"""
Classes used by developers of models to give them extra capabilities accessed by ArchIt.
"""
from typing import Optional, Union, Callable
from abc import abstractmethod, ABC
from torch import Tensor, nn
from numpy import ndarray
from collections import defaultdict

from contextlib import contextmanager
from contextvars import ContextVar
import torch
import transformers.initialization as hf_init
from transformers.modeling_utils import get_torch_context_manager_or_global_device
from transformers.training_args import TrainingArguments


__all__ = ["ReportDiagnosticsMixin", "StatefulLossMixin", "HuggingfaceInitialisationMixin", "LossState", "LoggingState"]


SupportedLoggingValue = Union[int, float, Tensor, ndarray]

class LoggingState:
    """
    Used for storing extra diagnostics by models that inherit from ReportMetricsMixin.
    Simplified from https://github.com/zipzou/hf-multitask-trainer/blob/main/hf_mtask_trainer/state.py.
    """

    def __init__(self, args: TrainingArguments):
        self._metrics: dict[str, list[SupportedLoggingValue]] = defaultdict(list)
        self._args = args

    def appendLogs(self, logs: dict[str,SupportedLoggingValue]):
        for k,v in logs.items():
            self._metrics[k].append(v)

    def compute(self,
                round_digits: Optional[int]=None,
                tensor_gathering_function: Optional[Callable[[Union[Tensor, list[Tensor]]], Tensor]]=None) -> dict[str, float]:
        """
        Compute, for each metric stored, its value that will be logged under the current global_step.

        If a metric's value was reported as a tensor or array of more than one number, obviously it can't be displayed in
        a 2D graph. To avoid this, a mean is computed for such values first.

        If multiple values were stored for the same metric (e.g. because it was logged once every .forward() and
        multiple forwards happened within one global_step, a.k.a. gradient accumulation), the mean is computed for all
        reported values (or, if those values were tensors/arrays, all means computed as previously mentioned).
        """
        name_to_many_unknowns = self._metrics

        # Convert all the stored values into floats. We do this in two steps:
        #   1. Convert all the non-number types into numbers. Now every metric is a flat list of numbers.
        name_to_many_floats = self._toUnsummarisedFloats(name_to_many_unknowns, tensor_gathering_function)

        #   2. Reduce across those lists.
        name_to_float = self._summarise(name_to_many_floats)

        # Rounding
        if round_digits is not None:
            name_to_float = {k: round(v, round_digits)
                             for k, v in name_to_float.items()}

        self._metrics.clear()
        return name_to_float

    def _toUnsummarisedFloats(self,
                              metrics: dict[str, list[SupportedLoggingValue]],
                              tensor_gathering_function: Optional[Callable[[Union[Tensor, list[Tensor]]], Tensor]]) -> dict[str, list[float]]:
        metrics_as_floats: dict[str, list[float]] = defaultdict(list)
        for k, values in metrics.items():
            for value in values:
                do_divide_across_accumulations = True
                if isinstance(value, Tensor):
                    value = (value if tensor_gathering_function is None else tensor_gathering_function(value)).mean().cpu().item()
                elif isinstance(value, ndarray):
                    value = value.mean().item()
                elif not isinstance(value, (int, float)):
                    do_divide_across_accumulations = False  # TODO: When? https://github.com/zipzou/hf-multitask-trainer/issues/7

                if do_divide_across_accumulations:
                    value /= self._args.gradient_accumulation_steps

                metrics_as_floats[k].append(value)
        return metrics_as_floats

    def _summarise(self, metrics_as_floats: dict[str, list[float]]) -> dict[str, float]:
        return {
            k: self._args.gradient_accumulation_steps * sum(values) / len(values)
            for k, values in metrics_as_floats.items()
        }


class ReportDiagnosticsMixin:
    """
    Interface for modules that want to log additional values (e.g. one particular activation inside a model).

    Adds two methods for two different users:
        1. `registerLog` to let an external user set a reference to an accumulator for the logs.
           The canonical case is that this is LaMoTO's ModelTrainer.
        2. `report` for the module itself to call without it needing to do safety checks.
    """

    def __init__(self):
        self._log: Optional[LoggingState] = None

    def registerLog(self, logging_state: LoggingState):
        self._log = logging_state

    def report(self, diagnostics: dict[str, SupportedLoggingValue]):
        if self._log is not None:
            self._log.appendLogs(diagnostics)


class LossState:
    """
    Used for storing an extra loss term by models that inherit from StatefulLossMixin.
    """

    def __init__(self):
        self._tensor: Union[Tensor,int] = 0  # We set it to 0 rather than torch.zeros(1) since for the latter you need to know the device (CPU/GPU) even before adding anything yet.

    def add(self, tensor: Tensor):
        self._tensor += tensor

    def compute(self) -> Union[Tensor,int]:
        """Return the accumulated loss and reset it to 0."""
        result = self._tensor
        self._tensor = 0
        return result


class StatefulLossMixin(ReportDiagnosticsMixin):
    """
    Interface for modules that generate unsupervised loss term(s) somewhere in their implementation. (For example, a
    regularisation term in one specific module.)

    Adds two methods for two different users:
        1. `registerLoss` to let an external user set a reference to an accumulator for the loss.
           The canonical case is that this is ArchIt's BaseModel.
        2. `addToLoss` for the module itself to call without it needing to do safety checks.

    This class also subclasses ReportDiagnosticsMixin so that if a reporter is connected, each (unscaled) loss term is
    tracked in its own logging entry.
    """

    def __init__(self):
        super().__init__()
        self._loss_state: Optional[LossState] = None

    def registerLoss(self, loss_state: LossState):
        self._loss_state = loss_state

    def addToLoss(self, report_as: str, tensor: Tensor, coefficient: float=1.0):
        if self._loss_state is not None:
            self._loss_state.add(coefficient*tensor)

        assert report_as
        self.report({report_as: tensor})


_no_archit_init_weights: ContextVar[bool] = ContextVar("_no_archit_init_weights", default=False)

@contextmanager
def no_weight_init():
    """
    Used in a with-statement to disable weight initialisation for both HuggingFace PretrainedModel and ArchIt's HuggingfaceInitialisationMixin.
    """
    old = _no_archit_init_weights.set(True)  # Handles ArchIt by temporarily making a value globally readable.
    try:
        with hf_init.no_init_weights(): # Handles HF by temporarily emptying the body of PretrainedModel.init_weights.
            yield
    finally:
        _no_archit_init_weights.reset(old)


def using_meta_device() -> bool:
    return get_torch_context_manager_or_global_device() == torch.device("meta")


class HuggingfaceInitialisationMixin(nn.Module, ABC):
    """
    Defines recursive nn.Module initialisation with the _is_hf_initialized flag like transformers's PretrainedModel,
    to avoid re-initialising weights twice (possibly using the wrong distribution the second time).
    """

    def post_init(self):
        """It is the responsibility of subclasses of this class to call this method at the end of their constructor."""
        self.init_weights()

    def init_weights(self):
        if not using_meta_device() and not _no_archit_init_weights.get():  # => Do nothing during the post_init() call of the constructor when checkpoint weights still need to be loaded in.
            self.initialize_weights()

    @torch.no_grad()
    def initialize_weights(self):
        with hf_init.guard_torch_init_functions():
            self.apply(self._initialize_weights)  # Recursively applies the given function to all child Modules and then self.

    def _initialize_weights(self, module: nn.Module):
        if not getattr(module, "_is_hf_initialized", False):
            self._init_weights(module)
            module._is_hf_initialized = True

    @abstractmethod
    def _init_weights(self, module: nn.Module):
        pass
