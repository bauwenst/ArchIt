"""
Classes used by developers of models to give them extra capabilities accessed by ArchIt.
"""
from typing import Optional, Union, Dict, List, Callable
from torch import Tensor
from numpy import ndarray
from collections import defaultdict
from transformers.training_args import TrainingArguments


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


class StatefulLossMixin:
    """
    Interface for modules that generate unsupervised loss term(s) somewhere in their implementation. (For example, a
    regularisation term in one specific module.)

    Adds two methods for two different users:
        1. `registerLoss` to let an external user set a reference to an accumulator for the loss.
           The canonical case is that this is ArchIt's BaseModel.
        2. `addToLoss` for the module itself to call without it needing to do safety checks.
    """

    def __init__(self):
        self._loss_state: Optional[LossState] = None

    def registerLoss(self, loss_state: LossState):
        self._loss_state = loss_state

    def addToLoss(self, tensor: Tensor):
        if self._loss_state is not None:
            self._loss_state.add(tensor)


SupportedLoggingValue = Union[int, float, Tensor, ndarray]

class LoggingState:
    """
    Used for storing extra diagnostics by models that inherit from ReportMetricsMixin.
    Simplified from https://github.com/zipzou/hf-multitask-trainer/blob/main/hf_mtask_trainer/state.py.
    """

    def __init__(self, args: TrainingArguments):
        self._metrics: Dict[str, List[SupportedLoggingValue]] = defaultdict(list)
        self._args = args

    def appendLogs(self, logs: Dict[str,SupportedLoggingValue]):
        for k,v in logs.items():
            self._metrics[k].append(v)

    def compute(self,
                round_digits: Optional[int]=None,
                tensor_gathering_function: Optional[Callable[[Union[Tensor, List[Tensor]]], Tensor]]=None) -> Dict[str, float]:
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
                              metrics: Dict[str, List[SupportedLoggingValue]],
                              tensor_gathering_function: Optional[Callable[[Union[Tensor, List[Tensor]]], Tensor]]) -> Dict[str, List[float]]:
        metrics_as_floats: Dict[str, List[float]] = defaultdict(list)
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

    def _summarise(self, metrics_as_floats: Dict[str, List[float]]) -> Dict[str, float]:
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

    def report(self, diagnostics: Dict[str, SupportedLoggingValue]):
        if self._log is not None:
            self._log.appendLogs(diagnostics)
