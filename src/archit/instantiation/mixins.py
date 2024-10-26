"""
Classes used by developers of models to give them extra capabilities accessed by ArchIt.
"""
from typing import Optional, Union
from torch import Tensor


class LossState:

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
        - registerLoss to let an external user set a reference to an accumulator for the loss.
        - addToLoss for the model itself to call without safety checks.
    """

    def __init__(self):
        self._loss_state: Optional[LossState] = None

    def registerLoss(self, loss_state: LossState):
        self._loss_state = loss_state

    def addToLoss(self, tensor: Tensor):
        if self._loss_state is not None:
            self._loss_state.add(tensor)
