"""
TODO:
    - We don't necessarily want to support .from_pretrained() on ModelWithHead checkpoints. It'd be nice to have it, but
      right now, we should focus on having a way to get the base model initialised from a checkpoint.
        - We have at least one way to do it, which is to override .from_pretrained() and just call ModelClass.from_pretrained()
          inside. (May require having the model stored in the same field of BaseModel every time so we can reassign it.)
        - Since .from_pretrained() parses a checkpoint by instantiating cls(config) and then querying the weights file,
          one thing you could try to do is to sneak an argument of the base model class into that constructor so that it
          constructs an instance of e.g. RobertaModel without knowing it will in advance.
"""
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional, TypeVar, Generic

from torch import Tensor, FloatTensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions as BMOWPACA
from transformers.utils import ModelOutput


@dataclass
class BaseModelConfig:
    """
    Unbelievably, HuggingFace's PretrainedConfig, the parent class for all config classes, does NOT provide
    a universal interface to get basic model properties like amount of layers, hidden size and so on.
    Instead, it says in its documentation that those properties "are common" in all subclasses. I don't play that way.
    """
    # Will be relevant for heads to inspect:
    hidden_size: int
    hidden_dropout_prob: float
    vocab_size: int

    # Will not be relevant, but should still be available on base model configs:
    num_hidden_layers: int
    num_attention_heads: int


class HeadConfig:
    pass

C = TypeVar("C", bound=HeadConfig)
P = TypeVar("P", bound=PretrainedConfig)


class BaseModel(PreTrainedModel, Generic[P], ABC):
    """
    Adapter around HuggingFace transformer encoders to have the same input-output signature.
    Crucially, the __call__() and forward() methods are type-annotated, unlike torch.nn.Module.

    To use any base model (e.g. RobertaModel), wrap it by this interface. Has a type parameter for proper handling of the config.
    """

    def __init__(self, config: P):
        super().__init__(config)
        self.config: P = self.config  # Type annotation for the existing field self.config. The annotation is not updated automatically if you just type-hint this class's constructor argument, because self.config gets its type from the signature of the super constructor. The alternative to a generic is that you repeat this expression here for each separate config.

    def __call__(self, *args, **kwargs) -> BMOWPACA:
        return super().__call__(*args, **kwargs)  # Just calls forward() which we know is adapted to the model to return the BMOWPACA.

    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        **kwargs
    ) -> BMOWPACA:
        pass

    @abstractmethod
    def convertConfig(self) -> BaseModelConfig:
        """
        Turn the HuggingFace config into a standardised interface.
        """
        pass


class Head(Module, Generic[C], ABC):
    """
    Adaapter around implementations of heads (or just implementations themselves) that take BMOWPACA as input and
    spit out a tensor as output.
    """

    def __init__(self, base_config: BaseModelConfig, head_config: C):
        super().__init__()
        self.base_config = base_config
        self.head_config = head_config

    @abstractmethod
    def forward(
        self,
        encoder_output: BMOWPACA,
        attention_mask: Tensor,
        **kwargs
    ) -> Tensor:
        pass


@dataclass
class ModelWithHeadOutput(ModelOutput):
    base_model_output: BMOWPACA
    logits: FloatTensor
    loss: Optional[FloatTensor]


class ModelWithHead(Module, ABC):
    """
    Although this class treats BaseModel and Head as two equals, it should be noted that when you defined a model task
    architecture as a subclass of this class, you will let the *user* supply the BaseModel while *you* build the Head
    for them and ask for a config instead.
    """

    def __init__(self, base_model: BaseModel, head: Head, loss: _Loss):
        super().__init__()
        self.base_model    = base_model
        self.head          = head
        self.loss_function = loss

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        **kwargs
    ):
        base_output = self.base_model(input_ids, attention_mask, **kwargs)
        logits      = self.head(base_output, attention_mask, **kwargs)
        return ModelWithHeadOutput(
            base_model_output=base_output,
            logits=logits,
            loss=self.getLoss(logits, labels) if labels is not None else None
        )

    @abstractmethod
    def getLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
        pass
