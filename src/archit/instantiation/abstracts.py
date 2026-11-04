"""
The three abstract classes needed to wrap HuggingFace models and put a head on top, while supporting a .from_pretrained() method:
    - BaseModel adapts every HuggingFace model to a general interface.
    - Head turns embeddings into logits.
    - ModelWithHead first runs a BaseModel and then runs a Head.

Two design choices that I will highlight:
    - All methods that have to do with constructing instances of the class (e.g. dealing with configs) are ALL
      tagged with @classmethod. This is because instances have to be constructible within .from_pretrained(), which is
      a class method that hence cannot be informed by instance fields or call instance methods.
    - I use generics to parameterise classes with their config classes, which makes sense considering that configs
      quite literally contain the hyperparameters that configure a model, and which hyperparameters are used is itself a hyperparameter of the class.

      Note that generic type variables are only useful OUTSIDE of the method whose signature they parameterise. As the
      developer, you won't notice this when you only use variable return types, because return type is not used for
      autocompletion inside the method itself. You WILL notice this for variable argument types: an argument tagged as "T"
      will always have the type checking (and autocompletion) of T's upper bound, EVEN IF you are inside a class that declares
      what T should be and is hence no longer generic! The only way you get type checking for a method's argument inside
      that method's body is to change the type hint when you override the method, but then it was useless to have the
      variable type hint first! All this to say: you should know when and when not to expect generic types to help with
      autocompletion, and if you need to override a variable signature with a type invocation, you're doing it wrong.
"""
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional, TypeVar, Generic, Union
from typing_extensions import Self

import re
import torch
from torch import Tensor, FloatTensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndNoAttention as AllHiddenStatesAndPooling
from transformers.utils import ModelOutput

Tensors = tuple[Tensor, ...]
OneOrMoreTensors = Union[Tensor, Tensors]
HeadOutput = OneOrMoreTensors  # In general, a head can return multiple tensors, see e.g. the dependency parsing head.

from .configs import *
from .configs import PC, HC  # Have to do this explicitly for the type checker not to complain.
from .mixins import *
from ..util import classproperty

__all__ = ["BaseModel", "BaseModelConfig", "Head", "HeadConfig", "ModelWithHead", "CombinedConfig", "AllHiddenStatesAndPooling"]


class BaseModel(PreTrainedModel, Generic[PC], ABC):
    """
    Adapter around HuggingFace transformer encoders to have the same input-output signature.
    Crucially, the __call__() and forward() methods are type-annotated, unlike torch.nn.Module.

    To use any base model (e.g. RobertaModel), wrap it by this interface. Has a type parameter for proper handling of the config.
    """

    def __init__(self, config: PC):
        super().__init__(config)
        self.config: PC = self.config  # Type annotation for the existing field self.config. The annotation is not updated automatically if you just type-hint this class's constructor argument, because self.config gets its type from the signature of the super constructor. The alternative to a generic is that you repeat this expression here for each separate config.
        self._core = self.buildCore(config)  # Private field because users should refer to .hf to access the HuggingFace model.

        self._accumulated_loss = LossState()  # Accumulates loss generated inside the core.
        self.activateCoreLoss()

    def __init_subclass__(cls, **kwargs):  # Also applies to grandchild classes.
        """
        Transformers used to expect model classes to declare a class field called 'config_class' to declare the type of
        their config. It would only be accessed on .from_pretrained, so you could override it with an abstract property
        to allow subclasses to declare it later on.

        Nowadays, they have a different way of declaring the config class, BUT, they STILL store the result of this
        declaration manually in cls.config_class whenever a subclass is created (including BaseModel). In doing so, they
        retrieve the config_class class field if it still exists, so you can no longer leave it abstract.
        You can leave it None, however, and no complaints will be made, so that's why we have no config_class field, and
        set cls.config_class ourselves.
        """
        super().__init_subclass__(**kwargs)
        cls.config_class: type[PC] = cls.configClass()  # This works even for subclasses that have not implemented this @abstractmethod yet. It just returns the None of the empty body.

    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        do_drop_intermediates: bool=True,
        **kwargs
    ) -> AllHiddenStatesAndPooling:
        pass

    def __call__(self, *args, **kwargs) -> AllHiddenStatesAndPooling:
        return super().__call__(*args, **kwargs)  # Just calls forward() which we know is adapted to the model to return the BMOWPACA. This method is here just for type annotation of 'model(x)'.

    @property
    def base_model(self) -> PreTrainedModel:
        """
        Returns a reference to the raw HuggingFace model beneath this BaseModel.
        Originally this method was called ".hf" but then you had .hf and .base_model both being suggested
        and that makes the interface cluttered. So now we have BaseModel.base_model... oh well.

        self._core is not always instantiated, that's why you shouldn't refer to it as a user.
        """
        return self._core

    def activateCoreLoss(self):
        """
        To prevent the core from accumulating any loss (which only happens if any of its modules is a StatefulLossMixin)
        and therefore prevent the core from adding anything to the downstream loss, override this method with an empty body.
        """
        def r(module: Module):
            if isinstance(module, StatefulLossMixin):
                module.registerLoss(self._accumulated_loss)
        self.base_model.apply(r)

    def computeLoss(self) -> Tensor:
        """DO NOT override this method. It is necessary to free the memory used for accumulating core loss if applicable."""
        return self._accumulated_loss.compute()

    #########################
    ##### CLASS METHODS #####
    #########################

    @classmethod
    def from_config(cls, config: Union["CombinedConfig",PC]) -> Self:
        return cls(config.base_model_config) if isinstance(config, CombinedConfig) else cls(config)

    # This used to have @property. But @property is evil, so we no longer have it.
    @classmethod  # So you can call it on the class. Must come before @abstractmethod.  https://stackoverflow.com/a/53417582/9352077
    @abstractmethod
    def configClass(cls) -> type[PC]:
        """Necessary so that when you call .from_pretrained() on this class, it knows how to parse the associated config."""
        pass

    @classmethod
    @abstractmethod
    def standardiseConfig(cls, raw_config: PC) -> BaseModelConfig:
        """
        Turn the HuggingFace config into a standardised interface.
        """
        pass

    @classmethod
    @abstractmethod
    def coreClass(cls) -> type[PreTrainedModel]:
        """The type that will be returned by buildCore()."""
        pass

    @classmethod
    def buildCore(cls, raw_config: PC) -> PreTrainedModel:
        """
        The core is where most of the computation for .forward() happens, and is the HuggingFace object that would
        be called .base_model otherwise.
        """
        return cls.coreClass()(raw_config)

    # In ArchIt, specifying that you want FA2 is NEVER done in the .from_pretrained() call and also NEVER as a field in
    # CombinedConfig (because CombinedConfig only has a few fields). If the user wants it, it will always be in the PretrainedConfig
    # of the base model.
    # Since it is the class instantiated with an FA2-requesting config that needs to have a flag set to prevent an
    # error, the above observation means that we need to set a flag in all the classes instantiated with the base model
    # PretrainedConfig. There are two such classes: the class of the core (which often multiplexes between attention
    # implementations, and thus the HuggingFace devs have already set the relevant flags when designing it) but ALSO the
    # BaseModel wrapper. Hence, we let class field accesses like BaseModel._supports_flash_attn_2 evaluate to the
    # core's ._supports_flash_attn_2 class field.

    @classproperty
    def _supports_sdpa(cls):
        return cls.coreClass()._supports_sdpa

    @classproperty
    def _supports_flash_attn(cls):
        return cls.coreClass()._supports_flash_attn

    @classproperty
    def _supports_flash_attn_2(cls):
        return cls.coreClass()._supports_flash_attn_2


class Head(HuggingfaceInitialisationMixin):
    """
    Adapter around implementations (or just implementations themselves) of heads that take BMOWPACA as input and
    spit out a tensor as output.

    Subclasses MUST abide by the following rules:
        1. The constructor arguments MUST NOT be changed. ModelWithHead relies on heads having a fixed constructor signature.
        2. The constructor MUST call post_init() at the end, in order for weights to be properly initialised.
    """

    def __init__(self, base_config: BaseModelConfig, head_config: HeadConfig):
        super().__init__()
        head_config.assertConstraints()

    def _init_weights(self, module: Module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # The 0.02 is normally a hyperparameter, but everyone uses 0.02. ModernBERT uses 1/sqrt(H) for its classification heads, which is 0.036 for H = 768.
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.constant_(module.weight, 1.0)

    @abstractmethod
    def forward(
        self,
        encoder_output: AllHiddenStatesAndPooling,
        attention_mask: Tensor,
        **kwargs
    ) -> HeadOutput:
        pass

    def __call__(self, *args, **kwargs) -> HeadOutput:
        return super().__call__(*args, **kwargs)

    #########################
    ##### CLASS METHODS #####
    #########################

    @classmethod
    def hfEquivalentSuffix(cls) -> str:
        """
        HuggingFace defines its heads only as part of a "...For..." model+head class. That means there is an equivalence
        between some of ArchIt's standalone head classes and HuggingFace's model+head classes. We assume that the same
        class suffix always implies the presence of the same head. The equivalence between ArchIt's model+head and
        HuggingFace's model+head architectures is less precise because ArchIt separates out tasks that still use the same head.
        """
        return "/"


@dataclass
class ModelWithHeadOutput(ModelOutput):  # The convention in HF is that loss is always the 1st element and logits the 2nd.
    loss: Optional[Tensor] = None
    logits: Optional[OneOrMoreTensors] = None  # Set to None during training because the accelerator converts all outputs to float32 and doing that with an N x L x V tensor can blow up memory.
    base_model_output: Optional[AllHiddenStatesAndPooling] = None


class ModelWithHead(PreTrainedModel, Generic[PC,HC], ABC):
    """
    Although this class treats BaseModel and Head as two equals, it should be noted that when you defined a model task
    architecture as a subclass of this class, you will let the *user* supply the BaseModel while *you* build the Head
    for them and ask for a config instead.
    """

    config_class = CombinedConfig  # In PretrainedModel.__init_subclass__, this class field gets priority for determining the config class. Used to be a classmethod-property in Transformers, but Python deprecated those.

    def __init__(self, combined_config: CombinedConfig[PC,HC], base_model_class: type[BaseModel[PC]]):  # Head class needn't be known because it is uniquely associated with the head config.
        """
        No user of this class should use the constructor, except the call to cls() inside .from_pretrained() (hence why
        the first argument is a config, despite the other arguments being existing instances that won't use that config).
        """
        super().__init__(combined_config)
        self.config: CombinedConfig[PC,HC] = self.config  # Same type annotation trick as in BaseModel.
        head_class = combined_config.head_config.headClass()
        assert combined_config.head_config.getInputSignature()  == self.headSignature().getInputSignature()
        assert combined_config.head_config.getOutputSignature() == self.headSignature().getOutputSignature()

        # Note: when you change the names of these fields, also change the regex in key_mapping below.
        self.model: BaseModel[PC] = base_model_class(combined_config.base_model_config)
        self.head: Head           = head_class(base_model_class.standardiseConfig(combined_config.base_model_config), combined_config.head_config)
        self._loss_function: _Loss = self.buildLoss()

        self.supports_gradient_checkpointing = self.model.base_model.supports_gradient_checkpointing
        self.post_init()  # Note: The base model and the head have their own post_init() that initialises their weights and does weight tying internally. What isn't done yet, is weight tying across the model and the head. What also isn't done yet is announcing the existing tied weights to the MwH wrapper. This post_init() does exactly that. It will not re-initialise weights because of HF's _is_hf_initialized flag.

    def _init_weights(self, module: Module):
        # For modules that are part of the head, switch to the head's initialiser. This doesn't happen automatically in smart_apply because the head is not a PretrainedModel, but ModelWithHead is, and so this _init_weights applies to all its children.
        if module in self.head.modules():
            self.head._init_weights(module)
        else:
            super()._init_weights(module)

    # base + head

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[OneOrMoreTensors]=None,  # To have a uniform interface across tasks (unlike what HF offers), we don't have separate arguments for separate labels, but rather a potential tuple of labels.
        output_hidden_states: bool=False,  # False by default because otherwise the output will be tuple-ified into (logits, hidden_states) rather than just the logits, and this breaks metrics relying on EvalPrediction objects.
        **kwargs
    ) -> ModelWithHeadOutput:
        base_output = self.callBaseModel(input_ids, attention_mask, do_drop_intermediates=not output_hidden_states, **kwargs)
        logits      = self.head(base_output, attention_mask, **kwargs)
        return ModelWithHeadOutput(  # Note: everything returned by this method will be converted to float32, so we have to be careful to only return the essentials.
            loss=self.model.computeLoss() + self.computeLoss(logits, labels)   if labels is not None else   self.model.computeLoss(),
            logits=logits if not self.training else None,
            base_model_output=base_output if output_hidden_states else None
        )

    def __call__(self, *args, **kwargs) -> ModelWithHeadOutput:
        return super().__call__(*args, **kwargs)

    # base

    @property
    def base_model(self) -> PreTrainedModel:
        return self.model.base_model

    def callBaseModel(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        do_drop_intermediates: bool,
        **kwargs
    ) -> AllHiddenStatesAndPooling:
        return self.model(input_ids, attention_mask, do_drop_intermediates=do_drop_intermediates, **kwargs)

    # loss

    @property
    def loss_function(self) -> _Loss:  # Need to override this property because Les Incompétents always find a way to mess up their design.  https://github.com/huggingface/transformers/pull/34191#issuecomment-2539200643
        return self._loss_function

    @abstractmethod
    def computeLoss(self, logits: OneOrMoreTensors, labels: OneOrMoreTensors) -> FloatTensor:
        pass

    #########################
    ##### CLASS METHODS #####
    #########################

    # This obviously needs to be a class-level method or field (a property is not allowed), so we choose method to tag it as abstract.
    @classmethod
    @abstractmethod
    def headSignature(cls) -> type[TensorSignature]:  # Note: not type[HC] because you can instantiate with many configs HC.
        """Heads of this ModelWithHead are allowed to be any subclass of this class."""
        pass

    @classmethod  # Not abstract because usually the task does not adjust the head config. Only in case of QA/AQA.
    def adjustHeadConfig(cls, head_config: HC):
        pass

    @abstractmethod
    def buildLoss(self) -> _Loss:
        pass

    @classmethod
    def fromModelAndHeadConfig(cls, base_model: BaseModel, head_config: HC) -> Self:
        """Expects a model that has already been loaded, and assumes you want the head randomised."""
        mwh = cls(
            CombinedConfig(base_model_config=base_model.config, head_config=head_config),
            base_model_class=type(base_model)
        )
        mwh.model = base_model

    @classmethod
    def from_pretrained(cls, checkpoint: str,
                        base_model_class: type[BaseModel[PC]], head_config: HC=None, head_config_class: type[HC]=None, **kwargs) -> "ModelWithHead[PC,HC]":  # Can't use Self here because it is trumped by the strange return type of the call inside the function.
        """
        Load the model from an existing checkpoint (of this ModelWithHead OR of just the base model), rather than a config.

        The additional parameters are necessary to correctly parse the content of the checkpoint. If the checkpoint
        contains a head config, only the head config class is needed. Otherwise a full head config is needed.
        """
        return super().from_pretrained(
            checkpoint,
            # Unnamed arguments (*args) passed to PretrainedModel.from_pretrained() are passed straight to the constructor of this class, except the checkpoint is replaced by a config.
            base_model_class,

            # Keyword arguments (**kwargs) are passed first to PretrainedConfig.from_pretrained() and only what remains ends up in PretrainedModel.from_pretrained(). Do note that PretrainedConfig.from_pretrained() does NOT pass these **kwargs to the constructor and only uses them for administrative stuff by default; I forcibly add them in get_config_dict().
            head_config=head_config,
            base_model_config_class=base_model_class.configClass(),
            head_config_class=head_config_class,

            # Popped kwarg by PretrainedModel.from_pretrained(). Adapts HF weights to where they live in ArchIt, while leaving ArchIt checkpoints untouched.
            key_mapping={
                # Rules:
                #  1. If key starts with "model.core." or "head.", do not do anything.
                #  2. For all other keys:
                #       - if you start with the prefix, remove it.
                #       - prepend "model.core." (even to the head in case of a heady model, because regex cannot tell the difference between the head of the heady model vs. the modules of the headless model)
                r"^(?!" + re.escape(cls.base_model_prefix) + r"\.|head\.)(?:" + re.escape(base_model_class.coreClass().base_model_prefix) + r"\.)?": cls.base_model_prefix + "."
            },
            **kwargs
        )

    base_model_prefix = "model._core"

    def __getattr__(self, item):
        """
        __getattr__ except it supports fields that have a "." in them, i.e. nested fields.
        Necessary to support the prefix consumption of _load_pretrained_model().
        """
        if "." not in item:  # As if we never overrode it.
            return super().__getattr__(item)
        else:
            # print("> Tried accessing nested field", item)
            obj = self
            for name in item.split("."):
                # print("\t> Accessing", obj.__class__.__name__ + "." + name)
                obj = getattr(obj, name)
            return obj

    def _get_name_of_module(self, module: torch.nn.Module) -> str:
        for name, module_reference in self.named_modules():
            if module_reference is module:
                return name
        else:
            raise ValueError(f"{module!r} is not a submodule of this {type(self).__name__}")
