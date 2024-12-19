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
from typing import Optional, TypeVar, Generic, Type, Union, Tuple
from typing_extensions import Self

import torch
from torch import Tensor, FloatTensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndNoAttention as AllHiddenStatesAndPooling
from transformers.utils import ModelOutput

Tensors = Tuple[Tensor, ...]
OneOrMoreTensors = Union[Tensor, Tensors]
HeadOutput = OneOrMoreTensors  # In general, a head can return multiple tensors, see e.g. the dependency parsing head.

from .configs import *
from .configs import PC, HC  # Have to do this explicitly for the type checker not to complain.
from .mixins import StatefulLossMixin, LossState
from ..util import dataclass_from_dict

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

    @classmethod  # So you can call it on the class. Must come before @abstractmethod and @property.  https://stackoverflow.com/a/53417582/9352077
    @property  # So you can call it without parentheses
    @abstractmethod  # This should force an implementation, but the @property causes ABC to allow the method to be left unimplemented in subclasses... Seems like a bug to me. Anyway, ABC does raise a compile-time error with this `raise` in the body, somehow! https://stackoverflow.com/a/74534218/9352077
    def config_class(cls) -> Type[PC]:
        """
        Necessary so that when you call .from_pretrained() on this class, it knows how to parse the associated config.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def standardiseConfig(cls, raw_config: PC) -> BaseModelConfig:
        """
        Turn the HuggingFace config into a standardised interface.
        """
        pass

    @classmethod
    @abstractmethod
    def buildCore(cls, raw_config: PC) -> PreTrainedModel:
        """
        The core is where most of the computation for .forward() happens, and is the HuggingFace object that would
        be called .base_model otherwise.
        """
        pass


class Head(Module, Generic[HC], ABC):
    """
    Adapter around implementations (or just implementations themselves) of heads that take BMOWPACA as input and
    spit out a tensor as output.
    """

    def __init__(self, base_config: BaseModelConfig, head_config: HC):
        super().__init__()
        self.assertConfigConstraints(base_config, head_config)

    def post_init(self):
        self.apply(self._init_weights)  # Recursively applies the given function to all child Modules and then self.

    def _init_weights(self, module: Module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)  # The 0.02 is normally a hyperparameter, but everyone uses 0.02...
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

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
    def fromConfigs(cls, base_config: BaseModelConfig, head_config: HC) -> Self:
        """
        When the developer overrides a method and changes the signature, he gets a warning. This is not the case for
        constructors. The only warning users get is when they don't call super().__init__(), but then constructors for
        different heads could still have *additional* arguments and we don't want that.
        Hence, we use this wrapper to force subclasses to be instantiable with no more than the base class's arguments.
        """
        return cls(base_config, head_config)

    @classmethod
    @abstractmethod
    def assertConfigConstraints(cls, base_config: BaseModelConfig, head_config: HC):
        pass

    @classmethod
    @property
    @abstractmethod  # Doesn't work due to the @property. Leaving it here so it will work one day.
    def config_class(cls) -> Type[HC]:
        """
        Necessary so that when you call .from_pretrained() on a ModelWithHead class, it knows how to parse the associated config.
        """
        raise NotImplementedError

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
    logits: OneOrMoreTensors = None  # Must be None due to some stupid rule in ModelOutput.__post_init__(), even though it never actually is None.
    base_model_output: Optional[AllHiddenStatesAndPooling] = None


class ModelWithHead(PreTrainedModel, Generic[PC,HC], ABC):
    """
    Although this class treats BaseModel and Head as two equals, it should be noted that when you defined a model task
    architecture as a subclass of this class, you will let the *user* supply the BaseModel while *you* build the Head
    for them and ask for a config instead.
    """

    def __init__(self, combined_config: CombinedConfig[PC,HC], model: BaseModel[PC], head: Head[HC], loss: _Loss):
        """
        No user of this class should use the constructor, except the call to cls() inside .from_pretrained() (hence why
        the first argument is a config, despite the other arguments being existing instances that won't use that config).
        """
        super().__init__(combined_config)
        self.config: CombinedConfig[PC,HC] = self.config  # Same type annotation trick as in BaseModel.
        self.supports_gradient_checkpointing = model.base_model.supports_gradient_checkpointing

        # Note: when you change the names of these fields, also change the strings in _load_pretrained_model().
        self.model: BaseModel[PC] = model
        self.head: Head[HC]       = head
        self._loss_function = loss

        assert isinstance(head, self.__class__.head_class)

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
        return ModelWithHeadOutput(
            logits=logits,
            base_model_output=base_output if output_hidden_states else None,
            loss=self.model.computeLoss() + self.computeLoss(logits, labels)   if labels is not None else   self.model.computeLoss()
        )

    def __call__(self, *args, **kwargs) -> ModelWithHeadOutput:
        return super().__call__(*args, **kwargs)

    def callBaseModel(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        do_drop_intermediates: bool,
        **kwargs
    ) -> AllHiddenStatesAndPooling:
        return self.model(input_ids, attention_mask, do_drop_intermediates=do_drop_intermediates, **kwargs)

    @property
    def base_model(self) -> PreTrainedModel:
        return self.model.base_model

    @property
    def loss_function(self) -> _Loss:  # Need to override this property because Les Incompétents always find a way to mess up their design.  https://github.com/huggingface/transformers/pull/34191#issuecomment-2539200643
        return self._loss_function

    @abstractmethod
    def computeLoss(self, logits: OneOrMoreTensors, labels: OneOrMoreTensors) -> FloatTensor:
        pass

    #########################
    ##### CLASS METHODS #####
    #########################

    @classmethod
    @property
    def config_class(cls) -> Type[PretrainedConfig]:  # Because .from_pretrained() uses it.
        return CombinedConfig

    @classmethod
    @property
    @abstractmethod  # As always, not working right now.
    def head_class(cls) -> Type[Head[HC]]:  # Because all Head classes have the same constructor method .fromConfigs(), all you need to .buildHead() is to know the class to call it on.
        pass

    @classmethod
    def buildHead(cls, base_model_config: BaseModelConfig, head_config: HC) -> Head[HC]:
        return cls.head_class.fromConfigs(base_model_config, head_config)

    @classmethod
    @abstractmethod
    def buildLoss(cls) -> _Loss:
        pass

    @classmethod
    def fromModelAndHeadConfig(cls, base_model: BaseModel, head_config: HC) -> Self:
        """Expects a model that has already been loaded, and assumes you want the head randomised."""
        return cls(
            CombinedConfig(base_model_config=base_model.config, head_config=head_config),
            base_model,
            cls.buildHead(base_model.__class__.standardiseConfig(base_model.config), head_config),
            cls.buildLoss()
        )

    @classmethod
    def from_pretrained(cls, checkpoint: str, base_model_class: Type[BaseModel[PC]], head_config: HC=None) -> "ModelWithHead[PC,HC]":  # Can't use Self here because it is trumped by the strange return type of the call inside the function.
        """
        Load the model from an existing checkpoint. Distinguishes between two cases:
            - The given checkpoint is of the correct model-with-head. In that case, no head config is needed.
            - The given checkpoint is of a model without a head. In that case, the head config is used to construct a new head.

        How HuggingFace's .from_pretrained() loads weights is that it first creates an instance of the class by calling
        its constructor, and then it reads the provided checkpoint file by querying whatever fields there exist in that
        instance. Usually, that constructor only takes a config and explicitly defines the PyTorch module tree.
        Because a ModelWithHead doesn't know which base model will be used beforehand, it doesn't have a pre-determined
        module tree and cannot just be constructed with a config alone. (Not even by saving the class name in the config,
        because we don't have a mapping between the names and file paths of custom classes.)

        In short: the goal is to construct all the objects needed for the ModelWithHead constructor (except for the
        CombinedConfig, since from_pretrained helps us to construct that): a base model, a head, and a loss function.
        The weights in each of these will be set in the super() call.
        The way we communicate those constructor arguments is that .from_pretrained() passes any *args and **kwargs
        through itself to the constructor. This method's responsibility is to create those *args and **kwargs.
        """
        # 1. Get the config needed to get the base model. (Note: similar work will be done again later in CombinedConfig.from_pretrained() when the checkpoint is unpacked a second time. It is what it is.)
        base_model_config, _ = PretrainedConfig.get_config_dict(checkpoint)  # Basically a static method of PretrainedConfig, not a class-specific method, used to read a JSON.
        if "base_model_config" in base_model_config:  # In this case: (1) the base model's config lives one level deeper, and (2) the head config can be imputed.
            head_config       = head_config or dataclass_from_dict(cls.head_class.config_class, base_model_config["head_config"])
            base_model_config = base_model_config["base_model_config"]

        base_model_config = base_model_class.config_class.from_dict(base_model_config)

        # 2. Instantiate the base model with that config, instantiate the head with that and the head config, and the loss.
        return super().from_pretrained(
            checkpoint,
            # Unnamed arguments (*args) passed to PretrainedModel.from_pretrained() are passed straight to the constructor of this class, except the checkpoint is replaced by a config.
            base_model_class(base_model_config),  # A mix between .from_config (which only exists on auto classes and expects a config from the user) and .from_pretrained (just to get the config so the user doesn't have to supply it).
            cls.buildHead(base_model_class.standardiseConfig(base_model_config), head_config),
            cls.buildLoss(),

            # Keyword arguments (**kwargs) are passed first to PretrainedConfig.from_pretrained() and only what remains ends up in PretrainedModel.from_pretrained(). Do note that PretrainedConfig.from_pretrained() does NOT pass these **kwargs to the constructor and only uses them for administrative stuff by default; I forcibly add them in get_config_dict().
            head_config=head_config,
            base_model_config_class=base_model_class.config_class,
            head_config_class=cls.head_class.config_class
        )

    @classmethod
    def _load_pretrained_model(cls, empty_model_with_head: "ModelWithHead", state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, **kwargs):
        """
        Indirection to trick PyTorch into recognising the base model in a checkpoint during .from_pretrained() on this class.
        To understand how this works, see https://bauwenst.github.io/posts/explainers/2024-08-31-How-from_pretrained-works/.
        """
        loaded_keys = set(state_dict.keys())
        if any(all(not key.startswith(prefix) for prefix in {"model", "head"}) for key in loaded_keys):  # => You've loaded a HF checkpoint.
            print("Loading HuggingFace checkpoint into ArchIt model.")
            state_dict = {k:v for k,v in state_dict.items() if not k.startswith("head.")}  # If there is a head in the checkpoint that is literally named "head", it will cause a "state dict is corrupted" error since it looks like the head of ModelWithHead. The reason: when HF thinks you load from a base model checkpoint into a base-with-head model, it checks if any of the checkpoint keys (which should be base model fields) are not in the base model but are in the head. That means your checkpoint is not a base model after all and hence counts as corruption.
            consume_prefix_in_state_dict_if_present(state_dict, empty_model_with_head.model.base_model.base_model_prefix + ".")  # In-place.
            loaded_keys = list(state_dict.keys())

        return super()._load_pretrained_model(empty_model_with_head, state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, **kwargs)

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
