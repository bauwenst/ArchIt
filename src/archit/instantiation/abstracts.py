"""
Abstract classes needed to wrap HuggingFace models and put a head on top, while supporting a .from_pretrained() method.
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

TODO:
    - Can you initialise the base model from a base model checkpoint, INSIDE a ModelWithHead.from_pretrained()?
        > What we can currently do is either accept a ModelWithHead checkpoint or accept a BaseModel checkpoint BUT only
          read the config. We can't read the weights for the latter.
        - We have at least one way to do it, which is to override .from_pretrained() and just call BaseModel.from_pretrained()
          inside. (May require having the model stored in the same field of BaseModel every time so we can reassign it.)
          The problem is of course that we want to support ModelWithHead.from_pretrained() from an ACTUAL ModelWithHead checkpoint too!
"""
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional, TypeVar, Generic, Type, Union, Tuple

from torch import Tensor, FloatTensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndNoAttention as AllHiddenStatesAndPooling
from transformers.utils import ModelOutput
from typing_extensions import Self

__all__ = ["BaseModel", "BaseModelConfig", "Head", "HeadConfig", "ModelWithHead"]


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


HC = TypeVar("HC", bound=HeadConfig)
PC = TypeVar("PC", bound=PretrainedConfig)


class BaseModel(PreTrainedModel, Generic[PC], ABC):
    """
    Adapter around HuggingFace transformer encoders to have the same input-output signature.
    Crucially, the __call__() and forward() methods are type-annotated, unlike torch.nn.Module.

    To use any base model (e.g. RobertaModel), wrap it by this interface. Has a type parameter for proper handling of the config.
    """

    def __init__(self, config: PC):
        super().__init__(config)
        self.config: PC = self.config  # Type annotation for the existing field self.config. The annotation is not updated automatically if you just type-hint this class's constructor argument, because self.config gets its type from the signature of the super constructor. The alternative to a generic is that you repeat this expression here for each separate config.
        self.core = self.buildCore(config)

    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        **kwargs
    ) -> AllHiddenStatesAndPooling:
        pass

    def __call__(self, *args, **kwargs) -> AllHiddenStatesAndPooling:
        return super().__call__(*args, **kwargs)  # Just calls forward() which we know is adapted to the model to return the BMOWPACA. This method is here just for type annotation of 'model(x)'.

    #########################
    ##### CLASS METHODS #####
    #########################

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

    @abstractmethod
    def forward(
        self,
        encoder_output: AllHiddenStatesAndPooling,
        attention_mask: Tensor,
        **kwargs
    ) -> Tensor:
        pass

    def __call__(self, *args, **kwargs) -> Tensor:
        return super().__call__(*args, **kwargs)

    #########################
    ##### CLASS METHODS #####
    #########################

    @classmethod
    def fromConfigs(cls, base_config: BaseModelConfig, head_config: HC) -> Self:
        """
        When the developer overrides a method and changes the signature, he gets a warning. This is not the case for
        constructors. Hence, we use this wrapper to force subclasses to have the same constructor arguments.

        This class has no constructor because we don't actually care about storing these configs inside the head.
        ModelWithHead already stores them anyway.
        """
        return cls(base_config, head_config)

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
class ModelWithHeadOutput(ModelOutput):
    logits: Tensor
    base_model_output: Optional[AllHiddenStatesAndPooling]=None  # Has to be optional because of some stupid rule in ModelOutput.__post_init__()
    loss: Optional[Tensor]=None


class CombinedConfig(PretrainedConfig, Generic[PC,HC]):

    model_type = "ArchIt (not recognised by AutoModel)"

    def __init__(self, base_model_config: Union[dict,PC]=None, head_config: Union[dict,HC]=None,
                 base_model_config_class: Type[PC]=None, head_config_class: Type[HC]=None, **kwargs):
        """
        This is where all the ugly config hardcoding happens thanks to HuggingFace being fake and gay. I hope to repent
        for the sin of polluting this overall beautiful OOP code with this class.

        Note: because HuggingFace's Config.from_pretrained() calls the constructor with only **kwargs and with dictionaries
        rather than objects as values, you need the default 'None' and you need to support dictionary values too.

        Also, because you can't (and shouldn't) deduce from the config which config classes were used, this class has
        to be informed about them. For the PretrainedModel.from_pretrained() call, those will all be in **kwargs,
        but the **kwargs in this constructor don't include them because we mention them explicitly in the signature.
        """
        super().__init__(**kwargs)
        self.is_composition = True  # HuggingFace's way of saying "run serialisation on the fields to serialise me" nested configs.

        # Catch cases where the arguments are coming from a JSON for e.g. RobertaConfig rather than a CombinedConfig.
        if base_model_config is None:
            if head_config is None:
                raise ValueError("When you unpack a PretrainedConfig and give the result to the constructor of a CombinedConfig, you should also supply a HeadConfig as an extra argument.")
            base_model_config = base_model_config_class(**kwargs)
        else:
            assert head_config is not None, "Unpacked a CombinedConfig but it was missing a head config without missing a base model config... That's impossible."
            if isinstance(base_model_config, dict):
                base_model_config = base_model_config_class(**base_model_config)

        if isinstance(head_config, dict):
            head_config = head_config_class(**head_config)

        assert isinstance(base_model_config, base_model_config_class)
        assert isinstance(head_config, head_config_class)
        self.base_model_config = base_model_config
        self.head_config       = head_config

    def to_dict(self) -> dict:
        d = super().to_dict()
        for k in list(d.keys()):
            if isinstance(d[k], HeadConfig):
                d[k] = d[k].__dict__  # If HeadConfig ever stores objects, replace this .__dict__ by a method .to_dict() that can manage the nesting.
        return d

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path: str, **kwargs) -> Tuple[dict, dict]:
        json_dict, remaining_kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)

        kwargs_to_add_to_dict = ["head_config", "base_model_config_class", "head_config_class"]
        for key in kwargs_to_add_to_dict:
            if key in remaining_kwargs:
                json_dict[key] = remaining_kwargs.pop(key)

        return json_dict, remaining_kwargs


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
        self.supports_gradient_checkpointing = model.core.supports_gradient_checkpointing

        self.model         = model
        self.head          = head
        self.loss_function = loss

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        **kwargs
    ) -> ModelWithHeadOutput:
        base_output = self.model(input_ids, attention_mask, **kwargs)
        logits      = self.head(base_output, attention_mask, **kwargs)
        return ModelWithHeadOutput(
            base_model_output=base_output,
            logits=logits,
            loss=self.computeLoss(logits, labels) if labels is not None else None
        )

    def __call__(self, *args, **kwargs) -> ModelWithHeadOutput:
        return super().__call__(*args, **kwargs)

    @abstractmethod
    def computeLoss(self, logits: Tensor, labels: Tensor) -> FloatTensor:
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
    def head_class(cls) -> Type[Head[HC]]:  # Because all heads have the same constructor method .fromConfigs(), all you need to .buildHead() is to know the class to call it on.
        pass

    @classmethod
    def buildHead(cls, base_model_config: BaseModelConfig, head_config: HC) -> Head[HC]:
        return cls.head_class.fromConfigs(base_model_config, head_config)

    @classmethod
    @abstractmethod
    def buildLoss(cls) -> _Loss:
        pass

    @classmethod
    def from_pretrained(cls, checkpoint: str, base_model_class: Type[BaseModel], head_config: HC=None) -> Self:
        """
        Load the model from an existing checkpoint. Distinguishes between two cases:
            - The given checkpoint is of the correct model-with-head. In that case, no head config is needed.
            - The given checkpoint is of a model without a head. In that case, the head config is used to construct a new head.

        How HuggingFace's .from_pretrained() loads weights is that it first creates an instance of the class by calling
        its constructor, and then it reads the provided checkpoint file by querying whatever fields there exist in that
        instance. Usually, that constructor only takes a config and explicitly defines the PyTorch module tree.
        Because a ModelWithHead doesn't know which base model will be used beforehand, it doesn't have a pre-determined
        module tree and cannot just be constructed with a config alone. (Not even by saving the class name in the config,
        because we don't have a mapping between the names and locations of custom classes.)

        .from_pretrained() passes through *args and **kwargs to the constructor, which we can use to inform it. This
        method's responsibility is to create those *args and **kwargs.
        """
        base_model_config = base_model_class.config_class.from_pretrained(checkpoint)  # FIXME: If the given checkpoint is for a model-with-head, this does not work because the model config is nested one level deep.
        base_model_shell = base_model_class(base_model_config)  # A mix between .from_config (which only exists on auto classes and expects a config from the user) and .from_pretrained (just to get the config so the user doesn't have to supply it).
        return super().from_pretrained(
            checkpoint,
            # Unnamed arguments (*args) passed to PretrainedModel.from_pretrained() are passed straight to the constructor of this class, except the checkpoint is replaced by a config.
            base_model_shell,
            cls.buildHead(base_model_class.standardiseConfig(base_model_config), head_config),
            cls.buildLoss(),

            # Keyword arguments (**kwargs) are passed first to PretrainedConfig.from_pretrained() and only what remains ends up in PretrainedModel.from_pretrained(). Do note that PretrainedConfig.from_pretrained() does NOT pass these **kwargs to the constructor and only uses them for administrative stuff by default; I forcibly add them in get_config_dict().
            head_config=head_config,
            base_model_config_class=base_model_class.config_class,
            head_config_class=cls.head_class.config_class
        )

    @classmethod
    def fromModelAndHeadConfig(cls, base_model: BaseModel, head_config: HC) -> Self:
        return cls(
            CombinedConfig(base_model_config=base_model.config, head_config=head_config),
            base_model,
            cls.buildHead(base_model.standardiseConfig(base_model.config), head_config),
            cls.buildLoss()
        )

    base_model_prefix = "model.core"

    @classmethod
    def _load_pretrained_model(cls, model, state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, **kwargs):
        """
        Indirection to trick PyTorch into recognising the base model in a checkpoint.

        Here's the situation: when a Module hierarchy is saved (resulting in a "state dictionary" with signature
        something like Dict[str, Module]), every Module is identified by a dot-separated sequence of fields to traverse to get to it.
        In HuggingFace, every model stores a "prefix" which is equal to the name of the variable you store the base model in
        when you pack it with a head. For example, the prefix for RoBERTa is "roberta" and RobertaForMaskedLM has two fields [roberta: RobertaModel, lm_head: RobertaLMHead].
        On the other hand, RobertaModel has no field "roberta" because it is the base model.

        The assumption is now that you only ever call .from_pretrained() between models that have the same prefix. The
        4 possible variations of this situation are supported by looking for the prefix and removing it where needed,
        and then assuming all module identifiers are an exact match.

        What we want to do, however, is load from a [roberta, lm_head] model into a [wrapper, head] where we want
        the roberta field to land in the field of model. This is impossible, because finding the model weights require trimming the
        "roberta" prefix from the state dict but trimming "wrapper.model" from the model skeleton. So, what we do is trim
        the "roberta" prefix from the state dict while pretending that "wrapper.model" is the prefix and that the [roberta, lm_head]
        model is actually a base model.
        """
        if set(state_dict.keys()) != {"model", "head"}:  # You've loaded a HF checkpoint.
            consume_prefix_in_state_dict_if_present(state_dict, model.model.core.base_model_prefix + ".")  # In-place.
            loaded_keys = list(state_dict.keys())

        return super()._load_pretrained_model(model, state_dict, loaded_keys, resolved_archive_file, pretrained_model_name_or_path, **kwargs)
