"""
Defines the three config objects for ArchIt's three main classes.
"""
from __future__ import annotations
from typing import TypeVar, Generic, Union, Protocol, Any, TYPE_CHECKING, Optional
from typing_extensions import Self, runtime_checkable
from abc import abstractmethod, ABC, ABCMeta
from dataclasses import dataclass, field
from transformers import PreTrainedConfig

import json

from ..util import is_abstract_method


__all__ = [
    "BaseModelConfig",
    "TensorSignature", "HeadConfig", "HeadConfigWithLabels", "HeadConfigWithModelTying", "TokenClassifierHeadConfig", "PooledClassifierHeadConfig", "TokenVocabularyHeadConfig",
    "ArchitConfig", "PC", "HC"
]


@dataclass
class BaseModelConfig:
    """
    Unbelievably, HuggingFace's PreTrainedConfig, the parent class for all config classes, does NOT provide
    a universal interface to get basic model properties like amount of layers, hidden size and so on.
    Instead, it says in its documentation that those properties "are common" in all subclasses. I don't play that way.
    """
    # Will be relevant for heads to inspect:
    hidden_size: int
    hidden_dropout_prob: float  # TODO: Should probably use the MLP dropout, but even better would be if the user had to specify a dropout for the head.
    vocab_size: int

    # Will not be relevant, but should still be available on base model configs:
    num_hidden_layers: int
    num_attention_heads: int
    context_length: int


class Dictable(ABC):
    """Anything that has a method to_dict() for being turned into a dictionary. Due to duck typing, this includes PreTrainedConfig."""
    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        pass


# "Extend" the JSON package's encoder to support Dictables.
_default_json_encoder = json.JSONEncoder.default

def _extended_default_json_encoder(self, obj):
    if isinstance(obj, Dictable):
        return obj.to_dict()
    return _default_json_encoder(self, obj)

json.JSONEncoder.default = _extended_default_json_encoder


class RecursiveSerialisable(Dictable):
    """
    Small class that implements a recursive to_dict() method on top of __dict__.
    """

    def _fields_to_dictables(self) -> dict[str, Union[Any,Dictable]]:
        """
        Non-recursively take whatever information exists in this object and turn it into a dictionary of things that
        potentially have a .to_dict() method to further turn them into dictionaries.
        """
        return dict(self.__dict__)

    def to_dict(self) -> dict[str, Any]:
        d = self._fields_to_dictables()
        for k in list(d.keys()):
            try:
                d[k] = d[k].to_dict()  # If d[k] has the method, it must be executed.
            except:
                pass  # If d[k] does not have the method, it is assumed to be immediately serialisable. No .__dict__ is called on it either.
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)


class RecursiveAccessible(ABC):
    """
    Small class that implements a recursive __getattr__ that goes looking for nested fields if a field doesn't exist.
    """

    @classmethod
    @abstractmethod
    def _fallback_fields(cls) -> list[str]:
        pass

    def __getattr__(self, item):
        """
        When you can't find an item on this object (self.item), go look inside the fallback fields in order, then error.
        """
        try:
            return super().__getattr__(item)  # The old implementation. Will find methods and fields if they exist. (Should basically never be the case because __getattr__ is by definition for when they don't exist.)
        except AttributeError:
            for fallback in self._fallback_fields():
                if fallback not in self.__dict__:  # This saves you from infinite recursion. Even if your fallback fields are defined in the constructor, they may not exist in self.__dict__ at some point, for example during a deepcopy which calls getattr before calling __init__.
                    continue
                try:
                    obj = getattr(self, fallback)  # Calling getattr() on self will call this same __getattr__ method, BUT because we know that the given name is an existing field in self, we know that this new call will ALWAYS succeed on super().__getattr__().
                    return getattr(obj, item)  # This can raise an AttributeError.
                except AttributeError:
                    pass
            else:
                raise AttributeError(item)

    def __delattr__(self, item):  # Deletes one, not all instances, just like __getattr__ returns only the first instance. NOTE: Better not delete a dataclass attribute. It has unexpected behaviour.
        try:
            return super().__delattr__(item)
        except AttributeError:
            for fallback in self._fallback_fields():
                if fallback not in self.__dict__:
                    continue
                try:
                    obj = getattr(self, fallback)
                    return delattr(obj, item)
                except AttributeError:
                    pass
            else:
                raise AttributeError(item)


class RecursivePreTrainedConfig(PreTrainedConfig, RecursiveSerialisable, RecursiveAccessible, ABC):
    """
    Allows recursively accessing fields and recursively serialising, and additionally provides all the functionality
    of HuggingFace's PreTrainedConfig. The assumption is that some of your fields are themselves a PreTrainedConfig.

    Deletes bullshit fields added by HuggingFace.
    Does not use HuggingFace's subconfig system because it adds bloat.
    Not a dataclass because you need the super() call in the constructor.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._deleteTopLevelFields()

    ### Abstracts

    @abstractmethod
    def _syncSubconfigs(self):
        """
        Synchronise corresponding values in the different subconfigs.

        Note: It is the CHILD class which is responsible for calling this.
        The reason this is an abstract method is not that it is already called anywhere, but simply to
        remind the implementers that there may be fields to be synchronised across multiple subconfigs.
        """
        pass

    @classmethod
    @abstractmethod
    def _serialised_fields(cls) -> list[str]:
        """Names of fields that should be kept when serialising."""
        pass

    @classmethod
    @abstractmethod
    def _constructor_runtime_arguments(cls) -> list[str]:
        """
        Names of constructor arguments which, rather than coming from a JSON, can also come from the user at runtime
        when calling .from_pretrained(). For example, if your constructor has arguments x and y, then x may come from
        the checkpoint and y from the user, so you would call .from_pretrained(checkpoint, y=...) and you would declare
        "y" in this method.

        If you save all information in the JSON, then likely you just want [] here.
        """
        pass

    ### HuggingFace overrides

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path: str, **kwargs) -> tuple[dict, dict]:
        json_dict, remaining_kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        if not json_dict:  # Since HF won't raise an error when the checkpoint literally doesn't fucking exist, we will.
            raise FileNotFoundError(f"No model config was found for checkpoint '{pretrained_model_name_or_path}'.")

        # Go through the kwargs and move any keys you recognise to the main dict. THIS WILL OVERRIDE the values of those keys coming from the JSON!
        for key in cls._constructor_runtime_arguments():
            if key in remaining_kwargs:
                json_dict[key] = remaining_kwargs.pop(key)

        return json_dict, remaining_kwargs

    def _get_non_default_generation_parameters(self) -> dict:  # Needed to avoid triggering this bug (which will be fixed in transformers after October 2024): https://github.com/huggingface/transformers/pull/33934
        return dict()

    def to_dict(self) -> dict[str, Any]:
        return RecursiveSerialisable.to_dict(self)  # PreTrainedConfig's to_dict otherwise has priority, so we override that behaviour.

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs) -> Self:
        return PreTrainedConfig.from_dict.__func__(cls, config_dict, **kwargs)  # Equivalent to cls(**config_dict) but with some extra HF stuff. The reason we use __func__ is so that the cls parameter is not bound to cls=PreTrainedConfig but rather the cls here.

    ### Implementations that hardcode some HuggingFace properties

    def _deleteTopLevelFields(self):
        """
        Unlink all top-level fields added by HuggingFace. I didn't ask for them. If you need a field, go look for it in
        the subconfigs.
        """
        protected_fields = set(self._serialised_fields()) | {"_name_or_path", "transformers_version", "architectures"}
        for k, v in list(self.__dict__.items()):
            # if v is None:
            if k not in protected_fields:
                delattr(self, k)

    def _fields_to_dictables(self) -> dict[str, Union[Any,Dictable]]:
        fields_as_dict = PreTrainedConfig.to_dict(self)  # Although confusing, this to_dict should be read as being equivalent to _fields_to_dictables, which is called by RecursiveSerialisable's to_dict.

        fields_to_keep = self._serialised_fields() + ["torch_dtype", "transformers_version"]
        return {k:v for k,v in fields_as_dict.items() if k in fields_to_keep}  # Pop the gigantic amount of shit we didn't ask for.


class _TensorSignatureABCMeta(ABCMeta):
    """
    Enforces that headClass() must be defined only after defining getInputSignature() and getOutputSignature().
    """
    def __new__(mcs, name, bases, namespace, **kwargs):  # Runs when a subclass is declared.
        __cls = super().__new__(mcs, name, bases, namespace, **kwargs)  # namespace is the code written in the specific subclass.

        # Now that the class exists, we can check method order.
        __earlier_methods = ["getInputSignature", "getOutputSignature"]
        __later_method = "headClass"
        if __later_method in namespace and not is_abstract_method(__cls, __later_method):  # => This class defines it concretely.
            if any(m in namespace for m in __earlier_methods):
                raise TypeError("Cannot define tensor signature (which is true for many head classes) at the same time as the head class itself. Do this in a parent class.")
            elif any(is_abstract_method(__cls, m) for m in __earlier_methods):  # If this check raises an AttributeError, you have put the metaclass on a class that does not even define abstractly define the earlier methods, which is bad.
                raise TypeError(f"Cannot define head class when tensor signature methods aren't implemented yet. (Do that in a parent class.)")

        return __cls


class TensorSignature(metaclass=_TensorSignatureABCMeta):
    """
    Declares the input and output shapes of the tensors of a module.

    In ArchIt, we use this to check whether a head is compatible with a task at runtime, after the user has potentially
    ignored the type checker's warnings about a wrong signature, but before any GPU operations have been performed.

    You can choose whether you implement the methods of this class in pure TensorSignature subclasses, or directly in a
    HeadConfig subclass. Either way, you should not implement them in leaf subclasses of HeadConfig, because those are
    for heads with a concrete forward() method and the whole point of signatures is that they stretch across more than one forward()
    implementation.
    """
    @classmethod
    @abstractmethod
    def getInputSignature(cls) -> tuple:
        pass

    @classmethod
    @abstractmethod
    def getOutputSignature(cls) -> tuple:
        pass


if TYPE_CHECKING:
    from .abstracts import Head


class HeadConfig(RecursiveSerialisable, TensorSignature):
    """
    Config that determines both which Head class will be put on top of the model,
    and how it is parameterised.
    """
    @classmethod
    @abstractmethod
    def headClass(cls) -> type[Head]:
        """Which class is constructed by this head config."""
        pass

    def assertConstraints(self):
        """Called at head construction time (not config construction time) to validate that it would produce a feasible head."""
        pass

    def buildHead(self, base_config: BaseModelConfig) -> Head:
        return self.headClass()(base_config, self)


@dataclass
class HeadConfigWithLabels(HeadConfig):   # Your IDE might complain about extensions of this class, but due to the kw_only flag, it's all legal.
    n_classes: int = field(default=0, kw_only=True)  # It has a default so that users don't have to specify this (tasks automate it).
    label_loss_weights: Optional[list[float]] = None

    def assertConstraints(self):
        assert self.n_classes > 0
        if self.label_loss_weights is not None:
            assert all(weight >= 0 for weight in self.label_loss_weights)
            assert len(self.label_loss_weights) == self.n_classes

    def from_dict(cls, data: dict[str, Any]) -> Self:
        if "num_labels" in data:  # num_labels was grandfathered in from HF, but it is a misnomer because single-label classification has num_labels > 1.
            data["n_classes"] = data["num_labels"]
        return super().from_dict(data)


@dataclass
class TokenClassifierHeadConfig(HeadConfigWithLabels):
    @classmethod
    def getInputSignature(self) -> tuple:
        return ("N", "L", "H")

    @classmethod
    def getOutputSignature(self) -> tuple:
        return ("N", "L", "C")


@dataclass
class PooledClassifierHeadConfig(HeadConfigWithLabels):
    @classmethod
    def getInputSignature(self) -> tuple:
        return ("N", "L", "H")

    @classmethod
    def getOutputSignature(self) -> tuple:
        return ("N", "C")


@dataclass
class HeadConfigWithModelTying(HeadConfig):
    """Head config allowing the relevant head weights to be tied to weight that do not live in the head."""
    tie_to_model: bool = False


@dataclass
class TokenVocabularyHeadConfig(HeadConfigWithModelTying):
    @classmethod
    def getInputSignature(self) -> tuple:
        return ("N", "L", "H")

    @classmethod
    def getOutputSignature(self) -> tuple:
        return ("N", "L", "V")


PC = TypeVar("PC", bound=PreTrainedConfig)
HC = TypeVar("HC", bound=HeadConfig)
class ArchitConfig(RecursivePreTrainedConfig, Generic[PC,HC]):

    model_type = "ArchIt (ignore this warning, it is raised by AutoModel)"

    def __init__(self, base_model_config: Union[dict,PC]=None, head_config: Union[dict,HC]=None,
                 base_model_config_class: type[PC]=None, head_config_class: type[HC]=None, **kwargs):
        """
        Note: because HuggingFace's Config.from_pretrained() calls the constructor with only **kwargs and with dictionaries
        rather than objects as values, you need the default parameter values 'None' and you need to support dictionary values too.

        Also, because you can't (and shouldn't) deduce from a JSON dictionary which base model class and head class were used,
        this class has to be informed about them. For the PretrainedModel.from_pretrained(checkpoint, **kwargs) call,
        those will all be in **kwargs. The implementation of RecursivePreTrainedConfig.get_config_dict(...) above will pop
        from those kwargs the parameters that appear this constructor's signature, so that the **kwargs received here are a subset of the original kwargs.
        """
        if head_config is None:
            raise ValueError("Missing head information. Provide it either as a dictionary (with a class) or as a HeadConfig object.")
        if base_model_config_class is None and isinstance(base_model_config, dict):
            raise ValueError("Cannot parse base model subconfig without base_model_config_class.")
        if head_config_class is None and isinstance(head_config, dict):
            raise ValueError("Cannot parse head subconfig without head_config_class.")

        # Base config
        if base_model_config is None:  # => Information must be in kwargs, coming from e.g. RobertaConfig checkpoint rather than an ArchitConfig.
            base_model_config: PC = base_model_config_class.from_dict(kwargs)
        else:
            if isinstance(base_model_config, dict):
                base_model_config: PC = base_model_config_class.from_dict(base_model_config)
            else:
                base_model_config_class = base_model_config.__class__

        # Head config
        if isinstance(head_config, dict):  # Deserialise head. Its class has to be known for this.
            head_config: HC = head_config_class.from_dict(head_config)
        else:  # Head is already deserialised. This happens when manually calling this constructor with a head config (e.g. from LaMoTO hyperparameters), but also during .from_pretrained() with kwargs that overwrite the deserialised dictionary.
            if head_config_class is not None and head_config_class != type(head_config):
                raise ValueError(f"Mismatch between class of given head config ({type(head_config).__name__}) and given head config class ({head_config_class.__name__}).")
            head_config_class = head_config.__class__

        assert isinstance(base_model_config, base_model_config_class)
        assert isinstance(head_config, head_config_class)

        # Now that we know both configs are of the types they should be, rewire weight tying from transformers v4 to v5.
        if hasattr(base_model_config, "tie_word_embeddings"):
            tied_weights_requested_base = base_model_config.tie_word_embeddings
            if isinstance(head_config, HeadConfigWithModelTying):
                if tied_weights_requested_base ^ head_config.tie_to_model:
                    raise ValueError(f"Head weights should{' ' if tied_weights_requested_base else ' not '}be tied according to the base model config (tie_word_embeddings), which contradicts the head config. Fix this mismatch.")
            else:
                pass  # This is probably quite normal during finetuning, so we don't raise an error nor a warning.
            # delattr(base_model_config, "tie_word_embeddings")  # This is unnecessary and also buggy, because in dataclasses you just reset the field to its default value.

        self.base_model_config: PC = base_model_config
        self.head_config: HC       = head_config

        super().__init__(**kwargs)

    @classmethod
    def _fallback_fields(cls) -> list[str]:  # We try to get missing fields from the raw PreTrainedConfig of the base model which is likely what the user meant.
        return ["base_model_config"]

    @classmethod
    def _serialised_fields(cls) -> list[str]:
        return ["base_model_config", "head_config"]

    @classmethod
    def _constructor_runtime_arguments(cls) -> list[str]:
        return ["head_config", "base_model_config_class", "head_config_class"]

    def _syncSubconfigs(self):
        pass

    @classmethod
    def isNativeArchit(cls, checkpoint: str) -> bool:
        """Checks whether a pretrained checkpoint is an ArchIt checkpoint. Throws an error if the checkpoint doesn't exist."""
        raw = cls.get_config_dict(checkpoint)
        return all(field in raw for field in cls._serialised_fields())

    # The attention and MoE implementations only matter for the base model. They should NOT be gettable from ArchitConfig
    # because then ArchitConfig will check ModelWithHead for _supports_XYZ properties. They should NOT be settable on
    # ArchitConfig, because that is not where the base model will look to decide its implementation (rather, it will
    # eventually read its subconfig to decide which implementation to use). However, because ModelWithHead.from_pretrained()
    # should still allow a kwarg to set these implementations, setting it on ArchitConfig is a valid operation which just
    # redirects to the base model config.

    def __getattr__(self, item: str):
        if item in {"_attn_implementation_internal", "_experts_implementation_internal"}:  # Explicitly forbid searching deeper.
            try:
                return self.__dict__[item]
            except KeyError:
                raise AttributeError
        else:  # The recursive implementation.
            return super().__getattr__(item)

    def __setattr__(self, key: str, value):
        if key in {"_attn_implementation_internal", "_experts_implementation_internal"}:  # Immediately go deeper.
            return setattr(self.base_model_config, key, value)
        else:  # The standard implementation.
            super().__setattr__(key, value)
