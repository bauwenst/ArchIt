"""
Defines the three config objects for ArchIt's three main classes.
"""
from abc import abstractmethod, ABC
from typing import TypeVar, Generic, Type, Union, Tuple, Protocol, Any
from dataclasses import dataclass

from transformers import PretrainedConfig


__all__ = ["BaseModelConfig", "HeadConfig", "CombinedConfig", "PC", "HC"]


@dataclass
class BaseModelConfig:
    """
    Unbelievably, HuggingFace's PretrainedConfig, the parent class for all config classes, does NOT provide
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


class Dictable(Protocol):
    """Anything that has a method to_dict() for being turned into a dictionary. Due to duck typing, this includes PretrainedConfig."""
    def to_dict(self) -> dict:
        pass


class RecursiveSerialisable(Dictable):
    """
    Small class that implements a recursive to_dict() method on top of __dict__.
    """

    def _fields_to_dictables(self) -> dict[str, Union[Any,Dictable]]:
        """
        Non-recursively take whatever information exists in this object and turn it into a dictionary of things that
        potentially have a .to_dict() method to further turn them into dictionaries.
        """
        return self.__dict__

    def to_dict(self) -> dict[str, Any]:
        d = self._fields_to_dictables()
        for k in list(d.keys()):
            try:
                d[k] = d[k].to_dict()  # If d[k] has the method, it must be executed.
            except:
                pass  # If d[k] does not have the method, it is assumed to be immediately serialisable. No .__dict__ is called on it either.
        return d


class RecursiveAccessible(ABC):
    """
    Small class that implements a recursive __getattr__ that goes looking for nested fields if a field doesn't exist.
    """

    @abstractmethod
    def _fallback_fields(self) -> list[str]:
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


class RecursivePretrainedConfig(PretrainedConfig, RecursiveSerialisable, RecursiveAccessible, ABC):
    """
    Allows recursively accessing fields and recursively serialising, and additionally provides all the functionality
    of HuggingFace's PretrainedConfig. The assumption is that some of your fields are themselves a PretrainedConfig.

    Deletes bullshit fields added by HuggingFace.
    Not a dataclass because you need the super() call in the constructor.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._deleteTopLevelFields()
        self.is_composition = True  # HuggingFace's way of saying "run serialisation on the fields when you serialise me" nested configs. TODO: I think this feature has since been removed.
        self.has_no_defaults_at_init = True  # Needed because otherwise the assertions below get triggered due to HF trying to re-instantiate the class without any constructor arguments.

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

    @abstractmethod
    def _serialised_fields(self) -> list[str]:
        """Names of fields that should be kept when serialising."""
        pass

    @abstractmethod
    def _constructor_runtime_arguments(self) -> list[str]:
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
    def get_config_dict(cls, pretrained_model_name_or_path: str, **kwargs) -> Tuple[dict, dict]:
        json_dict, remaining_kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)

        kwargs_to_add_to_dict = cls._constructor_runtime_arguments()
        for key in kwargs_to_add_to_dict:
            if key in remaining_kwargs:
                json_dict[key] = remaining_kwargs.pop(key)

        return json_dict, remaining_kwargs

    def _get_non_default_generation_parameters(self) -> dict:  # Needed to avoid triggering this bug (which will be fixed in transformers after October 2024): https://github.com/huggingface/transformers/pull/33934
        return dict()

    def to_dict(self) -> dict[str, Any]:  # PretrainedConfig's to_dict has priority, so we override that behaviour.
        return RecursiveSerialisable.to_dict(self)

    ### Implementations that hardcode some HuggingFace properties

    def _deleteTopLevelFields(self):
        """
        Unlink all top-level fields added by HuggingFace. I didn't ask for them. If you need a field, go look for it in
        the subconfigs.
        """
        for k, v in list(self.__dict__.items()):
            # if v is None:
            if k not in {"_name_or_path", "transformers_version", "architectures"}:
                delattr(self, k)

    def _fields_to_dictables(self) -> dict[str, Union[Any,Dictable]]:
        fields_as_dict = PretrainedConfig.to_dict(self)  # Although confusing, this to_dict should be read as being equivalent to _fields_to_dictables, which is called by RecursiveSerialisable's to_dict.

        fields_to_keep = self._serialised_fields() + ["torch_dtype", "transformers_version"]
        return {k:v for k,v in fields_as_dict.items() if k in fields_to_keep}  # Pop the gigantic amount of shit we didn't ask for.


class HeadConfig(RecursiveSerialisable):
    pass


PC = TypeVar("PC", bound=PretrainedConfig)
HC = TypeVar("HC", bound=HeadConfig)
class CombinedConfig(RecursivePretrainedConfig, Generic[PC,HC]):

    model_type = "ArchIt (ignore this warning, it is raised by AutoModel)"

    def __init__(self, base_model_config: Union[dict,PC]=None, head_config: Union[dict,HC]=None,
                 base_model_config_class: Type[PC]=None, head_config_class: Type[HC]=None, **kwargs):
        """
        Note: because HuggingFace's Config.from_pretrained() calls the constructor with only **kwargs and with dictionaries
        rather than objects as values, you need the default parameter values 'None' and you need to support dictionary values too.

        Also, because you can't (and shouldn't) deduce from a JSON dictionary which base model class and head class were used,
        this class has to be informed about them. For the PretrainedModel.from_pretrained(checkpoint, **kwargs) call,
        those will all be in **kwargs. The implementation of RecursivePretrainedConfig.get_config_dict(...) above will pop
        from those kwargs the parameters that appear this constructor's signature, so that the **kwargs received here are a subset of the original kwargs.
        """
        super().__init__(**kwargs)

        # Catch cases where the arguments are coming from a JSON for e.g. RobertaConfig rather than a CombinedConfig.
        if base_model_config is None:
            if head_config is None:
                raise ValueError("When you unpack a PretrainedConfig and give the result to the constructor of a CombinedConfig, you should also supply a HeadConfig as an extra argument.")
            base_model_config = base_model_config_class(**kwargs)
        else:
            assert head_config is not None, "Unpacked a CombinedConfig but it was missing a head config without missing a base model config... That's impossible."
            if isinstance(base_model_config, dict):
                base_model_config = base_model_config_class(**base_model_config)
            else:
                base_model_config_class = base_model_config.__class__

        if isinstance(head_config, dict):  # Deserialise head. Its class has to be known for this.
            head_config = head_config_class(**head_config)
        else:  # Head is already deserialised. We can impute its class.
            head_config_class = head_config.__class__

        assert isinstance(base_model_config, base_model_config_class)
        assert isinstance(head_config, head_config_class)
        self.base_model_config = base_model_config
        self.head_config       = head_config

    def _fallback_fields(self) -> list[str]:  # We try to get missing fields from the raw PretrainedConfig of the base model which is likely what the user meant.
        return ["base_model_config"]

    def _serialised_fields(self) -> list[str]:
        return ["base_model_config", "head_config"]

    def _constructor_runtime_arguments(self) -> list[str]:
        return ["head_config", "base_model_config_class", "head_config_class"]

    def _syncSubconfigs(self):
        pass
