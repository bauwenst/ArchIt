"""
Defines the three config objects for ArchIt's three main classes.
"""
from typing import TypeVar, Generic, Type, Union, Tuple
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


class RecursiveSerialisable:
    """
    Small class that implements a recursive to_dict() method on top of __dict__.
    """

    def _fields_to_dict(self):
        return self.__dict__

    def to_dict(self) -> dict:
        d = self._fields_to_dict()
        for k in list(d.keys()):
            try:
                d[k] = d[k].to_dict()  # If d[k] has the method, it must be executed.
            except:
                pass  # If d[k] does not have the method, it is assumed to be immediately serialisable. No .__dict__ is called on it either.
        return d


class HeadConfig(RecursiveSerialisable):
    pass


PC = TypeVar("PC", bound=PretrainedConfig)
HC = TypeVar("HC", bound=HeadConfig)
class CombinedConfig(PretrainedConfig, Generic[PC,HC], RecursiveSerialisable):

    model_type = "ArchIt (ignore this warning, it is raised by AutoModel)"

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
        self.deleteTopLevelFields()
        self.is_composition = True  # HuggingFace's way of saying "run serialisation on the fields when you serialise me" nested configs.
        self.has_no_defaults_at_init = True  # Needed because otherwise the assertions below get triggered due to HF trying to re-instantiate the class without any constructor arguments.

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

    def __getattr__(self, item):
        """
        This is used as the fallback when self.item does not exist.
        We try to get it from the raw PretrainedConfig of the base model which is likely what the user meant.
        """
        try:
            return super().__getattr__(item)  # The old implementation. Will find methods and fields if they exist.
        except AttributeError:
            if "base_model_config" in self.__dict__:
                return getattr(self.base_model_config, item)  # If the field self.base_model_config doesn't exist, you call this same __getattr__ method recursively, hence the "if" around it.
            else:
                raise AttributeError(item)

    def _fields_to_dict(self):
        fields_as_dict = PretrainedConfig.to_dict(self)  # same as super().to_dict() but with a specific super() because two of the parents have a to_dict().

        fields_to_keep = ["base_model_config", "head_config", "torch_dtype", "transformers_version"]
        return {k:v for k,v in fields_as_dict.items() if k in fields_to_keep}  # Pop the gigantic amount of shit we didn't ask for.

    def to_dict(self) -> dict:
        return RecursiveSerialisable.to_dict(self)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path: str, **kwargs) -> Tuple[dict, dict]:
        json_dict, remaining_kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)

        kwargs_to_add_to_dict = ["head_config", "base_model_config_class", "head_config_class"]
        for key in kwargs_to_add_to_dict:
            if key in remaining_kwargs:
                json_dict[key] = remaining_kwargs.pop(key)

        return json_dict, remaining_kwargs

    def _get_non_default_generation_parameters(self) -> dict:  # Needed to avoid triggering this bug (which will be fixed in transformers after October 2024): https://github.com/huggingface/transformers/pull/33934
        return dict()

    def deleteTopLevelFields(self):
        """
        Unlink all top-level fields added by HuggingFace. I didn't ask for them. If you need a field, go look for it in
        the subconfigs.
        """
        for k,v in list(self.__dict__.items()):
            # if v is None:
            if k not in {"_name_or_path", "transformers_version", "architectures"}:
                delattr(self, k)
