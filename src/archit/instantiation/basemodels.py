from torch import Tensor
from transformers.models.roberta import RobertaModel, RobertaConfig

from .abstracts import BaseModel, BaseModelConfig, BMOWPACA


class RobertaBaseModel(BaseModel[RobertaConfig]):

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        **kwargs
    ) -> BMOWPACA:
        return self.core(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )  # Note that this call is Any-typed. This is exactly we we need to type-cast by declaring the return type as BMOWPACA.

    @classmethod
    def buildCore(cls, raw_config: RobertaConfig):
        return RobertaModel(raw_config)

    @classmethod
    def standardiseConfig(cls, raw_config: RobertaConfig) -> BaseModelConfig:
        return BaseModelConfig(
            hidden_size=raw_config.hidden_size,
            hidden_dropout_prob=raw_config.hidden_dropout_prob,
            vocab_size=raw_config.vocab_size,

            num_hidden_layers=raw_config.num_hidden_layers,
            num_attention_heads=raw_config.num_attention_heads
        )

    @classmethod
    @property
    def config_class(cls):
        return RobertaConfig
