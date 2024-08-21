import torch
from torch import Tensor
from transformers.models.roberta import RobertaModel, RobertaConfig

from .abstracts import BaseModel, BaseModelConfig, BMOWPACA


class RobertaBaseModel(BaseModel[RobertaConfig]):

    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.roberta = RobertaModel(config)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        **kwargs
    ) -> BMOWPACA:
        return self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )  # Note that this call is Any-typed. This is exactly we we need to type-cast by declaring the return type as BMOWPACA.

    def convertConfig(self) -> BaseModelConfig:
        return BaseModelConfig(
            hidden_size=self.config.hidden_size,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            vocab_size=self.config.vocab_size,

            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads
        )
