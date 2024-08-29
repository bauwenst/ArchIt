from torch import Tensor

from .abstracts import BaseModel, BaseModelConfig, AllHiddenStatesAndPooling


from transformers.models.roberta import RobertaModel, RobertaConfig

class RobertaBaseModel(BaseModel[RobertaConfig]):

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        **kwargs
    ) -> AllHiddenStatesAndPooling:
        output = self.core(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        return AllHiddenStatesAndPooling(
            last_hidden_state=output.last_hidden_state,
            hidden_states=output.hidden_states,
            pooler_output=output.pooler_output
        )

    @classmethod
    def buildCore(cls, raw_config: RobertaConfig):
        return RobertaModel(raw_config, add_pooling_layer=False)

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


from transformers.models.gpt2 import GPT2Model, GPT2Config

class GPT2BaseModel(BaseModel[GPT2Config]):

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        **kwargs
    ) -> AllHiddenStatesAndPooling:
        output = self.core(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        return AllHiddenStatesAndPooling(
            last_hidden_state=output.last_hidden_state,
            hidden_states=output.hidden_states,
            pooler_output=None
        )

    @classmethod
    def buildCore(cls, raw_config: GPT2Config):
        return GPT2Model(raw_config)

    @classmethod
    def standardiseConfig(cls, raw_config: GPT2Config) -> BaseModelConfig:
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
        return GPT2Config
