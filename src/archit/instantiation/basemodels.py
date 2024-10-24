from torch import Tensor

from .abstracts import BaseModel, BaseModelConfig, AllHiddenStatesAndPooling

########################################################################################################################

from transformers.models.roberta import RobertaModel, RobertaConfig

class RobertaBaseModel(BaseModel[RobertaConfig]):

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        do_drop_intermediates: bool=True,
        **kwargs
    ) -> AllHiddenStatesAndPooling:
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=not do_drop_intermediates,
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
            num_attention_heads=raw_config.num_attention_heads,
            context_length=raw_config.max_position_embeddings
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
        do_drop_intermediates: bool=True,
        **kwargs
    ) -> AllHiddenStatesAndPooling:
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=not do_drop_intermediates,
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
            num_attention_heads=raw_config.num_attention_heads,
            context_length=raw_config.n_positions
        )

    @classmethod
    @property
    def config_class(cls):
        return GPT2Config
