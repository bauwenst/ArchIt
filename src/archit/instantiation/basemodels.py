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


########################################################################################################################

from dataclasses import dataclass
import torch
from transformers import PreTrainedModel
from supar import modules as snn  # Note: You must install supar from GitHub. The pip version is more than 2 years out of date! https://github.com/yzhangcs/parser
from supar.utils.fn import pad

from .abstracts import RecursiveSerialisable

@dataclass
class BaseModelExtendedConfig(RecursiveSerialisable):
    layer_pooling: int=1      # Embedding pooling per token across the last N layers (weighted sum with learnt softmax weights).
    word_pooling: str="mean"  # Embedding pooling per word across its tokens.
    stride: int=256           # Tokens to shift the context window when the input is too long.


class BaseModelExtended(BaseModel):
    """
    Wrapper around BaseModel that offers three advanced features for generating embeddings:
        - Layer pooling: the token embeddings that are then within-word-pooled.
        - Word-level pooling: whereas input_ids is normally B x pad(L_tokens), it is assumed to now be B x pad(L_words) x pad(L_tokens_per_word),
          i.e. there is an extra dimension that separates IDs by the word they belong to. This allows custom within-word embedding pooling.
        - Striding: the BaseModel has a finite context length to produce logits, but the head that processes these logits
          doesn't. Hence, rather than having your tokeniser truncate the input, you can stride across an arbitrarily long input in
          overlapping windows and concatenate the resulting embeddings for tokens that don't yet have one.

    Note that this is a subclass of BaseModel so that a ModelWithHead can store it in its model field rather than having
    the actual BaseModel there and then have a separate field for this class that *also* stores the actual BaseModel.
    Just don't call any of the class methods on this class because they will fail. (BaseModel class methods are only
    called during .from_pretrained() on a predetermined class anyway.)

    If I had to design this again, I would let BaseModelExtended instead be a subclass of ModelWithHead that just has a
    different forward() method, because it's actually not modelling a base model (i.e. adapting from HF to a universal
    interface), but rather modelling how to USE a base model (i.e. adapting from an already universal BaseModel interface
    to a different way of treating its inputs and outputs), which is what ModelWithHead is for.
    """

    def __init__(self, base_model: BaseModel, extended_config: BaseModelExtendedConfig):
        super().__init__(base_model.config)  # Copying the config of the core just like the BaseModel does.
        self.nested = base_model
        self.scalar_mix = snn.pretrained.ScalarMix(extended_config.layer_pooling) if extended_config.layer_pooling > 1 else lambda t: t[0]  # Has a dropout parameter, but I don't think it works properly.

        self.layer_pooling  = extended_config.layer_pooling
        self.pooling_method = extended_config.word_pooling
        self.stride         = extended_config.stride

        base_config = base_model.standardiseConfig(base_model.config)
        self.hidden_size    = base_config.hidden_size
        self.context_length = base_config.context_length
        self.pad_index      = base_model.base_model.config.pad_token_id

    @property
    def base_model(self) -> PreTrainedModel:
        return self.nested._core

    def forward(
        self,
        input_ids: Tensor,  # B x pad(L_words) x pad(L_tokens_per_word)
        attention_mask: Tensor,  # Not used because we generate our own. Still need it as argument though.
        do_drop_intermediates: bool=True,
        **kwargs
    ) -> AllHiddenStatesAndPooling:
        assert len(input_ids.shape) == 3

        attention_mask = input_ids.ne(self.pad_index)

        # First, concatenate all input_ids into a flat sequence across words. We will process these as usual and only at the end pool them per word.
        tokens_per_word = attention_mask.sum((1, 2))
        tokens     = pad(     input_ids[attention_mask].split(tokens_per_word.tolist()), self.pad_index, padding_side="right")
        token_mask = pad(attention_mask[attention_mask].split(tokens_per_word.tolist()), 0,              padding_side="right")

        # Generate embeddings of all tokens by striding across the tokens in windows of max context length with a given stride.
        # For each window, pool the token embeddings across multiple layers.
        all_hidden_states = self.nested(
            tokens[:, :self.context_length],
            attention_mask=token_mask[:, :self.context_length].float(),
            do_drop_intermediates=False
        ).hidden_states  # This is a tuple, not a tensor.
        x_new = self.scalar_mix(all_hidden_states[-self.layer_pooling:])
        x = x_new
        for i in range(self.stride, (tokens.shape[1]-self.context_length+self.stride-1)//self.stride*self.stride+1, self.stride):
            all_hidden_states = self.nested(
                tokens[:, i:i+self.context_length],
                attention_mask=token_mask[:, i:i+self.context_length].float(),
                do_drop_intermediates=False
            ).hidden_states
            x_new = self.scalar_mix(all_hidden_states[-self.layer_pooling:])[:, self.context_length-self.stride:]
            x = torch.cat( (x,x_new) , dim=1)
        # -> B x (W*S) x H

        # Not sure what's happening here.
        tokens_per_word = attention_mask.sum(-1)
        tokens_per_word = tokens_per_word.masked_fill_(tokens_per_word.eq(0), 1)
        # -> B x W

        # Set everything to zero that belongs to a pad token (pad-subwords for short words and pad-words for short sentences)
        x = x.new_zeros(*attention_mask.shape, self.hidden_size).masked_scatter_(attention_mask.unsqueeze(-1), x[token_mask])
        # -> B x W x S x H

        # So far, everything has been in terms of subwords. We now pool the subword dimension to get word embeddings.
        if self.pooling_method == 'first':
            x = x[:, :, 0]
        elif self.pooling_method == 'last':
            x = x.gather(2, (tokens_per_word-1).unsqueeze(-1).repeat(1, 1, self.hidden_size).unsqueeze(2)).squeeze(2)
        elif self.pooling_method == 'mean':
            x = x.sum(2) / tokens_per_word.unsqueeze(-1)
        elif self.pooling_method:
            raise RuntimeError(f'Unsupported pooling method "{self.pooling_method}"!')
        # -> B x W x H

        return AllHiddenStatesAndPooling(
            last_hidden_state=x,
            hidden_states=None,  # TODO: If you ever need these, you can figure something out for them.
            pooler_output=None
        )

    @classmethod
    @property
    def config_class(cls):
        return None

    @classmethod
    def standardiseConfig(cls, raw_config):
        return None

    @classmethod
    def buildCore(cls, raw_config):
        return None
