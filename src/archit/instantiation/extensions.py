"""
Partial implementations of ModelWithHead as a step-up from BaseModel towards ModelWithHead.
These are NOT subclasses of BaseModel because they are oblivious to which HuggingFace model is wrapped.
"""
from dataclasses import dataclass
import torch
from supar import modules as snn  # Note: You must install supar from GitHub. The pip version is more than 2 years out of date! https://github.com/yzhangcs/parser
from supar.utils.fn import pad
from enum import Enum

from .configs import RecursiveSerialisable, CombinedConfig, PC, HC
from .abstracts import BaseModel, Head, ModelWithHead, _Loss, Tensor, AllHiddenStatesAndPooling


class SubwordPooling(str, Enum):  # https://aclanthology.org/2021.eacl-main.194.pdf
    FIRST = 1
    LAST  = 2
    MEAN  = 3


@dataclass
class BaseModelExtendedConfig(RecursiveSerialisable):
    layer_pooling: int=1      # Embedding pooling per token across the last N layers (weighted sum with learnt softmax weights).
    word_pooling: SubwordPooling = SubwordPooling.MEAN  # Embedding pooling per word across its tokens.
    stride: int=256           # Tokens to shift the context window when the input is too long.


class BaseModelExtended(ModelWithHead[PC,HC]):
    """
    Wrapper around BaseModel that offers three advanced features for generating embeddings:
        - Layer pooling: the token embeddings that are then within-word-pooled.
        - Word-level pooling: whereas input_ids is normally B x pad(L_tokens), it is assumed to now be B x pad(L_words) x pad(L_tokens_per_word),
          i.e. there is an extra dimension that separates IDs by the word they belong to. This allows custom within-word embedding pooling.
        - Striding: the BaseModel has a finite context length to produce logits, but the head that processes these logits
          doesn't. Hence, rather than having your tokeniser truncate the input, you can stride across an arbitrarily long input in
          overlapping windows and concatenate the resulting embeddings for tokens that don't yet have one.

    This was originally a subclass of BaseModel WHILE WRAPPING an object of type BaseModel. This was the wrong
    approach since all except one of BaseModel's methods is an abstract CLASS method. That is: you should be able to use
    a subclass of BaseModel without ever instantiating it. You can't do that with a wrapper around a BaseModel instance.
    Since ModelWithHead is for modelling how to use a base model (i.e. adapting from an already universal BaseModel interface
    to a different way of treating its inputs and outputs), it is now a subclass of that instead.
    """

    def __init__(self, combined_config: CombinedConfig[PC,HC], extended_config: BaseModelExtendedConfig, model: BaseModel[PC], head: Head[HC], loss: _Loss):
        super().__init__(combined_config=combined_config, model=model, head=head, loss=loss)
        self.scalar_mix = snn.pretrained.ScalarMix(extended_config.layer_pooling) if extended_config.layer_pooling > 1 else lambda t: t[0]  # Has a dropout parameter, but I don't think it works properly.

        # Settings to save.
        self._layer_pooling  = extended_config.layer_pooling
        self._pooling_method = extended_config.word_pooling
        self._stride         = extended_config.stride

        base_config = model.__class__.standardiseConfig(model.config)
        self._hidden_size    = base_config.hidden_size
        self._context_length = base_config.context_length
        self._pad_index      = model.config.pad_token_id
        if self._pad_index is None:
            raise ValueError("Failed to find pad_token_id in the base model's config.")

    def callBaseModel(
        self,
        input_ids: Tensor,  # B x pad(L_words) x pad(L_tokens_per_word)
        attention_mask: Tensor,  # Not used because we generate our own. Still need it as argument though.
        do_drop_intermediates: bool,
        **kwargs
    ) -> AllHiddenStatesAndPooling:
        assert len(input_ids.shape) == 3

        attention_mask = input_ids.ne(self._pad_index)

        # First, concatenate all input_ids into a flat sequence across words. We will process these as usual and only at the end pool them per word.
        tokens_per_word = attention_mask.sum((1, 2))
        tokens     = pad(input_ids[attention_mask].split(tokens_per_word.tolist()), self._pad_index, padding_side="right")
        token_mask = pad(attention_mask[attention_mask].split(tokens_per_word.tolist()), 0,              padding_side="right")

        # Generate embeddings of all tokens by striding across the tokens in windows of max context length with a given stride.
        # For each window, pool the token embeddings across multiple layers.
        all_hidden_states = self.model(
            tokens[:, :self._context_length],
            attention_mask=token_mask[:, :self._context_length].float(),
            do_drop_intermediates=False
        ).hidden_states  # This is a tuple, not a tensor.
        x_new = self.scalar_mix(all_hidden_states[-self._layer_pooling:])
        x = x_new
        for i in range(self._stride, (tokens.shape[1] - self._context_length + self._stride - 1) // self._stride * self._stride + 1, self._stride):
            all_hidden_states = self.model(
                tokens[:, i:i+self._context_length],
                attention_mask=token_mask[:, i:i+self._context_length].float(),
                do_drop_intermediates=False
            ).hidden_states
            x_new = self.scalar_mix(all_hidden_states[-self._layer_pooling:])[:, self._context_length - self._stride:]
            x = torch.cat( (x,x_new) , dim=1)
        # -> B x (W*S) x H

        # Not sure what's happening here.
        tokens_per_word = attention_mask.sum(-1)
        tokens_per_word = tokens_per_word.masked_fill_(tokens_per_word.eq(0), 1)
        # -> B x W

        # Set everything to zero that belongs to a pad token (pad-subwords for short words and pad-words for short sentences)
        x = x.new_zeros(*attention_mask.shape, self._hidden_size).masked_scatter_(attention_mask.unsqueeze(-1), x[token_mask])
        # -> B x W x S x H

        # So far, everything has been in terms of subwords. We now pool the subword dimension to get word embeddings.
        if self._pooling_method == SubwordPooling.FIRST:
            x = x[:, :, 0]
        elif self._pooling_method == SubwordPooling.LAST:
            x = x.gather(2, (tokens_per_word-1).unsqueeze(-1).repeat(1, 1, self._hidden_size).unsqueeze(2)).squeeze(2)
        elif self._pooling_method == SubwordPooling.MEAN:
            x = x.sum(2) / tokens_per_word.unsqueeze(-1)
        elif self._pooling_method:
            raise RuntimeError(f'Unsupported pooling method "{self._pooling_method}"!')
        # -> B x W x H

        return AllHiddenStatesAndPooling(
            last_hidden_state=x,
            hidden_states=None,  # TODO: If you ever need these, you can figure something out for them.
            pooler_output=None
        )

    # @property
    # def _supports_sdpa(self) -> bool:
    #     """
    #     In newer versions of Transformers, model code explicitly supports multiple implementations for attention. One is
    #     "eager" attention built with several calls to PyTorch in Python, the other uses PyTorch's new native "sdpa".
    #     The implementation is chosen with the config field "attn_implementation". If it is None (which is the case
    #     in all model configs pre-September 2024, and for newer models there is no requirement to set it so it defaults
    #     to None), then when you call model_class.from_pretrained(), the method _autoset_attn_implementation() checks
    #     whether the _supports_sdpa field is True in the model class. If yes, and PyTorch has SDPA, then "sdpa" is set
    #     in the config.
    #
    #     By default, _supports_sdpa is always False. In ArchIt, when the BaseModel's core has _supports_sdpa True (like
    #     with a RoBERTa core), it is not actually an issue because the config is FIRST used to instantiate the BaseModel
    #     (attn_implementation None + _supports_sdpa False => no conflict and no change) and THEN the core
    #     (attn_implementation None + _supports_sdpa True => no conflict and attn_implementation changed to "sdpa"). However,
    #     if the config is used for a third time to construct an ArchIt model (e.g. a BaseModelExtended which copies the
    #     BaseModel's config), then you have attn_implementation "sdpa" + _supports_sdpa False => conflict.
    #
    #     This can be solved by having BaseModelExtended affirm that it _supports_sdpa when its core does.
    #     """
    #     return self.base_model._supports_sdpa