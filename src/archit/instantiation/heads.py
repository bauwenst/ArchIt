from dataclasses import dataclass, field
from typing import Tuple

import torch
from torch import nn, Tensor

from .abstracts import Head, HeadConfig, Tensors
from .basemodels import BaseModelConfig, AllHiddenStatesAndPooling
from .extensions import PoolingAndStridingConfig, flattenNestedBatches

__all__ = ["TokenClassificationHead", "TokenClassificationHeadConfig",
           "SequenceClassificationHead", "SequenceClassificationHeadConfig", "SequenceClassificationHeadForNestedBatches",
           "MaskedLMHead", "MaskedLMHeadConfig",
           "CausalLMHead", "CausalLMHeadConfig",
           "ExtractiveQAHead", "ExtractiveQAHeadConfig",
           "ExtractiveAQAHead", "ExtractiveAQAHeadConfig",
           "DependencyParsingHead", "DependencyParsingHeadConfig", "PoolingAndStridingConfig"]


@dataclass
class HeadConfigWithLabels(HeadConfig):
    """Your IDE might complain about extensions of this class, but due to the kw_only flag, it's all legal."""
    num_labels: int = field(default=0, kw_only=True)  # It has a default so that users don't have to specify this (tasks automate it).


@dataclass
class TokenClassificationHeadConfig(HeadConfigWithLabels):
    pass


class TokenClassificationHead(Head[TokenClassificationHeadConfig]):

    def __init__(self, base_config: BaseModelConfig, head_config: TokenClassificationHeadConfig):
        super().__init__(base_config, head_config)
        self.dropout = nn.Dropout(base_config.hidden_dropout_prob)
        self.dense   = nn.Linear(base_config.hidden_size, head_config.num_labels)
        self.post_init()

    def forward(
        self,
        encoder_output: AllHiddenStatesAndPooling,
        attention_mask: Tensor,
        **kwargs
    ) -> Tensor:
        x = encoder_output.last_hidden_state
        x = self.dropout(x)
        x = self.dense(x)  # B x L x H -> B x L x C
        return x

    @classmethod
    @property
    def config_class(cls):
        return TokenClassificationHeadConfig

    @classmethod
    def hfEquivalentSuffix(cls) -> str:
        return "ForTokenClassification"

    @classmethod
    def assertConfigConstraints(cls, base_config: BaseModelConfig, head_config: TokenClassificationHeadConfig):
        assert head_config.num_labels > 0


@dataclass
class SequenceClassificationHeadConfig(HeadConfigWithLabels):
    pass


class SequenceClassificationHead(Head[SequenceClassificationHeadConfig]):
    """
    Expects the model to have pooler output. Otherwise, uses mean pooling.
    """

    def __init__(self, base_config: BaseModelConfig, head_config: SequenceClassificationHeadConfig):
        super().__init__(base_config, head_config)
        self.dropout = nn.Dropout(base_config.hidden_dropout_prob)
        self.dense1  = nn.Linear(base_config.hidden_size, base_config.hidden_size)
        self.dense2  = nn.Linear(base_config.hidden_size, head_config.num_labels)

        self.pooler = MeanPooler()
        self.post_init()

    def forward(
        self,
        encoder_output: AllHiddenStatesAndPooling,
        attention_mask: Tensor,
        **kwargs
    ) -> Tensor:
        x = encoder_output.pooler_output or self.pooler(encoder_output.last_hidden_state, attention_mask)

        # Token classification head...
        x = self.dropout(x)
        x = self.dense1(x)

        # ...followed by tanh...
        x = torch.tanh(x)

        # ...followed by another token classification head.
        x = self.dropout(x)
        x = self.dense2(x)
        return x

    @classmethod
    @property
    def config_class(cls):
        return SequenceClassificationHeadConfig

    @classmethod
    def hfEquivalentSuffix(cls) -> str:
        return "ForSequenceClassification"

    @classmethod
    def assertConfigConstraints(cls, base_config: BaseModelConfig, head_config: SequenceClassificationHeadConfig):
        assert head_config.num_labels > 0


class SequenceClassificationHeadForNestedBatches(SequenceClassificationHead):

    def forward(
        self,
        encoder_output: AllHiddenStatesAndPooling,
        attention_mask: Tensor,
        **kwargs
    ) -> Tensor:
        # Flatten encoder output
        B1, B2, L, H = encoder_output.last_hidden_state.shape
        flat_encoder_output, attention_mask = flattenNestedBatches(encoder_output, attention_mask)

        # Pooled output B*C x H -> B*C x K  (possibly after B*C x L x H -> B*C x H)
        logits = super().forward(
            encoder_output=flat_encoder_output,
            attention_mask=attention_mask,
            **kwargs
        )

        # Unflatten B*C x K -> B x C x K
        return logits.view(B1, B2, -1)

    @classmethod
    def hfEquivalentSuffix(cls) -> str:
        return "/"  # ForMultipleChoice uses a linear+dropout head, not a sequence classification head.


@dataclass
class MaskedLMHeadConfig(HeadConfig):
    pass


class MaskedLMHead(Head[MaskedLMHeadConfig]):

    def __init__(self, base_config: BaseModelConfig, head_config: MaskedLMHeadConfig):
        super().__init__(base_config, head_config)
        self.dense1     = nn.Linear(base_config.hidden_size, base_config.hidden_size)
        self.layer_norm = nn.LayerNorm(base_config.hidden_size, eps=1e-5)
        self.dense2     = nn.Linear(base_config.hidden_size, base_config.vocab_size, bias=True)
        self.post_init()

    def forward(
        self,
        encoder_output: AllHiddenStatesAndPooling,
        attention_mask: Tensor,
        **kwargs
    ) -> Tensor:
        x = encoder_output.last_hidden_state
        x = self.dense1(x)
        x = nn.functional.gelu(x)
        x = self.layer_norm(x)
        x = self.dense2(x)
        return x

    @classmethod
    @property
    def config_class(cls):
        return MaskedLMHeadConfig

    @classmethod
    def hfEquivalentSuffix(cls) -> str:
        return "ForMaskedLM"

    @classmethod
    def assertConfigConstraints(cls, base_config: BaseModelConfig, head_config: MaskedLMHeadConfig):
        pass


@dataclass
class CausalLMHeadConfig(HeadConfig):
    pass


class CausalLMHead(Head[CausalLMHeadConfig]):

    def __init__(self, base_config: BaseModelConfig, head_config: CausalLMHeadConfig):
        super().__init__(base_config, head_config)
        self.dense = nn.Linear(base_config.hidden_size, base_config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        encoder_output: AllHiddenStatesAndPooling,
        attention_mask: Tensor,
        **kwargs
    ) -> Tensor:
        return self.dense(encoder_output.last_hidden_state)

    @classmethod
    @property
    def config_class(cls):
        return CausalLMHeadConfig

    @classmethod
    def hfEquivalentSuffix(cls) -> str:
        return "ForCausalLM"

    @classmethod
    def assertConfigConstraints(cls, base_config: BaseModelConfig, head_config: CausalLMHeadConfig):
        pass


@dataclass
class ExtractiveQAHeadConfig(HeadConfig):
    pass


class ExtractiveQAHead(Head[ExtractiveQAHeadConfig]):

    def __init__(self, base_config: BaseModelConfig, head_config: ExtractiveQAHeadConfig):
        super().__init__(base_config, head_config)
        self.dense = nn.Linear(base_config.hidden_size, 2)
        self.post_init()

    def forward(
        self,
        encoder_output: AllHiddenStatesAndPooling,
        attention_mask: Tensor,
        **kwargs
    ) -> Tensor:
        return self.dense(encoder_output.last_hidden_state)

    @classmethod
    @property
    def config_class(cls):
        return ExtractiveQAHeadConfig

    @classmethod
    def hfEquivalentSuffix(cls) -> str:
        return "ForQuestionAnswering"

    @classmethod
    def assertConfigConstraints(cls, base_config: BaseModelConfig, head_config: ExtractiveQAHeadConfig):
        pass


@dataclass
class ExtractiveAQAHeadConfig(HeadConfig):
   ua_loss_weight: float


class ExtractiveAQAHead(Head[ExtractiveAQAHeadConfig]):

    def __init__(self, base_config: BaseModelConfig, head_config: ExtractiveAQAHeadConfig):
        super().__init__(base_config, head_config)
        self.qa_heads = nn.Linear(base_config.hidden_size, 2)  # Two linear heads but they're stored as one.
        self.ua_head  = SequenceClassificationHead(base_config, SequenceClassificationHeadConfig(num_labels=2))
        self.post_init()

    def forward(
        self,
        encoder_output: AllHiddenStatesAndPooling,
        attention_mask: Tensor,
        **kwargs
    ) -> Tuple[Tensor,Tensor]:  # (QA logits, UA logits)
        return self.qa_heads(encoder_output.last_hidden_state), self.ua_head(encoder_output.last_hidden_state, attention_mask)

    @classmethod
    @property
    def config_class(cls):
        return ExtractiveAQAHeadConfig

    @classmethod
    def hfEquivalentSuffix(cls) -> str:
        return "/"

    @classmethod
    def assertConfigConstraints(cls, base_config: BaseModelConfig, head_config: ExtractiveAQAHeadConfig):
        pass


@dataclass
class DependencyParsingHeadConfig(HeadConfigWithLabels):  # If your IDE complains about this line, update your IDE :)
    extended_model_config: PoolingAndStridingConfig

    final_hidden_size_arcs: int=500
    final_hidden_size_relations: int=100
    head_dropout: float=0.33  # Dropout after the model and dropout inside the head.
    standardisation_exponent: int=0  # The scores of the biaffine part of the head can be scaled by 1/hidden_size^e with e some exponent. For some reason it must be an int.


class DependencyParsingHead(Head[DependencyParsingHeadConfig]):

    def __init__(self, base_config: BaseModelConfig, head_config: DependencyParsingHeadConfig):
        super().__init__(base_config, head_config)
        self.dropout = nn.Dropout(base_config.hidden_dropout_prob)

        from supar import modules as snn
        self.arc_mlp_d = snn.MLP(n_in=base_config.hidden_size, n_out=head_config.final_hidden_size_arcs,      dropout=head_config.head_dropout)
        self.arc_mlp_h = snn.MLP(n_in=base_config.hidden_size, n_out=head_config.final_hidden_size_arcs,      dropout=head_config.head_dropout)
        self.rel_mlp_d = snn.MLP(n_in=base_config.hidden_size, n_out=head_config.final_hidden_size_relations, dropout=head_config.head_dropout)
        self.rel_mlp_h = snn.MLP(n_in=base_config.hidden_size, n_out=head_config.final_hidden_size_relations, dropout=head_config.head_dropout)

        self.arc_attn = snn.Biaffine(n_in=head_config.final_hidden_size_arcs, scale=head_config.standardisation_exponent, bias_x=True, bias_y=False)
        self.rel_attn = snn.Biaffine(n_in=head_config.final_hidden_size_relations, n_out=head_config.num_labels, bias_x=True, bias_y=True)
        self.post_init()

    def forward(
        self,
        encoder_output: AllHiddenStatesAndPooling,
        attention_mask: Tensor,  # This is assumed to be at the word level, not token level.
        **kwargs
    ) -> Tuple[Tensor,Tensor]:
        x = encoder_output.last_hidden_state  # These are also assumed to be embeddings at the word level.
        x = self.dropout(x)

        arc_d = self.arc_mlp_d(x)
        arc_h = self.arc_mlp_h(x)
        rel_d = self.rel_mlp_d(x)
        rel_h = self.rel_mlp_h(x)

        s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~attention_mask.ge(0).unsqueeze(1), -1e32)  # batch_size x seq_len x seq_len
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)                                      # batch_size x seq_len x seq_len x n_rels
        return s_arc, s_rel

    @classmethod
    @property
    def config_class(cls):
        return DependencyParsingHeadConfig

    @classmethod
    def assertConfigConstraints(cls, base_config: BaseModelConfig, head_config: DependencyParsingHeadConfig):
        assert head_config.num_labels > 0


class MeanPooler:
    """
    Takes a batch of B examples with L tokens in the longest example and hidden size H, and per example and per dimension
    takes the mean across all valid tokens, mapping a B x L x H tensor to a B x H tensor.

    Due to the presence of invalid tokens, this is more difficult than just torch.mean(embeddings, dim=1).
    Implementation simplified from a switch statement in SentenceTransformers: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py
    """

    def __call__(
        self,
        embeddings: Tensor,  # B x L x H
        attention_mask: Tensor  # B x L
    ) -> Tensor:
        # Make the mask B x L x H too.
        attention_mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).to(embeddings.dtype)

        # Mask the embeddings so that only the relevant ones are non-zero, then sum.
        sum_embeddings = torch.sum(embeddings * attention_mask, dim=1)  # B x H

        # Get denominators. For examples with length 0 (which will have a sum of 0 anyway), don't divide by 0.
        sum_mask = attention_mask.sum(dim=1)  # B x H
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # Could've been as much as 0.999; the point is "all integers under this are not allowed".
        return sum_embeddings / sum_mask
