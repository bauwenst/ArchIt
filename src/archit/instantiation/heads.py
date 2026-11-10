"""
Defines head signatures and head architectures that satisfy them.

Note: our design policy is that if your task needs an existing head architecture but with a fixed output dimension, you
      do not create a new architecture and config for it, but rather let the task adjust the config's n_classes field.
      This prevents a needless explosion of architectures and combined architectures.
"""
from dataclasses import dataclass, field
from abc import abstractmethod

import torch
from torch import nn, Tensor

from .abstracts import Head, Tensors
from .configs import *
from .basemodels import BaseModelConfig, AllHiddenStatesAndPooling
from .extensions import PoolingAndStridingConfig, flattenNestedBatches
from .mixins import no_weight_init

__all__ = [
    # Abstract contracts
    "TokenClassifierHeadConfig",
    "PooledClassifierHeadConfig",
    "NestedPooledClassifierHeadConfig",
    "TokenAndPooledClassifierHeadConfig",
    "TokenVocabularyHeadConfig",
    "TokenToTokenHeadConfig",
    "TokenToTokenClassifierHeadConfig",
    "TokenToTokenDualHeadsConfig",

    # Concrete implementations
    "SingleLayerHead",
    "SingleLayerHeadConfig",
    "SingleLayerLMHead",
    "SingleLayerLMHeadConfig",
    "TwoLayerLMHead",
    "TwoLayerLMHeadConfig",
    "TwoLayerPoolingHead",
    "TwoLayerPoolingHeadConfig",
    "TwoLayerPoolingHeadForNestedBatches",
    "OneTokenHeadOnePooledHead",
    "OneTokenHeadOnePooledHeadConfig",
    "BiaffineTokenToTokenHead",
    "BiaffineTokenToTokenHeadConfig",
    "PoolingAndStridingConfig",
]


@dataclass
class SingleLayerHeadConfig(TokenClassifierHeadConfig):
    @classmethod
    def headClass(cls) -> type[Head]:
        return SingleLayerHead


class SingleLayerHead(Head):

    def __init__(self, base_config: BaseModelConfig, head_config: SingleLayerHeadConfig):
        super().__init__(base_config, head_config)
        self.dropout = nn.Dropout(base_config.hidden_dropout_prob)
        self.dense   = nn.Linear(base_config.hidden_size, head_config.n_classes)
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
    def hfEquivalentSuffix(cls) -> str:
        return "ForTokenClassification"  # Without dropout, also ForQuestionAnswering.


@dataclass
class TwoLayerPoolingHeadConfig(PooledClassifierHeadConfig):
    @classmethod
    def headClass(cls) -> type[Head]:
        return TwoLayerPoolingHead


class TwoLayerPoolingHead(Head):
    """
    Expects the model to have pooler output. Otherwise, uses mean pooling.
    """

    def __init__(self, base_config: BaseModelConfig, head_config: TwoLayerPoolingHeadConfig):
        super().__init__(base_config, head_config)
        self.dropout = nn.Dropout(base_config.hidden_dropout_prob)
        self.dense1  = nn.Linear(base_config.hidden_size, base_config.hidden_size)
        self.dense2  = nn.Linear(base_config.hidden_size, head_config.n_classes)

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
    def hfEquivalentSuffix(cls) -> str:
        return "ForSequenceClassification"


@dataclass
class NestedPooledClassifierHeadConfig(PooledClassifierHeadConfig):
    """
    For tasks where examples sent to the model belong together somehow, e.g. (question, option1), (question, option2), ...
    where each pair is scored with a scalar for how high it is predicted to be along each of K criteria.
    """
    @classmethod
    def getInputSignature(cls) -> tuple:
        return ("N1", "N2", "L", "H")

    @classmethod
    def getOutputSignature(cls) -> tuple:
        return ("N1", "N2", "K")


@dataclass
class TwoLayerPoolingHeadForNestedBatchesConfig(NestedPooledClassifierHeadConfig):
    @classmethod
    def headClass(cls) -> type[Head]:
        return TwoLayerPoolingHeadForNestedBatches


class TwoLayerPoolingHeadForNestedBatches(TwoLayerPoolingHead):

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
        return ""  # ForMultipleChoice uses a single-layer head, not a two-layer head.


class _LMHead(Head):

    @abstractmethod
    def get_embeddings(self) -> nn.Linear:
        """
        Returns a reference to the module whose .weight should be reassigned when weight tying is done.
        """
        pass

    @abstractmethod
    def set_embeddings(self, new_embeddings: nn.Linear):
        """
        Used when there is no weight tying, but you resize the embedding matrix.
        modeling_utils will make sure you receive a nn.Linear rather than a Tensor or an nn.Embedding.
        """
        pass


@dataclass
class TwoLayerLMHeadConfig(TokenVocabularyHeadConfig):
    @classmethod
    def headClass(cls) -> type[Head]:
        return TwoLayerLMHead


class TwoLayerLMHead(_LMHead):
    """
    Based on RoBERTa's LM head.

    TODO: ModernBERT's standard deviation at initialisation for this head is
        0.02 / math.sqrt(2 * self.config.num_hidden_layers)
     which is much smaller than the usual 0.02. So, if weight tying is not used, consider using those initial values instead.

    Note: weight tying is enabled in the class that contains the (head, base model) pair, not the head.
          Hence, no method implementations are dedicated to it here.
    """

    def __init__(self, base_config: BaseModelConfig, head_config: TwoLayerLMHeadConfig):
        super().__init__(base_config, head_config)
        self.dense1     = nn.Linear(base_config.hidden_size, base_config.hidden_size)
        self.layer_norm = nn.LayerNorm(base_config.hidden_size, eps=1e-5)
        self.dense2     = nn.Linear(base_config.hidden_size, base_config.vocab_size, bias=True)  # Note about weight tying: weight tying is only applied to the .weight field of this Linear. I don't really see why you would need a separate .bias field in this head, despite BERT and RoBERTa and DeBERTa having them. (DeBERTa sets bias=False, then makes a .bias and then sets the Linear's bias to that...)
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
    def hfEquivalentSuffix(cls) -> str:
        return "ForMaskedLM"

    def get_embeddings(self) -> nn.Linear:
        return self.dense2

    def set_embeddings(self, new_embeddings: nn.Linear):
        self.dense2 = new_embeddings


@dataclass
class SingleLayerLMHeadConfig(TokenVocabularyHeadConfig):
    @classmethod
    def headClass(cls) -> type[Head]:
        return SingleLayerLMHead


class SingleLayerLMHead(_LMHead):
    """
    Based on GPT's LM head.
    """

    def __init__(self, base_config: BaseModelConfig, head_config: SingleLayerLMHeadConfig):
        super().__init__(base_config, head_config)
        self.dense = nn.Linear(base_config.hidden_size, base_config.vocab_size, bias=False)  # GPT, LLaMA, Gemma ... all lack a bias prior, unlike most MLMs. Don't know why.
        self.post_init()

    def forward(
        self,
        encoder_output: AllHiddenStatesAndPooling,
        attention_mask: Tensor,
        **kwargs
    ) -> Tensor:
        return self.dense(encoder_output.last_hidden_state)

    @classmethod
    def hfEquivalentSuffix(cls) -> str:
        return "ForCausalLM"

    def get_embeddings(self) -> nn.Linear:
        return self.dense

    def set_embeddings(self, new_embeddings: nn.Linear):
        self.dense = new_embeddings


@dataclass
class TokenAndPooledClassifierHeadConfig(HeadConfig):  # Can be implemented as two separately configure heads but also as one very specific very big head, as is the case for BiaffineTokenToTokenHeadConfig extending TokenToTokenDualHeadsConfig.
    pooled_loss_weight: float

    @classmethod
    def getInputSignature(cls) -> tuple:
        return (TokenClassifierHeadConfig.getInputSignature(), PooledClassifierHeadConfig.getInputSignature())  # We steal the signatures despite these classes possibly not being used to build the head.

    @classmethod
    def getOutputSignature(cls) -> tuple:
        return (TokenClassifierHeadConfig.getOutputSignature(), PooledClassifierHeadConfig.getOutputSignature())


@dataclass
class OneTokenHeadOnePooledHeadConfig(TokenAndPooledClassifierHeadConfig):
    token_head: TokenClassifierHeadConfig
    pooled_head: PooledClassifierHeadConfig

    @classmethod
    def headClass(cls) -> type[Head]:
        return OneTokenHeadOnePooledHead


class OneTokenHeadOnePooledHead(Head):

    def __init__(self, base_config: BaseModelConfig, head_config: OneTokenHeadOnePooledHeadConfig):
        super().__init__(base_config, head_config)
        with no_weight_init():
            self.first  = head_config.token_head.buildHead(base_config)
            self.second = head_config.pooled_head.buildHead(base_config)
            self.qa_heads = nn.Linear(base_config.hidden_size, 2)  # Two linear heads but they're stored as one.
            self.ua_head  = TwoLayerPoolingHead(base_config, TwoLayerPoolingHeadConfig(n_classes=2))  # Binary classifier.
        self.post_init()

    def forward(
        self,
        encoder_output: AllHiddenStatesAndPooling,
        attention_mask: Tensor,
        **kwargs
    ) -> tuple[Tensor,Tensor]:  # (QA logits, UA logits)
        return self.qa_heads(encoder_output.last_hidden_state), self.ua_head(encoder_output.last_hidden_state, attention_mask)


@dataclass
class TokenToTokenHeadConfig(HeadConfig):
    def getInputSignature(cls) -> tuple:
        return ("N", "L", "H")

    def getOutputSignature(cls) -> tuple:
        return ("N", "L", "L")


@dataclass
class TokenToTokenClassifierHeadConfig(HeadConfigWithLabels):
    def getInputSignature(cls) -> tuple:
        return ("N", "L", "H")

    def getOutputSignature(cls) -> tuple:
        return ("N", "L", "L", "C")


@dataclass
class TokenToTokenDualHeadsConfig(HeadConfigWithLabels):
    def getInputSignature(cls) -> tuple:
        return ("N", "L", "H")

    def getOutputSignature(cls) -> tuple:
        return (("N", "L", "L"), ("N", "L", "L", "C"))


@dataclass
class BiaffineTokenToTokenHeadConfig(TokenToTokenDualHeadsConfig):
    extended_model_config: PoolingAndStridingConfig=field(default_factory=PoolingAndStridingConfig)

    final_hidden_size_arcs: int=500
    final_hidden_size_relations: int=100
    head_dropout: float=0.33  # Dropout after the model and dropout inside the head.
    standardisation_exponent: int=0  # The scores of the biaffine part of the head can be scaled by 1/hidden_size^e with e some exponent. For some reason it must be an int.

    @classmethod
    def headClass(cls) -> type[Head]:
        return BiaffineTokenToTokenHead


class BiaffineTokenToTokenHead(Head):
    """
    Based on supar's Dozat & Manning 2017 implementation.
    https://arxiv.org/abs/1611.01734
    """

    def __init__(self, base_config: BaseModelConfig, head_config: BiaffineTokenToTokenHeadConfig):
        super().__init__(base_config, head_config)
        self.dropout = nn.Dropout(base_config.hidden_dropout_prob)

        from supar import __version__, modules as snn
        if tuple(map(int, __version__.split("."))) < (1,2,0):
            raise ImportError("You have the wrong distribution of SuPar. DO NOT use 'pip install supar'. Use 'pip install supar_bauwenst'.")

        self.arc_mlp_d = snn.MLP(n_in=base_config.hidden_size, n_out=head_config.final_hidden_size_arcs,      dropout=head_config.head_dropout)
        self.arc_mlp_h = snn.MLP(n_in=base_config.hidden_size, n_out=head_config.final_hidden_size_arcs,      dropout=head_config.head_dropout)
        self.rel_mlp_d = snn.MLP(n_in=base_config.hidden_size, n_out=head_config.final_hidden_size_relations, dropout=head_config.head_dropout)
        self.rel_mlp_h = snn.MLP(n_in=base_config.hidden_size, n_out=head_config.final_hidden_size_relations, dropout=head_config.head_dropout)

        self.arc_attn = snn.Biaffine(n_in=head_config.final_hidden_size_arcs, scale=head_config.standardisation_exponent, bias_x=True, bias_y=False)
        self.rel_attn = snn.Biaffine(n_in=head_config.final_hidden_size_relations, n_out=head_config.n_classes, bias_x=True, bias_y=True)
        self.post_init()

    def forward(
        self,
        encoder_output: AllHiddenStatesAndPooling,
        attention_mask: Tensor,  # This is assumed to be at the word level, not token level.
        **kwargs
    ) -> tuple[Tensor,Tensor]:
        x = encoder_output.last_hidden_state  # These are also assumed to be embeddings at the word level.
        x = self.dropout(x)

        arc_d = self.arc_mlp_d(x)
        arc_h = self.arc_mlp_h(x)
        rel_d = self.rel_mlp_d(x)
        rel_h = self.rel_mlp_h(x)

        s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~attention_mask.ge(0).unsqueeze(1), -1e32)  # batch_size x seq_len x seq_len
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)                                      # batch_size x seq_len x seq_len x n_rels
        return s_arc, s_rel


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
