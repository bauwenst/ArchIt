from dataclasses import dataclass
from torch import nn, Tensor
import torch

from .abstracts import Head, HeadConfig, BaseModelConfig, AllHiddenStatesAndPooling


@dataclass
class TokenClassificationHeadConfig(HeadConfig):
    num_labels: int


class TokenClassificationHead(Head[TokenClassificationHeadConfig]):

    def __init__(self, base_config: BaseModelConfig, head_config: TokenClassificationHeadConfig):
        super().__init__()
        self.dropout = nn.Dropout(base_config.hidden_dropout_prob)
        self.dense   = nn.Linear(base_config.hidden_size, head_config.num_labels)

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


@dataclass
class SequenceClassificationHeadConfig(HeadConfig):
    num_labels: int


class SequenceClassificationHead(Head[SequenceClassificationHeadConfig]):
    """
    Expects the model to have pooler output. Otherwise, uses mean pooling.
    """

    def __init__(self, base_config: BaseModelConfig, head_config: SequenceClassificationHeadConfig):
        super().__init__()
        self.dropout = nn.Dropout(base_config.hidden_dropout_prob)
        self.dense1  = nn.Linear(base_config.hidden_size, base_config.hidden_size)
        self.dense2  = nn.Linear(base_config.hidden_size, head_config.num_labels)

        self.pooler = MeanPooler()

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
        return "ForTokenClassification"


@dataclass
class MaskedLMHeadConfig(HeadConfig):
    pass


class MaskedLMHead(Head[MaskedLMHeadConfig]):

    def __init__(self, base_config: BaseModelConfig, head_config: MaskedLMHeadConfig):
        super().__init__()
        self.dense1     = nn.Linear(base_config.hidden_size, base_config.hidden_size)
        self.layer_norm = nn.LayerNorm(base_config.hidden_size, eps=1e-5)
        self.dense2     = nn.Linear(base_config.hidden_size, base_config.vocab_size, bias=True)

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


@dataclass
class CausalLMHeadConfig(HeadConfig):
    pass


class CausalLMHead(Head[CausalLMHeadConfig]):

    def __init__(self, base_config: BaseModelConfig, head_config: CausalLMHeadConfig):
        super().__init__()
        self.dense = nn.Linear(base_config.hidden_size, base_config.vocab_size, bias=False)

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


@dataclass
class ExtractiveQAHeadConfig(HeadConfig):
    pass


class ExtractiveQAHead(Head[ExtractiveQAHeadConfig]):

    def __init__(self, base_config: BaseModelConfig, head_config: CausalLMHeadConfig):
        super().__init__()
        self.dense = nn.Linear(base_config.hidden_size, 2)

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
