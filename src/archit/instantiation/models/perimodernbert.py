"""
A version of BERT that uses peri-norm convention rather than post-norm.
https://arxiv.org/abs/2502.02732
"""
from typing import Optional, Literal

import torch
from torch import nn

from transformers.models.modernbert.modeling_modernbert import ModernBertModel, ModernBertConfig, ModernBertAttention, \
    ModernBertMLP, ModernBertPreTrainedModel


NormType = Literal["ln", "rms"]

class PeriModernBertConfig(ModernBertConfig):

    def __init__(self,
        # Model dimensions
        vocab_size=50368,
        max_position_embeddings=8192,
        hidden_size=768,
        intermediate_size=1152,
        num_hidden_layers=22,
        num_attention_heads=12,

        # Other architectural hyperparameters
        hidden_activation="gelu",
        norm_type: NormType="rms",
        global_attn_every_n_layers=3,
        local_attention=128,

        # Init
        initializer_range=0.02,
        initializer_cutoff_factor=2.0,

        # Dropouts
        embedding_dropout=0.0,
        mlp_dropout=0.0,
        softmax_dropout=0.0,  # For the softmax inside attention, which is a much more impactful dropout than the one on the output of attention. ModernBERT uses the attention_dropout config field for both of these. We disentangle them and let attention_dropout refer to ONLY the output and NOT the softmax.
        attention_dropout=0.0,
        classifier_dropout=0.0,

        # Biases
        norm_bias=False,  # Only relevant with LayerNorm.
        attention_bias=False,
        mlp_bias=False,
        decoder_bias=True,
        classifier_bias=False,

        # Pooler
        classifier_pooling: Literal["cls", "mean"] = "mean",
        classifier_activation="gelu",

        # Miscellaneous
        norm_eps=1e-5,
        global_rope_theta=160000.0,
        local_rope_theta=10000.0,
        deterministic_flash_attn=False,
        sparse_prediction=False,
        sparse_pred_ignore_index=-100,
        reference_compile=None,
        repad_logits_with_grad=False,

        # Specials
        pad_token_id=50283,
        eos_token_id=50282,
        bos_token_id=50281,
        cls_token_id=50281,
        sep_token_id=50282,

        # Normally a kwarg, but we add it explicitly to the interface.
        attn_implementation: Literal["sdpa", "flash_attention_2"]="sdpa",
        tie_word_embeddings: bool=True,

        **kwargs
    ):
        constructor_arguments = locals()  # locals() is shorthand for "dictionary of every variable that is not accessible right outside this function".
        constructor_arguments.pop("self")    # duhhh
        constructor_arguments.pop("kwargs")  # If you don't pop this, the super constructor will have a kwargs variable with value {"kwargs": dict(...)} whereas you want the kwargs variable to become that nested dictionary itself.
        super().__init__(**constructor_arguments, **kwargs)  # Note that these two do not correspond to what the super().__init__ sees as kwargs. You simply take the union of these two and anything unrecognised by super().__init__ becomes a kwarg there.

        self.softmax_dropout = softmax_dropout
        self.norm_type = norm_type


class ConserveDataType(nn.Module):
    """
    Wrapper around a PyTorch module that casts its output back to the input's data type.
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        input_dtype = x.dtype
        return self.module(x, *args, **kwargs).to(input_dtype)


class CastTo(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self._dtype = dtype

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return input.to(self._dtype)


def Norm(norm_type: NormType, hidden_size: int, eps: float, bias: bool=True) -> nn.Module:
    """
    Chooses between LayerNorm and RMSNorm at construction time.
    """
    if norm_type == "rms":
        return ConserveDataType(nn.RMSNorm(hidden_size, eps=eps, dtype=torch.float32))  # Very important to keep these weights fp32, otherwise in bf16 the gradients will be too small to kick them out of their quantisation pit.
    elif norm_type == "ln":
        return ConserveDataType(nn.LayerNorm(hidden_size, eps=eps, bias=bias, dtype=torch.float32))
    else:
        raise NotImplementedError(f"Norm type {norm_type} unknown.")


def PeriModernBertNorm(config: ModernBertConfig) -> nn.Module:
    return Norm(config.norm_type, config.hidden_size, config.norm_eps, config.norm_bias)


class EmbeddingDropout(nn.Module):
    """
    nn.Embedding wrapped in a Norm and Dropout.
    Very similar to ModernBertEmbeddings, but instantiable regardless of the kind of config you have.
    """

    def __init__(self, vocab_size: int, hidden_size: int, padding_idx: int, dropout: float,
                 norm_type: NormType="rms", norm_eps: float=1e-5, norm_bias: bool=True):
        super().__init__()
        self.matrix  = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=padding_idx)
        self.norm    = Norm(norm_type, hidden_size, norm_eps, norm_bias)
        self.dropout = nn.Dropout(dropout)

    # The reason we don't allow this module to be returned as if it is an nn.Embedding (i.e. the thing that has a .weight to be used as an LM head), is because set_input_embeddings() is usually implemented to set the same field to an nn.Embedding, and so you would GET an EmbeddingDropout but then SET it to an Embedding. Bad deal.
    # @property
    # def weight(self):
    #     return self.matrix.weight

    def forward(self, input_ids: torch.Tensor, **kwargs):
        return self.dropout(self.norm(self.matrix(input_ids)))


class PeriModernBertAttention(ModernBertAttention):
    """
    The core operator of a transformer block (attention/MLP) should not do dropout itself.
    ModernBERT's MLP respects this. Its attention does not. Hence we redefine its dropout away.
    """

    def __init__(self, config: PeriModernBertConfig, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        self.out_drop = nn.Identity()
        self.attention_dropout = config.softmax_dropout  # HF's software design practices are super shady, but basically the SDPA implementation accesses this field on this nn.Module to get the softmax dropout probability. In ModernBertAttention, the config field used for this is the SAME FIELD as is used for the attention block's output dropout, which makes no sense since they have wildly different effects.


class PeriModernBertEncoderLayer(nn.Module):
    """
    Encoder for ModernBERT but with peri-norming (and associated dropout) before the residual.
    """

    def __init__(self, config: PeriModernBertConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.attention_type = config.layer_types[layer_idx]

        self.pre_norm_att  = PeriModernBertNorm(config)
        self.attn          = PeriModernBertAttention(config, layer_idx=layer_idx)
        self.post_norm_att = PeriModernBertNorm(config)
        self.dropout_att   = nn.Dropout(config.attention_dropout)

        self.pre_norm_mlp  = PeriModernBertNorm(config)
        self.mlp           = ModernBertMLP(config)
        self.post_norm_mlp = PeriModernBertNorm(config)
        self.dropout_mlp   = nn.Dropout(config.mlp_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        output_attentions: Optional[bool] = False,
    ) -> torch.Tensor:
        # y = x + Dropout(Norm(Att(Norm(x))))
        attention_output_and_weights = self.attn(self.pre_norm_att(hidden_states),
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_attentions=output_attentions,
        )
        block_output, extra_output = attention_output_and_weights[0], attention_output_and_weights[1:]
        hidden_states = hidden_states + self.dropout_att(self.post_norm_att(block_output))

        # y = x + Dropout(Norm(MLP(Norm(x))))
        if not self.config.reference_compile:
            block_output = self.mlp(self.pre_norm_mlp(hidden_states))
            hidden_states = hidden_states + self.dropout_mlp(self.post_norm_mlp(block_output))
        else:
            hidden_states = hidden_states + self.dropout_mlp(self.compiled_mlp(hidden_states))

        return (hidden_states,) + extra_output

    @torch.compile(dynamic=True)
    def compiled_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.post_norm_mlp(self.mlp(self.pre_norm_mlp(hidden_states)))


class PeriModernBertPretrainedModel(ModernBertPreTrainedModel):
    config_class: PeriModernBertConfig

    supports_gradient_checkpointing = True
    _supports_sdpa         = True
    _supports_flash_attn   = True
    _supports_flash_attn_2 = True  # For some reason there are versions of transformers where the models set _supports_flash_attn whilst the utils check for _supports_flash_attn_2 ...

    def _init_weights(self, module: nn.Module):
        ModernBertPreTrainedModel._init_weights(self, module)  # Takes care of the attention and MLP matrices. The norms take care of themselves. Note: not super() because we work with multiple inheritance and super() is unsafe there.

        # For embeddings that either were or weren't initialised in the above super call, we define our own initialisation.
        if isinstance(module, nn.Embedding):
            self._init_weights_normdist(module, self.config.initializer_range)

    def _init_weights_normdist(self, module: nn.Module, std: float):
        """Copy of the local function defined by ModernBertPreTrainedModel."""
        cutoff_factor = self.config.initializer_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=+cutoff_factor * std,
        )
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)


class PeriModernBertModel(PeriModernBertPretrainedModel, ModernBertModel):  # Inherit the forward() but re-do some of the fields.

    def __init__(self, config: PeriModernBertConfig):
        PeriModernBertPretrainedModel.__init__(self, config)
        ModernBertModel.__init__(self, config)  # This sets some modules and initialises them, but we will overwrite them and just re-initialise (random is random).
        self.embeddings = EmbeddingDropout(config.vocab_size, config.hidden_size, config.pad_token_id, config.embedding_dropout, config.norm_type, config.norm_eps, config.norm_bias)
        self.layers = nn.ModuleList(
            [PeriModernBertEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_norm = PeriModernBertNorm(config)
        self.post_init()

    # def get_input_embeddings(self):  # Returns the thing that has a .weight property that can be used as an LM head.
    #     return self.embeddings.matrix

    # def set_input_embeddings(self, value: nn.Embedding):
    #     self.embeddings.matrix = value
