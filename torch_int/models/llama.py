import torch
from torch import nn
from torch_int.utils.decorator import exec_time
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    BaseModelOutputWithPast,
    LlamaRMSNorm
)
from typing import Optional, Tuple, List
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
from torch_int.nn.fused import LayerNormQ
from transformers.utils import logging
from transformers.activations import ACT2FN
from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T
logger = logging.get_logger(__name__)


class Int8LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self, config: LlamaConfig
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.attention_weight_scale = 1.0

        self.qk_bmm = BMM_S8T_S8N_F32T(1.0)
        self.pv_bmm = BMM_S8T_S8N_S8T(1.0)

        self.k_proj = W8A8B8O8Linear(self.embed_dim, self.embed_dim)
        self.v_proj = W8A8B8O8Linear(self.embed_dim, self.embed_dim)
        self.q_proj = W8A8B8O8Linear(self.embed_dim, self.embed_dim)
        self.out_proj = W8A8BFP32OFP32Linear(self.embed_dim, self.embed_dim)

    @staticmethod
    @torch.no_grad()
    def from_float(module: LlamaAttention,
                   input_scale: float,
                   q_output_scale: float,
                   k_output_scale: float,
                   v_output_scale: float,
                   out_input_scale: float):
        int8_module = Int8LlamaAttention(module.config)
        # Fuse the scaling into the q_proj output scale
        # q_output_scale = q_output_scale * module.scaling
        # module.q_proj.weight *= module.scaling
        # module.q_proj.bias *= module.scaling
        # import pdb; pdb.set_trace()
        int8_module.q_proj = W8A8B8O8Linear.from_float(module.q_proj, input_scale, q_output_scale)
        int8_module.k_proj = W8A8B8O8Linear.from_float(
            module.k_proj, input_scale, k_output_scale)
        int8_module.v_proj = W8A8B8O8Linear.from_float(
            module.v_proj, input_scale, v_output_scale)
        int8_module.out_proj = W8A8BFP32OFP32Linear.from_float(
            module.o_proj, out_input_scale)
        int8_module.qk_bmm = BMM_S8T_S8N_F32T.from_scale(
            q_output_scale, k_output_scale)

        # alpha = s_prob * s_v / s_out, where s_prob = 1 / 127
        int8_module.pv_bmm = BMM_S8T_S8N_S8T.from_scale(
            1.0 / 127, v_output_scale, out_input_scale)
        return int8_module

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # @exec_time
    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(
            query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = self.qk_bmm(query_states, key_states)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(
                torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len)

        attn_probs = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_probs = layer_head_mask.view(
                1, -1, 1, 1) * attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
            attn_probs = attn_probs.view(
                bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_probs_reshaped = attn_probs.view(
                bsz, self.num_heads, tgt_len, src_len)
            attn_probs = attn_probs_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_probs_reshaped = None

        # (A_row V_row)_row = (A_row V_col ^T)_row
        attn_probs.mul_(127).round_()
        attn_probs = attn_probs.to(torch.int8)

        value_states = value_states.transpose(1, 2).contiguous()
        attn_output = self.pv_bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(
            bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(
            bsz, tgt_len, self.embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_probs_reshaped, past_key_value


class Int8LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act

        self.gate_proj = W8A8B8O8LinearReLU(self.hidden_size, self.intermediate_size)
        self.up_proj = W8A8B8O8Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = W8A8BFP32OFP32Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = ACT2FN[self.hidden_act]

    @staticmethod
    @torch.no_grad()
    def from_float(module: LlamaMLP,
                   input_scale: float,
                   gate_out_scale: float,
                   up_out_scale: float,
                   down_in_scale: float):
        int8_module = Int8LlamaMLP(
            module.config
        )
        int8_module.gate_proj = W8A8B8O8LinearReLU.from_float(
            module.gate_proj, input_scale, gate_out_scale)
        int8_module.up_proj = W8A8B8O8Linear.from_float(
            module.up_proj, input_scale, up_out_scale)
        int8_module.down_proj = W8A8BFP32OFP32Linear.from_float(
            module.down_proj, down_in_scale)
        return int8_module

    # @exec_time
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return self.down_proj(self.gate_proj(hidden_states) * self.up_proj(hidden_states))


class Int8LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Int8LlamaAttention(config)
        self.mlp = Int8LlamaMLP(config)

        self.input_layernorm = LayerNormQ(self.embed_dim)
        self.post_attention_layernorm = LayerNormQ(self.embed_dim)

    @staticmethod
    def from_float(module: LlamaDecoderLayer,
                   attn_input_scale: float):
        int8_module = Int8LlamaDecoderLayer(module.self_attn.config
        )
        # int8_module.input_layernorm = LayerNormQ.from_float(
        #     torch.nn.LayerNorm(module.input_layernorm.weight.shape[0]), 0)
        # int8_module.post_attention_layernorm = LayerNormQ.from_float(
        #     torch.nn.LayerNorm(module.post_attention_layernorm.weight.shape[0]), 0)
        int8_module.self_attn = Int8LlamaAttention.from_float(
            module.self_attn, 0, 0, 0, 0, 0)
        int8_module.mlp = Int8LlamaMLP.from_float(
            module.mlp, 0, 0, 0, 0)
        return int8_module

    @exec_time
    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        residual.add_(hidden_states.to(residual.dtype))

        hidden_states = self.post_attention_layernorm(residual)

        hidden_states = self.mlp(hidden_states)
        residual.add_(hidden_states.to(residual.dtype))

        outputs = (residual,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Int8LlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Int8LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = Int8LlamaModel(module.config)
        return int8_module


class Int8LlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Int8LlamaModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(
            config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = Int8LlamaForCausalLM(module.config)
        return int8_module