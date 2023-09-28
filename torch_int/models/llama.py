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
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from typing import Optional, Tuple, List
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
from torch_int.nn.fused import LayerNormQ
from transformers.utils import logging
from transformers.activations import ACT2FN
from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T
logger = logging.get_logger(__name__)
import math
import torch.nn.functional as F


class Int8LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self, config: LlamaConfig
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.qk_bmm = BMM_S8T_S8N_F32T(1.0)
        self.pv_bmm = BMM_S8T_S8N_S8T(1.0)

        self.k_proj = W8A8B8O8Linear(self.hidden_size, self.hidden_size)
        self.v_proj = W8A8B8O8Linear(self.hidden_size, self.hidden_size)
        self.q_proj = W8A8B8O8Linear(self.hidden_size, self.hidden_size)
        self.o_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.hidden_size)

        self._init_rope()

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
        int8_module.o_proj = W8A8BFP32OFP32Linear.from_float(
            module.o_proj, out_input_scale)
        int8_module.qk_bmm = BMM_S8T_S8N_F32T.from_scale(
            q_output_scale, k_output_scale)

        # alpha = s_prob * s_v / s_out, where s_prob = 1 / 127
        int8_module.pv_bmm = BMM_S8T_S8N_S8T.from_scale(
            1.0 / 127, v_output_scale, out_input_scale)
        return int8_module

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # if self.config.pretraining_tp > 1:
        #     key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        #     query_slices = self.q_proj.weight.split(
        #         (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        #     )
        #     key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        #     value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        #     query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        #     query_states = torch.cat(query_states, dim=-1)

        #     key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        #     key_states = torch.cat(key_states, dim=-1)

        #     value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        #     value_states = torch.cat(value_states, dim=-1)

        # else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)


        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        proj_shape = (bsz * self.num_heads, q_len, self.head_dim)
        query_states = query_states.reshape(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        ori_shape = (bsz, self.num_heads, q_len, self.head_dim)
        query_states = query_states.reshape(*ori_shape)
        key_states = key_states.reshape(*ori_shape)
        value_states = value_states.reshape(*ori_shape)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states = query_states.reshape(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        attn_weights = self.qk_bmm(query_states, key_states) / math.sqrt(self.head_dim)

        attn_weights = attn_weights.reshape(bsz, self.num_heads, q_len, kv_seq_len)
        # import pdb; pdb.set_trace()
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            # attn_weights = attn_weights + attention_mask
            #! add by dongz, delete attention_mask
            attn_weights = attn_weights

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # attn_weights.mul_(127).round_()
        # attn_weights = attn_weights.to(torch.int8)
        attn_weights = attn_weights.reshape(bsz * self.num_heads, q_len, kv_seq_len)
        attn_output = self.pv_bmm(attn_weights, value_states.transpose(1, 2))
        attn_output = attn_output.reshape(bsz, self.num_heads, q_len, self.head_dim)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class Int8LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act

        self.gate_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = W8A8BFP32OFP32Linear(self.hidden_size, self.intermediate_size)
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
        int8_module.gate_proj = W8A8BFP32OFP32Linear.from_float(
            module.gate_proj, input_scale, gate_out_scale)
        int8_module.up_proj = W8A8BFP32OFP32Linear.from_float(
            module.up_proj, input_scale, up_out_scale)
        int8_module.down_proj = W8A8BFP32OFP32Linear.from_float(
            module.down_proj, down_in_scale)
        return int8_module

    # @exec_time
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x1 = self.act_fn(self.gate_proj(hidden_states))
        x2 = self.up_proj(hidden_states)
        x = x1 * x2
        return self.down_proj(x.round().clamp(-128, 127).to(torch.int8))
        # return self.down_proj(self.gate_proj(hidden_states) * self.up_proj(hidden_states))


class Int8LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Int8LlamaAttention(config)
        self.mlp = Int8LlamaMLP(config)
        # ln_output_int8 = ln_output_fp.round().clamp(-128, 127).to(torch.int8)

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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

    # @exec_time
    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states).round().clamp(-128, 127).to(torch.int8)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states).round().clamp(-128, 127).to(torch.int8)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Int8LlamaModel(LlamaPreTrainedModel):
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
    get_input_embeddings = LlamaModel.get_input_embeddings
    set_input_embeddings = LlamaModel.set_input_embeddings
    _prepare_decoder_attention_mask = LlamaModel._prepare_decoder_attention_mask
    forward = LlamaModel.forward

    @staticmethod
    def from_float(module, decoder_layer_scales):
        int8_module = Int8LlamaModel(module.config)
        return int8_module


class Int8LlamaForCausalLM(LlamaPreTrainedModel):
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

    get_input_embeddings = LlamaForCausalLM.get_input_embeddings
    set_input_embeddings = LlamaForCausalLM.set_input_embeddings
    get_output_embeddings = LlamaForCausalLM.get_output_embeddings
    set_output_embeddings = LlamaForCausalLM.set_output_embeddings
    set_decoder = LlamaForCausalLM.set_decoder
    get_decoder = LlamaForCausalLM.get_decoder
    forward = LlamaForCausalLM.forward
    prepare_inputs_for_generation = LlamaForCausalLM.prepare_inputs_for_generation
    _reorder_cache = LlamaForCausalLM._reorder_cache