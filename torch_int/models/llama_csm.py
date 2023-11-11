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
from torch_int.models.csm_module import CSMModule
from torch_int.models.llama import Int8LlamaAttention
from transformers.utils import logging
from transformers.activations import ACT2FN
from torch_int.nn.bmm import BMM_S8T_S8N_S8T, BMM_S8T_S8N_F32T
logger = logging.get_logger(__name__)
import math
import torch.nn.functional as F


def split_fc(num_split, fcs):
    if not isinstance(fcs, list):
        fcs = [fcs]
    for idx, fc in enumerate(fcs):
        fc.weight.data = torch.repeat_interleave(
            fc.weight.data, num_split.int(), dim=1
        )
        fc.in_features = fc.weight.shape[1]


def merge_fc(src_idx, dst_idx, fcs):
        if not isinstance(fcs, list):
            fcs = [fcs]
        for idx, fc in enumerate(fcs):
            cout, cin = fc.weight.shape
            dev = fc.weight.device
            dtype = fc.weight.dtype
            ori_src_idx = torch.arange(0, cin, 2, device=dev)
            ori_dst_idx = torch.arange(1, cin, 2, device=dev)
            src_idx_ = ori_src_idx[src_idx]
            dst_idx_ = ori_dst_idx[dst_idx]
            r = src_idx_.nelement()

            channel_mask = torch.ones(cin, device=dev, dtype=dtype)
            channel_mask[src_idx_] = 0.0

            src_weight = fc.weight.gather(dim=-1, index=src_idx_.expand(cout, r))
            fc.weight.data.scatter_reduce_(
                dim=-1, index=dst_idx_.expand(cout, r), src=src_weight, reduce="sum"
            )
            fc.weight.data = fc.weight.data.index_select(
                -1, (channel_mask != 0).nonzero().squeeze()
            )
            fc.in_features = (channel_mask != 0).sum()


def num_split_to_index(num_split):
    # Create a tensor cumulative_indices with the total number of elements required
    cumulative_indices = torch.zeros(num_split.sum().int(), dtype=torch.int32, device=num_split.device)

    # The first index is always 0
    cumulative_indices[0] = 0

    # Calculate the start indices for each new value
    cumulative_sums = num_split.cumsum(0)
    start_indices = torch.zeros_like(num_split)
    start_indices[1:] = cumulative_sums[:-1]  # Set the start index for each group

    # Scatter the start_indices into the cumulative_indices tensor
    cumulative_indices.scatter_(0, start_indices.long(), 1)

    # The cumulative sum of cumulative_indices now gives us the indices we want
    index_tensor = cumulative_indices.cumsum(0) - 1
    return index_tensor


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

        self.fc1_csm = CSMModule(self.up_proj, config.channel_ratio)

        # self.num_channels = int(config.channel_ratio * self.intermediate_size)
        # # split
        # outlier_channel_idx = torch.arange(self.num_channels)
        # num_split = torch.ones(self.intermediate_size)
        # num_split[0] = self.num_channels
        # scaling_factors = torch.ones(self.intermediate_size + self.num_channels - 1)
        # scaling_factors[0] = 1 / self.num_channels
        
        # del self.fc1_csm.outlier_channel_idx
        # del self.fc1_csm.num_split
        # del self.fc1_csm.scaling_factors
        # self.fc1_csm.register_buffer("outlier_channel_idx", outlier_channel_idx)
        # self.fc1_csm.register_buffer("num_split", num_split)
        # self.fc1_csm.register_buffer("scaling_factors", scaling_factors)
        # self.fc1_csm.num_additional_channels = self.num_channels - 1
        # self.fc1_csm.num_merged_channels = self.num_channels - 1

        # # merge
        # src_idx = torch.arange(self.num_channels - 1)
        # dst_idx = torch.arange(self.num_channels - 1)
        # del self.fc1_csm.src_idx
        # del self.fc1_csm.dst_idx
        # self.fc1_csm.register_buffer("src_idx", src_idx)
        # self.fc1_csm.register_buffer("dst_idx", dst_idx)
        # self.fc1_csm.have_merge = True

        # split_fc(self.fc1_csm.num_split, [self.down_proj])
        # merge_fc(self.fc1_csm.src_idx, self.fc1_csm.dst_idx, [self.down_proj])


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
        # x = self.fc1_csm(x)
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

        self.input_layernorm_csm = CSMModule(self.self_attn, config.channel_ratio)
        self.post_attention_layernorm_csm = CSMModule(self.mlp.up_proj, config.channel_ratio)

        self.num_channels = int(config.channel_ratio * config.hidden_size)
        # split
        outlier_channel_idx = torch.arange(self.num_channels)
        num_split = torch.ones(config.hidden_size)
        num_split[0] = self.num_channels
        scaling_factors = torch.ones(config.hidden_size + self.num_channels - 1)
        scaling_factors[0] = 1 / self.num_channels
        index = num_split_to_index(num_split)
        
        # input_layernorm_csm
        del self.input_layernorm_csm.outlier_channel_idx
        del self.input_layernorm_csm.num_split
        del self.input_layernorm_csm.scaling_factors
        self.input_layernorm_csm.register_buffer("outlier_channel_idx", outlier_channel_idx)
        self.input_layernorm_csm.register_buffer("num_split", num_split)
        self.input_layernorm_csm.register_buffer("scaling_factors", scaling_factors)
        self.input_layernorm_csm.register_buffer("index", index)
        self.input_layernorm_csm.num_additional_channels = self.num_channels - 1
        self.input_layernorm_csm.num_merged_channels = self.num_channels - 1

        # merge
        src_idx = torch.arange(self.num_channels - 1)
        dst_idx = torch.arange(self.num_channels - 1)
        del self.input_layernorm_csm.src_idx
        del self.input_layernorm_csm.dst_idx
        self.input_layernorm_csm.register_buffer("src_idx", src_idx)
        self.input_layernorm_csm.register_buffer("dst_idx", dst_idx)
        self.input_layernorm_csm.have_merge = True

        split_fc(self.input_layernorm_csm.num_split, [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj])
        merge_fc(self.input_layernorm_csm.src_idx, self.input_layernorm_csm.dst_idx, [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj])
        # print("Test")
        # self.self_attn.q_proj.weight.contiguous()
        # self.self_attn.k_proj.weight.contiguous()
        # self.self_attn.v_proj.weight.contiguous()

        # post_attention_layernorm_csm
        del self.post_attention_layernorm_csm.outlier_channel_idx
        del self.post_attention_layernorm_csm.num_split
        del self.post_attention_layernorm_csm.scaling_factors
        self.post_attention_layernorm_csm.register_buffer("outlier_channel_idx", outlier_channel_idx)
        self.post_attention_layernorm_csm.register_buffer("num_split", num_split)
        self.post_attention_layernorm_csm.register_buffer("scaling_factors", scaling_factors)
        self.post_attention_layernorm_csm.register_buffer("index", index)
        self.post_attention_layernorm_csm.num_additional_channels = self.num_channels - 1
        self.post_attention_layernorm_csm.num_merged_channels = self.num_channels - 1

        # merge
        src_idx = torch.arange(self.num_channels - 1)
        dst_idx = torch.arange(self.num_channels - 1)
        del self.post_attention_layernorm_csm.src_idx
        del self.post_attention_layernorm_csm.dst_idx
        self.post_attention_layernorm_csm.register_buffer("src_idx", src_idx)
        self.post_attention_layernorm_csm.register_buffer("dst_idx", dst_idx)
        self.post_attention_layernorm_csm.have_merge = True

        split_fc(self.post_attention_layernorm_csm.num_split, [self.mlp.gate_proj, self.mlp.up_proj])
        merge_fc(self.post_attention_layernorm_csm.src_idx, self.post_attention_layernorm_csm.dst_idx, [self.mlp.gate_proj, self.mlp.up_proj])
        # self.mlp.gate_proj.weight.contiguous()
        # self.mlp.up_proj.weight.contiguous()
        # print("Test")

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

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.input_layernorm_csm(hidden_states).round().clamp(-128, 127).to(torch.int8)
        # hidden_states = hidden_states.contiguous()

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
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.post_attention_layernorm_csm(hidden_states).round().clamp(-128, 127).to(torch.int8)
        # hidden_states = hidden_states.contiguous()
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