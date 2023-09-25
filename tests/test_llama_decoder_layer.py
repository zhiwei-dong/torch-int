import torch
from transformers.models.opt.modeling_opt import OPTDecoderLayer, OPTConfig
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaAttention, LlamaMLP, LlamaDecoderLayer
from torch_int.models.opt import Int8OPTDecoderLayer
from torch_int.models.llama import Int8LlamaDecoderLayer
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
from typing import Tuple
from icecream import ic
from functools import partial


def store_act(module, x, y, act_dict, name):
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(y, tuple):
        y = y[0]
    act_dict[name] = (x, y)


@torch.no_grad()
def test_llama_decoder_layer():
    config = LlamaConfig()
    B, L, D, H = 1, 2048, config.hidden_size, config.num_attention_heads

    x = torch.randn(B, L, D)
    layer = LlamaDecoderLayer(config)
    layer.eval()
    act_dict = {}
    for name, module in layer.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(
                partial(store_act, act_dict=act_dict, name=name))
    y = layer(x)[0]

    int8_layer = Int8LlamaDecoderLayer.from_float(
        layer, 0).cuda()
    int8_layer.eval()

    y_hat = int8_layer(x.cuda())[0].cpu()

    # # ic(y_hat)
    # r2 = (y - y_hat).pow(2).mean() / y.pow(2).mean()
    # ic(r2)


if __name__ == '__main__':
    test_llama_decoder_layer()
