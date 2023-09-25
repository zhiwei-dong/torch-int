import torch
from transformers.models.opt.modeling_opt import OPTDecoderLayer, OPTConfig
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaAttention, LlamaMLP, LlamaDecoderLayer, LlamaModel, LlamaForCausalLM
from torch_int.models.opt import Int8OPTDecoderLayer
from torch_int.models.llama import Int8LlamaDecoderLayer, Int8LlamaModel
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
from typing import Tuple
from icecream import ic
from functools import partial
from torch_int.utils.decorator import exec_time


def store_act(module, x, y, act_dict, name):
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(y, tuple):
        y = y[0]
    act_dict[name] = (x, y)


@torch.no_grad()
def test_llama_model():
    config = LlamaConfig()

    batch_size = 10
    sequence_length = 20

    embed_dim = 4096
    vocab_size = 32000

    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))
    layer = LlamaForCausalLM(config)
    int8_layer = Int8LlamaModel.from_float(
        layer, 0).cuda()
    int8_layer.eval()
    import pdb; pdb.set_trace()

    _ = int_infer(int8_layer, input_ids)

@exec_time
def int_infer(model, input):
    return model(input)


if __name__ == '__main__':
    test_llama_model()
