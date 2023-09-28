import torch
from transformers.models.opt.modeling_opt import OPTDecoderLayer, OPTConfig
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaAttention, LlamaMLP, LlamaDecoderLayer, LlamaModel, LlamaForCausalLM
from torch_int.models.opt import Int8OPTDecoderLayer
# from torch_int.models.llama import Int8LlamaDecoderLayer, Int8LlamaModel
from torch_int.models.llama_csm import Int8LlamaModel
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU
from typing import Tuple
from icecream import ic
from functools import partial
from torch_int.utils.decorator import exec_time
import json

with open("config.json", "r") as f:
    config_dict = json.load(f)

def store_act(module, x, y, act_dict, name):
    if isinstance(x, tuple):
        x = x[0]
    if isinstance(y, tuple):
        y = y[0]
    act_dict[name] = (x, y)


@torch.no_grad()
def test_llama_model():
    # act: relu
    config = LlamaConfig.from_dict(config_dict)

    batch_size = 1
    sequence_length = 512
    vocab_size = 32000

    input_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length))
    print('done load input_ids')
    layer = LlamaForCausalLM(config)
    layer.to(torch.float16)
    layer.eval()
    print('done load layer')
    
    # import pdb; pdb.set_trace()
    input_ids = input_ids.cuda()
    # layer = layer.cuda()
    # _ = fp16_infer_30(layer, input_ids)
    # layer = layer.cpu()

    int8_layer = Int8LlamaModel.from_float(layer, 0).cuda()
    print('done load int8_layer')
    int8_layer.eval()
    _ = int_infer_30(int8_layer, input_ids)

@exec_time
def int_infer_30(model, input):
    with torch.no_grad():
        for _ in range(30):
            model(input)
    return

@exec_time
def fp16_infer_30(model, input):
    with torch.no_grad():
        for _ in range(1):
            model(input)
    return

if __name__ == '__main__':
    test_llama_model()
