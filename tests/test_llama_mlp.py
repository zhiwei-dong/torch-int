import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTConfig
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaAttention, LlamaMLP
from torch_int.models.opt import Int8OPTAttention
from torch_int.models.llama import Int8LlamaMLP
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
def test_llama_mlp():
    #! add by dongz
    # init config
    config = LlamaConfig()
    B, L, D, H = 1, 4096, 4096, 1
    x = torch.randn(B, L, D)
    x_scale = x.abs().max() / 127
    mlp = LlamaMLP(config)
    mlp.eval()
    act_dict = {}
    for name, module in mlp.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(
                partial(store_act, act_dict=act_dict, name=name))
    y = mlp(x)[0]

    gate_output_scale = act_dict['gate_proj'][1].abs().max() / 127
    up_output_scale = act_dict['up_proj'][1].abs().max() / 127
    down_output_scale = act_dict['down_proj'][0].abs().max() / 127
    int8_mlp = Int8LlamaMLP.from_float(
        mlp, x_scale, gate_output_scale, up_output_scale, down_output_scale).cuda()
    int8_mlp.eval()
    q_act_dict = {}
    for name, module in int8_mlp.named_modules():
        if isinstance(module, (W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU)):
            module.register_forward_hook(
                partial(store_act, act_dict=q_act_dict, name=name))
    q_x = (x / x_scale).round().to(torch.int8)
    y_hat = int8_mlp(q_x.cuda())[0].cpu()

    # ic(y_hat)
    r2 = (y - y_hat).pow(2).mean() / y.pow(2).mean()
    import pdb; pdb.set_trace()
    # ic(r2)
    print(r2)


if __name__ == '__main__':
    test_llama_mlp()
