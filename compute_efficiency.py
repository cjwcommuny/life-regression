from typing import Tuple

import torch
from fvcore.nn import FlopCountAnalysis
from torch import nn

from tv.sota import get_model, get_model_configs


def compute_flops(model: nn.Module) -> float:
    input = torch.rand(1, 3, 224, 224)
    flops = FlopCountAnalysis(model, input)
    return flops.total()

def compute_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_model_complexity(model: nn.Module) -> Tuple[float, float]:
    flops = compute_flops(model)
    params = compute_params(model)
    return flops / 10 ** 9, params / 10 ** 6

def compute_result(model_config: dict) -> str:
    model = get_model(**model_config).eval()
    complexity = compute_model_complexity(model)
    return f'{model_config}: model complexity: {complexity}'


if __name__ == '__main__':
    results = list(map(compute_result, get_model_configs()))
    print('---')
    for result in results:
        print(result)
