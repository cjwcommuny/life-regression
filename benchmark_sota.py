import time

import torch
from torch import nn, Tensor

from tv.sota import get_model, get_model_configs


def speed_test(model: nn.Module, x: Tensor, n_times: int) -> float:
    start = time.monotonic()
    for _ in range(n_times):
        _ = model(x)
    end = time.monotonic()
    return end - start


def benchmark(model_config: dict) -> str:
    model = get_model(**model_config).eval().cuda()
    elpased = speed_test(model, x, n_times)
    return f'{model_config}: time_elpased: {elpased}'


if __name__ == '__main__':
    batch_size = 128
    n_times = 50
    x = torch.rand(batch_size, 3, 224, 224).cuda()
    with torch.no_grad():
        result = list(map(benchmark, get_model_configs()))
        print('---')
        for row in result:
            print(row)
