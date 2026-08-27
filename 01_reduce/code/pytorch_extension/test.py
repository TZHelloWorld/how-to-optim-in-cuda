import torch
import reduce_kernel

N = 32 * 1024 * 1024
x = torch.full((N,), 2.0, device='cuda')

result = reduce_kernel.reduce_sum(x)
print(f"自定义 kernel 结果: {result.item()}")   # 期望 67108864.0

expected = x.sum()
print(f"PyTorch sum 结果:   {expected.item()}")
