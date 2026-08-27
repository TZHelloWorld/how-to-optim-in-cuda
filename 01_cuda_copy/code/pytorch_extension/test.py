import torch
import copy_kernel

N = 32 * 1024 * 1024
x = torch.randn(N, device='cuda')

# 1) 纯拷贝：结果应与输入完全一致
y = copy_kernel.copy_float4(x)
print(f"copy_float4 一致: {torch.equal(y, x)}")   # 期望 True

# 2) 拷贝 + scale + ReLU 融合：结果应等于 relu(x * alpha)
alpha = 2.0
z = copy_kernel.copy_scale_relu(x, alpha)
expected = torch.relu(x * alpha)
print(f"copy_scale_relu 最大误差: {(z - expected).abs().max().item():.3e}")  # 期望 ~0
