# run_jit.py — 无需 `pip install -e .` 的 JIT 即时编译版
#
# 用 torch.utils.cpp_extension.load() 在首次运行时自动编译 copy_kernel.cu。
#
# 用法:
#   python run_jit.py
#
# 说明:
#   - 首次运行调用 nvcc 编译，缓存在 ~/.cache/torch_extensions/，二次运行秒开。
#   - 需要 CUDA 工具链（nvcc）与可用 GPU。

import os
import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))

copy_kernel = load(
    name="copy_kernel",
    sources=[os.path.join(_HERE, "copy_kernel.cu")],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

# ---- 以下与 test.py 相同的验证逻辑 ----
N = 32 * 1024 * 1024
x = torch.randn(N, device="cuda")

# 1) 纯拷贝：结果应与输入完全一致
y = copy_kernel.copy_float4(x)
print(f"copy_float4 一致: {torch.equal(y, x)}")   # 期望 True

# 2) 拷贝 + scale + ReLU 融合：结果应等于 relu(x * alpha)
alpha = 2.0
z = copy_kernel.copy_scale_relu(x, alpha)
expected = torch.relu(x * alpha)
print(f"copy_scale_relu 最大误差: {(z - expected).abs().max().item():.3e}")  # 期望 ~0
