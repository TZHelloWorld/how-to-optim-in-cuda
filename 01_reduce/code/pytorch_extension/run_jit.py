# run_jit.py — 无需 `pip install -e .` 的 JIT 即时编译版
#
# 用 torch.utils.cpp_extension.load() 在首次运行时自动编译 reduce_kernel.cu，
# 之后直接调用，无需 setup.py / pip install。
#
# 用法:
#   python run_jit.py
#
# 说明:
#   - 首次运行会调用 nvcc 编译，结果缓存在 ~/.cache/torch_extensions/，二次运行秒开。
#   - 需要环境中有 CUDA 工具链（nvcc）与可用 GPU。

import os
import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))

# JIT 编译：等价于 setup.py 里 CUDAExtension('reduce_kernel', ['reduce_kernel.cu'])
reduce_kernel = load(
    name="reduce_kernel",
    sources=[os.path.join(_HERE, "reduce_kernel.cu")],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

# ---- 以下与 test.py 相同的验证逻辑 ----
N = 32 * 1024 * 1024
x = torch.full((N,), 2.0, device="cuda")

result = reduce_kernel.reduce_sum(x)
print(f"自定义 kernel 结果: {result.item()}")   # 期望 67108864.0

expected = x.sum()
print(f"PyTorch sum 结果:   {expected.item()}")
