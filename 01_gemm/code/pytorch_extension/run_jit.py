# run_jit.py — 无需 `pip install -e .` 的 JIT 即时编译版
#
# 用 torch.utils.cpp_extension.load() 在首次运行时自动编译 gemm_kernel.cu。
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

gemm_kernel = load(
    name="gemm_kernel",
    sources=[os.path.join(_HERE, "gemm_kernel.cu")],
    extra_cuda_cflags=["-O3", "-arch=sm_70"],
    verbose=True,
)

# ---- 以下与 test.py 相同的验证逻辑 ----
# sgemm_v5 要求 M、N 被 128 整除，K 被 8 整除
M = N = K = 4096
A = torch.randn(M, K, device="cuda")
B = torch.randn(K, N, device="cuda")

C1 = gemm_kernel.my_matmul(A, B)
C2 = A @ B
print("allclose:", torch.allclose(C1, C2, rtol=1e-3, atol=1e-3))


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


t = bench(lambda: gemm_kernel.my_matmul(A, B))
tflops = 2 * M * N * K / (t * 1e-3) / 1e12
tc = bench(lambda: A @ B)
print(f"my_matmul: {t:.3f} ms, {tflops:.1f} TFLOPS   (cuBLAS: {tc:.3f} ms)")
