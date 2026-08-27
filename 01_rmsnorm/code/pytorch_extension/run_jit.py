# run_jit.py — 无需 `pip install -e .` 的 JIT 即时编译版
#
# 用 torch.utils.cpp_extension.load() 在首次运行时自动编译 rmsnorm_kernel.cu。
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

rmsnorm_kernel = load(
    name="rmsnorm_kernel",
    sources=[os.path.join(_HERE, "rmsnorm_kernel.cu")],
    extra_cuda_cflags=["-O3", "-arch=sm_70"],
    verbose=True,
)


def rms_norm_ref(x, gamma, eps=1e-6):
    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * rms).to(x.dtype) * gamma


def test_rmsnorm():
    N, H = 4096, 4096
    eps = 1e-6
    x = torch.randn(N, H, device="cuda")
    gamma = torch.rand(H, device="cuda") + 0.5

    y_ref = rms_norm_ref(x, gamma, eps)
    y_mine = rmsnorm_kernel.rmsnorm(x, gamma, eps)

    ok = torch.allclose(y_mine, y_ref, rtol=1e-4, atol=1e-4)
    max_err = (y_mine - y_ref).abs().max().item()
    print(f"[rmsnorm]           allclose={ok}  max_abs_err={max_err:.3e}")


def test_fused_add_rmsnorm():
    N, H = 4096, 4096
    eps = 1e-6
    x = torch.randn(N, H, device="cuda")
    residual = torch.randn(N, H, device="cuda")
    gamma = torch.rand(H, device="cuda") + 0.5

    h_ref = x + residual
    y_ref = rms_norm_ref(h_ref, gamma, eps)

    x_io = x.clone()
    residual_io = residual.clone()
    rmsnorm_kernel.fused_add_rmsnorm(x_io, residual_io, gamma, eps)

    ok_h = torch.allclose(residual_io, h_ref, rtol=1e-4, atol=1e-4)
    ok_y = torch.allclose(x_io, y_ref, rtol=1e-4, atol=1e-4)
    err_h = (residual_io - h_ref).abs().max().item()
    err_y = (x_io - y_ref).abs().max().item()
    print(f"[fused_add_rmsnorm] 残差 allclose={ok_h} max_err={err_h:.3e} | "
          f"输出 allclose={ok_y} max_err={err_y:.3e}")


if __name__ == "__main__":
    test_rmsnorm()
    test_fused_add_rmsnorm()
