# run_jit.py — 无需 `pip install -e .` 的 JIT 即时编译版
#
# 用 torch.utils.cpp_extension.load() 在首次运行时自动编译 layernorm_kernel.cu。
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

# 模块名与 setup.py 保持一致（layernorm_ext）
layernorm_ext = load(
    name="layernorm_ext",
    sources=[os.path.join(_HERE, "layernorm_kernel.cu")],
    extra_cuda_cflags=["-O3", "-arch=sm_70"],
    verbose=True,
)

# ---- 以下与 test.py 相同的验证逻辑 ----
N, H = 4096, 4096
eps = 1e-5

x = torch.randn(N, H, device="cuda")
ln = torch.nn.LayerNorm(H, eps=eps).cuda()
with torch.no_grad():
    ln.weight.copy_(torch.rand(H) + 0.5)
    ln.bias.copy_(torch.rand(H) - 0.5)

y_ref = ln(x)
y_mine = layernorm_ext.layernorm(x, ln.weight, ln.bias, eps)

max_err = (y_mine - y_ref).abs().max().item()
ok = torch.allclose(y_mine, y_ref, rtol=1e-4, atol=1e-4)

print(f"max abs err = {max_err:.3e}")
print("allclose:", ok)

# ---- 极端分布回归：μ >> σ ----
x_hard = torch.randn(N, H, device="cuda") + 1e4
y_ref_hard = ln(x_hard)
y_mine_hard = layernorm_ext.layernorm(x_hard, ln.weight, ln.bias, eps)
hard_err = (y_mine_hard - y_ref_hard).abs().max().item()
print(f"[hard μ>>σ] max abs err = {hard_err:.3e}  "
      f"(单遍 naive 版预期在此偏大；Welford 版才稳定)")
