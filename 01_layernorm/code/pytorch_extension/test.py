# test.py — 自定义 LayerNorm CUDA 扩展与 PyTorch nn.LayerNorm 对拍（对应文档第 9.3 节）
#
# 用法:
#   pip install -e .
#   python test.py

import torch
import layernorm_ext

N, H = 4096, 4096
eps = 1e-5

x = torch.randn(N, H, device="cuda")
ln = torch.nn.LayerNorm(H, eps=eps).cuda()
with torch.no_grad():
    ln.weight.copy_(torch.rand(H) + 0.5)     # 非平凡的 γ/β，避免仿射错误被 1/0 掩盖
    ln.bias.copy_(torch.rand(H) - 0.5)

y_ref = ln(x)
y_mine = layernorm_ext.layernorm(x, ln.weight, ln.bias, eps)

max_err = (y_mine - y_ref).abs().max().item()
ok = torch.allclose(y_mine, y_ref, rtol=1e-4, atol=1e-4)   # fp32 容差

print(f"max abs err = {max_err:.3e}")        # fp32 预期 1e-6 ~ 1e-5 量级
print("allclose:", ok)

# ---- 极端分布回归：μ >> σ，暴露单遍 naive 的灾难性抵消（文档 2.2 节）----
x_hard = torch.randn(N, H, device="cuda") + 1e4
y_ref_hard = ln(x_hard)
y_mine_hard = layernorm_ext.layernorm(x_hard, ln.weight, ln.bias, eps)
hard_err = (y_mine_hard - y_ref_hard).abs().max().item()
print(f"[hard μ>>σ] max abs err = {hard_err:.3e}  "
      f"(单遍 naive 版预期在此偏大；Welford 版才稳定)")
