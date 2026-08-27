import torch
import rmsnorm_kernel


def rms_norm_ref(x, gamma, eps=1e-6):
    # 参考实现：与 torch.nn.RMSNorm（PyTorch >= 2.4）一致
    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * rms).to(x.dtype) * gamma


def test_rmsnorm():
    N, H = 4096, 4096
    eps = 1e-6
    x = torch.randn(N, H, device='cuda')
    gamma = torch.rand(H, device='cuda') + 0.5          # 非平凡 γ，避开全 1/全 0

    y_ref = rms_norm_ref(x, gamma, eps)
    y_mine = rmsnorm_kernel.rmsnorm(x, gamma, eps)

    ok = torch.allclose(y_mine, y_ref, rtol=1e-4, atol=1e-4)
    max_err = (y_mine - y_ref).abs().max().item()
    print(f"[rmsnorm]           allclose={ok}  max_abs_err={max_err:.3e}")   # fp32 预期 ~1e-6 量级


def test_fused_add_rmsnorm():
    N, H = 4096, 4096
    eps = 1e-6
    x = torch.randn(N, H, device='cuda')
    residual = torch.randn(N, H, device='cuda')
    gamma = torch.rand(H, device='cuda') + 0.5

    # 参考：h = x + residual; y = RMSNorm(h) * gamma
    h_ref = x + residual
    y_ref = rms_norm_ref(h_ref, gamma, eps)

    # 融合版原地改写 x(→y)、residual(→h)，需保留副本对拍
    x_io = x.clone()
    residual_io = residual.clone()
    rmsnorm_kernel.fused_add_rmsnorm(x_io, residual_io, gamma, eps)

    ok_h = torch.allclose(residual_io, h_ref, rtol=1e-4, atol=1e-4)
    ok_y = torch.allclose(x_io, y_ref, rtol=1e-4, atol=1e-4)
    err_h = (residual_io - h_ref).abs().max().item()
    err_y = (x_io - y_ref).abs().max().item()
    print(f"[fused_add_rmsnorm] 残差 allclose={ok_h} max_err={err_h:.3e} | "
          f"输出 allclose={ok_y} max_err={err_y:.3e}")


if __name__ == '__main__':
    test_rmsnorm()
    test_fused_add_rmsnorm()
