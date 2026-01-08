"""
Tensor Hook 演示

演示 PyTorch 中 Tensor.register_hook() 的用法：
- 查看中间 Tensor 的梯度
- 修改 Tensor 的梯度
- 梯度翻倍/缩放
- 梯度裁剪
- 调试梯度计算

参考: https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html
"""

import torch
import torch.nn as nn


def demo_basic_tensor_hook():
    """演示基本的 tensor hook — 查看和修改梯度"""
    print("=" * 60)
    print("Demo 1: 基本 Tensor Hook — 查看中间 Tensor 梯度")
    print("=" * 60)

    # 创建需要梯度的 tensor
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # y = x^2，z = y * 3
    y = x ** 2
    z = (y * 3).sum()

    # 问题: y 是中间变量，backward 后 y.grad 默认是 None
    z.backward()
    print(f"\n  x.grad = {x.grad}")      # 有值: dy/dx * dz/dy = 2x * 3 = [6, 12, 18]
    print(f"  y.grad = {y.grad}")          # None! 中间变量的梯度默认不保留

    # 使用 tensor hook 来捕获中间 tensor 的梯度
    print("\n  使用 tensor hook 捕获 y 的梯度:")
    x.grad = None  # 清零
    y = x ** 2

    y_grad_captured = {}

    def capture_grad(grad):
        y_grad_captured['grad'] = grad.clone()
        # 返回 None 表示不修改梯度

    y.register_hook(capture_grad)

    z = (y * 3).sum()
    z.backward()

    print(f"  y 的梯度 (通过 hook 捕获): {y_grad_captured['grad']}")
    print(f"  x.grad = {x.grad}")
    print()


def demo_modify_gradient():
    """演示用 tensor hook 修改梯度"""
    print("=" * 60)
    print("Demo 2: Tensor Hook — 修改梯度")
    print("=" * 60)

    # === 正常梯度 ===
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x * 3).sum()
    y.backward()
    print(f"\n  正常梯度: x.grad = {x.grad}")  # [3, 3, 3]

    # === 使用 hook 将梯度翻倍 ===
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    h = x.register_hook(lambda grad: grad * 2)  # 梯度翻倍
    y = (x * 3).sum()
    y.backward()
    print(f"  梯度翻倍: x.grad = {x.grad}")  # [6, 6, 6]
    h.remove()

    # === 使用 hook 将梯度置零 (冻结参数) ===
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    h = x.register_hook(lambda grad: torch.zeros_like(grad))
    y = (x * 3).sum()
    y.backward()
    print(f"  梯度置零: x.grad = {x.grad}")  # [0, 0, 0]
    h.remove()

    # === 使用 hook 对梯度裁剪 ===
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    h = x.register_hook(lambda grad: torch.clamp(grad, -2.0, 2.0))
    y = (x * 5).sum()  # 梯度为 [5, 5, 5]
    y.backward()
    print(f"  梯度裁剪: x.grad = {x.grad}")  # [2, 2, 2]
    h.remove()
    print()


def demo_selective_gradient_scaling():
    """演示对不同参数使用不同的梯度缩放"""
    print("=" * 60)
    print("Demo 3: Tensor Hook — 参数级别的差异化学习率")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(4, 3),
        nn.ReLU(),
        nn.Linear(3, 1),
    )

    # 对第一层权重的梯度缩放 0.1 (模拟较小的学习率)
    # 对最后一层权重的梯度缩放 10.0 (模拟较大的学习率)
    # 保存 handle，用完及时移除（最佳实践）
    h1 = model[0].weight.register_hook(lambda grad: grad * 0.1)
    h2 = model[2].weight.register_hook(lambda grad: grad * 10.0)

    x = torch.randn(2, 4)
    output = model(x)
    loss = output.sum()
    loss.backward()

    print(f"\n  第一层 weight 梯度范数: {model[0].weight.grad.norm():.6f} (缩放了 0.1x)")
    print(f"  最后层 weight 梯度范数: {model[2].weight.grad.norm():.6f} (缩放了 10x)")
    print()

    h1.remove()
    h2.remove()


def demo_gradient_debug():
    """演示用 tensor hook 调试复杂计算图中的梯度"""
    print("=" * 60)
    print("Demo 4: Tensor Hook — 调试复杂计算图的梯度流")
    print("=" * 60)

    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)

    # 复杂计算: c = a*b, d = a+b, e = c*d, f = e^2
    c = a * b
    d = a + b
    e = c * d
    f = e ** 2

    # 为中间变量注册 hook 来查看梯度
    grad_log = {}

    def make_hook(name):
        def hook(grad):
            grad_log[name] = grad.item()
            print(f"  {name:>2s} 的梯度: {grad.item():.4f}")
        return hook

    c.register_hook(make_hook('c'))
    d.register_hook(make_hook('d'))
    e.register_hook(make_hook('e'))

    print(f"\n  计算: c=a*b={c.item():.1f}, d=a+b={d.item():.1f}, "
          f"e=c*d={e.item():.1f}, f=e^2={f.item():.1f}")
    print("\n  反向传播梯度流 (f -> e -> c,d -> a,b):")

    f.backward()

    print(f"\n  a 的梯度: {a.grad.item():.4f}")
    print(f"  b 的梯度: {b.grad.item():.4f}")

    # 手动验证
    # f = e^2, df/de = 2e = 2*30 = 60
    # e = c*d, de/dc = d = 5, de/dd = c = 6
    # df/dc = df/de * de/dc = 60 * 5 = 300
    # df/dd = df/de * de/dd = 60 * 6 = 360
    # c = a*b, dc/da = b = 3, dc/db = a = 2
    # d = a+b, dd/da = 1, dd/db = 1
    # df/da = df/dc * dc/da + df/dd * dd/da = 300*3 + 360*1 = 1260
    # df/db = df/dc * dc/db + df/dd * dd/db = 300*2 + 360*1 = 960
    print("\n  手动验证:")
    print(f"  df/da = 300*3 + 360*1 = 1260, 实际: {a.grad.item():.0f}")
    print(f"  df/db = 300*2 + 360*1 = 960,  实际: {b.grad.item():.0f}")
    print()


def demo_tensor_hook_in_training():
    """演示在训练循环中使用 tensor hook 监控梯度"""
    print("=" * 60)
    print("Demo 5: Tensor Hook — 训练过程中监控参数梯度")
    print("=" * 60)

    # 简单的线性回归
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 生成数据
    torch.manual_seed(42)
    X = torch.randn(10, 3)
    y_true = X @ torch.tensor([1.0, 2.0, 3.0]).unsqueeze(1) + 0.5

    # 监控 weight 梯度的历史
    weight_grad_norms = []

    def track_weight_grad(grad):
        weight_grad_norms.append(grad.norm().item())

    handle = model.weight.register_hook(track_weight_grad)

    # 训练 5 个 epoch
    print()
    for epoch in range(5):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = nn.MSELoss()(y_pred, y_true)
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, "
              f"weight_grad_norm={weight_grad_norms[-1]:.4f}")

    handle.remove()

    print(f"\n  梯度范数变化趋势: {['%.4f' % n for n in weight_grad_norms]}")
    if weight_grad_norms[-1] < weight_grad_norms[0]:
        print("  趋势: 梯度范数在减小 (模型正在收敛)")
    print()


if __name__ == '__main__':
    demo_basic_tensor_hook()
    demo_modify_gradient()
    demo_selective_gradient_scaling()
    demo_gradient_debug()
    demo_tensor_hook_in_training()
