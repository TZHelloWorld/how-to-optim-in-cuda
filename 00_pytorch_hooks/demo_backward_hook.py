"""
Backward Hook 演示

演示 PyTorch 中 Module 的反向传播 Hook 机制：
1. register_full_backward_hook: 在 backward 完成后执行，可查看/修改 grad_input
2. register_full_backward_pre_hook: 在 backward 开始前执行，可查看/修改 grad_output

参考:
- https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
- https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_pre_hook
"""

import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 3)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def demo_basic_backward_hook():
    """演示基本的 backward hook — 查看每层的梯度"""
    print("=" * 60)
    print("Demo 1: 基本 Backward Hook — 查看每层梯度信息")
    print("=" * 60)

    model = SimpleNet()
    x = torch.randn(2, 4, requires_grad=True)

    def backward_hook(name):
        def hook(module, grad_input, grad_output):
            print(f"\n  [{name}] ({module.__class__.__name__})")
            # grad_output: 输出梯度
            for i, go in enumerate(grad_output):
                if go is not None:
                    print(f"    grad_output[{i}]: shape={go.shape}, "
                          f"均值={go.mean():.6f}, 范数={go.norm():.6f}")
            # grad_input: 输入梯度
            for i, gi in enumerate(grad_input):
                if gi is not None:
                    print(f"    grad_input[{i}]:  shape={gi.shape}, "
                          f"均值={gi.mean():.6f}, 范数={gi.norm():.6f}")
        return hook

    # 为每一层注册 backward hook
    handles = []
    for name, module in model.named_modules():
        if name:  # 跳过根模块
            h = module.register_full_backward_hook(backward_hook(name))
            handles.append(h)

    # 前向 + 反向
    output = model(x)
    loss = output.sum()
    print(f"\n输出: {output.data.flatten()}")
    print(f"Loss: {loss.item():.4f}")
    print("\n--- 反向传播开始 (从 output 向 input 方向) ---")
    loss.backward()

    # 清理
    for h in handles:
        h.remove()
    print()


def demo_backward_pre_hook():
    """演示 backward pre hook — 在梯度计算前查看 grad_output"""
    print("=" * 60)
    print("Demo 2: Backward Pre Hook — 在梯度计算前查看 grad_output")
    print("=" * 60)

    model = SimpleNet()
    x = torch.randn(1, 4, requires_grad=True)

    def backward_pre_hook(module, grad_output):
        """在模块反向传播开始前调用"""
        print(f"  [Backward Pre Hook] {module.__class__.__name__}")
        print(f"    grad_output shape: {grad_output[0].shape}")
        print(f"    grad_output 值: {grad_output[0].data.flatten()[:5]}...")
        return None  # 不修改

    handle = model.linear1.register_full_backward_pre_hook(backward_pre_hook)

    output = model(x)
    loss = output.sum()
    print()
    loss.backward()

    handle.remove()
    print()


def demo_modify_gradient():
    """演示用 backward hook 修改梯度"""
    print("=" * 60)
    print("Demo 3: Backward Hook — 修改梯度")
    print("=" * 60)

    model = SimpleNet()
    x = torch.randn(1, 4, requires_grad=True)

    # === 无 hook 的梯度 ===
    output = model(x)
    loss = output.sum()
    loss.backward()
    grad_no_hook = x.grad.clone()
    print(f"\n无 Hook 时 x.grad: {grad_no_hook.data}")

    # === 添加 hook: 将所有梯度替换为 42 ===
    def set_grad_to_42(module, grad_input, grad_output):
        """将梯度全部设置为 42"""
        new_grad_input = tuple(
            torch.ones_like(gi) * 42.0 if gi is not None else None
            for gi in grad_input
        )
        return new_grad_input

    handle = model.linear1.register_full_backward_hook(set_grad_to_42)

    # 重新前向 + 反向
    model.zero_grad()
    x.grad = None
    output = model(x)
    loss = output.sum()
    loss.backward()
    grad_with_hook = x.grad.clone()
    print(f"有 Hook 时 x.grad: {grad_with_hook.data}")
    print(f"\n梯度被修改为 42: {torch.all(grad_with_hook == 42.0).item()}")

    handle.remove()
    print()


def demo_gradient_statistics():
    """演示用 backward hook 收集每层梯度的统计信息"""
    print("=" * 60)
    print("Demo 4: Backward Hook — 收集梯度统计信息")
    print("=" * 60)

    # 构建一个稍复杂的网络
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    grad_stats = {}

    def collect_grad_stats(name):
        def hook(module, grad_input, grad_output):
            stats = {}
            for i, go in enumerate(grad_output):
                if go is not None:
                    stats[f'grad_output_{i}'] = {
                        'mean': go.mean().item(),
                        'std': go.std().item(),
                        'max': go.max().item(),
                        'min': go.min().item(),
                    }
            grad_stats[name] = stats
        return hook

    # 注册到所有 Linear 层
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            h = module.register_full_backward_hook(collect_grad_stats(name))
            handles.append(h)

    # 前向 + 反向
    x = torch.randn(8, 10)
    output = model(x)
    loss = output.sum()
    loss.backward()

    # 打印统计信息
    print(f"\n{'层名':>5s} | {'均值':>10s} | {'标准差':>10s} | {'最大值':>10s} | {'最小值':>10s}")
    print("-" * 60)
    for name, stats in grad_stats.items():
        for key, values in stats.items():
            print(f"{name:>5s} | {values['mean']:>10.6f} | {values['std']:>10.6f} | "
                  f"{values['max']:>10.6f} | {values['min']:>10.6f}")

    for h in handles:
        h.remove()
    print()


def demo_gradient_flow_check():
    """演示用 backward hook 检测梯度消失/爆炸"""
    print("=" * 60)
    print("Demo 5: Backward Hook — 检测梯度消失/爆炸")
    print("=" * 60)

    # 构建一个深层网络 (容易出现梯度消失)
    layers = []
    for _ in range(10):
        layers.append(nn.Linear(16, 16))
        layers.append(nn.Sigmoid())  # Sigmoid 容易导致梯度消失
    layers.append(nn.Linear(16, 1))
    model = nn.Sequential(*layers)

    gradient_norms = {}

    def monitor_gradient(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradient_norms[name] = grad_output[0].norm().item()
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            h = module.register_full_backward_hook(monitor_gradient(name))
            handles.append(h)

    # 前向 + 反向
    x = torch.randn(4, 16)
    output = model(x)
    loss = output.sum()
    loss.backward()

    # 按层顺序打印梯度范数
    print(f"\n{'层名':>5s} | {'梯度范数':>15s} | 状态")
    print("-" * 50)
    for name in sorted(gradient_norms.keys(), key=int):
        norm = gradient_norms[name]
        if norm < 1e-6:
            status = "[警告] 梯度消失"
        elif norm > 100:
            status = "[警告] 梯度爆炸"
        else:
            status = "正常"
        print(f"{name:>5s} | {norm:>15.8f} | {status}")

    for h in handles:
        h.remove()
    print()


if __name__ == '__main__':
    torch.manual_seed(42)
    demo_basic_backward_hook()
    demo_backward_pre_hook()
    demo_modify_gradient()
    demo_gradient_statistics()
    demo_gradient_flow_check()
