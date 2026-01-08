"""
梯度裁剪/修改实战

演示使用 Hook 实现各种梯度操作：
1. 逐层梯度裁剪 (per-layer gradient clipping)
2. 梯度缩放 (gradient scaling)
3. 梯度监控与报警
4. 选择性冻结层的梯度

参考: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
"""

import torch
import torch.nn as nn


# ============================================================
# 构建测试网络
# ============================================================
class DeepNet(nn.Module):
    """一个较深的网络，用于演示梯度相关操作"""

    def __init__(self, num_layers=5, hidden_size=32):
        super().__init__()
        layers = [nn.Linear(16, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Demo 函数
# ============================================================
def demo_per_layer_gradient_clipping():
    """演示逐层梯度裁剪"""
    print("=" * 60)
    print("Demo 1: 逐层梯度裁剪 (Backward Hook)")
    print("=" * 60)

    torch.manual_seed(42)
    model = DeepNet(num_layers=5)
    x = torch.randn(4, 16)

    max_norm = 1.0
    clip_log = {}

    def gradient_clip_hook(name, clip_value):
        """创建梯度裁剪 hook（clip_value 避免与外层 max_norm 同名遮蔽）"""
        def hook(module, grad_input, grad_output):
            clipped_grads = []
            for gi in grad_input:
                if gi is not None:
                    norm_before = gi.norm().item()
                    clipped = torch.clamp(gi, -clip_value, clip_value)
                    norm_after = clipped.norm().item()
                    clip_log[name] = {
                        'before': norm_before,
                        'after': norm_after,
                        'clipped': norm_before > clip_value,
                    }
                    clipped_grads.append(clipped)
                else:
                    clipped_grads.append(None)
            return tuple(clipped_grads)
        return hook

    # 为所有 Linear 层注册梯度裁剪 hook
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            h = module.register_full_backward_hook(gradient_clip_hook(name, max_norm))
            handles.append(h)

    # 前向 + 反向
    output = model(x)
    loss = output.sum()
    loss.backward()

    print(f"\n  梯度裁剪阈值: {max_norm}")
    print(f"\n  {'层名':>15s} | {'裁剪前范数':>12s} | {'裁剪后范数':>12s} | 是否裁剪")
    print("  " + "-" * 60)
    for name, info in clip_log.items():
        status = "是" if info['clipped'] else "否"
        print(f"  {name:>15s} | {info['before']:>12.6f} | "
              f"{info['after']:>12.6f} | {status}")

    for h in handles:
        h.remove()
    print()


def demo_gradient_scaling_per_layer():
    """演示逐层梯度缩放 — 模拟不同层使用不同学习率"""
    print("=" * 60)
    print("Demo 2: 逐层梯度缩放 (模拟差异化学习率)")
    print("=" * 60)

    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(8, 16),   # 浅层: 小学习率 -> 梯度缩小
        nn.ReLU(),
        nn.Linear(16, 16),  # 中间层: 正常学习率
        nn.ReLU(),
        nn.Linear(16, 1),   # 深层: 大学习率 -> 梯度放大
    )

    # 为不同深度的层设置不同缩放系数
    layer_scales = {
        '0': 0.1,   # 浅层梯度缩小 10 倍
        '2': 1.0,   # 中间层不变
        '4': 5.0,   # 深层梯度放大 5 倍
    }

    def scale_gradient_hook(scale):
        def hook(module, grad_input, grad_output):
            return tuple(
                gi * scale if gi is not None else None
                for gi in grad_input
            )
        return hook

    handles = []
    for name, module in model.named_modules():
        if name in layer_scales:
            h = module.register_full_backward_hook(
                scale_gradient_hook(layer_scales[name])
            )
            handles.append(h)

    # 前向 + 反向
    x = torch.randn(4, 8)
    output = model(x)
    loss = output.sum()
    loss.backward()

    print(f"\n  {'层名':>6s} | {'缩放系数':>8s} | {'权重梯度范数':>15s}")
    print("  " + "-" * 40)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            grad_norm = module.weight.grad.norm().item()
            scale = layer_scales.get(name, '-')
            print(f"  {name:>6s} | {str(scale):>8s} | {grad_norm:>15.6f}")

    for h in handles:
        h.remove()
    print()


def demo_gradient_monitor():
    """演示梯度监控 — 检测梯度异常"""
    print("=" * 60)
    print("Demo 3: 梯度监控与异常检测")
    print("=" * 60)

    torch.manual_seed(42)

    # 构建一个可能出现梯度问题的网络
    model = nn.Sequential(
        nn.Linear(8, 32),
        nn.Sigmoid(),         # Sigmoid 容易导致梯度消失
        nn.Linear(32, 32),
        nn.Sigmoid(),
        nn.Linear(32, 32),
        nn.Sigmoid(),
        nn.Linear(32, 1),
    )

    warnings_log = []

    def gradient_monitor_hook(name, vanish_threshold=1e-7, explode_threshold=100):
        """梯度监控 hook: 检测梯度消失和爆炸"""
        def hook(module, grad_input, grad_output):
            for i, go in enumerate(grad_output):
                if go is None:
                    continue
                norm = go.norm().item()
                mean_abs = go.abs().mean().item()

                if norm < vanish_threshold:
                    msg = f"[警告] {name} grad_output[{i}] 梯度消失! norm={norm:.2e}"
                    warnings_log.append(msg)
                elif norm > explode_threshold:
                    msg = f"[警告] {name} grad_output[{i}] 梯度爆炸! norm={norm:.2e}"
                    warnings_log.append(msg)

                # 检查 NaN
                if torch.isnan(go).any():
                    msg = f"[严重] {name} grad_output[{i}] 包含 NaN!"
                    warnings_log.append(msg)

        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Sigmoid)):
            h = module.register_full_backward_hook(gradient_monitor_hook(name))
            handles.append(h)

    # 训练几步
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    print()

    for step in range(3):
        warnings_log.clear()
        x = torch.randn(4, 8)
        output = model(x)
        loss = output.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if warnings_log:
            print(f"  Step {step}: 发现 {len(warnings_log)} 个梯度警告")
            for w in warnings_log[:3]:  # 只显示前3个
                print(f"    {w}")
        else:
            print(f"  Step {step}: 梯度正常")

    for h in handles:
        h.remove()
    print()


def demo_selective_freeze():
    """演示使用 Tensor Hook 选择性冻结层"""
    print("=" * 60)
    print("Demo 4: 选择性冻结层 (Tensor Hook)")
    print("=" * 60)

    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
    )

    # 记录初始参数
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.data.clone()

    # 使用 tensor hook 冻结前两层的梯度 (将梯度置零)
    freeze_handles = []
    for name, param in model.named_parameters():
        if name.startswith('0.') or name.startswith('2.'):
            h = param.register_hook(lambda grad: torch.zeros_like(grad))
            freeze_handles.append(h)

    # 训练一步
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(4, 4)
    output = model(x)
    loss = output.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 检查哪些参数发生了变化
    print(f"\n  {'参数名':>12s} | {'参数变化范数':>12s} | 状态")
    print("  " + "-" * 45)
    for name, param in model.named_parameters():
        change = (param.data - initial_params[name]).norm().item()
        frozen = name.startswith('0.') or name.startswith('2.')
        status = "冻结 (不变)" if frozen else "训练 (已更新)"
        print(f"  {name:>12s} | {change:>12.6f} | {status}")

    for h in freeze_handles:
        h.remove()
    print()


def demo_gradient_accumulation_with_hook():
    """演示使用 Hook 实现自定义梯度累积"""
    print("=" * 60)
    print("Demo 5: 自定义梯度累积 (Forward + Backward Hook)")
    print("=" * 60)

    torch.manual_seed(42)
    model = nn.Linear(4, 2)

    # 追踪前向和反向的信息
    forward_count = [0]
    backward_count = [0]
    accumulated_grads = {}

    def forward_counter(module, args, output):
        forward_count[0] += 1

    def backward_accumulator(module, grad_input, grad_output):
        backward_count[0] += 1
        key = 'grad_output_sum'   # 累积的是 grad_output，而非权重梯度
        go = grad_output[0]
        if key not in accumulated_grads:
            accumulated_grads[key] = go.clone()
        else:
            accumulated_grads[key] += go

    h1 = model.register_forward_hook(forward_counter)
    h2 = model.register_full_backward_hook(backward_accumulator)

    # 模拟梯度累积 (accumulation_steps=4)
    accumulation_steps = 4
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    print()
    for step in range(accumulation_steps):
        x = torch.randn(2, 4)
        output = model(x)
        loss = output.sum() / accumulation_steps  # 平均 loss
        loss.backward()

    print(f"  前向传播次数: {forward_count[0]}")
    print(f"  反向传播次数: {backward_count[0]}")
    print(f"  累积的 grad_output 范数: {accumulated_grads['grad_output_sum'].norm():.4f}")
    print(f"  权重梯度范数: {model.weight.grad.norm():.4f}")

    optimizer.step()
    optimizer.zero_grad()

    h1.remove()
    h2.remove()
    print()


if __name__ == '__main__':
    demo_per_layer_gradient_clipping()
    demo_gradient_scaling_per_layer()
    demo_gradient_monitor()
    demo_selective_freeze()
    demo_gradient_accumulation_with_hook()
