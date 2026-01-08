"""
Forward Hook 演示

演示 PyTorch 中 Module 的前向 Hook 机制：
1. register_forward_pre_hook: 在 forward() 之前执行，可修改输入
2. register_forward_hook: 在 forward() 之后执行，可修改输出

参考: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
"""

import torch
import torch.nn as nn


# ============================================================
# 定义一个简单的网络用于演示
# ============================================================
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 3)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(3, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def demo_basic_forward_hook():
    """演示基本的 forward hook — 查看每层的输入和输出"""
    print("=" * 60)
    print("Demo 1: 基本 Forward Hook — 查看每层输入输出 shape")
    print("=" * 60)

    model = SimpleNet()
    x = torch.randn(2, 4)  # batch_size=2, features=4

    # 定义 hook 函数
    def forward_hook(module, args, output):
        """在 forward() 之后调用"""
        input_tensor = args[0]
        print(f"  [{module.__class__.__name__}] "
              f"输入 shape: {input_tensor.shape} -> 输出 shape: {output.shape}")

    # 为每一层注册 hook
    handles = []
    for name, module in model.named_modules():
        if name:  # 跳过根模块
            h = module.register_forward_hook(forward_hook)
            handles.append(h)

    # 前向传播
    print(f"\n输入 shape: {x.shape}")
    output = model(x)
    print(f"最终输出 shape: {output.shape}")

    # 清理
    for h in handles:
        h.remove()
    print()


def demo_forward_pre_hook():
    """演示 forward pre hook — 在 forward 之前修改输入"""
    print("=" * 60)
    print("Demo 2: Forward Pre Hook — 修改输入")
    print("=" * 60)

    model = SimpleNet()
    x = torch.ones(1, 4)  # 全 1 输入

    # 不加 hook 的输出
    output_no_hook = model(x)
    print(f"\n无 Hook 时输出: {output_no_hook.data}")

    # 定义 pre hook: 将输入加 1
    def add_one_pre_hook(module, args):
        """在 forward() 之前将输入 +1"""
        modified_input = args[0] + 1.0
        print(f"  [Pre Hook] 原始输入均值: {args[0].mean():.2f} "
              f"-> 修改后输入均值: {modified_input.mean():.2f}")
        return (modified_input,)  # 注意返回 tuple

    handle = model.linear1.register_forward_pre_hook(add_one_pre_hook)

    # 有 hook 的输出（输入被修改为全 2）
    output_with_hook = model(x)
    print(f"有 Pre Hook 时输出: {output_with_hook.data}")
    print(f"输出发生了变化: {not torch.allclose(output_no_hook, output_with_hook)}")

    handle.remove()
    print()


def demo_modify_output_hook():
    """演示用 forward hook 修改输出 — 实现类似 ResNet 的残差连接"""
    print("=" * 60)
    print("Demo 3: Forward Hook — 修改输出（残差连接）")
    print("=" * 60)

    model = SimpleNet()

    # 用 hook 给 linear1 层加一个 residual 连接
    def residual_hook(module, args, output):
        """将输入加到输出上 (类似残差连接)"""
        # 注意: 这里简化处理，实际中需要维度匹配
        input_tensor = args[0]
        # 截取匹配的维度
        residual = input_tensor[:, :output.shape[1]]
        modified_output = output + residual
        print(f"  [Residual Hook] output + input[:, :{output.shape[1]}]")
        return modified_output

    x = torch.randn(1, 4)

    print("\n无 Hook 时:")
    out1 = model(x)
    print(f"  输出: {out1.data}")

    handle = model.linear1.register_forward_hook(residual_hook)

    print("有残差 Hook 时:")
    out2 = model(x)
    print(f"  输出: {out2.data}")

    handle.remove()
    print()


def demo_save_activations():
    """演示用 forward hook 保存中间层激活值"""
    print("=" * 60)
    print("Demo 4: Forward Hook — 保存中间层激活值")
    print("=" * 60)

    model = SimpleNet()
    activations = {}

    def get_activation(name):
        """返回一个 hook 函数，将输出保存到 activations 字典"""
        def hook(module, args, output):
            activations[name] = output.detach().clone()
        return hook

    # 注册到感兴趣的层
    h1 = model.linear1.register_forward_hook(get_activation('linear1'))
    h2 = model.relu.register_forward_hook(get_activation('relu'))
    h3 = model.linear2.register_forward_hook(get_activation('linear2'))

    # 前向传播
    x = torch.randn(2, 4)
    output = model(x)

    # 查看保存的激活值
    print()
    for name, activation in activations.items():
        print(f"  {name:10s} -> shape: {activation.shape}, "
              f"均值: {activation.mean():.4f}, 标准差: {activation.std():.4f}")

    # 验证: 最后一层的激活值应该等于模型输出
    assert torch.allclose(activations['linear2'], output), "最后一层激活应等于输出"
    print("\n  ✓ 验证通过: activations['linear2'] == model output")

    h1.remove()
    h2.remove()
    h3.remove()
    print()


def demo_forward_hook_vs_forward():
    """演示: model(x) 触发 Hook, model.forward(x) 不触发"""
    print("=" * 60)
    print("Demo 5: model(x) vs model.forward(x) — Hook 触发区别")
    print("=" * 60)

    model = SimpleNet()
    hook_called = {"count": 0}

    def counting_hook(module, args, output):
        hook_called["count"] += 1

    handle = model.register_forward_hook(counting_hook)
    x = torch.randn(1, 4)

    # 方式 1: model(x) — 会触发 hook
    hook_called["count"] = 0
    _ = model(x)
    print(f"\n  model(x)         -> Hook 被调用 {hook_called['count']} 次")

    # 方式 2: model.forward(x) — 不会触发 hook
    hook_called["count"] = 0
    _ = model.forward(x)
    print(f"  model.forward(x) -> Hook 被调用 {hook_called['count']} 次")

    print("\n  结论: 始终使用 model(x) 而非 model.forward(x) 以确保 Hook 被触发")

    handle.remove()
    print()


if __name__ == '__main__':
    torch.manual_seed(42)   # 固定随机种子，保证输出可复现（与同目录其他 demo 一致）
    demo_basic_forward_hook()
    demo_forward_pre_hook()
    demo_modify_output_hook()
    demo_save_activations()
    demo_forward_hook_vs_forward()
