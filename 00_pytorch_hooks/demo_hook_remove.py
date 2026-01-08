"""
Hook 的注册、移除与批量管理

演示 Hook 生命周期管理的各种模式：
1. 基本注册与移除 (handle.remove())
2. 批量注册与移除
3. 上下文管理器模式 (推荐)
4. Hook 的执行顺序
5. 内存泄漏防范

参考: https://pytorch.org/docs/stable/notes/modules.html#module-hooks
"""

import torch
import torch.nn as nn
import contextlib


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


def demo_basic_remove():
    """演示基本的 hook 注册和移除"""
    print("=" * 60)
    print("Demo 1: 基本的 Hook 注册与移除")
    print("=" * 60)

    model = SimpleNet()
    x = torch.randn(1, 4)
    call_count = [0]

    def counting_hook(module, args, output):
        call_count[0] += 1

    # 注册 hook
    handle = model.linear1.register_forward_hook(counting_hook)
    print(f"\n  Handle 类型: {type(handle).__name__}")

    # 第一次前向: hook 应该被调用
    call_count[0] = 0
    model(x)
    print(f"  注册后前向传播: hook 调用 {call_count[0]} 次")

    # 移除 hook
    handle.remove()

    # 第二次前向: hook 不应该被调用
    call_count[0] = 0
    model(x)
    print(f"  移除后前向传播: hook 调用 {call_count[0]} 次")

    # 重复 remove 不会报错
    handle.remove()  # 安全的，不会抛异常
    print("  重复 remove: 不会报错 (安全操作)")
    print()


def demo_batch_register_remove():
    """演示批量注册和移除 hook"""
    print("=" * 60)
    print("Demo 2: 批量注册与移除")
    print("=" * 60)

    model = SimpleNet()
    x = torch.randn(1, 4)

    layer_outputs = {}

    def make_hook(name):
        def hook(module, args, output):
            layer_outputs[name] = output.detach().shape
        return hook

    # 批量注册
    handles = []
    for name, module in model.named_modules():
        if name:  # 跳过根模块
            h = module.register_forward_hook(make_hook(name))
            handles.append(h)

    print(f"\n  注册了 {len(handles)} 个 hook")

    # 前向传播
    model(x)
    print(f"  捕获到 {len(layer_outputs)} 个层的输出:")
    for name, shape in layer_outputs.items():
        print(f"    {name}: {shape}")

    # 批量移除
    for h in handles:
        h.remove()
    handles.clear()

    print("  已移除所有 hook")

    # 验证 hook 已移除
    layer_outputs.clear()
    model(x)
    print(f"  移除后捕获到: {len(layer_outputs)} 个输出 (应为 0)")
    print()


def demo_context_manager():
    """演示使用上下文管理器管理 hook (推荐模式)"""
    print("=" * 60)
    print("Demo 3: 上下文管理器模式 (推荐)")
    print("=" * 60)

    # 方法 1: 自定义上下文管理器
    @contextlib.contextmanager
    def hook_scope(module, hook_fn, hook_type='forward'):
        """上下文管理器: 自动注册和移除 hook"""
        if hook_type == 'forward':
            handle = module.register_forward_hook(hook_fn)
        elif hook_type == 'forward_pre':
            handle = module.register_forward_pre_hook(hook_fn)
        elif hook_type == 'backward':
            handle = module.register_full_backward_hook(hook_fn)
        else:
            raise ValueError(f"未知的 hook 类型: {hook_type}")
        try:
            yield handle
        finally:
            handle.remove()

    model = SimpleNet()
    x = torch.randn(1, 4)
    captured = {}

    def capture_output(module, args, output):
        captured['output'] = output.detach()

    # 使用上下文管理器: 自动管理 hook 生命周期
    print("\n  使用 hook_scope 上下文管理器:")
    with hook_scope(model.linear1, capture_output):
        model(x)
        if 'output' in captured:
            print(f"    with 块内: 捕获到输出 shape={captured['output'].shape}")

    captured.clear()
    model(x)
    print(f"    with 块外: 捕获到输出数量={len(captured)} (hook 已自动移除)")

    # 方法 2: 批量上下文管理器
    @contextlib.contextmanager
    def multi_hook_scope(model, hook_fn, layer_filter=None):
        """为模型中所有匹配的层注册 hook"""
        handles = []
        for name, module in model.named_modules():
            if layer_filter is None or layer_filter(name, module):
                h = module.register_forward_hook(hook_fn)
                handles.append(h)
        try:
            yield handles
        finally:
            for h in handles:
                h.remove()

    all_outputs = {}

    def capture_all(module, args, output):
        all_outputs[module.__class__.__name__] = output.detach().shape

    print("\n  使用 multi_hook_scope 批量管理:")
    with multi_hook_scope(model, capture_all,
                          layer_filter=lambda n, m: isinstance(m, nn.Linear)):
        model(x)
        print(f"    捕获到 {len(all_outputs)} 个 Linear 层的输出")

    all_outputs.clear()
    model(x)
    print(f"    退出后捕获到: {len(all_outputs)} 个 (hook 已自动移除)")
    print()


def demo_hook_execution_order():
    """演示多个 hook 的执行顺序"""
    print("=" * 60)
    print("Demo 4: Hook 执行顺序")
    print("=" * 60)

    model = SimpleNet()
    x = torch.randn(1, 4)
    order_log = []

    def make_ordered_hook(name):
        def hook(module, args, output):
            order_log.append(name)
        return hook

    # 注册多个 hook 到同一个层
    h1 = model.linear1.register_forward_hook(make_ordered_hook("hook_A"))
    h2 = model.linear1.register_forward_hook(make_ordered_hook("hook_B"))
    h3 = model.linear1.register_forward_hook(make_ordered_hook("hook_C"))

    model(x)
    print(f"\n  按注册顺序执行: {' -> '.join(order_log)}")

    # 使用 prepend=True 插入到最前面
    order_log.clear()
    h4 = model.linear1.register_forward_hook(
        make_ordered_hook("hook_D (prepend)"), prepend=True
    )

    model(x)
    print(f"  加入 prepend hook: {' -> '.join(order_log)}")

    h1.remove()
    h2.remove()
    h3.remove()
    h4.remove()

    # forward pre hook 和 forward hook 的相对顺序
    order_log.clear()

    def pre_hook(module, args):
        order_log.append("pre_hook")

    def post_hook(module, args, output):
        order_log.append("post_hook")

    hp = model.linear1.register_forward_pre_hook(pre_hook)
    hf = model.linear1.register_forward_hook(post_hook)

    model(x)
    print(f"  Pre + Post Hook: {' -> '.join(order_log)}")

    hp.remove()
    hf.remove()
    print()


def demo_memory_safety():
    """演示 Hook 中的内存安全注意事项"""
    print("=" * 60)
    print("Demo 5: Hook 内存安全 — detach() 的重要性")
    print("=" * 60)

    model = SimpleNet()

    # === 不安全方式: 存储带计算图的 tensor ===
    unsafe_store = []

    def unsafe_hook(module, args, output):
        unsafe_store.append(output)  # 注意: 没有 detach()!

    # === 安全方式: 使用 detach() ===
    safe_store = []

    def safe_hook(module, args, output):
        safe_store.append(output.detach())  # 使用 detach() 切断计算图

    print("\n  不安全方式 (不用 detach):")
    h = model.linear1.register_forward_hook(unsafe_hook)
    x = torch.randn(1, 4, requires_grad=True)
    _ = model(x)
    print(f"    存储的 tensor requires_grad: {unsafe_store[0].requires_grad}")
    print(f"    存储的 tensor 有 grad_fn: {unsafe_store[0].grad_fn is not None}")
    print("    -> 持有计算图引用，可能导致内存无法释放!")
    h.remove()

    print("\n  安全方式 (用 detach):")
    h = model.linear1.register_forward_hook(safe_hook)
    _ = model(x)
    print(f"    存储的 tensor requires_grad: {safe_store[0].requires_grad}")
    print(f"    存储的 tensor 有 grad_fn: {safe_store[0].grad_fn is not None}")
    print("    -> 已切断计算图引用，内存安全!")
    h.remove()

    # === 最佳实践: detach + clone ===
    best_store = []

    def best_hook(module, args, output):
        best_store.append(output.detach().clone())  # detach + clone 最安全

    h = model.linear1.register_forward_hook(best_hook)
    output = model(x)
    print("\n  最佳实践 (detach + clone):")
    print(f"    与原始 tensor 共享内存: {best_store[0].data_ptr() == output.data_ptr()}")
    print("    -> detach().clone() 创建完全独立的副本")
    h.remove()
    print()


def demo_complete_workflow():
    """演示完整的 Hook 工作流"""
    print("=" * 60)
    print("Demo 6: 完整工作流 — 训练中使用 Hook 监控")
    print("=" * 60)

    torch.manual_seed(42)
    model = SimpleNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 监控字典
    monitor = {
        'forward_activations': {},
        'backward_gradients': {},
    }

    @contextlib.contextmanager
    def training_hooks(model, monitor):
        """训练监控 hook 的上下文管理器"""
        handles = []

        def fwd_hook(name):
            def hook(module, args, output):
                monitor['forward_activations'][name] = {
                    'mean': output.detach().mean().item(),
                    'std': output.detach().std().item(),
                }
            return hook

        def bwd_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    monitor['backward_gradients'][name] = {
                        'norm': grad_output[0].norm().item(),
                    }
            return hook

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(fwd_hook(name)))
                handles.append(module.register_full_backward_hook(bwd_hook(name)))

        try:
            yield
        finally:
            for h in handles:
                h.remove()

    # 训练循环
    print()
    with training_hooks(model, monitor):
        for epoch in range(3):
            x = torch.randn(4, 4)
            y = torch.randn(4, 1)

            output = model(x)
            loss = nn.MSELoss()(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}")
            print("    激活值: ", end="")
            for name, stats in monitor['forward_activations'].items():
                print(f"{name}(mean={stats['mean']:.3f}) ", end="")
            print()
            print("    梯度:   ", end="")
            for name, stats in monitor['backward_gradients'].items():
                print(f"{name}(norm={stats['norm']:.3f}) ", end="")
            print()

    # 验证 hook 已被清理
    print("\n  退出上下文后, hook 已自动清理")
    print()


if __name__ == '__main__':
    demo_basic_remove()
    demo_batch_register_remove()
    demo_context_manager()
    demo_hook_execution_order()
    demo_memory_safety()
    demo_complete_workflow()
