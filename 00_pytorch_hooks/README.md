# PyTorch Hook 机制指南

> 本文系统介绍 PyTorch 的 Hook（钩子）机制：从触发原理出发，依次讲解 Module 级、Tensor 级、全局与 State Dict 四类 Hook 的用法与函数签名，再到 Hook 的生命周期管理、五类典型应用场景，最后总结常见陷阱与最佳实践。全文配有 6 个可运行的演示脚本，各章节与脚本一一对应。
>
> 参考官方文档：
> - https://pytorch.org/docs/stable/notes/modules.html#module-hooks
> - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
> - https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html

---

## 目录

- [第 1 章 概述：什么是 Hook](#第-1-章-概述什么是-hook)
- [第 2 章 触发原理](#第-2-章-触发原理)
- [第 3 章 Hook 全景分类](#第-3-章-hook-全景分类)
- [第 4 章 Module Forward Hook](#第-4-章-module-forward-hook)
- [第 5 章 Module Backward Hook](#第-5-章-module-backward-hook)
- [第 6 章 Tensor Hook](#第-6-章-tensor-hook)
- [第 7 章 State Dict Hook](#第-7-章-state-dict-hook)
- [第 8 章 Hook 的生命周期管理](#第-8-章-hook-的生命周期管理)
- [第 9 章 典型应用场景](#第-9-章-典型应用场景)
- [第 10 章 陷阱与最佳实践](#第-10-章-陷阱与最佳实践)
- [第 11 章 配套演示脚本](#第-11-章-配套演示脚本)

---

## 第 1 章 概述：什么是 Hook

Hook 是 PyTorch 提供的一种**不修改模型源码即可介入前向/反向传播过程**的机制：在计算流程的特定位置"挂上"一段自定义代码——数据流经该位置时，Hook 函数被自动调用，可以**观察**乃至**修改**输入/输出/梯度，随后数据继续正常流动。

**为什么需要 Hook？** 实际开发中的常见需求：

- 查看中间层的输出（特征可视化、感知损失）；
- 检查或修改梯度（梯度裁剪、梯度调试、选择性冻结）;
- 统计每一层的计算量（FLOPs 分析）；
- 在不改动第三方预训练模型代码的前提下提取特征。

没有 Hook，这些需求只能通过改写模型的 `forward()` 实现——对第三方模型极不方便。Hook 将"观察/干预逻辑"与"模型定义"彻底解耦。

**全文路线图**：

```
触发原理（第 2 章：Hook 挂在 __call__ 上，而非 forward 上）
    └─> 全景分类（第 3 章：Module / Tensor / 全局 / State Dict 四类）
            └─> 各类 Hook 详解（第 4~7 章：签名、返回值语义、触发条件）
                    └─> 生命周期管理（第 8 章：handle / 批量 / 上下文管理器）
                            └─> 应用场景（第 9 章）与最佳实践（第 10 章）
```

---

## 第 2 章 触发原理

### 2.1 前向传播中的 Hook

调用 `model(input)` 时，PyTorch 内部的执行流程：

```
model.__call__(input)
  │
  ├── 1. 执行所有 forward_pre_hook(module, input)
  │      → 可以修改 input
  │
  ├── 2. 执行 module.forward(input)
  │      → 得到 output
  │
  └── 3. 执行所有 forward_hook(module, input, output)
         → 可以修改 output
```

**关键结论：Hook 的管理逻辑在 `Module.__call__` 中，而不在 `forward` 中。** 因此：

```python
model(x)            # 触发 Hook ✓
model.forward(x)    # 不会触发 Hook！
```

这是 Hook 最容易踩的坑（第 10 章陷阱 1），也是"永远通过 `model(x)` 调用模型"这条通用规范的原因之一。

### 2.2 反向传播中的 Hook

调用 `loss.backward()` 时，对每个注册了 backward hook 的模块：

```
loss.backward()
  │
  ├── 计算 grad_output（该模块输出端的梯度）
  │
  ├── 1. 执行所有 full_backward_pre_hook(module, grad_output)
  │      → 可以修改 grad_output
  │
  ├── 2. 计算 grad_input（该模块输入端的梯度）
  │
  └── 3. 执行所有 full_backward_hook(module, grad_input, grad_output)
         → 可以修改 grad_input
```

### 2.3 Tensor 级别 Hook

比 Module 更细的粒度——挂在单个 Tensor 上：

```
loss.backward()
  │
  └── 对于每个 requires_grad=True 的 Tensor:
      ├── 计算该 Tensor 的梯度 grad
      └── 执行 tensor.register_hook 注册的所有 hook(grad)
          → 可以返回新的梯度替换原梯度
```

---

## 第 3 章 Hook 全景分类

PyTorch 的 Hook 按作用对象分为四大类。后续第 4~7 章逐类展开。

### 3.1 Module 级别 Hook

| Hook 类型 | 注册方法 | 触发时机 | 函数签名 |
|-----------|---------|---------|---------|
| Forward Pre Hook | `register_forward_pre_hook(hook)` | forward() 之前 | `hook(module, args) -> None or modified_args` |
| Forward Hook | `register_forward_hook(hook)` | forward() 之后 | `hook(module, args, output) -> None or modified_output` |
| Backward Pre Hook | `register_full_backward_pre_hook(hook)` | backward 计算梯度之前 | `hook(module, grad_output) -> None or modified_grad_output` |
| Backward Hook | `register_full_backward_hook(hook)` | backward 计算梯度之后 | `hook(module, grad_input, grad_output) -> None or modified_grad_input` |

### 3.2 Tensor 级别 Hook

| Hook 类型 | 注册方法 | 触发时机 | 函数签名 |
|-----------|---------|---------|---------|
| Tensor Backward Hook | `tensor.register_hook(hook)` | 计算该 tensor 梯度时 | `hook(grad) -> Tensor or None` |

### 3.3 全局 Hook

对**所有** Module 生效，无需逐个注册：

| Hook 类型 | 注册方法 |
|-----------|---------|
| Global Forward Pre Hook | `torch.nn.modules.module.register_module_forward_pre_hook()` |
| Global Forward Hook | `torch.nn.modules.module.register_module_forward_hook()` |
| Global Backward Hook | `torch.nn.modules.module.register_module_full_backward_hook()` |

### 3.4 State Dict Hook

介入模型保存与加载流程：

| Hook 类型 | 注册方法 | 触发时机 |
|-----------|---------|---------|
| State Dict Pre Hook | `register_state_dict_pre_hook(hook)` | `state_dict()` 调用前 |
| State Dict Post Hook | `register_state_dict_post_hook(hook)` | `state_dict()` 调用后 |
| Load State Dict Pre Hook | `register_load_state_dict_pre_hook(hook)` | `load_state_dict()` 调用前 |
| Load State Dict Post Hook | `register_load_state_dict_post_hook(hook)` | `load_state_dict()` 调用后 |

---

## 第 4 章 Module Forward Hook

> 配套脚本：[`demo_forward_hook.py`](./demo_forward_hook.py)

### 4.1 Forward Pre Hook

在 `forward()` 执行**之前**调用，用于检查或修改输入：

```python
def forward_pre_hook(module, args):
    """
    Args:
        module: 当前模块
        args: forward() 的输入参数 (tuple)
    Returns:
        None 或修改后的 input (tuple)
    """
    print(f"[Pre Hook] {module.__class__.__name__} 输入 shape: {args[0].shape}")
    # 返回 None 表示不修改输入
    return None

# 注册
handle = model.layer1.register_forward_pre_hook(forward_pre_hook)
```

注意输入以 **tuple** 形式传入；若要修改输入，返回值也必须是 tuple（哪怕只有一个元素，如 `return (modified_x,)`）。

### 4.2 Forward Hook

在 `forward()` 执行**之后**调用，用于检查或修改输出：

```python
def forward_hook(module, args, output):
    """
    Args:
        module: 当前模块
        args: forward() 的输入参数 (tuple)
        output: forward() 的输出 (as-is，不是 tuple)
    Returns:
        None 或修改后的 output
    """
    print(f"[Hook] {module.__class__.__name__} 输出 shape: {output.shape}")
    # 返回修改后的输出
    return output * 2  # 例如将输出翻倍

# 注册
handle = model.layer1.register_forward_hook(forward_hook)
```

利用"修改输出"的能力，甚至可以在不改模型代码的情况下外挂残差连接（`return output + args[0]`），配套脚本的 `demo_modify_output_hook` 演示了这一用法。

### 4.3 with_kwargs 参数

PyTorch 2.0 起，`with_kwargs=True` 可让 hook 接收关键字参数：

```python
def hook_with_kwargs(module, args, kwargs, output):
    print(f"kwargs: {kwargs}")
    return output

handle = model.register_forward_hook(hook_with_kwargs, with_kwargs=True)
```

---

## 第 5 章 Module Backward Hook

> 配套脚本：[`demo_backward_hook.py`](./demo_backward_hook.py)

### 5.1 Full Backward Pre Hook

在模块反向传播**开始前**调用，可修改输出端梯度：

```python
def backward_pre_hook(module, grad_output):
    """
    Args:
        module: 当前模块
        grad_output: 输出的梯度 (tuple of Tensor)
    Returns:
        None 或修改后的 grad_output
    """
    print(f"[Backward Pre Hook] grad_output shape: {grad_output[0].shape}")
    return None

handle = model.layer1.register_full_backward_pre_hook(backward_pre_hook)
```

### 5.2 Full Backward Hook

在模块反向传播**完成后**调用，可检查和修改输入端梯度：

```python
def backward_hook(module, grad_input, grad_output):
    """
    Args:
        module: 当前模块
        grad_input: 输入的梯度 (tuple of Tensor)
        grad_output: 输出的梯度 (tuple of Tensor)
    Returns:
        None 或修改后的 grad_input (tuple)
    """
    print(f"[Backward Hook] grad_input: {grad_input}")
    print(f"[Backward Hook] grad_output: {grad_output}")
    # 可以返回新的 grad_input
    return tuple(torch.ones_like(gi) * 42. for gi in grad_input if gi is not None)

handle = model.layer1.register_full_backward_hook(backward_hook)
```

> **弃用提示**：旧接口 `register_backward_hook()` 已被弃用（对多输入/多输出模块的梯度对应关系有缺陷），一律使用 `register_full_backward_hook()`。

### 5.3 触发条件

根据官方文档，backward hook 的触发规则：

1. **通常情况**：当计算出模块**输入**的梯度时触发；
2. **如果没有模块输入需要梯度**：当计算出模块**输出**的梯度时触发；
3. **如果没有模块输出需要梯度**：Hook 不会触发。

---

## 第 6 章 Tensor Hook

> 配套脚本：[`demo_tensor_hook.py`](./demo_tensor_hook.py)

`Tensor.register_hook()` 在该 Tensor 的梯度被计算出来时执行，返回非 None 值可**替换**原梯度：

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)

# 注册 hook: 将梯度翻倍
handle = x.register_hook(lambda grad: grad * 2)

y = (x * 3).sum()
y.backward()

print(x.grad)  # tensor([6., 6., 6.])  而不是 [3., 3., 3.]

handle.remove()  # 移除 hook
```

Tensor Hook 的一个独特价值：**捕获非叶子节点（中间变量）的梯度**。默认情况下反向传播后中间变量的 `.grad` 为 None（仅叶子节点保留梯度），用 `register_hook` 可以在梯度流经时截取它——这是调试复杂计算图的利器（配套脚本 `demo_gradient_debug` 用它逐项验证链式法则）。

**Tensor Hook vs Module Hook**：

| 特性 | Tensor Hook | Module Hook |
|------|-------------|-------------|
| 作用对象 | 单个 Tensor | 整个 Module |
| 触发时机 | 该 Tensor 梯度计算时 | Module 的 forward/backward |
| 可访问信息 | 仅该 Tensor 的梯度 | 输入/输出/梯度 |
| 典型用途 | 修改特定参数梯度、捕获中间变量梯度 | 特征提取/调试 |

---

## 第 7 章 State Dict Hook

用于在模型保存和加载时自定义行为（如过滤临时键、兼容旧版权重格式）：

```python
# 保存时的 hook
def state_dict_hook(module, state_dict, prefix, local_metadata):
    # 可以修改 state_dict，如移除某些键
    keys_to_remove = [k for k in state_dict if 'temporary' in k]
    for k in keys_to_remove:
        del state_dict[k]

model.register_state_dict_post_hook(state_dict_hook)

# 加载时的 hook
def load_state_dict_hook(module, incompatible_keys):
    # 处理不兼容的键
    incompatible_keys.missing_keys.clear()

model.register_load_state_dict_post_hook(load_state_dict_hook)
```

> 实际案例：许多量化/并行框架（如 Transformers 的量化层替换、FSDP）正是通过 state dict hook 在保存/加载时做权重格式转换。

---

## 第 8 章 Hook 的生命周期管理

> 配套脚本：[`demo_hook_remove.py`](./demo_hook_remove.py)

### 8.1 注册与移除：RemovableHandle

所有 `register_*_hook()` 方法都返回一个 `RemovableHandle` 对象，`remove()` 即可移除（重复调用 `remove()` 是安全的）：

```python
handle = model.layer.register_forward_hook(my_hook)
# ... 使用 ...
handle.remove()
```

### 8.2 批量管理

```python
# 注册多个 hook 并统一管理
handles = []
for name, module in model.named_modules():
    h = module.register_forward_hook(my_hook)
    handles.append(h)

# 用完后批量移除
for h in handles:
    h.remove()
```

### 8.3 上下文管理器（推荐）

将"注册-使用-移除"绑定为一个作用域，杜绝忘记清理：

```python
import contextlib

@contextlib.contextmanager
def hook_context(module, hook_fn):
    handle = module.register_forward_hook(hook_fn)
    try:
        yield handle
    finally:
        handle.remove()

# 使用
with hook_context(model.layer1, my_hook):
    output = model(input)
# 退出 with 块后 hook 自动移除
```

### 8.4 执行顺序与 prepend

同一挂载点的多个 hook 按**注册顺序**执行；`prepend=True`（PyTorch 2.0+）可将新 hook 插到队首：

```python
module.register_forward_hook(hook_a)
module.register_forward_hook(hook_b)                  # 执行顺序: a -> b
module.register_forward_hook(hook_c, prepend=True)    # 执行顺序: c -> a -> b
```

---

## 第 9 章 典型应用场景

### 9.1 中间层特征提取

> 配套脚本：[`demo_feature_extraction.py`](./demo_feature_extraction.py)

经典的"闭包工厂 + 字典收集"模式，从任意（含第三方）模型提取中间特征：

```python
features = {}

def get_features(name):
    def hook(module, input, output):
        features[name] = output.detach()      # detach 防止内存泄漏（第 10 章）
    return hook

# 注册到感兴趣的层
model.layer1.register_forward_hook(get_features('layer1'))
model.layer2.register_forward_hook(get_features('layer2'))

# 前向传播后 features 字典中就有了中间层输出
output = model(input_tensor)
print(features['layer1'].shape)
```

延伸用法（脚本中均有演示）：统计各层激活分布（均值/标准差/零值比例）、比较不同输入的特征余弦相似度、提取多层特征计算感知损失（perceptual loss）。

### 9.2 梯度可视化与调试

```python
def print_grad(name):
    def hook(module, grad_input, grad_output):
        print(f"[{name}] grad_output: {grad_output[0].shape}")
        print(f"[{name}] grad_output mean: {grad_output[0].mean():.6f}")
    return hook

for name, module in model.named_modules():
    module.register_full_backward_hook(print_grad(name))
```

配套脚本 `demo_gradient_flow_check`（demo_backward_hook.py）用该模式在深层 Sigmoid 网络中逐层监测梯度范数，直观展示**梯度消失**的检测方法。

### 9.3 逐层梯度裁剪与梯度监控

> 配套脚本：[`demo_gradient_clipping.py`](./demo_gradient_clipping.py)

```python
def gradient_clipping_hook(module, grad_input, grad_output):
    return tuple(
        torch.clamp(gi, -1.0, 1.0) if gi is not None else None
        for gi in grad_input
    )

model.layer1.register_full_backward_hook(gradient_clipping_hook)
```

与全局的 `torch.nn.utils.clip_grad_norm_` 相比，Hook 方案可以做到**逐层不同阈值**、逐层缩放（差异化学习率）、NaN/爆炸检测告警、选择性冻结（梯度置零）——脚本中各有一个 demo。

### 9.4 修改前向传播输入（数据增强/扰动注入）

```python
def add_noise_hook(module, args):
    noisy_input = args[0] + torch.randn_like(args[0]) * 0.1
    return (noisy_input,)     # 注意：必须返回 tuple

model.layer1.register_forward_pre_hook(add_noise_hook)
```

### 9.5 计算 FLOPs / 参数量

```python
total_flops = 0

def count_linear_flops(module, input, output):
    global total_flops
    # Linear 层的 FLOPs ≈ 2 * in_features * out_features * batch_size
    batch_size = input[0].shape[0]
    total_flops += 2 * module.in_features * module.out_features * batch_size

for module in model.modules():
    if isinstance(module, nn.Linear):
        module.register_forward_hook(count_linear_flops)
```

这也是 `thop`、`fvcore.nn.FlopCountAnalysis` 等 FLOPs 统计工具的底层实现原理。

---

## 第 10 章 陷阱与最佳实践

### 10.1 四个常见陷阱

**陷阱 1：直接调用 `forward()` 不会触发 Hook**（原理见第 2.1 节）

```python
model.forward(x)   # Hook 不会被调用!
model(x)           # Hook 会被调用 ✓
```

**陷阱 2：Backward Hook 不能就地修改 grad_input/grad_output**

```python
# 错误! 不要就地修改
def bad_hook(module, grad_input, grad_output):
    grad_input[0].mul_(0.5)  # RuntimeError!

# 正确: 返回新的 tuple
def good_hook(module, grad_input, grad_output):
    return (grad_input[0] * 0.5,)
```

**陷阱 3：Hook 中引用的 Tensor 可能导致内存泄漏**

```python
# 不好: 存储了保持计算图的 tensor
saved = []
def leaky_hook(module, input, output):
    saved.append(output)  # output 持有计算图引用，整张图无法释放

# 好: 使用 detach()
def safe_hook(module, input, output):
    saved.append(output.detach())

# 最佳: detach + clone，得到完全独立的副本（不共享存储）
def best_hook(module, input, output):
    saved.append(output.detach().clone())
```

配套脚本 `demo_memory_safety`（demo_hook_remove.py）打印了三种方式下 tensor 的 `requires_grad` / `grad_fn` / `data_ptr` 对比，可直观验证差异。

**陷阱 4：用完 Hook 忘记移除**

- 残留的 Hook 会在之后每次前向/反向传播中执行，既拖慢速度又可能污染结果；
- 使用第 8.3 节的上下文管理器模式可以从机制上杜绝遗忘。

### 10.2 性能考虑

- Hook 的执行叠加在前向/反向传播的关键路径上，Hook 内的大量计算或 I/O 会显著拖慢训练；
- 生产环境中，训练/调试完成后应移除所有调试用 Hook；
- 特征收集使用 `output.detach()`，避免无谓的计算图保持。

### 10.3 最佳实践总结

| 实践 | 说明 |
|------|------|
| 使用 `model(x)` 而非 `model.forward(x)` | 确保 Hook 被触发 |
| 特征提取时使用 `.detach()`（必要时再 `.clone()`） | 避免内存泄漏 |
| 保存 handle 并及时 `remove()` | 避免残留 Hook 的性能损耗 |
| 使用上下文管理器管理 Hook | 自动注册和移除 |
| backward hook 返回新 tuple | 不要就地修改梯度 |
| 修改输入时返回 tuple | forward pre hook 的输入输出均为 tuple |
| 调试完及时清理 | 生产环境不留调试 Hook |

---

## 第 11 章 配套演示脚本

### 11.1 脚本索引

| 脚本 | 内容 | 对应章节 |
|------|------|---------|
| [`demo_forward_hook.py`](./demo_forward_hook.py) | Forward Pre/Post Hook：查看 shape、修改输入输出、外挂残差、保存激活、`model(x)` vs `forward(x)` 对比 | 第 2、4 章 |
| [`demo_backward_hook.py`](./demo_backward_hook.py) | Backward Hook：查看/修改梯度、梯度统计、梯度消失检测 | 第 5、9.2 章 |
| [`demo_tensor_hook.py`](./demo_tensor_hook.py) | Tensor Hook：捕获中间变量梯度、差异化学习率、链式法则验证、训练中监控 | 第 6 章 |
| [`demo_feature_extraction.py`](./demo_feature_extraction.py) | 实战：从 CNN 提取中间特征、激活统计、特征相似度、感知损失 | 第 9.1 章 |
| [`demo_gradient_clipping.py`](./demo_gradient_clipping.py) | 实战：逐层梯度裁剪/缩放、梯度异常监控、选择性冻结、梯度累积观测 | 第 9.3 章 |
| [`demo_hook_remove.py`](./demo_hook_remove.py) | 生命周期：注册/移除、批量管理、上下文管理器、执行顺序与 prepend、内存安全对比 | 第 8、10 章 |

### 11.2 运行方式

```bash
# 依赖：PyTorch（demo_hook_remove.py 的 prepend=True 需 PyTorch 2.0+）
cd 00_pytorch_hooks

# 运行单个 demo
python demo_forward_hook.py
python demo_backward_hook.py
python demo_tensor_hook.py
python demo_feature_extraction.py
python demo_gradient_clipping.py
python demo_hook_remove.py

# 运行所有 demo
for f in demo_*.py; do echo "=== $f ===" && python "$f" && echo; done
```
