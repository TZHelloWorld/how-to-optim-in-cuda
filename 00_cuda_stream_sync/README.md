# CUDA Stream 与同步机制指南

> 本文系统介绍 CUDA 的异步执行模型、Stream 与同步机制：从 Host/Device 异步关系出发，依次讲解 Stream 的执行规则、三级显式同步原语、CUDA Event、Default Stream 语义、PyTorch 的 Stream 实现、隐式同步的触发条件与规避方法，以及 CUDA Graph；最后以 SGLang + Qwen3-0.6B 的真实 profiler trace 为案例，分析 LLM 推理中的多 Stream 架构与常见同步性能陷阱。
>
> 读完本文后，你应当能够：读懂 `torch.profiler` trace 中的 `cudaStreamSynchronize`、`cudaEventSynchronize`、`cudaGraphLaunch` 等事件；定位隐式同步的根因；并在实际项目中写出不阻塞 GPU 流水线的代码。

---

## 目录

- [第 1 章 异步执行模型：Host 与 Device](#第-1-章-异步执行模型host-与-device)
- [第 2 章 CUDA Stream：有序操作队列](#第-2-章-cuda-stream有序操作队列)
- [第 3 章 显式同步：三级等待与非阻塞查询](#第-3-章-显式同步三级等待与非阻塞查询)
- [第 4 章 CUDA Event：Stream 中的标记点](#第-4-章-cuda-eventstream-中的标记点)
- [第 5 章 Default Stream 语义：NULL Stream 与 Blocking/Non-Blocking](#第-5-章-default-stream-语义null-stream-与-blockingnon-blocking)
- [第 6 章 PyTorch 的 Stream 实现](#第-6-章-pytorch-的-stream-实现)
- [第 7 章 隐式同步：触发条件、识别与规避](#第-7-章-隐式同步触发条件识别与规避)
- [第 8 章 CUDA Graph：录制与重放](#第-8-章-cuda-graph录制与重放)
- [第 9 章 案例研究：LLM 推理中的多 Stream 架构](#第-9-章-案例研究llm-推理中的多-stream-架构)
- [第 10 章 优化最佳实践](#第-10-章-优化最佳实践)
- [第 11 章 配套实验代码](#第-11-章-配套实验代码)
- [第 12 章 参考资料](#第-12-章-参考资料)
- [附录：知识结构总图](#附录知识结构总图)

---

## 第 1 章 异步执行模型：Host 与 Device

### 1.1 两个独立的处理器

CPU（Host）与 GPU（Device）是两个独立的处理器，通过 PCIe 总线连接：

```
┌───────────────┐       PCIe Bus        ┌───────────────┐
│     CPU       │ ◄──────────────────►  │     GPU       │
│ (Host)        │      数据传输通道      │ (Device)      │
│ 运行 Python/  │                       │ 执行 CUDA     │
│ C++ 控制逻辑  │                       │ Kernel 计算   │
└───────────────┘                       └───────────────┘
```

在 CUDA 编程模型中，CPU 负责**提交任务**（kernel 启动、内存拷贝），GPU 负责**执行任务**。两者的分工决定了整个体系的核心特性——**异步**。

### 1.2 异步提交

CPU 向 GPU 提交操作后**立即返回**，不等待操作执行，甚至不等待操作开始：

> "Asynchronous calls usually return before the dispatched operation has completed and may return before the asynchronous operation has started."
>
> — [CUDA Programming Guide, Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)

这带来两个直接推论，也是本文全部内容的出发点：

1. **CPU 时间线与 GPU 时间线是分离的**——profiler trace 中 CPU 侧的 `cudaLaunchKernel` 与 GPU 侧的 kernel 执行往往相隔很远；
2. **凡是 CPU 需要使用 GPU 结果的地方，都必须引入同步**——同步用得对，开销可忽略；用错了，就是性能瓶颈。

后续章节围绕两个问题展开：GPU 上的操作按什么规则排队执行（Stream，第 2 章）；CPU 与 GPU、GPU 各队列之间如何协调等待（同步机制，第 3~4 章）。

---

## 第 2 章 CUDA Stream：有序操作队列

### 2.1 定义

**CUDA Stream 是 GPU 上的一个有序操作队列**。程序向 Stream 提交操作（kernel 启动、异步内存拷贝等），这些操作在 Stream 内部按提交顺序依次执行。

> "At the most basic level, a CUDA stream is an abstraction which allows the programmer to express a sequence of operations. A stream operates like a work-queue into which programs can add operations, such as memory copies or kernel launches, to be executed in order."
>
> — [CUDA Programming Guide, CUDA Streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams)

### 2.2 三条执行规则

| 规则 | 含义 |
|------|------|
| **Stream 内严格有序** | 同一 Stream 中，前一个操作完成后，下一个才能开始 |
| **Stream 间可并行** | 不同 Stream 上的操作没有顺序约束，可以并发执行 |
| **提交异步** | CPU 将操作入队后立即返回，不等待执行 |

三条规则的组合效果：

```
CPU 时间线（一路提交，不等待）：
  提交 A→Stream1   提交 B→Stream2   提交 C→Stream1   ...继续执行后续代码

GPU 时间线：
  Stream 1:  [ 任务 A ]──────────[ 任务 C ]   ← 同 Stream：C 必须等 A
  Stream 2:  [ 任务 B ]                        ← 不同 Stream：B 与 A 可并行
```

由此可见，Stream 是表达**任务依赖关系**的基本工具：需要有序的操作放进同一个 Stream，可以并行的操作放进不同的 Stream。

### 2.3 基本 API

**CUDA C：**

```c
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

kernelA<<<grid, block, 0, stream1>>>(...);  // 第 4 个配置参数指定 Stream
kernelB<<<grid, block, 0, stream2>>>(...);  // 与 kernelA 可并行
kernelC<<<grid, block, 0, stream1>>>(...);  // 必须等 kernelA 完成

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

**PyTorch：**

```python
import torch

stream_a = torch.cuda.Stream()
stream_b = torch.cuda.Stream()

with torch.cuda.stream(stream_a):
    result_a = tensor_a + tensor_b      # 提交到 stream_a

with torch.cuda.stream(stream_b):
    result_b = tensor_c * tensor_d      # 提交到 stream_b，可与上面并行
```

### 2.4 Stream 优先级

CUDA 支持为 Stream 设置优先级（数值越小优先级越高）：

```c
int minPriority, maxPriority;
cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority);
// 典型值：minPriority = -1（最高），maxPriority = 0（默认/最低）

cudaStream_t highPriorityStream;
cudaStreamCreateWithPriority(&highPriorityStream, cudaStreamDefault, minPriority);
```

需要注意：优先级只是调度**提示（hint）**，既不保证严格的执行顺序，也不会抢占（preempt）正在执行的 kernel。

---

## 第 3 章 显式同步：三级等待与非阻塞查询

### 3.1 何时需要同步

异步模型下，以下场景必须等待 GPU：

- **CPU 读取 GPU 结果**：打印、落盘、控制流判断（`.item()`、`.cpu()`）；
- **建立数据依赖**：后续操作依赖前一操作的输出，且二者不在同一个 Stream 上；
- **性能测量**：计时前必须确保被测操作真正完成。

同步的设计原则是：**等待范围越小越好**。CUDA 为此提供了从粗到细的三级原语。

### 3.2 三级显式同步

| 函数 | 等待范围 | 阻塞代价 | 典型场景 |
|------|----------|---------|----------|
| `cudaDeviceSynchronize()` | 所有 Stream 的所有工作 | 最重 | 调试、程序退出前 |
| `cudaStreamSynchronize(stream)` | 指定 Stream 的所有工作 | 中等 | 等待某条流水线完成 |
| `cudaEventSynchronize(event)` | Stream 中某个标记点之前的工作 | 最轻 | 精确等待特定操作 |

> "`cudaDeviceSynchronize()` waits until all preceding commands in all streams of all host threads have completed."
>
> "`cudaStreamSynchronize()` takes a stream as a parameter and waits until all preceding commands in the given stream have completed."
>
> — [CUDA Programming Guide, Explicit Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#explicit-synchronization)

PyTorch 中的对应关系：`torch.cuda.synchronize()` → Device 级；`stream.synchronize()` → Stream 级；`event.synchronize()` → Event 级。

### 3.3 非阻塞查询：Query

阻塞等待之外，CUDA 还提供"只看一眼、不等待"的查询接口：

```c
cudaError_t status = cudaStreamQuery(stream);
if (status == cudaSuccess) {
    // Stream 已全部完成，可以处理结果
} else if (status == cudaErrorNotReady) {
    // Stream 还在执行，CPU 先做别的工作
}
```

```python
event = torch.cuda.Event()
event.record(stream_a)

if event.query():        # 非阻塞检查
    ...  # GPU 已完成
else:
    ...  # GPU 未完成，CPU 继续做其他事（如准备下一个 batch）
```

在 LLM 推理的 overlap scheduling 中（第 9 章），CPU 正是通过 `cudaEventQuery` 轮询 GPU 状态，避免阻塞在等待上。

### 3.4 GPU 端同步：cudaStreamWaitEvent

以上原语都是 **CPU 等 GPU**。还有一类重要需求是 **GPU 的一个 Stream 等另一个 Stream**——此时 CPU 完全不需要参与：

```c
cudaEvent_t event;
cudaEventCreate(&event);
cudaEventRecord(event, streamA);          // 在 streamA 上打标记
cudaStreamWaitEvent(streamB, event, 0);   // streamB 等待该标记，CPU 立即返回
```

```python
event = torch.cuda.Event()
event.record(stream_a)          # stream_a 上打标记
stream_b.wait_event(event)      # stream_b 等待标记；CPU 不阻塞
```

区分两类等待是理解多 Stream 程序的关键：

| 原语 | 谁在等 | CPU 是否阻塞 |
|------|--------|-------------|
| `cudaEventSynchronize(event)` | CPU 等 GPU | **是** |
| `cudaStreamWaitEvent(stream, event)` | GPU Stream 等 GPU Stream | **否** |

PyTorch 的 `stream_b.wait_stream(stream_a)` 是 `wait_event` 的便捷形式：等价于在 `stream_a` 的**当前末尾**打一个 Event 并让 `stream_b` 等待它——注意它等待的是 `stream_a` 上**已提交的全部工作**，粒度较粗（相关陷阱见第 7、9 章）。

---

## 第 4 章 CUDA Event：Stream 中的标记点

### 4.1 定义

CUDA Event 是**插入到 Stream 中的标记**。当 Stream 执行到该标记位置时，Event 变为完成（completed）状态。

> "CUDA events are a mechanism for inserting markers into a CUDA stream. They are essentially like tracer particles that can be used to track the progress of tasks in a stream."
>
> — [CUDA Programming Guide, CUDA Events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-events)

### 4.2 生命周期

```
cudaEventCreate()      创建 Event 对象
cudaEventRecord()      插入到某个 Stream 的当前末尾
cudaEventQuery()       非阻塞查询是否完成
cudaEventSynchronize() 阻塞等待完成
cudaEventDestroy()     销毁
```

### 4.3 三大用途

**用途 1：精确计时。** Event 记录的是 GPU 时间线上的时刻，比 CPU 端计时准确：

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(...);
cudaEventRecord(stop, stream);

cudaStreamSynchronize(stream);

float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);   // 单位 ms
```

**用途 2：CPU 精确等待特定操作。** 相比 `cudaStreamSynchronize` 等待整个 Stream，Event 只等到标记处：

```c
kernel1<<<grid, block, 0, stream>>>(...);
cudaEventRecord(event, stream);          // 标记在 kernel1 之后
kernel2<<<grid, block, 0, stream>>>(...);

cudaEventSynchronize(event);             // 只等 kernel1；kernel2 可能还在执行
// 此时可安全使用 kernel1 的结果
```

**用途 3：Stream 间建立点对点依赖（CPU 不阻塞）。** 生产者-消费者模式的标准写法：

```c
kernelProduce<<<grid, block, 0, streamA>>>(...);   // Stream A 生产数据
cudaEventRecord(event, streamA);

cudaStreamWaitEvent(streamB, event, 0);            // GPU 端等待，CPU 不阻塞
kernelConsume<<<grid, block, 0, streamB>>>(...);   // Stream B 消费数据
```

配套实验 Case 8 演示了这三种用法的 trace 差异。

---

## 第 5 章 Default Stream 语义：NULL Stream 与 Blocking/Non-Blocking

### 5.1 NULL Stream（Legacy Default Stream）

启动 kernel 或调用内存拷贝时若不指定 Stream，操作进入 **Legacy Default Stream**（又称 NULL Stream、stream 0）。它带有特殊的**隐式同步语义**：

> "The legacy default stream is an implicit stream which synchronizes with all other streams in the same CUcontext except for non-blocking streams."
>
> — [CUDA Runtime API, Stream synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html)

具体表现为双向屏障：

1. NULL Stream 上的操作**启动前**：等待所有 Blocking Stream 的已有工作完成；
2. NULL Stream 上的操作**提交后**：所有 Blocking Stream 的后续工作必须等它完成。

换言之，**NULL Stream 是一个全局序列化点，一旦使用即破坏所有 Stream 间的并发性**：

```c
cudaStream_t s;
cudaStreamCreate(&s);       // Blocking Stream（默认属性）

kernel1<<<1,1,0,s>>>();     // stream s
kernel2<<<1,1>>>();         // 未指定 → NULL Stream
kernel3<<<1,1,0,s>>>();     // stream s

// 实际执行：kernel1 → kernel2 → kernel3 完全串行
// kernel2 若在独立 Stream 上，本可与 kernel1/kernel3 并行
```

### 5.2 Blocking 与 Non-Blocking Stream

Stream 是否受 NULL Stream 约束，取决于创建时的标志：

| 类型 | 创建方式 | 与 NULL Stream 的关系 |
|------|----------|---------------------|
| Blocking Stream | `cudaStreamCreate()` | 受隐式同步双向约束 |
| Non-Blocking Stream | `cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)` | **完全不受影响** |

```c
cudaStream_t s1, s2;
cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);

kernel1<<<1,1,0,s1>>>();
kernel2<<<1,1>>>();          // NULL Stream
kernel3<<<1,1,0,s2>>>();

// 三个 kernel 可以并行：s1、s2 不受 NULL Stream 约束
```

### 5.3 Per-Thread Default Stream

CUDA 7 起可启用"每线程独立默认 Stream"，使每个 CPU 线程拥有自己的 default stream，行为等同 Non-Blocking，消除全局序列化：

```c
// 编译选项：nvcc --default-stream per-thread
// 或在包含头文件前定义：
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime.h>
```

### 5.4 小结

NULL Stream 的隐式同步是历史遗留设计，现代高性能代码应遵循：**要么显式指定 Non-Blocking Stream，要么启用 per-thread default stream，避免任何操作落入 Legacy NULL Stream**。下一章将看到，PyTorch 在框架层面彻底贯彻了这一原则。

---

## 第 6 章 PyTorch 的 Stream 实现

### 6.1 关键设计：全部使用 Non-Blocking Stream

PyTorch 的一个重要设计决策：**所有 Stream（包括"默认 Stream"）都以 `cudaStreamNonBlocking` 标志创建**。这意味着：

- PyTorch 的 default stream **不是** CUDA 的 NULL Stream（stream 0），而是从预分配的 Stream Pool 中取出的普通 Non-Blocking Stream；
- 框架内**不存在** NULL Stream 的隐式同步行为；
- 所有跨 Stream 依赖都必须通过**显式机制**表达（`wait_stream()` / `Event`）。

`c10/cuda/CUDAStream.h` 中的 Stream Pool 结构（每个 GPU 设备）：

```
Pool 1: 默认 Stream（1 个）
Pool 2: 普通优先级 Stream（32 个，轮询分配）
Pool 3: 高优先级 Stream
```

### 6.2 设计动机

```
若 PyTorch 使用真正的 NULL Stream：
  每次在默认 Stream 上执行操作 → 隐式同步所有其他 Stream
  → 多 Stream overlap 不可能实现

PyTorch 的实际做法：
  所有 Stream 均为 Non-Blocking → 各 Stream 独立并发
  需要协调时，通过显式 Event / wait_stream 建立依赖
```

这一设计使 trace 分析变得简单：**所有跨 Stream 依赖都能追溯到一个显式的 `cudaStreamWaitEvent` 调用**，不存在"看不见的"同步。

### 6.3 从真实 Trace 验证

本目录附带的 SGLang 推理 trace 中，各 Stream 的 GPU 事件分布：

```
Stream 20:  30,358 个 GPU 事件   ← PyTorch 的 "default stream"（Non-Blocking）
Stream 48:   1,418 个 GPU 事件   ← 额外创建的 Stream（Non-Blocking）
Stream 0:        0 个 GPU 事件   ← NULL Stream，从未被使用
```

配套实验 Case 7 验证了 PyTorch default stream 的指针非 0，并演示了多 Stream 真正并行的行为。

### 6.4 何时仍需警惕 NULL Stream

- 直接调用 CUDA C API 且未指定 Stream；
- 使用同步版 `cudaMemcpy`（非 Async 版本，总是走 NULL Stream 语义）；
- 第三方 C/C++ 扩展库创建 Stream 时未加 `cudaStreamNonBlocking` 标志。

---

## 第 7 章 隐式同步：触发条件、识别与规避

显式同步是自己写的，容易察觉；**隐式同步**则藏在框架和驱动内部——代码里没有任何 `synchronize` 字样，trace 里却出现了 `cudaStreamSynchronize`。这是实际项目中最常见的同步性能问题来源。

### 7.1 根源：Pageable 内存与 Pinned 内存

理解隐式同步，先要区分两种 Host 内存：

| 类型 | 分配方式 | 特点 | GPU DMA 可直接访问？ |
|------|----------|------|---------------------|
| **Pageable**（可分页） | 普通 `malloc` / `new` / 默认 tensor | 可能被 OS 换出到磁盘 | **否** |
| **Pinned**（锁页） | `cudaMallocHost` / `tensor.pin_memory()` | 锁定在物理内存 | **是** |

> "In order for memory copies involving CPU memory to be carried out asynchronously, the host buffers must be pinned and page-locked. `cudaMemcpyAsync()` will function correctly if host memory which is not pinned and page-locked is used, but it will revert to a synchronous behavior."
>
> — [CUDA Programming Guide, Launching memory transfers in CUDA streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#launching-memory-transfers-in-cuda-streams)

机制解释：GPU 的 DMA 引擎只能访问物理地址固定的内存。Pageable 内存的页面可能随时被 OS 移动，驱动必须先把数据拷入内部的 pinned staging buffer 再启动 DMA——这一过程要求与目标 Stream 同步。因此：

- **Pageable → GPU 拷贝** = `cudaMemcpyAsync` + `cudaStreamSynchronize`（伪异步）；
- **Pinned → GPU 拷贝**（配合 `non_blocking=True`）= 仅 `cudaMemcpyAsync`（真异步）。

### 7.2 PyTorch 中的隐式同步触发规则

| Python 代码 | 内部实际操作 | 触发隐式 sync |
|-----------------|-------------|:---:|
| `tensor.to(device)`（默认参数） | `cudaMemcpyAsync` + `cudaStreamSynchronize` | **是** |
| `tensor.pin_memory().to(device, non_blocking=True)` | 仅 `cudaMemcpyAsync` | 否 |
| `tensor.item()` | DtoH 拷贝 + sync | **是** |
| `tensor.cpu()` / `tensor.numpy()` | DtoH 拷贝 + sync | **是** |
| `torch.tensor(data, device='cuda')` | 先建 Pageable CPU tensor，再 HtoD + sync | **是** |
| `torch.cuda.synchronize()` | `cudaDeviceSynchronize()` | **是**（显式，等全部） |

其中 `torch.tensor(data, device='cuda')` 最具迷惑性——看似"直接在 GPU 上创建"，实际等价于：

```python
cpu_tmp = torch.tensor(data)   # Pageable CPU tensor
gpu = cpu_tmp.to(device)       # Pageable HtoD 拷贝 + 隐式 sync
```

### 7.3 Sync 耗时为何差异巨大：等待被"放大"

同一个 `cudaStreamSynchronize`，在 trace 中有时 ~8us，有时长达 29ms。差异不在 sync 本身，而在**它被迫等待的排队工作量**：

```
短 sync（~8us）：
  目标 Stream 空闲 → sync 只有 API 调用本身的开销，立即返回

长 sync（~29ms）：
  Stream 20: [kernel1][kernel2]...[kernel325]     ← 大量推理 kernel 排队
  Stream 48: (wait_stream 依赖 Stream 20) → [memcpy]
  CPU: cudaStreamSynchronize(stream 48)
       → 必须等 Stream 20 全部完成 + Stream 48 完成
```

注意第二种情况中的**放大链条**：

1. `stream48.wait_stream(stream20)` 建立了跨 Stream 依赖；
2. 在 stream48 上做了一次 Pageable 拷贝，触发隐式 `cudaStreamSynchronize(stream48)`；
3. 该 sync 传递性地等待了 stream20 上的全部推理 kernel。

一次看似无害的小拷贝，阻塞了整条流水线。这正是配套实验 Case 4 与 Case 9 复现的 SGLang 真实案例。

### 7.4 在 Profiler Trace 中识别隐式同步

```
CPU 侧特征模式：
  aten::copy_
    ├── cudaMemcpyAsync           (dur ~10us)
    └── cudaStreamSynchronize     (dur 8us ~ 29ms)   ← 隐式 sync

GPU 侧对应事件：
  "Memcpy HtoD (Pageable -> Device)"    ← 关键词 "Pageable"
  "Memcpy HtoD (Pinned -> Device)"      ← Pinned：通常无后续 sync
```

诊断口诀：**看到 `aten::copy_` 下挂着 `cudaStreamSynchronize`，且 GPU 侧 Memcpy 标注 Pageable，即为隐式同步；sync 耗时长则检查该 Stream 的上游 `wait_stream`/`wait_event` 依赖**。

---

## 第 8 章 CUDA Graph：录制与重放

### 8.1 动机：CPU 提交开销

每次 kernel 启动都需要 CPU 调用 `cudaLaunchKernel`（约 5~20us）。当计算由大量小 kernel 组成、且同一模式反复执行时（典型如 LLM decode，每步几十个 kernel、执行上千次），CPU 提交开销会成为瓶颈：GPU 执行完一个 kernel 后，必须等 CPU 提交下一个，出现"GPU 等 CPU"的间隙。

### 8.2 原理：一次录制，多次重放

CUDA Graph 将一系列操作**录制**为图结构，之后每次执行只需一个 `cudaGraphLaunch` 调用：

> "Capturing or creating a graph can help reduce latency and CPU overhead of repeatedly invoking the same chain of API calls from the host thread. Instead, the APIs to specify the graph operations can be called once, and then the resulting graph executed many times."
>
> — [CUDA Programming Guide, CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#introduction-to-cuda-graphs-with-stream-capture)

三步流程：

```
1. 录制（Capture）      ：以 stream capture 方式记录所有操作，只做一次
2. 实例化（Instantiate）：预编译与运行时结构准备，只做一次
3. 重放（Launch/Replay）：之后每次一个 API 调用重放整图
```

效果对比：

```
普通模式：
  CPU: [launch k1][launch k2]...[launch k28]    ← 28 次 API 调用
  GPU: [k1]~gap~[k2]~gap~...~gap~[k28]          ← 每个 gap 5~20us

Graph 模式：
  CPU: [cudaGraphLaunch]                        ← 1 次 API 调用（~26us）
  GPU: [k1][k2]...[k28]                         ← 间隙仅 ~8us 硬件调度延迟
```

### 8.3 代码示例

**CUDA C（Stream Capture）：**

```c
bool graphCreated = false;
cudaGraph_t graph;
cudaGraphExec_t instance;

for (int step = 0; step < NSTEP; step++) {
    if (!graphCreated) {
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);   // 1. 录制
        for (int i = 0; i < NKERNEL; i++) {
            kernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
        }
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);          // 2. 实例化
        graphCreated = true;
    }
    cudaGraphLaunch(instance, stream);                                  // 3. 重放
    cudaStreamSynchronize(stream);
}
```

**PyTorch：**

```python
# 热身：确保内存分配稳定
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    z = (x_static @ y_static + x_static).relu()
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()

# 录制
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    z = (x_static @ y_static + x_static).relu()

# 重放
for _ in range(50):
    g.replay()
```

### 8.4 适用条件

CUDA Graph 的前提是**计算图静态**：

- 每次执行的 kernel 序列、shape、内存地址必须一致（输入通过预分配的 static tensor 原地更新）；
- 不支持图内的动态控制流和动态 shape；
- 适合 LLM **decode** 阶段（固定 batch 结构），不适合 **prefill** 阶段（变长输入）。

第 9 章的真实 trace 中，1,390 次 decode 迭代全部通过 `cudaGraphLaunch` 完成。

---

## 第 9 章 案例研究：LLM 推理中的多 Stream 架构

本章综合运用前面所有概念，分析本目录附带的真实 trace（SGLang 运行 Qwen3-0.6B）。

### 9.1 为什么需要多个 Stream

一次 LLM 推理迭代包含三类工作：

```
1. [CPU+GPU] 调度：接收请求、组 batch、分配 KV cache、数据上卡
2. [GPU]     前向：RMSNorm / GEMM / Attention / Sampling 等 kernel
3. [CPU+GPU] 收尾：取回结果、更新请求状态、释放资源
```

若全部放在一个 Stream 上，调度类 GPU 操作（拼接、拷贝）与模型前向串行执行。将两类工作分到两条 Stream，即可实现 **overlap**：GPU 在 forward stream 上推理 batch N 的同时，CPU 在 schedule stream 上为 batch N+1 做准备。

### 9.2 两 Stream 架构

```
Schedule Stream: [准备 batch N+1] ──────────────── [处理 batch N 结果] ─ [准备 batch N+2]
                       │                            ↑                        │
                       │ wait_stream                │ Event.synchronize      │ wait_stream
                       ↓                            │                        ↓
Forward Stream:      [batch N 前向推理 + DtoH + Event.record]            [batch N+1 前向]
```

跨 Stream 协调全部使用第 3~4 章的原语：`wait_stream` 保证前向读到就绪的输入；`Event.record + synchronize/query` 让 CPU 在恰当的时刻取回结果。

### 9.3 Trace 总体数据

| 项目 | 值 |
|------|-----|
| 模型 / GPU | Qwen3-0.6B / NVIDIA A800-SXM4-80GB |
| Forward Stream（Stream 20） | 28,397 个 kernel（推理主力） |
| Schedule Stream（Stream 48） | 1,418 个 kernel（调度辅助） |
| Prefill 迭代 | 74 次（常规 kernel 提交） |
| Decode 迭代 | 1,390 次（全部走 CUDA Graph） |
| GPU 利用率 | **98.7%** |

decode 阶段 CPU 每 ~512us 提交一个 Graph，GPU 每 ~5ms 执行完一个 Graph，提交与执行比约 1:10——CPU 远远跑在前面，GPU 流水线始终满载，这是 98.7% 利用率的直接来源。

### 9.4 同步事件统计

| 事件类型 | 次数 | 用途（对应章节） |
|----------|------|------|
| `cudaGraphLaunch` | 1,390 | decode 迭代重放（第 8 章） |
| `cudaStreamWaitEvent` | 74 | forward 等 schedule 的输入就绪（第 3.4 节） |
| `cudaEventSynchronize` | 74 | CPU 取回 prefill 结果（第 4.3 节用途 2） |
| `cudaEventQuery` | 383 | CPU 非阻塞轮询，实现 overlap（第 3.3 节） |
| `cudaStreamSynchronize` | 378 | 多为隐式触发，大部分 <10us（第 7 章） |

### 9.5 一次 Prefill 迭代的时间线

```
CPU 主线程:
  [准备数据: cat / HtoD copy / write_req] → [wait_stream] → [提交 forward kernels]
       │                                                        │
       ↓                                                        ↓
Schedule Stream (48):                        Forward Stream (20):
┌───────────────────────────────┐           ┌──────────────────────────────────────┐
│ CatBatchedCopy ×8（拼接输入） │           │ resolve_future_token_ids            │
│ Memcpy HtoD ×14（数据上卡）  │           │ compute_position_kernel              │
│ write_req_to_token_pool       │           │ [RMSNorm → QKV GEMM → Attention →  │
└───────────────────────────────┘           │  O_proj → RMSNorm → gate_up GEMM → │
        |←~5-9ms→|                          │  SwiGLU → down_proj] × 28 层        │
                                            │ → final_norm → lm_head → sample     │
                                            └──────────────────────────────────────┘
                                            |←──────── ~60ms 模型前向 ──────────→|
```

### 9.6 该架构中的典型陷阱：29ms 的隐式 sync

trace 中曾出现单次 29ms 的 `cudaStreamSynchronize`，根因是第 7.3 节描述的放大链条在真实代码中的体现：

```python
# schedule_stream 已通过 wait_stream 依赖 forward_stream
with torch.cuda.stream(schedule_stream):
    schedule_stream.wait_stream(forward_stream)
    # 问题代码：Pageable HtoD → 隐式 sync → 等待全部 forward kernel
    indices = torch.tensor(data, device='cuda', dtype=torch.uint64)
```

修复方式（配套实验 Case 9 给出完整对照）：

```python
# Pinned + 异步拷贝：不触发 sync，forward kernel 不受干扰
indices_cpu = torch.tensor(data, dtype=torch.uint64).pin_memory()
indices = indices_cpu.to('cuda', non_blocking=True)
```

---

## 第 10 章 优化最佳实践

### 10.1 问题—方案对照表

| 根因 | 问题代码 | 优化方案 | 效果 |
|------|----------|----------|------|
| Pageable HtoD 隐式 sync | `tensor.to(device)` | `tensor.pin_memory().to(device, non_blocking=True)` | 消除 sync |
| GPU 上直接建 tensor | `torch.tensor(data, device='cuda')` | CPU 创建 → `pin_memory()` → 异步拷贝 | 消除 sync |
| 循环内取标量 | `for: x.item()` | 结果留在 GPU，循环外统一同步后取回 | sync 次数 N→1 |
| DtoH 阻塞拷贝 | `tensor.cpu()` | 预分配 pinned buffer + `copy_(non_blocking=True)` | 异步化 |
| 跨流依赖放大 sync | `wait_stream` 后做隐式 sync 操作 | 用 Event 建立最小粒度依赖 | 缩短等待 |
| 全局同步 | `torch.cuda.synchronize()` | 改用 `stream.synchronize()` 或 `event.synchronize()` | 缩小范围 |

### 10.2 代码对照

**HtoD 拷贝：**

```python
# Bad：Pageable 拷贝，每次触发隐式 sync
gpu_tensor = torch.tensor([1, 2, 3], device='cuda')

# Good：Pinned + 异步拷贝
cpu_tensor = torch.tensor([1, 2, 3]).pin_memory()
gpu_tensor = cpu_tensor.to('cuda', non_blocking=True)

# Best（高频场景）：复用预分配的 pinned buffer
pinned_buf[:n] = torch.tensor(data)
gpu_buf.copy_(pinned_buf[:n], non_blocking=True)
```

**循环取结果：**

```python
# Bad：每次迭代 .item()，每次都 sync，完全串行化
results = []
for i in range(100):
    x = model(inputs[i])
    results.append(x[0].item())     # 100 次 sync

# Good：数据留在 GPU，最后一次性同步
results_gpu = []
for i in range(100):
    x = model(inputs[i])
    results_gpu.append(x[0])        # 不触发 sync

torch.cuda.synchronize()            # 仅 1 次
results = [r.item() for r in results_gpu]   # 数据已就绪，取值近乎免费
```

**等待粒度：**

```python
# Bad：等待整个 Stream（包含不需要的后续工作）
stream_a.synchronize()

# Good：Event 只等到需要的位置
event = torch.cuda.Event()
event.record(stream_a)      # 紧跟在需要的操作之后
event.synchronize()         # 后续工作继续在 GPU 上执行
```

### 10.3 通用原则

1. 频繁参与 HtoD 拷贝的 CPU tensor 一律 `pin_memory()`，拷贝一律 `non_blocking=True`；
2. 避免在 GPU 队列繁忙时执行任何 Pageable 拷贝（尤其是位于 `wait_stream` 下游的 Stream 上）；
3. 跨 Stream 依赖优先用 Event（点对点），而非 `wait_stream`（等待全部）；
4. 等待粒度从小到大依次选择：`event.query()` 轮询 → `event.synchronize()` → `stream.synchronize()` → `torch.cuda.synchronize()`；
5. 重复执行的静态计算图（如 LLM decode）使用 CUDA Graph 消除提交开销；
6. 定期用 `torch.profiler` 导出 trace，检查 `cudaStreamSynchronize` 的耗时分布——长 sync 必有放大链条。

---

## 第 11 章 配套实验代码

### 11.1 目录结构

```
00_cuda_stream_sync/
├── README.md                              # 本文档
├── test_cuda_stream_sync_tutorial.py      # 10 个渐进式实验用例
└── Qwen3-0.6B-*.trace.json.gz             # SGLang 真实 profiler trace（第 9 章数据来源）
```

### 11.2 实验用例与章节对应

| Case | 主题 | 验证的知识点 | 对应章节 |
|------|------|-------------|---------|
| 1 | Pageable 内存隐式 Sync | `.to(device)` 为何触发 sync，及 sync 耗时的放大 | 第 7 章 |
| 2 | Pinned 内存消除 Sync | pin_memory 实现真异步拷贝 | 第 7.1 节 |
| 3 | `non_blocking` 标志行为 | 三种组合的实测耗时差异 | 第 7.2 节 |
| 4 | 跨 Stream 依赖 | `wait_stream` 如何放大 sync 耗时 | 第 7.3 节 |
| 5 | `.item()/.cpu()` 陷阱 | 循环内取标量导致串行化 | 第 10.2 节 |
| 6 | `torch.tensor(device='cuda')` 陷阱 | 隐藏的 Pageable 拷贝与 sync | 第 7.2 节 |
| 7 | NULL Stream 行为 | PyTorch 如何规避隐式同步 | 第 5~6 章 |
| 8 | Event 细粒度同步 | 四种同步粒度的对比 | 第 3~4 章 |
| 9 | Overlap Scheduling 模式 | 问题模式与优化模式对照 | 第 9 章 |
| 10 | CUDA Graph 加速 | 录制-重放消除提交开销 | 第 8 章 |

### 11.3 运行方式

```bash
# 需要 CUDA GPU 环境
cd 00_cuda_stream_sync/

python test_cuda_stream_sync_tutorial.py             # 运行全部用例
python test_cuda_stream_sync_tutorial.py --list      # 列出所有用例
python test_cuda_stream_sync_tutorial.py --case 1    # 运行单个用例并生成 trace

# trace 输出目录
ls /tmp/cuda_stream_tutorial_traces/
```

### 11.4 查看 Trace

将生成的 `.json`（或本目录的 `.json.gz`）拖入以下任一工具：

- `chrome://tracing`
- https://ui.perfetto.dev

建议重点观察：CPU 侧 `cudaStreamSynchronize` 的耗时分布、GPU 侧 Memcpy 事件的 `Pageable/Pinned` 标注、以及各 Stream 泳道的并行情况。

---

## 第 12 章 参考资料

| 主题 | 链接 |
|------|------|
| CUDA 异步执行总览 | https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html |
| CUDA Stream 概念 | https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams |
| Stream 同步 | https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#stream-synchronization |
| CUDA Events | https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-events |
| 显式同步机制 | https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#explicit-synchronization |
| 隐式同步 | https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#implicit-synchronization |
| Blocking/Non-Blocking Stream | https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#blocking-and-non-blocking-streams-and-the-default-stream |
| Legacy Default Stream | https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#legacy-default-stream |
| 异步内存拷贝 | https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#launching-memory-transfers-in-cuda-streams |
| CUDA Graphs | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html |
| cudaStreamSynchronize API | https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html |
| Stream 同步行为（Runtime API） | https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html |

---

## 附录：知识结构总图

```
                       ┌────────────────────────────────────────────┐
                       │         CUDA 异步执行与同步体系             │
                       └───────────────────────┬────────────────────┘
                                               │
          ┌────────────────────────────────────┼────────────────────────────────────┐
          │                                    │                                    │
┌─────────▼────────────┐        ┌──────────────▼─────────────┐       ┌─────────────▼────────────┐
│  Stream（工作队列）   │        │   Event（标记点）           │       │   同步（等待机制）        │
│                      │        │                            │       │                          │
│ • Stream 内有序      │        │ • 插入 Stream 的标记        │       │ • Device 级：等全部       │
│ • Stream 间并行      │        │ • 可查询 / 可等待           │       │ • Stream 级：等一个队列   │
│ • CPU 异步提交       │        │ • 跨 Stream 点对点依赖      │       │ • Event 级：等一个标记    │
│ • 可设优先级(hint)   │        │ • GPU 时间线精确计时        │       │ • Query：非阻塞轮询       │
└──────────┬───────────┘        └────────────────────────────┘       └──────────────────────────┘
           │
┌──────────┼───────────────────────────────────────────┐
│          │                                           │
│  ┌───────▼─────────┐ ┌──────────────────┐ ┌──────────▼───────────┐
│  │ NULL Stream     │ │ Blocking Stream  │ │ Non-Blocking Stream  │
│  │ (stream 0)      │ │ (默认创建)        │ │ (cudaStreamNon-      │
│  │                 │ │                  │ │  Blocking)           │
│  │ 与 Blocking     │ │ 受 NULL Stream   │ │ 不受 NULL Stream     │
│  │ Stream 双向     │ │ 双向约束          │ │ 任何影响             │
│  │ 隐式同步        │ │                  │ │                      │
│  │ → 全局序列化    │ │                  │ │ → PyTorch 全部采用   │
│  └─────────────────┘ └──────────────────┘ └──────────────────────┘
│
│          ┌───────────────────────────────┐
└──────────▶  CUDA Graph                   │
           │  录制一次，重放多次            │
           │  消除 CPU 提交开销             │
           │  LLM decode GPU 利用率 98%+   │
           └───────────────────────────────┘
```

---

*最后更新: 2026-07-23*
