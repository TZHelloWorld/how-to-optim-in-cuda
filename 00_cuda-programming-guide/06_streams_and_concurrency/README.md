# 第 6 章 流与并发

> [← 上一章：内存模型](../05_memory_model/README.md) | [返回目录](../README.md) | [下一章：原子操作与 Warp 级原语 →](../07_atomics_and_warp/README.md)

前面几章的优化都发生在"一个内核内部"：怎么组织线程（第 2 章）、怎么排布 warp（第 4 章）、怎么访问内存（第 5 章）。但把单个内核优化到飞起之后，很多程序依然不快——时间线上一看，GPU 算完一段就停下来等数据，PCIe 传完数据计算单元又在发呆，**设备利用率可能连一半都不到**。本章把视角从"内核内部"拉高到"整个程序"：让传输和计算重叠、让多个内核并发，把每一件硬件的空闲时间都填上。

还记得第 1 章埋下的两颗种子吗——[内核启动是**异步**的](../01_introduction/README.md#110-cuda-程序的典型流程)（1.10 节），[CPU 与 GPU **可以同时执行**各自的代码](../01_introduction/README.md#13-异构计算cpu-与-gpu-的协同)（1.3.2 节）。本章就把这种"同时"用到极致。

本章目标：理解 CUDA 的异步执行模型与流（Stream）的概念，掌握"异步传输 + 多流流水线"这一内核之外最重要的系统级优化手段，并学会用事件管理跨流依赖、用清单排查并发被破坏的常见原因。

全章行进路线：

```text
┌─ 地基 ────────────────────────────────────────────────┐
│ 6.1 异步执行模型：哪些操作天生异步？硬件凭什么能并发？    │
│ 6.2 流：组织并发的操作队列（API、锁页内存、默认流）       │
└──────────────────────┬────────────────────────────────┘
┌─ 核心技术 ────────────▼────────────────────────────────┐
│ 6.3 多流流水线：重叠传输与计算（本章主菜）                │
│ 6.4 事件：跨流依赖   6.5 并发内核 / 主机回调 / Graph 预告 │
└──────────────────────┬────────────────────────────────┘
┌─ 工程收尾 ────────────▼────────────────────────────────┐
│ 6.6 常见陷阱：并发是怎么被悄悄破坏的                     │
│ 6.7 优化清单 → 6.8 小结 → 6.9 深度专题导读与练习         │
└───────────────────────────────────────────────────────┘
```

## 本章目录

- [6.1 异步执行：一切并发的起点](#61-异步执行一切并发的起点)
- [6.2 流：组织并发的操作队列](#62-流组织并发的操作队列)
- [6.3 多流流水线：重叠传输与计算](#63-多流流水线重叠传输与计算)
- [6.4 事件（Event）：计时与跨流依赖](#64-事件event计时与跨流依赖)
- [6.5 并发内核、主机回调与 CUDA Graph](#65-并发内核主机回调与-cuda-graph)
- [6.6 常见陷阱：并发是怎么被悄悄破坏的](#66-常见陷阱并发是怎么被悄悄破坏的)
- [6.7 主机-设备交互优化清单](#67-主机-设备交互优化清单)
- [6.8 本章小结](#68-本章小结)
- [6.9 深度专题导读与动手练习](#69-深度专题导读与动手练习)
- [深度专题：CUDA Stream 与同步机制（专题 1~12）](#深度专题cuda-stream-与同步机制)——异步执行模型、三级同步原语、Default Stream 语义、PyTorch Stream 实现、隐式同步陷阱、CUDA Graph 与 LLM 推理真实 trace 案例

---

## 6.1 异步执行：一切并发的起点

想让传输和计算重叠，先要回答两个更基本的问题：CUDA 里哪些操作是"发出去就不管"的（异步）？硬件上又是谁在同时干这些活？这一节把地基打牢。

### 6.1.1 CPU 与 GPU：两条独立的时间线

CPU 和 GPU 是两个独立的处理器。CPU 向 GPU **提交任务**（启动内核、发起拷贝），GPU **执行任务**——提交和执行发生在两条独立的时间线上。官方文档对"异步"的定义比直觉更激进：**异步调用通常在被派发的操作完成之前就返回，甚至可能在操作开始之前就返回**。

这带来两个直接推论：

1. **CPU 代码行的位置 ≠ GPU 执行的时刻**——用 profiler 看时间线，CPU 侧的 `cudaLaunchKernel` 调用点和 GPU 侧内核真正执行的位置往往相隔很远；
2. **凡是 CPU 要用 GPU 的结果，就必须显式同步**——同步用对了开销可忽略，用错了就是性能瓶颈（6.6 节和深度专题的专题 7 会集中讲怎么用错的）。

### 6.1.2 哪些 CUDA 操作天生异步

对照官方列表，以下操作**相对主机是异步的**（调用后 CPU 立即继续执行）：

| 操作 | 说明 |
|------|------|
| **内核启动** `<<<...>>>` | 永远异步——这是第 1 章就认识的老朋友 |
| **设备内部的内存拷贝** | 同一设备内 D2D 拷贝不占 CPU |
| **≤ 64 KB 的 H2D 小拷贝** | 即使用同步版 `cudaMemcpy`，小块主机→设备拷贝也可能异步返回 |
| **`*Async` 后缀的内存操作** | `cudaMemcpyAsync` / `cudaMemsetAsync` 等（有前提，见 6.2.3） |

而普通的 `cudaMemcpy`（大块）、`cudaMalloc`、`cudaDeviceSynchronize` 等则是同步/阻塞的。

> [!TIP]
> 调试时怀疑"异步把报错位置搞乱了"（3.8 节的异步错误甩锅现象），可以临时设置环境变量 `CUDA_LAUNCH_BLOCKING=1`，强制所有内核启动变成同步——报错立刻回到"案发现场"。只用于调试，性能会大幅下降。

### 6.1.3 硬件前提：独立的拷贝引擎

CPU 一侧"发完就走"只是异步的一半；另一半在 GPU 上——GPU 有多类可以**并行工作的硬件引擎**：

- 计算引擎（执行内核）；
- H2D 拷贝引擎（主机 → 设备 DMA）；
- D2H 拷贝引擎（设备 → 主机 DMA）。

拷贝引擎（copy engine）是独立于 SM 的 DMA 硬件，搬数据不占用计算资源——这正是"传输与计算能够重叠"的硬件前提。具体数量可通过设备属性 `cudaDeviceProp.asyncEngineCount` 查询：为 1 时可在执行内核的同时做单向拷贝，为 2 时可同时做双向拷贝（现代数据中心 GPU 普遍为 2 以上）。

默认写法（同步 `cudaMemcpy` + 单内核）让这些引擎**串行**排队：拷入时计算引擎闲着，计算时拷贝引擎闲着。对于"数据量大、需要反复传输"的应用，传输时间可能与计算时间同量级，白白浪费一半吞吐。

用工厂类比：计算引擎是车间，两个拷贝引擎是"进货卡车"和"出货卡车"——三者本可以同时开工，默认写法却让它们排成一列纵队，永远只有一个在动。想让三者同时忙起来，就需要一个"排班表"机制——这就是流。

## 6.2 流：组织并发的操作队列

### 6.2.1 定义与三条执行规则

**CUDA 流（Stream）是一个按顺序执行的操作队列**：程序把操作（内核启动、异步拷贝等）提交进流，流内按提交顺序依次执行。并发的全部秘密在于规则二：

| 规则 | 含义 |
|------|------|
| ① 流内有序（FIFO） | 同一流中的操作严格按提交顺序执行，前一个不完成后一个不开始 |
| ② 流间无序 | **不同流的操作没有顺序保证，可以并发/重叠执行**——这是一切重叠优化的抓手 |
| ③ 默认流特殊 | 不指定流时操作进入默认流，它有特殊的全局同步语义（见 6.2.4） |

打个比方：流就像超市的收银通道——同一条队里的顾客先来后到（流内有序），多开几条通道就能同时结账（流间并发）。程序员的工作，就是把"必须有序"的操作放进同一条流、把"可以并行"的操作放进不同的流，剩下的交给硬件调度。

### 6.2.2 基本 API：创建、提交、同步、销毁

围绕流的操作总共就几个动作：

```c++
cudaStream_t stream;
cudaStreamCreate(&stream);                    // 创建流

// 异步拷贝（要求主机内存是锁页内存！见 6.2.3）
cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);

// 在指定流上启动内核（执行配置第 4 个参数；第 3 个是动态共享内存字节数）
kernel<<<grid, block, 0, stream>>>(...);

cudaStreamSynchronize(stream);   // 阻塞 CPU，等待该流全部完成
cudaStreamQuery(stream);         // 非阻塞探询：完成返回 cudaSuccess，
                                 //   未完成返回 cudaErrorNotReady（不算错误）
cudaStreamDestroy(stream);       // 销毁流（流中尚有工作时会等工作完成后才释放资源）
```

三种"等待"的粒度从粗到细：`cudaDeviceSynchronize()`（等整个设备，代价最大）→ `cudaStreamSynchronize(s)`（等一条流）→ 事件同步（等某个标记点，见 6.4）。**能用细粒度就不用粗粒度**，这是贯穿本章的原则（三级同步的完整对比见深度专题的专题 3）。

### 6.2.3 异步拷贝的前提：锁页内存

`cudaMemcpyAsync` 名字里带 Async，但它不是无条件异步的：

> [!IMPORTANT]
> `cudaMemcpyAsync` 要真正异步，主机端缓冲区**必须是锁页内存**（`cudaMallocHost` 分配，见 5.1.7 节）。对可分页内存调用异步拷贝会退化为同步行为。

原因回顾 5.1.7 节：拷贝引擎做 DMA 传输要求物理地址固定，而可分页（pageable）内存的页面随时可能被操作系统移动，驱动只能先把数据搬进内部的锁页中转缓冲区再传输——这一步迫使拷贝与流同步，"异步"名存实亡。本章深度专题的专题 7 会用 PyTorch 实验和真实 trace 展示这个退化有多隐蔽、代价有多大。

### 6.2.4 默认流的特殊性：legacy 与 per-thread

不指定流参数时操作进入**默认流（stream 0）**。传统（legacy）默认流是同步"路障"：它与所有其他流互斥——默认流上的操作开始前要等其他流清空，其他流的后续操作也要等它完成。也就是说，**哪怕只有一个操作落进默认流，也会把辛苦建立的多流并发拦腰截断**。

两种规避方式：

- 编译时加 `--default-stream per-thread`，让每个主机线程拥有独立的、不带全局路障语义的默认流；
- 或者干脆**所有操作都显式指定非默认流**——工程上更推荐，意图清晰（完整语义与 Blocking/Non-Blocking 流的区别见深度专题的专题 5）。

### 6.2.5 流的标志与优先级

创建流还有两个进阶选项，在多流工程中很常用：

```c++
// ① Non-Blocking 流：明确声明"不与 legacy 默认流互相等待"
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

// ② 带优先级的流：数值越小优先级越高
int lo, hi;
cudaDeviceGetStreamPriorityRange(&lo, &hi);            // 查询可用范围
cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, hi);
```

> [!NOTE]
> 优先级是**调度提示而非抢占**：高优先级流中待执行的块会被优先调度到 SM，但不会中断已经在跑的块。典型用途：LLM 推理中让"解码"流优先于"预填充"流，降低生成延迟——深度专题的专题 9 会在真实 trace 中看到这种用法。

## 6.3 多流流水线：重叠传输与计算

工具齐了，开始搭本章最重要的优化模式。思路：把大数据切成 n 块，交给 n 个流，每个流独立完成自己那块的"拷入 → 计算 → 拷出"——不同流的不同阶段就会在硬件引擎上交错起来，像工厂流水线一样。

### 6.3.1 重叠的前提条件清单

动手之前先对照检查，四个条件缺一不可（前两个查硬件，后两个查代码）：

| # | 条件 | 如何确认 |
|---|------|---------|
| 1 | 设备支持拷贝-计算重叠 | `cudaDeviceProp.asyncEngineCount ≥ 1`（双向重叠需 ≥ 2） |
| 2 | （若要内核间并发）设备支持并发内核 | `cudaDeviceProp.concurrentKernels == 1` |
| 3 | 主机缓冲区是**锁页内存** | `cudaMallocHost` 分配（6.2.3 节） |
| 4 | 操作分布在**不同的非默认流**，且中间没有隐式同步 | 代码检查 + `nsys` 时间线验证（6.6 节陷阱清单） |

### 6.3.2 分块流水线：完整实现

```c++
const int nStreams = 4;
int chunk = n / nStreams;                       // 假设整除；否则最后一块单独处理
cudaStream_t streams[nStreams];
for (int i = 0; i < nStreams; i++) cudaStreamCreate(&streams[i]);

for (int i = 0; i < nStreams; i++) {
    int offset = i * chunk;
    // 每个流独立地：拷入自己的分块 → 处理 → 拷出
    cudaMemcpyAsync(dA + offset, hA + offset, chunk * sizeof(float),
                    cudaMemcpyHostToDevice, streams[i]);
    process<<<chunk / 256, 256, 0, streams[i]>>>(dA + offset, chunk);
    cudaMemcpyAsync(hR + offset, dA + offset, chunk * sizeof(float),
                    cudaMemcpyDeviceToHost, streams[i]);
}
cudaDeviceSynchronize();      // 等全部流完成

for (int i = 0; i < nStreams; i++) cudaStreamDestroy(streams[i]);
```

注意这段代码的两个隐含设计：`hA`/`hR` 必须是锁页内存（条件 3）；每个流处理**互不重叠的数据分块**，块间没有依赖——这让流之间可以完全自由地交错。

### 6.3.3 时间线分析：收益从哪里来

```text
单流（串行）：
  H2D₀ H2D₁ H2D₂ H2D₃ → K₀ K₁ K₂ K₃ → D2H₀ D2H₁ D2H₂ D2H₃
  总时间 = 拷入 + 计算 + 拷出

4 流流水线（重叠）：
  流0:  H2D₀ K₀ D2H₀
  流1:       H2D₁ K₁ D2H₁
  流2:            H2D₂ K₂ D2H₂
  流3:                 H2D₃ K₃ D2H₃
  总时间 ≈ max(拷入, 计算, 拷出) + 启动/收尾开销
```

规律与经验值：

- **理论上限**：三个阶段耗时相当时收益最大，可逼近 **3 倍**；若某一阶段远长于其他（如计算占 90%），其他阶段完全藏进去，收益趋近于"省掉传输时间"；
- **分块数的权衡**：块越多流水越满（启动/收尾的"斜坡"占比越小），但每块太小会放大内核启动开销、且小内核可能吃不满 GPU——常见起点是 4~8 个流，用 profiler 调优；
- **首尾无法重叠**：第一块的拷入和最后一块的拷出注定露在流水线外，这是公式里"启动/收尾开销"的来源。

> [!TIP]
> 重叠是否真的发生了，不要凭感觉——用 `nsys profile` 打开时间线，亲眼确认 H2D、Kernel、D2H 三行是否错开重叠（6.9 节练习 1）。

### 6.3.4 提交顺序：深度优先还是广度优先

上面的代码按"**深度优先**"提交（一个流的三步发完，再发下一个流）。也可以"**广度优先**"：先发所有流的 H2D，再发所有内核，最后发所有 D2H。历史上这个选择很重要——Fermi 时代硬件只有一个任务队列，深度优先会造成"虚假依赖"卡死重叠，必须用广度优先绕开；**Kepler 引入 Hyper-Q（32 个硬件队列）后，两种顺序在现代 GPU 上通常都能正确重叠**。今天的建议：默认用逻辑清晰的深度优先，重叠效果一律以 `nsys` 时间线为准。

## 6.4 事件（Event）：计时与跨流依赖

多流并发带来一个新问题：如果流 B 的内核依赖流 A 的结果怎么办？总不能用 `cudaDeviceSynchronize` 把整个 GPU 停下来等。**事件（Event）**就是为此准备的轻量工具——它除了计时（3.9 节），还能做**跨流依赖**。

### 6.4.1 跨流依赖：cudaStreamWaitEvent

```c++
cudaEvent_t ev;
cudaEventCreate(&ev);

kernelA<<<grid, block, 0, streamA>>>(...);
cudaEventRecord(ev, streamA);              // 在 streamA 中打点
cudaStreamWaitEvent(streamB, ev, 0);       // streamB 等待该事件（GPU 侧等待，不阻塞 CPU）
kernelB<<<grid, block, 0, streamB>>>(...); // B 依赖 A 的结果

cudaEventDestroy(ev);
```

`cudaStreamWaitEvent` 是**设备侧**同步：CPU 不被阻塞，依赖关系由 GPU 调度器保证。它把"流 = 链条"升级成"流 + 事件 = 任意有向无环图（DAG）"——这是构建复杂任务图的基础，也是 CUDA Graph 的思想源头（专题 8）。

事件自己也有一套主机侧等待/查询：`cudaEventSynchronize(ev)`（阻塞 CPU 等到打点处）与 `cudaEventQuery(ev)`（非阻塞探询）——粒度比等待整条流更细。

### 6.4.2 纯同步用途：禁用计时的事件更便宜

事件默认会记录时间戳。如果只拿它做依赖同步、不需要计时，创建时加 `cudaEventDisableTiming` 标志可以省掉时间戳开销、获得更好的同步性能：

```c++
cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);   // 纯同步事件（更轻量）
```

PyTorch 等框架内部的跨流依赖用的正是这种事件（深度专题的专题 6）。

## 6.5 并发内核、主机回调与 CUDA Graph

重叠不止发生在"传输 × 计算"之间，本节把剩下几块并发拼图补齐。

### 6.5.1 并发内核执行

计算与计算之间也能重叠：计算资源没吃满的小内核，可以通过不同流**并发执行**（自 Fermi 起支持；Kepler 的 Hyper-Q 提供 32 个硬件队列，消除了单队列的虚假依赖；每设备可并发的内核数上限随架构从 16 到 128 不等，可查 `concurrentKernels` 属性确认支持）：

```c++
smallKernelA<<<8, 128, 0, streamA>>>(...);   // 两个小内核
smallKernelB<<<8, 128, 0, streamB>>>(...);   // 可在 GPU 上同时跑
```

注意：如果单个内核已经占满所有 SM，并发内核不会带来收益——它适用的是"多个小任务填不满 GPU"的场景（LLM 推理里通信内核与计算内核的重叠就是典型应用，见专题 9）。

### 6.5.2 主机回调：cudaLaunchHostFunc

有时想在"流执行到某一点"时通知 CPU 做点事（比如释放缓冲区、触发下一批数据准备）。`cudaLaunchHostFunc` 可以把一个**主机函数**插进流的队列，轮到它时由 CUDA 在后台线程调用：

```c++
void CUDART_CB onDone(void *userData) {     // 在流中排队的主机回调
    // 通知、记账、投递下一批任务……
}
cudaLaunchHostFunc(stream, onDone, &ctx);
```

> [!WARNING]
> 回调函数体内**禁止调用任何 CUDA API**（官方规定，否则可能死锁）；回调执行期间它后面的流操作会等待，所以回调要短小。

### 6.5.3 启动开销的终极解法：CUDA Graph（预告）

多流流水线解决了"引擎空转"，但每个操作仍要 CPU 逐个提交——当内核本身只有几微秒时（LLM 解码就是如此），**launch 开销反而成了主角**。CUDA Graph 允许把一整段"拷贝+内核+依赖"录制成图、一次提交整图重放，把成百上千次 launch 压缩成一次。完整讲解与 PyTorch 实验见深度专题的**专题 8**。

## 6.6 常见陷阱：并发是怎么被悄悄破坏的

多流代码"写了但没并发"是本章最高频的翻车现场。对照排查（前三个本章已讲，后三个由深度专题展开）：

| 陷阱 | 现象 | 解法 |
|------|------|------|
| 主机缓冲区是可分页内存 | `cudaMemcpyAsync` 退化同步，时间线上拷贝全串行 | 换 `cudaMallocHost`（6.2.3） |
| 有操作落进 legacy 默认流 | 一个默认流操作把所有流拦腰截断 | 全部显式指定流 / `per-thread`（6.2.4） |
| 粗粒度同步用得太勤 | 每次迭代 `cudaDeviceSynchronize`，流水线每轮清空 | 用流/事件粒度同步（6.2.2/6.4） |
| 热路径上调用隐式同步 API | `cudaMalloc`/`cudaFree`/同步版 `cudaMemcpy`/`cudaMemset` 等会隐式同步设备 | 预分配复用缓冲区（框架的显存池就是为此），完整触发清单见**专题 7** |
| 依赖关系用"睡等"实现 | CPU 轮询/sleep 代替事件 | `cudaStreamWaitEvent`（6.4.1） |
| launch 开销淹没小内核 | 时间线上 GPU 大段空白、CPU 忙着发射 | 合并小内核 / CUDA Graph（**专题 8**） |

## 6.7 主机-设备交互优化清单

流、异步拷贝、事件都学完了，把本章（连同第 1 章、第 5 章涉及的主机-设备交互知识）沉淀成一张可以对照检查的清单：

| 原则 | 说明 |
|------|------|
| **能不传就不传** | 数据尽量常驻 GPU；中间结果不要来回搬运 |
| **合并小传输** | 多次小 `cudaMemcpy` 合并成一次大传输（启动开销显著） |
| **频繁传输的缓冲区用锁页内存** | 带宽约提升 2 倍，且是异步拷贝的前提 |
| **用多流重叠传输与计算** | 分块流水线，隐藏传输时间（前提条件见 6.3.1） |
| **避免不必要的同步** | `cudaDeviceSynchronize` 会清空全部流水线，尽量用流/事件粒度的同步 |
| **纯同步事件加 DisableTiming** | 不需要计时的事件更轻量 |
| **延迟敏感的流设优先级** | `cudaStreamCreateWithPriority`，调度提示非抢占 |
| **用 Nsight Systems 看时间线** | `nsys profile` 可视化验证重叠是否真的发生 |

## 6.8 本章小结

- **异步是地基**：内核启动、`*Async` 拷贝（锁页内存前提）等操作发出即返回，CPU 与 GPU 两条时间线分离；GPU 上计算引擎与 H2D/D2H 拷贝引擎（`asyncEngineCount`）可同时开工；
- **流是排班表**：流内 FIFO 有序、流间可并发；默认流是全局"路障"，工程上应全部显式指定非默认流，必要时用 `cudaStreamNonBlocking` 与优先级；
- **多流流水线**是本章主菜：分块 ×"拷入-计算-拷出"，总时间从"三者之和"压到"约三者最大值"，上限约 3 倍；重叠四前提——拷贝引擎、（并发内核）、锁页内存、非默认流且无隐式同步；
- **事件**把流从"链条"升级为"DAG"：`cudaStreamWaitEvent` 设备侧跨流依赖不阻塞 CPU；纯同步用 `cudaEventDisableTiming` 更轻；
- **并发内核**填空隙、**主机回调**做通知（体内禁调 CUDA API）、**CUDA Graph** 治 launch 开销（专题 8）；
- 并发被破坏的六大陷阱见 6.6 节清单；一切重叠效果**以 `nsys` 时间线为准**，不要凭感觉。

## 6.9 深度专题导读与动手练习

> ⭐ 本章后半部分附有**[深度专题：CUDA Stream 与同步机制](#深度专题cuda-stream-与同步机制)**：系统讲解异步执行模型、三级同步原语、Default Stream 语义、PyTorch 的 Stream 实现、隐式同步陷阱与 CUDA Graph，并以 SGLang + Qwen3-0.6B 的真实 profiler trace 分析 LLM 推理中的多 Stream 架构（配套 PyTorch 实验代码位于 [`code/`](code/) 目录）。做深度学习系统方向的读者必读。

1. 实现 6.3 节的多流流水线（4 流分块处理 256 MB 数据），用 `nsys profile` 截图验证 H2D/Kernel/D2H 三行时间线确实重叠；
2. 把主机缓冲区从 `cudaMallocHost` 换回 `malloc`，重新 profile，观察"异步"拷贝退化为串行的现象；
3. 在练习 1 的基础上把提交顺序改成广度优先（三个独立循环），对比两种顺序的时间线差异（6.3.4 节）；
4. 用两个流分别启动一个小内核（各占 1/4 SM），验证并发内核执行；再把内核改大（占满 SM），观察并发消失；
5. 创建高/低优先级两条流各跑一串内核，用 `nsys` 观察高优先级流的块是否被优先调度（6.2.5 节）；
6. 阅读专题 7（隐式同步），在 PyTorch 中复现一次 `cudaStreamSynchronize` 引起的流水线气泡并消除它。

---

# 深度专题：CUDA Stream 与同步机制

> 以下为本章的深度进阶部分（原独立专题文档，已整合于此），配套实验代码位于 [`code/`](code/) 目录。

> 本专题系统介绍 CUDA 的异步执行模型、Stream 与同步机制：从 Host/Device 异步关系出发，依次讲解 Stream 的执行规则、三级显式同步原语、CUDA Event、Default Stream 语义、PyTorch 的 Stream 实现、隐式同步的触发条件与规避方法，以及 CUDA Graph；最后以 SGLang + Qwen3-0.6B 的真实 profiler trace 为案例，分析 LLM 推理中的多 Stream 架构与常见同步性能陷阱。
>
> 读完本专题后，你应当能够：读懂 `torch.profiler` trace 中的 `cudaStreamSynchronize`、`cudaEventSynchronize`、`cudaGraphLaunch` 等事件；定位隐式同步的根因；并在实际项目中写出不阻塞 GPU 流水线的代码。

### 如何阅读本专题

十二个专题分三层，难度递进。**如果你刚读完本章教程（6.1~6.9），专题 1~4 的大部分内容会似曾相识**——这是有意的：教程给的是"够用版"，这里给的是"精确版"（API 完整语义 + PyTorch 对照写法），可以快读，重点看表格和 PyTorch 代码：

| 层次 | 专题 | 内容定位 | 与教程的关系 |
|------|------|---------|-------------|
| 基础层（可快读） | 专题 1~4 | 异步模型、Stream、三级同步、Event 的**精确语义** | 教程 6.1/6.2/6.4 的精讲版 + PyTorch 对照 |
| 核心层（建议精读） | 专题 5~8 | Default Stream 精确语义、PyTorch Stream 设计、**隐式同步**（全专题最重要）、CUDA Graph | 教程 6.2.4/6.5.3/6.6 只点了名，这里展开 |
| 实战层（综合运用） | 专题 9~11 | 真实 LLM 推理 trace 分析、优化对照表、可运行实验 | 前面所有概念的实战检验 |

两个前置说明，帮你扫清阅读障碍：

- **trace（性能追踪文件）**：用 `torch.profiler`（或 `nsys`）录制程序运行，导出的 JSON 时间线文件。拖进 [Perfetto](https://ui.perfetto.dev) 打开后，每个 CPU 线程、每条 GPU Stream 各占一条横向"泳道"，一眼看清谁在忙、谁在等。本专题的大量结论直接来自 trace 证据，配套 trace 文件在 [`code/`](code/) 目录；
- **本专题的主角是"看不见的等待"**：显式同步是自己写的，一眼可见；真正吃性能的是**隐式同步**——代码里没有任何 `synchronize` 字样，trace 里却冒出几十毫秒的等待。专题 1~6 都在为看懂专题 7 做铺垫。

---

## 目录

- [专题 1 异步执行模型：Host 与 Device](#专题-1-异步执行模型host-与-device)
- [专题 2 CUDA Stream：有序操作队列](#专题-2-cuda-stream有序操作队列)
- [专题 3 显式同步：三级等待与非阻塞查询](#专题-3-显式同步三级等待与非阻塞查询)
- [专题 4 CUDA Event：Stream 中的标记点](#专题-4-cuda-eventstream-中的标记点)
- [专题 5 Default Stream 语义：NULL Stream 与 Blocking/Non-Blocking](#专题-5-default-stream-语义null-stream-与-blockingnon-blocking)
- [专题 6 PyTorch 的 Stream 实现](#专题-6-pytorch-的-stream-实现)
- [专题 7 隐式同步：触发条件、识别与规避](#专题-7-隐式同步触发条件识别与规避)
- [专题 8 CUDA Graph：录制与重放](#专题-8-cuda-graph录制与重放)
- [专题 9 案例研究：LLM 推理中的多 Stream 架构](#专题-9-案例研究llm-推理中的多-stream-架构)
- [专题 10 优化最佳实践](#专题-10-优化最佳实践)
- [专题 11 配套实验代码](#专题-11-配套实验代码)
- [专题 12 参考资料](#专题-12-参考资料)
- [附录：知识结构总图](#附录知识结构总图)

---

## 专题 1 异步执行模型：Host 与 Device

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

CPU 向 GPU 提交操作后**立即返回**，不等待操作执行，甚至不等待操作开始——官方（[Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)）的措辞比直觉更激进：**异步调用通常在被派发的操作完成之前返回，甚至可能在操作开始之前就返回**。

这带来两个直接推论，也是本专题全部内容的出发点：

1. **CPU 时间线与 GPU 时间线是分离的**——profiler trace 中 CPU 侧的 `cudaLaunchKernel` 与 GPU 侧的 kernel 执行往往相隔很远；
2. **凡是 CPU 需要使用 GPU 结果的地方，都必须引入同步**——同步用得对，开销可忽略；用错了，就是性能瓶颈。

后续专题围绕两个问题展开：GPU 上的操作按什么规则排队执行（Stream，专题 2）；CPU 与 GPU、GPU 各队列之间如何协调等待（同步机制，专题 3~4）。

---

## 专题 2 CUDA Stream：有序操作队列

### 2.1 定义

**CUDA Stream 是 GPU 上的一个有序操作队列**。程序向 Stream 提交操作（kernel 启动、异步内存拷贝等），这些操作在 Stream 内部按提交顺序依次执行——官方（[CUDA Streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams)）把它定义为"一个像工作队列（work-queue）一样运作的抽象：程序把拷贝、内核启动等操作加入队列，队列按序执行"。

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

## 专题 3 显式同步：三级等待与非阻塞查询

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

官方语义（[Explicit Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#explicit-synchronization)）：`cudaDeviceSynchronize` 等待**所有主机线程、所有 Stream** 中已提交的全部命令完成；`cudaStreamSynchronize` 只等待**指定 Stream** 中已提交的命令完成。

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

在 LLM 推理的 overlap scheduling 中（专题 9），CPU 正是通过 `cudaEventQuery` 轮询 GPU 状态，避免阻塞在等待上。

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

PyTorch 的 `stream_b.wait_stream(stream_a)` 是 `wait_event` 的便捷形式：等价于在 `stream_a` 的**当前末尾**打一个 Event 并让 `stream_b` 等待它——注意它等待的是 `stream_a` 上**已提交的全部工作**，粒度较粗（相关陷阱见专题 7、9）。

---

## 专题 4 CUDA Event：Stream 中的标记点

### 4.1 定义

CUDA Event 是**插入到 Stream 中的标记**。当 Stream 执行到该标记位置时，Event 变为完成（completed）状态。官方（[CUDA Events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-events)）打的比方很形象：Event 就像投进水流的**示踪粒子（tracer particle）**——粒子漂到哪，就说明水流（Stream 中的任务）推进到了哪。

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

## 专题 5 Default Stream 语义：NULL Stream 与 Blocking/Non-Blocking

教程 6.2.4 节说"默认流是全局路障"，本专题给出它的精确语义——搞清楚到底谁挡谁、哪些流不受影响，是读懂后面 PyTorch 设计（专题 6）的前提。

### 5.1 NULL Stream（Legacy Default Stream）

启动 kernel 或调用内存拷贝时若不指定 Stream，操作进入 **Legacy Default Stream**（又称 NULL Stream、stream 0）。官方定义（[Stream synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html)）：**legacy 默认流是一个隐式流，它与同一上下文中除 non-blocking 流以外的所有其他流互相同步**。

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

NULL Stream 的隐式同步是历史遗留设计，现代高性能代码应遵循：**要么显式指定 Non-Blocking Stream，要么启用 per-thread default stream，避免任何操作落入 Legacy NULL Stream**。下一个专题将看到，PyTorch 在框架层面彻底贯彻了这一原则。

---

## 专题 6 PyTorch 的 Stream 实现

为什么要专门讲 PyTorch？因为绝大多数深度学习读者不直接写 CUDA C，而是隔着 PyTorch 用 GPU——**框架在 Stream 上做了什么手脚，决定了你在 trace 里看到什么**。这一专题也是专题 5 的最佳应用案例：看一个工业级框架如何彻底规避 NULL Stream 的坑。

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

## 专题 7 隐式同步：触发条件、识别与规避

显式同步是自己写的，容易察觉；**隐式同步**则藏在框架和驱动内部——代码里没有任何 `synchronize` 字样，trace 里却出现了 `cudaStreamSynchronize`。这是实际项目中最常见的同步性能问题来源。

### 7.1 根源：Pageable 内存与 Pinned 内存

理解隐式同步，先要区分两种 Host 内存：

| 类型 | 分配方式 | 特点 | GPU DMA 可直接访问？ |
|------|----------|------|---------------------|
| **Pageable**（可分页） | 普通 `malloc` / `new` / 默认 tensor | 可能被 OS 换出到磁盘 | **否** |
| **Pinned**（锁页） | `cudaMallocHost` / `tensor.pin_memory()` | 锁定在物理内存 | **是** |

官方结论（[Launching memory transfers in CUDA streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#launching-memory-transfers-in-cuda-streams)）：**涉及 CPU 内存的拷贝要想异步执行，主机缓冲区必须是锁页内存；对非锁页内存调用 `cudaMemcpyAsync` 不会报错，但会退化为同步行为**。

机制解释：GPU 的 DMA 引擎只能访问物理地址固定的内存。Pageable 内存的页面可能随时被 OS 移动，驱动必须先把数据拷入内部的 pinned staging buffer（锁页中转缓冲区）再启动 DMA——这一过程要求与目标 Stream 同步。因此：

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

用一个比喻说透这个"连坐"机制：sync 等的是 stream48，但 stream48 说"我在等 stream20"，于是 CPU 被迫陪着等完 stream20 的全部 325 个 kernel——**等待沿着依赖链向上传递，代价由链条上最忙的那条 Stream 决定**。一次看似无害的小拷贝，阻塞了整条流水线。这正是配套实验 Case 4 与 Case 9 复现的 SGLang 真实案例。

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

## 专题 8 CUDA Graph：录制与重放

### 8.1 动机：CPU 提交开销

每次 kernel 启动都需要 CPU 调用 `cudaLaunchKernel`（约 5~20us）。当计算由大量小 kernel 组成、且同一模式反复执行时（典型如 LLM decode，每步几十个 kernel、执行上千次），CPU 提交开销会成为瓶颈：GPU 执行完一个 kernel 后，必须等 CPU 提交下一个，出现"GPU 等 CPU"的间隙。

### 8.2 原理：一次录制，多次重放

CUDA Graph 将一系列操作**录制**为图结构，之后每次执行只需一个 `cudaGraphLaunch` 调用。官方的定位（[CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#introduction-to-cuda-graphs-with-stream-capture)）：**捕获或创建图，可以把"从主机反复发起同一串 API 调用"的延迟与 CPU 开销降下来——描述操作的 API 只调用一次，得到的图可以执行任意多次**。

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

专题 9 的真实 trace 中，1,390 次 decode 迭代全部通过 `cudaGraphLaunch` 完成。

---

## 专题 9 案例研究：LLM 推理中的多 Stream 架构

本专题综合运用前面所有概念，分析本目录附带的真实 trace（SGLang 运行 Qwen3-0.6B）。

> [!NOTE]
> 先解释两个贯穿本专题的 LLM 推理术语（不做深度学习方向的读者也能看懂案例）：
> - **Prefill（预填充）**：处理用户输入的整段 prompt，一次算完所有输入 token——计算量大、但只做一次，kernel 逐个常规提交；
> - **Decode（解码）**：逐个生成输出 token，每生成一个 token 跑一遍模型——单步计算量小、但要重复成百上千次，正是专题 8 所说"launch 开销成为主角"的场景，因此全部走 CUDA Graph。

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

跨 Stream 协调全部使用专题 3~4 的原语：`wait_stream` 保证前向读到就绪的输入；`Event.record + synchronize/query` 让 CPU 在恰当的时刻取回结果。

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
| `cudaGraphLaunch` | 1,390 | decode 迭代重放（专题 8） |
| `cudaStreamWaitEvent` | 74 | forward 等 schedule 的输入就绪（专题 3.4 节） |
| `cudaEventSynchronize` | 74 | CPU 取回 prefill 结果（专题 4.3 节用途 2） |
| `cudaEventQuery` | 383 | CPU 非阻塞轮询，实现 overlap（专题 3.3 节） |
| `cudaStreamSynchronize` | 378 | 多为隐式触发，大部分 <10us（专题 7） |

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

trace 中曾出现单次 29ms 的 `cudaStreamSynchronize`，根因是专题 7.3 节描述的放大链条在真实代码中的体现：

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

## 专题 10 优化最佳实践

专题 7~9 看过了各种翻车现场，本专题把修复手法沉淀成"问题 → 方案"对照表和可以直接抄的代码模板。

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

## 专题 11 配套实验代码

纸上得来终觉浅——本专题的十个渐进式实验覆盖前面全部知识点，每个用例都会生成自己的 trace，强烈建议边跑边用 Perfetto 对照观察。

### 11.1 目录结构

```
code/
├── test_cuda_stream_sync_tutorial.py      # 10 个渐进式实验用例
└── Qwen3-0.6B-*.trace.json.gz             # SGLang 真实 profiler trace（专题 9 数据来源）
```

### 11.2 实验用例与章节对应

| Case | 主题 | 验证的知识点 | 对应章节 |
|------|------|-------------|---------|
| 1 | Pageable 内存隐式 Sync | `.to(device)` 为何触发 sync，及 sync 耗时的放大 | 专题 7 |
| 2 | Pinned 内存消除 Sync | pin_memory 实现真异步拷贝 | 专题 7.1 节 |
| 3 | `non_blocking` 标志行为 | 三种组合的实测耗时差异 | 专题 7.2 节 |
| 4 | 跨 Stream 依赖 | `wait_stream` 如何放大 sync 耗时 | 专题 7.3 节 |
| 5 | `.item()/.cpu()` 陷阱 | 循环内取标量导致串行化 | 专题 10.2 节 |
| 6 | `torch.tensor(device='cuda')` 陷阱 | 隐藏的 Pageable 拷贝与 sync | 专题 7.2 节 |
| 7 | NULL Stream 行为 | PyTorch 如何规避隐式同步 | 专题 5~6 |
| 8 | Event 细粒度同步 | 四种同步粒度的对比 | 专题 3~4 |
| 9 | Overlap Scheduling 模式 | 问题模式与优化模式对照 | 专题 9 |
| 10 | CUDA Graph 加速 | 录制-重放消除提交开销 | 专题 8 |

### 11.3 运行方式

```bash
# 需要 CUDA GPU 环境
cd code/

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

## 专题 12 参考资料

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

最后用一张图串起全专题的知识体系——建议对照它自测：每个方框能不能用自己的话讲出来？

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

---

> [← 上一章：内存模型](../05_memory_model/README.md) | [返回目录](../README.md) | [下一章：原子操作与 Warp 级原语 →](../07_atomics_and_warp/README.md)
