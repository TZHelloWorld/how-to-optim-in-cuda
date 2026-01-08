# CUDA Reduce 算子优化指南

> 本文以 Reduce（归约）算子为例，系统介绍 CUDA kernel 的典型优化方法。从最朴素的基准实现出发，沿着"发现瓶颈 → 分析原理 → 针对性优化"的主线，逐步演进出 8 个版本（V0~V7），最终逼近硬件带宽的理论峰值；并进一步介绍如何将自定义 kernel 集成到 PyTorch，包括算子注册与 dtype 分发机制。

---

## 目录

- [第 1 章 问题定义：什么是 Reduce](#第-1-章-问题定义什么是-reduce)
- [第 2 章 预备知识：GPU 执行模型与内存层次](#第-2-章-预备知识gpu-执行模型与内存层次)
- [第 3 章 优化路线总览](#第-3-章-优化路线总览)
- [第 4 章 V0：基准实现——朴素树形归约](#第-4-章-v0基准实现朴素树形归约)
- [第 5 章 V1：消除 Warp Divergence——连续线程映射](#第-5-章-v1消除-warp-divergence连续线程映射)
- [第 6 章 V2：消除 Bank Conflict——顺序寻址](#第-6-章-v2消除-bank-conflict顺序寻址)
- [第 7 章 V3：提高线程利用率——加载时首次归约](#第-7-章-v3提高线程利用率加载时首次归约)
- [第 8 章 V4：减少同步开销——展开最后一个 Warp](#第-8-章-v4减少同步开销展开最后一个-warp)
- [第 9 章 V5：消除循环开销——模板完全展开](#第-9-章-v5消除循环开销模板完全展开)
- [第 10 章 V6：绕过共享内存——Warp Shuffle 两级归约](#第-10-章-v6绕过共享内存warp-shuffle-两级归约)
- [第 11 章 V7：榨干带宽——float4 向量化与 Grid Stride Loop](#第-11-章-v7榨干带宽float4-向量化与-grid-stride-loop)
- [第 12 章 工程化：多级归约与 PyTorch 扩展](#第-12-章-工程化多级归约与-pytorch-扩展)
- [第 13 章 PyTorch 算子注册机制](#第-13-章-pytorch-算子注册机制)
- [第 14 章 PyTorch 的 dtype 分发机制](#第-14-章-pytorch-的-dtype-分发机制)
- [第 15 章 总结与实践建议](#第-15-章-总结与实践建议)
- [附录：关键概念速查](#附录关键概念速查)

---

## 第 1 章 问题定义：什么是 Reduce

### 1.1 Reduce 的含义

Reduce 指用一个满足结合律的二元运算，将一组数据归约为单个值。常见形式包括求和、求最大/最小值、求乘积等。在 CPU 上，它只是一个简单的串行循环：

```c
float sum = 0;
for (int i = 0; i < n; i++) {
    sum += data[i];
}
```

Reduce 是深度学习与高性能计算中最基础的算子之一：`sum`、`mean`、`softmax` 的分母、`LayerNorm` 的均值方差，本质都是 Reduce。当数据量达到百万甚至千万级时，串行累加成为瓶颈，需要借助 GPU 的数千个线程并行完成。

### 1.2 并行归约的核心思路：树形归约

并行化 Reduce 的难点在于：数千个线程同时计算，最终却只需要一个结果。标准解法是**树形归约（Tree Reduction）**——每一轮将数据两两合并，规模减半，经过 log₂(N) 轮收敛为一个值：

```
 a0   a1   a2   a3   a4   a5   a6   a7
  \  /      \  /      \  /      \  /      ← 第 1 轮：8 → 4
 a0+a1    a2+a3    a4+a5    a6+a7
    \      /          \      /            ← 第 2 轮：4 → 2
  a0~a3              a4~a7
      \              /                    ← 第 3 轮：2 → 1
        a0~a7 (最终结果)
```

8 个数只需 3 轮，1024 个数只需 10 轮，复杂度从 O(N) 降至 O(log N)。

### 1.3 GPU 上的三段式流程

由于线程只能在 Block 内部通过共享内存高效通信，一个典型的 Block 级 Reduce kernel 分三步：

| 步骤 | 内容 | 说明 |
|------|------|------|
| 加载（Load） | 每个线程从全局内存读取数据到共享内存 | 全局内存 → 共享内存 |
| 归约（Reduce） | Block 内做树形归约 | 每轮活跃线程减半 |
| 写回（Store） | 线程 0 将 Block 局部和写回全局内存 | 每个 Block 产出一个部分和 |

各 Block 的部分和还需要进一步汇总（见第 12 章的多级归约）。本文的 V0~V7 主要优化 Block 内部的归约过程与数据加载方式。

---

## 第 2 章 预备知识：GPU 执行模型与内存层次

理解后续优化的前提，是掌握以下四个概念。

### 2.1 线程层次：Grid → Block → Thread

CUDA 将线程组织为三层结构：

```
Grid（网格）
 ├── Block 0（线程块，例如 256 线程）
 │    ├── Thread 0   (threadIdx.x = 0)
 │    ├── ...
 │    └── Thread 255 (threadIdx.x = 255)
 ├── Block 1
 └── ...
```

- **Thread**：最小执行单位；
- **Block**：一组线程，可通过共享内存通信、通过 `__syncthreads()` 同步；
- **Grid**：一次 kernel 启动的全部 Block，Block 之间无法直接同步。

常用内置变量：

| 变量 | 含义 |
|------|------|
| `threadIdx.x` | 线程在 Block 内的编号 |
| `blockIdx.x` | Block 在 Grid 中的编号 |
| `blockDim.x` | 每个 Block 的线程数 |
| `gridDim.x` | Grid 中的 Block 数 |

线程的全局编号计算为 `gid = blockIdx.x * blockDim.x + threadIdx.x`。

### 2.2 Warp：真正的硬件调度单位

GPU 硬件以 **Warp（32 个连续线程）** 为单位调度执行，同一 Warp 内的线程**锁步执行同一条指令**（SIMT 模型）。由此引出两条重要推论：

1. **Warp Divergence（线程束分化）**：若同一 Warp 内的线程走了不同分支，硬件必须依次执行所有分支路径（不参与的线程结果被屏蔽），分支越多性能越差。闲置线程并不省时间——它们只是结果被丢弃。
2. **Warp 内天然同步**：同一 Warp 的 32 个线程不需要显式同步就步调一致，这是 V4、V6 优化的理论基础。

一个 256 线程的 Block 包含 8 个 Warp：Warp 0 = tid 0~31，Warp 1 = tid 32~63，以此类推。

### 2.3 内存层次

```
速度：   慢 ──────────────────────────────────────────── 快
         全局内存(Global)  →  共享内存(Shared)  →  寄存器(Register)
大小：   几 GB              几十 KB / Block       几 KB / 线程
延迟：   ~400-600 cycles    ~20-30 cycles         ~1 cycle
可见性： 所有线程           同一 Block 内         仅本线程
```

优化的总体方向是：**让数据尽量在更快的层次上流转**——V0~V5 依赖共享内存，V6 进一步把归约搬进寄存器。

### 2.4 Shared Memory 的 Bank 结构

共享内存被划分为 **32 个 Bank**，地址按 4 字节交错分布：地址 `i`（以 float 计）属于 `bank = i % 32`。

```
地址:  0  1  2  ... 31 32 33 ... 63 64 ...
bank:  0  1  2  ... 31  0  1 ... 31  0 ...
```

同一 Warp 内的多个线程若同时访问**同一 Bank 的不同地址**，访问会被串行化，称为 **Bank Conflict**。n 个线程冲突则退化为 n 次串行访问（n-way conflict）。

---

## 第 3 章 优化路线总览

在深入代码之前，先给出全文的优化主线。每个版本都针对上一版暴露的一个具体瓶颈：

```
V0: 朴素树形归约（基准）
 │   瓶颈：取模选线程 → 活跃线程分散 → Warp Divergence 严重
 ▼
V1: Strided Index，让活跃线程连续
 │   瓶颈：跨步访问共享内存 → Bank Conflict
 ▼
V2: 步长从大到小 + tid < s，一并消除 Divergence 与 Bank Conflict
 │   瓶颈：一半线程只搬一次数据就闲置
 ▼
V3: 每线程加载 2 个元素并预先相加
 │   瓶颈：归约进入单 Warp 后，__syncthreads() 全部多余
 ▼
V4: 手动展开最后一个 Warp，省去 5~6 次同步
 │   瓶颈：剩余循环的控制开销与运行时分支
 ▼
V5: 模板参数 BLOCK_SIZE，编译期完全展开
 │   瓶颈：归约始终经过共享内存，延迟高于寄存器
 ▼
V6: Warp Shuffle 寄存器直传 + 两级归约
 │   瓶颈：加载粒度小（4 字节/次），Block 数量过多
 ▼
V7: float4 向量化加载 + Grid Stride Loop（带宽接近理论峰值）
```

对应的性能瓶颈分类：

| 瓶颈类别 | 具体问题 | 解决版本 |
|---------|---------|---------|
| 计算效率 | Warp Divergence | V1 / V2 |
| 计算效率 | 循环与分支开销 | V4 / V5 |
| 内存效率 | Bank Conflict | V2 |
| 内存效率 | 共享内存往返延迟 | V6 |
| 内存效率 | 全局内存带宽利用率 | V7 |
| 资源利用率 | 线程闲置 | V3 |
| 资源利用率 | 同步开销 | V4 |
| 资源利用率 | Block 数量过多 | V7 |

值得强调的是：Reduce 是典型的**访存受限（memory-bound）**算子，计算量（N 次加法）远小于访存量（N 次读取）。因此优化的终极目标不是提高算力利用率，而是**让全局内存带宽跑满**——这正是 V7 的方向。

---

## 第 4 章 V0：基准实现——朴素树形归约

### 4.1 实现

V0 直接把第 1 章的树形归约翻译成 CUDA 代码：

```cuda
__global__ void reduce_v0(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. 加载：每线程搬运一个元素到共享内存
    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // 2. 归约：步长从 1 开始逐轮翻倍
    for (int step = 1; step < blockDim.x; step *= 2) {
        if (tid % (2 * step) == 0) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }

    // 3. 写回：每个 Block 产出一个部分和
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}
```

几点说明：

- `extern __shared__ float smem[]`：动态共享内存，大小在 kernel 启动时由第三个配置参数指定（`<<<grid, block, smem_bytes>>>`）；
- 越界线程写入 `0.0f`，对求和无影响；
- 每轮归约后必须 `__syncthreads()`，保证本轮所有写入对下一轮可见。

### 4.2 执行过程

以 8 线程为例，条件 `tid % (2*step) == 0` 决定每轮的活跃线程：

```
初始:  smem = [a0, a1, a2, a3, a4, a5, a6, a7]

step=1（tid % 2 == 0，活跃 tid: 0,2,4,6）:
  smem[0]+=smem[1]  smem[2]+=smem[3]  smem[4]+=smem[5]  smem[6]+=smem[7]
  → [a0+a1, ·, a2+a3, ·, a4+a5, ·, a6+a7, ·]

step=2（tid % 4 == 0，活跃 tid: 0,4）:
  smem[0]+=smem[2]  smem[4]+=smem[6]
  → [a0~a3, ·, ·, ·, a4~a7, ·, ·, ·]

step=4（tid % 8 == 0，活跃 tid: 0）:
  smem[0]+=smem[4]
  → smem[0] = a0+a1+...+a7 ✓
```

### 4.3 瓶颈分析

V0 功能正确，但存在三个问题，它们分别驱动了后续版本的演进：

**问题 1：Warp Divergence 严重。** 活跃线程由取模条件选出，在 Warp 内呈交错分布。以 step=1 为例，一个 Warp 内 tid 为偶数的 16 个线程干活、奇数的 16 个闲置，且彼此交错：

```
tid:  0  1  2  3  4  5  ... 30 31
      ✓  ✗  ✓  ✗  ✓  ✗      ✓  ✗    ← 活跃/闲置交错，Warp 内部分化
```

由于 Warp 锁步执行，这种交错意味着每个 Warp 都要为分支付出代价——闲置线程占着执行资源却不产出结果。

**问题 2：Bank Conflict。** 随着 step 增大，活跃线程对共享内存的访问间距拉大，会出现多个线程命中同一 Bank 的情况（详见第 6 章的分析）。

**问题 3：线程利用率低。** 每个线程只加载 1 个元素，且第一轮归约后就有一半线程永久闲置。处理 100 万元素需要约 4000 个 Block，产生大量需要二次归约的部分和。

---

## 第 5 章 V1：消除 Warp Divergence——连续线程映射

### 5.1 优化思路

V0 的问题出在"选人方式"：用取模条件筛选活跃线程，选出的线程编号是分散的。V1 换一种映射：**不筛选线程编号，而是让编号连续的前若干个线程，通过乘法计算出各自要操作的位置**：

```
V0: tid 能被 2s 整除吗？ 能 → 操作 smem[tid]
V1: index = tid * 2s；   index 合法 → 操作 smem[index]
```

### 5.2 实现

```cuda
__global__ void reduce_v1(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = threadIdx.x * 2 * s;   // 连续 tid 映射到分散的位置
        if (index < blockDim.x) {
            smem[index] += smem[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}
```

### 5.3 效果：活跃线程变为连续段

同样是 8 线程的第一轮（s=1，index = tid×2）：

```
tid=0 → index=0: smem[0]+=smem[1]  ✓
tid=1 → index=2: smem[2]+=smem[3]  ✓
tid=2 → index=4: smem[4]+=smem[5]  ✓
tid=3 → index=6: smem[6]+=smem[7]  ✓
tid=4~7 → index ≥ 8，闲置          ✗
```

归约的数学过程与 V0 完全相同，但活跃线程从"交错分布"变成了"连续的前半段"。以 blockDim=256、s=1 为例：

```
V0:  tid 0~255 中偶数活跃、奇数闲置 → 8 个 Warp 全部分化
V1:  tid 0~127 全部活跃，tid 128~255 全部闲置
     → Warp 0~3 全员干活，Warp 4~7 全员闲置，没有任何 Warp 内部分化
```

整 Warp 闲置是无害的——硬件可以直接跳过它们去调度别的工作；有害的是 Warp **内部**分化。V1 将后者基本消除。

### 5.4 遗留问题：Bank Conflict 反而加重

V1 的写位置是 `index = tid * 2s`，随着 s 增大，相邻线程的访问间距成倍拉大。以 s=16 为例：

```
tid=0 → 访问 smem[0]  （bank 0）
tid=1 → 访问 smem[32] （bank 32%32 = 0）  ← 与 tid=0 冲突！
```

相邻线程以 32 的倍数间隔访问共享内存，大量命中同一 Bank，形成多路 Bank Conflict。这一问题交给 V2 解决。

---

## 第 6 章 V2：消除 Bank Conflict——顺序寻址

### 6.1 优化思路

V0/V1 的步长都是**从小到大**（1→2→4→...），V2 将方向反转为**从大到小**（blockDim/2 → ... → 2 → 1），并用最简洁的条件选择活跃线程：

```cuda
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        smem[tid] += smem[tid + s];
    }
    __syncthreads();
}
```

> `s >>= 1` 为右移一位，等价于 `s = s / 2`，是 CUDA 代码的惯用写法。

直观地看，每一轮相当于把数组**对折**：前半段累加后半段，有效数据规模减半。

### 6.2 实现

```cuda
__global__ void reduce_v2(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // 步长从 blockDim.x/2 开始，每轮减半
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}
```

以 8 线程为例：

```
初始:  [a0, a1, a2, a3, a4, a5, a6, a7]

s=4（tid<4 活跃）: [a0+a4, a1+a5, a2+a6, a3+a7 | a4, a5, a6, a7]  ← 对折
s=2（tid<2 活跃）: [a0+a2+a4+a6, a1+a3+a5+a7 | ...]              ← 再对折
s=1（tid<1 活跃）: smem[0] = 全部之和 ✓
```

### 6.3 为什么一个改动解决了两个问题

**（1）Warp Divergence：** 条件 `tid < s` 保证活跃线程永远是连续的前 s 个。以 blockDim=256 为例：

| 轮次 | s | 活跃线程 | Warp 状态 |
|------|-----|----------|-----------|
| 1 | 128 | tid 0~127 | Warp 0~3 全活跃，Warp 4~7 全闲置 |
| 2 | 64 | tid 0~63 | Warp 0~1 全活跃 |
| 3 | 32 | tid 0~31 | Warp 0 全活跃 |
| 4~8 | 16~1 | tid 0~15 起 | 仅剩单 Warp 内部（不可避免的尾部）|

**（2）Bank Conflict：** 活跃线程 tid 访问 `smem[tid]` 与 `smem[tid+s]`，两类地址均连续排列。

- 当 `s ≥ 32` 时，Warp 内 32 个线程访问 `smem[tid]`（tid=0~31），恰好落在 bank 0~31 各一个；访问 `smem[tid+s]` 同理（s 是 32 的倍数时偏移不改变 bank 分布的互异性）。零冲突。
- 当 `s < 32` 时，活跃线程数少于 32，`tid` 与 `tid+s` 模 32 必然不同（差值 s ∈ [1,31]），仍然零冲突。

**顺序寻址（sequential addressing）是共享内存访问的黄金法则**：让相邻线程访问相邻地址，Bank 冲突自然消失。

### 6.4 V0 / V1 / V2 对比

| 维度 | V0 | V1 | V2 |
|------|----|----|-----|
| 步长方向 | 小→大 | 小→大 | **大→小** |
| 活跃线程选择 | `tid%(2s)==0`（分散） | `tid*2s < N`（连续） | **`tid < s`（连续）** |
| Warp Divergence | 严重 | 已消除 | 已消除 |
| Bank Conflict | 有 | 有（更严重） | **无** |
| 代码复杂度 | 中 | 中 | **最简洁** |

### 6.5 遗留问题

V2 的归约阶段已经很干净，但注意第一轮（s=128）：tid 128~255 这一半线程**只在加载阶段搬了一次数据，之后全程闲置**。一半的线程资源仅用作"搬运工"，这是 V3 要解决的问题。

---

## 第 7 章 V3：提高线程利用率——加载时首次归约

### 7.1 优化思路

既然后半线程只承担一次搬运，不如直接砍掉它们，让保留下来的每个线程**加载 2 个元素并在寄存器中预先相加**。等价地说：把归约树的第一层合并进加载阶段。

```
V2：256 线程，每线程搬 1 个元素 → Block 处理 256 个元素
V3：256 线程，每线程搬 2 个元素 → Block 处理 512 个元素
```

### 7.2 实现

```cuda
__global__ void reduce_v3(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;  // 每 Block 覆盖 2×blockDim

    // 加载 2 个相距 blockDim.x 的元素，先在寄存器里相加
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    // 归约循环与 V2 相同
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}
```

注意两个元素的间距是 `blockDim.x` 而不是 1：这保证同一时刻 Warp 内 32 个线程访问的全局内存地址仍然连续，满足**合并访存（coalesced access）**的要求。

### 7.3 效果分析

以处理 8 个元素为例对比：

| | V2 | V3 |
|--|----|----|
| 所需线程数 | 8 | **4** |
| 共享内存占用 | 8 个 float | **4 个 float** |
| 归约轮数 | 3 | **2** |
| `__syncthreads()` 次数 | 3 | **2** |

收益来自三方面：

1. **Block 数量减半**——处理同样的数据，Grid 规模和二次归约的部分和数量都减半；
2. **消除"搬完即闲"**——每个线程从一开始就承担实际计算；
3. **少一轮归约**——少一次同步栅栏的等待开销。

这一思想可以推广：每线程处理 4 个、8 个元素通常还能进一步提速（代价是 Block 数减少可能影响占用率），V7 的 Grid Stride Loop 是它的一般化形式。

---

## 第 8 章 V4：减少同步开销——展开最后一个 Warp

### 8.1 瓶颈：单 Warp 阶段的同步全是多余的

观察 V3 的归约循环（blockDim=256）：

```
s=128: 活跃 tid 0~127（跨 4 个 Warp）→ __syncthreads() 必要
s=64:  活跃 tid 0~63 （跨 2 个 Warp）→ __syncthreads() 必要
s=32:  活跃 tid 0~31 （单 Warp）    → 同步多余！
s=16 / 8 / 4 / 2 / 1                → 同步全部多余！
```

当 `s ≤ 32` 时，活跃线程全部位于 Warp 0 内。根据第 2 章的结论，**Warp 内线程锁步执行，天然同步**，后 6 轮的 `__syncthreads()`（以及循环判断和 `if` 分支）纯属浪费。

### 8.2 优化思路：循环只跑到 s > 32，尾部手写展开

```cuda
// 辅助函数：单 Warp 内的归约，无需任何同步
__device__ void warpReduce(volatile float* smem, int tid) {
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid +  8];
    smem[tid] += smem[tid +  4];
    smem[tid] += smem[tid +  2];
    smem[tid] += smem[tid +  1];
}

__global__ void reduce_v4(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // 加载阶段继承 V3
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    // 循环仅执行到 s > 32（blockDim=256 时只有 2 轮）
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    // 最后 64 → 1 由单 Warp 完成，无循环、无分支、无同步
    if (tid < 32) {
        warpReduce(smem, tid);
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}
```

注意 `if (tid < 32)` 进入后，`warpReduce` 的第一步 `smem[tid] += smem[tid+32]` 处理的是 64→32，因此循环的终止条件是 `s > 32` 而非 `s > 16`。

### 8.3 关键细节：为什么参数必须是 `volatile`

去掉 `__syncthreads()` 后，线程间的数据可见性完全依赖"每次读写都真实落到共享内存"。若不加 `volatile`，编译器可能把 `smem[tid]` 缓存进寄存器做累加、最后才写回——期间其他线程读到的就是过期值，结果错误。

`volatile` 强制编译器：每次 `+=` 立即写回共享内存，每次读取都从共享内存取最新值。

> 说明：在现代 CUDA（9.0+）中，更规范的做法是使用 `__syncwarp()` 或直接改用 Warp Shuffle（见 V6）。`volatile` 写法源自经典的 NVIDIA reduction 教程，此处保留以呈现完整的演进脉络。

### 8.4 效果

| | V3 | V4 |
|--|----|----|
| `__syncthreads()`（blockDim=256）| 7 次 | **2 次** |
| 循环轮数 | 7 | 2 轮循环 + 6 行直线代码 |
| 尾部分支判断 | 每轮 `if (tid<s)` | **无** |

---

## 第 9 章 V5：消除循环开销——模板完全展开

### 9.1 瓶颈：运行时循环无法被编译器优化掉

V4 剩下的 for 循环虽然只有 2 轮，但 `blockDim.x` 是**运行时值**，编译器无法在编译期确定循环次数，只能生成通用的循环控制代码（条件判断、右移、跳转）。

### 9.2 优化思路：把 Block 大小提升为编译期常量

利用 C++ 模板，将 Block 大小作为模板参数传入：

```cuda
template <int BLOCK_SIZE>   // 编译期常量
__global__ void reduce_v5(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;

    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + BLOCK_SIZE < n) val += input[gid + BLOCK_SIZE];
    smem[tid] = val;
    __syncthreads();

    // 所有条件均在编译期求值，false 分支被整段删除
    if (BLOCK_SIZE >= 512) { if (tid < 256) smem[tid] += smem[tid + 256]; __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (tid <  64) smem[tid] += smem[tid +  64]; __syncthreads(); }

    if (tid < 32) {
        volatile float* vsmem = smem;
        if (BLOCK_SIZE >= 64) vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if (tid == 0) output[blockIdx.x] = smem[0];
}
```

### 9.3 编译器视角

以 `reduce_v5<256>` 为例，`BLOCK_SIZE >= 512` 在编译期即为 false，整段代码被死代码消除（dead code elimination）；其余条件为 true，外层 `if` 被直接去掉。最终生成的机器码等价于：

```cuda
if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads();
if (tid <  64) smem[tid] += smem[tid +  64]; __syncthreads();
// 直接进入 warp 展开部分
```

**零循环、零运行时多余分支**。不同的模板实参各自实例化一份最优代码：

| BLOCK_SIZE | 保留的归约步骤 |
|-----------|--------------|
| 512 | +256, +128, +64, Warp 展开 |
| 256 | +128, +64, Warp 展开 |
| 128 | +64, Warp 展开 |
| 64  | 仅 Warp 展开 |

### 9.4 调用方式与代价

模板参数必须是编译期常量，因此宿主端需要 switch 选择实例：

```cuda
switch (block_size) {
    case 512: reduce_v5<512><<<grid, 512, 512 * sizeof(float)>>>(d_in, d_out, n); break;
    case 256: reduce_v5<256><<<grid, 256, 256 * sizeof(float)>>>(d_in, d_out, n); break;
    case 128: reduce_v5<128><<<grid, 128, 128 * sizeof(float)>>>(d_in, d_out, n); break;
}
```

代价是**代码膨胀**：每种 BLOCK_SIZE 编译一份独立机器码。对于性能库而言，这是普遍接受的权衡（PyTorch、cuDNN 内部大量使用同类技巧）。

---

## 第 10 章 V6：绕过共享内存——Warp Shuffle 两级归约

### 10.1 瓶颈：归约数据始终在共享内存中往返

V0~V5 的每一轮归约都是"读共享内存 → 加 → 写共享内存"。共享内存延迟约 20~30 周期，而寄存器只要约 1 周期。如果线程之间能**直接交换寄存器中的值**，就能跳过共享内存这一层。

Kepler（SM 3.0）起，硬件提供了这一能力：**Warp Shuffle 指令**。

### 10.2 核心原语：`__shfl_down_sync`

```cuda
float v = __shfl_down_sync(0xffffffff, val, offset);
```

语义：在同一 Warp 内，每个线程获得 **lane 编号比自己大 offset 的那个线程的 `val` 值**，整个交换在寄存器级完成、不经过内存。参数依次为：参与掩码（`0xffffffff` 表示全部 32 个 lane）、要传递的值、偏移量。

基于它可以写出 5 步完成的 Warp 内归约：

```cuda
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;   // lane 0 持有 32 个线程的总和
}
```

执行过程（以 8 lane 简化示意，实际 offset 从 16 开始）：

```
初始:      lane:  0    1    2    3    4    5    6    7
           val: [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7]

offset=4:  lane0: a0+a4   lane1: a1+a5   lane2: a2+a6   lane3: a3+a7
offset=2:  lane0: a0+a2+a4+a6            lane1: a1+a3+a5+a7
offset=1:  lane0: 全部之和 ✓
```

高位 lane 的返回值未定义，但不影响 lane 0 收敛出正确结果。Shuffle 延迟仅 1~2 周期，比共享内存快一个数量级。

### 10.3 两级归约架构

Shuffle 只能在 Warp 内部（32 线程）工作，一个 Block 有多个 Warp，因此需要两级：

```
第一级：每个 Warp 内部 shuffle 归约 → 各 Warp 的 lane 0 得到本 Warp 部分和
中转：  各 lane 0 把部分和写入共享内存 warp_results[]（仅此一次使用共享内存）
第二级：Warp 0 读取这些部分和，再做一次 shuffle 归约 → tid 0 得到 Block 总和
```

```
┌────────────────────────────────────────────────────────┐
│              Block（256 线程 = 8 个 Warp）              │
│                                                        │
│  Warp0     Warp1     Warp2   ...   Warp7               │
│  shuffle   shuffle   shuffle       shuffle   ← 第一级  │
│    ↓          ↓         ↓             ↓      （寄存器）│
│  lane0     lane0     lane0         lane0               │
│    └──────────┴─────────┴─────────────┘                │
│              warp_results[0..7]              ← 写 smem │
│                     ↓                                  │
│     Warp 0 再做一次 shuffle 归约              ← 第二级  │
│                     ↓                                  │
│           tid 0 = Block 总和                           │
└────────────────────────────────────────────────────────┘
```

### 10.4 实现

```cuda
__global__ void reduce_v6(float* input, float* output, int n) {
    int tid  = threadIdx.x;
    int gid  = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int lane = tid % 32;   // Warp 内编号
    int wid  = tid / 32;   // Warp 编号

    // 加载阶段继承 V3
    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];

    // 第一级：Warp 内归约（寄存器直传）
    val = warpReduceSum(val);

    // 中转：各 Warp 的部分和写入共享内存
    __shared__ float warp_results[32];   // 最多 32 个 Warp（1024 线程）
    if (lane == 0) {
        warp_results[wid] = val;
    }
    __syncthreads();                     // 整个 kernel 唯一一次 Block 级同步

    // 第二级：Warp 0 汇总所有 Warp 的部分和
    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        val = warpReduceSum(val);
    }

    if (tid == 0) output[blockIdx.x] = val;
}
```

### 10.5 与 V5 的对比

| 维度 | V5 | V6 |
|------|----|----|
| 数据交换方式 | 共享内存反复读写 | **寄存器直传（Shuffle）** |
| 共享内存用量 | blockDim × 4 字节 | **固定 128 字节** |
| `__syncthreads()` | 2~3 次 | **1 次** |
| 是否需要 volatile | 是 | **否**（不经过内存） |
| 是否需要模板 | 是 | **否**（对 blockDim 通用） |
| 硬件要求 | 任意 | SM 3.0+ |

V6 结构更简单、通用性更好，是现代 CUDA 代码（包括 PyTorch 内部实现）做 Block 级归约的标准写法。

---

## 第 11 章 V7：榨干带宽——float4 向量化与 Grid Stride Loop

### 11.1 瓶颈：归约已近最优，加载成为短板

V6 之后，Block 内归约的开销已降至很低。回到"Reduce 是 memory-bound 算子"这一本质：性能上限由**全局内存带宽利用率**决定。V6 的加载阶段还有两个短板：

1. **加载粒度小**：每条加载指令只取 1 个 float（4 字节），指令数多，难以喂饱 128 字节宽的内存事务；
2. **Block 数量随数据量线性增长**：16M 元素、blockDim=256 时需要 32768 个 Block，调度开销大，且留下 32768 个部分和等待二次归约。

V7 用两个手段分别解决。

### 11.2 优化 1：float4 向量化加载

`float4` 是 CUDA 内置的 16 字节向量类型（成员 `.x .y .z .w`）。将 `float*` 重新解释为 `float4*`，一条 `LDG.128` 指令即可加载 4 个 float：

```cuda
float4* input4 = reinterpret_cast<float4*>(input);
int n4 = n / 4;                       // 完整 float4 的个数

float4 data = input4[idx];            // 一条指令，16 字节
val += data.x + data.y + data.z + data.w;
```

对比：V6 一条 `LDG.32` 指令搬 4 字节，V7 一条 `LDG.128` 搬 16 字节——**同样的指令开销，4 倍的数据吞吐**。一个 Warp 的 32 个线程各取一个 float4 共 512 字节，恰好由 4 个 128 字节内存事务完成，带宽利用率最优。

前提是地址按 16 字节对齐（`cudaMalloc` 与 PyTorch 张量默认满足）。

### 11.3 优化 2：Grid Stride Loop

不再按数据量启动 Block，而是**固定启动适量的 Block**（例如恰好填满所有 SM），每个线程以"全 GPU 线程总数"为步长循环，遍历完整个数组：

```cuda
for (int idx = blockIdx.x * blockDim.x + tid;
     idx < n4;
     idx += gridDim.x * blockDim.x)   // 步长 = 全 GPU 线程总数
{
    float4 data = input4[idx];
    val += data.x + data.y + data.z + data.w;
}
```

示意（4 个线程处理 12 个 float4）：

```
数据:   0   1   2   3   4   5   6   7   8   9  10  11
第1轮:  t0  t1  t2  t3
第2轮:                  t0  t1  t2  t3
第3轮:                                  t0  t1  t2  t3
```

步长设计保证覆盖完整且不重复；每一轮内相邻线程访问相邻地址，合并访存依然成立。

Grid 大小的经验取法是 `SM 数量 × 每 SM 可驻留 Block 数`，让 GPU"刚好填满"。例如 blockDim=256、gridDim=128 处理 16M float 时：总线程 32768 个，每线程循环约 122 次、累加约 488 个 float——大量工作在寄存器内完成，只做一次 Block 级归约。

### 11.4 完整实现

```cuda
__global__ void reduce_v7(float* input, float* output, int n) {
    int tid  = threadIdx.x;
    int lane = tid % 32;
    int wid  = tid / 32;

    float4* input4 = reinterpret_cast<float4*>(input);
    int n4 = n / 4;

    float val = 0.0f;

    // 主体：float4 + Grid Stride Loop
    for (int idx = blockIdx.x * blockDim.x + tid;
         idx < n4;
         idx += gridDim.x * blockDim.x)
    {
        float4 data = input4[idx];
        val += data.x + data.y + data.z + data.w;
    }

    // 尾部：n 不是 4 的倍数时，剩余 1~3 个元素按标量处理
    int tail_start = n4 * 4;
    for (int idx = tail_start + blockIdx.x * blockDim.x + tid;
         idx < n;
         idx += gridDim.x * blockDim.x)
    {
        val += input[idx];
    }

    // Block 内两级归约（同 V6）
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    __shared__ float warp_results[32];
    if (lane == 0) warp_results[wid] = val;
    __syncthreads();

    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }

    if (tid == 0) output[blockIdx.x] = val;
}
```

### 11.5 完整执行流程（16M float，blockDim=256，gridDim=128）

```
阶段 1  加载 + 局部累加（float4 + Grid Stride Loop）
        32768 个线程，每线程循环 ~122 次 → 32768 个寄存器局部和

阶段 2  Warp 内 Shuffle 归约
        1024 个 Warp，各自的 lane 0 得到部分和

阶段 3  Warp 间归约（Block 内两级归约的第二级）
        128 个 Block → 128 个部分和

阶段 4  二次归约
        仅 128 个值，一个 Block 一次 kernel 即可收尾
```

对比 V6（需要对 32768 个部分和做二次归约），V7 的收尾工作量可以忽略不计。

### 11.6 与 V6 的对比

| 维度 | V6 | V7 |
|------|----|----|
| 每线程处理量 | 2 个 float | **数百个 float**（循环累加） |
| 单次加载宽度 | 4 字节 | **16 字节** |
| Block 数量 | 随数据量增长（可达数万） | **固定数百个** |
| 二次归约规模 | 数万个值 | **数百个值** |
| 带宽利用率 | 中高 | **接近理论峰值** |

至此，Reduce kernel 的主要瓶颈全部消除：归约在寄存器完成，加载以最大宽度进行，带宽利用率接近硬件上限——这就是 memory-bound 算子优化的终点。

---

## 第 12 章 工程化：多级归约与 PyTorch 扩展

### 12.1 Block 部分和的收尾：多级（递归）归约

单次 kernel 启动只能得到每个 Block 的部分和。要得到最终标量，通用做法是**将输出作为下一轮输入，反复调用 kernel 直到只剩 1 个值**。以 blockDim=256、N=33554432 为例：

```
第 1 次调用: 33,554,432 元素 → 131,072 个部分和
第 2 次调用:    131,072 元素 →     512 个部分和
第 3 次调用:        512 元素 →       2 个部分和
第 4 次调用:          2 元素 →       1 个值 ✓
```

每轮规模缩小 blockDim 倍。若使用 V7（固定 Grid），第一轮后就只剩几百个值，第二轮即可收尾。

### 12.2 完整的 PyTorch CUDA 扩展

下面以 V0 kernel 为例（换成任意版本只需替换 kernel 函数），给出可编译运行的最小扩展。

**reduce_kernel.cu**

```cuda
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void reduce_v0(float *g_idata, float *g_odata, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// 多级归约：反复调用 kernel 直到只剩 1 个值
torch::Tensor reduce_sum(torch::Tensor input) {
    auto current = input.contiguous().to(torch::kFloat32);
    int n = current.numel();

    while (n > 1) {
        int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        auto output = torch::zeros(grid_size, current.options());

        reduce_v0<<<grid_size, BLOCK_SIZE>>>(
            current.data_ptr<float>(),
            output.data_ptr<float>(),
            n
        );

        current = output;   // 本轮输出作为下一轮输入
        n = grid_size;
    }

    return current;   // shape [1]
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_sum", &reduce_sum, "Reduce sum with V0 kernel");
}
```

**setup.py**

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='reduce_kernel',
    ext_modules=[
        CUDAExtension('reduce_kernel', ['reduce_kernel.cu'])
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

**test.py**

```python
import torch
import reduce_kernel

N = 32 * 1024 * 1024
x = torch.full((N,), 2.0, device='cuda')

result = reduce_kernel.reduce_sum(x)
print(f"自定义 kernel 结果: {result.item()}")   # 期望 67108864.0

expected = x.sum()
print(f"PyTorch sum 结果:   {expected.item()}")
```

**编译与运行**

```bash
pip install -e .
python test.py
```

---

## 第 13 章 PyTorch 算子注册机制

把 C++/CUDA 函数暴露给 PyTorch 有多种方式，按工程化程度从低到高依次介绍。

### 13.1 方式 1：PYBIND11_MODULE（最简单）

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_sum", &reduce_sum, "Reduce sum");
}
```

- pybind11 原生接口，直接把 C++ 函数导出为 Python 函数；
- **不经过 PyTorch dispatcher**，不支持按 device 自动分发，也不兼容 `torch.compile`；
- 调用方式：`import my_ext; my_ext.reduce_sum(x)`；
- 适合学习与简单场景。

### 13.2 方式 2：TORCH_LIBRARY + TORCH_LIBRARY_IMPL（生产级）

接口声明与各后端实现分离，接入 PyTorch dispatcher：

```cpp
// 声明算子 schema（每个 library 名只能声明一次）
TORCH_LIBRARY(my_ops, m) {
    m.def("reduce_sum(Tensor input) -> Tensor");
}

// 为不同 dispatch key 提供实现（可分散在不同文件）
TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
    m.impl("reduce_sum", &reduce_sum_cuda);
}

TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
    m.impl("reduce_sum", &reduce_sum_cpu);
}

// 可选：autograd 实现
TORCH_LIBRARY_IMPL(my_ops, AutogradCUDA, m) {
    m.impl("reduce_sum", &reduce_sum_autograd);
}
```

- dispatcher 根据输入 tensor 的 device 自动路由到对应实现；
- 兼容 `torch.compile`、FakeTensor 等基础设施；
- 调用方式：`torch.ops.my_ops.reduce_sum(x)`。

### 13.3 方式 3：def 与 impl 写在一起

`TORCH_LIBRARY` 块内的 `m`（`torch::Library` 类型）同时具有 `def` 与 `impl` 方法，可以就地绑定（sgl-kernel 等项目的实际用法）：

```cpp
TORCH_LIBRARY(sgl_kernel, m) {
    m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
    m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

    // 无需 dispatch 的工具函数可直接绑定（CatchAll）
    m.def("dispose", &dispose);
    m.def("meta_size", &meta_size);
}
```

> 注意区分：此处的 `m` 是 `torch::Library`，与 `PYBIND11_MODULE` 中的 `pybind11::module` 是完全不同的对象。

### 13.4 方式 4：TORCH_LIBRARY_FRAGMENT（跨文件追加）

```cpp
// 在另一个编译单元向已存在的 library 追加算子
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
    m.def("silu_and_mul_cpu(Tensor input) -> Tensor");
    m.impl("silu_and_mul_cpu", torch::kCPU, &silu_and_mul_cpu);
}
```

三个宏的出现次数规则：

| 宏 | 作用 | 允许出现次数 |
|----|------|------------|
| `TORCH_LIBRARY(name, m)` | 创建算子库 | 每个 name **一次** |
| `TORCH_LIBRARY_FRAGMENT(name, m)` | 向已有库追加算子 | 多次 |
| `TORCH_LIBRARY_IMPL(name, key, m)` | 为指定 key 补充实现 | 多次 |

### 13.5 Dispatch Key 如何决定路由

dispatcher 根据输入 tensor 的属性计算 dispatch key，选择实现：

```
tensor.device=cuda, requires_grad=True  → AutogradCUDA
tensor.device=cuda, requires_grad=False → CUDA
tensor.device=cpu,  requires_grad=True  → AutogradCPU
tensor.device=cpu,  requires_grad=False → CPU
FakeTensor（torch.compile 追踪用）       → Meta
```

### 13.6 选型建议

| 机制 | 适用场景 |
|----|---------|
| `PYBIND11_MODULE` | 学习、原型、单设备简单扩展 |
| `TORCH_LIBRARY` 系列 | 生产级算子：多后端、autograd、torch.compile |

---

## 第 14 章 PyTorch 的 dtype 分发机制

### 14.1 核心结论

> PyTorch dispatcher **只按 device（dispatch key）分发，不按 dtype 分发**。
> dtype 的选择在实现函数**内部**完成，主要工具是 `AT_DISPATCH_*` 宏（本质是 switch + 模板实例化）。

完整的两层分发流程：

```
Python: torch.ops.my_ops.reduce_sum(x)
            │
            ▼
  第一层：PyTorch Dispatcher（框架负责）
     依据 x 的 dispatch key（device、autograd 状态）
     x.device == cuda → 路由到 CUDA 实现函数
            │
            ▼
  第二层：实现函数内部（开发者负责）
     依据 x.scalar_type() 选择模板实例
     kFloat32  → kernel<float>
     kFloat16  → kernel<half>
     kBFloat16 → kernel<nv_bfloat16>
```

### 14.2 主流方式：AT_DISPATCH_* 宏（约 90% 场景）

```cpp
AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(), "reduce", [&] {
    // lambda 内 scalar_t 依次被实例化为 float / double / half / bf16
    reduce_kernel<scalar_t><<<grid, block>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        n
    );
});
```

宏展开后本质是一个 switch，每个 case 里用 `using scalar_t = xxx` 定义类型别名，lambda 体针对每种类型各实例化一份：

```cpp
switch (input.scalar_type()) {
    case at::kFloat: {
        using scalar_t = float;
        reduce_kernel<float><<<grid, block>>>(input.data_ptr<float>(), ...);
        break;
    }
    case at::kDouble: {
        using scalar_t = double;
        reduce_kernel<double><<<grid, block>>>(input.data_ptr<double>(), ...);
        break;
    }
    default:
        AT_ERROR("reduce not implemented for '", toString(input.scalar_type()), "'");
}
```

常用宏变体：

| 宏 | 覆盖类型 |
|----|--------|
| `AT_DISPATCH_FLOATING_TYPES` | float, double |
| `AT_DISPATCH_FLOATING_TYPES_AND_HALF` | float, double, half |
| `AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, ...)` | float, double, half, bf16 |
| `AT_DISPATCH_ALL_TYPES` | 整型 + 浮点 |
| `AT_DISPATCH_ALL_TYPES_AND(kHalf, ...)` | 所有数值类型 + half |
| `AT_DISPATCH_SWITCH` | 完全自定义 |

### 14.3 其他方式

**直接 if/switch（约 10% 场景）**：当不同 dtype 的**算法本身不同**（而非仅类型参数不同）时，直接写分支：

```cpp
if (self.scalar_type() == ScalarType::Float) {
    return my_op_float_impl(self, other);    // float 专用路径
} else if (self.scalar_type() == ScalarType::Half) {
    return my_op_half_impl(self, other);     // half 专用路径（如精度补偿）
}
```

**TensorIterator 自动类型提升**：逐元素类算子借助 `TensorIterator` 自动完成 dtype promotion（如 float32 + float64 → float64），开发者只需对 `iter.common_dtype()` 做分发，无需处理混合输入。

### 14.4 为什么 dtype 不设计成 dispatch key

1. **组合爆炸**：device(≈5) × dtype(≈12) × autograd(2) 会产生 120+ 个 key，dispatch table 不可维护；
2. **实现高度同构**：多数算子在不同 dtype 下只是类型参数不同，一份模板即可覆盖；
3. **不影响调度语义**：float32 与 float64 的 autograd、编译策略完全一致，没有分流的必要。

---

## 第 15 章 总结与实践建议

### 15.1 八个版本回顾

| 版本 | 核心手段 | 解决的瓶颈 |
|------|---------|-----------|
| V0 | 朴素树形归约 | —（基准） |
| V1 | Strided Index 连续线程映射 | Warp Divergence |
| V2 | 步长从大到小 + `tid < s` | Warp Divergence + Bank Conflict |
| V3 | 每线程加载 2 元素预相加 | 线程闲置 |
| V4 | 手动展开最后一个 Warp | 冗余的 `__syncthreads()` |
| V5 | 模板参数编译期展开 | 循环与运行时分支开销 |
| V6 | Warp Shuffle 两级归约 | 共享内存往返延迟 |
| V7 | float4 向量化 + Grid Stride Loop | 全局内存带宽利用率、Block 过多 |

### 15.2 关键指标对比（blockDim=256）

| 指标 | V0 | V1 | V2 | V3 | V4 | V5 | V6 | V7 |
|------|----|----|----|----|----|----|----|----|
| `__syncthreads()` | 8 | 8 | 8 | 7 | 2 | 2 | 1 | 1 |
| Warp Divergence | 严重 | 无 | 无 | 无 | 无 | 无 | 无 | 无 |
| Bank Conflict | 有 | 有 | 无 | 无 | 无 | 无 | 无 | 无 |
| 每线程加载量 | 1 | 1 | 1 | 2 | 2 | 2 | 2 | N×4 |
| 数据交换方式 | smem | smem | smem | smem | smem | smem | shuffle | shuffle |
| 带宽利用率 | 低 | 低 | 低 | 中 | 中 | 中 | 中高 | 最高 |

### 15.3 通用优化方法论

Reduce 的优化过程体现了 CUDA 性能调优的一般规律，可归纳为三条主线：

1. **顺应硬件执行模型**：让同一 Warp 的线程走同一分支（V1/V2）、访问相邻地址（V2）、利用 Warp 内天然同步（V4/V6）；
2. **把数据留在更快的存储层次**：全局内存 → 共享内存 → 寄存器（V6），并用最大粒度访问最慢的层次（V7 的 float4）；
3. **让每个线程做足够多的工作**：减少纯搬运线程（V3）、用编译期信息替代运行时开销（V5）、以固定 Grid 循环覆盖数据（V7）。

### 15.4 版本选择建议

| 场景 | 推荐版本 |
|------|---------|
| 学习并行归约原理 | V0 → V2 |
| 快速实现、性能够用 | V2 / V3 |
| 生产环境、追求性能 | V6 / V7 |
| 兼容旧 GPU（SM < 3.0） | V5 |
| 超大数据量（> 100M 元素） | V7 |

实际工程中，也可以直接使用 CUB（`cub::DeviceReduce` / `cub::BlockReduce`）或 Thrust 库——它们内部实现与 V7 思路一致且经过充分调优。手写 kernel 的价值在于理解原理，以及在融合算子（fused kernel）中嵌入归约逻辑时无法直接使用库的场景。

---

## 附录：关键概念速查

| 概念 | 含义 | 相关章节 |
|------|------|---------|
| Warp | 32 线程的硬件调度单元，锁步执行、天然同步 | 第 2 章 |
| Warp Divergence | 同一 Warp 内线程走不同分支导致串行化 | 第 4~5 章 |
| Bank | 共享内存的 32 个存储体，`bank = 地址 % 32` | 第 2、6 章 |
| Bank Conflict | 多线程同时访问同一 Bank 的不同地址，被串行化 | 第 5~6 章 |
| 合并访存（Coalescing） | Warp 内线程访问连续地址，合并为最少内存事务 | 第 7、11 章 |
| `__syncthreads()` | Block 级栅栏同步，有等待开销 | 第 4、8 章 |
| `volatile` | 禁止编译器把变量缓存进寄存器，强制读写内存 | 第 8 章 |
| `__shfl_down_sync` | Warp 内寄存器直传，延迟 1~2 周期 | 第 10 章 |
| lane | 线程在 Warp 内的编号，`lane = tid % 32` | 第 10 章 |
| float4 | 16 字节向量类型，一条指令加载 4 个 float | 第 11 章 |
| Grid Stride Loop | 固定 Grid 规模，线程以全局线程总数为步长循环处理数据 | 第 11 章 |
| `template <int N>` | 将参数提升为编译期常量，触发死代码消除与完全展开 | 第 9 章 |
| memory-bound | 性能受限于访存带宽而非算力的算子类型 | 第 3、11 章 |
| Dispatch Key | PyTorch dispatcher 的路由键（CPU/CUDA/Autograd 等） | 第 13 章 |
| `TORCH_LIBRARY` | 创建算子库并声明 schema（每库一次） | 第 13 章 |
| `TORCH_LIBRARY_IMPL` | 为指定 dispatch key 提供实现（可多次） | 第 13 章 |
| `TORCH_LIBRARY_FRAGMENT` | 跨编译单元向已有库追加算子（可多次） | 第 13 章 |
| `AT_DISPATCH_*` | 实现内部按 dtype 做模板分发的宏 | 第 14 章 |
