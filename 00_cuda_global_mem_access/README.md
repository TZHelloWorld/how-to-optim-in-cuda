# CUDA 全局内存访问机制：内存段、事务粒度与缓存行

> 本文系统解析 CUDA 全局内存访问的硬件机制：从 DRAM 的内存段（Segment）结构出发，依次讲清事务粒度（Transaction Granularity）、缓存行（Cache Line）三个核心概念及其关系，深入 L1/L2 两级缓存行大小不同的设计原因与协同命中流程，最后总结访存优化的最佳实践。全文论断均标注官方文档出处，并配有可运行的带宽基准测试程序验证结论。
>
> 参考文档：
> - CUDA C++ Programming Guide（§8.3.2 Device Memory Accesses，§20.4~20.8 Compute Capability 各章节）
> - CUDA C++ Best Practices Guide（§10.2.1 Coalesced Access to Global Memory，§10.2.2 L2 Cache）
> - PTX ISA Reference（§9.7.9.1 Cache Operators）

---

## 目录

- [第 1 章 概述：三个概念与一条主线](#第-1-章-概述三个概念与一条主线)
- [第 2 章 内存段（Memory Segment）](#第-2-章-内存段memory-segment)
- [第 3 章 事务粒度与数据访问单元](#第-3-章-事务粒度与数据访问单元)
- [第 4 章 缓存行与 L1/L2 设计差异](#第-4-章-缓存行与-l1l2-设计差异)
- [第 5 章 完整访问流程：L1/L2 协同命中过程](#第-5-章-完整访问流程l1l2-协同命中过程)
- [第 6 章 概念体系总览与架构演进](#第-6-章-概念体系总览与架构演进)
- [第 7 章 最佳实践](#第-7-章-最佳实践)
- [第 8 章 配套基准测试](#第-8-章-配套基准测试)

---

## 第 1 章 概述：三个概念与一条主线

全局内存是 CUDA kernel 最主要的性能瓶颈来源。理解"一次访存到底发生了什么"，需要区分三个经常被混用的概念：

| 概念 | 一句话定义 | 描述视角 |
|------|-----------|---------|
| **内存段（Segment）** | DRAM 地址空间按固定大小切分的对齐地址块 | DRAM 物理地址结构 |
| **事务粒度（Transaction Granularity）** | 内存总线一次物理传输的最小数据量 | 硬件总线视角 |
| **缓存行（Cache Line）** | 缓存内部一个存储槽位的大小 | 缓存存储结构视角 |

三者围绕同一条主线——**一个 warp 的访存请求如何被硬件拆解、传输与缓存**：

```
warp 发出访存指令
    └─> 合并（Coalesce）为若干个对齐的段请求      ← 第 2 章：段与对齐
            └─> 以固定事务粒度在总线上传输          ← 第 3 章：事务粒度
                    └─> 数据以缓存行为单位驻留 L1/L2 ← 第 4~5 章：缓存行与协同
```

程序员无法改变段大小、事务粒度、缓存行大小（均为硬件固化），**唯一能优化的是访问模式**——让 warp 的请求覆盖尽量少的段。这就是"合并访存（Coalesced Access）"优化的全部本质，第 7 章的所有实践原则都由此推出。

---

## 第 2 章 内存段（Memory Segment）

### 2.1 官方定义

CUDA C++ Programming Guide（§8.3.2）原文：

> *"Global memory resides in device memory and device memory is accessed via **32-, 64-, or 128-byte memory transactions**. These memory transactions must be **naturally aligned**: Only the 32-, 64-, or 128-byte **segments** of device memory that are aligned to their size (i.e., whose first address is a multiple of their size) can be read or written by memory transactions."*

**内存段（segment）** 是硬件将 DRAM 地址空间按固定大小切分后的每一个地址区块。全局内存的段大小有三种：**32 字节、64 字节、128 字节**。

### 2.2 三种段大小同时存在

这三种大小并非互斥，而是同一块 DRAM 的不同"视角"，硬件根据访问情况选择使用哪种粒度：

```
同一段 DRAM 地址空间（0~256 字节）的三种划分视角：

32 字节段：
  [0,32) [32,64) [64,96) [96,128) [128,160) [160,192) [192,224) [224,256)
   段0    段1     段2     段3       段4        段5        段6        段7

64 字节段：
  [0,64)          [64,128)          [128,192)          [192,256)
   段0              段1               段2                段3

128 字节段：
  [0,128)                            [128,256)
   段0                                段1
```

### 2.3 为什么段的起始地址必须是段大小的整数倍

这不是人为规定，而是**硬件寻址位运算的数学必然结果**。

以 32 字节段（32 = 2⁵）为例，硬件将地址的低 5 位作为"段内偏移"，高位作为"段号"：

```
地址位结构（32 字节段，低5位为段内偏移）：

 地址(十进制) | 段号(高位) | 段内偏移(低5位) | 说明
      0       |     0      |     00000       | 段0 起点
      4       |     0      |     00100       | 段0 内部
     31       |     0      |     11111       | 段0 末尾
     32       |     1      |     00000       | 段1 起点 ← 32 是 32 的整数倍
     64       |     2      |     00000       | 段2 起点 ← 64 是 32 的整数倍
    128       |     4      |     00000       | 段4 起点 ← 128 是 32 的整数倍
    100       |     3      |     00100       | 段3 内部 ← 100 不是段的起点
```

**段起始地址 = 段号 × 段大小**，因此段的起始地址必然是段大小的整数倍。这是位运算寻址结构的天然结果，不可更改。

### 2.4 非对齐访问触发额外事务

官方 Best Practices Guide（§10.2.1.2）说明：

> *"If sequential threads in a warp access memory that is sequential but not aligned with a 32-byte segment, **five 32-byte segments** will be requested."*

示例对比（warp 访问 128 字节数据）：

```
对齐访问（地址 [0, 128)，从 0 开始）：
  触发段：[0,32) [32,64) [64,96) [96,128) → 共 4 个段，100% 利用率

非对齐访问（地址 [4, 132)，从 4 开始）：
  触发段：[0,32)★ [32,64) [64,96) [96,128) [128,160)★ → 共 5 个段
  ★ 表示该段中有部分字节不是所需数据，但仍须加载整个段 → 带宽浪费
```

Programming Guide（§8.3.2）进一步量化了影响：

> *"For example, if a **32-byte memory transaction** is generated for each thread's **4-byte access**, throughput is **divided by 8**."*

即：段是"全取或不取"的最小单位，无法只取其中一部分。

### 2.5 数据地址可以不对齐内存段吗

**答案：可以不对齐，硬件会自动处理，代价是触发更多事务、浪费带宽；但 8/16 字节宽类型非对齐会产生错误结果。**

#### 2.5.1 "必须自然对齐"说的是事务，不是数据指针

Programming Guide（§8.3.2）原文中 "must be naturally aligned" 描述的主语是**内存事务本身**，而非用户数据的起始地址：

> *"These memory transactions must be naturally aligned: Only the 32-, 64-, or 128-byte segments of device memory that are **aligned to their size** can be read or written by memory transactions."*

含义是：

- **你的数据指针**：可以是任意地址，不强制对齐；
- **硬件执行内存事务时**：只能以对齐的段为单位读写 DRAM；
- 如果数据地址跨越了段边界，硬件会**自动拆成多个对齐事务**来覆盖整个数据范围。

#### 2.5.2 非对齐时硬件的实际行为

Best Practices Guide（§10.2.1.2）：

> *"If sequential threads in a warp access memory that is sequential but **not aligned** with a 32-byte segment, **five 32-byte segments** will be requested."*

硬件处理过程如下：

```
数据地址 [4, 132)（不对齐，起始地址 4 不是 32 的整数倍）：

物理内存段边界：  0     32    64    96    128   160
                  |      |     |     |      |     |
你的数据范围：         [4 ........................ 132)

硬件必须覆盖的对齐段：
  [0,  32) ← 段0：其中 [0,4)  共 4 字节不属于你的数据，但必须整段加载
  [32, 64) ← 段1：全部有效
  [64, 96) ← 段2：全部有效
  [96,128) ← 段3：全部有效
  [128,160)← 段4：其中 [132,160) 共 28 字节不属于你的数据，但必须整段加载

结果：5 个事务（对齐时只需 4 个），实际有效数据 128 字节，加载 160 字节，浪费 32 字节
```

#### 2.5.3 非对齐访问的性能影响

Best Practices Guide（§10.2.1.3）给出了 Tesla V100 上的实测数据：

> *"Otherwise, five 32-byte segments are loaded per warp, and we would expect approximately **4/5th** of the memory throughput achieved with no offsets."*

> *"...the offset memory throughput achieved is, however, approximately **9/10th**, because **adjacent warps reuse the cache lines their neighbors fetched**."*

理论损失为 4/5 = 80%，实测约为 9/10 = 90%，因为 L1 的 128 字节大缓存行使得相邻 warp 可以复用彼此"多取"的数据，从而弥补了部分性能损失（机制详见第 5.3 节）。

| 情况 | 事务数 | 加载字节 | 有效字节 | 带宽利用率 | 实测吞吐（V100）|
|------|--------|---------|---------|-----------|----------------|
| 对齐访问（地址 % 32 == 0） | 4 | 128 B | 128 B | 100% | 基准 |
| 非对齐访问（地址 % 32 != 0）| 5 | 160 B | 128 B | 80% | ~90%（L1 缓存行复用补偿）|

> 该结论可用本目录的基准测试程序在你自己的 GPU 上复现，见第 8 章。

#### 2.5.4 特殊情况：8/16 字节宽类型非对齐会产生错误结果

Programming Guide（§8.3.2）有明确警告：

> *"Reading **non-naturally aligned 8-byte or 16-byte words** produces **incorrect results** (off by a few words), so special care must be taken to maintain alignment of the starting address of any value or array of values of these types."*

对于 `double`（8 字节）、`float4`/`double2`（16 字节）等宽类型，地址非对齐**不只是性能问题，而是会读出错误数据**，必须严格保证对齐：

```c
// 危险：double 要求 8 字节对齐
double *ptr = (double *)((char *)base + 4);  // 地址偏移 4，不是 8 的倍数
double val = *ptr;  // 产生错误结果！

// 安全：cudaMalloc 保证 ≥256 字节对齐，天然满足所有类型
double *safe_ptr;
cudaMalloc(&safe_ptr, N * sizeof(double));  // 地址保证 8 字节对齐
```

#### 2.5.5 小结

| 问题 | 答案 |
|------|------|
| 数据地址能不对齐吗？ | **能**，程序不会崩溃（8/16 字节宽类型除外） |
| 硬件如何处理非对齐？ | 自动拆成多个对齐的 32 字节事务覆盖目标地址范围 |
| 代价是什么？ | 事务数增加，有效带宽利用率下降（理论 80%，实测约 90%）|
| 8/16 字节类型非对齐？ | **产生错误结果**，必须严格对齐 |
| `cudaMalloc` 分配的内存？ | 保证 ≥256 字节对齐，天然满足所有基本类型对齐要求 |

---

## 第 3 章 事务粒度与数据访问单元

### 3.1 定义

**事务粒度（Transaction Granularity）** 是内存总线一次物理传输的最小数据量，由底层内存控制器的硬件总线宽度决定，与上层软件无关。CUDA 全局内存支持三种事务粒度：32 字节、64 字节、128 字节（与内存段大小对应）。

**数据访问单元（Data Access Unit）** 是从 warp 视角描述的最小请求单位，与事务粒度在 CC 6.0+ 中数值相同（均为 32 字节），但描述侧重不同：

- **事务粒度** → 硬件总线视角，强调"物理传输"；
- **数据访问单元** → warp/软件视角，强调"一次请求触发的最小数据量"。

### 3.2 三种段大小选项本身是固定的

这三个数值（32/64/128 字节）**由 GPU 芯片的内存控制器物理总线宽度决定，是硬件固化的，程序员无法修改，也无法自定义其他值**（比如 48 字节或 96 字节的段）。

它们的来源是内存系统分级总线的物理设计：

```
硬件层级          传输粒度      对应段大小
─────────────────────────────────────────
内存控制器 ↔ L2    32 字节       32 字节段（现代 GPU 的基础事务单元）
L2         ↔ L1   32 字节       32 字节段
L1 缓存行填充      128 字节      128 字节段（L1 一次预取的完整块）
```

Programming Guide（§8.3.2）明确列出了这三种值：

> *"Global memory resides in device memory and device memory is accessed via **32-, 64-, or 128-byte memory transactions**."*

**程序员能做的不是改变段大小，而是通过优化访问模式来减少触发的段数量。**

### 3.3 实际生效的粒度：由 Compute Capability 决定

不同架构代际使用的有效事务粒度不同（Best Practices Guide §10.2.1）：

> *"For certain devices of compute capability 5.2, L1-caching of accesses to global memory can be optionally enabled. If L1-caching is enabled on these devices, the number of required transactions is equal to the number of required **128-byte aligned segments**."*

> *"On devices of compute capability 6.0 or higher, L1-caching is the default, however the **data access unit is 32-byte** regardless of whether global loads are cached in L1 or not."*

| Compute Capability | 有效事务粒度 / 数据访问单元 | L1 缓存默认状态 |
|--------------------|------------|------|
| CC 5.0 | 128 字节（L1 未开启时按 L2 段计）| 默认关闭，全走 L2 |
| CC 5.2 | 128 字节（开启 L1 时）| 默认关闭，`-dlcm=ca` 可开启 |
| **CC 6.0+（Pascal 及以后）** | **32 字节** | **默认开启**；与是否经过 L1 无关 |

**结论**：CC 6.0+ 的现代 GPU 统一以 **32 字节**作为全局内存的基本事务粒度。

### 3.4 一次访问产生多少个事务：由两个因素共同决定

这是最核心的问题。Programming Guide（§8.3.2）给出了决定事务数的两个因素：

> *"When a warp executes an instruction that accesses global memory, it coalesces the memory accesses of the threads within the warp into one or more of these memory transactions **depending on:***
> 1. ***the size of the word accessed by each thread***（每个线程访问的数据类型大小）
> 2. ***the distribution of the memory addresses across the threads***"*（线程之间的地址分布）

**因素一：数据类型大小（word size）**

Programming Guide（§8.3.2 "Size and Alignment Requirement"）：

> *"Global memory instructions support reading or writing words of size equal to **1, 2, 4, 8, or 16 bytes**. Any access to data residing in global memory compiles to a **single** global memory instruction if and only if the size of the data type is 1, 2, 4, 8, or 16 bytes and the data is naturally aligned."*

> *"If this size and alignment requirement is not fulfilled, the access **compiles to multiple instructions** with interleaved access patterns that **prevent these instructions from fully coalescing**."*

数据类型大小决定了整个 warp 的总请求量：

```
数据类型   单线程字节   warp总请求（32线程）   最少触发段数（32字节段）
char       1 字节       32 字节               1 个
short      2 字节       64 字节               2 个
float      4 字节       128 字节              4 个（最常见情况）
double     8 字节       256 字节              8 个
float4    16 字节       512 字节             16 个
```

不满足 1/2/4/8/16 字节约束时，一条访问语句会被编译成多条指令，进一步阻碍合并。

**因素二：地址分布（address distribution）**

地址分布决定 warp 的总请求究竟跨越了几个对齐的 32 字节段：

```
情况1：连续对齐（stride=1，起始地址对齐32字节）
  32线程访问 float：地址 [0, 4, 8, ..., 124]
  覆盖段：[0,32) [32,64) [64,96) [96,128) → 4 个事务（最优，100% 利用率）

情况2：连续但非对齐（起始地址偏移4字节）
  32线程访问 float：地址 [4, 8, 12, ..., 128]
  覆盖段：[0,32) [32,64) [64,96) [96,128) [128,160) → 5 个事务（80% 利用率）

情况3：跨步访问（stride=2，间隔访问）
  32线程访问 float：地址 [0, 8, 16, ..., 248]
  覆盖段：[0,32) [32,64) ... [224,256) → 8 个事务（50% 利用率）

情况4：极端跨步（stride=32）
  32线程访问 float：地址 [0, 128, 256, ..., 3968]
  覆盖段：32 个不相邻的 32 字节段 → 32 个事务（~3% 利用率）
```

Best Practices Guide（§10.2.1.1）：

> *"For example, if the threads of a warp access adjacent 4-byte words (e.g., adjacent `float` values), **four coalesced 32-byte transactions** will service that memory access."*

> *"If from any of the four 32-byte segments only a subset of the words are requested... the **full segment is fetched anyway**."*

即段是"全取或不取"的，只要触碰到该段的任意地址，整个段都必须加载。

### 3.5 小结：固定与可变的边界

| 维度 | 是否固定 | 由什么决定 |
|------|---------|-----------|
| 段大小的三个选项（32/64/128B） | **固定** | GPU 芯片内存控制器总线宽度，硬件决定 |
| 实际生效的事务粒度 | **随架构变化** | Compute Capability：CC 6.0+ 统一为 32 字节 |
| 一次访问产生的事务数 | **动态，运行时决定** | ① 数据类型大小 × 32 线程 = warp 总请求量；② 地址分布决定跨越几个对齐段 |
| 程序员能控制的 | **事务数（间接）** | 通过对齐访问、连续访问、合理选择数据类型来最小化事务数 |

---

## 第 4 章 缓存行与 L1/L2 设计差异

### 4.1 缓存行的定义

缓存行是**缓存内部存储数据的最小单位**，即一个缓存槽位能存放的数据大小。缓存以缓存行为单位进行分配和替换，不能存半行。

PTX ISA 文档对缓存行的重要性有直接描述：

> *"Cache line size can be safely ignored when designing for correctness but must be considered in the code structure when designing for peak performance."*

### 4.2 L1 与 L2 的缓存行大小

| 缓存层级 | 缓存行大小 | 文档依据 |
|---------|---------|----------|
| **L1 Cache** | **128 字节** | CC 5.2 开启 L1 时，事务数等于所需 128-byte aligned segments 的数量 |
| **L2 Cache** | **32 字节**  | CC 6.0+ 统一 data access unit 为 32 字节；L2 以 32 字节段为原子单位操作 |

### 4.3 缓存行与事务粒度的关系

两者描述同一物理行为的不同视角：

```
事务粒度（Transaction Granularity）
  → 内存总线视角：一次物理传输搬运多少字节
  → 强调"传输行为"

缓存行（Cache Line）
  → 缓存存储视角：一个缓存槽位存放多少字节
  → 强调"存储结构"

对应关系（CC 6.0+）：
  L2 缓存行（32 字节） ≡ 事务粒度（32 字节）
  L1 缓存行（128 字节）= 4 × L2 缓存行 = 4 × 32 字节事务的聚合
```

### 4.4 架构演进背景（官方文档各代架构描述）

**CC 5.x（Maxwell）**（Programming Guide §20.4）：

> *"a unified L1/texture cache of **24 KB** used to cache reads from global memory"*
> *"There is also an **L2 cache shared by all SMs**..."*
> *"**Global memory accesses are always cached in L2.** Data that is not read-only... **cannot be cached in the unified L1/texture cache** for devices of compute capability 5.0."*

Maxwell 架构默认全局内存只走 L2，L1 仅对只读数据（`__ldg()`）开放。

**CC 6.x（Pascal）**（Programming Guide §20.5）：

> *"a unified L1/texture cache for reads from global memory of size **24 KB (6.0/6.2) or 48 KB (6.1)**"*
> *"Global memory behaves the same way as in devices of compute capability 5.x"*

**CC 7.x（Volta/Turing）**（Programming Guide §20.6）：

> *"a **unified data cache and shared memory** with a total size of **128 KB (Volta) or 96 KB (Turing)**"*
> *"Shared memory is partitioned out of unified data cache... The remaining data cache serves as an **L1 cache**"*

Volta 的重大变化：**L1 Cache 和 Shared Memory 共享同一片片上 SRAM**，两者的容量分配可按 kernel 需求动态配置：

> *"For the Volta architecture (compute capability 7.0), the unified data cache has a size of **128 KB**, and the shared memory capacity can be set to **0, 8, 16, 32, 64 or 96 KB**. The remaining data cache serves as L1."*

**CC 8.x（Ampere）**（Programming Guide §20.7）：

> *"a unified data cache and shared memory with a total size of **192 KB** for devices of compute capability 8.0 and 8.7"*

### 4.5 两级缓存行大小不同的设计原因

| 设计维度 | L1 Cache（128 字节缓存行）| L2 Cache（32 字节粒度）|
|---------|--------------------------|------------------------|
| **物理位置** | 每个 SM 私有，片上（极低延迟，~20-30 周期） | 全芯片共享，片上（低延迟，~200 周期） |
| **容量** | 小（数十 KB，与 Shared Memory 共享） | 大（数 MB，A100 为 40 MB） |
| **服务对象** | 单个 SM 的多个 warp | 全部 SM 共享访问 |
| **设计目标** | 利用**空间局部性**进行大块预取，减少 SM 等待 DRAM 的次数 | 精确按需管理，减少带宽浪费，支持多 SM 公平竞争 |
| **缓存行设计** | 大行（128 字节）：一次填充，覆盖更多相邻数据，提高命中率 | 小行（32 字节）：精确粒度，避免无效数据占用宝贵共享容量 |

**核心设计逻辑**：

- **L1 用大缓存行（128 字节）**：L1 是 SM 私有的极小缓存，容量珍贵但延迟最低。通过大块预取（128 字节），期望同一线程块中的后续 warp 能直接命中，从而用空间换时间，最大化缓存命中率。这与 CPU 的 L1 缓存设计哲学相同。

- **L2 用小粒度（32 字节）**：L2 是全芯片所有 SM 共享的，必须精确管理。若 L2 也用 128 字节大粒度，每个 SM 的访问都会填充大量可能永远不会被用到的数据，造成缓存污染和带宽浪费。32 字节的细粒度使 L2 能精确匹配各种访问模式，减少无效传输。

### 4.6 PTX Cache Operators：对 L1/L2 的独立控制

PTX ISA（§9.7.9.1）提供了对两级缓存独立控制的机制：

```
Load 指令的 Cache Operator（控制 L1/L2 缓存行为）：

.ca  → Cache at ALL levels（L1 + L2）默认行为，CC 6.0+ 的 ld 指令默认
.cg  → Cache at Global level ONLY（仅 L2，绕过 L1）
.cs  → Cache Streaming（L1 + L2，但 evict-first 策略，防止污染）
.lu  → Last Use（类似 .cs，用于寄存器溢出恢复场景）
.cv  → Don't cache，每次重新从内存 fetch

Store 指令的 Cache Operator：

.wb  → Write-Back（默认，写回 L1+L2）
.cg  → 仅写入 L2，绕过 L1
.cs  → Cache Streaming（evict-first）
.wt  → Write-Through（直写到系统内存）
```

> *"`.ca` — The default load instruction cache operation is `ld.ca`, which **allocates cache lines in all levels (L1 and L2)** with normal eviction policy."*

> *"`.cg` — Use `ld.cg` to cache loads only globally, **bypassing the L1 cache**, and cache only in the L2 cache."*

这表明 L1 和 L2 是**独立**的两级缓存，可分别选择是否缓存，缓存行大小不同但协同工作——协同的具体过程见下一章。

---

## 第 5 章 完整访问流程：L1/L2 协同命中过程

### 5.1 对齐访问的完整流程（以 CC 7.0 Volta 为例）

场景：warp 的 32 个线程各访问一个 `float`（4 字节），地址范围 **[0, 128)**（对齐，`.ca` 默认缓存模式）。

```
第 1 步：warp 发出访问请求 & 合并（Coalesce）
─────────────────────────────────────────────────────────────────
32 个线程的地址：[0,4), [4,8), ..., [124,128)
硬件合并为 4 个 32 字节事务请求（数据访问单元 = 32 字节）：
  请求1: [0,   32)
  请求2: [32,  64)
  请求3: [64,  96)
  请求4: [96, 128)

第 2 步：查询 L1 Cache（缓存行 = 128 字节）
─────────────────────────────────────────────────────────────────
L1 以 128 字节为缓存行大小。
查询目标：地址 [0, 128) 所在的 128 字节对齐缓存行块 = [0, 128)

  ┌─ L1 命中 ──────────────────────────────────────────────────────┐
  │ [0, 128) 这整个 128 字节缓存行在 L1 中存在                     │
  │ → 直接从 L1 返回所需数据给 warp 寄存器                          │
  │ → 延迟极低（约 20~30 个时钟周期）                               │
  └────────────────────────────────────────────────────────────────┘

  ┌─ L1 未命中（进入第 3 步）──────────────────────────────────────┐
  │ [0, 128) 缓存行不在 L1 中                                       │
  │ → 向 L2 发起 4 次 32 字节查询                                   │
  └────────────────────────────────────────────────────────────────┘

第 3 步：查询 L2 Cache（缓存行 = 32 字节）
─────────────────────────────────────────────────────────────────
L2 以 32 字节为粒度，对每个 32 字节请求独立查询：

  [0,  32)  → L2 命中？ ┬─ 是 → 返回 32 字节给 L1
  [32, 64)  → L2 命中？ │
  [64, 96)  → L2 命中？ │  L2 全部命中：
  [96, 128) → L2 命中？ ┘  4 × 32 字节 = 128 字节 传到 L1
                            L1 将其填入一个 128 字节缓存行槽位
                            → 返回数据给 warp（约 200 个周期）

                         ┬─ 否 → 向 DRAM 发起 32 字节事务
                         │       DRAM 返回 32 字节数据
                         │       → 填入 L2（32 字节槽位）
                         └       → 聚合后填入 L1（128 字节缓存行）
                                  → 返回数据（约 600~800 个周期）
```

### 5.2 L1 的 128 字节缓存行如何从 L2 的 32 字节数据"拼凑"而来

```
从 L2（或 DRAM）分 4 次收到的 32 字节段：

  [0,   32) ██████████  (32 字节)
  [32,  64) ██████████  (32 字节)   →  L1 缓存行 [0, 128):
  [64,  96) ██████████  (32 字节)      ████████████████████████████████ (128 字节)
  [96, 128) ██████████  (32 字节)

L1 将 4 个连续 32 字节段合并，填入 1 个 128 字节的缓存行槽位。
这就是两级缓存行大小不同却能协同工作的核心机制。
```

### 5.3 非对齐访问的命中分析

场景：warp 访问地址 **[4, 132)**（偏移 4 字节，跨越 5 个 32 字节段）：

```
触发的 5 个 32 字节事务：
  [0,  32) [32, 64) [64, 96) [96, 128) [128, 160)
  ←──────────── 128字节块A ──────────────→ ←─ 128字节块B ─

L2 层：5 次 32 字节查询（比对齐时多 1 次 → 带宽多消耗 25%）

L1 层：涉及两个 128 字节缓存行：
  缓存行A: [0,   128) → 包含 4 个 32 字节段（[0,32) [32,64) [64,96) [96,128)）
  缓存行B: [128, 256) → 包含 1 个 32 字节段（[128,160)）

关键：相邻 warp 的缓存复用机会
  如果下一个 warp 访问 [128, 256) 范围的数据：
  → 缓存行B 已在 L1 中 → L1 命中！
  → 这正是 Best Practices Guide §10.2.1.3 所说：
    "adjacent warps reuse the cache lines their neighbors fetched"
    （相邻 warp 复用彼此预取的缓存行）
```

正因为 L1 缓存行是 128 字节，比 L2 的 32 字节大 4 倍，相邻 warp 的访问地址往往都落在同一个 128 字节缓存行内，从而产生大量缓存复用，弥补了非对齐带来的部分性能损失——这解释了第 2.5.3 节"理论 80%、实测 90%"的差距来源。

### 5.4 跨步访问（Strided Access）的缓存命中影响

Best Practices Guide（§10.2.1.4）：

> *"This action leads to a load of **eight L2 cache segments** per warp on the Tesla V100 (compute capability 7.0)."*（stride=2 的情况）
> *"A stride of 2 results in a **50% of load/store efficiency** since half the elements in the transaction are not used and represent wasted bandwidth."*

```
stride=1（连续访问）：4 个 32 字节段 → 128 字节 → 1 个 L1 缓存行，100% 利用率
stride=2（间隔访问）：8 个 32 字节段 → 256 字节 → 跨 2 个 L1 缓存行，50% 利用率
stride=32（极端跨步）：32 个 32 字节段 → 1024 字节 → 跨 8 个 L1 缓存行，~3% 利用率
```

跨步越大，浪费越严重，且与非对齐不同——**跨步浪费无法被相邻 warp 复用弥补**（跳过的数据没有任何线程需要）。这是"避免跨步访问"比"避免非对齐"更重要的原因。

---

## 第 6 章 概念体系总览与架构演进

### 6.1 概念定义对照表

| 概念 | 定义 | 大小（CC 6.0+）| 描述视角 |
|------|------|----------------|---------|
| **内存段（Segment）** | DRAM 按固定大小切分的地址块，起始地址 = n × 段大小 | 32 / 64 / 128 字节 | DRAM 物理地址结构 |
| **事务粒度** | 内存总线一次物理传输的最小数据量 | **32 字节** | 硬件内存总线视角 |
| **数据访问单元** | warp 一次内存请求触发的最小数据量 | **32 字节** | warp/软件请求视角 |
| **L2 缓存行** | L2 内部一个存储槽位的大小 | **32 字节** | L2 缓存存储结构 |
| **L1 缓存行** | L1 内部一个存储槽位的大小 | **128 字节** | L1 缓存存储结构 |

### 6.2 各代 GPU 缓存体系演进（官方文档汇总）

| 架构 | CC | L1 大小（全局内存可用部分）| L2 | 全局内存 L1 默认策略 |
|------|----|--------------------------|-----|---------------------|
| Maxwell | 5.0 | 24 KB（仅只读数据） | 共享 | 关闭；仅 `__ldg()` 走 L1 |
| Maxwell | 5.2 | 24 KB | 共享 | 关闭；`-dlcm=ca` 可开启 |
| Pascal  | 6.0/6.2 | 24 KB | 共享 | **默认开启**；32 字节访问单元 |
| Pascal  | 6.1 | 48 KB | 共享 | **默认开启**；32 字节访问单元 |
| Volta   | 7.0 | 最多 128 KB（与 Shared Memory 共享，可配 0~96 KB 给 Shared）| 共享 | 默认开启 |
| Turing  | 7.5 | 最多 96 KB（可配 32/64 KB 给 Shared）| 共享 | 默认开启 |
| Ampere  | 8.0/8.7 | 最多 192 KB（与 Shared Memory 共享）| 40 MB（A100）| 默认开启 |

### 6.3 层次结构全景图

```
warp 发出 load 指令（访问 float arr[tid]）
              │
              ▼
  ┌─────────────────────────────────────┐
  │       数据访问单元 = 32 字节         │  ← warp 的每次请求以 32 字节为单位
  │  warp 的 32 线程 × 4字节 = 128字节  │    合并为 4 个 32 字节请求
  └──────────────┬──────────────────────┘
                 │ 4 个 32 字节请求
                 ▼
  ┌─────────────────────────────────────┐
  │            L1 Cache                 │  ← SM 私有，片上
  │       缓存行大小 = 128 字节          │    一个槽位存 128 字节
  │       (~20-30 周期延迟)             │    4 个 32B 请求 → 查 1 个 128B 缓存行
  └──────────────┬──────────────────────┘
                 │ L1 miss → 向 L2 发 4 个 32 字节请求
                 ▼
  ┌─────────────────────────────────────┐
  │            L2 Cache                 │  ← 全芯片共享，片上
  │       缓存行大小 = 32 字节           │    每个 32B 请求独立查询
  │       (~200 周期延迟)               │    4 次 32B 命中 → 返回给 L1 填缓存行
  └──────────────┬──────────────────────┘
                 │ L2 miss → 向 DRAM 发 32 字节事务
                 ▼
  ┌─────────────────────────────────────┐
  │         Global Memory（DRAM）        │  ← 片外，高延迟
  │       事务粒度 = 32 字节             │    (~600-800 周期延迟)
  │       内存段 = 32/64/128 字节        │
  └─────────────────────────────────────┘
```

---

## 第 7 章 最佳实践

基于以上分析，提高全局内存访问效率的关键原则：

| 原则 | 说明 | 文档依据 |
|------|------|---------|
| **使用 `cudaMalloc` 分配内存** | 自动保证 ≥256 字节对齐，所有基本类型天然对齐 | Programming Guide §8.3.2 |
| **线程块大小为 warp 大小的倍数** | 保证每个 warp 的访问地址在同一对齐块内 | Best Practices Guide §10.2.1.2 |
| **连续线程访问连续地址（stride=1）** | 4 个 32 字节事务，100% 带宽利用率 | Best Practices Guide §10.2.1.1 |
| **避免跨步访问** | stride=2 → 50% 效率；用 Shared Memory 中转 | Best Practices Guide §10.2.1.4 |
| **结构体使用 `__align__(8/16)`** | 防止多指令访问，保证自然对齐 | Programming Guide §8.3.2 |
| **2D 数组使用 `cudaMallocPitch`** | 自动 padding，保证每行地址对齐 | Programming Guide §8.3.2 |
| **频繁重用数据用 L2 持久化（CC 8.0+）** | `cudaAccessPropertyPersisting`，减少 DRAM 访问 | Programming Guide §6.2.3 |
| **只读数据使用 `__ldg()` 或 `const __restrict__`** | 走 L1 read-only cache，带宽更高 | Programming Guide §20.4.2 |

> 这些原则在本仓库其他章节中反复出现：`01_reduce` 的 V3（相邻线程访问相邻地址）与 V7（float4 向量化 + 对齐加载）正是"连续对齐 + 宽类型"两条原则的实战应用。

---

## 第 8 章 配套基准测试

### 8.1 程序说明

[`main.cu`](./main.cu) 复现了 Best Practices Guide §10.2.1.3（offset copy）与 §10.2.1.4（strided access）的经典实验，实测两类非理想访问模式对有效带宽的影响：

| 实验 | Kernel | 验证的结论 |
|------|--------|-----------|
| Offset 实验 | `offset_access`：`out[i+offset] = in[i+offset] + 1` | 非对齐访问约损失 10~20% 带宽（第 2.5.3 节） |
| Stride 实验 | `stride_access`：`out[i*stride] = in[i*stride] + 1` | stride=2 约 50% 效率，随 stride 增大急剧恶化（第 5.4 节） |

### 8.2 编译与运行

```bash
cd 00_cuda_global_mem_access

# 编译（-O3 优化；架构参数按设备调整，或省略使用默认值）
nvcc -O3 -o mem_benchmark main.cu

# 运行
./mem_benchmark
```

### 8.3 预期输出形态

```
Device: NVIDIA A100-SXM4-80GB (CC 8.0)
Offset array: 64 MB, stride array: 16 MB (x32), block size: 256

=== Offset access (out[i+offset] = in[i+offset] + 1) ===
offset      bandwidth (GB/s)    relative
     0             1550.23        100.0%
     1             1402.11         90.4%      <- 非对齐，约 9/10（L1 复用补偿）
     ...
    32             1548.90         99.9%      <- 偏移 32 元素 = 128B，重新对齐

=== Strided access (out[i*stride] = in[i*stride] + 1) ===
stride      bandwidth (GB/s)    relative
     1             1549.87        100.0%
     2              812.44         52.4%      <- 约 1/2
     4              401.15         25.9%      <- 约 1/4
     ...
    32              48.33           3.1%      <- 极端跨步，~3%
```

具体数值随 GPU 型号变化，但**相对趋势**应与第 2.5.3、5.4 节的官方数据一致。

> **测量注意**：数组规模应显著大于 L2 容量（A100 为 40 MB），否则测得的是缓存带宽而非 DRAM 带宽；显存较小的设备可在源码中调小 `kStrideElems`（stride 实验的寻址上界为 `kStrideElems × 32` 个 float）。
