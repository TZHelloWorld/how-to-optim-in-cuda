# CUDA GEMM（矩阵乘）算子优化指南

> 本文以 SGEMM（单精度通用矩阵乘）为例，系统介绍 CUDA 上**计算受限（compute-bound）算子**的典型优化方法。GEMM 优化的目标是"跑满算力"，核心手段是**数据复用**——让每个从内存搬进来的字节参与尽可能多的计算。全文从最朴素的实现出发，沿着"分析数据流 → 定位复用不足的层次 → 在该层次建立分块"的主线，逐步演进出 8 个版本（V0~V7），从不足 cuBLAS 2% 的性能一路逼近硬件峰值，最后过渡到 Tensor Core 与工程实践。
>
> 本文内容完全自包含：理解全文所需的 GPU 执行模型、内存层次、合并访存、Bank Conflict、占用率等基础概念，都在第 2 章从零讲起，无需先阅读其他资料。

---

## 目录

- [第 1 章 问题定义：什么是 GEMM](#第-1-章-问题定义什么是-gemm)
- [第 2 章 预备知识：GPU 执行模型、内存层次与 Roofline 模型](#第-2-章-预备知识gpu-执行模型内存层次与-roofline-模型)
- [第 3 章 优化路线总览](#第-3-章-优化路线总览)
- [第 4 章 V0：基准实现——一线程一元素](#第-4-章-v0基准实现一线程一元素)
- [第 5 章 V1：合并访存——修正线程到矩阵的映射](#第-5-章-v1合并访存修正线程到矩阵的映射)
- [第 6 章 V2：共享内存分块——Block Tiling](#第-6-章-v2共享内存分块block-tiling)
- [第 7 章 V3：寄存器分块入门——一维 Thread Tiling](#第-7-章-v3寄存器分块入门一维-thread-tiling)
- [第 8 章 V4：二维 Thread Tiling——外积累加](#第-8-章-v4二维-thread-tiling外积累加)
- [第 9 章 V5：float4 向量化与共享内存布局重排](#第-9-章-v5float4-向量化与共享内存布局重排)
- [第 10 章 V6：双缓冲——用计算掩盖访存延迟](#第-10-章-v6双缓冲用计算掩盖访存延迟)
- [第 11 章 V7：Tensor Core——WMMA 半精度矩阵乘](#第-11-章-v7tensor-corewmma-半精度矩阵乘)
- [第 12 章 工程化：PyTorch 扩展与 cuBLAS 对比](#第-12-章-工程化pytorch-扩展与-cublas-对比)
- [第 13 章 总结与实践建议](#第-13-章-总结与实践建议)
- [附录：关键概念速查](#附录关键概念速查)

---

## 第 1 章 问题定义：什么是 GEMM

### 1.1 GEMM 的含义

GEMM（GEneral Matrix Multiplication，通用矩阵乘）是 BLAS Level-3 的核心例程，完整定义为：

```
C = α · A × B + β · C      A: M×K,  B: K×N,  C: M×N
```

为聚焦优化本身，本文取 α=1、β=0，即纯矩阵乘 `C = A × B`，且约定三个矩阵都按**行主序（row-major）**存储——即同一行的元素在内存中连续排列，`A[m][k]` 的线性地址为 `m * K + k`。CPU 上的参考实现是三重循环：

```c
for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += A[m * K + k] * B[k * N + n];
        C[m * N + n] = acc;
    }
```

三层循环的含义：外两层枚举输出矩阵 C 的每个位置 (m, n)，最内层沿 K 维做**点积**——A 的第 m 行与 B 的第 n 列逐元素相乘并累加。

GEMM 是深度学习中当之无愧的第一算子：全连接层、卷积（im2col / implicit GEMM）、Attention 中的 QKᵀ 与 PV、LoRA 的低秩投影，本质都是矩阵乘。GPU 上 90% 以上的算力通常花在 GEMM 及其变体上——这也是为什么 NVIDIA 会为它设计专用硬件（Tensor Core）。

### 1.2 计算量与访存量：GEMM 最重要的规模特征

先算两笔账（以 fp32、每元素 4 字节计）：

| 量 | 表达式 | M=N=K=4096 时 |
|---|---|---|
| 计算量（FLOP） | 2·M·N·K（每个输出元素做 K 次乘加，1 次乘加 = 2 FLOP） | 2×4096³ ≈ **137 GFLOP** |
| 最少访存量（字节） | 4·(MK + KN + MN)（三个矩阵各读/写一遍） | 4096²×3×4 B ≈ **201 MB** |

**计算量是数据量的近 700 倍**。这意味着同一个数据"理论上"会被反复使用：A 的每个元素要参与 N 次乘加（与 B 的每一列各配对一次），B 的每个元素要参与 M 次。

能不能把这种"理论上的复用"变成"硬件上的复用"（让数据留在快速存储中被反复读取，而不是反复去全局内存搬），就是 GEMM 优化的全部内容。

作为对照：向量求和（Reduce）这类算子对 N 个数只做 N 次加法、却要做 N 次读取，计算量与访存量同阶，性能天花板是内存带宽，优化的主题是"消除浪费、跑满带宽"。GEMM 恰好相反——**优化的主题是"建立复用、跑满算力"**。两类算子的判别标准将在 2.8 节用算术强度精确刻画。

### 1.3 并行化的基本形状

矩阵乘的输出天然是二维的，最直接的并行方案是**每个线程负责 C 的一个元素**，二维 Grid/Block 与矩阵形状一一对应：

```
        N (列, x 方向)
      ┌────────────────────┐
      │  Block(0,0) Block(1,0) ...   ← 每个 Block 负责 C 的一个矩形子块
   M  │  Block(0,1) ...              ← 每个 Thread 负责子块中的一个元素
(行,y)│  ...
      └────────────────────┘
```

这个"一线程一元素"的方案就是 V0。它功能正确，但性能通常只有 cuBLAS 的 1%~2%——差距全部来自数据复用与访存方式，后续 7 个版本将逐一补齐。

---

## 第 2 章 预备知识：GPU 执行模型、内存层次与 Roofline 模型

本章从零介绍理解全文所必需的全部硬件概念。已熟悉 CUDA 基础的读者可以只读 2.6、2.8~2.10 四节（贯穿全文的分析方法、性能模型与指令视角），其余跳过。

### 2.1 线程层次：Grid → Block → Thread

CUDA 将一次 kernel 启动的所有线程组织为三层结构：

```
Grid（网格）—— 一次 kernel 启动的全部线程
 ├── Block 0（线程块，例如 256 或 1024 线程）
 │    ├── Thread 0
 │    ├── ...
 │    └── Thread 255
 ├── Block 1
 └── ...
```

- **Thread**：最小执行单位，有自己的寄存器；
- **Block**：一组线程，运行在同一个 SM（Streaming Multiprocessor，GPU 的"核心"）上，可通过**共享内存**通信、用 `__syncthreads()` 栅栏同步；
- **Grid**：全部 Block 的集合，Block 之间无法直接同步（只能通过结束 kernel 或全局内存原子操作）。

常用内置变量（Grid 与 Block 都可以是一维、二维或三维的，用 `dim3` 指定）：

| 变量 | 含义 |
|------|------|
| `threadIdx.x / .y / .z` | 线程在 Block 内的坐标 |
| `blockIdx.x / .y / .z` | Block 在 Grid 中的坐标 |
| `blockDim.x / .y / .z` | Block 各维度的线程数 |
| `gridDim.x / .y / .z` | Grid 各维度的 Block 数 |

GEMM 常用二维配置，例如 `dim3 block(32, 32)` 表示每个 Block 有 32×32 = 1024 个线程。一个常用的全局坐标算法：

```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

**一个后文反复用到的细节**：多维 Block 内的线程会被"拉平"成一维编号，规则是 **x 维变化最快**：

```
线性编号 tid = threadIdx.x + threadIdx.y * blockDim.x (+ threadIdx.z * blockDim.x * blockDim.y)
```

也就是说，Block(32, 32) 中，`(threadIdx.x=0..31, threadIdx.y=0)` 这 32 个线程的线性编号是 0~31——它们是"连续"的线程。这个拉平规则直接决定了 Warp 的构成方式（见下节），进而决定访存性能（见 2.4 节），是 V0→V1 优化的关键。

### 2.2 Warp：真正的硬件调度单位

GPU 硬件并不逐个调度线程，而是以 **Warp（按线性编号连续的 32 个线程）** 为单位调度执行。同一 Warp 内的线程**锁步执行同一条指令**，作用于各自的数据——这一模型称为 SIMT（Single Instruction, Multiple Threads）。

由此引出三条重要推论：

1. **Warp Divergence（线程束分化）**：若同一 Warp 内的线程走了不同分支，硬件必须依次执行所有分支路径（不参与的线程结果被屏蔽），分支越多性能越差；
2. **访存以 Warp 为单位发出**：一个 Warp 的 32 个线程各自的地址会被硬件合并成内存事务（详见 2.4 节），32 个地址的"形状"决定了访存效率；
3. **Warp 内天然同步**：同一 Warp 的 32 个线程步调一致，Warp 内部通信不需要 `__syncthreads()`。

一个 256 线程的 Block 包含 8 个 Warp：Warp 0 = tid 0~31，Warp 1 = tid 32~63，以此类推。对于二维 Block(32, 32)，按 2.1 节的拉平规则，**每个 Warp 恰好是 `threadIdx.y` 相同、`threadIdx.x` 从 0 到 31 的一行线程**。

### 2.3 内存层次

GPU 的存储从慢到快分为多层，容量与速度反向变化：

```
速度：   慢 ──────────────────────────────────────────────────────── 快
         全局内存(Global)   L2 缓存    共享内存(Shared)    寄存器(Register)
容量：   几十 GB            几十 MB    ~100 KB / SM        每线程最多 255 个
延迟：   ~400-600 cycles    ~200      ~20-30 cycles       ~1 cycle
可见性： 所有线程           所有线程   同一 Block 内        仅本线程
控制：   cudaMalloc 分配    硬件自动   __shared__ 声明      编译器分配
```

几个要点：

- **全局内存（显存）** 是 kernel 输入输出所在的地方，也是最慢的一层。它的带宽（如 1 TB/s 量级）看似巨大，但相对 GPU 的算力（几十 TFLOPS）仍然远远不够——见 2.8 节的量化；
- **共享内存**是每个 SM 上的一块高速便签内存（scratchpad），由程序员显式管理，同一 Block 的所有线程可见。它是"让 Block 内线程共享数据"的唯一高效手段，也是 GEMM 分块优化（V2 起）的物理载体；
- **寄存器**最快但只属于单个线程。每个 SM 有一个总大小固定的寄存器文件（如 65536 个 32-bit 寄存器），被驻留在该 SM 上的所有线程瓜分——单线程用得越多，能同时驻留的线程越少（见 2.7 节的占用率）。

优化的总体方向：**让数据尽量在更快的层次上被反复使用**。GEMM 的三级分块（Block Tiling → Thread Tiling）就是把复用逐级建立到共享内存和寄存器上。

### 2.4 合并访存（Memory Coalescing）

全局内存的读写以 **内存事务（transaction）** 为单位进行，事务粒度通常为 32 字节（一个 sector），一次最多合并到 128 字节。当一个 Warp 执行一条加载指令时，硬件收集 32 个线程各自的地址，把落在同一事务范围内的请求合并：

- **最好情况**：32 个线程访问 32 个连续的 float（共 128 字节，且对齐）→ 合并为 4 个 32 字节事务，每个字节都是有用的，带宽利用率 100%；
- **最坏情况**：32 个线程的地址彼此相距很远（如相隔一整行矩阵）→ 需要 32 个独立事务，每个 32 字节的事务只有 4 个字节有用，带宽利用率 12.5%，等效带宽只剩 1/8 甚至更低；
- **特殊情况——广播**：32 个线程读同一个地址 → 1 个事务即可服务全部线程，效率很高。

由此得出全局内存访问的黄金法则：

> **让 Warp 内相邻的线程（threadIdx.x 相邻）访问相邻的内存地址。**

对行主序矩阵而言，"相邻地址"就是"同一行内相邻的列"。这条法则是 V1 的全部内容，也是后续所有版本加载数据时始终要维持的性质。

### 2.5 共享内存的 Bank 结构与 Bank Conflict

共享内存被划分为 **32 个 Bank（存储体）**，地址按 4 字节交错分布：以 float 为单位，地址 `i` 属于 `bank = i % 32`：

```
地址:  0  1  2  ... 31 32 33 ... 63 64 ...
bank:  0  1  2  ... 31  0  1 ... 31  0 ...
```

每个 Bank 每周期只能服务一次访问。同一 Warp 内的多个线程若同时访问**同一 Bank 的不同地址**，访问会被串行化，称为 **Bank Conflict**：n 个线程冲突则退化为 n 次串行访问（n-way conflict）。两个不算冲突的例外：

- 多个线程读**同一地址** → 硬件广播，一次完成；
- 32 个线程访问的地址恰好落在 32 个不同 Bank → 一次并行完成。

实用推论：Warp 内 32 个线程访问共享内存中 32 个**连续的 float**（如 `smem[tid]`），bank 编号恰好 0~31 各一个，零冲突；而如果访问间距是 32 的倍数（如 `smem[tid * 32]`），所有线程全部命中 bank 0，退化为 32 路串行——这正是 V5 中"As 按列访问"要避免的陷阱。

### 2.6 从代码到访存行为：Warp 访存模式四步分析法

2.4 与 2.5 两节给出的是"判据"：什么样的地址模式是好的、什么样是坏的。但初学者拿到一段 kernel 代码时，最大的困难往往不是记不住判据，而是**不知道如何从代码推导出"硬件实际看到的地址模式"**——代码里写的是二维下标和循环变量，硬件看到的却是某个瞬间 32 个线程发出的 32 个一维地址。本节给出一套机械化的分析流程，把这个推导过程标准化。后文对每个版本的访存分析（第 4~9 章）都是这套流程的反复应用。

#### 前提认知一：内存是一维的，多维数组是人为约定

内存中不存在二维数组，只有一条按字节编址的一维地址空间。所谓 M×K 的行主序矩阵 A，是"每 K 个元素折为一行"的约定：

```
逻辑视图（二维）:                  物理视图（一维纸带）:

     k=0  k=1  k=2  k=3
row0  a    b    c    d            [a b c d │ e f g h │ i j k l]
row1  e    f    g    h             ↑第0行    ↑第1行     ↑第2行
row2  i    j    k    l
                                  A[row][k] 的线性地址 = row * K + k
```

由地址公式可直接读出两条**步长铁律**：

- 沿**行方向**（k 变化 1）：地址变化 **1**——相邻；
- 沿**列方向**（row 变化 1）：地址变化 **K**——跳过一整行（K=4096、fp32 时一步 16 KB）。

任何多维数组访问，第一步都应先在头脑中（或纸上）还原成这样的线性地址表达式。

#### 前提认知二：分析单位是 Warp，不是单个线程

由 2.2 节，硬件以 Warp 为单位锁步执行：**同一时刻，Warp 内 32 个线程执行同一条语句，唯一的区别是各自的 threadIdx 不同**。因此"这条加载语句的访存行为"不由单个线程决定，而由 32 个线程代入各自线程号后得到的 **32 个地址的集合形状**决定。

又因为所有 Warp 的行为在结构上对称（只差一个整体偏移），**分析一个代表 Warp 即可推知全局**。

#### 四步分析法

对 kernel 中任意一条内存访问语句，依次执行：

| 步骤 | 操作 | 说明 |
|------|------|------|
| ① 拍扁 | 把访问改写成线性地址表达式 | 如 `A[row][k]` → `row * K + k`，并把 `row`/`col` 代换成含 `threadIdx` 的式子 |
| ② 抓一个代表 Warp | 取 `threadIdx.x = 0..31`，其余线程坐标固定为常数 | 依据 2.1 节拉平规则，这 32 个线程构成一个 Warp |
| ③ 冻结一个时刻 | 把所有循环变量固定为具体值（如 k=0） | 锁步执行意味着"同一时刻"即"同一循环迭代" |
| ④ 求相邻线程的地址差 Δ | 计算 t 号与 t+1 号线程的地址之差 | Δ 即地址表达式中 **threadIdx.x 的系数**，查下表判定 |

第④步的判定表（以 fp32、4 字节元素计）：

| Δ（元素） | 访存模式 | 硬件行为 | 效率 |
|-----------|---------|---------|------|
| 0 | 全 Warp 同地址 | 广播，1 次传输 | 优 |
| 1 | 32 个连续地址（128 B） | 完全合并，4 个 32B 事务 | 优（100%） |
| 2~7 | 有间隙的跨步访问 | 部分合并，事务数按跨度增加 | 中，应尽量避免 |
| ≥8（如 K、N） | 每线程独占一个事务 | 32 个分散事务，每 32B 只用 4B | 差（≤12.5%） |

一条实用捷径：地址表达式几乎总是 threadIdx.x 的**仿射函数**（形如 `a·threadIdx.x + b`），所以不必真的写出 32 个地址——**直接看 threadIdx.x 的系数 a**：系数为 0 是广播，为 1 是完全合并，为大常数（K、N、BK…）是灾难。

#### 变体：共享内存的 Bank 分析

同一套流程稍作修改即可分析 Bank Conflict：前三步不变，第④步改为**把 32 个地址对 32 取模，统计落在同一 Bank 的线程数**（依据 2.5 节 `bank = 地址 % 32`）：

| 地址模式 | Bank 分布 | 结论 |
|---------|----------|------|
| Δ = 0（同地址） | 同 Bank 同地址 | 广播，无冲突 |
| Δ = 1 | 恰好铺满 32 个 Bank | 无冲突 |
| Δ 为奇数 | 32 个地址模 32 互不相同 | 无冲突 |
| Δ = 32 的倍数 | 全部命中同一 Bank | **32 路冲突，串行化** |

#### 用 profiler 验证纸面分析

纸面推导应当与实测互相印证。Nsight Compute（`ncu`）中的对应指标：

- 全局内存：`l1tex__average_t_sectors_per_request`（每请求扇区数，完全合并时为 4）、旧指标 `gld_efficiency`（V0 约 12.5%，V1 接近 100%）；
- 共享内存：`l1tex__data_bank_conflicts_pipe_lsu`（Bank 冲突次数）。

> 第 4 章（V0）与第 5 章（V1）将把这套四步法完整地走一遍，读者可以先自行对 `A[row * K + k]` 在两种映射下分别执行四步，再对照正文验证。

### 2.7 占用率（Occupancy）与延迟隐藏

GPU 掩盖长延迟操作（如一次全局内存读要等几百个周期）的基本机制是**超额订阅**：每个 SM 上同时驻留远多于执行单元数量的 Warp，某个 Warp 因等待数据而停顿时，调度器立即切换到其他就绪的 Warp——切换零开销，因为所有驻留线程的寄存器都常驻在寄存器文件中。

**占用率** = SM 上实际驻留的 Warp 数 / 硬件支持的最大驻留 Warp 数。驻留数量受三种资源限制，取最小者：

| 资源 | 每 SM 总量（典型值） | 限制方式 |
|------|-----|---------|
| 寄存器文件 | 65536 个 | 每线程用 R 个 → 最多驻留 65536/R 线程 |
| 共享内存 | ~100 KB | 每 Block 用 S KB → 最多驻留 100/S 个 Block |
| 线程槽位 | 1536~2048 线程 | 硬性上限 |

**但高占用率不是目的，掩盖延迟才是**。掩盖延迟有两条途径：

- **TLP（线程级并行）**：靠大量 Warp 轮转——访存受限算子的主要手段，需要高占用率；
- **ILP（指令级并行）**：靠单个线程内多条**互不依赖**的指令连续发射——如果一个线程有 64 条独立的乘加链，即使占用率只有 25%，流水线也能填满。

GEMM 恰好走第二条路：V4 起每个线程持有 8×8 = 64 个独立累加器，寄存器用量大、占用率低，但 ILP 极其充足。**"用低占用率换每线程高复用"是 GEMM 优化的标志性权衡**，与访存受限算子的调优方向相反。

### 2.8 算术强度（Arithmetic Intensity）与 Roofline 模型

有了以上硬件图景，现在回答一个根本问题：**一个算子的性能上限由什么决定？**

定义 **算术强度** 为计算量与访存量之比：

```
AI = FLOP 数 / 访存字节数    （单位：FLOP/Byte）
```

硬件有一个平衡点：`峰值算力 / 峰值带宽`。以一块典型 GPU 为例（fp32 算力 ~35 TFLOPS，显存带宽 ~1 TB/s），平衡点约为 **35 FLOP/Byte**。Roofline 模型把两者画在一张图上：

```
可达性能
(FLOP/s)                    ┌───────────────  算力屋顶（35 TFLOPS）
   │                   ／
   │              ／   ↑
   │         ／        平衡点 AI ≈ 35
   │    ／ ← 带宽屋顶（斜率 = 1 TB/s）
   └──────┴────────────────────────→ 算术强度 (FLOP/Byte)
      memory-bound │ compute-bound
```

- 算子 AI < 平衡点：性能撞上带宽斜线，**memory-bound（访存受限）**——如向量求和，读 4 字节做 1 次加法，AI = 0.25，无论怎么优化计算都没用，目标只能是把带宽跑满；
- 算子 AI > 平衡点：性能撞上算力横线，**compute-bound（计算受限）**——目标是把算力跑满。

GEMM 的理论算术强度（按 1.2 节的最少访存量计算）：

```
AI = 2MNK / 4(MK + KN + MN)     （M=N=K 时 ≈ K/6）
```

K=4096 时 AI ≈ 683 FLOP/Byte，远超平衡点——**GEMM 天生是 compute-bound 算子，理论上限是算力峰值**。

**但朴素实现是 memory-bound 的。** 上面的 AI 用的是"理论最少访存量"。V0 每个线程独立地从全局内存读 2K 个数做 K 次乘加，谁也不复用谁的数据，**实际**访存量是 2·M·N·K 个 float：

```
实际 AI(V0) = 2MNK / (4 · 2MNK) = 0.25 FLOP/Byte
```

与向量求和一样落在带宽斜线的最底端。也就是说：**GEMM 的优化过程，本质是把实际 AI 从 0.25 一路提升到接近理论值 K/6 的过程**，手段是在各级存储上建立数据复用：

```
复用层次           容量           带宽/延迟         建立复用的手段
────────────────────────────────────────────────────────────────
全局内存 → L2      几十 MB        自动，不可控       （Block 调度顺序影响命中率）
全局内存 → 共享内存 ~100KB/SM     ~20-30 cycles     Block Tiling（V2）
共享内存 → 寄存器   255 个/线程    ~1 cycle          Thread Tiling（V3/V4）
```

### 2.9 指令视角：FFMA、LDS/LDG 与"发射"

前面各节都在讨论"数据在哪、怎么搬"；本节下沉到**指令层面**——kernel 里的每一行 C++ 最终都变成一条条机器指令在 SM 上排队执行，而 GEMM 优化的后半程（V4~V6）恰恰是在指令层面展开的。本文正文反复出现的 FFMA、LDS、LDG、"发射端口"等名词，都在此一次讲清。

#### 2.9.1 从 C++ 到 SASS：文中指令名的来历

nvcc 的编译分两级：C++ → **PTX**（跨代虚拟指令集）→ **SASS**（特定架构的真实机器码，可用 `cuobjdump --dump-sass` 查看）。本文出现的大写指令名都是 SASS 指令：

| SASS 指令 | 含义 | 对应的 C++ 写法 |
|----------|------|----------------|
| `FFMA d,a,b,c` | fp32 融合乘加 d = a×b + c | `acc += x * y` |
| `LDG.32 / LDG.128` | 从**全局内存**加载 4 / 16 字节（G = global） | 读 `A[i]` / 读 `float4` |
| `STG.32 / STG.128` | 向全局内存写 4 / 16 字节 | 写 `C[i]` |
| `LDS.32 / LDS.128` | 从**共享内存**加载 4 / 16 字节（S = shared） | 读 `As[i]` |
| `STS.32 / STS.128` | 向共享内存写 | `As[i] = ...` |
| `HMMA`（PTX 层 `mma`） | Tensor Core 小矩阵乘加 | `wmma::mma_sync`（第 11 章） |

后缀 `.32/.64/.128` 表示**一条指令搬运的位宽**：一条 LDS.128 顶四条 LDS.32。记住这一点，V5 向量化的收益就一目了然——数据量不变，**指令条数变为 1/4**。

#### 2.9.2 FMA：为什么乘加算"一条指令、2 FLOP"

FMA（Fused Multiply-Add，融合乘加）把 `d = a×b + c` 的乘法与加法合成**一条指令、一次舍入**。它是 GPU 算力的基本计量单位：

- 1 条 FMA = 2 FLOP（一乘 + 一加）；
- GPU 标称的 fp32 峰值算力 = CUDA Core 数 × 2 FLOP × 主频，其隐含假设是**每个 core 每周期恰好完成一条 FFMA**；
- 命名规则：FFMA 是 fp32 版本（首字母 F = float），fp64 为 DFMA，fp16 为 HFMA2（一条算两对），Tensor Core 的 HMMA 则一条指令完成一整个小矩阵乘加。

由此得到一个贯穿全文的判据：**峰值算力是"指令流全是 FFMA、且每周期都能发射一条"时的极限**。任何挤占 FFMA 发射机会的指令，都在直接扣减可达算力。

#### 2.9.3 发射（Issue）：每周期一个名额的稀缺资源

SM 内部划分为 4 个调度分区（processing block），每个分区有一个 Warp 调度器，**每个时钟周期最多"发射"一条指令**——从驻留的 Warp 中挑一个就绪的，把它的下一条指令派发给对应的执行单元。执行单元有多种、各自独立干活：

```
Warp 调度器（每周期发射 1 条）
        │
        ├──► FP32 单元        ← FFMA 在这里执行
        ├──► LSU（Load/Store）← LDG / LDS / STS / STG 在这里执行
        ├──► SFU              ← exp、rsqrt 等特殊函数
        └──► Tensor Core      ← HMMA
```

关键在于：**执行单元各自独立，但"发射口"只有一个**。一条 LDS 虽然由 LSU 执行、并不占用 FP32 单元，却占掉了这个周期唯一的发射名额——FFMA 只能排到下一个周期。这就是文中"LDS 与 FFMA 竞争同一个指令发射端口"的确切含义。

由此可以直接算出算力上限（这也是 2.10 节工具的数学根据）：若指令流中每 1 条 FFMA 配 2 条 LDS，则每 3 个发射周期只有 1 个给了 FFMA：

```
可达算力 ≤ 峰值 × FFMA 在指令流中的占比 = 峰值 × 1/3
```

- V2 的 LDS/FMA = 2 → 上限 1/3，正是这么算出来的（6.7 节）；
- V4 压到 0.25 → 每 5 条指令 4 条是 FFMA，上限 80%（8.1 节）；
- V5 用 .128 位宽把 LDS 条数再削 4 倍，为流水线腾出更多发射名额（9.4 节）。

#### 2.9.4 发射 ≠ 执行完成：延迟、吞吐与记分板

三个必须区分的概念：

| 概念 | 含义 | FFMA | LDS | LDG |
|------|------|------|-----|-----|
| **发射（issue）** | 占用调度器名额的那**一个**周期 | 1 slot | 1 slot | 1 slot |
| **延迟（latency）** | 发射后多少周期结果才可用 | ~4 | ~20-30 | ~400-600 |
| **吞吐（throughput）** | 流水线稳定后平均每周期完成几条 | 高 | 中（受 Bank 带宽限制） | 低（受 HBM 带宽限制） |

注意访存指令的一个重要行为：**发射之后线程并不停下**。硬件用**记分板（scoreboard）**记住"某寄存器的数据还在路上"，只有当后续某条指令**真正要用**那个寄存器时，Warp 才停顿等待。两个直接推论：

1. 2.7 节说的"掩盖延迟"（TLP/ILP），本质都是**在等待期间让调度器找得到别的可发射指令**——要么换一个 Warp（TLP），要么本 Warp 还有不依赖该数据的指令可发（ILP）；
2. V6 双缓冲的指令编排——先发出下一块的 LDG、隔上几百条 FFMA 再使用其数据——正是**手动拉开"发射"与"使用"的距离**，让 LDG 的数百周期延迟被中间的计算流填满（10.1 节）。

#### 2.9.5 用工具验证

纸面分析应与实测互相印证：

- `cuobjdump --dump-sass`（或 Nsight Compute 的 Source/SASS 页）：直接数内层循环里 FFMA 与 LDS 的真实条数与配比，检验编译器是否生成了预期的 `.128` 指令；
- Nsight Compute 指标：`smsp__issue_active`（发射口利用率）、`smsp__inst_executed_pipe_fma / _lsu`（各执行管线的指令数）。你在 2.10 节算出的 LDS/FMA 比值，应当与这些计数一致。

### 2.10 另一条贯穿全文的分析工具：每次 FMA 需要几次访存

2.6 节的四步法回答访存的**形状**问题（地址模式是否合并）；本节基于 2.9 节的发射模型，回答访存的**数量**问题。后文每个版本都会用同一个问题来定位瓶颈：**平均每做一次 FMA，需要从"上一级存储"读几个数？**

这个视角之所以关键，正如 2.9.3 节所算：访存指令与计算指令竞争同一个发射口——

- 每次 FMA 配 2 次 LDS 时，指令流中 2/3 是访存指令，算力最多发挥约 1/3；
- 优化目标是把 **LDS/FMA 比值**降到远小于 1，让发射名额几乎全部留给 FFMA。

这个比值将解释：为什么 V2（已经用上共享内存）依然很慢；为什么 Thread Tiling 必须做成"二维"才有效（V3 → V4）；以及向量化访存（V5）的收益从哪来。

---

## 第 3 章 优化路线总览

每个版本针对上一版暴露的一个具体瓶颈：

```
V0: 朴素实现，一线程一元素（基准）
 │   瓶颈：线程到矩阵的映射不当 → 全局内存访问完全不合并
 ▼
V1: 交换行列映射，Warp 内线程访问连续地址
 │   瓶颈：数据零复用，每次 FMA 配 2 次全局内存读，实际 AI = 0.25
 ▼
V2: Block Tiling，子块搬入共享内存复用
 │   瓶颈：每次 FMA 仍配 2 次共享内存读，指令发射被 LDS 占满
 ▼
V3: 一维 Thread Tiling，每线程算 8 个输出，中间值驻留寄存器
 │   瓶颈：只在一个维度复用，每 8 次 FMA 仍需 9 次 LDS
 ▼
V4: 二维 Thread Tiling（8×8），外积累加，LDS/FMA 降到 1/4
 │   瓶颈：加载共享内存是标量指令；As 列访问方向与存储方向不一致
 ▼
V5: float4 向量化读写 + As 转置存储，访存指令数减为 1/4
 │   瓶颈：加载与计算串行：算完一块 → 停下来搬下一块 → 再算
 ▼
V6: 双缓冲（Double Buffering），搬运与计算流水线重叠
 │   瓶颈：CUDA Core 的 fp32 FFMA 吞吐本身成为天花板
 ▼
V7: Tensor Core（WMMA），专用矩阵乘硬件，半精度吞吐 ~8-16 倍
```

各版本解决的瓶颈分类：

| 瓶颈类别 | 具体问题 | 解决版本 |
|---------|---------|---------|
| 访存效率 | 全局内存不合并 | V1 |
| 数据复用 | 全局内存 → 共享内存 | V2 |
| 数据复用 | 共享内存 → 寄存器 | V3 / V4 |
| 指令效率 | 访存指令占比过高 | V4 / V5 |
| 延迟隐藏 | 加载与计算串行 | V6 |
| 硬件能力 | CUDA Core 吞吐上限 | V7 |

一个有用的直觉：**memory-bound 算子的优化是"消除浪费"（分支分化、Bank 冲突、线程闲置），compute-bound 算子的优化是"建立复用"（分块、驻留、流水）**。前者做减法，后者做加法。GEMM 是后者的教科书案例。

---

## 第 4 章 V0：基准实现——一线程一元素

### 4.1 实现

把 CPU 三重循环的外两层交给线程网格，每个线程执行最内层的 K 次乘加：

```cuda
__global__ void sgemm_v0(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    // 注意这个映射：x 方向对应行 —— 这是一个刻意保留的错误，V1 修正
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

// 启动配置
dim3 block(32, 32);
dim3 grid((M + 31) / 32, (N + 31) / 32);
sgemm_v0<<<grid, block>>>(M, N, K, dA, dB, dC);
```

代码与数学定义一一对应，非常直观。但在 4096³ 的规模上，它的性能通常只有 cuBLAS 的 **1%~2%**。

### 4.2 瓶颈分析：四步分析法的第一次完整应用

下面按 2.6 节的四步分析法，对内层循环中读 A 的语句 `A[row * K + k]` 走一遍完整流程。

**① 拍扁成线性地址。** 代入 `row = blockIdx.x * 32 + threadIdx.x`（取 blockIdx.x = 0 不失一般性）：

```
地址(threadIdx.x, k) = (threadIdx.x) * K + k
```

**② 抓一个代表 Warp。** 由 2.1 节的拉平规则，Block(32,32) 中 `threadIdx.x = 0..31、threadIdx.y = 0` 恰好构成 Warp 0——这 32 个线程 `row` 连续、`col` 相同。

**③ 冻结一个时刻。** 取 k = 0（Warp 锁步执行，32 个线程此刻在同一轮循环）。

**④ 求相邻线程的地址差。** 地址表达式中 threadIdx.x 的系数是 **K**——相邻线程地址相差一整行（K=4096 时相差 16 KB）：

```
线程 t0: A[ 0 * K + 0]     ┐
线程 t1: A[ 1 * K + 0]     │  Δ = K，查判定表：
线程 t2: A[ 2 * K + 0]     │  每线程独占一个内存事务 → 最坏情况
...                        │
线程 t31: A[31 * K + 0]    ┘
```

对 A 的访问**跨度为整行**：一个 Warp 的一次读取分散在 32 个不同的缓存行上，需要 32 个独立的内存事务；每个 32 字节的事务只有 4 字节被用到，带宽利用率 12.5%，这是 2.4 节描述的最坏情况。

对另外两条访问语句重复同样的流程（只需看 threadIdx.x 的系数）：

```
读 B:  B[k * N + col]        ← threadIdx.x 的系数为 0 → 全 Warp 同地址，硬件广播，无问题
写 C:  C[row * N + col]      ← threadIdx.x 的系数为 N → 与 A 同病，32 个分散事务
```

主循环执行 K 轮，每一轮都发生一次对 A 的分散读——低效被放大了 4096 倍。

> 注意：这里的问题不是分支分化，也不是计算写错了，而是"**线程编号与数据布局的映射关系**"选错了。计算逻辑完全不动，只改映射就能有数倍提升——这是 V1。

---

## 第 5 章 V1：合并访存——修正线程到矩阵的映射

### 5.1 优化思路

行主序存储下，矩阵**同一行的元素地址连续**。要让 Warp 内（threadIdx.x 连续）的线程访问连续地址，就应该让 **threadIdx.x 对应列方向（同行不同列）**：

```
V0:  threadIdx.x → 行   （Warp 跨 32 行，地址间距 K）
V1:  threadIdx.x → 列   （Warp 在同一行内横向排开，地址连续）
```

### 5.2 实现

只交换两行代码：

```cuda
__global__ void sgemm_v1(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // x → 列
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // y → 行

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}
```

### 5.3 效果：三类访问全部变优

对交换映射后的代码重复 2.6 节的四步法。现在同一 Warp 内 `col` 连续、`row` 相同，三条访问语句中 threadIdx.x 的系数分别变为：

```
读 A:  A[row * K + k]            ← 系数 0：全 Warp 同地址 → 一次广播
读 B:  B[k * N + (col+0..31)]    ← 系数 1：32 个连续地址 → 合并为 4 个 32B 事务（128B 全部有用）
写 C:  C[row * N + (col+0..31)]  ← 系数 1：连续 → 完全合并
```

| 访问 | V0（系数） | V1（系数） |
|------|----|----|
| A | 32 个分散事务（K） | **1 次广播（0）** |
| B | 广播（0） | **完全合并（1）** |
| C | 32 个分散事务（N） | **完全合并（1）** |

一行代码的交换通常带来 **5~8 倍**提升。这是 2.4 节"相邻线程访问相邻地址"黄金法则最典型的案例——在写任何 CUDA kernel 时，都应先用四步法检查这一条。

### 5.4 遗留问题：零复用

V1 的访存已经"姿势正确"，但**总量**一点没少：每个线程仍然独立读 2K 个数，实际 AI 仍是 0.25 FLOP/Byte（见 2.8 节）。同一 Block 内，行相同的线程重复读同样的 A 行，列相同的线程重复读同样的 B 列——这些重复读取只能指望 L2/L1 缓存兜底，而缓存的命中行为不可控。

下一步：**把复用显式地组织起来**——用共享内存。

---

## 第 6 章 V2：共享内存分块——Block Tiling

从本章起，kernel 开始显式使用共享内存。共享内存的规则（谁拥有、谁可见、活多久）决定了本章方案的每一处设计，也决定了它的能力边界，因此先花一节把这些规则彻底理清，再进入优化本身。

### 6.1 预备理解：共享内存是"绑定在 Block 上的便签纸"

**物理上**，共享内存是每个 SM 芯片内部的一小块 SRAM（约 100 KB），不在显存里——这是它比全局内存快约 20 倍的根本原因。**逻辑上**，CUDA 为它定了三条铁律：

| 规则 | 内容 | 对本章的意义 |
|------|------|------------|
| 谁拥有 | **每个 Block 独占一份**。`__shared__` 声明的是"每个 Block 各自的一份"，不是全局一份 | 代码里只写了一次 `As[TILE][TILE]`，但 Grid 中 16384 个 Block 实际各有一份互相独立的 As/Bs |
| 谁可见 | 只有**本 Block 内的线程**能读写，Block A 永远摸不到 Block B 的共享内存 | 复用只能在 Block 内部组织（跨 Block 的问题见 6.5 节） |
| 活多久 | 生命周期 = **Block 的执行期**。Block 结束，空间立即回收给下一个 Block，内容作废 | 不能指望"上一个 Block 留下的数据"，每个 Block 必须自己搬自己的 |

**一句话定位**：共享内存 = **程序员手动管理的缓存**。L1 缓存与它共用同一块物理 SRAM，区别在于 L1 由硬件自动决定"留谁、踢谁"（不可控），共享内存由程序员显式决定"什么数据、什么时候进来、驻留多久"（完全可控）。当你**确切知道**一批数据会被反复使用时，手动管理稳定可靠，自动缓存则看运气——这正是 GEMM 选择共享内存而不是指望 L1 的原因。

### 6.2 优化思路：把 K 维切成小段，子块驻留共享内存

一个 Block 负责 C 的一个 32×32 子块。要算出这个子块，需要 A 的 32 行 × 全部 K 列、B 的全部 K 行 × 32 列——以 K=4096 计有 2×32×4096×4B = 1 MB，远超共享内存容量，放不进去。解法是**沿 K 维分段**：

```
把 K 切成 K/32 段。每一段：
  ① 全 Block 协作，把 A 的 32×32 子块、B 的 32×32 子块搬进共享内存
  ② 每个线程用共享内存中的数据做 32 次乘加，累加到自己的寄存器
  ③ 同步，进入下一段

        A (M×K)                B (K×N)
   ┌────┬────┬────┐        ┌───────────┐
   │    │    │    │        ├───┼▓▓▓┼───┤  ▓ = 当前段搬入共享内存的子块
   ├────┼────┼────┤        │   │▓▓▓│   │
   │▓▓▓▓│ →  │ →  │        ├───┼─↓─┼───┤
   └────┴────┴────┘        └───────────┘
    A 子块沿行向右滑动        B 子块沿列向下滑动
```

正确性不受影响：点积 `Σₖ A[m][k]·B[k][n]` 只是被按 k 分了组，每组的部分和累加进同一个寄存器。

复用从哪来？每个 A 子块元素搬进共享内存后，被**同一 Block 中与它同行的 32 个线程各用一次**——复用 32 次；B 子块元素被同列的 32 个线程复用，同样 32 次。于是全局内存访问量降为 V1 的 1/32。

**理解这个方案的关键心智转换：As/Bs 是"舞台"，不是"仓库"。** `__shared__ float As[TILE][TILE]` 只在 Block 启动时分配一次，但它的**内容**在主循环中被覆盖 K/32 = 128 次。每一轮的生命周期如下：

```
主循环第 t 轮（t = 0, 32, 64, ...）:

  ┌──────────────────────────────────────────────────┐
  │ ① 搬入：32×32=1024 个线程，每人从 global 搬 1 个    │
  │    A 元素 + 1 个 B 元素，写进 As/Bs（覆盖上一轮）     │
  │ ② __syncthreads()  ← 等 1024 人全部搬完            │
  │ ③ 计算：每人做 32 次乘加，只读 As/Bs，不碰 global    │
  │    部分和累加进自己私有的寄存器 acc                   │
  │ ④ __syncthreads()  ← 等 1024 人全部用完，才许覆盖    │
  └──────────────────────────────────────────────────┘
                    ↓ 重复 K/32 = 128 轮
  acc 里攒齐完整点积，写回 C
```

注意其中的角色分工：**中间结果（acc）从头到尾住在每个线程私有的寄存器里，共享内存只放"原材料"**。As/Bs 像流水线工位上的料盒，一批料加工完就换下一批；acc 才是每个工人手里越攒越多的半成品。"每轮都要重新搬"不是没优化掉的浪费，而是滑动窗口的正常工作方式——它省下了什么，6.4 节用账本算清。

### 6.3 实现

```cuda
#define TILE 32

__global__ void sgemm_v2(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    __shared__ float As[TILE][TILE];    // 2 × 32×32×4B = 8 KB 共享内存
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;                    // 列方向（保持 V1 的合并映射）
    int ty = threadIdx.y;                    // 行方向
    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;

    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        // ① 协作加载：每线程各搬 A、B 的一个元素（越界补 0）
        As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx]   : 0.0f;
        Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
        __syncthreads();                     // 等全部数据就位

        // ② 用共享内存中的子块做 TILE 次乘加
        for (int k = 0; k < TILE; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();                     // 防止下一轮加载覆盖还在使用的数据

        // 循环推进：A 子块右移、B 子块下移（由 t 体现）
    }

    if (row < M && col < N) C[row * N + col] = acc;
}
```

几个细节：

- **两次 `__syncthreads()` 缺一不可**：第一次保证"算之前数据已就位"（读后写依赖），第二次保证"覆盖之前大家都用完了"（写后读依赖）。可以用一个口诀记忆：**第一次是"没到齐不许吃"，第二次是"没吃完不许撤"**。漏掉任何一次都会产生随机错误结果，且往往小规模测试碰巧全对、大规模才崩——这是 CUDA 新手最经典的 bug 之一；
- 加载阶段 `As[ty][tx] = A[row*K + t + tx]`：tx 连续 → 全局地址连续，V1 建立的合并访存在加载路径上依然成立（Bs 同理）；
- 计算阶段的共享内存访问按 2.6 节四步法的 Bank 变体检查：`As[ty][k]` 中 threadIdx.x（即 tx）的系数为 0——Warp 内广播；`Bs[k][tx]` 的系数为 1——32 个连续地址恰好铺满 32 个不同 Bank——**无 Bank Conflict**。

### 6.4 效果与代价的量化

**先算宏观账**。全局内存访问量：每个 Block 每段搬入 2×32×32 个 float，共 K/32 段，Block 总数 (M/32)(N/32)：

```
总访存 = (M/32)(N/32) · (K/32) · 2·32·32 · 4B = MNK/4 字节
实际 AI = 2MNK / (MNK/4) = 8 FLOP/Byte      （V1 的 32 倍）
```

**再从单个数据的视角看"每轮重搬到底省在哪"**，对比每个 float 的"搬运次数 : 使用次数"：

```
V1（无共享内存）:
  同一 Block 内 col=0..31 的 32 个线程，各自去 global
  读了一遍同一个 A[row][k]
  → 1 个数据：32 次 global 读取，各用 1 次

V2（共享内存）:
  A[row][k] 从 global 只搬 1 次进 As，
  然后同行的 32 个线程从 As 各读 1 次
  → 1 个数据：1 次 global 读取 + 32 次 smem 读取
    （smem 快约 20 倍，且不占显存带宽）
```

准确地说：**没有变少的是 片上→寄存器 的读取次数，变少的是 global→片上 的搬运次数**（降为 1/32）。而 global 恰恰是最慢、最容易成为瓶颈的一层——每轮"重新搬"的成本，被"搬进来的每个数都被用 32 次"稳稳赚回。

一般化结论：**BM×BN 的 Block Tile 把全局访存降为原来的 (1/BM + 1/BN)/2 倍量级**——tile 越大，复用越充分；上限受共享内存容量与线程数约束。

### 6.5 一个自然的疑问：不同 Block 之间能不能复用 As/Bs？

看 C 的 Block 布局，会发现跨 Block 的数据重叠是真实存在的：

```
            B 列 0..31    B 列 32..63
           ┌───────────┬───────────┐
A 行 0..31 │ Block(0,0) │ Block(1,0)│ ← 同一行的 N/32=128 个 Block
           ├───────────┼───────────┤    需要的 A 数据完全相同
A 行32..63 │ Block(0,1) │ Block(1,1)│
           └───────────┴───────────┘
                 ↑
        同一列的 128 个 Block 需要的 B 数据完全相同
```

理论上 A 的每一行只该从显存读 1 次，而 V2 中它被同行的 128 个 Block 各读了 1 次。既然重叠这么多，能不能让 Block 之间共享 As/Bs？**不能——共享内存被硬件与编程模型双重限定为 Block 私有**，原因有三：

1. **物理上没有通路**：Block(0,0) 可能跑在 SM 3 上，Block(1,0) 可能跑在 SM 57 上，而共享内存是各 SM 内部的 SRAM，芯片上不存在让 SM 57 读 SM 3 便签纸的连线；
2. **时间上未必共存**：Block 总数（16384）远多于 SM 数量，Block 是分批上机的。Block(1,0) 被调度时，Block(0,0) 可能早已执行完毕、共享内存被回收并让给了别的 Block——想共享，连"对方还活着"都无法保证；
3. **这是可扩展性的刻意设计**："Block 相互独立、无执行顺序保证"是 CUDA 编程模型的根基。正因如此，同一份代码不加修改就能跑在 10 个 SM 的笔记本 GPU 和 132 个 SM 的 H100 上——硬件只管把 Block 随意扔向空闲 SM。允许 Block 间共享 smem，这种自由调度就崩塌了。

那这些跨 Block 的重复读取就白白浪费了吗？不完全是——它们由另外两层机制兜底：

- **L2 缓存（自动，全 GPU 共享）**：所有 SM 的全局内存访问都经过 L2（几十 MB）。Block(0,0) 读过 A 的第 0 行后，数据会留在 L2 中；稍后 Block(1,0) 再读时很可能直接命中 L2，不必真的到显存。这正是 2.8 节复用层次表中"全局内存 → L2：自动，不可控"一行的含义——有效，但命中与否取决于 Block 调度时机与 L2 置换策略，**程序员不能依赖它**；
- **进阶手段**（了解即可）：
  - **Block Swizzle**：重排 blockIdx 到 C 子块的映射，让"同时在跑"的 Block 集中在 C 的一个局部区域（它们所需的 A/B 行列高度重叠），人为提高 L2 命中率——cuBLAS / CUTLASS 都做了这件事；
  - **Thread Block Cluster（Hopper，sm_90+）**：新硬件真正开了口子——同一 Cluster 内的 Block 保证同时调度到相邻 SM，可通过 **分布式共享内存（Distributed Shared Memory）** 直接读写彼此的 smem。这正是硬件对"跨 Block 复用"诉求的回应，但它是带严格约束的新特性，不是通用机制。

一句话总结：**Block 内复用靠共享内存（显式、可控），Block 间复用靠 L2（隐式、尽力而为）**。6.4 节算出的 1/32 缩减，指的仅是前者。

### 6.6 共享内存使用范式小结

V2 是共享内存的第一次出场，也是它的标准用法模板。此后所有版本（乃至绝大多数用到 smem 的 kernel）都遵循同一范式：

```
声明（每 Block 一份） → 循环 { 协作装载 → 同步 → 集体使用 → 同步 } → 结果写回
```

其中"协作装载"的精髓在于：**装载的分工与使用的分工可以完全不同**。V2 中线程 (tx, ty) 搬运的是 `As[ty][tx]` 这一个固定位置，计算时读的却是 `As[ty][0..31]` 一整行——"我搬的不只是我用的，我用的不只是我搬的"。正因为中间隔着同步栅栏，这种交叉才是安全的，而这正是共享内存"共享"二字的价值所在（V3 起两种分工将进一步彻底解耦，见 7.2 节）。

判断"该不该用共享内存"的三条准则，对任何 kernel 都适用：

| 问题 | V2 的答案 |
|------|----------|
| 这块数据会被同一 Block 的多个线程用到吗？ | 是——A 的一行被 32 个线程用 |
| 用的次数够多，值回一次搬运 + 两次同步的成本吗？ | 是——每个数用 32 次 |
| 容量放得下吗？放不下怎么切？ | 放不下整条 K，沿 K 切成 32 一段滑动 |

### 6.7 遗留问题：共享内存成为新瓶颈

用 2.10 节的分析工具检查内层循环：

```cuda
acc += As[ty][k] * Bs[k][tx];   // 1 次 FMA = 2 次 LDS（共享内存加载）+ 1 次 FFMA
```

**每次 FMA 配 2 次共享内存读，LDS/FMA = 2**。LDS 指令与 FFMA 指令共享发射端口（2.9.3 节），指令流中 2/3 是访存指令，算力上限被压到约 1/3。此外每个线程只积累 1 个 acc，相邻两条 FFMA 之间存在写后读依赖（都要先读上一条算出的 acc），流水线上无法连续发射，延迟也掩盖不住。

另外，6.2 节的时间线里还藏着一处浪费：每一轮中"搬入"与"计算"是**串行**的——搬的时候算力闲着，算的时候搬运通道闲着。这个问题本章先按下不表，留给第 10 章的双缓冲解决。

数据复用的问题在共享内存这一层重演了：**共享内存里的数据没有被寄存器复用**。解法与上一层完全同构——再分一次块，这次分到寄存器。

---

## 第 7 章 V3：寄存器分块入门——一维 Thread Tiling

### 7.1 优化思路：一个线程算一小列（8 个输出）

让每个线程负责 C 子块中**同一列的 TM=8 个元素**。关键收益在内层循环的访存结构：

```
对固定的 k：
    b = Bs[k][tx]           ← 读 1 次 Bs，存入寄存器
    acc[0] += As[row0][k] * b
    acc[1] += As[row1][k] * b     ← b 在寄存器中被复用 8 次！
    ...
    acc[7] += As[row7][k] * b
```

8 次 FMA 消耗 8 次 As 读 + **1 次** Bs 读 = 9 次 LDS，LDS/FMA 从 2 降到约 1.1；且 8 个 acc 是互不依赖的累加链，FFMA 可以流水发射（2.7 节所说的 ILP 开始起作用）。

### 7.2 分块参数与实现

引入一组贯穿后文的分块记号：

| 记号 | 含义 | V3 取值 |
|------|------|---------|
| BM × BN | 一个 Block 负责的 C 子块 | 64 × 64 |
| BK | K 维每段长度 | 8 |
| TM | 每线程负责的输出行数 | 8 |

线程数 = (BM×BN)/TM = 512。共享内存 = (64×8 + 8×64)×4B = 4 KB。

```cuda
template <int BM, int BN, int BK, int TM>
__global__ void sgemm_v3(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    __shared__ float As[BM * BK];      // 64×8
    __shared__ float Bs[BK * BN];      // 8×64

    // 把指针推进到本 Block 负责的子块起点，后续用局部坐标
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // 计算映射：512 线程排成 8 行 × 64 列，每线程纵向负责 TM=8 个输出
    const int threadCol = threadIdx.x % BN;        // 0..63
    const int threadRow = threadIdx.x / BN;        // 0..7

    // 加载映射：按各自矩阵的形状重新分工（与计算映射无关！）
    const int innerColA = threadIdx.x % BK;        // 0..7
    const int innerRowA = threadIdx.x / BK;        // 0..63
    const int innerColB = threadIdx.x % BN;        // 0..63
    const int innerRowB = threadIdx.x / BN;        // 0..7

    float acc[TM] = {0.0f};

    for (int t = 0; t < K; t += BK) {
        // ① 协作加载 64×8 的 As 与 8×64 的 Bs（每线程各 1 个元素）
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();
        A += BK;                                   // A 子块右移
        B += BK * N;                               // B 子块下移

        // ② 内层：对每个 k，Bs 读 1 次进寄存器，复用 TM 次
        for (int k = 0; k < BK; k++) {
            float b = Bs[k * BN + threadCol];
            for (int i = 0; i < TM; i++) {
                acc[i] += As[(threadRow * TM + i) * BK + k] * b;
            }
        }
        __syncthreads();
    }

    // ③ 写回 TM 个结果
    for (int i = 0; i < TM; i++) {
        C[(threadRow * TM + i) * N + threadCol] = acc[i];
    }
}
// 启动：dim3 grid(N/64, M/64);  sgemm_v3<64,64,8,8><<<grid, 512>>>(...)
```

两个容易被忽略的设计点：

1. **加载分工与计算分工解耦**。加载时 512 个线程按 `(64行 × 8列)` 铺满 As、按 `(8行 × 64列)` 铺满 Bs，都保证 threadIdx.x 连续 → 全局地址连续（合并访存）；计算时又按另一种形状（8 行 × 64 列，每线程管一小列）分工。中间隔着共享内存与 `__syncthreads()`，两种映射互不干扰——**这是 GEMM kernel 的通用手法**，此后每个版本都在用；
2. **BK 缩小到 8**。共享内存复用率由 BM/BN 决定，与 BK 无关；BK 越小，共享内存占用越少（有利于 SM 上驻留更多 Block，见 2.7 节），代价是主循环轮数变多、同步更频繁。BK=8 是经验平衡点。

### 7.3 效果与遗留问题

| 每 8 次 FMA | V2 | V3 |
|---|----|----|
| As 读取 | 8 | 8 |
| Bs 读取 | 8 | **1** |
| 独立累加链 | 1 条 | **8 条** |

性能通常再翻 2~3 倍。但 As 的读取还是"每 FMA 一次"——复用只建立在了 Bs 一侧。对称地想：如果线程同时在**行、列两个方向**各负责多个输出，As 和 Bs 就都能复用——这就是二维 Thread Tiling。

---

## 第 8 章 V4：二维 Thread Tiling——外积累加

### 8.1 优化思路：从"点积"视角切换到"外积"视角

每个线程负责 C 的一个 **TM×TN = 8×8 小方块**。对固定的 k，先把 As 的 8 个数、Bs 的 8 个数读进寄存器，然后做一次 **8×8 外积**累加：

```
        regB[0..7]  (来自 Bs 的第 k 行)
         b0  b1 ... b7
regA a0 │ ×   ×  ...  ×     acc[i][j] += regA[i] * regB[j]
(As  a1 │ ×   ×  ...  ×
 第 k…  │
 列) a7 │ ×   ×  ...  ×     ← 16 次 LDS 支撑 64 次 FMA
```

**LDS/FMA 比值 = (TM+TN)/(TM×TN) = 16/64 = 0.25**——终于反转过来，指令流以 FFMA 为主。这就是为什么 Thread Tiling 必须做成二维：一维时比值是 (TM+1)/TM ≈ 1，无论怎么加大 TM 都不可能低于 1；二维时分母是乘积、分子是和，比值随 tile 变大迅速下降。

### 8.2 分块参数与实现

| 记号 | V4 取值 | 说明 |
|------|---------|------|
| BM × BN | 128 × 128 | Block 子块 |
| BK | 8 | |
| TM × TN | 8 × 8 | 每线程输出小方块 |
| 线程数 | (128×128)/(8×8) = 256 | 逻辑上排成 16×16 的线程网格 |

加载侧的变化：每段需要搬 128×8（As）+ 8×128（Bs）= 2048 个 float，256 个线程**每人搬 4+4 个**，用跨步循环完成。

```cuda
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v4(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    __shared__ float As[BM * BK];               // 128×8
    __shared__ float Bs[BK * BN];               // 8×128

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const int threadCol = threadIdx.x % (BN / TN);   // 0..15
    const int threadRow = threadIdx.x / (BN / TN);   // 0..15

    // 加载映射（256 线程 → 128×8 的 As：每次覆盖 32 行，循环 4 次）
    const int innerRowA = threadIdx.x / BK;     // 0..31
    const int innerColA = threadIdx.x % BK;     // 0..7
    const int strideA   = blockDim.x / BK;      // 32
    const int innerRowB = threadIdx.x / BN;     // 0..1
    const int innerColB = threadIdx.x % BN;     // 0..127
    const int strideB   = blockDim.x / BN;      // 2

    float acc[TM][TN] = {{0.0f}};
    float regA[TM], regB[TN];

    for (int t = 0; t < K; t += BK) {
        // ① 协作加载（每线程多个元素，跨步循环）
        for (int off = 0; off < BM; off += strideA)
            As[(innerRowA + off) * BK + innerColA] = A[(innerRowA + off) * K + innerColA];
        for (int off = 0; off < BK; off += strideB)
            Bs[(innerRowB + off) * BN + innerColB] = B[(innerRowB + off) * N + innerColB];
        __syncthreads();
        A += BK;
        B += BK * N;

        // ② 外积累加
        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i++)         // As 第 k 列的 8 个数 → 寄存器
                regA[i] = As[(threadRow * TM + i) * BK + k];
            for (int j = 0; j < TN; j++)         // Bs 第 k 行的 8 个数 → 寄存器
                regB[j] = Bs[k * BN + threadCol * TN + j];
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i] * regB[j];   // 64 次独立 FMA
        }
        __syncthreads();
    }

    // ③ 写回 8×8 小方块
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            C[(threadRow * TM + i) * N + threadCol * TN + j] = acc[i][j];
}
// 启动：dim3 grid(N/128, M/128);  sgemm_v4<128,128,8,8,8><<<grid, 256>>>(...)
```

### 8.3 寄存器压力：一个必须正视的代价

数一数每线程的寄存器：acc 64 个 + regA/regB 16 个 + 地址计算若干 ≈ **100+ 个**。按 2.7 节的资源账：SM 的寄存器文件共 65536 个，每 SM 能驻留的线程数被压到约 512~768（占用率 25%~37%）。

这是 GEMM 优化中的经典权衡：**降低占用率，换取每线程更高的数据复用**。GEMM 的延迟主要靠 64 条独立 FMA 链的指令级并行（ILP）掩盖，而不是靠线程级并行（TLP），所以低占用率是可接受的——这一点与 memory-bound 算子（需要高占用率轮转掩盖访存延迟）恰好相反（2.7 节已预告了这一对比）。

### 8.4 三级分块的全景图

至此，经典 SGEMM 的三级分块结构已经完整：

```
Grid 级:   C 被切成 (M/128)×(N/128) 个 Block Tile      ← 复用发生在 L2
Block 级:  128×128 tile 沿 K 每次前进 8，子块驻留共享内存 ← 复用发生在 smem
Thread 级: 每线程 8×8 小方块，操作数驻留寄存器            ← 复用发生在寄存器
每级访存缩减:  global→smem 降 64 倍;  smem→reg 降 4 倍
```

```
C 全矩阵 ──切──► Block Tile (128×128) ──切──► Thread Tile (8×8)
              沿 K 分段 (BK=8)              固定 k 做外积
              数据源: 共享内存               数据源: 寄存器
```

### 8.5 遗留问题

用 profiler（如 Nsight Compute）观察 V4，会发现两个访存层面的低效：

1. **标量访存指令**：加载 As/Bs（global→smem）与读取 regA/regB（smem→reg）都是 4 字节一条的指令，指令数偏多；
2. **As 的访问方向别扭**：计算时读 `As[(row+i)*BK + k]`，即按**列**方向取 8 个数，但 As 按行主序存储——这 8 个数在共享内存中相距 BK=8 个 float，既不能向量化，访问模式也不理想。

---

## 第 9 章 V5：float4 向量化与共享内存布局重排

### 9.1 优化 1：global→smem 用 float4 搬运

`float4` 是 CUDA 内置的 16 字节向量类型（4 个 float，成员 `.x .y .z .w`）。把 `float*` 重新解释为 `float4*` 后，一条加载指令（SASS 层面的 `LDG.128`）即可搬运 16 字节——**与 4 字节的标量加载指令数相同，数据吞吐是 4 倍**。前提是地址按 16 字节对齐（`cudaMalloc` 与 PyTorch 张量默认满足）。

As 每线程原来要用 4 条标量指令搬 4 个 float，现在一条指令搬完：

```cuda
float4 tmp = reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA * 4])[0];
```

BK=8 恰好等于 2 个 float4 的宽度，128×8 的 As 由 256 线程一轮搬完（每人一条 float4），Bs 同理。

### 9.2 优化 2：As 转置存储，让"取一列"变成"取一行"

V4 计算时需要 As 的一**列**（固定 k，行方向连续取 TM 个数）。若把 As **转置**存储为 `As[BK][BM]`（k 为行、m 为列），这 TM 个数就变成共享内存中的**连续地址**，可以用 float4 一次取 4 个：

```cuda
// 加载时顺手转置：全局内存读出的 float4 拆成 4 个标量，按转置位置写入
As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;
```

```
V4:  As[m][k]  计算时沿列取数 → 间距 BK，无法向量化
V5:  As[k][m]  计算时沿行取数 → 连续，LDS.128 一条指令取 4 个
```

Bs 本来就是"固定 k 取一行"，天然连续，无需转置。

### 9.3 核心片段

```cuda
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v5(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    __shared__ float As[BK * BM];    // 注意：转置布局 [BK][BM]
    __shared__ float Bs[BK * BN];    // 正常布局 [BK][BN]

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    const int innerRowA = threadIdx.x / (BK / 4);    // 0..127
    const int innerColA = threadIdx.x % (BK / 4);    // 0..1
    const int innerRowB = threadIdx.x / (BN / 4);    // 0..7
    const int innerColB = threadIdx.x % (BN / 4);    // 0..31

    float acc[TM][TN] = {{0.0f}};
    float regA[TM], regB[TN];

    for (int t = 0; t < K; t += BK) {
        // ① float4 加载 + As 转置写入
        float4 ta = reinterpret_cast<const float4*>(
                        &A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = ta.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = ta.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = ta.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = ta.w;

        reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();
        A += BK;
        B += BK * N;

        // ② 外积累加：regA/regB 均可由编译器生成 LDS.128
        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i++)
                regA[i] = As[k * BM + threadRow * TM + i];   // 连续!
            for (int j = 0; j < TN; j++)
                regB[j] = Bs[k * BN + threadCol * TN + j];   // 连续!
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
    }

    // ③ 写回也用 float4（每行 8 个输出 = 2 条 ST.128）
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
            float4 out = {acc[i][j], acc[i][j+1], acc[i][j+2], acc[i][j+3]};
            reinterpret_cast<float4*>(
                &C[(threadRow * TM + i) * N + threadCol * TN + j])[0] = out;
        }
}
```

### 9.4 效果与代价

| 指令类型（每主循环轮） | V4 | V5 |
|---|----|----|
| global load | 8 条 LDG.32 | **3 条 LDG.128** |
| smem load（每 k） | 16 条 LDS.32 | **4 条 LDS.128** |
| smem store（As，每轮） | 4 条 STS.32 | 4 条 STS.32（转置后无法合并成向量，属于代价） |

> 进阶说明：转置写入 As 时，相邻线程写入的地址相距 BM=128 个 float——按 2.6 节 Bank 变体的判定表，128 是 32 的倍数，多个线程命中同一 Bank，存在写入侧 Bank Conflict。生产级 kernel 会用 padding（把 As 声明为 `As[BK][BM+4]`，错开 bank 相位）或 swizzle 布局消除，此处为保持代码可读暂不展开——profiler 中的 `shared_st_bank_conflict` 指标可以观察到它。

### 9.5 遗留问题：加载与计算的串行化

观察主循环的时间线：

```
[加载 tile 0] → sync → [计算 tile 0] → sync → [加载 tile 1] → sync → [计算 tile 1] → ...
     ↑                                          ↑
  计算单元闲着                               访存单元闲着
```

加载与计算像"红绿灯"一样交替，两类硬件资源互相等待。GPU 的访存与计算本可以并行——只要它们操作的不是同一块缓冲区。

---

## 第 10 章 V6：双缓冲——用计算掩盖访存延迟

### 10.1 优化思路：乒乓缓冲

开**两份**共享内存缓冲区，形成流水线：计算第 t 块时，同时预取第 t+1 块到另一份缓冲区：

```
缓冲 0:  [加载 T0]        [计算 T0]         [加载 T2]        [计算 T2]
缓冲 1:            [加载 T1]        [计算 T1]        [加载 T3]  ...
                       ↑ 加载与计算重叠，访存延迟被计算掩盖
```

关键在于把"发出全局内存读请求"与"使用读到的数据"拆开：LDG 指令发出后线程并不阻塞，**只有在使用目标寄存器时才真正等待数据到达**（记分板机制，2.9.4 节）。因此正确的指令编排是：

```
① 发出 tile t+1 的 LDG（读全局内存 → 寄存器 tmp，不等待）
② 用缓冲区 cur 完成 tile t 的全部外积计算（几百条 FFMA，足够掩盖 LDG 延迟）
③ 把 tmp 写入另一份缓冲区 next（STS）
④ __syncthreads()；交换 cur/next，进入下一轮
```

### 10.2 核心片段

在 V5 的基础上改造主循环（省略与 V5 相同的索引计算）：

```cuda
    __shared__ float As[2][BK * BM];     // 双缓冲
    __shared__ float Bs[2][BK * BN];

    float4 ta, tb;                        // 预取暂存寄存器

    // 序幕：加载第 0 块到缓冲 0
    ta = reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA * 4])[0];
    tb = reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];
    As[0][(innerColA * 4 + 0) * BM + innerRowA] = ta.x;
    As[0][(innerColA * 4 + 1) * BM + innerRowA] = ta.y;
    As[0][(innerColA * 4 + 2) * BM + innerRowA] = ta.z;
    As[0][(innerColA * 4 + 3) * BM + innerRowA] = ta.w;
    reinterpret_cast<float4*>(&Bs[0][innerRowB * BN + innerColB * 4])[0] = tb;
    __syncthreads();

    int cur = 0;
    for (int t = 0; t < K; t += BK) {
        int next = cur ^ 1;

        // ① 发出下一块的全局内存读（最后一轮除外）——立即返回，不阻塞
        if (t + BK < K) {
            ta = reinterpret_cast<const float4*>(
                     &A[innerRowA * K + (t + BK) + innerColA * 4])[0];
            tb = reinterpret_cast<const float4*>(
                     &B[(t + BK + innerRowB) * N + innerColB * 4])[0];
        }

        // ② 计算当前块（长串 FFMA 掩盖 ① 的延迟）
        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i++)
                regA[i] = As[cur][k * BM + threadRow * TM + i];
            for (int j = 0; j < TN; j++)
                regB[j] = Bs[cur][k * BN + threadCol * TN + j];
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i] * regB[j];
        }

        // ③ 把预取的数据写入另一份缓冲
        if (t + BK < K) {
            As[next][(innerColA * 4 + 0) * BM + innerRowA] = ta.x;
            As[next][(innerColA * 4 + 1) * BM + innerRowA] = ta.y;
            As[next][(innerColA * 4 + 2) * BM + innerRowA] = ta.z;
            As[next][(innerColA * 4 + 3) * BM + innerRowA] = ta.w;
            reinterpret_cast<float4*>(
                &Bs[next][innerRowB * BN + innerColB * 4])[0] = tb;
            __syncthreads();             // ④ 每轮只需一次同步
        }
        cur = next;
    }
```

### 10.3 收益的三个来源

1. **访存-计算重叠**：LDG 的数百周期延迟被 ② 中约 512 次 FFMA 完全掩盖；
2. **同步次数减半**：V5 每轮 2 次 `__syncthreads()`（等加载完成 + 等使用完成），双缓冲下"正在使用"与"正在写入"是两块内存，写后读依赖消失，每轮只需 1 次；
3. **地址推进更直白**：A/B 改用 t 显式索引，去掉了指针滑动带来的额外状态。

代价是共享内存翻倍（8 KB → 16 KB）与几个额外的暂存寄存器——按 2.7 节的资源账，对占用率的影响通常可以接受。

> 现代架构（Ampere+）提供 `cp.async` 指令，能把 global→smem 的拷贝完全绕过寄存器、异步进行，配合 `cuda::pipeline` 可以实现更深的多级流水；Hopper 更进一步提供 TMA 硬件拷贝引擎。双缓冲是理解这一切的概念原型。

### 10.4 至此的性能位置

V6 在多数架构上可达 cuBLAS SGEMM 的 **80%~90%**。剩余差距来自更精细的技巧：Warp Tiling（在 Block 与 Thread 之间再加一级 warp 级分块，优化寄存器缓存局部性与 Bank 访问模式）、swizzle 布局、K 维 Split-K 并行等——它们收益递减、复杂度陡增，工程上通常直接交给 CUTLASS。

而**数量级**的下一次跃迁来自硬件：Tensor Core。

---

## 第 11 章 V7：Tensor Core——WMMA 半精度矩阵乘

### 11.1 为什么需要专用硬件

V6 之后，指令流已以 FFMA 为主，性能上限就是 CUDA Core 的 fp32 FMA 吞吐。但 GEMM 太重要了，NVIDIA 从 Volta 起加入 **Tensor Core**：一条指令完成一个**小矩阵块的乘加**（如 16×16×16），而不是一个标量乘加：

```
CUDA Core:   d = a * b + c          （标量 FMA，每周期每 core 2 FLOP）
Tensor Core: D = A × B + C          （16×16×16 矩阵 FMA，一条指令 8192 FLOP）
```

同代硬件上，半精度 Tensor Core 吞吐通常是 fp32 CUDA Core 的 **8~16 倍**。深度学习训练/推理中的 GEMM 几乎全部跑在 Tensor Core 上（fp16/bf16/tf32/fp8 输入，fp32 累加）。

### 11.2 编程接口：WMMA

CUDA 通过 `nvcuda::wmma` 命名空间暴露 Tensor Core，以 **Warp 为操作单位**——一个 Warp 的 32 个线程协作持有一个小矩阵块（数据如何分布在 32 个线程的寄存器中由硬件决定，程序员不可见，这个抽象叫 **fragment**）：

| API | 作用 |
|-----|------|
| `wmma::fragment<>` | 声明矩阵片段（matrix_a / matrix_b / accumulator） |
| `wmma::load_matrix_sync` | 全 Warp 协作，从内存加载一个 16×16 块到 fragment |
| `wmma::mma_sync` | 执行 D = A×B + C |
| `wmma::store_matrix_sync` | 把累加器 fragment 写回内存 |
| `wmma::fill_fragment` | 初始化累加器（通常置 0） |

### 11.3 最小可用实现

每个 Warp 负责 C 的一个 16×16 块，结构与 V0 惊人地相似——只是"线程算一个标量"换成了"Warp 算一个 16×16 块"：

```cuda
#include <mma.h>
using namespace nvcuda;

// A: M×K (half, row-major)  B: K×N (half, row-major)  C: M×N (float)
// 要求 M、N、K 均为 16 的倍数
__global__ void hgemm_wmma_v7(int M, int N, int K,
                              const half* A, const half* B, float* C) {
    // 每个 Warp 负责一个 16×16 输出块
    // blockDim = (128, 4)：x 方向 4 个 Warp，y 方向 4 个 Warp，共 16 块/Block
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;  // 块列号
    int warpM = blockIdx.y * blockDim.y + threadIdx.y;         // 块行号

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag;
    wmma::fill_fragment(cFrag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        // 全 Warp 协作加载 A、B 的 16×16 块（第二个参数是行跨度）
        wmma::load_matrix_sync(aFrag, A + warpM * 16 * K + k, K);
        wmma::load_matrix_sync(bFrag, B + k * N + warpN * 16, N);
        // 一条 mma：16×16×16 = 4096 次乘加
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    wmma::store_matrix_sync(C + warpM * 16 * N + warpN * 16, cFrag,
                            N, wmma::mem_row_major);
}
```

注意三个与 CUDA Core 编程的本质差异：

1. **粒度上移**：编程单位从 Thread 变成 Warp，`*_sync` 后缀提醒这些调用必须由整个 Warp 一致执行；
2. **混合精度**：输入 half、累加 float 是标准配置——K 很大时 fp16 累加会损失精度，fp32 累加器是精度保障；
3. **数据布局不透明**：fragment 内部的数据-线程映射由架构决定，不能对 fragment 逐元素索引（除了 `fill_fragment` 这类统一操作）。

### 11.4 优化思路的完全复现

这个最小实现相当于 Tensor Core 世界的 "V0"——每次 `load_matrix_sync` 都直接打到全局内存，数据零复用。**前十章的所有优化在这里原样重演一遍**：

| CUDA Core 版本 | Tensor Core 对应物 |
|---|---|
| V2 共享内存分块 | Block 内多 Warp 共享 As/Bs，`load_matrix_sync` 改从共享内存读 |
| V4 二维 Thread Tiling | Warp Tiling：每 Warp 负责多个 16×16 块（如 64×64） |
| V5 向量化 + 布局重排 | smem swizzle 布局消除 `ldmatrix` 的 Bank Conflict |
| V6 双缓冲 | `cp.async` 多级流水 / Hopper TMA + `wgmma` |

这正是 CUTLASS 的组织方式：它把上述每一层分块抽象成可组合的 C++ 模板组件。理解了 V0~V7，就能读懂 CUTLASS 的架构图。

---

## 第 12 章 工程化：PyTorch 扩展与 cuBLAS 对比

写出高性能 kernel 只是一半，另一半是把它接入实际框架并正确地度量。本章给出一个完整可编译运行的 PyTorch CUDA 扩展，并讨论正确性验证与性能测量中的常见陷阱。

### 12.1 完整的 PyTorch 扩展

一个最小扩展由三个文件组成：CUDA 源文件、`setup.py`、测试脚本。

**gemm_kernel.cu**

```cuda
#include <torch/extension.h>
#include <cuda_runtime.h>

// …… 前文任意版本的 kernel 定义，此处以 sgemm_v5 为例 ……

torch::Tensor my_matmul(torch::Tensor A, torch::Tensor B) {
    // 输入检查：设备、维度、形状匹配
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "expect CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "expect 2-D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "shape mismatch");

    // contiguous()：kernel 假定行主序连续存储，转置视图等必须先物化
    auto Ac = A.contiguous().to(torch::kFloat32);
    auto Bc = B.contiguous().to(torch::kFloat32);
    int M = Ac.size(0), K = Ac.size(1), N = Bc.size(1);
    auto C = torch::empty({M, N}, Ac.options());

    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block((BM * BN) / (TM * TN));      // 256

    sgemm_v5<BM, BN, BK, TM, TN><<<grid, block>>>(
        M, N, K,
        Ac.data_ptr<float>(), Bc.data_ptr<float>(), C.data_ptr<float>());
    return C;
}

// PYBIND11_MODULE：把 C++ 函数导出为 Python 模块中的函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_matmul", &my_matmul, "SGEMM v5");
}
```

**setup.py**

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gemm_kernel',
    ext_modules=[
        CUDAExtension('gemm_kernel', ['gemm_kernel.cu'])
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

**编译与安装**

```bash
pip install -e .        # 调用 nvcc 编译并安装为 Python 可导入的模块
```

> 关于算子注册方式的说明：`PYBIND11_MODULE` 是最简单的导出方式，直接把 C++ 函数变成 Python 函数，适合学习与原型验证；它不经过 PyTorch dispatcher，不支持按 device 自动分发，也不兼容 `torch.compile`。生产级算子应使用 `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` 宏注册进 dispatcher（声明 schema 后按 CPU/CUDA/Autograd 等 dispatch key 分别提供实现，调用方式为 `torch.ops.xxx.yyy`）；若要支持多种 dtype（fp32/fp16/bf16），惯用做法是把 kernel 写成模板，在实现函数内部用 `AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, ...)` 宏按 `scalar_type()` 实例化。本文聚焦 kernel 本身，扩展机制点到为止。

### 12.2 正确性与性能验证

**test.py**

```python
import torch, gemm_kernel

M = N = K = 4096
A = torch.randn(M, K, device='cuda')
B = torch.randn(K, N, device='cuda')

# 正确性：与 cuBLAS 对比（浮点求和顺序不同，需容忍相对误差）
C1 = gemm_kernel.my_matmul(A, B)
C2 = A @ B
print(torch.allclose(C1, C2, rtol=1e-3, atol=1e-3))

# 性能：CUDA event 计时，注意预热与同步
def bench(fn, iters=20):
    for _ in range(3): fn()                 # 预热（含 JIT/缓存效应）
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters        # ms

t = bench(lambda: gemm_kernel.my_matmul(A, B))
tflops = 2 * M * N * K / (t * 1e-3) / 1e12
print(f"{t:.3f} ms, {tflops:.1f} TFLOPS   (cuBLAS: {bench(lambda: A @ B):.3f} ms)")
```

四个常见的测量陷阱：

- **验证矩阵不要用全 1 / 全同值**——很多索引错误（如行列写反、跨度写错）在对称输入下算出的结果恰好正确，务必用随机数据；
- **误差判断用相对容差**——K=4096 时每个输出累加 4096 项，fp32 下与 cuBLAS 逐位一致是不可能的（求和顺序不同），`rtol=1e-3` 量级是合理预期；
- **必须预热**——首次调用包含 kernel 加载、cuBLAS 句柄初始化等一次性开销；
- **必须同步**——kernel 启动是异步的，不 `synchronize()` 或用 CUDA event，计到的只是"提交时间"。

### 12.3 什么时候手写 GEMM

| 场景 | 建议 |
|------|------|
| 通用矩阵乘 | 直接用 cuBLAS / `torch.matmul`，不要手写 |
| 特殊形状（超小 M、极度瘦长） | cuBLAS 可能不优，可试 CUTLASS 调参或手写 |
| 融合需求（GEMM + bias + activation + …） | CUTLASS Epilogue，或手写融合 kernel |
| 特殊数据类型 / 稀疏 / 量化 | CUTLASS / 手写 |
| 学习原理 | 手写 V0~V7（本文） |

手写 GEMM 的最大价值不是替代库，而是**具备读懂和修改 CUTLASS / FlashAttention 这类代码的能力**——它们的内核结构正是本文三级分块 + 双缓冲 + Tensor Core 的组合。

---

## 第 13 章 总结与实践建议

### 13.1 八个版本回顾

| 版本 | 核心手段 | 解决的瓶颈 | 复用建立在哪一层 |
|------|---------|-----------|----------------|
| V0 | 一线程一元素 | —（基准） | 无 |
| V1 | 交换线程行列映射 | 全局内存不合并 | 无（只是访问变合并） |
| V2 | Block Tiling + 共享内存 | 全局内存零复用 | global → smem |
| V3 | 一维 Thread Tiling | LDS 指令占比过高 | smem → reg（单侧） |
| V4 | 二维 Thread Tiling / 外积 | LDS/FMA 仍 ≥ 1 | smem → reg（双侧） |
| V5 | float4 + As 转置 | 标量访存指令过多 | （提高各层搬运效率） |
| V6 | 双缓冲 | 加载与计算串行 | （时间维度的重叠） |
| V7 | Tensor Core (WMMA) | CUDA Core 吞吐上限 | 硬件级矩阵运算 |

### 13.2 关键指标演进（M=N=K=4096 量级的典型相对性能）

| 指标 | V0 | V1 | V2 | V3 | V4 | V5 | V6 | V7(fp16) |
|------|----|----|----|----|----|----|----|----|
| 全局访存合并 | 否 | 是 | 是 | 是 | 是 | 是 | 是 | 是 |
| 实际 AI (FLOP/B) | 0.25 | 0.25 | 8 | ~32 | ~64 | ~64 | ~64 | ~64 |
| LDS / FMA | — | — | 2 | ~1.1 | 0.25 | 0.25(向量) | 0.25 | ldmatrix |
| 每线程输出数 | 1 | 1 | 1 | 8 | 64 | 64 | 64 | Warp 级 |
| 相对 cuBLAS(fp32) | ~1% | ~8% | ~15% | ~35% | ~55% | ~70% | ~85% | >100%* |

> \* V7 与 fp32 cuBLAS 比较是跨精度的，仅示意 Tensor Core 的吞吐量级；与 fp16 cuBLAS 相比，朴素 WMMA 实现仍需第 11.4 节的全套优化才能接近。具体数字随架构（占用率、频率、缓存）浮动，表中比值取多个公开复现实验的典型量级，请以自己机器上的实测为准。

### 13.3 通用优化方法论

GEMM 的优化过程给出三条对一切 compute-bound 算子适用的一般规律：

1. **先把访存"姿势"做对，再谈复用**：合并访存（V1）是所有后续优化的地基，映射错误会淹没一切其他努力；
2. **核心问题是"每次计算配几次访存"**：在每一级存储上建立分块，把这个比值逐级压低（V2 压 global，V4 压 smem，V5 压指令数）；分块参数的本质是**用片上资源（smem、寄存器）换访存量**，代价是占用率——ILP 充足时低占用率完全可行；
3. **让不同硬件单元的工作在时间上重叠**：双缓冲/流水线（V6）不减少任何工作量，却能显著缩短总时间——这一思想向上延伸就是 `cp.async`、TMA、以及 kernel 间的 stream 并行。

### 13.4 版本选择与进阶路径

| 场景 | 建议 |
|------|------|
| 理解 GPU 存储层次 | 精读 V0 → V2 |
| 理解现代 GEMM kernel 结构 | 精读 V4 → V6（三级分块 + 双缓冲） |
| 进阶：Warp Tiling / swizzle / Split-K | siboehm 博客、CUTLASS 文档 |
| 进阶：Tensor Core 深入 | `mma.sync` PTX、`ldmatrix`、CUTLASS CuTe |
| 生产环境 | cuBLAS / cuBLASLt / CUTLASS，融合场景用 Triton 或手写 |

---

## 附录：关键概念速查

| 概念 | 含义 | 相关章节 |
|------|------|---------|
| GEMM | 通用矩阵乘 C = αAB + βC，计算量 2MNK | 第 1 章 |
| Warp | 32 个连续线程的硬件调度单元，锁步执行（SIMT） | 第 2 章 |
| 合并访存 | Warp 内线程访问连续地址，合并为最少内存事务 | 第 2、4~5 章 |
| Bank Conflict | Warp 内多线程访问共享内存同一 Bank 的不同地址被串行化 | 第 2、9 章 |
| 四步分析法 | 拍扁地址 → 抓代表 Warp → 冻结时刻 → 看 threadIdx.x 系数，判定访存模式 | 第 2、4~5 章 |
| 占用率（Occupancy） | SM 上驻留 Warp 数与上限之比，受寄存器/共享内存约束 | 第 2、8 章 |
| TLP / ILP | 线程级 / 指令级并行，两条掩盖延迟的途径 | 第 2、7~8 章 |
| 算术强度（AI） | FLOP 数 / 访存字节数，决定算子是 compute- 还是 memory-bound | 第 2 章 |
| Roofline 模型 | 以 AI 为横轴、可达性能为纵轴的性能上限模型 | 第 2 章 |
| compute-bound | 性能受限于算力而非带宽；GEMM 的理论属性 | 第 2 章 |
| Block Tiling | C 按 Block 分块、K 维分段，子块驻留共享内存复用 | 第 6 章 |
| 共享内存三铁律 | 每 Block 独占一份；仅 Block 内可见；生命周期随 Block 结束 | 第 6 章 |
| Block Swizzle | 重排 blockIdx 映射使同时运行的 Block 数据重叠，提高 L2 命中率 | 第 6 章 |
| Thread Block Cluster | Hopper 特性：Cluster 内 Block 同时调度、可互访共享内存（DSM） | 第 6 章 |
| Thread Tiling | 每线程负责多个输出元素，操作数驻留寄存器复用 | 第 7~8 章 |
| 外积累加 | 固定 k，regA×regB 的所有组合一次算完；TM×TN 次 FMA 只需 TM+TN 次 LDS | 第 8 章 |
| LDS / FMA 比 | 每次乘加所需的共享内存加载次数，衡量指令流效率 | 第 2、6~8 章 |
| FFMA | fp32 融合乘加 SASS 指令，1 条 = 2 FLOP；峰值算力按"每 core 每周期一条 FFMA"标定 | 第 2 章 |
| LDG / LDS / STS | 全局加载 / 共享内存加载 / 共享内存存储指令；后缀 .32/.128 为一条指令的位宽 | 第 2、9 章 |
| 发射（issue） | 调度器每周期派发一条指令的名额；访存与计算指令竞争同一发射口 | 第 2 章 |
| 记分板（scoreboard） | 访存指令发射后不阻塞、使用结果寄存器时才等待的硬件机制；双缓冲的基础 | 第 2、10 章 |
| float4 / LDG.128 | 16 字节向量访存，同等指令数 4 倍吞吐 | 第 9 章 |
| smem 转置存储 | 让计算期的访问方向与存储方向一致，使 LDS 可向量化 | 第 9 章 |
| 双缓冲 | 两份缓冲乒乓切换，加载与计算重叠 | 第 10 章 |
| cp.async / TMA | Ampere/Hopper 的异步拷贝机制，双缓冲的硬件化 | 第 10~11 章 |
| Tensor Core | 以小矩阵块为单位的乘加硬件，半精度吞吐 8~16 倍 | 第 11 章 |
| WMMA / fragment | Warp 级矩阵乘 API；数据在 Warp 内的分布对程序员不透明 | 第 11 章 |
| CUTLASS | NVIDIA 开源 GEMM 模板库，本文各级分块的组件化实现 | 第 11~12 章 |
| Split-K | K 维切给多个 Block 并行、结果归约；小 M/N 大 K 时提高并行度 | 第 13 章 |
| PYBIND11_MODULE | 把 C++/CUDA 函数导出为 Python 模块的最简方式 | 第 12 章 |
| TORCH_LIBRARY | PyTorch 生产级算子注册宏，接入 dispatcher 按 device 分发 | 第 12 章 |
| AT_DISPATCH_* | 实现函数内部按 dtype 实例化模板 kernel 的宏 | 第 12 章 |
