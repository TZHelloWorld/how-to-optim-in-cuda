# Roofline 模型：从零开始理解性能分析

> 本文面向性能优化新人，结合 UC Berkeley 原始论文、劳伦斯伯克利国家实验室 (LBL/NERSC) 与 NVIDIA Nsight Compute 官方文档，力求用直白的语言把 Roofline 模型讲清楚。

---

## 目录

- [一、为什么需要 Roofline 模型](#一为什么需要-roofline-模型)
- [二、三个核心概念](#二三个核心概念)
- [三、Roofline 公式与那张图](#三roofline-公式与那张图)
- [四、脊点：Memory-bound 还是 Compute-bound](#四脊点memory-bound-还是-compute-bound)
- [五、一个手算例子](#五一个手算例子)
- [六、怎么读懂 Roofline 图 —— 优化决策](#六怎么读懂-roofline-图--优化决策)
- [七、进阶：天花板 (Ceilings)](#七进阶天花板-ceilings)
- [八、进阶：分层 Roofline (Hierarchical Roofline)](#八进阶分层-roofline-hierarchical-roofline)
- [九、在 GPU 上实测 Roofline](#九在-gpu-上实测-roofline)
- [十、常见误区与速查表](#十常见误区与速查表)
- [参考资料](#参考资料)

---

## 一、为什么需要 Roofline 模型

假设你写了一个 CUDA kernel，跑出来是 2 TFLOP/s。这个数字是好是坏？

- 如果 GPU 峰值算力是 15 TFLOP/s，你可能觉得「只用了 13%，太差了，得优化」。
- 但如果这个 kernel 本质上是**搬运数据**为主（比如向量加法），它可能**根本不可能**接近 15 TFLOP/s —— 因为它会先把内存带宽用满。此时 2 TFLOP/s 可能已经接近它的理论上限了，继续「优化计算」是白费力气。

**Roofline 模型就是用来回答这个问题的**：

> 我的程序当前性能，距离这台机器**在这种负载下**能给的上限，还有多远？我到底该优化「计算」还是「访存」？

它把三件事画在**同一张图**里：
1. 机器的**峰值算力**（能算多快）
2. 机器的**峰值内存带宽**（能供数据多快）
3. 你的程序的**算术强度**（每搬 1 字节数据，要做多少次计算）

一图看穿瓶颈，这就是 Roofline 的价值。

---

## 二、三个核心概念

Roofline 建立在三个基础量之上（Berkeley 原始论文定义）：

### 1. Work（工作量，W）
kernel 执行的**操作总数**，绝大多数场景用**浮点操作数 FLOPs** 来衡量。

> ⚠️ 注意区分：
> - **FLOPs**（小写 s）= 浮点操作的**数量**（count），是一个计数，比如「这个 kernel 一共做了 10 亿次浮点运算」。
> - **FLOP/s**（有斜杠）= 每秒浮点操作数，是**速率/吞吐**，比如「这个 GPU 每秒能做 15 万亿次浮点运算」。
> 二者极易混淆，务必分清。

W 主要由算法本身决定，基本不随平台变化。

### 2. Memory Traffic（内存流量，Q）
kernel 运行期间在内存之间搬运的**总字节数**（读 + 写），单位 Bytes。

与 W 不同，**Q 高度依赖平台**（尤其是缓存层次）：同一个算法，缓存命中率不同，实际从 DRAM 搬的字节数就不同。

### 3. Arithmetic Intensity（算术强度 / 操作强度，I）
这是 Roofline 的灵魂，定义为：

```
算术强度 I = Work / Memory Traffic = FLOPs / Bytes
```

**物理意义**：每从内存搬运 1 个字节，能「顺便」做多少次浮点运算。单位是 **FLOPs/Byte**。

- **I 小**（比如 0.1）：搬得多、算得少 → 程序是「数据搬运工」，容易受**内存带宽**限制。
- **I 大**（比如 50）：搬一点数据能算很久 → 程序是「计算密集型」，容易受**算力**限制。

几个直观的例子：

| 操作 | 大致算术强度 | 类型 |
|------|-------------|------|
| 向量加法 `c[i] = a[i] + b[i]` | 极低 (~0.08) | 访存受限 |
| SAXPY `y[i] = a*x[i] + y[i]` | 低 (~0.17) | 访存受限 |
| 稠密矩阵乘法 GEMM | 高（随矩阵增大而升高） | 计算受限 |
| FFT | 中等 | 视规模而定 |

---

## 三、Roofline 公式与那张图

有了三个概念，核心公式只有一行（朴素 Roofline / Naive Roofline）：

```
可达性能 P = min( π , β × I )
```

其中：
- **P** = Attainable Performance，可达到的性能（GFLOP/s）
- **π** = Peak Performance，机器峰值算力（GFLOP/s），一般通过 benchmark 测得
- **β** = Peak Bandwidth，机器峰值内存带宽（GB/s），从手册或工具测得
- **I** = 算术强度（FLOPs/Byte）

这个 `min` 表达了一个朴素但深刻的事实：**你的性能，要么被算力卡住，要么被带宽卡住，取谁小算谁**。

把它画出来（两个轴都用**对数刻度**）：

```
  性能 (GFLOP/s)  [对数轴]
   ▲
 π ┤ - - - - - - - - ┌──────────────────────  ← 峰值算力屋顶 (水平线): P = π
   │              ╱   │
   │           ╱      │        Compute-bound
   │        ╱         │        (计算受限区域)
   │     ╱            │
   │  ╱   Memory-     │
   │╱    bound        │
   │    (访存受限)     │
   │  斜率 = β        │
   │  P = β × I       │
   └───────┬──────────┴──────────────────────►
          脊点            算术强度 I (FLOPs/Byte) [对数轴]
       (Ridge Point)
```

- **左边的斜线**（斜率 = β）：内存带宽屋顶。在这个区域，`β × I < π`，性能被带宽卡住，`P = β × I`。
- **右边的水平线**（高度 = π）：峰值算力屋顶。在这个区域，性能被算力卡住，`P = π`。
- 两条线像「房顶」一样罩住所有可能的性能点 —— 这就是 **Roofline（屋顶线）** 名字的由来。你的程序性能点永远在这个屋顶**下方**。

---

## 四、脊点：Memory-bound 还是 Compute-bound

斜线和水平线的交点叫做**脊点 (Ridge Point)**，LBL 也称它为**机器平衡点 (Machine Balance)**。

脊点的横坐标（临界算术强度）为：

```
I_ridge = π / β  =  峰值算力 / 峰值带宽
```

脊点把整张图分成两个区域，**你的程序算术强度落在哪边，就决定了瓶颈是什么**：

| 条件 | 区域 | 瓶颈 | 优化方向 |
|------|------|------|----------|
| `I < π/β` | 脊点左侧（斜线下） | **访存受限 Memory-bound** | 提高数据局部性、缓存复用、减少内存流量、提高算术强度 |
| `I > π/β` | 脊点右侧（水平线下） | **计算受限 Compute-bound** | 提高向量化/并行度、用更快指令 (FMA/Tensor Core) |

**脊点的另一层含义**：它是「达到峰值算力所需的**最小**算术强度」。脊点越靠右，说明这台机器要「喂饱」它的算力单元，需要越高的算术强度 —— 现代 GPU 算力增长远快于带宽，所以脊点越来越靠右，越来越多的 kernel 落入 memory-bound。

---

## 五、一个手算例子

假设一台 GPU：
- 峰值算力 π = **14,000 GFLOP/s**（14 TFLOP/s，FP32）
- 峰值带宽 β = **900 GB/s**（HBM）

**脊点** = π / β = 14000 / 900 ≈ **15.6 FLOPs/Byte**。
也就是说，算术强度要 ≥ 15.6 才可能吃满算力，否则一定是 memory-bound。

现在分析两个 kernel：

**Kernel A：向量加法 `c[i] = a[i] + b[i]`（FP32）**
- 每个元素：1 次浮点加法 → 1 FLOP
- 每个元素：读 a、读 b、写 c = 3 × 4 字节 = 12 Bytes
- 算术强度 I = 1 / 12 ≈ **0.083 FLOPs/Byte**
- 0.083 << 15.6 → **严重 memory-bound**
- 可达性能 P = min(14000, 900 × 0.083) = min(14000, **75**) = **75 GFLOP/s**

结论：这个 kernel 上限只有 75 GFLOP/s，只有峰值算力的 0.5%。**这不是你写得差，而是算法本质决定的**。想提速，只能从减少内存流量 / 提高带宽利用入手，优化计算毫无意义。

**Kernel B：稠密矩阵乘法 GEMM，N=1024（FP32）**
- 计算量 ≈ 2 × N³ = 2 × 1024³ ≈ 2.15 × 10⁹ FLOPs
- 理想访存（三个矩阵各读/写一次）≈ 3 × N² × 4 = 3 × 1024² × 4 ≈ 1.26 × 10⁷ Bytes
- 算术强度 I ≈ 2.15e9 / 1.26e7 ≈ **170 FLOPs/Byte**
- 170 >> 15.6 → **compute-bound**
- 可达性能 P = min(14000, 900 × 170) = min(14000, 153000) = **14000 GFLOP/s**

结论：GEMM 理论上能吃满算力，是 compute-bound。优化重点是提高计算单元利用率（用 Tensor Core、提高 occupancy 等），而不是省内存。

> 这也解释了为什么深度学习里大矩阵乘法能把 GPU 算力用得很满，而 element-wise 操作（激活、加法）总是被带宽卡住 —— 于是有了「算子融合 (kernel fusion)」来提高算术强度。

---

## 六、怎么读懂 Roofline 图 —— 优化决策

把你的 kernel 实测点（横坐标 = 算术强度，纵坐标 = 实测 GFLOP/s = FLOPs / 运行时间）画到 Roofline 图上，按位置决策：

```
性能
 ▲
π┤ - - - - - -┌──────────────
 │         ╱  │        ● C  ← 点在算力屋顶附近：已很优，难再快
 │      ╱     │
 │   ╱  ● B   │   ← 点贴着斜线但没到脊点高度：带宽已用满，
 │ ╱          │       只有提高算术强度(往右移)才能更快
 │╱   ↑       │
 │  ● A       │   ← 点远低于屋顶：还有很大空间，
 │  距屋顶远   │       先分析是访存还是计算没做好
 └────┴───────┴──────────►  算术强度
     脊点
```

- **点 A（远低于屋顶）**：性能没到上限，有优化空间。先看它在脊点哪边判断瓶颈类型，再针对性优化（提高并行度、改善访存模式、减少同步等）。
- **点 B（贴着斜线、但低于脊点高度）**：带宽已经基本用满了。此时**光提高带宽利用没用**，必须**增大算术强度**（让点向右移，比如 kernel fusion、增加数据复用），才能爬到更高的 FLOP/s。
- **点 C（贴着水平屋顶）**：已接近算力峰值，优化收益很小，可以收工了。

**实测点到屋顶的垂直距离，就是你的优化空间**（Nsight Compute 里用白色虚线标出）。越靠近屋顶越好。

---

## 七、进阶：天花板 (Ceilings)

朴素 Roofline 只给了理论上界。现实中往往达不到，因为缺了某些优化。于是可以在图里加「天花板」，指导优化优先级（Berkeley 论文）：

1. **带宽天花板 (Bandwidth Ceilings)**：在理想带宽斜线下方再画几条更低的斜线，代表因为**缺少软件预取、NUMA 不友好、并发不足**等导致达不到峰值带宽。
2. **核内天花板 (In-core Ceilings)**：在算力屋顶下方画更低的水平线，代表因为**没用 FMA、没做向量化 (SIMD)、指令级并行 (ILP) 不足**而达不到峰值算力。只有补上对应的并行性，性能才能突破这条天花板。
3. **局部性墙 (Locality Walls)**：垂直的墙，代表当前算术强度的极限。想突破就必须改善数据局部性（减少 cache miss）把算术强度做上去。

天花板的意义在于：**它告诉你「下一步做哪个优化能突破哪条线」**，让优化不再盲目。

---

## 八、进阶：分层 Roofline (Hierarchical Roofline)

前面的公式里，「内存流量 Q」默认指 DRAM/HBM。但现代 GPU/CPU 有多级缓存 (L1/L2/DRAM)，**同一个 kernel 在不同层级看到的字节数不同，因此算术强度也不同**。

分层 Roofline 就是**把多条斜线（每级内存/缓存各一条带宽屋顶）叠加在同一张图**上：

```
性能
 ▲                    ┌────────── 峰值算力
 │        ╱   ╱   ╱   │
 │      ╱   ╱   ╱     │   斜率从陡到缓:
 │    ╱   ╱   ╱       │   L1 (最快, 最陡)
 │  ╱   ╱   ╱         │   L2 (中)
 │╱   ╱   ╱           │   DRAM/HBM (最慢, 最缓)
 └────────────────────┴───────►  算术强度
```

- **每一级用它自己的字节数算算术强度**：例如 L2 层用「L1↔L2 之间搬运的字节」做分母，DRAM 层用「DRAM 读写字节」做分母。
- 同一个 kernel 会在图上产生**多个点**（每级一个），通过看它们分别贴近哪条屋顶，可以精细诊断**是哪一级内存成了瓶颈**、缓存复用做得好不好。

这是分析真实复杂 kernel 数据局部性的强力工具。

---

## 九、在 GPU 上实测 Roofline

### 方式一：Nsight Compute（推荐）

NVIDIA Nsight Compute 已**内置 Roofline 图**（`ncu` 工具），是当前生产级首选：

```bash
# 采集包含 Roofline 所需指标的完整 section 集合
ncu --set full -o report ./your_program
# 然后用 Nsight Compute UI 打开 report.ncu-rep，查看 GPU Speed Of Light -> Roofline Chart
```

在 Nsight Compute 的 Roofline 图里：
- **纵轴** = FLOP/s（对数），**横轴** = 算术强度 FLOPs/Byte（对数）
- **斜线** = 内存带宽边界，**水平线** = 峰值性能边界，交点是**脊点**
- **实测点 (Achieved Value)** 落在**蓝色区域**表示 memory-bound，落在**绿色区域**表示 compute-bound
- 结合 **baseline** 功能，可以叠加多次优化的实测点，直观追踪优化进度

> 注意：Nsight Compute 默认的 Roofline 主要针对 device memory (HBM) 层级；要做分层 Roofline，需要自定义 section 文件或基于 metrics 采集。

### 方式二：nvprof（旧卡 / 老工具链）

在较老的 GPU 上可用 `nvprof` 手动采集，然后手算算术强度。以 V100 的 HBM 层级为例：

```bash
nvprof --kernels "your_kernel" \
       --metrics flop_count_dp \
       --metrics dram_read_transactions \
       --metrics dram_write_transactions ./your_program
```

- `flop_count_dp` / `flop_count_sp` / `flop_count_hp`：FP64 / FP32 / FP16 的**总 FLOP 数**
- `dram_read_transactions` + `dram_write_transactions`：DRAM 读写事务数，**每个事务 32 字节**

算术强度（HBM 层级）：

```
AI(HBM) = flop_count_dp / ((dram_read_transactions + dram_write_transactions) × 32)

实测性能 GFLOP/s = flop_count_dp / 运行时间
```

### 方式三：ERT（机器峰值特征化）

厂商标称的峰值算力/带宽往往达不到。**Empirical Roofline Toolkit (ERT)** 通过跑一系列 micro-kernel、扫描各种配置，实测出**各级缓存的峰值带宽 + 峰值 GFLOP/s**，得到更真实的「屋顶」。
仓库：`bitbucket.org/berkeleylab/cs-roofline-toolkit`

---

## 十、常见误区与速查表

### 常见误区

- ❌ **「性能没到峰值就是写得差」**：错。如果 kernel 是 memory-bound，它根本到不了算力峰值，要看它有没有贴近**带宽屋顶**。
- ❌ **「优化就是让计算更快」**：错。memory-bound 的 kernel 优化计算毫无用处，要减少访存 / 提高算术强度。
- ❌ **「算术强度是算法固有属性」**：不完全对。它依赖缓存行为，同一算法在不同缓存/不同实现下算术强度可以变化（这正是分层 Roofline 要处理的）。
- ❌ **「FLOPs 和 FLOP/s 一样」**：错。前者是计数，后者是速率（÷时间）。
- ✅ **提高算术强度的常用手段**：算子融合 (kernel fusion)、分块 (tiling / blocking) 提高缓存复用、循环展开、增大 batch 等。

### 核心公式速查

```
算术强度      I = FLOPs / Bytes                     (FLOPs/Byte)
可达性能      P = min( π , β × I )                  (GFLOP/s)
脊点          I_ridge = π / β                       (FLOPs/Byte)
判定          I < π/β → memory-bound
              I > π/β → compute-bound
实测纵坐标    GFLOP/s = FLOPs / Runtime
```

### 术语速查表

| 术语 | 含义 | 单位 |
|------|------|------|
| Work (W) | 操作总数（通常 FLOPs 计数） | FLOPs |
| Memory Traffic (Q) | 内存搬运总字节（读+写） | Bytes |
| Arithmetic Intensity (I) | I = W/Q，每字节的操作数 | FLOPs/Byte |
| Peak Performance (π) | 机器峰值算力 | GFLOP/s |
| Peak Bandwidth (β) | 机器峰值带宽 | GB/s |
| Attainable Performance (P) | P = min(π, β·I) | GFLOP/s |
| Ridge Point（脊点/机器平衡点） | 两屋顶交点，横坐标 π/β | FLOPs/Byte |
| Memory-bound | I ≤ π/β，受带宽限制 | — |
| Compute-bound | I ≥ π/β，受算力限制 | — |
| Hierarchical Roofline | 叠加 L1/L2/DRAM 多级屋顶 | — |

---

## 参考资料

1. **原始论文**：S. Williams, A. Waterman, D. Patterson, *"Roofline: An Insightful Visual Performance Model for Multicore Architectures"*, Communications of the ACM, 52(4):65–76, 2009.
2. **Wikipedia**：<https://en.wikipedia.org/wiki/Roofline_model>
3. **LBL / NERSC Roofline 文档**：NERSC Performance / Roofline 团队维护，含 Hierarchical Roofline、ERT、V100 实测方法。
4. **NVIDIA Nsight Compute Profiling Guide**：Roofline Charts 章节，<https://docs.nvidia.com/nsight-compute/ProfilingGuide/>
5. **配套论文**：*Hierarchical Roofline Analysis: How to Collect Data using Performance Tools on Intel CPUs and NVIDIA GPUs*，arXiv:2009.02449.
6. **工具**：ERT (`bitbucket.org/berkeleylab/cs-roofline-toolkit`)、NERSC Roofline 示例 (`github.com/cyanguwa/nersc-roofline`)。
