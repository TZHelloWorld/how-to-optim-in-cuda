# CUDA Attention 算子优化指南

> 本文以 Scaled Dot-Product Attention 为例，系统介绍 CUDA 上**算子融合（kernel fusion）**与**分块流式计算**的典型优化方法。Attention 的特殊之处在于：它由两个矩阵乘夹着一个 softmax 构成，朴素实现会在显存中生成两个 N×N 的中间矩阵——**序列越长，这两个矩阵的读写越是压倒性的瓶颈**。全文从最朴素的"三个 kernel"实现出发，沿着"减少 N² 中间数据在显存中的往返"这条主线，经过算子融合（V1）、Online Softmax（V2），最终演进到 FlashAttention（V3）与 FlashAttention-2（V4）——让 N×N 矩阵**从头到尾不落地**，显存占用从 O(N²) 降到 O(N)，速度提升数倍。
>
> 本文内容完全自包含：理解全文所需的 GPU 执行模型、内存层次、归约、数值稳定性、算术强度等基础概念，都在第 2 章从零讲起，无需先阅读其他资料。

---

## 目录

- [第 1 章 问题定义：什么是 Attention](#第-1-章-问题定义什么是-attention)
- [第 2 章 预备知识：GPU 基础、归约、数值稳定性与流量分析](#第-2-章-预备知识gpu-基础归约数值稳定性与流量分析)
- [第 3 章 优化路线总览](#第-3-章-优化路线总览)
- [第 4 章 V0：基准实现——三个 kernel，中间矩阵落地](#第-4-章-v0基准实现三个-kernel中间矩阵落地)
- [第 5 章 V1：算子融合入门——scale + mask + softmax 单 kernel](#第-5-章-v1算子融合入门scale--mask--softmax-单-kernel)
- [第 6 章 V2：Online Softmax——把三遍扫描压成一遍](#第-6-章-v2online-softmax把三遍扫描压成一遍)
- [第 7 章 V3：FlashAttention——分块融合，N² 矩阵永不落地](#第-7-章-v3flashattention分块融合n-矩阵永不落地)
- [第 8 章 V4：FlashAttention-2——并行度与指令效率的工程改进](#第-8-章-v4flashattention-2并行度与指令效率的工程改进)
- [第 9 章 场景扩展：推理 Decode 阶段与 Flash-Decoding](#第-9-章-场景扩展推理-decode-阶段与-flash-decoding)
- [第 10 章 工程化：PyTorch SDPA 与正确性验证](#第-10-章-工程化pytorch-sdpa-与正确性验证)
- [第 11 章 总结与实践建议](#第-11-章-总结与实践建议)
- [附录：关键概念速查](#附录关键概念速查)

---

## 第 1 章 问题定义：什么是 Attention

### 1.1 Scaled Dot-Product Attention 的定义

Attention 是 Transformer 的核心算子，定义为：

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V

Q（query）: N×d      N = 序列长度（token 数），d = 每个注意力头的维度
K（key）  : N×d      （本文聚焦单个注意力头；多头即把此计算独立重复 h 次）
V（value）: N×d
输出 O    : N×d
```

计算分三步，每一步都有明确的语义：

```
① S = Q·Kᵀ / √d        S: N×N   每个 query 与每个 key 的"相似度分数"
② P = softmax(S)        P: N×N   逐行归一化成概率分布（每行和为 1）
③ O = P·V               O: N×d   按概率对 value 加权求和
```

其中 softmax 对矩阵的**每一行独立**进行：

```
softmax(x)_j = exp(x_j) / Σₖ exp(x_k)
```

CPU/PyTorch 上的参考实现只有三行：

```python
S = Q @ K.T / math.sqrt(d)        # ① N×N 分数矩阵
P = torch.softmax(S, dim=-1)      # ② 逐行归一化
O = P @ V                         # ③ 加权求和
```

实际使用中第①②步之间通常还有 **mask**：最常见的是**因果掩码（causal mask）**——生成式模型中位置 i 的 query 只允许看位置 j ≤ i 的 key，实现为把 S 中 j > i 的元素置为 -∞（softmax 后即为 0）。本文的 kernel 都会带上这个可选项。

### 1.2 为什么要除以 √d：一个完整的数学推导

这个缩放因子不是经验参数，而是有严格推导的。先备好两条方差运算规律（X、Y 独立时）：

```
Var(cX)    = c² · Var(X)                 —— 常数平方地放大方差
Var(X + Y) = Var(X) + Var(Y)             —— 独立随机变量方差相加
Var(X · Y) = E[X²]·E[Y²] − (E[X]E[Y])²   —— 独立乘积；均值为 0 时 = Var(X)·Var(Y)
```

**问题设定**：假设 q 和 k 的每个分量都是独立随机变量，均值 0、方差 1（经过 LayerNorm 与合理初始化后的近似成立）。看未缩放的点积：

```
s = Σₗ qₗ·kₗ    （l = 1..d）
```

**第一步：单项的方差。** qₗ、kₗ 独立且均值为 0：

```
Var(qₗ·kₗ) = Var(qₗ) · Var(kₗ) = 1 × 1 = 1
```

**第二步：求和的方差。** d 个独立项相加，方差线性累加：

```
Var(s) = Σₗ Var(qₗ·kₗ) = d        →  标准差 std(s) = √d
```

也就是说，**点积的典型幅度随 √d 增长**：d=64 时分数普遍散布在 ±8×3 = ±24 的范围（3 个标准差），d=512 时达到 ±68。

**第三步：为什么这会出问题——softmax 饱和。** softmax 中的 exp 对大输入极其敏感：当一行分数中最大值比其他值大出几十，exp 之后最大项独占几乎全部权重，softmax 输出趋近 one-hot。此时：

- **前向**：注意力退化为"硬选择"，失去了对多个位置软性加权的能力；
- **反向**：softmax 的梯度 ∂P/∂S 在饱和区趋近于 0，梯度无法流动，训练停滞；
- **数值**：fp32 下 exp(89) 直接上溢为 inf（fp16 上溢点更低，exp(11.1)）。

**解决：除以 √d，把方差归一回 1。**

```
Var(s/√d) = Var(s)/(√d)² = d/d = 1
```

无论 d 是 64 还是 512，缩放后分数的尺度恒定，softmax 始终工作在梯度正常流动的"敏感区"。这就是分母 √d 的全部来由——**它是标准差归一化，不是工程调参**。

> 本目录下的《attention_sqrt_dk_derivation.md》给出了这一推导的独立完整版（含均值/方差全部运算规律），可作为参考。

### 1.3 计算量、访存量与显存：N² 是万恶之源

以 N=4096、d=64、fp32 为例给三笔账（单个注意力头）：

**计算量**：两个矩阵乘各 2N²d，softmax 约 5N²（求最大、减、exp、求和、除）：

```
FLOP ≈ 2N²d (QKᵀ) + 2N²d (PV) + 5N² (softmax)
     ≈ 4N²d = 4 × 4096² × 64 ≈ 4.3 GFLOP
```

**数据规模**——这是 Attention 与 GEMM 最大的不同：

| 数据 | 元素数 | 大小（fp32） |
|------|--------|-------------|
| Q、K、V、O（各） | N×d = 26 万 | **各 1 MB** |
| S、P（各） | N² = 1678 万 | **各 64 MB** |

输入输出总共 4 MB，而两个**中间矩阵**各 64 MB——中间数据比有效输入输出大 30 倍，且随 N **平方**增长（N=32K 时 S 单个就要 4 GB）。

**访存量（朴素实现）**：S 和 P 每次在 kernel 之间传递都要在显存（HBM）走一个来回。即使 softmax 只算一遍读写，最少也有：

```
写 S + 读 S + 写 P + 读 P = 4 × 64 MB = 256 MB 的 N² 级流量
```

**算术强度**只有 4.3 GFLOP / 268 MB ≈ **16 FLOP/Byte**，低于典型 GPU 约 35 FLOP/Byte 的平衡点（见 2.6 节）——**朴素 Attention 是 memory-bound 的**，尽管它内部有两个大矩阵乘。瓶颈不在计算，而在 N² 中间矩阵反复过显存。

于是优化的主线呼之欲出：**让 S 和 P 不要落地**。如果 N² 的数据能在片上（共享内存/寄存器）随产随消，访存量就只剩 Q、K、V、O 这 4 MB 的量级，算子回到 compute-bound，同时 O(N²) 显存占用也消失。这正是 V1 → V3 逐步实现的目标。

### 1.4 并行化的基本形状

Attention 的天然并行维度有三个：

```
batch × heads    —— 完全独立，最外层并行（映射到 Grid 的一个维度）
序列维 N（行）    —— softmax 逐行独立，行与行可以并行
head 维 d        —— 输出的 d 个通道共享同一行概率 P
```

softmax **行内**则是"先全行归约（求 max、求 sum）、再逐元素变换"的模式——行内并行需要归约原语支撑（2.4 节）。后文所有版本的线程组织都是这三个维度的不同分配方式。

---

## 第 2 章 预备知识：GPU 基础、归约、数值稳定性与流量分析

本章从零介绍理解全文所必需的概念。已熟悉 CUDA 的读者可只读 2.4~2.7（Attention 特有的工具），其余跳过。

### 2.1 GPU 执行模型速览

CUDA 将一次 kernel 启动的线程组织为三层：

```
Grid（一次启动的全部线程）
 └── Block（同一 SM 上的一组线程，可用共享内存通信、__syncthreads() 同步）
      └── Thread（最小执行单位，私有寄存器）
```

| 内置变量 | 含义 |
|----------|------|
| `threadIdx` / `blockIdx` | 线程在 Block 内 / Block 在 Grid 中的坐标 |
| `blockDim` / `gridDim` | Block / Grid 各维度的大小 |

硬件真正的调度单位是 **Warp**——编号连续的 32 个线程，**锁步执行同一条指令**（SIMT）。两条推论贯穿全文：

1. 一条访存指令实际发出的是"32 个线程的 32 个地址"，地址连续才能合并成最少的内存事务（见 2.2）；
2. Warp 内 32 个线程天然同步，彼此交换数据可以用寄存器级的 shuffle 指令，无需经过内存（见 2.4）。

### 2.2 内存层次与合并访存

```
速度：   慢 ──────────────────────────────────────────────── 快
         全局内存(HBM)     L2 缓存    共享内存(SRAM)      寄存器
容量：   几十 GB           几十 MB    ~100 KB / SM        255 个 / 线程
延迟：   ~400-600 cycles   ~200      ~20-30 cycles       ~1 cycle
可见性： 所有线程          所有线程   同一 Block 内        仅本线程
```

三条使用规则：

- **合并访存**：同一 Warp 的 32 个线程访问 32 个**连续**地址时，硬件合并为最少的内存事务（效率 100%）；地址间距一整行时退化为 32 次独立事务（效率 ≤12.5%）。写 kernel 时始终让 `threadIdx.x` 对应数据中**连续**的那个维度；
- **共享内存**是程序员手动管理的片上缓存，每 Block 独占一份，生命周期随 Block 结束，用于组织 Block 内线程间的数据复用——FlashAttention 中 K/V 分块正是驻留在这里；
- **寄存器**最快但线程私有，累加器一类"越攒越多的私有状态"应放这里。

一个对本文至关重要的事实：**HBM 带宽（~1 TB/s）与片上带宽（共享内存 ~20 TB/s、寄存器更高）之间有 1~2 个数量级的鸿沟**。同一份数据在 HBM 多走一个来回，就要多付一份最贵的运费——这是全文一切优化的动机。

### 2.3 算子融合（Kernel Fusion）的收益模型

深度学习框架默认"一个算子一个 kernel"。相邻 kernel 之间传递数据的唯一通道是**全局内存**：

```
kernel A 算出 X → 写 HBM → kernel B 从 HBM 读 X → 计算
```

若把 A、B 融合成一个 kernel，X 就能以寄存器/共享内存为载体直接传递，省掉一写一读。收益的量化：

```
省下的时间 ≈ 2 × sizeof(X) / HBM带宽    （X 越大越划算）
```

对 Attention 而言 X 是 64 MB 的 N² 矩阵，一写一读约 128 MB ≈ 0.13 ms（1 TB/s 时），而整个 attention 的计算在理想算力下不足 0.15 ms——**省一次中间矩阵往返，收益与全部计算时间同量级**。此外融合还附带省去 kernel 启动开销与显存占用。

融合的难点在于：**后一个算子往往需要前一个算子的"全局结果"**。softmax 就是典型——归一化分母需要整行的和，看似必须"先算完整行、再归一化"，这正是 V2（Online Softmax）要拆掉的锁。

### 2.4 归约与两级归约：Attention kernel 的核心组件

softmax 的行内求 max、求 sum 都是**归约（reduction）**：把一组数缩成一个数。GPU 上的高效实现分两级：

**第一级：Warp 内归约（寄存器直传）。** `__shfl_xor_sync` 让 Warp 内线程直接交换寄存器值（延迟 1~2 周期，不经内存），蝶形（butterfly）模式 5 步完成 32 个数的归约，且**每个线程都持有最终结果**（省去广播）：

```cuda
__device__ float warpReduceMax(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    return v;   // 32 个 lane 都得到全 Warp 最大值
}
__device__ float warpReduceSum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}
```

**第二级：Block 内归约。** 各 Warp 先内部归约，代表（lane 0）把部分结果写进共享内存，再由第一个 Warp 对这些部分结果做一次 Warp 归约：

```cuda
__device__ float blockReduceMax(float v) {
    __shared__ float warpRes[32];                 // 最多 32 个 Warp
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    v = warpReduceMax(v);
    if (lane == 0) warpRes[wid] = v;
    __syncthreads();
    int nWarp = (blockDim.x + 31) / 32;
    v = (lane < nWarp) ? warpRes[lane] : -INFINITY;
    if (wid == 0) v = warpReduceMax(v);
    // 把结果广播给全 Block
    __shared__ float result;
    if (threadIdx.x == 0) result = v;
    __syncthreads();
    return result;
}
```

`blockReduceSum` 同理（初值换成 0、`fmaxf` 换成 `+`）。这对组件将在 V1 的融合 softmax 中直接使用。

### 2.5 数值稳定性：exp 的上溢与 Safe Softmax

直接按定义计算 softmax 在浮点下会爆炸：

```
fp32: exp(x) 在 x > 88.7 时上溢为 inf     （inf/inf = NaN，全行报废）
fp16: 上溢点更低，x > 11.1
```

而 1.2 节说过缩放后的分数方差为 1，但 N 很大时一行的**最大值**仍可能达到几十。标准解法是 **Safe Softmax**——利用恒等式，给每个元素减去该行最大值 m：

```
softmax(x)_j = exp(x_j) / Σ exp(x_k)
             = exp(x_j − m) / Σ exp(x_k − m)      其中 m = max(x)
```

分子分母同乘 exp(−m)，数学上恒等；数值上 x_j − m ≤ 0，exp 的输入永不为正，**彻底杜绝上溢**（下溢为 0 是无害的）。代价是多了一遍"求 max"扫描，于是标准 softmax 需要**三遍扫描**：

```
pass 1: m = max(x)                     （读一遍 x）
pass 2: l = Σ exp(x_j − m)             （再读一遍 x）
pass 3: y_j = exp(x_j − m) / l         （第三遍读 x，写 y）
```

"三遍扫描"意味着三倍读流量——这个数字在 2.7 节的流量账本和 V2 的优化中都会反复出现。

### 2.6 算术强度与 Roofline：给 Attention 定位

**算术强度 AI = 计算量（FLOP）/ 访存量（Byte）**。硬件的平衡点是 `峰值算力/峰值带宽`（典型 GPU：~35 TFLOPS fp32 / ~1 TB/s ≈ 35 FLOP/B）：

- AI < 35：**memory-bound**，上限是带宽，优化目标是减流量、跑满带宽；
- AI > 35：**compute-bound**，上限是算力，优化目标是提高数据复用、跑满算力。

Attention 三个阶段各自的定位（N=4096, d=64, fp32）：

| 阶段 | FLOP | 独立成 kernel 时的最少 HBM 流量 | AI (FLOP/B) | 定位 |
|------|------|-------------------------------|-------------|------|
| ① S = QKᵀ | 2N²d | 读 Q,K (2MB) + 写 S (64MB) | ~32 | 被写 S 拖累 |
| ② softmax | ~5N² | 读 S (64MB×3 遍) + 写 P (64MB) | **~0.3** | 重度 memory-bound |
| ③ O = PV | 2N²d | 读 P (64MB) + 读 V (1MB) + 写 O (1MB) | ~31 | 被读 P 拖累 |

单看矩阵乘本身，2N²d 的计算配 Nd 级的输入本应是高 AI 的 compute-bound 计算（同 GEMM）；**是 N² 中间矩阵的落地把整条链拖进了 memory-bound 区**。反过来说：把 S/P 留在片上，AI 立即回升——这就是 FlashAttention 的 Roofline 逻辑。

### 2.7 一条贯穿全文的分析工具：数"N² 流量过 HBM 几遍"

与逐 kernel 分析相比，一个更快的定位方法是只盯着**最大的数据（N² 矩阵）在 HBM 上走了几个来回**——因为它比其他所有数据大 30 倍以上，其他流量几乎可以忽略：

```
一遍 = N² × 4 字节 = 64 MB（本文默认参数下）
```

| 版本 | S/P 的 HBM 读写遍数 | N² 流量 |
|------|-----|---------|
| V0（三 kernel + 三遍扫描 softmax） | 写S 1 + 读S 3 + 写P 1 + 读P 1 = **6 遍** | ~384 MB |
| V1（融合 softmax，行驻留片上） | 写S 1 + 读S 1 + 写P 1 + 读P 1 = **4 遍** | ~256 MB |
| V2（online softmax，理论意义） | 同 V1（单独用时省的是行内扫描遍数） | ~256 MB |
| V3（FlashAttention） | **0 遍**（S/P 只存在于片上） | ~0 |

后文每个版本的效果，都可以先用这张表预估，再看实现细节。

---

## 第 3 章 优化路线总览

每个版本针对上一版暴露的一个具体瓶颈：

```
V0: 朴素实现：三个 kernel（QKᵀ → softmax → PV），S/P 落地 HBM
 │   瓶颈：N² 中间矩阵 6 遍过 HBM；显存占用 O(N²)
 ▼
V1: 融合 scale + mask + softmax 为单 kernel，一行一 Block、块内两级归约
 │   瓶颈：融合只省了 softmax 内部的重复扫描，S、P 本身仍要落地（4 遍）
 ▼
V2: Online Softmax：max 与分母 l 在一遍扫描中同时递推得到
 │   意义：softmax 从"必须先看完全行"变成"可以流式逐段处理"
 │        ——单独使用收益有限，但它是分块融合的数学钥匙
 ▼
V3: FlashAttention：K/V 沿序列分块驻留片上，O 在线重缩放
 │   S/P 永不落地：N² 流量归零，显存 O(N)，速度数倍提升
 │   瓶颈：（FA-1 风格实现）并行度与 Warp 分工仍有工程优化空间
 ▼
V4: FlashAttention-2：交换内外循环、序列维并行、Warp 间 split-Q、
    延迟归一化——逼近硬件矩阵乘的效率上限
```

各版本解决的瓶颈分类：

| 瓶颈类别 | 具体问题 | 解决版本 |
|---------|---------|---------|
| 访存量 | softmax 三遍扫描 | V1 / V2 |
| 访存量 | N² 中间矩阵落地 | V3 |
| 显存容量 | O(N²) 中间矩阵 | V3 |
| 算法结构 | softmax 的全局依赖阻碍分块 | V2 |
| 并行度/指令效率 | 循环顺序、Warp 分工、非矩阵乘运算 | V4 |
| 特殊场景 | decode 阶段并行度骤减 | 第 9 章 |

一个有用的对照：**GEMM 的优化是在"计算受限"算子上建立数据复用；Attention 的优化是先用"融合 + 分块"把算子从 memory-bound 拉回 compute-bound，然后才轮到 GEMM 式的优化**。FlashAttention 内部的每个分块小矩阵乘，用的正是 GEMM 的全套技术。

---

## 第 4 章 V0：基准实现——三个 kernel，中间矩阵落地

### 4.1 实现

最直接的实现是把三步各交给一个 kernel（矩阵乘可以调 cuBLAS，这里为完整性给出朴素版）：

```cuda
// kernel 1: S = Q·Kᵀ / √d    （每线程算 S 的一个元素）
__global__ void qk_kernel(const float* Q, const float* K, float* S,
                          int N, int d, float scale) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // key 序号 j
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // query 序号 i
    if (row < N && col < N) {
        float s = 0.0f;
        for (int x = 0; x < d; x++)
            s += Q[row * d + x] * K[col * d + x];      // K 按行存，Kᵀ 即按行取 K
        S[row * N + col] = s * scale;                  // scale = 1/√d
    }
}

// kernel 2: P = softmax(S) 逐行（此处为示意的三遍扫描；每线程处理一行——效率极低）
__global__ void softmax_kernel(const float* S, float* P, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    float m = -INFINITY;
    for (int j = 0; j < N; j++) m = fmaxf(m, S[row * N + j]);        // pass 1
    float l = 0.0f;
    for (int j = 0; j < N; j++) l += expf(S[row * N + j] - m);       // pass 2
    for (int j = 0; j < N; j++)                                      // pass 3
        P[row * N + j] = expf(S[row * N + j] - m) / l;
}

// kernel 3: O = P·V    （每线程算 O 的一个元素）
__global__ void pv_kernel(const float* P, const float* V, float* O,
                          int N, int d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // 0..d-1
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // 0..N-1
    if (row < N && col < d) {
        float o = 0.0f;
        for (int k = 0; k < N; k++)
            o += P[row * N + k] * V[k * d + col];
        O[row * d + col] = o;
    }
}
```

宿主端按顺序启动三个 kernel，中间结果 S、P 各分配一块 N² 的显存。

### 4.2 瓶颈分析

按 2.7 节的流量账本：

```
kernel 1 写 S:        64 MB
kernel 2 读 S 三遍:   192 MB     ← safe softmax 的三遍扫描（2.5 节）
kernel 2 写 P:        64 MB
kernel 3 读 P:        64 MB
──────────────────────────────
N² 级流量合计:        384 MB     （有效输入输出 Q/K/V/O 合计仅 4 MB）
```

在 1 TB/s 带宽下仅数据搬运就需要 ~0.4 ms，而 4.3 GFLOP 的计算在 35 TFLOPS 下只要 ~0.12 ms——**四分之三的时间在等内存**。此外：

- **显存占用 O(N²)**：N=32K 时单头单批就要 2×4 GB，长序列直接 OOM；
- 三个 kernel 三次启动开销，且 kernel 之间无法重叠（有数据依赖）。

两个矩阵乘 kernel 本身也远未优化（无分块复用），但**先解决 N² 流量才是主要矛盾**——即使把两个 GEMM 换成 cuBLAS，S/P 的 6 遍落地照旧，总时间改善有限。这个判断本身就是一课：**优化前先算流量账，找最大的搬运项下手**。

---

## 第 5 章 V1：算子融合入门——scale + mask + softmax 单 kernel

### 5.1 优化思路

V0 的 softmax kernel 有两个显而易见的低效：

1. **每线程独自处理一整行**：行内 4096 个元素串行扫描，且相邻线程（处理相邻行）同一时刻访问的地址相距一整行 N——完全不合并（违反 2.2 节规则）；
2. **三遍扫描三倍读流量**。

V1 的改法：**一行分配一个 Block**（256 线程协作处理 4096 个元素），行内归约用 2.4 节的两级归约完成；同时把 scale 和 causal mask 一并融合进来（它们本来还要各自多走一遍 N² 读写）。三遍扫描在 Block 内进行，行数据在三遍之间大概率驻留 L1/L2，HBM 层面接近一读一写。

```
线程组织：  grid = N（一行一 Block）,  block = 256
每线程分工：跨步处理本行的 N/256 = 16 个元素
行内归约：  blockReduceMax / blockReduceSum（2.4 节组件）
```

### 5.2 实现

```cuda
// P = softmax(mask(S · scale))，逐行；一行一个 Block
__global__ void fused_softmax_kernel(const float* S, float* P,
                                     int N, float scale, bool causal) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* srow = S + (size_t)row * N;
    float*       prow = P + (size_t)row * N;

    // pass 1: 求行最大值（scale 与 mask 在读取时现场应用，不额外过显存）
    float m = -INFINITY;
    for (int j = tid; j < N; j += blockDim.x) {
        float x = srow[j] * scale;
        if (causal && j > row) x = -INFINITY;        // 因果掩码
        m = fmaxf(m, x);
    }
    m = blockReduceMax(m);

    // pass 2: 求分母 l = Σ exp(x − m)
    float l = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float x = srow[j] * scale;
        if (causal && j > row) x = -INFINITY;
        l += __expf(x - m);                          // exp(-inf) = 0，mask 自然生效
    }
    l = blockReduceSum(l);

    // pass 3: 写出归一化结果
    float inv_l = 1.0f / l;
    for (int j = tid; j < N; j += blockDim.x) {
        float x = srow[j] * scale;
        if (causal && j > row) x = -INFINITY;
        prow[j] = __expf(x - m) * inv_l;
    }
}
// 启动：fused_softmax_kernel<<<N, 256>>>(S, P, N, 1.0f/sqrtf(d), true);
```

三个值得注意的设计点：

- **跨步循环 `j = tid; j += blockDim.x`**：同一时刻 Warp 内 32 个线程访问的 j 连续 → 地址连续 → 完全合并，同时任意行长 N 都能被 256 线程覆盖；
- **scale/mask 融合进读取路径**：它们不再是独立 kernel，省掉了各自的 N² 读写各一遍；`-INFINITY` 经 exp 变成 0，mask 的语义自动正确；
- **`__expf`** 是硬件特殊函数单元（SFU）上的快速指数，softmax 场景精度足够。

### 5.3 效果与遗留问题

| | V0 softmax | V1 融合 softmax |
|--|-----------|----------------|
| 行内并行 | 无（单线程串行） | 256 线程 + 两级归约 |
| 访存合并 | 完全不合并 | 完全合并 |
| scale/mask | 需独立 kernel（各 +2 遍 N²） | 融合，零额外流量 |
| S 的 HBM 读 | 3 遍 | ~1 遍（三遍扫描命中缓存） |

softmax 这一段通常可加速一个数量级。但从全局账本看，**S 和 P 这两个 N² 矩阵本身还在**：QKᵀ 写 S 一遍、softmax 读 S 写 P 各一遍、PV 读 P 一遍——4 遍 N² 流量（256 MB）与 O(N²) 显存原封未动。

要消灭它们，就必须把三步彻底融成**一个** kernel：S 的每个分块算出来就地喂给 softmax，softmax 的结果就地喂给 PV 累加。拦路的正是 softmax 的**全局依赖**——分母要等整行算完才知道。下一章拆掉这把锁。

---

## 第 6 章 V2：Online Softmax——把三遍扫描压成一遍

### 6.1 问题：safe softmax 的"先全局后局部"结构

safe softmax 的 m（行最大值）和 l（分母）都是全行的函数。按 2.5 节的三遍扫描，处理任何一个元素之前都要先扫完整行拿到 m——这意味着 S 的一行必须**完整存在**于某处。若想分块流式处理（每次只看一小段），必须回答：

> 看到一半时用的是"临时最大值"，后面来了更大的数怎么办？

### 6.2 递推公式：错了可以补救

Online Softmax（Milakov & Gimelshein, 2018）的核心发现是：**基于旧 m 算出的部分和，可以用一个指数因子事后修正**。逐个读入 x₁, x₂, ... 时维护两个量：

```
m_t = 前 t 个元素的最大值
l_t = Σᵢ₌₁..t exp(xᵢ − m_t)      （以当前最大值为基准的部分分母）
```

递推：读入新元素 x_{t+1}：

```
m_{t+1} = max(m_t, x_{t+1})
l_{t+1} = l_t · exp(m_t − m_{t+1}) + exp(x_{t+1} − m_{t+1})
          └────────┬────────┘
           修正因子：旧部分和统一"换基准"
```

**正确性**：l_t 中每一项都是 exp(xᵢ − m_t)，乘以 exp(m_t − m_{t+1}) 后变成 exp(xᵢ − m_{t+1})——所有旧项被一次性换到新基准下，再加上新项，恰好是定义式。两种情形自查：

- 新元素不更大（m_{t+1} = m_t）：修正因子 = exp(0) = 1，退化为普通累加；
- 新元素更大：旧项统一乘上一个 < 1 的因子缩小——它们相对新的最大值变"不重要"了，且永不上溢。

扫完一遍后 m_N、l_N 与三遍扫描的结果**逐位一致**（不是近似）。代码只有几行：

```cuda
float m = -INFINITY, l = 0.0f;
for (int j = 0; j < N; j++) {
    float m_new = fmaxf(m, x[j]);
    l = l * __expf(m - m_new) + __expf(x[j] - m_new);
    m = m_new;
}
// 此时 m、l 与"先求 max 再求 sum"完全一致
```

于是 softmax 从三遍扫描降为**两遍**（一遍得 m/l，一遍写归一化结果）。

### 6.3 更进一步：连输出的加权和也能在线递推

单独优化 softmax 只省一遍扫描，真正的威力在于把第③步（O = P·V）也拉进递推。注意输出行：

```
o = Σⱼ softmax(s)_j · v_j = ( Σⱼ exp(s_j − m) · v_j ) / l
```

分子同样是"以 m 为基准的指数加权和"——**和 l 服从完全相同的修正规律**。维护第三个量（d 维向量）：

```
o~_t = Σᵢ₌₁..t exp(sᵢ − m_t) · vᵢ       （未归一化的输出）

递推：o~_{t+1} = o~_t · exp(m_t − m_{t+1}) + exp(s_{t+1} − m_{t+1}) · v_{t+1}
最后：o = o~_N / l_N
```

这意味着：**处理完 s 的一段并丢弃它之后，照样能得到精确的最终输出**——每个分数 s_j 只在算出来的那一刻被用一次（喂进 o~ 和 l 的递推），之后再也不需要。P 矩阵从数学上就不必存在了。

### 6.4 意义与遗留问题

V2 本身若做成独立 softmax kernel，收益只是 3 遍读变 2 遍。它真正的角色是**钥匙**：

| softmax 的锁 | Online Softmax 的解 |
|--------------|-------------------|
| 分母需要全行的和 | l 可流式递推，随时可修正 |
| safe 化需要全行的 max | m 可流式递推，旧结果按 exp(Δm) 补偿 |
| P·V 需要完整的 P | o~ 与 l 同规律递推，P 逐块用完即弃 |

剩下的只是一个工程问题：递推是逐元素描述的，直接实现毫无并行度与数据复用。把它**按块**组织——每次处理 K/V 的一个 tile，块内是小矩阵乘（可以用满 GPU 算力），块间做一次修正——就是 FlashAttention。

---

## 第 7 章 V3：FlashAttention——分块融合，N² 矩阵永不落地

### 7.1 核心思想

把 V2 的逐元素递推升级为**逐块递推**，并让所有中间量驻留片上：

```
外层：每个 Q 行块（Br 行）独立处理（可并行）
内层：沿序列遍历 K/V 的列块（每块 Bc 行），对每一块：
  ① 片上算分块分数   S_ij = Q_i · K_jᵀ · scale        （Br×Bc，只存在于片上）
  ② 片上算分块指数   P~ = exp(S_ij − m_new)
  ③ 在线修正累加     o~ ← o~ · exp(m_old − m_new) + P~ · V_j
                     l  ← l  · exp(m_old − m_new) + rowsum(P~)
内层结束：O_i = o~ / l，写回 HBM（唯一一次 N×d 级写出）
```

```
      K/V 分块 j=0    j=1    j=2   ...
       ┌──────┬──────┬──────┬────┐
 Q 块 i│ tile │ tile │ tile │    │   每个 tile：小矩阵乘 + 在线修正
 (Br行)│ (i,0)→ (i,1)→ (i,2)→ ...│   S/P~ 只活在共享内存/寄存器里
       └──────┴──────┴──────┴────┘
         m, l, o~ 随块滑动持续更新（驻留寄存器）
```

对照 2.7 节账本：S 和 P 的 HBM 读写**归零**；HBM 流量只剩 Q 读 1 遍、K/V 各被每个 Q 块读 1 遍、O 写 1 遍。显存占用从 O(N²) 降到 O(N)（只需给每行留 m/l 统计量，甚至可以不留）。

### 7.2 完整算法（forward，单头）

记 Tr = N/Br（Q 块数）、Tc = N/Bc（K/V 块数）：

```
for i = 0 .. Tr−1:                         # 每个 Q 块——Grid 并行
    加载 Q_i (Br×d) 到片上
    初始化 m = −inf (Br维), l = 0 (Br维), o~ = 0 (Br×d)
    for j = 0 .. Tc−1:                     # 内层顺序循环
        加载 K_j, V_j (各 Bc×d) 到共享内存
        S = Q_i · K_jᵀ · scale                     # Br×Bc，片上
        （causal：屏蔽 j 块中列号 > 行号的元素为 −inf）
        m_new   = max(m, rowmax(S))                # 逐行
        P~      = exp(S − m_new)                   # Br×Bc，片上
        corr    = exp(m − m_new)                   # 逐行修正因子
        l       = l · corr + rowsum(P~)
        o~      = o~ · corr + P~ · V_j             # Br×d 累加
        m       = m_new
    O_i = o~ / l                                    # 写回 HBM
```

正确性完全由 6.2/6.3 节的递推保证——FlashAttention 是**精确算法**，不是近似。

### 7.3 教学版 CUDA 实现

下面给出一个以可读性为先的实现：每个 Block 处理一个（batch×head 的）Q 行块，`blockDim.x = Br`，**每个线程负责块内一行**——q、o~、m、l 全部驻留该线程的寄存器，K/V tile 由全 Block 协作装入共享内存：

```cuda
// 教学版 FlashAttention forward（fp32，单头版式）
// Q/K/V/O: [BH, N, D]（BH = batch*heads），要求 N 是 Bc 的倍数、D ≤ 模板上限
template <int Br, int Bc, int D>
__global__ void flash_attn_v3(const float* Q, const float* K,
                              const float* V, float* O,
                              int N, float scale, bool causal) {
    int qb  = blockIdx.x;                 // Q 行块编号 0..N/Br-1
    int bh  = blockIdx.y;                 // batch*head 编号
    int tid = threadIdx.x;                // 0..Br-1：本线程负责的块内行

    // 指针推进到本 (batch, head)
    size_t off = (size_t)bh * N * D;
    Q += off; K += off; V += off; O += off;

    __shared__ float Ks[Bc][D];
    __shared__ float Vs[Bc][D];

    // ① 本线程的 Q 行与在线统计量 → 寄存器（全程驻留）
    int qRow = qb * Br + tid;
    float q[D], o[D];
    for (int x = 0; x < D; x++) { q[x] = Q[qRow * D + x]; o[x] = 0.0f; }
    float m = -INFINITY, l = 0.0f;

    // ② 内层：沿序列遍历 K/V 块
    for (int j0 = 0; j0 < N; j0 += Bc) {
        if (causal && j0 > qRow) break;   // 整块都在掩码之外，直接结束（见 7.5）

        // 协作装载 K/V tile（Br 个线程搬 Bc×D 个元素，跨步循环、地址连续）
        for (int idx = tid; idx < Bc * D; idx += Br) {
            Ks[idx / D][idx % D] = K[(size_t)(j0 + idx / D) * D + idx % D];
            Vs[idx / D][idx % D] = V[(size_t)(j0 + idx / D) * D + idx % D];
        }
        __syncthreads();

        // 本行 vs tile 内 Bc 个 key：算分数 + 在线修正累加
        for (int c = 0; c < Bc; c++) {
            if (causal && j0 + c > qRow) break;      // 块内尾部掩码

            float s = 0.0f;                          // s = q · k_c
            for (int x = 0; x < D; x++) s += q[x] * Ks[c][x];
            s *= scale;

            float m_new = fmaxf(m, s);
            float p     = __expf(s - m_new);         // 本分数的指数权重
            float corr  = __expf(m - m_new);         // 旧累积的修正因子
            l = l * corr + p;
            for (int x = 0; x < D; x++)
                o[x] = o[x] * corr + p * Vs[c][x];   // o~ 递推
            m = m_new;
        }
        __syncthreads();                             // 用完才许下一轮覆盖 tile
    }

    // ③ 归一化并写回（合并写出的行主序连续段）
    float inv_l = 1.0f / l;
    for (int x = 0; x < D; x++) O[qRow * D + x] = o[x] * inv_l;
}
// 启动：dim3 grid(N/Br, batch*heads);
//       flash_attn_v3<64, 64, 64><<<grid, 64>>>(Q, K, V, O, N, 1.0f/sqrtf(64), true);
```

结构上与 GEMM 优化中经典的共享内存分块（Block Tiling，见本仓库 `01_gemm` 指南第 6 章）完全同构：**K/V tile 是"料盒"（共享内存，滑动覆盖），q/o~/m/l 是"越攒越多的私有状态"（寄存器）**，两次 `__syncthreads()` 的语义也一样（没到齐不许吃 / 没吃完不许撤）。

> 教学版为清晰做了取舍：块内分数逐个（`c` 循环）处理，S_ij 未显式成块；点积 `q·k_c` 未做 Warp 级并行；未用 Tensor Core。生产实现（flash-attn 库 / CUTLASS）把 ①② 换成分块矩阵乘 + Tensor Core（即本仓库 `01_gemm` 指南 V4~V7 的全套技术），骨架不变。

### 7.4 流量与显存账本

按 2.7 节的口径（N=4096, d=64, fp32, Br=64）：

```
V1（融合 softmax，S/P 仍落地）:
  N² 流量 = 4 遍 × 64 MB = 256 MB

V3（FlashAttention）:
  Q 读 1 遍 + O 写 1 遍           = 2 MB
  K/V 被每个 Q 块各读一遍全量：
    (N/Br) × 2 × N×d×4B = 64 × 2 MB = 128 MB
  N² 流量                          = 0
```

约 2 倍的直接流量削减（Br 越大、d 越小、或 K/V 命中 L2 越多，优势越大），更重要的是三项**结构性收益**：

1. **显存 O(N²) → O(N)**：长序列（32K、128K）从"根本放不下"变成常规可跑，这是长上下文 LLM 的使能技术；
2. **三步计算融合进一个 kernel**：矩阵乘与 softmax 的运算交错填充流水线，实测加速常达 2~4 倍；
3. **AI 回到 compute-bound 区**：瓶颈从带宽转回算力，Tensor Core 等算力优化（V4）才有了用武之地。

理论上界（FlashAttention 论文）：HBM 访问量从 Θ(N² ) 降为 Θ(N²d²/M)（M 为片上存储大小），且证明了在此模型下是最优的。

### 7.5 causal mask 的免费午餐

因果掩码下，Q 块 i 只需要遍历 j ≤ i 的 K/V 块——右上三角的块**整块跳过**（代码中的 `break`）。计算量与 K/V 流量直接减半，且不同 Q 块的工作量从相同变为线性递增，这也是 V4 中负载均衡讨论的背景。

---

## 第 8 章 V4：FlashAttention-2——并行度与指令效率的工程改进

V3 已经解决了访存的主要矛盾，V4（FlashAttention-2, 2023）的改进全部在"把片上计算做到极致"，可归纳为四点。

### 8.1 交换循环，最大化序列维并行

FA-1 原始实现的外层循环在 **K/V 块**上（每个 Block 负责一段 K/V，内层扫 Q），这带来两个问题：不同 Block 要向同一个 O 累加（需要原子/额外同步），且并行度由 K/V 块数决定。FA-2 交换为**外层 Q 块**（即 7.3 教学版采用的方向）：

```
FA-1: 外层 K/V 块 → O 是共享的累加目标，写冲突
FA-2: 外层 Q 块   → 每个 Q 块的 (m, l, o~) 完全私有，Block 间零通信
      Grid = (N/Br) × batch × heads —— 序列维成为并行维度
```

后者在长序列、小 batch 的场景（正是长上下文推理/训练的典型形态）下能启动足够多的 Block 填满 SM，这是 FA-2 相对 FA-1 约 2 倍提速的最大来源。

### 8.2 Warp 间分工：split-K 改 split-Q

Block 内部（如 4 个 Warp）如何分担一个 tile 的计算：

```
FA-1（split-K）: 4 个 Warp 各算 S 的一段列 → 每个 Warp 持有部分和，
                 需经共享内存互相归约 —— 同步 + 读写开销
FA-2（split-Q）: 4 个 Warp 各负责 Q 块的一段行 → 各自的行独立走完
                 全流程，Warp 间零通信
```

原则与 8.1 相同：**让并行单元之间不共享可写状态**——通信是并行的敌人，能靠划分消掉就消掉。

### 8.3 减少非矩阵乘运算：延迟归一化

Tensor Core 时代矩阵乘吞吐极高，夹在中间的**逐元素运算**（exp、修正乘法、除法）反而成为新短板。FA-2 的两处削减：

- **归一化推迟到最后**：内层循环只维护未归一化的 o~（V2 的 6.3 节本就如此），不在每块后除以当前 l；
- **修正因子合并**：rescale 乘法尽量批量地作用于 o~，减少每块的标量乘次数。

这一条与 GEMM 优化中"让指令流以 FFMA/MMA 为主"的思想一脉相承——**统计非核心指令的占比，是 profile 计算受限 kernel 的通用视角**。

### 8.4 causal mask 下的负载均衡

因果掩码使 Q 块的工作量随行号线性递增（7.5 节）。若调度不当，先完成的 SM 会闲等。FA-2 依靠"Q 块数 × batch × heads 远多于 SM 数"的超额并行让硬件调度自然填坑，后续工作（如 FA-3、各类推理引擎）进一步引入块粒度的动态调度。

### 8.5 性能位置

在 A100/H100 上，FA-2 的 forward 可达理论算力的 50%~70%（fp16/bf16，Tensor Core），达到与高度调优的 GEMM 同一量级——考虑到它还内嵌了 softmax 与掩码逻辑，这已接近该算法结构的硬件上限。后续的 FlashAttention-3 进一步利用 Hopper 的 TMA 与 wgmma 异步流水，属于"双缓冲 → 硬件异步化"这条思想脉络（`01_gemm` 指南第 10 章）的延伸。

---

## 第 9 章 场景扩展：推理 Decode 阶段与 Flash-Decoding

### 9.1 Decode：每步只有一个新 query

自回归生成时，历史 token 的 K/V 缓存在显存中（**KV Cache**），每生成一个新 token 只需用**1 行 query** 对全部历史做 attention：

```
q: 1×d,  K/V: N×d（缓存）  →  o: 1×d
计算量:  4Nd FLOP          访存量: 读 K/V 共 2Nd 个元素
AI ≈ 4Nd / (2Nd × 4B) = 0.5 FLOP/B      → 极度 memory-bound
```

decode attention 的本质是**读一遍 KV Cache 的带宽测试**，优化目标变回"跑满带宽"，与 prefill（第 4~8 章的 compute-bound 优化）完全不同。

### 9.2 新问题：并行度塌缩

拿 V4 的并行方案套 decode：Q 只有 1 行 → Q 块只有 1 个 → Grid = 1 × batch × heads。小 batch 时只有几十个 Block，**上百个 SM 大面积闲置**——不是带宽不够，而是没有足够的 Block 去发起访存。

### 9.3 Flash-Decoding：沿 KV 维切分再归并

解法是把唯一能切的维度——**KV 序列维**——切给多个 Block：

```
① 把长度 N 的 KV 缓存切成 S 段，S 个 Block 并行，各自对本段
   做 flash attention，输出部分结果 (m⁽ˢ⁾, l⁽ˢ⁾, o~⁽ˢ⁾)
② 一个轻量 kernel 归并 S 份部分结果
```

归并公式正是 Online Softmax 递推的"两两合并"形式（与 6.2 节同源，可再次体会该递推的普适性）：

```
m  = max(m⁽¹⁾, m⁽²⁾)
l  = l⁽¹⁾·exp(m⁽¹⁾−m) + l⁽²⁾·exp(m⁽²⁾−m)
o~ = o~⁽¹⁾·exp(m⁽¹⁾−m) + o~⁽²⁾·exp(m⁽²⁾−m)      最后 o = o~ / l
```

这与 GEMM 中的 Split-K（K 维切给多 Block 再归约）是同一族技巧：**当天然并行维度不足时，牺牲一次归并换取并行度**。

### 9.4 相关技术一览（了解即可）

| 技术 | 一句话 |
|------|--------|
| MQA / GQA | 多个 query 头共享一组 KV 头，成倍削减 KV Cache 的容量与读流量 |
| PagedAttention (vLLM) | KV Cache 按页非连续存放，kernel 按页表寻址，消除显存碎片 |
| FA-3 / FlashMLA | Hopper TMA + 异步流水在 attention 上的延伸 |

---

## 第 10 章 工程化：PyTorch SDPA 与正确性验证

### 10.1 用好现成实现：scaled_dot_product_attention

PyTorch 2.x 已内置多后端的 attention（含 FlashAttention），日常使用**不需要手写**：

```python
import torch
import torch.nn.functional as F

q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
k, v = torch.randn_like(q), torch.randn_like(q)

o = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # 自动选后端

# 需要指定后端时（如强制 FlashAttention）：
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

三个内置后端与本文版本的对应：`MATH`（V0/V1 级的落地实现，兜底）、`EFFICIENT_ATTENTION`（memory-efficient，分块思想同 V3）、`FLASH_ATTENTION`（V3/V4）。

### 10.2 手写 kernel 的正确性验证

以 7.3 的教学版为例，与 PyTorch 参考实现对拍：

```python
import torch, math

def reference(q, k, v, causal):
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        n = q.shape[-2]
        s = s.masked_fill(torch.triu(torch.ones(n, n, device=q.device, dtype=torch.bool), 1), float('-inf'))
    return torch.softmax(s, dim=-1) @ v

o_mine = my_ext.flash_attn(q, k, v, causal=True)     # 自己的扩展
o_ref  = reference(q, k, v, causal=True)
print(torch.allclose(o_mine, o_ref, rtol=1e-3, atol=1e-3))
```

Attention 特有的验证陷阱，比 GEMM 更多：

- **必须用随机数据**：全同值输入下 softmax 输出是均匀分布，mask 错位、归一化错误等 bug 全部隐形；
- **causal 要单独测**，且用**非方阵场景**（如 N 不等于块大小的整数倍）测边界——mask 与边界的交互是 bug 高发区；
- **容差按精度定**：fp32 用 rtol≈1e-3；fp16/bf16 下在线递推与参考实现的求和顺序不同，rtol≈1e-2 属正常，不要按逐位一致排查；
- **测极端分布**：给一行注入一个特别大的分数（如 +50），验证 safe/online softmax 没有 NaN——这正是 2.5 节问题的回归测试；
- **性能计时**用 CUDA event + 预热，报告带宽（decode 场景）或 TFLOPS（prefill 场景），并与 `F.scaled_dot_product_attention` 同条件对比。

### 10.3 什么时候手写 Attention

| 场景 | 建议 |
|------|------|
| 标准 attention（训练/推理） | 直接用 SDPA / flash-attn 库，不要手写 |
| 变体：自定义 mask、bias、稀疏模式 | FlexAttention（PyTorch）/ Triton 手写 |
| 极端场景：特殊 KV 布局、量化 KV、投机解码 | 参考 vLLM/SGLang kernel 修改 |
| 学习原理 | 手写 V0~V3（本文） |

与 GEMM 一样，手写的最大价值是**获得读懂并修改 flash-attn / vLLM 这类生产 kernel 的能力**——它们的骨架就是本文的分块递推 + GEMM 篇的分块矩阵乘。

---

## 第 11 章 总结与实践建议

### 11.1 版本回顾

| 版本 | 核心手段 | 解决的瓶颈 | N² 流量（遍） |
|------|---------|-----------|------------|
| V0 | 三个独立 kernel | —（基准） | 6 |
| V1 | 融合 scale+mask+softmax，行内两级归约 | softmax 多遍扫描、不合并访存 | 4 |
| V2 | Online Softmax 递推（m, l, o~） | softmax 的全局依赖（分块的数学障碍） | 4（钥匙作用） |
| V3 | FlashAttention：K/V 分块 + 在线重缩放 | N² 中间矩阵落地、O(N²) 显存 | **0** |
| V4 | FA-2：循环交换、split-Q、延迟归一化 | 并行度、Warp 通信、非矩阵乘指令 | 0 |
| 第 9 章 | Flash-Decoding：KV 维切分 + 归并 | decode 阶段并行度塌缩 | 0 |

### 11.2 通用优化方法论

Attention 的优化过程沉淀出三条可迁移的规律：

1. **先算流量账，向最大的搬运项开刀**。N² 中间矩阵比有效数据大 30 倍，在它面前优化矩阵乘细节是徒劳——"数最大的数据过 HBM 几遍"（2.7 节）应当成为分析任何算子链的第一步；
2. **融合的障碍常在数学结构，解锁靠代数变换**。softmax 的全局依赖看似不可分块，Online Softmax 用"可修正的部分量 (m, l, o~)"将其改写为流式递推——遇到"必须先全局后局部"的算子，先找有没有等价的递推形式（同类例子：Welford 方差递推之于 LayerNorm）；
3. **memory-bound 与 compute-bound 会随场景切换，优化目标要跟着换**。同一个 attention，prefill 是 compute-bound（拼算力利用率），decode 是 memory-bound（拼带宽与并行度），长序列小 batch 还要拼调度——先判形态，再选武器。

### 11.3 学习路径建议

| 目标 | 建议 |
|------|------|
| 理解 attention 为什么慢 | 精读第 1、4 章，自己算一遍流量账 |
| 理解 FlashAttention | 精读第 6 章（数学）→ 第 7 章（实现），手推一遍 6.2 递推 |
| 动手实践 | 实现 7.3 教学版，用 10.2 的方法对拍，再用 ncu 看流量 |
| 进阶 | FlashAttention-1/2/3 论文、flash-attn 源码、Triton 教程的 fused attention |
| 生产 | PyTorch SDPA / flash-attn / vLLM，变体用 FlexAttention 或 Triton |

---

## 附录：关键概念速查

| 概念 | 含义 | 相关章节 |
|------|------|---------|
| Scaled Dot-Product Attention | softmax(QKᵀ/√d)·V，两个矩阵乘夹一个逐行 softmax | 第 1 章 |
| √d 缩放 | 点积方差 = d，除以 √d 归一为 1，防止 softmax 饱和 | 第 1 章 |
| causal mask | 位置 i 只可见 j ≤ i；置 −inf，softmax 后为 0；可整块跳过 | 第 1、7 章 |
| N² 中间矩阵 | S、P 各 N² 元素，随序列长平方增长，朴素实现的根本瓶颈 | 第 1、4 章 |
| 算子融合 | 相邻 kernel 合一，中间数据经寄存器/共享内存而非 HBM 传递 | 第 2、5 章 |
| 两级归约 | Warp 内 shuffle 蝶形归约 + Warp 间经共享内存汇总 | 第 2、5 章 |
| Safe Softmax | 全行减最大值再 exp，杜绝上溢；代价是多一遍求 max | 第 2 章 |
| Online Softmax | m/l 一遍流式递推，旧部分和乘 exp(Δm) 修正，结果精确 | 第 6 章 |
| 未归一化输出 o~ | Σ exp(sⱼ−m)·vⱼ，与 l 同规律递推，最后一除即得 O | 第 6~7 章 |
| FlashAttention | K/V 分块驻留片上 + 在线重缩放，S/P 永不落地，显存 O(N) | 第 7 章 |
| 重缩放（rescale） | 块间最大值更新时，累积量统一乘 exp(m_old − m_new) | 第 6~7 章 |
| split-Q / split-K | Block 内 Warp 分工方式；FA-2 用 split-Q 消除 Warp 间通信 | 第 8 章 |
| KV Cache | 推理时缓存历史 K/V；decode 即对其做单行 attention | 第 9 章 |
| Flash-Decoding | KV 维切分给多 Block，部分结果按 (m,l,o~) 归并 | 第 9 章 |
| MQA / GQA | 多 query 头共享 KV 头，削减 KV Cache 容量与带宽 | 第 9 章 |
| SDPA | PyTorch 的 scaled_dot_product_attention，多后端自动分发 | 第 10 章 |
| 流量分析法 | 数"最大的数据过 HBM 几遍"，定位算子链的主要矛盾 | 第 2、4 章 |
