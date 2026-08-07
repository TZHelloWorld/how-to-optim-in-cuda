# CUDA LayerNorm 算子优化指南

> 本文以 LayerNorm 为例，系统介绍**归一化类算子**在 CUDA 上的实现与优化。LayerNorm 的结构是"两个统计量归约 + 一次逐元素变换"，麻雀虽小五脏俱全：既有归约的并行技巧，又有"均值/方差怎么算才又快又稳"的算法选择，还有访存遍数的逐版削减。全文从最朴素的三遍扫描出发，沿"发现瓶颈 → 分析原理 → 针对性优化"的主线演进出 5 个版本（V0~V4），把 $x$ 的显存读取从 3 遍压到 1 遍；最后给出数值稳健的 Welford 版本与 PyTorch 对拍方法。
>
> 本文假定读者已熟悉 CUDA 基础（Grid/Block/Warp、共享内存、合并访存、warp shuffle 与两级归约），预备知识不再单独成章，着重算法思想与代码。

---

## 目录

- [第 1 章 问题定义：什么是 LayerNorm](#第-1-章-问题定义什么是-layernorm)
- [第 2 章 算法核心：计算均值与方差的三种方法](#第-2-章-算法核心计算均值与方差的三种方法)
- [第 3 章 优化路线总览](#第-3-章-优化路线总览)
- [第 4 章 V0：基准实现——两遍统计 + 树形归约](#第-4-章-v0基准实现两遍统计--树形归约)
- [第 5 章 V1：单遍化——一次归约同时得到均值与方差](#第-5-章-v1单遍化一次归约同时得到均值与方差)
- [第 6 章 V2：Warp Shuffle 两级归约](#第-6-章-v2warp-shuffle-两级归约)
- [第 7 章 V3：float4 向量化](#第-7-章-v3float4-向量化)
- [第 8 章 V4：行驻留寄存器——显存只读一遍](#第-8-章-v4行驻留寄存器显存只读一遍)
- [第 9 章 工程化：Welford 稳健版、benchmark 与 PyTorch 对拍](#第-9-章-工程化welford-稳健版benchmark-与-pytorch-对拍)
- [第 10 章 总结与速查](#第-10-章-总结与速查)

---

## 第 1 章 问题定义：什么是 LayerNorm

### 1.1 公式与语义

LayerNorm 对输入的**最后一维**（hidden 维，记长度为 $H$）做归一化。对形状 $[N, H]$ 的输入（$N$ = batch × 序列长度，即"行数"），**每一行独立**计算：

$$
\mu = \frac{1}{H}\sum_{i=1}^{H} x_i,
\qquad
\sigma^2 = \frac{1}{H}\sum_{i=1}^{H} (x_i - \mu)^2
$$

$$
y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}} \cdot \gamma_i + \beta_i
$$

- $\mu$、$\sigma^2$：本行的均值与（有偏）方差，把该行拉到"均值 0、方差 1"的标准分布；
- $\varepsilon$：防除零的小常数（典型 $10^{-5}$）；
- $\gamma, \beta \in \mathbb{R}^{H}$：可学习的缩放与平移参数（仿射变换），所有行共享。

它的作用是稳定训练：把每个 token 的激活分布拉回标准范围，缓解梯度消失/爆炸。与 BatchNorm 的本质区别在于**归一化的维度**——LN 沿 hidden 维、每行自给自足，不依赖 batch 内其他样本，因此训练/推理行为一致、天然适合变长序列，成为 Transformer 的标配。

CPU 参考实现只有几行：

```c
// 对每一行：
float mean = 0.f, var = 0.f;
for (int i = 0; i < H; i++) mean += x[i];
mean /= H;
for (int i = 0; i < H; i++) var += (x[i] - mean) * (x[i] - mean);
var /= H;
float rstd = 1.f / sqrtf(var + eps);
for (int i = 0; i < H; i++) y[i] = (x[i] - mean) * rstd * gamma[i] + beta[i];
```

三个循环对应三步：**求均值（归约）→ 求方差（归约）→ 归一化（逐元素）**。

### 1.2 计算特征与流量账本

以典型规模 $N = 4096$（batch 8 × 序列 512）、$H = 4096$、fp32 为例：

| 项目 | 量 |
|------|-----|
| 计算量 | 每元素约 7 FLOP（两次累加、减、乘、仿射等）≈ $7NH \approx 0.12$ GFLOP |
| 最少访存 | 读 $x$ 一遍 + 写 $y$ 一遍 = $2NH \times 4\,\mathrm{B} = 128$ MB（$\gamma/\beta$ 仅 $H$ 级、可忽略） |
| 算术强度 | $\approx 0.9$ FLOP/B —— **深度 memory-bound** |

结论与所有归一化/逐元素算子相同：**瓶颈在访存，优化目标是"$x$ 过显存的遍数"逼近下限（读 1 写 1），并把达成带宽推向硬件峰值**。全文的记账口径就是一个数字——**$x$ 的显存读取遍数**：

$$
\text{朴素三遍扫描} = 3 \text{ 遍读} \quad\longrightarrow\quad \text{理论下限} = 1 \text{ 遍读}
$$

### 1.3 并行化的基本形状

- **行间（$N$ 维）**：每行的 $\mu$、$\sigma^2$、$y$ 只依赖本行数据，$N$ 行完全独立——天然的 Grid 级并行，一行分给一个 Block（或一个 Warp，见 RMSNorm 场景的讨论）；
- **行内（$H$ 维）**：求 $\mu$、$\sigma^2$ 是**归约**（多个数缩成一个），需要线程协作——树形归约 / warp shuffle 两级归约；归一化阶段则是纯逐元素，线程各管各的；
- **一个隐含的依赖**：归一化需要**全行**的 $\mu$、$\sigma^2$——统计量算完之前，任何元素都写不出去。这个"先全局统计、后逐元素消费"的结构决定了 kernel 至少要过两个阶段（中间隔一次 Block 同步），也引出了第 2 章的核心问题：统计量本身怎么算最好？

---

## 第 2 章 算法核心：计算均值与方差的三种方法

同一对 $(\mu, \sigma^2)$，数学上至少有三种算法。它们的扫描遍数与数值行为不同，直接决定了 kernel 的结构——这是 LayerNorm 与普通归约最大的不同，值得先于一切 CUDA 技巧讲清楚。

### 2.1 方法一：两遍扫描（two-pass）

按定义直译：第一遍算 $\mu$，第二遍算 $\sigma^2 = \frac{1}{H}\sum (x_i - \mu)^2$。

- 优点：**数值最稳**——每一项 $(x_i-\mu)$ 都是小量，平方求和不丢精度；
- 缺点：$x$ 要**读两遍**（加上归一化第三遍）。

### 2.2 方法二：单遍 naive——$E[x^2] - \mu^2$

利用恒等式，方差可以由两个"可单遍累加"的量拼出：

$$
\sigma^2 = \frac{1}{H}\sum x_i^2 - \mu^2 = E[x^2] - (E[x])^2
$$

一遍扫描同时累加 $\sum x_i$ 与 $\sum x_i^2$ 即可——扫描遍数从 2 降到 1。但它藏着一个经典的数值陷阱：**灾难性抵消（catastrophic cancellation）**。当 $|\mu| \gg \sigma$ 时，$E[x^2]$ 与 $\mu^2$ 是两个巨大而几乎相等的数，相减后有效位数所剩无几。

用一个具体数值看清楚：设某行 $x = [10000.00,\ 10000.01,\ 10000.02,\ 10000.03]$，真实方差约 $1.25 \times 10^{-4}$。而 $x_i^2 \approx 10^8$——fp32 在 $10^8$ 量级的量化步长（ULP）是 $2^{26-23} = 8$：**单是把 $x_i^2$ 存成 fp32，误差就已达 ±4，比要求的答案大 4 个数量级**，相减后结果完全是噪声（甚至为负，开方直接 NaN）。

工程结论：

- 激活值场景（LN 的实际输入，$\mu$ 通常 $O(1)$ 量级）下，**fp32 累加 + 结果下界保护 `max(var, 0)`** 的单遍 naive 是可用的，也是很多生产 kernel 的选择；
- 但对输入分布不做假设的通用实现（如 PyTorch 官方），需要下一个方法。

### 2.3 方法三：Welford 在线算法与并行合并

Welford 算法逐元素**增量维护** $(n, \mu, M_2)$（$M_2 = \sum (x_i - \mu)^2$），每一步都只对"小量"做运算，无抵消问题：

$$
n \leftarrow n+1,\qquad
\delta = x - \mu,\qquad
\mu \leftarrow \mu + \frac{\delta}{n},\qquad
M_2 \leftarrow M_2 + \delta\,(x - \mu_{\text{new}})
$$

扫完后 $\sigma^2 = M_2 / n$。它天生是串行递推，但存在精确的**并行合并公式**（Chan et al.）——两个子集的统计量可以一步合并：

$$
n_{ab} = n_a + n_b,\qquad
\delta = \mu_b - \mu_a
$$

$$
\mu_{ab} = \mu_a + \delta \cdot \frac{n_b}{n_{ab}},
\qquad
M_{2,ab} = M_{2,a} + M_{2,b} + \delta^2 \cdot \frac{n_a\, n_b}{n_{ab}}
$$

合并满足结合律——**每线程先串行 Welford 自己名下的元素，线程间再树形/蝶形合并**，与普通求和归约同构，只是"合并操作"从加法换成上面的公式。单遍扫描 + 数值稳健兼得，代价是每步合并约 10 FLOP（对 memory-bound 的 LN 无感）。PyTorch 的 CUDA LayerNorm（ATen）与 NVIDIA Apex 用的正是它。

### 2.4 三种方法对比

| 方法 | $x$ 扫描遍数 | 数值稳健性 | 每元素代价 | 适用 |
|------|-------------|-----------|-----------|------|
| 两遍扫描 | 2 | 最好 | 最低 | 教学基准（V0） |
| 单遍 $E[x^2]-\mu^2$ | 1 | 差（$\mu \gg \sigma$ 时崩溃） | 低 | 激活值场景 + fp32 累加（V1~V4） |
| Welford + 并行合并 | 1 | 好 | 略高 | 通用/官方实现（第 9 章） |

本文 V1~V4 用单遍 naive 讲清访存优化主线（代码最短），第 9 章给出可直接替换的 Welford 版。

---

## 第 3 章 优化路线总览

每个版本针对上一版暴露的一个具体瓶颈：

```
V0: 一行一 Block；两遍统计 + 共享内存树形归约 + 归一化
 │   x 读 3 遍；两次完整的树形归约、两轮 __syncthreads 串行链
 ▼
V1: 单遍化：一遍同时累加 (Σx, Σx²)，方差 = E[x²] − μ²
 │   x 读 3 → 2 遍，归约次数 2 → 1（代价：数值风险，见 2.2 节）
 ▼
V2: 归约本身换 Warp Shuffle 两级归约：寄存器直传、少同步、省共享内存
 │   归约不再是短板；访存效率成为新短板（标量 4B 访问）
 ▼
V3: float4 向量化：一条指令搬 16B，在途字节 ×4，逼近带宽上限
 │   仍剩最后一遍冗余：统计与归一化两个阶段各读一次 x
 ▼
V4: 行驻留寄存器：加载时顺手累加统计量，归一化直接用寄存器里的数据
     x 读 2 → 1 遍 —— 达到理论下限（读 1 写 1）
```

流量账本预览（记账口径：$x$ 的显存读取遍数，见 1.2 节）：

| 版本 | x 读取遍数 | 归约方式 | 关键改动 |
|------|-----------|---------|---------|
| V0 | 3 | 共享内存树形 ×2 | 基准 |
| V1 | 2 | 共享内存树形 ×1 | $(\Sigma x, \Sigma x^2)$ 单遍 |
| V2 | 2 | Warp Shuffle 两级 | 归约提速、省共享内存 |
| V3 | 2 | Warp Shuffle 两级 | float4 向量化 |
| V4 | **1（下限）** | Warp Shuffle 两级 | 行驻留寄存器 |

---

## 第 4 章 V0：基准实现——两遍统计 + 树形归约

### 4.1 实现

最直接的映射：**一行分配一个 Block**，行内 $H$ 个元素由 `blockDim.x` 个线程跨步分摊；统计量按 2.1 节的两遍扫描计算，每遍用共享内存树形归约收拢：

```cuda
// V0：一行一 Block；两遍统计 + 共享内存树形归约
// 启动：layernorm_v0<<<N, 256, 256 * sizeof(float)>>>(x, y, gamma, beta, H, 1e-5f);
__global__ void layernorm_v0(const float* __restrict__ x, float* __restrict__ y,
                             const float* __restrict__ gamma, const float* __restrict__ beta,
                             int H, float eps) {
    extern __shared__ float sdata[];                  // blockDim.x 个 float
    const float* row = x + (size_t)blockIdx.x * H;    // 本 Block 负责的行
    float*       out = y + (size_t)blockIdx.x * H;
    int tid = threadIdx.x;

    // ---- pass 1：求均值 ----
    float acc = 0.f;
    for (int i = tid; i < H; i += blockDim.x)         // 跨步循环：warp 内地址连续（合并访存）
        acc += row[i];
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {    // 树形归约：每轮活跃线程减半
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    __shared__ float s_mean;
    if (tid == 0) s_mean = sdata[0] / H;
    __syncthreads();
    float mean = s_mean;                              // 广播给全 Block

    // ---- pass 2：求方差（再读一遍 x）----
    acc = 0.f;
    for (int i = tid; i < H; i += blockDim.x) {
        float d = row[i] - mean;
        acc += d * d;
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    __shared__ float s_rstd;
    if (tid == 0) s_rstd = rsqrtf(sdata[0] / H + eps);
    __syncthreads();
    float rstd = s_rstd;

    // ---- pass 3：归一化 + 仿射（第三遍读 x）----
    for (int i = tid; i < H; i += blockDim.x)
        out[i] = (row[i] - mean) * rstd * gamma[i] + beta[i];
}
```

### 4.2 瓶颈分析

按 1.2 节的账本口径：

| 问题 | 具体表现 |
|------|---------|
| **$x$ 读 3 遍** | pass1/2/3 各读一遍显存（行较短时 L1/L2 会吸收一部分，行长时是真实流量） |
| **两次完整树形归约** | 每次 $\log_2(\mathrm{blockDim})$ 轮 `__syncthreads()`，两次归约串行排队 |
| **树形归约本身低效** | 每轮活跃线程减半，大量线程围观；数据在共享内存反复读写 |

三个问题分别由 V1（砍一遍统计）、V2（换归约算法）解决。

---
## 第 5 章 V1：单遍化——一次归约同时得到均值与方差

### 5.1 优化思路

用 2.2 节的恒等式 $\sigma^2 = E[x^2] - \mu^2$：一遍扫描同时累加 $\Sigma x$ 与 $\Sigma x^2$，统计阶段从两遍变一遍、两次归约变一次。两个累加量打包成 `float2` 一起归约即可。

### 5.2 实现

```cuda
// V1：单遍统计——一次扫描同时累加 (Σx, Σx²)
__global__ void layernorm_v1(const float* __restrict__ x, float* __restrict__ y,
                             const float* __restrict__ gamma, const float* __restrict__ beta,
                             int H, float eps) {
    extern __shared__ float2 sdata2[];                // blockDim.x 个 float2
    const float* row = x + (size_t)blockIdx.x * H;
    float*       out = y + (size_t)blockIdx.x * H;
    int tid = threadIdx.x;

    // 一遍扫描，双累加
    float2 acc = make_float2(0.f, 0.f);               // (Σx, Σx²)
    for (int i = tid; i < H; i += blockDim.x) {
        float v = row[i];
        acc.x += v;
        acc.y += v * v;
    }
    sdata2[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {    // 树形归约（float2 一起归）
        if (tid < s) {
            sdata2[tid].x += sdata2[tid + s].x;
            sdata2[tid].y += sdata2[tid + s].y;
        }
        __syncthreads();
    }

    __shared__ float s_mean, s_rstd;
    if (tid == 0) {
        float mean = sdata2[0].x / H;
        float var  = fmaxf(sdata2[0].y / H - mean * mean, 0.f);  // 下界保护（2.2 节）
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    float mean = s_mean, rstd = s_rstd;

    for (int i = tid; i < H; i += blockDim.x)
        out[i] = (row[i] - mean) * rstd * gamma[i] + beta[i];
}
```

### 5.3 效果与遗留问题

- $x$ 读取 3 → 2 遍；`__syncthreads()` 轮数减半；
- `fmaxf(var, 0.f)` 是必须的：浮点误差可能让 $E[x^2] - \mu^2$ 轻微为负，不加保护 `rsqrtf` 直接产出 NaN；
- 数值前提要记住（2.2 节）：输入均值远大于标准差时此法失真，通用实现请换第 9 章的 Welford；
- 遗留：树形归约本身仍在共享内存里反复读写、反复同步——这是纯粹的归约算法问题，下一版解决。

---

## 第 6 章 V2：Warp Shuffle 两级归约

### 6.1 优化思路

树形归约的两个固有开销——共享内存往返、每轮全 Block 同步——用 **warp shuffle 两级归约**消除：

1. **第一级（Warp 内）**：`__shfl_xor_sync` 让 32 个 lane 在寄存器间直接交换数据，5 步蝶形完成归约，零共享内存、零 `__syncthreads`；
2. **第二级（Warp 间）**：每 Warp 的部分结果（最多 32 个）写入共享内存，由第一个 Warp 再做一次 warp 归约。

对 LayerNorm 的特殊点：归约对象是 `float2`（$\Sigma x$ 与 $\Sigma x^2$ 一对），两个分量各自 shuffle 即可。

### 6.2 实现

```cuda
// (Σx, Σx²) 成对的 warp / block 归约组件
__device__ __forceinline__ float2 warpReduceSum2(float2 v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v.x += __shfl_xor_sync(0xffffffff, v.x, offset);
        v.y += __shfl_xor_sync(0xffffffff, v.y, offset);
    }
    return v;                                          // 32 个 lane 都持有完整和
}

__device__ __forceinline__ float2 blockReduceSum2(float2 v) {
    __shared__ float2 warpRes[32];                     // 至多 32 个 Warp
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warpReduceSum2(v);                             // 第一级：Warp 内
    if (lane == 0) warpRes[wid] = v;
    __syncthreads();
    int nWarp = (blockDim.x + 31) >> 5;
    v = (lane < nWarp) ? warpRes[lane] : make_float2(0.f, 0.f);
    if (wid == 0) v = warpReduceSum2(v);               // 第二级：Warp 间
    __shared__ float2 result;
    if (threadIdx.x == 0) result = v;
    __syncthreads();                                   // 广播给全 Block
    return result;
}

// V2：归约换成两级 shuffle，主体骤然清爽
__global__ void layernorm_v2(const float* __restrict__ x, float* __restrict__ y,
                             const float* __restrict__ gamma, const float* __restrict__ beta,
                             int H, float eps) {
    const float* row = x + (size_t)blockIdx.x * H;
    float*       out = y + (size_t)blockIdx.x * H;

    float2 acc = make_float2(0.f, 0.f);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float v = row[i];
        acc.x += v;
        acc.y += v * v;
    }
    acc = blockReduceSum2(acc);                        // 一行代码完成全部归约

    float mean = acc.x / H;
    float rstd = rsqrtf(fmaxf(acc.y / H - mean * mean, 0.f) + eps);

    for (int i = threadIdx.x; i < H; i += blockDim.x)
        out[i] = (row[i] - mean) * rstd * gamma[i] + beta[i];
}
```

### 6.3 效果与遗留问题

- 归约的同步从 $\log_2(\mathrm{blockDim})$ 轮 `__syncthreads` 降到 2 轮；中间值走寄存器不走共享内存；共享内存占用从 `blockDim×8B` 降到固定 `32×8B`；
- `blockReduceSum2` 返回后**每个线程都持有结果**，均值/方差的计算不再需要"线程 0 算完再广播"；
- 遗留：归约已不是瓶颈，profile 会显示时间花在访存上——而标量 4 B 访问的在途字节数不足以打满带宽，下一版向量化。

---

## 第 7 章 V3：float4 向量化

### 7.1 优化思路

memory-bound kernel 打满带宽的标准手段：把每线程的访问宽度从 4 B 提到 16 B（`float4`，一条 `LDG.128`），单条指令的在途字节 ×4，用更少的指令喂满内存管线。要求 $H$ 是 4 的倍数（Transformer 的 hidden 维天然满足）。

### 7.2 实现

```cuda
// V3：float4 向量化（要求 H % 4 == 0，且指针 16B 对齐）
__global__ void layernorm_v3(const float* __restrict__ x, float* __restrict__ y,
                             const float* __restrict__ gamma, const float* __restrict__ beta,
                             int H, float eps) {
    const float4* row4 = reinterpret_cast<const float4*>(x + (size_t)blockIdx.x * H);
    float4*       out4 = reinterpret_cast<float4*>(y + (size_t)blockIdx.x * H);
    const float4* g4   = reinterpret_cast<const float4*>(gamma);
    const float4* b4   = reinterpret_cast<const float4*>(beta);
    int H4 = H >> 2;

    float2 acc = make_float2(0.f, 0.f);
    for (int i = threadIdx.x; i < H4; i += blockDim.x) {
        float4 v = row4[i];                                    // 一条指令搬 16B
        acc.x += v.x + v.y + v.z + v.w;
        acc.y += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    acc = blockReduceSum2(acc);

    float mean = acc.x / H;
    float rstd = rsqrtf(fmaxf(acc.y / H - mean * mean, 0.f) + eps);

    for (int i = threadIdx.x; i < H4; i += blockDim.x) {
        float4 v = row4[i], g = g4[i], b = b4[i], o;
        o.x = (v.x - mean) * rstd * g.x + b.x;
        o.y = (v.y - mean) * rstd * g.y + b.y;
        o.z = (v.z - mean) * rstd * g.z + b.z;
        o.w = (v.w - mean) * rstd * g.w + b.w;
        out4[i] = o;                                           // 一条指令写 16B
    }
}
```

### 7.3 效果与遗留问题

访存指令数减为 1/4，达成带宽显著上升（具体收益依卡而定，建议用第 9 章的 benchmark 实测）。遗留最后一个结构性冗余：**统计阶段读了一遍 $x$，归一化阶段又读了一遍**——同一份数据进 SM 两次。能不能只读一次？

---

## 第 8 章 V4：行驻留寄存器——显存只读一遍

### 8.1 优化思路

归一化阶段用的每个元素，恰好就是统计阶段**同一个线程**读过的那些（跨步映射相同）。既然如此，第一遍读的时候就把数据留在**线程私有的寄存器数组**里；统计量归约完成后，直接从寄存器取数归一化——第二遍显存读取整个消失，$x$ 的读取达到理论下限 1 遍。

要求每线程负责的元素数是编译期常量（寄存器数组必须静态索引），因此把 `ITEMS = H / blockDim.x` 做成模板参数、循环全展开。

### 8.2 实现

```cuda
// V4：行驻留寄存器——加载时顺手累加，归一化直接用寄存器数据
// 要求 H = ITEMS * blockDim.x（H=4096, block=256 → ITEMS=16）
// 启动：layernorm_v4<16><<<N, 256>>>(x, y, gamma, beta, H, eps);
template <int ITEMS>
__global__ void layernorm_v4(const float* __restrict__ x, float* __restrict__ y,
                             const float* __restrict__ gamma, const float* __restrict__ beta,
                             int H, float eps) {
    const float* row = x + (size_t)blockIdx.x * H;
    float*       out = y + (size_t)blockIdx.x * H;

    float buf[ITEMS];                                  // 本线程名下的元素，驻留寄存器
    float2 acc = make_float2(0.f, 0.f);
    #pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        int i = threadIdx.x + k * blockDim.x;          // 同一轮各线程地址连续 → 合并访存
        buf[k] = row[i];                               // 唯一一次显存读
        acc.x += buf[k];
        acc.y += buf[k] * buf[k];                      // 加载时顺手累加统计量
    }
    acc = blockReduceSum2(acc);

    float mean = acc.x / H;
    float rstd = rsqrtf(fmaxf(acc.y / H - mean * mean, 0.f) + eps);

    #pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        int i = threadIdx.x + k * blockDim.x;
        out[i] = (buf[k] - mean) * rstd * gamma[i] + beta[i];  // 数据来自寄存器，不再读 x
    }
}
```

（向量化版本同理：`buf` 换成 `float4 buf[ITEMS/4]`，加载/写出用 `float4`——两项优化正交，可叠加。）

### 8.3 效果、代价与适用边界

- $x$ 显存读取 2 → 1 遍，加上写 $y$ 一遍，**流量达到 1.2 节的理论下限**；
- 代价是**寄存器压力**：每线程 `ITEMS` 个 float（H=4096、256 线程时 16 个，尚可）。$H$ 很大或想用更小的 Block 时，寄存器装不下会溢出到 local memory（反而变慢），此时退而求其次：
  - 改存**共享内存**（容量更大，但要占用 smem、多两次同步）；
  - 或干脆回退 V3 的两遍读（第二遍大概率命中 L2，实际代价低于纸面）；
- 生产实现（如 PyTorch ATen、Apex）正是按 $H$ 的大小在"寄存器驻留 / 共享内存驻留 / 两遍读"之间分派——**没有普适最优，只有按规模分派**。

---

## 第 9 章 工程化：Welford 稳健版、benchmark 与 PyTorch 对拍

### 9.1 Welford 并行版 kernel

把 2.3 节的公式落成代码，可直接替换 V2~V4 中的"$(\Sigma x, \Sigma x^2)$ + blockReduceSum2"组合：

```cuda
// Welford 状态：(mean, m2, n)。每线程先串行 update，线程间再蝶形 merge
__device__ __forceinline__ void welfordUpdate(float v, float& mean, float& m2, float& n) {
    n += 1.f;
    float d = v - mean;
    mean += d / n;
    m2 += d * (v - mean);                              // 注意用更新后的 mean
}

__device__ __forceinline__ void welfordMerge(float& mean, float& m2, float& n,
                                             float mean_b, float m2_b, float n_b) {
    float n_ab = n + n_b;
    if (n_ab == 0.f) return;
    float d = mean_b - mean;                           // 2.3 节的并行合并公式
    mean += d * (n_b / n_ab);
    m2   += m2_b + d * d * (n * n_b / n_ab);
    n = n_ab;
}

__device__ __forceinline__ void warpWelford(float& mean, float& m2, float& n) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {  // 蝶形：合并操作换成 welfordMerge
        float mb = __shfl_xor_sync(0xffffffff, mean, offset);
        float sb = __shfl_xor_sync(0xffffffff, m2,   offset);
        float nb = __shfl_xor_sync(0xffffffff, n,    offset);
        welfordMerge(mean, m2, n, mb, sb, nb);
    }
}

// Block 级：结构与 blockReduceSum2 相同（各 Warp → 共享内存 → Warp 0 再合并），
// 只是元素从 float2 换成 (mean, m2, n) 三元组，合并用 welfordMerge。
```

kernel 主体中：线程内循环 `welfordUpdate(row[i], mean, m2, n)`，归约后 `var = m2 / n`。归约结构一个字不改——这正是"满足结合律的合并"这一抽象的威力：**换统计算法 = 换合并算子，归约骨架复用**。

### 9.2 benchmark 骨架与正确性校验

```cuda
// 校验 + 计时骨架（各版本 kernel 同一套）
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

int main() {
    const int N = 4096, H = 4096;
    const size_t sz = (size_t)N * H;
    float *hx = new float[sz], *hy = new float[sz], *href = new float[sz];
    float *hg = new float[H], *hb = new float[H];
    for (size_t i = 0; i < sz; i++) hx[i] = (rand() % 2000 - 1000) / 500.f;   // 必须随机！
    for (int i = 0; i < H; i++) { hg[i] = 1.f + i % 3 * 0.1f; hb[i] = i % 5 * 0.01f; }

    // CPU 参考（double 累加当金标准）
    for (int r = 0; r < N; r++) {
        double s = 0, ss = 0;
        for (int i = 0; i < H; i++) s += hx[(size_t)r * H + i];
        double mean = s / H;
        for (int i = 0; i < H; i++) { double d = hx[(size_t)r * H + i] - mean; ss += d * d; }
        double rstd = 1.0 / sqrt(ss / H + 1e-5);
        for (int i = 0; i < H; i++)
            href[(size_t)r * H + i] = (float)((hx[(size_t)r * H + i] - mean) * rstd * hg[i] + hb[i]);
    }

    float *x, *y, *g, *b;
    cudaMalloc(&x, sz * 4); cudaMalloc(&y, sz * 4);
    cudaMalloc(&g, H * 4);  cudaMalloc(&b, H * 4);
    cudaMemcpy(x, hx, sz * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(g, hg, H * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(b, hb, H * 4, cudaMemcpyHostToDevice);

    // -------- 以 V4 为例：校验 + 计时 --------
    layernorm_v4<16><<<N, 256>>>(x, y, g, b, H, 1e-5f);
    cudaMemcpy(hy, y, sz * 4, cudaMemcpyDeviceToHost);
    float max_err = 0.f;
    for (size_t i = 0; i < sz; i++) max_err = fmaxf(max_err, fabsf(hy[i] - href[i]));
    printf("max_err = %.3e\n", max_err);               // fp32 预期 ~1e-5 量级

    cudaEvent_t beg, end; cudaEventCreate(&beg); cudaEventCreate(&end);
    cudaEventRecord(beg);
    for (int it = 0; it < 100; it++) layernorm_v4<16><<<N, 256>>>(x, y, g, b, H, 1e-5f);
    cudaEventRecord(end); cudaEventSynchronize(end);
    float ms; cudaEventElapsedTime(&ms, beg, end); ms /= 100;
    printf("%.3f ms  %.1f GB/s\n", ms, 2.0 * sz * 4 / ms / 1e6);  // 有效带宽 = (读+写)/t
    return 0;
}
```

两条校验要点：

- **必须用随机数据**：全同值输入下均值恰好等于元素、方差为 0，统计量算错也测不出来；
- **达成带宽是横向比较的唯一指标**：所有版本都按"最少流量 $2NH \times 4$B / 时间"计——V0 因多读而虚低，恰好反映其劣势；最优版本应逼近拷贝算子实测出的带宽水位。

### 9.3 PyTorch 对拍

```python
import torch

N, H = 4096, 4096
x = torch.randn(N, H, device='cuda')
ln = torch.nn.LayerNorm(H, eps=1e-5).cuda()
with torch.no_grad():
    ln.weight.copy_(torch.rand(H) + 0.5)     # 非平凡的 γ/β，避免仿射错误被 1/0 掩盖
    ln.bias.copy_(torch.rand(H) - 0.5)

y_ref = ln(x)
# y_mine = my_ext.layernorm(x, ln.weight, ln.bias, 1e-5)   # 自己的扩展
# print(torch.allclose(y_mine, y_ref, rtol=1e-4, atol=1e-4))
# print((y_mine - y_ref).abs().max().item())               # fp32 预期 1e-6~1e-5 量级
```

工程注意：

- **fp16/bf16 输入务必 fp32 累加**：IO 用半精度、统计量与中间计算用 fp32（`__half2float` 后累加），否则 $\Sigma x^2$ 在长行上迅速丢精度；
- 对拍容差按精度定：fp32 用 `rtol=1e-4`；fp16 IO 下 `rtol=1e-2` 属正常；
- 极端分布回归测试：造一行 $\mu \gg \sigma$ 的输入（如全体加 1e4），单遍 naive 版会暴露 2.2 节的抵消问题，Welford 版应稳定通过。

---

## 第 10 章 总结与速查

### 10.1 版本回顾

| 版本 | 核心手段 | 解决的瓶颈 | x 读取遍数 |
|------|---------|-----------|-----------|
| V0 | 两遍统计 + 树形归约 | —（基准） | 3 |
| V1 | $(\Sigma x, \Sigma x^2)$ 单遍 | 统计多扫一遍 | 2 |
| V2 | Warp Shuffle 两级归约 | 归约的同步与共享内存往返 | 2 |
| V3 | float4 向量化 | 标量访问喂不满带宽 | 2 |
| V4 | 行驻留寄存器 | 统计/归一化两阶段重复读 | **1（下限）** |
| Welford 版 | 换合并算子，骨架不变 | 单遍 naive 的数值风险 | 1~2 |

### 10.2 可迁移的经验

1. **归一化算子的通用形状**："全局统计（归约）→ 逐元素消费"，两阶段之间隔一次同步；优化主线是数"输入过显存几遍"；
2. **算法选择先于 CUDA 技巧**：均值/方差的三种算法（两遍/naive 单遍/Welford）决定了扫描遍数与数值行为，比任何访存优化的影响都靠前；
3. **归约骨架与合并算子解耦**：sum、(sum, sumsq)、Welford 三元组共用同一套两级归约结构，只换合并操作——写一次骨架，处处复用；
4. **"加载时顺手做事 + 数据驻留"**是消灭重复读的两板斧：统计在加载时累加、消费阶段用寄存器/共享内存里的副本；
5. **按规模分派**：寄存器驻留、共享内存驻留、两遍读各有适用区间，生产 kernel 用 $H$ 的大小选择路径。

### 10.3 关键概念速查

| 概念 | 含义 | 出处 |
|------|------|------|
| LayerNorm | 沿 hidden 维逐行归一化 + 仿射，$y=(x-\mu)/\sqrt{\sigma^2+\varepsilon}\cdot\gamma+\beta$ | 1.1 节 |
| 灾难性抵消 | $E[x^2]-\mu^2$ 两大数相减丢失有效位，$\mu\gg\sigma$ 时崩溃 | 2.2 节 |
| Welford 算法 | 增量维护 $(n,\mu,M_2)$，数值稳健的单遍方差 | 2.3 节 |
| 并行合并公式 | 两个子集统计量一步合并（Chan et al.），满足结合律 | 2.3 节 |
| 两级归约 | Warp 内 shuffle + Warp 间共享内存汇总 | 第 6 章 |
| 行驻留 | 数据加载后留在寄存器/共享内存，消费阶段不再读显存 | 第 8 章 |
| 有效带宽 | $(读+写最少字节)/时间$，归一化算子的横向比较指标 | 9.2 节 |
