# CUDA RMSNorm 算子优化指南

> 本文以 RMSNorm 为例，介绍这个 LLaMA/Qwen/DeepSeek 等主流大模型标配归一化算子的 CUDA 实现与优化。RMSNorm 是 LayerNorm 的精简变体——去掉均值中心化、只保留 RMS 缩放，统计量从两个减到一个，是归一化家族里结构最干净的成员，也因此成为**算子融合**最常见的载体（residual + RMSNorm 融合是推理框架的标准操作）。全文从朴素实现出发演进出 5 个版本（V0~V4），最后落到生产框架中真实存在的融合 kernel 形态。
>
> 本文假定读者已熟悉 CUDA 基础（Grid/Block/Warp、共享内存、合并访存、warp shuffle 与两级归约），预备知识不再单独成章，着重算法思想与代码。

---

## 目录

- [第 1 章 问题定义：什么是 RMSNorm](#第-1-章-问题定义什么是-rmsnorm)
- [第 2 章 与 LayerNorm 的对比：少了什么、为什么可以少](#第-2-章-与-layernorm-的对比少了什么为什么可以少)
- [第 3 章 优化路线总览](#第-3-章-优化路线总览)
- [第 4 章 V0：基准实现——一行一 Block 树形归约](#第-4-章-v0基准实现一行一-block-树形归约)
- [第 5 章 V1：Warp Shuffle 两级归约](#第-5-章-v1warp-shuffle-两级归约)
- [第 6 章 V2：float4 向量化 + 行驻留寄存器](#第-6-章-v2float4-向量化--行驻留寄存器)
- [第 7 章 V3：行级调度——短行一 Warp](#第-7-章-v3行级调度短行一-warp)
- [第 8 章 V4：算子融合——residual + RMSNorm](#第-8-章-v4算子融合residual--rmsnorm)
- [第 9 章 工程化：混合精度、变体与 PyTorch 对拍](#第-9-章-工程化混合精度变体与-pytorch-对拍)
- [第 10 章 总结与速查](#第-10-章-总结与速查)

---

## 第 1 章 问题定义：什么是 RMSNorm

### 1.1 公式与语义

RMSNorm（Root Mean Square Normalization，Zhang & Sennrich, 2019）对输入的最后一维（hidden 维，长度 $H$）做归一化。对形状 $[N, H]$ 的输入，**每一行独立**计算：

$$
\mathrm{rms}(x) = \sqrt{\frac{1}{H}\sum_{i=1}^{H} x_i^2 + \varepsilon}
\qquad\Longrightarrow\qquad
y_i = \frac{x_i}{\mathrm{rms}(x)} \cdot \gamma_i
$$

与 LayerNorm 相比它做了两处减法：**不减均值**（不做中心化）、**没有平移参数 $\beta$**。只剩一个统计量（平方和）、一个可学习参数（$\gamma$）。

CPU 参考实现：

```c
// 对每一行：
float ss = 0.f;
for (int i = 0; i < H; i++) ss += x[i] * x[i];        // ① 平方和（归约）
float rrms = 1.f / sqrtf(ss / H + eps);               // ② 一次标量运算
for (int i = 0; i < H; i++) y[i] = x[i] * rrms * gamma[i];   // ③ 缩放（逐元素）
```

结构是"**一次归约 + 一次逐元素**"——比 LayerNorm 少一个统计量，是归一化算子里最简单的形状。

### 1.2 计算特征与流量账本

以典型规模 $N = 4096$、$H = 4096$、fp32 为例：

| 项目 | 量 |
|------|-----|
| 计算量 | 每元素约 4 FLOP（平方累加 2 + 缩放 2）≈ $4NH \approx 67$ MFLOP |
| 最少访存 | 读 $x$ 一遍 + 写 $y$ 一遍 = $2NH \times 4\,\mathrm{B} = 128$ MB（$\gamma$ 仅 $H$ 级，可忽略） |
| 算术强度 | $\approx 0.5$ FLOP/B —— 深度 memory-bound |

优化目标与所有归一化算子一致：**让 $x$ 过显存的遍数逼近下限（读 1 写 1），达成带宽逼近硬件峰值**。全文记账口径即"$x$ 的显存读取遍数"。

### 1.3 并行化的基本形状

- **行间**：$N$ 行完全独立 → Grid 级并行，一行一个 Block（或一个 Warp，第 7 章）；
- **行内**：平方和是归约（线程协作），缩放是逐元素（各管各的）；
- **依赖结构**：缩放需要**全行**的 $\mathrm{rms}$——归约完成前一个元素都写不出去，kernel 必然分"归约 → 同步 → 消费"两个阶段。这个形状与 softmax、LayerNorm 完全同族。

---

## 第 2 章 与 LayerNorm 的对比：少了什么、为什么可以少

### 2.1 逐项对比

| | LayerNorm | RMSNorm |
|--|-----------|---------|
| 公式 | $\dfrac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}\cdot\gamma+\beta$ | $\dfrac{x}{\sqrt{E[x^2]+\varepsilon}}\cdot\gamma$ |
| 统计量 | 2 个（$\mu$、$\sigma^2$） | **1 个**（$E[x^2]$） |
| 归约内容 | $(\Sigma x,\ \Sigma x^2)$ 成对归约 | $\Sigma x^2$ 单量归约 |
| 可学习参数 | $\gamma, \beta$（$2H$ 个） | $\gamma$（$H$ 个） |
| 数值陷阱 | 单遍方差有灾难性抵消风险（$E[x^2]-\mu^2$） | **无**——平方和本身就是要的量，不存在大数相减 |
| 每元素 FLOP | ~7 | ~4 |
| 代表模型 | BERT、GPT-2、ViT | T5、LLaMA、Qwen、DeepSeek、Mistral |

### 2.2 为什么可以去掉均值

Zhang & Sennrich 的实验结论：LayerNorm 的收益主要来自**缩放不变性**（re-scaling invariance，把激活拉回统一量级），而**中心化**（re-centering，减均值）的贡献很小——把它去掉，模型质量几乎不变，计算却明显省了。此外 Transformer 中 LN 前面往往紧跟线性层/残差和，激活分布本身接近零均值，中心化更显多余。

对 kernel 工程师，这个"减法"带来三重红利：

1. **归约减半**：单量归约比 $(\Sigma x, \Sigma x^2)$ 成对归约更快更省寄存器；
2. **数值陷阱消失**：不存在"两个大数相减"，fp32 累加即可，无需 Welford 这类补救；
3. **融合更顺手**：结构简单，往 residual add、量化等相邻操作上融合的代码量小——这正是第 8 章的主题。

---

## 第 3 章 优化路线总览

每个版本针对上一版暴露的一个具体瓶颈：

```
V0: 一行一 Block；共享内存树形归约 Σx²，再逐元素缩放
 │   瓶颈：树形归约反复过共享内存、反复同步；x 读 2 遍
 ▼
V1: 归约换 Warp Shuffle 两级归约：寄存器直传、2 次同步
 │   瓶颈：标量 4B 访问喂不满带宽；两阶段重复读 x
 ▼
V2: float4 向量化 + 行驻留寄存器：加载时顺手累加，缩放用寄存器副本
 │   x 读 2 → 1 遍（理论下限）；带宽逼近峰值
 │   瓶颈：decode 场景行数少（N 小），一行一 Block 喂不饱 SM——
 │        调度粒度与负载形状不匹配
 ▼
V3: 行级调度：短行改一行一 Warp，一个 Block 装 8 行，并行度 ×8
 │   单算子已到头；放眼算子链：RMSNorm 前面总跟着 residual add
 ▼
V4: 融合 residual + RMSNorm 为单 kernel：省一遍中间结果的读写
     （vLLM / SGLang 的 fused_add_rms_norm 即此形态）
```

流量账本预览（记账口径：$x$ 的显存读取遍数）：

| 版本 | x 读取遍数 | 关键改动 |
|------|-----------|---------|
| V0 | 2 | 基准：树形归约 |
| V1 | 2 | Warp Shuffle 两级归约 |
| V2 | **1（单算子下限）** | 向量化 + 行驻留 |
| V3 | 1 | 短行一 Warp（并行度适配） |
| V4 | 融合视角：算子链总流量 5 遍 → 4 遍 | residual 融合 |

---

## 第 4 章 V0：基准实现——一行一 Block 树形归约

### 4.1 实现

```cuda
// V0：一行一 Block；共享内存树形归约
// 启动：rmsnorm_v0<<<N, 256, 256 * sizeof(float)>>>(x, y, gamma, H, 1e-6f);
__global__ void rmsnorm_v0(const float* __restrict__ x, float* __restrict__ y,
                           const float* __restrict__ gamma, int H, float eps) {
    extern __shared__ float sdata[];                  // blockDim.x 个 float
    const float* row = x + (size_t)blockIdx.x * H;    // 本 Block 负责的行
    float*       out = y + (size_t)blockIdx.x * H;
    int tid = threadIdx.x;

    // ① 每线程跨步累加平方和
    float acc = 0.f;
    for (int i = tid; i < H; i += blockDim.x)         // 跨步循环：warp 内地址连续（合并访存）
        acc += row[i] * row[i];

    // ② Block 内树形归约
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {    // 每轮活跃线程减半
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // ③ 线程 0 算 1/rms，经共享内存广播
    __shared__ float s_rrms;
    if (tid == 0) s_rrms = rsqrtf(sdata[0] / H + eps);
    __syncthreads();
    float rrms = s_rrms;

    // ④ 逐元素缩放（第二遍读 x）
    for (int i = tid; i < H; i += blockDim.x)
        out[i] = row[i] * rrms * gamma[i];
}
```

### 4.2 瓶颈分析

| 问题 | 具体表现 |
|------|---------|
| 树形归约低效 | $\log_2(\mathrm{blockDim})$ 轮循环，每轮活跃线程减半、其余围观；部分和在共享内存反复读写；每轮一次 `__syncthreads()` |
| $x$ 读 2 遍 | 归约阶段读一遍，缩放阶段再读一遍（行短时 L1/L2 可吸收，行长时是真实流量） |
| 标量访问 | 每线程每次只搬 4 B，在途字节数不足以喂满 HBM 管线 |

三个问题依次由 V1、V2 解决。

---
## 第 5 章 V1：Warp Shuffle 两级归约

### 5.1 优化思路

树形归约的共享内存往返与逐轮同步，用两级归约消除：Warp 内 32 个 lane 经 `__shfl_xor_sync` 在寄存器间蝶形归约（5 步、零同步）；Warp 间只剩至多 32 个部分和，写入共享内存后由 Warp 0 再归约一次。全程只需 2 次 `__syncthreads()`。

### 5.2 实现

```cuda
__device__ __forceinline__ float warpReduceSum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);  // 蝶形：5 步，寄存器直传
    return v;                                         // 32 个 lane 都持有完整和
}

__device__ __forceinline__ float blockReduceSum(float v) {
    __shared__ float warpRes[32];                     // 至多 32 个 Warp 的部分和
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warpReduceSum(v);                             // 第一级：Warp 内
    if (lane == 0) warpRes[wid] = v;
    __syncthreads();
    int nWarp = (blockDim.x + 31) >> 5;
    v = (lane < nWarp) ? warpRes[lane] : 0.f;
    if (wid == 0) v = warpReduceSum(v);               // 第二级：Warp 间
    __shared__ float result;
    if (threadIdx.x == 0) result = v;
    __syncthreads();                                  // 广播给全 Block
    return result;
}

// V1：主体归约一行搞定
__global__ void rmsnorm_v1(const float* __restrict__ x, float* __restrict__ y,
                           const float* __restrict__ gamma, int H, float eps) {
    const float* row = x + (size_t)blockIdx.x * H;
    float*       out = y + (size_t)blockIdx.x * H;

    float acc = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        acc += row[i] * row[i];
    acc = blockReduceSum(acc);                        // 所有线程都拿到 Σx²

    float rrms = rsqrtf(acc / H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        out[i] = row[i] * rrms * gamma[i];
}
```

### 5.3 效果与遗留问题

同步从 $\log_2(\mathrm{blockDim})$ 轮降到 2 轮；归约中间值全程走寄存器。此后归约不再是瓶颈，profile 会显示时间花在访存上——两处访存低效留给 V2：标量 4 B 访问、$x$ 两阶段重复读。

---

## 第 6 章 V2：float4 向量化 + 行驻留寄存器

### 6.1 优化思路

两项正交的访存优化一次做完：

1. **float4 向量化**：一条 `LDG.128` 搬 16 B，单指令在途字节 ×4，喂满内存管线（要求 $H \bmod 4 = 0$）；
2. **行驻留寄存器**：缩放阶段用的元素恰好是归约阶段**同一线程**读过的那些——第一遍读进来就存入线程私有的寄存器数组，缩放时直接取用，第二遍显存读取整个消失。$x$ 读取达到单算子理论下限 **1 遍**。

行驻留要求每线程元素数是编译期常量（寄存器数组需静态索引），因此 `ITEMS = H / (blockDim.x × 4)` 做成模板参数。

### 6.2 实现

```cuda
// V2：float4 向量化 + 行驻留寄存器
// 要求 H = ITEMS * blockDim.x * 4（H=4096, block=256 → ITEMS=4）
// 启动：rmsnorm_v2<4><<<N, 256>>>(x, y, gamma, H, eps);
template <int ITEMS>
__global__ void rmsnorm_v2(const float* __restrict__ x, float* __restrict__ y,
                           const float* __restrict__ gamma, int H, float eps) {
    const float4* row4 = reinterpret_cast<const float4*>(x + (size_t)blockIdx.x * H);
    float4*       out4 = reinterpret_cast<float4*>(y + (size_t)blockIdx.x * H);
    const float4* g4   = reinterpret_cast<const float4*>(gamma);

    float4 buf[ITEMS];                                // 本线程名下的数据，驻留寄存器
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        int i = threadIdx.x + k * blockDim.x;         // 同一轮各线程地址连续 → 合并访存
        float4 v = row4[i];                           // 唯一一次显存读（16B/指令）
        buf[k] = v;
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;   // 加载时顺手累加
    }
    acc = blockReduceSum(acc);
    float rrms = rsqrtf(acc / H + eps);

    #pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        int i = threadIdx.x + k * blockDim.x;
        float4 v = buf[k], g = g4[i], o;              // 数据来自寄存器，不再读 x
        o.x = v.x * rrms * g.x;
        o.y = v.y * rrms * g.y;
        o.z = v.z * rrms * g.z;
        o.w = v.w * rrms * g.w;
        out4[i] = o;
    }
}
```

### 6.3 效果、代价与适用边界

- $x$ 读 1 遍 + 写 $y$ 1 遍 = 流量下限；向量化后达成带宽应逼近本卡拷贝算子的实测水位；
- 代价是寄存器压力：每线程 `ITEMS` 个 float4（$H{=}4096$、256 线程时 4 个 = 16 个 float，轻松）。$H$ 特别大时寄存器装不下会溢出到 local memory，此时回退"两遍读"（第二遍多半命中 L2）或改共享内存驻留——按 $H$ 分派；
- 训练场景（$N$ 大，几千上万行）到此已基本到头。**但推理 decode 场景暴露出新问题**：$N$ = batch size，可能只有几十——几十个 Block 铺在上百个 SM 上，大片硬件闲置。这不是访存问题，是**调度粒度**问题，下一版解决。

---

## 第 7 章 V3：行级调度——短行一 Warp

### 7.1 优化思路

一行一个 Block 的隐含假设是"行足够长、行数足够多"。行不长（$H$ 几千）而行数又少时，更细的分配是**一行一个 Warp**：一个 256 线程的 Block 装 8 行，Block 数不变的情况下并行粒度细 8 倍；且行内归约只需 Warp 内 shuffle——**零共享内存、零 `__syncthreads`**。

### 7.2 实现

```cuda
// V3：一行一 Warp（每 Block 处理 blockDim.x/32 行）
// 要求 H = ITEMS * 32 * 4（H=4096 → ITEMS=32）
// 启动：行数 N，每 Block 8 行：rmsnorm_v3<32><<<(N+7)/8, 256>>>(x, y, gamma, N, H, eps);
template <int ITEMS>
__global__ void rmsnorm_v3(const float* __restrict__ x, float* __restrict__ y,
                           const float* __restrict__ gamma, int N, int H, float eps) {
    int lane = threadIdx.x & 31;
    int row_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // 全局 Warp 编号 = 行号
    if (row_id >= N) return;

    const float4* row4 = reinterpret_cast<const float4*>(x + (size_t)row_id * H);
    float4*       out4 = reinterpret_cast<float4*>(y + (size_t)row_id * H);
    const float4* g4   = reinterpret_cast<const float4*>(gamma);

    float4 buf[ITEMS];
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        int i = lane + k * 32;                        // Warp 内 32 个 lane 地址连续
        float4 v = row4[i];
        buf[k] = v;
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    acc = warpReduceSum(acc);                         // 只需 Warp 内归约：零同步
    float rrms = rsqrtf(acc / H + eps);

    #pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        int i = lane + k * 32;
        float4 v = buf[k], g = g4[i], o;
        o.x = v.x * rrms * g.x;  o.y = v.y * rrms * g.y;
        o.z = v.z * rrms * g.z;  o.w = v.w * rrms * g.w;
        out4[i] = o;
    }
}
```

### 7.3 效果与取舍

- decode 场景（$N$ 小）并行度 ×8，SM 利用率显著回升；归约路径更短（5 步 shuffle，无 Block 同步）；
- 代价：每 Warp 扛整行，`ITEMS` 变大（此例 32 个 float4 = 128 寄存器/线程量级——已经偏高，`ITEMS` 再大需拆分或回退）；
- 生产实现的常见做法正是**按 $(N, H)$ 分派**：行长且行多 → 一行一 Block；行短或行少 → 一行一 Warp；超长行 → 两遍读回退。没有单一最优 kernel，只有调度策略。

---

## 第 8 章 V4：算子融合——residual + RMSNorm

### 8.1 优化思路：看算子链，不看单算子

Transformer 层里 RMSNorm 从不单独出现，它的标准上下文是**残差流**：

```
h = x + residual        # 残差相加，h 还要作为新的 residual 传给下一层
y = RMSNorm(h) * γ      # 归一化结果进入下一个子层
```

分开两个 kernel 的显存流量（按行、单位 $H$）：

| kernel | 读 | 写 |
|--------|----|----|
| add | $x$、$residual$ | $h$ |
| rmsnorm | $h$ | $y$ |
| **合计** | **3** | **2** |

融合成一个 kernel：$h$ 在寄存器里随算随用，**只写不再读**：

| kernel | 读 | 写 |
|--------|----|----|
| fused_add_rmsnorm | $x$、$residual$ | $h$（新残差，必须落地）、$y$ |
| **合计** | **2** | **2** |

流量 5 遍 → 4 遍（省 20%），外加省一次 kernel 启动。这正是 vLLM / SGLang 中 `fused_add_rms_norm` 的形态——推理框架的标准算子。

### 8.2 实现

```cuda
// V4：residual + RMSNorm 融合（vLLM/SGLang fused_add_rms_norm 的教学版）
// x:        [N, H] 输入，kernel 结束后被覆写为归一化输出 y（原地）
// residual: [N, H] 旧残差，kernel 结束后被覆写为新残差 h = x + residual
// 要求 H = ITEMS * blockDim.x * 4
template <int ITEMS>
__global__ void fused_add_rmsnorm(float* __restrict__ x, float* __restrict__ residual,
                                  const float* __restrict__ gamma, int H, float eps) {
    float4* xrow = reinterpret_cast<float4*>(x + (size_t)blockIdx.x * H);
    float4* rrow = reinterpret_cast<float4*>(residual + (size_t)blockIdx.x * H);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);

    float4 buf[ITEMS];
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        int i = threadIdx.x + k * blockDim.x;
        float4 a = xrow[i], b = rrow[i];
        float4 h;                                     // ① h = x + residual
        h.x = a.x + b.x;  h.y = a.y + b.y;
        h.z = a.z + b.z;  h.w = a.w + b.w;
        rrow[i] = h;                                  // ② 新残差写回（下一层要用）
        buf[k] = h;                                   //    同时驻留寄存器
        acc += h.x * h.x + h.y * h.y + h.z * h.z + h.w * h.w;
    }
    acc = blockReduceSum(acc);
    float rrms = rsqrtf(acc / H + eps);

    #pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        int i = threadIdx.x + k * blockDim.x;
        float4 h = buf[k], g = g4[i], o;              // ③ 归一化，h 来自寄存器
        o.x = h.x * rrms * g.x;  o.y = h.y * rrms * g.y;
        o.z = h.z * rrms * g.z;  o.w = h.w * rrms * g.w;
        xrow[i] = o;                                  // ④ 结果原地写回 x
    }
}
```

三个值得注意的设计点：

- **原地（in-place）语义**：`x` 与 `residual` 既是输入也是输出——省两块中间缓冲，也是 vLLM/SGLang 接口的真实约定；
- **加法藏进加载路径**：残差相加发生在数据"反正要读进来"的那一刻，零额外访存——与"逐元素计算搭归约的车"是同一个原理；
- **可继续外延**：同一骨架上还能继续融合量化（写出 int8/fp8 的 $y$ + scale，即 SGLang 的 fused rmsnorm-quant 系列）、或把 $\gamma$ 乘进后续 GEMM 的权重——融合的边界只取决于下游算子的形态。

---

## 第 9 章 工程化：混合精度、变体与 PyTorch 对拍

### 9.1 fp16/bf16：IO 半精度、累加全精度

生产中激活是 fp16/bf16。正确姿势是**读写用半精度、平方和用 fp32 累加**——半精度直接累加长行的 $\Sigma x^2$ 会迅速丢精度甚至上溢（fp16 最大 65504）：

```cuda
// 半精度 IO 的累加模式（以 half2 为例，其余结构与 V2 相同）
__half2 v2 = row_h2[i];                    // 读：半精度
float2 vf = __half22float2(v2);            // 升精度
acc += vf.x * vf.x + vf.y * vf.y;          // 累加：fp32
...
out_h2[i] = __float22half2_rn(make_float2(vf.x * rrms * g.x,
                                          vf.y * rrms * g.y));   // 写：半精度
```

### 9.2 常见变体

| 变体 | 公式差异 | 出现在 |
|------|---------|--------|
| 标准 RMSNorm | $y = x/\mathrm{rms}\cdot\gamma$ | LLaMA、Qwen、Mistral |
| Gemma 风格 | $y = x/\mathrm{rms}\cdot(1+\gamma)$（$\gamma$ 零初始化） | Gemma 系列 |
| 前置转精度 | 先把 $x$ 转 fp32 再归一化、最后转回（与"半精度直接算"结果有差） | 各框架实现细节不一 |

移植权重时**变体必须与训练一致**：Gemma 的 $(1+\gamma)$ 用标准 kernel 加载会静默算错——对拍时用非平凡的 $\gamma$（避开全 1/全 0）才能暴露这类错误。

### 9.3 PyTorch 对拍与 benchmark

```python
import torch

def rms_norm_ref(x, gamma, eps=1e-6):
    # 参考实现：与 torch.nn.RMSNorm（PyTorch >= 2.4）一致
    rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * rms).to(x.dtype) * gamma

N, H = 4096, 4096
x = torch.randn(N, H, device='cuda')
gamma = torch.rand(H, device='cuda') + 0.5          # 非平凡 γ

y_ref = rms_norm_ref(x, gamma)
# y_mine = my_ext.rmsnorm(x, gamma, 1e-6)           # 自己的扩展
# print(torch.allclose(y_mine, y_ref, rtol=1e-4, atol=1e-4))
# print((y_mine - y_ref).abs().max().item())        # fp32 预期 ~1e-6 量级

# 融合版对拍：h 与 y 都要验
# h_ref = x + residual; y_ref = rms_norm_ref(h_ref, gamma)
# my_ext.fused_add_rmsnorm(x, residual, gamma, 1e-6)  # 原地改写 x(→y)、residual(→h)
# 分别 allclose(residual, h_ref)、allclose(x, y_ref)
```

CUDA 侧计时/校验骨架与常规 memory-bound 算子相同（CUDA event + 预热 + 随机数据 + CPU double 参考），报告**有效带宽** $2NH \times 4\mathrm{B}/t$ 并与本卡拷贝算子的实测水位对比——RMSNorm 优化到位的标志就是贴着拷贝的带宽跑。融合版按 8.1 节口径算 $4NH \times 4\mathrm{B}/t$。

三条验证提醒：

- **必须随机数据**：全同值输入下 $\mathrm{rms} = |x|$，缩放退化成符号运算，归约错误全被掩盖；
- **fp16 容差放宽**到 `rtol=1e-2` 量级（累加顺序不同）；
- **测大数值行**：整行乘 $10^3$ 后半精度直接累加会上溢——这是 9.1 节"fp32 累加"的回归测试。

---

## 第 10 章 总结与速查

### 10.1 版本回顾

| 版本 | 核心手段 | 解决的瓶颈 | x 读取遍数 |
|------|---------|-----------|-----------|
| V0 | 树形归约 | —（基准） | 2 |
| V1 | Warp Shuffle 两级归约 | 归约的同步与共享内存往返 | 2 |
| V2 | float4 + 行驻留寄存器 | 标量访问、两阶段重复读 | **1（单算子下限）** |
| V3 | 短行一 Warp | decode 场景并行度不足 | 1 |
| V4 | residual 融合 | 算子链上 $h$ 的一读 | 链路 5 遍 → 4 遍 |

### 10.2 可迁移的经验

1. **结构减法即工程红利**：RMSNorm 砍掉均值后，归约减半、数值陷阱消失、融合变容易——理解算子的数学形状，比堆 CUDA 技巧优先；
2. **"加载时顺手做事 + 数据驻留"**：平方和在加载时累加、残差加法藏进加载路径、缩放用寄存器副本——三处优化同一原理：数据进来一趟，把能做的都做完；
3. **调度按负载形状分派**：一行一 Block / 一行一 Warp 没有绝对优劣，训练（行多）与 decode（行少）各取所需；
4. **融合看算子链**：单算子到流量下限后，收益藏在相邻算子之间的中间结果里——residual、量化、仿射都能并进来。

### 10.3 关键概念速查

| 概念 | 含义 | 出处 |
|------|------|------|
| RMSNorm | $y = x / \sqrt{E[x^2]+\varepsilon} \cdot \gamma$，无中心化无 $\beta$ | 1.1 节 |
| 与 LN 的差别 | 少一个统计量、少一组参数、无抵消风险 | 第 2 章 |
| 两级归约 | Warp 内 shuffle + Warp 间共享内存汇总 | 第 5 章 |
| 行驻留 | 加载后数据留在寄存器，消费阶段不再读显存 | 第 6 章 |
| 行级调度 | 一行一 Block vs 一行一 Warp，按 $(N,H)$ 分派 | 第 7 章 |
| fused_add_rms_norm | residual 相加 + 归一化单 kernel，原地写回 | 第 8 章 |
| 混合精度累加 | IO 半精度、$\Sigma x^2$ 用 fp32 | 9.1 节 |
| Gemma 变体 | $(1+\gamma)$ 缩放，与标准版权重不通用 | 9.2 节 |
