# Attention 算子优化代码

## 目录结构

```
code/
├── attention.cu               # CUDA V0/V1/V3/V4：QKᵀ→softmax→PV 到 FlashAttention-2
├── flash_decoding.cu          # CUDA Flash-Decoding：decode 阶段 KV 维切分 + 归并
├── flash_attention_sim.py     # PyTorch 模拟 FA-1 vs FA-2 循环顺序（写 CUDA 前的热身）
├── attention_variants.py      # PyTorch SDPA/MHA/MQA/GQA/MLA + 完整验证链
└── README.md
```

## 一、CUDA 部分

### `attention.cu`（对应文档第 4~8 章）

| 版本 | 核心手段 | 解决的瓶颈 |
|------|---------|-----------|
| V0 | 朴素三 kernel（QKᵀ → softmax → PV），N×N 矩阵落地 HBM | —（基准） |
| V1 | 融合 scale+mask+softmax 为单 kernel，一行一 Block + 两级归约 | softmax 多遍扫描、不合并访存 |
| V3 | FlashAttention 教学版：K/V 分块驻留片上，`(m,l,õ)` 在线递推（每线程一行） | N² 中间矩阵落地、O(N²) 显存 |
| V4 | FlashAttention-2 教学版：split-Q，一个 Warp 负责一行，lane 分摊 head 维 | 片上并行度、Warp 通信、非矩阵乘指令 |

单头版式，`Q/K/V/O` 形状 `[N, D]`（V3/V4 kernel 支持 `[BH, N, D]`，demo 中 `BH=1`）。
程序依次跑 V0~V4，用 **CPU 参考实现**（`cpu_reference`）做正确性校验，并用 `cudaEvent` 计时。

模板 kernel 固定 `D=64`；`V3` 块大小 `Br=Bc=64`（每线程一行），`V4` `Br=8`（8 个 Warp、每 Warp 一行）、`Bc=64`。
要求 `N` 为 `Bc`、`V3Br`、`V4Br` 的倍数。

```bash
nvcc -O3 -arch=sm_70 attention.cu -o attention
./attention                 # 默认 N=1024, D=64, causal=1
./attention 2048 64 1       # 指定 N D causal
```

### `flash_decoding.cu`（对应文档第 9 章）

推理 decode 阶段：每步只有 1 行 query 对全部历史 KV Cache 做 attention。并行度塌缩时，
把 KV 序列维切成 `S` 段给多个 Block 并行，各输出部分结果 `(m,l,õ)`，再由一个轻量 kernel 归并。

- Kernel 1 `decode_partial`：每个 Block（1 个 Warp）负责一段 KV，输出未归一化的部分结果；
- Kernel 2 `decode_reduce`：每个 Block 归并一份 `(batch,head)` 的 `S` 份部分结果，收尾一次除法。

模板固定 `D=128`（须为 32 的倍数）。用 CPU 参考实现校验、`cudaEvent` 计时。

```bash
nvcc -O3 -arch=sm_70 flash_decoding.cu -o flash_decoding
./flash_decoding                 # 默认 BH=8, N=2048, D=128, S=16
./flash_decoding 8 4096 128 32   # 指定 BH N D S
```

> `-arch=sm_70` 请按 GPU 计算能力调整（V3/V4/Flash-Decoding 依赖 Warp Shuffle，需 SM 3.0+）。
> 校验通过标准：`max|diff|` 应为 ~1e-6 量级，容差 1e-3 内即打印 `OK`。

## 二、PyTorch 部分

### `flash_attention_sim.py`（对应文档第 10.3 节）

写 CUDA 前的热身：纯 PyTorch 同时实现 FA-1 风格（外层 KV）与 FA-2 风格（外层 Q），
核心数学一致，只交换循环内外与状态存放位置。验证：

- FA-1 与 FA-2 输出**逐位一致**（合并公式精确）；
- 两者与朴素参考实现相差 ~1e-6（浮点求和顺序不同）；
- 计时对比（FA-2 通常更快）。

```bash
python flash_attention_sim.py
```

有 GPU 用 GPU，否则用 CPU（CPU 上默认 `N=4096` 稍慢，可在文件顶部改小 `N`）。

### `attention_variants.py`（对应变体文档全篇）

包含 SDPA 原子操作、`MultiHeadAttention`(MHA)、`MultiQueryAttention`(MQA)、
`GroupedQueryAttention`(GQA)、`MultiHeadLatentAttention`(MLA) 五个类，以及 RoPE、因果掩码。

`__main__` 把文档各章的内嵌测试串成一条验证链：

- 手写 SDPA ≡ 官方 `F.scaled_dot_product_attention`（第 2 章）；
- MHA 逐步 decode ≡ 一次 prefill（第 4 章，KV Cache 正确性）；
- MQA/GQA 前向输出形状合理（第 5/6 章）；
- MLA 逐步 decode ≡ 一次 prefill（第 7 章，RoPE 位置经 `past_len` 衔接）；
- GQA(g=h) ≡ MHA、GQA(g=1) ≡ MQA（第 8 章，权重拷贝 + 逐位对拍）。

```bash
python attention_variants.py
```

只依赖 `torch` 标准 API，有 GPU 用 GPU，否则自动用 CPU。

## KV Cache 变体开销速查

| | MHA | MQA | GQA | MLA |
|--|-----|-----|-----|-----|
| K/V 份数 | `h` | 1 | `g` | 每头一份（由 latent 重建） |
| 每 token 每层 Cache（元素） | `2hd` | `2d` | `2gd` | `d_c + d_r` |
| 质量 | 基准 | 有损 | ≈ MHA（`g≥8`） | 报告 ≥ MHA |
| 代表模型 | GPT-2、LLaMA-1 | PaLM、StarCoder | LLaMA-2/3、Qwen2 | DeepSeek-V2/V3/R1 |

## 用 Nsight Compute（ncu）分析 CUDA kernel

Attention 优化的核心是**用 kernel 融合消除 N×N 中间矩阵的 HBM 往返**（V0 三 kernel → V1 融合 softmax → V3/V4 FlashAttention）。`ncu` 能逐 kernel 报告耗时、访存量、占用率、warp 通信开销，定量印证融合到底省了多少显存流量。

> 建议加 `-lineinfo`：`nvcc -O3 -arch=sm_70 -lineinfo attention.cu -o attention`。`ncu` 若不在 PATH，用 `/usr/local/NVIDIA-Nsight-Compute/ncu` 或 `/usr/local/cuda/bin/ncu`。V0 会启动 QKᵀ/softmax/PV 三个 kernel，融合版只有一个——报告里的 kernel 数量本身就是融合效果的直接体现。

```bash
# 总览：逐 kernel 对比 Duration 与 Memory Throughput
ncu --set basic ./attention 1024 64 1

# 只测某个版本（各版本 kernel 名不同，先用 --set basic 跑一遍看实际名字，再用 -k 过滤）
ncu -k "regex:.*flash.*|.*attn_v4.*" --set full ./attention 1024 64 1

# Flash-Decoding 的两个 kernel（partial + reduce）
ncu -k "decode_partial|decode_reduce" --set full ./flash_decoding 8 2048 128 16
```

针对 Attention 的关键指标：

```bash
ncu --set full \
    --metrics \
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    ./attention 1024 64 1
```

| 指标 | 观察点 |
|------|--------|
| `dram__bytes.sum` | 全 kernel 显存总流量：V0 因 N×N 矩阵落地 HBM 最大，Flash 版（V3/V4）显著更小 |
| `gpu__dram_throughput...pct_of_peak` | 融合后是否转为计算受限 |
| `sm__pipe_fma_cycles_active...pct_of_peak` | FlashAttention 计算流水线繁忙度 |
| `sm__warps_active...pct_of_peak` | 占用率；V4 split-Q（一 Warp 一行）改善片上并行度 |

```bash
ncu -o attention_report -f --set full ./attention 1024 64 1   # 存 .ncu-rep 用 ncu-ui 打开
```

> PyTorch 侧（`attention_variants.py` / `flash_attention_sim.py`）属于框架算子，用 `ncu python xxx.py` 也能抓，但更推荐用 `torch.profiler` 或 `nsys` 看整体时间线；ncu 更适合上面这些自写 CUDA kernel 的单核深挖。若报 `ERR_NVGPUCTRPERM`，用 `sudo ncu ...` 或让管理员放开性能计数器权限。
