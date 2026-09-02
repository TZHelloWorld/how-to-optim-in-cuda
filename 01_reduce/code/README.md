# Reduce 算子优化代码

对应文档: [`../cuda_reduce_optimization_guide.md`](../cuda_reduce_optimization_guide.md)

从文档中提取的可运行代码，覆盖 V0~V7 全部 8 个优化版本。

## 目录结构

```
code/
├── reduce.cu                    # V0~V7 全部 kernel + 计时/校验驱动
├── pytorch_extension/
│   ├── reduce_kernel.cu         # PyTorch CUDA 扩展（第 12 章）
│   ├── setup.py
│   └── test.py
└── README.md
```

## 版本一览

| 版本 | 核心手段 | 解决的瓶颈 |
|------|---------|-----------|
| V0 | 朴素树形归约 | —（基准） |
| V1 | Strided Index 连续线程映射 | Warp Divergence |
| V2 | 步长从大到小 + `tid < s` | Warp Divergence + Bank Conflict |
| V3 | 每线程加载 2 元素预相加 | 线程闲置 |
| V4 | 手动展开最后一个 Warp | 冗余的 `__syncthreads()` |
| V5 | 模板参数编译期展开 | 循环与运行时分支开销 |
| V6 | Warp Shuffle 两级归约 | 共享内存往返延迟 |
| V7 | float4 向量化 + Grid Stride Loop | 带宽利用率、Block 过多 |

## 编译与运行（standalone）

```bash
nvcc -O3 -arch=sm_70 reduce.cu -o reduce
./reduce              # 默认 32M 元素
./reduce 1048576      # 指定元素个数
```

输入全部填 `1.0`，因此期望的求和结果等于元素个数。程序会依次跑
V0~V7，打印每个版本的结果、耗时与等效带宽，并校验正确性（OK/FAIL）。

> `-arch=sm_70` 请按你的 GPU 计算能力调整（V6/V7 依赖 Warp Shuffle，需 SM 3.0+）。

## PyTorch 扩展

**方式 A：提前编译安装（setup.py）**

```bash
cd pytorch_extension
pip install -e .
python test.py
```

**方式 B：JIT 即时编译（无需 `pip install -e .`）**

```bash
cd pytorch_extension
python run_jit.py
```

`run_jit.py` 用 `torch.utils.cpp_extension.load()` 在首次运行时自动调用 nvcc
编译，编译结果缓存在 `~/.cache/torch_extensions/`，二次运行秒开。它与
`test.py` 的验证逻辑完全一致，只是省去了 setup.py / pip 安装步骤。

期望输出（N = 32M，全部填 2.0）：

```
自定义 kernel 结果: 67108864.0
PyTorch sum 结果:   67108864.0
```

## 用 Nsight Compute（ncu）分析各版本 kernel

`ncu` 是 NVIDIA 官方的 kernel 级 profiler，能逐个 kernel 报告耗时、访存带宽、占用率、warp 分化、bank 冲突等硬件指标——正好用来定量验证 V0~V7 每一步优化到底改善了哪个瓶颈。

> 前提：编译时加 `-lineinfo` 可让报告关联到源码行（`nvcc -O3 -arch=sm_70 -lineinfo reduce.cu -o reduce`）。若 `ncu` 不在 PATH，用绝对路径 `/usr/local/NVIDIA-Nsight-Compute/ncu` 或 `/usr/local/cuda/bin/ncu`；非 root 环境如提示权限不足，见文末说明。

### 快速总览（所有 kernel 的耗时与吞吐）

```bash
# --set basic 只收集轻量指标，开销小、适合先看全局
ncu --set basic ./reduce 1048576
```

程序会依次跑 V0~V7 八个 kernel，`ncu` 为每个 kernel 各打印一份报告——对比它们的 Duration 与 Compute/Memory Throughput，即可看出逐版优化的效果。

### 只测某几个版本（按 kernel 名过滤）

reduce 的 8 个版本 kernel 名各不相同（形如 `reduce_v0`、`reduce_v1`…）。用 `-k` 正则只测感兴趣的版本，避免全量分析太慢：

```bash
# 只分析 V0 和 V2（正则匹配 kernel 名）
ncu -k "reduce_v0|reduce_v2" --set full ./reduce 1048576

# 每个匹配到的 kernel 只测第一次启动，避免重复
ncu -k "regex:reduce_v.*" -c 8 --set full ./reduce 1048576
```

### 针对 reduce 瓶颈的关键指标

reduce 是典型的**访存受限 + 归约分化/冲突**算子，重点看这几组指标：

```bash
ncu -k "regex:reduce_v.*" \
    --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__average_inst_executed_per_warp.ratio,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    ./reduce 1048576
```

| 指标 | 含义 | 在 reduce 里对应的优化点 |
|------|------|------------------------|
| `gpu__dram_throughput...pct_of_peak` | DRAM 带宽利用率 | V7 float4 + grid-stride 后应接近峰值 |
| `smsp__average_inst_executed_per_warp` | 每 warp 平均指令数 | 分化越重该值越高（V0→V1 明显下降） |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared` | 共享内存 bank 冲突次数 | V2 消除冲突后应降为 0 |
| `sm__warps_active...pct_of_peak` | 达成占用率 | 反映延迟隐藏是否充分 |

### 常用配套选项

```bash
ncu -o reduce_report -f --set full ./reduce 1048576   # 结果存成 reduce_report.ncu-rep，用 ncu-ui 打开
ncu --section SpeedOfLight ./reduce 1048576            # 只看 Speed-of-Light 概览
ncu --launch-count 1 -k reduce_v7 ./reduce 1048576     # 只抓一次启动
```

> **权限问题**：若报 `ERR_NVGPUCTRPERM`（无法访问性能计数器），需 root 运行（`sudo ncu ...`）或让管理员放开权限（Linux 参见 NVIDIA 文档设置 `NVreg_RestrictProfilingToAdminUsers=0`）。仅收集 `--set basic`/launch 级指标时通常不需要特殊权限。
