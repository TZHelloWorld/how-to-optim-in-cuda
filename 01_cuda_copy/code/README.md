# CUDA 拷贝算子优化代码

对应文档: [`../cuda_copy_operator_guide.md`](../cuda_copy_operator_guide.md)

从文档中提取的可运行代码，覆盖 SM 拷贝 kernel 的逐版优化、拷贝融合、
kernel 内拷贝原语（cp.async）、H2D 分块流水，以及 Copy Engine 查询。

## 目录结构

```
code/
├── copy_bench.cu                # D2D 拷贝四版对比 + 融合（§4.3 / §6.1）
├── ce_query.cu                  # 查询 Copy Engine 数量（§2.2）
├── cp_async_tile.cu             # kernel 内 cp.async tile 装载（§5.2）
├── h2d_pipeline.cu              # 大批量 H2D 分块流水，CE 与 SM 重叠（§6.3）
├── pytorch_extension/
│   ├── copy_kernel.cu           # PyTorch CUDA 扩展：float4 拷贝 + 融合
│   ├── setup.py
│   ├── test.py
│   └── profile_copy_paths.py    # profiler 区分三种拷贝路径（§3.4）
└── README.md
```

## 版本一览（copy_bench.cu）

| 版本 | 核心手段 | 说明 | 章节 |
|------|---------|------|------|
| cudaMemcpy | 驱动 / Copy Engine 路径 | 不占 SM，规整大块拷贝的基准上限 | §3.2 / §4.3 |
| V0 naive | 一线程一元素（4 B） | 并发全靠海量线程 | §4.3 |
| V1 gridstride | grid-stride loop | 固定网格反复扫、线程复用；并发不足反而最慢 | §4.3 |
| V2 float4 | float4 向量化（LDG.128） | 在途字节 ×4，补上 V1 的坑 | §4.3 |
| V3 fused | 拷贝 + scale + ReLU 融合 | 计算藏进访存影子里，带宽同 V2 | §6.1 |

> 有效带宽定义（读 + 写）：`BW_eff = 2N / t`，即 CUDA Best Practices Guide
> 的标准算法（§1.2）。

## 编译与运行（standalone）

```bash
# 主 benchmark：四版拷贝 + 融合对比，含正确性校验与带宽 GB/s
nvcc -O3 -arch=native copy_bench.cu -o copy_bench
./copy_bench            # 默认 2^28 个 float = 1 GiB
./copy_bench 67108864   # 指定元素个数（会向下取整到 4 的倍数）

# 查询 Copy Engine 数量
nvcc ce_query.cu -o ce_query
./ce_query

# 大批量 H2D 分块流水（串行 vs 双流重叠）
nvcc -O3 -arch=native h2d_pipeline.cu -o h2d_pipeline
./h2d_pipeline

# kernel 内 cp.async（需要计算能力 >= 8.0 / Ampere）
nvcc -O3 -arch=sm_80 cp_async_tile.cu -o cp_async_tile
./cp_async_tile
```

`copy_bench` 会依次跑 cudaMemcpy、V0~V3，打印每个版本的耗时、等效带宽，
并校验拷贝结果与输入一致（融合版校验 `relu(x * alpha)`），输出 OK/FAIL。

> `-arch=native` 会按当前机器的 GPU 自动选择；无该 GPU 时可改为具体架构，
> 如 `-arch=sm_80`。`cp_async_tile.cu` 依赖 Ampere 的 `cp.async`，
> 在计算能力 < 8.0 的设备上会自动跳过。

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

`run_jit.py` 用 `torch.utils.cpp_extension.load()` 首次运行时自动 nvcc 编译并缓存，
验证逻辑与 `test.py` 完全一致。

`copy_kernel` 提供两个入口，kernel 算法与 `copy_bench.cu` 中的 V2 / V3 一致：

- `copy_float4(x)`：float4 向量化纯拷贝，结果应与输入完全一致；
- `copy_scale_relu(x, alpha)`：拷贝 + scale + ReLU 融合，结果应等于 `relu(x * alpha)`。

期望输出：

```
copy_float4 一致: True
copy_scale_relu 最大误差: 0.000e+00
```

用 profiler 区分 CE 路径与 SM 路径（§3.4，需要 GPU）：

```bash
python profile_copy_paths.py
# 表中可见 Memcpy DtoD（CE）与 vectorized_elementwise_kernel（SM）
```

## 用 Nsight Compute（ncu）分析拷贝 kernel

拷贝是**纯访存受限**的算子，唯一目标就是逼近 DRAM 峰值带宽。`ncu` 能逐 kernel 报告实际带宽、在途请求、指令数，定量对比 V0~V3。

> 注意：`cudaMemcpy` 走的是 **Copy Engine（CE）**，不是 SM kernel，`ncu` 抓不到它（它不是 kernel launch）——CE 路径请用 `nsys` 或上面的 `profile_copy_paths.py` 观察。`ncu` 只分析 V0~V3 这些**走 SM 的拷贝 kernel**。建议加 `-lineinfo`：`nvcc -O3 -arch=native -lineinfo copy_bench.cu -o copy_bench`。`ncu` 若不在 PATH，用 `/usr/local/NVIDIA-Nsight-Compute/ncu` 或 `/usr/local/cuda/bin/ncu`。

```bash
# 总览：逐 kernel 对比 Duration 与 Memory Throughput
ncu --set basic ./copy_bench 67108864

# 只测某几版（kernel 名：copy_naive / copy_gridstride / copy_float4 / copy_scale_relu）
ncu -k "copy_float4|copy_scale_relu" --set full ./copy_bench 67108864
```

纯访存算子最该盯的就是带宽和访存效率：

```bash
ncu -k "regex:copy_.*" \
    --metrics \
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
smsp__inst_executed.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    ./copy_bench 67108864
```

| 指标 | 观察点 |
|------|--------|
| `gpu__dram_throughput...pct_of_peak` | 头号指标：`copy_float4`（LDG.128）应接近峰值，`copy_gridstride` 并发不足时偏低 |
| `dram__bytes.sum` | 读+写总流量，应 ≈ `2N`；明显偏大说明有非合并访问拉了无用数据 |
| `smsp__inst_executed.sum` | `copy_float4` 让指令数降到约 `copy_naive` 的 1/4（一条指令搬 16 字节） |
| `sm__warps_active...pct_of_peak` | 占用率，解释 `copy_gridstride` 为何反而最慢（并发不足） |

`h2d_pipeline`（H2D 分块流水，CE 与 SM 重叠）属于**多流重叠**场景，重点是时间线上引擎是否并行——这类分析用 `nsys profile ./h2d_pipeline` 看时间线更直观，`ncu` 适合深挖其中单个 SM kernel。

```bash
ncu -o copy_report -f --set full ./copy_bench 67108864   # 存 .ncu-rep 用 ncu-ui 打开
```

> 若报 `ERR_NVGPUCTRPERM`（性能计数器权限不足），用 `sudo ncu ...` 或让管理员放开权限；仅 `--set basic` 通常无需特殊权限。
