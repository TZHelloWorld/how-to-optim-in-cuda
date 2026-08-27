# GEMM 算子优化代码

对应文档: [`../cuda_gemm_optimization_guide.md`](../cuda_gemm_optimization_guide.md)

从文档中提取的可运行代码，覆盖 V0~V7 全部 8 个优化版本（SGEMM `C = A × B`，行主序）。

## 目录结构

```
code/
├── sgemm.cu                     # V0~V6 全部 fp32 kernel + 计时/CPU 校验驱动
├── hgemm_wmma.cu                # V7 Tensor Core / WMMA 半精度实现（第 11 章）
├── pytorch_extension/
│   ├── gemm_kernel.cu           # PyTorch CUDA 扩展，封装 sgemm_v5（第 12 章）
│   ├── setup.py
│   └── test.py
└── README.md
```

## 版本一览

| 版本 | 文件 | 核心手段 | 解决的瓶颈 | 复用建立层次 |
|------|------|---------|-----------|-------------|
| V0 | sgemm.cu | 一线程一元素（映射错误：threadIdx.x→行） | —（基准） | 无 |
| V1 | sgemm.cu | 交换行列映射（threadIdx.x→列） | 全局内存不合并 | 无（仅访问变合并） |
| V2 | sgemm.cu | Block Tiling + 共享内存（TILE=32） | 全局内存零复用 | global → smem |
| V3 | sgemm.cu | 一维 Thread Tiling（TM=8） | LDS 指令占比过高 | smem → reg（单侧） |
| V4 | sgemm.cu | 二维 Thread Tiling / 8×8 外积 | LDS/FMA 仍 ≥ 1 | smem → reg（双侧） |
| V5 | sgemm.cu | float4 向量化 + As 转置存储 | 标量访存指令过多 | 提高各层搬运效率 |
| V6 | sgemm.cu | 双缓冲（Double Buffering） | 加载与计算串行 | 时间维度重叠 |
| V7 | hgemm_wmma.cu | Tensor Core（WMMA，half×half→float） | CUDA Core fp32 吞吐上限 | 硬件级矩阵运算 |

## 分块参数

| 版本 | BM×BN | BK | TM×TN | 线程数/Block | 形状约束 |
|------|-------|----|----|-------------|---------|
| V0/V1 | 32×32 | — | — | 1024 | 任意（有边界判断） |
| V2 | 32×32 | 32 | — | 1024 | 任意（有边界判断） |
| V3 | 64×64 | 8 | 8×1 | 512 | M,N % 64==0；K % 8==0 |
| V4/V5/V6 | 128×128 | 8 | 8×8 | 256 | M,N % 128==0；K % 8==0 |
| V7 | 64×64（16×16 块×4×4 Warp） | 16 | 16×16/Warp | 512 | M,N,K % 16==0 |

> V3~V6 是分块 kernel，没有边界判断，对形状有硬性整除约束。驱动程序 `sgemm.cu`
> 会对不满足约束的版本自动跳过（打印 `SKIP(shape)`）。默认 `M=N=K=1024` 满足
> V0~V6 全部约束。

## 编译与运行（standalone）

### V0~V6（fp32）

```bash
nvcc -O3 -arch=sm_70 sgemm.cu -o sgemm
./sgemm              # 默认 M=N=K=1024（便于 CPU 校验）
./sgemm 2048         # 方阵边长 2048
./sgemm 1024 768 512 # 指定 M N K
```

程序用随机数据填充 A、B（不用全同值，以暴露行列写反等索引错误），先算一遍 CPU
参考结果，再依次跑 V0~V6，打印每个版本的耗时、GFLOPS，并按相对误差校验正确性
（PASS/FAIL，容差 `rtol=1e-3`）。

### V7（Tensor Core，half）

```bash
nvcc -O3 -arch=sm_70 hgemm_wmma.cu -o hgemm_wmma
./hgemm_wmma         # 默认 M=N=K=1024
./hgemm_wmma 2048
```

输入 half、累加 float。因半精度动态范围有限，校验容差放宽到 `rtol=2e-2`。

> `-arch=sm_70` 请按你的 GPU 计算能力调整。Tensor Core（WMMA）需 Volta（sm_70）及
> 以上；V0~V6 只需 sm_60+。

## PyTorch 扩展

**方式 A：提前编译安装（setup.py）**

```bash
cd pytorch_extension
pip install -e .     # 调用 nvcc 编译并安装为 Python 可导入模块
python test.py
```

**方式 B：JIT 即时编译（无需 `pip install -e .`）**

```bash
cd pytorch_extension
python run_jit.py
```

`run_jit.py` 用 `torch.utils.cpp_extension.load()` 首次运行时自动 nvcc 编译并缓存，
验证逻辑与 `test.py` 完全一致。

`test.py` / `run_jit.py` 用 `M=N=K=4096`（满足 sgemm_v5 的 128/8 整除约束），
与 `A @ B`（cuBLAS）对比正确性，并测量 my_matmul 与 cuBLAS 的耗时与 TFLOPS。

## 说明

- 所有 kernel 算法逻辑严格忠实于文档，仅补充了可编译运行所需的上下文：头文件、
  `CUDA_CHECK` 宏、`main`、主机端矩阵初始化、kernel launch、CPU 参考 GEMM、
  cudaEvent 计时与 GFLOPS 计算。
- CPU 参考实现即文档第 1 章的三重循环点积。
- 矩阵默认取 1024×1024，CPU 校验约需数秒，规模适中。
