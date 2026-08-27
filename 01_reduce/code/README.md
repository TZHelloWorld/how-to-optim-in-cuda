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
