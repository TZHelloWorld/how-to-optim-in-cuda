# LayerNorm 算子优化代码

对应文档: [`../cuda_layernorm_optimization_guide.md`](../cuda_layernorm_optimization_guide.md)

从文档中提取的可运行代码，覆盖 V0~V4 全部优化版本，外加第 9 章的 Welford 稳健版与 PyTorch 对拍扩展。

## 目录结构

```
code/
├── layernorm.cu                 # V0~V4 + Welford 全部 kernel + 计时/校验驱动
├── pytorch_extension/
│   ├── layernorm_kernel.cu      # PyTorch CUDA 扩展（第 9.3 节）
│   ├── setup.py
│   └── test.py
└── README.md
```

## 版本一览

| 版本 | 核心手段 | 解决的瓶颈 | x 读取遍数 | 对应章节 |
|------|---------|-----------|-----------|---------|
| V0 | 两遍统计 + 共享内存树形归约 | —（基准） | 3 | 第 4 章 |
| V1 | `(Σx, Σx²)` 单遍统计 + 树形归约 | 统计多扫一遍 | 2 | 第 5 章 |
| V2 | Warp Shuffle 两级归约（`float2` 成对） | 归约同步与共享内存往返 | 2 | 第 6 章 |
| V3 | float4 向量化 | 标量访问喂不满带宽 | 2 | 第 7 章 |
| V4 | 行驻留寄存器（模板 `ITEMS`） | 统计/归一化两阶段重复读 | **1（下限）** | 第 8 章 |
| Welford | 换合并算子，归约骨架不变 | 单遍 naive 的数值风险 | 1~2 | 第 9 章 |

LayerNorm 沿最后一维（hidden 维 H）逐行归一化 + 仿射:

```
y = (x - mean) / sqrt(var + eps) * gamma + beta
```

## 编译与运行（standalone）

```bash
nvcc -O3 -arch=sm_70 layernorm.cu -o layernorm
./layernorm              # 默认 N=4096, H=4096
./layernorm 8192 2048    # 指定 N H
```

程序会：

1. 用**随机数据**初始化 `x`（文档 9.2 节强调：全同值输入方差为 0，测不出统计错误）；
2. 用 **double 累加的 CPU 参考**做金标准；
3. 依次跑 V0~V4 + Welford，逐元素对比 CPU 参考，打印 `max_err`、耗时（cudaEvent 计时，100 次平均）与等效带宽，并给出 OK/FAIL。

有效带宽按"最少流量 `2NH*4B`（读 x + 写 y）/ 时间"计——V0 因多读显存实际带宽虚低，恰好反映其劣势。

### 版本约束说明

- **V3（float4）** 要求 `H % 4 == 0`；不满足时该行标 `SKIP`。
- **V4（行驻留寄存器）** 要求 `H = ITEMS * blockDim.x` 且 `ITEMS` 是编译期常量。
  驱动中针对 `blockDim.x=256` 预实例化了 `ITEMS = 4 / 8 / 16`（即 `H = 1024 / 2048 / 4096`）。
  其它 `H` 该行标 `SKIP (H!=ITEMS*256)`。默认 `H=4096 → ITEMS=16`，正是文档示例配置。

> `-arch=sm_70` 请按你的 GPU 计算能力调整（V2/V3/V4/Welford 依赖 Warp Shuffle，需 SM 3.0+）。

## PyTorch 扩展（第 9.3 节）

扩展导出 `layernorm(x, gamma, beta, eps)`，kernel 采用文档 V2 的
Warp Shuffle 两级归约 + 单遍 `(Σx, Σx²)`，与官方 `nn.LayerNorm` 对拍。

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

`test.py` / `run_jit.py` 做两组对拍：

- 常规输入：fp32 容差 `rtol=1e-4, atol=1e-4`，`max abs err` 预期 `1e-6 ~ 1e-5` 量级；
- 极端分布（`μ >> σ`，全体加 `1e4`）：暴露单遍 naive 版的灾难性抵消（文档 2.2 节），
  该场景下误差偏大属预期——需要 `layernorm.cu` 里的 Welford 版才稳定。

## 数值方法对照（文档第 2 章）

| 方法 | x 扫描遍数 | 数值稳健性 | 本仓库实现 |
|------|-----------|-----------|-----------|
| 两遍扫描 | 2 | 最好 | V0 |
| 单遍 `E[x²]-μ²` | 1 | 差（μ≫σ 时崩溃） | V1~V4、扩展 kernel |
| Welford + 并行合并 | 1 | 好 | `layernorm_welford` |
