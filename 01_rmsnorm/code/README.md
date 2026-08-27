# RMSNorm 算子优化代码

对应文档: [`../cuda_rmsnorm_optimization_guide.md`](../cuda_rmsnorm_optimization_guide.md)

从文档中提取的可运行代码，覆盖 V0~V4 全部 5 个优化版本，外加第 9 章的
PyTorch 对拍扩展。

公式（对形状 `[N, H]` 的输入，每一行独立）：

```
rms(x) = sqrt( (1/H) * Σ x_i^2 + eps )
y_i    = x_i / rms(x) * gamma_i
```

## 目录结构

```
code/
├── rmsnorm.cu                   # V0~V3 全部 kernel + 计时/校验驱动（第 4~7 章）
├── fused_add_rmsnorm.cu         # V4 residual + RMSNorm 融合 kernel（第 8 章）
├── pytorch_extension/
│   ├── rmsnorm_kernel.cu        # PyTorch CUDA 扩展（第 9 章对拍）
│   ├── setup.py
│   └── test.py
└── README.md
```

## 版本一览

| 版本 | 核心手段 | 解决的瓶颈 | x 读取遍数 | 文件 / 章节 |
|------|---------|-----------|-----------|-------------|
| V0 | 一行一 Block；共享内存树形归约 | —（基准） | 2 | `rmsnorm.cu` / 第 4 章 |
| V1 | Warp Shuffle 两级归约 | 归约的同步与共享内存往返 | 2 | `rmsnorm.cu` / 第 5 章 |
| V2 | float4 向量化 + 行驻留寄存器 | 标量访问、两阶段重复读 | **1（单算子下限）** | `rmsnorm.cu` / 第 6 章 |
| V3 | 短行一 Warp（一 Block 装 8 行） | decode 场景并行度不足 | 1 | `rmsnorm.cu` / 第 7 章 |
| V4 | residual 融合，原地写回 | 算子链上 h 的一读 | 链路 5 遍 → 4 遍 | `fused_add_rmsnorm.cu` / 第 8 章 |

## 编译与运行（standalone）

### V0~V3 对比

```bash
nvcc -O3 -arch=sm_70 rmsnorm.cu -o rmsnorm
./rmsnorm              # 默认 N=4096, H=4096
./rmsnorm 8192 4096    # 指定 N 与 H
```

程序用**随机数据**初始化 `x`（文档强调：全同值输入会掩盖归约错误），
用非平凡 `γ ∈ [0.5, 1.5]` 暴露缩放错误，并以 double 累加的 CPU 参考校验。
依次跑 V0~V3，打印每个版本与 CPU 参考的最大绝对误差、耗时与有效带宽（OK/FAIL）。

> V2/V3 的每线程元素数 `ITEMS` 是编译期常量，本文件按默认 `H=4096`
> （`ITEMS`：V2=4、V3=32）固定；`H != 4096` 时 V2/V3 会跳过，V0/V1 仍正确运行。
> 有效带宽口径：读 x 一遍 + 写 y 一遍 = `2*N*H*4B / t`。

### V4 融合版

```bash
nvcc -O3 -arch=sm_70 fused_add_rmsnorm.cu -o fused_add_rmsnorm
./fused_add_rmsnorm    # 默认 N=4096, H=4096
```

原地（in-place）语义：`x` 结束后被覆写为归一化输出 `y`，`residual` 被覆写为
新残差 `h = x + residual`。程序分别校验 `h` 与 `y` 两个输出。
融合口径有效带宽：读 x、residual + 写 h、y = `4*N*H*4B / t`。

> `-arch=sm_70` 请按你的 GPU 计算能力调整（Warp Shuffle 需 SM 3.0+）。

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

提供两个算子：

- `rmsnorm(x, gamma, eps)` —— 标准 RMSNorm（V2 kernel），返回 `y`；
- `fused_add_rmsnorm(x, residual, gamma, eps)` —— V4 融合，**原地**改写
  `x`（→ 归一化输出）、`residual`（→ 新残差 h）。

`test.py` 用 `torch.nn.RMSNorm`（PyTorch >= 2.4）等价的参考实现对拍，
fp32 预期最大绝对误差在 `~1e-6` 量级。

## 混合精度说明（第 9 章）

文档第 9.1 节给出 fp16/bf16 的正确姿势：**读写用半精度、平方和用 fp32 累加**
（半精度直接累加长行的 Σx² 会迅速丢精度甚至上溢，fp16 最大 65504）。本目录的
standalone 与扩展均以 fp32 为例；迁移到半精度时按该节替换加载/写回路径即可，
归约与缩放的 fp32 逻辑保持不变。对拍时 fp16 容差应放宽到 `rtol=1e-2` 量级。
