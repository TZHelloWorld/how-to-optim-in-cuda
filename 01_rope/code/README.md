# RoPE（旋转位置编码）代码

对应文档: [`../rope.md`](../rope.md)

从文档中提取的可运行 Python 代码，覆盖两条主线：第 6 章的**最小实现**（用于验证核心数学性质），以及第 4 章的 **SGLang 风格 `RotaryEmbedding`**（把工程源码逐段组织为一个自洽的类）。

## 目录结构

```
code/
├── rope.py      # 最小实现 rope() + SGLang 风格 RotaryEmbedding + 对拍测试
└── README.md
```

## 文件 / 函数一览

| 符号 | 用途 | 对应文档章节 |
|------|------|-------------|
| `rope(x, pos, base)` | 最小 RoPE 实现（NeoX 半半式配对），逐元素旋转公式 | 第 6 章 / 3.4 节 |
| `apply_rotary_emb(x, cos, sin, is_neox_style)` | 旋转应用原语，支持 NeoX（半半）与 GPT-J（交错）两种配对风格 | 4.4 节 |
| `RotaryEmbedding._compute_inv_freq()` | 逆频率计算 θ_i = base^(-2i/d) | 4.2 节 |
| `RotaryEmbedding._compute_cos_sin_cache()` | 离线预计算 cos/sin 缓存表 `[max_pos, rotary_dim]` | 4.3 节 |
| `RotaryEmbedding.forward_native()` | 完整前向：查表 → 部分旋转 → 对 Q/K 施加旋转 | 4.5 节 |

## 测试块（`__main__`）说明

`rope.py` 的 `if __name__ == "__main__":` 依次运行三组测试：

1. **`_test_minimal_properties`（第 6 章）** — 用最小 `rope()` 验证三条核心性质：
   - 性质 1：分数只依赖相对距离 `m-n`（整体平移位置分数不变，`allclose`）；
   - 性质 2：保范性（旋转不改变模长，2.1 节正交性）；
   - 性质 3：位置 0 是恒等变换（所有旋转角为 0）。
2. **`_test_shapes_and_equivalence`（第 4 章）** — 构造随机 `query`/`key` 张量 `[num_tokens, num_heads, head_size]`，跑 `RotaryEmbedding.forward_native`，打印输入/输出 shape 与缓存 shape；再用 `torch.allclose` 验证 **NeoX 风格的 `forward_native` 与逐 token/head 的最小 `rope()` 数值一致**（两个等价实现对拍）。
3. **`_test_neox_vs_gptj_differ`（3.4 / 4.4 节）** — 验证 NeoX 与 GPT-J 两种配对风格对同一输入布局给出**不同**结果（混用会静默算错），且在 `pos=0` 时两者都退化为恒等。

## 运行命令

```bash
python rope.py
```

期望输出（节选）：

```
性质1 分数(相对距离4): -0.386794  -0.386793   对照(距离6): 0.662847
性质2 保范性: 11.768735  11.768736
性质3 pos=0 恒等: max_err = 0.00e+00
对拍 forward_native(NeoX) vs 最小 rope(): max_err = 0.00e+00
NeoX vs GPT-J 输出 max_diff = 2.1737（应显著非零）
全部测试通过。
```

## 依赖

仅依赖 PyTorch 标准 API：

```bash
pip install torch
```

## 说明

代码忠实于文档 4.2~4.5 节与第 6 章的实现，核心算法逻辑未做改动；仅补充了
使其能运行/自洽的上下文（`import torch`、把四段源码封装为 `RotaryEmbedding`
类、构造函数、`__main__` 测试块）。文档中数学上等价的实现（NeoX `forward_native`
与最小 `rope()`）在测试块里用 `torch.allclose` 验证一致。
