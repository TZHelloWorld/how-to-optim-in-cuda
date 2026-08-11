# Attention 变体入门：SDPA、MHA、MQA、GQA 与 MLA

> 本文面向初学者，讲清楚注意力机制的五种常见形态：**SDPA**（缩放点积注意力）、**MHA**（多头注意力）、**MQA**（多查询注意力）、**GQA**（分组查询注意力）和 **MLA**（多头潜在注意力）。每一种都包含三部分：它想解决什么问题、公式长什么样、以及一份**可以直接运行的 PyTorch 代码**（有 GPU 用 GPU，没有就自动用 CPU，正确性都做过验证）。
>
> 先说一个贯穿全文的关键认识：**这五种结构不是五个孤立的发明，而是同一个问题的一步步改进**。这个问题就是——大模型生成文字时需要一块叫 **KV Cache** 的缓存，它太占显存了。"怎么把 KV Cache 变小，同时尽量不影响模型效果"，就是 MHA → MQA → GQA → MLA 这条演进路线的全部动机。带着这个问题去看，每种结构都很好理解。
>
> **阅读路线**：第 1 章先把问题（KV Cache）讲清楚；第 2 章从最简单的单份 Q/K/V 计算（SDPA）入手；第 3 章是一个预备知识插曲，讲透代码里到处出现的 `view`/`transpose` 形状操作（不懂这个，多头的代码只能死记硬背）；第 4~7 章依次讲 MHA、MQA、GQA、MLA；第 8 章横向对比，并用代码验证它们之间的关系。
---

## 目录

- [第 1 章 问题定义与主线：KV Cache](#第-1-章-问题定义与主线kv-cache)
- [第 2 章 SDPA：缩放点积注意力](#第-2-章-sdpa缩放点积注意力)
- [第 3 章 形状操作预备：view、transpose 与 matmul 的批量语义](#第-3-章-形状操作预备viewtranspose-与-matmul-的批量语义)
- [第 4 章 MHA：多头注意力与 KV Cache](#第-4-章-mha多头注意力与-kv-cache)
- [第 5 章 MQA：多查询注意力](#第-5-章-mqa多查询注意力)
- [第 6 章 GQA：分组查询注意力](#第-6-章-gqa分组查询注意力)
- [第 7 章 MLA：多头潜在注意力](#第-7-章-mla多头潜在注意力)
- [第 8 章 横向对比与统一验证](#第-8-章-横向对比与统一验证)

---

## 第 1 章 问题定义与主线：KV Cache

### 1.1 符号约定

全文统一使用下面这套记号。现在不需要记住，读到后面随时回来查即可（掩码的约定：bool 张量，**True 的位置表示"被屏蔽、不许看"**）：

| 符号 | 形状/取值 | 含义 |
|------|----------|------|
| $B$ | 标量 | batch size，一次同时处理多少条序列 |
| $N$ | 标量 | query 的序列长度（prefill 时是整段长度，decode 时是 1，见 4.3 节） |
| $M$ | 标量 | key/value 的总长度（包含缓存的历史；prefill 时 $M = N$） |
| $h$ | 标量 | query 的头数（num_heads） |
| $g$ | 标量 | key/value 的头数（num_kv_heads），要求 $g$ 能整除 $h$ |
| $d$ | 标量 | 每个头的维度（head_dim） |
| $D$ | 标量 | 隐藏层宽度 hidden_size，$D = h \cdot d$ |
| $X$ | $N \times D$ | 一层的输入隐状态（单个样本；代码里带 batch 维，是 $B \times N \times D$） |
| $S$、$P$ | $N \times M$ | 注意力分数矩阵、softmax 之后的权重矩阵 |

### 1.2 演进时间线

先看全景，心里有个地图：

| 变体 | 年份 | 论文 | 核心思想 | 代表模型 |
|------|------|------|---------|---------|
| SDPA | 2014/2017 | 注意力机制源于 [Bahdanau et al.](https://arxiv.org/abs/1409.0473)（加性形式）；缩放点积形式由 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 确立 | $\mathrm{softmax}(QK^\top/\sqrt{d})V$，一切的基础运算 | —（是构件，不是完整模型结构） |
| MHA | 2017 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | $h$ 个头，每个头有自己独立的 Q/K/V | GPT 系、BERT、LLaMA-1 |
| MQA | 2019 | [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150) | 所有头**共用一份** K/V，Cache 缩小 $h$ 倍 | PaLM、StarCoder、Gemini |
| GQA | 2023 | [GQA 论文](https://arxiv.org/abs/2305.13245) | $h$ 个头分成 $g$ 组，每组共用一份 K/V，是 MHA 和 MQA 的折中 | LLaMA-2/3-70B、Qwen2、Mistral |
| MLA | 2024 | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 不减少 K/V 的份数，改为把 K/V **压缩**成一个小向量来缓存 | DeepSeek-V2/V3/R1 |

### 1.3 主线：KV Cache 为什么是个大问题

大模型生成文字是**一个词一个词往外蹦**的（这叫自回归生成）。生成第 $t$ 个词时，这个词的 query 要和**前面所有词**的 key、value 做注意力计算。

关键观察：前面那些词的 key 和 value，**跟新生成的词没有任何关系**——第 $j$ 个词的 $k_j, v_j$ 只由第 $j$ 个词自己算出来，算一次就永远有效。既然如此，就没必要每生成一个词都重算一遍历史，把它们**缓存**下来反复使用就行。这块缓存就叫 **KV Cache**（第 4.3 节会详细讲原理和代码）。

KV Cache 的大小可以精确算出来：

$$
\mathrm{KVCache} = \underbrace{2}_{K \text{ 和 } V \text{ 各一份}} \times B \times M \times \underbrace{g \times d}_{\text{每层每个 token}} \times N_{\mathrm{layers}} \times \mathrm{bytes}
$$

拿 LLaMA-2-70B 的真实规格（$h = 64$、$d = 128$、80 层、fp16 即每个数 2 字节）算一笔账：

| 方案 | $g$ | 每个 token 的 Cache | 4K 上下文 × batch 32 |
|------|-----|---------------|---------------------|
| MHA | 64 | 2.62 MB | **344 GB —— 一台 8 卡 A100 都放不下** |
| GQA（$g = 8$，LLaMA-2-70B 实际采用） | 8 | 0.33 MB | 43 GB |
| MQA | 1 | 0.04 MB | 5.4 GB |

为什么这么在意 Cache 大小？两个原因：

1. **Cache 决定并发上限**：显存就那么大，Cache 越大，能同时服务的请求（batch）就越少，吞吐直接受限；
2. **Cache 决定生成速度**：生成每个新词都要把整个 Cache 从显存里完整读一遍。这一步的瓶颈不是计算而是显存带宽——Cache 越小，每个词的延迟越低。

所以 MQA、GQA、MLA 优化的都是同一个量：**每个 token 每层需要缓存多少个数**（MHA 是 $2hd$ 个）。它们的区别只是压缩手段不同、以及为保住效果做了什么补救。记住这条主线，后面每一章都是在回答"怎么把 $2hd$ 变小"。

---

## 第 2 章 SDPA：缩放点积注意力

### 2.1 定义

SDPA（Scaled Dot-Product Attention）是后面所有变体共用的基础运算，定义只有一行：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

三个输入可以用"查资料"来打比方：query 是你提的问题，key 是每份资料的标签，value 是资料的内容。注意力做的事就是：拿问题和每个标签比对相似度，再按相似度加权把内容混合起来。

| 符号 | 形状 | 含义 |
|------|------|------|
| $Q$（query） | $N \times d$ | 发起查询的一方，共 $N$ 个查询 |
| $K$（key） | $M \times d$ | 被检索的标签，共 $M$ 个 |
| $V$（value） | $M \times d$ | 被加权混合的内容，和 key 一一对应 |
| $O$（输出） | $N \times d$ | 每一行是 value 的一个加权和 |

计算分三步：

| 步骤 | 公式 | 形状 | 含义 |
|------|------|------|------|
| ① | $S = QK^\top / \sqrt{d}$ | $N \times M$ | 每个 query 和每个 key 算相似度分数 |
| ② | $P = \mathrm{softmax}(S)$ | $N \times M$ | 每行归一化成权重（每行加起来等于 1） |
| ③ | $O = PV$ | $N \times d$ | 按权重对 value 加权求和 |

关于 $S$ 的形状为什么是 $N \times M$（而不是 $M \times N$），可以从两个角度确认：

- **矩阵乘法角度**：$Q$ 是 $N \times d$，$K^\top$ 是 $d \times M$，乘出来自然是 $N \times M$；
- **语义角度**：一共 $N$ 个 query，每个 query 要对全部 $M$ 个 key 各打一个分，所以是 $N$ 行 $M$ 列——**第 $i$ 行放的是"第 $i$ 个 query 对所有 key 的分数"**。softmax 沿行方向（$M$ 那一维）归一化，正因为每一行是同一个 query 的权重分布；第③步再用这行权重去加权 $M$ 行 value，得到该 query 的输出（$1 \times d$），$N$ 个 query 叠起来就是 $N \times d$。

两个细节，后面会反复用到：

- **为什么要除以 $\sqrt{d}$**：假设 $q$、$k$ 的每个分量都近似独立、方差为 1，那么点积 $q \cdot k$ 是 $d$ 个乘积之和，方差是 $d$。$d$ 一大，分数就会很大，softmax 会"饱和"——最大的那项权重接近 1，其余接近 0，梯度也几乎消失。除以 $\sqrt{d}$ 把方差拉回 1，避免这个问题；
- **掩码（mask）**：第①②步之间通常要加掩码，最常见的是**因果掩码（causal mask）**——位置 $i$ 的 query 只允许看位置 $j \le i$ 的 key（生成文字时不能"偷看未来"）。实现方法：把 $S$ 中 $j > i$ 的位置填上一个极大的负数，softmax 之后这些位置的权重就是 0。

一个重要的提前预告：后面 MHA/MQA/GQA/MLA 的所有差异，**都发生在 SDPA 的外面**——差别只在 Q/K/V 怎么投影出来、有几份、缓存什么。上面这三步计算本身从头到尾不变。所以我们先把它写成一个可复用的函数。

### 2.2 实现

```python
import math
import torch
from torch import nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"   # 全文代码默认在 CUDA 上运行

def scaled_dot_product_attention(query, key, value, attn_mask=None):
    """SDPA：本文所有变体共用的原子操作
    query: (..., N, d)   key/value: (..., M, d)
    attn_mask: bool 张量，True 的位置被屏蔽，形状可广播到 (..., N, M)
    """
    d_k = query.size(-1)
    scores = query @ key.transpose(-1, -2) / math.sqrt(d_k)   # ① S = QKᵀ/√d，(..., N, M)
    if attn_mask is not None:
        # 用有限大负数而非 -inf：万一某行被全部屏蔽（如全 padding），softmax 退化为
        # 均匀分布而不是 NaN
        scores = scores.masked_fill(attn_mask, -1e9)
    probs = torch.softmax(scores, dim=-1)                     # ② P = softmax(S)
    return probs @ value                                      # ③ O = PV

def causal_mask(N, device=None):
    """因果掩码：上三角（j > i）为 True，屏蔽未来位置"""
    return torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)

# ---- 与 PyTorch 官方实现对拍（CUDA 上运行；官方版会自动分发到融合后端）----
torch.manual_seed(0)
q = torch.randn(2, 8, 16, 64, device=device)   # (B, h, N, d)
k, v = torch.randn_like(q), torch.randn_like(q)
out = scaled_dot_product_attention(q, k, v, causal_mask(16, device))
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(torch.allclose(out, ref, atol=1e-5))     # True
```

三个实现细节：

- **前面的维度自动当批量处理**：这个函数只关心最后两维 $(N, d)$ 和 $(M, d)$。前面不管是 $(B)$ 还是 $(B, h)$，`matmul` 都会自动把它们当成批量维（3.4 节详细解释）。正因为这样，这一个函数能被后面所有变体直接复用；
- **掩码用 bool 加 `masked_fill`**：True 表示屏蔽，语义清楚，而且和 padding 掩码可以直接用 `|` 组合；
- **填 `-1e9` 而不是 `-inf`**：如果某一行被全部屏蔽（比如整句都是 padding），用 `-inf` 会让 softmax 算出 NaN，用有限大负数则退化成均匀分布，程序不会崩。

### 2.3 与 `F.scaled_dot_product_attention` 的关系

PyTorch 2.x 自带的 `F.scaled_dot_product_attention` 做的是同样的数学运算，但它是**融合实现**——会自动选择 FlashAttention 等高效后端，速度快得多。生产代码应该直接用它。本文手写一遍只是为了看清结构，两者随时可以用 `allclose` 互相验证（上面已经演示了）。

到这里，"单份 Q/K/V"的世界已经讲完了。但真实模型是**多头**的：$h$ 份 Q/K/V 同时做 $h$ 次 SDPA，张量会多出 batch 和 head 两个维度。在写多头代码之前，先花一章把形状操作的原理弄明白——否则 `view(B, N, h, d).transpose(1, 2)` 这行在所有 attention 代码里都会出现的写法，就只能死记硬背了。

---
## 第 3 章 形状操作预备：view、transpose 与 matmul 的批量语义

本章回答三个问题：

1. 形状操作的底层原理是什么？（3.1）
2. `view` / `reshape` / `transpose` / `permute` 各自做了什么、有什么坑？（3.2~3.3）
3. matmul 遇到高维张量按什么规则计算？为什么多头计算前要把形状摆成 $(B, h, N, d)$？（3.4~3.5）

已经熟悉 stride 机制的读者可以直接跳到 3.4 节。

### 3.1 底层模型：storage + stride

理解一切形状操作，只需要一个心智模型：**张量 = 一维连续内存（storage）+ 一套"解读方式"**。不管逻辑上有多少维，数据实际都平铺在一维内存里；"多少维、什么形状"只是三个元信息定义出来的解读规则：

- `shape`：每个维度多长；
- `stride`（步长）：**沿某个维度前进 1 步，在一维内存里要跳过几个元素**；
- `offset`：从内存的第几个元素开始读。

逻辑坐标换算成内存位置的公式：

$$
\mathrm{addr}(i_0, i_1, \dots) = \mathrm{offset} + i_0 \cdot s_0 + i_1 \cdot s_1 + \cdots
$$

```python
x = torch.arange(12).view(3, 4)
# 内存里就是 [0,1,...,11]，解读成 3×4：
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]
print(x.stride())            # (4, 1)：行方向走 1 步跳 4 个元素，列方向跳 1 个
print(x.is_contiguous())     # True：stride 恰为行主序默认值，内存顺序 = 逻辑顺序
```

有了这个模型，所有形状操作就只分两类：

| 类别 | 操作 | 做了什么 | 代价 |
|------|------|---------|------|
| **视图类** | `view`、`transpose`、`permute`、`expand`、切片、`squeeze`/`unsqueeze` | 只改 shape/stride/offset，数据不动 | 零拷贝，O(1) |
| **拷贝类** | `contiguous`、`repeat`、（必要时的）`reshape` | 分配新内存、搬运数据 | 一次读 + 一次写 |

### 3.2 四个核心操作

**`view`：重新分组，数据不动。** 把同一段内存按新形状重新切分。要求元素总数不变、且新形状能和当前内存布局对得上（通常要求张量是 contiguous 的）。`-1` 表示这一维自动推断：

```python
y = x.view(2, 6)                      # 重新分组：[[0..5], [6..11]]
z = x.view(4, -1)                     # -1 自动推断为 3
print(x.data_ptr() == y.data_ptr())   # True：同一块存储，零拷贝
```

**`transpose(i, j)`：交换两个维度，只改 stride。** 数据一个字节都不动，但从此逻辑顺序和内存顺序就错开了：

```python
t = x.transpose(0, 1)                 # shape (3,4)→(4,3)，stride (4,1)→(1,4)
print(t.stride())                     # (1, 4)
print(t.is_contiguous())              # False：逻辑顺序 ≠ 内存顺序
print(t.data_ptr() == x.data_ptr())   # True：零拷贝
```

二维张量可以简写成 `.t()`；`.mT` 等价于 `transpose(-1, -2)`（转置最后两维，attention 里最常用）。

**`permute`：一次性重排所有维度。** 是 `transpose` 的一般化，同样零拷贝：

```python
p = torch.randn(2, 3, 4, 5).permute(0, 2, 1, 3)
print(p.shape, p.stride())            # torch.Size([2, 4, 3, 5]) (60, 5, 20, 1)
```

**`reshape`：更宽容的 view。** 布局对得上时内部就是 `view`（零拷贝）；对不上时（比如转置之后）会**悄悄拷贝一份**再变形。经验法则：确定不需要拷贝时用 `view`（条件不满足它会报错提醒你），只想要结果、不在乎是否拷贝时用 `reshape`。

### 3.3 高频陷阱

**陷阱一：view 不能代替 transpose。** 两者结果完全不同。可以想象元素是串在一条线上的珠子：view 是"同一串珠子换个方式分组"，transpose 是"换一条数珠子的路线"：

```python
a = torch.arange(6).view(2, 3)   # [[0,1,2],[3,4,5]]
print(a.view(3, 2))              # [[0,1],[2,3],[4,5]]  重新分组，元素顺序未变
print(a.t())                     # [[0,3],[1,4],[2,5]]  真正的转置
```

**陷阱二：转置之后 view 会报错。** `view` 要求新形状能在旧 stride 上零拷贝地表达出来，转置后的布局做不到：

```python
# a.t().view(6)                 # RuntimeError: view size is not compatible ...
print(a.t().contiguous().view(6))   # 方案一：显式物化   → [0, 3, 1, 4, 2, 5]
print(a.t().reshape(6))             # 方案二：reshape 代劳（内部同样拷贝了）
```

这个报错其实是好事：它强迫你意识到这里必须发生一次真实的数据搬运。

**陷阱三：`expand` 和 `repeat` 不一样。** `expand` 把长度为 1 的维度"虚拟复制"——把 stride 置 0，零拷贝，但**只读**（多个逻辑位置指向同一块内存，往里写会出诡异结果）；`repeat` 是真实复制：

```python
b = torch.tensor([[1.], [2.], [3.]])   # (3, 1)
print(b.expand(3, 4).stride())          # (1, 0)：stride=0，4 列读的是同一个元素
print(b.repeat(1, 4).stride())          # (4, 1)：物化后的正常布局
```

"stride=0"这个机制值得记住：**广播（broadcast）的本质就是 stride 为 0 的虚拟维度**。下一节 matmul 的批量广播、第 5 章 MQA 里"多个头共享一份 K/V"，底层都是它。

### 3.4 matmul 的批量语义：只乘最后两维

`torch.matmul`（就是 `@` 运算符）处理高维张量的规则只有一条：

> **最后两维做矩阵乘法；前面的所有维度一律当作批量维，每个批量独立做一次乘法；批量维之间遵循广播规则。**

$$
(\ldots, m, k) \times (\ldots, k, n) \to (\ldots, m, n)
$$

```python
A = torch.randn(2, 8, 5, 3)
B = torch.randn(2, 8, 3, 7)
print((A @ B).shape)        # torch.Size([2, 8, 5, 7])
# 语义：2×8 = 16 个独立的 (5,3)@(3,7)，一次批量完成

C = torch.randn(2, 1, 3, 7)
print((A @ C).shape)        # torch.Size([2, 8, 5, 7])
# 批量维 (2,8) 与 (2,1) 广播：C 的 1 份矩阵被 8 个批量共享（3.3 节的 stride=0 机制）
```

批量维广播的**精确规则**：把最后两维去掉之后，两边剩下的批量维**右对齐、一维一维地比较**，每一对必须满足"相等，或者其中一边是 1（或这一维干脆没有）"。可以同时广播多个维度，两边的维数也可以不一样：

```python
print((A @ torch.randn(1, 1, 3, 7)).shape)     # (2,8,5,7)：两个批量维同时广播
print((A @ torch.randn(3, 7)).shape)           # (2,8,5,7)：批量维整体缺失，视为全 1
print((A @ torch.randn(6, 1, 8, 3, 7)).shape)  # (6,2,8,5,7)：C 维数更多，反过来广播 A

# 反例：A @ torch.randn(1, 4, 3, 7) 会报错——逐维比较 (2,8) vs (1,4)，
# 第二维 8 vs 4 既不相等、也没有一边是 1。广播不是"总数凑得上就行"：
# 长度 1 展开到 8 语义唯一（stride=0，8 份读同一个），4 展开到 8 则无法定义
```

在 GPU 上，这样一次调用会映射成 **batched GEMM**（cuBLAS 的 strided-batched 接口）：16 个小矩阵乘打包进一个 kernel 一起算，比循环 16 次快得多。

### 3.5 为什么 (B, N, h, d) 要变成 (B, h, N, d)

多头注意力做完投影、拆完头之后，张量形状是 $(B, N, h, d)$。而我们**想要**的计算是：每个 batch 的每个头，各自拿自己的 $N \times d$ 矩阵去算 $Q_i K_i^\top$——也就是 $B \times h$ 个独立的矩阵乘法。

对照 3.4 节的规则（matmul 只乘最后两维），两种形状的含义天差地别：

| 形状 | 最后两维 | matmul 实际算的是什么 |
|------|---------|------------------|
| $(B, N, h, d)$（不变换，**错误**） | $(h, d)$ | $B \times N$ 个 $(h,d) \times (d,h)$——算的是**同一个 token 内部、头和头之间**的相似度，根本不是注意力 |
| $(B, h, N, d)$（transpose 之后，正确） | $(N, d)$ | $B \times h$ 个 $(N,d) \times (d,N)$——每个头独立的 token 之间的注意力 ✓ |

所以所有 attention 代码里都有的那行惯用写法，做的事情就是一句话：**把希望独立并行的维度（$B$、$h$）摆到前面当批量维，把真正参与矩阵乘的维度（$N$、$d$）摆到最后两位**：

```python
B_, N, h, d = 2, 10, 8, 8
q = torch.randn(B_, N, h * d)              # 投影结果 (B, N, D)
q = q.view(B_, N, h, d).transpose(1, 2)    # 拆头 + 挪维 → (B, h, N, d)
print(q.shape)                             # torch.Size([2, 8, 10, 8])
```

再补三个细节，把整条流水线里"哪一步拷贝了数据"彻底讲清楚：

1. **拆头为什么用 `view`**：$(B, N, D)$ 的内存里，$D$ 这一维本来就是 `[头0 的 d 个数 | 头1 的 d 个数 | ...]` 连续排着的。`view(B, N, h, d)` 恰好就是按这个既有布局分组——零拷贝，而且语义正确；
2. **挪维为什么用 `transpose`**：交换头维和序列维必须改变遍历路线，`view` 做不到（陷阱一）；`transpose` 只改 stride，也是零拷贝。结果不连续没关系——matmul 认 stride，能直接算；
3. **合头是全流程唯一一次真实拷贝**：输出 $(B, h, N, d)$ 要变回 $(B, N, D)$，做法是先 `transpose(1, 2)` 再 `reshape(B, N, D)`。transpose 之后张量不连续，`view` 会报错，`reshape` 在这里会悄悄物化一份（等价于 `.contiguous().view()`）。**整条 attention 形状流水线里，只有这一步真的搬了数据。**

下一章 MHA 的代码就是这套流水线的完整版本；第 5 章 MQA 里 $(B, h, N, d) \times (B, 1, d, M)$ 的写法，则是 3.4 节批量广播的直接应用。

### 3.6 速查表

| 操作 | 作用 | 拷贝？ | 典型场景 |
|------|------|--------|---------|
| `view(shape)` | 重新分组解读 | 否（不行则报错） | 拆头 $(B,N,D) \to (B,N,h,d)$ |
| `reshape(shape)` | 同 view，语义宽容 | 必要时是 | 合头（转置后） |
| `transpose(i, j)` | 交换两维 | 否（结果常非连续） | $(B,N,h,d) \to (B,h,N,d)$、`k.transpose(-1,-2)` |
| `permute(...)` | 任意重排全部维 | 否 | 多维一次到位 |
| `contiguous()` | 物化当前视图 | 是 | 转置后需要连续内存时 |
| `expand` | 虚拟广播（stride=0） | 否（只读） | 共享 K/V、掩码广播 |
| `repeat` | 真实平铺复制 | 是 | 需要可写的复制 |

三条判断口诀：

- **想要新形状**：布局兼容用 `view`，不确定用 `reshape`；
- **想换维度顺序**：两个维度用 `transpose`，多个用 `permute`（本身免费，但代价会记到下一个需要连续内存的操作头上）;
- **要喂给 matmul**：独立并行的维度摆前面当批量，参与乘法的两维摆最后。

---
## 第 4 章 MHA：多头注意力与 KV Cache

### 4.1 定义

MHA 的想法：与其让所有维度一起做一次注意力，不如把 hidden 维切成 $h$ 份，每份（叫一个"头"）拥有**独立的** Q/K/V 投影，在各自 $d$ 维的小空间里做一次 SDPA，最后把 $h$ 个结果拼起来、再过一个输出投影混合：

$$
Q = XW_q,\qquad K = XW_k,\qquad V = XW_v
$$

$$
\mathrm{head}_i = \mathrm{Attention}(Q_i, K_i, V_i) = \mathrm{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d}}\right)V_i,
\qquad i = 1, \dots, h
$$

$$
\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\, W_o
$$

| 符号 | 形状 | 含义 |
|------|------|------|
| $W_q, W_k, W_v, W_o$ | $D \times D$ | 四个投影参数矩阵 |
| $Q, K, V$ | $N \times D$ | 投影结果（还没拆头） |
| $Q_i, K_i, V_i$ | $N \times d$ | 第 $i$ 个头的三元组（把 $Q$ 沿列切成 $h$ 份，取第 $i$ 份） |
| $\mathrm{head}_i$ | $N \times d$ | 第 $i$ 个头的 SDPA 输出 |
| 输出 | $N \times D$ | 拼接后经 $W_o$ 混合各头的信息 |

计算分四步，形状变化如下（带 batch 维，对应 3.5 节的流水线）：

| 步骤 | 操作 | 形状变化 | 拷贝？ |
|------|------|---------|--------|
| ① 投影 | 三个 `nn.Linear` | $(B,N,D) \to (B,N,D)$ | GEMM |
| ② 拆头 | `view` + `transpose` | $(B,N,D) \to (B,h,N,d)$ | 零拷贝 |
| ③ 每头 SDPA | 2.2 节的函数 | $(B,h,N,d) \to (B,h,N,d)$ | batched GEMM |
| ④ 合头 + 输出投影 | `transpose` + `reshape` + `nn.Linear` | $(B,h,N,d) \to (B,N,D)$ | reshape 处一次拷贝 |

为什么要多头？因为不同的头在训练中会学到不同的关注模式（有的管语法结构、有的管指代关系、有的管相对位置……），多个视角并行，表达力更强。另外注意：工程实现上并不会真的循环 $h$ 次——按 3.5 节的摆法，一次批量 matmul 就把所有头一起算完了。

### 4.2 实现（含 KV Cache）

```python
class MultiHeadAttention(nn.Module):
    """MHA：h 个头各有独立的 K/V（g = h）"""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, attn_mask=None, past_key_value=None, use_cache=False):
        B, N, _ = x.shape
        # ①② 投影 + 拆头：(B, N, D) → (B, h, N, d)
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # KV Cache：拼接历史 K/V（decode 时 N=1，k/v 只算了新 token 的那一列）
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)   # (B, h, M, d)，M = 历史+当前
            v = torch.cat([past_key_value[1], v], dim=2)
        present = (k, v) if use_cache else None

        # ③ 每头独立 SDPA：B×h 个 (N,d)@(d,M)
        out = scaled_dot_product_attention(q, k, v, attn_mask)   # (B, h, N, d)
        # ④ 合头（此处发生全流程唯一一次拷贝）+ 输出投影
        out = out.transpose(1, 2).reshape(B, N, -1)              # (B, N, D)
        out = self.o_proj(out)
        return (out, present) if use_cache else out
```

### 4.3 KV Cache：为什么可以缓存

把 1.3 节的论证再展开一点。在因果掩码下，位置 $t$ 的输出只依赖 $k_{\le t}, v_{\le t}$；而 $k_j, v_j$ 只是第 $j$ 个 token 自己的线性投影，**和后面生成的任何 token 都没有关系**——算出来一次，永远有效。所以自回归生成分成两个阶段：

- **Prefill（预填充）**：用户的 prompt 一次性整段算完，同时把全部 K/V 存进 Cache；
- **Decode（逐词生成）**：每步只为新 token 算一行 $q, k, v$；新的 $k/v$ 追加进 Cache，$q$ 和整个 Cache 做注意力。

这样每步的计算量从 $O(t^2)$ 降到 $O(t)$，代价就是 1.3 节算过的那笔显存开销。

下面验证"逐步 decode 的结果 ≡ 一次 prefill 的结果"——这是检验任何 KV Cache 实现是否正确的标准方法：

```python
torch.manual_seed(0)
mha = MultiHeadAttention(64, 4).to(device)
x = torch.randn(2, 6, 64, device=device)

full = mha(x, attn_mask=causal_mask(6, device))   # 一次 prefill（带因果掩码）

cache, steps = None, []
for t in range(6):                             # 逐 token decode（无需掩码：历史都可见）
    step, cache = mha(x[:, t:t+1], past_key_value=cache, use_cache=True)
    steps.append(step)

print(torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5))   # True
```

（decode 时为什么不需要掩码？因为每步只有 1 个 query，它面对的 key 全都是历史，本来就全部可见。）

### 4.4 MHA 的开销与遗留问题

| 项目 | 量 |
|------|-----|
| 参数量 | $4D^2$（四个投影矩阵） |
| 每 token 每层 KV Cache | $2 \times h \times d = 2D$ 个元素 |

注意 **Cache 的宽度和 hidden 一样宽**——这就是 1.3 节 344 GB 的来源。想给它瘦身，最自然的问题是：$h$ 份 K/V 真的每份都需要吗？下一章给出最激进的回答。

---

## 第 5 章 MQA：多查询注意力

### 5.1 定义：所有头共享一份 K/V

MQA 的观察是：多头的"多样性"主要由 query 承载，K/V 未必需要每个头都独立一份。于是它**保留 $h$ 份 query 投影，但 K/V 只投影一份**，所有头对着同一份 K/V 做注意力：

$$
Q = XW_q,\qquad K = XW_k,\qquad V = XW_v
\qquad\big(W_q \in \mathbb{R}^{D \times D},\ \ W_k, W_v \in \mathbb{R}^{D \times d}\big)
$$

$$
\mathrm{head}_i = \mathrm{softmax}\!\left(\frac{Q_i K^\top}{\sqrt{d}}\right)V,
\qquad K, V \in \mathbb{R}^{M \times d} \text{（只有一份，所有头共用）}
$$

和 MHA 相比，唯一的差别就是 $W_k, W_v$ 的输出宽度从 $D$ 缩到了 $d$。效果立竿见影：KV Cache 直接缩小 $h$ 倍（LLaMA-70B 规格下就是 64 倍）。

### 5.2 实现

```python
class MultiQueryAttention(nn.Module):
    """MQA：所有头共享一份 K/V（g = 1）"""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.head_dim)    # 输出只有 d 宽！
        self.v_proj = nn.Linear(hidden_size, self.head_dim)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, attn_mask=None, past_key_value=None, use_cache=False):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, 1, self.head_dim).transpose(1, 2)   # (B, 1, N, d)
        v = self.v_proj(x).view(B, N, 1, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present = (k, v) if use_cache else None

        # (B, h, N, d) @ (B, 1, d, M)：头维 1 自动广播到 h，无需显式复制（3.4 节）
        out = scaled_dot_product_attention(q, k, v, attn_mask)
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.o_proj(out)
        return (out, present) if use_cache else out
```

实现要点：K/V 的头维是 1，和 query 的 $(B, h, N, d)$ 做 matmul 时**批量维会自动广播**（3.4 节的规则、3.3 节的 stride=0 机制）——不需要用 `expand`/`repeat` 复制数据。"共享"这个概念，在张量语义上就这样直接表达出来了。

### 5.3 开销与遗留问题

| 项目 | 量 |
|------|-----|
| 参数量 | $\big(2 + 2/h\big)D^2$（$W_k, W_v$ 从 $D{\times}D$ 缩到 $D{\times}d$） |
| 每 token 每层 KV Cache | $2d$ 个元素（MHA 的 $1/h$） |

但天下没有免费的午餐：GQA 论文的对照实验显示，MQA 相比 MHA 有可以感知到的效果下降，而且训练更容易不稳定——**压得太狠了**。那能不能在"$h$ 份"和"1 份"之间找一个中间点？这就是下一章。

---

## 第 6 章 GQA：分组查询注意力

### 6.1 定义：在 MHA 与 MQA 之间取中间点

GQA 把 $h$ 个 query 头分成 $g$ 组（要求 $g$ 整除 $h$），**每组共享一份 K/V**。记头 $i$ 所属的组为 $\mathrm{grp}(i) = \lceil i / (h/g) \rceil$：

$$
\mathrm{head}_i = \mathrm{softmax}\!\left(\frac{Q_i\, K_{\mathrm{grp}(i)}^\top}{\sqrt{d}}\right) V_{\mathrm{grp}(i)},
\qquad W_k, W_v \in \mathbb{R}^{D \times g d}
$$

$g$ 就像一个可以连续调节的旋钮，两端恰好就是前两章的结构：

```
g = h ────────────── 1 < g < h ────────────── g = 1
 MHA                    GQA                    MQA
每头一份 KV          每组一份 KV             全体一份 KV
Cache = 2hd          Cache = 2gd             Cache = 2d
质量最好             质量 ≈ MHA               质量有损
```

论文结论：$g$ 取 8 左右，质量就能拉回到接近 MHA 的水平，同时保住绝大部分压缩收益。LLaMA-2-70B 选 $g = 8$ 还有一层部署上的考虑：8 组 KV 恰好对应单机 8 卡的张量并行——**每张卡独占一组完整的 K/V**，attention 计算不需要跨卡通信。

### 6.2 实现

```python
class GroupedQueryAttention(nn.Module):
    """GQA：h 个 query 头分 g 组共享 K/V（g = h 即 MHA，g = 1 即 MQA）"""

    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)   # g 份
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, attn_mask=None, past_key_value=None, use_cache=False):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:                 # Cache 里只存 g 份
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        present = (k, v) if use_cache else None

        # 把 g 份 K/V 逐组复制到 h 份，对齐 query 的头数（repeat_kv）
        rep = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(rep, dim=1)            # (B, g, M, d) → (B, h, M, d)
        v = v.repeat_interleave(rep, dim=1)

        out = scaled_dot_product_attention(q, k, v, attn_mask)
        out = out.transpose(1, 2).reshape(B, N, -1)
        out = self.o_proj(out)
        return (out, present) if use_cache else out
```

关于 `repeat_kv`（把 g 份复制成 h 份）这一步，三点说明：

- `repeat_interleave` 会**真实复制数据**（HF Transformers 的教学实现也是这么写的）。想零拷贝也可以：把 $q$ 重排成 $(B, g, h/g, N, d)$，和 $(B, g, 1, M, d)$ 的 K/V 广播相乘——数学上完全等价；
- PyTorch 2.5+ 的 `F.scaled_dot_product_attention(..., enable_gqa=True)` 可以直接接受 $g \ne h$ 的 K/V，内部完成分组匹配，不用手动 repeat；
- 生产推理 kernel（FlashAttention/PagedAttention）在索引层面直接让 $h/g$ 个 query 头去读同一块 KV Cache，**从头到尾不复制**——复制只存在于教学代码里。

### 6.3 从 MHA 检查点低成本改造：均值池化初始化

GQA 论文还有一个很实用的贡献：**不用从头训练**。做法是把已有 MHA 检查点里每组内的 $h/g$ 份 $W_k, W_v$ **按组取平均**，合成一份，作为 GQA 的初始化；然后用大约 5% 的原始预训练计算量继续训练（叫 uptraining），就能基本追平原模型。这也是"把开源 MHA 模型改造成 GQA 版"的标准做法。

### 6.4 开销与遗留问题

| 项目 | 量 |
|------|-----|
| 每 token 每层 KV Cache | $2gd$ 个元素（MHA 的 $g/h$） |

到这里，"减少 K/V 份数"这条路已经走到头了：份数从 $h$ 到 $g$ 再到 1，往下没有空间了。**还想继续压缩，就必须换一个思路**——不减份数，而是把每份的表示本身压小。这就是 MLA。

---
## 第 7 章 MLA：多头潜在注意力

### 7.1 核心思想：不是共享，而是压缩

DeepSeek-V2 提出的 MLA（Multi-head Latent Attention，多头潜在注意力）换了一个方向做减法：**份数不减**（每个头仍然有自己的 K/V），但 K/V 不直接存进缓存，而是做**低秩压缩**——把每个 token 的 K/V 信息压进一个小向量 $c^{KV}$（叫 latent，潜在向量），缓存里只存这个小向量，需要用的时候再投影"解压"回来。可以粗略地类比成：缓存里存的不是原始文件，而是压缩包。

对第 $t$ 个 token 的隐状态 $x_t \in \mathbb{R}^{D}$（提醒一个记号差异：前几章写 $K = XW_k$，是"矩阵 $X$ 的每一行右乘 $W$"；本章跟随 DeepSeek 论文的写法 $c_t = W x_t$，是"列向量左乘 $W$"，所以本章 $W$ 的形状是"输出维 × 输入维"，两种写法互为转置，本质相同）：

$$
c_t^{KV} = W^{DKV} x_t
\qquad \text{（down：压缩。缓存的就是它）}
$$

$$
k_t^{C} = W^{UK} c_t^{KV},
\qquad
v_t^{C} = W^{UV} c_t^{KV}
\qquad \text{（up：需要时重建出每个头的 K/V）}
$$

query 也做同样的低秩分解（不过 query 不进缓存，这一支省的不是 Cache，而是参数量和训练时的激活显存）：

$$
c_t^{Q} = W^{DQ} x_t,
\qquad
q_t^{C} = W^{UQ} c_t^{Q}
$$

| 符号 | 形状 | 含义 | DeepSeek-V2 取值 |
|------|------|------|------------------|
| $c_t^{KV}$ | $d_c$ | K 和 V 共用的 latent 向量（**缓存对象**） | $d_c = 512 = 4d$ |
| $c_t^{Q}$ | $d_c'$ | query 的 latent | $d_c' = 1536$ |
| $W^{DKV}$ | $d_c \times D$ | KV 压缩（down）投影 | — |
| $W^{UK}, W^{UV}$ | $hd \times d_c$ | K/V 重建（up）投影 | — |
| $W^{DQ}, W^{UQ}$ | $d_c' \times D$、$hd \times d_c'$ | query 的两段低秩投影 | — |
| $d_r$ | 标量 | 位置分支的每头维度（7.2 节） | $d_r = 64$ |

从另一个角度看，这相当于给 $W_k, W_v$ 做了一个共享底座的低秩分解。再配合一个关键技巧——"重建这一步可以被吸收掉"（7.3 节），推理时**根本不需要真的解压**。

### 7.2 解耦 RoPE：为什么需要单独一条位置分支

上面的压缩方案有一个致命的冲突：K 需要加 RoPE 位置编码，而 RoPE 是一个**随位置变化**的旋转变换 $R_t$。如果直接写成 $k_t = R_t\, W^{UK} c_t^{KV}$，那么位置 $s$ 的 query 和位置 $t$ 的 key 的分数里会出现：

$$
q_s^\top k_t = (c_s^Q)^\top \underbrace{(W^{UQ})^\top R_{s}^\top R_{t}\, W^{UK}}_{\text{随 } (s,t) \text{ 变化，无法预先合并}}\, c_t^{KV}
$$

中间那个矩阵随位置 $(s, t)$ 变化，没办法提前算好合并成一个固定矩阵——这就堵死了 7.3 节的吸收技巧。

MLA 的解法：把位置信息**拆出去，单独走一条小分支**。于是每个头的 q/k 都分成两段：

- **内容分支（nope，no position embedding）**：$q^C, k^C$，每头 $d$ 维，不带位置信息，走上面的低秩压缩；
- **位置分支（rope）**：$q^R_t = \mathrm{RoPE}(W^{QR} c^Q_t)$，每头一份；$k^R_t = \mathrm{RoPE}(W^{KR} x_t)$，**所有头共享一份**，维度只有 $d_r$（很小）。

完整的 q/k 由两段拼接而成，注意力分数自然拆成"内容分 + 位置分"两项（缩放分母相应变成 $\sqrt{d + d_r}$）：

$$
q_{t,i} = \big[\,q^C_{t,i};\ q^R_{t,i}\,\big],
\qquad
k_{t,i} = \big[\,k^C_{t,i};\ k^R_t\,\big],
\qquad
\mathrm{score} \propto q^{C\top} k^C + q^{R\top} k^R
$$

到这里，**缓存清单就定下来了：每个 token 每层只存两样东西——$c^{KV}$（$d_c$ 维）和 $k^R$（$d_r$ 维）**。代入 DeepSeek-V2 的实际配置（$h = 128$、$d = 128$、$d_c = 512$、$d_r = 64$）：

$$
\frac{\text{MLA Cache}}{\text{MHA Cache}} = \frac{d_c + d_r}{2\,h\,d} = \frac{576}{32768} \approx \frac{1}{57}
$$

这个缓存量相当于 $g = 2.25$ 的 GQA，但论文报告的效果反而**优于 MHA**（一种解释：低秩共享底座起到了某种正则化作用）。

### 7.3 吸收（absorption）：推理时永远不用重建 K/V

看到"低秩重建"，直觉反应是：那 decode 每一步岂不是都要把缓存解压回全尺寸的 K/V？其实不用——**两个 up 投影矩阵都可以被"吸收"进旁边的矩阵里**，解压这一步从来不会真的发生：

- **$W^{UK}$ 吸进 query 一侧**。看内容分数的展开：

$$
q^{C\top} k^C = (W^{UQ}c^Q)^\top (W^{UK} c^{KV}) = \big((W^{UK})^\top W^{UQ}\, c^Q\big)^\top c^{KV}
$$

  也就是说，把 $(W^{UK})^\top$ 提前乘进 query 的投影里，**分数就可以直接拿 $c^{KV}$ 来算**，根本不用重建 $k^C$；

- **$W^{UV}$ 吸进输出投影**。加权求和满足 $\sum_t p_t\, v_t = W^{UV} \sum_t p_t\, c_t^{KV}$——先对 latent 加权求和，最后再一起过 $W^{UV}$ 和 $W^O$。

所以 decode 每步实际读的就是 $c^{KV}$ 和 $k^R$ 本身。换个说法：**MLA 的 Cache 不是"压缩过、要解压的 K/V"，它就是 K/V 参与计算的最终形态**。这也反过来解释了 7.2 节为什么必须解耦 RoPE：吸收要求 up 投影是一个与位置无关的固定矩阵，所以位置信息必须拆到另一条分支上去。

### 7.4 实现

下面的教学版按 7.1/7.2 的公式直观展开（显式重建 K/V），但缓存语义和生产实现完全一致（只存 $c^{KV}$ 和 $k^R$）。吸收版只是纯代数变换，正确性由上一节的推导保证：

```python
def apply_rope(x, positions, base=10000.0):
    """RoPE（NeoX 半半式配对）。x: (B, heads, N, d)，d 为偶数；positions: (N,)"""
    d = x.size(-1)
    inv_freq = 1.0 / base ** (torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d)
    ang = positions.float()[:, None] * inv_freq[None, :]          # (N, d/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

class MultiHeadLatentAttention(nn.Module):
    """MLA 教学版（DeepSeek-V2 结构；默认超参为等比缩小的教学配置）
    缓存只有两样：c^KV (B, M, kv_lora_rank) 与共享的 k^R (B, 1, M, rope_dim)
    """

    def __init__(self, hidden_size, num_heads,
                 q_lora_rank=384, kv_lora_rank=128,
                 qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.dn, self.dr, self.dv = qk_nope_head_dim, qk_rope_head_dim, v_head_dim
        self.kv_rank = kv_lora_rank
        # Q 侧：低秩两段（c^Q = W_DQ x；[q^C, q^R 原料] = W_UQ c^Q）
        self.q_down = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_up   = nn.Linear(q_lora_rank, num_heads * (self.dn + self.dr), bias=False)
        # KV 侧：一次投影同时产出 c^KV 与共享 k^R 的原料
        self.kv_down = nn.Linear(hidden_size, kv_lora_rank + self.dr, bias=False)
        self.kv_up   = nn.Linear(kv_lora_rank, num_heads * (self.dn + self.dv), bias=False)
        self.o_proj  = nn.Linear(num_heads * self.dv, hidden_size, bias=False)

    def forward(self, x, attn_mask=None, past_cache=None, use_cache=False):
        B, N, _ = x.shape
        h, dn, dr, dv = self.num_heads, self.dn, self.dr, self.dv
        past_len = past_cache[0].size(1) if past_cache is not None else 0
        pos = torch.arange(past_len, past_len + N, device=x.device)   # 绝对位置（接着缓存数）

        # ---- Q：低秩重建，拆成内容/位置两段，位置段加 RoPE ----
        q = self.q_up(self.q_down(x)).view(B, N, h, dn + dr).transpose(1, 2)
        q_nope, q_rope = q.split([dn, dr], dim=-1)     # (B, h, N, dn) / (B, h, N, dr)
        q_rope = apply_rope(q_rope, pos)

        # ---- KV：只产出 latent c^KV 与所有头共享的 k^R ----
        ckv, k_rope = self.kv_down(x).split([self.kv_rank, dr], dim=-1)
        k_rope = apply_rope(k_rope.unsqueeze(1), pos)  # (B, 1, N, dr)，头维为 1 = 共享

        # ---- KV Cache：只缓存这两样（7.2 节的清单）----
        if past_cache is not None:
            ckv    = torch.cat([past_cache[0], ckv], dim=1)      # (B, M, kv_rank)
            k_rope = torch.cat([past_cache[1], k_rope], dim=2)   # (B, 1, M, dr)
        present = (ckv, k_rope) if use_cache else None
        M = ckv.size(1)

        # ---- 教学版：从 latent 重建 k^C 与 v（生产推理走 7.3 节的吸收，不做这一步）----
        kv = self.kv_up(ckv).view(B, M, h, dn + dv).transpose(1, 2)
        k_nope, v = kv.split([dn, dv], dim=-1)
        k = torch.cat([k_nope, k_rope.expand(B, h, M, dr)], dim=-1)   # (B, h, M, dn+dr)
        qh = torch.cat([q_nope, q_rope], dim=-1)                      # (B, h, N, dn+dr)

        out = scaled_dot_product_attention(qh, k, v, attn_mask)      # 缩放 1/√(dn+dr)
        out = out.transpose(1, 2).reshape(B, N, h * dv)
        out = self.o_proj(out)
        return (out, present) if use_cache else out

# ---- 验证：逐步 decode ≡ 一次 prefill（RoPE 位置经 past_len 正确衔接）----
torch.manual_seed(0)
mla = MultiHeadLatentAttention(128, 4).to(device)
x = torch.randn(2, 6, 128, device=device)
full = mla(x, attn_mask=causal_mask(6, device))
cache, steps = None, []
for t in range(6):
    o, cache = mla(x[:, t:t+1], past_cache=cache, use_cache=True)
    steps.append(o)
print(torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5))   # True
print("cache: c^KV", tuple(cache[0].shape), " k^R", tuple(cache[1].shape))
```

拿这份代码和第 4 章的 MHA 对照，差异只有三处，而这三处正是 MLA 的全部要点：

1. **缓存的东西变了**：从 K/V 变成 $c^{KV}$ + 共享的 $k^R$；
2. **k/v 是重建出来的**：教学版显式重建，生产版通过吸收跳过这一步；
3. **每个头的 q/k 由内容段和位置段拼接而成**：缩放分母相应变为 $\sqrt{d_n + d_r}$。

---

## 第 8 章 横向对比与统一验证

### 8.1 总对比表

沿着主线，把前面各章的开销合到一张表里（$h$ = 头数、$g$ = KV 头数、$d$ = 头维、$D = hd$）：

| | MHA | MQA | GQA | MLA |
|--|-----|-----|-----|-----|
| K/V 份数 | $h$ | 1 | $g$ | 每头一份（由 latent 重建） |
| K/V 投影形状 | $D \times D$ | $D \times d$ | $D \times gd$ | 低秩两段（7.1 节） |
| 每 token 每层 Cache（元素） | $2hd$ | $2d$ | $2gd$ | $d_c + d_r$ |
| LLaMA-70B/DSV2 规格下的相对量 | 1× | 1/64 | 1/8（$g{=}8$） | ~1/57 |
| 质量 | 基准 | 有损 | ≈ MHA（$g{\ge}8$） | 报告 ≥ MHA |
| 特殊代价 | — | — | — | 结构复杂；RoPE 需解耦 |
| 代表模型 | GPT-2、LLaMA-1 | PaLM、StarCoder | LLaMA-2/3、Qwen2、Mistral | DeepSeek-V2/V3/R1 |

一句话总结整个谱系：**MQA/GQA 在"份数"这个维度做减法（共享），MLA 在"表示"这个维度做减法（低秩压缩）**。GQA 用一个旋钮 $g$ 覆盖了从 MHA 到 MQA 的整段光谱；MLA 则证明了压缩和质量可以兼得——代价是结构复杂得多，还需要解耦 RoPE、矩阵吸收这两个配套技巧。

### 8.2 统一验证：GQA 的两端就是 MHA 与 MQA

本文四个类的参数命名和形状是刻意对齐的，所以"GQA 两端退化成 MHA/MQA"这个谱系关系可以直接用**权重拷贝 + 输出逐位对拍**来验证：

```python
torch.manual_seed(0)
D, h = 64, 8
x = torch.randn(2, 10, D, device=device)
m = causal_mask(10, device)

mha = MultiHeadAttention(D, h).to(device)
mqa = MultiQueryAttention(D, h).to(device)
gqa_h = GroupedQueryAttention(D, h, num_kv_heads=h).to(device)   # g = h
gqa_1 = GroupedQueryAttention(D, h, num_kv_heads=1).to(device)   # g = 1

gqa_h.load_state_dict(mha.state_dict())   # 形状完全一致，直接加载
gqa_1.load_state_dict(mqa.state_dict())

print(torch.allclose(gqa_h(x, m), mha(x, m), atol=1e-6))   # True：GQA(g=h) ≡ MHA
print(torch.allclose(gqa_1(x, m), mqa(x, m), atol=1e-6))   # True：GQA(g=1) ≡ MQA
```

加上第 2、4、7 章的三个内嵌测试（手写 SDPA ≡ 官方 SDPA、MHA 与 MLA 的 decode ≡ prefill），全文代码构成一套自洽的验证链——把所有代码块按顺序拼进一个文件就可以直接运行。

### 8.3 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)（MHA / 缩放点积形式）
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)（MQA）
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)（MLA）
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)（注意力机制起源）
