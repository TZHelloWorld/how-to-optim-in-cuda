# RoPE（旋转位置编码）原理与实现解析

本文系统讲解 **RoPE（Rotary Position Embedding，旋转位置编码）**——当前主流大模型（LLaMA、Qwen、DeepSeek 等）的标准位置编码方案，由苏剑林等人在 2021 年的论文 [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) 中提出。

全文按"由浅入深"组织，共三条主线：

- **原理**（第 1～3 章）：从"为什么需要位置编码"出发，先在二维平面上把旋转的核心性质讲透，再推广到 $d$ 维的完整数学形式，给出全部公式与证明；
- **实现**（第 4 章）：以 SGLang 源码为例，把公式逐行对应到工程代码，并从性能视角分析这个算子；
- **扩展与验证**（第 5～6 章）：长上下文场景下的 RoPE 缩放家族（PI / NTK / YaRN / Llama-3），以及可直接运行的最小实现与性质验证。

RoPE 施加在 attention 的 $Q$、$K$ 上、发生在注意力计算之前。

---

## 目录

- [第 1 章 问题定义：为什么需要位置编码](#第-1-章-问题定义为什么需要位置编码)
- [第 2 章 RoPE 的核心思想：在二维平面上旋转](#第-2-章-rope-的核心思想在二维平面上旋转)
- [第 3 章 完整数学：从 2D 到 d 维](#第-3-章-完整数学从-2d-到-d-维)
- [第 4 章 工程实现：SGLang 源码解析](#第-4-章-工程实现sglang-源码解析)
- [第 5 章 长上下文扩展：RoPE 缩放家族](#第-5-章-长上下文扩展rope-缩放家族)
- [第 6 章 动手验证：最小实现与性质对拍](#第-6-章-动手验证最小实现与性质对拍)
- [第 7 章 总结与速查表](#第-7-章-总结与速查表)

---

## 第 1 章 问题定义：为什么需要位置编码

### 1.1 自注意力的置换不变性

Transformer 的自注意力机制本身是**置换不变的（permutation invariant）**——它把输入当成一个"无序集合"处理。形式化地看：设输入 token 序列的表示为 $X \in \mathbb{R}^{N \times d}$，$P$ 为任意置换矩阵（交换行的顺序），则：

$$
\mathrm{Attn}(PX) = P \cdot \mathrm{Attn}(X)
$$

推一遍即知：$Q$、$K$、$V$ 都是 $X$ 的逐行线性投影，输入置换后变为 $PQ$、$PK$、$PV$；分数矩阵 $S' = (PQ)(PK)^\top = P S P^\top$（行、列同时被置换）；softmax 逐行进行，与行置换可交换；最终 $O' = P O$。也就是说，**交换 token 的输入顺序，每个 token 得到的输出向量一个数都不变**，只是跟着换了个位置——模型完全"感觉不到"顺序变了。

但语言有序列性："猫吃鱼"和"鱼吃猫"含义完全不同。因此必须以某种方式把"每个 token 处于第几个位置"注入模型。

### 1.2 绝对位置编码及其局限

最早的方案是**绝对位置编码（APE）**：给第 $m$ 个位置准备一个位置向量 $p_m$，直接加在词向量上：

$$
x_m \leftarrow x_m + p_m
$$

$p_m$ 有两种来源：

- **正弦位置编码**（Transformer 原论文）：

$$
p_{m,\,2i} = \sin\!\left(\frac{m}{10000^{2i/d}}\right), \qquad
p_{m,\,2i+1} = \cos\!\left(\frac{m}{10000^{2i/d}}\right)
$$

- **可学习位置嵌入**（BERT/GPT 系）：$p_m$ 作为参数直接训练。

绝对位置编码的问题在于：位置信息以"加法"混入内容后，注意力分数展开为四项：

$$
(x_m + p_m)^\top W_q^\top W_k (x_n + p_n)
= \underbrace{x_m^\top W_q^\top W_k x_n}_{\text{内容-内容}}
+ \underbrace{x_m^\top W_q^\top W_k p_n}_{\text{内容-位置}}
+ \underbrace{p_m^\top W_q^\top W_k x_n}_{\text{位置-内容}}
+ \underbrace{p_m^\top W_q^\top W_k p_n}_{\text{位置-位置}}
$$

位置与内容纠缠在一起，且分数同时依赖绝对位置 $m$、$n$——而语言建模真正关心的往往是**相对关系**（"形容词修饰它后面的名词"与这对词出现在句首还是句尾无关）。此外，可学习位置嵌入在超出训练长度时根本没有对应向量，无法外推。

### 1.3 目标：让分数只依赖相对位置

理想的位置编码应该满足：对位置 $m$ 处的 query 向量 $q$ 与位置 $n$ 处的 key 向量 $k$，各自施加某种只依赖自身位置的变换 $f$ 之后，它们的内积（即注意力分数）**只通过相对距离 $m - n$ 依赖位置**：

$$
\boxed{\ \langle f(q, m),\ f(k, n) \rangle = g(q,\ k,\ m - n)\ }
$$

这就是 RoPE 要精确求解的方程。RoFormer 论文证明了：在二维情形下，满足该方程的（一类自然的）解正是**旋转**——这就是第 2 章的主角。

---

## 第 2 章 RoPE 的核心思想：在二维平面上旋转

$d$ 维的 RoPE 本质上是许多个二维旋转的拼装，所以先把 $d = 2$ 的情形彻底讲清楚——本章结束时，RoPE 的核心性质就已经全部到手了。

### 2.1 二维旋转矩阵

把二维向量 $x = [x_1, x_2]^\top$ 逆时针旋转角度 $\alpha$，用旋转矩阵表示为：

$$
R(\alpha) =
\begin{bmatrix}
\cos\alpha & -\sin\alpha \\
\sin\alpha & \cos\alpha
\end{bmatrix},
\qquad
R(\alpha)\, x =
\begin{bmatrix}
\cos\alpha & -\sin\alpha \\
\sin\alpha & \cos\alpha
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
=
\begin{bmatrix}
x_1\cos\alpha - x_2\sin\alpha \\
x_2\cos\alpha + x_1\sin\alpha
\end{bmatrix}
$$

旋转矩阵有三条后文反复使用的性质：

| 性质 | 公式 | 含义 |
|------|------|------|
| 可加性 | $R(\alpha)\,R(\beta) = R(\alpha + \beta)$ | 连转两次 = 转角度之和 |
| 正交性 | $R(\alpha)^\top = R(\alpha)^{-1} = R(-\alpha)$ | 转置即反向旋转 |
| 保范性 | $\lVert R(\alpha)\,x \rVert = \lVert x \rVert$ | 旋转不改变向量长度 |

### 2.2 定义与关键性质：内积只看相对角度

RoPE 的定义极其简单：**位置 $m$ 处的向量，旋转 $m\theta$ 角度**（$\theta$ 为固定常数）：

$$
f(x, m) = R(m\theta)\, x
$$

验证它满足 1.3 节的目标方程——利用正交性与可加性：

$$
\langle f(q, m),\ f(k, n) \rangle
= q^\top R(m\theta)^\top R(n\theta)\, k
= q^\top R\big((n - m)\theta\big)\, k
$$

结果**只通过 $n - m$ 依赖位置**，绝对位置 $m$、$n$ 单独出现的项完全消失。目标方程被精确满足（不是近似），且由保范性，$\lVert f(q,m) \rVert = \lVert q \rVert$——位置编码不改变向量的模长，softmax 的数值尺度不受任何影响。

**几何直觉**：内积等于"模长之积 × 夹角余弦"。$q$ 转 $m\theta$、$k$ 转 $n\theta$ 后，两者模长不变，夹角恰好改变了 $(m-n)\theta$：

$$
\langle f(q, m),\ f(k, n) \rangle = \lVert q \rVert\, \lVert k \rVert\, \cos\big(\varphi + (m-n)\theta\big), \qquad \varphi = q,k \text{ 的原始夹角}
$$

**复数视角**（推导 $d$ 维形式与理解论文时更方便）：把二维向量 $[x_1, x_2]$ 看作复数 $x = x_1 + \mathrm{i}\,x_2$，旋转 $m\theta$ 就是乘上单位复数：

$$
f(x, m) = x\, e^{\mathrm{i} m\theta}, \qquad
\langle f(q,m), f(k,n) \rangle = \mathrm{Re}\!\left[\, q\, \overline{k}\, e^{\mathrm{i}(m-n)\theta} \,\right]
$$

$e^{\mathrm{i} m\theta}\cdot \overline{e^{\mathrm{i} n\theta}} = e^{\mathrm{i}(m-n)\theta}$，相对位置性质一望即知。

### 2.3 一个可手算的数值例子

取 $\theta = 30°$，$q = [1, 0]$，$k = [0, 1]$。按上面的展开（3.5 节有逐项推导），这一对向量的分数为：

$$
\langle f(q,m),\ f(k,n) \rangle = \sin\big((m-n)\cdot 30°\big)
$$

| $(m, n)$ | $m - n$ | 分数 |
|----------|---------|------|
| $(1, 0)$ | 1 | $\sin 30° = 0.5$ |
| $(3, 2)$ | 1 | $\sin 30° = 0.5$ |
| $(101, 100)$ | 1 | $\sin 30° = 0.5$ |
| $(2, 0)$ | 2 | $\sin 60° \approx 0.866$ |
| $(4, 0)$ | 4 | $\sin 120° \approx 0.866$ |
| $(14, 0)$ | 14 | $\sin 420° = \sin 60° \approx 0.866$ |

前三行印证核心性质：**把两个位置整体平移（$1{-}0$、$3{-}2$、$101{-}100$），分数完全不变**——分数是相对距离的函数。

后三行则暴露了单一频率的"分不清"问题，但要小心区分两层：

- **浅层巧合（距离 2 vs 4）**：$\sin 60° = \sin 120°$ 使两者分数相同——这是本例特殊取 $q$、$k$（分数恰好只剩 $\sin$ 项）造成的巧合。一般的 $q$、$k$ 下分数还含 $\cos$ 项，而 $\cos 60° \ne \cos 120°$，仍然分得开——毕竟 $R(60°)$ 和 $R(120°)$ 本来就不是同一个矩阵；
- **真正的歧义（距离 2 vs 14）**：$\theta = 30°$ 的旋转每 $360°/30° = 12$ 个 token 转回原点，$R(420°)$ 与 $R(60°)$ 是**同一个矩阵**——距离 2 和距离 14 对**任何** $q$、$k$ 都完全无法区分。单一频率是周期的，相对距离一旦超过一个波长就开始重复。

解决办法像时钟：只看分针，12:25 和 13:25 无法区分；加上时针，两个时刻立刻分开。要让"多根针"仍然整体混淆，必须**每根针同时回到原位**——距离差得是所有波长的**公倍数**。第 3 章会给 $d/2$ 组旋转配上几何级数的频率（波长从几个 token 到几万个 token），它们互相几乎不成整数比，公倍数远超任何实际上下文长度——这正是从 2D 走向 $d$ 维时多频率设计的动机。

---

## 第 3 章 完整数学：从 2D 到 d 维

### 3.1 分组旋转：分块对角矩阵

真实模型中每个注意力头的维度 $d$（head_dim）通常是 64、128 等。RoPE 把这 $d$ 个维度**两两一组，分成 $d/2$ 组，每组是一个独立的二维平面，各用各的旋转角速度 $\theta_i$**：

```
维度组 0：      (x_0, x_1)        旋转角度 = m·θ_0
维度组 1：      (x_2, x_3)        旋转角度 = m·θ_1
维度组 2：      (x_4, x_5)        旋转角度 = m·θ_2
...
维度组 d/2-1：  (x_{d-2}, x_{d-1}) 旋转角度 = m·θ_{d/2-1}
```

写成矩阵，就是一个**分块对角**的旋转矩阵（每块是一个 2.1 节的二维旋转）：

$$
R_{\Theta, m} =
\begin{bmatrix}
\cos m\theta_0 & -\sin m\theta_0 & & & & \\
\sin m\theta_0 & \cos m\theta_0 & & & & \\
& & \cos m\theta_1 & -\sin m\theta_1 & & \\
& & \sin m\theta_1 & \cos m\theta_1 & & \\
& & & & \ddots & \\
& & & & & \begin{matrix} \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\ \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1} \end{matrix}
\end{bmatrix}
$$

$d$ 维 RoPE 的完整定义即：

$$
f(x, m) = R_{\Theta, m}\, x
$$

（此处按 RoFormer 论文的**相邻配对** $(x_{2i}, x_{2i+1})$ 写出；工程实现中配对方式还有另一种等价布局，见 3.4 节。）

### 3.2 频率的几何级数设计

各组的角速度 $\theta_i$ 取**几何级数**（与 1.2 节正弦位置编码的频谱完全相同）：

$$
\theta_i = \mathrm{base}^{-2i/d}, \qquad i = 0, 1, \dots, \tfrac{d}{2}-1
$$

$\mathrm{base}$ 是超参数（即配置文件中的 `rope_theta`），LLaMA-1/2 取 10000，LLaMA-3 取 500000，Qwen2 取 1000000——base 越大低频越慢，天然支持更长的上下文。

每组的**波长**（旋转一整圈需要走过的 token 数）为：

$$
\lambda_i = \frac{2\pi}{\theta_i} = 2\pi \cdot \mathrm{base}^{2i/d}
$$

以 $d = 128$、$\mathrm{base} = 10000$ 为例（注意 $\theta_i = 10000^{-2i/128} = 10^{-i/16}$）：

| 组号 $i$ | $\theta_i$ | 波长 $\lambda_i$（token 数） | 角色 |
|---------|-----------|------------------------------|------|
| 0 | $1.0$ | $\approx 6.3$ | 最高频：几个 token 就转一圈，分辨相邻 token |
| 1 | $10^{-1/16} \approx 0.866$ | $\approx 7.3$ | 次高频 |
| 16 | $0.1$ | $\approx 63$ | 中频：分辨句子内部的距离 |
| 32 | $0.01$ | $\approx 628$ | 中低频：分辨段落级距离 |
| 63 | $10^{-63/16} \approx 1.15 \times 10^{-4}$ | $\approx 54410$ | 最低频：5 万 token 才转一圈，编码全局位置 |

### 3.3 频率直觉：一只多针时钟

把 $d/2$ 组旋转想象成一只有 $d/2$ 根针的时钟：**秒针（高频）分辨小间隔但很快重复，时针（低频）不重复但分辨率粗**。单独任何一根针都会产生 2.3 节那种"距离超过一个波长就重复"的周期性歧义；而多根转速呈几何级数的针**组合**起来，只有当距离差是**所有波长的公倍数**时才会整体重复——上表的 64 个波长（从约 6 到约 5.4 万 token）互相几乎不成整数比，公倍数远超任何实际上下文，因此在最长波长覆盖的范围内每个相对距离都能被唯一分辨——这与十进制计数"个位、十位、百位"是同一个原理。

同时，不同频率有天然分工：高频组主要贡献**近距离**（局部语法）的分辨能力，低频组主要贡献**远距离**（篇章结构）的感知——分数展开后（见 3.5 节）每组贡献一个 $\cos((m-n)\theta_i + \varphi_i)$ 项，高频项随距离剧烈振荡、低频项缓慢变化。

### 3.4 高效计算形式：不需要矩阵乘法

$R_{\Theta,m}$ 是极度稀疏的（每行只有 2 个非零元），实际计算从不构造矩阵，而是用逐元素运算。以第 $i$ 组的一对分量 $(x^{(1)}_i, x^{(2)}_i)$ 为例：

$$
\begin{aligned}
o^{(1)}_i &= x^{(1)}_i \cos(m\theta_i) - x^{(2)}_i \sin(m\theta_i) \\
o^{(2)}_i &= x^{(2)}_i \cos(m\theta_i) + x^{(1)}_i \sin(m\theta_i)
\end{aligned}
$$

把 $d/2$ 组拼起来，可以写成紧凑的向量形式：

$$
f(x, m) = x \odot \cos_m + \mathrm{rotate\_half}(x) \odot \sin_m
$$

其中 $\cos_m$、$\sin_m$ 是把各组的 $\cos(m\theta_i)$、$\sin(m\theta_i)$ 按分量排布好的向量，$\mathrm{rotate\_half}(x)$ 把每对分量变成 $(-x^{(2)}_i,\ x^{(1)}_i)$（即"转 90°"）。**每对分量只需 4 次乘、2 次加（全向量共 $2d$ 次乘、$d$ 次加）**，代价与一次向量加法同数量级。

"每对分量"具体是哪两个维度，业界有两种排列约定（数学上等价，只是维度的重排）：

| 风格 | 配对方式 | 使用模型 |
|------|---------|---------|
| **NeoX style**（半半式） | $(x_j,\ x_{j + d/2})$：前半段与后半段对应位置配对 | LLaMA、Qwen 等绝大多数 |
| **GPT-J style**（交错式） | $(x_{2j},\ x_{2j+1})$：相邻两个配对 | GPT-J、ChatGLM 等 |

两种风格各自都满足全部数学性质，但**配对方案不同意味着 $g$ 函数不同**——同一套权重必须自始至终使用训练时的风格，混用会静默地算错（这是移植模型时的高频 bug，见 4.4 节）。

### 3.5 相对位置性质的完整证明

沿用 2.1 节的正交性与可加性。因为 $R_{\Theta,m}$ 是分块对角的，性质逐块继承：

$$
R_{\Theta, m}^\top\, R_{\Theta, n} = R_{\Theta,\, n - m}
\qquad \text{（每个 } 2\times 2 \text{ 块上就是 } R(m\theta_i)^\top R(n\theta_i) = R((n-m)\theta_i) \text{）}
$$

于是对任意 $q$、$k$：

$$
\langle f(q, m),\ f(k, n) \rangle
= q^\top R_{\Theta,m}^\top R_{\Theta,n}\, k
= q^\top R_{\Theta,\,n-m}\, k
= g(q, k, m-n)
$$

**逐分量展开**（记 $\Delta = m - n$，第 $i$ 组的分量为 $q_1, q_2, k_1, k_2$，略去组号下标）：

$$
\begin{aligned}
\langle R(m\theta)\,q,\ R(n\theta)\,k \rangle
&= \big(q_1\cos m\theta - q_2\sin m\theta\big)\big(k_1\cos n\theta - k_2\sin n\theta\big) \\
&\quad + \big(q_2\cos m\theta + q_1\sin m\theta\big)\big(k_2\cos n\theta + k_1\sin n\theta\big) \\[4pt]
&= (q_1 k_1 + q_2 k_2)\big(\cos m\theta \cos n\theta + \sin m\theta \sin n\theta\big) \\
&\quad + (q_1 k_2 - q_2 k_1)\big(\sin m\theta \cos n\theta - \cos m\theta \sin n\theta\big) \\[4pt]
&= (q_1 k_1 + q_2 k_2)\cos(\Delta\theta) + (q_1 k_2 - q_2 k_1)\sin(\Delta\theta)
\end{aligned}
$$

（第二步为乘开后按 $q_i k_j$ 归并同类项；第三步用恒等式 $\cos(a{-}b) = \cos a\cos b + \sin a \sin b$ 与 $\sin(a{-}b) = \sin a\cos b - \cos a\sin b$。）全部 $d/2$ 组求和：

$$
\langle f(q,m),\ f(k,n) \rangle
= \sum_{i=0}^{d/2-1} \Big[ \big(q^{(1)}_i k^{(1)}_i + q^{(2)}_i k^{(2)}_i\big) \cos(\Delta\theta_i) + \big(q^{(1)}_i k^{(2)}_i - q^{(2)}_i k^{(1)}_i\big) \sin(\Delta\theta_i) \Big]
$$

用复数记号更紧凑——把第 $i$ 组看作复数 $q_i$、$k_i$，则：

$$
\langle f(q,m),\ f(k,n) \rangle = \mathrm{Re}\!\left[ \sum_{i=0}^{d/2-1} q_i\, \overline{k_i}\; e^{\mathrm{i}\Delta\theta_i} \right]
= \sum_{i=0}^{d/2-1} |q_i|\,|k_i| \cos\big(\varphi_i + \Delta\theta_i\big)
$$

（$\varphi_i$ 为该组内 $q_i$、$k_i$ 的原始夹角。）这就是 3.3 节"每组贡献一个余弦项"的出处。

### 3.6 一个常见误解：能否拆成 $(q \cdot k) \times f(m-n)$？

看到"分数只依赖相对位置"，容易误以为 RoPE 是"原始内积乘上一个只与距离有关的系数"。**然而并不是。** 3.5 节的 2D 展开有两项：

$$
\underbrace{(q_1 k_1 + q_2 k_2)}_{\text{原始内积 } q \cdot k} \cos(\Delta\theta)
\ +\
\underbrace{(q_1 k_2 - q_2 k_1)}_{\text{交叉项（2D 叉积）}} \sin(\Delta\theta)
$$

第二项的系数 $q_1 k_2 - q_2 k_1$ 是 $q$、$k$ 张成的平行四边形的**有向面积**（二维叉积），与内积 $q \cdot k$ 是相互独立的量——所以整体**无法**写成 $(q \cdot k) \cdot f(m-n)$ 的可分离形式。

正确的理解用矩阵形式最清楚：

$$
\langle f(q,m),\ f(k,n) \rangle = q^\top\, R_{\Theta,\,n-m}\, k
$$

RoPE 相当于在 $q$ 与 $k$ 的内积中间**插入了一个取决于相对距离的旋转矩阵**（不是标量）：这个矩阵改变了 $q$、$k$ 的相对取向，从而让分数随相对距离变化——**内容信息（$q$、$k$）与位置信息（$m-n$）交织在一起，不可分离**。这正是 RoPE 表达能力的来源：位置不是简单地缩放分数，而是参与了内容匹配的方向。

### 3.7 远程衰减性质

RoFormer 论文还证明了一条工程上重要的性质：在 3.2 节的几何级数频率下，把分数写成 3.5 节的复数求和 $\mathrm{Re}\big[\sum_i h_i\, e^{\mathrm{i}\Delta\theta_i}\big]$（$h_i = q_i \overline{k_i}$），用 Abel 变换可以给出其上界随 $|\Delta|$ 增大而**衰减**。直觉上：$\Delta$ 越大，各组相位 $\Delta\theta_i$ 在单位圆上散得越开，正负项相互抵消得越厉害——**相距很远的两个随机 token，注意力分数的期望幅度自然变小**。这与语言的局部性先验相符，也是几何级数频率（而非其他频率分布）被选中的理由之一。

---

## 第 4 章 工程实现：SGLang 源码解析

原理部分至此完备。本章把第 3 章的公式逐条对应到 SGLang 的真实源码，最后从性能视角审视这个算子。本章代码均位于 SGLang 仓库的 `python/sglang/srt/layers/rotary_embedding/` 目录：基类与线性缩放在 `base.py`，旋转原语在 `utils.py`，工厂函数 `get_rope()` 在 `factory.py` 中按 `rope_type` 分发，YaRN 与各模型变体在 `yarn.py`、`rope_variant.py`，多模态在 `mrope.py`，Triton 加速 kernel 在 `triton_kernels.py`。

### 4.1 RoPE 作用在哪个张量的哪个维度上

先把形状定位清楚。以 LLaMA-3-8B 为例：

```
hidden_dim = 4096,  num_heads = 32,  num_kv_heads = 8 (GQA),  head_dim = 4096 / 32 = 128
```

一个 token 经过 QKV 投影后：

```
Q: [1, 4096] → reshape → [1, 32, 128]     # 1 个 token、32 个 Q head、每 head 128 维
K: [1, 1024] → reshape → [1,  8, 128]     # GQA：8 个 KV head
```

**RoPE 的作用单位是每个 head 内部的 head_dim 维向量**（此处 128 维 = 64 个旋转组），不是整个 hidden_dim：

```
token 在 position = 5 时：
Q 的第 0 个 head: [q_0, ..., q_127] → 各组旋转 5·θ_i → [q'_0, ..., q'_127]
Q 的第 1 个 head: 同样用 position=5 的角度旋转
...（32 个 Q head、8 个 KV head 各自独立旋转，角度全部相同）
```

所有 head 查同一张 cos/sin 表、用相同的角度；但各 head 向量内容不同，旋转结果自然不同。对应代码中的形状处理：

```python
# radix_attention.py 中的 reshape
k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)   # [num_tokens, num_kv_heads, head_dim]

# rotary_embedding/base.py forward_native:
query = query.view(num_tokens, -1, self.head_size)     # [num_tokens, 32, 128]，逐 head 旋转
```

### 4.2 逆频率计算：`_compute_inv_freq()`

```python
def _compute_inv_freq(self, base):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim)
    )
    return inv_freq
```

这就是 3.2 节的公式 $\theta_i = \mathrm{base}^{-2i/d}$：`torch.arange(0, d, 2)` 生成 $[0, 2, 4, \dots, d-2]$，即公式里的 $2i$。以 `head_size=128, base=10000` 为例（$\theta_i = 10^{-i/16}$）：

```
inv_freq[0]  = 1 / 10000^(0/128)   = 1.0
inv_freq[1]  = 1 / 10000^(2/128)   ≈ 0.866
inv_freq[2]  = 1 / 10000^(4/128)   ≈ 0.750
...
inv_freq[63] = 1 / 10000^(126/128) ≈ 1.155e-4
```

（这几个数值可用 `10 ** (-i/16)` 心算校验。）

### 4.3 预计算 cos/sin 缓存：`_compute_cos_sin_cache()`

三角函数在 GPU 上由 SFU 计算、吞吐有限，而"位置 × 频率"的角度组合总共只有 `max_pos × d/2` 种——**离线全部算好存表，推理时查表**：

```python
def _compute_cos_sin_cache(self):
    inv_freq = self._compute_inv_freq(self.base)
    t = torch.arange(self.max_position_embeddings, dtype=torch.float)  # [0, 1, ..., max_pos-1]

    freqs = torch.einsum("i,j -> ij", t, inv_freq)  # 外积: [max_pos, rotary_dim/2]
    # freqs[pos][i] = pos * θ_i  ← 每个位置、每个维度组的旋转角度（3.1 节的 m·θ_i）

    cos = freqs.cos()                               # [max_pos, rotary_dim/2]
    sin = freqs.sin()                               # [max_pos, rotary_dim/2]
    cache = torch.cat((cos, sin), dim=-1)           # [max_pos, rotary_dim]，布局 [cos | sin]
    return cache
```

第 5 章所有缩放变体（PI/NTK/YaRN/Llama-3）的差异，最终都体现为**这张表的 `inv_freq` 或 `t` 算法不同**——表算好之后，后续的旋转应用代码完全一致。

### 4.4 旋转应用：`apply_rotary_emb()`

```python
def apply_rotary_emb(x, cos, sin, is_neox_style):
    """
    x:   [num_tokens, num_heads, head_size]
    cos: [num_tokens, head_size // 2]
    sin: [num_tokens, head_size // 2]
    """
    cos = cos.unsqueeze(-2).to(x.dtype)   # [num_tokens, 1, d/2]，沿 head 维广播
    sin = sin.unsqueeze(-2).to(x.dtype)

    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)  # NeoX：前半 x1 = x[0:d/2]，后半 x2 = x[d/2:d]
    else:
        x1 = x[..., ::2]                    # GPT-J：偶数位 x1 = [x_0, x_2, ...]
        x2 = x[..., 1::2]                   #        奇数位 x2 = [x_1, x_3, ...]

    o1 = x1 * cos - x2 * sin   # 3.4 节的旋转公式：o1 = x1·cos(mθ) - x2·sin(mθ)
    o2 = x2 * cos + x1 * sin   #                  o2 = x2·cos(mθ) + x1·sin(mθ)

    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)            # 拼回 [前半 | 后半]
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)  # 交错还原 [o1_0, o2_0, o1_1, o2_1, ...]
```

两种 `is_neox_style` 分支就是 3.4 节的两种配对约定。再次强调：**权重用哪种风格训练，推理就必须用哪种风格**——两种风格对同一输入给出不同的旋转结果，混用不报错但分数全错。

### 4.5 完整前向流程：`forward_native()`

```python
def forward_native(self, positions, query, key, offsets=None):   # 摘录自 base.py，签名有省略
    # ① 按每个 token 的位置查表，取出对应行的 cos/sin
    cos_sin = self.cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)          # 拆开 [cos | sin] 布局

    # ② 部分旋转：只旋转前 rotary_dim 维，剩余维度原样保留
    query_rot  = query[..., :self.rotary_dim]
    query_pass = query[..., self.rotary_dim:]    # 不参与旋转的"无位置"通道

    # ③ 对 Q 应用旋转
    query_rot = apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1)

    # ④ 对 K 应用相同的旋转
    key_rot  = key[..., :self.rotary_dim]
    key_pass = key[..., self.rotary_dim:]
    key_rot = apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
    key = torch.cat((key_rot, key_pass), dim=-1)

    return query, key
```

两个值得展开的设计点：

- **部分旋转（partial rotary）**：`rotary_dim` 可以小于 `head_size`（如 GPT-NeoX 系模型取 `rotary_pct = 0.25`，只旋转前 1/4 维度）。被旋转的维度携带位置信息，剩余维度成为纯内容通道——一种让模型自行分配"位置敏感度"的设计。LLaMA 系模型取 `rotary_dim = head_size`（全旋转）。
- **`positions` 与 `offsets`**：prefill 阶段 `positions` 是 `[0, 1, ..., L-1]` 的整段；decode 阶段每步只有当前 token 的一个位置值。多轮对话/投机解码等场景用 `offsets` 对位置做统一平移——由第 2 章的相对位置性质，整体平移不改变已有 token 之间的分数。

### 4.6 为什么只旋转 Q/K，不旋转 V

注意 `forward_native` 只处理 `query` 和 `key`。原因回到第 1 章的目标：位置信息只需要影响**注意力分数**（谁该关注谁），而分数完全由 $Q \cdot K$ 决定——旋转 Q/K 后，分数已经带上了相对位置。V 是被加权的"内容"，保持原样，attention 输出就是纯内容的加权和：

$$
o_m = \sum_n \mathrm{softmax}_n\big(\underbrace{q_m^\top R_{\Theta,\,n-m}\, k_n}_{\text{含相对位置}}\big)\cdot \underbrace{v_n}_{\text{纯内容}}
$$

若对 V 也旋转，输出会混入各 token 的**绝对**相位 $R_{\Theta,n} v_n$，反而破坏了"只依赖相对位置"的设计。

工程上这还带来一个重要便利：**K 旋转后直接存入 KV Cache**。decode 阶段历史 K 的旋转结果永远有效（它只依赖自己的绝对位置），每步只需旋转当前新 token 的 q/k 各一行——RoPE 的计算量在 decode 时是 $O(d)$ 每头每步，几乎可以忽略。

### 4.7 性能视角：一个典型的 memory-bound 小算子

给 RoPE 算一笔计算量与访存量的账（fp16，单 token 单 head，$d = 128$）：

| 项目 | 量 |
|------|-----|
| 计算量 | $6 \times d/2 \times 2 = 6d$ FLOP（4 乘 2 加 × d/2 组 × Q/K 两个）≈ 768 FLOP |
| 访存量 | 读 q,k（$2d \times 2$B）+ 读 cos/sin（$d \times 4$B）+ 写 q,k（$2d \times 2$B）≈ 1.5 KB |
| 算术强度（FLOP/Byte） | **< 1** —— 远低于 GPU 的算力/带宽平衡点（约数十 FLOP/B），深度 memory-bound，瓶颈在访存而非计算 |

结论与优化手段与逐元素算子一族相同：

1. **查表代替计算**（4.3 节）：省掉 SFU 三角函数，纯访存；
2. **Q/K 融合进同一个 kernel**：一次 kernel 启动、cos/sin 只读一遍——SGLang/vLLM 的 CUDA op `rotary_embedding` 与 `triton_kernels.py` 都是这么做的，且**原地（in-place）写回**，不分配新张量；
3. **向量化访存**：head_dim 连续、天然对齐，按 `float4`/128-bit 宽度读写；
4. **进一步融合**：把 RoPE 融进前面 QKV 投影的 epilogue 或后面 attention 的 prologue（如 FlashInfer 支持 fused-RoPE attention），彻底消掉这一遍独立的读写。RoPE 本身耗时极小，但在 decode 这种对 kernel 启动开销敏感的场景，省一次 launch 与一遍 q/k 读写仍然可观。

---

## 第 5 章 长上下文扩展：RoPE 缩放家族

### 5.1 外推问题：超出训练长度会发生什么

设模型训练时的最大长度为 $L_{\mathrm{train}}$（如 8K）。推理时若位置 $m > L_{\mathrm{train}}$，直接代入 RoPE 公式在数学上没有任何障碍——但效果会**急剧崩坏**（困惑度爆炸）。原因在 3.2 节的频谱上：

- **高频组**（$\lambda_i \ll L_{\mathrm{train}}$）：训练中已经转过成百上千圈，$[0, 2\pi)$ 的每个相位都见过——外推时没有新鲜相位，表现稳定；
- **低频组**（$\lambda_i \gtrsim L_{\mathrm{train}}$）：训练中只转过一小段弧（$m\theta_i \in [0,\ L_{\mathrm{train}}\theta_i]$），**外推位置对应的角度是模型从未见过的输入区域**，attention 分布随之失控。

所以问题精确地说是：**低频维度组的角度出界了**。所有扩展方法的本质，都是把出界的角度"塞回"训练时见过的范围，区别只在"怎么塞"——这也解释了为什么它们全部可以实现为"改一改 4.3 节那张 cos/sin 表"。

### 5.2 位置内插（PI / Linear Scaling）

[Position Interpolation（Chen et al., 2023）](https://arxiv.org/abs/2306.15595)：最直接的做法，把位置**等比压缩** $s$ 倍（$s = L_{\mathrm{new}} / L_{\mathrm{train}}$）：

$$
f'(x, m) = f\!\left(x, \frac{m}{s}\right)
\qquad \Longleftrightarrow \qquad
\theta_i' = \frac{\theta_i}{s}
$$

压缩后最大位置 $L_{\mathrm{new}}/s = L_{\mathrm{train}}$，所有角度都回到训练范围内。

- 优点：简单，配少量微调效果好；
- 缺点：**高频也被同等压缩**——相邻 token 的角度差从 $\theta_0$ 缩到 $\theta_0/s$，局部分辨率受损（模型分辨"隔 1 个 token"与"隔 2 个 token"变难），微调前困惑度上升明显。

对应 SGLang `base.py` 中的 `LinearScalingRotaryEmbedding`。

### 5.3 NTK-aware 缩放：只改 base

[NTK-aware scaling（bloc97, 2023）](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/)洞察到 PI 的问题在于"高频不该被压缩"，于是不改位置、只改 base：

$$
\mathrm{base}' = \mathrm{base} \cdot s^{\frac{d}{d-2}}
\qquad \Longrightarrow \qquad
\theta_i' = \mathrm{base}'^{-2i/d} = \theta_i \cdot s^{-\frac{2i}{d-2}}
$$

看两个端点就明白了它的精妙：

- $i = 0$（最高频）：$\theta_0' = \theta_0 = 1$，**完全不动**——局部分辨率无损；
- $i = \frac{d}{2}-1$（最低频）：$\theta' = \theta \cdot s^{-1}$，**恰好等价于内插 $s$ 倍**——出界角度被塞回去。

即"**高频外推、低频内插**"，且中间频率平滑过渡。无需微调也有不错的外推效果。**Dynamic NTK** 进一步让 $s$ 随当前实际序列长度动态调整（短序列不缩放、超长时按需放大），避免短序列被无谓劣化。

### 5.4 YaRN：按波长分段处理

[YaRN（Peng et al., 2023）](https://arxiv.org/abs/2309.00071)把"高频外推、低频内插"的思想做成显式的**分段规则**。定义每组的"训练长度/波长"比：

$$
r_i = \frac{L_{\mathrm{train}}}{\lambda_i}
\qquad \text{（该组在训练长度内转过的圈数）}
$$

用斜坡函数 $\gamma_i$ 把两种策略线性混合（默认 $\alpha = 1$、$\beta = 32$）：

$$
\theta_i' = \big(1 - \gamma_i\big)\, \frac{\theta_i}{s} + \gamma_i\, \theta_i,
\qquad
\gamma_i = \mathrm{clamp}\!\left(\frac{r_i - \alpha}{\beta - \alpha},\ 0,\ 1\right)
$$

- $r_i > \beta$（训练中已转 32 圈以上的高频组）：$\gamma_i = 1$，**保持原样**；
- $r_i < \alpha$（训练中不足 1 圈的低频组）：$\gamma_i = 0$，**完全内插** $\theta_i / s$；
- 中间：线性过渡。

YaRN 还引入**注意力温度校正**：把 cos/sin 缓存整体乘上

$$
m_s = 0.1 \ln s + 1
$$

（等效于给注意力分数乘 $m_s^2$），补偿长上下文下注意力分布变平的趋势。实测 YaRN 用极少的微调数据即可扩展到 64K/128K。对应 SGLang 的 `yarn.py`（`_yarn_get_mscale()` 即上式）。

### 5.5 Llama-3 风格缩放：按波长三段分治

Llama-3.1 采用与 YaRN 同源、但更直白的分段规则（`rope_variant.py`，配置默认 `factor = 8`、`low_freq_factor = 1`、`high_freq_factor = 4`、`original_max_position = 8192`）。按各组波长 $\lambda_i$ 与训练长度 $L$ 的关系分三段：

$$
\theta_i' =
\begin{cases}
\theta_i, & \lambda_i < \dfrac{L}{\beta_{\mathrm{high}}} & \text{（高频：不动）}\\[8pt]
\dfrac{\theta_i}{s}, & \lambda_i > \dfrac{L}{\beta_{\mathrm{low}}} & \text{（低频：完全内插）}\\[8pt]
(1 - \gamma_i)\dfrac{\theta_i}{s} + \gamma_i\,\theta_i, & \text{其间}, \quad \gamma_i = \dfrac{L/\lambda_i - \beta_{\mathrm{low}}}{\beta_{\mathrm{high}} - \beta_{\mathrm{low}}} & \text{（中频：平滑过渡）}
\end{cases}
$$

其中 $s = 8$、$\beta_{\mathrm{low}} = 1$、$\beta_{\mathrm{high}} = 4$。配合 $\mathrm{base} = 500000$ 的大基数，Llama-3.1 将上下文扩展到 128K。

### 5.6 缩放方法对比

| 方法 | 修改对象 | 公式核心 | 高频（局部分辨率） | 是否需要微调 | SGLang 实现 |
|------|---------|---------|------------------|-------------|-------------|
| PI / Linear | 位置 $m$ | $m \to m/s$ | 受损（同等压缩） | 建议 | `base.py` LinearScaling |
| NTK-aware | base | $\mathrm{base} \cdot s^{d/(d-2)}$ | 无损（$\theta_0$ 不变） | 可免 | `factory.py` 按 rope_type 分发 |
| Dynamic NTK | base（动态） | $s$ 随当前长度调整 | 无损 | 可免 | 同上 |
| YaRN | 逐组 $\theta_i$ | 斜坡混合 + 温度 $m_s$ | 无损（$\gamma = 1$ 段） | 少量 | `yarn.py` |
| Llama-3 | 逐组 $\theta_i$ | 按波长三段分治 | 无损 | 官方已训练 | `rope_variant.py` |

一条统一的观察：**全部方法都只改 4.3 节 cos/sin 表的生成逻辑，旋转应用（4.4 节）与 kernel（4.7 节）一行不用动**——这正是"预计算查表"这一实现架构的扩展性红利。

### 5.7 MRoPE：多模态的位置编码

Qwen2-VL 等多模态模型使用 **MRoPE（Multimodal RoPE，`mrope.py`）**：位置编号从标量 $m$ 变成三元组 $(t, h, w)$（时间、高、宽），把 head_dim 的旋转组切成三段，分别用三个坐标驱动——文本 token 三个坐标相同（退化回普通 RoPE），图像/视频 patch 则按其时空网格位置编码。核心的旋转数学与本文完全一致，只是"位置"的语义扩展了。

---

## 第 6 章 动手验证：最小实现与性质对拍

理论与源码都齐了，最后用一段可直接运行的代码把核心性质验证一遍——先用 PyTorch 把算法逻辑调对，再下沉到 kernel，是开发这类算子的推荐工作流。

```python
import torch

def rope(x, pos, base=10000.0):
    """最小 RoPE 实现（NeoX 半半式配对），x: [..., d]（d 为偶数），pos: 标量位置"""
    d = x.shape[-1]
    inv_freq = 1.0 / base ** (torch.arange(0, d, 2, dtype=torch.float32) / d)  # θ_i（4.2 节）
    ang = pos * inv_freq                        # 各组旋转角 m·θ_i，[d/2]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]  # NeoX 配对：(x_j, x_{j+d/2})
    return torch.cat([x1 * cos - x2 * sin,       # 3.4 节旋转公式
                      x2 * cos + x1 * sin], dim=-1)

torch.manual_seed(0)
d = 128
q, k = torch.randn(d), torch.randn(d)

# ---- 性质 1：分数只依赖相对距离 m-n（整体平移位置，分数不变）----
s1 = rope(q, 7)   @ rope(k, 3)     # m-n = 4
s2 = rope(q, 104) @ rope(k, 100)   # m-n = 4，整体平移 97
s3 = rope(q, 9)   @ rope(k, 3)     # m-n = 6（对照组）
print(f"{s1.item():.6f}  {s2.item():.6f}  {s3.item():.6f}")
# 前两个相等（差异仅浮点舍入，~1e-6），第三个明显不同

# ---- 性质 2：保范性（旋转不改变模长，2.1 节的正交性）----
print(q.norm().item(), rope(q, 12345).norm().item())   # 两个数相同

# ---- 性质 3：位置 0 是恒等变换（所有旋转角为 0）----
print((rope(q, 0) - q).abs().max().item())             # 0.0
```

三个断言分别对应第 2～3 章的三条数学性质。验证自己手写的 RoPE kernel 时，把 `rope()` 当参考实现对拍即可；再补两条对拍时的提醒：

- **风格必须对齐**：上面是 NeoX 式，若被测实现是 GPT-J 式（交错配对），两者输出**不相等**且分数也不同——对拍前先确认配对风格一致；
- **精度容差**：fp16 下查表 + 逐元素乘加的误差在 1e-3 量级，`allclose` 用 `rtol=1e-3` 起步；性质 1 的"平移不变"在低精度下同样只近似成立。

---

## 第 7 章 总结与速查表

### 7.1 一句话总结

**RoPE = 按 token 的位置，把 Q/K 向量的每个二维分组旋转 $m\theta_i$ 的角度**。位置越靠后转得越多；两个 token 做 attention 时，内积中的绝对相位相互抵消，只留下相对距离 $m-n$——位置信息就这样以乘性、保范、可外推的方式进入了注意力分数。

### 7.2 关键公式速查

| 概念 | 公式 | 出处 |
|------|------|------|
| 目标方程 | $\langle f(q,m), f(k,n)\rangle = g(q,k,m-n)$ | 1.3 节 |
| 2D 定义 | $f(x,m) = R(m\theta)\,x$，$R$ 为旋转矩阵 | 2.2 节 |
| $d$ 维定义 | $f(x,m) = R_{\Theta,m}\,x$（分块对角） | 3.1 节 |
| 频率 | $\theta_i = \mathrm{base}^{-2i/d}$，波长 $\lambda_i = 2\pi/\theta_i$ | 3.2 节 |
| 高效形式 | $x \odot \cos_m + \mathrm{rotate\_half}(x) \odot \sin_m$ | 3.4 节 |
| 相对性质 | $q^\top R_{\Theta,m}^\top R_{\Theta,n} k = q^\top R_{\Theta,n-m} k$ | 3.5 节 |
| 2D 展开 | $(q\!\cdot\!k)\cos\Delta\theta + (q_1k_2\!-\!q_2k_1)\sin\Delta\theta$，$\Delta = m\!-\!n$ | 3.5/3.6 节 |
| PI | $m \to m/s$ | 5.2 节 |
| NTK-aware | $\mathrm{base}' = \mathrm{base}\cdot s^{d/(d-2)}$ | 5.3 节 |
| YaRN | $\theta_i' = (1-\gamma_i)\theta_i/s + \gamma_i\theta_i$，温度 $m_s = 0.1\ln s + 1$ | 5.4 节 |

### 7.3 关键事实速查

| 事实 | 说明 | 出处 |
|------|------|------|
| 只旋转 Q/K，不旋转 V | 位置信息经分数传递即可；K 旋转后可直接进 KV Cache | 4.6 节 |
| 作用单位是 head_dim | 每个 head 独立旋转，所有 head 共用同一角度表 | 4.1 节 |
| 两种配对风格不可混用 | NeoX（半半）vs GPT-J（交错），须与训练一致 | 3.4/4.4 节 |
| cos/sin 离线预计算 | 所有缩放变体都只改这张表的生成逻辑 | 4.3/5.6 节 |
| RoPE 是 memory-bound 小算子 | AI < 1 FLOP/B；优化靠查表、Q/K 融合、原地写回、融进 attention | 4.7 节 |
| 外推崩坏的根源 | 低频组遇到训练时没见过的角度区间 | 5.1 节 |

### 7.4 参考资料

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)（RoPE 原论文）
- [Transformer 升级之路：博采众长的旋转式位置编码（苏剑林）](https://kexue.fm/archives/8265)
- [Extending Context Window via Position Interpolation](https://arxiv.org/abs/2306.15595)（PI）
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
- [十分钟读懂旋转编码（RoPE）](https://zhuanlan.zhihu.com/p/647109286)
