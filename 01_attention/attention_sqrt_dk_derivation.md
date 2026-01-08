# 为什么 Attention 要除以 √dₖ：完整数学推导

> 本笔记独立完整地推导 Scaled Dot-Product Attention 中缩放因子 $\sqrt{d_k}$ 的来由：它不是经验调参，而是对点积做**标准差归一化**的严格结果。笔记分三部分：先备齐所需的均值/方差运算规律，再给出推导，最后解释其对训练的意义。

---

## 1. 预备知识：均值与方差的运算规律

### 1.1 基本定义

**均值（期望）**：

$$E[X] = \mu = \frac{1}{n} \sum_{i=1}^{n} x_i$$

**方差**（两种等价形式，第二种在推导中更常用）：

$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

### 1.2 运算规律

**（1）常数运算**

$$E[c] = c, \qquad E[cX] = cE[X]$$

$$\text{Var}(c) = 0, \qquad \text{Var}(cX) = c^2\,\text{Var}(X)$$

注意最后一条：**常数是平方地放大方差的**——这正是后文"除以 $\sqrt{d_k}$ 恰好把方差除以 $d_k$"的原因。

**（2）加法**

$$E[X + Y] = E[X] + E[Y] \quad \text{（无条件成立）}$$

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\,\text{Cov}(X, Y)$$

当 $X$、$Y$ **独立**（协方差为 0）时：

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$$

**（3）乘法（独立变量）**

$$E[XY] = E[X] \cdot E[Y]$$

$$\text{Var}(XY) = E[X^2]E[Y^2] - (E[X]E[Y])^2$$

特别地，当 $E[X] = E[Y] = 0$ 时，$E[X^2] = \text{Var}(X)$、$E[Y^2] = \text{Var}(Y)$，于是：

$$\text{Var}(XY) = \text{Var}(X) \cdot \text{Var}(Y)$$

**（4）线性组合**

$$E[aX + bY] = aE[X] + bE[Y]$$

$$\text{Var}(aX + bY) = a^2\text{Var}(X) + b^2\text{Var}(Y) + 2ab\,\text{Cov}(X, Y)$$

---

## 2. 核心问题

Attention 的分数矩阵由 query 与 key 的点积构成：

$$\alpha = q_i \cdot k_j = \sum_{l=1}^{d_k} q_{il}\, k_{jl}$$

当维度 $d_k$ 很大时，点积由 $d_k$ 个随机项累加而成，**幅度随维度增长**。过大的分数送入 softmax 会导致输出饱和（趋近 one-hot）、梯度消失。问题是：点积的幅度到底随 $d_k$ 怎样增长？

## 3. 推导

**假设**：$q_i$ 与 $k_j$ 的各分量是独立随机变量，均值为 0、方差为 1（经 LayerNorm 与合理初始化后近似成立）。

**步骤 1：单项 $q_{il} k_{jl}$ 的方差。** 由 1.2 节规律（3），两因子独立且均值为 0：

$$\text{Var}(q_{il} k_{jl}) = E[q_{il}^2] \cdot E[k_{jl}^2] - 0^2 = 1 \times 1 = 1$$

**步骤 2：点积（求和）的方差。** $d_k$ 个乘积项相互独立，由规律（2）方差线性累加：

$$\text{Var}(\alpha) = \sum_{l=1}^{d_k} \text{Var}(q_{il} k_{jl}) = d_k$$

**步骤 3：标准差。**

$$\text{std}(\alpha) = \sqrt{\text{Var}(\alpha)} = \sqrt{d_k}$$

即：未缩放的点积典型幅度是 $\pm$ 数倍的 $\sqrt{d_k}$（ $d_k = 64$ 时约 $\pm 24$（3 个标准差），$d_k = 512$ 时约 $\pm 68$）。

**解决：除以 $\sqrt{d_k}$。** 由规律（1）的 $\text{Var}(cX) = c^2\text{Var}(X)$，取 $c = 1/\sqrt{d_k}$：

$$\text{Var}\!\left(\frac{\alpha}{\sqrt{d_k}}\right) = \frac{\text{Var}(\alpha)}{(\sqrt{d_k})^2} = \frac{d_k}{d_k} = 1$$

**关键结论：除以 $\sqrt{d_k}$ 后方差恒为 1，与维度无关。**

---

## 4. 为什么这很重要

1. **防止 softmax 饱和**：当一行分数中最大值比其余大出几十，$e^x$ 使最大项独占几乎全部权重，softmax 输出趋近 one-hot；数值上 fp32 的 $e^x$ 在 $x > 88.7$ 时直接上溢（fp16 在 $x > 11.1$）；
2. **保持梯度流动**：softmax 在饱和区的梯度 $\partial P/\partial S$ 趋近 0，训练停滞；方差为 1 时输入落在 softmax 的"敏感区"，梯度正常回传；
3. **维度无关性**：无论 $d_k = 64$ 还是 $512$，注意力分数尺度一致，同一套超参数（学习率、初始化）可跨模型规模复用。

**直观总结**：点积是 $d_k$ 个独立项之和，方差随 $d_k$ **线性**增长，因此除以标准差 $\sqrt{d_k}$（而不是 $d_k$）恰好完成标准化。
