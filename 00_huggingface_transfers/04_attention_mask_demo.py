#!/usr/bin/env python3
"""
04_attention_mask_demo.py - 注意力机制与掩码（Mask）完整演示

本脚本演示了 Transformer 注意力机制的核心计算过程：
1. Scaled Dot-Product Attention 的完整计算（NumPy 手算版）
2. Padding Mask 的作用和实现
3. Look-ahead Mask（因果掩码）的作用和实现
4. 带掩码的注意力计算
5. PyTorch 版本的注意力实现
6. Full-precision softmax 的作用

核心公式:
  Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

使用前请确保：
- pip install numpy torch
"""

import numpy as np
import torch
import torch.nn.functional as F


def numpy_softmax(x):
    """
    数值稳定的 softmax 函数（按行计算）

    通过减去每行最大值来防止指数溢出：
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def demo_basic_attention():
    """
    演示基本的 Scaled Dot-Product Attention 计算过程

    公式: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    步骤:
    1. 计算注意力分数: scores = Q @ K^T
    2. 缩放: scaled_scores = scores / sqrt(d_k)
    3. Softmax 归一化: weights = softmax(scaled_scores)
    4. 加权求和: output = weights @ V
    """
    print("=" * 60)
    print("[1] 基本 Scaled Dot-Product Attention 计算")
    print("=" * 60)

    # 定义 Q、K、V 矩阵（3 个位置，每个 4 维）
    Q = np.array([
        [1, 1, 2, 1],  # 位置 0 的 Query
        [2, 2, 2, 1],  # 位置 1 的 Query
        [1, 2, 3, 2]   # 位置 2 的 Query
    ], dtype=np.float32)

    K = np.array([
        [1, 1, 2, 1],  # 位置 0 的 Key
        [2, 2, 2, 1],  # 位置 1 的 Key
        [1, 2, 3, 2]   # 位置 2 的 Key
    ], dtype=np.float32)

    V = np.array([
        [1, 1, 2, 1],  # 位置 0 的 Value
        [2, 2, 2, 1],  # 位置 1 的 Value
        [1, 2, 3, 2]   # 位置 2 的 Value
    ], dtype=np.float32)

    d_k = Q.shape[-1]  # Key 的维度
    print(f"\nQ shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")
    print(f"d_k = {d_k}")

    # Step 1: 计算注意力分数 Q @ K^T
    scores = Q @ K.T
    print(f"\nStep 1 - 注意力分数矩阵 (Q @ K^T):")
    print(f"  shape: {scores.shape}")
    print(scores)

    # Step 2: 缩放（除以 sqrt(d_k)）
    # 目的：防止点积值过大导致 softmax 梯度消失
    scaled_scores = scores / np.sqrt(d_k)
    print(f"\nStep 2 - 缩放后 (/ sqrt({d_k}) = / {np.sqrt(d_k):.4f}):")
    print(np.round(scaled_scores, 4))

    # Step 3: Softmax 归一化
    attention_weights = numpy_softmax(scaled_scores)
    print(f"\nStep 3 - 注意力权重 (softmax):")
    print(np.round(attention_weights, 4))
    print(f"  每行之和: {np.round(attention_weights.sum(axis=-1), 4)}")  # 应该都是 1

    # Step 4: 加权求和
    output = attention_weights @ V
    print(f"\nStep 4 - 最终输出 (weights @ V):")
    print(np.round(output, 4))

    return output


def demo_padding_mask():
    """
    演示 Padding Mask（填充掩码）

    问题：不同长度的序列 padding 到相同长度后，
    padding 位置的 token 不应该参与注意力计算。

    解决：将 padding 位置的注意力分数设为 -inf，
    softmax 后这些位置的权重趋近于 0。
    """
    print("\n" + "=" * 60)
    print("[2] Padding Mask（填充掩码）")
    print("=" * 60)

    # 假设 batch 中有两个序列：
    # 序列 1: [Hello, World, !]          长度 3
    # 序列 2: [Hi]                       长度 1
    # Padding 后: [Hello, World, !, PAD, PAD] 和 [Hi, PAD, PAD, PAD, PAD]

    seq_len = 5
    print(f"\n序列 1 (实际长度 3): [Hello, World, !, PAD, PAD]")
    print(f"序列 2 (实际长度 1): [Hi, PAD, PAD, PAD, PAD]")

    # Padding Mask: True 表示有效位置，False 表示 padding
    padding_mask_1 = np.array([True, True, True, False, False])
    padding_mask_2 = np.array([True, False, False, False, False])

    print(f"\nPadding Mask 序列 1: {padding_mask_1}")
    print(f"Padding Mask 序列 2: {padding_mask_2}")

    # 生成注意力掩码矩阵（seq_len x seq_len）
    # 每个 query 位置都不应该关注 padding 位置的 key
    attn_mask_1 = padding_mask_1[np.newaxis, :].repeat(seq_len, axis=0)
    print(f"\n注意力掩码矩阵 (序列 1):")
    print(attn_mask_1.astype(int))
    print("(1=可关注, 0=被屏蔽)")


def demo_causal_mask():
    """
    演示 Look-ahead Mask（因果掩码 / Causal Mask）

    问题：在自回归生成中，位置 t 只能看到 t 及之前的 token，
    不能看到未来的 token（防止信息泄露）。

    实现：使用上三角矩阵，上三角部分设为 True（被屏蔽）。
    """
    print("\n" + "=" * 60)
    print("[3] Look-ahead Mask（因果掩码）")
    print("=" * 60)

    seq_len = 5

    # 方法 1: 使用 torch.triu
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    print(f"\n因果掩码矩阵 (1=被屏蔽):")
    print(causal_mask)
    print("""
含义:
  位置 0: 只能看自己           [0, 1, 1, 1, 1]
  位置 1: 能看位置 0, 1        [0, 0, 1, 1, 1]
  位置 2: 能看位置 0, 1, 2     [0, 0, 0, 1, 1]
  位置 3: 能看位置 0, 1, 2, 3  [0, 0, 0, 0, 1]
  位置 4: 能看所有位置          [0, 0, 0, 0, 0]
    """)

    # 方法 2: 使用布尔掩码（True = 可关注）
    causal_mask_bool = torch.tril(torch.ones(seq_len, seq_len)).bool()
    print(f"布尔因果掩码 (True=可关注):")
    print(causal_mask_bool.int())


def demo_masked_attention():
    """
    演示带掩码的完整注意力计算（NumPy 手算版）

    核心步骤:
    1. scores = Q @ K^T
    2. scores_masked = where(mask, scores, -1e9)  # 屏蔽位置设为极小值
    3. scaled_scores = scores_masked / sqrt(d_k)
    4. weights = softmax(scaled_scores)  # 极小值 -> 权重接近 0
    5. output = weights @ V
    """
    print("\n" + "=" * 60)
    print("[4] 带掩码的完整注意力计算")
    print("=" * 60)

    Q = np.array([
        [1, 1, 2, 1],
        [2, 2, 3, 1],
        [1, 2, 3, 2]
    ], dtype=np.float32)

    K = np.array([
        [1, 1, 2, 1],
        [2, 2, 3, 1],
        [1, 2, 3, 2]
    ], dtype=np.float32)

    V = np.array([
        [1, 1, 2, 1],
        [2, 2, 3, 1],
        [1, 2, 3, 2]
    ], dtype=np.float32)

    # 因果掩码：True = 可关注
    causal_mask = np.array([
        [True,  False, False],  # 位置 0 只能看位置 0
        [True,  True,  False],  # 位置 1 能看位置 0, 1
        [True,  True,  True]    # 位置 2 能看所有
    ])

    d_k = Q.shape[-1]

    # Step 1: 计算注意力分数
    scores = Q @ K.T
    print(f"\n原始注意力分数矩阵:")
    print(scores)

    # Step 2: 应用掩码
    masked_scores = np.where(causal_mask, scores, -1e9)
    print(f"\n应用因果掩码后:")
    print(np.where(causal_mask, np.round(scores, 2), "-inf"))

    # Step 3: 缩放
    scaled_scores = masked_scores / np.sqrt(d_k)

    # Step 4: Softmax
    attention_weights = numpy_softmax(scaled_scores)
    print(f"\n注意力权重 (softmax 后):")
    print(np.round(attention_weights, 4))
    print("(注意: 被掩码的位置权重接近 0)")

    # Step 5: 加权求和
    output = attention_weights @ V
    print(f"\n最终输出:")
    print(np.round(output, 4))

    # 对比无掩码的结果
    attention_weights_no_mask = numpy_softmax(scores / np.sqrt(d_k))
    output_no_mask = attention_weights_no_mask @ V
    print(f"\n对比 - 无掩码的输出:")
    print(np.round(output_no_mask, 4))
    print("\n差异（展示掩码的效果）:")
    print(np.round(output - output_no_mask, 4))


def demo_pytorch_attention():
    """
    演示 PyTorch 版本的注意力计算

    提供两种实现：
    1. 手动实现（适合理解原理）
    2. 使用 F.scaled_dot_product_attention（推荐，会自动选择最优后端）
    """
    print("\n" + "=" * 60)
    print("[5] PyTorch 版本的注意力计算")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 64
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)

    print(f"\nQ shape: {Q.shape}")
    print(f"  含义: (batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim})")

    # --- 手动实现 ---
    def manual_attention(Q, K, V, mask=None):
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

        if mask is not None:
            # mask: 1 表示被屏蔽的位置
            scores = scores.masked_fill(mask.bool(), float('-inf'))

        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)
        return output, weights

    # 创建因果掩码
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)  # 上三角为 1
    output_manual, weights_manual = manual_attention(Q, K, V, causal_mask)
    print(f"\n手动实现 - 输出 shape: {output_manual.shape}")
    print(f"手动实现 - 权重 shape: {weights_manual.shape}")
    print(f"手动实现 - 权重示例 (batch=0, head=0, query=0):")
    print(f"  {weights_manual[0, 0, 0, :].tolist()[:5]}...")
    print(f"  (第一个 query 只关注第一个 key，其余权重为 0)")

    # --- 使用 PyTorch 内置 SDPA ---
    # F.scaled_dot_product_attention 会自动选择最优后端:
    # - Flash Attention (最快，需要 CUDA)
    # - Memory-Efficient Attention
    # - Math (通用回退)
    causal_mask_sdpa = torch.triu(
        torch.full((seq_len, seq_len), float('-inf')), diagonal=1
    )
    output_sdpa = F.scaled_dot_product_attention(Q, K, V, attn_mask=causal_mask_sdpa)
    print(f"\nSDPA 实现 - 输出 shape: {output_sdpa.shape}")

    # 验证两种实现结果一致
    diff = (output_manual - output_sdpa).abs().max().item()
    print(f"两种实现的最大差异: {diff:.2e}")

    # --- 使用 is_causal 参数（更简洁）---
    output_causal = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    diff2 = (output_manual - output_causal).abs().max().item()
    print(f"is_causal=True 与手动实现的最大差异: {diff2:.2e}")


def demo_full_precision_softmax():
    """
    演示 full-precision softmax 的作用

    在 FP16/BF16 下计算 softmax 可能导致精度损失，
    因此部分模型会先将 attention scores 转为 FP32 计算 softmax，
    再转回原始精度。
    """
    print("\n" + "=" * 60)
    print("[6] Full-Precision Softmax")
    print("=" * 60)

    torch.manual_seed(42)

    # 模拟大数值的注意力分数（FP16 下容易溢出）
    scores = torch.randn(1, 1, 16, 16) * 10  # 较大的分数值

    # FP16 下直接计算 softmax
    scores_fp16 = scores.to(torch.float16)
    weights_fp16 = F.softmax(scores_fp16, dim=-1)

    # Full-precision: 先转 FP32 计算，再转回 FP16
    weights_full = F.softmax(scores_fp16.float(), dim=-1).to(torch.float16)

    # 对比差异
    diff = (weights_fp16.float() - weights_full.float()).abs()
    print(f"\nFP16 直接计算 vs Full-precision 计算:")
    print(f"  最大差异: {diff.max().item():.6f}")
    print(f"  平均差异: {diff.mean().item():.6f}")
    print(f"  差异 > 0 的位置数: {(diff > 0).sum().item()} / {diff.numel()}")

    print("""
结论:
  - Full-precision softmax 在 FP32 下计算 softmax，避免精度损失
  - 代码中通常写作: F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
  - 推荐在低精度推理中使用，以保证数值稳定性
    """)


def main():
    print("Attention Mechanism & Mask Demo")
    print("=" * 60)

    demo_basic_attention()
    demo_padding_mask()
    demo_causal_mask()
    demo_masked_attention()
    demo_pytorch_attention()
    demo_full_precision_softmax()


if __name__ == "__main__":
    main()
