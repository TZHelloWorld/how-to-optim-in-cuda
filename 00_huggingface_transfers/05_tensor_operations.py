#!/usr/bin/env python3
"""
05_tensor_operations.py - PyTorch 张量维度变换操作演示

本脚本演示了大模型代码中常见的张量维度变换操作：
1. permute - 重新排列所有维度
2. transpose - 交换两个维度
3. view - 改变形状（要求内存连续）
4. reshape - 改变形状（自动处理连续性）
5. contiguous - 确保内存连续
6. einops.rearrange - 更直观的维度变换

这些操作在 Attention 计算中非常频繁，理解它们对于阅读和编写大模型代码至关重要。

使用前请确保：
- pip install torch einops
"""

import torch
import torch.nn as nn


def demo_permute_transpose():
    """
    演示 permute 和 transpose 操作

    permute: 重新排列所有维度的顺序
    transpose: 交换两个指定维度

    两者都不复制数据，返回的是原张量的视图（view）。
    """
    print("=" * 60)
    print("[1] permute 与 transpose")
    print("=" * 60)

    x = torch.randn(2, 8, 64, 128)  # (batch, heads, seq_len, head_dim)
    print(f"\n原始张量 shape: {x.shape}")
    print(f"  含义: (batch=2, heads=8, seq_len=64, head_dim=128)")

    # --- permute: 调换维度顺序 ---
    # 常见场景：将 (batch, seq, heads, head_dim) 转为 (batch, heads, seq, head_dim)
    x_bshd = torch.randn(2, 64, 8, 128)  # (batch, seq_len, heads, head_dim)
    x_bhsd = x_bshd.permute(0, 2, 1, 3)  # -> (batch, heads, seq_len, head_dim)
    print(f"\npermute(0, 2, 1, 3):")
    print(f"  {x_bshd.shape} -> {x_bhsd.shape}")
    print(f"  (batch, seq, heads, dim) -> (batch, heads, seq, dim)")

    # --- transpose: 交换两个维度 ---
    x_transposed = x.transpose(1, 2)  # 交换 heads 和 seq_len
    print(f"\ntranspose(1, 2):")
    print(f"  {x.shape} -> {x_transposed.shape}")
    print(f"  (batch, heads, seq, dim) -> (batch, seq, heads, dim)")

    # --- 连续性检查 ---
    print(f"\n原始张量是否连续: {x.is_contiguous()}")
    print(f"permute 后是否连续: {x_bhsd.is_contiguous()}")
    print(f"transpose 后是否连续: {x_transposed.is_contiguous()}")

    # permute 和 transpose 都不复制数据
    print(f"\n原始张量 data_ptr: {x.data_ptr()}")
    print(f"transpose 后 data_ptr: {x_transposed.data_ptr()}")
    print(f"(相同的 data_ptr 说明共享内存，未复制数据)")


def demo_view_reshape():
    """
    演示 view 和 reshape 的区别

    view: 要求张量内存连续，否则报错
    reshape: 自动处理，不连续时会复制数据

    常见场景：
    - 将 (batch, seq, hidden) 拆分为 (batch, seq, heads, head_dim)
    - 将 (batch, heads, seq, head_dim) 合并为 (batch, seq, hidden)
    """
    print("\n" + "=" * 60)
    print("[2] view 与 reshape")
    print("=" * 60)

    # --- 基本用法 ---
    x = torch.randn(2, 64, 512)  # (batch, seq_len, hidden_size)
    num_heads = 8
    head_dim = 512 // num_heads  # 64

    # 拆分 hidden_size 为 heads * head_dim
    x_split = x.view(2, 64, num_heads, head_dim)
    print(f"\nview 拆分维度:")
    print(f"  {x.shape} -> {x_split.shape}")
    print(f"  (batch, seq, hidden=512) -> (batch, seq, heads=8, head_dim=64)")

    # 合并 heads 和 head_dim
    x_merged = x_split.view(2, 64, -1)  # -1 表示自动计算
    print(f"\nview 合并维度 (-1 自动计算):")
    print(f"  {x_split.shape} -> {x_merged.shape}")

    # --- view 在非连续张量上会失败 ---
    x_permuted = x_split.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
    print(f"\npermute 后:")
    print(f"  shape: {x_permuted.shape}")
    print(f"  is_contiguous: {x_permuted.is_contiguous()}")

    try:
        x_permuted.view(2, 8, -1)
        print("  view 成功")
    except RuntimeError as e:
        print(f"  view 失败: {str(e)[:80]}...")

    # 方案 1: 先 contiguous 再 view
    x_contig = x_permuted.contiguous().view(2, 8, -1)
    print(f"\n方案 1 - contiguous().view(): {x_contig.shape}")

    # 方案 2: 使用 reshape（自动处理）
    x_reshaped = x_permuted.reshape(2, 8, -1)
    print(f"方案 2 - reshape(): {x_reshaped.shape}")

    # 验证结果一致
    print(f"两种方案结果一致: {torch.allclose(x_contig, x_reshaped)}")


def demo_contiguous():
    """
    深入理解 contiguous（内存连续性）

    PyTorch 张量在内存中按 stride 排列。
    permute/transpose 改变 stride 但不移动数据，导致内存不连续。
    contiguous() 会重新排列数据使其在内存中连续。
    """
    print("\n" + "=" * 60)
    print("[3] contiguous 内存连续性")
    print("=" * 60)

    x = torch.randn(3, 4, 5)
    print(f"\n原始张量:")
    print(f"  shape: {x.shape}")
    print(f"  stride: {x.stride()}")
    print(f"  is_contiguous: {x.is_contiguous()}")

    # permute 改变 stride 但不移动数据
    x_p = x.permute(2, 0, 1)
    print(f"\npermute(2, 0, 1) 后:")
    print(f"  shape: {x_p.shape}")
    print(f"  stride: {x_p.stride()}")
    print(f"  is_contiguous: {x_p.is_contiguous()}")
    print(f"  data_ptr 相同: {x.data_ptr() == x_p.data_ptr()}")

    # contiguous 复制数据使其连续
    x_c = x_p.contiguous()
    print(f"\ncontiguous() 后:")
    print(f"  shape: {x_c.shape}")
    print(f"  stride: {x_c.stride()}")
    print(f"  is_contiguous: {x_c.is_contiguous()}")
    print(f"  data_ptr 相同: {x.data_ptr() == x_c.data_ptr()}")
    print(f"  (data_ptr 不同说明数据被复制了)")

    print("""
Stride 解释:
  对于 shape (3, 4, 5) 的连续张量，stride 为 (20, 5, 1)
  含义: 沿第 0 维移动 1 步需要跳 20 个元素，
       沿第 1 维移动 1 步需要跳 5 个元素，
       沿第 2 维移动 1 步需要跳 1 个元素。

  permute 后 stride 变为 (1, 20, 5)，
  在内存中不再是连续排列的，所以 view() 无法直接使用。
    """)


def demo_einops_rearrange():
    """
    演示 einops.rearrange 的使用

    einops 提供更直观的语法来表达维度变换，
    在大模型代码中（如 sglang、vllm）被广泛使用。
    """
    print("=" * 60)
    print("[4] einops.rearrange")
    print("=" * 60)

    try:
        from einops import rearrange
    except ImportError:
        print("需要安装 einops: pip install einops")
        return

    # --- 示例 1: 交换维度 ---
    x = torch.randn(64, 2, 512)  # (seq_len, batch, dim)
    y = rearrange(x, "s b d -> b s d")
    print(f"\n交换维度: 's b d -> b s d'")
    print(f"  {x.shape} -> {y.shape}")
    # 等价于: x.permute(1, 0, 2)

    # --- 示例 2: 拆分维度 ---
    x = torch.randn(2, 64, 512)  # (batch, seq, hidden)
    y = rearrange(x, "b s (h d) -> b h s d", h=8)
    print(f"\n拆分维度: 'b s (h d) -> b h s d', h=8")
    print(f"  {x.shape} -> {y.shape}")
    # 等价于: x.view(2, 64, 8, 64).permute(0, 2, 1, 3)

    # --- 示例 3: 合并维度 ---
    x = torch.randn(2, 8, 64, 64)  # (batch, heads, seq, head_dim)
    y = rearrange(x, "b h s d -> b s (h d)")
    print(f"\n合并维度: 'b h s d -> b s (h d)'")
    print(f"  {x.shape} -> {y.shape}")
    # 等价于: x.permute(0, 2, 1, 3).contiguous().view(2, 64, -1)

    # --- 示例 4: 分离展平的 batch 和 seq ---
    bsz = 4
    x = torch.randn(bsz * 32, 8, 64)  # (batch*seq, heads, head_dim)
    y = rearrange(x, "(b s) h d -> b h s d", b=bsz)
    print(f"\n分离展平维度: '(b s) h d -> b h s d', b={bsz}")
    print(f"  {x.shape} -> {y.shape}")

    # --- 示例 5: 转置 K 用于注意力计算 ---
    k = torch.randn(2, 8, 64, 128)  # (batch, heads, seq, head_dim)
    k_t = rearrange(k, "b h s d -> b h d s")
    print(f"\n转置 K: 'b h s d -> b h d s'")
    print(f"  {k.shape} -> {k_t.shape}")
    # 等价于: k.transpose(-2, -1)

    # --- 对比: 使用 einops 处理 Q, K, V ---
    print("\n对比 einops vs PyTorch 原生:")
    q = torch.randn(64, 2, 8, 128)  # (seq, batch, heads, dim)
    k = torch.randn(64, 2, 8, 128)
    v = torch.randn(64, 2, 8, 128)

    # einops 方式
    q1, k1, v1 = [rearrange(t, "s b h d -> b h s d").contiguous() for t in (q, k, v)]

    # PyTorch 原生方式
    q2, k2, v2 = [t.permute(1, 2, 0, 3).contiguous() for t in (q, k, v)]

    print(f"  einops 结果:  q={q1.shape}, k={k1.shape}, v={v1.shape}")
    print(f"  原生结果:     q={q2.shape}, k={k2.shape}, v={v2.shape}")
    print(f"  结果一致: {torch.allclose(q1, q2) and torch.allclose(k1, k2) and torch.allclose(v1, v2)}")


def demo_common_patterns():
    """
    展示大模型代码中常见的维度变换模式
    """
    print("\n" + "=" * 60)
    print("[5] 大模型代码中常见的维度变换模式")
    print("=" * 60)

    batch, seq_len, hidden_size = 2, 64, 512
    num_heads = 8
    head_dim = hidden_size // num_heads

    # --- 模式 1: Q/K/V 投影后拆分多头 ---
    print("\n模式 1: Q/K/V 投影后拆分多头")
    x = torch.randn(batch, seq_len, hidden_size)
    q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    q = q_proj(x)  # (batch, seq, hidden)
    q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    print(f"  (batch, seq, hidden) -> view -> (batch, seq, heads, dim) -> transpose -> (batch, heads, seq, dim)")
    print(f"  {x.shape} -> {q.shape}")

    # --- 模式 2: 注意力输出合并多头 ---
    print("\n模式 2: 注意力输出合并多头")
    attn_output = torch.randn(batch, num_heads, seq_len, head_dim)
    merged = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, hidden_size)
    print(f"  (batch, heads, seq, dim) -> transpose -> contiguous -> view -> (batch, seq, hidden)")
    print(f"  {attn_output.shape} -> {merged.shape}")

    # --- 模式 3: GQA (Grouped Query Attention) 的 key/value 扩展 ---
    print("\n模式 3: GQA key/value 扩展")
    num_kv_heads = 2  # KV 头数少于 Q 头数
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim)
    # 扩展 KV 头数以匹配 Q 头数
    num_groups = num_heads // num_kv_heads
    k_expanded = k.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
    k_expanded = k_expanded.reshape(batch, num_heads, seq_len, head_dim)
    print(f"  KV heads={num_kv_heads}, Q heads={num_heads}, groups={num_groups}")
    print(f"  {k.shape} -> expand -> reshape -> {k_expanded.shape}")


def main():
    demo_permute_transpose()
    demo_view_reshape()
    demo_contiguous()
    demo_einops_rearrange()
    demo_common_patterns()


if __name__ == "__main__":
    main()
