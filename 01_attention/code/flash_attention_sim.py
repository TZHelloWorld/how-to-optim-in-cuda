"""flash_attention_sim.py — 用 PyTorch 模拟 FlashAttention 的分块递推与循环顺序

对应文档: ../cuda_attention_optimization_guide.md 第 10.3 节

写 CUDA 之前先用纯 PyTorch 把分块递推"演"一遍：
  - 验证第 6 章的数学（在线合并是否精确）；
  - 直观感受循环顺序（FA-1 外层 KV vs FA-2 外层 Q）的影响。

两种循环顺序核心数学完全一致，只交换了循环内外与状态的存放位置。
FA-1 与 FA-2 输出逐位一致（合并公式精确）；与朴素参考实现相差 ~1e-6（浮点求和顺序不同）。

运行:
    python flash_attention_sim.py
"""

import torch

torch.manual_seed(0)
N, D = 4096, 128            # 序列长度 / head 维度（无 GPU 时可改小，如 512/64）
Br = Bc = 64                # Q 块大小 / KV 块大小
Tr, Tc = N // Br, N // Bc   # Q 块数 / KV 块数
scale = D ** -0.5           # 1/√d
dev = "cuda" if torch.cuda.is_available() else "cpu"

Q = torch.randn(N, D, device=dev)
K = torch.randn(N, D, device=dev)
V = torch.randn(N, D, device=dev)


def fa1_outer_kv(Q, K, V):
    """FA-1 风格：外层 KV，内层 Q —— (õ, m, l) 存于全局张量，反复读改写"""
    O = torch.zeros(N, D, device=dev)                    # 未归一化输出 õ（收尾统一除以 l）
    m = torch.full((N, 1), float("-inf"), device=dev)    # 全 N 行的最大值（共享进度）
    l = torch.zeros(N, 1, device=dev)                    # 全 N 行的分母（共享进度）
    for j in range(Tc):                                  # 外层：KV 块
        K_j, V_j = K[j * Bc:(j + 1) * Bc], V[j * Bc:(j + 1) * Bc]
        for i in range(Tr):                              # 内层：Q 块
            sl = slice(i * Br, (i + 1) * Br)
            O_i, m_i, l_i = O[sl], m[sl], l[sl]          # 灾难点：取回历史进度（HBM 读）
            S = Q[sl] @ K_j.T * scale                    # Br×Bc 分块分数
            m_blk = S.max(dim=-1, keepdim=True).values   # 本块的逐行最大值（Br×1）
            m_new = torch.maximum(m_i, m_blk)            # 逐行在线更新基准
            P = torch.exp(S - m_new)                     # 直接以 m_new 为基准（6.5 节的 P~）
            corr = torch.exp(m_i - m_new)                # 旧累积的补偿系数（Br×1）
            l[sl] = l_i * corr + P.sum(dim=-1, keepdim=True)
            O[sl] = O_i * corr + P @ V_j                 # 灾难点：写回全局进度（HBM 写）
            m[sl] = m_new
    return O / l                                         # 延迟归一化：除法只在最后做一次


def fa2_outer_q(Q, K, V):
    """FA-2 风格：外层 Q，内层 KV —— (õ, m, l) 是局部变量，收尾只写一次"""
    O = torch.zeros(N, D, device=dev)
    for i in range(Tr):                                  # 外层：Q 块
        Q_i = Q[i * Br:(i + 1) * Br]
        o = torch.zeros(Br, D, device=dev)               # 本 Q 块的私有状态（模拟片上驻留）
        m = torch.full((Br, 1), float("-inf"), device=dev)
        l = torch.zeros(Br, 1, device=dev)
        for j in range(Tc):                              # 内层：KV 块（更新公式逐行相同）
            K_j, V_j = K[j * Bc:(j + 1) * Bc], V[j * Bc:(j + 1) * Bc]
            S = Q_i @ K_j.T * scale
            m_blk = S.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m, m_blk)
            P = torch.exp(S - m_new)
            corr = torch.exp(m - m_new)
            l = l * corr + P.sum(dim=-1, keepdim=True)
            o = o * corr + P @ V_j
            m = m_new
        O[i * Br:(i + 1) * Br] = o / l                   # 每个 O 块只写 1 次
    return O


def bench(fn, iters=5):
    """CUDA event + 预热计时（10.2 节的计时建议）；CPU 上退化为墙钟计时"""
    for _ in range(2):                    # 预热：排除首次运行的一次性开销
        fn(Q, K, V)
    if dev == "cuda":
        torch.cuda.synchronize()
        beg = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        beg.record()
        for _ in range(iters):
            fn(Q, K, V)
        end.record()
        torch.cuda.synchronize()
        return beg.elapsed_time(end) / iters      # 毫秒
    import time
    t0 = time.time()
    for _ in range(iters):
        fn(Q, K, V)
    return (time.time() - t0) / iters * 1e3        # 毫秒


if __name__ == "__main__":
    # ---- 正确性：两种循环顺序 + 朴素参考实现，三方对拍 ----
    out1, out2 = fa1_outer_kv(Q, K, V), fa2_outer_q(Q, K, V)
    ref = torch.softmax(Q @ K.T * scale, dim=-1) @ V
    print("device:", dev)
    print("max|FA1 - FA2| =", (out1 - out2).abs().max().item(),
          " (期望 0：两种顺序逐位一致，合并公式精确)")
    print("max|FA2 - ref| =", (out2 - ref).abs().max().item(),
          " (~1e-6 量级：与参考实现的浮点求和顺序差异)")

    # ---- 计时 ----
    print(f"FA-1 风格（外层 KV）: {bench(fa1_outer_kv):.2f} ms")
    print(f"FA-2 风格（外层 Q） : {bench(fa2_outer_q):.2f} ms")
