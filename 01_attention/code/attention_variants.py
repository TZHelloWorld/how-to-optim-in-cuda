"""attention_variants.py — Attention 变体的 PyTorch 实现：SDPA / MHA / MQA / GQA / MLA

对应文档: ../attention_variants_mha_mqa_gqa_mla.md

五种结构不是孤立的发明，而是同一个问题（KV Cache 太大）的逐步改进：
    MHA → MQA → GQA 在"份数"维度做减法（共享 K/V）；
    MLA 在"表示"维度做减法（低秩压缩 K/V）。

本文件把文档中的 5 个类 + SDPA 原子操作 + 全部内嵌测试整理成一个自包含、
可 `python attention_variants.py` 直接运行的脚本（有 GPU 用 GPU，否则用 CPU）。

内含的验证链（对应文档各章的内嵌测试）：
    - 手写 SDPA ≡ 官方 F.scaled_dot_product_attention（第 2 章）
    - MHA 的逐步 decode ≡ 一次 prefill（第 4 章）
    - MLA 的逐步 decode ≡ 一次 prefill（第 7 章）
    - GQA(g=h) ≡ MHA、GQA(g=1) ≡ MQA（第 8 章，权重拷贝 + 逐位对拍）

运行:
    python attention_variants.py
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"   # 有 GPU 用 GPU，否则 CPU


# ===========================================================================
# 第 2 章 SDPA：后面所有变体共用的原子操作
# ===========================================================================
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


# ===========================================================================
# 第 4 章 MHA：多头注意力（含 KV Cache）—— h 个头各有独立的 K/V（g = h）
# ===========================================================================
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


# ===========================================================================
# 第 5 章 MQA：多查询注意力 —— 所有头共享一份 K/V（g = 1）
# ===========================================================================
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


# ===========================================================================
# 第 6 章 GQA：分组查询注意力 —— h 个 query 头分 g 组共享 K/V
# ===========================================================================
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


# ===========================================================================
# 第 7 章 MLA：多头潜在注意力 —— 不减份数，把 K/V 低秩压缩成 latent 缓存
# ===========================================================================
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
        self.q_up = nn.Linear(q_lora_rank, num_heads * (self.dn + self.dr), bias=False)
        # KV 侧：一次投影同时产出 c^KV 与共享 k^R 的原料
        self.kv_down = nn.Linear(hidden_size, kv_lora_rank + self.dr, bias=False)
        self.kv_up = nn.Linear(kv_lora_rank, num_heads * (self.dn + self.dv), bias=False)
        self.o_proj = nn.Linear(num_heads * self.dv, hidden_size, bias=False)

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
            ckv = torch.cat([past_cache[0], ckv], dim=1)         # (B, M, kv_rank)
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


# ===========================================================================
# 测试块：把文档中各章的内嵌验证串成一条自洽的验证链
# ===========================================================================
def test_sdpa():
    """第 2 章：手写 SDPA ≡ 官方 F.scaled_dot_product_attention"""
    torch.manual_seed(0)
    q = torch.randn(2, 8, 16, 64, device=device)   # (B, h, N, d)
    k, v = torch.randn_like(q), torch.randn_like(q)
    out = scaled_dot_product_attention(q, k, v, causal_mask(16, device))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ok = torch.allclose(out, ref, atol=1e-5)
    print(f"[SDPA] out.shape={tuple(out.shape)}  手写 ≡ 官方: {ok}")
    assert ok


def test_mha_decode():
    """第 4 章：MHA 逐步 decode ≡ 一次 prefill"""
    torch.manual_seed(0)
    mha = MultiHeadAttention(64, 4).to(device)
    x = torch.randn(2, 6, 64, device=device)

    full = mha(x, attn_mask=causal_mask(6, device))   # 一次 prefill（带因果掩码）

    cache, steps = None, []
    for t in range(6):                             # 逐 token decode（无需掩码：历史都可见）
        step, cache = mha(x[:, t:t + 1], past_key_value=cache, use_cache=True)
        steps.append(step)
    ok = torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5)
    print(f"[MHA] out.shape={tuple(full.shape)}  decode ≡ prefill: {ok}")
    assert ok


def test_mqa_gqa_shapes():
    """第 5/6 章：MQA / GQA 前向输出形状合理"""
    torch.manual_seed(0)
    x = torch.randn(2, 10, 64, device=device)
    m = causal_mask(10, device)
    mqa = MultiQueryAttention(64, 8).to(device)
    gqa = GroupedQueryAttention(64, 8, num_kv_heads=2).to(device)
    o_mqa = mqa(x, m)
    o_gqa = gqa(x, m)
    print(f"[MQA] out.shape={tuple(o_mqa.shape)}")
    print(f"[GQA g=2] out.shape={tuple(o_gqa.shape)}")
    assert o_mqa.shape == x.shape and o_gqa.shape == x.shape


def test_mla_decode():
    """第 7 章：MLA 逐步 decode ≡ 一次 prefill（RoPE 位置经 past_len 正确衔接）"""
    torch.manual_seed(0)
    mla = MultiHeadLatentAttention(128, 4).to(device)
    x = torch.randn(2, 6, 128, device=device)
    full = mla(x, attn_mask=causal_mask(6, device))
    cache, steps = None, []
    for t in range(6):
        o, cache = mla(x[:, t:t + 1], past_cache=cache, use_cache=True)
        steps.append(o)
    ok = torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5)
    print(f"[MLA] out.shape={tuple(full.shape)}  decode ≡ prefill: {ok}")
    print(f"      cache: c^KV {tuple(cache[0].shape)}  k^R {tuple(cache[1].shape)}")
    assert ok


def test_gqa_endpoints():
    """第 8 章：GQA 两端退化 —— GQA(g=h) ≡ MHA、GQA(g=1) ≡ MQA"""
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

    ok_h = torch.allclose(gqa_h(x, m), mha(x, m), atol=1e-6)
    ok_1 = torch.allclose(gqa_1(x, m), mqa(x, m), atol=1e-6)
    print(f"[谱系] GQA(g=h) ≡ MHA: {ok_h}   GQA(g=1) ≡ MQA: {ok_1}")
    assert ok_h and ok_1


if __name__ == "__main__":
    print("device:", device)
    test_sdpa()
    test_mha_decode()
    test_mqa_gqa_shapes()
    test_mla_decode()
    test_gqa_endpoints()
    print("\n全部测试通过：SDPA/MHA/MQA/GQA/MLA 输出形状与数值均合理，谱系关系成立。")
