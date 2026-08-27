"""RoPE（旋转位置编码）最小实现与 SGLang 风格实现的对拍。

对应文档: ../rope.md
    - 第 4 章 工程实现：SGLang 源码解析（RotaryEmbedding 类）
        - 4.2 _compute_inv_freq()        逆频率计算   θ_i = base^(-2i/d)
        - 4.3 _compute_cos_sin_cache()   cos/sin 缓存预计算
        - 4.4 apply_rotary_emb()         旋转应用（NeoX / GPT-J 两种风格）
        - 4.5 forward_native()           完整前向流程
    - 第 6 章 动手验证：最小实现与性质对拍（rope() 函数）

代码忠实于文档实现，仅补充了使其能运行/自洽的上下文（import、类封装、
__main__ 测试块），核心算法逻辑未改动。

运行:
    python rope.py
"""

import torch


# =====================================================================
# 第 6 章：最小 RoPE 实现（NeoX 半半式配对）
# =====================================================================
def rope(x, pos, base=10000.0):
    """最小 RoPE 实现（NeoX 半半式配对），x: [..., d]（d 为偶数），pos: 标量位置"""
    d = x.shape[-1]
    inv_freq = 1.0 / base ** (torch.arange(0, d, 2, dtype=torch.float32) / d)  # θ_i（4.2 节）
    ang = pos * inv_freq                        # 各组旋转角 m·θ_i，[d/2]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]  # NeoX 配对：(x_j, x_{j+d/2})
    return torch.cat([x1 * cos - x2 * sin,       # 3.4 节旋转公式
                      x2 * cos + x1 * sin], dim=-1)


# =====================================================================
# 第 4.4 节：旋转应用原语 apply_rotary_emb()
# =====================================================================
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


# =====================================================================
# 第 4 章：SGLang 风格的 RotaryEmbedding
# 把 4.2 / 4.3 / 4.4 / 4.5 节的四段源码组织成一个自洽的类
# =====================================================================
class RotaryEmbedding:
    """SGLang forward_native 风格的旋转位置编码。

    忠实于文档 4.2~4.5 节的源码；构造函数与个别形状处理为补充的运行上下文。
    """

    def __init__(
        self,
        head_size,
        rotary_dim,
        max_position_embeddings,
        base=10000.0,
        is_neox_style=True,
    ):
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.cos_sin_cache = self._compute_cos_sin_cache()

    # ---------- 4.2 逆频率计算 ----------
    def _compute_inv_freq(self, base):
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim)
        )
        return inv_freq

    # ---------- 4.3 预计算 cos/sin 缓存 ----------
    def _compute_cos_sin_cache(self):
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)  # [0, 1, ..., max_pos-1]

        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # 外积: [max_pos, rotary_dim/2]
        # freqs[pos][i] = pos * θ_i  ← 每个位置、每个维度组的旋转角度（3.1 节的 m·θ_i）

        cos = freqs.cos()                               # [max_pos, rotary_dim/2]
        sin = freqs.sin()                               # [max_pos, rotary_dim/2]
        cache = torch.cat((cos, sin), dim=-1)           # [max_pos, rotary_dim]，布局 [cos | sin]
        return cache

    # ---------- 4.5 完整前向流程 ----------
    def forward_native(self, positions, query, key, offsets=None):  # 摘录自 base.py，签名有省略
        # ① 按每个 token 的位置查表，取出对应行的 cos/sin
        pos = positions if offsets is None else positions + offsets
        cos_sin = self.cos_sin_cache.index_select(0, pos)
        cos, sin = cos_sin.chunk(2, dim=-1)          # 拆开 [cos | sin] 布局

        # ② 部分旋转：只旋转前 rotary_dim 维，剩余维度原样保留
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]    # 不参与旋转的"无位置"通道

        # ③ 对 Q 应用旋转
        query_rot = apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1)

        # ④ 对 K 应用相同的旋转
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1)

        return query, key


# =====================================================================
# 测试与性质对拍
# =====================================================================
def _test_minimal_properties():
    """第 6 章：验证 RoPE 的三条核心数学性质。"""
    print("=" * 60)
    print("第 6 章 最小实现 rope() —— 三条性质验证")
    print("=" * 60)

    torch.manual_seed(0)
    d = 128
    q, k = torch.randn(d), torch.randn(d)

    # ---- 性质 1：分数只依赖相对距离 m-n（整体平移位置，分数不变）----
    s1 = rope(q, 7)   @ rope(k, 3)     # m-n = 4
    s2 = rope(q, 104) @ rope(k, 100)   # m-n = 4，整体平移 97
    s3 = rope(q, 9)   @ rope(k, 3)     # m-n = 6（对照组）
    print(f"性质1 分数(相对距离4): {s1.item():.6f}  {s2.item():.6f}   对照(距离6): {s3.item():.6f}")
    # 前两个相等（差异仅浮点舍入，~1e-6），第三个明显不同
    assert torch.allclose(s1, s2, rtol=1e-4, atol=1e-4), "相对距离相同，分数应一致"
    assert not torch.allclose(s1, s3, rtol=1e-2), "相对距离不同，分数应不同"

    # ---- 性质 2：保范性（旋转不改变模长，2.1 节的正交性）----
    n0, n1 = q.norm().item(), rope(q, 12345).norm().item()
    print(f"性质2 保范性: {n0:.6f}  {n1:.6f}")
    assert torch.allclose(q.norm(), rope(q, 12345).norm(), rtol=1e-4, atol=1e-4)

    # ---- 性质 3：位置 0 是恒等变换（所有旋转角为 0）----
    id_err = (rope(q, 0) - q).abs().max().item()
    print(f"性质3 pos=0 恒等: max_err = {id_err:.2e}")
    assert id_err < 1e-5

    print("三条性质全部通过。\n")


def _test_shapes_and_equivalence():
    """构造随机 query/key，跑 RotaryEmbedding 并打印 shape；
    验证 forward_native(NeoX) 与最小 rope() 数值一致。"""
    print("=" * 60)
    print("第 4 章 RotaryEmbedding —— 形状 + 与最小实现对拍")
    print("=" * 60)

    torch.manual_seed(0)
    num_tokens, num_heads, head_size = 4, 8, 128
    max_pos = 2048
    rotary_dim = head_size  # LLaMA 系：全旋转

    query = torch.randn(num_tokens, num_heads, head_size)
    key = torch.randn(num_tokens, num_heads, head_size)
    positions = torch.arange(num_tokens)

    rope_mod = RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_pos,
        base=10000.0,
        is_neox_style=True,
    )
    q_out, k_out = rope_mod.forward_native(positions, query.clone(), key.clone())
    print(f"query in : {tuple(query.shape)}   query out: {tuple(q_out.shape)}")
    print(f"key   in : {tuple(key.shape)}   key   out: {tuple(k_out.shape)}")
    print(f"cos_sin_cache: {tuple(rope_mod.cos_sin_cache.shape)}")

    # ---- 对拍：RotaryEmbedding(NeoX) 应与逐 token 逐 head 的最小 rope() 等价 ----
    ref = torch.empty_like(query)
    for t in range(num_tokens):
        for h in range(num_heads):
            ref[t, h] = rope(query[t, h], int(positions[t]), base=10000.0)
    print(f"对拍 forward_native(NeoX) vs 最小 rope(): "
          f"max_err = {(q_out - ref).abs().max().item():.2e}")
    assert torch.allclose(q_out, ref, rtol=1e-4, atol=1e-4), \
        "NeoX 风格的 forward_native 应与最小 rope() 数值一致"
    print("对拍通过。\n")


def _test_neox_vs_gptj_differ():
    """3.4 / 4.4 / 第 6 章提醒：NeoX 与 GPT-J 两种配对风格数学上等价但对同一
    输入布局给出不同结果，混用会静默算错。这里验证它们确实不相等。"""
    print("=" * 60)
    print("第 3.4 / 4.4 节 —— NeoX vs GPT-J 风格差异")
    print("=" * 60)

    torch.manual_seed(0)
    num_tokens, num_heads, head_size = 2, 4, 64
    query = torch.randn(num_tokens, num_heads, head_size)
    key = torch.randn(num_tokens, num_heads, head_size)
    positions = torch.arange(num_tokens)

    common = dict(head_size=head_size, rotary_dim=head_size,
                  max_position_embeddings=512, base=10000.0)
    neox = RotaryEmbedding(is_neox_style=True, **common)
    gptj = RotaryEmbedding(is_neox_style=False, **common)

    q_neox, _ = neox.forward_native(positions, query.clone(), key.clone())
    q_gptj, _ = gptj.forward_native(positions, query.clone(), key.clone())
    diff = (q_neox - q_gptj).abs().max().item()
    print(f"NeoX vs GPT-J 输出 max_diff = {diff:.4f}（应显著非零）")
    assert not torch.allclose(q_neox, q_gptj, rtol=1e-2), \
        "两种风格对同一输入应给出不同结果（不可混用）"

    # 但位置 0 时两种风格都退化为恒等，输出应相同
    p0 = torch.zeros(num_tokens, dtype=torch.long)
    q0_neox, _ = neox.forward_native(p0, query.clone(), key.clone())
    q0_gptj, _ = gptj.forward_native(p0, query.clone(), key.clone())
    assert torch.allclose(q0_neox, query, atol=1e-6)
    assert torch.allclose(q0_gptj, query, atol=1e-6)
    print("pos=0 时两种风格均退化为恒等，符合预期。\n")


if __name__ == "__main__":
    _test_minimal_properties()
    _test_shapes_and_equivalence()
    _test_neox_vs_gptj_differ()
    print("全部测试通过。")
