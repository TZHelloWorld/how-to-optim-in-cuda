# profile_copy_paths.py — 用 PyTorch profiler 区分三种拷贝路径（对应文档 §3.4）
#
# 对应文档: ../../cuda_copy_operator_guide.md
# 三种拷贝在时间线上一眼可辨（§2.5）:
#   Memcpy DtoD 条目           = Copy Engine 路径（不占 SM）
#   vectorized_elementwise_kernel = SM 路径（elementwise / cast）
#
# 运行:
#   python profile_copy_paths.py

import torch
from torch.profiler import profile, ProfilerActivity

x = torch.randn(4096, 4096, device='cuda')

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    a = x.clone()                    # 连续、同 dtype   -> Memcpy DtoD（CE，不占 SM）
    b = x.t().contiguous()           # 非连续（转置）   -> elementwise copy kernel（SM）
    c = x.to(torch.float16)          # dtype 转换       -> 转型拷贝 kernel（SM）

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# 表中可看到：Memcpy DtoD (Device -> Device) 一行，
# 以及两条 vectorized_elementwise_kernel<...>（名字含 copy/cast 字样）
