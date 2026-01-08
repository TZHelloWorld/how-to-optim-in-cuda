#!/usr/bin/env python3
"""在目录下所有 .safetensors 中搜索 key, 打印详细信息。

用法:
    # 子串匹配(默认, 不区分大小写), 默认 pattern 为 mtp
    python3 find_safetensors_key.py /path/to/model_dir --pattern o_proj

    # 打印所有 key(匹配的标 *), 同时输出匹配 key 的详细信息
    python3 find_safetensors_key.py /path/to/model_dir --pattern mtp --show-all

    # 精确匹配完整 key
    python3 find_safetensors_key.py /path/to/model_dir \
        --pattern model.mtp.layers.0.self_attn.o_proj.weight --exact

    # 正则匹配
    python3 find_safetensors_key.py /path/to/model_dir \
        --pattern "mtp\\.layers\\.\\d+\\.self_attn" --regex

    # 加载张量并打印数值统计(用于判断权重是否正常初始化)
    python3 find_safetensors_key.py /path/to/model_dir --pattern o_proj.weight --stats
"""

import argparse
import json
import re
import struct
from pathlib import Path


def read_safetensors_header(path: Path) -> dict:
    """只读 header, 不加载权重数据, 速度快。

    safetensors 格式: 前 8 字节是小端 uint64 的 header 长度, 后面跟 JSON header。
    """
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    # header: {key: {"dtype": ..., "shape": [...], "data_offsets": [...]}}
    return header


def tensor_stats(path: Path, key: str) -> str:
    """加载单个张量并计算统计值(需要安装 safetensors + torch)。"""
    try:
        import torch
        from safetensors import safe_open
    except ImportError:
        return "(--stats 需要 pip install safetensors torch)"
    with safe_open(str(path), framework="pt") as f:
        t = f.get_tensor(key)
    tf = t.float()
    return (
        f"mean={tf.mean().item():.6g}  std={tf.std().item():.6g}  "
        f"min={tf.min().item():.6g}  max={tf.max().item():.6g}  "
        f"nan={torch.isnan(tf).sum().item()}  inf={torch.isinf(tf).sum().item()}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path, help="包含 .safetensors 的目录")
    parser.add_argument(
        "--pattern", default="mtp", help="筛选关键字(不区分大小写), 默认 mtp"
    )
    parser.add_argument(
        "--show-all", action="store_true", help="同时打印所有 key(默认只打印匹配的)"
    )
    parser.add_argument("--exact", action="store_true", help="精确匹配完整 key")
    parser.add_argument("--regex", action="store_true", help="按正则表达式匹配")
    parser.add_argument(
        "--stats", action="store_true", help="加载匹配的张量并打印数值统计(较慢)"
    )
    args = parser.parse_args()

    files = sorted(args.model_dir.glob("*.safetensors"))
    if not files:
        print(f"[!] {args.model_dir} 下没有找到 .safetensors 文件")
        return

    if args.exact:
        match_fn = lambda k: k == args.pattern
    elif args.regex:
        prog = re.compile(args.pattern)
        match_fn = lambda k: prog.search(k) is not None
    else:
        needle = args.pattern.lower()
        match_fn = lambda k: needle in k.lower()

    total_keys = 0
    matched = []  # (file_path, key, info)

    for fp in files:
        header = read_safetensors_header(fp)
        total_keys += len(header)
        print(f"\n===== {fp.name} ({len(header)} keys) =====")
        for key in sorted(header):
            info = header[key]
            hit = match_fn(key)
            if hit:
                matched.append((fp, key, info))
            if args.show_all or hit:
                mark = "* " if hit else "  "
                print(f"{mark}  {key:<80s} {info['dtype']:>8s}  {info['shape']}")

    print(f"\n===== 汇总: 共 {len(files)} 个文件, {total_keys} 个 key =====")
    print(f"匹配 '{args.pattern}' 的 key 共 {len(matched)} 个:\n")

    if not matched:
        print("    提示: 去掉 --exact 试试子串匹配, 或用 --regex")
        return

    # 组装表格行
    headers = ["key", "file", "dtype", "shape", "numel", "size(MiB)"]
    if args.stats:
        headers.append("stats")
    rows = []
    n_params = 0
    total_bytes = 0
    for fp, key, info in matched:
        shape = info["shape"]
        numel = 1
        for d in shape:
            numel *= d
        n_params += numel
        offs = info.get("data_offsets", [0, 0])
        nbytes = offs[1] - offs[0]
        total_bytes += nbytes
        row = [
            key,
            fp.name,
            info["dtype"],
            str(shape),
            f"{numel:,}",
            f"{nbytes / 1024 / 1024:.2f}",
        ]
        if args.stats:
            row.append(tensor_stats(fp, key))
        rows.append(row)

    # 按列宽对齐打印
    widths = [
        max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))
    ]
    sep = "-+-".join("-" * w for w in widths)
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print(sep)
    for r in rows:
        print(" | ".join(r[i].ljust(widths[i]) for i in range(len(headers))))

    print(
        f"\n匹配参数量: {n_params / 1e6:.2f} M, "
        f"合计 {total_bytes / 1024 / 1024:.2f} MiB"
    )


if __name__ == "__main__":
    main()
