#!/usr/bin/env python3
"""
06_safetensors_inspector.py - safetensors 权重文件分析工具

本脚本提供了 safetensors 权重文件的分析功能：
1. 查看权重文件中的所有张量信息（名称、形状、数据类型）
2. 按关键字过滤张量
3. 统计参数总量和内存占用
4. 从文件中提取指定张量
5. 分析分片索引文件 (model.safetensors.index.json)

safetensors 格式说明:
  safetensors 是 HuggingFace 推出的模型权重存储格式。
  相比 pickle-based 的 .bin 文件，safetensors 具有：
  - 安全性：不使用 pickle，避免代码注入风险
  - 速度：支持零拷贝读取（zero-copy），加载更快
  - 内存效率：支持 memory-mapped 读取

使用前请确保：
- pip install safetensors torch
"""

import os
import sys
import json
import argparse
from typing import List, Optional, Dict


def get_dtype_bytes(dtype_str: str) -> int:
    """根据 dtype 字符串返回每个元素占用的字节数"""
    dtype_map = {
        "torch.float32": 4, "torch.float": 4, "F32": 4,
        "torch.float16": 2, "torch.half": 2, "F16": 2,
        "torch.bfloat16": 2, "BF16": 2,
        "torch.int8": 1, "I8": 1,
        "torch.uint8": 1, "U8": 1,
        "torch.int32": 4, "I32": 4,
        "torch.int64": 8, "I64": 8,
        "torch.float8_e4m3fn": 1, "F8_E4M3": 1,
        "torch.float8_e5m2": 1, "F8_E5M2": 1,
    }
    return dtype_map.get(str(dtype_str), 4)  # 默认 4 bytes


def format_size(num_bytes: int) -> str:
    """将字节数格式化为可读字符串"""
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / 1024**3:.2f} GB"
    elif num_bytes >= 1024 ** 2:
        return f"{num_bytes / 1024**2:.2f} MB"
    elif num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes} B"


def inspect_safetensors(
    file_path: str,
    include_keywords: Optional[List[str]] = None,
    exclude_keywords: Optional[List[str]] = None,
    show_details: bool = True,
) -> Dict:
    """
    分析 safetensors 文件中的张量信息

    Args:
        file_path: safetensors 文件路径
        include_keywords: 仅显示包含这些关键字的张量
        exclude_keywords: 排除包含这些关键字的张量
        show_details: 是否打印详细信息

    Returns:
        统计信息字典
    """
    try:
        from safetensors import safe_open
    except ImportError:
        print("需要安装 safetensors: pip install safetensors")
        sys.exit(1)

    stats = {
        "file": file_path,
        "file_size": os.path.getsize(file_path),
        "total_params": 0,
        "total_bytes": 0,
        "tensor_count": 0,
        "dtypes": {},
        "tensors": [],
    }

    if show_details:
        print(f"\n{'='*80}")
        print(f"分析文件: {file_path}")
        print(f"文件大小: {format_size(stats['file_size'])}")
        print(f"{'='*80}")
        print(f"\n{'张量名称':<60s} {'形状':<25s} {'参数量':>12s} {'类型':>10s}")
        print("-" * 110)

    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in sorted(f.keys()):
            # 过滤逻辑
            if include_keywords:
                if not any(kw in key for kw in include_keywords):
                    continue
            if exclude_keywords:
                if any(kw in key for kw in exclude_keywords):
                    continue

            tensor = f.get_tensor(key)
            numel = tensor.numel()
            dtype = str(tensor.dtype)
            shape = tuple(tensor.shape)
            byte_size = numel * get_dtype_bytes(dtype)

            stats["total_params"] += numel
            stats["total_bytes"] += byte_size
            stats["tensor_count"] += 1
            stats["dtypes"][dtype] = stats["dtypes"].get(dtype, 0) + 1
            stats["tensors"].append({
                "name": key,
                "shape": shape,
                "numel": numel,
                "dtype": dtype,
            })

            if show_details:
                shape_str = str(shape)
                print(f"  {key:<58s} {shape_str:<25s} {numel:>12,} {dtype:>10s}")

    if show_details:
        print(f"\n{'='*80}")
        print(f"统计摘要")
        print(f"{'='*80}")
        print(f"  张量数量: {stats['tensor_count']}")
        print(f"  总参数量: {stats['total_params']:,}")
        print(f"  估算内存: {format_size(stats['total_bytes'])}")
        print(f"  数据类型分布: {stats['dtypes']}")
        if include_keywords:
            print(f"  包含关键字: {include_keywords}")
        if exclude_keywords:
            print(f"  排除关键字: {exclude_keywords}")

    return stats


def load_tensor_from_safetensors(file_path: str, tensor_key: str):
    """
    从 safetensors 文件中加载指定张量

    Args:
        file_path: safetensors 文件路径
        tensor_key: 张量的键名

    Returns:
        torch.Tensor
    """
    from safetensors import safe_open

    with safe_open(file_path, framework="pt", device="cpu") as f:
        available_keys = list(f.keys())
        if tensor_key not in available_keys:
            print(f"键 '{tensor_key}' 不存在。可用的键:")
            for k in available_keys[:20]:
                print(f"  {k}")
            if len(available_keys) > 20:
                print(f"  ... 共 {len(available_keys)} 个")
            return None
        return f.get_tensor(tensor_key)


def analyze_index_file(index_path: str) -> Dict:
    """
    分析 model.safetensors.index.json 分片索引文件

    该文件记录了每个参数（key）存储在哪个分片文件中。
    大模型的权重通常被分片存储（如 model-00001-of-00063.safetensors）。

    Args:
        index_path: index.json 文件路径

    Returns:
        分析结果字典
    """
    print(f"\n{'='*80}")
    print(f"分析分片索引: {index_path}")
    print(f"{'='*80}")

    with open(index_path, 'r') as f:
        index = json.load(f)

    # 元数据
    metadata = index.get("metadata", {})
    print(f"\n元数据:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")

    # 分片统计
    weight_map = index.get("weight_map", {})
    shard_to_keys: Dict[str, List[str]] = {}
    for key, shard in weight_map.items():
        if shard not in shard_to_keys:
            shard_to_keys[shard] = []
        shard_to_keys[shard].append(key)

    print(f"\n分片文件统计:")
    print(f"  总参数（key）数量: {len(weight_map)}")
    print(f"  分片文件数量: {len(shard_to_keys)}")
    print(f"\n  {'分片文件':<45s} {'包含张量数':>10s}")
    print(f"  {'-'*60}")
    for shard, keys in sorted(shard_to_keys.items()):
        print(f"  {shard:<45s} {len(keys):>10d}")

    return {
        "metadata": metadata,
        "total_keys": len(weight_map),
        "num_shards": len(shard_to_keys),
        "shard_to_keys": shard_to_keys,
    }


def batch_inspect(directory: str, exclude_keywords: Optional[List[str]] = None):
    """
    批量分析目录下所有 safetensors 文件

    Args:
        directory: 包含 safetensors 文件的目录
        exclude_keywords: 排除的关键字
    """
    print(f"\n{'='*80}")
    print(f"批量分析目录: {directory}")
    print(f"{'='*80}")

    safetensor_files = sorted([
        f for f in os.listdir(directory) if f.endswith('.safetensors')
    ])

    if not safetensor_files:
        print("目录中没有找到 .safetensors 文件")
        return

    total_params = 0
    total_bytes = 0
    total_file_size = 0

    for filename in safetensor_files:
        filepath = os.path.join(directory, filename)
        file_size = os.path.getsize(filepath)
        total_file_size += file_size

        stats = inspect_safetensors(
            filepath,
            exclude_keywords=exclude_keywords,
            show_details=False
        )
        total_params += stats["total_params"]
        total_bytes += stats["total_bytes"]

        print(f"  {filename:<50s} {stats['tensor_count']:>5d} 张量  "
              f"{stats['total_params']:>15,} 参数  {format_size(file_size):>10s}")

    print(f"\n总计:")
    print(f"  文件数量: {len(safetensor_files)}")
    print(f"  总参数量: {total_params:,}")
    print(f"  磁盘占用: {format_size(total_file_size)}")
    print(f"  估算 FP32 大小: {format_size(total_params * 4)}")
    print(f"  估算 FP16 大小: {format_size(total_params * 2)}")

    # 检查是否有 index 文件
    index_file = os.path.join(directory, "model.safetensors.index.json")
    if os.path.exists(index_file):
        analyze_index_file(index_file)


def main():
    parser = argparse.ArgumentParser(
        description="safetensors 权重文件分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析单个 safetensors 文件
  python 06_safetensors_inspector.py --file model.safetensors

  # 分析并过滤（仅显示 attention 相关）
  python 06_safetensors_inspector.py --file model.safetensors --include attn

  # 排除 MLP 层
  python 06_safetensors_inspector.py --file model.safetensors --exclude mlp

  # 批量分析目录
  python 06_safetensors_inspector.py --dir ./model_weights/

  # 提取指定张量
  python 06_safetensors_inspector.py --file model.safetensors --extract "model.embed_tokens.weight"

  # 分析 index 文件
  python 06_safetensors_inspector.py --index model.safetensors.index.json
        """
    )
    parser.add_argument("--file", type=str, help="单个 safetensors 文件路径")
    parser.add_argument("--dir", type=str, help="包含 safetensors 文件的目录路径")
    parser.add_argument("--index", type=str, help="model.safetensors.index.json 路径")
    parser.add_argument("--include", nargs="*", help="仅显示包含这些关键字的张量")
    parser.add_argument("--exclude", nargs="*", help="排除包含这些关键字的张量")
    parser.add_argument("--extract", type=str, help="提取指定张量并显示信息")
    args = parser.parse_args()

    if not any([args.file, args.dir, args.index]):
        parser.print_help()
        print("\n请指定 --file, --dir 或 --index 参数")
        return

    if args.file:
        if not os.path.exists(args.file):
            print(f"文件不存在: {args.file}")
            return

        if args.extract:
            tensor = load_tensor_from_safetensors(args.file, args.extract)
            if tensor is not None:
                print(f"\n张量: {args.extract}")
                print(f"  Shape: {tensor.shape}")
                print(f"  Dtype: {tensor.dtype}")
                print(f"  Min: {tensor.min().item():.6f}")
                print(f"  Max: {tensor.max().item():.6f}")
                print(f"  Mean: {tensor.float().mean().item():.6f}")
                print(f"  Std: {tensor.float().std().item():.6f}")
        else:
            inspect_safetensors(
                args.file,
                include_keywords=args.include,
                exclude_keywords=args.exclude,
            )

    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"目录不存在: {args.dir}")
            return
        batch_inspect(args.dir, exclude_keywords=args.exclude)

    if args.index:
        if not os.path.exists(args.index):
            print(f"文件不存在: {args.index}")
            return
        analyze_index_file(args.index)


if __name__ == "__main__":
    main()
