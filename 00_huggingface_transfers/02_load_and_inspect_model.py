#!/usr/bin/env python3
"""
02_load_and_inspect_model.py - 模型加载与结构查看演示

本脚本演示了多种加载和查看模型结构的方式：
1. 从配置文件创建模型（不加载权重）
2. 使用 init_empty_weights 零内存查看超大模型结构
3. 加载预训练模型到 GPU
4. 使用 device_map 自动分配设备
5. 查看模型参数统计信息

使用前请确保：
- pip install transformers accelerate torch
"""

import argparse
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM


def demo_from_config(model_name_or_path: str, trust_remote_code: bool = False):
    """
    从配置文件创建模型结构（不加载权重）

    这种方式只创建模型的骨架结构，权重是随机初始化的。
    适用场景：
    - 快速查看模型结构
    - 测试模型是否能正确初始化
    - 了解参数形状和数量
    """
    print("=" * 60)
    print("[1] 从配置文件创建模型结构（不加载权重）")
    print("=" * 60)

    # 加载配置
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    print(f"\n模型类型: {config.model_type}")
    print(f"配置内容:\n{config}\n")

    # 从配置创建模型（随机初始化权重）
    model = AutoModel.from_config(config, trust_remote_code=trust_remote_code)

    # 打印模型结构
    print("模型结构:")
    print(model)

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小 (FP32): {total_params * 4 / 1024**3:.2f} GB")
    print(f"模型大小 (FP16): {total_params * 2 / 1024**3:.2f} GB")

    return model


def demo_init_empty_weights(model_name_or_path: str, trust_remote_code: bool = False):
    """
    使用 init_empty_weights 零内存查看模型结构

    原理：使用 torch.device('meta') 创建所有张量，
    只记录形状和 dtype，不分配实际内存。
    适用于查看超大模型（如 70B、456B 参数）的结构。
    """
    print("\n" + "=" * 60)
    print("[2] 使用 init_empty_weights 零内存查看模型结构")
    print("=" * 60)

    try:
        from accelerate import init_empty_weights
    except ImportError:
        print("需要安装 accelerate: pip install accelerate")
        return None

    with init_empty_weights():
        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
        model = AutoModel.from_config(config, trust_remote_code=trust_remote_code)

    # 打印模型结构
    print("\n模型结构:")
    print(model)

    # 打印每个参数的名称和形状
    print("\n参数详情:")
    total_params = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel
        print(f"  {name:60s} => {str(param.shape):30s} ({numel:>12,} params)")

    print(f"\n总参数量: {total_params:,}")
    print(f"估算模型大小 (FP32): {total_params * 4 / 1024**3:.2f} GB")
    print(f"估算模型大小 (BF16): {total_params * 2 / 1024**3:.2f} GB")

    return model


def demo_load_pretrained(model_name_or_path: str, trust_remote_code: bool = False):
    """
    加载预训练模型到 GPU

    这是最常用的模型加载方式，会从磁盘或 Hub 加载完整权重。
    """
    print("\n" + "=" * 60)
    print("[3] 加载预训练模型")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"目标设备: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype="auto",  # 自动选择 dtype
        # device_map="auto",   # 如需自动分配设备，取消注释
    )

    if device == "cuda":
        model = model.to(device)

    # 查看各层设备
    print("\n参数设备分布（前10个）:")
    for i, (name, param) in enumerate(model.named_parameters()):
        if i >= 10:
            print("  ...")
            break
        print(f"  {name}: device={param.device}, dtype={param.dtype}, shape={param.shape}")

    return model


def demo_device_map(model_name_or_path: str, trust_remote_code: bool = False):
    """
    使用 device_map="auto" 自动分配设备

    当模型太大无法放入单张 GPU 时，accelerate 会自动将模型层
    分配到多张 GPU、CPU 甚至磁盘上。

    device_map 选项:
    - "auto": 自动分配
    - "balanced": 均匀分配到所有 GPU
    - "balanced_low_0": 均匀分配但少用 GPU 0
    - "sequential": 按顺序填满每张 GPU
    - "cuda:0": 全部放到 GPU 0
    """
    print("\n" + "=" * 60)
    print("[4] 使用 device_map='auto' 自动分配设备")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA 不可用，跳过此示例")
        return None

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        device_map="auto",
        torch_dtype="auto",
    )

    # 查看设备分配情况
    print("\n设备分配详情:")
    if hasattr(model, 'hf_device_map'):
        for layer_name, device in model.hf_device_map.items():
            print(f"  {layer_name}: {device}")
    else:
        # 如果没有 hf_device_map，逐参数查看
        devices = set()
        for name, param in model.named_parameters():
            devices.add(str(param.device))
        print(f"  使用的设备: {devices}")

    return model


def print_model_summary(model):
    """
    打印模型的详细统计信息
    """
    print("\n" + "=" * 60)
    print("模型统计信息")
    print("=" * 60)

    # 按层类型统计参数
    layer_stats = {}
    for name, param in model.named_parameters():
        # 提取层类型（取最后一个 . 之前的部分作为模块名）
        parts = name.rsplit('.', 1)
        layer_type = parts[0] if len(parts) > 1 else name
        if layer_type not in layer_stats:
            layer_stats[layer_type] = {"count": 0, "params": 0}
        layer_stats[layer_type]["count"] += 1
        layer_stats[layer_type]["params"] += param.numel()

    # 打印前 20 个最大的层
    sorted_layers = sorted(layer_stats.items(), key=lambda x: x[1]["params"], reverse=True)
    print(f"\n{'模块名':<60s} {'参数数':<15s} {'张量数'}")
    print("-" * 90)
    for name, stats in sorted_layers[:20]:
        print(f"{name:<60s} {stats['params']:>12,}   {stats['count']}")
    if len(sorted_layers) > 20:
        print(f"... 共 {len(sorted_layers)} 个模块")


def main():
    parser = argparse.ArgumentParser(
        description="模型加载与结构查看演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 HuggingFace Hub 查看 BERT 模型结构
  python 02_load_and_inspect_model.py --model bert-base-uncased --demo config

  # 零内存查看本地大模型结构
  python 02_load_and_inspect_model.py --model ./local_model --demo empty --trust-remote-code

  # 加载预训练模型
  python 02_load_and_inspect_model.py --model ./local_model --demo load

  # 使用 device_map 加载
  python 02_load_and_inspect_model.py --model ./local_model --demo device-map --trust-remote-code
        """
    )
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="模型名称或本地路径")
    parser.add_argument("--demo", choices=["config", "empty", "load", "device-map", "all"],
                        default="config", help="要运行的演示")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="信任并执行远程代码（自定义模型需要）")
    args = parser.parse_args()

    if args.demo in ("config", "all"):
        model = demo_from_config(args.model, args.trust_remote_code)
        if model:
            print_model_summary(model)

    if args.demo in ("empty", "all"):
        demo_init_empty_weights(args.model, args.trust_remote_code)

    if args.demo in ("load", "all"):
        demo_load_pretrained(args.model, args.trust_remote_code)

    if args.demo in ("device-map", "all"):
        demo_device_map(args.model, args.trust_remote_code)


if __name__ == "__main__":
    main()
