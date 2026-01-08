#!/usr/bin/env python3
"""
01_download_model.py - HuggingFace 模型/数据集下载演示

本脚本演示了从 HuggingFace Hub 下载模型和数据集的多种方式：
1. 使用 snapshot_download 下载完整仓库
2. 使用 hf_hub_download 下载单个文件
3. 下载数据集
4. 高级选项（过滤、指定版本等）

使用前请确保：
- pip install -U huggingface_hub
- 如在国内，建议设置镜像：export HF_ENDPOINT=https://hf-mirror.com
- 如需下载受限模型，请先登录：huggingface-cli login
"""

import os
import argparse
from huggingface_hub import snapshot_download, hf_hub_download, login


def demo_snapshot_download(repo_id: str, local_dir: str, exclude_patterns: list = None):
    """
    使用 snapshot_download 下载完整模型仓库

    snapshot_download 会下载仓库的完整快照，支持：
    - ignore_patterns: 排除匹配的文件（如 *.pt）
    - allow_patterns: 仅下载匹配的文件
    - revision: 指定版本（分支/tag/commit）

    Args:
        repo_id: HuggingFace 仓库 ID，如 "THUDM/chatglm2-6b"
        local_dir: 本地保存路径
        exclude_patterns: 排除的文件模式列表
    """
    print(f"[snapshot_download] 开始下载: {repo_id}")
    print(f"  保存路径: {local_dir}")
    if exclude_patterns:
        print(f"  排除模式: {exclude_patterns}")

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=exclude_patterns,
        # resume_download=True,    # 默认已支持断点续传
        # revision="main",         # 指定分支/tag/commit
        # max_workers=8,           # 最大并行下载线程数
    )
    print(f"  下载完成: {path}")
    return path


def demo_single_file_download(repo_id: str, filename: str, local_dir: str):
    """
    使用 hf_hub_download 下载单个文件

    适用场景：
    - 只需要 config.json 查看模型配置
    - 只需要 tokenizer.json 测试分词器
    - 只需要某个特定的权重分片

    Args:
        repo_id: HuggingFace 仓库 ID
        filename: 要下载的文件名
        local_dir: 本地保存路径
    """
    print(f"\n[hf_hub_download] 下载单文件: {repo_id}/{filename}")

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
    )
    print(f"  文件保存至: {path}")
    return path


def demo_download_dataset(repo_id: str, local_dir: str):
    """
    下载 HuggingFace 数据集

    与模型下载类似，只需指定 repo_type="dataset"

    Args:
        repo_id: 数据集仓库 ID，如 "roneneldan/TinyStories"
        local_dir: 本地保存路径
    """
    print(f"\n[download_dataset] 下载数据集: {repo_id}")

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",  # 关键：指定为数据集类型
        local_dir=local_dir,
    )
    print(f"  数据集保存至: {path}")
    return path


def demo_download_specific_files(repo_id: str, local_dir: str):
    """
    下载仓库中的特定文件（通过 allow_patterns 过滤）

    适用场景：
    - 只需要配置文件和分词器（不需要权重）
    - 只需要 safetensors 格式（排除 .pt/.bin）
    """
    print(f"\n[download_specific] 仅下载配置和分词器文件: {repo_id}")

    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=[
            "config.json",
            "tokenizer*.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "*.jinja",
        ],
    )
    print(f"  文件保存至: {path}")

    # 列出下载的文件
    if os.path.exists(local_dir):
        print("  下载的文件:")
        for f in sorted(os.listdir(local_dir)):
            filepath = os.path.join(local_dir, f)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                print(f"    {f} ({size:,} bytes)")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace 模型/数据集下载演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载完整模型
  python 01_download_model.py --mode full --repo-id THUDM/chatglm2-6b --local-dir ./chatglm2-6b

  # 仅下载配置和分词器
  python 01_download_model.py --mode config-only --repo-id THUDM/chatglm2-6b --local-dir ./chatglm2-config

  # 下载单个文件
  python 01_download_model.py --mode single --repo-id THUDM/chatglm2-6b --filename config.json --local-dir ./

  # 下载数据集
  python 01_download_model.py --mode dataset --repo-id roneneldan/TinyStories --local-dir ./TinyStories

  # 下载模型但排除 .pt 文件
  python 01_download_model.py --mode full --repo-id mistralai/Mixtral-8x7B-v0.1 --local-dir ./Mixtral --exclude "*.pt"
        """
    )
    parser.add_argument("--mode", choices=["full", "single", "dataset", "config-only"],
                        default="config-only", help="下载模式")
    parser.add_argument("--repo-id", type=str, default="THUDM/chatglm2-6b",
                        help="仓库 ID")
    parser.add_argument("--local-dir", type=str, default="./downloaded_model",
                        help="本地保存路径")
    parser.add_argument("--filename", type=str, default="config.json",
                        help="单文件下载时的文件名")
    parser.add_argument("--exclude", type=str, nargs="*", default=None,
                        help="排除的文件模式（如 '*.pt' '*.bin'）")
    parser.add_argument("--login", action="store_true",
                        help="下载前先登录 HuggingFace")
    args = parser.parse_args()

    # 如果需要登录
    if args.login:
        print("请输入你的 HuggingFace Access Token:")
        login()

    # 根据模式执行下载
    if args.mode == "full":
        demo_snapshot_download(args.repo_id, args.local_dir, args.exclude)
    elif args.mode == "single":
        demo_single_file_download(args.repo_id, args.filename, args.local_dir)
    elif args.mode == "dataset":
        demo_download_dataset(args.repo_id, args.local_dir)
    elif args.mode == "config-only":
        demo_download_specific_files(args.repo_id, args.local_dir)


if __name__ == "__main__":
    main()
