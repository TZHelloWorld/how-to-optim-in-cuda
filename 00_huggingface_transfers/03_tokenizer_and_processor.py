#!/usr/bin/env python3
"""
03_tokenizer_and_processor.py - Tokenizer 与 Processor 使用演示

本脚本演示了 HuggingFace Tokenizer 的核心功能：
1. 基本的编码/解码操作
2. 批量处理与 padding
3. 对话模板 (Chat Template)
4. 特殊 token 的使用
5. Tokenizer 的保存与加载
6. Processor（多模态前处理）的基本用法

使用前请确保：
- pip install transformers
"""

import argparse
import os

from transformers import AutoTokenizer


def demo_basic_tokenization(model_name: str):
    """
    演示 Tokenizer 的基本编码和解码操作

    Tokenizer 是将文本转为模型可处理的数字 ID 序列的核心组件。
    它的工作流程：文本 -> 分词 -> 转为 ID -> 模型输入
    """
    print("=" * 60)
    print("[1] 基本编码/解码操作")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text = "Hello, how are you doing today?"

    # --- 编码方式 1: tokenizer() 返回 BatchEncoding ---
    encoded = tokenizer(text)
    print(f"\n原始文本: '{text}'")
    print(f"input_ids:      {encoded['input_ids']}")
    print(f"attention_mask: {encoded['attention_mask']}")
    print(f"token 数量: {len(encoded['input_ids'])}")

    # --- 编码方式 2: tokenizer.encode() 仅返回 ID 列表 ---
    ids = tokenizer.encode(text)
    print(f"\ntokenizer.encode(): {ids}")

    # --- 编码方式 3: 不加特殊 token ---
    ids_no_special = tokenizer.encode(text, add_special_tokens=False)
    print(f"encode(add_special_tokens=False): {ids_no_special}")

    # --- 解码 ---
    decoded = tokenizer.decode(encoded['input_ids'])
    print(f"\n解码结果: '{decoded}'")

    decoded_no_special = tokenizer.decode(encoded['input_ids'], skip_special_tokens=True)
    print(f"解码(跳过特殊token): '{decoded_no_special}'")

    # --- 查看分词结果 ---
    tokens = tokenizer.tokenize(text)
    print(f"\n分词结果: {tokens}")

    # --- 查看词汇表信息 ---
    print(f"\n词汇表大小: {tokenizer.vocab_size}")
    print(f"模型最大长度: {tokenizer.model_max_length}")

    return tokenizer


def demo_batch_encoding(tokenizer):
    """
    演示批量编码和 padding

    在实际使用中，通常需要对多个句子进行批量处理，
    并 padding 到相同长度以便组成 batch。
    """
    print("\n" + "=" * 60)
    print("[2] 批量编码与 Padding")
    print("=" * 60)

    texts = [
        "Hello!",
        "How are you doing today?",
        "This is a much longer sentence that contains more tokens than the others.",
    ]

    # --- 不 padding（各序列长度不同）---
    encoded_no_pad = tokenizer(texts)
    print("\n不使用 padding:")
    for i, ids in enumerate(encoded_no_pad['input_ids']):
        print(f"  文本 {i}: 长度={len(ids)}, ids={ids[:10]}...")

    # --- padding 到最长序列 ---
    encoded_padded = tokenizer(texts, padding=True, return_tensors="pt")
    print(f"\npadding=True (pad 到 batch 最长):")
    print(f"  input_ids shape: {encoded_padded['input_ids'].shape}")
    print(f"  attention_mask shape: {encoded_padded['attention_mask'].shape}")
    print(f"  attention_mask[0]: {encoded_padded['attention_mask'][0].tolist()}")

    # --- padding + truncation ---
    encoded_trunc = tokenizer(
        texts, padding=True, truncation=True, max_length=10, return_tensors="pt"
    )
    print(f"\npadding=True, truncation=True, max_length=10:")
    print(f"  input_ids shape: {encoded_trunc['input_ids'].shape}")
    for i in range(len(texts)):
        print(f"  文本 {i}: {encoded_trunc['input_ids'][i].tolist()}")


def demo_special_tokens(tokenizer):
    """
    演示特殊 token 的使用

    特殊 token 是模型约定的特殊标记，如：
    - [CLS] / <s>: 序列开始
    - [SEP] / </s>: 序列结束
    - [PAD]: 填充
    - [UNK]: 未知词
    - [MASK]: 掩码（BERT 用）
    """
    print("\n" + "=" * 60)
    print("[3] 特殊 Token")
    print("=" * 60)

    print(f"\nPAD token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    print(f"EOS token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    print(f"BOS token: '{tokenizer.bos_token}' (id={tokenizer.bos_token_id})")
    print(f"UNK token: '{tokenizer.unk_token}' (id={tokenizer.unk_token_id})")

    if hasattr(tokenizer, 'sep_token') and tokenizer.sep_token:
        print(f"SEP token: '{tokenizer.sep_token}' (id={tokenizer.sep_token_id})")
    if hasattr(tokenizer, 'cls_token') and tokenizer.cls_token:
        print(f"CLS token: '{tokenizer.cls_token}' (id={tokenizer.cls_token_id})")
    if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token:
        print(f"MASK token: '{tokenizer.mask_token}' (id={tokenizer.mask_token_id})")

    # 查看所有特殊 token
    print(f"\n所有特殊 token: {tokenizer.all_special_tokens}")
    print(f"对应 ID: {tokenizer.all_special_ids}")


def demo_chat_template(model_name: str):
    """
    演示对话模板（Chat Template）

    对话模板用于将多轮对话格式化为模型所需的特定格式。
    不同模型的对话格式不同，chat_template 提供了统一的接口。
    """
    print("\n" + "=" * 60)
    print("[4] 对话模板 (Chat Template)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."},
        {"role": "user", "content": "Can you give me an example?"},
    ]

    # 检查是否支持 chat template
    if tokenizer.chat_template is None:
        print(f"\n{model_name} 不支持 chat template，跳过此演示")
        return

    # 方式 1: 返回格式化后的字符串
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,               # 返回字符串而不是 token IDs
        add_generation_prompt=True     # 添加 assistant 的开始标记
    )
    print(f"\n格式化后的文本:\n{text}")

    # 方式 2: 直接返回 token IDs
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    print(f"\nToken IDs shape: {token_ids.shape}")
    print(f"Token 数量: {token_ids.shape[-1]}")


def demo_save_tokenizer(tokenizer, save_dir: str = "./saved_tokenizer"):
    """
    演示 Tokenizer 的保存

    保存后会生成以下文件：
    - tokenizer.json: 完整分词器数据
    - tokenizer_config.json: 分词器配置
    - vocab.json: 词汇表
    - merges.txt: BPE 合并规则
    - special_tokens_map.json: 特殊 token 映射
    - added_tokens.json: 额外添加的 token
    """
    print("\n" + "=" * 60)
    print("[5] 保存 Tokenizer")
    print("=" * 60)

    tokenizer.save_pretrained(save_dir)

    print(f"\nTokenizer 已保存到: {save_dir}")
    print("保存的文件:")
    if os.path.exists(save_dir):
        for f in sorted(os.listdir(save_dir)):
            filepath = os.path.join(save_dir, f)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                print(f"  {f} ({size:,} bytes)")


def demo_processor_info():
    """
    介绍 Processor（多模态前处理器）的基本概念

    Processor = Tokenizer + ImageProcessor + VideoProcessor
    用于多模态模型（如 Qwen2.5-VL, LLaVA 等）的输入预处理。
    """
    print("\n" + "=" * 60)
    print("[6] Processor（多模态前处理器）概念说明")
    print("=" * 60)

    print("""
Processor 是多模态模型的统一前处理接口，它负责将不同类型的输入
（文本、图像、视频、音频等）转换为模型可以处理的统一格式。

使用方式:
  from transformers import AutoProcessor
  processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B")

核心方法:
  - processor(text=..., images=..., videos=...)  # 处理多模态输入
  - processor.apply_chat_template(messages)       # 应用对话模板
  - processor.save_pretrained(path)               # 保存

输入格式示例 (Qwen2.5-VL):
  messages = [{
      "role": "user",
      "content": [
          {"type": "image", "image": "path/to/image.jpg"},
          {"type": "text", "text": "描述这张图片"}
      ]
  }]

返回类型:
  BatchFeature - 包含模型所需的所有输入张量
  - input_ids: 文本 token IDs
  - attention_mask: 注意力掩码
  - pixel_values: 图像像素值（经过预处理）
  - image_grid_thw: 图像网格信息（部分模型）

注意事项:
  - 不同模型的 Processor 实现不同（resize、裁剪、归一化等策略不同）
  - 微调/校准时需要使用与训练时一致的 Processor
  - 对于纯文本模型，AutoProcessor 等价于 AutoTokenizer
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Tokenizer 与 Processor 使用演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 BERT tokenizer 演示基本功能
  python 03_tokenizer_and_processor.py --model bert-base-uncased

  # 使用 GPT-2 tokenizer
  python 03_tokenizer_and_processor.py --model gpt2

  # 使用本地模型的 tokenizer
  python 03_tokenizer_and_processor.py --model ./local_model
        """
    )
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="模型名称或本地路径")
    parser.add_argument("--save-dir", type=str, default="./saved_tokenizer",
                        help="Tokenizer 保存路径")
    args = parser.parse_args()

    # 运行所有演示
    tokenizer = demo_basic_tokenization(args.model)
    demo_batch_encoding(tokenizer)
    demo_special_tokens(tokenizer)
    demo_chat_template(args.model)
    demo_save_tokenizer(tokenizer, args.save_dir)
    demo_processor_info()


if __name__ == "__main__":
    main()
