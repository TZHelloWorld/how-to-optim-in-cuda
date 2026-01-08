#!/usr/bin/env python3
"""
07_inference_demo.py - 模型推理与文本生成完整演示

本脚本演示了使用 HuggingFace Transformers 进行模型推理的完整流程：
1. 基本文本生成（简单 prompt）
2. 对话式推理（Chat Template）
3. 生成参数详解（temperature, top_p, top_k 等）
4. 流式输出（Streaming）
5. KV Cache 的使用
6. 从修改后的配置重建模型并推理

使用前请确保：
- pip install transformers torch accelerate
- 准备好模型权重（本地路径或 HuggingFace Hub 模型名）
"""

import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
    TextStreamer,
)


def demo_basic_generation(model_id: str, trust_remote_code: bool = False):
    """
    基本文本生成

    最简单的推理流程：加载模型 -> 编码输入 -> 生成 -> 解码输出
    """
    print("=" * 60)
    print("[1] 基本文本生成")
    print("=" * 60)

    # 选择设备和数据类型
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"设备: {device}, 数据类型: {dtype}")

    # 加载模型和分词器
    print(f"加载模型: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
    )

    if device == "cpu":
        model = model.to(device)

    # 编码输入
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"\n输入: '{prompt}'")
    print(f"Token IDs: {inputs['input_ids'].tolist()}")
    print(f"Token 数量: {inputs['input_ids'].shape[-1]}")

    # 生成
    # model.generate() 实际调用的是 GenerationMixin.generate()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,          # 最多生成 50 个新 token
            do_sample=False,             # 贪心解码（确定性输出）
        )

    # 解码输出
    # 方式 1: 解码全部（包含输入）
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\n完整输出: {full_text}")

    # 方式 2: 仅解码生成的部分（去掉输入）
    output_ids = generated_ids[0][len(inputs['input_ids'][0]):]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"仅生成部分: {response}")

    return model, tokenizer


def demo_chat_inference(model, tokenizer):
    """
    对话式推理（使用 Chat Template）

    对话式模型需要将消息列表格式化为模型特定的格式，
    这通过 tokenizer.apply_chat_template() 实现。
    """
    print("\n" + "=" * 60)
    print("[2] 对话式推理")
    print("=" * 60)

    if tokenizer.chat_template is None:
        print("当前模型不支持 chat template，跳过此示例")
        return

    # 构造多轮对话
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Answer concisely."},
        {"role": "user", "content": "What is Python?"},
    ]

    # 应用对话模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # 添加 assistant 的开始标记
    )
    print(f"\n格式化后的输入:\n{text[:200]}...")

    # 编码并生成
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

    # 解码（去掉输入部分）
    output_ids = [
        out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(f"\nAssistant: {response}")


def demo_generation_params(model, tokenizer):
    """
    生成参数详解

    generation_config 控制了文本生成的行为，关键参数：
    - max_new_tokens: 最大生成 token 数
    - temperature: 温度（越高越随机）
    - top_p: nucleus sampling（概率累积阈值）
    - top_k: top-k sampling（只考虑概率最高的 k 个 token）
    - do_sample: 是否采样（False=贪心解码）
    - repetition_penalty: 重复惩罚
    - num_beams: beam search 的束宽
    """
    print("\n" + "=" * 60)
    print("[3] 生成参数对比")
    print("=" * 60)

    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    configs = {
        "贪心解码 (Greedy)": GenerationConfig(
            max_new_tokens=30,
            do_sample=False,
        ),
        "温度=0.3 (保守)": GenerationConfig(
            max_new_tokens=30,
            do_sample=True,
            temperature=0.3,
        ),
        "温度=1.0 (标准)": GenerationConfig(
            max_new_tokens=30,
            do_sample=True,
            temperature=1.0,
        ),
        "温度=1.5 (创意)": GenerationConfig(
            max_new_tokens=30,
            do_sample=True,
            temperature=1.5,
        ),
        "Top-p=0.9": GenerationConfig(
            max_new_tokens=30,
            do_sample=True,
            top_p=0.9,
        ),
        "Top-k=50": GenerationConfig(
            max_new_tokens=30,
            do_sample=True,
            top_k=50,
        ),
    }

    print(f"\n输入: '{prompt}'\n")
    for name, gen_config in configs.items():
        with torch.no_grad():
            output = model.generate(**inputs, generation_config=gen_config)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"  [{name}]")
        print(f"    {text}\n")


def demo_streaming(model, tokenizer):
    """
    流式输出（Streaming）

    使用 TextStreamer 实现逐 token 输出，
    模拟 ChatGPT 的打字效果。
    """
    print("=" * 60)
    print("[4] 流式输出 (Streaming)")
    print("=" * 60)

    prompt = "Explain machine learning in simple terms:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # TextStreamer 会在每个 token 生成后立即打印
    streamer = TextStreamer(tokenizer, skip_special_tokens=True)

    print(f"\n输入: '{prompt}'")
    print("输出 (流式): ", end="", flush=True)

    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            streamer=streamer,  # 关键：传入 streamer
        )
    print()


def demo_kv_cache(model, tokenizer):
    """
    KV Cache 的使用

    在自回归生成中，KV Cache 缓存已计算的 Key 和 Value，
    避免每一步都重新计算所有位置的注意力。

    流程：
    1. Prefill: 处理完整输入，生成初始 KV Cache
    2. Decode: 每步只处理新 token，复用 KV Cache
    """
    print("\n" + "=" * 60)
    print("[5] KV Cache 使用演示")
    print("=" * 60)

    from transformers import DynamicCache

    prompt = "Hello, world!"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"\n输入: '{prompt}'")
    print(f"输入 token 数: {inputs['input_ids'].shape[-1]}")

    # 创建空的 KV Cache
    past_key_values = DynamicCache()

    # Prefill: 一次性处理所有输入 token
    with torch.no_grad():
        outputs = model(
            **inputs,
            past_key_values=past_key_values,
            use_cache=True,
        )

    # 查看 KV Cache 信息
    cache = outputs.past_key_values
    print(f"\nPrefill 后的 KV Cache:")
    print(f"  缓存层数: {len(cache)}")

    if len(cache) > 0:
        # DynamicCache 存储方式
        if hasattr(cache, 'key_cache') and len(cache.key_cache) > 0:
            print(f"  第 0 层 Key shape: {cache.key_cache[0].shape}")
            print(f"  第 0 层 Value shape: {cache.value_cache[0].shape}")
            seq_len = cache.key_cache[0].shape[-2]
            print(f"  缓存的序列长度: {seq_len}")

    print(f"\n  KV Cache 类型: {type(cache).__name__}")
    print("""
KV Cache 说明:
  - Prefill 阶段: 处理完整序列，计算并缓存所有 K, V
  - Decode 阶段: 每步只输入 1 个新 token，
    新 token 的 Q 与缓存的 K, V 做注意力计算
  - 优势: 避免重复计算，将 O(n^2) 降为 O(n)
    """)


def demo_rebuild_and_infer(model_dir: str, trust_remote_code: bool = True):
    """
    从修改后的配置重建模型并推理

    流程：
    1. 修改 config.json 中的参数（如缩减层数）
    2. 使用修改后的 config 创建模型（随机权重）
    3. 保存模型权重
    4. 加载并推理

    适用于：模型结构测试、缩减版模型构建
    """
    print("\n" + "=" * 60)
    print("[6] 从配置重建模型")
    print("=" * 60)

    print(f"\n模型路径: {model_dir}")

    # 加载配置
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    print(f"模型类型: {config.model_type}")

    # 创建模型（随机初始化）
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    model = model.to(torch.bfloat16)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    # 保存
    save_dir = model_dir + "_rebuilt"
    print(f"保存到: {save_dir}")
    model.save_pretrained(save_dir)

    # 重新加载
    model = AutoModelForCausalLM.from_pretrained(
        save_dir,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # 推理测试
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    inputs = tokenizer("Hello!", return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"推理结果（随机权重，无意义）: {result}")


def main():
    parser = argparse.ArgumentParser(
        description="模型推理与文本生成演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 GPT-2 运行所有演示
  python 07_inference_demo.py --model gpt2

  # 使用本地模型
  python 07_inference_demo.py --model ./local_model --trust-remote-code

  # 仅运行基本生成
  python 07_inference_demo.py --model gpt2 --demo basic

  # 运行流式输出演示
  python 07_inference_demo.py --model gpt2 --demo streaming

  # 重建模型并推理
  python 07_inference_demo.py --model ./modified_model --demo rebuild --trust-remote-code
        """
    )
    parser.add_argument("--model", type=str, default="gpt2",
                        help="模型名称或本地路径")
    parser.add_argument("--demo", choices=["basic", "chat", "params", "streaming", "kvcache", "rebuild", "all"],
                        default="basic", help="运行哪个演示")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="信任并执行远程代码")
    args = parser.parse_args()

    model, tokenizer = None, None

    if args.demo in ("basic", "chat", "params", "streaming", "kvcache", "all"):
        model, tokenizer = demo_basic_generation(args.model, args.trust_remote_code)

    if args.demo in ("chat", "all") and model:
        demo_chat_inference(model, tokenizer)

    if args.demo in ("params", "all") and model:
        demo_generation_params(model, tokenizer)

    if args.demo in ("streaming", "all") and model:
        demo_streaming(model, tokenizer)

    if args.demo in ("kvcache", "all") and model:
        demo_kv_cache(model, tokenizer)

    if args.demo == "rebuild":
        demo_rebuild_and_infer(args.model, args.trust_remote_code)


if __name__ == "__main__":
    main()
