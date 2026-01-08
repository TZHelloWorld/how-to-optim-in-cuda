# Tokenizer 与 Chat Template 深度详解

> 本文详细介绍 LLM 推理流程中的核心组件 —— Tokenizer（分词器）和 Chat Template（对话模板），结合 HuggingFace 官方文档与 Qwen3-0.6B 模型实例，从原理到代码进行全面解析。
>
> 本文是 [`README.md`](./README.md) 第 6 章的深入阅读材料；Tokenizer 的基础用法与 API 速查参见 README 第 6 章及配套脚本 [`03_tokenizer_and_processor.py`](./03_tokenizer_and_processor.py)。

---

## 目录

- [1. 什么是 Tokenizer](#1-什么是-tokenizer)
  - [1.1 核心定位](#11-核心定位)
  - [1.2 为什么需要 Tokenizer](#12-为什么需要-tokenizer)
  - [1.3 分词算法演进](#13-分词算法演进)
- [2. 子词分词算法详解](#2-子词分词算法详解)
  - [2.1 Byte Pair Encoding (BPE)](#21-byte-pair-encoding-bpe)
  - [2.2 Byte-level BPE（Qwen3 使用的算法）](#22-byte-level-bpeqwen3-使用的算法)
  - [2.3 其他算法简介](#23-其他算法简介)
- [3. Tokenizer 核心操作详解](#3-tokenizer-核心操作详解)
  - [3.1 加载 Tokenizer](#31-加载-tokenizer)
  - [3.2 Tokenize：文本切分为 token](#32-tokenize文本切分为-token)
  - [3.3 Encode：文本编码为 ID](#33-encode文本编码为-id)
  - [3.4 Decode：ID 解码为文本](#34-decode-id-解码为文本)
  - [3.5 完整的编解码流程](#35-完整的编解码流程)
- [4. 特殊 Token 与词表结构](#4-特殊-token-与词表结构)
  - [4.1 什么是特殊 Token](#41-什么是特殊-token)
  - [4.2 Qwen3 的特殊 Token](#42-qwen3-的特殊-token)
  - [4.3 Byte-level BPE 的显示约定](#43-byte-level-bpe-的显示约定)
- [5. 什么是 Chat Template](#5-什么是-chat-template)
  - [5.1 核心概念](#51-核心概念)
  - [5.2 为什么需要 Chat Template](#52-为什么需要-chat-template)
  - [5.3 不同模型的格式差异](#53-不同模型的格式差异)
- [6. Chat Template 完整使用流程](#6-chat-template-完整使用流程)
  - [6.1 基本用法](#61-基本用法)
  - [6.2 add_generation_prompt 参数详解](#62-add_generation_prompt-参数详解)
  - [6.3 continue_final_message 参数](#63-continue_final_message-参数)
  - [6.4 Jinja2 模板原理](#64-jinja2-模板原理)
- [7. 推理全流程：从用户输入到模型输出](#7-推理全流程从用户输入到模型输出)
  - [7.1 流程图](#71-流程图)
  - [7.2 完整可运行代码](#72-完整可运行代码)
- [8. 多请求拼接与 Batch 推理](#8-多请求拼接与-batch-推理)
- [9. 总结](#9-总结)

---

## 1. 什么是 Tokenizer

### 1.1 核心定位

Tokenizer（分词器）是 LLM 系统中**文本与数字之间的桥梁**。它负责：

```
人类可读文本  ←──→  模型可处理的整数序列（token IDs）
```

LLM 本质上是一个数学函数，只能接受数字张量作为输入，输出也是数字（概率分布）。Tokenizer 就是连接人类语言和模型数学世界的翻译官。

### 1.2 为什么需要 Tokenizer

一个直观的问题：为什么不直接用字符的 ASCII/Unicode 编码？

| 方案 | 词表大小 | 序列长度 | 语义信息 | 缺点 |
|------|----------|----------|----------|------|
| 字符级 | ~256 | 极长 | 极少 | 单字符无法表达语义，序列过长导致计算量爆炸 |
| 词级 | 数十万+ | 短 | 丰富 | 词表巨大，无法处理未知词（OOV），无法处理词形变化 |
| **子词级** | 3~15万 | 适中 | 适中 | **最佳平衡**：常见词完整保留，生僻词拆分为有意义的子词 |

**子词分词的核心思想**：

- 高频词保持完整，如 `the`、`is`、`hello`
- 低频词拆分为高频子词组合，如 `annoyingly` → `["annoying", "ly"]`
- 保证任何文本都可以被表示（不会出现 `<unk>`）

### 1.3 分词算法演进

```
字符级分词 → 词级分词 → 子词分词（BPE / Unigram / WordPiece）→ Byte-level BPE
                                                                    ↑
                                                              当前主流方案
                                                         (GPT、Llama、Qwen 等)
```

---

## 2. 子词分词算法详解

### 2.1 Byte Pair Encoding (BPE)

BPE 是当前最流行的分词算法，被 GPT、Llama、Qwen 系列等主流模型采用。

**训练过程（构建词表）**：

```
第1步：从训练语料中提取所有唯一词及其频率
       ("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)

第2步：将每个词拆分为字符序列，作为初始词表
       基础词表 = ["b", "g", "h", "n", "p", "s", "u"]
       ("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)

第3步：统计所有相邻字符对的出现频率，合并最频繁的对
       "u"+"g" 出现 10+5+5=20 次 → 合并为 "ug"
       词表 = ["b", "g", "h", "n", "p", "s", "u", "ug"]
       ("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)

第4步：继续合并下一个最频繁的对
       "u"+"n" 出现 12+4=16 次 → 合并为 "un"
       词表 = ["b", "g", "h", "n", "p", "s", "u", "ug", "un"]

第5步：重复直到达到目标词表大小
       ...
```

**推理过程（分词）**：按照训练时学到的合并规则，对输入文本执行相同的合并操作。

### 2.2 Byte-level BPE（Qwen3 使用的算法）

标准 BPE 的基础词表是字符集，但要覆盖所有 Unicode 字符需要巨大的基础词表。**Byte-level BPE** 的改进是：

- **基础词表**：仅使用 256 个字节值（0x00~0xFF）
- **优势**：任何文本（包括中文、emoji、二进制数据）都能被编码，永远不会出现 `<unk>` token
- **代价**：多字节字符（如中文）初始时会被拆分为多个字节 token，但经过足够多的合并步骤后，常见的多字节字符组合会被合并为单个 token

**Qwen3 的词表**：约 151,000+ token，包含：
- 256 个字节基础 token
- ~150,000 个通过 BPE 合并学到的子词 token
- 若干特殊 token（如 `<|im_start|>`、`<|im_end|>` 等）

### 2.3 其他算法简介

| 算法 | 核心思路 | 代表模型 |
|------|----------|----------|
| **BPE** | 自底向上合并最频繁的字符对 | GPT、Llama、Qwen |
| **Unigram** | 自顶向下删除贡献最小的子词 | T5、mBART |
| **WordPiece** | 类似 BPE，但按似然增益选择合并对 | BERT、DistilBERT |
| **SentencePiece** | 直接在原始字节流上应用 BPE/Unigram，支持无空格语言 | Llama、T5 |

---

## 3. Tokenizer 核心操作详解

### 3.1 加载 Tokenizer

```python
from transformers import AutoTokenizer

# 方式一：从 HuggingFace Hub 加载（需要网络）
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# 方式二：从本地目录加载（推荐，无需网络）
tokenizer = AutoTokenizer.from_pretrained("/home/models/Qwen3-0.6B/")

# 查看词表大小
print(f"词表大小: {tokenizer.vocab_size}")
# 词表大小: 151665

# 查看特殊 token
print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
```

### 3.2 Tokenize：文本切分为 token

`tokenize()` 方法将文本切分为 token 字符串列表（不转换为 ID）：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/models/Qwen3-0.6B/")

text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', '?']

# 中文文本
text_cn = "你好世界"
tokens_cn = tokenizer.tokenize(text_cn)
print(tokens_cn)
# ['ä½', 'ł', 'å¥½', 'ä¸ĸ', 'çŀ', 'Į']  (多字节字符的 Byte-level BPE 表示)
```

**关键符号约定**：

| 显示符号 | 实际含义 | 说明 |
|----------|----------|------|
| `Ġ` | 空格（` `） | 表示该 token 前面有一个空格，如 `Ġare` = ` are` |
| `Ċ` | 换行符（`\n`） | 表示该位置是一个换行 |
| `ĉ` | 制表符（`\t`） | 表示该位置是一个 Tab |

这些是 Byte-level BPE 将不可见字符映射为可见 Unicode 字符的编码方式。

### 3.3 Encode：文本编码为 ID

将文本一步到位转换为模型需要的整数 ID 序列：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/models/Qwen3-0.6B/")

text = "Hello, how are you?"

# 方式一：使用 __call__（最常用，返回 BatchEncoding 字典）
encoded = tokenizer(text, return_tensors=None)
print(encoded)
# {'input_ids': [9707, 11, 1246, 525, 498, 30], 'attention_mask': [1, 1, 1, 1, 1, 1]}

# 方式二：使用 encode（仅返回 ID 列表）
ids = tokenizer.encode(text)
print(ids)
# [9707, 11, 1246, 525, 498, 30]

# 方式三：分步操作（tokenize → convert_tokens_to_ids）
tokens = tokenizer.tokenize(text)          # ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', '?']
ids = tokenizer.convert_tokens_to_ids(tokens)  # [9707, 11, 1246, 525, 498, 30]
print(ids)
```

**三种方式的区别**：

| 方式 | 返回类型 | 包含 attention_mask | 支持 return_tensors |
|------|----------|--------------------|--------------------|
| `tokenizer(text)` | BatchEncoding（dict） | 是 | 是（"pt"/"tf"/"np"） |
| `tokenizer.encode(text)` | List[int] | 否 | 否 |
| `tokenize() + convert_tokens_to_ids()` | List[str] → List[int] | 否 | 否 |

### 3.4 Decode：ID 解码为文本

将 token ID 序列还原为人类可读文本：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/models/Qwen3-0.6B/")

# 包含特殊 token 的 ID 序列
input_ids = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
             151644, 872, 198, 9309, 220, 16, 10, 16, 28, 18, 30, 151645, 198,
             151644, 77091, 198]

# 保留特殊 token（查看完整的对话结构）
text_with_special = tokenizer.decode(input_ids, skip_special_tokens=False)
print(text_with_special)
# <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# when 1+1=3?<|im_end|>
# <|im_start|>assistant

# 跳过特殊 token（获取纯文本内容）
text_without_special = tokenizer.decode(input_ids, skip_special_tokens=True)
print(text_without_special)
# system
# You are a helpful assistant.
# user
# when 1+1=3?
# assistant
```

### 3.5 完整的编解码流程

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/models/Qwen3-0.6B/")

# ===== 编码方向：文本 → Token IDs =====
original_text = "You are a helpful assistant."

# Step 1: 文本 → token 字符串
tokens = tokenizer.tokenize(original_text)
print(f"Tokens: {tokens}")
# Tokens: ['You', 'Ġare', 'Ġa', 'Ġhelpful', 'Ġassistant', '.']

# Step 2: token 字符串 → token IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Token IDs: {token_ids}")
# Token IDs: [2610, 525, 264, 10950, 17847, 13]

# ===== 解码方向：Token IDs → 文本 =====

# Step 3: token IDs → token 字符串
tokens_back = tokenizer.convert_ids_to_tokens(token_ids)
print(f"Tokens back: {tokens_back}")
# Tokens back: ['You', 'Ġare', 'Ġa', 'Ġhelpful', 'Ġassistant', '.']

# Step 4: token IDs → 文本（一步到位）
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded: {decoded_text}")
# Decoded: You are a helpful assistant.

# 验证完整性
assert decoded_text == original_text
print("编解码一致性验证通过！")
```

---

## 4. 特殊 Token 与词表结构

### 4.1 什么是特殊 Token

特殊 Token 是词表中**不对应自然语言文本**的控制符号。它们在模型训练（SFT/RLHF）阶段被引入，让模型学会识别对话结构、消息边界等元信息。

```
普通 token：   "hello"  "world"  "1+1"   ← 对应自然语言
特殊 token：   <|im_start|>  <|im_end|>  <|endoftext|>   ← 对应结构信号
```

### 4.2 Qwen3 的特殊 Token

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/models/Qwen3-0.6B/")

# 查看所有特殊 token
print(f"所有特殊 token: {tokenizer.all_special_tokens}")
print(f"对应 IDs: {tokenizer.all_special_ids}")

# Qwen3 关键特殊 token
special_tokens = {
    "<|im_start|>": 151644,   # 消息开始标记
    "<|im_end|>": 151645,     # 消息结束标记
    "<|endoftext|>": 151643,  # 文本结束标记（EOS）
}

# 验证
for token, expected_id in special_tokens.items():
    actual_id = tokenizer.convert_tokens_to_ids(token)
    print(f"{token} → ID: {actual_id} (expected: {expected_id})")
```

**特殊 Token 在对话中的作用**：

```
<|im_start|>system\n        ← 标记 system 消息开始
You are a helpful assistant.
<|im_end|>\n                ← 标记消息结束
<|im_start|>user\n          ← 标记 user 消息开始
when 1+1=3?
<|im_end|>\n                ← 标记消息结束
<|im_start|>assistant\n     ← 标记 assistant 消息开始（模型从这里开始生成）
```

### 4.3 Byte-level BPE 的显示约定

当我们用 `tokenizer.tokenize()` 查看 token 时，会看到一些特殊的 Unicode 字符：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/models/Qwen3-0.6B/")

# 包含特殊 token 的完整对话文本
input_text = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nwhen 1+1=3?<|im_end|>\n<|im_start|>assistant\n'

tokens = tokenizer.tokenize(input_text)
print(tokens)
# ['<|im_start|>', 'system', 'Ċ', 'You', 'Ġare', 'Ġa', 'Ġhelpful', 'Ġassistant', '.',
#  '<|im_end|>', 'Ċ', '<|im_start|>', 'user', 'Ċ', 'when', 'Ġ', '1', '+', '1', '=', '3', '?',
#  '<|im_end|>', 'Ċ', '<|im_start|>', 'assistant', 'Ċ']
```

**符号对照表**：

| Token 显示 | 实际文本 | Unicode 码点 | 说明 |
|-----------|----------|-------------|------|
| `Ġ` | ` `（空格） | U+0120 | 单词前的空格前缀 |
| `Ċ` | `\n`（换行） | U+010A | 换行符 |
| `ĉ` | `\t`（制表符） | U+0109 | 制表符 |
| `<\|im_start\|>` | 特殊标记 | - | 消息开始（不可分割） |
| `<\|im_end\|>` | 特殊标记 | - | 消息结束（不可分割） |

**为什么这样设计？**

Byte-level BPE 需要将所有 256 个字节值映射到"可显示"的 Unicode 字符。空格（0x20）被映射到 `Ġ`（U+0120），换行符（0x0A）被映射到 `Ċ`（U+010A）。这只是显示层面的约定，解码时会正确还原。

---

## 5. 什么是 Chat Template

### 5.1 核心概念

**关键洞察**：所有因果语言模型（无论是否经过 Chat 微调）本质上都在做一件事 —— **续写 token 序列**。

Chat 模型并没有"对话"的天然能力，它之所以能进行对话，是因为：

1. **训练数据格式化**：SFT 阶段将对话数据按特定格式组织，插入角色标记（如 `<|user|>`、`<|assistant|>`）
2. **模型学会识别结构**：模型学会了"看到 `<|assistant|>` 标记后就该输出回复"
3. **推理时复现格式**：推理时必须用**完全相同的格式**组织输入，模型才能正确响应

**Chat Template 就是这个"格式规则"的标准化定义** —— 它将通用的消息列表 `[{"role": ..., "content": ...}]` 转换为模型训练时使用的特定格式字符串。

### 5.2 为什么需要 Chat Template

如果你给模型输入错误的格式，即使模型能力很强，也会表现异常：

```
❌ 错误格式（缺少角色标记）：
"You are a helpful assistant. What is 1+1?"
→ 模型可能继续补全这句话，而不是回答问题

✅ 正确格式（使用 Chat Template）：
"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n"
→ 模型识别到 assistant 角色开始，正确生成回复
```

### 5.3 不同模型的格式差异

**同一个问题，不同模型需要完全不同的格式**：

```python
from transformers import AutoTokenizer

# ===== Qwen3 格式 =====
# <|im_start|>user
# Hello, how are you?<|im_end|>
# <|im_start|>assistant

# ===== Mistral 格式 =====
# <s>[INST] Hello, how are you? [/INST]

# ===== Llama 3 格式 =====
# <|begin_of_text|><|start_header_id|>user<|end_header_id|>
# Hello, how are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

# ===== Zephyr 格式 =====
# <|user|>
# Hello, how are you?</s>
# <|assistant|>
```

Chat Template 消除了这些差异 —— 开发者只需要用标准的 messages 格式，`apply_chat_template()` 自动处理不同模型的格式转换。

---

## 6. Chat Template 完整使用流程

### 6.1 基本用法

```python
from transformers import AutoTokenizer

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/models/Qwen3-0.6B/")

# 构建标准的 messages 列表
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "when 1+1=3?"},
]

# 应用 Chat Template —— 生成格式化文本（不 tokenize）
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,            # False: 返回字符串；True: 返回 token IDs
    add_generation_prompt=True  # 在末尾添加 assistant 角色开始标记
)

print(formatted_text)
# <|im_start|>system
# You are a helpful assistant.<|im_end|>
# <|im_start|>user
# when 1+1=3?<|im_end|>
# <|im_start|>assistant
```

**输出解析**：

```
<|im_start|>system\n         ← 角色标记 + 换行
You are a helpful assistant.  ← 消息内容
<|im_end|>\n                  ← 消息结束 + 换行
<|im_start|>user\n            ← 角色标记 + 换行
when 1+1=3?                   ← 消息内容
<|im_end|>\n                  ← 消息结束 + 换行
<|im_start|>assistant\n       ← 生成提示（add_generation_prompt=True 的效果）
```

### 6.2 add_generation_prompt 参数详解

这个参数决定了是否在末尾添加 assistant 角色的开始标记。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/models/Qwen3-0.6B/")

messages = [
    {"role": "user", "content": "Hello!"},
]

# ===== add_generation_prompt=False =====
text_no_prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
print("Without generation prompt:")
print(text_no_prompt)
# <|im_start|>user
# Hello!<|im_end|>
#                    ← 到此结束，模型不知道该做什么

print("---")

# ===== add_generation_prompt=True =====
text_with_prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print("With generation prompt:")
print(text_with_prompt)
# <|im_start|>user
# Hello!<|im_end|>
# <|im_start|>assistant
#                    ← 模型看到这个标记，知道该开始生成回复了
```

**使用场景**：

| 场景 | add_generation_prompt | 原因 |
|------|----------------------|------|
| **推理（生成回复）** | `True` | 告诉模型"现在轮到你说话了" |
| **训练（构造训练数据）** | `False` | 训练数据已包含完整对话，不需要额外提示 |
| **计算 loss/困惑度** | `False` | 评估时不需要生成提示 |

### 6.3 continue_final_message 参数

用于"预填充"（prefilling）场景 —— 你已经知道回复的开头，想让模型继续：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/models/Qwen3-0.6B/")

# 预填充 JSON 格式的开头
messages = [
    {"role": "user", "content": "List 3 colors in JSON format"},
    {"role": "assistant", "content": '{"colors": ["'},  # 预填充回复开头
]

# continue_final_message=True：不结束最后一条消息，让模型继续
text = tokenizer.apply_chat_template(
    messages, tokenize=False, continue_final_message=True
)
print(text)
# <|im_start|>user
# List 3 colors in JSON format<|im_end|>
# <|im_start|>assistant
# {"colors": ["
#              ← 没有 <|im_end|>，模型会继续从这里生成
```

> **注意**：`add_generation_prompt=True` 和 `continue_final_message=True` 不能同时使用，前者添加新消息开头，后者延续最后消息，逻辑冲突。

### 6.4 Jinja2 模板原理

Chat Template 本质上是一个 **Jinja2 模板**，存储在 `tokenizer_config.json` 的 `chat_template` 字段中（或独立的 `chat_template.jinja` 文件）。

Qwen3 的 Chat Template 大致逻辑（简化版）：

```jinja2
{%- for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}
```

你可以查看实际模板：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/models/Qwen3-0.6B/")

# 查看 chat template 内容
print(tokenizer.chat_template)
```

---

## 7. 推理全流程：从用户输入到模型输出

### 7.1 流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LLM 推理完整流程                                  │
└─────────────────────────────────────────────────────────────────────────┘

Step 1: 用户构建 messages
┌──────────────────────────────────────────┐
│ messages = [                             │
│   {"role": "system", "content": "..."},  │
│   {"role": "user", "content": "..."},    │
│ ]                                        │
└──────────────────────────────────────────┘
                    │
                    ▼
Step 2: apply_chat_template() —— 格式化
┌──────────────────────────────────────────────────────────────┐
│ "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n..."  │
│                  （格式化后的纯文本字符串）                      │
└──────────────────────────────────────────────────────────────┘
                    │
                    ▼
Step 3: tokenizer.encode() —— 编码为 token IDs
┌──────────────────────────────────────────────────────────────┐
│ [151644, 8948, 198, 2610, 525, 264, ..., 151644, 77091, 198]│
│                （整数序列，模型的实际输入）                       │
└──────────────────────────────────────────────────────────────┘
                    │
                    ▼
Step 4: model.generate() —— 模型推理
┌──────────────────────────────────────────────────────────────┐
│ 模型逐 token 生成：                                           │
│ [...原始IDs...] → [新token1] → [新token2] → ... → [EOS]      │
│                                                              │
│ Prefill: 一次性处理所有输入 token                               │
│ Decode:  逐步生成新 token，直到 EOS 或 max_length               │
└──────────────────────────────────────────────────────────────┘
                    │
                    ▼
Step 5: tokenizer.decode() —— 解码为文本
┌──────────────────────────────────────────────────────────────┐
│ "The equation 1+1=3 is mathematically incorrect..."           │
│                   （人类可读的回复文本）                          │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 完整可运行代码

```python
"""
完整的 LLM 推理流程演示
模型: Qwen3-0.6B
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==================== Step 0: 加载模型和 Tokenizer ====================
model_path = "/home/models/Qwen3-0.6B/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
)

# ==================== Step 1: 构建 messages ====================
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "when 1+1=3?"},
]

# ==================== Step 2: 应用 Chat Template ====================
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print("=== 格式化后的文本 ===")
print(formatted_text)
print()

# ==================== Step 3: 编码为 Token IDs ====================
# 方式一：直接使用 tokenizer（推荐）
inputs = tokenizer(formatted_text, return_tensors="pt").to(model.device)
print(f"=== Token IDs (共 {inputs.input_ids.shape[1]} 个) ===")
print(inputs.input_ids[0].tolist())
print()

# 方式二：分步操作（等价，用于理解原理）
tokens = tokenizer.tokenize(formatted_text)       # 文本 → token 字符串列表
token_ids = tokenizer.convert_tokens_to_ids(tokens)  # token 字符串 → IDs
print(f"=== 分步操作结果一致: {token_ids == inputs.input_ids[0].tolist()} ===")
print()

# ==================== Step 4: 模型推理 ====================
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

# ==================== Step 5: 解码输出 ====================
# 去掉输入部分，只保留模型新生成的 token
output_ids = generated_ids[0][inputs.input_ids.shape[1]:]
response = tokenizer.decode(output_ids, skip_special_tokens=True)

print("=== 模型回复 ===")
print(response)
```

---

## 8. 多请求拼接与 Batch 推理

在推理引擎（如 vLLM、SGLang）中，多个请求的 token 会被紧凑地拼接在一起处理。以下代码展示了这种场景的 token 结构：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/models/Qwen3-0.6B/")

# ===== 场景：两条不同请求的 token 片段被拼接在一起 =====
# 这模拟了推理引擎中 continuous batching 的场景

input_ids = [
    # 请求 A 的末尾部分
    3391, 220, 16, 10, 16, 28, 18, 30, 151645, 198, 151644, 77091, 198,
    # 请求 B 的末尾部分
    927, 268, 220, 16, 10, 16, 28, 18, 30, 151645, 198, 151644, 77091, 198
]

input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
print("=== 拼接后的文本 ===")
print(repr(input_text))
# 'then 1+1=3?<|im_end|>\n<|im_start|>assistant\nshen 1+1=3?<|im_end|>\n<|im_start|>assistant\n'

print()
print("=== 可视化显示 ===")
print(input_text)
# then 1+1=3?<|im_end|>
# <|im_start|>assistant
# shen 1+1=3?<|im_end|>
# <|im_start|>assistant

# 查看 token 切分
tokens = tokenizer.tokenize(input_text)
print()
print("=== Token 列表 ===")
print(tokens)
# ['then', 'Ġ', '1', '+', '1', '=', '3', '?', '<|im_end|>', 'Ċ',
#  '<|im_start|>', 'assistant', 'Ċ',
#  'sh', 'en', 'Ġ', '1', '+', '1', '=', '3', '?', '<|im_end|>', 'Ċ',
#  '<|im_start|>', 'assistant', 'Ċ']
```

**理解要点**：

1. `then` 和 `shen` 分别是两个不同请求的最后几个字符（截取片段）
2. 每个请求都以 `<|im_end|>\n<|im_start|>assistant\n` 结尾
3. 这种结构说明两个请求的消息内容不同（`then...` vs `shen...`），但格式模板相同
4. 推理引擎通过 attention mask 和 position IDs 来区分不同请求，避免跨请求注意力

---

## 9. 总结

### 核心概念速查表

| 概念 | 一句话解释 |
|------|-----------|
| **Tokenizer** | 文本 ↔ token ID 的双向转换器，使用 BPE 等算法将文本切分为子词 |
| **Token** | 模型处理文本的最小单位，可以是字、词、子词或特殊标记 |
| **词表（Vocabulary）** | Token 到 ID 的映射表，Qwen3 约有 151,000+ 条目 |
| **BPE** | 自底向上合并频繁字符对的分词算法 |
| **Byte-level BPE** | 以 256 字节为基础词表的 BPE，保证任何文本都能编码 |
| **Special Token** | 非自然语言的控制符号（如 `<\|im_start\|>`），标识对话结构 |
| **Chat Template** | Jinja2 模板，将 messages 列表转为模型训练时使用的特定格式 |
| **add_generation_prompt** | 在输入末尾添加 assistant 角色开始标记，提示模型开始回复 |
| **apply_chat_template()** | 应用模板的核心方法，一步完成格式转换 |

### 关键方法对照

```python
# ===== Tokenizer 核心 API =====
tokenizer = AutoTokenizer.from_pretrained(path)    # 加载
tokens = tokenizer.tokenize(text)                  # 文本 → token 字符串列表
ids = tokenizer.encode(text)                       # 文本 → token ID 列表
ids = tokenizer.convert_tokens_to_ids(tokens)      # token 字符串 → IDs
text = tokenizer.decode(ids)                       # IDs → 文本
encoded = tokenizer(text, return_tensors="pt")     # 文本 → 模型输入（推荐）

# ===== Chat Template API =====
text = tokenizer.apply_chat_template(              # messages → 格式化文本
    messages, tokenize=False, add_generation_prompt=True
)
ids = tokenizer.apply_chat_template(               # messages → token IDs
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
)
```

### 推理流程总结

```
messages → apply_chat_template() → tokenizer() → model.generate() → tokenizer.decode() → 回复文本
   │              │                      │               │                    │
   │         格式化为模型           编码为整数         自回归生成          解码为文本
   │         期望的格式             张量             新 token
   │
[{"role":"user",
  "content":"Hi"}]
```

---

## 参考资料

| 资源 | 链接 |
|------|------|
| HuggingFace Tokenizer 算法总结 | https://huggingface.co/docs/transformers/tokenizer_summary |
| HuggingFace Chat Templates 文档 | https://huggingface.co/docs/transformers/chat_templating |
| Qwen3 模型页面 | https://huggingface.co/Qwen/Qwen3-0.6B |
| BPE 原始论文 | https://huggingface.co/papers/1508.07909 |
| LLM Course - Tokenizers | https://huggingface.co/learn/llm-course/chapter6/1 |
