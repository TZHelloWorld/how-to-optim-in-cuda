# HuggingFace Transformers 实践指南

> 本文系统介绍 HuggingFace 生态的核心工作流：从模型下载、仓库文件格式、模型加载，到 Tokenizer/Processor 前处理、推理生成，再深入注意力机制、KV Cache 等原理，最后覆盖自定义模型重建、Transformers 源码架构与常见错误排查。全文配有 7 个可运行的演示脚本，各章节与脚本一一对应。

---

## 目录

- [第 1 章 概述与环境准备](#第-1-章-概述与环境准备)
- [第 2 章 模型下载](#第-2-章-模型下载)
- [第 3 章 模型仓库文件结构与 safetensors](#第-3-章-模型仓库文件结构与-safetensors)
- [第 4 章 模型加载的四种方式](#第-4-章-模型加载的四种方式)
- [第 5 章 模型结构显示原理](#第-5-章-模型结构显示原理)
- [第 6 章 Tokenizer 与 Chat Template](#第-6-章-tokenizer-与-chat-template)
- [第 7 章 Processor：多模态前处理](#第-7-章-processor多模态前处理)
- [第 8 章 模型推理与文本生成](#第-8-章-模型推理与文本生成)
- [第 9 章 张量维度变换](#第-9-章-张量维度变换)
- [第 10 章 注意力机制与掩码](#第-10-章-注意力机制与掩码)
- [第 11 章 多头注意力、KV Cache 与位置编码](#第-11-章-多头注意力kv-cache-与位置编码)
- [第 12 章 自定义模型结构与权重重建](#第-12-章-自定义模型结构与权重重建)
- [第 13 章 Transformers 源码架构解析](#第-13-章-transformers-源码架构解析)
- [第 14 章 常见错误与解决方案](#第-14-章-常见错误与解决方案)
- [第 15 章 参考资料与配套脚本](#第-15-章-参考资料与配套脚本)

---

## 第 1 章 概述与环境准备

### 1.1 全文路线图

使用 HuggingFace 生态的完整工作流可以概括为一条主线，本文各章按此展开：

```
获取模型（第 2 章）
    └─> 认识仓库文件（第 3 章：config / safetensors / tokenizer 文件）
            └─> 加载模型（第 4 章：from_config / from_pretrained / device_map / meta 设备）
                    └─> 查看结构（第 5 章：__repr__ 机制与常见层 IO）
                            └─> 输入前处理（第 6~7 章：Tokenizer / Chat Template / Processor）
                                    └─> 推理生成（第 8 章：generate / 生成参数 / 流式输出）
                                            └─> 原理深入（第 9~11 章：张量操作 / Attention / KV Cache）
                                                    └─> 工程进阶（第 12~13 章：模型重建 / 源码架构）
                                                            └─> 排错（第 14 章）
```

### 1.2 核心库与安装

| 库名 | 用途 | 安装命令 |
|------|------|----------|
| `huggingface_hub` | 模型/数据集下载与管理 | `pip install -U huggingface_hub` |
| `transformers` | 模型加载、推理、训练 | `pip install transformers` |
| `accelerate` | 大模型分布式加载、device_map | `pip install accelerate` |
| `safetensors` | 高效安全的权重文件格式 | `pip install safetensors` |
| `flash-attn` | Flash Attention 加速 | `pip install flash-attn --no-build-isolation` |
| `einops` | 张量维度变换工具 | `pip install einops` |

```bash
# 一键安装所有依赖
pip install -U huggingface_hub transformers accelerate safetensors einops
```

> **官方文档参考**：
> - huggingface_hub: https://huggingface.co/docs/huggingface_hub
> - Transformers: https://huggingface.co/docs/transformers
> - Accelerate: https://huggingface.co/docs/accelerate

---

## 第 2 章 模型下载

> 配套脚本：[`01_download_model.py`](./01_download_model.py)

### 2.1 配置缓存路径与镜像源

**默认缓存地址**：HuggingFace 将下载的模型缓存到 `~/.cache/huggingface/`，可通过 `HF_HOME` 环境变量自定义：

```bash
export HF_HOME="/your/custom/cache/path"
```

**国内镜像加速**：通过 `HF_ENDPOINT` 环境变量使用镜像源：

```bash
# 使用 hf-mirror.com 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

> **注意**：建议将环境变量写入 `~/.bashrc`（或 `~/.zshrc`）持久化，否则每次打开新终端都需要重新设置：
>
> ```bash
> echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
> source ~/.bashrc
> ```

### 2.2 登录与 Token 认证

部分模型仓库（如 Meta 的 Llama 系列）是**受限访问（gated）**的，需要先在官网完成授权，再用 Access Token 登录。

**获取 Token**：

1. 访问 https://huggingface.co/settings/tokens
2. 点击 "Create new token"
3. 下载场景选择 `Read` 权限即可

**CLI 登录与验证**：

```bash
huggingface-cli login
# 按提示输入你的 Access Token

huggingface-cli whoami
# 如果已登录，会返回当前用户名；否则提示未找到登录信息
```

> **安全提示**：Token 是敏感信息，切勿提交到公开仓库。登录后 Token 保存在 `~/.cache/huggingface/token` 文件中。

### 2.3 使用 CLI 下载

`huggingface-cli download` 的完整参数：

```
huggingface-cli download [-h]
    [--repo-type {model,dataset,space}]    # 仓库类型，默认 model
    [--revision REVISION]                   # Git 版本（分支名/tag/commit hash）
    [--include [INCLUDE ...]]               # glob 模式，指定要下载的文件
    [--exclude [EXCLUDE ...]]               # glob 模式，排除不需要的文件
    [--cache-dir CACHE_DIR]                 # 缓存目录
    [--local-dir LOCAL_DIR]                 # 下载到指定本地目录
    [--force-download]                      # 强制重新下载
    [--token TOKEN]                         # Access Token
    [--quiet]                               # 静默模式
    [--max-workers MAX_WORKERS]             # 最大并行下载线程数（默认 8）
    repo_id                                 # 仓库 ID（如 username/repo-name）
    [filenames ...]                         # 指定要下载的文件
```

**示例 1：下载完整模型到本地目录**

```bash
huggingface-cli download \
    --resume-download \
    THUDM/chatglm2-6b \
    --local-dir chatglm2-6b
```

- `--resume-download`：断点续传，中断后可继续下载而非从头开始；
- `--local-dir`：下载到指定文件夹（而非缓存目录）。

**示例 2：排除特定文件**

```bash
huggingface-cli download \
    --resume-download \
    mistralai/Mixtral-8x7B-v0.1 \
    --local-dir /workspace/model/Mixtral-8x7B-v0.1 \
    --exclude "*.pt"
```

`--exclude "*.pt"` 排除 `.pt` 格式权重（只保留 `.safetensors`），节省磁盘空间——同一模型仓库常同时存有多种格式的权重，重复下载没有意义。

**示例 3：仅下载特定文件**

```bash
huggingface-cli download \
    THUDM/chatglm2-6b \
    config.json tokenizer.json \
    --local-dir chatglm2-6b-config
```

### 2.4 使用 Python API 下载

适合集成到自动化脚本。

**方式一：`snapshot_download` 下载整仓快照（推荐）**

```python
from huggingface_hub import snapshot_download

SAVED_DIR = "/path/to/save"

snapshot_download(
    repo_id="mistralai/Mixtral-8x7B-v0.1",
    ignore_patterns=["*.pt"],     # 排除匹配的文件
    # allow_patterns=[...],       # 或仅下载匹配的文件
    # revision="main",            # 指定分支/tag/commit
    # max_workers=8,              # 并行下载线程数
    local_dir=SAVED_DIR,
)
```

**方式二：`hf_hub_download` 下载单个文件**

```python
from huggingface_hub import hf_hub_download

# 仅下载 config.json（适合只想看模型配置的场景）
hf_hub_download(
    repo_id="THUDM/chatglm2-6b",
    filename="config.json",
    local_dir="./chatglm2-6b-config"
)
```

### 2.5 下载数据集

与模型下载相同，仅需指定仓库类型为 `dataset`：

```bash
huggingface-cli download \
    --repo-type dataset \
    --resume-download \
    roneneldan/TinyStories \
    --local-dir ./TinyStories
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="roneneldan/TinyStories",
    repo_type="dataset",      # 关键：指定为数据集类型
    local_dir="./TinyStories"
)
```

---

## 第 3 章 模型仓库文件结构与 safetensors

> 配套脚本：[`06_safetensors_inspector.py`](./06_safetensors_inspector.py)

### 3.1 仓库文件清单

从 HuggingFace 下载的模型目录通常包含以下文件，理解每个文件的职责是后续加载与定制的基础：

| 文件名 | 作用 | 说明 |
|--------|------|------|
| `config.json` | 模型配置 | 定义模型结构参数（层数、隐藏维度、激活函数等），是模型初始化的核心依据 |
| `model.safetensors` | 模型权重 | safetensors 格式的模型参数。大模型分片为 `model-00001-of-00063.safetensors` 等多个文件 |
| `model.safetensors.index.json` | 分片索引 | 记录每个参数（key）存储在哪个分片文件中 |
| `tokenizer.json` | 分词器数据 | 完整的分词器信息（词汇表 + 分词规则），fast tokenizer 的核心文件 |
| `tokenizer_config.json` | 分词器配置 | 分词器类型、特殊 token 定义、模板配置等 |
| `vocab.json` | 词汇表 | 词汇到 ID 的映射表（部分模型使用） |
| `merges.txt` | BPE 合并规则 | BPE（Byte-Pair Encoding）分词算法的合并规则文件 |
| `generation_config.json` | 生成配置 | 文本生成相关参数（max_length、temperature、top_p 等） |
| `special_tokens_map.json` | 特殊 token 映射 | 定义 `[PAD]`、`[CLS]`、`[SEP]`、`[EOS]` 等特殊 token |
| `added_tokens.json` | 额外 token | 用户通过 `tokenizer.add_tokens()` 添加的自定义 token |
| `chat_template.jinja` | 对话模板 | Jinja2 格式的对话模板，供 `apply_chat_template` 使用 |
| `configuration_xxx.py` | 自定义配置类 | `trust_remote_code=True` 时使用的自定义配置代码 |
| `modeling_xxx.py` | 自定义模型类 | `trust_remote_code=True` 时使用的自定义模型结构代码 |

按职责可归为四组：**结构定义**（config.json + 自定义 `.py`）、**权重**（safetensors 及索引）、**分词**（tokenizer 系列文件）、**生成行为**（generation_config.json、chat_template.jinja）。

### 3.2 safetensors 格式

safetensors 是 HuggingFace 推出的新一代权重存储格式，相比传统的 PyTorch `.bin`（pickle 序列化）文件：

- **安全性**：不使用 pickle，避免反序列化时的代码注入风险；
- **速度**：支持零拷贝（zero-copy）读取，加载更快；
- **内存效率**：支持惰性加载与 memory-mapped 读取——这正是下一节"不加载模型也能看权重"的基础。

> 官方仓库：https://github.com/huggingface/safetensors

### 3.3 不加载模型查看权重

利用 safetensors 的惰性读取能力，可以在**不加载完整模型**的情况下查看权重文件的元信息：

```python
from safetensors import safe_open

file_path = "model-00001-of-00007.safetensors"

with safe_open(file_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(f"Name: {key}")
        print(f"Shape: {tensor.shape}")
        print(f"Dtype: {tensor.dtype}")
        print("-" * 50)
```

配套脚本 [`06_safetensors_inspector.py`](./06_safetensors_inspector.py) 在此基础上实现了完整的分析工具：

```bash
# 分析单个 safetensors 文件（张量名称/形状/参数量/类型 + 统计摘要）
python 06_safetensors_inspector.py --file model.safetensors

# 按关键字过滤（仅显示 attention 相关张量）
python 06_safetensors_inspector.py --file model.safetensors --include attn

# 排除 MLP 层
python 06_safetensors_inspector.py --file model.safetensors --exclude mlp

# 批量分析目录（自动汇总所有分片 + 解析 index.json）
python 06_safetensors_inspector.py --dir ./model_weights/

# 提取指定张量并显示统计信息（min/max/mean/std）
python 06_safetensors_inspector.py --file model.safetensors --extract "model.embed_tokens.weight"

# 分析分片索引文件
python 06_safetensors_inspector.py --index model.safetensors.index.json
```

---

## 第 4 章 模型加载的四种方式

> 配套脚本：[`02_load_and_inspect_model.py`](./02_load_and_inspect_model.py)

四种加载方式按"是否加载权重、权重放在哪"区分，适用于不同场景：

| 方式 | 加载权重？ | 内存占用 | 适用场景 |
|------|:---:|---------|---------|
| `from_config` | 否（随机初始化） | 完整结构内存 | 查看结构、测试初始化 |
| `from_pretrained` + `.to(device)` | 是 | 单卡显存 | 常规单卡推理 |
| `from_pretrained` + `device_map` | 是 | 多卡/CPU 分摊 | 超大模型 |
| `init_empty_weights` | 否（meta 设备） | **零内存** | 查看超大模型结构 |

### 4.1 从配置文件创建模型结构（不加载权重）

仅从 `config.json` 创建模型骨架，权重随机初始化：

```python
from transformers import AutoConfig, AutoModel

# 从 config 创建模型结构（权重随机初始化）
# AutoConfig 既可以传模型目录，也可以直接指向 config.json 文件
config = AutoConfig.from_pretrained("/mnt/model/DeepSeek-V3-0324/config.json")
model = AutoModel.from_config(config)

# 验证模型结构
print(model)

# 查看参数总量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# 或者输出每个参数 key 和对应的 shape
for name, param in model.named_parameters():
    print(f"{name} :==> {param.shape}")
```

两点源码层面的说明（详见第 13 章）：

- `AutoModel.from_config(config)` 实际调用的是 `PreTrainedModel._from_config()` 方法。**该路径不包含量化逻辑**——即使 `config.json` 中带有 `quantization_config`，也不会替换量化 Linear。量化流程只挂在 `from_pretrained()` 路径上（第 13.6 节）；
- `print(config)` 调用的是 `config.__repr__()`，**不会显示 `quantization_config`**；需要通过 `config.quantization_config` 显式访问，或用 `dir(config)` 查看全部属性。

### 4.2 加载预训练模型到 GPU

最常规的方式——加载完整权重并放到 GPU：

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_path = "./local_model_dir"  # 需包含 config.json 和权重文件

tokenizer = AutoTokenizer.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(model_path).to(device)

# 验证设备
for name, param in model.named_parameters():
    print(f"{name} device: {param.device}")
    break  # 只看第一个参数即可
```

### 4.3 大模型加载：device_map

对于数十/数百 GB 的超大模型，单卡放不下。`accelerate` 提供自动设备映射，将模型层分配到多张 GPU 甚至 CPU：

```bash
pip install accelerate
```

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",       # 自动分配到可用 GPU/CPU（由 accelerate 实现）
    torch_dtype="auto",      # 自动选择数据类型（通常为 bf16/fp16）
    # low_cpu_mem_usage=True,  # 加载时降低 CPU 内存峰值（分片加载、避免整份权重驻留内存）
)
```

> **参数命名说明**：新版本 Transformers 已将 `torch_dtype` 重命名为 `dtype`（`dtype="auto"`），旧名称目前仍兼容。`device_map` 指定时会隐式启用 `low_cpu_mem_usage=True`。

**`device_map` 的可选值**：

| 值 | 行为 |
|----|------|
| `"auto"` | 自动将模型层分配到所有可用 GPU，溢出部分放到 CPU |
| `"balanced"` | 尽量均匀地分配到所有 GPU |
| `"balanced_low_0"` | 均匀分配但尽量少用 GPU 0（留显存给推理） |
| `"sequential"` | 按顺序填满每张 GPU |
| `"cuda:0"` | 全部放到 GPU 0 |
| 自定义 dict | 手动指定每一层的设备 |

**查看各层设备分布**：

```python
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")
```

可能的输出：

```
model.layers.0.self_attn.q_proj.weight: cuda:0
model.layers.15.block_sparse_moe.experts.10.w2.weight: cuda:2
...
model.layers.79.block_sparse_moe.experts.31.w3.weight: meta  # offload 到 CPU
```

> **注意**：出现 `meta` 设备说明该层被 offload 到 CPU 或磁盘。显存不足时会有如下警告：
>
> ```
> UserWarning: Current model requires XXX bytes of buffer for offloaded layers,
> which seems does not fit any GPU's remaining memory.
> ```

也可以通过 `model.hf_device_map` 直接查看层级分配表（使用 device_map 加载时该属性自动生成）。

### 4.4 init_empty_weights：零内存查看结构

`accelerate` 的 `init_empty_weights` 上下文管理器在**不分配任何内存**的情况下创建模型，适合查看 70B、456B 等超大模型的结构与参数量：

```python
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModel

with init_empty_weights():
    config = AutoConfig.from_pretrained("path/to/config.json", trust_remote_code=True)
    model = AutoModel.from_config(config, trust_remote_code=True)
    print(model)

# 打印参数总量（在 with 外部也可以访问）
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# 查看每个参数的形状
for name, param in model.named_parameters():
    print(f"{name} :==> {param.shape}")
```

**原理**：内部使用 `torch.device('meta')` 创建所有张量。meta 是 PyTorch 的虚拟设备，只记录张量元数据（shape、dtype），不分配实际内存。

**参数量与显存估算**（脚本中的实用技巧）：

```python
total_params = sum(p.numel() for p in model.parameters())
print(f"模型大小 (FP32): {total_params * 4 / 1024**3:.2f} GB")
print(f"模型大小 (BF16): {total_params * 2 / 1024**3:.2f} GB")
```

---

## 第 5 章 模型结构显示原理

`print(model)` 输出的层级结构是怎么来的？理解该机制，既能读懂任意模型的结构输出，也能为自定义模块添加有用的显示信息。

### 5.1 核心机制：nn.Module 的 `__repr__`

`print(model)` 实际调用模型的 `__repr__()` 方法（PyTorch 中 `__str__ = __repr__`）：

```python
# nn.Module 中的 __repr__ 方法（简化版）
def __repr__(self):
    extra_lines = []
    extra_repr = self.extra_repr()
    if extra_repr:
        extra_lines = extra_repr.split('\n')

    child_lines = []
    for key, module in self._modules.items():
        mod_str = repr(module)
        child_lines.append('(' + key + '): ' + mod_str)

    lines = extra_lines + child_lines
    main_str = self._get_name() + '('
    if lines:
        main_str += '\n  ' + '\n  '.join(lines) + '\n'
    main_str += ')'
    return main_str
```

三个关键点：

- `self._modules`：存储所有子模块的有序字典（`OrderedDict`），递归 `repr` 形成层级缩进；
- `extra_repr()`：每个模块可重写此方法提供额外信息（如 `Linear(in_features=512, out_features=512)` 中括号内的内容）；
- `_get_name()`：模块类名。

### 5.2 `_modules` 的自动注册机制

在 `__init__` 中将 `nn.Module` 实例赋值给 `self` 的属性时，PyTorch 通过 `__setattr__` 拦截并自动注册到 `_modules`：

```python
# 以下两种写法等价：
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)       # 自动注册到 self._modules
        self.relu = nn.ReLU()               # 自动注册到 self._modules

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_module('fc1', nn.Linear(10, 20))       # 显式注册
        self.register_module('relu', nn.ReLU())          # 同上
```

### 5.3 自定义 extra_repr：以 Qwen2RMSNorm 为例

```python
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
```

重写 `extra_repr` 后，`print(model)` 中该模块显示为 `Qwen2RMSNorm((896,), eps=1e-06)`，一眼可见维度与超参。

RMSNorm 的数学公式：

$$x' = \frac{x}{\sqrt{\text{RMS}^2 + \epsilon}} \times \gamma$$

其中 $\gamma$ 是可学习参数 `self.weight`，$\epsilon$ 防止除零。与 LayerNorm 的区别：RMSNorm **不减去均值**，只用均方值归一化，计算更高效。

### 5.4 常见网络层的输入输出

读懂结构输出后，还需要知道各类层对输入形状的要求。

**Linear 层**：

```python
linear = nn.Linear(in_features=8, out_features=4)
```

| 输入形状 | 输出形状 | 说明 |
|----------|----------|------|
| `(8,)` | `(4,)` | 1D 输入 |
| `(6, 8)` | `(6, 4)` | 2D 输入（batch） |
| `(10, 6, 8)` | `(10, 6, 4)` | 3D 输入，保留前导维度 |
| `(*, 8)` | `(*, 4)` | 任意前导维度，仅最后一维变换 |

数学操作：$y = xW^T + b$，其中 $W$ 的形状为 `(out_features, in_features)`。

**Conv2d 层**：

```python
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
```

| 输入形状 | 输出形状 | 说明 |
|----------|----------|------|
| `(3, 32, 32)` | `(16, 30, 30)` | 单张图像（unbatched） |
| `(64, 3, 32, 32)` | `(64, 16, 30, 30)` | 批量图像 |
| `(7, 64, 3, 32, 32)` | 报错 | 仅支持 3D/4D 输入 |

输出尺寸计算：$H_{out} = \lfloor\frac{H_{in} + 2 \times \text{padding} - \text{kernel\_size}}{\text{stride}}\rfloor + 1$

**TransformerEncoderLayer**：

```python
transformer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
```

| 输入形状 | 输出形状 | 说明 |
|----------|----------|------|
| `(512,)` | 报错 | 至少需要 2D |
| `(32, 512)` | `(32, 512)` | `(seq_len, d_model)` |
| `(64, 32, 512)` | `(64, 32, 512)` | `(batch, seq_len, d_model)` 或 `(seq_len, batch, d_model)` |
| `(8, 64, 32, 512)` | 报错 | 最多 3D |

> **注意**：默认 `batch_first=False`，即输入格式为 `(seq_len, batch_size, d_model)`；设置 `batch_first=True` 后变为 `(batch_size, seq_len, d_model)`。这是维度错误的高发点。

---

## 第 6 章 Tokenizer 与 Chat Template

> 配套脚本：[`03_tokenizer_and_processor.py`](./03_tokenizer_and_processor.py)
>
> **深入阅读**：本目录的 [`tokenizer_and_chat_template.md`](./tokenizer_and_chat_template.md) 对 BPE/Byte-level BPE 分词算法、特殊 Token、Chat Template 的 Jinja2 原理、推理全流程做了完整的深度解析（以 Qwen3-0.6B 为实例），本章仅覆盖核心用法。

### 6.1 基本使用

Tokenizer 是文本与 token ID 序列之间的双向转换器，通过 `AutoTokenizer` 统一加载：

```python
from transformers import AutoTokenizer

model_path = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

prompt = "Hello, how are you today?"
encoded = tokenizer(prompt, add_special_tokens=False)

# encoded 是 BatchEncoding 类型，包含以下字段：
print(encoded.input_ids)       # token ID 序列，如 [7592, 1010, 2129, 2024, 2017, 2651, 1029]
print(encoded.attention_mask)  # 注意力掩码，1 表示有效 token，0 表示 padding
```

### 6.2 核心 API 一览

| 方法 | 功能 | 说明 |
|------|------|------|
| `tokenizer(text)` | 编码文本 | 返回 `input_ids` + `attention_mask`，支持 `return_tensors="pt"` |
| `tokenizer.encode(text)` | 编码为 ID 列表 | 仅返回 `[101, 7592, ...]` |
| `tokenizer.tokenize(text)` | 切分为 token 字符串 | 查看分词结果（不转 ID） |
| `tokenizer.decode(ids)` | ID 序列解码回文本 | `skip_special_tokens=True` 可去掉特殊 token |
| `tokenizer.batch_decode(ids_list)` | 批量解码 | 返回文本列表 |
| `tokenizer.apply_chat_template(messages)` | 应用对话模板 | 将多轮对话格式化为模型输入 |
| `tokenizer.add_tokens(new_tokens)` | 添加新 token | 扩展词汇表 |
| `tokenizer.save_pretrained(path)` | 保存分词器 | 导出所有分词器文件 |

**批量编码与 padding**（组 batch 的标准操作）：

```python
texts = ["Hello!", "How are you doing today?", "A much longer sentence ..."]

# padding 到 batch 内最长序列，返回张量
encoded = tokenizer(texts, padding=True, return_tensors="pt")
# encoded.input_ids.shape      -> (3, max_len)
# encoded.attention_mask       -> padding 位置为 0

# 加上截断
encoded = tokenizer(texts, padding=True, truncation=True, max_length=10, return_tensors="pt")
```

**特殊 token 查看**：

```python
print(tokenizer.pad_token, tokenizer.pad_token_id)
print(tokenizer.eos_token, tokenizer.eos_token_id)
print(tokenizer.all_special_tokens)   # 所有特殊 token
print(tokenizer.all_special_ids)      # 对应 ID
```

### 6.3 对话模板（Chat Template）

对话式模型的训练数据按特定格式组织（插入 `<|im_start|>` 等角色标记），推理时必须用**完全相同的格式**组织输入。`apply_chat_template` 将标准的 messages 列表自动转换为该模型的专属格式：

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
]

# 将消息格式化为模型输入文本
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,            # False: 返回字符串; True: 返回 token IDs
    add_generation_prompt=True # 自动添加 assistant 回复的起始标记
)

# 编码为模型输入张量
model_inputs = tokenizer(text, return_tensors="pt")
```

`add_generation_prompt` 的使用场景：

| 场景 | 取值 | 原因 |
|------|------|------|
| 推理（生成回复） | `True` | 告诉模型"现在轮到 assistant 说话了" |
| 训练（构造训练数据） | `False` | 训练数据已包含完整对话 |
| 计算 loss/困惑度 | `False` | 评估时不需要生成提示 |

> 模板本质、`continue_final_message` 预填充、不同模型格式差异等，详见 [`tokenizer_and_chat_template.md`](./tokenizer_and_chat_template.md) 第 5~6 章。

### 6.4 Tokenizer 的保存与重导出

部署或微调前，常需要将 Tokenizer 独立保存：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/path/to/original/model")
tokenizer.save_pretrained("./saved_tokenizer")
```

保存后的文件：

```
./saved_tokenizer/
├── added_tokens.json          # 额外添加的 token
├── chat_template.jinja        # 对话模板
├── merges.txt                 # BPE 合并规则
├── special_tokens_map.json    # 特殊 token 映射
├── tokenizer_config.json      # 分词器配置
├── tokenizer.json             # 完整分词器数据
└── vocab.json                 # 词汇表
```

> **注意**：`save_pretrained` 会自动生成 `added_tokens.json`、`chat_template.jinja` 等额外文件，这些是 Transformers 根据模型配置自动生成的默认文件。

---

## 第 7 章 Processor：多模态前处理

对于多模态模型（如 Qwen2.5-VL），输入不仅有文本，还有图像、视频。Processor 将多种输入统一处理为模型可接受的格式，本质是各种预处理器的组合：

> **Processor vs Tokenizer**：
> - Tokenizer 仅处理文本；
> - Processor = Tokenizer + ImageProcessor + VideoProcessor 的组合；
> - 对于纯文本模型，`AutoProcessor` 实际上等价于 `AutoTokenizer`。

```python
from transformers import AutoProcessor

model_path = "Qwen/Qwen2.5-VL-3B"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
```

**多模态输入格式**（以 Qwen2.5-VL 为例）：

```python
from qwen_vl_utils import process_vision_info

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image1.jpg"},
            {"type": "image", "image": "path/to/image2.jpg"},
            {"type": "text", "text": "请描述这两张图片的区别。"}
        ]
    }
]

# 应用对话模板
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 处理视觉信息
image_inputs, video_inputs, video_kwargs = process_vision_info(
    [messages], return_video_kwargs=True
)

# 统一处理为模型输入（返回 BatchFeature 类型）
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    fps=video_kwargs.get('fps'),
    padding=True,
    return_tensors="pt"
)
```

返回的 `BatchFeature` 包含模型所需的全部输入张量：`input_ids`、`attention_mask`、`pixel_values`（预处理后的图像像素）、`image_grid_thw`（部分模型的图像网格信息）等。

**注意事项**：

- 不同模型的 Processor 实现不同（resize、裁剪、归一化策略各异）；
- 微调/量化校准时必须使用与训练时一致的 Processor；
- Processor 同样支持 `save_pretrained("./saved_processor")` 独立保存。

---

## 第 8 章 模型推理与文本生成

> 配套脚本：[`07_inference_demo.py`](./07_inference_demo.py)

### 8.1 基本推理流程

推理五步曲：加载 → 编码 → 生成 → 截取 → 解码。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "path/to/model"

# 1. 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# 2. 编码输入
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 3. 生成（实际调用 GenerationMixin.generate 方法）
generated_ids = model.generate(inputs.input_ids, max_new_tokens=100)

# 4. 解码输出（去掉输入部分——generate 返回的序列包含输入）
output_ids = generated_ids[0][len(inputs.input_ids[0]):]
response = tokenizer.decode(output_ids, skip_special_tokens=True)
print(response)

# 也可以直接 batch_decode 完整序列（包含输入部分）
full_text = tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
```

两个易混参数：

| 参数 | 含义 |
|------|------|
| `max_new_tokens` | 限制**新生成**的 token 数（推荐使用，语义明确） |
| `max_length` | 限制**输入 + 输出的总长度**（如 `max_length=30` 表示总共最多 30 个 token） |

### 8.2 对话式推理（Chat Template + GenerationConfig）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_id = "path/to/chat/model"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="cuda:0",
    torch_dtype="auto"
)

# 构造对话消息
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

# 应用对话模板
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

# 配置生成参数
generation_config = GenerationConfig(
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

# 生成
generated_ids = model.generate(**model_inputs, generation_config=generation_config)
output_ids = [
    out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
print(response)
```

### 8.3 生成参数详解

`GenerationConfig` 控制生成行为的关键参数：

| 参数 | 作用 | 典型取值 |
|------|------|---------|
| `max_new_tokens` | 最大生成 token 数 | 按需 |
| `do_sample` | 是否采样（`False` = 贪心解码，确定性输出） | 推理常用 `True` |
| `temperature` | 温度：缩放 logits，越高分布越平（越随机） | 0.3 保守 / 1.0 标准 / 1.5 创意 |
| `top_p` | nucleus sampling：仅从累积概率达 p 的候选集采样 | 0.9 |
| `top_k` | 仅从概率最高的 k 个 token 中采样 | 50 |
| `repetition_penalty` | 重复惩罚 | 1.0~1.3 |
| `num_beams` | beam search 束宽 | 需要确定性高质量输出时使用 |

配套脚本的 `--demo params` 会用同一 prompt 对比不同参数组合（贪心 / temperature 0.3/1.0/1.5 / top-p / top-k）的输出差异，直观感受各参数的效果。

### 8.4 流式输出（Streaming）

使用 `TextStreamer` 实现逐 token 输出（ChatGPT 式打字机效果）：

```python
from transformers import TextStreamer

streamer = TextStreamer(tokenizer, skip_special_tokens=True)

model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    streamer=streamer,     # 关键：传入 streamer，每个 token 生成后立即打印
)
```

### 8.5 高层封装：pipeline

`pipeline` 是 Transformers 提供的最高层 API，将"加载模型 + 前处理 + 推理 + 后处理"封装为一行调用，适合快速验证任务效果：

```python
>>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

>>> # Sentiment analysis pipeline
>>> analyzer = pipeline("sentiment-analysis")

>>> # Question answering pipeline, specifying the checkpoint identifier
>>> oracle = pipeline(
...     "question-answering", model="distilbert/distilbert-base-cased-distilled-squad", tokenizer="google-bert/bert-base-cased"
... )

>>> # Named entity recognition pipeline, passing in a specific model and tokenizer
>>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
>>> recognizer = pipeline("ner", model=model, tokenizer=tokenizer)
```

pipeline 内部组合的仍是本文介绍的组件（AutoModel + AutoTokenizer/Processor + generate），只是隐藏了中间步骤。需要控制生成细节、KV Cache、批处理策略时，应使用第 8.1~8.4 节的显式流程。

---

## 第 9 章 张量维度变换

> 配套脚本：[`05_tensor_operations.py`](./05_tensor_operations.py)

实现注意力机制离不开高频的维度变换。本章是第 10~11 章的准备知识。

### 9.1 permute 与 transpose

**`permute()`**：重新排列所有维度：

```python
x = torch.randn(64, 8, 1024)     # (batch, heads, dim)
x_permuted = x.permute(1, 0, 2)  # -> (8, 64, 1024) = (heads, batch, dim)
```

**`transpose()`**：交换两个指定维度：

```python
x_transposed = x.transpose(0, 1)  # -> (8, 64, 1024)
```

> **关键注意点**：两者都不复制数据，返回原张量的**视图（view）**——`data_ptr()` 相同，仅 stride 改变。代价是张量在内存中不再连续，后续若要使用 `view()`，**必须先调用 `.contiguous()`**。

**stride 与连续性的原理**：

```
shape (3, 4, 5) 的连续张量，stride 为 (20, 5, 1)：
  沿第 0 维移动 1 步跳 20 个元素，沿第 1 维跳 5 个，沿第 2 维跳 1 个。

permute(2, 0, 1) 后 stride 变为 (1, 20, 5)：
  数据没动，但按新维度顺序读取时内存不连续 → view() 无法直接使用。

contiguous() 会按新布局复制数据（data_ptr 改变），使 stride 恢复标准排列。
```

### 9.2 view 与 reshape

| 操作 | 内存要求 | 失败行为 |
|------|----------|----------|
| `view()` | 要求内存连续 | 抛出 `RuntimeError` |
| `reshape()` | 自动处理 | 不连续时自动复制数据 |

```python
x = torch.randn(64, 8, 1024)

# permute 后内存不连续
x_p = x.permute(1, 0, 2)
# x_p.view(8, -1)             # 报错: RuntimeError
x_p.contiguous().view(8, -1)  # 正确
x_p.reshape(8, -1)            # 也正确（自动 contiguous）
```

> **最佳实践**：显式调用 `.contiguous()` 后使用 `view()`，比 `reshape()` 的隐式复制更可控——阅读代码时能一眼看出哪里发生了数据拷贝。

### 9.3 einops.rearrange

`einops` 用模式字符串直观表达变换，在 SGLang、vLLM 等大模型代码中被广泛使用：

```python
from einops import rearrange

# 交换维度：(seq, batch, dim) -> (batch, seq, dim)
q, k, v = [rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v)]

# 等价于 PyTorch 原生写法：
q, k, v = [x.permute(1, 0, 2).contiguous() for x in (q, k, v)]
```

**高级用法——拆分与重组维度**：

```python
# 拆分：(batch, seq, hidden) -> (batch, heads, seq, head_dim)
y = rearrange(x, "b s (h d) -> b h s d", h=8)
# 等价于: x.view(b, s, 8, d).permute(0, 2, 1, 3)

# 合并：(batch, heads, seq, head_dim) -> (batch, seq, hidden)
y = rearrange(x, "b h s d -> b s (h d)")
# 等价于: x.permute(0, 2, 1, 3).contiguous().view(b, s, -1)

# 分离展平的 batch*seq：(batch*seq, heads, dim) -> (batch, heads, seq, dim)
q, k, v = [rearrange(x, "(b s) h d -> b h s d", b=bsz) for x in [q, k, v]]

# 转置 key 用于注意力计算
k_transposed = rearrange(k, "b h s d -> b h d s")
```

### 9.4 大模型代码中的三个高频模式

**模式 1：Q/K/V 投影后拆分多头**

```python
# (batch, seq, hidden) -> view -> (batch, seq, heads, dim) -> transpose -> (batch, heads, seq, dim)
q = q_proj(x)
q = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
```

**模式 2：注意力输出合并多头**

```python
# (batch, heads, seq, dim) -> transpose -> contiguous -> view -> (batch, seq, hidden)
merged = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, hidden_size)
```

**模式 3：GQA（Grouped Query Attention）的 KV 头扩展**

```python
# KV 头数少于 Q 头数（如 2 vs 8），需扩展 KV 以匹配
num_groups = num_heads // num_kv_heads
k_expanded = k.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
k_expanded = k_expanded.reshape(batch, num_heads, seq_len, head_dim)
# (batch, kv_heads, seq, dim) -> (batch, heads, seq, dim)
```

---

## 第 10 章 注意力机制与掩码

> 配套脚本：[`04_attention_mask_demo.py`](./04_attention_mask_demo.py)（含 NumPy 手算逐步演示）

### 10.1 Attention 计算公式

Scaled Dot-Product Attention：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：

- $Q$（Query）：查询矩阵，形状 `[batch, heads, seq_len, d_k]`
- $K$（Key）：键矩阵，形状 `[batch, heads, seq_len, d_k]`
- $V$（Value）：值矩阵，形状 `[batch, heads, seq_len, d_v]`
- $d_k$：Key 的维度。除以 $\sqrt{d_k}$ 是为了防止点积值过大，导致 softmax 进入饱和区、梯度消失

四步计算：`scores = QK^T` → 缩放 → softmax → 与 `V` 加权求和。

### 10.2 Padding Mask（填充掩码）

**问题**：不同长度的序列组 batch 时需要填充（padding）到相同长度，但填充位置不应参与注意力计算。

```python
# 原始输入（padding 到长度 5）
input_sequence = [1, 2, 3, 0, 0]  # 0 是填充 token
padding_mask = [False, False, False, True, True]  # True 表示需要被屏蔽的位置
```

**效果**：被 mask 的位置在 softmax 后权重趋近于 0，不影响输出。这正是 Tokenizer 返回的 `attention_mask`（第 6.1 节）的用途。

### 10.3 Look-ahead Mask（因果掩码）

**问题**：自回归模型（如 GPT）生成第 $t$ 个 token 时不能看到位置 $t$ 之后的 token（防止信息泄露）。

```python
import torch
# 生成上三角掩码矩阵
seq_len = 5
look_ahead_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
```

结果：

```
tensor([[0., 1., 1., 1., 1.],   # token 0 只能看自己
        [0., 0., 1., 1., 1.],   # token 1 能看 0, 1
        [0., 0., 0., 1., 1.],   # token 2 能看 0, 1, 2
        [0., 0., 0., 0., 1.],   # token 3 能看 0, 1, 2, 3
        [0., 0., 0., 0., 0.]])  # token 4 能看所有
```

### 10.4 Mask 与 Q/K/V 的结合计算

Mask 在 $QK^T$ 之后、softmax 之前应用：

```
1. scores = Q @ K^T / sqrt(d_k)                # [batch, heads, seq_len, seq_len]
2. scores_masked = scores + mask * (-1e9)      # 被 mask 位置设为极小值
3. attention_weights = softmax(scores_masked)  # 极小值 -> 权重接近 0
4. output = attention_weights @ V
```

完整实现：

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores + mask * (-1e9)  # 将 mask 位置设为极小值

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

生产代码中推荐使用 PyTorch 原生 SDPA，它会自动选择最优后端（Flash Attention / Memory-Efficient / Math 回退）：

```python
# 显式传 mask
output = F.scaled_dot_product_attention(Q, K, V, attn_mask=causal_mask)

# 因果掩码场景更简洁的写法（无需手动构造 mask）
output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
```

> 配套脚本 [`04_attention_mask_demo.py`](./04_attention_mask_demo.py) 用 NumPy 逐步打印每一步的中间矩阵（分数矩阵 → 掩码后 → softmax 权重 → 输出），并验证手动实现与 SDPA 的结果一致性（最大差异 ~1e-7 量级）。

### 10.5 use_full_precision_softmax

FP16/BF16 下计算 softmax 可能损失精度，部分模型实现提供选项：先转 `float32` 计算 softmax，再转回原始精度：

```python
if self.use_full_precision_softmax:
    # 手动计算 attention（全精度 softmax）
    scale = self.head_size ** -0.5
    k_transposed = rearrange(k, "b h s d -> b h d s")
    attn_weights = torch.matmul(q, k_transposed) * scale
    attn_weights = attn_weights + attention_mask

    # 关键：在 float32 精度下计算 softmax，然后转回原始精度
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_weights = F.dropout(attn_weights, p=self.dropout, training=False)
    output = torch.matmul(attn_weights, v)
else:
    # 使用 PyTorch 原生 SDPA（自动选择最优后端）
    output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=self.dropout)
```

### 10.6 案例：Qwen2.5-VL 的 Mask 缓存优化

Qwen2.5-VL 的视觉编码器使用窗口注意力，通过 `cu_seqlens`（cumulative sequence lengths）而非直接传 Mask。为避免重复构造 Mask，实现中采用了缓存策略。

**早期版本——手动字典缓存**：

```python
class VisionSdpaAttention(nn.Module):
    _mask_cache = {}

    def generate_patch_attention_mask(self, s, bsz, device, cu_seqlens, ...):
        cache_key = (s, bsz, flatten_batch, tuple(cu_seqlens.cpu().tolist()))
        if cache_key in VisionSdpaAttention._mask_cache:
            return VisionSdpaAttention._mask_cache[cache_key].to(device=device)

        # ... 计算 mask ...
        VisionSdpaAttention._mask_cache[cache_key] = mask
        return mask
```

**新版本——`lru_cache` 装饰器**：

```python
from functools import lru_cache

class VisionSdpaAttention(nn.Module):
    @staticmethod
    @lru_cache(maxsize=128)
    def _generate_mask_cache(s, flatten_batch, cu_seqlens):
        # ... 计算 mask ...
        return mask
```

`lru_cache`（Least Recently Used Cache）自动缓存函数的输入输出，相同输入直接返回缓存结果。

**两种 Mask 模式**：

| 模式 | 输入处理 | 掩码形状 | 适用场景 |
|------|----------|----------|----------|
| `flatten_batch=True` | 所有序列展平为单个维度 | `(1, 1, s, s)` | 变长序列（如不同分辨率图像） |
| `flatten_batch=False` | 保持批次独立 | `(b, 1, s, s)` | 固定长度序列 |

---

## 第 11 章 多头注意力、KV Cache 与位置编码

### 11.1 Multi-Head Attention（MHA）

多头注意力将 Q、K、V 分成多个"头"（head），每个头独立计算注意力后拼接输出：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

其中 $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$。

参数维度关系：

- `num_heads (h)`：注意力头数
- `head_dim (d_k)`：每个头的维度
- `hidden_size = num_heads * head_dim`：总隐藏维度

拆头与合头的具体维度操作见第 9.4 节的模式 1 与模式 2；KV 头数少于 Q 头数的 GQA 变体见模式 3。

> **参考资料**：
> - The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
> - Transformer 详解: https://cuijiahua.com/blog/2021/01/dl-basics-3.html

### 11.2 KV Cache

自回归生成中，每个新 token 都要对所有历史 token 计算注意力。KV Cache 缓存已计算的 Key/Value，避免重复计算，将每步复杂度从 O(n²) 降为 O(n)：

- **Prefill 阶段**：一次性处理完整输入序列，计算并缓存所有 K、V；
- **Decode 阶段**：每步只计算新 token 的 Q，与缓存的 K、V 做注意力。

```python
from transformers import DynamicCache

# 创建 KV Cache
past_key_values = DynamicCache(config=model.config)

# Prefill（一次性处理所有输入 token）
outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)

# Decode（逐 token 生成，复用 KV Cache）
# outputs.past_key_values 包含了更新后的缓存
```

Prefill 后可查看缓存内容（配套脚本 `--demo kvcache`）：

```python
cache = outputs.past_key_values
print(f"缓存层数: {len(cache)}")
print(f"第 0 层 Key shape: {cache.key_cache[0].shape}")     # (batch, kv_heads, seq_len, head_dim)
print(f"第 0 层 Value shape: {cache.value_cache[0].shape}")
```

KV Cache 在 Transformers `forward` 中的传递机制见第 13.5 节。

> **参考资料**：KV Cache 详解 https://zhuanlan.zhihu.com/p/662498827

### 11.3 位置编码

Transformer 结构本身不包含位置信息，需要显式添加位置编码。常见方案：

| 方案 | 特点 | 代表模型 |
|------|------|----------|
| 正弦位置编码 | 固定编码，无需学习 | 原始 Transformer |
| 可学习位置编码 | 参数化，端到端学习 | BERT、GPT-2 |
| RoPE（旋转位置编码） | 相对位置编码，支持外推 | LLaMA、Qwen |
| ALiBi | 注意力线性偏置 | BLOOM |

> **参考资料**：位置编码详解 https://zhuanlan.zhihu.com/p/454482273

---

## 第 12 章 自定义模型结构与权重重建

> 配套脚本：[`07_inference_demo.py`](./07_inference_demo.py) `--demo rebuild`

修改模型结构（如缩减层数构建测试用小模型）并重新生成权重的完整流程。

### 12.1 理解 auto_map 配置

`config.json` 中的 `auto_map` 字段定义自定义类的映射关系：

```json
{
  "auto_map": {
    "AutoConfig": "configuration_minimax_text_01.MiniMaxText01Config",
    "AutoModelForCausalLM": "modeling_minimax_text_01.MiniMaxText01ForCausalLM"
  }
}
```

含义：使用 `trust_remote_code=True` 时，Transformers 从同目录的 `.py` 文件加载自定义的配置类和模型类（而非内置模型代码）。

### 12.2 必要文件清单

```
model_dir/
├── config.json                          # 模型配置（必须，可修改层数等参数）
├── configuration_minimax_text_01.py     # 自定义配置类（trust_remote_code 时需要）
├── modeling_minimax_text_01.py          # 自定义模型类（trust_remote_code 时需要）
├── tokenizer.json                       # 分词器
├── tokenizer_config.json
├── vocab.json
├── merges.txt
└── special_tokens_map.json
```

### 12.3 从修改后的 config 创建并保存模型

```python
import torch
from transformers import AutoModelForCausalLM, AutoConfig

model_id = "./modified_model_dir/"

# 加载修改后的配置
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

# 创建模型（随机初始化权重）
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# 转为 bf16（某些模型要求，如 FlashAttention 仅支持 fp16/bf16）
model = model.to(torch.bfloat16)

# 保存权重（会覆盖同名文件，但不会删除无关文件）
model.save_pretrained(model_id)
```

保存后新增的文件：

```
model_dir/
├── model-00001-of-00007.safetensors    # 分片权重文件
├── model-00002-of-00007.safetensors
├── ...
├── model.safetensors.index.json         # 分片索引文件
└── generation_config.json               # 自动生成的生成配置
```

> **注意**：`save_pretrained` 会自动复制 `configuration_xxx.py` 和 `modeling_xxx.py` 到保存目录，确保模型可独立加载。重新加载后即可推理验证（输出无意义——权重是随机的，但可用于验证结构与推理链路）。

---

## 第 13 章 Transformers 源码架构解析

理解 Transformers 内部架构，是调试问题与开发新模型的基础。本章按"入口路由 → Config 体系 → Model 体系 → 输出包装 → KV Cache → 量化 → 并行"的顺序，梳理从 `from_pretrained()` 到 `forward()` 的完整源码链路。

> **参考**：如何添加新模型 https://huggingface.co/docs/transformers/zh/custom_models
>
> **版本提示**：官方文档中流传较广的 transformers 架构总览图已经过时——例如 `generate()` 方法已从 `PreTrainedModel` 重构到 `GenerationMixin` 中。阅读源码时应以当前版本代码为准。

### 13.1 入口：Auto 类与完整加载链路

`Auto*` 类是按 `config.model_type` 路由到具体模型类的工厂。模型适配的入口在 `transformers/models/auto/modeling_auto.py`：

```python
# transformers/models/auto/modeling_auto.py
class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING     # model_type -> 模型类 的映射表

    # override to give better return typehint
    @classmethod
    def from_pretrained(
        cls: type["AutoModelForCausalLM"],
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        *model_args,
        **kwargs,
    ) -> "_BaseModelWithGenerate":
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

# 注意：auto_class_update 会动态更新 from_config 和 from_pretrained 方法
AutoModelForCausalLM = auto_class_update(AutoModelForCausalLM, head_doc="causal language modeling")
```

**完整加载链路**（决定"模型的类型"发生在加载 config 之后）：

```
AutoModelForCausalLM.from_pretrained(path)
    └─> _BaseAutoModelClass.from_pretrained()
            ├─> AutoConfig.from_pretrained(path)          # 加载 config；很多地方会对
            │       └─> PreTrainedConfig.from_pretrained() #   config 做判断，缺失则重新加载
            │           （此时从磁盘 config.json 初始化 quantization_config）
            ├─> 按 config.model_type 查 _model_mapping    # 路由到具体模型类
            └─> xxxForCausalLM.from_pretrained()          # 即 PreTrainedModel.from_pretrained()
                    └─> 权重加载 / 量化 / device_map 等核心逻辑（transformers/modeling_utils.py）
```

**两条初始化路径的重要差异**：

| 路径 | 实际调用 | 加载权重 | 量化逻辑 |
|------|---------|:---:|:---:|
| `AutoModel.from_config(config)` | `PreTrainedModel._from_config()` | 否（随机初始化） | **无**（不替换量化层） |
| `AutoModel.from_pretrained(path)` | `PreTrainedModel.from_pretrained()` | 是 | 有（见第 13.6 节） |

### 13.2 PreTrainedConfig 体系

```
PreTrainedConfig <---- xxxConfig（configuration_[model_name].py）
```

Config 类主要定义推理所需的属性变量：

- `model_type`：模型类型标识符（如 `"llama"`、`"qwen2"`、`"gpt_oss"`），Auto 类路由的依据；
- `base_model_tp_plan` / `base_model_pp_plan`：张量并行 / 流水线并行策略（见第 13.7 节）；
- 其他模型超参（如 GptOss 的 `default_theta = 150000.0`）在 `__init__()` 中初始化；
- 核心方法：`from_pretrained()` / `save_pretrained()`。

```python
config = AutoConfig.from_pretrained("model_path")
print(config)              # 调用的是 config.__repr__() 方法
dir(config)                # 查看 config 对象的所有属性
```

> **易踩的坑**：`print(config)` 看不到 `quantization_config` 的内容（`__repr__` 不显示它），必须通过 `config.quantization_config` 显式查看。

### 13.3 PreTrainedModel 体系

```
PreTrainedModel <---- xxxPreTrainedModel <---- xxxModel（modeling_[model_name].py）

class xxxForCausalLM(xxxPreTrainedModel, GenerationMixin)
```

`GenerationMixin` 提供 `generate()` 方法——这就是第 8 章 `model.generate()` 的来源（早期版本中该方法位于 `PreTrainedModel`，后被重构到 Mixin）。

**关键属性**：

- `_tp_plan` / `_pp_plan`：运行时的并行计划。其来源与赋值路径（现写在 `xxxForCausalLM` 类中）：
  1. 在模型类中直接声明该属性（替换基类中的 `None`）；
  2. 模型 `__init__()` 中显式调用 `self.post_init()`，将 config 中的 `base_model_tp_plan` / `base_model_pp_plan` 赋给这两个属性；
  3. 赋值经过 `tp_plan` 的 **setter 校验**（并行风格合法性 + 层模式匹配检查，代码见第 13.7 节）。
- `_keep_in_fp32_modules` / `_keep_in_fp32_modules_strict`：精度保持与跳过量化的标记（见第 13.6 节）。

**核心方法调用链**：

```
from_pretrained()
  -> __init__()
    -> post_init()          # 赋值 _tp_plan/_pp_plan、执行权重初始化收尾
      -> init_weights()
        -> _init_weights()
```

此外 `save_pretrained()` 与 config 的同名方法配套，负责权重与配置的落盘。

### 13.4 模型输出的包装：ModelOutput 体系

模型的输出基本都有包装。一个模型中主要有两类输出：

| 输出类 | 由谁返回 | 内容 |
|--------|---------|------|
| `xxxModelOutput[WithPast]` | `xxxModel`（骨干网络） | `last_hidden_state`（+ 缓存） |
| `xxxCausalLMOutput[WithPast]` | `xxxForCausalLM`（带 LM head） | `logits`、`loss`（+ 缓存） |

两者都继承自 `ModelOutput`（`transformers/utils/generic.py`），它同时支持属性访问（`outputs.logits`）、字典访问（`outputs["logits"]`）和元组解包。

**`WithPast` 后缀的含义**：区分是否携带/复用 KV Cache——带 `WithPast` 的输出包含 `past_key_values` 字段，下一步 decode 时回传即可复用缓存。

### 13.5 KV Cache 在 forward 中的使用

KV Cache 的多次 decode 复用，体现在 `xxxModel.forward()` 中——通过 `past_key_values` 参数衔接传递：

```python
class xxxModel(xxxPreTrainedModel):
    # xxxxx

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,      # KV Cache
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> xxxModelOutputWithPast:
        # input_ids 与 inputs_embeds 二选一（异或校验）
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # 如果启用缓存但未提供，创建新的 DynamicCache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # cache_position：本次输入在完整序列中的绝对位置
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # ... 各层 forward，更新 past_key_values ...

        return xxxModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,  # 返回更新后的缓存
        )
```

两点补充：

- 模型文件中的 `eager_attention_forward` 计算的仍是标准 full attention 逻辑，只是在计算前把 `past_key_values` 中缓存的 K/V 拼接进来——缓存改变的是"算多少"，不是"怎么算"；
- **与 SGLang/vLLM 的区别**：Transformers 的推理是批量处理模式——一开始确定所有请求、组成 batch 一起推理，不支持运行时动态插入新请求，因此无需请求拼接与复杂的 KV Cache 管理（如 PagedAttention）。

### 13.6 量化逻辑

**适配新模型时只写 `nn.Linear`。** 模型文件中的线性层统一用 `torch.nn.Linear`，不感知任何量化：

```python
class xxxMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```

实际加载量化模型时，这些 `nn.Linear` 会被**原地替换**为对应的量化 Linear。替换逻辑由量化器（基类 `HfQuantizer`）驱动，配套一个 `quantization_config` 配置。

**完整量化流程**（挂在 `PreTrainedModel.from_pretrained()` 内）：

```
1. AutoConfig.from_pretrained()
   -> 从磁盘 config.json 初始化 quantization_config

2. PreTrainedModel.from_pretrained()
   -> quantization_config = kwargs.pop("quantization_config", None)   # 也可从参数传入
   -> hf_quantizer, config, device_map = get_hf_quantizer(
          config, quantization_config, device_map, weights_only, user_agent
      )                                          # 获取对应的 HfQuantizer 实例

3. 模型初始化（meta 设备上创建骨架）
   with ContextManagers(model_init_context):
       # 初始化调用链：__init__() -> post_init() -> init_weights() -> _init_weights()
       model = cls(config, *model_args, **model_kwargs)

       if hf_quantizer is not None:   # 替换量化模块（不触碰权重）
           hf_quantizer.preprocess_model(
               model=model, dtype=dtype, device_map=device_map,
               checkpoint_files=checkpoint_files, use_kernels=use_kernels,
           )

4. 加载磁盘权重
   _load_pretrained_model()

5. 后处理
   hf_quantizer.postprocess_model()
   # 部分量化器子类重写了该方法做收尾处理，部分子类中是空实现
```

**量化器基类与 FP8-Block 实例**。`preprocess_model` 在 meta 设备阶段自由改写模型骨架，具体替换逻辑由子类的 `_process_model_before_weight_loading` 实现。以 DeepSeek 风格的 FP8 Block 量化（`[128, 128]` block size）为例，对应 `FineGrainedFP8HfQuantizer`（注册表在 `transformers/quantizers/auto.py`）：

```python
class HfQuantizer(ABC):
    def preprocess_model(self, model: "PreTrainedModel", dtype=None, **kwargs):
        """
        Setting model attributes and/or converting model before weights loading. At this point
        the model should be initialized on the meta device so you can freely manipulate the skeleton
        of the model in order to replace modules in-place.
        """
        model.is_quantized = True
        model.quantization_method = self.quantization_config.quant_method
        if self.pre_quantized:
            self._convert_model_for_quantization(model)
        self._process_model_before_weight_loading(model, **kwargs)


class FineGrainedFP8HfQuantizer(HfQuantizer):
    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        from ..integrations.finegrained_fp8 import replace_with_fp8_linear

        # 这里一定会返回一个包含 'lm_head' 的列表（输出层默认不量化）
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )

        model = replace_with_fp8_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )
```

**命名陷阱：跳过量化的层，字段名并不统一。** 初始化 `hf_quantizer` 需要的是 `modules_to_not_convert`，但不同量化配置中该信息的字段名各异：

| 来源 | 字段名 |
|------|--------|
| `HfQuantizer` 初始化参数 | `modules_to_not_convert` |
| `CompressedTensorsConfig` | `ignore` |
| DeepSeek FP8 Block（[128,128]）量化配置 | `ignored_layers` |

对接新量化格式时需要注意这些字段间的转换。

**`_keep_in_fp32_modules`：名义精度保持，实际是跳过量化的标记**：

```python
# PreTrainedModel 基类中的定义
_keep_in_fp32_modules = None
# the _keep_in_fp32_modules will avoid casting to anything other than float32, except bfloat16
# to also prevent bfloat16 casting, use the _keep_in_fp32_modules_strict flag
_keep_in_fp32_modules_strict = None

# 模型类中覆盖，通常是数值敏感的归一化层
_keep_in_fp32_modules = ["post_attention_layernorm", "input_layernorm", "norm"]
```

两点实测观察：

- 语义上 `_keep_in_fp32_modules` 阻止转换到 float32 **以外**的类型，但**例外允许 bfloat16**；若要连 bf16 也禁止，需使用 `_keep_in_fp32_modules_strict`；
- 因此实际检查 GPT-OSS 等模型时，被该属性标记的层可能仍是 BF16 而非 FP32——它更可靠的作用是**作为跳过量化的字典标记**（上面 `get_modules_to_not_convert` 的第三个参数正是它）。是否存在将其强制 update 到 float32 的代码路径，需结合具体版本进一步确认。

### 13.7 并行策略（TP/PP）

**策略组件注册表**。张量并行的各种策略组件统一注册在 `transformers/integrations/tensor_parallel.py` 的 `ParallelInterface` 中：

```python
class ParallelInterface(GeneralInterface):
    # Class instance object, so that a call to `register` can be reflected into all other files correctly,
    # even if a new instance is created (in order to locally override a given entry)
    _global_mapping = (
        {
            "colwise": ColwiseParallel(),                    # 列并行
            "rowwise": RowwiseParallel(),                    # 行并行
            "colwise_rep": ColwiseParallelReplicate(),
            "rowwise_rep": RowwiseParallelReplicate(),
            "local_colwise": LocalColwiseParallel(),
            "local_rowwise": LocalRowwiseParallel(),
            "local": IsolatedParallel(),
            "gather": GatherParallel(),
            "local_packed_rowwise": LocalPackedRowwiseParallel(),
            "sequence_parallel": SequenceParallel(),
            "replicate": ReplicateParallel(),
            "grouped_gemm": GroupedGemmParallel(),
            "ep_router": RouterParallel(),
        }
        if is_torch_greater_or_equal("2.5") and _torch_distributed_available
        else {}
    )
```

注意注册表的可用性条件：`torch >= 2.5` 且 `torch.distributed` 可用，否则为空表。

**模型级策略声明**。具体模型的并行策略写在 configuration 文件中。以 `transformers/models/gpt_oss/configuration_gpt_oss.py` 为例（注意 Attention 的经典组合：QKV 列并行 + O 投影行并行；MoE 专家使用 grouped_gemm + ep_router）：

```python
class GptOssConfig(PreTrainedConfig):
    model_type = "gpt_oss"
    default_theta = 150000.0

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.sinks": "local_rowwise",
        "layers.*.mlp.experts": "gather",
        "layers.*.mlp.router": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.gate_up_proj_bias": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj_bias": "grouped_gemm",
    }
```

**运行时校验：tp_plan 的 setter**。config 中的计划在 `post_init()` 时赋给模型的 `_tp_plan`，赋值经过 setter 做两层校验——并行风格必须在注册表中、层模式必须能匹配到真实参数：

```python
@tp_plan.setter
def tp_plan(self, plan: dict[str, str] | None):
    if plan is None:
        self._tp_plan = {}
        return
    if not isinstance(plan, dict):
        raise ValueError("Can only set a dictionary as `tp_plan`")

    # Ensure the styles are all valid
    for layer_pattern, parallel_style in plan.items():
        if parallel_style not in ALL_PARALLEL_STYLES:
            raise ValueError(
                f"Unsupported tensor parallel style '{parallel_style}' for layer '{layer_pattern}'. "
                f"Supported styles are {list(ALL_PARALLEL_STYLES.keys())}"
            )

    # Validate that the layer patterns match existing model structure. We check this by getting all
    # parameter names and seeing if any match the patterns
    model_param_names = [name for name, _ in self.named_parameters()]
    for layer_pattern in plan.keys():
        # Convert pattern to regex (replace * with .*)
        regex_pattern = layer_pattern.replace("*", r"\d+")
        pattern_matched = False
        for param_name in model_param_names:
            if re.match(regex_pattern, param_name):
                pattern_matched = True
                break
        if not pattern_matched:
            warnings.warn(
                f"Layer pattern '{layer_pattern}' does not match any parameters in the model. This rule may not "
                "be applied during tensor parallelization, or may lead to dimension mismatches"
            )

    # Set the plan
    self._tp_plan = plan
```

`_pp_plan` 的处理同理。模式中的 `*` 通配层号（转换为正则 `\d+`），未匹配到任何参数的模式会给出 warning——这是排查"并行策略未生效 / 维度不匹配"问题的第一线索。

---


## 第 14 章 常见错误与解决方案

### 14.1 NameError: `_flash_supports_window_size` is not defined

**原因**：`flash-attn` 未安装或版本不兼容。模型代码在导入时检查 `flash_attn` 是否可用：

```python
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    _flash_supports_window_size = "window_size" in list(
        inspect.signature(flash_attn_func).parameters
    )
```

**解决方案**：

```bash
pip install flash-attn --no-build-isolation
```

### 14.2 NotImplementedError: `flash_attn::_flash_attn_forward` on CPU

**原因**：Flash Attention 仅支持 CUDA 设备，不能在 CPU 上运行。

**解决方案**：

- 使用 GPU：`model = model.to("cuda")`；
- 使用 `device_map="auto"` 自动分配设备；
- 必须在 CPU 上运行时，改用不依赖 Flash Attention 的实现：`attn_implementation="eager"` 或 `"sdpa"`。

### 14.3 CUDA Out of Memory

**原因**：模型参数超过 GPU 显存容量（可先用第 4.4 节的方法估算所需显存）。

**解决方案**：

- 使用 `device_map="auto"` 分布到多 GPU/CPU（第 4.3 节）；
- 使用量化：`load_in_8bit=True` 或 `load_in_4bit=True`；
- 使用 `torch_dtype=torch.float16` 或 `torch.bfloat16`；
- 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解显存碎片。

### 14.4 ConnectTimeout / LocalEntryNotFoundError

**原因**：无法连接 HuggingFace Hub（网络问题）。

**解决方案**：

- 配置镜像：`export HF_ENDPOINT=https://hf-mirror.com`（第 2.1 节）；
- 使用离线模式：`TRANSFORMERS_OFFLINE=1`；
- 确保 `trust_remote_code=True` 时，自定义代码文件存在于本地目录。

### 14.5 RuntimeError: view size is not compatible

**原因**：对非连续内存的张量调用 `view()`（原理见第 9.1 节的 stride 解释）。

**解决方案**：

```python
# 方案 1：先 contiguous 再 view
x = x.permute(1, 0, 2).contiguous().view(...)

# 方案 2：使用 reshape（自动处理）
x = x.permute(1, 0, 2).reshape(...)
```

---

## 第 15 章 参考资料与配套脚本

### 15.1 配套脚本索引

| 脚本 | 内容 | 对应章节 |
|------|------|---------|
| [`01_download_model.py`](./01_download_model.py) | 模型/数据集下载的多种方式（full/single/dataset/config-only 四种模式） | 第 2 章 |
| [`02_load_and_inspect_model.py`](./02_load_and_inspect_model.py) | 四种加载方式、结构查看、参数统计 | 第 4 章 |
| [`03_tokenizer_and_processor.py`](./03_tokenizer_and_processor.py) | Tokenizer 编解码、批量 padding、特殊 token、对话模板、保存 | 第 6~7 章 |
| [`04_attention_mask_demo.py`](./04_attention_mask_demo.py) | 注意力与掩码的 NumPy 手算 + PyTorch/SDPA 实现对比 | 第 10 章 |
| [`05_tensor_operations.py`](./05_tensor_operations.py) | permute/transpose/view/reshape/einops 与大模型高频模式 | 第 9 章 |
| [`06_safetensors_inspector.py`](./06_safetensors_inspector.py) | safetensors 权重文件分析工具（过滤/统计/提取/索引解析） | 第 3 章 |
| [`07_inference_demo.py`](./07_inference_demo.py) | 推理全流程、生成参数对比、流式输出、KV Cache、模型重建 | 第 8、11、12 章 |

运行示例：

```bash
# 各脚本均支持 --help 查看完整用法
python 02_load_and_inspect_model.py --model bert-base-uncased --demo config
python 04_attention_mask_demo.py
python 07_inference_demo.py --model gpt2 --demo all
```

### 15.2 深入阅读

本目录的 [`tokenizer_and_chat_template.md`](./tokenizer_and_chat_template.md) 以 Qwen3-0.6B 为实例，深度解析：

- BPE / Byte-level BPE 分词算法的训练与推理过程；
- `Ġ`/`Ċ` 等 Byte-level BPE 显示约定；
- 特殊 Token 与词表结构；
- Chat Template 的 Jinja2 模板原理与 `add_generation_prompt`/`continue_final_message` 参数；
- 从用户输入到模型输出的推理全流程；
- 推理引擎中多请求拼接（continuous batching）的 token 结构。

### 15.3 参考资料

| 资源 | 链接 |
|------|------|
| HuggingFace Transformers 官方文档 | https://huggingface.co/docs/transformers |
| HuggingFace Hub 文档 | https://huggingface.co/docs/huggingface_hub |
| Accelerate 文档 | https://huggingface.co/docs/accelerate |
| safetensors 仓库 | https://github.com/huggingface/safetensors |
| The Illustrated Transformer | https://jalammar.github.io/illustrated-transformer/ |
| Transformer 详解 | https://cuijiahua.com/blog/2021/01/dl-basics-3.html |
| KV Cache 详解 | https://zhuanlan.zhihu.com/p/662498827 |
| 位置编码详解 | https://zhuanlan.zhihu.com/p/454482273 |
| HuggingFace Model 详解 | https://www.huaxiaozhuan.com/工具/huggingface_transformer/chapters/3_model.html |
| 自定义模型 | https://huggingface.co/docs/transformers/zh/custom_models |
| AFT (Attention Free Transformer) | https://arxiv.org/abs/2105.14103 |
| 手撕 Attention | https://hwcoder.top/Manual-Coding-1 |
