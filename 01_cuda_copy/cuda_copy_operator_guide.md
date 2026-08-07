# CUDA 拷贝算子：Tensor 拷贝的底层实现与优化

本文系统讲解 GPU 上 **拷贝（copy）** 这件看似最简单的事：一个 tensor 拷贝到一块新地址，底层到底发生了什么？由哪个硬件执行？数据会不会流经 SM？拷贝怎么与计算融合？数据量很大或者拷贝很碎的时候又该怎么办？

拷贝值得单独一篇的原因是：**它是所有算子中"计算/访存比"的极端下界——0 FLOP、纯搬运**。GPU 程序的时间要么花在算、要么花在搬，而拷贝把"搬"这一半剥离出来单独呈现：读懂了拷贝，就读懂了所有 memory-bound 算子的性能模型；反过来，深度学习负载里大量隐藏的拷贝（`contiguous()`、dtype 转换、KV Cache 写入、H2D 数据加载）也正是常见的性能刺客。

全文按"由浅入深"组织：

- **原理**（第 1～2 章）：一次拷贝的软件路径与性能模型；GPU 里两类"搬运工"（Copy Engine 与 SM）；
- **拆解**（第 3～4 章）：逐条分析 H2D/D2H/D2D/跨卡/PyTorch `copy_` 的真实执行路径；再深入 SM 拷贝 kernel，回答"数据怎么经过 SM"，并用可运行的 benchmark 逐版优化到带宽上限；
- **进阶**（第 5～6 章）：kernel 内部的拷贝原语（`cp.async`、TMA）；拷贝与计算的融合、大批量与碎片化拷贝的实践。

---

## 目录

- [第 1 章 问题定义：一次 tensor 拷贝到底发生了什么](#第-1-章-问题定义一次-tensor-拷贝到底发生了什么)
- [第 2 章 预备知识：GPU 里的两类搬运硬件](#第-2-章-预备知识gpu-里的两类搬运硬件)
- [第 3 章 拷贝路径逐一拆解：从 cudaMemcpy 到 PyTorch copy_](#第-3-章-拷贝路径逐一拆解从-cudamemcpy-到-pytorch-copy_)
- [第 4 章 SM 拷贝 kernel：数据如何经过 SM](#第-4-章-sm-拷贝-kernel数据如何经过-sm)
- [第 5 章 kernel 内的拷贝原语：cp.async 与 TMA](#第-5-章-kernel-内的拷贝原语cpasync-与-tma)
- [第 6 章 拷贝融合与大数据量拷贝实践](#第-6-章-拷贝融合与大数据量拷贝实践)
- [第 7 章 总结与速查表](#第-7-章-总结与速查表)

---

## 第 1 章 问题定义：一次 tensor 拷贝到底发生了什么

### 1.1 从 `y = x.clone()` 说起

在 PyTorch 里写下一行再普通不过的代码：

```python
y = x.clone()          # x 是 GPU 上的 tensor，把它拷到一块新显存
```

这一行触发的完整链条是：

1. **分配目的地址**：缓存分配器（CachingAllocator）划出一块与 $x$ 等大的显存（通常命中缓存池，不真正调 `cudaMalloc`）；
2. **分发到 copy 实现**：`clone` → `copy_` → ATen 的 CUDA 拷贝实现（源码在 `aten/src/ATen/native/cuda/Copy.cu`）；
3. **二选一的执行路径**（第 3 章详细拆解决策树）：
   - 源和目的**同卡、同 dtype、内存布局完全一致** → 调 `cudaMemcpyAsync(DeviceToDevice)`，作为一次**内存操作**提交，**不占用 SM**；
   - 否则（非连续、转 dtype、broadcast……）→ 启动一个**逐元素拷贝 kernel**，由 **SM** 执行：数据从显存读进寄存器、再写回显存。

也就是说，"拷贝会不会经过 SM"这个问题**没有唯一答案——取决于走哪条路径**。把每条路径的硬件行为弄清楚，正是本文的主线。

### 1.2 拷贝的性能模型：纯带宽，0 计算

拷贝 $N$ 字节，无论谁来执行，都必须**从源地址读 $N$ 字节、向目的地址写 $N$ 字节**——总计 $2N$ 字节流过内存系统，而计算量是 0 FLOP：

$$
\mathrm{AI} = \frac{\text{FLOP}}{\text{Byte}} = \frac{0}{2N} = 0
\qquad\Longrightarrow\qquad
t_{\min} = \frac{2N}{BW}
$$

算术强度恰好为 0——拷贝位于 Roofline 图最左端，是**最纯粹的 memory-bound 算子**：性能上限完全由带宽 $BW$ 决定，与算力无关。评价一个拷贝实现好不好只有一个指标——**达成带宽**：

$$
BW_{\mathrm{effective}} = \frac{\text{读字节数} + \text{写字节数}}{\text{时间}} = \frac{2N}{t}
$$

（这也是 CUDA Best Practices Guide 中 effective bandwidth 的标准算法。）以 A100-80GB 为例，HBM 峰值约 2 TB/s，拷贝 1 GiB 数据的理论下限约 $2 \times 1\,\mathrm{GiB} / 2\,\mathrm{TB/s} \approx 1.07\,\mathrm{ms}$——第 4 章的 benchmark 将逐版逼近这个数字。

由此得出两条贯穿全文的推论：

1. **拷贝无法靠"优化计算"加速**，只能：跑满带宽（第 4 章）、换更快的通路（第 2、3 章）、或**干脆不拷/顺路捎带**（第 6 章的融合与消除）；
2. **任何逐元素计算都可以免费搭拷贝的车**——数据反正要进出一趟，路过时做几个乘加不增加任何时间（第 6 章将实测验证）。

### 1.3 两类搬运工：Copy Engine 与 SM

GPU 上有**两类完全不同的硬件**可以执行拷贝，这是理解全文的钥匙：

| | **Copy Engine（CE）** | **SM（流式多处理器）** |
|--|----------------------|----------------------|
| 本质 | 独立的 DMA 引擎（profiler 中的 async engine） | 执行 kernel 的计算单元 |
| 工作方式 | 给定源/目的地址与长度，硬件自主搬运 | 线程执行 load/store 指令，数据**经过寄存器** |
| 数据路径 | 内存 → 总线/L2 → 内存，**不经过任何 SM** | HBM → L2 → SM 寄存器 → L2 → HBM |
| 擅长 | 跨 PCIe/NVLink 的传输；与 kernel **并行**执行 | 灵活：任意布局变换、类型转换、随算随拷（融合） |
| 局限 | 只会线性/简单二维搬运，不会计算 | 占用 SM 资源（与计算 kernel 抢） |
| 触发方式 | `cudaMemcpy(Async)` 等内存操作 API | 任何含 load/store 的 kernel |

一句话预告后文结论：**规整的大块拷贝交给 CE（不占 SM、可与计算重叠）；带任何"加工"（布局、类型、计算）的拷贝由 SM kernel 完成（数据实打实地流经 SM）；而高性能 kernel 内部的分块装载，在新架构上还有专门的第三类硬件（Hopper 的 TMA，第 5 章）**。

（严格说，现代 GPU 系统里的"搬运工"远不止两类——2.6 节给出完整图谱；但 CE 与 SM 是卡内拷贝的两条基准路径，其余成员都可以拿这两极作参照来理解。）

---

## 第 2 章 预备知识：GPU 里的两类搬运硬件

### 2.1 内存层次与各级带宽

拷贝的"路"由内存层次决定。各级容量、带宽的数量级（以 A100/H100 世代为参照）：

| 层级 | 容量 | 带宽（数量级） | 备注 |
|------|------|---------------|------|
| 寄存器 | 每线程 ≤ 255 个 | 最高 | SM 拷贝 kernel 的数据中转站 |
| 共享内存 / L1 | ~100–228 KB / SM | ~20 TB/s（全卡聚合） | kernel 内 tile 拷贝的目的地 |
| L2 缓存 | 40–50 MB | 数 TB/s | 所有拷贝路径都要经过 |
| HBM 显存 | 40–80 GB | ~2–3.3 TB/s | 卡内拷贝的带宽上限 |
| NVLink | — | 600–900 GB/s（双向总和） | 跨卡拷贝 |
| PCIe 4.0/5.0 x16 | — | ~32 / ~64 GB/s（单向理论值） | CPU↔GPU 拷贝的瓶颈 |
| CPU 内存（DDR） | 数百 GB | ~100–400 GB/s | pageable 拷贝还要受它约束 |

两条立刻能读出的结论：

- **卡内拷贝（D2D）与 CPU↔GPU 拷贝（H2D/D2H）相差近两个数量级**（2 TB/s vs 32 GB/s）——所以"数据留在卡上"永远是第一优化；
- 同一块数据在层次间每多走一跳就多付一份带宽——第 3 章 pageable 内存拷贝慢的根源就在多走了一跳。

### 2.2 Copy Engine：独立的 DMA 引擎

**Copy Engine（拷贝引擎，也叫 async engine / DMA engine）** 是 GPU 上独立于 SM 的搬运硬件。用 `deviceQuery` 能看到它的数量：

```
Device 0: "NVIDIA A100-SXM4-80GB"
  ...
  Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
```

对应 `cudaDeviceProp.asyncEngineCount`。`deviceQuery` 来自官方 [cuda-samples](https://github.com/NVIDIA/cuda-samples) 仓库（`Samples/1_Utilities/deviceQuery`，CUDA ≤ 11.6 的工具包则自带于 `/usr/local/cuda/samples/`）；嫌构建麻烦的话，自己查询只要几行：

```cuda
// ce_query.cu ：nvcc ce_query.cu -o ce_query && ./ce_query
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("%s: asyncEngineCount = %d\n", p.name, p.asyncEngineCount);
    return 0;
}
```

它的意义（CUDA C++ Programming Guide, *Asynchronous Concurrent Execution* 一节）：

- **≥ 1 个 CE**：拷贝可以与 kernel 执行**并行**——SM 在算，CE 在搬，互不占用；
- **≥ 2 个 CE**：H2D 和 D2H 两个方向的拷贝还能**彼此并行**（各占一个引擎），流水线双向传输。

CE 的工作方式是经典 DMA：驱动把"源地址、目的地址、长度"写入引擎的命令队列，引擎自主完成搬运并在结束时通知。全程**没有任何指令在 SM 上执行、没有任何数据进入 SM**——这就是"拷贝不经过 SM"的那一半情形。

CE 的局限同样明显：它只做**地址到地址的原样搬运**（线性或简单跨步），不会转 dtype、不会改布局、更不会计算。所以它天生匹配的是"规整大块"的拷贝。

### 2.3 SM 的访存通路：拷贝 kernel 靠什么搬数据

SM 没有"拷贝指令"，它执行拷贝的方式就是最普通的**读 + 写**：

```
LDG  R0, [src_addr]    // load global：显存 → 寄存器
STG  [dst_addr], R0    // store global：寄存器 → 显存
```

一条 `LDG` 从发射到数据可用的完整旅程（第 4 章还会细讲）：

```
warp 调度器发射 LDG → LSU（load/store 单元）做地址合并
  → 32 个 lane 的地址归并成若干 32B sector 请求
  → 查 L1：命中则直接返回
  → 未命中：经片上互连发往对应 L2 分片 → 查 L2
  → 再未命中：HBM 内存控制器排队、DRAM 读出
  → 数据原路返回：L2 → L1 → 写入该线程的寄存器
```

关键点：**kernel 方式的拷贝，每一个字节都真实地流经"SM 的寄存器文件"**，并占用 warp 槽位、寄存器、LSU 发射带宽等 SM 资源。作为交换，它获得了 CE 没有的能力——数据在寄存器里的那一刻，**做什么都可以**（转类型、乘个系数、换个布局再写出），这是第 6 章融合的物理基础。

### 2.4 pinned 与 pageable：主机内存的两种身份

CPU↔GPU 拷贝还有一个决定性因素：主机内存是否**页锁定（pinned / page-locked）**。

- **pageable（默认 `malloc`/`new` 得到的）**：操作系统随时可能把这页换出/搬家，DMA 引擎不能直接对它工作。驱动的处理办法是：先由 **CPU** 把数据拷进驱动自带的一小块 pinned 中转缓冲，再让 CE 从中转缓冲 DMA 到显存——**两跳、占用 CPU、无法真正异步**；
- **pinned（`cudaMallocHost` / `torch.Tensor.pin_memory()` 得到的）**：物理页被锁定，CE 直接一跳 DMA，带宽可达 PCIe 上限，且 `cudaMemcpyAsync` 才能真正异步（这正是 NVIDIA 官方博客 [How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/) 的核心内容）。

实测差距通常在 2 倍上下（pageable 还引入 CPU 占用与不确定延迟）。代价是 pinned 内存分配慢、占物理内存，不宜滥用。

### 2.5 用 profiler 区分两类拷贝

Nsight Systems / PyTorch profiler 的时间线上，两类拷贝一眼可辨：

- **CE 执行的拷贝**：显示为独立的内存操作条目，如 `Memcpy DtoD (Device -> Device)`、`Memcpy HtoD (Pinned -> Device)`，位于 memory 行，不占 kernel 行；
- **SM 执行的拷贝**：显示为一个 kernel，名字里通常带 elementwise/copy 字样，如 `vectorized_elementwise_kernel<..., direct_copy_kernel_cuda...>`，位于 kernel 行。

第 3 章的 PyTorch 实验会让这两种条目同框出现。

### 2.6 补充：搬运工其实不止两类

本章把 CE 与 SM 立为两极，是因为它们是**卡内拷贝**的两条基准路径。放宽视野到整个 GPU 系统，官方文档里能数出来的"搬运工"是一张更大的图谱：

| 搬运工 | 位置 | 驱动方式 | 数据路径 | 典型用途 | 官方出处 |
|--------|------|---------|---------|---------|---------|
| SM：LDG/STG | SM 内 | 线程逐条指令 | HBM↔L2↔L1↔**寄存器** | 一切 kernel 访存；可顺路加工 | CUDA Programming Guide |
| SM：`cp.async` | SM 的 LSU | 线程指令（异步） | global→L2(→L1)→shared，**绕寄存器** | kernel 内 tile 装载流水 | Ampere 架构白皮书 |
| SM：DSMEM | SM 间网络 | 线程直接访问对端 shared | SM shared ↔ SM shared，**不下 HBM** | thread block cluster 内交换数据 | Hopper 架构白皮书 |
| TMA | 每个 SM 的专用单元 | 单线程 + 张量描述符 | global↔shared，硬件自主寻址 | 整 tile 异步搬运（第 5 章） | Hopper 白皮书 / Tuning Guide |
| Copy Engine | GPU 级 DMA | 驱动命令队列 | 内存↔内存，**不经 SM** | `cudaMemcpy` 系、与计算重叠 | CUDA Programming Guide |
| 页迁移引擎 | GPU 级 | 缺页自动触发 | 主机页↔显存页 | Unified Memory（`cudaMallocManaged`）按需迁移 | Pascal 架构白皮书 |
| 解压引擎（DE） | GPU 级 | 专用引擎 | 边搬运边解压 | 压缩数据加载 | Blackwell 架构介绍 |
| CPU | 主机 | `memcpy` | pageable→pinned 中转缓冲 | pageable 拷贝的第一跳（2.4 节） | Best Practices Guide |
| 网卡 DMA | GPU 之外 | NIC 自主 DMA | 网络↔显存，**GPU 全程旁观** | 多机通信零拷贝 | GPUDirect RDMA 文档 |
| 存储控制器 DMA | GPU 之外 | NVMe DMA（cuFile） | SSD↔显存，跳过 CPU 内存 | 数据集/权重直读 | GPUDirect Storage 文档 |
| NVSwitch 在网计算 | 交换网络 | 交换机多播/规约 | 数据在**网络内**被复制、聚合 | all-reduce/broadcast 加速 | NVLink SHARP（H100/NCCL NVLS） |

两条贯穿性的观察：

- **演进方向是一致的：把 SM 从搬运里解放出来。** 从占线程、占寄存器（LDG/STG），到绕过寄存器（`cp.async`），到 SM 内专职 DMA（TMA），再到完全不劳 GPU 计算部分的第三方 DMA（GPUDirect 家族）与在网计算（NVLink SHARP）——搬运工作被一步步移交给专用硬件，SM 的每个周期都留给计算；
- **分类心法只有两问**：这条路径**谁产生地址（谁驱动）**？数据**物理上流经哪里**？用这两问去套上表任何一行，就能自己推出它"经不经过 SM"、能不能与计算重叠、可不可以顺路加工——第 7 章的速查表正是这两问的批量答案。

本文主线聚焦卡内拷贝，后文继续以 CE 与 SM 两极展开；TMA 在第 5 章详述，其余成员知道其存在与定位即可。

---
## 第 3 章 拷贝路径逐一拆解：从 cudaMemcpy 到 PyTorch copy_

本章把常见拷贝按"路径"逐一过一遍：每种拷贝由谁执行、走什么硬件通路、经不经过 SM。

### 3.1 H2D / D2H：CPU 与 GPU 之间

```cpp
cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice);              // 同步版
cudaMemcpyAsync(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice, stream); // 异步版
```

- **执行硬件**：Copy Engine，经 PCIe（或 Grace Hopper 的 NVLink-C2C）搬运，**不经过 SM**；
- **pageable 源**：如 2.4 节所述，多一跳 CPU 中转，`cudaMemcpyAsync` 也会退化成"提交时就分块搬"的近同步行为；
- **pinned 源**：一跳 DMA 直达，真异步——提交后立即返回，CE 在后台搬，SM 可同时执行 kernel；
- **同步语义**：`cudaMemcpy` 阻塞 CPU 直到完成；`cudaMemcpyAsync` 只是提交，完成与否由 stream/event 同步。

这条路径的优化三板斧（第 6.5 节给完整代码）：**pinned 内存、分块流水（拷贝与计算重叠）、双向传输用两个 CE**。

### 3.2 D2D 同卡：一条 HBM 总线上的读与写

```cpp
cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
```

同一张卡内的拷贝，源和目的都在同一块 HBM 上——**读 $N$ 和写 $N$ 都走这条总线**，这是硬约束：不管由谁执行，理论上限都是 $BW/2$ 的"拷贝速率"（即 $2N/t = BW$）。

执行方式上，`cudaMemcpy(DtoD)` 由驱动作为**内存操作**提交（profiler 中是 `Memcpy DtoD` 条目，不出现在 kernel 时间线上、不占用你的 SM 计算窗口）；而你也完全可以**自己写 SM kernel** 做同样的事（第 4 章）。两者受同一条 HBM 带宽约束、量级相同，实测中驱动路径通常还略胜手写 kernel（4.3 节的 benchmark 给出具体数字）。选择的依据不是快慢，而是：

- 想**与计算重叠**、或拷贝就是"原样搬"→ 用 `cudaMemcpyAsync`（CE 路径）；
- 拷贝需要**顺路加工**（转 dtype、变布局、乘系数）→ 必须 SM kernel，而且加工是免费的（1.2 节推论 2）。

### 3.3 D2D 跨卡：P2P 与 NVLink

```cpp
cudaDeviceCanAccessPeer(&ok, dev0, dev1);         // 查询 P2P 能力
cudaDeviceEnablePeerAccess(dev1, 0);              // 开启后 dev0 可直接访问 dev1 显存
cudaMemcpyPeerAsync(dst1, dev1, src0, dev0, bytes, stream);
```

- 两卡间有 **NVLink** 或同一 PCIe switch 且开启 P2P：CE 直接卡到卡 DMA，不经过 CPU 内存；
- 无 P2P 能力：驱动退化为"D2H 到主机中转缓冲 + H2D"，两跳 PCIe，带宽腰斩以上；
- 开启 P2P 后，kernel 里甚至可以直接解引用对端指针（SM 的 load 经 NVLink 取数）——NCCL 的很多通信 kernel 正是这么工作的。

### 3.4 PyTorch `copy_` 的完整决策树

PyTorch 所有拷贝入口（`clone()`、`contiguous()`、`.to()`、`copy_()`）最终汇合到 `aten/src/ATen/native/cuda/Copy.cu`，决策逻辑可以画成一棵树：

```
x.copy_(src)  /  clone()  /  contiguous()  /  to(...)
│
├─ 源、目的在同一张卡？
│   ├─ 是：dtype 相同 且 两边内存布局逐字节对应（连续或步长完全一致）？
│   │   ├─ 是 → cudaMemcpyAsync(DtoD)            ← CE 路径，不占 SM
│   │   └─ 否 → TensorIterator 逐元素拷贝 kernel  ← SM 路径
│   │           （非连续 / 转 dtype / broadcast 全走这里）
│   └─ 否（跨卡）→ P2P 可用则 cudaMemcpyPeerAsync（CE 走 NVLink/PCIe），
│                   否则经主机内存中转（两跳）
│
└─ CPU ↔ GPU → cudaMemcpyAsync(H2D / D2H)         ← CE 路径
        ├─ 源为 pinned（pin_memory=True）且 non_blocking=True → 真异步
        └─ pageable → 驱动经中转缓冲两跳，non_blocking 名存实亡
```

三个高频实践推论：

- `x.t().contiguous()`、`x.to(torch.float16)`、`x[::2].clone()` 这类操作**都是 SM kernel 拷贝**——数据全部流经 SM 寄存器，还常常因为非合并访存而达不到带宽上限；
- DataLoader 的 `pin_memory=True` + `tensor.to('cuda', non_blocking=True)` 是一套组合拳：前者把主机侧换成 pinned，后者让 H2D 交给 CE 后台搬、与计算重叠（PyTorch 官方教程 *A guide on good usage of non_blocking and pin_memory* 专讲这件事）；
- **不确定时看 profiler**（2.5 节）：`Memcpy` 条目 = CE，elementwise kernel = SM。

用 PyTorch profiler 把三种路径同框验证（可直接运行）：

```python
import torch
from torch.profiler import profile, ProfilerActivity

x = torch.randn(4096, 4096, device='cuda')

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    a = x.clone()                    # 连续、同 dtype   → Memcpy DtoD（CE，不占 SM）
    b = x.t().contiguous()           # 非连续（转置）   → elementwise copy kernel（SM）
    c = x.to(torch.float16)          # dtype 转换       → 转型拷贝 kernel（SM）

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# 表中可看到：Memcpy DtoD (Device -> Device) 一行，
# 以及两条 vectorized_elementwise_kernel<...>（名字含 copy/cast 字样）
```

---

## 第 4 章 SM 拷贝 kernel：数据如何经过 SM

CE 路径没什么可优化的（给定总线，硬件全自动）；真正值得写、也值得优化的是 **SM 拷贝 kernel**——它是所有 memory-bound kernel 的原型。本章先把"数据经过 SM"的微观过程讲透，再用一个 benchmark 从朴素版逐步优化到带宽上限。

### 4.1 一个字节的旅程：LDG 与 STG 的微观过程

以一个 warp 执行 `dst[i] = src[i]`（fp32）为例：

**读（LDG）**：

1. **发射**：warp 调度器把 `LDG.E.SYS R0, [R2]` 发给 LSU；该 warp 进入等待（scoreboard 标记 R0 未就绪），调度器立刻切换其他 warp——这就是 GPU 用**并发掩盖延迟**的方式；
2. **地址合并**：LSU 检查 32 个 lane 的地址。若连续（lane $j$ 访问 `src[i+j]`），32 × 4 B = 128 B 恰好合并成 4 个 32 B sector 请求——**满效率**；若地址分散（如跨步访问），最坏退化成 32 个独立 sector——同样的数据要付 8 倍流量；
3. **逐级查找**：L1 未命中 → 经片上互连路由到该物理地址对应的 L2 分片 → L2 未命中 → HBM 控制器排队，DRAM 激活行、读出；
4. **数据返回**：HBM → L2（顺手驻留）→ L1 → **写入寄存器文件**，scoreboard 置就绪，warp 恢复可调度。

**写（STG）**：数据从寄存器发出，经 L1（global 写不驻留 L1）进入 **L2，以 write-back 策略聚合**后写往 HBM。写操作是"发后不理"（fire-and-forget），一般不阻塞 warp。

所以"经过 SM"的确切含义是：**每个字节都物理地流经 SM 的寄存器文件**，且整个搬运过程消耗 SM 的 warp 槽位、寄存器和 LSU 发射带宽。这与 CE 路径（数据只在内存系统内流动）形成鲜明对照。

### 4.2 打满带宽需要多少并发：Little's Law

HBM 延迟约 500 ns、带宽约 2 TB/s。要让总线不空转，任意时刻的**在途（in-flight）字节数**必须达到：

$$
\text{在途字节} = BW \times t_{\text{延迟}} \approx 2\,\mathrm{TB/s} \times 500\,\mathrm{ns} = 1\,\mathrm{MB}
$$

摊到 A100 的 108 个 SM，每个 SM 要随时挂起约 9 KB、即约 **72 个 128 B 事务**。一个 warp 的一条 `LDG.128`（16 B/lane）贡献 512 B 在途——每 SM 需要约 18 条这样的未决指令。凑够它靠两件事：

- **足够多的 warp**（occupancy）：一个 warp 等数据时别的 warp 接着发请求；
- **每线程更宽的访问**（向量化 `float4`）与**每线程多个独立请求**（ILP）：单个 warp 一次挂起更多字节。

这就是下面 benchmark 三个版本的优化逻辑。

### 4.3 benchmark：三版拷贝 kernel 对比 cudaMemcpy

完整可编译运行（`nvcc -O3 -arch=native copy_bench.cu -o copy_bench`）：

```cuda
// copy_bench.cu ：D2D 拷贝的四种做法与带宽实测
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(1); } } while (0)

// V0：一线程一元素（4 B），并发全靠海量线程
__global__ void copy_naive(const float* __restrict__ src, float* __restrict__ dst,
                           size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];                 // LDG 4B + STG 4B
}

// V1：grid-stride loop——固定网格反复扫，线程复用、启动参数与数据量解耦
__global__ void copy_gridstride(const float* __restrict__ src, float* __restrict__ dst,
                                size_t n) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        dst[i] = src[i];
}

// V2：float4 向量化——一条 LDG.128 搬 16 B，在途字节数 ×4（4.2 节的处方）
__global__ void copy_float4(const float4* __restrict__ src, float4* __restrict__ dst,
                            size_t n4) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride)
        dst[i] = src[i];                        // LDG.128 + STG.128
}

template <typename F>
float bench_ms(F launch, int iters = 20) {
    launch();                                   // 预热
    cudaEvent_t beg, end;
    CHECK(cudaEventCreate(&beg)); CHECK(cudaEventCreate(&end));
    CHECK(cudaEventRecord(beg));
    for (int i = 0; i < iters; i++) launch();
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));
    float ms; CHECK(cudaEventElapsedTime(&ms, beg, end));
    CHECK(cudaEventDestroy(beg)); CHECK(cudaEventDestroy(end));
    return ms / iters;
}

int main() {
    const size_t N = 1ull << 28;                // 2^28 个 float = 1 GiB
    const size_t bytes = N * sizeof(float);
    float *src, *dst;
    CHECK(cudaMalloc(&src, bytes));
    CHECK(cudaMalloc(&dst, bytes));
    CHECK(cudaMemset(src, 1, bytes));

    auto report = [&](const char* name, float ms) {   // 有效带宽 = (读 + 写) / 时间
        printf("%-14s %8.3f ms   %8.1f GB/s\n", name, ms, 2.0 * bytes / ms / 1e6);
    };

    report("cudaMemcpy", bench_ms([&] {
        CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice)); }));

    const int block = 256;
    report("V0 naive", bench_ms([&] {
        copy_naive<<<(int)((N + block - 1) / block), block>>>(src, dst, N); }));

    const int grid = 2048;                      // 铺满全部 SM 若干倍即可
    report("V1 gridstride", bench_ms([&] {
        copy_gridstride<<<grid, block>>>(src, dst, N); }));
    report("V2 float4", bench_ms([&] {
        copy_float4<<<grid, block>>>((const float4*)src, (float4*)dst, N / 4); }));

    CHECK(cudaFree(src)); CHECK(cudaFree(dst));
    return 0;
}
```

一次真实实测（HBM 峰值约 2 TB/s 级别的卡，拷贝 1 GiB）：

| 版本 | 关键改动 | 实测耗时 | 达成带宽 | 峰值占比 |
|------|---------|---------|---------|---------|
| cudaMemcpy（驱动/CE） | — | 1.219 ms | **1761 GB/s** | ~86% |
| V0 一线程一元素 | 海量 block 堆并发 | 1.423 ms | 1509 GB/s | ~74% |
| V1 grid-stride | 固定 2048 block，线程循环复用 | 1.708 ms | **1257 GB/s（反而最慢！）** | ~62% |
| V2 float4 向量化 | 在 V1 基础上在途字节 ×4 | 1.279 ms | 1679 GB/s | ~82% |

四点解读（这组数字比"预期排名"有教益得多）：

- **`cudaMemcpy` 是最快的**：驱动的拷贝路径经过多年调优，86% 峰值几乎就是读写混流（DRAM read/write turnaround 开销）下的实际上限。结论：**规整大块拷贝，直接用 `cudaMemcpy`，不必手写**——手写 SM kernel 的价值在第 6 章的"顺路加工"，不在纯拷贝本身；
- **V1 比 V0 慢是 Little's Law 的活例子**（4.2 节）：固定 2048 block × 256 线程后，靠每线程"标量 4 B load → 循环回跳 → 再 load"的节奏攒不够在途字节（循环控制拖慢了发射密度）；而 V0 的百万个"一次性"block 靠硬件调度器高速轮转，反而维持了更高的在途请求量。**grid-stride 本身不是加速手段**，它换来的是启动参数与数据量解耦的灵活性——用它时必须配足并发（更大的 grid，或下一条的向量化）；
- **V2 用向量化补上了 V1 挖的坑**：同样的 grid-stride 结构，`LDG.128` 让每条指令的在途字节 ×4，带宽从 1257 → 1679 GB/s，一举超过 V0（对应官方博客 [CUDA Pro Tip: Increase Performance with Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)）；
- 所有版本都到不了纸面 2 TB/s：拷贝是 50% 读 + 50% 写的混流，DRAM 方向翻转有固有开销，80%～90% 峰值即为此类负载的实际水位（STREAM/BabelStream 基准同此）。

（不同卡、不同驱动版本的具体数字会有出入，但上述**相对关系与成因**是稳定的；V1 的 `grid` 若调大数倍或让每线程展开多路独立访问，也能明显回升——不妨作为练习自行验证。）

### 4.4 非连续拷贝：SM 路径的真正难点

`x.t().contiguous()` 这类**布局变换拷贝**才是 SM 拷贝的硬骨头：转置访问中，读、写总有一侧地址不连续，naive 实现的合并效率跌到 1/8（4 B / 32 B sector），带宽掉一个数量级。标准解法是**经共享内存中转的分块转置**：按 tile 读入共享内存（读侧合并）、转置后再写出（写侧也合并），配合 padding 消 bank conflict。这一族技巧与 GEMM 的分块一脉相承，此处点到为止——要点是：**看到"非连续拷贝很慢"，第一反应应该是布局问题，而不是拷贝本身慢**。

---
## 第 5 章 kernel 内的拷贝原语：cp.async 与 TMA

前四章讨论的都是"拷贝作为一个独立算子"。但高性能 kernel **内部**同样充满拷贝——分块算法（GEMM、FlashAttention）每一步都要把 tile 从全局内存装进共享内存。这类"kernel 内拷贝"有自己专门的硬件演进路线，也是"数据怎么经过 SM"这个问题最精彩的部分。

### 5.1 传统路径：绕道寄存器

Ampere 之前，global → shared 只有一条路：

```cuda
// 两条指令、一次寄存器中转：
float tmp = g_ptr[i];    // LDG：global → 寄存器
s_buf[j] = tmp;          // STS：寄存器 → shared
```

问题有三：数据**绕道寄存器**（占用宝贵的寄存器预算）；两条指令都占发射带宽；`LDG` 是同步语义——想实现"边算当前块、边搬下一块"的流水，要靠软件多缓冲 + 精心排布指令，寄存器压力更大。

### 5.2 cp.async（Ampere）：绕过寄存器、异步搬运

Ampere 引入 `cp.async`（PTX 指令，CUDA C++ 侧对应 `cuda::memcpy_async` / `__pipeline_memcpy_async`）：**一条指令让数据从全局内存直达共享内存，不经过寄存器文件**，且为异步语义：

```cuda
#include <cuda/pipeline>

__global__ void tile_load(const float* __restrict__ g, float* __restrict__ out, int n) {
    __shared__ float s_buf[1024];
    auto pipe = cuda::make_pipeline();

    pipe.producer_acquire();
    // 每线程提交一段 16B 的异步拷贝：global → shared，绕过寄存器
    cuda::memcpy_async(&s_buf[threadIdx.x * 4],
                       &g[blockIdx.x * 4096 + threadIdx.x * 4],
                       sizeof(float4), pipe);
    pipe.producer_commit();

    /* ... 这里可以先干别的活（计算上一块），拷贝在后台进行 ... */

    pipe.consumer_wait();          // 需要数据时才等待
    __syncthreads();
    out[blockIdx.x * blockDim.x + threadIdx.x] = s_buf[threadIdx.x] * 2.0f;
}
```

收益（Ampere 架构白皮书与官方博客 [Controlling Data Movement to Boost Performance on the NVIDIA Ampere Architecture](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/)）：

- **省寄存器**：数据不再经寄存器中转（`cp.async.cg` 变体还可绕过 L1，只走 L2）；
- **省指令**：LDG+STS 两条变一条；
- **真异步**：提交后 warp 继续执行，配合 commit/wait 分组构成硬件支持的多级流水——FlashAttention、CUTLASS 的 Ampere kernel 全部建立在它之上。

注意：`cp.async` 仍由 SM 的 LSU 发起和追踪，数据走 L2 → shared，只是**不再进寄存器文件**——"经过 SM 但不经过寄存器"。

### 5.3 TMA（Hopper）：SM 里的专职 DMA

Hopper（H100）更进一步，给每个 SM 配了一个 **TMA（Tensor Memory Accelerator）**——可以理解为 SM 内部的小型 Copy Engine：

- **描述符驱动**：主机侧先用 `cuTensorMapEncodeTiled` 把张量的形状/步长/tile 尺寸/swizzle 方式编码成 TensorMap 描述符；kernel 里**单个线程**发出一条 `cp.async.bulk.tensor` 指令，TMA 硬件按描述符自主完成整个多维 tile（最多 5 维）的搬运，包括地址计算与越界处理；
- **彻底解放线程**：传统方式全 block 的线程忙于算地址、发 load；TMA 只需 1 个线程发一条指令，其余线程与寄存器全部留给计算，完成经 `mbarrier` 异步通知；
- **双向 + 多播**：支持 global↔shared 双向、跨 block 集群（thread block cluster）的 shared→shared 多播。

三代路径对比：

| | 数据路径 | 指令开销 | 地址计算 | 异步性 |
|--|---------|---------|---------|--------|
| LDG + STS | global → L2 → L1 → **寄存器** → shared | 2 条/每 16B/每线程 | 每线程自算 | 同步（软件流水） |
| cp.async（Ampere） | global → L2(→L1) → shared，**绕寄存器** | 1 条/每 16B/每线程 | 每线程自算 | 异步（commit/wait） |
| TMA（Hopper） | global → L2 → shared，**硬件自主** | **1 条/整个 tile/单线程** | TMA 硬件按描述符自算 | 异步（mbarrier） |

可以看到一条清晰的演进逻辑：**把"拷贝"从线程的工作里一步步剥离，还给专用硬件**——kernel 内的拷贝正在变得越来越像 2.2 节的 Copy Engine。FlashAttention-3、Hopper 版 CUTLASS 的性能跃升，很大程度上就来自 TMA + 异步流水把 SM 彻底腾给了 Tensor Core。

---

## 第 6 章 拷贝融合与大数据量拷贝实践

### 6.1 融合的收益模型：拷贝是"免费车道"

设一个逐元素算子链 `y = f(x)` 后面跟一次拷贝（或反过来）。分开执行的代价：

$$
t_{\text{分开}} = \underbrace{\frac{2N}{BW}}_{f\ \text{读写}} + \underbrace{\frac{2N}{BW}}_{\text{拷贝读写}} = \frac{4N}{BW}
\qquad\Longrightarrow\qquad
t_{\text{融合}} = \frac{2N}{BW}
$$

融合直接**省一半时间**——因为拷贝没有任何计算，它的全部成本就是那一读一写，而这一读一写完全可以搭在相邻算子身上。反方向同样成立（1.2 节推论 2）：拷贝 kernel 里顺路做逐元素计算是**零成本**的。实测验证——给第 4 章的 `copy_float4` 加上 scale + ReLU：

```cuda
// 拷贝 + 逐元素计算融合：与纯拷贝耗时几乎完全相同（ALU 全程在等访存）
__global__ void copy_scale_relu(const float4* __restrict__ src, float4* __restrict__ dst,
                                size_t n4, float alpha) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = src[i];                       // 数据既然已经进了寄存器……
        v.x = fmaxf(v.x * alpha, 0.f);           // ……顺手做任何逐元素计算都免费
        v.y = fmaxf(v.y * alpha, 0.f);
        v.z = fmaxf(v.z * alpha, 0.f);
        v.w = fmaxf(v.w * alpha, 0.f);
        dst[i] = v;
    }
}
```

把它加进 4.3 节的 benchmark，会看到它与 `V2 float4` 的带宽几乎一致——**计算藏进了访存的影子里**。

### 6.2 融合的三种形态

**形态一：拷贝吸收计算（上面的例子）。** dtype 转换 + 缩放 + 激活……只要是逐元素的，全能塞进一个拷贝 kernel。PyTorch 的 `x.to(torch.float16)` 本质就是"转型拷贝"单 kernel，而不是"先拷贝再转型"两个。

**形态二：计算吸收拷贝（消除拷贝）。** 更高级的融合是让拷贝**消失**：

- 算子直接把结果写进"本来要拷去"的目的缓冲（`out=` 参数、in-place 操作）；
- `torch.compile` 的内存规划：编译期发现 `y = f(x); z = y.clone()` 之类的模式，直接让 `f` 写两份或复用缓冲；
- 经典案例：残差连接 `y = x + f(x)` 中，融合 kernel 读 `x` 一次、写 `y` 一次，中间结果从不落地。

**形态三：许多小拷贝合并成一个 kernel（batching）。** kernel/memcpy 的启动开销约 3–10 μs，而 4 KB 数据在 2 TB/s 下只需 **4 ns**——碎片化拷贝的时间几乎全花在启动上，差三个数量级。解法是把"很多个源/目的指针"打包给**一个** kernel：

- **LLM 推理的 KV Cache 写入**是最典型的生产例子：每步 decode 要把新 token 的 K/V 写进 cache 的分页布局里——vLLM/SGLang 的 `reshape_and_cache` kernel 一次处理整个 batch 所有层的写入，融合了"拷贝 + 布局变换 + （可选的）量化"三件事；
- PyTorch 的 `torch._foreach_*` 系列与优化器的 `foreach=True`：把几百个参数张量的更新（本质是"读-算-写"的小拷贝）合并成少数几个 multi-tensor kernel；
- CUDA Graphs：拷贝数量实在多且模式固定时，把整串操作录制成图、一次提交，摊薄全部启动开销。

### 6.3 大数据量 H2D：分块流水，让 CE 与 SM 重叠

单卡 D2D 的大拷贝没什么可切的——一个拷贝就能吃满 HBM 带宽，切成多流并不会更快。真正需要"操作"的是**大批量 H2D/D2H**（数据加载、参数交换、offload）：PCIe 只有几十 GB/s，硬吞吐省不掉，但可以让它**藏在计算后面**。标准做法（官方博客 [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)）：pinned 内存 + 分块 + 多流：

```cuda
// 大批量 H2D 的分块流水：第 c+1 块在 CE 上传输时，第 c 块正在 SM 上计算
const int NCHUNK = 8;
size_t chunk = n / NCHUNK;                       // 每块元素数（假设整除）

float* h_buf;                                    // pinned 主机内存：CE 可直接 DMA
CHECK(cudaMallocHost(&h_buf, n * sizeof(float)));

cudaStream_t s[2];
CHECK(cudaStreamCreate(&s[0])); CHECK(cudaStreamCreate(&s[1]));

for (int c = 0; c < NCHUNK; c++) {
    cudaStream_t st = s[c & 1];                  // 两条流交替
    CHECK(cudaMemcpyAsync(d_buf + c * chunk, h_buf + c * chunk,
                          chunk * sizeof(float), cudaMemcpyHostToDevice, st));
    process<<<grid, block, 0, st>>>(d_buf + c * chunk, chunk);
}
// 同一条流内：拷贝 → 计算天然有序；
// 两条流之间：块 c 的计算（SM）与块 c+1 的拷贝（CE）并行——
// 这正是 Copy Engine 独立于 SM 存在的意义（2.2 节）
```

理想情况下总时间从 `T_拷贝 + T_计算` 降到 `max(T_拷贝, T_计算) + 一个块的启动延迟`。PyTorch 侧的等价形态就是 3.4 节的 `pin_memory + non_blocking` 组合加上预取（prefetch 下一个 batch）。

### 6.4 实践 checklist

| 情形 | 该做的事 |
|------|---------|
| 大块规整 D2D | 交给 `cudaMemcpyAsync` / `clone()`（CE），别自己写 kernel |
| 拷贝带任何加工 | 写（或让框架生成）融合 kernel，加工免费 |
| 反复出现的 `contiguous()` | 检查上游布局，能不转置就不转置；必须转置则确认走的是分块转置 kernel |
| H2D/D2H 慢 | pinned 内存 → 异步 → 分块流水与计算重叠 |
| 成百上千个小拷贝 | 合并成 multi-tensor kernel / CUDA Graphs |
| 拷贝本身可疑 | 先问"这份拷贝能不能不存在"（out=、in-place、内存规划），再谈优化 |

---

## 第 7 章 总结与速查表

### 7.1 核心问题回答

**"拷贝会经过 SM 吗？"** ——取决于路径：

| 拷贝方式 | 执行硬件 | 数据是否流经 SM | 备注 |
|---------|---------|----------------|------|
| `cudaMemcpy` H2D/D2H（pinned） | Copy Engine | **否** | 可与 kernel 并行 |
| H2D/D2H（pageable） | CPU + Copy Engine | **否**（但多一跳 CPU 中转） | 慢且伪异步 |
| `cudaMemcpy` D2D 同卡 | 驱动提交的内存操作 | **否**（不占 kernel 时间线） | 上限 = HBM 带宽 |
| `cudaMemcpyPeer` 跨卡（P2P） | Copy Engine 走 NVLink/PCIe | **否** | 无 P2P 则经主机两跳 |
| PyTorch 非连续 / 转 dtype 拷贝 | SM kernel | **是**：HBM→L2→**寄存器**→L2→HBM | 占用 SM 资源 |
| kernel 内 LDG+STS 装载 | SM | **是**（经寄存器） | 传统 tile 装载 |
| kernel 内 `cp.async`（Ampere） | SM 的 LSU | **是**（经 L2 直达 shared，**绕过寄存器**） | 异步流水 |
| kernel 内 TMA（Hopper） | SM 内专用 DMA 单元 | 数据直达 shared，**不占线程与寄存器** | 单线程发起整 tile |

**"怎么经过 SM？"** ——warp 发射 LDG → LSU 地址合并成 32B sector → L1 → L2 → HBM，数据原路返回**写入寄存器文件**；STG 反向经 L2 write-back 回 HBM。全过程占用 warp 槽位、寄存器与 LSU 带宽，靠海量在途请求（Little's Law：约 1 MB 在途）打满带宽（4.1/4.2 节）。

**"拷贝融合怎么做？"** ——拷贝 0 FLOP，融合的本质是让那一读一写被别人复用：拷贝吸收逐元素计算（免费）、计算直接写目的地址（拷贝消失）、碎片拷贝合并成一个 kernel（摊销启动开销）（6.1/6.2 节）。

**"数据量大怎么办？"** ——卡内：一个拷贝即可打满带宽，无需切分；跨 PCIe：pinned + 分块 + 多流，让 CE 的传输躲进 SM 计算的影子里（6.3 节）。

### 7.2 关键公式与数字速查

| 概念 | 公式 / 数值 | 出处 |
|------|------------|------|
| 拷贝时间下限 | $t_{\min} = 2N / BW$（读 + 写） | 1.2 节 |
| 有效带宽 | $BW_{\mathrm{eff}} = 2N / t$ | 1.2 节 |
| 算术强度 | $\mathrm{AI} = 0$，Roofline 最左端 | 1.2 节 |
| 打满带宽的在途字节 | $BW \times t_{\text{延迟}} \approx 2\,\mathrm{TB/s} \times 500\,\mathrm{ns} = 1\,\mathrm{MB}$ | 4.2 节 |
| 融合收益 | 逐元素算子 + 拷贝：$4N/BW \to 2N/BW$，省一半 | 6.1 节 |
| 小拷贝的启动开销比 | 4 KB 拷贝 4 ns vs 启动 ~5 μs，差 3 个数量级 | 6.2 节 |
| 带宽鸿沟 | HBM ~2 TB/s vs PCIe 4.0 ~32 GB/s，差约 60 倍 | 2.1 节 |

### 7.3 参考资料

- [CUDA C++ Programming Guide — Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)（Copy Engine、流与重叠的官方语义）
- [CUDA C++ Best Practices Guide — Data Transfer Between Host and Device](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)（pinned 内存、有效带宽的官方定义）
- [How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)（Mark Harris：pageable vs pinned）
- [How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)（分块流水重叠）
- [CUDA Pro Tip: Increase Performance with Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)（float4 向量化）
- [Controlling Data Movement to Boost Performance on the NVIDIA Ampere Architecture](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/)（cp.async）
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)（TMA）
- PyTorch 源码 `aten/src/ATen/native/cuda/Copy.cu`（copy_ 决策逻辑）；官方教程 [A guide on good usage of non_blocking and pin_memory](https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
