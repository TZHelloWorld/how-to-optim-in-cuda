# CUDA 编程指南：从入门到精通

> 一份面向新人小白的 CUDA 学习指南：从异构计算的基本概念出发，逐步深入编程模型、执行模型、内存模型，最终掌握全局内存与共享内存的性能优化技术。全书共 8 章，**一个目录对应一个章节**，每章包含教程正文（`README.md`）与可运行示例代码（`code/`），由浅到深、由简到繁编排，可顺序通读，也可按需查阅。

## 📂 目录结构

```text
00_cuda-programming-guide/
├── README.md                        ← 本文件：总目录 + 学习路线
├── images/                          ← 全书共享配图
├── 01_introduction/                 ← 第 1 章：CUDA 与异构计算入门
│   ├── README.md                    │    教程正文
│   └── code/                        │    hello.cu、vec_add.cu
├── 02_programming_model/            ← 第 2 章：编程模型（线程组织）
│   ├── README.md
│   └── code/                        │    check_index.cu、mat_add.cu、grid_stride.cu
├── 03_syntax_and_api/               ← 第 3 章：基本语法与常用 API
│   ├── README.md
│   └── code/                        │    device_query.cu、event_timing.cu
├── 04_execution_model/              ← 第 4 章：CUDA 执行模型
│   ├── README.md
│   └── code/                        │    reduce_divergence.cu
├── 05_memory_model/                 ← 第 5 章：CUDA 内存模型
│   ├── README.md                    │    5.1 层次总览 + 5.2 全局内存深入 + 5.3 共享内存深入
│   └── code/                        │    main.cu（带宽基准）、transpose_smem.cu、reduce_smem.cu
├── 06_streams_and_concurrency/      ← 第 6 章：流与并发
│   ├── README.md                    │    教程正文 + ⭐ 深度专题（专题 1~12）：Stream/同步/
│   │                                │       CUDA Graph + SGLang LLM 推理真实 trace 案例分析
│   └── code/                        │    深度专题配套 PyTorch 实验代码与 trace 文件
├── 07_atomics_and_warp/             ← 第 7 章：原子操作与 Warp 级原语
│   ├── README.md
│   └── code/                        │    histogram.cu、reduce_final.cu
└── 08_best_practices/               ← 第 8 章：性能优化最佳实践
    └── README.md
```

## 📚 章节目录

| 章节 | 内容概要 | 难度 |
|------|---------|------|
| [第 1 章 CUDA 与异构计算入门](01_introduction/README.md) | 并行计算基础、GPU vs CPU、异构计算与 Amdahl 定律、CUDA 平台全貌、可扩展编程模型、环境搭建、nvcc 与 PTX/cubin、Hello World、向量加法 | ⭐ 入门 |
| [第 2 章 编程模型：线程组织](02_programming_model/README.md) | Grid/Block/Thread 层次、线程索引计算、2D 矩阵示例、grid-stride loop、执行配置选择 | ⭐ 入门 |
| [第 3 章 基本语法与常用 API](03_syntax_and_api/README.md) | 函数/变量修饰符、内存管理 API、同步、错误检查、事件计时、统一内存、设备查询 | ⭐⭐ 基础 |
| [第 4 章 CUDA 执行模型](04_execution_model/README.md) | SM 架构、warp 与 SIMT、延迟隐藏、占用率、线程束分化、并行归约、循环展开 | ⭐⭐⭐ 进阶 |
| [第 5 章 CUDA 内存模型](05_memory_model/README.md) | 5.1 内存层次总览、锁页与零拷贝、L1/L2；5.2 合并访问法则、访问模式图解、AoS/SoA、矩阵转置、带宽基准 + 进阶：段/事务/缓存行；5.3 Bank 冲突、内存填充、共享内存转置与归约 | ⭐⭐⭐~⭐⭐⭐⭐ 进阶到深入 |
| [第 6 章 流与并发](06_streams_and_concurrency/README.md) | CUDA 流、异步拷贝、计算与传输重叠、事件同步、多流流水线；深度专题：同步机制/CUDA Graph/LLM 推理案例 | ⭐⭐⭐⭐ 深入 |
| [第 7 章 原子操作与 Warp 级原语](07_atomics_and_warp/README.md) | 原子函数、直方图实例、warp shuffle、warp 级归约、协作组简介 | ⭐⭐⭐⭐ 深入 |
| [第 8 章 性能优化最佳实践](08_best_practices/README.md) | APOD 方法论、四大优化清单、性能分析工具、常见错误与陷阱、综合实战与进阶路线 | ⭐⭐⭐⭐⭐ 精通 |

## ⭐ 深度专题（进阶阅读）

教程正文之外，两个附带**可运行验证代码**的深度专题已整合进对应章节的正文中，适合读完章节基础部分后深入研读：

| 专题 | 位置 | 内容 |
|------|------|------|
| [配套基准测试：实测验证访问模式的影响](05_memory_model/README.md#526-配套基准测试实测验证访问模式的影响) | 第 5 章 5.2.6 节 | 带宽基准测试程序（代码位于 `05_memory_model/code/main.cu`），用实测数据验证对齐/跨步/合并访问的性能差异 |
| [深度专题：CUDA Stream 与同步机制（专题 1~12）](06_streams_and_concurrency/README.md#深度专题cuda-stream-与同步机制) | 第 6 章后半部分 | 异步执行模型、三级同步原语、Default Stream 语义、PyTorch Stream 实现、隐式同步陷阱、CUDA Graph，以及 SGLang + Qwen3 真实 profiler trace 的 LLM 推理多流案例分析（PyTorch 实验代码位于 `06_streams_and_concurrency/code/`） |

## 🗺️ 学习路线建议

```text
第一阶段：会写能跑（1~2 天）
  第 1 章 → 第 2 章 → 第 3 章
  目标：独立写出向量加法、矩阵加法，会编译、计时、查错
  练习：跑通各章 code/ 目录中的示例，完成章末练习

第二阶段：理解硬件（3~5 天）
  第 4 章 → 第 5 章（5.1 内存层次总览）
  目标：理解 warp/SM/占用率/内存层次，能解释"为什么这么写更快"

第三阶段：性能优化（1~2 周）
  第 5 章（5.2 全局内存 → 5.3 共享内存，+ 5.2.6 带宽基准实测）
  目标：掌握合并访问与 bank 冲突，独立优化矩阵转置/归约到接近带宽上限

第四阶段：高级特性（按需）
  第 6 章 → 第 7 章 → 第 8 章（+ 第 6 章 Stream 深度专题）
  目标：多流重叠、原子操作、warp 原语，形成完整的优化方法论
```

> [!TIP]
> 学习 CUDA 最重要的一条原则：**什么时候我们沿着硬件设计的思路设计程序，就会得到好的结果；什么时候背离了硬件设计的思路，就得不到好结果。** 所以不要只背 API——每学一个特性，都要问一句"硬件上发生了什么"。

## 🛠️ 环境要求

- NVIDIA GPU（计算能力 5.0+，即 Maxwell 及以后架构）
- NVIDIA 驱动 + [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)（含 `nvcc`）
- 验证：`nvidia-smi` 和 `nvcc --version` 均能正常输出（详见[第 1 章](01_introduction/README.md)）

各章示例代码的编译方式统一为：

```bash
nvcc -O3 xxx.cu -o xxx && ./xxx
```

## 📖 主要参考资料

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [谭升的博客：CUDA 系列教程](https://face2ai.com/program-blog/)
- [知乎专栏：CUDA 编程](https://zhuanlan.zhihu.com/c_1556913741089976320)
- [CUDA 内存访问](https://zhuanlan.zhihu.com/p/632244210)
- [SM 详解与 Warp Scheduler，合理块和线程的数量对 GPU 利用率非常重要](https://zhuanlan.zhihu.com/p/670063380)
- 《Professional CUDA C Programming》（CUDA C 编程权威指南）
