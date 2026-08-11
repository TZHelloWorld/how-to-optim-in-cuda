# 第 7 章 原子操作与 Warp 级原语

> [← 上一章：流与并发](../06_streams_and_concurrency/README.md) | [返回目录](../README.md) | [下一章：最佳实践 →](../08_best_practices/README.md)

本章目标：解决"多个线程要写同一个位置"的竞争问题（原子操作），并掌握 warp 内线程直接交换寄存器数据的高级原语（shuffle），完成归约优化的最后一块拼图。

先回顾一下走到这里的路：第 4 章我们认识了 warp 与 SIMT，用"消除分化"把归约优化了一轮；第 5 章把归约的中间迭代搬进了共享内存；第 6 章跳出内核，解决了主机与设备之间的并发问题。本章重新回到内核内部，补上线程协作的最后两块拼图——

- **线程往同一个位置写，怎么保证不出错？** 前几章的例子里，每个线程都写自己独享的位置（`C[i] = A[i] + B[i]`），井水不犯河水；可一旦多个线程要更新**同一个**变量（比如计数器、直方图的 bin），就会踩进"竞争条件"的坑。答案是**原子操作**。
- **warp 内的 32 个线程，能不能不经共享内存直接对话？** 第 5 章里线程交换数据要"写共享内存 → `__syncthreads()` → 读共享内存"三步走；而 warp 内的线程本来就步调一致，硬件提供了让它们**直接互读寄存器**的捷径。答案是 **warp shuffle**。

学完这两样，我们就能写出贯穿全书的"终极归约内核"，把第 2、4、5 章的技巧串成一条完整的优化链。

## 本章目录

- [7.1 竞争条件与原子操作](#71-竞争条件与原子操作)
- [7.2 常用原子函数](#72-常用原子函数)
- [7.3 实例：直方图统计](#73-实例直方图统计)
- [7.4 Warp 级原语：Shuffle 指令](#74-warp-级原语shuffle-指令)
- [7.5 综合实例：三层归约内核](#75-综合实例三层归约内核)
- [7.6 协作组（Cooperative Groups）简介](#76-协作组cooperative-groups简介)
- [7.7 本章小结](#77-本章小结)
- [7.8 动手练习](#78-动手练习)

---

## 7.1 竞争条件与原子操作

一切从一个看似无害的 `counter++` 说起——它在单线程的 CPU 代码里天经地义，放到 GPU 上却是一颗地雷。这一节先看地雷怎么炸，再看原子操作如何拆弹，最后算清拆弹的代价。

### 7.1.1 竞争条件从哪里来

CUDA 中成千上万个线程并发执行，当**多个线程读-改-写同一个内存位置**时就产生竞争条件（race condition）：

```c++
// 错误示例：多个线程同时执行 counter++ 会丢失更新
__global__ void bad_count(int *counter, const int *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && data[i] > 0) {
        (*counter)++;        // 读-改-写三步会交错，结果错误！
    }
}
```

问题出在哪？`(*counter)++` 这一行 C 代码在硬件上并不是一步完成的，而是**三步**：① 从内存**读**出当前值；② 在寄存器里**加一**；③ 把结果**写**回内存。单线程里这三步天然连贯；但成千上万个线程同时做这三步时，它们会**交错**。用一张时序表看两个线程如何"弄丢"一次更新（假设 `counter` 当前是 5）：

| 时刻 | 线程 A | 线程 B | 内存中的 counter |
|------|--------|--------|------------------|
| t0 | 读到 5 | — | 5 |
| t1 | — | 读到 5 | 5 |
| t2 | 算出 6 | 算出 6 | 5 |
| t3 | 写回 6 | — | 6 |
| t4 | — | 写回 6 | 6（正确答案应为 7） |

两个线程都"如实"完成了自己的三步，结果却少加了一次——线程 B 的写回**覆盖**了线程 A 的成果。打个比方：两个人同时查同一个银行账户，都看到余额 100 元，各自存入 1 元后都把"101"写了回去——账户凭空少了 1 元。参与的线程越多、交错越密集，丢失的更新就越多，而且**每次运行丢多少都不一样**——这类 bug 不但结果错，还难以复现，是并行编程中最阴险的一类。

> [!WARNING]
> 竞争条件的可怕之处在于它**可能偶尔算对**：线程恰好没有交错时结果就是正确的。"跑了一次结果对"绝不代表代码没有竞争——凡是多个线程写同一个位置，就必须显式处理。

### 7.1.2 原子操作：不可分割的读-改-写

**原子操作（atomic operation）**把"读-改-写"合并为一个**不可分割**的硬件操作，保证正确性：

```c++
if (i < n && data[i] > 0) {
    atomicAdd(counter, 1);   // 正确：硬件保证互斥
}
```

"原子"（atomic）一词取自希腊语"不可再分"的本义：对其他线程而言，`atomicAdd` 的读、改、写三步**要么全部发生、要么全部没发生**，绝不会有第二个线程能插在中间读到"半成品"。继续用银行的比方：原子操作相当于把"查余额、加钱、写回"打包成柜台的**一笔业务**——柜员办理期间锁定账户，下一个人只能等这笔业务完整结束后再办，"读到旧余额再互相覆盖"的事故就不可能发生了。

原子函数可以作用于**全局内存或共享内存**中的地址。硬件层面，全局内存的原子操作由 L2 缓存中的专用原子单元执行，共享内存的原子操作则在 SM 片上完成——这个区别在 7.3 节的直方图优化中将成为关键。

### 7.1.3 原子操作的代价：冲突时串行化

原子操作不是免费的午餐。柜台业务"一次只办一个人"保证了正确性，但也意味着——**当很多线程对同一个地址做原子操作（冲突，contention）时，这些操作会被硬件串行化**，一个接一个地执行。想象 GPU 里几十万个线程排在同一个柜台前，并行度瞬间归零，性能可想而知。

> [!NOTE]
> 性能特征：原子操作在**冲突（同地址竞争）激烈时**会串行化，成为瓶颈；冲突稀疏时开销可接受（现代 GPU 的 L2 原子单元相当快）。优化思路是**分层聚合**：先在块内/共享内存聚合，最后再对全局内存做一次原子操作。

"分层聚合"翻译成生活语言：与其让全厂几万名工人都挤到总部的一个柜台交报表，不如**每个车间先在自己的小黑板上汇总，最后各车间只派一个人去总部交一张汇总单**。7.3 节的直方图和 7.5 节的归约，用的都是这一招。

竞争问题解决了、代价也算清了，接下来把工具箱打开——除了 `atomicAdd`，CUDA 还提供了哪些原子函数？

## 7.2 常用原子函数

这一节过一遍原子函数家族：先看一览表，再讲它们共同的"返回旧值"约定，最后请出其中的"万能钥匙"——`atomicCAS`。

### 7.2.1 原子函数一览

以下函数均作用于全局内存或共享内存地址：

| 函数 | 作用 | 说明 |
|------|------|------|
| `atomicAdd(addr, v)` | `*addr += v` | 支持 int/uint/ull/float（CC 6.0+ 支持 double） |
| `atomicSub(addr, v)` | `*addr -= v` | 整数 |
| `atomicMax / atomicMin` | 取最大/最小 | 整数（浮点可用 CAS 模拟） |
| `atomicExch(addr, v)` | 交换值，返回旧值 | 常用于简单锁 |
| `atomicAnd / atomicOr / atomicXor` | 位运算 | 整数 |
| `atomicCAS(addr, cmp, v)` | Compare-And-Swap | 万能原语，可实现任意原子逻辑 |

几点补充说明：

- 表中"整数"指 `int`、`unsigned int`、`unsigned long long` 等整型；`atomicAdd` 在较新的计算能力上还支持 `__half`、`__nv_bfloat16` 等半精度类型，具体型号与类型的对应关系以官方 [CUDA C++ Programming Guide 的 Atomic Functions 一节](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)为准；
- CC 6.0+ 还提供了带作用域后缀的变体：`atomicAdd_block`（只保证对**同块**线程原子）和 `atomicAdd_system`（对**整个系统**原子，包括 CPU 和其他 GPU）——不带后缀的默认版本作用域是**单个 GPU 内的全部线程**，日常使用默认版本即可。

> [!WARNING]
> 原子函数只保证"这一个操作"不可分割，**不充当内存栅栏，也不隐含任何同步或顺序约束**——它管不了这条语句前后的其他内存读写顺序。另外，多个线程的原子操作**执行顺序是不确定的**：对整数加法无所谓（结合律严格成立），但对浮点数，不同的求和顺序会产生不同的舍入误差，因此**浮点原子累加的结果可能每次运行有微小差异**——深度学习框架中"同样的代码两次结果不完全一致"常源于此。

### 7.2.2 共同约定：返回操作前的旧值

所有原子函数都返回**操作前的旧值**。这不是可有可无的设计，而是很多算法的关键素材——"旧值"告诉你"在你之前世界是什么样的"。一个典型用法是**原子分配任务编号**：

```c++
// 每个线程领到一个全局唯一的递增编号：旧值就是"我排到的号"
int my_slot = atomicAdd(queue_tail, 1);
output[my_slot] = my_result;   // 各线程写入互不重叠的位置
```

好比取号机：`atomicAdd` 把号码牌拨到下一个数，而**吐给你的是拨动之前的那张号**——每个人拿到的号互不相同，凭号入座就不会互相覆盖。这个"原子取号"模式是并行压缩（stream compaction）、并行队列等算法的基石。

### 7.2.3 atomicCAS：万能原语

表格最后一行的 `atomicCAS`（Compare-And-Swap，比较并交换）值得单独一讲，因为它是整个家族的"万能钥匙"。它的语义是一步原子地完成："**如果 `*addr` 当前等于 `cmp`，就把它换成 `v`；无论换没换成，都返回 `*addr` 的旧值**"：

```c++
old = atomicCAS(addr, cmp, v);
// 等价于原子地执行：old = *addr; if (old == cmp) *addr = v;
```

为什么说它万能？因为任何"读-改-写"逻辑都可以套用一个**"乐观重试"循环**来实现：先读出旧值，在本地算出新值，然后用 CAS 尝试提交——"如果内存里还是我刚才读到的那个旧值（没人动过），就换成我的新值"；若提交失败（说明有别的线程抢先改了），就拿着最新值重来一遍。官方文档给出的经典例子，是在不支持硬件 double 原子加的老设备（CC < 6.0）上用 CAS 实现 `atomicAdd(double*, double)`：

```c++
__device__ double atomicAdd(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;   // 记住"我看到的旧值"
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
        // 若返回值 == assumed，说明提交成功，循环结束；
        // 否则 old 已更新为最新值，带着它重试
    } while (assumed != old);   // 用整数比较，避免 NaN != NaN 导致死循环
    return __longlong_as_double(old);
}
```

三个细节：① CAS 只支持整数类型，所以浮点数要先用 `__double_as_longlong` 按位重解释成整数参与比较；② 循环条件用**整数比较**而不是浮点比较，是为了绕开"NaN 不等于自身"这个浮点陷阱；③ 冲突激烈时重试会变多，性能不如硬件原生原子指令——所以 CAS 是"兜底方案"而非首选。同样的套路可以实现浮点版 `atomicMax`、原子乘法等任何标准库没有提供的原子逻辑（见 7.8 节练习 6）。

工具备齐了，接下来在一个经典问题上实战：直方图统计——它既是原子操作最典型的应用，也是"分层聚合"套路最好的教材。

## 7.3 实例：直方图统计

这一节把 7.1 的正确性和 7.1.3 的性能考量合到一个完整例子里：先写一个"能跑但慢"的朴素版本，再用共享内存两级聚合提速，代码来自本章配套的 [`code/histogram.cu`](code/histogram.cu)。

### 7.3.1 为什么直方图绕不开原子操作

直方图统计：给定一大堆字节数据，统计每个取值（0~255）各出现了多少次——图像处理里统计亮度分布、数据分析里统计频次，都是这个模型。它是原子操作的经典应用，因为**哪个线程会命中哪个 bin 完全由数据决定**，任意多个线程都可能同时对同一个 bin 加一——这正是 7.1.1 节的竞争条件现场，没有原子操作必然算错。

### 7.3.2 版本 1：朴素全局原子

最直接的写法：一个线程处理一个元素，直接对全局内存里的直方图做原子加：

```c++
#define NUM_BINS 256

// 版本 1：直接对全局内存做原子加（同 bin 冲突激烈时慢）
__global__ void histNaive(const unsigned char *data, int n, unsigned int *hist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&hist[data[i]], 1);
}
```

结果正确，但性能被数据分布攥在手里：如果数据均匀分布在 256 个 bin 上，冲突还算分散；可一旦分布**偏斜**——比如一张大部分是夜空的照片，几乎所有像素都落在最暗的几个 bin 里——几百万个线程就会挤在少数几个地址上排队，原子操作全面串行化。这就是 7.1.3 节说的"全厂工人挤一个柜台"。

### 7.3.3 版本 2：共享内存两级聚合

解法正是"分层聚合"：**每个线程块先在共享内存里维护一份自己的"私有直方图"，统计完后再一次性合并进全局直方图**：

```c++
// 版本 2：共享内存私有直方图 → 块内聚合 → 一次性合并到全局
__global__ void histSmem(const unsigned char *data, int n, unsigned int *hist) {
    __shared__ unsigned int smem[NUM_BINS];

    // 1. 清零共享内存直方图
    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) smem[b] = 0;
    __syncthreads();

    // 2. grid-stride loop 统计到"块私有"直方图
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&smem[data[i]], 1);
    __syncthreads();

    // 3. 每块只对全局内存做 NUM_BINS 次原子加
    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        atomicAdd(&hist[b], smem[b]);
}
```

三个步骤逐一拆解：

1. **清零私有直方图**。`__shared__` 数组的初始内容是未定义的，必须先清零。这里用了一个小技巧：`for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)` 让块内线程**分摊**清零工作（第 2 章周期划分思想的块内版本）——块有 256 个线程、直方图恰有 256 个 bin 时每人清一格；线程数少于 bin 数时每人多清几格，代码对任何块大小都正确。清完必须 `__syncthreads()`，确保没人在直方图还没清干净时就开始统计；
2. **统计到块私有直方图**。外层是标准的 [grid-stride loop](../02_programming_model/README.md#25-grid-stride-loop让固定线程数处理任意规模数据)（第 2 章），让固定数量的块吃下任意规模的数据；内层的 `atomicAdd(&smem[...], 1)` 作用在**共享内存**上——冲突的范围从"全 GPU 几十万线程"缩小到"块内几百个线程"，而且共享内存原子操作在 SM 片上完成、由硬件直接支持，比全局原子快得多。统计完再来一次 `__syncthreads()`，确保块内所有线程的账都记完了才能交汇总单；
3. **合并到全局直方图**。每个块只需对全局内存做 `NUM_BINS` 次原子加（还是用"分摊"技巧），把私有直方图整体加进总账。

版本 2 把全局原子操作的次数从 `n` 次降到 `gridDim.x × NUM_BINS` 次——以配套代码的参数算：64M 次降到约 `32 × SM 数 × 256` 次，缩减了三个数量级以上。这种**"先局部聚合、再全局提交"**的分层模式是原子操作优化的通用套路。

### 7.3.4 完整程序与实测

配套代码 [`code/histogram.cu`](code/histogram.cu) 用 CUDA 事件（第 6 章）给两个版本计时。主机端有两处值得注意：

```c++
// 故意让数据分布偏斜（低 bin 冲突激烈），拉开两版本差距
for (int i = 0; i < n; i++) h_data[i] = (unsigned char)(rand() % 16);

// 版本 1：一个线程一个元素，块数按数据量取
int threads = 256, blocks = (n + threads - 1) / threads;
histNaive<<<blocks, threads>>>(d_data, n, d_hist);

// 版本 2：grid-stride loop，块数只按 SM 数取（每 SM 32 个块）
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
histSmem<<<32 * numSMs, threads>>>(d_data, n, d_hist);
```

- **数据故意偏斜**：`rand() % 16` 让 64M 个元素全部挤进前 16 个 bin，把版本 1 的冲突推向最坏情况，两版本的差距因此格外醒目（改成均匀分布会怎样？见 7.8 节练习 1）；
- **两种启动配置**：版本 1"一个线程一个元素"，块数随数据量走；版本 2 用 grid-stride loop，块数只取 `32 × SM 数`——块数固定的好处是"每块合并一次"的全局原子总量可控，块开得越多、合并的开销反而越大；
- 程序最后把 256 个 bin 求和与 `n` 比对（每个元素恰好落入一个 bin，总数必须等于 `n`），输出 `PASS/FAIL` 验证正确性——性能优化的前提永远是结果正确。

原子操作解决的是"多个线程**写同一处**"的竞争；接下来换一个角度——warp 内的线程之间想**互相读数据**，有没有比共享内存更快的通道？

## 7.4 Warp 级原语：Shuffle 指令

第 4 章讲过，warp 内的 32 个线程以 SIMT 方式同步推进，像 32 个同排就座、齐步做题的同学。既然本来就肩并肩，传个数据何必"写到教室后面的黑板（共享内存）、等老师喊停（`__syncthreads()`）、再走过去抄回来"？这一节介绍的 shuffle 指令，就是让同桌之间**直接递纸条**。

### 7.4.1 Shuffle 家族

**Shuffle（洗牌）指令**允许同一 warp 内的线程**直接读取彼此的寄存器**，不经过共享内存、无需块级同步，延迟比共享内存更低：

```c++
// 主要原语（mask 通常取 0xffffffff，表示全 warp 参与）
__shfl_sync(mask, var, srcLane);        // 读取指定 lane 的 var（广播）
__shfl_up_sync(mask, var, delta);       // 读取 lane_id - delta 的 var
__shfl_down_sync(mask, var, delta);     // 读取 lane_id + delta 的 var
__shfl_xor_sync(mask, var, laneMask);   // 读取 lane_id ^ laneMask 的 var（蝶形交换）
```

四个变体的区别只在"读谁"（`lane_id` 指线程在 warp 内的编号，0~31，即 `threadIdx.x % warpSize`）：

| 原语 | 我读谁的 `var` | 典型用途 |
|------|----------------|----------|
| `__shfl_sync` | 指定的 `srcLane`（全 warp 读同一人 = 广播） | 把 lane 0 算出的值发给全 warp |
| `__shfl_up_sync` | 比我编号小 `delta` 的 lane | 前缀和（扫描） |
| `__shfl_down_sync` | 比我编号大 `delta` 的 lane | 归约（见 7.4.3） |
| `__shfl_xor_sync` | 编号与我异或 `laneMask` 的 lane | 蝶形交换、全员归约 |

理解它的关键是换一个视角：**shuffle 是"我去读别人的寄存器"，而不是"别人发给我"**——每个线程各自执行同一条 shuffle 指令，指令算出一个源 lane，然后取回源 lane 手里那份 `var` 的值。若 `__shfl_up_sync`/`__shfl_down_sync` 算出的源 lane 越界（比如 lane 30 想读 lane 30+16=46），该线程**拿回的就是自己原来的 `var`**，不会读到垃圾值——这个细节在 7.4.3 节的归约里会用到。

关于 `mask` 参数：它是一个 32 位掩码，声明**warp 内哪些线程参与本次操作**，第 n 位为 1 表示 lane n 参与，`0xffffffff` 即全员参与。执行时硬件会先让 mask 指定的线程会齐（同步），再交换数据——这就是函数名里 `_sync` 后缀的含义。

> [!IMPORTANT]
> 为什么必须显式传 mask？因为从 Volta（CC 7.0）架构起，GPU 支持**独立线程调度（Independent Thread Scheduling）**：warp 内发生分支后，两侧线程可能**交错推进**，不再保证"warp 天然齐步走"。旧式不带 `_sync` 后缀的 `__shfl` 等原语依赖这种隐式齐步假设，在新架构上已不安全（并已被弃用移除）。写新代码一律使用 `_sync` 版本，并确保 mask 覆盖的线程**都会执行到这条指令**——让 mask 里的线程"缺席会合"是未定义行为。

### 7.4.2 投票原语：warp 内的举手表决

除了交换数据，硬件还提供一组"表决"原语，一条指令收集全 warp 的判断结果：

```c++
__ballot_sync(mask, pred);    // 收集全 warp 的谓词到一个 32 位掩码
__any_sync(mask, pred);       // warp 内任一线程满足条件？
__all_sync(mask, pred);       // warp 内全部线程满足条件？
__activemask();               // 当前活跃线程的掩码
```

把 warp 想成一个 32 人小班：`__ballot_sync` 是"满足条件的举手，我拍张照"——返回值的第 n 位记录 lane n 的 `pred` 真假；`__any_sync`/`__all_sync` 则直接回答"有人举手吗？"/"全举了吗？"。`__ballot_sync` 配合位计数函数 `__popc`（population count，数一个整数里有几个 1）能玩出高效花样：`__popc(__ballot_sync(0xffffffff, pred))` 一次拍照加一次数数，就统计出 warp 内满足条件的线程数——比 32 次原子加便宜得多（见 7.8 节练习 3）。`__activemask()` 返回"此刻真正活跃的线程"掩码，常用于分支内确定可用的 mask。

### 7.4.3 Warp 级归约

shuffle 最经典的用武之地就是归约。用 `__shfl_down_sync` 实现 warp 内 32 个数的归约——**5 条指令完成，无共享内存、无 `__syncthreads()`**：

```c++
__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;   // lane 0 持有全 warp 的和
}
```

归约过程示意：

```text
offset=16:  lane0 += lane16,  lane1 += lane17, ...   （32 → 16 个部分和）
offset=8 :  lane0 += lane8, ...                       （16 → 8）
offset=4 :  ...                                       （8 → 4）
offset=2 :  ...                                       （4 → 2）
offset=1 :  lane0 += lane1                            （2 → 1，结果在 lane0）
```

思路和第 4 章的交错配对归约一模一样——跨步减半、层层对折——只是"内存位置"换成了"lane"。每一轮，前半部分的 lane 把后半部分对应 lane 的值加到自己身上；32 = 2⁵，所以对折 5 轮、恰好 5 条 shuffle 指令，最终 lane 0 手握全 warp 的总和。

> [!NOTE]
> 细心的读者会问：`offset=16` 那一轮，lane 16~31 去读 lane 32~47——越界了怎么办？回忆 7.4.1 节的规则：源 lane 越界时线程拿回**自己的值**，于是 lane 16 执行的是 `val += val`（自己翻倍）。这些"翻倍"的值确实是错的，但没关系——**我们只取 lane 0 的结果**，而 lane 0 每一轮读到的都是正确的部分和。高位 lane 只是"陪跑"，不污染答案。

对比第 5 章的共享内存归约（[5.3.8 节](../05_memory_model/README.md#538-综合实例二共享内存优化并行归约)），warp 级归约的优势立竿见影：

| 维度 | 共享内存归约（块内最后 32 个元素） | warp shuffle 归约 |
|------|------------------------------------|-------------------|
| 数据通道 | 寄存器 → 共享内存 → 寄存器 | 寄存器 → 寄存器 |
| 同步 | 每轮 `__syncthreads()`（全块会齐） | 无需块级同步（`_sync` 只涉及本 warp） |
| 共享内存占用 | 需要一块 `__shared__` 数组 | 零（省下的容量可提高占用率，见第 4 章） |
| 指令数（32 个数） | 每轮"读 + 加 + 写 + 同步" | 5 条 shuffle + 5 条加法 |

不过 shuffle 的射程只有一个 warp（32 个线程）——要归约整个块、整个网格，还得把它和共享内存、原子操作组合起来。这正是下一节的主角。

## 7.5 综合实例：三层归约内核

从第 4 章一路优化过来的归约，终于要在这里收官。这一节把全书的技巧组装成一个内核：**warp 内用 shuffle，块内用共享内存，块间用原子加**——三层结构（warp → block → grid），一次内核启动出最终结果。代码来自 [`code/reduce_final.cu`](code/reduce_final.cu)。

### 7.5.1 设计思路

回顾归约的痛点演进：第 4 章版本的块内每一轮都要 `__syncthreads()`，且结果是"每块一个部分和"，还得再启动一次内核（或拷回 CPU）做最终求和；第 5 章把迭代搬进共享内存省了全局内存读写，但同步和二次归约的问题仍在。现在手里的新武器正好各管一段：

- **grid-stride loop（第 2 章）**：让每个线程先串行累加多个元素，把百万级数据浓缩到"每线程一个部分和"——串行累加不需要任何同步，是最便宜的归约；
- **warp shuffle（7.4 节）**：把"每线程一个"浓缩到"每 warp 一个"，零共享内存、零块级同步；
- **共享内存（第 5 章）**：块内各 warp 的部分和存进共享内存，凑成不超过 32 个数，交给第一个 warp 再来一轮 shuffle 归约——"每块一个"；
- **原子加（7.1 节）**：每块只剩 1 个数，直接 `atomicAdd` 进全局结果——每块仅一次原子操作，冲突可忽略不计，却省掉了整个"第二次内核启动"。

### 7.5.2 完整内核

结合第 4 章、第 5 章 5.3 节的归约演进，最终形态如下：

```c++
__global__ void reduceFinal(const int *g_idata, int *g_odata, unsigned int n) {
    // 1. 每线程用 grid-stride loop 先串行累加多个元素（提高算术强度）
    int sum = 0;
    int stride = gridDim.x * blockDim.x;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        sum += g_idata[i];

    // 2. warp 内归约（shuffle，无同步开销）
    sum = warpReduceSum(sum);

    // 3. 各 warp 的部分和经共享内存汇聚，再由第一个 warp 归约
    __shared__ int warpSums[32];               // 最多 1024/32 = 32 个 warp
    int lane   = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) warpSums[warpId] = sum;
    __syncthreads();

    int nWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (warpId == 0) {
        sum = (lane < nWarps) ? warpSums[lane] : 0;
        sum = warpReduceSum(sum);
        // 4. 块结果直接原子加到最终结果（省去第二次内核启动）
        if (lane == 0) atomicAdd(g_odata, sum);
    }
}
```

### 7.5.3 逐段拆解

**第 1 段：grid-stride 串行累加**。相邻线程读相邻元素（合并访问，第 5 章），跨一整个网格再读下一批；循环结束时每个线程的寄存器里有一个部分和。让每个线程"多干几个元素的活"提高了算术强度，也让后续归约的规模与数据量无关——数据再大，进入第 2 段时也只有"线程数"个部分和。

**第 2 段：warp 内归约**。就是 7.4.3 节的 `warpReduceSum`，5 条 shuffle 把 32 个部分和折成 1 个，落在每个 warp 的 lane 0 手里。

**第 3 段：块内跨 warp 汇聚**。shuffle 出不了 warp，跨 warp 只能借道共享内存——但这次共享内存的用量小得可怜：块最多 1024 个线程、也就是最多 1024 / 32 = 32 个 warp，所以 `warpSums[32]` 固定 32 个 int（128 字节）就够了，对占用率几乎零影响。每个 warp 的 lane 0（部分和的持有者）把结果存入 `warpSums[warpId]`，`__syncthreads()` 确保全部存完——这是整个内核**唯一一次**块级同步。随后第一个 warp（`warpId == 0`）把这些部分和读进寄存器再做一轮 `warpReduceSum`。注意 `sum = (lane < nWarps) ? warpSums[lane] : 0;` 这个边界处理：块大小是 256 时只有 8 个 warp，`warpSums[8..31]` 从未被写过，超出部分的 lane 必须补 0，否则会把未初始化的垃圾值卷进总和。

**第 4 段：原子聚合**。块内归约完成后，lane 0 手握全块总和，直接 `atomicAdd(g_odata, sum)`。全网格只有"块数"次原子加（配套代码里是 `32 × SM 数` 次），摊到整个内核的运行时间里微不足道，却让归约**一次内核启动就得到最终标量**——第 4、5 章"每块一个部分和，再二次归约"的尾巴被彻底剪掉了。

这个版本综合了本指南讲过的全部技巧：**grid-stride loop（第 2 章）→ 无分化归约（第 4 章）→ 共享内存汇聚（第 5 章）→ warp shuffle + 原子聚合（本章）**。

> [!TIP]
> 使用原子聚合时别忘了**启动内核前把 `g_odata` 清零**（配套代码用 `cudaMemset`）——原子加是累加语义，残留的旧值会直接加进结果里。另外，本例归约的是 int，原子加满足结合律，结果确定；若归约 float 并以 `atomicAdd` 聚合，块间提交顺序不定会带来舍入级的非确定性（回顾 7.2.1 节的警告）。

### 7.5.4 主机端与性能度量

配套代码 [`code/reduce_final.cu`](code/reduce_final.cu) 的主机端套路与直方图一致：`blocks = 32 * numSMs` 配合 grid-stride loop；64M 个 1 求和并与期望值比对输出 `PASS/FAIL`；用 CUDA 事件计时后按"归约只读一遍数据"折算**有效带宽**：

```c++
// 有效带宽（归约只读一遍数据）
double gbps = (double)bytes / (ms * 1e-3) / 1e9;
```

为什么用带宽做指标？因为归约每个元素只做一次加法，是典型的**访存受限**问题（第 5 章）——它的理论极限就是"把数据从全局内存读一遍"的时间。这个终极版本在现代 GPU 上通常能逼近实测内存带宽的九成上下，意味着优化空间已经见底。把它与第 4、5 章各版本的带宽画在一条曲线上（7.8 节练习 2），就是一部浓缩的 CUDA 优化史。

写到这里你可能有个感受：`__syncthreads()`、shuffle、原子加各管一段，"哪些线程在协作"全靠脑内记账。CUDA 9 引入的协作组，正是把这笔账变成显式代码的现代化接口。

## 7.6 协作组（Cooperative Groups）简介

CUDA 9 引入的**协作组**把"线程组"抽象成一等公民，提供比 `__syncthreads()`/warp 原语更灵活、更安全的协作接口。传统写法里，"组"是隐式的：`__syncthreads()` 隐含"全块"，shuffle 隐含"本 warp"，mask 靠手写十六进制——组的边界只存在于程序员脑子里，编译器无从检查。协作组把它变成**显式的对象**：

```c++
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel(...) {
    cg::thread_block block = cg::this_thread_block();     // 整个线程块
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);  // 划分成 warp

    block.sync();                          // 等价 __syncthreads()
    float v = warp.shfl_down(val, 8);      // 等价 __shfl_down_sync
    cg::thread_block_tile<8> sub = cg::tiled_partition<8>(warp);  // 还能再细分！
}
```

几个核心概念：

- **`cg::this_thread_block()`**：拿到"我所在的线程块"这个组对象，`block.sync()` 等价于 `__syncthreads()`，此外还有 `block.thread_rank()`（我在组内的编号）、`block.size()` 等成员，把散落的内建变量收进统一接口；
- **`cg::tiled_partition<N>(group)`**：把一个组切成大小为 N 的子组（N 须为 2 的幂且不超过 32 的 tile 拥有 shuffle 能力），`tiled_partition<32>` 切出来的就是 warp。子组对象自带 `shfl_down`、`any`、`ballot` 等方法——**不再需要手写 mask**，"哪些线程参与"由组对象自己描述，从根源上避免了 7.4.1 节警告的 mask 写错问题；
- **组还能再分**：对 warp tile 再做 `tiled_partition<8>` 得到 8 线程的小组，各小组独立同步、独立归约——这是传统原语很难表达的粒度。

优点：组大小显式化（编译期检查）、支持任意 2 的幂子组、可表达网格级同步（`cg::this_grid().sync()`，需要 cooperative launch）。新代码建议优先使用协作组接口。

> [!NOTE]
> 关于**网格级同步**多说一句：第 1 章立下的规矩是"块间不能有依赖"，而 `cg::this_grid().sync()` 是这条规矩的官方例外——代价是内核必须通过 `cudaLaunchCooperativeKernel` 以协作方式启动，且所有块必须能**同时**驻留在 GPU 上（块数受硬件限制）。它适合迭代求解器这类"每轮全网格同步一次"的场景，属于高级特性，入门阶段知道存在即可。

用协作组重写本章的归约（7.8 节练习 5），你会发现代码逻辑不变，但"谁和谁协作"从注释变成了类型——这正是它的价值所在。

## 7.7 本章小结

- 多线程"读-改-写"同一位置必须用**原子操作**；所有原子函数返回旧值；`atomicCAS` 是万能原语；
- 原子操作**不可分割但有代价**：同地址冲突激烈时硬件串行化；它也不充当内存栅栏，浮点原子累加因顺序不定而有舍入级非确定性；
- 原子操作的优化套路是**分层聚合**：先共享内存块内聚合，最后一次性提交全局（直方图从 n 次全局原子降到 blocks×bins 次）；
- **warp shuffle** 让 warp 内线程直接互读寄存器：无共享内存、无同步开销，5 条指令完成 warp 归约；Volta 起须用带 mask 的 `_sync` 版本；
- 投票原语（`__ballot_sync`/`__any_sync`/`__all_sync`）一条指令完成 warp 内表决，配合 `__popc` 可高效计数；
- 终极归约 = grid-stride 串行累加 + warp shuffle + 共享内存汇聚 + 原子聚合，综合了全书技巧，一次内核启动逼近带宽上限；
- 新代码建议使用**协作组（Cooperative Groups）**：组大小显式、可任意细分、支持网格级同步。

## 7.8 动手练习

> 本章示例代码位于 [`code/`](code/) 目录：`histogram.cu`（两版本对比）、`reduce_final.cu`（终极归约）。

1. 运行 `histogram.cu`，把数据分布从 `rand() % 16`（偏斜）改为 `rand() % 256`（均匀），观察两版本差距的变化，解释原因；
2. 运行 `reduce_final.cu`，对比第 4 章与第 5 章各版归约的带宽，绘制"优化演进曲线"；
3. 用 `__ballot_sync` + `__popc` 实现"统计 warp 内满足条件的线程数"，替代 32 次原子加；
4. 把 `warpReduceSum` 改用 `__shfl_xor_sync` 实现蝶形归约（所有 lane 都得到总和），验证与 `__shfl_down_sync` 版本的结果一致性；
5. 用协作组接口重写 `reduce_final.cu`，体会 `tiled_partition<32>` 的显式组语义；
6. 参考 7.2.3 节的 CAS 重试循环，实现浮点版 `atomicMax(float*, float)`，并构造多线程并发场景验证其正确性。

---

> [← 上一章：流与并发](../06_streams_and_concurrency/README.md) | [返回目录](../README.md) | [下一章：最佳实践 →](../08_best_practices/README.md)
