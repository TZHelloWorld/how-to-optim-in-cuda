# 第 3 章 基本语法与常用 API

> [← 上一章：编程模型](../02_programming_model/README.md) | [返回目录](../README.md) | [下一章：执行模型 →](../04_execution_model/README.md)

本章目标：系统掌握 CUDA C++ 的语法要素与日常开发必备的 Runtime API——修饰符、内存管理、同步、错误检查、计时和设备查询。

第 1 章建立了"CPU 指挥、GPU 干活"的全局图景，第 2 章解决了"海量线程如何组织、每个线程如何知道自己是谁"。到目前为止，你已经能看懂并写出向量加法这样的完整程序——但用到的语法和 API 都是"照葫芦画瓢"学来的。本章就来正式补齐这块地基：把散落在前两章代码里的 `__global__`、`<<<...>>>`、`cudaMalloc`、`CHECK` 宏等元素一一摊开讲清楚，再补上真实开发中天天要用、但前两章还没出场的几件工具——同步原语、错误检查、内核计时和设备查询。

可以把本章看成一次"工具箱盘点"，内容分三块，由浅入深：

```text
写内核要用的语法：函数修饰符 → 内核定义与执行配置 → 内置变量 → 边界检查
                   3.1          3.2               3.3       3.4
管数据要用的语法与 API：变量修饰符与内存体系 → 内存管理 API（三件套/锁页/统一内存）
                        3.5                  3.6
保证正确与度量性能的 API：同步原语 → 错误检查 → Event 计时 → 设备信息查询
                          3.7       3.8       3.9         3.10
```

本章的知识点偏"工具性"，第一遍通读建立框架即可，之后写代码时随时回来查——但 3.7（同步）和 3.8（错误检查）两节建议精读，那是新人 Bug 的重灾区。

## 本章目录

- [3.1 函数执行空间修饰符](#31-函数执行空间修饰符)
- [3.2 内核定义与执行配置](#32-内核定义与执行配置)
- [3.3 内置变量：线程的身份与坐标](#33-内置变量线程的身份与坐标)
- [3.4 边界检查与块数计算](#34-边界检查与块数计算)
- [3.5 变量修饰符与内存的作用域、生命周期](#35-变量修饰符与内存的作用域生命周期)
- [3.6 常用内存管理 API](#36-常用内存管理-api)
- [3.7 同步原语](#37-同步原语)
- [3.8 错误检查](#38-错误检查)
- [3.9 用 CUDA Event 给内核计时](#39-用-cuda-event-给内核计时)
- [3.10 GPU 设备信息查询](#310-gpu-设备信息查询)
- [3.11 本章小结](#311-本章小结)
- [3.12 动手练习](#312-动手练习)

---

## 3.1 函数执行空间修饰符

第 1 章讲过，一个 CUDA 程序是一份"双语文档"：主机代码和设备代码住在同一个 `.cu` 文件里，由 `nvcc` 分开编译（1.8 节）。那么编译器怎么知道哪个函数属于哪边？答案就是**函数执行空间修饰符（Function Execution Space Specifiers）**——相当于给每个函数贴上一张"工作地点"标签。

### 3.1.1 三种修饰符

| 修饰符 | 执行位置 | 调用位置 | 说明 |
|--------|---------|---------|------|
| `__global__` | 设备 | 主机（或设备，动态并行） | 即内核函数，返回值必须是 `void` |
| `__device__` | 设备 | 设备 | 只能被内核或其他设备函数调用 |
| `__host__` | 主机 | 主机 | 普通 CPU 函数（默认，可省略）；可与 `__device__` 联用生成两份代码 |

用第 1 章"总经理与车间"的类比：`__host__` 函数是总经理办公室里的日常事务，`__global__` 函数是总经理下达给车间的"开工指令"（跨越了主机与设备的边界，所以最特殊），`__device__` 函数则是车间内部的工序——只有车间里的人（其他设备函数或内核）才能调用它。

看一个三种修饰符同台的例子：

```c++
__device__ float square(float x) { return x * x; }        // GPU 内部工具函数

__host__ __device__ float lerp(float a, float b, float t) // 主机/设备两用
{ return a + t * (b - a); }

__global__ void kernel(float *out, const float *in) {     // 内核
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = square(in[i]);
}
```

- `square` 只在 GPU 上存在，内核里可以像普通函数一样调用它；
- `lerp` 同时贴了两张标签，`nvcc` 会**把它编译成两份代码**——一份 CPU 指令、一份 GPU 指令，主机代码和设备代码都能调用，特别适合数学工具函数这类"两边都用得上"的逻辑，避免写两遍；
- `kernel` 是内核，由主机通过 `<<<...>>>` 启动（下一节展开）。

### 3.1.2 组合规则与常见限制

修饰符不能随意组合，规则如下：

- **`__global__` 的返回值必须是 `void`**——一次内核启动会创建成千上万个线程，"返回值"根本无从谈起（几万个线程各返回一个值给谁？），计算结果只能写入设备内存再拷回主机；
- **`__global__` 不能与 `__host__` 或 `__device__` 联用**——内核的身份是唯一的；
- **`__host__` 与 `__device__` 可以联用**，如上面的 `lerp`，生成主机、设备两份代码；
- 从主机调用 `__global__` 函数是**异步**的（1.9 节 Hello World 里已经领教过）；而调用 `__device__` 函数就是普通的函数调用，没有异步语义。

> [!NOTE]
> 不写任何修饰符的函数默认就是 `__host__` 函数——这保证了普通 C++ 代码放进 `.cu` 文件里行为不变。换句话说，你以前写的所有 C++ 函数在 CUDA 的世界里都自动持有"主机工作证"。

有了"函数住在哪边"的概念，接下来看本章第一位主角——`__global__` 内核函数——的完整定义与启动方式。

## 3.2 内核定义与执行配置

第 2 章已经写过不少内核，但一直只用了执行配置最简单的形式 `<<<blocks, threads>>>`。这一节把内核的定义、启动语法和限制一次性讲全。

### 3.2.1 定义与启动语法

```c++
// Kernel definition
__global__ void vecAdd(float *A, float *B, float *C) {
    ...
}

// Kernel invocation
dim3 grid(16, 16);
dim3 block(8, 8);
vecAdd<<<grid, block>>>(A, B, C);
```

这里表示使用 $16 \times 16$ 的线程块网格启动，并且每个线程块中线程是 $8 \times 8$（也就是启动这个内核用了 $16 \times 16$ 个线程块，每个线程块包含 $8 \times 8$ 个线程）。总线程数 = $16 \times 16 \times 8 \times 8 = 16384$——回忆第 2 章：`dim3` 是三维向量类型，未指定的维度默认为 1，所以这是一个二维网格套二维块的配置，天然适合矩阵类数据。

### 3.2.2 执行配置的完整形式

执行配置的完整形式为 `<<<grid, block, sharedMemBytes, stream>>>`，共 4 个参数，后两个可选：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `grid` | 网格维度（块数） | 必填 |
| `block` | 块维度（每块线程数） | 必填 |
| `sharedMemBytes` | 动态共享内存字节数（见第 5 章 5.3 节） | 0 |
| `stream` | 所属 CUDA 流（见第 6 章） | 0（默认流） |

前两个参数你已经很熟了；后两个现在只需混个脸熟——第 5 章讲共享内存、第 6 章讲流的时候，它们会正式登场。

### 3.2.3 内核的限制与约定

内核毕竟运行在另一个世界（GPU）里，比普通 C++ 函数多一些限制与约定：

- 返回类型必须是 `void`（结果只能写入设备内存带回，见 3.1.2 节）；
- 参数通过常量内存自动传入，按值传递（**指针传的是设备地址**——把主机指针传进内核是新人经典错误，内核解引用它就是非法访问）；
- 不能使用可变参数、静态局部变量、函数指针（受限）；
- 支持模板、重载、`printf`、断言 `assert`——所以调试时可以在内核里直接打印，但请记得配合线程索引过滤（如 `if (i == 0) printf(...)`），否则一百万个线程齐刷刷打印，输出会淹没一切。

内核定义好了、也启动了，块内的每个线程拿到的是同一份代码——它们靠什么区分彼此？靠下一节的内置变量。

## 3.3 内置变量：线程的身份与坐标

第 2 章已经用 `threadIdx` 和 `blockIdx` 计算过全局索引，这里把全部内置变量（Built-in Variables）正式列全。以下内置变量**只能在设备代码中使用**（前四个都有 `.x`、`.y`、`.z` 属性），不需要声明、不需要赋值，每个线程一出生就带着：

- `threadIdx`：返回线程在其线程块内的索引；
- `blockDim`：给出线程块的尺寸，该尺寸在内核启动的执行配置中指定；
- `blockIdx`：返回网格中线程块的索引；
- `gridDim`：给出网格的尺寸，该尺寸在内核启动的执行配置中指定；
- `warpSize`：线程束大小（目前所有架构均为 32）。

一个好记的方式：**`Idx` 结尾的是"我在哪"（坐标），`Dim` 结尾的是"一共有多大"（尺寸）**。两个坐标（`threadIdx`、`blockIdx`）加两个尺寸（`blockDim`、`gridDim`），恰好构成第 2 章全局索引公式的全部原料：

```c++
int i = blockIdx.x * blockDim.x + threadIdx.x;   // 一维全局索引
```

`warpSize` 与前四个不同——它不描述线程的位置，而是硬件的属性（1.5.5 节介绍的 warp 大小）。第 4 章讲执行模型、第 7 章讲 warp 级原语时会真正用到它。

每个线程都有了唯一身份，但还剩一个现实问题：数据量往往不是线程数的整数倍，"多出来的线程"怎么办？

## 3.4 边界检查与块数计算

数据总量往往不是线程块大小的整数倍——1M 个元素除以 256 恰好除尽是运气好，现实中的 `n` 可能是任何数。因此每个 CUDA 程序都要处理好一对标准搭配：

1. **向上取整计算块数**。宁可多分一个"不满员"的块，也不能少分——否则末尾的数据就没有线程处理了。CUDA 提供了 `cuda::ceil_div` 用于执行向上取整以计算内核启动所需的块数（也可以手写 `(n + threads - 1) / threads`）：

    ```c++
    #include <cuda/cmath>

    // vectorLength is an integer storing number of elements in the vector
    int threads = 256;
    int blocks  = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    ```

2. **内核中进行边界检查**，防止多出来的线程越界访问：

    ```c++
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { ... }   // 越界的线程直接跳过
    ```

两步是配套的：向上取整保证"每个数据都有线程管"，边界检查保证"没有数据的线程不添乱"。第 1 章向量加法（1.11 节）里已经见过这对搭配，第 2 章的 grid-stride loop 是它的进阶形态——以后你会写无数遍。

> [!WARNING]
> 忘写边界检查的后果是**越界访问**——而且它属于 3.8 节将讲的"异步错误"，往往不会立刻报错，而是在之后某个同步点才爆出来，甚至默默写坏别的数据。养成"算全局索引后第一件事就是判界"的肌肉记忆。

至此，"写内核"所需的语法都齐了。接下来转向数据这一侧：变量放在 GPU 的哪种内存里、归谁可见、能活多久。

## 3.5 变量修饰符与内存的作用域、生命周期

函数有"工作地点"标签，变量也有"存储地点"标签。1.5.5 节预览过 GPU 的内存层次（寄存器 → 共享内存 → 缓存 → 全局内存），这一节回答的问题是：**在代码里如何把变量放进指定的那一层？**

### 3.5.1 变量内存空间修饰符

| 修饰符 | 存储位置 | 说明 |
|--------|---------|------|
| `__device__` | 全局内存 | 设备全局变量，所有线程可见，与应用同生命周期 |
| `__constant__` | 常量内存 | 只读，经常量缓存广播，warp 内所有线程读同一地址时效率极高 |
| `__shared__` | 共享内存 | 块内线程共享，与线程块同生命周期 |
| `__managed__` | 统一内存 | 主机与设备均可直接访问，由驱动自动迁移 |
| （无修饰符的内核局部变量） | 寄存器 / 本地内存 | 编译器自动分配；寄存器放不下时溢出到本地内存 |

> [!NOTE]
> `__device__` 出现了两次——修饰**函数**时表示"设备函数"（3.1 节），修饰**变量**时表示"放在全局内存的设备全局变量"。同名不同义，读代码时看它修饰的是什么。

用第 1 章"车间"的类比串起来：无修饰符的局部变量放在**寄存器**——工人手边的工具台，只有自己能碰；`__shared__` 变量放在**共享内存**——车间内部的公共物料架，本车间（线程块）的工人共用；`__device__` 和 `__constant__` 变量放在**显存**——全厂共用的中央仓库，只是后者带一条只读的"广播快线"（常量缓存）。

### 3.5.2 内存类型、作用域与生命周期总表

把上面的修饰符与其背后的内存类型对齐，可以汇总成一张必须烂熟于心的表：

| 内存类型 | 作用域 | 生命周期 | 物理位置 |
|----------|--------|----------|----------|
| Global（全局内存） | Grid | Application | Device（DRAM） |
| Constant（常量内存） | Grid | Application | Device（DRAM，片上有常量缓存） |
| Shared（共享内存） | Block | Kernel | SM（片上） |
| Local（本地内存） | Thread | Kernel | Device（DRAM，经 L1/L2 缓存） |
| Register（寄存器） | Thread | Kernel | SM（片上） |

> [!NOTE]
> 作用域为 Grid 表示网格内所有线程均可访问；作用域为 Block/Thread 则分别只对块内线程/单个线程可见。生命周期为 Application 的内存跨内核持续存在，Kernel 级的则在内核结束后失效。

观察这张表的规律：**作用域越大，离计算单元越远、速度越慢**——Thread 级的寄存器最快，Block 级的共享内存次之，Grid 级的全局内存最慢。这个"快的小而私有、慢的大而公共"的层次结构，正是第 5 章内存优化的全部舞台。

修饰符解决的是"变量放哪"的**静态**声明；而真实程序中最大头的数据——那些百万级的数组——是在运行时**动态**分配的，这就要请出内存管理 API。

## 3.6 常用内存管理 API

回忆第 1 章的六步流程（1.10 节）：分配设备内存、拷入数据、算完拷回、释放——其中三步都是内存管理 API 的工作。本节从最常用的"三件套"讲起，再介绍两种进阶的内存分配方式：锁页内存和统一内存。

### 3.6.1 基本三件套

```c++
cudaError_t cudaMalloc(void **devPtr, size_t size);          // 分配设备内存
cudaError_t cudaMemcpy(void *dst, const void *src,
                       size_t count, cudaMemcpyKind kind);   // 同步拷贝
// kind: cudaMemcpyHostToDevice / DeviceToHost / DeviceToDevice / HostToHost
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
cudaError_t cudaFree(void *devPtr);                          // 释放设备内存
cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch,
                            size_t width, size_t height);    // 2D 数组，自动行对齐
```

几个容易踩坑的细节：

- `cudaMalloc` 接收的是**二级指针**（`void **`）——因为它要修改你的指针本身（把分配到的设备地址写进去），所以调用时要取地址：`cudaMalloc(&dA, bytes)`；
- `cudaMalloc` 返回的指针指向**设备内存**，主机代码不能直接解引用它，只能把它传给内核或 `cudaMemcpy` 等 API 使用；
- `cudaMemset` 与 C 标准库的 `memset` 一样按**字节**填充——用它把浮点数组清零没问题，但想填成 `1.0f` 是办不到的；
- 注意所有 API 都返回 `cudaError_t` 错误码——这为 3.8 节的错误检查埋下伏笔。

### 3.6.2 cudaMemcpy 的方向与同步语义

`cudaMemcpy` 的第四个参数 `cudaMemcpyKind` 指明拷贝方向，共四种枚举值：

| 枚举值 | 方向 | 典型场景 |
|--------|------|---------|
| `cudaMemcpyHostToDevice` | 主机 → 设备 | 把输入数据"寄"给 GPU（六步流程第 3 步） |
| `cudaMemcpyDeviceToHost` | 设备 → 主机 | 把计算结果"寄"回来（第 5 步） |
| `cudaMemcpyDeviceToDevice` | 设备 → 设备 | 显存内部搬运，不走 PCIe，速度快得多 |
| `cudaMemcpyHostToHost` | 主机 → 主机 | 等价于 `memcpy`，很少用 |

> [!NOTE]
> 还有一个 `cudaMemcpyDefault`：让运行时根据指针的虚拟地址自动推断方向（依赖 1.5.5 节提到的统一虚拟地址空间）。写反方向参数是新人常见错误，用 `cudaMemcpyDefault` 可以避开，但显式写方向的代码可读性更好。

另一个必须建立的概念是**同步 API 与异步 API 的区分**——它决定了"这行代码返回时，活到底干完没有"：

| 操作 | 对主机而言 | 说明 |
|------|-----------|------|
| 内核启动 `<<<...>>>` | **异步** | 发出指令立即返回，GPU 在后台执行（1.9 节已见过） |
| `cudaMemcpy` | **同步** | 拷贝完成才返回，并会先等待同一流中先前的操作（如内核）完成——所以它自带"隐式同步"效果 |
| `cudaMemcpyAsync` | **异步** | 立即返回，需配合流与锁页内存使用（第 6 章） |
| `cudaMalloc` / `cudaFree` | **同步** | 返回时分配/释放已完成 |
| `cudaDeviceSynchronize` | **阻塞** | 专职等待：设备上所有先前任务完成才返回 |

这解释了第 1 章向量加法为什么不需要显式调用 `cudaDeviceSynchronize`——拷回结果的 `cudaMemcpy` 顺带完成了等待。

### 3.6.3 锁页内存（Pinned Memory）

`malloc` 分配的普通主机内存是**可分页（pageable）**的——操作系统随时可能把它换出到磁盘或搬到别的物理页，好比一个"随时可能搬家的临时仓库"，GPU 的 DMA 引擎不敢直接来取货。因此每次 `cudaMemcpy` 都要先经过一个内部锁页缓冲区中转。用 `cudaMallocHost` 直接分配**锁页（page-locked / pinned）**内存——挂了牌、保证不搬家的固定仓库——可以省去中转：

```c++
float *h_data;
cudaMallocHost(&h_data, bytes);      // 分配锁页主机内存
...                                   // 传输带宽显著高于 pageable 内存
cudaFreeHost(h_data);
```

- 优点：**传输带宽更高**（通常提升 2 倍左右）；且是异步拷贝 `cudaMemcpyAsync`（第 6 章）的前提；
- 缺点：占用不可换出的物理内存，分配过多会拖慢整个系统——**只对频繁参与传输的缓冲区使用**。

### 3.6.4 统一内存（Unified Memory）

前面的方式都要求你亲自管理两个世界的内存、来回 `cudaMemcpy`。统一内存提供了另一种思路——"一个指针、CPU/GPU 都能访问"的编程便利，由驱动按需自动迁移数据：

```c++
float *data;
cudaMallocManaged(&data, bytes);     // 统一内存：主机、设备代码都可直接解引用

for (int i = 0; i < n; i++) data[i] = 1.0f;   // CPU 直接写
kernel<<<blocks, threads>>>(data, n);          // GPU 直接用，无需 cudaMemcpy
cudaDeviceSynchronize();                       // 同步后 CPU 又可直接读
printf("%f\n", data[0]);

cudaFree(data);
```

对比六步流程，代码明显清爽了：不再需要 `hA`/`dA` 两套指针，也没有显式拷贝。但便利不是免费的——数据迁移并没有消失，只是从"你手动寄快递"变成了"驱动自动配送"，配送时机不再由你精确掌控。

> [!IMPORTANT]
> 注意示例中内核启动后的 `cudaDeviceSynchronize()`：内核是异步的，在确认 GPU 用完这块内存之前，CPU 不应去访问它——统一内存把拷贝省了，**同步一步都不能省**。

> [!TIP]
> 统一内存非常适合**快速原型开发**和不规则数据结构（链表、树）；但对性能敏感的规整数据，显式 `cudaMalloc` + `cudaMemcpy`（或配合 `cudaMemPrefetchAsync` 预取）通常更快、行为更可控。

内存的事讲完了。上面已经两次出现 `cudaDeviceSynchronize()` 这个"等一等"的角色，是时候系统盘点 CUDA 的同步原语了。

## 3.7 同步原语

CUDA 的世界里到处是并行与异步：块内成百上千个线程各跑各的，主机发完指令也不等 GPU。**同步原语的作用就是在需要"对齐进度"的地方画一条线**——1.5.2 节三大抽象中的"栅栏同步"，落到代码上就是这几个 API。它们作用的范围从小到大：

- `__syncwarp()`：warp 级同步（现代架构中 warp 内线程可独立调度，需要时显式同步）；
- `__syncthreads()`：**仅同步单个线程块内的线程**（块级栅栏），并保证块内共享内存/全局内存写入对块内其他线程可见。不同块之间无法用它同步；
- `cudaDeviceSynchronize()`：主机端阻塞等待设备上所有先前任务完成。

注意前两个是**设备代码里用的**（线程之间互相等），最后一个是**主机代码里用的**（CPU 等 GPU），不要混淆。用"包饺子"类比：`__syncthreads()` 是本桌桌长喊"都停一下，等最慢的人擀完皮再开始包"；`cudaDeviceSynchronize()` 是总经理站在车间门口，等所有桌全部交货才走。

为什么没有"全网格同步"？回忆 1.5.4 节：**块间独立是 CUDA 可扩展性的根基**——块可能先后被调度、根本不同时存在，自然无法互相等待。如果确实需要"所有块都算完再进行下一步"，标准做法是把工作拆成两个内核，内核启动的先后顺序天然构成全局同步点。

> [!WARNING]
> `__syncthreads()` 必须被块内**所有**线程执行到。把它放进只有部分线程进入的 `if` 分支中会导致死锁或未定义行为：
>
> ```c++
> if (threadIdx.x < 16) {
>     __syncthreads();   // 错误！其余线程永远到不了这里
> }
> ```
>
> 桌长喊了"人到齐再开工"，可有一半人根本不会来——全桌人就永远等下去了。

同步保证了"程序按你想的顺序跑"；但跑错了怎么知道？这就是错误检查的职责。

## 3.8 错误检查

CUDA API 出错时不抛异常、不打日志，**只是默默返回一个错误码**——像一张塞进你手里的回执单，你不看，它就永远沉默。第 1 章的 `CHECK` 宏已经替你挡过枪了，这一节讲清它的原理，以及内核这种"没有返回值的调用"怎么查错。

### 3.8.1 两类错误：同步错误与异步错误

CUDA Runtime API 都返回 `cudaError_t`，检查返回值即可；麻烦的是**内核启动本身不返回错误码**（`<<<...>>>` 不是函数调用），需要用 `cudaGetLastError()` 捕获。而且内核的错误还分两类，出现的时机不同：

| 类别 | 例子 | 捕获方式 |
|------|------|---------|
| **同步错误**（启动时即可发现） | 块大小超限、参数非法 | `cudaGetLastError()` 紧跟内核启动 |
| **异步错误**（执行中才发生） | 越界访问、非法地址 | 之后任意同步点（如 `cudaDeviceSynchronize`）才返回 |

道理不难理解：内核启动是异步的（3.6.2 节），主机发完指令就走了——"块大小超过 1024"这类配置问题在发指令那一刻就能发现（同步错误）；而"第 100 万号线程越界写"要等 GPU 真正跑到那里才会暴露，错误只能记在账上，由**下一个**跟 GPU 打交道的调用带回来（异步错误）。这就是新人常见的灵异现象——"报错的那行代码明明没问题"：**异步错误会甩锅给之后无辜的同步点**。

### 3.8.2 CHECK 宏：把检查变成习惯

每个调用都手写 `if (err != cudaSuccess)` 太啰嗦，工程实践中建议统一用宏封装：

```c++
#define CHECK(call)                                                       \
do {                                                                      \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(err));                                 \
        exit(1);                                                          \
    }                                                                     \
} while (0)

kernel<<<grid, block>>>(...);
CHECK(cudaGetLastError());        // 捕获启动错误（如配置非法）
CHECK(cudaDeviceSynchronize());   // 捕获内核执行期错误
```

宏里的 `__FILE__` 和 `__LINE__` 会自动展开成出错的文件名与行号，`cudaGetErrorString` 把错误码翻译成人类可读的描述——三样合起来，出错时一眼定位。内核启动后的两行是标准搭配：第一行查"指令发出去了没有"，第二行等 GPU 干完活、查"干的过程中出没出事"，正好对应 3.8.1 节的两类错误。

> [!NOTE]
> 与 `cudaGetLastError()` 相对的还有 `cudaPeekAtLastError()`：两者都返回最近一次的错误，区别是**前者会把错误状态重置回 `cudaSuccess`（取走回执），后者只看不取（错误状态保留）**。日常用前者即可；只想探查、不想清除状态时用后者。

> [!TIP]
> 调试越界访问的利器是 `compute-sanitizer`（老版本叫 `cuda-memcheck`）：
>
> ```bash
> compute-sanitizer ./app     # 精确定位非法内存访问的内核与代码行
> ```
>
> 生产代码中每个同步的 `cudaDeviceSynchronize()` 都有性能代价，不必在每次启动后都加；但**调试阶段**大方地加，配合 `compute-sanitizer`，能把"甩锅"的异步错误钉死在案发现场。

程序能跑对了，下一个问题自然是：跑得多快？

## 3.9 用 CUDA Event 给内核计时

性能优化的第一步是测量——测不准，一切优化都是盲人摸象。本节介绍 CUDA 官方的计时工具：**CUDA 事件（Event）**，也顺便回答一个常见疑问：为什么不直接用 `std::chrono`？

### 3.9.1 为什么 CPU 计时器不合适

CPU 端计时器（`std::chrono` 等）测内核有两个先天缺陷：

1. **内核启动是异步的**——不加同步的话，你测到的只是"发指令"的时间（微秒级），而不是内核真正的执行时间；必须先 `cudaDeviceSynchronize()` 再停表，而同步本身也有开销，会混进测量结果；
2. **测量位置隔着一条马路**——CPU 的表掐的是"主机视角"，包含启动开销、驱动调度等噪声。

CUDA Event 的思路不同：**让 GPU 自己打卡**。`cudaEventRecord` 把一个"打卡点"插进 GPU 的任务队列（流）里，GPU 执行到那里时记下自己时间线上的时刻——两个打卡点之间的间隔就是纯粹的 GPU 耗时，与主机端的噪声无关，分辨率约为 0.5 微秒。

### 3.9.2 Event 计时的基本骨架

```c++
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);                    // 在默认流中打起点
kernel<<<grid, block>>>(...);
cudaEventRecord(stop);                     // 打终点
cudaEventSynchronize(stop);                // 等待终点事件完成

float ms = 0.f;
cudaEventElapsedTime(&ms, start, stop);    // 两事件间隔（毫秒）
printf("kernel time: %.3f ms\n", ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

流程是固定的五步：创建两个事件 → 起点打卡 → 干活 → 终点打卡 → `cudaEventSynchronize` 等终点事件完成后取间隔。注意 `cudaEventElapsedTime` 返回的单位是**毫秒**。

### 3.9.3 完整示例与有效带宽

实际测量还要讲究两点，见本章示例 [`code/event_timing.cu`](code/event_timing.cu) 的核心片段：

```c++
// 预热（首次启动包含初始化开销，不计入测量）
vecAdd<<<blocks, threads>>>(dA, dB, dC, n);
CHECK(cudaDeviceSynchronize());

// 用 CUDA Event 在 GPU 时间线上打点
cudaEvent_t start, stop;
CHECK(cudaEventCreate(&start));
CHECK(cudaEventCreate(&stop));

const int iters = 100;
CHECK(cudaEventRecord(start));
for (int i = 0; i < iters; i++)
    vecAdd<<<blocks, threads>>>(dA, dB, dC, n);
CHECK(cudaEventRecord(stop));
CHECK(cudaEventSynchronize(stop));

float ms = 0.f;
CHECK(cudaEventElapsedTime(&ms, start, stop));
ms /= iters;
```

- **预热（warmup）**：首次内核启动包含 JIT、上下文初始化等一次性开销（1.8.3 节），先空跑一次再计时；
- **多次迭代取平均**：单次测量抖动大，跑 100 次取平均值更稳定。

拿到时间之后怎么判断"快不快"？衡量访存型内核时，习惯把时间换算成**有效带宽**并与硬件峰值对比：

$$\text{Effective Bandwidth (GB/s)} = \frac{\text{读取字节数} + \text{写入字节数}}{\text{耗时}}$$

例如向量加法读 2 个数组、写 1 个数组，共 `3 * n * 4` 字节；若达到峰值带宽的 80% 以上，说明该内核已接近访存极限，进一步优化空间不大。这个"与硬件上限对比"的思路会贯穿后面所有性能章节——毕竟内核不是和自己比快，而是和硬件的物理极限比。

那么"峰值带宽"这些硬件数字从哪来？这就是本章最后一件工具的用武之地。

## 3.10 GPU 设备信息查询

GPU 的很多参数信息对性能有直接影响：SM 数量决定该开多少块、每块共享内存上限决定优化策略、峰值带宽是 3.9 节的对比基准……这些数字每张卡都不同，写程序前先"摸底"。1.4.5 节留过一个伏笔——在代码里查询计算能力的办法，就是本节的 `cudaGetDeviceProperties`。

### 3.10.1 示例：枚举并查询所有设备

完整代码见 [`code/device_query.cu`](code/device_query.cu)：先用 `cudaGetDeviceCount` 数一数机器上有几块 GPU，再对每块调用 `cudaGetDeviceProperties` 把属性结构体填满、逐项打印：

```c++
#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Detected %d CUDA capable device(s)\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute capability:            %d.%d\n", prop.major, prop.minor);
        printf("  Global memory:                 %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
        printf("  GPU clock rate:                %.0f MHz\n", prop.clockRate * 1e-3f);
        printf("  Memory clock rate:             %.0f MHz\n", prop.memoryClockRate * 1e-3f);
        printf("  Memory bus width:              %d-bit\n", prop.memoryBusWidth);
        printf("  Peak memory bandwidth:         %.1f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  L2 cache size:                 %d KB\n", prop.l2CacheSize / 1024);
        printf("  Constant memory:               %zu KB\n", prop.totalConstMem / 1024);
        printf("  Shared memory per block:       %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("  Shared memory per SM:          %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("  Registers per block:           %d\n", prop.regsPerBlock);
        printf("  Warp size:                     %d\n", prop.warpSize);
        printf("  Max threads per SM:            %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max threads per block:         %d\n", prop.maxThreadsPerBlock);
        printf("  Max block dimensions:          (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max grid dimensions:           (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Number of SMs:                 %d\n", prop.multiProcessorCount);
        printf("\n");
    }
    return 0;
}
```

代码里有一处小计算值得说明：峰值带宽那行的 `2.0 * memoryClockRate * (busWidth / 8)`——内存频率 × 总线字节宽度得到每周期传输量，乘 2 是因为 DDR 类显存每个时钟周期传输两次数据。这个数字正是 3.9.3 节有效带宽的对比基准。

### 3.10.2 值得关注的字段

`cudaDeviceProp` 结构体有上百个字段，值得关注的参数包括：

1. CUDA 驱动/运行时版本；
2. 设备计算能力（Compute Capability）编号；
3. 全局内存大小；
4. GPU 主频；
5. GPU 内存带宽（总线宽度 × 内存频率）；
6. L2 缓存大小；
7. 纹理维度最大值（各维度）；
8. 层叠纹理维度最大值；
9. 常量内存大小；
10. 块内共享内存大小；
11. 块内寄存器数量上限；
12. 线程束大小（warpSize = 32）；
13. 每个 SM 可处理的最大线程数；
14. 每个块可容纳的最大线程数；
15. 块的最大尺寸（各维度）；
16. 网格的最大尺寸（各维度）；
17. 最大连续线性内存等。

其中和后续章节关系最密切的几个：**SM 数量**（`multiProcessorCount`，第 4 章讨论占用率时的分母）、**每 SM 最大线程数与每块最大线程数**（执行配置的硬上限，块大小超过 `maxThreadsPerBlock` 正是 3.8.1 节"同步错误"的典型例子）、**共享内存大小**（第 5 章优化的资源预算）、**峰值带宽**（3.9 节的对比基准）。

> [!TIP]
> CUDA Toolkit 自带的 `deviceQuery` 示例程序（以及 `nvidia-smi -q`）也能查看这些信息；但自己写一遍这几十行，你会对"哪些硬件参数影响性能"有真切得多的体感。

## 3.11 本章小结

- 三类函数修饰符：`__global__`（内核，返回值必须为 `void`，不可与其他修饰符联用）、`__device__`（设备函数）、`__host__`（主机函数，可与 `__device__` 联用生成两份代码）；
- 执行配置完整形式 `<<<grid, block, sharedMemBytes, stream>>>`；内置变量两坐标（`threadIdx`/`blockIdx`）加两尺寸（`blockDim`/`gridDim`），另有 `warpSize`；
- 块数**向上取整**（`cuda::ceil_div` 或 `(n + threads - 1) / threads`）与内核内**边界检查**（`if (i < n)`）是标准搭配；
- 五类变量存储：寄存器（默认局部变量）、`__shared__`、`__device__`、`__constant__`、`__managed__`，牢记作用域与生命周期表——**作用域越大、速度越慢**；
- 内存管理三件套 `cudaMalloc / cudaMemcpy / cudaFree`；`cudaMemcpy` 有四种方向枚举且是**同步** API，内核启动与 `cudaMemcpyAsync` 是**异步**的；频繁传输用锁页内存（`cudaMallocHost`），快速原型用统一内存（`cudaMallocManaged`）；
- `__syncthreads()` 只同步块内且必须全员到达；主机端等待整个设备用 `cudaDeviceSynchronize()`；块间无同步原语——需要时拆成两个内核；
- 错误检查双保险：`cudaGetLastError()`（启动错误，读取并重置）+ 同步点（执行期错误）；`cudaPeekAtLastError()` 只读不重置；越界排查用 `compute-sanitizer`；
- 内核计时用 **CUDA Event**（GPU 时间线打卡，先预热、多次取平均），并换算成有效带宽与硬件峰值对比；硬件参数用 `cudaGetDeviceProperties` 查询。

工具箱备齐了。下一章我们潜入硬件内部，看这些线程在 SM 上究竟是如何被调度执行的——warp、SIMT、延迟隐藏，第 1 章埋下的伏笔将逐一展开。

## 3.12 动手练习

> 本章示例代码位于 [`code/`](code/) 目录：`device_query.cu`、`event_timing.cu`。

1. 运行 `device_query.cu`，记下你的 GPU 的 SM 数量、每 SM 最大线程数、共享内存大小和峰值带宽——后续章节都会用到这些数字；
2. 运行 `event_timing.cu`，计算向量加法达到峰值带宽的百分比；再把 `n` 缩小到 `1 << 10`，观察带宽数字为何暴跌（提示：内核太小，启动开销占主导）；
3. 把 `vec_add` 改为统一内存版本（`cudaMallocManaged`），对比代码简洁度与性能；
4. 写一段代码触发一个异步错误（内核里越界写），验证错误只在 `cudaDeviceSynchronize()` 处被报出，再用 `compute-sanitizer` 精确定位；
5. 试着把一个内核的返回类型从 `void` 改成 `float`，或给 `__global__` 函数同时加上 `__host__`，观察 `nvcc` 的报错信息——亲眼见过这些编译错误，以后排查会快得多；
6. 用 `<<<1, 2048>>>` 启动任意内核（超过 `maxThreadsPerBlock` 上限），验证这是一个**同步错误**：紧跟其后的 `cudaGetLastError()` 就能捕获，不需要等到同步点。

---

> [← 上一章：编程模型](../02_programming_model/README.md) | [返回目录](../README.md) | [下一章：执行模型 →](../04_execution_model/README.md)
