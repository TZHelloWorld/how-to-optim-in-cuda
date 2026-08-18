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
- [3.13 参考资料](#313-参考资料)

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
```

逐个把参数讲清楚：

- **`cudaMalloc(void **devPtr, size_t size)`**：在设备全局内存上分配 `size` 字节的**线性内存**，并把首地址写入 `*devPtr`。返回的地址保证至少 **256 字节对齐**，天然满足各种向量化访存（如 `float4`）的对齐要求；
- **`cudaMemcpy(dst, src, count, kind)`**：从 `src` 向 `dst` 拷贝 `count` 字节，`kind` 指明方向（下一小节展开）。注意参数顺序与 C 标准库 `memcpy` 一致——**目的在前、源在后**，写反了方向参数和指针就全乱了；
- **`cudaMemset(devPtr, value, count)`**：把设备内存的每个**字节**设置为 `value`，常用于结果缓冲区清零；
- **`cudaFree(devPtr)`**：释放 `cudaMalloc`（或 `cudaMallocManaged` 等）分配的内存。对同一指针 `cudaFree` 两次会返回错误；传入 `nullptr` 则是合法的空操作。

二维/三维数据还有一组带对齐的变体：

```c++
cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch,
                            size_t width, size_t height);    // 2D 数组，自动行对齐
cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
                         size_t width, size_t height, cudaMemcpyKind kind);
cudaError_t cudaMalloc3D(cudaPitchedPtr *pitchedDevPtr, cudaExtent extent);
```

`cudaMallocPitch` 会把每一行的起始地址**填充（padding）对齐**到硬件友好的边界，实际行宽（字节）通过 `pitch` 返回——访问第 `r` 行第 `c` 列元素时要用 `(float*)((char*)devPtr + r * pitch) + c`，而不能假设行与行紧密相连。官方推荐 2D/3D 数组用这组 API 分配，以保证按行访问时满足合并访存的对齐条件（第 5 章）。

此外还有一个"摸底"工具，写自适应程序（如根据剩余显存决定分块大小）时很有用：

```c++
size_t freeBytes, totalBytes;
cudaMemGetInfo(&freeBytes, &totalBytes);   // 查询当前设备的空闲/总显存（字节）
```

几个容易踩坑的细节：

- `cudaMalloc` 接收的是**二级指针**（`void **`）——因为它要修改你的指针本身（把分配到的设备地址写进去），所以调用时要取地址：`cudaMalloc(&dA, bytes)`；
- `cudaMalloc` 返回的指针指向**设备内存**，主机代码不能直接解引用它，只能把它传给内核或 `cudaMemcpy` 等 API 使用；
- `cudaMemset` 与 C 标准库的 `memset` 一样按**字节**填充——用它把浮点数组清零没问题，但想填成 `1.0f` 是办不到的（`1.0f` 的四个字节各不相同）；
- 注意所有 API 都返回 `cudaError_t` 错误码——这为 3.8 节的错误检查埋下伏笔。

> [!NOTE]
> CUDA 11.2 起还提供了**流有序内存分配器**（Stream-Ordered Memory Allocator）：`cudaMallocAsync` / `cudaFreeAsync`。它们把分配/释放作为流中的操作异步执行，并带有内存池复用机制，适合"频繁分配释放临时缓冲区"的场景（详见官方 [Stream Ordered Memory Allocator](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator) 一节）。初学阶段用同步版三件套即可。

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

锁页内存家族还有两个进阶成员，混个脸熟即可：

```c++
// 带标志的分配版本（cudaMallocHost 等价于 flags = cudaHostAllocDefault）
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
// 把 malloc/new 已分配好的普通内存"就地注册"为锁页内存（不搬数据）
cudaError_t cudaHostRegister(void *ptr, size_t size, unsigned int flags);
cudaError_t cudaHostUnregister(void *ptr);
```

`cudaHostAlloc` 的常用标志（可按位或组合）：

| 标志 | 含义 | 典型场景 |
|------|------|---------|
| `cudaHostAllocDefault` | 普通锁页内存 | 等价于 `cudaMallocHost` |
| `cudaHostAllocPortable` | 对**所有** CUDA 上下文均为锁页 | 多 GPU 程序 |
| `cudaHostAllocMapped` | 映射进设备地址空间，内核可直接访问（零拷贝内存），设备侧指针用 `cudaHostGetDevicePointer` 获取 | 集成显卡、数据只被 GPU 读写一次的场景 |
| `cudaHostAllocWriteCombined` | 写合并内存：CPU 写入更快、跨 PCIe 传输效率更高，但 **CPU 读取极慢** | 只由 CPU 写入、GPU 读取的单向缓冲区 |

`cudaHostRegister` 则适合"缓冲区不是我分配的"的场景——比如第三方库交给你一块 `malloc` 内存，注册后同样能享受高带宽传输与异步拷贝，用完记得 `cudaHostUnregister`。

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

统一内存的"自动配送"也可以人工干预。当你清楚数据接下来会在哪一侧使用时，两个 API 能显著减少按需迁移（缺页）的开销：

```c++
// 提前把数据搬到指定设备（dstDevice 填 cudaCpuDeviceId 表示搬回主机）
cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count,
                                 int dstDevice, cudaStream_t stream = 0);
// 给驱动提供访问模式提示，影响迁移策略
cudaError_t cudaMemAdvise(const void *devPtr, size_t count,
                          cudaMemoryAdvise advice, int device);
// advice 常用值：
//   cudaMemAdviseSetReadMostly     数据以只读为主，可在多处保留副本
//   cudaMemAdviseSetPreferredLocation  设定数据的"常驻地"
//   cudaMemAdviseSetAccessedBy     声明某设备会频繁访问，提前建立映射
```

典型用法是在内核启动前预取：`cudaMemPrefetchAsync(data, bytes, deviceId)`——相当于把"驱动按需零散配送"变成"提前整车发货"，统一内存的性能常常能因此接近显式拷贝。

> [!TIP]
> 统一内存非常适合**快速原型开发**和不规则数据结构（链表、树）；但对性能敏感的规整数据，显式 `cudaMalloc` + `cudaMemcpy`（或配合 `cudaMemPrefetchAsync` 预取）通常更快、行为更可控。

### 3.6.5 本节 API 速查

把本节出场的分配方式排在一起对比：

| 分配 API | 内存位置 | 主机可访问 | 设备可访问 | 释放 API | 适用场景 |
|----------|---------|-----------|-----------|---------|---------|
| `malloc` / `new` | 主机（可分页） | ✔ | ✘ | `free` / `delete` | 普通主机数据 |
| `cudaMalloc` | 设备全局内存 | ✘ | ✔ | `cudaFree` | 常规设备缓冲区（性能首选） |
| `cudaMallocHost` / `cudaHostAlloc` | 主机（锁页） | ✔ | 仅 Mapped 标志时 | `cudaFreeHost` | 频繁参与传输的主机缓冲区 |
| `cudaMallocManaged` | 统一内存（自动迁移） | ✔ | ✔ | `cudaFree` | 快速原型、不规则数据结构 |

内存的事讲完了。上面已经两次出现 `cudaDeviceSynchronize()` 这个"等一等"的角色，是时候系统盘点 CUDA 的同步原语了。

## 3.7 同步原语

CUDA 的世界里到处是并行与异步：块内成百上千个线程各跑各的，主机发完指令也不等 GPU。**同步原语的作用就是在需要"对齐进度"的地方画一条线**——1.5.2 节三大抽象中的"栅栏同步"，落到代码上就是这几个 API。它们作用的范围从小到大：

- `__syncwarp(mask = 0xffffffff)`：warp 级同步（现代架构中 warp 内线程可独立调度，需要时显式同步，`mask` 指定参与的线程）；
- `__syncthreads()`：**仅同步单个线程块内的线程**（块级栅栏），并保证块内共享内存/全局内存写入对块内其他线程可见。不同块之间无法用它同步；
- `cudaDeviceSynchronize()`：主机端阻塞等待设备上所有先前任务完成。

`__syncthreads()` 还有三个带"投票"功能的变体，同步之余顺便做一次块内统计（第 7 章讲 warp 级原语时会再遇到类似接口）：

```c++
int __syncthreads_count(int predicate);  // 返回 predicate 非零的线程数
int __syncthreads_and(int predicate);    // 所有线程 predicate 均非零才返回非零
int __syncthreads_or(int predicate);     // 任一线程 predicate 非零即返回非零
```

主机侧的"等待"其实也是一个家族，等待范围从大到小（后两个的主场在第 6 章）：

| API | 等待范围 | 说明 |
|-----|---------|------|
| `cudaDeviceSynchronize()` | 整个设备 | 所有流中所有先前任务完成才返回，最重量级 |
| `cudaStreamSynchronize(stream)` | 单个流 | 只等指定流，不打扰其他流（第 6 章） |
| `cudaEventSynchronize(event)` | 单个事件 | 只等流中某个"打卡点"，粒度最细（3.9 节） |

注意设备代码里的 `__syncwarp`/`__syncthreads` 是**线程之间互相等**，主机侧三个 API 是 **CPU 等 GPU**，不要混淆。用"包饺子"类比：`__syncthreads()` 是本桌桌长喊"都停一下，等最慢的人擀完皮再开始包"；`cudaDeviceSynchronize()` 是总经理站在车间门口，等所有桌全部交货才走。

> [!NOTE]
> 还有一族容易与栅栏混淆的**内存栅栏（memory fence）**函数：`__threadfence_block()` / `__threadfence()` / `__threadfence_system()`。区别在于：栅栏同步（`__syncthreads`）是"**等人**"——大家都到齐才继续；内存栅栏是"**等数据**"——只保证自己之前的写入按顺序对指定范围（块内/设备内/全系统）可见，**并不等待其他线程**。日常开发中 `__syncthreads()` 已自带块内可见性保证，内存栅栏主要用于无锁算法、块间通过全局内存传递数据等高级场景（第 7 章原子操作时会再遇到）。

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
> 生产代码中每个同步的 `cudaDeviceSynchronize()` 都有性能代价，不必在每次启动后都加；但**调试阶段**大方地加，配合下一小节的 `compute-sanitizer`，能把"甩锅"的异步错误钉死在案发现场。

### 3.8.3 排错利器 compute-sanitizer：从"灵异报错"到精确定位

`CHECK` 宏能告诉你"**出事了**"，但对越界访问这类异步错误，它最多告诉你"在某个同步点收到了 `an illegal memory access was encountered`"——**是哪个内核、哪一行代码、哪个线程、访问了哪个地址**，一概不知。补上这块短板的官方工具就是 **Compute Sanitizer**。

#### 它是什么

`compute-sanitizer` 是 CUDA Toolkit 自带的**功能正确性检查工具集**（位于 `$CUDA_HOME/bin/`，装好 Toolkit 即可用），前身是老工具 `cuda-memcheck`（CUDA 12 起已移除，被 compute-sanitizer 完全取代）。它的工作方式类似 CPU 世界的 Valgrind/AddressSanitizer：**以你的程序为参数启动它**，它在运行时对每一次设备内存访问、每一次同步调用做插桩检查，出错时精确报告到内核名、线程坐标甚至源码行号——不需要修改一行代码。

它其实是四个工具的集合，用 `--tool` 参数切换：

| 工具 | 选项 | 检查内容 | 典型症状 |
|------|------|---------|---------|
| **memcheck**（默认） | `--tool memcheck` | 越界/未对齐的内存访问、CUDA API 错误、硬件异常、内存泄漏（配 `--leak-check full`） | 结果错乱、`illegal memory access`、程序偶发崩溃 |
| **racecheck** | `--tool racecheck` | **共享内存**上的数据竞争（如漏写 `__syncthreads()`） | 结果不稳定、每次跑不一样 |
| **initcheck** | `--tool initcheck` | 读取**未初始化**的设备全局内存 | 结果里混入垃圾值 |
| **synccheck** | `--tool synccheck` | 同步原语的非法使用（如 3.7 节警告过的"分支里的 `__syncthreads()`"） | 死锁、结果未定义 |

常用命令行选项：

```bash
compute-sanitizer [options] ./app [app的参数]

# 常用 options：
#   --tool <name>          选择工具（默认 memcheck）
#   --leak-check full      memcheck 附带检查设备内存泄漏（有 cudaMalloc 没 cudaFree）
#   --log-file out.log     报告写入文件（%p 可展开为进程号）
#   --error-exitcode 1     检出错误时以非零码退出，方便接入 CI 脚本
#   --kernel-name kns=<子串>  只检查名字含指定子串的内核（大程序提速用），
#                          键值对语法，key 还可用 kne=<完整修饰名>、regex=<正则>
#   --launch-skip N        跳过前 N 次内核启动
#   --print-limit N        每类错误最多打印 N 条（默认 100，0 表示不限）
#   --padding N            在每个 CUDA 分配之后加 N 字节"隔离带"——相邻分配
#                          背靠背时越界会踩进邻居而漏报，加隔离带可抓出来
```

> [!NOTE]
> 想让报告显示**源码文件名与行号**，编译时要加 `-lineinfo`（保留行号信息，几乎不影响性能，可与 `-O3` 共存）或 `-G`（完整设备端调试信息，会关闭优化）。调试排错阶段建议养成 `nvcc -O3 -lineinfo` 的习惯。

#### 实战：一次完整的越界排查

下面用一个"每一行看起来都很合理"的错误程序，把排查过程完整走一遍。完整代码见 [`code/sanitizer_oob.cu`](code/sanitizer_oob.cu)，核心部分如下：

```c++
// 数组逆序：out[i] = in[n - 1 - i]，但埋了一个经典 bug
__global__ void reverseArray(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n) {                       // BUG：应为 i < n，差一错误（off-by-one）
        out[i] = in[n - 1 - i];         // i == n 时：写 out[n] 越界，读 in[-1] 越界
    }
}

int main(void) {
    const int n = 1000;                 // 故意取一个除不尽 256 的规模
    const size_t bytes = n * sizeof(float);
    ...
    CHECK(cudaMalloc(&dIn, bytes));
    CHECK(cudaMalloc(&dOut, bytes));
    CHECK(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;   // ceil(1000/256) = 4 块，共 1024 线程
    reverseArray<<<blocks, threads>>>(dIn, dOut, n);
    CHECK(cudaGetLastError());          // 启动配置合法，这里查不出任何问题
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
    ...
}
```

启动了 1024 个线程处理 1000 个元素——本该由 `if (i < n)` 拦下多余的 24 个线程，但 `<=` 放走了其中一个：线程 `i == 1000` 会写 `out[1000]`（合法下标只有 0~999）、读 `in[-1]`。

**第一步：正常运行，观察"症状"。**

```bash
$ nvcc -O3 -lineinfo sanitizer_oob.cu -o sanitizer_oob
$ ./sanitizer_oob
Verify: PASS (1000 elements checked)
```

程序居然 **跑通了，校验还全对** ！这正是越界写最阴险的地方：`out[n]` 恰好落在 `cudaMalloc` 分配粒度的填充区里，这次没砸到任何人。但它是一颗地雷——换个分配顺序、换张卡、数据规模一变，它可能写坏隔壁缓冲区，让一个毫不相干的内核输出错误结果，或触发 `illegal memory access` 让整个上下文报废。**"能跑通"不等于"没问题"。**

**第二步：交给 compute-sanitizer 复跑。**

```bash
$ compute-sanitizer ./sanitizer_oob
```

典型输出（不同环境地址与偏移会不同）：

```text
========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 4 bytes
=========     at reverseArray(const float *, float *, int)+0x70 in sanitizer_oob.cu:25
=========     by thread (232,0,0) in block (3,0,0)
=========     Address 0x7f5c92600fa0 is out of bounds
=========     and is 0 bytes after the nearest allocation at 0x7f5c92600000 of size 4000 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: main [0x8a2e] in sanitizer_oob
=========
========= Invalid __global__ read of size 4 bytes
=========     at reverseArray(const float *, float *, int)+0x60 in sanitizer_oob.cu:25
=========     by thread (232,0,0) in block (3,0,0)
=========     Address 0x7f5c923ffffc is out of bounds
=========     and is 4 bytes before the nearest allocation at 0x7f5c92400000 of size 4000 bytes
=========
========= ERROR SUMMARY: 2 errors
```

**第三步：读报告。** 这份报告的每一行都是破案线索，逐条解读：

| 报告内容 | 解读 |
|----------|------|
| `Invalid __global__ write of size 4 bytes` | 错误类型：对**全局内存**的非法**写**，宽度 4 字节——正好是一个 `float`，对应 `out[i] = ...` |
| `at reverseArray(...)+0x70 in sanitizer_oob.cu:25` | 案发内核与**源码行号**（编译时加了 `-lineinfo` 才有）：第 25 行，就是 `out[i] = in[n - 1 - i]` |
| `by thread (232,0,0) in block (3,0,0)` | 肇事线程坐标：块 3 的 232 号线程。反算全局索引 `i = 3 × 256 + 232 = 1000 = n`——**恰好越界 1 个元素**，差一错误的铁证 |
| `0 bytes after the nearest allocation ... of size 4000 bytes` | 非法地址紧贴在一块 4000 字节（= 1000 个 `float`，就是 `dOut`）分配的**末尾之后 0 字节**——"刚好多写了一个"的标准特征 |
| 第二条 `Invalid __global__ read` ... `4 bytes before` | 同一行还有一次非法**读**：`in[n - 1 - i]` 在 `i == n` 时变成 `in[-1]`，落在 `dIn` 首地址**之前 4 字节** |

三条线索（线程 `i == n`、写越界"尾后 0 字节"、读越界"头前 4 字节"）交叉印证，指向同一个结论：**边界条件把 `<` 写成了 `<=`**。

**第四步：修复并复验。** 把第 24 行改回 `if (i < n)`，重新编译后再跑一遍：

```bash
$ compute-sanitizer --leak-check full ./sanitizer_oob
========= COMPUTE-SANITIZER
Verify: PASS (1000 elements checked)
========= ERROR SUMMARY: 0 errors
```

`ERROR SUMMARY: 0 errors` 才是真正的通过。这里顺手加了 `--leak-check full`——如果程序里有 `cudaMalloc` 忘了配对 `cudaFree`，它会在这一步一并报出 `Leaked 4,000 bytes at 0x...`。

#### 在 PyTorch 自定义算子中使用

真实工程里，你的 CUDA 内核往往不是独立程序，而是编译成 PyTorch 扩展、由 Python 调用的自定义算子。compute-sanitizer **同样适用**——它拦截的是**进程内的所有 CUDA 活动**，不关心宿主程序是 C++ 还是 Python 解释器，直接把 `python` 当作被检查的程序即可：

```bash
compute-sanitizer python test_my_op.py
```

但 PyTorch 场景有三个特有的坑，直接照搬前面的流程很可能"查不出、慢到跑不完、报告没行号"。逐个拆解。

**坑一：缓存分配器会"掩护"越界（假阴性）。**

PyTorch 并不是每创建一个 tensor 就调一次 `cudaMalloc`——它的 **缓存分配器（caching allocator）** 会先向 CUDA 申请一大块显存池，再从池里切小块分给各个 tensor：

```text
compute-sanitizer 看到的：  [============ 一块 512 MB 的合法 cudaMalloc ============]
PyTorch 实际的切分：        [tensor A][tensor B][tensor C][........空闲........]
```

你的算子越界写穿了 tensor A、踩进了隔壁的 tensor B——但在 memcheck 眼里，这次访问**仍落在那块 512 MB 的合法分配之内**，一个错误都不会报。这是比前面实战里"分配粒度填充区"更彻底的假阴性：数据实实在在被写坏了（训练 loss 变 NaN、结果随机错乱），工具却说没问题。

解决办法是 PyTorch 官方专门为此保留的环境变量——**关掉缓存分配器**，让每个 tensor 都独立 `cudaMalloc`，边界恢复为真实的分配边界：

```bash
PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer python test_my_op.py
```

代价是分配退化为逐次 `cudaMalloc`/`cudaFree`，程序明显变慢——**只在调试时使用**。若仍怀疑"越界恰好踩进背靠背的相邻分配"而漏报，可以再加 `--padding 32`：sanitizer 会在每个分配之后垫一段永远非法的隔离带，踩进去必报错。

**坑二：PyTorch 自身的内核太多，插桩太慢。**

sanitizer 会插桩进程里的**每一个**内核——包括 PyTorch 内部的 elementwise、reduction、cuBLAS 调用等，一个训练 step 可能有成百上千次内核启动，整体慢几十倍。两个对策配合使用：

- **写最小复现脚本**：不要在完整训练脚本上跑 sanitizer；单独写十几行的测试脚本——构造小输入 → 调一次你的算子 → `torch.cuda.synchronize()`；
- **按内核名过滤**：用 `--kernel-name kns=<子串>` 只对你自己的内核做完整检查，跳过 PyTorch 内部内核的大部分开销。注意值是 `key=value` 格式：`kns`（`kernel_substring`）按子串匹配、`kne`（`kernel_name`）按完整名精确匹配、`regex` 按正则搜索——且匹配的都是**名字修饰（mangled）后**的内核名（如 `_Z14reverse_kernelPKfPfi`），所以用 `kns` 写个独特的子串最省事；还可以配 `--kernel-name-exclude` 反向排除。

**坑三：报告没有源码行号。**

和纯 C++ 一样，编译扩展时要给 `nvcc` 传 `-lineinfo`。两种常见构建方式的写法：

```python
# 方式一：setup.py（CUDAExtension）
CUDAExtension(
    name="my_op", sources=["my_op.cu", ...],
    extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lineinfo"]},
)

# 方式二：JIT 编译（load / load_inline）
mod = load_inline(..., extra_cuda_cflags=["-O3", "-lineinfo"])
```

三个坑都填上，来看一个完整的最小复现脚本 [`code/sanitizer_torch_op.py`](code/sanitizer_torch_op.py)——它用 `load_inline` JIT 编译了一个埋着同款 off-by-one bug 的"数组逆序"算子：

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
__global__ void reverse_kernel(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n)                          // BUG：应为 i < n
        out[i] = in[n - 1 - i];
}

torch::Tensor reverse_op(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    reverse_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

mod = load_inline(name="my_op",
                  cpp_sources="torch::Tensor reverse_op(torch::Tensor x);",
                  cuda_sources=cuda_src, functions=["reverse_op"],
                  extra_cuda_cflags=["-O3", "-lineinfo"])   # 行号信息

x = torch.arange(1000, dtype=torch.float32, device="cuda")  # 1000 除不尽 256
y = mod.reverse_op(x)
torch.cuda.synchronize()
print("correct:", torch.equal(y, x.flip(0)))
```

排查命令与典型输出：

```bash
$ PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
  compute-sanitizer --kernel-name kns=reverse_kernel --error-exitcode 1 \
      python sanitizer_torch_op.py

========= COMPUTE-SANITIZER
========= Invalid __global__ write of size 4 bytes
=========     at reverse_kernel(const float *, float *, int)+0x70 in cuda.cu:8
=========     by thread (232,0,0) in block (3,0,0)
=========     Address 0x7f... is out of bounds
=========     and is 0 bytes after the nearest allocation at 0x7f... of size 4000 bytes
...
========= ERROR SUMMARY: 2 errors
```

读报告的方法和前面完全一样：内核名、线程坐标（`3 × 256 + 232 = 1000 = n`）、"尾后 0 字节"——铁证依旧。唯一的差别是文件名：JIT 编译时你的 CUDA 源码被写进了缓存目录（`~/.cache/torch_extensions/<name>/cuda.cu`）里的生成文件，报告中的行号指向该文件（`load_inline` 会在你的源码前拼接若干 `#include`，行号相对你的字符串会整体偏移）；用 `setup.py` 编译独立 `.cu` 文件则没有这个问题，行号直指你的源文件。

> [!NOTE]
> 试着把脚本里的 `1000` 改成 `1024`（能被 256 整除），你会发现即使关了缓存分配器也报不出错——原因与前面实战用例相同：`i == n` 的越界线程根本没被启动，bug 被整除"藏"住了。给自定义算子写测试时，**务必覆盖除不尽块大小的形状**。

PyTorch 场景的排查节奏总结成三步：

1. **先定位是哪个算子**：正常运行加 `CUDA_LAUNCH_BLOCKING=1 python train.py`，强制内核同步启动，让异步错误回到肇事算子对应的 Python 调用栈；
2. **再写最小复现**：用小 tensor 单独调用该算子（记得测除不尽的形状）；
3. **最后上 sanitizer 精查**：`PYTORCH_NO_CUDA_MEMORY_CACHING=1` + `--kernel-name` 过滤 + `-lineinfo` 编译，定位到行。

另外两点提醒：`--leak-check full` 在 PyTorch 下噪声很大（缓存分配器持有的显存退出时不归还，会被误报为泄漏），一般不用；racecheck / initcheck / synccheck 对自定义算子同样有效，用法不变。

#### 使用建议

- **写完任何新内核，第一次跑就挂上 memcheck**——它把"越界了但侥幸没炸"这类未来的地雷当场引爆，成本只是程序变慢（通常数倍到几十倍，调试时完全可接受）；
- **结果时对时错、和线程数/运行次数有关**，优先怀疑共享内存竞争，跑 `--tool racecheck`；
- **结果里混入奇怪的垃圾值**，跑 `--tool initcheck`，检查是不是漏了初始化（注意 `cudaMalloc` 分配的内存**内容是未定义的**，并不保证为 0）；
- **程序卡死不动**，怀疑分支里的 `__syncthreads()`，跑 `--tool synccheck`；
- 在 CI 中用 `--error-exitcode 1` 让检出错误直接判为测试失败；
- compute-sanitizer 关注**功能正确性**；性能分析请用 Nsight Systems / Nsight Compute（第 8 章），两类工具各司其职。

程序能跑对了，下一个问题自然是：跑得多快？

## 3.9 用 CUDA Event 给内核计时

性能优化的第一步是测量——测不准，一切优化都是盲人摸象。本节介绍 CUDA 官方的计时工具：**CUDA 事件（Event）**，也顺便回答一个常见疑问：为什么不直接用 `std::chrono`？

### 3.9.1 为什么 CPU 计时器不合适

CPU 端计时器（`std::chrono` 等）测内核有两个先天缺陷：

1. **内核启动是异步的**——不加同步的话，你测到的只是"发指令"的时间（微秒级），而不是内核真正的执行时间；必须先 `cudaDeviceSynchronize()` 再停表，而同步本身也有开销，会混进测量结果；
2. **测量位置隔着一条马路**——CPU 的表掐的是"主机视角"，包含启动开销、驱动调度等噪声。

CUDA Event 的思路不同：**让 GPU 自己打卡**。`cudaEventRecord` 把一个"打卡点"插进 GPU 的任务队列（流）里，GPU 执行到那里时记下自己时间线上的时刻——两个打卡点之间的间隔就是 GPU 时间线上的耗时，与主机端的噪声无关，分辨率约为 0.5 微秒。（"与主机端无关"这句话有一个重要的限定条件，3.9.3 节专门算这笔细账。）

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

把 Event 家族的 API 列全（后两个进阶接口第 6 章还会用到）：

| API | 作用 | 备注 |
|-----|------|------|
| `cudaEventCreate(&e)` | 创建事件 | 用完须 `cudaEventDestroy` |
| `cudaEventCreateWithFlags(&e, flags)` | 带标志创建 | `cudaEventBlockingSync`：等待时让出 CPU（阻塞而非自旋）；`cudaEventDisableTiming`：不记时间戳，专用于第 6 章的跨流依赖，开销更小 |
| `cudaEventRecord(e, stream)` | 把"打卡点"插入流 | `stream` 省略即默认流 |
| `cudaEventSynchronize(e)` | 阻塞等待事件完成 | 计时前必须等终点事件 |
| `cudaEventQuery(e)` | **非阻塞**探询事件是否完成 | 完成返回 `cudaSuccess`，未完成返回 `cudaErrorNotReady`（这不是"出错"，注意别被 CHECK 宏误杀） |
| `cudaEventElapsedTime(&ms, start, stop)` | 两事件间隔（毫秒） | 两个事件都须已完成，且不能用 `cudaEventDisableTiming` 创建 |
| `cudaEventDestroy(e)` | 销毁事件 | — |

### 3.9.3 Event 测的究竟是什么：两条时间线的细账

一个常见疑问：`cudaEventRecord` 本身也是 CPU 调用、也要"入队"，内核启动也有开销——这些难道不会影响测量结果吗？答案分两层：**`cudaEventRecord` 的 CPU 开销不会计入，但内核在 GPU 侧的启动延迟会**。把两条时间线画开就清楚了：

```text
CPU 时间线:  [record(start)入队][launch K 入队][record(stop)入队]   ← 这些 API 调用
                    │                │               │               各花几微秒，但
                    ▼                ▼               ▼               不计入测量结果
GPU 队列:    [start 打卡] ─────→ [内核 K] ─────→ [stop 打卡]
GPU 时间线:      t0 ──(启动延迟)──→ K 执行完 ──→ t1
                  └────────── 测到的 = t1 - t0 ──────────┘
```

`cudaEventRecord` 是**异步**的：它只负责把"记录时间戳"这条命令排进流里就立刻返回，时间戳是 GPU 自己执行到那条命令时才记的。所以三次 CPU 调用各花多少微秒、CPU 有没有卡顿，都不直接进入 `t1 - t0`——这就是"免疫主机端噪声"的含义。

但 `t1 - t0` 也**并非纯粹的内核执行时间**，它是 GPU 时间线上两个打卡点之间的**墙钟时间**，拆开看包含四部分：

| 成分 | 量级 | 说明 |
|------|------|------|
| 内核真正的执行时间 | 主体 | 你想测的东西 |
| GPU 侧的内核启动/调度延迟 | 约 3~10 μs | GPU 从"处理完 start 打卡"到"内核第一个块开跑"之间的命令分发、块调度开销——**躲不掉，会被计入** |
| 打卡命令本身的执行 | 亚微秒 | 只是写一个时间戳（分辨率 ~0.5 μs 即由此而来），可忽略 |
| 两个打卡点之间 GPU 的任何**空转** | 视情况，可以很大 | 见下面的坑 |

最后一条是真正的坑：如果两个打卡点之间 GPU 在干等，**等待时间也被如实算进去**。典型反例：

```c++
cudaEventRecord(start);
do_some_cpu_work();              // CPU 先忙了 2 ms 才提交内核
kernel<<<grid, block>>>(...);    // GPU 处理完 start 打卡后一直闲着等命令
cudaEventRecord(stop);
// 测出来 ≈ 2 ms + 内核时间——CPU 的拖延以"GPU 空转"的形式混了进来
```

所以准确的说法是：Event 计时**免疫 CPU 的 API 调用开销，但不免疫"CPU 提交不及时导致的 GPU 空等"**——打卡与发工作必须背靠背写，中间别夹 CPU 逻辑。

这笔细账也解释了下一小节两个测量守则的由来：**预热**剔除首次启动的 JIT/初始化等一次性大头；**循环多次夹在一对 Event 之间取平均**，把每次几微秒的 GPU 侧启动延迟摊薄到可忽略——而且循环内 CPU 只做入队（远快于内核执行），队列始终喂饱，GPU 不会空转。反之，若**单次**测量一个只跑几微秒的小内核，启动延迟与内核时间同量级，结果会显著偏大。

> [!NOTE]
> 如果需要严格排除启动延迟的"纯内核硬件执行时长"，请用 profiler——Nsight Compute / Nsight Systems 直接从硬件读取内核的起止时间戳（第 8 章）。日常优化迭代用 Event（快、可嵌入代码、结果足够做相对比较），精细分析用 profiler，两者互补。

### 3.9.4 完整示例与有效带宽

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
- **多次迭代取平均**：单次测量抖动大，跑 100 次取平均值更稳定，同时把每次启动的 GPU 侧调度延迟（3.9.3 节）摊薄到可忽略。

拿到时间之后怎么判断"快不快"？衡量访存型内核时，习惯把时间换算成**有效带宽**并与硬件峰值对比：

$$\text{Effective Bandwidth (GB/s)} = \frac{\text{读取字节数} + \text{写入字节数}}{\text{耗时}}$$

例如向量加法读 2 个数组、写 1 个数组，共 `3 * n * 4` 字节；若达到峰值带宽的 80% 以上，说明该内核已接近访存极限，进一步优化空间不大。这个"与硬件上限对比"的思路会贯穿后面所有性能章节——毕竟内核不是和自己比快，而是和硬件的物理极限比。

那么"峰值带宽"这些硬件数字从哪来？这就是本章最后一件工具的用武之地。

## 3.10 GPU 设备信息查询

GPU 的很多参数信息对性能有直接影响：SM 数量决定该开多少块、每块共享内存上限决定优化策略、峰值带宽是 3.9 节的对比基准……这些数字每张卡都不同，写程序前先"摸底"。1.4.5 节留过一个伏笔——在代码里查询计算能力的办法，就是本节的 `cudaGetDeviceProperties`。

先把设备管理这组 API 列全：

```c++
cudaError_t cudaGetDeviceCount(int *count);                       // 有几块 GPU
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int dev); // 查属性
cudaError_t cudaSetDevice(int dev);      // 选定后续操作的目标 GPU（默认 0 号）
cudaError_t cudaGetDevice(int *dev);     // 查询当前选定的 GPU
cudaError_t cudaDriverGetVersion(int *v);   // 驱动支持的 CUDA 版本（如 12040 = 12.4）
cudaError_t cudaRuntimeGetVersion(int *v);  // 链接的 Runtime 版本
```

单卡程序通常只用前两个；多 GPU 时用 `cudaSetDevice` 切换目标设备——之后的 `cudaMalloc`、内核启动都作用于选定的卡。

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

代码里有一处小计算值得说明：峰值带宽那行的 `2.0 * memoryClockRate * (busWidth / 8)`——内存频率 × 总线字节宽度得到每周期传输量，乘 2 是因为 DDR 类显存每个时钟周期传输两次数据。这个数字正是 3.9.4 节有效带宽的对比基准。

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
- 内存管理三件套 `cudaMalloc / cudaMemcpy / cudaFree`；`cudaMemcpy` 有四种方向枚举且是**同步** API，内核启动与 `cudaMemcpyAsync` 是**异步**的；频繁传输用锁页内存（`cudaMallocHost` / `cudaHostAlloc`，已有内存可 `cudaHostRegister` 就地注册），快速原型用统一内存（`cudaMallocManaged`，可用 `cudaMemPrefetchAsync` / `cudaMemAdvise` 干预迁移）；
- `__syncthreads()` 只同步块内且必须全员到达；主机端等待按范围从大到小有 `cudaDeviceSynchronize` / `cudaStreamSynchronize` / `cudaEventSynchronize`；块间无同步原语——需要时拆成两个内核；
- 错误检查双保险：`cudaGetLastError()`（启动错误，读取并重置）+ 同步点（执行期错误）；`cudaPeekAtLastError()` 只读不重置；
- **compute-sanitizer** 是功能正确性检查的官方工具集：memcheck（越界/泄漏，默认）、racecheck（共享内存竞争）、initcheck（未初始化读）、synccheck（非法同步）；编译加 `-lineinfo` 可定位到源码行，新内核第一次跑就应挂上 memcheck；排查 PyTorch 自定义算子时直接 `compute-sanitizer python ...`，但要记得 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 关掉缓存分配器，否则越界会被显存池"掩护"成假阴性；
- 内核计时用 **CUDA Event**（GPU 时间线打卡，先预热、多次取平均），并换算成有效带宽与硬件峰值对比；注意 Event 免疫 CPU 调用开销，但测的是两打卡点间的 GPU **墙钟时间**——内核启动延迟和打卡点之间的 GPU 空转都会计入；硬件参数用 `cudaGetDeviceProperties` 查询。

工具箱备齐了。下一章我们潜入硬件内部，看这些线程在 SM 上究竟是如何被调度执行的——warp、SIMT、延迟隐藏，第 1 章埋下的伏笔将逐一展开。

## 3.12 动手练习

> 本章示例代码位于 [`code/`](code/) 目录：`device_query.cu`、`event_timing.cu`、`sanitizer_oob.cu`、`sanitizer_torch_op.py`。

1. 运行 `device_query.cu`，记下你的 GPU 的 SM 数量、每 SM 最大线程数、共享内存大小和峰值带宽——后续章节都会用到这些数字；
2. 运行 `event_timing.cu`，计算向量加法达到峰值带宽的百分比；再把 `n` 缩小到 `1 << 10`，观察带宽数字为何暴跌（提示：内核太小，3.9.3 节的启动延迟占了主导）；
3. 把 `vec_add` 改为统一内存版本（`cudaMallocManaged`），对比代码简洁度与性能；再加上 `cudaMemPrefetchAsync` 预取，看性能能否追回显式拷贝版本；
4. 编译运行 `sanitizer_oob.cu`（记得加 `-lineinfo`），复现 3.8.3 节的完整排查流程：先正常跑（大概率"侥幸"通过校验），再用 `compute-sanitizer` 定位到肇事线程与源码行，修复后复验至 `ERROR SUMMARY: 0 errors`；
5. 在 `sanitizer_oob.cu` 里注释掉一个 `cudaFree`，用 `compute-sanitizer --leak-check full` 验证泄漏能被查出；再把 `n` 改成 256 的整数倍（如 `1 << 20`），先预测 `compute-sanitizer` 还能不能查出越界，再运行验证（提示：想想这时启动了多少个线程、`i == n` 的线程还存不存在——bug 没被修好，只是被"藏"起来了）；
6. （装有 PyTorch 的环境）运行 `sanitizer_torch_op.py`，做一组对照实验：① 直接挂 `compute-sanitizer` 跑；② 加上 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 再跑。观察缓存分配器如何把越界"掩护"掉，体会为什么排查 PyTorch 算子必须关掉它；
7. 试着把一个内核的返回类型从 `void` 改成 `float`，或给 `__global__` 函数同时加上 `__host__`，观察 `nvcc` 的报错信息——亲眼见过这些编译错误，以后排查会快得多；
8. 用 `<<<1, 2048>>>` 启动任意内核（超过 `maxThreadsPerBlock` 上限），验证这是一个**同步错误**：紧跟其后的 `cudaGetLastError()` 就能捕获，不需要等到同步点。

## 3.13 参考资料

| 主题 | 官方文档 |
|------|---------|
| CUDA C++ 编程指南（语法、修饰符、内置变量） | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html |
| CUDA Runtime API 参考（本章全部 `cuda*` 函数的权威签名） | https://docs.nvidia.com/cuda/cuda-runtime-api/index.html |
| 内存管理 API（Memory Management） | https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html |
| 统一内存编程（Unified Memory Programming） | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd |
| 错误处理（Error Handling） | https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html |
| Compute Sanitizer 用户手册 | https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html |
| PyTorch CUDA 语义（缓存分配器与调试环境变量） | https://pytorch.org/docs/stable/notes/cuda.html |
| PyTorch 自定义 C++/CUDA 扩展 | https://pytorch.org/tutorials/advanced/cpp_extension.html |
| Event 管理（Event Management） | https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html |
| 设备管理与 `cudaDeviceProp` 字段 | https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html |

---

> [← 上一章：编程模型](../02_programming_model/README.md) | [返回目录](../README.md) | [下一章：执行模型 →](../04_execution_model/README.md)
