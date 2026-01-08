# Python 调用 CUDA Kernel 完全指南

> 本文系统介绍 Python 调用 CUDA Kernel 的完整技术链路。从 C/C++ 直接调用 CUDA Kernel 出发，自底向上依次讲解三种 Python/C++ 互通方案——CPython 原生 C Extension、pybind11、PyTorch C++ 扩展——并深入 PyTorch 算子注册与 Dispatcher 分发机制，结合 sgl-kernel 生产级案例分析工程实践；最后给出本目录可运行示例工程的构建系统（CMake）详解与完整的调试方法（ipdb / rpdb / cuda-gdb 联合调试）。
>
> 核心调用链路：
>
> ```
> Python  --->  绑定层 (pybind11 / C Extension)  --->  C/C++ 封装  --->  CUDA Kernel
> ```

---

## 目录

- [第 1 章 概述：调用链路与方案选型](#第-1-章-概述调用链路与方案选型)
- [第 2 章 前置知识：C/C++ 调用 CUDA Kernel](#第-2-章-前置知识cc-调用-cuda-kernel)
- [第 3 章 底层机制：Python C Extension](#第-3-章-底层机制python-c-extension)
- [第 4 章 pybind11：现代绑定方案](#第-4-章-pybind11现代绑定方案)
- [第 5 章 pybind11 调用 CUDA Kernel](#第-5-章-pybind11-调用-cuda-kernel)
- [第 6 章 PyTorch C++ 扩展：JIT 与 AOT 编译](#第-6-章-pytorch-c-扩展jit-与-aot-编译)
- [第 7 章 PyTorch 算子注册与 Dispatcher](#第-7-章-pytorch-算子注册与-dispatcher)
- [第 8 章 生产案例：sgl-kernel](#第-8-章-生产案例sgl-kernel)
- [第 9 章 示例工程：结构、构建与运行](#第-9-章-示例工程结构构建与运行)
- [第 10 章 构建系统：CMake 详解](#第-10-章-构建系统cmake-详解)
- [第 11 章 调试：从 Python 层到 Kernel 层](#第-11-章-调试从-python-层到-kernel-层)
- [第 12 章 总结与参考资料](#第-12-章-总结与参考资料)

---

## 第 1 章 概述：调用链路与方案选型

### 1.1 问题的本质

Python 解释器无法直接执行 GPU 代码。所谓"Python 调用 CUDA Kernel"，本质是一套**多层语言互通机制**：

```
Python (import xxx.so)
    └─> 绑定层：将 C/C++ 函数注册为 Python 可调用对象
            └─> C++ 封装函数：管理内存、启动 kernel、检查错误
                    └─> CUDA Kernel：<<<grid, block>>> 在 GPU 上并行执行
```

其中每一层解决一个独立的问题：

| 层次 | 解决的问题 | 关键技术 |
|------|-----------|---------|
| 绑定层 | Python 解释器如何"认识"C++ 函数 | `PyInit_*` 入口 / `PYBIND11_MODULE` |
| C++ 封装层 | kernel 的启动配置、错误处理 | `<<<>>>` 语法、`cudaGetLastError` |
| CUDA Kernel 层 | GPU 上的并行计算逻辑 | `__global__` 函数 |
| 构建系统 | 两种编译器（nvcc / g++）的协作与链接 | CMake / setuptools / Ninja |

### 1.2 三种主流方案

| 方案 | 定位 | 优点 | 代价 | 适用场景 |
|------|------|------|------|---------|
| **C Extension**（CPython 原生） | 底层机制 | 零依赖、最直接的控制力 | 样板代码多、手动管理引用计数 | 理解原理、构建基础库 |
| **pybind11** | C Extension 的现代封装 | 纯头文件、自动类型转换、代码量最少 | 需 C++11+ | 通用 Python/C++ 绑定首选 |
| **PyTorch C++ 扩展** | 面向 PyTorch 生态 | 自动处理 Torch 编译参数、可接入 Dispatcher | 依赖 PyTorch | 自定义算子、深度学习 kernel |

三者是**递进关系**而非并列关系：pybind11 底层生成的就是 C Extension 的 `PyInit_*` 入口；PyTorch 扩展底层使用 pybind11（`torch/extension.h` 包含了 pybind11 头文件）。本文按此顺序自底向上展开。

> 若 C++ 代码没有任何绑定入口而直接编译为 `.so`，Python 导入时会报错：
> `ImportError: dynamic module does not define module export function (PyInit_<module_name>)`
> ——这条错误信息正揭示了绑定层的本质：**Python 加载扩展模块时，寻找并调用约定命名的 `PyInit_<模块名>` 入口函数**。

### 1.3 全文路线图

```
第 2 章   C/C++ 如何启动 CUDA Kernel        ← GPU 侧基础
第 3 章   CPython 如何加载 C 扩展模块       ← Python 侧底层机制
第 4~5 章 pybind11 打通两侧                 ← 最小可用方案
第 6~7 章 PyTorch 扩展与 Dispatcher        ← 深度学习工程方案
第 8 章   sgl-kernel                        ← 生产级组合拳
第 9~10 章 本目录示例工程与 CMake           ← 动手实践
第 11 章  跨层调试                          ← 排查问题的能力
```

---

## 第 2 章 前置知识：C/C++ 调用 CUDA Kernel

在打通 Python 之前，先掌握纯 C/C++ 环境下 CUDA Kernel 的编写、启动与错误处理——后续所有方案的 C++ 封装层都构建在这些基础之上。

### 2.1 最小示例

CUDA 程序的基本要素是**核函数（Kernel Function）**：用 `__global__` 修饰，从 Host 端调用、在 Device（GPU）上由大量线程并行执行。

```c++
// example.cu
#include <stdio.h>

// __global__：声明核函数，在 GPU 上由每个线程执行一份
__global__ void demo_kernel() {
    printf("hello, world!!!");
}

int main(int argc, char *argv[]) {
    demo_kernel<<<1, 1>>>();     // 启动 1 个线程块 x 1 个线程
    cudaDeviceSynchronize();     // kernel 启动是异步的，等待 GPU 执行完成
    return 0;
}
```

```bash
nvcc example.cu -o example && ./example
```

注意：作为独立可执行程序编译时必须有 `main` 入口，否则链接报错：

```bash
/usr/bin/ld: /usr/lib/gcc/x86_64-linux-gnu/11/../../../x86_64-linux-gnu/Scrt1.o: in function `_start':
(.text+0x1b): undefined reference to `main'
collect2: error: ld returned 1 exit status
```

而编译为共享库（后续章节的场景）时则不需要 `main`。

### 2.2 Kernel 启动语法

完整的启动语法包含四个配置参数：

```c++
kernel_name<<<gridDim, blockDim, sharedMemBytes, stream>>>(args...);
```

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `gridDim` | `dim3` 或 `int` | Grid 维度：线程块的数量与排列 | 必填 |
| `blockDim` | `dim3` 或 `int` | Block 维度：每块的线程数与排列 | 必填 |
| `sharedMemBytes` | `size_t` | 每块动态共享内存字节数 | `0`（可省略） |
| `stream` | `cudaStream_t` | 执行所在的 CUDA Stream | `0`（默认流，可省略） |

**`dim3` 结构体**：最多三维，未指定的维度默认为 1：

```c++
dim3 grid(4, 4);                // 等价于 dim3(4,4,1)，共 16 个线程块
dim3 block(16, 16);             // 每块 256 个线程
kernel<<<grid, block>>>(args);  // 共 16 x 256 = 4096 个线程
```

**动态共享内存**（第三参数）：

```c++
__global__ void shared_mem_kernel(float* data, int N) {
    extern __shared__ float smem[];   // 大小由启动参数决定
    int idx = threadIdx.x;
    if (idx < N) {
        smem[idx] = data[idx];
        __syncthreads();              // Block 内同步，确保加载完成
        data[idx] = smem[idx] * 2.0f;
    }
}

int N = 256;
shared_mem_kernel<<<1, N, N * sizeof(float)>>>(d_data, N);
```

**CUDA Stream**（第四参数）：

```c++
cudaStream_t stream;
cudaStreamCreate(&stream);

kernel<<<grid, block, 0, stream>>>(args);   // 在指定流上异步执行

cudaStreamSynchronize(stream);
cudaStreamDestroy(stream);
```

> Stream 实现 Host 与 Device、多 Kernel 之间的异步并发，是吞吐量优化的重要手段。系统性介绍参见本仓库 `00_cuda_stream_sync/` 章节。

### 2.3 错误检查：必不可少的一环

CUDA 的错误报告机制有两个特点，决定了错误检查必须**主动**进行：

1. **kernel 启动是异步的**：launch 阶段的错误（如配置非法、架构不匹配）不会抛出异常，需调用 `cudaGetLastError()` 主动获取；
2. **执行期错误延迟暴露**：kernel 运行期间的错误（如非法内存访问）要到下一个同步点（如 `cudaDeviceSynchronize()`）才能捕获。

标准做法是封装检查宏：

```c++
// 通用 CUDA 运行时错误检查宏
#define CUDA_CHECK(expr_to_check) do {                       \
    cudaError_t result = (expr_to_check);                    \
    if (result != cudaSuccess) {                             \
        fprintf(stderr,                                      \
                "CUDA error at %s:%d code=%d(%s)\n",         \
                __FILE__, __LINE__,                          \
                result, cudaGetErrorString(result));         \
    }                                                        \
} while (0)

void launch_kernel() {
    demo_kernel<<<1, 1>>>();
    CUDA_CHECK(cudaGetLastError());        // 检查 launch 错误
    CUDA_CHECK(cudaDeviceSynchronize());   // 等待完成并检查执行期错误
}
```

**不加检查的后果**：kernel 静默失败，程序无任何提示。典型案例——编译时指定了设备不支持的架构：

```bash
# 设备不支持 sm_90，但编译不报错
nvcc -arch=sm_90 --shared ... example.cu

# 运行时无任何输出，kernel 实际未执行
>>> import example; example.run()

# 加上错误检查后才能看到真正的原因：
CUDA launch error: no kernel image is available for execution on the device
```

---

## 第 3 章 底层机制：Python C Extension

pybind11 等一切绑定方案的底层，都是 CPython 官方的 **C Extension Module 机制**。理解它，就理解了"Python 如何加载 C 代码"这一根本问题。

### 3.1 三要素

CPython 通过三个约定结构将 C/C++ 代码封装为 Python 模块：

| 要素 | 作用 |
|------|------|
| `PyMethodDef` | 方法表：函数名、C 函数指针、调用约定、文档字符串 |
| `PyModuleDef` | 模块元信息：模块名、文档、状态空间大小、方法表 |
| `PyInit_<模块名>` | 模块入口：解释器 `import` 时按命名约定查找并调用 |

加载流程：`import example_ops` → 解释器定位 `example_ops*.so` → `dlopen` 加载 → 查找符号 `PyInit_example_ops` → 调用它获得模块对象。

### 3.2 完整示例

```c
// example_ops.c
#include <Python.h>

// C 函数实现：所有导出函数的签名遵循 PyObject* (PyObject*, PyObject*)
static PyObject* hello_world(PyObject* self, PyObject* args) {
    printf("hello world\n");
    fflush(stdout);      // 强制刷新，避免输出滞留在 C 层缓冲区
    Py_RETURN_NONE;      // 返回 Python 的 None（自动处理引用计数）
}

// 方法表：{python 函数名, C 函数指针, 调用约定, 文档字符串}
static PyMethodDef ExampleMethods[] = {
    {"hello", hello_world, METH_NOARGS, "Print 'hello world' from C."},
    {NULL, NULL, 0, NULL}    // 哨兵：标记数组结束
};

// 模块入口：命名必须为 PyInit_<模块名>
PyMODINIT_FUNC PyInit_example_ops(void) {
    static struct PyModuleDef example_ops_module = {
        PyModuleDef_HEAD_INIT,
        "example_ops",                     // 模块名称
        "A simple C extension module.",    // 模块文档
        0,                                 // 模块状态空间（0 = 无状态）
        ExampleMethods
    };
    return PyModule_Create(&example_ops_module);
}
```

使用 setuptools 编译（AOT 预编译）：

```python
# setup.py
from setuptools import setup, Extension

setup(
    name='example_ops',
    version='1.0',
    ext_modules=[Extension('example_ops', sources=['example_ops.c'])],
)
```

```bash
python3 setup.py build_ext --inplace   # 当前目录生成 example_ops.cpython-*.so
```

编译输出示例（可以看到 setuptools 实际调用的 gcc 编译与链接命令）：

```bash
running build_ext
building 'example_ops' extension
creating build/temp.linux-x86_64-cpython-310
x86_64-linux-gnu-gcc -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -fPIC -I/usr/include/python3.10 -c example_ops.c -o build/temp.linux-x86_64-cpython-310/example_ops.o
creating build/lib.linux-x86_64-cpython-310
x86_64-linux-gnu-gcc -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -g -fwrapv -O2 build/temp.linux-x86_64-cpython-310/example_ops.o -L/usr/lib/x86_64-linux-gnu -o build/lib.linux-x86_64-cpython-310/example_ops.cpython-310-x86_64-linux-gnu.so
copying build/lib.linux-x86_64-cpython-310/example_ops.cpython-310-x86_64-linux-gnu.so ->
```

```python
import example_ops
example_ops.hello()    # 输出: hello world
```

### 3.3 定位与取舍

C Extension 无第三方依赖、直接操作 CPython API，提供**最底层的控制力**（自定义类型、精细内存管理、GIL 控制），是 NumPy、PyTorch 等高性能库的基石。代价是：参数解析（`PyArg_ParseTuple`）、引用计数（`Py_INCREF/DECREF`）、异常传播都需手写，样板代码多、易出错。

**实践建议**：理解其机制（尤其是 `PyInit_*` 约定，第 8 章的 sgl-kernel 会直接用到），日常开发使用 pybind11。

---

## 第 4 章 pybind11：现代绑定方案

### 4.1 定位

[pybind11](https://pybind11.readthedocs.io/en/stable/) 是一个**纯头文件（header-only）**的 C++ 库，用现代 C++（模板元编程）把第 3 章的样板代码全部自动化：`PYBIND11_MODULE` 宏在编译期展开为完整的 `PyInit_*` 入口、方法表与类型转换代码。

- 要求 C++11 及以上；支持 CPython 3.8+、PyPy3 7.3.17+、GraalPy 24.1+；
- 核心头文件仅约 4K 行，是 Boost.Python 的轻量替代品。

**pybind11 vs Boost.Python：**

| 对比项 | pybind11 | Boost.Python |
|--------|----------|--------------|
| 依赖 | 纯头文件，无外部依赖 | 依赖完整 Boost 库（数十 MB） |
| 编译速度 | 快 | 慢（Boost 头文件庞大） |
| 二进制体积 | 小 | 大 |
| C++ 标准 | C++11 起 | C++03 起 |
| 维护活跃度 | 活跃 | 相对停滞 |
| 适用场景 | 新项目首选 | 与 Boost 重度集成的遗留项目 |

### 4.2 最小示例

```c++
// example.cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;

int add(int i = 10, int j = 30) {
    return i + j + 100;
}

// 定义模块入口（模块名 example 必须与 .so 文件名前缀一致）
PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin";
    m.def("add", &add, "A function which adds two numbers",
          py::arg("i") = 51, py::arg("j") = 10);   // 支持关键字参数与默认值
}
```

编译：

```bash
c++ -O3 -Wall -shared -std=c++11 -fPIC \
    $(python3 -m pybind11 --includes) \
    example.cpp \
    -o example$(python3-config --extension-suffix)
```

<details>
<summary>编译参数说明</summary>

| 参数 | 说明 |
|------|------|
| `-O3` | 最高级别优化 |
| `-Wall` | 启用所有警告 |
| `-shared` | 生成共享库 |
| `-std=c++11` | pybind11 最低 C++ 标准要求 |
| `-fPIC` | 位置无关代码，共享库必需 |
| `$(python3 -m pybind11 --includes)` | 展开为 pybind11 与 Python 头文件的 `-I` 路径 |
| `$(python3-config --extension-suffix)` | 展开为符合规范的后缀，如 `.cpython-310-x86_64-linux-gnu.so` |

</details>

调用：

```python
import example
print(example.add(1, 2))   # 输出: 103 (1+2+100)
```

### 4.3 三个高频错误

**错误 1：缺少头文件路径。** 未指定 `$(python3 -m pybind11 --includes)` 时：

```
fatal error: Python.h: No such file or directory
```

**错误 2：模块名与 `.so` 文件名不一致。** `PYBIND11_MODULE(example, m)` 的模块名必须与 `.so` 前缀一致，否则解释器找不到 `PyInit_example`：

```
ImportError: dynamic module does not define module export function (PyInit_example)
```

验证 `.so` 导出的入口符号：

```bash
nm example.cpython-310-x86_64-linux-gnu.so | grep PyInit_
```

**错误 3：Python 版本不匹配。** 多 Python 环境下，编译头文件版本与运行时解释器版本必须一致。统一使用当前激活的 Python 获取编译参数：

```bash
which python && python --version

$(python -m pybind11 --includes)                                          # 而非 python3
$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
```

版本不匹配（如运行 3.12、`.so` 命名为 3.10）会导致 `ImportError` 或静默加载失败。

---

## 第 5 章 pybind11 调用 CUDA Kernel

将第 2 章（CUDA）与第 4 章（pybind11）合并：绑定层与 kernel 写入同一个 `.cu` 文件，用 nvcc 一次编译。

### 5.1 单文件实现

```c++
// example.cu
#include <stdio.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// CUDA 核函数：在 GPU 上执行
__global__ void demo_kernel() {
    printf("hello, world!!! \n\n");
}

// C++ 封装函数：启动 kernel + 完整错误检查（见第 2.3 节）
void launch_kernel() {
    demo_kernel<<<1, 1>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA launch error: %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA DeviceSynchronize error: %s\n", cudaGetErrorString(err));
    }
}

// pybind11 绑定：只暴露 C++ 封装函数，不直接暴露 kernel
PYBIND11_MODULE(example, m) {
    m.def("run", &launch_kernel, "Launch the CUDA kernel");
}
```

### 5.2 用 nvcc 编译

nvcc 负责编译 CUDA 代码，宿主 C++ 代码交给底层 C++ 编译器处理；`-Xcompiler` 用于将选项透传给宿主编译器：

```bash
# 推荐：动态获取编译参数
nvcc --shared -Xcompiler -fPIC \
    -Xcompiler "$(python3 -m pybind11 --includes)" \
    -o example$(python3-config --extension-suffix) example.cu
```

```python
import example
example.run()    # GPU 上执行 demo_kernel，打印 hello, world!!!
```

### 5.3 工程化改进：分离编译

单文件方案简单，但生产项目通常将 **CUDA 层与绑定层分离编译**（本目录示例工程即采用此结构，见第 9 章）：

```
cuda_hello.cu       ──nvcc──>  libcuda_functions.so   （纯 CUDA，不含 Python 依赖）
pybind_wrapper.cpp  ──g++───>  cuda_hello.so           （纯 C++，链接上面的库）
```

好处：

1. **职责单一**：kernel 代码不感知 Python；绑定代码不感知 CUDA 语法；
2. **编译提速**：修改绑定层无需重新走 nvcc；
3. **复用性**：CUDA 库可同时被 C++ 程序和 Python 扩展链接。

两层之间通过一个普通 C++ 头文件（如 `cuda_hello.h`）作为契约。

---

## 第 6 章 PyTorch C++ 扩展：JIT 与 AOT 编译

当扩展需要操作 `torch.Tensor` 时，应使用 PyTorch 提供的 [`torch.utils.cpp_extension`](https://docs.pytorch.org/docs/stable/cpp_extension.html) 工具链。它在 pybind11 之上自动处理 Torch 头文件路径、ABI 标志、CUDA 库链接等所有编译细节，并提供两种编译模式。

### 6.1 JIT 即时编译：`load()`

[`load()`](https://docs.pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) 在运行时调用 Ninja 完成编译-链接-加载，适合开发迭代阶段：

```python
from torch.utils.cpp_extension import load

example = load(
    name="example",
    sources=["example.cu"],
    verbose=True,          # 输出 Ninja 构建日志
)
example.run()
```

首次调用的构建输出：

```bash
Using /root/.cache/torch_extensions/py310_cu126 as PyTorch extensions root...
Creating extension directory /root/.cache/torch_extensions/py310_cu126/example...
Detected CUDA files, patching ldflags
Emitting ninja build file ...
Building extension module example...
[1/2] /usr/local/cuda/bin/nvcc ... -c example.cu -o example.cuda.o
[2/2] c++ example.cuda.o -shared ... -o example.so
Loading extension module example...

hello,world!!!
```

构建产物缓存于 `~/.cache/torch_extensions/<py版本>_<cuda版本>/<name>/`（可用环境变量 `TORCH_EXTENSIONS_DIR` 自定义）。**源码未变更时直接命中缓存，不重复编译**：

```bash
Using /root/.cache/torch_extensions/py310_cu126 as PyTorch extensions root...
...
ninja: no work to do.
Loading extension module example...
```

### 6.2 AOT 预编译：`setup.py` + `CUDAExtension`

AOT（Ahead-of-Time）方式适合发布为 wheel 包的场景，构建时一次性编译，用户安装即用：

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='example',
    ext_modules=[
        CUDAExtension(name='example', sources=['example.cu']),
    ],
    cmdclass={'build_ext': BuildExtension},
)
```

```bash
python setup.py build_ext --inplace
```

> 完整工程模板参考官方示例仓库：[pytorch/extension-cpp](https://github.com/pytorch/extension-cpp)。

### 6.3 两种模式对比

| | JIT `load()` | AOT `setup.py` |
|--|-------------|----------------|
| 编译时机 | 首次运行时 | 打包/安装时 |
| 增量构建 | Ninja 缓存自动处理 | 手动重新执行 |
| 分发形式 | 随源码分发 | wheel 二进制包 |
| 适用阶段 | 开发调试、教学 | 生产发布 |

---

## 第 7 章 PyTorch 算子注册与 Dispatcher

> 参考：[PyTorch Custom Operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)、[Let's talk about the PyTorch dispatcher](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)

### 7.1 为什么 pybind11 直接暴露还不够

用 pybind11 暴露一个接收 `data_ptr()` 的函数虽然能跑通，但它对 PyTorch 而言是一个"黑盒"——无法参与框架的任何子系统：

| 能力 | 直接 pybind11 | 注册到 Dispatcher |
|------|:---:|:---:|
| 基本调用 | ✅ | ✅ |
| `torch.compile` | ❌（图中断） | ✅ |
| `autograd` 反向传播 | ❌（需手动实现） | ✅（注册 `Autograd` key） |
| `torch.vmap` 批量化 | ❌ | ✅（注册 `Batched` key） |
| `torch.jit.script` | ❌ | ✅ |
| 多后端自动路由 | ❌ | ✅ |
| AOTInductor（无 Python 推理） | ❌ | ✅ |

要获得这些能力，必须通过 Python 的 [`torch.library`](https://docs.pytorch.org/docs/stable/library.html) 或 C++ 的 [`TORCH_LIBRARY`](https://docs.pytorch.org/cppdocs/library.html) 将算子**注册**到 PyTorch。

### 7.2 Dispatcher 的工作原理

调用 `torch.add(a, b)` 时，PyTorch 并非调用某个固定的 C++ 函数，而是经过**分发（Dispatch）机制**，根据输入 Tensor 的属性（设备、是否需要梯度等）动态路由到对应实现：

```
Python: torch.add(a, b)
    └─> C++ Dispatcher（按 dispatch key 路由）
            ├─> CPU 实现          （Tensor 在 CPU 上）
            ├─> CUDA 实现         （Tensor 在 CUDA 上）
            ├─> Autograd 实现     （requires_grad=True）
            └─> 其他后端          （XLA、MPS、量化等）
```

常见的 `DispatchKey`：

| DispatchKey | 含义 |
|-------------|------|
| `CPU` / `CUDA` | 对应设备上的算子实现 |
| `AutogradCPU` / `AutogradCUDA` | 自动微分包装实现 |
| `CompositeImplicitAutograd` | 由基础算子组合实现（梯度自动推导） |
| `Meta` | 只推断形状，不分配内存 |
| `Functionalize` | functorch / `torch.compile` 的函数化变换 |

查看某个算子注册了哪些 key：

```python
import torch
print(torch._C._dispatch_dump("aten::add.Tensor"))
```

### 7.3 两种注册途径

**途径一：Python `torch.library.custom_op()`** —— 将 Python 函数注册为不透明算子（防止 `torch.compile` 追踪进函数内部造成图中断），配合 `torch.library.register_autograd` 可添加自动微分。适合逻辑在 Python 侧的场景。

**途径二：C++ `TORCH_LIBRARY` 系列宏** —— 性能要求高、或需要 AOTInductor（无 Python 运行时）部署的场景。核心是两个宏：

```c++
// 1. 声明算子 schema（一个命名空间只能出现一次 TORCH_LIBRARY）
//    Schema 语法参考：aten/src/ATen/native/README.md
TORCH_LIBRARY(myops, m) {
    // 注意：schema 中的 "float" 对应 C++ 的 double / Python 的 float
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
}

// 2. 为各后端绑定实现（TORCH_LIBRARY_IMPL 可出现多次、可跨文件）
TORCH_LIBRARY_IMPL(myops, CPU, m)  { m.impl("mymuladd", &mymuladd_cpu); }
TORCH_LIBRARY_IMPL(myops, CUDA, m) { m.impl("mymuladd", &mymuladd_cuda); }
```

注册后算子通过 `torch.ops.myops.mymuladd` 访问。

> **版本提示**：PyTorch 2.9+ 引入 [Stable Torch API](https://docs.pytorch.org/cppdocs/stable.html)，提供 `STABLE_TORCH_LIBRARY` / `STABLE_TORCH_LIBRARY_IMPL` / `STABLE_TORCH_LIBRARY_FRAGMENT` 等具有 ABI 稳定性保证的注册接口，扩展无需随 PyTorch 小版本重新编译。

### 7.4 端到端示例：自定义 scalar_mul 算子

以下示例完整展示"CUDA Kernel → C++ 封装 → 注册 → Python 调用"四步。

**第一步：CUDA Kernel 与封装（`myops_cuda.cu`）**

```c++
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel：逐元素乘以标量 c
__global__ void scalar_mul_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * c;
    }
}

// C++ 封装：校验输入、管理输出 Tensor、启动 kernel
torch::Tensor scalar_mul_cuda(torch::Tensor input, double c) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    scalar_mul_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(c),
        n
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "scalar_mul_cuda kernel failed");
    return output;
}
```

**第二步：注册（`myops.cpp`）**

```c++
#include <torch/library.h>
#include <torch/extension.h>

// 声明 CUDA 实现（定义在 myops_cuda.cu）
torch::Tensor scalar_mul_cuda(torch::Tensor input, double c);

// CPU 实现（演示用，直接复用原生运算）
torch::Tensor scalar_mul_cpu(torch::Tensor input, double c) {
    return input * static_cast<float>(c);
}

TORCH_LIBRARY(myops, m) {
    m.def("scalar_mul(Tensor input, float c) -> Tensor");
}
TORCH_LIBRARY_IMPL(myops, CPU, m)  { m.impl("scalar_mul", &scalar_mul_cpu); }
TORCH_LIBRARY_IMPL(myops, CUDA, m) { m.impl("scalar_mul", &scalar_mul_cuda); }
```

**第三步：编译**

```python
# setup.py（AOT）
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='myops',
    ext_modules=[
        CUDAExtension(name='myops', sources=['myops.cpp', 'myops_cuda.cu']),
    ],
    cmdclass={'build_ext': BuildExtension},
)
```

或 JIT：`load(name="myops", sources=["myops.cpp", "myops_cuda.cu"])`。

**第四步：Python 调用（自动分发）**

```python
import torch
import myops    # import 时触发 TORCH_LIBRARY 静态注册

x_cpu = torch.tensor([1.0, 2.0, 3.0])
print(torch.ops.myops.scalar_mul(x_cpu, 3.0))          # → CPU 实现

x_cuda = x_cpu.cuda()
print(torch.ops.myops.scalar_mul(x_cuda, 3.0))         # → CUDA 实现
```

CUDA Tensor 的完整分发链路：

```
torch.ops.myops.scalar_mul(x_cuda, 3.0)
    └─> Dispatcher 检测 x_cuda.device = CUDA
            └─> 路由到 TORCH_LIBRARY_IMPL(myops, CUDA) 注册的实现
                    └─> scalar_mul_cuda(input, c)
                            └─> scalar_mul_kernel<<<blocks, threads>>>(...)
```

> dtype 层面的分发（`AT_DISPATCH_*` 宏）不经过 Dispatcher，在实现函数内部完成，详见本仓库 `01_reduce/` 章节第 14 章。

---

## 第 8 章 生产案例：sgl-kernel

> 参考：[sgl-kernel README](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/README.md)

`sgl-kernel` 是 SGLang 框架的高性能 CUDA kernel 库。它的绑定层设计把前面三章的机制组合到了一起，是学习生产级工程组织的绝佳样本。

### 8.1 编译安装

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang/sgl-kernel
make build -j8
pip install dist/sgl_*.whl --force-reinstall
```

### 8.2 目录组织

| 位置 | 职责 |
|------|------|
| `csrc/` | CUDA kernel 实现（`.cu` / `.cuh`） |
| `*.cc`（如 `common_extension.cc`） | 声明函数接口 + 注册到 PyTorch 分发系统 + Python 模块入口 |
| `python/sgl_kernel/__init__.py` | 对 `.so` 二次封装，提供用户友好的 Python API |

### 8.3 绑定层的双重注册

`common_extension.cc` 的核心逻辑（精简）：

```c++
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>
#include "sgl_kernel_ops.h"

// TORCH_LIBRARY_FRAGMENT：允许多个编译单元向同一命名空间追加注册
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
    m.def(
        "lightning_attention_decode(Tensor q, Tensor k, Tensor v, "
        "Tensor past_kv, Tensor slope, Tensor! output, Tensor! new_kv) -> ()");
    m.impl("lightning_attention_decode", torch::kCUDA, &lightning_attention_decode);
    // ... 更多算子
}

// Python 模块入口（C Extension 机制）
REGISTER_EXTENSION(common_ops)
```

`REGISTER_EXTENSION` 宏展开后正是第 3 章的标准 `PyInit_*` 入口：

```c++
#define REGISTER_EXTENSION(NAME)                                                                       \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                             \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr};  \
    return PyModule_Create(&module);                                                                   \
  }
```

### 8.4 设计要点

这里的双重注册体现了两套机制的分工：

- **`PyModuleDef` 注册**（面向 Python 解释器）：让 `.so` 可以被 `import`。注意方法表为空——模块本身不导出任何函数；
- **`TORCH_LIBRARY_FRAGMENT` 注册**（面向 PyTorch Dispatcher）：`import` 触发 `.so` 加载时，C++ 静态初始化执行注册宏，所有算子挂到 `torch.ops.sgl_kernel.*` 下。

即：`import` 的唯一作用是**触发加载与注册**，实际调用全部走 `torch.ops` 路径，从而获得 Dispatcher 的全部能力（第 7.1 节表格）。

---

## 第 9 章 示例工程：结构、构建与运行

本目录提供一个可直接运行的最小工程，实现第 5.3 节的**分离编译**架构。

### 9.1 项目结构

```
00_how_python_call_cuda_kernel/
├── CMakeLists.txt          # CMake 构建配置（两个编译目标）
├── build.sh                # 一键构建脚本（Release / Debug）
├── src/
│   ├── cuda_hello.h        # C++ 接口声明（CUDA 层与绑定层的契约）
│   ├── cuda_hello.cu       # CUDA kernel + 启动封装（nvcc 编译）
│   └── pybind_wrapper.cpp  # pybind11 绑定层（g++ 编译）
└── python/
    └── test_cuda_hello.py  # Python 测试入口
```

分层与产物：

| 层次 | 文件 | 编译器 | 产物 |
|------|------|--------|------|
| CUDA Kernel 层 | `src/cuda_hello.cu` | nvcc | `libcuda_functions.so` |
| pybind11 绑定层 | `src/pybind_wrapper.cpp` | g++ | `cuda_hello.cpython-*.so`（链接上者） |
| Python 层 | `python/test_cuda_hello.py` | — | — |

示例 kernel 以 `grid(2,2) × block(4,4)`（共 64 线程）启动，每个线程打印自己的 `threadIdx` / `blockIdx` / `gridDim` / `blockDim` 坐标，直观展示 CUDA 线程层次。

### 9.2 快速运行

```bash
git clone https://github.com/TZHelloWorld/how-to-optim-in-cuda.git
cd how-to-optim-in-cuda/00_how_python_call_cuda_kernel

# 安装 pybind11（推荐 pip 方式，与当前 Python 解释器版本严格匹配）
python -m pip install pybind11

# 编译（CMake + make，产物在 build/ 目录）
bash build.sh

# 运行测试
python python/test_cuda_hello.py
```

预期输出（64 行线程坐标，顺序不定）：

```
[test_cuda_hello.py] 成功导入 cuda_hello 模块
[test_cuda_hello.py] 调用 CUDA hello kernel...
Hello, cuda kernel; Thread (0,0,0) in Block (0,0,0), Grid (2,2,1), BlockSize (4,4,1)
Hello, cuda kernel; Thread (1,0,0) in Block (0,0,0), Grid (2,2,1), BlockSize (4,4,1)
...
[test_cuda_hello.py] CUDA kernel 执行完成!
```

### 9.3 Debug 构建

调试 CUDA kernel（第 11 章）需要带调试信息的产物：

```bash
python -m pip install pybind11 ipdb
bash build.sh debug        # 为 device 代码加 -G -g，为 host 代码加 -g
```

---

## 第 10 章 构建系统：CMake 详解

示例工程用 CMake 协调两种编译器的协作。`CMakeLists.txt` 的四个关键设计：

### 10.1 双目标结构

```cmake
cmake_minimum_required(VERSION 3.18)
project(CudaHello CUDA CXX)          # 声明 CUDA + C++ 双语言

# 目标 1：CUDA kernel 共享库（nvcc）
add_library(cuda_functions SHARED src/cuda_hello.cu)
set_target_properties(cuda_functions PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

# 目标 2：pybind11 模块（g++），链接目标 1
pybind11_add_module(cuda_hello src/pybind_wrapper.cpp)
target_link_libraries(cuda_hello PRIVATE cuda_functions)
```

### 10.2 GPU 架构自动检测

通过 `nvidia-smi` 查询当前 GPU 的 compute capability，避免手写 `-arch` 不匹配导致的 `no kernel image is available` 运行时错误：

```cmake
option(USE_NATIVE_CUDA_ARCH "Detect CUDA architecture from nvidia-smi" ON)

execute_process(
    COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv
    OUTPUT_VARIABLE COMPUTE_CAP_OUTPUT
)
# 解析 "8.9" → "89" → CUDA_ARCHITECTURES
```

关闭自动检测手动指定：`cmake -DUSE_NATIVE_CUDA_ARCH=OFF -DCUDA_ARCHITECTURES=89 ..`

### 10.3 pybind11 查找策略

```cmake
find_package(Python COMPONENTS Interpreter Development REQUIRED)
# HINTS 指向 Python site-packages：多个 pybind11 共存时优先 pip 安装的版本，
# 保证与当前解释器严格匹配（对应第 4.3 节错误 3）
find_package(pybind11 REQUIRED HINTS "${Python_SITELIB}")
```

`pybind11_add_module()` 是 [pybind11 官方推荐](https://pybind11.readthedocs.io/en/stable/compiling.html)的目标创建方式，自动处理：Python 版本相关的编译/链接标志、`.so` 命名后缀、Release 模式 LTO 与符号裁剪、默认隐藏符号可见性（避免多模块符号冲突）。

### 10.4 Debug 模式

```cmake
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # -G：device 代码调试信息（同时禁用优化）；-g：host 代码调试符号
    target_compile_options(cuda_functions PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g>)
    add_compile_options(-g)
endif()
```

### 10.5 构建命令

```bash
bash build.sh              # Release（默认）
bash build.sh debug        # Debug

# 等价的手动流程
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

---

## 第 11 章 调试：从 Python 层到 Kernel 层

跨语言调用链的调试需要两套工具的配合：Python 层用 `pdb/ipdb/rpdb`，CUDA Kernel 层用 `cuda-gdb`，两者可以**联合调试**。

### 11.1 Python 层：ipdb 与 rpdb

**ipdb（单机交互调试）：**

```bash
pip install ipdb

python -m ipdb script.py     # 非侵入式：以调试模式启动脚本
```

或侵入式插入断点：

```python
import ipdb; ipdb.set_trace()
```

常用命令：

| 命令 | 说明 | 命令 | 说明 |
|------|------|------|------|
| `c` | 继续到下一断点 | `p <var>` | 打印变量 |
| `n` | 单步（不进函数） | `w` | 查看调用栈 |
| `s` | 单步（进函数） | `u` / `d` | 上/下移栈帧 |
| `l` | 查看附近代码 | `b <line>` | 设断点 |
| `q` | 退出 | `h` | 帮助 |

断点处常用的 Python 内省：

```python
locals() / globals()      # 当前作用域变量
type(a) / dir(a) / help(a)
id(a)                     # 对象地址
import sys
sys.getsizeof(a)          # 内存占用
sys.getrefcount(a)        # 引用计数
```

**rpdb（多进程 / 后台进程调试）：**

`ipdb.set_trace()` 在两类场景下会失败：

1. `nohup` 后台启动——`stdin` 重定向到 `/dev/null`，报 `OSError: Bad file descriptor`；
2. 多进程子进程——无法访问父进程终端，报 `Input is not a terminal`，随即抛出 `bdb.BdbQuit`。

复现示例：

```python
import multiprocessing
import ipdb

def crash_in_subprocess():
    print(f"Process {multiprocessing.current_process().name} is running")
    ipdb.set_trace()      # 子进程中失败 → bdb.BdbQuit
    print("This line won't be reached")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    ps = [multiprocessing.Process(target=crash_in_subprocess, name=f"Worker-{i}")
          for i in range(2)]
    for p in ps: p.start()
    for p in ps: p.join()
```

运行结果：

```bash
Process Worker-1:
...
bdb.BdbQuit
```

[rpdb](https://github.com/tamentis/rpdb)（Remote PDB）改用网络端口提供调试接口：

```python
import rpdb
rpdb.set_trace(addr="0.0.0.0", port=5555)
```

另开终端连接：

```bash
nc -t 0.0.0.0 5555                    # 方式一：netcat
python -m telnetlib 0.0.0.0 5555      # 方式二：telnetlib
```

### 11.2 Kernel 层：cuda-gdb

[cuda-gdb](https://docs.nvidia.com/cuda/cuda-gdb/index.html) 基于 GNU GDB 扩展，支持**同时调试 Host 代码与 Device 代码**。

**编译要求**——必须带调试信息：

```bash
nvcc -g -G -lineinfo -arch=sm_75 matrix_add.cu -o matrix_add
```

| 标志 | 说明 |
|------|------|
| `-g` | Host 代码调试信息 |
| `-G` | Device 代码调试信息，同时禁用优化（增大体积、变慢） |
| `-lineinfo` | Device 代码行号信息（不禁用优化，适合性能分析场景） |
| `-arch=sm_XX` | 显式指定目标架构，避免 `no kernel image available` |

> `-G` 与 `-lineinfo` 同时给出时，`-G` 优先，`-lineinfo` 被忽略。

**常用命令速查：**

```bash
# ===== 基础设置 =====
(cuda-gdb) set cuda launch_blocking on   # kernel launch 变为阻塞，便于调试
(cuda-gdb) help set cuda                 # 查看所有 CUDA 相关设置

# ===== 断点管理 =====
(cuda-gdb) break kernel_name                        # kernel 入口断点
(cuda-gdb) break file.cu:20                         # 文件:行号断点
(cuda-gdb) break kernel_name if threadIdx.x == 0    # 条件断点
(cuda-gdb) info breakpoints                         # 查看断点
(cuda-gdb) delete 1 / disable 2 / enable 2

# 断点触发时自动执行命令
(cuda-gdb) break my_kernel
(cuda-gdb) commands
> print threadIdx.x
> continue
> end

# ===== 执行控制 =====
(cuda-gdb) run / continue / next / step / finish / until [line]

# ===== 状态查看 =====
(cuda-gdb) backtrace          # 调用栈
(cuda-gdb) frame N            # 切换栈帧
(cuda-gdb) info registers     # 寄存器

# ===== CUDA 特有命令 =====
(cuda-gdb) info cuda devices / kernels / blocks / threads
(cuda-gdb) info cuda sms / warps / lanes
(cuda-gdb) info cuda launch trace          # kernel 启动跟踪

# ===== 线程焦点切换（调试以 warp 为单位）=====
(cuda-gdb) cuda kernel 0 grid 1 block (0,0,0) thread (0,0,0)   # 按逻辑坐标
(cuda-gdb) cuda kernel 0 sm 2 warp 43 lane 16                  # 按硬件坐标

# ===== 查看变量 =====
(cuda-gdb) print threadIdx / blockIdx / variable_name
```

**典型调试会话**（以矩阵加法为例）：

```bash
cuda-gdb ./matrix_add

(cuda-gdb) set cuda launch_blocking on
On the next run, the CUDA kernel launches will be blocking.

(cuda-gdb) break matrixAdd
Breakpoint 1 at 0xb2f1: file matrix_add.cu, line 5.

(cuda-gdb) run
Running matrix addition for 1024x1024 matrices
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), ...]
CUDA thread hit Breakpoint 1, matrixAdd<<<(64,64,1),(16,16,1)>>> (...) at matrix_add.cu:7
7       int row = blockIdx.y * blockDim.y + threadIdx.y;

(cuda-gdb) info cuda launch trace
  Lvl  Kernel Dev Grid  Status    GridDim   BlockDim   Invocation
* #  0   0    0   1    Active   (64,64,1) (16,16,1)  matrixAdd(...)

(cuda-gdb) print blockIdx
$1 = {x = 0, y = 0, z = 0}
(cuda-gdb) print threadIdx
$2 = {x = 0, y = 0, z = 0}
(cuda-gdb) next
```

### 11.3 联合调试：cuda-gdb + ipdb

同时调试 Python 层与 Kernel 层的完整流程（以本目录示例工程为例）：

```bash
# 1. Debug 构建（生成带 -g -G 的 .so）
bash build.sh debug

# 2. 以 python 解释器为入口启动 cuda-gdb
cuda-gdb python --quiet
Reading symbols from python...
(No debugging symbols found in python)
```

进入交互界面后：

```bash
# 3. 启用同步 launch，便于断点定位
(cuda-gdb) set cuda launch_blocking on
On the next run, the CUDA kernel launches will be blocking.

# 4. 为 kernel 设 pending 断点（.so 尚未加载，选择 y 挂起等待）
(cuda-gdb) break cuda_hello_kernel
Function "cuda_hello_kernel" not defined.
Make breakpoint pending on future shared library load? (y or [n]) y
Breakpoint 1 (cuda_hello_kernel) pending.

# 5. 以 ipdb 模式运行 Python 脚本
(cuda-gdb) run -m ipdb python/test_cuda_hello.py
Starting program: /usr/bin/python -m ipdb python/test_cuda_hello.py
...
> /xxx/how-to-optim-in-cuda/00_how_python_call_cuda_kernel/python/test_cuda_hello.py(1)<module>()
----> 1 """test_cuda_hello.py — Python 测试入口。
      2

# 6. 在 ipdb 中调试 Python 层（可设 Python 断点、单步执行）
#    第 39 行即 cuda_hello.hello() 调用处
ipdb> b 39
Breakpoint 1 at /xxx/python/test_cuda_hello.py:39
ipdb> c

# 7. 执行到 kernel launch 时，cuda-gdb 自动接管，切入 GPU 上下文
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), ...]
CUDA thread hit Breakpoint 1, cuda_hello_kernel<<<(2,2,1),(4,4,1)>>> () at cuda_hello.cu:...
(cuda-gdb)      # 此处可用全部 cuda-gdb 命令调试 kernel
```

调试链路总结：

```
cuda-gdb 启动 python 解释器
    └─> ipdb 调试 Python 层（Python 断点、单步）
            └─> 执行触发 CUDA kernel launch
                    └─> cuda-gdb 命中 pending 断点，接管 GPU 上下文
```

### 11.4 VSCode 集成

将上述流程脚本化：`tasks.json` 定义构建任务，`launch.json` 定义调试配置。

**`.vscode/tasks.json`**（调试前自动触发 Debug 构建）：

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build-debug",
            "type": "shell",
            "command": "bash ${workspaceFolder}/00_how_python_call_cuda_kernel/build.sh debug",
            "group": { "kind": "build", "isDefault": true },
            "presentation": { "reveal": "always", "panel": "shared" },
            "problemMatcher": []
        }
    ]
}
```

**`.vscode/launch.json`**（两种调试模式）：

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: test_cuda_hello",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/00_how_python_call_cuda_kernel/python/test_cuda_hello.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "preLaunchTask": "build-debug"
        },
        {
            "name": "CUDA-GDB: test_cuda_hello",
            "type": "cuda-gdb",
            "request": "launch",
            "program": "/usr/bin/python",
            "args": ["-m", "ipdb",
                     "${workspaceFolder}/00_how_python_call_cuda_kernel/python/test_cuda_hello.py"],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}/00_how_python_call_cuda_kernel",
            "preLaunchTask": "build-debug",
            "cuda": { "breakOnLaunch": true }
        }
    ]
}
```

> `cuda-gdb` 类型的配置需安装 NVIDIA 官方插件 [Nsight Visual Studio Code Edition](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition)；未安装时可用模式一（debugpy）调试 Python 层，Kernel 层调试在终端手动执行 cuda-gdb（第 11.3 节）。

---

## 第 12 章 总结与参考资料

### 12.1 方案选型速查

| 需求 | 推荐方案 | 关键章节 |
|------|---------|---------|
| 理解底层加载机制 | C Extension（`PyInit_*` 三要素） | 第 3 章 |
| 通用 Python/C++ 绑定 | pybind11 | 第 4~5 章 |
| 开发迭代 PyTorch 扩展 | `cpp_extension.load()` JIT | 第 6.1 节 |
| 发布 PyTorch 扩展 wheel | `setup.py` + `CUDAExtension` AOT | 第 6.2 节 |
| 与 compile/autograd/vmap 集成 | `TORCH_LIBRARY` 注册到 Dispatcher | 第 7 章 |
| 生产级多算子 kernel 库 | `TORCH_LIBRARY_FRAGMENT` + `PyInit_*` 双注册 | 第 8 章 |

### 12.2 高频错误速查

| 错误信息 | 根因 | 解决 |
|---------|------|------|
| `undefined reference to 'main'` | 独立程序编译却无 main 入口 | 补 main 或改编译为共享库 |
| `fatal error: Python.h: No such file` | 缺少 Python/pybind11 头文件路径 | `$(python -m pybind11 --includes)` |
| `PyInit_<name>` 未定义 | 模块名与 `.so` 前缀不一致 / 无绑定入口 | 对齐名称；`nm .so \| grep PyInit_` 验证 |
| `no kernel image is available` | `-arch` 与设备不匹配 | 自动检测架构（第 10.2 节）+ 错误检查宏 |
| kernel 静默无输出 | 未做 CUDA 错误检查 | `CUDA_CHECK(cudaGetLastError())`（第 2.3 节） |
| `bdb.BdbQuit`（子进程断点） | 无终端环境使用 ipdb | 改用 rpdb（第 11.1 节） |

### 12.3 参考资料

| 主题 | 链接 |
|------|------|
| pybind11 官方文档 | https://pybind11.readthedocs.io/en/stable/ |
| pybind11 中文翻译 | https://github.com/charlotteLive/pybind11-Chinese-docs |
| pybind11 编译指南（CMake） | https://pybind11.readthedocs.io/en/stable/compiling.html |
| CPython C Extension 文档 | https://docs.python.org/3/extending/extending.html |
| PyTorch Custom Operators | https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html |
| torch.utils.cpp_extension | https://docs.pytorch.org/docs/stable/cpp_extension.html |
| torch.library（Python 注册） | https://docs.pytorch.org/docs/stable/library.html |
| TORCH_LIBRARY（C++ 注册） | https://docs.pytorch.org/cppdocs/library.html |
| Stable Torch API（2.9+） | https://docs.pytorch.org/cppdocs/stable.html |
| 算子 Schema 语法 | https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func |
| PyTorch Dispatcher 设计（ezyang） | https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/ |
| pytorch/extension-cpp 示例 | https://github.com/pytorch/extension-cpp |
| sgl-kernel | https://github.com/sgl-project/sglang/tree/main/sgl-kernel |
| CUDA-GDB User Manual | https://docs.nvidia.com/cuda/cuda-gdb/index.html |
| Nsight VSCode Edition | https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition |
| rpdb | https://github.com/tamentis/rpdb |
| Boost.Python | https://www.boost.org/doc/libs/release/libs/python/doc/ |

### 12.4 知识结构总图

```
                    ┌──────────────────────────────────────────────┐
                    │        Python 调用 CUDA Kernel 技术栈         │
                    └──────────────────────┬───────────────────────┘
                                           │
        ┌──────────────────────────────────┼──────────────────────────────────┐
        │                                  │                                  │
┌───────▼─────────┐              ┌─────────▼──────────┐             ┌─────────▼─────────┐
│  绑定机制        │              │  构建系统           │             │  调试工具          │
│                 │              │                    │             │                   │
│ C Extension     │              │ 手动 nvcc/g++      │             │ ipdb（单机）       │
│  └ PyInit_* 三件套│             │ CMake              │             │ rpdb（远程/多进程）│
│ pybind11        │              │  └ pybind11_add_   │             │ cuda-gdb          │
│  └ PYBIND11_MODULE│            │    module          │             │  └ pending 断点    │
│ PyTorch 扩展     │              │ setuptools         │             │  └ 线程焦点切换    │
│  └ cpp_extension │              │  └ CUDAExtension   │             │ 联合调试           │
│  └ TORCH_LIBRARY │              │ Ninja（JIT load）  │             │  └ cuda-gdb+ipdb  │
└───────┬─────────┘              └────────────────────┘             └───────────────────┘
        │
        │    ┌────────────────────────────────────────────┐
        └────▶  PyTorch Dispatcher                        │
             │  按 DispatchKey（CPU/CUDA/Autograd...）路由 │
             │  TORCH_LIBRARY / _IMPL / _FRAGMENT         │
             │  → compile / autograd / vmap 全兼容        │
             └────────────────────────────────────────────┘
```
