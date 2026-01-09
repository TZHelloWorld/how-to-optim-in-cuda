# 运行

如果只是单纯的运行，则参考：

```bash
git clone https://github.com/TZHelloWorld/how-to-optim-in-cuda.git
cd how-to-optim-in-cuda/00_how_python_call_cuda_kernel

# 确保是通过运行 python 安装的 pybind11
python -m pip install pybind11

# 编译 c++ 和 cuda kernel
bash build.sh

# 执行
python python/test_cuda_hello.py
```

调试运行：

```bash
git clone https://github.com/TZHelloWorld/how-to-optim-in-cuda.git
cd how-to-optim-in-cuda/00_how_python_call_cuda_kernel

# 确保是通过运行 python 安装的 pybind11
python -m pip install pybind11 ipdb

# 添加调试信息到编译文件中
bash build.sh debug

# 测试执行情况：
python python/test_cuda_hello.py

# 使用 cuda-gdb + ipdb 进行调试
cuda-gdb python --quiet
```

此时需要交互操作：

```bash
# 给 kernel 打断点
(cuda-gdb) break cuda_hello_kernel
Function "cuda_hello_kernel" not defined.
Make breakpoint pending on future shared library load? (y or [n]) y
Breakpoint 1 (cuda_hello_kernel) pending.

(cuda-gdb) run -m ipdb python/test_cuda_hello.py 
Starting program: /usr/bin/python -m ipdb python/test_cuda_hello.py
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
<frozen runpy>:128: RuntimeWarning: 'ipdb.__main__' found in sys.modules after import of package 'ipdb', but prior to execution of 'ipdb.__main__'; this may result in unpredictable behaviour
[New Thread 0x7ffff5838640 (LWP 4291)]
> /xxx/xxx/how-to-optim-in-cuda/00_how_python_call_cuda_kernel/python/test_cuda_hello.py(1)<module>()
----> 1 """
      2 测试CUDA Hello World模块
      3 """

[New Thread 0x7ffff4ce3640 (LWP 4292)]
ipdb>
```

# 说明
其实可以理解为一种 `python ---> pybind11 ---> C/C++ ---> CUDA Kernel` 的调用逻辑。

## 关于 `C/C++ ---> CUDA Kernel` 的调用

创建一个 `example.cu` 文件，内容为：

```c++ 
#include <stdio.h>

// 核函数定义，定义每个 thread 干什么
__global__ void demo_kernel(){
    printf("hello,world!!!");
}

int main(int argc, char *argv[]){
    demo_kernel<<<1,1>>>(); // 调用核函数启动1个线程块，这个线程块包含1个线程。
    cudaDeviceSynchronize();
    return 0;
}
```

然后使用 `nvcc` 工具进行编译并执行即可：

```bash
nvcc example.cu -o example && ./example
```

### 注意

上述这种调用必须符合 `C/C++` 编译规则(需要一个入口函数)，否则会提示错误：

```bash
// /usr/bin/ld: /usr/lib/gcc/x86_64-linux-gnu/11/../../../x86_64-linux-gnu/Scrt1.o: in function `_start':
// (.text+0x1b): undefined reference to `main'
// collect2: error: ld returned 1 exit status
```

## 关于 `python ---> pybind11 ---> C/C++` 的调用

创建一个 `example.cpp` 的文件，内容为：

```c++
#include <pybind11/pybind11.h>
namespace py = pybind11;

// 定义 c++ 代码逻辑
int add(int i = 10, int j = 30) {
    return i + j + 100; 
}

// 定义绑定 python 的逻辑
PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", &add, "A function which adds two numbers", py::arg("i") = 51,py::arg("j") = 10);
}
```
然后使用**编译工具**（很多，这里以`c++`为例）进行编译：

```bash
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)
```

<details>

<summary></summary>
说明：

- `--shared` 表示要求生成 **共享库**（`Shared Library`）
- `-fPIC` 表示要求生成 **位置无关代码**（`Position Independent Code`），这是共享库的必要条件。
- `python3 -m pybind11 --includes` :查看对应的 `python` 依赖库头文件地址, 内容为 `-I/usr/xxx/xxx` 路径，用于指定头文件的搜索路径（`Include Path`），就是告诉编译器，需要在该路径下查找头文件。
- `example$(python3-config --extension-suffix)`: 如果使用 `pybind11` 作为 `python` 和 `c++` 的胶水，则编译的 `.so` 需要满足 `[module_name].cpythono-[version]-x86_64-linux-gnu.so` 的命名规则。才会被 `python` 程序导入。
</details>
<p>

会在当前目录生成一个类似 `example.cpython-310-x86_64-linux-gnu.so`的文件，然后在该同目录下编写 `python` 调用测试程序执行即可，内容为：

```python
import example
print(example.add(1,2)) # ===> 输出结果为 103 (1+2+100)
```

### 注意

1. 需显示设置 `include` 依赖库头文件地址（如 `$(python3 -m pybind11 --includes)`），否则可能会出现错误：
    ```bash
    In file included from /usr/include/pybind11/pytypes.h:12,
                    from /usr/include/pybind11/cast.h:13,
                    from /usr/include/pybind11/attr.h:13,
                    from /usr/include/pybind11/pybind11.h:13,
                    from example.cpp:2:
    /usr/include/pybind11/detail/common.h:215:10: fatal error: Python.h: No such file or directory
    215 | #include <Python.h>
        |          ^~~~~~~~~~
    compilation terminated.
    ```
2.  为了能够让 `python` 能够导入生成的 `.so` 文件，需要确保 `pybind11` 配置的 `module` 名字和 `.so` 文件的模块名字一样。如错误示例：
    ```bash
    >>> c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o demo$(python3-config --extension-suffix)

    # 查看内容 demo.xxx.so 名字和 PyInit_example 中不一致
    >>> nm demo.cpython-310-x86_64-linux-gnu.so | grep PyInit_example
    ```
    编译的时候没问题，但是在 `python` 导入包的时候提示：
    ```bash
    ImportError: dynamic module does not define module export function (PyInit_xxx)
    ```

3. 类似于 2 ，因为 `Linux` 系统中，存在多个 `python` 版本，可能也会出现命名错误。如：`python` 使用的 `3.12` 版本，但是生成的 `.so` 文件命名为 `3.10` 版本。
   
    ```bash
    # 查看 python 路径
    >>> which python
        /usr/bin/python

    # 可以发现 python 用的是 3.12 ，但是 python3-config 用的是 3.10
    >>> ll /usr/bin/python*
        lrwxrwxrwx 1 root root      19 Nov  6 16:51 /usr/bin/python -> /usr/bin/python3.12*
        lrwxrwxrwx 1 root root      25 Nov  6 16:50 /usr/bin/python3 -> /etc/alternatives/python3*
        -rwxr-xr-x 1 root root 5937768 Aug 15 14:32 /usr/bin/python3.10*
        lrwxrwxrwx 1 root root      34 Aug 15 14:32 /usr/bin/python3.10-config -> x86_64-linux-gnu-python3.10-config*
        -rwxr-xr-x 1 root root 7914288 Oct 10 08:52 /usr/bin/python3.12*
        lrwxrwxrwx 1 root root      34 Oct 10 08:52 /usr/bin/python3.12-config -> x86_64-linux-gnu-python3.12-config*
        lrwxrwxrwx 1 root root      17 Aug  8  2024 /usr/bin/python3-config -> python3.10-config*

    # 优先使用的 3.12 版本的编译头文件
    >>> echo $(python -m pybind11 --includes)
        -I/usr/include/python3.12 -I/usr/local/lib/python3.12/dist-packages/pybind11/include

    # 生成的却是 3.10 的 .so 文件
    >>> echo example$(python3-config --extension-suffix)
        example.cpython-310-x86_64-linux-gnu.so
    ```



## 关于 `python ---> pybind11 ---> C/C++ ---> CUDA Kernel` 的调用
创建一个 `example.cu` 文件，文件内容为：
```c++
#include <stdio.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// 核函数定义，定义每个 thread 干什么
__global__ void demo_kernel(){
    printf("hello,world!!! \n\n");
}

// 封装 CUDA 调用的函数
void launch_kernel() {
    cudaError_t err; 
    
    demo_kernel<<<1,1>>>();
    err = cudaGetLastError(); // 执行kernel launch 之后判断
    if (err != cudaSuccess) {
        printf("CUDA launch error: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA DeviceSynchronize error: %s\n", cudaGetErrorString(err));
    }
}

// pybind11 模块定义
PYBIND11_MODULE(example, m) {
    m.def("run", &launch_kernel, "Launch the CUDA kernel");
}
```

然后使用 `nvcc` 工具进行编译：
```bash
nvcc --shared -Xcompiler -fPIC -Xcompiler -I/usr/include/python3.10 -I/usr/local/lib/python3.10/dist-packages/pybind11/include -o example.cpython-310-x86_64-linux-gnu.so example.cu

# 或者根据环境动态设置编译环境
nvcc --shared -Xcompiler -fPIC -Xcompiler $(python3 -m pybind11 --includes)  -o example$(python3-config --extension-suffix) example.cu
```

然后在该同目录下编写 `python` 调用测试程序执行即可，内容为：

```python
import example
print(example.run())
```

### 注意
上述中有个 `CUDA Launch` 检错逻辑。如果不添加这些功能，可能会导致编译通过，运行通过，但是 `cuda kernel` 没执行。如：上述代码删除掉检测代码,进行编译：
```bash
>>> nvcc -arch=sm_90 --shared -Xcompiler -fPIC -Xcompiler $(python3 -m pybind11 --includes)  -o example$(python3-config --extension-suffix) example.cu

>>> python -m IPython
Python 3.10.12 (main, Aug 15 2025, 14:32:43) [GCC 11.4.0]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.38.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import example

In [2]: example.run() # ===> 不检错，无任何输出结果

In [2]: example.run() # ===> 如果添加检错代码，则会提示：
CUDA launch error: no kernel image is available for execution on the device
```

因为检错逻辑很常见，可以将检错逻辑定义成一个宏：
```c++
// 宏定义，用于检测 运行时 错误的。
#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)
```
检测代码有：
```c++
// 核函数定义，定义每个 thread 干什么
__global__ void demo_kernel(){
    printf("hello,world!!! \n\n");
}

void launch_kernel() {

    demo_kernel<<<1,1>>>();

    // check error state after kernel launch
    CUDA_CHECK(cudaGetLastError());
    // wait for kernel execution to complete
    // The CUDA_CHECK will report errors that occurred during execution of the kernel
    CUDA_CHECK(cudaDeviceSynchronize());
}
```


# vscode 调试

代码调试主要是 `python` 代码调试 加 `c++` 代码调试(`cuda-gdb`)。然后基于 `vscode` 将上述过程脚本化（通过配置 `.vscode` 目录下的 `launch.json` 和 `tasks.json` 实现）

## 使用 cuda-gdb

编写一个简单的加法 `cuda kernel` 算子，文件 `matrix_add.cu` 内容为：

```c++
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for matrix addition
__global__ void matrixAdd(int* A, int* B, int* C, int width) {
    // Calculate global thread index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread index is within matrix bounds
    if (row < width && col < width) {
        int index = row * width + col;
        C[index] = A[index] + B[index];
    }
}

void runMatrixAddition(int N) {
    // Initialize matrices on host
    int* h_A = new int[N*N];
    int* h_B = new int[N*N];
    int* h_C = new int[N*N];

    // Initialize input matrices with sample values
    for (int i = 0; i < N*N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Pointers for device matrices
    int* d_A, *d_B, *d_C;

    // Allocate memory on device
    cudaMalloc((void**)&d_A, N*N * sizeof(int));
    cudaMalloc((void**)&d_B, N*N * sizeof(int));
    cudaMalloc((void**)&d_C, N*N * sizeof(int));

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, N*N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);  // 256 threads per block
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                      (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }

    // Copy result back to host
    cudaMemcpy(h_C, d_C, N*N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print first few elements as verification
    std::cout << "First 5x5 elements of result matrix C:" << std::endl;
    for (int row = 0; row < 5 && row < N; row++) {
        for (int col = 0; col < 5 && col < N; col++) {
            std::cout << h_C[row*N + col] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

int main() {
    int N = 1024;  // Matrix size (can be adjusted)
    std::cout << "Running matrix addition for " << N << "x" << N << " matrices" << std::endl;
    runMatrixAddition(N);
    return 0;
}
```

编译：

```bash
>>> nvcc -g -G -lineinfo -arch=sm_75 matrix_add.cu -o matrix_add
nvcc warning : '--device-debug (-G)' overrides '--generate-line-info (-lineinfo)'
ptxas warning : Conflicting options --device-debug and --generate-line-info specified, ignoring --generate-line-info option
```

说明：
 - `-g`：生成主机(`host`)代码调试信息
 - `-G`：生成设备(`device`)代码调试信息（禁用优化）
 - `-lineinfo`: 为设备代码生成 **行号信息**（`Line Number Information`）, 对于后续精确定位很有帮助
 - `-arch=sm_75`：指定目标 `GPU` 的 **计算能力**（`Compute Capability`），尽量显示指定，否则容易在 `cuda-gdb` 调试运行时报错



启动 `cuda-gdb` 调试：

```bash
>>> cuda-gdb ./matrix_add

NVIDIA (R) cuda-gdb 12.9
Portions Copyright (C) 2007-2024 NVIDIA Corporation
Based on GNU gdb 13.2
Copyright (C) 2023 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This CUDA-GDB was configured as "x86_64-pc-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<https://forums.developer.nvidia.com/c/developer-tools/cuda-developer-tools/cuda-gdb>.
Find the CUDA-GDB manual and other documentation resources online at:
    <https://docs.nvidia.com/cuda/cuda-gdb/index.html>.

For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from ./matrix_add...
(cuda-gdb) 
```

然后就可以开始调试了，具体的操作可以参考 [CUDA-GDB 官网](https://docs.nvidia.com/cuda/cuda-gdb/index.html#compiling-the-application)，常用的操作如下：

```bash
# 查看设置
(cuda-gdb) help set cuda

# 同步内核启动( cpu 和 gpu 同步)，便于调试
(cuda-gdb) set cuda launch_blocking on

# 设置断点
(cuda-gdb) break kernel_name                                # 在内核函数设置断点
(cuda-gdb) break file.cu:20                                 # 在特定文件行号设置断点
(cuda-gdb) break kernel_name if threadIdx.x == 0 && ..      # 设置条件断点

(cuda-gdb) info breakpoints         # 查看所有断点
(cuda-gdb) info b                   # 查看所有断点

(cuda-gdb) delete 1                 # 删除1号断点
(cuda-gdb) delete                   # 删除所有断点
(cuda-gdb) disable 2 / enable 2     # 禁用/启用断点


# 运行程序 & 控制
(cuda-gdb) run                      # 运行程序
(cuda-gdb) continue                 # 继续执行
(cuda-gdb) next                     # 单步执行（不进入函数）
(cuda-gdb) step                     # 单步执行（进入函数）
(cuda-gdb) finish                   # 跳出当前函数
(cuda-gdb) until [line_number]      # 执行直到指定行

# 进入设备(GPU)上下文
(cuda-gdb) cuda kernel 0


# 查看调用栈（调用栈分析）
(cuda-gdb) backtrace                # 或者 bt, 查看调用栈
(cuda-gdb) where                    # 显示当前位置
(cuda-gdb) frame N                  # 切换到第N层栈帧
(cuda-gdb) info registers           # 查看寄存器状态
(cuda-gdb) info cuda devices        # 查看设备
(cuda-gdb) info cuda kernels        # 查看 kernel
(cuda-gdb) info cuda blocks
(cuda-gdb) info cuda threads        # 断点触发后，查看线程信息
(cuda-gdb) info cuda sms
(cuda-gdb) info cuda warps
(cuda-gdb) info cuda lanes
(cuda-gdb) info cuda launch trace   # 显示当前 kernel 的内核启动跟踪信息


# 断点触发后，可以切换到特定的线程进行调试
# 注意，线程调试是以 线程束 为单位进行的
(cuda-gdb) set cuda thread (0,0,0)  # 聚焦(0,0,0)线程
(cuda-gdb) print value              # 查看该线程的变量值


# 可为断点设置触发时自动执行的命令:
(cuda-gdb) break my_kernel
(cuda-gdb) commands
> print threadIdx.x
> continue
> end
```

基于上述内容，开始调试我们生成的 `./matrix_add`:

```bash
>>> cuda-gdb ./matrix_add

# 同步内核启动( cpu 和 gpu 同步)，便于调试
(cuda-gdb) set cuda launch_blocking on
On the next run, the CUDA kernel launches will be blocking.

# 添加断点
(cuda-gdb) break matrixAdd
Breakpoint 1 at 0xb2f1: file /xxx/xxx/matrix_add.cu, line 5.

# 查看断点
(cuda-gdb) info breakpoints 
Num     Type           Disp Enb Address            What
1       breakpoint     keep y   <MULTIPLE>         
1.1                         n   0x000000000000b2f1 in matrixAdd(int*, int*, int*, int) at /xxx/xxx/matrix_add.cu:5

# 执行 cuda 算子 
(cuda-gdb) run
Starting program: /xxx/xxx/matrix_add 
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
Running matrix addition for 1024x1024 matrices
[New Thread 0x7ffff15bf000 (LWP 2268)]
[New Thread 0x7fffebfff000 (LWP 2269)]
[Detaching after fork from child process 2270]
[New Thread 0x7fffd855a000 (LWP 2278)]
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0]

CUDA thread hit Breakpoint 1.2, matrixAdd<<<(64,64,1),(16,16,1)>>> (A=0x7fffb3600000, B=0x7fffb3a00000, C=0x7fffc9200000, width=1024) at matrix_add.cu:7
7	    int row = blockIdx.y * blockDim.y + threadIdx.y;


# 查看当前 kernel 的配置信息
(cuda-gdb) info cuda launch trace
    Lvl  Kernel Dev Grid  Status     GridDim  BlockDim   Invocation                                                                  
* #   0     0    0   1    Active    (64,64,1) (16,16,1)  matrixAdd(A=0x7fffb3600000, B=0x7fffb3a00000, C=0x7fffc9200000, width=1024)

# 从 cpu上下文 到 GPU 上下文, 此时其实还在cpu上，需要切换， 先查看下 当前的坐标：
(cuda-gdb) cuda kernel grid sm warp lane device block thread
kernel 0, grid 1, block (44,1,0), thread (3,4,0), device 0, sm 1, warp 2, lane 3

# 切换到其他坐标（好像得分开切换，否则会提示 Invalid coordinates requested. CUDA focus unchanged.）
# 先查看下能切换到哪些：
# 直接一下切换，（推荐分开切换，否则很容易提示 Invalid coordinates requested. CUDA focus unchanged. ）
(cuda-gdb) cuda kernel 0 grid 1 block (44,1,0) thread (3,4,0) device 0 sm 1 warp 2 lane 3
[Switching focus to CUDA kernel 0, grid 1, block (44,1,0), thread (3,4,0), device 0, sm 1, warp 2, lane 3]
7	    int row = blockIdx.y * blockDim.y + threadIdx.y;


# 先切换 grid block thread
(cuda-gdb) cuda device 0 grid 1 block (49,6,0) thread (15,5,0)
[Switching focus to CUDA kernel 0, grid 1, block (49,6,0), thread (15,5,0), device 0, sm 0, warp 43, lane 31]
0x00007fffcf26a8e0	5	__global__ void matrixAdd(int* A, int* B, int* C, int width) {

# 然后切换其中的 sm warp lane 等等，不确定是否可以一起切换。
(cuda-gdb) cuda kernel 0 sm 2 warp 43 lane 16
[Switching focus to CUDA kernel 0, grid 1, block (51,6,0), thread (0,5,0), device 0, sm 2, warp 43, lane 16]
7	    int row = blockIdx.y * blockDim.y + threadIdx.y;

# 查看当前的 block
(cuda-gdb) print blockIdx
$7 = {x = 44, y = 1, z = 0}

# 查看当前的 thread
(cuda-gdb) print threadIdx
$8 = {x = 3, y = 4, z = 0}

# 执行下一步
(cuda-gdb) next
```

## 使用 pdb/ipdb/rpdb/...

对于 `python` 的调试器，太多了。单机版本可以考虑使用 `ipdb`:

```bash
pip install ipdb
```

使用方案可以嵌入到自己代码中：

```python
import ipdb
ipdb.set_trace()
```

完整的 `matrix_add.py` 文件内容为：

```python
if __name__ == '__main__':
    # 定义两个3x3矩阵
    matrix_a = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]

    matrix_b = [[9, 8, 7],
                [6, 5, 4],
                [3, 2, 1]]
    
    # import ipdb
    # ipdb.set_trace()

    # 初始化结果矩阵
    result = [[0 for _ in range(len(matrix_a[0]))] for _ in range(len(matrix_a))]

    # 执行矩阵加法
    for i in range(len(matrix_a)):  # 遍历行
        for j in range(len(matrix_a[0])):  # 遍历列
            result[i][j] = matrix_a[i][j] + matrix_b[i][j]

    # 打印结果
    print("矩阵加法结果：")
    for row in result:
        print(row)
```

又或者启动程序的时候设置（不侵入代码）：

```bash
python -m ipdb matrix_add.py
```

那么在断点处就会出现交互窗口，可通过命令去调试代码：

- **c**：继续执行代码，直到遇到下一个断点或程序结束。
- **n**：单步执行下一行代码（不会进入函数内部）。
- **s**：单步进入下一行代码（如果有函数调用，则进入函数内部）。
- **q**：退出调试器并终止程序的执行。
- **l**：查看当前位置附近的代码。
- **p**：打印变量的值，例如p variable_name。
- **h**：查看帮助信息，例如h command_name。
- **w**：查看当前的调用栈。
- **u**：向上移动一层调用栈。
- **d**：向下移动一层调用栈。
- **exit**:

除此之外，当到达断点的时候可以查看：

```python

locals() 
globals()

# 对于变量a
type(a)
dir(a)

# 可能还有
help(a)  # 适用于模块、类、函数等

# 查看内存
id(a)
sys.getsizeof(a)
sys.getrefcount(a)
```


### why rpdb?

1. `nohup` 启动抛出 `OSError: Bad file descriptor` 错误, 错误的根本原因是 交互式调试器无法在后台进程（无终端）中运行 ，- 当调用 `ipdb.set_trace()` 时，调试器会尝试从当前进程的标准输入（文件描述符 0）读取用户指令。- 但通过 `nohup` 在后台运行时，标准输入会被关闭（`stdin` 被重定向到 `/dev/null`），导致调试器无法读取输入，抛出 `Bad file descriptor` 错误。
2. 错误提示 `Input is not a terminal (fd=44)`。表明程序运行在一个没有关联到交互式终端的环境（如后台进程、多线程/多进程调度、服务端框架等）。`ipdb` 需要直接与终端交互以接收用户的调试指令，但在这些环境中，标准输入（`stdin`）被重定向或不可用，导致调试器无法启动。当 `ipdb.set_trace()` 尝试启动调试器时，由于无法获取终端输入，调试器会立即退出并抛出 `bdb.BdbQuit` 异常，导致程序终止。

这里以多进程的方式复现上述内容, 编辑文件 `test_ipdb_in_subprocess.py`, 内容为：

```python
import multiprocessing
import ipdb

def crash_in_subprocess():
    print(f"Process {multiprocessing.current_process().name} is running")
    ipdb.set_trace()  # 在子进程中触发调试断点
    print("This line won't be reached")

if __name__ == "__main__":
    # 使用 spawn 方式启动进程（Linux/Windows 通用）
    multiprocessing.set_start_method('spawn')
    
    # 创建两个子进程
    processes = [multiprocessing.Process(target=crash_in_subprocess, name=f"Worker-{i}") 
                for i in range(2)]
    
    # 启动子进程
    for p in processes:
        p.start()
    
    # 等待子进程结束
    for p in processes:
        p.join()
```

运行：

```bash
>>> python test_ipdb_in_subprocess.py

Process Worker-1:
Traceback (most recent call last):
  File "/usr/lib/python3.10/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/lib/python3.10/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  # ....
  File "/usr/lib/python3.10/bdb.py", line 90, in trace_dispatch
    return self.dispatch_line(frame)
  File "/usr/lib/python3.10/bdb.py", line 115, in dispatch_line
    if self.quitting: raise BdbQuit
bdb.BdbQuit
```

### use rpdb

官网：[rpdb github](https://github.com/tamentis/rpdb.git)

安装：
```bash
pip install rpdb
```

使用(一般喜欢注入代码使用，在想要断点处，添加代码即可)：

```bash
import rpdb
rpdb.set_trace(addr="0.0.0.0",port=5555)
```

然后开启一个新的终端连接这个 `5555` 端口:
```bash
# 使用 nc 进行连接
# Ubuntu安装
apt-get install netcat
# 连接
nc -t 0.0.0.0 5555

# 或者使用 python -m telnetlib 连接
python -m telnetlib 0.0.0.0 5555
```


## 使用 `cuda-gdb` + `ipdb`

复用之前的代码：创建一个 `example.cu` 文件，文件内容为：
```c++
#include <stdio.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// 核函数定义，定义每个 thread 干什么
__global__ void demo_kernel(){
    printf("hello,world!!! \n\n");
}

// 封装 CUDA 调用的函数
void launch_kernel() {
    cudaError_t err; 
    
    demo_kernel<<<1,1>>>();
    err = cudaGetLastError(); // 执行kernel launch 之后判断
    if (err != cudaSuccess) {
        printf("CUDA launch error: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA DeviceSynchronize error: %s\n", cudaGetErrorString(err));
    }
}

// pybind11 模块定义
PYBIND11_MODULE(example, m) {
    m.def("run", &launch_kernel, "Launch the CUDA kernel");
}
```

然后使用 `nvcc` 工具进行编译：
```bash
# 或者根据环境动态设置编译环境
nvcc -g -G -lineinfo -arch=sm_75 --shared -Xcompiler -fPIC -Xcompiler $(python3 -m pybind11 --includes)  -o example$(python3-config --extension-suffix) example.cu
```

然后在该同目录下编写 `test.py` 文件用于调用测试程序，内容为：

```python
import example
print(example.run())
```

对于此时就可以进行调试了:
```bash
# 查看目录结构
>>> tree .
.
├── example.cpython-310-x86_64-linux-gnu.so
├── example.cu
└── test.py

# 添加 python 解释器作为执行入口
>>> cuda-gdb python --quiet
Reading symbols from python...
(No debugging symbols found in python)

# 同步内核启动( cpu 和 gpu 同步)，便于调试
(cuda-gdb) set cuda launch_blocking on

# 给 cuda kernel 打断点： 或者直接给算子打 break demo_kernel
(cuda-gdb) break example.cu:8
No symbol table is loaded.  Use the "file" command.
Make breakpoint pending on future shared library load? (y or [n]) y
Breakpoint 1 (example.cu:8) pending.

# 查看断点
(cuda-gdb) info breakpoints  
Num     Type           Disp Enb Address    What
1       breakpoint     keep y   <PENDING>  example.cu:8

# 执行程序，类似于 python -m ipdb test.py
(cuda-gdb) run -m ipdb test.py
Starting program: /usr/bin/python -m ipdb test.py
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib/x86_64-linux-gnu/libthread_db.so.1".
/usr/lib/python3.10/runpy.py:126: RuntimeWarning: 'ipdb.__main__' found in sys.modules after import of package 'ipdb', but prior to execution of 'ipdb.__main__'; this may result in unpredictable behaviour
  warn(RuntimeWarning(msg))
[New Thread 0x7ffff5893640 (LWP 21255)]
> /xxx/xxx/test.py(1)<module>()
----> 1 import example
      2 print(example.run())

[New Thread 0x7ffff4c0c640 (LWP 21256)]

# 给 python 程序的 第二行打断点
ipdb> b 2
Breakpoint 1 at /xxx/xxx/test.py:2

# 查看 python 的所有断点
ipdb> b
Num Type         Disp Enb   Where
1   breakpoint   keep yes   at /xxx/xxx/test.py:2

# 直接执行，调试 python 代码一行一行执行也成
# 不过在运行到对应的 kernel的时候 step into
ipdb> c
[New Thread 0x7fffedc35640 (LWP 21490)]
[New Thread 0x7fffec9b5640 (LWP 21491)]
[Detaching after fork from child process 21492]
[New Thread 0x7fffe0f9c640 (LWP 21499)]
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (0,0,0), device 0, sm 0, warp 0, lane 0]

CUDA thread hit Breakpoint 1, demo_kernel<<<(1,1,1),(1,1,1)>>> () at example.cu:8
8	    printf("hello,world!!! \n\n");

(cuda-gdb)  # ====> 注意，此时是直接回到了 cuda-gdb 调试的。 就可以参考之前的 cuda-gdb 了
```

此时就差不多能够调试 `python` 和 `cuda` 了。 

