# 运行
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
2.  为了能够让 `python` 能够导入生成的 `.so` 文件，需要确保 `pybind11` 配置的 `module` 名字和 `.so`问价的模块名字一样。如错误示例：
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

    cudaError_t err = cudaDeviceSynchronize();
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



