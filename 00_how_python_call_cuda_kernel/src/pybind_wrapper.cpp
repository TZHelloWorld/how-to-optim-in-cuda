// ===========================================================================
// pybind_wrapper.cpp — pybind11 绑定层
//
// 职责：将 C++ 接口 launch_cuda_hello() 暴露为 Python 模块 cuda_hello
//       中的函数 hello()。
//
// 关键约束：PYBIND11_MODULE 的第一个参数（模块名）必须与最终生成的
//           .so 文件名前缀一致（cuda_hello.cpython-XXX.so），否则
//           Python 导入时报错：
//           "dynamic module does not define module export function
//            (PyInit_cuda_hello)"
//
// 本文件由普通 C++ 编译器（g++）编译，通过链接 libcuda_functions.so
// 获得 CUDA 功能，自身不包含任何 CUDA 语法。
// ===========================================================================
#include <pybind11/pybind11.h>

#include "cuda_hello.h"

PYBIND11_MODULE(cuda_hello, m) {
    m.doc() = "Minimal example: calling a CUDA kernel from Python via pybind11";

    // Python 侧调用：import cuda_hello; cuda_hello.hello()
    m.def("hello", &launch_cuda_hello,
          "Launch a CUDA kernel that prints per-thread coordinates");
}
