// ===========================================================================
// cuda_hello.cu — CUDA Kernel 实现层
//
// 职责：定义 __global__ 核函数，并提供 C++ 封装函数 launch_cuda_hello()
//       负责 kernel 启动、错误检查与设备同步。
//
// 分层说明：本文件由 nvcc 编译为共享库 libcuda_functions.so，
//           绑定层（pybind_wrapper.cpp）通过 cuda_hello.h 中声明的
//           普通 C++ 接口调用，不直接接触 CUDA 语法。
// ===========================================================================
#include <stdio.h>

#include <cuda_runtime_api.h>

#include "cuda_hello.h"

// ---------------------------------------------------------------------------
// CUDA 运行时错误检查宏
//
// 用法：CUDA_CHECK(cudaXxx(...));
// 出错时打印文件名、行号、错误码与错误描述。
//
// 注意：CUDA kernel 启动是异步的，launch 阶段的错误需要通过
//       cudaGetLastError() 主动获取；执行阶段的错误则要等到下一次
//       同步点（如 cudaDeviceSynchronize）才会暴露。
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr_to_check)                              \
    do {                                                       \
        cudaError_t result = (expr_to_check);                  \
        if (result != cudaSuccess) {                           \
            fprintf(stderr,                                    \
                    "CUDA error at %s:%d code=%d(%s)\n",       \
                    __FILE__,                                  \
                    __LINE__,                                  \
                    result,                                    \
                    cudaGetErrorString(result));               \
        }                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel 1：打印每个线程的完整坐标信息
//
// threadIdx / blockIdx / blockDim / gridDim 均为 CUDA 内置变量，
// 用于定位当前线程在整个 Grid 中的位置。
// ---------------------------------------------------------------------------
__global__ void cuda_hello_kernel() {
    printf("Hello, cuda kernel; "
           "Thread (%d,%d,%d) in Block (%d,%d,%d), "
           "Grid (%d,%d,%d), BlockSize (%d,%d,%d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           gridDim.x, gridDim.y, gridDim.z,
           blockDim.x, blockDim.y, blockDim.z);
}

// ---------------------------------------------------------------------------
// Kernel 2：最精简版本，仅打印一条消息（与 Kernel 1 对比学习用）
//
// 如需单独测试，取消 launch_cuda_hello() 中对应调用的注释即可。
// ---------------------------------------------------------------------------
__global__ void simple_cuda_hello_kernel() {
    printf("Hello, CUDA kernel!\n");
}

// ---------------------------------------------------------------------------
// C++ 封装函数：供绑定层调用的普通 C++ 接口
//
// 执行流程：
//   1. 以 <<<grid, block>>> 配置启动 kernel（异步提交）
//   2. cudaGetLastError()      检查 launch 阶段错误
//   3. cudaDeviceSynchronize() 等待 GPU 执行完成并捕获运行期错误
// ---------------------------------------------------------------------------
void launch_cuda_hello() {
    // grid(2,2) x block(4,4)：共 4 个线程块，每块 16 个线程，合计 64 个线程
    dim3 grid(2, 2);
    dim3 block(4, 4);
    cuda_hello_kernel<<<grid, block>>>();

    // simple_cuda_hello_kernel<<<1, 1>>>();

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
