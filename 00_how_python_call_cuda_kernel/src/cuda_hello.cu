#include <stdio.h>
#include <cuda_runtime_api.h>

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


//  内置变量： 用于定位各个 grid 中的 block 和 block 中的 thread
__global__ void cuda_hello_kernel() {
    printf("Hello, cuda kernel; Thread (%d,%d,%d) in Block (%d,%d,%d), Grid (%d,%d,%d), BlockSize (%d,%d,%d)\n",
       threadIdx.x, threadIdx.y, threadIdx.z,
       blockIdx.x, blockIdx.y, blockIdx.z,
       gridDim.x, gridDim.y, gridDim.z,
       blockDim.x, blockDim.y, blockDim.z);
}

__global__ void simple_cuda_hello_kernel() {
    printf("Hello, CUDA kernel!\n");
}

extern "C" void launch_cuda_hello() {
    
    dim3 grid(2,2);
    dim3 block(4,4);
    cuda_hello_kernel<<<grid, block>>>();

    // simple_cuda_hello_kernel<<<1, 1>>>();

    // check error state after kernel launch
    CUDA_CHECK(cudaGetLastError());
    // wait for kernel execution to complete
    // The CUDA_CHECK will report errors that occurred during execution of the kernel
    CUDA_CHECK(cudaDeviceSynchronize());

}