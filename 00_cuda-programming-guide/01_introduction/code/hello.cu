// 第 1 章示例：Hello World
// 编译运行：nvcc -O3 hello.cu -o hello && ./hello
#include <cstdio>

// __global__ 声明这是一个内核函数：在 GPU 上执行，从 CPU 调用
__global__ void helloFromGPU(void) {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main(void) {
    printf("Hello World from CPU!\n");

    helloFromGPU<<<1, 8>>>();     // 启动内核：1 个线程块 × 8 个线程
    cudaDeviceSynchronize();      // 等待 GPU 执行完（否则程序可能提前退出）
    return 0;
}
