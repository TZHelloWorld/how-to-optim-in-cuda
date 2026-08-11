// 第 2 章示例：打印线程索引，理解 Grid/Block/Thread 层次
// 编译运行：nvcc -O3 check_index.cu -o check_index && ./check_index
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex(void) {
    printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d) gridDim:(%d,%d,%d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char **argv) {
    dim3 grid(3, 3);    // 3×3 = 9 个线程块
    dim3 block(2, 2);   // 每块 2×2 = 4 个线程，共 36 个线程

    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

    checkIndex<<<grid, block>>>();
    cudaDeviceReset();   // 同步并复位设备
    return 0;
}
