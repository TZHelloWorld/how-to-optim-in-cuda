// 第 3 章示例：GPU 设备信息查询
// 编译运行：nvcc -O3 device_query.cu -o device_query && ./device_query
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
