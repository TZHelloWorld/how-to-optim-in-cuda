// 第 9 章示例：终极归约内核——grid-stride + warp shuffle + 共享内存汇聚 + 原子聚合
// 编译运行：nvcc -O3 reduce_final.cu -o reduce_final && ./reduce_final
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call)                                                       \
do {                                                                      \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
        printf("CUDA Error: %s:%d, %s\n", __FILE__, __LINE__,             \
               cudaGetErrorString(err));                                  \
        exit(1);                                                          \
    }                                                                     \
} while (0)

// warp 内 32 个数归约：5 条 shuffle 指令完成，无共享内存、无 __syncthreads()
__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;   // lane 0 持有全 warp 的和
}

__global__ void reduceFinal(const int *g_idata, int *g_odata, unsigned int n) {
    // 1. grid-stride loop：每线程先串行累加多个元素
    int sum = 0;
    int stride = gridDim.x * blockDim.x;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        sum += g_idata[i];

    // 2. warp 内归约（shuffle）
    sum = warpReduceSum(sum);

    // 3. 各 warp 部分和经共享内存汇聚，由第一个 warp 再归约
    __shared__ int warpSums[32];
    int lane   = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) warpSums[warpId] = sum;
    __syncthreads();

    int nWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (warpId == 0) {
        sum = (lane < nWarps) ? warpSums[lane] : 0;
        sum = warpReduceSum(sum);
        // 4. 块结果直接原子加到最终结果，省去第二次内核启动
        if (lane == 0) atomicAdd(g_odata, sum);
    }
}

int main(void) {
    unsigned int n = 1 << 26;
    size_t bytes = n * sizeof(int);

    int *h_idata = (int *)malloc(bytes);
    long long expect = 0;
    for (unsigned int i = 0; i < n; i++) { h_idata[i] = 1; expect++; }

    int *d_idata, *d_result;
    CHECK(cudaMalloc(&d_idata, bytes));
    CHECK(cudaMalloc(&d_result, sizeof(int)));
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_result, 0, sizeof(int)));

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int threads = 256, blocks = 32 * numSMs;

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    reduceFinal<<<blocks, threads>>>(d_idata, d_result, n);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    int result;
    CHECK(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    // 有效带宽（归约只读一遍数据）
    double gbps = (double)bytes / (ms * 1e-3) / 1e9;
    printf("reduceFinal: %.3f ms, %.1f GB/s, sum = %d (%s)\n",
           ms, gbps, result, (long long)result == expect ? "PASS" : "FAIL");

    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_idata)); CHECK(cudaFree(d_result));
    free(h_idata);
    return 0;
}
