// 第 9 章示例：直方图统计——朴素全局原子 vs 共享内存分层聚合
// 编译运行：nvcc -O3 histogram.cu -o histogram && ./histogram
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

#define NUM_BINS 256

// 版本 1：直接对全局内存做原子加（同 bin 冲突激烈时慢）
__global__ void histNaive(const unsigned char *data, int n, unsigned int *hist) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&hist[data[i]], 1);
}

// 版本 2：共享内存私有直方图 → 块内聚合 → 一次性合并到全局
__global__ void histSmem(const unsigned char *data, int n, unsigned int *hist) {
    __shared__ unsigned int smem[NUM_BINS];

    // 1. 清零共享内存直方图
    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) smem[b] = 0;
    __syncthreads();

    // 2. grid-stride loop 统计到"块私有"直方图
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&smem[data[i]], 1);
    __syncthreads();

    // 3. 每块只对全局内存做 NUM_BINS 次原子加
    for (int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)
        atomicAdd(&hist[b], smem[b]);
}

int main(void) {
    int n = 1 << 26;   // 64M 字节
    unsigned char *h_data = (unsigned char *)malloc(n);
    // 故意让数据分布偏斜（低 bin 冲突激烈），拉开两版本差距
    for (int i = 0; i < n; i++) h_data[i] = (unsigned char)(rand() % 16);

    unsigned char *d_data;
    unsigned int *d_hist;
    CHECK(cudaMalloc(&d_data, n));
    CHECK(cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int)));
    CHECK(cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    unsigned int h_hist[NUM_BINS];
    float ms;

    // --- 版本 1 ---
    CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    int threads = 256, blocks = (n + threads - 1) / threads;
    CHECK(cudaEventRecord(start));
    histNaive<<<blocks, threads>>>(d_data, n, d_hist);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaMemcpy(h_hist, d_hist, sizeof(h_hist), cudaMemcpyDeviceToHost));
    long long total1 = 0; for (int b = 0; b < NUM_BINS; b++) total1 += h_hist[b];
    printf("histNaive: %8.3f ms, total = %lld (%s)\n",
           ms, total1, total1 == n ? "PASS" : "FAIL");

    // --- 版本 2 ---
    CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    CHECK(cudaEventRecord(start));
    histSmem<<<32 * numSMs, threads>>>(d_data, n, d_hist);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaMemcpy(h_hist, d_hist, sizeof(h_hist), cudaMemcpyDeviceToHost));
    long long total2 = 0; for (int b = 0; b < NUM_BINS; b++) total2 += h_hist[b];
    printf("histSmem:  %8.3f ms, total = %lld (%s)\n",
           ms, total2, total2 == n ? "PASS" : "FAIL");

    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_data)); CHECK(cudaFree(d_hist));
    free(h_data);
    return 0;
}
