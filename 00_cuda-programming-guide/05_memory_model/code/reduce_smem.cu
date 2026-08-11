// 第 7 章示例：共享内存归约（对比第 4 章的全局内存版本）
// 编译运行：nvcc -O3 reduce_smem.cu -o reduce_smem && ./reduce_smem
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

#define BLOCK_SIZE 256

__global__ void reduceSmem(const int *g_idata, int *g_odata, unsigned int n) {
    __shared__ int smem[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. 全局内存 → 共享内存（合并读）
    smem[tid] = (idx < n) ? g_idata[idx] : 0;
    __syncthreads();

    // 2. 共享内存内交错配对归约（无分化 + 无 bank 冲突）
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // 3. 块结果写回全局内存
    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

int main(void) {
    unsigned int n = 1 << 24;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t bytes = n * sizeof(int);

    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(blocks * sizeof(int));
    long long expect = 0;
    for (unsigned int i = 0; i < n; i++) { h_idata[i] = 1; expect++; }

    int *d_idata, *d_odata;
    CHECK(cudaMalloc(&d_idata, bytes));
    CHECK(cudaMalloc(&d_odata, blocks * sizeof(int)));
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    reduceSmem<<<blocks, BLOCK_SIZE>>>(d_idata, d_odata, n);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    CHECK(cudaMemcpy(h_odata, d_odata, blocks * sizeof(int), cudaMemcpyDeviceToHost));
    long long sum = 0;
    for (int i = 0; i < blocks; i++) sum += h_odata[i];   // 最后一步在 CPU 上完成

    printf("reduceSmem: %.3f ms, sum = %lld (%s)\n",
           ms, sum, sum == expect ? "PASS" : "FAIL");

    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_idata)); CHECK(cudaFree(d_odata));
    free(h_idata); free(h_odata);
    return 0;
}
