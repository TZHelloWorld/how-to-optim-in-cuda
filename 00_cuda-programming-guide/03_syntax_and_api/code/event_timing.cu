// 第 3 章示例：cudaEvent 计时 + 有效带宽计算
// 编译运行：nvcc -O3 event_timing.cu -o event_timing && ./event_timing
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

__global__ void vecAdd(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

int main(void) {
    int n = 1 << 26;    // 64M 元素，保证内核耗时可测
    size_t bytes = n * sizeof(float);

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));
    CHECK(cudaMemset(dA, 0, bytes));
    CHECK(cudaMemset(dB, 0, bytes));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // 预热（首次启动包含初始化开销，不计入测量）
    vecAdd<<<blocks, threads>>>(dA, dB, dC, n);
    CHECK(cudaDeviceSynchronize());

    // 用 CUDA Event 在 GPU 时间线上打点
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    const int iters = 100;
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        vecAdd<<<blocks, threads>>>(dA, dB, dC, n);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    // 有效带宽 = (读字节数 + 写字节数) / 耗时
    // 向量加法：读 2 个数组 + 写 1 个数组 = 3 * bytes
    double gbps = 3.0 * bytes / (ms * 1e-3) / 1e9;
    printf("kernel time: %.3f ms, effective bandwidth: %.1f GB/s\n", ms, gbps);
    printf("(对比 device_query 输出的峰值带宽，达到 80%% 以上说明已接近访存极限)\n");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    return 0;
}
