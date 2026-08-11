// 第 4 章示例：并行归约三个版本对比——观察线程束分化的性能影响
// 编译运行：nvcc -O3 reduce_divergence.cu -o reduce_divergence && ./reduce_divergence
// 进一步分析：ncu --metrics smsp__average_inst_executed_per_warp ./reduce_divergence
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

// 版本 1：相邻配对（tid % (2*stride) == 0 引发严重 warp 分化）
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// 版本 2：重排线程索引（干活的始终是块内前段连续线程）
__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            idata[index] += idata[index + stride];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// 版本 3：交错配对（跨步从大到小，分化最小 + 访存合并）
__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

typedef void (*Kernel)(int *, int *, unsigned int);

float benchKernel(Kernel k, const char *name, const int *h_idata,
                  int *d_idata, int *d_odata, int *h_odata,
                  unsigned int n, int blocks, int threads, long long expect) {
    size_t bytes = n * sizeof(int);
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    k<<<blocks, threads>>>(d_idata, d_odata, n);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    CHECK(cudaMemcpy(h_odata, d_odata, blocks * sizeof(int), cudaMemcpyDeviceToHost));
    long long sum = 0;
    for (int i = 0; i < blocks; i++) sum += h_odata[i];

    printf("%-24s %8.3f ms   sum = %lld (%s)\n",
           name, ms, sum, sum == expect ? "PASS" : "FAIL");
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms;
}

int main(void) {
    unsigned int n = 1 << 24;
    int threads = 512;
    int blocks = (n + threads - 1) / threads;
    size_t bytes = n * sizeof(int);

    int *h_idata = (int *)malloc(bytes);
    int *h_odata = (int *)malloc(blocks * sizeof(int));
    long long expect = 0;
    for (unsigned int i = 0; i < n; i++) { h_idata[i] = 1; expect += 1; }

    int *d_idata, *d_odata;
    CHECK(cudaMalloc(&d_idata, bytes));
    CHECK(cudaMalloc(&d_odata, blocks * sizeof(int)));

    // 预热
    reduceInterleaved<<<blocks, threads>>>(d_idata, d_odata, n);
    CHECK(cudaDeviceSynchronize());

    benchKernel(reduceNeighbored,     "reduceNeighbored",     h_idata, d_idata, d_odata, h_odata, n, blocks, threads, expect);
    benchKernel(reduceNeighboredLess, "reduceNeighboredLess", h_idata, d_idata, d_odata, h_odata, n, blocks, threads, expect);
    benchKernel(reduceInterleaved,    "reduceInterleaved",    h_idata, d_idata, d_odata, h_odata, n, blocks, threads, expect);

    CHECK(cudaFree(d_idata)); CHECK(cudaFree(d_odata));
    free(h_idata); free(h_odata);
    return 0;
}
