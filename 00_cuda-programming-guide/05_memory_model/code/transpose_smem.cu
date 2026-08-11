// 第 7 章示例：矩阵转置三个版本对比——朴素版 / 共享内存版 / 共享内存+padding 版
// 编译运行：nvcc -O3 transpose_smem.cu -o transpose_smem && ./transpose_smem
// bank 冲突分析：ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared ./transpose_smem
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

#define TILE_DIM 32
#define INDX(row, col, ld) (((row) * (ld)) + (col))

// 版本 1：朴素转置（读合并，写交叉——不合并）
__global__ void naiveTranspose(int m, const float *a, float *c) {
    int myCol = blockDim.x * blockIdx.x + threadIdx.x;
    int myRow = blockDim.y * blockIdx.y + threadIdx.y;
    if (myRow < m && myCol < m)
        c[INDX(myCol, myRow, m)] = a[INDX(myRow, myCol, m)];
}

// 版本 2：共享内存中转（读写全局内存均合并，但共享内存按列访问有 32 路 bank 冲突）
__global__ void smemTranspose(int m, const float *a, float *c) {
    __shared__ float tile[TILE_DIM][TILE_DIM];        // 无 padding

    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    if (row < m && col < m)
        tile[threadIdx.y][threadIdx.x] = a[row * m + col];
    __syncthreads();

    col = blockIdx.y * TILE_DIM + threadIdx.x;        // 块坐标交换
    row = blockIdx.x * TILE_DIM + threadIdx.y;
    if (row < m && col < m)
        c[row * m + col] = tile[threadIdx.x][threadIdx.y];   // 按列读 → bank 冲突
}

// 版本 3：共享内存 + padding（+1 列彻底消除 bank 冲突）
__global__ void smemPadTranspose(int m, const float *a, float *c) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];    // ← 关键：33 = 32 + 1

    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    if (row < m && col < m)
        tile[threadIdx.y][threadIdx.x] = a[row * m + col];
    __syncthreads();

    col = blockIdx.y * TILE_DIM + threadIdx.x;
    row = blockIdx.x * TILE_DIM + threadIdx.y;
    if (row < m && col < m)
        c[row * m + col] = tile[threadIdx.x][threadIdx.y];   // 无冲突
}

typedef void (*Kernel)(int, const float *, float *);

void bench(Kernel k, const char *name, int m, const float *dA, float *dC,
           dim3 grid, dim3 block, size_t bytes) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    k<<<grid, block>>>(m, dA, dC);      // 预热
    CHECK(cudaDeviceSynchronize());

    const int iters = 50;
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) k<<<grid, block>>>(m, dA, dC);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;
    double gbps = 2.0 * bytes / (ms * 1e-3) / 1e9;   // 读 + 写
    printf("%-20s %8.3f ms   %8.1f GB/s\n", name, ms, gbps);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
}

int main(void) {
    int m = 4096;
    size_t bytes = (size_t)m * m * sizeof(float);

    float *hA = (float *)malloc(bytes);
    for (int i = 0; i < m * m; i++) hA[i] = (float)i;

    float *dA, *dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dC, bytes));
    CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((m + TILE_DIM - 1) / TILE_DIM, (m + TILE_DIM - 1) / TILE_DIM);

    bench(naiveTranspose,   "naive",      m, dA, dC, grid, block, bytes);
    bench(smemTranspose,    "smem",       m, dA, dC, grid, block, bytes);
    bench(smemPadTranspose, "smem+pad",   m, dA, dC, grid, block, bytes);

    // 校验最后一个版本的正确性
    float *hC = (float *)malloc(bytes);
    CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int r = 0; r < m && ok; r += 997)
        for (int c = 0; c < m && ok; c += 991)
            if (hC[INDX(r, c, m)] != hA[INDX(c, r, m)]) ok = false;
    printf("correctness: %s\n", ok ? "PASS" : "FAIL");

    CHECK(cudaFree(dA)); CHECK(cudaFree(dC));
    free(hA); free(hC);
    return 0;
}
