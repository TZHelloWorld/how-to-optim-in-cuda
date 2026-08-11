// 第 2 章示例：2D 矩阵加法（2D Grid + 2D Block 映射）
// 编译运行：nvcc -O3 mat_add.cu -o mat_add && ./mat_add
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

// C[ny][nx] = A[ny][nx] + B[ny][nx]，行主序存储
__global__ void matAdd(const float *A, const float *B, float *C,
                       int nx, int ny) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // x → 列（内层连续维度）
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // y → 行

    if (col < nx && row < ny) {                        // 两个维度都要边界检查
        int idx = row * nx + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    int nx = 4096, ny = 4096;
    size_t bytes = (size_t)nx * ny * sizeof(float);

    float *hA = (float *)malloc(bytes), *hB = (float *)malloc(bytes),
          *hC = (float *)malloc(bytes);
    for (int i = 0; i < nx * ny; i++) { hA[i] = 1.0f; hB[i] = 2.0f; }

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));
    CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    // block.x = 32（warp 大小）保证 warp 内线程访问同一行连续 32 列 → 合并访问
    dim3 block(32, 8);                                 // 每块 256 线程
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);
    matAdd<<<grid, block>>>(dA, dB, dC, nx, ny);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    printf("C[0] = %f, C[last] = %f (expect 3.0)\n", hC[0], hC[nx * ny - 1]);

    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC);
    return 0;
}
