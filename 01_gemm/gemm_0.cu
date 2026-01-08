#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// CUDA API 错误检查宏：每个 CUDA 调用都应该检查返回值
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// V0：一线程一输出元素（x 方向映射到行——刻意保留的低效映射，V1 修正）
__global__ void gemm_v0(int M, int N, int K,
                        const float* A, const float* B, float* C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

// CPU 参考实现：用于验证 GPU 结果
void gemm_cpu(int M, int N, int K,
              const float* A, const float* B, float* C) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

// 打印矩阵（只用于小矩阵）
void print_matrix(const char* name, const float* mat, int rows, int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("  ");
        for (int j = 0; j < cols; j++) {
            printf("%8.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    printf("===============start===================\n");

    // 故意取不能被 block 整除的形状，顺便测试边界分支 if(row<M && col<N)
    const int M = 6, N = 7, K = 5;

    // ---------- 1. 分配并初始化主机内存 ----------
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    float* hA    = (float*)malloc(bytesA);
    float* hB    = (float*)malloc(bytesB);
    float* hC    = (float*)malloc(bytesC);   // GPU 结果
    float* hCref = (float*)malloc(bytesC);   // CPU 参考结果

    // 用随机数填充（不要用全 1：行列写反等索引错误在对称输入下测不出来）
    srand(42);
    for (int i = 0; i < M * K; i++) hA[i] = (float)(rand() % 10) - 5.0f;
    for (int i = 0; i < K * N; i++) hB[i] = (float)(rand() % 10) - 5.0f;

    // ---------- 2. 分配设备内存并拷贝输入 ----------
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    CHECK_CUDA(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    // ---------- 3. 启动 kernel ----------
    // 注意 V0 的映射：x → 行(M)，y → 列(N)，grid 形状要与之对应
    dim3 block(4, 4);
    dim3 grid((M + block.x - 1) / block.x,    // 向上取整覆盖 M
              (N + block.y - 1) / block.y);   // 向上取整覆盖 N
    printf("grid=(%d,%d), block=(%d,%d)\n\n", grid.x, grid.y, block.x, block.y);

    gemm_v0<<<grid, block>>>(M, N, K, dA, dB, dC);
    CHECK_CUDA(cudaGetLastError());           // 捕获 kernel 启动错误
    CHECK_CUDA(cudaDeviceSynchronize());      // 等待 kernel 完成并捕获执行错误

    // ---------- 4. 拷回结果并与 CPU 参考对比 ----------
    CHECK_CUDA(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));
    gemm_cpu(M, N, K, hA, hB, hCref);

    print_matrix("A", hA, M, K);
    print_matrix("B", hB, K, N);
    print_matrix("C (GPU)", hC, M, N);
    print_matrix("C (CPU ref)", hCref, M, N);

    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        if (fabsf(hC[i] - hCref[i]) > 1e-4f) {
            if (errors < 5) {  // 最多打印前 5 个错误
                printf("MISMATCH at [%d,%d]: GPU=%f, CPU=%f\n",
                       i / N, i % N, hC[i], hCref[i]);
            }
            errors++;
        }
    }
    printf(errors == 0 ? "PASS: GPU result matches CPU reference.\n"
                       : "FAIL: %d mismatches.\n", errors);

    // ---------- 5. 释放资源 ----------
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    free(hA); free(hB); free(hC); free(hCref);

    printf("===============end=====================\n");
    return errors == 0 ? 0 : 1;
}
