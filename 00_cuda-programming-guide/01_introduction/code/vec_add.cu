// 第 1 章示例：向量加法（CUDA 程序 6 步标准流程）
// 编译运行：nvcc -O3 vec_add.cu -o vec_add && ./vec_add
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 错误检查宏：CUDA API 均返回错误码，建议始终检查
#define CHECK(call)                                                       \
do {                                                                      \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
        printf("CUDA Error: %s:%d, %s\n", __FILE__, __LINE__,             \
               cudaGetErrorString(err));                                  \
        exit(1);                                                          \
    }                                                                     \
} while (0)

// 内核定义：__global__ 表示该函数在 GPU 上执行，从主机端调用
__global__ void vecAdd(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程索引
    if (i < n) {                                    // 边界检查
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    int n = 1 << 20;                    // 1M 个元素
    size_t bytes = n * sizeof(float);

    // 1. 分配主机内存并初始化
    float *hA = (float *)malloc(bytes);
    float *hB = (float *)malloc(bytes);
    float *hC = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) { hA[i] = 1.0f; hB[i] = 2.0f; }

    // 2. 分配设备内存
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));

    // 3. 主机 -> 设备 数据拷贝
    CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    // 4. 启动内核
    int threads = 256;
    int blocks = (n + threads - 1) / threads;   // 向上取整
    vecAdd<<<blocks, threads>>>(dA, dB, dC, n);
    CHECK(cudaGetLastError());

    // 5. 设备 -> 主机 拷贝结果（cudaMemcpy 隐式同步）
    CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    // 验证结果
    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (hC[i] != 3.0f) { ok = false; break; }
    }
    printf("Result: %s (C[0] = %f)\n", ok ? "PASS" : "FAIL", hC[0]);

    // 6. 释放资源
    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC);
    return 0;
}
