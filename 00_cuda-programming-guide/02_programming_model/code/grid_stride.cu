// 第 2 章示例：Grid-Stride Loop——固定线程数处理任意规模数据
// 编译运行：nvcc -O3 grid_stride.cu -o grid_stride && ./grid_stride
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

__global__ void vecAddGS(const float *A, const float *B, float *C, int n) {
    int stride = gridDim.x * blockDim.x;      // 网格内线程总数
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += stride) {                       // 每个线程处理多个元素
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    int n = (1 << 24) + 7;   // 故意取非 2 的幂，验证任意规模都正确
    size_t bytes = n * sizeof(float);

    float *hA = (float *)malloc(bytes), *hB = (float *)malloc(bytes),
          *hC = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) { hA[i] = 1.0f; hB[i] = 2.0f; }

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));
    CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    // 块数取 SM 数量的若干倍即可，不必覆盖全部数据
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    vecAddGS<<<32 * numSMs, 256>>>(dA, dB, dC, n);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; i++)
        if (hC[i] != 3.0f) { ok = false; break; }
    printf("n = %d, blocks = %d, result: %s\n", n, 32 * numSMs, ok ? "PASS" : "FAIL");

    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC);
    return 0;
}
