// 第 3 章示例：用 compute-sanitizer 排查越界访问
// 本文件刻意埋了一个 off-by-one 越界 bug，用于练习 compute-sanitizer 的排查流程。
// 编译：nvcc -O3 -lineinfo sanitizer_oob.cu -o sanitizer_oob
// 运行：./sanitizer_oob                     （很可能"侥幸"通过校验）
// 排查：compute-sanitizer ./sanitizer_oob   （精确定位越界的内核/行号/线程）
// 修复：把下面标注 BUG 的 i <= n 改为 i < n，重新编译复验
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call)                                                       \
do {                                                                      \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(err));                                 \
        exit(1);                                                          \
    }                                                                     \
} while (0)

// 数组逆序：out[i] = in[n - 1 - i]
__global__ void reverseArray(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n) {                       // BUG：应为 i < n，差一错误（off-by-one）
        out[i] = in[n - 1 - i];         // i == n 时：写 out[n] 越界，读 in[-1] 越界
    }
}

int main(void) {
    const int n = 1000;                 // 故意取一个除不尽 256 的规模
    const size_t bytes = n * sizeof(float);

    float *hIn  = (float *)malloc(bytes);
    float *hOut = (float *)malloc(bytes);
    for (int i = 0; i < n; i++) hIn[i] = (float)i;

    float *dIn, *dOut;
    CHECK(cudaMalloc(&dIn, bytes));
    CHECK(cudaMalloc(&dOut, bytes));
    CHECK(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    reverseArray<<<blocks, threads>>>(dIn, dOut, n);
    CHECK(cudaGetLastError());          // 启动配置合法：这里查不出越界
    CHECK(cudaDeviceSynchronize());     // 越界"侥幸"未触发硬件异常时，这里也不报错

    CHECK(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));

    // 校验前 n 个元素——注意：即使全对，也不代表没有越界（地雷可能没被踩响）
    int errors = 0;
    for (int i = 0; i < n; i++)
        if (hOut[i] != hIn[n - 1 - i]) errors++;
    if (errors == 0)
        printf("Verify: PASS (%d elements checked)\n", n);
    else
        printf("Verify: FAIL (%d errors)\n", errors);

    CHECK(cudaFree(dIn));
    CHECK(cudaFree(dOut));
    free(hIn); free(hOut);
    return 0;
}
