// hgemm_wmma.cu — CUDA HGEMM（半精度矩阵乘）Tensor Core / WMMA 实现
//
// 对应文档: ../cuda_gemm_optimization_guide.md 第 11 章（V7）
//   V7 Tensor Core（WMMA）—— 一条 mma 指令完成 16×16×16 小矩阵乘加
//
// A: M×K (half, row-major)  B: K×N (half, row-major)  C: M×N (float)
// 要求 M、N、K 均为 16 的倍数。
//
// 编译（Tensor Core 需 Volta 及以上，sm_70+）:
//   nvcc -O3 -arch=sm_70 hgemm_wmma.cu -o hgemm_wmma
// 运行:
//   ./hgemm_wmma            # 默认 M=N=K=1024
//   ./hgemm_wmma 2048
//   ./hgemm_wmma 1024 512 256

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// WMMA 小矩阵块尺寸
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// ===========================================================================
// V7：Tensor Core WMMA 最小可用实现（第 11.3 节）
// 每个 Warp 负责 C 的一个 16×16 块；每次 load_matrix_sync 直接打到全局内存
// （相当于 Tensor Core 世界的 "V0"，数据零复用）
// blockDim = (128, 4)：x 方向 4 个 Warp，y 方向 4 个 Warp，共 16 块/Block
// ===========================================================================
__global__ void hgemm_wmma_v7(int M, int N, int K,
                              const half* A, const half* B, float* C) {
    // 每个 Warp 负责一个 16×16 输出块
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;  // 块列号
    int warpM = blockIdx.y * blockDim.y + threadIdx.y;         // 块行号

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;
    wmma::fill_fragment(cFrag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        // 全 Warp 协作加载 A、B 的 16×16 块（第三个参数是行跨度）
        wmma::load_matrix_sync(aFrag, A + warpM * WMMA_M * K + k, K);
        wmma::load_matrix_sync(bFrag, B + k * N + warpN * WMMA_N, N);
        // 一条 mma：16×16×16 = 4096 次乘加
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    wmma::store_matrix_sync(C + warpM * WMMA_M * N + warpN * WMMA_N, cFrag,
                            N, wmma::mem_row_major);
}

// ===========================================================================
// CPU 参考实现（用 float 累加，输入是 half 转来的 float）
// ===========================================================================
static void gemm_cpu(int M, int N, int K,
                     const float* A, const float* B, float* C) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += A[m * K + k] * B[k * N + n];
            C[m * N + n] = acc;
        }
}

int main(int argc, char** argv) {
    int M, N, K;
    if (argc >= 4) {
        M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]);
    } else if (argc == 2) {
        M = N = K = atoi(argv[1]);
    } else {
        M = N = K = 1024;
    }

    if (M % 16 || N % 16 || K % 16) {
        fprintf(stderr, "M, N, K 必须都是 16 的倍数\n");
        return 1;
    }
    printf("HGEMM (WMMA)  C[%dx%d] = A[%dx%d] x B[%dx%d]  (half x half -> float)\n\n",
           M, N, M, K, K, N);

    size_t elemsA = (size_t)M * K, elemsB = (size_t)K * N, elemsC = (size_t)M * N;

    // 主机端：float 原始数据 + half 输入
    float* hA_f    = (float*)malloc(elemsA * sizeof(float));
    float* hB_f    = (float*)malloc(elemsB * sizeof(float));
    half*  hA_h    = (half*) malloc(elemsA * sizeof(half));
    half*  hB_h    = (half*) malloc(elemsB * sizeof(half));
    float* hC      = (float*)malloc(elemsC * sizeof(float));   // GPU 结果
    float* hCref   = (float*)malloc(elemsC * sizeof(float));   // CPU 参考

    // 随机小数据（half 动态范围有限，控制在 [-1,1)）
    srand(42);
    for (size_t i = 0; i < elemsA; i++) {
        hA_f[i] = (float)(rand() % 200 - 100) / 100.0f;
        hA_h[i] = __float2half(hA_f[i]);
    }
    for (size_t i = 0; i < elemsB; i++) {
        hB_f[i] = (float)(rand() % 200 - 100) / 100.0f;
        hB_h[i] = __float2half(hB_f[i]);
    }

    // CPU 参考（用 half 舍入后的值以贴近硬件计算）
    for (size_t i = 0; i < elemsA; i++) hA_f[i] = __half2float(hA_h[i]);
    for (size_t i = 0; i < elemsB; i++) hB_f[i] = __half2float(hB_h[i]);
    printf("Computing CPU reference...\n");
    gemm_cpu(M, N, K, hA_f, hB_f, hCref);

    half *dA, *dB;
    float* dC;
    CUDA_CHECK(cudaMalloc(&dA, elemsA * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dB, elemsB * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC, elemsC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA_h, elemsA * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB_h, elemsB * sizeof(half), cudaMemcpyHostToDevice));

    // 启动配置：blockDim=(128,4) → x 方向 4 warp、y 方向 4 warp
    dim3 block(128, 4);
    dim3 grid((N + (WMMA_N * 4) - 1) / (WMMA_N * 4),   // 每 Block 覆盖 4 个 16 宽块 = 64 列
              (M + (WMMA_M * 4) - 1) / (WMMA_M * 4));  // 每 Block 覆盖 4 个 16 高块 = 64 行

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热
    hgemm_wmma_v7<<<grid, block>>>(M, N, K, dA, dB, dC);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    CUDA_CHECK(cudaEventRecord(start));
    hgemm_wmma_v7<<<grid, block>>>(M, N, K, dA, dB, dC);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double gflops = 2.0 * M * N * K / 1e9 / (ms * 1e-3);

    CUDA_CHECK(cudaMemcpy(hC, dC, elemsC * sizeof(float), cudaMemcpyDeviceToHost));

    // 相对误差校验（half 精度，容差放宽）
    double max_rel = 0.0;
    int errors = 0;
    for (size_t idx = 0; idx < elemsC; idx++) {
        float ref = hCref[idx];
        float rel = fabsf(hC[idx] - ref) / (fabsf(ref) + 1e-2f);
        if (rel > max_rel) max_rel = rel;
        if (rel > 2e-2f) errors++;
    }

    printf("\n%-8s %12s %12s %10s\n", "Ver", "time(ms)", "GFLOPS", "check");
    printf("--------------------------------------------------\n");
    printf("%-8s %12.4f %12.1f %10s (max_rel=%.2e)\n",
           "V7(fp16)", ms, gflops, errors == 0 ? "PASS" : "FAIL", max_rel);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA_f); free(hB_f); free(hA_h); free(hB_h); free(hC); free(hCref);
    return 0;
}
