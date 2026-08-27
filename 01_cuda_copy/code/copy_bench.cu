// copy_bench.cu — D2D 拷贝的多种做法与带宽实测（对应文档第 4 章、第 6.1 节）
//
// 对应文档: ../cuda_copy_operator_guide.md
// 覆盖版本:
//   cudaMemcpy    驱动 / Copy Engine 路径（不占 SM，基准上限）        (§3.2 / §4.3)
//   V0 naive      一线程一元素（4 B），并发全靠海量线程               (§4.3)
//   V1 gridstride grid-stride loop，固定网格反复扫、线程复用          (§4.3)
//   V2 float4     float4 向量化，一条 LDG.128 搬 16 B，在途字节 ×4    (§4.3)
//   V3 fused      拷贝 + scale + ReLU 融合，计算藏进访存影子里         (§6.1)
//
// 有效带宽定义（读 + 写）: BW_eff = 2N / t （CUDA Best Practices Guide, §1.2）
//
// 编译:
//   nvcc -O3 -arch=native copy_bench.cu -o copy_bench
// 运行:
//   ./copy_bench            # 默认 2^28 个 float = 1 GiB

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <utility>   // std::swap（本文件未直接使用，保留以对齐参考样例风格）
#include <cuda_runtime.h>

// 文档中的 CHECK 宏（这里统一命名为 CUDA_CHECK，与参考样例一致）
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// V0：一线程一元素（4 B），并发全靠海量线程
// ---------------------------------------------------------------------------
__global__ void copy_naive(const float* __restrict__ src, float* __restrict__ dst,
                           size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];                 // LDG 4B + STG 4B
}

// ---------------------------------------------------------------------------
// V1：grid-stride loop——固定网格反复扫，线程复用、启动参数与数据量解耦
// ---------------------------------------------------------------------------
__global__ void copy_gridstride(const float* __restrict__ src, float* __restrict__ dst,
                                size_t n) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// V2：float4 向量化——一条 LDG.128 搬 16 B，在途字节数 ×4（§4.2 的处方）
// ---------------------------------------------------------------------------
__global__ void copy_float4(const float4* __restrict__ src, float4* __restrict__ dst,
                            size_t n4) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride)
        dst[i] = src[i];                        // LDG.128 + STG.128
}

// ---------------------------------------------------------------------------
// V3：拷贝 + 逐元素计算融合（scale + ReLU）——与纯拷贝耗时几乎完全相同
//     （ALU 全程在等访存，计算藏进访存的影子里，§6.1）
// ---------------------------------------------------------------------------
__global__ void copy_scale_relu(const float4* __restrict__ src, float4* __restrict__ dst,
                                size_t n4, float alpha) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = src[i];                       // 数据既然已经进了寄存器……
        v.x = fmaxf(v.x * alpha, 0.f);           // ……顺手做任何逐元素计算都免费
        v.y = fmaxf(v.y * alpha, 0.f);
        v.z = fmaxf(v.z * alpha, 0.f);
        v.w = fmaxf(v.w * alpha, 0.f);
        dst[i] = v;
    }
}

// ===========================================================================
// 计时辅助：预热一次，再计时 iters 次取平均（对应文档 bench_ms）
// ===========================================================================
template <typename F>
static float bench_ms(F launch, int iters = 20) {
    launch();                                   // 预热
    CUDA_CHECK(cudaGetLastError());
    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(beg));
    for (int i = 0; i < iters; i++) launch();
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms / iters;
}

// 校验：dst 应与 src 逐元素一致（纯拷贝版本）
static bool verify_copy(const float* h_src, const float* d_dst, size_t n) {
    float* h_dst = (float*)malloc(sizeof(float) * n);
    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, sizeof(float) * n, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < n; ++i) {
        if (h_dst[i] != h_src[i]) { ok = false; break; }
    }
    free(h_dst);
    return ok;
}

// 校验：融合版本 dst == max(src * alpha, 0)
static bool verify_fused(const float* h_src, const float* d_dst, size_t n, float alpha) {
    float* h_dst = (float*)malloc(sizeof(float) * n);
    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, sizeof(float) * n, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < n; ++i) {
        float expect = fmaxf(h_src[i] * alpha, 0.f);
        if (fabsf(h_dst[i] - expect) > 1e-5f) { ok = false; break; }
    }
    free(h_dst);
    return ok;
}

int main(int argc, char** argv) {
    // 默认 2^28 个 float = 1 GiB
    size_t N = (argc > 1) ? (size_t)strtoull(argv[1], nullptr, 10) : (1ull << 28);
    N &= ~((size_t)3);                          // 向下取整到 4 的倍数，便于 float4
    const size_t bytes = N * sizeof(float);

    printf("D2D copy benchmark: N = %zu floats (%.2f MiB), 2N bytes moved\n\n",
           N, bytes / (1024.0 * 1024.0));

    // 主机端初始化：填入可判别的数据（含负数以检验 ReLU）
    float* h_src = (float*)malloc(bytes);
    for (size_t i = 0; i < N; ++i)
        h_src[i] = ((i & 7) == 0) ? -1.0f : (float)(i % 13) + 0.5f;

    float *src, *dst;
    CUDA_CHECK(cudaMalloc(&src, bytes));
    CUDA_CHECK(cudaMalloc(&dst, bytes));
    CUDA_CHECK(cudaMemcpy(src, h_src, bytes, cudaMemcpyHostToDevice));

    // 有效带宽 = (读 + 写) / 时间 = 2N / t
    auto report = [&](const char* name, float ms, bool ok) {
        printf("%-16s %8.3f ms   %8.1f GB/s   %s\n",
               name, ms, 2.0 * bytes / ms / 1e6, ok ? "OK" : "FAIL");
    };

    const int block = 256;
    const int grid  = 2048;                     // 铺满全部 SM 若干倍即可
    const float alpha = 2.0f;

    // --- cudaMemcpy（驱动 / CE 路径） -----------------------------------
    CUDA_CHECK(cudaMemset(dst, 0, bytes));
    float t_memcpy = bench_ms([&] {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice));
    });
    report("cudaMemcpy", t_memcpy, verify_copy(h_src, dst, N));

    // --- V0 naive -------------------------------------------------------
    CUDA_CHECK(cudaMemset(dst, 0, bytes));
    float t_v0 = bench_ms([&] {
        copy_naive<<<(int)((N + block - 1) / block), block>>>(src, dst, N);
    });
    report("V0 naive", t_v0, verify_copy(h_src, dst, N));

    // --- V1 gridstride --------------------------------------------------
    CUDA_CHECK(cudaMemset(dst, 0, bytes));
    float t_v1 = bench_ms([&] {
        copy_gridstride<<<grid, block>>>(src, dst, N);
    });
    report("V1 gridstride", t_v1, verify_copy(h_src, dst, N));

    // --- V2 float4 ------------------------------------------------------
    CUDA_CHECK(cudaMemset(dst, 0, bytes));
    float t_v2 = bench_ms([&] {
        copy_float4<<<grid, block>>>((const float4*)src, (float4*)dst, N / 4);
    });
    report("V2 float4", t_v2, verify_copy(h_src, dst, N));

    // --- V3 fused (scale + ReLU) ----------------------------------------
    CUDA_CHECK(cudaMemset(dst, 0, bytes));
    float t_v3 = bench_ms([&] {
        copy_scale_relu<<<grid, block>>>((const float4*)src, (float4*)dst, N / 4, alpha);
    });
    report("V3 fused", t_v3, verify_fused(h_src, dst, N, alpha));

    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
    free(h_src);
    printf("\n注: 融合版 (V3) 带宽应与 V2 几乎一致——计算藏进了访存的影子里 (§6.1)\n");
    return 0;
}
