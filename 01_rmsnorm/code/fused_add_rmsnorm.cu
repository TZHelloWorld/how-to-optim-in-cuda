// fused_add_rmsnorm.cu — V4：算子融合 residual + RMSNorm（第 8 章）
//
// 对应文档: ../cuda_rmsnorm_optimization_guide.md 第 8 章
//
// Transformer 层里 RMSNorm 从不单独出现，标准上下文是残差流:
//   h = x + residual        # 残差相加，h 还要作为新的 residual 传给下一层
//   y = RMSNorm(h) * gamma  # 归一化结果进入下一个子层
//
// 融合成一个 kernel：h 在寄存器里随算随用，只写不再读。
// 流量 5 遍 → 4 遍（省 20%），外加省一次 kernel 启动。
// 这正是 vLLM / SGLang 中 fused_add_rms_norm 的形态。
//
// 原地（in-place）语义:
//   x:        [N, H] 输入，kernel 结束后被覆写为归一化输出 y
//   residual: [N, H] 旧残差，kernel 结束后被覆写为新残差 h = x + residual
//
// 编译:
//   nvcc -O3 -arch=sm_70 fused_add_rmsnorm.cu -o fused_add_rmsnorm
// 运行:
//   ./fused_add_rmsnorm            # 默认 N=4096, H=4096

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// Warp / Block 两级归约（与 V1/V2 相同）
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warpReduceSum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

__device__ __forceinline__ float blockReduceSum(float v) {
    __shared__ float warpRes[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warpReduceSum(v);
    if (lane == 0) warpRes[wid] = v;
    __syncthreads();
    int nWarp = (blockDim.x + 31) >> 5;
    v = (lane < nWarp) ? warpRes[lane] : 0.f;
    if (wid == 0) v = warpReduceSum(v);
    __shared__ float result;
    if (threadIdx.x == 0) result = v;
    __syncthreads();
    return result;
}

// ===========================================================================
// V4：residual + RMSNorm 融合（vLLM/SGLang fused_add_rms_norm 的教学版）
// 要求 H = ITEMS * blockDim.x * 4
// ===========================================================================
template <int ITEMS>
__global__ void fused_add_rmsnorm(float* __restrict__ x, float* __restrict__ residual,
                                  const float* __restrict__ gamma, int H, float eps) {
    float4* xrow = reinterpret_cast<float4*>(x + (size_t)blockIdx.x * H);
    float4* rrow = reinterpret_cast<float4*>(residual + (size_t)blockIdx.x * H);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);

    float4 buf[ITEMS];
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        int i = threadIdx.x + k * blockDim.x;
        float4 a = xrow[i], b = rrow[i];
        float4 h;                                     // ① h = x + residual
        h.x = a.x + b.x;  h.y = a.y + b.y;
        h.z = a.z + b.z;  h.w = a.w + b.w;
        rrow[i] = h;                                  // ② 新残差写回（下一层要用）
        buf[k] = h;                                   //    同时驻留寄存器
        acc += h.x * h.x + h.y * h.y + h.z * h.z + h.w * h.w;
    }
    acc = blockReduceSum(acc);
    float rrms = rsqrtf(acc / H + eps);

    #pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        int i = threadIdx.x + k * blockDim.x;
        float4 h = buf[k], g = g4[i], o;              // ③ 归一化，h 来自寄存器
        o.x = h.x * rrms * g.x;  o.y = h.y * rrms * g.y;
        o.z = h.z * rrms * g.z;  o.w = h.w * rrms * g.w;
        xrow[i] = o;                                  // ④ 结果原地写回 x
    }
}

// ===========================================================================
// CPU 参考实现（double 累加）
//   h_ref = x + residual
//   y_ref = h_ref / rms(h_ref) * gamma
// ===========================================================================
static void fused_add_rmsnorm_cpu(const float* x, const float* residual,
                                  const float* gamma, float* h_out, float* y_out,
                                  int N, int H, float eps) {
    for (int r = 0; r < N; ++r) {
        const float* xr = x + (size_t)r * H;
        const float* rr = residual + (size_t)r * H;
        float* ho = h_out + (size_t)r * H;
        float* yo = y_out + (size_t)r * H;

        double ss = 0.0;
        for (int i = 0; i < H; ++i) {
            float h = xr[i] + rr[i];
            ho[i] = h;
            ss += (double)h * (double)h;
        }
        double rrms = 1.0 / sqrt(ss / H + (double)eps);
        for (int i = 0; i < H; ++i)
            yo[i] = (float)((double)ho[i] * rrms * (double)gamma[i]);
    }
}

// ===========================================================================
// 宿主端驱动
// ===========================================================================
#define ITEMS_DEFAULT 4    // H=4096, block=256 → ITEMS=4

static float max_abs_err(const float* a, const float* b, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    int H = (argc > 2) ? atoi(argv[2]) : 4096;
    const float eps = 1e-6f;

    if (H != ITEMS_DEFAULT * BLOCK_SIZE * 4) {
        fprintf(stderr, "本教学版要求 H = %d（ITEMS=%d, block=%d），当前 H=%d\n",
                ITEMS_DEFAULT * BLOCK_SIZE * 4, ITEMS_DEFAULT, BLOCK_SIZE, H);
        return EXIT_FAILURE;
    }

    printf("Fused add + RMSNorm  N = %d, H = %d, eps = %g\n\n", N, H, eps);

    size_t total = (size_t)N * H;
    size_t bytes = total * sizeof(float);

    // 主机数据：随机初始化
    float* h_x     = (float*)malloc(bytes);
    float* h_res   = (float*)malloc(bytes);
    float* h_gamma = (float*)malloc(H * sizeof(float));
    float* h_h_ref = (float*)malloc(bytes);   // 新残差参考
    float* h_y_ref = (float*)malloc(bytes);   // 归一化输出参考
    float* h_h_gpu = (float*)malloc(bytes);
    float* h_y_gpu = (float*)malloc(bytes);

    srand(1234);
    for (size_t i = 0; i < total; ++i) {
        h_x[i]   = ((float)rand() / RAND_MAX) * 2.f - 1.f;    // [-1, 1]
        h_res[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;
    }
    for (int i = 0; i < H; ++i)
        h_gamma[i] = ((float)rand() / RAND_MAX) + 0.5f;       // 非平凡 γ ∈ [0.5, 1.5]

    // CPU 参考
    fused_add_rmsnorm_cpu(h_x, h_res, h_gamma, h_h_ref, h_y_ref, N, H, eps);

    float *d_x, *d_res, *d_g;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_res, bytes));
    CUDA_CHECK(cudaMalloc(&d_g, H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_g, h_gamma, H * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热（原地改写，需每次重置输入）
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_res, h_res, bytes, cudaMemcpyHostToDevice));
    fused_add_rmsnorm<ITEMS_DEFAULT><<<N, BLOCK_SIZE>>>(d_x, d_res, d_g, H, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 正式计时（重置输入）
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_res, h_res, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(start));
    fused_add_rmsnorm<ITEMS_DEFAULT><<<N, BLOCK_SIZE>>>(d_x, d_res, d_g, H, eps);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // 原地语义：x -> y, residual -> h
    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_x, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_h_gpu, d_res, bytes, cudaMemcpyDeviceToHost));

    float err_h = max_abs_err(h_h_gpu, h_h_ref, total);   // 新残差
    float err_y = max_abs_err(h_y_gpu, h_y_ref, total);   // 归一化输出

    // 融合口径有效带宽：读 x、residual + 写 h、y = 4NH*4B
    double gbps = 4.0 * total * sizeof(float) / (ms * 1e-3) / 1e9;

    bool ok = (err_h < 1e-3f) && (err_y < 1e-3f);
    printf("%-18s %14s %14s %12s %10s\n", "kernel", "err_h(残差)", "err_y(输出)", "time(ms)", "GB/s");
    printf("--------------------------------------------------------------------------\n");
    printf("%-18s %14.2e %14.2e %12.4f %10.1f  %s\n",
           "fused_add_rmsnorm", err_h, err_y, ms, gbps, ok ? "OK" : "FAIL");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_g));
    free(h_x); free(h_res); free(h_gamma);
    free(h_h_ref); free(h_y_ref); free(h_h_gpu); free(h_y_gpu);

    printf("\n融合口径有效带宽: 读 x、residual + 写 h、y = 4*N*H*4B / t\n");
    return 0;
}
