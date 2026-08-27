// layernorm.cu — CUDA LayerNorm 从 V0 到 V4（含 Welford 稳健版）的完整可运行实现
//
// 对应文档: ../cuda_layernorm_optimization_guide.md
// 覆盖版本:
//   V0 基准实现   —— 一行一 Block；两遍统计 + 共享内存树形归约（第 4 章）
//   V1 单遍化     —— 一次扫描同时累加 (Σx, Σx²)，树形归约（第 5 章）
//   V2 Warp Shuffle 两级归约（float2 成对归约）（第 6 章）
//   V3 float4 向量化（第 7 章）
//   V4 行驻留寄存器 —— 显存只读一遍，达理论下限（第 8 章）
//   Welford 稳健版 —— 换合并算子，归约骨架不变（第 9 章）
//
// LayerNorm 沿最后一维（hidden 维 H）逐行归一化 + 仿射:
//   y = (x - mean) / sqrt(var + eps) * gamma + beta
//
// 编译:
//   nvcc -O3 -arch=sm_70 layernorm.cu -o layernorm
// 运行:
//   ./layernorm            # 默认 N=4096, H=4096
//   ./layernorm 8192 2048  # 指定 N H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <utility>   // std::swap
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

// ===========================================================================
// V0：基准实现——一行一 Block；两遍统计 + 共享内存树形归约（第 4 章）
// 启动：layernorm_v0<<<N, 256, 256 * sizeof(float)>>>(x, y, gamma, beta, H, 1e-5f);
// ===========================================================================
__global__ void layernorm_v0(const float* __restrict__ x, float* __restrict__ y,
                             const float* __restrict__ gamma, const float* __restrict__ beta,
                             int H, float eps) {
    extern __shared__ float sdata[];                  // blockDim.x 个 float
    const float* row = x + (size_t)blockIdx.x * H;    // 本 Block 负责的行
    float*       out = y + (size_t)blockIdx.x * H;
    int tid = threadIdx.x;

    // ---- pass 1：求均值 ----
    float acc = 0.f;
    for (int i = tid; i < H; i += blockDim.x)         // 跨步循环：warp 内地址连续（合并访存）
        acc += row[i];
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {    // 树形归约：每轮活跃线程减半
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    __shared__ float s_mean;
    if (tid == 0) s_mean = sdata[0] / H;
    __syncthreads();
    float mean = s_mean;                              // 广播给全 Block

    // ---- pass 2：求方差（再读一遍 x）----
    acc = 0.f;
    for (int i = tid; i < H; i += blockDim.x) {
        float d = row[i] - mean;
        acc += d * d;
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    __shared__ float s_rstd;
    if (tid == 0) s_rstd = rsqrtf(sdata[0] / H + eps);
    __syncthreads();
    float rstd = s_rstd;

    // ---- pass 3：归一化 + 仿射（第三遍读 x）----
    for (int i = tid; i < H; i += blockDim.x)
        out[i] = (row[i] - mean) * rstd * gamma[i] + beta[i];
}

// ===========================================================================
// V1：单遍统计——一次扫描同时累加 (Σx, Σx²)，树形归约（第 5 章）
// 启动：layernorm_v1<<<N, 256, 256 * sizeof(float2)>>>(x, y, gamma, beta, H, 1e-5f);
// ===========================================================================
__global__ void layernorm_v1(const float* __restrict__ x, float* __restrict__ y,
                             const float* __restrict__ gamma, const float* __restrict__ beta,
                             int H, float eps) {
    extern __shared__ float2 sdata2[];                // blockDim.x 个 float2
    const float* row = x + (size_t)blockIdx.x * H;
    float*       out = y + (size_t)blockIdx.x * H;
    int tid = threadIdx.x;

    // 一遍扫描，双累加
    float2 acc = make_float2(0.f, 0.f);               // (Σx, Σx²)
    for (int i = tid; i < H; i += blockDim.x) {
        float v = row[i];
        acc.x += v;
        acc.y += v * v;
    }
    sdata2[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {    // 树形归约（float2 一起归）
        if (tid < s) {
            sdata2[tid].x += sdata2[tid + s].x;
            sdata2[tid].y += sdata2[tid + s].y;
        }
        __syncthreads();
    }

    __shared__ float s_mean, s_rstd;
    if (tid == 0) {
        float mean = sdata2[0].x / H;
        float var  = fmaxf(sdata2[0].y / H - mean * mean, 0.f);  // 下界保护（2.2 节）
        s_mean = mean;
        s_rstd = rsqrtf(var + eps);
    }
    __syncthreads();
    float mean = s_mean, rstd = s_rstd;

    for (int i = tid; i < H; i += blockDim.x)
        out[i] = (row[i] - mean) * rstd * gamma[i] + beta[i];
}

// ===========================================================================
// V2：Warp Shuffle 两级归约（第 6 章）
// (Σx, Σx²) 成对的 warp / block 归约组件
// ===========================================================================
__device__ __forceinline__ float2 warpReduceSum2(float2 v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v.x += __shfl_xor_sync(0xffffffff, v.x, offset);
        v.y += __shfl_xor_sync(0xffffffff, v.y, offset);
    }
    return v;                                          // 32 个 lane 都持有完整和
}

__device__ __forceinline__ float2 blockReduceSum2(float2 v) {
    __shared__ float2 warpRes[32];                     // 至多 32 个 Warp
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warpReduceSum2(v);                             // 第一级：Warp 内
    if (lane == 0) warpRes[wid] = v;
    __syncthreads();
    int nWarp = (blockDim.x + 31) >> 5;
    v = (lane < nWarp) ? warpRes[lane] : make_float2(0.f, 0.f);
    if (wid == 0) v = warpReduceSum2(v);               // 第二级：Warp 间
    __shared__ float2 result;
    if (threadIdx.x == 0) result = v;
    __syncthreads();                                   // 广播给全 Block
    return result;
}

// V2：归约换成两级 shuffle，主体骤然清爽
// 启动：layernorm_v2<<<N, 256>>>(x, y, gamma, beta, H, 1e-5f);
__global__ void layernorm_v2(const float* __restrict__ x, float* __restrict__ y,
                             const float* __restrict__ gamma, const float* __restrict__ beta,
                             int H, float eps) {
    const float* row = x + (size_t)blockIdx.x * H;
    float*       out = y + (size_t)blockIdx.x * H;

    float2 acc = make_float2(0.f, 0.f);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float v = row[i];
        acc.x += v;
        acc.y += v * v;
    }
    acc = blockReduceSum2(acc);                        // 一行代码完成全部归约

    float mean = acc.x / H;
    float rstd = rsqrtf(fmaxf(acc.y / H - mean * mean, 0.f) + eps);

    for (int i = threadIdx.x; i < H; i += blockDim.x)
        out[i] = (row[i] - mean) * rstd * gamma[i] + beta[i];
}

// ===========================================================================
// V3：float4 向量化（第 7 章）
// 要求 H % 4 == 0，且指针 16B 对齐（cudaMalloc 返回的指针满足对齐）
// 启动：layernorm_v3<<<N, 256>>>(x, y, gamma, beta, H, 1e-5f);
// ===========================================================================
__global__ void layernorm_v3(const float* __restrict__ x, float* __restrict__ y,
                             const float* __restrict__ gamma, const float* __restrict__ beta,
                             int H, float eps) {
    const float4* row4 = reinterpret_cast<const float4*>(x + (size_t)blockIdx.x * H);
    float4*       out4 = reinterpret_cast<float4*>(y + (size_t)blockIdx.x * H);
    const float4* g4   = reinterpret_cast<const float4*>(gamma);
    const float4* b4   = reinterpret_cast<const float4*>(beta);
    int H4 = H >> 2;

    float2 acc = make_float2(0.f, 0.f);
    for (int i = threadIdx.x; i < H4; i += blockDim.x) {
        float4 v = row4[i];                                    // 一条指令搬 16B
        acc.x += v.x + v.y + v.z + v.w;
        acc.y += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    acc = blockReduceSum2(acc);

    float mean = acc.x / H;
    float rstd = rsqrtf(fmaxf(acc.y / H - mean * mean, 0.f) + eps);

    for (int i = threadIdx.x; i < H4; i += blockDim.x) {
        float4 v = row4[i], g = g4[i], b = b4[i], o;
        o.x = (v.x - mean) * rstd * g.x + b.x;
        o.y = (v.y - mean) * rstd * g.y + b.y;
        o.z = (v.z - mean) * rstd * g.z + b.z;
        o.w = (v.w - mean) * rstd * g.w + b.w;
        out4[i] = o;                                           // 一条指令写 16B
    }
}

// ===========================================================================
// V4：行驻留寄存器——加载时顺手累加，归一化直接用寄存器数据（第 8 章）
// 要求 H = ITEMS * blockDim.x（H=4096, block=256 → ITEMS=16）
// 启动：layernorm_v4<16><<<N, 256>>>(x, y, gamma, beta, H, eps);
// ===========================================================================
template <int ITEMS>
__global__ void layernorm_v4(const float* __restrict__ x, float* __restrict__ y,
                             const float* __restrict__ gamma, const float* __restrict__ beta,
                             int H, float eps) {
    const float* row = x + (size_t)blockIdx.x * H;
    float*       out = y + (size_t)blockIdx.x * H;

    float buf[ITEMS];                                  // 本线程名下的元素，驻留寄存器
    float2 acc = make_float2(0.f, 0.f);
    #pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        int i = threadIdx.x + k * blockDim.x;          // 同一轮各线程地址连续 → 合并访存
        buf[k] = row[i];                               // 唯一一次显存读
        acc.x += buf[k];
        acc.y += buf[k] * buf[k];                      // 加载时顺手累加统计量
    }
    acc = blockReduceSum2(acc);

    float mean = acc.x / H;
    float rstd = rsqrtf(fmaxf(acc.y / H - mean * mean, 0.f) + eps);

    #pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        int i = threadIdx.x + k * blockDim.x;
        out[i] = (buf[k] - mean) * rstd * gamma[i] + beta[i];  // 数据来自寄存器，不再读 x
    }
}

// ===========================================================================
// Welford 稳健版（第 9 章）
// 换合并算子，归约骨架复用；数值稳健的单遍方差
// ===========================================================================
__device__ __forceinline__ void welfordUpdate(float v, float& mean, float& m2, float& n) {
    n += 1.f;
    float d = v - mean;
    mean += d / n;
    m2 += d * (v - mean);                              // 注意用更新后的 mean
}

__device__ __forceinline__ void welfordMerge(float& mean, float& m2, float& n,
                                             float mean_b, float m2_b, float n_b) {
    float n_ab = n + n_b;
    if (n_ab == 0.f) return;
    float d = mean_b - mean;                           // 2.3 节的并行合并公式
    mean += d * (n_b / n_ab);
    m2   += m2_b + d * d * (n * n_b / n_ab);
    n = n_ab;
}

__device__ __forceinline__ void warpWelford(float& mean, float& m2, float& n) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {  // 蝶形：合并操作换成 welfordMerge
        float mb = __shfl_xor_sync(0xffffffff, mean, offset);
        float sb = __shfl_xor_sync(0xffffffff, m2,   offset);
        float nb = __shfl_xor_sync(0xffffffff, n,    offset);
        welfordMerge(mean, m2, n, mb, sb, nb);
    }
}

// Block 级 Welford：结构与 blockReduceSum2 相同（各 Warp → 共享内存 → Warp 0 再合并），
// 只是元素从 float2 换成 (mean, m2, n) 三元组，合并用 welfordMerge。
__device__ __forceinline__ void blockWelford(float& mean, float& m2, float& n) {
    __shared__ float s_mean[32], s_m2[32], s_n[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    warpWelford(mean, m2, n);                          // 第一级：Warp 内
    if (lane == 0) { s_mean[wid] = mean; s_m2[wid] = m2; s_n[wid] = n; }
    __syncthreads();
    int nWarp = (blockDim.x + 31) >> 5;
    if (lane < nWarp) { mean = s_mean[lane]; m2 = s_m2[lane]; n = s_n[lane]; }
    else              { mean = 0.f; m2 = 0.f; n = 0.f; }
    if (wid == 0) warpWelford(mean, m2, n);            // 第二级：Warp 间
    __shared__ float r_mean, r_m2, r_n;
    if (threadIdx.x == 0) { r_mean = mean; r_m2 = m2; r_n = n; }
    __syncthreads();                                   // 广播给全 Block
    mean = r_mean; m2 = r_m2; n = r_n;
}

// Welford kernel：线程内串行 update，线程间蝶形 merge；var = m2 / n
// 启动：layernorm_welford<<<N, 256>>>(x, y, gamma, beta, H, 1e-5f);
__global__ void layernorm_welford(const float* __restrict__ x, float* __restrict__ y,
                                  const float* __restrict__ gamma, const float* __restrict__ beta,
                                  int H, float eps) {
    const float* row = x + (size_t)blockIdx.x * H;
    float*       out = y + (size_t)blockIdx.x * H;

    float mean = 0.f, m2 = 0.f, n = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        welfordUpdate(row[i], mean, m2, n);
    blockWelford(mean, m2, n);                          // 归约结构复用

    float var  = m2 / n;
    float rstd = rsqrtf(var + eps);

    for (int i = threadIdx.x; i < H; i += blockDim.x)
        out[i] = (row[i] - mean) * rstd * gamma[i] + beta[i];
}

// ===========================================================================
// 宿主端驱动：CPU 参考实现（double 累加当金标准）+ 校验 + cudaEvent 计时
// ===========================================================================

enum Version { V0, V1, V2, V3, V4, WELFORD };

// 依据版本发起一次 kernel 启动
static void launch_once(Version v, const float* x, float* y,
                        const float* g, const float* b, int N, int H, float eps) {
    switch (v) {
        case V0:
            layernorm_v0<<<N, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(x, y, g, b, H, eps);
            break;
        case V1:
            layernorm_v1<<<N, BLOCK_SIZE, BLOCK_SIZE * sizeof(float2)>>>(x, y, g, b, H, eps);
            break;
        case V2:
            layernorm_v2<<<N, BLOCK_SIZE>>>(x, y, g, b, H, eps);
            break;
        case V3:
            layernorm_v3<<<N, BLOCK_SIZE>>>(x, y, g, b, H, eps);
            break;
        case V4:
            // V4 要求 H = ITEMS * blockDim.x；ITEMS 是编译期常量。
            // 这里针对 H = 16 * 256 = 4096 的默认配置实例化 ITEMS=16。
            if (H == 16 * BLOCK_SIZE)
                layernorm_v4<16><<<N, BLOCK_SIZE>>>(x, y, g, b, H, eps);
            else if (H == 8 * BLOCK_SIZE)
                layernorm_v4<8><<<N, BLOCK_SIZE>>>(x, y, g, b, H, eps);
            else if (H == 4 * BLOCK_SIZE)
                layernorm_v4<4><<<N, BLOCK_SIZE>>>(x, y, g, b, H, eps);
            // 其它 H 不满足 ITEMS 编译期常量约束，跳过（在 main 中标注 SKIP）
            break;
        case WELFORD:
            layernorm_welford<<<N, BLOCK_SIZE>>>(x, y, g, b, H, eps);
            break;
    }
}

// V4 是否支持给定的 H（ITEMS 必须是编译期常量且 H = ITEMS * BLOCK_SIZE）
static bool v4_supported(int H) {
    return H == 16 * BLOCK_SIZE || H == 8 * BLOCK_SIZE || H == 4 * BLOCK_SIZE;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 4096;
    int H = (argc > 2) ? atoi(argv[2]) : 4096;
    const float eps = 1e-5f;
    const size_t sz = (size_t)N * H;

    printf("LayerNorm over [N=%d, H=%d], eps=%g\n\n", N, H, eps);

    if (H % 4 != 0)
        printf("警告: H 不是 4 的倍数，V3（float4）不适用，将跳过。\n\n");

    // ---- 主机数据：必须用随机数据（全同值输入方差为 0，测不出统计错误）----
    float* hx   = new float[sz];
    float* hy   = new float[sz];
    float* href = new float[sz];
    float* hg   = new float[H];
    float* hb   = new float[H];
    for (size_t i = 0; i < sz; i++) hx[i] = (rand() % 2000 - 1000) / 500.f;
    for (int i = 0; i < H; i++) { hg[i] = 1.f + (i % 3) * 0.1f; hb[i] = (i % 5) * 0.01f; }

    // ---- CPU 参考（double 累加当金标准）----
    for (int r = 0; r < N; r++) {
        double s = 0, ss = 0;
        for (int i = 0; i < H; i++) s += hx[(size_t)r * H + i];
        double mean = s / H;
        for (int i = 0; i < H; i++) { double d = hx[(size_t)r * H + i] - mean; ss += d * d; }
        double rstd = 1.0 / sqrt(ss / H + eps);
        for (int i = 0; i < H; i++)
            href[(size_t)r * H + i] =
                (float)((hx[(size_t)r * H + i] - mean) * rstd * hg[i] + hb[i]);
    }

    // ---- 设备内存 ----
    float *x, *y, *g, *b;
    CUDA_CHECK(cudaMalloc(&x, sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y, sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g, H  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, H  * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(x, hx, sz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g, hg, H  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b, hb, H  * sizeof(float), cudaMemcpyHostToDevice));

    const char* names[] = {"V0", "V1", "V2", "V3", "V4", "Welford"};
    Version vers[]      = {V0, V1, V2, V3, V4, WELFORD};

    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));

    const int ITERS = 100;

    printf("%-8s %12s %12s %12s   %s\n", "Ver", "max_err", "time(ms)", "GB/s", "校验");
    printf("------------------------------------------------------------------\n");

    for (int vi = 0; vi < 6; vi++) {
        Version v = vers[vi];

        // 跳过不适用的版本
        if (v == V3 && (H % 4 != 0)) {
            printf("%-8s %12s %12s %12s   %s\n", names[vi], "-", "-", "-", "SKIP (H%4!=0)");
            continue;
        }
        if (v == V4 && !v4_supported(H)) {
            printf("%-8s %12s %12s %12s   %s\n", names[vi], "-", "-", "-",
                   "SKIP (H!=ITEMS*256)");
            continue;
        }

        // ---- 校验：跑一次，逐元素对比 CPU 参考 ----
        CUDA_CHECK(cudaMemset(y, 0, sz * sizeof(float)));
        launch_once(v, x, y, g, b, N, H, eps);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hy, y, sz * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err = 0.f;
        for (size_t i = 0; i < sz; i++)
            max_err = fmaxf(max_err, fabsf(hy[i] - href[i]));

        // ---- 计时：重复 ITERS 次取平均 ----
        CUDA_CHECK(cudaEventRecord(beg));
        for (int it = 0; it < ITERS; it++)
            launch_once(v, x, y, g, b, N, H, eps);
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));

        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
        ms /= ITERS;

        // 有效带宽 = 最少流量 (读 x + 写 y = 2NH*4B) / 时间
        double gbps = 2.0 * (double)sz * sizeof(float) / (ms * 1e-3) / 1e9;

        bool ok = (max_err < 1e-3f);   // fp32 单遍 naive 预期 ~1e-5 量级，放宽到 1e-3
        printf("%-8s %12.3e %12.4f %12.1f   %s\n",
               names[vi], max_err, ms, gbps, ok ? "OK" : "FAIL");
    }

    printf("\n提示: 有效带宽按最少流量 2NH*4B 计；V0 因多读显存实际带宽更低。\n");

    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(g));
    CUDA_CHECK(cudaFree(b));
    delete[] hx; delete[] hy; delete[] href; delete[] hg; delete[] hb;
    return 0;
}
