// 第 13 章教学 kernel：用同一 fp32 计算核心隔离调度、布局与融合的作用。
#pragma once
#include <cuda_runtime.h>
#include "schedule.h"

namespace advanced_gemm {

struct Problem {
    const float *a, *b, *c, *bias;
    float* d;                       // 与 a/b/c/bias 不重叠，c 是只读旧值
    int m, n, k, lda, ldb, ldc, ldd;
    float alpha, beta;
};

template<bool UseBeta, bool Bias, bool Relu>
__device__ float epilogue(const Problem& p, int row, int col, float acc) {
    float out = p.alpha * acc;
    if constexpr (UseBeta) out += p.beta * p.c[static_cast<std::size_t>(row) * p.ldc + col];
    if constexpr (Bias) out += p.bias[col];
    if constexpr (Relu) out = fmaxf(out, 0.0f);
    return out;
}

template<int M, int N, int K, int TM_, int TN_>
struct Tile {
    static constexpr int BM = M, BN = N, BK = K, TM = TM_, TN = TN_;
    static constexpr int Threads = BM * BN / (TM * TN);
    static_assert(BM % TM == 0 && BN % TN == 0, "invalid thread tile");
    static_assert(Threads == 256, "examples use 256 threads per block");
    struct Shared { float a[BM * BK], b[BK * BN]; };
    struct Acc { float x[TM][TN]; };

    template<bool Packed, bool FastInterior>
    __device__ static void compute(const Problem& p, int m0, int n0,
                                    int k_begin, int k_end, Shared& smem, Acc& acc) {
        const int tr = (threadIdx.x / (BN / TN)) * TM;
        const int tc = (threadIdx.x % (BN / TN)) * TN;
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            #pragma unroll
            for (int j = 0; j < TN; ++j) acc.x[i][j] = 0.0f;

        for (int tile_k = k_begin; tile_k < k_end; ++tile_k) {
            const int k0 = tile_k * BK;
            // 对整个 Block 一致。FastInterior=false 保留逐元素 mask 基线。
            const bool full = FastInterior && m0 + BM <= p.m &&
                              n0 + BN <= p.n && k0 + BK <= p.k;
            for (int ix = threadIdx.x; ix < BM * BK; ix += Threads) {
                const int r = ix / BK, c = ix % BK;
                smem.a[ix] = (full || (m0 + r < p.m && k0 + c < p.k))
                    ? p.a[static_cast<std::size_t>(m0 + r) * p.lda + k0 + c] : 0.0f;
            }
            for (int ix = threadIdx.x; ix < BK * BN; ix += Threads) {
                const int r = ix / BN, c = ix % BN;
                if constexpr (Packed) {
                    // pack kernel 已对 K/N 尾部补零；每个 tile 总是完整可读。
                    smem.b[ix] = p.b[packed_b_index(n0 / BN, tile_k, r, c,
                                                   ceil_div(p.k, BK), BK, BN)];
                } else {
                    smem.b[ix] = (full || (k0 + r < p.k && n0 + c < p.n))
                        ? p.b[static_cast<std::size_t>(k0 + r) * p.ldb + n0 + c] : 0.0f;
                }
            }
            __syncthreads();             // 等加载完成
            #pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                float ra[TM], rb[TN];
                #pragma unroll
                for (int i = 0; i < TM; ++i) ra[i] = smem.a[(tr + i) * BK + kk];
                #pragma unroll
                for (int j = 0; j < TN; ++j) rb[j] = smem.b[kk * BN + tc + j];
                #pragma unroll
                for (int i = 0; i < TM; ++i)
                    #pragma unroll
                    for (int j = 0; j < TN; ++j)
                        acc.x[i][j] = fmaf(ra[i], rb[j], acc.x[i][j]);
            }
            __syncthreads();             // 等所有读取完成，才可覆盖 shared
        }
    }

    template<bool UseBeta, bool Bias, bool Relu>
    __device__ static void store(const Problem& p, int m0, int n0, const Acc& acc) {
        const int tr = (threadIdx.x / (BN / TN)) * TM;
        const int tc = (threadIdx.x % (BN / TN)) * TN;
        #pragma unroll
        for (int i = 0; i < TM; ++i)
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int r = m0 + tr + i, c = n0 + tc + j;
                if (r < p.m && c < p.n)
                    p.d[static_cast<std::size_t>(r) * p.ldd + c] =
                        epilogue<UseBeta, Bias, Relu>(p, r, c, acc.x[i][j]);
            }
    }
};

using Small = Tile<32, 64, 16, 2, 4>;
using Base = Tile<64, 64, 16, 4, 4>;
using Wide = Tile<64, 128, 16, 4, 8>;

template<class T, bool Persistent = false, bool Packed = false,
         bool FastInterior = true, bool UseBeta = true, bool Bias = false, bool Relu = false>
__global__ void gemm_kernel(Problem p, int group_rows = 1) {
    __shared__ typename T::Shared smem;
    typename T::Acc acc;
    const int mt = ceil_div(p.m, T::BM), nt = ceil_div(p.n, T::BN);
    for (int id = blockIdx.x; id < mt * nt; id += gridDim.x) {
        const TileCoord coord = tile_coord(id, mt, nt, group_rows);
        T::template compute<Packed, FastInterior>(p, coord.m * T::BM, coord.n * T::BN,
                                                  0, ceil_div(p.k, T::BK), smem, acc);
        T::template store<UseBeta, Bias, Relu>(p, coord.m * T::BM, coord.n * T::BN, acc);
        if constexpr (!Persistent) break;
        __syncthreads();                 // 所有线程一起进入下一个输出 tile
    }
}

template<class T>
__global__ void pack_b_kernel(Problem p, float* packed, std::size_t count) {
    for (std::size_t ix = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         ix < count; ix += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const int c = ix % T::BN;
        const int r = (ix / T::BN) % T::BK;
        const int kt = ceil_div(p.k, T::BK);
        const int tk = (ix / (T::BN * T::BK)) % kt;
        const int tn = ix / (static_cast<std::size_t>(T::BN) * T::BK * kt);
        const int row = tk * T::BK + r, col = tn * T::BN + c;
        packed[ix] = row < p.k && col < p.n
            ? p.b[static_cast<std::size_t>(row) * p.ldb + col] : 0.0f;
    }
}

template<class T>
__global__ void split_k_kernel(Problem p, int splits, float* partial) {
    __shared__ typename T::Shared smem;
    typename T::Acc acc;
    const int nt = ceil_div(p.n, T::BN);
    const int m0 = (blockIdx.x / nt) * T::BM, n0 = (blockIdx.x % nt) * T::BN;
    const Range r = partition(ceil_div(p.k, T::BK), blockIdx.z, splits);
    T::template compute<false, true>(p, m0, n0, r.begin, r.end, smem, acc);
    // 部分和始终原样写出；alpha/beta/bias/activation 留给归约阶段。
    p.d = partial + static_cast<std::size_t>(blockIdx.z) * p.m * p.n;
    p.ldd = p.n;
    p.alpha = 1.0f;
    T::template store<false, false, false>(p, m0, n0, acc);
}

template<bool Bias = false, bool Relu = false>
__global__ void split_reduce_kernel(Problem p, int splits, const float* partial) {
    const std::size_t count = static_cast<std::size_t>(p.m) * p.n;
    for (std::size_t ix = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         ix < count; ix += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        float sum = 0;
        for (int s = 0; s < splits; ++s) sum += partial[s * count + ix];
        const int r = ix / p.n, c = ix % p.n;
        p.d[static_cast<std::size_t>(r) * p.ldd + c] = epilogue<true, Bias, Relu>(p, r, c, sum);
    }
}

// Sliced-K 教学版：16×32 输出，256 线程分成 4 组，每组 2 个 Warp。
// 所有组共享 A/B，但每组只计算每个 BK=32 段中的 8 个 k，最后在 Block 内归约。
__global__ void sliced_k_kernel(Problem p) {
    constexpr int BM = 16, BN = 32, BK = 32, Parts = 4;
    __shared__ float a[BM * BK], b[BK * BN], partial[Parts * BM * BN];
    const int nt = ceil_div(p.n, BN);
    const int m0 = (blockIdx.x / nt) * BM, n0 = (blockIdx.x % nt) * BN;
    const int part = threadIdx.x / 64, local = threadIdx.x % 64;
    const int row_base = local / BN, col = local % BN;
    float acc[8] = {};
    for (int k0 = 0; k0 < p.k; k0 += BK) {
        for (int ix = threadIdx.x; ix < BM * BK; ix += 256) {
            const int r = ix / BK, c = ix % BK;
            a[ix] = m0 + r < p.m && k0 + c < p.k
                ? p.a[static_cast<std::size_t>(m0 + r) * p.lda + k0 + c] : 0;
        }
        for (int ix = threadIdx.x; ix < BK * BN; ix += 256) {
            const int r = ix / BN, c = ix % BN;
            b[ix] = k0 + r < p.k && n0 + c < p.n
                ? p.b[static_cast<std::size_t>(k0 + r) * p.ldb + n0 + c] : 0;
        }
        __syncthreads();
        for (int k = part * 8; k < (part + 1) * 8; ++k) {
            const float bv = b[k * BN + col];
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] = fmaf(a[(row_base + 2 * i) * BK + k], bv, acc[i]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) partial[(part * BM + row_base + 2 * i) * BN + col] = acc[i];
    __syncthreads();
    for (int ix = threadIdx.x; ix < BM * BN; ix += 256) {
        const int r = m0 + ix / BN, c = n0 + ix % BN;
        if (r < p.m && c < p.n) {
            float sum = 0;
            for (int s = 0; s < Parts; ++s) sum += partial[s * BM * BN + ix];
            p.d[static_cast<std::size_t>(r) * p.ldd + c] = epilogue<true, false, false>(p, r, c, sum);
        }
    }
}

template<class T>
__global__ void streamk_partials_kernel(Problem p, const Segment* segments,
                                       const int* worker_offsets, float* partial) {
    __shared__ typename T::Shared smem;
    typename T::Acc acc;
    const int nt = ceil_div(p.n, T::BN);
    for (int slot = worker_offsets[blockIdx.x]; slot < worker_offsets[blockIdx.x + 1]; ++slot) {
        const Segment task = segments[slot];
        T::template compute<false, true>(p, (task.tile / nt) * T::BM,
                                         (task.tile % nt) * T::BN,
                                         task.k_begin, task.k_end, smem, acc);
        const int tr = (threadIdx.x / (T::BN / T::TN)) * T::TM;
        const int tc = (threadIdx.x % (T::BN / T::TN)) * T::TN;
        #pragma unroll
        for (int i = 0; i < T::TM; ++i)
            #pragma unroll
            for (int j = 0; j < T::TN; ++j)
                partial[(static_cast<std::size_t>(slot) * T::BM + tr + i) * T::BN + tc + j]
                    = acc.x[i][j];
        __syncthreads();
    }
}

template<class T>
__global__ void streamk_reduce_kernel(Problem p, const float* partial,
                                     const int* tile_offsets, const int* tile_slots) {
    const std::size_t count = static_cast<std::size_t>(p.m) * p.n;
    const int nt = ceil_div(p.n, T::BN);
    for (std::size_t ix = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         ix < count; ix += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const int r = ix / p.n, c = ix % p.n;
        const int tile = (r / T::BM) * nt + c / T::BN;
        const int local = (r % T::BM) * T::BN + c % T::BN;
        float sum = 0;
        for (int j = tile_offsets[tile]; j < tile_offsets[tile + 1]; ++j)
            sum += partial[static_cast<std::size_t>(tile_slots[j]) * T::BM * T::BN + local];
        p.d[static_cast<std::size_t>(r) * p.ldd + c] = epilogue<true, false, false>(p, r, c, sum);
    }
}

struct BatchStrides { std::size_t a, b, c, d, bias; };

__host__ __device__ inline Problem batch_problem(Problem p, BatchStrides s, int batch) {
    p.a += batch * s.a; p.b += batch * s.b; p.c += batch * s.c;
    p.d += batch * s.d; p.bias += batch * s.bias;
    return p;
}

template<class T>
__global__ void batched_kernel(Problem base, BatchStrides strides) {
    const Problem p = batch_problem(base, strides, blockIdx.z);
    __shared__ typename T::Shared smem;
    typename T::Acc acc;
    const int nt = ceil_div(p.n, T::BN);
    const int m0 = (blockIdx.x / nt) * T::BM, n0 = (blockIdx.x % nt) * T::BN;
    T::template compute<false, true>(p, m0, n0, 0, ceil_div(p.k, T::BK), smem, acc);
    T::template store<true, false, false>(p, m0, n0, acc);
}

template<class T>
__global__ void grouped_kernel(const Problem* problems, const int* prefix, int count) {
    __shared__ typename T::Shared smem;
    typename T::Acc acc;
    for (int id = blockIdx.x; id < prefix[count]; id += gridDim.x) {
        const int g = find_problem(id, prefix, count);
        const Problem p = problems[g];
        const int local = id - prefix[g], nt = ceil_div(p.n, T::BN);
        const int m0 = (local / nt) * T::BM, n0 = (local % nt) * T::BN;
        T::template compute<false, true>(p, m0, n0, 0, ceil_div(p.k, T::BK), smem, acc);
        T::template store<true, false, false>(p, m0, n0, acc);
        __syncthreads();
    }
}

__global__ void bias_kernel(Problem p) {
    const std::size_t count = static_cast<std::size_t>(p.m) * p.n;
    for (std::size_t ix = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         ix < count; ix += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const int r = ix / p.n, c = ix % p.n;
        p.d[static_cast<std::size_t>(r) * p.ldd + c] += p.bias[c];
    }
}

__global__ void relu_kernel(Problem p) {
    const std::size_t count = static_cast<std::size_t>(p.m) * p.n;
    for (std::size_t ix = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         ix < count; ix += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const std::size_t address = (ix / p.n) * p.ldd + ix % p.n;
        p.d[address] = fmaxf(p.d[address], 0.0f);
    }
}

}  // namespace advanced_gemm
