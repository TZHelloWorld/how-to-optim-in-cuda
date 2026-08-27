// flash_decoding.cu — Flash-Decoding 教学版（推理 decode 阶段的 attention）
//
// 对应文档: ../cuda_attention_optimization_guide.md 第 9 章
// 场景: 自回归生成的 decode 阶段——每步只有 1 行 query，对全部历史 KV Cache 做 attention。
//   并行度塌缩（Q 只有 1 行 → Block 太少）时，把唯一能切的 KV 序列维切给多个 Block，
//   各 Block 输出部分结果 (m, l, õ)，再由一个轻量 kernel 归并。
//
// 两个 kernel:
//   Kernel 1 decode_partial：每个 Block（1 个 Warp）负责一段 KV，输出本段部分结果
//   Kernel 2 decode_reduce ：每个 Block（1 个 Warp）归并一份 (batch,head) 的 S 份部分结果
//
// q: [BH, D]，K/V: [BH, N, D]（KV Cache），out: [BH, D]
// 工作区 Mp/Lp: [BH, S]，Op: [BH, S, D]
//
// 编译:
//   nvcc -O3 -arch=sm_70 flash_decoding.cu -o flash_decoding
// 运行:
//   ./flash_decoding                 # 默认 BH=8, N=2048, D=128, S=16
//   ./flash_decoding 8 4096 128 32   # 指定 BH N D S

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

// ---------------------------------------------------------------------------
// Kernel 1：每个 Block（1 个 Warp）负责一段 KV，输出本段部分结果 (m, l, õ)
// ---------------------------------------------------------------------------
template <int D>
__global__ void decode_partial(const float* q, const float* K, const float* V,
                               float* Mp, float* Lp, float* Op,
                               int N, int S, float scale) {
    constexpr int DL = D / 32;            // 每 lane 分担的 head 维分量数
    int lane = threadIdx.x;               // blockDim = 32：一个 Warp
    int seg  = blockIdx.x;                // 段编号 0..S-1
    int bh   = blockIdx.y;                // batch*head 编号

    int len   = (N + S - 1) / S;          // 每段长度（向上取整）
    int begin = seg * len;
    int end   = min(begin + len, N);

    const float* Kb = K + (size_t)bh * N * D;
    const float* Vb = V + (size_t)bh * N * D;

    // q 与 õ 沿 head 维切给 32 个 lane
    float qr[DL], o[DL];
    for (int x = 0; x < DL; x++) { qr[x] = q[(size_t)bh * D + x * 32 + lane]; o[x] = 0.0f; }
    float m = -INFINITY, l = 0.0f;

    for (int c = begin; c < end; c++) {   // 流式扫过本段的 key
        float s = 0.0f;                   // Warp 级并行点积 + 蝶形归约
        for (int x = 0; x < DL; x++) s += qr[x] * Kb[(size_t)c * D + x * 32 + lane];
        for (int d2 = 16; d2 > 0; d2 >>= 1)
            s += __shfl_xor_sync(0xffffffff, s, d2);
        s *= scale;

        float m_new = fmaxf(m, s);        // 逐元素在线递推（块大小 = 1）
        float p     = __expf(s - m_new);
        float corr  = __expf(m - m_new);
        l = l * corr + p;
        for (int x = 0; x < DL; x++)
            o[x] = o[x] * corr + p * Vb[(size_t)c * D + x * 32 + lane];
        m = m_new;
    }

    // 写出部分结果（未归一化：归一化了归并时还得乘回去）
    size_t ws = (size_t)bh * S + seg;
    if (lane == 0) { Mp[ws] = m; Lp[ws] = l; }    // m/l 各 lane 一致，一个人写即可
    for (int x = 0; x < DL; x++)
        Op[ws * D + x * 32 + lane] = o[x];
}

// ---------------------------------------------------------------------------
// Kernel 2：每个 Block（1 个 Warp）归并一个 (batch, head) 的 S 份部分结果
// ---------------------------------------------------------------------------
template <int D>
__global__ void decode_reduce(const float* Mp, const float* Lp, const float* Op,
                              float* out, int S) {
    constexpr int DL = D / 32;
    int lane = threadIdx.x;
    int bh   = blockIdx.x;

    float m = -INFINITY, l = 0.0f;
    float o[DL];
    for (int x = 0; x < DL; x++) o[x] = 0.0f;

    for (int s = 0; s < S; s++) {                  // 逐段两两归并（9.3 的公式）
        size_t ws = (size_t)bh * S + s;
        float ms = Mp[ws], ls = Lp[ws];
        float m_new = fmaxf(m, ms);
        float c_old = __expf(m - m_new);           // 旧累积的补偿系数
        float c_new = __expf(ms - m_new);          // 新一段的补偿系数
        l = l * c_old + ls * c_new;
        for (int x = 0; x < DL; x++)
            o[x] = o[x] * c_old + Op[ws * D + x * 32 + lane] * c_new;
        m = m_new;
    }

    float inv_l = 1.0f / l;                        // 全程唯一一次除法
    for (int x = 0; x < DL; x++)
        out[(size_t)bh * D + x * 32 + lane] = o[x] * inv_l;
}

// ---------------------------------------------------------------------------
// CPU 参考：单行 query 对全部 N 个 key/value 做 safe softmax attention
// ---------------------------------------------------------------------------
static void cpu_reference(const float* q, const float* K, const float* V,
                          float* out, int BH, int N, int D) {
    float scale = 1.0f / sqrtf((float)D);
    float* s = (float*)malloc(sizeof(float) * N);
    for (int bh = 0; bh < BH; bh++) {
        const float* qb = q + (size_t)bh * D;
        const float* Kb = K + (size_t)bh * N * D;
        const float* Vb = V + (size_t)bh * N * D;
        float m = -INFINITY;
        for (int c = 0; c < N; c++) {
            float acc = 0.0f;
            for (int x = 0; x < D; x++) acc += qb[x] * Kb[(size_t)c * D + x];
            s[c] = acc * scale;
            m = fmaxf(m, s[c]);
        }
        float l = 0.0f;
        for (int c = 0; c < N; c++) { s[c] = expf(s[c] - m); l += s[c]; }
        for (int x = 0; x < D; x++) {
            float acc = 0.0f;
            for (int c = 0; c < N; c++) acc += (s[c] / l) * Vb[(size_t)c * D + x];
            out[(size_t)bh * D + x] = acc;
        }
    }
    free(s);
}

static float max_abs_diff(const float* a, const float* b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) mx = fmaxf(mx, fabsf(a[i] - b[i]));
    return mx;
}

int main(int argc, char** argv) {
    int BH = (argc > 1) ? atoi(argv[1]) : 8;
    int N  = (argc > 2) ? atoi(argv[2]) : 2048;
    int D  = (argc > 3) ? atoi(argv[3]) : 128;
    int S  = (argc > 4) ? atoi(argv[4]) : 16;

    constexpr int TD = 128;              // 模板固定 D=128（须为 32 的倍数）
    if (D != TD) {
        fprintf(stderr, "本 demo 的模板 kernel 固定 D=%d（可修改模板实参）。\n", TD);
        return 1;
    }
    if (N < S) { fprintf(stderr, "要求 N >= S（每段非空）。\n"); return 1; }

    printf("Flash-Decoding: BH=%d, N=%d, D=%d, S=%d\n\n", BH, N, D, S);
    float scale = 1.0f / sqrtf((float)D);

    size_t szQ  = (size_t)BH * D;
    size_t szKV = (size_t)BH * N * D;
    size_t szWp = (size_t)BH * S;         // Mp / Lp
    size_t szOp = (size_t)BH * S * D;     // Op

    float* h_q   = (float*)malloc(sizeof(float) * szQ);
    float* h_K   = (float*)malloc(sizeof(float) * szKV);
    float* h_V   = (float*)malloc(sizeof(float) * szKV);
    float* h_out = (float*)malloc(sizeof(float) * szQ);
    float* h_ref = (float*)malloc(sizeof(float) * szQ);
    srand(0);
    for (size_t i = 0; i < szQ;  i++) h_q[i] = (float)rand() / RAND_MAX - 0.5f;
    for (size_t i = 0; i < szKV; i++) { h_K[i] = (float)rand() / RAND_MAX - 0.5f;
                                        h_V[i] = (float)rand() / RAND_MAX - 0.5f; }

    cpu_reference(h_q, h_K, h_V, h_ref, BH, N, D);

    float *d_q, *d_K, *d_V, *d_out, *d_Mp, *d_Lp, *d_Op;
    CUDA_CHECK(cudaMalloc(&d_q,   sizeof(float) * szQ));
    CUDA_CHECK(cudaMalloc(&d_K,   sizeof(float) * szKV));
    CUDA_CHECK(cudaMalloc(&d_V,   sizeof(float) * szKV));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * szQ));
    CUDA_CHECK(cudaMalloc(&d_Mp,  sizeof(float) * szWp));
    CUDA_CHECK(cudaMalloc(&d_Lp,  sizeof(float) * szWp));
    CUDA_CHECK(cudaMalloc(&d_Op,  sizeof(float) * szOp));
    CUDA_CHECK(cudaMemcpy(d_q, h_q, sizeof(float) * szQ,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, sizeof(float) * szKV, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, sizeof(float) * szKV, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    decode_partial<TD><<<dim3(S, BH), 32>>>(d_q, d_K, d_V, d_Mp, d_Lp, d_Op, N, S, scale);
    decode_reduce <TD><<<BH, 32>>>(d_Mp, d_Lp, d_Op, d_out, S);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(float) * szQ, cudaMemcpyDeviceToHost));
    float diff = max_abs_diff(h_out, h_ref, szQ);

    printf("time(ms) = %.4f\n", ms);
    printf("max|diff| vs CPU reference = %.2e  %s\n", diff, diff < 1e-3f ? "OK" : "FAIL");
    printf("\nGrid: kernel1 = S×BH = %d Blocks（对症解决并行度塌缩）; kernel2 = BH = %d Blocks\n",
           S * BH, BH);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_K)); CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_Mp)); CUDA_CHECK(cudaFree(d_Lp));
    CUDA_CHECK(cudaFree(d_Op));
    free(h_q); free(h_K); free(h_V); free(h_out); free(h_ref);
    return 0;
}
