// attention.cu — CUDA Attention（Scaled Dot-Product Attention）从 V0 到 V4 的完整可运行实现
//
// 对应文档: ../cuda_attention_optimization_guide.md
// 覆盖版本:
//   V0 朴素三 kernel（QKᵀ → softmax → PV），N×N 中间矩阵落地 HBM（基准）
//   V1 融合 scale + mask + softmax 为单 kernel（一行一 Block + 两级归约）
//   V3 FlashAttention 教学版（K/V 分块驻留片上，(m,l,õ) 在线递推；每线程一行）
//   V4 FlashAttention-2 教学版（split-Q：一个 Warp 负责一行，lane 分摊 head 维）
//
// 说明:
//   - 本文件为单头（single head）版式，Q/K/V/O 形状为 [N, D]（V3/V4 支持 [BH, N, D]，本 demo BH=1）；
//   - CPU 参考实现 (cpu_reference) 做正确性校验；cudaEvent 计时各 kernel。
//   - 数学推导（online softmax 递推、分块合并、延迟归一化）见文档第 6、7、8 章。
//
// 编译:
//   nvcc -O3 -arch=sm_70 attention.cu -o attention
// 运行:
//   ./attention                 # 默认 N=1024, D=64, causal=1
//   ./attention 2048 64 1       # 指定 N D causal

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
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
// 2.4 节：两级归约组件（Warp 内蝶形归约 + Block 内经共享内存汇总）
// ===========================================================================

__device__ float warpReduceMax(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    return v;
}

__device__ float warpReduceSum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

__device__ float blockReduceMax(float v) {
    __shared__ float warpRes[32];
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    v = warpReduceMax(v);                                  // 步骤 1：Warp 内归约
    if (lane == 0) warpRes[wid] = v;                       // 步骤 2：代表写共享内存
    __syncthreads();

    int nWarp = (blockDim.x + 31) / 32;                    // 步骤 3：Warp 0 收尾归约
    v = (lane < nWarp) ? warpRes[lane] : -INFINITY;
    if (wid == 0) v = warpReduceMax(v);

    __shared__ float result;                               // 步骤 4：广播给全 Block
    if (threadIdx.x == 0) result = v;
    __syncthreads();
    return result;
}

__device__ float blockReduceSum(float v) {
    __shared__ float warpRes[32];
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    v = warpReduceSum(v);
    if (lane == 0) warpRes[wid] = v;
    __syncthreads();

    int nWarp = (blockDim.x + 31) / 32;
    v = (lane < nWarp) ? warpRes[lane] : 0.0f;
    if (wid == 0) v = warpReduceSum(v);

    __shared__ float result;
    if (threadIdx.x == 0) result = v;
    __syncthreads();
    return result;
}

// ===========================================================================
// V0：朴素三 kernel（对应文档 4.1 节）
// ===========================================================================

// kernel 1: S = Q·Kᵀ / √d    （每线程算 S 的一个元素）
__global__ void qk_kernel(const float* Q, const float* K, float* S,
                          int N, int d, float scale) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // key 序号 j
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // query 序号 i
    if (row < N && col < N) {
        float s = 0.0f;
        for (int x = 0; x < d; x++)
            s += Q[row * d + x] * K[col * d + x];      // K 按行存，Kᵀ 即按行取 K
        S[row * N + col] = s * scale;                  // scale = 1/√d
    }
}

// kernel 2: P = softmax(S) 逐行（示意的三遍扫描；每线程处理一行——效率极低）
// 这里把 causal mask 也加进来（j > i 置 -inf），与 V1/V3 对齐
__global__ void softmax_kernel(const float* S, float* P, int N, bool causal) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    float m = -INFINITY;
    for (int j = 0; j < N; j++) {                                    // pass 1
        float x = S[row * N + j];
        if (causal && j > row) x = -INFINITY;
        m = fmaxf(m, x);
    }
    float l = 0.0f;
    for (int j = 0; j < N; j++) {                                    // pass 2
        float x = S[row * N + j];
        if (causal && j > row) x = -INFINITY;
        l += expf(x - m);
    }
    for (int j = 0; j < N; j++) {                                    // pass 3
        float x = S[row * N + j];
        if (causal && j > row) x = -INFINITY;
        P[row * N + j] = expf(x - m) / l;
    }
}

// kernel 3: O = P·V    （每线程算 O 的一个元素）
__global__ void pv_kernel(const float* P, const float* V, float* O,
                          int N, int d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // 0..d-1
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // 0..N-1
    if (row < N && col < d) {
        float o = 0.0f;
        for (int k = 0; k < N; k++)
            o += P[row * N + k] * V[k * d + col];
        O[row * d + col] = o;
    }
}

// ===========================================================================
// V1：融合 scale + mask + softmax 单 kernel（对应文档 5.2 节）
// 一行一个 Block，行内两级归约完成三遍扫描
// ===========================================================================

// P = softmax(mask(S · scale))，逐行；一行一个 Block
__global__ void fused_softmax_kernel(const float* S, float* P,
                                     int N, float scale, bool causal) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* srow = S + (size_t)row * N;
    float*       prow = P + (size_t)row * N;

    // pass 1: 求行最大值（scale 与 mask 在读取时现场应用）
    float m = -INFINITY;
    for (int j = tid; j < N; j += blockDim.x) {
        float x = srow[j] * scale;
        if (causal && j > row) x = -INFINITY;
        m = fmaxf(m, x);
    }
    m = blockReduceMax(m);

    // pass 2: 求分母 l = Σ exp(x − m)
    float l = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float x = srow[j] * scale;
        if (causal && j > row) x = -INFINITY;
        l += __expf(x - m);                          // exp(-inf) = 0，mask 自然生效
    }
    l = blockReduceSum(l);

    // pass 3: 写出归一化结果
    float inv_l = 1.0f / l;
    for (int j = tid; j < N; j += blockDim.x) {
        float x = srow[j] * scale;
        if (causal && j > row) x = -INFINITY;
        prow[j] = __expf(x - m) * inv_l;
    }
}

// ===========================================================================
// V3：FlashAttention 教学版（对应文档 7.3 节）
// 每个 Block 处理一个 Q 行块，blockDim.x = Br，每个线程负责块内一行
// Q/K/V/O: [BH, N, D]（BH = batch*heads）；要求 N 是 Bc 的倍数、Bc 是 Br 的倍数
// ===========================================================================

template <int Br, int Bc, int D>
__global__ void flash_attn_v3(const float* Q, const float* K,
                              const float* V, float* O,
                              int N, float scale, bool causal) {
    int qb  = blockIdx.x;                 // Q 行块编号 0..N/Br-1
    int bh  = blockIdx.y;                 // batch*head 编号
    int tid = threadIdx.x;                // 0..Br-1：本线程负责的块内行

    size_t off = (size_t)bh * N * D;      // 指针推进到本 (batch, head)
    Q += off; K += off; V += off; O += off;

    __shared__ float Ks[Bc][D];
    __shared__ float Vs[Bc][D];

    // ① 本线程的 Q 行与在线统计量 → 寄存器（全程驻留）
    int qRow = qb * Br + tid;
    float q[D], o[D];
    for (int x = 0; x < D; x++) { q[x] = Q[qRow * D + x]; o[x] = 0.0f; }
    float m = -INFINITY, l = 0.0f;

    // ② 内层：沿序列遍历 K/V 块
    for (int j0 = 0; j0 < N; j0 += Bc) {
        if (causal && j0 > qRow) break;   // 整块都在掩码之外，直接结束

        // 协作装载 K/V tile（Br 个线程搬 Bc×D 个元素，跨步循环、地址连续）
        for (int idx = tid; idx < Bc * D; idx += Br) {
            Ks[idx / D][idx % D] = K[(size_t)(j0 + idx / D) * D + idx % D];
            Vs[idx / D][idx % D] = V[(size_t)(j0 + idx / D) * D + idx % D];
        }
        __syncthreads();

        // 本行 vs tile 内 Bc 个 key：算分数 + 在线修正累加
        for (int c = 0; c < Bc; c++) {
            if (causal && j0 + c > qRow) break;      // 块内尾部掩码

            float s = 0.0f;                          // s = q · k_c
            for (int x = 0; x < D; x++) s += q[x] * Ks[c][x];
            s *= scale;

            float m_new = fmaxf(m, s);
            float p     = __expf(s - m_new);         // 本分数的指数权重（以 m_new 为基准）
            float corr  = __expf(m - m_new);         // 旧累积的补偿系数
            l = l * corr + p;
            for (int x = 0; x < D; x++)
                o[x] = o[x] * corr + p * Vs[c][x];   // õ 递推（6.4 节的输出递推）
            m = m_new;
        }
        __syncthreads();                             // 用完才许下一轮覆盖 tile
    }

    // ③ 归一化并写回（合并写出的行主序连续段）
    float inv_l = 1.0f / l;
    for (int x = 0; x < D; x++) O[qRow * D + x] = o[x] * inv_l;
}

// ===========================================================================
// V4：FlashAttention-2 教学版（对应文档 8.7 节）
// split-Q：一个 Warp 负责一行，行内 head 维 D 切给 32 个 lane 并行
// Br = 每 Block 的 Warp 数（每 Warp 一行）；要求 D 是 32 的倍数、Bc 是 Br 的倍数
// ===========================================================================

template <int Br, int Bc, int D>
__global__ void flash_attn_v4(const float* Q, const float* K,
                              const float* V, float* O,
                              int N, float scale, bool causal) {
    constexpr int DL = D / 32;            // 每 lane 分担的 head 维分量数
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;          // Warp 编号 = 本 Warp 负责的块内行
    int qb   = blockIdx.x;                // Q 行块编号
    int bh   = blockIdx.y;                // batch*head 编号

    size_t off = (size_t)bh * N * D;
    Q += off; K += off; V += off; O += off;

    __shared__ float Ks[Bc][D];
    __shared__ float Vs[Bc][D];

    int qRow = qb * Br + wid;

    // ① q 与 õ 沿 head 维切给 32 个 lane（每 lane DL 个分量，驻留寄存器）
    float q[DL], o[DL];
    for (int x = 0; x < DL; x++) {
        q[x] = Q[qRow * D + x * 32 + lane];      // lane 连续 → 合并访存
        o[x] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;               // 本行统计量（每 lane 冗余一份）

    // ② 内层：沿序列遍历 K/V 块
    for (int j0 = 0; j0 < N; j0 += Bc) {
        if (causal && j0 > qRow) break;          // Bc % Br == 0 时全 Block 同步跳出

        // 全 Block（32×Br 线程）协作装载 K/V tile
        for (int idx = threadIdx.x; idx < Bc * D; idx += 32 * Br) {
            Ks[idx / D][idx % D] = K[(size_t)(j0 + idx / D) * D + idx % D];
            Vs[idx / D][idx % D] = V[(size_t)(j0 + idx / D) * D + idx % D];
        }
        __syncthreads();

        for (int c = 0; c < Bc; c++) {
            if (causal && j0 + c > qRow) break;  // Warp 内 32 lane 行号相同，不发散

            // Warp 级并行点积：各 lane 算 DL 个分量的部分和，蝶形归约拼出完整 s
            float s = 0.0f;
            for (int x = 0; x < DL; x++) s += q[x] * Ks[c][x * 32 + lane];
            for (int d2 = 16; d2 > 0; d2 >>= 1)
                s += __shfl_xor_sync(0xffffffff, s, d2);   // 蝶形归约
            s *= scale;                                     // 32 个 lane 都持有完整 s

            // 在线递推与 V3 完全相同（m、l 在各 lane 上冗余但数值一致）
            float m_new = fmaxf(m, s);
            float p     = __expf(s - m_new);
            float corr  = __expf(m - m_new);
            l = l * corr + p;
            for (int x = 0; x < DL; x++)
                o[x] = o[x] * corr + p * Vs[c][x * 32 + lane];
            m = m_new;
        }
        __syncthreads();
    }

    // ③ 延迟归一化（唯一一次除法）+ 写回：各 lane 写自己的 DL 个分量，地址连续
    float inv_l = 1.0f / l;
    for (int x = 0; x < DL; x++)
        O[qRow * D + x * 32 + lane] = o[x] * inv_l;
}

// ===========================================================================
// CPU 参考实现（对应文档 10.2 节 reference）：物化 N×N 矩阵，慢但直观
// ===========================================================================
static void cpu_reference(const float* Q, const float* K, const float* V,
                          float* O, int N, int D, bool causal) {
    float scale = 1.0f / sqrtf((float)D);
    float* s = (float*)malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
        // ① 分数 + mask
        float m = -INFINITY;
        for (int j = 0; j < N; j++) {
            if (causal && j > i) { s[j] = -INFINITY; continue; }
            float acc = 0.0f;
            for (int x = 0; x < D; x++) acc += Q[i * D + x] * K[j * D + x];
            s[j] = acc * scale;
            m = fmaxf(m, s[j]);
        }
        // ② safe softmax
        float l = 0.0f;
        for (int j = 0; j < N; j++) {
            s[j] = (s[j] == -INFINITY) ? 0.0f : expf(s[j] - m);
            l += s[j];
        }
        // ③ O = P·V
        for (int x = 0; x < D; x++) {
            float acc = 0.0f;
            for (int j = 0; j < N; j++) acc += (s[j] / l) * V[j * D + x];
            O[i * D + x] = acc;
        }
    }
    free(s);
}

static float max_abs_diff(const float* a, const float* b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) mx = fmaxf(mx, fabsf(a[i] - b[i]));
    return mx;
}

// ===========================================================================
// 宿主端驱动
// ===========================================================================
int main(int argc, char** argv) {
    int  N      = (argc > 1) ? atoi(argv[1]) : 1024;
    int  D      = (argc > 2) ? atoi(argv[2]) : 64;
    bool causal = (argc > 3) ? (atoi(argv[3]) != 0) : true;

    // 编译期分块参数（模板要求编译期常量），要求 D == 64
    constexpr int TD  = 64;
    constexpr int V3Br = 64, V3Bc = 64;   // V3：每 Block 64 线程（每线程一行）
    constexpr int V4Br = 8,  V4Bc = 64;   // V4：每 Block 8 个 Warp（每 Warp 一行）

    if (D != TD) {
        fprintf(stderr, "本 demo 的模板 kernel 固定 D=%d（可修改模板实参）。\n", TD);
        return 1;
    }
    if (N % V3Bc != 0 || N % V3Br != 0 || N % V4Br != 0) {
        fprintf(stderr, "要求 N 同时为 Bc(%d)、V3Br(%d)、V4Br(%d) 的倍数。\n",
                V3Bc, V3Br, V4Br);
        return 1;
    }

    printf("Scaled Dot-Product Attention: N=%d, D=%d, causal=%d\n\n", N, D, causal);
    float scale = 1.0f / sqrtf((float)D);

    // 主机随机输入（文档 10.2 节：必须用随机数据，否则 bug 全部隐形）
    size_t szIn = (size_t)N * D;
    float* h_Q = (float*)malloc(sizeof(float) * szIn);
    float* h_K = (float*)malloc(sizeof(float) * szIn);
    float* h_V = (float*)malloc(sizeof(float) * szIn);
    float* h_O = (float*)malloc(sizeof(float) * szIn);
    float* h_ref = (float*)malloc(sizeof(float) * szIn);
    srand(0);
    for (size_t i = 0; i < szIn; i++) {
        h_Q[i] = (float)rand() / RAND_MAX - 0.5f;
        h_K[i] = (float)rand() / RAND_MAX - 0.5f;
        h_V[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    // CPU 参考（裁判）
    cpu_reference(h_Q, h_K, h_V, h_ref, N, D, causal);

    // 设备内存
    float *d_Q, *d_K, *d_V, *d_O, *d_S, *d_P;
    CUDA_CHECK(cudaMalloc(&d_Q, sizeof(float) * szIn));
    CUDA_CHECK(cudaMalloc(&d_K, sizeof(float) * szIn));
    CUDA_CHECK(cudaMalloc(&d_V, sizeof(float) * szIn));
    CUDA_CHECK(cudaMalloc(&d_O, sizeof(float) * szIn));
    CUDA_CHECK(cudaMalloc(&d_S, sizeof(float) * (size_t)N * N));   // N×N 中间矩阵
    CUDA_CHECK(cudaMalloc(&d_P, sizeof(float) * (size_t)N * N));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, sizeof(float) * szIn, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, sizeof(float) * szIn, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, sizeof(float) * szIn, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("%-28s %12s %12s %8s\n", "Version", "time(ms)", "max|diff|", "check");
    printf("----------------------------------------------------------------\n");

    // ---- V0：三 kernel ----
    {
        dim3 blk(16, 16);
        dim3 grdQK((N + 15) / 16, (N + 15) / 16);
        dim3 grdPV((D + 15) / 16, (N + 15) / 16);
        int  grdSM = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        CUDA_CHECK(cudaEventRecord(start));
        qk_kernel<<<grdQK, blk>>>(d_Q, d_K, d_S, N, D, scale);
        softmax_kernel<<<grdSM, BLOCK_SIZE>>>(d_S, d_P, N, causal);
        pv_kernel<<<grdPV, blk>>>(d_P, d_V, d_O, N, D);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaMemcpy(h_O, d_O, sizeof(float) * szIn, cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(h_O, h_ref, szIn);
        printf("%-28s %12.4f %12.2e %8s\n", "V0 (3 kernels)", ms, diff,
               diff < 1e-3f ? "OK" : "FAIL");
    }

    // ---- V1：融合 softmax（QKᵀ 与 PV 仍复用 V0 的 kernel）----
    {
        dim3 blk(16, 16);
        dim3 grdQK((N + 15) / 16, (N + 15) / 16);
        dim3 grdPV((D + 15) / 16, (N + 15) / 16);

        CUDA_CHECK(cudaEventRecord(start));
        qk_kernel<<<grdQK, blk>>>(d_Q, d_K, d_S, N, D, 1.0f);         // scale 挪进融合 kernel
        fused_softmax_kernel<<<N, BLOCK_SIZE>>>(d_S, d_P, N, scale, causal);
        pv_kernel<<<grdPV, blk>>>(d_P, d_V, d_O, N, D);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaMemcpy(h_O, d_O, sizeof(float) * szIn, cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(h_O, h_ref, szIn);
        printf("%-28s %12.4f %12.2e %8s\n", "V1 (fused softmax)", ms, diff,
               diff < 1e-3f ? "OK" : "FAIL");
    }

    // ---- V3：FlashAttention（每线程一行）----
    {
        dim3 grid(N / V3Br, 1);          // BH = 1
        CUDA_CHECK(cudaEventRecord(start));
        flash_attn_v3<V3Br, V3Bc, TD><<<grid, V3Br>>>(d_Q, d_K, d_V, d_O, N, scale, causal);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaMemcpy(h_O, d_O, sizeof(float) * szIn, cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(h_O, h_ref, szIn);
        printf("%-28s %12.4f %12.2e %8s\n", "V3 (FlashAttention)", ms, diff,
               diff < 1e-3f ? "OK" : "FAIL");
    }

    // ---- V4：FlashAttention-2（split-Q，一 Warp 一行）----
    {
        dim3 grid(N / V4Br, 1);          // BH = 1
        CUDA_CHECK(cudaEventRecord(start));
        flash_attn_v4<V4Br, V4Bc, TD><<<grid, V4Br * 32>>>(d_Q, d_K, d_V, d_O, N, scale, causal);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaMemcpy(h_O, d_O, sizeof(float) * szIn, cudaMemcpyDeviceToHost));
        float diff = max_abs_diff(h_O, h_ref, szIn);
        printf("%-28s %12.4f %12.2e %8s\n", "V4 (FlashAttention-2)", ms, diff,
               diff < 1e-3f ? "OK" : "FAIL");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K)); CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O)); CUDA_CHECK(cudaFree(d_S)); CUDA_CHECK(cudaFree(d_P));
    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_ref);
    printf("\n（max|diff| 应为 ~1e-6 量级；容差 1e-3 通过即 OK）\n");
    return 0;
}
