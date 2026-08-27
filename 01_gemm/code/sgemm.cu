// sgemm.cu — CUDA SGEMM（单精度矩阵乘）从 V0 到 V6 的完整可运行实现
//
// 对应文档: ../cuda_gemm_optimization_guide.md
// 覆盖版本（fp32，C = A × B，A:M×K, B:K×N, C:M×N，行主序）:
//   V0 一线程一元素，映射错误（threadIdx.x → 行），全局访存不合并（基准）  —— 第 4 章
//   V1 交换行列映射（threadIdx.x → 列），合并访存                          —— 第 5 章
//   V2 Block Tiling，子块搬入共享内存复用                                  —— 第 6 章
//   V3 一维 Thread Tiling，每线程算 8 个输出（TM=8）                       —— 第 7 章
//   V4 二维 Thread Tiling，8×8 外积累加，LDS/FMA=0.25                      —— 第 8 章
//   V5 float4 向量化 + As 转置存储                                         —— 第 9 章
//   V6 双缓冲（Double Buffering），加载与计算流水重叠                        —— 第 10 章
//
// Tensor Core（V7，half 精度）见 hgemm_wmma.cu
//
// 编译:
//   nvcc -O3 -arch=sm_70 sgemm.cu -o sgemm
// 运行:
//   ./sgemm            # 默认 M=N=K=1024（便于 CPU 校验）
//   ./sgemm 2048       # 指定方阵边长
//   ./sgemm 1024 768 512   # 指定 M N K

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

// ===========================================================================
// V0：基准实现——一线程一元素（第 4 章）
// 刻意保留的错误映射：threadIdx.x → 行，导致对 A/C 的全局访问不合并
// ===========================================================================
__global__ void sgemm_v0(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;   // x → 行（错误映射）
    int col = blockIdx.y * blockDim.y + threadIdx.y;   // y → 列

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

// ===========================================================================
// V1：合并访存——修正线程到矩阵的映射（第 5 章）
// threadIdx.x → 列，让 Warp 内相邻线程访问连续地址
// ===========================================================================
__global__ void sgemm_v1(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // x → 列
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // y → 行

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

// ===========================================================================
// V2：共享内存分块——Block Tiling（第 6 章）
// 一个 Block 负责 C 的一个 TILE×TILE 子块，沿 K 维分段搬入共享内存
// ===========================================================================
#define TILE 32

__global__ void sgemm_v2(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    __shared__ float As[TILE][TILE];    // 2 × 32×32×4B = 8 KB 共享内存
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;                    // 列方向（保持 V1 的合并映射）
    int ty = threadIdx.y;                    // 行方向
    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;

    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        // ① 协作加载：每线程各搬 A、B 的一个元素（越界补 0）
        As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx]   : 0.0f;
        Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
        __syncthreads();                     // 等全部数据就位

        // ② 用共享内存中的子块做 TILE 次乘加
        for (int k = 0; k < TILE; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();                     // 防止下一轮加载覆盖还在使用的数据
    }

    if (row < M && col < N) C[row * N + col] = acc;
}

// ===========================================================================
// V3：一维 Thread Tiling（第 7 章）
// 每线程负责同一列的 TM=8 个输出；Bs 读 1 次进寄存器，复用 TM 次
// 分块参数: BM=BN=64, BK=8, TM=8, 线程数 = (BM*BN)/TM = 512
// 要求 M、N 被 BM/BN 整除，K 被 BK 整除
// ===========================================================================
template <int BM, int BN, int BK, int TM>
__global__ void sgemm_v3(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    __shared__ float As[BM * BK];      // 64×8
    __shared__ float Bs[BK * BN];      // 8×64

    // 把指针推进到本 Block 负责的子块起点，后续用局部坐标
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // 计算映射：512 线程排成 8 行 × 64 列，每线程纵向负责 TM=8 个输出
    const int threadCol = threadIdx.x % BN;        // 0..63
    const int threadRow = threadIdx.x / BN;        // 0..7

    // 加载映射：按各自矩阵的形状重新分工（与计算映射无关）
    const int innerColA = threadIdx.x % BK;        // 0..7
    const int innerRowA = threadIdx.x / BK;        // 0..63
    const int innerColB = threadIdx.x % BN;        // 0..63
    const int innerRowB = threadIdx.x / BN;        // 0..7

    float acc[TM] = {0.0f};

    for (int t = 0; t < K; t += BK) {
        // ① 协作加载 64×8 的 As 与 8×64 的 Bs（每线程各 1 个元素）
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();
        A += BK;                                   // A 子块右移
        B += BK * N;                               // B 子块下移

        // ② 内层：对每个 k，Bs 读 1 次进寄存器，复用 TM 次
        for (int k = 0; k < BK; k++) {
            float b = Bs[k * BN + threadCol];
            for (int i = 0; i < TM; i++) {
                acc[i] += As[(threadRow * TM + i) * BK + k] * b;
            }
        }
        __syncthreads();
    }

    // ③ 写回 TM 个结果
    for (int i = 0; i < TM; i++) {
        C[(threadRow * TM + i) * N + threadCol] = acc[i];
    }
}

// ===========================================================================
// V4：二维 Thread Tiling——外积累加（第 8 章）
// 每线程负责 C 的一个 TM×TN=8×8 小方块，固定 k 做 8×8 外积
// 分块参数: BM=BN=128, BK=8, TM=TN=8, 线程数 = (BM*BN)/(TM*TN) = 256
// ===========================================================================
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v4(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    __shared__ float As[BM * BK];               // 128×8
    __shared__ float Bs[BK * BN];               // 8×128

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const int threadCol = threadIdx.x % (BN / TN);   // 0..15
    const int threadRow = threadIdx.x / (BN / TN);   // 0..15

    // 加载映射（256 线程 → 128×8 的 As：每次覆盖 32 行，循环 4 次）
    const int innerRowA = threadIdx.x / BK;     // 0..31
    const int innerColA = threadIdx.x % BK;     // 0..7
    const int strideA   = blockDim.x / BK;      // 32
    const int innerRowB = threadIdx.x / BN;     // 0..1
    const int innerColB = threadIdx.x % BN;     // 0..127
    const int strideB   = blockDim.x / BN;      // 2

    float acc[TM][TN] = {{0.0f}};
    float regA[TM], regB[TN];

    for (int t = 0; t < K; t += BK) {
        // ① 协作加载（每线程多个元素，跨步循环）
        for (int off = 0; off < BM; off += strideA)
            As[(innerRowA + off) * BK + innerColA] = A[(innerRowA + off) * K + innerColA];
        for (int off = 0; off < BK; off += strideB)
            Bs[(innerRowB + off) * BN + innerColB] = B[(innerRowB + off) * N + innerColB];
        __syncthreads();
        A += BK;
        B += BK * N;

        // ② 外积累加
        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i++)         // As 第 k 列的 8 个数 → 寄存器
                regA[i] = As[(threadRow * TM + i) * BK + k];
            for (int j = 0; j < TN; j++)         // Bs 第 k 行的 8 个数 → 寄存器
                regB[j] = Bs[k * BN + threadCol * TN + j];
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i] * regB[j];   // 64 次独立 FMA
        }
        __syncthreads();
    }

    // ③ 写回 8×8 小方块
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            C[(threadRow * TM + i) * N + threadCol * TN + j] = acc[i][j];
}

// ===========================================================================
// V5：float4 向量化 + 共享内存布局重排（第 9 章）
// global→smem 用 float4（LDG.128）搬运；As 转置存储 [BK][BM]，使计算期"取一列"
// 变为连续访问，可用 LDS.128 向量化
// 分块参数: BM=BN=128, BK=8, TM=TN=8, 线程数 = 256
// ===========================================================================
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v5(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    __shared__ float As[BK * BM];    // 注意：转置布局 [BK][BM]
    __shared__ float Bs[BK * BN];    // 正常布局 [BK][BN]

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    const int innerRowA = threadIdx.x / (BK / 4);    // 0..127
    const int innerColA = threadIdx.x % (BK / 4);    // 0..1
    const int innerRowB = threadIdx.x / (BN / 4);    // 0..7
    const int innerColB = threadIdx.x % (BN / 4);    // 0..31

    float acc[TM][TN] = {{0.0f}};
    float regA[TM], regB[TN];

    for (int t = 0; t < K; t += BK) {
        // ① float4 加载 + As 转置写入
        float4 ta = reinterpret_cast<const float4*>(
                        &A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = ta.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = ta.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = ta.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = ta.w;

        reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();
        A += BK;
        B += BK * N;

        // ② 外积累加：regA/regB 均可由编译器生成 LDS.128
        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i++)
                regA[i] = As[k * BM + threadRow * TM + i];   // 连续!
            for (int j = 0; j < TN; j++)
                regB[j] = Bs[k * BN + threadCol * TN + j];   // 连续!
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
    }

    // ③ 写回也用 float4（每行 8 个输出 = 2 条 ST.128）
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
            float4 out = {acc[i][j], acc[i][j+1], acc[i][j+2], acc[i][j+3]};
            reinterpret_cast<float4*>(
                &C[(threadRow * TM + i) * N + threadCol * TN + j])[0] = out;
        }
}

// ===========================================================================
// V6：双缓冲——用计算掩盖访存延迟（第 10 章）
// 在 V5 基础上开两份共享内存缓冲区乒乓切换，计算第 t 块时预取第 t+1 块
// 分块参数: BM=BN=128, BK=8, TM=TN=8, 线程数 = 256
// ===========================================================================
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v6(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    __shared__ float As[2][BK * BM];     // 双缓冲，转置布局
    __shared__ float Bs[2][BK * BN];

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    const int innerRowA = threadIdx.x / (BK / 4);    // 0..127
    const int innerColA = threadIdx.x % (BK / 4);    // 0..1
    const int innerRowB = threadIdx.x / (BN / 4);    // 0..7
    const int innerColB = threadIdx.x % (BN / 4);    // 0..31

    float acc[TM][TN] = {{0.0f}};
    float regA[TM], regB[TN];
    float4 ta, tb;                        // 预取暂存寄存器

    // 序幕：加载第 0 块到缓冲 0
    ta = reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA * 4])[0];
    tb = reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];
    As[0][(innerColA * 4 + 0) * BM + innerRowA] = ta.x;
    As[0][(innerColA * 4 + 1) * BM + innerRowA] = ta.y;
    As[0][(innerColA * 4 + 2) * BM + innerRowA] = ta.z;
    As[0][(innerColA * 4 + 3) * BM + innerRowA] = ta.w;
    reinterpret_cast<float4*>(&Bs[0][innerRowB * BN + innerColB * 4])[0] = tb;
    __syncthreads();

    int cur = 0;
    for (int t = 0; t < K; t += BK) {
        int next = cur ^ 1;

        // ① 发出下一块的全局内存读（最后一轮除外）——立即返回，不阻塞
        //    A/B 固定在 Block 起点，用 (t + BK) 作为列偏移显式索引下一块
        if (t + BK < K) {
            ta = reinterpret_cast<const float4*>(
                     &A[innerRowA * K + (t + BK) + innerColA * 4])[0];
            tb = reinterpret_cast<const float4*>(
                     &B[(t + BK + innerRowB) * N + innerColB * 4])[0];
        }

        // ② 计算当前块（长串 FFMA 掩盖 ① 的延迟）
        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i++)
                regA[i] = As[cur][k * BM + threadRow * TM + i];
            for (int j = 0; j < TN; j++)
                regB[j] = Bs[cur][k * BN + threadCol * TN + j];
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i] * regB[j];
        }

        // ③ 把预取的数据写入另一份缓冲
        if (t + BK < K) {
            As[next][(innerColA * 4 + 0) * BM + innerRowA] = ta.x;
            As[next][(innerColA * 4 + 1) * BM + innerRowA] = ta.y;
            As[next][(innerColA * 4 + 2) * BM + innerRowA] = ta.z;
            As[next][(innerColA * 4 + 3) * BM + innerRowA] = ta.w;
            reinterpret_cast<float4*>(
                &Bs[next][innerRowB * BN + innerColB * 4])[0] = tb;
            __syncthreads();             // ④ 每轮只需一次同步
        }
        cur = next;
    }

    // 写回 8×8 小方块（用 float4）
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
            float4 out = {acc[i][j], acc[i][j+1], acc[i][j+2], acc[i][j+3]};
            reinterpret_cast<float4*>(
                &C[(threadRow * TM + i) * N + threadCol * TN + j])[0] = out;
        }
}

// ===========================================================================
// CPU 参考实现：用于验证 GPU 结果（第 1 章三重循环）
// ===========================================================================
static void sgemm_cpu(int M, int N, int K,
                      const float* A, const float* B, float* C) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += A[m * K + k] * B[k * N + n];
            C[m * N + n] = acc;
        }
}

// ===========================================================================
// 宿主端驱动
// ===========================================================================

enum Version { V0, V1, V2, V3, V4, V5, V6 };

// 检查某版本对给定形状的整除约束（分块版本 V3~V6 有硬约束）
static bool version_supported(Version v, int M, int N, int K) {
    switch (v) {
        case V0: case V1: case V2:
            return true;                                  // 有边界判断，任意形状
        case V3:
            return (M % 64 == 0) && (N % 64 == 0) && (K % 8 == 0);
        case V4: case V5: case V6:
            return (M % 128 == 0) && (N % 128 == 0) && (K % 8 == 0);
    }
    return false;
}

static void launch(Version v, int M, int N, int K,
                   const float* dA, const float* dB, float* dC) {
    switch (v) {
        case V0: {
            // 注意 V0 的映射：x → 行(M)，y → 列(N)
            dim3 block(32, 32);
            dim3 grid((M + 31) / 32, (N + 31) / 32);
            sgemm_v0<<<grid, block>>>(M, N, K, dA, dB, dC);
            break;
        }
        case V1: {
            // V1 修正映射：x → 列(N)，y → 行(M)
            dim3 block(32, 32);
            dim3 grid((N + 31) / 32, (M + 31) / 32);
            sgemm_v1<<<grid, block>>>(M, N, K, dA, dB, dC);
            break;
        }
        case V2: {
            dim3 block(TILE, TILE);
            dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
            sgemm_v2<<<grid, block>>>(M, N, K, dA, dB, dC);
            break;
        }
        case V3: {
            constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
            dim3 grid(N / BN, M / BM);
            sgemm_v3<BM, BN, BK, TM><<<grid, (BM * BN) / TM>>>(M, N, K, dA, dB, dC);
            break;
        }
        case V4: {
            constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
            dim3 grid(N / BN, M / BM);
            sgemm_v4<BM, BN, BK, TM, TN><<<grid, (BM * BN) / (TM * TN)>>>(M, N, K, dA, dB, dC);
            break;
        }
        case V5: {
            constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
            dim3 grid(N / BN, M / BM);
            sgemm_v5<BM, BN, BK, TM, TN><<<grid, (BM * BN) / (TM * TN)>>>(M, N, K, dA, dB, dC);
            break;
        }
        case V6: {
            constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
            dim3 grid(N / BN, M / BM);
            sgemm_v6<BM, BN, BK, TM, TN><<<grid, (BM * BN) / (TM * TN)>>>(M, N, K, dA, dB, dC);
            break;
        }
    }
}

int main(int argc, char** argv) {
    int M, N, K;
    if (argc >= 4) {
        M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]);
    } else if (argc == 2) {
        M = N = K = atoi(argv[1]);
    } else {
        M = N = K = 1024;          // 默认适中大小，便于 CPU 校验
    }
    printf("SGEMM  C[%dx%d] = A[%dx%d] x B[%dx%d]\n\n", M, N, M, K, K, N);

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float* hA    = (float*)malloc(bytesA);
    float* hB    = (float*)malloc(bytesB);
    float* hC    = (float*)malloc(bytesC);   // GPU 结果
    float* hCref = (float*)malloc(bytesC);   // CPU 参考结果

    // 用随机数填充（不要用全 1：行列写反等索引错误在对称输入下测不出来）
    srand(42);
    for (int i = 0; i < M * K; i++) hA[i] = (float)(rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < K * N; i++) hB[i] = (float)(rand() % 200 - 100) / 100.0f;

    // CPU 参考（1024³ 约需数秒，可接受）
    printf("Computing CPU reference...\n");
    sgemm_cpu(M, N, K, hA, hB, hCref);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    const char* names[] = {"V0", "V1", "V2", "V3", "V4", "V5", "V6"};
    Version vers[]      = {V0, V1, V2, V3, V4, V5, V6};

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double gflop = 2.0 * M * N * K / 1e9;   // 2MNK FLOP
    printf("\n%-6s %12s %12s %10s\n", "Ver", "time(ms)", "GFLOPS", "check");
    printf("------------------------------------------------\n");

    for (int i = 0; i < 7; ++i) {
        if (!version_supported(vers[i], M, N, K)) {
            printf("%-6s %12s %12s %10s\n", names[i], "-", "-", "SKIP(shape)");
            continue;
        }

        CUDA_CHECK(cudaMemset(dC, 0, bytesC));

        // 预热
        launch(vers[i], M, N, K, dA, dB, dC);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 计时
        CUDA_CHECK(cudaEventRecord(start));
        launch(vers[i], M, N, K, dA, dB, dC);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        double gflops = gflop / (ms * 1e-3);

        CUDA_CHECK(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));

        // 相对误差校验
        double max_rel = 0.0;
        int errors = 0;
        for (int idx = 0; idx < M * N; idx++) {
            float ref = hCref[idx];
            float rel = fabsf(hC[idx] - ref) / (fabsf(ref) + 1e-5f);
            if (rel > max_rel) max_rel = rel;
            if (rel > 1e-3f) errors++;
        }
        printf("%-6s %12.4f %12.1f %10s (max_rel=%.2e)\n",
               names[i], ms, gflops, errors == 0 ? "PASS" : "FAIL", max_rel);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC); free(hCref);
    return 0;
}
