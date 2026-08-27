// rmsnorm.cu — CUDA RMSNorm（均方根归一化）从 V0 到 V3 的完整可运行实现
//
// 对应文档: ../cuda_rmsnorm_optimization_guide.md
// 覆盖版本:
//   V0 基准实现：一行一 Block，共享内存树形归约（第 4 章）
//   V1 Warp Shuffle 两级归约（第 5 章）
//   V2 float4 向量化 + 行驻留寄存器（第 6 章）
//   V3 行级调度：短行一 Warp，一个 Block 装 8 行（第 7 章）
//
// 公式（对形状 [N, H] 的输入，每一行独立）:
//   rms(x)  = sqrt( (1/H) * Σ x_i^2 + eps )
//   y_i     = x_i / rms(x) * gamma_i
//
// 编译:
//   nvcc -O3 -arch=sm_70 rmsnorm.cu -o rmsnorm
// 运行:
//   ./rmsnorm            # 默认 N=4096, H=4096
//   ./rmsnorm 8192 2048  # 指定 N 与 H（要求 H % 4 == 0）

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
// V0：基准实现——一行一 Block；共享内存树形归约（第 4 章）
// 启动：rmsnorm_v0<<<N, 256, 256 * sizeof(float)>>>(x, y, gamma, H, 1e-6f);
// ===========================================================================
__global__ void rmsnorm_v0(const float* __restrict__ x, float* __restrict__ y,
                           const float* __restrict__ gamma, int H, float eps) {
    extern __shared__ float sdata[];                  // blockDim.x 个 float
    const float* row = x + (size_t)blockIdx.x * H;    // 本 Block 负责的行
    float*       out = y + (size_t)blockIdx.x * H;
    int tid = threadIdx.x;

    // ① 每线程跨步累加平方和
    float acc = 0.f;
    for (int i = tid; i < H; i += blockDim.x)         // 跨步循环：warp 内地址连续（合并访存）
        acc += row[i] * row[i];

    // ② Block 内树形归约
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {    // 每轮活跃线程减半
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // ③ 线程 0 算 1/rms，经共享内存广播
    __shared__ float s_rrms;
    if (tid == 0) s_rrms = rsqrtf(sdata[0] / H + eps);
    __syncthreads();
    float rrms = s_rrms;

    // ④ 逐元素缩放（第二遍读 x）
    for (int i = tid; i < H; i += blockDim.x)
        out[i] = row[i] * rrms * gamma[i];
}

// ===========================================================================
// V1：Warp Shuffle 两级归约（第 5 章）
// ===========================================================================
__device__ __forceinline__ float warpReduceSum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);  // 蝶形：5 步，寄存器直传
    return v;                                         // 32 个 lane 都持有完整和
}

__device__ __forceinline__ float blockReduceSum(float v) {
    __shared__ float warpRes[32];                     // 至多 32 个 Warp 的部分和
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warpReduceSum(v);                             // 第一级：Warp 内
    if (lane == 0) warpRes[wid] = v;
    __syncthreads();
    int nWarp = (blockDim.x + 31) >> 5;
    v = (lane < nWarp) ? warpRes[lane] : 0.f;
    if (wid == 0) v = warpReduceSum(v);               // 第二级：Warp 间
    __shared__ float result;
    if (threadIdx.x == 0) result = v;
    __syncthreads();                                  // 广播给全 Block
    return result;
}

// V1：主体归约一行搞定
__global__ void rmsnorm_v1(const float* __restrict__ x, float* __restrict__ y,
                           const float* __restrict__ gamma, int H, float eps) {
    const float* row = x + (size_t)blockIdx.x * H;
    float*       out = y + (size_t)blockIdx.x * H;

    float acc = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        acc += row[i] * row[i];
    acc = blockReduceSum(acc);                        // 所有线程都拿到 Σx²

    float rrms = rsqrtf(acc / H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        out[i] = row[i] * rrms * gamma[i];
}

// ===========================================================================
// V2：float4 向量化 + 行驻留寄存器（第 6 章）
// 要求 H = ITEMS * blockDim.x * 4（H=4096, block=256 → ITEMS=4）
// 启动：rmsnorm_v2<4><<<N, 256>>>(x, y, gamma, H, eps);
// ===========================================================================
template <int ITEMS>
__global__ void rmsnorm_v2(const float* __restrict__ x, float* __restrict__ y,
                           const float* __restrict__ gamma, int H, float eps) {
    const float4* row4 = reinterpret_cast<const float4*>(x + (size_t)blockIdx.x * H);
    float4*       out4 = reinterpret_cast<float4*>(y + (size_t)blockIdx.x * H);
    const float4* g4   = reinterpret_cast<const float4*>(gamma);

    float4 buf[ITEMS];                                // 本线程名下的数据，驻留寄存器
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        int i = threadIdx.x + k * blockDim.x;         // 同一轮各线程地址连续 → 合并访存
        float4 v = row4[i];                           // 唯一一次显存读（16B/指令）
        buf[k] = v;
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;   // 加载时顺手累加
    }
    acc = blockReduceSum(acc);
    float rrms = rsqrtf(acc / H + eps);

    #pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        int i = threadIdx.x + k * blockDim.x;
        float4 v = buf[k], g = g4[i], o;              // 数据来自寄存器，不再读 x
        o.x = v.x * rrms * g.x;
        o.y = v.y * rrms * g.y;
        o.z = v.z * rrms * g.z;
        o.w = v.w * rrms * g.w;
        out4[i] = o;
    }
}

// ===========================================================================
// V3：行级调度——短行一 Warp（第 7 章）
// 每 Block 处理 blockDim.x/32 行
// 要求 H = ITEMS * 32 * 4（H=4096 → ITEMS=32）
// 启动：行数 N，每 Block 8 行：rmsnorm_v3<32><<<(N+7)/8, 256>>>(x, y, gamma, N, H, eps);
// ===========================================================================
template <int ITEMS>
__global__ void rmsnorm_v3(const float* __restrict__ x, float* __restrict__ y,
                           const float* __restrict__ gamma, int N, int H, float eps) {
    int lane = threadIdx.x & 31;
    int row_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // 全局 Warp 编号 = 行号
    if (row_id >= N) return;

    const float4* row4 = reinterpret_cast<const float4*>(x + (size_t)row_id * H);
    float4*       out4 = reinterpret_cast<float4*>(y + (size_t)row_id * H);
    const float4* g4   = reinterpret_cast<const float4*>(gamma);

    float4 buf[ITEMS];
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        int i = lane + k * 32;                        // Warp 内 32 个 lane 地址连续
        float4 v = row4[i];
        buf[k] = v;
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    acc = warpReduceSum(acc);                         // 只需 Warp 内归约：零同步
    float rrms = rsqrtf(acc / H + eps);

    #pragma unroll
    for (int k = 0; k < ITEMS; k++) {
        int i = lane + k * 32;
        float4 v = buf[k], g = g4[i], o;
        o.x = v.x * rrms * g.x;  o.y = v.y * rrms * g.y;
        o.z = v.z * rrms * g.z;  o.w = v.w * rrms * g.w;
        out4[i] = o;
    }
}

// ===========================================================================
// CPU 参考实现（double 累加，保证精度基准）
// ===========================================================================
static void rmsnorm_cpu(const float* x, float* y, const float* gamma,
                        int N, int H, float eps) {
    for (int r = 0; r < N; ++r) {
        const float* row = x + (size_t)r * H;
        float* out = y + (size_t)r * H;
        double ss = 0.0;
        for (int i = 0; i < H; ++i) ss += (double)row[i] * (double)row[i];
        double rrms = 1.0 / sqrt(ss / H + (double)eps);
        for (int i = 0; i < H; ++i)
            out[i] = (float)((double)row[i] * rrms * (double)gamma[i]);
    }
}

// ===========================================================================
// 宿主端驱动：随机数据 + CPU 参考校验 + cudaEvent 计时 + 有效带宽
// ===========================================================================

enum Version { V0, V1, V2, V3 };

// 编译期固定 ITEMS 供 V2/V3 模板使用（H=4096 时：V2 ITEMS=4，V3 ITEMS=32）
#define V2_ITEMS 4
#define V3_ITEMS 32

static void launch(Version v, const float* d_x, float* d_y, const float* d_g,
                   int N, int H, float eps) {
    switch (v) {
        case V0:
            rmsnorm_v0<<<N, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
                d_x, d_y, d_g, H, eps);
            break;
        case V1:
            rmsnorm_v1<<<N, BLOCK_SIZE>>>(d_x, d_y, d_g, H, eps);
            break;
        case V2:
            rmsnorm_v2<V2_ITEMS><<<N, BLOCK_SIZE>>>(d_x, d_y, d_g, H, eps);
            break;
        case V3:
            rmsnorm_v3<V3_ITEMS><<<(N + 7) / 8, BLOCK_SIZE>>>(d_x, d_y, d_g, N, H, eps);
            break;
    }
}

// 校验：与 CPU 参考的最大绝对误差
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

    if (H % 4 != 0) {
        fprintf(stderr, "H 必须是 4 的倍数（float4 向量化要求），当前 H=%d\n", H);
        return EXIT_FAILURE;
    }
    // V2/V3 的 ITEMS 是编译期常量，只在默认 H=4096 时严格匹配。
    // H != 4096 时 V0/V1 仍正确；V2/V3 依赖 H = V2_ITEMS*BLOCK_SIZE*4 = V3_ITEMS*32*4。
    bool v2v3_valid = (H == V2_ITEMS * BLOCK_SIZE * 4) && (H == V3_ITEMS * 32 * 4);

    printf("RMSNorm  N = %d, H = %d, eps = %g\n\n", N, H, eps);

    size_t total = (size_t)N * H;
    size_t bytes = total * sizeof(float);

    // 主机数据：随机初始化（文档强调必须随机数据，全同值会掩盖归约错误）
    float* h_x     = (float*)malloc(bytes);
    float* h_gamma = (float*)malloc(H * sizeof(float));
    float* h_y_ref = (float*)malloc(bytes);
    float* h_y_gpu = (float*)malloc(bytes);

    srand(1234);
    for (size_t i = 0; i < total; ++i)
        h_x[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;      // [-1, 1]
    for (int i = 0; i < H; ++i)
        h_gamma[i] = ((float)rand() / RAND_MAX) + 0.5f;       // 非平凡 γ ∈ [0.5, 1.5]

    // CPU 参考
    rmsnorm_cpu(h_x, h_y_ref, h_gamma, N, H, eps);

    float *d_x, *d_y, *d_g;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_g, H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, h_gamma, H * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const char* names[] = {"V0", "V1", "V2", "V3"};
    Version vers[] = {V0, V1, V2, V3};

    printf("%-6s %14s %12s %10s\n", "Ver", "max_abs_err", "time(ms)", "GB/s");
    printf("------------------------------------------------------\n");

    for (int i = 0; i < 4; ++i) {
        Version v = vers[i];
        if ((v == V2 || v == V3) && !v2v3_valid) {
            printf("%-6s %14s %12s %10s  (需 H=4096)\n", names[i], "-", "-", "-");
            continue;
        }

        // 预热
        CUDA_CHECK(cudaMemset(d_y, 0, bytes));
        launch(v, d_x, d_y, d_g, N, H, eps);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 计时
        CUDA_CHECK(cudaEventRecord(start));
        launch(v, d_x, d_y, d_g, N, H, eps);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, bytes, cudaMemcpyDeviceToHost));
        float err = max_abs_err(h_y_gpu, h_y_ref, total);

        // 有效带宽：读 x 一遍 + 写 y 一遍 = 2NH*4B
        double gbps = 2.0 * total * sizeof(float) / (ms * 1e-3) / 1e9;

        bool ok = err < 1e-3f;
        printf("%-6s %14.2e %12.4f %10.1f  %s\n",
               names[i], err, ms, gbps, ok ? "OK" : "FAIL");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_g));
    free(h_x); free(h_gamma); free(h_y_ref); free(h_y_gpu);

    printf("\n有效带宽口径: 读 x 一遍 + 写 y 一遍 = 2*N*H*4B / t\n");
    return 0;
}
