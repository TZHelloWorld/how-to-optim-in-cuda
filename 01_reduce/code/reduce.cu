// reduce.cu — CUDA Reduce（求和归约）从 V0 到 V7 的完整可运行实现
//
// 对应文档: ../cuda_reduce_optimization_guide.md
// 覆盖版本:
//   V0 朴素树形归约（基准）
//   V1 Strided Index 连续线程映射（消除 Warp Divergence）
//   V2 步长从大到小 + tid<s（消除 Bank Conflict）
//   V3 每线程加载 2 元素预相加（提高线程利用率）
//   V4 手动展开最后一个 Warp（减少同步开销）
//   V5 模板参数编译期完全展开
//   V6 Warp Shuffle 两级归约（绕过共享内存）
//   V7 float4 向量化 + Grid Stride Loop（榨干带宽）
//
// 编译:
//   nvcc -O3 -arch=sm_70 reduce.cu -o reduce
// 运行:
//   ./reduce            # 默认 32M 元素
//   ./reduce 1048576    # 指定元素个数

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

// ---------------------------------------------------------------------------
// V0：朴素树形归约（基准）
// ---------------------------------------------------------------------------
__global__ void reduce_v0(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (int step = 1; step < blockDim.x; step *= 2) {
        if (tid % (2 * step) == 0) {
            smem[tid] += smem[tid + step];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// ---------------------------------------------------------------------------
// V1：连续线程映射（消除 Warp Divergence）
// ---------------------------------------------------------------------------
__global__ void reduce_v1(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = threadIdx.x * 2 * s;
        if (index < blockDim.x) {
            smem[index] += smem[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// ---------------------------------------------------------------------------
// V2：顺序寻址，步长从大到小（消除 Bank Conflict）
// ---------------------------------------------------------------------------
__global__ void reduce_v2(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    smem[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// ---------------------------------------------------------------------------
// V3：每线程加载 2 个元素并预相加（提高线程利用率）
// 注意: 此版本每个 Block 处理 2*blockDim.x 个元素
// ---------------------------------------------------------------------------
__global__ void reduce_v3(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// ---------------------------------------------------------------------------
// V4：手动展开最后一个 Warp（减少同步开销）
// ---------------------------------------------------------------------------
__device__ void warpReduce(volatile float* smem, int tid) {
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid +  8];
    smem[tid] += smem[tid +  4];
    smem[tid] += smem[tid +  2];
    smem[tid] += smem[tid +  1];
}

__global__ void reduce_v4(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];
    smem[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(smem, tid);
    }

    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// ---------------------------------------------------------------------------
// V5：模板参数编译期完全展开
// ---------------------------------------------------------------------------
template <int BLK>
__global__ void reduce_v5(float* input, float* output, int n) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * (BLK * 2) + threadIdx.x;

    float val = 0.0f;
    if (gid < n)        val += input[gid];
    if (gid + BLK < n)  val += input[gid + BLK];
    smem[tid] = val;
    __syncthreads();

    if (BLK >= 512) { if (tid < 256) smem[tid] += smem[tid + 256]; __syncthreads(); }
    if (BLK >= 256) { if (tid < 128) smem[tid] += smem[tid + 128]; __syncthreads(); }
    if (BLK >= 128) { if (tid <  64) smem[tid] += smem[tid +  64]; __syncthreads(); }

    if (tid < 32) {
        volatile float* vsmem = smem;
        if (BLK >= 64) vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if (tid == 0) output[blockIdx.x] = smem[0];
}

// ---------------------------------------------------------------------------
// V6：Warp Shuffle 两级归约（绕过共享内存）
// ---------------------------------------------------------------------------
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_v6(float* input, float* output, int n) {
    int tid  = threadIdx.x;
    int gid  = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int lane = tid % 32;
    int wid  = tid / 32;

    float val = 0.0f;
    if (gid < n)              val += input[gid];
    if (gid + blockDim.x < n) val += input[gid + blockDim.x];

    val = warpReduceSum(val);

    __shared__ float warp_results[32];
    if (lane == 0) {
        warp_results[wid] = val;
    }
    __syncthreads();

    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        val = warpReduceSum(val);
    }

    if (tid == 0) output[blockIdx.x] = val;
}

// ---------------------------------------------------------------------------
// V7：float4 向量化 + Grid Stride Loop（榨干带宽）
// 单次 kernel 处理任意长度 n，产出 gridDim.x 个部分和
// ---------------------------------------------------------------------------
__global__ void reduce_v7(float* input, float* output, int n) {
    int tid  = threadIdx.x;
    int lane = tid % 32;
    int wid  = tid / 32;

    float4* input4 = reinterpret_cast<float4*>(input);
    int n4 = n / 4;

    float val = 0.0f;

    for (int idx = blockIdx.x * blockDim.x + tid;
         idx < n4;
         idx += gridDim.x * blockDim.x) {
        float4 data = input4[idx];
        val += data.x + data.y + data.z + data.w;
    }

    int tail_start = n4 * 4;
    for (int idx = tail_start + blockIdx.x * blockDim.x + tid;
         idx < n;
         idx += gridDim.x * blockDim.x) {
        val += input[idx];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    __shared__ float warp_results[32];
    if (lane == 0) warp_results[wid] = val;
    __syncthreads();

    int num_warps = blockDim.x / 32;
    if (wid == 0) {
        val = (lane < num_warps) ? warp_results[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }

    if (tid == 0) output[blockIdx.x] = val;
}

// ===========================================================================
// 宿主端驱动：多级归约（反复调用 kernel 直到只剩 1 个值），并计时
// ===========================================================================

enum Version { V0, V1, V2, V3, V4, V5, V6, V7 };

// 每个 Block 消费的元素个数：V0/V1/V2 为 blockDim；V3~V6 为 2*blockDim；V7 固定网格
static int elems_per_block(Version v) {
    switch (v) {
        case V0: case V1: case V2: return BLOCK_SIZE;
        default:                   return BLOCK_SIZE * 2;
    }
}

static void launch_once(Version v, float* d_in, float* d_out, int n, int grid) {
    size_t smem = BLOCK_SIZE * sizeof(float);
    switch (v) {
        case V0: reduce_v0<<<grid, BLOCK_SIZE, smem>>>(d_in, d_out, n); break;
        case V1: reduce_v1<<<grid, BLOCK_SIZE, smem>>>(d_in, d_out, n); break;
        case V2: reduce_v2<<<grid, BLOCK_SIZE, smem>>>(d_in, d_out, n); break;
        case V3: reduce_v3<<<grid, BLOCK_SIZE, smem>>>(d_in, d_out, n); break;
        case V4: reduce_v4<<<grid, BLOCK_SIZE, smem>>>(d_in, d_out, n); break;
        case V5: reduce_v5<BLOCK_SIZE><<<grid, BLOCK_SIZE, smem>>>(d_in, d_out, n); break;
        case V6: reduce_v6<<<grid, BLOCK_SIZE>>>(d_in, d_out, n); break;
        case V7: reduce_v7<<<grid, BLOCK_SIZE>>>(d_in, d_out, n); break;
    }
}

// 反复调用直到收敛为单值，返回最终标量（存于 d_buf_a[0]）
static float reduce_full(Version v, float* d_buf_a, float* d_buf_b, int n) {
    float* cur = d_buf_a;
    float* nxt = d_buf_b;
    int m = n;

    if (v == V7) {
        // V7 用固定网格：第一轮大幅缩小规模，后续用 V2 收尾
        int grid = 1024;
        reduce_v7<<<grid, BLOCK_SIZE>>>(cur, nxt, m);
        CUDA_CHECK(cudaGetLastError());
        std::swap(cur, nxt);
        m = grid;
        // 收尾：用 V2 逐级归约
        while (m > 1) {
            int g = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
            reduce_v2<<<g, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(cur, nxt, m);
            CUDA_CHECK(cudaGetLastError());
            std::swap(cur, nxt);
            m = g;
        }
    } else {
        int epb = elems_per_block(v);
        while (m > 1) {
            int grid = (m + epb - 1) / epb;
            launch_once(v, cur, nxt, m, grid);
            CUDA_CHECK(cudaGetLastError());
            std::swap(cur, nxt);
            m = grid;
        }
    }

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&result, cur, sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : (32 * 1024 * 1024);
    printf("Reduce sum over N = %d elements\n\n", n);

    // 主机数据：全部填 1.0，期望和 == n
    float* h_in = (float*)malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i) h_in[i] = 1.0f;
    double expected = (double)n;

    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_a, sizeof(float) * n));
    // 输出缓冲区最多需要 ceil(n / epb_min) 个元素；用 n 保守分配
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(float) * n));

    const char* names[] = {"V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"};
    Version vers[] = {V0, V1, V2, V3, V4, V5, V6, V7};

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("%-6s %18s %12s %10s\n", "Ver", "result", "time(ms)", "GB/s");
    printf("--------------------------------------------------------\n");

    for (int i = 0; i < 8; ++i) {
        // 每次重置输入
        CUDA_CHECK(cudaMemcpy(d_a, h_in, sizeof(float) * n, cudaMemcpyHostToDevice));

        // 预热
        {
            float* a = d_a; float* b = d_b;
            // 拷贝一份到临时，避免破坏 d_a（此处 warmup 直接跑，随后重置）
            (void)a; (void)b;
        }

        CUDA_CHECK(cudaEventRecord(start));
        float result = reduce_full(vers[i], d_a, d_b, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        double gbps = (double)n * sizeof(float) / (ms * 1e-3) / 1e9;

        bool ok = fabs((double)result - expected) < 1e-3 * expected + 1.0;
        printf("%-6s %18.1f %12.4f %10.1f  %s\n",
               names[i], result, ms, gbps, ok ? "OK" : "FAIL");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_in);
    printf("\nExpected sum = %.1f\n", expected);
    return 0;
}
