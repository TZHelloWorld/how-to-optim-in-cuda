// ===========================================================================
// main.cu — 全局内存访问模式带宽基准测试
//
// 复现 CUDA C++ Best Practices Guide 的两个经典实验：
//   1. Offset 实验（§10.2.1.3）：非对齐访问对有效带宽的影响
//      预期：offset 非 0 时带宽降至基准的 ~90%（L1 缓存行复用补偿，
//            理论值为 4/5 = 80%），offset=32 个 float（128 字节）时重新对齐。
//   2. Stride 实验（§10.2.1.4）：跨步访问对有效带宽的影响
//      预期：stride=2 约 50% 效率，stride=32 时降至 ~3%。
//
// 原理详见同目录 README.md 第 2.5.3 节与第 5.4 节。
//
// 编译运行：
//   nvcc -O3 -o mem_benchmark main.cu && ./mem_benchmark
// ===========================================================================
#include <cstdio>

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUDA 运行时错误检查宏
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr_to_check)                              \
    do {                                                       \
        cudaError_t result = (expr_to_check);                  \
        if (result != cudaSuccess) {                           \
            fprintf(stderr,                                    \
                    "CUDA error at %s:%d code=%d(%s)\n",       \
                    __FILE__,                                  \
                    __LINE__,                                  \
                    result,                                    \
                    cudaGetErrorString(result));               \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// 实验配置
//
// 两组实验的规模分开设置：
//   - offset 实验寻址上界 = kOffsetElems + kMaxOffset（约 64 MB）
//   - stride 实验寻址上界 = kStrideElems * kMaxStride（4M * 32 = 128M 元素，
//     512 MB/缓冲区），如显存不足可进一步调小 kStrideElems
// 注意：数组应显著大于 L2 容量，否则测得的是缓存带宽而非 DRAM 带宽。
// ---------------------------------------------------------------------------
constexpr int kBlockSize   = 256;                   // 线程块大小（warp 的整数倍）
constexpr int kOffsetElems = 16 * 1024 * 1024;      // offset 实验：16M float = 64 MB
constexpr int kStrideElems = 4 * 1024 * 1024;       // stride 实验：4M float
constexpr int kMaxStride   = 32;                    // 最大跨步（元素数）
constexpr int kMaxOffset   = 32;                    // 最大偏移（元素数）
constexpr int kRepeat      = 20;                    // 每个配置的计时重复次数

// ---------------------------------------------------------------------------
// Kernel 1：Offset 访问
//
// 每个线程读写地址 (tid + offset)。offset 以 float 元素为单位：
//   offset=0                     -> 完全对齐（基准）
//   offset=1..31                 -> warp 请求跨越 5 个 32 字节段（非对齐）
//   offset=8 (32B) / 32 (128B)   -> 重新落在段/缓存行边界上
// ---------------------------------------------------------------------------
__global__ void offset_access(float* out, const float* in, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;
    out[i] = in[i] + 1.0f;
}

// ---------------------------------------------------------------------------
// Kernel 2：Strided 访问
//
// 每个线程读写地址 (tid * stride)。stride 越大，warp 的 128 字节总请求
// 分散到越多的 32 字节段中，每个段的有效字节越少：
//   stride=1  -> 4 个段，100% 利用率
//   stride=2  -> 8 个段， 50% 利用率
//   stride=32 -> 32 个段， ~3% 利用率
// ---------------------------------------------------------------------------
__global__ void stride_access(float* out, const float* in, int stride) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    out[i] = in[i] + 1.0f;
}

// ---------------------------------------------------------------------------
// 计时工具：用 cudaEvent 测量 kernel 的平均执行时间（毫秒）
// ---------------------------------------------------------------------------
template <typename LaunchFn>
static float time_kernel_ms(LaunchFn launch) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    launch();                                   // 热身（不计时）
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < kRepeat; ++r) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / kRepeat;
}

// 有效带宽（GB/s）：每个元素 1 读 + 1 写，共 2 * n * sizeof(float) 字节
static double effective_bandwidth_gbs(float ms, int n) {
    double bytes = 2.0 * n * sizeof(float);
    return bytes / (ms * 1e-3) / 1e9;
}

int main() {
    // -----------------------------------------------------------------------
    // 设备信息
    // -----------------------------------------------------------------------
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Offset array: %zu MB, stride array: %zu MB (x%d), block size: %d\n\n",
           (size_t)kOffsetElems * sizeof(float) / (1024 * 1024),
           (size_t)kStrideElems * sizeof(float) / (1024 * 1024),
           kMaxStride, kBlockSize);

    // -----------------------------------------------------------------------
    // 内存分配
    // 按两组实验寻址上界的较大者分配，两组实验复用同一块缓冲区。
    // cudaMalloc 保证 >=256 字节对齐（README 第 2.5.4 节），offset=0 即为对齐基准。
    // -----------------------------------------------------------------------
    size_t offset_upper = (size_t)kOffsetElems + kMaxOffset;
    size_t stride_upper = (size_t)kStrideElems * kMaxStride;
    size_t alloc_elems  = offset_upper > stride_upper ? offset_upper : stride_upper;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, alloc_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, alloc_elems * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_in, 0, alloc_elems * sizeof(float)));

    // -----------------------------------------------------------------------
    // 实验 1：Offset 访问
    // -----------------------------------------------------------------------
    printf("=== Offset access (out[i+offset] = in[i+offset] + 1) ===\n");
    printf("%6s  %20s  %10s\n", "offset", "bandwidth (GB/s)", "relative");

    int grid_offset = (kOffsetElems + kBlockSize - 1) / kBlockSize;
    double base_bw = 0.0;
    for (int offset = 0; offset <= kMaxOffset; ++offset) {
        float ms = time_kernel_ms([&] {
            offset_access<<<grid_offset, kBlockSize>>>(d_out, d_in, offset);
        });
        double bw = effective_bandwidth_gbs(ms, kOffsetElems);
        if (offset == 0) {
            base_bw = bw;
        }
        printf("%6d  %20.2f  %9.1f%%\n", offset, bw, bw / base_bw * 100.0);
    }

    // -----------------------------------------------------------------------
    // 实验 2：Strided 访问
    // -----------------------------------------------------------------------
    printf("\n=== Strided access (out[i*stride] = in[i*stride] + 1) ===\n");
    printf("%6s  %20s  %10s\n", "stride", "bandwidth (GB/s)", "relative");

    int grid_stride = (kStrideElems + kBlockSize - 1) / kBlockSize;
    for (int stride = 1; stride <= kMaxStride; stride *= 2) {
        float ms = time_kernel_ms([&] {
            stride_access<<<grid_stride, kBlockSize>>>(d_out, d_in, stride);
        });
        double bw = effective_bandwidth_gbs(ms, kStrideElems);
        if (stride == 1) {
            base_bw = bw;
        }
        printf("%6d  %20.2f  %9.1f%%\n", stride, bw, bw / base_bw * 100.0);
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
