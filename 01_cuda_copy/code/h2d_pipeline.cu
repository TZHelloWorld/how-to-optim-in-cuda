// h2d_pipeline.cu — 大批量 H2D 的分块流水：让 CE 与 SM 重叠（对应文档 §6.3）
//
// 对应文档: ../cuda_copy_operator_guide.md
// 思路（官方博客 How to Overlap Data Transfers in CUDA C/C++）:
//   pinned 内存 + 分块 + 多流：
//   第 c+1 块在 Copy Engine 上传输时，第 c 块正在 SM 上计算。
//   理想总时间从 T_拷贝 + T_计算 降到 max(T_拷贝, T_计算) + 一个块的启动延迟。
//
// 本文件把文档 §6.3 的核心循环补齐成可运行程序：
//   - 对比"单流串行 H2D+计算"与"双流分块流水"的耗时
//   - process kernel 做一个简单的逐元素变换（x -> x * 2 + 1），便于正确性校验
//
// 编译:
//   nvcc -O3 -arch=native h2d_pipeline.cu -o h2d_pipeline
// 运行:
//   ./h2d_pipeline

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

// 一个占用一定时间的逐元素处理，代表"拷进来之后要做的计算"
__global__ void process(float* data, size_t n) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float v = data[i];
        // 人为增加一点算术负载，让计算时间可观察
        #pragma unroll
        for (int k = 0; k < 32; ++k) v = v * 1.0000001f + 0.0f;
        data[i] = v * 2.0f + 1.0f;               // 最终结果: x*2 + 1（近似）
    }
}

static bool verify(const float* h_src, const float* d_buf, size_t n) {
    float* h_out = (float*)malloc(sizeof(float) * n);
    CUDA_CHECK(cudaMemcpy(h_out, d_buf, sizeof(float) * n, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < n; ++i) {
        float expect = h_src[i] * 2.0f + 1.0f;
        if (fabsf(h_out[i] - expect) > 1e-2f) { ok = false; break; }
    }
    free(h_out);
    return ok;
}

int main() {
    const size_t n = 1ull << 26;                 // 2^26 个 float = 256 MiB
    const int block = 256;
    const int grid  = 1024;

    // pinned 主机内存：CE 可直接一跳 DMA（§2.4）
    float* h_buf;
    CUDA_CHECK(cudaMallocHost(&h_buf, n * sizeof(float)));
    for (size_t i = 0; i < n; ++i) h_buf[i] = (float)(i % 100) * 0.01f;

    float* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, n * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ---------------- 基线：单流串行（先整块拷贝，再计算） ----------------
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, n * sizeof(float), cudaMemcpyHostToDevice));
    process<<<grid, block>>>(d_buf, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_serial;
    CUDA_CHECK(cudaEventElapsedTime(&ms_serial, start, stop));
    bool ok_serial = verify(h_buf, d_buf, n);

    // ---------------- 分块流水：两条流交替，拷贝与计算重叠 ----------------
    // 以下循环结构忠实于文档 §6.3
    const int NCHUNK = 8;
    size_t chunk = n / NCHUNK;                    // 每块元素数（假设整除）

    cudaStream_t s[2];
    CUDA_CHECK(cudaStreamCreate(&s[0]));
    CUDA_CHECK(cudaStreamCreate(&s[1]));

    CUDA_CHECK(cudaMemset(d_buf, 0, n * sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    for (int c = 0; c < NCHUNK; c++) {
        cudaStream_t st = s[c & 1];               // 两条流交替
        CUDA_CHECK(cudaMemcpyAsync(d_buf + c * chunk, h_buf + c * chunk,
                                   chunk * sizeof(float), cudaMemcpyHostToDevice, st));
        process<<<grid, block, 0, st>>>(d_buf + c * chunk, chunk);
        // 同一条流内：拷贝 -> 计算 天然有序；
        // 两条流之间：块 c 的计算（SM）与块 c+1 的拷贝（CE）并行——
        // 这正是 Copy Engine 独立于 SM 存在的意义（§2.2）
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_pipe;
    CUDA_CHECK(cudaEventElapsedTime(&ms_pipe, start, stop));
    bool ok_pipe = verify(h_buf, d_buf, n);

    printf("H2D pipeline: N = %zu floats (%.1f MiB), NCHUNK = %d\n\n",
           n, n * sizeof(float) / (1024.0 * 1024.0), NCHUNK);
    printf("%-24s %10s   %s\n", "scheme", "time(ms)", "verify");
    printf("-------------------------------------------------\n");
    printf("%-24s %10.3f   %s\n", "serial (copy then compute)", ms_serial,
           ok_serial ? "OK" : "FAIL");
    printf("%-24s %10.3f   %s\n", "pipelined (2 streams)", ms_pipe,
           ok_pipe ? "OK" : "FAIL");

    CUDA_CHECK(cudaStreamDestroy(s[0]));
    CUDA_CHECK(cudaStreamDestroy(s[1]));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFreeHost(h_buf));
    printf("\n注: 流水版应快于串行版——H2D 传输躲进了 SM 计算的影子里 (§6.3)\n");
    return 0;
}
