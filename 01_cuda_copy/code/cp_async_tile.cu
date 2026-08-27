// cp_async_tile.cu — kernel 内拷贝原语 cp.async 演示（对应文档 §5.2）
//
// 对应文档: ../cuda_copy_operator_guide.md
// Ampere（SM 8.0+）引入 cp.async：一条指令让数据从全局内存直达共享内存，
// 不经过寄存器文件，且为异步语义（cuda::memcpy_async / cuda::pipeline）。
//   - 省寄存器：数据不再经寄存器中转
//   - 省指令：LDG + STS 两条变一条
//   - 真异步：提交后 warp 继续执行，配合 commit/wait 构成硬件流水
//
// 编译（需要 CUDA 11+ 且 GPU 计算能力 >= 8.0）:
//   nvcc -O3 -arch=sm_80 cp_async_tile.cu -o cp_async_tile
// 运行:
//   ./cp_async_tile

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda/pipeline>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// 每个 block 处理 1024 个 float（256 线程 × 每线程 float4）
// 忠实于文档 §5.2 的示例结构：每线程提交一段 16B 的异步拷贝 global -> shared，
// 消费时对 shared 里的数据做 *2.0f 后写出。
__global__ void tile_load(const float* __restrict__ g, float* __restrict__ out, int n) {
    __shared__ float s_buf[1024];
    auto pipe = cuda::make_pipeline();

    pipe.producer_acquire();
    // 每线程提交一段 16B 的异步拷贝：global -> shared，绕过寄存器
    cuda::memcpy_async(&s_buf[threadIdx.x * 4],
                       &g[blockIdx.x * 4096 + threadIdx.x * 4],
                       sizeof(float4), pipe);
    pipe.producer_commit();

    /* ... 这里可以先干别的活（计算上一块），拷贝在后台进行 ... */

    pipe.consumer_wait();          // 需要数据时才等待
    __syncthreads();
    out[blockIdx.x * blockDim.x + threadIdx.x] = s_buf[threadIdx.x] * 2.0f;
}

int main() {
    // 校验设备是否支持 cp.async（计算能力 >= 8.0）
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 8) {
        printf("%s: 计算能力 %d.%d < 8.0，cp.async 不受支持，跳过。\n",
               prop.name, prop.major, prop.minor);
        return 0;
    }

    const int block  = 256;                      // 每 block 256 线程
    const int blocks = 64;                       // 64 个 block
    // 注意：kernel 中每 block 从 g 读 4096 个 float（threadIdx.x*4 + blockIdx.x*4096），
    // 但只写出 blockDim.x = 256 个 float（对应 s_buf 的前 256 个元素）。
    const int g_elems   = blocks * 4096;         // 输入元素数
    const int out_elems = blocks * block;        // 输出元素数

    float* h_g = (float*)malloc(sizeof(float) * g_elems);
    for (int i = 0; i < g_elems; ++i) h_g[i] = (float)(i % 97) * 0.1f;

    float *d_g, *d_out;
    CUDA_CHECK(cudaMalloc(&d_g, sizeof(float) * g_elems));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * out_elems));
    CUDA_CHECK(cudaMemcpy(d_g, h_g, sizeof(float) * g_elems, cudaMemcpyHostToDevice));

    tile_load<<<blocks, block>>>(d_g, d_out, g_elems);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_out = (float*)malloc(sizeof(float) * out_elems);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(float) * out_elems, cudaMemcpyDeviceToHost));

    // 校验：out[b*256 + t] == g[b*4096 + t*4] * 2.0f
    //  （s_buf[t] 由线程 t 写入的 float4 的首元素，即 g[b*4096 + t*4]）
    bool ok = true;
    for (int b = 0; b < blocks && ok; ++b) {
        for (int t = 0; t < block; ++t) {
            float expect = h_g[b * 4096 + t * 4] * 2.0f;
            if (fabsf(h_out[b * block + t] - expect) > 1e-4f) { ok = false; break; }
        }
    }
    printf("cp.async tile_load: %s\n", ok ? "OK" : "FAIL");

    free(h_g);
    free(h_out);
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
