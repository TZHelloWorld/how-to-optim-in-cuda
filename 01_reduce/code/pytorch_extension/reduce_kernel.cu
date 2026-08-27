// reduce_kernel.cu — PyTorch CUDA 扩展（对应文档第 12 章）
// 以 V0 kernel 为例，通过多级归约实现 reduce_sum。
// 换成任意版本只需替换 kernel 函数。

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void reduce_v0(float *g_idata, float *g_odata, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// 多级归约：反复调用 kernel 直到只剩 1 个值
torch::Tensor reduce_sum(torch::Tensor input) {
    auto current = input.contiguous().to(torch::kFloat32);
    int n = current.numel();

    while (n > 1) {
        int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        auto output = torch::zeros(grid_size, current.options());

        reduce_v0<<<grid_size, BLOCK_SIZE>>>(
            current.data_ptr<float>(),
            output.data_ptr<float>(),
            n
        );

        current = output;   // 本轮输出作为下一轮输入
        n = grid_size;
    }

    return current;   // shape [1]
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reduce_sum", &reduce_sum, "Reduce sum with V0 kernel");
}
