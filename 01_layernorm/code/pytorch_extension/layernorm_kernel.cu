// layernorm_kernel.cu — LayerNorm 的 PyTorch CUDA 扩展（对应文档第 9.3 节）
//
// 导出接口: layernorm(x, gamma, beta, eps) -> y
//   x     : [N, H] fp32 CUDA tensor（最后一维做归一化）
//   gamma : [H]    fp32 CUDA tensor
//   beta  : [H]    fp32 CUDA tensor
//   eps   : float
//
// kernel 采用文档 V2 的 Warp Shuffle 两级归约 + 单遍 (Σx, Σx²)，
// 与 PyTorch 官方 nn.LayerNorm 对拍（见 test.py）。

#include <torch/extension.h>
#include <cuda_runtime.h>

// (Σx, Σx²) 成对的 warp / block 归约组件（同 layernorm.cu 中的 V2）
__device__ __forceinline__ float2 warpReduceSum2(float2 v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v.x += __shfl_xor_sync(0xffffffff, v.x, offset);
        v.y += __shfl_xor_sync(0xffffffff, v.y, offset);
    }
    return v;
}

__device__ __forceinline__ float2 blockReduceSum2(float2 v) {
    __shared__ float2 warpRes[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warpReduceSum2(v);
    if (lane == 0) warpRes[wid] = v;
    __syncthreads();
    int nWarp = (blockDim.x + 31) >> 5;
    v = (lane < nWarp) ? warpRes[lane] : make_float2(0.f, 0.f);
    if (wid == 0) v = warpReduceSum2(v);
    __shared__ float2 result;
    if (threadIdx.x == 0) result = v;
    __syncthreads();
    return result;
}

__global__ void layernorm_kernel(const float* __restrict__ x, float* __restrict__ y,
                                 const float* __restrict__ gamma, const float* __restrict__ beta,
                                 int H, float eps) {
    const float* row = x + (size_t)blockIdx.x * H;
    float*       out = y + (size_t)blockIdx.x * H;

    float2 acc = make_float2(0.f, 0.f);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float v = row[i];
        acc.x += v;
        acc.y += v * v;
    }
    acc = blockReduceSum2(acc);

    float mean = acc.x / H;
    float rstd = rsqrtf(fmaxf(acc.y / H - mean * mean, 0.f) + eps);

    for (int i = threadIdx.x; i < H; i += blockDim.x)
        out[i] = (row[i] - mean) * rstd * gamma[i] + beta[i];
}

torch::Tensor layernorm(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dim");

    auto x_c = x.contiguous();
    auto g_c = gamma.contiguous();
    auto b_c = beta.contiguous();

    int H = x_c.size(x_c.dim() - 1);
    int N = x_c.numel() / H;

    auto y = torch::empty_like(x_c);

    const int block = 256;
    layernorm_kernel<<<N, block>>>(
        x_c.data_ptr<float>(), y.data_ptr<float>(),
        g_c.data_ptr<float>(), b_c.data_ptr<float>(),
        H, (float)eps);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm", &layernorm, "LayerNorm forward (CUDA)",
          py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5);
}
