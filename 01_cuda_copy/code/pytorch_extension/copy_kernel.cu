// copy_kernel.cu — PyTorch CUDA 扩展：SM 拷贝 kernel（对应文档第 4、6 章）
//
// 提供两个入口:
//   copy_float4(x)             纯拷贝，float4 向量化 + grid-stride loop（§4.3 V2）
//   copy_scale_relu(x, alpha)  拷贝 + scale + ReLU 融合（§6.1）
//
// kernel 算法逻辑与 standalone 的 copy_bench.cu 完全一致，此处只是接入 PyTorch。

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// float4 向量化拷贝：一条 LDG.128 搬 16 B，在途字节 ×4（§4.3 V2）
__global__ void copy_float4_kernel(const float4* __restrict__ src,
                                   float4* __restrict__ dst, size_t n4) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride)
        dst[i] = src[i];                        // LDG.128 + STG.128
}

// 拷贝 + scale + ReLU 融合（§6.1）：计算藏进访存的影子里
__global__ void copy_scale_relu_kernel(const float4* __restrict__ src,
                                        float4* __restrict__ dst,
                                        size_t n4, float alpha) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = src[i];
        v.x = fmaxf(v.x * alpha, 0.f);
        v.y = fmaxf(v.y * alpha, 0.f);
        v.z = fmaxf(v.z * alpha, 0.f);
        v.w = fmaxf(v.w * alpha, 0.f);
        dst[i] = v;
    }
}

// 纯拷贝（要求元素数为 4 的倍数以走 float4 路径）
torch::Tensor copy_float4(torch::Tensor input) {
    auto x = input.contiguous().to(torch::kFloat32);
    int64_t n = x.numel();
    TORCH_CHECK(n % 4 == 0, "元素数必须是 4 的倍数（float4 向量化）");
    auto out = torch::empty_like(x);

    const int block = 256;
    const int grid  = 2048;
    copy_float4_kernel<<<grid, block>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        n / 4);
    return out;
}

// 拷贝 + scale + ReLU 融合
torch::Tensor copy_scale_relu(torch::Tensor input, double alpha) {
    auto x = input.contiguous().to(torch::kFloat32);
    int64_t n = x.numel();
    TORCH_CHECK(n % 4 == 0, "元素数必须是 4 的倍数（float4 向量化）");
    auto out = torch::empty_like(x);

    const int block = 256;
    const int grid  = 2048;
    copy_scale_relu_kernel<<<grid, block>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        n / 4, (float)alpha);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("copy_float4", &copy_float4, "float4 vectorized copy (V2)");
    m.def("copy_scale_relu", &copy_scale_relu,
          "fused copy + scale + ReLU", py::arg("input"), py::arg("alpha") = 1.0);
}
