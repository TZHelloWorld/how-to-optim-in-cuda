// rmsnorm_kernel.cu — PyTorch CUDA 扩展（对应文档第 9 章 PyTorch 对拍）
//
// 提供两个算子:
//   rmsnorm(x, gamma, eps)                  —— 标准 RMSNorm（V2：float4 + 行驻留）
//   fused_add_rmsnorm(x, residual, gamma, eps) —— V4 融合，原地改写 x(→y)、residual(→h)
//
// 与文档第 6/8 章的 kernel 一致，仅补充 PyTorch 绑定上下文。
// 要求 H 是 (blockDim.x * 4) 的倍数（float4 向量化）；这里按 ITEMS 通过循环处理任意
// H = ITEMS * blockDim.x * 4，ITEMS 在 host 侧计算并作为运行时上界（保持算法逻辑不变）。

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// Warp / Block 两级归约（第 5 章）
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warpReduceSum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

__device__ __forceinline__ float blockReduceSum(float v) {
    __shared__ float warpRes[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warpReduceSum(v);
    if (lane == 0) warpRes[wid] = v;
    __syncthreads();
    int nWarp = (blockDim.x + 31) >> 5;
    v = (lane < nWarp) ? warpRes[lane] : 0.f;
    if (wid == 0) v = warpReduceSum(v);
    __shared__ float result;
    if (threadIdx.x == 0) result = v;
    __syncthreads();
    return result;
}

// ---------------------------------------------------------------------------
// V2：float4 向量化 + 行驻留寄存器（第 6 章）
// 运行时版本：H4 = H/4 个 float4，各线程跨步累加/缩放（等价于文档模板逻辑）。
// ---------------------------------------------------------------------------
__global__ void rmsnorm_v2_kernel(const float* __restrict__ x, float* __restrict__ y,
                                  const float* __restrict__ gamma, int H, float eps) {
    const float4* row4 = reinterpret_cast<const float4*>(x + (size_t)blockIdx.x * H);
    float4*       out4 = reinterpret_cast<float4*>(y + (size_t)blockIdx.x * H);
    const float4* g4   = reinterpret_cast<const float4*>(gamma);
    int H4 = H >> 2;                                  // float4 个数

    float acc = 0.f;
    for (int i = threadIdx.x; i < H4; i += blockDim.x) {
        float4 v = row4[i];
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    acc = blockReduceSum(acc);
    float rrms = rsqrtf(acc / H + eps);

    for (int i = threadIdx.x; i < H4; i += blockDim.x) {
        float4 v = row4[i], g = g4[i], o;
        o.x = v.x * rrms * g.x;  o.y = v.y * rrms * g.y;
        o.z = v.z * rrms * g.z;  o.w = v.w * rrms * g.w;
        out4[i] = o;
    }
}

// ---------------------------------------------------------------------------
// V4：residual + RMSNorm 融合（第 8 章），原地改写
// ---------------------------------------------------------------------------
__global__ void fused_add_rmsnorm_kernel(float* __restrict__ x, float* __restrict__ residual,
                                         const float* __restrict__ gamma, int H, float eps) {
    float4* xrow = reinterpret_cast<float4*>(x + (size_t)blockIdx.x * H);
    float4* rrow = reinterpret_cast<float4*>(residual + (size_t)blockIdx.x * H);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    int H4 = H >> 2;

    float acc = 0.f;
    for (int i = threadIdx.x; i < H4; i += blockDim.x) {
        float4 a = xrow[i], b = rrow[i];
        float4 h;
        h.x = a.x + b.x;  h.y = a.y + b.y;
        h.z = a.z + b.z;  h.w = a.w + b.w;
        rrow[i] = h;                                  // 新残差写回
        acc += h.x * h.x + h.y * h.y + h.z * h.z + h.w * h.w;
    }
    acc = blockReduceSum(acc);
    float rrms = rsqrtf(acc / H + eps);

    for (int i = threadIdx.x; i < H4; i += blockDim.x) {
        float4 h = rrow[i], g = g4[i], o;             // 从写回的 residual 取 h（原地语义）
        o.x = h.x * rrms * g.x;  o.y = h.y * rrms * g.y;
        o.z = h.z * rrms * g.z;  o.w = h.w * rrms * g.w;
        xrow[i] = o;
    }
}

// ---------------------------------------------------------------------------
// Host 端绑定
// ---------------------------------------------------------------------------
torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor gamma, double eps) {
    TORCH_CHECK(x.is_cuda(), "x 必须在 CUDA 上");
    TORCH_CHECK(x.dim() == 2, "x 必须是 [N, H]");
    auto xc = x.contiguous().to(torch::kFloat32);
    auto gc = gamma.contiguous().to(torch::kFloat32);
    int N = xc.size(0), H = xc.size(1);
    TORCH_CHECK(H % 4 == 0, "H 必须是 4 的倍数（float4 向量化）");

    auto y = torch::empty_like(xc);
    rmsnorm_v2_kernel<<<N, BLOCK_SIZE>>>(
        xc.data_ptr<float>(), y.data_ptr<float>(),
        gc.data_ptr<float>(), H, (float)eps);
    return y;
}

// 原地改写：x -> y（归一化输出），residual -> h（新残差）
void fused_add_rmsnorm(torch::Tensor x, torch::Tensor residual,
                       torch::Tensor gamma, double eps) {
    TORCH_CHECK(x.is_cuda() && residual.is_cuda(), "输入必须在 CUDA 上");
    TORCH_CHECK(x.dim() == 2, "x 必须是 [N, H]");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "本教学版仅支持 fp32");
    auto xc = x.contiguous();
    auto rc = residual.contiguous();
    auto gc = gamma.contiguous().to(torch::kFloat32);
    int N = xc.size(0), H = xc.size(1);
    TORCH_CHECK(H % 4 == 0, "H 必须是 4 的倍数（float4 向量化）");

    fused_add_rmsnorm_kernel<<<N, BLOCK_SIZE>>>(
        xc.data_ptr<float>(), rc.data_ptr<float>(),
        gc.data_ptr<float>(), H, (float)eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &rmsnorm, "RMSNorm (V2: float4 + row-resident)");
    m.def("fused_add_rmsnorm", &fused_add_rmsnorm,
          "Fused residual add + RMSNorm (V4, in-place)");
}
