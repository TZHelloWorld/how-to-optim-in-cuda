# 第 3 章示例：在 PyTorch 自定义算子中使用 compute-sanitizer
# 本脚本用 load_inline JIT 编译一个埋了 off-by-one 越界 bug 的"数组逆序"算子。
#
# 正常运行（大概率"侥幸"通过）：
#   python sanitizer_torch_op.py
#
# 用 compute-sanitizer 排查（三件套：关缓存分配器 + 内核名过滤 + -lineinfo）：
#   PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
#   compute-sanitizer --kernel-name kns=reverse_kernel --error-exitcode 1 \
#       python sanitizer_torch_op.py
#
# 修复：把 cuda_src 中标注 BUG 的 i <= n 改为 i < n，删除扩展缓存后复验
#   （JIT 缓存目录：~/.cache/torch_extensions/）
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
__global__ void reverse_kernel(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n)                          // BUG：应为 i < n，差一错误（off-by-one）
        out[i] = in[n - 1 - i];          // i == n 时：写 out[n] 越界，读 in[-1] 越界
}

torch::Tensor reverse_op(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dtype() == torch::kFloat32,
                "expect contiguous float32 CUDA tensor");
    auto out = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    reverse_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

mod = load_inline(
    name="my_op",
    cpp_sources="torch::Tensor reverse_op(torch::Tensor x);",
    cuda_sources=cuda_src,
    functions=["reverse_op"],
    extra_cuda_cflags=["-O3", "-lineinfo"],   # -lineinfo：让 sanitizer 报告源码行号
)

# 故意取除不尽 256 的规模：启动 1024 个线程，i == 1000 的线程恰好越界
# （若改成 1024，越界线程根本不会被启动，bug 会被整除"藏"住——测试要覆盖这类形状）
n = 1000
x = torch.arange(n, dtype=torch.float32, device="cuda")
y = mod.reverse_op(x)
torch.cuda.synchronize()

# 校验通过也不代表没越界——越界写可能落在分配的填充区或隔壁 tensor 里
print("correct:", torch.equal(y, x.flip(0)))
