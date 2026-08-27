// gemm_kernel.cu — PyTorch CUDA 扩展（对应文档第 12 章）
// 以 sgemm_v5（float4 向量化 + As 转置存储）为例封装成 Python 可调用算子。

#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// sgemm_v5：float4 向量化 + 共享内存布局重排（第 9 章）
// 分块参数在下面 my_matmul 中固定为 BM=BN=128, BK=8, TM=TN=8
// 要求 M、N 被 128 整除，K 被 8 整除。
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_v5(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    __shared__ float As[BK * BM];    // 转置布局 [BK][BM]
    __shared__ float Bs[BK * BN];    // 正常布局 [BK][BN]

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    const int innerRowA = threadIdx.x / (BK / 4);
    const int innerColA = threadIdx.x % (BK / 4);
    const int innerRowB = threadIdx.x / (BN / 4);
    const int innerColB = threadIdx.x % (BN / 4);

    float acc[TM][TN] = {{0.0f}};
    float regA[TM], regB[TN];

    for (int t = 0; t < K; t += BK) {
        float4 ta = reinterpret_cast<const float4*>(
                        &A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = ta.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = ta.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = ta.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = ta.w;

        reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();
        A += BK;
        B += BK * N;

        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < TM; i++)
                regA[i] = As[k * BM + threadRow * TM + i];
            for (int j = 0; j < TN; j++)
                regB[j] = Bs[k * BN + threadCol * TN + j];
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j += 4) {
            float4 out = {acc[i][j], acc[i][j+1], acc[i][j+2], acc[i][j+3]};
            reinterpret_cast<float4*>(
                &C[(threadRow * TM + i) * N + threadCol * TN + j])[0] = out;
        }
}

torch::Tensor my_matmul(torch::Tensor A, torch::Tensor B) {
    // 输入检查：设备、维度、形状匹配
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "expect CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "expect 2-D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "shape mismatch");

    // contiguous()：kernel 假定行主序连续存储，转置视图等必须先物化
    auto Ac = A.contiguous().to(torch::kFloat32);
    auto Bc = B.contiguous().to(torch::kFloat32);
    int M = Ac.size(0), K = Ac.size(1), N = Bc.size(1);

    // sgemm_v5 分块要求 M、N 被 128 整除，K 被 8 整除
    TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 8 == 0,
                "M/N must be multiples of 128 and K a multiple of 8 for sgemm_v5");

    auto C = torch::empty({M, N}, Ac.options());

    constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block((BM * BN) / (TM * TN));      // 256

    sgemm_v5<BM, BN, BK, TM, TN><<<grid, block>>>(
        M, N, K,
        Ac.data_ptr<float>(), Bc.data_ptr<float>(), C.data_ptr<float>());
    return C;
}

// PYBIND11_MODULE：把 C++ 函数导出为 Python 模块中的函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_matmul", &my_matmul, "SGEMM v5");
}
