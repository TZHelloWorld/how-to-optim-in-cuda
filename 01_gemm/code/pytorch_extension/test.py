import torch, gemm_kernel

# sgemm_v5 要求 M、N 被 128 整除，K 被 8 整除
M = N = K = 4096
A = torch.randn(M, K, device='cuda')
B = torch.randn(K, N, device='cuda')

# 正确性：与 cuBLAS 对比（浮点求和顺序不同，需容忍相对误差）
C1 = gemm_kernel.my_matmul(A, B)
C2 = A @ B
print("allclose:", torch.allclose(C1, C2, rtol=1e-3, atol=1e-3))


# 性能：CUDA event 计时，注意预热与同步
def bench(fn, iters=20):
    for _ in range(3):
        fn()                                 # 预热（含 JIT/缓存效应）
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters         # ms


t = bench(lambda: gemm_kernel.my_matmul(A, B))
tflops = 2 * M * N * K / (t * 1e-3) / 1e12
tc = bench(lambda: A @ B)
print(f"my_matmul: {t:.3f} ms, {tflops:.1f} TFLOPS   (cuBLAS: {tc:.3f} ms)")
