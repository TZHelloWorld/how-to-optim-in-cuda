// ce_query.cu — 查询设备的 Copy Engine 数量（对应文档 §2.2）
//
// 对应文档: ../cuda_copy_operator_guide.md
// asyncEngineCount 即 Copy Engine（async engine / DMA engine）的数量:
//   >= 1：拷贝可与 kernel 执行并行（SM 在算，CE 在搬）
//   >= 2：H2D 与 D2H 两个方向的拷贝还能彼此并行
//
// 编译:
//   nvcc ce_query.cu -o ce_query
// 运行:
//   ./ce_query

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("%s: asyncEngineCount = %d\n", p.name, p.asyncEngineCount);
    return 0;
}
