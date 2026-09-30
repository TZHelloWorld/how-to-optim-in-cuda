# GEMM 通用优化的可运行示例

对应主文档第 13 章：13.2～13.11 将各算法与代码、数字例子逐项对照，[13.13 节](../../cuda_gemm_optimization_guide.md#1313-代码实践把优化方向落实到同一套计算核心) 汇总源码索引、运行和校验方法。这些示例使用同一 fp32 分块核心，让调度、数据布局与融合的作用容易观察；实际收益需在目标 GPU 上测量。

## 文件与计算约定

| 文件 | 内容 |
|------|------|
| `kernels.cuh` | 分块计算、形状/边界特化、Split-K、Sliced-K、Stream-K 两阶段、打包、融合、Batched/Grouped |
| `schedule.h` | CPU/GPU 共用索引；host 端 Stream-K 任务计划 |
| `gemm_advanced.cu` | 参数、资源管理、CPU 参考、GPU 校验、计时、L2 策略、调参缓存、Streams/Graphs |
| `schedule_test.cpp` | 不需要 GPU 的索引、覆盖、任务平衡和矩阵重建测试 |
| `CMakeLists.txt` | 有 CUDA 时构建全部；无 CUDA 时只构建 CPU 调度测试 |

普通输出为 `D = alpha * A @ B + beta * C_old`；融合输出再加列 bias 并执行 ReLU。A/B/C/D 均为 fp32、行主序，使用独立的 leading dimension。`C_old` 只读，D 与全部输入不重叠。

驱动故意设置 `lda=K+3、ldb=N+5、ldc=N+7、ldd=N+9`，并用非对称随机数据填充有效区域。输入 padding 为 NaN，输出每个测试前也填 NaN，防止遗漏写入和错误跨行访问被全零数据掩盖。

单矩阵参数要求 `1 <= M,N,K <= 65536`，内存和 CPU 参考计算时间仍受设备/主机资源限制。教学程序不处理零维 GEMM，也不接受转置布局或别名输出。

## 编译

使用 CUDA Toolkit 12+ 和它支持的主机 C++ 编译器。下面命令在本目录执行，`sm_80` / `80` 请改为目标 GPU 支持的架构：

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_80 gemm_advanced.cu -o gemm_advanced
g++ -std=c++17 -O2 -Wall -Wextra -pedantic schedule_test.cpp -o schedule_test
```

或者使用 CMake：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/gemm_advanced --check-suite
```

CMake 默认只将 CPU 测试注册到 CTest。有可用 GPU 时可增加 `-DGEMM_ENABLE_GPU_TESTS=ON`，将 GPU 校验也注册进去。没有 CUDA 编译器时，CMake 会明确提示并只生成 `schedule_test`。

## 一次运行全部方向

```bash
./gemm_advanced
./gemm_advanced --m 129 --n 193 --k 65 --iters 20 --demo all
./gemm_advanced --check-suite
```

`--check-suite` 强制运行所有 demo、每项计时迭代数为 1；覆盖 `(1,17,3)`、`(31,65,17)`、`(64,64,32)`、`(129,193,65)`、`(32,48,513)`，并运行固定的批处理、异构分组和 QKV 合并案例。它用于正确性，不用于比较性能。

每个版本先与 double 累加的 CPU 参考比较，再检查 D 的行尾 padding 未被覆盖，成功才打印 `PASS`。任何 CUDA API/执行错误或数值错误都会令进程非零退出。缓存策略不被设备暴露时，`cache` demo 打印 `SKIP`；其他 demo 仍然执行。

CPU 参考的复杂度也是 O(MNK)，大矩阵上可能很慢。先用小规模验证实现，再为正式测量准备适合的参考/基准工具。

## 分方向实验

| 选项 | 输出标签/对照 | 关注点 |
|------|---------------|--------|
| `--demo core` | masked、interior-fast、三个 shape、swizzle、persistent、beta-zero | 相同数据的线程分块、边界与调度变化 |
| `--demo split` | split-k-two-pass、split-k-fused-reduce、sliced-k-block-reduce | 跨 Block 归约 vs Block 内归约，epilogue 执行一次 |
| `--demo streamk` | stream-k-two-pass | 按总 BK 迭代数平衡工作，段结果再归约 |
| `--demo packed` | unpacked-B、packed-B-steady、pack-plus-gemm | 打包成本、稳态收益及回本次数 |
| `--demo fusion` | 3 次 launch vs fused epilogue | GEMM→bias→ReLU 的完整链路 |
| `--demo cache` | L2-window-control vs L2-persisting-window | 给 B 的一段区域设置 L2 持久化偏好 |
| `--demo batch` | 顺序、strided batch、多 stream、grouped、K 排序、QKV 合并 | 多问题调度与共同输入重组 |
| `--demo tune` | 三个候选及缓存命中 | 数值验证、三轮计时中位数、进程内配置复用 |
| `--demo graph` | 相同 3 节点链的普通提交 vs graph replay | GPU 时间、同步后的主机墙钟时间和构图成本 |

示例命令：

```bash
# 边缘与长宽形状
./gemm_advanced --demo core --m 129 --n 257 --k 65 --group-rows 4

# 少输出、长 K；workers 是 GPU 工作 Block 数，不是每 Block 线程数
./gemm_advanced --demo split --m 64 --n 64 --k 4097 --split 16
./gemm_advanced --demo streamk --m 64 --n 64 --k 4097 --workers 32

# 固定权重的预打包，报告预处理与稳态两种时间
./gemm_advanced --demo packed --m 128 --n 257 --k 257 --iters 30

# 短调用链
./gemm_advanced --demo fusion --m 64 --n 64 --k 32 --iters 100
./gemm_advanced --demo graph --m 64 --n 64 --k 32 --iters 100
./gemm_advanced --demo batch --iters 100

# 极端任务划分仍需正确
./gemm_advanced --check-suite --workers 1 --split 64 --group-rows 64
./gemm_advanced --check-suite --workers 100 --split 3 --group-rows 3
```

`--workers 0`（默认）根据 Base persistent kernel 的 occupancy 上限估计工作 Block 数。它是教学启发式，不是 Stream-K / Grouped kernel 各自的最优占用率；可显式指定来比较。`--split` 会被限制到实际 BK 迭代数，避免空分段。`--group-rows` 范围是 1~64；1 表示普通逐行 tile 顺序。

`batch` 使用固定形状，独立于 `--m/--n/--k`：8 个 `(33,65,49)` GEMM、5 个异构问题，以及 17×49 输入到三个 35 列投影的合并案例。

## 计时口径

- `time_gpu` 在指定 stream 上预热、同步，再用 CUDA events 测量完整命名操作。非常短的操作仍可能包含 GPU 等待主机提交的空隙。
- Split-K / Stream-K 时间包括部分和 kernel **和归约 kernel**。workspace 分配、Stream-K host 计划及上传在计时外。
- `packed-B-steady` 不含打包；`pack-plus-gemm` 包含每次重新打包。驱动单独报告 pack 时间，并只在实测稳态变快时计算回本点。
- `cache` 两边都会预热，测的是重复使用 B 的场景。小 B 本来就可能命中 L2，提示策略未必带来收益。设置影响当前 CUDA 上下文；示例结束后清除窗口、重置持久化缓存并恢复原预算。MIG/MPS 等环境的支持和配置权限以 Runtime 返回为准。
- `graph` 保留 GEMM、bias、ReLU 三个节点，不把 Graph 的效果与融合混为一谈。资源创建、capture 和 instantiate 不进入 replay 的稳态时间，但单独报告构图墙钟时间。
- `8-streams-with-join` 用一个起始事件将所有工作流接到计时流，再让计时流等待全部完成事件。计时结束意味着整批完成，包含扇出/汇合的事件开销。
- Batched/Grouped 输出的是每批时间；标量 GEMM 的 GFLOP/s 使用有效 `2MNK`，不把 padding 或归约的额外运算算成有用 FLOP。
- `Fixture::reset()` 中的设备同步用于测试准备，不在稳态计时内。正式应用不应在每次算子调用前照搬这次测试同步。

## 代码边界与实现选择

1. **公共核心有意保持简单**：256 线程、BK=16、fp32 FMA 和 shared 分块；没有再叠加 Tensor Core、异步流水线或架构专用汇编。与生产库的绝对吞吐不是本示例的目标。
2. **Stream-K 是两阶段教学实现**：按工作量均分和跨 tile 切段是真实实现；每个 segment 都写 workspace，再由第二个 kernel 归约。它没有 CUTLASS 的整 tile 直写、混合调度、细粒度完成协议，不能据它推断优化版 Stream-K 的性能。
3. **Sliced-K 使用独立的 16×32 输出 tile**：4 个 64 线程子组各算每轮 32 个 k 中的 8 个，再在 shared 归约。它用于说明 Block 内 K 分片，与 Base 64×64 的计时不构成只改变一个变量的消融实验。
4. **B 打包布局属于特定 tile 配置**：`[tile_n][tile_k][BK][BN]` 必须与消费端 BN/BK 一致；换算法、改权重或换尺寸要重新打包。当前 padding 并非 `float4` 对齐示例，程序使用标量协作加载。
5. **调参缓存是有限示例**：仅限当前进程、fp32、固定普通 epilogue、相同 alpha/beta 语义和标量指针对齐条件。扩展到转置、融合、量化或其他 workspace 策略时，要扩展缓存键与合法性检查。
6. **所有数据依赖都显式处理**：没有跨 Block 自旋等待，不依赖 Block 按编号执行；尾部线程参与 shared 栅栏，只有 global 访问和写回按有效坐标屏蔽。

## 检查与分析

```bash
./schedule_test
./gemm_advanced --check-suite
compute-sanitizer --tool memcheck ./gemm_advanced --check-suite
compute-sanitizer --tool racecheck ./gemm_advanced --demo core --m 65 --n 97 --k 33 --iters 1
compute-sanitizer --tool synccheck ./gemm_advanced --demo split --m 17 --n 33 --k 19 --iters 1
ncu --set basic ./gemm_advanced --demo core --m 256 --n 256 --k 256 --iters 5
nsys profile -o gemm_graph ./gemm_advanced --demo graph --iters 100
```

`schedule_test` 验证 CPU/GPU 共用索引的覆盖性、Stream-K 的负载差不超过一个 BK 单位、归约列表、Grouped 任务归属，以及非整齐矩阵从打包数据与分段任务重建出的结果。它不执行 CUDA，不能验证 GPU 上的 barrier、访存或性能。
