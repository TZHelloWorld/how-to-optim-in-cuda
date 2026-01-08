"""
===========================================================================
CUDA Stream 与同步机制 —— PyTorch Profiler 实战教程
===========================================================================

本教程通过 10 个渐进式用例，帮助你理解和优化 CUDA Stream 同步行为。

核心知识点：
  1. 什么是 CUDA Stream 以及同一 Stream 内有序、不同 Stream 间并行的规则
  2. cudaStreamSynchronize 何时被隐式调用
  3. Pageable 内存 vs Pinned 内存对同步的影响
  4. 跨 Stream 依赖（wait_stream）如何放大同步开销
  5. NULL Stream 的特殊隐式同步语义
  6. CUDA Event 的细粒度同步
  7. Overlap Scheduling（重叠调度）模式
  8. CUDA Graph 消除 CPU 提交开销

使用方式：
    python test_cuda_stream_sync_tutorial.py            # 运行全部 case
    python test_cuda_stream_sync_tutorial.py --case 1   # 运行单个 case（生成trace）
    python test_cuda_stream_sync_tutorial.py --list      # 列出所有 case

每个 case 生成的 trace.json 可以用以下工具查看：
    chrome://tracing  或  https://ui.perfetto.dev
"""

import argparse
import os
import time
from contextlib import contextmanager

import torch
import torch.nn as nn

# ===========================================================================
# 环境检查与全局配置
# ===========================================================================
assert torch.cuda.is_available(), "本教程需要 CUDA GPU 环境"

DEVICE = torch.device("cuda:0")
OUTPUT_DIR = "/tmp/cuda_stream_tutorial_traces"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# 工具函数
# ===========================================================================
@contextmanager
def profile_and_export(trace_name: str):
    """对代码块进行 profiling 并导出 chrome trace。"""
    trace_path = os.path.join(OUTPUT_DIR, trace_name)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
        record_shapes=True,
    ) as prof:
        yield prof

    json_path = f"{trace_path}.json"
    prof.export_chrome_trace(json_path)
    print(f"  [Trace] {json_path}")
    print(f"  [View]  chrome://tracing 或 https://ui.perfetto.dev")
    print()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def print_case_header(case_num: int, title: str, description: str):
    """打印用例标题。"""
    print()
    print("=" * 72)
    print(f"  Case {case_num}: {title}")
    print("=" * 72)
    print()
    print(f"  {description}")
    print()


def print_explanation(text: str):
    """打印解释说明。"""
    print()
    print("-" * 72)
    print("  解析:")
    for line in text.strip().split("\n"):
        print(f"  {line}")
    print("-" * 72)
    print()


# ===========================================================================
# Case 1: Pageable 内存的隐式 cudaStreamSynchronize
# ===========================================================================
def case1_pageable_implicit_sync():
    """
    知识点: 普通CPU内存(Pageable) → GPU 的拷贝会触发隐式 cudaStreamSynchronize。

    当你用 tensor.to(device) 且 non_blocking=False（默认）时，PyTorch 内部执行：
      1. cudaMemcpyAsync(dst, src, size, HtoD, stream)  — 提交异步拷贝
      2. cudaStreamSynchronize(stream)                   — 等待该 stream 完成

    原因：Pageable 内存可能被 OS 换出到磁盘，GPU DMA 引擎无法直接访问。
    CUDA 驱动必须先将数据复制到内部的 pinned staging buffer，这要求
    目标 stream 上没有其他操作在执行。

    关键观察：如果目标 stream 此时没有待完成的工作，sync 几乎立即返回
    (~5-10us)——看起来像"异步的"，但其实 CPU 确实被短暂阻塞了。
    """
    print_case_header(
        1,
        "Pageable 内存隐式 Sync",
        "演示：普通 CPU tensor → GPU 拷贝时，为什么会自动触发 cudaStreamSynchronize",
    )

    with profile_and_export("case1_pageable_sync") as prof:
        # 热身，确保 CUDA context 初始化完毕
        _ = torch.zeros(100, device=DEVICE)
        torch.cuda.synchronize()

        # --- 场景 A: Stream 空闲时做 Pageable 拷贝 ---
        # sync 很快（~5-10us），因为 stream 上没有排队的工作
        cpu_tensor_a = torch.randn(1024)  # Pageable 内存
        gpu_a = cpu_tensor_a.to(DEVICE)  # 触发: cudaMemcpyAsync + cudaStreamSynchronize
        # ↑ 此时 stream 空闲，sync 立即返回

        torch.cuda.synchronize()  # 分隔两个场景

        # --- 场景 B: Stream 有大量排队工作时做 Pageable 拷贝 ---
        # 先往 stream 上堆积大量 GPU 计算
        big_matrix = torch.randn(2048, 2048, device=DEVICE)
        for _ in range(20):
            big_matrix = big_matrix @ big_matrix  # 20 次矩阵乘法入队

        # 现在做 pageable 拷贝 — sync 必须等上面所有 matmul 完成！
        cpu_tensor_b = torch.randn(1024)
        gpu_b = cpu_tensor_b.to(DEVICE)  # 这次 sync 会很慢！
        # ↑ cudaStreamSynchronize 等待 20 次 matmul 全部执行完毕

    print_explanation("""
    场景A: sync 约 5-10us（stream 空闲，没有要等的工作）
    场景B: sync 可能数十ms（必须等 20 次 matmul 做完）

    这解释了为什么 trace 中有些 sync 极短（看起来像异步），
    有些 sync 极长（阻塞整个流水线）。

    根本原因：Pageable 内存 → GPU 拷贝 = cudaMemcpyAsync + cudaStreamSynchronize
    """)


# ===========================================================================
# Case 2: Pinned 内存消除隐式 Sync
# ===========================================================================
def case2_pinned_memory_no_sync():
    """
    知识点: 使用 Pinned (page-locked) 内存可以实现真正的异步拷贝。

    Pinned 内存被锁定在物理内存中，GPU DMA 引擎可以直接访问，
    不需要通过 staging buffer，因此不需要 stream 同步。

    关键操作：tensor.pin_memory() + .to(device, non_blocking=True)
    """
    print_case_header(
        2,
        "Pinned 内存消除 Sync",
        "演示：使用 pin_memory + non_blocking=True 实现真正的异步拷贝",
    )

    with profile_and_export("case2_pinned_no_sync") as prof:
        torch.cuda.synchronize()

        # --- 对比 A: Pageable 拷贝（有 sync） ---
        cpu_pageable = torch.randn(1024 * 1024)  # 4MB pageable
        gpu_a = cpu_pageable.to(DEVICE)  # 有隐式 sync

        torch.cuda.synchronize()

        # --- 对比 B: Pinned 拷贝（无 sync） ---
        cpu_pinned = torch.randn(1024 * 1024).pin_memory()  # 4MB pinned
        gpu_b = cpu_pinned.to(DEVICE, non_blocking=True)  # 真正异步！
        # ↑ 只有 cudaMemcpyAsync，没有 cudaStreamSynchronize

        # CPU 立即继续 — 拷贝在后台进行
        # 你可以在这里做 CPU 计算，与 GPU 拷贝重叠
        cpu_work = sum(range(10000))

        torch.cuda.synchronize()  # 最终确保拷贝完成

    print_explanation("""
    Pageable: aten::copy_ = cudaMemcpyAsync + cudaStreamSynchronize  (阻塞)
    Pinned:   aten::copy_ = cudaMemcpyAsync                          (真正异步)

    优化模式：
      # 差: Pageable（触发 sync）
      tensor = torch.randn(N)
      gpu_tensor = tensor.to(device)

      # 好: Pinned（真正异步）
      tensor = torch.randn(N).pin_memory()
      gpu_tensor = tensor.to(device, non_blocking=True)
    """)


# ===========================================================================
# Case 3: non_blocking=True 的行为差异
# ===========================================================================
def case3_non_blocking_behavior():
    """
    知识点: non_blocking=True 的效果取决于内存类型。

    - non_blocking=True + Pageable 内存: PyTorch 不显式调用 sync，
      但 CUDA 驱动内部可能仍需同步（行为依赖驱动实现）
    - non_blocking=True + Pinned 内存: 真正异步，无任何阻塞
    - non_blocking=False (默认): 始终在 cudaMemcpyAsync 后加 sync
    """
    print_case_header(
        3,
        "non_blocking 标志行为",
        "演示：non_blocking=True 与 False 在不同内存类型下的真实差异",
    )

    with profile_and_export("case3_non_blocking") as prof:
        torch.cuda.synchronize()

        # 先入队一些 GPU 计算
        x = torch.randn(1024, 1024, device=DEVICE)
        for _ in range(5):
            x = x @ x

        # --- non_blocking=False + Pageable: 最慢（显式 sync，等所有 matmul） ---
        pageable = torch.randn(1024)
        t0 = time.perf_counter()
        gpu1 = pageable.to(DEVICE, non_blocking=False)
        t1 = time.perf_counter()
        print(f"  non_blocking=False + Pageable: {(t1-t0)*1000:.2f}ms")

        torch.cuda.synchronize()

        # 再入队一些 GPU 计算
        for _ in range(5):
            x = x @ x

        # --- non_blocking=True + Pageable: 中等（不显式 sync，但驱动可能内部阻塞）---
        t0 = time.perf_counter()
        gpu2 = pageable.to(DEVICE, non_blocking=True)
        t1 = time.perf_counter()
        print(f"  non_blocking=True  + Pageable: {(t1-t0)*1000:.2f}ms")

        torch.cuda.synchronize()

        # 再入队一些 GPU 计算
        for _ in range(5):
            x = x @ x

        # --- non_blocking=True + Pinned: 最快（真正异步） ---
        pinned = torch.randn(1024).pin_memory()
        t0 = time.perf_counter()
        gpu3 = pinned.to(DEVICE, non_blocking=True)
        t1 = time.perf_counter()
        print(f"  non_blocking=True  + Pinned:   {(t1-t0)*1000:.2f}ms")

        torch.cuda.synchronize()

    print_explanation("""
    non_blocking=False + Pageable: 慢（sync 等待所有排队的 GPU 工作）
    non_blocking=True  + Pageable: 较快（无显式 sync，但驱动可能阻塞）
    non_blocking=True  + Pinned:   最快（真正异步，CPU 立即返回）

    规则: 始终使用 non_blocking=True + pin_memory() 组合获得最佳性能。
    """)


# ===========================================================================
# Case 4: 跨 Stream 依赖 (wait_stream 放大 sync 耗时)
# ===========================================================================
def case4_cross_stream_dependency():
    """
    知识点: stream.wait_stream() 建立跨 Stream 依赖，会使后续 sync 被放大。

    这是 LLM 推理框架中最常见的性能陷阱之一：
    1. compute_stream 上排了大量推理 kernel
    2. copy_stream.wait_stream(compute_stream) — 建立依赖
    3. 在 copy_stream 上做 pageable 拷贝触发 sync
    4. 这个 sync 不得不等 compute_stream 全部完成！

    这正是 SGLang trace 中 29ms sync 的根因。
    """
    print_case_header(
        4,
        "跨 Stream 依赖",
        "演示：wait_stream 如何使一个 stream 上的 sync 被迫等待另一个 stream",
    )

    compute_stream = torch.cuda.Stream()
    copy_stream = torch.cuda.Stream()

    with profile_and_export("case4_cross_stream") as prof:
        torch.cuda.synchronize()

        # 步骤 1: 在 compute_stream 上排大量计算
        with torch.cuda.stream(compute_stream):
            x = torch.randn(2048, 2048, device=DEVICE)
            for _ in range(30):
                x = x @ x  # 30 次重型矩阵乘法

        # 步骤 2: 建立跨 stream 依赖
        copy_stream.wait_stream(compute_stream)
        # ↑ 从此刻起，copy_stream 上所有后续操作必须等 compute_stream 完成

        # 步骤 3: 在 copy_stream 上做 pageable 拷贝 — 触发 sync
        with torch.cuda.stream(copy_stream):
            cpu_data = torch.randn(1024)
            gpu_data = cpu_data.to(DEVICE)  # sync 会等整个 compute_stream！
            # ↑ 这个 sync 的耗时 ≈ compute_stream 30 次 matmul 的总时间

        torch.cuda.synchronize()

    print_explanation("""
    因果链：
    1. compute_stream: 30 次 matmul 排队（~Xms GPU 工作）
    2. copy_stream.wait_stream(compute_stream): 建立依赖
    3. cpu_data.to(DEVICE) 在 copy_stream 上:
       → cudaMemcpyAsync(copy_stream)
       → cudaStreamSynchronize(copy_stream)
       → 但 copy_stream 必须等 compute_stream 完成才能执行！
       → 所以 sync 阻塞了 compute_stream 的全部执行时间

    优化方案:
    - 使用 pinned memory + non_blocking=True 避免触发 sync
    - 用 Event 建立最小粒度依赖（而非 wait_stream 等全部）
    """)


# ===========================================================================
# Case 5: .item() / .cpu() 同步陷阱
# ===========================================================================
def case5_item_cpu_trap():
    """
    知识点: .item()、.cpu()、.numpy() 都会触发 cudaStreamSynchronize。

    这些操作需要把数据从 GPU 传回 CPU，且 CPU 需要立即使用结果，
    所以必须等 GPU 完成所有在该 stream 上排队的工作。

    在循环中频繁调用这些函数 = 每次迭代都 sync = 完全串行化。
    """
    print_case_header(
        5,
        ".item() / .cpu() 同步陷阱",
        "演示：为什么在循环中调用 .item() 会严重拖慢性能",
    )

    with profile_and_export("case5_item_trap") as prof:
        torch.cuda.synchronize()

        # --- 差: 每次迭代都 .item()，每次都 sync ---
        x = torch.randn(2048, 2048, device=DEVICE)
        results_bad = []
        t0 = time.perf_counter()
        for i in range(5):
            x = x @ x
            results_bad.append(x[0, 0].item())  # 每次强制 sync！
        t_bad = time.perf_counter() - t0
        print(f"  差的写法 (.item() 每次): {t_bad*1000:.2f}ms")

        torch.cuda.synchronize()

        # --- 好: 收集 GPU tensor，最后一次性取 ---
        x = torch.randn(2048, 2048, device=DEVICE)
        scalars = []
        t0 = time.perf_counter()
        for i in range(5):
            x = x @ x
            scalars.append(x[0, 0].clone())  # 保持在 GPU，不触发 sync
        torch.cuda.synchronize()  # 只 sync 一次
        results_good = [s.item() for s in scalars]  # 此时数据已就绪
        t_good = time.perf_counter() - t0
        print(f"  好的写法 (最后 sync): {t_good*1000:.2f}ms")
        print(f"  加速比: {t_bad/t_good:.1f}x")

    print_explanation("""
    .item()/.cpu()/.numpy() 内部都会触发 cudaStreamSynchronize:
    - CPU 需要立即获取数值 → 必须等 GPU 算完
    - 如果 stream 上有排队的工作 → 每次都要等

    优化模式:
    1. 循环中保持数据在 GPU，最后统一取回
    2. 使用 pinned buffer + copy_(non_blocking=True) 做异步 DtoH
    3. 用 Event 检查完成状态，避免不必要的等待
    """)


# ===========================================================================
# Case 6: torch.tensor(device='cuda') 的隐藏 Sync
# ===========================================================================
def case6_torch_tensor_hidden_sync():
    """
    知识点: torch.tensor(data, device='cuda') 内部会触发隐式 sync。

    执行过程：
    1. 在 CPU 上创建 Pageable tensor
    2. 执行 aten::copy_ (HtoD, Pageable → Device)
    3. cudaMemcpyAsync + cudaStreamSynchronize

    这在 SGLang 的代码中造成了 29ms 的阻塞：
      torch.tensor([t.data_ptr() for t in prefix_tensors],
                   device=req_to_token_pool.device, dtype=torch.uint64)
    """
    print_case_header(
        6,
        "torch.tensor(device='cuda') 隐藏 Sync",
        "演示：直接在 GPU 上创建 tensor 时，隐藏的同步开销",
    )

    with profile_and_export("case6_torch_tensor") as prof:
        torch.cuda.synchronize()

        # 先入队大量 GPU 计算
        x = torch.randn(2048, 2048, device=DEVICE)
        for _ in range(20):
            x = x @ x

        # --- 差: 直接 torch.tensor(device='cuda') ---
        data = list(range(100))
        t0 = time.perf_counter()
        gpu_bad = torch.tensor(data, device=DEVICE, dtype=torch.int64)
        t1 = time.perf_counter()
        # ↑ 内部: CPU tensor → aten::copy_ → Pageable HtoD + sync
        # sync 必须等上面 20 次 matmul 全部完成！
        print(f"  差: torch.tensor(device='cuda') = {(t1-t0)*1000:.2f}ms")

        torch.cuda.synchronize()

        # 再入队大量 GPU 计算
        for _ in range(20):
            x = x @ x

        # --- 好: CPU 创建 + pin + 异步拷贝 ---
        t0 = time.perf_counter()
        cpu_tensor = torch.tensor(data, dtype=torch.int64).pin_memory()
        gpu_good = cpu_tensor.to(DEVICE, non_blocking=True)
        t1 = time.perf_counter()
        # ↑ 无 sync！拷贝异步进行，GPU 的 matmul 继续执行不受干扰
        print(f"  好: pin_memory + non_blocking = {(t1-t0)*1000:.2f}ms")

        torch.cuda.synchronize()

    print_explanation("""
    torch.tensor(data, device='cuda') 等价于：
      1. cpu_tmp = torch.tensor(data)          # Pageable CPU tensor
      2. gpu = cpu_tmp.to(device)              # aten::copy_ + sync!

    如果此时 stream 上有大量 GPU 工作排队 → sync 等待全部完成 → 长时间阻塞

    修复：
      # 差（阻塞 GPU 流水线）:
      t = torch.tensor(data, device='cuda')

      # 好（异步，不阻塞）:
      t = torch.tensor(data).pin_memory().to('cuda', non_blocking=True)

      # 最好（复用预分配 pinned buffer）:
      pinned_buf[:len(data)] = torch.tensor(data)
      gpu_buf.copy_(pinned_buf, non_blocking=True)
    """)


# ===========================================================================
# Case 7: NULL Stream 的特殊行为
# ===========================================================================
def case7_null_stream_behavior():
    """
    知识点: NULL Stream (Legacy Default Stream, stream 0) 有隐式同步语义。

    CUDA 中的 NULL Stream 会与所有 Blocking Stream 产生双向依赖：
    - NULL Stream 上的操作启动前：等待所有 Blocking Stream 完成
    - NULL Stream 上的操作提交后：所有 Blocking Stream 等待 NULL Stream 完成

    但在 PyTorch 中，所有 stream 都是 Non-Blocking，所以：
    - NULL Stream 的隐式同步不会影响 PyTorch 创建的 stream
    - 这也是为什么 trace 中 stream 0 上没有任何事件
    """
    print_case_header(
        7,
        "NULL Stream 行为",
        "演示：PyTorch 如何规避 NULL Stream 的隐式同步问题",
    )

    with profile_and_export("case7_null_stream") as prof:
        torch.cuda.synchronize()

        # --- 验证: PyTorch 的 default stream 不是 NULL stream ---
        default_stream = torch.cuda.default_stream()
        current_stream = torch.cuda.current_stream()
        explicit_stream = torch.cuda.Stream()

        print(f"  PyTorch default stream:  {default_stream}")
        print(f"  PyTorch current stream:  {current_stream}")
        print(f"  Explicit stream:         {explicit_stream}")
        print(f"  Default stream ptr:      {default_stream.cuda_stream}")
        print(f"  (NULL stream 的 ptr = 0)")
        print()

        # --- 证明: PyTorch 的多个 stream 确实可以并行 ---
        # 如果有任何一个是 NULL stream，它们就无法并行
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()

        # 在两个 stream 上同时排大量计算
        with torch.cuda.stream(stream_a):
            x = torch.randn(2048, 2048, device=DEVICE)
            for _ in range(10):
                x = x @ x

        with torch.cuda.stream(stream_b):
            y = torch.randn(2048, 2048, device=DEVICE)
            for _ in range(10):
                y = y @ y

        # 两个 stream 并行执行，因为都不是 NULL stream
        torch.cuda.synchronize()

        # --- 模拟: NULL Stream 的串行化效果 ---
        # 如果真的使用 NULL Stream，效果等同于所有 stream 完全串行
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()
        simulated_null = torch.cuda.Stream()

        with torch.cuda.stream(stream_a):
            x = torch.randn(1024, 1024, device=DEVICE)
            x = x @ x

        # 模拟 NULL stream: 等所有 stream 完成，再执行
        simulated_null.wait_stream(stream_a)
        simulated_null.wait_stream(stream_b)
        with torch.cuda.stream(simulated_null):
            z = torch.randn(1024, 1024, device=DEVICE)
            z = z @ z

        # 模拟: 所有 stream 都要等 NULL stream
        stream_a.wait_stream(simulated_null)
        stream_b.wait_stream(simulated_null)
        with torch.cuda.stream(stream_b):
            y = torch.randn(1024, 1024, device=DEVICE)
            y = y @ y

        torch.cuda.synchronize()

    print_explanation("""
    NULL Stream (stream 0) 的隐式同步规则：
      规则1: NULL stream 上的操作启动前 → 等所有 Blocking stream 完成
      规则2: NULL stream 上的操作提交后 → 所有 Blocking stream 等它完成
      效果: 一旦使用 NULL stream，所有 stream 变成完全串行

    PyTorch 的规避策略：
      - 所有 stream（包括 default）都用 cudaStreamNonBlocking 创建
      - Non-Blocking stream 不受 NULL stream 的隐式同步约束
      - 所以 PyTorch 中多 stream 可以真正并行
      - trace 中 stream 0 上事件数 = 0

    何时可能遇到 NULL stream 问题:
      - 调用原始 CUDA C API 时不指定 stream
      - 使用 cudaMemcpy（非 Async 版本）— 始终用 NULL stream
      - 第三方 C 扩展库未使用 cudaStreamNonBlocking
    """)


# ===========================================================================
# Case 8: CUDA Event 细粒度同步
# ===========================================================================
def case8_event_fine_grained_sync():
    """
    知识点: CUDA Event 允许比 stream sync 更精细的同步控制。

    三种 Event 用法：
    1. event.synchronize() — CPU 等待特定标记点（而非整个 stream）
    2. stream.wait_event(event) — Stream 间建立点对点依赖（CPU不阻塞）
    3. event.query() — 非阻塞查询某个标记是否完成

    关键优势: 相比 stream.synchronize() 等待全部工作，
    event.synchronize() 只等到特定操作完成即可。
    """
    print_case_header(
        8,
        "Event 细粒度同步",
        "演示：使用 CUDA Event 实现比 stream sync 更精确的等待",
    )

    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    with profile_and_export("case8_event_sync") as prof:
        torch.cuda.synchronize()

        # --- 模式 A: 粗粒度同步（差） ---
        with torch.cuda.stream(stream1):
            x = torch.randn(2048, 2048, device=DEVICE)
            for _ in range(10):
                x = x @ x
            needed_result = x.sum()  # 我们只需要这个结果
            for _ in range(10):
                x = x @ x  # 但这些额外计算也在 stream 上

        # 粗: 等 stream1 全部完成（包括不需要的后 10 次 matmul）
        stream1.synchronize()
        val_a = needed_result.item()

        # --- 模式 B: Event 细粒度同步（好） ---
        with torch.cuda.stream(stream1):
            x = torch.randn(2048, 2048, device=DEVICE)
            for _ in range(10):
                x = x @ x
            needed_result2 = x.sum()  # 我们需要的结果

            # 在需要的结果之后打 Event 标记
            event = torch.cuda.Event()
            event.record()

            # 后续不需要的计算继续在 stream 上
            for _ in range(10):
                x = x @ x

        # 细: 只等到 Event 处（前 10 次 matmul + sum），后面的继续跑
        event.synchronize()
        val_b = needed_result2.item()

        # --- 模式 C: Stream 间 Event 依赖（CPU不阻塞） ---
        with torch.cuda.stream(stream1):
            a = torch.randn(1024, 1024, device=DEVICE)
            a = a @ a
            event_a = torch.cuda.Event()
            event_a.record()  # 标记: 第一次 matmul 完成
            # stream1 继续更多计算...
            for _ in range(5):
                a = a @ a

        with torch.cuda.stream(stream2):
            # stream2 只等 event_a（第一次 matmul），不等 stream1 全部
            stream2.wait_event(event_a)
            b = a + 1  # 第一次 matmul 完成后就能开始

        torch.cuda.synchronize()

        # --- 模式 D: 非阻塞查询 ---
        with torch.cuda.stream(stream1):
            x = torch.randn(2048, 2048, device=DEVICE)
            x = x @ x
            check_event = torch.cuda.Event()
            check_event.record()

        # CPU 不阻塞，只是"看一眼"
        if check_event.query():
            print("  Event 查询: GPU 已完成")
        else:
            print("  Event 查询: GPU 还在执行中，CPU 可以继续做其他事")

        torch.cuda.synchronize()

    print_explanation("""
    同步粒度对比：
      torch.cuda.synchronize()  → 等所有 stream 的所有工作（最重）
      stream.synchronize()      → 等一个 stream 的所有工作
      event.synchronize()       → 只等到某个标记点（最轻）
      event.query()             → 不等，只查一下（非阻塞）
      stream.wait_event(event)  → GPU端等待，CPU完全不阻塞

    优化建议:
    - 用 event 建立最小必要的跨 stream 依赖
    - 而非 stream.wait_stream()（会等整个 stream）
    - 用 event.query() 轮询代替 event.synchronize() 阻塞
    """)


# ===========================================================================
# Case 9: Overlap Scheduling 重叠调度模式
# ===========================================================================
def case9_overlap_scheduling():
    """
    知识点: LLM 推理中的 Overlap Scheduling（重叠调度）模式。

    核心思想：当前 batch 在 forward_stream 上做推理时，
    CPU 同时在 schedule_stream 上为下一个 batch 做准备。

    问题模式（SGLang 29ms sync 的根因）：
      1. forward_stream 排大量推理 kernel
      2. schedule_stream.wait_stream(forward_stream)
      3. 在 schedule_stream 上用 torch.tensor(device='cuda')
      4. 触发 sync → 等待全部推理 kernel！

    正确模式：
      使用 pinned memory + non_blocking 避免在 schedule_stream 上触发 sync
    """
    print_case_header(
        9,
        "Overlap Scheduling 模式",
        "演示：LLM 推理中的重叠调度 — 问题模式与优化模式对比",
    )

    schedule_stream = torch.cuda.Stream()  # 调度流
    forward_stream = torch.cuda.Stream()  # 前向计算流

    # 模拟一个小模型
    model = nn.Sequential(
        nn.Linear(512, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 512),
    ).to(DEVICE)

    with profile_and_export("case9_overlap") as prof:
        torch.cuda.synchronize()

        # ===== 问题模式（SGLang trace 中的实际情况）=====
        print("  --- 问题模式 ---")

        # 步骤 1: schedule_stream 准备数据
        with torch.cuda.stream(schedule_stream):
            input_data = torch.randn(32, 512, device=DEVICE)

        # 步骤 2: forward_stream 做推理（大量 kernel）
        with torch.cuda.stream(forward_stream):
            forward_stream.wait_stream(schedule_stream)
            output = model(input_data)

        # 步骤 3: 准备下一个 batch — 这里出问题！
        with torch.cuda.stream(schedule_stream):
            schedule_stream.wait_stream(forward_stream)  # 等 forward 完成
            # 差: 用 torch.tensor(device='cuda') → 触发 sync → 等全部 forward！
            t0 = time.perf_counter()
            indices = torch.tensor(
                [1, 2, 3, 4, 5], device=DEVICE, dtype=torch.int64
            )
            t1 = time.perf_counter()
            print(f"  问题模式 torch.tensor(device=cuda): {(t1-t0)*1000:.3f}ms")

        torch.cuda.synchronize()

        # ===== 优化模式 =====
        print("  --- 优化模式 ---")

        # 步骤 1: 同样准备数据
        with torch.cuda.stream(schedule_stream):
            input_data = torch.randn(32, 512, device=DEVICE)

        # 步骤 2: 同样做推理
        with torch.cuda.stream(forward_stream):
            forward_stream.wait_stream(schedule_stream)
            output = model(input_data)

        # 步骤 3: 优化的下一 batch 准备
        with torch.cuda.stream(schedule_stream):
            schedule_stream.wait_stream(forward_stream)
            # 好: 用 pinned memory + non_blocking
            t0 = time.perf_counter()
            indices_cpu = torch.tensor(
                [1, 2, 3, 4, 5], dtype=torch.int64
            ).pin_memory()
            indices = indices_cpu.to(DEVICE, non_blocking=True)
            t1 = time.perf_counter()
            print(f"  优化模式 pin + non_blocking:        {(t1-t0)*1000:.3f}ms")

        torch.cuda.synchronize()

    print_explanation("""
    SGLang Overlap Scheduling 架构:
      schedule_stream: [准备batch N+1] ─────────── [处理batch N结果] ── [准备batch N+2]
                           │                              ↑                    │
                           │ wait_stream                  │ Event              │ wait_stream
                           ↓                              │                    ↓
      forward_stream:     [推理 batch N ──── Event.record]                  [推理 batch N+1]

    问题: 如果在 schedule_stream 上用 pageable 拷贝触发 sync，
          而此时有 wait_stream 依赖 → sync 等全部 forward kernel

    解决: 所有 CPU→GPU 拷贝用 pin_memory + non_blocking，彻底避免 sync
    """)


# ===========================================================================
# Case 10: CUDA Graph 消除 CPU 提交开销
# ===========================================================================
def case10_cuda_graph():
    """
    知识点: CUDA Graph 将一系列操作"录制"后"重放"，消除 CPU 端 kernel 提交开销。

    普通模式: 每个 kernel 都需要 CPU 调用 cudaLaunchKernel (~5-20us)
    Graph 模式: 一次 cudaGraphLaunch (~26us) 重放所有 kernel

    在 LLM decode 阶段，每次迭代执行相同的计算图（只有输入不同），
    非常适合用 CUDA Graph 加速。
    """
    print_case_header(
        10,
        "CUDA Graph 加速",
        "演示：CUDA Graph 如何消除 CPU 提交开销，提升 GPU 利用率",
    )

    # 模拟一个简单计算
    x_static = torch.randn(1024, 1024, device=DEVICE)
    y_static = torch.randn(1024, 1024, device=DEVICE)

    with profile_and_export("case10_cuda_graph") as prof:
        torch.cuda.synchronize()

        # --- 普通模式: 每次迭代逐个提交 kernel ---
        print("  --- 普通模式（逐个提交 kernel）---")
        t0 = time.perf_counter()
        for _ in range(50):
            z = x_static @ y_static
            z = z + x_static
            z = z.relu()
        torch.cuda.synchronize()
        t_normal = time.perf_counter() - t0
        print(f"  普通模式 50 次迭代: {t_normal*1000:.2f}ms")

        # --- CUDA Graph 模式: 录制一次，重放多次 ---
        print("  --- CUDA Graph 模式（录制 + 重放）---")

        # 步骤 1: 热身（确保 GPU 内存分配已稳定）
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            z = x_static @ y_static
            z = z + x_static
            z = z.relu()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # 步骤 2: 录制 Graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            z = x_static @ y_static
            z = z + x_static
            z = z.relu()

        # 步骤 3: 重放 Graph
        t0 = time.perf_counter()
        for _ in range(50):
            g.replay()  # 一次 API 调用重放全部操作
        torch.cuda.synchronize()
        t_graph = time.perf_counter() - t0
        print(f"  Graph 模式 50 次迭代: {t_graph*1000:.2f}ms")
        print(f"  加速比: {t_normal/t_graph:.2f}x")

    print_explanation("""
    CUDA Graph 三步流程:
      1. 录制(Capture): 把一系列操作记录成图（只做一次）
      2. 实例化(Instantiate): 预编译优化（只做一次）
      3. 重放(Replay/Launch): 之后每次只需一个 API 调用

    效果:
      普通模式: CPU [launch k1] [launch k2] [launch k3]... (多次 API 调用)
      Graph:    CPU [cudaGraphLaunch]                       (一次 API 调用)

    在 LLM decode 阶段:
      - CPU 每 ~500us 提交一个 Graph
      - GPU 每 ~5ms 执行一个 Graph
      - 比例 1:10 → GPU 流水线始终满载
      - GPU 利用率可达 98%+

    注意事项:
      - Graph 中的操作必须是确定性的（相同的 shape、相同的路径）
      - 不能在 Graph 中做条件分支或动态 shape
      - 适合 LLM decode（固定 batch size），不适合 prefill（变长输入）
    """)


# ===========================================================================
# 主程序
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="CUDA Stream 与同步机制 —— PyTorch Profiler 实战教程"
    )
    parser.add_argument(
        "--case",
        type=int,
        default=None,
        help="运行指定 case (1-10)。默认: 运行全部。",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用的 case。",
    )
    args = parser.parse_args()

    cases = {
        1: ("Pageable 内存隐式 Sync", case1_pageable_implicit_sync),
        2: ("Pinned 内存消除 Sync", case2_pinned_memory_no_sync),
        3: ("non_blocking 标志行为", case3_non_blocking_behavior),
        4: ("跨 Stream 依赖（wait_stream放大sync）", case4_cross_stream_dependency),
        5: (".item()/.cpu() 同步陷阱", case5_item_cpu_trap),
        6: ("torch.tensor(device='cuda') 隐藏 Sync", case6_torch_tensor_hidden_sync),
        7: ("NULL Stream 行为", case7_null_stream_behavior),
        8: ("Event 细粒度同步", case8_event_fine_grained_sync),
        9: ("Overlap Scheduling 模式", case9_overlap_scheduling),
        10: ("CUDA Graph 加速", case10_cuda_graph),
    }

    if args.list:
        print()
        print("可用的 Case 列表:")
        print("-" * 60)
        for i, (name, _) in cases.items():
            print(f"  Case {i:2d}: {name}")
        print("-" * 60)
        print(f"\n运行示例: python {os.path.basename(__file__)} --case 1")
        return

    print()
    print("=" * 72)
    print("  CUDA Stream 与同步机制 —— PyTorch Profiler 实战教程")
    print("=" * 72)
    print(f"  设备: {torch.cuda.get_device_name()}")
    print(f"  输出: {OUTPUT_DIR}")
    print()

    if args.case:
        if args.case not in cases:
            print(f"  错误: Case {args.case} 不存在。可用范围: 1-{len(cases)}")
            return
        name, func = cases[args.case]
        print(f"  运行 Case {args.case}: {name}")
        func()
    else:
        print("  运行全部 Case...\n")
        for i, (name, func) in cases.items():
            try:
                func()
            except Exception as e:
                print(f"\n  [ERROR] Case {i} ({name}): {e}")
                import traceback
                traceback.print_exc()

    print()
    print("=" * 72)
    print("  教程完成！")
    print(f"  Trace 文件保存在: {OUTPUT_DIR}")
    print("  用 chrome://tracing 或 https://ui.perfetto.dev 打开 .json 文件查看")
    print("=" * 72)


if __name__ == "__main__":
    main()
