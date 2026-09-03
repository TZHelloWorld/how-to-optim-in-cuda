"""
demo_04_multiprocessing.py
==========================
用多进程绕过 GIL，真正利用多核并行计算。

运行:
    python demo_04_multiprocessing.py

本示例展示:
    1. multiprocessing.Pool 的基本用法
    2. 每个子进程有独立的 PID 和独立的 GIL
    3. 进程间通过序列化(pickle)传递数据，无法直接共享内存对象
"""

import os
import time
import multiprocessing as mp


def heavy_compute(n: int) -> tuple:
    """CPU 密集型工作，返回 (进程PID, 结果)。"""
    total = 0
    for i in range(n):
        total += (i * i) % 7
    return os.getpid(), total


N = 15_000_000
TASKS = 4


def main():
    print("=" * 60)
    print("使用 multiprocessing 绕过 GIL")
    print(f"主进程 PID: {os.getpid()}, CPU 核心数: {mp.cpu_count()}")
    print("=" * 60)

    start = time.perf_counter()
    with mp.Pool(processes=TASKS) as pool:
        results = pool.map(heavy_compute, [N] * TASKS)
    elapsed = time.perf_counter() - start

    print("各任务由不同的子进程完成 (注意 PID 各不相同):")
    pids = set()
    for idx, (pid, res) in enumerate(results):
        print(f"  任务 {idx}: 子进程 PID={pid}, 部分结果={res}")
        pids.add(pid)

    print("-" * 60)
    print(f"用到的独立子进程数: {len(pids)}")
    print(f"总耗时: {elapsed:.3f} 秒")
    print("结论:")
    print("  每个子进程拥有独立的 Python 解释器与独立的 GIL，")
    print("  因此可以在多个 CPU 核心上真正并行计算。")
    print("  代价: 进程创建开销较大，数据需序列化传递，内存不共享。")
    print("=" * 60)


if __name__ == "__main__":
    # Windows / macOS(spawn) 上必须有 __main__ 保护，否则会递归创建子进程
    main()
