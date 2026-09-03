"""
demo_02_cpu_bound.py
====================
CPU 密集型任务：对比 单线程 / 多线程 / 多进程 的耗时。

运行:
    python demo_02_cpu_bound.py

预期结论:
    对于 CPU 密集型任务（纯 Python 计算），由于 GIL 的存在:
      - 多线程 (threading) 并不能加速，甚至因为切换开销而变慢；
      - 多进程 (multiprocessing) 每个进程有独立 GIL，可以真正利用多核加速。
"""

import time
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def cpu_task(n: int) -> int:
    """一个纯 CPU 密集型任务：累加 0..n 的平方。"""
    total = 0
    for i in range(n):
        total += i * i
    return total


N = 20_000_000   # 每个任务的计算量
TASKS = 4        # 任务个数


def run_serial():
    """单线程串行执行所有任务。"""
    start = time.perf_counter()
    for _ in range(TASKS):
        cpu_task(N)
    return time.perf_counter() - start


def run_threaded():
    """使用多线程执行 —— 受 GIL 限制，无法真正并行。"""
    start = time.perf_counter()
    threads = [threading.Thread(target=cpu_task, args=(N,)) for _ in range(TASKS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - start


def run_multiprocess():
    """使用多进程执行 —— 每个进程独立 GIL，可真正并行。"""
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=TASKS) as executor:
        list(executor.map(cpu_task, [N] * TASKS))
    return time.perf_counter() - start


def main():
    print("=" * 60)
    print(f"CPU 密集型任务对比 (每个任务 N={N:,}, 共 {TASKS} 个任务)")
    print(f"CPU 核心数: {multiprocessing.cpu_count()}")
    print("=" * 60)

    t_serial = run_serial()
    print(f"[单线程串行]  耗时: {t_serial:.3f} 秒")

    t_thread = run_threaded()
    print(f"[多线程   ]  耗时: {t_thread:.3f} 秒  "
          f"(相对串行加速比: {t_serial / t_thread:.2f}x)")

    t_proc = run_multiprocess()
    print(f"[多进程   ]  耗时: {t_proc:.3f} 秒  "
          f"(相对串行加速比: {t_serial / t_proc:.2f}x)")

    print("-" * 60)
    print("结论:")
    print("  多线程几乎没有加速甚至更慢 —— 这就是 GIL 对 CPU 密集型任务的限制。")
    print("  多进程能显著加速 —— 因为每个进程有独立的 GIL 和解释器。")
    print("=" * 60)


if __name__ == "__main__":
    # multiprocessing 在某些平台需要 __main__ 保护
    main()
