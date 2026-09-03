"""
demo_03_io_bound.py
===================
I/O 密集型任务：展示多线程在 I/O 场景下的加速效果。

运行:
    python demo_03_io_bound.py

预期结论:
    对于 I/O 密集型任务，线程在阻塞等待（如 sleep / 网络 / 磁盘）时会
    主动释放 GIL，因此其他线程可以并发运行 —— 多线程在这里非常有效。
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor


def io_task(task_id: int, delay: float = 0.5):
    """模拟一次 I/O 阻塞（例如网络请求、读写磁盘）。

    time.sleep() 在等待期间会释放 GIL，因此多个线程可以“同时”等待。
    """
    time.sleep(delay)
    return task_id


TASKS = 8
DELAY = 0.5  # 每个 I/O 任务阻塞 0.5 秒


def run_serial():
    start = time.perf_counter()
    for i in range(TASKS):
        io_task(i, DELAY)
    return time.perf_counter() - start


def run_threaded():
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=TASKS) as executor:
        list(executor.map(lambda i: io_task(i, DELAY), range(TASKS)))
    return time.perf_counter() - start


def main():
    print("=" * 60)
    print(f"I/O 密集型任务对比 (共 {TASKS} 个任务, 每个阻塞 {DELAY} 秒)")
    print("=" * 60)

    t_serial = run_serial()
    print(f"[单线程串行]  耗时: {t_serial:.3f} 秒  "
          f"(理论 = {TASKS * DELAY:.1f} 秒)")

    t_thread = run_threaded()
    print(f"[多线程   ]  耗时: {t_thread:.3f} 秒  "
          f"(相对串行加速比: {t_serial / t_thread:.2f}x)")

    print("-" * 60)
    print("结论:")
    print("  多线程在 I/O 密集型任务上有显著加速。")
    print("  因为线程执行阻塞 I/O 时会释放 GIL，多个任务的等待时间被重叠了。")
    print("=" * 60)


if __name__ == "__main__":
    main()
