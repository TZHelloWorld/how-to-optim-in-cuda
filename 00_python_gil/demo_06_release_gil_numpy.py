"""
demo_06_release_gil_numpy.py
============================
演示：NumPy 等库在底层 C 计算时会释放 GIL，从而让多线程也能并行加速。

运行:
    python demo_06_release_gil_numpy.py

依赖:
    pip install numpy

要点:
    虽然纯 Python 的 CPU 密集型多线程会被 GIL 拖累(见 demo_02)，
    但 NumPy 的向量化运算在进入底层 C/BLAS 代码时会 **主动释放 GIL**，
    因此多个线程可以同时在不同核心上跑 NumPy 计算，实现真正的并行。
"""

import time
import threading

try:
    import numpy as np
except ImportError:
    raise SystemExit("本示例需要 numpy，请先运行: pip install numpy")


SIZE = 4000     # 矩阵维度 SIZE x SIZE
TASKS = 4       # 任务/线程数


def numpy_task():
    """一次较大的矩阵乘法。NumPy 在底层会释放 GIL。"""
    a = np.random.rand(SIZE, SIZE)
    b = np.random.rand(SIZE, SIZE)
    c = a @ b   # 矩阵乘法 -> 底层 BLAS，释放 GIL
    return float(c[0, 0])


def run_serial():
    start = time.perf_counter()
    for _ in range(TASKS):
        numpy_task()
    return time.perf_counter() - start


def run_threaded():
    start = time.perf_counter()
    threads = [threading.Thread(target=numpy_task) for _ in range(TASKS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - start


def main():
    print("=" * 60)
    print(f"NumPy 释放 GIL 演示 (矩阵 {SIZE}x{SIZE}, 共 {TASKS} 个任务)")
    print("=" * 60)
    print("提示: NumPy/BLAS 本身可能已是多线程的，")
    print("      可用环境变量 OMP_NUM_THREADS=1 限制其内部线程以更清晰地观察效果。")
    print("-" * 60)

    t_serial = run_serial()
    print(f"[单线程串行]  耗时: {t_serial:.3f} 秒")

    t_thread = run_threaded()
    print(f"[多线程   ]  耗时: {t_thread:.3f} 秒  "
          f"(相对串行加速比: {t_serial / t_thread:.2f}x)")

    print("-" * 60)
    print("结论:")
    print("  与纯 Python 的 CPU 密集型任务不同(见 demo_02)，")
    print("  NumPy 多线程通常能加速，因为底层 C 计算释放了 GIL。")
    print("  这也是为什么很多科学计算库能在 Python 里高效利用多核。")
    print("=" * 60)


if __name__ == "__main__":
    main()
