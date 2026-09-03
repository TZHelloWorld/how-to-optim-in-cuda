"""
demo_05_race_condition.py
=========================
演示：GIL 并不能保证复合操作的线程安全（竞态条件）。

运行:
    python demo_05_race_condition.py

常见误区:
    很多人以为“有了 GIL，同一时刻只有一个线程执行，所以 Python 多线程一定线程安全”。
    这是错误的！GIL 只保证单条字节码的执行不被打断，
    但像 `counter += 1` 这样的操作实际上由多条字节码组成:
        LOAD  -> ADD -> STORE
    在这几步之间，GIL 可能被切换给别的线程，从而产生竞态条件、丢失更新。
"""

import sys
import threading
import dis

# 把 GIL 切换间隔调到极小，逼迫解释器在“读-改-写”之间频繁切换线程，
# 从而稳定地复现竞态条件（否则默认 5ms 的间隔下循环太快，往往观察不到）。
sys.setswitchinterval(1e-9)


# 一个被多个线程共享的全局计数器
counter = 0

N_THREADS = 8
INCREMENTS = 100_000


def unsafe_increment():
    """没有加锁的“读-改-写”操作 —— 存在竞态条件。

    这里刻意用一个局部中间变量来放大“读取 -> 计算 -> 写回”之间的时间窗口，
    使得 GIL 在这几步之间被切换给别的线程的概率增大，从而更容易复现丢失更新。
    在真实代码里 `counter += 1` 同样不是原子的，只是现代解释器下窗口更短。
    """
    global counter
    for _ in range(INCREMENTS):
        tmp = counter      # 读
        tmp = tmp + 1      # 改
        # 让出执行权，制造一个明显的切换窗口，使竞态更容易复现
        # （真实代码里 counter += 1 同样非原子，只是窗口更短、更“看运气”）
        for _ in range(3):
            pass
        counter = tmp      # 写


lock = threading.Lock()


def safe_increment():
    """使用锁保护复合操作 —— 线程安全。"""
    global counter
    for _ in range(INCREMENTS):
        with lock:
            counter += 1


def run(worker):
    global counter
    counter = 0
    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return counter


def main():
    expected = N_THREADS * INCREMENTS

    print("=" * 60)
    print("演示 GIL 不保证复合操作的线程安全")
    print("=" * 60)
    print("`counter += 1` 对应的字节码 (由多条指令组成，可能被 GIL 切换打断):")
    dis.dis("counter += 1")
    print()

    # 无锁版本 —— 结果通常会小于期望值（发生了丢失更新）
    result_unsafe = run(unsafe_increment)
    print(f"[无锁 读-改-写   ] 期望={expected:,}, 实际={result_unsafe:,}, "
          f"{'✅ 正确(碰巧)' if result_unsafe == expected else '❌ 出现丢失更新!'}")

    # 有锁版本 —— 结果一定正确
    result_safe = run(safe_increment)
    print(f"[加锁 Lock 保护  ] 期望={expected:,}, 实际={result_safe:,}, "
          f"{'✅ 正确' if result_safe == expected else '❌ 错误'}")

    print("-" * 60)
    print("结论:")
    print("  GIL 只保证单条字节码原子执行，不保证 `counter += 1` 这种复合操作原子。")
    print("  需要线程安全时，仍必须使用 Lock / RLock / Queue 等同步原语。")
    print("=" * 60)


if __name__ == "__main__":
    main()
