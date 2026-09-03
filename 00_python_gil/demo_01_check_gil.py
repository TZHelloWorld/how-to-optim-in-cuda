"""
demo_01_check_gil.py
====================
检查当前 Python 解释器的 GIL 相关信息。

运行:
    python demo_01_check_gil.py

本示例展示:
    1. 当前 Python 实现与版本
    2. GIL 的线程切换间隔 (switch interval)
    3. 在 Python 3.13+ 上检查 GIL 是否启用 (free-threading)
"""

import sys
import platform


def main():
    print("=" * 60)
    print("Python GIL 基础信息")
    print("=" * 60)

    # 1. Python 实现与版本
    impl = platform.python_implementation()
    print(f"Python 实现        : {impl}")
    print(f"Python 版本        : {platform.python_version()}")
    print(f"运行平台           : {platform.platform()}")
    print(f"CPU 核心数         : {__import__('os').cpu_count()}")

    # 只有 CPython 才有 GIL；Jython / IronPython 没有 GIL
    if impl == "CPython":
        print("\n[说明] 你正在使用 CPython，它带有 GIL（全局解释器锁）。")
    else:
        print(f"\n[说明] 你正在使用 {impl}，它可能没有 GIL。")

    # 2. GIL 线程切换间隔
    #    Python 3.2+ 使用基于时间的切换，默认约 5ms
    interval = sys.getswitchinterval()
    print(f"\nGIL 线程切换间隔    : {interval} 秒 ({interval * 1000:.1f} ms)")
    print("  -> 解释器大约每隔这么久就会强制当前线程释放 GIL，让其他线程有机会运行。")

    # 演示如何修改切换间隔（这里改完再改回来）
    old = sys.getswitchinterval()
    sys.setswitchinterval(0.001)
    print(f"  已临时把切换间隔改为: {sys.getswitchinterval()} 秒")
    sys.setswitchinterval(old)
    print(f"  已恢复切换间隔为    : {sys.getswitchinterval()} 秒")

    # 3. Python 3.13+ 的 free-threading 检查
    print()
    if hasattr(sys, "_is_gil_enabled"):
        enabled = sys._is_gil_enabled()
        print(f"sys._is_gil_enabled(): {enabled}")
        if enabled:
            print("  -> 当前构建启用了 GIL（普通构建）。")
        else:
            print("  -> 当前构建禁用了 GIL（free-threading 构建，PEP 703）！")
    else:
        print("当前 Python 版本 < 3.13，没有 sys._is_gil_enabled()。")
        print("  -> 这是一个带 GIL 的普通 CPython 构建。")

    print("=" * 60)


if __name__ == "__main__":
    main()
