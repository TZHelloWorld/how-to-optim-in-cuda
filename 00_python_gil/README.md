# Python GIL（全局解释器锁）详解

本目录通过一系列可运行的示例，深入介绍 Python 的 GIL（Global Interpreter Lock，全局解释器锁）的原理、影响以及应对策略。

## 目录

- [什么是 GIL](#什么是-gil)
- [为什么会有 GIL](#为什么会有-gil)
- [GIL 如何工作](#gil-如何工作)
- [GIL 的影响](#gil-的影响)
- [如何绕过 GIL](#如何绕过-gil)
- [Python 3.13 的无 GIL（free-threading）模式](#python-313-的无-gilfree-threading模式)
- [示例文件说明](#示例文件说明)
- [如何运行](#如何运行)

---

## 什么是 GIL

GIL（Global Interpreter Lock，全局解释器锁）是 **CPython** 解释器中的一把互斥锁。它保证在任意时刻，**只有一个线程**能够执行 Python 字节码。

需要强调的是：

- GIL 是 **CPython 实现的产物**，而不是 Python 语言规范的一部分。
- 其它实现如 Jython（基于 JVM）、IronPython（基于 .NET）没有 GIL。
- 这意味着：即使你的机器有 16 个核心，一个纯 Python 的多线程程序在做 CPU 密集型计算时，也只能用满 1 个核心。

```
        多核 CPU
   ┌─────┬─────┬─────┬─────┐
   │ CPU0│ CPU1│ CPU2│ CPU3│
   └─────┴─────┴─────┴─────┘
      ▲
      │  同一时刻只有持有 GIL 的线程能执行字节码
   ┌──┴──────────────────────┐
   │  GIL（全局解释器锁）      │
   └─────────────────────────┘
   Thread-1  Thread-2  Thread-3   （轮流抢 GIL）
```

---

## 为什么会有 GIL

1. **简化内存管理（引用计数）**
   CPython 使用引用计数（reference counting）来管理内存。每个对象都有一个 `ob_refcnt` 计数器。如果多个线程同时修改同一个对象的引用计数，就会产生竞态条件（race condition），导致对象被过早释放或内存泄漏。GIL 保证了引用计数操作的原子性，避免了对每个对象都加锁的开销。

2. **简化 C 扩展的编写**
   有了 GIL，C 扩展开发者不必担心线程安全问题，很多历史遗留的 C 库因此能够安全地被集成。

3. **单线程性能好**
   基于单一全局锁的方案，在单线程场景下开销极小，避免了细粒度锁带来的大量加锁/解锁开销。

> 简言之：GIL 是一个「用单线程性能与实现简单性换取多线程并行能力」的历史权衡。

---

## GIL 如何工作

- 线程在执行 Python 字节码前必须先获取 GIL。
- CPython 会周期性地强制当前线程释放 GIL，让其他线程有机会运行。
  - 在 Python 2 中，这个切换基于「执行的字节码指令数」（默认每 100 条检查一次）。
  - 在 Python 3.2+ 中，改为基于**时间间隔**，默认约 **5 毫秒**（可通过 `sys.getswitchinterval()` / `sys.setswitchinterval()` 查看和修改）。
- 当线程执行**阻塞式 I/O**（读写文件、网络请求、`time.sleep` 等）时，会**主动释放 GIL**，因此其他线程可以趁机运行。这就是为什么多线程对 **I/O 密集型** 任务仍然有效。

```python
import sys
print(sys.getswitchinterval())  # 0.005 （秒）
```

---

## GIL 的影响

| 任务类型      | 多线程是否有效 | 原因                                             |
|--------------|--------------|--------------------------------------------------|
| CPU 密集型    | ❌ 基本无效   | 线程需要一直持有 GIL 执行字节码，无法真正并行     |
| I/O 密集型    | ✅ 有效       | 阻塞 I/O 时线程会释放 GIL，其他线程可并发运行     |

- **CPU 密集型**（如大量数学计算、循环）：多线程不但不能加速，反而因为线程切换和抢锁的开销可能**更慢**。
- **I/O 密集型**（如网络爬虫、读写磁盘）：多线程能有效地重叠等待时间，显著提升吞吐。

---

## 如何绕过 GIL

1. **`multiprocessing`（多进程）**
   每个进程有独立的 Python 解释器和独立的 GIL，可以真正利用多核。代价是进程间通信（IPC）和内存不能直接共享。

2. **使用释放 GIL 的 C 扩展 / 库**
   - `NumPy`、`SciPy`、`PyTorch` 等在执行底层计算时会**释放 GIL**，因此这些库内部的向量化运算能利用多核。
   - 自己写 C 扩展时，可用 `Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS` 宏在耗时的纯 C 计算段释放 GIL。

3. **`concurrent.futures.ProcessPoolExecutor`**
   对 `multiprocessing` 的高层封装，接口更简洁。

4. **异步编程 `asyncio`**
   单线程事件循环，适用于高并发 I/O，避免了线程切换开销（但不能加速 CPU 计算）。

5. **换用无 GIL 的解释器或语言互操作**
   如 Cython（`nogil`）、Numba、以及下面提到的 free-threading Python。

---

## Python 3.13 的无 GIL（free-threading）模式

从 **Python 3.13** 开始，CPython 引入了**实验性**的「free-threading」构建（PEP 703），可以在编译时禁用 GIL。

- 需要专门的构建版本（`python3.13t`）。
- 可通过 `sys._is_gil_enabled()`（3.13+）检查当前是否启用 GIL。
- 目前仍是实验特性，部分 C 扩展需要适配，单线程性能可能略有下降。

```python
import sys
if hasattr(sys, "_is_gil_enabled"):
    print("GIL enabled:", sys._is_gil_enabled())
```

---

## 示例文件说明

| 文件 | 说明 |
|------|------|
| `demo_01_check_gil.py`         | 检查当前解释器的 GIL 状态、切换间隔等基本信息 |
| `demo_02_cpu_bound.py`         | CPU 密集型任务：对比单线程 / 多线程 / 多进程的耗时，直观展示 GIL 对 CPU 任务的限制 |
| `demo_03_io_bound.py`          | I/O 密集型任务：展示多线程在 I/O 场景下的加速效果 |
| `demo_04_multiprocessing.py`   | 用多进程绕过 GIL，真正利用多核并行计算 |
| `demo_05_race_condition.py`    | 演示 GIL 并不能保证复合操作的线程安全（竞态条件） |
| `demo_06_release_gil_numpy.py` | 演示 NumPy 等库在底层计算时释放 GIL，从而让多线程也能并行 |

---

## 如何运行

```bash
# 进入目录
cd 00_python_gil

# 逐个运行示例
python demo_01_check_gil.py
python demo_02_cpu_bound.py
python demo_03_io_bound.py
python demo_04_multiprocessing.py
python demo_05_race_condition.py
python demo_06_release_gil_numpy.py   # 需要安装 numpy
```

> 建议在多核机器上运行 `demo_02` 和 `demo_04`，才能明显看到多进程相对多线程在 CPU 密集型任务上的优势。
