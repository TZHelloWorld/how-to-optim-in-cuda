"""test_cuda_hello.py — Python 测试入口。

验证完整调用链路：
    Python -> pybind11 绑定层 (cuda_hello.so) -> C++ 封装 -> CUDA Kernel

运行前需先编译扩展模块：
    bash build.sh          # Release 构建
    bash build.sh debug    # Debug 构建（含 -g -G 调试信息，供 cuda-gdb 使用）

运行方式：
    python python/test_cuda_hello.py
"""

import os
import sys

# 编译产物 cuda_hello.cpython-*.so 位于 build/ 目录，将其加入模块搜索路径。
# 置于模块顶层，保证 `python -m ipdb` 等间接执行方式同样生效。
_BUILD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "build")
)
sys.path.append(_BUILD_DIR)


def main() -> int:
    """导入扩展模块并触发 CUDA kernel，返回进程退出码。"""
    try:
        import cuda_hello
    except ImportError as e:
        print(f"[test_cuda_hello.py] 导入模块失败: {e}")
        print(f"[test_cuda_hello.py] 模块搜索路径: {_BUILD_DIR}")
        print("[test_cuda_hello.py] 请先编译模块: bash build.sh")
        return 1

    print("[test_cuda_hello.py] 成功导入 cuda_hello 模块")

    try:
        print("[test_cuda_hello.py] 调用 CUDA hello kernel...")
        cuda_hello.hello()
        print("[test_cuda_hello.py] CUDA kernel 执行完成!")
    except Exception as e:  # noqa: BLE001 — 演示脚本，捕获所有运行时错误
        print(f"[test_cuda_hello.py] 执行出错: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
