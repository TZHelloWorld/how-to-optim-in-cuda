"""
测试CUDA Hello World模块
"""

import sys
import os

if __name__ == '__main__':

    # 将 build 目录添加到 Python 路径
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

    try:
        import cuda_hello
        print("[test_cuda_hello.py] 成功导入 cuda_hello 模块")

        # 调用hello函数
        print("[test_cuda_hello.py] 调用 CUDA hello kernel...")
        cuda_hello.hello()
        print("[test_cuda_hello.py] CUDA kernel 执行完成!")

    except ImportError as e:
        print(f"[test_cuda_hello.py] 导入模块失败: {e}")
        print("[test_cuda_hello.py] 请确保已经编译了模块: mkdir build && cd build && cmake .. && make")
        sys.exit(1)
    except Exception as e:
        print(f"[test_cuda_hello.py] 执行出错: {e}")
        sys.exit(1)