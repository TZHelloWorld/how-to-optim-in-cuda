#!/bin/bash

# CUDA Hello World 构建脚本

# 获取脚本所在目录的绝对路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# 设置构建类型（默认为 Release）
BUILD_TYPE="Release"
if [ "$1" == "debug" ]; then
    BUILD_TYPE="Debug"
fi

echo "构建CUDA共享库..."
echo "构建类型: $BUILD_TYPE"

# 检查依赖
echo "检查依赖..."
if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到cmake，请先安装cmake"
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到nvcc，请先安装CUDA Toolkit"
    exit 1
fi

# 清理旧的构建
echo "清理旧的构建文件..."
rm -rf "${SCRIPT_DIR}/build"

# 创建构建目录
mkdir -p "${SCRIPT_DIR}/build"
cd "${SCRIPT_DIR}/build" || { echo "无法进入构建目录"; exit 1; }



# 配置 CMake
echo "配置 CMake..."
cmake -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" "${SCRIPT_DIR}"

if [ $? -ne 0 ]; then
    echo "CMake配置失败"
    exit 1
fi

# 编译
echo "编译共享库..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "编译失败"
    exit 1
fi

# 复制共享库到python目录
# echo "复制共享库到python目录..."
# cp libcuda_functions.so ../python/

cd "${SCRIPT_DIR}"

echo "构建完成!"
echo ""
echo "运行测试:"
echo "      python python/test_cuda_hello.py"
# echo "==================================================="
# python python/test_cuda_hello.py
# echo "==================================================="
# echo "done!!!"

