#!/bin/bash

# CUDA Hello World 构建脚本

echo "构建CUDA共享库..."

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
rm -rf build
# rm -f python/*.so

# 创建构建目录
mkdir -p build
cd build

# 配置CMake
echo "配置CMake..."
cmake ..

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

cd ..

echo "构建完成!"
echo ""
echo "运行测试:"
echo "      python python/test_cuda_hello.py"
# echo "==================================================="
# python python/test_cuda_hello.py
# echo "==================================================="
# echo "done!!!"

