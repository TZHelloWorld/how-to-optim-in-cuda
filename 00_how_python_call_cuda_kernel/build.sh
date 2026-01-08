#!/bin/bash
# ===========================================================================
# build.sh — 一键构建脚本
#
# 用法：
#   bash build.sh          # Release 构建（默认）
#   bash build.sh debug    # Debug 构建（含 -g -G 调试信息，供 cuda-gdb 使用）
#
# 产物：build/cuda_hello.cpython-<ver>-<arch>.so
# ===========================================================================
set -e  # 任意命令失败立即退出

# 脚本所在目录的绝对路径（支持从任意工作目录调用）
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# --------------------------------------------------------------------------
# 解析构建类型
# --------------------------------------------------------------------------
BUILD_TYPE="Release"
if [ "$1" == "debug" ]; then
    BUILD_TYPE="Debug"
fi

echo "构建 CUDA 共享库..."
echo "构建类型: ${BUILD_TYPE}"

# --------------------------------------------------------------------------
# 依赖检查
# --------------------------------------------------------------------------
echo "检查依赖..."
if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到 cmake，请先安装 cmake"
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到 nvcc，请先安装 CUDA Toolkit"
    exit 1
fi

# --------------------------------------------------------------------------
# 构建（每次全量重建，保证 Release/Debug 切换干净；
#       如需增量编译，可注释掉 rm -rf 行）
# --------------------------------------------------------------------------
echo "清理旧的构建文件..."
rm -rf "${SCRIPT_DIR}/build"

mkdir -p "${SCRIPT_DIR}/build"
cd "${SCRIPT_DIR}/build"

echo "配置 CMake..."
cmake -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" "${SCRIPT_DIR}"

echo "编译共享库..."
make -j"$(nproc)"

cd "${SCRIPT_DIR}"

echo "构建完成!"
echo ""
echo "运行测试:"
echo "      python python/test_cuda_hello.py"
