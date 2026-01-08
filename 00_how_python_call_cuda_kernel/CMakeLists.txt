# ===========================================================================
# CMakeLists.txt — Python 调用 CUDA Kernel 示例的构建配置
#
# 构建产物（位于 build/ 目录）：
#   1. libcuda_functions.so                    CUDA kernel 共享库（nvcc 编译）
#   2. cuda_hello.cpython-<ver>-<arch>.so      pybind11 扩展模块（g++ 编译，
#                                              链接 libcuda_functions.so）
#
# 常用命令：
#   cmake -DCMAKE_BUILD_TYPE=Release ..        # Release 构建
#   cmake -DCMAKE_BUILD_TYPE=Debug ..          # Debug 构建（-g -G）
#   cmake -DUSE_NATIVE_CUDA_ARCH=OFF \
#         -DCUDA_ARCHITECTURES=89 ..           # 关闭自动检测，手动指定架构
# ===========================================================================

cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------
# 构建类型
# 不加 FORCE：命令行 -DCMAKE_BUILD_TYPE=Debug 可正常覆盖默认值
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (Release/Debug)")
endif()

# project() 中声明语言会自动调用 enable_language()，无需重复声明
project(CudaHello CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# ---------------------------------------------------------------------------
# CUDA 架构检测
#
# 通过 `nvidia-smi --query-gpu=compute_cap --format=csv` 获取当前 GPU 的
# 计算能力（如 8.9 -> sm_89），避免手动指定 -arch 导致的
# "no kernel image is available" 运行时错误。
# ---------------------------------------------------------------------------
option(USE_NATIVE_CUDA_ARCH "Detect CUDA architecture from nvidia-smi" ON)

if(USE_NATIVE_CUDA_ARCH)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv
            OUTPUT_VARIABLE COMPUTE_CAP_OUTPUT
            ERROR_QUIET
        )
        # 从输出中提取形如 "8.9" 的 compute capability，并转换为 "89"
        string(REGEX MATCHALL "[0-9]+\\.[0-9]+" COMPUTE_CAP_VALUES ${COMPUTE_CAP_OUTPUT})
        if(COMPUTE_CAP_VALUES)
            list(GET COMPUTE_CAP_VALUES 0 COMPUTE_CAP)
            string(REPLACE "." "" SM_VERSION ${COMPUTE_CAP})
            message(STATUS "Detected CUDA SM_VERSION: ${SM_VERSION}")
            set(CUDA_ARCHITECTURES ${SM_VERSION} CACHE STRING "CUDA architecture" FORCE)
        else()
            message(WARNING "No compute capability detected from nvidia-smi. Using default value.")
        endif()
    else()
        message(WARNING "nvidia-smi not found. Using default CUDA architecture.")
    endif()
endif()

# 自动检测失败或被禁用时的默认架构
if(NOT CUDA_ARCHITECTURES)
    set(CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architecture")
endif()

# ---------------------------------------------------------------------------
# 依赖查找：Python 与 pybind11
#
# 以 Python_SITELIB（site-packages 路径）作为 HINTS：当系统中存在多个
# pybind11（apt 安装的系统版 / pip 安装的 Python 包版）时，优先使用
# pip 安装的版本，保证与当前 Python 解释器匹配。
# ---------------------------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED HINTS "${Python_SITELIB}")

# ---------------------------------------------------------------------------
# 目标 1：CUDA kernel 共享库（nvcc 编译 src/cuda_hello.cu）
# ---------------------------------------------------------------------------
add_library(cuda_functions SHARED src/cuda_hello.cu)
set_target_properties(cuda_functions PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
)

# Debug 模式：为 device 代码加 -G -g（cuda-gdb 需要），为 host 代码加 -g
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cuda_functions PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -g>
    )
    add_compile_options(-g)
endif()

# ---------------------------------------------------------------------------
# 目标 2：pybind11 扩展模块（g++ 编译 src/pybind_wrapper.cpp）
#
# pybind11_add_module() 自动处理：Python 相关编译/链接标志、
# .so 命名后缀、符号可见性、Release 模式 LTO 等。
# ---------------------------------------------------------------------------
pybind11_add_module(cuda_hello src/pybind_wrapper.cpp)
target_link_libraries(cuda_hello PRIVATE cuda_functions)
