# 设置 CMake 的最低版本要求
cmake_minimum_required(VERSION 3.18)

# 设置构建类型为 Debug
set(CMAKE_BUILD_TYPE Debug)

# 设置 CUDA 主机编译器为 g++
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++)

# 定义项目名称和支持的语言
project(CudaHello CUDA CXX)

# 设置 C++ 标准为 C++14
set(CMAKE_CXX_STANDARD 14)

# 设置 CUDA 标准为 C++14
set(CMAKE_CUDA_STANDARD 14)

# 启用 CUDA 语言支持
enable_language(CUDA)



# 设置 CUDA 架构（根据 GPU 调整这个值）
# set(CUDA_ARCHITECTURES 89)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 自动检测 CUDA 架构的选项 
option(USE_NATIVE_CUDA_ARCH "Detect CUDA architecture from nvidia-smi" ON)

# 自动检测逻辑, 原理就是根据 nvidia-smi --query-gpu=compute_cap --format=csv 命令获取。
if(USE_NATIVE_CUDA_ARCH)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv
            OUTPUT_VARIABLE COMPUTE_CAP_OUTPUT
            ERROR_QUIET
        )

        # 提取 compute_cap 值
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

# 如果未检测到或用户禁用自动检测，则使用默认值
if(NOT CUDA_ARCHITECTURES)
    set(CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architecture")
endif() 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# 查找 Python 解释器和开发组件
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# 这个主要是针对 Linux 上多个 python 版本导致的多个 pybind11 出现的问题进行修复：
# ==============================================================================================
# 根据 $(python3 -m pybind11 --includes) 动态获取 pybind11 的包含路径, 即优先使用 pip 安装的 pybind 
# 而不是优先使用 CMake 的 find_package 配置（系统中通过 apt 或 yum 安装了 pybind11-dev）
execute_process( # 用于运行 Python 命令，动态获取 pybind11 的包含路径；
    COMMAND ${Python_EXECUTABLE} -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDE_OUTPUT
    ERROR_VARIABLE PYBIND11_INCLUDE_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_CONFIG_RESULT
)

if(NOT PYBIND11_CONFIG_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to get pybind11 include paths: ${PYBIND11_INCLUDE_ERROR}")
endif()

# 解析输出中的 -I 路径
string(REGEX MATCHALL "-I([^ ]+)" PYBIND11_INCLUDE_FLAGS ${PYBIND11_INCLUDE_OUTPUT})
set(PYBIND11_INCLUDE_DIRS "")
foreach(flag ${PYBIND11_INCLUDE_FLAGS})
    string(SUBSTRING ${flag} 2 -1 dir)
    list(APPEND PYBIND11_INCLUDE_DIRS ${dir})
endforeach()

# 添加 pybind11 的包含路径
include_directories(${PYBIND11_INCLUDE_DIRS})

message(STATUS "add include dirctories dir: ${PYBIND11_INCLUDE_DIRS}")
# ==============================================================================================

# 查找 pybind11 包
find_package(pybind11 CONFIG REQUIRED)

# 添加 CUDA 文件并创建共享库
add_library(cuda_functions SHARED src/cuda_hello.cu)

# 设置目标属性，指定 CUDA 架构
set_target_properties(cuda_functions PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})

# 如果是 Debug 模式，为 CUDA 编译器添加调试选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cuda_functions PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G -g>)
endif()

# 创建 pybind11 模块
pybind11_add_module(cuda_hello src/pybind_wrapper.cpp)

# 将 CUDA 函数库链接到 pybind11 模块
target_link_libraries(cuda_hello PRIVATE cuda_functions)