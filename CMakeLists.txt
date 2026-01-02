# ==============================================================================
# Project Setup
# ==============================================================================
cmake_minimum_required(VERSION 3.18)

project(FSI_Simulator LANGUAGES CXX CUDA C)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# 防止release模式下的LTO问题
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug, Release, RelWithDebInfo, MinSizeRel)." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
  message(STATUS "No build type specified, defaulting to ${CMAKE_BUILD_TYPE}.")
endif()

# 为所有目标设置可执行文件输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)

# 为所有目标设置库文件输出目录
# 这同时影响静态库和共享库
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)

# 静态库的输出目录（可选，如果想和共享库分开）
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)

message(STATUS "Configuring FSI_Simulator version 0.1.0")
message(STATUS "Compiler CXX: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler CUDA: ${CMAKE_CUDA_COMPILER}")

# 启用 compile_commands.json 生成，用于编辑器IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)


# ==============================================================================
# Find Dependencies (via vcpkg)
# ==============================================================================
message(STATUS "Finding required packages...")

find_package(Python COMPONENTS Interpreter REQUIRED)
message(STATUS "Found Python Interpreter: ${Python_EXECUTABLE}")
message(STATUS "Found Python Version: ${Python_VERSION}")

find_package(Python ${Python_VERSION} EXACT
    COMPONENTS Development.Module # .Module 通常是pybind11需要的
    REQUIRED)
message(STATUS "Found Python Development Libraries: ${Python_LIBRARIES}")

find_package(Eigen3 3.3 REQUIRED CONFIG)
find_package(pybind11 2.6 REQUIRED CONFIG)
find_package(spdlog REQUIRED)
find_package(nlohmann_json 3.2 REQUIRED)
find_package(glm REQUIRED)
find_package(VTK REQUIRED)
find_package(OpenMP REQUIRED)

option(USE_CUDSS "Enable the cuDSS sparse direct solver" ON)
option(USE_SUITESPARSE "Enable the CholmodLDLT solver" ON)

if(USE_SUITESPARSE)
    message(STATUS "Attempting to find SuiteSparse/CHOLMOD...")
    find_package(CHOLMOD CONFIG REQUIRED) 
    set(USE_SUITESPARSE_LIBS SuiteSparse::CHOLMOD_static) 
else()
    message(STATUS "Skipping SuiteSparse/CHOLMOD as per configuration.")
    set(USE_SUITESPARSE_LIBS "")
endif()

if(USE_CUDSS)
    # 现代CMake(3.18+)可以找到CUDA自带的库
    # CUDA::cudss 是CMake 3.25+引入的官方目标名
    # 对于旧版CMake, 我们可能需要回退到find_library
    find_package(CUDAToolkit)
    if(TARGET CUDA::cudss)
        message(STATUS "Found cuDSS via CMake's CUDA support (TARGET CUDA::cudss)")
        set(USE_CUDSS_LIBS CUDA::cudss)
    else()
        # 手动查找，适用于较旧的CMake版本
        find_library(CUDSS_LIBRARY NAMES cudss
                     HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib/x64)
        if(CUDSS_LIBRARY)
            message(STATUS "Found cuDSS library manually: ${CUDSS_LIBRARY}")
            set(USE_CUDSS_LIBS ${CUDSS_LIBRARY})
        else()
            message(FATAL_ERROR "cuDSS library not found, but USE_CUDSS is ON. Please check your CUDA installation.")
        endif()
    endif()
else()
    message(STATUS "Skipping cuDSS as per configuration.")
    set(USE_CUDSS_LIBS "")
endif()

# find_package(tinyobjloader REQUIRED) # 如果需要
# find_package(GAIA REQUIRED) # 假设你的VBD库也通过vcpkg或find_package管理

message(STATUS "  - Eigen3: ${EIGEN3_VERSION}")
message(STATUS "  - pybind11: ${pybind11_VERSION}")
message(STATUS "  - spdlog: ${spdlog_VERSION}")
message(STATUS "  - nlohmann_json: ${nlohmann_json_VERSION}")
message(STATUS "  - glm: ${GLM_VERSION_STRING}")
message(STATUS "  - OpenMP: ${OpenMP_CXX_FLAGS}")

# ==============================================================================
# Project Structure and Source Files
# ==============================================================================

# 定义头文件目录
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 自动查找所有源文件 (.cpp, .cu)
# 注意：当添加或删除文件时，需要重新运行CMake
file(GLOB_RECURSE CORE_SOURCES
    "src/*.cpp"
    "src/*.cu"
)

# 排除Python绑定文件，它将单独处理
file(GLOB PYTHON_BINDINGS "src/python/*.cpp")
list(REMOVE_ITEM CORE_SOURCES ${PYTHON_BINDINGS})

# 排除main.cpp，它将用于可执行文件
file(GLOB MAIN_EXECUTABLE "src/app/*.cpp")
list(REMOVE_ITEM CORE_SOURCES ${MAIN_EXECUTABLE})

# ==============================================================================
# Build Core Library (Static Library)
# ==============================================================================
# 我们将核心功能编译成一个静态库，这样可以被Python绑定和C++可执行文件共享

add_library(fsi_core OBJECT ${CORE_SOURCES})

set_target_properties(fsi_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 为fsi_core设置包含目录
target_include_directories(fsi_core PUBLIC
    # 当安装时，头文件在'include'目录下
    $<INSTALL_INTERFACE:include>
    # 当在构建树中时，头文件在'${PROJECT_SOURCE_DIR}/include'目录下
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

if(USE_SUITESPARSE)
    target_compile_definitions(fsi_core PUBLIC USE_SUITESPARSE)
endif()

if(USE_CUDSS)
    target_compile_definitions(fsi_core PUBLIC USE_CUDSS)
endif()

target_compile_definitions(fsi_core PUBLIC GLM_ENABLE_EXPERIMENTAL)

# 为fsi_core链接它需要的所有库
# 使用PUBLIC关键字，这样任何链接到fsi_core的目标也会自动链接到这些库
target_link_libraries(fsi_core PUBLIC
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    glm::glm
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    ${CUDA_LIBRARIES}
    VTK::CommonCore
    VTK::CommonDataModel
    VTK::IOXML # 用于写入.vtu, .vts等XML格式文件
    ${USE_SUITESPARSE_LIBS}
    ${USE_CUDSS_LIBS}
    CUDA::cusparse
    CUDA::cusolver
)

# 为fsi_core设置CUDA可重定位代码
set_target_properties(fsi_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ==============================================================================
# Build Python Bindings
# ==============================================================================
if(PYTHON_BINDINGS)
    message(STATUS "Building Python bindings...")
    
    # 使用 pybind11 提供的函数来创建 Python 模块
    # 第一个参数是模块名，也是最终生成的文件名 (e.g., fsi_simulator.so)
    pybind11_add_module(fsi_simulator ${PYTHON_BINDINGS} $<TARGET_OBJECTS:fsi_core>)

    set_property(TARGET fsi_simulator PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)


    # 链接核心库和 pybind11 库
    target_link_libraries(fsi_simulator PRIVATE
        $<TARGET_PROPERTY:fsi_core,INTERFACE_LINK_LIBRARIES>
        pybind11::module # 链接pybind11
    )

    install(TARGETS fsi_simulator
            LIBRARY  # 表示这是一个库文件
            DESTINATION . # 安装到包的根目录
            COMPONENT python_module # 匹配 pyproject.toml 中的组件名
    )
    
    # 2. 创建并安装 __init__.py 文件
    # 这样做可以让用户直接 `import fsi_simulator`
    # 并从.so文件中导入所有内容
    set(INIT_PY_CONTENT "from .fsi_simulator import *")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" "${INIT_PY_CONTENT}")
    
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/__init__.py"
            DESTINATION . # 和.so文件安装到同一个目录下
            COMPONENT python_module
    )

endif()


# ==============================================================================
# Build C++ Executable (for testing or C++-only runs)
# ==============================================================================
if(MAIN_EXECUTABLE)
    message(STATUS "Building C++ executable...")

    add_executable(fsi_main ${MAIN_EXECUTABLE} $<TARGET_OBJECTS:fsi_core>)

    set_property(TARGET fsi_main PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    # 链接核心库
    target_link_libraries(fsi_main PRIVATE
        $<TARGET_PROPERTY:fsi_core,INTERFACE_LINK_LIBRARIES>
    )

endif()


# ==============================================================================
# Final Touches
# ==============================================================================
# 打印目标信息
message(STATUS "Configuration finished. Targets to be built:")
if(TARGET fsi_core)
    message(STATUS "  - Library: fsi_core")
endif()
if(TARGET fsi_simulator)
    message(STATUS "  - Python Module: fsi_simulator")
endif()
if(TARGET fsi_main)
    message(STATUS "  - Executable: fsi_main")
endif()