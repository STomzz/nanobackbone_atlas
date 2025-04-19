# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(ACL_NAnotrack)

# set(CURRENT_DIR ${CMAKE_CURRENT_LIST_DIR})


# Compile options
add_compile_options(-std=c++17)

set(CMAKE_BUILD_TYPE Debug CACHE STRING "default is Debug")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

#设置可执行文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_SOURCE_DIR}/bin)

set(INC_PATH $ENV{DDK_PATH})

set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else()
    message(STATUS "set INC_PATH: ${INC_PATH}")
endif ()


set(LIB_PATH $ENV{NPU_HOST_LIB})

set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else()
    message(STATUS "set LIB_PATH: ${LIB_PATH}")
endif ()

# 自动查找 OpenCV
find_package(OpenCV 4 REQUIRED)
# 包含 OpenCV 的头文件路径
include_directories(${OpenCV_INCLUDE_DIRS})

#查找thread
find_package(Threads REQUIRED)
# include_directories(${Threads_INCLUDE_DIRS})


include_directories(
   ${INC_PATH}/runtime/include/
   ${CMAKE_SOURCE_DIR}/include
   ${CMAKE_SOURCE_DIR}/kcf/include
)

link_directories(
    ${LIB_PATH}
)

add_subdirectory(${CMAKE_SOURCE_DIR}/src)
