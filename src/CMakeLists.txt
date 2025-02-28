cmake_minimum_required(VERSION 3.5.1)

project(nanotrack_backbone)

add_compile_options(-std=c++11)


# 设置构建类型为Debug
set(CMAKE_BUILD_TYPE Debug)


#设置可执行文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../../../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else()
    message(STATUS "set INC_PATH: ${INC_PATH}")
endif ()

set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else()
    message(STATUS "set LIB_PATH: ${LIB_PATH}")
endif ()

include_directories(
    ${INC_PATH}/runtime/include/
)

link_directories(
    ${LIB_PATH}
)

find_package(OpenCV 4 REQUIRED)
message(STATUS "\n[FindLibrary]  ---------OpenCV library status-----------")
message(STATUS "        version: ${OpenCV_VERSION}")
message(STATUS "        libraries: ${OpenCV_LIBS}")
message(STATUS "        include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "-------------------------------------------------------")
include_directories(${OpenCV_INCLUDE_DIRS})



add_executable(main main.cpp)
target_link_libraries(main ascendcl stdc++ opencv_core opencv_imgproc opencv_imgcodecs dl rt)


# install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

