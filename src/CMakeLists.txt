
FILE(GLOB_RECURSE nanosrcfiles "${CMAKE_SOURCE_DIR}/src/*.cpp")

FILE(GLOB_RECURSE kcfsrcfiles "${CMAKE_SOURCE_DIR}/kcf/src/*.cpp")

set(TEST_CODE_THREADPOOL OFF CACHE BOOL "enable test threadpool code")
if(TEST_CODE_THREADPOOL)
        add_executable(test_thread ${CMAKE_SOURCE_DIR}/test/test_thread.cpp ${CMAKE_SOURCE_DIR}/src/ThreadPool.cpp)
        target_link_libraries(test_thread stdc++ Threads::Threads)

endif()
set(TEST_CODE_KCF OFF CACHE BOOL "enable test kcf code")
if(TEST_CODE_KCF)
        add_executable(test_kcf ${CMAKE_SOURCE_DIR}/test/test_kcf.cpp ${kcfsrcfiles})
        target_link_libraries(test_kcf stdc++ ${OpenCV_LIBS})
endif()

# add_executable(main
#         ${CMAKE_CURRENT_SOURCE_DIR}/ACNNModel_B.cpp
#         ${CMAKE_CURRENT_SOURCE_DIR}/ACNNModel_N.cpp
#         ${CMAKE_CURRENT_SOURCE_DIR}/nanotrack.cpp
#         ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
add_executable(main
        ${nanosrcfiles}
        ${kcfsrcfiles}
        )


# target_link_libraries(main ascendcl stdc++ opencv_core opencv_imgproc opencv_imgcodecs dl rt)
target_link_libraries(main ${OpenCV_LIBS} ascendcl stdc++ dl rt)


# install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})