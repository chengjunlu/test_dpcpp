
find_package(LevelZero REQUIRED)

add_executable(triton_matmul main.cpp)
target_compile_options(triton_matmul PUBLIC "-fsycl")
target_compile_options(triton_matmul PUBLIC "-O3")
target_link_libraries(triton_matmul PUBLIC "sycl")
target_link_options(triton_matmul PUBLIC "-fsycl")

target_link_libraries(triton_matmul PRIVATE ze_loader)