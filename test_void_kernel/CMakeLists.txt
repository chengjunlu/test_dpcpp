
find_package(LevelZero REQUIRED)

add_executable(triton_void_kernel main.cpp)
target_compile_options(triton_void_kernel PUBLIC "-fsycl")
target_compile_options(triton_void_kernel PUBLIC "-O3")
target_link_libraries(triton_void_kernel PUBLIC "sycl")
target_link_options(triton_void_kernel PUBLIC "-fsycl")

target_link_libraries(triton_void_kernel PRIVATE ze_loader)