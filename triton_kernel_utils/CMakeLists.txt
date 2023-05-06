
find_package(LevelZero REQUIRED)

add_executable(triton_launch main.cpp)
target_compile_options(triton_launch PUBLIC "-fsycl")
target_compile_options(triton_launch PUBLIC "-O3")
target_link_libraries(triton_launch PUBLIC "sycl")
target_link_options(triton_launch PUBLIC "-fsycl")

target_link_libraries(triton_launch PRIVATE ze_loader)