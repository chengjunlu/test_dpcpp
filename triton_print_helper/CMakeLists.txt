
find_package(LevelZero REQUIRED)

add_executable(print_float main.cpp print_helper.cpp)
target_compile_options(print_float PUBLIC "-fsycl")
target_compile_options(print_float PUBLIC "-O3")
#target_compile_options(print_float PUBLIC "-fsycl-targets=spirv64_spirv1.4-unknown-unknown")
#target_compile_options(print_float PUBLIC "-fsycl-targets=spirv64-spirv1.4")

target_link_libraries(print_float PUBLIC "sycl")
target_link_options(print_float PUBLIC "-fsycl")

target_link_libraries(print_float PRIVATE ze_loader)