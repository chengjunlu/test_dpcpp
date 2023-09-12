
find_package(LevelZero REQUIRED)

add_executable(test_invoke_esimd_layout main.cpp)
#compile flag
target_compile_options(test_invoke_esimd_layout PUBLIC "-fsycl")
target_compile_options(test_invoke_esimd_layout PUBLIC "-O3")

#linker flag
target_link_options(test_invoke_esimd_layout PUBLIC "-fsycl")
target_link_options(test_invoke_esimd_layout PUBLIC "-O3")

#library
target_link_libraries(test_invoke_esimd_layout PUBLIC "sycl")
target_link_libraries(test_invoke_esimd_layout PRIVATE ze_loader)