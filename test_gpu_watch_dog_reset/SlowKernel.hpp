//
// Created by john on 2022/1/26.
//
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <stdint.h>

void test_slow_kernel(sycl::queue& queue, uint32_t loop_cnt);