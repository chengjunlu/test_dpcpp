//
// Created by john on 2022/1/26.
//
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <stdint.h>


#define ATS 0
#define PVC 1

template <int arch>
void test_threads_spawn_overhead(sycl::queue& queue);