//
// Created by guangyey on 5/11/23.
//
#include <sycl.hpp>

extern "C" {
SYCL_EXTERNAL void print_cur_float(uint32_t tid, uint32_t stage, float f);

SYCL_EXTERNAL void print_acc_float(uint32_t tid, uint32_t stage, float f);

SYCL_EXTERNAL void print_output_float(uint32_t tid, uint32_t stage, float f);
}