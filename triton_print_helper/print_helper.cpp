//
// Created by guangyey on 5/11/23.
//
#include "print_helper.h"



extern "C" {
SYCL_EXTERNAL void print_cur_float(uint32_t tid, uint32_t stage, float f) {
  static const __attribute__((opencl_constant)) char var[] = "tid %d stage %d cur = %f\n";
  sycl::ext::oneapi::experimental::printf(var, tid, stage, f);
}

SYCL_EXTERNAL void print_acc_float(uint32_t tid, uint32_t stage, float f) {
  static const __attribute__((opencl_constant)) char var[] = "tid %d stage %d acc = %f\n";
  sycl::ext::oneapi::experimental::printf(var, tid, stage, f);
}

SYCL_EXTERNAL void print_output_float(uint32_t tid, uint32_t stage, float f) {
  static const __attribute__((opencl_constant)) char var[] = "tid %d stage %d output = %f\n";
  sycl::ext::oneapi::experimental::printf(var, tid, stage, f);
}
}