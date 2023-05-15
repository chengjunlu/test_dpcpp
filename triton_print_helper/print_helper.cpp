//
// Created by guangyey on 5/11/23.
//
#include "print_helper.h"



extern "C" {
SYCL_EXTERNAL void print_cur_float(uint32_t tid, uint32_t stage, sycl::_V1::local_ptr<float>::pointer_t ptr, float f) {
  static const __attribute__((opencl_constant)) char var[] = "tid %d stage %d ptr %p cur = %f\n";
  sycl::ext::oneapi::experimental::printf(var, tid, stage, ptr, f);
}

SYCL_EXTERNAL void print_acc_float(uint32_t tid, uint32_t stage, sycl::_V1::local_ptr<float>::pointer_t ptr, float f) {
  static const __attribute__((opencl_constant)) char var[] = "tid %d stage %d ptr %p acc = %f\n";
  sycl::ext::oneapi::experimental::printf(var, tid, stage, ptr, f);
}

SYCL_EXTERNAL void print_output_float(uint32_t tid, uint32_t stage, sycl::_V1::local_ptr<float>::pointer_t ptr, float f) {
  static const __attribute__((opencl_constant)) char var[] = "tid %d stage %d ptr %p output = %f\n";
  sycl::ext::oneapi::experimental::printf(var, tid, stage, ptr, f);
}

SYCL_EXTERNAL void print_write_index(uint32_t tid, uint32_t stage, uint32_t index) {
  static const __attribute__((opencl_constant)) char var[] = "tid %d stage %d write_index = %d\n";
  sycl::ext::oneapi::experimental::printf(var, tid, stage, index);
}

//SYCL_EXTERNAL void dpcpp_barrier(sycl::nd_item<1> item) {
//  item.barrier(sycl::access::fence_space::local_space);
//}

}