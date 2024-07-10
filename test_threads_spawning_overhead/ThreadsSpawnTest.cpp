//
// Created by john on 2022/1/27.
//
#include "ThreadsSpawnTest.hpp"

#define NUM_WORK_GROUP 64
#define SLM_SIZE 64 * 1024
#define OVERLOADABLE __attribute__((overloadable))

#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL extern "C" ulong __builtin_spirv_OpReadClockKHR_i64_i32(uint scope);
SYCL_EXTERNAL extern "C" OVERLOADABLE ulong intel_get_cycle_counter( void );
SYCL_EXTERNAL uint OVERLOADABLE intel_get_subslice_id( void );
SYCL_EXTERNAL uint OVERLOADABLE intel_get_eu_id( void );
#define __SYCL_CONSTANT_AS __attribute__((opencl_constant))
#else
ulong intel_get_cycle_counter( void ) {
  return 0;
}
ulong __builtin_spirv_OpReadClockKHR_i64_i32(uint scope) {
  return 0;
}
uint intel_get_subslice_id( void ) {
  return 0;
}
uint intel_get_eu_id( void ) {
  return 0;
}
#define __SYCL_CONSTANT_AS
#endif

void print_time(const char* prefix, sycl::queue& queue, ulong* time_stamp, unsigned size) {
  ulong* out_host = (ulong*)malloc(sizeof(ulong)* NUM_WORK_GROUP);
  auto e = queue.memcpy(out_host, time_stamp, sizeof(ulong)* NUM_WORK_GROUP);
  e.wait();
  for (unsigned i = 0; i < size; i++) {
    printf("%s[%d] = %ld\n", prefix, i, out_host[i]);
  }
}

void test_subgroup_ballot(sycl::queue& queue){

  ulong* time_entry = (ulong*)malloc_device(sizeof(ulong)* NUM_WORK_GROUP, queue);
  ulong* time_exit = (ulong*)malloc_device(sizeof(ulong)* NUM_WORK_GROUP, queue);
  ulong* sub_slice_id = (ulong*)malloc_device(sizeof(ulong)* NUM_WORK_GROUP, queue);
  ulong* eu_id = (ulong*)malloc_device(sizeof(ulong)* NUM_WORK_GROUP, queue);

  auto cgf = [&](sycl::handler & cgh) {
      using share_mem_t = sycl::local_accessor<int8_t, 1>;
      share_mem_t local_buffer = share_mem_t(SLM_SIZE, cgh);
      cgh.parallel_for(
              sycl::nd_range<1>(NUM_WORK_GROUP, 1),
              [=](sycl::nd_item<1> id) {
                  local_buffer[0] = 0;
                  auto work_group_linear_id = id.get_global_linear_id();
                  auto start_time = __builtin_spirv_OpReadClockKHR_i64_i32(0);
                  time_entry[work_group_linear_id] = start_time;

                  sub_slice_id[work_group_linear_id] = intel_get_subslice_id();
                  eu_id[work_group_linear_id] = intel_get_eu_id();

                  auto end_time = __builtin_spirv_OpReadClockKHR_i64_i32(0);
                  time_exit[work_group_linear_id] = end_time;
              });
  };

  queue.submit(cgf);
  queue.wait();
  print_time("time_entry", queue, time_entry, NUM_WORK_GROUP);
  print_time("time_exit", queue, time_exit, NUM_WORK_GROUP);
  print_time("sub_slice_id", queue, sub_slice_id, NUM_WORK_GROUP);
  print_time("eu_id", queue, eu_id, NUM_WORK_GROUP);
}