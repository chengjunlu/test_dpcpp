//
// Created by john on 2022/1/27.
//
#include "ThreadsSpawnTest.hpp"
#include <map>
#include <vector>

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

ulong* to_host(sycl::queue& queue, ulong* time_stamp, unsigned size) {
  ulong* out_host = (ulong*)malloc(sizeof(ulong)* NUM_WORK_GROUP);
  auto e = queue.memcpy(out_host, time_stamp, sizeof(ulong)* NUM_WORK_GROUP);
  e.wait();
  return out_host;
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
//  print_time("time_entry", queue, time_entry, NUM_WORK_GROUP);
//  print_time("time_exit", queue, time_exit, NUM_WORK_GROUP);
//  print_time("sub_slice_id", queue, sub_slice_id, NUM_WORK_GROUP);
//  print_time("eu_id", queue, eu_id, NUM_WORK_GROUP);
  ulong* time_entry_host = to_host(queue, time_entry, NUM_WORK_GROUP);
  ulong* time_exit_host = to_host(queue, time_exit, NUM_WORK_GROUP);
  ulong* sub_slice_id_host = to_host(queue, sub_slice_id, NUM_WORK_GROUP);
  ulong* eu_id_host = to_host(queue, eu_id, NUM_WORK_GROUP);

  struct time_stamp {
    ulong work_group_id;
    ulong entry;
    ulong exit;
    ulong sub_slice_id;
    ulong eu_id;
  };

  std::map<unsigned, std::vector<struct time_stamp>> time_map;

  for (unsigned i = 0; i < NUM_WORK_GROUP; ++i) {
    struct time_stamp ts;
    ts.work_group_id = i;
    ts.entry = time_entry_host[i];
    ts.exit = time_exit_host[i];
    ts.sub_slice_id = sub_slice_id_host[i];
    ts.eu_id = eu_id_host[i];
    time_map[ts.sub_slice_id].push_back(ts);
  }

  for (auto& kv : time_map) {
    printf("sub_slice_id: %d, total work group num: %d\n", kv.first, kv.second.size());
      for (auto& ts : kv.second) {
        printf("work_group_id: %ld, eu_id: %ld, entry: %ld, exit: %ld\n", ts.work_group_id, ts.eu_id, ts.entry,
               ts.exit);
      }
  }
}