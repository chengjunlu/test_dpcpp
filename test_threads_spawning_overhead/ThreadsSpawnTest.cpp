//
// Created by john on 2022/1/27.
//
#include "ThreadsSpawnTest.hpp"
#include <map>
#include <vector>

#define NUM_WORK_GROUP 256
#define OVERLOADABLE __attribute__((overloadable))
#define FREQUENCY 1.6e9

#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL extern "C" ulong __builtin_spirv_OpReadClockKHR_i64_i32(uint scope);
SYCL_EXTERNAL extern "C" OVERLOADABLE ulong intel_get_cycle_counter( void );
SYCL_EXTERNAL uint OVERLOADABLE intel_get_dual_subslice_id( void );
SYCL_EXTERNAL uint OVERLOADABLE intel_get_slice_id( void );
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
uint intel_get_dual_subslice_id( void ) {
  return 0;
}
uint intel_get_slice_id( void ) {
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

template <int arch>
void test_threads_spawn_overhead(sycl::queue& queue){

  ulong* time_entry = (ulong*)malloc_device(sizeof(ulong)* NUM_WORK_GROUP, queue);
  ulong* time_exit = (ulong*)malloc_device(sizeof(ulong)* NUM_WORK_GROUP, queue);
  ulong* dual_subslice_id;
  if constexpr (arch == ATS) {
    dual_subslice_id = (ulong*)malloc_device(sizeof(ulong)* NUM_WORK_GROUP, queue);
  }
  ulong* slice_id = (ulong*)malloc_device(sizeof(ulong)* NUM_WORK_GROUP, queue);
  ulong* sub_slice_id = (ulong*)malloc_device(sizeof(ulong)* NUM_WORK_GROUP, queue);
  ulong* eu_id = (ulong*)malloc_device(sizeof(ulong)* NUM_WORK_GROUP, queue);

  auto slm_size = queue.get_device().get_info<sycl::info::device::local_mem_size>();

  auto cgf = [&](sycl::handler & cgh) {
      using share_mem_t = sycl::local_accessor<int8_t, 1>;
      share_mem_t local_buffer = share_mem_t(slm_size, cgh);
      cgh.parallel_for(
              sycl::nd_range<1>(NUM_WORK_GROUP, 1),
              [=](sycl::nd_item<1> id) {
                  auto work_group_linear_id = id.get_group_linear_id();
                  auto start_time = __builtin_spirv_OpReadClockKHR_i64_i32(0);
                  time_entry[work_group_linear_id] = start_time;

                  slice_id[work_group_linear_id] = intel_get_slice_id();

                  if constexpr (arch == ATS)
                    dual_subslice_id[work_group_linear_id] = intel_get_dual_subslice_id();

                  sub_slice_id[work_group_linear_id] = intel_get_subslice_id();
                  eu_id[work_group_linear_id] = intel_get_eu_id();
                  // touch the SLM
                  auto a = local_buffer[work_group_linear_id];
                  local_buffer[work_group_linear_id] = a + 1;

                  auto end_time = __builtin_spirv_OpReadClockKHR_i64_i32(0);
                  time_exit[work_group_linear_id] = end_time;
              });
  };

  queue.submit(cgf);
  queue.wait();
//  print_time("time_entry", queue, time_entry, NUM_WORK_GROUP);
//  print_time("time_exit", queue, time_exit, NUM_WORK_GROUP);
//  print_time("slice_id", queue, slice_id, NUM_WORK_GROUP);
//  print_time("sub_slice_id", queue, sub_slice_id, NUM_WORK_GROUP);
//  print_time("eu_id", queue, eu_id, NUM_WORK_GROUP);
  ulong* time_entry_host = to_host(queue, time_entry, NUM_WORK_GROUP);
  ulong* time_exit_host = to_host(queue, time_exit, NUM_WORK_GROUP);
  ulong* slice_id_host = to_host(queue, slice_id, NUM_WORK_GROUP);

  ulong* dual_subslice_id_host;
  if constexpr (arch == ATS)
    dual_subslice_id_host = to_host(queue, dual_subslice_id, NUM_WORK_GROUP);

  ulong* sub_slice_id_host = to_host(queue, sub_slice_id, NUM_WORK_GROUP);
  ulong* eu_id_host = to_host(queue, eu_id, NUM_WORK_GROUP);

  struct time_stamp {
    ulong work_group_id;
    ulong entry;
    ulong exit;
    ulong dual_subslice_id;
    ulong slice_id;
    ulong sub_slice_id;
    ulong eu_id;
  };

  std::map<std::tuple<unsigned, unsigned, unsigned>, std::vector<struct time_stamp>> time_map;

  for (unsigned i = 0; i < NUM_WORK_GROUP; ++i) {
    struct time_stamp ts;
    ts.work_group_id = i;
    ts.entry = time_entry_host[i];
    ts.exit = time_exit_host[i];
    ts.slice_id = slice_id_host[i];
    if constexpr (arch == ATS)
      ts.dual_subslice_id = dual_subslice_id_host[i];
    else
      ts.dual_subslice_id = 0;
    ts.sub_slice_id = sub_slice_id_host[i];
    ts.eu_id = eu_id_host[i];
    time_map[{ts.slice_id, ts.dual_subslice_id, ts.sub_slice_id}].push_back(ts);
  }

  std::vector<struct time_stamp> time_stamp;
  for (auto& kv : time_map) {
    time_stamp.insert(time_stamp.end(), kv.second.begin(), kv.second.end());
  }

  std::sort(time_stamp.begin(), time_stamp.end(), [](const struct time_stamp& a, const struct time_stamp& b) {
    return a.entry < b.entry;
  });

#if 0
  for (unsigned i = 0; i < time_stamp.size(); i++) {
    auto& ts = time_stamp[i];
    if (i == 0) {
      printf("john lu work_group_id: %ld, slice_id:%ld, dual_subslice_id: %ld, sub_slice_id: %ld, eu_id: %ld, entry: %ld, exit: %ld\n", ts.work_group_id, ts.slice_id, ts.dual_subslice_id, ts.sub_slice_id, ts.eu_id, ts.entry,
             ts.exit);
      continue;
    }
    auto& prev_ts = time_stamp[i-1];
    printf("john lu work_group_id: %ld, slice_id:%ld, dual_subslice_id: %ld, sub_slice_id: %ld, eu_id: %ld, entry: %ld, exit: %ld, diff start: %ld\n", ts.work_group_id, ts.slice_id, ts.dual_subslice_id, ts.sub_slice_id, ts.eu_id, ts.entry,
           ts.exit, ts.entry - prev_ts.entry);
  }
#endif

  for (auto& kv : time_map) {
    printf("slice_id: %d, dual_subslice_id: %d, sub_slice_id: %d, total work group num: %zu\n", std::get<0>(kv.first), std::get<1>(kv.first), std::get<2>(kv.first), kv.second.size());
    std::vector<long> diff;
    auto time_stamp = kv.second;
    std::sort(time_stamp.begin(), time_stamp.end(), [](const struct time_stamp& a, const struct time_stamp& b) {
      return a.entry < b.entry;
    });
    for (unsigned i = 0; i < time_stamp.size(); i++) {
      auto& ts = time_stamp[i];
      if (i == 0) {
        printf("work_group_id: %ld, eu_id: %ld, entry: %ld, exit: %ld\n", ts.work_group_id, ts.eu_id, ts.entry,
               ts.exit);
        continue;
      }
      auto& prev_ts = time_stamp[i-1];
      printf("work_group_id: %ld, eu_id: %ld, entry: %ld, exit: %ld, diff: %ld\n", ts.work_group_id, ts.eu_id, ts.entry,
             ts.exit, ts.entry - prev_ts.exit);
      diff.push_back(ts.entry - prev_ts.exit);
    }
    long total_diff = std::accumulate(diff.begin(), diff.end(), 0);
    long avg_diff = total_diff / (time_stamp.size() - 1);
    printf("total_diff: %ld, avg_diff:%ld\n", total_diff, avg_diff);
    printf("avg_overhead:%fus\n", avg_diff / FREQUENCY * 1e6);
  }
}

template void test_threads_spawn_overhead<ATS>(sycl::queue& queue);
template void test_threads_spawn_overhead<PVC>(sycl::queue& queue);