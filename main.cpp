#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <stdint.h>
#include "FunctionTraits.h"

using namespace sycl;

template <
        int vec_size,
        typename func_t,
        typename array_t>
static inline void launch_unrolled_kernel(
        cl::sycl::queue& dpcpp_queue,
        int64_t N,
        const func_t& f,
        array_t data) {
  using traits = function_traits<func_t>;
  using ret_t = typename traits::result_type;
  int thread_num = (N + vec_size - 1) / vec_size;
  int strides[12];
  for (int i = 0; i < sizeof(strides); i++)
    strides[i]= i;

  auto cgf = [&](handler &cgh) {
    auto kfn = [=](cl::sycl::item<1> item_id) {
      int thread_idx = item_id.get_linear_id();
      for (int i = 0; i < 12; i++) {
        data[thread_idx][i] = strides[i];
      }
    };

    cgh.parallel_for(cl::sycl::range</*dim=*/1>(thread_num), kfn);
  };
  dpcpp_queue.submit(cgf);
}

int main() {
  constexpr int size=16;
  assert(0 && "cannot run");

  char* data[3];
// Create queue on implementation-chosen default device
  queue Q;
// Create buffer using host allocated "data" array
  launch_unrolled_kernel<4>(Q,
          size,
          [=](int a, int b){ return a + b;},
          data);
  return 0;
}