#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <stdint.h>
#include "FunctionTraits.h"

using namespace sycl;

#define ARRAY_SIZE 12

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
  int strides[ARRAY_SIZE];
  for (int i = 0; i < sizeof(strides); i++)
    strides[i]= i;

  char* data_ptr[ARRAY_SIZE];
  for(int i =0; i < ARRAY_SIZE; i++)
    data_ptr[i] = data[i];

  auto cgf = [&](handler &cgh) {
    auto kfn = [=](cl::sycl::item<1> item_id) {
      int thread_idx = item_id.get_linear_id();
      for (int i = 0; i < ARRAY_SIZE; i++) {
        auto ptr = data_ptr[i];
        ptr[thread_idx] = strides[i];
      }
    };

    cgh.parallel_for(cl::sycl::range</*dim=*/1>(thread_num), kfn);
  };
  dpcpp_queue.submit(cgf);
}

int main() {
  constexpr int size=16;
  assert(0 && "cannot run");

  char* data[ARRAY_SIZE];
// Create queue on implementation-chosen default device
  queue Q;
// Create buffer using host allocated "data" array
  launch_unrolled_kernel<4>(Q,
          size,
          [=](int a, int b){ return a + b;},
          data);
  return 0;
}