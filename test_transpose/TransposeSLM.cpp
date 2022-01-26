//
// Created by john on 2022/1/26.
//
#include "TransposeSLM.hpp"

#define VEC_SIZE 4
#define NUM_WORK_ITEM 16

using namespace TransposeShareLocalMemroy;

struct TensorStoreWithOutCast {
  template <typename scalar_t>
  void store(scalar_t&& value, void* base_ptr, uint32_t offset) {
    *(reinterpret_cast<std::decay_t<scalar_t>*>(base_ptr) + offset) = value;
  }
};

struct VectorizedPolicy {
  void* out_data;
  TensorStoreWithOutCast storer;
  TransposeSLM transpose;

  VectorizedPolicy(
          void* out_data,
          TransposeSLM transpose)
          : out_data(out_data),
            storer(),
            transpose(transpose){};

  template <typename scalar_t, int unroll_size>
  void store(sycl::vec<scalar_t, unroll_size>& value, uint32_t linear_idx) {
    transpose.template transpose<scalar_t, unroll_size>(value);
    storer.store(value, out_data, (linear_idx + unroll_size * 0) / unroll_size);
  };
};


template <int unroll_size, typename scalar_t, typename policy_t>
void distribution_kernel(
        sycl::nd_item<1> item_id,
        int rounded_size,
        int thread_num,
        policy_t p) {
  size_t thread_idx = item_id.get_global_linear_id();

  for (int linear_idx = thread_idx * unroll_size; linear_idx < rounded_size;
       linear_idx += thread_num * unroll_size) {
    sycl::vec<scalar_t, unroll_size> result;
#pragma unroll
    for (int i = 0; i < unroll_size; i++) {
      result[i] = static_cast<scalar_t>(thread_idx);
    }

    p.store(result, linear_idx);
  }
}

void test_transpose_slm(sycl::queue& queue){

  float* data_ptr = (float*)malloc_device(sizeof(float)* NUM_WORK_ITEM * VEC_SIZE, queue);
  auto numel = NUM_WORK_ITEM*VEC_SIZE;
  auto vec_size = VEC_SIZE;
  auto thread_num = NUM_WORK_ITEM*VEC_SIZE;
  auto work_group_size = NUM_WORK_ITEM*VEC_SIZE;

  auto cgf = [&](sycl::handler & cgh) {
    VectorizedPolicy p(
            data_ptr, {work_group_size, cgh});
    cgh.parallel_for(
            sycl::nd_range<1>(thread_num, work_group_size),
            [=](sycl::nd_item<1> id) {
                distribution_kernel<VEC_SIZE, float>(
                        id, numel, thread_num, p);
            });
  };

  queue.submit(cgf);
  float* out_host = (float*)malloc(sizeof(float)* NUM_WORK_ITEM * VEC_SIZE);
  auto e = queue.memcpy(out_host, data_ptr, sizeof(float)* NUM_WORK_ITEM * VEC_SIZE);

  e.wait();

  for(int i =0; i < NUM_WORK_ITEM; i++ ) {
    for(int j =0; j < VEC_SIZE; j++ ) {
      std::cout << " " << out_host[i*VEC_SIZE + j];
    }
    std::cout << std::endl;
  }
}
