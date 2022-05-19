//
// Created by john on 2022/1/27.
//
#include "SlowKernel.hpp"


#define NUM_WORK_ITEM 16
#define GROUP_CNT 2

void test_slow_kernel(sycl::queue& queue, uint32_t loop_cnt){

  float* data_ptr = (float*)malloc_device(sizeof(float)* NUM_WORK_ITEM, queue);
  auto numel = NUM_WORK_ITEM;
  auto thread_num = NUM_WORK_ITEM;
  auto work_group_size = NUM_WORK_ITEM;

  auto cgf = [&](sycl::handler & cgh) {
      cgh.parallel_for(
              sycl::nd_range<1>(thread_num, work_group_size),
              [=](sycl::nd_item<1> id) {

                size_t thread_idx = id.get_global_linear_id();
                for(int i=0; i<loop_cnt; i++) {
                  for(int j=0; j<loop_cnt; j++) {
                    data_ptr[thread_idx] = i + j;
                  }
                }
              });
  };

  auto evt=queue.submit(cgf);
  evt.wait();
  float* out_host = (float*)malloc(sizeof(float)* NUM_WORK_ITEM);
  auto e = queue.memcpy(out_host, data_ptr, sizeof(float)* NUM_WORK_ITEM);

  e.wait();

  for(int i =0; i < NUM_WORK_ITEM; i++ ) {
    std::cout << " " << out_host[i];
    std::cout << std::endl;
  }
  queue.throw_asynchronous();
}