//
// Created by john on 2022/1/27.
//
#include "SubgroupBallot.hpp"


#define NUM_WORK_ITEM 16
#define GROUP_CNT 2

void test_subgroup_ballot(sycl::queue& queue){

  float* data_ptr = (float*)malloc_device(sizeof(float)* NUM_WORK_ITEM, queue);
  auto numel = NUM_WORK_ITEM;
  auto thread_num = NUM_WORK_ITEM;
  auto work_group_size = NUM_WORK_ITEM;

  auto cgf = [&](sycl::handler & cgh) {
      cgh.parallel_for(
              sycl::nd_range<1>(thread_num, work_group_size),
              [=](sycl::nd_item<1> id) {
//                if (id < 4)
//                  flat = ture
//                  else
//                    flag= false
                  printf("work item %zu here\n", id.get_global_linear_id());
                if (id.get_local_linear_id() < 5) {
                  auto sub_group = sycl::ext::oneapi::experimental::this_sub_group();
                  auto mask = sycl::ext::oneapi::group_ballot(sub_group, true);
                  auto count = mask.count();
                  printf("pass1 work item %zu get vote count is %d\n", id.get_global_linear_id(), count);
                } else {
//                  auto sub_group = sycl::ext::oneapi::experimental::this_sub_group();
//                  auto mask = sycl::ext::oneapi::group_ballot(sub_group, false);
//                  auto count = mask.count();
//                  printf("pass2 work item %zu get vote count is %d\n", id.get_global_linear_id(), count);
                }

              });
  };

  queue.submit(cgf);
  float* out_host = (float*)malloc(sizeof(float)* NUM_WORK_ITEM);
  auto e = queue.memcpy(out_host, data_ptr, sizeof(float)* NUM_WORK_ITEM);

  e.wait();

//  for(int i =0; i < NUM_WORK_ITEM; i++ ) {
//    std::cout << " " << out_host[i];
//  }
}