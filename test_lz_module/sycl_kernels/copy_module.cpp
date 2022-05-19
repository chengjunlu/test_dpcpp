//
// Created by john on 2022/1/27.
//
#include <CL/sycl.hpp>

class copy_data {};
void test_copy(sycl::queue& queue){
  auto size = 8;
  auto offset = 0;
  auto thread_num = 1;
  int* out_ptr = (int*)malloc_shared(sizeof(int)*size, queue);
  int* in_ptr = (int*)malloc_shared(sizeof(int)*size, queue);

  auto cgf = [&](sycl::handler & cgh) {
      cgh.parallel_for<copy_data>(
              sycl::range<1>(thread_num),
              [=](sycl::item<1> id) {
                auto linear_id = id.get_linear_id();
                if (linear_id == 0) {
                  for (int i = 0; i < size - offset; i++) {
                    out_ptr[i + offset] = in_ptr[i];
                  }
                }
              });
  };

  auto e = queue.submit(cgf);
  e.wait();
}