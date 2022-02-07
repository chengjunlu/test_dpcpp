//
// Created by john on 2022/1/26.
//
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <stdint.h>
#include "TransposeUtils.h"

namespace TransposeShareLocalMemroy{
  using namespace sycl;

class TransposeSLM {
public:
  TransposeSLM(int work_group_size, sycl::handler& cgh)
          : local_buffer(work_group_size * max_vec_size, cgh) {
  };

  template <typename T, int vec_size>
  void transpose(sycl::vec<T, vec_size>& data) {
    auto sub_group = sycl::ext::oneapi::experimental::this_sub_group();
    auto sub_group_size = sub_group.get_local_linear_range();
    auto sub_group_id = sub_group.get_group_linear_id();
    auto sub_group_local_id = sub_group.get_local_linear_id();
#ifdef PASS_WITH_PRINTF_IN_KERNEL
    DPCPP_K_PRINT(
        "before transpose sub_group_local_id %d, data[0] %f, data[1] %f, data[2] %f, data[3] %f\n ",
        sub_group_local_id,
        data[0],
        data[1],
        data[2],
        data[3]);
#endif
    auto offset =
            (sub_group_size * sizeof(sycl::vec<T, vec_size>)) * sub_group_id;
    auto store_base = local_buffer.get_pointer() + offset;
    T* store_ptr = reinterpret_cast<T*>(store_base.get());
    sub_group.store<vec_size>(sycl::local_ptr<T>(store_ptr), data);

    // add the barrier to sync the execution in the sub-group.
    group_barrier(sub_group);

    auto load_base = store_base.get();
    sycl::vec<T, vec_size>* load_ptr =
            reinterpret_cast<sycl::vec<T, vec_size>*>(load_base) +
            sub_group_local_id;
    data = *(load_ptr);

#ifdef PASS_WITH_PRINTF_IN_KERNEL
    DPCPP_K_PRINT(
        "sub_group_id %d, sub_group_local_id %d, sub_group_size %d offset %d, store_ptr %p, load_ptr %p \n ",
        sub_group_id,
        sub_group_local_id,
        sub_group_size,
        offset,
        store_ptr,
        load_ptr);

    DPCPP_K_PRINT(
        "after transpose sub_group_local_id %d, data[0] %f, data[1] %f, data[2] %f, data[3] %f\n ",
        sub_group_local_id,
        data[0],
        data[1],
        data[2],
        data[3]);
#endif
  }

private:
  constexpr static int max_vec_size = 16;
  sycl::accessor<uint32_t, 1, sycl::access::mode::read_write, sycl::access::target::local>
          local_buffer;
};
}

void test_transpose(sycl::queue& queue);