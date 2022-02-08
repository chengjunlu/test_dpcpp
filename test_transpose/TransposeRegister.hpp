//
// Created by john on 2022/1/26.
//
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <stdint.h>
#include "TransposeUtils.h"

namespace TransposeRegister{

using namespace sycl;

// Assume the sycl::vec in vertica and the SIMD lane in horizontal.
template <
        int rows,
        int grouped_rows,
        int grouped_columns,
        typename T,
        int vec_size>
void Transpose(sycl::vec<T, vec_size>& data) {
  auto sub_group = sycl::ext::oneapi::experimental::this_sub_group();
  auto sub_group_id = sub_group.get_local_linear_id();
  auto sub_group_size = sub_group.get_local_linear_range();
  if (sub_group_size <= grouped_columns)
    return;


  DPCPP_K_PRINT("sub group %d sub_group_size!!! %d \n ",sub_group_id, sub_group_size);
  constexpr int row_stride = grouped_rows * rows;
#pragma unroll
  for (int i = 1; i < rows; i++) {
    auto tgt_idx = sub_group_id ^ (i * grouped_columns);
#pragma unroll
    for (int row_base = 0; row_base < vec_size; row_base += row_stride) {
#pragma unroll
      for (int row = 0; row < grouped_rows; row++) {
        auto row_idx =
                (((sub_group_id / grouped_columns) % rows) ^ i) * grouped_rows +
                row_base + row;
#if 1

//        DPCPP_K_PRINT("sub group %d send data %f -> %d \n ", sub_group_id,
//                      data[row_idx],
//                      tgt_idx);

        data[row_idx] = sub_group.shuffle(data[row_idx], tgt_idx);

//        DPCPP_K_PRINT("sub group %d recv data %f \n ", sub_group_id,
//                      data[row_idx]);
#else
        auto swapped_data = sub_group.shuffle(data[row_idx], tgt_idx);
        if (row_idx == 0)
          data[0] = swapped_data;
        if (row_idx == 1)
          data[1] = swapped_data;
        if (row_idx == 2)
          data[2] = swapped_data;
        if (row_idx == 3)
          data[3] = swapped_data;
#endif
      }
    }
  }
}

}

void test_transpose(sycl::queue& queue);