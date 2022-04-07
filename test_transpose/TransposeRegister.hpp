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
template <int rows, int grouped_rows, int columns, typename T, int vec_size>
sycl::vec<T, vec_size> Transpose(const sycl::vec<T, vec_size>& data) {
  auto sub_group = sycl::ext::oneapi::experimental::this_sub_group();
  auto sub_group_id = sub_group.get_local_linear_id();
  auto mask = sycl::ext::oneapi::group_ballot(sub_group, true);
  auto sub_group_size = mask.count();

  auto grouped_columns = (sub_group_size + columns - 1) / columns;
  printf(
          "pass1 sub_group_id %d sub_group_size is %d grouped_columns %d\n",
          sub_group_id,
          sub_group_size,
          grouped_columns);

  sycl::vec<T, vec_size> result = data;

  constexpr int row_stride = grouped_rows * rows;
#pragma unroll
  for (int i = 1; i < rows; i++) {
    auto from_column = sub_group_id ^ (grouped_columns);
    if (!(from_column <sub_group_size))
      from_column = sub_group_id;
    printf(
            "i %d sub_group_id %d from column is %d\n",
            i, sub_group_id,
            from_column);
#pragma unroll
    for (int row_base = 0; row_base < vec_size; row_base += row_stride) {
#pragma unroll
      for (int row = 0; row < grouped_rows; row++) {
        auto row_idx =
                (((sub_group_id / grouped_columns) % rows) ^ i) * grouped_rows +
                row_base + row;
#if 1
        result[row_idx] = sub_group.shuffle(result[row_idx], from_column);
#else
        auto swapped_data = sub_group.shuffle_xor(result[row_idx], grouped_columns);
        if (row_idx == 0)
          result[0] = swapped_data;
        if (row_idx == 1)
          result[1] = swapped_data;
        if (row_idx == 2)
          result[2] = swapped_data;
        if (row_idx == 3)
          result[3] = swapped_data;
#endif
      }
    }
  }

  return result;
}

template <int group_cnt,
        typename T,
        int vec_size>
sycl::vec<T, vec_size> VerticsShuffle(const sycl::vec<T, vec_size>& data) {

  auto sub_group = sycl::ext::oneapi::experimental::this_sub_group();
  auto sub_group_local_id = sub_group.get_local_linear_id();
  auto sub_group_size = sub_group.get_local_linear_range();

  auto shuffle_id = sub_group_local_id;
  auto suffle_cnt = sub_group_size / 2;
  suffle_cnt = suffle_cnt / group_cnt;
  shuffle_id = (shuffle_id / group_cnt);
  shuffle_id = (shuffle_id % 2) * suffle_cnt + shuffle_id / 2;
  shuffle_id = shuffle_id * group_cnt + sub_group_local_id % group_cnt;

  return sub_group.shuffle(data, shuffle_id);
}

};

void test_transpose(sycl::queue& queue);