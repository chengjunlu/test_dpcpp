//
// Created by john on 2022/1/26.
//
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <stdint.h>

#define X 64*16*16
#define Y 32
#define VECSIZE 16
#define SUBGROUPSIZE 16

namespace TransposeRegister{

using namespace sycl;

// Assume the sycl::vec in vertica and the SIMD lane in horizontal.
template <
        int rows,
        int grouped_rows,
        int grouped_columns,
        typename T,
        int vec_size>
void TransposeRegister(sycl::vec<T, vec_size>& data) {
  auto sub_group = sycl::ext::oneapi::experimental::this_sub_group();
  auto sub_group_id = sub_group.get_local_linear_id();

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
        auto swap_ed = sub_group.shuffle_xor(data[row_idx], tgt_idx);
#if 0
        data[row_idx] = sub_group.shuffle(data[row_idx], tgt_idx);
#else
        if (row_idx == 0)
          data[0] = swap_ed;
        if (row_idx == 1)
          data[1] = swap_ed;
        if (row_idx == 2)
          data[2] = swap_ed;
        if (row_idx == 3)
          data[3] = swap_ed;
#endif
      }
    }
  }
}

template <typename T>
sycl::event ShuffleTranspose(sycl::queue &q, T *in, T *out, int KX, int KY) {
  constexpr int NUMSUBGROUPX = 64;
  constexpr int NUMSUBGROUPY = 1;
  if (KX % (SUBGROUPSIZE * NUMSUBGROUPX) != 0)
    printf(
            "SubgroupTransposeRowToCol only support when KX is a mutiple of SUBGROUPSIZE * "
            "NUMSUBGROUPX\n");
  if (KY % (VECSIZE * NUMSUBGROUPY) != 0)
    printf(
            "SubgroupTransposeRowToCol only support when KY is a mutiple of VECSIZE * NUMSUBGROUPY\n");

  auto event = q.submit([&](sycl::handler &h) {
      h.parallel_for<class ShuffleTranspose>(
              sycl::nd_range<2>(sycl::range<2> {size_t(KX), size_t(KY / VECSIZE)},
                                sycl::range<2> {SUBGROUPSIZE * NUMSUBGROUPX, NUMSUBGROUPY}),
              [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(SUBGROUPSIZE)]] {
                  auto global_x = item.get_group(0) * SUBGROUPSIZE * NUMSUBGROUPX;
                  auto global_y = item.get_group(1) * VECSIZE * NUMSUBGROUPY;

                  auto sg = item.get_sub_group();
                  auto sg_id = sg.get_group_linear_id();
                  auto sg_id_x = sg_id / NUMSUBGROUPY;
                  auto sg_id_y = sg_id - sg_id_x * NUMSUBGROUPY;
                  auto local_x = sg_id_x * SUBGROUPSIZE;
                  auto local_y = sg_id_y * VECSIZE;

                  uint32_t lid_in_sg = sg.get_local_linear_id();

                  auto in_offset = (global_x + local_x) * KY + global_y + local_y;
                  auto out_offset = (global_y + local_y) * KX + global_x + local_x;

                  sycl::vec<T, VECSIZE> data;
                  data = *(reinterpret_cast<sycl::vec<T, VECSIZE> *>(in + in_offset + lid_in_sg * KY));

                  TransposeRegister<VECSIZE, 1, 1>(data);

                  *(reinterpret_cast<sycl::vec<T, VECSIZE> *>(out + out_offset + lid_in_sg * KX)) = data;
              });
  });
  return event;
}

}