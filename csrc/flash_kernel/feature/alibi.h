// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)
#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <mctlass/mctlass.h>
#include <mctlass/array.h>

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_causal>
struct Alibi {

    const float alibi_slope;
    const int max_seqlen_k, max_seqlen_q;

    __forceinline__ __device__ Alibi(const float alibi_slope, const int max_seqlen_k, const int max_seqlen_q)
        : alibi_slope(alibi_slope)
        , max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q) {
    };


    template <typename Engine, typename Layout>
    __forceinline__ __device__ void apply_alibi(Tensor<Engine, Layout> &tensor,
                                      const int col_idx_offset_,
                                      const int row_idx_offset,
                                      const int warp_row_stride,
                                      const int warp_col_stride = 16) {
        // tensor has shape (nrow=(1, MMA_M), ncol=(4, MMA_N))
        static_assert(Layout::rank == 2, "Only support 2D Tensor");
        static_assert(decltype(size<0, 0>(tensor))::value == 1);
        static_assert(decltype(size<1, 0>(tensor))::value == 4);
        const int col_idx_offset = col_idx_offset_ + ((__lane_id() >> 4) << 2);
        if constexpr (Is_causal) {  // Simpler, we add the same bias vector to all rows
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * warp_col_stride;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    #pragma unroll
                    for (int mi = 0; mi < size<0>(tensor); ++mi) {
                        tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                    }
                }
            }
        } else {  // Bias depends on both row_idx and col_idx
            #pragma unroll
            for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                const int row_idx = row_idx_offset + mi * warp_row_stride;
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * warp_col_stride;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        tensor(make_coord(0, mi), make_coord(j, nj)) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                    }
                }
            }
        }
    }

};

}  // namespace flash
