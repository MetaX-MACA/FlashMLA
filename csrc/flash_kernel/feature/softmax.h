// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <mctlass/numeric_types.h>

#include "philox.cuh"
#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<64>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    reduce_(tensor, sum, sum_op);
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void thread_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    thread_reduce_(tensor, sum, sum_op);
}

template<typename Engine0, typename Layout0>
__device__ __forceinline__ void quadreduce_sum(Tensor<Engine0, Layout0>&sum) {
    SumOp<float> sum_op;
    quad_allreduce_(sum, sum, sum_op);
}

// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    static_assert(decltype(size<1>(tensor))::value % 2 == 0);
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    typedef __NATIVE_VECTOR__(2, float) Float2;
    Float2 scale_vec = {scale, scale};
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        /*#pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            //tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            tensor(mi, ni) = __builtin_exp2f(tensor(mi, ni) * scale - max_scaled);
        }*/
        Float2 max_scale_vec = {-max_scaled, -max_scaled};
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ni += 2) {
            Float2 x_vec = {tensor(mi, ni), tensor(mi, ni + 1)};
            x_vec = __builtin_mxc_pk_fma_f32(x_vec, scale_vec, max_scale_vec);
            tensor(mi, ni) = __builtin_exp2f(x_vec[0]);
            tensor(mi, ni + 1) = __builtin_exp2f(x_vec[1]);
        }
    }
}

// Apply the exp to all the elements.
template <bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void max_scale_exp2_sum(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> &max, Tensor<Engine1, Layout1> &sum, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        MaxOp<float> max_op;
        max(mi) = zero_init ? tensor(mi, 0) : max_op(max(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            max(mi) = max_op(max(mi), tensor(mi, ni));
        }
        max(mi) = Allreduce<4>::run(max(mi), max_op);
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * scale;
        sum(mi) = 0;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            //tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
            tensor(mi, ni) = __builtin_exp2f(tensor(mi, ni) * scale - max_scaled);
            sum(mi) += tensor(mi, ni);
        }
        SumOp<float> sum_op;
        sum(mi) = Allreduce<4>::run(sum(mi), sum_op);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows>
struct Softmax {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;

    __forceinline__ __device__ Softmax() {};

    template<bool Is_first, bool Check_inf=false, bool Syncthreads=false, bool AddVec=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
        MaxOp<float> max_op;
        static_assert(decltype(size<0>(scores))::value == kNRows);
        static_assert(decltype(size<1>(scores))::value % 2 == 0);
        typedef __NATIVE_VECTOR__(2, float) Float2;
        if (Is_first) {
            //flash::template reduce_max</*zero_init=*/true>(scores, row_max);
            flash::template thread_reduce_</*zero_init=*/true>(scores, row_max, max_op);
            //if (Syncthreads) __syncthreads();
            if (Syncthreads) flash::sync_threads();
            flash::template quad_allreduce_(row_max, row_max, max_op);
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            //flash::reduce_sum(scores, row_sum);
            if constexpr(AddVec) {
                #pragma unroll
                for (int mi = 0; mi < size<0>(scores); mi++) {
                    Float2 x_vec = { 0.0f, 0.0f};
                    Float2 scale_vec = {1.0f, 1.0f};
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(scores); ni += 2) {
                        Float2 beta_vec = {scores(mi, ni), scores(mi, ni + 1)};
                        x_vec = __builtin_mxc_pk_fma_f32(x_vec, scale_vec, beta_vec);
                    }
                    row_sum(mi) = x_vec[0] + x_vec[1];
                }
            }
            else {
                SumOp<float> sum_op;
                flash::thread_reduce_</*zero_init=*/true>(scores, row_sum, sum_op);
            }
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            //flash::template reduce_max</*zero_init=*/false>(scores, row_max);
            flash::template thread_reduce_</*zero_init=*/false>(scores, row_max, max_op);
            //if (Syncthreads) __syncthreads();
            if (Syncthreads) flash::sync_threads();
            flash::template quad_allreduce_(row_max, row_max, max_op);
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            static_assert(decltype(size<1>(acc_o_rowcol))::value % 2 == 0);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_scale = __builtin_exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;
                // #pragma unroll
                // for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
                Float2 scale_vec = {scores_scale , scores_scale};
                Float2 beta_vec = {0.0f, 0.0f};
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ni += 2) {
                    Float2 x_vec = {acc_o_rowcol(mi, ni),  acc_o_rowcol(mi, ni + 1)};
                    x_vec = __builtin_mxc_pk_fma_f32(x_vec, scale_vec, beta_vec);
                    acc_o_rowcol(mi, ni) = x_vec[0];
                    acc_o_rowcol(mi, ni + 1) = x_vec[1];
                }
            }
            flash::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            #pragma unroll
            for (int mi = 0; mi < size<0>(scores); mi++) {
                if constexpr(AddVec) {
                    Float2 x_vec = { 0.0f, 0.0f};
                    Float2 scale_vec = {1.0f, 1.0f};
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(scores); ni += 2) {
                        Float2 beta_vec = {scores(mi, ni), scores(mi, ni + 1)};
                        x_vec = __builtin_mxc_pk_fma_f32(x_vec, scale_vec, beta_vec);
                    }
                    row_sum(mi) += x_vec[0] + x_vec[1];
                }
                else {
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(scores); ni++) {
                        row_sum(mi) += scores(mi, ni);
                    }
                }
            }
        }
    };

    template<bool Is_dropout=false, bool Return_lse=true, bool Split=false, typename Tensor0>
    __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0) {
        flash::quadreduce_sum(row_sum);
        TensorT lse = make_fragment_like(row_sum);
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
        static_assert(decltype(size<1>(acc_o_rowcol))::value % 2 == 0);

        typedef __NATIVE_VECTOR__(2, float) Float2;
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            if (Return_lse)
                lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
            float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
            // #pragma unroll
            // for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
            //     acc_o_rowcol(mi,  ni) *= scale;
            // }
            Float2 scale_vec = {scale, scale};
            Float2 beta_vec = {0.0f, 0.0f};
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ni += 2) {
                Float2 x_vec = {acc_o_rowcol(mi, ni), acc_o_rowcol(mi, ni + 1)};
                x_vec = __builtin_mxc_pk_fma_f32(x_vec, scale_vec, beta_vec);
                acc_o_rowcol(mi, ni) = x_vec[0];
                acc_o_rowcol(mi, ni + 1) = x_vec[1];
            }
        }
        return lse;
    };
};

}  // namespace flash
