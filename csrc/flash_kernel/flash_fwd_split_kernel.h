// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)

#pragma once

#include <cute/algorithm/copy.hpp>

#include <mctlass/mctlass.h>
#include <mctlass/array.h>
#include <mctlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "rotary.h"
#include "attn_mask.h"
#include "flash_fwd_split_kernel_k64_32x16_4waves.h"
#include "flash_fwd_split_kernel_k64_64x16_8waves.h"

namespace flash {

using namespace cute;

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, typename Params>
__forceinline__ __device__ void compute_attn_splitkv(const Params &params, const int m_block_max) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
    // The block index for the head.
    const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
    const int n_split_idx = Split ? blockIdx.y : 0;
    const int num_n_splits = Split ? gridDim.y : 1;
    if constexpr (Kernel_traits::kBlockM == 32 && Kernel_traits::kBlockN == 16 && Kernel_traits::kNWarps == 4) {
        compute_attn_1rowblock_splitkv_k64_mla_32x16_4waves<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split>(
                params, bidb, bidh, m_block, n_split_idx, num_n_splits);
    } else if constexpr (Kernel_traits::kBlockM == 64 && Kernel_traits::kBlockN == 16 && Kernel_traits::kNWarps == 8) {
        compute_attn_1rowblock_splitkv_k64_mla_64x16_8waves<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split>(
                params, bidb, bidh, m_block, n_split_idx, num_n_splits);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K, typename Params>
__forceinline__ __device__ void combine_attn_seqk_parallel(const Params &params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    constexpr int kMaxSplits = 1 << Log_max_splits;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNThreads = 256;/*Kernel_traits::kNThreads*/;

    static_assert(kMaxSplits <= 128, "kMaxSplits must be <= 128");
    static_assert(kBlockM == 4 || kBlockM == 8 || kBlockM == 16 || kBlockM == 32, "kBlockM must be 4, 8, 16 or 32");
    static_assert(kNThreads == 128 || kNThreads == 256, "We assume that each block has 128 or 256 threads");

    // Shared memory.
    // kBlockM + 1 instead of kBlockM to reduce bank conflicts.
    __shared__ ElementAccum sLSE[kMaxSplits][kBlockM + 1];

    // The thread and block index.
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    const index_t lse_size = params.b * params.h * params.seqlen_q;

    const index_t row_offset_lse = bidx * kBlockM;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lse),
                                   Shape<Int<kMaxSplits>, Int<kBlockM>>{},
                                   make_stride(lse_size, _1{}));
    // LSE format is different depending on params.unpadded_lse and params.seqlenq_ngroups_swapped, see comment in get_lse_tile.
    // This tensor's layout maps row_offset_lse to {bidb, bidh, lse_size}.
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    // This layout maps row_offset_lse to {bidh, lse_size, bidb} or {bidh, bidb, lse_size}.
    Layout flat_layout = make_layout(lse_size);
    Layout orig_layout = make_layout(make_shape(params.seqlen_q, params.h, params.b));
    auto transposed_stride = make_stride(params.b, params.seqlen_q * params.b, params.seqlen_q / params.ngroups);
    Layout remapped_layout = make_layout(make_shape(params.seqlen_q, params.h, params.b), transposed_stride);
    Layout final_layout = cute::composition(remapped_layout, cute::composition(orig_layout, flat_layout));

    Tensor gLSE_unpadded = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr)), final_layout);

    constexpr int kNLsePerThread = (kMaxSplits * kBlockM + kNThreads - 1) / kNThreads;

    // Read the LSE values from gmem and store them in shared memory, then tranpose them.
    constexpr int kRowsPerLoadLSE = kNThreads / kBlockM;
    typedef __NATIVE_VECTOR__(1, ElementAccum) B32Type;
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadLSE + tidx / kBlockM;
        const int col = tidx % kBlockM;
        ElementAccum lse = (row < params.num_splits && col < lse_size - bidx * kBlockM) ? gLSEaccum(row, col) : -INFINITY;
        if (row < kMaxSplits) { sLSE[row][col] = lse; }
    }

    flash::sync_threads();
    Tensor lse_accum = make_tensor<ElementAccum>(Shape<Int<kNLsePerThread>>{});
    constexpr int kRowsPerLoadTranspose = std::min(kRowsPerLoadLSE, kMaxSplits);
    // To make sure that kMaxSplits is within 1 warp: we decide how many elements within kMaxSplits
    // each thread should hold. If kMaxSplits = 16, then each thread holds 2 elements (128 threads,
    // kBlockM rows, so each time we load we can load 128 / kBlockM rows).
    // constexpr int kThreadsPerSplit = kMaxSplits / kRowsPerLoadTranspose;
    // static_assert(kThreadsPerSplit <= 32);
    //static_assert(kRowsPerLoadTranspose <= 32);
    static_assert(kRowsPerLoadTranspose <= 64);
    static_assert(kNLsePerThread * kRowsPerLoadTranspose <= kMaxSplits);
    const int lse_base_row = tidx % kRowsPerLoadTranspose;
    const int lse_base_col = tidx / kRowsPerLoadTranspose;
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + lse_base_row;
        const int col = lse_base_col;
        lse_accum(l) = (row < kMaxSplits && col < kBlockM) ? sLSE[row][col] : -INFINITY;
        // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse_accum(l)); }
    }

    // Compute the logsumexp of the LSE along the split dimension.
    ElementAccum lse_max = lse_accum(0);
    #pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_max = max(lse_max, lse_accum(l)); }
    MaxOp<float> max_op;
    lse_max = Allreduce<kRowsPerLoadTranspose>::run(lse_max, max_op);
    lse_max = lse_max == -INFINITY ? 0.0f : lse_max;  // In case all local LSEs are -inf
    float lse_sum = __expf(lse_accum(0) - lse_max);
    #pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_sum += __expf(lse_accum(l) - lse_max); }
    SumOp<float> sum_op;
    lse_sum = Allreduce<kRowsPerLoadTranspose>::run(lse_sum, sum_op);
    // For the case where all local lse == -INFINITY, we want to set lse_logsum to INFINITY. Otherwise
    // lse_logsum is log(0.0) = -INFINITY and we get NaN when we do lse_accum(l) - lse_logsum.
    ElementAccum lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : __logf(lse_sum) + lse_max;
    if (tidx % kRowsPerLoadTranspose == 0 && tidx / kRowsPerLoadTranspose < kBlockM) {
        if (params.unpadded_lse) {
            const index_t lse_offset = row_offset_lse + tidx / kRowsPerLoadTranspose;
            if (lse_offset < lse_size) {
                gLSE_unpadded(lse_offset) = lse_logsum;
            }
        } else {
            gLSE(tidx / kRowsPerLoadTranspose) = lse_logsum;
        }
    }
    // Store the scales exp(lse - lse_logsum) in shared memory.
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + lse_base_row;
        const int col = lse_base_col;
        if (row < params.num_splits && col < kBlockM) { sLSE[row][col] = __expf(lse_accum(l) - lse_logsum); }
    }

    const index_t row_offset_oaccum = bidx * kBlockM * params.d_v;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                 Stride<Int<kHeadDimV>, _1>{});
    constexpr int kBlockN = kNThreads / kBlockM;
    using GmemLayoutAtomOaccum = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    clear(tOrO);
    flash::sync_threads();

    typedef __NATIVE_VECTOR__(2, float) Float2;

    // Predicates
    Tensor cOaccum = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});
    // Repeat the partitioning with identity layouts
    Tensor tOcOaccum = gmem_thr_copy_Oaccum.partition_S(cOaccum);
    static_assert(decltype(size<0>(tOrOaccum))::value % 2 == 0);
    // Load Oaccum in then scale and accumulate to O
    for (int split = 0; split < params.num_splits; ++split) {
        flash::copy_b128</*Is_even_MN=*/false, Is_even_K>(
            tOgOaccum, tOrOaccum, tOcOaccum, params.d_v, lse_size - bidx * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOrOaccum); ++m) {
            int row = get<0>(tOcOaccum(0, m, 0));
            ElementAccum lse_scale = sLSE[split][row];
            Float2 lse_scale_vec = {lse_scale, lse_scale};
            #pragma unroll
            for (int k = 0; k < size<2>(tOrOaccum); ++k) {
                #pragma unroll
                for (int i = 0; i < size<0>(tOrOaccum); i += 2) {
                    Float2 x_vec = {tOrOaccum(i, m, k), tOrOaccum(i + 1, m, k)};
                    Float2 y_vec = {tOrO(i, m, k), tOrO(i + 1, m, k)};
                    y_vec = __builtin_mxc_pk_fma_f32(x_vec, lse_scale_vec, y_vec);
                    tOrO(i, m, k) = y_vec[0];
                    tOrO(i + 1, m, k) = y_vec[1];
                }
            }
        }
        tOgOaccum.data() = tOgOaccum.data() + lse_size * params.d_v;
    }

    //Tensor rO = flash::convert_type<Element>(tOrO);
    CONVERT_TENSOR_TYPE(ElementAccum, Element, tOrO, rO)
    const int q_head_offset = params.h * params.seqlen_q;
    // Write to gO
    #pragma unroll
    for (int m = 0; m < size<1>(rO); ++m) {
        const int idx = bidx * kBlockM + get<0>(tOcOaccum(0, m, 0));
        const int batch_idx = idx / q_head_offset;
        const int head_idx = (idx - batch_idx * q_head_offset) / params.seqlen_q;
        // The index to the rows of Q
        const int row = idx - batch_idx * q_head_offset - head_idx * params.seqlen_q;
        auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride
            + head_idx * params.o_head_stride + row * params.o_row_stride;
        #pragma unroll
        for (int k = 0; k < size<2>(rO); ++k) {
            const int col = get<1>(tOcOaccum(0, m, k));
            Tensor gO = make_tensor(make_gmem_ptr(o_ptr + col),
                                    Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
            auto gO_ptr = reinterpret_cast<uint64_t *>(gO.data().ptr_);
            auto rO_ptr = reinterpret_cast<uint64_t *>(rO(_, m, k).data().ptr_);
            __builtin_mxc_stg_b64_predicator(gO_ptr, 0, rO_ptr[0], true, false, false, idx < lse_size && (Is_even_K || col < params.d_v), 1, MACA_ICMP_EQ);
        }
    }
}

} // namespace flash
