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

namespace flash {

using namespace cute;

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_k64_mla_32x16_4waves(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, const int num_n_splits) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;
    const int warp_idx = tidx / 64;
    const int lane_idx = tidx % 64;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kBlockKSmem = Kernel_traits::kBlockKSmem;
    constexpr int kAtomLayoutMS = Kernel_traits::kAtomLayoutMS;
    constexpr int kAtomLayoutMO = Kernel_traits::kAtomLayoutMO;
    constexpr int Num_Stages = Kernel_traits::Num_Stages;

    static_assert(kBlockKSmem == 64);

    using GmemTiledCopyO = std::conditional_t<
        !Split,
        typename Kernel_traits::GmemTiledCopyO,
        typename Kernel_traits::GmemTiledCopyOaccum
    >;
    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
    // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_blocks_per_split = ((binfo.actual_seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
    const int n_block_min = !Is_local
        ? n_split_idx * n_blocks_per_split
        : std::max(n_split_idx * n_blocks_per_split, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q / params.ngroups + params.window_size_right, kBlockN));
    }

    if (n_block_min >= n_block_max) {  // This also covers the case where n_block_max <= 0
        // We exit early and write 0 to gOaccum and -inf to gLSEaccum.
        // Otherwise we might read OOB elements from gK and gV,
        // or get wrong results when we combine gOaccum from different blocks.
        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
            + m_block * kBlockM) * params.d_v;
        const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                      Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                     make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                      Shape<Int<kBlockM>>{}, Stride<_1>{});

        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        clear(tOrOaccum);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d_v; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgOaccum); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSEaccum(row) = Split ? -INFINITY : INFINITY; }
        }
        return;
    }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).
    const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
        + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    // We move K and V to the last block.
    const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
    const int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
    const int block_table_idx = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN / params.page_block_size;
    const int block_table_offset = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN - block_table_idx * params.page_block_size;
    const index_t row_offset_k = block_table == nullptr
        ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
        : (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = block_table == nullptr
        ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
          + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
        : (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDimV>>{},
                            make_stride(params.v_row_stride, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutVtNoSwizzle{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyB128 gmem_tiled_copy_Q;
    auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);

    typename Kernel_traits::GmemTiledCopyB64 gmem_tiled_copy_KV;
    auto gmem_thr_copy_KV = gmem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tKgK = gmem_thr_copy_KV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_KV.partition_D(sK);
    // wave0 and wave2 compute the same S, wave1 and wave3 compute the same S
    int tidx_mma_s = tidx & 0x7F;
    typename Kernel_traits::TiledMmaS tiled_mma_s;
    auto thr_mma_s = tiled_mma_s.get_thread_slice(tidx_mma_s);
    Tensor tSrQ  = thr_mma_s.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma_s.partition_fragment_B(sK(_, _, 0));                           // (MMA,MMA_N,MMA_K)
    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    // Tensor tOrVt  = thr_mma_o.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)
    Tensor tOrVt = make_tensor<Element>(Shape<_4, Shape<_4, _4>, _1>{});

    Tensor acc_o = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomB64{}, tiled_mma_s);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx_mma_s);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomB64{}, tiled_mma_s);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx_mma_s);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    int warp_offset = warp_idx / kAtomLayoutMO * 16 * 64;
    int thread_offset = lane_idx / 16 * 4 * 64;
    Element *Vtsmem_ptr_lds = reinterpret_cast<Element *>(sVt.data().get()) + warp_offset + thread_offset;
    Tensor tOsVt = make_tensor(make_smem_ptr(Vtsmem_ptr_lds), make_layout(Shape<_4, _4, Int<Num_Stages>>{},                  // MMA  MMA_N  NUM_STAGES
                                                                          Stride<_1, Int<16*128>, Int<kBlockN*kHeadDim>>{}));

    // PREDICATES

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_KV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)


    // Prologue


    // Read Q from gmem to smem, optionally apply rotary embedding.
    Tensor tQrQ = make_fragment_like(tQgQ);
    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    flash::copy_b128<Is_even_MN, Is_even_K>(tQgQ, tQrQ, tQcQ, params.d, binfo.actual_seqlen_q - m_block * kBlockM);
    cute::copy(tQrQ, tQsQ);

    if constexpr (Kernel_traits::Is_Q_in_regs) {
        flash::sync_threads();
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ);
        flash::sync_threads();
    }


    int n_block = n_block_max - 1;
    int Ksmem_read_index = 0;
    int Ksmem_write_index = 0;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor tKrK = make_fragment_like(tKgK);
    flash::copy_b64_page_one<Kernel_traits, Is_even_MN, Is_even_K>(gK, tKgK, tKrK, tKVcKV, params.d, n_block,
                                                                        block_table, params.k_batch_stride, params.k_row_stride, params.page_block_size, binfo.actual_seqlen_k - n_block * kBlockN);


    // flash::cp_async_wait<0>();
    // __syncthreads();
    // if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) { print(tKsK); }
    // __syncthreads();

    clear(acc_o);

    flash::Softmax<size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.ngroups, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        cute::copy(tKrK, tKsK(_, _, _, Ksmem_write_index));
        Ksmem_write_index ^= 1;
        clear(acc_s);

        flash::sync_threads();
        if (n_block > n_block_min) {
            flash::copy_b64_page_one<Kernel_traits, /*Is_even_MN=*/true, Is_even_K>(gK, tKgK, tKrK, tKVcKV, params.d, n_block - 1,
                                                                                block_table, params.k_batch_stride, params.k_row_stride, params.page_block_size);
        }

        flash::gemm_opt</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, Ksmem_read_index), tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) % kAtomLayoutMS * 16 + (tidx & 0xf), kAtomLayoutMS * 16
        );

        // We have key_padding_mask so we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local || !Is_even_MN, true, true>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || !Is_even_MN, true, true>(acc_s, acc_o, params.scale_softmax_log2);
        // if (cute::thread0()) { print(scores_max); print(scores_sum); print(scores); }

        // Convert acc_s from fp32 to fp16/bf16
        //Tensor rP = flash::convert_type<Element>(acc_s);
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        //Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        lds4x4_with_swizzle424(tOsVt(_, _, Ksmem_read_index), tOrVt);
        CUTE_STATIC_ASSERT_V(size<2>(tOrVt) == _1{}); // only support MMA_K = 1
        Tensor tOrVt_permute_view = make_tensor(tOrVt.data(), make_layout(make_shape(size<0>(tOrVt), size<1, 0>(tOrVt), size<1, 1>(tOrVt))));
        permute_4x4_b16(tOrVt_permute_view);
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());
        flash::gemm_rr(acc_o, tOrP, tOrVt, tiled_mma_o);
        Ksmem_read_index ^= 1;

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        cute::copy(tKrK, tKsK(_, _, _, Ksmem_write_index));
        Ksmem_write_index ^= 1;
        clear(acc_s);
        flash::sync_threads();
        if (n_block > n_block_min) {
            // Advance gK
            flash::copy_b64_page_one<Kernel_traits, /*Is_even_MN=*/true, Is_even_K>(gK, tKgK, tKrK, tKVcKV, params.d, n_block - 1,
                                                                                block_table, params.k_batch_stride, params.k_row_stride, params.page_block_size);
        }

        flash::gemm_opt</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK(_, _, _, Ksmem_read_index), tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );


        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) % kAtomLayoutMS * 16 + (tidx & 0xf), kAtomLayoutMS * 16
        );
        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || !Is_even_MN, true, true>(acc_s, acc_o, params.scale_softmax_log2);

        //Tensor rP = flash::convert_type<Element>(acc_s);
        CONVERT_TENSOR_TYPE(ElementAccum, Element, acc_s, rP)
        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        //Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        lds4x4_with_swizzle424(tOsVt(_, _, Ksmem_read_index), tOrVt);
        CUTE_STATIC_ASSERT_V(size<2>(tOrVt) == _1{}); // only support MMA_K = 1
        Tensor tOrVt_permute_view = make_tensor(tOrVt.data(), make_layout(make_shape(size<0>(tOrVt), size<1, 0>(tOrVt), size<1, 1>(tOrVt))));
        permute_4x4_b16(tOrVt_permute_view);
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());

        flash::gemm_rr(acc_o, tOrP, tOrVt, tiled_mma_o);
        Ksmem_read_index ^= 1;
    }

    // Epilogue


    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, /*Return_lse*/true, Split>(acc_o, params.scale_softmax);
    Tensor acc_o_view = make_tensor(acc_o.data(), make_layout(Shape<_4, Shape<_4, _4>>{},
                                                                Stride<_1, Shape<_4, _16>>{}));
    Tensor acc_o_copy = make_fragment_like(acc_o_view);
    #pragma unroll
    for (int k = 0; k < size<1, 1>(acc_o_view); k++) {
        #pragma unroll
        for (int idx = 0; idx < 16; idx++) {
            int row = idx / 4;
            int col = idx % 4;
            acc_o_copy(row, make_coord(col, k)) = acc_o_view(col, make_coord(row, k));
        }
    }
    // if (cute::thread0()) { print(lse); }
    if constexpr (!Split) {
        // use smem for O (mtreg->smem->mtreg->global)
        Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
        // Partition sO to match the accumulator partitioning
        using SmemTiledCopyO = typename Kernel_traits::SmemCopyAtomO;
        CONVERT_TENSOR_TYPE(ElementAccum, ElementO, acc_o_copy, rO)
        int warp_offset = warp_idx * 16 * 64;
        int thread_offset = lane_idx % 16 * 64 + lane_idx / 16 * 16;
        Element *Osmem_ptr_sts = reinterpret_cast<ElementO *>(smem_) + warp_offset + thread_offset;
        Tensor tOsO = make_tensor(make_smem_ptr(Osmem_ptr_sts), make_layout(Shape<_16, _4>{},
                                                                          Stride<_1, Int<16*64*kNWarps>>{}));
        Tensor tOrO = make_tensor(rO.data(), make_layout(Shape<_16, _4>{},
                                                                Stride<_1, _16>{}));


        // sOaccum is larger than sQ, so we need to syncthreads here
        // TODO: allocate enough smem for sOaccum
        if constexpr (Kernel_traits::Share_Q_K_smem) { flash::sync_threads(); }

        cute::copy(tOrO, tOsO);

        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.o_ptr) + (row_offset_o)),
                                    Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                    make_stride(params.o_row_stride, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lseaccum),
                                    Shape<Int<kBlockM>>{}, Stride<_1>{});
        // if (tidx == 0) { printf("row_offset_o = %d, bidh = %d, gOaccum = %p\n", row_offset_o, bidh, gOaccum.data()); }

        GmemTiledCopyO gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

        flash::sync_threads();

        Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
        cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma_o.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        static_assert(decltype(size<0>(taccOcO))::value == 4);
        // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
        if (get<1>(taccOcO_row(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO_row(mi));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
            }
        }

        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy_reg_to_global<Is_even_MN, Is_even_K>(
            tOrOaccum, tOgOaccum, tOcO, params.d_v, binfo.actual_seqlen_q - m_block * kBlockM
        );
    } else {
        // don't use smem for O (mtreg->global)
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
                                            + m_block * kBlockM) * params.d_v;
        const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

        // Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.oaccum_ptr) + row_offset_oaccum),
        //                             Shape<Int<kBlockM>, Int<kHeadDimV>>{},
        //                             make_stride(kHeadDimV, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                    Shape<Int<kBlockM>>{}, Stride<_1>{});
        // if (tidx == 0) { printf("row_offset_o = %d, bidh = %d, gOaccum = %p\n", row_offset_o, bidh, gOaccum.data()); }
        Tensor taccOrOaccum = make_tensor(acc_o_copy.data(), make_layout(Shape<_16, _4>{},
                                                                    Stride<_1, _16>{}));
        int warp_offset = warp_idx / kAtomLayoutMO * 64 + warp_idx % kAtomLayoutMO * 16 * kHeadDimV;
        int thread_offset = lane_idx % 16 * kHeadDimV + lane_idx / 16 * 16;
        ElementO *Osmem_ptr_stg = reinterpret_cast<ElementO *>(params.oaccum_ptr) + row_offset_oaccum + warp_offset + thread_offset;
        Tensor taccOgOaccum = make_tensor(make_gmem_ptr(Osmem_ptr_stg), make_layout(Shape<_16, _4>{},
                                                                          Stride<_1, Int<128>>{}));
        // Tensor taccOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);



        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma_o.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        static_assert(decltype(size<0>(taccOcO))::value == 4);
        // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_4>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
        if (get<1>(taccOcO_row(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO_row(mi));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
            }
        }

        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy_reg_to_global4x4fp32<Kernel_traits, Is_even_MN, Is_even_K>(
            taccOrOaccum, taccOgOaccum, params.d_v, binfo.actual_seqlen_q - m_block * kBlockM
        );
    }
}

} // namespace flash
