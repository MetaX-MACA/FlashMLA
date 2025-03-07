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

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Is_page_attn, typename Params>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_k64_mla_V1x8(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, const int num_n_splits) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kBlockKSmem = Kernel_traits::kBlockKSmem;
    constexpr int kBlockKGmem = Kernel_traits::UseWarpsNx1 ? Kernel_traits::kBlockKSmem : 128;
    constexpr int kAtomLayoutMS = Kernel_traits::kAtomLayoutMS;
    constexpr int kAtomLayoutMO = Kernel_traits::kAtomLayoutMO;

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
    //Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutK{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutVtNoSwizzle{});
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
    Tensor tVgV = gmem_thr_copy_KV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_KV.partition_D(sV);
    Tensor tVrV = make_fragment_like(tVgV);
    // wave0 and wave2 compute the same S, wave1 and wave3 compute the same S
    int tidx_mma_s = tidx & 0x7F;
    typename Kernel_traits::TiledMmaS tiled_mma_s;
    auto thr_mma_s = tiled_mma_s.get_thread_slice(tidx_mma_s);
    Tensor tSrQ  = thr_mma_s.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma_s.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tOrVt  = thr_mma_o.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

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
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVtNoSwizzle);

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
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    Tensor tKrK = make_fragment_like(tKgK);
    if constexpr (!Is_page_attn) {
        flash::copy_b64<Is_even_MN, Is_even_K>(tKgK, tKrK, tKVcKV, params.d, binfo.actual_seqlen_k - n_block * kBlockN);
    } else {
        flash::copy_b64_page_one<Kernel_traits, Is_even_MN, Is_even_K>(gK, tKgK, tKrK, tKVcKV, params.d, n_block,
                                                                            block_table, params.k_batch_stride, params.k_row_stride, params.page_block_size, binfo.actual_seqlen_k - n_block * kBlockN);
    }

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
        cute::copy(tKrK, tKsK);
        clear(acc_s);

        // Advance gV
        if (masking_step > 0) {
            if constexpr (!Is_page_attn) {
                tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
                flash::copy_b64</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tKVcKV, params.d_v);
            } else {
                flash::copy_b64_page_one<Kernel_traits, /*Is_even_MN=*/true, Is_even_K>(gV, tVgV, tVrV, tKVcKV, params.d_v, n_block,
                                                                                    block_table, params.v_batch_stride, params.v_row_stride, params.page_block_size);
            }
        } else {
            if constexpr (!Is_page_attn) {
                // Clear the smem tiles to account for predicated off loads
                flash::copy_b64<Is_even_MN, Is_even_K>(
                    tVgV, tVrV, tKVcKV, params.d_v, binfo.actual_seqlen_k - n_block * kBlockN
                );
            } else {
                flash::copy_b64_page_one<Kernel_traits, Is_even_MN, Is_even_K>(gV, tVgV, tVrV, tKVcKV, params.d_v, n_block,
                                                                            block_table, params.v_batch_stride, params.v_row_stride, params.page_block_size, binfo.actual_seqlen_k - n_block * kBlockN);
            }
        }
        flash::sync_threads();

        flash::gemm_opt</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 64) % kAtomLayoutMS * 16 + (tidx & 0xf), kAtomLayoutMS * 16
        );

        cute::copy(tVrV, tVsV);

        if (n_block > n_block_min) {
            // Advance gK
            if constexpr (!Is_page_attn) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
                flash::copy_b64</*Is_even_MN=*/true, Is_even_K>(tKgK, tKrK, tKVcKV, params.d);
            } else {
                flash::copy_b64_page_one<Kernel_traits, /*Is_even_MN=*/true, Is_even_K>(gK, tKgK, tKrK, tKVcKV, params.d, n_block - 1,
                                                                                    block_table, params.k_batch_stride, params.k_row_stride, params.page_block_size);
            }
        }

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
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());

        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma_o, smem_tiled_copy_V, smem_thr_copy_V);

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma_s, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        cute::copy(tKrK, tKsK);
        clear(acc_s);
        // Advance gV
        if constexpr (!Is_page_attn) {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
            flash::copy_b64</*Is_even_MN=*/true, Is_even_K>(tVgV, tVrV, tKVcKV, params.d_v);
        } else {
            flash::copy_b64_page_one<Kernel_traits, /*Is_even_MN=*/true, Is_even_K>(gV, tVgV, tVrV, tKVcKV, params.d_v, n_block,
                                                                                block_table, params.v_batch_stride, params.v_row_stride, params.page_block_size);
        }
        flash::sync_threads();

        flash::gemm_opt</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_s, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        cute::copy(tVrV, tVsV);

        if (n_block > n_block_min) {
            // Advance gK
            if constexpr (!Is_page_attn) {
                tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
                flash::copy_b64</*Is_even_MN=*/true, Is_even_K>(tKgK, tKrK, tKVcKV, params.d);
            } else {
                flash::copy_b64_page_one<Kernel_traits, /*Is_even_MN=*/true, Is_even_K>(gK, tKgK, tKrK, tKVcKV, params.d, n_block - 1,
                                                                                    block_table, params.k_batch_stride, params.k_row_stride, params.page_block_size);
            }
        }

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
        Tensor tOrP = make_tensor(rP.data(), acc_s.layout());

        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma_o, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue


    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, /*Return_lse*/true, Split>(acc_o, params.scale_softmax);
    // if (cute::thread0()) { print(lse); }
    if constexpr (!Split) {
        // use smem for O (mtreg->smem->mtreg->global)
        Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
        // Partition sO to match the accumulator partitioning
        using SmemTiledCopyO = typename Kernel_traits::SmemCopyAtomO;
        auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma_o);
        auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
        //Tensor rO = flash::convert_type<ElementO>(acc_o);
        CONVERT_TENSOR_TYPE(ElementAccum, ElementO, acc_o, rO)
        Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        // sOaccum is larger than sQ, so we need to syncthreads here
        // TODO: allocate enough smem for sOaccum
        if constexpr (Kernel_traits::Share_Q_K_smem) { flash::sync_threads(); }

        cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

        const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
            + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
        const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
                                            + m_block * kBlockM) * params.d_v;
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

        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(params.oaccum_ptr) + row_offset_oaccum),
                                    Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                    make_stride(kHeadDimV, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                    Shape<Int<kBlockM>>{}, Stride<_1>{});
        // if (tidx == 0) { printf("row_offset_o = %d, bidh = %d, gOaccum = %p\n", row_offset_o, bidh, gOaccum.data()); }
        using GmemCopyAtomOaccum = typename Kernel_traits::SmemCopyAtomOaccum;
        auto gmem_tiled_copy_Oaccum = make_tiled_copy_C(GmemCopyAtomOaccum{}, tiled_mma_o);
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor taccOrOaccum = gmem_thr_copy_Oaccum.retile_S(acc_o);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);



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
        flash::copy_reg_to_global<Is_even_MN, Is_even_K>(
            taccOrOaccum, taccOgOaccum, taccOcO, params.d_v, binfo.actual_seqlen_q - m_block * kBlockM
        );
    }
}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, bool Is_page_attn, typename Params>
__forceinline__ __device__ void compute_attn_splitkv(const Params &params, const int m_block_max) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
    // The block index for the head.
    const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
    const int n_split_idx = Split ? blockIdx.y : 0;
    const int num_n_splits = Split ? gridDim.y : 1;
    compute_attn_1rowblock_splitkv_k64_mla_V1x8<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN,Is_even_K, Is_softcap, Split, Append_KV, Is_page_attn>(
            params, bidb, bidh, m_block, n_split_idx, num_n_splits);
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
