// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)

#pragma once

#include <cuda.h>

#include "flash_mla.h"
#include "static_switch.h"
#include "flash_fwd_split_kernel.h"
#include "feature/attn_mask.h"

using namespace mcFlashAttn;

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split>
__global__ void flash_fwd_splitkv_kernel(const Flash_fwd_mla_params params, const int num_m_block) {
    flash::compute_attn_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split>(params, num_m_block);
}

template<typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K>
__global__ void flash_fwd_splitkv_combine_kernel(const Flash_fwd_mla_params params) {
    static_assert(Log_max_splits >= 1);
    flash::combine_attn_seqk_parallel<Kernel_traits, kBlockM, Log_max_splits, Is_even_K>(params);
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd(Flash_fwd_mla_params &params, cudaStream_t stream) {

    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);
    static_assert(Kernel_traits::kHeadDim == 576 && Kernel_traits::kHeadDimV == 512);

    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim && params.d_v == Kernel_traits::kHeadDimV;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        LOCAL_SWITCH_AND_CONST_PRECOND((!Is_causal), (params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
            BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                    auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, false, false, false, IsEvenKConst, false, Split>;
                    if (smem_size >= 32 * 1024) {
                        CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    }
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params, num_m_block);
                    CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
    if (params.num_splits > 1) {
        // We want kBlockM to be as small as possible for more parallelism.
        // With 128 threads we can load 512 elements at a time, so if headdim is divisible by 128, kBlockM = 4.
        // If headdim is divisible by 64, then we set kBlockM = 8, etc.
        //constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 4 : (Kernel_traits::kHeadDim % 64 == 0 ? 8 : 16);
        constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 8 : (Kernel_traits::kHeadDim % 64 == 0 ? 16 : 32);
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        const int kNThreads = 256; /*Kernel_traits::kNThreads;*/
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            NUMSPLITS_SWITCH(params.num_splits, kLogMaxSplits, [&] {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, kLogMaxSplits, IsEvenKConst><<<grid_combine, kNThreads, 0, stream>>>(params);
                CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    }

}
