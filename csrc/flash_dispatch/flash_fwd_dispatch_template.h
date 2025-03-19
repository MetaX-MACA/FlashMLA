// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)
/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include "flash_mla.h"
#include "static_switch.h"

using namespace mcFlashAttn;

template<
    int kHeadDim,
    int kBlockM,
    int kBlockN,
    int kNWarps,
    bool Is_Q_in_regs,
    bool Share_Q_K_smem,
    typename elem_type,
    bool Is_splits = false,
    int kHeadDimV = kHeadDim,
    int Num_Stages = 1
>
void run_flash_splitkv_fwd_template(Flash_fwd_mla_params &params, cudaStream_t stream);

namespace mcFlashAttn {

    template<int Headdim>
    void run_mha_fwd_splitkv_dispatch(Flash_fwd_mla_params &params, const cudaStream_t stream);

    template<>
    inline void run_mha_fwd_splitkv_dispatch<576>(Flash_fwd_mla_params &params, const cudaStream_t stream) {
        constexpr static int HeaddimQK = 576;
        constexpr static int HeaddimVO = 512;

        constexpr static int Num_Stages = 2;
        FP16_SWITCH(!params.is_bf16, [&] {
            BOOL_SWITCH(params.num_splits > 1, Is_splits, [&] {
                if (params.seqlen_q >= 64) {
                    constexpr static int kBlockM = 64;
                    constexpr static int kBlockN = 16;
                    constexpr static int kNWarps = 8;
                    run_flash_splitkv_fwd_template<HeaddimQK, kBlockM, kBlockN, kNWarps, true, true, elem_type, Is_splits, HeaddimVO, Num_Stages>(params, stream);
                } else {
                    constexpr static int kBlockM = 32;
                    constexpr static int kBlockN = 16;
                    constexpr static int kNWarps = 4;
                    run_flash_splitkv_fwd_template<HeaddimQK, kBlockM, kBlockN, kNWarps, true, true, elem_type, Is_splits, HeaddimVO, Num_Stages>(params, stream);
                }
            });
        });
    }

} // namespace mcFlashAttn end
