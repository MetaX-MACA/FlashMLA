// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)
#pragma once

#include "flash_mla.h"

void run_mha_fwd(mcFlashAttn::Flash_fwd_mla_params &params, cudaStream_t stream, bool force_split_kernel=false);
