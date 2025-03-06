// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)

#include <mctlass/numeric_types.h>
#include "run_mha.h"
#include "flash_fwd_dispatch_template.h"

void run_mha_fwd(mcFlashAttn::Flash_fwd_mla_params &params, cudaStream_t stream, bool force_split_kernel) {

    constexpr int kHeadDim = 576;
    mcFlashAttn::run_mha_fwd_splitkv_dispatch<kHeadDim>(params, stream);
}

