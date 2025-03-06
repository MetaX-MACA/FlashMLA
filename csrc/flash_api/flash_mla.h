// Adapted from deepseek-ai/FlashMLA(https://github.com/deepseek-ai/FlashMLA)
/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <vector>

constexpr int maxValidBlockSizeM = 128;

namespace mcFlashAttn {

struct Qkv_params {
    using index_t = int64_t;

    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,

};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_mla_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, d_v, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q;
    int ngroups;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ leftpad_k;

    // If provided, the actual length of each k sequence.
    int * __restrict__ seqused_k;

    int *__restrict__ blockmask;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // kv cache dequant
    index_t kscale_batch_stride;
    index_t vscale_batch_stride;
    index_t kscale_row_stride;
    index_t vscale_row_stride;
    index_t kscale_head_stride;
    index_t vscale_head_stride;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;

    // The indices to index into the KV cache.
    int * __restrict__ cache_batch_idx;

    // Paged KV cache
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    // when page attn is not enable, page_block_size will has default value 0.
    int page_block_size;

    // KV Cache dequant
    int dequant_group;
    void *__restrict__ k_scale_ptr;
    void *__restrict__ v_scale_ptr;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    // uint32_t p_dropout_in_uint;
    // uint16_t p_dropout_in_uint16_t;
    uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_softmax_rp_dropout;

    // Local window size
    int window_size_left, window_size_right;

    // ratio of softcapping attention
    // S = exp2(log2(e) * softcap * tanh(S * softmax_scale / softcap))
    // only value > 0.0 will take effect
    float softcap;

    // Random state.
    // at::PhiloxCudaState philox_args;

    // the RNG seed and offset .
    uint64_t rng_state_seed = 0;
    uint64_t rng_state_offset = 0;

    bool is_bf16;
    bool is_causal;

    // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
    // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
    bool is_seqlens_k_cumulative;

    bool is_rotary_interleaved;

    int num_splits;  // For split-KV version

    void * __restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;

    // attn_mask support for bert model Jira[C500-21935]
    bool has_attn_mask;
    void * __restrict__ attn_mask_ptr = nullptr;
    index_t attn_mask_batch_stride = 0;
    index_t attn_mask_nheads_stride = 0;
    index_t attn_mask_row_stride = 0;
    index_t attn_mask_col_stride = 1;

    index_t attn_mask_batch_shape = 1;
    index_t attn_mask_nheads_shape = 1;
    index_t attn_mask_row_shape = 1;
    index_t attn_mask_col_shape = 1;

    bool unpadded_lse;  // For varlen paths: LSE is in [nheads, total_seqlen_q] format instead of [b, nheads, seqlen_q].
    bool seqlenq_ngroups_swapped;  // q has been transposed from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d).

    int d_value;
    int d_value_rounded;

    bool is_support_splitkv = false;
};


struct Flash_launch_params {
    bool is_balance;
    int rowblock_parallel;
    int block_type;

    bool performance_mode; // from offline

    Flash_launch_params():
        is_balance(false),rowblock_parallel(0),block_type(0),performance_mode(false){}
};

}

static constexpr int TileSchedulerMetaDataSize = 8;
// [begin_idx, begin_seqlen, end_idx, end_seqlen, begin_n_split_idx, _, _, _]

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Mla_metadata_params {
    int *__restrict__ seqlens_k_ptr;
    int *__restrict__ tile_scheduler_metadata_ptr;
    int *__restrict__ num_splits_ptr;
    int batch_size;
    int block_size_n;
    int fixed_overhead_num_blocks;
    int num_sm_parts;
};

void get_mla_metadata_func(Mla_metadata_params &params, cudaStream_t stream);
