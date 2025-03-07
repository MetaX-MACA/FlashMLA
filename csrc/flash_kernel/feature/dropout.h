// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)

/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "philox.cuh"
#include "utils.h"

namespace flash {

struct Dropout {

    const unsigned long long seed, offset;
    const uint8_t p_dropout_in_uint8_t;

    __forceinline__ __device__ Dropout(const unsigned long long seed, const unsigned long long offset,
                              const uint8_t p_dropout_in_uint8_t,
                              const int bid, const int hid, const int tid, const int nheads)
            : seed(seed)
            , offset(offset + (bid * nheads + hid) * 64 + tid % 64)
            , p_dropout_in_uint8_t(p_dropout_in_uint8_t) {
    }

    template <bool encode_dropout_in_sign_bit=false, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_dropout(Tensor<Engine, Layout> &tensor_,
                                         int block_row_start, int block_col_start, int block_row_stride) {
        // convert shape from (4, MMA_M, MMA_N) to (8, MMA_M, MMA_N / 2)
        Tensor tensor = make_tensor(tensor_.data(), flash::convert_layout_acc_dropout(tensor_.layout()));
        using T = typename Engine::value_type;
        auto encode_dropout = [](bool keep, T val) {
            return keep ? val : (encode_dropout_in_sign_bit ? -val : T(0));
        };
        static_assert(decltype(size<2>(tensor))::value % 2 == 0);
        const uint16_t p_dropout_8bit_in_uint16_t = uint16_t(p_dropout_in_uint8_t);
        const uint32_t p_dropout_8bit_in_uint32_t = (uint32_t(p_dropout_8bit_in_uint16_t) << 16) | uint32_t(p_dropout_8bit_in_uint16_t);
        // if (cute::thread0()) { printf("threshold2 = 0x%x\n", p_dropout_8bit_in_uint32_t); }
        #pragma unroll
        for (int m = 0; m < size<1>(tensor); ++m, block_row_start += block_row_stride) {
            uint2 rowcol = make_uint2(block_row_start, block_col_start);
            #pragma unroll
            for (int n = 0; n < size<2>(tensor) / 2; ++n, ++rowcol.y) {
                // if (cute::thread(32, 0)) { printf("m = %d, n = %d, row = %d, col = %d\n", m, n, int(rowcol.x), int(rowcol.y));}
                uint4 random_uint4 = flash::philox(seed, reinterpret_cast<unsigned long long&>(rowcol), offset);
                // if (cute::thread0()) { printf("philox = %u, %d, %d, %d\n", random_uint4.x, random_uint4.y, random_uint4.z, random_uint4.w);}
                uint8_t (&rnd_8)[16] = reinterpret_cast<uint8_t (&)[16]>(random_uint4);
                // Special implementation for 16-bit types: we duplicate the threshold to the
                // low and high 16 bits of a 32-bit value, then use the f16x2 comparison instruction
                // to get a mask. The low 16 bits of the mask will be either 0xffff or 0x0000,
                // and the high 16 bits will be either 0xffff or 0x0000, depending on whether
                // the random value is less than the threshold.
                // We then do a bit-wise AND between the mask and the original value (in 32-bit).
                // We're exploiting the fact that floating point comparison is equivalent to integer
                // comparison, since we're comparing unsigned integers whose top 8-bits are zero.
                if (!encode_dropout_in_sign_bit
                    && (std::is_same<T, mctlass::half_t>::value || std::is_same<T, mctlass::bfloat16_t>::value)) {
                    uint16_t rnd_16[16];
                    #pragma unroll
                    for (int i = 0; i < 16; i++) { rnd_16[i] = uint16_t(rnd_8[i]); }
                    uint32_t (&rnd_32)[8] = reinterpret_cast<uint32_t (&)[8]>(rnd_16);
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        Tensor tensor_uint32 = recast<uint32_t>(tensor(_, m, n * 2 + j));
                        // if (cute::thread0()) { printf("random = 0x%x, 0x%x, 0x%x, 0x%x\n", rnd_32[j * 4 + 0], rnd_32[j * 4 + 1], rnd_32[j * 4 + 2], rnd_32[j * 4 + 3]); }
                        // if (cute::thread0()) { printf("tensor_uint32 = 0x%x, 0x%x, 0x%x, 0x%x\n", tensor_uint32(0), tensor_uint32(1), tensor_uint32(2), tensor_uint32(3)); }
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            uint32_t mask;
                            //asm volatile("set.le.u32.f16x2 %0, %1, %2;\n" : "=r"(mask) : "r"(rnd_32[j * 4 + i]), "r"(p_dropout_8bit_in_uint32_t));
                            tensor_uint32(i) &= mask;
                        }
                        // if (cute::thread0()) { printf("tensor_uint32 = 0x%x, 0x%x, 0x%x, 0x%x\n", tensor_uint32(0), tensor_uint32(1), tensor_uint32(2), tensor_uint32(3)); }
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 2; j++) {
                        #pragma unroll
                        for (int i = 0; i < 8; i++) {
                            tensor(i, m, n * 2 + j) = encode_dropout(rnd_8[j * 8 + i] <= p_dropout_in_uint8_t, tensor(i, m, n * 2 + j));
                        }
                        Tensor tensor_uint32 = recast<uint32_t>(tensor(_, m, n * 2 + j));
                        // if (cute::thread0()) { printf("tensor_uint32 = 0x%x, 0x%x, 0x%x, 0x%x\n", tensor_uint32(0), tensor_uint32(1), tensor_uint32(2), tensor_uint32(3)); }
                    }
                }
                // // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
                // //     printf("n = %d, ph  Philox: %u, %u, %u, %u\n", n, rnd_8.x, rnd_8.y, rnd_8.z, rnd_8.w);
                // // }
            }
        }
    }

    template <bool encode_dropout_in_sign_bit = false, int AtomLayoutNS = 1, typename Engine, typename Layout>
    __forceinline__ __device__ void mc_apply_dropout(Tensor<Engine, Layout> &tensor,
                                            int block_row_start, int block_col_start,
                                            int block_row_stride,
                                            int kBlockN, int n_block) {
        using T = typename Engine::value_type;
        auto encode_dropout = [](bool keep, T val) {
            return keep ? val : (encode_dropout_in_sign_bit ? -val : T(0));
        };
        static_assert(decltype(size<0>(tensor))::value == 4);
        int warp_id = threadIdx.x / 64;

        #pragma unroll
        for (int m = 0; m < size<1>(tensor); ++m, block_row_start += block_row_stride) {
            // use block_col_offset to control rnd_8 apply
            int block_col_offset = ((kBlockN * n_block) % 64) / 16;
            int warp_col_offset = warp_id / block_row_stride;
            int block_col_start_tmp = block_col_start;

            #pragma unroll
            for (int n = 0; n < size<2>(tensor); ++n, warp_col_offset += AtomLayoutNS) {
                // when blockN=128, one block contain two 64 in col, so we need to update block_col_start_tmp
                block_col_start_tmp += warp_col_offset / 4;
                warp_col_offset %= 4;
                // rnd_8 contains 16 nums for kBlockN 64 in 1 block, to process 64 in col
                // if use kBlockN 32 in 2 block to process 64 in col, rnd_8 should apply in two differen block
                uint2 rowcol = make_uint2(block_row_start, block_col_start_tmp);
                uint4 random_uint4 = flash::philox(seed, reinterpret_cast<unsigned long long &>(rowcol), offset);
                uint8_t (&rnd_8)[16] = reinterpret_cast<uint8_t (&)[16]>(random_uint4);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int rng_idx = (block_col_offset + warp_col_offset) * 4 + i;
                    // This implementation is a native perf version,here assembly instruction will have ldp_u8.
                    tensor(i, m, n) = encode_dropout(rnd_8[rng_idx] <= p_dropout_in_uint8_t, tensor(i, m, n));
                }
            }
        }
    }

    // it works for:
    // blockN=64, waves layout: 4x1, 2x2, 4x2, 1x4, 2x4
    // blockN=128, waves layout: 1x4, 2x4
    template <bool encode_dropout_in_sign_bit = false, int AtomLayoutMS = 2, int AtomLayoutNS = 2, typename Engine, typename Layout>
    __forceinline__ __device__ void mc_apply_dropout(Tensor<Engine, Layout> &tensor, int block_row_start, int block_col_start) {
        using T = typename Engine::value_type;
        auto encode_dropout = [](bool keep, T val) {
            return keep ? val : (encode_dropout_in_sign_bit ? -val : T(0));
        };
        static_assert(decltype(size<0>(tensor))::value == 4);
        const int wave_col = threadIdx.x / 64 / AtomLayoutMS;

        if constexpr (AtomLayoutNS == 1) {
            #pragma unroll
            for (int m = 0; m < size<1>(tensor); ++m, block_row_start += AtomLayoutMS) {
                uint2 rowcol = make_uint2(block_row_start, block_col_start);
                uint4 random_uint4 = flash::philox(seed, reinterpret_cast<unsigned long long &>(rowcol), offset);
                uint8_t (&rnd_8)[16] = reinterpret_cast<uint8_t (&)[16]>(random_uint4);
                #pragma unroll
                for (int n = 0; n < size<2>(tensor); ++n) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        // w0|w0|w0|w0
                        tensor(i, m, n) = encode_dropout(rnd_8[n * 4 + i] <= p_dropout_in_uint8_t, tensor(i, m, n));
                    }
                }
            }
        } else if constexpr (AtomLayoutNS == 2) {
            #pragma unroll
            for (int m = 0; m < size<1>(tensor); ++m, block_row_start += AtomLayoutMS) {
                uint2 rowcol = make_uint2(block_row_start, block_col_start);
                uint4 random_uint4 = flash::philox(seed, reinterpret_cast<unsigned long long &>(rowcol), offset);
                uint8_t (&rnd_8)[16] = reinterpret_cast<uint8_t (&)[16]>(random_uint4);
                #pragma unroll
                for (int n = 0; n < size<2>(tensor); ++n) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        // e.g., w0|w2|w0|w2
                        if (wave_col == 0) {
                            tensor(i, m, n) = encode_dropout(rnd_8[n * 8 + i] <= p_dropout_in_uint8_t, tensor(i, m, n));
                        } else {
                            tensor(i, m, n) = encode_dropout(rnd_8[n * 8 + i + 4] <= p_dropout_in_uint8_t, tensor(i, m, n));
                        }
                    }
                }
            }
        } else if constexpr (AtomLayoutNS == 4) {
            #pragma unroll
            for (int m = 0; m < size<1>(tensor); ++m, block_row_start += AtomLayoutMS) {
                #pragma unroll
                for (int n = 0; n < size<2>(tensor); ++n, block_col_start += 1) {
                    uint2 rowcol = make_uint2(block_row_start, block_col_start);
                    uint4 random_uint4 = flash::philox(seed, reinterpret_cast<unsigned long long &>(rowcol), offset);
                    uint8_t (&rnd_8)[16] = reinterpret_cast<uint8_t (&)[16]>(random_uint4);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        // e.g., w0|w2|w4|w6
                        if (wave_col == 0) {
                            tensor(i, m, n) = encode_dropout(rnd_8[i] <= p_dropout_in_uint8_t, tensor(i, m, n));
                        } else if (wave_col == 1) {
                            tensor(i, m, n) = encode_dropout(rnd_8[i + 4] <= p_dropout_in_uint8_t, tensor(i, m, n));
                        } else if (wave_col == 2) {
                            tensor(i, m, n) = encode_dropout(rnd_8[i + 8] <= p_dropout_in_uint8_t, tensor(i, m, n));
                        } else {
                            tensor(i, m, n) = encode_dropout(rnd_8[i + 12] <= p_dropout_in_uint8_t, tensor(i, m, n));
                        }
                    }
                }
            }
        }
    }
};

} // namespace flash
