// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/algorithm/copy.hpp>

#include "utils.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true, bool Clear_OOB_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_rotary_interleaved(Tensor<Engine0, Layout0> const &S,
                                               Tensor<Engine1, Layout1> &D,
                                               Tensor<Engine2, Layout2> const &Cos,
                                               Tensor<Engine2, Layout2> const &Sin,
                                               Tensor<Engine3, Layout3> const &identity_MN,
                                               const int max_MN, const int min_MN,
                                               const int dim, const int rotary_dim) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Cos));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Cos));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Sin));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Sin));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<0>(Cos) == size<0>(Sin));                     // MMA_K
    static_assert(decltype(size<0>(S))::value == decltype(size<0>(Cos))::value * 2);
    static_assert(decltype(size<0>(Cos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
    Tensor rCos = make_fragment_like(Cos);
    Tensor rSin = make_fragment_like(Sin);
    Tensor rS = make_fragment_like(S);
    typedef __NATIVE_VECTOR__(2, float) Float2;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || get<1>(identity_MN(0, 0, k)) < dim) {
                    cute::copy(S(_, m, k), rS(_, m, k));
                    if (get<1>(identity_MN(0, 0, k)) < rotary_dim) {
                        cute::copy(Cos(_, m, k), rCos(_, m, k));
                        cute::copy(Sin(_, m, k), rSin(_, m, k));
                        // Tensor S_fp32 = convert_type<float>(rS(_, m, k));
                        // Tensor cos_fp32 = convert_type<float>(rCos(_, m, k));
                        // Tensor sin_fp32 = convert_type<float>(rSin(_, m, k));
                        using T = typename Engine0::value_type;
                        using T_rotary = typename Engine2::value_type;
                        CONVERT_TENSOR_TYPE(T, float, rS(_, m, k), S_fp32)
                        CONVERT_TENSOR_TYPE(T_rotary, float, rCos(_, m, k), cos_fp32)
                        CONVERT_TENSOR_TYPE(T_rotary, float, rSin(_, m, k), sin_fp32)

                        #pragma unroll
                        for (int i = 0; i < size<0>(rS) / 2; ++i) {
                            Float2 x_vec = {S_fp32(2 * i), S_fp32(2 * i + 1)};
                            Float2 real_vec = {cos_fp32(i), sin_fp32(i)};
                            Float2 imag_vec = {sin_fp32(i), cos_fp32(i)};
                            Float2 beta_vec = {0.0f, 0.0f};
                            real_vec = __builtin_mxc_pk_fma_f32(x_vec, real_vec, beta_vec);
                            imag_vec = __builtin_mxc_pk_fma_f32(x_vec, imag_vec, beta_vec);
                            S_fp32(2 * i) = real_vec[0] - real_vec[1];
                            S_fp32(2 * i + 1) = imag_vec[0] + imag_vec[1];
                            //float real = S_fp32(2 * i) * cos_fp32(i) - S_fp32(2 * i + 1) * sin_fp32(i);
                            //float imag = S_fp32(2 * i) * sin_fp32(i) + S_fp32(2 * i + 1) * cos_fp32(i);
                            //S_fp32(2 * i) = real;
                            //S_fp32(2 * i + 1) = imag;
                        }
                        // Idk but I need to copy for the convert_type to work
                        Tensor S_fp32_copy = make_fragment_like(S_fp32);
                        cute::copy(S_fp32, S_fp32_copy);
                        //Tensor S_og_type = convert_type<T>(S_fp32_copy);
                        CONVERT_TENSOR_TYPE(float, T, S_fp32_copy, S_og_type)
                        cute::copy(S_og_type, rS(_, m, k));
                    }
                    cute::copy(rS(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true, bool Clear_OOB_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_rotary_contiguous(Tensor<Engine0, Layout0> const &S,
                                              Tensor<Engine1, Layout1> &D,
                                              Tensor<Engine2, Layout2> const &Cos,
                                              Tensor<Engine2, Layout2> const &Sin,
                                              Tensor<Engine3, Layout3> const &identity_MN,
                                              const int max_MN, const int min_MN,
                                              const int dim, const int rotary_dim) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Cos));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Cos));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Sin));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Sin));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(Cos));                     // MMA
    CUTE_STATIC_ASSERT_V(size<0>(Cos) == size<0>(Sin));
    static_assert(decltype(size<0>(Cos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
    Tensor rCos = make_fragment_like(Cos);
    Tensor rSin = make_fragment_like(Sin);
    Tensor rS = make_fragment_like(S);
    Tensor rS_other = make_fragment_like(rS(_, 0, 0));
    typedef __NATIVE_VECTOR__(2, float) Float2;
    Float2 beta_vec = {0.0f, 0.0f};

    const int rotary_dim_half = rotary_dim >> 1;

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || get<1>(identity_MN(0, 0, k)) < dim) {
                    cute::copy(S(_, m, k), rS(_, m, k));
                    if (get<1>(identity_MN(0, 0, k)) < rotary_dim) {
                        const bool is_left = get<1>(identity_MN(0, 0, k)) < rotary_dim_half;
                        Tensor gS_other = make_tensor(S(_, m, k).data() + (is_left ? rotary_dim_half : -rotary_dim_half), S(_, m, k).layout());
                        cute::copy(gS_other, rS_other);
                        // if (cute::thread0()) { print_tensor(rS(_, m, k)); print_tensor(rS_other); }
                        Tensor gCos = make_tensor(Cos(_, m, k).data() + (is_left ? 0 : -rotary_dim_half), Cos(_, m, k).layout());
                        Tensor gSin = make_tensor(Sin(_, m, k).data() + (is_left ? 0 : -rotary_dim_half), Sin(_, m, k).layout());
                        cute::copy(gCos, rCos(_, m, k));
                        cute::copy(gSin, rSin(_, m, k));
                        // if (cute::thread0()) { print_tensor(rCos(_, m, k)); print_tensor(rSin(_, m, k)); }
                        // Tensor S_fp32 = convert_type<float>(rS(_, m, k));
                        // Tensor S_other_fp32 = convert_type<float>(rS_other);
                        // Tensor cos_fp32 = convert_type<float>(rCos(_, m, k));
                        // Tensor sin_fp32 = convert_type<float>(rSin(_, m, k));

                        using T = typename Engine0::value_type;
                        using T_rotary = typename Engine2::value_type;
                        CONVERT_TENSOR_TYPE(T, float, rS(_,m,k), S_fp32)
                        CONVERT_TENSOR_TYPE(T, float, rS_other, S_other_fp32)
                        CONVERT_TENSOR_TYPE(T_rotary, float, rCos(_, m, k), cos_fp32)
                        CONVERT_TENSOR_TYPE(T_rotary, float, rSin(_, m, k), sin_fp32)

                        #pragma unroll
                        for (int i = 0; i < size<0>(rS); ++i) {
                            //S_fp32(i) = S_fp32(i) * cos_fp32(i) + S_other_fp32(i) * (is_left ? -sin_fp32(i) : sin_fp32(i));
                            Float2 x_vec = {S_fp32(i), S_other_fp32(i)};
                            Float2 alpha_vec = {cos_fp32(i), is_left ? -sin_fp32(i) : sin_fp32(i)};
                            Float2 y_vec = __builtin_mxc_pk_fma_f32(x_vec, alpha_vec, beta_vec);
                            S_fp32(i) = y_vec[0] + y_vec[1];
                        }
                        // Idk but I need to copy for the convert_type to work
                        Tensor S_fp32_copy = make_fragment_like(S_fp32);
                        cute::copy(S_fp32, S_fp32_copy);
                        //Tensor S_og_type = convert_type<T>(S_fp32_copy);
                        CONVERT_TENSOR_TYPE(float, T, S_fp32_copy, S_og_type)

                        cute::copy(S_og_type, rS(_, m, k));
                        // if (cute::thread0()) { print_tensor(rS(_, m, k)); }
                    }
                    cute::copy(rS(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        }
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_rotary_interleaved_to_reg(Tensor<Engine0, Layout0> const &S,
                                               uint32_t *D_ptr,
                                               Tensor<Engine1, Layout1> const &Cos,
                                               Tensor<Engine1, Layout1> const &Sin,
                                               Tensor<Engine2, Layout2> const &identity_MN,
                                               const int max_MN, const int min_MN,
                                               const int dim, const int rotary_dim) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Cos));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Cos));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Sin));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Sin));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<0>(Cos) == size<0>(Sin));                     // MMA_K
    static_assert(decltype(size<0>(S))::value == decltype(size<0>(Cos))::value * 2);
    static_assert(decltype(size<0>(Cos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
    Tensor rCos = make_fragment_like(Cos);
    Tensor rSin = make_fragment_like(Sin);
    Tensor rS = make_fragment_like(S);

    typedef __NATIVE_VECTOR__(2, float) Float2;
    typedef __NATIVE_VECTOR__(4, int) VecTypeB128;
    typedef __NATIVE_VECTOR__(2, int) VecTypeB64;

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN;
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            bool mask = row_mask && (Is_even_K || get<1>(identity_MN(0, 0, k)) < dim);
            auto S_ptr = (VecTypeB128 *)(S(_, m, k).data().ptr_);
            auto rS_ptr = (VecTypeB128 *)(rS(_, m, k).data());
            rS_ptr[0] = __builtin_mxc_ldg_b128_predicator(S_ptr, 0, true, true, false, false,
                                                            mask, 1, MACA_ICMP_EQ);
            bool rotary_mask = mask && (get<1>(identity_MN(0, 0, k)) < rotary_dim);

            auto gCos_ptr = (VecTypeB64 *)(Cos(_, m, k).data().ptr_);
            auto gSin_ptr = (VecTypeB64 *)(Sin(_, m, k).data().ptr_);
            auto rCos_ptr = (VecTypeB64 *)(rCos(_, m, k).data());
            auto rSin_ptr = (VecTypeB64 *)(rSin(_, m, k).data());
            rCos_ptr[0] = __builtin_mxc_ldg_b64_predicator(gCos_ptr, 0, true, true, false, false,
                                                            rotary_mask, 1, MACA_ICMP_EQ);
            rSin_ptr[0] = __builtin_mxc_ldg_b64_predicator(gSin_ptr, 0, true, true, false, false,
                                                            rotary_mask, 1, MACA_ICMP_EQ);

            if (rotary_mask) {
                using T = typename Engine0::value_type;
                using T_rotary = typename Engine1::value_type;
                CONVERT_TENSOR_TYPE(T, float, rS(_, m, k), S_fp32)
                CONVERT_TENSOR_TYPE(T_rotary, float, rCos(_, m, k), cos_fp32)
                CONVERT_TENSOR_TYPE(T_rotary, float, rSin(_, m, k), sin_fp32)

                #pragma unroll
                for (int i = 0; i < size<0>(rS) / 2; ++i) {
                    Float2 x_vec = {S_fp32(2 * i), S_fp32(2 * i + 1)};
                    Float2 real_vec = {cos_fp32(i), sin_fp32(i)};
                    Float2 imag_vec = {sin_fp32(i), cos_fp32(i)};
                    Float2 beta_vec = {0.0f, 0.0f};
                    real_vec = __builtin_mxc_pk_fma_f32(x_vec, real_vec, beta_vec);
                    imag_vec = __builtin_mxc_pk_fma_f32(x_vec, imag_vec, beta_vec);
                    S_fp32(2 * i) = real_vec[0] - real_vec[1];
                    S_fp32(2 * i + 1) = imag_vec[0] + imag_vec[1];
                }
                // Idk but I need to copy for the convert_type to work
                Tensor S_fp32_copy = make_fragment_like(S_fp32);
                cute::copy(S_fp32, S_fp32_copy);
                //Tensor S_og_type = convert_type<T>(S_fp32_copy);
                CONVERT_TENSOR_TYPE(float, T, S_fp32_copy, S_og_type)
                cute::copy(S_og_type, rS(_, m, k));
            }
            const int idx = (m * size<2>(S) + k) << 2;
            auto D = (VecTypeB128 *)(D_ptr + idx);
            D[0] = rS_ptr[0];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_rotary_contiguous_to_reg(Tensor<Engine0, Layout0> const &S,
                                              uint32_t *D_ptr,
                                              Tensor<Engine1, Layout1> const &Cos,
                                              Tensor<Engine1, Layout1> const &Sin,
                                              Tensor<Engine2, Layout2> const &identity_MN,
                                              const int max_MN, const int min_MN,
                                              const int dim, const int rotary_dim) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Cos));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Cos));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Sin));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Sin));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(Cos));                     // MMA
    CUTE_STATIC_ASSERT_V(size<0>(Cos) == size<0>(Sin));
    static_assert(decltype(size<0>(Cos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
    Tensor rCos = make_fragment_like(Cos);
    Tensor rSin = make_fragment_like(Sin);
    Tensor rS = make_fragment_like(S);
    Tensor rS_other = make_fragment_like(rS(_, 0, 0));
    typedef __NATIVE_VECTOR__(2, float) Float2;
    Float2 beta_vec = {0.0f, 0.0f};

    const int rotary_dim_half = rotary_dim >> 1;

    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN;
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            bool mask = row_mask && (Is_even_K || get<1>(identity_MN(0, 0, k)) < dim);
            auto S_ptr = (VecType *)(S(_, m, k).data().ptr_);
            auto rS_ptr = (VecType *)(rS(_, m, k).data());
            rS_ptr[0] = __builtin_mxc_ldg_b128_predicator(S_ptr, 0, true, true, false, false,
                                                            mask, 1, MACA_ICMP_EQ);

            bool rotary_mask = mask && (get<1>(identity_MN(0, 0, k)) < rotary_dim);
            const bool is_left = get<1>(identity_MN(0, 0, k)) < rotary_dim_half;
            auto gS_ptr = (VecType *)(S(_, m, k).data().ptr_ + (is_left ? rotary_dim_half : -rotary_dim_half));
            auto rS_other_ptr = (VecType *)(rS_other.data());
            rS_other_ptr[0] = __builtin_mxc_ldg_b128_predicator(gS_ptr, 0, true, true, false, false,
                                                            rotary_mask, 1, MACA_ICMP_EQ);

            auto gCos_ptr = (VecType *)(Cos(_, m, k).data().ptr_ + (is_left ? 0 : -rotary_dim_half));
            auto gSin_ptr = (VecType *)(Sin(_, m, k).data().ptr_ + (is_left ? 0 : -rotary_dim_half));
            auto rCos_ptr = (VecType *)(rCos(_, m, k).data());
            auto rSin_ptr = (VecType *)(rSin(_, m, k).data());
            rCos_ptr[0] = __builtin_mxc_ldg_b128_predicator(gCos_ptr, 0, true, true, false, false,
                                                            rotary_mask, 1, MACA_ICMP_EQ);
            rSin_ptr[0] = __builtin_mxc_ldg_b128_predicator(gSin_ptr, 0, true, true, false, false,
                                                            rotary_mask, 1, MACA_ICMP_EQ);
            if (rotary_mask) {
                using T = typename Engine0::value_type;
                using T_rotary = typename Engine1::value_type;
                CONVERT_TENSOR_TYPE(T, float, rS(_, m, k), S_fp32)
                CONVERT_TENSOR_TYPE(T, float, rS_other, S_other_fp32)
                CONVERT_TENSOR_TYPE(T_rotary, float, rCos(_, m, k), cos_fp32)
                CONVERT_TENSOR_TYPE(T_rotary, float, rSin(_, m, k), sin_fp32)

                #pragma unroll
                for (int i = 0; i < size<0>(rS); ++i) {
                    // S_fp32(i) = S_fp32(i) * cos_fp32(i) + S_other_fp32(i) * (is_left ? -sin_fp32(i) : sin_fp32(i));
                    Float2 x_vec = {S_fp32(i), S_other_fp32(i)};
                    Float2 alpha_vec = {cos_fp32(i), is_left ? -sin_fp32(i) : sin_fp32(i)};
                    Float2 y_vec = __builtin_mxc_pk_fma_f32(x_vec, alpha_vec, beta_vec);
                    S_fp32(i) = y_vec[0] + y_vec[1];
                }
                // Idk but I need to copy for the convert_type to work
                Tensor S_fp32_copy = make_fragment_like(S_fp32);
                cute::copy(S_fp32, S_fp32_copy);
                //Tensor S_og_type = convert_type<T>(S_fp32_copy);
                CONVERT_TENSOR_TYPE(float, T, S_fp32_copy, S_og_type)
                cute::copy(S_og_type, rS(_, m, k));
            }
            const int idx = (m * size<2>(S) + k) * 4;
            auto D = (VecType *)(D_ptr + idx);
            D[0] = rS_ptr[0];
        }
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_rotary_interleaved_to_global(Tensor<Engine0, Layout0> const &S,
                                               Tensor<Engine1, Layout1> &D,
                                               Tensor<Engine2, Layout2> const &Cos,
                                               Tensor<Engine2, Layout2> const &Sin,
                                               Tensor<Engine3, Layout3> const &identity_MN,
                                               const int max_MN, const int min_MN,
                                               const int dim, const int rotary_dim) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Cos));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Cos));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Sin));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Sin));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<0>(Cos) == size<0>(Sin));                     // MMA_K
    static_assert(decltype(size<0>(S))::value == decltype(size<0>(Cos))::value * 2);
    static_assert(decltype(size<0>(Cos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
    Tensor rCos = make_fragment_like(Cos);
    Tensor rSin = make_fragment_like(Sin);
    Tensor rS = make_fragment_like(S);
    typedef __NATIVE_VECTOR__(2, float) Float2;
    typedef __NATIVE_VECTOR__(4, int) VecTypeB128;
    typedef __NATIVE_VECTOR__(2, int) VecTypeB64;

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN;
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            bool mask = row_mask && (Is_even_K || get<1>(identity_MN(0, 0, k)) < dim);
            auto S_ptr = (VecTypeB128 *)(S(_, m, k).data().ptr_);
            auto rS_ptr = (VecTypeB128 *)(rS(_, m, k).data());
            rS_ptr[0] = __builtin_mxc_ldg_b128_predicator(S_ptr, 0, true, true, false, false,
                                                            mask, 1, MACA_ICMP_EQ);
            bool rotary_mask = mask && (get<1>(identity_MN(0, 0, k)) < rotary_dim);

            auto gCos_ptr = (VecTypeB64 *)(Cos(_, m, k).data().ptr_);
            auto gSin_ptr = (VecTypeB64 *)(Sin(_, m, k).data().ptr_);
            auto rCos_ptr = (VecTypeB64 *)(rCos(_, m, k).data());
            auto rSin_ptr = (VecTypeB64 *)(rSin(_, m, k).data());
            rCos_ptr[0] = __builtin_mxc_ldg_b64_predicator(gCos_ptr, 0, true, true, false, false,
                                                            rotary_mask, 1, MACA_ICMP_EQ);
            rSin_ptr[0] = __builtin_mxc_ldg_b64_predicator(gSin_ptr, 0, true, true, false, false,
                                                            rotary_mask, 1, MACA_ICMP_EQ);

            if (rotary_mask) {
                using T = typename Engine0::value_type;
                using T_rotary = typename Engine1::value_type;
                CONVERT_TENSOR_TYPE(T, float, rS(_, m, k), S_fp32)
                CONVERT_TENSOR_TYPE(T_rotary, float, rCos(_, m, k), cos_fp32)
                CONVERT_TENSOR_TYPE(T_rotary, float, rSin(_, m, k), sin_fp32)

                #pragma unroll
                for (int i = 0; i < size<0>(rS) / 2; ++i) {
                    Float2 x_vec = {S_fp32(2 * i), S_fp32(2 * i + 1)};
                    Float2 real_vec = {cos_fp32(i), sin_fp32(i)};
                    Float2 imag_vec = {sin_fp32(i), cos_fp32(i)};
                    Float2 beta_vec = {0.0f, 0.0f};
                    real_vec = __builtin_mxc_pk_fma_f32(x_vec, real_vec, beta_vec);
                    imag_vec = __builtin_mxc_pk_fma_f32(x_vec, imag_vec, beta_vec);
                    S_fp32(2 * i) = real_vec[0] - real_vec[1];
                    S_fp32(2 * i + 1) = imag_vec[0] + imag_vec[1];
                }
                // Idk but I need to copy for the convert_type to work
                Tensor S_fp32_copy = make_fragment_like(S_fp32);
                cute::copy(S_fp32, S_fp32_copy);
                //Tensor S_og_type = convert_type<T>(S_fp32_copy);
                CONVERT_TENSOR_TYPE(float, T, S_fp32_copy, S_og_type)
                cute::copy(S_og_type, rS(_, m, k));
            }
            auto D_ptr = (VecTypeB128 *)(D(_, m, k).data().ptr_);
            __builtin_mxc_stg_b128_predicator(D_ptr, 0, rS_ptr[0], true, false, true, mask, 1, MACA_ICMP_EQ);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_rotary_contiguous_to_global(Tensor<Engine0, Layout0> const &S,
                                              Tensor<Engine1, Layout1> &D,
                                              Tensor<Engine2, Layout2> const &Cos,
                                              Tensor<Engine2, Layout2> const &Sin,
                                              Tensor<Engine3, Layout3> const &identity_MN,
                                              const int max_MN, const int min_MN,
                                              const int dim, const int rotary_dim) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Cos));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Cos));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Sin));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Sin));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(Cos));                     // MMA
    CUTE_STATIC_ASSERT_V(size<0>(Cos) == size<0>(Sin));
    static_assert(decltype(size<0>(Cos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
    Tensor rCos = make_fragment_like(Cos);
    Tensor rSin = make_fragment_like(Sin);
    Tensor rS = make_fragment_like(S);
    Tensor rS_other = make_fragment_like(rS(_, 0, 0));
    typedef __NATIVE_VECTOR__(2, float) Float2;
    Float2 beta_vec = {0.0f, 0.0f};

    const int rotary_dim_half = rotary_dim >> 1;

    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN;
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            bool mask = row_mask && (Is_even_K || get<1>(identity_MN(0, 0, k)) < dim);
            auto S_ptr = (VecType *)(S(_, m, k).data().ptr_);
            auto rS_ptr = (VecType *)(rS(_, m, k).data());
            rS_ptr[0] = __builtin_mxc_ldg_b128_predicator(S_ptr, 0, true, true, false, false,
                                                            mask, 1, MACA_ICMP_EQ);

            bool rotary_mask = mask && (get<1>(identity_MN(0, 0, k)) < rotary_dim);
            const bool is_left = get<1>(identity_MN(0, 0, k)) < rotary_dim_half;
            auto gS_ptr = (VecType *)(S(_, m, k).data().ptr_ + (is_left ? rotary_dim_half : -rotary_dim_half));
            auto rS_other_ptr = (VecType *)(rS_other.data());
            rS_other_ptr[0] = __builtin_mxc_ldg_b128_predicator(gS_ptr, 0, true, true, false, false,
                                                            rotary_mask, 1, MACA_ICMP_EQ);

            auto gCos_ptr = (VecType *)(Cos(_, m, k).data().ptr_ + (is_left ? 0 : -rotary_dim_half));
            auto gSin_ptr = (VecType *)(Sin(_, m, k).data().ptr_ + (is_left ? 0 : -rotary_dim_half));
            auto rCos_ptr = (VecType *)(rCos(_, m, k).data());
            auto rSin_ptr = (VecType *)(rSin(_, m, k).data());
            rCos_ptr[0] = __builtin_mxc_ldg_b128_predicator(gCos_ptr, 0, true, true, false, false,
                                                            rotary_mask, 1, MACA_ICMP_EQ);
            rSin_ptr[0] = __builtin_mxc_ldg_b128_predicator(gSin_ptr, 0, true, true, false, false,
                                                            rotary_mask, 1, MACA_ICMP_EQ);
            if (rotary_mask) {
                using T = typename Engine0::value_type;
                using T_rotary = typename Engine1::value_type;
                CONVERT_TENSOR_TYPE(T, float, rS(_, m, k), S_fp32)
                CONVERT_TENSOR_TYPE(T, float, rS_other, S_other_fp32)
                CONVERT_TENSOR_TYPE(T_rotary, float, rCos(_, m, k), cos_fp32)
                CONVERT_TENSOR_TYPE(T_rotary, float, rSin(_, m, k), sin_fp32)

                #pragma unroll
                for (int i = 0; i < size<0>(rS); ++i) {
                    // S_fp32(i) = S_fp32(i) * cos_fp32(i) + S_other_fp32(i) * (is_left ? -sin_fp32(i) : sin_fp32(i));
                    Float2 x_vec = {S_fp32(i), S_other_fp32(i)};
                    Float2 alpha_vec = {cos_fp32(i), is_left ? -sin_fp32(i) : sin_fp32(i)};
                    Float2 y_vec = __builtin_mxc_pk_fma_f32(x_vec, alpha_vec, beta_vec);
                    S_fp32(i) = y_vec[0] + y_vec[1];
                }
                // Idk but I need to copy for the convert_type to work
                Tensor S_fp32_copy = make_fragment_like(S_fp32);
                cute::copy(S_fp32, S_fp32_copy);
                //Tensor S_og_type = convert_type<T>(S_fp32_copy);
                CONVERT_TENSOR_TYPE(float, T, S_fp32_copy, S_og_type)
                cute::copy(S_og_type, rS(_, m, k));
            }
            auto D_ptr = (VecType *)(D(_, m, k).data().ptr_);
            __builtin_mxc_stg_b128_predicator(D_ptr, 0, rS_ptr[0], true, false, true, mask, 1, MACA_ICMP_EQ);
        }
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
