// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)
/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <mcr/mc_runtime_api.h>

#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <mctlass/array.h>
#include <mctlass/mctlass.h>
#include <mctlass/numeric_conversion.h>
#include <mctlass/numeric_types.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline__ __device__ dim3 get_bidInfo(const int& blockType) {

    int m_block = blockIdx.y;
    int bidb = blockIdx.z;
    int bidh = blockIdx.x;

    if (blockType == 0) {
        int m_block = blockIdx.x;
        int bidb = blockIdx.z;
        int bidh = blockIdx.y;
        return dim3(m_block, bidb, bidh);
    }
    if (blockType == 1) {
        int m_block = blockIdx.x;
        int bidb = blockIdx.y;
        int bidh = blockIdx.z;
        return dim3(m_block, bidb, bidh);
    }

    if (blockType == 2) {
        int m_block = blockIdx.y;
        int bidb = blockIdx.z;
        int bidh = blockIdx.x;
        return dim3(m_block, bidb, bidh);
    }

    if (blockType == 3) {
        int m_block = blockIdx.y;
        int bidb = blockIdx.x;
        int bidh = blockIdx.z;
        return dim3(m_block, bidb, bidh);
    }

    if (blockType == 4) {
        int m_block = blockIdx.z;
        int bidb = blockIdx.x;
        int bidh = blockIdx.y;
        return dim3(m_block, bidb, bidh);
    }

    if (blockType == 5) {
        int m_block = blockIdx.z;
        int bidb = blockIdx.y;
        int bidh = blockIdx.x;
        return dim3(m_block, bidb, bidh);
    }

    return dim3(0, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Split>
__forceinline__ __device__ dim3 get_bidInfo(const int& blockType, const int& h, int& n_split_idx) {

    if constexpr(Split) {
        int m_block = blockIdx.y;
        int bidb = blockIdx.x / h;
        int bidh = blockIdx.x - bidb * h;
        n_split_idx = blockIdx.z;
        if (blockType == 0) {
            m_block = blockIdx.z;
            n_split_idx = blockIdx.x;
            bidb = blockIdx.y / h;
            bidh = blockIdx.y - bidb * h;
            return dim3(m_block, bidb, bidh);
        }
        if (blockType == 1) {
            m_block = blockIdx.y;
            n_split_idx = blockIdx.x;
            bidb = blockIdx.z / h;
            bidh = blockIdx.z - bidb * h;
            return dim3(m_block, bidb, bidh);
        }
        if (blockType == 2) {
            m_block = blockIdx.z;
            n_split_idx = blockIdx.y;
            bidb = blockIdx.x / h;
            bidh = blockIdx.x - bidb * h;
            return dim3(m_block, bidb, bidh);
        }
        if (blockType == 3) {
            m_block = blockIdx.x;
            n_split_idx = blockIdx.y;
            bidb = blockIdx.z / h;
            bidh = blockIdx.z - bidb * h;
            return dim3(m_block, bidb, bidh);
        }
        if (blockType == 4) {
            m_block = blockIdx.y;
            n_split_idx = blockIdx.z;
            bidb = blockIdx.x / h;
            bidh = blockIdx.x - bidb * h;
            return dim3(m_block, bidb, bidh);
        }
        if (blockType == 5) {
            m_block = blockIdx.x;
            n_split_idx = blockIdx.z;
            bidb = blockIdx.y / h;
            bidh = blockIdx.y - bidb * h;
            return dim3(m_block, bidb, bidh);
        }
        return dim3(m_block, bidb, bidh);
    }
    n_split_idx = 0;
    return get_bidInfo(blockType);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint64_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<64> {
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        #if 0
        constexpr int OFFSET = 32;
        x = op(x, __shfl_xor_sync(uint64_t(-1), x, OFFSET));
        x = op(x, __shfl_xor_sync(uint64_t(-1), x, OFFSET / 2));
        return x;
        #endif

        /**********************************************
         ** Using one addtional __shfl_xor_sync can
         ** reduce the time of waiting arrive inst
        **********************************************/
        auto x1 = __shfl_xor_sync(uint64_t(-1), x, 48);
        auto x2 = __shfl_xor_sync(uint64_t(-1), x, 32);
        auto x3 = __shfl_xor_sync(uint64_t(-1), x, 16);
        return op(op(op(x, x1), x2), x3);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator>
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint64_t(-1), x, 1));
    return x;
}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm_opt(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        //ToDo: remove this after compiler has been updated
        __builtin_mxc_schedbound_begin();
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
        __builtin_mxc_schedbound_end();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                    // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tensor0, typename Tensor1, typename Tensor2, typename TiledMma>
__forceinline__ __device__ void gemm_rr(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, TiledMma tiled_mma) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                    // MMA_K
    cute::gemm(tiled_mma, tCrA, tCrB, acc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// cu: Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
// mc: Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(1, MMA_M), ncol=(4, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    //auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    //return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
    return make_layout(make_layout(cute::Layout<_1>{}, get<1>(acc_layout)), make_layout(get<0>(acc_layout), get<2>(acc_layout)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    mctlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const mctlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

#define CONVERT_TENSOR_TYPE(type_s, type_d, tensor_s, tensor_d)                                                                         \
    constexpr int tensor_d##_numel = decltype(size(tensor_s))::value;                                                                   \
    mctlass::NumericArrayConverter<type_d, type_s, tensor_d##_numel > tensor_d##_convert_op;                                            \
    auto tensor_d##_frag = tensor_d##_convert_op(*reinterpret_cast<const mctlass::Array<type_s, tensor_d##_numel> *>(tensor_s.data())); \
    Tensor tensor_d = make_tensor(make_rmem_ptr<type_d>(&tensor_d##_frag), tensor_s.layout());

////////////////////////////////////////////////////////////////////////////////////////////////////

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
__forceinline__ __device__ void cp_async_wait() {
    __builtin_mxc_arrive_gvmcnt(N);
}

__forceinline__ __device__ void sync_threads() {
    __builtin_mxc_arrive_bsmcnt(0);
    __builtin_mxc_barrier_inst();
}

__forceinline__ __device__ void barrier() {
    __builtin_mxc_barrier_inst();
}

template <int N>
__forceinline__ __device__ void barrier_gvm() {
    __builtin_mxc_arrive_gvmcnt(N);
    __builtin_mxc_barrier_inst();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            const int& d, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || get<1>(identity_MN(0, 0, k)) < d) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN = true, bool Is_even_K = true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_reg_to_global(Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            const int &d, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto D_ptr = (VecType *)(reinterpret_cast<int32_t *>(D(_, m, k).data().ptr_));
            auto S_ptr = (VecType const *)(reinterpret_cast<int32_t const *>(S(_, m, k).data()));
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
            __builtin_mxc_stg_b128_predicator(D_ptr, 0, S_ptr[0], true, false, false, col_mask && row_mask, 1, MACA_ICMP_EQ);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_even_MN = true, bool Is_even_K = true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void copy_reg_to_global4x4fp32(Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, const int &d, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<2>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<2>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
    CUTE_STATIC_ASSERT_V(size<0>(S) == _16{});
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_K
    typedef __NATIVE_VECTOR__(4, int) VecType;
    constexpr int kAtomLayoutMO = Kernel_traits::kAtomLayoutMO;
    const int tidx = threadIdx.x;
    const int warp_idx = tidx / 64;
    const int lane_idx = tidx % 64;
    int row_idx = warp_idx % kAtomLayoutMO * 16 + lane_idx % 16;
    #pragma unroll
    for (int k = 0; k < size<1>(S); ++k) {
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int col_idx = lane_idx / 16 * 16 + k * 128 + m * 4;
            auto D_ptr = (VecType *)(reinterpret_cast<int32_t *>(&D(4*m, k)));
            auto S_ptr = (VecType const *)(reinterpret_cast<int32_t const *>(&S(4*m, k)));
            bool col_mask = Is_even_K || col_idx < d;
            bool row_mask = Is_even_MN || row_idx < max_MN;
            __builtin_mxc_stg_b128_predicator(D_ptr, 0, S_ptr[0], true, false, false, col_mask && row_mask, 1, MACA_ICMP_EQ);
        }
    }
}


template <bool Is_even_K = true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const &S,
                                      Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                      const int &d, const int max_MN=0, const int min_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    Tensor reg = make_fragment_like(S);
    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            bool row_mask = get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN;
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            auto src_ptr = (VecType *)(S(_, m, k).data().ptr_);
            auto reg_ptr = (VecType *)(reg(_, m, k).data());
            reg_ptr[0] = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, false, true, false, false,
                                                            col_mask && row_mask, 1, MACA_ICMP_EQ);

            auto dst_ptr = (VecType *)(D(_, m, k).data().ptr_);
            __builtin_mxc_stg_b128_predicator(dst_ptr, 0, reg_ptr[0], true, false, false, col_mask && row_mask, 1, MACA_ICMP_EQ);
        }
    }
}

template <typename T>
__forceinline__ __device__ void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

// for tensor shape is (cols=8, m, k).
template <bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_b128(Tensor<Engine0, Layout0> const &S,
                                          Tensor<Engine1, Layout1> &D,
                                          Tensor<Engine2, Layout2> const &identity_MN,
                                          const int d,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S(_, m, k).data().get());    // gmem
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b128(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, true, true, false, false,
                                                         row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}

// for tensor shape is (cols=4, m, k).
template <bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2>
__forceinline__ __device__ void copy_b64(Tensor<Engine0, Layout0> const &S,
                                          Tensor<Engine1, Layout1> &D,
                                          Tensor<Engine2, Layout2> const &identity_MN,
                                          const int d,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    typedef __NATIVE_VECTOR__(2, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S(_, m, k).data().get());    // gmem
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b64(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, false,
                                                            row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}

template <typename Engine, typename Layout>
__forceinline__ __device__ void swap_fragment(Tensor<Engine, Layout> &S) {
    using data_type = typename Engine::value_type;
    static_assert(decltype(size<0>(S))::value == 8);
    static_assert(std::is_same_v<data_type, mctlass::half_t> || std::is_same_v<data_type, mctlass::bfloat16_t>);

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        #pragma unroll
        for (int n = 0; n < size<2>(S); ++n) {
            uint64_t *first = reinterpret_cast<uint64_t *>(S(_, m, n).data());
            uint64_t *second = first + 1;
            uint64_t tmp = *first;
            *first = *second;
            *second = tmp;
        }
    }
}

#define SWIZZLE_STORE_QDO(smem_s, reg, smem_d)      \
    cute::copy(smem_s, reg);                        \
    if (tidx / 8 % 2 == 1) {                        \
        flash::swap_fragment(reg);                  \
    }                                               \
    cute::copy(reg, smem_d);

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_softcap(Tensor<Engine, Layout> &tensor, const float softcap){
    // #pragma unroll
    // for (int i = 0; i < size(tensor); ++i) {
    //     tensor(i) = mctlass::fast_tanh(tensor(i) * softcap);
    // }
    static_assert(decltype(size(tensor))::value % 2 == 0);
    typedef __NATIVE_VECTOR__(2, float) Float2;
    Float2 scale_vec = {softcap, softcap};
    Float2 beta_vec = {0.0f, 0.0f};
    #pragma unroll
    for (int i = 0; i < size(tensor); i += 2) {
        // tensor(i) = mctlass::fast_tanh(tensor(i) * softcap);
        Float2 x_vec = {tensor(i), tensor(i + 1)};
        x_vec = __builtin_mxc_pk_fma_f32(x_vec, scale_vec, beta_vec);
        tensor(i) = mctlass::fast_tanh(x_vec[0]);
        tensor(i + 1) = mctlass::fast_tanh(x_vec[1]);
    }
}



// resolves offset of a slice of a paged kv copy from gmem.
// assumes that the tensor has already been positioned at the correct head.
__forceinline__ __device__
int resolve_thread_kv_page_slice_offset(const int page_block_size, const int* block_table, const int page_stride, const int row_stride, const int row_offset, const int col_offset) {
    const int virtual_page_idx = row_offset / page_block_size;
    const int page_offset = row_offset - virtual_page_idx * page_block_size;

    return block_table[virtual_page_idx] * page_stride
        + page_offset * row_stride
        + col_offset;
}



template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_b128_page_one(Tensor<Engine0, Layout0> const &S_base,
                                          Tensor<Engine1, Layout1> &S,
                                          Tensor<Engine2, Layout2> &D,
                                          Tensor<Engine3, Layout3> const &identity_MN,
                                          const int d,
                                          const int n_block,
                                          const int *block_table,
                                          const int page_stride,
                                          const int row_stride,
                                          const int page_block_size,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kElementPerThread = 8;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kBlockKSmem / kElementPerThread;
    constexpr int kGmemRowsPerThread = 1;
    // load 1x8 per thread
    int tidx = threadIdx.x;

    typedef __NATIVE_VECTOR__(4, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        const int row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread + kNThreads / kGmemThreadsPerRow * m + n_block * kBlockN;
        const int col_offset = tidx % kGmemThreadsPerRow * kElementPerThread;
        const int global_kv_page_offset = flash::resolve_thread_kv_page_slice_offset(page_block_size, block_table, page_stride, row_stride, row_offset, col_offset);
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S_base.data().get() + global_kv_page_offset + get<2>(S.stride()) * k);
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b128(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, true, true, false, false,
                                                         row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_b64_page_one(Tensor<Engine0, Layout0> const &S_base,
                                          Tensor<Engine1, Layout1> &S,
                                          Tensor<Engine2, Layout2> &D,
                                          Tensor<Engine3, Layout3> const &identity_MN,
                                          const int d,
                                          const int n_block,
                                          const int *block_table,
                                          const int page_stride,
                                          const int row_stride,
                                          const int page_block_size,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kElementPerThread = 4;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kBlockKSmem / kElementPerThread;
    constexpr int kGmemRowsPerThread = 1;
    // load 1x4 per thread
    int tidx = threadIdx.x;

    typedef __NATIVE_VECTOR__(2, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        const int row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread + kNThreads / kGmemThreadsPerRow * m + n_block * kBlockN;
        const int col_offset = tidx % kGmemThreadsPerRow * kElementPerThread;
        const int global_kv_page_offset = flash::resolve_thread_kv_page_slice_offset(page_block_size, block_table, page_stride, row_stride, row_offset, col_offset);
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S_base.data().get() + global_kv_page_offset + get<2>(S.stride()) * k);
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b64(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, false,
                                                         row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}

template <typename Kernel_traits, bool Is_even_MN=true, bool Is_even_K=true, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_b32_page_one(Tensor<Engine0, Layout0> const &S_base,
                                          Tensor<Engine1, Layout1> &S,
                                          Tensor<Engine2, Layout2> &D,
                                          Tensor<Engine3, Layout3> const &identity_MN,
                                          const int d,
                                          const int n_block,
                                          const int *block_table,
                                          const int page_stride,
                                          const int row_stride,
                                          const int page_block_size,
                                          const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kElementPerThread = 2;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kBlockKSmem / kElementPerThread;
    constexpr int kGmemRowsPerThread = 1;
    // load 1x2 per thread
    int tidx = threadIdx.x;

    typedef __NATIVE_VECTOR__(1, int) VecType;
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        bool row_mask = Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN;
        const int row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread + kNThreads / kGmemThreadsPerRow * m + n_block * kBlockN;
        const int col_offset = tidx % kGmemThreadsPerRow * kElementPerThread;
        const int global_kv_page_offset = flash::resolve_thread_kv_page_slice_offset(page_block_size, block_table, page_stride, row_stride, row_offset, col_offset);
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
            auto src_ptr = (VecType *)(S_base.data().get() + global_kv_page_offset + get<2>(S.stride()) * k);
            auto dst_ptr = (VecType *)(D(_, m, k).data());          // rf
            bool col_mask = Is_even_K || get<1>(identity_MN(0, 0, k)) < d;
            if constexpr (Is_even_MN && Is_even_K) {
                *dst_ptr = __builtin_mxc_ldg_b32(src_ptr, 0, -1, true, true, false, false);
            } else {
                *dst_ptr = __builtin_mxc_ldg_b32_predicator(src_ptr, 0, true, true, false, false,
                                                         row_mask && col_mask, 1, MACA_ICMP_EQ);
            }
        }
    }
}

template<typename Tensor0, typename Tensor1, typename Tensor2>
__forceinline__ __device__ void concat(Tensor0 &lhs, Tensor1 &rhs, Tensor2 &out) {
    CUTE_STATIC_ASSERT_V(rank(lhs) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(rhs) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(out) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(lhs) == size<0>(rhs));                     // MMA
    CUTE_STATIC_ASSERT_V(size<0>(lhs) == size<0>(out));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(lhs) == size<1>(rhs));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(lhs) == size<1>(out));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(lhs) + size<2>(rhs) == size<2>(out));      // MMA_K
    #pragma unroll
    for (int k = 0; k < size<2>(lhs); k++) {
        #pragma unroll
        for (int m = 0; m < size<1>(out); m++) {
            #pragma unroll
            for (int i = 0; i < size<0>(out); i++) {
                out(i, m, k) = lhs(i, m, k);
            }
        }
    }

    #pragma unroll
    for (int k = 0; k < size<2>(rhs); k++) {
        #pragma unroll
        for (int m = 0; m < size<1>(out); m++) {
            #pragma unroll
            for (int i = 0; i < size<0>(out); i++) {
                out(i, m, k + size<2>(lhs)) = rhs(i, m, k);
            }
        }
    }

}

template<typename Tensor0, typename Tensor1>
__forceinline__ __device__ void lds4x4_with_swizzle424(Tensor0 const& tCsA, Tensor1& tCrA) {
    CUTE_STATIC_ASSERT_V(size<0>(tCsA) == size<0>(tCrA));
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == (size<1, 1>(tCrA)));
    CUTE_STATIC_ASSERT_V((size<1, 0>(tCrA)) == _4{});
    const int lane_idx = threadIdx.x % 64;
    const int Vt_swizzle_row = lane_idx / 16 * 4;
    const int Vt_swizzle_col = lane_idx % 16;

    #pragma unroll
    for (int m = 0; m < size<1>(tCsA); m++) {
        uint64_t* src_ptr = reinterpret_cast<uint64_t *>(&tCsA(0, m));
        #pragma unroll
        for (int row = 0; row < 4; row++) {
            int col_idx = Vt_swizzle_col ^ (Vt_swizzle_row + row);
            uint64_t* dst_ptr = reinterpret_cast<uint64_t *>(&tCrA(0, make_coord(row, m), 0));
            *dst_ptr = *(src_ptr + row * 16 + col_idx);
        }
    }
}


// resolves offset of a slice of a paged kv copy from gmem.
// assumes that the tensor has already been positioned at the correct head.
template <typename Kernel_traits>
__forceinline__ __device__
int resolve_thread_kv_page_slice_offset(const int tidx, const int n_block_max, const int page_block_size,
                            const int* block_table, const int page_stride, const int row_stride, const int row_idx = 0) {
    constexpr int kGmemThreadsPerRow = Kernel_traits::kGmemThreadsPerRow;
    constexpr int kGmemRowsPerThread = Kernel_traits::kGmemRowsPerThread;
    constexpr int kGmemElemsPerLoad = Kernel_traits::kGmemElemsPerLoad;
    constexpr int kBlockN = Kernel_traits::kBlockN;

    const int col_offset = tidx % kGmemThreadsPerRow * kGmemElemsPerLoad;
    const int block_row_offset = tidx / kGmemThreadsPerRow * kGmemRowsPerThread;
    const int global_row_offset = block_row_offset + (n_block_max - 1) * kBlockN;
    const int page_offset = global_row_offset % page_block_size;
    const int virtual_page_idx = global_row_offset / page_block_size + row_idx;

    return block_table[virtual_page_idx] * page_stride
        + page_offset * row_stride
        + col_offset;
}

template <typename Engine, typename Layout>
__forceinline__ __device__ decltype(auto) permute_4x4_b16(Tensor<Engine, Layout> &t) {
    using data_type = typename Engine::value_type;
    Tensor tPerm = make_tensor<data_type>(Shape<_4, _4>{});
    uint32_t v1, v2;
    uint32_t *dest;

    #pragma unroll
    for (int i = 0; i < size<2>(t); ++i) {
        v1 = *(reinterpret_cast<uint32_t *>(t(_, 0, i).data()));
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 1, i).data()));
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 0).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 1).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 0, i).data()) + 1);
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 1, i).data()) + 1);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 2).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 3).data());
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 2, i).data()));
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 3, i).data()));
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 0).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 1).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        v1 = *(reinterpret_cast<uint32_t *>(t(_, 2, i).data()) + 1);
        v2 = *(reinterpret_cast<uint32_t *>(t(_, 3, i).data()) + 1);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 2).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x05040100);
        dest = reinterpret_cast<uint32_t *>(tPerm(_, 3).data()) + 1;
        *dest = __builtin_mxc_byte_perm(v2, v1, 0x07060302);

        cute::copy(tPerm, t(_, _, i));
    }
    return t;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ mcDeviceProp_t mcGetCurrentDeviceProperties() {
    int deviceId{};
    mcGetDevice(&deviceId);
    mcDeviceProp_t dprops;
    mcGetDeviceProperties(&dprops, deviceId);
    return dprops;
}

// solving the bank conflict of swizzle<2,3,3>, match with offset_swz233
template <typename Engine0, typename Layout0>
__forceinline__ __device__ void swap_swz233(Tensor<Engine0, Layout0> &t) {
    if (__lane_id() >= 32) {
        flash::swap_fragment(t);
    }
}

// solving the bank conflict of swizzle<3,3,4>, match with offset_swz334. todo: need to combine with SWIZZLE_STORE_QDO
template <typename Engine0, typename Layout0>
__forceinline__ __device__ void swap_swz334(Tensor<Engine0, Layout0> &t) {
    if (__lane_id() / 8 % 2 == 1) {
        flash::swap_fragment(t);
    }
}

#define UNPACK_GRID(bidx, bidh, bidb, type)                                             \
    if (type == 0)         {bidx =  blockIdx.x; bidh =  blockIdx.y; bidb = blockIdx.z;} \
    else if (type == 1)    {bidx =  blockIdx.x; bidh =  blockIdx.z; bidb = blockIdx.y;} \
    else if (type == 2)    {bidx =  blockIdx.y; bidh =  blockIdx.x; bidb = blockIdx.z;} \
    else if (type == 3)    {bidx =  blockIdx.y; bidh =  blockIdx.z; bidb = blockIdx.x;} \
    else if (type == 4)    {bidx =  blockIdx.z; bidh =  blockIdx.y; bidb = blockIdx.x;} \
    else if (type == 5)    {bidx =  blockIdx.z; bidh =  blockIdx.x; bidb = blockIdx.y;} \
    else                   {bidx =  blockIdx.x; bidh =  blockIdx.y; bidb = blockIdx.z;} \

#define CHECK_MSG(x, ...) do { if((x) == false) {throw std::invalid_argument(__VA_ARGS__);} }while(0)
#define CUDA_CHECK(expr) {auto x = (expr); CHECK_MSG(x == cudaSuccess, #expr + std::string(" check failed!"));}
#define CUDA_KERNEL_LAUNCH_CHECK() CUDA_CHECK(cudaGetLastError())

}  // namespace flash
