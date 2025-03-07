// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)

/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/algorithm/copy.hpp"

#include "mctlass/mctlass.h"
#include "mctlass/layout/layout.h"
#include <mctlass/numeric_types.h>

using namespace cute;

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=mctlass::half_t>
struct Flash_kernel_traits {

#if defined(__MACA_ARCH__)
    using Element = elem_type;
    static constexpr bool Has_cp_async = false;
#else
    using Element = mctlass::half_t;
    static constexpr bool Has_cp_async = false;
#endif

    using ElementAccum = float;
    using index_t = int64_t;

#if defined(__MACA_ARCH__)
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, mctlass::half_t>,
        MMA_Atom<MACA_16x16x16_F32F16F16F32>,
        MMA_Atom<MACA_16x16x16_F32BF16BF16F32>
    >;
    using ValLayoutMNK = Layout<Shape<_1, _1, _1>>;
#else
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
    using ValLayoutMNK = Layout<Shape<_1, _2, _2>>;
#endif

    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyB64 = Copy_Atom<UniversalCopy<uint64_t>, elem_type>;
    using UniversalCopyAtom32 = Copy_Atom<UniversalCopy<uint32_t>, elem_type>;
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, typename elem_type=mctlass::half_t, bool Is_Splits_=false,
         int kHeadDimV_=kHeadDim_, typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;

    using SmemCopyAtomB64 = typename Base::SmemCopyB64;
    using UniversalCopyAtom32 = typename Base::UniversalCopyAtom32;

    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 64;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHeadDimV = kHeadDimV_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKSmemV = kHeadDimV % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    static constexpr int MBase = 3;
    static constexpr int SShift = 3;
    static constexpr int SShift_OPT = kBlockKSmem == 32 ? 3 : 4;    // for bank conflict free
    static constexpr int kAtomLayoutMS = std::min(kBlockM / 16, kNWarps);
    static constexpr int kAtomLayoutMO = 2;

    using TiledMmaS = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kAtomLayoutMS>,_1,_1>>,  // 2x1x1
        typename Base::ValLayoutMNK>;

    using TiledMmaO = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kAtomLayoutMO>,Int<kNWarps / kAtomLayoutMO>,_1>>,  // 2x2x1
        typename Base::ValLayoutMNK>;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, MBase, SShift>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutAtomK = decltype(
        composition(Swizzle<4, 2, 4>{},
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDimV>>{}));

    // This has to be kBlockN and not 8, otherwise we get wrong results for d=128
    using SmemLayoutAtomVtransposedNoSwizzle = Layout<Shape<Int<kBlockKSmemV>, Int<kBlockN>>,
                                                      Stride<_1, Int<kBlockKSmemV>>>;
    using SmemLayoutAtomVtransposed = decltype(
        composition(Swizzle<4, 2, 4>{}, SmemLayoutAtomVtransposedNoSwizzle{}));
    using SmemLayoutVtransposed = decltype(tile_to_shape(
        SmemLayoutAtomVtransposed{},
        Shape<Int<kHeadDimV>, Int<kBlockN>>{}));
    // Maybe the VtransposeNoSwizzle just needs to have the right shape
    // And the strides don't matter?
    using SmemLayoutVtransposedNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomVtransposedNoSwizzle{},
        Shape<Int<kHeadDimV>, Int<kBlockN>>{}));

    using SmemLayoutVtNoSwizzle = decltype(tile_to_shape(
        Layout<Shape<_16, Int<kBlockKSmemV>>,
               Stride<Int<kBlockKSmemV>, _1>>{},
        make_shape(Int<kBlockN>{}, Int<kHeadDimV>{})));

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, MBase, SShift>{},
                    Layout<Shape<Int<16>, Int<kBlockKSmemV>>,
                           Stride<Int<kBlockKSmemV>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<UniversalCopy<uint64_t>, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<UniversalCopy<uint128_t>, ElementAccum>;

 static constexpr int kBlockKSmemMask = kBlockN % 64 == 0 ? 64 : (kBlockN % 32 == 0 ? 32: 16);
    static constexpr int kSwizzleMask = kBlockKSmemMask == 32 ? 3 : 4;
    using SmemLayoutAtomMask = decltype(
        composition(Swizzle<kSwizzleMask, 2, kSwizzleMask>{},
                    Layout<Shape<Int<kBlockM>, Int<kBlockKSmemMask>>,
                           Stride<Int<kBlockKSmemMask>, _1>>{}));
    using SmemLayoutMask = decltype(tile_to_shape(
        SmemLayoutAtomMask{},
        Shape<Int<kBlockM>, Int<kBlockN>>{}));

    static constexpr int kSmemMaskSize = size(SmemLayoutMask{}) * sizeof(Element);
    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKSize = size(SmemLayoutK{}) * sizeof(Element);
    static constexpr int kSmemVSize = size(SmemLayoutV{}) * sizeof(Element);
    static constexpr int kSmemKVSize = kSmemKSize + kSmemVSize;
    static constexpr int kSmemSize = Share_Q_K_smem ? std::max((Is_Splits_ ? 2 : 1) * kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize;
    static constexpr int kRegSize = kSmemSize / sizeof(uint32_t) / kNThreads;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kGmemElemsPerLoadB64 = sizeof(cute::uint64_t) / sizeof(Element);
    static constexpr int kGmemElemsQuantPerLoadB64 = sizeof(cute::uint64_t) / sizeof(int8_t);

    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static_assert(kHeadDim % kGmemElemsPerLoadB64 == 0, "kHeadDim must be a multiple of kGmemElemsPerLoadB64");
    static_assert(kHeadDimV % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");

    // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
    // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
    // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
    // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
    // to the same banks.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static constexpr int kGmemThreadsPerRowB64 = kBlockKSmem / kGmemElemsPerLoadB64;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemLayoutAtomB64 = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowB64>, Int<kGmemThreadsPerRowB64>>,
                                  Stride<Int<kGmemThreadsPerRowB64>, _1>>;

    static constexpr int kGmemThreadsPerRowV = kBlockKSmemV / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRowV == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtomV = Layout<Shape <Int<kNThreads / kGmemThreadsPerRowV>, Int<kGmemThreadsPerRowV>>,
                                   Stride<Int<kGmemThreadsPerRowV>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using GmemTiledCopyB128 = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    using GmemTiledCopyB64 = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomB64{},
                        Layout<Shape<_1, _4>>{}));  // Val layout, 8 vals per read
    static constexpr bool UseWarpsNx1 = kBlockN >= 16 * kNWarps;
    // from how many rows does each thread have to fetch
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);

    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, Element>{},
                        GmemLayoutAtomV{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <Int<kNThreads / 8>, _8>,  // Thread layout, 8 threads per row
               Stride< _8, _1>>,
        Layout<Shape <Int<kNThreads / 16>, _16>,  // Thread layout, 16 threads per row
               Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
    using GmemTiledCopyRotcossinPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _4>, Stride<_4, _1>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinContPaged = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));  // Val layout, 8 vals per load
};
////////////////////////////////////////////////////////////////////////////////////////////////////
