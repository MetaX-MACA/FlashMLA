// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)

#pragma once

#include "flash_headdim.h"

constexpr static int Headdim32 = FlashHeaddim<32>::Headdim;
constexpr static int Headdim64 = FlashHeaddim<64>::Headdim;
constexpr static int Headdim96 = FlashHeaddim<96>::Headdim;
constexpr static int Headdim128 = FlashHeaddim<128>::Headdim;
constexpr static int Headdim160 = FlashHeaddim<160>::Headdim;
constexpr static int Headdim192 = FlashHeaddim<192>::Headdim;
constexpr static int Headdim224 = FlashHeaddim<224>::Headdim;
constexpr static int Headdim256 = FlashHeaddim<256>::Headdim;
constexpr static int Headdim512 = FlashHeaddim<512>::Headdim;

#ifndef MCFLASHINFER

#define FWD_HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
    if (HEADDIM <= 32) {                   \
      constexpr static int kHeadDim = Headdim32;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 64) {            \
      constexpr static int kHeadDim = Headdim64;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 96) {            \
      constexpr static int kHeadDim = Headdim96;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 128) {           \
      constexpr static int kHeadDim = Headdim128; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 160) {           \
      constexpr static int kHeadDim = Headdim160; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 192) {           \
      constexpr static int kHeadDim = Headdim192; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 224) {           \
      constexpr static int kHeadDim = Headdim224; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 256) {           \
      constexpr static int kHeadDim = Headdim256; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 512) {           \
      constexpr static int kHeadDim = Headdim512; \
      return __VA_ARGS__();                \
    }                                      \
  }()

#define BWD_HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
    if (HEADDIM <= 32) {                   \
      constexpr static int kHeadDim = Headdim32;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 64) {            \
      constexpr static int kHeadDim = Headdim64;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 96) {            \
      constexpr static int kHeadDim = Headdim96;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 128) {           \
      constexpr static int kHeadDim = Headdim128; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 160) {           \
      constexpr static int kHeadDim = Headdim160; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 192) {           \
      constexpr static int kHeadDim = Headdim192; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 224) {           \
      constexpr static int kHeadDim = Headdim224; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 256) {           \
      constexpr static int kHeadDim = Headdim256; \
      return __VA_ARGS__();                \
    }                                      \
  }()

#else
#define FWD_HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
    if (HEADDIM <= 64) {                   \
      constexpr static int kHeadDim = Headdim64;  \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 128) {           \
      constexpr static int kHeadDim = Headdim128; \
      return __VA_ARGS__();                \
    } else if (HEADDIM <= 256) {           \
      constexpr static int kHeadDim = Headdim256; \
      return __VA_ARGS__();                \
    }                                      \
  }()
#endif
