// Inspired by
// https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

#pragma once

/// CONST_PRECOND && COND
#define BOOL_SWITCH_AND_CONST_PRECOND(CONST_PRECOND, COND, CONST_NAME, ...) \
  [&] {                                                                      \
    if constexpr (CONST_PRECOND) {                                           \
      if (COND) {                                                            \
        constexpr static bool CONST_NAME = true;                             \
        return __VA_ARGS__();                                                \
      } else {                                                               \
        constexpr static bool CONST_NAME = false;                            \
        return __VA_ARGS__();                                                \
      }                                                                      \
    } else {                                                                 \
      constexpr static bool CONST_NAME = false;                              \
      return __VA_ARGS__();                                                  \
    }                                                                        \
  }()

/// CONST_PRECOND || COND
#define BOOL_SWITCH_OR_CONST_PRECOND(CONST_PRECOND, COND, CONST_NAME, ...)   \
  [&] {                                                                      \
    if constexpr (CONST_PRECOND) {                                           \
      constexpr static bool CONST_NAME = true;                               \
      return __VA_ARGS__();                                                  \
    } else {                                                                 \
      if (COND) {                                                            \
        constexpr static bool CONST_NAME = true;                             \
        return __VA_ARGS__();                                                \
      } else {                                                               \
        constexpr static bool CONST_NAME = false;                            \
        return __VA_ARGS__();                                                \
      }                                                                      \
    }                                                                        \
  }()

/// @param COND       - a boolean expression to switch by
/// @param CONST_NAME - a name given for the constexpr bool variable.
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// BOOL_SWITCH(flag, BoolConst, [&] {
///     some_function<BoolConst>(...);
/// });
/// ```

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  BOOL_SWITCH_AND_CONST_PRECOND(true, COND, CONST_NAME, __VA_ARGS__)

#ifdef FLASHATTENTION_DISABLE_DROPOUT
  #define DROPOUT_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define DROPOUT_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_SOFTCAP
  #define SOFTCAP_SWITCH(CONST_PRECOND, COND, CONST_NAME) \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define SOFTCAP_SWITCH BOOL_SWITCH_AND_CONST_PRECOND
#endif

#ifdef FLASHATTENTION_DISABLE_ALIBI
  #define ALIBI_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;   \
    return __VA_ARGS__();                       \
  }()
#else
  #define ALIBI_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  #define EVENK_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = true;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define EVENK_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_LOCAL
  #define LOCAL_SWITCH(COND, CONST_NAME, ...)   \
  [&] {                                         \
    constexpr static bool CONST_NAME = false;    \
    return __VA_ARGS__();                       \
  }()
#else
  #define LOCAL_SWITCH BOOL_SWITCH
#endif

#ifdef FLASHATTENTION_DISABLE_LOCAL
  #define LOCAL_SWITCH(COND, CONST_NAME, ...)   \
  BOOL_SWITCH_AND_CONST_PRECOND(false, COND, CONST_NAME, __VA_ARGS__)
#else
  #define LOCAL_SWITCH_AND_CONST_PRECOND BOOL_SWITCH_AND_CONST_PRECOND
#endif

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = mctlass::half_t;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = mctlass::bfloat16_t; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

// #define HEADDIM_SWITCH(HEADDIM, ...)   \
//   [&] {                                    \
//     if (HEADDIM <= 32) {                   \
//       constexpr static int kHeadDim = 32;  \
//       return __VA_ARGS__();                \
//     } else if (HEADDIM <= 64) {            \
//       constexpr static int kHeadDim = 64;  \
//       return __VA_ARGS__();                \
//     } else if (HEADDIM <= 96) {            \
//       constexpr static int kHeadDim = 96;  \
//       return __VA_ARGS__();                \
//     } else if (HEADDIM <= 128) {           \
//       constexpr static int kHeadDim = 128; \
//       return __VA_ARGS__();                \
//     } else if (HEADDIM <= 160) {           \
//       constexpr static int kHeadDim = 160; \
//       return __VA_ARGS__();                \
//     } else if (HEADDIM <= 192) {           \
//       constexpr static int kHeadDim = 192; \
//       return __VA_ARGS__();                \
//     } else if (HEADDIM <= 224) {           \
//       constexpr static int kHeadDim = 224; \
//       return __VA_ARGS__();                \
//     } else if (HEADDIM <= 256) {           \
//       constexpr static int kHeadDim = 256; \
//       return __VA_ARGS__();                \
//     }                                      \
//   }()

#define ROWNUM_SWITCH(COND, CONST_NAME, ...) \
  [&] {                                      \
    if (COND)                                \
    {                                        \
      constexpr static int CONST_NAME = 2;   \
      return __VA_ARGS__();                  \
    }                                        \
    else                                     \
    {                                        \
      constexpr static int CONST_NAME = 1;   \
      return __VA_ARGS__();                  \
    }                                        \
  }()
