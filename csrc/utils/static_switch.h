// Adapted from Dao-AILab/flash-attention (https://github.com/Dao-AILab/flash-attention/tree/v2.6.3)
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

#define NUMSPLITS_SWITCH(NUMSPLITS, ...)       \
  [&] {                                        \
    if (NUMSPLITS <= 2) {                      \
      constexpr static int kLogMaxSplits = 1;  \
      return __VA_ARGS__();                    \
    } else if (NUMSPLITS <= 4) {               \
      constexpr static int kLogMaxSplits = 2;  \
      return __VA_ARGS__();                    \
    } else if (NUMSPLITS <= 8) {               \
      constexpr static int kLogMaxSplits = 3;  \
      return __VA_ARGS__();                    \
    } else if (NUMSPLITS <= 16) {              \
      constexpr static int kLogMaxSplits = 4;  \
      return __VA_ARGS__();                    \
    } else if (NUMSPLITS <= 32) {              \
      constexpr static int kLogMaxSplits = 5;  \
      return __VA_ARGS__();                    \
    } else if (NUMSPLITS <= 64) {              \
      constexpr static int kLogMaxSplits = 6;  \
      return __VA_ARGS__();                    \
    } else if (NUMSPLITS <= 128) {             \
      constexpr static int kLogMaxSplits = 7;  \
      return __VA_ARGS__();                    \
    }                                          \
  }()

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
