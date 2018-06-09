/*
 * osdep.h
 *
 * Description of this file:
 *    platform-specific code functions definition of the xavs2 library
 *
 * --------------------------------------------------------------------------
 *
 *    xavs2 - video encoder of AVS2/IEEE1857.4 video coding standard
 *    Copyright (C) 2018~ VCL, NELVT, Peking University
 *
 *    Authors: Falei LUO <falei.luo@gmail.com>
 *             etc.
 *
 *    Homepage1: http://vcl.idm.pku.edu.cn/xavs2
 *    Homepage2: https://github.com/pkuvcl/xavs2
 *    Homepage3: https://gitee.com/pkuvcl/xavs2
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *
 *    This program is also available under a commercial proprietary license.
 *    For more information, contact us at sswang @ pku.edu.cn.
 */

#ifndef XAVS2_OSDEP_H
#define XAVS2_OSDEP_H


/**
 * ===========================================================================
 * includes
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * disable warning C4996: functions or variables may be unsafe.
 */
#if defined(_MSC_VER)
#define WIN32_LEAN_AND_MEAN
#define _CRT_NONSTDC_NO_DEPRECATE
#define _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_WARNINGS
#include <intrin.h>
#include <windows.h>
#endif

#define _LARGEFILE_SOURCE       1
#define _FILE_OFFSET_BITS       64
#if defined(__ICL) || defined(_MSC_VER)
#include "configw.h"
#else
#include "config.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#if HAVE_STDINT_H
#include <stdint.h>
#else
#include <inttypes.h>
#endif

#if defined(__INTEL_COMPILER)
#include <mathimf.h>
#else
#include <math.h>
#endif

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#include <float.h>
#endif


/* ---------------------------------------------------------------------------
 * disable warning C4100: unreferenced formal parameter
 */
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#define UNUSED_PARAMETER(P)     (P)   /* same as UNREFERENCED_PARAMETER */
#else
#define UNUSED_PARAMETER(P)
#endif


/**
 * ===========================================================================
 * const defines
 * ===========================================================================
 */
/* ---------------------------------------------------------------------------
 * Specifies the number of bits per pixel that xavs2 encoder uses. This is also the
 * bit depth that xavs2 encoder encodes in. If this value is > 8, xavs2 encoder will read
 * two bytes of input data for each pixel sample, and expect the upper
 * (16-XAVS2_BIT_DEPTH) bits to be zero.
 * Note: The flag XAVS2_CSP_HIGH_DEPTH must be used to specify the
 * colorspace depth as well.
 */
#define XAVS2_BIT_DEPTH       BIT_DEPTH

#define WORD_SIZE               sizeof(void*)
#define asm                     __asm__


/**
 * ===========================================================================
 * const defines
 * ===========================================================================
 */

#if defined(__GNUC__) && (__GNUC__ > 3 || __GNUC__ == 3 && __GNUC_MINOR__ > 0)
#define UNINIT(x)               x = x
#define UNUSED                  __attribute__((unused))
#define ALWAYS_INLINE           __attribute__((always_inline)) inline
#define NOINLINE                __attribute__((noinline))
#define MAY_ALIAS               __attribute__((may_alias))
#define xavs2_constant_p(x)   __builtin_constant_p(x)
#define xavs2_nonconstant_p(x)    (!__builtin_constant_p(x))
#define INLINE                  __inline
#else
#define UNINIT(x)               x
#if defined(__ICL)
#define ALWAYS_INLINE           __forceinline
#define NOINLINE                __declspec(noinline)
#else
#define ALWAYS_INLINE           INLINE
#define NOINLINE
#endif
#define UNUSED
#define MAY_ALIAS
#define xavs2_constant_p(x)       0
#define xavs2_nonconstant_p(x)    0
#endif

#if defined(__ICL) || defined(_MSC_VER)
#define INLINE                  __inline
#define strcasecmp              _stricmp
#define strncasecmp             _strnicmp
#if !HAVE_POSIXTHREAD
#define strtok_r                strtok_s
#endif
#define S_ISREG(x)              (((x) & S_IFMT) == S_IFREG)
#endif

#if (defined(__GNUC__) || defined(__INTEL_COMPILER)) && (ARCH_X86 || ARCH_X86_64)
#ifndef HAVE_X86_INLINE_ASM
#define HAVE_X86_INLINE_ASM     1
#endif
#endif

/* ---------------------------------------------------------------------------
 * align
 */
/* align a pointer */
#  define CACHE_LINE_SIZE       32    /* for x86-64 and x86 */
#  define ALIGN_POINTER(p)      (p) = (uint8_t *)((intptr_t)((p) + (CACHE_LINE_SIZE - 1)) & (~(intptr_t)(CACHE_LINE_SIZE - 1)))
#  define CACHE_LINE_256B       32    /* for x86-64 and x86 */
#  define ALIGN_256_PTR(p)      (p) = (uint8_t *)((intptr_t)((p) + (CACHE_LINE_256B - 1)) & (~(intptr_t)(CACHE_LINE_256B - 1)))

#if defined(_MSC_VER)
#pragma warning(disable:4324)   /* disable warning C4324: 由于 __declspec(align())，结构被填充 */
#define DECLARE_ALIGNED(var, n) __declspec(align(n)) var
#else
#define DECLARE_ALIGNED(var, n) var __attribute__((aligned(n)))
#endif
#define ALIGN32(var)            DECLARE_ALIGNED(var, 32)
#define ALIGN16(var)            DECLARE_ALIGNED(var, 16)
#define ALIGN8(var)             DECLARE_ALIGNED(var, 8)


// ARM compiliers don't reliably align stack variables
// - EABI requires only 8 byte stack alignment to be maintained
// - gcc can't align stack variables to more even if the stack were to be correctly aligned outside the function
// - armcc can't either, but is nice enough to actually tell you so
// - Apple gcc only maintains 4 byte alignment
// - llvm can align the stack, but only in svn and (unrelated) it exposes bugs in all released GNU binutils...

#define ALIGNED_ARRAY_EMU( mask, type, name, sub1, ... )\
    uint8_t name##_u [sizeof(type sub1 __VA_ARGS__) + mask]; \
    type (*name) __VA_ARGS__ = (void*)((intptr_t)(name##_u+mask) & ~mask)

#if ARCH_ARM && SYS_MACOSX
#define ALIGNED_ARRAY_8( ... ) ALIGNED_ARRAY_EMU( 7, __VA_ARGS__ )
#else
#define ALIGNED_ARRAY_8( type, name, sub1, ... ) \
    ALIGN8( type name sub1 __VA_ARGS__ )
#endif

#if ARCH_ARM
#define ALIGNED_ARRAY_16( ... ) ALIGNED_ARRAY_EMU( 15, __VA_ARGS__ )
#else
#define ALIGNED_ARRAY_16( type, name, sub1, ... ) \
    ALIGN16( type name sub1 __VA_ARGS__ )
#endif

#define EXPAND(x)               x

#if defined(STACK_ALIGNMENT) && STACK_ALIGNMENT >= 32
#define ALIGNED_ARRAY_32( type, name, sub1, ... ) \
    ALIGN32( type name sub1 __VA_ARGS__ )
#else
#define ALIGNED_ARRAY_32(...)   EXPAND(ALIGNED_ARRAY_EMU(31, __VA_ARGS__))
#endif

#define ALIGNED_ARRAY_64(...)   EXPAND(ALIGNED_ARRAY_EMU(63, __VA_ARGS__))

/* For AVX2 */
#if ARCH_X86 || ARCH_X86_64
#define NATIVE_ALIGN            32
#define ALIGNED_N               ALIGN32
#define ALIGNED_ARRAY_N         ALIGNED_ARRAY_32
#else
#define NATIVE_ALIGN            16
#define ALIGNED_N               ALIGN16
#define ALIGNED_ARRAY_N         ALIGNED_ARRAY_16
#endif


/* ---------------------------------------------------------------------------
 * threads
 */
#if HAVE_BEOSTHREAD
#include <kernel/OS.h>
#define xavs2_thread_t       thread_id
static int ALWAYS_INLINE
xavs2_thread_create(xavs2_thread_t *t, void *a, void *(*f)(void *), void *d)
{
    *t = spawn_thread(f, "", 10, d);
    if (*t < B_NO_ERROR) {
        return -1;
    }
    resume_thread(*t);
    return 0;
}

#define xavs2_thread_join(t,s) \
{\
    long tmp; \
    wait_for_thread(t,(s)?(long*)(*(s)):&tmp);\
}

#elif HAVE_POSIXTHREAD
#if defined(_MSC_VER) || defined(__ICL)
#if _MSC_VER >= 1900
#define HAVE_STRUCT_TIMESPEC    1       /* for struct timespec */
#endif
#pragma comment(lib, "pthread_lib.lib")
#endif
#include <pthread.h>
#define xavs2_thread_t                   pthread_t
#define xavs2_thread_create              pthread_create
#define xavs2_thread_join                pthread_join
#define xavs2_thread_mutex_t             pthread_mutex_t
#define xavs2_thread_mutex_init          pthread_mutex_init
#define xavs2_thread_mutex_destroy       pthread_mutex_destroy
#define xavs2_thread_mutex_lock          pthread_mutex_lock
#define xavs2_thread_mutex_unlock        pthread_mutex_unlock
#define xavs2_thread_cond_t              pthread_cond_t
#define xavs2_thread_cond_init           pthread_cond_init
#define xavs2_thread_cond_destroy        pthread_cond_destroy
#define xavs2_thread_cond_signal         pthread_cond_signal
#define xavs2_thread_cond_broadcast      pthread_cond_broadcast
#define xavs2_thread_cond_wait           pthread_cond_wait
#define xavs2_thread_attr_t              pthread_attr_t
#define xavs2_thread_attr_init           pthread_attr_init
#define xavs2_thread_attr_destroy        pthread_attr_destroy
#define xavs2_thread_attr_setdetachstate pthread_attr_setdetachstate
#define xavs2_thread_num_processors_np   pthread_num_processors_np
#define XAVS2_PTHREAD_MUTEX_INITIALIZER   PTHREAD_MUTEX_INITIALIZER

#elif HAVE_WIN32THREAD
#include "win32thread.h"

#else
#define xavs2_thread_t                   int
#define xavs2_thread_create(t,u,f,d)     0
#define xavs2_thread_join(t,s)
#endif // HAVE_*THREAD

#if !HAVE_POSIXTHREAD && !HAVE_WIN32THREAD
#define xavs2_thread_mutex_t             int
#define xavs2_thread_mutex_init(m,f)     0
#define xavs2_thread_mutex_destroy(m)
#define xavs2_thread_mutex_lock(m)
#define xavs2_thread_mutex_unlock(m)
#define xavs2_thread_cond_t              int
#define xavs2_thread_cond_init(c,f)      0
#define xavs2_thread_cond_destroy(c)
#define xavs2_thread_cond_broadcast(c)
#define xavs2_thread_cond_wait(c,m)
#define xavs2_thread_attr_t              int
#define xavs2_thread_attr_init(a)        0
#define xavs2_thread_attr_destroy(a)
#define XAVS2_PTHREAD_MUTEX_INITIALIZER   0
#endif

#if HAVE_POSIXTHREAD
#if SYS_WINDOWS
#define xavs2_lower_thread_priority(p) \
{\
    xavs2_thread_t handle = pthread_self();\
    struct sched_param sp;\
    int policy = SCHED_OTHER;\
    pthread_getschedparam(handle, &policy, &sp);\
    sp.sched_priority -= p;\
    pthread_setschedparam(handle, policy, &sp);\
}

#else
#include <unistd.h>
#define xavs2_lower_thread_priority(p) { UNUSED int nice_ret = nice(p); }
#endif /* SYS_WINDOWS */
#elif HAVE_WIN32THREAD
#define xavs2_lower_thread_priority(p) SetThreadPriority(GetCurrentThread(), XAVS2_MAX(-2, -p))
#else
#define xavs2_lower_thread_priority(p)
#endif

#if SYS_WINDOWS
#define xavs2_sleep_ms(x)              Sleep(x)
#else
#define xavs2_sleep_ms(x)              usleep(x * 1000)
#endif


/**
 * ===========================================================================
 * inline functions
 * ===========================================================================
 */

#if defined(__GNUC__) && (__GNUC__ > 3 || __GNUC__ == 3 && __GNUC_MINOR__ > 3)
#define xavs2_clz(x)      __builtin_clz(x)
#define xavs2_ctz(x)      __builtin_ctz(x)
#elif defined(_MSC_VER) && defined(_WIN32)
static int ALWAYS_INLINE xavs2_clz(const uint32_t x)
{
    DWORD r;
    _BitScanReverse(&r, (DWORD)x);
    return (r ^ 31);
}

static int ALWAYS_INLINE xavs2_ctz(const uint32_t x)
{
    DWORD r;
    _BitScanForward(&r, (DWORD)x);
    return r;
}

#else
static int ALWAYS_INLINE xavs2_clz(uint32_t x)
{
    static uint8_t lut[16] = {4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    int y, z = (((x >> 16) - 1) >> 27) & 16;
    x >>= z ^ 16;
    z += y = ((x - 0x100) >> 28) & 8;
    x >>= y ^ 8;
    z += y = ((x - 0x10) >> 29) & 4;
    x >>= y ^ 4;
    return z + lut[x];
}

static int ALWAYS_INLINE xavs2_ctz(uint32_t x)
{
    static uint8_t lut[16] = {4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};
    int y, z = (((x & 0xffff) - 1) >> 27) & 16;
    x >>= z;
    z += y = (((x & 0xff) - 1) >> 28) & 8;
    x >>= y;
    z += y = (((x & 0xf) - 1) >> 29) & 4;
    x >>= y;
    return z + lut[x & 0xf];
}
#endif


/* ---------------------------------------------------------------------------
 * prefetch
 */
#if HAVE_X86_INLINE_ASM && HAVE_MMX
/* Don't use __builtin_prefetch; even as recent as 4.3.4, GCC seems incapable
 * of using complex address modes properly unless we use inline asm. */
static void ALWAYS_INLINE xavs2_prefetch(void *p)
{
    asm volatile("prefetcht0 %0"::"m"(*(uint8_t *)p));
}
/* We require that prefetch not fault on invalid reads, so we only enable it
 * on known architectures. */
#elif defined(__GNUC__) && (__GNUC__ > 3 || __GNUC__ == 3 && __GNUC_MINOR__ > 1) &&\
      (ARCH_X86 || ARCH_X86_64 || ARCH_ARM || ARCH_PPC)
#define xavs2_prefetch(x)     __builtin_prefetch(x)
#elif defined(_MSC_VER)
#define xavs2_prefetch(x)     _mm_prefetch((const char*)(x), _MM_HINT_T0)
#else
#define xavs2_prefetch(x)
#endif


/* ---------------------------------------------------------------------------
 * log2/log2f
 */
#if !HAVE_LOG2F
#define log2f(x)    (logf(x)/0.693147180559945f)
#define log2(x)     (log(x)/0.693147180559945)
#endif

#endif /* XAVS2_OSDEP_H */
