/*****************************************************************************
 * Copyright (C) 2013-2017 MulticoreWare, Inc
 * Copyright (C) 2018~ VCL, NELVT, Peking University
 *
 * Authors: Loren Merritt <lorenm@u.washington.edu>
 *          Laurent Aimar <fenrir@via.ecp.fr>
 *          Fiona Glaser  <fiona@x264.com>
 *          Steve Borho   <steve@borho.org>
 *          Falei LUO     <falei.luo@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at license @ x265.com.
 *****************************************************************************/

#include "common.h"
#include "cpu.h"

#if SYS_MACOSX || SYS_FREEBSD
#include <sys/types.h>
#include <sys/sysctl.h>
#endif
#if SYS_OPENBSD
#include <sys/param.h>
#include <sys/sysctl.h>
#include <machine/cpu.h>
#endif

#if ARCH_ARM
#include <signal.h>
#include <setjmp.h>
static sigjmp_buf jmpbuf;
static volatile sig_atomic_t canjump = 0;

static void sigill_handler(int sig)
{
    if (!canjump) {
        signal(sig, SIG_DFL);
        raise(sig);
    }

    canjump = 0;
    siglongjmp(jmpbuf, 1);
}

#endif // if ARCH_ARM

/* ---------------------------------------------------------------------------
 */
typedef struct {
    const char name[16];
    int flags;
} xavs2_cpu_name_t;

/* ---------------------------------------------------------------------------
 */
static const xavs2_cpu_name_t xavs2_cpu_names[] = {
#if ARCH_X86 || ARCH_X86_64
#define MMX2            XAVS2_CPU_MMX | XAVS2_CPU_MMX2 | XAVS2_CPU_CMOV
    { "MMX2",           MMX2 },
    { "MMXEXT",         MMX2 },
    { "SSE",            MMX2 | XAVS2_CPU_SSE },
#define SSE2            MMX2 | XAVS2_CPU_SSE | XAVS2_CPU_SSE2
    { "SSE2Slow",       SSE2 | XAVS2_CPU_SSE2_IS_SLOW },
    { "SSE2",           SSE2 },
    { "SSE2Fast",       SSE2 | XAVS2_CPU_SSE2_IS_FAST },
    { "SSE3",           SSE2 | XAVS2_CPU_SSE3 },
    { "SSSE3",          SSE2 | XAVS2_CPU_SSE3 | XAVS2_CPU_SSSE3 },
    { "SSE4.1",         SSE2 | XAVS2_CPU_SSE3 | XAVS2_CPU_SSSE3 | XAVS2_CPU_SSE4 },
    { "SSE4",           SSE2 | XAVS2_CPU_SSE3 | XAVS2_CPU_SSSE3 | XAVS2_CPU_SSE4 },
    { "SSE4.2",         SSE2 | XAVS2_CPU_SSE3 | XAVS2_CPU_SSSE3 | XAVS2_CPU_SSE4 | XAVS2_CPU_SSE42 },
#define AVX             SSE2 | XAVS2_CPU_SSE3 | XAVS2_CPU_SSSE3 | XAVS2_CPU_SSE4 | XAVS2_CPU_SSE42 | XAVS2_CPU_AVX
    { "AVX",            AVX },
    { "XOP",            AVX | XAVS2_CPU_XOP },
    { "FMA4",           AVX | XAVS2_CPU_FMA4 },
    { "AVX2",           AVX | XAVS2_CPU_AVX2 },
    { "FMA3",           AVX | XAVS2_CPU_FMA3 },
#undef AVX
#undef SSE2
#undef MMX2
    { "Cache32",        XAVS2_CPU_CACHELINE_32 },
    { "Cache64",        XAVS2_CPU_CACHELINE_64 },
    { "LZCNT",          XAVS2_CPU_LZCNT },
    { "BMI1",           XAVS2_CPU_BMI1 },
    { "BMI2",           XAVS2_CPU_BMI1 | XAVS2_CPU_BMI2 },
    { "SlowCTZ",        XAVS2_CPU_SLOW_CTZ },
    { "SlowAtom",       XAVS2_CPU_SLOW_ATOM },
    { "SlowPshufb",     XAVS2_CPU_SLOW_PSHUFB },
    { "SlowPalignr",    XAVS2_CPU_SLOW_PALIGNR },
    { "SlowShuffle",    XAVS2_CPU_SLOW_SHUFFLE },
    { "UnalignedStack", XAVS2_CPU_STACK_MOD4 },

#elif ARCH_ARM
    { "ARMv6",          XAVS2_CPU_ARMV6 },
    { "NEON",           XAVS2_CPU_NEON },
    { "FastNeonMRC",    XAVS2_CPU_FAST_NEON_MRC },
#endif // if XAVS2_ARCH_X86
    { "", 0 }
};

/* ---------------------------------------------------------------------------
 */
char *xavs2_get_simd_capabilities(char *buf, int cpuid)
{
    char *p = buf;
    for (int i = 0; xavs2_cpu_names[i].flags; i++) {
        if (!strcmp(xavs2_cpu_names[i].name, "SSE")
            && (cpuid & XAVS2_CPU_SSE2))
            continue;
        if (!strcmp(xavs2_cpu_names[i].name, "SSE2")
            && (cpuid & (XAVS2_CPU_SSE2_IS_FAST | XAVS2_CPU_SSE2_IS_SLOW)))
            continue;
        if (!strcmp(xavs2_cpu_names[i].name, "SSE3")
            && (cpuid & XAVS2_CPU_SSSE3 || !(cpuid & XAVS2_CPU_CACHELINE_64)))
            continue;
        if (!strcmp(xavs2_cpu_names[i].name, "SSE4.1")
            && (cpuid & XAVS2_CPU_SSE42))
            continue;
        if (!strcmp(xavs2_cpu_names[i].name, "BMI1")
            && (cpuid & XAVS2_CPU_BMI2))
            continue;
        if ((cpuid & xavs2_cpu_names[i].flags) == xavs2_cpu_names[i].flags
            && (!i || xavs2_cpu_names[i].flags != xavs2_cpu_names[i - 1].flags))
            p += sprintf(p, " %s", xavs2_cpu_names[i].name);
    }

    if (p == buf)
        sprintf(p, " none! (%08x)", cpuid);
    return buf;
}

#if HAVE_MMX
/* ---------------------------------------------------------------------------
 */
uint32_t xavs2_cpu_detect(void)
{
    uint32_t cpuid = 0;

    uint32_t eax, ebx, ecx, edx;
    uint32_t vendor[4] = { 0 };
    uint32_t max_extended_cap, max_basic_cap;

#if !ARCH_X86_64
    if (!xavs2_cpu_cpuid_test()) {
        return 0;
    }
#endif

    xavs2_cpu_cpuid(0, &eax, vendor + 0, vendor + 2, vendor + 1);
    max_basic_cap = eax;
    if (max_basic_cap == 0) {
        return 0;
    }

    xavs2_cpu_cpuid(1, &eax, &ebx, &ecx, &edx);
    if (edx & 0x00800000) {
        cpuid |= XAVS2_CPU_MMX;
    } else {
        return cpuid;
    }

    if (edx & 0x02000000) {
        cpuid |= XAVS2_CPU_MMX2 | XAVS2_CPU_SSE;
    }
    if (edx & 0x00008000) {
        cpuid |= XAVS2_CPU_CMOV;
    } else {
        return cpuid;
    }

    if (edx & 0x04000000) {
        cpuid |= XAVS2_CPU_SSE2;
    }
    if (ecx & 0x00000001) {
        cpuid |= XAVS2_CPU_SSE3;
    }
    if (ecx & 0x00000200) {
        cpuid |= XAVS2_CPU_SSSE3;
    }
    if (ecx & 0x00080000) {
        cpuid |= XAVS2_CPU_SSE4;
    }
    if (ecx & 0x00100000) {
        cpuid |= XAVS2_CPU_SSE42;
    }

    /* Check OXSAVE and AVX bits */
    if ((ecx & 0x18000000) == 0x18000000) {
        /* Check for OS support */
        xavs2_cpu_xgetbv(0, &eax, &edx);
        if ((eax & 0x6) == 0x6) {
            cpuid |= XAVS2_CPU_AVX;
            if (ecx & 0x00001000) {
                cpuid |= XAVS2_CPU_FMA3;
            }
        }
    }

    if (max_basic_cap >= 7) {
        xavs2_cpu_cpuid(7, &eax, &ebx, &ecx, &edx);
        /* AVX2 requires OS support, but BMI1/2 don't. */
        if ((cpuid & XAVS2_CPU_AVX) && (ebx & 0x00000020)) {
            cpuid |= XAVS2_CPU_AVX2;
        }
        if (ebx & 0x00000008) {
            cpuid |= XAVS2_CPU_BMI1;
            if (ebx & 0x00000100) {
                cpuid |= XAVS2_CPU_BMI2;
            }
        }
    }

    if (cpuid & XAVS2_CPU_SSSE3) {
        cpuid |= XAVS2_CPU_SSE2_IS_FAST;
    }

    xavs2_cpu_cpuid(0x80000000, &eax, &ebx, &ecx, &edx);
    max_extended_cap = eax;

    if (max_extended_cap >= 0x80000001) {
        xavs2_cpu_cpuid(0x80000001, &eax, &ebx, &ecx, &edx);

        if (ecx & 0x00000020)
            cpuid |= XAVS2_CPU_LZCNT;               /* Supported by Intel chips starting with Haswell */
        if (ecx & 0x00000040) {                     /* SSE4a, AMD only */
            int family = ((eax >> 8) & 0xf) + ((eax >> 20) & 0xff);
            cpuid |= XAVS2_CPU_SSE2_IS_FAST;        /* Phenom and later CPUs have fast SSE units */
            if (family == 0x14) {
                cpuid &= ~XAVS2_CPU_SSE2_IS_FAST;   /* SSSE3 doesn't imply fast SSE anymore... */
                cpuid |= XAVS2_CPU_SSE2_IS_SLOW;    /* Bobcat has 64-bit SIMD units */
                cpuid |= XAVS2_CPU_SLOW_PALIGNR;    /* palignr is insanely slow on Bobcat */
            }
            if (family == 0x16) {
                cpuid |= XAVS2_CPU_SLOW_PSHUFB;     /* Jaguar's pshufb isn't that slow, but it's slow enough
                                                     * compared to alternate instruction sequences that this
                                                     * is equal or faster on almost all such functions. */
            }
        }

        if (cpuid & XAVS2_CPU_AVX) {
            if (ecx & 0x00000800) {   /* XOP */
                cpuid |= XAVS2_CPU_XOP;
            }
            if (ecx & 0x00010000) {   /* FMA4 */
                cpuid |= XAVS2_CPU_FMA4;
            }
        }

        if (!strcmp((char*)vendor, "AuthenticAMD")) {
            if (edx & 0x00400000) {
                cpuid |= XAVS2_CPU_MMX2;
            }
            if (!(cpuid & XAVS2_CPU_LZCNT)) {
                cpuid |= XAVS2_CPU_SLOW_CTZ;
            }
            if ((cpuid & XAVS2_CPU_SSE2) && !(cpuid & XAVS2_CPU_SSE2_IS_FAST)) {
                cpuid |= XAVS2_CPU_SSE2_IS_SLOW; /* AMD CPUs come in two types: terrible at SSE and great at it */
            }
        }
    }

    if (!strcmp((char*)vendor, "GenuineIntel")) {
        int family, model;
        xavs2_cpu_cpuid(1, &eax, &ebx, &ecx, &edx);
        family = ((eax >> 8) & 0xf) + ((eax >> 20) & 0xff);
        model = ((eax >> 4) & 0xf) + ((eax >> 12) & 0xf0);
        if (family == 6) {
            /* 6/9 (pentium-m "banias"), 6/13 (pentium-m "dothan"), and 6/14 (core1 "yonah")
             * theoretically support sse2, but it's significantly slower than mmx for
             * almost all of x264's functions, so let's just pretend they don't. */
            if (model == 9 || model == 13 || model == 14) {
                cpuid &= ~(XAVS2_CPU_SSE2 | XAVS2_CPU_SSE3);
                //XAVS2_CHECK(!(cpuid & (XAVS2_CPU_SSSE3 | XAVS2_CPU_SSE4)), "unexpected CPU ID %d\n", cpuid);
            } else if (model == 28) {
                /* Detect Atom CPU */
                cpuid |= XAVS2_CPU_SLOW_ATOM;
                cpuid |= XAVS2_CPU_SLOW_CTZ;
                cpuid |= XAVS2_CPU_SLOW_PSHUFB;
            } else if ((cpuid & XAVS2_CPU_SSSE3) && !(cpuid & XAVS2_CPU_SSE4) && model < 23) {
                /* Conroe has a slow shuffle unit. Check the model number to make sure not
                 * to include crippled low-end Penryns and Nehalems that don't have SSE4. */
                cpuid |= XAVS2_CPU_SLOW_SHUFFLE;
            }
        }
    }

    if ((!strcmp((char*)vendor, "GenuineIntel") || !strcmp((char*)vendor, "CyrixInstead")) && !(cpuid & XAVS2_CPU_SSE42)) {
        /* cacheline size is specified in 3 places, any of which may be missing */
        int cache;
        xavs2_cpu_cpuid(1, &eax, &ebx, &ecx, &edx);
        cache = (ebx & 0xff00) >> 5; // cflush size
        if (!cache && max_extended_cap >= 0x80000006) {
            xavs2_cpu_cpuid(0x80000006, &eax, &ebx, &ecx, &edx);
            cache = ecx & 0xff; // cacheline size
        }
        if (!cache && max_basic_cap >= 2) {
            // Cache and TLB Information
            static const char cache32_ids[] = { 0x0a, 0x0c, 0x41, 0x42, 0x43, 0x44, 0x45, 0x82, 0x83, 0x84, 0x85, 0 };
            static const char cache64_ids[] = { 0x22, 0x23, 0x25, 0x29, 0x2c, 0x46, 0x47, 0x49, 0x60, 0x66, 0x67,
                                                0x68, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7c, 0x7f, 0x86, 0x87, 0
                                              };
            uint32_t buf[4];
            int max, i = 0, j;
            do {
                xavs2_cpu_cpuid(2, buf + 0, buf + 1, buf + 2, buf + 3);
                max = buf[0] & 0xff;
                buf[0] &= ~0xff;
                for (j = 0; j < 4; j++) {
                    if (!(buf[j] >> 31)) {
                        while (buf[j]) {
                            if (strchr(cache32_ids, buf[j] & 0xff)) {
                                cache = 32;
                            }
                            if (strchr(cache64_ids, buf[j] & 0xff)) {
                                cache = 64;
                            }
                            buf[j] >>= 8;
                        }
                    }
                }
            } while (++i < max);
        }

        if (cache == 32) {
            cpuid |= XAVS2_CPU_CACHELINE_32;
        } else if (cache == 64) {
            cpuid |= XAVS2_CPU_CACHELINE_64;
        } else {
            xavs2_log(NULL, XAVS2_LOG_WARNING, "unable to determine cacheline size\n");
        }
    }

#ifdef BROKEN_STACK_ALIGNMENT
    cpuid |= XAVS2_CPU_STACK_MOD4;
#endif

    return cpuid;
}

#endif // if HAVE_MMX

#if SYS_LINUX && !(defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7__))
/* ---------------------------------------------------------------------------
 */
int sched_getaffinity(pid_t pid, size_t cpusetsize, cpu_set_t *mask);
#endif

/* ---------------------------------------------------------------------------
 */
int xavs2_cpu_num_processors(void)
{
#if !HAVE_THREAD
    return 1;
#elif defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7__)
    return 2;
#elif SYS_WINDOWS
    return xavs2_thread_num_processors_np();
#elif SYS_LINUX
    unsigned int bit;
    int np = 0;
    cpu_set_t p_aff;

    memset(&p_aff, 0, sizeof(p_aff));
    sched_getaffinity(0, sizeof(p_aff), &p_aff);
    for (bit = 0; bit < sizeof(p_aff); bit++) {
        np += (((uint8_t *)& p_aff)[bit / 8] >> (bit % 8)) & 1;
    }
    return np;

#elif SYS_BEOS
    system_info info;

    get_system_info(&info);
    return info.cpu_count;

#elif SYS_MACOSX || SYS_FREEBSD || SYS_OPENBSD
    int numberOfCPUs;
    size_t length = sizeof (numberOfCPUs);
#if SYS_OPENBSD
    int mib[2] = { CTL_HW, HW_NCPU };
    if(sysctl(mib, 2, &numberOfCPUs, &length, NULL, 0))
#else
    if(sysctlbyname("hw.ncpu", &numberOfCPUs, &length, NULL, 0))
#endif
    {
        numberOfCPUs = 1;
    }
    return numberOfCPUs;

#else
    return 1;
#endif
}
