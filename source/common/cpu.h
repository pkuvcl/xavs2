/*
 * cpu.h
 *
 * Description of this file:
 *    CPU-Processing functions definition of the xavs2 library
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



#ifndef XAVS2_CPU_H
#define XAVS2_CPU_H

/**
 * ===========================================================================
 * const defines
 * ===========================================================================
 */
/* CPU flags */

/* x86 */
#define XAVS2_CPU_CMOV            0x0000001
#define XAVS2_CPU_MMX             0x0000002
#define XAVS2_CPU_MMX2            0x0000004   /* MMX2 aka MMXEXT aka ISSE */
#define XAVS2_CPU_MMXEXT          XAVS2_CPU_MMX2
#define XAVS2_CPU_SSE             0x0000008
#define XAVS2_CPU_SSE2            0x0000010
#define XAVS2_CPU_SSE3            0x0000020
#define XAVS2_CPU_SSSE3           0x0000040
#define XAVS2_CPU_SSE4            0x0000080   /* SSE4.1 */
#define XAVS2_CPU_SSE42           0x0000100   /* SSE4.2 */
#define XAVS2_CPU_LZCNT           0x0000200   /* Phenom support for "leading zero count" instruction. */
#define XAVS2_CPU_AVX             0x0000400   /* AVX support: requires OS support even if YMM registers aren't used. */
#define XAVS2_CPU_XOP             0x0000800   /* AMD XOP */
#define XAVS2_CPU_FMA4            0x0001000   /* AMD FMA4 */
#define XAVS2_CPU_AVX2            0x0002000   /* AVX2 */
#define XAVS2_CPU_FMA3            0x0004000   /* Intel FMA3 */
#define XAVS2_CPU_BMI1            0x0008000   /* BMI1 */
#define XAVS2_CPU_BMI2            0x0010000   /* BMI2 */
/* x86 modifiers */
#define XAVS2_CPU_CACHELINE_32    0x0020000   /* avoid memory loads that span the border between two cachelines */
#define XAVS2_CPU_CACHELINE_64    0x0040000   /* 32/64 is the size of a cacheline in bytes */
#define XAVS2_CPU_SSE2_IS_SLOW    0x0080000   /* avoid most SSE2 functions on Athlon64 */
#define XAVS2_CPU_SSE2_IS_FAST    0x0100000   /* a few functions are only faster on Core2 and Phenom */
#define XAVS2_CPU_SLOW_SHUFFLE    0x0200000   /* The Conroe has a slow shuffle unit (relative to overall SSE performance) */
#define XAVS2_CPU_STACK_MOD4      0x0400000   /* if stack is only mod4 and not mod16 */
#define XAVS2_CPU_SLOW_CTZ        0x0800000   /* BSR/BSF x86 instructions are really slow on some CPUs */
#define XAVS2_CPU_SLOW_ATOM       0x1000000   /* The Atom is terrible: slow SSE unaligned loads, slow
                                               * SIMD multiplies, slow SIMD variable shifts, slow pshufb,
                                               * cacheline split penalties -- gather everything here that
                                               * isn't shared by other CPUs to avoid making half a dozen
                                               * new SLOW flags. */
#define XAVS2_CPU_SLOW_PSHUFB     0x2000000   /* such as on the Intel Atom */
#define XAVS2_CPU_SLOW_PALIGNR    0x4000000   /* such as on the AMD Bobcat */

/* ARM */
#define XAVS2_CPU_ARMV6           0x0000001
#define XAVS2_CPU_NEON            0x0000002   /* ARM NEON */
#define XAVS2_CPU_FAST_NEON_MRC   0x0000004   /* Transfer from NEON to ARM register is fast (Cortex-A9) */


/**
* ===========================================================================
* declarations
* ===========================================================================
*/
#define xavs2_cpu_detect FPFX(cpu_detect)
uint32_t xavs2_cpu_detect(void);
#define xavs2_cpu_num_processors FPFX(cpu_num_processors)
int  xavs2_cpu_num_processors(void);
#define xavs2_cpu_emms FPFX(cpu_emms)
void xavs2_cpu_emms(void);
#define xavs2_cpu_sfence FPFX(cpu_sfence)
void xavs2_cpu_sfence(void);
#define xavs2_get_simd_capabilities FPFX(get_simd_capabilities)
char *xavs2_get_simd_capabilities(char *buf, int cpuid);

#if HAVE_MMX
#define xavs2_cpu_cpuid_test FPFX(cpu_cpuid_test)
int xavs2_cpu_cpuid_test(void);
#define xavs2_cpu_cpuid FPFX(cpu_cpuid)
uint32_t xavs2_cpu_cpuid(uint32_t op, uint32_t * eax, uint32_t * ebx, uint32_t * ecx, uint32_t * edx);
#define xavs2_cpu_xgetbv FPFX(cpu_xgetbv)
void xavs2_cpu_xgetbv(uint32_t op, uint32_t *eax, uint32_t *edx);
#define xavs2_emms() xavs2_cpu_emms()
#else
#define xavs2_emms()
#endif

#endif  // XAVS2_CPU_H
