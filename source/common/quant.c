/*
 * quant.h
 *
 * Description of this file:
 *    Quant functions definition of the xavs2 library
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

#include "common.h"
#include "primitives.h"
#include "block_info.h"
#include "cpu.h"

#if HAVE_MMX
#include "x86/quant8.h"
#endif


/**
 * ===========================================================================
 * local/global variables
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
const uint16_t tab_Q_TAB[80] = {
    32768, 29775, 27554, 25268, 23170, 21247, 19369, 17770,
    16302, 15024, 13777, 12634, 11626, 10624,  9742,  8958,
     8192,  7512,  6889,  6305,  5793,  5303,  4878,  4467,
     4091,  3756,  3444,  3161,  2894,  2654,  2435,  2235,
     2048,  1878,  1722,  1579,  1449,  1329,  1218,  1117,
     1024,   939,   861,   790,   724,   664,   609,   558,
      512,   470,   430,   395,   362,   332,   304,   279,
      256,   235,   215,   197,   181,   166,   152,   140,
      128,   116,   108,    99,    91,    83,    76,    69,
        64,   59,    54,    49,    45,    41,    38,    35
};

/* ---------------------------------------------------------------------------
 */
const uint16_t tab_IQ_TAB[80] = {
    32768, 36061, 38968, 42495, 46341, 50535, 55437, 60424,
    32932, 35734, 38968, 42495, 46177, 50535, 55109, 59933,
    65535, 35734, 38968, 42577, 46341, 50617, 55027, 60097,
    32809, 35734, 38968, 42454, 46382, 50576, 55109, 60056,
    65535, 35734, 38968, 42495, 46320, 50515, 55109, 60076,
    65535, 35744, 38968, 42495, 46341, 50535, 55099, 60087,
    65535, 35734, 38973, 42500, 46341, 50535, 55109, 60097,
    32771, 35734, 38965, 42497, 46341, 50535, 55109, 60099,
    32768, 36061, 38968, 42495, 46341, 50535, 55437, 60424,
    32932, 35734, 38968, 42495, 46177, 50535, 55109, 59933
};

/* ---------------------------------------------------------------------------
 */
const uint8_t tab_IQ_SHIFT[80] = {
    15, 15, 15, 15, 15, 15, 15, 15,
    14, 14, 14, 14, 14, 14, 14, 14,
    14, 13, 13, 13, 13, 13, 13, 13,
    12, 12, 12, 12, 12, 12, 12, 12,
    12, 11, 11, 11, 11, 11, 11, 11,
    11, 10, 10, 10, 10, 10, 10, 10,
    10,  9,  9,  9,  9,  9,  9,  9,
     8,  8,  8,  8,  8,  8,  8,  8,
     7,  7,  7,  7,  7,  7,  7,  7,
     6,  6,  6,  6,  6,  6,  6,  6
};


/**
 * ===========================================================================
 * function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * i_coef - number of coeffs, 16 <= i_coef <= 1024
 */
static
int quant_c(coeff_t *coef, const int i_coef, const int scale, const int shift, const int add)
{
    int num_non_zero = 0;
    int i;

    for (i = 0; i < i_coef; i++) {
        coef[i] = (coeff_t)(xavs2_sign2(coef[i]) * ((XAVS2_ABS(coef[i]) * scale + add) >> shift));
        num_non_zero += coef[i] != 0;
    }

    return num_non_zero;
}


/* ---------------------------------------------------------------------------
 * i_coef - number of coeffs, 16 <= i_coef <= 1024
 */
static
void abs_coeff_c(coeff_t *dst, const coeff_t *src, const int i_coef)
{
    int i;

    for (i = 0; i < i_coef; i++) {
        dst[i] = (coeff_t)abs(src[i]);
    }
}


/* ---------------------------------------------------------------------------
 * i_coef - number of coeffs, 16 <= i_coef <= 1024
 */
static
int add_sign_c(coeff_t *dst, const coeff_t *abs_val, const int i_coef)
{
    int nz = 0;
    int i;

    for (i = 0; i < i_coef; i++) {
        dst[i] = (dst[i] > 0) ? abs_val[i] : -abs_val[i];
        nz += (!!abs_val[i]);
    }

    return nz;
}


/* ---------------------------------------------------------------------------
 * adaptive frequency weighting quantization
 */
static int quant_weighted_c(coeff_t *coef, const int i_coef, const int scale, const int shift, const int add, int *levelscale)
{
    int num_non_zero = 0;
    int i;

    for (i = 0; i < i_coef; i++) {
        coef[i] = (coeff_t)(xavs2_sign2(coef[i]) * ((((XAVS2_ABS(coef[i]) * levelscale[i] + (1 << 18)) >> 19) * scale + add) >> shift));
        num_non_zero += coef[i] != 0;
    }

    return num_non_zero;
}

/* ---------------------------------------------------------------------------
 */
static void dequant_c(coeff_t *coef, const int i_coef, const int scale, const int shift)
{
    const int add = (1 << (shift - 1));
    int k;

    for (k = 0; k < i_coef; k++) {
        if (coef[k] != 0) {
            // dequantization & descale
            coef[k] = (coeff_t)XAVS2_CLIP3(-32768, 32767, (coef[k] * scale + add) >> shift);
        }
    }
}

#if ENABLE_WQUANT
/* ---------------------------------------------------------------------------
 */
static void dequant_weighted_c(coeff_t *coef, int i_coef, int scale, int shift, int wqm_shift, int wqm_stride, int xy_shift, int16_t *wq_matrix, const int16_t(*AVS_SCAN)[2])
{
    const int add = (1 << (shift - 1));
    const int wqm_mask = wqm_stride - 1;
    int xx, yy;
    int k;
    int16_t wqm_coef = 0;

    for (k = 0; k < i_coef; k++) {
        xx = AVS_SCAN[k][0] >> xy_shift;
        yy = AVS_SCAN[k][1] >> xy_shift;
        wqm_coef = wq_matrix[(yy & wqm_mask) * wqm_stride + (xx & wqm_mask)];

        if (coef[k] != 0) {
            // dequantization & descale
            coef[k] = (coeff_t)XAVS2_CLIP3(-32768, 32767, (((((coef[k] * wqm_coef) >> wqm_shift) * scale) >> 4) + add) >> shift);
        }
    }
}
#endif

/* ---------------------------------------------------------------------------
 */
void xavs2_quant_init(uint32_t cpuid, dct_funcs_t *dctf)
{
    /* init c function handles */
    dctf->quant   = quant_c;
    dctf->dequant = dequant_c;
    dctf->wquant  = quant_weighted_c;

    dctf->abs_coeff = abs_coeff_c;
    dctf->add_sign  = add_sign_c;

    /* init asm function handles */
#if HAVE_MMX
    if (cpuid & XAVS2_CPU_SSE4) {
        dctf->quant     = FPFX(quant_sse4);
        dctf->dequant   = FPFX(dequant_sse4);
        dctf->abs_coeff = abs_coeff_sse128;
        dctf->add_sign  = add_sign_sse128;
    }

    if (cpuid & XAVS2_CPU_AVX2) {
        dctf->quant     = quant_c_avx2;
        dctf->dequant   = dequant_c_avx2;
        dctf->abs_coeff = abs_coeff_avx2;
        dctf->add_sign  = add_sign_avx2;

#if _MSC_VER
        dctf->quant     = FPFX(quant_avx2);   // would cause mis-match on some machine/system
#endif
#if ARCH_X86_64
        dctf->dequant   = FPFX(dequant_avx2);
#endif
    }
#else
    UNUSED_PARAMETER(cpuid);
#endif  // if HAVE_MMX
}
