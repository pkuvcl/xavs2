/*
 * transform.c
 *
 * Description of this file:
 *    transform functions definition of the xavs2 library
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
#include "transform.h"
#include "block_info.h"
#include "cpu.h"

#if HAVE_MMX
#include "x86/dct8.h"
#include "vec/intrinsic.h"
#endif
// ---------------------------------------------------------------------------
#define LOT_MAX_WLT_TAP     2   // number of wavelet transform tap (5-3)


/**
 * ===========================================================================
 * local/global variables
 * ===========================================================================
 */


/* ---------------------------------------------------------------------------
 */
static const int16_t g_T4[4][4] = {
    { 32,  32,  32,  32 },
    { 42,  17, -17, -42 },
    { 32, -32, -32,  32 },
    { 17, -42,  42, -17 }
};

/* ---------------------------------------------------------------------------
 */
static const int16_t g_T8[8][8] = {
    { 32,  32,  32,  32,  32,  32,  32,  32 },
    { 44,  38,  25,   9,  -9, -25, -38, -44 },
    { 42,  17, -17, -42, -42, -17,  17,  42 },
    { 38,  -9, -44, -25,  25,  44,   9, -38 },
    { 32, -32, -32,  32,  32, -32, -32,  32 },
    { 25, -44,   9,  38, -38,  -9,  44, -25 },
    { 17, -42,  42, -17, -17,  42, -42,  17 },
    {  9, -25,  38, -44,  44, -38,  25,  -9 }
};

/* ---------------------------------------------------------------------------
 */
static const int16_t g_T16[16][16] = {
    { 32,  32,  32,  32,  32,  32,  32,  32,  32,  32,  32,  32,  32,  32,  32,  32 }, // 0
    { 45,  43,  40,  35,  29,  21,  13,   4,  -4, -13, -21, -29, -35, -40, -43, -45 }, // 1
    { 44,  38,  25,   9,  -9, -25, -38, -44, -44, -38, -25,  -9,   9,  25,  38,  44 }, // 2
    { 43,  29,   4, -21, -40, -45, -35, -13,  13,  35,  45,  40,  21,  -4, -29, -43 }, // 3
    { 42,  17, -17, -42, -42, -17,  17,  42,  42,  17, -17, -42, -42, -17,  17,  42 }, // 4
    { 40,   4, -35, -43, -13,  29,  45,  21, -21, -45, -29,  13,  43,  35,  -4, -40 }, // 5
    { 38,  -9, -44, -25,  25,  44,   9, -38, -38,   9,  44,  25, -25, -44,  -9,  38 }, // 6
    { 35, -21, -43,   4,  45,  13, -40, -29,  29,  40, -13, -45,  -4,  43,  21, -35 }, // 7
    { 32, -32, -32,  32,  32, -32, -32,  32,  32, -32, -32,  32,  32, -32, -32,  32 }, // 8
    { 29, -40, -13,  45,  -4, -43,  21,  35, -35, -21,  43,   4, -45,  13,  40, -29 }, // 9
    { 25, -44,   9,  38, -38,  -9,  44, -25, -25,  44,  -9, -38,  38,   9, -44,  25 }, // 10
    { 21, -45,  29,  13, -43,  35,   4, -40,  40,  -4, -35,  43, -13, -29,  45, -21 }, // 11
    { 17, -42,  42, -17, -17,  42, -42,  17,  17, -42,  42, -17, -17,  42, -42,  17 }, // 12
    { 13, -35,  45, -40,  21,   4, -29,  43, -43,  29,  -4, -21,  40, -45,  35, -13 }, // 13
    {  9, -25,  38, -44,  44, -38,  25,  -9,  -9,  25, -38,  44, -44,  38, -25,   9 }, // 14
    {  4, -13,  21, -29,  35, -40,  43, -45,  45, -43,  40, -35,  29, -21,  13,  -4 }  // 15
};

/* ---------------------------------------------------------------------------
 */
static const int16_t g_T32[32][32] = {
    { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 },//0
    { 45, 45, 44, 43, 41, 39, 36, 34, 30, 27, 23, 19, 15, 11,  7,  2, -2, -7,-11,-15,-19,-23,-27,-30,-34,-36,-39,-41,-43,-44,-45,-45 },//1
    { 45, 43, 40, 35, 29, 21, 13,  4, -4,-13,-21,-29,-35,-40,-43,-45,-45,-43,-40,-35,-29,-21,-13, -4,  4, 13, 21, 29, 35, 40, 43, 45 },//2
    { 45, 41, 34, 23, 11, -2,-15,-27,-36,-43,-45,-44,-39,-30,-19, -7,  7, 19, 30, 39, 44, 45, 43, 36, 27, 15,  2,-11,-23,-34,-41,-45 },//3
    { 44, 38, 25,  9, -9,-25,-38,-44,-44,-38,-25, -9,  9, 25, 38, 44, 44, 38, 25,  9, -9,-25,-38,-44,-44,-38,-25, -9,  9, 25, 38, 44 },//4
    { 44, 34, 15, -7,-27,-41,-45,-39,-23, -2, 19, 36, 45, 43, 30, 11,-11,-30,-43,-45,-36,-19,  2, 23, 39, 45, 41, 27,  7,-15,-34,-44 },//5
    { 43, 29,  4,-21,-40,-45,-35,-13, 13, 35, 45, 40, 21, -4,-29,-43,-43,-29, -4, 21, 40, 45, 35, 13,-13,-35,-45,-40,-21,  4, 29, 43 },//
    { 43, 23, -7,-34,-45,-36,-11, 19, 41, 44, 27, -2,-30,-45,-39,-15, 15, 39, 45, 30,  2,-27,-44,-41,-19, 11, 36, 45, 34, 7, -23,-43 },//
    { 42, 17,-17,-42,-42,-17, 17, 42, 42, 17,-17,-42,-42,-17, 17, 42, 42, 17,-17,-42,-42,-17, 17, 42, 42, 17,-17,-42,-42,-17, 17, 42 },//8
    { 41, 11,-27,-45,-30,  7, 39, 43, 15,-23,-45,-34,  2, 36, 44, 19,-19,-44,-36, -2, 34, 45, 23,-15,-43,-39, -7, 30, 45, 27,-11,-41 },//
    { 40,  4,-35,-43,-13, 29, 45, 21,-21,-45,-29, 13, 43, 35, -4,-40,-40, -4, 35, 43, 13,-29,-45,-21, 21, 45, 29,-13,-43,-35,  4, 40 },//10
    { 39, -2,-41,-36,  7, 43, 34,-11,-44,-30, 15, 45, 27,-19,-45,-23, 23, 45, 19,-27,-45,-15, 30, 44, 11,-34,-43, -7, 36, 41,  2,-39 },//
    { 38, -9,-44,-25, 25, 44,  9,-38,-38,  9, 44, 25,-25,-44, -9, 38, 38, -9,-44,-25, 25, 44,  9,-38,-38,  9, 44, 25,-25,-44, -9, 38 },//12
    { 36,-15,-45,-11, 39, 34,-19,-45, -7, 41, 30,-23,-44, -2, 43, 27,-27,-43,  2, 44, 23,-30,-41,  7, 45, 19,-34,-39, 11, 45, 15,-36 },//
    { 35,-21,-43,  4, 45, 13,-40,-29, 29, 40,-13,-45, -4, 43, 21,-35,-35, 21, 43, -4,-45,-13, 40, 29,-29,-40, 13, 45,  4,-43,-21, 35 },//14
    { 34,-27,-39, 19, 43,-11,-45,  2, 45,  7,-44,-15, 41, 23,-36,-30, 30, 36,-23,-41, 15, 44, -7,-45, -2, 45, 11,-43,-19, 39, 27,-34 },//
    { 32,-32,-32, 32, 32,-32,-32, 32, 32,-32,-32, 32, 32,-32,-32, 32, 32,-32,-32, 32, 32,-32,-32, 32, 32,-32,-32, 32, 32,-32,-32, 32 },//16
    { 30,-36,-23, 41, 15,-44, -7, 45, -2,-45, 11, 43,-19,-39, 27, 34,-34,-27, 39, 19,-43,-11, 45,  2,-45,  7, 44,-15,-41, 23, 36,-30 },//
    { 29,-40,-13, 45, -4,-43, 21, 35,-35,-21, 43,  4,-45, 13, 40,-29,-29, 40, 13,-45,  4, 43,-21,-35, 35, 21,-43, -4, 45,-13,-40, 29 },//18
    { 27,-43, -2, 44,-23,-30, 41,  7,-45, 19, 34,-39,-11, 45,-15,-36, 36, 15,-45, 11, 39,-34,-19, 45, -7,-41, 30, 23,-44,  2, 43,-27 },//
    { 25,-44,  9, 38,-38, -9, 44,-25,-25, 44, -9,-38, 38,  9,-44, 25, 25,-44,  9, 38,-38, -9, 44,-25,-25, 44, -9,-38, 38,  9,-44, 25 },//20
    { 23,-45, 19, 27,-45, 15, 30,-44, 11, 34,-43,  7, 36,-41,  2, 39,-39, -2, 41,-36, -7, 43,-34,-11, 44,-30,-15, 45,-27,-19, 45,-23 },//
    { 21,-45, 29, 13,-43, 35,  4,-40, 40, -4,-35, 43,-13,-29, 45,-21,-21, 45,-29,-13, 43,-35, -4, 40,-40,  4, 35,-43, 13, 29,-45, 21 },//22
    { 19,-44, 36, -2,-34, 45,-23,-15, 43,-39,  7, 30,-45, 27, 11,-41, 41,-11,-27, 45,-30, -7, 39,-43, 15, 23,-45, 34,  2,-36, 44,-19 },//
    { 17,-42, 42,-17,-17, 42,-42, 17, 17,-42, 42,-17,-17, 42,-42, 17, 17,-42, 42,-17,-17, 42,-42, 17, 17,-42, 42,-17,-17, 42,-42, 17 },//24
    { 15,-39, 45,-30,  2, 27,-44, 41,-19,-11, 36,-45, 34, -7,-23, 43,-43, 23,  7,-34, 45,-36, 11, 19,-41, 44,-27, -2, 30,-45, 39,-15 },//
    { 13,-35, 45,-40, 21,  4,-29, 43,-43, 29, -4,-21, 40,-45, 35,-13,-13, 35,-45, 40,-21, -4, 29,-43, 43,-29,  4, 21,-40, 45,-35, 13 },//26
    { 11,-30, 43,-45, 36,-19, -2, 23,-39, 45,-41, 27, -7,-15, 34,-44, 44,-34, 15,  7,-27, 41,-45, 39,-23,  2, 19,-36, 45,-43, 30,-11 },//
    {  9,-25, 38,-44, 44,-38, 25, -9, -9, 25,-38, 44,-44, 38,-25,  9,  9,-25, 38,-44, 44,-38, 25, -9, -9, 25,-38, 44,-44, 38,-25,  9 },//28
    {  7,-19, 30,-39, 44,-45, 43,-36, 27,-15,  2, 11,-23, 34,-41, 45,-45, 41,-34, 23,-11, -2, 15,-27, 36,-43, 45,-44, 39,-30, 19, -7 },//
    {  4,-13, 21,-29, 35,-40, 43,-45, 45,-43, 40,-35, 29,-21, 13, -4, -4, 13,-21, 29,-35, 40,-43, 45,-45, 43,-40, 35,-29, 21,-13,  4 },//30
    {  2, -7, 11,-15, 19,-23, 27,-30, 34,-36, 39,-41, 43,-44, 45,-45, 45,-45, 44,-43, 41,-39, 36,-34, 30,-27, 23,-19, 15,-11,  7, -2 } //31
};

/* ---------------------------------------------------------------------------
 * secondary transform
 */
ALIGN16(const int16_t g_2T[SEC_TR_SIZE * SEC_TR_SIZE]) = {
    123,  -35,  -8,  -3,
    -32, -120,  30,  10,
     14,   25, 123, -22,
      8,   13,  19, 126
};

/* ---------------------------------------------------------------------------
 * secondary transform (only for 4x4)
 */
ALIGN16(const int16_t g_2T_C[SEC_TR_SIZE * SEC_TR_SIZE]) = {
    34,  58,  72,  81,
    77,  69,  -7, -75,
    79, -33, -75,  58,
    55, -84,  73, -28
};


/**
 * ===========================================================================
 * local function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * dst = g_T4 x src_T
 */
static void partialButterfly4(const coeff_t *src, coeff_t *dst, int shift, int line)
{
    int E[2], O[2];
    const int add = ((1 << shift) >> 1);
    int j;

    for (j = 0; j < line; j++) {
        /* E and O */
        E[0] = src[0] + src[3];
        O[0] = src[0] - src[3];
        E[1] = src[1] + src[2];
        O[1] = src[1] - src[2];

        dst[0       ] = (coeff_t)((g_T4[0][0] * E[0] + g_T4[0][1] * E[1] + add) >> shift);
        dst[2 * line] = (coeff_t)((g_T4[2][0] * E[0] + g_T4[2][1] * E[1] + add) >> shift);
        dst[    line] = (coeff_t)((g_T4[1][0] * O[0] + g_T4[1][1] * O[1] + add) >> shift);
        dst[3 * line] = (coeff_t)((g_T4[3][0] * O[0] + g_T4[3][1] * O[1] + add) >> shift);

        src += 4;
        dst++;
    }
}

/* ---------------------------------------------------------------------------
 */
static void partialButterflyInverse4(const coeff_t *src, coeff_t *dst, int shift, int line, int clip_depth)
{
    int E[2], O[2];
    const int max_val = ((1 << clip_depth) >> 1) - 1;
    const int min_val = -max_val - 1;
    const int add     = ((1 << shift) >> 1);
    int j;

    for (j = 0; j < line; j++) {
        /* utilizing symmetry properties to the maximum to
         * minimize the number of multiplications */
        O[0] = g_T4[1][0] * src[line] + g_T4[3][0] * src[3 * line];
        O[1] = g_T4[1][1] * src[line] + g_T4[3][1] * src[3 * line];
        E[0] = g_T4[0][0] * src[0   ] + g_T4[2][0] * src[2 * line];
        E[1] = g_T4[0][1] * src[0   ] + g_T4[2][1] * src[2 * line];

        /* combining even and odd terms at each hierarchy levels to
         * calculate the final spatial domain vector */
        dst[0] = (coeff_t)XAVS2_CLIP3(min_val, max_val, ((E[0] + O[0] + add) >> shift));
        dst[1] = (coeff_t)XAVS2_CLIP3(min_val, max_val, ((E[1] + O[1] + add) >> shift));
        dst[2] = (coeff_t)XAVS2_CLIP3(min_val, max_val, ((E[1] - O[1] + add) >> shift));
        dst[3] = (coeff_t)XAVS2_CLIP3(min_val, max_val, ((E[0] - O[0] + add) >> shift));

        src++;
        dst += 4;
    }
}

/* ---------------------------------------------------------------------------
 */
static void partialButterfly8(const coeff_t *src, coeff_t *dst, int shift, int line)
{
    int E[4], O[4];
    int EE[2], EO[2];
    const int add = ((1 << shift) >> 1);
    int j, k;

    for (j = 0; j < line; j++) {
        /* E and O*/
        for (k = 0; k < 4; k++) {
            E[k] = src[k] + src[7 - k];
            O[k] = src[k] - src[7 - k];
        }

        /* EE and EO */
        EE[0] = E[0] + E[3];
        EO[0] = E[0] - E[3];
        EE[1] = E[1] + E[2];
        EO[1] = E[1] - E[2];

        dst[0       ] = (coeff_t)((g_T8[0][0] * EE[0] + g_T8[0][1] * EE[1] + add) >> shift);
        dst[4 * line] = (coeff_t)((g_T8[4][0] * EE[0] + g_T8[4][1] * EE[1] + add) >> shift);
        dst[2 * line] = (coeff_t)((g_T8[2][0] * EO[0] + g_T8[2][1] * EO[1] + add) >> shift);
        dst[6 * line] = (coeff_t)((g_T8[6][0] * EO[0] + g_T8[6][1] * EO[1] + add) >> shift);

        dst[    line] = (coeff_t)((g_T8[1][0] * O[0] + g_T8[1][1] * O[1] + g_T8[1][2] * O[2] + g_T8[1][3] * O[3] + add) >> shift);
        dst[3 * line] = (coeff_t)((g_T8[3][0] * O[0] + g_T8[3][1] * O[1] + g_T8[3][2] * O[2] + g_T8[3][3] * O[3] + add) >> shift);
        dst[5 * line] = (coeff_t)((g_T8[5][0] * O[0] + g_T8[5][1] * O[1] + g_T8[5][2] * O[2] + g_T8[5][3] * O[3] + add) >> shift);
        dst[7 * line] = (coeff_t)((g_T8[7][0] * O[0] + g_T8[7][1] * O[1] + g_T8[7][2] * O[2] + g_T8[7][3] * O[3] + add) >> shift);

        src += 8;
        dst++;
    }
}

/* ---------------------------------------------------------------------------
 */
static void partialButterflyInverse8(const coeff_t *src, coeff_t *dst, int shift, int line, int clip_depth)
{
    int E[4], O[4];
    int EE[2], EO[2];
    const int max_val = ((1 << clip_depth) >> 1) - 1;
    const int min_val = -max_val - 1;
    const int add     = ((1 << shift) >> 1);
    int j, k;

    for (j = 0; j < line; j++) {
        /* utilizing symmetry properties to the maximum to
         * minimize the number of multiplications */
        for (k = 0; k < 4; k++) {
            O[k] = g_T8[1][k] * src[    line] +
                   g_T8[3][k] * src[3 * line] +
                   g_T8[5][k] * src[5 * line] +
                   g_T8[7][k] * src[7 * line];
        }

        EO[0] = g_T8[2][0] * src[2 * line] + g_T8[6][0] * src[6 * line];
        EO[1] = g_T8[2][1] * src[2 * line] + g_T8[6][1] * src[6 * line];
        EE[0] = g_T8[0][0] * src[0       ] + g_T8[4][0] * src[4 * line];
        EE[1] = g_T8[0][1] * src[0       ] + g_T8[4][1] * src[4 * line];

        /* combining even and odd terms at each hierarchy levels to
         * calculate the final spatial domain vector */
        E[0] = EE[0] + EO[0];
        E[3] = EE[0] - EO[0];
        E[1] = EE[1] + EO[1];
        E[2] = EE[1] - EO[1];

        for (k = 0; k < 4; k++) {
            dst[k    ] = (coeff_t)XAVS2_CLIP3(min_val, max_val, ((E[k    ] + O[k    ] + add) >> shift));
            dst[k + 4] = (coeff_t)XAVS2_CLIP3(min_val, max_val, ((E[3 - k] - O[3 - k] + add) >> shift));
        }

        src++;
        dst += 8;
    }
}

/* ---------------------------------------------------------------------------
 */
static void partialButterfly16(const coeff_t *src, coeff_t *dst, int shift, int line)
{
    int E[8], O[8];
    int EE[4], EO[4];
    int EEE[2], EEO[2];
    const int add = ((1 << shift) >> 1);
    int j, k;

    for (j = 0; j < line; j++) {
        /* E and O*/
        for (k = 0; k < 8; k++) {
            E[k] = src[k] + src[15 - k];
            O[k] = src[k] - src[15 - k];
        }

        /* EE and EO */
        for (k = 0; k < 4; k++) {
            EE[k] = E[k] + E[7 - k];
            EO[k] = E[k] - E[7 - k];
        }

        /* EEE and EEO */
        EEE[0] = EE[0] + EE[3];
        EEO[0] = EE[0] - EE[3];
        EEE[1] = EE[1] + EE[2];
        EEO[1] = EE[1] - EE[2];

        dst[ 0       ] = (coeff_t)((g_T16[ 0][0] * EEE[0] + g_T16[ 0][1] * EEE[1] + add) >> shift);
        dst[ 8 * line] = (coeff_t)((g_T16[ 8][0] * EEE[0] + g_T16[ 8][1] * EEE[1] + add) >> shift);
        dst[ 4 * line] = (coeff_t)((g_T16[ 4][0] * EEO[0] + g_T16[ 4][1] * EEO[1] + add) >> shift);
        dst[12 * line] = (coeff_t)((g_T16[12][0] * EEO[0] + g_T16[12][1] * EEO[1] + add) >> shift);

        for (k = 2; k < 16; k += 4) {
            dst[k * line] = (coeff_t)((g_T16[k][0] * EO[0] +
                                       g_T16[k][1] * EO[1] +
                                       g_T16[k][2] * EO[2] +
                                       g_T16[k][3] * EO[3] + add) >> shift);
        }

        for (k = 1; k < 16; k += 2) {
            dst[k * line] = (coeff_t)((g_T16[k][0] * O[0] +
                                       g_T16[k][1] * O[1] +
                                       g_T16[k][2] * O[2] +
                                       g_T16[k][3] * O[3] +
                                       g_T16[k][4] * O[4] +
                                       g_T16[k][5] * O[5] +
                                       g_T16[k][6] * O[6] +
                                       g_T16[k][7] * O[7] + add) >> shift);
        }

        src += 16;
        dst++;
    }
}

/* ---------------------------------------------------------------------------
 */
static void partialButterflyInverse16(const coeff_t *src, coeff_t *dst, int shift, int line, int clip_depth)
{
    int E[8], O[8];
    int EE[4], EO[4];
    int EEE[2], EEO[2];
    const int max_val = ((1 << clip_depth) >> 1) - 1;
    const int min_val = -max_val - 1;
    const int add     = ((1 << shift) >> 1);
    int j, k;

    for (j = 0; j < line; j++) {
        /* utilizing symmetry properties to the maximum to
         * minimize the number of multiplications */
        for (k = 0; k < 8; k++) {
            O[k] = g_T16[ 1][k] * src[     line] +
                   g_T16[ 3][k] * src[ 3 * line] +
                   g_T16[ 5][k] * src[ 5 * line] +
                   g_T16[ 7][k] * src[ 7 * line] +
                   g_T16[ 9][k] * src[ 9 * line] +
                   g_T16[11][k] * src[11 * line] +
                   g_T16[13][k] * src[13 * line] +
                   g_T16[15][k] * src[15 * line];
        }

        for (k = 0; k < 4; k++) {
            EO[k] = g_T16[ 2][k] * src[ 2 * line] +
                    g_T16[ 6][k] * src[ 6 * line] +
                    g_T16[10][k] * src[10 * line] +
                    g_T16[14][k] * src[14 * line];
        }

        EEO[0] = g_T16[4][0] * src[4 * line] + g_T16[12][0] * src[12 * line];
        EEE[0] = g_T16[0][0] * src[0       ] + g_T16[ 8][0] * src[ 8 * line];
        EEO[1] = g_T16[4][1] * src[4 * line] + g_T16[12][1] * src[12 * line];
        EEE[1] = g_T16[0][1] * src[0       ] + g_T16[ 8][1] * src[ 8 * line];

        /* combining even and odd terms at each hierarchy levels to
         * calculate the final spatial domain vector */
        for (k = 0; k < 2; k++) {
            EE[k    ] = EEE[k    ] + EEO[k    ];
            EE[k + 2] = EEE[1 - k] - EEO[1 - k];
        }

        for (k = 0; k < 4; k++) {
            E[k    ] = EE[k    ] + EO[k    ];
            E[k + 4] = EE[3 - k] - EO[3 - k];
        }

        for (k = 0; k < 8; k++) {
            dst[k    ] = (coeff_t)XAVS2_CLIP3(min_val, max_val, ((E[k    ] + O[k    ] + add) >> shift));
            dst[k + 8] = (coeff_t)XAVS2_CLIP3(min_val, max_val, ((E[7 - k] - O[7 - k] + add) >> shift));
        }

        src++;
        dst += 16;
    }
}

/* ---------------------------------------------------------------------------
 */
static void partialButterfly32(const coeff_t *src, coeff_t *dst, int shift, int line)
{
    int E[16], O[16];
    int EE[8], EO[8];
    int EEE[4], EEO[4];
    int EEEE[2], EEEO[2];
    const int add = ((1 << shift) >> 1);
    int j, k;

    for (j = 0; j < line; j++) {
        /* E and O*/
        for (k = 0; k < 16; k++) {
            E[k] = src[k] + src[31 - k];
            O[k] = src[k] - src[31 - k];
        }

        /* EE and EO */
        for (k = 0; k < 8; k++) {
            EE[k] = E[k] + E[15 - k];
            EO[k] = E[k] - E[15 - k];
        }

        /* EEE and EEO */
        for (k = 0; k < 4; k++) {
            EEE[k] = EE[k] + EE[7 - k];
            EEO[k] = EE[k] - EE[7 - k];
        }

        /* EEEE and EEEO */
        EEEE[0] = EEE[0] + EEE[3];
        EEEO[0] = EEE[0] - EEE[3];
        EEEE[1] = EEE[1] + EEE[2];
        EEEO[1] = EEE[1] - EEE[2];

        dst[ 0       ] = (coeff_t)((g_T32[ 0][0] * EEEE[0] + g_T32[ 0][1] * EEEE[1] + add) >> shift);
        dst[ 8 * line] = (coeff_t)((g_T32[ 8][0] * EEEO[0] + g_T32[ 8][1] * EEEO[1] + add) >> shift);
        dst[16 * line] = (coeff_t)((g_T32[16][0] * EEEE[0] + g_T32[16][1] * EEEE[1] + add) >> shift);
        dst[24 * line] = (coeff_t)((g_T32[24][0] * EEEO[0] + g_T32[24][1] * EEEO[1] + add) >> shift);

        for (k = 4; k < 32; k += 8) {
            dst[k * line] = (coeff_t)((g_T32[k][0] * EEO[0] +
                                       g_T32[k][1] * EEO[1] +
                                       g_T32[k][2] * EEO[2] +
                                       g_T32[k][3] * EEO[3] + add) >> shift);
        }

        for (k = 2; k < 32; k += 4) {
            dst[k * line] = (coeff_t)((g_T32[k][0] * EO[0] +
                                       g_T32[k][1] * EO[1] +
                                       g_T32[k][2] * EO[2] +
                                       g_T32[k][3] * EO[3] +
                                       g_T32[k][4] * EO[4] +
                                       g_T32[k][5] * EO[5] +
                                       g_T32[k][6] * EO[6] +
                                       g_T32[k][7] * EO[7] + add) >> shift);
        }

        for (k = 1; k < 32; k += 2) {
            dst[k * line] = (coeff_t)((g_T32[k][ 0] * O[ 0] +
                                       g_T32[k][ 1] * O[ 1] +
                                       g_T32[k][ 2] * O[ 2] +
                                       g_T32[k][ 3] * O[ 3] +
                                       g_T32[k][ 4] * O[ 4] +
                                       g_T32[k][ 5] * O[ 5] +
                                       g_T32[k][ 6] * O[ 6] +
                                       g_T32[k][ 7] * O[ 7] +
                                       g_T32[k][ 8] * O[ 8] +
                                       g_T32[k][ 9] * O[ 9] +
                                       g_T32[k][10] * O[10] +
                                       g_T32[k][11] * O[11] +
                                       g_T32[k][12] * O[12] +
                                       g_T32[k][13] * O[13] +
                                       g_T32[k][14] * O[14] +
                                       g_T32[k][15] * O[15] + add) >> shift);
        }

        src += 32;
        dst++;
    }
}

/* ---------------------------------------------------------------------------
 */
static void partialButterflyInverse32(const coeff_t *src, coeff_t *dst, int shift, int line, int clip_depth)
{
    int E[16], O[16];
    int EE[8], EO[8];
    int EEE[4], EEO[4];
    int EEEE[2], EEEO[2];
    const int max_val = ((1 << clip_depth) >> 1) - 1;
    const int min_val = -max_val - 1;
    const int add     = ((1 << shift) >> 1);
    int j, k;

    for (j = 0; j < line; j++) {
        /* utilizing symmetry properties to the maximum to
         * minimize the number of multiplications */
        for (k = 0; k < 16; k++) {
            O[k] = g_T32[ 1][k] * src[     line] +
                   g_T32[ 3][k] * src[ 3 * line] +
                   g_T32[ 5][k] * src[ 5 * line] +
                   g_T32[ 7][k] * src[ 7 * line] +
                   g_T32[ 9][k] * src[ 9 * line] +
                   g_T32[11][k] * src[11 * line] +
                   g_T32[13][k] * src[13 * line] +
                   g_T32[15][k] * src[15 * line] +
                   g_T32[17][k] * src[17 * line] +
                   g_T32[19][k] * src[19 * line] +
                   g_T32[21][k] * src[21 * line] +
                   g_T32[23][k] * src[23 * line] +
                   g_T32[25][k] * src[25 * line] +
                   g_T32[27][k] * src[27 * line] +
                   g_T32[29][k] * src[29 * line] +
                   g_T32[31][k] * src[31 * line];
        }

        for (k = 0; k < 8; k++) {
            EO[k] = g_T32[ 2][k] * src[ 2 * line] +
                    g_T32[ 6][k] * src[ 6 * line] +
                    g_T32[10][k] * src[10 * line] +
                    g_T32[14][k] * src[14 * line] +
                    g_T32[18][k] * src[18 * line] +
                    g_T32[22][k] * src[22 * line] +
                    g_T32[26][k] * src[26 * line] +
                    g_T32[30][k] * src[30 * line];
        }

        for (k = 0; k < 4; k++) {
            EEO[k] = g_T32[ 4][k] * src[ 4 * line] +
                     g_T32[12][k] * src[12 * line] +
                     g_T32[20][k] * src[20 * line] +
                     g_T32[28][k] * src[28 * line];
        }

        EEEO[0] = g_T32[8][0] * src[8 * line] + g_T32[24][0] * src[24 * line];
        EEEO[1] = g_T32[8][1] * src[8 * line] + g_T32[24][1] * src[24 * line];
        EEEE[0] = g_T32[0][0] * src[0       ] + g_T32[16][0] * src[16 * line];
        EEEE[1] = g_T32[0][1] * src[0       ] + g_T32[16][1] * src[16 * line];

        /* combining even and odd terms at each hierarchy levels to
         * calculate the final spatial domain vector */
        EEE[0] = EEEE[0] + EEEO[0];
        EEE[3] = EEEE[0] - EEEO[0];
        EEE[1] = EEEE[1] + EEEO[1];
        EEE[2] = EEEE[1] - EEEO[1];
        for (k = 0; k < 4; k++) {
            EE[k    ] = EEE[k    ] + EEO[k    ];
            EE[k + 4] = EEE[3 - k] - EEO[3 - k];
        }

        for (k = 0; k < 8; k++) {
            E[k    ] = EE[k    ] + EO[k    ];
            E[k + 8] = EE[7 - k] - EO[7 - k];
        }

        for (k = 0; k < 16; k++) {
            dst[k     ] = (coeff_t)XAVS2_CLIP3(min_val, max_val, ((E[k     ] + O[k     ] + add) >> shift));
            dst[k + 16] = (coeff_t)XAVS2_CLIP3(min_val, max_val, ((E[15 - k] - O[15 - k] + add) >> shift));
        }

        src++;
        dst += 32;
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void wavelet_64x64_c(const coeff_t *src, coeff_t *dst)
{
    ALIGN32(coeff_t row_buf[64 + LOT_MAX_WLT_TAP * 2]);
    coeff_t *pExt = row_buf + LOT_MAX_WLT_TAP;
    const int N0  = 64;
    const int N1  = 64 >> 1;
    int x, y, offset;

    /* step 1: horizontal transform */
    for (y = 0, offset = 0; y < N0; y++, offset += N0) {
        /* copy */
        memcpy(pExt, src + offset, N0 * sizeof(coeff_t));

        /* reflection */
        pExt[-1    ] = pExt[1     ];
        pExt[-2    ] = pExt[2     ];
        pExt[N0    ] = pExt[N0 - 2];
        pExt[N0 + 1] = pExt[N0 - 3];

        /* filtering (H) */
        for (x = -1; x < N0; x += 2) {
            pExt[x] -= (pExt[x - 1] + pExt[x + 1]) >> 1;
        }

        /* filtering (L) */
        for (x = 0; x < N0; x += 2) {
            pExt[x] += (pExt[x - 1] + pExt[x + 1] + 2) >> 2;
        }

        /* copy */
        for (x = 0; x < N1; x++) {
            dst[x + offset] = pExt[x << 1];
        }
    }

    /* step 2: vertical transform */
    for (x = 0; x < N1; x++) {
        /* copy */
        for (y = 0, offset = 0; y < N0; y++, offset += N0) {
            pExt[y] = dst[x + offset];
        }

        /* reflection */
        pExt[-1    ] = pExt[1     ];
        pExt[-2    ] = pExt[2     ];
        pExt[N0    ] = pExt[N0 - 2];
        pExt[N0 + 1] = pExt[N0 - 3];

        /* filtering (H) */
        for (y = -1; y < N0; y += 2) {
            pExt[y] -= (pExt[y - 1] + pExt[y + 1]) >> 1;
        }

        /* filtering (L) */
        for (y = 0; y < N0; y += 2) {
            pExt[y] = (pExt[y] << 1) + ((pExt[y - 1] + pExt[y + 1] + 1) >> 1);
        }

        /* copy */
        for (y = 0, offset = 0; y < N1; y++, offset += 32) {
            dst[x + offset] = pExt[y << 1];
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void wavelet_64x16_c(const coeff_t *src, coeff_t *dst)
{
    ALIGN32(coeff_t row_buf[64 + LOT_MAX_WLT_TAP * 2]);
    coeff_t *pExt = row_buf + LOT_MAX_WLT_TAP;
    const int N0  = 64;
    const int N1  = 16;
    int x, y, offset;

    /* step 1: horizontal transform */
    for (y = 0, offset = 0; y < N1; y++, offset += N0) {
        /* copy */
        memcpy(pExt, src + offset, N0 * sizeof(coeff_t));

        /* reflection */
        pExt[-1    ] = pExt[1     ];
        pExt[-2    ] = pExt[2     ];
        pExt[N0    ] = pExt[N0 - 2];
        pExt[N0 + 1] = pExt[N0 - 3];

        /* filtering (H) */
        for (x = -1; x < N0; x += 2) {
            pExt[x] -= (pExt[x - 1] + pExt[x + 1]) >> 1;
        }

        /* filtering (L) */
        for (x = 0; x < N0; x += 2) {
            pExt[x] += (pExt[x - 1] + pExt[x + 1] + 2) >> 2;
        }

        /* copy */
        for (x = 0; x < N0 >> 1; x++) {
            dst[x + offset] = pExt[x << 1];
        }
    }

    /* step 2: vertical transform */
    for (x = 0; x < (N0 >> 1); x++) {
        /* copy */
        for (y = 0, offset = 0; y < N1; y++, offset += N0) {
            pExt[y] = dst[x + offset];
        }

        /* reflection */
        pExt[-1    ] = pExt[1     ];
        pExt[-2    ] = pExt[2     ];
        pExt[N1    ] = pExt[N1 - 2];
        pExt[N1 + 1] = pExt[N1 - 3];

        /* filtering (H) */
        for (y = -1; y < N1; y += 2) {
            pExt[y] -= (pExt[y - 1] + pExt[y + 1]) >> 1;
        }

        /* filtering (L) */
        for (y = 0; y < N1; y += 2) {
            pExt[y] = (pExt[y] << 1) + ((pExt[y - 1] + pExt[y + 1] + 1) >> 1);
        }

        /* copy */
        for (y = 0, offset = 0; y < N1 >> 1; y++, offset += N0) {
            dst[x + (offset >> 1)] = pExt[y << 1];
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void wavelet_16x64_c(const coeff_t *src, coeff_t *dst)
{
    ALIGN32(coeff_t row_buf[64 + LOT_MAX_WLT_TAP * 2]);
    coeff_t *pExt = row_buf + LOT_MAX_WLT_TAP;
    const int N0  = 16;
    const int N1  = 64;
    int x, y, offset;

    /* step 1: horizontal transform */
    for (y = 0, offset = 0; y < N1; y++, offset += N0) {
        /* copy */
        memcpy(pExt, src + offset, N0 * sizeof(coeff_t));

        /* reflection */
        pExt[-1    ] = pExt[1     ];
        pExt[-2    ] = pExt[2     ];
        pExt[N0    ] = pExt[N0 - 2];
        pExt[N0 + 1] = pExt[N0 - 3];

        /* filtering (H) */
        for (x = -1; x < N0; x += 2) {
            pExt[x] -= (pExt[x - 1] + pExt[x + 1]) >> 1;
        }

        /* filtering (L) */
        for (x = 0; x < N0; x += 2) {
            pExt[x] += (pExt[x - 1] + pExt[x + 1] + 2) >> 2;
        }

        /* copy */
        for (x = 0; x < N0 >> 1; x++) {
            dst[x + offset] = pExt[x << 1];
        }
    }

    /* step 2: vertical transform */
    for (x = 0; x < (N0 >> 1); x++) {
        /* copy */
        for (y = 0, offset = 0; y < N1; y++, offset += N0) {
            pExt[y] = dst[x + offset];
        }

        /* reflection */
        pExt[-1    ] = pExt[1     ];
        pExt[-2    ] = pExt[2     ];
        pExt[N1    ] = pExt[N1 - 2];
        pExt[N1 + 1] = pExt[N1 - 3];

        /* filtering (H) */
        for (y = -1; y < N1; y += 2) {
            pExt[y] -= (pExt[y - 1] + pExt[y + 1]) >> 1;
        }

        /* filtering (L) */
        for (y = 0; y < N1; y += 2) {
            pExt[y] = (pExt[y] << 1) + ((pExt[y - 1] + pExt[y + 1] + 1) >> 1);
        }

        /* copy */
        for (y = 0, offset = 0; y < N1 >> 1; y++, offset += N0) {
            dst[x + (offset >> 1)] = pExt[y << 1];
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void inv_wavelet_64x64_c(coeff_t *coeff)
{
    ALIGN32(coeff_t row_buf[64 + LOT_MAX_WLT_TAP * 2]);
    coeff_t *pExt = row_buf + LOT_MAX_WLT_TAP;
    const int N0  = 64;
    const int N1  = 64 >> 1;
    int x, y, offset;

    /* step 1: vertical transform */
    for (x = 0; x < N0; x++) {
        /* copy */
        for (y = 0, offset = 0; y < N1; y++, offset += 32) {
            pExt[y << 1] = coeff[x + offset];
        }

        /* reflection */
        pExt[N0] = pExt[N0 - 2];

        /* filtering (even pixel) */
        for (y = 0; y <= N0; y += 2) {
            pExt[y] >>= 1;
        }

        /* filtering (odd pixel) */
        for (y = 1; y < N0; y += 2) {
            pExt[y] = (pExt[y - 1] + pExt[y + 1]) >> 1;
        }

        /* copy */
        for (y = 0, offset = 0; y < N0; y++, offset += N0) {
            coeff[x + offset] = pExt[y];
        }
    }

    /* step 2: horizontal transform */
    for (y = 0, offset = 0; y < N0; y++, offset += N0) {
        /* copy */
        for (x = 0; x < N1; x++) {
            pExt[x << 1] = coeff[offset + x];
        }

        /* reflection */
        pExt[N0] = pExt[N0 - 2];

        /* filtering (odd pixel) */
        for (x = 1; x < N0; x += 2) {
            pExt[x] = (pExt[x - 1] + pExt[x + 1]) >> 1;
        }

        /* copy */
        memcpy(coeff + offset, pExt, N0 * sizeof(coeff_t));
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void inv_wavelet_64x16_c(coeff_t *coeff)
{
    ALIGN32(coeff_t row_buf[64 + LOT_MAX_WLT_TAP * 2]);
    coeff_t *pExt = row_buf + LOT_MAX_WLT_TAP;
    const int N0  = 64;
    const int N1  = 16;
    int x, y, offset;

    /* step 1: vertical transform */
    for (x = 0; x < (N0 >> 1); x++) {
        /* copy */
        for (y = 0, offset = 0; y < N1 >> 1; y++, offset += (N0 >> 1)) {
            pExt[y << 1] = coeff[x + offset];
        }

        /* reflection */
        pExt[N1] = pExt[N1 - 2];

        /* filtering (even pixel) */
        for (y = 0; y <= N1; y += 2) {
            pExt[y] >>= 1;
        }

        /* filtering (odd pixel) */
        for (y = 1; y < N1; y += 2) {
            pExt[y] = (pExt[y - 1] + pExt[y + 1]) >> 1;
        }

        /* copy */
        for (y = 0, offset = 0; y < N1; y++, offset += N0) {
            coeff[x + offset] = pExt[y];
        }
    }

    /* step 2: horizontal transform */
    for (y = 0, offset = 0; y < N1; y++, offset += N0) {
        /* copy */
        for (x = 0; x < N0 >> 1; x++) {
            pExt[x << 1] = coeff[offset + x];
        }

        /* reflection */
        pExt[N0] = pExt[N0 - 2];

        /* filtering (odd pixel) */
        for (x = 1; x < N0; x += 2) {
            pExt[x] = (pExt[x - 1] + pExt[x + 1]) >> 1;
        }

        /* copy */
        memcpy(coeff + offset, pExt, N0 * sizeof(coeff_t));
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void inv_wavelet_16x64_c(coeff_t *coeff)
{
    ALIGN32(coeff_t row_buf[64 + LOT_MAX_WLT_TAP * 2]);
    coeff_t *pExt = row_buf + LOT_MAX_WLT_TAP;
    const int N0 = 16;
    const int N1 = 64;
    int x, y, offset;

    /* step 1: vertical transform */
    for (x = 0; x < (N0 >> 1); x++) {
        /* copy */
        for (y = 0, offset = 0; y < N1 >> 1; y++, offset += (N0 >> 1)) {
            pExt[y << 1] = coeff[x + offset];
        }

        /* reflection */
        pExt[N1] = pExt[N1 - 2];

        /* filtering (even pixel) */
        for (y = 0; y <= N1; y += 2) {
            pExt[y] >>= 1;
        }

        /* filtering (odd pixel) */
        for (y = 1; y < N1; y += 2) {
            pExt[y] = (pExt[y - 1] + pExt[y + 1]) >> 1;
        }

        /* copy */
        for (y = 0, offset = 0; y < N1; y++, offset += N0) {
            coeff[x + offset] = pExt[y];
        }
    }

    /* step 2: horizontal transform */
    for (y = 0, offset = 0; y < N1; y++, offset += N0) {
        /* copy */
        for (x = 0; x < N0 >> 1; x++) {
            pExt[x << 1] = coeff[offset + x];
        }

        /* reflection */
        pExt[N0] = pExt[N0 - 2];

        /* filtering (odd pixel) */
        for (x = 1; x < N0; x += 2) {
            pExt[x] = (pExt[x - 1] + pExt[x + 1]) >> 1;
        }

        /* copy */
        memcpy(coeff + offset, pExt, N0 * sizeof(coeff_t));
    }
}


/**
 * ===========================================================================
 * local function defines for secondary transform
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static void xTr2nd_4_1d_Hor(coeff_t *coeff, int i_coeff, int i_shift, const int16_t *tc)
{
    int tmp_dct[SEC_TR_SIZE * SEC_TR_SIZE];
    const int add = (1 << i_shift) >> 1;
    int i, j, k, sum;

    for (i = 0; i < SEC_TR_SIZE; i++) {
        for (j = 0; j < SEC_TR_SIZE; j++) {
            tmp_dct[i * SEC_TR_SIZE + j] = coeff[i * i_coeff + j];
        }
    }

    for (i = 0; i < SEC_TR_SIZE; i++) {
        for (j = 0; j < SEC_TR_SIZE; j++) {
            sum = add;
            for (k = 0; k < SEC_TR_SIZE; k++) {
                sum += tc[i * SEC_TR_SIZE + k] * tmp_dct[j * SEC_TR_SIZE + k];
            }
            coeff[j * i_coeff + i] = (coeff_t)XAVS2_CLIP3(-32768, 32767, sum >> i_shift);
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void xTr2nd_4_1d_Ver(coeff_t *coeff, int i_coeff, int i_shift, const int16_t *tc)
{
    int tmp_dct[SEC_TR_SIZE * SEC_TR_SIZE];
    const int add = (1 << i_shift) >> 1;
    int i, j, k, sum;

    for (i = 0; i < SEC_TR_SIZE; i++) {
        for (j = 0; j < SEC_TR_SIZE; j++) {
            tmp_dct[i * SEC_TR_SIZE + j] = coeff[i * i_coeff + j];
        }
    }

    for (i = 0; i < SEC_TR_SIZE; i++) {
        for (j = 0; j < SEC_TR_SIZE; j++) {
            sum = add;
            for (k = 0; k < SEC_TR_SIZE; k++) {
                sum += tc[i * SEC_TR_SIZE + k] * tmp_dct[k * SEC_TR_SIZE + j];
            }
            coeff[i * i_coeff + j] = (coeff_t)XAVS2_CLIP3(-32768, 32767, sum >> i_shift);
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void xTr2nd_4_1d_Inv_Ver(coeff_t *coeff, int i_coeff, int i_shift, const int16_t *tc)
{
    int tmp_dct[SEC_TR_SIZE * SEC_TR_SIZE];
    const int add = (1 << i_shift) >> 1;
    int i, j, k, sum;

    for (i = 0; i < SEC_TR_SIZE; i++) {
        for (j = 0; j < SEC_TR_SIZE; j++) {
            tmp_dct[i * SEC_TR_SIZE + j] = coeff[i * i_coeff + j];
        }
    }

    for (i = 0; i < SEC_TR_SIZE; i++) {
        for (j = 0; j < SEC_TR_SIZE; j++) {
            sum = add;
            for (k = 0; k < SEC_TR_SIZE; k++) {
                sum += tc[k * SEC_TR_SIZE + i] * tmp_dct[k * SEC_TR_SIZE + j];
            }
            coeff[i * i_coeff + j] = (coeff_t)XAVS2_CLIP3(-32768, 32767, sum >> i_shift);
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void xTr2nd_4_1d_Inv_Hor(coeff_t *coeff, int i_coeff, int i_shift, int clip_depth, const int16_t *tc)
{
    int tmp_dct[SEC_TR_SIZE * SEC_TR_SIZE];
    const int max_val = ((1 << clip_depth) >> 1) - 1;
    const int min_val = -max_val - 1;
    const int add = (1 << i_shift) >> 1;
    int i, j, k, sum;

    for (i = 0; i < SEC_TR_SIZE; i++) {
        for (j = 0; j < SEC_TR_SIZE; j++) {
            tmp_dct[i * SEC_TR_SIZE + j] = coeff[i * i_coeff + j];
        }
    }

    for (i = 0; i < SEC_TR_SIZE; i++) {
        for (j = 0; j < SEC_TR_SIZE; j++) {
            sum = add;
            for (k = 0; k < SEC_TR_SIZE; k++) {
                sum += tc[k * SEC_TR_SIZE + i] * tmp_dct[j * SEC_TR_SIZE + k];
            }
            coeff[j * i_coeff + i] = (coeff_t)XAVS2_CLIP3(min_val, max_val, sum >> i_shift);
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void transform_4x4_2nd_c(coeff_t *coeff, int i_coeff)
{
    const int shift1 = B4X4_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT + 1;
    const int shift2 = B4X4_IN_BIT + FACTO_BIT + 1;

    xTr2nd_4_1d_Hor(coeff, i_coeff, shift1, g_2T_C);
    xTr2nd_4_1d_Ver(coeff, i_coeff, shift2, g_2T_C);
}

/* ---------------------------------------------------------------------------
 */
static void inv_transform_4x4_2nd_c(coeff_t *coeff, int i_coeff)
{
    const int shift1 = 5;
    const int shift2 = 20 - g_bit_depth + 2;
    const int clip_depth2 = g_bit_depth + 1;

    xTr2nd_4_1d_Inv_Ver(coeff, i_coeff, shift1, g_2T_C);
    xTr2nd_4_1d_Inv_Hor(coeff, i_coeff, shift2, clip_depth2, g_2T_C);
}

/* ---------------------------------------------------------------------------
 * i_mode - real intra mode (luma)
 * b_top  - block top available?
 * b_left - block left available?
 */
static void transform_2nd_c(coeff_t *coeff, int i_coeff, int i_mode, int b_top, int b_left)
{
    int vt = (i_mode >=  0 && i_mode <= 23);
    int ht = (i_mode >= 13 && i_mode <= 32) || (i_mode >= 0 && i_mode <= 2);

    if (vt && b_top) {
        xTr2nd_4_1d_Ver(coeff, i_coeff, 7, g_2T);
    }
    if (ht && b_left) {
        xTr2nd_4_1d_Hor(coeff, i_coeff, 7, g_2T);
    }
}

/* ---------------------------------------------------------------------------
 * i_mode - real intra mode (luma)
 * b_top  - block top available?
 * b_left - block left available?
 */
static void inv_transform_2nd_c(coeff_t *coeff, int i_coeff, int i_mode, int b_top, int b_left)
{
    int vt = (i_mode >=  0 && i_mode <= 23);
    int ht = (i_mode >= 13 && i_mode <= 32) || (i_mode >= 0 && i_mode <= 2);

    if (ht && b_left) {
        xTr2nd_4_1d_Inv_Hor(coeff, i_coeff, 7, 16, g_2T);
    }
    if (vt && b_top) {
        xTr2nd_4_1d_Inv_Ver(coeff, i_coeff, 7, g_2T);
    }
}

/* ---------------------------------------------------------------------------
 */
static void dct_4x4_c(const coeff_t *src, coeff_t *dst, int i_src)
{
#define BSIZE   4
    ALIGN32(coeff_t coeff[BSIZE * BSIZE]);
    ALIGN32(coeff_t block[BSIZE * BSIZE]);
    int shift1 = B4X4_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT;   // 0
    int shift2 = B4X4_IN_BIT + FACTO_BIT;                               // 7
    int i;

    for (i = 0; i < BSIZE; i++) {
        memcpy(&block[i * BSIZE], &src[i * i_src], BSIZE * sizeof(coeff_t));
    }

    // coeff = g_T4 x block^T
    partialButterfly4(block, coeff, shift1, BSIZE);
    // dst = g_T4 x coeff^T = g_T4 x (g_T4 x block ^T)^T = g_T4 x block x g_T4^T
    partialButterfly4(coeff, dst,   shift2, BSIZE);
#undef BSIZE
}

/* ---------------------------------------------------------------------------
 */
static void idct_4x4_c(const coeff_t *src, coeff_t *dst, int i_dst)
{
#define BSIZE   4
    ALIGN32(coeff_t coeff[BSIZE * BSIZE]);
    ALIGN32(coeff_t block[BSIZE * BSIZE]);
    int shift1 = 5;
    int shift2 = 20 - g_bit_depth;
    int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = g_bit_depth + 1;
    int i;

    partialButterflyInverse4(src,   coeff, shift1, BSIZE, clip_depth1);
    partialButterflyInverse4(coeff, block, shift2, BSIZE, clip_depth2);

    for (i = 0; i < BSIZE; i++) {
        memcpy(&dst[i * i_dst], &block[i * BSIZE], BSIZE * sizeof(coeff_t));
    }
#undef BSIZE
}

/* ---------------------------------------------------------------------------
 */
static void dct_8x8_c(const coeff_t *src, coeff_t *dst, int i_src)
{
#define BSIZE   8
    ALIGN32(coeff_t coeff[BSIZE * BSIZE]);
    ALIGN32(coeff_t block[BSIZE * BSIZE]);
    int shift1 = B8X8_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT;
    int shift2 = B8X8_IN_BIT + FACTO_BIT;
    int i;

    for (i = 0; i < BSIZE; i++) {
        memcpy(&block[i * BSIZE], &src[i * i_src], BSIZE * sizeof(coeff_t));
    }

    partialButterfly8(block, coeff, shift1, BSIZE);
    partialButterfly8(coeff, dst,   shift2, BSIZE);
#undef BSIZE
}

/* ---------------------------------------------------------------------------
 */
static void idct_8x8_c(const coeff_t *src, coeff_t *dst, int i_dst)
{
#define BSIZE   8
    ALIGN32(coeff_t coeff[BSIZE * BSIZE]);
    ALIGN32(coeff_t block[BSIZE * BSIZE]);
    int shift1 = 5;
    int shift2 = 20 - g_bit_depth;
    int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = g_bit_depth + 1;
    int i;

    partialButterflyInverse8(src,   coeff, shift1, BSIZE, clip_depth1);
    partialButterflyInverse8(coeff, block, shift2, BSIZE, clip_depth2);

    for (i = 0; i < BSIZE; i++) {
        memcpy(&dst[i * i_dst], &block[i * BSIZE], BSIZE * sizeof(coeff_t));
    }
#undef BSIZE
}

/* ---------------------------------------------------------------------------
 */
static void dct_16x16_c(const coeff_t *src, coeff_t *dst, int i_src)
{
#define BSIZE   16
    ALIGN32(coeff_t coeff[BSIZE * BSIZE]);
    ALIGN32(coeff_t block[BSIZE * BSIZE]);
    int shift1 = B16X16_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT;
    int shift2 = B16X16_IN_BIT + FACTO_BIT;
    int i;

    for (i = 0; i < BSIZE; i++) {
        memcpy(&block[i * BSIZE], &src[i * i_src], BSIZE * sizeof(coeff_t));
    }

    partialButterfly16(block, coeff, shift1, BSIZE);
    partialButterfly16(coeff, dst, shift2, BSIZE);
#undef BSIZE
}

/* ---------------------------------------------------------------------------
 */
static void idct_16x16_c(const coeff_t *src, coeff_t *dst, int i_dst)
{
#define BSIZE   16
    ALIGN32(coeff_t coeff[BSIZE * BSIZE]);
    ALIGN32(coeff_t block[BSIZE * BSIZE]);
    int shift1 = 5;
    int shift2 = 20 - g_bit_depth;
    int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = g_bit_depth + 1;
    int i;

    partialButterflyInverse16(src,   coeff, shift1, BSIZE, clip_depth1);
    partialButterflyInverse16(coeff, block, shift2, BSIZE, clip_depth2);

    for (i = 0; i < BSIZE; i++) {
        memcpy(&dst[i * i_dst], &block[i * BSIZE], BSIZE * sizeof(coeff_t));
    }
#undef BSIZE
}

/* ---------------------------------------------------------------------------
 * NOTE:
 * i_src - the stride of src (the lowest bit is additional wavelet flag)
 */
static void dct_32x32_c(const coeff_t *src, coeff_t *dst, int i_src)
{
#define BSIZE   32
    ALIGN32(coeff_t coeff[BSIZE * BSIZE]);
    ALIGN32(coeff_t block[BSIZE * BSIZE]);
    int shift1 = B32X32_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT + (i_src & 0x01);
    int shift2 = B32X32_IN_BIT + FACTO_BIT;
    int i;

    i_src &= 0xFE;    /* remember to remove the flag bit */
    for (i = 0; i < BSIZE; i++) {
        memcpy(&block[i * BSIZE], &src[i * i_src], BSIZE * sizeof(coeff_t));
    }

    partialButterfly32(block, coeff, shift1, BSIZE);
    partialButterfly32(coeff, dst,   shift2, BSIZE);
#undef BSIZE
}

/* ---------------------------------------------------------------------------
 * NOTE:
 * i_src - the stride of src (the lowest bit is additional wavelet flag)
 */
static void dct_32x32_half_c(const coeff_t *src, coeff_t *dst, int i_src)
{
    int i;
    dct_32x32_c(src, dst, i_src);

    for (i = 0; i < 16; i++) {
        memset(dst + 16, 0, 16 * sizeof(coeff_t));
        dst += 32;
    }
    memset(dst, 0, 32 * 16 * sizeof(coeff_t));
}

/* ---------------------------------------------------------------------------
 * NOTE:
 * i_dst - the stride of dst (the lowest bit is additional wavelet flag)
 */
static void idct_32x32_c(const coeff_t *src, coeff_t *dst, int i_dst)
{
#define BSIZE   32
    ALIGN32(coeff_t coeff[BSIZE * BSIZE]);
    ALIGN32(coeff_t block[BSIZE * BSIZE]);
    int a_flag = i_dst & 0x01;
    int shift1 = 5;
    int shift2 = 20 - g_bit_depth - a_flag;
    int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = g_bit_depth + 1 + a_flag;
    int i;

    i_dst &= 0xFE;    /* remember to remove the flag bit */
    partialButterflyInverse32(src,   coeff, shift1, BSIZE, clip_depth1);
    partialButterflyInverse32(coeff, block, shift2, BSIZE, clip_depth2);

    for (i = 0; i < BSIZE; i++) {
        memcpy(&dst[i * i_dst], &block[i * BSIZE], BSIZE * sizeof(coeff_t));
    }
#undef BSIZE
}

/* ---------------------------------------------------------------------------
 */
static void dct_16x4_c(const coeff_t *src, coeff_t *dst, int i_src)
{
#define BSIZE_H   16
#define BSIZE_V   4
    ALIGN32(coeff_t coeff[BSIZE_H * BSIZE_V]);
    ALIGN32(coeff_t block[BSIZE_H * BSIZE_V]);
    int shift1 = B16X16_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT;
    int shift2 = B16X16_IN_BIT + FACTO_BIT - 2;
    int i;

    for (i = 0; i < BSIZE_V; i++) {
        memcpy(&block[i * BSIZE_H], &src[i * i_src], BSIZE_H * sizeof(coeff_t));
    }

    partialButterfly16(block, coeff, shift1, BSIZE_V);
    partialButterfly4 (coeff, dst,   shift2, BSIZE_H);
#undef BSIZE_H
#undef BSIZE_V
}

/* ---------------------------------------------------------------------------
 */
static void idct_16x4_c(const coeff_t *src, coeff_t *dst, int i_dst)
{
#define BSIZE_H   16
#define BSIZE_V   4
    ALIGN32(coeff_t coeff[BSIZE_H * BSIZE_V]);
    ALIGN32(coeff_t block[BSIZE_H * BSIZE_V]);
    int shift1 = 5;
    int shift2 = 20 - g_bit_depth;
    int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = g_bit_depth + 1;
    int i;

    partialButterflyInverse4 (src,   coeff, shift1, BSIZE_H, clip_depth1);
    partialButterflyInverse16(coeff, block, shift2, BSIZE_V, clip_depth2);

    for (i = 0; i < BSIZE_V; i++) {
        memcpy(&dst[i * i_dst], &block[i * BSIZE_H], BSIZE_H * sizeof(coeff_t));
    }
#undef BSIZE_H
#undef BSIZE_V
}

/* ---------------------------------------------------------------------------
 */
static void dct_4x16_c(const coeff_t *src, coeff_t *dst, int i_src)
{
#define BSIZE_H   4
#define BSIZE_V   16
    ALIGN32(coeff_t coeff[BSIZE_H * BSIZE_V]);
    ALIGN32(coeff_t block[BSIZE_H * BSIZE_V]);
    int shift1 = B16X16_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT - 2;
    int shift2 = B16X16_IN_BIT + FACTO_BIT;
    int i;

    for (i = 0; i < BSIZE_V; i++) {
        memcpy(&block[i * BSIZE_H], &src[i * i_src], BSIZE_H * sizeof(coeff_t));
    }

    partialButterfly4 (block, coeff, shift1, BSIZE_V);
    partialButterfly16(coeff, dst,   shift2, BSIZE_H);
#undef BSIZE_H
#undef BSIZE_V
}

/* ---------------------------------------------------------------------------
 */
static void idct_4x16_c(const coeff_t *src, coeff_t *dst, int i_dst)
{
#define BSIZE_H   4
#define BSIZE_V   16
    ALIGN32(coeff_t coeff[BSIZE_H * BSIZE_V]);
    ALIGN32(coeff_t block[BSIZE_H * BSIZE_V]);
    int shift1 = 5;
    int shift2 = 20 - g_bit_depth;
    int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = g_bit_depth + 1;
    int i;

    partialButterflyInverse16(src,   coeff, shift1, BSIZE_H, clip_depth1);
    partialButterflyInverse4 (coeff, block, shift2, BSIZE_V, clip_depth2);

    for (i = 0; i < BSIZE_V; i++) {
        memcpy(&dst[i * i_dst], &block[i * BSIZE_H], BSIZE_H * sizeof(coeff_t));
    }
#undef BSIZE_H
#undef BSIZE_V
}

/* ---------------------------------------------------------------------------
 * NOTE:
 * i_src - the stride of src (the lowest bit is additional wavelet flag)
 */
static void dct_32x8_c(const coeff_t *src, coeff_t *dst, int i_src)
{
#define BSIZE_H   32
#define BSIZE_V   8
    ALIGN32(coeff_t coeff[BSIZE_H * BSIZE_V]);
    ALIGN32(coeff_t block[BSIZE_H * BSIZE_V]);
    int shift1 = B32X32_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT;
    int shift2 = B32X32_IN_BIT + FACTO_BIT - 2 - (i_src & 0x01);
    int i;

    i_src &= 0xFE;
    for (i = 0; i < BSIZE_V; i++) {
        memcpy(&block[i * BSIZE_H], &src[i * i_src], BSIZE_H * sizeof(coeff_t));
    }

    partialButterfly32(block, coeff, shift1, BSIZE_V);
    partialButterfly8 (coeff, dst,   shift2, BSIZE_H);
#undef BSIZE_H
#undef BSIZE_V
}

/* ---------------------------------------------------------------------------
 * NOTE:
 * i_dst - the stride of dst (the lowest bit is additional wavelet flag)
 */
static void idct_32x8_c(const coeff_t *src, coeff_t *dst, int i_dst)
{
#define BSIZE_H   32
#define BSIZE_V   8
    ALIGN32(coeff_t coeff[BSIZE_H * BSIZE_V]);
    ALIGN32(coeff_t block[BSIZE_H * BSIZE_V]);
    int shift1 = 5;
    int shift2 = 20 - g_bit_depth - (i_dst & 0x01);
    int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = g_bit_depth + 1 + (i_dst & 0x01);
    int i;

    partialButterflyInverse8 (src,   coeff, shift1, BSIZE_H, clip_depth1);
    partialButterflyInverse32(coeff, block, shift2, BSIZE_V, clip_depth2);

    i_dst &= 0xFE;
    for (i = 0; i < BSIZE_V; i++) {
        memcpy(&dst[i * i_dst], &block[i * BSIZE_H], BSIZE_H * sizeof(coeff_t));
    }
#undef BSIZE_H
#undef BSIZE_V
}

/* ---------------------------------------------------------------------------
 * NOTE:
 * i_src - the stride of src (the lowest bit is additional wavelet flag)
 */
static void dct_8x32_c(const coeff_t *src, coeff_t *dst, int i_src)
{
#define BSIZE_H   8
#define BSIZE_V   32
    ALIGN32(coeff_t coeff[BSIZE_H * BSIZE_V]);
    ALIGN32(coeff_t block[BSIZE_H * BSIZE_V]);
    int shift1 = B32X32_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT - 2 + (i_src & 0x01);
    int shift2 = B32X32_IN_BIT + FACTO_BIT;
    int i;

    i_src &= 0xFE;
    for (i = 0; i < BSIZE_V; i++) {
        memcpy(&block[i * BSIZE_H], &src[i * i_src], BSIZE_H * sizeof(coeff_t));
    }

    partialButterfly8 (block, coeff, shift1, BSIZE_V);
    partialButterfly32(coeff, dst,   shift2, BSIZE_H);
#undef BSIZE_H
#undef BSIZE_V
}

/* ---------------------------------------------------------------------------
 * NOTE:
 * i_dst - the stride of dst (the lowest bit is additional wavelet flag)
 */
static void idct_8x32_c(const coeff_t *src, coeff_t *dst, int i_dst)
{
#define BSIZE_H   8
#define BSIZE_V   32
    ALIGN32(coeff_t coeff[BSIZE_H * BSIZE_V]);
    ALIGN32(coeff_t block[BSIZE_H * BSIZE_V]);
    int shift1 = 5;
    int shift2 = 20 - g_bit_depth - (i_dst & 0x01);
    int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = g_bit_depth + 1 + (i_dst & 0x01);
    int i;

    partialButterflyInverse32(src,   coeff, shift1, BSIZE_H, clip_depth1);
    partialButterflyInverse8 (coeff, block, shift2, BSIZE_V, clip_depth2);

    i_dst &= 0xFE;
    for (i = 0; i < BSIZE_V; i++) {
        memcpy(&dst[i * i_dst], &block[i * BSIZE_H], BSIZE_H * sizeof(coeff_t));
    }
#undef BSIZE_H
#undef BSIZE_V
}

/* ---------------------------------------------------------------------------
 * NOTE:
 * i_src - the stride of src (the lowest bit is additional wavelet flag)
 */
static void dct_64x64_c(const coeff_t *src, coeff_t *dst, int i_src)
{
    UNUSED_PARAMETER(i_src);
    wavelet_64x64_c(src, dst);
    dct_32x32_c(dst, dst, 32 | 0x01);  /* 32x32 dct */
}

/* ---------------------------------------------------------------------------
 * NOTE:
 * i_src - the stride of src (the lowest bit is additional wavelet flag)
 */
static void dct_64x64_half_c(const coeff_t *src, coeff_t *dst, int i_src)
{
    UNUSED_PARAMETER(i_src);
    wavelet_64x64_c(src, dst);
    dct_32x32_half_c(dst, dst, 32 | 0x01);  /* 32x32 dct */
}


/* ---------------------------------------------------------------------------
 * NOTE:
 * i_dst - the stride of dst (the lowest bit is additional wavelet flag)
 */
static void idct_64x64_c(const coeff_t *src, coeff_t *dst, int i_dst)
{
    UNUSED_PARAMETER(i_dst);
    idct_32x32_c(src, dst, 32 | 0x01); /* 32x32 idct */
    inv_wavelet_64x64_c(dst);
}

/* ---------------------------------------------------------------------------
 * NOTE:
 * i_src - the stride of src (the lowest bit is additional wavelet flag)
 */
static void dct_64x16_c(const coeff_t *src, coeff_t *dst, int i_src)
{
    UNUSED_PARAMETER(i_src);
    wavelet_64x16_c(src, dst);
    dct_32x8_c(dst, dst, 32 | 0x01);
}

/* ---------------------------------------------------------------------------
 * NOTE:
 * i_dst - the stride of dst (the lowest bit is additional wavelet flag)
 */
static void idct_64x16_c(const coeff_t *src, coeff_t *dst, int i_dst)
{
    UNUSED_PARAMETER(i_dst);
    idct_32x8_c(src, dst, 32 | 0x01);
    inv_wavelet_64x16_c(dst);
}


/* ---------------------------------------------------------------------------
 * NOTE:
 * i_src - the stride of src (the lowest bit is additional wavelet flag)
 */
static void dct_16x64_c(const coeff_t *src, coeff_t *dst, int i_src)
{
    UNUSED_PARAMETER(i_src);
    wavelet_16x64_c(src, dst);
    dct_8x32_c(dst, dst, 8 | 0x01);
}

/* ---------------------------------------------------------------------------
 * NOTE:
 * i_dst - the stride of dst (the lowest bit is additional wavelet flag)
 */
static void idct_16x64_c(const coeff_t *src, coeff_t *dst, int i_dst)
{
    UNUSED_PARAMETER(i_dst);
    idct_8x32_c(src, dst, 8 | 0x01);
    inv_wavelet_16x64_c(dst);
}


/**
 * ===========================================================================
 * interface function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * init dct function handles
 */
void xavs2_dct_init(uint32_t cpuid, dct_funcs_t *dctf)
{
    /* -------------------------------------------------------------
     * set handles with default c functions
     */

    /* dct: square */
    dctf->dct [LUMA_4x4  ] = dct_4x4_c;
    dctf->dct [LUMA_8x8  ] = dct_8x8_c;
    dctf->dct [LUMA_16x16] = dct_16x16_c;
    dctf->dct [LUMA_32x32] = dct_32x32_c;
    dctf->dct [LUMA_64x64] = dct_64x64_c;

    /* dct: asymmetrical */
    dctf->dct [LUMA_16x4 ] = dct_16x4_c;
    dctf->dct [LUMA_4x16 ] = dct_4x16_c;
    dctf->dct [LUMA_32x8 ] = dct_32x8_c;
    dctf->dct [LUMA_8x32 ] = dct_8x32_c;
    dctf->dct [LUMA_64x16] = dct_64x16_c;
    dctf->dct [LUMA_16x64] = dct_16x64_c;

    /* idct: square */
    dctf->idct[LUMA_4x4  ] = idct_4x4_c;
    dctf->idct[LUMA_8x8  ] = idct_8x8_c;
    dctf->idct[LUMA_16x16] = idct_16x16_c;
    dctf->idct[LUMA_32x32] = idct_32x32_c;
    dctf->idct[LUMA_64x64] = idct_64x64_c;

    /* idct: asymmetrical */
    dctf->idct[LUMA_16x4 ] = idct_16x4_c;
    dctf->idct[LUMA_4x16 ] = idct_4x16_c;
    dctf->idct[LUMA_32x8 ] = idct_32x8_c;
    dctf->idct[LUMA_8x32 ] = idct_8x32_c;
    dctf->idct[LUMA_64x16] = idct_64x16_c;
    dctf->idct[LUMA_16x64] = idct_16x64_c;

    /* 2nd transform */
    dctf->transform_4x4_2nd     = transform_4x4_2nd_c;
    dctf->inv_transform_4x4_2nd = inv_transform_4x4_2nd_c;
    dctf->transform_2nd         = transform_2nd_c;
    dctf->inv_transform_2nd     = inv_transform_2nd_c;

    /* DCT half */
    dctf->dct_half[LUMA_32x32] = dct_32x32_half_c;
    dctf->dct_half[LUMA_64x64] = dct_64x64_half_c;

#if HAVE_MMX
    /* -------------------------------------------------------------
     * set handles with asm functions
     */

    /* functions defined in file intrinsic_dct.c */
    if (cpuid & XAVS2_CPU_SSE42) {
        /* dct: square */
        dctf->dct [LUMA_4x4  ] = dct_c_4x4_sse128;
        dctf->dct [LUMA_8x8  ] = dct_c_8x8_sse128;
        dctf->dct [LUMA_16x16] = dct_c_16x16_sse128;
        dctf->dct [LUMA_32x32] = dct_c_32x32_sse128;
        dctf->dct [LUMA_64x64] = dct_c_64x64_sse128;

        /* dct: asymmetrical */
        dctf->dct[LUMA_16x4 ] = dct_c_16x4_sse128;
        dctf->dct[LUMA_4x16 ] = dct_c_4x16_sse128;//第一次变换没写移位
        dctf->dct[LUMA_32x8 ] = dct_c_32x8_sse128;
        dctf->dct[LUMA_8x32 ] = dct_c_8x32_sse128;
        dctf->dct[LUMA_64x16] = dct_c_64x16_sse128;
        dctf->dct[LUMA_16x64] = dct_c_16x64_sse128;

        /* idct: square */
        dctf->idct[LUMA_4x4  ] = idct_c_4x4_sse128;
        dctf->idct[LUMA_8x8  ] = idct_c_8x8_sse128;
        dctf->idct[LUMA_16x16] = idct_c_16x16_sse128;
        dctf->idct[LUMA_32x32] = idct_c_32x32_sse128;
        dctf->idct[LUMA_64x64] = idct_c_64x64_sse128;

        /* idct: asymmetrical */
        dctf->idct[LUMA_16x4 ] = idct_c_16x4_sse128;
        dctf->idct[LUMA_4x16 ] = idct_c_4x16_sse128;
        dctf->idct[LUMA_32x8 ] = idct_c_32x8_sse128;
        dctf->idct[LUMA_8x32 ] = idct_c_8x32_sse128;
        dctf->idct[LUMA_64x16] = idct_c_64x16_sse128;
        dctf->idct[LUMA_16x64] = idct_c_16x64_sse128;

        /* 2nd transform */
        dctf->transform_4x4_2nd     = transform_4x4_2nd_sse128;
        dctf->inv_transform_4x4_2nd = inv_transform_4x4_2nd_sse128;
        dctf->transform_2nd         = transform_2nd_sse128;
        dctf->inv_transform_2nd     = inv_transform_2nd_sse128;

        // half  transform
        dctf->dct_half[LUMA_32x32] = dct_c_32x32_half_sse128;
        dctf->dct_half[LUMA_64x64] = dct_c_64x64_half_sse128;
    }

    if (cpuid & XAVS2_CPU_SSE2) {
        dctf->dct [LUMA_4x4  ] = xavs2_dct_4x4_sse2;
        dctf->dct [LUMA_8x8  ] = xavs2_dct_8x8_sse2;

        dctf->idct[LUMA_4x4  ] = xavs2_idct_4x4_sse2;
#if ARCH_X86_64
        dctf->idct[LUMA_8x8  ] = xavs2_idct_8x8_sse2;
#endif
    }

    if (cpuid & XAVS2_CPU_SSSE3) {
        dctf->idct[LUMA_8x8  ] = xavs2_idct_8x8_ssse3;
    }

    if (cpuid & XAVS2_CPU_SSE4) {
        dctf->dct[LUMA_8x8   ] = xavs2_dct_8x8_sse4;
    }

    if (cpuid & XAVS2_CPU_AVX2) {

        dctf->dct [LUMA_4x4   ] = xavs2_dct_4x4_avx2;
#if ARCH_X86_64
        dctf->dct [LUMA_8x8   ] = xavs2_dct_8x8_avx2;
        dctf->dct [LUMA_16x16 ] = xavs2_dct_16x16_avx2; // slower than dct_16x16_avx2
        dctf->dct [LUMA_32x32 ] = xavs2_dct_32x32_avx2;

        dctf->idct[LUMA_4x4   ] = xavs2_idct_4x4_avx2;
        dctf->idct[LUMA_8x8   ] = xavs2_idct_8x8_avx2;
        dctf->idct[LUMA_16x16 ] = xavs2_idct_16x16_avx2;
        dctf->idct[LUMA_32x32 ] = xavs2_idct_32x32_avx2;
#endif
    }


#if ARCH_X86_64
    if (cpuid & XAVS2_CPU_AVX2) {
        // dctf->dct[LUMA_4x4 ] = dct_c_4x4_avx2;   /* futl: dct_4x4_avx2的速度比dct_4x4_sse128略慢一点 */
        // dctf->dct[LUMA_8x8 ] = dct_c_8x8_avx2;   /* futl: dct_8x8_avx2的速度比xavs2_dct_8x8_avx2慢 */
        // dctf->dct[LUMA_4x16] = dct_c_4x16_avx2; /* futl: dct_4x16_avx2的速度比dct_4x16_sse128慢 */
        dctf->dct[LUMA_16x4 ] = dct_c_16x4_avx2;   /* 姜波：速度比sse128快两倍 */
        dctf->dct[LUMA_8x32 ] = dct_c_8x32_avx2;
        dctf->dct[LUMA_32x8 ] = dct_c_32x8_avx2;
        dctf->dct[LUMA_16x16] = dct_c_16x16_avx2;

        // dctf->dct[LUMA_32x32] = dct_c_32x32_avx2; /* asm faster than intrinsic */

        dctf->dct[LUMA_64x64] = dct_c_64x64_avx2;
        dctf->dct[LUMA_64x16] = dct_c_64x16_avx2;
        dctf->dct[LUMA_16x64] = dct_c_16x64_avx2;

        dctf->idct[LUMA_8x8]   = idct_c_8x8_avx2;
        dctf->idct[LUMA_16x16] = idct_c_16x16_avx2;
        dctf->idct[LUMA_32x32] = idct_c_32x32_avx2;
        dctf->idct[LUMA_64x64] = idct_c_64x64_avx2;
        dctf->idct[LUMA_64x16] = idct_c_64x16_avx2;
        dctf->idct[LUMA_16x64] = idct_c_16x64_avx2;

        dctf->dct_half[LUMA_32x32] = dct_c_32x32_half_avx2;
        dctf->dct_half[LUMA_64x64] = dct_c_64x64_half_avx2;
    }
#endif  // ARCH_X86_64
#else
    UNUSED_PARAMETER(cpuid);
#endif  // if HAVE_MMX
}

