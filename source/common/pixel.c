/*
 * pixel.c
 *
 * Description of this file:
 *    Pixel processing functions definition of the xavs2 library
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
#include "pixel.h"
#include "cpu.h"
#include <stdlib.h>

#if HAVE_MMX
#include "vec/intrinsic.h"
#include "x86/pixel.h"
#include "x86/blockcopy8.h"
#include "x86/pixel-util.h"
#endif


/**
 * ===========================================================================
 * global variables
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * partition map table
 */
#define INVALID LUMA_INVALID
const uint8_t g_partition_map_tab[] = {
    //  4         8             12          16         20         24         28           32        36       40      44         48        52       56       60       64
    LUMA_4x4,   LUMA_4x8,     INVALID,  LUMA_4x16,   INVALID,    INVALID,   INVALID,    INVALID,  INVALID, INVALID, INVALID,    INVALID, INVALID, INVALID, INVALID,    INVALID,      // 4
    LUMA_8x4,   LUMA_8x8,     INVALID,  LUMA_8x16,   INVALID,    INVALID,   INVALID,  LUMA_8x32,  INVALID, INVALID, INVALID,    INVALID, INVALID, INVALID, INVALID,    INVALID,      // 8
    INVALID,    INVALID,     INVALID, LUMA_12x16,   INVALID,    INVALID,   INVALID,    INVALID,  INVALID, INVALID, INVALID,    INVALID, INVALID, INVALID, INVALID,    INVALID,      // 12
    LUMA_16x4,  LUMA_16x8,  LUMA_16x12, LUMA_16x16,   INVALID,    INVALID,   INVALID, LUMA_16x32,  INVALID, INVALID, INVALID,    INVALID, INVALID, INVALID, INVALID, LUMA_16x64,   // 16
    INVALID,    INVALID,     INVALID,    INVALID,   INVALID,    INVALID,   INVALID,    INVALID,  INVALID, INVALID, INVALID,    INVALID, INVALID, INVALID, INVALID,    INVALID,      // 20
    INVALID,    INVALID,     INVALID,    INVALID,   INVALID,    INVALID,   INVALID, LUMA_24x32,  INVALID, INVALID, INVALID,    INVALID, INVALID, INVALID, INVALID,    INVALID,      // 24
    INVALID,    INVALID,     INVALID,    INVALID,   INVALID,    INVALID,   INVALID,    INVALID,  INVALID, INVALID, INVALID,    INVALID, INVALID, INVALID, INVALID,    INVALID,      // 28
    INVALID,  LUMA_32x8,     INVALID, LUMA_32x16,   INVALID, LUMA_32x24,   INVALID, LUMA_32x32,  INVALID, INVALID, INVALID,    INVALID, INVALID, INVALID, INVALID, LUMA_32x64,   // 32
    INVALID,    INVALID,     INVALID,    INVALID,   INVALID,    INVALID,   INVALID,    INVALID,  INVALID, INVALID, INVALID,    INVALID, INVALID, INVALID, INVALID,    INVALID,      // 36
    INVALID,    INVALID,     INVALID,    INVALID,   INVALID,    INVALID,   INVALID,    INVALID,  INVALID, INVALID, INVALID,    INVALID, INVALID, INVALID, INVALID,    INVALID,      // 40
    INVALID,    INVALID,     INVALID,    INVALID,   INVALID,    INVALID,   INVALID,    INVALID,  INVALID, INVALID, INVALID,    INVALID, INVALID, INVALID, INVALID,    INVALID,      // 44
    INVALID,    INVALID,     INVALID,    INVALID,   INVALID,    INVALID,   INVALID,    INVALID,  INVALID, INVALID, INVALID,    INVALID, INVALID, INVALID, INVALID, LUMA_48x64,   // 48
    INVALID,    INVALID,     INVALID,    INVALID,   INVALID,    INVALID,   INVALID,    INVALID,  INVALID, INVALID, INVALID,    INVALID, INVALID, INVALID, INVALID,    INVALID,      // 52
    INVALID,    INVALID,     INVALID,    INVALID,   INVALID,    INVALID,   INVALID,    INVALID,  INVALID, INVALID, INVALID,    INVALID, INVALID, INVALID, INVALID,    INVALID,      // 56
    INVALID,    INVALID,     INVALID,    INVALID,   INVALID,    INVALID,   INVALID,    INVALID,  INVALID, INVALID, INVALID,    INVALID, INVALID, INVALID, INVALID,    INVALID,      // 60
    INVALID,    INVALID,     INVALID, LUMA_64x16,   INVALID,    INVALID,   INVALID, LUMA_64x32,  INVALID, INVALID, INVALID, LUMA_64x48, INVALID, INVALID, INVALID, LUMA_64x64    // 64
};
#undef INVALID


/**
 * ===========================================================================
 * local function defines
 * ===========================================================================
 */

/**
 * ---------------------------------------------------------------------------
 * SAD
 * ---------------------------------------------------------------------------
 */
#define PIXEL_SAD_C(w, h) \
static cmp_dist_t xavs2_pixel_sad_##w##x##h(const pel_t *pix1, intptr_t i_pix1, const pel_t *pix2, intptr_t i_pix2)\
{\
    cmp_dist_t sum = 0;\
    int x, y;\
    for (y = 0; y < h; y++) {\
        for (x = 0; x < w; x += 4) {\
            sum += abs(pix1[x]     - pix2[x]);\
            sum += abs(pix1[x + 1] - pix2[x + 1]);\
            sum += abs(pix1[x + 2] - pix2[x + 2]);\
            sum += abs(pix1[x + 3] - pix2[x + 3]);\
        }\
        pix1 += i_pix1;\
        pix2 += i_pix2;\
    }\
    return sum;\
}

PIXEL_SAD_C(64, 64)     /* 64x64 */
PIXEL_SAD_C(64, 32)
PIXEL_SAD_C(32, 64)
PIXEL_SAD_C(64, 16)
PIXEL_SAD_C(64, 48)
PIXEL_SAD_C(16, 64)
PIXEL_SAD_C(48, 64)
PIXEL_SAD_C(32, 32)     /* 32x32 */
PIXEL_SAD_C(32, 16)
PIXEL_SAD_C(16, 32)
PIXEL_SAD_C(32,  8)
PIXEL_SAD_C(32, 24)
PIXEL_SAD_C( 8, 32)
PIXEL_SAD_C(24, 32)
PIXEL_SAD_C(16, 16)     /* 16x16 */
PIXEL_SAD_C(16,  8)
PIXEL_SAD_C( 8, 16)
PIXEL_SAD_C(16,  4)
PIXEL_SAD_C(16, 12)
PIXEL_SAD_C( 4, 16)
PIXEL_SAD_C(12, 16)
PIXEL_SAD_C( 8,  8)     /* 8x8 */
PIXEL_SAD_C( 8,  4)
PIXEL_SAD_C( 4,  8)
PIXEL_SAD_C( 4,  4)     /* 4x4 */


/**
 * ---------------------------------------------------------------------------
 * SAD x3
 * ---------------------------------------------------------------------------
 */
#define PIXEL_SAD_X3_C(w, h) \
void xavs2_pixel_sad_x3_##w##x##h(const pel_t* pix1, const pel_t* pix2, const pel_t* pix3, const pel_t* pix4, intptr_t i_fref_stride, int32_t* res)\
{\
    int x, y;\
    res[0] = 0;\
    res[1] = 0;\
    res[2] = 0;\
    for (y = 0; y < h; y++) {\
        for (x = 0; x < w; x++) {\
            res[0] += abs(pix1[x] - pix2[x]);\
            res[1] += abs(pix1[x] - pix3[x]);\
            res[2] += abs(pix1[x] - pix4[x]);\
        }\
        pix1 += FENC_STRIDE;\
        pix2 += i_fref_stride;\
        pix3 += i_fref_stride;\
        pix4 += i_fref_stride;\
    }\
}

PIXEL_SAD_X3_C(64, 64)  /* 64x64 */
PIXEL_SAD_X3_C(64, 32)
PIXEL_SAD_X3_C(32, 64)
PIXEL_SAD_X3_C(64, 16)
PIXEL_SAD_X3_C(64, 48)
PIXEL_SAD_X3_C(16, 64)
PIXEL_SAD_X3_C(48, 64)
PIXEL_SAD_X3_C(32, 32)  /* 32x32 */
PIXEL_SAD_X3_C(32, 16)
PIXEL_SAD_X3_C(16, 32)
PIXEL_SAD_X3_C(32,  8)
PIXEL_SAD_X3_C(32, 24)
PIXEL_SAD_X3_C( 8, 32)
PIXEL_SAD_X3_C(24, 32)
PIXEL_SAD_X3_C(16, 16)  /* 16x16 */
PIXEL_SAD_X3_C(16,  8)
PIXEL_SAD_X3_C( 8, 16)
PIXEL_SAD_X3_C(16,  4)
PIXEL_SAD_X3_C(16, 12)
PIXEL_SAD_X3_C( 4, 16)
PIXEL_SAD_X3_C(12, 16)
PIXEL_SAD_X3_C( 8,  8)  /* 8x8 */
PIXEL_SAD_X3_C( 8,  4)
PIXEL_SAD_X3_C( 4,  8)
PIXEL_SAD_X3_C( 4,  4)  /* 4x4 */


/**
 * ---------------------------------------------------------------------------
 * SAD x4
 * ---------------------------------------------------------------------------
 */

#define PIXEL_SAD_X4_C(w, h) \
void xavs2_pixel_sad_x4_##w##x##h(const pel_t* pix1, const pel_t* pix2, const pel_t* pix3, const pel_t* pix4, const pel_t* pix5, intptr_t i_fref_stride, int32_t* res)\
{\
    int x, y;\
    res[0] = 0;\
    res[1] = 0;\
    res[2] = 0;\
    res[3] = 0;\
    for (y = 0; y < h; y++) {\
        for (x = 0; x < w; x++) {\
            res[0] += abs(pix1[x] - pix2[x]);\
            res[1] += abs(pix1[x] - pix3[x]);\
            res[2] += abs(pix1[x] - pix4[x]);\
            res[3] += abs(pix1[x] - pix5[x]);\
        }\
        pix1 += FENC_STRIDE;\
        pix2 += i_fref_stride;\
        pix3 += i_fref_stride;\
        pix4 += i_fref_stride;\
        pix5 += i_fref_stride;\
    }\
}

PIXEL_SAD_X4_C(64, 64)  /* 64x64 */
PIXEL_SAD_X4_C(64, 32)
PIXEL_SAD_X4_C(32, 64)
PIXEL_SAD_X4_C(64, 16)
PIXEL_SAD_X4_C(64, 48)
PIXEL_SAD_X4_C(16, 64)
PIXEL_SAD_X4_C(48, 64)
PIXEL_SAD_X4_C(32, 32)  /* 32x32 */
PIXEL_SAD_X4_C(32, 16)
PIXEL_SAD_X4_C(16, 32)
PIXEL_SAD_X4_C(32,  8)
PIXEL_SAD_X4_C(32, 24)
PIXEL_SAD_X4_C( 8, 32)
PIXEL_SAD_X4_C(24, 32)
PIXEL_SAD_X4_C(16, 16)  /* 16x16 */
PIXEL_SAD_X4_C(16,  8)
PIXEL_SAD_X4_C( 8, 16)
PIXEL_SAD_X4_C(16,  4)
PIXEL_SAD_X4_C(16, 12)
PIXEL_SAD_X4_C( 4, 16)
PIXEL_SAD_X4_C(12, 16)
PIXEL_SAD_X4_C( 8,  8)  /* 8x8 */
PIXEL_SAD_X4_C( 8,  4)
PIXEL_SAD_X4_C( 4,  8)
PIXEL_SAD_X4_C( 4,  4)  /* 4x4 */


/**
 * ---------------------------------------------------------------------------
 * SATD
 * ---------------------------------------------------------------------------
 */
#define BITS_PER_SUM (8 * sizeof(uint16_t))

#define HADAMARD4(d0, d1, d2, d3, s0, s1, s2, s3) \
{\
    uint32_t t0 = s0 + s1;\
    uint32_t t1 = s0 - s1;\
    uint32_t t2 = s2 + s3;\
    uint32_t t3 = s2 - s3;\
    d0 = t0 + t2;\
    d2 = t0 - t2;\
    d1 = t1 + t3;\
    d3 = t1 - t3;\
}

#define HADAMARD4_10bit(d0, d1, d2, d3, s0, s1, s2, s3) \
{\
    uint64_t t0 = s0 + s1; \
    uint64_t t1 = s0 - s1; \
    uint64_t t2 = s2 + s3; \
    uint64_t t3 = s2 - s3; \
    d0 = t0 + t2; \
    d2 = t0 - t2; \
    d1 = t1 + t3; \
    d3 = t1 - t3; \
}



/* ---------------------------------------------------------------------------
 * in: a pseudo-simd number of the form x+(y<<16)
 * return: abs(x) + (abs(y)<<16)
 */
ALWAYS_INLINE uint32_t abs2(uint32_t a)
{
    uint32_t s = ((a >> (BITS_PER_SUM - 1)) & (((uint32_t)1 << BITS_PER_SUM) + 1)) * ((uint16_t)-1);
    return (a + s) ^ s;
}

ALWAYS_INLINE uint64_t abs2_10bit(uint64_t a)
{
    uint64_t s = ((a >> (BITS_PER_SUM - 1)) & (((uint64_t)1 << BITS_PER_SUM) + 1)) * ((uint64_t)-1);
    return (a + s) ^ s;
}

/* ---------------------------------------------------------------------------
 */
static cmp_dist_t xavs2_pixel_satd_4x4(const pel_t *pix1, intptr_t i_pix1, const pel_t *pix2, intptr_t i_pix2)
{
    uint32_t tmp[4][2];
    uint32_t a0, a1, a2, a3, b0, b1;
    cmp_dist_t sum = 0;
    int i;

    for (i = 0; i < 4; i++, pix1 += i_pix1, pix2 += i_pix2) {
        a0 = pix1[0] - pix2[0];
        a1 = pix1[1] - pix2[1];
        b0 = (a0 + a1) + ((a0 - a1) << BITS_PER_SUM);
        a2 = pix1[2] - pix2[2];
        a3 = pix1[3] - pix2[3];
        b1 = (a2 + a3) + ((a2 - a3) << BITS_PER_SUM);
        tmp[i][0] = b0 + b1;
        tmp[i][1] = b0 - b1;
    }

    for (i = 0; i < 2; i++) {
        HADAMARD4(a0, a1, a2, a3, tmp[0][i], tmp[1][i], tmp[2][i], tmp[3][i]);
        a0 = abs2(a0) + abs2(a1) + abs2(a2) + abs2(a3);
        sum += ((uint16_t)a0) + (a0 >> BITS_PER_SUM);
    }

    return (sum >> 1);
}

/* ---------------------------------------------------------------------------
 * SWAR version of satd 8x4, performs two 4x4 SATDs at once
 */
static cmp_dist_t xavs2_pixel_satd_8x4(const pel_t *pix1, intptr_t i_pix1, const pel_t *pix2, intptr_t i_pix2)
{
    uint32_t tmp[4][4];
    uint32_t a0, a1, a2, a3;
    cmp_dist_t sum = 0;
    int i;

    for (i = 0; i < 4; i++, pix1 += i_pix1, pix2 += i_pix2) {
        a0 = (pix1[0] - pix2[0]) + ((uint32_t)(pix1[4] - pix2[4]) << BITS_PER_SUM);
        a1 = (pix1[1] - pix2[1]) + ((uint32_t)(pix1[5] - pix2[5]) << BITS_PER_SUM);
        a2 = (pix1[2] - pix2[2]) + ((uint32_t)(pix1[6] - pix2[6]) << BITS_PER_SUM);
        a3 = (pix1[3] - pix2[3]) + ((uint32_t)(pix1[7] - pix2[7]) << BITS_PER_SUM);
        HADAMARD4(tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], a0, a1, a2, a3);
    }

    for (i = 0; i < 4; i++) {
        HADAMARD4(a0, a1, a2, a3, tmp[0][i], tmp[1][i], tmp[2][i], tmp[3][i]);
        sum += abs2(a0) + abs2(a1) + abs2(a2) + abs2(a3);
    }

    return (((uint16_t)sum) + (sum >> BITS_PER_SUM)) >> 1;
}


/* ---------------------------------------------------------------------------
 * calculate satd in blocks of 4x4
 */
#define PIXEL_SATD4_C(w, h) \
static cmp_dist_t xavs2_pixel_satd_##w##x##h(const pel_t *pix1, intptr_t i_pix1, const pel_t *pix2, intptr_t i_pix2)\
{\
    cmp_dist_t satd = 0;\
    int y, x;\
    for (y = 0; y < h; y += 4) {\
        for (x = 0; x < w; x += 4) {\
            satd += xavs2_pixel_satd_4x4(pix1 + y * i_pix1 + x, i_pix1,\
                                       pix2 + y * i_pix2 + x, i_pix2);\
        }\
    }\
    return satd;\
}

/* ---------------------------------------------------------------------------
 * calculate satd in blocks of 8x4
 */
#define PIXEL_SATD8_C(w, h) \
static cmp_dist_t xavs2_pixel_satd_##w##x##h(const pel_t *pix1, intptr_t i_pix1, const pel_t *pix2, intptr_t i_pix2)\
{\
    cmp_dist_t satd = 0;\
    int y, x;\
    for (y = 0; y < h; y += 4) {\
        for (x = 0; x < w; x += 8) {\
            satd += xavs2_pixel_satd_8x4(pix1 + y * i_pix1 + x, i_pix1,\
                                       pix2 + y * i_pix2 + x, i_pix2);\
        }\
    }\
    return satd;\
}

PIXEL_SATD8_C(64, 64) /* 64x64 */
PIXEL_SATD8_C(64, 32)
PIXEL_SATD8_C(32, 64)
PIXEL_SATD8_C(64, 16)
PIXEL_SATD8_C(64, 48)
PIXEL_SATD8_C(16, 64)
PIXEL_SATD8_C(48, 64)
PIXEL_SATD8_C(32, 32) /* 32x32 */
PIXEL_SATD8_C(32, 16)
PIXEL_SATD8_C(16, 32)
PIXEL_SATD8_C(32,  8)
PIXEL_SATD8_C(32, 24)
PIXEL_SATD8_C( 8, 32)
PIXEL_SATD8_C(24, 32)
PIXEL_SATD8_C(16, 16) /* 16x16 */
PIXEL_SATD8_C(16,  8)
PIXEL_SATD8_C( 8, 16)
PIXEL_SATD8_C(16,  4)
PIXEL_SATD8_C(16, 12)
PIXEL_SATD4_C( 4, 16)
PIXEL_SATD4_C(12, 16)
PIXEL_SATD8_C( 8,  8) /* 8x8 */
PIXEL_SATD4_C( 4,  8)


/**
 * ---------------------------------------------------------------------------
 * SA8D
 * ---------------------------------------------------------------------------
 */

int _sa8d_8x8(const pel_t* pix1, intptr_t i_pix1, const pel_t* pix2, intptr_t i_pix2)
{
    sum2_t tmp[8][4];
    sum2_t a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3;
    sum2_t sum = 0;

    for (int i = 0; i < 8; i++, pix1 += i_pix1, pix2 += i_pix2) {
        a0 = pix1[0] - pix2[0];
        a1 = pix1[1] - pix2[1];
        b0 = (a0 + a1) + ((a0 - a1) << BITS_PER_SUM);
        a2 = pix1[2] - pix2[2];
        a3 = pix1[3] - pix2[3];
        b1 = (a2 + a3) + ((a2 - a3) << BITS_PER_SUM);
        a4 = pix1[4] - pix2[4];
        a5 = pix1[5] - pix2[5];
        b2 = (a4 + a5) + ((a4 - a5) << BITS_PER_SUM);
        a6 = pix1[6] - pix2[6];
        a7 = pix1[7] - pix2[7];
        b3 = (a6 + a7) + ((a6 - a7) << BITS_PER_SUM);
        HADAMARD4(tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], b0, b1, b2, b3);
    }

    for (int i = 0; i < 4; i++) {
        HADAMARD4(a0, a1, a2, a3, tmp[0][i], tmp[1][i], tmp[2][i], tmp[3][i]);
        HADAMARD4(a4, a5, a6, a7, tmp[4][i], tmp[5][i], tmp[6][i], tmp[7][i]);
        b0  = abs2(a0 + a4) + abs2(a0 - a4);
        b0 += abs2(a1 + a5) + abs2(a1 - a5);
        b0 += abs2(a2 + a6) + abs2(a2 - a6);
        b0 += abs2(a3 + a7) + abs2(a3 - a7);
        sum += (sum_t)b0 + (b0 >> BITS_PER_SUM);
    }

    return (cmp_dist_t)sum;
}

/* ---------------------------------------------------------------------------
 */
static
cmp_dist_t xavs2_pixel_sa8d_8x8(const pel_t* pix1, intptr_t i_pix1, const pel_t* pix2, intptr_t i_pix2)
{
    return (cmp_dist_t)((_sa8d_8x8(pix1, i_pix1, pix2, i_pix2) + 2) >> 2);
}

/* ---------------------------------------------------------------------------
 */
static
cmp_dist_t xavs2_pixel_sa8d_16x16(const pel_t* pix1, intptr_t i_pix1, const pel_t* pix2, intptr_t i_pix2)
{
    cmp_dist_t sum = _sa8d_8x8(pix1, i_pix1, pix2, i_pix2)
                     + _sa8d_8x8(pix1 + 8, i_pix1, pix2 + 8, i_pix2)
                     + _sa8d_8x8(pix1 + 8 * i_pix1, i_pix1, pix2 + 8 * i_pix2, i_pix2)
                     + _sa8d_8x8(pix1 + 8 + 8 * i_pix1, i_pix1, pix2 + 8 + 8 * i_pix2, i_pix2);

    // This matches x264 sa8d_16x16, but is slightly different from HM's behavior because
    // this version only rounds once at the end
    return (sum + 2) >> 2;
}

/* ---------------------------------------------------------------------------
 * calculate sa8d in blocks of 8x8
 */
#define PIXEL_SA8D_C8(w, h) \
static cmp_dist_t xavs2_pixel_sa8d_##w##x##h(const pel_t *pix1, intptr_t i_pix1, const pel_t *pix2, intptr_t i_pix2)\
{\
    cmp_dist_t sa8d = 0;\
    int y, x;\
    for (y = 0; y < h; y += 8) {\
        for (x = 0; x < w; x += 8) {\
            sa8d += xavs2_pixel_sa8d_8x8(pix1 + y * i_pix1 + x, i_pix1,\
                                         pix2 + y * i_pix2 + x, i_pix2);\
        }\
    }\
    return sa8d;\
}

/* ---------------------------------------------------------------------------
 * calculate sa8d in blocks of 16x16
 */
#define PIXEL_SA8D_C16(w, h) \
static cmp_dist_t xavs2_pixel_sa8d_##w##x##h(const pel_t *pix1, intptr_t i_pix1, const pel_t *pix2, intptr_t i_pix2)\
{\
    cmp_dist_t sa8d = 0;\
    int y, x;\
    for (y = 0; y < h; y += 16) {\
        for (x = 0; x < w; x += 16) {\
            sa8d += xavs2_pixel_sa8d_16x16(pix1 + y * i_pix1 + x, i_pix1,\
                                           pix2 + y * i_pix2 + x, i_pix2);\
        }\
    }\
    return sa8d;\
}

#define xavs2_pixel_sa8d_4x4    xavs2_pixel_satd_4x4
#define xavs2_pixel_sa8d_4x8    xavs2_pixel_satd_4x8
#define xavs2_pixel_sa8d_8x4    xavs2_pixel_satd_8x4
#define xavs2_pixel_sa8d_16x4   xavs2_pixel_satd_16x4
#define xavs2_pixel_sa8d_4x16   xavs2_pixel_satd_4x16
#define xavs2_pixel_sa8d_12x16  xavs2_pixel_satd_12x16
#define xavs2_pixel_sa8d_16x12  xavs2_pixel_satd_16x12
PIXEL_SA8D_C8(8, 16)
PIXEL_SA8D_C8(8, 32)
PIXEL_SA8D_C8(16, 8)
PIXEL_SA8D_C8(32, 8)
PIXEL_SA8D_C16(32, 16)
PIXEL_SA8D_C8(32, 24)
PIXEL_SA8D_C8(24, 32)
PIXEL_SA8D_C16(32, 32)
PIXEL_SA8D_C16(16, 32)
PIXEL_SA8D_C16(64, 16)
PIXEL_SA8D_C16(64, 32)
PIXEL_SA8D_C16(64, 48)
PIXEL_SA8D_C16(16, 64)
PIXEL_SA8D_C16(32, 64)
PIXEL_SA8D_C16(48, 64)
PIXEL_SA8D_C16(64, 64)

/**
 * ---------------------------------------------------------------------------
 * SSD
 * ---------------------------------------------------------------------------
 */
dist_t xavs2_get_block_ssd_c(const pel_t *pix1, intptr_t i_pix1, const pel_t *pix2, intptr_t i_pix2, int width, int height)
{
    dist_t sum = 0;
    int x, y, tmp;
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            tmp = pix1[x] - pix2[x];
            sum += (tmp * tmp);
        }
        pix1 += i_pix1;
        pix2 += i_pix2;
    }
    return sum;
}

#define PIXEL_SSD_C(w, h) \
static dist_t xavs2_pixel_ssd_##w##x##h(const pel_t *pix1, intptr_t i_pix1, const pel_t *pix2, intptr_t i_pix2)\
{\
    dist_t sum = 0;\
    int x, y, tmp;\
    for (y = 0; y < h; y++) {\
        for (x = 0; x < w; x++) {\
            tmp  = pix1[x] - pix2[x];\
            sum += (tmp * tmp);\
        }\
        pix1 += i_pix1;\
        pix2 += i_pix2;\
    }\
    return sum;\
}

PIXEL_SSD_C(64, 64)  /* 64x64 */
PIXEL_SSD_C(64, 32)
PIXEL_SSD_C(32, 64)
PIXEL_SSD_C(64, 16)
PIXEL_SSD_C(64, 48)
PIXEL_SSD_C(16, 64)
PIXEL_SSD_C(48, 64)
PIXEL_SSD_C(32, 32)  /* 32x32 */
PIXEL_SSD_C(32, 16)
PIXEL_SSD_C(16, 32)
PIXEL_SSD_C(32,  8)
PIXEL_SSD_C(32, 24)
PIXEL_SSD_C( 8, 32)
PIXEL_SSD_C(24, 32)
PIXEL_SSD_C(16, 16)  /* 16x16 */
PIXEL_SSD_C(16,  8)
PIXEL_SSD_C( 8, 16)
PIXEL_SSD_C(16,  4)
PIXEL_SSD_C(16, 12)
PIXEL_SSD_C( 4, 16)
PIXEL_SSD_C(12, 16)
PIXEL_SSD_C( 8,  8)  /* 8x8 */
PIXEL_SSD_C( 8,  4)
PIXEL_SSD_C( 4,  8)
PIXEL_SSD_C( 4,  4)  /* 4x4 */

/* ---------------------------------------------------------------------------
 * ssd for one plane of frame
 */
#if XAVS2_STAT
uint64_t xavs2_pixel_ssd_wxh(pixel_funcs_t *pf,
                             pel_t *p_pix1, intptr_t i_pix1,
                             pel_t *p_pix2, intptr_t i_pix2,
                             int i_width, int i_height,
                             int inout_shift)
{
    uint64_t i_ssd = 0;
    int align = !(((intptr_t)p_pix1 | (intptr_t)p_pix2 | i_pix1 | i_pix2) & 15);
    int x, y;
    pixel_ssd_t cal_ssd[2];

    if (inout_shift > 0) {
        int inout_offset = 1 << (inout_shift - 1);

        for (y = 0; y < i_height; y++) {
            for (x = 0; x < i_width; x++) {
                int d = ((p_pix1[x] + inout_offset) >> inout_shift) - ((p_pix2[x] + inout_offset) >> inout_shift);
                i_ssd += d * d;
            }
            p_pix1 += i_pix1;
            p_pix2 += i_pix2;
        }
    } else {
        cal_ssd[0] = pf->ssd[LUMA_8x8];  /*  8 x  8 */
        cal_ssd[1] = pf->ssd[LUMA_16x16];  /* 16 x 16 */

#define SSD(id) i_ssd += cal_ssd[id](p_pix1 + y*i_pix1 + x, i_pix1, p_pix2 + y*i_pix2 + x, i_pix2)

        for (y = 0; y < i_height - 15;) {
            if (align) {
                for (x = 0; x < i_width - 15; x += 16) {
                    SSD(1);         /* 16x16 */
                }
                y += 16;
            } else {
                for (x = 0; x < i_width - 7; x += 8) {
                    SSD(0);         /* 8x8 */
                }
                y += 8;
                for (x = 0; x < i_width - 7; x += 8) {
                    SSD(0);         /* 8x8 */
                }
                y += 8;
            }
        }
        if (y < i_height - 7) {
            for (x = 0; x < i_width - 7; x += 8) {
                SSD(0);             /* 8x8 */
            }
        }
#undef SSD

        /* sum the rest ssd */
#define SSD1    { int d = p_pix1[y*i_pix1+x] - p_pix2[y*i_pix2+x]; i_ssd += d*d; }
        if (i_width & 7) {
            for (y = 0; y < (i_height & ~7); y++) {
                for (x = i_width & ~7; x < i_width; x++) {
                    SSD1;
                }
            }
        }
        if (i_height & 7) {
            for (y = i_height & ~7; y < i_height; y++) {
                for (x = 0; x < i_width; x++) {
                    SSD1;
                }
            }
        }
#undef SSD1
    }

    return i_ssd;
}
#endif


/**
 * ---------------------------------------------------------------------------
 * AVG
 * ---------------------------------------------------------------------------
 */

#define PIXEL_AVG_C(w, h) \
static void xavs2_pixel_avg_##w##x##h(pel_t* dst, intptr_t dstride, const pel_t* src0, intptr_t sstride0, const pel_t* src1, intptr_t sstride1, int weight)\
{\
    int x, y;\
    UNUSED_PARAMETER(weight); \
    for (y = 0; y < h; y++) {\
        for (x = 0; x < w; x++) {\
            dst[x] = (src0[x] + src1[x] + 1) >> 1;\
        }\
        dst  += dstride;\
        src0 += sstride0;\
        src1 += sstride1;\
    }\
}

PIXEL_AVG_C(64, 64) /* 64x64 */
PIXEL_AVG_C(64, 32)
PIXEL_AVG_C(32, 64)
PIXEL_AVG_C(64, 16)
PIXEL_AVG_C(64, 48)
PIXEL_AVG_C(16, 64)
PIXEL_AVG_C(48, 64)
PIXEL_AVG_C(32, 32) /* 32x32 */
PIXEL_AVG_C(32, 16)
PIXEL_AVG_C(16, 32)
PIXEL_AVG_C(32,  8)
PIXEL_AVG_C(32, 24)
PIXEL_AVG_C( 8, 32)
PIXEL_AVG_C(24, 32)
PIXEL_AVG_C(16, 16) /* 16x16 */
PIXEL_AVG_C(16,  8)
PIXEL_AVG_C( 8, 16)
PIXEL_AVG_C(16,  4)
PIXEL_AVG_C(16, 12)
PIXEL_AVG_C( 4, 16)
PIXEL_AVG_C(12, 16)
PIXEL_AVG_C( 8,  8) /* 8x8 */
PIXEL_AVG_C( 8,  4)
PIXEL_AVG_C( 4,  8)
PIXEL_AVG_C( 4,  4) /* 4x4 */


/**
 * ---------------------------------------------------------------------------
 * block operation: copy/add/sub (p: pixel, s: short)
 * ---------------------------------------------------------------------------
 */
#define BLOCKCOPY_PP_C(w, h) \
static void xavs2_blockcopy_pp_##w##x##h(pel_t *a, intptr_t stridea, const pel_t *b, intptr_t strideb)\
{\
    int x, y;\
    for (y = 0; y < h; y++) {\
        for (x = 0; x < w; x++) {\
            a[x] = b[x];\
        }\
        a += stridea;\
        b += strideb;\
    }\
}

#define BLOCKCOPY_SS_C(w, h) \
static void xavs2_blockcopy_ss_##w##x##h(coeff_t* a, intptr_t stridea, const coeff_t* b, intptr_t strideb)\
{\
    int x, y;\
    for (y = 0; y < h; y++) {\
        for (x = 0; x < w; x++) {\
            a[x] = b[x];\
        }\
        a += stridea;\
        b += strideb;\
    }\
}

#define BLOCKCOPY_SP_C(w, h) \
static void xavs2_blockcopy_sp_##w##x##h(pel_t *a, intptr_t stridea, const coeff_t* b, intptr_t strideb)\
{\
    int x, y;\
    for (y = 0; y < h; y++) {\
        for (x = 0; x < w; x++) {\
            assert((b[x] >= 0) && (b[x] <= ((1 << 8) - 1)));\
            a[x] = (pel_t)b[x];\
        }\
        a += stridea;\
        b += strideb;\
    }\
}

#define BLOCKCOPY_PS_C(w, h) \
static void xavs2_blockcopy_ps_##w##x##h(coeff_t *a, intptr_t stridea, const pel_t *b, intptr_t strideb)\
{\
    int x, y;\
    for (y = 0; y < h; y++) {\
        for (x = 0; x < w; x++) {\
            a[x] = (int16_t)b[x];\
        }\
        a += stridea;\
        b += strideb;\
    }\
}\
 
#define PIXEL_SUB_PS_C(w, h) \
static void xavs2_pixel_sub_ps_##w##x##h(coeff_t *a, intptr_t dstride, const pel_t *b0, const pel_t *b1, intptr_t sstride0, intptr_t sstride1)\
{\
    int x, y;\
    for (y = 0; y < h; y++) {\
        for (x = 0; x < w; x++) {\
            a[x] = (int16_t)(b0[x] - b1[x]);\
        }\
        b0 += sstride0;\
        b1 += sstride1;\
        a  += dstride;\
    }\
}

#define PIXEL_ADD_PS_C(w, h) \
static void xavs2_pixel_add_ps_##w##x##h(pel_t *a, intptr_t dstride, const pel_t *b0, const coeff_t* b1, intptr_t sstride0, intptr_t sstride1)\
{\
    int x, y;\
    for (y = 0; y < h; y++) {\
        for (x = 0; x < w; x++) {\
            a[x] = (pel_t)XAVS2_CLIP1(b0[x] + b1[x]);\
        }\
        b0 += sstride0;\
        b1 += sstride1;\
        a  += dstride;\
    }\
}

#define BLOCK_OP_C(w, h) \
    BLOCKCOPY_PP_C(w, h);\
    BLOCKCOPY_SS_C(w, h);\
    BLOCKCOPY_SP_C(w, h);\
    BLOCKCOPY_PS_C(w, h);\
    PIXEL_SUB_PS_C(w, h);\
    PIXEL_ADD_PS_C(w, h);

BLOCK_OP_C(64, 64)  /* 64x64 */
BLOCK_OP_C(64, 32)
BLOCK_OP_C(32, 64)
BLOCK_OP_C(64, 16)
BLOCK_OP_C(64, 48)
BLOCK_OP_C(16, 64)
BLOCK_OP_C(48, 64)
BLOCK_OP_C(32, 32)  /* 32x32 */
BLOCK_OP_C(32, 16)
BLOCK_OP_C(16, 32)
BLOCK_OP_C(32,  8)
BLOCK_OP_C(32, 24)
BLOCK_OP_C( 8, 32)
BLOCK_OP_C(24, 32)
BLOCK_OP_C(16, 16)  /* 16x16 */
BLOCK_OP_C(16,  8)
BLOCK_OP_C( 8, 16)
BLOCK_OP_C(16,  4)
BLOCK_OP_C(16, 12)
BLOCK_OP_C( 4, 16)
BLOCK_OP_C(12, 16)
BLOCK_OP_C( 8,  8)  /* 8x8 */
BLOCK_OP_C( 8,  4)
BLOCK_OP_C( 4,  8)
BLOCK_OP_C( 4,  4)  /* 4x4 */

/* ---------------------------------------------------------------------------
 */
static void xavs2_pixel_average(pel_t *dst, int i_dst, pel_t *src1, int i_src1, pel_t *src2, int i_src2, int width, int height)
{
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            dst[j] = (pel_t)((src1[j] + src2[j] + 1) >> 1);
        }
        dst  += i_dst;
        src1 += i_src1;
        src2 += i_src2;
    }
}

/* ---------------------------------------------------------------------------
 * init functions of block operation : copy / add / sub
 */
static void init_block_opreation_funcs(uint32_t cpuid, pixel_funcs_t* pixf)
{
#define ALL_LUMA_CU(name1, name2, cpu) \
    pixf->name1[LUMA_64x64] = xavs2_ ## name2 ## _64x64 ## cpu;\
    pixf->name1[LUMA_32x32] = xavs2_ ## name2 ## _32x32 ## cpu;\
    pixf->name1[LUMA_16x16] = xavs2_ ## name2 ## _16x16 ## cpu;\
    pixf->name1[LUMA_8x8  ] = xavs2_ ## name2 ## _8x8   ## cpu;\
    pixf->name1[LUMA_4x4  ] = xavs2_ ## name2 ## _4x4   ## cpu

#define ALL_LUMA_PU(name1, name2, cpu) \
    pixf->name1[LUMA_64x64] = xavs2_ ## name2 ## _64x64 ## cpu;  /* 64x64 */ \
    pixf->name1[LUMA_64x32] = xavs2_ ## name2 ## _64x32 ## cpu;\
    pixf->name1[LUMA_32x64] = xavs2_ ## name2 ## _32x64 ## cpu;\
    pixf->name1[LUMA_64x16] = xavs2_ ## name2 ## _64x16 ## cpu;\
    pixf->name1[LUMA_64x48] = xavs2_ ## name2 ## _64x48 ## cpu;\
    pixf->name1[LUMA_16x64] = xavs2_ ## name2 ## _16x64 ## cpu;\
    pixf->name1[LUMA_48x64] = xavs2_ ## name2 ## _48x64 ## cpu;\
    pixf->name1[LUMA_32x32] = xavs2_ ## name2 ## _32x32 ## cpu;  /* 32x32 */ \
    pixf->name1[LUMA_32x16] = xavs2_ ## name2 ## _32x16 ## cpu;\
    pixf->name1[LUMA_16x32] = xavs2_ ## name2 ## _16x32 ## cpu;\
    pixf->name1[LUMA_32x8 ] = xavs2_ ## name2 ## _32x8  ## cpu;\
    pixf->name1[LUMA_32x24] = xavs2_ ## name2 ## _32x24 ## cpu;\
    pixf->name1[LUMA_8x32 ] = xavs2_ ## name2 ## _8x32  ## cpu;\
    pixf->name1[LUMA_24x32] = xavs2_ ## name2 ## _24x32 ## cpu;\
    pixf->name1[LUMA_16x16] = xavs2_ ## name2 ## _16x16 ## cpu;  /* 16x16 */ \
    pixf->name1[LUMA_16x8 ] = xavs2_ ## name2 ## _16x8  ## cpu;\
    pixf->name1[LUMA_8x16 ] = xavs2_ ## name2 ## _8x16  ## cpu;\
    pixf->name1[LUMA_16x4 ] = xavs2_ ## name2 ## _16x4  ## cpu;\
    pixf->name1[LUMA_16x12] = xavs2_ ## name2 ## _16x12 ## cpu;\
    pixf->name1[LUMA_4x16 ] = xavs2_ ## name2 ## _4x16  ## cpu;\
    pixf->name1[LUMA_12x16] = xavs2_ ## name2 ## _12x16 ## cpu;\
    pixf->name1[LUMA_8x8  ] = xavs2_ ## name2 ## _8x8   ## cpu;  /* 8x8 */ \
    pixf->name1[LUMA_8x4  ] = xavs2_ ## name2 ## _8x4   ## cpu;\
    pixf->name1[LUMA_4x8  ] = xavs2_ ## name2 ## _4x8   ## cpu;\
    pixf->name1[LUMA_4x4  ] = xavs2_ ## name2 ## _4x4   ## cpu  /* 4x4 */

    /* -------------------------------------------------------------
     * init all c functions
     */
    //ALL_LUMA_CU(add_ps,  pixel_add_ps, );
    ALL_LUMA_PU(add_ps,  pixel_add_ps, );
//    ALL_LUMA_CU(sub_ps,  pixel_sub_ps, );
    ALL_LUMA_PU(sub_ps,  pixel_sub_ps, );
    ALL_LUMA_PU(copy_sp, blockcopy_sp, );
    ALL_LUMA_PU(copy_ps, blockcopy_ps, );
    ALL_LUMA_PU(copy_ss, blockcopy_ss, );
    ALL_LUMA_PU(copy_pp, blockcopy_pp, );
    pixf->ssd_block = xavs2_get_block_ssd_c;

    /* -------------------------------------------------------------
     * init all SIMD functions
     */
#if HAVE_MMX
    if (cpuid & XAVS2_CPU_SSE2) {
        ALL_LUMA_PU(copy_sp, blockcopy_sp, _sse2);
        ALL_LUMA_PU(copy_ss, blockcopy_ss, _sse2);
        ALL_LUMA_PU(copy_pp, blockcopy_pp, _sse2);
    }

    if (cpuid & XAVS2_CPU_SSE4) {
        pixf->add_ps [LUMA_4x4  ] = xavs2_pixel_add_ps_4x4_sse4;
        pixf->add_ps [LUMA_4x8  ] = xavs2_pixel_add_ps_4x8_sse4;
        pixf->add_ps [LUMA_4x16 ] = xavs2_pixel_add_ps_4x16_sse4;
        pixf->add_ps [LUMA_8x8  ] = xavs2_pixel_add_ps_8x8_sse4;
        pixf->add_ps [LUMA_8x16 ] = xavs2_pixel_add_ps_8x16_sse4;
        pixf->add_ps [LUMA_8x32 ] = xavs2_pixel_add_ps_8x32_sse4;
        pixf->add_ps [LUMA_16x4 ] = xavs2_pixel_add_ps_16x4_sse4;
        pixf->add_ps [LUMA_16x8 ] = xavs2_pixel_add_ps_16x8_sse4;
        pixf->add_ps [LUMA_16x12] = xavs2_pixel_add_ps_16x12_sse4;
        pixf->add_ps [LUMA_16x16] = xavs2_pixel_add_ps_16x16_sse4;
        pixf->add_ps [LUMA_16x64] = xavs2_pixel_add_ps_16x64_sse4;
        pixf->add_ps [LUMA_32x8 ] = xavs2_pixel_add_ps_32x8_sse4;
        //  pixf->add_ps [LUMA_32x16] = xavs2_pixel_add_ps_32x16_sse4;
        //  pixf->add_ps [LUMA_32x24] = xavs2_pixel_add_ps_32x24_sse4;
        pixf->add_ps [LUMA_32x32] = xavs2_pixel_add_ps_32x32_sse4;
        pixf->add_ps [LUMA_32x64] = xavs2_pixel_add_ps_32x64_sse4;
        pixf->add_ps [LUMA_64x16] = xavs2_pixel_add_ps_64x16_sse4;
        //  pixf->add_ps [LUMA_64x32] = xavs2_pixel_add_ps_64x32_sse4;
        //  pixf->add_ps [LUMA_64x48] = xavs2_pixel_add_ps_64x48_sse4;
        pixf->add_ps [LUMA_64x64] = xavs2_pixel_add_ps_64x64_sse4;

        pixf->sub_ps [LUMA_4x4  ] = xavs2_pixel_sub_ps_4x4_sse4;
        pixf->sub_ps [LUMA_4x8  ] = xavs2_pixel_sub_ps_4x8_sse4;
        pixf->sub_ps [LUMA_4x16 ] = xavs2_pixel_sub_ps_4x16_sse4;
        pixf->sub_ps [LUMA_8x8  ] = xavs2_pixel_sub_ps_8x8_sse4;
        pixf->sub_ps [LUMA_8x16 ] = xavs2_pixel_sub_ps_8x16_sse4;
        pixf->sub_ps [LUMA_8x32 ] = xavs2_pixel_sub_ps_8x32_sse4;
        pixf->sub_ps [LUMA_16x4 ] = xavs2_pixel_sub_ps_16x4_sse4;
        //  pixf->sub_ps [LUMA_16x8 ] = xavs2_pixel_sub_ps_16x8_sse4;
        //  pixf->sub_ps [LUMA_16x12] = xavs2_pixel_sub_ps_16x12_sse4;
        pixf->sub_ps [LUMA_16x16] = xavs2_pixel_sub_ps_16x16_sse4;
        pixf->sub_ps [LUMA_16x64] = xavs2_pixel_sub_ps_16x64_sse4;
        pixf->sub_ps [LUMA_32x8 ] = xavs2_pixel_sub_ps_32x8_sse4;
        //  pixf->sub_ps [LUMA_32x16] = xavs2_pixel_sub_ps_32x16_sse4;
        //  pixf->sub_ps [LUMA_32x24] = xavs2_pixel_sub_ps_32x24_sse4;
        pixf->sub_ps [LUMA_32x32] = xavs2_pixel_sub_ps_32x32_sse4;
        pixf->sub_ps [LUMA_32x64] = xavs2_pixel_sub_ps_32x64_sse4;
        pixf->sub_ps [LUMA_64x16] = xavs2_pixel_sub_ps_64x16_sse4;
        //  pixf->sub_ps [LUMA_64x32] = xavs2_pixel_sub_ps_64x32_sse4;
        //  pixf->sub_ps [LUMA_64x48] = xavs2_pixel_sub_ps_64x48_sse4;
        pixf->sub_ps [LUMA_64x64] = xavs2_pixel_sub_ps_64x64_sse4;

        ALL_LUMA_PU(copy_ps, blockcopy_ps, _sse4);
    }

    if (cpuid & XAVS2_CPU_AVX) {
        pixf->copy_pp[LUMA_64x64] = xavs2_blockcopy_pp_64x64_avx;
        pixf->copy_pp[LUMA_64x32] = xavs2_blockcopy_pp_64x32_avx;
        pixf->copy_pp[LUMA_32x64] = xavs2_blockcopy_pp_32x64_avx;
        pixf->copy_pp[LUMA_64x16] = xavs2_blockcopy_pp_64x16_avx;
        pixf->copy_pp[LUMA_64x48] = xavs2_blockcopy_pp_64x48_avx;
        pixf->copy_pp[LUMA_48x64] = xavs2_blockcopy_pp_48x64_avx;
        pixf->copy_pp[LUMA_32x32] = xavs2_blockcopy_pp_32x32_avx;
        pixf->copy_pp[LUMA_32x16] = xavs2_blockcopy_pp_32x16_avx;
        pixf->copy_pp[LUMA_32x8 ] = xavs2_blockcopy_pp_32x8_avx;
        pixf->copy_pp[LUMA_32x24] = xavs2_blockcopy_pp_32x24_avx;

        pixf->copy_ss[LUMA_64x64] = xavs2_blockcopy_ss_64x64_avx;
        pixf->copy_ss[LUMA_64x32] = xavs2_blockcopy_ss_64x32_avx;
        pixf->copy_ss[LUMA_32x64] = xavs2_blockcopy_ss_32x64_avx;
        pixf->copy_ss[LUMA_64x16] = xavs2_blockcopy_ss_64x16_avx;
        pixf->copy_ss[LUMA_64x48] = xavs2_blockcopy_ss_64x48_avx;
        pixf->copy_ss[LUMA_16x64] = xavs2_blockcopy_ss_16x64_avx;
        pixf->copy_ss[LUMA_48x64] = xavs2_blockcopy_ss_48x64_avx;
        pixf->copy_ss[LUMA_32x32] = xavs2_blockcopy_ss_32x32_avx;
        pixf->copy_ss[LUMA_32x16] = xavs2_blockcopy_ss_32x16_avx;
        pixf->copy_ss[LUMA_16x32] = xavs2_blockcopy_ss_16x32_avx;
        pixf->copy_ss[LUMA_32x8 ] = xavs2_blockcopy_ss_32x8_avx;
        pixf->copy_ss[LUMA_32x24] = xavs2_blockcopy_ss_32x24_avx;
        pixf->copy_ss[LUMA_24x32] = xavs2_blockcopy_ss_24x32_avx;
        pixf->copy_ss[LUMA_16x16] = xavs2_blockcopy_ss_16x16_avx;
        pixf->copy_ss[LUMA_16x8 ] = xavs2_blockcopy_ss_16x8_avx;
        pixf->copy_ss[LUMA_16x4 ] = xavs2_blockcopy_ss_16x4_avx;
        pixf->copy_ss[LUMA_16x12] = xavs2_blockcopy_ss_16x12_avx;
    }

    if (cpuid & XAVS2_CPU_AVX2) {
        pixf->add_ps [LUMA_16x4 ] = xavs2_pixel_add_ps_16x4_avx2;
        pixf->add_ps [LUMA_16x8 ] = xavs2_pixel_add_ps_16x8_avx2;
        pixf->add_ps [LUMA_16x16] = xavs2_pixel_add_ps_16x16_avx2;
        pixf->add_ps [LUMA_16x32] = xavs2_pixel_add_ps_16x32_avx2;
        pixf->add_ps [LUMA_16x64] = xavs2_pixel_add_ps_16x64_avx2;
#if ARCH_X86_64
        pixf->add_ps [LUMA_32x8 ] = xavs2_pixel_add_ps_32x8_avx2;
        pixf->add_ps [LUMA_32x16] = xavs2_pixel_add_ps_32x16_avx2;
        pixf->add_ps [LUMA_32x24] = xavs2_pixel_add_ps_32x24_avx2;
        pixf->add_ps [LUMA_32x32] = xavs2_pixel_add_ps_32x32_avx2;
        pixf->add_ps [LUMA_32x64] = xavs2_pixel_add_ps_32x64_avx2;
#endif
        pixf->add_ps [LUMA_64x16] = xavs2_pixel_add_ps_64x16_avx2;
        pixf->add_ps [LUMA_64x32] = xavs2_pixel_add_ps_64x32_avx2;
        pixf->add_ps [LUMA_64x48] = xavs2_pixel_add_ps_64x48_avx2;
        pixf->add_ps [LUMA_64x64] = xavs2_pixel_add_ps_64x64_avx2;

#if ARCH_X86_64
        pixf->sub_ps [LUMA_16x16] = xavs2_pixel_sub_ps_16x16_avx2;
        pixf->sub_ps [LUMA_16x32] = xavs2_pixel_sub_ps_16x32_avx2;
        pixf->sub_ps [LUMA_16x64] = xavs2_pixel_sub_ps_16x64_avx2;
        pixf->sub_ps [LUMA_32x8 ] = xavs2_pixel_sub_ps_32x8_avx2;
        pixf->sub_ps [LUMA_32x16] = xavs2_pixel_sub_ps_32x16_avx2;
        pixf->sub_ps [LUMA_32x32] = xavs2_pixel_sub_ps_32x32_avx2;
        pixf->sub_ps [LUMA_32x64] = xavs2_pixel_sub_ps_32x64_avx2;
#endif
        pixf->sub_ps [LUMA_64x16] = xavs2_pixel_sub_ps_64x16_avx2;
        pixf->sub_ps [LUMA_64x64] = xavs2_pixel_sub_ps_64x64_avx2;

        pixf->copy_sp[LUMA_64x64] = xavs2_blockcopy_sp_64x64_avx2;
        pixf->copy_sp[LUMA_32x64] = xavs2_blockcopy_sp_32x64_avx2;
        pixf->copy_sp[LUMA_32x32] = xavs2_blockcopy_sp_32x32_avx2;
        pixf->copy_sp[LUMA_16x32] = xavs2_blockcopy_sp_16x32_avx2;
        pixf->copy_sp[LUMA_16x16] = xavs2_blockcopy_sp_16x16_avx2;

        pixf->copy_ps[LUMA_64x64] = xavs2_blockcopy_ps_64x64_avx2;
        pixf->copy_ps[LUMA_32x64] = xavs2_blockcopy_ps_32x64_avx2;
        pixf->copy_ps[LUMA_32x32] = xavs2_blockcopy_ps_32x32_avx2;
        pixf->copy_ps[LUMA_16x32] = xavs2_blockcopy_ps_16x32_avx2;
        pixf->copy_ps[LUMA_16x16] = xavs2_blockcopy_ps_16x16_avx2;
    }
#endif // if HAVE_MMX

#undef ALL_LUMA_CU
#undef ALL_LUMA_PU
}

/**
 * ---------------------------------------------------------------------------
 * pixel init
 * ---------------------------------------------------------------------------
 */
void xavs2_pixel_init(uint32_t cpuid, pixel_funcs_t* pixf)
{
    /* -------------------------------------------------------------
     */
#define INIT_PIXEL_FUNC(name, cpu) \
    /* 64x64 */                                                  \
    pixf->name[LUMA_64x64] = xavs2_pixel_ ## name ## _64x64 ## cpu;\
    pixf->name[LUMA_64x32] = xavs2_pixel_ ## name ## _64x32 ## cpu;\
    pixf->name[LUMA_32x64] = xavs2_pixel_ ## name ## _32x64 ## cpu;\
    pixf->name[LUMA_64x16] = xavs2_pixel_ ## name ## _64x16 ## cpu;\
    pixf->name[LUMA_64x48] = xavs2_pixel_ ## name ## _64x48 ## cpu;\
    pixf->name[LUMA_16x64] = xavs2_pixel_ ## name ## _16x64 ## cpu;\
    pixf->name[LUMA_48x64] = xavs2_pixel_ ## name ## _48x64 ## cpu;\
    /* 32x32 */                                                  \
    pixf->name[LUMA_32x32] = xavs2_pixel_ ## name ## _32x32 ## cpu;\
    pixf->name[LUMA_32x16] = xavs2_pixel_ ## name ## _32x16 ## cpu;\
    pixf->name[LUMA_16x32] = xavs2_pixel_ ## name ## _16x32 ## cpu;\
    pixf->name[LUMA_32x8 ] = xavs2_pixel_ ## name ## _32x8  ## cpu;\
    pixf->name[LUMA_32x24] = xavs2_pixel_ ## name ## _32x24 ## cpu;\
    pixf->name[LUMA_8x32 ] = xavs2_pixel_ ## name ## _8x32  ## cpu;\
    pixf->name[LUMA_24x32] = xavs2_pixel_ ## name ## _24x32 ## cpu;\
    /* 16x16 */                                                  \
    pixf->name[LUMA_16x16] = xavs2_pixel_ ## name ## _16x16 ## cpu;\
    pixf->name[LUMA_16x8 ] = xavs2_pixel_ ## name ## _16x8  ## cpu;\
    pixf->name[LUMA_8x16 ] = xavs2_pixel_ ## name ## _8x16  ## cpu;\
    pixf->name[LUMA_16x4 ] = xavs2_pixel_ ## name ## _16x4  ## cpu;\
    pixf->name[LUMA_16x12] = xavs2_pixel_ ## name ## _16x12 ## cpu;\
    pixf->name[LUMA_4x16 ] = xavs2_pixel_ ## name ## _4x16  ## cpu;\
    pixf->name[LUMA_12x16] = xavs2_pixel_ ## name ## _12x16 ## cpu;\
    /* 8x8 */                                                    \
    pixf->name[LUMA_8x8  ] = xavs2_pixel_ ## name ## _8x8   ## cpu;\
    pixf->name[LUMA_8x4  ] = xavs2_pixel_ ## name ## _8x4   ## cpu;\
    pixf->name[LUMA_4x8  ] = xavs2_pixel_ ## name ## _4x8   ## cpu;\
    /* 4x4 */                                                    \
    pixf->name[LUMA_4x4  ] = xavs2_pixel_ ## name ## _4x4   ## cpu;


    /* -------------------------------------------------------------
     */
#define INIT_SATD(cpu) \
    pixf->satd[LUMA_64x64] = xavs2_pixel_satd_64x64_ ## cpu;  /* 64x64 */ \
    pixf->satd[LUMA_64x32] = xavs2_pixel_satd_64x32_ ## cpu;\
    pixf->satd[LUMA_32x64] = xavs2_pixel_satd_32x64_ ## cpu;\
    pixf->satd[LUMA_64x16] = xavs2_pixel_satd_64x16_ ## cpu;\
    pixf->satd[LUMA_64x48] = xavs2_pixel_satd_64x48_ ## cpu;\
    pixf->satd[LUMA_16x64] = xavs2_pixel_satd_16x64_ ## cpu;\
    pixf->satd[LUMA_48x64] = xavs2_pixel_satd_48x64_ ## cpu;\
    pixf->satd[LUMA_32x32] = xavs2_pixel_satd_32x32_ ## cpu;  /* 32x32 */ \
    pixf->satd[LUMA_32x16] = xavs2_pixel_satd_32x16_ ## cpu;\
    pixf->satd[LUMA_16x32] = xavs2_pixel_satd_16x32_ ## cpu;\
    pixf->satd[LUMA_32x8 ] = xavs2_pixel_satd_32x8_  ## cpu;\
    pixf->satd[LUMA_32x24] = xavs2_pixel_satd_32x24_ ## cpu;\
    pixf->satd[LUMA_8x32 ] = xavs2_pixel_satd_8x32_  ## cpu;\
    pixf->satd[LUMA_24x32] = xavs2_pixel_satd_24x32_ ## cpu;\
    pixf->satd[LUMA_16x16] = xavs2_pixel_satd_16x16_ ## cpu;  /* 16x16 */ \
    pixf->satd[LUMA_16x8 ] = xavs2_pixel_satd_16x8_  ## cpu;\
    pixf->satd[LUMA_8x16 ] = xavs2_pixel_satd_8x16_  ## cpu;\
    pixf->satd[LUMA_16x4 ] = xavs2_pixel_satd_16x4_  ## cpu;\
    pixf->satd[LUMA_16x12] = xavs2_pixel_satd_16x12_ ## cpu;\
    pixf->satd[LUMA_4x16 ] = xavs2_pixel_satd_4x16_  ## cpu;\
    pixf->satd[LUMA_12x16] = xavs2_pixel_satd_12x16_ ## cpu;\
    pixf->satd[LUMA_8x8  ] = xavs2_pixel_satd_8x8_   ## cpu;  /* 8x8 */   \
    pixf->satd[LUMA_8x4  ] = xavs2_pixel_satd_8x4_   ## cpu;\
    pixf->satd[LUMA_4x8  ] = xavs2_pixel_satd_4x8_   ## cpu;

    /* -------------------------------------------------------------
     */
#define INIT_SSD(cpu) \
    pixf->ssd[LUMA_32x64] = xavs2_pixel_ssd_32x64_ ## cpu;\
    pixf->ssd[LUMA_16x64] = xavs2_pixel_ssd_16x64_ ## cpu;\
    pixf->ssd[LUMA_32x32] = xavs2_pixel_ssd_32x32_ ## cpu;\
    pixf->ssd[LUMA_32x16] = xavs2_pixel_ssd_32x16_ ## cpu;\
    pixf->ssd[LUMA_16x32] = xavs2_pixel_ssd_16x32_ ## cpu;\
    pixf->ssd[LUMA_32x24] = xavs2_pixel_ssd_32x24_ ## cpu;\
    pixf->ssd[LUMA_32x8 ] = xavs2_pixel_ssd_32x8_  ## cpu;\
    pixf->ssd[LUMA_8x32 ] = xavs2_pixel_ssd_8x32_  ## cpu;\
    pixf->ssd[LUMA_16x16] = xavs2_pixel_ssd_16x16_ ## cpu;\
    pixf->ssd[LUMA_16x8 ] = xavs2_pixel_ssd_16x8_  ## cpu;\
    pixf->ssd[LUMA_8x16 ] = xavs2_pixel_ssd_8x16_  ## cpu;\
    pixf->ssd[LUMA_16x12] = xavs2_pixel_ssd_16x12_ ## cpu;\
    pixf->ssd[LUMA_16x4 ] = xavs2_pixel_ssd_16x4_  ## cpu;\
/* SIMD_ERROR pixf->ssd[LUMA_8x8  ] = xavs2_pixel_ssd_8x8_   ## cpu;*/\
    pixf->ssd[LUMA_8x4  ] = xavs2_pixel_ssd_8x4_   ## cpu


    /* clear */
    memset(pixf, 0, sizeof(pixel_funcs_t));

    /* -------------------------------------------------------------
     * init all c functions
     */
    INIT_PIXEL_FUNC(sad,    );        // sad
    INIT_PIXEL_FUNC(sad_x3, );        // sad_x3
    INIT_PIXEL_FUNC(sad_x4, );        // sad_x4
    INIT_PIXEL_FUNC(satd,   );        // satd
    INIT_PIXEL_FUNC(ssd,    );        // ssd
    INIT_PIXEL_FUNC(avg,    );        // avg
    INIT_PIXEL_FUNC(sa8d,   );        // sa8d

    pixf->average = xavs2_pixel_average;// block average

    /* -------------------------------------------------------------
     * init SIMD functions
     */
#if HAVE_MMX
    if (cpuid & XAVS2_CPU_MMX2) {
        pixf->sad   [LUMA_16x16] = xavs2_pixel_sad_16x16_mmx2;
        pixf->sad   [LUMA_16x8 ] = xavs2_pixel_sad_16x8_mmx2;
        pixf->sad   [LUMA_8x16 ] = xavs2_pixel_sad_8x16_mmx2;
        pixf->sad   [LUMA_16x4 ] = xavs2_pixel_sad_16x4_mmx2;
        pixf->sad   [LUMA_4x16 ] = xavs2_pixel_sad_4x16_mmx2;
        pixf->sad   [LUMA_8x8  ] = xavs2_pixel_sad_8x8_mmx2;
        pixf->sad   [LUMA_8x4  ] = xavs2_pixel_sad_8x4_mmx2;
        pixf->sad   [LUMA_4x8  ] = xavs2_pixel_sad_4x8_mmx2;
        pixf->sad   [LUMA_4x4  ] = xavs2_pixel_sad_4x4_mmx2;


        pixf->sad_x3[LUMA_16x16] = xavs2_pixel_sad_x3_16x16_mmx2;
        pixf->sad_x3[LUMA_16x8 ] = xavs2_pixel_sad_x3_16x8_mmx2;
        pixf->sad_x3[LUMA_8x16 ] = xavs2_pixel_sad_x3_8x16_mmx2;
        pixf->sad_x3[LUMA_8x8  ] = xavs2_pixel_sad_x3_8x8_mmx2;
        pixf->sad_x3[LUMA_8x4  ] = xavs2_pixel_sad_x3_8x4_mmx2;
        pixf->sad_x3[LUMA_4x16 ] = xavs2_pixel_sad_x3_4x16_mmx2;
        pixf->sad_x3[LUMA_4x8  ] = xavs2_pixel_sad_x3_4x8_mmx2;
        pixf->sad_x3[LUMA_4x4  ] = xavs2_pixel_sad_x3_4x4_mmx2;

        pixf->sad_x4[LUMA_16x16] = xavs2_pixel_sad_x4_16x16_mmx2;
        pixf->sad_x4[LUMA_16x8 ] = xavs2_pixel_sad_x4_16x8_mmx2;
        pixf->sad_x4[LUMA_8x16 ] = xavs2_pixel_sad_x4_8x16_mmx2;
        pixf->sad_x4[LUMA_8x8  ] = xavs2_pixel_sad_x4_8x8_mmx2;
        pixf->sad_x4[LUMA_8x4  ] = xavs2_pixel_sad_x4_8x4_mmx2;
        pixf->sad_x4[LUMA_4x16 ] = xavs2_pixel_sad_x4_4x16_mmx2;
        pixf->sad_x4[LUMA_4x8  ] = xavs2_pixel_sad_x4_4x8_mmx2;
        pixf->sad_x4[LUMA_4x4  ] = xavs2_pixel_sad_x4_4x4_mmx2;

        pixf->ssd   [LUMA_16x16] = xavs2_pixel_ssd_16x16_mmx;
        pixf->ssd   [LUMA_16x8 ] = xavs2_pixel_ssd_16x8_mmx;
        pixf->ssd   [LUMA_8x16 ] = xavs2_pixel_ssd_8x16_mmx;
        pixf->ssd   [LUMA_4x16 ] = xavs2_pixel_ssd_4x16_mmx;
        pixf->ssd   [LUMA_8x8  ] = xavs2_pixel_ssd_8x8_mmx;
        pixf->ssd   [LUMA_8x4  ] = xavs2_pixel_ssd_8x4_mmx;
        pixf->ssd   [LUMA_4x8  ] = xavs2_pixel_ssd_4x8_mmx;
        pixf->ssd   [LUMA_4x4  ] = xavs2_pixel_ssd_4x4_mmx;

        pixf->satd  [LUMA_16x16] = xavs2_pixel_satd_16x16_mmx2;
        pixf->satd  [LUMA_16x8 ] = xavs2_pixel_satd_16x8_mmx2;
        pixf->satd  [LUMA_8x16 ] = xavs2_pixel_satd_8x16_mmx2;
        pixf->satd  [LUMA_4x16 ] = xavs2_pixel_satd_4x16_mmx2;
        pixf->satd  [LUMA_8x8  ] = xavs2_pixel_satd_8x8_mmx2;
        pixf->satd  [LUMA_8x4  ] = xavs2_pixel_satd_8x4_mmx2;
        pixf->satd  [LUMA_4x8  ] = xavs2_pixel_satd_4x8_mmx2;
        pixf->satd  [LUMA_4x4  ] = xavs2_pixel_satd_4x4_mmx2;

        //pixf->sa8d  [LUMA_16x16] = xavs2_pixel_satd_16x16_mmx2; // not found in x265
        //pixf->sa8d  [LUMA_16x8 ] = xavs2_pixel_satd_16x8_mmx2;
        //pixf->sa8d  [LUMA_8x16 ] = xavs2_pixel_satd_8x16_mmx2;
        //pixf->sa8d  [LUMA_4x16 ] = xavs2_pixel_satd_4x16_mmx2;
        //pixf->sa8d  [LUMA_8x8  ] = xavs2_pixel_satd_8x8_mmx2;
        //pixf->sa8d  [LUMA_8x4  ] = xavs2_pixel_satd_8x4_mmx2;
        //pixf->sa8d  [LUMA_4x8  ] = xavs2_pixel_satd_4x8_mmx2;
        pixf->sa8d  [LUMA_4x4  ] = xavs2_pixel_satd_4x4_mmx2;
    }

    if (cpuid & XAVS2_CPU_SSE2) {
        pixf->sad   [LUMA_16x16] = xavs2_pixel_sad_16x16_sse2;
        pixf->sad   [LUMA_16x8 ] = xavs2_pixel_sad_16x8_sse2;
        pixf->sad   [LUMA_16x12] = xavs2_pixel_sad_16x12_sse2;
        pixf->sad   [LUMA_16x32] = xavs2_pixel_sad_16x32_sse2;
        pixf->sad   [LUMA_16x64] = xavs2_pixel_sad_16x64_sse2;
        pixf->sad   [LUMA_16x4 ] = xavs2_pixel_sad_16x4_sse2;
        pixf->sad   [LUMA_32x8 ] = xavs2_pixel_sad_32x8_sse2;

        pixf->sad   [LUMA_32x24] = xavs2_pixel_sad_32x24_sse2;
        pixf->sad   [LUMA_32x32] = xavs2_pixel_sad_32x32_sse2;
        pixf->sad   [LUMA_32x16] = xavs2_pixel_sad_32x16_sse2;
        pixf->sad   [LUMA_32x64] = xavs2_pixel_sad_32x64_sse2;
        pixf->sad   [LUMA_8x32 ] = xavs2_pixel_sad_8x32_sse2;

        pixf->sad   [LUMA_64x16] = xavs2_pixel_sad_64x16_sse2;
        pixf->sad   [LUMA_64x32] = xavs2_pixel_sad_64x32_sse2;
        pixf->sad   [LUMA_64x48] = xavs2_pixel_sad_64x48_sse2;
        pixf->sad   [LUMA_64x64] = xavs2_pixel_sad_64x64_sse2;
        pixf->sad   [LUMA_48x64] = xavs2_pixel_sad_48x64_sse2;
        pixf->sad   [LUMA_24x32] = xavs2_pixel_sad_24x32_sse2;
        pixf->sad   [LUMA_12x16] = xavs2_pixel_sad_12x16_sse2;
        pixf->sa8d  [LUMA_64x16] = xavs2_pixel_sa8d_64x16_sse2;
        pixf->sa8d  [LUMA_64x32] = xavs2_pixel_sa8d_64x32_sse2;
        pixf->sa8d  [LUMA_64x48] = xavs2_pixel_sa8d_64x48_sse2;
        pixf->sa8d  [LUMA_48x64] = xavs2_pixel_sa8d_48x64_sse2;
        pixf->sa8d  [LUMA_24x32] = xavs2_pixel_sa8d_24x32_sse2;
        pixf->sa8d  [LUMA_8x16 ] = xavs2_pixel_sa8d_8x16_sse2;
        pixf->sa8d  [LUMA_16x32] = xavs2_pixel_sa8d_16x32_sse2;
        pixf->sa8d  [LUMA_32x64] = xavs2_pixel_sa8d_32x64_sse2;
        pixf->sa8d  [LUMA_8x8  ] = xavs2_pixel_sa8d_8x8_sse2;
        pixf->sa8d  [LUMA_16x16] = xavs2_pixel_sa8d_16x16_sse2;
        pixf->sa8d  [LUMA_32x32] = xavs2_pixel_sa8d_32x32_sse2;
        pixf->sa8d  [LUMA_64x64] = xavs2_pixel_sa8d_64x64_sse2;


        INIT_SATD(sse2);

        pixf->sad_x3[LUMA_16x16] = xavs2_pixel_sad_x3_16x16_sse2;
        pixf->sad_x3[LUMA_16x8 ] = xavs2_pixel_sad_x3_16x8_sse2;
        pixf->sad_x3[LUMA_8x16 ] = xavs2_pixel_sad_x3_8x16_sse2;
        pixf->sad_x3[LUMA_8x8  ] = xavs2_pixel_sad_x3_8x8_sse2;
        pixf->sad_x3[LUMA_8x4  ] = xavs2_pixel_sad_x3_8x4_sse2;

        pixf->sad_x4[LUMA_16x16] = xavs2_pixel_sad_x4_16x16_sse2;
        pixf->sad_x4[LUMA_16x8 ] = xavs2_pixel_sad_x4_16x8_sse2;
        pixf->sad_x4[LUMA_8x16 ] = xavs2_pixel_sad_x4_8x16_sse2;
        pixf->sad_x4[LUMA_8x8  ] = xavs2_pixel_sad_x4_8x8_sse2;
        pixf->sad_x4[LUMA_8x4  ] = xavs2_pixel_sad_x4_8x4_sse2;

        INIT_SSD (sse2);

    }

    if (cpuid & XAVS2_CPU_SSE3) {
        pixf->sad   [LUMA_16x16] = xavs2_pixel_sad_16x16_sse3;
        pixf->sad   [LUMA_16x8 ] = xavs2_pixel_sad_16x8_sse3;
        pixf->sad   [LUMA_16x12] = xavs2_pixel_sad_16x12_sse3;
        pixf->sad   [LUMA_16x32] = xavs2_pixel_sad_16x32_sse3;
        pixf->sad   [LUMA_16x64] = xavs2_pixel_sad_16x64_sse3;
        pixf->sad   [LUMA_16x4 ] = xavs2_pixel_sad_16x4_sse3;
        pixf->sad   [LUMA_32x8 ] = xavs2_pixel_sad_32x8_sse3;
        pixf->sad   [LUMA_32x24] = xavs2_pixel_sad_32x24_sse3;
        pixf->sad   [LUMA_32x32] = xavs2_pixel_sad_32x32_sse3;
        pixf->sad   [LUMA_32x16] = xavs2_pixel_sad_32x16_sse3;
        pixf->sad   [LUMA_32x64] = xavs2_pixel_sad_32x64_sse3;
        pixf->sad   [LUMA_8x32 ] = xavs2_pixel_sad_8x32_sse3;
        pixf->sad   [LUMA_64x16] = xavs2_pixel_sad_64x16_sse3;
        pixf->sad   [LUMA_64x32] = xavs2_pixel_sad_64x32_sse3;
        pixf->sad   [LUMA_64x48] = xavs2_pixel_sad_64x48_sse3;
        pixf->sad   [LUMA_64x64] = xavs2_pixel_sad_64x64_sse3;
        pixf->sad   [LUMA_48x64] = xavs2_pixel_sad_48x64_sse3;
        pixf->sad   [LUMA_24x32] = xavs2_pixel_sad_24x32_sse3;
        pixf->sad   [LUMA_12x16] = xavs2_pixel_sad_12x16_sse3;

        pixf->sad_x3[LUMA_16x16] = xavs2_pixel_sad_x3_16x16_sse3;
        pixf->sad_x3[LUMA_16x8 ] = xavs2_pixel_sad_x3_16x8_sse3;
        pixf->sad_x3[LUMA_16x4 ] = xavs2_pixel_sad_x3_16x4_sse3;

        pixf->sad_x4[LUMA_16x16] = xavs2_pixel_sad_x4_16x16_sse3;
        pixf->sad_x4[LUMA_16x8 ] = xavs2_pixel_sad_x4_16x8_sse3;
        pixf->sad_x4[LUMA_16x4 ] = xavs2_pixel_sad_x4_16x4_sse3;

    }

    if (cpuid & XAVS2_CPU_SSSE3) {
        INIT_SATD(ssse3);

        pixf->sad_x3[LUMA_64x64] = xavs2_pixel_sad_x3_64x64_ssse3;    /* 64x64 */
        pixf->sad_x3[LUMA_64x32] = xavs2_pixel_sad_x3_64x32_ssse3;
        pixf->sad_x3[LUMA_32x64] = xavs2_pixel_sad_x3_32x64_ssse3;
        pixf->sad_x3[LUMA_64x16] = xavs2_pixel_sad_x3_64x16_ssse3;
        pixf->sad_x3[LUMA_64x48] = xavs2_pixel_sad_x3_64x48_ssse3;
        pixf->sad_x3[LUMA_16x64] = xavs2_pixel_sad_x3_16x64_ssse3;
        pixf->sad_x3[LUMA_48x64] = xavs2_pixel_sad_x3_48x64_ssse3;
        pixf->sad_x3[LUMA_32x32] = xavs2_pixel_sad_x3_32x32_ssse3;    /* 32x32 */
        pixf->sad_x3[LUMA_32x16] = xavs2_pixel_sad_x3_32x16_ssse3;
        pixf->sad_x3[LUMA_16x32] = xavs2_pixel_sad_x3_16x32_ssse3;
        pixf->sad_x3[LUMA_32x8 ] = xavs2_pixel_sad_x3_32x8_ssse3;
        pixf->sad_x3[LUMA_32x24] = xavs2_pixel_sad_x3_32x24_ssse3;
        pixf->sad_x3[LUMA_8x32 ] = xavs2_pixel_sad_x3_8x32_ssse3;
        pixf->sad_x3[LUMA_24x32] = xavs2_pixel_sad_x3_24x32_ssse3;
        pixf->sad_x3[LUMA_16x16] = xavs2_pixel_sad_x3_16x16_ssse3;    /* 16x16 */
        pixf->sad_x3[LUMA_16x8 ] = xavs2_pixel_sad_x3_16x8_ssse3;
        pixf->sad_x3[LUMA_8x16 ] = xavs2_pixel_sad_x3_8x16_ssse3;
        pixf->sad_x3[LUMA_12x16] = xavs2_pixel_sad_x3_12x16_ssse3;

        pixf->sad_x4[LUMA_64x64] = xavs2_pixel_sad_x4_64x64_ssse3;    /* 64x64 */
        pixf->sad_x4[LUMA_64x32] = xavs2_pixel_sad_x4_64x32_ssse3;
        pixf->sad_x4[LUMA_32x64] = xavs2_pixel_sad_x4_32x64_ssse3;
        pixf->sad_x4[LUMA_64x16] = xavs2_pixel_sad_x4_64x16_ssse3;
        pixf->sad_x4[LUMA_64x48] = xavs2_pixel_sad_x4_64x48_ssse3;
        pixf->sad_x4[LUMA_16x64] = xavs2_pixel_sad_x4_16x64_ssse3;
        pixf->sad_x4[LUMA_48x64] = xavs2_pixel_sad_x4_48x64_ssse3;
        pixf->sad_x4[LUMA_32x32] = xavs2_pixel_sad_x4_32x32_ssse3;    /* 32x32 */
        pixf->sad_x4[LUMA_32x16] = xavs2_pixel_sad_x4_32x16_ssse3;
        pixf->sad_x4[LUMA_16x32] = xavs2_pixel_sad_x4_16x32_ssse3;
        pixf->sad_x4[LUMA_32x8 ] = xavs2_pixel_sad_x4_32x8_ssse3;
        pixf->sad_x4[LUMA_32x24] = xavs2_pixel_sad_x4_32x24_ssse3;
        pixf->sad_x4[LUMA_8x32 ] = xavs2_pixel_sad_x4_8x32_ssse3;
        pixf->sad_x4[LUMA_24x32] = xavs2_pixel_sad_x4_24x32_ssse3;
        pixf->sad_x4[LUMA_16x16] = xavs2_pixel_sad_x4_16x16_ssse3;    /* 16x16 */
        pixf->sad_x4[LUMA_16x8 ] = xavs2_pixel_sad_x4_16x8_ssse3;
        pixf->sad_x4[LUMA_8x16 ] = xavs2_pixel_sad_x4_8x16_ssse3;
        pixf->sad_x4[LUMA_12x16] = xavs2_pixel_sad_x4_12x16_ssse3;

        INIT_SSD (ssse3);

        pixf->sa8d  [LUMA_4x4  ] = xavs2_pixel_satd_4x4_ssse3;
        pixf->sa8d  [LUMA_8x8  ] = xavs2_pixel_sa8d_8x8_ssse3;
        pixf->sa8d  [LUMA_16x16] = xavs2_pixel_sa8d_16x16_ssse3;
        pixf->sa8d  [LUMA_32x32] = xavs2_pixel_sa8d_32x32_ssse3;
        pixf->sa8d  [LUMA_8x16 ] = xavs2_pixel_sa8d_8x16_ssse3;
        pixf->sa8d  [LUMA_16x32] = xavs2_pixel_sa8d_16x32_ssse3;
        pixf->sa8d  [LUMA_32x64] = xavs2_pixel_sa8d_32x64_ssse3;

    }

    if (cpuid & XAVS2_CPU_SSE4) {
        INIT_SATD(sse4);
        pixf->ssd   [LUMA_12x16] = xavs2_pixel_ssd_12x16_sse4;
        pixf->ssd   [LUMA_24x32] = xavs2_pixel_ssd_24x32_sse4;
        pixf->ssd   [LUMA_48x64] = xavs2_pixel_ssd_48x64_sse4;
        pixf->ssd   [LUMA_64x16] = xavs2_pixel_ssd_64x16_sse4;
        pixf->ssd   [LUMA_64x32] = xavs2_pixel_ssd_64x32_sse4;
        pixf->ssd   [LUMA_64x48] = xavs2_pixel_ssd_64x48_sse4;
        pixf->ssd   [LUMA_64x64] = xavs2_pixel_ssd_64x64_sse4;

        pixf->sa8d  [LUMA_4x4  ] = xavs2_pixel_satd_4x4_sse4;
        pixf->sa8d  [LUMA_8x8  ] = xavs2_pixel_sa8d_8x8_sse4;
        pixf->sa8d  [LUMA_16x16] = xavs2_pixel_sa8d_16x16_sse4;
        pixf->sa8d  [LUMA_32x32] = xavs2_pixel_sa8d_32x32_sse4;
        pixf->sa8d  [LUMA_8x16 ] = xavs2_pixel_sa8d_8x16_sse4;
        pixf->sa8d  [LUMA_16x32] = xavs2_pixel_sa8d_16x32_sse4;
        pixf->sa8d  [LUMA_32x64] = xavs2_pixel_sa8d_32x64_sse4;

    }

    if (cpuid & XAVS2_CPU_AVX) {
        INIT_SATD(avx);
        pixf->sad_x3[LUMA_64x64] = xavs2_pixel_sad_x3_64x64_avx;  /* 64x64 */
        pixf->sad_x3[LUMA_64x32] = xavs2_pixel_sad_x3_64x32_avx;
        pixf->sad_x3[LUMA_32x64] = xavs2_pixel_sad_x3_32x64_avx;
        pixf->sad_x3[LUMA_64x16] = xavs2_pixel_sad_x3_64x16_avx;
        pixf->sad_x3[LUMA_64x48] = xavs2_pixel_sad_x3_64x48_avx;
        pixf->sad_x3[LUMA_48x64] = xavs2_pixel_sad_x3_48x64_avx;
        pixf->sad_x3[LUMA_16x64] = xavs2_pixel_sad_x3_16x64_avx;
        pixf->sad_x3[LUMA_32x32] = xavs2_pixel_sad_x3_32x32_avx;  /* 32x32 */
        pixf->sad_x3[LUMA_32x16] = xavs2_pixel_sad_x3_32x16_avx;
        pixf->sad_x3[LUMA_16x32] = xavs2_pixel_sad_x3_16x32_avx;
        pixf->sad_x3[LUMA_32x8 ] = xavs2_pixel_sad_x3_32x8_avx;
        pixf->sad_x3[LUMA_32x24] = xavs2_pixel_sad_x3_32x24_avx;
        pixf->sad_x3[LUMA_24x32] = xavs2_pixel_sad_x3_24x32_avx;
        pixf->sad_x3[LUMA_16x16] = xavs2_pixel_sad_x3_16x16_avx;  /* 16x16 */
        pixf->sad_x3[LUMA_16x8 ] = xavs2_pixel_sad_x3_16x8_avx;
        pixf->sad_x3[LUMA_16x4 ] = xavs2_pixel_sad_x3_16x4_avx;
        pixf->sad_x3[LUMA_16x12] = xavs2_pixel_sad_x3_16x12_avx;
        pixf->sad_x3[LUMA_12x16] = xavs2_pixel_sad_x3_12x16_avx;

        pixf->sad_x4[LUMA_64x64] = xavs2_pixel_sad_x4_64x64_avx;  /* 64x64 */
        pixf->sad_x4[LUMA_64x32] = xavs2_pixel_sad_x4_64x32_avx;
        pixf->sad_x4[LUMA_32x64] = xavs2_pixel_sad_x4_32x64_avx;
        pixf->sad_x4[LUMA_64x16] = xavs2_pixel_sad_x4_64x16_avx;
        pixf->sad_x4[LUMA_64x48] = xavs2_pixel_sad_x4_64x48_avx;
        pixf->sad_x4[LUMA_16x64] = xavs2_pixel_sad_x4_16x64_avx;
        pixf->sad_x4[LUMA_48x64] = xavs2_pixel_sad_x4_48x64_avx;
        pixf->sad_x4[LUMA_32x32] = xavs2_pixel_sad_x4_32x32_avx;  /* 32x32 */
        pixf->sad_x4[LUMA_32x16] = xavs2_pixel_sad_x4_32x16_avx;
        pixf->sad_x4[LUMA_16x32] = xavs2_pixel_sad_x4_16x32_avx;
        pixf->sad_x4[LUMA_32x8 ] = xavs2_pixel_sad_x4_32x8_avx;
        pixf->sad_x4[LUMA_32x24] = xavs2_pixel_sad_x4_32x24_avx;
        pixf->sad_x4[LUMA_24x32] = xavs2_pixel_sad_x4_24x32_avx;
        pixf->sad_x4[LUMA_16x16] = xavs2_pixel_sad_x4_16x16_avx;  /* 16x16 */
        pixf->sad_x4[LUMA_16x8 ] = xavs2_pixel_sad_x4_16x8_avx;
        pixf->sad_x4[LUMA_16x4 ] = xavs2_pixel_sad_x4_16x4_avx;
        pixf->sad_x4[LUMA_16x12] = xavs2_pixel_sad_x4_16x12_avx;
        pixf->sad_x4[LUMA_12x16] = xavs2_pixel_sad_x4_12x16_avx;

        INIT_SSD (avx);

        pixf->sa8d  [LUMA_4x4  ] = xavs2_pixel_satd_4x4_avx;
        pixf->sa8d  [LUMA_8x8  ] = xavs2_pixel_sa8d_8x8_avx;
        pixf->sa8d  [LUMA_16x16] = xavs2_pixel_sa8d_16x16_avx;
        pixf->sa8d  [LUMA_32x32] = xavs2_pixel_sa8d_32x32_avx;
        pixf->sa8d  [LUMA_8x16 ] = xavs2_pixel_sa8d_8x16_avx;
        pixf->sa8d  [LUMA_16x32] = xavs2_pixel_sa8d_16x32_avx;
        pixf->sa8d  [LUMA_32x64] = xavs2_pixel_sa8d_32x64_avx;
        pixf->sa8d  [LUMA_64x64] = xavs2_pixel_sa8d_64x64_avx;

    }

    if (cpuid & XAVS2_CPU_XOP) {
        INIT_SATD(xop);
        pixf->ssd   [LUMA_16x16] = xavs2_pixel_ssd_16x16_xop;
        pixf->ssd   [LUMA_16x8 ] = xavs2_pixel_ssd_16x8_xop;
        pixf->ssd   [LUMA_8x16 ] = xavs2_pixel_ssd_8x16_xop;
        pixf->ssd   [LUMA_8x8  ] = xavs2_pixel_ssd_8x8_xop;
        pixf->ssd   [LUMA_8x4  ] = xavs2_pixel_ssd_8x4_xop;

        //pixf->sa8d  [LUMA_4x4  ] = xavs2_pixel_satd_4x4_xop; // in x265, this one is broken
        pixf->sa8d  [LUMA_8x8  ] = xavs2_pixel_sa8d_8x8_xop;
        pixf->sa8d  [LUMA_16x16] = xavs2_pixel_sa8d_16x16_xop;
        pixf->sa8d  [LUMA_32x32] = xavs2_pixel_sa8d_32x32_xop;
        pixf->sa8d  [LUMA_8x16 ] = xavs2_pixel_sa8d_8x16_xop;
        pixf->sa8d  [LUMA_16x32] = xavs2_pixel_sa8d_16x32_xop;
        pixf->sa8d  [LUMA_32x64] = xavs2_pixel_sa8d_32x64_xop;

    }

#if ARCH_X86_64
    if (cpuid & XAVS2_CPU_AVX2) {
        pixf->sad   [LUMA_32x8 ] = xavs2_pixel_sad_32x8_avx2;
        pixf->sad   [LUMA_32x16] = xavs2_pixel_sad_32x16_avx2;
        pixf->sad   [LUMA_32x24] = xavs2_pixel_sad_32x24_avx2;
        pixf->sad   [LUMA_32x32] = xavs2_pixel_sad_32x32_avx2;
        pixf->sad   [LUMA_32x64] = xavs2_pixel_sad_32x64_avx2;
        pixf->sad   [LUMA_48x64] = xavs2_pixel_sad_48x64_avx2;
        pixf->sad   [LUMA_64x16] = xavs2_pixel_sad_64x16_avx2;
        pixf->sad   [LUMA_64x32] = xavs2_pixel_sad_64x32_avx2;
        pixf->sad   [LUMA_64x48] = xavs2_pixel_sad_64x48_avx2;
        pixf->sad   [LUMA_64x64] = xavs2_pixel_sad_64x64_avx2;

        pixf->sad_x3[LUMA_8x16 ] = xavs2_pixel_sad_x3_8x16_avx2;
        pixf->sad_x3[LUMA_8x8  ] = xavs2_pixel_sad_x3_8x8_avx2;
        pixf->sad_x3[LUMA_8x4  ] = xavs2_pixel_sad_x3_8x4_avx2;
        pixf->sad_x4[LUMA_8x8  ] = xavs2_pixel_sad_x4_8x8_avx2;

        pixf->ssd   [LUMA_64x64] = xavs2_pixel_ssd_64x64_avx2;
        pixf->ssd   [LUMA_32x32] = xavs2_pixel_ssd_32x32_avx2;
        pixf->ssd   [LUMA_16x16] = xavs2_pixel_ssd_16x16_avx2;
        pixf->ssd   [LUMA_16x8 ] = xavs2_pixel_ssd_16x8_avx2;

        pixf->satd  [LUMA_16x16] = xavs2_pixel_satd_16x16_avx2;
        pixf->satd  [LUMA_16x8 ] = xavs2_pixel_satd_16x8_avx2;
        pixf->satd  [LUMA_8x16 ] = xavs2_pixel_satd_8x16_avx2;
        pixf->satd  [LUMA_8x8  ] = xavs2_pixel_satd_8x8_avx2;
        pixf->satd  [LUMA_64x64] = xavs2_pixel_satd_64x64_avx2;
        pixf->satd  [LUMA_64x32] = xavs2_pixel_satd_64x32_avx2;
        pixf->satd  [LUMA_32x64] = xavs2_pixel_satd_32x64_avx2;
        pixf->satd  [LUMA_64x16] = xavs2_pixel_satd_64x16_avx2;
        pixf->satd  [LUMA_64x48] = xavs2_pixel_satd_64x48_avx2;
        pixf->satd  [LUMA_16x64] = xavs2_pixel_satd_16x64_avx2;
        pixf->satd  [LUMA_48x64] = xavs2_pixel_satd_48x64_avx2;
        pixf->satd  [LUMA_32x32] = xavs2_pixel_satd_32x32_avx2;
        pixf->satd  [LUMA_32x16] = xavs2_pixel_satd_32x16_avx2;
        pixf->satd  [LUMA_16x32] = xavs2_pixel_satd_16x32_avx2;
        pixf->satd  [LUMA_32x24] = xavs2_pixel_satd_32x24_avx2;
        pixf->satd  [LUMA_16x4 ] = xavs2_pixel_satd_16x4_avx2;
        pixf->satd  [LUMA_16x12] = xavs2_pixel_satd_16x12_avx2;

        pixf->sad_x3[LUMA_32x8 ] = xavs2_pixel_sad_x3_32x8_avx2;
        pixf->sad_x3[LUMA_32x16] = xavs2_pixel_sad_x3_32x16_avx2;
        pixf->sad_x3[LUMA_32x24] = xavs2_pixel_sad_x3_32x24_avx2;
        pixf->sad_x3[LUMA_32x32] = xavs2_pixel_sad_x3_32x32_avx2;
        pixf->sad_x3[LUMA_32x64] = xavs2_pixel_sad_x3_32x64_avx2;
        pixf->sad_x3[LUMA_48x64] = xavs2_pixel_sad_x3_48x64_avx2;
        pixf->sad_x3[LUMA_64x16] = xavs2_pixel_sad_x3_64x16_avx2;
        pixf->sad_x3[LUMA_64x32] = xavs2_pixel_sad_x3_64x32_avx2;
        pixf->sad_x3[LUMA_64x48] = xavs2_pixel_sad_x3_64x48_avx2;
        pixf->sad_x3[LUMA_64x64] = xavs2_pixel_sad_x3_64x64_avx2;

        pixf->sad_x4[LUMA_16x8 ] = xavs2_pixel_sad_x4_16x8_avx2;
        pixf->sad_x4[LUMA_16x12] = xavs2_pixel_sad_x4_16x12_avx2;
        pixf->sad_x4[LUMA_16x16] = xavs2_pixel_sad_x4_16x16_avx2;
        pixf->sad_x4[LUMA_16x32] = xavs2_pixel_sad_x4_16x32_avx2;
        pixf->sad_x4[LUMA_32x8 ] = xavs2_pixel_sad_x4_32x8_avx2;
        pixf->sad_x4[LUMA_32x16] = xavs2_pixel_sad_x4_32x16_avx2;
        pixf->sad_x4[LUMA_32x24] = xavs2_pixel_sad_x4_32x24_avx2;
        pixf->sad_x4[LUMA_32x32] = xavs2_pixel_sad_x4_32x32_avx2;
        pixf->sad_x4[LUMA_32x64] = xavs2_pixel_sad_x4_32x64_avx2;
        pixf->sad_x4[LUMA_48x64] = xavs2_pixel_sad_x4_48x64_avx2;
        pixf->sad_x4[LUMA_64x16] = xavs2_pixel_sad_x4_64x16_avx2;
        pixf->sad_x4[LUMA_64x32] = xavs2_pixel_sad_x4_64x32_avx2;
        pixf->sad_x4[LUMA_64x48] = xavs2_pixel_sad_x4_64x48_avx2;
        pixf->sad_x4[LUMA_64x64] = xavs2_pixel_sad_x4_64x64_avx2;

        pixf->sa8d  [LUMA_8x8  ] = xavs2_pixel_sa8d_8x8_avx2;
        pixf->sa8d  [LUMA_16x16] = xavs2_pixel_sa8d_16x16_avx2;
        pixf->sa8d  [LUMA_32x32] = xavs2_pixel_sa8d_32x32_avx2;
    }
#endif

    /* -------------------------------------------------------------
     * init AVG functions
     */
#define INIT_PIXEL_AVG(w, h, suffix) \
    pixf->avg[LUMA_## w ##x## h] = xavs2_pixel_avg_##w##x##h##_##suffix

    if (cpuid & XAVS2_CPU_MMX2) {
        INIT_PIXEL_AVG(64, 64, mmx2);
        INIT_PIXEL_AVG(64, 16, mmx2);
        INIT_PIXEL_AVG(64, 48, mmx2);
        INIT_PIXEL_AVG(16, 64, mmx2);
        INIT_PIXEL_AVG(48, 64, mmx2);
        INIT_PIXEL_AVG(32, 32, mmx2);
        INIT_PIXEL_AVG(32, 16, mmx2);
        INIT_PIXEL_AVG(16, 32, mmx2);
        INIT_PIXEL_AVG(32,  8, mmx2);
        INIT_PIXEL_AVG(32, 24, mmx2);
        INIT_PIXEL_AVG(24, 32, mmx2);
        INIT_PIXEL_AVG(16, 16, mmx2);
        INIT_PIXEL_AVG(16,  8, mmx2);
        INIT_PIXEL_AVG(16,  4, mmx2);
        INIT_PIXEL_AVG(16, 12, mmx2);
        INIT_PIXEL_AVG( 8, 32, mmx2);
        INIT_PIXEL_AVG( 8, 16, mmx2);
        INIT_PIXEL_AVG( 4, 16, mmx2);
        INIT_PIXEL_AVG(12, 16, mmx2);
        INIT_PIXEL_AVG( 8,  8, mmx2);
        INIT_PIXEL_AVG( 8,  4, mmx2);
        INIT_PIXEL_AVG( 4,  8, mmx2);
        INIT_PIXEL_AVG( 4,  4, mmx2);
    }

    if (cpuid & XAVS2_CPU_SSE2) {
        INIT_PIXEL_AVG(64, 64, sse2);
        INIT_PIXEL_AVG(64, 32, sse2);
        INIT_PIXEL_AVG(32, 64, sse2);
        INIT_PIXEL_AVG(64, 16, sse2);
        INIT_PIXEL_AVG(64, 48, sse2);
        INIT_PIXEL_AVG(16, 64, sse2);
        INIT_PIXEL_AVG(48, 64, sse2);
        INIT_PIXEL_AVG(32, 32, sse2);
        INIT_PIXEL_AVG(32, 16, sse2);
        INIT_PIXEL_AVG(16, 32, sse2);
        INIT_PIXEL_AVG(32,  8, sse2);
        INIT_PIXEL_AVG(32, 24, sse2);
        INIT_PIXEL_AVG(16, 16, sse2);
        INIT_PIXEL_AVG(16,  8, sse2);
        INIT_PIXEL_AVG(16,  4, sse2);
        INIT_PIXEL_AVG(16, 12, sse2);
        INIT_PIXEL_AVG( 8, 32, sse2);
        INIT_PIXEL_AVG(24, 32, sse2);
        INIT_PIXEL_AVG( 8, 16, sse2);
        INIT_PIXEL_AVG(12, 16, sse2);
        INIT_PIXEL_AVG( 8,  8, sse2);
        INIT_PIXEL_AVG( 8,  4, sse2);
    }

    if (cpuid & XAVS2_CPU_SSE3) {
        INIT_PIXEL_FUNC(avg, _ssse3);
    }

    if (cpuid & XAVS2_CPU_AVX2) {
#if ARCH_X86_64
        INIT_PIXEL_AVG(64, 64, avx2);
        INIT_PIXEL_AVG(64, 32, avx2);
        INIT_PIXEL_AVG(64, 16, avx2);
        INIT_PIXEL_AVG(64, 48, avx2);
        INIT_PIXEL_AVG(32, 32, avx2);
        INIT_PIXEL_AVG(32, 64, avx2);
        INIT_PIXEL_AVG(32, 16, avx2);
        INIT_PIXEL_AVG(32,  8, avx2);
        INIT_PIXEL_AVG(32, 24, avx2);
#endif
        INIT_PIXEL_AVG(16, 64, avx2);
        INIT_PIXEL_AVG(16, 32, avx2);
        INIT_PIXEL_AVG(16, 16, avx2);
        INIT_PIXEL_AVG(16,  8, avx2);
        INIT_PIXEL_AVG(16,  4, avx2);
        INIT_PIXEL_AVG(16, 12, avx2);
    }

    /* block average */
    if (cpuid & XAVS2_CPU_SSE42) {
        pixf->average = xavs2_pixel_average_sse128;
    }
#if _MSC_VER
    if (cpuid & XAVS2_CPU_AVX2) {
        pixf->average = xavs2_pixel_average_avx;
    }
#endif
#endif

    /* init functions of block operation : copy/add/sub */
    init_block_opreation_funcs(cpuid, pixf);


#undef INIT_PIXEL_AVG
#undef INIT_PIXEL_FUNC
#undef INIT_SATD
#undef INIT_SSD
}


/* ---------------------------------------------------------------------------
 */
static int mad_NxN_c(pel_t *p_src, int i_src, int cu_size)
{
    pel_t *p_src_base = p_src;
    int num_pix = cu_size * cu_size;
    int x, y;
    int sum = 0;
    int f_avg = 0;                 /* average of all pixels in current block */
    int mad = 0;

    /* cal average */
    for (y = 0; y < cu_size; ++y) {
        for (x = 0; x < cu_size; ++x) {
            sum += p_src[x];
        }
        p_src += i_src;
    }
    f_avg = (sum + (num_pix >> 1)) / num_pix;

    /* cal mad */
    p_src = p_src_base;
    for (y = 0; y < cu_size; ++y) {
        for (x = 0; x < cu_size; ++x) {
            int f_pxl = p_src[x];
            mad += XAVS2_ABS(f_pxl - f_avg);
        }
        p_src += i_src;
    }

    return mad;
}


/* ---------------------------------------------------------------------------
 */
void xavs2_mad_init(uint32_t cpuid, mad_funcs_t *madf)
{
    madf[B16X16_IN_BIT - MIN_CU_SIZE_IN_BIT] = mad_NxN_c;
    madf[B32X32_IN_BIT - MIN_CU_SIZE_IN_BIT] = mad_NxN_c;
    madf[B64X64_IN_BIT - MIN_CU_SIZE_IN_BIT] = mad_NxN_c;

    /* init asm function handles */
#if HAVE_MMX
    /* functions defined in file intrinsic_mad.c */
    if (cpuid & XAVS2_CPU_SSE2) {
        madf[B16X16_IN_BIT - MIN_CU_SIZE_IN_BIT] = mad_16x16_sse128;
        madf[B32X32_IN_BIT - MIN_CU_SIZE_IN_BIT] = mad_32x32_sse128;
        madf[B64X64_IN_BIT - MIN_CU_SIZE_IN_BIT] = mad_64x64_sse128;
    }
#endif //if HAVE_MMX
}

