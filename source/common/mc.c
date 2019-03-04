/*
 * mc.c
 *
 * Description of this file:
 *    MC functions definition of the xavs2 library
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
#include "predict.h"
#include "wrapper.h"
#include "frame.h"
#include "cpu.h"
#include "mc.h"

#if HAVE_MMX
#include "x86/mc.h"
#include "vec/intrinsic.h"
#endif


/**
 * ===========================================================================
 * global/local variables
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
const int16_t tab_dmh_pos[DMH_MODE_NUM + DMH_MODE_NUM - 1][2] = {
    { 0,  0 },
    { 1,  0 },
    { 0,  1 },
    { 1, -1 },
    { 1,  1 },
    { 2,  0 },
    { 0,  2 },
    { 2, -2 },
    { 2,  2 }
};

/* ---------------------------------------------------------------------------
 * interpolate filter (luma) */
ALIGN16(static const int8_t INTPL_FILTERS[4][8]) = {
    {  0, 0,   0, 64,  0,  0,  0,  0 }, /* for full-pixel, no use */
    { -1, 4, -10, 57, 19, -7,  3, -1 },
    { -1, 4, -11, 40, 40, -11, 4, -1 },
    { -1, 3,  -7, 19, 57, -10, 4, -1 }
};

/* ---------------------------------------------------------------------------
 * interpolate filter (chroma) */
ALIGN16(static const int8_t INTPL_FILTERS_C[8][4]) = {
    {  0, 64,  0,  0 },                 /* for full-pixel, no use */
    { -4, 62,  6,  0 },
    { -6, 56, 15, -1 },
    { -5, 47, 25, -3 },
    { -4, 36, 36, -4 },
    { -3, 25, 47, -5 },
    { -1, 15, 56, -6 },
    {  0,  6, 62, -4 }
};

/* ---------------------------------------------------------------------------
 * interpolate offsets */
static const int MC_OFFSET = 8;
static const int PAD_OFFSET = 4;


/* ---------------------------------------------------------------------------
 * luma interpolating position
 */
enum intpl_pos_e {
    INTPL_POS_0  = 0,           /* decoded luma full pel plane  */
    INTPL_POS_A  = 1,           /* interpolating position: a    */
    INTPL_POS_B  = 2,           /*                              */
    INTPL_POS_C  = 3,           /*        |                     */
    INTPL_POS_D  = 4,           /*        |  0   1   2   3      */
    INTPL_POS_E  = 5,           /*    ----+------------------   */
    INTPL_POS_F  = 6,           /*        |                     */
    INTPL_POS_G  = 7,           /*      0 |  *   a   b   c      */
    INTPL_POS_H  = 8,           /*        |                     */
    INTPL_POS_I  = 9,           /*      1 |  d   e   f   g      */
    INTPL_POS_J  = 10,          /*        |                     */
    INTPL_POS_K  = 11,          /*      2 |  h   i   j   k      */
    INTPL_POS_N  = 12,          /*        |                     */
    INTPL_POS_P  = 13,          /*      3 |  n   p   q   r      */
    INTPL_POS_Q  = 14,          /*        |                     */
    INTPL_POS_R  = 15           /*                              */
};


/**
 * ===========================================================================
 * macros
 * ===========================================================================
 */

#define FLT_8TAP_HOR(src, i, coef) ( \
    (src)[i - 3] * (coef)[0] + \
    (src)[i - 2] * (coef)[1] + \
    (src)[i - 1] * (coef)[2] + \
    (src)[i    ] * (coef)[3] + \
    (src)[i + 1] * (coef)[4] + \
    (src)[i + 2] * (coef)[5] + \
    (src)[i + 3] * (coef)[6] + \
    (src)[i + 4] * (coef)[7])

#define FLT_8TAP_VER(src, i, i_src, coef) ( \
    (src)[i - 3 * i_src] * (coef)[0] + \
    (src)[i - 2 * i_src] * (coef)[1] + \
    (src)[i - 1 * i_src] * (coef)[2] + \
    (src)[i            ] * (coef)[3] + \
    (src)[i + 1 * i_src] * (coef)[4] + \
    (src)[i + 2 * i_src] * (coef)[5] + \
    (src)[i + 3 * i_src] * (coef)[6] + \
    (src)[i + 4 * i_src] * (coef)[7])

#define FLT_4TAP_HOR(src, i, coef) ( \
    (src)[i - 1] * (coef)[0] + \
    (src)[i    ] * (coef)[1] + \
    (src)[i + 1] * (coef)[2] + \
    (src)[i + 2] * (coef)[3])

#define FLT_4TAP_VER(src, i, i_src, coef) ( \
    (src)[i - 1 * i_src] * (coef)[0] + \
    (src)[i            ] * (coef)[1] + \
    (src)[i + 1 * i_src] * (coef)[2] + \
    (src)[i + 2 * i_src] * (coef)[3])


/**
 * ===========================================================================
 * local function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static void
mc_copy_c(pel_t *dst, intptr_t i_dst, pel_t *src, intptr_t i_src, int w, int h)
{
    while (h--) {
        memcpy(dst, src, w * sizeof(pel_t));
        dst += i_dst;
        src += i_src;
    }
}

/* ---------------------------------------------------------------------------
 * plane copy
 */
static void
plane_copy_c(pel_t *dst, intptr_t i_dst, pel_t *src, intptr_t i_src, int w, int h)
{
    while (h--) {
        memcpy(dst, src, w * sizeof(pel_t));
        dst += i_dst;
        src += i_src;
    }
}

#define PLANE_COPY(align, cpu) \
void plane_copy_##cpu(pel_t *dst, intptr_t i_dst, pel_t *src, intptr_t i_src, int w, int h)\
{\
    int c_w = (align) / sizeof(pel_t) - 1;\
    if (w < 256) { /* tiny resolutions don't want non-temporal hints. dunno the exact threshold. */\
        plane_copy_c( dst, i_dst, src, i_src, w, h );\
    } else if (!(w & c_w)) {\
        xavs2_plane_copy_core_##cpu( dst, i_dst, src, i_src, w, h );\
    } else {\
        if (--h > 0) {\
            if( i_src > 0 ) {\
                xavs2_plane_copy_core_##cpu( dst, i_dst, src, i_src, (w+c_w)&~c_w, h );\
                dst += i_dst * h;\
                src += i_src * h;\
            } else {\
                xavs2_plane_copy_core_##cpu( dst+i_dst, i_dst, src+i_src, i_src, (w+c_w)&~c_w, h );\
            }\
        }\
        /* use plain memcpy on the last line (in memory order) to avoid overreading src. */\
        memcpy( dst, src, w*sizeof(pel_t) );\
    }\
}

#if HAVE_MMX
PLANE_COPY(16, mmx2)
#endif

/* ---------------------------------------------------------------------------
 * deinterleave copy, for chroma planes
 */
static void
plane_copy_deinterleave_c(pel_t *dstu, intptr_t i_dstu, pel_t *dstv, intptr_t i_dstv, pel_t *src, intptr_t i_src, int w, int h)
{
    int x, y;

    for (y = 0; y < h; y++, dstu += i_dstu, dstv += i_dstv, src += i_src) {
        for (x = 0; x < w; x++) {
            dstu[x] = src[2*x    ];
            dstv[x] = src[2*x + 1];
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void *memzero_aligned_c(void *dst, size_t n)
{
    return memset(dst, 0, n);
}


/* ---------------------------------------------------------------------------
 */
void mem_repeat_i_c(void *dst, int val, size_t count)
{
    int *p = (int *)dst;

    for (; count != 0; count--) {
        *p++ = val;
    }
}

/* ---------------------------------------------------------------------------
 */
void mem_repeat_8i_c(void *dst, int val, size_t count)
{
    int64_t *p = (int64_t *)dst;
    int64_t val64 = val;

    val64 = (val64 << 32) | val;
    count = (count + 7) >> 3;

    for (; count != 0; count--) {
        *p++ = val64;
        *p++ = val64;
        *p++ = val64;
        *p++ = val64;
    }
}

/* ---------------------------------------------------------------------------
 */
static void
intpl_chroma_block_hor_c(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, int8_t const *coeff)
{
    int x, y, v;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            v = (FLT_4TAP_HOR(src, x, coeff) + 32) >> 6;
            dst[x] = (pel_t)XAVS2_CLIP1(v);
        }
        src += i_src;
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static void
intpl_chroma_block_ver_c(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, int8_t const *coeff)
{
    int x, y, v;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            v = (FLT_4TAP_VER(src, x, i_src, coeff) + 32) >> 6;
            dst[x] = (pel_t)XAVS2_CLIP1(v);
        }
        src += i_src;
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static void
intpl_chroma_block_ext_c(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff_h, const int8_t *coeff_v)
{
    ALIGN16(int32_t tmp_res[(32 + 3) * 32]);
    int32_t *tmp = tmp_res;
    const int shift1 = g_bit_depth - 8;
    const int add1   = (1 << shift1) >> 1;
    const int shift2 = 20 - g_bit_depth;
    const int add2   = 1 << (shift2 - 1); // 1<<(19-g_bit_depth)
    int x, y, v;

    src -= i_src;
    for (y = -1; y < height + 2; y++) {
        for (x = 0; x < width; x++) {
            v = FLT_4TAP_HOR(src, x, coeff_h);
            tmp[x] = (v + add1) >> shift1;
        }
        src += i_src;
        tmp += 32;
    }
    tmp = tmp_res + 32;
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            v = (FLT_4TAP_VER(tmp, x, 32, coeff_v) + add2) >> shift2;
            dst[x] = (pel_t)XAVS2_CLIP1(v);
        }
        dst += i_dst;
        tmp += 32;
    }
}

/* ---------------------------------------------------------------------------
 */
static void
intpl_luma_block_hor_c(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, int8_t const *coeff)
{
    int x, y, v;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            v = (FLT_8TAP_HOR(src, x, coeff) + 32) >> 6;
            dst[x] = (pel_t)XAVS2_CLIP1(v);
        }
        src += i_src;
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
#define intpl_luma_block_ver_c intpl_luma_ver_c

/* ---------------------------------------------------------------------------
 */
static void
intpl_luma_block_ext_c(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff_h, const int8_t *coeff_v)
{
#define TMP_STRIDE      64

    const int shift1 = g_bit_depth - 8;
    const int add1   = (1 << shift1) >> 1;
    const int shift2 = 20 - g_bit_depth;
    const int add2   = 1 << (shift2 - 1);//1<<(19-bit_depth)

    ALIGN16(mct_t tmp_buf[(64 + 7) * TMP_STRIDE]);
    mct_t *tmp = tmp_buf;
    int x, y, v;

    src -= 3 * i_src;
    for (y = -3; y < height + 4; y++) {
        for (x = 0; x < width; x++) {
            v = FLT_8TAP_HOR(src, x, coeff_h);
            tmp[x] = (mct_t)((v + add1) >> shift1);
        }
        src += i_src;
        tmp += TMP_STRIDE;
    }

    tmp = tmp_buf + 3 * TMP_STRIDE;
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            v = (FLT_8TAP_VER(tmp, x, TMP_STRIDE, coeff_v) + add2) >> shift2;
            dst[x] = (pel_t)XAVS2_CLIP1(v);
        }

        dst += i_dst;
        tmp += TMP_STRIDE;
    }

#undef TMP_STRIDE
}

/* ---------------------------------------------------------------------------
 */
static void
intpl_luma_hor_c(pel_t *dst, int i_dst, mct_t *tmp, int i_tmp, pel_t *src, int i_src, int width, int height, int8_t const *coeff)
{
    int x, y, v;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            v = FLT_8TAP_HOR(src, x, coeff);
            tmp[x] = (mct_t)v;
            dst[x] = (pel_t)XAVS2_CLIP1((v + 32) >> 6);
        }
        src += i_src;
        tmp += i_tmp;
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static void
intpl_luma_ver_c(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, int8_t const *coeff)
{
    int x, y;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            int v = FLT_8TAP_VER(src, x, i_src, coeff);
            v = (v + 32) >> 6;
            dst[x] = (pel_t)XAVS2_CLIP1(v);
        }
        src += i_src;
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static void
intpl_luma_ver_x3_c(pel_t *const dst[3], int i_dst, pel_t *src, int i_src, int width, int height, int8_t const **coeff)
{
    int x, y, v;
    pel_t *dst0 = dst[0];
    pel_t *dst1 = dst[1];
    pel_t *dst2 = dst[2];

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            v = FLT_8TAP_VER(src, x, i_src, coeff[0]);
            dst0[x] = (pel_t)XAVS2_CLIP1((v + 32) >> 6);
            v = FLT_8TAP_VER(src, x, i_src, coeff[1]);
            dst1[x] = (pel_t)XAVS2_CLIP1((v + 32) >> 6);
            v = FLT_8TAP_VER(src, x, i_src, coeff[2]);
            dst2[x] = (pel_t)XAVS2_CLIP1((v + 32) >> 6);
        }
        src  += i_src;
        dst0 += i_dst;
        dst1 += i_dst;
        dst2 += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static void
intpl_luma_hor_x3_c(pel_t *const dst[3], int i_dst, mct_t *const tmp[3], int i_tmp, pel_t *src, int i_src, int width, int height, const int8_t **coeff)
{
    int x, y, v;
    pel_t *dst0 = dst[0];
    pel_t *dst1 = dst[1];
    pel_t *dst2 = dst[2];
    mct_t *tmp0 = tmp[0];
    mct_t *tmp1 = tmp[1];
    mct_t *tmp2 = tmp[2];

    for (y = 0; y < height; y++) {
        for(x = 0; x < width; x++) {
            v = FLT_8TAP_HOR(src, x, coeff[0]);
            tmp0[x] = (mct_t)v;
            dst0[x] = (pel_t)XAVS2_CLIP1((v + 32) >> 6);
            v = FLT_8TAP_HOR(src, x, coeff[1]);
            tmp1[x] = (mct_t)v;
            dst1[x] = (pel_t)XAVS2_CLIP1((v + 32) >> 6);
            v = FLT_8TAP_HOR(src, x, coeff[2]);
            tmp2[x] = (mct_t)v;
            dst2[x] = (pel_t)XAVS2_CLIP1((v + 32) >> 6);
        }
        src  += i_src;
        tmp0 += i_tmp;
        tmp1 += i_tmp;
        tmp2 += i_tmp;
        dst0 += i_dst;
        dst1 += i_dst;
        dst2 += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static void
intpl_luma_ext_c(pel_t *dst, int i_dst, mct_t *tmp, int i_tmp, int width, int height, const int8_t *coeff)
{
    const int MC_SHIFT = 20 - g_bit_depth;
    const int MC_ADD = 1 << (MC_SHIFT - 1);   // (1 << (19-g_bit_depth))
    int x, y;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            int v = FLT_8TAP_VER(tmp, x, i_tmp, coeff);
            v = (v + MC_ADD) >> MC_SHIFT;
            dst[x] = (pel_t)XAVS2_CLIP1(v);
        }
        dst += i_dst;
        tmp += i_tmp;
    }
}

static void
intpl_luma_ext_x3_c(pel_t *const dst[3], int i_dst, mct_t *tmp, int i_tmp, int width, int height, const int8_t **coeff)
{
    const int MC_SHIFT = 20 - g_bit_depth;
    const int MC_ADD = 1 << (MC_SHIFT - 1);   // (1 << (19-g_bit_depth))
    int x, y;

    pel_t *dst0 = dst[0];
    pel_t *dst1 = dst[1];
    pel_t *dst2 = dst[2];

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            int v;
            v = FLT_8TAP_VER(tmp, x, i_tmp, coeff[0]);
            v = (v + MC_ADD) >> MC_SHIFT;
            dst0[x] = (pel_t)XAVS2_CLIP1(v);
            v = FLT_8TAP_VER(tmp, x, i_tmp, coeff[1]);
            v = (v + MC_ADD) >> MC_SHIFT;
            dst1[x] = (pel_t)XAVS2_CLIP1(v);
            v = FLT_8TAP_VER(tmp, x, i_tmp, coeff[2]);
            v = (v + MC_ADD) >> MC_SHIFT;
            dst2[x] = (pel_t)XAVS2_CLIP1(v);
        }
        dst0 += i_dst;
        dst1 += i_dst;
        dst2 += i_dst;
        tmp  += i_tmp;
    }
}

/**
 * ===========================================================================
 * interface function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * predict one component of a luma block
 *   ref_idx - reference frame (0.. / -1:backward)
 */
void mc_luma(pel_t *p_pred, int i_pred, int pix_quad_x, int pix_quad_y,
             int width, int height, const xavs2_frame_t *p_ref_frm)
{
    int x = (pix_quad_x >> 2);
    int y = (pix_quad_y >> 2);
    int dx = pix_quad_x & 3;
    int dy = pix_quad_y & 3;
    int i_src = p_ref_frm->i_stride[0];
    pel_t *src = p_ref_frm->filtered[(dy << 2) + dx];

    /* fetch prediction result */
#if ENABLE_FRAME_SUBPEL_INTPL
    if (src != NULL) {
        src += y * i_src + x;
        g_funcs.pixf.copy_pp[PART_INDEX(width, height)](p_pred, i_pred, src, i_src);
    } else {
#endif
        src = p_ref_frm->filtered[0] + y * i_src + x;
        if (dx == 0 && dy == 0) {
            g_funcs.pixf.copy_pp[PART_INDEX(width, height)](p_pred, i_pred, src, i_src);
        } else if (dy == 0) {
            g_funcs.intpl_luma_block_hor(p_pred, i_pred, src, i_src, width, height, INTPL_FILTERS[dx]);
        } else if (dx == 0) {
            g_funcs.intpl_luma_block_ver(p_pred, i_pred, src, i_src, width, height, INTPL_FILTERS[dy]);
        } else {
            g_funcs.intpl_luma_block_ext(p_pred, i_pred, src, i_src, width, height, INTPL_FILTERS[dx], INTPL_FILTERS[dy]);
        }
#if ENABLE_FRAME_SUBPEL_INTPL
    }
#endif
}

/* ---------------------------------------------------------------------------
 */
void interpolate_sample_rows(xavs2_t *h, xavs2_frame_t* frm, int start_y, int height, int b_start, int b_end)
{
    int stride  = frm->i_stride[IMG_Y];         // for src and dst
    int i_tmp   = frm->i_width[IMG_Y] + 2 * XAVS2_PAD;
    int width   = frm->i_width[IMG_Y] + 2 * PAD_OFFSET;
    int off_dst = start_y * stride - PAD_OFFSET;
    pel_t *src  = frm->planes[IMG_Y] + off_dst; // reconstructed luma plane
    pel_t *p_dst[3];
    const int8_t *p_coeffs[3];
    pel_t *dst;
    mct_t *intpl_tmp[3];

    /* -------------------------------------------------------------
     * init */

    intpl_tmp[0] = h->img4Y_tmp[0] + (XAVS2_PAD + start_y) * i_tmp + XAVS2_PAD - PAD_OFFSET;
    intpl_tmp[1] = h->img4Y_tmp[1] + (XAVS2_PAD + start_y) * i_tmp + XAVS2_PAD - PAD_OFFSET;
    intpl_tmp[2] = h->img4Y_tmp[2] + (XAVS2_PAD + start_y) * i_tmp + XAVS2_PAD - PAD_OFFSET;

    /* -------------------------------------------------------------
     * interpolate horizontal positions: a,b,c;
     * 4 more rows needed for vertical interpolation */

    // SSE x3 Optimization
    /* decoded luma full pel plane  */
    /* interpolating position: a    */
    /*                              */
    /*        |                     */
    /*        |  0   1   2   3      */
    /*    ----+------------------   */
    /*        |                     */
    /*      0 |  *   a   b   c      */
    /*        |                     */
    /*      1 |  d   e   f   g      */
    /*        |                     */
    /*      2 |  h   i   j   k      */
    /*        |                     */
    /*      3 |  n   p   q   r      */
    /*        |                     */
    /*                              */

    /* -------------------------------------------------------------
     * interpolate horizontal positions: a.b,c */
    {
        const int shift_h = 4;   // 往上偏移4行重新插值以并行
        intpl_tmp[0] -= shift_h * i_tmp;
        intpl_tmp[1] -= shift_h * i_tmp;
        intpl_tmp[2] -= shift_h * i_tmp;
        src          -= shift_h * stride;
        if (h->use_fractional_me > 1) {
            p_dst[0] = frm->filtered[INTPL_POS_A] + off_dst - shift_h * stride;  // a
            p_coeffs[0] = INTPL_FILTERS[INTPL_POS_A];         // a

            p_dst[1] = frm->filtered[INTPL_POS_B] + off_dst - shift_h * stride;  // b
            p_coeffs[1] = INTPL_FILTERS[INTPL_POS_B];         // b

            p_dst[2] = frm->filtered[INTPL_POS_C] + off_dst - shift_h * stride;  // c
            p_coeffs[2] = INTPL_FILTERS[INTPL_POS_C];         // c

            g_funcs.intpl_luma_hor_x3(p_dst, stride, intpl_tmp, i_tmp, src, stride, width, height + 4 + shift_h, p_coeffs);
        } else {
            // b
            dst = frm->filtered[INTPL_POS_B] + off_dst - 4 * stride;
            g_funcs.intpl_luma_hor(dst, stride, intpl_tmp[1], i_tmp, src, stride, width, height + 4 + shift_h, INTPL_FILTERS[INTPL_POS_B]);
        }
        src          += shift_h * stride;
        intpl_tmp[0] += shift_h * i_tmp;
        intpl_tmp[1] += shift_h * i_tmp;
        intpl_tmp[2] += shift_h * i_tmp;
    }

    /* -------------------------------------------------------------
     * interpolate vertical positions: d,h,n */
    if (h->use_fractional_me > 1) {
        p_dst[0] = frm->filtered[INTPL_POS_D] + off_dst;  // d
        p_coeffs[0] = INTPL_FILTERS[INTPL_POS_D >> 2];    // d

        p_dst[1] = frm->filtered[INTPL_POS_H] + off_dst;  // h
        p_coeffs[1] = INTPL_FILTERS[INTPL_POS_H >> 2];    // h

        p_dst[2] = frm->filtered[INTPL_POS_N] + off_dst;  // n
        p_coeffs[2] = INTPL_FILTERS[INTPL_POS_N >> 2];    // n

        g_funcs.intpl_luma_ver_x3(p_dst, stride, src, stride, width, height, p_coeffs);
    } else {
        p_dst[1] = frm->filtered[INTPL_POS_H] + off_dst;  // h
        g_funcs.intpl_luma_ver(p_dst[1], stride, src, stride, width, height, INTPL_FILTERS[INTPL_POS_H >> 2]);
    }

    /* -------------------------------------------------------------
     * interpolate tilt positions: [e,f,g; i,j,k; p,q,r] */
    if (h->use_fractional_me > 1) {
        // --- for e,i,p ---
        p_dst[0] = frm->filtered[INTPL_POS_E] + off_dst;  // e
        p_coeffs[0] = INTPL_FILTERS[INTPL_POS_E >> 2];    // e

        p_dst[1] = frm->filtered[INTPL_POS_I] + off_dst;  // i
        p_coeffs[1] = INTPL_FILTERS[INTPL_POS_I >> 2];    // i

        p_dst[2] = frm->filtered[INTPL_POS_P] + off_dst;  // p
        p_coeffs[2] = INTPL_FILTERS[INTPL_POS_P >> 2];    // p

        g_funcs.intpl_luma_ext_x3(p_dst, stride, intpl_tmp[0], i_tmp, width, height, p_coeffs);

        // --- for f,j,q ---
        p_dst[0] = frm->filtered[INTPL_POS_F] + off_dst;  // f
        p_coeffs[0] = INTPL_FILTERS[INTPL_POS_F >> 2];    // f

        p_dst[1] = frm->filtered[INTPL_POS_J] + off_dst;  // j
        p_coeffs[1] = INTPL_FILTERS[INTPL_POS_J >> 2];    // j

        p_dst[2] = frm->filtered[INTPL_POS_Q] + off_dst;  // q
        p_coeffs[2] = INTPL_FILTERS[INTPL_POS_Q >> 2];    // q

        g_funcs.intpl_luma_ext_x3(p_dst, stride, intpl_tmp[1], i_tmp, width, height, p_coeffs);

        // --- for g,k,r ---
        p_dst[0] = frm->filtered[INTPL_POS_G] + off_dst;  // g
        p_coeffs[0] = INTPL_FILTERS[INTPL_POS_G >> 2];    // g

        p_dst[1] = frm->filtered[INTPL_POS_K] + off_dst;  // k
        p_coeffs[1] = INTPL_FILTERS[INTPL_POS_K >> 2];    // k

        p_dst[2] = frm->filtered[INTPL_POS_R] + off_dst;  // r
        p_coeffs[2] = INTPL_FILTERS[INTPL_POS_R >> 2];    // r

        g_funcs.intpl_luma_ext_x3(p_dst, stride, intpl_tmp[2], i_tmp, width, height, p_coeffs);
    } else {
        // j
        dst = frm->filtered[INTPL_POS_J] + off_dst;
        g_funcs.intpl_luma_ext(dst, stride, intpl_tmp[1], i_tmp, width, height, INTPL_FILTERS[INTPL_POS_J >> 2]);
    }

    /* ---------------------------------------------------------------------------
     * expand border for all 15 filtered planes */
    {
        const int padh = XAVS2_PAD - PAD_OFFSET;
        const int padv = XAVS2_PAD - PAD_OFFSET;
        int i;

        width  = frm->i_width[IMG_Y] + PAD_OFFSET * 2;

        /* loop over all 15 filtered planes */
        for (i = 1; i < 16; i++) {
            pel_t *pix = frm->filtered[i];
            if (pix != NULL) {
                pix += start_y * stride - PAD_OFFSET;
                plane_expand_border(pix, stride, width, height, padh, padv, b_start, b_end);
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void interpolate_lcu_row(xavs2_t *h, xavs2_frame_t* frm, int i_lcu_y)
{
    int b_start = !i_lcu_y;
    int b_end   = i_lcu_y == h->i_height_in_lcu - 1;
    int y_start = (i_lcu_y + 0) << h->i_lcu_level;
    int y_end   = (i_lcu_y + 1) << h->i_lcu_level;
    int height;
    slice_t *slice = h->slices[h->i_slice_index];

    /* 有效插值像素区域的起始和结束行号 */
    if (b_start) {
        y_start -= PAD_OFFSET;
    } else {
        y_start -= MC_OFFSET;
    }
    if (b_end) {
        y_end = h->i_height + MC_OFFSET - PAD_OFFSET;
    } else {
        y_end -= MC_OFFSET;
    }

    /* 多slice时减少冗余运算 */
    if (h->param->slice_num > 1 && !b_start && !b_end) {
        if (slice->i_first_lcu_y == i_lcu_y) {
            /* Slice的上边界 */
            y_start += (MC_OFFSET + PAD_OFFSET);
        }
        if (slice->i_last_lcu_y == i_lcu_y) {
            /* Slice的下边界 */
            y_end += PAD_OFFSET;
        }
    }

    height = y_end - y_start;
    // xavs2_log(NULL, XAVS2_LOG_DEBUG, "Intpl POC [%3d], Slice %2d, Row %2d, [%3d, %3d)\n",
    //           h->fenc->i_frame, h->i_slice_index, i_lcu_y, y_start, y_end);
    interpolate_sample_rows(h, frm, y_start, height, b_start, b_end);
}


/**
 * ===========================================================================
 * interpolating for chroma
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * predict one component of a chroma block
 */
void mc_chroma(pel_t *p_pred_u, pel_t *p_pred_v, int i_pred,
               int pix_quad_x, int pix_quad_y, int width, int height,
               const xavs2_frame_t *p_ref_frm)
{
    int posx = pix_quad_x & 7;
    int posy = pix_quad_y & 7;
    int i_src = p_ref_frm->i_stride[IMG_U];
    pel_t *p_src_u = p_ref_frm->planes[IMG_U];
    pel_t *p_src_v = p_ref_frm->planes[IMG_V];
    int src_offset = (pix_quad_y >> 3) * i_src + (pix_quad_x >> 3);

    p_src_u += src_offset;
    p_src_v += src_offset;

    if (posy == 0 && posx == 0) {
        if (width != 2 && width != 6 && height != 2 && height != 6) {
            g_funcs.pixf.copy_pp[PART_INDEX(width, height)](p_pred_u, i_pred, p_src_u, i_src);
            g_funcs.pixf.copy_pp[PART_INDEX(width, height)](p_pred_v, i_pred, p_src_v, i_src);
        } else {
            g_funcs.align_copy(p_pred_u, i_pred, p_src_u, i_src, width, height);
            g_funcs.align_copy(p_pred_v, i_pred, p_src_v, i_src, width, height);
        }
    } else if (posy == 0) {
        g_funcs.intpl_chroma_block_hor(p_pred_u, i_pred, p_src_u, i_src, width, height, INTPL_FILTERS_C[posx]);
        g_funcs.intpl_chroma_block_hor(p_pred_v, i_pred, p_src_v, i_src, width, height, INTPL_FILTERS_C[posx]);
    } else if (posx == 0) {
        g_funcs.intpl_chroma_block_ver(p_pred_u, i_pred, p_src_u, i_src, width, height, INTPL_FILTERS_C[posy]);
        g_funcs.intpl_chroma_block_ver(p_pred_v, i_pred, p_src_v, i_src, width, height, INTPL_FILTERS_C[posy]);
    } else {
        g_funcs.intpl_chroma_block_ext(p_pred_u, i_pred, p_src_u, i_src, width, height, INTPL_FILTERS_C[posx], INTPL_FILTERS_C[posy]);
        g_funcs.intpl_chroma_block_ext(p_pred_v, i_pred, p_src_v, i_src, width, height, INTPL_FILTERS_C[posx], INTPL_FILTERS_C[posy]);
    }
}


/**
 * ===========================================================================
 * low resolution (down sampling)
 * ===========================================================================
 */


/* ---------------------------------------------------------------------------
 */
static void lowres_filter_core_c(pel_t *src, int i_src, pel_t *dst, int i_dst, int width, int height)
{
#define FILTER(a,b,c,d) ((((a+b+1)>>1) + ((c+d+1)>>1) + 1) >> 1)
    int i_src2 = i_src << 1;    // stride of 2 src lines
    int x, y;
    pel_t *dwn;

    for (y = 0; y < height; y++) {
        dwn = src + i_src;      // point to down line of src
        for (x = 0; x < width; x++) {
            dst[x] = FILTER(src[2 * x], dwn[2 * x], src[2 * x + 1], dwn[2 * x + 1]);
        }
        src += i_src2;
        dst += i_dst;
    }
#undef FILTER
}

/**
 * ===========================================================================
 * interface function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * global function set initial
 */
void xavs2_mem_oper_init(uint32_t cpuid, intrinsic_func_t *pf)
{
    pf->fast_memcpy     = memcpy;
    pf->memcpy_aligned  = memcpy;
    pf->fast_memset     = memset;
    pf->fast_memzero    = memzero_aligned_c;
    pf->memzero_aligned = memzero_aligned_c;
    pf->mem_repeat_i    = mem_repeat_i_c;
    pf->mem_repeat_p    = memset;
    pf->lowres_filter   = lowres_filter_core_c;

#if ARCH_X86_64
    pf->mem_repeat_i    = mem_repeat_8i_c;  // x64架构下，减少循环次数同时使用64位打包赋值
#endif

#if HAVE_MMX
    if (cpuid & XAVS2_CPU_MMX) {
        pf->fast_memcpy     = xavs2_fast_memcpy_mmx;
        pf->memcpy_aligned  = xavs2_memcpy_aligned_mmx;
        pf->fast_memset     = xavs2_fast_memset_mmx;
        pf->fast_memzero    = xavs2_fast_memzero_mmx;
        pf->memzero_aligned = xavs2_fast_memzero_mmx;
    }
    if (cpuid & XAVS2_CPU_MMX2) {
        pf->lowres_filter = xavs2_lowres_filter_core_mmx2;
    }

    if (cpuid & XAVS2_CPU_SSE) {
        // pf->memcpy_aligned  = xavs2_memcpy_aligned_sse;
        // pf->memzero_aligned = xavs2_memzero_aligned_sse;
    }

    if (cpuid & XAVS2_CPU_SSE2) {
        pf->memzero_aligned = xavs2_memzero_aligned_c_sse2;
        // pf->memcpy_aligned  = xavs2_memcpy_aligned_c_sse2;
        pf->lowres_filter  = xavs2_lowres_filter_core_sse2;
        // pf->mem_repeat_i  = xavs2_mem_repeat_i_c_sse2;  // TODO: 比C版本慢，禁用
    }

    if (cpuid & XAVS2_CPU_SSSE3) {
        pf->lowres_filter = xavs2_lowres_filter_core_ssse3;
    }

    if (cpuid & XAVS2_CPU_AVX2) {
        pf->memzero_aligned = xavs2_memzero_aligned_c_avx;
        // pf->mem_repeat_i    = xavs2_mem_repeat_i_c_avx;  // TODO: 比C版本慢，禁用
        pf->lowres_filter   = xavs2_lowres_filter_core_avx;
    }
#else
    UNUSED_PARAMETER(cpuid);
#endif
}

/* ---------------------------------------------------------------------------
 */
void xavs2_mc_init(uint32_t cpuid, intrinsic_func_t *pf)
{
    /* align copy */
    pf->align_copy = mc_copy_c;

    /* plane copy */
    pf->plane_copy = plane_copy_c;
    pf->plane_copy_deinterleave = plane_copy_deinterleave_c;

    /* interpolate */
    pf->intpl_luma_hor = intpl_luma_hor_c;
    pf->intpl_luma_ver = intpl_luma_ver_c;
    pf->intpl_luma_ext = intpl_luma_ext_c;

    pf->intpl_luma_ver_x3 = intpl_luma_ver_x3_c;
    pf->intpl_luma_hor_x3 = intpl_luma_hor_x3_c;
    pf->intpl_luma_ext_x3 = intpl_luma_ext_x3_c;

    pf->intpl_luma_block_hor   = intpl_luma_block_hor_c;
    pf->intpl_luma_block_ver   = intpl_luma_block_ver_c;
    pf->intpl_luma_block_ext   = intpl_luma_block_ext_c;
    pf->intpl_chroma_block_hor = intpl_chroma_block_hor_c;
    pf->intpl_chroma_block_ver = intpl_chroma_block_ver_c;
    pf->intpl_chroma_block_ext = intpl_chroma_block_ext_c;

#if HAVE_MMX
    if (cpuid & XAVS2_CPU_MMX2) {
        pf->plane_copy = plane_copy_mmx2;
        pf->plane_copy_deinterleave = xavs2_plane_copy_deinterleave_mmx;
    }

    if (cpuid & XAVS2_CPU_SSE42) {
        pf->intpl_luma_hor = intpl_luma_hor_sse128;
        pf->intpl_luma_ver = intpl_luma_ver_sse128;
        pf->intpl_luma_ext = intpl_luma_ext_sse128;

        pf->intpl_luma_hor_x3 = intpl_luma_hor_x3_sse128;
        pf->intpl_luma_ver_x3 = intpl_luma_ver_x3_sse128;
        pf->intpl_luma_ext_x3 = intpl_luma_ext_x3_sse128;

        pf->intpl_luma_block_hor   = intpl_luma_block_hor_sse128;
        pf->intpl_luma_block_ver   = intpl_luma_block_ver_sse128;
        pf->intpl_luma_block_ext   = intpl_luma_block_ext_sse128;
        pf->intpl_chroma_block_hor = intpl_chroma_block_hor_sse128;
        pf->intpl_chroma_block_ver = intpl_chroma_block_ver_sse128;
        pf->intpl_chroma_block_ext = intpl_chroma_block_ext_sse128;
    }

    if (cpuid & XAVS2_CPU_AVX2) {
        pf->intpl_luma_hor = intpl_luma_hor_avx2;
        pf->intpl_luma_ver = intpl_luma_ver_avx2;
        pf->intpl_luma_ext = intpl_luma_ext_avx2;

        pf->intpl_luma_ver_x3 = intpl_luma_ver_x3_avx2;
        pf->intpl_luma_hor_x3 = intpl_luma_hor_x3_avx2;
        pf->intpl_luma_ext_x3 = intpl_luma_ext_x3_avx2;

        pf->intpl_luma_block_hor = intpl_luma_block_hor_avx2;
        pf->intpl_luma_block_ver = intpl_luma_block_ver_avx2;
        pf->intpl_luma_block_ext = intpl_luma_block_ext_avx2;

        pf->intpl_chroma_block_ver = intpl_chroma_block_ver_avx2;
        pf->intpl_chroma_block_hor = intpl_chroma_block_hor_avx2;
        pf->intpl_chroma_block_ext = intpl_chroma_block_ext_avx2;
    }
#else
    UNUSED_PARAMETER(cpuid);
#endif
}
