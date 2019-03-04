/*
 * intra.c
 *
 * Description of this file:
 *    Intra prediction functions definition of the xavs2 library
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
#include "block_info.h"
#include "cudata.h"
#include "cpu.h"

#if HAVE_MMX
#include "vec/intrinsic.h"
#endif


// ---------------------------------------------------------------------------
// disable warning
#if defined(_MSC_VER) || defined(__ICL)
#pragma warning(disable: 4100)  // unreferenced formal parameter
#endif


/**
 * ===========================================================================
 * local tables
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static const int16_t tab_log2size[MAX_CU_SIZE + 1] = {
    -1, -1, -1, -1,  2, -1, -1, -1,
     3, -1, -1, -1, -1, -1, -1, -1,
     4, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
     5, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    6
};

/* ---------------------------------------------------------------------------
 */
static const char tab_auc_dir_dx[NUM_INTRA_MODE] = {
      0, 0,  0,
     11, 2, 11, 1,  8, 1,  4, 1,  1,                    /* X  */
      0,
      1, 1,  4, 1,  8, 1, 11, 2, 11, 4, 8,              /* XY */
      0,
      8, 4, 11, 2, 11, 1,  8, 1                         /* Y  */
};

/* ---------------------------------------------------------------------------
 */
static const char tab_auc_dir_dy[NUM_INTRA_MODE] = {
      0,  0,  0,
     -4, -1, -8, -1, -11, -2, -11, -4, -8,              /* X  */
      0,
      8,  4, 11,  2,  11,  1,   8,  1,  4,  1,  1,      /* XY */
      0,
     -1, -1, -4, -1,  -8, -1, -11, -2                   /* Y  */
};

/* ---------------------------------------------------------------------------
 */
static const char tab_auc_dir_dxdy[2][NUM_INTRA_MODE][2] = {
    {
        // dx/dy
        { 0,0}, {0,0}, { 0,0},
        {11,2}, {2,0}, {11,3}, {1,0}, {93,7}, {1,1}, {93,8}, {1,2}, { 1,3},                 /* X  */
        { 0,0},
        { 1,3}, {1,2}, {93,8}, {1,1}, {93,7}, {1,0}, {11,3}, {2,0}, {11,2}, {4,0}, {8,0},   /* XY */
        { 0,0},
        { 8,0}, {4,0}, {11,2}, {2,0}, {11,3}, {1,0}, {93,7}, {1,1},                         /* Y  */
    }, {
        // dy/dx
        { 0,0}, {0,0}, { 0,0},
        {93,8}, {1,1}, {93,7}, {1,0}, {11,3}, {2,0}, {11,2}, {4,0}, { 8,0},                 /* X  */
        { 0,0},
        { 8,0}, {4,0}, {11,2}, {2,0}, {11,3}, {1,0}, {93,7}, {1,1}, {93,8}, {1,2}, {1,3},   /* XY */
        { 0,0},
        { 1,3}, {1,2}, {93,8}, {1,1}, {93,7}, {1,0}, {11,3}, {2,0}                          /* Y  */
    }
};


/**
 * ===========================================================================
 * local function definition
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ver_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    pel_t *p_src = src + 1;
    int y;

    for (y = 0; y < bsy; y++) {
        g_funcs.fast_memcpy(dst, p_src, bsx * sizeof(pel_t));
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_hor_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    pel_t *p_src = src - 1;

    while (bsy-- != 0) {
        g_funcs.mem_repeat_p(dst, *p_src--, bsx);
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 * NOTE: dir_mode = (bAboveAvail << 8) + (bLeftAvail)
 */
static void intra_pred_dc_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int avail_top  = dir_mode >> 8;
    int avail_left = dir_mode & 0xFF;
    int sum_left   = 0;
    int sum_above  = 0;
    int dc_value;
    int x, y;
    pel_t *p_src;

    p_src = src - 1;
    for (y = 0; y < bsy; y++) {
        sum_left += p_src[-y];
    }

    p_src = src + 1;
    for (x = 0; x < bsx; x++) {
        sum_above += p_src[x];
    }

    if (avail_left && avail_top) {
        x = bsx + bsy;
        dc_value = ((sum_left + sum_above + (x >> 1)) * (512 / x)) >> 9;
    } else if (avail_left) {
        dc_value = (sum_left  + (bsy >> 1)) >> xavs2_log2u(bsy);
    } else if (avail_top) {
        dc_value = (sum_above + (bsx >> 1)) >> xavs2_log2u(bsx);
    } else {
        dc_value = g_dc_value;
    }

    for (y = 0; y < bsy; y++) {
        g_funcs.mem_repeat_p(dst, (pel_t)dc_value, bsx);
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_plane_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    /*                 size in bits:       2   3   4   5   6 */
    static const int ib_mult [8] = { 0, 0, 13, 17,  5, 11, 23, 0 };
    static const int ib_shift[8] = { 0, 0,  7, 10, 11, 15, 19, 0 };
    const int mult_h  = ib_mult [tab_log2size[bsx]];
    const int mult_v  = ib_mult [tab_log2size[bsy]];
    const int shift_h = ib_shift[tab_log2size[bsx]];
    const int shift_v = ib_shift[tab_log2size[bsy]];
    const int W2   = bsx >> 1;              /* half block width */
    const int H2   = bsy >> 1;              /* half block height */
    const int vmax = (1 << g_bit_depth) - 1;  /* max value of pixel */
    int H = 0;
    int V = 0;
    int a, b, c;
    int x, y;
    pel_t *p_src;

    /* calculate H and V */
    p_src = src + W2;
    for (x = 1; x < W2 + 1; x++) {
        H += x * (p_src[x] - p_src[-x]);
    }
    p_src = src - H2;
    for (y = 1; y < H2 + 1; y++) {
        V += y * (p_src[-y] - p_src[y]);
    }

    a  = (src[-bsy] + src[bsx]) << 4;
    b  = ((H << 5) * mult_h + (1 << (shift_h - 1))) >> shift_h;
    c  = ((V << 5) * mult_v + (1 << (shift_v - 1))) >> shift_v;
    a += 16 - b * (W2 - 1) - c * (H2 - 1);

    for (y = 0; y < bsy; y++) {
        int pix = a;
        for (x = 0; x < bsx; x++) {
            dst[x] = (pel_t)XAVS2_CLIP3(0, vmax, pix >> 5);
            pix   += b;
        }
        dst += i_dst;
        a   += c;
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_bilinear_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    itr_t pTop[MAX_CU_SIZE], pLeft[MAX_CU_SIZE], pT[MAX_CU_SIZE], pL[MAX_CU_SIZE], wy[MAX_CU_SIZE];
    int shift_x  = tab_log2size[bsx];
    int shift_y  = tab_log2size[bsy];
    int shift    = XAVS2_MIN(shift_x, shift_y);
    int shift_xy = shift_x + shift_y + 1;
    int offset   = 1 << (shift_x + shift_y);
    int vmax     = max_pel_value;    // max value of pixel
    int a, b, c, t, wxy, temp;
    int predx, val;
    int x, y;
    pel_t *p_src;

    p_src = src + 1;
    for (x = 0; x < bsx; x++) {
        pTop[x] = p_src[x];
    }
    p_src = src - 1;
    for (y = 0; y < bsy; y++) {
        pLeft[y] = p_src[-y];
    }

    a = pTop [bsx - 1];
    b = pLeft[bsy - 1];
    c = (bsx == bsy) ? (a + b + 1) >> 1 : (((a << shift_x) + (b << shift_y)) * 13 + (1 << (shift + 5))) >> (shift + 6);
    t = (c << 1) - a - b;

    for (x = 0; x < bsx; x++) {
        pT  [x]   = (itr_t)(b - pTop[x]);
        pTop[x] <<= shift_y;
    }

    temp = 0;
    for (y = 0; y < bsy; y++) {
        pL   [y]   = (itr_t)(a - pLeft[y]);
        pLeft[y] <<= shift_x;
        wy   [y]   = (itr_t)temp;
        temp      += t;
    }

    for (y = 0; y < bsy; y++) {
        predx = pLeft[y];
        wxy   = -wy[y];
        for (x = 0; x < bsx; x++) {
            predx   += pL[y];
            wxy     += wy[y];
            pTop[x] += pT[x];
            val      = ((predx << shift_y) + (pTop[x] << shift_x) + wxy + offset) >> shift_xy;
            dst[x]   = (pel_t)XAVS2_CLIP3(0, vmax, val);
        }
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE int get_context_pixel(int dir_mode, int xy_flag, int temp_d, int *offset)
{
    const int mult  = tab_auc_dir_dxdy[xy_flag][dir_mode][0];
    const int shift = tab_auc_dir_dxdy[xy_flag][dir_mode][1];
    int temp_dn;

    temp_d *= mult;
    temp_dn = temp_d >> shift;
    *offset = ((temp_d << 5) >> shift) - (temp_dn << 5);

    return temp_dn;
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_x_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int f0, f1, f2, f3;
    int i, j;
    int iX;

    for (j = 0; j < bsy; j++) {
        iX = get_context_pixel(dir_mode, 0, j + 1, &f3);
        f0 = 32 - f3;
        f1 = 64 - f3;
        f2 = 32 + f3;

        for (i = 0; i < bsx; i++) {
            dst[i] = (pel_t)((src[iX    ] * f0 +
                              src[iX + 1] * f1 +
                              src[iX + 2] * f2 +
                              src[iX + 3] * f3 + 64) >> 7);
            iX++;
        }

        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_y_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int offsets[64];
    int xsteps[64];
    int offset;
    int i, j;
    int iY;

    for (i = 0; i < bsx; i++) {
        xsteps[i] = get_context_pixel(dir_mode, 1, i + 1, &offsets[i]);
    }

    for (j = 0; j < bsy; j++) {
        for (i = 0; i < bsx; i++) {
            iY     = j + xsteps[i];
            offset = offsets[i];
            dst[i] = (pel_t)((src[-iY    ] * (32 - offset) +
                              src[-iY - 1] * (64 - offset) +
                              src[-iY - 2] * (32 + offset) +
                              src[-iY - 3] * (     offset) + 64) >> 7);
        }
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_xy_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(int xoffsets[64]);
    ALIGN16(int xsteps[64]);
    int i, j, iXx, iYy;
    int offsetx, offsety;

    for (i = 0; i < bsx; i++) {
        xsteps[i] = get_context_pixel(dir_mode, 1, i + 1, &xoffsets[i]);
    }

    for (j = 0; j < bsy; j++) {
        iXx = -get_context_pixel(dir_mode, 0, j + 1, &offsetx);

        for (i = 0; i < bsx; i++) {
            iYy = j - xsteps[i];

            if (iYy <= -1) {
                dst[i] = (pel_t)((src[ iXx + 2] * (32 - offsetx) +
                                  src[ iXx + 1] * (64 - offsetx) +
                                  src[ iXx    ] * (32 + offsetx) +
                                  src[ iXx - 1] * (     offsetx) + 64) >> 7);
            } else {
                offsety = xoffsets[i];
                dst[i] = (pel_t)((src[-iYy - 2] * (32 - offsety) +
                                  src[-iYy - 1] * (64 - offsety) +
                                  src[-iYy    ] * (32 + offsety) +
                                  src[-iYy + 1] * (     offsety) + 64) >> 7);
            }
            iXx++;
        }
        dst += i_dst;
    }
}


/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_x_3_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[(64 + 176) << 2]);
    int line_size = bsx + (bsy >> 2) * 11 - 1;

    int aligned_line_size = 64 + 176;
    int i_dst4 = i_dst << 2;
    int i;
    pel_t *pfirst[4];

    pfirst[0] = first_line;
    pfirst[1] = pfirst[0] + aligned_line_size;
    pfirst[2] = pfirst[1] + aligned_line_size;
    pfirst[3] = pfirst[2] + aligned_line_size;

    for (i = 0; i < line_size; i++, src++) {
        pfirst[0][i] = (pel_t)((    src[2] + 5 * src[3] + 7 * src[4] + 3 * src[5] + 8) >> 4);
        pfirst[1][i] = (pel_t)((    src[5] + 3 * src[6] + 3 * src[7] +     src[8] + 4) >> 3);
        pfirst[2][i] = (pel_t)((3 * src[8] + 7 * src[9] + 5 * src[10] +     src[11] + 8) >> 4);
        pfirst[3][i] = (pel_t)((    src[11] + 2 * src[12] +   src[13] + 0 * src[14] + 2) >> 2);
    }

    bsy >>= 2;
    for (i = 0; i < bsy; i++) {
        memcpy(dst            , pfirst[0] + i * 11, bsx * sizeof(pel_t));
        memcpy(dst +     i_dst, pfirst[1] + i * 11, bsx * sizeof(pel_t));
        memcpy(dst + 2 * i_dst, pfirst[2] + i * 11, bsx * sizeof(pel_t));
        memcpy(dst + 3 * i_dst, pfirst[3] + i * 11, bsx * sizeof(pel_t));
        dst += i_dst4;
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_x_4_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[64 + 128]);
    int line_size = bsx + ((bsy - 1) << 1);
    int iHeight2 = bsy << 1;
    int i;

    src += 3;
    for (i = 0; i < line_size; i++, src++) {
        first_line[i] = (pel_t)((src[-1] + src[0] * 2 + src[1] + 2) >> 2);
    }

    for (i = 0; i < iHeight2; i += 2) {
        memcpy(dst, first_line + i, bsx * sizeof(pel_t));
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_x_5_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;

    if (((bsy > 4) && (bsx > 8))) {
        ALIGN16(pel_t first_line[(64 + 80) << 3]);
        int line_size = bsx + (((bsy - 8) * 11) >> 3);
        int aligned_line_size = ((line_size + 15) >> 4) << 4;
        pel_t *pfirst[8];

        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        pel_t *dst5 = dst4 + i_dst;
        pel_t *dst6 = dst5 + i_dst;
        pel_t *dst7 = dst6 + i_dst;
        pel_t *dst8 = dst7 + i_dst;

        pfirst[0] = first_line;
        pfirst[1] = pfirst[0] + aligned_line_size;
        pfirst[2] = pfirst[1] + aligned_line_size;
        pfirst[3] = pfirst[2] + aligned_line_size;
        pfirst[4] = pfirst[3] + aligned_line_size;
        pfirst[5] = pfirst[4] + aligned_line_size;
        pfirst[6] = pfirst[5] + aligned_line_size;
        pfirst[7] = pfirst[6] + aligned_line_size;

        for (i = 0; i < line_size; src++, i++) {
            pfirst[0][i] = (pel_t)((5 * src[1] + 13 * src[2] + 11 * src[3] + 3 * src[4] + 16) >> 5);
            pfirst[1][i] = (pel_t)((    src[2] +  5 * src[3] +  7 * src[4] + 3 * src[5] + 8) >> 4);
            pfirst[2][i] = (pel_t)((7 * src[4] + 15 * src[5] +  9 * src[6] +     src[7] + 16) >> 5);
            pfirst[3][i] = (pel_t)((    src[5] +  3 * src[6] +  3 * src[7] +     src[8] + 4) >> 3);

            pfirst[4][i] = (pel_t)((     src[6] +  9 * src[7]  + 15 * src[8]  +  7 * src[9]  + 16) >> 5);
            pfirst[5][i] = (pel_t)(( 3 * src[8] +  7 * src[9]  +  5 * src[10] +      src[11] +  8) >> 4);
            pfirst[6][i] = (pel_t)(( 3 * src[9] + 11 * src[10] + 13 * src[11] +  5 * src[12] + 16) >> 5);
            pfirst[7][i] = (pel_t)((    src[11] +  2 * src[12] +      src[13]                 + 2) >> 2);
        }

        bsy  >>= 3;
        for (i = 0; i < bsy; i++) {
            memcpy(dst1, pfirst[0] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst2, pfirst[1] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst3, pfirst[2] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst4, pfirst[3] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst5, pfirst[4] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst6, pfirst[5] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst7, pfirst[6] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst8, pfirst[7] + i * 11, bsx * sizeof(pel_t));

            dst1 = dst8 + i_dst;
            dst2 = dst1 + i_dst;
            dst3 = dst2 + i_dst;
            dst4 = dst3 + i_dst;
            dst5 = dst4 + i_dst;
            dst6 = dst5 + i_dst;
            dst7 = dst6 + i_dst;
            dst8 = dst7 + i_dst;
        }
    } else if (bsx == 16) {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;

        for (i = 0; i < bsx; i++, src++) {
            dst1[i]  = (pel_t)((5 * src[1] + 13 * src[2] + 11 * src[3] + 3 * src[4] + 16) >> 5);
            dst2[i]  = (pel_t)((    src[2] +  5 * src[3] +  7 * src[4] + 3 * src[5] + 8) >> 4);
            dst3[i]  = (pel_t)((7 * src[4] + 15 * src[5] +  9 * src[6] +     src[7] + 16) >> 5);
            dst4[i]  = (pel_t)((    src[5] +  3 * src[6] +  3 * src[7] +     src[8] + 4) >> 3);
        }
    } else if (bsx == 8) {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        pel_t *dst5 = dst4 + i_dst;
        pel_t *dst6 = dst5 + i_dst;
        pel_t *dst7 = dst6 + i_dst;
        pel_t *dst8 = dst7 + i_dst;

        for (i = 0; i < 8; src++, i++) {
            dst1[i]  = (pel_t)((5 * src[1] + 13 * src[2] + 11 * src[3] + 3 * src[4] + 16) >> 5);
            dst2[i]  = (pel_t)((    src[2] +  5 * src[3] +  7 * src[4] + 3 * src[5] + 8) >> 4);
            dst3[i]  = (pel_t)((7 * src[4] + 15 * src[5] +  9 * src[6] +     src[7] + 16) >> 5);
            dst4[i]  = (pel_t)((    src[5] +  3 * src[6] +  3 * src[7] +     src[8] + 4) >> 3);

            dst5[i] = (pel_t)((     src[6] +  9 * src[7]  + 15 * src[8]  +  7 * src[9]  + 16) >> 5);
            dst6[i] = (pel_t)(( 3 * src[8] +  7 * src[9]  +  5 * src[10] +      src[11] + 8) >> 4);
            dst7[i] = (pel_t)(( 3 * src[9] + 11 * src[10] + 13 * src[11] +  5 * src[12] + 16) >> 5);
            dst8[i] = (pel_t)((    src[11] +  2 * src[12] +      src[13]                 + 2) >> 2);
        }
        if (bsy == 32) {
            //src -> 8,src[8] -> 16
            pel_t pad1 = src[8];
            dst1 = dst8 + i_dst;
            int j;
            for (j = 0; j < 24; j++) {
                for (i = 0; i < 8; i++) {
                    dst1[i] = pad1;
                }
                dst1 += i_dst;
            }

            dst1 = dst8 + i_dst;
            dst2 = dst1 + i_dst;
            dst3 = dst2 + i_dst;

            src += 4;
            dst1[0] = (pel_t)((5 * src[0] + 13 * src[1] + 11 * src[2] + 3 * src[3] + 16) >> 5);
            dst1[1] = (pel_t)((5 * src[1] + 13 * src[2] + 11 * src[3] + 3 * src[4] + 16) >> 5);
            dst1[2] = (pel_t)((5 * src[2] + 13 * src[3] + 11 * src[4] + 3 * src[5] + 16) >> 5);
            dst1[3] = (pel_t)((5 * src[3] + 13 * src[4] + 11 * src[5] + 3 * src[6] + 16) >> 5);
            dst2[0] = (pel_t)((src[1] + 5 * src[2] + 7 * src[3] + 3 * src[4] + 8) >> 4);
            dst2[1] = (pel_t)((src[2] + 5 * src[3] + 7 * src[4] + 3 * src[5] + 8) >> 4);
            dst2[2] = (pel_t)((src[3] + 5 * src[4] + 7 * src[5] + 3 * src[6] + 8) >> 4);
            dst3[0] = (pel_t)((7 * src[3] + 15 * src[4] +  9 * src[5] +     src[6] + 16) >> 5);
        }
    } else {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;

        for (i = 0; i < 4; i++, src++) {
            dst1[i]  = (pel_t)((5 * src[1] + 13 * src[2] + 11 * src[3] + 3 * src[4] + 16) >> 5);
            dst2[i]  = (pel_t)((    src[2] +  5 * src[3] +  7 * src[4] + 3 * src[5] + 8) >> 4);
            dst3[i]  = (pel_t)((7 * src[4] + 15 * src[5] +  9 * src[6] +     src[7] + 16) >> 5);
            dst4[i]  = (pel_t)((    src[5] +  3 * src[6] +  3 * src[7] +     src[8] + 4) >> 3);
        }
        if (bsy == 16) {
            pel_t *dst5 = dst4 + i_dst;

            src += 4;
            pel_t pad1 = src[0];

            int j;
            for (j = 0; j < 12; j++) {
                for (i = 0; i < 4; i++) {
                    dst5[i] = pad1;
                }
                dst5 += i_dst;
            }
            dst5 = dst4 + i_dst;
            dst5[0] = (pel_t)((src[-2] + 9 * src[-1] + 15 * src[0] + 7 * src[1] + 16) >> 5);
            dst5[1] = (pel_t)((src[-1] + 9 * src[ 0] + 15 * src[1] + 7 * src[2] + 16) >> 5);
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_x_6_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[64 + 64]);
    int line_size = bsx + bsy - 1;
    int i;

    for (i = 0; i < line_size; i++, src++) {
        first_line[i] = (pel_t)((src[1] + (src[2] << 1) + src[3] + 2) >> 2);
    }

    for (i = 0; i < bsy; i++) {
        memcpy(dst, first_line + i, bsx * sizeof(pel_t));
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_x_7_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;
    pel_t *dst1 = dst;
    pel_t *dst2 = dst1 + i_dst;
    pel_t *dst3 = dst2 + i_dst;
    pel_t *dst4 = dst3 + i_dst;
    if (bsy == 4) {
        for (i = 0; i < bsx; src++, i++) {
            dst1[i] = (pel_t)((src[0] *  9 + src[1] * 41 + src[2] * 55 + src[3] * 23 + 64) >> 7);
            dst2[i] = (pel_t)((src[1] *  9 + src[2] * 25 + src[3] * 23 + src[4] *  7 + 32) >> 6);
            dst3[i] = (pel_t)((src[2] * 27 + src[3] * 59 + src[4] * 37 + src[5] *  5 + 64) >> 7);
            dst4[i] = (pel_t)((src[2] *  3 + src[3] * 35 + src[4] * 61 + src[5] * 29 + 64) >> 7);
        }
    } else if (bsy == 8) {
        pel_t *dst5 = dst4 + i_dst;
        pel_t *dst6 = dst5 + i_dst;
        pel_t *dst7 = dst6 + i_dst;
        pel_t *dst8 = dst7 + i_dst;
        for (i = 0; i < bsx; src++, i++) {
            dst1[i] = (pel_t)((src[0] *  9 + src[1] * 41 + src[2] * 55 + src[3] * 23 + 64) >> 7);
            dst2[i] = (pel_t)((src[1] *  9 + src[2] * 25 + src[3] * 23 + src[4] *  7 + 32) >> 6);
            dst3[i] = (pel_t)((src[2] * 27 + src[3] * 59 + src[4] * 37 + src[5] *  5 + 64) >> 7);
            dst4[i] = (pel_t)((src[2] *  3 + src[3] * 35 + src[4] * 61 + src[5] * 29 + 64) >> 7);
            dst5[i] = (pel_t)((src[3] *  3 + src[4] * 11 + src[5] * 13 + src[6] *  5 + 16) >> 5);
            dst6[i] = (pel_t)((src[4] * 21 + src[5] * 53 + src[6] * 43 + src[7] * 11 + 64) >> 7);
            dst7[i] = (pel_t)((src[5] * 15 + src[6] * 31 + src[7] * 17 + src[8] + 32)      >> 6);
            dst8[i] = (pel_t)((src[5] *  3 + src[6] * 19 + src[7] * 29 + src[8] * 13 + 32) >> 6);
        }
    } else {
        intra_pred_ang_x_c(src, dst, i_dst, dir_mode, bsx, bsy);
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_x_8_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[2 * (64 + 32)]);
    int line_size = bsx + (bsy >> 1) - 1;
    int aligned_line_size = ((line_size + 15) >> 4) << 4;
    int i_dst2 = i_dst << 1;
    int i;
    pel_t *pfirst[2];

    pfirst[0] = first_line;
    pfirst[1] = first_line + aligned_line_size;
    for (i = 0; i < line_size; i++, src++) {
        pfirst[0][i] = (pel_t)((src[0] + (src[1] + src[2]) * 3 + src[3] + 4) >> 3);
        pfirst[1][i] = (pel_t)((src[1] + (src[2] << 1)         + src[3] + 2) >> 2);
    }

    bsy >>= 1;
    for (i = 0; i < bsy; i++) {
        memcpy(dst        , pfirst[0] + i, bsx * sizeof(pel_t));
        memcpy(dst + i_dst, pfirst[1] + i, bsx * sizeof(pel_t));
        dst += i_dst2;
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_x_9_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    if (bsy > 8) {
        intra_pred_ang_x_c(src, dst, i_dst, dir_mode, bsx, bsy);
        /*
        ALIGN16(pel_t first_line[(64 + 32) * 11]);
        int line_size = bsx + (bsy * 93 >> 8) - 1;
        int real_size = XAVS2_MIN(line_size, bsx * 2);
        int aligned_line_size = ((line_size + 31) >> 5) << 5;
        int i_dst11 = i_dst * 11;
        int i;
        pel_t pad1, pad2, pad3, pad4, pad5, pad6, pad7, pad8, pad9, pad10, pad11;
        pel_t *pfirst[11];

        pfirst[0] = first_line;
        pfirst[1] = pfirst[0] + aligned_line_size;
        pfirst[2] = pfirst[1] + aligned_line_size;
        pfirst[3] = pfirst[2] + aligned_line_size;
        pfirst[4] = pfirst[3] + aligned_line_size;
        pfirst[5] = pfirst[4] + aligned_line_size;
        pfirst[6] = pfirst[5] + aligned_line_size;
        pfirst[7] = pfirst[6] + aligned_line_size;
        pfirst[8] = pfirst[7] + aligned_line_size;
        pfirst[9] = pfirst[8] + aligned_line_size;
        pfirst[10] = pfirst[9] + aligned_line_size;
        for (i = 0; i < real_size; i++, src++) {
            pfirst[0][i] = (pel_t)((21 * src[0] + 53 * src[1] + 43 * src[2] + 11 * src[3] + 64) >> 7);
            pfirst[1][i] = (pel_t)((9 * src[0] + 41 * src[1] + 55 * src[2] + 23 * src[3] + 64) >> 7);
            pfirst[2][i] = (pel_t)((15 * src[1] + 31 * src[2] + 17 * src[3] + 1 * src[4] + 32) >> 6);
            pfirst[3][i] = (pel_t)((9 * src[1] + 25 * src[2] + 23 * src[3] + 7 * src[4] + 32) >> 6);
            pfirst[4][i] = (pel_t)((3 * src[1] + 19 * src[2] + 29 * src[3] + 13 * src[4] + 32) >> 6);
            pfirst[5][i] = (pel_t)((27 * src[2] + 59 * src[3] + 37 * src[4] + 5 * src[5] + 64) >> 7);
            pfirst[6][i] = (pel_t)((15 * src[2] + 47 * src[3] + 49 * src[4] + 17 * src[5] + 64) >> 7);
            pfirst[7][i] = (pel_t)((3 * src[2] + 35 * src[3] + 61 * src[4] + 29 * src[5] + 64) >> 7);
            pfirst[8][i] = (pel_t)((3 * src[3] + 7 * src[4] + 5 * src[5] + 1 * src[6] + 8) >> 4);
            pfirst[9][i] = (pel_t)((3 * src[3] + 11 * src[4] + 13 * src[5] + 5 * src[6] + 16) >> 5);
            pfirst[10][i] = (pel_t)((1 * src[3] + 33 * src[4] + 63 * src[5] + 31 * src[6] + 64) >> 7);
        }

        // padding
        if (real_size < line_size) {
            pfirst[8][real_size - 3] = pfirst[8][real_size - 4];
            pfirst[9][real_size - 3] = pfirst[9][real_size - 4];
            pfirst[10][real_size - 3] = pfirst[10][real_size - 4];
            pfirst[8][real_size - 2] = pfirst[8][real_size - 3];
            pfirst[9][real_size - 2] = pfirst[9][real_size - 3];
            pfirst[10][real_size - 2] = pfirst[10][real_size - 3];
            pfirst[8][real_size - 1] = pfirst[8][real_size - 2];
            pfirst[9][real_size - 1] = pfirst[9][real_size - 2];
            pfirst[10][real_size - 1] = pfirst[10][real_size - 2];

            pfirst[5][real_size - 2] = pfirst[5][real_size - 3];
            pfirst[6][real_size - 2] = pfirst[6][real_size - 3];
            pfirst[7][real_size - 2] = pfirst[7][real_size - 3];
            pfirst[5][real_size - 1] = pfirst[5][real_size - 2];
            pfirst[6][real_size - 1] = pfirst[6][real_size - 2];
            pfirst[7][real_size - 1] = pfirst[7][real_size - 2];

            pfirst[2][real_size - 1] = pfirst[2][real_size - 2];
            pfirst[3][real_size - 1] = pfirst[3][real_size - 2];
            pfirst[4][real_size - 1] = pfirst[4][real_size - 2];


            pad1 = pfirst[0][real_size - 1];
            pad2 = pfirst[1][real_size - 1];
            pad3 = pfirst[2][real_size - 1];
            pad4 = pfirst[3][real_size - 1];
            pad5 = pfirst[4][real_size - 1];
            pad6 = pfirst[5][real_size - 1];
            pad7 = pfirst[6][real_size - 1];
            pad8 = pfirst[7][real_size - 1];
            pad9 = pfirst[8][real_size - 1];
            pad10 = pfirst[9][real_size - 1];
            pad11 = pfirst[10][real_size - 1];
            for (; i < line_size; i++) {
                pfirst[0][i] = pad1;
                pfirst[1][i] = pad2;
                pfirst[2][i] = pad3;
                pfirst[3][i] = pad4;
                pfirst[4][i] = pad5;
                pfirst[5][i] = pad6;
                pfirst[6][i] = pad7;
                pfirst[7][i] = pad8;
                pfirst[8][i] = pad9;
                pfirst[9][i] = pad10;
                pfirst[10][i] = pad11;
            }
        }

        int bsy_b = bsy / 11;
        for (i = 0; i < bsy_b; i++) {
            memcpy(dst, pfirst[0] + i, bsx * sizeof(pel_t));
            memcpy(dst + i_dst, pfirst[1] + i, bsx * sizeof(pel_t));
            memcpy(dst + 2 * i_dst, pfirst[2] + i, bsx * sizeof(pel_t));
            memcpy(dst + 3 * i_dst, pfirst[3] + i, bsx * sizeof(pel_t));
            memcpy(dst + 4 * i_dst, pfirst[4] + i, bsx * sizeof(pel_t));
            memcpy(dst + 5 * i_dst, pfirst[5] + i, bsx * sizeof(pel_t));
            memcpy(dst + 6 * i_dst, pfirst[6] + i, bsx * sizeof(pel_t));
            memcpy(dst + 7 * i_dst, pfirst[7] + i, bsx * sizeof(pel_t));
            memcpy(dst + 8 * i_dst, pfirst[8] + i, bsx * sizeof(pel_t));
            memcpy(dst + 9 * i_dst, pfirst[9] + i, bsx * sizeof(pel_t));
            memcpy(dst + 10 * i_dst, pfirst[10] + i, bsx * sizeof(pel_t));
            dst += i_dst11;
        }
        int bsy_r = bsy - bsy_b * 11;
        for (i = 0; i < bsy_r; i++) {
            memcpy(dst, pfirst[i] + bsy_b, bsx * sizeof(pel_t));
            dst += i_dst;
        }
        */
    } else if (bsy == 8) {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        pel_t *dst5 = dst4 + i_dst;
        pel_t *dst6 = dst5 + i_dst;
        pel_t *dst7 = dst6 + i_dst;
        pel_t *dst8 = dst7 + i_dst;
        for (int i = 0; i < bsx; i++, src++) {
            dst1[i] = (pel_t)((21 * src[0] + 53 * src[1] + 43 * src[2] + 11 * src[3] + 64) >> 7);
            dst2[i] = (pel_t)((9  * src[0] + 41 * src[1] + 55 * src[2] + 23 * src[3] + 64) >> 7);
            dst3[i] = (pel_t)((15 * src[1] + 31 * src[2] + 17 * src[3] +      src[4] + 32) >> 6);
            dst4[i] = (pel_t)((9  * src[1] + 25 * src[2] + 23 * src[3] + 7  * src[4] + 32) >> 6);

            dst5[i] = (pel_t)((3  * src[1] + 19 * src[2] + 29 * src[3] + 13 * src[4] + 32) >> 6);
            dst6[i] = (pel_t)((27 * src[2] + 59 * src[3] + 37 * src[4] + 5  * src[5] + 64) >> 7);
            dst7[i] = (pel_t)((15 * src[2] + 47 * src[3] + 49 * src[4] + 17 * src[5] + 64) >> 7);
            dst8[i] = (pel_t)((3  * src[2] + 35 * src[3] + 61 * src[4] + 29 * src[5] + 64) >> 7);
        }
    } else { /*if (bsy == 4)*/
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        for (int i = 0; i < bsx; i++, src++) {
            dst1[i] = (pel_t)((21 * src[0] + 53 * src[1] + 43 * src[2] + 11 * src[3] + 64) >> 7);
            dst2[i] = (pel_t)((9  * src[0] + 41 * src[1] + 55 * src[2] + 23 * src[3] + 64) >> 7);
            dst3[i] = (pel_t)((15 * src[1] + 31 * src[2] + 17 * src[3] +      src[4] + 32) >> 6);
            dst4[i] = (pel_t)((9  * src[1] + 25 * src[2] + 23 * src[3] + 7  * src[4] + 32) >> 6);
        }
    }

}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_x_10_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    pel_t *dst1 = dst;
    pel_t *dst2 = dst1 + i_dst;
    pel_t *dst3 = dst2 + i_dst;
    pel_t *dst4 = dst3 + i_dst;
    int i;

    if (bsy != 4) {
        ALIGN16(pel_t first_line[4 * (64 + 16)]);
        int line_size = bsx + bsy / 4 - 1;
        int aligned_line_size = ((line_size + 15) >> 4) << 4;
        pel_t *pfirst[4];

        pfirst[0] = first_line;
        pfirst[1] = first_line + aligned_line_size;
        pfirst[2] = first_line + aligned_line_size * 2;
        pfirst[3] = first_line + aligned_line_size * 3;

        for (i = 0; i < line_size; i++, src++) {
            pfirst[0][i] = (pel_t)((src[0] * 3 +  src[1] * 7 + src[2]  * 5 + src[3]     + 8) >> 4);
            pfirst[1][i] = (pel_t)((src[0]     + (src[1]     + src[2]) * 3 + src[3]     + 4) >> 3);
            pfirst[2][i] = (pel_t)((src[0]     +  src[1] * 5 + src[2]  * 7 + src[3] * 3 + 8) >> 4);
            pfirst[3][i] = (pel_t)((src[1]     +  src[2] * 2 + src[3]                   + 2) >> 2);
        }

        bsy   >>= 2;
        i_dst <<= 2;
        for (i = 0; i < bsy; i++) {
            memcpy(dst1, pfirst[0] + i, bsx * sizeof(pel_t));
            memcpy(dst2, pfirst[1] + i, bsx * sizeof(pel_t));
            memcpy(dst3, pfirst[2] + i, bsx * sizeof(pel_t));
            memcpy(dst4, pfirst[3] + i, bsx * sizeof(pel_t));
            dst1 += i_dst;
            dst2 += i_dst;
            dst3 += i_dst;
            dst4 += i_dst;
        }
    } else {
        for (i = 0; i < bsx; i++, src++) {
            dst1[i] = (pel_t)((src[0] * 3 +  src[1] * 7 + src[2]  * 5 + src[3]     + 8) >> 4);
            dst2[i] = (pel_t)((src[0]     + (src[1]     + src[2]) * 3 + src[3]     + 4) >> 3);
            dst3[i] = (pel_t)((src[0]     +  src[1] * 5 + src[2]  * 7 + src[3] * 3 + 8) >> 4);
            dst4[i] = (pel_t)((src[1]     +  src[2] * 2 + src[3]                   + 2) >> 2);
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_x_11_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;
    if (bsy > 8) {
        ALIGN16(pel_t first_line[(64 + 16) << 3]);
        int line_size = bsx + (bsy >> 3) - 1;
        int aligned_line_size = ((line_size + 15) >> 4) << 4;
        int i_dst8 = i_dst << 3;
        pel_t *pfirst[8];

        pfirst[0] = first_line;
        pfirst[1] = pfirst[0] + aligned_line_size;
        pfirst[2] = pfirst[1] + aligned_line_size;
        pfirst[3] = pfirst[2] + aligned_line_size;
        pfirst[4] = pfirst[3] + aligned_line_size;
        pfirst[5] = pfirst[4] + aligned_line_size;
        pfirst[6] = pfirst[5] + aligned_line_size;
        pfirst[7] = pfirst[6] + aligned_line_size;
        for (i = 0; i < line_size; i++, src++) {
            pfirst[0][i] = (pel_t)((7 * src[0] + 15 * src[1] +  9 * src[2] +     src[3] + 16) >> 5);
            pfirst[1][i] = (pel_t)((3 * src[0] +  7 * src[1] +  5 * src[2] +     src[3] +  8) >> 4);
            pfirst[2][i] = (pel_t)((5 * src[0] + 13 * src[1] + 11 * src[2] + 3 * src[3] + 16) >> 5);
            pfirst[3][i] = (pel_t)((    src[0] +  3 * src[1] +  3 * src[2] +     src[3] +  4) >> 3);

            pfirst[4][i] = (pel_t)((3 * src[0] + 11 * src[1] + 13 * src[2] + 5 * src[3] + 16) >> 5);
            pfirst[5][i] = (pel_t)((    src[0] +  5 * src[1] +  7 * src[2] + 3 * src[3] +  8) >> 4);
            pfirst[6][i] = (pel_t)((    src[0] +  9 * src[1] + 15 * src[2] + 7 * src[3] + 16) >> 5);
            pfirst[7][i] = (pel_t)((    src[1] +  2 * src[2] +      src[3] + 0 * src[4] +  2) >> 2);
        }

        bsy >>= 3;
        for (i = 0; i < bsy; i++) {
            memcpy(dst            , pfirst[0] + i, bsx * sizeof(pel_t));
            memcpy(dst +     i_dst, pfirst[1] + i, bsx * sizeof(pel_t));
            memcpy(dst + 2 * i_dst, pfirst[2] + i, bsx * sizeof(pel_t));
            memcpy(dst + 3 * i_dst, pfirst[3] + i, bsx * sizeof(pel_t));
            memcpy(dst + 4 * i_dst, pfirst[4] + i, bsx * sizeof(pel_t));
            memcpy(dst + 5 * i_dst, pfirst[5] + i, bsx * sizeof(pel_t));
            memcpy(dst + 6 * i_dst, pfirst[6] + i, bsx * sizeof(pel_t));
            memcpy(dst + 7 * i_dst, pfirst[7] + i, bsx * sizeof(pel_t));
            dst += i_dst8;
        }
    } else if (bsy == 8) {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        pel_t *dst5 = dst4 + i_dst;
        pel_t *dst6 = dst5 + i_dst;
        pel_t *dst7 = dst6 + i_dst;
        pel_t *dst8 = dst7 + i_dst;
        for (i = 0; i < bsx; i++, src++) {
            dst1[i] = (pel_t)((7 * src[0] + 15 * src[1] +  9 * src[2] +     src[3] + 16) >> 5);
            dst2[i] = (pel_t)((3 * src[0] +  7 * src[1] +  5 * src[2] +     src[3] + 8) >> 4);
            dst3[i] = (pel_t)((5 * src[0] + 13 * src[1] + 11 * src[2] + 3 * src[3] + 16) >> 5);
            dst4[i] = (pel_t)((    src[0] +  3 * src[1] +  3 * src[2] +     src[3] + 4) >> 3);

            dst5[i] = (pel_t)((3 * src[0] + 11 * src[1] + 13 * src[2] + 5 * src[3] + 16) >> 5);
            dst6[i] = (pel_t)((    src[0] +  5 * src[1] +  7 * src[2] + 3 * src[3] +  8) >> 4);
            dst7[i] = (pel_t)((    src[0] +  9 * src[1] + 15 * src[2] + 7 * src[3] + 16) >> 5);
            dst8[i] = (pel_t)((    src[1] +  2 * src[2] +      src[3] +            +  2) >> 2);
        }
    } else {
        for (i = 0; i < bsx; i++, src++) {
            pel_t *dst1 = dst;
            pel_t *dst2 = dst1 + i_dst;
            pel_t *dst3 = dst2 + i_dst;
            pel_t *dst4 = dst3 + i_dst;
            dst1[i] = (pel_t)(( 7 * src[0] + 15 * src[1] +  9 * src[2] +      src[3] + 16) >> 5);
            dst2[i] = (pel_t)(( 3 * src[0] +  7 * src[1] +  5 * src[2] +      src[3] +  8) >> 4);
            dst3[i] = (pel_t)(( 5 * src[0] + 13 * src[1] + 11 * src[2] +  3 * src[3] + 16) >> 5);
            dst4[i] = (pel_t)((     src[0] +  3 * src[1] +  3 * src[2] +      src[3] +  4) >> 3);
        }
    }
}


/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_y_25_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;

    if (bsx > 8) {
        ALIGN16(pel_t first_line[64 + (64 << 3)]);
        int line_size = bsx + ((bsy - 1) << 3);
        int iHeight8 = bsy << 3;
        for (i = 0; i < line_size; i += 8, src--) {
            first_line[0 + i] = (pel_t)((src[0] * 7 + src[-1] * 15 + src[-2] *  9 + src[-3] * 1 + 16) >> 5);
            first_line[1 + i] = (pel_t)((src[0] * 3 + src[-1] * 7  + src[-2] *  5 + src[-3] * 1 + 8) >> 4);
            first_line[2 + i] = (pel_t)((src[0] * 5 + src[-1] * 13 + src[-2] * 11 + src[-3] * 3 + 16) >> 5);
            first_line[3 + i] = (pel_t)((src[0] * 1 + src[-1] * 3  + src[-2] *  3 + src[-3] * 1 + 4) >> 3);

            first_line[4 + i] = (pel_t)((src[0] * 3 + src[-1] * 11 + src[-2] * 13 + src[-3] * 5 + 16) >> 5);
            first_line[5 + i] = (pel_t)((src[0] * 1 + src[-1] *  5 + src[-2] *  7 + src[-3] * 3 + 8) >> 4);
            first_line[6 + i] = (pel_t)((src[0] * 1 + src[-1] *  9 + src[-2] * 15 + src[-3] * 7 + 16) >> 5);
            first_line[7 + i] = (pel_t)((             src[-1] *  1 + src[-2] *  2 + src[-3] * 1 + 2) >> 2);
        }
        for (i = 0; i < iHeight8; i += 8) {
            memcpy(dst, first_line + i, bsx * sizeof(pel_t));
            dst += i_dst;
        }
    } else if (bsx == 8) {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((src[0] * 7 + src[-1] * 15 + src[-2] *  9 + src[-3] * 1 + 16) >> 5);
            dst[1] = (pel_t)((src[0] * 3 + src[-1] *  7 + src[-2] *  5 + src[-3] * 1 + 8) >> 4);
            dst[2] = (pel_t)((src[0] * 5 + src[-1] * 13 + src[-2] * 11 + src[-3] * 3 + 16) >> 5);
            dst[3] = (pel_t)((src[0] * 1 + src[-1] *  3 + src[-2] *  3 + src[-3] * 1 + 4) >> 3);

            dst[4] = (pel_t)((src[0] * 3 + src[-1] * 11 + src[-2] * 13 + src[-3] * 5 + 16) >> 5);
            dst[5] = (pel_t)((src[0] * 1 + src[-1] *  5 + src[-2] *  7 + src[-3] * 3 + 8) >> 4);
            dst[6] = (pel_t)((src[0] * 1 + src[-1] *  9 + src[-2] * 15 + src[-3] * 7 + 16) >> 5);
            dst[7] = (pel_t)((             src[-1] *  1 + src[-2] *  2 + src[-3] * 1 + 2) >> 2);
            dst += i_dst;
        }
    } else {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((src[0] * 7 + src[-1] * 15 + src[-2] *  9 + src[-3] * 1 + 16) >> 5);
            dst[1] = (pel_t)((src[0] * 3 + src[-1] *  7 + src[-2] *  5 + src[-3] * 1 + 8) >> 4);
            dst[2] = (pel_t)((src[0] * 5 + src[-1] * 13 + src[-2] * 11 + src[-3] * 3 + 16) >> 5);
            dst[3] = (pel_t)((src[0] * 1 + src[-1] *  3 + src[-2] *  3 + src[-3] * 1 + 4) >> 3);
            dst += i_dst;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_y_26_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;

    if (bsx != 4) {
        ALIGN16(pel_t first_line[64 + 256]);
        int line_size = bsx + ((bsy - 1) << 2);
        int iHeight4 = bsy << 2;

        for (i = 0; i < line_size; i += 4, src--) {
            first_line[i    ] = (pel_t)((src[ 0] * 3 +  src[-1] * 7 + src[-2]  * 5 + src[-3]     + 8) >> 4);
            first_line[i + 1] = (pel_t)((src[ 0]     + (src[-1]     + src[-2]) * 3 + src[-3]     + 4) >> 3);
            first_line[i + 2] = (pel_t)((src[ 0]     +  src[-1] * 5 + src[-2]  * 7 + src[-3] * 3 + 8) >> 4);
            first_line[i + 3] = (pel_t)((src[-1]     +  src[-2] * 2 + src[-3]                    + 2) >> 2);
        }

        for (i = 0; i < iHeight4; i += 4) {
            memcpy(dst, first_line + i, bsx * sizeof(pel_t));
            dst += i_dst;
        }
    } else {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((src[ 0] * 3 +  src[-1] * 7 + src[-2]  * 5 + src[-3]     + 8) >> 4);
            dst[1] = (pel_t)((src[ 0]     + (src[-1]     + src[-2]) * 3 + src[-3]     + 4) >> 3);
            dst[2] = (pel_t)((src[ 0]     +  src[-1] * 5 + src[-2]  * 7 + src[-3] * 3 + 8) >> 4);
            dst[3] = (pel_t)((src[-1]     +  src[-2] * 2 + src[-3]                    + 2) >> 2);
            dst += i_dst;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_y_27_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;
    if (bsx > 8) {
        intra_pred_ang_y_c(src, dst, i_dst, dir_mode, bsx, bsy);
    } else if (bsx == 8) {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((21 * src[0] +  53 * src[-1] + 43 * src[-2] + 11 * src[-3] + 64) >> 7);
            dst[1] = (pel_t)(( 9 * src[0] +  41 * src[-1] + 55 * src[-2] + 23 * src[-3] + 64) >> 7);
            dst[2] = (pel_t)((15 * src[-1] + 31 * src[-2] + 17 * src[-3] +  1 * src[-4] + 32) >> 6);
            dst[3] = (pel_t)(( 9 * src[-1] + 25 * src[-2] + 23 * src[-3] +  7 * src[-4] + 32) >> 6);

            dst[4] = (pel_t)(( 3 * src[-1] + 19 * src[-2] + 29 * src[-3] + 13 * src[-4] + 32) >> 6);
            dst[5] = (pel_t)((27 * src[-2] + 59 * src[-3] + 37 * src[-4] +  5 * src[-5] + 64) >> 7);
            dst[6] = (pel_t)((15 * src[-2] + 47 * src[-3] + 49 * src[-4] + 17 * src[-5] + 64) >> 7);
            dst[7] = (pel_t)(( 3 * src[-2] + 35 * src[-3] + 61 * src[-4] + 29 * src[-5] + 64) >> 7);
            dst += i_dst;
        }
    } else {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((21 * src[0]  + 53 * src[-1] + 43 * src[-2] + 11 * src[-3] + 64) >> 7);
            dst[1] = (pel_t)(( 9 * src[0]  + 41 * src[-1] + 55 * src[-2] + 23 * src[-3] + 64) >> 7);
            dst[2] = (pel_t)((15 * src[-1] + 31 * src[-2] + 17 * src[-3] +  1 * src[-4] + 32) >> 6);
            dst[3] = (pel_t)(( 9 * src[-1] + 25 * src[-2] + 23 * src[-3] +  7 * src[-4] + 32) >> 6);
            dst += i_dst;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_y_28_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[64 + 128]);
    int line_size = bsx + ((bsy - 1) << 1);
    int iHeight2 = bsy << 1;
    int i;

    for (i = 0; i < line_size; i += 2, src--) {
        first_line[i    ] = (pel_t)((src[ 0] + (src[-1] + src[-2]) * 3 + src[-3] + 4) >> 3);
        first_line[i + 1] = (pel_t)((src[-1] + (src[-2] << 1)          + src[-3] + 2) >> 2);
    }

    for (i = 0; i < iHeight2; i += 2) {
        memcpy(dst, first_line + i, bsx * sizeof(pel_t));
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_y_29_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;
    if (bsx > 8) {
        intra_pred_ang_y_c(src, dst, i_dst, dir_mode, bsx, bsy);
    } else if (bsx == 8) {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((src[0] * 9 + src[-1] * 41 + src[-2] * 55 + src[-3] * 23 + 64) >> 7);
            dst[1] = (pel_t)((src[-1] * 9 + src[-2] * 25 + src[-3] * 23 + src[-4] * 7 + 32) >> 6);
            dst[2] = (pel_t)((src[-2] * 27 + src[-3] * 59 + src[-4] * 37 + src[-5] * 5 + 64) >> 7);
            dst[3] = (pel_t)((src[-2] * 3 + src[-3] * 35 + src[-4] * 61 + src[-5] * 29 + 64) >> 7);

            dst[4] = (pel_t)((src[-3] * 3 + src[-4] * 11 + src[-5] * 13 + src[-6] * 5 + 16) >> 5);
            dst[5] = (pel_t)((src[-4] * 21 + src[-5] * 53 + src[-6] * 43 + src[-7] * 11 + 64) >> 7);
            dst[6] = (pel_t)((src[-5] * 15 + src[-6] * 31 + src[-7] * 17 + src[-8] + 32) >> 6);
            dst[7] = (pel_t)((src[-5] * 3 + src[-6] * 19 + src[-7] * 29 + src[-8] * 13 + 32) >> 6);
            dst += i_dst;
        }
    } else {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((src[0] * 9 + src[-1] * 41 + src[-2] * 55 + src[-3] * 23 + 64) >> 7);
            dst[1] = (pel_t)((src[-1] * 9 + src[-2] * 25 + src[-3] * 23 + src[-4] * 7 + 32) >> 6);
            dst[2] = (pel_t)((src[-2] * 27 + src[-3] * 59 + src[-4] * 37 + src[-5] * 5 + 64) >> 7);
            dst[3] = (pel_t)((src[-2] * 3 + src[-3] * 35 + src[-4] * 61 + src[-5] * 29 + 64) >> 7);
            dst += i_dst;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_y_30_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[64 + 64]);
    int line_size = bsx + bsy - 1;
    int i;

    src -= 2;
    for (i = 0; i < line_size; i++, src--) {
        first_line[i] = (pel_t)((src[-1] + (src[0] << 1) + src[1] + 2) >> 2);
    }

    for (i = 0; i < bsy; i++) {
        memcpy(dst, first_line + i, bsx * sizeof(pel_t));
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_y_31_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t dst_tran[MAX_CU_SIZE * MAX_CU_SIZE]);
    ALIGN16(pel_t src_tran[MAX_CU_SIZE << 3]);
    int i;
    if (bsx >= bsy) {
        // transposition
        // i < (bsx * 19 / 8 + 3)
        for (i = 0; i < (bsy + bsx * 11 / 8 + 3); i++) {
            src_tran[i] = src[-i];
        }
        intra_pred_ang_x_5_c(src_tran, dst_tran, bsy, 5, bsy, bsx);
        for (i = 0; i < bsy; i++) {
            for (int j = 0; j < bsx; j++) {
                dst[j + i_dst * i] = dst_tran[i + bsy * j];
            }
        }
    } else if (bsx == 8) {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((5 * src[-1] + 13 * src[-2] + 11 * src[-3] + 3 * src[-4] + 16) >> 5);
            dst[1] = (pel_t)((1 * src[-2] + 5 * src[-3] + 7 * src[-4] + 3 * src[-5] + 8) >> 4);
            dst[2] = (pel_t)((7 * src[-4] + 15 * src[-5] + 9 * src[-6] + 1 * src[-7] + 16) >> 5);
            dst[3] = (pel_t)((1 * src[-5] + 3 * src[-6] + 3 * src[-7] + 1 * src[-8] + 4) >> 3);

            dst[4] = (pel_t)((1 * src[-6] + 9 * src[-7] + 15 * src[-8] + 7 * src[-9] + 16) >> 5);
            dst[5] = (pel_t)((3 * src[-8] + 7 * src[-9] + 5 * src[-10] + 1 * src[-11] + 8) >> 4);
            dst[6] = (pel_t)((3 * src[-9] + 11 * src[-10] + 13 * src[-11] + 5 * src[-12] + 16) >> 5);
            dst[7] = (pel_t)((1 * src[-11] + 2 * src[-12] + 1 * src[-13] + 0 * src[-14] + 2) >> 2);
            dst += i_dst;
        }
    } else {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((5 * src[-1] + 13 * src[-2] + 11 * src[-3] + 3 * src[-4] + 16) >> 5);
            dst[1] = (pel_t)((1 * src[-2] + 5 * src[-3] + 7 * src[-4] + 3 * src[-5] + 8) >> 4);
            dst[2] = (pel_t)((7 * src[-4] + 15 * src[-5] + 9 * src[-6] + 1 * src[-7] + 16) >> 5);
            dst[3] = (pel_t)((1 * src[-5] + 3 * src[-6] + 3 * src[-7] + 1 * src[-8] + 4) >> 3);
            dst += i_dst;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_y_32_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[2 * (32 + 64)]);
    int line_size = (bsy >> 1) + bsx - 1;
    int aligned_line_size = ((line_size + 15) >> 4) << 4;
    int i_dst2 = i_dst << 1;
    int i;
    pel_t *pfirst[2];

    pfirst[0] = first_line;
    pfirst[1] = first_line + aligned_line_size;

    src -= 3;
    for (i = 0; i < line_size; i++, src -= 2) {
        pfirst[0][i] = (pel_t)((src[1] + (src[ 0] << 1) + src[-1] + 2) >> 2);
        pfirst[1][i] = (pel_t)((src[0] + (src[-1] << 1) + src[-2] + 2) >> 2);
    }

    bsy >>= 1;
    for (i = 0; i < bsy; i++) {
        memcpy(dst        , pfirst[0] + i, bsx * sizeof(pel_t));
        memcpy(dst + i_dst, pfirst[1] + i, bsx * sizeof(pel_t));
        dst += i_dst2;
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_xy_13_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;
    if (bsy > 8) {
        ALIGN16(pel_t first_line[(64 + 16) << 3]);
        int line_size = bsx + (bsy >> 3) - 1;
        int left_size = line_size - bsx;
        int aligned_line_size = ((line_size + 15) >> 4) << 4;
        pel_t *pfirst[8];

        pfirst[0] = first_line;
        pfirst[1] = pfirst[0] + aligned_line_size;
        pfirst[2] = pfirst[1] + aligned_line_size;
        pfirst[3] = pfirst[2] + aligned_line_size;
        pfirst[4] = pfirst[3] + aligned_line_size;
        pfirst[5] = pfirst[4] + aligned_line_size;
        pfirst[6] = pfirst[5] + aligned_line_size;
        pfirst[7] = pfirst[6] + aligned_line_size;

        src -= bsy - 8;
        for (i = 0; i < left_size; i++, src += 8) {
            pfirst[0][i] = (pel_t)((src[6] + (src[7] << 1) + src[8] + 2) >> 2);
            pfirst[1][i] = (pel_t)((src[5] + (src[6] << 1) + src[7] + 2) >> 2);
            pfirst[2][i] = (pel_t)((src[4] + (src[5] << 1) + src[6] + 2) >> 2);
            pfirst[3][i] = (pel_t)((src[3] + (src[4] << 1) + src[5] + 2) >> 2);

            pfirst[4][i] = (pel_t)((src[2] + (src[3] << 1) + src[4] + 2) >> 2);
            pfirst[5][i] = (pel_t)((src[1] + (src[2] << 1) + src[3] + 2) >> 2);
            pfirst[6][i] = (pel_t)((src[0] + (src[1] << 1) + src[2] + 2) >> 2);
            pfirst[7][i] = (pel_t)((src[-1] + (src[0] << 1) + src[1] + 2) >> 2);
        }

        for (; i < line_size; i++, src++) {
            pfirst[0][i] = (pel_t)((7 * src[2] + 15 * src[1] + 9 * src[0] + src[-1] + 16) >> 5);
            pfirst[1][i] = (pel_t)((3 * src[2] + 7 * src[1] + 5 * src[0] + src[-1] + 8) >> 4);
            pfirst[2][i] = (pel_t)((5 * src[2] + 13 * src[1] + 11 * src[0] + 3 * src[-1] + 16) >> 5);
            pfirst[3][i] = (pel_t)((src[2] + 3 * src[1] + 3 * src[0] + src[-1] + 4) >> 3);

            pfirst[4][i] = (pel_t)((3 * src[2] + 11 * src[1] + 13 * src[0] + 5 * src[-1] + 16) >> 5);
            pfirst[5][i] = (pel_t)((src[2] + 5 * src[1] + 7 * src[0] + 3 * src[-1] + 8) >> 4);
            pfirst[6][i] = (pel_t)((src[2] + 9 * src[1] + 15 * src[0] + 7 * src[-1] + 16) >> 5);
            pfirst[7][i] = (pel_t)((src[1] + 2 * src[0] + src[-1] + 2) >> 2);
        }

        pfirst[0] += left_size;
        pfirst[1] += left_size;
        pfirst[2] += left_size;
        pfirst[3] += left_size;
        pfirst[4] += left_size;
        pfirst[5] += left_size;
        pfirst[6] += left_size;
        pfirst[7] += left_size;

        bsy >>= 3;
        for (i = 0; i < bsy; i++) {
            memcpy(dst, pfirst[0] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[1] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[2] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[3] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[4] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[5] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[6] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[7] - i, bsx * sizeof(pel_t));
            dst += i_dst;
        }
    } else if (bsy == 8) {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        pel_t *dst5 = dst4 + i_dst;
        pel_t *dst6 = dst5 + i_dst;
        pel_t *dst7 = dst6 + i_dst;
        pel_t *dst8 = dst7 + i_dst;
        for (i = 0; i < bsx; i++, src++) {
            dst1[i] = (pel_t)((7 * src[2] + 15 * src[1] + 9 * src[0] + src[-1] + 16) >> 5);
            dst2[i] = (pel_t)((3 * src[2] + 7 * src[1] + 5 * src[0] + src[-1] + 8) >> 4);
            dst3[i] = (pel_t)((5 * src[2] + 13 * src[1] + 11 * src[0] + 3 * src[-1] + 16) >> 5);
            dst4[i] = (pel_t)((src[2] + 3 * src[1] + 3 * src[0] + src[-1] + 4) >> 3);

            dst5[i] = (pel_t)((3 * src[2] + 11 * src[1] + 13 * src[0] + 5 * src[-1] + 16) >> 5);
            dst6[i] = (pel_t)((src[2] + 5 * src[1] + 7 * src[0] + 3 * src[-1] + 8) >> 4);
            dst7[i] = (pel_t)((src[2] + 9 * src[1] + 15 * src[0] + 7 * src[-1] + 16) >> 5);
            dst8[i] = (pel_t)((src[1] + 2 * src[0] + src[-1]  + 2) >> 2);
        }
    } else {
        for (i = 0; i < bsx; i++, src++) {
            pel_t *dst1 = dst;
            pel_t *dst2 = dst1 + i_dst;
            pel_t *dst3 = dst2 + i_dst;
            pel_t *dst4 = dst3 + i_dst;
            dst1[i] = (pel_t)((7 * src[2] + 15 * src[1] +  9 * src[0] +     src[-1] + 16) >> 5);
            dst2[i] = (pel_t)((3 * src[2] +  7 * src[1] +  5 * src[0] +     src[-1] + 8) >> 4);
            dst3[i] = (pel_t)((5 * src[2] + 13 * src[1] + 11 * src[0] + 3 * src[-1] + 16) >> 5);
            dst4[i] = (pel_t)((    src[2] +  3 * src[1] +  3 * src[0] +     src[-1] + 4) >> 3);
        }
    }
}
static void intra_pred_ang_xy_14_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;

    if (bsy != 4) {
        ALIGN16(pel_t first_line[4 * (64 + 16)]);
        int line_size = bsx + (bsy >> 2) - 1;
        int left_size = line_size - bsx;
        int aligned_line_size = ((line_size + 15) >> 4) << 4;
        pel_t *pfirst[4];

        pfirst[0] = first_line;
        pfirst[1] = first_line + aligned_line_size;
        pfirst[2] = first_line + aligned_line_size * 2;
        pfirst[3] = first_line + aligned_line_size * 3;

        src -= bsy - 4;
        for (i = 0; i < left_size; i++, src += 4) {
            pfirst[0][i] = (pel_t)((src[ 2] + (src[3] << 1) + src[4] + 2) >> 2);
            pfirst[1][i] = (pel_t)((src[ 1] + (src[2] << 1) + src[3] + 2) >> 2);
            pfirst[2][i] = (pel_t)((src[ 0] + (src[1] << 1) + src[2] + 2) >> 2);
            pfirst[3][i] = (pel_t)((src[-1] + (src[0] << 1) + src[1] + 2) >> 2);
        }

        for (; i < line_size; i++, src++) {
            pfirst[0][i] = (pel_t)((src[-1]     +  src[0] * 5 + src[1]  * 7 + src[2] * 3 + 8) >> 4);
            pfirst[1][i] = (pel_t)((src[-1]     + (src[0]     + src[1]) * 3 + src[2]     + 4) >> 3);
            pfirst[2][i] = (pel_t)((src[-1] * 3 +  src[0] * 7 + src[1]  * 5 + src[2]     + 8) >> 4);
            pfirst[3][i] = (pel_t)((src[-1]     +  src[0] * 2 + src[1]                   + 2) >> 2);
        }

        pfirst[0] += left_size;
        pfirst[1] += left_size;
        pfirst[2] += left_size;
        pfirst[3] += left_size;

        bsy >>= 2;
        for (i = 0; i < bsy; i++) {
            memcpy(dst, pfirst[0] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[1] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[2] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[3] - i, bsx * sizeof(pel_t));
            dst += i_dst;
        }
    } else {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;

        for (i = 0; i < bsx; i++, src++) {
            dst1[i] = (pel_t)((src[-1]     +  src[0] * 5 + src[1]  * 7 + src[2] * 3 + 8) >> 4);
            dst2[i] = (pel_t)((src[-1]     + (src[0]     + src[1]) * 3 + src[2]     + 4) >> 3);
            dst3[i] = (pel_t)((src[-1] * 3 +  src[0] * 7 + src[1]  * 5 + src[2]     + 8) >> 4);
            dst4[i] = (pel_t)((src[-1]     +  src[0] * 2 + src[1]                   + 2) >> 2);
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_xy_16_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[2 * (64 + 32)]);
    int line_size = bsx + (bsy >> 1) - 1;
    int left_size = line_size - bsx;
    int aligned_line_size = ((line_size + 15) >> 4) << 4;
    int i_dst2 = i_dst << 1;
    pel_t *pfirst[2];
    int i;

    pfirst[0] = first_line;
    pfirst[1] = first_line + aligned_line_size;

    src -= bsy - 2;
    for (i = 0; i < left_size; i++, src += 2) {
        pfirst[0][i] = (pel_t)((src[ 0] + (src[1] << 1) + src[2] + 2) >> 2);
        pfirst[1][i] = (pel_t)((src[-1] + (src[0] << 1) + src[1] + 2) >> 2);
    }

    for (; i < line_size; i++, src++) {
        pfirst[0][i] = (pel_t)((src[-1] + (src[0]       + src[1]) * 3 + src[2] + 4) >> 3);
        pfirst[1][i] = (pel_t)((src[-1] + (src[0] << 1) + src[1]               + 2) >> 2);
    }

    pfirst[0] += left_size;
    pfirst[1] += left_size;

    bsy >>= 1;
    for (i = 0; i < bsy; i++) {
        memcpy(dst        , pfirst[0] - i, bsx * sizeof(pel_t));
        memcpy(dst + i_dst, pfirst[1] - i, bsx * sizeof(pel_t));
        dst += i_dst2;
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_xy_18_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[64 + 64]);
    int line_size = bsx + bsy - 1;
    int i;
    pel_t *pfirst = first_line + bsy - 1;

    src -= bsy - 1;
    for (i = 0; i < line_size; i++, src++) {
        first_line[i] = (pel_t)((src[-1] + (src[0] << 1) + src[1] + 2) >> 2);
    }

    for (i = 0; i < bsy; i++) {
        memcpy(dst, pfirst, bsx * sizeof(pel_t));
        pfirst--;
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_xy_20_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[64 + 128]);
    int left_size = ((bsy - 1) << 1) + 1;
    int top_size = bsx - 1;
    int line_size = left_size + top_size;
    int i;
    pel_t *pfirst = first_line + left_size - 1;

    src -= bsy;
    for (i = 0; i < left_size; i += 2, src++) {
        first_line[i    ] = (pel_t)((src[-1] + (src[0] +  src[1]) * 3  + src[2] + 4) >> 3);
        first_line[i + 1] = (pel_t)((           src[0] + (src[1] << 1) + src[2] + 2) >> 2);
    }
    i--;

    for (; i < line_size; i++, src++) {
        first_line[i] = (pel_t)((src[-1] + (src[0] << 1) + src[1] + 2) >> 2);
    }

    for (i = 0; i < bsy; i++) {
        memcpy(dst, pfirst, bsx * sizeof(pel_t));
        pfirst -= 2;
        dst    += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static void intra_pred_ang_xy_22_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;

    if (bsx != 4) {
        src -= bsy;
        ALIGN16(pel_t first_line[64 + 256]);
        int left_size = ((bsy - 1) << 2) + 3;
        int top_size  = bsx - 3;
        int line_size = left_size + top_size;
        pel_t *pfirst = first_line + left_size - 3;

        for (i = 0; i < left_size; i += 4, src++) {
            first_line[i    ] = (pel_t)((src[-1] * 3 +  src[0] * 7 + src[1]  * 5 + src[2]     + 8) >> 4);
            first_line[i + 1] = (pel_t)((src[-1]     + (src[0]     + src[1]) * 3 + src[2]     + 4) >> 3);
            first_line[i + 2] = (pel_t)((src[-1]     +  src[0] * 5 + src[1]  * 7 + src[2] * 3 + 8) >> 4);
            first_line[i + 3] = (pel_t)((               src[0]     + src[1]  * 2 + src[2]     + 2) >> 2);
        }
        i--;

        for (; i < line_size; i++, src++) {
            first_line[i] = (pel_t)((src[-1] + (src[0] << 1) + src[1] + 2) >> 2);
        }

        for (i = 0; i < bsy; i++) {
            memcpy(dst, pfirst, bsx * sizeof(pel_t));
            dst    += i_dst;
            pfirst -= 4;
        }
    } else {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((src[-2] * 3 +  src[-1] * 7 + src[0]  * 5 + src[1]     + 8) >> 4);
            dst[1] = (pel_t)((src[-2]     + (src[-1]     + src[0]) * 3 + src[1]     + 4) >> 3);
            dst[2] = (pel_t)((src[-2]     +  src[-1] * 5 + src[0]  * 7 + src[1] * 3 + 8) >> 4);
            dst[3] = (pel_t)((               src[-1]     + src[0]  * 2 + src[1]     + 2) >> 2);
            dst += i_dst;
        }
        // needn't pad, (3,0) is equal for ang_x and ang_y
    }
}

/* ---------------------------------------------------------------------------
*/
static void intra_pred_ang_xy_23_c(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;

    if (bsx > 8) {
        ALIGN16(pel_t first_line[64 + 512]);
        int left_size = (bsy << 3) - 1;
        int top_size = bsx - 7;
        int line_size = left_size + top_size;
        pel_t *pfirst = first_line + left_size - 7;

        src -= bsy;
        for (i = 0; i < left_size; i += 8, src++) {
            first_line[i    ] = (pel_t)((7 * src[-1] + 15 * src[0] +  9 * src[1] +     src[2] + 16) >> 5);
            first_line[i + 1] = (pel_t)((3 * src[-1] +  7 * src[0] +  5 * src[1] +     src[2] +  8) >> 4);
            first_line[i + 2] = (pel_t)((5 * src[-1] + 13 * src[0] + 11 * src[1] + 3 * src[2] + 16) >> 5);
            first_line[i + 3] = (pel_t)((    src[-1] +  3 * src[0] +  3 * src[1] +     src[2] +  4) >> 3);

            first_line[i + 4] = (pel_t)((3 * src[-1] + 11 * src[0] + 13 * src[1] + 5 * src[2] + 16) >> 5);
            first_line[i + 5] = (pel_t)((    src[-1] +  5 * src[0] +  7 * src[1] + 3 * src[2] +  8) >> 4);
            first_line[i + 6] = (pel_t)((    src[-1] +  9 * src[0] + 15 * src[1] + 7 * src[2] + 16) >> 5);
            first_line[i + 7] = (pel_t)((    src[ 0] +  2 * src[1] +      src[2] + 0 * src[3] +  2) >> 2);
        }
        i--;

        for (; i < line_size; i++, src++) {
            first_line[i] = (pel_t)((src[1] + (src[0] << 1) + src[-1] + 2) >> 2);
        }

        for (i = 0; i < bsy; i++) {
            memcpy(dst, pfirst, bsx * sizeof(pel_t));
            dst += i_dst;
            pfirst -= 8;
        }
    } else if (bsx == 8) {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((7 * src[-2] + 15 * src[-1] +  9 * src[0] +     src[1] + 16) >> 5);
            dst[1] = (pel_t)((3 * src[-2] +  7 * src[-1] +  5 * src[0] +     src[1] +  8) >> 4);
            dst[2] = (pel_t)((5 * src[-2] + 13 * src[-1] + 11 * src[0] + 3 * src[1] + 16) >> 5);
            dst[3] = (pel_t)((    src[-2] +  3 * src[-1] +  3 * src[0] +     src[1] +  4) >> 3);

            dst[4] = (pel_t)((3 * src[-2] + 11 * src[-1] + 13 * src[0] + 5 * src[1] + 16) >> 5);
            dst[5] = (pel_t)((    src[-2] +  5 * src[-1] +  7 * src[0] + 3 * src[1] +  8) >> 4);
            dst[6] = (pel_t)((    src[-2] +  9 * src[-1] + 15 * src[0] + 7 * src[1] + 16) >> 5);
            dst[7] = (pel_t)((    src[-1] +  2 * src[ 0] +      src[1] + 0 * src[2] +  2) >> 2);
            dst += i_dst;
        }
        // needn't pad, (7,0) is equal for ang_x and ang_y
    } else {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((7 * src[-2] + 15 * src[-1] + 9 * src[0] + src[1] + 16) >> 5);
            dst[1] = (pel_t)((3 * src[-2] + 7 * src[-1] + 5 * src[0] + src[1] + 8) >> 4);
            dst[2] = (pel_t)((5 * src[-2] + 13 * src[-1] + 11 * src[0] + 3 * src[1] + 16) >> 5);
            dst[3] = (pel_t)((src[-2] + 3 * src[-1] + 3 * src[0] + src[1] + 4) >> 3);
            dst += i_dst;
        }
    }
}

/* ---------------------------------------------------------------------------
 * fill reference samples for intra prediction
 * LCU内在上边界的PU
 */
static
void fill_reference_samples_0_c(const pel_t *pTL, int i_TL, const pel_t *pLcuEP, pel_t *EP, uint32_t i_avai, int bsx, int bsy)
{
    int num_padding = 0;

    /* fill default value */
    g_funcs.mem_repeat_p(&EP[-(bsy << 1)], g_dc_value, ((bsy + bsx) << 1) + 1);

    /* get prediction pixels ---------------------------------------
     * extra pixels          | left-down pixels   | left pixels   | top-left | top pixels  | top-right pixels  | extra pixels
     * -2*bsy-4 ... -2*bsy-1 | -bsy-bsy ... -bsy-1| -bsy -3 -2 -1 |     0    | 1 2 ... bsx | bsx+1 ... bsx+bsx | 2*bsx+1 ... 2*bsx+4
     */

    /* fill top & top-right pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        /* fill top pixels */
        g_funcs.fast_memcpy(&EP[1], &pLcuEP[1], bsx * sizeof(pel_t));
    }

    /* fill top-right pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_RIGHT)) {
        g_funcs.fast_memcpy(&EP[bsx + 1], &pLcuEP[bsx + 1], bsx * sizeof(pel_t));
    } else {
        g_funcs.mem_repeat_p(&EP[bsx + 1], EP[bsx], bsx);   // repeat the last pixel
    }

    /* fill extra pixels */
    num_padding = bsy * 11 / 4 - bsx + 4;
    if (num_padding > 0) {
        g_funcs.mem_repeat_p(&EP[2 * bsx + 1], EP[2 * bsx], num_padding); // from (2*bsx) to (iX + 3) = (bsy *11/4 + bsx - 1) + 3
    }

    /* fill left & left-down pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        /* fill left pixels */
        memcpy(&EP[-bsy], &pLcuEP[-bsy], bsy * sizeof(pel_t));
    }

    /* fill left-down pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT_DOWN)) {
        memcpy(&EP[-2 * bsy], &pLcuEP[-2 * bsy], bsy * sizeof(pel_t));
    } else {
        g_funcs.mem_repeat_p(&EP[-(bsy << 1)], EP[-bsy], bsy);
    }

    /* fill top-left pixel */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_LEFT)) {
        EP[0] = pLcuEP[0];
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        EP[0] = pLcuEP[1];
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        EP[0] = pLcuEP[-1];
    }

    /* fill extra pixels */
    num_padding = bsx * 11 / 4 - bsy + 4;
    if (num_padding > 0) {
        g_funcs.mem_repeat_p(&EP[-2 * bsy - num_padding], EP[-2 * bsy], num_padding); // from (-2*bsy) to (-iY - 3) = -(bsx *11/4 + bsy - 1) - 3
    }
}

/* ---------------------------------------------------------------------------
 * fill reference samples for intra prediction
 * LCU内在上边界的PU
 */
static
void fill_reference_samples_x_c(const pel_t *pTL, int i_TL, const pel_t *pLcuEP, pel_t *EP, uint32_t i_avai, int bsx, int bsy)
{
    const pel_t *pL = pTL + i_TL;
    int num_padding = 0;

    /* fill default value */
    g_funcs.mem_repeat_p(&EP[-(bsy << 1)], g_dc_value, ((bsy + bsx) << 1) + 1);

    /* get prediction pixels ---------------------------------------
     * extra pixels          | left-down pixels   | left pixels   | top-left | top pixels  | top-right pixels  | extra pixels
     * -2*bsy-4 ... -2*bsy-1 | -bsy-bsy ... -bsy-1| -bsy -3 -2 -1 |     0    | 1 2 ... bsx | bsx+1 ... bsx+bsx | 2*bsx+1 ... 2*bsx+4
     */

    /* fill top & top-right pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        /* fill top pixels */
        g_funcs.fast_memcpy(&EP[1], &pLcuEP[1], bsx * sizeof(pel_t));
    }

    /* fill top-right pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_RIGHT)) {
        g_funcs.fast_memcpy(&EP[bsx + 1], &pLcuEP[bsx + 1], bsx * sizeof(pel_t));
    } else {
        g_funcs.mem_repeat_p(&EP[bsx + 1], EP[bsx], bsx);   // repeat the last pixel
    }

    /* fill extra pixels */
    num_padding = bsy * 11 / 4 - bsx + 4;
    if (num_padding > 0) {
        g_funcs.mem_repeat_p(&EP[2 * bsx + 1], EP[2 * bsx], num_padding); // from (2*bsx) to (iX + 3) = (bsy *11/4 + bsx - 1) + 3
    }

    /* fill left & left-down pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        const pel_t *p_l = pL;
        int y;
        /* fill left pixels */
        for (y = 0; y < bsy; y++) {
            EP[-1 - y] = *p_l;
            p_l += i_TL;
        }
    }

    /* fill left-down pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT_DOWN)) {
        int y;
        const pel_t *p_l = pL + bsy * i_TL;

        for (y = 0; y < bsy; y++) {
            EP[-bsy - 1 - y] = *p_l;
            p_l += i_TL;
        }
    } else {
        g_funcs.mem_repeat_p(&EP[-(bsy << 1)], EP[-bsy], bsy);
    }

    /* fill top-left pixel */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_LEFT)) {
        EP[0] = pLcuEP[0];
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        EP[0] = pLcuEP[1];
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        EP[0] = pL[0];
    }

    /* fill extra pixels */
    num_padding = bsx * 11 / 4 - bsy + 4;
    if (num_padding > 0) {
        g_funcs.mem_repeat_p(&EP[-2 * bsy - num_padding], EP[-2 * bsy], num_padding); // from (-2*bsy) to (-iY - 3) = -(bsx *11/4 + bsy - 1) - 3
    }
}

/* ---------------------------------------------------------------------------
 * fill reference samples for intra prediction
 * LCU内在左边界上的PU
 */
static
void fill_reference_samples_y_c(const pel_t *pTL, int i_TL, const pel_t *pLcuEP, pel_t *EP, uint32_t i_avai, int bsx, int bsy)
{
    const pel_t *pT = pTL + 1;
    int num_padding = 0;

    /* fill default value */
    g_funcs.mem_repeat_p(&EP[-(bsy << 1)], g_dc_value, ((bsy + bsx) << 1) + 1);

    /* get prediction pixels ---------------------------------------
     * extra pixels          | left-down pixels   | left pixels   | top-left | top pixels  | top-right pixels  | extra pixels
     * -2*bsy-4 ... -2*bsy-1 | -bsy-bsy ... -bsy-1| -bsy -3 -2 -1 |     0    | 1 2 ... bsx | bsx+1 ... bsx+bsx | 2*bsx+1 ... 2*bsx+4
     */

    /* fill top & top-right pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        /* fill top pixels */
        g_funcs.fast_memcpy(&EP[1], pT, bsx * sizeof(pel_t));
    }

    /* fill top-right pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_RIGHT)) {
        g_funcs.fast_memcpy(&EP[bsx + 1], &pT[bsx], bsx * sizeof(pel_t));
    } else {
        g_funcs.mem_repeat_p(&EP[bsx + 1], EP[bsx], bsx);   // repeat the last pixel
    }

    /* fill extra pixels */
    num_padding = bsy * 11 / 4 - bsx + 4;
    if (num_padding > 0) {
        g_funcs.mem_repeat_p(&EP[2 * bsx + 1], EP[2 * bsx], num_padding); // from (2*bsx) to (iX + 3) = (bsy *11/4 + bsx - 1) + 3
    }

    /* fill left & left-down pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        /* fill left pixels */
        memcpy(&EP[-bsy], &pLcuEP[-bsy], bsy * sizeof(pel_t));
    }

    /* fill left-down pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT_DOWN)) {
        memcpy(&EP[-2 * bsy], &pLcuEP[-2 * bsy], bsy * sizeof(pel_t));
    } else {
        g_funcs.mem_repeat_p(&EP[-(bsy << 1)], EP[-bsy], bsy);
    }

    /* fill top-left pixel */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_LEFT)) {
        EP[0] = pLcuEP[0];
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        EP[0] = pT[0];
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        EP[0] = pLcuEP[-1];
    }

    /* fill extra pixels */
    num_padding = bsx * 11 / 4 - bsy + 4;
    if (num_padding > 0) {
        g_funcs.mem_repeat_p(&EP[-2 * bsy - num_padding], EP[-2 * bsy], num_padding); // from (-2*bsy) to (-iY - 3) = -(bsx *11/4 + bsy - 1) - 3
    }
}

/* ---------------------------------------------------------------------------
 * fill reference samples for intra prediction
 * LCU内不在边界上的PU
 */
static
void fill_reference_samples_xy_c(const pel_t *pTL, int i_TL, const pel_t *pLcuEP, pel_t *EP, uint32_t i_avai, int bsx, int bsy)
{
    const pel_t *pT = pTL + 1;
    const pel_t *pL = pTL + i_TL;
    int num_padding = 0;

    /* fill default value */
    g_funcs.mem_repeat_p(&EP[-(bsy << 1)], g_dc_value, ((bsy + bsx) << 1) + 1);

    /* get prediction pixels ---------------------------------------
     * extra pixels          | left-down pixels   | left pixels   | top-left | top pixels  | top-right pixels  | extra pixels
     * -2*bsy-4 ... -2*bsy-1 | -bsy-bsy ... -bsy-1| -bsy -3 -2 -1 |     0    | 1 2 ... bsx | bsx+1 ... bsx+bsx | 2*bsx+1 ... 2*bsx+4
     */

    /* fill top & top-right pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        /* fill top pixels */
        g_funcs.fast_memcpy(&EP[1], pT, bsx * sizeof(pel_t));
    }

    /* fill top-right pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_RIGHT)) {
        g_funcs.fast_memcpy(&EP[bsx + 1], &pT[bsx], bsx * sizeof(pel_t));
    } else {
        g_funcs.mem_repeat_p(&EP[bsx + 1], EP[bsx], bsx);   // repeat the last pixel
    }

    /* fill extra pixels */
    num_padding = bsy * 11 / 4 - bsx + 4;
    if (num_padding > 0) {
        g_funcs.mem_repeat_p(&EP[2 * bsx + 1], EP[2 * bsx], num_padding); // from (2*bsx) to (iX + 3) = (bsy *11/4 + bsx - 1) + 3
    }

    /* fill left & left-down pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        const pel_t *p_l = pL;
        int y;
        /* fill left pixels */
        for (y = 0; y < bsy; y++) {
            EP[-1 - y] = *p_l;
            p_l += i_TL;
        }
    }

    /* fill left-down pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT_DOWN)) {
        int y;
        const pel_t *p_l = pL + bsy * i_TL;

        for (y = 0; y < bsy; y++) {
            EP[-bsy - 1 - y] = *p_l;
            p_l += i_TL;
        }
    } else {
        g_funcs.mem_repeat_p(&EP[-(bsy << 1)], EP[-bsy], bsy);
    }

    /* fill top-left pixel */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_LEFT)) {
        EP[0] = pTL[0];
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        EP[0] = pT[0];
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        EP[0] = pL[0];
    }

    /* fill extra pixels */
    num_padding = bsx * 11 / 4 - bsy + 4;
    if (num_padding > 0) {
        g_funcs.mem_repeat_p(&EP[-2 * bsy - num_padding], EP[-2 * bsy], num_padding); // from (-2*bsy) to (-iY - 3) = -(bsx *11/4 + bsy - 1) - 3
    }
}

/**
 * ===========================================================================
 * interface function definition
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
void xavs2_intra_pred_init(uint32_t cpuid, intrinsic_func_t *pf)
{
#define ANG_X_OFFSET    3
#define ANG_XY_OFFSET   13
#define ANG_Y_OFFSET    25
    int i;

    intra_pred_t *ipred = pf->intraf;

    pf->fill_edge_f[0] = fill_reference_samples_0_c;
    pf->fill_edge_f[1] = fill_reference_samples_x_c;
    pf->fill_edge_f[2] = fill_reference_samples_y_c;
    pf->fill_edge_f[3] = fill_reference_samples_xy_c;
    ipred[DC_PRED   ] = intra_pred_dc_c;                // 0
    ipred[PLANE_PRED] = intra_pred_plane_c;             // 1
    ipred[BI_PRED   ] = intra_pred_bilinear_c;          // 2

    for (i = ANG_X_OFFSET; i < VERT_PRED; i++) {
        ipred[i     ] = intra_pred_ang_x_c;             // 3 ~ 11
    }
    ipred[VERT_PRED ] = intra_pred_ver_c;               // 12

    for (i = ANG_XY_OFFSET; i < HOR_PRED; i++) {
        ipred[i     ] = intra_pred_ang_xy_c;            // 13 ~ 23
    }

    ipred[HOR_PRED  ] = intra_pred_hor_c;               // 24
    for (i = ANG_Y_OFFSET; i < NUM_INTRA_MODE; i++) {
        ipred[i     ] = intra_pred_ang_y_c;             // 25 ~ 32
    }

    ipred[INTRA_ANG_X_3 ]  = intra_pred_ang_x_3_c;
    ipred[INTRA_ANG_X_4 ]  = intra_pred_ang_x_4_c;
    ipred[INTRA_ANG_X_5 ]  = intra_pred_ang_x_5_c;
    ipred[INTRA_ANG_X_6 ]  = intra_pred_ang_x_6_c;
    ipred[INTRA_ANG_X_7 ]  = intra_pred_ang_x_7_c;
    ipred[INTRA_ANG_X_8 ]  = intra_pred_ang_x_8_c;
    ipred[INTRA_ANG_X_9 ]  = intra_pred_ang_x_9_c;
    ipred[INTRA_ANG_X_10]  = intra_pred_ang_x_10_c;
    ipred[INTRA_ANG_X_11]  = intra_pred_ang_x_11_c;

    ipred[INTRA_ANG_XY_13] = intra_pred_ang_xy_13_c;
    ipred[INTRA_ANG_XY_14] = intra_pred_ang_xy_14_c;
    ipred[INTRA_ANG_XY_16] = intra_pred_ang_xy_16_c;
    ipred[INTRA_ANG_XY_18] = intra_pred_ang_xy_18_c;
    ipred[INTRA_ANG_XY_20] = intra_pred_ang_xy_20_c;
    ipred[INTRA_ANG_XY_22] = intra_pred_ang_xy_22_c;
    ipred[INTRA_ANG_XY_23] = intra_pred_ang_xy_23_c;

    ipred[INTRA_ANG_Y_25]  = intra_pred_ang_y_25_c;
    ipred[INTRA_ANG_Y_26]  = intra_pred_ang_y_26_c;
    ipred[INTRA_ANG_Y_27]  = intra_pred_ang_y_27_c;
    ipred[INTRA_ANG_Y_28]  = intra_pred_ang_y_28_c;
    ipred[INTRA_ANG_Y_29]  = intra_pred_ang_y_29_c;
    ipred[INTRA_ANG_Y_30]  = intra_pred_ang_y_30_c;
    ipred[INTRA_ANG_Y_31]  = intra_pred_ang_y_31_c;
    ipred[INTRA_ANG_Y_32]  = intra_pred_ang_y_32_c;

    // TODO: 8bit情况下角度7、9、11性能不一致   20170716
#if HAVE_MMX
    if (cpuid & XAVS2_CPU_SSE42) {
        ipred[DC_PRED        ] = intra_pred_dc_sse128;
        ipred[HOR_PRED       ] = intra_pred_hor_sse128;
        ipred[VERT_PRED      ] = intra_pred_ver_sse128;
        ipred[PLANE_PRED     ] = intra_pred_plane_sse128;
        ipred[BI_PRED        ] = intra_pred_bilinear_sse128;
        ipred[INTRA_ANG_X_3  ] = intra_pred_ang_x_3_sse128;
        ipred[INTRA_ANG_X_4  ] = intra_pred_ang_x_4_sse128;
        ipred[INTRA_ANG_X_5  ] = intra_pred_ang_x_5_sse128;
        ipred[INTRA_ANG_X_6  ] = intra_pred_ang_x_6_sse128;
        ipred[INTRA_ANG_X_7  ] = intra_pred_ang_x_7_sse128;
        ipred[INTRA_ANG_X_8  ] = intra_pred_ang_x_8_sse128;
        ipred[INTRA_ANG_X_9  ] = intra_pred_ang_x_9_sse128;
        ipred[INTRA_ANG_X_10 ] = intra_pred_ang_x_10_sse128;
        ipred[INTRA_ANG_X_11 ] = intra_pred_ang_x_11_sse128;
        ipred[INTRA_ANG_XY_13] = intra_pred_ang_xy_13_sse128;
        ipred[INTRA_ANG_XY_14] = intra_pred_ang_xy_14_sse128;
        ipred[INTRA_ANG_XY_16] = intra_pred_ang_xy_16_sse128;
        ipred[INTRA_ANG_XY_18] = intra_pred_ang_xy_18_sse128;
        ipred[INTRA_ANG_XY_20] = intra_pred_ang_xy_20_sse128;
        ipred[INTRA_ANG_XY_22] = intra_pred_ang_xy_22_sse128;
        ipred[INTRA_ANG_XY_23] = intra_pred_ang_xy_23_sse128;
        ipred[INTRA_ANG_Y_25 ] = intra_pred_ang_y_25_sse128;
        ipred[INTRA_ANG_Y_26 ] = intra_pred_ang_y_26_sse128;
        ipred[INTRA_ANG_Y_28 ] = intra_pred_ang_y_28_sse128;
        ipred[INTRA_ANG_Y_30 ] = intra_pred_ang_y_30_sse128;
        ipred[INTRA_ANG_Y_31 ] = intra_pred_ang_y_31_sse128;
        ipred[INTRA_ANG_Y_32 ] = intra_pred_ang_y_32_sse128;
        pf->fill_edge_f[0] = fill_edge_samples_0_sse128;
        pf->fill_edge_f[1] = fill_edge_samples_x_sse128;
        pf->fill_edge_f[2] = fill_edge_samples_y_sse128;
        pf->fill_edge_f[3] = fill_edge_samples_xy_sse128;
    }

    /* 8/10bit assemble*/
    if (cpuid & XAVS2_CPU_AVX2) {
        ipred[DC_PRED        ] = intra_pred_dc_avx;
        ipred[HOR_PRED       ] = intra_pred_hor_avx;
        ipred[VERT_PRED      ] = intra_pred_ver_avx;

        ipred[PLANE_PRED     ] = intra_pred_plane_avx;
        ipred[BI_PRED        ] = intra_pred_bilinear_avx;

        ipred[INTRA_ANG_X_3  ] = intra_pred_ang_x_3_avx;
        ipred[INTRA_ANG_X_4  ] = intra_pred_ang_x_4_avx;
        ipred[INTRA_ANG_X_5  ] = intra_pred_ang_x_5_avx;
        ipred[INTRA_ANG_X_6  ] = intra_pred_ang_x_6_avx;
        ipred[INTRA_ANG_X_7  ] = intra_pred_ang_x_7_avx;
        ipred[INTRA_ANG_X_8  ] = intra_pred_ang_x_8_avx;
        ipred[INTRA_ANG_X_9  ] = intra_pred_ang_x_9_avx;
        ipred[INTRA_ANG_X_10 ] = intra_pred_ang_x_10_avx;
        ipred[INTRA_ANG_X_11 ] = intra_pred_ang_x_11_avx;

        ipred[INTRA_ANG_XY_13] = intra_pred_ang_xy_13_avx;
        ipred[INTRA_ANG_XY_14] = intra_pred_ang_xy_14_avx;
        ipred[INTRA_ANG_XY_16] = intra_pred_ang_xy_16_avx;
        ipred[INTRA_ANG_XY_18] = intra_pred_ang_xy_18_avx;
        ipred[INTRA_ANG_XY_20] = intra_pred_ang_xy_20_avx;
        ipred[INTRA_ANG_XY_22] = intra_pred_ang_xy_22_avx;
        ipred[INTRA_ANG_XY_23] = intra_pred_ang_xy_23_avx;

        ipred[INTRA_ANG_Y_25 ] = intra_pred_ang_y_25_avx;
        ipred[INTRA_ANG_Y_26 ] = intra_pred_ang_y_26_avx;
        ipred[INTRA_ANG_Y_28 ] = intra_pred_ang_y_28_avx;
        ipred[INTRA_ANG_Y_30 ] = intra_pred_ang_y_30_avx;
        ipred[INTRA_ANG_Y_31 ] = intra_pred_ang_y_31_avx;
        ipred[INTRA_ANG_Y_32 ] = intra_pred_ang_y_32_avx;

    }
#endif //if HAVE_MMX
#undef ANG_X_OFFSET
#undef ANG_XY_OFFSET
#undef ANG_Y_OFFSET
}
