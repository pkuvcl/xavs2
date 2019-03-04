/*
 * filter_alf.h
 *
 * Description of this file:
 *    ALF filter functions definition of the xavs2 library
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
#include "filter.h"
#include "cudata.h"
#include "cpu.h"

/**
 * ===========================================================================
 * function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static
void alf_filter_block1(pel_t *p_dst, int i_dst, pel_t *p_src, int i_src,
                       int lcu_pix_x, int lcu_pix_y, int lcu_width, int lcu_height,
                       int *alf_coeff, int b_top_avail, int b_down_avail)
{
    const int pel_add  = 1 << (ALF_NUM_BIT_SHIFT - 1);
    int startPos = b_top_avail  ? (lcu_pix_y - 4) : lcu_pix_y;
    int endPos   = b_down_avail ? (lcu_pix_y + lcu_height - 4) : (lcu_pix_y + lcu_height);
    int xPosEnd  = lcu_pix_x + lcu_width;
    int min_x    = lcu_pix_x - 3;
    int max_x    = xPosEnd - 1 + 3;
    int yUp, yBottom;
    int xLeft, xRight;
    int x, y, pel_val;
    pel_t *p_src1, *p_src2, *p_src3, *p_src4, *p_src5, *p_src6;

    p_src += (startPos * i_src);
    p_dst += (startPos * i_dst);

    for (y = startPos; y < endPos; y++) {
        yUp     = XAVS2_CLIP3(startPos, endPos - 1, y - 1);
        yBottom = XAVS2_CLIP3(startPos, endPos - 1, y + 1);
        p_src1 = p_src + (yBottom - y) * i_src;
        p_src2 = p_src + (yUp     - y) * i_src;

        yUp     = XAVS2_CLIP3(startPos, endPos - 1, y - 2);
        yBottom = XAVS2_CLIP3(startPos, endPos - 1, y + 2);
        p_src3 = p_src + (yBottom - y) * i_src;
        p_src4 = p_src + (yUp     - y) * i_src;

        yUp     = XAVS2_CLIP3(startPos, endPos - 1, y - 3);
        yBottom = XAVS2_CLIP3(startPos, endPos - 1, y + 3);
        p_src5 = p_src + (yBottom - y) * i_src;
        p_src6 = p_src + (yUp     - y) * i_src;

        for (x = lcu_pix_x; x < xPosEnd; x++) {
            pel_val  = alf_coeff[0] * (p_src5[x] + p_src6[x]);
            pel_val += alf_coeff[1] * (p_src3[x] + p_src4[x]);

            xLeft    = XAVS2_CLIP3(min_x, max_x, x - 1);
            xRight   = XAVS2_CLIP3(min_x, max_x, x + 1);
            pel_val += alf_coeff[2] * (p_src1[xRight] + p_src2[xLeft ]);
            pel_val += alf_coeff[3] * (p_src1[x     ] + p_src2[x     ]);
            pel_val += alf_coeff[4] * (p_src1[xLeft ] + p_src2[xRight]);
            pel_val += alf_coeff[7] * (p_src [xRight] + p_src [xLeft ]);

            xLeft    = XAVS2_CLIP3(min_x, max_x, x - 2);
            xRight   = XAVS2_CLIP3(min_x, max_x, x + 2);
            pel_val += alf_coeff[6] * (p_src [xRight] + p_src [xLeft ]);

            xLeft    = XAVS2_CLIP3(min_x, max_x, x - 3);
            xRight   = XAVS2_CLIP3(min_x, max_x, x + 3);
            pel_val += alf_coeff[5] * (p_src [xRight] + p_src [xLeft ]);
            pel_val += alf_coeff[8] * (p_src [x     ]);

            pel_val   = (pel_val + pel_add) >> ALF_NUM_BIT_SHIFT;
            p_dst[x] = (pel_t)XAVS2_CLIP1(pel_val);
        }

        p_src += i_src;
        p_dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
static
void alf_filter_block2(pel_t *p_dst, int i_dst, pel_t *p_src, int i_src,
                       int lcu_pix_x, int lcu_pix_y, int lcu_width, int lcu_height,
                       int *alf_coeff, int b_top_avail, int b_down_avail)
{
    pel_t *p_src1, *p_src2, *p_src3, *p_src4, *p_src5, *p_src6;
    int pixelInt;
    int startPos = b_top_avail ? (lcu_pix_y - 4) : lcu_pix_y;
    int endPos = b_down_avail ? (lcu_pix_y + lcu_height - 4) : (lcu_pix_y + lcu_height);

    /* first line */
    p_src += (startPos * i_src) + lcu_pix_x;
    p_dst += (startPos * i_dst) + lcu_pix_x;

    if (p_src[0] != p_src[-1]) {
        p_src1 = p_src + 1 * i_src;
        p_src2 = p_src;
        p_src3 = p_src + 2 * i_src;
        p_src4 = p_src;
        p_src5 = p_src + 3 * i_src;
        p_src6 = p_src;

        pixelInt  = alf_coeff[0] * (p_src5[ 0] + p_src6[ 0]);
        pixelInt += alf_coeff[1] * (p_src3[ 0] + p_src4[ 0]);
        pixelInt += alf_coeff[2] * (p_src1[ 1] + p_src2[ 0]);
        pixelInt += alf_coeff[3] * (p_src1[ 0] + p_src2[ 0]);
        pixelInt += alf_coeff[4] * (p_src1[-1] + p_src2[ 1]);
        pixelInt += alf_coeff[7] * (p_src [ 1] + p_src [-1]);
        pixelInt += alf_coeff[6] * (p_src [ 2] + p_src [-2]);
        pixelInt += alf_coeff[5] * (p_src [ 3] + p_src [-3]);
        pixelInt += alf_coeff[8] * (p_src [ 0]);

        pixelInt = (int)((pixelInt + 32) >> 6);
        p_dst[0] = (pel_t)XAVS2_CLIP1(pixelInt);
    }

    p_src += lcu_width - 1;
    p_dst += lcu_width - 1;

    if (p_src[0] != p_src[1]) {
        p_src1 = p_src + 1 * i_src;
        p_src2 = p_src;
        p_src3 = p_src + 2 * i_src;
        p_src4 = p_src;
        p_src5 = p_src + 3 * i_src;
        p_src6 = p_src;

        pixelInt  = alf_coeff[0] * (p_src5[ 0] + p_src6[ 0]);
        pixelInt += alf_coeff[1] * (p_src3[ 0] + p_src4[ 0]);
        pixelInt += alf_coeff[2] * (p_src1[ 1] + p_src2[-1]);
        pixelInt += alf_coeff[3] * (p_src1[ 0] + p_src2[ 0]);
        pixelInt += alf_coeff[4] * (p_src1[-1] + p_src2[ 0]);
        pixelInt += alf_coeff[7] * (p_src [ 1] + p_src [-1]);
        pixelInt += alf_coeff[6] * (p_src [ 2] + p_src [-2]);
        pixelInt += alf_coeff[5] * (p_src [ 3] + p_src [-3]);
        pixelInt += alf_coeff[8] * (p_src [ 0]);

        pixelInt = (int)((pixelInt + 32) >> 6);
        p_dst[0] = (pel_t)XAVS2_CLIP1(pixelInt);
    }

    /* last line */
    p_src -= lcu_width - 1;
    p_dst -= lcu_width - 1;
    p_src += ((endPos - startPos - 1) * i_src);
    p_dst += ((endPos - startPos - 1) * i_dst);

    if (p_src[0] != p_src[-1]) {
        p_src1 = p_src;
        p_src2 = p_src - 1 * i_src;
        p_src3 = p_src;
        p_src4 = p_src - 2 * i_src;
        p_src5 = p_src;
        p_src6 = p_src - 3 * i_src;

        pixelInt  = alf_coeff[0] * (p_src5[ 0] + p_src6[ 0]);
        pixelInt += alf_coeff[1] * (p_src3[ 0] + p_src4[ 0]);
        pixelInt += alf_coeff[2] * (p_src1[ 1] + p_src2[-1]);
        pixelInt += alf_coeff[3] * (p_src1[ 0] + p_src2[ 0]);
        pixelInt += alf_coeff[4] * (p_src1[ 0] + p_src2[ 1]);
        pixelInt += alf_coeff[7] * (p_src [ 1] + p_src [-1]);
        pixelInt += alf_coeff[6] * (p_src [ 2] + p_src [-2]);
        pixelInt += alf_coeff[5] * (p_src [ 3] + p_src [-3]);
        pixelInt += alf_coeff[8] * (p_src [ 0]);

        pixelInt = (int)((pixelInt + 32) >> 6);
        p_dst[0] = (pel_t)XAVS2_CLIP1(pixelInt);
    }

    p_src += lcu_width - 1;
    p_dst += lcu_width - 1;

    if (p_src[0] != p_src[1]) {
        p_src1 = p_src;
        p_src2 = p_src - 1 * i_src;
        p_src3 = p_src;
        p_src4 = p_src - 2 * i_src;
        p_src5 = p_src;
        p_src6 = p_src - 3 * i_src;

        pixelInt  = alf_coeff[0] * (p_src5[ 0] + p_src6[ 0]);
        pixelInt += alf_coeff[1] * (p_src3[ 0] + p_src4[ 0]);
        pixelInt += alf_coeff[2] * (p_src1[ 0] + p_src2[-1]);
        pixelInt += alf_coeff[3] * (p_src1[ 0] + p_src2[ 0]);
        pixelInt += alf_coeff[4] * (p_src1[-1] + p_src2[ 1]);
        pixelInt += alf_coeff[7] * (p_src [ 1] + p_src [-1]);
        pixelInt += alf_coeff[6] * (p_src [ 2] + p_src [-2]);
        pixelInt += alf_coeff[5] * (p_src [ 3] + p_src [-3]);
        pixelInt += alf_coeff[8] * (p_src [ 0]);

        pixelInt = (int)((pixelInt + 32) >> 6);
        p_dst[0] = (pel_t)XAVS2_CLIP1(pixelInt);
    }
}

/* ---------------------------------------------------------------------------
 */
void xavs2_alf_init(uint32_t cpuid, intrinsic_func_t *pf)
{
    /* set function handles */
    pf->alf_flt[0] = alf_filter_block1;
    pf->alf_flt[1] = alf_filter_block2;
#if HAVE_MMX
    if (cpuid & XAVS2_CPU_SSE42) {
        pf->alf_flt[0] = alf_flt_one_block_sse128;
    }
#else
    UNUSED_PARAMETER(cpuid);
#endif
}
