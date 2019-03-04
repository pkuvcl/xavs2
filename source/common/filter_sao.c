/*
 * filter_sao.h
 *
 * Description of this file:
 *    SAO filter functions definition of the xavs2 library
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
static void sao_block_c(pel_t *p_dst, int i_dst, pel_t *p_src, int i_src,
                        int i_block_w, int i_block_h,
                        int *lcu_avail, SAOBlkParam *sao_param)
{
    int8_t SIGN_BUF[MAX_CU_SIZE + 32];  // sign of top line
    int8_t *UPROW_S = SIGN_BUF + 16;
    int  *sao_offset = sao_param->offset;
    const int max_pel_val = (1 << g_bit_depth) - 1;
    int reg = 0;
    int sx, sy, ex, ey;               // start/end (x, y)
    int sx_0, ex_0, sx_n, ex_n;       // start/end x for first and last row
    int left_sign, right_sign, top_sign, down_sign;
    int edge_type;
    int pel_diff;
    int x, y;

    assert(sao_param->typeIdc != SAO_TYPE_OFF);
    switch (sao_param->typeIdc) {
    case SAO_TYPE_EO_0:
        sx = lcu_avail[SAO_L] ? 0 : 1;
        ex = lcu_avail[SAO_R] ? i_block_w : (i_block_w - 1);
        for (y = 0; y < i_block_h; y++) {
            left_sign = xavs2_sign3(p_src[sx] - p_src[sx - 1]);
            for (x = sx; x < ex; x++) {
                right_sign = xavs2_sign3(p_src[x] - p_src[x + 1]);
                edge_type = left_sign + right_sign + 2;
                left_sign = -right_sign;
                p_dst[x] = (pel_t)XAVS2_CLIP3(0, max_pel_val, p_src[x] + sao_offset[edge_type]);
            }
            p_src += i_src;
            p_dst += i_dst;
        }
        break;
    case SAO_TYPE_EO_90: {
        sy = lcu_avail[SAO_T] ? 0 : 1;
        ey = lcu_avail[SAO_D] ? i_block_h : (i_block_h - 1);
        for (x = 0; x < i_block_w; x++) {
            pel_diff = p_src[sy * i_src + x] - p_src[(sy - 1) * i_src + x];
            top_sign = pel_diff > 0 ? 1 : (pel_diff < 0 ? -1 : 0);
            for (y = sy; y < ey; y++) {
                pel_diff = p_src[y * i_src + x] - p_src[(y + 1) * i_src + x];
                down_sign = pel_diff > 0 ? 1 : (pel_diff < 0 ? -1 : 0);
                edge_type = down_sign + top_sign + 2;
                top_sign = -down_sign;
                p_dst[y * i_dst + x] = (pel_t)XAVS2_CLIP3(0, max_pel_val, p_src[y * i_src + x] + sao_offset[edge_type]);
            }
        }
        break;
    }
    case SAO_TYPE_EO_135:
        sx = lcu_avail[SAO_L] ? 0 : 1;
        ex = lcu_avail[SAO_R] ? i_block_w : (i_block_w - 1);

        // init the line buffer
        for (x = sx; x < ex; x++) {
            pel_diff = p_src[i_src + x + 1] - p_src[x];
            top_sign = pel_diff > 0 ? 1 : (pel_diff < 0 ? -1 : 0);
            UPROW_S[x + 1] = (int8_t)top_sign;
        }

        // first row
        sx_0 = lcu_avail[SAO_TL] ? 0 : 1;
        ex_0 = lcu_avail[SAO_T] ? (lcu_avail[SAO_R] ? i_block_w : (i_block_w - 1)) : 1;
        for (x = sx_0; x < ex_0; x++) {
            pel_diff = p_src[x] - p_src[-i_src + x - 1];
            top_sign = pel_diff > 0 ? 1 : (pel_diff < 0 ? -1 : 0);
            edge_type = top_sign - UPROW_S[x + 1] + 2;
            p_dst[x] = (pel_t)XAVS2_CLIP3(0, max_pel_val, p_src[x] + sao_offset[edge_type]);
        }

        // middle rows
        for (y = 1; y < i_block_h - 1; y++) {
            p_src += i_src;
            p_dst += i_dst;
            for (x = sx; x < ex; x++) {
                if (x == sx) {
                    pel_diff = p_src[x] - p_src[-i_src + x - 1];
                    top_sign = pel_diff > 0 ? 1 : (pel_diff < 0 ? -1 : 0);
                    UPROW_S[x] = (int8_t)top_sign;
                }
                pel_diff = p_src[x] - p_src[i_src + x + 1];
                down_sign = pel_diff > 0 ? 1 : (pel_diff < 0 ? -1 : 0);
                edge_type = down_sign + UPROW_S[x] + 2;
                p_dst[x] = (pel_t)XAVS2_CLIP3(0, max_pel_val, p_src[x] + sao_offset[edge_type]);
                UPROW_S[x] = (int8_t)reg;
                reg = -down_sign;
            }
        }

        // last row
        sx_n = lcu_avail[SAO_D] ? (lcu_avail[SAO_L] ? 0 : 1) : (i_block_w - 1);
        ex_n = lcu_avail[SAO_DR] ? i_block_w : (i_block_w - 1);
        p_src += i_src;
        p_dst += i_dst;
        for (x = sx_n; x < ex_n; x++) {
            if (x == sx) {
                pel_diff = p_src[x] - p_src[-i_src + x - 1];
                top_sign = pel_diff > 0 ? 1 : (pel_diff < 0 ? -1 : 0);
                UPROW_S[x] = (int8_t)top_sign;
            }
            pel_diff = p_src[x] - p_src[i_src + x + 1];
            down_sign = pel_diff > 0 ? 1 : (pel_diff < 0 ? -1 : 0);
            edge_type = down_sign + UPROW_S[x] + 2;
            p_dst[x] = (pel_t)XAVS2_CLIP3(0, max_pel_val, p_src[x] + sao_offset[edge_type]);
        }
        break;
    case SAO_TYPE_EO_45:
        sx = lcu_avail[SAO_L] ? 0 : 1;
        ex = lcu_avail[SAO_R] ? i_block_w : (i_block_w - 1);

        // init the line buffer
        for (x = sx; x < ex; x++) {
            pel_diff = p_src[i_src + x - 1] - p_src[x];
            top_sign = pel_diff > 0 ? 1 : (pel_diff < 0 ? -1 : 0);
            UPROW_S[x - 1] = (int8_t)top_sign;
        }

        // first row
        sx_0 = lcu_avail[SAO_T] ? (lcu_avail[SAO_L] ? 0 : 1) : (i_block_w - 1);
        ex_0 = lcu_avail[SAO_TR] ? i_block_w : (i_block_w - 1);
        for (x = sx_0; x < ex_0; x++) {
            pel_diff = p_src[x] - p_src[-i_src + x + 1];
            top_sign = pel_diff > 0 ? 1 : (pel_diff < 0 ? -1 : 0);
            edge_type = top_sign - UPROW_S[x - 1] + 2;
            p_dst[x] = (pel_t)XAVS2_CLIP3(0, max_pel_val, p_src[x] + sao_offset[edge_type]);
        }

        // middle rows
        for (y = 1; y < i_block_h - 1; y++) {
            p_src += i_src;
            p_dst += i_dst;
            for (x = sx; x < ex; x++) {
                if (x == ex - 1) {
                    pel_diff = p_src[x] - p_src[-i_src + x + 1];
                    top_sign = pel_diff > 0 ? 1 : (pel_diff < 0 ? -1 : 0);
                    UPROW_S[x] = (int8_t)top_sign;
                }
                pel_diff = p_src[x] - p_src[i_src + x - 1];
                down_sign = pel_diff > 0 ? 1 : (pel_diff < 0 ? -1 : 0);
                edge_type = down_sign + UPROW_S[x] + 2;
                p_dst[x] = (pel_t)XAVS2_CLIP3(0, max_pel_val, p_src[x] + sao_offset[edge_type]);
                UPROW_S[x - 1] = (int8_t)(-down_sign);
            }
        }

        // last row
        sx_n = lcu_avail[SAO_DL] ? 0 : 1;
        ex_n = lcu_avail[SAO_D] ? (lcu_avail[SAO_R] ? i_block_w : (i_block_w - 1)) : 1;
        p_src += i_src;
        p_dst += i_dst;
        for (x = sx_n; x < ex_n; x++) {
            if (x == ex - 1) {
                pel_diff = p_src[x] - p_src[-i_src + x + 1];
                top_sign = pel_diff > 0 ? 1 : (pel_diff < 0 ? -1 : 0);
                UPROW_S[x] = (int8_t)top_sign;
            }
            pel_diff = p_src[x] - p_src[i_src + x - 1];
            down_sign = pel_diff > 0 ? 1 : (pel_diff < 0 ? -1 : 0);
            edge_type = down_sign + UPROW_S[x] + 2;
            p_dst[x] = (pel_t)XAVS2_CLIP3(0, max_pel_val, p_src[x] + sao_offset[edge_type]);
        }
        break;
    case SAO_TYPE_BO:
        pel_diff = g_bit_depth - NUM_SAO_BO_CLASSES_IN_BIT;
        for (y = 0; y < i_block_h; y++) {
            for (x = 0; x < i_block_w; x++) {
                edge_type = p_src[x] >> pel_diff;
                p_dst[x] = (pel_t)XAVS2_CLIP3(0, max_pel_val, p_src[x] + sao_offset[edge_type]);
            }
            p_src += i_src;
            p_dst += i_dst;
        }
        break;
    default:
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Not a supported SAO types.");
        assert(0);
        exit(-1);
    }
}

/* ---------------------------------------------------------------------------
 */
void xavs2_sao_init(uint32_t cpuid, intrinsic_func_t *pf)
{
    pf->sao_block = sao_block_c;
#if HAVE_MMX
    if (cpuid & XAVS2_CPU_SSE4) {
        pf->sao_block = SAO_on_block_sse128;
    }
#ifdef _MSC_VER
    if (cpuid & XAVS2_CPU_AVX2) {
        pf->sao_block = SAO_on_block_sse256;
    }
#endif // if _MSC_VER
#endif // HAVE_MMX
}
