/*
 * filter_deblock.h
 *
 * Description of this file:
 *    Deblock filter functions definition of the xavs2 library
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
 * global/local variables
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
const uint8_t tab_deblock_alpha[64] = {
     0,  0,  0,  0,  0,  0,  1,  1,
     1,  1,  1,  2,  2,  2,  3,  3,
     4,  4,  5,  5,  6,  7,  8,  9,
    10, 11, 12, 13, 15, 16, 18, 20,
    22, 24, 26, 28, 30, 33, 33, 35,
    35, 36, 37, 37, 39, 39, 42, 44,
    46, 48, 50, 52, 53, 54, 55, 56,
    57, 58, 59, 60, 61, 62, 63, 64
};

/* ---------------------------------------------------------------------------
 */
const uint8_t tab_deblock_beta[64] = {
     0,  0,  0,  0,  0,  0,  1,  1,
     1,  1,  1,  1,  1,  2,  2,  2,
     2,  2,  3,  3,  3,  3,  4,  4,
     4,  4,  5,  5,  5,  5,  6,  6,
     6,  7,  7,  7,  8,  8,  8,  9,
     9, 10, 10, 11, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22,
    23, 23, 24, 24, 25, 25, 26, 27
};

/* ---------------------------------------------------------------------------
 * edge direction for deblock
 */
enum edge_direction_e {
    EDGE_HOR = 1,           /* horizontal */
    EDGE_VER = 0            /* vertical */
};

/* ---------------------------------------------------------------------------
 * edge type for fitler control
 */
enum edge_type_e {
    EDGE_TYPE_NOFILTER  = 0,  /* no deblock filter */
    EDGE_TYPE_ONLY_LUMA = 1,  /* TU boundary in CU (chroma block does not have such boundaries) */
    EDGE_TYPE_BOTH      = 2   /* CU boundary and PU boundary */
};


/**
 * ===========================================================================
 * function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static void lf_set_edge_filter_param(xavs2_t *h, int i_level, int scu_x, int scu_y, int dir, int edge_type)
{
    const int w_in_scu = h->i_width_in_mincu;
    // const int h_in_scu = h->i_height_in_mincu;
    const int y_in_lcu = scu_y - h->lcu.i_scu_y;
    int scu_num  = 1 << (i_level - MIN_CU_SIZE_IN_BIT);
    int i;

    if (dir == EDGE_VER) {
        /* set flag of vertical edges */
        if (scu_x == 0) {
            return;
        }
        /* TODO: Is left border Slice border?
         */

        /* set filter type */
        // scu_num = XAVS2_MIN(scu_num, h_in_scu - scu_y);
        for (i = 0; i < scu_num; i++) {
            if (h->p_deblock_flag[EDGE_VER][(y_in_lcu + i) * w_in_scu + scu_x] != EDGE_TYPE_NOFILTER) {
                break;
            }
            h->p_deblock_flag[EDGE_VER][(y_in_lcu + i) * w_in_scu + scu_x] = (uint8_t)edge_type;
        }
    } else {
        /* set flag of horizontal edges */
        if (scu_y == 0) {
            return;
        }

        /* Is this border a slice border inside the picture? */
        if (cu_get_slice_index(h, scu_x, scu_y) != cu_get_slice_index(h, scu_x, scu_y - 1)) {
            if (!h->param->b_cross_slice_loop_filter) {
                return;
            }
        }

        /* set filter type */
        // scu_num = XAVS2_MIN(scu_num, w_in_scu - scu_x);
        for (i = 0; i < scu_num; i++) {
            if (h->p_deblock_flag[EDGE_HOR][y_in_lcu * w_in_scu + scu_x + i] != EDGE_TYPE_NOFILTER) {
                break;
            }
            h->p_deblock_flag[EDGE_HOR][y_in_lcu * w_in_scu + scu_x + i] = (uint8_t)edge_type;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static
void lf_lcu_set_edge_filter(xavs2_t *h, int i_level, int scu_x, int scu_y, int scu_xy)
{
    cu_info_t *p_cu_info = &h->cu_info[scu_xy];
    int i;

    assert(p_cu_info->i_level >= MIN_CU_SIZE_IN_BIT);

    if (p_cu_info->i_level < i_level) {
        const int w_in_scu = h->i_width_in_mincu;
        const int h_in_scu = h->i_height_in_mincu;
        // 4 sub-cu
        for (i = 0; i < 4; i++) {
            int sub_cu_x = (i  & 1) << (i_level - MIN_CU_SIZE_IN_BIT - 1);
            int sub_cu_y = (i >> 1) << (i_level - MIN_CU_SIZE_IN_BIT - 1);
            int pos;

            if (scu_x + sub_cu_x >= w_in_scu || scu_y + sub_cu_y >= h_in_scu) {
                continue;       // is outside of the frame
            }

            pos = scu_xy + sub_cu_y * w_in_scu + sub_cu_x;
            lf_lcu_set_edge_filter(h, i_level - 1, scu_x + sub_cu_x, scu_y + sub_cu_y, pos);
        }
    } else {
        // set the first left and top edge filter parameters
        lf_set_edge_filter_param(h, i_level, scu_x, scu_y, EDGE_VER, EDGE_TYPE_BOTH);  // left edge
        lf_set_edge_filter_param(h, i_level, scu_x, scu_y, EDGE_HOR, EDGE_TYPE_BOTH);  // top  edge

        // set other edge filter parameters
        if (p_cu_info->i_level > MIN_CU_SIZE_IN_BIT) {
            /* set prediction boundary */
            i = i_level - MIN_CU_SIZE_IN_BIT - 1;
            switch (p_cu_info->i_mode) {
            case PRED_2NxN:
                lf_set_edge_filter_param(h, i_level, scu_x, scu_y + (1 << i), EDGE_HOR, EDGE_TYPE_BOTH);
                break;
            case PRED_Nx2N:
                lf_set_edge_filter_param(h, i_level, scu_x + (1 << i), scu_y, EDGE_VER, EDGE_TYPE_BOTH);
                break;
            case PRED_I_NxN:
                lf_set_edge_filter_param(h, i_level, scu_x + (1 << i), scu_y, EDGE_VER, EDGE_TYPE_BOTH);
                lf_set_edge_filter_param(h, i_level, scu_x, scu_y + (1 << i), EDGE_HOR, EDGE_TYPE_BOTH);
                break;
            case PRED_I_2Nxn:
                if (i > 0) {
                    lf_set_edge_filter_param(h, i_level, scu_x, scu_y + (1 << (i - 1)),     EDGE_HOR, EDGE_TYPE_ONLY_LUMA);
                    lf_set_edge_filter_param(h, i_level, scu_x, scu_y + (1 << (i - 1)) * 2, EDGE_HOR, EDGE_TYPE_ONLY_LUMA);
                    lf_set_edge_filter_param(h, i_level, scu_x, scu_y + (1 << (i - 1)) * 3, EDGE_HOR, EDGE_TYPE_ONLY_LUMA);
                } else {
                    lf_set_edge_filter_param(h, i_level, scu_x, scu_y + (1 << (i    )),     EDGE_HOR, EDGE_TYPE_ONLY_LUMA);
                }
                break;
            case PRED_I_nx2N:
                if (i > 0) {
                    lf_set_edge_filter_param(h, i_level, scu_x + (1 << (i - 1)),     scu_y, EDGE_VER, EDGE_TYPE_ONLY_LUMA);
                    lf_set_edge_filter_param(h, i_level, scu_x + (1 << (i - 1)) * 2, scu_y, EDGE_VER, EDGE_TYPE_ONLY_LUMA);
                    lf_set_edge_filter_param(h, i_level, scu_x + (1 << (i - 1)) * 3, scu_y, EDGE_VER, EDGE_TYPE_ONLY_LUMA);
                } else {
                    lf_set_edge_filter_param(h, i_level, scu_x + (1 << (i    )),     scu_y, EDGE_VER, EDGE_TYPE_ONLY_LUMA);
                }
                break;
            case PRED_2NxnU:
                if (i > 0) {
                    lf_set_edge_filter_param(h, i_level, scu_x, scu_y + (1 << (i - 1)), EDGE_HOR, EDGE_TYPE_BOTH);
                }
                break;
            case PRED_2NxnD:
                if (i > 0) {
                    lf_set_edge_filter_param(h, i_level, scu_x, scu_y + (1 << (i - 1)) * 3, EDGE_HOR, EDGE_TYPE_BOTH);
                }
                break;
            case PRED_nLx2N:
                if (i > 0) {
                    lf_set_edge_filter_param(h, i_level, scu_x + (1 << (i - 1)), scu_y, EDGE_VER, EDGE_TYPE_BOTH);
                }
                break;
            case PRED_nRx2N:
                if (i > 0) {
                    lf_set_edge_filter_param(h, i_level, scu_x + (1 << (i - 1)) * 3, scu_y, EDGE_VER, EDGE_TYPE_BOTH);
                }
                break;
            default:
                // for other modes: direct/skip, 2Nx2N inter, 2Nx2N intra, no need to set
                break;
            }

            /* set transform block boundary */
            if (p_cu_info->i_mode != PRED_I_NxN && p_cu_info->i_tu_split && p_cu_info->i_cbp != 0) {
                if (h->param->enable_nsqt && IS_HOR_PU_PART(p_cu_info->i_mode)) {
                    if (p_cu_info->i_level == B16X16_IN_BIT) {
                        lf_set_edge_filter_param(h, i_level, scu_x, scu_y + (1 << (i    )),                  EDGE_HOR, EDGE_TYPE_ONLY_LUMA);
                    } else {
                        lf_set_edge_filter_param(h, i_level, scu_x, scu_y + (1 << (i - 1)),                  EDGE_HOR, EDGE_TYPE_ONLY_LUMA);
                        lf_set_edge_filter_param(h, i_level, scu_x, scu_y + (1 << (i    )),                  EDGE_HOR, EDGE_TYPE_ONLY_LUMA);
                        lf_set_edge_filter_param(h, i_level, scu_x, scu_y + (1 << (i    )) + (1 << (i - 1)), EDGE_HOR, EDGE_TYPE_ONLY_LUMA);
                    }
                } else if (h->param->enable_nsqt && IS_VER_PU_PART(p_cu_info->i_mode)) {
                    if (p_cu_info->i_level == B16X16_IN_BIT) {
                        lf_set_edge_filter_param(h, i_level, scu_x + (1 << (i    )),                  scu_y, EDGE_VER, EDGE_TYPE_ONLY_LUMA);
                    } else {
                        lf_set_edge_filter_param(h, i_level, scu_x + (1 << (i - 1)),                  scu_y, EDGE_VER, EDGE_TYPE_ONLY_LUMA);
                        lf_set_edge_filter_param(h, i_level, scu_x + (1 << (i    )),                  scu_y, EDGE_VER, EDGE_TYPE_ONLY_LUMA);
                        lf_set_edge_filter_param(h, i_level, scu_x + (1 << (i    )) + (1 << (i - 1)), scu_y, EDGE_VER, EDGE_TYPE_ONLY_LUMA);
                    }
                } else {
                    lf_set_edge_filter_param(h, i_level, scu_x + (1 << i), scu_y, EDGE_VER, EDGE_TYPE_ONLY_LUMA);
                    lf_set_edge_filter_param(h, i_level, scu_x, scu_y + (1 << i), EDGE_HOR, EDGE_TYPE_ONLY_LUMA);
                }
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * Return 1 if skip filtering is needed
 */
static INLINE
uint8_t lf_skip_filter(xavs2_t *h, cu_info_t *MbP, cu_info_t *MbQ, int dir, int block_x, int block_y)
{
    if (h->i_type == SLICE_TYPE_P || h->i_type == SLICE_TYPE_F) {
        const mv_t *p_mv_buf = h->fwd_1st_mv;
        const int8_t *p_ref_buf = h->fwd_1st_ref;
        int w_in_4x4 = h->i_width_in_minpu;
        int block_x2 = block_x - !dir;
        int block_y2 = block_y -  dir;
        int pos1 = block_y  * w_in_4x4 + block_x;
        int pos2 = block_y2 * w_in_4x4 + block_x2;

        if ((MbP->i_cbp == 0) && (MbQ->i_cbp == 0) &&
            (XAVS2_ABS(p_mv_buf[pos1].x - p_mv_buf[pos2].x) < 4) &&
            (XAVS2_ABS(p_mv_buf[pos1].y - p_mv_buf[pos2].y) < 4) &&
            (p_ref_buf[pos1] != INVALID_REF && p_ref_buf[pos1] == p_ref_buf[pos2])) {
            return 0;
        }
    }

    return 1;
}

/* ---------------------------------------------------------------------------
 */
static void lf_edge_core(pel_t *src, int b_chroma, int ptr_inc, int inc1, int alpha, int beta, uint8_t *flt_flag)
{
    int pel;
    int abs_delta;
    int L2, L1, L0, R0, R1, R2;
    int fs; // fs stands for filtering strength. The larger fs is, the stronger filter is applied.
    int FlatnessL, FlatnessR;
    int inc2, inc3;
    int flag = 0;

    inc2 = inc1 << 1;
    inc3 = inc1 + inc2;
    for (pel = 0; pel < MIN_CU_SIZE; pel++) {
        L2 = src[-inc3];
        L1 = src[-inc2];
        L0 = src[-inc1];
        R0 = src[    0];
        R1 = src[ inc1];
        R2 = src[ inc2];

        abs_delta = XAVS2_ABS(R0 - L0);
        flag = (pel < 4) ? flt_flag[0] : flt_flag[1];
        if (flag && (abs_delta < alpha) && (abs_delta > 1)) {
            FlatnessL = (XAVS2_ABS(L1 - L0) < beta) ? 2 : 0;
            if (XAVS2_ABS(L2 - L0) < beta) {
                FlatnessL += 1;
            }

            FlatnessR = (XAVS2_ABS(R0 - R1) < beta) ? 2 : 0;
            if (XAVS2_ABS(R0 - R2) < beta) {
                FlatnessR += 1;
            }

            switch (FlatnessL + FlatnessR) {
            case 6:
                fs = (R1 == R0 && L0 == L1) ? 4 : 3;
                break;
            case 5:
                fs = (R1 == R0 && L0 == L1) ? 3 : 2;
                break;
            case 4:
                fs = (FlatnessL == 2) ? 2 : 1;
                break;
            case 3:
                fs = (XAVS2_ABS(L1 - R1) < beta) ? 1 : 0;
                break;
            default:
                fs = 0;
                break;
            }

            if (b_chroma && fs > 0) {
                fs--;
            }

            switch (fs) {
            case 4:
                src[-inc1] = (pel_t)((L0 + ((L0 + L2) << 3) + L2 + (R0 << 3) + (R2 << 2) + (R2 << 1) + 16) >> 5);   // L0
                src[-inc2] = (pel_t)(((L0 << 3) - L0 + (L2 << 2) + (L2 << 1) + R0 + (R0 << 1) + 8) >> 4);           // L1
                src[-inc3] = (pel_t)(((L0 << 2) + L2 + (L2 << 1) + R0 + 4) >> 3);                                   // L2
                src[    0] = (pel_t)((R0 + ((R0 + R2) << 3) + R2 + (L0 << 3) + (L2 << 2) + (L2 << 1) + 16) >> 5);   // R0
                src[ inc1] = (pel_t)(((R0 << 3) - R0 + (R2 << 2) + (R2 << 1) + L0 + (L0 << 1) + 8) >> 4);           // R1
                src[ inc2] = (pel_t)(((R0 << 2) + R2 + (R2 << 1) + L0 + 4) >> 3);                                   // R2
                break;
            case 3:
                src[-inc1] = (pel_t)((L2 + (L1 << 2) + (L0 << 2) + (L0 << 1) + (R0 << 2) + R1 + 8) >> 4);   // L0
                src[    0] = (pel_t)((L1 + (L0 << 2) + (R0 << 2) + (R0 << 1) + (R1 << 2) + R2 + 8) >> 4);   // R0
                src[-inc2] = (pel_t)((L2 * 3 + L1 * 8 + L0 * 4 + R0 + 8) >> 4);
                src[ inc1] = (pel_t)((R2 * 3 + R1 * 8 + R0 * 4 + L0 + 8) >> 4);
                break;
            case 2:
                src[-inc1] = (pel_t)(((L1 << 1) + L1 + (L0 << 3) + (L0 << 1) + (R0 << 1) + R0 + 8) >> 4);
                src[    0] = (pel_t)(((L0 << 1) + L0 + (R0 << 3) + (R0 << 1) + (R1 << 1) + R1 + 8) >> 4);
                break;
            case 1:
                src[-inc1] = (pel_t)((L0 * 3 + R0 + 2) >> 2);
                src[    0] = (pel_t)((R0 * 3 + L0 + 2) >> 2);
                break;
            default:
                break;
            }
        }

        src += ptr_inc;    // next row or column
        pel += b_chroma;
    }
}

/* ---------------------------------------------------------------------------
 */
static void deblock_edge_hor(pel_t *src, int stride, int alpha, int beta, uint8_t *flt_flag)
{
    lf_edge_core(src, 0, 1, stride, alpha, beta, flt_flag);
}

/* ---------------------------------------------------------------------------
 */
static void deblock_edge_ver(pel_t *src, int stride, int alpha, int beta, uint8_t *flt_flag)
{
    lf_edge_core(src, 0, stride, 1, alpha, beta, flt_flag);
}

/* ---------------------------------------------------------------------------
 */
static void deblock_edge_ver_c(pel_t *src_u, pel_t *src_v, int stride, int alpha, int beta, uint8_t *flt_flag)
{
    lf_edge_core(src_u, 1, stride, 1, alpha, beta, flt_flag);
    lf_edge_core(src_v, 1, stride, 1, alpha, beta, flt_flag);
}

/* ---------------------------------------------------------------------------
 */
static void deblock_edge_hor_c(pel_t *src_u, pel_t *src_v, int stride, int alpha, int beta, uint8_t *flt_flag)
{
    lf_edge_core(src_u, 1, 1, stride, alpha, beta, flt_flag);
    lf_edge_core(src_v, 1, 1, stride, alpha, beta, flt_flag);
}

/* ---------------------------------------------------------------------------
 */
static
void lf_scu_deblock(xavs2_t *h, pel_t *p_rec[3], int i_stride, int i_stride_c, int scu_x, int scu_y, int dir)
{
    static const int max_qp_deblock = 63;
    cu_info_t *MbQ = &h->cu_info[scu_y * h->i_width_in_mincu + scu_x];  /* current SCU */
    int edge_type = h->p_deblock_flag[dir][(scu_y - h->lcu.i_scu_y) * h->i_width_in_mincu + scu_x];

    if (edge_type != EDGE_TYPE_NOFILTER) {
        pel_t *src_y = p_rec[0] + (scu_y << MIN_CU_SIZE_IN_BIT) * i_stride + (scu_x << MIN_CU_SIZE_IN_BIT);
        cu_info_t *MbP = dir ? (MbQ - h->i_width_in_mincu) : (MbQ - 1); /* MbP = Mb of the remote 4x4 block */
        int QP = (cu_get_qp(h, MbP) + cu_get_qp(h, MbQ) + 1) >> 1;                /* average QP of the two blocks */
        int shift = h->param->sample_bit_depth - 8;
        int offset = shift << 3;  /* coded as 10/12 bit, QP is added by (8 * (h->param->sample_bit_depth - 8)) in config file */
        int alpha, beta;
        uint8_t b_filter_edge[2];

        b_filter_edge[0] = lf_skip_filter(h, MbP, MbQ, dir, (scu_x << 1), (scu_y << 1));
        b_filter_edge[1] = lf_skip_filter(h, MbP, MbQ, dir, (scu_x << 1) + dir, (scu_y << 1) + !dir);

        if (b_filter_edge[0] == 0 && b_filter_edge[1] == 0) {
            return;
        }

        /* deblock luma edge */
        alpha = tab_deblock_alpha[XAVS2_CLIP3(0, max_qp_deblock, QP - offset + h->param->alpha_c_offset)] << shift;
        beta  = tab_deblock_beta [XAVS2_CLIP3(0, max_qp_deblock, QP - offset + h->param->beta_offset)] << shift;

        g_funcs.deblock_luma[dir](src_y, i_stride, alpha, beta, b_filter_edge);

        assert(h->param->chroma_format == CHROMA_420 || h->param->chroma_format == CHROMA_400);   /* only support I420/I400 now */
        /* deblock chroma edge */
        if (edge_type == EDGE_TYPE_BOTH && h->param->chroma_format == CHROMA_420)
            if ((((scu_y & 1) == 0) && dir) || (((scu_x & 1) == 0) && (!dir))) {
                pel_t *src_u = p_rec[1] + (scu_y << (MIN_CU_SIZE_IN_BIT - 1)) * i_stride_c + (scu_x << (MIN_CU_SIZE_IN_BIT - 1));
                pel_t *src_v = p_rec[2] + (scu_y << (MIN_CU_SIZE_IN_BIT - 1)) * i_stride_c + (scu_x << (MIN_CU_SIZE_IN_BIT - 1));

                int alpha_c, beta_c;
                QP = cu_get_chroma_qp(h, QP, 0) - offset;
                alpha_c = tab_deblock_alpha[XAVS2_CLIP3(0, max_qp_deblock, QP + h->param->alpha_c_offset)] << shift;
                beta_c  = tab_deblock_beta [XAVS2_CLIP3(0, max_qp_deblock, QP + h->param->beta_offset)] << shift;
                g_funcs.deblock_chroma[dir](src_u, src_v, i_stride_c, alpha_c, beta_c, b_filter_edge);
            }
    }
}

/**
 * ===========================================================================
 * interface function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
void xavs2_lcu_deblock(xavs2_t *h, xavs2_frame_t *frm)
{
    const int i_stride   = frm->i_stride[0];
    const int i_stride_c = frm->i_stride[1];
    const int w_in_scu   = h->i_width_in_mincu;
    int scu_x = h->lcu.i_scu_x;
    int scu_y = h->lcu.i_scu_y;
    int num_of_scu_hor = h->lcu.i_pix_width  >> MIN_CU_SIZE_IN_BIT;
    int num_of_scu_ver = h->lcu.i_pix_height >> MIN_CU_SIZE_IN_BIT;
    uint8_t *p_fbuf0 = h->p_deblock_flag[0] + scu_x;
    uint8_t *p_fbuf1 = h->p_deblock_flag[1] + scu_x;
    int i, j;

    /* clear edge flags in one LCU */
    int size_setzero = num_of_scu_hor * sizeof(uint8_t);
    for (j = 0; j < num_of_scu_ver; j++) {
        g_funcs.fast_memzero(p_fbuf0, size_setzero);
        g_funcs.fast_memzero(p_fbuf1, size_setzero);
        p_fbuf0 += w_in_scu;
        p_fbuf1 += w_in_scu;
    }

    /* set edge flags in one LCU */
    lf_lcu_set_edge_filter(h, h->i_lcu_level, h->lcu.i_scu_x, h->lcu.i_scu_y, h->lcu.i_scu_xy);

    /* deblock all vertical edges in one LCU */
    for (j = 0; j < num_of_scu_ver; j++) {
        for (i = 0; i < num_of_scu_hor; i++) {
            lf_scu_deblock(h, frm->planes, i_stride, i_stride_c, scu_x + i, scu_y + j, EDGE_VER);
        }
    }

    /* adjust the value of scu_x and num_of_scu_hor */
    if (scu_x == 0) {
        /* the current LCU is the first LCU in a LCU row */
        num_of_scu_hor--; /* leave the last horizontal edge */
    } else {
        /* the current LCU is one of the rest LCUs in a row */
        if (scu_x + num_of_scu_hor == w_in_scu) {
            /* the current LCU is the last LCUs in a row,
             * need deblock one horizontal edge more */
            num_of_scu_hor++;
        }
        scu_x--;        /* begin from the last horizontal edge of previous LCU */
    }

    /* deblock all horizontal edges in one LCU */
    for (j = 0; j < num_of_scu_ver; j++) {
        for (i = 0; i < num_of_scu_hor; i++) {
            lf_scu_deblock(h, frm->planes, i_stride, i_stride_c, scu_x + i, scu_y + j, EDGE_HOR);
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void xavs2_deblock_init(uint32_t cpuid, intrinsic_func_t* lf)
{
    lf->deblock_luma  [0] = deblock_edge_ver;
    lf->deblock_luma  [1] = deblock_edge_hor;
    lf->deblock_chroma[0] = deblock_edge_ver_c;
    lf->deblock_chroma[1] = deblock_edge_hor_c;

#if HAVE_MMX
    if (cpuid & XAVS2_CPU_SSE42) {
        lf->deblock_luma[0] = deblock_edge_ver_sse128;
        lf->deblock_luma[1] = deblock_edge_hor_sse128;
        // lf->deblock_chroma[0] = deblock_edge_ver_c_sse128;
        // lf->deblock_chroma[1] = deblock_edge_hor_c_sse128;
    }
    if (cpuid & XAVS2_CPU_AVX2) {
        // In some machines, avx is slower than SSE
        // lf->deblock_luma[0]   = deblock_edge_ver_avx2;
        // lf->deblock_luma[1]   = deblock_edge_hor_avx2;
        // lf->deblock_chroma[0] = deblock_edge_ver_c_avx2;
        // lf->deblock_chroma[1] = deblock_edge_hor_c_avx2;
    }
#else
    UNUSED_PARAMETER(cpuid);
#endif
}
