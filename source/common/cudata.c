/*
 * cudata.c
 *
 * Description of this file:
 *    CU-Data functions definition of the xavs2 library
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
#include "cudata.h"
#include "header.h"
#include "block_info.h"
#include "transform.h"
#include "me.h"
#include "rdo.h"
#include "predict.h"
#include "bitstream.h"
#include "ratecontrol.h"


/**
 * ===========================================================================
 * local/global variables
 * ===========================================================================
 */

#if XAVS2_TRACE
extern int g_sym_count;         /* global symbol count for trace */
extern int g_bit_count;         /* global bit    count for trace */
#endif

/* ---------------------------------------------------------------------------
 */
static const uint8_t BLOCK_STEPS[MAX_PRED_MODES][2] = { // [mode][h/v]
    { 2, 2 },                   // 8x8 (PRED_SKIP   )
    { 2, 2 },                   // 8x8 (PRED_2Nx2N  )
    { 2, 1 },                   // 8x4 (PRED_2NxN   )
    { 1, 2 },                   // 4x8 (PRED_Nx2N   )
    { 2, 1 },                   // 8x2 (PRED_2NxnU  )
    { 2, 1 },                   // 8x6 (PRED_2NxnD  )
    { 1, 2 },                   // 2x8 (PRED_nLx2N  )
    { 1, 2 },                   // 6x8 (PRED_nRx2N  )
    { 2, 2 },                   // 8x8 (PRED_I_2Nx2N)
    { 1, 1 },                   // 4x4 (PRED_I_NxN  )
    { 2, 1 },                   // 8x2 (PRED_I_2Nxn )
    { 1, 2 }                    // 2x8 (PRED_I_nx2N )
};

/* ---------------------------------------------------------------------------
 */
const uint8_t tab_split_tu_pos[MAX_PRED_MODES][4][2] = { // [mode][block][x/y]
    // x0,y0     x1,y1     x2,y2     x3,y3         CU   TU0  TU1  TU2  TU3
    { { 0, 0 }, { 4, 0 }, { 0, 4 }, { 4, 4 } }, // 8x8: 4x4, 4x4, 4x4, 4x4 (PRED_SKIP   )
    { { 0, 0 }, { 4, 0 }, { 0, 4 }, { 4, 4 } }, // 8x8: 4x4, 4x4, 4x4, 4x4 (PRED_2Nx2N  )
    { { 0, 0 }, { 0, 2 }, { 0, 4 }, { 0, 6 } }, // 8x4: 8x2, 8x2, 8x2, 8x2 (PRED_2NxN   )
    { { 0, 0 }, { 2, 0 }, { 4, 0 }, { 6, 0 } }, // 4x8: 2x8, 2x8, 2x8, 2x8 (PRED_Nx2N   )
    { { 0, 0 }, { 0, 2 }, { 0, 4 }, { 0, 6 } }, // 8x2: 8x2, 8x2, 8x2, 8x2 (PRED_2NxnU  )
    { { 0, 0 }, { 0, 2 }, { 0, 4 }, { 0, 6 } }, // 8x6: 8x2, 8x2, 8x2, 8x2 (PRED_2NxnD  )
    { { 0, 0 }, { 2, 0 }, { 4, 0 }, { 6, 0 } }, // 2x8: 2x8, 2x8, 2x8, 2x8 (PRED_nLx2N  )
    { { 0, 0 }, { 2, 0 }, { 4, 0 }, { 6, 0 } }, // 6x8: 2x8, 2x8, 2x8, 2x8 (PRED_nRx2N  )
    { { 0, 0 }, { 4, 0 }, { 0, 4 }, { 4, 4 } }, // 8x8: 4x4, 4x4, 4x4, 4x4 (PRED_I_2Nx2N)
    { { 0, 0 }, { 4, 0 }, { 0, 4 }, { 4, 4 } }, // 4x4: 4x4, 4x4, 4x4, 4x4 (PRED_I_NxN  )
    { { 0, 0 }, { 0, 2 }, { 0, 4 }, { 0, 6 } }, // 8x2: 8x2, 8x2, 8x2, 8x2 (PRED_I_2Nxn )
    { { 0, 0 }, { 2, 0 }, { 4, 0 }, { 6, 0 } }  // 2x8: 2x8, 2x8, 2x8, 2x8 (PRED_I_nx2N )
};

/* ---------------------------------------------------------------------------
 */
const uint8_t tab_qp_scale_chroma[64] = {
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 42, 43, 43, 44, 44, 45, 45,
    46, 46, 47, 47, 48, 48, 48, 49, 49, 49,
    50, 50, 50, 51,
};

/**
 * ===========================================================================
 * function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void cu_mvd_derivation(xavs2_t *h, mv_t *mvd, const mv_t *mv, const mv_t *mvp)
{
    if (h->param->enable_pmvr) {
        mv_t ctr;

        ctr.x = (mvp->x >> 1) << 1;
        ctr.y = (mvp->y >> 1) << 1;
        if (XAVS2_ABS(mv->x - ctr.x) > TH_PMVR) {
            mvd->x = (int16_t)((mv->x + ctr.x + xavs2_sign2(mv->x - ctr.x) * TH_PMVR) >> 1) - mvp->x;
            mvd->y = (mv->y - ctr.y) >> 1;
        } else if (XAVS2_ABS(mv->y - ctr.y) > TH_PMVR) {
            mvd->x = (mv->x - ctr.x) >> 1;
            mvd->y = (int16_t)((mv->y + ctr.y + xavs2_sign2(mv->y - ctr.y) * TH_PMVR) >> 1) - mvp->y;
        } else {
            mvd->x = mv->x - mvp->x;
            mvd->y = mv->y - mvp->y;
        }
    } else {
        mvd->x = mv->x - mvp->x;
        mvd->y = mv->y - mvp->y;
    }
}

/* ---------------------------------------------------------------------------
 * get mvds, only for inter cu mode
 */
void cu_get_mvds(xavs2_t *h, cu_t *p_cu)
{
    int mode = p_cu->cu_info.i_mode;
    int pdir;
    int k, blk_idx;
    cu_mode_t *p_mode = cu_get_layer_mode(h, p_cu->cu_info.i_level);
    cu_mv_mode_t *p_mvmode = p_mode->mvs[mode];

    assert(IS_INTER_MODE(mode) && !IS_SKIP_MODE(mode));

    for (k = 0; k < p_cu->cu_info.num_pu; k++) {
        mv_t mv_fwd,  mv_bwd;
        mv_t mvp_fwd, mvp_bwd;
        mv_t mvd_fwd, mvd_bwd;

#if XAVS2_TRACE
        mv_fwd.v  = mv_bwd.v  = 0;
        mvp_fwd.v = mvp_bwd.v = 0;
#endif
        mvd_fwd.v = mvd_bwd.v = 0;

        blk_idx = pu_get_mv_index(mode, k);
        pdir = p_cu->cu_info.b8pdir[k];
        /* forward motion vectors */
        if (pdir != PDIR_BWD) {
            int ref_fwd = p_cu->cu_info.ref_idx_1st[k];
            mv_fwd  = p_cu->mc.mv[k][0];
            mvp_fwd = p_mvmode[blk_idx].all_mvp[ref_fwd];
            cu_mvd_derivation(h, &mvd_fwd, &mv_fwd, &mvp_fwd);
        }

        /* backward motion vectors */
        if (pdir == PDIR_BWD || pdir == PDIR_BID) { // has backward vector
            mv_bwd  = p_cu->mc.mv[k][1];
            mvp_bwd = p_mvmode[blk_idx].all_mvp[B_BWD];
            cu_mvd_derivation(h, &mvd_bwd, &mv_bwd, &mvp_bwd);
        }

        // store (oversampled) mvd
        p_cu->cu_info.mvd[0][k] = mvd_fwd;
        p_cu->cu_info.mvd[1][k] = mvd_bwd;
#if XAVS2_TRACE
        p_cu->cu_info.mv [0][k] =  mv_fwd;
        p_cu->cu_info.mvp[0][k] = mvp_fwd;
        p_cu->cu_info.mv [1][k] =  mv_bwd;
        p_cu->cu_info.mvp[1][k] = mvp_bwd;
#endif
    }
}

/* ---------------------------------------------------------------------------
 * copy one block (multi-planes)
 */
static void block_copy_x3(pel_t *p_dst[], int i_dst[], pel_t *p_src[], int i_src[], int i_width[], int i_height[], int i_planes)
{
    pel_t *dst, *src;
    int y, k;

    for (k = 0; k < i_planes; k++) {
        int i_size = i_width[k] * sizeof(pel_t);
        memcpy_t f_memcpy = i_size & 15 ? memcpy : g_funcs.memcpy_aligned;
        dst = p_dst[k];
        src = p_src[k];
        for (y = i_height[k]; y != 0; y--) {
            f_memcpy(dst, src, i_size);
            dst += i_dst[k];
            src += i_src[k];
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void
xavs2_copy_col1(pel_t *dst, pel_t *src, const int height, const int stride)
{
    int i;
    int k = 0;

    for (i = height; i != 0; i--) {
        dst[k] = src[k];
        k += stride;
    }
}

/* ---------------------------------------------------------------------------
 * cache CTU border
 */
static INLINE
void xavs2_cache_lcu_border(pel_t *p_dst, const pel_t *p_top,
                            const pel_t *p_left, int i_left,
                            int lcu_width, int lcu_height)
{
    int i;
    /* top, top-right */
    memcpy(p_dst, p_top, (2 * lcu_width + 1) * sizeof(pel_t));
    /* left */
    for (i = 1; i <= lcu_height; i++) {
        p_dst[-i] = p_left[0];
        p_left += i_left;
    }
}

/* ---------------------------------------------------------------------------
 * cache CTU border (UV components together)
 */
static INLINE
void xavs2_cache_lcu_border_uv(pel_t *p_dst_u, const pel_t *p_top_u, const pel_t *p_left_u,
                               pel_t *p_dst_v, const pel_t *p_top_v, const pel_t *p_left_v,
                               int i_left, int lcu_width, int lcu_height)
{
    int i;
    /* top, top-right */
    memcpy(p_dst_u, p_top_u, (2 * lcu_width + 1) * sizeof(pel_t));
    memcpy(p_dst_v, p_top_v, (2 * lcu_width + 1) * sizeof(pel_t));
    /* left */
    for (i = 1; i <= lcu_height; i++) {
        p_dst_u[-i] = p_left_u[0];
        p_dst_v[-i] = p_left_v[0];
        p_left_u += i_left;
        p_left_v += i_left;
    }
}

/* ---------------------------------------------------------------------------
 * start encoding a lcu (initializing)
 */
void lcu_start_init_pos(xavs2_t *h, int i_lcu_x, int i_lcu_y)
{
    const int scu_x = i_lcu_x << (h->i_lcu_level - MIN_CU_SIZE_IN_BIT);
    const int scu_y = i_lcu_y << (h->i_lcu_level - MIN_CU_SIZE_IN_BIT);
    const int pix_x = scu_x << MIN_CU_SIZE_IN_BIT;
    const int pix_y = scu_y << MIN_CU_SIZE_IN_BIT;
    int lcu_width   = 1 << h->i_lcu_level;
    int lcu_height  = 1 << h->i_lcu_level;
    cu_t  *p_cu = h->lcu.p_ctu; /* point to the CTU */
#if ENABLE_RATE_CONTROL_CU
    int w_in_scu;               /* width  in SCU of current lcu */
    int h_in_scu;               /* height in SCU of current lcu */
    int x, y;
#endif

    /* -------------------------------------------------------------
     * 1, update the coordinates for the next lcu
     */

    /* update the coordinates */
    h->lcu.i_lcu_xy = i_lcu_y * h->i_width_in_lcu + i_lcu_x;
    h->lcu.i_scu_xy = p_cu->i_scu_xy = scu_y * h->i_width_in_mincu + scu_x;
    h->lcu.i_scu_x  = p_cu->cu_info.i_scu_x  = scu_x;
    h->lcu.i_scu_y  = p_cu->cu_info.i_scu_y  = scu_y;
    h->lcu.i_pix_x  = p_cu->i_pix_x  = pix_x;
    h->lcu.i_pix_y  = p_cu->i_pix_y  = pix_y;

    /* update actual width and height */
    lcu_width  = XAVS2_MIN( lcu_width, h->i_width  - pix_x);
    lcu_height = XAVS2_MIN(lcu_height, h->i_height - pix_y);
    h->lcu.i_pix_width  = (int16_t)lcu_width;
    h->lcu.i_pix_height = (int16_t)lcu_height;

    /* -------------------------------------------------------------
     * 2, init qp for current CTU
     */

#if ENABLE_RATE_CONTROL_CU
    if (h->param->i_rc_method == XAVS2_RC_CBR_SCU) {
        h->i_qp = xavs2_rc_get_lcu_qp(h, h->fenc->i_frame, h->i_qp);
    }
#endif

    /* -------------------------------------------------------------
     * 3, init all SCU in current CTU
     */
    h->lcu_slice_idx[h->lcu.i_lcu_xy] = (int8_t)(h->i_slice_index);

#if ENABLE_RATE_CONTROL_CU
    w_in_scu = lcu_width  >> MIN_CU_SIZE_IN_BIT;
    h_in_scu = lcu_height >> MIN_CU_SIZE_IN_BIT;
    for (y = 0; y < h_in_scu; y++) {
        cu_info_t *p_cu_info = &h->cu_info[h->lcu.i_scu_xy + y * h->i_width_in_mincu];  /* point to a SCU */
        for (x = w_in_scu; x != 0; x--, p_cu_info++) {
            p_cu_info->i_delta_qp  = 0;
            p_cu_info->i_cu_qp     = (int8_t)(h->i_qp);   // needed in loop filter (even if constant QP is used)

            // reset syntax element entries in cu_info_t
            // 这些元素在编码每个LCU时会设置，所以此处不需要修改
            // p_cu_info->i_mode  = PRED_SKIP;
            // p_cu_info->i_cbp   = 0;
            // p_cu_info->i_level = MIN_CU_SIZE_IN_BIT;
        }
    }
#endif
}

/* ---------------------------------------------------------------------------
 * start encoding a lcu (initializing)
 */
void lcu_start_init_pixels(xavs2_t *h, int i_lcu_x, int i_lcu_y)
{
    const int scu_x = i_lcu_x << (h->i_lcu_level - MIN_CU_SIZE_IN_BIT);
    const int scu_y = i_lcu_y << (h->i_lcu_level - MIN_CU_SIZE_IN_BIT);
    int img_x = scu_x << MIN_CU_SIZE_IN_BIT;
    int img_y = scu_y << MIN_CU_SIZE_IN_BIT;
    int lcu_width   = h->lcu.i_pix_width;
    int lcu_height  = h->lcu.i_pix_height;
    int blk_w[3];
    int blk_h[3];
    int i_src[3];
    int i_dst[3];
    pel_t *p_src[3];
    pel_t *p_dst[3];

    /* -------------------------------------------------------------
     * 1, copy LCU pixel data from original image buffer
     */
    i_src[0] = h->fenc->i_stride[0];
    i_src[1] = h->fenc->i_stride[1];
    i_src[2] = h->fenc->i_stride[2];
    p_src[0] = h->fenc->planes[0] + (img_y     ) * i_src[0] + (img_x     );
    p_src[1] = h->fenc->planes[1] + (img_y >> 1) * i_src[1] + (img_x >> 1);
    p_src[2] = h->fenc->planes[2] + (img_y >> 1) * i_src[2] + (img_x >> 1);

    i_dst[0] = i_dst[1] = i_dst[2] = FENC_STRIDE;
    p_dst[0] = h->lcu.p_fenc[0];
    p_dst[1] = h->lcu.p_fenc[1];
    p_dst[2] = h->lcu.p_fenc[2];

    blk_w[0] = lcu_width;
    blk_h[0] = lcu_height;
    blk_w[1] = blk_w[2] = lcu_width  >> 1;
    blk_h[1] = blk_h[2] = lcu_height >> 1;
    block_copy_x3(p_dst, i_dst, p_src, i_src, blk_w, blk_h, 3);

    /* first CTU of LCU row */
    if (h->fenc->b_enable_intra || h->fenc->i_frm_type == XAVS2_TYPE_I) {
        if (img_x == 0) {
            memcpy(h->lcu.ctu_border[0].rec_top + 1, h->intra_border[0], lcu_width * 2 * sizeof(pel_t));
            memcpy(h->lcu.ctu_border[1].rec_top + 1, h->intra_border[1], lcu_width * sizeof(pel_t));
            memcpy(h->lcu.ctu_border[2].rec_top + 1, h->intra_border[2], lcu_width * sizeof(pel_t));
        } else if (h->param->i_lcurow_threads > 1) {
            /* top-right pixels */
            memcpy(h->lcu.ctu_border[0].rec_top + 1 + lcu_width,        h->intra_border[0] + img_x + lcu_width, lcu_width * sizeof(pel_t));
            memcpy(h->lcu.ctu_border[1].rec_top + 1 + (lcu_width >> 1), h->intra_border[1] + ((img_x + lcu_width) >> 1), (lcu_width >> 1) * sizeof(pel_t));
            memcpy(h->lcu.ctu_border[2].rec_top + 1 + (lcu_width >> 1), h->intra_border[2] + ((img_x + lcu_width) >> 1), (lcu_width >> 1) * sizeof(pel_t));
        }
    }
}

/* ---------------------------------------------------------------------------
 * terminate processing of the current LCU depending on the chosen slice mode
 */
void lcu_end(xavs2_t *h, int i_lcu_x, int i_lcu_y)
{
    const int img_y = h->lcu.i_pix_y;
    const int img_y_c = img_y >> 1;
    const int img_x = h->lcu.i_pix_x;
    const int img_x_c = img_x >> 1;
    const int lcu_width = h->lcu.i_pix_width;    /* width  of lcu (in pixel) */
    const int lcu_height = h->lcu.i_pix_height;   /* height of lcu (in pixel) */
    const int lcu_width_c = lcu_width >> 1;
    const int lcu_height_c = lcu_height >> 1;
    int blk_w[3];
    int blk_h[3];
    int i_src[3];
    int i_dst[3];
    pel_t *p_src[3];
    pel_t *p_dst[3];

    /* -------------------------------------------------------------
     * 1, copy decoded LCU to frame buffer
     */
    i_dst[0] = h->fdec->i_stride[0];
    i_dst[1] = h->fdec->i_stride[1];
    i_dst[2] = h->fdec->i_stride[2];
    p_dst[0] = h->fdec->planes[0] + (img_y) * i_dst[0] + (img_x);
    p_dst[1] = h->fdec->planes[1] + (img_y_c) * i_dst[1] + (img_x_c);
    p_dst[2] = h->fdec->planes[2] + (img_y_c) * i_dst[2] + (img_x_c);

    i_src[0] = i_src[1] = i_src[2] = FDEC_STRIDE;
    p_src[0] = h->lcu.p_fdec[0];
    p_src[1] = h->lcu.p_fdec[1];
    p_src[2] = h->lcu.p_fdec[2];

    blk_w[0] = lcu_width;
    blk_h[0] = lcu_height;
    blk_w[1] = blk_w[2] = lcu_width_c;
    blk_h[1] = blk_h[2] = lcu_height_c;
    block_copy_x3(p_dst, i_dst, p_src, i_src, blk_w, blk_h, 3);

    /* -------------------------------------------------------------
     * 2, backup right col and bottom row pixels for intra coding
     */
    if (h->fenc->b_enable_intra || h->fenc->i_frm_type == XAVS2_TYPE_I) {
        // backup intra pred mode of bottom 4x4 row
        int i_pred_mode_stride = h->i_width_in_minpu + 16;
        int i_pred_mode_width_in_lcu = (1 << h->i_lcu_level) >> MIN_PU_SIZE_IN_BIT;
        memcpy(h->ipredmode - i_pred_mode_stride + i_lcu_x * i_pred_mode_width_in_lcu,
               h->ipredmode + i_pred_mode_stride * (i_pred_mode_width_in_lcu - 1) + i_lcu_x * i_pred_mode_width_in_lcu,
               i_pred_mode_width_in_lcu * sizeof(int8_t));

        /* cache top and left samples for intra prediction of next CTU */
        xavs2_cache_lcu_border(h->lcu.ctu_border[0].rec_top, h->intra_border[0] + img_x + lcu_width - 1, p_src[0] + lcu_width - 1,
                               FDEC_STRIDE, lcu_width, lcu_height);
        xavs2_cache_lcu_border_uv(h->lcu.ctu_border[1].rec_top, h->intra_border[1] + img_x_c + lcu_width_c - 1, p_src[1] + lcu_width_c - 1,
                                  h->lcu.ctu_border[2].rec_top, h->intra_border[2] + img_x_c + lcu_width_c - 1, p_src[2] + lcu_width_c - 1,
                                  FDEC_STRIDE, lcu_width_c, lcu_height_c);

        /* 2.2, backup bottom row pixels */
        if (i_lcu_y < h->i_height_in_lcu - 1) {
            g_funcs.fast_memcpy(h->intra_border[0] + img_x,   p_src[0] + (lcu_height   - 1) * FDEC_STRIDE, lcu_width   * sizeof(pel_t));
            g_funcs.fast_memcpy(h->intra_border[1] + img_x_c, p_src[1] + (lcu_height_c - 1) * FDEC_STRIDE, lcu_width_c * sizeof(pel_t));
            g_funcs.fast_memcpy(h->intra_border[2] + img_x_c, p_src[2] + (lcu_height_c - 1) * FDEC_STRIDE, lcu_width_c * sizeof(pel_t));
        }
    }

}
