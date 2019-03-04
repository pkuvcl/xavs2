/*
 * predict.h
 *
 * Description of this file:
 *    Prediction functions definition of the xavs2 library
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

#ifndef XAVS2_PREDICT_H
#define XAVS2_PREDICT_H


/**
 * ===========================================================================
 * local/global variables
 * ===========================================================================
 */

static const int REF_BITS[32] = {
    1, 1, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9
};

#define REF_COST(ref_idx)    WEIGHTED_COST(h->i_lambda_factor, REF_BITS[ref_idx + 1])


/**
 * ===========================================================================
 * inline function defines
 * ===========================================================================
 */


/* ---------------------------------------------------------------------------
* determine the mv value (1/4 pixel) is legal or not
* Return: 0: out of the legal mv range;
*         1: in the legal mv range
*/
static ALWAYS_INLINE
int check_mv_range(xavs2_t *h, const mv_t *mv, int ref_idx, int pix_x, int pix_y, int bsx, int bsy)
{
    int bsize = 1 << h->i_lcu_level;        /* valid padding size */
    int min_x = -((pix_x + bsize) << 2);
    int min_y = -((pix_y + bsize) << 2);
    int max_x = ((h->i_width  - (pix_x + bsx)) + bsize) << 2;
    int max_y = ((h->i_height - (pix_y + bsy)) + bsize) << 2;

    min_x = XAVS2_MAX(min_x, h->min_mv_range[0]);
    min_y = XAVS2_MAX(min_y, h->min_mv_range[1]);
    max_x = XAVS2_MIN(max_x, h->max_mv_range[0]);
    max_y = XAVS2_MIN(max_y, h->max_mv_range[1]);

    /* 帧级并行时，当前块最大依赖的LCU行行号 */
    int dep_lcu_y = (pix_y + bsy + ((mv->y >> 2) + 4) + 4) >> h->i_lcu_level;
    int dep_lcu_x = (pix_x + bsx + ((mv->x >> 2) + 4) + 4) >> h->i_lcu_level;
    int dep_lcu_row_avail;

    dep_lcu_y = XAVS2_MAX(0, dep_lcu_y);
    dep_lcu_x = XAVS2_MAX(0, dep_lcu_x);
    dep_lcu_y = XAVS2_MIN(h->i_height_in_lcu - 1, dep_lcu_y);
    dep_lcu_x = XAVS2_MIN(h->i_width_in_lcu - 1, dep_lcu_x);
    dep_lcu_row_avail = h->fref[ref_idx]->num_lcu_coded_in_row[dep_lcu_y] > dep_lcu_x;

    return dep_lcu_row_avail && (mv->x <= max_x && mv->x >= min_x && mv->y <= max_y && mv->y >= min_y);
}

/* ---------------------------------------------------------------------------
 * get distance for a reference frame
 */
static ALWAYS_INLINE int calculate_distance(xavs2_t *h, int blkref)
{
    assert(blkref >= 0 && blkref < MAX_REFS);
    return h->fdec->ref_dpoc[blkref];
}


/* ---------------------------------------------------------------------------
* 用于场编码中Y分量缩放的偏置
*/
static ALWAYS_INLINE void getDeltas(xavs2_t *h, int *delt, int *delt2, int OriPOC, int OriRefPOC, int ScaledPOC, int ScaledRefPOC)
{
    const int factor = 2;

    if (h->b_field_sequence == 0) {
        *delt = *delt2 = 0;
        return;
    }

    OriPOC = (OriPOC + 512) & 511;    // % 512
    OriRefPOC = (OriRefPOC + 512) & 511;
    ScaledPOC = (ScaledPOC + 512) & 511;
    ScaledRefPOC = (ScaledRefPOC + 512) & 511;
    assert((OriPOC % factor) + (OriRefPOC % factor) + (ScaledPOC % factor) + (ScaledRefPOC % factor) == 0);

    OriPOC /= factor;
    OriRefPOC /= factor;
    ScaledPOC /= factor;
    ScaledRefPOC /= factor;

    if (h->b_top_field) {   // scaled is top field
        *delt2 = (ScaledRefPOC & 1) != (ScaledPOC & 1) ? 2 : 0;

        if ((ScaledPOC & 1) == (OriPOC & 1)) { // ori is top
            *delt = (OriRefPOC & 1) != (OriPOC & 1) ? 2 : 0;
        } else {
            *delt = (OriRefPOC & 1) != (OriPOC & 1) ? -2 : 0;
        }
    } else {                // scaled is bottom field
        *delt2 = (ScaledRefPOC & 1) != (ScaledPOC & 1) ? -2 : 0;
        if ((ScaledPOC & 1) == (OriPOC & 1)) { // ori is bottom
            *delt = (OriRefPOC & 1) != (OriPOC & 1) ? -2 : 0;
        } else {
            *delt = (OriRefPOC & 1) != (OriPOC & 1) ? 2 : 0;
        }
    }
}

// ----------------------------------------------------------
// MV scaling for Skip/Direct Mode
static ALWAYS_INLINE int16_t scale_mv_skip(int mv, int dist_dst, int dist_src)
{
    return (int16_t)((mv * dist_dst * (MULTI / dist_src) + HALF_MULTI) >> OFFSET);
}

static ALWAYS_INLINE int16_t scale_mv_skip_y(xavs2_t *h, int mvy, int dist_dst, int dist_src)
{
    if (h->b_field_sequence == 0) {
        return scale_mv_skip(mvy, dist_dst, dist_src);
    } else {
        int oriPOC       = h->fdec->i_frm_poc;
        int oriRefPOC    = oriPOC - dist_src;
        int scaledPOC    = h->fdec->i_frm_poc;
        int scaledRefPOC = scaledPOC - dist_dst;
        int delta, delta2;

        getDeltas(h, &delta, &delta2, oriPOC, oriRefPOC, scaledPOC, scaledRefPOC);

        return (int16_t)(scale_mv_skip(mvy + delta, dist_dst, dist_src) - delta2);
    }
}

// ----------------------------------------------------------
// MV scaling for Bi-Skip/Direct Mode
static ALWAYS_INLINE int16_t scale_mv_biskip(int mv, int dist_dst, int dist_src_scale)
{
    return (int16_t)(xavs2_sign3(mv) * ((dist_src_scale * (1 + XAVS2_ABS(mv) * dist_dst) - 1) >> OFFSET));
}

static ALWAYS_INLINE int16_t scale_mv_biskip_y(xavs2_t *h, int mvy, int dist_dst, int dist_src, int dist_src_scale)
{
    int oriPOC = h->fdec->i_frm_poc;
    int oriRefPOC = oriPOC - dist_src;
    int scaledPOC = h->fdec->i_frm_poc;
    int scaledRefPOC = scaledPOC - dist_dst;
    int delta, delta2;

    getDeltas(h, &delta, &delta2, oriPOC, oriRefPOC, scaledPOC, scaledRefPOC);

    return (int16_t)(scale_mv_biskip(mvy + delta, dist_dst, dist_src_scale) - delta2);
}


/**
 * ===========================================================================
 * interface function declares
 * ===========================================================================
 */

#define get_mv_predictors_bskip FPFX(get_mv_predictors_bskip)
int  get_mv_predictors_bskip(xavs2_t *h, cu_t *p_cu);
#define get_mv_predictors_pskip FPFX(get_mv_predictors_pskip)
int  get_mv_predictors_pskip(xavs2_t *h, cu_t *p_cu);

#define get_mvp_default FPFX(get_mvp_default)
void get_mvp_default         (xavs2_t *h, const neighbor_inter_t *p_neighbors, mv_t *pmv, int bwd_2nd, cb_t *p_cb, int ref_idx);

#define pred_inter_search_single FPFX(pred_inter_search_single)
int  pred_inter_search_single(xavs2_t *h, cu_t *p_cu, cb_t *p_cb, xavs2_me_t *p_me, dist_t *fwd_cost, dist_t *bwd_cost);
#define pred_inter_search_bi FPFX(pred_inter_search_bi)
void pred_inter_search_bi    (xavs2_t *h, cu_t *p_cu, cb_t *p_cb, xavs2_me_t *p_me, dist_t *sym_mcost, dist_t *bid_mcost);
#define pred_inter_search_dual FPFX(pred_inter_search_dual)
void pred_inter_search_dual  (xavs2_t *h, cu_t *p_cu, cb_t *p_cb, xavs2_me_t *p_me, dist_t *dual_mcost, int *dual_best_fst_ref, int *dual_best_snd_ref);

#endif  // XAVS2_PREDICT_H
