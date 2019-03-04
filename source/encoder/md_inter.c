/*
 * md_inter.c
 *
 * Description of this file:
 *    Mode decision functions definition for Inter prediction of the xavs2 library
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
#include "block_info.h"
#include "cudata.h"
#include "me.h"


/**
 * ===========================================================================
 * global variables
 * ===========================================================================
 */
static const double tab_umh_alpha_2nd[MAX_INTER_MODES] = {
    0.0f, 0.01f, 0.01f, 0.01f, 0.02f, 0.03f, 0.03f, 0.04f
};
static const double tab_umh_alpha_3rd[MAX_INTER_MODES] = {
    0.0f, 0.06f, 0.07f, 0.07f, 0.08f, 0.12f, 0.11f, 0.15f
};


/**
 * ===========================================================================
 * local function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * determine the MVD value (1/4 pixel) is legal or not
 * Return: 0: out of the legal mv range; 1: in the legal mv range
 */
static ALWAYS_INLINE
int check_mvd(xavs2_t *h, int mvd_x, int mvd_y)
{
    if (h->param->i_frame_threads > 1) {
        return (mvd_x < 4096 && mvd_x >= -4096 &&
                mvd_y < ((1 << h->i_lcu_level) << 2) && mvd_y >= -((1 << h->i_lcu_level) << 2));
    }

    return (mvd_x < 4096 && mvd_x >= -4096 && mvd_y < 4096 && mvd_y >= -4096);
}

/* ---------------------------------------------------------------------------
 * determine the forward and backward mv value (1/4 pixel) is legal or not
 * return: 0: out of the legal mv range; 1: in the legal mv range
 */
static
int check_mv_range_sym(xavs2_t *h, mv_t *mv, int pix_x, int pix_y, int bsx, int bsy, int distance_fwd, int distance_bwd)
{
    int bsize = 1 << h->i_lcu_level;  /* valid padding size */
    int min_x = -((pix_x + bsize) << 2);
    int min_y = -((pix_y + bsize) << 2);
    int max_x = ((h->i_width  - (pix_x + bsx)) + bsize) << 2;
    int max_y = ((h->i_height - (pix_y + bsy)) + bsize) << 2;
    int bwd_mvx, bwd_mvy;

    min_x = XAVS2_MAX(min_x, h->min_mv_range[0]);
    min_y = XAVS2_MAX(min_y, h->min_mv_range[1]);
    max_x = XAVS2_MIN(max_x, h->max_mv_range[0]);
    max_y = XAVS2_MIN(max_y, h->max_mv_range[1]);

    if (h->i_type == SLICE_TYPE_B) {
        bwd_mvx = -scale_mv_skip(     mv->x, distance_bwd, distance_fwd);
        bwd_mvy = -scale_mv_skip_y(h, mv->y, distance_bwd, distance_fwd);
    } else {    /* SLICE_TYPE_F or SLICE_TYPE_P */
        bwd_mvx = scale_mv_skip(     mv->x, distance_bwd, distance_fwd);
        bwd_mvy = scale_mv_skip_y(h, mv->y, distance_bwd, distance_fwd);
    }

    if (mv->x > max_x || mv->x < min_x || mv->y > max_y || mv->y < min_y) {
        return 0;
    }

    if (bwd_mvx > max_x || bwd_mvx < min_x || bwd_mvy > max_y || bwd_mvy < min_y) {
        return 0;
    }

    return 1;
}

/**
 * ---------------------------------------------------------------------------
 * Function   : get temporal motion vector predictor for SKIP/DIRECT mode in B frame
 * Parameters :
 *      [in ] : h - encoder handler
 *      [out] : none
 * Return     : none
 * ---------------------------------------------------------------------------
 */
static INLINE
void get_bskip_mv_spatial(cu_mode_t *p_cumode, const neighbor_inter_t *p_neighbors)
{
    int bid_flag = 0, bw_flag = 0, fwd_flag = 0, sym_flag = 0, bid2 = 0;
    int j;

    g_funcs.fast_memset(p_cumode->skip_mv_1st, 0, sizeof(p_cumode->skip_mv_1st)
                        + sizeof(p_cumode->skip_mv_2nd));

    for (j = 0; j < 6; j++) {
        if (p_neighbors[j].i_dir_pred == PDIR_BID) {
            p_cumode->skip_mv_2nd[DS_B_BID] = p_neighbors[j].mv[1];
            p_cumode->skip_mv_1st[DS_B_BID] = p_neighbors[j].mv[0];
            bid_flag++;
            if (bid_flag == 1) {
                bid2 = j;
            }
        } else if (p_neighbors[j].i_dir_pred == PDIR_SYM) {
            p_cumode->skip_mv_2nd[DS_B_SYM] = p_neighbors[j].mv[1];
            p_cumode->skip_mv_1st[DS_B_SYM] = p_neighbors[j].mv[0];
            sym_flag++;
        } else if (p_neighbors[j].i_dir_pred == PDIR_BWD) {
            p_cumode->skip_mv_2nd[DS_B_BWD] = p_neighbors[j].mv[1];
            bw_flag++;
        } else if (p_neighbors[j].i_dir_pred == PDIR_FWD) {
            p_cumode->skip_mv_1st[DS_B_FWD] = p_neighbors[j].mv[0];
            fwd_flag++;
        }
    }

    /* 相邻块不存在双向预测块时，双向Skip/Direct模式的填充 */
    if (bid_flag == 0 && fwd_flag != 0 && bw_flag != 0) {
        p_cumode->skip_mv_2nd[DS_B_BID] = p_cumode->skip_mv_2nd[DS_B_BWD];
        p_cumode->skip_mv_1st[DS_B_BID] = p_cumode->skip_mv_1st[DS_B_FWD];
    }
    p_cumode->skip_ref_1st[DS_B_BID] = B_FWD;
    p_cumode->skip_ref_2nd[DS_B_BID] = B_BWD;

    /* 相邻块不存在对称预测块时，对称Skip/Direct模式的填充 */
    if (sym_flag == 0) {
        if (bid_flag > 1) {  /* 若存在双向预测块，则使用双向预测块生成 */
            p_cumode->skip_mv_2nd[DS_B_SYM] = p_neighbors[bid2].mv[1];
            p_cumode->skip_mv_1st[DS_B_SYM] = p_neighbors[bid2].mv[0];
        } else if (bw_flag != 0) {  /* 若存在后向预测块，则使用后向预测块生成 */
            p_cumode->skip_mv_2nd[DS_B_SYM]   =  p_cumode->skip_mv_2nd[DS_B_BWD];
            p_cumode->skip_mv_1st[DS_B_SYM].x = -p_cumode->skip_mv_2nd[DS_B_BWD].x;
            p_cumode->skip_mv_1st[DS_B_SYM].y = -p_cumode->skip_mv_2nd[DS_B_BWD].y;
        } else if (fwd_flag != 0) {  /* 若存在前向预测块，则使用前向预测块生成 */
            p_cumode->skip_mv_2nd[DS_B_SYM].x = -p_cumode->skip_mv_1st[DS_B_FWD].x;
            p_cumode->skip_mv_2nd[DS_B_SYM].y = -p_cumode->skip_mv_1st[DS_B_FWD].y;
            p_cumode->skip_mv_1st[DS_B_SYM]   =  p_cumode->skip_mv_1st[DS_B_FWD];
        }
    }
    p_cumode->skip_ref_1st[DS_B_SYM] = B_FWD;
    p_cumode->skip_ref_2nd[DS_B_SYM] = B_BWD;
    /* 后向预测块不存在时后向Skip/Direct模式的填充 */
    if (bw_flag == 0 && bid_flag > 1) {  /* 如果存在双向预测块，则使用双向预测块逆序的最后一个元素 */
        p_cumode->skip_mv_2nd[DS_B_BWD] = p_neighbors[bid2].mv[1];
    } else if (bw_flag == 0 && bid_flag != 0) {  /* 只有一个双向预测块时，使用双向列表的后向 */
        p_cumode->skip_mv_2nd[DS_B_BWD] = p_cumode->skip_mv_2nd[DS_B_BID];
    }
    p_cumode->skip_ref_1st[DS_B_BWD] = INVALID_REF;
    p_cumode->skip_ref_2nd[DS_B_BWD] = B_BWD;

    /* 前向预测块不存在时前向Skip/Direct模式的填充，类似后向Skip/Direct模式 */
    if (fwd_flag == 0 && bid_flag > 1) {
        p_cumode->skip_mv_1st[DS_B_FWD] = p_neighbors[bid2].mv[0];
    } else if (fwd_flag == 0 && bid_flag != 0) {
        p_cumode->skip_mv_1st[DS_B_FWD] = p_cumode->skip_mv_1st[DS_B_BID];
    }
    p_cumode->skip_ref_1st[DS_B_FWD] = B_FWD;
    p_cumode->skip_ref_2nd[DS_B_FWD] = INVALID_REF;
}

/**
 * ---------------------------------------------------------------------------
 * Function   : get spatial motion vector predictor for SKIP/DIRECT mode in P/F frame
 * Parameters :
 *      [in ] : h         - encoder handler
 *      [in ] : blocksize - size of current block
 *      [out] :           -
 * Return     :
 * ---------------------------------------------------------------------------
 */
static void get_pskip_mv_spatial(cu_mode_t *p_cumode, const neighbor_inter_t *p_neighbors)
{
    int bid_flag = 0, fwd_flag = 0, bid2 = 0, fwd2 = 0;
    int j;

    g_funcs.fast_memset(p_cumode->skip_mv_1st, 0, sizeof(p_cumode->skip_mv_1st)
                        + sizeof(p_cumode->skip_mv_2nd)
                        + sizeof(p_cumode->skip_ref_1st)
                        + sizeof(p_cumode->skip_ref_2nd));

    for (j = 0; j < 6; j++) {
        if (p_neighbors[j].ref_idx[0] != INVALID_REF && p_neighbors[j].ref_idx[1] != INVALID_REF) {
            // dual prediction
            p_cumode->skip_ref_1st[DS_DUAL_1ST] = p_neighbors[j].ref_idx[0];
            p_cumode->skip_ref_2nd[DS_DUAL_1ST] = p_neighbors[j].ref_idx[1];
            p_cumode->skip_mv_1st[DS_DUAL_1ST] = p_neighbors[j].mv[0];
            p_cumode->skip_mv_2nd[DS_DUAL_1ST] = p_neighbors[j].mv[1];
            bid_flag++;
            if (bid_flag == 1) {
                bid2 = j;
            }
        } else if (p_neighbors[j].ref_idx[0] != INVALID_REF && p_neighbors[j].ref_idx[1] == INVALID_REF) {
            // fwd
            p_cumode->skip_ref_1st[DS_SINGLE_1ST] = p_neighbors[j].ref_idx[0];
            p_cumode->skip_ref_2nd[DS_SINGLE_1ST] = INVALID_REF;
            p_cumode->skip_mv_1st[DS_SINGLE_1ST] = p_neighbors[j].mv[0];
            fwd_flag++;
            if (fwd_flag == 1) {
                fwd2 = j;
            }
        }
    }

    // first dual
    if (bid_flag == 0 && fwd_flag > 1) {
        p_cumode->skip_ref_1st[DS_DUAL_1ST] = p_cumode->skip_ref_1st[DS_SINGLE_1ST];
        p_cumode->skip_ref_2nd[DS_DUAL_1ST] = p_neighbors[fwd2].ref_idx[0];
        p_cumode->skip_mv_1st[DS_DUAL_1ST] = p_cumode->skip_mv_1st[DS_SINGLE_1ST];
        p_cumode->skip_mv_2nd[DS_DUAL_1ST] = p_neighbors[fwd2].mv[0];
    }

    // second dual
    if (bid_flag > 1) {
        p_cumode->skip_ref_1st[DS_DUAL_2ND] = p_neighbors[bid2].ref_idx[0];
        p_cumode->skip_ref_2nd[DS_DUAL_2ND] = p_neighbors[bid2].ref_idx[1];
        p_cumode->skip_mv_1st[DS_DUAL_2ND] = p_neighbors[bid2].mv[0];
        p_cumode->skip_mv_2nd[DS_DUAL_2ND] = p_neighbors[bid2].mv[1];
    } else if (bid_flag == 1 && fwd_flag > 1) {
        p_cumode->skip_ref_1st[DS_DUAL_2ND] = p_cumode->skip_ref_1st[DS_SINGLE_1ST];
        p_cumode->skip_ref_2nd[DS_DUAL_2ND] = p_neighbors[fwd2].ref_idx[0];
        p_cumode->skip_mv_1st[DS_DUAL_2ND] = p_cumode->skip_mv_1st[DS_SINGLE_1ST];
        p_cumode->skip_mv_2nd[DS_DUAL_2ND] = p_neighbors[fwd2].mv[0];
    }

    // first fwd
    p_cumode->skip_ref_2nd[DS_SINGLE_1ST] = INVALID_REF;
    if (fwd_flag == 0 && bid_flag > 1) {
        p_cumode->skip_ref_1st[DS_SINGLE_1ST] = p_neighbors[bid2].ref_idx[0];
        p_cumode->skip_mv_1st [DS_SINGLE_1ST] = p_neighbors[bid2].mv[0];
    } else if (fwd_flag == 0 && bid_flag == 1) {
        p_cumode->skip_ref_1st[DS_SINGLE_1ST] = p_cumode->skip_ref_1st[DS_DUAL_1ST];
        p_cumode->skip_mv_1st [DS_SINGLE_1ST] = p_cumode->skip_mv_1st[DS_DUAL_1ST];
    }

    // second fwd
    p_cumode->skip_ref_2nd[DS_SINGLE_2ND] = INVALID_REF;
    if (fwd_flag > 1) {
        p_cumode->skip_ref_1st[DS_SINGLE_2ND] = p_neighbors[fwd2].ref_idx[0];
        p_cumode->skip_mv_1st [DS_SINGLE_2ND] = p_neighbors[fwd2].mv[0];
    } else if (bid_flag > 1) {
        p_cumode->skip_ref_1st[DS_SINGLE_2ND] = p_neighbors[bid2].ref_idx[1];
        p_cumode->skip_mv_1st [DS_SINGLE_2ND] = p_neighbors[bid2].mv[1];
    } else if (bid_flag == 1) {
        p_cumode->skip_ref_1st[DS_SINGLE_2ND] = p_cumode->skip_ref_2nd[DS_DUAL_1ST];
        p_cumode->skip_mv_1st [DS_SINGLE_2ND] = p_cumode->skip_mv_2nd[DS_DUAL_1ST];
    }
}

/**
 * ---------------------------------------------------------------------------
 * Function   : get temporal motion vector predictor for SKIP/DIRECT mode in P/F frame
 * Parameters :
 *      [in ] : h         - encoder handler
 *      [in ] : p_cu      - current encoding CU
 *      [out] : none
 * Return     : none
 * ---------------------------------------------------------------------------
 */
static void get_pskip_mv_temporal(xavs2_t *h, cu_t *p_cu)
{
    cu_mode_t *p_cumode = cu_get_layer_mode(h, p_cu->cu_info.i_level);
    int blocksize2 = p_cu->i_size >> 1;
    int w_in_16x16 = (h->i_width_in_minpu + 3) >> 2;
    const int8_t *col_ref = h->fref[0]->pu_ref;
    const mv_t   *col_mv = h->fref[0]->pu_mv;
    int pic_x = p_cu->i_pix_x;
    int pic_y = p_cu->i_pix_y;
    int refframe;
    int curT, colT;
    int k;

    for (k = 0; k < 4; k++) {
        int b_pix_x = pic_x + blocksize2 * (k  & 1);
        int b_pix_y = pic_y + blocksize2 * (k >> 1);
        int col_pos = (b_pix_y >> 4) * w_in_16x16 + (b_pix_x >> 4);
        mv_t mv_1st;

        refframe = col_ref[col_pos];
        if (refframe >= 0) {
            curT = calculate_distance(h, 0);
            colT = h->fref[0]->ref_dpoc[refframe];
            mv_1st.x = scale_mv_skip(col_mv[col_pos].x, curT, colT);
            mv_1st.y = scale_mv_skip(col_mv[col_pos].y, curT, colT);
        } else {
            mv_1st.v = 0;
        }

        p_cumode->tskip_mv[k][0].v = mv_1st.v;
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
int get_mvp_type_default(int ref_frame, int rFrameL, int rFrameU, int rFrameUR, cb_t *p_cb)
{
    int mvp_type = MVP_MEDIAN;

    if ((rFrameL != INVALID_REF) && (rFrameU == INVALID_REF) && (rFrameUR == INVALID_REF)) {
        mvp_type = MVP_LEFT;
    } else if ((rFrameL == INVALID_REF) && (rFrameU != INVALID_REF) && (rFrameUR == INVALID_REF)) {
        mvp_type = MVP_TOP;
    } else if ((rFrameL == INVALID_REF) && (rFrameU == INVALID_REF) && (rFrameUR != INVALID_REF)) {
        mvp_type = MVP_TR;
    } else if (p_cb->w < p_cb->h) {
        if (p_cb->x == 0) {
            if (rFrameL == ref_frame) {
                mvp_type = MVP_LEFT;
            }
        } else {
            if (rFrameUR == ref_frame) {
                mvp_type = MVP_TR;
            }
        }
    } else if (p_cb->w > p_cb->h) {
        if (p_cb->y == 0) {
            if (rFrameU == ref_frame) {
                mvp_type = MVP_TOP;
            }
        } else {
            if (rFrameL == ref_frame) {
                mvp_type = MVP_LEFT;
            }
        }
    }

    return mvp_type;
}

/* ---------------------------------------------------------------------------
 */
static int derive_median_mv(int16_t mva, int16_t mvb, int16_t mvc, int16_t *pmv)
{
    int mvp_type;

    if (((mva < 0) && (mvb > 0) && (mvc > 0)) || ((mva > 0) && (mvb < 0) && (mvc < 0))) {
        *pmv = (mvb + mvc) / 2;
        mvp_type = 1;           // b
    } else if (((mvb < 0) && (mva > 0) && (mvc > 0)) || ((mvb > 0) && (mva < 0) && (mvc < 0))) {
        *pmv = (mvc + mva) / 2;
        mvp_type = 2;           // c
    } else if (((mvc < 0) && (mva > 0) && (mvb > 0)) || ((mvc > 0) && (mva < 0) && (mvb < 0))) {
        *pmv = (mva + mvb) / 2;
        mvp_type = 0;           // a
    } else {
        const int dAB = XAVS2_ABS(mva - mvb);  // for Ax
        const int dBC = XAVS2_ABS(mvb - mvc);  // for Bx
        const int dCA = XAVS2_ABS(mvc - mva);  // for Cx
        const int min_diff = XAVS2_MIN(dAB, XAVS2_MIN(dBC, dCA));

        if (min_diff == dAB) {
            *pmv = (mva + mvb) / 2;
            mvp_type = 0;       // a;
        } else if (min_diff == dBC) {
            *pmv = (mvb + mvc) / 2;
            mvp_type = 1;       // b;
        } else {
            *pmv = (mvc + mva) / 2;
            mvp_type = 2;       // c;
        }
    }

    return mvp_type;
}

/* ---------------------------------------------------------------------------
 * MV scaling for Normal Inter Mode (MVP + MVD)
 */
static ALWAYS_INLINE
int16_t scale_mv_default(int mv, int dist_dst, int dist_src_scale)
{
    mv = xavs2_sign3(mv) * ((XAVS2_ABS(mv) * dist_dst * dist_src_scale + HALF_MULTI) >> OFFSET);
    return (int16_t)(XAVS2_CLIP3(-32768, 32767, mv));
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
int16_t scale_mv_default_y(xavs2_t *h, int16_t mvy, int dist_dst, int dist_src, int dist_src_scale)
{
    int oriPOC       = h->fdec->i_frm_poc;
    int oriRefPOC    = oriPOC - dist_src;
    int scaledPOC    = h->fdec->i_frm_poc;
    int scaledRefPOC = scaledPOC - dist_dst;
    int delta, delta2;

    getDeltas(h, &delta, &delta2, oriPOC, oriRefPOC, scaledPOC, scaledRefPOC);

    return (int16_t)(scale_mv_default(mvy + delta, dist_dst, dist_src_scale) - delta2);
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
void check_scaling_neighbor_mv_b(xavs2_t *h, mv_t *mv, int dist_dst, int dist_src_scale, int ref_neighbor)
{
    if (ref_neighbor >= 0) {
        if (h->b_field_sequence == 0) {
            mv->y = scale_mv_default(mv->y, dist_dst, dist_src_scale);
            mv->x = scale_mv_default(mv->x, dist_dst, dist_src_scale);
        } else {
            mv->y = scale_mv_default_y(h, mv->y, dist_dst, dist_dst, dist_src_scale);
            mv->x = scale_mv_default  (   mv->x, dist_dst, dist_src_scale);
        }
    } else {
        mv->v = 0;
    }
}
/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
void check_scaling_neighbor_mv(xavs2_t *h, mv_t *mv, int dist_dst, int ref_neighbor)
{
    if (ref_neighbor >= 0) {
        int dist_src_scale = h->fdec->ref_dpoc_multi[ref_neighbor];

        if (h->b_field_sequence == 0) {
            mv->y = scale_mv_default(mv->y, dist_dst, dist_src_scale);
            mv->x = scale_mv_default(mv->x, dist_dst, dist_src_scale);
        } else {
            int dist_src = h->fdec->ref_dpoc[ref_neighbor];
            mv->y = scale_mv_default_y(h, mv->y, dist_dst, dist_src, dist_src_scale);
            mv->x = scale_mv_default  (   mv->x, dist_dst, dist_src_scale);
        }
    } else {
        mv->v = 0;
    }
}

/* ---------------------------------------------------------------------------
 */
void get_mvp_default(xavs2_t *h, const neighbor_inter_t *p_neighbors, mv_t *pmv, int bwd_2nd, cb_t *p_cb, int ref_idx)
{
    int is_available_UR = p_neighbors[BLK_TOPRIGHT].is_available;
    int rFrameL, rFrameU, rFrameUR, rFrameUL;
    int mvp_type;
    mv_t mva, mvb, mvc, mvd;

    rFrameL  = p_neighbors[BLK_LEFT    ].ref_idx[bwd_2nd];
    rFrameU  = p_neighbors[BLK_TOP     ].ref_idx[bwd_2nd];
    rFrameUL = p_neighbors[BLK_TOPLEFT ].ref_idx[bwd_2nd];
    rFrameUR = is_available_UR ? p_neighbors[BLK_TOPRIGHT].ref_idx[bwd_2nd] : rFrameUL;

    mva = p_neighbors[BLK_LEFT    ].mv[bwd_2nd];
    mvb = p_neighbors[BLK_TOP     ].mv[bwd_2nd];
    mvd = p_neighbors[BLK_TOPLEFT ].mv[bwd_2nd];
    mvc = is_available_UR ? p_neighbors[BLK_TOPRIGHT].mv[bwd_2nd] : mvd;

    mvp_type = get_mvp_type_default(ref_idx, rFrameL, rFrameU, rFrameUR, p_cb);

    if (h->i_type == SLICE_TYPE_B) {
        int mult_distance  = h->fdec->ref_dpoc      [bwd_2nd ? B_BWD : B_FWD];
        int dist_src_scale = h->fdec->ref_dpoc_multi[bwd_2nd ? B_BWD : B_FWD];
        check_scaling_neighbor_mv_b(h, &mva, mult_distance, dist_src_scale, rFrameL);
        check_scaling_neighbor_mv_b(h, &mvb, mult_distance, dist_src_scale, rFrameU);
        check_scaling_neighbor_mv_b(h, &mvc, mult_distance, dist_src_scale, rFrameUR);
    } else {
        int mult_distance = calculate_distance(h, ref_idx);
        check_scaling_neighbor_mv(h, &mva, mult_distance, rFrameL);
        check_scaling_neighbor_mv(h, &mvb, mult_distance, rFrameU);
        check_scaling_neighbor_mv(h, &mvc, mult_distance, rFrameUR);
    }

    switch (mvp_type) {
    case MVP_MEDIAN:
        // for x component
        derive_median_mv(mva.x, mvb.x, mvc.x, &pmv->x);
        // for y component
        derive_median_mv(mva.y, mvb.y, mvc.y, &pmv->y);
        break;
    case MVP_LEFT:
        pmv->v = mva.v;
        break;
    case MVP_TOP:
        pmv->v = mvb.v;
        break;
    case MVP_TR:
        pmv->v = mvc.v;
        break;
    default:
        assert(0);
        break;
    }
}

/* ---------------------------------------------------------------------------
 */
static INLINE
void get_mvp_default_sad(xavs2_t *h, const neighbor_inter_t *p_neighbors, cu_t *p_cu, xavs2_me_t *p_me, mv_t *pmv, int bwd_2nd, cb_t *p_cb, int ref_idx)
{
    int mode = p_cu->cu_info.i_mode;
    int pic_block_x = (p_cu->i_pix_x + p_cb->x) >> MIN_PU_SIZE_IN_BIT;
    int pic_block_y = (p_cu->i_pix_y + p_cb->y) >> MIN_PU_SIZE_IN_BIT;
    int width_in_4x4 = h->i_width_in_minpu;
    dist_t SAD[4] = { 0, 0, 0, 0 };
    int is_available_UL = p_neighbors[BLK_TOPLEFT ].is_available;
    int is_available_UR = p_neighbors[BLK_TOPRIGHT].is_available;
    int rFrameL, rFrameU, rFrameUR, rFrameUL;
    int mvp_type;
    dist_t sad_space;
    mv_t mva, mvb, mvc, mvd;

    rFrameL  = p_neighbors[BLK_LEFT    ].ref_idx[bwd_2nd];
    rFrameU  = p_neighbors[BLK_TOP     ].ref_idx[bwd_2nd];
    rFrameUL = p_neighbors[BLK_TOPLEFT ].ref_idx[bwd_2nd];
    rFrameUR = is_available_UR ? p_neighbors[BLK_TOPRIGHT].ref_idx[bwd_2nd] : rFrameUL;

    mva = p_neighbors[BLK_LEFT    ].mv[bwd_2nd];
    mvb = p_neighbors[BLK_TOP     ].mv[bwd_2nd];
    mvd = p_neighbors[BLK_TOPLEFT ].mv[bwd_2nd];
    mvc = is_available_UR ? p_neighbors[BLK_TOPRIGHT].mv[bwd_2nd] : mvd;

    SAD[0] = pic_block_x > 0 ? h->all_mincost[(pic_block_y    ) * width_in_4x4 + pic_block_x - 1][mode][ref_idx] : 0;
    SAD[1] = pic_block_y > 0 ? h->all_mincost[(pic_block_y - 1) * width_in_4x4 + pic_block_x    ][mode][ref_idx] : 0;
    SAD[2] = is_available_UR ? h->all_mincost[(pic_block_y - 1) * width_in_4x4 + pic_block_x + 1][mode][ref_idx] : 0;
    SAD[3] = is_available_UL ? h->all_mincost[(pic_block_y - 1) * width_in_4x4 + pic_block_x - 1][mode][ref_idx] : 0;

    mvp_type = get_mvp_type_default(ref_idx, rFrameL, rFrameU, rFrameUR, p_cb);

    if (h->i_type == SLICE_TYPE_B) {
        int mult_distance  = h->fdec->ref_dpoc      [bwd_2nd ? B_BWD : B_FWD];
        int dist_src_scale = h->fdec->ref_dpoc_multi[bwd_2nd ? B_BWD : B_FWD];
        check_scaling_neighbor_mv_b(h, &mva, mult_distance, dist_src_scale, rFrameL);
        check_scaling_neighbor_mv_b(h, &mvb, mult_distance, dist_src_scale, rFrameU);
        check_scaling_neighbor_mv_b(h, &mvc, mult_distance, dist_src_scale, rFrameUR);
    } else {
        int mult_distance = calculate_distance(h, ref_idx);
        check_scaling_neighbor_mv(h, &mva, mult_distance, rFrameL);
        check_scaling_neighbor_mv(h, &mvb, mult_distance, rFrameU);
        check_scaling_neighbor_mv(h, &mvc, mult_distance, rFrameUR);
    }

    switch (mvp_type) {
    case MVP_MEDIAN:
        // for x component
        derive_median_mv(mva.x, mvb.x, mvc.x, &pmv->x);
        // for y component
        sad_space = SAD[derive_median_mv(mva.y, mvb.y, mvc.y, &pmv->y)];
        break;
    case MVP_LEFT:
        pmv->v = mva.v;
        sad_space = SAD[0];     // a
        break;
    case MVP_TOP:
        pmv->v = mvb.v;
        sad_space = SAD[1];     // b
        break;
    case MVP_TR:
        pmv->v = mvc.v;
        sad_space = SAD[2];     // c
        break;
    default:
        sad_space = 0;
        assert(0);
        break;
    }
    p_me->pred_sad_space = sad_space;
}

/* ---------------------------------------------------------------------------
*/
static void
fast_me_prepare_info_remove_mvp(xavs2_t *h, xavs2_me_t *p_me, int mode, int ref_idx,
                                dist_t mincosts[MAX_INTER_MODES][MAX_REFS])
{
    dist_t pred_sad, sad_reference, sad_uplayer;

    /* get mvp & sad in upper layer */
    if (mode == PRED_2NxnU || mode == PRED_2NxnD) {
        sad_uplayer = mincosts[PRED_2NxN][ref_idx] / 2;  // sad in upper layer
    } else if (mode == PRED_nLx2N || mode == PRED_nRx2N) {
        sad_uplayer = mincosts[PRED_Nx2N][ref_idx] / 2;  // sad in upper layer
    } else if (mode > PRED_2Nx2N) {
        sad_uplayer = mincosts[PRED_2Nx2N][ref_idx] / 2; // sad in upper layer
    } else {
        sad_uplayer = 0;       // set flag, the up layer cannot be used
    }
    p_me->pred_sad_uplayer = sad_uplayer;

    /* get mvp & sad in nearest reference frame */
    if (h->i_type == SLICE_TYPE_B && ref_idx == B_FWD) {
        sad_reference = 0;
    } else if (ref_idx > 0) {
        sad_reference = mincosts[mode][ref_idx - 1];    // sad in nearest reference frame
    } else {
        sad_reference = 0;
    }
    p_me->pred_sad_ref = sad_reference;

    /* get pred sad */
    if (h->i_type != SLICE_TYPE_B && ref_idx > 0) {
        pred_sad = p_me->pred_sad_ref;
    } else if (mode == PRED_2Nx2N) {
        pred_sad = p_me->pred_sad_space;
    } else {
        pred_sad = p_me->pred_sad_uplayer;
    }
    p_me->pred_sad = pred_sad;
}

/* ---------------------------------------------------------------------------
 */
static void
fast_me_prepare_info(xavs2_t *h, xavs2_me_t *p_me, int mode, int ref_idx,int pu_idx,
                     dist_t mincosts[MAX_INTER_MODES][MAX_REFS])
{
    dist_t pred_sad, sad_reference, sad_uplayer;
    mv_t(*best_mvs)[4][MAX_REFS] = p_me->all_best_mv;

    /* get mvp & sad in upper layer */
    if (mode == PRED_2NxnU || mode == PRED_2NxnD) {
        p_me->mvp1  = best_mvs[PRED_2NxN][pu_idx][ref_idx];
        sad_uplayer = mincosts[PRED_2NxN][ref_idx] / 2;  // sad in upper layer
    } else if (mode == PRED_nLx2N || mode == PRED_nRx2N) {
        p_me->mvp1  = best_mvs[PRED_Nx2N][pu_idx][ref_idx];
        sad_uplayer = mincosts[PRED_Nx2N][ref_idx] / 2;  // sad in upper layer
    } else if (mode > PRED_2Nx2N) {
        p_me->mvp1  = best_mvs[PRED_2Nx2N][pu_idx][ref_idx];
        sad_uplayer = mincosts[PRED_2Nx2N][ref_idx] / 2; // sad in upper layer
    } else {
        p_me->mvp1.v = 0;
        sad_uplayer  = 0;       // set flag, the up layer cannot be used
    }
    p_me->pred_sad_uplayer = sad_uplayer;

    /* get mvp & sad in nearest reference frame */
    if (h->i_type == SLICE_TYPE_B && ref_idx == B_FWD) {
        mv_t mv_bwd = best_mvs[mode][pu_idx][B_BWD];
        p_me->mvp2.x  = (int16_t)(-mv_bwd.x);
        p_me->mvp2.y  = (int16_t)(-mv_bwd.y);
        sad_reference = mincosts[mode][B_BWD];
    } else if (ref_idx > 0) {
        mv_t mv_last = best_mvs[mode][pu_idx][ref_idx - 1];
        int dpoc_last = h->fdec->ref_dpoc[ref_idx - 1];
        int dpoc_curr = h->fdec->ref_dpoc[ref_idx];
        p_me->mvp2.x  = (int16_t)(mv_last.x * dpoc_curr / (float)dpoc_last);
        p_me->mvp2.y  = (int16_t)(mv_last.y * dpoc_curr / (float)dpoc_last);
        sad_reference = mincosts[mode][ref_idx - 1];    // sad in nearest reference frame
    } else {
        sad_reference = 0;
    }
    p_me->pred_sad_ref = sad_reference;

    /* get MV of collocated block */
    if (h->fref[0] != NULL && h->fref[0]->i_frm_type != XAVS2_TYPE_I) {
        int stride_mvstore = (h->i_width_in_minpu + 3) >> 2;
        int pu_pos = (p_me->i_pix_y >> 4) * stride_mvstore + (p_me->i_pix_x >> 4);
        mv_t mv_col = h->fref[0]->pu_mv [pu_pos];
        int ref_idx_col = h->fref[0]->pu_ref[pu_pos];
        if (ref_idx_col >= 0) {
            int dpoc_col = h->fref[0]->ref_dpoc[ref_idx_col];
            int dpoc_curr = h->fdec->ref_dpoc[ref_idx];
            p_me->mvp3.x = (int16_t)(mv_col.x * dpoc_curr / (float)dpoc_col);
            p_me->mvp3.y = (int16_t)(mv_col.y * dpoc_curr / (float)dpoc_col);
        } else {
            p_me->mvp3.v = 0;
        }
    } else {
        p_me->mvp3.v = 0;
    }

    /* get pred sad */
    if (h->i_type != SLICE_TYPE_B && ref_idx > 0) {
        pred_sad = p_me->pred_sad_ref;
    } else if (mode == PRED_2Nx2N) {
        pred_sad = p_me->pred_sad_space;
    } else {
        pred_sad = p_me->pred_sad_uplayer;
    }
    p_me->pred_sad = pred_sad;

    /* init beta parameters for UMH */
    if (pred_sad != 0) {
        double threshold = h->umh_bsize[mode] / (pred_sad * pred_sad);
        p_me->beta2 = threshold - tab_umh_alpha_2nd[mode];
        p_me->beta3 = threshold - tab_umh_alpha_3rd[mode];
    } else {
        p_me->beta2 = 0;
        p_me->beta3 = 0;
    }
}


/**
 * ===========================================================================
 * interface function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
int get_mv_predictors_bskip(xavs2_t *h, cu_t *p_cu)
{
    cu_mode_t *p_cumode = cu_get_layer_mode(h, p_cu->cu_info.i_level);
    neighbor_inter_t *p_neighbors = cu_get_layer(h, p_cu->cu_info.i_level)->neighbor_inter;
    mv_t mv_1st, mv_2nd;
    cb_t cur_cb;
    const int8_t *col_ref = h->fref[0]->pu_ref;
    const mv_t   *col_mv = h->fref[0]->pu_mv;
    int w_in_16x16 = (h->i_width_in_minpu + 3) >> 2;
    int i_level = p_cu->cu_info.i_level;
    int pix_x = p_cu->i_pix_x;
    int pix_y = p_cu->i_pix_y;
    int pic_block_x, pic_block_y;
    int col_mv_pos;
    int col_blk_ref;
    int k;
    int blocksize  = 1 << i_level;
    int blocksize2 = 1 << (i_level - 1);

    assert(SLICE_TYPE_B == h->i_type);
    cur_cb.x = cur_cb.y = 0;
    cur_cb.w = cur_cb.h = (int8_t)blocksize;

    for (k = 0; k < 4; k++) {
        pic_block_y = pix_y + (k >> 1) * blocksize2;
        pic_block_x = pix_x + (k  & 1) * blocksize2;

        col_mv_pos  = (pic_block_y >> 4) * w_in_16x16 + (pic_block_x >> 4);
        col_blk_ref = col_ref[col_mv_pos];
        if (col_blk_ref == INVALID_REF) {
            ///! 9.5.8.4.3 运动矢量导出方法2：如果编码 单元子类型为 B_Skip_Bi，且时域PU的参考索引为 INVALID_REF
            get_mvp_default(h, p_neighbors, &mv_1st, 0, &cur_cb, B_FWD);  // 这里传递的ref_idx影响p_me->pred_sad_space，但不被使用
            get_mvp_default(h, p_neighbors, &mv_2nd, 1, &cur_cb, B_BWD);
        } else {
            int TRp = h->fref[B_BWD]->ref_dpoc[col_blk_ref];
            int dst_src_scale = h->fref[B_BWD]->ref_dpoc_multi[col_blk_ref];
            int TRd = calculate_distance(h, B_BWD);
            int TRb = calculate_distance(h, B_FWD);

            mv_t mv_col = col_mv[col_mv_pos];

            if (h->b_field_sequence == 0) {
                mv_1st.x = scale_mv_biskip(mv_col.x, TRb, dst_src_scale);
                mv_1st.y = scale_mv_biskip(mv_col.y, TRb, dst_src_scale);

                mv_2nd.x = -scale_mv_biskip(mv_col.x, TRd, dst_src_scale);
                mv_2nd.y = -scale_mv_biskip(mv_col.y, TRd, dst_src_scale);
            } else {
                mv_1st.x = scale_mv_biskip(     mv_col.x, TRb, dst_src_scale);
                mv_1st.y = scale_mv_biskip_y(h, mv_col.y, TRb, TRp, dst_src_scale);

                mv_2nd.x = -scale_mv_biskip(     mv_col.x, TRd, dst_src_scale);
                mv_2nd.y = -scale_mv_biskip_y(h, mv_col.y, TRd, TRp, dst_src_scale);
            }
        }

        p_cumode->tskip_mv[k][0] = mv_1st;
        p_cumode->tskip_mv[k][1] = mv_2nd;
        // only calculate block 0 for smallest CU, need copy MV of block 0 to block 1/2/3
        if (i_level == MIN_CU_SIZE_IN_BIT) {
            for (k = 1; k < 4; k++) {
                p_cumode->tskip_mv[k][0] = mv_1st;
                p_cumode->tskip_mv[k][1] = mv_2nd;
            }
            break;
        }
    }

    get_bskip_mv_spatial(p_cumode, p_neighbors);
    return 1;
}

/* ---------------------------------------------------------------------------
 */
int get_mv_predictors_pskip(xavs2_t *h, cu_t *p_cu)
{
    int i, k;

    get_pskip_mv_temporal(h, p_cu);

    if (h->i_type == SLICE_TYPE_F) {
        cu_mode_t        *p_cu_mode = cu_get_layer_mode(h, p_cu->cu_info.i_level);
        neighbor_inter_t *p_neighbors = cu_get_layer(h, p_cu->cu_info.i_level)->neighbor_inter;
        if (h->i_ref > 1) {
            int *delta_P = h->fdec->ref_dpoc;

            for (k = 0; k < 4; k++) {
                mv_t mv_1st = p_cu_mode->tskip_mv[k][0];
                for (i = 1; i < h->i_ref; i++) {
                    mv_t mv_2nd;
                    mv_2nd.x = scale_mv_skip  (   mv_1st.x, delta_P[i], delta_P[0]);
                    mv_2nd.y = scale_mv_skip_y(h, mv_1st.y, delta_P[i], delta_P[0]);
                    p_cu_mode->tskip_mv[k][i] = mv_2nd;
                }
            }
        }

        get_pskip_mv_spatial(p_cu_mode, p_neighbors);

        if (h->param->enable_wsm) {
            return h->i_ref;
        }
    }

    return 1;
}

/* ---------------------------------------------------------------------------
 */
static INLINE
int add_one_mv_candidate(xavs2_me_t *p_me, int16_t (*mvc)[2], int i_mvc, int x, int y)
{
    int mv_x_min = p_me->mv_min_fpel[0];
    int mv_y_min = p_me->mv_min_fpel[1];
    int mv_x_max = p_me->mv_max_fpel[0];
    int mv_y_max = p_me->mv_max_fpel[1];
    int i;

    x = IPEL(x);
    y = IPEL(y);
    x = XAVS2_CLIP3(mv_x_min, mv_x_max, x);
    y = XAVS2_CLIP3(mv_y_min, mv_y_max, y);

    for (i = 0; i < i_mvc; i++) {
        if (mvc[i][0] == x && mvc[i][1] == y) {
            break;
        }
    }
    if (i == i_mvc) {
        mvc[i][0] = (int16_t)x;
        mvc[i][1] = (int16_t)y;
        return i_mvc + 1;
    } else {
        return i_mvc;
    }
}


/* ---------------------------------------------------------------------------
 */
int pred_inter_search_single(xavs2_t *h, cu_t *p_cu, cb_t *p_cb, xavs2_me_t *p_me, dist_t *fwd_cost, dist_t *bwd_cost)
{
    int16_t mvc[8][2] = {{0}};
    int i_mvc = 0;
    int pu_size_shift = p_cu->cu_info.i_level - MIN_CU_SIZE_IN_BIT;
    int mode = p_cu->cu_info.i_mode;
    int ref_idx;
    int best_ref_idx = 0;
    dist_t cost;
    int mv_mempos_x;
    int mv_mempos_y;
    mv_t mv;
    int b_mv_valid;              // MV是否有效：大小取值是否在标准规定的有效范围内
    int pu_idx_x = p_cb->x != 0; // PU index in CU
    int pu_idx_y = p_cb->y != 0;
    int pu_idx = (pu_idx_y << 1) + pu_idx_x;
    int pix_x = p_cu->i_pix_x + p_cb->x;
    int pix_y = p_cu->i_pix_y + p_cb->y;
    int bsx = p_cb->w;
    int bsy = p_cb->h;
    int i, j, m, n, k;
    cu_mv_mode_t *p_mode_mvs = cu_get_layer_mode(h, p_cu->cu_info.i_level)->mvs[mode];
    neighbor_inter_t *p_neighbors = cu_get_layer(h, p_cu->cu_info.i_level)->neighbor_inter;
    dist_t(*all_min_costs)[MAX_INTER_MODES][MAX_REFS];
    int width_in_4x4 = h->i_width_in_minpu;
    int max_ref = h->i_ref;

    *fwd_cost = MAX_DISTORTION;
    mv_mempos_x = (pix_x + MIN_PU_SIZE - 1) >> MIN_PU_SIZE_IN_BIT;  // 考虑到8x8块的非对称划分，需要做一个补偿再移位
    mv_mempos_y = (pix_y + MIN_PU_SIZE - 1) >> MIN_PU_SIZE_IN_BIT;
    all_min_costs = &h->all_mincost[mv_mempos_y * width_in_4x4 + mv_mempos_x];

    /* make p_fenc point to the start address of the current PU */
    p_me->p_fenc  = h->lcu.p_fenc[0] + (pix_y - h->lcu.i_pix_y) * FENC_STRIDE + pix_x - h->lcu.i_pix_x;
    p_me->i_pixel = PART_INDEX(bsx, bsy);
    p_me->i_pix_x   = pix_x;
    p_me->i_pix_y   = pix_y;
    p_me->i_block_w = bsx;
    p_me->i_block_h = bsy;

    /* calculate max allowed MV range
     * limit motion search to a slightly smaller range than the theoretical limit,
     * since the search may go a few iterations past its given range */
    m = 6;  // UMH: 1 for diamond, 2 for octagon, 2 for subpel
    i = (-MAX_CU_SIZE - pix_x) << 2;                     // mv min
    j = (h->i_width + MAX_CU_SIZE - pix_x - bsx) << 2;   // mv max
    p_me->mv_min[0] = XAVS2_CLIP3(h->min_mv_range[0], h->max_mv_range[0], i);
    p_me->mv_max[0] = XAVS2_CLIP3(h->min_mv_range[0], h->max_mv_range[0], j);
    p_me->mv_min_fpel[0] = (p_me->mv_min[0] >> 2) + m;
    p_me->mv_max_fpel[0] = (p_me->mv_max[0] >> 2) - m;

    i = (-MAX_CU_SIZE - pix_y) << 2;                     // mv min
    j = (h->i_height + MAX_CU_SIZE - pix_y - bsy) << 2;  // mv max
    p_me->mv_min[1] = XAVS2_CLIP3(h->min_mv_range[1], h->max_mv_range[1], i);
    p_me->mv_max[1] = XAVS2_CLIP3(h->min_mv_range[1], h->max_mv_range[1], j);
    p_me->mv_min_fpel[1] = (p_me->mv_min[1] >> 2) + m;
    p_me->mv_max_fpel[1] = (p_me->mv_max[1] >> 2) - m;

    // loop over all reference frames
    for (ref_idx = 0; ref_idx < max_ref; ref_idx++) {
        int bwd_2nd = h->i_type == SLICE_TYPE_B && ref_idx == B_BWD;
        xavs2_frame_t *p_ref_frm = h->fref[ref_idx];
        mv_t *pred_mv = &p_mode_mvs[pu_idx].all_mvp[ref_idx];

        /* get MVP (motion vector predictor) */
        if (h->param->me_method == XAVS2_ME_UMH) {
            get_mvp_default_sad(h, p_neighbors, p_cu, p_me, pred_mv, bwd_2nd, p_cb, ref_idx);
        } else {
            get_mvp_default(h, p_neighbors, pred_mv, bwd_2nd, p_cb, ref_idx);
        }

        // 需在 MVP 获取之后执行，两者都会设置 p_me 状态
        p_me->i_ref_idx = (int16_t)ref_idx;
        if (h->param->me_method == XAVS2_ME_UMH) {
            fast_me_prepare_info(h, p_me, mode, ref_idx, pu_idx, all_min_costs[0]);
        }

        /* set reference index and pointer */
        p_me->i_bias = pix_y * p_ref_frm->i_stride[IMG_Y] + pix_x;
        p_me->p_fref_1st = p_ref_frm;
        p_me->mvp.v  = pred_mv->v;

        /* 限制MVP的取值，如果MVP值过大，则不做ME */
        b_mv_valid = check_mv_range(h, pred_mv, ref_idx, pix_x, pix_y, bsx, bsy);
        b_mv_valid &= check_mvd(h, pred_mv->x, pred_mv->y);

        /* 默认必须搜索的点位置 */
        i_mvc = 0;
        i_mvc = add_one_mv_candidate(p_me, mvc, i_mvc, p_me->mvp.x, p_me->mvp.y);
        i_mvc = add_one_mv_candidate(p_me, mvc, i_mvc, 0, 0);

        if (b_mv_valid) {
            cost = xavs2_me_search(h, p_me, mvc, i_mvc);
        } else {
            p_me->bmv = p_me->mvp;  // MVP越界时，最优MV设置成和MVP一样大小
            cost = MAX_DISTORTION;
        }
        mv = p_me->bmv;

        /* store motion vectors and reference frame (for motion vector prediction) */
        p_me->all_best_imv[ref_idx] = p_me->bmv2;
        m = XAVS2_MAX(bsx >> (MIN_PU_SIZE_IN_BIT + pu_size_shift), 1);
        n = XAVS2_MAX(bsy >> (MIN_PU_SIZE_IN_BIT + pu_size_shift), 1);

        if (h->param->me_method == XAVS2_ME_UMH) {
            for (j = 0; j < n; j++) {
                for (i = 0; i < m; i++) {
                    k = ((pu_idx_y + j) << 1) + (pu_idx_x + i);
                    assert(mode >= 0 && mode < MAX_INTER_MODES && k < 4 && k >= 0);
                    p_mode_mvs[k].all_single_mv[ref_idx] = mv;
                    p_me->all_best_mv[mode][k][ref_idx] = mv;
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                for (i = 0; i < m; i++) {
                    k = ((pu_idx_y + j) << 1) + (pu_idx_x + i);
                    assert(mode >= 0 && mode < MAX_INTER_MODES && k < 4 && k >= 0);
                    p_mode_mvs[k].all_single_mv[ref_idx] = mv;
                }
            }
        }


        if (h->param->me_method == XAVS2_ME_UMH) {
            m = XAVS2_MAX(bsx >> MIN_PU_SIZE_IN_BIT, 1);
            n = XAVS2_MAX(bsy >> MIN_PU_SIZE_IN_BIT, 1);
            for (j = 0; j < n; j++) {
                for (i = 0; i < m; i++) {
                    all_min_costs[j * width_in_4x4 + i][mode][ref_idx] = p_me->bcost2;
                }
            }
        }

        b_mv_valid &= check_mv_range(h, &mv, ref_idx, pix_x, pix_y, bsx, bsy);
        b_mv_valid &= check_mvd(h, (mv.x - pred_mv->x), (mv.y - pred_mv->y));
        if (!b_mv_valid) {
            cost = MAX_DISTORTION;
        }

        if (h->i_type == SLICE_TYPE_B) {
            // for SLICE_TYPE_B: only get the forward cost
            if (ref_idx == B_FWD) {
                *fwd_cost = cost;     // forward cost
                p_me->bmvcost[PDIR_FWD] = p_me->mvcost[PDIR_FWD];
            } else {
                *bwd_cost = cost; // backward cost
                p_me->bmvcost[PDIR_BWD] = p_me->mvcost[PDIR_FWD];
            }
        } else {
            // for SLICE_TYPE_F or SLICE_TYPE_P
            cost += REF_COST(ref_idx);
            if (cost < *fwd_cost) {
                *fwd_cost    = cost;
                best_ref_idx = ref_idx;
                p_me->bmvcost[PDIR_FWD] = p_me->mvcost[PDIR_FWD];
            }
        }
    }

    return best_ref_idx;
}

/* ---------------------------------------------------------------------------
 * get cost for symirectional prediction
 */
void pred_inter_search_bi(xavs2_t *h, cu_t *p_cu, cb_t *p_cb, xavs2_me_t *p_me, dist_t *sym_mcost, dist_t *bid_mcost)
{
    int mode = p_cu->cu_info.i_mode;
    mv_t mvp, mv;
    cu_parallel_t *p_enc = cu_get_enc_context(h, p_cu->cu_info.i_level);
    cu_mv_mode_t *p_mode_mv = cu_get_layer_mode(h, p_cu->cu_info.i_level)->mvs[mode];
    pel_t *buf_pixel_temp = p_enc->buf_pixel_temp;
    int pu_size_shift = p_cu->cu_info.i_level - MIN_CU_SIZE_IN_BIT;
    dist_t cost, cost_bid;
    int m, n, i, j;
    int b_mv_valid;                    // MV是否有效：大小取值是否在标准规定的有效范围内
    int pu_idx_x = p_cb->x != 0;       // PU index in CU
    int pu_idx_y = p_cb->y != 0;
    int k = (pu_idx_y << 1) + pu_idx_x;
    int pix_x = p_cu->i_pix_x + p_cb->x;
    int pix_y = p_cu->i_pix_y + p_cb->y;
    int bsx = p_cb->w;
    int bsy = p_cb->h;
    int distance_fwd = calculate_distance(h, B_FWD);
    int distance_bwd = calculate_distance(h, B_BWD);

    // get fullpel search results
    mv_t fwd_mv = p_me->all_best_imv[B_FWD];
    mv_t bwd_mv = p_mode_mv[k].all_single_mv[B_BWD];

    assert(mode >= 0 && mode < MAX_INTER_MODES && k < 4 && k >= 0);
    // get MVP (motion vector predicator
    p_me->mvp1 = p_mode_mv[k].all_mvp[B_FWD];
    p_me->mvp2 = p_mode_mv[k].all_mvp[B_BWD];
    mvp = p_me->mvp1;

    // init motion vectors
    fwd_mv.x <<= 2;
    fwd_mv.y <<= 2;
    mv = fwd_mv;

    /* set reference index and pointer */
    p_me->i_ref_idx = B_BWD;
    p_me->p_fref_1st = h->fref[B_FWD];
    p_me->p_fref_2nd = h->fref[B_BWD];
    p_me->i_distance_1st = distance_fwd;
    p_me->i_distance_2nd = distance_bwd;

    b_mv_valid  = check_mv_range_sym(h, &mvp, pix_x, pix_y, bsx, bsy, distance_fwd, distance_bwd);
    b_mv_valid &= check_mv_range_sym(h,  &mv, pix_x, pix_y, bsx, bsy, distance_fwd, distance_bwd);
    b_mv_valid &= check_mvd(h, mvp.x, mvp.y);  // avoid mv-bits calculation error

    if (b_mv_valid) {
        cost = xavs2_me_search_sym(h, p_me, buf_pixel_temp, &mv);
    } else {
        cost = MAX_DISTORTION;
    }

    b_mv_valid  = check_mv_range(h, &fwd_mv, B_FWD, pix_x, pix_y, bsx, bsy);
    b_mv_valid &= check_mv_range(h, &bwd_mv, B_BWD, pix_x, pix_y, bsx, bsy);
    b_mv_valid &= check_mvd(h, p_me->mvp1.x, p_me->mvp1.y);  // avoid mv-bits calculation error
    b_mv_valid &= check_mvd(h, p_me->mvp2.x, p_me->mvp2.y);
    if (b_mv_valid) {
        cost_bid = xavs2_me_search_bid(h, p_me, buf_pixel_temp, &fwd_mv, &bwd_mv, p_enc);
    } else {
        cost_bid = MAX_DISTORTION;
    }

    // store motion vectors
    m = XAVS2_MAX((bsx >> (MIN_PU_SIZE_IN_BIT + pu_size_shift)), 1);
    n = XAVS2_MAX((bsy >> (MIN_PU_SIZE_IN_BIT + pu_size_shift)), 1);
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            k = ((pu_idx_y + j) << 1) + (pu_idx_x + i);
            p_mode_mv[k].all_sym_mv     [0] = mv;
            p_mode_mv[k].all_dual_mv_1st[0] = fwd_mv;
            p_mode_mv[k].all_dual_mv_2nd[0] = bwd_mv;
        }
    }

    if (!(check_mv_range(h, &fwd_mv, B_FWD, pix_x, pix_y, bsx, bsy) &&
          check_mvd(h, (fwd_mv.x - p_me->mvp1.x), (fwd_mv.y - p_me->mvp1.y)))) {
        cost_bid = MAX_DISTORTION;
    }

    if (!(check_mv_range(h, &bwd_mv, B_BWD, pix_x, pix_y, bsx, bsy) &&
          check_mvd(h, (bwd_mv.x - p_me->mvp2.x), (bwd_mv.y - p_me->mvp2.y)))) {
        cost_bid = MAX_DISTORTION;
    }

    if (!(check_mv_range_sym(h, &mv, pix_x, pix_y, bsx, bsy, distance_fwd, distance_bwd) &&
          check_mvd(h, (mv.x - mvp.x), (mv.y - mvp.y)))) {
        cost = MAX_DISTORTION;
    }
    p_me->bmvcost[PDIR_SYM] = p_me->mvcost[PDIR_SYM];
    p_me->bmvcost[PDIR_BID] = p_me->mvcost[PDIR_BID];

    *sym_mcost = cost;
    *bid_mcost = cost_bid;
}

/* ---------------------------------------------------------------------------
 * get cost for dual hypothesis prediction
 */
void pred_inter_search_dual(xavs2_t *h, cu_t *p_cu, cb_t *p_cb, xavs2_me_t *p_me,
                            dist_t *dual_mcost, int *dual_best_fst_ref, int *dual_best_snd_ref)
{
    int mode = p_cu->cu_info.i_mode;
    mv_t fst_dual, snd_dual;
    cu_parallel_t *p_enc = cu_get_enc_context(h, p_cu->cu_info.i_level);
    cu_mv_mode_t *p_mode_mv = cu_get_layer_mode(h, p_cu->cu_info.i_level)->mvs[mode];
    pel_t *buf_pixel_temp = p_enc->buf_pixel_temp;
    int pix_x = p_cu->i_pix_x + p_cb->x;
    int pix_y = p_cu->i_pix_y + p_cb->y;
    int pu_idx_x = p_cb->x != 0;           // PU index
    int pu_idx_y = p_cb->y != 0;
    int bsx = p_cb->w;  // block size
    int bsy = p_cb->h;
    int pu_size_shift = p_cu->cu_info.i_level - MIN_CU_SIZE_IN_BIT;
    int ref_idx;
    dist_t cost;
    int distance_fwd, distance_bwd;
    int b_mv_valid;        // MV是否有效：大小取值是否在标准规定的有效范围内
    int m, n, i, j, k;
    int max_ref = h->i_ref;

    *dual_mcost = MAX_DISTORTION;

    // loop over reference frames
    for (ref_idx = 0; ref_idx < max_ref; ref_idx++) {
        int snd_ref = !ref_idx;

        // get MVPs(motion vector predictors)
        k = (pu_idx_y << 1) + pu_idx_x;
        assert(mode >= 0 && mode < MAX_INTER_MODES && k < 4 && k >= 0);
        p_me->mvp1 = p_mode_mv[k].all_mvp[ref_idx];

        /* set reference index and pointer */
        p_me->i_ref_idx = (int16_t)ref_idx;
        p_me->i_distance_1st = distance_fwd = calculate_distance(h, ref_idx);
        p_me->i_distance_2nd = distance_bwd = calculate_distance(h, snd_ref);
        p_me->p_fref_1st = h->fref[ref_idx];
        p_me->p_fref_2nd = h->fref[snd_ref];

        // get the best fullpel search result
        fst_dual = p_me->all_best_imv[ref_idx];  // only for F frame, B frame are not called here
        fst_dual.x <<= 2;
        fst_dual.y <<= 2;

        // get the min motion cost for dual hypothesis prediction
        b_mv_valid  = check_mv_range_sym(h, &fst_dual, pix_x, pix_y, bsx, bsy, distance_fwd, distance_bwd);
        b_mv_valid &= check_mvd(h, (fst_dual.x - p_me->mvp1.x), (fst_dual.y - p_me->mvp1.y));
        b_mv_valid &= check_mvd(h, p_me->mvp1.x, p_me->mvp1.y);
        b_mv_valid &= check_mvd(h, p_me->mvp.x, p_me->mvp.y);
        if (b_mv_valid) {
            cost = xavs2_me_search_sym(h, p_me, buf_pixel_temp, &fst_dual);
        } else {
            cost = MAX_DISTORTION;
        }

        /* store motion vectors and reference frame (for motion vector prediction) */
        snd_dual.v = MAKEDWORD(scale_mv_skip  (   fst_dual.x, distance_bwd, distance_fwd),
                               scale_mv_skip_y(h, fst_dual.y, distance_bwd, distance_fwd));

        m = XAVS2_MAX((bsx >> (MIN_PU_SIZE_IN_BIT + pu_size_shift)), 1);
        n = XAVS2_MAX((bsy >> (MIN_PU_SIZE_IN_BIT + pu_size_shift)), 1);
        for (j = 0; j < n; j++) {
            for (i = 0; i < m; i++) {
                k = ((pu_idx_y + j) << 1) + (pu_idx_x + i);
                p_mode_mv[k].all_dual_mv_1st[ref_idx] = fst_dual;
                p_mode_mv[k].all_dual_mv_2nd[ref_idx] = snd_dual;
            }
        }

        b_mv_valid &= check_mv_range_sym(h, &fst_dual, pix_x, pix_y, bsx, bsy, distance_fwd, distance_bwd);
        b_mv_valid &= check_mvd(h, (fst_dual.x - p_me->mvp1.x), (fst_dual.y - p_me->mvp1.y));
        if (!b_mv_valid) {
            cost = MAX_DISTORTION;
        } else {
            cost += REF_COST(ref_idx);
            if (cost < *dual_mcost) {
                *dual_mcost = cost;
                *dual_best_fst_ref = ref_idx;
                *dual_best_snd_ref = !ref_idx;
                p_me->bmvcost[PDIR_DUAL] = p_me->mvcost[PDIR_SYM];
            }
        }
    }
}


