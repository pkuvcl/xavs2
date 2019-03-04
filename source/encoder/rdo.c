/*
 * rdo.c
 *
 * Description of this file:
 *    RDO functions definition of the xavs2 library
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
#include "rdo.h"
#include "cudata.h"
#include "aec.h"
#include "common/mc.h"
#include "transform.h"
#include "block_info.h"
#include "wquant.h"
#include "me.h"
#include "cpu.h"
#include "predict.h"
#include "ratecontrol.h"
#include "rdoq.h"


/**
 * ===========================================================================
 * local/global variables
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static const float SUBCU_COST_RATE[2][4] = {
    {0.50f, 0.75f, 0.97f, 1.0f},   /* 帧内CU的Cost一般都较大 */
    {0.75f, 0.90f, 0.99f, 1.0f},   /* 帧间情况下，Skip块Cost很小 */
};

static const int tab_pdir_bskip[DS_MAX_NUM] = {
    PDIR_BID, PDIR_BWD, PDIR_SYM, PDIR_FWD
};

/* ---------------------------------------------------------------------------
 */
static const int8_t NUM_PREDICTION_UNIT[MAX_PRED_MODES] = {// [mode]
    1, // 0: 8x8, ---, ---, --- (PRED_SKIP   )
    1, // 1: 8x8, ---, ---, --- (PRED_2Nx2N  )
    2, // 2: 8x4, 8x4, ---, --- (PRED_2NxN   )
    2, // 3: 4x8, 4x8, ---, --- (PRED_Nx2N   )
    2, // 4: 8x2, 8x6, ---, --- (PRED_2NxnU  )
    2, // 5: 8x6, 8x2, ---, --- (PRED_2NxnD  )
    2, // 6: 2x8, 6x8, ---, --- (PRED_nLx2N  )
    2, // 7: 6x8, 2x8, ---, --- (PRED_nRx2N  )
    1, // 8: 8x8, ---, ---, --- (PRED_I_2Nx2N)
    4, // 9: 4x4, 4x4, 4x4, 4x4 (PRED_I_NxN  )
    4, //10: 8x2, 8x2, 8x2, 8x2 (PRED_I_2Nxn )
    4  //11: 2x8, 2x8, 2x8, 2x8 (PRED_I_nx2N )
};

static const cb_t CODING_BLOCK_INFO[MAX_PRED_MODES + 1][4] = {// [mode][block]
    //   x, y, w, h        x, y, w, h        x, y, w, h        x, y, w, h          for block 0, 1, 2 and 3
    { { {0, 0, 8, 8} }, { {0, 0, 0, 0} }, { {0, 0, 0, 0} }, { {0, 0, 0, 0} } }, // 0: 8x8, ---, ---, --- (PRED_SKIP   )
    { { {0, 0, 8, 8} }, { {0, 0, 0, 0} }, { {0, 0, 0, 0} }, { {0, 0, 0, 0} } }, // 1: 8x8, ---, ---, --- (PRED_2Nx2N  )
    { { {0, 0, 8, 4} }, { {0, 4, 8, 4} }, { {0, 0, 0, 0} }, { {0, 0, 0, 0} } }, // 2: 8x4, 8x4, ---, --- (PRED_2NxN   )
    { { {0, 0, 4, 8} }, { {4, 0, 4, 8} }, { {0, 0, 0, 0} }, { {0, 0, 0, 0} } }, // 3: 4x8, 4x8, ---, --- (PRED_Nx2N   )
    { { {0, 0, 8, 2} }, { {0, 2, 8, 6} }, { {0, 0, 0, 0} }, { {0, 0, 0, 0} } }, // 4: 8x2, 8x6, ---, --- (PRED_2NxnU  )
    { { {0, 0, 8, 6} }, { {0, 6, 8, 2} }, { {0, 0, 0, 0} }, { {0, 0, 0, 0} } }, // 5: 8x6, 8x2, ---, --- (PRED_2NxnD  )
    { { {0, 0, 2, 8} }, { {2, 0, 6, 8} }, { {0, 0, 0, 0} }, { {0, 0, 0, 0} } }, // 6: 2x8, 6x8, ---, --- (PRED_nLx2N  )
    { { {0, 0, 6, 8} }, { {6, 0, 2, 8} }, { {0, 0, 0, 0} }, { {0, 0, 0, 0} } }, // 7: 6x8, 2x8, ---, --- (PRED_nRx2N  )
    { { {0, 0, 8, 8} }, { {0, 0, 0, 0} }, { {0, 0, 0, 0} }, { {0, 0, 0, 0} } }, // 8: 8x8, ---, ---, --- (PRED_I_2Nx2N)
    { { {0, 0, 4, 4} }, { {4, 0, 4, 4} }, { {0, 4, 4, 4} }, { {4, 4, 4, 4} } }, // 9: 4x4, 4x4, 4x4, 4x4 (PRED_I_NxN  )
    { { {0, 0, 8, 2} }, { {0, 2, 8, 2} }, { {0, 4, 8, 2} }, { {0, 6, 8, 2} } }, //10: 8x2, 8x2, 8x2, 8x2 (PRED_I_2Nxn )
    { { {0, 0, 2, 8} }, { {2, 0, 2, 8} }, { {4, 0, 2, 8} }, { {6, 0, 2, 8} } }, //11: 2x8, 2x8, 2x8, 2x8 (PRED_I_nx2N )
    { { {0, 0, 4, 4} }, { {4, 0, 4, 4} }, { {0, 4, 4, 4} }, { {4, 4, 4, 4} } }, // X: 4x4, 4x4, 4x4, 4x4
};

static const int8_t TU_SPLIT_TYPE[MAX_PRED_MODES][2] = {  // [mode][(NsqtEnable or SdipEnables) and cu_level > B8X8_IN_BIT]
    //  split_type for block non-sdip/nsqt:[0] and sdip/nsqt:[1]
    { TU_SPLIT_CROSS,   TU_SPLIT_CROSS   },   // 0: 8x8, ---, ---, --- (PRED_SKIP   )
    { TU_SPLIT_CROSS,   TU_SPLIT_CROSS   },   // 1: 8x8, ---, ---, --- (PRED_2Nx2N  )
    { TU_SPLIT_CROSS,   TU_SPLIT_HOR     },   // 2: 8x4, 8x4, ---, --- (PRED_2NxN   )
    { TU_SPLIT_CROSS,   TU_SPLIT_VER     },   // 3: 4x8, 4x8, ---, --- (PRED_Nx2N   )
    { TU_SPLIT_CROSS,   TU_SPLIT_HOR     },   // 4: 8x2, 8x6, ---, --- (PRED_2NxnU  )
    { TU_SPLIT_CROSS,   TU_SPLIT_HOR     },   // 5: 8x6, 8x2, ---, --- (PRED_2NxnD  )
    { TU_SPLIT_CROSS,   TU_SPLIT_VER     },   // 6: 2x8, 6x8, ---, --- (PRED_nLx2N  )
    { TU_SPLIT_CROSS,   TU_SPLIT_VER     },   // 7: 6x8, 2x8, ---, --- (PRED_nRx2N  )
    { TU_SPLIT_NON,     TU_SPLIT_INVALID },   // 8: 8x8, ---, ---, --- (PRED_I_2Nx2N)
    { TU_SPLIT_CROSS,   TU_SPLIT_CROSS   },   // 9: 4x4, 4x4, 4x4, 4x4 (PRED_I_NxN  )
    { TU_SPLIT_INVALID, TU_SPLIT_HOR     },   //10: 8x2, 8x2, 8x2, 8x2 (PRED_I_2Nxn )
    { TU_SPLIT_INVALID, TU_SPLIT_VER     }    //11: 2x8, 2x8, 2x8, 2x8 (PRED_I_nx2N )
};

static const int8_t headerbits_skipmode[8] = { 2, 3, 4, 4, 3, 4, 5, 5 };//temporal wsm1 wsm2 wsm3 spatial_direct0 spatial_direct1 spatial_direct2 spatial_direct3
/**
 * ===========================================================================
 * local function defines (utilities)
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * 依据CU划分模式确定当前CU包含的PU数量和大小（帧间划分）
 */
static ALWAYS_INLINE
void cu_init_pu_inter(xavs2_t *h, cu_info_t *p_cu_info, int i_level, int i_mode)
{
    const int shift_bits = i_level - MIN_CU_SIZE_IN_BIT;
    const int8_t block_num = NUM_PREDICTION_UNIT[i_mode];
    int ds_mode = p_cu_info->directskip_mhp_idx;
    int i;
    cb_t *p_cb = p_cu_info->cb;

    // set for each block
    if (i_mode == PRED_SKIP) {
        ///! 一些特殊的Skip/Direct模式下如果CU超过8x8，则PU划分成4个
        if (i_level > 3 && (h->i_type == SLICE_TYPE_P || (h->i_type == SLICE_TYPE_F && ds_mode == DS_NONE)
                            || (h->i_type == SLICE_TYPE_B && ds_mode == DS_NONE))) {
            p_cu_info->num_pu = 4;
            for (i = 0; i < 4; i++) {
                p_cb[i].v = CODING_BLOCK_INFO[PRED_I_nx2N + 1][i].v << shift_bits;
            }
        } else {
            p_cu_info->num_pu = 1;
            memset(p_cu_info->cb, 0, sizeof(p_cu_info->cb));
            p_cb[0].v = CODING_BLOCK_INFO[PRED_SKIP][0].v << shift_bits;
        }
    } else {
        p_cu_info->num_pu = block_num;
        for (i = 0; i < block_num; i++) {
            p_cb[i].v = CODING_BLOCK_INFO[i_mode][i].v << shift_bits;
        }
    }
}

/* ---------------------------------------------------------------------------
 * 依据CU划分模式确定当前CU包含的PU数量和大小（帧内划分）
 */
static ALWAYS_INLINE
void cu_init_pu_intra(xavs2_t *h, cu_info_t *p_cu_info, int i_level, int i_mode)
{
    const int shift_bits = i_level - MIN_CU_SIZE_IN_BIT;
    const int8_t block_num = NUM_PREDICTION_UNIT[i_mode];
    int i;
    cb_t *p_cb = p_cu_info->cb;

    UNUSED_PARAMETER(h);
    // set for each block
    p_cu_info->num_pu = block_num;
    for (i = 0; i < 4; i++) {
        p_cb[i].v = CODING_BLOCK_INFO[i_mode][i].v << shift_bits;
    }
}

/* ---------------------------------------------------------------------------
 * TU split type when TU split is enabled for current CU
 */
static ALWAYS_INLINE
void cu_set_tu_split_type(xavs2_t *h, cu_info_t *p_cu_info, int transform_split_flag)
{
    int mode = p_cu_info->i_mode;
    int level = p_cu_info->i_level;
    int enable_nsqt_sdip = IS_INTRA_MODE(mode) ? h->param->enable_sdip : h->param->enable_nsqt;

    enable_nsqt_sdip = enable_nsqt_sdip && level > B8X8_IN_BIT;
    p_cu_info->i_tu_split = transform_split_flag ? TU_SPLIT_TYPE[mode][enable_nsqt_sdip] : TU_SPLIT_NON;
    assert(p_cu_info->i_tu_split != TU_SPLIT_INVALID);
}


/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE uint32_t
cu_get_valid_modes(xavs2_t *h, int frm_type, int level)
{
    return h->valid_modes[frm_type][level - MIN_CU_SIZE_IN_BIT];
}

/* ---------------------------------------------------------------------------
 */
static INLINE
void cu_init(xavs2_t *h, cu_t *p_cu, cu_info_t *best, int i_level)
{
    cu_layer_t *p_layer  = cu_get_layer(h, i_level);
    int i;

    /* Ping-pong buffer */
    p_layer->buf_pred_inter      = p_layer->buf_pred_inter_luma[0];
    p_layer->buf_pred_inter_best = p_layer->buf_pred_inter_luma[1];

    /* init rec and coeff pointer */
    p_cu->cu_info.p_rec  [0]      = p_layer->rec_buf_y [0];
    p_cu->cu_info.p_coeff[0]      = p_layer->coef_buf_y[0];
    p_layer->p_rec_tmp   [0]      = p_layer->rec_buf_y [1];
    p_layer->p_coeff_tmp [0]      = p_layer->coef_buf_y[1];
    best->p_rec          [0]      = p_layer->rec_buf_y [2];
    best->p_coeff        [0]      = p_layer->coef_buf_y[2];

    p_cu->cu_info.p_rec  [1]      = p_layer->rec_buf_uv [0][0];
    p_cu->cu_info.p_coeff[1]      = p_layer->coef_buf_uv[0][0];
    p_layer->p_rec_tmp   [1]      = p_layer->rec_buf_uv [0][1];
    p_layer->p_coeff_tmp [1]      = p_layer->coef_buf_uv[0][1];
    best->p_rec          [1]      = p_layer->rec_buf_uv [0][2];
    best->p_coeff        [1]      = p_layer->coef_buf_uv[0][2];

    p_cu->cu_info.p_rec  [2]      = p_layer->rec_buf_uv [1][0];
    p_cu->cu_info.p_coeff[2]      = p_layer->coef_buf_uv[1][0];
    p_layer->p_rec_tmp   [2]      = p_layer->rec_buf_uv [1][1];
    p_layer->p_coeff_tmp [2]      = p_layer->coef_buf_uv[1][1];
    best->p_rec          [2]      = p_layer->rec_buf_uv [1][2];
    best->p_coeff        [2]      = p_layer->coef_buf_uv[1][2];

    /* init basic properties */
    p_cu->cu_info.i_cbp = 0;

#if ENABLE_RATE_CONTROL_CU
    /* set qp needed in loop filter (even if constant QP is used) */
    p_cu->cu_info.i_cu_qp = h->i_qp;

    if (h->param->i_rc_method == XAVS2_RC_CBR_SCU) {
        int i_left_cu_qp;
        if (p_cu->i_pix_x > 0) {
            i_left_cu_qp = h->cu_info[p_cu->i_scu_xy - 1].i_cu_qp;
        } else {
            i_left_cu_qp = h->i_qp;
        }

        p_cu->cu_info.i_delta_qp = p_cu->cu_info.i_cu_qp - i_left_cu_qp;
    } else {
        p_cu->cu_info.i_delta_qp = 0;
    }
#endif

    /* ref_idx_1st[], ref_idx_2nd[] 内存连续 */
    memset(p_cu->cu_info.ref_idx_1st, INVALID_REF, sizeof(p_cu->cu_info.ref_idx_1st) + sizeof(p_cu->cu_info.ref_idx_2nd));

    /* init position for 4 sub-CUs */
    if (i_level > B8X8_IN_BIT) {
        for (i = 0; i < 4; i++) {
            cu_t *p_sub_cu = p_cu->sub_cu[i];

            p_sub_cu->i_pix_x          = p_cu->i_pix_x + ((i &  1) << (i_level - 1));
            p_sub_cu->i_pix_y          = p_cu->i_pix_y + ((i >> 1) << (i_level - 1));
            p_sub_cu->cu_info.i_scu_x  = p_sub_cu->i_pix_x >> MIN_CU_SIZE_IN_BIT;
            p_sub_cu->cu_info.i_scu_y  = p_sub_cu->i_pix_y >> MIN_CU_SIZE_IN_BIT;
            p_sub_cu->i_scu_xy         = p_sub_cu->cu_info.i_scu_y * h->i_width_in_mincu + p_sub_cu->cu_info.i_scu_x;
        }
    }

    /* set neighbor CUs */
    check_neighbor_cu_avail(h, p_cu, p_cu->cu_info.i_scu_x, p_cu->cu_info.i_scu_y, p_cu->i_scu_xy);
    p_cu->b_cbp_direct = 0;
}

/* ---------------------------------------------------------------------------
 * copy information of CU
 */
static ALWAYS_INLINE
void cu_copy_info(cu_info_t *p_dst, const cu_info_t *p_src)
{
    const int num_bytes = sizeof(cu_info_t) - (int)((uint8_t *)&p_dst->i_level - (uint8_t *)p_dst);
    memcpy(&p_dst->i_level, &p_src->i_level, num_bytes);
}

/* ---------------------------------------------------------------------------
 * store cu parameters to best
 */
static
void cu_store_parameters(xavs2_t *h, cu_t *p_cu, cu_info_t *best)
{
    int mode = p_cu->cu_info.i_mode;
    cu_mode_t *p_mode = cu_get_layer_mode(h, p_cu->cu_info.i_level);

    // store best mode
    cu_copy_info(best, &p_cu->cu_info);

    /* --- reconstructed blocks ---- */
    XAVS2_SWAP_PTR(best->p_rec[0], p_cu->cu_info.p_rec[0]);
    XAVS2_SWAP_PTR(best->p_rec[1], p_cu->cu_info.p_rec[1]);
    XAVS2_SWAP_PTR(best->p_rec[2], p_cu->cu_info.p_rec[2]);

    /* ---- residual (coefficients) ---- */
    XAVS2_SWAP_PTR(best->p_coeff[0], p_cu->cu_info.p_coeff[0]);
    XAVS2_SWAP_PTR(best->p_coeff[1], p_cu->cu_info.p_coeff[1]);
    XAVS2_SWAP_PTR(best->p_coeff[2], p_cu->cu_info.p_coeff[2]);

    /* ---- prediction information ---- */
    if (!IS_INTRA_MODE(mode)) {
        memcpy(&p_mode->best_mc, &p_cu->mc, sizeof(p_cu->mc));
    }
}

/* ---------------------------------------------------------------------------
 * sets motion vectors and reference indexes for an CU
 */
static void cu_save_mvs_refs(xavs2_t *h, cu_info_t *p_cu_info)
{
    int8_t *p_dirpred = h->dir_pred;
    int8_t *p_ref_1st = h->fwd_1st_ref;
    int8_t *p_ref_2nd = h->bwd_2nd_ref;
    mv_t   *p_mv_1st  = h->fwd_1st_mv;
    mv_t   *p_mv_2nd  = h->bwd_2nd_mv;
    int bx_4x4_cu     = p_cu_info->i_scu_x << (MIN_CU_SIZE_IN_BIT - MIN_PU_SIZE_IN_BIT);
    int by_4x4_cu     = p_cu_info->i_scu_y << (MIN_CU_SIZE_IN_BIT - MIN_PU_SIZE_IN_BIT);
    int w_in_4x4      = h->i_width_in_minpu;
    cu_mode_t *p_cu_mode = cu_get_layer_mode(h, p_cu_info->i_level);
    cu_mc_param_t *p_mc = &p_cu_mode->best_mc;
    int width, height;
    int k, by, bx;
    int r, c;

    for (k = 0; k < p_cu_info->num_pu; k++) {
        int8_t i_dir_pred = (int8_t)p_cu_info->b8pdir[k];
        int8_t ref_1st = (int8_t)p_cu_info->ref_idx_1st[k];
        int8_t ref_2nd = (int8_t)p_cu_info->ref_idx_2nd[k];
        mv_t   mv_1st  = p_mc->mv[k][0];
        mv_t   mv_2nd  = p_mc->mv[k][1];
        cb_t   cur_cb  = p_cu_info->cb[k];

        cur_cb.v >>= 2;
        bx         = cur_cb.x;
        by         = cur_cb.y;
        width      = cur_cb.w;
        height     = cur_cb.h;

        bx += bx_4x4_cu;
        by += by_4x4_cu;

        for (r = 0; r < height; r++) {
            int offset = (by + r) * w_in_4x4 + bx;
            for (c = 0; c < width; c++) {
                p_dirpred[offset + c] = i_dir_pred;

                p_mv_1st [offset + c] = mv_1st;
                p_mv_2nd [offset + c] = mv_2nd;

                p_ref_1st[offset + c] = ref_1st;
                p_ref_2nd[offset + c] = ref_2nd;
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * set stored cu parameters
 */
static INLINE
void cu_copy_stored_parameters(xavs2_t *h, cu_t *p_cu, cu_info_t *best)
{
    int mode       = best->i_mode;
    int w_in_4x4   = h->i_width_in_minpu;
    int scu_xy     = p_cu->i_scu_xy;
    int b4x4_x     = p_cu->i_pix_x >> MIN_PU_SIZE_IN_BIT;
    int b4x4_y     = p_cu->i_pix_y >> MIN_PU_SIZE_IN_BIT;
    int pix_x      = p_cu->i_pos_x;
    int pix_y      = p_cu->i_pos_y;
    int pix_cx     = pix_x >> 1;
    int pix_cy     = pix_y >> 1;
    int blocksize  = p_cu->i_size;
    int ip_stride  = h->i_width_in_minpu + 16;
    int part_idx_c = PART_INDEX(blocksize >> 1, blocksize >> 1);
    int8_t *p_intramode   = h->ipredmode + (pix_y >> MIN_PU_SIZE_IN_BIT) * ip_stride + b4x4_x;
    const int size_in_spu = (blocksize >> MIN_PU_SIZE_IN_BIT);
    const int size_in_scu = (blocksize >> MIN_CU_SIZE_IN_BIT);
    int i, j;

    cu_copy_info(&p_cu->cu_info, best);

    //===== reconstruction values =====
    g_funcs.pixf.copy_pp[PART_INDEX(blocksize, blocksize)](h->lcu.p_fdec[0] + pix_y * FDEC_STRIDE + pix_x, FDEC_STRIDE,
            best->p_rec[0], FREC_STRIDE);
    g_funcs.pixf.copy_ss[PART_INDEX(blocksize, blocksize)](h->lcu.lcu_coeff[0] + (p_cu->idx_zorder << 6), blocksize,
            best->p_coeff[0], blocksize);

    g_funcs.pixf.copy_pp[part_idx_c](h->lcu.p_fdec[1] + pix_cy * FDEC_STRIDE + pix_cx, FDEC_STRIDE,
                                     best->p_rec[1], FREC_CSTRIDE / 2);
    g_funcs.pixf.copy_pp[part_idx_c](h->lcu.p_fdec[2] + pix_cy * FDEC_STRIDE + pix_cx, FDEC_STRIDE,
                                     best->p_rec[2], FREC_CSTRIDE / 2);
    g_funcs.pixf.copy_ss[part_idx_c](h->lcu.lcu_coeff[1] + (p_cu->idx_zorder << 4), blocksize >> 1,
                                     best->p_coeff[1], blocksize >> 1);
    g_funcs.pixf.copy_ss[part_idx_c](h->lcu.lcu_coeff[2] + (p_cu->idx_zorder << 4), blocksize >> 1,
                                     best->p_coeff[2], blocksize >> 1);

    //===============   cbp and mode   ===============
    for (j = 0; j < size_in_scu; j++) {
        cu_info_t *p_cu_info = &h->cu_info[j * h->i_width_in_mincu + scu_xy];  // save data to cu_info
        for (i = size_in_scu; i != 0; i--) {
            cu_copy_info(p_cu_info++, best);
        }
    }

    //===============   intra pred mode   ===============
    if (IS_INTRA_MODE(mode)) {
        int n_size4 = size_in_spu >> 2;
        int k;
        int8_t intra_pred_mode;

        switch (mode) {
        case PRED_I_2Nxn:
            for (i = 0; i < 4; i++) {
                for (j = i * n_size4; j < (i + 1) * n_size4; j++) {
                    g_funcs.fast_memset(p_intramode, p_cu->cu_info.real_intra_modes[i], size_in_spu * sizeof(int8_t));
                    p_intramode += ip_stride;
                }
            }
            break;
        case PRED_I_nx2N:
            for (j = 0; j < size_in_spu; j++) {
                for (i = 0; i < 4; i++) {
                    k = i * n_size4;
                    g_funcs.fast_memset(p_intramode + k, p_cu->cu_info.real_intra_modes[i], n_size4 * sizeof(int8_t));
                }
                p_intramode += ip_stride;
            }
            break;
        case PRED_I_NxN:
            n_size4 = size_in_spu >> 1;
            for (j = 0; j < n_size4; j++) {
                for (i = 0; i < 2; i++) {
                    k = i * n_size4;
                    g_funcs.fast_memset(p_intramode + k, p_cu->cu_info.real_intra_modes[i], n_size4 * sizeof(int8_t));
                }
                p_intramode += ip_stride;
            }
            for (j = n_size4; j < size_in_spu; j++) {
                for (i = 0; i < 2; i++) {
                    k = i * n_size4;
                    g_funcs.fast_memset(p_intramode + k, p_cu->cu_info.real_intra_modes[i + 2], n_size4 * sizeof(int8_t));
                }
                p_intramode += ip_stride;
            }
            break;
        default: // PRED_2Nx2N
            intra_pred_mode = p_cu->cu_info.real_intra_modes[0];
            for (j = size_in_spu - 1; j != 0; j--) {
                p_intramode[size_in_spu - 1] = intra_pred_mode;
                p_intramode += ip_stride;
            }
            g_funcs.fast_memset(p_intramode, intra_pred_mode, size_in_spu * sizeof(int8_t));
            break;
        }
    } else if (h->i_type != SLICE_TYPE_I && h->fenc->b_enable_intra) {
        for (j = size_in_spu - 1; j != 0; j--) {
            p_intramode[size_in_spu - 1] = -1;
            p_intramode += ip_stride;
        }
        g_funcs.fast_memset(p_intramode, -1, size_in_spu * sizeof(int8_t));
    }

    //============== inter prediction information =========================
    if (h->i_type != SLICE_TYPE_I) {
        if (IS_INTER_MODE(p_cu->cu_info.i_mode)) {
            cu_save_mvs_refs(h, &p_cu->cu_info); // store mv
        } else {
            int8_t *p_dirpred = h->dir_pred    + b4x4_y * w_in_4x4 + b4x4_x;
            int8_t *p_ref_1st = h->fwd_1st_ref + b4x4_y * w_in_4x4 + b4x4_x;
            int8_t *p_ref_2nd = h->bwd_2nd_ref + b4x4_y * w_in_4x4 + b4x4_x;
            int size_b4 = blocksize >> MIN_PU_SIZE_IN_BIT;

            for (i = size_b4; i != 0; i--) {
                for (j = 0; j < size_b4; j++) {
                    p_ref_1st[j] = INVALID_REF;
                    p_ref_2nd[j] = INVALID_REF;
                    p_dirpred[j] = PDIR_INVALID;
                }
                p_dirpred += w_in_4x4;
                p_ref_1st += w_in_4x4;
                p_ref_2nd += w_in_4x4;
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * get spatial neighboring MV
 */
static ALWAYS_INLINE
void cu_get_neighbor_spatial(xavs2_t *h, int cur_slice_idx, neighbor_inter_t *p_neighbor, int x4, int y4)
{
    int pos           = y4 * h->i_width_in_minpu + x4;
    int y_outside_pic = y4 < 0 || y4 >= h->i_height_in_minpu;
    int x_outside_pic = x4 < 0 || x4 >= h->i_width_in_minpu;

    // scu_xy = XAVS2_MIN(h->i_width_in_mincu * h->i_height_in_mincu - 1, XAVS2_MAX(0, scu_xy));
    if (y_outside_pic || x_outside_pic || cu_get_slice_index(h, x4 >> 1, y4 >> 1) != cur_slice_idx) {
        p_neighbor->is_available = 0;
        p_neighbor->i_dir_pred = PDIR_INVALID;
        p_neighbor->ref_idx[0] = INVALID_REF;
        p_neighbor->ref_idx[1] = INVALID_REF;
        p_neighbor->mv[0].v    = 0;
        p_neighbor->mv[1].v    = 0;
    } else {
        p_neighbor->is_available = 1;
        p_neighbor->i_dir_pred = h->dir_pred[pos];
        p_neighbor->ref_idx[0] = h->fwd_1st_ref[pos];
        p_neighbor->ref_idx[1] = h->bwd_2nd_ref[pos];
        p_neighbor->mv[0]      = h->fwd_1st_mv[pos];
        p_neighbor->mv[1]      = h->bwd_2nd_mv[pos];
    }
}

/* ---------------------------------------------------------------------------
 * get temporal MV predictor
 */
static ALWAYS_INLINE
void cu_get_neighbor_temporal(xavs2_t *h, neighbor_inter_t *p_neighbor, int x4, int y4)
{
    int w_in_16x16 = (h->i_width_in_minpu + 3) >> 2;
    int pos = (y4 >> 2) * w_in_16x16 + (x4 >> 2);

    p_neighbor->is_available = 1;
    p_neighbor->i_dir_pred   = PDIR_FWD;
    p_neighbor->ref_idx[0]   = h->fref[0]->pu_ref[pos];
    p_neighbor->mv[0]        = h->fref[0]->pu_mv[pos];
    p_neighbor->ref_idx[1]   = INVALID_REF;
    p_neighbor->mv[1].v      = 0;
}

/* ---------------------------------------------------------------------------
 * get neighboring MVs for MVP
 */
void cu_get_neighbors(xavs2_t *h, cu_t *p_cu, cb_t *p_cb)
{
    neighbor_inter_t *neighbors = cu_get_layer(h, p_cu->cu_info.i_level)->neighbor_inter;
    int cur_slice_idx = cu_get_slice_index(h, p_cu->i_pix_x >> MIN_CU_SIZE_IN_BIT, p_cu->i_pix_y >> MIN_CU_SIZE_IN_BIT);
    int bx_4x4 = p_cu->i_pix_x >> MIN_PU_SIZE_IN_BIT;
    int by_4x4 = p_cu->i_pix_y >> MIN_PU_SIZE_IN_BIT;
    int xx0 = (p_cb->x >> MIN_PU_SIZE_IN_BIT) + bx_4x4;
    int yy0 = (p_cb->y >> MIN_PU_SIZE_IN_BIT) + by_4x4;
    int xx1 = (p_cb->w >> MIN_PU_SIZE_IN_BIT) + xx0 - 1;
    int yy1 = (p_cb->h >> MIN_PU_SIZE_IN_BIT) + yy0 - 1;

    /* 1. check whether the top-right 4x4 block is reconstructed */
    int x_TR_4x4_in_lcu = xx1 - (h->lcu.i_scu_x << (MIN_CU_SIZE_IN_BIT - MIN_PU_SIZE_IN_BIT));
    int y_TR_4x4_in_lcu = yy0 - (h->lcu.i_scu_y << (MIN_CU_SIZE_IN_BIT - MIN_PU_SIZE_IN_BIT));
    int b_available_TR  = h->tab_avail_TR[(y_TR_4x4_in_lcu << (h->i_lcu_level - B4X4_IN_BIT)) + x_TR_4x4_in_lcu];

    /* 2. get neighboring blocks */
    /* 左上 */
    cu_get_neighbor_spatial(h, cur_slice_idx, &neighbors[BLK_TOPLEFT ], xx0 - 1, yy0 - 1);

    /* 左邻的PU信息 */
    if (IS_VER_PU_PART(p_cu->cu_info.i_mode) && p_cb->x != 0) {  // CU垂直划分为两个PU，且当前PU为右边一个
        neighbor_inter_t *p_neighbor = neighbors + BLK_LEFT;
        p_neighbor->is_available = 1;
        // cu_get_neighbor_spatial(h, cur_slice_idx, &neighbors[BLK_LEFT], xx0 - 1, yy0);
        p_neighbor->i_dir_pred = p_cu->cu_info.b8pdir[0];
        p_neighbor->ref_idx[0] = p_cu->cu_info.ref_idx_1st[0];
        p_neighbor->ref_idx[1] = p_cu->cu_info.ref_idx_2nd[0];
        p_neighbor->mv[0]      = p_cu->mc.mv[0][0];
        p_neighbor->mv[1]      = p_cu->mc.mv[0][1];
        memcpy(&neighbors[BLK_LEFT2], p_neighbor, sizeof(neighbor_inter_t));
    } else {
        cu_get_neighbor_spatial(h, cur_slice_idx, &neighbors[BLK_LEFT], xx0 - 1, yy0);
        cu_get_neighbor_spatial(h, cur_slice_idx, &neighbors[BLK_LEFT2], xx0 - 1, yy1);
    }

    /* 上邻的PU信息 */
    if (IS_HOR_PU_PART(p_cu->cu_info.i_mode) && p_cb->y != 0) {  // CU水平划分为两个PU，且当前PU为下边一个
        neighbor_inter_t *p_neighbor = neighbors + BLK_TOP;
        p_neighbor->is_available = 1;
        // cu_get_neighbor_spatial(h, cur_slice_idx, &neighbors[BLK_LEFT], xx0 - 1, yy0);
        p_neighbor->i_dir_pred = p_cu->cu_info.b8pdir[0];
        p_neighbor->ref_idx[0] = p_cu->cu_info.ref_idx_1st[0];
        p_neighbor->ref_idx[1] = p_cu->cu_info.ref_idx_2nd[0];
        p_neighbor->mv[0]      = p_cu->mc.mv[0][0];
        p_neighbor->mv[1]      = p_cu->mc.mv[0][1];
        memcpy(&neighbors[BLK_TOP2], p_neighbor, sizeof(neighbor_inter_t));
    } else {
        cu_get_neighbor_spatial(h, cur_slice_idx, &neighbors[BLK_TOP], xx0, yy0 - 1);
        cu_get_neighbor_spatial(h, cur_slice_idx, &neighbors[BLK_TOP2], xx1, yy0 - 1);
    }

    /* 右上 */
    cu_get_neighbor_spatial(h, cur_slice_idx, &neighbors[BLK_TOPRIGHT], b_available_TR ? xx1 + 1 : -1, yy0 - 1);

    cu_get_neighbor_temporal(h, &neighbors[BLK_COL], xx0, yy0);
}


/* ---------------------------------------------------------------------------
 * return: number of reference frames
 */
static ALWAYS_INLINE
int cu_get_mvs_for_mc(xavs2_t *h, cu_t *p_cu, int pu_idx,
                      mv_t *p_mv_1st, mv_t *p_mv_2nd,
                      int *p_ref_idx1, int *p_ref_idx2)
{
    int num_ref;            // number of reference frames
    int dmh_mode = p_cu->cu_info.dmh_mode;
    int ref_1st = p_cu->cu_info.ref_idx_1st[pu_idx]; // 第一（前向或者B帧单向预测）运动矢量
    int ref_2nd = p_cu->cu_info.ref_idx_2nd[pu_idx]; // 第二（B帧双向的后向）
    mv_t mv_1st, mv_2nd;    // 第一（前向或者B帧单向预测）和第二（后向）运动矢量

    if (h->i_type != SLICE_TYPE_B) {
        num_ref = (ref_1st != INVALID_REF) + (ref_2nd != INVALID_REF);
        mv_1st  = p_cu->mc.mv[pu_idx][0];
        mv_2nd  = p_cu->mc.mv[pu_idx][1];
        if (dmh_mode > 0) {
            num_ref = 2;
            ref_2nd = ref_1st;
            mv_2nd  = mv_1st;
            mv_1st.x -= tab_dmh_pos[dmh_mode][0];
            mv_1st.y -= tab_dmh_pos[dmh_mode][1];
            mv_2nd.x += tab_dmh_pos[dmh_mode][0];
            mv_2nd.y += tab_dmh_pos[dmh_mode][1];
        }
    } else {
        num_ref = (ref_1st != INVALID_REF) + (ref_2nd != INVALID_REF);
        if (ref_1st == INVALID_REF) {
            ref_1st = B_BWD;
            ref_2nd = INVALID_REF;
            mv_1st  = p_cu->mc.mv[pu_idx][1];
            mv_2nd.v = 0;
        } else {
            mv_1st  = p_cu->mc.mv[pu_idx][0];
            mv_2nd  = p_cu->mc.mv[pu_idx][1];
            ref_1st = B_FWD;
            ref_2nd = B_BWD;
        }
    }

    *p_mv_1st   = mv_1st;
    *p_mv_2nd   = mv_2nd;
    *p_ref_idx1 = ref_1st;
    *p_ref_idx2 = ref_2nd;

    return num_ref;
}

/* ---------------------------------------------------------------------------
 * forward quantization
 */
static INLINE
int tu_quant_forward(xavs2_t *h, aec_t *p_aec, cu_t *p_cu, coeff_t *p_coeff, int i_level, int bsx, int bsy, int qp, int b_intra, int b_luma, int intra_mode)
{
    /* ((1 << x) * 5) / 31 */
    static const int tab_quant_fwd_add[] = {
        0,      0,      0,      1,      2,      5,     10,     20,
        41,     82,    165,    330,    660,   1321,   2642,   5285,
        10570,  21140,  42281,  84562, 169125, 338250, 676500, 1353001,
    };
    const int shift = 15 + LIMIT_BIT - (h->param->sample_bit_depth + 1) - i_level;
    const int add = tab_quant_fwd_add[shift + b_intra];

    if (h->lcu.b_enable_rdoq) {
        if ((IS_ALG_ENABLE(OPT_CODE_OPTIMZATION) && b_luma) || (IS_ALG_ENABLE(OPT_RDOQ_AZPC))) {
            const int i_coef = bsx * bsy;
            const int th_RDOQ = (int)(((1 << shift) - add) / (double)(tab_Q_TAB[qp])); //ljr
            int i;

            for (i = 0; i < i_coef; i++) {
                if (XAVS2_ABS(p_coeff[i]) >= th_RDOQ) {
                    break;
                }
            }

            if (i_coef == i) {
                p_coeff[0] = 0;
                return 0;
            }
        }

        return rdoq_block(h, p_aec, p_cu, p_coeff, bsx, bsy, i_level, qp, b_luma, intra_mode);
    } else {
#if !ENABLE_WQUANT
        return g_funcs.dctf.quant(p_coeff, bsx * bsy, tab_Q_TAB[qp], shift, add);
#else
        if (!h->WeightQuantEnable) {
            return g_funcs.dctf.quant(p_coeff, bsx * bsy, tab_Q_TAB[qp], shift, add);
        } else {
            int *levelscale = h->wq_data.levelScale[i_level - B4X4_IN_BIT][b_intra];

            return g_funcs.dctf.wquant(p_coeff, bsx * bsy, tab_Q_TAB[qp], shift, add, levelscale);
        }
#endif
    }
}

/* ---------------------------------------------------------------------------
 * inverse quantization
 */
static INLINE
void tu_quant_inverse(xavs2_t *h, cu_t *p_cu, coeff_t *coef, int num_coeff, int i_level, int qp, int b_luma)
{
    const int scale = tab_IQ_TAB[qp];
    const int shift = tab_IQ_SHIFT[qp] + (h->param->sample_bit_depth + 1) + i_level - LIMIT_BIT;

#if !ENABLE_WQUANT
    UNUSED_PARAMETER(h);
    UNUSED_PARAMETER(b_luma);
    UNUSED_PARAMETER(p_cu);
    g_funcs.dctf.dequant(coef, num_coeff, scale, shift);
#else
    if (!h->WeightQuantEnable) {
        g_funcs.dctf.dequant(coef, num_coeff, scale, shift);
    } else {
        int b_hor = b_luma && p_cu->cu_info.i_tu_split == TU_SPLIT_HOR;
        int b_ver = b_luma && p_cu->cu_info.i_tu_split == TU_SPLIT_VER;
        const int16_t(*AVS_SCAN)[2] = NULL;
        int wqm_size_id = 0;
        int wqm_stride = 0;
        int wqm_shift = h->param->PicWQDataIndex == 1 ? 3 : 0;
        int xy_shift = 0;
        int16_t *wq_matrix;

        // adaptive frequency weighting quantization
        if ((h->param->enable_sdip || h->param->enable_nsqt) && (b_hor || b_ver)) {
            xy_shift = XAVS2_MIN(2, i_level - B4X4_IN_BIT);
            wqm_size_id = xy_shift + 1;

            if (b_hor) {
                AVS_SCAN = tab_coef_scan_list_hor[XAVS2_MIN(2, i_level - 2)];
            } else {
                AVS_SCAN = tab_coef_scan_list_ver[XAVS2_MIN(2, i_level - 2)];
            }
        } else {
            xy_shift = XAVS2_MIN(3, i_level - B4X4_IN_BIT);
            wqm_size_id = xy_shift + 1;

            AVS_SCAN = tab_coef_scan_list[XAVS2_MIN(3, i_level - 2)];
        }

        wqm_stride = 1 << (wqm_size_id + B4X4_IN_BIT);
        if (wqm_size_id == 2) {
            wqm_stride >>= 1;
        } else if (wqm_size_id == 3) {
            wqm_stride >>= 2;
        }
        wq_matrix = h->wq_data.cur_wq_matrix[wqm_size_id];

        dequant_weighted_c(coef, num_coeff, scale, shift, wqm_shift, wqm_stride, xy_shift, wq_matrix, AVS_SCAN);
    }
#endif
}

/* ---------------------------------------------------------------------------
 */
static INLINE
void tu_get_dct_coeff(xavs2_t *h, coeff_t *cur_blk, int pu_size_idx, int bsx, int bsy)
{
    if (IS_ALG_ENABLE(OPT_BIT_EST_PSZT) && !h->lcu.b_2nd_rdcost_pass && bsx >= 32 && bsy >= 32) {
        g_funcs.dctf.dct_half[pu_size_idx](cur_blk, cur_blk, bsx);
    } else {
        g_funcs.dctf.dct[pu_size_idx](cur_blk, cur_blk, bsx);
    }
}



/**
 * ===========================================================================
 * local function defines (chroma)
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * finish transform, quantization, inverse-transform, inverse-quantization
 * and reconstruction pixel generation of chroma block
 */
static int cu_recon_chroma(xavs2_t *h, aec_t *p_aec, cu_t *p_cu, dist_t *distortion)
{
    int b_intra = IS_INTRA_MODE(p_cu->cu_info.i_mode);
    int pix_x_c = p_cu->i_pos_x >> 1;
    int pix_y_c = p_cu->i_pos_y >> CHROMA_V_SHIFT;
    int level_c = p_cu->cu_info.i_level - CHROMA_V_SHIFT;
    int bsize_c = 1 << level_c;
    int partidx_c = PART_INDEX(bsize_c, bsize_c);
    int cbp_c = 0;     // Coding Block Pattern (CBP) of chroma blocks
    int num_nonzero;   // number of non-zero coefficients
    int qp_c;
    int uv;
    cu_parallel_t *p_enc = cu_get_enc_context(h, p_cu->cu_info.i_level);
    coeff_t *cur_blk = p_enc->coeff_blk;
    pel_t *p_pred;

    /* prediction buffer of chroma blocks */
    if (b_intra) {
        p_pred = p_enc->intra_pred_c[p_cu->cu_info.i_intra_mode_c];
    } else {
        p_pred = p_enc->buf_pred_inter_c;
    }

    for (uv = 0; uv < 2; uv++) {
        pel_t *p_fdec = p_cu->cu_info.p_rec[uv + 1];
        pel_t *p_fenc = h->lcu.p_fenc[uv + 1] + pix_y_c * FENC_STRIDE + pix_x_c;

        g_funcs.pixf.sub_ps[partidx_c](cur_blk, bsize_c, p_fenc, p_pred, FENC_STRIDE, FREC_CSTRIDE);

        // DCT, quantization, inverse quantization, IDCT, and reconstruction
        tu_get_dct_coeff(h, cur_blk, partidx_c, bsize_c, bsize_c);

        qp_c = cu_get_qp(h, &p_cu->cu_info);
#if ENABLE_WQUANT
        qp_c += (uv == 0 ? h->param->chroma_quant_param_delta_u : h->param->chroma_quant_param_delta_v);
#endif

        qp_c = cu_get_chroma_qp(h, qp_c, uv);

        num_nonzero = tu_quant_forward(h, p_aec, p_cu, cur_blk, level_c, bsize_c, bsize_c, qp_c, b_intra, 0, DC_PRED);
        cbp_c |= (num_nonzero != 0) << (4 + uv);

        if (num_nonzero) {
            g_funcs.pixf.copy_ss[partidx_c](p_cu->cu_info.p_coeff[uv + 1], bsize_c, cur_blk, bsize_c);

            tu_quant_inverse(h, p_cu, cur_blk, bsize_c * bsize_c, level_c, qp_c, 0);
            g_funcs.dctf.idct[partidx_c](cur_blk, cur_blk, bsize_c);

            g_funcs.pixf.add_ps[partidx_c](p_fdec, FREC_CSTRIDE / 2, p_pred, cur_blk, FREC_CSTRIDE, bsize_c);
        } else {
            g_funcs.pixf.copy_pp[partidx_c](p_fdec, FREC_CSTRIDE / 2, p_pred, FREC_CSTRIDE);
        }

        *distortion += g_funcs.pixf.ssd[partidx_c](p_fenc, FENC_STRIDE, p_fdec, FREC_CSTRIDE / 2);

        p_pred += (FREC_CSTRIDE >> 1);  // uvoffset
    }

    return cbp_c;
}

/* ---------------------------------------------------------------------------
 * get max available bits for left residual coding
 */
static ALWAYS_INLINE
int rdo_get_left_bits(xavs2_t *h, rdcost_t min_rdcost, dist_t distortion)
{
    rdcost_t f_lambda = h->f_lambda_mode;
    double f_left_bits = ((min_rdcost - distortion) * h->f_lambda_1th) + 1;
    int left_bits;

    left_bits = (int)XAVS2_CLIP3F(0.0f, 32766.0f, f_left_bits);    // clip到一个合理的区间内
    if (left_bits * f_lambda + distortion <= min_rdcost) {
        left_bits++;    // 避免浮点数运算误差，保证比特数达到该值时rdcost大于min_rdcost
    }

    return left_bits;
}


/**
 * ===========================================================================
 * local function defines (intra)
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * finish transform, quantization, inverse-transform, inverse-quantization
 * and reconstruction pixel generation of a intra luma block
 */
static INLINE
int cu_recon_intra_luma(xavs2_t *h, aec_t *p_aec, cu_t *p_cu, pel_t *p_pred, int bsx, int bsy,
                        int block_x, int block_y, int idx_tu, int intra_pred_mode, dist_t *distortion)
{
    int used_wavelet = (p_cu->cu_info.i_level == B64X64_IN_BIT && p_cu->cu_info.i_tu_split != TU_SPLIT_CROSS);
    int i_tu_level = p_cu->cu_info.i_level - (p_cu->cu_info.i_tu_split != TU_SPLIT_NON);
    int pos_x = p_cu->i_pos_x + block_x;
    int pos_y = p_cu->i_pos_y + block_y;
    int part_idx = PART_INDEX(bsx, bsy);
    int w_tr = bsx >> used_wavelet;
    int h_tr = bsy >> used_wavelet;
    int num_non_zero;
    int b_2nd_trans = h->param->enable_secT;
    cu_parallel_t *p_enc  = cu_get_enc_context(h, p_cu->cu_info.i_level);
    pel_t      *p_fenc    = h->lcu.p_fenc[0] + pos_y * FENC_STRIDE + pos_x;
    pel_t      *p_fdec    = p_cu->cu_info.p_rec[0] + block_y * FREC_STRIDE + block_x;
    coeff_t    *cur_blk   = p_enc->coeff_blk;
    coeff_t    *p_coeff_y = p_cu->cu_info.p_coeff[0] + (idx_tu << ((p_cu->cu_info.i_level - 1) << 1));
    int b_top  = IS_NEIGHBOR_AVAIL(p_cu->block_avail, MD_I_TOP);
    int b_left = IS_NEIGHBOR_AVAIL(p_cu->block_avail, MD_I_LEFT);

    // get prediction and prediction error
    g_funcs.pixf.sub_ps[PART_INDEX(bsx, bsy)](cur_blk, bsx, p_fenc, p_pred, FENC_STRIDE, bsx);

    // block transform
    if (part_idx == LUMA_4x4) {
        if (b_2nd_trans) {
            g_funcs.dctf.transform_4x4_2nd(cur_blk, w_tr);
        } else {
            g_funcs.dctf.dct[LUMA_4x4](cur_blk, cur_blk, 4);     /* 4x4 dct */
        }
    } else {
        tu_get_dct_coeff(h, cur_blk, part_idx, w_tr, h_tr);

        if (b_2nd_trans) {
            g_funcs.dctf.transform_2nd(cur_blk, w_tr, intra_pred_mode, b_top, b_left);
        }
    }

    // quantization
    num_non_zero = tu_quant_forward(h, p_aec, p_cu, cur_blk, i_tu_level, w_tr, h_tr, cu_get_qp(h, &p_cu->cu_info), 1, 1, intra_pred_mode);

    if (num_non_zero) {
        g_funcs.pixf.copy_ss[PART_INDEX(w_tr, h_tr)](p_coeff_y, w_tr, cur_blk, w_tr);

        // inverse quantization
        tu_quant_inverse(h, p_cu, cur_blk, w_tr * h_tr, i_tu_level, cu_get_qp(h, &p_cu->cu_info), 1);

        // inverse transform
        if (part_idx == LUMA_4x4) {
            if (b_2nd_trans) {
                g_funcs.dctf.inv_transform_4x4_2nd(cur_blk, w_tr);
            } else {
                g_funcs.dctf.idct[LUMA_4x4](cur_blk, cur_blk, 4);    /* 4x4 idct */
            }
        } else {
            if (b_2nd_trans) {
                g_funcs.dctf.inv_transform_2nd(cur_blk, w_tr, intra_pred_mode, b_top, b_left);
            }

            g_funcs.dctf.idct[part_idx](cur_blk, cur_blk, w_tr);
        }

        g_funcs.pixf.add_ps[part_idx](p_fdec, FREC_STRIDE, p_pred, cur_blk, bsx, bsx);
    } else {
        g_funcs.pixf.copy_pp[part_idx](p_fdec, FREC_STRIDE, p_pred, bsx);
    }

    // get distortion (SSD) of current block
    *distortion = g_funcs.pixf.ssd[part_idx](p_fenc, FENC_STRIDE, p_fdec, FREC_STRIDE);

    return num_non_zero;
}

/* ---------------------------------------------------------------------------
 * get the MPMs of an intra block at (pos_x, pos_y) as 4x4 address
 */
static ALWAYS_INLINE
void xavs2_get_mpms(xavs2_t *h, cu_t *p_cu, int blockidx, int pos_y, int pos_x, int mpm[2])
{
    int ip_stride = h->i_width_in_minpu + 16;
    int top_mode  = h->ipredmode[((pos_y >> MIN_PU_SIZE_IN_BIT) - 1) * ip_stride + pos_x];
    int left_mode = h->ipredmode[(pos_y >> MIN_PU_SIZE_IN_BIT) * ip_stride + pos_x - 1];

    if (blockidx != 0) {
        if (p_cu->cu_info.i_mode == PRED_I_2Nxn) {
            top_mode = p_cu->cu_info.real_intra_modes[blockidx - 1];
        } else if (p_cu->cu_info.i_mode == PRED_I_nx2N) {
            left_mode = p_cu->cu_info.real_intra_modes[blockidx - 1];
        } else if (p_cu->cu_info.i_mode == PRED_I_NxN) {
            switch (blockidx) {
            case 1:
                left_mode = p_cu->cu_info.real_intra_modes[0];
                break;
            case 2:
                top_mode  = p_cu->cu_info.real_intra_modes[0];
                break;
            case 3:
                top_mode  = p_cu->cu_info.real_intra_modes[1];
                left_mode = p_cu->cu_info.real_intra_modes[2];
                break;
            default: // case 0:
                break;
            }
        }
    }

    top_mode  = (top_mode  < 0) ? DC_PRED : top_mode;
    left_mode = (left_mode < 0) ? DC_PRED : left_mode;
    mpm[0] = XAVS2_MIN(top_mode, left_mode);
    mpm[1] = XAVS2_MAX(top_mode, left_mode);

    if (mpm[0] == mpm[1]) {
        mpm[0] = DC_PRED;
        mpm[1] = (mpm[1] == DC_PRED) ? BI_PRED : mpm[1];
    }
}


/* ---------------------------------------------------------------------------
 * 检查帧内PU划分方式的RDCost并更新最优的PU划分方式
 */
static void cu_check_intra(xavs2_t *h, aec_t *p_aec, cu_t *p_cu, cu_info_t *best, int mode, rdcost_t *min_rdcost)
{
    int level = p_cu->cu_info.i_level;
    cu_layer_t    *p_layer = cu_get_layer(h, level);
    cu_parallel_t *p_enc   = cu_get_enc_context(h, level);
    rdcost_t rdcost_luma = 0;
    rdcost_t rdcost = MAX_COST;
    rdcost_t min_mode_rdcost = MAX_COST;
    pel_t *rec_bak_y = best->p_rec[0];
    pel_t *p_best_part[4];
    int blockidx;
    int num_luma_block = mode != PRED_I_2Nx2N ? 4 : 1;
    int b_need_swap_buf = 0;
    int pix_x_c = p_cu->i_pos_x >> 1;
    int pix_y_c = p_cu->i_pos_y >> CHROMA_V_SHIFT;
    intra_candidate_t *p_candidates = p_layer->intra_candidates;

    /* 确定PU划分类型 */
    cu_init_pu_intra(h, &p_cu->cu_info, level, mode);

    /* 确定TU划分类型 */
    cu_set_tu_split_type(h, &p_cu->cu_info, mode != PRED_I_2Nx2N);

    h->copy_aec_state_rdo(&p_layer->cs_rdo, p_aec);
    p_cu->cu_info.i_cbp = 0;

    p_cu->intra_avail = (uint8_t)xavs2_intra_get_cu_neighbors(h, p_cu, p_cu->i_pix_x, p_cu->i_pix_y, p_cu->i_size);

    /* 1, intra luma prediction and mode decision */
    for (blockidx = 0; blockidx < num_luma_block; blockidx++) {
        int mpm[2];  // most probable modes (MPMs) for current luma block
        int block_x = p_cu->cu_info.cb[blockidx].x;
        int block_y = p_cu->cu_info.cb[blockidx].y;
        int block_w = p_cu->cu_info.cb[blockidx].w;
        int block_h = p_cu->cu_info.cb[blockidx].h;
        int pos_x = p_cu->i_pos_x + block_x;
        int pos_y = p_cu->i_pos_y + block_y;
        int b4x4_x = (p_cu->i_pix_x + block_x) >> MIN_PU_SIZE_IN_BIT;
        dist_t best_dist = MAX_DISTORTION;
        int best_rate = INT_MAX;
        int best_mode = 0;
        int best_pmode = 0;
        int best_cbp = 0;
        pel_t *p_fenc = h->lcu.p_fenc[0] + pos_y * FENC_STRIDE + pos_x;
        rdcost_t best_rdcost = MAX_COST;
        int i;
        int num_for_rdo;
        p_candidates = p_layer->intra_candidates;   // candidate list, reserving the cost

        /* init */
        xavs2_get_mpms(h, p_cu, blockidx, pos_y, b4x4_x, mpm);

        for (i = 0; i < INTRA_MODE_NUM_FOR_RDO; i++) {
            p_candidates[i].mode = 0;
            p_candidates[i].cost = MAX_COST;
        }

        /* conduct prediction and get intra prediction direction candidates for RDO */
        num_for_rdo = h->lcu.get_intra_dir_for_rdo_luma(h, p_cu, p_candidates, p_fenc, mpm, blockidx,
                      block_x, block_y, block_w, block_h);

        // store the coding state
        h->copy_aec_state_rdo(&p_enc->cs_pu_init, p_aec);

        /* RDO */
        for (i = 0; i < num_for_rdo; i++) {
            //rdcost_t rdcost;
            dist_t dist_curr;     // 当前亮度帧内块的失真
            int rate_curr = 0; // 当前亮度帧内块的码率（比特数）
            int Mode = p_candidates[i].mode;
            pel_t *p_pred = p_enc->intra_pred[Mode];

            // get and check rate_chroma-distortion cost
            int mode_idx_aec = (mpm[0] == Mode) ? -2 : ((mpm[1] == Mode) ? -1 : (mpm[0] > Mode ? Mode : (mpm[1] > Mode ? Mode - 1 : Mode - 2)));
            int num_nonzero;

            num_nonzero = cu_recon_intra_luma(h, p_aec, p_cu, p_pred,
                                              block_w, block_h, block_x, block_y,
                                              blockidx, Mode, &dist_curr);
            num_nonzero = !!num_nonzero;
            {
                int used_wavelet = (p_cu->cu_info.i_level == B64X64_IN_BIT && p_cu->cu_info.i_tu_split != TU_SPLIT_CROSS);
                int w_tr = block_w >> used_wavelet;
                int i_tu_level = p_cu->cu_info.i_level - (p_cu->cu_info.i_tu_split != TU_SPLIT_NON) - used_wavelet;
                int rate_luma_mode;
                coeff_t *p_coeff_y = p_cu->cu_info.p_coeff[0] + (blockidx << ((p_cu->cu_info.i_level - 1) << 1));

                // get rate for intra prediction mode
                rate_luma_mode = p_aec->binary.write_intra_pred_mode(p_aec, mode_idx_aec);

                // get rate for luminance coefficients
                if (num_nonzero) {
                    int bits_left = rdo_get_left_bits(h, best_rdcost, dist_curr) - rate_luma_mode;
                    rate_curr = p_aec->binary.est_luma_block_coeff(h, p_aec, p_cu, p_coeff_y, &p_enc->runlevel, i_tu_level, xavs2_log2u(w_tr),
                                1, Mode, bits_left);
                    rate_luma_mode += rate_curr;
                }

                // calculate RD-cost and return it
                rdcost = dist_curr + h->f_lambda_mode * rate_luma_mode;
            }

            // choose best mode
            if (rdcost < best_rdcost) {
                XAVS2_SWAP_PTR(p_cu->cu_info.p_rec[0], p_layer->p_rec_tmp[0]);
                XAVS2_SWAP_PTR(p_cu->cu_info.p_coeff[0], p_layer->p_coeff_tmp[0]);

                // set best mode update minimum cost
                best_dist = dist_curr;
                best_rate = rate_curr;
                best_rdcost = rdcost;
                best_mode = Mode;
                best_pmode = mode_idx_aec;
                best_cbp = num_nonzero;   // flag if dct-coefficients must be coded
                h->copy_aec_state_rdo(&p_enc->cs_tu, p_aec);
            }

            h->copy_aec_state_rdo(p_aec, &p_enc->cs_pu_init);

            if (IS_ALG_ENABLE(OPT_ET_RDO_INTRA_L)) {
                if (rdcost > best_rdcost * 1.2) {
                    break;
                }
            }
        }   // for (i = 0; i < num_for_rdo; i++)

        /* change the coding state to BEST */
        if (best_rate < INT_MAX) {
            if (p_cu->cu_info.i_mode != PRED_I_2Nx2N) {
                g_funcs.pixf.copy_pp[PART_INDEX(block_w, block_h)](h->lcu.p_fdec[0] + pos_y * FDEC_STRIDE + pos_x, FDEC_STRIDE,
                        p_layer->p_rec_tmp[0] + block_y * FREC_STRIDE + block_x, FREC_STRIDE);
            }

            /* copy coefficients and reconstructed data for best mode */
            XAVS2_SWAP_PTR(p_cu->cu_info.p_rec[0], p_layer->p_rec_tmp[0]);
            XAVS2_SWAP_PTR(p_cu->cu_info.p_coeff[0], p_layer->p_coeff_tmp[0]);
            p_best_part[blockidx] = p_cu->cu_info.p_rec[0];

            /* set intra mode prediction */
            p_cu->cu_info.pred_intra_modes[blockidx] = (int8_t)best_pmode;
            p_cu->cu_info.real_intra_modes[blockidx] = (int8_t)best_mode;

            /* copy coding state */
            h->copy_aec_state_rdo(p_aec, &p_enc->cs_tu);
        }

        /* 保存最优模式的状态：失真、亮度分量比特数（排除掉亮度预测模式），CBP */
        rdcost_luma += best_dist + h->f_lambda_mode * best_rate;
        p_cu->cu_info.i_cbp |= (best_cbp) << blockidx;

        /* 亮度块RDO的提前终止 */
        if (rdcost_luma >= *min_rdcost) {
            p_layer->mode_rdcost[mode] = MAX_COST;        /* set the cost for SDIP fast algorithm */
            h->copy_aec_state_rdo(p_aec, &p_layer->cs_rdo);
            return;  // 亮度块的最优rdcost已经超过当前最优值，停止后续色度块的模式遍历
        }
    }
    p_cu->feature.rdcost_luma = rdcost_luma;

    /* 2, store best luma reconstruction pixels */
    for (blockidx = 0; blockidx < num_luma_block; blockidx++) {
        if (p_best_part[blockidx] != p_cu->cu_info.p_rec[0]) {
            int offset = p_cu->cu_info.cb[blockidx].y * FREC_STRIDE + p_cu->cu_info.cb[blockidx].x;
            int offset_coeff = blockidx << ((p_cu->cu_info.i_level - 1) << 1);
            int w_tr = p_cu->cu_info.cb[0].w;
            int h_tr = p_cu->cu_info.cb[0].h;
            int part_idx = PART_INDEX(w_tr, h_tr);
            g_funcs.pixf.copy_pp[part_idx](p_cu->cu_info.p_rec[0]   + offset, FREC_STRIDE, p_layer->p_rec_tmp[0]  + offset, p_cu->i_size);
            g_funcs.pixf.copy_ss[part_idx](p_cu->cu_info.p_coeff[0] + offset_coeff, w_tr, p_layer->p_coeff_tmp[0] + offset_coeff, w_tr);
        }
    }

    /* 3, Chroma mode decision and CU mode updating */
    if (h->param->chroma_format != CHROMA_400) {
        int lmode;
        int num_rdo_chroma_mode;
        int idx_chroma_mode;
        int tmp_cbp_luma = p_cu->cu_info.i_cbp;

        lmode = tab_intra_mode_luma2chroma[p_cu->cu_info.real_intra_modes[0]];
        num_rdo_chroma_mode = h->get_intra_candidates_chroma(h, p_cu, level - 1, pix_y_c, pix_x_c, p_candidates);

        for (idx_chroma_mode = 0; idx_chroma_mode < num_rdo_chroma_mode; idx_chroma_mode++) {
            dist_t dist_chroma = 0;  // 色度块的指针
            int rate_chroma = 0;
            int bits_left;
            int predmode_c = p_candidates[idx_chroma_mode].mode;
            int cbp_c;

            /* 跳过色度分量第二次调用过程中的模式选择，直接选到最优模式完成RDOQ */
            if ((h->param->i_rdoq_level == RDOQ_CU_LEVEL && h->lcu.b_enable_rdoq) && predmode_c != best->i_intra_mode_c) {
                continue;
            }
            if (predmode_c != DM_PRED_C && predmode_c == lmode) {
                continue;
            }
            p_cu->cu_info.i_intra_mode_c = (int8_t)predmode_c;

            /* 完成RDO过程的色度块的重构过程（变换、量化、反变换反量化及求重构值） */
            cbp_c = cu_recon_chroma(h, p_aec, p_cu, &dist_chroma);

            p_cu->cu_info.i_cbp = (int8_t)(tmp_cbp_luma + cbp_c);

            /* ------- GET RATE -------- */
            rate_chroma = p_aec->binary.est_cu_header(h, p_aec, p_cu);
#if ENABLE_RATE_CONTROL_CU
            rate_chroma += p_aec->binary.write_cu_cbp_dqp(h, p_aec, &p_cu->cu_info, h->i_slice_index, h->last_dquant);
#else
            rate_chroma += p_aec->binary.write_cu_cbp(p_aec, &p_cu->cu_info, h->i_slice_index, h);
#endif

            bits_left = rdo_get_left_bits(h, *min_rdcost - rdcost_luma, dist_chroma);

            if (p_cu->cu_info.i_cbp & (1 << 4)) {
                int cur_bits_left = bits_left - rate_chroma;
                rate_chroma += p_aec->binary.est_chroma_block_coeff(h, p_aec, p_cu, p_cu->cu_info.p_coeff[1], &p_enc->runlevel, level - 1, cur_bits_left);
            }
            if (p_cu->cu_info.i_cbp & (1 << 5)) {
                int cur_bits_left = bits_left - rate_chroma;
                rate_chroma += p_aec->binary.est_chroma_block_coeff(h, p_aec, p_cu, p_cu->cu_info.p_coeff[2], &p_enc->runlevel, level - 1, cur_bits_left);
            }

            rdcost = dist_chroma + h->f_lambda_mode * rate_chroma + rdcost_luma;

            min_mode_rdcost = XAVS2_MIN(rdcost, min_mode_rdcost);

            if (rdcost < *min_rdcost) {
                *min_rdcost = rdcost;
                h->copy_aec_state_rdo(&p_layer->cs_cu, p_aec);    /* store coding state for the best mode */
                cu_store_parameters(h, p_cu, best);
                b_need_swap_buf = 1;
            }

            h->copy_aec_state_rdo(p_aec, &p_enc->cs_tu);   /* revert to AEC context of best Luma mode */

            if (IS_ALG_ENABLE(OPT_FAST_RDO_INTRA_C)) {
                if (rdcost > *min_rdcost * 2 ||
                    cbp_c == 0) {
                    break;
                }
            }
        }
    } else {   /* YUV400 */
        /* ------- GET RATE -------- */
        int rate_hdr = p_aec->binary.est_cu_header(h, p_aec, p_cu);
#if ENABLE_RATE_CONTROL_CU
        rate_hdr += p_aec->binary.write_cu_cbp_dqp(h, p_aec, &p_cu->cu_info, h->i_slice_index, h->last_dquant);
#else
        rate_hdr += p_aec->binary.write_cu_cbp(p_aec, &p_cu->cu_info, h->i_slice_index, h);
#endif
        rdcost = h->f_lambda_mode * rate_hdr + rdcost_luma;

        if (rdcost < *min_rdcost) {
            *min_rdcost = rdcost;
            h->copy_aec_state_rdo(&p_layer->cs_cu, p_aec);    /* store coding state for the best mode */
            cu_store_parameters(h, p_cu, best);
            b_need_swap_buf = 1;
        }
    }

    h->copy_aec_state_rdo(p_aec, &p_layer->cs_rdo);  /* revert to initial AEC context */

    /* 4, confirm the buffer pointers and record the best information */
    if (best->p_rec[0] == rec_bak_y && b_need_swap_buf) {
        XAVS2_SWAP_PTR(best->p_rec[0],   p_cu->cu_info.p_rec[0]);
        XAVS2_SWAP_PTR(best->p_coeff[0], p_cu->cu_info.p_coeff[0]);
    }

    p_layer->mode_rdcost[mode] = min_mode_rdcost;    /* store the cost for SDIP fast algorithm */
}

//#if OPT_BYPASS_SDIP
/* ---------------------------------------------------------------------------
 * SDIP fast
 */
static ALWAYS_INLINE int sdip_early_bypass(xavs2_t *h, cu_layer_t *p_layer, int i_mode)
{
    UNUSED_PARAMETER(h);
    return i_mode == PRED_I_nx2N && (p_layer->mode_rdcost[PRED_I_2Nxn] < p_layer->mode_rdcost[PRED_I_2Nx2N] * 0.9);
}
//#endif

/**
 * ===========================================================================
 * local function defines (inter)
 * ===========================================================================
 */

//#if OPT_FAST_ZBLOCK || OPT_ECU
static const int tab_th_zero_block_sad[][5] = {
    {    7,   19,   72,  281,  1115 }, {    7,   19,   73,  281,  1116 }, {    7,   20,   73,  282,  1118 },
    {    8,   20,   74,  283,  1120 }, {    8,   20,   74,  284,  1122 }, {    8,   20,   75,  285,  1124 },
    {    8,   21,   75,  286,  1126 }, {    8,   21,   76,  288,  1129 }, {    9,   21,   77,  289,  1132 },
    {    9,   22,   77,  291,  1135 }, {    9,   22,   78,  292,  1138 }, {   10,   23,   79,  294,  1142 },
    {   10,   23,   80,  296,  1146 }, {   10,   24,   81,  298,  1150 }, {   11,   24,   82,  301,  1155 },
    {   11,   25,   84,  303,  1160 }, {   12,   26,   85,  306,  1166 }, {   12,   26,   87,  309,  1172 },
    {   13,   27,   88,  312,  1179 }, {   13,   28,   90,  316,  1186 }, {   14,   29,   92,  320,  1194 },
    {   15,   30,   94,  325,  1203 }, {   15,   31,   97,  329,  1213 }, {   16,   33,   99,  334,  1223 },
    {   17,   34,  102,  340,  1235 }, {   18,   36,  105,  346,  1247 }, {   20,   37,  109,  353,  1260 },
    {   21,   39,  112,  360,  1275 }, {   22,   41,  116,  368,  1292 }, {   24,   43,  121,  377,  1309 },
    {   25,   46,  125,  386,  1328 }, {   27,   48,  131,  397,  1349 }, {   29,   51,  136,  408,  1372 },
    {   31,   54,  142,  420,  1397 }, {   33,   58,  149,  434,  1424 }, {   36,   61,  156,  448,  1453 },
    {   38,   65,  164,  464,  1485 }, {   41,   70,  173,  482,  1520 }, {   45,   74,  183,  501,  1559 },
    {   48,   79,  193,  521,  1600 }, {   52,   85,  204,  544,  1646 }, {   56,   91,  217,  569,  1696 },
    {   61,   98,  230,  596,  1750 }, {   66,  105,  245,  625,  1809 }, {   71,  113,  261,  657,  1873 },
    {   77,  122,  278,  692,  1944 }, {   83,  132,  297,  729,  2020 }, {   90,  142,  318,  771,  2104 },
    {   98,  153,  341,  816,  2195 }, {  106,  166,  365,  865,  2294 }, {  116,  179,  392,  919,  2403 },
    {  126,  194,  422,  978,  2521 }, {  136,  210,  454, 1042,  2649 }, {  148,  227,  488, 1111,  2790 },
    {  161,  246,  526, 1187,  2943 }, {  175,  267,  568, 1270,  3110 }, {  191,  290,  613, 1360,  3292 },
    {  207,  314,  662, 1459,  3491 }, {  225,  341,  716, 1566,  3707 }, {  245,  370,  775, 1683,  3944 },
    {  267,  402,  839, 1811,  4201 }, {  291,  437,  909, 1950,  4482 }, {  316,  475,  985, 2102,  4788 },
    {  345,  517, 1068, 2268,  5123 }, {  375,  562, 1158, 2448,  5487 }, {  412,  617, 1268, 2667,  5928 },
    {  445,  665, 1364, 2860,  6317 }, {  485,  724, 1482, 3094,  6790 }, {  528,  788, 1610, 3350,  7305 },
    {  576,  858, 1749, 3628,  7867 }, {  631,  939, 1912, 3954,  8524 }, {  687, 1022, 2078, 4285,  9192 },
    {  748, 1113, 2259, 4647,  9920 }, {  812, 1206, 2446, 5019, 10671 }, {  884, 1313, 2661, 5448, 11537 },
    {  964, 1431, 2895, 5917, 12482 }, { 1047, 1553, 3140, 6406, 13469 }, { 1145, 1698, 3430, 6985, 14636 },
    { 1248, 1850, 3735, 7592, 15862 }, { 1357, 2011, 4055, 8233, 17154 }
};

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
bool_t isZeroCuFast(xavs2_t *h, cu_t *p_cu)
{
    int i_level = p_cu->cu_info.i_level - MIN_PU_SIZE_IN_BIT;
    int i_qp = cu_get_qp(h, &p_cu->cu_info);
    int thres_satd = (int)(tab_th_zero_block_sad[i_qp][i_level] * h->param->factor_zero_block);

    return p_cu->sum_satd < thres_satd;
}
//#endif

/* ---------------------------------------------------------------------------
 * int scrFlag = 0;             // 0=noSCR, 1=strongSCR, 2=jmSCR
 */
static INLINE int
tu_recon_inter_luma(xavs2_t *h, aec_t *p_aec, cu_t *p_cu,
                    int i_level, int8_t *cbp, int blockidx, coeff_t *cur_blk,
                    int x_pu, int y_pu, int w_pu, int h_pu)
{
    cu_layer_t *p_layer = cu_get_layer(h, p_cu->cu_info.i_level);
    int used_wavelet = (p_cu->cu_info.i_level == B64X64_IN_BIT && p_cu->cu_info.i_tu_split != TU_SPLIT_CROSS);
    int part_idx = PART_INDEX(w_pu, h_pu);
    int w_tr = w_pu >> used_wavelet;
    int h_tr = h_pu >> used_wavelet;
    int num_non_zero = 0;
    pel_t *p_fdec = p_cu->cu_info.p_rec[0] + y_pu * FREC_STRIDE + x_pu;
    pel_t *p_pred = p_layer->buf_pred_inter + y_pu * FREC_STRIDE + x_pu;
    coeff_t *coeff_y = p_cu->cu_info.p_coeff[0] + (blockidx << ((p_cu->cu_info.i_level - 1) << 1));

    tu_get_dct_coeff(h, cur_blk, part_idx, w_tr, h_tr);

    num_non_zero = tu_quant_forward(h, p_aec, p_cu, cur_blk, i_level, w_tr, h_tr,
                                    cu_get_qp(h, &p_cu->cu_info), 0, 1, DC_PRED);

    if (num_non_zero != 0) {
        *cbp |= (1 << blockidx);    // 指定位设置为 1
        g_funcs.pixf.copy_ss[PART_INDEX(w_tr, h_tr)](coeff_y, w_tr, cur_blk, w_tr);

        tu_quant_inverse(h, p_cu, cur_blk, w_tr * h_tr, i_level, cu_get_qp(h, &p_cu->cu_info), 1);
        g_funcs.dctf.idct[part_idx](cur_blk, cur_blk, w_tr);

        g_funcs.pixf.add_ps[part_idx](p_fdec, FREC_STRIDE, p_pred, cur_blk, FREC_STRIDE, w_pu);
    } else {
        /* 清除CBP指定位的值，这里CBP初始值为0，因而无需操作 */
        // 全零块不必做反变换反量化，只需拷贝预测值为重构值
        coeff_y[0] = 0;
        if (p_cu->cu_info.i_tu_split) {
            g_funcs.pixf.copy_pp[part_idx](p_fdec, FREC_STRIDE, p_pred, FREC_STRIDE);
        }
    }

    return num_non_zero;
}


/* ---------------------------------------------------------------------------
 * 以指定方式重构帧间预测方式的CU的亮度分量；
 * 返回当前CU地失真（加上色度块失真）
 */
static
dist_t cu_recon_inter_luma(xavs2_t *h, aec_t *p_aec, cu_t *p_cu,
                           int is_non_residual, int b_tu_split,
                           int cbp_c, dist_t dist_chroma)
{
    cu_layer_t *p_layer = cu_get_layer(h, p_cu->cu_info.i_level);
    cu_parallel_t *p_enc = cu_get_enc_context(h, p_cu->cu_info.i_level);
    coeff_t *cur_blk   = p_enc->coeff_blk;
    coeff_t *coeff_bak = p_enc->coeff_bak;
    coeff_t *p_resi;
    int level = p_cu->cu_info.i_level;
    int num_nonzero = 0;
    int sum_dc_coeff = 0;
    int b_zero_block = 0;
    int blockidx;
    int pix_x = p_cu->i_pos_x;
    int pix_y = p_cu->i_pos_y;
    int cu_size = p_cu->i_size;
    int cu_size_2 = cu_size >> 1;
    int cu_size_4 = cu_size_2 >> 1;
    dist_t distortion;
    pel_t *p_fenc;
    pel_t *p_fdec;

    /* clear CBP */
    p_cu->cu_info.i_cbp = 0;

    /* encode for luma */
    cu_set_tu_split_type(h, &p_cu->cu_info, b_tu_split);

    if (is_non_residual) {  /* SKIP mode (or no residual coding) */
        int uvoffset = (FREC_CSTRIDE >> 1);
        int part_idx_c = PART_INDEX(cu_size_2, cu_size_2);
        int pix_x_c = pix_x >> 1;
        int pix_y_c = pix_y >> CHROMA_V_SHIFT;

        h->lcu.bypass_all_dmh |= (p_cu->cu_info.dmh_mode == 0);
        /* copy Y component and get distortion */
        p_fenc = h->lcu.p_fenc[0] + pix_y * FENC_STRIDE + pix_x;
        p_fdec = p_cu->cu_info.p_rec[0];
        g_funcs.pixf.copy_pp[PART_INDEX(cu_size, cu_size)](p_fdec, FREC_STRIDE, p_layer->buf_pred_inter, FREC_STRIDE);
        distortion = g_funcs.pixf.ssd[PART_INDEX(cu_size, cu_size)](p_fenc, FENC_STRIDE, p_fdec, FREC_STRIDE);

        /* chroma distortion */
        if (cbp_c) {
            /* copy U component and get distortion */
            p_fenc = h->lcu.p_fenc[1] + pix_y_c * FENC_STRIDE + pix_x_c;
            p_fdec = p_cu->cu_info.p_rec[1];
            g_funcs.pixf.copy_pp[part_idx_c](p_fdec, FREC_CSTRIDE / 2, p_enc->buf_pred_inter_c, FREC_CSTRIDE);
            distortion += g_funcs.pixf.ssd[part_idx_c](p_fenc, FENC_STRIDE, p_fdec, FREC_CSTRIDE / 2);

            /* copy V component and get distortion */
            p_fenc = h->lcu.p_fenc[2] + pix_y_c * FENC_STRIDE + pix_x_c;
            p_fdec = p_cu->cu_info.p_rec[2];
            g_funcs.pixf.copy_pp[part_idx_c](p_fdec, FREC_CSTRIDE / 2, p_enc->buf_pred_inter_c + uvoffset, FREC_CSTRIDE);
            distortion += g_funcs.pixf.ssd[part_idx_c](p_fenc, FENC_STRIDE, p_fdec, FREC_CSTRIDE / 2);
        } else {
            distortion += dist_chroma;
        }

        return distortion;
    } else if (p_cu->cu_info.i_tu_split) {
        int pix_cu_x = 0;
        int pix_cu_y = 0;

        switch (p_cu->cu_info.i_tu_split) {
        case TU_SPLIT_HOR:
            g_funcs.pixf.copy_ss[PART_INDEX(cu_size, cu_size)](cur_blk, cu_size, coeff_bak, cu_size);
            for (blockidx = 0; blockidx < 4; blockidx++, pix_cu_y += cu_size_4) {
                p_resi = cur_blk + pix_cu_y * cu_size + pix_cu_x;
                num_nonzero += tu_recon_inter_luma(h, p_aec, p_cu, level - 1, &p_cu->cu_info.i_cbp, blockidx, p_resi, pix_cu_x, pix_cu_y, cu_size, cu_size_4);
                sum_dc_coeff += XAVS2_ABS(p_cu->cu_info.p_coeff[0][pix_cu_y * cu_size + pix_cu_x]);
            }
            break;
        case TU_SPLIT_VER:
            for (blockidx = 0; blockidx < 4; blockidx++, pix_cu_x += cu_size_4) {
                p_resi = coeff_bak + pix_cu_y * cu_size + pix_cu_x;
                g_funcs.pixf.copy_ss[PART_INDEX(cu_size_4, cu_size)](cur_blk, cu_size_4, p_resi, cu_size);
                num_nonzero += tu_recon_inter_luma(h, p_aec, p_cu, level - 1, &p_cu->cu_info.i_cbp, blockidx, cur_blk, pix_cu_x, pix_cu_y, cu_size_4, cu_size);
                sum_dc_coeff += XAVS2_ABS(p_cu->cu_info.p_coeff[0][pix_cu_y * cu_size + pix_cu_x]);
            }
            break;
        default:
            for (blockidx = 0; blockidx < 4; blockidx++) {
                pix_cu_x = (blockidx & 1) * cu_size_2;
                pix_cu_y = (blockidx >> 1) * cu_size_2;
                p_resi = coeff_bak + pix_cu_y * cu_size + pix_cu_x;
                g_funcs.pixf.copy_ss[PART_INDEX(cu_size_2, cu_size_2)](cur_blk, cu_size_2, p_resi, cu_size);
                num_nonzero += tu_recon_inter_luma(h, p_aec, p_cu, level - 1, &p_cu->cu_info.i_cbp, blockidx, cur_blk, pix_cu_x, pix_cu_y, cu_size_2, cu_size_2);
                sum_dc_coeff += XAVS2_ABS(p_cu->cu_info.p_coeff[0][pix_cu_y * cu_size + pix_cu_x]);
            }
            break;
        }

        // 当前CU非零系数不大于 LUMA_COEFF_COST 个，且DC系数并不大的情况下，可认定为全零块
        b_zero_block = (num_nonzero <= LUMA_COEFF_COST && sum_dc_coeff <= MAX_COEFF_QUASI_ZERO);
    } else {
        if (IS_ALG_ENABLE(OPT_FAST_ZBLOCK) && p_cu->is_zero_block) {
            b_zero_block = 1;
        } else {
            num_nonzero += tu_recon_inter_luma(h, p_aec, p_cu, level, &p_cu->cu_info.i_cbp, 0, coeff_bak, 0, 0, cu_size, cu_size);

            // 当前CU的所有变换块的非零系数数量，不大于 LUMA_COEFF_COST 个，且DC系数并不大的情况下，可认定为全零块
            sum_dc_coeff = XAVS2_ABS(p_cu->cu_info.p_coeff[0][0]);
            b_zero_block = (num_nonzero <= LUMA_COEFF_COST && sum_dc_coeff <= MAX_COEFF_QUASI_ZERO);
        }
    }

    if (b_zero_block) {
        h->lcu.bypass_all_dmh |= (h->i_type == SLICE_TYPE_F && p_cu->cu_info.dmh_mode == 0);
        p_cu->cu_info.i_cbp = 0;
        g_funcs.pixf.copy_pp[PART_INDEX(cu_size, cu_size)](p_cu->cu_info.p_rec[0], FREC_STRIDE,
                p_layer->buf_pred_inter, FREC_STRIDE);
    }

    /* set CBP */
    p_cu->cu_info.i_cbp += (int8_t)cbp_c;

    /* luma distortion */
    p_fenc = h->lcu.p_fenc[0] + pix_y * FENC_STRIDE + pix_x;
    p_fdec = p_cu->cu_info.p_rec[0];
    distortion = dist_chroma;
    distortion += g_funcs.pixf.ssd[PART_INDEX(cu_size, cu_size)](p_fenc, FENC_STRIDE, p_fdec, FREC_STRIDE);
    return distortion;
}

/* ---------------------------------------------------------------------------
 * R-D cost for a inter cu whether split or not
 * Return: rate-distortion cost of cu when TU is split or not
 */
static int tu_rdcost_inter(xavs2_t *h, aec_t *p_aec, cu_t *p_cu,
                           dist_t distortion, int rate_chroma, rdcost_t *rdcost)
{
    int mode = p_cu->cu_info.i_mode;
    int level = p_cu->cu_info.i_level;
    int rate;
    int block_idx;
    cu_parallel_t *p_enc = cu_get_enc_context(h, level);

    /* -------------------------------------------------------------
     * get rate */

    /* rate of cu header */
    rate = p_aec->binary.est_cu_header(h, p_aec, p_cu);

    /* rate of motion information */
    rate += p_aec->binary.est_cu_refs_mvds(h, p_aec, p_cu);

    /* tu information */
    if (mode != PRED_SKIP || p_cu->cu_info.i_cbp) {
        int bits_left = rdo_get_left_bits(h, *rdcost, distortion);
        int cur_bits_left;
        /* rate of cbp & dqp */
#if ENABLE_RATE_CONTROL_CU
        rate += p_aec->binary.write_cu_cbp_dqp(h, p_aec, &p_cu->cu_info, h->i_slice_index, h->last_dquant);
#else
        rate += p_aec->binary.write_cu_cbp(p_aec, &p_cu->cu_info, h->i_slice_index, h);
#endif

        /* rate of luma coefficients */
        if (p_cu->cu_info.i_tu_split != TU_SPLIT_NON) {
            int use_wavelet = (level == B64X64_IN_BIT && p_cu->cu_info.i_tu_split != TU_SPLIT_CROSS);
            int i_tu_level = level - 1 - use_wavelet;
            for (block_idx = 0; block_idx < 4; block_idx++) {
                if (p_cu->cu_info.i_cbp & (1 << block_idx)) {
                    cb_t tb;
                    cur_bits_left = bits_left - rate;

                    cu_init_transform_block(p_cu->cu_info.i_level, p_cu->cu_info.i_tu_split, block_idx, &tb);

                    rate += p_aec->binary.est_luma_block_coeff(h, p_aec, p_cu,
                            p_cu->cu_info.p_coeff[0] + (block_idx << ((p_cu->cu_info.i_level - 1) << 1)),
                            &p_enc->runlevel, i_tu_level, xavs2_log2u(tb.w) - use_wavelet, 0, 0, cur_bits_left);
                }
            }
        } else {
            if (p_cu->cu_info.i_cbp & 15) {
                int i_tu_level = level - (level == B64X64_IN_BIT);
                cur_bits_left = bits_left - rate;
                rate += p_aec->binary.est_luma_block_coeff(h, p_aec, p_cu, p_cu->cu_info.p_coeff[0],
                        &p_enc->runlevel, i_tu_level, i_tu_level, 0, 0, cur_bits_left);
            }
        }

        /* rate of chroma coefficients */
        if (IS_ALG_ENABLE(OPT_ADVANCE_CHROMA_AEC)) {
            if (p_cu->cu_info.i_cbp != 0) {  // not skip mode
                rate += rate_chroma;
            }
        } else {
            level--;
            if (p_cu->cu_info.i_cbp & (1 << 4)) {
                cur_bits_left = bits_left - rate;
                rate += p_aec->binary.est_chroma_block_coeff(h, p_aec, p_cu, p_cu->cu_info.p_coeff[1], &p_enc->runlevel, level, cur_bits_left);
            }
            if (p_cu->cu_info.i_cbp & (1 << 5)) {
                cur_bits_left = bits_left - rate;
                rate += p_aec->binary.est_chroma_block_coeff(h, p_aec, p_cu, p_cu->cu_info.p_coeff[2], &p_enc->runlevel, level, cur_bits_left);
            }
        }
    }

    /* -------------------------------------------------------------
     * get rate-distortion cost */
    *rdcost = distortion + h->f_lambda_mode * rate;

    return p_cu->cu_info.i_cbp;
}

/* ---------------------------------------------------------------------------
 * 获取亮度、色度分量的预测像素值，返回MV是否在有效范围内
 */
static ALWAYS_INLINE
int rdo_get_pred_inter(xavs2_t *h, cu_t *p_cu, int cal_luma_chroma)
{
    cu_parallel_t *p_enc = cu_get_enc_context(h, p_cu->cu_info.i_level);
    cu_layer_t *p_layer = cu_get_layer(h, p_cu->cu_info.i_level);
    int blockidx;

    /* get prediction data */
    for (blockidx = 0; blockidx < p_cu->cu_info.num_pu; blockidx++) {
        cb_t cur_cb = p_cu->cu_info.cb[blockidx];
        int start_x = cur_cb.x;
        int start_y = cur_cb.y;
        int width   = cur_cb.w;
        int height  = cur_cb.h;
        int pix_x   = p_cu->i_pix_x + start_x;
        int pix_y   = p_cu->i_pix_y + start_y;

        mv_t mv_1st, mv_2nd;   // 第一（前向或者B帧单向预测）和第二（后向）运动矢量
        int ref_1st, ref_2nd;  // 第一（前向或者B帧单向预测）和第二（后向）参考帧号
        int num_mvs;
        int b_mv_valid;        // MV是否有效：大小取值是否在标准规定的有效范围内
        pel_t *p_temp = p_enc->buf_pixel_temp;
        pel_t *p_pred;
        xavs2_frame_t *p_ref1 = NULL;
        xavs2_frame_t *p_ref2 = NULL;

        /* MV的数量，大于1为双参考帧/DMH的预测 */
        num_mvs = cu_get_mvs_for_mc(h, p_cu, blockidx, &mv_1st, &mv_2nd, &ref_1st, &ref_2nd);
        b_mv_valid = check_mv_range(h, &mv_1st, ref_1st, pix_x, pix_y, width, height);
        if (num_mvs > 1) {
            b_mv_valid &= check_mv_range(h, &mv_2nd, ref_2nd, pix_x, pix_y, width, height);
            get_mv_for_mc(h, &mv_2nd, pix_x, pix_y, width, height);
            p_ref2 = h->fref[ref_2nd];
        }
        get_mv_for_mc(h, &mv_1st, pix_x, pix_y, width, height);
        p_ref1 = h->fref[ref_1st];

        if (!b_mv_valid && p_cu->cu_info.i_mode != PRED_SKIP) {
            return 0;
        }

        /* y component */
        if (cal_luma_chroma & 1) {
            p_pred = p_layer->buf_pred_inter + start_y * FREC_STRIDE + start_x;

            mc_luma(p_pred, FREC_STRIDE, mv_1st.x, mv_1st.y, width, height, p_ref1);
            if (num_mvs > 1) {
                mc_luma(p_temp, width, mv_2nd.x, mv_2nd.y, width, height, p_ref2);
                g_funcs.pixf.avg[PART_INDEX(width, height)](p_pred, FREC_STRIDE, p_pred, FREC_STRIDE, p_temp, width, 32);
            }
        }

        /* u and v component */
        if (h->param->chroma_format == CHROMA_420 && (cal_luma_chroma & 2)) {
            int uvoffset = (FREC_CSTRIDE >> 1);
            start_x >>= 1;
            width   >>= 1;
            pix_x   >>= 1;
            start_y >>= CHROMA_V_SHIFT;
            pix_y   >>= CHROMA_V_SHIFT;
            height  >>= CHROMA_V_SHIFT;

            p_pred = p_enc->buf_pred_inter_c + start_y * FREC_CSTRIDE + start_x;

            /* u component */
            mc_chroma(p_pred, p_pred + uvoffset, FREC_CSTRIDE,
                      mv_1st.x, mv_1st.y, width, height, p_ref1);

            if (num_mvs > 1) {
                mc_chroma(p_temp, p_temp + uvoffset, FREC_CSTRIDE,
                          mv_2nd.x, mv_2nd.y, width, height, p_ref2);

                if (width != 2 && width != 6 && height != 2 && height != 6) {
                    pixel_avg_pp_t func_avg = g_funcs.pixf.avg[PART_INDEX(width, height)];
                    func_avg(p_pred           , FREC_CSTRIDE, p_pred           , FREC_CSTRIDE, p_temp           , FREC_CSTRIDE, 32);
                    func_avg(p_pred + uvoffset, FREC_CSTRIDE, p_pred + uvoffset, FREC_CSTRIDE, p_temp + uvoffset, FREC_CSTRIDE, 32);
                } else {
                    g_funcs.pixf.average(p_pred, FREC_CSTRIDE / 2, p_pred, FREC_CSTRIDE / 2, p_temp, FREC_CSTRIDE / 2, width, height * 2);
                }
            }
        }
    }

    return 1;
}


/* ---------------------------------------------------------------------------
 * compute rd-cost for inter cu
 * return 1, means it is the best mode
 *        0, means it is not the best mode
 */
static
int cu_rdcost_inter(xavs2_t *h, aec_t *p_aec, cu_t *p_cu, rdcost_t *min_rdcost, cu_info_t *p_best)
{
    static int8_t tab_try_2level_tu[4][2] = {
        /* try non-split; try split */
        {1, 0},  /* 8x8 */
        {1, 0},  /* 16x16 */
        {1, 0},  /* 32x32 */
        {1, 0},  /* 64x64 */
    };
    bool_t b_try_tu_nonsplit = h->param->b_fast_2lelvel_tu ? tab_try_2level_tu[p_cu->cu_info.i_level - MIN_CU_SIZE_IN_BIT][0] : 1;
    bool_t b_try_tu_split    = h->param->b_fast_2lelvel_tu ? tab_try_2level_tu[p_cu->cu_info.i_level - MIN_CU_SIZE_IN_BIT][1] : 1;
    int mode = p_cu->cu_info.i_mode;
    int cu_size = p_cu->i_size;
    int tmp_cbp;                /* cbp for i_tu_split = 1*/
    int cbp_c = 0;
    int rate_chroma = 0;
    dist_t dist_chroma   = 0;
    dist_t dist_split    = 0;
    dist_t dist_notsplit = 0;
    dist_t best_dist_cur = 0;
    rdcost_t rdcost = *min_rdcost;   // 初始化为最大可允许的RDCost
    rdcost_t rdcost_split = rdcost;
    pel_t *p_fenc = h->lcu.p_fenc[0] + p_cu->i_pos_y * FENC_STRIDE + p_cu->i_pos_x;
    cu_layer_t *p_layer  = cu_get_layer(h, p_cu->cu_info.i_level);
    cu_parallel_t *p_enc = cu_get_enc_context(h, p_cu->cu_info.i_level);

    /* -------------------------------------------------------------
     * 1, begin
     */
    p_cu->cu_info.i_cbp = 0;
    p_cu->cu_info.i_tu_split = TU_SPLIT_NON;  // cu_set_tu_split_type(h, &p_cu->cu_info, 0);

    /* set reference frame and block mode */
    cu_init_pu_inter(h, &p_cu->cu_info, p_cu->cu_info.i_level, mode);

    /* store coding state */
    h->copy_aec_state_rdo(&p_layer->cs_rdo, p_aec);

    /* -------------------------------------------------------------
    * 2. get prediction data
    */
    if (!rdo_get_pred_inter(h, p_cu, 3)) {
        return 0;
    }

    /* -------------------------------------------------------------
     * 3, tu decision
     */
    /* 3.1, check chroma residual coding */
    if (h->param->chroma_format == CHROMA_420) {
        cbp_c = cu_recon_chroma(h, p_aec, p_cu, &dist_chroma);

        if (IS_ALG_ENABLE(OPT_ADVANCE_CHROMA_AEC)) {
            int bits_left = rdo_get_left_bits(h, *min_rdcost, dist_chroma);
            int i_level_c = p_cu->cu_info.i_level - 1;
            int cur_bits_left;
            if (cbp_c & (1 << 4)) {
                cur_bits_left = bits_left - rate_chroma;
                rate_chroma += p_aec->binary.est_chroma_block_coeff(h, p_aec, p_cu, p_cu->cu_info.p_coeff[1], &p_enc->runlevel, i_level_c, cur_bits_left);
            }
            if (cbp_c & (1 << 5)) {
                cur_bits_left = bits_left - rate_chroma;
                rate_chroma += p_aec->binary.est_chroma_block_coeff(h, p_aec, p_cu, p_cu->cu_info.p_coeff[2], &p_enc->runlevel, i_level_c, cur_bits_left);
            }
        }
    }

    /* 3.2, check luma CU tu-split type and CBP */
    /* 3.2.1, get luma residual */
    g_funcs.pixf.sub_ps[PART_INDEX(cu_size, cu_size)](p_enc->coeff_bak, cu_size,
            p_fenc, p_layer->buf_pred_inter,
            FENC_STRIDE, FREC_STRIDE);

    /* 3.2.2, Fast algorithm, check whether TU split is essential */
    if (IS_ALG_ENABLE(OPT_FAST_ZBLOCK) || IS_ALG_ENABLE(OPT_ECU)) {
        p_cu->sum_satd = g_funcs.pixf.sad[PART_INDEX(cu_size, cu_size)](p_layer->buf_pred_inter, FREC_STRIDE, p_fenc, FENC_STRIDE);
        p_cu->is_zero_block = isZeroCuFast(h, p_cu);
    }

    /* only get cost with tu depth equals 1 */
    if ((h->enable_tu_2level == 1) || ((h->enable_tu_2level == 3) && (p_best->i_tu_split != 0))) {
        if (b_try_tu_split && b_try_tu_nonsplit && (IS_ALG_ENABLE(OPT_FAST_ZBLOCK) && p_cu->is_zero_block)) {
            b_try_tu_split = FALSE;
        }

        if (b_try_tu_split) {
            h->copy_aec_state_rdo(&p_enc->cs_tu, p_aec); /* store coding state for tu depth = 1 */

            dist_split = cu_recon_inter_luma(h, &p_enc->cs_tu, p_cu, 0, 1, cbp_c, dist_chroma);
            tmp_cbp = tu_rdcost_inter(h, &p_enc->cs_tu, p_cu, dist_split, rate_chroma, &rdcost_split);

            /* store dct coefficients, rec data and coding state for tu depth = 1*/
            XAVS2_SWAP_PTR(p_cu->cu_info.p_rec[0], p_layer->p_rec_tmp[0]);
            XAVS2_SWAP_PTR(p_cu->cu_info.p_coeff[0], p_layer->p_coeff_tmp[0]);
        } else {
            rdcost_split = MAX_COST;
            tmp_cbp = 0;
        }
        if (rdcost_split >= *min_rdcost) {
            h->copy_aec_state_rdo(p_aec, &p_layer->cs_rdo);
            return 0;  /* return code = 0, means it is not the best mode */
        } else {
            XAVS2_SWAP_PTR(p_cu->cu_info.p_rec[0], p_layer->p_rec_tmp[0]);
            XAVS2_SWAP_PTR(p_cu->cu_info.p_coeff[0], p_layer->p_coeff_tmp[0]);
            p_layer->mode_rdcost[mode] = XAVS2_MIN(rdcost_split, p_layer->mode_rdcost[mode]);
            /* restore the cbp, dct coefficients, rec data and coding state for tu depth = 1*/
            p_cu->cu_info.i_cbp = (int8_t)tmp_cbp;
            *min_rdcost = rdcost_split;
            p_cu->best_dist_total = dist_split;
            h->copy_aec_state_rdo(&p_layer->cs_cu, &p_enc->cs_tu);
            h->copy_aec_state_rdo(p_aec, &p_layer->cs_rdo);
            cu_store_parameters(h, p_cu, p_best);
            return 1;  /* return code = 1, means it is the best mode */
        }
    } else if ((h->enable_tu_2level == 0) || ((h->enable_tu_2level == 3) && (p_best->i_tu_split == 0))) {   /* only get cost with tu depth equals 0 */
        dist_notsplit = cu_recon_inter_luma(h, p_aec, p_cu, 0, 0, cbp_c, dist_chroma);
        tu_rdcost_inter(h, p_aec, p_cu, dist_notsplit, rate_chroma, &rdcost);
    } else {
        if (b_try_tu_split && b_try_tu_nonsplit && (IS_ALG_ENABLE(OPT_FAST_ZBLOCK) && p_cu->is_zero_block)) {
            b_try_tu_split = FALSE;
        }

        if (b_try_tu_split) {
            h->copy_aec_state_rdo(&p_enc->cs_tu, p_aec); /* store coding state for tu depth = 1 */

            dist_split = cu_recon_inter_luma(h, &p_enc->cs_tu, p_cu, 0, 1, cbp_c, dist_chroma);
            tmp_cbp = tu_rdcost_inter(h, &p_enc->cs_tu, p_cu, dist_split, rate_chroma, &rdcost_split);

            /* store dct coefficients, rec data and coding state for tu depth = 1*/
            XAVS2_SWAP_PTR(p_cu->cu_info.p_rec[0], p_layer->p_rec_tmp[0]);
            XAVS2_SWAP_PTR(p_cu->cu_info.p_coeff[0], p_layer->p_coeff_tmp[0]);
        } else {
            rdcost_split = MAX_COST;
            tmp_cbp = 0;
        }

        /* 3.2.4, get cost with tu depth equals 0 */
        if (b_try_tu_nonsplit) {
            dist_notsplit = cu_recon_inter_luma(h, p_aec, p_cu, 0, 0, cbp_c, dist_chroma);
            tu_rdcost_inter(h, p_aec, p_cu, dist_notsplit, rate_chroma, &rdcost);
        }

        /* 3.2.5, choose the best tu depth (whether split or not) */
        if (rdcost > rdcost_split) {
            /* the best tu depth is 1 */
            rdcost = rdcost_split;
            best_dist_cur = dist_split;
            cu_set_tu_split_type(h, &p_cu->cu_info, 1);

            /* restore the cbp, dct coefficients, rec data and coding state for tu depth = 1*/
            p_cu->cu_info.i_cbp = (int8_t)tmp_cbp;
            XAVS2_SWAP_PTR(p_cu->cu_info.p_rec[0], p_layer->p_rec_tmp[0]);
            XAVS2_SWAP_PTR(p_cu->cu_info.p_coeff[0], p_layer->p_coeff_tmp[0]);

            h->copy_aec_state_rdo(p_aec, &p_enc->cs_tu); /* restore coding state */
        } else {
            best_dist_cur = dist_notsplit;
        }
    }

    if (IS_ALG_ENABLE(OPT_CBP_DIRECT) && IS_SKIP_MODE(mode)) {
        /* Skip/Direct模式的残差经过变换量化后为全零块：
         * 此时终止下层CU划分可以得到较多时间节省且损失较小，
         * 但跳过普通PU划分模式并不能带来更多的加速。
         */
        p_cu->b_cbp_direct = (p_cu->cu_info.i_cbp == 0);
    }

    /* 3.3, check skip mode for PRED_SKIP when CBP is nonzero */
    if (IS_SKIP_MODE(p_cu->cu_info.i_mode) && p_cu->cu_info.i_cbp != 0) {
        rdcost_t rdcost_skip = MAX_COST;
        dist_t dist_total_skip;
        int best_tu_split_type = p_cu->cu_info.i_tu_split;

        if (best_tu_split_type == TU_SPLIT_NON) {
            h->copy_aec_state_rdo(&p_enc->cs_tu, p_aec); /* store coding state for best Direct mode */
        }

        h->copy_aec_state_rdo(p_aec, &p_layer->cs_rdo);/* restore coding state */

        tmp_cbp = p_cu->cu_info.i_cbp;
        /* backup reconstruction buffers, prepare for SKIP mode */
        XAVS2_SWAP_PTR(p_cu->cu_info.p_rec[0], p_layer->p_rec_tmp[0]);
        if (cbp_c != 0) {
            XAVS2_SWAP_PTR(p_cu->cu_info.p_rec[1], p_layer->p_rec_tmp[1]);
            XAVS2_SWAP_PTR(p_cu->cu_info.p_rec[2], p_layer->p_rec_tmp[2]);
        }

        /* check SKIP Mode */
        dist_total_skip = cu_recon_inter_luma(h, p_aec, p_cu, 1, 0, cbp_c, dist_chroma);
        tu_rdcost_inter(h, p_aec, p_cu, dist_total_skip, rate_chroma, &rdcost_skip);

        if (rdcost_skip <= rdcost) {
            rdcost = rdcost_skip;    /* skip mode is the best */
            best_dist_cur = dist_total_skip;
            p_cu->cu_info.i_tu_split = TU_SPLIT_NON;
        } else {
            h->copy_aec_state_rdo(p_aec, &p_enc->cs_tu); /* restore coding state */
            /* revert buffers */
            XAVS2_SWAP_PTR(p_cu->cu_info.p_rec[0], p_layer->p_rec_tmp[0]);
            if (cbp_c != 0) {
                XAVS2_SWAP_PTR(p_cu->cu_info.p_rec[1], p_layer->p_rec_tmp[1]);
                XAVS2_SWAP_PTR(p_cu->cu_info.p_rec[2], p_layer->p_rec_tmp[2]);
            }

            p_cu->cu_info.i_cbp = (int8_t)(tmp_cbp);
            p_cu->cu_info.i_tu_split = (int8_t)(best_tu_split_type);
        }
    }

    /* -------------------------------------------------------------
     * 4, store the min cost for current cu mode
     */
    p_layer->mode_rdcost[mode] = XAVS2_MIN(rdcost, p_layer->mode_rdcost[mode]);

    /* -------------------------------------------------------------
     * 5, update the min cost, restore the coding state and return
     */
    if (rdcost >= *min_rdcost) {
        h->copy_aec_state_rdo(p_aec, &p_layer->cs_rdo);
        return 0;  /* return code = 0, means it is not the best mode */
    } else {
        if (mode == PRED_SKIP && IS_ALG_ENABLE(OPT_ROUGH_SKIP_SEL)) {
            /* re-cover best skip prediction data */
            XAVS2_SWAP_PTR(p_layer->buf_pred_inter, p_layer->buf_pred_inter_best);
        }
        *min_rdcost = rdcost;
        p_cu->best_dist_total = best_dist_cur;
        /* store coding state for the best mode */
        h->copy_aec_state_rdo(&p_layer->cs_cu, p_aec);
        h->copy_aec_state_rdo(p_aec, &p_layer->cs_rdo);
        /* update best CU information */
        cu_store_parameters(h, p_cu, p_best);
        return 1;  /* return code = 1, means it is the best mode */
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void cu_set_mvs_noskip(cu_t *p_cu, int blockidx, int ref1, mv_t *pmv1, int ref2, mv_t *pmv2)
{
    const int mode = p_cu->cu_info.i_mode;

    if (mode == PRED_2Nx2N) {
        int k;
        for (k = 0; k < 4; k++) {
            p_cu->mc.mv[k][0].v = pmv1->v;
            p_cu->mc.mv[k][1].v = pmv2->v;
            p_cu->cu_info.ref_idx_1st[k] = (int8_t)ref1;
            p_cu->cu_info.ref_idx_2nd[k] = (int8_t)ref2;
        }
    } else {
        p_cu->mc.mv              [blockidx][0].v = pmv1->v;
        p_cu->cu_info.ref_idx_1st[blockidx] = (int8_t)ref1;

        p_cu->mc.mv              [blockidx][1].v = pmv2->v;
        p_cu->cu_info.ref_idx_2nd[blockidx] = (int8_t)ref2;
    }
}

/* ---------------------------------------------------------------------------
 */
static
rdcost_t cu_rdo_motion_estimation(xavs2_t *h, cu_t *p_cu, xavs2_me_t *p_me, int dualpred_enabled)
{
    const int mode = p_cu->cu_info.i_mode;
    const int block_num = mode == PRED_2Nx2N ? 1 : 2;
    int best_fwd_ref = 0;
    int best_pdir = PDIR_FWD;
    int dual_best_fst_ref = 0;
    int dual_best_snd_ref = 0;
    int block, b8;
    dist_t fwd_cost = MAX_DISTORTION;
    dist_t bwd_cost = 0;
    rdcost_t total_cost = 0;
    int ref1, ref2;  // best references
    mv_t mv1, mv2;   // best mvs
    cb_t *p_cb;
    cu_mode_t *p_mode = cu_get_layer_mode(h, p_cu->cu_info.i_level);
    cu_mv_mode_t *p_mv_mode;

    p_cu->cu_info.b8pdir[0] = p_cu->cu_info.b8pdir[1] = p_cu->cu_info.b8pdir[2] = p_cu->cu_info.b8pdir[3] = (int8_t)PDIR_FWD;
    p_me->bmvcost[0] = p_me->bmvcost[1] = p_me->bmvcost[2] = p_me->bmvcost[3] = MAX_DISTORTION;
    p_me->b_search_dmh = (dualpred_enabled == -1);

    // motion estimation for 2Nx2N, 2NxN, Nx2N, AMP blocks
    for (block = 0; block < block_num; block++) {
        mv1.v = mv2.v = 0;
        p_cb = &p_cu->cu_info.cb[block];
        cu_get_neighbors(h, p_cu, p_cb);

        /* 第一个PU不需要重新进行ME（MVP不变） */
        if (dualpred_enabled < 0 && block == 0) {
            best_fwd_ref = p_mode->ref_idx_single[0];
        } else {
            best_fwd_ref = pred_inter_search_single(h, p_cu, p_cb, p_me, &fwd_cost, &bwd_cost);
        }

        b8 = pu_get_mv_index(mode, block);
        p_mv_mode = &p_mode->mvs[mode][b8];

        if (dualpred_enabled < 0) {
            best_pdir = PDIR_FWD;
            mv1 = p_mv_mode->all_single_mv[best_fwd_ref];
            cu_set_mvs_noskip(p_cu, block, best_fwd_ref, &mv1, INVALID_REF, &mv2);
        } else if (h->i_type == SLICE_TYPE_F) {
            dist_t dual_mcost = MAX_DISTORTION;
            if (dualpred_enabled && (!(p_cu->cu_info.i_level == B8X8_IN_BIT && mode >= PRED_2NxN && mode <= PRED_nRx2N))) {
                pred_inter_search_dual(h, p_cu, p_cb, p_me, &dual_mcost, &dual_best_fst_ref, &dual_best_snd_ref);
            }

            if (fwd_cost <= dual_mcost) {
                best_pdir = PDIR_FWD;
                ref1 = best_fwd_ref;
                mv1 = p_mv_mode->all_single_mv[ref1];
                cu_set_mvs_noskip(p_cu, block, ref1, &mv1, INVALID_REF, &mv2);
                p_cu->mvcost[block] = p_me->bmvcost[PDIR_FWD];
            } else {
                fwd_cost = dual_mcost;
                best_pdir = PDIR_DUAL;
                ref1 = dual_best_fst_ref;
                ref2 = dual_best_snd_ref;
                mv1 = p_mv_mode->all_dual_mv_1st[ref1];
                mv2 = p_mv_mode->all_dual_mv_2nd[ref1];
                cu_set_mvs_noskip(p_cu, block, ref1, &mv1, ref2, &mv2);
                p_cu->mvcost[block] = p_me->bmvcost[PDIR_DUAL];
            }
        } else if (h->i_type == SLICE_TYPE_B) {
            dist_t sym_mcost = MAX_DISTORTION;
            dist_t bid_mcost = MAX_DISTORTION;

            best_fwd_ref = 0;               // must reset
            if (!((p_cu->cu_info.i_level == B8X8_IN_BIT) && (mode >= PRED_2NxN && mode <= PRED_nRx2N))) {
                pred_inter_search_bi(h, p_cu, p_cb, p_me, &sym_mcost, &bid_mcost);
            }

            if (fwd_cost <= bwd_cost && fwd_cost <= sym_mcost && fwd_cost <= bid_mcost) {
                best_pdir = PDIR_FWD;
                ref1 = B_FWD;
                mv1 = p_mv_mode->all_single_mv[ref1];
                cu_set_mvs_noskip(p_cu, block, ref1, &mv1, INVALID_REF, &mv2);
                p_cu->mvcost[block] = p_me->bmvcost[PDIR_FWD];
            } else if (bwd_cost <= fwd_cost && bwd_cost <= sym_mcost && bwd_cost <= bid_mcost) {
                fwd_cost = bwd_cost;
                best_pdir = PDIR_BWD;
                ref2 = B_BWD;
                mv2 = p_mv_mode->all_single_mv[B_BWD];
                cu_set_mvs_noskip(p_cu, block, INVALID_REF, &mv1, ref2, &mv2);
                p_cu->mvcost[block] = p_me->bmvcost[PDIR_BWD];
            } else if (sym_mcost <= fwd_cost && sym_mcost <= bwd_cost && sym_mcost <= bid_mcost) {
                int dist_fwd = calculate_distance(h, B_FWD);  // fwd
                int dist_bwd = calculate_distance(h, B_BWD);  // bwd
                fwd_cost = sym_mcost;
                best_pdir = PDIR_SYM;
                ref1 = B_FWD;
                ref2 = B_BWD;
                mv1 = p_mv_mode->all_sym_mv[0];
                mv2.x = -scale_mv_skip  (   mv1.x, dist_bwd, dist_fwd);
                mv2.y = -scale_mv_skip_y(h, mv1.y, dist_bwd, dist_fwd);
                cu_set_mvs_noskip(p_cu, block, ref1, &mv1, ref2, &mv2);
                p_cu->mvcost[block] = p_me->bmvcost[PDIR_SYM];
            } else {
                fwd_cost = bid_mcost;
                best_pdir = PDIR_BID;
                ref1 = B_FWD;
                ref2 = B_BWD;
                mv1 = p_mv_mode->all_dual_mv_1st[0];
                mv2 = p_mv_mode->all_dual_mv_2nd[0];
                cu_set_mvs_noskip(p_cu, block, ref1, &mv1, ref2, &mv2);
                p_cu->mvcost[block] = p_me->bmvcost[PDIR_BID];
            }
        } else {
            ref1 = best_fwd_ref;
            mv1 = p_mv_mode->all_single_mv[ref1];
            cu_set_mvs_noskip(p_cu, block, ref1, &mv1, INVALID_REF, &mv2);
            p_cu->mvcost[block] = p_me->mvcost[PDIR_FWD];
        }

        total_cost += fwd_cost;

        // store reference frame index and direction parameters
        p_mode->ref_idx_single[block] = (int8_t)best_fwd_ref;
        p_cu->cu_info.b8pdir[block] = (int8_t)best_pdir;
    }

    cu_get_mvds(h, p_cu);  // 生成MVD

    return total_cost;  // 返回最小Cost
}

//#if OPT_DMH_CANDIDATE
/* ---------------------------------------------------------------------------
 * 提前获取最优的DMH模式候选，减少RDO次数
 */
static int dmh_bits[9] = {
//  0, 3, 3, 4, 4, 5, 5, 5, 5
    0, 0, 0, 0, 0, 0, 0, 0, 0
};

static int rdo_get_dmh_candidate(xavs2_t *h, cu_t *p_cu, rdcost_t rdcost_non_dmh)
{
    const int num_dmh_modes = DMH_MODE_NUM + DMH_MODE_NUM - 1;
    int cu_size = 1 << p_cu->cu_info.i_level;
    pixel_ssd_t cmp_dmh = g_funcs.pixf.ssd[PART_INDEX(cu_size, cu_size)];
    rdcost_t min_distotion = MAX_COST;
    dist_t distortion;
    rdcost_t cost;
    int best_dmh_cand = -1;
    cu_layer_t *p_layer = cu_get_layer(h, p_cu->cu_info.i_level);
    pel_t *p_fenc = h->lcu.p_fenc[0] + p_cu->i_pos_y * FENC_STRIDE + p_cu->i_pos_x;
    int i;
    int rate;
    /* 遍历DMH模式执行预测并计算失真，取失真最小的一个模式作为DMH候选集 */
    for (i = 1; i < num_dmh_modes; i++) {
        /* get prediction data and luma distortion */
        p_cu->cu_info.dmh_mode = (int8_t)(i);
        if (rdo_get_pred_inter(h, p_cu, 1)) {
            rate = dmh_bits[i];
            distortion = cmp_dmh(p_fenc, FENC_STRIDE, p_layer->buf_pred_inter, FREC_STRIDE);
            cost = distortion + h->f_lambda_mode * rate;
            if (cost < min_distotion) {
                min_distotion = cost;
                best_dmh_cand = i;
            }
        }
    }

    if (IS_ALG_ENABLE(OPT_SKIP_DMH_THRES) && min_distotion > (rdcost_t)(1.2 * rdcost_non_dmh)) {
        /* 不考虑残差编码带来的distortion减少 */
        return -1;
    } else {
        return best_dmh_cand;
    }
}
//#endif


/* ---------------------------------------------------------------------------
 * 尝试所有帧间预测块划分方式，选择一个最优的划分
 */
static int cu_select_inter_partition(xavs2_t *h, cu_t *p_cu, int i_level, uint32_t inter_modes,
                                     cu_info_t *best, rdcost_t *p_min_rdcost,
                                     int b_dhp_enabled, int b_check_dmh)
{
    cu_layer_t *p_layer  = cu_get_layer(h, p_cu->cu_info.i_level);
    cu_parallel_t *p_enc = cu_get_enc_context(h, p_cu->cu_info.i_level);
    int best_cu_mode = 1;
    int mode;
    int cu_size = 1 << i_level;
    int cu_size_c = cu_size >> 1;
    int pix_x = p_cu->i_pix_x;
    int pix_y = p_cu->i_pix_y;
    int pix_x_c = pix_x >> 1;
    int pix_y_c = pix_y >> CHROMA_V_SHIFT;
    pel_t *p_fenc[3];
    int i;
    int64_t min_cost = MAX_COST;
    int64_t mecost;
    int ref1, ref2;

    UNUSED_PARAMETER(b_check_dmh);
    UNUSED_PARAMETER(p_min_rdcost);
    memcpy(best, &p_cu->cu_info, sizeof(cu_info_t));

    //inter_modes |= (uint32_t)((1 << PRED_2NxN) | (1 << PRED_Nx2N));

    for (mode = 1; mode < MAX_INTER_MODES; mode++) {
        /* 执行运动估计 */

        if (!(inter_modes & (1 << mode))) {
            continue;           // 直接跳过不可用模式的决策
        }

        /* 快速决策(OPT_BYPASS_AMP)：如果P2NxN未获得最优，直接跳过相同划分方向的PRED_2NxnU/PRED_2NxnD; PNx2N同理 */
        if (IS_ALG_ENABLE(OPT_BYPASS_AMP) && i_level > B16X16_IN_BIT) {
            if ((mode == PRED_2NxnU || mode == PRED_2NxnD) && best_cu_mode != PRED_2NxN) {
                continue;
            } else if ((mode == PRED_nLx2N || mode == PRED_nRx2N) && best_cu_mode != PRED_Nx2N) {
                continue;
            }
        }

        p_cu->cu_info.i_mode = (int8_t)mode;
        cu_init_pu_inter(h, &p_cu->cu_info, i_level, mode);
        cu_rdo_motion_estimation(h, p_cu, &h->me_state, b_dhp_enabled);

        /* 估计Cost选取最小的 */
        p_cu->cu_info.directskip_wsm_idx = 0;
        p_cu->cu_info.directskip_mhp_idx = DS_NONE;
        p_cu->cu_info.dmh_mode = 0;

        rdo_get_pred_inter(h, p_cu, 3);
        p_fenc[0] = h->lcu.p_fenc[0] + pix_y   * FENC_STRIDE + pix_x;
        p_fenc[1] = h->lcu.p_fenc[1] + pix_y_c * FENC_STRIDE + pix_x_c;
        p_fenc[2] = h->lcu.p_fenc[2] + pix_y_c * FENC_STRIDE + pix_x_c;

        mecost  = g_funcs.pixf.sa8d[PART_INDEX(cu_size, cu_size)](p_layer->buf_pred_inter, FREC_STRIDE, p_fenc[0], FENC_STRIDE);
        mecost += g_funcs.pixf.sa8d[PART_INDEX(cu_size_c, cu_size_c)](p_enc->buf_pred_inter_c, FREC_CSTRIDE, p_fenc[1], FENC_STRIDE);
        mecost += g_funcs.pixf.sa8d[PART_INDEX(cu_size_c, cu_size_c)](p_enc->buf_pred_inter_c + (FREC_CSTRIDE >> 1), FREC_CSTRIDE, p_fenc[2], FENC_STRIDE);

        for (i = 0; i < p_cu->cu_info.num_pu; i++) {
            mecost += p_cu->mvcost[i];
            ref1 = p_cu->cu_info.ref_idx_1st[i];
            ref2= p_cu->cu_info.ref_idx_2nd[i];
            if (h->i_type != SLICE_TYPE_B) {
                mecost += (ref1 == INVALID_REF? 0: REF_COST(ref1));
                mecost += (ref2 == INVALID_REF? 0: REF_COST(ref2));
            }
        }

        if (mecost < min_cost) {
            memcpy(&p_layer->cu_mode.best_mc_tmp, &p_cu->mc, sizeof(p_cu->mc));
            memcpy(best, &p_cu->cu_info, sizeof(cu_info_t));
            min_cost     = mecost;
            best_cu_mode = mode;
        }
    }

    return best_cu_mode;
}

/* ---------------------------------------------------------------------------
 * 尝试普通帧间预测块划分方式，并计算相应的Cost
 */
static
void cu_check_inter_partition(xavs2_t *h, aec_t *p_aec, cu_t *p_cu, int mode, int i_level,
                              cu_info_t *best, rdcost_t *p_min_rdcost,
                              int b_dhp_enabled, int b_check_dmh)
{
    /* set reference frame and block mode */
    cu_init_pu_inter(h, &p_cu->cu_info, i_level, mode);

    /* ME */
    cu_rdo_motion_estimation(h, p_cu, &h->me_state, b_dhp_enabled);

    h->lcu.bypass_all_dmh = 0;

    /* 计算一个帧间划分模式的RDCost，以确定最优编码模式 */
    p_cu->cu_info.directskip_wsm_idx = 0;
    p_cu->cu_info.directskip_mhp_idx = DS_NONE;
    p_cu->cu_info.dmh_mode = 0;
    cu_rdcost_inter(h, p_aec, p_cu, p_min_rdcost, best);

    /* 检查DMH模式 */
    if (h->i_type == SLICE_TYPE_F && h->param->enable_dmh && !h->lcu.bypass_all_dmh && b_check_dmh
        && !(i_level == B8X8_IN_BIT && mode != PRED_2Nx2N)) {  // disable 8x4 or 4x8 2MVs/PU mode
        int dmh_mode_candidate = 0;
        int max_dmh_mode;
        int best_dmh_mode = 0;
        int dmh_mode;

        if (p_cu->cu_info.b8pdir[0] == PDIR_FWD && p_cu->cu_info.b8pdir[1] == PDIR_FWD &&
            p_cu->cu_info.b8pdir[2] == PDIR_FWD && p_cu->cu_info.b8pdir[3] == PDIR_FWD) {
            /* ME确定的最优的PU预测方向均为单前向，此时只需要检查后续DMH模式 */
            dmh_mode = 1;
        } else { // DHP 开启且参考帧数量为2时才有可能上述条件不成立
            /* 最优的PU中包含双前向块，此时需要计算PU均为单前向时的RDCosts，再遍历后续DMH模式 */
            /* 此时需重新ME，同时第一个PU不需要重新搜索 */
            cu_rdo_motion_estimation(h, p_cu, &h->me_state, -1);
            dmh_mode = 0;
        }

        /* 总计 2 * (DMH_MODE_NUM - 1) + 1 个模式 */
        max_dmh_mode = DMH_MODE_NUM + DMH_MODE_NUM - 1;

        /* 快速算法，从DMH可选模式中估计最需要做的模式
            * 避免依次遍历所有模式巨大的计算量
            */
        if (IS_ALG_ENABLE(OPT_DMH_CANDIDATE)) {
            dmh_mode_candidate = rdo_get_dmh_candidate(h, p_cu, *p_min_rdcost);
        }

        // 当某个模式下的残差为全零时，跳过所有后续dmh模式
        for (; dmh_mode < max_dmh_mode && !h->lcu.bypass_all_dmh; dmh_mode++) {
            if (IS_ALG_ENABLE(OPT_DMH_CANDIDATE)) {
                if (dmh_mode != 0 && dmh_mode != dmh_mode_candidate) {
                    continue;
                }
            } else {
                if (dmh_mode > (DMH_MODE_NUM - 1)) {
                    if (best_dmh_mode != (dmh_mode - (DMH_MODE_NUM - 1))) { // 只在同方向上扩展，其他跳过
                        continue;
                    }
                }
            }

            p_cu->cu_info.dmh_mode = (int8_t)dmh_mode;
            if (cu_rdcost_inter(h, p_aec, p_cu, p_min_rdcost, best)) {
                best_dmh_mode = dmh_mode;
            }
        }  // end loop of DMH modes
    }  // end of check DMH modes

}


/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
void cu_set_mvs_skip(xavs2_t *h, cu_t *p_cu)
{
    int weighted_skip_mode = p_cu->cu_info.directskip_wsm_idx;
    int ds_mode = p_cu->cu_info.directskip_mhp_idx;
    const cu_mode_t *p_cu_mode = cu_get_layer_mode(h, p_cu->cu_info.i_level);
    int k;

    assert(p_cu->cu_info.i_mode == PRED_SKIP);
    assert(h->i_type != SLICE_TYPE_I);

    if (ds_mode != DS_NONE) {
        mv_t mv1 = p_cu_mode->skip_mv_1st[ds_mode];
        mv_t mv2 = p_cu_mode->skip_mv_2nd[ds_mode];
        int8_t ref1 = p_cu_mode->skip_ref_1st[ds_mode];
        int8_t ref2 = p_cu_mode->skip_ref_2nd[ds_mode];

        int i_dir_pred = tab_pdir_bskip[ds_mode];

        for (k = 0; k < 4; k++) {
            p_cu->cu_info.b8pdir[k] = (int8_t)i_dir_pred;
            p_cu->mc.mv[k][0] = mv1;
            p_cu->mc.mv[k][1] = mv2;
            p_cu->cu_info.ref_idx_1st[k] = ref1;
            p_cu->cu_info.ref_idx_2nd[k] = ref2;
        }
    } else if (weighted_skip_mode) {
        for (k = 0; k < 4; k++) {
            p_cu->cu_info.b8pdir[k] = PDIR_FWD;
            p_cu->mc.mv[k][0] = p_cu_mode->tskip_mv[k][0];
            p_cu->mc.mv[k][1] = p_cu_mode->tskip_mv[k][weighted_skip_mode];
            p_cu->cu_info.ref_idx_1st[k] = 0;
            p_cu->cu_info.ref_idx_2nd[k] = (int8_t)weighted_skip_mode;
        }
    } else if (h->i_type != SLICE_TYPE_B) {
        for (k = 0; k < 4; k++) {
            p_cu->cu_info.b8pdir[k] = PDIR_FWD;
            p_cu->mc.mv[k][0] = p_cu_mode->tskip_mv[k][0];
            p_cu->mc.mv[k][1].v = 0;
            p_cu->cu_info.ref_idx_1st[k] = 0;
            p_cu->cu_info.ref_idx_2nd[k] = INVALID_REF;
        }
    } else {
        for (k = 0; k < 4; k++) {
            p_cu->cu_info.b8pdir[k] = PDIR_SYM;
            p_cu->mc.mv[k][0] = p_cu_mode->tskip_mv[k][0];
            p_cu->mc.mv[k][1] = p_cu_mode->tskip_mv[k][1];
            p_cu->cu_info.ref_idx_1st[k] = B_FWD;
            p_cu->cu_info.ref_idx_2nd[k] = B_BWD;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
typedef struct cu_skip_mc_t {
    mv_t         mv_1st[4];
    mv_t         mv_2nd[4];
    int8_t       ref_1st[4];
    int8_t       ref_2nd[4];
} cu_skip_mc_t;

/* ---------------------------------------------------------------------------
 * 更新Skip的MV集，以检测当前模式的MV是否被遍历过
 */
static ALWAYS_INLINE
int is_same_skip_mc_param(const cu_skip_mc_t *p_src1, const cu_skip_mc_t *p_src2)
{
    uint32_t *p1 = (uint32_t *)p_src1;
    uint32_t *p2 = (uint32_t *)p_src2;
    int num = sizeof(cu_skip_mc_t) >> 2;
    int i;

    for (i = 0; i < num; i++) {
        if (p1[i] != p2[i]) {
            return 0;
        }
    }

    return 1;
}

/* ---------------------------------------------------------------------------
 * 更新Skip的MV集，以检测当前模式的MV是否被遍历过
 */
static
int update_skip_mv_list(cu_skip_mc_t *p_skip_mvs, int i_num, cu_t *p_cu)
{
    cu_skip_mc_t cur_mc;
    int i;
    for (i = 0; i < 4; i++) {
        cur_mc.mv_1st[i].v = p_cu->mc.mv[i][0].v;
        cur_mc.mv_2nd[i].v = p_cu->mc.mv[i][1].v;
        cur_mc.ref_1st[i] = p_cu->cu_info.ref_idx_1st[i];
        cur_mc.ref_2nd[i] = p_cu->cu_info.ref_idx_2nd[i];
    }

    for (i = 0; i < i_num; i++) {
        if (is_same_skip_mc_param(p_skip_mvs + i, &cur_mc)) {
            break;
        }
    }

    if (i != i_num) {
        return 0;
    } else {
        memcpy(p_skip_mvs + i_num, &cur_mc, sizeof(cu_skip_mc_t));
        return 1;
    }
}

/* ---------------------------------------------------------------------------
 * 检查Skip/Direct模式的编码代价（依据预测残差），选取最优的Skip子模式进行一次RDO
 */
static
void cu_check_skip_direct_rough2(xavs2_t *h, aec_t *p_aec, cu_info_t *p_best, cu_t *p_cu, rdcost_t *p_min_rdcost)
{
    cu_skip_mc_t skip_mc_params[DS_MAX_NUM + XAVS2_MAX_REFS];
    int num_mc_params = 0;
    int max_skip_mode_num, i;
    int cu_size = p_cu->i_size;
    pixel_ssd_t cmp_skip = g_funcs.pixf.sa8d[PART_INDEX(cu_size, cu_size)];
    cu_layer_t *p_layer = cu_get_layer(h, p_cu->cu_info.i_level);
    pel_t *p_fenc = h->lcu.p_fenc[0] + p_cu->i_pos_y * FENC_STRIDE + p_cu->i_pos_x;
    dist_t distortion;
    rdcost_t rdcost;
    rdcost_t min_rdcost = MAX_COST;
    int best_skip_mode = DS_NONE;
    int best_weighted_skip = -1;   // also used to verify an additional skip mode is found

    cb_t cur_cb = { { 0 } };
    cur_cb.w = cur_cb.h = (int8_t)p_cu->i_size;
    cu_get_neighbors(h, p_cu, &cur_cb);

    /* get Skip/Direct MVs and temporal SKIP mode number */
    max_skip_mode_num = h->lcu.get_skip_mvs(h, p_cu);

    /* 0, init cu data */
    p_cu->cu_info.dmh_mode = 0;
    p_cu->cu_info.i_cbp = 0;
    int rate;

    /* 1, temporal skip mode, derive MV from temporal */
    p_cu->cu_info.directskip_mhp_idx = DS_NONE;
    p_cu->cu_info.directskip_wsm_idx = 0;

    /* 时域MVP预测的直接算RDCost，再跟空域的最优的RDCost做比较，增益 3%左右，时间增加 20%~30% */
    cu_set_mvs_skip(h, p_cu);
    cu_rdcost_inter(h, p_aec, p_cu, p_min_rdcost, p_best);

    /* 2, Weighted skip mode, derive MV from temporal and scaling */
    for (i = 1; i < max_skip_mode_num; i++) {
        int need_check_mv;
        p_cu->cu_info.directskip_wsm_idx = (int8_t)i;
        cu_set_mvs_skip(h, p_cu);
        cu_init_pu_inter(h, &p_cu->cu_info, p_cu->cu_info.i_level, PRED_SKIP);
        need_check_mv = update_skip_mv_list(skip_mc_params, num_mc_params, p_cu);
        num_mc_params += need_check_mv;
        if (need_check_mv && rdo_get_pred_inter(h, p_cu, 1)) {
            rate = p_aec->binary.est_cu_header(h, p_aec, p_cu);
            distortion = cmp_skip(p_fenc, FENC_STRIDE, p_layer->buf_pred_inter, FREC_STRIDE);
            rdcost = distortion + h->f_lambda_mode * rate;
            if (rdcost < min_rdcost) {
                XAVS2_SWAP_PTR(p_layer->buf_pred_inter, p_layer->buf_pred_inter_best);
                min_rdcost = rdcost;
                best_weighted_skip = i;
            }
        }
    }

    /* 3, 四个spatial direct类型 (single first, single second, dual first, dual second) */
    if ((h->i_type == SLICE_TYPE_B || (h->i_type == SLICE_TYPE_F && h->param->enable_mhp_skip)) && (!h->fdec->rps.referd_by_others && h->i_type == SLICE_TYPE_B)) {
        p_cu->cu_info.directskip_wsm_idx = 0;
        for (i = 0; i < DS_MAX_NUM; i++) {
            int need_check_mv;
            p_cu->cu_info.directskip_mhp_idx = (int8_t)i;
            cu_init_pu_inter(h, &p_cu->cu_info, p_cu->cu_info.i_level, PRED_SKIP);
            cu_set_mvs_skip(h, p_cu);
            need_check_mv = update_skip_mv_list(skip_mc_params, num_mc_params, p_cu);
            num_mc_params += need_check_mv;
            if (need_check_mv && rdo_get_pred_inter(h, p_cu, 1)) {
                rate = headerbits_skipmode[4+i];
                distortion = cmp_skip(p_fenc, FENC_STRIDE, p_layer->buf_pred_inter, FREC_STRIDE);
                rdcost = distortion + h->f_lambda_mode * rate;
                if (rdcost < min_rdcost) {
                    XAVS2_SWAP_PTR(p_layer->buf_pred_inter, p_layer->buf_pred_inter_best);
                    min_rdcost = rdcost;
                    best_weighted_skip = 0;
                    best_skip_mode = i;
                }
            }
        }
        /* 在distortion最小的模式中选择一个最优的 */
        p_cu->cu_info.directskip_mhp_idx = (int8_t)best_skip_mode;
        p_cu->cu_info.directskip_wsm_idx = (int8_t)best_weighted_skip;
        cu_set_mvs_skip(h, p_cu);
        cu_rdcost_inter(h, p_aec, p_cu, p_min_rdcost, p_best);
    } else if ((h->i_type == SLICE_TYPE_B || (h->i_type == SLICE_TYPE_F && h->param->enable_mhp_skip)) && (h->fdec->rps.poc == 2 || h->fdec->rps.poc == 6)) {
        if (p_cu->p_left_cu != NULL && p_cu->p_topA_cu != NULL && p_cu->p_topL_cu != NULL && p_cu->p_topR_cu != NULL) {
            if ((p_cu->p_left_cu->i_mode == 0 && p_cu->p_topA_cu->i_mode == 0 && p_cu->p_topL_cu->i_mode == 0 && p_cu->p_topR_cu->i_mode == 0) && (p_cu->p_left_cu->i_cbp == 0 || p_cu->p_topA_cu->i_cbp == 0 || p_cu->p_topL_cu->i_cbp == 0 || p_cu->p_topR_cu->i_cbp == 0)) {
                p_cu->cu_info.directskip_wsm_idx = 0;
                for (i = 0; i < DS_MAX_NUM; i++) {
                    int need_check_mv;
                    p_cu->cu_info.directskip_mhp_idx = (int8_t)i;
                    cu_init_pu_inter(h, &p_cu->cu_info, p_cu->cu_info.i_level, PRED_SKIP);
                    cu_set_mvs_skip(h, p_cu);
                    need_check_mv = update_skip_mv_list(skip_mc_params, num_mc_params, p_cu);
                    num_mc_params += need_check_mv;
                    if (need_check_mv && rdo_get_pred_inter(h, p_cu, 1)) {
                        rate = headerbits_skipmode[4 + i];
                        distortion = cmp_skip(p_fenc, FENC_STRIDE, p_layer->buf_pred_inter, FREC_STRIDE);
                        rdcost = distortion + h->f_lambda_mode * rate;
                        if (rdcost < min_rdcost) {
                            XAVS2_SWAP_PTR(p_layer->buf_pred_inter, p_layer->buf_pred_inter_best);
                            min_rdcost = rdcost;
                            best_weighted_skip = 0;
                            best_skip_mode = i;
                        }
                    }
                }
                /* 在distortion最小的模式中选择一个最优的 */
                p_cu->cu_info.directskip_mhp_idx = (int8_t)best_skip_mode;
                p_cu->cu_info.directskip_wsm_idx = (int8_t)best_weighted_skip;
                cu_set_mvs_skip(h, p_cu);
                cu_rdcost_inter(h, p_aec, p_cu, p_min_rdcost, p_best);

            } else {
                p_cu->cu_info.directskip_wsm_idx = 0;
                for (i = 0; i < DS_MAX_NUM; i++) {
                    p_cu->cu_info.directskip_mhp_idx = (int8_t)i;
                    cu_init_pu_inter(h, &p_cu->cu_info, p_cu->cu_info.i_level, PRED_SKIP);
                    cu_set_mvs_skip(h, p_cu);
                    cu_rdcost_inter(h, p_aec, p_cu, p_min_rdcost, p_best);
                }
            }
        } else {
            p_cu->cu_info.directskip_wsm_idx = 0;
            for (i = 0; i < DS_MAX_NUM; i++) {
                p_cu->cu_info.directskip_mhp_idx = (int8_t)i;
                cu_init_pu_inter(h, &p_cu->cu_info, p_cu->cu_info.i_level, PRED_SKIP);
                cu_set_mvs_skip(h, p_cu);
                cu_rdcost_inter(h, p_aec, p_cu, p_min_rdcost, p_best);
            }
        }
    } else if (h->i_type == SLICE_TYPE_B || (h->i_type == SLICE_TYPE_F && h->param->enable_mhp_skip)) {
        if (p_cu->p_left_cu != NULL && p_cu->p_topA_cu != NULL && p_cu->p_topL_cu != NULL && p_cu->p_topR_cu != NULL) {
            if ((p_cu->p_left_cu->i_mode == 0 && p_cu->p_topA_cu->i_mode == 0 && p_cu->p_topL_cu->i_mode == 0 && p_cu->p_topR_cu->i_mode == 0) && (p_cu->p_left_cu->i_cbp == 0 && p_cu->p_topA_cu->i_cbp == 0 && p_cu->p_topL_cu->i_cbp == 0 && p_cu->p_topR_cu->i_cbp == 0)) {
                p_cu->cu_info.directskip_wsm_idx = 0;
                for (i = 0; i < DS_MAX_NUM; i++) {
                    int need_check_mv;
                    p_cu->cu_info.directskip_mhp_idx = (int8_t)i;
                    cu_init_pu_inter(h, &p_cu->cu_info, p_cu->cu_info.i_level, PRED_SKIP);
                    cu_set_mvs_skip(h, p_cu);
                    need_check_mv = update_skip_mv_list(skip_mc_params, num_mc_params, p_cu);
                    num_mc_params += need_check_mv;
                    if (need_check_mv && rdo_get_pred_inter(h, p_cu, 1)) {
                        rate = headerbits_skipmode[4 + i];
                        distortion = cmp_skip(p_fenc, FENC_STRIDE, p_layer->buf_pred_inter, FREC_STRIDE);
                        rdcost = distortion + h->f_lambda_mode * rate;
                        if (rdcost < min_rdcost) {
                            XAVS2_SWAP_PTR(p_layer->buf_pred_inter, p_layer->buf_pred_inter_best);
                            min_rdcost = rdcost;
                            best_weighted_skip = 0;
                            best_skip_mode = i;
                        }
                    }
                }
                /* 在distortion最小的模式中选择一个最优的 */
                p_cu->cu_info.directskip_mhp_idx = (int8_t)best_skip_mode;
                p_cu->cu_info.directskip_wsm_idx = (int8_t)best_weighted_skip;
                cu_set_mvs_skip(h, p_cu);
                cu_rdcost_inter(h, p_aec, p_cu, p_min_rdcost, p_best);

            } else {
                p_cu->cu_info.directskip_wsm_idx = 0;
                for (i = 0; i < DS_MAX_NUM; i++) {
                    p_cu->cu_info.directskip_mhp_idx = (int8_t)i;
                    cu_init_pu_inter(h, &p_cu->cu_info, p_cu->cu_info.i_level, PRED_SKIP);
                    cu_set_mvs_skip(h, p_cu);
                    cu_rdcost_inter(h, p_aec, p_cu, p_min_rdcost, p_best);
                }
            }
        } else {
            p_cu->cu_info.directskip_wsm_idx = 0;
            for (i = 0; i < DS_MAX_NUM; i++) {
                p_cu->cu_info.directskip_mhp_idx = (int8_t)i;
                cu_init_pu_inter(h, &p_cu->cu_info, p_cu->cu_info.i_level, PRED_SKIP);
                cu_set_mvs_skip(h, p_cu);
                cu_rdcost_inter(h, p_aec, p_cu, p_min_rdcost, p_best);
            }
        }
    }
}

static
void cu_check_skip_direct_rough1(xavs2_t *h, aec_t *p_aec, cu_info_t *p_best, cu_t *p_cu, rdcost_t *p_min_rdcost)
{
    cu_skip_mc_t skip_mc_params[DS_MAX_NUM + XAVS2_MAX_REFS];
    int num_mc_params = 0;
    int max_skip_mode_num, i;
    int cu_size = p_cu->i_size;
    pixel_ssd_t cmp_skip = g_funcs.pixf.sa8d[PART_INDEX(cu_size, cu_size)];
    cu_layer_t *p_layer = cu_get_layer(h, p_cu->cu_info.i_level);
    pel_t *p_fenc = h->lcu.p_fenc[0] + p_cu->i_pos_y * FENC_STRIDE + p_cu->i_pos_x;
    dist_t distortion;
    rdcost_t rdcost;
    rdcost_t min_rdcost = MAX_COST;
    int best_skip_mode = DS_NONE;
    int best_weighted_skip = -1;   // also used to verify an additional skip mode is found

    cb_t cur_cb = { { 0 } };
    cur_cb.w = cur_cb.h = (int8_t)p_cu->i_size;
    cu_get_neighbors(h, p_cu, &cur_cb);

    /* get Skip/Direct MVs and temporal SKIP mode number */
    max_skip_mode_num = h->lcu.get_skip_mvs(h, p_cu);

    /* 0, init cu data */
    p_cu->cu_info.dmh_mode = 0;
    p_cu->cu_info.i_cbp = 0;
    int rate;

    /* 1, temporal skip mode, derive MV from temporal */
    p_cu->cu_info.directskip_mhp_idx = DS_NONE;
    p_cu->cu_info.directskip_wsm_idx = 0;

    /* 时域MVP预测的直接算RDCost，再跟空域的最优的RDCost做比较，增益 3%左右，时间增加 20%~30% */
    cu_set_mvs_skip(h, p_cu);
    cu_init_pu_inter(h, &p_cu->cu_info, p_cu->cu_info.i_level, PRED_SKIP);
    num_mc_params += update_skip_mv_list(skip_mc_params, num_mc_params, p_cu);
    if (rdo_get_pred_inter(h, p_cu, 1)) {
        rate = headerbits_skipmode[0];//p_aec->binary.est_cu_header(h, p_aec, p_cu);
        distortion = cmp_skip(p_fenc, FENC_STRIDE, p_layer->buf_pred_inter, FREC_STRIDE);
        min_rdcost = distortion + h->f_lambda_mode * rate;
        XAVS2_SWAP_PTR(p_layer->buf_pred_inter, p_layer->buf_pred_inter_best);
        best_weighted_skip = 0;
        best_skip_mode = DS_NONE;
    }

    /* 2, Weighted skip mode, derive MV from temporal and scaling */
    for (i = 1; i < max_skip_mode_num; i++) {
        int need_check_mv;
        p_cu->cu_info.directskip_wsm_idx = (int8_t)i;
        cu_set_mvs_skip(h, p_cu);
        cu_init_pu_inter(h, &p_cu->cu_info, p_cu->cu_info.i_level, PRED_SKIP);
        need_check_mv = update_skip_mv_list(skip_mc_params, num_mc_params, p_cu);
        num_mc_params += need_check_mv;
        if (need_check_mv && rdo_get_pred_inter(h, p_cu, 1)) {
            rate = headerbits_skipmode[i];//p_aec->binary.est_cu_header(h, p_aec, p_cu);
            distortion = cmp_skip(p_fenc, FENC_STRIDE, p_layer->buf_pred_inter, FREC_STRIDE);
            rdcost = distortion + h->f_lambda_mode * rate;
            if (rdcost < min_rdcost) {
                XAVS2_SWAP_PTR(p_layer->buf_pred_inter, p_layer->buf_pred_inter_best);
                min_rdcost = rdcost;
                best_weighted_skip = i;
            }
        }
    }

    /* 3, 四个spatial direct类型 (single first, single second, dual first, dual second) */
    if (h->i_type == SLICE_TYPE_B || (h->i_type == SLICE_TYPE_F && h->param->enable_mhp_skip)) {
        p_cu->cu_info.directskip_wsm_idx = 0;
        for (i = 0; i < DS_MAX_NUM; i++) {
            int need_check_mv;
            p_cu->cu_info.directskip_mhp_idx = (int8_t)i;
            cu_init_pu_inter(h, &p_cu->cu_info, p_cu->cu_info.i_level, PRED_SKIP);
            cu_set_mvs_skip(h, p_cu);
            need_check_mv = update_skip_mv_list(skip_mc_params, num_mc_params, p_cu);
            num_mc_params += need_check_mv;
            if (need_check_mv && rdo_get_pred_inter(h, p_cu, 1)) {
                rate = headerbits_skipmode[4 + i];//p_aec->binary.est_cu_header(h, p_aec, p_cu);
                distortion = cmp_skip(p_fenc, FENC_STRIDE, p_layer->buf_pred_inter, FREC_STRIDE);
                rdcost = distortion + h->f_lambda_mode * rate;
                if (rdcost < min_rdcost) {
                    XAVS2_SWAP_PTR(p_layer->buf_pred_inter, p_layer->buf_pred_inter_best);
                    min_rdcost = rdcost;
                    best_weighted_skip = 0;
                    best_skip_mode = i;
                }
            }
        }
    }

    /* 在distortion最小的模式中选择一个最优的 */
    p_cu->cu_info.directskip_mhp_idx = (int8_t)best_skip_mode;
    p_cu->cu_info.directskip_wsm_idx = (int8_t)best_weighted_skip;
    cu_set_mvs_skip(h, p_cu);
    cu_rdcost_inter(h, p_aec, p_cu, p_min_rdcost, p_best);
}


/* ---------------------------------------------------------------------------
 * 检查Skip/Direct模式的编码代价（RDO），选取最优的Skip子模式
 */
static
void cu_check_skip_direct_fullrdo(xavs2_t *h, aec_t *p_aec, cu_info_t *p_best, cu_t *p_cu, rdcost_t *p_min_rdcost)
{
    int max_skip_mode_num, i;

    cb_t cur_cb = { { 0 } };
    cur_cb.w = cur_cb.h = (int8_t)p_cu->i_size;
    cu_get_neighbors(h, p_cu, &cur_cb);

    /* get Skip/Direct MVs and temporal SKIP mode number */
    max_skip_mode_num = h->lcu.get_skip_mvs(h, p_cu);

    /* 0, init cu data */
    p_cu->cu_info.dmh_mode = 0;

    /* 1, temporal skip mode, derive MV from temporal */
    p_cu->cu_info.directskip_mhp_idx = DS_NONE;
    p_cu->cu_info.directskip_wsm_idx = 0;

    /* 时域MVP预测的直接算RDCost，再跟空域的最优的RDCost做比较，增益 3%左右，时间增加 20%~30% */
    cu_set_mvs_skip(h, p_cu);
    cu_rdcost_inter(h, p_aec, p_cu, p_min_rdcost, p_best);

    /* 2, Weighted skip mode, derive MV from temporal and scaling */
    for (i = 1; i < max_skip_mode_num; i++) {
        p_cu->cu_info.directskip_wsm_idx = (int8_t)i;
        cu_set_mvs_skip(h, p_cu);
        cu_rdcost_inter(h, p_aec, p_cu, p_min_rdcost, p_best);
    }

    /* 3, 四个spatial direct类型 (single first, single second, dual first, dual second) */
    if (h->i_type == SLICE_TYPE_B || (h->i_type == SLICE_TYPE_F && h->param->enable_mhp_skip)) {
        p_cu->cu_info.directskip_wsm_idx = 0;
        for (i = 0; i < DS_MAX_NUM; i++) {
            p_cu->cu_info.directskip_mhp_idx = (int8_t)i;
            cu_init_pu_inter(h, &p_cu->cu_info, p_cu->cu_info.i_level, PRED_SKIP);
            cu_set_mvs_skip(h, p_cu);
            cu_rdcost_inter(h, p_aec, p_cu, p_min_rdcost, p_best);
        }
    }
}

#if SAVE_CU_INFO
//#if OPT_EARLY_SKIP
/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
int is_early_skip(xavs2_t *h, cu_t *p_cu)
{
#define IS_EARLY_SKIP_CU(mode, cbp)     (((mode) <= PRED_2Nx2N || (mode) == PRED_I_2Nx2N ) && (cbp) == 0)    // faster
//#define IS_EARLY_SKIP_CU(mode, cbp)   ((mode) == PRED_SKIP  && (cbp) == 0)    // fast
#define IS_EAYLY_SKIP(p_cu_info)        IS_EARLY_SKIP_CU(p_cu_info->i_mode, p_cu_info->i_cbp)
    // each neighbor block (left, top, top-left, top-right, col) is skip mode?
    int left_skip     = p_cu->p_left_cu && IS_EAYLY_SKIP(p_cu->p_left_cu);
    int top_skip      = p_cu->p_topA_cu && IS_EAYLY_SKIP(p_cu->p_topA_cu);
    int topleft_skip  = p_cu->p_topL_cu && IS_EAYLY_SKIP(p_cu->p_topL_cu);
    int topright_skip = p_cu->p_topR_cu && IS_EAYLY_SKIP(p_cu->p_topR_cu);
    xavs2_frame_t *p_col_ref = h->fref[0];
    int i_scu_xy = p_cu->i_scu_xy;
    int col_skip = IS_EARLY_SKIP_CU(p_col_ref->cu_mode[i_scu_xy], p_col_ref->cu_cbp[i_scu_xy]);
#undef IS_EARLY_SKIP_CU
#undef IS_EAYLY_SKIP

    return (left_skip + top_skip + topleft_skip + topright_skip + col_skip > 4) && p_cu->cu_info.i_mode <= PRED_2Nx2N;
    // return left_skip && top_skip && topleft_skip && topright_skip && col_skip && p_cu->cu_info.i_mode == PRED_SKIP;
}
//#endif
#endif

//#if OPT_PSC_MD || OPT_TR_KEY_FRAME_MD
/* ---------------------------------------------------------------------------
 */
static void update_valid_modes_by_complexity(xavs2_t *h, cu_t *p_cu, uint32_t *valid_pred_modes)
{
    static const int mode_weight_factor[3][MAX_PRED_MODES + 1] = {
        { 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 },    // neighbor bitsize <  cur bitsize
        { 1, 1, 1, 2, 2, 4, 4, 4, 4, 1, 4, 4, 4 },    // neighbor bitsize == cur bitsize
        { 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }     // neighbor bitsize >  cur bitsize
    };
#define REAL_CU_TYPE(cu_type, cbp)  (cu_type + 1 - (!cu_type && !!cbp))
    static const double thres_complexity_min = 1.0f;
    static const double thres_complexity_max = 4.0f;
    static const uint32_t modes_simple_tex = (1 << PRED_SKIP) | (1 << PRED_2Nx2N) | (1 << PRED_I_2Nx2N);
    static const uint32_t mask_complex_tex = (uint32_t)~((1 << PRED_I_2Nx2N) | (1 << PRED_2Nx2N) | (1 << PRED_2NxN) | (1 << PRED_Nx2N));
    static const uint32_t mask_non_keyframe_modes = 0x0F;

    int cu_weight_sum     = 0;
    int mode_complexity   = 0;
    int i_level           = p_cu->cu_info.i_level;
    cu_info_t *cu_left    = p_cu->p_left_cu;
    cu_info_t *cu_up      = p_cu->p_topA_cu;
    cu_info_t *cu_upleft  = p_cu->p_topL_cu;
    cu_info_t *cu_upright = p_cu->p_topR_cu;
#if SAVE_CU_INFO
    int scu_xy             = p_cu->i_scu_xy;
    xavs2_frame_t *ref_fwd = h->i_type == SLICE_TYPE_B ? h->fref[1] : h->fref[0];
    xavs2_frame_t *ref_bwd = h->i_type == SLICE_TYPE_B ? h->fref[0] : NULL;
#endif
    int sign_idx, type_idx;

    if (IS_ALG_ENABLE(OPT_TR_KEY_FRAME_MD)) {
        if (h->fdec->i_frm_poc & 2) {
            *valid_pred_modes &= mask_non_keyframe_modes;
            return;
        }
    }

    /* IS_ALG_ENABLE(OPT_PSC_MD) */
    // left
    if (cu_left) {
        cu_weight_sum += 6;
        sign_idx = xavs2_sign3(cu_left->i_level - i_level) + 1;
        type_idx = REAL_CU_TYPE(cu_left->i_mode, cu_left->i_cbp);
        mode_complexity += 6 * mode_weight_factor[sign_idx][type_idx];
    }

    // up
    if (cu_up) {
        cu_weight_sum += 6;
        sign_idx = xavs2_sign3(cu_up->i_level - i_level) + 1;
        type_idx = REAL_CU_TYPE(cu_up->i_mode, cu_up->i_cbp);
        mode_complexity += 6 * mode_weight_factor[sign_idx][type_idx];
    }

    // upleft
    if (cu_upleft) {
        cu_weight_sum += 2;
        sign_idx = xavs2_sign3(cu_upleft->i_level - i_level) + 1;
        type_idx = REAL_CU_TYPE(cu_upleft->i_mode, cu_upleft->i_cbp);
        mode_complexity += 2 * mode_weight_factor[sign_idx][type_idx];
    }

    // upright
    if (cu_upright) {
        cu_weight_sum += 2;
        sign_idx = xavs2_sign3(cu_upright->i_level - i_level) + 1;
        type_idx = REAL_CU_TYPE(cu_upright->i_mode, cu_upright->i_cbp);
        mode_complexity += 2 * mode_weight_factor[sign_idx][type_idx];
    }

#if SAVE_CU_INFO
    // temporal forward
    if (ref_fwd) {
        cu_weight_sum += 1;
        sign_idx = xavs2_sign3(ref_fwd->cu_level[scu_xy] - i_level) + 1;
        type_idx = REAL_CU_TYPE(ref_fwd->cu_mode[scu_xy], ref_fwd->cu_cbp[scu_xy]);
        mode_complexity += 1 * mode_weight_factor[sign_idx][type_idx];
    }

    // temporal backward
    if (ref_bwd) {
        cu_weight_sum += 1;
        sign_idx = xavs2_sign3(ref_bwd->cu_level[scu_xy] - i_level) + 1;
        type_idx = REAL_CU_TYPE(ref_bwd->cu_mode[scu_xy], ref_bwd->cu_cbp[scu_xy]);
        mode_complexity += 1 * mode_weight_factor[sign_idx][type_idx];
    }
#else
    mode_complexity += 2;
#endif

    if (mode_complexity < thres_complexity_min * cu_weight_sum) {
        *valid_pred_modes &= modes_simple_tex;
    } else if (mode_complexity >= thres_complexity_max * cu_weight_sum) {
        *valid_pred_modes &= mask_complex_tex;
    }

#undef REAL_CU_TYPE
}
//#endif

//#if OPT_ET_HOMO_MV
/* ---------------------------------------------------------------------------
 */
static int is_ET_inter_recur(xavs2_t *h, cu_t *p_cu, cu_info_t *curr_best)
{
    int b_avail_up   = p_cu->p_topA_cu != NULL;
    int b_avail_left = p_cu->p_left_cu != NULL;
#if SAVE_CU_INFO
    int b_avail_col  = IS_INTER_MODE(curr_best->i_mode) & IS_INTER_MODE(h->fref[0]->cu_mode[p_cu->i_scu_xy]);
#else
    int b_avail_col  = FALSE;
#endif
    int num_blk_pixels[4];
    float mv_avg_x[4], mv_avg_y[4];
    int start_b4_x, start_b4_y;

    if (b_avail_up && b_avail_left && b_avail_col) {
        cu_mode_t *p_mode = cu_get_layer_mode(h, curr_best->i_level);
        int w_in_4x4 = h->i_width_in_minpu;
        int b4_x = p_cu->i_pix_x >> MIN_PU_SIZE_IN_BIT;
        int b4_y = p_cu->i_pix_y >> MIN_PU_SIZE_IN_BIT;
        int b4_size = 1 << (p_cu->cu_info.i_level - MIN_PU_SIZE_IN_BIT);
        int b4_num  = b4_size * b4_size;
        float mvs_avg_x = 0, mvs_avg_y = 0;
        float mvs_var_x = 0, mvs_var_y = 0;
        mv_t cur_blk_mvs[4];
        const mv_t *p_mv_1st = h->fwd_1st_mv + b4_y * w_in_4x4 + b4_x;
        const mv_t *col_mv = h->fref[0]->pu_mv;
        const int w_in_16x16 = (h->i_width_in_minpu + 3) >> 2;
        int i, j, k;
        assert(curr_best->i_mode >= 0 && curr_best->i_mode < MAX_INTER_MODES);

        for (i = 0; i < 4; i++) {
            mv_avg_x[i] = 0;
            mv_avg_y[i] = 0;
        }

        // left column & top row
        start_b4_x = -1;
        start_b4_y = 0;
        for (j = 0; j < b4_size; j++) {
            mv_avg_x[0] += p_mv_1st[j * w_in_4x4 - 1].x;
            mv_avg_y[0] += p_mv_1st[j * w_in_4x4 - 1].y;
            mv_avg_x[1] += p_mv_1st[j - w_in_4x4].x;
            mv_avg_y[1] += p_mv_1st[j - w_in_4x4].y;
        }
        mv_avg_x[0] *= b4_size;
        mv_avg_y[0] *= b4_size;
        mv_avg_x[1] *= b4_size;
        mv_avg_y[1] *= b4_size;

        // collocated
        start_b4_x = b4_x;
        start_b4_y = b4_y;
        for (j = 0; j < b4_size; j++) {
            for (i = 0; i < b4_size; i++) {
                mv_avg_x[2] += col_mv[((start_b4_y + j) >> 2) * w_in_16x16 + ((start_b4_x + i) >> 2)].x;
                mv_avg_y[2] += col_mv[((start_b4_y + j) >> 2) * w_in_16x16 + ((start_b4_x + i) >> 2)].y;
            }
        }

        // current cu
        for (k = 0; k < curr_best->num_pu; k++) {
            mv_t mv_1st = p_mode->best_mc.mv[k][0];
            cb_t cur_cb = curr_best->cb[k];
            int width, height;

            cur_cb.v >>= 2;
            width  = cur_cb.w;
            height = cur_cb.h;

            cur_blk_mvs[k] = mv_1st;
            num_blk_pixels[k] = width * height;

            mv_avg_x[3] += mv_1st.x * num_blk_pixels[k];
            mv_avg_y[3] += mv_1st.y * num_blk_pixels[k];
        }

        for (; k < 4; k++) {
            num_blk_pixels[k] = 0;
            cur_blk_mvs[k].v = 0;
        }

        for (i = 0; i < 4; i++) {
            mv_avg_x[i] /= b4_num;
            mv_avg_y[i] /= b4_num;
            mvs_avg_x += mv_avg_x[i];
            mvs_avg_y += mv_avg_y[i];
        }

        // left column & top row
        for (j = 0; j < b4_size; j++) {
            mvs_var_x += XAVS2_ABS(p_mv_1st[j * w_in_4x4 - 1].x - mvs_avg_x);
            mvs_var_y += XAVS2_ABS(p_mv_1st[j * w_in_4x4 - 1].y - mvs_avg_y);
            mvs_var_x += XAVS2_ABS(p_mv_1st[j - w_in_4x4].x - mvs_avg_x);
            mvs_var_y += XAVS2_ABS(p_mv_1st[j - w_in_4x4].y - mvs_avg_y);
        }
        mvs_var_x *= b4_size;
        mvs_var_y *= b4_size;

        // collocated
        start_b4_x = b4_x;
        start_b4_y = b4_y;
        for (j = 0; j < b4_size; j++) {
            for (i = 0; i < b4_size; i++) {
                mvs_var_x += XAVS2_ABS(col_mv[((start_b4_y + j) >> 2) * w_in_16x16 + ((start_b4_x + i) >> 2)].x - mvs_avg_x);
                mvs_var_y += XAVS2_ABS(col_mv[((start_b4_y + j) >> 2) * w_in_16x16 + ((start_b4_x + i) >> 2)].y - mvs_avg_y);
            }
        }

        // current
        for (i = 0; i < 4; i++) {
            mvs_var_x += XAVS2_ABS(cur_blk_mvs[i].x - mvs_avg_x) * num_blk_pixels[i];
            mvs_var_y += XAVS2_ABS(cur_blk_mvs[i].y - mvs_avg_y) * num_blk_pixels[i];
        }

        return (mvs_var_x < 4 * b4_num && mvs_var_y < 4 * b4_num);
    }

    return 0;
}
//#endif

/* ---------------------------------------------------------------------------
 * encode an intra cu (for I-picture)
 */
static
rdcost_t compress_cu_intra(xavs2_t *h, aec_t *p_aec, cu_t *p_cu, cu_info_t *best, rdcost_t cost_limit)
{
    int i_level = p_cu->cu_info.i_level;
    cu_layer_t *p_layer = cu_get_layer(h, i_level);
    uint32_t intra_modes;   // valid intra modes
    rdcost_t split_flag_cost = 0;
    rdcost_t min_rdcost = MAX_COST;
    int mode;

    UNUSED_PARAMETER(cost_limit);
    if (i_level > MIN_CU_SIZE_IN_BIT) {
        split_flag_cost = h->f_lambda_mode * p_aec->binary.write_ctu_split_flag(p_aec, 0, i_level);
    }

    h->lcu.b_enable_rdoq     = (h->param->i_rdoq_level == RDOQ_ALL);
    h->lcu.b_2nd_rdcost_pass = 1;
    h->lcu.get_intra_dir_for_rdo_luma = h->get_intra_candidates_luma;

    //===== SET VALID MODES =====
    intra_modes = cu_get_valid_modes(h, h->i_type, i_level);

    // reset default parameters for chroma intra predictor
    p_cu->cu_info.i_intra_mode_c     = DC_PRED_C;
    p_cu->cu_info.directskip_wsm_idx = 0;
    p_cu->cu_info.directskip_mhp_idx = DS_NONE;

    //===== GET BEST MACROBLOCK MODE =====
    for (mode = PRED_I_2Nx2N; mode <= PRED_I_nx2N; mode++) {
        if (!(intra_modes & (1 << mode))) {
            continue;           // 直接跳过不可用模式
        }

        if (IS_ALG_ENABLE(OPT_BYPASS_SDIP)) {
            // 最后一个非对称帧内模式的提前跳过
            if (sdip_early_bypass(h, p_layer, mode)) {
                continue;
            }
        }

        // init coding block(s)
        p_cu->cu_info.i_mode = (int8_t)mode;    // set cu type

        cu_check_intra(h, p_aec, p_cu, best, mode, &min_rdcost);
    }

    /* 检查最优模式，带RDOQ */
    if (h->param->i_rdoq_level == RDOQ_CU_LEVEL && best->i_cbp > 0) {
        h->lcu.get_intra_dir_for_rdo_luma = rdo_get_pred_intra_luma_2nd_pass;
        h->lcu.b_enable_rdoq = 1;
        mode = best->i_mode;
        cu_copy_info(&p_cu->cu_info, best);
        cu_check_intra(h, p_aec, p_cu, best, mode, &min_rdcost);
    }

    min_rdcost += split_flag_cost;
    return p_layer->best_rdcost = min_rdcost;
}

/* ---------------------------------------------------------------------------
 * encode an inter cu (for none I-picture)
 */
static
rdcost_t compress_cu_inter(xavs2_t *h, aec_t *p_aec, cu_t *p_cu, cu_info_t *best,
                           uint32_t avail_modes,  // available prediction partition modes
                           rdcost_t min_rdcost,   // Cost of Skip/Direct mode
                           rdcost_t cost_limit)
{
    int b_dhp_enabled   = h->param->enable_dhp && h->i_type == SLICE_TYPE_F && h->i_ref > 1;
    int i_level = p_cu->cu_info.i_level;
    int b_bypass_intra  = 0;
    int b_check_dmh     = 1;
    int mode;
    cu_layer_t *p_layer  = cu_get_layer(h, p_cu->cu_info.i_level);

    /* -------------------------------------------------------------
     * 1, 初始化
     */
    UNUSED_PARAMETER(cost_limit);
    h->lcu.get_intra_dir_for_rdo_luma = h->get_intra_candidates_luma;
    h->enable_tu_2level = IS_ALG_ENABLE(OPT_TU_LEVEL_DEC) ? 0 : 2;
    h->lcu.b_enable_rdoq      = (h->param->i_rdoq_level == RDOQ_ALL);
    h->lcu.b_2nd_rdcost_pass  = 0;

    for (mode = 0; mode < MAX_PRED_MODES; mode++) {
        p_layer->mode_rdcost[mode] = MAX_COST;
    }

    /* reset chroma intra predictor to default */
    p_cu->cu_info.i_intra_mode_c = DC_PRED_C;   // @luofl：请勿移除此行，否则会导致不匹配问题；20170304 19:52:32

    /* -------------------------------------------------------------
     * 2, 检查Skip和Direct模式
     */
    /* 检查所有SKIP/Direct子模式 */
    p_cu->cu_info.i_mode = PRED_SKIP;

    if (IS_ALG_ENABLE(OPT_ROUGH_SKIP_SEL) && h->skip_rough_improved) {
        cu_check_skip_direct_rough2(h, p_aec, best, p_cu, &min_rdcost);
    } else if (IS_ALG_ENABLE(OPT_ROUGH_SKIP_SEL)) {
        cu_check_skip_direct_rough1(h, p_aec, best, p_cu, &min_rdcost);
    } else {
        cu_check_skip_direct_fullrdo(h, p_aec, best, p_cu, &min_rdcost);
    }

    p_layer->best_rdcost = min_rdcost;

    // update valid modes
    if (IS_ALG_ENABLE(OPT_PSC_MD) || IS_ALG_ENABLE(OPT_TR_KEY_FRAME_MD)) {
        update_valid_modes_by_complexity(h, p_cu, &avail_modes);
    }

    if (IS_ALG_ENABLE(OPT_ROUGH_MODE_SKIP)) {
        if (h->i_type == SLICE_TYPE_B && !h->fdec->rps.referd_by_others && (i_level == B64X64_IN_BIT || i_level == B32X32_IN_BIT)) {
            avail_modes &= (uint32_t)~((1 << PRED_2NxN) | (1 << PRED_Nx2N));
        }
    }


    /* -------------------------------------------------------------
     * 3, 非Skip/Direct的帧间模式
     */
    for (mode = 1; mode < MAX_INTER_MODES; mode++) {
        if (!(avail_modes & (1 << mode))) {
            continue;           // 直接跳过不可用模式的决策
        }

        /* -------------------------------------------------------------
         * 3.1 与Skip/Direct模式相关的快速模式决策算法放在此处
         */

#if SAVE_CU_INFO
        if (IS_ALG_ENABLE(OPT_EARLY_SKIP)) {
            if (is_early_skip(h, p_cu)) {
                b_bypass_intra = 1;
                break;              // bypass all rest inter & intra modes
            }
        }
#endif

        /* 快速PU划分模式决策：
         * 如果P2NxN未获得最优，直接跳过相同划分方向的PRED_2NxnU/PRED_2NxnD; PNx2N同理 */
        if (IS_ALG_ENABLE(OPT_BYPASS_AMP) && i_level > B16X16_IN_BIT) {
            if ((mode == PRED_2NxnU || mode == PRED_2NxnD) && best->i_mode != PRED_2NxN) {
                continue;
            } else if ((mode == PRED_nLx2N || mode == PRED_nRx2N) && best->i_mode != PRED_Nx2N) {
                continue;
            }
        }


        /* -------------------------------------------------------------
         * 3.2, 尝试编码当前PU划分模式
         */
        p_cu->cu_info.i_mode = (int8_t)mode;
        if (IS_ALG_ENABLE(OPT_ROUGH_PU_SEL) && mode == PRED_2Nx2N) {
            cu_info_t cur_best;
            cu_select_inter_partition(h, p_cu, i_level, avail_modes, &cur_best, &min_rdcost, b_dhp_enabled, b_check_dmh);
            mode = cur_best.i_mode;
            cu_copy_info(&p_cu->cu_info, &cur_best);
            memcpy(&p_cu->mc, &p_layer->cu_mode.best_mc_tmp, sizeof(p_cu->mc));  /* 拷贝MV信息用于补偿 */
            cu_rdcost_inter(h, p_aec, p_cu, &min_rdcost, best);
            avail_modes &= ~0xfe;   // 禁用掉剩余帧间划分模式
        } else {
            cu_check_inter_partition(h, p_aec, p_cu, mode, i_level, best, &min_rdcost, b_dhp_enabled, b_check_dmh);
        }

        /* -------------------------------------------------------------
         * 3.3, 当前普通PU划分模式编码后的快速决策算法
         */

        if (best->i_mode == mode) {
            if (best->dmh_mode != 0) {
                if (IS_ALG_ENABLE(OPT_BYPASS_MODE_FPIC)) {
                    b_bypass_intra = 1;
                }
            }
        }

        if (IS_ALG_ENABLE(OPT_FAST_CBF_MODE) && p_cu->cu_info.i_cbp == 0) {
            if (mode == PRED_2Nx2N && best->i_mode == PRED_SKIP) {
                b_bypass_intra = 1;
                break;              // bypass all rest inter & intra modes
            }
            if (mode >= PRED_2Nx2N && best->i_mode == mode) {
                b_bypass_intra = 1;
                break;              // bypass all rest inter modes
            }
        }

        if (IS_ALG_ENABLE(OPT_FAST_PU_SEL) && p_cu->cu_info.i_cbp == 0) {
            if (mode == PRED_2Nx2N && best->i_mode == PRED_SKIP) {
                b_bypass_intra = 1;
                break;              // bypass all rest inter & intra modes
            }
        }
    }

    /* 做第二层TU划分，选出最优模式 */
    if (IS_ALG_ENABLE(OPT_TU_LEVEL_DEC) && best->i_cbp > 0) {
        h->enable_tu_2level = 1;
        mode = best->i_mode;
        cu_copy_info(&p_cu->cu_info, best);
        memcpy(&p_cu->mc, &p_layer->cu_mode.best_mc, sizeof(p_cu->mc));  /* 拷贝MV信息用于补偿 */
        cu_rdcost_inter(h, p_aec, p_cu, &min_rdcost, best);
    }// end of checking inter PU partitions

    /* 通过帧级预分析判定，此帧不需要做帧内预测时，跳过后续帧内模式 */
    if (!h->fenc->b_enable_intra) {
        b_bypass_intra = 1;
    }

    if (IS_ALG_ENABLE(OPT_BYPASS_INTRA_BPIC)) {
        b_bypass_intra |= (h->i_type == SLICE_TYPE_B && best->i_cbp == 0);   // 禁用B帧的帧内预测模式
    }

    /* 条件禁用部分帧内划分模式 */
    if (IS_ALG_ENABLE(OPT_CMS_ETMD)) {
        /* 帧间模式做完之后，若最优模式的CBP为零，则不再遍历所有帧内预测模式 */
        b_bypass_intra |= ((best->i_cbp == 0) && (best->i_mode == 0));
        /* 依据帧间最优划分模式，筛选不需要遍历的模式 */
        // if (IS_HOR_PU_PART(best->i_mode)) {
        //     avail_modes &= !(1 << PRED_I_nx2N);
        // } else if (IS_VER_PU_PART(best->i_mode)) {
        //     avail_modes &= !(1 << PRED_I_2Nxn);
        // } else if (best->i_mode == PRED_SKIP) {
        //     avail_modes &= (1 << PRED_I_2Nx2N);
        // }
    }

    if (IS_ALG_ENABLE(OPT_ROUGH_MODE_SKIP)) {
        if (h->i_type == SLICE_TYPE_B && i_level == B64X64_IN_BIT) {
            b_bypass_intra = 1;
        }

        if (!h->fdec->rps.referd_by_others && h->i_type == SLICE_TYPE_B && i_level != B16X16_IN_BIT) {
            b_bypass_intra = 1;
        }
    }

    /* 若当前最小RDCost小于了某个阈值，表明帧间预测模式已经能够较好地预测，此时不再继续尝试帧内模式 */
    if (IS_ALG_ENABLE(OPT_FAST_INTRA_IN_INTER) && min_rdcost < h->thres_qsfd_cu[1][i_level - MIN_CU_SIZE_IN_BIT]) {
        b_bypass_intra = 1;
    }

    /* -------------------------------------------------------------
     * 4, get best intra mode
     */
    if (!b_bypass_intra) {
        for (mode = PRED_I_2Nx2N; mode <= PRED_I_nx2N; mode++) {
            if (!(avail_modes & (1 << mode))) {
                continue;           // 直接跳过不可用模式的决策
            }

            if (IS_ALG_ENABLE(OPT_BYPASS_SDIP)) {
                // 最后一个非对称帧内模式的提前跳过
                if (sdip_early_bypass(h, p_layer, mode)) {
                    continue;
                }
            }

            // init coding block(s)
            p_cu->cu_info.i_mode = (int8_t)mode;

            // cal rd-cost
            cu_check_intra(h, p_aec, p_cu, best, mode, &min_rdcost);

            if (IS_ALG_ENABLE(OPT_CMS_ETMD)) {
                if (best->i_mode != PRED_I_2Nx2N && mode == PRED_I_2Nx2N) {
                    break;
                }
            }
        }
    }

    /* 检查最优模式,包括TU划分还是不划分的确定，带RDOQ */
    if (h->param->i_rdoq_level == RDOQ_CU_LEVEL&& best->i_cbp > 0) {
        if (IS_ALG_ENABLE(OPT_TU_LEVEL_DEC)) {
            h->enable_tu_2level = 3;
        } else {
            h->enable_tu_2level = 2;
        }
        h->lcu.get_intra_dir_for_rdo_luma = rdo_get_pred_intra_luma_2nd_pass;
        h->lcu.b_enable_rdoq = 1;
        h->lcu.b_2nd_rdcost_pass = 1;
        mode = best->i_mode;
        cu_copy_info(&p_cu->cu_info, best);
        if (IS_INTRA_MODE(mode)) {
            if((!IS_ALG_ENABLE(OPT_BYPASS_INTRA_RDOQ)) || h->i_type == SLICE_TYPE_F) {
                cu_check_intra(h, p_aec, p_cu, best, mode, &min_rdcost);
            }
        } else {
            memcpy(&p_cu->mc, &p_layer->cu_mode.best_mc, sizeof(p_cu->mc));  /* 拷贝MV信息用于补偿 */
            cu_rdcost_inter(h, p_aec, p_cu, &min_rdcost, best);
        }
    } else if (IS_ALG_ENABLE(OPT_BIT_EST_PSZT) && i_level >= 5 && (best->i_mode != PRED_SKIP || best->i_cbp != 0)) {
        h->enable_tu_2level = 2;
        h->lcu.get_intra_dir_for_rdo_luma = rdo_get_pred_intra_luma_2nd_pass;
        h->lcu.b_2nd_rdcost_pass = 1;
        // recheck RDCost
        mode = best->i_mode;
        cu_copy_info(&p_cu->cu_info, best);
        if (IS_INTRA_MODE(mode)) {
            cu_check_intra(h, p_aec, p_cu, best, mode, &min_rdcost);
        } else {
            memcpy(&p_cu->mc, &p_layer->cu_mode.best_mc, sizeof(p_cu->mc));  /* 拷贝MV信息用于补偿 */
            cu_rdcost_inter(h, p_aec, p_cu, &min_rdcost, best);
        }
    }

    return p_layer->best_rdcost = min_rdcost;
}

/* ---------------------------------------------------------------------------
 */
static INLINE
int ctu_intra_depth_pred_mad(xavs2_t *h, int level, int pix_x, int pix_y)
{
    static const int MAD_TH0[] = {
        2, 2 * 256, 2 * 1024, 3 * 4096
    };
    pel_t *p_src_base = h->lcu.p_fenc[0] + pix_y * FENC_STRIDE + pix_x;
    int cu_size = 1 << level;

    int mad = g_funcs.pixf.madf[level - MIN_CU_SIZE_IN_BIT](p_src_base, FENC_STRIDE, cu_size);

    return mad >= MAD_TH0[level - MIN_CU_SIZE_IN_BIT];
}


/**
 * ===========================================================================
 * interface function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * RDOPT初始化时，设置不同帧和CU大小可用的模式，后续直接查表
 */
void xavs2_init_valid_mode_table(xavs2_t *h)
{
    int frm_type;
    int level;

    for (frm_type = 0; frm_type < SLICE_TYPE_NUM; frm_type++) {
        int inter_frame = (frm_type != SLICE_TYPE_I);
        for (level = MIN_CU_SIZE_IN_BIT; level <= MAX_CU_SIZE_IN_BIT; level++) {
            uint32_t valid_pred_modes = 0;

            /* set validity of inter modes */
            if (inter_frame) {
                valid_pred_modes |= 1 << PRED_SKIP;
                valid_pred_modes |= 1 << PRED_2Nx2N;
                valid_pred_modes |= h->param->inter_2pu << PRED_2NxN;
                valid_pred_modes |= h->param->inter_2pu << PRED_Nx2N;
                if (h->param->enable_amp && level > MIN_CU_SIZE_IN_BIT) {
                    valid_pred_modes |= 1 << PRED_2NxnU;
                    valid_pred_modes |= 1 << PRED_2NxnD;
                    valid_pred_modes |= 1 << PRED_nLx2N;
                    valid_pred_modes |= 1 << PRED_nRx2N;
                }
            }

            /* set validity of intra modes */
            if (!inter_frame || h->param->enable_intra) {
                valid_pred_modes |= 1 << PRED_I_2Nx2N;
                valid_pred_modes |= (level == MIN_CU_SIZE_IN_BIT) << PRED_I_NxN;

                // only valid for 32x8,8x32, 16x4,4x16
                if (h->param->enable_sdip && (level == B16X16_IN_BIT || level == B32X32_IN_BIT)) {
                    valid_pred_modes |= 1 << PRED_I_2Nxn;
                    valid_pred_modes |= 1 << PRED_I_nx2N;
                }
            }

            // @luofl: SDIP is disabled here for speedup in inter frames
            if (inter_frame && level != MIN_CU_SIZE_IN_BIT) {
                valid_pred_modes &= ~((1 << PRED_I_2Nxn) | (1 << PRED_I_nx2N) | (1 << PRED_I_NxN));
            }

            if (inter_frame && IS_ALG_ENABLE(OPT_PU_RMS)) {
                if (level == B8X8_IN_BIT || level == B16X16_IN_BIT) {
                    valid_pred_modes &= (uint32_t)((1 << PRED_2Nx2N) | (1 << PRED_I_2Nx2N));
                }
            }
            h->valid_modes[frm_type][level - MIN_CU_SIZE_IN_BIT] = valid_pred_modes;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
rdcost_t compress_ctu_intra(xavs2_t *h, aec_t *p_aec, cu_t *p_cu, int i_level, int i_min_level, int i_max_level, rdcost_t cost_limit)
{
    aec_t cs_aec;
    cu_layer_t *p_layer    = cu_get_layer(h, i_level);
    cu_info_t *best        = &p_layer->cu_best;
    rdcost_t large_cu_cost = MAX_COST;
    rdcost_t split_cu_cost = MAX_COST;
    int b_inside_pic       = (p_cu->i_pix_x + p_cu->i_size <= h->i_width) && (p_cu->i_pix_y + p_cu->i_size <= h->i_height);
    int b_split_ctu        = (i_level > i_min_level || !b_inside_pic);
    int b_check_large_cu   = (b_inside_pic && i_level <= i_max_level);

    /* init current CU ---------------------------------------------
     */
    cu_init(h, p_cu, best, i_level);

    /* coding current CU -------------------------------------------
     */
    if (b_check_large_cu) {
        if (IS_ALG_ENABLE(OPT_ET_INTRA_DEPTH) && b_split_ctu) {
            b_split_ctu &= ctu_intra_depth_pred_mad(h, i_level, p_cu->i_pos_x, p_cu->i_pos_y);
        }

        h->copy_aec_state_rdo(&cs_aec, p_aec);
        large_cu_cost = compress_cu_intra(h, &cs_aec, p_cu, best, cost_limit);

        /* QSFD, skip smaller CU partitions */
        if (IS_ALG_ENABLE(OPT_CU_QSFD)) {
            if (p_cu->cu_info.i_level > 3 && large_cu_cost < h->thres_qsfd_cu[1][p_cu->cu_info.i_level - 3]) {
                b_split_ctu = FALSE;
            }
        }
    }

    /* coding 4 sub-CUs --------------------------------------------
     */
    if (b_split_ctu) {
        int i;
        split_cu_cost = 0;

        // cal split cost
        if (b_inside_pic) {
            split_cu_cost += h->f_lambda_mode * p_aec->binary.write_ctu_split_flag(p_aec, 1, i_level);
        }

        for (i = 0; i < 4; i++) {
            cu_t *p_sub_cu = p_cu->sub_cu[i];

            if (p_sub_cu->i_pix_x >= h->i_width || p_sub_cu->i_pix_y >= h->i_height) {
                continue;       // current sub CU is outside the frame+*
            }

            split_cu_cost += compress_ctu_intra(h, p_aec, p_sub_cu, i_level - 1, i_min_level, i_max_level, large_cu_cost - split_cu_cost);

            if (split_cu_cost > large_cu_cost ||
                (i != 3 && IS_ALG_ENABLE(OPT_CODE_OPTIMZATION) && (split_cu_cost >= SUBCU_COST_RATE[0][i] * large_cu_cost))) {
                split_cu_cost = MAX_COST; // guide RDO to select large CU
                break;
            }
        }
    }

    /* decide split or not -----------------------------------------
     */
    if (large_cu_cost < split_cu_cost) {
        /* the larger cu is selected */
        cu_copy_stored_parameters(h, p_cu, best);
        h->copy_aec_state_rdo(p_aec, &p_layer->cs_cu);
        split_cu_cost = large_cu_cost;
    }

    return split_cu_cost;
}

/* ---------------------------------------------------------------------------
 */
rdcost_t compress_ctu_inter(xavs2_t *h, aec_t *p_aec, cu_t *p_cu, int i_level, int i_min_level, int i_max_level, rdcost_t cost_limit)
{
    aec_t cs_aec;
    cu_layer_t *p_layer      = cu_get_layer(h, i_level);
    cu_info_t *best          = &p_layer->cu_best;
    rdcost_t large_cu_cost   = MAX_COST;
    rdcost_t split_cu_cost   = MAX_COST;
    rdcost_t split_flag_cost = 0;
    uint32_t avail_modes     = cu_get_valid_modes(h, h->i_type, i_level);
    int b_inside_pic         = (p_cu->i_pix_x + p_cu->i_size <= h->i_width) && (p_cu->i_pix_y + p_cu->i_size <= h->i_height);
    int b_split_ctu          = (i_level > i_min_level || !b_inside_pic);
    int b_check_large_cu     = (b_inside_pic && i_level <= i_max_level);

    /* init current CU ---------------------------------------------
     */
    cu_init(h, p_cu, best, i_level);

    /* coding current CU -------------------------------------------
     */
    if (b_check_large_cu) {
        h->copy_aec_state_rdo(&cs_aec, p_aec);
        if (i_level > MIN_CU_SIZE_IN_BIT) {
            split_flag_cost = h->f_lambda_mode * p_aec->binary.write_ctu_split_flag(&cs_aec, 0, i_level);
        }

        large_cu_cost = compress_cu_inter(h, &cs_aec, p_cu, best, avail_modes, large_cu_cost, cost_limit);
        large_cu_cost += split_flag_cost;

        if (IS_ALG_ENABLE(OPT_ET_HOMO_MV) && i_level > i_min_level) {
            b_split_ctu &= !is_ET_inter_recur(h, p_cu, best);
        }

        /* 当前CU和上一层CU的最优模式均为SKIP模式，则跳过下层CU的划分 @张玉槐 */
        if (IS_ALG_ENABLE(OPT_CU_CSET) &&
            ((p_cu->i_size <= 16 && h->i_type == SLICE_TYPE_B) || (p_cu->i_size <= 32 && h->fdec->rps.referd_by_others == 0))) {
            cu_layer_t *p_ulayer = cu_get_layer(h, i_level + 1);
            cu_info_t *curr_ubest = &p_ulayer->cu_best;
            if (IS_SKIP_MODE(curr_ubest->i_mode) && IS_SKIP_MODE(best->i_mode)) {
                b_split_ctu = 0;
            }
        }

        /* QSFD, skip smaller CU partitions */
        if (IS_ALG_ENABLE(OPT_CU_QSFD)) {
            if (p_cu->cu_info.i_level != 3 && large_cu_cost < h->thres_qsfd_cu[0][p_cu->cu_info.i_level - 3]) {
                b_split_ctu = FALSE;
            }
        }

        if (IS_ALG_ENABLE(OPT_ECU) && i_level > i_min_level) {
            // int i_level_left = p_cu->p_left_cu ? p_cu->p_left_cu->i_level : MAX_CU_SIZE_IN_BIT;
            // int i_level_top  = p_cu->p_topA_cu ? p_cu->p_topA_cu->i_level : MAX_CU_SIZE_IN_BIT;

            // b_split_ctu &= !(i_level_left >= i_level && i_level_top >= i_level && (best->i_mode == PRED_SKIP));
            b_split_ctu &= !((best->i_mode == PRED_SKIP) && (best->i_cbp == 0) && p_cu->is_zero_block);
        }
    }


    /* coding 4 sub-CUs --------------------------------------------
     */
    if (b_split_ctu) {
        int i;
        split_cu_cost = 0;

        // cal split cost
        if (b_inside_pic) {
            split_cu_cost += h->f_lambda_mode * p_aec->binary.write_ctu_split_flag(p_aec, 1, i_level);
        }

        for (i = 0; i < 4; i++) {
            cu_t *p_sub_cu = p_cu->sub_cu[i];


            if (p_sub_cu->i_pix_x >= h->i_width || p_sub_cu->i_pix_y >= h->i_height) {
                continue;       // current sub CU is outside the frame
            }

            split_cu_cost += compress_ctu_inter(h, p_aec, p_sub_cu, i_level - 1, i_min_level, i_max_level, large_cu_cost - split_cu_cost);

            if (split_cu_cost > large_cu_cost ||
                (i != 3 && IS_ALG_ENABLE(OPT_CODE_OPTIMZATION) && (split_cu_cost >= SUBCU_COST_RATE[1][i] * large_cu_cost))) {
                split_cu_cost = MAX_COST; // guide RDO to select large CU
                break;
            }
        }
    }
    if (IS_ALG_ENABLE(OPT_SUBCU_SPLIT)) {
        if ((p_cu->sub_cu[0] != NULL) && (p_cu->sub_cu[1] != NULL) && (p_cu->sub_cu[2] != NULL) && (p_cu->sub_cu[3] != NULL)) {
            if (((p_cu->sub_cu[0]->is_ctu_split + p_cu->sub_cu[1]->is_ctu_split + p_cu->sub_cu[2]->is_ctu_split + p_cu->sub_cu[3]->is_ctu_split) >= 3)) {
                b_check_large_cu = FALSE;   // 1080p 20% 节省，约1.7%损失，preset 6，1080p
            }
            /* else if (((!p_cu->sub_cu[0]->is_ctu_split) && ((p_cu->sub_cu[0]->cu_info.i_mode == PRED_SKIP || p_cu->sub_cu[0]->cu_info.i_mode == PRED_2Nx2N) && (p_cu->sub_cu[0]->cu_info.i_cbp == 0)))
            && ((!p_cu->sub_cu[1]->is_ctu_split) && ((p_cu->sub_cu[1]->cu_info.i_mode == PRED_SKIP || p_cu->sub_cu[1]->cu_info.i_mode == PRED_2Nx2N) && (p_cu->sub_cu[1]->cu_info.i_cbp == 0)))
            && ((!p_cu->sub_cu[2]->is_ctu_split) && ((p_cu->sub_cu[2]->cu_info.i_mode == PRED_SKIP || p_cu->sub_cu[2]->cu_info.i_mode == PRED_2Nx2N) && (p_cu->sub_cu[2]->cu_info.i_cbp == 0)))
            && ((!p_cu->sub_cu[3]->is_ctu_split) && ((p_cu->sub_cu[3]->cu_info.i_mode == PRED_SKIP || p_cu->sub_cu[3]->cu_info.i_mode == PRED_2Nx2N) && (p_cu->sub_cu[3]->cu_info.i_cbp == 0)))) {
            avail_modes &= (1<<PRED_2Nx2N);
            }*/
        }
    }

    /* decide split or not -----------------------------------------
     */
    if (large_cu_cost < split_cu_cost) {
        /* the larger cu is selected */
        cu_copy_stored_parameters(h, p_cu, best);
        h->copy_aec_state_rdo(p_aec, &p_layer->cs_cu);
        split_cu_cost = large_cu_cost;
        p_cu->is_ctu_split = FALSE;
    } else {
        p_cu->is_ctu_split = TRUE;
    }
    p_cu->rdcost = split_cu_cost;

    return split_cu_cost;
}

/* ---------------------------------------------------------------------------
 */
void xavs2_rdo_init(uint32_t cpuid, intrinsic_func_t *pf)
{
    UNUSED_PARAMETER(cpuid);
    pf->compress_ctu[SLICE_TYPE_I] = compress_ctu_intra;
    pf->compress_ctu[SLICE_TYPE_P] = compress_ctu_inter;
    pf->compress_ctu[SLICE_TYPE_F] = compress_ctu_inter;
    pf->compress_ctu[SLICE_TYPE_B] = compress_ctu_inter;

    pf->get_skip_mv_predictors[SLICE_TYPE_I] = NULL;
    pf->get_skip_mv_predictors[SLICE_TYPE_P] = get_mv_predictors_pskip;
    pf->get_skip_mv_predictors[SLICE_TYPE_F] = get_mv_predictors_pskip;
    pf->get_skip_mv_predictors[SLICE_TYPE_B] = get_mv_predictors_bskip;
}
