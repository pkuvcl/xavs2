/*
 * slice.c
 *
 * Description of this file:
 *    Slice Processing functions definition of the xavs2 library
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
#include "mc.h"
#include "aec.h"
#include "nal.h"
#include "wrapper.h"
#include "slice.h"
#include "header.h"
#include "bitstream.h"
#include "cudata.h"
#include "rdo.h"
#include "tdrdo.h"
#include "wrapper.h"
#include "frame.h"
#include "alf.h"
#include "sao.h"

/**
 * ===========================================================================
 * global variables
 * ===========================================================================
 */
slice_row_index_t g_slice_lcu_row_order[1024];


#if XAVS2_TRACE
extern int g_sym_count;         /* global symbol count for trace */
extern int g_bit_count;         /* global bit    count for trace */
#endif


/* ---------------------------------------------------------------------------
 * 初始化LCU行的编码顺序
 */
void slice_lcu_row_order_init(xavs2_t *h)
{
    slice_row_index_t *lcurow = g_slice_lcu_row_order;
    int num_lcu_row = h->i_height_in_lcu;
    int idx_slice = 0;
    int i;

    if (h->param->i_lcurow_threads > 1 && h->param->slice_num > 1) {
        int slice_num = h->param->slice_num;
        int set_new_lcu_row = 1;
        int k;

        /* set task table. the order of encoding task priority:
         * 1) first LCU row in each slice;
         * 2) other LCU rows. */
        for (i = 0, idx_slice = 0; idx_slice < slice_num; idx_slice++) {
            lcurow[i].lcu_y     = (int16_t)(h->slices[idx_slice]->i_first_lcu_y);
            lcurow[i].row_type  = 0;
            lcurow[i].slice_idx = (int8_t)idx_slice;
            i++;
        }

        for (k = 0; set_new_lcu_row; k++) {
            set_new_lcu_row = 0;
            for (idx_slice = 0; idx_slice < slice_num && i < num_lcu_row; idx_slice++) {
                slice_t *p_slice = h->slices[idx_slice];
                int bottom_row_y = p_slice->i_first_lcu_y + p_slice->i_lcu_row_num - 1;
                int new_row = lcurow[k * slice_num + idx_slice].lcu_y + 1;  /* next LCU row in same slice */
                if (new_row > p_slice->i_first_lcu_y && new_row <= bottom_row_y) {
                    lcurow[i].lcu_y     = (int16_t)(new_row);
                    lcurow[i].row_type  = 1 + (new_row == bottom_row_y);
                    lcurow[i].slice_idx = (int8_t)(idx_slice);
                    set_new_lcu_row = 1;    /* set a new LCU row */
                    i++;
                }
            }
        }
    } else {
        slice_t *p_slice = h->slices[idx_slice];

        for (i = 0; i < num_lcu_row; i++) {
            int row_type = (i != p_slice->i_first_lcu_y) + (i == p_slice->i_first_lcu_y + p_slice->i_lcu_row_num - 1);

            lcurow[i].lcu_y     = (int16_t)(i);
            lcurow[i].row_type  = (int8_t)row_type;
            lcurow[i].slice_idx = (int8_t)idx_slice;

            if (row_type == 2) {
                idx_slice++;                       /* a new slice appear */
                p_slice = h->slices[idx_slice];
            }
        }
    }   // 默认行级顺序
}

/* ---------------------------------------------------------------------------
 * initializes the parameters for all slices
 */
void xavs2_slices_init(xavs2_t *h)
{
    slice_t *p_slice;

    if (h->param->slice_num < 2) {
        /* single slice per frame */
        p_slice = h->slices[0];

        /* set slice properties */
        p_slice->i_first_lcu_xy = 0;
        p_slice->i_last_lcu_xy  = h->i_height_in_lcu * h->i_width_in_lcu - 1;
        p_slice->i_first_scu_y  = 0;
        p_slice->i_first_lcu_y  = 0;
        p_slice->i_lcu_row_num  = h->i_height_in_lcu;
        p_slice->i_last_lcu_y   = p_slice->i_first_lcu_y + p_slice->i_lcu_row_num - 1;
        p_slice->p_slice_bs_buf = h->p_bs_buf_slice;
        p_slice->len_slice_bs_buf = h->i_bs_buf_slice;
    } else {
        /* multi-slice per frame */
        uint8_t *p_bs_start   = h->p_bs_buf_slice;
        const int i_slice_num = h->param->slice_num;
        int i_rest_rows       = h->i_height_in_lcu;
        int i_len_per_row     = (h->i_bs_buf_slice - i_slice_num * CACHE_LINE_256B) / i_rest_rows;
        int i_first_row_id    = 0;
        int i_left_slice_num  = i_slice_num;
        int i_scus_in_lcu     = 1 << (h->i_lcu_level - MIN_CU_SIZE_IN_BIT);
        int i_avg_rows;
        int i_bs_len;
        int i;

        /* set properties for each slice */
        for (i = 0; i < i_slice_num; i++) {
            p_slice = h->slices[i];

            /* compute lcu-row number in a slice */
            i_avg_rows = (i_rest_rows + i_left_slice_num - 1) / i_left_slice_num;
            i_rest_rows -= i_avg_rows;/* left lcu rows */
            i_left_slice_num--;       /* left slice number */

            /* set slice properties */
            p_slice->i_first_lcu_xy = i_first_row_id * h->i_width_in_lcu;
            p_slice->i_first_scu_y  = i_first_row_id * i_scus_in_lcu;
            p_slice->i_first_lcu_y  = i_first_row_id;
            p_slice->i_lcu_row_num  = i_avg_rows;
            p_slice->i_last_lcu_xy  = p_slice->i_first_lcu_xy + (p_slice->i_lcu_row_num * h->i_width_in_lcu - 1);
            p_slice->i_last_lcu_y   = p_slice->i_first_lcu_y + p_slice->i_lcu_row_num - 1;

            /* init slice bs, start at align 128-byte */
            ALIGN_256_PTR(p_bs_start);/* align 256B */
            i_bs_len = i_len_per_row * p_slice->i_lcu_row_num;
            i_bs_len = (i_bs_len >> 8) << 8;   /* the length is a multiple of 256 */
            p_slice->p_slice_bs_buf   = p_bs_start;
            p_slice->len_slice_bs_buf = i_bs_len;
            p_bs_start += i_bs_len;

            /* update row id for next slice */
            i_first_row_id += i_avg_rows;
            assert(i_first_row_id <= h->i_height_in_lcu);
        }
    }
}

/* ---------------------------------------------------------------------------
 * estimate CU depth range
 */
//#if OPT_CU_DEPTH_CTRL
static void est_cu_depth_range(xavs2_t *h, int *min_level, int *max_level)
{
    static const int L_WEIGHT[] = {3,2,0,1,5};   // [Left Top TopLeft TopRight Col]
    static const int TH_WEIGHT[3] = {25, 15, 5};
    int b_left_cu = h->lcu.i_pix_x > 0 && (cu_get_slice_index(h, h->lcu.i_scu_x, h->lcu.i_scu_y) == cu_get_slice_index(h, h->lcu.i_scu_x - 1, h->lcu.i_scu_y));
    int b_top_cu  = h->lcu.i_pix_y > 0 && (cu_get_slice_index(h, h->lcu.i_scu_x, h->lcu.i_scu_y) == cu_get_slice_index(h, h->lcu.i_scu_x, h->lcu.i_scu_y - 1));
#if SAVE_CU_INFO
    int b_col_cu  = (h->i_type != SLICE_TYPE_I) && (h->fref[0]->cu_mode[h->lcu.i_scu_xy] < PRED_I_2Nx2N);
#else
    int b_col_cu  = FALSE;
#endif
    int min_level_ctrl = h->i_scu_level;
    int max_level_ctrl = h->i_lcu_level;
    int min_level_pred = h->i_lcu_level - 3;
    int max_level_pred = h->i_lcu_level - 0;
    int min_left_level = h->i_lcu_level;
    int min_top_level  = h->i_lcu_level;
    int i = 0;
    int cu_with_of_lcu = 1 << (h->i_lcu_level - MIN_CU_SIZE_IN_BIT);

    if (b_left_cu && b_top_cu) {
        // check left CTU's max depth
        int i_left_cu_y = h->lcu.i_scu_y;
        int i_top_cu_x = h->lcu.i_scu_x;
        cu_info_t *p_left = &h->cu_info[h->lcu.i_scu_xy - 1];
        cu_info_t *p_top  = &h->cu_info[h->lcu.i_scu_xy - h->i_width_in_mincu];
        for (i = cu_with_of_lcu; i != 0; i--) {
            if (i_left_cu_y++ < h->i_height_in_mincu) {
                min_left_level = XAVS2_MIN(min_left_level, p_left->i_level);
                p_left += h->i_width_in_mincu;
            }

            if (i_top_cu_x++ < h->i_width_in_mincu) {
                min_top_level = XAVS2_MIN(min_top_level, p_top->i_level);
                p_top++;
            }
        }
        min_left_level = (min_left_level >= h->i_lcu_level - 1);
        min_top_level  = (min_top_level  >= h->i_lcu_level - 1);
        if (min_left_level && min_top_level) {
            min_level_pred = h->i_lcu_level - 2;
            max_level_pred = h->i_lcu_level - 0;  // depth range limited to [0, 1, 2]
        } else if (!min_left_level && !min_top_level) {
            min_level_pred = h->i_lcu_level - 3;
            max_level_pred = h->i_lcu_level - 1;  // depth range limited to [1, 2, 3]
        }
    }

    min_level_pred = XAVS2_MAX(min_level_pred, h->i_scu_level);

    if (b_left_cu && b_top_cu && b_col_cu) {
#if SAVE_CU_INFO
        int level_T  = h->i_lcu_level - h->cu_info[h->lcu.i_scu_xy - h->i_width_in_mincu].i_level;       // top
        int level_L  = h->i_lcu_level - h->cu_info[h->lcu.i_scu_xy - 1].i_level;                         // left
        int level_TL = h->i_lcu_level - h->cu_info[h->lcu.i_scu_xy - 1 - h->i_width_in_mincu].i_level;   // top-left
        int level_TR = h->i_lcu_level - h->cu_info[h->lcu.i_scu_xy + 1 - h->i_width_in_mincu].i_level;   // top-right
        int level_C  = h->i_lcu_level - h->fref[0]->cu_level[h->lcu.i_scu_xy];                           // col-located
        int weight = L_WEIGHT[0] * level_L + L_WEIGHT[1] * level_T+L_WEIGHT[2] * level_TL + L_WEIGHT[3] * level_TR+L_WEIGHT[4] * level_C;

        if (weight >= TH_WEIGHT[0]) {
            min_level_ctrl = -3;
            max_level_ctrl = -2;
        } else if (weight >= TH_WEIGHT[1]) {
            min_level_ctrl = -3;
            max_level_ctrl = -1;
        } else if (weight >= TH_WEIGHT[2]) {
            min_level_ctrl = -2;
            max_level_ctrl = 0;
        } else {
            min_level_ctrl = -1;
            max_level_ctrl = 0;
        }
        min_level_ctrl = XAVS2_MAX(h->i_scu_level, min_level_ctrl + h->i_lcu_level);
        max_level_ctrl = max_level_ctrl + h->i_lcu_level;
#endif
    } else {
        min_level_ctrl = h->i_scu_level;
        max_level_ctrl = h->i_lcu_level;
    }

    *min_level = XAVS2_MAX(min_level_ctrl, min_level_pred);
    *max_level = XAVS2_MIN(max_level_ctrl, max_level_pred);
    assert(*min_level <= *max_level);
}
//#endif

/* ---------------------------------------------------------------------------
 * store cu info for one LCU row
 */
static void store_cu_info_row(row_info_t *row)
{
    int i, j, k, l;
    xavs2_t *h = row->h;

    int lcu_height_in_scu = 1 << (h->i_lcu_level - MIN_CU_SIZE_IN_BIT);
    int last_lcu_row = ((h->lcu.i_scu_y + lcu_height_in_scu) < h->i_height_in_mincu ? 0 : 1);
    int num_scu_y = last_lcu_row == 0 ? lcu_height_in_scu : h->i_height_in_mincu - h->lcu.i_scu_y;

#if SAVE_CU_INFO
    /* store cu info (one lcu row) of reference frame */
    for (i = 0; i < num_scu_y; i++) {
        int scu_offset = (h->lcu.i_scu_y + i) * h->i_width_in_mincu;
        cu_info_t *p_cu_info = &h->cu_info[scu_offset];

        for (j = 0; j < h->i_width_in_mincu; j++) {
            h->fdec->cu_level[scu_offset + j] = (int8_t)p_cu_info->i_level;
            h->fdec->cu_mode[scu_offset + j] = (int8_t)p_cu_info->i_mode;
            h->fdec->cu_cbp[scu_offset + j] = (int8_t)p_cu_info->i_cbp;

            p_cu_info++;
        }
    }
#endif

    if (h->i_type != SLICE_TYPE_I) {
        // store motion information for temporal prediction
        const int w0_in_16x16 = h->i_width_in_minpu >> 2;
        const int h0_in_16x16 = h->i_height_in_minpu >> 2;
        const int w_in_16x16 = (h->i_width_in_minpu + 3) >> 2;
        const int h_in_16x16 = (h->i_height_in_minpu + 3) >> 2;

        const int w_in_4x4 = h->i_width_in_minpu;
        const int h_in_4x4 = h->i_height_in_minpu;

        int start_16x16_y = h->lcu.i_scu_y >> 1;
        int num_16x16_y   = num_scu_y >> 1;

        const mv_t   *src_mv  = h->fwd_1st_mv;
        const int8_t *src_ref = h->fwd_1st_ref;

        mv_t   *dst_mv  = h->fdec->pu_mv;
        int8_t *dst_ref = h->fdec->pu_ref;

        assert(num_16x16_y >= 1 || last_lcu_row);

        // store middle pixel's motion information
        for (i = start_16x16_y; i < start_16x16_y + num_16x16_y; i++) {
            k = ((i << 2) + 2) * w_in_4x4;
            for (j = 0; j < w0_in_16x16; j++) {
                l = (j << 2) + 2;
                dst_mv[i * w_in_16x16 + j]  = src_mv[k + l];
                dst_ref[i * w_in_16x16 + j] = src_ref[k + l];
            }
        }

        ///! last LCU row
        if (last_lcu_row && (h0_in_16x16 < h_in_16x16)) {
            k = (((h0_in_16x16 << 2) + h_in_4x4) >> 1) * w_in_4x4;
            for (j = 0; j < w0_in_16x16; j++) {
                l = (j << 2) + 2;
                dst_mv[h0_in_16x16 * w_in_16x16 + j]  = src_mv[k + l];
                dst_ref[h0_in_16x16 * w_in_16x16 + j] = src_ref[k + l];
            }

            if (w0_in_16x16 < w_in_16x16) {
                l = ((w0_in_16x16 << 2) + w_in_4x4) >> 1;
                dst_mv[h0_in_16x16 * w_in_16x16 + w0_in_16x16]  = src_mv[k + l];
                dst_ref[h0_in_16x16 * w_in_16x16 + w0_in_16x16] = src_ref[k + l];
            }
        }

        ///! last column
        if (w0_in_16x16 < w_in_16x16) {
            i = ((w0_in_16x16 << 2) + w_in_4x4) >> 1;

            for (j = start_16x16_y; j < start_16x16_y + num_16x16_y; j++) {
                dst_mv[j * w_in_16x16 + w0_in_16x16] = src_mv[(j * 4 + 2) * w_in_4x4 + i];
                dst_ref[j * w_in_16x16 + w0_in_16x16] = src_ref[(j * 4 + 2) * w_in_4x4 + i];
            }
        }
    }
}



/* ---------------------------------------------------------------------------
 * encode one lcu row
 */
void *xavs2_lcu_row_write(void *arg)
{
    row_info_t  *row      = (row_info_t *)arg;
    xavs2_t     *h        = row->h;
    slice_t     *slice    = h->slices[h->i_slice_index];
    aec_t       *p_aec    = &h->aec;
    const int    i_lcu_y  = row->row;
    row_info_t  *last_row = (i_lcu_y > slice->i_first_lcu_y) ? &h->frameinfo->rows[i_lcu_y - 1] : 0;
    lcu_analyse_t lcu_analyse = g_funcs.compress_ctu[h->i_type];
    const bool_t b_enable_wpp = h->param->i_lcurow_threads > 1;
    int min_level = h->i_scu_level;
    int max_level = h->i_lcu_level;
    int i_lcu_x;
#if ENABLE_RATE_CONTROL_CU
    int temp_dquant;
#endif

    h->lcu.get_skip_mvs = g_funcs.get_skip_mv_predictors[h->i_type];
    if (h->param->slice_num > 1) {
        slice_init_bufer(h, slice);
    }

    /* loop over all LCUs in current lcu row ------------------------
     */
    for (i_lcu_x = 0; i_lcu_x < h->i_width_in_lcu; i_lcu_x++) {
        /* 0, initialization before sync */
        lcu_info_t  *lcu = &row->lcus[i_lcu_x];
        lcu_start_init_pos(h, i_lcu_x, i_lcu_y);

        lcu->slice_index = h->i_slice_index;
        lcu->scu_xy      = h->lcu.i_scu_xy;
        lcu->pix_x       = h->lcu.i_pix_x;
        lcu->pix_y       = h->lcu.i_pix_y;

        h->lcu.lcu_coeff[0] = lcu->coeffs_y;
        h->lcu.lcu_coeff[1] = lcu->coeffs_uv[0];
        h->lcu.lcu_coeff[2] = lcu->coeffs_uv[1];
#if ENABLE_RATE_CONTROL_CU
        h->last_dquant     = &lcu->last_dqp;
#endif

        /* 1, sync */
        wait_lcu_row_coded(last_row, XAVS2_MIN(h->i_width_in_lcu - 1, i_lcu_x + 1));

        if (b_enable_wpp && last_row != NULL && i_lcu_x == 0) {
            aec_copy_aec_state(p_aec, &last_row->aec_set);
        }

        /* 3, start */
        lcu_start_init_pixels(h, i_lcu_x, i_lcu_y);

        if (h->td_rdo != NULL) {
            tdrdo_lcu_adjust_lambda(h, &h->f_lambda_mode);
        }

#if ENABLE_RATE_CONTROL_CU
        temp_dquant = *h->last_dquant;
#endif

        /* 4, analyze */
        if (IS_ALG_ENABLE(OPT_CU_DEPTH_CTRL)) {
            est_cu_depth_range(h, &min_level, &max_level);
        }

        lcu_analyse(h, p_aec, h->lcu.p_ctu, h->i_lcu_level, min_level, max_level, MAX_COST);

        if (h->td_rdo != NULL) {
            tdrdo_lcu_update(h);
        }

#if ENABLE_RATE_CONTROL_CU
        *h->last_dquant = temp_dquant;
#endif

        /* 5, lcu end */
        lcu_end(h, i_lcu_x, i_lcu_y);

        if (b_enable_wpp && i_lcu_x == 1) {
            /* backup aec contexts for the next row */
            aec_copy_aec_state(&row->aec_set, p_aec);
        }

        /* 4, deblock on lcu */
#if XAVS2_DUMP_REC
        if (!h->param->loop_filter_disable) {
            xavs2_lcu_deblock(h, h->fdec);
        }
#else
        /* no need to do loop-filter without dumping, but at this time,
         * the PSNR is computed not correctly if XAVS2_STAT is on. */
        if (!h->param->loop_filter_disable && h->fdec->rps.referd_by_others) {
            xavs2_lcu_deblock(h, h->fdec);
        }
#endif

        /* copy reconstruction pixels when the last LCU is reconstructed */
        if (h->param->enable_sao) {
            if (i_lcu_x > 0) {
                sao_get_lcu_param_after_deblock(h, p_aec, i_lcu_x - 1, i_lcu_y);
                sao_filter_lcu(h, h->sao_blk_params[i_lcu_y * h->i_width_in_lcu + i_lcu_x - 1], i_lcu_x - 1, i_lcu_y);
            }
            if (i_lcu_x == h->i_width_in_lcu - 1) {
                sao_get_lcu_param_after_deblock(h, p_aec, i_lcu_x, i_lcu_y);
                sao_filter_lcu(h, h->sao_blk_params[i_lcu_y * h->i_width_in_lcu + i_lcu_x], i_lcu_x, i_lcu_y);
            }
        }

        xavs2_thread_mutex_lock(&row->mutex);    /* lock */
        row->coded = i_lcu_x;
        // h->fdec->num_lcu_coded_in_row[row->row]++;
        xavs2_thread_mutex_unlock(&row->mutex);  /* unlock */

        /* signal to the next row */
        if (i_lcu_x >= 1) {
            xavs2_thread_cond_signal(&row->cond);
        }
    }

    /* post-processing for current lcu row -------------------------
     */
    if (h->param->enable_sao && (h->slice_sao_on[0] || h->slice_sao_on[1] || h->slice_sao_on[2])) {
        int sao_off_num_y = 0;
        int sao_off_num_u = 0;
        int sao_off_num_v = 0;
        int idx_lcu = i_lcu_y * h->i_width_in_lcu;
        for (i_lcu_x = 0; i_lcu_x < h->i_width_in_lcu; i_lcu_x++, idx_lcu++) {
            if (h->sao_blk_params[idx_lcu][0].typeIdc == SAO_TYPE_OFF) {
                sao_off_num_y++;
            }
            if (h->sao_blk_params[idx_lcu][1].typeIdc == SAO_TYPE_OFF) {
                sao_off_num_u++;
            }
            if (h->sao_blk_params[idx_lcu][2].typeIdc == SAO_TYPE_OFF) {
                sao_off_num_v++;
            }
        }
        h->num_sao_lcu_off[i_lcu_y][0] = sao_off_num_y;
        h->num_sao_lcu_off[i_lcu_y][1] = sao_off_num_u;
        h->num_sao_lcu_off[i_lcu_y][2] = sao_off_num_v;
    } else {
        int num_lcu = h->i_width_in_lcu;
        h->num_sao_lcu_off[i_lcu_y][0] = num_lcu;
        h->num_sao_lcu_off[i_lcu_y][1] = num_lcu;
        h->num_sao_lcu_off[i_lcu_y][2] = num_lcu;
    }

    if (h->param->enable_alf && (h->pic_alf_on[0] || h->pic_alf_on[1] || h->pic_alf_on[2])) {
        if (h->i_type == SLICE_TYPE_B && IS_ALG_ENABLE(OPT_FAST_ALF)) {
            i_lcu_x = ((i_lcu_y + h->fenc->i_frm_coi) & 1);
            for (; i_lcu_x < h->i_width_in_lcu; i_lcu_x += 2) {
                alf_get_statistics_lcu(h, i_lcu_x, i_lcu_y, h->fenc, h->fdec);
            }
        } else {
            for (i_lcu_x = 0; i_lcu_x < h->i_width_in_lcu; i_lcu_x++) {
                alf_get_statistics_lcu(h, i_lcu_x, i_lcu_y, h->fenc, h->fdec);
            }
        }
    }

    /* reference frame */
    if (h->fdec->rps.referd_by_others) {
        /* store cu info */
        store_cu_info_row(row);

        /* expand border */
        xavs2_frame_expand_border_lcurow(h, h->fdec, i_lcu_y);

        /* interpolate (after finished expanding border) */
#if ENABLE_FRAME_SUBPEL_INTPL
        if (h->use_fractional_me != 0) {
            interpolate_lcu_row(h, h->fdec, i_lcu_y);
        }
#endif

        if (last_row) {
            /* make sure the top row have finished interpolation and padding */
            xavs2_frame_t *fdec = h->fdec;

            xavs2_thread_mutex_lock(&fdec->mutex);   /* lock */
            while (fdec->num_lcu_coded_in_row[last_row->row] < h->i_width_in_lcu) {
                xavs2_thread_cond_wait(&fdec->cond, &fdec->mutex);
            }
            xavs2_thread_mutex_unlock(&fdec->mutex); /* unlock */
        }
    }

    /* release task */
    xavs2e_release_row_task(row);

    return 0;
}

/* ---------------------------------------------------------------------------
 * start encodes one slice
 */
void xavs2_slice_write_start(xavs2_t *h)
{
    aec_t   *p_aec = &h->aec;
    slice_t *slice = h->slices[h->i_slice_index];

    /* init slice */
#if ENABLE_RATE_CONTROL_CU
    h->frameinfo->rows[slice->i_first_lcu_y].lcus[0].last_dqp = 0;
#endif
    slice->i_qp = h->i_qp;

    /* init bs_t, reserve space to store the length of bitstream */
    xavs2_bs_init(&slice->bs, slice->p_slice_bs_buf, slice->len_slice_bs_buf);

    sao_slice_onoff_decision(h, h->slice_sao_on);

    /* write slice header */
    xavs2_slice_header_write(h, slice);
    bs_byte_align(&slice->bs);

    /* init AEC */
    aec_start(h, p_aec, slice->bs.p_start + PSEUDO_CODE_SIZE, slice->bs.p_end, 0);

    /* init slice buffers */
    slice_init_bufer(h, slice);

    /* prediction mode is set to -1 outside the frame,
     * indicating that no prediction can be made from this part */
    {
        int ip_stride = h->i_width_in_minpu + 16;
        int lcu_height_in_pu = ((1 << h->i_lcu_level) >> MIN_PU_SIZE_IN_BIT);
        g_funcs.fast_memset((h->ipredmode - ip_stride - 16), -1, (lcu_height_in_pu + 1) * ip_stride * sizeof(int8_t));
    }
}
