/*
 * slice.h
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


#ifndef XAVS2_SLICE_H
#define XAVS2_SLICE_H

/**
 * ===========================================================================
 * structures
 * ===========================================================================
 */
typedef struct slice_row_index_t {
    int16_t lcu_y;       /* 行编号 */
    int8_t  slice_idx;   /* 行所在的Slice索引号 */
    int8_t  row_type;    /* 0: Slice开始位置的行；1:普通；2: Slice结束位置的行 */
} slice_row_index_t;

extern slice_row_index_t g_slice_lcu_row_order[1024];

/* ---------------------------------------------------------------------------
 * 初始化Slice级的buffer指针
 */
static ALWAYS_INLINE
void slice_init_bufer(xavs2_t *h, slice_t *slice)
{
    /* init slice buffers */
    h->ipredmode         = slice->slice_ipredmode;
    h->intra_border[0]   = slice->slice_intra_border[0];
    h->intra_border[1]   = slice->slice_intra_border[1];
    h->intra_border[2]   = slice->slice_intra_border[2];
    h->p_deblock_flag[0] = slice->slice_deblock_flag[0];
    h->p_deblock_flag[1] = slice->slice_deblock_flag[1];
}


/* ---------------------------------------------------------------------------
 * 等待一行LCU编码完指定数量的LCU
 */
static ALWAYS_INLINE
void wait_lcu_row_coded(row_info_t *last_row, int wait_lcu_coded)
{
    if (last_row != NULL && last_row->coded < wait_lcu_coded) {
        xavs2_thread_mutex_lock(&last_row->mutex);   /* lock */
        while (last_row->coded < wait_lcu_coded) {
            xavs2_thread_cond_wait(&last_row->cond, &last_row->mutex);
        }
        xavs2_thread_mutex_unlock(&last_row->mutex); /* unlock */
    }
}


/* ---------------------------------------------------------------------------
 * 查询一行LCU是否已编码完毕
 */
static ALWAYS_INLINE
int is_lcu_row_finished(xavs2_t *h, xavs2_frame_t *frm, int lcu_row)
{
    return (frm->num_lcu_coded_in_row[lcu_row] > h->i_width_in_lcu);
}

/* ---------------------------------------------------------------------------
 * 查询一行LCU是否已编码完毕
 */
static ALWAYS_INLINE
void set_lcu_row_finished(xavs2_t *h, xavs2_frame_t *frm, int lcu_row)
{
    frm->num_lcu_coded_in_row[lcu_row] = h->i_width_in_lcu + 1;
}


/* ---------------------------------------------------------------------------
 * release a row task
 */
static INLINE
void xavs2e_release_row_task(row_info_t *row)
{
    if (row) {
        xavs2_t         *h     = row->h;
        xavs2_frame_t   *fdec  = h->fdec;
        xavs2_handler_t *h_mgr = h->h_top;
        int b_slice_boundary_done = FALSE;

        /* 如果此时Slice边界的相邻行已处理完，则直接进行插值，不需要加锁
         * 否则，需要加锁后进行处理，避免出现问题 */
        if (h->param->b_cross_slice_loop_filter == FALSE) {
            if (row->b_top_slice_border && row->row > 0) {
                if (is_lcu_row_finished(h, fdec, row->row - 1)) {
                    int y_start = (row->row << h->i_lcu_level) - 4;
                    interpolate_sample_rows(h, h->fdec, y_start, 8, 0, 0);
                    b_slice_boundary_done = TRUE;
                }
            } else if (row->b_down_slice_border && row->row < h->i_height_in_lcu - 1) {
                if (is_lcu_row_finished(h, fdec, row->row + 1)) {
                    int y_start = ((row->row + 1) << h->i_lcu_level) - 4;
                    interpolate_sample_rows(h, h->fdec, y_start, 8, 0, 0);
                    b_slice_boundary_done = TRUE;
                }
            }
        } else {
            /* TODO: 多Slice并行时，对Slice边界的处理 */
            if (h->param->slice_num > 1) {
                xavs2_log(NULL, XAVS2_LOG_ERROR, "CrossSliceLoopFilter not supported now!\n");
                assert(0);
            }
        }

        xavs2_thread_mutex_lock(&fdec->mutex);           /* lock */
        if (h->param->b_cross_slice_loop_filter == FALSE) {
            if (b_slice_boundary_done == FALSE && row->b_top_slice_border && row->row > 0) {
                if (is_lcu_row_finished(h, fdec, row->row - 1)) {
                    int y_start = (row->row << h->i_lcu_level) - 4;
                    interpolate_sample_rows(h, h->fdec, y_start, 8, 0, 0);
                    // xavs2_log(NULL, XAVS2_LOG_DEBUG, "Intp2 POC [%3d], Slice %2d, Row %2d, [%3d, %3d)\n",
                    //           h->fenc->i_frame, h->i_slice_index, row->row, y_start, y_start + 8);
                }
            } else if (b_slice_boundary_done == FALSE && row->b_down_slice_border && row->row < h->i_height_in_lcu - 1) {
                if (is_lcu_row_finished(h, fdec, row->row + 1)) {
                    int y_start = ((row->row + 1) << h->i_lcu_level) - 4;
                    interpolate_sample_rows(h, h->fdec, y_start, 8, 0, 0);
                    // xavs2_log(NULL, XAVS2_LOG_DEBUG, "Intp3 POC [%3d], Slice %2d, Row %2d, [%3d, %3d)\n",
                    //           h->fenc->i_frame, h->i_slice_index, row->row, y_start, y_start + 8);
                }
            }
        } else {
            /* TODO: 多Slice并行时，对Slice边界的处理 */
        }
        set_lcu_row_finished(h, fdec, row->row);
        xavs2_thread_mutex_unlock(&fdec->mutex);         /* unlock */

        /* broadcast to the aec thread and all waiting contexts */
        xavs2_thread_cond_broadcast(&fdec->cond);

        if (h->task_type == XAVS2_TASK_ROW) {
            xavs2_thread_mutex_lock(&h_mgr->mutex);   /* lock */
            h->task_status = XAVS2_TASK_FREE;
            xavs2_thread_mutex_unlock(&h_mgr->mutex); /* unlock */
            /* signal a free row context available */
            xavs2_thread_cond_signal(&h_mgr->cond[SIG_ROW_CONTEXT_RELEASED]);
        }
    }
}

/* ---------------------------------------------------------------------------
 * sync of frame parallel coding
 */
static ALWAYS_INLINE
void xavs2e_inter_sync(xavs2_t *h, int lcu_y, int lcu_x)
{
    if (h->i_type != SLICE_TYPE_I && h->h_top->i_frm_threads > 1) {
        int num_lcu_delay = ((h->param->search_range + (1 << h->i_lcu_level) - 1) >> h->i_lcu_level) + 1;
        int low_bound  = XAVS2_MAX(lcu_y - num_lcu_delay, 0);
        int up_bound = XAVS2_MIN(lcu_y + num_lcu_delay, h->i_height_in_lcu - 1);
        int col_coded = h->i_width_in_lcu;
        int i, j;

        UNUSED_PARAMETER(lcu_x);

        for (i = 0; i < h->i_ref; i++) {
            xavs2_frame_t *p_ref = h->fref[i];

            for (j = low_bound; j <= up_bound; j++) {
                xavs2_thread_mutex_lock(&p_ref->mutex);    /* lock */
                while (p_ref->num_lcu_coded_in_row[j] < col_coded) {
                    xavs2_thread_cond_wait(&p_ref->cond, &p_ref->mutex);
                }
                xavs2_thread_mutex_unlock(&p_ref->mutex);  /* unlock */
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * get a row encoder handle
 */
static INLINE
xavs2_t *xavs2e_alloc_row_task(xavs2_t *h)
{
    xavs2_handler_t *h_mgr = h->h_top;
    int i;

    assert(h->task_type == XAVS2_TASK_FRAME && h->frameinfo);

    xavs2_thread_mutex_lock(&h_mgr->mutex);   /* lock */

    /* wait until we successfully get one free row context */
    for (; h_mgr->i_exit_flag != XAVS2_EXIT_THREAD;) {
        for (i = 0; i < h_mgr->num_row_contexts; i++) {
            xavs2_t *h_row_coder = &h_mgr->row_contexts[i];

            if (h_row_coder->task_status == XAVS2_TASK_FREE) {
                h_row_coder->task_status = XAVS2_TASK_BUSY;
                h_row_coder->frameinfo = h->frameinfo;   /* duplicate frame info */

                /* sync row contexts */
                memcpy(&h_row_coder->row_vars_1, &h->row_vars_1, (uint8_t *)&h->row_vars_2 - (uint8_t *)&h->row_vars_1);

                /* make the state of the aec engine same as the one when the slice starts */
                /* 这里h->aec的位置不同导致性能不一样，但是在LCU行编码时重新做了同步保证了一致性 */
                aec_copy_aec_state(&h_row_coder->aec, &h->aec);
                /* unlock */
                xavs2_thread_mutex_unlock(&h_mgr->mutex);

                return h_row_coder;
            }
        }

        xavs2_thread_cond_wait(&h_mgr->cond[SIG_ROW_CONTEXT_RELEASED], &h_mgr->mutex);
    }

    /* unlock */
    xavs2_thread_mutex_unlock(&h_mgr->mutex);
    return NULL;
}

#define xavs2_slices_init FPFX(slices_init)
void  xavs2_slices_init(xavs2_t *h);

#define xavs2_slice_write_start FPFX(slice_write_start)
void  xavs2_slice_write_start(xavs2_t *h);

#define xavs2_lcu_row_write FPFX(lcu_row_write)
void *xavs2_lcu_row_write(void *arg);

#define slice_lcu_row_order_init FPFX(slice_lcu_row_order_init)
void  slice_lcu_row_order_init(xavs2_t *h);

#define xavs2e_encode_one_frame FPFX(xavs2e_encode_one_frame)
void *xavs2e_encode_one_frame(void *arg);

#endif  // XAVS2_SLICE_H
