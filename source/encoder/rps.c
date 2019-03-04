/*
 * rps.c
 *
 * Description of this file:
 *    RPS functions definition of the xavs2 library
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

#include "common/common.h"
#include "cudata.h"
#include "wrapper.h"
#include "ratecontrol.h"
#include "rps.h"

/**
 * ===========================================================================
 * local definitions
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static void ALWAYS_INLINE
set_ref_man(xavs2_rps_t *p_refman, int idx, int poc, int qp_offset, int refered_by_others,
            int num_of_ref, int ref_pic[4], int num_to_rm, int rm_pic[4], int temporal_id)
{
    p_refman->idx_in_gop       = idx;
    p_refman->poc              = poc;
    p_refman->qp_offset        = qp_offset;
    p_refman->referd_by_others = refered_by_others;

    p_refman->num_of_ref       = num_of_ref;
    memcpy(p_refman->ref_pic,    ref_pic, XAVS2_MAX_REFS * sizeof(int));

    p_refman->num_to_rm        = num_to_rm;
    memcpy(p_refman->rm_pic,     rm_pic,  XAVS2_MAX_REFS * sizeof(int));

    p_refman->temporal_id      = temporal_id;
}

/* ---------------------------------------------------------------------------
 * low delay configuration for reference management
 */
static void default_reference_management_ldp(xavs2_rps_t *p_refman)
{
    int ref_pic0[4]    = {  1,  5,  9, 13 };
    int ref_pic1[4]    = {  1,  2,  6, 10 };
    int ref_pic2[4]    = {  1,  3,  7, 11 };
    int ref_pic3[4]    = {  1,  4,  8, 12 };
    int remove_pic0[4] = {  2, -1, -1, -1 };
    int remove_pic1[4] = {  4, -1, -1, -1 };
    int remove_pic2[4] = {  2, -1, -1, -1 };
    int remove_pic3[4] = { 12, -1, -1, -1 };

    memset(p_refman, -1, sizeof(xavs2_rps_t) * XAVS2_MAX_GOPS);

    set_ref_man(&p_refman[0], 0, 1, 5, 1, 4, ref_pic0, 1, remove_pic0, -1);
    set_ref_man(&p_refman[1], 1, 2, 4, 1, 4, ref_pic1, 1, remove_pic1, -1);
    set_ref_man(&p_refman[2], 2, 3, 5, 1, 4, ref_pic2, 1, remove_pic2, -1);
    set_ref_man(&p_refman[3], 3, 4, 2, 1, 4, ref_pic3, 1, remove_pic3, -1);
}

/* ---------------------------------------------------------------------------
 * random access configuration (GOP 4) for reference management
 * (the max default reference frame number is 4)
 * coding order index in mini-GOP:                      2---1---3---0
 * type+POC: I0 B1 B2 B3 P4 B5 B6 B7 P8 B9 B10 B11 P12 B13 B14 B15 P16 B17 B18 B19 P20
 *      COI:  0  3  2  4  1  7  6  8  5 11  10  12   9  15  14  16  13  19  18  20  17
 *               3     4                                |   |   |   |
 *                  2                                   |   |   |   +-- ref_pic: 4: 13 -  4 = 9
 *            0           1                             |   |   |   |            3: 13 -  3 = 10
 *            encoding layer                            |   |   |   |            8: 13 -  8 = 5
 *                                                      |   |   |   |           12: 13 - 12 = 1
 *                                                      |   |   |   |
 * rm_pic = { 9,12 }, in order to                       |   |   |   +--  rm_pic: 9: 13 -  9 = 4
 *  9: remove I0 after P12 encoded                      |   |   |               12: 13 - 12 = 1
 * 12: remove the most front reference frame            |   |   |
 *                                                      |   |   +------ ref_pic: 3: 16 -  3 = 13
 *                                                      |   |                    2: 16 -  2 = 14
 *                                                      |   |
 *                                                      |   +---------- ref_pic: 1: 14 -  1 = 13
 *                                                      |   |                    5: 14 -  5 = 9
 *                                                      |   |
 * rm_pic = { 4 }, in order to                          |   +----------  rm_pic: 4: 14 -  4 = 10
 * remove the reference picture-B in last mini-GOP      |
 *                                                      +-------------- ref_pic: 1: 15 -  1 = 14
 *                                                                               6: 15 -  6 = 9
 */
static void default_reference_management_ra_gop4(xavs2_rps_t *p_refman)
{
    int ref_pic0[4]    = {  4,  3,  8, 12 };
    int ref_pic1[4]    = {  1,  5, -1, -1 };
    int ref_pic2[4]    = {  1,  6, -1, -1 };
    int ref_pic3[4]    = {  3,  2, -1, -1 };
    int remove_pic0[4] = {  9, 12, -1, -1 };
    int remove_pic1[4] = {  4, -1, -1, -1 };
    int remove_pic2[4] = { -1, -1, -1, -1 };
    int remove_pic3[4] = { -1, -1, -1, -1 };

    memset(p_refman, -1, sizeof(xavs2_rps_t) * XAVS2_MAX_GOPS);

    set_ref_man(&p_refman[0], 0, 4, 2, 1, 4, ref_pic0, 2, remove_pic0, -1);
    set_ref_man(&p_refman[1], 1, 2, 3, 1, 2, ref_pic1, 1, remove_pic1, -1);
    set_ref_man(&p_refman[2], 2, 1, 4, 0, 2, ref_pic2, 0, remove_pic2, -1);
    set_ref_man(&p_refman[3], 3, 3, 4, 0, 2, ref_pic3, 0, remove_pic3, -1);
}

/* ---------------------------------------------------------------------------
 * random access configuration (GOP 8) for reference management
 */
static void default_reference_management_ra_gop8(xavs2_rps_t *p_refman)
{
    int ref_pic0[4]    = {  8,  3,  7, 16 };
    int ref_pic1[4]    = {  1,  9, -1, -1 };
    int ref_pic2[4]    = {  1, 10, -1, -1 };
    int ref_pic3[4]    = {  1, 11, -1, -1 };
    int ref_pic4[4]    = {  3,  2, -1, -1 };
    int ref_pic5[4]    = {  5,  4, -1, -1 };
    int ref_pic6[4]    = {  1,  5, -1, -1 };
    int ref_pic7[4]    = {  7,  2, -1, -1 };
    int remove_pic0[4] = { 16, 17, -1, -1 };
    int remove_pic1[4] = {  4, -1, -1, -1 };
    int remove_pic2[4] = {  9, -1, -1, -1 };
    int remove_pic3[4] = { -1, -1, -1, -1 };
    int remove_pic4[4] = { -1, -1, -1, -1 };
    int remove_pic5[4] = { -1, -1, -1, -1 };
    int remove_pic6[4] = {  4, -1, -1, -1 };
    int remove_pic7[4] = { -1, -1, -1, -1 };

    memset(p_refman, -1, sizeof(xavs2_rps_t) * XAVS2_MAX_GOPS);

    set_ref_man(&p_refman[0], 0, 8, 1, 1, 4, ref_pic0, 2, remove_pic0, -1);
    set_ref_man(&p_refman[1], 1, 4, 1, 1, 2, ref_pic1, 1, remove_pic1, -1);
    set_ref_man(&p_refman[2], 2, 2, 2, 1, 2, ref_pic2, 1, remove_pic2, -1);
    set_ref_man(&p_refman[3], 3, 1, 4, 0, 2, ref_pic3, 0, remove_pic3, -1);
    set_ref_man(&p_refman[4], 4, 3, 4, 0, 2, ref_pic4, 0, remove_pic4, -1);
    set_ref_man(&p_refman[5], 5, 6, 2, 1, 2, ref_pic5, 0, remove_pic5, -1);
    set_ref_man(&p_refman[6], 6, 5, 4, 0, 2, ref_pic6, 1, remove_pic6, -1);
    set_ref_man(&p_refman[7], 7, 7, 4, 0, 2, ref_pic7, 0, remove_pic7, -1);
}

/* ---------------------------------------------------------------------------
 * find a frame in DPB with the specific COI
 */
static
xavs2_frame_t *find_frame_by_coi(xavs2_frame_buffer_t *frm_buf, int coi)
{
    xavs2_frame_t *frame;
    int i;

    for (i = 0; i < frm_buf->num_frames; i++) {
        if ((frame = frm_buf->frames[i]) != NULL) {
            xavs2_thread_mutex_lock(&frame->mutex);        /* lock */

            if (frame->i_frm_coi == coi) {
                xavs2_thread_mutex_unlock(&frame->mutex);  /* unlock */
                return frame;
            }

            xavs2_thread_mutex_unlock(&frame->mutex);      /* unlock */
        }
    }

    return NULL;
}

/* ---------------------------------------------------------------------------
 * get RPS of one frame
 */
static
int xavs2e_get_frame_rps(const xavs2_t *h, xavs2_frame_buffer_t *frm_buf,
                         xavs2_frame_t *cur_frm, xavs2_rps_t *p_rps)
{
    const xavs2_rps_t *p_seq_rps = h->param->cfg_ref_all;
    xavs2_frame_t     *frame = NULL;
    int rps_idx = 0;
    int j;

    if (cur_frm->i_frm_type == XAVS2_TYPE_I) {
        if (h->param->intra_period_max == 1) {
            memcpy(p_rps, &p_seq_rps[0], sizeof(xavs2_rps_t));
            p_rps->num_of_ref       = 0;  // clear reference frames for I frame
            p_rps->referd_by_others = 0;
        } else {
            p_rps->idx_in_gop       = -1;
            p_rps->num_of_ref       = 0;
            p_rps->num_to_rm        = 0;
            p_rps->referd_by_others = 1;

            if (!h->param->b_open_gop || !h->param->num_bframes) {
                // IDR refresh
                for (j = 0; j < frm_buf->num_frames; j++) {
                    if ((frame = frm_buf->frames[j]) != NULL && cur_frm->i_frame != frame->i_frame) {
                        xavs2_thread_mutex_lock(&frame->mutex);      /* lock */
                        assert(p_rps->num_to_rm < sizeof(p_rps->rm_pic) / sizeof(p_rps->rm_pic[0]));
                        if (frame->rps.referd_by_others == 1 && frame->cnt_refered > 0) {
                            if (cur_frm->i_frm_coi - frame->i_frm_coi < 64) {
                                /* only 6 bits for delta coi */
                                p_rps->rm_pic[p_rps->num_to_rm++] = cur_frm->i_frm_coi - frame->i_frm_coi;
                            }
                        }

                        xavs2_thread_mutex_unlock(&frame->mutex);    /* unlock */
                        if (p_rps->num_to_rm == sizeof(p_rps->rm_pic) / sizeof(p_rps->rm_pic[0])) {
                            break;
                        }

                        /* at most 7 frames can be removed limited to the "num of removed picture" in sequence/picture header */
                        if (p_rps->num_to_rm == 7) {
                            break;
                        }
                    }
                }
            } else {
                // RA OpenGOP, I帧出现在P/F帧的位置，按P/F帧的移除帧列表处理即可
                memcpy(p_rps, &p_seq_rps[0], sizeof(xavs2_rps_t));
                p_rps->idx_in_gop       = -1;
                p_rps->num_of_ref       = 0;
                p_rps->referd_by_others = 1;
            }

            p_rps->qp_offset        = 0;
        }
    } else {
        rps_idx = (cur_frm->i_frm_coi - 1 - ((!h->param->b_open_gop && h->param->num_bframes > 0) ? frm_buf->COI_IDR : 0)) % h->i_gop_size;
        memcpy(p_rps, &p_seq_rps[rps_idx], sizeof(xavs2_rps_t));

        if (cur_frm->i_frame > frm_buf->POC_IDR && (!h->param->b_open_gop || !h->param->num_bframes)) {
            /* clear frames before IDR frame */
            for (j = 0; j < frm_buf->num_frames; j++) {
                if ((frame = frm_buf->frames[j]) != NULL) {
                    xavs2_thread_mutex_lock(&frame->mutex);      /* lock */
                    assert(p_rps->num_to_rm < sizeof(p_rps->rm_pic) / sizeof(p_rps->rm_pic[0]));
                    if (frame->rps.referd_by_others == 1 && frame->cnt_refered > 0) {
                        /* only 6 bits for delta coi */
                        if (frame->i_frame < frm_buf->POC_IDR && cur_frm->i_frm_coi - frame->i_frm_coi < 64) {
                            p_rps->rm_pic[p_rps->num_to_rm++] = cur_frm->i_frm_coi - frame->i_frm_coi;
                        }
                    }

                    xavs2_thread_mutex_unlock(&frame->mutex);    /* unlock */
                    if (p_rps->num_to_rm == sizeof(p_rps->rm_pic) / sizeof(p_rps->rm_pic[0])) {
                        break;
                    }

                    /* at most 7 frames can be removed limited to the "num of removed picture" in sequence/picture header */
                    if (p_rps->num_to_rm == 7) {
                        break;
                    }
                }
            }
        }
    }

    return rps_idx;
}

/* ---------------------------------------------------------------------------
 * build reference list according to RPS, returns the number of frames found
 */
static INLINE
int rps_init_reference_list(const xavs2_t *h, xavs2_frame_buffer_t *frm_buf,
                            xavs2_frame_t *cur_frm,
                            xavs2_rps_t *p_rps, xavs2_frame_t *frefs[XAVS2_MAX_REFS])
{
    xavs2_frame_t *frame;
    int i, k, m;
    int num_ref = 0;

    for (i = 0; i < XAVS2_MAX_REFS; i++) {
        frefs[i] = NULL;
    }
    assert(p_rps->num_of_ref <= XAVS2_MAX_REFS);

    for (i = 0; i < p_rps->num_of_ref; i++) {
        for (m = 0;; m++) {
            int coi = cur_frm->i_frm_coi - p_rps->ref_pic[i];

            frame = find_frame_by_coi(frm_buf, coi);

            if (frame != NULL) {
                int b_could_be_referenced;
                xavs2_thread_mutex_lock(&frame->mutex);          /* lock */

                /* check whether the frame is already in the reference list */
                for (k = 0; k < num_ref; k++) {
                    if (frefs[k] == frame) {
                        // already in the reference list
                        p_rps->idx_in_gop = -1;
                        break;
                    }
                }

                /* check whether the frame could be referenced by current frame */
                b_could_be_referenced = frame->i_frame >= frm_buf->POC_IDR ||
                                        (frame->i_frame < frm_buf->POC_IDR && cur_frm->i_frm_type == XAVS2_TYPE_B && h->param->b_open_gop);

                if (k == num_ref &&
                    frame->i_frm_coi == coi &&
                    frame->cnt_refered > 0 &&
                    b_could_be_referenced) {

                    // put in the reference list
                    assert(frame->cnt_refered > 0);
                    assert(frame->rps.referd_by_others != 0);

                    // hold reference to this frame
                    frame->cnt_refered++;

                    frefs[num_ref] = frame;
                    p_rps->ref_pic[num_ref] = cur_frm->i_frm_coi - frame->i_frm_coi;

                    num_ref++;

                    xavs2_thread_mutex_unlock(&frame->mutex);    /* unlock */
                    /* a reference frame found */
                    break;
                }

                xavs2_thread_mutex_unlock(&frame->mutex);    /* unlock */
            }

            /* reference frame not found in the second run, break now */
            if (m > 0) {
                break;
            }

            // reference frame not found, fall back on the IDR frame.
            p_rps->ref_pic[i] = cur_frm->i_frm_coi - frm_buf->COI_IDR;
            p_rps->idx_in_gop = -1;
        }
    }

    return num_ref;
}

/* ---------------------------------------------------------------------------
 * fix reference list of B frame, returns the number of reference frames
 */
static INLINE
int rps_fix_reference_list_b(const xavs2_t *h, xavs2_frame_buffer_t *frm_buf,
                             xavs2_frame_t *cur_frm,
                             xavs2_rps_t *p_rps, xavs2_frame_t *frefs[XAVS2_MAX_REFS])
{
    xavs2_frame_t **DPB = frm_buf->frames;
    xavs2_frame_t *frame;
    int i;
    // reassign reference frames for this B frame.
    int max_fwd_idx = -1, min_bwd_idx = -1;
    dist_t max_fwd_poi = 0, min_bwd_poi = MAX_DISTORTION;

    UNUSED_PARAMETER(h);

    for (i = 0; i < frm_buf->num_frames; i++) {
        if ((frame = DPB[i]) != NULL) {
            xavs2_thread_mutex_lock(&frame->mutex);      /* lock */

            if (frame->rps.referd_by_others != 0 && frame->cnt_refered > 0 &&
                frame->i_frame < cur_frm->i_frame && frame->i_frame > max_fwd_poi) {
                if (max_fwd_idx != -1) {
                    xavs2_thread_mutex_lock(&DPB[max_fwd_idx]->mutex);   /* lock */
                    DPB[max_fwd_idx]->cnt_refered--;
                    assert(DPB[max_fwd_idx]->cnt_refered >= 0);
                    xavs2_thread_mutex_unlock(&DPB[max_fwd_idx]->mutex); /* unlock */
                }

                frame->cnt_refered++;

                max_fwd_idx = i;
                max_fwd_poi = frame->i_frame;
            }

            xavs2_thread_mutex_unlock(&frame->mutex);    /* unlock */
        }
    }

    assert(max_fwd_idx >= 0);
    assert(DPB[max_fwd_idx]->cnt_refered > 0);

    xavs2_thread_mutex_lock(&frefs[1]->mutex);     /* lock */
    frefs[1]->cnt_refered--;
    assert(frefs[1]->cnt_refered >= 0);
    xavs2_thread_mutex_unlock(&frefs[1]->mutex);   /* unlock */

    frefs[1] = DPB[max_fwd_idx];

    for (i = 0; i < frm_buf->num_frames; i++) {
        if ((frame = DPB[i]) != NULL) {
            xavs2_thread_mutex_lock(&frame->mutex);  /* lock */

            if (frame->rps.referd_by_others != 0 && frame->cnt_refered > 0 &&
                frame->i_frame > cur_frm->i_frame && frame->i_frame < min_bwd_poi) {
                if (min_bwd_idx != -1) {
                    xavs2_thread_mutex_lock(&DPB[min_bwd_idx]->mutex);   /* lock */
                    DPB[min_bwd_idx]->cnt_refered--;
                    assert(DPB[min_bwd_idx]->cnt_refered >= 0);
                    xavs2_thread_mutex_unlock(&DPB[min_bwd_idx]->mutex); /* unlock */
                }

                frame->cnt_refered++;

                min_bwd_idx = i;
                min_bwd_poi = frame->i_frame;
            }

            xavs2_thread_mutex_unlock(&frame->mutex);/* unlock */
        }
    }

    assert(min_bwd_idx >= 0);
    assert(DPB[min_bwd_idx]->cnt_refered > 0);

    xavs2_thread_mutex_lock(&frefs[0]->mutex);     /* lock */
    frefs[0]->cnt_refered--;
    assert(frefs[0]->cnt_refered >= 0);
    xavs2_thread_mutex_unlock(&frefs[0]->mutex);   /* unlock */

    frefs[0] = DPB[min_bwd_idx];

    p_rps->ref_pic[0] = cur_frm->i_frm_coi - frefs[0]->i_frm_coi;
    p_rps->ref_pic[1] = cur_frm->i_frm_coi - frefs[1]->i_frm_coi;
    p_rps->idx_in_gop = -1;

    return 2;   // number of reference frames for B frame
}


/* ---------------------------------------------------------------------------
 * fix reference list of P/F frame, returns the number of reference frames
 */
static INLINE
int rps_fix_reference_list_pf(const xavs2_t *h, xavs2_frame_buffer_t *frm_buf,
                              xavs2_frame_t *cur_frm,
                              xavs2_rps_t *p_rps, xavs2_frame_t *frefs[XAVS2_MAX_REFS])
{
    xavs2_frame_t **DPB = frm_buf->frames;
    xavs2_frame_t *frame;
    int i, j, k;
    int num_ref = p_rps->num_of_ref;

    for (i = num_ref; i < h->i_max_ref; i++) {
        int max_fwd_idx = -1;
        int max_fwd_poi = frm_buf->POC_IDR;
        int switch_flag = 0;

        for (j = 0; j < frm_buf->num_frames; j++) {
            if ((frame = DPB[j]) != NULL && frame->rps.referd_by_others) {
                xavs2_thread_mutex_lock(&frame->mutex);          /* lock */
                int poi = frame->i_frame;

                for (k = 0; k < num_ref; k++) {
                    if (frefs[k] == frame) {
                        break;
                    }
                }

                if (k < num_ref) {
                    xavs2_thread_mutex_unlock(&frame->mutex);    /* unlock */
                    continue;
                }

                if (poi < cur_frm->i_frame && poi > max_fwd_poi &&
                    XAVS2_ABS(poi - cur_frm->i_frame) < 128 && frame->cnt_refered > 0 &&
                    (h->param->temporal_id_exist_flag == 0 || h->i_layer >= frame->rps.temporal_id)) {
                    if (max_fwd_idx != -1) {
                        xavs2_thread_mutex_lock(&DPB[max_fwd_idx]->mutex);   /* lock */
                        DPB[max_fwd_idx]->cnt_refered--;
                        assert(DPB[max_fwd_idx]->cnt_refered >= 0);
                        xavs2_thread_mutex_unlock(&DPB[max_fwd_idx]->mutex); /* unlock */
                    }

                    assert(frame->rps.referd_by_others != 0);

                    frame->cnt_refered++;

                    max_fwd_idx = j;
                    max_fwd_poi = poi;
                    switch_flag = 1;
                }

                xavs2_thread_mutex_unlock(&frame->mutex);    /* unlock */
            }
        }

        if (switch_flag == 0) {
            break;
        }

        assert(DPB[max_fwd_idx]->cnt_refered > 0);

        frefs[i] = DPB[max_fwd_idx];
        p_rps->ref_pic[i] = cur_frm->i_frm_coi - frefs[i]->i_frm_coi;
        p_rps->idx_in_gop = -1;

        num_ref++;
    }

    return num_ref;   // number of reference frames for B frame
}

/* ---------------------------------------------------------------------------
 * check whether a frame is writable:
 *    no one is referencing this frame
 */
static ALWAYS_INLINE
int frame_is_writable(const xavs2_handler_t *h_mgr, xavs2_frame_t *frame)
{
    UNUSED_PARAMETER(h_mgr);
    return frame->cnt_refered == 0;
}

/* ---------------------------------------------------------------------------
 * check whether a frame is free to use
 */
static ALWAYS_INLINE
int frame_is_free(const xavs2_handler_t *h_mgr, int cur_poc, xavs2_frame_t *frame)
{
    if (frame_is_writable(h_mgr, frame)) {
        return 1;  /* this frame will never be used */
    } else {
        return (XAVS2_ABS(cur_poc - frame->i_frame) >= 128) /* is too long-ago frame ? */;
    }
}

/* ---------------------------------------------------------------------------
 * find a free frame for encoding
 */
static INLINE
xavs2_frame_t *frame_buffer_find_free_frame_dpb(xavs2_handler_t *h_mgr, xavs2_t *h, xavs2_frame_buffer_t *frm_buf,
        xavs2_frame_t *cur_frm, xavs2_rps_t *p_rps)
{
    xavs2_frame_t **DPB = frm_buf->frames;
    xavs2_frame_t *fdec_frm = NULL;
    int num_frames = frm_buf->num_frames;
    int i;

    // find a free frame for the fdec
    for (i = 0; i < num_frames; i++) {
        xavs2_frame_t *frame = DPB[i];
        if (frame != NULL) {
            xavs2_thread_mutex_lock(&frame->mutex);          /* lock */

            if (frame_is_free(h_mgr, cur_frm->i_frame, frame)) {
                frame->cnt_refered++;  // RDO
                frame->cnt_refered++;  // reconstruction output
                frame->cnt_refered++;  // Entropy encoding
                fdec_frm = frame;
                xavs2_thread_mutex_unlock(&frame->mutex);    /* unlock */
                break;
            }

            xavs2_thread_mutex_unlock(&frame->mutex);        /* unlock */
        }
    }

    // fdec must exist
    for (; fdec_frm == NULL;) {
        for (i = 0; i < num_frames; i++) {
            xavs2_frame_t *frame = DPB[i];
            if (frame != NULL) {
                xavs2_thread_mutex_lock(&frame->mutex);          /* unlock */

                if (frame_is_writable(h_mgr, frame)) {
                    p_rps->rm_pic[p_rps->num_to_rm++] = cur_frm->i_frm_coi - frame->i_frm_coi;
                    p_rps->idx_in_gop = -1;

                    frame->cnt_refered++;  // RDO
                    frame->cnt_refered++;  // reconstruction output
                    frame->cnt_refered++;  // Entropy encoding

                    fdec_frm = frame;

                    xavs2_thread_mutex_unlock(&frame->mutex);    /* unlock */

                    break;
                }

                xavs2_thread_mutex_unlock(&frame->mutex);        /* unlock */
            }
        }

        if (fdec_frm) {
            break;
        }

        xavs2_thread_cond_wait(&h_mgr->cond[SIG_FRM_BUFFER_RELEASED], &h_mgr->mutex);
    }

    if (fdec_frm) {
        memcpy(&fdec_frm->rps, p_rps, sizeof(xavs2_rps_t));
        fdec_frm->i_frame      = -1;
        fdec_frm->i_frm_coi    = -1;
        fdec_frm->cnt_refered += fdec_frm->rps.referd_by_others;

        memset(fdec_frm->num_lcu_coded_in_row, 0, h->i_height_in_lcu * sizeof(fdec_frm->num_lcu_coded_in_row[0]));
    }

    return fdec_frm;
}


/* ---------------------------------------------------------------------------
 * find a free frame for encoding
 */
static INLINE
void rps_determine_remove_frames(xavs2_frame_buffer_t *frm_buf, xavs2_frame_t *cur_frm)
{
    int i, k;

    // remove the frames that will never be referenced
    for (i = 0, k = 0; i < cur_frm->rps.num_to_rm; i++) {
        int coi = cur_frm->i_frm_coi - cur_frm->rps.rm_pic[i];
        xavs2_frame_t *frame;

        if (coi < 0) {
            continue;
        }

        frame = find_frame_by_coi(frm_buf, coi);

        if (frame != NULL) {
            xavs2_thread_mutex_lock(&frame->mutex);              /* lock */
            if (frame->i_frm_coi == coi && frame->cnt_refered > 0) {
                // can not remove frames with lower layers
                assert(cur_frm->rps.temporal_id <= frame->rps.temporal_id);

                cur_frm->rps.rm_pic[k++] = cur_frm->rps.rm_pic[i];

                xavs2_thread_mutex_unlock(&frame->mutex);    /* unlock */
                continue;
            }

            xavs2_thread_mutex_unlock(&frame->mutex);    /* unlock */
        }
    }

    if (cur_frm->rps.num_to_rm != k) {
        cur_frm->rps.num_to_rm = k;
        cur_frm->rps.idx_in_gop = -1;
    }
}

/* ---------------------------------------------------------------------------
 * update frame buffer, record frames to be removed
 */
void frame_buffer_update_remove_frames(xavs2_frame_buffer_t *frm_buf, xavs2_frame_t *cur_frm)
{
    int i;

    frm_buf->num_frames_to_remove = cur_frm->rps.num_to_rm;

    for (i = 0; i < frm_buf->num_frames_to_remove; i++) {
        frm_buf->coi_remove_frame[i] = cur_frm->i_frm_coi - cur_frm->rps.rm_pic[i];
    }

    // xavs2_log(NULL, XAVS2_LOG_INFO, "RPS remove[%d]:  %d,  [%d %d]\n",
    //           cur_frm->i_frm_coi, cur_frm->rps.num_to_rm, cur_frm->rps.rm_pic[0], cur_frm->rps.rm_pic[1]);
}

/* ---------------------------------------------------------------------------
 * update frame buffer, remove frames
 */
void frame_buffer_remove_frames(xavs2_frame_buffer_t *frm_buf)
{
    int i;

    for (i = 0; i < frm_buf->num_frames_to_remove; i++) {
        int coi_frame_to_remove = frm_buf->coi_remove_frame[i];
        xavs2_frame_t *frame = find_frame_by_coi(frm_buf, coi_frame_to_remove);

        if (frame != NULL) {
            xavs2_thread_mutex_lock(&frame->mutex);          /* lock */

            if (frame->i_frm_coi == coi_frame_to_remove && frame->cnt_refered > 0) {
                frame->cnt_refered--;
                // xavs2_log(NULL, XAVS2_LOG_DEBUG, "remove frame COI: %3d, POC %3d\n",
                //           frame->i_frm_coi, frame->i_frame);
                xavs2_thread_mutex_unlock(&frame->mutex);    /* unlock */
                break;
            }

            xavs2_thread_mutex_unlock(&frame->mutex);        /* unlock */
        }
    }
}


/**
 * ===========================================================================
 * functions
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
xavs2_frame_t *frame_buffer_get_free_frame_ipb(xavs2_handler_t *h_mgr)
{
    xavs2_frame_t *frame = NULL;

    frame = (xavs2_frame_t *)xl_remove_head(&h_mgr->list_frames_free, 1);

    return frame;
}


/* ---------------------------------------------------------------------------
 * build rps of a frame
 */
int rps_build(const xavs2_t *h, xavs2_frame_buffer_t *frm_buf,
              xavs2_frame_t *cur_frm,
              xavs2_rps_t *p_rps, xavs2_frame_t *frefs[XAVS2_MAX_REFS])
{
    // initialize current RPS
    cur_frm->rps_index_in_gop = xavs2e_get_frame_rps(h, frm_buf, cur_frm, p_rps);

    // get encoding layer of current frame
    if (h->param->temporal_id_exist_flag == 1 && cur_frm->i_frm_type != XAVS2_TYPE_I) {
        if (p_rps->temporal_id < 0 || p_rps->temporal_id >= TEMPORAL_MAXLEVEL) {
            p_rps->temporal_id = TEMPORAL_MAXLEVEL - 1;    // the lowest level
        }
    } else {
        p_rps->temporal_id = 0;
    }

    // prepare the reference list

    p_rps->num_of_ref = rps_init_reference_list(h, frm_buf, cur_frm, p_rps, frefs);

    if (cur_frm->i_frm_type == XAVS2_TYPE_B && p_rps->num_of_ref != 2) {
        return -1;
    }

    if (cur_frm->i_frm_type == XAVS2_TYPE_B &&
        (frefs[0]->i_frame <= cur_frm->i_frame || frefs[1]->i_frame >= cur_frm->i_frame)) {
        // for B frames with wrong reference frames
        p_rps->num_of_ref = rps_fix_reference_list_b(h, frm_buf, cur_frm, p_rps, frefs);
    } else if (cur_frm->i_frm_type == XAVS2_TYPE_P || cur_frm->i_frm_type == XAVS2_TYPE_F) {
        // for P/F-frame
        p_rps->num_of_ref = rps_fix_reference_list_pf(h, frm_buf, cur_frm, p_rps, frefs);
    }

    rps_determine_remove_frames(frm_buf, cur_frm);

    return 0;
}

/* ---------------------------------------------------------------------------
 * initializes the parameters for a new frame
 */
xavs2_frame_t *find_fdec_and_build_rps(xavs2_t *h, xavs2_frame_buffer_t *frm_buf,
                                       xavs2_frame_t *cur_frm,
                                       xavs2_frame_t *frefs[XAVS2_MAX_REFS])
{
    xavs2_handler_t *h_mgr = h->h_top;
    xavs2_frame_t   *frame = NULL;

    /* remove frames before current frame encoding */
    frame_buffer_remove_frames(frm_buf);

    if (rps_build(h, frm_buf, cur_frm, &cur_frm->rps, frefs) < 0) {
        return NULL;
    }

    /* find a frame for encoding */
    frame = frame_buffer_find_free_frame_dpb(h_mgr, h, frm_buf, cur_frm, &cur_frm->rps);

    /* label frames to be removed */
    frame_buffer_update_remove_frames(frm_buf, cur_frm);

    return frame;
}

/* ---------------------------------------------------------------------------
 * set picture reorder delay
 */
void rps_set_picture_reorder_delay(xavs2_t *h)
{
    h->i_gop_size = h->param->i_gop_size;

    if (!h->param->low_delay) {
        int delta_dd = 1000;
        int tmp_delta_dd;
        int i;

        for (i = 0; i < h->i_gop_size; i++) {
            tmp_delta_dd = h->param->cfg_ref_all[i].poc - (i + 1);
            if (tmp_delta_dd < delta_dd) {
                delta_dd = tmp_delta_dd;
            }
        }

        // set picture reorder delay
        if (delta_dd < 0) {
            h->picture_reorder_delay = -delta_dd;
        } else {
            h->picture_reorder_delay = 0;
        }
    }
}


/* ---------------------------------------------------------------------------
 * check RPS config
 */
static
int update_rps_config(xavs2_param_t *param)
{
    xavs2_rps_t *p_seq_rps = param->cfg_ref_all;

    if (param->i_gop_size < 0) {
        param->i_gop_size = XAVS2_ABS(param->i_gop_size);

        /* set default configuration for reference_management */
        memset(p_seq_rps, -1, XAVS2_MAX_GOPS * sizeof(xavs2_rps_t));

        if (param->num_bframes == 0) {
            /* LDP */
            default_reference_management_ldp(&p_seq_rps[0]);
        } else {
            /* RA */
            if (param->i_gop_size == 4) {
                default_reference_management_ra_gop4(&p_seq_rps[0]);
                param->num_bframes = 3;
            } else if (param->i_gop_size == 8) {
                default_reference_management_ra_gop8(&p_seq_rps[0]);
                param->num_bframes = 7;
            } else {
                /* GOP size error */
                return -1;
            }
        }
    }

    return 0;
}

/* ---------------------------------------------------------------------------
 * config RPS
 */
int rps_check_config(xavs2_param_t *param)
{
    xavs2_rps_t *p_seq_rps = param->cfg_ref_all;
    int rps_idx;

    if (update_rps_config(param) < 0) {
        return -1;
    }

    // set index
    for (rps_idx = 0; rps_idx < param->i_gop_size; rps_idx++) {
        p_seq_rps[rps_idx].idx_in_gop = rps_idx;
    }

    if (param->num_max_ref < 4) {
        for (rps_idx = 0; rps_idx < param->i_gop_size; rps_idx++) {
            p_seq_rps[rps_idx].num_of_ref = XAVS2_MIN(param->num_max_ref, p_seq_rps[rps_idx].num_of_ref);
        }
    }

    return 0;
}
