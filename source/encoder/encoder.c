/*
 * encoder.c
 *
 * Description of this file:
 *    Encoder functions definition of the xavs2 library
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
#include "wrapper.h"
#include "encoder.h"
#include "frame.h"
#include "aec.h"
#include "slice.h"
#include "nal.h"
#include "ratecontrol.h"
#include "tdrdo.h"
#include "me.h"
#include "cpu.h"
#include "rdo.h"
#include "rps.h"
#include "wquant.h"
#include "header.h"
#include "cudata.h"
#include "block_info.h"
#include "presets.h"
#include "version.h"
#include "alf.h"
#include "sao.h"
#include "ratecontrol.h"

/* "video_sequence_end_code", 0xB1 */
static const uint8_t end_code[16] = {
    0x00, 0x00, 0x01, 0xB1
};
static const int     len_end_code = 4;

/**
 * ===========================================================================
 * local tables
 * ===========================================================================
 */

static const int tab_frm_type_to_slice_type[] = {
    0, 0, 0, 1, 3, 2, 0, 0, 0
};

// ---------------------------------------------------------------------------
static const int tab_LambdaQ[4] = {     // [slice_type]
    -2, 0, 0, 0
};

// ---------------------------------------------------------------------------
static const double tab_LambdaF[4] = {  // [slice_type]
    0.60, 0.60, 0.85, 0.60
};

// ---------------------------------------------------------------------------
static const int8_t tab_cu_bfs_order[] = {
    21, 5, 1, 0
};

extern double tab_qsfd_thres[MAX_QP][2][CTU_DEPTH];

/* ---------------------------------------------------------------------------
 * QSFD threshold
 */
static ALWAYS_INLINE
void qsfd_calculate_threshold_of_a_frame(xavs2_t *h)
{
    assert(sizeof(h->thres_qsfd_cu) == sizeof(tab_qsfd_thres[0]));

    memcpy(h->thres_qsfd_cu, tab_qsfd_thres[h->i_qp], sizeof(h->thres_qsfd_cu));
}


/* ---------------------------------------------------------------------------
 * decrease the reference count by one
 */
static void release_one_frame(xavs2_t *h, xavs2_frame_t *frame)
{
    xavs2_handler_t *h_mgr = h->h_top;

    xavs2_thread_mutex_lock(&frame->mutex);      /* lock */

    assert(frame->cnt_refered > 0);
    frame->cnt_refered--;

    xavs2_thread_mutex_unlock(&frame->mutex);    /* unlock */

    if (frame->cnt_refered == 0) {
        /* signal to the h_mgr */
        xavs2_thread_cond_signal(&h_mgr->cond[SIG_FRM_BUFFER_RELEASED]);
    }
}


/* ---------------------------------------------------------------------------
 * fill packet data for output
 */
static ALWAYS_INLINE
void encoder_fill_packet_data(xavs2_handler_t *h_mgr, xavs2_outpacket_t *packet, xavs2_frame_t *frame)
{
    assert(packet != NULL);
    packet->private_data = frame;
    packet->opaque       = h_mgr->user_data;

    if (frame == NULL) {
        packet->stream   = end_code;
        packet->len      = 0;
        packet->state    = XAVS2_STATE_FLUSH_END;
        packet->type     = 0;
        packet->pts      = h_mgr->max_out_pts;
        packet->dts      = h_mgr->max_out_dts;
        if (h_mgr->b_seq_end == 0) {
            packet->len = len_end_code;
            h_mgr->b_seq_end = 1;
        }
    } else {
        assert(frame->i_bs_len > 0);

        packet->stream   = frame->p_bs_buf;
        packet->len      = frame->i_bs_len;
        packet->state    = XAVS2_STATE_ENCODED;
        packet->type     = frame->i_frm_type;
        packet->pts      = frame->i_pts;
        packet->dts      = frame->i_dts;
        h_mgr->max_out_pts = XAVS2_MAX(h_mgr->max_out_pts, frame->i_pts);
        h_mgr->max_out_dts = XAVS2_MAX(h_mgr->max_out_dts, frame->i_dts);
    }
}

/**
 * ---------------------------------------------------------------------------
 * Function   : output encoded data from the encoder
 * Parameters :
 *      [in]  : opaque - user data
 *      [in]  : frame - pointer to the frame
 * Return     : none
 * ---------------------------------------------------------------------------
 */
static INLINE
void encoder_output_frame_bitstream(xavs2_handler_t *h_mgr, xavs2_frame_t *frame)
{
    if (frame != NULL) {
        xl_append(&h_mgr->list_frames_output, frame);
    }
}

/**
 * ---------------------------------------------------------------------------
 * Function   : fetch bit-stream of one encoded frame
 * Parameters :
 *      [in ] : h_mgr - pointer to xavs2_handler_t
 *      [out] : packet of one encoded frame
 * Return     : none
 * ---------------------------------------------------------------------------
 */
void encoder_fetch_one_encoded_frame(xavs2_handler_t *h_mgr, xavs2_outpacket_t *packet, int is_flush)
{
    int num_encoding_frames = h_mgr->num_encode - h_mgr->num_output;  // 正在编码帧数
    int num_frames_threads  = h_mgr->i_frm_threads;      // 并行帧数

    /* clear packet data */
    packet->len          = 0;
    packet->private_data = NULL;

    if (is_flush && h_mgr->num_input == h_mgr->num_output) {
        /* all frames are encoded and have been output;
         * return video_end_code to finish encoding */
        encoder_fill_packet_data(h_mgr, packet, NULL);
    } else if (num_encoding_frames > num_frames_threads || is_flush) {
        /* now we should wait for one frame output */
        xavs2_frame_t *frame = (xavs2_frame_t *)xl_remove_head(&h_mgr->list_frames_output, 1);

        if (frame != NULL) {
            encoder_fill_packet_data(h_mgr, packet, frame);
            h_mgr->num_output++;
            assert(frame->i_bs_len > 0);
        }
    }
}

/* ---------------------------------------------------------------------------
 * check pseudo code and merge slice data with slice header bits
 */
static ALWAYS_INLINE
void check_pseudo_code_and_merge_slice_data(bs_t *p_bs, aec_t *p_aec)
{
    uint8_t *dst = p_bs->p;           /* point to the end of previous bitstream */
    uint8_t *src = p_aec->p_start;    /* point to the start position of source */
    uint8_t *end = p_aec->p;          /* point to the end   position of source */

    assert(p_bs->i_left == 8);

    /* check pseudo code */
    p_bs->p = nal_escape_c(dst, src, end);
}

/* ---------------------------------------------------------------------------
 * calculate lambda for one frame
 */
void xavs2e_get_frame_lambda(xavs2_t *h, xavs2_frame_t *cur_frm, int i_qp)
{
    double lambda;
    int i_type = tab_frm_type_to_slice_type[cur_frm->i_frm_type];
    int qp = i_qp - SHIFT_QP;
    int rps_idx = cur_frm->rps_index_in_gop;
#if ENABLE_WQUANT
    // adaptive frequency weighting quantization
    if (h->WeightQuantEnable) {
        qp += tab_LambdaQ[i_type];
    }
#endif
    lambda = pow(2, qp / 4.0);
#if ENABLE_WQUANT
    if (h->WeightQuantEnable) {
        lambda *= tab_LambdaF[i_type];
    } else {
        lambda *= 0.85 *  LAM_2Level_TU;
    }
#else
    lambda *= 0.85 *  LAM_2Level_TU;
#endif

    if (h->param->intra_period_max != 1) {
        if (h->param->num_bframes > 0) {
            if (i_type != SLICE_TYPE_I && rps_idx != 0) {
                if (i_type == SLICE_TYPE_B) {
                    lambda *= 1.2;
                }
                lambda *= XAVS2_CLIP3F(2.00, 4.00, qp / 8.0);
            } else if (i_type == SLICE_TYPE_I) {
                lambda *= 0.8;
            }
        } else if (i_type != SLICE_TYPE_I) {
            lambda *= 0.8;
            if ((rps_idx + 1) % h->i_gop_size != 0) {
                lambda *= XAVS2_CLIP3F(2.00, 4.00, qp / 8.0) * 0.8;
            }
        }
    }

    /* only use for RA configure */
#if AQPO
    if (h->param->is_enable_AQPO && h->param->intra_period_max != 1 && h->param->i_cfg_type == 2) {
        int gop_size;
        int num_poc;
        int index;
        int intra_period_num;
        int temp_a, temp_b, temp_c, temp_d;
        float temp_e, temp_f, temp_g;
        gop_size = h->param->i_gop_size;
        num_poc = (h->curr_coi >> 8) << 8;
        if (cur_frm->i_frame + num_poc == 0) {
            h->param->GopQpOffset_old = 0;
        }

        if (cur_frm->i_frame + num_poc != 0) {
            if ((cur_frm->i_frame + num_poc) % gop_size == 0) {
                intra_period_num = h->param->intra_period_max;
                index = ((cur_frm->i_frame + num_poc + gop_size - 1) / gop_size) % intra_period_num;

                temp_a = (intra_period_num - index) >> 1;
                temp_b = intra_period_num >> 1;
                temp_c = (intra_period_num - 1) >> 1;
                temp_d = (intra_period_num - index - 1) >> 1;
                temp_e = 7.5 * (1 / (float)pow(2, temp_b)) + 5 * (1 / (float)pow(2, temp_c));
                temp_f = 7.5 * (1 / (float)pow(2, temp_a)) + 5 * (1 / (float)pow(2, temp_d));
                temp_g = (temp_f - temp_e) / (14 - temp_e);
                temp_g = (temp_g + 1.0)*intra_period_num / (intra_period_num + 2.0);
                if (temp_g < 0.8001) {
                    h->param->GopQpOffset = 0;
                } else if (temp_g < 1.0) {
                    h->param->GopQpOffset = 1;
                } else {
                    h->param->GopQpOffset = 2;
                }
                h->param->GopQpOffset = XAVS2_CLIP3(0, h->param->GopQpOffset_old + 1, h->param->GopQpOffset);
                h->param->GopQpOffset_old = h->param->GopQpOffset;
            }
            lambda *= exp(h->param->GopQpOffset / 5.661);
        }

    }
#endif

    cur_frm->f_frm_lambda_ssd = lambda;
    cur_frm->i_frm_lambda_sad = LAMBDA_FACTOR(sqrt(lambda));
}


/* ---------------------------------------------------------------------------
 * calculate lambda for RDO
 */
static void xavs2e_update_lambda(xavs2_t *h, int i_type, double lambda)
{
    /* get lambda for RDO */
    h->f_lambda_mode   = (rdcost_t)lambda;
    h->i_lambda_factor = LAMBDA_FACTOR(sqrt(lambda));
    h->f_lambda_1th    = 1.0 / h->f_lambda_mode;

    /* get lambda for RDOQ */
    if (h->param->i_rdoq_level != RDOQ_OFF) {
        h->f_lambda_rdoq = (lambda * h->param->lambda_factor_rdoq + 50) / 100;
        if (i_type == SLICE_TYPE_P || i_type == SLICE_TYPE_F) {
            h->f_lambda_rdoq = (h->f_lambda_rdoq * h->param->lambda_factor_rdoq_p + 50) / 100;
        } else if (i_type == SLICE_TYPE_B) {
            h->f_lambda_rdoq = (h->f_lambda_rdoq * h->param->lambda_factor_rdoq_b + 50) / 100;
        }
    }
}


/* ---------------------------------------------------------------------------
 * initializes the parameters for a new frame
 */
static void init_frame(xavs2_t *h, xavs2_frame_t *frame)
{
    int frame_size_in_4x4 = h->i_height_in_minpu * h->i_width_in_minpu;
    dist_t *all_mincost = &h->all_mincost[0][0][0];

    h->fenc  = frame;
    h->i_nal = 0;

    switch (frame->i_frm_type) {
    case XAVS2_TYPE_I:
        h->i_nal_type    = NAL_SLICE;
        h->i_nal_ref_idc = NAL_PRIORITY_HIGHEST;
        h->i_type        = SLICE_TYPE_I;
        break;
    case XAVS2_TYPE_P:
        h->i_nal_type    = NAL_SLICE;
        h->i_nal_ref_idc = NAL_PRIORITY_HIGH;
        h->i_type        = SLICE_TYPE_P;
        break;
    case XAVS2_TYPE_F:
        h->i_nal_type    = NAL_SLICE;
        h->i_nal_ref_idc = NAL_PRIORITY_HIGH;
        h->i_type        = SLICE_TYPE_F;
        break;
    default:
        h->i_nal_type    = NAL_SLICE;
        h->i_nal_ref_idc = NAL_PRIORITY_DISPOSABLE;
        h->i_type        = SLICE_TYPE_B;
        break;
    }

    // initialize slice index of each CTU/LCU
    g_funcs.fast_memset(h->lcu_slice_idx, -1, h->i_width_in_lcu * h->i_height_in_lcu * sizeof(int8_t));

    // initialize MVs, references and prediction direction
    if (h->param->me_method == XAVS2_ME_UMH) {
        g_funcs.mem_repeat_i(all_mincost, 1 << 30, frame_size_in_4x4 * MAX_INTER_MODES * MAX_REFS * sizeof(dist_t) / sizeof(int32_t));
    }
}

/* ---------------------------------------------------------------------------
 * get the next input frame order
 */
static int Advance2NextFrame(xavs2_handler_t *h_mgr, int frame)
{
    return (frame + 1) % h_mgr->i_frm_threads;
}

/* ---------------------------------------------------------------------------
 * get a frame encoder handle
 */
static xavs2_t *encoder_alloc_frame_task(xavs2_handler_t *h_mgr, xavs2_frame_t *frame)
{
    int refs_unavailable = 0;
    int i, j;

    xavs2_thread_mutex_lock(&h_mgr->mutex);   /* lock */

    /* wait until we successfully get one frame context */
    for (;;) {
        for (i = 0; i < h_mgr->i_frm_threads; i++) {
            /* alloc a frame task */
            xavs2_t *h = h_mgr->frm_contexts[i];
            assert(h->task_type == XAVS2_TASK_FRAME);

            if (h->task_status == XAVS2_TASK_FREE) {
                /* initialize the task */
                h->task_status  = XAVS2_TASK_BUSY;
                h->i_frame_b    = h_mgr->dpb.i_frame_b;
                h->ip_pic_idx   = h_mgr->dpb.ip_pic_idx;
                h->i_aec_frm    = h_mgr->i_frame_in;
                h->b_all_row_ctx_released = 0;

#if XAVS2_STAT
                /* reset frame statistics */
                memset(&h->frameinfo->frame_stat, 0, sizeof(frame_stat_t));
#endif

                /* reset all rows */
                for (j = 0; j < h->i_height_in_lcu; j++) {
                    row_info_t *row = &h->frameinfo->rows[j];

                    row->h     = 0;
                    row->row   = j;
                    row->coded = -1;
                }

                /* init caches */
                init_frame(h, frame);
                h->fenc->b_random_access_decodable = (h->fenc->i_frame >= h_mgr->dpb.POC_IDR);

                /* update the task manager */
                frame_buffer_update(h, &h_mgr->dpb, h->fenc);

                /* advance to the next input frame */
                h_mgr->i_frame_in = Advance2NextFrame(h_mgr, h_mgr->i_frame_in);

                /* prepare the reference list */
                h->fdec = find_fdec_and_build_rps(h, &h_mgr->dpb, h->fenc, h->fref);

                if (h->fdec == NULL) {
                    xavs2_log(NULL, XAVS2_LOG_DEBUG, "find FDEC or build reference lists fail\n");
                    refs_unavailable = 1;
                    break;
                }

                /* decide frame QP and lambdas */
                h->fenc->i_frm_qp = clip_qp(h, xavs2_rc_get_base_qp(h) + h->fenc->rps.qp_offset);
                xavs2e_get_frame_lambda(h, h->fenc, h->fenc->i_frm_qp);

                h->i_qp = h->fenc->i_frm_qp;
                /* update lambda in encoder handler (h) */
                xavs2e_update_lambda(h, h->i_type, h->fenc->f_frm_lambda_ssd);
                h->frameinfo->frame_stat.stat_frm.f_lambda_frm = h->f_lambda_mode;

                /* refine qp */
                if (h->param->enable_refine_qp && h->param->intra_period_min > 1) {
                    h->i_qp = (int)(5.661 * log((double)(h->f_lambda_mode)) + 13.131 + 0.5);
                }
                /* udpdate some properties */
                h->i_ref = h->fenc->rps.num_of_ref;
                h->i_layer = h->fenc->rps.temporal_id;
                assert(h->i_ref <= XAVS2_MAX_REFS);

                xavs2_thread_mutex_unlock(&h_mgr->mutex); /* unlock */
                /* signal to the aec thread */
                xavs2_thread_cond_signal(&h_mgr->cond[SIG_FRM_CONTEXT_ALLOCATED]);

                return h;
            }
        }

        if (refs_unavailable) {
            break;
        }

        xavs2_thread_cond_wait(&h_mgr->cond[SIG_FRM_CONTEXT_RELEASED], &h_mgr->mutex);
    }

    xavs2_thread_mutex_unlock(&h_mgr->mutex); /* unlock */

    return 0;
}

/* ---------------------------------------------------------------------------
 * set frame task status
 */
static void encoder_set_task_status(xavs2_t *h, task_status_e status)
{
    xavs2_handler_t *h_mgr = h->h_top;

    assert(h->task_type == XAVS2_TASK_FRAME);

    xavs2_thread_mutex_lock(&h_mgr->mutex);   /* lock */

    if ((status == XAVS2_TASK_RDO_DONE && h->task_status == XAVS2_TASK_AEC_DONE) ||
        (status == XAVS2_TASK_AEC_DONE && h->task_status == XAVS2_TASK_RDO_DONE)) {
        h->task_status = XAVS2_TASK_FREE;
    } else {
        h->task_status = status;
    }

    xavs2_thread_mutex_unlock(&h_mgr->mutex); /* unlock */

    if (status == XAVS2_TASK_AEC_DONE) {
        /* signal to the output proc */
        xavs2_thread_cond_signal(&h_mgr->cond[SIG_FRM_AEC_COMPLETED]);
    }

    if (h->task_status == XAVS2_TASK_FREE) {
        /* broadcast to the task manager & flush */
        xavs2_thread_cond_broadcast(&h_mgr->cond[SIG_FRM_CONTEXT_RELEASED]);
    }
}

/* ---------------------------------------------------------------------------
 */
static
void encoder_write_rec_frame(xavs2_handler_t *h_mgr)
{
    xavs2_t        *h     = h_mgr->p_coder;
    xavs2_frame_t **DPB   = h_mgr->dpb.frames;
    int size_dpb = h_mgr->dpb.num_frames;
    int i = 0;
    int j;

    xavs2_thread_mutex_lock(&h_mgr->mutex);   /* lock */

    while (i < size_dpb) {
        int next_output_frame_idx;
        xavs2_frame_t  *frame = DPB[i];
        if (frame == NULL) {
            i++;
            continue;
        }

        xavs2_thread_mutex_lock(&frame->mutex);  /* lock */

        next_output_frame_idx = get_next_frame_id(h_mgr->i_output);
        if (frame->i_frame == next_output_frame_idx) {
            /* has the frame already been reconstructed ? */
            for (j = 0; j < h->i_height_in_lcu; j++) {
                if (frame->num_lcu_coded_in_row[j] < h->i_width_in_lcu) {
                    break;
                }
            }

            if (j < h->i_height_in_lcu) {
                /* frame doesn't finish reconstruction */
                xavs2_thread_mutex_unlock(&frame->mutex);   /* unlock */
                break;
            }

            /* update output frame index */
            h_mgr->i_output = next_output_frame_idx;

#if XAVS2_DUMP_REC
            dump_yuv_out(h, h_mgr->h_rec_file, frame, h->param->org_width, h->param->org_height);
#endif //if XAVS2_DUMP_REC
            xavs2_thread_mutex_unlock(&frame->mutex);   /* unlock */

            /* release one frame */
            release_one_frame(h, frame);   // write reconstruction file

            /* start over for the next reconstruction frame */
            i = 0;
            continue;
        }

        xavs2_thread_mutex_unlock(&frame->mutex);    /* unlock */
        i++;
    }

    xavs2_thread_mutex_unlock(&h_mgr->mutex);     /* unlock */
}


/* ---------------------------------------------------------------------------
 * the aec encoding
 */
static INLINE
void encoder_encode_frame_header(xavs2_t *h)
{
    bs_t   *p_bs = &h->header_bs;
    int overhead = 30;  /* number of overhead bytes (include I/P/B picture header) */

    /* init bitstream context */
    xavs2_bs_init(p_bs, h->p_bs_buf_header, h->i_bs_buf_header);

    /* create sequence header if need ------------------------------
     */
    if (h->fenc->b_keyframe) {
        if (h->fenc->i_frm_coi == 0 || h->param->intra_period_min > 1) {
            /* generate sequence parameters */
            nal_start(h, NAL_SPS, NAL_PRIORITY_HIGHEST);
            xavs2_sequence_write(h, p_bs);
            nal_end(h);

            overhead += h->p_nal[h->i_nal - 1].i_payload;

            /* generate user data */
            nal_start(h, NAL_AUD, NAL_PRIORITY_HIGHEST);
            xavs2_user_data_write(p_bs);
            nal_end(h);

            overhead += h->p_nal[h->i_nal - 1].i_payload;
        }
    }

    /* nal start for picture header */
    nal_start(h, NAL_PPS, NAL_PRIORITY_HIGHEST);
    if (h->i_type == SLICE_TYPE_I) {
        xavs2_intra_picture_header_write(h, p_bs);
    } else {
        xavs2_inter_picture_header_write(h, p_bs);
    }
    // write picture header (ALF)
    xavs2_picture_header_alf_write(h, h->pic_alf_params, p_bs);
    bs_stuff_bits(p_bs);        // stuff bits after finishing ALF
    nal_end(h);                 // nal for picture header
}


/* ---------------------------------------------------------------------------
 * the aec encoding
 */
static void *encoder_aec_encode_one_frame(xavs2_t *h)
{
    aec_t            aec;
    frame_info_t    *frame = h->frameinfo;
    xavs2_frame_t   *fdec  = h->fdec;
    row_info_t      *row   = NULL;
    lcu_info_t      *lcu   = NULL;
    slice_t         *slice = NULL;
    aec_t           *p_aec = &aec;
    outputframe_t    output_frame;
#if XAVS2_STAT
    frame_stat_t *frm_stat = &frame->frame_stat;
    int i = 0;
#endif
    int lcu_xy = 0;
    int lcu_x = 0, lcu_y = 0;

    /* encode frame header */
    encoder_encode_frame_header(h);

    /* encode all LCUs */
    for (lcu_y = 0; lcu_y < h->i_height_in_lcu; lcu_y++) {
        row = &frame->rows[lcu_y];

        /* wait until the row finishes RDO */
        xavs2_thread_mutex_lock(&fdec->mutex);   /* lock */
        while (fdec->num_lcu_coded_in_row[lcu_y] < h->i_width_in_lcu) {
            xavs2_thread_cond_wait(&fdec->cond, &fdec->mutex);
        }
        xavs2_thread_mutex_unlock(&fdec->mutex); /* unlock */

        /* row is clear: start aec for every LCU */
        for (lcu_x = 0; lcu_x < h->i_width_in_lcu; lcu_x++, lcu_xy++) {
            lcu   = &row->lcus[lcu_x];
            slice = h->slices[lcu->slice_index];

            // while (fdec->num_lcu_coded_in_row[lcu_y] <= lcu_x) {
            //     xavs2_sleep_ms(1);
            // }

            if (lcu_xy == slice->i_first_lcu_xy) {
                /* slice start : initialize the aec engine */
                aec_start(h, p_aec, slice->bs.p_start + PSEUDO_CODE_SIZE, slice->bs.p_end, 1);
                p_aec->b_writting = 1;
            }

            if (h->param->enable_sao) {
                write_saoparam_one_lcu(h, p_aec, lcu_x, lcu_y, h->slice_sao_on, h->sao_blk_params[lcu_y * h->i_width_in_lcu + lcu_x]);
            }

            if (h->param->enable_alf) {
                int alf_comp;
                for (alf_comp = 0; alf_comp < 3; alf_comp++) {
                    if (h->pic_alf_on[alf_comp]) {
                        p_aec->binary.write_alf_lcu_ctrl(p_aec, h->is_alf_lcu_on[lcu_xy][alf_comp]);
                    }
                }
            }

            xavs2_lcu_write(h, p_aec, lcu, h->i_lcu_level, lcu->pix_x, lcu->pix_y);

            /* for the last LCU in SLice, write 1, otherwise write 0 */
            xavs2_lcu_terminat_bit_write(p_aec, lcu_xy == slice->i_last_lcu_xy);
        }

        /* 仅考虑LCU行级的Slice划分方式 */
        if (lcu_xy >= slice->i_last_lcu_xy) {
            int bs_len;
            /* slice done */
            aec_done(p_aec);

            /* check pseudo start code, and store bit stream length */
            check_pseudo_code_and_merge_slice_data(&slice->bs, p_aec);
            bs_len = xavs2_bs_pos(&slice->bs) / 8;

            nal_merge_slice(h, slice->p_slice_bs_buf, bs_len, h->i_nal_type, h->i_nal_ref_idc);
        }
    }

    h->fenc->i_bs_len = (int)encoder_encapsulate_nals(h, h->fenc, 0);

#if XAVS2_STAT
    /* collect frame properties */
    frm_stat->i_type  = h->i_type;
    frm_stat->i_frame = h->fenc->i_frame;
    frm_stat->i_qp    = h->i_qp;
    frm_stat->i_ref   = h->i_ref;

    for (i = 0; i < h->i_ref; i++) {
        frm_stat->ref_poc_set[i] = h->fref[i]->i_frm_poc >> 1;
    }

    h->fenc->i_time_end = xavs2_mdate();

    if (h->param->enable_psnr) {
        encoder_cal_psnr(h, &frm_stat->stat_frm.f_psnr[0], &frm_stat->stat_frm.f_psnr[1], &frm_stat->stat_frm.f_psnr[2]);
    } else {
        frm_stat->stat_frm.f_psnr[0] = 0;
        frm_stat->stat_frm.f_psnr[1] = 0;
        frm_stat->stat_frm.f_psnr[2] = 0;
    }

    if (h->param->enable_ssim) {
        encoder_cal_ssim(h, &frm_stat->stat_frm.f_ssim[0], &frm_stat->stat_frm.f_ssim[1], &frm_stat->stat_frm.f_ssim[2]);
    } else {
        frm_stat->stat_frm.f_ssim[0] = 0;
        frm_stat->stat_frm.f_ssim[1] = 0;
        frm_stat->stat_frm.f_ssim[2] = 0;
    }
#endif

    /* make sure all row context has been released */
    while (h->b_all_row_ctx_released == 0) {
        xavs2_sleep_ms(1);
    }

    /* release the reconstructed frame */
    release_one_frame(h, h->fdec);

    /* update rate control */
    xavs2_rc_update_after_frame_coded(h, h->fenc->i_bs_len * 8, h->i_qp, h->fenc->i_frm_type, h->fenc->i_frame);

    /* output this encoded frame */
    output_frame.frm_enc = h->fenc;
    output_frame.next    = NULL;
#if XAVS2_STAT
    memcpy(&output_frame.out_frm_stat, &h->frameinfo->frame_stat, sizeof(output_frame.out_frm_stat));

    /* report frame encoding */
    if (output_frame.frm_enc->i_bs_len > 0) {
        encoder_report_one_frame(h, &output_frame);
        if (output_frame.frm_enc->i_bs_len >= (output_frame.frm_enc->i_bs_buf >> 2)) {
            h->h_top->stat.num_frame_small_qp++;
            if ((h->h_top->stat.num_frame_small_qp & 128) == 1) {
                if (output_frame.frm_enc->i_bs_len > output_frame.frm_enc->i_bs_buf) {
                    xavs2_log(h, XAVS2_LOG_ERROR, "Frame bitstream exceeds the BS buffer size. num:%d\n",
                              h->h_top->stat.num_frame_small_qp);
                } else {
                    xavs2_log(h, XAVS2_LOG_WARNING, "Frame bitstream exceeds 1/4 BS buffer size. num %d\n",
                              h->h_top->stat.num_frame_small_qp);
                }
            }
        }
    }
#endif

    /* output bitstream and recycle input frame */
    {
        xavs2_handler_t *h_mgr = h->h_top;
        while (h_mgr->i_exit_flag != XAVS2_EXIT_THREAD) {
            /* wait until it is time for output of this frame */
            if (h_mgr->i_frame_aec == h->i_aec_frm) {
                break;
            }
        }

        xavs2_thread_mutex_lock(&h_mgr->mutex); /* lock */
        encoder_output_frame_bitstream(h_mgr, output_frame.frm_enc);
        h_mgr->i_frame_aec = Advance2NextFrame(h_mgr, h_mgr->i_frame_aec);
        xavs2_thread_mutex_unlock(&h_mgr->mutex); /* unlock */
    }

    /* set task status */
    encoder_set_task_status(h, XAVS2_TASK_AEC_DONE);

    return NULL;
}


/**
 * ---------------------------------------------------------------------------
 * Function   : flush reconstructed frames from the encoder
 * Parameters :
 *      [in ] : h   - pointer to struct xavs2_t, the xavs2 encoder
 *      [out] : none
 * Return     : none
 * ---------------------------------------------------------------------------
 */
static void encoder_flush(xavs2_handler_t *h_mgr)
{
    int i = 0;

    if (h_mgr == NULL) {
        return;
    }

    xavs2_thread_mutex_lock(&h_mgr->mutex);   /* lock */

    /* wait until all tasks free */
    while (i < h_mgr->i_frm_threads) {
        xavs2_t *h_frm_coder = h_mgr->frm_contexts[i];
        if (h_frm_coder && h_frm_coder->task_status != XAVS2_TASK_FREE) {
            /* use 'sleep()' instead ? */
            xavs2_thread_cond_wait(&h_mgr->cond[SIG_FRM_CONTEXT_RELEASED], &h_mgr->mutex);
            /* recheck all */
            i = 0;
            continue;
        }

        i++;
    }

    xavs2_thread_mutex_unlock(&h_mgr->mutex); /* unlock */

    encoder_write_rec_frame(h_mgr);
}

/**
 * ---------------------------------------------------------------------------
 * Function   : determine the MVD's value (1/4 pixel) is legal or not
 * Parameters :
 *      [in ] : h   - pointer to struct xavs2_t, the xavs2 encoder
 *      [out] :     - (-1) fail, (0) success
 * Return     : none
 * ---------------------------------------------------------------------------
 */
static int encoder_decide_mv_range(xavs2_t *h)
{
    /* set horizontal mv range */
    h->min_mv_range[0] = -8192;
    h->max_mv_range[0] =  8191;

    if (h->param->profile_id == MAIN10_PROFILE || h->param->profile_id == MAIN_PROFILE) {
        if (h->param->i_frame_threads > 1) {
            /* set vertical mv range */
            h->min_mv_range[1] = -((1 << h->i_lcu_level) << 2);
            h->max_mv_range[1] =  ((1 << h->i_lcu_level) << 2) - 1;
        } else {
            /* set vertical mv range */
            if (h->param->level_id >= 0x40) {
                h->min_mv_range[1] = -2048;
                h->max_mv_range[1] =  2047;
            } else if (h->param->level_id >= 0x20) {
                h->min_mv_range[1] = -1024;
                h->max_mv_range[1] =  1023;
            } else if (h->param->level_id >= 0x10) {
                h->min_mv_range[1] = -512;
                h->max_mv_range[1] =  511;
            } else {
                return -1;
            }
        }

        /* scale for field coding */
        if (h->param->InterlaceCodingOption == FIELD_CODING) {
            h->min_mv_range[1] >>= 1;
            h->max_mv_range[1] >>= 1;
        }

        return 0;
    }

    return -1;
}


/**
 * ---------------------------------------------------------------------------
 * Function   : determine the appropriate LevelID
 * ---------------------------------------------------------------------------
 */
static void encoder_decide_level_id(xavs2_param_t *param)
{
    const int tab_level_restriction[][5] = {
        /* LevelID, MaxWidth, MaxHeight, MaxFps, MaxKBps */
        { 0x00, 8192, 8192,   0,      0 },  // 禁止
        { 0x10,  352,  288,  15,   1500 },  // 2.0.15
        { 0x12,  352,  288,  30,   2000 },  // 2.0.30
        { 0x14,  352,  288,  60,   2500 },  // 2.0.60
        { 0x20,  720,  576,  30,   6000 },  // 4.0.30
        { 0x22,  720,  576,  60,  10000 },  // 4.0.60
        { 0x40, 2048, 1152,  30,  12000 },  // 6.0.30
        { 0x42, 2048, 1152,  30,  30000 },  // 6.0.60
        { 0x44, 2048, 1152,  60,  20000 },  // 6.0.120
        { 0x46, 2048, 1152,  60,  50000 },  // 6.2.120
        { 0x48, 2048, 1152, 120,  25000 },  // 6.0.120
        { 0x4A, 2048, 1152, 120, 100000 },  // 6.2.120
        { 0x50, 4090, 2304,  30,  25000 },  // 8.0.30
        { 0x52, 4090, 2304,  30,  25000 },  // 8.2.30
        { 0x54, 4090, 2304,  60,  40000 },  // 8.0.60
        { 0x56, 4090, 2304,  60, 160000 },  // 8.2.60
        { 0x58, 4090, 2304, 120,  60000 },  // 8.0.120
        { 0x5A, 4090, 2304, 120, 240000 },  // 8.2.120
        { 0x60, 8192, 4608,  30,  60000 },  // 10.0.30
        { 0x62, 8192, 4608,  30, 240000 },  // 10.2.30
        { 0x64, 8192, 4608,  60, 120000 },  // 10.0.60
        { 0x66, 8192, 4608,  60, 480000 },  // 10.2.60
        { 0x68, 8192, 4608, 120, 240000 },  // 10.0.120
        { 0x6A, 8192, 4608, 120, 800000 },  // 10.2.120
        { 0x00, 16384, 8192, 120, 8000000 },  // 禁止
    };

    int i = 1;
    int i_last_level = 0;

    for (; tab_level_restriction[i][4] != 0;) {
        /* 未开启码控时，设置为最大 */
        if (param->i_rc_method == 0 &&
            param->org_width <= tab_level_restriction[i_last_level][1] &&
            param->org_height <= tab_level_restriction[i_last_level][2] &&
            param->org_width <= tab_level_restriction[i][1] &&
            param->org_height <= tab_level_restriction[i][2] &&
            tab_level_restriction[i_last_level][1] < tab_level_restriction[i][1] &&
            tab_level_restriction[i_last_level][2] < tab_level_restriction[i][2]) {
            /* 码率控制未开启时，选择满足条件的分辨率下的最高档 */
            i = i_last_level;
            break;
        }
        /* 分辨率、帧率符合要求 */
        if (param->org_width <= tab_level_restriction[i][1] &&
            param->org_height <= tab_level_restriction[i][2] &&
            param->frame_rate <= tab_level_restriction[i][3]) {
            i_last_level = i;
            /* 比特率已设定，可根据最大码率设置LevelID */
            if (param->i_rc_method != 0 &&
                param->i_target_bitrate * 1.5 <= tab_level_restriction[i][4] * 1000 &&
                param->bitrate_upper <= tab_level_restriction[i][4] * 1000) {
                break;
            }
        }
        i++;
    }

    param->level_id = tab_level_restriction[i][0];
}

/* ---------------------------------------------------------------------------
 */
static INLINE
void encoder_check_fps_param(xavs2_param_t *param)
{
    float cur_fps = param->frame_rate;
    float min_error = 1000;
    int min_idx = 0;
    int i;
    for (i = 0; i < 8; i++) {
        float f_err = (float)fabs(FRAME_RATE[i] - cur_fps);
        if (f_err < min_error) {
            min_error = f_err;
            min_idx = i;
        }
    }
    param->frame_rate_code = min_idx + 1;
    param->frame_rate      = FRAME_RATE[min_idx];
    if (min_error >= 0.1) {
        xavs2_log(NULL, XAVS2_LOG_WARNING, "Framerate has been fixed: %.3f => %.3f\n",
                  cur_fps, param->frame_rate);
    }
}

/* ---------------------------------------------------------------------------
 */
int encoder_check_parameters(xavs2_param_t *param)
{
    int num_max_slice = ((param->org_height + (1 << param->lcu_bit_level) - 1) >> param->lcu_bit_level) >> 1;
    num_max_slice = XAVS2_MAX(2, num_max_slice);

    /* check number of threaded frames */
    if (param->i_frame_threads > MAX_PARALLEL_FRAMES) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "too many threaded frames : %d. increase MAX_PARALLEL_FRAMES (%d) and recompile.\n",
                  param->i_frame_threads, MAX_PARALLEL_FRAMES);
        return -1;
    }

    /* check slice number */
    if (param->slice_num > MAX_SLICES || param->slice_num > num_max_slice) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "too many slices : %d. exceeds MAX_SLICES (%d) or LcuRows/2 (%d).\n",
                  param->slice_num, MAX_SLICES, num_max_slice);
        return -1;
    }

    /* 多Slice下不能开启 cross slice loop filter，会影响并行效率
     * TODO: 后续可支持 */
    if (param->slice_num > 1 && param->b_cross_slice_loop_filter != FALSE) {
        xavs2_log(NULL, XAVS2_LOG_WARNING, "Un-supported cross slice loop filter, forcing not filtering\n");
        param->b_cross_slice_loop_filter = FALSE;
    }

    /* check frame rate */
    encoder_check_fps_param(param);

    /* check LCU size */
    if (param->lcu_bit_level < B16X16_IN_BIT || param->lcu_bit_level > B64X64_IN_BIT) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "MaxSizeInBit must be in 4..6 (LCU size: 16x16, 32x32, 64x64)\n");
        return -1;
    }

    /* check intra period */
    xavs2_log(NULL, XAVS2_LOG_DEBUG, "IntraPeriod { Min %d Max %d }, BFrames %d, OpenGOP %d\n",
              param->intra_period_min,
              param->intra_period_max,
              param->num_bframes,
              param->b_open_gop);
    if (param->intra_period_max == -1) {
        param->intra_period_max = (int)param->frame_rate;
    }
    if (param->intra_period_min == -1) {
        param->intra_period_min = param->intra_period_max;
    }
    if (param->intra_period_min > param->intra_period_max) {
        xavs2_log(NULL, XAVS2_LOG_WARNING, "IntraPeriod: swapped Min/Max\n");
        XAVS2_SWAP(param->intra_period_max, param->intra_period_min);
    }
    /* Only support GOP size divisible by 8 while using RA with openGOP */
    if (param->b_open_gop && param->num_bframes) {
        int period = param->intra_period_max / XAVS2_ABS(param->i_gop_size);
        if (param->intra_period_max % XAVS2_ABS(param->i_gop_size)) {
            param->intra_period_max = (period + 1) * XAVS2_ABS(param->i_gop_size);
            xavs2_log(NULL, XAVS2_LOG_WARNING, "IntraPeriodMax Fixed for OpenGOP => %d\n",
                      param->intra_period_max);
        }
    }
    if (param->profile_id == MAIN_PICTURE_PROFILE &&
        (param->intra_period_max != 1 || param->intra_period_min != 1)) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "MAIN picture file only supports intra picture coding!\n");
        return -1;
    }

    /* update profile id */
    if (param->sample_bit_depth == 8) {
        param->profile_id = MAIN_PROFILE;
    } else {
        if (param->profile_id != MAIN10_PROFILE && param->sample_bit_depth > 8) {
            xavs2_log(NULL, XAVS2_LOG_WARNING, "Forcing Main10 Profile for high bit-depth coding\n");
            param->profile_id = MAIN10_PROFILE;
        }
    }

    /* check bit depth */
    if (param->profile_id != MAIN_PROFILE) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Not Supported profile \"%d\", HIGH_BIT_DEPTH macro haven`t turn on!\n",
                  param->profile_id);
        return -1;
    }
    /* check LevelID */
    encoder_decide_level_id(param);
    if (param->level_id <= 0 || param->level_id > 0x6A) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Not Supported LevelID: %dx%d, %.3f fps, %d bps!\n",
                  param->org_width, param->org_height, param->frame_rate, param->i_target_bitrate);
        return -1;
    }

    /* check chroma format */
    if (param->chroma_format != CHROMA_420) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "invalid chroma format %d; Only YUV420 is supported for %s\n",
                  param->chroma_format, xavs2_avs2_standard_version);
        return -1;
    }

    /* check reference configuration */
    if (param->num_bframes >= XAVS2_MAX_GOP_SIZE) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "The number of successive B-frame is too big!\n");
        return -1;
    }
    if (param->num_bframes > 0 && param->num_bframes + 1 != XAVS2_ABS(param->i_gop_size)) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "The number of successive B-frame is wrong!\n");
        return -1;
    }
    if (rps_check_config(param) < 0) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Error found in RPS configuration!\n");
        return -1;
    }
    /* GOP parallel encoding */
    if (param->num_parallel_gop < 1) {
        param->num_parallel_gop = 1;
    } else if (param->num_parallel_gop > 1 && param->b_open_gop) {
        xavs2_log(NULL, XAVS2_LOG_WARNING, "Only ClosedGOP can be utilized with GOP parallel encoding\n");
        param->b_open_gop = FALSE;
    }

    /* check preset level */
    if (param->preset_level < 0 || param->preset_level > 9) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Error input parameter preset_level, check configuration file\n");
        return -1;
    } else {
        if (param->is_preset_configured == FALSE) {
            /* modify configurations according to the input preset level */
            parse_preset_level(param, param->preset_level);
        }
    }

    /* check QP */
    if (param->i_initial_qp > MAX_QP || param->i_initial_qp < MIN_QP) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Error input parameter quant_0, check configuration file\n");
        return -1;
    }
    if (param->i_initial_qp < 25 + 8 * (param->sample_bit_depth - 8)) {
        xavs2_log(NULL, XAVS2_LOG_WARNING, "A small QP is configured: QP: %d, EncodingBitDepth: %d, Suggested QP: >=%d\n",
                  param->i_initial_qp, param->sample_bit_depth, 25 + 8 * (param->sample_bit_depth - 8));
    }
    if (param->i_max_qp > 63 + (param->sample_bit_depth - 8) * 8) {
        xavs2_log(NULL, XAVS2_LOG_WARNING, "A too large max QP is configured: QP: %d, EncodingBitDepth: %d, Available QP: <=%d\n",
                  param->i_max_qp, param->sample_bit_depth, 63 + 8 * (param->sample_bit_depth - 8));
        param->i_max_qp = 63 + (param->sample_bit_depth - 8) * 8;
    }
    if (param->i_min_qp < 0) {
        param->i_min_qp = 0;
    }
    param->i_initial_qp = XAVS2_CLIP3(param->i_min_qp, param->i_max_qp, param->i_initial_qp);

    /* check LCU level */
    if (param->lcu_bit_level > 6 || param->lcu_bit_level < 3) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Error input parameter MaxSizeInBit, check configuration file\n");
        return -1;
    }

    /* check range of filter offsets */
    if (param->alpha_c_offset > 8 || param->alpha_c_offset < -8) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Error input parameter LFAlphaC0Offset, check configuration file\n");
        return -1;
    }
    if (param->beta_offset > 8 || param->beta_offset < -8) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Error input parameter LFBetaOffset, check configuration file\n");
        return -1;
    }

    /* check ALF configuration */
    if (param->i_frame_threads != 1 && param->enable_alf != 0) {
        param->enable_alf = 0;
        xavs2_log(NULL, XAVS2_LOG_WARNING, "ALF disabled since frame parallel encoding is enabled.\n");
    }

    /* FIXME: set bitrate (lower and upper) */
    param->bitrate_lower = (param->i_target_bitrate / 400) & 0x3FFFF;   /* lower 18 bits */
    param->bitrate_upper = (param->i_target_bitrate / 400) >> 18;       /* upper 12 bits */

    /* set for field coding */
    if (param->InterlaceCodingOption == FIELD_CODING) {
        param->org_height   = param->org_height   >> 1;
        param->intra_period_max = param->intra_period_max << 1;
        param->intra_period_min = param->intra_period_min << 1;
    }

    /* low delay? */
    if (param->num_bframes == 0) {
        param->low_delay = TRUE;
    } else {
        param->low_delay = FALSE;
    }

    /* Rate-Control */
#if !ENABLE_RATE_CONTROL_CU
    if (param->i_rc_method == XAVS2_RC_CBR_SCU) {
        xavs2_log(NULL, XAVS2_LOG_WARNING, "Rate Control with CU level control disabled in this version.\n");
        param->i_rc_method = XAVS2_RC_CBR_FRM;
    }
#endif

    if (param->i_rc_method == XAVS2_RC_CBR_SCU) {
        param->fixed_picture_qp = FALSE;
    } else {
        param->fixed_picture_qp = TRUE;
    }

    /* consistency check num_max_ref */
    if (param->num_max_ref < 1) {
        param->num_max_ref = 1;
    }

    /* enable TDRDO? TDRDO is only just for low delay */
    if (param->num_bframes != 0 || param->intra_period_min > 0) {
        param->enable_tdrdo = 0;
    }

    /* set display properties */
    // param->display_horizontal_size  = param->org_width;
    // param->display_vertical_size    = param->org_height;
    param->sample_precision         = ((param->input_sample_bit_depth - 6) / 2);
    param->aspect_ratio_information = 1;

#if !ENABLE_WQUANT
    /* weighting quantization */
    param->enable_wquant           = 0;   /* disable */
#endif

    return 0;
}

/* ---------------------------------------------------------------------------
 * assign pointers for all coding tree units (till 4x4 CU)
 */
static void build_coding_tree(xavs2_t *h, cu_t *p_cu, int idx_zorder, int i_level, int i_pos_x, int i_pos_y)
{
    int i;
    int idx_cu_bfs = 0;

    p_cu->i_size          = 1 << i_level;
    p_cu->cu_info.i_level = (int8_t)i_level;
    p_cu->i_pos_x        = i_pos_x;
    p_cu->i_pos_y        = i_pos_y;
    p_cu->in_lcu_edge    = ((i_pos_y != 0) << 1) + (i_pos_x != 0);
    p_cu->idx_zorder     = (int8_t)idx_zorder;

    idx_cu_bfs = tab_cu_bfs_order[i_level - MIN_CU_SIZE_IN_BIT];
    idx_cu_bfs += (i_pos_y >> i_level) * (MAX_CU_SIZE >> i_level) + (i_pos_x >> i_level);
    p_cu->idx_cu_bfs = (int8_t)idx_cu_bfs;

    if (i_level > B8X8_IN_BIT) {
        int num_parts = 1 << ((i_level - B16X16_IN_BIT) << 1);
        i_level--;
        for (i = 0; i < 4; i++) {
            p_cu->sub_cu[i] = &h->lcu.all_cu[h->lcu.i_scu_xy++];

            i_pos_x = p_cu->i_pos_x + ((i &  1) << i_level);
            i_pos_y = p_cu->i_pos_y + ((i >> 1) << i_level);
            build_coding_tree(h, p_cu->sub_cu[i], idx_zorder + i * num_parts, i_level, i_pos_x, i_pos_y);
        }
    } else {
        for (i = 0; i < 4; i++) {
            p_cu->sub_cu[i] = NULL;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static
xavs2_t *encoder_create_frame_context(const xavs2_param_t *param, int idx_frm_encoder)
{
    const int num_slices = param->slice_num;
    xavs2_t *h = NULL;
    int frame_w  = ((param->org_width  + MIN_CU_SIZE - 1) >> MIN_CU_SIZE_IN_BIT) << MIN_CU_SIZE_IN_BIT;
    int frame_h  = ((param->org_height + MIN_CU_SIZE - 1) >> MIN_CU_SIZE_IN_BIT) << MIN_CU_SIZE_IN_BIT;
    int size_lcu = 1 << param->lcu_bit_level;       /* size of a LCU (largest coding unit) */
    int w_in_lcu = (frame_w + size_lcu - 1) >> param->lcu_bit_level;
    int h_in_lcu = (frame_h + size_lcu - 1) >> param->lcu_bit_level;
    int w_in_scu = frame_w >> MIN_CU_SIZE_IN_BIT;
    int h_in_scu = frame_h >> MIN_CU_SIZE_IN_BIT;
    int w_in_4x4 = frame_w >> MIN_PU_SIZE_IN_BIT;
    int h_in_4x4 = frame_h >> MIN_PU_SIZE_IN_BIT;
    int bs_size  = frame_w * frame_h * 2;
    int ipm_size = (w_in_4x4 + 16) * ((size_lcu >> MIN_PU_SIZE_IN_BIT) + 1);
    int size_4x4 = w_in_4x4 * h_in_4x4;
    int qpel_frame_size = (frame_w + 2 * XAVS2_PAD) * (frame_h + 2 * XAVS2_PAD);
    int info_size = sizeof(frame_info_t) + h_in_lcu * sizeof(row_info_t) + w_in_lcu * h_in_lcu * sizeof(lcu_info_t);

    int size_sao_stats = w_in_lcu * h_in_lcu * sizeof(SAOStatData[NUM_SAO_COMPONENTS][NUM_SAO_NEW_TYPES]);
    int size_sao_param = w_in_lcu * h_in_lcu * sizeof(SAOBlkParam[NUM_SAO_COMPONENTS]);
    int size_sao_onoff = h_in_lcu * sizeof(int[NUM_SAO_COMPONENTS]);

    size_t size_alf = alf_get_buffer_size(param);
    int frame_size_in_scu = w_in_scu * h_in_scu;
    int num_me_bytes = (w_in_4x4 * h_in_4x4)* sizeof(dist_t[MAX_INTER_MODES][MAX_REFS]);
    size_t size_extra_frame_buffer = 0;
    int i, j;
    int scu_xy = 0;
    cu_info_t *p_cu_info;
    size_t mem_size = 0;
    uint8_t *mem_base;

    num_me_bytes = (num_me_bytes + 255) >> 8 << 8;    /* align number of bytes to 256 */
    qpel_frame_size = (qpel_frame_size + CACHE_LINE_SIZE - 1) & ~(CACHE_LINE_SIZE - 1);
    size_extra_frame_buffer = (param->enable_tdrdo + param->enable_sao + param->enable_alf) * xavs2_frame_buffer_size(param, FT_TEMP);

    /* compute the space size and alloc buffer */
    mem_size = sizeof(xavs2_t)                       +  /* xavs2_t */
               sizeof(nal_t)   * (MAX_SLICES + 6)    +  /* all nal units */
               sizeof(uint8_t) * XAVS2_BS_HEAD_LEN   +  /* bitstream buffer (frame header only) */
               sizeof(uint8_t) * bs_size             +  /* bitstream buffer for all slices */
               sizeof(slice_t) * MAX_SLICES          +  /* slice array */
               sizeof(pel_t)   * (frame_w * 2) * num_slices + /* buffer for intra_border */
               sizeof(uint8_t) * w_in_scu * 32 * num_slices + /* buffer for edge filter flag (of one LCU row) */
               sizeof(int8_t)  * ipm_size      * num_slices + /* intra prediction mode buffer */
               sizeof(int8_t)  * size_4x4            +  /* inter prediction direction */
               sizeof(int8_t)  * size_4x4 * 2        +  /* reference frames */
               sizeof(mv_t)    * size_4x4 * 2        +  /* reference motion vectors */
               CACHE_LINE_SIZE * (MAX_SLICES + 32);
    mem_size +=
        qpel_frame_size * 3 * sizeof(mct_t)   +  /* temporary buffer for 1/4 interpolation: a,1,b */
        xavs2_me_get_buf_size(param)          +  /* buffers in me module */
        info_size                             +  /* the frame info structure */
        frame_size_in_scu * sizeof(cu_info_t) +  /* CU data */
        num_me_bytes                          +  /* Motion Estimation */
        w_in_lcu * h_in_lcu * sizeof(int8_t)  +  /* CTU slice index */
        size_extra_frame_buffer               +  /* extra frame buffer: TDRDO, SAO, ALF */

        size_sao_stats + CACHE_LINE_SIZE      +  /* SAO stat data */
        size_sao_param + CACHE_LINE_SIZE      +  /* SAO parameters */
        size_sao_onoff + CACHE_LINE_SIZE      +  /* SAO on/off number of LCU row */

        size_alf + CACHE_LINE_SIZE            +  /* ALF encoder contexts */
        CACHE_LINE_SIZE * 30;                    /* used for align buffer */

    /* alloc memory space */
    mem_size = ((mem_size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;
    CHECKED_MALLOC(mem_base, uint8_t *, mem_size);

    /* assign handle pointer of the xavs2 encoder */
    h = (xavs2_t *)mem_base;
    memset(h, 0, sizeof(xavs2_t));
    mem_base += sizeof(xavs2_t);
    ALIGN_POINTER(mem_base);          /* align pointer */

    /* init log module */
    h->module_log.i_log_level = param->i_log_level;
    sprintf(h->module_log.module_name, "Enc[%2d] %06llx", idx_frm_encoder, (uintptr_t)(h));

    /* copy the input parameters */
    h->param = param;

    /* const properties */
    h->i_width           = frame_w;
    h->i_height          = frame_h;
    h->i_width_in_lcu    = w_in_lcu;
    h->i_height_in_lcu   = h_in_lcu;
    h->i_width_in_mincu  = w_in_scu;
    h->i_height_in_mincu = h_in_scu;
    h->i_width_in_minpu  = w_in_4x4;
    h->i_height_in_minpu = h_in_4x4;

    h->framerate         = h->param->frame_rate;

    h->i_lcu_level       = h->param->lcu_bit_level;
    h->i_scu_level       = h->param->scu_bit_level;
    h->i_chroma_v_shift  = h->param->chroma_format == CHROMA_420;
    h->i_max_ref         = h->param->num_max_ref;
    h->b_progressive     = (bool_t)h->param->progressive_frame;
    h->b_field_sequence  = (h->param->InterlaceCodingOption == FIELD_CODING);

    /* set table which indicates numbers of intra prediction modes for RDO */
    for (i = 0; i < MAX_CU_SIZE_IN_BIT; i++) {
        h->tab_num_intra_rdo[i] = 1;                 /* this will later be set according to the preset level */
    }
    h->num_rdo_intra_chroma = NUM_INTRA_MODE_CHROMA;

    /* -------------------------------------------------------------
     * assign buffer pointers of xavs2 encoder
     */

    /* point to all nal units */
    h->p_nal  = (nal_t *)mem_base;
    mem_base += sizeof(nal_t) * (MAX_SLICES + 6);
    ALIGN_POINTER(mem_base);    /* align pointer */

    /* bitstream buffer (frame header) */
    h->p_bs_buf_header = mem_base;
    h->i_bs_buf_header = sizeof(uint8_t) * XAVS2_BS_HEAD_LEN;
    mem_base          += sizeof(uint8_t) * XAVS2_BS_HEAD_LEN;
    ALIGN_POINTER(mem_base);    /* align pointer */

    /* bitstream buffer for all slices */
    h->p_bs_buf_slice = mem_base;
    h->i_bs_buf_slice = sizeof(uint8_t) * bs_size;
    mem_base         += sizeof(uint8_t) * bs_size;
    ALIGN_POINTER(mem_base);    /* align pointer */

    /* slice array */
    for (i = 0; i < num_slices; i++) {
        slice_t *p_slice = (slice_t *)mem_base;
        h->slices[i] = p_slice;
        mem_base    += sizeof(slice_t);
        ALIGN_POINTER(mem_base);    /* align pointer */

        /* intra prediction mode buffer */
        p_slice->slice_ipredmode  = (int8_t *)mem_base;
        mem_base                 += sizeof(int8_t) * ipm_size;
        p_slice->slice_ipredmode += (h->i_width_in_minpu + 16) + 16;
        ALIGN_POINTER(mem_base);    /* align pointer */

        /* assign pointer to intra_border buffer */
        p_slice->slice_intra_border[0] = (pel_t *)mem_base;
        mem_base          += h->i_width * sizeof(pel_t);
        ALIGN_POINTER(mem_base);
        p_slice->slice_intra_border[1] = (pel_t *)mem_base;
        mem_base          += (h->i_width / 2) * sizeof(pel_t);
        ALIGN_POINTER(mem_base);
        p_slice->slice_intra_border[2] = (pel_t *)mem_base;
        mem_base          += (h->i_width / 2) * sizeof(pel_t);
        ALIGN_POINTER(mem_base);

        /* buffer for edge filter flag (of one LCU row) */
        p_slice->slice_deblock_flag[0] = (uint8_t *)mem_base;
        mem_base            += h->i_width_in_mincu * (MAX_CU_SIZE / MIN_PU_SIZE) * sizeof(uint8_t);
        p_slice->slice_deblock_flag[1] = (uint8_t *)mem_base;
        mem_base            += h->i_width_in_mincu * (MAX_CU_SIZE / MIN_PU_SIZE) * sizeof(uint8_t);
        ALIGN_POINTER(mem_base);
    }

    slice_init_bufer(h, h->slices[0]);

    /* -------------------------------------------------------------
     *      fenc                fdec
     *      Y Y Y Y             Y Y Y Y
     *      Y Y Y Y             Y Y Y Y
     *      Y Y Y Y             Y Y Y Y
     *      Y Y Y Y             Y Y Y Y
     *      U U V V             U U V V
     *      U U V V             U U V V
     */

    /* assign pointers for p_fenc (Y/U/V pointers) */
    h->lcu.p_fenc[0] = h->lcu.fenc_buf;
    h->lcu.p_fenc[1] = h->lcu.fenc_buf + FENC_STRIDE * MAX_CU_SIZE;
    h->lcu.p_fenc[2] = h->lcu.fenc_buf + FENC_STRIDE * MAX_CU_SIZE + (FENC_STRIDE / 2);

    /* assign pointers for p_fdec (Y/U/V pointers) */
    h->lcu.p_fdec[0] = h->lcu.fdec_buf;
    h->lcu.p_fdec[1] = h->lcu.fdec_buf + FDEC_STRIDE * MAX_CU_SIZE;
    h->lcu.p_fdec[2] = h->lcu.fdec_buf + FDEC_STRIDE * MAX_CU_SIZE + (FDEC_STRIDE / 2);

    /* slice index of CTUs */
    h->lcu_slice_idx = (int8_t *)mem_base;
    mem_base += w_in_lcu * h_in_lcu * sizeof(int8_t);
    ALIGN_POINTER(mem_base);    /* align pointer */

    /* inter prediction mode */
    h->dir_pred = (int8_t *)mem_base;
    mem_base += sizeof(int8_t) * size_4x4;
    ALIGN_POINTER(mem_base);    /* align pointer */

    /* reference frames */
    h->fwd_1st_ref = (int8_t *)mem_base;
    mem_base      += sizeof(int8_t) * size_4x4;
    ALIGN_POINTER(mem_base);    /* align pointer */
    h->bwd_2nd_ref = (int8_t *)mem_base;
    mem_base      += sizeof(int8_t) * size_4x4;
    ALIGN_POINTER(mem_base);    /* align pointer */

    /* reference motion vectors */
    h->fwd_1st_mv = (mv_t *)mem_base;
    mem_base     += sizeof(mv_t) * size_4x4;
    ALIGN_POINTER(mem_base);    /* align pointer */
    h->bwd_2nd_mv = (mv_t *)mem_base;
    mem_base     += sizeof(mv_t) * size_4x4;
    ALIGN_POINTER(mem_base);    /* align pointer */

    /* temporary buffer for 1/4 interpolation: a,1,b, alone buffer */
    h->img4Y_tmp[0] = (mct_t *)mem_base;
    h->img4Y_tmp[1] = h->img4Y_tmp[0] + qpel_frame_size;
    h->img4Y_tmp[2] = h->img4Y_tmp[0] + qpel_frame_size * 2;
    mem_base       += qpel_frame_size * 3 * sizeof(mct_t);
    ALIGN_POINTER(mem_base);

    /* SAO data */
    h->sao_stat_datas = (SAOStatData (*)[NUM_SAO_COMPONENTS][NUM_SAO_NEW_TYPES])mem_base;
    memset(h->sao_stat_datas[0], 0, size_sao_stats);
    mem_base += size_sao_stats;
    ALIGN_POINTER(mem_base);

    h->sao_blk_params = (SAOBlkParam (*)[NUM_SAO_COMPONENTS])mem_base;
    memset(h->sao_blk_params[0], 0, size_sao_param);
    mem_base += size_sao_param;
    ALIGN_POINTER(mem_base);

    h->num_sao_lcu_off = (int (*)[NUM_SAO_COMPONENTS])mem_base;
    memset(h->num_sao_lcu_off[0], 0, size_sao_onoff);
    mem_base += size_sao_onoff;
    ALIGN_POINTER(mem_base);


    /* init memory space in me module */
    xavs2_me_init(h, &mem_base);

    /* allocate frame_info_t (one for each frame context) */
    h->frameinfo = (frame_info_t *)mem_base;
    mem_base    += sizeof(frame_info_t);
    ALIGN_POINTER(mem_base);    /* align pointer */

    h->frameinfo->rows = (row_info_t *)mem_base;
    mem_base          += sizeof(row_info_t) * h_in_lcu;
    ALIGN_POINTER(mem_base);    /* align pointer */

    /* set available tables */
    set_available_tables(h);

    /* assign pointers for all coding tree units */
    h->lcu.p_ctu    = &h->lcu.all_cu[0];
    h->lcu.i_scu_xy = 1;        // borrowed
    build_coding_tree(h, h->lcu.p_ctu, 0, h->i_lcu_level, 0, 0);
    h->lcu.i_scu_xy = 0;        // reset

    /* set row info */
    for (i = 0; i < h_in_lcu; i++) {
        row_info_t *row = &h->frameinfo->rows[i];

        row->h     = 0;
        row->row   = i;
        row->coded = -1;
        row->lcus  = (lcu_info_t *)mem_base;
        mem_base  += sizeof(lcu_info_t) * w_in_lcu;

        if (xavs2_thread_mutex_init(&row->mutex, NULL)) {
            goto fail;
        }

        if (xavs2_thread_cond_init(&row->cond, NULL)) {
            goto fail;
        }
    }

    /* check memory size */
    ALIGN_POINTER(mem_base);    /* align pointer */

    /* -------------------------------------------------------------
     * allocate other alone spaces for xavs2 encoder
     */

    h->cu_info = (cu_info_t *)mem_base;
    mem_base  += frame_size_in_scu * sizeof(cu_info_t);
    ALIGN_POINTER(mem_base);

    p_cu_info = h->cu_info;
    for (j = 0; j < h_in_scu; j++) {
        for (i = 0; i < w_in_scu; i++) {
            scu_xy++;
            p_cu_info->i_scu_x = i;
            p_cu_info->i_scu_y = j;
            p_cu_info++;
        }
    }

    /* motion estimation buffer */
    h->all_mincost = (dist_t(*)[MAX_INTER_MODES][MAX_REFS])mem_base;
    mem_base += num_me_bytes;
    ALIGN_POINTER(mem_base);

    // allocate memory for current frame
    if (h->param->enable_tdrdo) {
        h->img_luma_pre = xavs2_frame_new(h, &mem_base, FT_TEMP);
        ALIGN_POINTER(mem_base);
    } else {
        h->img_luma_pre = NULL;
    }

    if (h->param->enable_sao) {
        h->img_sao = xavs2_frame_new(h, &mem_base, FT_TEMP);
        ALIGN_POINTER(mem_base);
    } else {
        h->img_sao = NULL;
    }

    if (h->param->enable_alf) {
        h->img_alf = xavs2_frame_new(h, &mem_base, FT_TEMP);
        ALIGN_POINTER(mem_base);
        alf_init_buffer(h, mem_base);
        mem_base += size_alf;
        ALIGN_POINTER(mem_base);
    } else {
        h->img_alf = NULL;
    }

    if ((uintptr_t)(h) + mem_size < (uintptr_t)(mem_base)) {
        /* malloc size allocation error: no enough memory */
        goto fail;
    }
    /* -------------------------------------------------------------
     * init other properties/modules for xavs2 encoder
     */

    /* init all slices */
    xavs2_slices_init(h);

#if ENABLE_WQUANT
    /* adaptive frequency weighting quantization */
    if (h->param->enable_wquant) {
        xavs2_wq_init_seq_quant_param(h);
    }
#endif

    return h;

fail:
    return NULL;
}

/* ---------------------------------------------------------------------------
 */
static void encoder_destroy_frame_context(xavs2_t *h)
{
    int i;
    assert(h != NULL);
    assert(h->task_type == XAVS2_TASK_FRAME);

    h->img_luma_pre = NULL;
    h->img_sao      = NULL;
    h->img_alf = NULL;
    h->enc_alf = NULL;

    /* free frame_info_t & row_info_t */
    if (h->frameinfo) {
        for (i = 0; i < h->i_height_in_lcu; i++) {
            /* free a row */
            row_info_t *row = &h->frameinfo->rows[i];
            if (row) {
                xavs2_thread_mutex_destroy(&row->mutex);
                xavs2_thread_cond_destroy(&row->cond);
            }
        }
    }

    xavs2_free(h);
}

/* ---------------------------------------------------------------------------
 * allocate memory for multiple threads (slices/frames parallel)
 */
int encoder_contexts_init(xavs2_t *h, xavs2_handler_t *h_mgr)
{
    int i;

    /* -------------------------------------------------------------
     * build lcu row encoding contexts */
    if (h_mgr->num_row_contexts > 1) {
        CHECKED_MALLOC(h_mgr->row_contexts, xavs2_t *, h_mgr->num_row_contexts * sizeof(xavs2_t));

        for (i = 0; i < h_mgr->num_row_contexts; i++) {
            xavs2_t *h_row_coder = &h_mgr->row_contexts[i];

            memcpy(&h_row_coder->communal_vars_1, &h->communal_vars_1,
                   (uint8_t *)&h->communal_vars_2 - (uint8_t *)&h->communal_vars_1);

            /* identify ourself */
            h_row_coder->task_type = XAVS2_TASK_ROW;

            /* we are free */
            h_row_coder->i_aec_frm = -1;

            /* assign pointers for all coding tree units */
            h_row_coder->lcu.p_ctu     = &h_row_coder->lcu.all_cu[0];
            h_row_coder->lcu.i_scu_xy  = 1;     // borrowed
            build_coding_tree(h_row_coder, h_row_coder->lcu.p_ctu, 0, h_row_coder->i_lcu_level, 0, 0);
            h_row_coder->lcu.i_scu_xy  = 0;     // reset

            /* assign pointers for p_fenc (Y/U/V pointers) */
            h_row_coder->lcu.p_fenc[0] = h_row_coder->lcu.fenc_buf;
            h_row_coder->lcu.p_fenc[1] = h_row_coder->lcu.fenc_buf + FENC_STRIDE * MAX_CU_SIZE;
            h_row_coder->lcu.p_fenc[2] = h_row_coder->lcu.fenc_buf + FENC_STRIDE * MAX_CU_SIZE + FENC_STRIDE / 2;

            /* assign pointers for p_fdec (Y/U/V pointers) */
            h_row_coder->lcu.p_fdec[0] = h_row_coder->lcu.fdec_buf;
            h_row_coder->lcu.p_fdec[1] = h_row_coder->lcu.fdec_buf + FDEC_STRIDE * MAX_CU_SIZE;
            h_row_coder->lcu.p_fdec[2] = h_row_coder->lcu.fdec_buf + FDEC_STRIDE * MAX_CU_SIZE + FDEC_STRIDE / 2;
        }
    }

    /* -------------------------------------------------------------
     * build frame encoding contexts */
    h_mgr->frm_contexts[0] = h; /* context 0 is the main encoder handle */
    for (i = 1; i < h_mgr->i_frm_threads; i++) {
        if ((h_mgr->frm_contexts[i] = encoder_create_frame_context(h->param, i)) == 0) {
            goto fail;
        }

        memcpy(&h_mgr->frm_contexts[i]->communal_vars_1, &h->communal_vars_1,
               (uint8_t *)&h->communal_vars_2 - (uint8_t *)&h->communal_vars_1);
    }

    return 0;

fail:
    return -1;
}

/* ---------------------------------------------------------------------------
 * free all contexts except for the main context : xavs2_handler_t::contexts[0]
 */
static void encoder_contexts_free(xavs2_handler_t *h_mgr)
{
    int i = 0;

    /* free all row contexts */
    if (h_mgr->row_contexts != NULL) {
        xavs2_free(h_mgr->row_contexts);
        h_mgr->row_contexts = NULL;
    }

    /* free frame contexts */
    for (i = 0; i < h_mgr->i_frm_threads; i++) {
        /* free the xavs2 encoder */
        if (h_mgr->frm_contexts[i] != NULL) {
            encoder_destroy_frame_context(h_mgr->frm_contexts[i]);
            h_mgr->frm_contexts[i] = NULL;
        }
    }
}

/* ---------------------------------------------------------------------------
 * free the task manager
 */
void encoder_task_manager_free(xavs2_handler_t *h_mgr)
{
    int i = 0;

    assert(h_mgr != NULL);

    /* signal to exit */
    h_mgr->i_exit_flag = XAVS2_EXIT_THREAD;

    /* wait until the aec thread finish its job */
    xavs2_thread_cond_signal(&h_mgr->cond[SIG_FRM_CONTEXT_ALLOCATED]);

    /* destroy the AEC thread pool */
    if (h_mgr->threadpool_aec != NULL) {
        xavs2_threadpool_delete(h_mgr->threadpool_aec);
    }

    /* wait until the output thread finish its job */
    xavs2_thread_cond_signal(&h_mgr->cond[SIG_FRM_AEC_COMPLETED]);

    xavs2_thread_mutex_destroy(&h_mgr->mutex);

    for (i = 0; i < SIG_COUNT; i++) {
        xavs2_thread_cond_destroy(&h_mgr->cond[i]);
    }

    /* destroy the RDO thread pool */
    if (h_mgr->i_frm_threads > 1 || h_mgr->i_row_threads > 1) {
        xavs2_threadpool_delete(h_mgr->threadpool_rdo);
    }

#if XAVS2_STAT
    /* report everything */
    encoder_report_stat_info(h_mgr->p_coder);
#endif

    /* destroy TDRDO */
    tdrdo_destroy(h_mgr->td_rdo);

    /* destroy the rate control */
    xavs2_rc_destroy(h_mgr->rate_control);

#if XAVS2_DUMP_REC
    /* close rec file */
    if (h_mgr->h_rec_file) {
        fclose(h_mgr->h_rec_file);
        h_mgr->h_rec_file = NULL;
    }
#endif
    /* free contexts */
    encoder_contexts_free(h_mgr);

    /* free memory of all lists */
    destroy_all_lists(h_mgr);

    frame_buffer_destroy(h_mgr, &h_mgr->dpb);
}


/* ---------------------------------------------------------------------------
 * reset the decoding frame
 */
static void init_decoding_frame(xavs2_t *h)
{
#if SAVE_CU_INFO
    int frame_size_in_mincu = h->i_width_in_mincu * h->i_height_in_mincu;
#endif
    int frame_size_in_mvstore = ((h->i_width_in_minpu + 3) >> 2) * ((h->i_height_in_minpu + 3) >> 2);
    int i;

    /* set frame properties */
    h->fdec->i_frame         = h->fenc->i_frame;
    h->fdec->i_frm_type      = h->fenc->i_frm_type;
    h->fdec->i_pts           = h->fenc->i_pts;
    h->fdec->i_dts           = h->fenc->i_dts;
    h->fdec->i_frm_coi       = h->fenc->i_frm_coi;
    h->fdec->i_gop_idr_coi   = h->fenc->i_gop_idr_coi;
    h->fdec->rps.temporal_id = h->i_layer;

    if (h->b_field_sequence == 0) {
        h->fdec->i_frm_poc = h->fdec->i_frame << 1;
    } else {
        assert(0);  // field sequences
    }

    /* set ref_dpoc */
    for (i = 0; i < sizeof(h->fdec->ref_dpoc) / sizeof(h->fdec->ref_dpoc[0]); i++) {
        h->fdec->ref_dpoc[i] = MULTIx2;
        h->fdec->ref_dpoc_multi[i] = 1;
    }

    if (h->i_type == SLICE_TYPE_B) {
        h->fdec->ref_dpoc[B_BWD] = ((h->fref[B_BWD]->i_frm_poc - h->fdec->i_frm_poc) + 512) & 511;
        h->fdec->ref_dpoc[B_FWD] = ((h->fdec->i_frm_poc - h->fref[B_FWD]->i_frm_poc) + 512) & 511;

        h->fdec->ref_dpoc_multi[B_BWD] = MULTI / h->fdec->ref_dpoc[B_BWD];
        h->fdec->ref_dpoc_multi[B_FWD] = MULTI / h->fdec->ref_dpoc[B_FWD];
    } else if (h->i_type != SLICE_TYPE_I) {    /* F/P frame */
        for (i = 0; i < h->i_ref; i++) {
            h->fdec->ref_dpoc[i] = (h->fdec->i_frm_poc - h->fref[i]->i_frm_poc + 512) & 511;
            h->fdec->ref_dpoc_multi[i] = MULTI / h->fdec->ref_dpoc[i];
        }
    }

    /* reset mv buffer */
    g_funcs.fast_memzero(h->fdec->pu_mv, frame_size_in_mvstore * sizeof(mv_t));

    /* reset ref buffer */
    g_funcs.fast_memset(h->fdec->pu_ref, INVALID_REF, frame_size_in_mvstore * sizeof(int8_t));

#if SAVE_CU_INFO
    /* reset CU BitSize buffer */
    g_funcs.fast_memzero(h->fdec->cu_level, frame_size_in_mincu * sizeof(int8_t));

    /* reset CU type buffer */
    g_funcs.fast_memzero(h->fdec->cu_mode, frame_size_in_mincu * sizeof(int8_t));

    /* reset CU cbp buffer */
    g_funcs.fast_memzero(h->fdec->cu_cbp, frame_size_in_mincu * sizeof(int8_t));
#endif
}

/* ---------------------------------------------------------------------------
 * init function handles
 */
static void encoder_init_func_handles(xavs2_t *h)
{
    /* set some function handles according option or preset level */
    if (h->param->enable_hadamard) {
        g_funcs.pixf.intra_cmp = g_funcs.pixf.satd;
        g_funcs.pixf.fpel_cmp  = g_funcs.pixf.satd;
    } else {
        g_funcs.pixf.intra_cmp = g_funcs.pixf.sad;
        g_funcs.pixf.fpel_cmp  = g_funcs.pixf.sad;
    }
}

/**
 * ===========================================================================
 * encoder function defines
 * ===========================================================================
 */

/**
 * ---------------------------------------------------------------------------
 * Function   : create and initialize a xavs2 video encoder
 * Parameters :
 *      [in ] : param   - pointer to struct xavs2_param_t
 *            : h_mgr   - pointer to top handler
 *      [out] : none
 * Return     : handle of xavs2 encoder, none zero for success, otherwise false
 * ---------------------------------------------------------------------------
 */
xavs2_t *encoder_open(xavs2_param_t *param, xavs2_handler_t *h_mgr)
{
    xavs2_t *h = NULL;

#if XAVS2_STAT
    /* show header info */
    encoder_show_head_info(param);
#endif
    /* decide ultimaete coding parameters by preset level */
    decide_ultimate_paramters(param);

    /* init frame context */
    if ((h = encoder_create_frame_context(param, 0)) == NULL) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "create frame context fail\n");
        goto fail;
    }

    /* set fast algorithms according to the input preset level */
    encoder_set_fast_algorithms(h);

    /* init top handler */
    h->h_top       = h_mgr;
    h->rc          = h_mgr->rate_control;
    h->td_rdo      = h_mgr->td_rdo;
    h->task_type   = XAVS2_TASK_FRAME;      /* we are a frame task */
    h->task_status = XAVS2_TASK_FREE;       /* ready for encoding */
    h->i_aec_frm   = -1;                    /* ready to be allocated */

    h_mgr->frm_contexts[0] = h;   /* point to the xavs2_t handle */

#if XAVS2_TRACE
    xavs2_trace_init(h->param);    /* init trace */
#endif



    if (encoder_decide_mv_range(h) < 0) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "check mv range fail\n");
        goto fail;
    }

    encoder_init_func_handles(h);     /* init function handles */

    xavs2_init_valid_mode_table(h);

    xavs2_me_init_umh_threshold(h, h->umh_bsize, h->param->i_initial_qp + 1);

#if CTRL_OPT_AEC
    init_aec_context_tab();
#endif

    /* parse RPS */
    rps_set_picture_reorder_delay(h);

#if XAVS2_STAT
    encoder_show_frame_info_tab(h, h_mgr);
#endif

    /* init LCU row order */
    slice_lcu_row_order_init(h);

    return h;

fail:
    encoder_close(h_mgr);

    return 0;
}

/**
 * ---------------------------------------------------------------------------
 * Function   : init frame coding (init bitstream and picture header)
 * Parameters :
 *      [in ] : h   - pointer to struct xavs2_t, the xavs2 encoder
 *      [out] : none
 * Return     : the length of bitstream
 * ---------------------------------------------------------------------------
 */
static INLINE
void xavs2e_frame_coding_init(xavs2_t *h)
{
    /* prepare to encode -------------------------------------------
     */
#if ENABLE_WQUANT
    if (h->param->intra_period_min != 0 && h->i_type == SLICE_TYPE_I) {
        // adaptive frequency weighting quantization
        if (h->param->enable_wquant) {
            xavs2_wq_init_seq_quant_param(h);
        }
    }

    if (h->param->enable_wquant && h->param->PicWQEnable) {
        xavs2_wq_init_pic_quant_param(h);
        xavs2_wq_update_pic_matrix(h);
    }
#endif

    /* frame picture? */
    if (h->param->InterlaceCodingOption == FIELD_CODING) {
        h->b_top_field = (h->fenc->i_frm_poc & 1) == 0;
    }

    /* get QP to encode --------------------------------------------
     */

    /* enable TD-RDO? */
    if (h->td_rdo != NULL) {
        tdrdo_frame_start(h);
    }

    /* get frame level qp */
    if (h->param->i_rc_method != XAVS2_RC_CQP) {
        int new_qp = h->i_qp;
        new_qp = xavs2_rc_get_frame_qp(h, h->fenc->i_frame, h->fenc->i_frm_type, h->fenc->i_qpplus1);

        /* calculate the lambda again */
        if (new_qp != h->i_qp) {
            h->i_qp = new_qp;
            xavs2e_get_frame_lambda(h, h->fenc, new_qp);
            xavs2e_update_lambda(h, h->i_type, h->fenc->f_frm_lambda_ssd);
        }
    }

    /* confirm the encoding QP in the right range */
    h->i_qp = XAVS2_CLIP3(h->param->i_min_qp, h->param->i_max_qp, h->i_qp);
    h->i_qp = clip_qp(h, h->i_qp);

    /* encoding begin ----------------------------------------------
     */

    /* 帧级其他参数初始化 */
    if (IS_ALG_ENABLE(OPT_CU_QSFD)) {
        qsfd_calculate_threshold_of_a_frame(h);
    }

    if (h->param->enable_intra || h->fenc->i_frm_type == XAVS2_TYPE_I) {
        h->fenc->b_enable_intra = 1;
    } else {
        h->fenc->b_enable_intra = 0;
    }
}

/**
 * ---------------------------------------------------------------------------
 * Function   : encode a video frame
 * Parameters :
 *      [in ] : h   - pointer to struct xavs2_t, the xavs2 encoder
 *            : frm - pointer to struct xavs2_picture_t
 *      [out] : none
 * Return     : the length of bitstream
 * ---------------------------------------------------------------------------
 */
void *xavs2e_encode_one_frame(void *arg)
{
    xavs2_t    *h    = (xavs2_t *)arg;
    row_info_t *rows = h->frameinfo->rows;
    const int enable_wpp = h->h_top->i_row_threads > 1;
    int i;

    /* (1) init frame properties for frame coding -------------------------
     */
    xavs2e_frame_coding_init(h);

    h->pic_alf_on[0] = h->param->enable_alf;
    h->pic_alf_on[1] = h->param->enable_alf;
    h->pic_alf_on[2] = h->param->enable_alf;
    if (h->param->enable_alf && IS_ALG_ENABLE(OPT_FAST_ALF)) {
        if ((!h->fdec->rps.referd_by_others && h->i_type == SLICE_TYPE_B)) {
            h->pic_alf_on[0] = 0;
            h->pic_alf_on[1] = 0;
            h->pic_alf_on[2] = 0;
        }
    }

    /* start AEC frame coding */
    if (h->h_top->threadpool_aec != NULL && !h->param->enable_alf) {
        xavs2_threadpool_run(h->h_top->threadpool_aec, encoder_aec_encode_one_frame, h, 0);
    }

    /* (3) encode all LCU rows in current frame ---------------------------
     */
    for (i = 0; i < h->i_height_in_lcu; i++) {
        int lcu_y       = g_slice_lcu_row_order[i].lcu_y;
        int row_type    = g_slice_lcu_row_order[i].row_type;
        row_info_t *row = &rows[lcu_y];
        row_info_t *last_row;

        h->i_slice_index = g_slice_lcu_row_order[i].slice_idx;

        /* 是否需要额外处理Slice边界 */
        row->b_top_slice_border  = 0;
        row->b_down_slice_border = 0;

        /* 当前帧内的依赖行 */
        if (row_type) {
            last_row = &rows[lcu_y - 1];
            row->b_down_slice_border = (row_type == 2 && lcu_y != h->i_height_in_lcu - 1);
        } else {
            xavs2_slice_write_start(h);  /* Slice的第一行，初始化 */
            last_row = NULL;
            row->b_top_slice_border = (lcu_y > 0);
        }

        /* 等待参考帧中依赖的行编码完毕 */
        xavs2e_inter_sync(h, lcu_y, 0);

        /* encode one lcu row */
        if (enable_wpp && i != h->i_height_in_lcu - 1) {
            /* 1, 分配一个行级的线程进行编码 */
            if ((row->h = xavs2e_alloc_row_task(h)) == NULL) {
                return NULL;
            }

            /* 2, 检查当前行是否应立刻启动；
             *    规则为等待上一行至少完成两个LCU才启动线程，这里至少等待1个
             */
            wait_lcu_row_coded(last_row, 0);

            /* 3, 使用该行级线程进行编码 */
            xavs2_threadpool_run(h->h_top->threadpool_rdo, xavs2_lcu_row_write, row, 0);
        } else {
            row->h = h;
            xavs2_lcu_row_write(row);
        }

        /* 对Slice的最后一行LCU来说，需要合并多个Slice的码流
         * 但在RDO阶段，并不需要 */
        // if (h->param->slice_num > 1 && row_type == 2) {
        //     nal_merge_slice(h, h->slices[h->i_slice_index]->p_bs_buf, h->i_nal_type, h->i_nal_ref_idc);
        // }
    }   // for all LCU rows

    /* (4) Make sure that all LCU row are finished */
    if (h->param->slice_num > 1) {
        xavs2_frame_t *p_fdec = h->fdec;

        for (i = 0; i < h->i_height_in_lcu; i++) {
            xavs2_thread_mutex_lock(&p_fdec->mutex);    /* lock */
            while (p_fdec->num_lcu_coded_in_row[i] < h->i_width_in_lcu) {
                xavs2_thread_cond_wait(&p_fdec->cond, &p_fdec->mutex);
            }
            xavs2_thread_mutex_unlock(&p_fdec->mutex);  /* unlock */
        }
    }

    /* (5) 统计SAO的开启和开关比率 */
    if (h->param->enable_sao && (h->slice_sao_on[0] || h->slice_sao_on[1] || h->slice_sao_on[2])) {
        int sao_off_num_y = 0;
        int sao_off_num_u = 0;
        int sao_off_num_v = 0;
        for (i = 0; i < h->i_height_in_lcu; i++) {
            sao_off_num_y += h->num_sao_lcu_off[i][0];
            sao_off_num_u += h->num_sao_lcu_off[i][1];
            sao_off_num_v += h->num_sao_lcu_off[i][2];
        }
        h->fdec->num_lcu_sao_off[0] = sao_off_num_y;
        h->fdec->num_lcu_sao_off[1] = sao_off_num_u;
        h->fdec->num_lcu_sao_off[2] = sao_off_num_v;
    } else {
        int num_lcu = h->i_width_in_lcu * h->i_height_in_lcu;
        h->fdec->num_lcu_sao_off[0] = num_lcu;
        h->fdec->num_lcu_sao_off[1] = num_lcu;
        h->fdec->num_lcu_sao_off[2] = num_lcu;
    }

    /* (6) ALF */
    if (h->param->enable_alf) {
        xavs2_frame_copy_planes(h, h->img_alf, h->fdec);
        xavs2_frame_expand_border_frame(h, h->img_alf);
        alf_filter_one_frame(h);
        /* 重新对重构图像边界进行扩展 */
        if (h->pic_alf_on[0] || h->pic_alf_on[1] || h->pic_alf_on[2]) {
            xavs2_frame_expand_border_frame(h, h->fdec);
        }

#if ENABLE_FRAME_SUBPEL_INTPL
        if (h->pic_alf_on[0] && h->use_fractional_me != 0) {
            /* interpolate (after finished expanding border) */
            for (i = 0; i < h->i_height_in_lcu; i++) {
                interpolate_lcu_row(h, h->fdec, i);
            }
        }
#endif

        if (h->h_top->threadpool_aec != NULL) {
            xavs2_threadpool_run(h->h_top->threadpool_aec, encoder_aec_encode_one_frame, h, 0);
        }
    }


    /* (7) after encoding ... ------------------------------------------
     */

    if (h->td_rdo != NULL) {
        tdrdo_frame_done(h);
    }

    encoder_write_rec_frame(h->h_top);

    /* update encoding information */
    xavs2_reconfigure_encoder(h);

    /* release all reference frames */
    for (i = 0; i < h->i_ref; i++) {
        release_one_frame(h, h->fref[i]);
    }

    /* make sure all row context to release */
    if (h->param->i_lcurow_threads > 1) {
        int *num_lcu_coded = h->fdec->num_lcu_coded_in_row;

        for (i = 0; i < h->i_height_in_lcu; i++) {
            if (num_lcu_coded[i] <= h->i_width_in_lcu) {
                xavs2_sleep_ms(1);
            }
        }
    }
    h->b_all_row_ctx_released = 1;

    /* release the reconstructed frame */
    release_one_frame(h, h->fdec);

    /* set task status */
    encoder_set_task_status(h, XAVS2_TASK_RDO_DONE);

    if (h->h_top->threadpool_aec == NULL) {
        encoder_aec_encode_one_frame(h);
    }

    return 0;
}

/**
 * ---------------------------------------------------------------------------
 * Function   : encode a video frame
 * Parameters :
 *      [in ] : h   - pointer to struct xavs2_t, the xavs2 encoder
 *            : frm - pointer to struct xavs2_picture_t
 *      [out] : none
 * Return     : return 0 on success, -1 on failure
 * ---------------------------------------------------------------------------
 */
int encoder_encode(xavs2_handler_t *h_mgr, xavs2_frame_t *frame)
{
    if (frame->i_state != XAVS2_FLUSH) {
        xavs2_t *p_coder;

#if XAVS2_STAT
        frame->i_time_start = xavs2_mdate();
#endif

        /* prepare the encoding context.
         * get a frame encoder handle (initialized already) */
        if ((p_coder = encoder_alloc_frame_task(h_mgr, frame)) == NULL) {
            return -1;
        }

        init_decoding_frame(p_coder);

        /* encode the input frame: parallel or not */
        if (h_mgr->i_frm_threads > 1) {
            /* frame level parallel processing enabled */
            xavs2_threadpool_run(h_mgr->threadpool_rdo, xavs2e_encode_one_frame, p_coder, 0);
        } else {
            xavs2e_encode_one_frame(p_coder);
        }
    } else {
        /* flush output */
        encoder_flush(h_mgr);

        /* flush stream-end */
        encoder_output_frame_bitstream(h_mgr, NULL);
    }

    return 0;
}

/**
 * ---------------------------------------------------------------------------
 * Function   : destroy the xavs2 video encoder
 * Parameters :
 *      [in ] : h_mgr - pointer to struct xavs2_handler_t, the xavs2 encoder
 *      [out] : none
 * Return     : none
 * ---------------------------------------------------------------------------
 */
void encoder_close(xavs2_handler_t *h_mgr)
{
    if (h_mgr == NULL) {
        return;
    }

    /* flush delayed frames, so that every process could be stopped */
    encoder_flush(h_mgr);

    /* now, destroy everything! */
#if XAVS2_TRACE
    /* destroy trace */
    xavs2_trace_destroy();
#endif

    /* free the task manager and all encoding contexts */
    encoder_task_manager_free(h_mgr);
}
