/*
 * pre_encode.c
 *
 * Description of this file:
 *    Pre-Encode functions definition of the xavs2 library
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
#include "wrapper.h"
#include "frame.h"
#include "encoder.h"
#include "cpu.h"
#include "ratecontrol.h"
#include "tdrdo.h"
#include "presets.h"
#include "rps.h"

/* ---------------------------------------------------------------------------
 */
static
int slice_type_analyse(xavs2_handler_t *h_mgr, xavs2_frame_t *frm)
{
    /* GOP structures
     * openGOP:  I B......B F (...) B......B F  B......B I  B......B F (...)
     *            |<-subGOP->| ..  |<-subGOP->||<-subGOP->||<-subGOP->|
     *           |<---------- GOP0 ---------->||<---------- GOP1 ---------->|
     *
     * closeGOP: I B......B F (...) B......B F  I B......B F (...) B......B F
     *            |<-subGOP->| ..  |<-subGOP->|  |<-subGOP->|     |<-subGOP->|
     *           |<---------- GOP0 ---------->||<----------- GOP1 ---------->|
     *
     */
    lookahead_t *lookahead     = &h_mgr->lookahead;
    const xavs2_param_t *param = h_mgr->p_coder->param;
    int b_delayed = 0;            // the frame is normal to be encoded default

    /* slice type decision */
    if (lookahead->start) {
        int p_frm_type = param->enable_f_frame ? XAVS2_TYPE_F : XAVS2_TYPE_P;
        if (param->intra_period_max == 1) {
            // for AI (All Intra)
            frm->i_frm_type = XAVS2_TYPE_I;
            frm->b_keyframe = 1;
        } else if (param->intra_period_max == 0 || param->num_bframes == 0) {
            // for LDP (with no intra period)
            frm->i_frm_type = p_frm_type;
            frm->b_keyframe = 0;
            lookahead->gopframes++;
            // when intra period is non-zero, set key frames
            if (lookahead->gopframes - 1 == param->intra_period_max) {
                frm->i_frm_type    = XAVS2_TYPE_I;
                frm->b_keyframe    = 1;
                lookahead->gopframes = 1;
            }
        } else {
            // for RA (with any intra period) or LDP (with an intra period > 1),
            // buffer all these frames
            lookahead->gopframes++;
            b_delayed = 1;     // the frame is delayed to be encoded
            frm->b_keyframe = 0;

            --lookahead->bpframes;

            if (param->b_open_gop && lookahead->gopframes - 1 == param->intra_period_max) {
                // new sequence start
                // note: this i-frame's POI does NOT equal to its COI
                frm->i_frm_type = XAVS2_TYPE_I;
                frm->b_keyframe = 1;

                lookahead->gopframes = 1;
                lookahead->bpframes = param->i_gop_size;
            } else if (!param->b_open_gop && lookahead->gopframes == param->intra_period_max) {
                frm->i_frm_type = p_frm_type;
                lookahead->start = 0;
                lookahead->bpframes = param->i_gop_size;
            } else if (lookahead->bpframes > 0) {
                // the first 'bpframes - 1' frames is of type B
                frm->i_frm_type = XAVS2_TYPE_B;
            } else {
                frm->i_frm_type = p_frm_type;
                lookahead->bpframes = param->i_gop_size;
            }
        }
    } else {
        // the very first frame of an open GOP stream or the first frame (IDR) of a close GOP stream
        frm->i_frm_type     = XAVS2_TYPE_I;
        frm->b_keyframe     = 1;
        lookahead->start    = 1;   // set flag
        lookahead->bpframes = param->i_gop_size;
        lookahead->gopframes= 1;
    }

    return b_delayed;
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
void decide_frame_dts(xavs2_handler_t *h_mgr, xavs2_frame_t *frm)
{
    int num_bframes_delay = h_mgr->p_coder->picture_reorder_delay;
    int num_encoded_frame = h_mgr->num_encoded_frames_for_dts;
    if (num_bframes_delay) {
        if (num_encoded_frame > num_bframes_delay) {
            frm->i_dts = h_mgr->prev_reordered_pts_set[num_encoded_frame % num_bframes_delay];
        } else {
            frm->i_dts = frm->i_reordered_pts - num_bframes_delay;
        }
        h_mgr->prev_reordered_pts_set[num_encoded_frame % num_bframes_delay] = frm->i_reordered_pts;
    } else {
        frm->i_dts = frm->i_reordered_pts;
    }
    h_mgr->num_encoded_frames_for_dts++;
    /* ±ÜÃâÔ½½ç */
    if (h_mgr->num_encoded_frames_for_dts > 32768) {
        h_mgr->num_encoded_frames_for_dts -= 16384;
    }
}


/* ---------------------------------------------------------------------------
 */
static INLINE
void lookahead_append_frame(xavs2_handler_t *h_mgr, xlist_t *list_out, xavs2_frame_t *fenc,
                            int num_bframes, int idx_in_gop)
{
    if (fenc->i_state != XAVS2_EXIT_THREAD && fenc->i_state != XAVS2_FLUSH) {
        fenc->i_frm_coi = h_mgr->ipb.COI;
        h_mgr->ipb.COI++;

        frame_buffer_update(h_mgr->p_coder, &h_mgr->ipb, fenc);
        fenc->i_gop_idr_coi = h_mgr->ipb.COI_IDR;

        decide_frame_dts(h_mgr, fenc);

        UNUSED_PARAMETER(num_bframes);
        UNUSED_PARAMETER(idx_in_gop);

    }

    if (fenc != NULL) {
        xl_append(list_out, fenc);
    }
}


/* append a group of frames to the output list */
static INLINE
void lookahead_append_subgop_frames(xavs2_handler_t *h_mgr, xlist_t *list_out,
                                    xavs2_frame_t **blocked_frm_set, int64_t *blocked_pts_set,
                                    int num_frames)
{
    xavs2_t         *h               = h_mgr->p_coder;
    const xavs2_param_t *param       = h->param;
    int i;

    /* append all frames one by one to output list */
    if (param->i_gop_size == num_frames) {
        for (i = 0; i < num_frames; i++) {
            int k = param->cfg_ref_all[i].poc;

            if (k > 0) {
                /* get a frame to encode */
                xavs2_frame_t *frm = blocked_frm_set[k];
                if (frm == NULL) {
                    break;
                }

                /* clear */
                blocked_frm_set[k] = NULL;

                /* set DTS */
                frm->i_reordered_pts = blocked_pts_set[i + 1];

                /* append to output list to be encoded */
                lookahead_append_frame(h_mgr, list_out, frm, param->num_bframes, i + 1);
                h_mgr->num_encode++;
            } else {
                break;
            }
        }
#if !RELEASE_BUILD
        /* check the buffer */
        for (i = 1; i <= num_frames; i++) {
            assert(blocked_frm_set[i] == NULL);
        }
#endif
    } else  if (num_frames > 0) {
        static const int tab_poc_order[][8] = {
            { 1, 0, 0, 0, 0, 0, 0, 0 },  // 1
            { 2, 1, 0, 0, 0, 0, 0, 0 },  // 2: 1 B frame
            { 3, 1, 2, 0, 0, 0, 0, 0 },  // 3: 2 B frames
            { 4, 2, 1, 3, 0, 0, 0, 0 },  // 4: 3 B frames
            { 5, 2, 1, 3, 4, 0, 0, 0 },  // 5: 4 B frames
            { 6, 3, 1, 2, 4, 5, 0, 0 },  // 6: 5 B frames
            { 7, 3, 1, 2, 5, 4, 6, 0 },  // 7: 6 B frames
            { 8, 4, 2, 1, 3, 6, 5, 7 },  // 8: 7 B frames
        };
        const int *p_tab_poc = tab_poc_order[num_frames - 1];

        for (i = 0; i < num_frames; i++) {
            int k = p_tab_poc[i];

            if (k > 0) {
                /* get a frame to encode */
                xavs2_frame_t *frm = blocked_frm_set[k];
                if (frm == NULL) {
                    break;
                }

                /* clear */
                blocked_frm_set[k] = NULL;

                /* set frame type */
                if (i == 0) {
                    frm->i_frm_type = h_mgr->p_coder->param->enable_f_frame ? XAVS2_TYPE_F : XAVS2_TYPE_P;
                }

                /* set DTS */
                frm->i_reordered_pts = blocked_pts_set[i + 1];

                /* append to output list to be encoded */
                lookahead_append_frame(h_mgr, list_out, frm, param->num_bframes, i + 1);
                h_mgr->num_encode++;
            } else {
                break;
            }
        }
    }

    /* reset the index */
    h_mgr->num_blocked_frames = 0; /* the buffer is empty now */
}


/**
 * ===========================================================================
 * interface function defines (xavs2 encoder library APIs for AVS2 video encoder)
 * ===========================================================================
 */

/**
 * ---------------------------------------------------------------------------
 * Function   : complexity analysis and slice type decision of one frame,
 *              then send the frame into encoding queue
 * Parameters :
 *      [in ] : h_mgr - pointer to xavs2_handler_t
 *      [out] : end of encoding
 * Return     : none
 * ---------------------------------------------------------------------------
 */
int send_frame_to_enc_queue(xavs2_handler_t *h_mgr, xavs2_frame_t *frm)
{
    xavs2_t         *h               = h_mgr->p_coder;
    const xavs2_param_t *param       = h->param;
    xavs2_frame_t  **blocked_frm_set = h_mgr->blocked_frm_set;
    int64_t         *blocked_pts_set = h_mgr->blocked_pts_set;
    xlist_t         *list_out        = &h_mgr->list_frames_ready;

    /* check state */
    if (frm->i_state == XAVS2_EXIT_THREAD) {
        /* 1, estimate frame complexity and append rest frames */
        lookahead_append_subgop_frames(h_mgr, list_out, blocked_frm_set, blocked_pts_set, h_mgr->num_blocked_frames);

        /* 2, append current frame */
        lookahead_append_frame(h_mgr, list_out, frm, 0, 0);

        /* 3, exit this thread */
        return -1;
    }

    /* process... */
    if (frm->i_state != XAVS2_FLUSH) {
        /* decide the slice type of current frame */
        int b_delayed = slice_type_analyse(h_mgr, frm);          // is frame delayed to be encoded (B frame) ?

        if (b_delayed) {
            /* block a whole GOP until the last frame(I/P/F) of current GOP
             * a GOP should look somewhat like(POC order): B...BP */

            h_mgr->num_blocked_frames++;
            assert(h_mgr->num_blocked_frames <= param->i_gop_size);

            /* store the frame in blocked buffers */
            blocked_frm_set[h_mgr->num_blocked_frames] = frm;
            blocked_pts_set[h_mgr->num_blocked_frames] = frm->i_pts;

            /* is the last frame(I/P/F) of current GOP? */
            if (frm->i_frm_type != XAVS2_TYPE_B) {
                lookahead_append_subgop_frames(h_mgr, list_out, blocked_frm_set, blocked_pts_set, h_mgr->num_blocked_frames);
            }
        } else {
            assert(h_mgr->num_blocked_frames == 0);
            frm->i_reordered_pts = frm->i_pts;     /* DTS is same as PTS */

            lookahead_append_frame(h_mgr, list_out, frm, param->num_bframes, h_mgr->num_blocked_frames);
            h_mgr->num_encode++;
        }
    } else {
        /* flushing... */
        lookahead_append_subgop_frames(h_mgr, list_out, blocked_frm_set, blocked_pts_set, h_mgr->num_blocked_frames);
        h_mgr->num_blocked_frames = 0;

        /* append current frame to label flushing */
        lookahead_append_frame(h_mgr, list_out, frm, 0, 0);
    }

    return 0;
}

