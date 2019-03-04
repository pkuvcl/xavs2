/*
 * wrapper.c
 *
 * Description of this file:
 *    encoder wrapper functions definition of the xavs2 library
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
#include "wrapper.h"
#include "frame.h"
#include "encoder.h"
#include "rps.h"

/* ---------------------------------------------------------------------------
 */
void frame_buffer_init(xavs2_handler_t *h_mgr, uint8_t **mem_base, xavs2_frame_buffer_t *frm_buf,
                       int num_frm, int frm_type)
{
    int i;

    memset(frm_buf, 0, sizeof(xavs2_frame_buffer_t));

    frm_buf->COI     = 0;
    frm_buf->COI_IDR = 0;
    frm_buf->POC_IDR = 0;
    frm_buf->num_frames = num_frm;
    frm_buf->i_frame_b  = 0;
    frm_buf->ip_pic_idx = 0;

    if (mem_base == NULL) {
        for (i = 0; i < num_frm; i++) {
            frm_buf->frames[i] = xavs2_frame_new(h_mgr->p_coder, NULL, frm_type);
        }
    } else {
        uint8_t *mem_ptr = *mem_base;
        for (i = 0; i < num_frm; i++) {
            frm_buf->frames[i] = xavs2_frame_new(h_mgr->p_coder, &mem_ptr, frm_type);
            ALIGN_POINTER(mem_ptr);
        }
        *mem_base = mem_ptr;
    }
}

/* ---------------------------------------------------------------------------
 * destroy frame buffer
 */
void frame_buffer_destroy(xavs2_handler_t *h_mgr, xavs2_frame_buffer_t *frm_buf)
{
    int i;

    for (i = 0; i < frm_buf->num_frames; i++) {
        xavs2_frame_delete(h_mgr, frm_buf->frames[i]);
        frm_buf->frames[i] = NULL;
    }
}

/* ---------------------------------------------------------------------------
 * update frame buffer information
 */
void frame_buffer_update(xavs2_t *h, xavs2_frame_buffer_t *frm_buf, xavs2_frame_t *frm)
{
    /* update the task manager */
    if (h->param->intra_period_max != 0 && frm->i_frm_type == XAVS2_TYPE_I) {
        frm_buf->COI_IDR = frm->i_frm_coi;
        frm_buf->POC_IDR = frm->i_frame;
    }

    if (frm->i_frm_type == XAVS2_TYPE_B) {
        frm_buf->i_frame_b++;      /* encoded B-picture index */
    } else {
        frm_buf->i_frame_b = 0;    /* reset */
        frm_buf->ip_pic_idx++;     /* encoded I/P/F-picture index */
    }
}


/**
 * ---------------------------------------------------------------------------
 * Function   : destroy all lists used by the AVS video encoder
 * Parameters :
 *      [in ] : h_mgr - pointer of struct xavs2_handler_t, the AVS encoder
 *      [out] : none
 * Return     : none
 * ---------------------------------------------------------------------------
 */
void destroy_all_lists(xavs2_handler_t *h_mgr)
{
    int i;

    assert(h_mgr != NULL);

    xl_destroy(&h_mgr->list_frames_output);
    xl_destroy(&h_mgr->list_frames_ready);
    xl_destroy(&h_mgr->list_frames_free);

    for (i = 0; i < XAVS2_INPUT_NUM; i++) {
        xavs2_frame_destroy_objects(h_mgr, h_mgr->ipb.frames[i]);
    }
}

/**
 * ---------------------------------------------------------------------------
 * Function   : proceeding of wrapper thread
 * Parameters :
 *      [in ] : h_mgr - pointer to xavs2_handler_t
 *      [out] : none
 * Return     : none
 * ---------------------------------------------------------------------------
 */
void *proc_wrapper_thread(void *args)
{
    xavs2_handler_t *h_mgr     = (xavs2_handler_t *)args;
    xlist_t         *list_in   = &h_mgr->list_frames_ready;
    xlist_t         *list_idle = &h_mgr->list_frames_free;

    for (;;) {
        /* fetch one node from input list */
        xavs2_frame_t *frame = (xavs2_frame_t *)xl_remove_head(list_in, 1);
        int            state = frame->i_state;
        if (state == XAVS2_EXIT_THREAD) {
            xl_append(list_idle, frame);
            break;              /* exit this thread */
        }

        /* encoding... */
        if (encoder_encode(h_mgr, frame) < 0) {
            xavs2_log(NULL, XAVS2_LOG_ERROR, "encode frame fail\n");
            break;              /* exit on error */
        }

        /* throw it into idle list */
        if (state == XAVS2_FLUSH) {
            xl_append(list_idle, frame);
        }
    }

    return NULL;
}
