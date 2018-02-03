/*
 * yuv_writer.c
 *
 * Description of this file:
 *    YUV Writing functions definition of the xavs2 library
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
#include "encoder.h"

/* ---------------------------------------------------------------------------
 */
static INLINE
void dump_yuv_out(xavs2_t *h, FILE *fp, xavs2_frame_t *frame, int img_w, int img_h)
{
    int j;

    if (fp != NULL) {
        UNUSED_PARAMETER(h);
        for (j = 0; j < img_h; j++) {
            fwrite(frame->planes[0] + j * frame->i_stride[0], img_w, 1, fp);
        }

        if (frame->i_plane == 3) {
            for (j = 0; j < (img_h >> 1); j++) {
                fwrite(frame->planes[1] + j * frame->i_stride[1], img_w >> 1, 1, fp);
            }

            for (j = 0; j < (img_h >> 1); j++) {
                fwrite(frame->planes[2] + j * frame->i_stride[2], img_w >> 1, 1, fp);
            }
        }

    }
}

/* ---------------------------------------------------------------------------
 */
void encoder_write_rec_frame(xavs2_handler_t *h_mgr)
{
    xavs2_t        *h     = h_mgr->p_coder;
    xavs2_frame_t **DPB   = h_mgr->dpb.frames;
    int size_dpb = h_mgr->dpb.num_frames;
    int i = 0;
    int j;

    xavs2_pthread_mutex_lock(&h_mgr->mutex);   /* lock */

    while (i < size_dpb) {
        int next_output_frame_idx;
        xavs2_frame_t  *frame = DPB[i];
        if (frame == NULL) {
            i++;
            continue;
        }

        xavs2_pthread_mutex_lock(&frame->mutex);  /* lock */

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
                xavs2_pthread_mutex_unlock(&frame->mutex);   /* unlock */
                break;
            }

            /* update output frame index */
            h_mgr->i_output = next_output_frame_idx;

#if XAVS2_DUMP_REC
            dump_yuv_out(h, h_mgr->h_rec_file, frame, h->param->org_width, h->param->org_height);
#endif //if XAVS2_DUMP_REC
            xavs2_pthread_mutex_unlock(&frame->mutex);   /* unlock */

            /* start over for the next reconstruction frame */
            i = 0;
            continue;
        }

        xavs2_pthread_mutex_unlock(&frame->mutex);    /* unlock */
        i++;
    }

    xavs2_pthread_mutex_unlock(&h_mgr->mutex);     /* unlock */
}
