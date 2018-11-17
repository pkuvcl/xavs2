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
