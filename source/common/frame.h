/*
 * frame.h
 *
 * Description of this file:
 *    Frame handling functions definition of the xavs2 library
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

#ifndef XAVS2_FRAME_H
#define XAVS2_FRAME_H


/**
 * ===========================================================================
 * function declares
 * ===========================================================================
 */

#define xavs2_frame_new FPFX(frame_new)
xavs2_frame_t *xavs2_frame_new(xavs2_t *h, uint8_t **mem_base, int alloc_type);
#define xavs2_frame_delete FPFX(frame_delete)
void xavs2_frame_delete(xavs2_handler_t *h_mgr, xavs2_frame_t *frame);

#define xavs2_frame_buffer_size FPFX(frame_buffer_size)
size_t xavs2_frame_buffer_size(const xavs2_param_t *param, int alloc_type);
#define xavs2_frame_destroy_objects FPFX(frame_destroy_objects)
void xavs2_frame_destroy_objects(xavs2_handler_t *h_mgr, xavs2_frame_t *frame);

#define xavs2_frame_copy_planes FPFX(frame_copy_planes)
void xavs2_frame_copy_planes(xavs2_t *h, xavs2_frame_t *dst, xavs2_frame_t *src);

#define xavs2_frame_expand_border_frame FPFX(frame_expand_border_frame)
void plane_expand_border(pel_t *p_pix, int i_stride, int i_width, int i_height,
                         int i_padh, int i_padv, int b_pad_top, int b_pad_bottom);
void xavs2_frame_expand_border_frame(xavs2_t *h, xavs2_frame_t *frame);
#define xavs2_frame_expand_border_lcurow FPFX(frame_expand_border_lcurow)
void xavs2_frame_expand_border_lcurow(xavs2_t *h, xavs2_frame_t *frame, int i_lcu_y);

#define xavs2_frame_expand_border_mod8 FPFX(frame_expand_border_mod8)
void xavs2_frame_expand_border_mod8(xavs2_t *h, xavs2_frame_t *frame);

#endif  /* XAVS2_FRAME_H */
