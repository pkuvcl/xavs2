/*
 * header.h
 *
 * Description of this file:
 *    Header writing functions definition of the xavs2 library
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


#ifndef XAVS2_HEADER_H
#define XAVS2_HEADER_H

#define xavs2_sequence_write FPFX(sequence_write)
int  xavs2_sequence_write(xavs2_t *h, bs_t *p_bs);
#define xavs2_user_data_write FPFX(user_data_write)
int  xavs2_user_data_write(bs_t *p_bs);
#define xavs2_intra_picture_header_write FPFX(intra_picture_header_write)
int  xavs2_intra_picture_header_write(xavs2_t *h, bs_t *p_bs);
#define xavs2_inter_picture_header_write FPFX(inter_picture_header_write)
int  xavs2_inter_picture_header_write(xavs2_t *h, bs_t *p_bs);
#define xavs2_picture_header_alf_write FPFX(picture_header_alf_write)
void xavs2_picture_header_alf_write(xavs2_t *h, ALFParam *alfPictureParam, bs_t *p_bs);
#define xavs2_slice_header_write FPFX(slice_header_write)
int  xavs2_slice_header_write(xavs2_t *h, slice_t *p_slice);

#endif  // XAVS2_HEADER_H
