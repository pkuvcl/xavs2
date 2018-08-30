/*
 * alf.h
 *
 * Description of this file:
 *    ALF functions definition of the xavs2 library
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

#ifndef XAVS2_ALF_H
#define XAVS2_ALF_H

#define alf_get_buffer_size FPFX(alf_get_buffer_size)
int  alf_get_buffer_size(const xavs2_param_t *param);

#define alf_init_buffer FPFX(alf_init_buffer)
void alf_init_buffer(xavs2_t *h, uint8_t *mem_base);

#define alf_filter_one_frame FPFX(alf_filter_one_frame)
void alf_filter_one_frame(xavs2_t *h);

#define alf_get_statistics_lcu FPFX(alf_get_statistics_lcu)
void alf_get_statistics_lcu(xavs2_t *h, int lcu_x, int lcu_y,
                            xavs2_frame_t *p_org, xavs2_frame_t *p_rec);

#endif  // XAVS2_ALF_H
