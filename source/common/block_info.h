/*
 * block_info.h
 *
 * Description of this file:
 *    Block Infomation functions definition of the xavs2 library
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


#ifndef XAVS2_BLOCK_INFO_H
#define XAVS2_BLOCK_INFO_H

#define get_neighbor_cbp_y FPFX(get_neighbor_cbp_y)
int  get_neighbor_cbp_y(xavs2_t *h, cu_info_t *p_cur, int slice_idx_cur_cu, int x_4x4, int y_4x4);
#define set_available_tables FPFX(set_available_tables)
void set_available_tables(xavs2_t *h);
#define check_neighbor_cu_avail FPFX(check_neighbor_cu_avail)
void check_neighbor_cu_avail(xavs2_t *h, cu_t *p_cu, int scu_x, int scu_y, int scu_xy);

#endif  // XAVS2_BLOCK_INFO_H
