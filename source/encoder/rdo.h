/*
 * rdo.h
 *
 * Description of this file:
 *    RDO functions definition of the xavs2 library
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

#ifndef XAVS2_RDO_H
#define XAVS2_RDO_H

#define xavs2_init_valid_mode_table FPFX(init_valid_mode_table)
void xavs2_init_valid_mode_table(xavs2_t *h);

#define compress_ctu_inter FPFX(compress_ctu_inter)
rdcost_t compress_ctu_inter        (xavs2_t *h, aec_t *p_aec, cu_t *p_cu, int i_level, int i_min_level, int i_max_level, rdcost_t cost_limit);
#define compress_ctu_intra_topdown FPFX(compress_ctu_intra_topdown)
rdcost_t compress_ctu_intra_topdown(xavs2_t *h, aec_t *p_aec, cu_t *p_cu, int i_level, int i_min_level, int i_max_level, rdcost_t cost_limit);
#define compress_ctu_intra_downtop FPFX(compress_ctu_intra_downtop)
rdcost_t compress_ctu_intra_downtop(xavs2_t *h, aec_t *p_aec, cu_t *p_cu, int i_level, int i_min_level, int i_max_level, rdcost_t cost_limit);

typedef rdcost_t(*lcu_analyse_t)(xavs2_t *h, aec_t *p_aec, cu_t *p_cu, int i_level, int i_min_level, int i_max_level, rdcost_t cost_limit);

#endif  // XAVS2_RDO_H
