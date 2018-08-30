/*
 * sao.h
 *
 * Description of this file:
 *    SAO functions definition of the xavs2 library
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


#ifndef XAVS2_SAO_H
#define XAVS2_SAO_H

#define write_saoparam_one_lcu FPFX(write_saoparam_one_lcu)
void write_saoparam_one_lcu(xavs2_t *h, aec_t *p_aec, int lcu_x, int lcu_y, bool_t *slice_sao_on, SAOBlkParam *saoBlkParam);
#define sao_slice_onoff_decision FPFX(sao_slice_onoff_decision)
void sao_slice_onoff_decision(xavs2_t *h, bool_t *slice_sao_on);

/* decide sao parameters directly after one lcu reconstruction */
#define sao_get_lcu_param_after_deblock FPFX(sao_get_lcu_param_after_deblock)
void sao_get_lcu_param_after_deblock(xavs2_t *h, aec_t *p_aec, int i_lcu_x, int i_lcu_y);

/* conduct SAO filtering after one lcu row coding */
#define sao_filter_lcu FPFX(sao_filter_lcu)
void sao_filter_lcu(xavs2_t *h, SAOBlkParam blk_param[NUM_SAO_COMPONENTS], int lcu_x, int lcu_y);
#endif  // XAVS2_SAO_H
