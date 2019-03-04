/*
 * avs2tab.h
 *
 * Description of this file:
 *    AVS2 tables definition of the xavs2 library (this file is ONLY included by block_info.c)
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

#ifndef XAVS2_AVS2_TABLE_H
#define XAVS2_AVS2_TABLE_H

/* 变换系数扫描顺序表 */
extern ALIGN32(const int16_t    tab_scan_4x4  [ 16][2]);
extern ALIGN32(const int16_t    tab_scan_4x4_yx[16][2]);
extern ALIGN32(const int16_t    tab_1d_scan_4x4[16]);

extern const int16_t   *tab_coef_scan1_list_nxn[2][4];
extern const int16_t   *tab_coef_scan1_list_hor[3];
extern const int16_t   *tab_coef_scan1_list_ver[3];

extern const int16_t  (*tab_coef_scan_list[4])[2];
extern const int16_t  (*tab_coef_scan_list_hor[3])[2];
extern const int16_t  (*tab_coef_scan_list_ver[3])[2];

extern const int16_t  (*tab_cg_scan_list_nxn[4])[2];
extern const int16_t  (*tab_cg_scan_list_hor[3])[2];
extern const int16_t  (*tab_cg_scan_list_ver[3])[2];

/* 变换块大小查找表 */
extern const uint8_t    tab_split_tu_pos[MAX_PRED_MODES][4][2];

/* 滤波 */
extern const uint8_t    tab_deblock_alpha[64];
extern const uint8_t    tab_deblock_beta[64];
extern const int        tab_saoclip[NUM_SAO_OFFSET][3];

extern const uint16_t   tab_Q_TAB   [80];
extern const uint16_t   tab_IQ_TAB  [80];
extern const uint8_t    tab_IQ_SHIFT[80];
extern const uint8_t    tab_qp_scale_chroma[64];

extern const int8_t     tab_intra_mode_luma2chroma[NUM_INTRA_MODE];

extern const int16_t    tab_dmh_pos[DMH_MODE_NUM + DMH_MODE_NUM - 1][2];

extern const float      FRAME_RATE[8];

extern const char *     xavs2_preset_names[];

extern const char *     xavs2_avs2_standard_version;  // standard version

#endif  // XAVS2_AVS2_TABLE_H
