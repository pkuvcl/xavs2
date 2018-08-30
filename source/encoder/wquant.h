/*
 * wquant.h
 *
 * Description of this file:
 *    Weighted Quant functions definition of the xavs2 library
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


#ifndef XAVS2_WQUANT_H
#define XAVS2_WQUANT_H

#if ENABLE_WQUANT

#define PARAM_NUM           6
#define WQ_MODEL_NUM        3
#define SCENE_MODE_NUM      4

#define UNDETAILED          0
#define DETAILED            1

#define WQ_MODE_F           0
#define WQ_MODE_U           1
#define WQ_MODE_D           2

#define FRAME_WQ_DEFAULT    0
#define USER_DEF_UNDETAILED 1
#define USER_DEF_DETAILED   2


/**
 * ===========================================================================
 * interface function declares
 * ===========================================================================
 */

#define xavs2_wq_init_seq_quant_param FPFX(wq_init_seq_quant_param)
void xavs2_wq_init_seq_quant_param(xavs2_t *h);
#define xavs2_wq_init_pic_quant_param FPFX(wq_init_pic_quant_param)
void xavs2_wq_init_pic_quant_param(xavs2_t *h);
#define xavs2_wq_update_pic_matrix FPFX(wq_update_pic_matrix)
void xavs2_wq_update_pic_matrix(xavs2_t *h);

extern const short tab_wq_param_default[2][6];

#endif

#endif  // XAVS2_WQUANT_H
