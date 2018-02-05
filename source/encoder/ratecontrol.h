/*
 * ratecontrol.h
 *
 * Description of this file:
 *    Ratecontrol functions definition of the xavs2 library
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


#ifndef XAVS2_RATECONTRAL_H
#define XAVS2_RATECONTRAL_H

int  xavs2_rc_get_buffer_size(xavs2_param_t *h);
int  xavs2_ratecontrol_init(ratectrl_t *rc, xavs2_param_t *param);

int  xavs2_ratecontrol_base_qp(xavs2_t *h);

int  xavs2_ratecontrol_qp(xavs2_t *h, int frm_idx, int frm_type, int force_qp);
void xavs2_ratecontrol_end(xavs2_t *h, int frm_bits, int frm_qp, int frm_type, int frm_idx);


#if ENABLE_RATE_CONTROL_CU
int  xavs2_ratecontrol_qp_lcu(xavs2_t *h, int frm_idx, int qp);

void xavs2_ratecontrol_end_lcu(xavs2_t *h, int frm_idx, int qp);
#endif  // ENABLE_RATE_CONTROL_CU

void xavs2_ratecontrol_destroy(ratectrl_t *rc);

#endif  // XAVS2_RATECONTRAL_H
