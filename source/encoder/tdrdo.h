/*
 * tdrdo.h
 *
 * Description of this file:
 *    TDRDO functions definition of the xavs2 library
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


#ifndef __XAVS2_TDRDO_H__
#define __XAVS2_TDRDO_H__


/**
 * ===========================================================================
 * function declares
 * ===========================================================================
 */
int  tdrdo_get_buffer_size(xavs2_param_t *param);
int  tdrdo_init(td_rdo_t *td_rdo, xavs2_param_t *param);
void tdrdo_destroy(td_rdo_t *td_rdo);

void tdrdo_frame_start(xavs2_t *h);
void tdrdo_frame_done(xavs2_t *h);
void tdrdo_lcu_adjust_lambda(xavs2_t *h, rdcost_t *new_lambda);
void tdrdo_lcu_update(xavs2_t *h);

#endif  // __XAVS2_TDRDO_H__
