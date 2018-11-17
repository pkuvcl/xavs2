/*
 * encoder.h
 *
 * Description of this file:
 *    Encoder functions definition of the xavs2 library
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

#ifndef XAVS2_ENCODER_H
#define XAVS2_ENCODER_H

/**
 * ===========================================================================
 * interface function defines (pre-encode processing)
 * ===========================================================================
 */

int      send_frame_to_enc_queue(xavs2_handler_t *h_mgr, xavs2_frame_t *frm);

void     xavs2e_get_frame_lambda(xavs2_t *h, xavs2_frame_t *cur_frm, int i_qp);

/**
 * ===========================================================================
 * interface function defines (encode processing)
 * ===========================================================================
 */

int      encoder_check_parameters(xavs2_param_t *param);

xavs2_t *encoder_open  (xavs2_param_t *param, xavs2_handler_t *h_mgr);
int      encoder_encode(xavs2_handler_t *h_mgr, xavs2_frame_t *frame);
void     encoder_close (xavs2_handler_t *h_mgr);

int      encoder_contexts_init(xavs2_t *h, xavs2_handler_t *h_mgr);
void     dump_yuv_out(xavs2_t *h, FILE *fp, xavs2_frame_t *frame, int img_w, int img_h);
void     encoder_fetch_one_encoded_frame(xavs2_handler_t *h_mgr, xavs2_outpacket_t *packet, int is_flush);

void     xavs2_reconfigure_encoder(xavs2_t *h);

#if XAVS2_STAT
/**
 * ===========================================================================
 * interface function defines (encode report)
 * ===========================================================================
 */
void     encoder_show_head_info(xavs2_param_t *param);
void     encoder_show_frame_info_tab(xavs2_t *h, xavs2_handler_t *mgr);

void     encoder_cal_psnr(xavs2_t *h, double *psnr_y, double *psnr_u, double *psnr_v);
void     encoder_cal_ssim(xavs2_t *h, double *ssim_y, double *ssim_u, double *ssim_v);

void     encoder_report_one_frame(xavs2_t *h, outputframe_t *frame);

void     encoder_report_stat_info(xavs2_t *h);
#endif

#endif  // XAVS2_ENCODER_H
