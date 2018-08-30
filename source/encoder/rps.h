/*
 * rps.h
 *
 * Description of this file:
 *    RPS functions definition of the xavs2 library
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

#ifndef XAVS2_RPS_H
#define XAVS2_RPS_H

// (1 - LDP, 2 - RA, 3 - RAP, 4 - AI)
enum xavs2e_rps_cfg_e {
    XAVS2_RPS_CFG_LDP = 1,
    XAVS2_RPS_CFG_RA  = 2,
    XAVS2_RPS_CFG_RAP = 3,
    XAVS2_RPS_CFG_AI  = 4
};

#define frame_buffer_get_free_frame_ipb FPFX(frame_buffer_get_free_frame_ipb)
xavs2_frame_t *frame_buffer_get_free_frame_ipb(xavs2_handler_t *h_mgr);
#define frame_buffer_update_remove_frames FPFX(frame_buffer_update_remove_frames)
void frame_buffer_update_remove_frames(xavs2_frame_buffer_t *frm_buf, xavs2_frame_t *cur_frm);
#define frame_buffer_remove_frames FPFX(frame_buffer_remove_frames)
void frame_buffer_remove_frames(xavs2_frame_buffer_t *frm_buf);

#define rps_build FPFX(rps_build)
int rps_build(const xavs2_t *h, xavs2_frame_buffer_t *frm_buf,
              xavs2_frame_t *cur_frm,
              xavs2_rps_t *p_rps, xavs2_frame_t *frefs[XAVS2_MAX_REFS]);

#define find_fdec_and_build_rps FPFX(find_fdec_and_build_rps)
xavs2_frame_t *find_fdec_and_build_rps(xavs2_t *h, xavs2_frame_buffer_t *frm_buf,
                                       xavs2_frame_t *cur_frm,
                                       xavs2_frame_t *frefs[XAVS2_MAX_REFS]);


#define rps_check_config FPFX(rps_check_config)
int rps_check_config(xavs2_param_t *param);

#define rps_set_picture_reorder_delay FPFX(rps_set_picture_reorder_delay)
void rps_set_picture_reorder_delay(xavs2_t *h);

#endif  // XAVS2_RPS_H
