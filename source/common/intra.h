/*
 * intra.h
 *
 * Description of this file:
 *    Intra prediction functions definition of the xavs2 library
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

#ifndef XAVS2_INTRA_H
#define XAVS2_INTRA_H

typedef struct intra_candidate_t intra_candidate_t;

#define xavs2_intra_get_cu_neighbors FPFX(intra_get_cu_neighbors)
uint32_t xavs2_intra_get_cu_neighbors(xavs2_t *h, cu_t *p_cu, int img_x, int img_y, int cu_size);

#define xavs2_intra_fill_ref_samples_luma FPFX(intra_fill_ref_samples_luma)
void xavs2_intra_fill_ref_samples_luma(xavs2_t *h, cu_t *p_cu, int img_x, int img_y,
                                       int block_x, int block_y, int bsx, int bsy);

#define rdo_get_pred_intra_luma FPFX(rdo_get_pred_intra_luma)
int rdo_get_pred_intra_luma(xavs2_t *h, cu_t *p_cu, intra_candidate_t *p_candidates,
                            pel_t *p_fenc, int mpm[], int blockidx,
                            int block_x, int block_y, int block_w, int block_h);

#define rdo_get_pred_intra_luma_rmd FPFX(rdo_get_pred_intra_luma_rmd)
int rdo_get_pred_intra_luma_rmd(xavs2_t *h, cu_t *p_cu, intra_candidate_t *p_candidates,
                                pel_t *p_fenc, int mpm[], int blockidx,
                                int block_x, int block_y, int block_w, int block_h);

#define rdo_get_pred_intra_luma_cuda FPFX(rdo_get_pred_intra_luma_cuda)
int rdo_get_pred_intra_luma_cuda(xavs2_t *h, cu_t *p_cu, intra_candidate_t *p_candidates,
                                 pel_t *p_fenc, int mpm[], int blockidx,
                                 int block_x, int block_y, int block_w, int block_h);

#define rdo_get_pred_intra_luma_2nd_pass FPFX(rdo_get_pred_intra_luma_2nd_pass)
int rdo_get_pred_intra_luma_2nd_pass(xavs2_t *h, cu_t *p_cu, intra_candidate_t *p_candidates,
                                     pel_t *p_fenc, int mpm[], int blockidx,
                                     int block_x, int block_y, int block_w, int block_h);

#define rdo_get_pred_intra_chroma FPFX(rdo_get_pred_intra_chroma)
int rdo_get_pred_intra_chroma(xavs2_t *h, cu_t *p_cu, int i_level, int pix_y_c, int pix_x_c,
                              intra_candidate_t *p_candidate_list);

#define rdo_get_pred_intra_chroma_fast FPFX(rdo_get_pred_intra_chroma_fast)
int rdo_get_pred_intra_chroma_fast(xavs2_t *h, cu_t *p_cu, int i_level, int pix_y_c, int pix_x_c,
                                   intra_candidate_t *p_candidate_list);

#endif  // XAVS2_INTRA_H
