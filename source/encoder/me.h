/*
 * me.h
 *
 * Description of this file:
 *    ME functions definition of the xavs2 library
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


#ifndef XAVS2_ME_H
#define XAVS2_ME_H


/**
 * ===========================================================================
 * macros
 * ===========================================================================
 */

#define pack16to32_mask(x,y)    (((x) << 16)|((y) & 0xFFFF))
#define pack16to32_mask2(mx,my) (((mx) << 16) | ((my) & 0x7FFF))
#define CHECK_MV_RANGE(mx,my)   (!(((pack16to32_mask2(mx,my) + mv_min) | (mv_max - pack16to32_mask2(mx,my))) & 0x80004000))
#define CHECK_MV_RANGE_X4(x0,y0,x1,y1,x2,y2,x3,y3) (!((                          \
    (pack16to32_mask2(x0, y0) + mv_min) | (mv_max - pack16to32_mask2(x0, y0)) | \
    (pack16to32_mask2(x1, y1) + mv_min) | (mv_max - pack16to32_mask2(x1, y1)) | \
    (pack16to32_mask2(x2, y2) + mv_min) | (mv_max - pack16to32_mask2(x2, y2)) | \
    (pack16to32_mask2(x3, y3) + mv_min) | (mv_max - pack16to32_mask2(x3, y3))   \
    ) & 0x80004000))

/* ---------------------------------------------------------------------------
 * conversion */
#define IPEL(mv)    (((mv) + 2) >> 2) /* convert fractional pixel MV to integer    pixel with rounding */
#define FPEL(mv)    ((mv) << 2)       /* convert integer    pixel MV to fractional pixel */

/* ---------------------------------------------------------------------------
 */
#define COPY1_IF_LT(x, y) \
    if ((y) < (x)) {\
        (x) = (y);\
    }

#define COPY2_IF_LT(x, y, a, b) \
    if ((y) < (x)) {\
        (x) = (y);\
        (a) = (b);\
    }

#define COPY3_IF_LT(x, y, a, b, c, d) \
    if ((y) < (x)) {\
        (x) = (y);\
        (a) = (b);\
        (c) = (d);\
    }

#define COPY4_IF_LT(x, y, a, b, c, d, e, f) \
    if ((y) < (x)) {\
        (x) = (y);\
        (a) = (b);\
        (c) = (d);\
        (e) = (f);\
    }

#define COPY5_IF_LT(x, y, a, b, c, d, e, f, g, h) \
    if ((y) < (x)) {\
        (x) = (y);\
        (a) = (b);\
        (c) = (d);\
        (e) = (f);\
        (g) = (h);\
    }

/* ---------------------------------------------------------------------------
 * MV cost */
#define MV_COST_IPEL(mx,my)     (WEIGHTED_COST(lambda, p_cost_mvx[(mx) << 2] + p_cost_mvy[(my) << 2]))
#define MV_COST_FPEL(mx,my)     (WEIGHTED_COST(lambda, p_cost_mvx[mx] + p_cost_mvy[my]))
#define MV_COST_FPEL_BID(mx,my) (WEIGHTED_COST(lambda, p_cost_bix[mx] + p_cost_biy[my]))


/**
 * ===========================================================================
 * function declares
 * ===========================================================================
 */

#define xavs2_me_get_buf_size FPFX(me_get_buf_size)
int  xavs2_me_get_buf_size(const xavs2_param_t *param);
#define xavs2_me_init FPFX(me_init)
void xavs2_me_init(xavs2_t *h, uint8_t **mem_base);
#define xavs2_me_init_umh_threshold FPFX(me_init_umh_threshold)
void xavs2_me_init_umh_threshold(xavs2_t *h, double *bsize, int i_qp);

#define xavs2_me_search FPFX(me_search)
dist_t xavs2_me_search(xavs2_t *h, xavs2_me_t *p_me, int16_t(*mvc)[2], int i_mvc);

#define xavs2_me_search_sym FPFX(me_search_sym)
dist_t xavs2_me_search_sym(xavs2_t *h, xavs2_me_t *p_me, pel_t *buf_pixel_temp, mv_t *mv);
#define xavs2_me_search_bid FPFX(me_search_bid)
dist_t xavs2_me_search_bid(xavs2_t *h, xavs2_me_t *p_me, pel_t *buf_pixel_temp, mv_t *fwd_mv, mv_t *bwd_mv, cu_parallel_t *p_enc);

#endif  // XAVS2_ME_H
