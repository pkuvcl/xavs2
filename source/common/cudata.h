/*
 * cudata.h
 *
 * Description of this file:
 *    CU-Data functions definition of the xavs2 library
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


#ifndef XAVS2_CUDATA_H
#define XAVS2_CUDATA_H

void cu_get_mvds(xavs2_t *h, cu_t *p_cu);

void lcu_start_init_pos   (xavs2_t *h, int i_lcu_x, int i_lcu_y);
void lcu_start_init_pixels(xavs2_t *h, int i_lcu_x, int i_lcu_y);
void lcu_end(xavs2_t *h, int i_lcu_x, int i_lcu_y);


/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE int clip_qp(xavs2_t *h, int i_qp)
{
    /* AVS2-P2： 图像量化因子  picture_qp */
    int max_qp = MAX_QP + (h->param->sample_bit_depth - 8) * 8;
    return XAVS2_MAX(MIN_QP, XAVS2_MIN(max_qp, i_qp));
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void cu_init_transform_block(int i_cu_level, int i_tu_split, int b8, cb_t *p_tb)
{
    static const cb_t TRANS_BLOCK_INFO[TU_SPLIT_TYPE_NUM][4] = {// [tu_split_type][block]
        //    x, y, w, h          x, y, w, h          x, y, w, h          x, y, w, h           for block 0, 1, 2 and 3
        { { { 0, 0, 8, 8 } }, { { 0, 0, 0, 0 } }, { { 0, 0, 0, 0 } }, { { 0, 0, 0, 0 } } }, // 0: 8x8, ---, ---, --- (TU_SPLIT_NON   )
        { { { 0, 0, 8, 2 } }, { { 0, 2, 8, 2 } }, { { 0, 4, 8, 2 } }, { { 0, 6, 8, 2 } } }, // 2: 8x2, 8x2, 8x2, 8x2 (TU_SPLIT_HOR   )
        { { { 0, 0, 2, 8 } }, { { 2, 0, 2, 8 } }, { { 4, 0, 2, 8 } }, { { 6, 0, 2, 8 } } }, // 3: 2x8, 2x8, 2x8, 2x8 (TU_SPLIT_VER   )
        { { { 0, 0, 4, 4 } }, { { 4, 0, 4, 4 } }, { { 0, 4, 4, 4 } }, { { 4, 4, 4, 4 } } }  // 1: 4x4, 4x4, 4x4, 4x4 (TU_SPLIT_CROSS )
    };
    static const cb_t CHROMAW_BLOCK_INFO[2] = {
        { { 0, 8, 4, 4 } }, { { 4, 8, 4, 4 } }
    };
    const int shift_bits = (i_cu_level - MIN_CU_SIZE_IN_BIT);

    if (b8 < 4) {
        p_tb->v = TRANS_BLOCK_INFO[i_tu_split][b8].v << shift_bits;
    } else {
        p_tb->v = CHROMAW_BLOCK_INFO[b8 - 4].v << shift_bits;
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE int pu_get_mv_index(int i_mode, int pu_idx)
{
    int i_shift = IS_HOR_PU_PART(i_mode);
    return pu_idx << i_shift;
}


/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE int cu_get_qp(xavs2_t *h, cu_info_t *p_cu_info)
{
#if ENABLE_RATE_CONTROL_CU
    UNUSED_PARAMETER(h);
    return p_cu_info->i_cu_qp;
#else
    UNUSED_PARAMETER(p_cu_info);
    return h->i_qp;
#endif
}


/* ---------------------------------------------------------------------------
 *
 */
static ALWAYS_INLINE
int cu_get_slice_index(xavs2_t *h, int scu_x, int scu_y)
{
    int lcu_shift = (h->i_lcu_level - MIN_CU_SIZE_IN_BIT);
    int lcu_xy = (scu_y >> lcu_shift) * h->i_width_in_lcu
                 + (scu_x >> lcu_shift);
    return h->lcu_slice_idx[lcu_xy];
}


/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE int cu_get_chroma_qp(xavs2_t *h, int luma_qp, int uv)
{
    int QP;
    UNUSED_PARAMETER(uv);
    UNUSED_PARAMETER(h);
    QP = tab_qp_scale_chroma[XAVS2_CLIP3(0, 63, luma_qp)];
    return QP;
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
cu_layer_t *cu_get_layer(xavs2_t *h, int i_cu_level)
{
    return &h->lcu.cu_layer[i_cu_level - MIN_CU_SIZE_IN_BIT];
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
cu_mode_t *cu_get_layer_mode(xavs2_t *h, int i_cu_level)
{
    return &h->lcu.cu_layer[i_cu_level - MIN_CU_SIZE_IN_BIT].cu_mode;
}

static ALWAYS_INLINE
cu_parallel_t *cu_get_enc_context(xavs2_t *h, int i_cu_level)
{
#if PARALLEL_INSIDE_CTU
    return &h->lcu.cu_enc[i_cu_level - MIN_CU_SIZE_IN_BIT];
#else
    UNUSED_PARAMETER(i_cu_level);
    return &h->lcu.cu_enc[0];
#endif
}


#endif  // XAVS2_CUDATA_H
