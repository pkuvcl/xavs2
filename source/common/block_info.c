/*
 * block_info.c
 *
 * Description of this file:
 *    Block-infomation functions definition of the xavs2 library
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

#include "common.h"
#include "block_info.h"
#include "cudata.h"
#include "avs2tab.h"            // AVS2 tables

/**
 * ===========================================================================
 * global variables (const tables)
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static const uint8_t tab_DL_Avail64[16 * 16] = {
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/* ---------------------------------------------------------------------------
 */
static const uint8_t tab_DL_Avail32[8 * 8] = {
    1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0
};

/* ---------------------------------------------------------------------------
 */
static const uint8_t tab_DL_Avail16[4 * 4] = {
    1, 0, 1, 0,
    1, 0, 0, 0,
    1, 0, 1, 0,
    0, 0, 0, 0
};

/* ---------------------------------------------------------------------------
 */
static const uint8_t tab_DL_Avail8[2 * 2] = {
    1, 0,
    0, 0
};

static const uint8_t tab_TR_Avail64[16 * 16] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
};

/* ---------------------------------------------------------------------------
 */
static const uint8_t tab_TR_Avail32[8 * 8] = {
    // 0: 8 1:16 2: 32  pu size
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0
};

/* ---------------------------------------------------------------------------
 */
static const uint8_t tab_TR_Avail16[4 * 4] = {
    1, 1, 1, 1,
    1, 0, 1, 0,
    1, 1, 1, 0,
    1, 0, 1, 0
};

/* ---------------------------------------------------------------------------
 */
static const uint8_t tab_TR_Avail8[2 * 2] = {
    1, 1,
    1, 0
};


/**
 * ===========================================================================
 * function definition
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
cu_info_t *get_neighbor_cu_in_slice(xavs2_t *h, cu_info_t *p_cur, int slice_index_cur_cu, int x4x4, int y4x4)
{
    const int shift_4x4 = MIN_CU_SIZE_IN_BIT - MIN_PU_SIZE_IN_BIT;

    if (x4x4 < 0 || y4x4 < 0 || x4x4 >= h->i_width_in_minpu || y4x4 >= h->i_height_in_minpu) {
        return NULL;
    } else if ((p_cur->i_scu_x << shift_4x4) <= x4x4 && (p_cur->i_scu_y << shift_4x4) <= y4x4) {
        return p_cur;
    } else {
        cu_info_t *p_neighbor = &h->cu_info[(y4x4 >> 1) * h->i_width_in_mincu + (x4x4 >> 1)];
        return cu_get_slice_index(h, x4x4 >> 1, y4x4 >> 1) == slice_index_cur_cu ? p_neighbor : NULL;
    }
}

/* ---------------------------------------------------------------------------
 * get neighboring CBP
 */
int get_neighbor_cbp_y(xavs2_t *h, cu_info_t *p_cur, int slice_index_cur_cu, int x_4x4, int y_4x4)
{
    cu_info_t *p_neighbor = get_neighbor_cu_in_slice(h, p_cur, slice_index_cur_cu, x_4x4, y_4x4);

    if (p_neighbor == NULL) {
        return 0;
    } else if (p_neighbor->i_tu_split == TU_SPLIT_NON) {
        return p_neighbor->i_cbp & 1;   // TU不划分时，直接返回对应亮度块CBP
    } else {
        int cbp     = p_neighbor->i_cbp;
        int level   = p_neighbor->i_level - MIN_PU_SIZE_IN_BIT;
        int cu_mask = (1 << level) - 1;

        /* 4x4块在CU内的相对地址 */
        x_4x4 &= cu_mask;
        y_4x4 &= cu_mask;
        /* 求对应4x4块所在的变换块的CTP */
        if (p_neighbor->i_tu_split == TU_SPLIT_VER) {           // 垂直划分
            x_4x4 >>= (level - 2);
            return (cbp >> x_4x4) & 1;
        } else if (p_neighbor->i_tu_split == TU_SPLIT_HOR) {    // 水平划分
            y_4x4 >>= (level - 2);
            return (cbp >> y_4x4) & 1;
        } else {                                                // 四叉划分
            x_4x4 >>= (level - 1);
            y_4x4 >>= (level - 1);
            return (cbp >> (x_4x4 + (y_4x4 << 1))) & 1;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void set_available_tables(xavs2_t *h)
{
    switch (h->i_lcu_level) {
    case B64X64_IN_BIT:
        h->tab_avail_DL = (uint8_t *)tab_DL_Avail64;
        h->tab_avail_TR = (uint8_t *)tab_TR_Avail64;
        break;
    case B32X32_IN_BIT:
        h->tab_avail_DL = (uint8_t *)tab_DL_Avail32;
        h->tab_avail_TR = (uint8_t *)tab_TR_Avail32;
        break;
    case B16X16_IN_BIT:
        h->tab_avail_DL = (uint8_t *)tab_DL_Avail16;
        h->tab_avail_TR = (uint8_t *)tab_TR_Avail16;
        break;
    default:
        h->tab_avail_DL = (uint8_t *)tab_DL_Avail8;
        h->tab_avail_TR = (uint8_t *)tab_TR_Avail8;
        break;
    }
}

/* ---------------------------------------------------------------------------
 * check for available neighbor CUs and set pointers in current CU
 */
void check_neighbor_cu_avail(xavs2_t *h, cu_t *p_cu, int scu_x, int scu_y, int scu_xy)
{
    const int first_scu_y = h->slices[h->i_slice_index]->i_first_scu_y;
    int slice_index_of_cur_cu = cu_get_slice_index(h, scu_x, scu_y);

    /* reset */
    p_cu->p_topA_cu = p_cu->p_left_cu = NULL;
    p_cu->p_topL_cu = p_cu->p_topR_cu = NULL;

    /* check top row */
    if (scu_y > first_scu_y) {
        const int width_in_scu    = h->i_width_in_mincu;
        const int right_cu_offset = 1 << (p_cu->cu_info.i_level - MIN_CU_SIZE_IN_BIT);

        /* check top */
        p_cu->p_topA_cu = h->cu_info + (scu_xy - width_in_scu);

        /* check top-left */
        if (scu_x > 0) {
            p_cu->p_topL_cu = p_cu->p_topA_cu - 1;
        }

        /* check top-right */
        if (scu_x + right_cu_offset < width_in_scu) {
            if (slice_index_of_cur_cu == cu_get_slice_index(h, scu_x + right_cu_offset, scu_y - 1)) {
                cu_info_t *p_tmp_cu = p_cu->p_topA_cu + right_cu_offset;
                p_cu->p_topR_cu = p_tmp_cu;
            }
        }
    }

    /* check left */
    if (scu_x > 0) {
        p_cu->p_left_cu = &h->cu_info[scu_xy - 1];
    }
}
