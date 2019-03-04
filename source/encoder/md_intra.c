/*
 * md_intra.c
 *
 * Description of this file:
 *    Mode decision functions definition for Intra prediction of the xavs2 library
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


/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
bool_t is_block_available(xavs2_t *h, int x_4x4, int y_4x4, int dx_4x4, int dy_4x4, int cur_slice_idx)
{
    int x2_4x4 = x_4x4 + dx_4x4;
    int y2_4x4 = y_4x4 + dy_4x4;

    if (x2_4x4 < 0 || y2_4x4 < 0 || x2_4x4 >= h->i_width_in_minpu || y2_4x4 >= h->i_height_in_minpu) {
        return 0;
    } else {
        return cur_slice_idx == cu_get_slice_index(h, x2_4x4 >> 1, y2_4x4 >> 1);
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
uint32_t get_intra_neighbors(xavs2_t *h, int x_4x4, int y_4x4, int bsx, int bsy, int cur_slice_idx)
{
    const int lcu_mask = (1 << (h->i_lcu_level - 2)) - 1;
    int leftdown, topright;

    /* 1. 检查相邻块是否属于同一个Slice */
    uint32_t b_LEFT      = is_block_available(h, x_4x4, y_4x4, -1,  0, cur_slice_idx);
    uint32_t b_TOP       = is_block_available(h, x_4x4, y_4x4,  0, -1, cur_slice_idx);
    uint32_t b_TOP_LEFT  = is_block_available(h, x_4x4, y_4x4, -1, -1, cur_slice_idx);
    uint32_t b_TOP_RIGHT = is_block_available(h, x_4x4, y_4x4, (bsx >> 1) - 1, -1, cur_slice_idx);   // (bsx >> MIN_PU_SIZE_IN_BIT << 1)
    uint32_t b_LEFT_DOWN = is_block_available(h, x_4x4, y_4x4, -1, (bsy >> 1) - 1, cur_slice_idx);   // (bsy >> MIN_PU_SIZE_IN_BIT << 1)

    /* 2. 检查相邻块是否在当前块之前重构 */
    x_4x4   &= lcu_mask;
    y_4x4   &= lcu_mask;
    leftdown = h->tab_avail_DL[((y_4x4 + (bsy >> 2) - 1) << (h->i_lcu_level - B4X4_IN_BIT)) + (x_4x4)];
    topright = h->tab_avail_TR[((y_4x4) << (h->i_lcu_level - B4X4_IN_BIT)) + (x_4x4 + (bsx >> 2) - 1)];

    b_LEFT_DOWN = b_LEFT_DOWN && leftdown;
    b_TOP_RIGHT = b_TOP_RIGHT && topright;

    return (b_LEFT << MD_I_LEFT) | (b_TOP << MD_I_TOP) | (b_TOP_LEFT << MD_I_TOP_LEFT) |
           (b_TOP_RIGHT << MD_I_TOP_RIGHT) | (b_LEFT_DOWN << MD_I_LEFT_DOWN);
}

/* ---------------------------------------------------------------------------
 * get intra PU availability
 */
static ALWAYS_INLINE
uint32_t get_intra_pu_avail(cu_t *p_cu, int block_x, int block_y, int bsx, int bsy)
{
    int cu_size = p_cu->i_size;
    uint32_t cu_avail = p_cu->intra_avail;
    uint32_t avail;

    if (block_x == 0 && block_y == 0) {
        avail = cu_avail;
        if (bsx < cu_size) {
            avail = (avail & (~(1 << MD_I_TOP_RIGHT))) | (!!IS_NEIGHBOR_AVAIL(cu_avail, MD_I_TOP) << MD_I_TOP_RIGHT);
        }
        if (bsy < cu_size) {
            avail = (avail & (~(1 << MD_I_LEFT_DOWN))) | (!!IS_NEIGHBOR_AVAIL(cu_avail, MD_I_LEFT) << MD_I_LEFT_DOWN);
        }
    } else if (block_y == 0) {
        avail = (cu_avail & (1 << MD_I_TOP));  // 上边界由CU的上边界决定；左下均不可用
        avail |= (1 << MD_I_LEFT);             // 左边界均可用
        avail |= ((cu_avail >> MD_I_TOP) & 1) << MD_I_TOP_LEFT;  // 左上由CU上边界可用性决定
        if (block_x + bsx < cu_size) {  // 右上由CU上边界和右上边界决定
            avail |= (!!IS_NEIGHBOR_AVAIL(cu_avail, MD_I_TOP)) << MD_I_TOP_RIGHT;
        } else {
            avail |= cu_avail & (1 << MD_I_TOP_RIGHT);
        }
    } else if (block_x == 0) {
        avail = (cu_avail & (1 << MD_I_LEFT)); // 左边界由CU的左边界决定
        avail |= (1 << MD_I_TOP);              // 上边界均可用
        avail |= ((cu_avail >> MD_I_LEFT) & 1) << MD_I_TOP_LEFT;  // 左上由CU上边界可用性决定
        if (bsx < cu_size && bsy < cu_size) {  // 右上
            avail |= 1 << MD_I_TOP_RIGHT;
        }
        // 左下
        if (block_y + bsy < cu_size) {
            avail |= (!!IS_NEIGHBOR_AVAIL(cu_avail, MD_I_LEFT)) << MD_I_LEFT_DOWN;
        } else {
            avail |= cu_avail & (1 << MD_I_LEFT_DOWN);
        }
    } else {
        // 右上、左下不可用
        avail = (1 << MD_I_LEFT) | (1 << MD_I_TOP) | (1 << MD_I_TOP_LEFT);
    }

    return avail;
}

/* ---------------------------------------------------------------------------
 * fill reference samples for luma component
 */
static INLINE
void fill_ref_samples_luma(xavs2_t *h, cu_t *p_cu, pel_t *EP,
                           int img_x, int img_y,
                           int block_x, int block_y,
                           int bsx, int bsy)
{
    int pos_x = (img_x - h->lcu.i_pix_x - 1);
    int pos_y = (img_y - h->lcu.i_pix_y - 1);
    pel_t *pTL = h->lcu.p_fdec[0] + pos_y * FDEC_STRIDE + pos_x;
    int xy = (((pos_y + 1) != 0) << 1) + ((pos_x + 1) != 0);
    uint32_t avail;

    /* 1, 检查参考边界有效性 */
    if (img_x + 2 * bsx <= h->i_width && img_y + 2 * bsy <= h->i_height
        && 0) {  // TODO: 高档次下不匹配，仍采用原先默认模式
        avail = get_intra_pu_avail(p_cu, block_x, block_y, bsx, bsy);
    } else {
        int cur_slice_idx = cu_get_slice_index(h, img_x >> MIN_CU_SIZE_IN_BIT, img_y >> MIN_CU_SIZE_IN_BIT);
        int b8_x = img_x >> MIN_PU_SIZE_IN_BIT;
        int b8_y = img_y >> MIN_PU_SIZE_IN_BIT;

        avail = get_intra_neighbors(h, b8_x, b8_y, bsx, bsy, cur_slice_idx);
    }

    p_cu->block_avail = (uint8_t)avail;

    /* 2, 完成参考边界像素的填充 */
    g_funcs.fill_edge_f[xy](pTL, FDEC_STRIDE, h->lcu.ctu_border[0].rec_top + pos_x - pos_y, EP, avail, bsx, bsy);
}

/* ---------------------------------------------------------------------------
 * \param h  : handle of the encoder
 * \param src: (src + 1) is aligned to 32-byte, src[1] is the 1st pixel in top reference row
 * \param dst: aligned to 32-byte
 */
static INLINE
void xavs2_intra_prediction(xavs2_t *h, pel_t *src, pel_t *dst, int i_dst, int dir_mode, int i_avail, int bsx, int bsy)
{
    UNUSED_PARAMETER(h);

    if (dir_mode != DC_PRED) {
        g_funcs.intraf[dir_mode](src, dst, i_dst, dir_mode, bsx, bsy);
    } else {
        int b_top  = !!IS_NEIGHBOR_AVAIL(i_avail, MD_I_TOP);
        int b_left = !!IS_NEIGHBOR_AVAIL(i_avail, MD_I_LEFT);
        int mode_ex = ((b_top << 8) + b_left);

        g_funcs.intraf[dir_mode](src, dst, i_dst, mode_ex, bsx, bsy);
    }
}

/**
 * ===========================================================================
 * interface function definition
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
void update_candidate_list(int mode, rdcost_t cost, int max_num, intra_candidate_t *p_candidates)
{
    int shift = 0;

    p_candidates += max_num - 1;
    while (shift < max_num && cost < p_candidates->cost) {
        p_candidates[0].mode = p_candidates[-1].mode;
        p_candidates[0].cost = p_candidates[-1].cost;
        shift++;
        p_candidates--;
    }

    p_candidates[1].mode = mode;
    p_candidates[1].cost = cost;
}

/* ---------------------------------------------------------------------------
 * used for generating intra luma prediction samples
 */
#define PREDICT_ADD_LUMA(MODE_IDX) \
{\
    pel_t *p_pred = p_enc->intra_pred[MODE_IDX];\
    int mode_bits = (mpm[0] == (MODE_IDX) || mpm[1] == (MODE_IDX)) ? 2 : 6;\
    rdcost_t cost = h->f_lambda_mode * mode_bits; \
    \
    xavs2_intra_prediction(h, edge_pixels, p_pred, block_w, MODE_IDX,\
        p_cu->block_avail, block_w, block_h);\
    cost += intra_cmp(p_fenc, FENC_STRIDE, p_pred, block_w);\
    update_candidate_list(MODE_IDX, cost, INTRA_MODE_NUM_FOR_RDO, p_candidates);\
}

/* ---------------------------------------------------------------------------
 * return numbers for RDO and candidate list by scanning all the intra modes
 */
int rdo_get_pred_intra_luma(xavs2_t *h, cu_t *p_cu, intra_candidate_t *p_candidates,
                            pel_t *p_fenc, int mpm[], int blockidx,
                            int block_x, int block_y, int block_w, int block_h)
{
    pixel_cmp_t intra_cmp = g_funcs.pixf.intra_cmp[PART_INDEX(block_w, block_h)];
    cu_parallel_t *p_enc = cu_get_enc_context(h, p_cu->cu_info.i_level);
    pel_t *edge_pixels   = &p_enc->buf_edge_pixels[(MAX_CU_SIZE << 2) - 1];
    int mode;
    int img_x = h->lcu.i_pix_x + p_cu->i_pos_x + block_x;
    int img_y = h->lcu.i_pix_y + p_cu->i_pos_y + block_y;

    /* get edge samples for intra prediction */
    fill_ref_samples_luma(h, p_cu, edge_pixels, img_x, img_y, block_x, block_y, block_w, block_h);

    UNUSED_PARAMETER(blockidx);

    /* loop over all intra predication modes */
    for (mode = 0; mode < NUM_INTRA_MODE; mode++) {
        PREDICT_ADD_LUMA(mode);
    }

    p_cu->feature.intra_had_cost = p_candidates[0].cost;
    return h->tab_num_intra_rdo[p_cu->cu_info.i_level - (p_cu->cu_info.i_tu_split != TU_SPLIT_NON)];
}

/* ---------------------------------------------------------------------------
 * return numbers for RDO and candidate list by rough scanning
 */
int rdo_get_pred_intra_luma_rmd(xavs2_t *h, cu_t *p_cu, intra_candidate_t *p_candidates,
                                pel_t *p_fenc, int mpm[], int blockidx,
                                int block_x, int block_y, int block_w, int block_h)
{
    int visited[NUM_INTRA_MODE] = { 0 };    /* 0: not visited yet
                                             * 1: visited in the first phase
                                             * 2: visited in final_mode */
    pixel_cmp_t intra_cmp = g_funcs.pixf.intra_cmp[PART_INDEX(block_w, block_h)];
    cu_parallel_t *p_enc  = cu_get_enc_context(h, p_cu->cu_info.i_level);
    pel_t *edge_pixels    = &p_enc->buf_edge_pixels[(MAX_CU_SIZE << 2) - 1];
    int mode, i, j;
    int num_angle = 0;
    int num_for_rdo;
    int num_to_add;
    int img_x = h->lcu.i_pix_x + p_cu->i_pos_x + block_x;
    int img_y = h->lcu.i_pix_y + p_cu->i_pos_y + block_y;

    /* get edge samples for intra prediction */
    fill_ref_samples_luma(h, p_cu, edge_pixels, img_x, img_y, block_x, block_y, block_w, block_h);

    UNUSED_PARAMETER(blockidx);

    /* 1, 遍历基础模式，
     * (1.1) 几个关键的角度 */
    for (mode = 0; mode < 3; mode++) {
        PREDICT_ADD_LUMA(mode);
        visited[mode] = 1;
    }
    /* (1.2) 角度预测模式 */
    for (mode = 4; mode < NUM_INTRA_MODE; mode += 4) {
        PREDICT_ADD_LUMA(mode);
        visited[mode] = 1;
    }

    /* 2, 遍历N个最优的模式的距离为二的模式，如果较优则放到CandModeList中 */
    num_to_add = h->num_intra_rmd_dist2;
    for (i = 0; i < num_to_add; i++) {
        mode = p_candidates[i].mode;
        if (mode <= 2) {
            continue;
        }

        if (mode > 3 && !visited[mode - 2]) {
            j = mode - 2;
            PREDICT_ADD_LUMA(j);
            visited[j] = 1;
        }

        if (mode < NUM_INTRA_MODE - 2 && !visited[mode + 2]) {
            j = mode + 2;
            PREDICT_ADD_LUMA(j);
            visited[j] = 1;
        }
    }

    /* 3, 把以上得到的最佳的两个模式的距离为一的模式放在CandModeList中 */
    num_to_add = h->num_intra_rmd_dist1;
    for (i = 0, num_angle = 0; num_angle < num_to_add && i < INTRA_MODE_NUM_FOR_RDO; i++) {
        mode = p_candidates[i].mode;
        if (mode <= 2) {
            continue;
        }

        if (mode > 3 && !visited[mode - 1]) {
            j = mode - 1;
            PREDICT_ADD_LUMA(j);
            visited[j] = 1;
            num_angle++;
        }

        if (mode < NUM_INTRA_MODE - 1 && !visited[mode + 1]) {
            j = mode + 1;
            PREDICT_ADD_LUMA(j);
            visited[j] = 1;
            num_angle++;
        }
    }

    /* 4, 查找最优列表中是否有MPMs，若没有，则加入，若有则不用加入 */
    if (!visited[mpm[0]]) {
        mode = mpm[0];
        PREDICT_ADD_LUMA(mode);
        visited[mode] = 1;
    }

    if (!visited[mpm[1]]) {
        mode = mpm[1];
        PREDICT_ADD_LUMA(mode);
        visited[mode] = 1;
    }

    num_for_rdo = h->tab_num_intra_rdo[p_cu->cu_info.i_level - (p_cu->cu_info.i_tu_split != TU_SPLIT_NON)];

    /* 若当前局部最优的两个模式是MPM之一，则减少RDO模式数量 */
    if (p_candidates[0].mode == mpm[0] || p_candidates[0].mode == mpm[1] ||
        p_candidates[1].mode == mpm[0] || p_candidates[1].mode == mpm[1]) {
        num_for_rdo = XAVS2_MIN(num_for_rdo, 3);
        return num_for_rdo;
    }

    /* 从M个最优模式中选定最终参加RDO的模式，即去重 */
    visited[p_candidates[0].mode] = 2;
    visited[p_candidates[1].mode] = 2;

    for (i = 2, j = 2; i < INTRA_MODE_NUM_FOR_RDO && j < num_for_rdo; i++) {
        mode = p_candidates[i].mode;
        if (!visited[mode]) {
            continue;
        }
        if (mode <= 2) {
            p_candidates[j++].mode = mode;
            visited[mode] = 2;
        } else if (mode == 3) {
            if (visited[4] == 1) {
                p_candidates[j++].mode = 3;
                visited[3] = 2;
            }
        } else if (mode == 32) {
            if (visited[31] == 1) {
                p_candidates[j++].mode = 32;
                visited[32] = 2;
            }
        } else {
            if (visited[mode - 1] == 1 && visited[mode + 1] == 1) {
                p_candidates[j++].mode = mode;
                visited[mode] = 2;
            }
        }
        if (visited[0] == 2 && visited[1] == 2 && visited[2] == 2) {
            break;
        }
    }

    p_cu->feature.intra_had_cost = p_candidates[0].cost;
    return XAVS2_MIN(num_for_rdo, j);
}


/* ---------------------------------------------------------------------------
 * return the best intra prediction mode from the 1st run
 */
int rdo_get_pred_intra_luma_2nd_pass(xavs2_t *h, cu_t *p_cu, intra_candidate_t *p_candidates,
                                     pel_t *p_fenc, int mpm[], int blockidx,
                                     int block_x, int block_y, int block_w, int block_h)
{
    cu_parallel_t *p_enc = cu_get_enc_context(h, p_cu->cu_info.i_level);
    int best_intra_mode = p_cu->cu_info.real_intra_modes[blockidx];
    pel_t *edge_pixels = &p_enc->buf_edge_pixels[(MAX_CU_SIZE << 2) - 1];
    pel_t *p_pred = p_enc->intra_pred[best_intra_mode];
    int img_x = h->lcu.i_pix_x + p_cu->i_pos_x + block_x;
    int img_y = h->lcu.i_pix_y + p_cu->i_pos_y + block_y;

    /* get edge samples for intra prediction */
    fill_ref_samples_luma(h, p_cu, edge_pixels, img_x, img_y, block_x, block_y, block_w, block_h);

    UNUSED_PARAMETER(p_fenc);
    UNUSED_PARAMETER(mpm);

    xavs2_intra_prediction(h, edge_pixels, p_pred, block_w, best_intra_mode, p_cu->block_avail, block_w, block_h);
    p_candidates[0].mode = best_intra_mode;
    p_candidates[0].cost = 0;

    return 1;
}

#undef PREDICT_ADD_LUMA

//#if OPT_FAST_RDO_INTRA_C
/* ---------------------------------------------------------------------------
 * predict an intra chroma block (fast)
 */
int rdo_get_pred_intra_chroma_fast(xavs2_t *h, cu_t *p_cu, int i_level, int pix_y_c, int pix_x_c,
                                   intra_candidate_t *p_candidate_list)
{
    cu_parallel_t *p_enc = cu_get_enc_context(h, i_level + 1);
    pel_t *p_fenc_u = h->lcu.p_fenc[IMG_U] + pix_y_c * FENC_STRIDE + pix_x_c;
    pel_t *p_fenc_v = h->lcu.p_fenc[IMG_V] + pix_y_c * FENC_STRIDE + pix_x_c;
    int blksize = 1 << i_level;
    pixel_cmp_t intra_chroma_cost = g_funcs.pixf.intra_cmp[PART_INDEX(blksize, blksize)];
    int num_for_rdo = 0;

    int LUMA_MODE[5] = { -1, DC_PRED, HOR_PRED, VERT_PRED, BI_PRED }; // map chroma mode to luma mode
    pel_t *EP_u = &p_enc->buf_edge_pixels[(MAX_CU_SIZE << 1) - 1];
    pel_t *EP_v = EP_u + (MAX_CU_SIZE << 2);
    int xy = p_cu->in_lcu_edge;

    /* 计算U、V分量的左上角像素点的位置 */
    pel_t *pTL_u = h->lcu.p_fdec[1] + (pix_y_c - 1) * FDEC_STRIDE + pix_x_c - 1;
    pel_t *pTL_v = h->lcu.p_fdec[2] + (pix_y_c - 1) * FDEC_STRIDE + pix_x_c - 1;
    int offset = (FREC_CSTRIDE >> 1);
    int m;

    /* 检查边界有效性 */
    uint32_t avail = p_cu->intra_avail;

    /* 计算每个模式号对应的预测模式 */
    LUMA_MODE[0] = p_cu->cu_info.real_intra_modes[0];

    /* 2.1, 获取参考边界像素 */
    g_funcs.fill_edge_f[xy](pTL_u, FDEC_STRIDE, h->lcu.ctu_border[1].rec_top + pix_x_c - pix_y_c, EP_u, avail, blksize, blksize);
    g_funcs.fill_edge_f[xy](pTL_v, FDEC_STRIDE, h->lcu.ctu_border[2].rec_top + pix_x_c - pix_y_c, EP_v, avail, blksize, blksize);

    for (m = 0; m < NUM_INTRA_MODE_CHROMA; m++) {
        p_candidate_list[m].mode = DM_PRED_C;
        p_candidate_list[m].cost = MAX_COST;
    }

    /* 2.2, 执行预测 */
    for (m = 0; m < NUM_INTRA_MODE_CHROMA; m++) {
        pel_t *p_pred_u = p_enc->intra_pred_c[m];
        pel_t *p_pred_v = p_enc->intra_pred_c[m] + offset;
        rdcost_t est_cost;

        xavs2_intra_prediction(h, EP_u, p_pred_u, FREC_CSTRIDE, LUMA_MODE[m], avail, blksize, blksize);
        xavs2_intra_prediction(h, EP_v, p_pred_v, FREC_CSTRIDE, LUMA_MODE[m], avail, blksize, blksize);

        est_cost  = intra_chroma_cost(p_fenc_u, FENC_STRIDE, p_pred_u, FREC_CSTRIDE);
        est_cost += intra_chroma_cost(p_fenc_v, FENC_STRIDE, p_pred_v, FREC_CSTRIDE);

        update_candidate_list(m, est_cost, NUM_INTRA_MODE_CHROMA, p_candidate_list);
    }

    if (h->i_type != SLICE_TYPE_I) {
        num_for_rdo = NUM_INTRA_C_FULL_RD;
        if (i_level == 6) {
            num_for_rdo -= 2;
        } else if (i_level == 5) {
            num_for_rdo -= 1;
        }
    } else {
        num_for_rdo = NUM_INTRA_MODE_CHROMA;
    }

    if (p_candidate_list[0].mode == DM_PRED_C) {
        num_for_rdo = 1;
    }

    num_for_rdo = XAVS2_MIN(h->num_rdo_intra_chroma, num_for_rdo);

    return num_for_rdo;
}
//#endif

/* ---------------------------------------------------------------------------
 * predict an intra chroma block
 */
int rdo_get_pred_intra_chroma(xavs2_t *h, cu_t *p_cu, int i_level_c, int pix_y_c, int pix_x_c,
                              intra_candidate_t *p_candidate_list)
{
    int LUMA_MODE[5] = { -1, DC_PRED, HOR_PRED, VERT_PRED, BI_PRED }; // map chroma mode to luma mode
    cu_parallel_t *p_enc = cu_get_enc_context(h, i_level_c + 1);
    pel_t *EP_u = &p_enc->buf_edge_pixels[(MAX_CU_SIZE << 1) - 1];
    pel_t *EP_v = EP_u + (MAX_CU_SIZE << 2);
    int bsize   = 1 << i_level_c;
    int xy = p_cu->in_lcu_edge;

    /* 计算U、V分量的左上角像素点的位置 */
    pel_t *pTL_u = h->lcu.p_fdec[1] + (pix_y_c - 1) * FDEC_STRIDE + pix_x_c - 1;
    pel_t *pTL_v = h->lcu.p_fdec[2] + (pix_y_c - 1) * FDEC_STRIDE + pix_x_c - 1;
    int offset = (FREC_CSTRIDE >> 1);
    int m;

    /* 检查边界有效性 */
    uint32_t avail = p_cu->intra_avail;

    /* 计算每个模式号对应的预测模式 */
    LUMA_MODE[0] = p_cu->cu_info.real_intra_modes[0];

    /* 2.1, 获取参考边界像素 */
    g_funcs.fill_edge_f[xy](pTL_u, FDEC_STRIDE, h->lcu.ctu_border[1].rec_top + pix_x_c - pix_y_c, EP_u, avail, bsize, bsize);
    g_funcs.fill_edge_f[xy](pTL_v, FDEC_STRIDE, h->lcu.ctu_border[2].rec_top + pix_x_c - pix_y_c, EP_v, avail, bsize, bsize);

    /* 2.2, 执行预测 */
    for (m = 0; m < NUM_INTRA_MODE_CHROMA; m++) {
        xavs2_intra_prediction(h, EP_u, p_enc->intra_pred_c[m] + 0,      FREC_CSTRIDE, LUMA_MODE[m], avail, bsize, bsize);
        xavs2_intra_prediction(h, EP_v, p_enc->intra_pred_c[m] + offset, FREC_CSTRIDE, LUMA_MODE[m], avail, bsize, bsize);

        p_candidate_list[m].mode = m;
        p_candidate_list[m].cost = MAX_COST;
    }

    return NUM_INTRA_MODE_CHROMA;
}

/* ---------------------------------------------------------------------------
 */
uint32_t xavs2_intra_get_cu_neighbors(xavs2_t *h, cu_t *p_cu, int img_x, int img_y, int cu_size)
{
    UNUSED_PARAMETER(p_cu);
    int cur_slice_idx = cu_get_slice_index(h, img_x >> MIN_CU_SIZE_IN_BIT, img_y >> MIN_CU_SIZE_IN_BIT);
    int b8_x = img_x >> MIN_PU_SIZE_IN_BIT;
    int b8_y = img_y >> MIN_PU_SIZE_IN_BIT;

    return get_intra_neighbors(h, b8_x, b8_y, cu_size, cu_size, cur_slice_idx);
}
