/*
 * sao.c
 *
 * Description of this file:
 *    SAO functions definition of the xavs2 library
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

#include "aec.h"
#include "sao.h"
#include "filter.h"
#include "cpu.h"
#include "cudata.h"
#include "vec/intrinsic.h"

static const int tab_sao_check_mode_fast[3][5] = {
    1, 1, 0, 0, 0,
    0, 0, 0, 0, 1,
    0, 0, 0, 0, 1
};

/**
 * ===========================================================================
 * local function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void sao_init_stat_data(SAOStatData *p_stats)
{
    memset(p_stats, 0, sizeof(SAOStatData));
}

/* ---------------------------------------------------------------------------
 */
static
void sao_get_stat_block_EO_0(xavs2_frame_t *frm_rec, xavs2_frame_t *frm_org,
                             SAOStatData *p_stats, sao_region_t *p_region, int compIdx)
{
    int start_x, end_x, start_y, end_y;
    int x, y;
    int leftsign, rightsign;
    /* the size of SAO region max be larger than MAX_CU_SIZE on right/down of picture */
    int edgetype;

    int pix_x = p_region->pix_x[compIdx];
    int pix_y = p_region->pix_y[compIdx];
    int width = p_region->width[compIdx];
    int height = p_region->height[compIdx];

    int i_rec = frm_rec->i_stride[compIdx];
    int i_org = frm_org->i_stride[compIdx];
    const pel_t *p_rec = frm_rec->planes[compIdx] + pix_y * i_rec + pix_x;
    const pel_t *p_org = frm_org->planes[compIdx] + pix_y * i_org + pix_x;
    const pel_t *p_org_iter;
    const pel_t *p_rec_iter;
    sao_init_stat_data(p_stats);
    p_org_iter = p_org;
    p_rec_iter = p_rec;
    start_y = 0;
    end_y = height;
    start_x = p_region->b_left ? 0 : 1;
    end_x = p_region->b_right ? width : (width - 1);
    p_org_iter = p_org + start_y * i_org;
    p_rec_iter += start_y * i_rec;
    for (y = start_y; y < end_y; y++) {
        leftsign = xavs2_sign3(p_rec_iter[start_x] - p_rec_iter[start_x - 1]);
        for (x = start_x; x < end_x; x++) {
            rightsign = xavs2_sign3(p_rec_iter[x] - p_rec_iter[x + 1]);
            edgetype = leftsign + rightsign;
            leftsign = -rightsign;
            p_stats->diff[edgetype + 2] += (p_org_iter[x] - p_rec_iter[x]);
            p_stats->count[edgetype + 2]++;
        }
        p_rec_iter += i_rec;
        p_org_iter += i_org;
    }
}

/* ---------------------------------------------------------------------------
*/
static
void sao_get_stat_block_EO_90(xavs2_frame_t *frm_rec, xavs2_frame_t *frm_org,
                              SAOStatData *p_stats, sao_region_t *p_region, int compIdx)
{
    int start_x, end_x, start_y, end_y;
    int x, y;
    int upsign, downsign;
    /* the size of SAO region max be larger than MAX_CU_SIZE on right/down of picture */
    int edgetype;

    int pix_x = p_region->pix_x[compIdx];
    int pix_y = p_region->pix_y[compIdx];
    int width = p_region->width[compIdx];
    int height = p_region->height[compIdx];

    int i_rec = frm_rec->i_stride[compIdx];
    int i_org = frm_org->i_stride[compIdx];
    const pel_t *p_rec = frm_rec->planes[compIdx] + pix_y * i_rec + pix_x;
    const pel_t *p_org = frm_org->planes[compIdx] + pix_y * i_org + pix_x;
    const pel_t *p_org_iter;
    const pel_t *p_rec_iter;

    sao_init_stat_data(p_stats);

    p_org_iter = p_org;
    p_rec_iter = p_rec;
    start_x = 0;
    end_x = width;
    start_y = p_region->b_top ? 0 : 1;
    end_y = p_region->b_down ? height : (height - 1);
    for (x = start_x; x < end_x; x++) {
        upsign = xavs2_sign3(p_rec_iter[start_y * i_rec + x] - p_rec_iter[(start_y - 1) * i_rec + x]);
        for (y = start_y; y < end_y; y++) {
            downsign = xavs2_sign3(p_rec_iter[y * i_rec + x] - p_rec_iter[(y + 1) * i_rec + x]);
            edgetype = downsign + upsign;
            upsign = -downsign;
            p_stats->diff[edgetype + 2] += (p_org_iter[y * i_org + x] - p_rec_iter[y * i_rec + x]);
            p_stats->count[edgetype + 2]++;
        }
    }
}

/* ---------------------------------------------------------------------------
*/
static
void sao_get_stat_block_EO_135(xavs2_frame_t *frm_rec, xavs2_frame_t *frm_org,
                               SAOStatData *p_stats, sao_region_t *p_region, int compIdx)
{
    int start_x_r0, end_x_r0, start_x_r, end_x_r, start_x_rn, end_x_rn;
    int x, y;
    int upsign, downsign;
    /* the size of SAO region max be larger than MAX_CU_SIZE on right/down of picture */
    int signupline[MAX_CU_SIZE << 1];
    int reg = 0;
    int edgetype;

    int pix_x = p_region->pix_x[compIdx];
    int pix_y = p_region->pix_y[compIdx];
    int width = p_region->width[compIdx];
    int height = p_region->height[compIdx];

    int i_rec = frm_rec->i_stride[compIdx];
    int i_org = frm_org->i_stride[compIdx];
    const pel_t *p_rec = frm_rec->planes[compIdx] + pix_y * i_rec + pix_x;
    const pel_t *p_org = frm_org->planes[compIdx] + pix_y * i_org + pix_x;
    const pel_t *p_org_iter;
    const pel_t *p_rec_iter;

    sao_init_stat_data(p_stats);

    p_org_iter = p_org;
    p_rec_iter = p_rec;
    start_x_r0 = p_region->b_top_left ? 0 : 1;
    end_x_r0 = p_region->b_top ? (p_region->b_right ? width : (width - 1)) : 1;
    start_x_r = p_region->b_left ? 0 : 1;
    end_x_r = p_region->b_right ? width : (width - 1);
    start_x_rn = p_region->b_down ? (p_region->b_left ? 0 : 1) : (width - 1);
    end_x_rn = p_region->b_right_down ? width : (width - 1);

    // init the line buffer
    for (x = start_x_r + 1; x < end_x_r + 1; x++) {
        upsign = xavs2_sign3(p_rec_iter[x + i_rec] - p_rec_iter[x - 1]);
        signupline[x] = upsign;
    }
    // first row
    for (x = start_x_r0; x < end_x_r0; x++) {
        upsign = xavs2_sign3(p_rec_iter[x] - p_rec_iter[x - 1 - i_rec]);
        edgetype = upsign - signupline[x + 1];
        p_stats->diff[edgetype + 2] += (p_org_iter[x] - p_rec_iter[x]);
        p_stats->count[edgetype + 2]++;
    }

    // middle rows
    p_rec_iter += i_rec;
    p_org_iter += i_org;
    for (y = 1; y < height - 1; y++) {
        for (x = start_x_r; x < end_x_r; x++) {
            if (x == start_x_r) {
                upsign = xavs2_sign3(p_rec_iter[x] - p_rec_iter[x - 1 - i_rec]);
                signupline[x] = upsign;
            }
            downsign = xavs2_sign3(p_rec_iter[x] - p_rec_iter[x + 1 + i_rec]);
            edgetype = downsign + signupline[x];
            p_stats->diff[edgetype + 2] += (p_org_iter[x] - p_rec_iter[x]);
            p_stats->count[edgetype + 2]++;
            signupline[x] = (char)reg;
            reg = -downsign;
        }
        p_rec_iter += i_rec;
        p_org_iter += i_org;
    }
    // last row
    for (x = start_x_rn; x < end_x_rn; x++) {
        if (x == start_x_r) {
            upsign = xavs2_sign3(p_rec_iter[x] - p_rec_iter[x - 1 - i_rec]);
            signupline[x] = upsign;
        }
        downsign = xavs2_sign3(p_rec_iter[x] - p_rec_iter[x + 1 + i_rec]);
        edgetype = downsign + signupline[x];
        p_stats->diff[edgetype + 2] += (p_org_iter[x] - p_rec_iter[x]);
        p_stats->count[edgetype + 2]++;
    }
}

/* ---------------------------------------------------------------------------
*/
static
void sao_get_stat_block_EO_45(xavs2_frame_t *frm_rec, xavs2_frame_t *frm_org,
                              SAOStatData *p_stats, sao_region_t *p_region, int compIdx)
{
    int start_x_r0, end_x_r0, start_x_r, end_x_r, start_x_rn, end_x_rn;
    int x, y;
    int upsign, downsign;
    /* the size of SAO region max be larger than MAX_CU_SIZE on right/down of picture */
    int signupline[MAX_CU_SIZE << 1];
    int *signupline1;
    int edgetype;

    int pix_x = p_region->pix_x[compIdx];
    int pix_y = p_region->pix_y[compIdx];
    int width = p_region->width[compIdx];
    int height = p_region->height[compIdx];

    int i_rec = frm_rec->i_stride[compIdx];
    int i_org = frm_org->i_stride[compIdx];
    const pel_t *p_rec = frm_rec->planes[compIdx] + pix_y * i_rec + pix_x;
    const pel_t *p_org = frm_org->planes[compIdx] + pix_y * i_org + pix_x;
    const pel_t *p_org_iter;
    const pel_t *p_rec_iter;

    sao_init_stat_data(p_stats);

    p_org_iter = p_org;
    p_rec_iter = p_rec;
    start_x_r0 = p_region->b_top ? (p_region->b_left ? 0 : 1) : (width - 1);
    end_x_r0 = p_region->b_top_right ? width : (width - 1);
    start_x_r = p_region->b_left ? 0 : 1;
    end_x_r = p_region->b_right ? width : (width - 1);
    start_x_rn = p_region->b_down_left ? 0 : 1;
    end_x_rn = p_region->b_down ? (p_region->b_right ? width : (width - 1)) : 1;

    // init the line buffer
    signupline1 = signupline + 1;
    for (x = start_x_r - 1; x < XAVS2_MAX(end_x_r - 1, end_x_r0 - 1); x++) {
        upsign = xavs2_sign3(p_rec_iter[x + i_rec] - p_rec_iter[x + 1]);
        signupline1[x] = upsign;
    }
    // first row
    for (x = start_x_r0; x < end_x_r0; x++) {
        upsign = xavs2_sign3(p_rec_iter[x] - p_rec_iter[x + 1 - i_rec]);
        edgetype = upsign - signupline1[x - 1];
        p_stats->diff[edgetype + 2] += (p_org_iter[x] - p_rec_iter[x]);
        p_stats->count[edgetype + 2]++;
    }

    // middle rows
    p_rec_iter += i_rec;
    p_org_iter += i_org;
    for (y = 1; y < height - 1; y++) {
        for (x = start_x_r; x < end_x_r; x++) {
            if (x == end_x_r - 1) {
                upsign = xavs2_sign3(p_rec_iter[x] - p_rec_iter[x + 1 - i_rec]);
                signupline1[x] = upsign;
            }
            downsign = xavs2_sign3(p_rec_iter[x] - p_rec_iter[x - 1 + i_rec]);
            edgetype = downsign + signupline1[x];
            p_stats->diff[edgetype + 2] += (p_org_iter[x] - p_rec_iter[x]);
            p_stats->count[edgetype + 2]++;
            signupline1[x - 1] = -downsign;
        }
        p_rec_iter += i_rec;
        p_org_iter += i_org;
    }
    for (x = start_x_rn; x < end_x_rn; x++) {
        if (x == end_x_r - 1) {
            upsign = xavs2_sign3(p_rec_iter[x] - p_rec_iter[x + 1 - i_rec]);
            signupline1[x] = upsign;
        }
        downsign = xavs2_sign3(p_rec_iter[x] - p_rec_iter[x - 1 + i_rec]);
        edgetype = downsign + signupline1[x];
        p_stats->diff[edgetype + 2] += (p_org_iter[x] - p_rec_iter[x]);
        p_stats->count[edgetype + 2]++;
    }
}

/* ---------------------------------------------------------------------------
*/
static
void sao_get_stat_block_BO(xavs2_frame_t *frm_rec, xavs2_frame_t *frm_org,
                           SAOStatData *p_stats, sao_region_t *p_region, int compIdx)
{
    int start_x, end_x, start_y, end_y;
    int x, y;
    /* the size of SAO region max be larger than MAX_CU_SIZE on right/down of picture */
    int bandtype;
    int band_shift;

    int pix_x = p_region->pix_x[compIdx];
    int pix_y = p_region->pix_y[compIdx];
    int width = p_region->width[compIdx];
    int height = p_region->height[compIdx];

    int i_rec = frm_rec->i_stride[compIdx];
    int i_org = frm_org->i_stride[compIdx];
    const pel_t *p_rec = frm_rec->planes[compIdx] + pix_y * i_rec + pix_x;
    const pel_t *p_org = frm_org->planes[compIdx] + pix_y * i_org + pix_x;
    const pel_t *p_org_iter;
    const pel_t *p_rec_iter;

    sao_init_stat_data(p_stats);

    p_org_iter = p_org;
    p_rec_iter = p_rec;
    band_shift = (g_bit_depth - NUM_SAO_BO_CLASSES_IN_BIT);
    start_x = 0;
    end_x = width;
    start_y = 0;
    end_y = height;
    for (y = start_y; y < end_y; y++) {
        for (x = start_x; x < end_x; x++) {
            bandtype = p_rec_iter[x] >> band_shift;
            p_stats->diff[bandtype] += (p_org_iter[x] - p_rec_iter[x]);
            p_stats->count[bandtype]++;
        }
        p_rec_iter += i_rec;
        p_org_iter += i_org;
    }
}

/* ---------------------------------------------------------------------------
*/
typedef void(*sao_pf)(xavs2_frame_t *frm_rec, xavs2_frame_t *frm_org,
                      SAOStatData *stat_datas, sao_region_t *p_region, int compIdx);

sao_pf gf_sao_stat[5] = {
    sao_get_stat_block_EO_0,
    sao_get_stat_block_EO_90,
    sao_get_stat_block_EO_135,
    sao_get_stat_block_EO_45,
    sao_get_stat_block_BO
};

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
long distortion_cal(long count, int offset, long diff)
{
    return (count * offset * offset - diff * offset * 2);
}

/* ---------------------------------------------------------------------------
 */
static
int offset_estimation(int typeIdx, int classIdx, rdcost_t lambda, long offset_ori, int count, long diff, rdcost_t *bestCost)
{
    const int tab_EO_OFFSET_MAP[8] = {4, 2, 1, 3, 5, 6, 7, 7};  // -1, 0, ..., 6
    int cur_offset = offset_ori;
    int offset_best = 0;
    int lower_bd, upper_bd, Th;
    int temp_offset, start_offset, end_offset;
    int temprate;
    long tempdist;
    rdcost_t tempcost, mincost;
    const int *eo_offset_bins = &(tab_EO_OFFSET_MAP[1]);
    int offset_type;

    if (typeIdx == SAO_TYPE_BO) {
        offset_type = SAO_CLASS_BO;
    } else {
        offset_type = classIdx;
    }
    lower_bd = tab_saoclip[offset_type][0];
    upper_bd = tab_saoclip[offset_type][1];
    Th       = tab_saoclip[offset_type][2];
    cur_offset = XAVS2_CLIP3(lower_bd, upper_bd, cur_offset);
    if (typeIdx == SAO_TYPE_BO) {
        start_offset = XAVS2_MIN(cur_offset, 0);
        end_offset   = XAVS2_MAX(cur_offset, 0);
    } else {
        assert(typeIdx >= SAO_TYPE_EO_0 && typeIdx <= SAO_TYPE_EO_45);
        switch (classIdx) {
        case SAO_CLASS_EO_FULL_VALLEY:
            start_offset = -1;
            end_offset = XAVS2_MAX(cur_offset, 1);
            break;
        case SAO_CLASS_EO_HALF_VALLEY:
            start_offset = 0;
            end_offset = 1;
            break;
        case SAO_CLASS_EO_HALF_PEAK:
            start_offset = -1;
            end_offset = 0;
            break;
        case SAO_CLASS_EO_FULL_PEAK:
            start_offset = XAVS2_MIN(cur_offset, -1);
            end_offset = 1;
            break;
        default:
            xavs2_log(NULL, XAVS2_LOG_ERROR, "Not a supported SAO mode offset_estimation\n");
            exit(-1);
        }
    }

    mincost = MAX_COST;
    for (temp_offset = start_offset; temp_offset <= end_offset; temp_offset++) {
        if (typeIdx == SAO_TYPE_BO) {
            assert(offset_type == SAO_CLASS_BO);
            temprate = XAVS2_ABS(temp_offset);
            temprate = temprate ? (temprate + 1) : 0;
        } else if (classIdx == SAO_CLASS_EO_HALF_VALLEY || classIdx == SAO_CLASS_EO_HALF_PEAK) {
            temprate = XAVS2_ABS(temp_offset);
        } else {
            temprate = eo_offset_bins[classIdx == SAO_CLASS_EO_FULL_VALLEY ? temp_offset : -temp_offset];
        }
        temprate = (temprate == Th) ? temprate : (temprate + 1);

        tempdist = distortion_cal(count, temp_offset, diff);
        tempcost = tempdist + lambda * temprate;
        if (tempcost < mincost) {
            mincost = tempcost;
            offset_best = temp_offset;
            *bestCost = tempcost;
        }
    }

    return offset_best;
}

/* ---------------------------------------------------------------------------
 */
static void find_offset(int typeIdc, SAOStatData *p_stat, SAOBlkParam *p_param, rdcost_t lambda)
{
    int class_i;
    rdcost_t classcost[MAX_NUM_SAO_CLASSES];
    rdcost_t offth;
    rdcost_t mincost_bandsum, cost_bandsum;
    int num_class = (typeIdc == SAO_TYPE_BO) ? NUM_SAO_BO_CLASSES : NUM_SAO_EO_CLASSES;
    static const int deltaband_cost[] = { -1, -1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 16 };

    int start_band1, start_band2, delta_band12;

    for (class_i = 0; class_i < num_class; class_i++) {
        if ((typeIdc != SAO_TYPE_BO) && (class_i == SAO_CLASS_EO_PLAIN)) {
            p_param->offset[class_i] = 0;
            continue;
        }
        if (p_stat[typeIdc].count[class_i] == 0) {
            p_param->offset[class_i] = 0; //offset will be zero
            continue;
        }
        offth = p_stat[typeIdc].diff[class_i] > 0 ? 0.5 : (p_stat[typeIdc].diff[class_i] < 0 ? -0.5 : 0);
        p_param->offset[class_i] = (int8_t)((double)p_stat[typeIdc].diff[class_i] / (double)p_stat[typeIdc].count[class_i] + offth);
    }
    if (typeIdc == SAO_TYPE_BO) {
        int best_start_band1 = 0;
        int best_start_band2 = 0;
        for (class_i = 0; class_i < num_class; class_i++) {
            p_param->offset[class_i] = (int8_t)offset_estimation(typeIdc, class_i, lambda, p_param->offset[class_i], p_stat[typeIdc].count[class_i],
                                       p_stat[typeIdc].diff[class_i], &(classcost[class_i]));
        }
        mincost_bandsum = MAX_DOUBLE;
        for (start_band1 = 0; start_band1 < (NUM_SAO_BO_CLASSES - 1); start_band1++) {
            for (start_band2 = start_band1 + 2; start_band2 < (NUM_SAO_BO_CLASSES - 1); start_band2++) {
                cost_bandsum = classcost[start_band1] + classcost[start_band1 + 1] + classcost[start_band2] + classcost[start_band2 + 1];
                delta_band12 = (start_band2 - start_band1) >(NUM_SAO_BO_CLASSES >> 1) ? (32 - start_band2 + start_band1) : (start_band2 - start_band1);
                assert(delta_band12 >= 0 && delta_band12 <= (NUM_SAO_BO_CLASSES >> 1));
                cost_bandsum += lambda * deltaband_cost[delta_band12];
                if (cost_bandsum < mincost_bandsum) {
                    mincost_bandsum  = cost_bandsum;
                    best_start_band1 = start_band1;
                    best_start_band2 = start_band2;
                }
            }
        }

        for (class_i = 0; class_i < num_class; class_i++) {
            if ((class_i >= best_start_band1 && class_i <= best_start_band1 + 1) || (class_i >= best_start_band2 &&
                    class_i <= best_start_band2 + 1)) {
                continue;
            }
            p_param->offset[class_i] = 0;
        }

        start_band1 = XAVS2_MIN(best_start_band1, best_start_band2);
        start_band2 = XAVS2_MAX(best_start_band1, best_start_band2);
        delta_band12 = (start_band2 - start_band1);
        if (delta_band12 > (NUM_SAO_BO_CLASSES >> 1)) {
            p_param->deltaBand = 32 - delta_band12;  // TODO: 这里应该是 (32 + delta_band12)
            p_param->startBand = start_band2;
        } else {
            p_param->deltaBand = delta_band12;
            p_param->startBand = start_band1;
        }
    } else {
        assert(typeIdc >= SAO_TYPE_EO_0 && typeIdc <= SAO_TYPE_EO_45);
        for (class_i = 0; class_i < num_class; class_i++) {
            if (class_i == SAO_CLASS_EO_PLAIN) {
                p_param->offset[class_i] = 0;
                classcost[class_i] = 0;
            } else {
                p_param->offset[class_i] = (int8_t)offset_estimation(typeIdc, class_i, lambda,
                                           p_param->offset[class_i], p_stat[typeIdc].count[class_i],
                                           p_stat[typeIdc].diff[class_i], &(classcost[class_i]));
            }
        }
        p_param->startBand = 0;
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
long get_distortion(int compIdx, int type, SAOStatData stat_data[NUM_SAO_COMPONENTS][NUM_SAO_NEW_TYPES], SAOBlkParam *sao_cur_param)
{
    int classIdc, bandIdx;
    long dist = 0;
    switch (type) {
    case SAO_TYPE_EO_0:
    case SAO_TYPE_EO_90:
    case SAO_TYPE_EO_135:
    case SAO_TYPE_EO_45:
        for (classIdc = 0; classIdc < NUM_SAO_EO_CLASSES; classIdc++) {
            dist += distortion_cal(stat_data[compIdx][type].count[classIdc], sao_cur_param[compIdx].offset[classIdc], stat_data[compIdx][type].diff[classIdc]);
        }
        break;
    case SAO_TYPE_BO:
        for (classIdc = 0; classIdc < NUM_BO_OFFSET; classIdc++) {
            bandIdx = classIdc % NUM_SAO_BO_CLASSES;
            dist += distortion_cal(stat_data[compIdx][type].count[bandIdx], sao_cur_param[compIdx].offset[bandIdx], stat_data[compIdx][type].diff[bandIdx]);
        }
        break;
    default:
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Not a supported type in get_distortion()");
        exit(-1);
    }

    return dist;
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void copy_sao_param_lcu(SAOBlkParam *saopara_dst, SAOBlkParam *saopara_src)
{
    memcpy(saopara_dst, saopara_src, NUM_SAO_COMPONENTS * sizeof(SAOBlkParam));
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void copy_sao_param_one_comp(SAOBlkParam *saopara_dst, SAOBlkParam *saopara_src)
{
    memcpy(saopara_dst, saopara_src, sizeof(SAOBlkParam));
}

/* ---------------------------------------------------------------------------
 */
static
rdcost_t sao_rdo_new_params(xavs2_t *h, aec_t *p_aec, int avail_left, int avail_up, bool_t *slice_sao_on, rdcost_t sao_lambda,
                            SAOStatData stat_data[NUM_SAO_COMPONENTS][NUM_SAO_NEW_TYPES], SAOBlkParam *sao_cur_param)
{
    ALIGN16(SAOBlkParam temp_sao_param[NUM_SAO_COMPONENTS]);
    rdcost_t total_rdcost = 0;
    int bits;
    int compIdx, type;

    sao_cur_param[SAO_Y].mergeIdx = SAO_MERGE_NONE; // SET AS NOT MERGE MODE
    if (avail_left + avail_up) {
        bits = p_aec->binary.write_sao_mergeflag(p_aec, avail_left, avail_up, &(sao_cur_param[SAO_Y]));
        total_rdcost += bits * sao_lambda;
    }
    for (compIdx = 0; compIdx < 3; compIdx++) {
        if (slice_sao_on[compIdx]) {
            rdcost_t mincost;
            rdcost_t curcost;
            aec_copy_coding_state_sao(&h->cs_data.cs_sao_start, p_aec);
            // for off mode
            sao_cur_param[compIdx].mergeIdx = SAO_MERGE_NONE;
            sao_cur_param[compIdx].typeIdc = SAO_TYPE_OFF;
            bits = p_aec->binary.write_sao_mode(p_aec, &(sao_cur_param[compIdx]));
            mincost = sao_lambda * bits;
            aec_copy_coding_state_sao(&h->cs_data.cs_sao_temp, p_aec);
            // for other normal mode
            for (type = 0; type < 5; type++) {
                if (!h->param->b_fast_sao || tab_sao_check_mode_fast[compIdx][type]) {
                    if (((!IS_ALG_ENABLE(OPT_FAST_SAO)) || (!(!h->fdec->rps.referd_by_others && h->i_type == SLICE_TYPE_B)))) {
                        aec_copy_coding_state_sao(p_aec, &h->cs_data.cs_sao_start);
                        temp_sao_param[compIdx].mergeIdx = SAO_MERGE_NONE;
                        temp_sao_param[compIdx].typeIdc = type;
                        find_offset(type, stat_data[compIdx], &temp_sao_param[compIdx], sao_lambda);
                        curcost = get_distortion(compIdx, type, stat_data, temp_sao_param);

                        bits = p_aec->binary.write_sao_mode(p_aec, &(temp_sao_param[compIdx]));
                        bits += p_aec->binary.write_sao_offset(p_aec, &(temp_sao_param[compIdx]));
                        bits += p_aec->binary.write_sao_type(p_aec, &(temp_sao_param[compIdx]));

                        curcost += sao_lambda * bits;

                        if (curcost < mincost) {
                            mincost = curcost;
                            copy_sao_param_one_comp(&sao_cur_param[compIdx], &temp_sao_param[compIdx]);
                            aec_copy_coding_state_sao(&h->cs_data.cs_sao_temp, p_aec);
                        }
                    }
                }
            }
            aec_copy_coding_state_sao(p_aec, &h->cs_data.cs_sao_temp);
            total_rdcost += mincost;
        }
    }
    return total_rdcost;
}

/* ---------------------------------------------------------------------------
 */
static
void getMergeNeighbor(xavs2_t *h, int lcu_x, int lcu_y, SAOBlkParam (*blk_param)[NUM_SAO_COMPONENTS],
                      int *MergeAvail, SAOBlkParam sao_merge_param[][NUM_SAO_COMPONENTS])
{
    int mb_y = lcu_y << (h->i_lcu_level - MIN_CU_SIZE_IN_BIT);
    int mb_x = lcu_x << (h->i_lcu_level - MIN_CU_SIZE_IN_BIT);
    int mergeup_avail, mergeleft_avail;
    int width_in_smb = h->i_width_in_lcu;

    mergeup_avail   = (mb_y == 0) ? 0 : (cu_get_slice_index(h, mb_x, mb_y) == cu_get_slice_index(h, mb_x, mb_y - 1));
    mergeleft_avail = (mb_x == 0) ? 0 : (cu_get_slice_index(h, mb_x, mb_y) == cu_get_slice_index(h, mb_x - 1, mb_y));

    if (blk_param != NULL) {
        if (mergeleft_avail) {
            copy_sao_param_lcu(sao_merge_param[SAO_MERGE_LEFT], blk_param[-1]);
        }
        if (mergeup_avail) {
            copy_sao_param_lcu(sao_merge_param[SAO_MERGE_ABOVE], blk_param[-width_in_smb]);
        }
    }
    MergeAvail[SAO_MERGE_LEFT] = mergeleft_avail;
    MergeAvail[SAO_MERGE_ABOVE] = mergeup_avail;
}

/* ---------------------------------------------------------------------------
 */
static
rdcost_t sao_rdcost_merge(xavs2_t *h, aec_t *p_aec, rdcost_t sao_labmda,
                          SAOStatData stat_data[NUM_SAO_COMPONENTS][NUM_SAO_NEW_TYPES],
                          SAOBlkParam *sao_cur_param, int merge_avail[NUM_SAO_MERGE_TYPES], int mergeIdx,
                          SAOBlkParam merge_candidate[NUM_SAO_MERGE_TYPES][NUM_SAO_COMPONENTS])
{
    int compIdx;
    int type;
    int currate;
    rdcost_t curcost = 0;

    assert(merge_avail[mergeIdx]);

    copy_sao_param_lcu(sao_cur_param, merge_candidate[mergeIdx]);
    for (compIdx = 0; compIdx < NUM_SAO_COMPONENTS; compIdx++) {
        type = merge_candidate[mergeIdx][compIdx].typeIdc;
        sao_cur_param[compIdx].mergeIdx = SAO_MERGE_LEFT + mergeIdx;

        if (type != SAO_TYPE_OFF && h->slice_sao_on[compIdx] != 0) {
            curcost += get_distortion(compIdx, type, stat_data, sao_cur_param);
        }
    }
    currate = p_aec->binary.write_sao_mergeflag(p_aec, merge_avail[SAO_MERGE_LEFT], merge_avail[SAO_MERGE_ABOVE], sao_cur_param);
    curcost += sao_labmda * currate;

    return curcost;
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void off_sao(SAOBlkParam *saoblkparam)
{
    int i;

    for (i = 0; i < NUM_SAO_COMPONENTS; i++) {
        saoblkparam[i].mergeIdx = SAO_MERGE_NONE;
        saoblkparam[i].typeIdc = SAO_TYPE_OFF;
        saoblkparam[i].startBand = -1;
        saoblkparam[i].deltaBand = -1;
        memset(saoblkparam[i].offset, 0, sizeof(saoblkparam[0].offset));
    }
}

/* ---------------------------------------------------------------------------
 */
static void sao_filter_region(xavs2_t *h, SAOBlkParam *blk_param, int compIdx, sao_region_t *p_region)
{
    ALIGN16(int signupline[MAX_CU_SIZE + SAO_SHIFT_PIX_NUM]);
    const int max_val = ((1 << h->param->sample_bit_depth) - 1);
    int start_x, end_x, start_y, end_y;
    int start_x_r0, end_x_r0, start_x_r, end_x_r, start_x_rn, end_x_rn;
    int x, y;
    int leftsign, rightsign, upsign, downsign;
    int reg = 0;
    int edgetype, bandtype;
    int band_shift;

    int pix_x = p_region->pix_x[compIdx];
    int pix_y = p_region->pix_y[compIdx];
    int width = p_region->width[compIdx];
    int height = p_region->height[compIdx];

    int i_src = h->img_sao->i_stride[compIdx];
    int i_dst = h->fdec->i_stride[compIdx];
    pel_t *dst = h->fdec->planes[compIdx] + pix_y * i_dst + pix_x;
    pel_t *src = h->img_sao->planes[compIdx] + pix_y * i_src + pix_x;

    assert(blk_param->typeIdc != SAO_TYPE_OFF);

    switch (blk_param->typeIdc) {
    case SAO_TYPE_EO_0:
        end_y = height;
        start_x = p_region->b_left ? 0 : 1;
        end_x = p_region->b_right ? width : (width - 1);
        for (y = 0; y < end_y; y++) {
            leftsign = xavs2_sign3(src[start_x] - src[start_x - 1]);
            for (x = start_x; x < end_x; x++) {
                rightsign = xavs2_sign3(src[x] - src[x + 1]);
                edgetype = leftsign + rightsign;
                leftsign = -rightsign;
                dst[x] = (pel_t)XAVS2_CLIP3(0, max_val, src[x] + blk_param->offset[edgetype + 2]);
            }
            src += i_src;
            dst += i_dst;
        }
        break;
    case SAO_TYPE_EO_90: {
        pel_t *src_base = src;
        pel_t *dst_base = dst;
        start_x = 0;
        end_x = width;
        start_y = p_region->b_top ? 0 : 1;
        end_y = p_region->b_down ? height : (height - 1);

        src_base += start_y * i_src;
        dst_base += start_y * i_dst;
        for (x = start_x; x < end_x; x++) {
            src = src_base;
            dst = dst_base;
            upsign = xavs2_sign3(src[0] - src[-i_src]);
            for (y = start_y; y < end_y; y++) {
                downsign = xavs2_sign3(src[0] - src[i_src]);
                edgetype = downsign + upsign;
                upsign = -downsign;
                *dst = (pel_t)XAVS2_CLIP3(0, max_val, src[0] + blk_param->offset[edgetype + 2]);
                src += i_src;
                dst += i_dst;
            }
            src_base++;
            dst_base++;
        }
        break;
    }
    case SAO_TYPE_EO_135: {
        start_x_r0 = p_region->b_top_left ? 0 : 1;
        end_x_r0 = p_region->b_top ? (p_region->b_right ? width : (width - 1)) : 1;
        start_x_r = p_region->b_left ? 0 : 1;
        end_x_r = p_region->b_right ? width : (width - 1);
        start_x_rn = p_region->b_down ? (p_region->b_left ? 0 : 1) : (width - 1);
        end_x_rn = p_region->b_right_down ? width : (width - 1);

        // init the line buffer
        for (x = start_x_r + 1; x < end_x_r + 1; x++) {
            signupline[x] = xavs2_sign3(src[x + i_src] - src[x - 1]);
        }
        // first row
        for (x = start_x_r0; x < end_x_r0; x++) {
            upsign = xavs2_sign3(src[x] - src[x - 1 - i_src]);
            edgetype = upsign - signupline[x + 1];
            dst[x] = (pel_t)XAVS2_CLIP3(0, max_val, src[x] + blk_param->offset[edgetype + 2]);
        }
        // middle rows
        src += i_src;
        dst += i_dst;
        for (y = 1; y < height - 1; y++) {
            x = start_x_r;
            signupline[x] = xavs2_sign3(src[x] - src[x - 1 - i_src]);
            for (; x < end_x_r; x++) {
                downsign = xavs2_sign3(src[x] - src[x + 1 + i_src]);
                edgetype = downsign + signupline[x];
                dst[x] = (pel_t)XAVS2_CLIP3(0, max_val, src[x] + blk_param->offset[edgetype + 2]);
                signupline[x] = reg;
                reg = -downsign;
            }
            dst += i_dst;
            src += i_src;
        }
        // last row
        x = start_x_rn;
        signupline[x] = xavs2_sign3(src[x] - src[x - 1 - i_src]);
        for (; x < end_x_rn; x++) {
            downsign = xavs2_sign3(src[x] - src[x + 1 + i_src]);
            edgetype = downsign + signupline[x];
            dst[x] = (pel_t)XAVS2_CLIP3(0, max_val, src[x] + blk_param->offset[edgetype + 2]);
        }
    }
    break;
    case SAO_TYPE_EO_45: {
        start_x_r0 = p_region->b_top ? (p_region->b_left ? 0 : 1) : (width - 1);
        end_x_r0 = p_region->b_top_right ? width : (width - 1);
        start_x_r = p_region->b_left ? 0 : 1;
        end_x_r = p_region->b_right ? width : (width - 1);
        start_x_rn = p_region->b_down_left ? 0 : 1;
        end_x_rn = p_region->b_down ? (p_region->b_right ? width : (width - 1)) : 1;

        // init the line buffer
        for (x = start_x_r; x < end_x_r; x++) {
            signupline[x] = xavs2_sign3(src[x - 1 + i_src] - src[x]);
        }
        // first row
        for (x = start_x_r0; x < end_x_r0; x++) {
            upsign = xavs2_sign3(src[x] - src[x + 1 - i_src]);
            edgetype = upsign - signupline[x];
            dst[x] = (pel_t)XAVS2_CLIP3(0, max_val, src[x] + blk_param->offset[edgetype + 2]);
        }
        // middle rows
        src += i_src;
        dst += i_dst;
        for (y = 1; y < height - 1; y++) {
            signupline[end_x_r] = xavs2_sign3(src[end_x_r - 1] - src[end_x_r - i_src]);
            for (x = start_x_r; x < end_x_r; x++) {
                downsign = xavs2_sign3(src[x] - src[x - 1 + i_src]);
                edgetype = downsign + signupline[x + 1];
                dst[x] = (pel_t)XAVS2_CLIP3(0, max_val, src[x] + blk_param->offset[edgetype + 2]);
                signupline[x] = -downsign;
            }
            src += i_src;
            dst += i_dst;
        }
        //last row
        for (x = start_x_rn; x < end_x_rn; x++) {
            if (x == end_x_r - 1) {
                upsign = xavs2_sign3(src[x] - src[x + 1 - i_src]);
                signupline[x + 1] = upsign;
            }
            downsign = xavs2_sign3(src[x] - src[x - 1 + i_src]);
            edgetype = downsign + signupline[x + 1];
            dst[x] = (pel_t)XAVS2_CLIP3(0, max_val, src[x] + blk_param->offset[edgetype + 2]);
        }
        break;
    }
    case SAO_TYPE_BO:
        band_shift = (h->param->sample_bit_depth - NUM_SAO_BO_CLASSES_IN_BIT);
        start_x = 0;
        end_x = width;
        start_y = 0;
        end_y = height;
        src += start_y * i_src;
        dst += start_y * i_dst;
        for (y = start_y; y < end_y; y++) {
            for (x = start_x; x < end_x; x++) {
                bandtype = src[x] >> band_shift;
                dst[x] = (pel_t)XAVS2_CLIP3(0, max_val, src[x] + blk_param->offset[bandtype]);
            }
            src += i_src;
            dst += i_dst;
        }
        break;
    default:
        xavs2_log(h, XAVS2_LOG_ERROR, "Not a supported SAO types for SAO_on_Block\n");
        exit(-1);
    }
}

/* ---------------------------------------------------------------------------
 */
static void sao_get_neighbor_avail(xavs2_t *h, sao_region_t *p_avail, int i_lcu_x, int i_lcu_y)
{
    int i_lcu_level = h->i_lcu_level;
    int pix_x = i_lcu_x << i_lcu_level;
    int pix_y = i_lcu_y << i_lcu_level;
    int width  = XAVS2_MIN(1 << i_lcu_level, h->i_width - pix_x);
    int height = XAVS2_MIN(1 << i_lcu_level, h->i_height - pix_y);
    int pix_x_c = pix_x >> 1;
    int pix_y_c = pix_y >> CHROMA_V_SHIFT;
    int width_c = width >> 1;
    int height_c = height >> 1;

    /* 可用性获取 */
    p_avail->b_left = i_lcu_x != 0;
    p_avail->b_top  = i_lcu_y != 0;
    p_avail->b_right = (i_lcu_x < h->i_width_in_lcu - 1);
    p_avail->b_down  = (i_lcu_y < h->i_height_in_lcu - 1);

    if (h->param->b_cross_slice_loop_filter == FALSE) {
        slice_t *slice = h->slices[h->i_slice_index];
        if (p_avail->b_top) {
            p_avail->b_top = (slice->i_first_lcu_y != i_lcu_y);
        }
        if (p_avail->b_down) {
            p_avail->b_down = (slice->i_last_lcu_y != i_lcu_y);
        }
    }

    p_avail->b_top_left = p_avail->b_top && p_avail->b_left;
    p_avail->b_top_right = p_avail->b_top && p_avail->b_right;
    p_avail->b_down_left = p_avail->b_down && p_avail->b_left;
    p_avail->b_right_down = p_avail->b_down && p_avail->b_right;

    /* 滤波区域的调整 */
    if (!p_avail->b_right) {
        width += SAO_SHIFT_PIX_NUM;
        width_c += SAO_SHIFT_PIX_NUM;
    }

    if (!p_avail->b_down) {
        height += SAO_SHIFT_PIX_NUM;
        height_c += SAO_SHIFT_PIX_NUM;
    }

    if (p_avail->b_left) {
        pix_x -= SAO_SHIFT_PIX_NUM;
        pix_x_c -= SAO_SHIFT_PIX_NUM;
    } else {
        width -= SAO_SHIFT_PIX_NUM;
        width_c -= SAO_SHIFT_PIX_NUM;
    }

    if (p_avail->b_top) {
        pix_y -= SAO_SHIFT_PIX_NUM;
        pix_y_c -= SAO_SHIFT_PIX_NUM;
    } else {
        height -= SAO_SHIFT_PIX_NUM;
        height_c -= SAO_SHIFT_PIX_NUM;
    }

    /* make sure the width and height is not outside a picture */
    width    = XAVS2_MIN(width  ,  h->i_width - pix_x);
    width_c  = XAVS2_MIN(width_c, (h->i_width >> 1) - pix_x_c);
    height   = XAVS2_MIN(height  ,  h->i_height - pix_y);
    height_c = XAVS2_MIN(height_c, (h->i_height >> 1) - pix_y_c);

    /* luma component */
    p_avail->pix_x[0] = pix_x;
    p_avail->pix_y[0] = pix_y;
    p_avail->width[0] = width;
    p_avail->height[0] = height;

    /* chroma components */
    p_avail->pix_x[1]  = p_avail->pix_x[2]  = pix_x_c;
    p_avail->pix_y[1]  = p_avail->pix_y[2]  = pix_y_c;
    p_avail->width[1]  = p_avail->width[2]  = width_c;
    p_avail->height[1] = p_avail->height[2] = height_c;
}

/**
 * ===========================================================================
 * interface function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static
void sao_get_param_lcu(xavs2_t *h, aec_t *p_aec, int lcu_x, int lcu_y, bool_t *slice_sao_on,
                       SAOStatData   stat_data [NUM_SAO_COMPONENTS][NUM_SAO_NEW_TYPES],
                       SAOBlkParam (*blk_param)[NUM_SAO_COMPONENTS], rdcost_t sao_labmda)
{
    if (slice_sao_on[0] || slice_sao_on[1] || slice_sao_on[2]) {
        SAOBlkParam sao_cur_param[NUM_SAO_COMPONENTS];
        SAOBlkParam merge_candidate[NUM_SAO_MERGE_TYPES][NUM_SAO_COMPONENTS];
        int merge_avail[NUM_SAO_MERGE_TYPES];
        rdcost_t mcost;
        rdcost_t mincost = MAX_COST;

        getMergeNeighbor(h, lcu_x, lcu_y, blk_param, merge_avail, merge_candidate);

        // backup AEC contexts
        aec_copy_coding_state_sao(&h->cs_data.cs_sao_start, p_aec);

        // MERGE MODE
        if (merge_avail[SAO_MERGE_LEFT]) {
            mincost = sao_rdcost_merge(h, p_aec, sao_labmda, stat_data, sao_cur_param, merge_avail, SAO_MERGE_LEFT, merge_candidate);
            copy_sao_param_lcu(blk_param[0], sao_cur_param);
            aec_copy_coding_state_sao(&h->cs_data.cs_sao_best, p_aec);
            aec_copy_coding_state_sao(p_aec, &h->cs_data.cs_sao_start);
        }
        if (merge_avail[SAO_MERGE_ABOVE]) {
            mcost = sao_rdcost_merge(h, p_aec, sao_labmda, stat_data, sao_cur_param, merge_avail, SAO_MERGE_ABOVE, merge_candidate);
            if (mcost < mincost) {
                mincost = mcost;
                copy_sao_param_lcu(blk_param[0], sao_cur_param);
                aec_copy_coding_state_sao(&h->cs_data.cs_sao_best, p_aec);
            }
            aec_copy_coding_state_sao(p_aec, &h->cs_data.cs_sao_start);
        }

        // NEW MODE
        mcost = sao_rdo_new_params(h, p_aec, merge_avail[SAO_MERGE_LEFT], merge_avail[SAO_MERGE_ABOVE],
                                   slice_sao_on, sao_labmda, stat_data, sao_cur_param);
        if (mcost < mincost) {
            mincost = mcost;
            copy_sao_param_lcu(blk_param[0], sao_cur_param);
            aec_copy_coding_state_sao(&h->cs_data.cs_sao_best, &h->cs_data.cs_sao_temp);
        }

        // RESET ENTROPY CODING
        aec_copy_coding_state_sao(p_aec, &h->cs_data.cs_sao_best);
    } else {
        off_sao(blk_param[0]);
    }
}

/* ---------------------------------------------------------------------------
 */
void write_saoparam_one_lcu(xavs2_t *h, aec_t *p_aec, int lcu_x, int lcu_y, bool_t *slice_sao_on, SAOBlkParam sao_cur_param[NUM_SAO_COMPONENTS])
{
    if (slice_sao_on[0] || slice_sao_on[1] || slice_sao_on[2]) {
        int merge_avail[NUM_SAO_MERGE_TYPES];
        int avail_left, avail_up;

        getMergeNeighbor(h, lcu_x, lcu_y, NULL, merge_avail, NULL);
        avail_left = merge_avail[0];
        avail_up   = merge_avail[1];

        if (avail_left || avail_up) {
            p_aec->binary.write_sao_mergeflag(p_aec, avail_left, avail_up, &sao_cur_param[SAO_Y]);
        }

        if (sao_cur_param[SAO_Y].mergeIdx == SAO_MERGE_NONE) {
            int compIdx;
            for (compIdx = SAO_Y; compIdx < NUM_SAO_COMPONENTS; compIdx++) {
                if (slice_sao_on[compIdx]) {
                    p_aec->binary.write_sao_mode(p_aec, &sao_cur_param[compIdx]);
                    if (sao_cur_param[compIdx].typeIdc != SAO_TYPE_OFF) {
                        p_aec->binary.write_sao_offset(p_aec, &sao_cur_param[compIdx]);
                        p_aec->binary.write_sao_type(p_aec, &sao_cur_param[compIdx]);
                    }
                }
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void sao_slice_onoff_decision(xavs2_t *h, bool_t *slice_sao_on)
{
    const double saorate[NUM_SAO_COMPONENTS] = {SAO_RATE_THR, SAO_RATE_CHROMA_THR, SAO_RATE_CHROMA_THR};
    const int num_lcu = h->i_width_in_lcu * h->i_height_in_lcu;
    int compIdx;

    for (compIdx = 0; compIdx < NUM_SAO_COMPONENTS; compIdx++) {
        if (h->param->chroma_format == CHROMA_420 || compIdx == IMG_Y) {
            slice_sao_on[compIdx] = TRUE;
            if (h->fref[0] != NULL && h->fref[0]->num_lcu_sao_off[compIdx] > num_lcu * saorate[compIdx]) {
                slice_sao_on[compIdx] = FALSE;
            }
        } else {
            slice_sao_on[compIdx] = FALSE;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static
void sao_copy_lcu(xavs2_t *h, xavs2_frame_t *frm_dst, xavs2_frame_t *frm_src, int lcu_x, int lcu_y)
{
    int i_src = frm_src->i_stride[0];
    int i_dst = frm_dst->i_stride[0];
    int start_y = lcu_y << h->i_lcu_level;
    int start_x = lcu_x << h->i_lcu_level;
    int end_y   = XAVS2_MIN(h->i_height, ((lcu_y + 1) << h->i_lcu_level));
    int end_x   = XAVS2_MIN(h->i_width,  ((lcu_x + 1) << h->i_lcu_level));
    int lcu_width  = end_x - start_x;
    int lcu_height;
    int i_first_lcu_y_for_filter = h->param->b_cross_slice_loop_filter ? 0 : h->slices[h->i_slice_index]->i_first_lcu_y;
    int start_y_shift = (lcu_y != i_first_lcu_y_for_filter) ? SAO_SHIFT_PIX_NUM : 0;
    pel_t *p_src;
    pel_t *p_dst;
    pel_t *p_src2, *p_dst2;

    /* luma component */
    start_y -= start_y_shift;
    lcu_height = end_y - start_y;
    p_src = frm_src->planes[0] + start_y * i_src + start_x;
    p_dst = frm_dst->planes[0] + start_y * i_dst + start_x;
    g_funcs.plane_copy(p_dst, i_dst, p_src, i_src, lcu_width, lcu_height);

    /* chroma component */
    start_y = lcu_y << (h->i_lcu_level - CHROMA_V_SHIFT);
    start_y -= start_y_shift;
    end_y   >>= CHROMA_V_SHIFT;
    start_x >>= CHROMA_V_SHIFT;
    end_x   >>= CHROMA_V_SHIFT;

    lcu_width  = end_x - start_x;
    lcu_height = end_y - start_y;
    i_src = frm_src->i_stride[1];
    i_dst = frm_dst->i_stride[1];
    p_src  = frm_src->planes[1] + start_y * i_src + start_x;
    p_src2 = frm_src->planes[2] + start_y * i_src + start_x;
    p_dst  = frm_dst->planes[1] + start_y * i_dst + start_x;
    p_dst2 = frm_dst->planes[2] + start_y * i_dst + start_x;
    g_funcs.plane_copy(p_dst, i_dst, p_src, i_src, lcu_width, lcu_height);
    g_funcs.plane_copy(p_dst2, i_dst, p_src2, i_src, lcu_width, lcu_height);
}

/* ---------------------------------------------------------------------------
 */
void sao_get_lcu_param_after_deblock(xavs2_t *h, aec_t *p_aec, int i_lcu_x, int i_lcu_y)
{
    sao_region_t region;
    int i_lcu_xy = i_lcu_y * h->i_width_in_lcu + i_lcu_x;
    int compIdx, type;

    sao_copy_lcu(h, h->img_sao, h->fdec, i_lcu_x, i_lcu_y);
    sao_get_neighbor_avail(h, &region, i_lcu_x, i_lcu_y);

    for (compIdx = 0; compIdx < 3; compIdx++) {
        if (h->slice_sao_on[compIdx]) {
            for (type = 0; type < 5; type++) {
                if (!h->param->b_fast_sao || tab_sao_check_mode_fast[compIdx][type]) {
                    if (((!IS_ALG_ENABLE(OPT_FAST_SAO)) || (!(!h->fdec->rps.referd_by_others && h->i_type == SLICE_TYPE_B)))) {
                        gf_sao_stat[type](h->img_sao, h->fenc, &h->sao_stat_datas[i_lcu_xy][compIdx][type], &region, compIdx);
                    }
                    // SAOStatData tmp;
                    // memset(&tmp, 0, sizeof(tmp));
                    // gf_sao_stat[type](h->fdec, h->fenc, &tmp, &region, compIdx);
                    // if (memcmp(&tmp, &h->sao_stat_datas[i_lcu_xy][compIdx][type], sizeof(tmp)) != 0) {
                    //     xavs2_log(h, XAVS2_LOG_ERROR, "SAO mismatch!\n");
                    //     gf_sao_stat[type](h->img_sao, h->fenc, &h->sao_stat_datas[i_lcu_xy][compIdx][type], &region, compIdx);
                    //     gf_sao_stat[type](h->fdec, h->fenc, &tmp, &region, compIdx);
                    // }
                }
            }
        }
    }

    sao_get_param_lcu(h, p_aec, i_lcu_x, i_lcu_y, h->slice_sao_on,
                      h->sao_stat_datas[i_lcu_xy],
                      &h->sao_blk_params[i_lcu_xy], h->f_lambda_mode);
}

/* ---------------------------------------------------------------------------
 */
void sao_filter_lcu(xavs2_t *h, SAOBlkParam blk_param[NUM_SAO_COMPONENTS], int lcu_x, int lcu_y)
{
    sao_region_t region;
    SAOBlkParam *p_param = blk_param;
    int compIdx;

    sao_get_neighbor_avail(h, &region, lcu_x, lcu_y);

    for (compIdx = 0; compIdx < NUM_SAO_COMPONENTS; compIdx++) {
        if (h->slice_sao_on[compIdx] == 0 || p_param[compIdx].typeIdc == SAO_TYPE_OFF) {
            continue;
        }
        int pix_y = region.pix_y[compIdx];
        int pix_x = region.pix_x[compIdx];
        int i_dst = h->fdec->i_stride[compIdx];
        int i_src = h->img_sao->i_stride[compIdx];
        pel_t *dst = h->fdec->planes[compIdx]    + pix_y * i_dst + pix_x;
        pel_t *src = h->img_sao->planes[compIdx] + pix_y * i_src + pix_x;
        int avail[8];
        avail[0] = region.b_top;
        avail[1] = region.b_down;
        avail[2] = region.b_left;
        avail[3] = region.b_right;
        avail[4] = region.b_top_left;
        avail[5] = region.b_top_right;
        avail[6] = region.b_down_left;
        avail[7] = region.b_right_down;
        g_funcs.sao_block(dst, i_dst, src, i_src,
                          region.width[compIdx], region.height[compIdx],
                          avail, &p_param[compIdx]);

    }
}


