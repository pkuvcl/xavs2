/*
 * me.c
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

#include "common.h"
#include "block_info.h"
#include "me.h"
#include "common/cpu.h"
#include "common/mc.h"
#include "predict.h"


/**
 * ===========================================================================
 * type defines
 * ===========================================================================
 */

typedef struct mv_info {
    int     bmx;                /* best mv (x)    */
    int     bmy;                /* best mv (y)    */
    int     bdir;               /* best direction */
    dist_t  bcost;              /* best cost      */
    dist_t  bdist;              /* best distance  */
} mv_info;


/**
 * ===========================================================================
 * local/global variables
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * big hexagon for UMH */
static const int8_t HEX4[16][2] = {
    {  0, -4 }, { 0,  4 }, { -2, -3 }, { 2, -3 },
    { -4, -2 }, { 4, -2 }, { -4, -1 }, { 4, -1 },
    { -4,  0 }, { 4,  0 }, { -4,  1 }, { 4,  1 },
    { -4,  2 }, { 4,  2 }, { -2,  3 }, { 2,  3 }
};

static const int8_t FAST_HEX4[8][2] = {
    {  0, -4 }, { 0,  4 }, { -2, -3 }, { 2, -3 },
    { -4,  0 }, { 4,  0 }, { -2,  3 }, { 2,  3 }
};

/* ---------------------------------------------------------------------------
 * radius 2 hexagon
 * repeated entries are to avoid having to compute mod6 every time */
const int8_t HEX2[8][2] = {
    {-1, -2},       /* 0,       0(6)   5        */
    {-2,  0},       /* 1,                       */
    {-1,  2},       /* 2,                       */
    { 1,  2},       /* 3,   1(7)   *       4    */
    { 2,  0},       /* 4,                       */
    { 1, -2},       /* 5,                       */
    {-1, -2},       /* 6,       2      3        */
    {-2,  0}        /* 7,                       */
};

/* ---------------------------------------------------------------------------
 * (x - 1) % 6 */
const int8_t M1MOD6[8] = { 5, 0, 1, 2, 3, 4, 5, 0 };

/* ---------------------------------------------------------------------------
 * radius 1 diamond
 * repeated entries are to avoid having to compute mod4 every time */
const int8_t DIA1[6][2] = {
    { 0, -1},       /* 0,                       */
    {-1,  0},       /* 1,           0(4)        */
    { 0,  1},       /* 2,                       */
    { 1,  0},       /* 3,      1(5) *   3       */
    { 0, -1},       /* 4,                       */
    {-1,  0}        /* 5,           2           */
};

/* ---------------------------------------------------------------------------
 * (x - 1) % 4 */
const int8_t M1MOD4[6] = { 3, 0, 1, 2, 3, 0 };

/* ---------------------------------------------------------------------------
 * uneven multi-hexagon-grid: 5x5 */
static int8_t GRID[24][2] = {
    { -1, -1 }, {  0, -1 }, { 1, -1 },  /* inside 8 points */
    { -1,  0 },             { 1,  0 },
    { -1,  1 }, {  0,  1 }, { 1,  1 },
    { -2, -2 }, { -1, -2 }, { 0,  2 }, { 1, -2 }, { 2, -2 }, /* outside 16 points */
    { -2, -1 },                                   { 2, -1 },
    { -2,  0 },                                   { 2,  0 },
    { -2,  1 },                                   { 2,  1 },
    { -2,  2 }, { -1,  2 }, { 0,  2 }, { 1,  2 }, { 2,  2 }
};

/* ---------------------------------------------------------------------------
 * 用于分像素搜索的正方形搜索 */
static const int8_t Spiral[9][2] = {
    {  0,  0 }, {  0, -1 }, {  0, 1 },
    { -1, -1 }, {  1, -1 }, { -1, 0 },
    {  1,  0 }, { -1,  1 }, {  1, 1 }
};

static const int8_t Spiral2[9][2] = {
    {  0,  0 }, {  0, -1 }, { -1, -1 },      /* 2 1 8 */
    { -1,  0 }, { -1,  1 }, {  0,  1 },      /* 3 0 7 */
    {  1,  1 }, {  1,  0 }, {  1, -1 }       /* 4 5 6 */
};


/* ---------------------------------------------------------------------------
 * offsets for Two Point Search (TZ) */
static const int offsets[16][2] = {
    { -1,  0 }, {  0, -1 },
    { -1, -1 }, {  1, -1 },
    { -1,  0 }, {  1,  0 },
    { -1,  1 }, { -1, -1 },
    {  1, -1 }, {  1,  1 },
    { -1,  0 }, {  0,  1 },
    { -1,  1 }, {  1,  1 },
    {  1,  0 }, {  0,  1 },
};

static const int i_org = FENC_STRIDE;

/**
 * ===========================================================================
 * macros
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * early termination */
#define EARLY_TERMINATION(pred_sad) \
    if (bcost < (pred_sad) * beta3) {\
        goto umh_step_3;\
    } else if (bcost < (pred_sad) * beta2) {\
        goto umh_step_2;\
    }


/**
 * ===========================================================================
 * calculate cost for integer pixel motion search
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
#define CAL_COST_IPEL(mx, my) \
    g_funcs.pixf.sad[i_pixel](p_org, i_org,\
        p_fref + (my) * i_fref + (mx), i_fref) + MV_COST_IPEL(mx, my)

/* ---------------------------------------------------------------------------
 */
#define ME_COST_IPEL(mx, my) \
    if (CHECK_MV_RANGE(mx, my)) {\
        int cost = g_funcs.pixf.sad[i_pixel](p_org, i_org,\
                   p_fref + (my) * i_fref + (mx), i_fref) + MV_COST_IPEL(mx, my);\
        COPY3_IF_LT(bcost, cost, bmx, mx, bmy, my);\
    }

/* ---------------------------------------------------------------------------
 */
#define ME_COST_IPEL_DIR(mx, my, d) \
    if (CHECK_MV_RANGE(mx, my)) {\
        int cost = g_funcs.pixf.sad[i_pixel](p_org, i_org,\
                   p_fref + (my) * i_fref + (mx), i_fref) + MV_COST_IPEL(mx, my);\
        COPY4_IF_LT(bcost, cost, bmx, mx, bmy, my, dir, d);\
    }

/* ---------------------------------------------------------------------------
 */
#define ME_COST_IPEL_X3(m0x, m0y, m1x, m1y, m2x, m2y) \
{\
    pel_t *pix_base = p_fref + omy * i_fref + omx;\
    g_funcs.pixf.sad_x3[i_pixel](p_org,\
        pix_base + (m0y) * i_fref + (m0x),\
        pix_base + (m1y) * i_fref + (m1x),\
        pix_base + (m2y) * i_fref + (m2x),\
        i_fref, costs);\
    costs[0] += MV_COST_IPEL(omx + (m0x), omy + (m0y));\
    costs[1] += MV_COST_IPEL(omx + (m1x), omy + (m1y));\
    costs[2] += MV_COST_IPEL(omx + (m2x), omy + (m2y));\
    COPY3_IF_LT(bcost, costs[0], bmx, omx + (m0x), bmy, omy + (m0y));\
    COPY3_IF_LT(bcost, costs[1], bmx, omx + (m1x), bmy, omy + (m1y));\
    COPY3_IF_LT(bcost, costs[2], bmx, omx + (m2x), bmy, omy + (m2y));\
}

/* ---------------------------------------------------------------------------
 */
#define ME_COST_IPEL_X3_DIR(m0x, m0y, d0, m1x, m1y, d1, m2x, m2y, d2) \
{\
    pel_t *pix_base = p_fref + omy * i_fref + omx;\
    g_funcs.pixf.sad_x3[i_pixel](p_org,\
        pix_base + (m0y) * i_fref + (m0x),\
        pix_base + (m1y) * i_fref + (m1x),\
        pix_base + (m2y) * i_fref + (m2x),\
        i_fref, costs);\
    costs[0] += MV_COST_IPEL(omx + (m0x), omy + (m0y));\
    costs[1] += MV_COST_IPEL(omx + (m1x), omy + (m1y));\
    costs[2] += MV_COST_IPEL(omx + (m2x), omy + (m2y));\
    COPY4_IF_LT(bcost, costs[0], bmx, omx + (m0x), bmy, omy + (m0y), dir, d0);\
    COPY4_IF_LT(bcost, costs[1], bmx, omx + (m1x), bmy, omy + (m1y), dir, d1);\
    COPY4_IF_LT(bcost, costs[2], bmx, omx + (m2x), bmy, omy + (m2y), dir, d2);\
}

/* ---------------------------------------------------------------------------
 */
#define ME_COST_IPEL_X4(m0x, m0y, m1x, m1y, m2x, m2y, m3x, m3y) \
{\
    if (CHECK_MV_RANGE_X4(m0x, m0y, m1x, m1y, m2x, m2y, m3x, m3y)) {  \
        pel_t *pix_base = p_fref + omy * i_fref + omx;\
        g_funcs.pixf.sad_x4[i_pixel](p_org,\
            pix_base + (m0y) * i_fref + (m0x),\
            pix_base + (m1y) * i_fref + (m1x),\
            pix_base + (m2y) * i_fref + (m2x),\
            pix_base + (m3y) * i_fref + (m3x),\
            i_fref, costs);\
        costs[0] += MV_COST_IPEL(omx + (m0x), omy + (m0y));\
        costs[1] += MV_COST_IPEL(omx + (m1x), omy + (m1y));\
        costs[2] += MV_COST_IPEL(omx + (m2x), omy + (m2y));\
        costs[3] += MV_COST_IPEL(omx + (m3x), omy + (m3y));\
        COPY3_IF_LT(bcost, costs[0], bmx, omx + (m0x), bmy, omy + (m0y));\
        COPY3_IF_LT(bcost, costs[1], bmx, omx + (m1x), bmy, omy + (m1y));\
        COPY3_IF_LT(bcost, costs[2], bmx, omx + (m2x), bmy, omy + (m2y));\
        COPY3_IF_LT(bcost, costs[3], bmx, omx + (m3x), bmy, omy + (m3y));\
    } else {                    \
        ME_COST_IPEL(m0x, m0y); \
        ME_COST_IPEL(m1x, m1y); \
        ME_COST_IPEL(m2x, m2y); \
        ME_COST_IPEL(m3x, m3y); \
    } \
}

/* ---------------------------------------------------------------------------
 */
#define ME_COST_IPEL_X4_DIR(m0x, m0y, d0, m1x, m1y, d1, m2x, m2y, d2, m3x, m3y, d3) \
{\
    pel_t *pix_base = p_fref + omy * i_fref + omx;\
    g_funcs.pixf.sad_x4[i_pixel](p_org,\
        pix_base + (m0y) * i_fref + (m0x),\
        pix_base + (m1y) * i_fref + (m1x),\
        pix_base + (m2y) * i_fref + (m2x),\
        pix_base + (m3y) * i_fref + (m3x), i_fref, costs);\
    costs[0] += MV_COST_IPEL(omx + (m0x), omy + (m0y));\
    costs[1] += MV_COST_IPEL(omx + (m1x), omy + (m1y));\
    costs[2] += MV_COST_IPEL(omx + (m2x), omy + (m2y));\
    costs[3] += MV_COST_IPEL(omx + (m3x), omy + (m3y));\
    COPY4_IF_LT(bcost, costs[0], bmx, omx + (m0x), bmy, omy + (m0y), dir, d0);\
    COPY4_IF_LT(bcost, costs[1], bmx, omx + (m1x), bmy, omy + (m1y), dir, d1);\
    COPY4_IF_LT(bcost, costs[2], bmx, omx + (m2x), bmy, omy + (m2y), dir, d2);\
    COPY4_IF_LT(bcost, costs[3], bmx, omx + (m3x), bmy, omy + (m3y), dir, d3);\
}

/* ---------------------------------------------------------------------------
 * for TZ */
#define ME_COST_IPEL_DIR_DIST(mx, my, direction, dist) \
    if (CHECK_MV_RANGE(mx, my)) {\
        int cost = g_funcs.pixf.sad[i_pixel](p_org, i_org,\
                   p_fref + (my) * i_fref + (mx), i_fref) + MV_COST_IPEL(mx, my);\
        COPY5_IF_LT(mv->bcost, cost, mv->bmx, mx, mv->bmy, my, mv->bdir, direction, mv->bdist, dist);\
    }

/* ---------------------------------------------------------------------------
 * for TZ */
#define ME_COST_IPEL_X4_DIR_DIST(m0x, m0y, p0, d0, m1x, m1y, p1, d1, m2x, m2y, p2, d2, m3x, m3y, p3, d3) \
{\
    g_funcs.pixf.sad_x4[i_pixel](p_org,\
        p_fref + (m0x) + (m0y) * i_fref,\
        p_fref + (m1x) + (m1y) * i_fref,\
        p_fref + (m2x) + (m2y) * i_fref,\
        p_fref + (m3x) + (m3y) * i_fref,\
        i_fref, costs);\
    (costs)[0] += MV_COST_IPEL(m0x, m0y);\
    (costs)[1] += MV_COST_IPEL(m1x, m1y);\
    (costs)[2] += MV_COST_IPEL(m2x, m2y);\
    (costs)[3] += MV_COST_IPEL(m3x, m3y);\
    if (CHECK_MV_RANGE(m0x,m0y)) {\
        COPY5_IF_LT(mv->bcost, costs[0], mv->bmx, m0x, mv->bmy, m0y, mv->bdir, p0, mv->bdist, d0);\
    }\
    if (CHECK_MV_RANGE(m1x,m1y)) {\
        COPY5_IF_LT(mv->bcost, costs[1], mv->bmx, m1x, mv->bmy, m1y, mv->bdir, p1, mv->bdist, d1);\
    }\
    if (CHECK_MV_RANGE(m2x,m2y)) {\
        COPY5_IF_LT(mv->bcost, costs[2], mv->bmx, m2x, mv->bmy, m2y, mv->bdir, p2, mv->bdist, d2);\
    }\
    if (CHECK_MV_RANGE(m3x,m3y)) {\
        COPY5_IF_LT(mv->bcost, costs[3], mv->bmx, m3x, mv->bmy, m3y, mv->bdir, p3, mv->bdist, d3);\
    }\
}

/* ---------------------------------------------------------------------------
 * diamond:     1
 *            1 0 1
 *              1    */
#define DIA_ITER(mx, my) \
{\
    omx = mx;\
    omy = my;\
    ME_COST_IPEL_X4(0,-1, -1,0, 1,0, 0,1);\
}


/**
 * ===========================================================================
 * calculate cost for fractional pixel refine
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
#define ME_COST_QPEL(mx, my) \
{\
    pel_t *p_pred = p_filtered[(((my) & 3) << 2) + ((mx) & 3)] + i_offset\
                  + ((my) >> 2) * i_fref + ((mx) >> 2); \
    cost = g_funcs.pixf.fpel_cmp[i_pixel](p_org, i_org, p_pred, i_fref) + MV_COST_FPEL(mx, my);\
}

/* ---------------------------------------------------------------------------
 */
#define ME_COST_QPEL_SYM \
{\
    int mx_sym;\
    int my_sym;\
    \
    cost = MAX_DISTORTION;\
    if (h->i_type == SLICE_TYPE_B) {\
        mx_sym = -scale_mv_skip  (   mx, distance_bwd, distance_fwd);\
        my_sym = -scale_mv_skip_y(h, my, distance_bwd, distance_fwd);\
    } else {\
        mx_sym = scale_mv_skip  (   mx, distance_bwd, distance_fwd);\
        my_sym = scale_mv_skip_y(h, my, distance_bwd, distance_fwd);\
    }\
    \
    if (CHECK_MV_RANGE(mx, my) && CHECK_MV_RANGE(mx_sym, my_sym)) {\
        int xx1 = mx     >> 2;\
        int yy1 = my     >> 2;\
        int xx2 = mx_sym >> 2;\
        int yy2 = my_sym >> 2;\
        pel_t *p_src1 = p_filtered1[((my     & 3) << 2) + (mx     & 3)]; \
        pel_t *p_src2 = p_filtered2[((my_sym & 3) << 2) + (mx_sym & 3)]; \
        pel_t *p_pred = buf_pixel_temp;\
        \
        if (p_src1 != NULL && p_src2 != NULL) { \
            p_src1 += i_offset + yy1 * i_fref + xx1;\
            p_src2 += i_offset + yy2 * i_fref + xx2;\
            g_funcs.pixf.avg[i_pixel](p_pred, 64, p_src1, i_fref, p_src2, i_fref, 32); \
            cost = g_funcs.pixf.fpel_cmp[i_pixel](p_org, i_org, p_pred, MAX_CU_SIZE)\
                 + MV_COST_FPEL(mx, my);\
        } \
    }\
}

/* ---------------------------------------------------------------------------
 */
#define ME_COST_QPEL_BID \
    if (CHECK_MV_RANGE(mx, my) && CHECK_MV_RANGE(mx_bid, my_bid)) {\
        int xx1 = mx     >> 2;\
        int yy1 = my     >> 2;\
        pel_t *p_src1 = p_filtered1[((my     & 3) << 2) + (mx     & 3)] + i_offset + yy1 * i_fref + xx1;\
        int distortion = g_funcs.pixf.fpel_cmp[i_pixel](buf_pixel_temp, MAX_CU_SIZE, p_src1, i_fref) >> 1;\
        \
        cost = distortion + MV_COST_FPEL(mx, my) + mv_bid_bit;\
    } else {\
        cost = MAX_DISTORTION;\
    }


/**
 * ===========================================================================
 * local function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * 若candMV超过1/4精度的阈值，则新的MV应采用2倍步长，若此时新的MV在阈值范围内，则返回1，表示新的MV不应继续搜索
 * 若candMV在1/4精度阈值范围内，则新的MV采用单倍步长，此时若新MV超过阈值范围，则返回1，表示新的MV不应继续搜索
 * 否则，返回0值表示新的MV应该继续被搜索
 */
static int pmvr_adapt_mv(int *mx, int *my, int ctr_x, int ctr_y,
                         int mv_x, int mv_y, int step_x, int step_y)
{
    if (XAVS2_ABS(mv_x - ctr_x) > TH_PMVR || XAVS2_ABS(mv_y - ctr_y) > TH_PMVR) {
        *mx = mv_x + step_x * 2;
        *my = mv_y + step_y * 2;
        return (XAVS2_ABS(*mx - ctr_x) <= TH_PMVR && XAVS2_ABS(*my - ctr_y) <= TH_PMVR);
    } else {
        *mx = mv_x + step_x;
        *my = mv_y + step_y;
        return (XAVS2_ABS(*mx - ctr_x) > TH_PMVR || XAVS2_ABS(*my - ctr_y) > TH_PMVR);
    }
}

/* ---------------------------------------------------------------------------
 */
static int ALWAYS_INLINE
mv_roundclip(int16_t (*dst)[2], int16_t (*mvc)[2], int i_mvc, int mv_min[2], int mv_max[2], uint32_t pmv)
{
    int cnt = 0;
    int i;

    for (i = 0; i < i_mvc; i++) {
        int mx = IPEL(mvc[i][0]);
        int my = IPEL(mvc[i][1]);
        uint32_t mv = MAKEDWORD(mx, my);

        if (!mv || mv == pmv) {
            continue;
        }

        dst[cnt][0] = (int16_t)XAVS2_CLIP3(mv_min[0], mv_max[0], mx);
        dst[cnt][1] = (int16_t)XAVS2_CLIP3(mv_min[1], mv_max[1], my);
        cnt++;
    }

    return cnt;
}

/* ---------------------------------------------------------------------------
 */
static int ALWAYS_INLINE
mv_clip(int16_t (*dst)[2], int16_t (*mvc)[2], int i_mvc, int mv_min[2], int mv_max[2], uint32_t pmv)
{
    int cnt = 0;
    int i;

    for (i = 0; i < i_mvc; i++) {
        int mx = mvc[i][0];
        int my = mvc[i][1];
        uint32_t mv = M32(mvc[i]);

        if (!mv || mv == pmv) {
            continue;
        }

        dst[cnt][0] = (int16_t)XAVS2_CLIP3(mv_min[0], mv_max[0], mx);
        dst[cnt][1] = (int16_t)XAVS2_CLIP3(mv_min[1], mv_max[1], my);
        cnt++;
    }

    return cnt;
}

/* ---------------------------------------------------------------------------
 * sub pixel block motion search
 */
static dist_t
me_subpel_refine(xavs2_t *h, xavs2_me_t *p_me)
{
#if !ENABLE_FRAME_SUBPEL_INTPL
    ALIGN32(pel_t p_pred[MAX_CU_SIZE * MAX_CU_SIZE]);
#endif
    pel_t  *p_org     = p_me->p_fenc;
    pel_t **p_filtered = p_me->p_fref_1st->filtered;
    int i_fref   = p_me->p_fref_1st->i_stride[IMG_Y];
    int pmx      = p_me->mvp.x;
    int pmy      = p_me->mvp.y;
    int i_pixel  = p_me->i_pixel;
    int i_offset = p_me->i_bias;
    const uint16_t *p_cost_mvx = h->mvbits - p_me->mvp.x;
    const uint16_t *p_cost_mvy = h->mvbits - p_me->mvp.y;
    int lambda = h->i_lambda_factor;
    const int search_pos2 = 9;
    const int search_pos4 = 9;
    const int search_step = h->use_fast_sub_me ? 2 : 1;
    const int8_t(*search_pattern)[2] = h->use_fast_sub_me ? Spiral2 : Spiral;
    dist_t bcost;
    int ctr_x = (pmx >> 1) << 1;
    int ctr_y = (pmy >> 1) << 1;
    int pos, cost;
    int mx, my, bmx, bmy;
    mv_t bmv;

    // convert search center to quarter-pel units
    bmx = p_me->bmv.x;
    bmy = p_me->bmv.y;
    bmv = p_me->bmv;

    if (h->param->enable_hadamard) {
        ME_COST_QPEL(bmx, bmy);
        bcost = cost;
    } else {
        bcost = p_me->bcost;
    }

    /* -------------------------------------------------------------
     * half-pel refine */

    // loop over search positions
    for (pos = 1; pos < search_pos2; pos += search_step) {
        mx = bmx + (search_pattern[pos][0] << 1);
        my = bmy + (search_pattern[pos][1] << 1);
#if ENABLE_FRAME_SUBPEL_INTPL
        ME_COST_QPEL(mx, my);
#else
        mv_t mvt;
        mvt.v = MAKEDWORD(mx, my);
        get_mv_for_mc(h, &mvt, p_me->i_pix_x, p_me->i_pix_y, p_me->i_block_w, p_me->i_block_h);
        mc_luma(p_pred, MAX_CU_SIZE, mvt.x, mvt.y, p_me->i_block_w, p_me->i_block_h, p_me->p_fref_1st);
        cost = g_funcs.pixf.fpel_cmp[i_pixel](p_org, i_org, p_pred, MAX_CU_SIZE) + MV_COST_FPEL(mx, my);
#endif
        if (cost < bcost) {
            bcost = cost;
            bmv.v = MAKEDWORD(mx, my);
        }
    }

    bmx = bmv.x;
    bmy = bmv.y;

    /* -------------------------------------------------------------
     * quarter-pel refine */

    if (h->use_fractional_me > 1) {
        // loop over search positions
        for (pos = 1; pos < search_pos4; pos += search_step) {
            if (h->param->enable_pmvr) {
                if (pmvr_adapt_mv(&mx, &my, ctr_x, ctr_y, bmx, bmy, search_pattern[pos][0], search_pattern[pos][1])) {
                    continue;
                }
            } else {
                mx = bmx + search_pattern[pos][0];    // quarter-pel units
                my = bmy + search_pattern[pos][1];    // quarter-pel units
            }

            // set motion vector cost
#if ENABLE_FRAME_SUBPEL_INTPL
            ME_COST_QPEL(mx, my);
#else
            mv_t mvt;
            mvt.v = MAKEDWORD(mx, my);
            get_mv_for_mc(h, &mvt, p_me->i_pix_x, p_me->i_pix_y, p_me->i_block_w, p_me->i_block_h);
            mc_luma(p_pred, MAX_CU_SIZE, mvt.x, mvt.y, p_me->i_block_w, p_me->i_block_h, p_me->p_fref_1st);
            cost = g_funcs.pixf.fpel_cmp[i_pixel](p_org, i_org, p_pred, MAX_CU_SIZE) + MV_COST_FPEL(mx, my);
#endif
            if (cost < bcost) {
                bcost = cost;
                bmv.v = MAKEDWORD(mx, my);
            }
        }
    }
    // save the results
    p_me->bmv   = bmv;
    p_me->bcost = bcost;
    p_me->mvcost[PDIR_FWD] = MV_COST_FPEL(bmv.x,bmv.y);
    return bcost;
}


/**
 * ===========================================================================
 * interface function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * initialize the motion search
 */
int xavs2_me_get_buf_size(const xavs2_param_t *param)
{
    int me_range    = XAVS2_MAX(256, param->search_range);
    int subpel_num  = 4 * (2 * me_range + 3);
    int max_mv_bits = 5 + 2 * (int)ceil(log(subpel_num + 1) / log(2) + 1e-10);
    int max_mvd     = (1 << ((max_mv_bits >> 1))) - 1;
    int mem_size;

    /* buffer size for mvbits */
    mem_size = (max_mvd * 2 + 1) * sizeof(uint16_t) + CACHE_LINE_SIZE;

    return mem_size;
}

/**
 * ---------------------------------------------------------------------------
 * Function   : initialize the motion search module
 * Parameters :
 *      [in ] : h   - pointer to struct xavs2_t, the HiPE encoder
 *      [out] : none
 * Return     : none
 * ---------------------------------------------------------------------------
 */
void xavs2_me_init(xavs2_t *h, uint8_t **mem_base)
{
    uint8_t *mbase  = *mem_base;
    int me_range    = XAVS2_MAX(256, h->param->search_range);
    int subpel_num  = 4 * (2 * me_range + 3);
    int max_mv_bits = 5 + 2 * (int)ceil(log(subpel_num + 1) / log(2) + 1e-10);
    int max_mvd     = (1 << ((max_mv_bits >> 1))) - 1;
    int bits, i, imin, imax;

    /* set pointer of mvbits */
    h->mvbits  = (uint16_t *)mbase;
    h->mvbits += max_mvd;       // reset the array offset
    mbase     += (max_mvd * 2 + 1) * sizeof(uint16_t);
    ALIGN_POINTER(mbase);

    *mem_base = mbase;

    // init array of motion vector bits
    h->mvbits[0] = 1;
    for (bits = 3; bits <= max_mv_bits; bits += 2) {
        imax = 1 << (bits >> 1);
        imin = imax >> 1;

        for (i = imin; i < imax; i++) {
            h->mvbits[-i] = h->mvbits[i] = (uint16_t)bits;
        }
    }
}

/* ---------------------------------------------------------------------------
 * i_qp: QP of P/F frame
 * TODO: call this function before encoding P/F frame?
 */
void xavs2_me_init_umh_threshold(xavs2_t *h, double *bsize, int i_qp)
{
    const int quant_coef[6] = { 13107, 11916, 10082, 9362, 8192, 7282 };
    int gb_qp_per = i_qp / 6;
    int gb_qp_rem = i_qp % 6;
    int gb_q_bits = 15 + gb_qp_per;
    int gb_qp_const = (h->i_type == SLICE_TYPE_I) ? ((1 << gb_q_bits) / 3) : ((1 << gb_q_bits) / 6);
    int threshold_4x4 = ((1 << gb_q_bits) - gb_qp_const) / quant_coef[gb_qp_rem];
    double quantize_step = threshold_4x4 / (4 * 5.61f);

    memset(bsize, 0, MAX_INTER_MODES * sizeof(double));
    bsize[PRED_nRx2N] = (16 * 16) * quantize_step;
    bsize[PRED_nLx2N] = 4 * bsize[PRED_nRx2N];
    bsize[PRED_2NxnD] = 4 * bsize[PRED_nRx2N];
    bsize[PRED_2NxnU] = 4 * bsize[PRED_2NxnD];
    bsize[PRED_Nx2N ] = 4 * bsize[PRED_2NxnU];
    bsize[PRED_2NxN ] = 4 * bsize[PRED_2NxnU];
    bsize[PRED_2Nx2N] = 4 * bsize[PRED_2NxN ];
}

/* ---------------------------------------------------------------------------
 */
static void tz_pattern_search(xavs2_t* h,
                              xavs2_me_t *p_me,
                              pel_t*    p_org,
                              pel_t*    p_fref,
                              mv_info*  mv,
                              int       mv_x_min,
                              int       mv_y_min,
                              int       mv_x_max,
                              int       mv_y_max,
                              int       i_pixel,
                              int       i_fref,
                              int       earlyExitIters,
                              int       merange)
{
    ALIGN16(int costs[16]);
    const uint32_t mv_min = pack16to32_mask2(-mv_x_min, -mv_y_min);
    const uint32_t mv_max = pack16to32_mask2(mv_x_max, mv_y_max) | 0x8000;
    const uint16_t *p_cost_mvx = h->mvbits - p_me->mvp.x;
    const uint16_t *p_cost_mvy = h->mvbits - p_me->mvp.y;
    int lambda = h->i_lambda_factor;
    int rounds = 0;
    int dist   = 1;
    int omx    = mv->bmx;
    int omy    = mv->bmx;
    dist_t bcost  = mv->bcost;
    int top    = omy - dist;
    int bottom = omy + dist;
    int left   = omx - dist;
    int right  = omx + dist;
    int top2, bottom2, left2, right2;
    int posYT, posYB, posXL, posXR;
    int idx;

    if (top >= mv_y_min && left >= mv_x_min && right <= mv_x_max && bottom <= mv_y_max) {
        ME_COST_IPEL_X4_DIR_DIST(omx,   top,    2, dist,      /* direction */
                                 left,  omy,    4, dist,      /*     2     */
                                 right, omy,    5, dist,      /*   4 * 5   */
                                 omx,   bottom, 7, dist);     /*     7     */
    } else {
        if (top >= mv_y_min) {        // check top
            ME_COST_IPEL_DIR_DIST(omx, top, 2, dist);
        }
        if (left >= mv_x_min) {       // check middle left
            ME_COST_IPEL_DIR_DIST(left, omy, 4, dist);
        }
        if (right <= mv_x_max) {      // check middle right
            ME_COST_IPEL_DIR_DIST(right, omy, 5, dist);
        }
        if (bottom <= mv_y_max) {     // check bottom
            ME_COST_IPEL_DIR_DIST(omx, bottom, 7, dist);
        }
    }

    if (mv->bcost < bcost) {
        rounds = 0;
    } else if (++rounds >= earlyExitIters) {
        return;
    }

    for (dist = 2; dist <= 8; dist <<= 1) {
        /*          2           points 2, 4, 5, 7 are dist
         *        1   3         points 1, 3, 6, 8 are dist/2
         *      4   *   5
         *        6   8
         *          7           */
        omx     = mv->bmx;
        omy     = mv->bmx;
        bcost   = mv->bcost;
        top     = omy - dist;
        bottom  = omy + dist;
        left    = omx - dist;
        right   = omx + dist;
        top2    = omy - (dist >> 1);
        bottom2 = omy + (dist >> 1);
        left2   = omx - (dist >> 1);
        right2  = omx + (dist >> 1);

        // check border
        if (top >= mv_y_min && left >= mv_x_min && right <= mv_x_max && bottom <= mv_y_max) {
            ME_COST_IPEL_X4_DIR_DIST(omx,    top,     2, dist,
                                     left2,  top2,    1, dist >> 1,
                                     right2, top2,    3, dist >> 1,
                                     left,   omy,     4, dist);
            ME_COST_IPEL_X4_DIR_DIST(right,  omy,     5, dist,
                                     left2,  bottom2, 6, dist >> 1,
                                     right2, bottom2, 8, dist >> 1,
                                     omx,    bottom,  7, dist);
        } else {
            if (top >= mv_y_min) {              // check top
                ME_COST_IPEL_DIR_DIST(omx, top, 2, dist);
            }
            if (top2 >= mv_y_min) {             // check half top
                if (left2 >= mv_x_min) {        // check half left
                    ME_COST_IPEL_DIR_DIST(left2, top2, 1, (dist >> 1));
                }
                if (right2 <= mv_x_max) {       // check half right
                    ME_COST_IPEL_DIR_DIST(right2, top2, 3, (dist >> 1));
                }
            }
            if (left >= mv_x_min) {             // check left
                ME_COST_IPEL_DIR_DIST(left, omy, 4, dist);
            }
            if (right <= mv_x_max) {            // check right
                ME_COST_IPEL_DIR_DIST(right, omy, 5, dist);
            }
            if (bottom2 <= mv_y_max) {          // check half bottom
                if (left2 >= mv_x_min) {        // check half left
                    ME_COST_IPEL_DIR_DIST(left2, bottom2, 6, (dist >> 1));
                }
                if (right2 <= mv_x_max) {       // check half right
                    ME_COST_IPEL_DIR_DIST(right2, bottom2, 8, (dist >> 1));
                }
            }
            if (bottom <= mv_y_max) {           // check bottom
                ME_COST_IPEL_DIR_DIST(omx, bottom, 7, dist);
            }
        }
        if (mv->bcost < bcost) {
            rounds = 0;
        } else if (++rounds >= earlyExitIters) {
            return;
        }
    }

    for (dist = 16; dist <= merange; dist <<= 1) {
        omx    = mv->bmx;
        omy    = mv->bmx;
        bcost  = mv->bcost;
        top    = omy - dist;
        bottom = omy + dist;
        left   = omx - dist;
        right  = omx + dist;

        if (top >= mv_y_min && left >= mv_x_min && right <= mv_x_max && bottom <= mv_y_max) { // check border
            /* index:        0
             *               3
             *               2
             *               1
             *       0 3 2 1 * 1 2 3 0
             *               1
             *               2
             *               3
             *               0                  */
            ME_COST_IPEL_X4_DIR_DIST(omx,   top,    0, dist,
                                     left,  omy,    0, dist,
                                     right, omy,    0, dist,
                                     omx,   bottom, 0, dist);
            for (idx = 1; idx < 4; idx++) {
                posYT = top    + ((dist >> 2) * idx);
                posYB = bottom - ((dist >> 2) * idx);
                posXL = omx    - ((dist >> 2) * idx);
                posXR = omx    + ((dist >> 2) * idx);
                ME_COST_IPEL_X4_DIR_DIST(posXL, posYT, 0, dist,
                                         posXR, posYT, 0, dist,
                                         posXL, posYB, 0, dist,
                                         posXR, posYB, 0, dist);
            }
        } else {
            // check border for each mv
            if (top >= mv_y_min) {              // check top
                ME_COST_IPEL_DIR_DIST(omx, top, 0, dist);
            }
            if (left >= mv_x_min) {             // check left
                ME_COST_IPEL_DIR_DIST(left, omy, 0, dist);
            }
            if (right <= mv_x_max) {            // check right
                ME_COST_IPEL_DIR_DIST(right, omy, 0, dist);
            }
            if (bottom <= mv_y_max) {           // check bottom
                ME_COST_IPEL_DIR_DIST(omx, bottom, 0, dist);
            }

            for (idx = 1; idx < 4; idx++) {
                posYT = top    + ((dist >> 2) * idx);
                posYB = bottom - ((dist >> 2) * idx);
                posXL = omx    - ((dist >> 2) * idx);
                posXR = omx    + ((dist >> 2) * idx);

                if (posYT >= mv_y_min) {        // check top
                    if (posXL >= mv_x_min) {    // check left
                        ME_COST_IPEL_DIR_DIST(posXL, posYT, 0, dist);
                    }
                    if (posXR <= mv_x_max) {    // check right
                        ME_COST_IPEL_DIR_DIST(posXR, posYT, 0, dist);
                    }
                }
                if (posYB <= mv_y_max) {        // check bottom
                    if (posXL >= mv_x_min) {    // check left
                        ME_COST_IPEL_DIR_DIST(posXL, posYB, 0, dist);
                    }
                    if (posXR <= mv_x_max) {    // check right
                        ME_COST_IPEL_DIR_DIST(posXR, posYB, 0, dist);
                    }
                }
            }
        }

        if (mv->bcost < bcost) {
            rounds = 0;
        } else if (++rounds >= earlyExitIters) {
            return;
        }
    }
}

// int g_me_time[4] = { 0 };

/* ---------------------------------------------------------------------------
 * return minimum motion cost after search
 */
dist_t xavs2_me_search(xavs2_t *h, xavs2_me_t *p_me, int16_t(*mvc)[2], int i_mvc)
{
    /* special version of pack to allow shortcuts in CHECK_MV_RANGE */
    ALIGNED_ARRAY_16(int, costs,[8]);
    double beta2  = p_me->beta2 + 1;
    double beta3  = p_me->beta3 + 1;
    pel_t *p_org = p_me->p_fenc;
    pel_t *p_fref = p_me->p_fref_1st->planes[IMG_Y] + p_me->i_bias;
    int i_fref    = p_me->p_fref_1st->i_stride[IMG_Y];
    int i_pixel   = p_me->i_pixel;
    int mv_x_min  = p_me->mv_min_fpel[0];
    int mv_y_min  = p_me->mv_min_fpel[1];
    int mv_x_max  = p_me->mv_max_fpel[0];
    int mv_y_max  = p_me->mv_max_fpel[1];
    int me_range  = h->param->search_range;
    int lambda    = h->i_lambda_factor; // factor for determining Lagrangian's motion cost
    const uint32_t mv_min = pack16to32_mask2(-mv_x_min, -mv_y_min);
    const uint32_t mv_max = pack16to32_mask2(mv_x_max, mv_y_max) | 0x8000;
    const uint16_t *p_cost_mvx = h->mvbits - p_me->mvp.x;
    const uint16_t *p_cost_mvy = h->mvbits - p_me->mvp.y;
    uint32_t pmv;
    dist_t bcost = MAX_DISTORTION;
    int bmx = 0, bmy = 0;
    int omx, omy;
    int i, j, dir, idx;

    const int umh_1_3_step = h->UMH_big_hex_level == 2 ? 16 : 8;
    const int8_t(*search_patern)[2] = h->UMH_big_hex_level == 2 ? HEX4 : FAST_HEX4;

    // g_me_time[0]++;
    /* -------------------------------------------------------------
     * try MVP and some key searching points */
    pmv = MAKEDWORD(mvc[0][0], mvc[0][1]);   /* mvc[0][] is the MVP */

    for (i = 0; i < i_mvc; i++) {
        int mx = mvc[i][0];
        int my = mvc[i][1];
        ME_COST_IPEL(mx, my);
    }

    if (bcost == MAX_DISTORTION) {
        goto _me_error;         /* me failed */
    }

    /* -------------------------------------------------------------
     * search using different method */
    switch (h->param->me_method) {
    case XAVS2_ME_TZ: {       /* TZ */
        const int RasterDistance = 16;
        const int MaxIters = 32;
        const int EarlyExitIters = 3;
        dist_t bdist;
        int mv1_x, mv1_y, mv2_x, mv2_y;
        mv_info mvinfo;

        omx = bmx;
        omy = bmy;
        ME_COST_IPEL_X3(-2, 0, -1,  2,  1,  2);
        ME_COST_IPEL_X3( 2, 0,  1, -2, -1, -2);

        if (CHECK_MV_RANGE(bmx, bmy)) {
            DIA_ITER(bmx, bmy);
        }

        mvinfo.bcost = bcost;
        mvinfo.bdist = 0;
        mvinfo.bmx   = bmx;
        mvinfo.bmy   = bmy;
        mvinfo.bdir  = 0;
        tz_pattern_search(h, p_me, p_org, p_fref, &mvinfo, mv_x_min, mv_y_min, mv_x_max, mv_y_max, i_pixel, i_fref, EarlyExitIters, me_range);
        bcost = mvinfo.bcost;
        bdist = mvinfo.bdist;
        bmx   = mvinfo.bmx;
        bmy   = mvinfo.bmy;
        dir   = mvinfo.bdir;

        if (bdist == 1) {
            if (!dir) {
                break;
            }

            /* if best distance was only 1, check two missing points.
             * for a given direction 1 to 8, check nearest two outer X pixels*/
            mv1_x = bmx + offsets[(dir - 1) * 2    ][0];    /*     X   X     */
            mv1_y = bmy + offsets[(dir - 1) * 2    ][1];    /*   X 1 2 3 X   */
            mv2_x = bmx + offsets[(dir - 1) * 2 + 1][0];    /*     4 * 5     */
            mv2_y = bmy + offsets[(dir - 1) * 2 + 1][1];    /*   X 6 7 8 X   */
            if (CHECK_MV_RANGE(mv1_x, mv1_y)) {             /*     X   X     */
                ME_COST_IPEL(mv1_x, mv1_y);
            }
            if (CHECK_MV_RANGE(mv2_x, mv2_y)) {
                ME_COST_IPEL(mv2_x, mv2_y);
            }

            /* if no new point is found, stop */
            if (bcost == mvinfo.bcost) {
                break;      /* the bcost is not changed */
            }
        }

        /* raster search refinement if original search distance was too big */
        if (bdist > RasterDistance) {
            const int iRasterDist  = RasterDistance >> 1;
            const int iRasterDist2 = RasterDistance >> 2;
            int rmv_y_min = XAVS2_MAX(mv_y_min, bmy - RasterDistance + 2);
            int rmv_y_max = XAVS2_MIN(mv_y_max, bmy + RasterDistance - 2);
            int rmv_x_min = XAVS2_MAX(mv_x_min, bmx - RasterDistance + 2);
            int rmv_x_max = XAVS2_MIN(mv_x_max, bmx + RasterDistance - 2);
            for (j = rmv_y_min; j < rmv_y_max; j += iRasterDist) {
                for (i = rmv_x_min; i < rmv_x_max; i += iRasterDist) {
                    ME_COST_IPEL_X4(i, j, i, j + iRasterDist2, i + iRasterDist2, j, i + iRasterDist2, j + iRasterDist2);
                }
            }
        }

        while (bdist > 0) {
            // center a new search around current best
            mvinfo.bcost = bcost;
            mvinfo.bdist = 0;
            mvinfo.bmx   = bmx;
            mvinfo.bmy   = bmy;
            mvinfo.bdir  = 0;
            tz_pattern_search(h, p_me, p_org, p_fref, &mvinfo, mv_x_min, mv_y_min, mv_x_max, mv_y_max, i_pixel, i_fref, MaxIters, me_range);
            bcost = mvinfo.bcost;
            bdist = mvinfo.bdist;
            bmx   = mvinfo.bmx;
            bmy   = mvinfo.bmy;
            dir   = mvinfo.bdir;

            if (bdist == 1) {
                /* for a given direction 1 to 8, check nearest 2 outer X pixels */
                if (dir) {                                       /*    X   X    */
                    mv1_x = bmx + offsets[(dir - 1) * 2    ][0]; /*  X 1 2 3 X  */
                    mv1_y = bmy + offsets[(dir - 1) * 2    ][1]; /*    4 * 5    */
                    mv2_x = bmx + offsets[(dir - 1) * 2 + 1][0]; /*  X 6 7 8 X  */
                    mv2_y = bmy + offsets[(dir - 1) * 2 + 1][1]; /*    X   X    */
                    if (CHECK_MV_RANGE(mv1_x, mv1_y)) {
                        ME_COST_IPEL(mv1_x, mv1_y);
                    }
                    if (CHECK_MV_RANGE(mv2_x, mv2_y)) {
                        ME_COST_IPEL(mv2_x, mv2_y);
                    }
                }
                break;
            }
        }

        /* equivalent to the above, but eliminates duplicate candidates */
        goto umh_step_2;
    }
    case XAVS2_ME_UMH:        /* UMH */
        /* http://www.cnblogs.com/TaigaCon/archive/2014/06/16/3788984.html
         * 0. 初始点搜索 */
        DIA_ITER(mvc[0][0], mvc[0][1]);
        if (pmv && (bmx != mvc[0][0] || bmy != mvc[0][1])) {
            DIA_ITER(bmx, bmy);
            pmv = MAKEDWORD(bmx, bmy);
        }

        // select different step according to the different cost from upper layer
        if (p_me->mvp1.v != 0) {
            int mx = IPEL(p_me->mvp1.x);
            int my = IPEL(p_me->mvp1.y);
            ME_COST_IPEL(mx, my);
        }
        EARLY_TERMINATION(p_me->pred_sad_uplayer);
        // g_me_time[1]++;

        // prediction using mv of last ref_idx motion vector
        if (p_me->i_ref_idx > 0) {
            ME_COST_IPEL(IPEL(p_me->mvp2.x), IPEL(p_me->mvp2.y));
        }
        if (p_me->mvp3.v != 0) {
            ME_COST_IPEL(IPEL(p_me->mvp3.x), IPEL(p_me->mvp3.y));
        }

        /* 当前最优MV不是 MVP，搜索其周围一个小窗口 */
        if (pmv != (uint32_t)MAKEDWORD(bmx, bmy)) {
            DIA_ITER(bmx, bmy);
        }

        // early termination algorithm
        EARLY_TERMINATION(p_me->pred_sad);

        // umh_step_1:
        /* UMH 1. Unsymmetrical-cross search （非对称十字搜索） */
        // g_me_time[2]++;
        omx = bmx;
        omy = bmy;
        for (i = 1; i <= me_range; i += 2) {
            ME_COST_IPEL(omx + i, omy);
            ME_COST_IPEL(omx - i, omy);
        }
        for (j = 1; j <= me_range / 2; j += 2) {
            ME_COST_IPEL(omx, omy + j);
            ME_COST_IPEL(omx, omy - j);
        }

        // early termination algorithm
        EARLY_TERMINATION(p_me->pred_sad);

        /* UMH 2. Spiral search （螺旋搜索） */
        omx = bmx;
        omy = bmy;
        for (i = 0; i < 24; i++) {
            ME_COST_IPEL(omx + GRID[i][0], omy + GRID[i][1]);
        }

        // early termination algorithm
        EARLY_TERMINATION(p_me->pred_sad);

        // big hexagon
        if (h->UMH_big_hex_level) {
            for (j = 1; j <= me_range / 4; j++) {
                omx = bmx;
                omy = bmy;
                for (i = 0; i < umh_1_3_step; i++) {
                    ME_COST_IPEL(omx + search_patern[i][0] * j, omy + search_patern[i][1] * j);
                }
                if (bmx != omx || bmy != omy) {
                    EARLY_TERMINATION(p_me->pred_sad);
                }
            }
        }
        /* !!! NO break statement here */
    case XAVS2_ME_HEX:        /* hexagon search */
umh_step_2 :                  /* UMH 3. Uneven Multi-Hexagon-grid Search （不规律六边形模板搜索） */
        // g_me_time[3]++;
        dir = 0;                                        /*   6   5   */
        omx = bmx;                                      /*           */
        omy = bmy;                                      /* 1   *   4 */
        ME_COST_IPEL_X3_DIR(-1,-2,6,  1,-2,5, -2,0,1);  /*           */
        ME_COST_IPEL_X3_DIR( 2, 0,4, -1, 2,2,  1,2,3);  /*   2   3   */

        if (dir) {
            const int8_t (*hex)[2];
            /* UMH 4. Extended Hexagon-based Search （六边形模板反复搜索） */
            idx = dir - 1;      /* start array index */
            /* half hexagon, not overlapping the previous iteration */
            for (i = 0; i < me_range && CHECK_MV_RANGE(bmx, bmy); i++) {
                dir = 0;
                omx = bmx;
                omy = bmy;
                hex = &HEX2[idx];
                ME_COST_IPEL_X3_DIR(hex[0][0],hex[0][1],1, hex[1][0],hex[1][1],2, hex[2][0],hex[2][1],3);
                if (!dir) {
                    break;      /* early terminate */
                }
                idx = M1MOD6[dir + idx - 1];    /* next start array index */
            }
        }
        /* !!! NO break statement here */
    case XAVS2_ME_DIA:        /* diamond search */
umh_step_3:                   /* UMH 5. the third step with a small search pattern （小菱形模板反复搜索） */
        dir = 0;
        if (CHECK_MV_RANGE(bmx, bmy)) {
            omx = bmx;                                          /*    4    */
            omy = bmy;                                          /*  1 * 3  */
            ME_COST_IPEL_X4_DIR(0,-1,4, -1,0,1, 1,0,3, 0,1,2);  /*    2    */
        }
        if (dir) {
            const int8_t (*dia)[2];
            idx = dir - 1;      /* start array index */
            /* half diamond, not overlapping the previous iteration */
            for (i = 0; i < me_range && CHECK_MV_RANGE(bmx, bmy); i++) {
                dir = 0;
                omx = bmx;
                omy = bmy;
                dia = &DIA1[idx];
                ME_COST_IPEL_X3_DIR(dia[0][0],dia[0][1],1, dia[1][0],dia[1][1],2, dia[2][0],dia[2][1],3);
                if (!dir) {
                    break;      /* early terminate */
                }
                idx = M1MOD4[dir + idx - 1];    /* next start array index */
            }
        }
        break;
    default:                    /* XAVS2_ME_FS: full search */
        omx = bmx;
        omy = bmy;
        for (j = -me_range; j < me_range; j++) {
            for (i = -me_range; i < me_range; i++) {
                ME_COST_IPEL(omx + i, omy + j);
            }
        }
        break;
    }

    /* -------------------------------------------------------------
     * store the results of fullpel search */
    p_me->bmv.v  = MAKEDWORD(FPEL(bmx), FPEL(bmy));
    p_me->bmv2.v = MAKEDWORD(bmx, bmy);
    p_me->bcost  = bcost;
    p_me->bcost2 = bcost;
    p_me->mvcost[PDIR_FWD] = MV_COST_IPEL(bmx, bmy);

    /* -------------------------------------------------------------
     * sub-pel refine */
    if (h->use_fractional_me) {
        bcost = me_subpel_refine(h, p_me);
    }

_me_error:
    return bcost;
}


/* ---------------------------------------------------------------------------
 * find motion vector for forward dual hypothesis prediction (sub-pel search)
 * return minimum motion cost after search
 */
dist_t xavs2_me_search_sym(xavs2_t *h, xavs2_me_t *p_me, pel_t *buf_pixel_temp, mv_t *mv)
{
    const int search_pos2 = 5;  // search positions for    half-pel search  (default: 9)
    const int search_pos4 = 5;  // search positions for quarter-pel search  (default: 9)
    pel_t **p_filtered1 = p_me->p_fref_1st->filtered;
    pel_t **p_filtered2 = p_me->p_fref_2nd->filtered;
    pel_t *p_org = p_me->p_fenc;
    int distance_fwd = p_me->i_distance_1st;
    int distance_bwd = p_me->i_distance_2nd;
    int i_pixel  = p_me->i_pixel;
    int i_offset = p_me->i_bias;
    int ctr_x    = (p_me->mvp1.x >> 1) << 1;
    int ctr_y    = (p_me->mvp1.y >> 1) << 1;
    int mv_x_min = p_me->mv_min[0];
    int mv_y_min = p_me->mv_min[1];
    int mv_x_max = p_me->mv_max[0];
    int mv_y_max = p_me->mv_max[1];
    int lambda   = h->i_lambda_factor;
    int min_pos2 = (h->param->enable_hadamard ? 0 : 1);
    int max_pos2 = (h->param->enable_hadamard ? XAVS2_MAX(1, search_pos2) : search_pos2);
    const uint32_t mv_min = pack16to32_mask2(-mv_x_min, -mv_y_min);
    const uint32_t mv_max = pack16to32_mask2(mv_x_max, mv_y_max) | 0x8000;
    const uint16_t *p_cost_mvx = h->mvbits - p_me->mvp.x;
    const uint16_t *p_cost_mvy = h->mvbits - p_me->mvp.y;
    mv_t bmv = *mv;  // best mv
    dist_t bcost = MAX_DISTORTION;
    dist_t cost;
    int pos;
    int mx, my;
    int i_fref = p_me->p_fref_1st->i_stride[IMG_Y];

    if (!h->use_fractional_me) {
        mx = mv->x;
        my = mv->y;

        ME_COST_QPEL_SYM;
        bcost = cost;
        bmv.v = MAKEDWORD(mx, my);
        return bcost;
    }

    // loop over search positions
    for (pos = min_pos2; pos < max_pos2; pos++) {
        mx = mv->x + (Spiral[pos][0] << 1);    // quarter-pel units
        my = mv->y + (Spiral[pos][1] << 1);    // quarter-pel units

        ME_COST_QPEL_SYM;
        if (cost < bcost) {
            bcost = cost;
            bmv.v = MAKEDWORD(mx, my);
        }
    }

    mv->v = bmv.v;

    /* -------------------------------------------------------------
     * quarter-pel refine */

    // loop over search positions
    if (h->use_fractional_me >= 2) {
        for (pos = 1; pos < search_pos4; pos++) {
            if (h->param->enable_pmvr) {
                if (pmvr_adapt_mv(&mx, &my, ctr_x, ctr_y, mv->x, mv->y, Spiral[pos][0], Spiral[pos][1])) {
                    continue;
                }
            } else {
                mx = mv->x + Spiral[pos][0];    // quarter-pel units
                my = mv->y + Spiral[pos][1];    // quarter-pel units
            }

            ME_COST_QPEL_SYM;
            if (cost < bcost) {
                bcost = cost;
                bmv.v = MAKEDWORD(mx, my);
            }
        }
    }

    mv->v = bmv.v;
    p_me->mvcost[PDIR_SYM] = MV_COST_FPEL(bmv.x, bmv.y);

    // return minimum motion cost
    return bcost;
}

/* ---------------------------------------------------------------------------
 * return minimum motion cost after search (sub-pel search)
 */
dist_t xavs2_me_search_bid(xavs2_t *h, xavs2_me_t *p_me, pel_t *buf_pixel_temp, mv_t *fwd_mv, mv_t *bwd_mv, cu_parallel_t *p_enc)
{
    pel_t **p_filtered1 = p_me->p_fref_1st->filtered;
    pel_t **p_filtered2 = p_me->p_fref_2nd->filtered;
    pel_t *p_org = p_me->p_fenc;
    const int search_pos2 = 9;  // search positions for    half-pel search  (default: 9)
    const int search_pos4 = 9;  // search positions for quarter-pel search  (default: 9)
    int i_pixel  = p_me->i_pixel;
    int i_offset = p_me->i_bias;
    int ctr_x    = (p_me->mvp1.x >> 1) << 1;
    int ctr_y    = (p_me->mvp1.y >> 1) << 1;
    int mv_x_min = p_me->mv_min[0];
    int mv_y_min = p_me->mv_min[1];
    int mv_x_max = p_me->mv_max[0];
    int mv_y_max = p_me->mv_max[1];
    int lambda   = h->i_lambda_factor;
    int min_pos2 = (h->param->enable_hadamard ? 0 : 1);
    int max_pos2 = (h->param->enable_hadamard ? XAVS2_MAX(1, search_pos2) : search_pos2);
    int block_w = p_me->i_block_w;
    int xx2;
    int yy2;
    int mv_bid_bit;
    const uint32_t mv_min = pack16to32_mask2(-mv_x_min, -mv_y_min);
    const uint32_t mv_max = pack16to32_mask2(mv_x_max, mv_y_max) | 0x8000;
    const uint16_t *p_cost_mvx = h->mvbits - p_me->mvp1.x;
    const uint16_t *p_cost_mvy = h->mvbits - p_me->mvp1.y;
    const uint16_t *p_cost_bix = h->mvbits - p_me->mvp2.x;
    const uint16_t *p_cost_biy = h->mvbits - p_me->mvp2.y;
    mv_t bmv = *fwd_mv; // best mv
    dist_t bcost = MAX_DISTORTION;
    dist_t cost;
    int mx, my, mx_bid, my_bid;
    int pos;
    int i_fref = p_me->p_fref_1st->i_stride[IMG_Y];
    coeff_t *cur_blk = p_enc->coeff_blk;

    mx_bid = bwd_mv->x;
    my_bid = bwd_mv->y;

    //在这里把编码值与预测值的计算公式换算为2倍编码值-后向预测值
    xx2 = mx_bid >> 2;
    yy2 = my_bid >> 2;
    mv_bid_bit = MV_COST_FPEL_BID(mx_bid, my_bid);

    if (CHECK_MV_RANGE(mx_bid, my_bid)) {
        pel_t *p_src2 = p_filtered2[((my_bid & 3) << 2) + (mx_bid & 3)];

        if (p_src2 != NULL) {
            p_src2 += i_offset + yy2 * i_fref + xx2;
            g_funcs.pixf.sub_ps[i_pixel](cur_blk, block_w, p_org, p_src2, FENC_STRIDE, i_fref);//M-A
        } else {
            ALIGN32(pel_t tmp_pred[MAX_CU_SIZE * MAX_CU_SIZE]);
            mv_t mvt;
            mvt.x = (int16_t)mx_bid;
            mvt.y = (int16_t)my_bid;
            get_mv_for_mc(h, &mvt, p_me->i_pix_x, p_me->i_pix_y, block_w, p_me->i_block_h);
            mc_luma(tmp_pred, MAX_CU_SIZE, mvt.x, mvt.y, block_w, p_me->i_block_h, p_me->p_fref_2nd);
            g_funcs.pixf.sub_ps[i_pixel](cur_blk, block_w, p_org, tmp_pred, FENC_STRIDE, MAX_CU_SIZE);//M-A
        }
        g_funcs.pixf.add_ps[i_pixel](buf_pixel_temp, MAX_CU_SIZE, p_org, cur_blk, FENC_STRIDE, block_w);//M-A+M
    }

    if (!h->use_fractional_me) {
        mx = fwd_mv->x;
        my = fwd_mv->y;

        ME_COST_QPEL_BID;
        bcost = cost;
        bmv.v = MAKEDWORD(mx, my);
        return bcost;
    }

    // loop over search positions
    for (pos = min_pos2; pos < max_pos2; pos++) {
        mx = fwd_mv->x + (Spiral[pos][0] << 1);    // quarter-pel units
        my = fwd_mv->y + (Spiral[pos][1] << 1);    // quarter-pel units

        ME_COST_QPEL_BID;
        if (cost < bcost) {
            bcost = cost;
            bmv.v = MAKEDWORD(mx, my);
        }
    }

    fwd_mv->v = bmv.v;

    /* -------------------------------------------------------------
     * quarter-pel refine */

    // loop over search positions
    if (h->use_fractional_me >= 2) {
        for (pos = 1; pos < search_pos4; pos++) {
            if (h->param->enable_pmvr) {
                if (pmvr_adapt_mv(&mx, &my, ctr_x, ctr_y, fwd_mv->x, fwd_mv->y, Spiral[pos][0], Spiral[pos][1])) {
                    continue;
                }
            } else {
                mx = fwd_mv->x + Spiral[pos][0];    // quarter-pel units
                my = fwd_mv->y + Spiral[pos][1];    // quarter-pel units
            }

            ME_COST_QPEL_BID;
            if (cost < bcost) {
                bcost = cost;
                bmv.v = MAKEDWORD(mx, my);
            }
        }
    }

    fwd_mv->v = bmv.v;
    p_me->mvcost[PDIR_BID] = MV_COST_FPEL(bmv.x, bmv.y) + MV_COST_FPEL_BID(mx_bid, my_bid);

    // return minimum motion cost
    return bcost;
}
