/*
 * presets.c
 *
 * Description of this file:
 *    parse preset level functions definition of the xavs2 library
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

#include "common/common.h"
#include "encoder/aec.h"
#include "presets.h"

/**
 * ===========================================================================
 * macros
 * ===========================================================================
 */
/* macros for enable/disable algorithms */
#define SWITCH_OFF(m)   enable_algs &= (~(1LL << (m)))
#define SWITCH_ON(m)    enable_algs |=   (1LL << (m))

/**
 * ===========================================================================
 * local tables
 * ===========================================================================
 */
/* ---------------------------------------------------------------------------
 * 帧内亮度块的RDO模式数量，对应不同preset档次
 */
static const uint8_t INTRA_FULL_RDO_NUM[][MAX_CU_SIZE_IN_BIT + 1] = {
    { 0, 0, 1, 1, 1, 1, 1 },         /* 0:  1x1, 2x2, 4x4, 8x8, 16x16, 32x32, 64x64 */
    { 0, 0, 2, 2, 2, 2, 1 },         /* 1:  1x1, 2x2, 4x4, 8x8, 16x16, 32x32, 64x64 */
    { 0, 0, 3, 3, 3, 3, 2 },         /* 2:  1x1, 2x2, 4x4, 8x8, 16x16, 32x32, 64x64 */
    { 0, 0, 4, 4, 3, 3, 2 },         /* 3:  1x1, 2x2, 4x4, 8x8, 16x16, 32x32, 64x64 */
    { 0, 0, 5, 5, 5, 4, 3 },         /* 4:  1x1, 2x2, 4x4, 8x8, 16x16, 32x32, 64x64 */
    { 0, 0, 6, 6, 6, 4, 3 },         /* 5:  1x1, 2x2, 4x4, 8x8, 16x16, 32x32, 64x64 */
    { 0, 0, 7, 7, 7, 6, 5 },         /* 6:  1x1, 2x2, 4x4, 8x8, 16x16, 32x32, 64x64 */
    { 0, 0, 8, 8, 8, 6, 5 },         /* 7:  1x1, 2x2, 4x4, 8x8, 16x16, 32x32, 64x64 */
    { 0, 0, 9, 9, 9, 9, 9 },         /* 8:  1x1, 2x2, 4x4, 8x8, 16x16, 32x32, 64x64 */
    { 0, 0, 9, 9, 9, 9, 9 },         /* 9:  1x1, 2x2, 4x4, 8x8, 16x16, 32x32, 64x64 */
};

/* ---------------------------------------------------------------------------
 * 帧内色度块 RDO 的最大模式数量 (不同preset档次)
 */
static const int8_t tab_num_rdo_chroma_intra_mode[] = {
    1, 2, 2, 2, 3, 3, 4, 4, 5, 5
};

/* 帧内RMD搜索的阈值，步长为2和1搜索的角度数量 */
static const int8_t tab_num_angle_dist2[] = {
    0, 0, 4, 4, 4, 4, 5, 5, 6, 6
};
static const int8_t tab_num_angle_dist1[] = {
    0, 0, 0, 0, 2, 2, 3, 3, 4, 4
};

/* ---------------------------------------------------------------------------
 * 全零块检测时的判定阈值倍率
 */
static const float tab_th_zero_block_factor[] = {
    6, 6, 6, 6, 6, 6, 5, 5, 5, 5
};

/* ---------------------------------------------------------------------------
 * QSFD算法的阈值计算系数（不同preset）
 */
const static double tab_qsfd_s_presets[][10] = {
    /* preset_level:
     * 0    1    2    3    4    5    6    7    8    9 */
    { 2.3, 1.8, 1.6, 1.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},   /* inter */
    { 0.9, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.3, 0.2, 0.2},   /* intra */
};
const static double tab_qsfd_cu_size_weight[4] = {
    0.25, 1.0, 3.0, 7.5  /* 8x8, 16x16, 32x32, 64x64 */
};

double tab_qsfd_thres[MAX_QP][2][CTU_DEPTH];

/*--------------------------------------------------------------------------
 */
static INLINE
void algorithm_init_thresholds(xavs2_param_t *p_param)
{
    int i_preset_level = p_param->preset_level;
    //trade-off encoding time and performance
    const double s_inter = tab_qsfd_s_presets[0][i_preset_level];
    const double s_intra = tab_qsfd_s_presets[1][i_preset_level];
    int i;

    /* QSFD threasholds */
    for (i = 0; i < MAX_QP; i++) {
        double qstep = 32768.0 / tab_Q_TAB[i];
        double th_base = 350 * pow(qstep, 0.9);
        double th__8 = th_base * tab_qsfd_cu_size_weight[0];
        double th_16 = th_base * tab_qsfd_cu_size_weight[1];
        double th_32 = th_base * tab_qsfd_cu_size_weight[2];
        double th_64 = th_base * tab_qsfd_cu_size_weight[3];

        /* inter frame */
        tab_qsfd_thres[i][0][0] = th__8 * s_inter;
        tab_qsfd_thres[i][0][1] = th_16 * s_inter;
        tab_qsfd_thres[i][0][2] = th_32 * s_inter;
        tab_qsfd_thres[i][0][3] = th_64 * s_inter;
        if (i_preset_level < 2) {
            tab_qsfd_thres[i][0][1] *= 2.0;
        }
        /* intra frame */
        tab_qsfd_thres[i][1][0] = th__8;
        tab_qsfd_thres[i][1][1] = th_16 * s_intra * 1.4;
        tab_qsfd_thres[i][1][2] = th_32 * s_intra * 1.2;
        tab_qsfd_thres[i][1][3] = th_64 * s_intra * 1.0;
    }

    /* 全零块检测 */
    p_param->factor_zero_block = tab_th_zero_block_factor[i_preset_level];
}

/* ---------------------------------------------------------------------------
 * Function   : modify configurations according to different preset levels.
 * Parameters :
 *   [in/out] : p_param        - the coding parameter to be set
 *      [in ] : i_preset_level - the preset level
 * Return     : none
 * ---------------------------------------------------------------------------
 */
void parse_preset_level(xavs2_param_t *p_param, int i_preset_level)
{
    /* special settings */
    if (i_preset_level < 2) {
        /* only for level: 0,1 */
        p_param->search_range = XAVS2_MIN(p_param->search_range, 57);
        p_param->num_max_ref = 2;
    } else {
        /* only for level: 2,3,4,5,6,7,8,9 */
        p_param->num_max_ref = XAVS2_MIN(i_preset_level, 4);
    }

    /* --------------------------- CU结构 ---------------------------
    | preset          |  0  |  1  |  2  |   3 |   4 |   5 |   6  |   7  |   8  |  9   |
    +=================+=====+=====+=====+=====+=====+=====+======+======+======+======+
    | ctu             | 32  | 32  | 64  |  64 |  64 |  64 |  64  |  64  |  64  | 64   |
    | min-cu-size     |  8  |  8  |  8  |   8 |   8 |   8 |   8  |   8  |   8  |  8   |
    */
    p_param->lcu_bit_level = XAVS2_MIN(p_param->lcu_bit_level, 5 + (i_preset_level > 1));

    /* --------------------------- 预测 ---------------------------
    */
    p_param->inter_2pu       = i_preset_level > 1;
    p_param->enable_amp      = i_preset_level > 5;  // NSQT
    p_param->enable_intra    = i_preset_level > 0;
    p_param->enable_f_frame  = i_preset_level > -1;
    p_param->enable_mhp_skip = i_preset_level > -1 && p_param->enable_f_frame;
    p_param->enable_wsm      = i_preset_level > 7 && p_param->enable_f_frame;
    p_param->enable_dhp      = i_preset_level > 7 && p_param->enable_f_frame;
    p_param->enable_dmh      = i_preset_level > 6 && p_param->enable_f_frame;

    /* --------------------------- 变换 --------------------------- */
    p_param->enable_sdip       = i_preset_level > 5;
    p_param->enable_nsqt       = i_preset_level > 5;
    p_param->enable_secT       = i_preset_level > -1;
    p_param->b_fast_2lelvel_tu = i_preset_level < 4;

    /* --------------------------- 量化 ---------------------------
     * Level: All for preset 9, Off for preset 0~2 */
    p_param->i_rdoq_level = i_preset_level > 8 ? RDOQ_ALL : i_preset_level > 5 ? RDOQ_CU_LEVEL : RDOQ_OFF;

    /* --------------------------- RDO档次 ---------------------------
    */
    if (i_preset_level < 0) {
        p_param->i_rd_level = RDO_OFF;
    } else if (i_preset_level < 1) {
        p_param->i_rd_level = RDO_CU_LEVEL1;
    } else if (i_preset_level < 5) {
        p_param->i_rd_level = RDO_CU_LEVEL2;
    } else {
        p_param->i_rd_level = RDO_ALL;
    }

    /* --------------------------- 熵编码 ---------------------------
     */
    if (i_preset_level <= 3) {
        p_param->rdo_bit_est_method = 2;
    } else if (i_preset_level < 5) {
        p_param->rdo_bit_est_method = 1;
    } else {
        p_param->rdo_bit_est_method = 0;
    }

    /* --------------------------- 滤波 ---------------------------
    */
    p_param->enable_alf = p_param->enable_alf && i_preset_level > 4;
    p_param->enable_sao = p_param->enable_sao && i_preset_level > 1;
    p_param->b_fast_sao = i_preset_level < 5;  // 档次4以下开启快速SAO编码决策

    /* --------------------------- 其他 ---------------------------
    */
    p_param->enable_hadamard = i_preset_level > 0;
    p_param->enable_tdrdo    = i_preset_level > 4 && p_param->enable_tdrdo;

    /* tell the encoder preset configuration is utilized */
    p_param->is_preset_configured = TRUE;
}

/* ---------------------------------------------------------------------------
 * reconfigure encoder after one frame has been encoded
 */
void xavs2_reconfigure_encoder(xavs2_t *h)
{
    UNUSED_PARAMETER(h);
}

/* ---------------------------------------------------------------------------
 * fast algorithms for different presets
 */
static INLINE
uint64_t get_fast_algorithms(xavs2_t *h, int i_preset_level)
{
    uint64_t enable_algs = 0;  // disable all algorithms

    UNUSED_PARAMETER(h);

    switch (i_preset_level) {
    case 0:     // ultra fast
        SWITCH_ON(OPT_ET_INTRA_DEPTH);
        SWITCH_ON(OPT_SKIP_DMH_THRES);
        SWITCH_ON(OPT_EARLY_SKIP);
        SWITCH_ON(OPT_BYPASS_MODE_FPIC);
        SWITCH_ON(OPT_BYPASS_SDIP);
        SWITCH_ON(OPT_BYPASS_INTRA_BPIC);
    case 1:     // super fast
        SWITCH_ON(OPT_ECU);
    case 2:     // very fast
        SWITCH_ON(OPT_FAST_ZBLOCK);
        SWITCH_ON(OPT_FAST_RDO_INTRA_C);
    case 3:     // faster
        SWITCH_ON(OPT_FAST_CBF_MODE);
        SWITCH_ON(OPT_ET_RDO_INTRA_L);
        SWITCH_ON(OPT_BYPASS_INTRA_RDOQ);
        SWITCH_ON(OPT_RDOQ_AZPC);
        SWITCH_ON(OPT_PU_RMS);
    case 4:     // fast
        SWITCH_ON(OPT_CU_DEPTH_CTRL);
        SWITCH_ON(OPT_SUBCU_SPLIT);
        SWITCH_ON(OPT_FAST_PU_SEL);
        SWITCH_ON(OPT_CMS_ETMD);
    case 5:
        SWITCH_ON(OPT_ROUGH_SKIP_SEL);
        SWITCH_ON(OPT_BIT_EST_PSZT);
        SWITCH_ON(OPT_FAST_ALF);
        SWITCH_ON(OPT_FAST_SAO);
        SWITCH_ON(OPT_CBP_DIRECT);
        SWITCH_ON(OPT_FAST_INTRA_IN_INTER);
    case 6:     // slow
        SWITCH_ON(OPT_BYPASS_AMP);
        SWITCH_ON(OPT_CODE_OPTIMZATION);
    case 7:     // slower
        SWITCH_ON(OPT_CU_QSFD);
        SWITCH_ON(OPT_TU_LEVEL_DEC);
        SWITCH_ON(OPT_TR_KEY_FRAME_MD);
    case 8:     // very slow
        // fast inter
        SWITCH_ON(OPT_DMH_CANDIDATE);
        SWITCH_ON(OPT_ADVANCE_CHROMA_AEC);
        SWITCH_ON(OPT_ROUGH_MODE_SKIP);
        SWITCH_ON(OPT_PSC_MD);
        // fast intra
        SWITCH_ON(OPT_FAST_INTRA_MODE);
        break;
    case 9:     // placebo
        enable_algs = 0;           /* switch off all fast algorithms */
        break;
    default:
        assert(0);
        break;
    }

    return enable_algs;
}

/**
 * ---------------------------------------------------------------------------
 * Function   : set fast algorithms enabled according to different preset levels
 * Parameters :
 *      [in ] : h - pointer to struct xavs2_t, the xavs2 encoder
 * Return     : none
 * ---------------------------------------------------------------------------
 */
void encoder_set_fast_algorithms(xavs2_t *h)
{
    const int num_algorithm = NUM_FAST_ALGS;
    int i_preset_level = h->param->preset_level;
    uint64_t enable_algs = 0;  // disable all algorithms

    if (num_algorithm > 64) {
        xavs2_log(h, XAVS2_LOG_ERROR, "Algorithms error: too many flags: %d\n", num_algorithm);
        exit(0);
    }

    /* -------------------------------------------------------------
     * 1, switch on some algorithms with little efficiency loss
     */

    /* 是否需要分像素运动搜索
     * 参考帧数量大于1个时，会出现MV的缩放而导致MV像素精度达到1/4
     */
    if (i_preset_level < 2) {
        h->use_fractional_me = 1;
    } else {
        h->use_fractional_me = 2;
    }
    h->use_fast_sub_me = (i_preset_level < 5);
    h->UMH_big_hex_level = (i_preset_level < 5) ? 0 : (i_preset_level < 9) ? 1 : 2;
    h->skip_rough_improved = (i_preset_level > 3);
    /* -------------------------------------------------------------
     * 2, switch off part of fast algorithms according to different preset levels
     */
    enable_algs = get_fast_algorithms(h, i_preset_level);

    SWITCH_OFF(OPT_ROUGH_PU_SEL);

    /* apply the settings */
    h->i_fast_algs = enable_algs;

    if (IS_ALG_ENABLE(OPT_ET_RDO_INTRA_L)) {
        memcpy(h->tab_num_intra_rdo, INTRA_FULL_RDO_NUM[i_preset_level >> 1], sizeof(h->tab_num_intra_rdo));
    } else {
        memcpy(h->tab_num_intra_rdo, INTRA_FULL_RDO_NUM[i_preset_level >> 0], sizeof(h->tab_num_intra_rdo));
    }
    /* RMD算法的搜索角度数量 */
    h->num_intra_rmd_dist2  = tab_num_angle_dist2[i_preset_level];
    h->num_intra_rmd_dist1  = tab_num_angle_dist1[i_preset_level];
    h->num_rdo_intra_chroma = tab_num_rdo_chroma_intra_mode[i_preset_level];

    /* 帧内预测模式 */
    if (IS_ALG_ENABLE(OPT_FAST_INTRA_MODE)) {
        h->get_intra_candidates_luma = rdo_get_pred_intra_luma_rmd;
    } else {
        h->get_intra_candidates_luma = rdo_get_pred_intra_luma;
    }
    if (IS_ALG_ENABLE(OPT_FAST_RDO_INTRA_C)) {
        h->get_intra_candidates_chroma = rdo_get_pred_intra_chroma_fast;
    } else {
        h->get_intra_candidates_chroma = rdo_get_pred_intra_chroma;
    }

    /* AEC */
    switch (h->param->rdo_bit_est_method) {
    case 1:
    case 2:
        h->size_aec_rdo_copy = sizeof(aec_t) - sizeof(ctx_set_t);
        h->copy_aec_state_rdo = aec_copy_aec_state_rdo;
        break;
    default:
        h->size_aec_rdo_copy = sizeof(aec_t);
        h->copy_aec_state_rdo = aec_copy_aec_state;
        break;
    }
}

/**
* ---------------------------------------------------------------------------
* Function   : decide the ultimate parameters used by encoders
* Parameters :
*      [in ] : p_param - the ultimate coding parameter to be set
* Return     : none
* ---------------------------------------------------------------------------
*/
void decide_ultimate_paramters(xavs2_param_t *p_param)
{
    algorithm_init_thresholds(p_param);

    if (p_param->preset_level < 4) {
        p_param->me_method = XAVS2_ME_HEX;
    }
}

#undef SWITCH_OFF
#undef SWITCH_ON
