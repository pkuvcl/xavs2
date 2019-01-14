/*
 * aec.h
 *
 * Description of this file:
 *    AEC functions definition of the xavs2 library
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

#ifndef XAVS2_AEC_H
#define XAVS2_AEC_H

#include "common.h"

/*
 * ===========================================================================
 * global variables
 * ===========================================================================
 */
extern binary_t gf_aec_default;
extern binary_t gf_aec_rdo;
extern binary_t gf_aec_fastrdo;
extern binary_t gf_aec_vrdo;

#define tab_intra_mode_scan_type FPFX(tab_intra_mode_scan_type)
extern const int tab_intra_mode_scan_type[NUM_INTRA_MODE];

#if CTRL_OPT_AEC
extern context_t g_tab_ctx_mps[4096 * 5];    /* [2 * lg_pmps + mps + cycno * 4096] */
extern context_t g_tab_ctx_lps[4096 * 5];    /* [2 * lg_pmps + mps + cycno * 4096] */
#endif

/* ---------------------------------------------------------------------------
 * number of maximum flush bits in p_aec->reg_flush_bits
 */
static const uint32_t NUM_FLUSH_BITS = 24;

/* ---------------------------------------------------------------------------
 * AC ENGINE PARAMETERS
 */
static const uint32_t tab_cwr[4] = {
    197, 197, 95, 46
};

/* ---------------------------------------------------------------------------
 * cu type mapping
 */
static const int MAP_CU_TYPE[MAX_PRED_MODES] = {
    1, 2, 3, 4, 3, 3, 4, 4, 6, 6, 6, 6
};

/* ---------------------------------------------------------------------------
 * coefficient offset [stride][line]
 */
static const int tab_coeff_offset[9][3] = {
    { 1, 2, 3 },
    { 2, 4, 6 },
    { 4, 8, 12 },
    { 8, 16, 24 },
    { 16, 32, 48 },
    { 32, 64, 96 },
    { 64, 128, 192 },
    { 128, 256, 384 },
    { 256, 512, 768 },
};

/* ---------------------------------------------------------------------------
 * macros
 */
#define NUM_OF_COEFFS_IN_CG     16

#define CHECK_EARLY_RETURN_RUNLEVEL(aec) \
    if ((cur_bits = rdo_get_written_bits(aec) - org_bits) > maxvalue) {\
        return cur_bits;\
    }

#define MAKE_CONTEXT(lg_pmps, mps, cycno)  (((uint16_t)(cycno) << 12) | ((uint16_t)(mps) << 0) | (uint16_t)(lg_pmps << 1))

/* ---------------------------------------------------------------------------
 * AC ENGINE PARAMETERS
 */
#define B_BITS              10
#define QUARTER             (1 << (B_BITS-2))
#define LG_PMPS_SHIFTNO     2

/*
 * ===========================================================================
 * inline function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * copy coding state
 */
static ALWAYS_INLINE
void aec_copy_aec_state(aec_t *dst, aec_t *src)
{
    memcpy(dst, src, sizeof(aec_t));
    dst->p_ctx_set = &dst->ctx_set;
}


/* ---------------------------------------------------------------------------
 * copy coding state
 */
static ALWAYS_INLINE
void aec_copy_aec_state_rdo(aec_t *dst, aec_t *src)
{
    memcpy(dst, src, sizeof(aec_t) - sizeof(ctx_set_t));
    (dst)->p_ctx_set = (src)->p_ctx_set;
}

/* ---------------------------------------------------------------------------
 * 仅拷贝SAO相关的上下文状态和码流状态
 */
static ALWAYS_INLINE
void aec_copy_coding_state_sao(aec_t *p_dst, aec_t *p_src)
{
    int num_bytes_aec = (int)((uint8_t *)&p_dst->ctx_set - (uint8_t *)p_dst);
    int num_bytes_context = (int)(sizeof(ctx_set_t) - ((uint8_t *)&p_dst->ctx_set.sao_merge_type_index[0] - (uint8_t *)&p_dst->ctx_set));
    memcpy(p_dst, p_src, num_bytes_aec);
    p_dst->p_ctx_set = &p_dst->ctx_set;
    memcpy(&p_dst->ctx_set.sao_merge_type_index[0], &p_src->ctx_set.sao_merge_type_index[0], num_bytes_context);
}


/* ---------------------------------------------------------------------------
 * returns the number of currently written bits
 */
static ALWAYS_INLINE
int aec_get_written_bits(aec_t *p_aec)
{
    return (int)(((p_aec->p - p_aec->p_start) << 3) + p_aec->i_bits_to_follow + NUM_FLUSH_BITS - p_aec->num_left_flush_bits);
}

/* ---------------------------------------------------------------------------
 * returns the number of currently written bits
 */
static ALWAYS_INLINE
int rdo_get_written_bits(aec_t *p_aec)
{
    return (int)p_aec->i_bits_to_follow;
}


/* ---------------------------------------------------------------------------
 * 向码流文件中输出flush bits
 */
static INLINE
void bitstr_flush_bits(aec_t *p_aec)
{
    switch (NUM_FLUSH_BITS) {
    case 24:
        p_aec->p[0] = (uint8_t)(p_aec->reg_flush_bits >> 16);
        p_aec->p[1] = (uint8_t)(p_aec->reg_flush_bits >> 8);
        p_aec->p[2] = (uint8_t)(p_aec->reg_flush_bits);
        p_aec->p += 3;
        break;
    case 16:
        p_aec->p[0] = (uint8_t)(p_aec->reg_flush_bits >> 8);
        p_aec->p[1] = (uint8_t)(p_aec->reg_flush_bits);
        p_aec->p += 2;
        break;
    case 8:
        p_aec->p[0] = (uint8_t)p_aec->reg_flush_bits;
        p_aec->p += 1;
        break;
    default:
        fprintf(stderr, "Unsupported number of flush bits %d\n", NUM_FLUSH_BITS);
        assert(0);
        break;
    }

    p_aec->reg_flush_bits = 0;
}


/* ---------------------------------------------------------------------------
 * 向码流文件中输出one bit
 */
static INLINE
void bitstr_put_one_bit(aec_t *p_aec, uint32_t b)
{
    p_aec->reg_flush_bits |= ((b) << --p_aec->num_left_flush_bits);
    if (!p_aec->num_left_flush_bits) {
        bitstr_flush_bits(p_aec);
        p_aec->num_left_flush_bits = NUM_FLUSH_BITS;
    }
}

/* ---------------------------------------------------------------------------
 * 判断CG是否为全零块。
 * 是则返回1，否则返回0
 */
static ALWAYS_INLINE
int aec_is_cg_allzero(const coeff_t *src_coeff, int i_stride_shift)
{
    assert(sizeof(coeff_t) * 4 == sizeof(uint64_t));
    /* 64 bit */
    return (*(uint64_t *)(src_coeff) == 0 &&
            *(uint64_t *)(src_coeff + (uint64_t)(tab_coeff_offset[i_stride_shift][0])) == 0 &&
            *(uint64_t *)(src_coeff + (uint64_t)(tab_coeff_offset[i_stride_shift][1])) == 0 &&
            *(uint64_t *)(src_coeff + (uint64_t)(tab_coeff_offset[i_stride_shift][2])) == 0);
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
int tu_get_cg_run_level_info(runlevel_t *runlevel,
                             const coeff_t *quant_coeff, int i_stride_shift,
                             const int b_hor)
{
    uint64_t c1 = *(uint64_t *)(quant_coeff);
    uint64_t c2 = *(uint64_t *)(quant_coeff + (intptr_t)(tab_coeff_offset[i_stride_shift][0]));
    uint64_t c3 = *(uint64_t *)(quant_coeff + (intptr_t)(tab_coeff_offset[i_stride_shift][1]));
    uint64_t c4 = *(uint64_t *)(quant_coeff + (intptr_t)(tab_coeff_offset[i_stride_shift][2]));
    if (c1 == 0 && c2 == 0 && c3 == 0 && c4 == 0) {
        return 0;
    } else {
        ALIGN32(coeff_t res[16]);
        runlevel_pair_t *p_runlevel = runlevel->runlevels_cg;
        int8_t   run = 0;         // here is not -1
        int      num_runlevel_pair = 0;
        int   i;

        g_funcs.transpose_coeff_4x4[b_hor](res, c1, c2, c3, c4);
        /* prepare run-level pairs in one CG */
        for (i = 0; i < 16; i++) {
            coeff_t level = res[i];

            if (level != 0) {
                num_runlevel_pair++;
                p_runlevel->level = level;
                p_runlevel->run   = run;
                p_runlevel++;
                run = 0;
            } else {
                run++;
            }
        }

        runlevel->last_pos_cg = run;
        return num_runlevel_pair;
    }
}


/*
 * ===========================================================================
 * function declares
 * ===========================================================================
 */

#if CTRL_OPT_AEC
/* init AEC context table */
#define init_aec_context_tab FPFX(init_aec_context_tab)
void init_aec_context_tab(void);
#endif

/* ---------------------------------------------------------------------------
 * coding state initialization (no need to destroy, just free the space is OK)
 */
#define aec_init_coding_state FPFX(aec_init_coding_state)
void aec_init_coding_state   (aec_t *p_aec);


/* ---------------------------------------------------------------------------
 * aec functions
 */
#define aec_start FPFX(aec_start)
void aec_start(xavs2_t *h, aec_t *p_aec, uint8_t *p_bs_start, uint8_t *p_bs_end, int b_writing);
#define aec_done FPFX(aec_done)
void aec_done(aec_t *p_aec);

/* AEC */
#define xavs2_lcu_write FPFX(lcu_write)
void xavs2_lcu_write(xavs2_t *h, aec_t *p_aec, lcu_info_t *lcu_info, int i_level, int img_x, int img_y);
#define xavs2_lcu_terminat_bit_write FPFX(lcu_terminat_bit_write)
void xavs2_lcu_terminat_bit_write(aec_t *p_aec, uint8_t bit);

#endif  // XAVS2_AEC_H
