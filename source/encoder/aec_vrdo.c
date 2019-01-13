/*
 * aec_vrdo.c
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

#include "common.h"
#include "aec.h"
#include "bitstream.h"
#include "block_info.h"
#include "cudata.h"

/**
 * ===========================================================================
 * binary
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE int aec_get_shift(uint32_t v)
{
#if SYS_WINDOWS && !ARCH_X86_64
    __asm {
        bsr     eax, v
        mov     v, eax
    }

    return 8 - v;
#else
    int i;

    for (i = 0; !(v & 0x100); i++) {
        v <<= 1;
    }

    return i;
#endif
}

/* ---------------------------------------------------------------------------
 */
#define biari_encode_symbol_vrdo(p_aec, symbol, p_ctx)  \
                                 p_aec->i_bits_to_follow++
#define DECLARE_CONTEXT(p_ctx)  

/* ---------------------------------------------------------------------------
 */
#define biari_encode_tu_vrdo(p_aec, num_zeros, max_len, p_ctx) \
    p_aec->i_bits_to_follow += num_zeros + (max_len != num_zeros)

/* ---------------------------------------------------------------------------
 */
#define biari_encode_symbol_eq_prob_vrdo(p_aec, symbol)  p_aec->i_bits_to_follow++

/* ---------------------------------------------------------------------------
 */
#define biari_encode_symbols_eq_prob_vrdo(p_aec, val, len) p_aec->i_bits_to_follow += len

/* ---------------------------------------------------------------------------
 */
static INLINE
void biari_encode_symbol_final_vrdo(aec_t *p_aec, uint8_t symbol)
{
    if (symbol) {
        p_aec->i_bits_to_follow += 8;  // (!p_aec->i_t1) + 8;
    }
}

/**
 * ===========================================================================
 * syntax coding
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * cu type for B/F/P frame
 */
static INLINE
int aec_write_cutype_vrdo(aec_t *p_aec, int i_cu_type, int i_cu_level, int i_cu_cbp, int is_amp_enabled)
{
    DECLARE_CONTEXT(context_t *p_ctx = p_aec->p_ctx_set->cu_type_contexts);
    int org_bits = rdo_get_written_bits(p_aec);
    int act_sym = MAP_CU_TYPE[i_cu_type];

    if (i_cu_type == PRED_SKIP && i_cu_cbp == 0) {
        act_sym = 0;
    }

    switch (act_sym) {
    case 0:     // SKIP
        biari_encode_symbol_vrdo(p_aec, 1, p_ctx + 0);
        break;
    case 1:     // DIRECT
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 0);
        biari_encode_symbol_vrdo(p_aec, 1, p_ctx + 1);
        break;
    case 2:     // 2Nx2N
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 0);
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 1);
        biari_encode_symbol_vrdo(p_aec, 1, p_ctx + 2);
        break;
    case 3:     // 2NxN, 2NxnU, 2NxnD
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 0);
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 1);
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 2);
        biari_encode_symbol_vrdo(p_aec, 1, p_ctx + 3);
        if (is_amp_enabled && i_cu_level >= B16X16_IN_BIT) {
            DECLARE_CONTEXT(p_ctx = p_aec->p_ctx_set->shape_of_partition_index);
            if (i_cu_type == PRED_2NxN) {
                biari_encode_symbol_vrdo(p_aec, 1, p_ctx);   // SMP - AMP signal bit
            } else {
                biari_encode_symbol_vrdo(p_aec, 0, p_ctx);   // SMP - AMP signal bit
                biari_encode_symbol_vrdo(p_aec, (uint8_t)(i_cu_type == PRED_2NxnU), p_ctx + 1);  // AMP shape
            }
        }
        break;
    case 4:     // Nx2N, nLx2N, nRx2N
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 0);
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 1);
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 2);
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 3);
        biari_encode_symbol_vrdo(p_aec, 1, p_ctx + 4);
        if (is_amp_enabled && i_cu_level >= B16X16_IN_BIT) {
            DECLARE_CONTEXT(p_ctx = p_aec->p_ctx_set->shape_of_partition_index);
            if (i_cu_type == PRED_Nx2N) {
                biari_encode_symbol_vrdo(p_aec, 1, p_ctx);   // SMP - AMP signal bit
            } else {
                biari_encode_symbol_vrdo(p_aec, 0, p_ctx);   // SMP - AMP signal bit
                biari_encode_symbol_vrdo(p_aec, (uint8_t)(i_cu_type == PRED_nLx2N), p_ctx + 1);  // AMP shape
            }
        }
        break;
    //case 5:     // NxN, not enabled
    //    biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 0);
    //    biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 1);
    //    biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 2);
    //    biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 3);
    //    biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 4);
    //    if (i_cu_level > B8X8_IN_BIT) {
    //        biari_encode_symbol_final_vrdo(p_aec, 1);
    //    }
    //    break;
    default:    // case 6:  // Intra
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 0);
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 1);
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 2);
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 3);
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 4);
        if (i_cu_level > B8X8_IN_BIT) {
            biari_encode_symbol_final_vrdo(p_aec, 0);
        }
        break;
    }

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}

/* ---------------------------------------------------------------------------
 * arithmetically encode a pair of intra prediction modes of a given cu
 */
static
int aec_write_intra_pred_mode_vrdo(aec_t *p_aec, int ipmode)
{
    DECLARE_CONTEXT(context_t *p_ctx = p_aec->p_ctx_set->intra_luma_pred_mode);
    int org_bits = rdo_get_written_bits(p_aec);

    if (ipmode >= 0) {
        biari_encode_symbol_vrdo(p_aec, 0,                               p_ctx    );
        biari_encode_symbol_vrdo(p_aec, (uint8_t)((ipmode & 0x10) >> 4), p_ctx + 1);
        biari_encode_symbol_vrdo(p_aec, (uint8_t)((ipmode & 0x08) >> 3), p_ctx + 2);
        biari_encode_symbol_vrdo(p_aec, (uint8_t)((ipmode & 0x04) >> 2), p_ctx + 3);
        biari_encode_symbol_vrdo(p_aec, (uint8_t)((ipmode & 0x02) >> 1), p_ctx + 4);
        biari_encode_symbol_vrdo(p_aec, (uint8_t)((ipmode & 0x01)     ), p_ctx + 5);
    } else {
        biari_encode_symbol_vrdo(p_aec, 1,                     p_ctx    );
        biari_encode_symbol_vrdo(p_aec, (uint8_t)(ipmode + 2), p_ctx + 6);
    }

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}

/* ---------------------------------------------------------------------------
 * arithmetically encode the reference parameter of a given cu
 */
static INLINE
int aec_write_ref_vrdo(xavs2_t *h, aec_t *p_aec, int ref_idx)
{
    context_t *p_ctx = p_aec->p_ctx_set->pu_reference_index;
    int org_bits = rdo_get_written_bits(p_aec);
    int act_sym  = ref_idx;

    /* 第0位用0号上下文，第1位用1号上下文，其他用2号上下文 */
    if (act_sym == 0) {
        biari_encode_symbol_vrdo(p_aec, 1, p_ctx);
    } else {
        int act_ctx = 0;
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx++);

        while (--act_sym != 0) {
            biari_encode_symbol_vrdo(p_aec, 0, p_ctx);
            if (!act_ctx) {
                p_ctx++;
            }
        }

        if (ref_idx < h->i_ref - 1) {
            biari_encode_symbol_vrdo(p_aec, 1, p_ctx);
        }
    }

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}

/* ---------------------------------------------------------------------------
 * arithmetically encode the motion vector data
 */
static INLINE
int aec_write_mvd_vrdo(aec_t *p_aec, int mvd, int xy)
{
    int org_bits = rdo_get_written_bits(p_aec);
    uint32_t act_sym = XAVS2_ABS(mvd);

    UNUSED_PARAMETER(xy);

    if (act_sym == 0) {
        p_aec->i_bits_to_follow += 1;
    } else if (act_sym < 3) { // 1, 2
        p_aec->i_bits_to_follow += act_sym + 2;
    } else {
        int exp_golomb_order = 0;
        // odds > 3: (act_sym - 3) >> 1
        // even > 3: (act_sym - 4) >> 1
        act_sym = (act_sym - 3) >> 1;

        /* exp_golomb part */
        while (act_sym >= (uint32_t)(1 << exp_golomb_order)) {
            act_sym -= (1 << exp_golomb_order);
            exp_golomb_order++;
        }
        // Exp-Golomb: (prefix) + 1 + (suffix)
        p_aec->i_bits_to_follow += 5 + (exp_golomb_order + 1 + exp_golomb_order);
    }

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}

/* ---------------------------------------------------------------------------
 */
static INLINE
int aec_write_dmh_mode_vrdo(aec_t *p_aec, int i_cu_level, int dmh_mode)
{
    static const int iEncMapTab[9] = { 0, 5, 6, 1, 2, 7, 8, 3, 4 };
    context_t *p_ctx = p_aec->p_ctx_set->pu_type_index + 3;
    int org_bits = rdo_get_written_bits(p_aec);
    int symbol   = dmh_mode != 0;

    p_ctx += (i_cu_level - MIN_CU_SIZE_IN_BIT) * 3;
    biari_encode_symbol_vrdo(p_aec, (uint8_t)symbol, p_ctx);

    if (symbol) {
        int iMapVal = iEncMapTab[dmh_mode];

        if (iMapVal < 3) {
            symbol = (iMapVal != 1);
            biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 1);
            biari_encode_symbol_eq_prob_vrdo(p_aec, (uint8_t)symbol);
        } else if (iMapVal < 5) {
            symbol = (iMapVal != 3);
            biari_encode_symbol_vrdo(p_aec, 1, p_ctx + 1);
            biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 2);
            biari_encode_symbol_eq_prob_vrdo(p_aec, (uint8_t)symbol);
        } else {
            biari_encode_symbol_vrdo(p_aec, 1, p_ctx + 1);
            biari_encode_symbol_vrdo(p_aec, 1, p_ctx + 2);
            symbol = (iMapVal >= 7);
            biari_encode_symbol_eq_prob_vrdo(p_aec, (uint8_t)symbol);
            symbol = !(iMapVal & 1);
            biari_encode_symbol_eq_prob_vrdo(p_aec, (uint8_t)symbol);
        }
    }

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}

/* ---------------------------------------------------------------------------
 * write "transform_split_flag" and SDIP type for intra CU
 */
static INLINE
int aec_write_intra_cutype_vrdo(aec_t *p_aec, int i_cu_type, int i_cu_level, int i_tu_split, int is_sdip_enabled)
{
    int org_bits = rdo_get_written_bits(p_aec);

    UNUSED_PARAMETER(i_cu_type);
    if (i_cu_level == B8X8_IN_BIT) {
        p_aec->i_bits_to_follow++;
    } else if (is_sdip_enabled && (i_cu_level == B32X32_IN_BIT || i_cu_level == B16X16_IN_BIT)) {
        uint8_t transform_split_flag = i_tu_split != TU_SPLIT_NON;  /* just write split or not */

        p_aec->i_bits_to_follow += 1 + transform_split_flag;
    }

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}

/* ---------------------------------------------------------------------------
 */
static INLINE
int aec_write_pdir_vrdo(aec_t *p_aec, int i_cu_type, int i_cu_level, int pdir0, int pdir1)
{
    int new_pdir[4] = { 2, 1, 3, 0 };
    context_t *p_ctx = p_aec->p_ctx_set->pu_type_index;
    int org_bits = rdo_get_written_bits(p_aec);
    int act_ctx  = 0;
    int act_sym;
    int symbol;

    if (i_cu_type == PRED_2Nx2N) {
        /* 一个CU只有一个PU的情况，这个PU可以有四个方向，使用上下文3个，编号: 0, 1, 2 */
        act_sym = pdir0;
        while (act_sym >= 1) {
            biari_encode_symbol_vrdo(p_aec, 0, p_ctx + act_ctx);
            act_sym--;
            act_ctx++;
        }
        if (pdir0 != 3) {
            biari_encode_symbol_vrdo(p_aec, 1, p_ctx + act_ctx);
        }
    } else if ((i_cu_type >= PRED_2NxN && i_cu_type <= PRED_nRx2N) && i_cu_level == B8X8_IN_BIT) {
        /* 一个CU分为两个PU，且CU大小为8x8，这时预测块为4x8或8x4，每个PU只能是单方向的预测，
         * 总计有4种组合，需要编码两位Bit，编码 b_pu_type_min_index 共计使用两个上下文 */
        p_ctx = p_aec->p_ctx_set->b_pu_type_min_index;
        pdir0 = new_pdir[pdir0];
        pdir1 = new_pdir[pdir1];

        act_sym = (pdir0 != 1);
        biari_encode_symbol_vrdo(p_aec, (int8_t)act_sym, p_ctx + 0);
        act_sym = (pdir0 == pdir1);
        biari_encode_symbol_vrdo(p_aec, (int8_t)act_sym, p_ctx + 1);
    } else if (i_cu_type >= PRED_2NxN || i_cu_type <= PRED_nRx2N) { //1010
        /* act_ctx: 3,...,14 */
        pdir0 = new_pdir[pdir0];
        pdir1 = new_pdir[pdir1];
        act_sym = pdir0;
        act_ctx = 3;

        /* 3,4,5 */
        while (act_sym >= 1) {
            biari_encode_symbol_vrdo(p_aec, 0, p_ctx + act_ctx);
            act_sym--;
            act_ctx++;
        }
        if (pdir0 != 3) {
             biari_encode_symbol_vrdo(p_aec, 1, p_ctx + act_ctx);
        }

        symbol = (pdir0 == pdir1);
        biari_encode_symbol_vrdo(p_aec, (uint8_t)symbol, p_ctx + 6);

        /* 7,...,14 */
        if (!symbol) {
            switch (pdir0) {
            case 0:
                symbol = (pdir1 == 1);
                biari_encode_symbol_vrdo(p_aec, (uint8_t)symbol, p_ctx + 7);
                if (!symbol) {
                    symbol = (pdir1 == 2);
                    biari_encode_symbol_vrdo(p_aec, (uint8_t)symbol, p_ctx + 8);
                }
                break;
            case 1:
                symbol = (pdir1 == 0);
                biari_encode_symbol_vrdo(p_aec, (uint8_t)symbol, p_ctx + 9);
                if (!symbol) {
                    symbol = (pdir1 == 2);
                    biari_encode_symbol_vrdo(p_aec, (uint8_t)symbol, p_ctx + 10);
                }
                break;
            case 2:
                symbol = (pdir1 == 0);
                biari_encode_symbol_vrdo(p_aec, (uint8_t)symbol, p_ctx + 11);
                if (!symbol) {
                    symbol = (pdir1 == 1);
                    biari_encode_symbol_vrdo(p_aec, (uint8_t)symbol, p_ctx + 12);
                }
                break;
            case 3:
                symbol = (pdir1 == 0);
                biari_encode_symbol_vrdo(p_aec, (uint8_t)symbol, p_ctx + 13);
                if (!symbol) {
                    symbol = (pdir1 == 1);
                    biari_encode_symbol_vrdo(p_aec, (uint8_t)symbol, p_ctx + 14);
                }
                break;
            }
        }
    }

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}

/* ---------------------------------------------------------------------------
 */
static INLINE
int aec_write_pdir_dhp_vrdo(aec_t *p_aec, int i_cu_type, int pdir0, int pdir1)
{
    DECLARE_CONTEXT(context_t *p_ctx = p_aec->p_ctx_set->pu_type_index);
    int org_bits = rdo_get_written_bits(p_aec);

    pdir0 = (pdir0 != 0);
    pdir1 = (pdir1 != 0);

    if (i_cu_type == PRED_2Nx2N) {
        biari_encode_symbol_vrdo(p_aec, (uint8_t)pdir0, p_ctx);
    } else if (i_cu_type >= PRED_2NxN || i_cu_type <= PRED_nRx2N) { // 1010
        biari_encode_symbol_vrdo(p_aec, (uint8_t)pdir0,            p_ctx + 1);
        biari_encode_symbol_vrdo(p_aec, (uint8_t)(pdir0 == pdir1), p_ctx + 2);
    }

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}

/* ---------------------------------------------------------------------------
 */
static INLINE
int aec_write_wpm_vrdo(aec_t *p_aec, int ref_idx, int num_ref)
{
    DECLARE_CONTEXT(context_t *p_ctx = p_aec->p_ctx_set->weighted_skip_mode);
    int org_bits = rdo_get_written_bits(p_aec);
    int i, idx_bin = 0;

    for (i = 0; i < ref_idx; i++) {
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + idx_bin);
        idx_bin = XAVS2_MIN(idx_bin + 1, 2);
    }

    if (ref_idx < num_ref - 1) {
        biari_encode_symbol_vrdo(p_aec, 1, p_ctx + idx_bin);
    }

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}

/* ---------------------------------------------------------------------------
 */
static INLINE
int aec_write_spatial_skip_mode_vrdo(aec_t *p_aec, int mode)
{
    DECLARE_CONTEXT(context_t *p_ctx = p_aec->p_ctx_set->cu_subtype_index);
    int org_bits = rdo_get_written_bits(p_aec);
    int offset;

    for (offset = 0; offset < mode; offset++) {
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + offset);
    }

    if (mode < DS_MAX_NUM) {
        biari_encode_symbol_vrdo(p_aec, 1, p_ctx + offset);
    }

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}

/* ---------------------------------------------------------------------------
 * arithmetically encode the chroma intra prediction mode of an 8x8 block
 */
static INLINE
int aec_write_intra_pred_cmode_vrdo(aec_t *p_aec, cu_info_t *p_cu_info, int i_left_cmode)
{
    int i_chroma_mode = p_cu_info->i_intra_mode_c;
    int org_bits      = rdo_get_written_bits(p_aec);
    UNUSED_PARAMETER(i_left_cmode);

    if (i_chroma_mode == DM_PRED_C) {
        biari_encode_symbol_vrdo(p_aec, 1, p_ctx + act_ctx);
    } else {
        int lmode = tab_intra_mode_luma2chroma[p_cu_info->real_intra_modes[0]];
        int is_redundant = lmode >= 0;

        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + act_ctx);
        i_chroma_mode -= (1 + (is_redundant && i_chroma_mode > lmode));

        p_aec->i_bits_to_follow += i_chroma_mode;
        if (i_chroma_mode < 3) {
            p_aec->i_bits_to_follow++;
        }
    }

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}

/* ---------------------------------------------------------------------------
 * arithmetically encode the coded block pattern of an luma CB
 */
#define write_cbp_bit_vrdo(h, p_aec, p_cu_info, b8, bit) p_aec->i_bits_to_follow++

/* ---------------------------------------------------------------------------
 * arithmetically encode the coded block pattern of a cu
 */
static INLINE
int aec_write_cu_cbp_vrdo(aec_t *p_aec, cu_info_t *p_cu_info, int slice_index_cur_cu, xavs2_t *h)
{
    DECLARE_CONTEXT(context_t *p_ctx = p_aec->p_ctx_set->cbp_contexts + 4);
    int org_bits = rdo_get_written_bits(p_aec);
    int i_cu_cbp = p_cu_info->i_cbp;
    int i_cu_type = p_cu_info->i_mode;
    int transform_split_flag = p_cu_info->i_tu_split != TU_SPLIT_NON;

    UNUSED_PARAMETER(slice_index_cur_cu);

    if (IS_INTER_MODE(i_cu_type)) {
        /* write cbp for inter pred mode ---------------------------
         */
        if (!IS_SKIP_MODE(i_cu_type)) {
            write_cbp_bit_vrdo(h, p_aec, p_cu_info, 4, i_cu_cbp == 0);
        }

        if (i_cu_cbp) {
            // write tr_size
            biari_encode_symbol_vrdo(p_aec, (uint8_t)transform_split_flag, p_aec->p_ctx_set->transform_split_flag);

            // write cbp for chroma
            if (h->param->chroma_format != CHROMA_400) {
                switch ((i_cu_cbp >> 4) & 0x03) {
                case 0:
                    biari_encode_symbol_vrdo(p_aec, 0, p_ctx);
                    break;
                case 1:
                    biari_encode_symbol_vrdo(p_aec, 1, p_ctx);
                    biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 2);
                    biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 2);
                    break;
                case 2:
                    biari_encode_symbol_vrdo(p_aec, 1, p_ctx);
                    biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 2);
                    biari_encode_symbol_vrdo(p_aec, 1, p_ctx + 2);
                    break;
                case 3:
                    biari_encode_symbol_vrdo(p_aec, 1, p_ctx);
                    biari_encode_symbol_vrdo(p_aec, 1, p_ctx + 2);
                    break;
                }
            }

            // write cbp for luma
            if (transform_split_flag == 0) {
                if (i_cu_cbp > 15) {
                    write_cbp_bit_vrdo(h, p_aec, p_cu_info, 0, (i_cu_cbp & 1) != 0);
                }
            } else {
                write_cbp_bit_vrdo(h, p_aec, p_cu_info, 0, (i_cu_cbp & 1) != 0);
                write_cbp_bit_vrdo(h, p_aec, p_cu_info, 1, (i_cu_cbp & 2) != 0);
                write_cbp_bit_vrdo(h, p_aec, p_cu_info, 2, (i_cu_cbp & 4) != 0);
                write_cbp_bit_vrdo(h, p_aec, p_cu_info, 3, (i_cu_cbp & 8) != 0);
            }
        }
    } else {
        /* write cbp for intra pred mode ---------------------------
         */

        // write bits for luma
        if (transform_split_flag == 0 || i_cu_type == PRED_I_2Nx2N) {
            write_cbp_bit_vrdo(h, p_aec, p_cu_info, 0, (i_cu_cbp & 0x0F) != 0);
        } else {
            write_cbp_bit_vrdo(h, p_aec, p_cu_info, 0, (i_cu_cbp & 1) != 0);
            write_cbp_bit_vrdo(h, p_aec, p_cu_info, 1, (i_cu_cbp & 2) != 0);
            write_cbp_bit_vrdo(h, p_aec, p_cu_info, 2, (i_cu_cbp & 4) != 0);
            write_cbp_bit_vrdo(h, p_aec, p_cu_info, 3, (i_cu_cbp & 8) != 0);
        }

        // write bits for chroma
        if (h->param->chroma_format != CHROMA_400) {
            switch ((i_cu_cbp >> 4) & 0x03) {
            case 0:
                biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 1);
                break;
            case 1:
                biari_encode_symbol_vrdo(p_aec, 1, p_ctx + 1);
                biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 3);
                biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 3);
                break;
            case 2:
                biari_encode_symbol_vrdo(p_aec, 1, p_ctx + 1);
                biari_encode_symbol_vrdo(p_aec, 0, p_ctx + 3);
                biari_encode_symbol_vrdo(p_aec, 1, p_ctx + 3);
                break;
            case 3:
                biari_encode_symbol_vrdo(p_aec, 1, p_ctx + 1);
                biari_encode_symbol_vrdo(p_aec, 1, p_ctx + 3);
                break;
            }
        }
    }

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}

#if ENABLE_RATE_CONTROL_CU
/* ---------------------------------------------------------------------------
 */
static INLINE
int aec_write_dqp_vrdo(aec_t *p_aec, int delta_qp, int last_dqp)
{
    context_t *p_ctx = p_aec->p_ctx_set->delta_qp_contexts;
    int org_bits = rdo_get_written_bits(p_aec);
    int act_ctx  = (last_dqp) ? 1 : 0;
    int act_sym  = (delta_qp > 0) ? (2 * delta_qp - 1) : (-2 * delta_qp);

    if (act_sym == 0) {
        biari_encode_symbol_vrdo(p_aec, 1, p_ctx + act_ctx);
    } else {
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx + act_ctx);
        act_ctx = 2;
        if (act_sym == 1) {
            biari_encode_symbol_vrdo(p_aec, 1, p_ctx + act_ctx);
        } else {
            biari_encode_symbol_vrdo(p_aec, 0, p_ctx + act_ctx);
            act_ctx++;
            while (act_sym > 2) {
                biari_encode_symbol_vrdo(p_aec, 0, p_ctx + act_ctx);
                act_sym--;
            }
            biari_encode_symbol_vrdo(p_aec, 1, p_ctx + act_ctx);
        }
    }

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}
#endif

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
void aec_write_last_cg_pos(aec_t *p_aec, int b_luma, int b_dc_diag, 
                           int i_cg, int cg_last_x, int cg_last_y, 
                           int num_cg, int num_cg_x_minus1, int num_cg_y_minus1)
{
    int count;

    if (num_cg == 4) {   // 8x8
        p_aec->i_bits_to_follow += XAVS2_MIN(3, i_cg + 1);
    } else {
        if (b_luma && b_dc_diag) {
            XAVS2_SWAP(cg_last_x, cg_last_y);
            XAVS2_SWAP(num_cg_x_minus1, num_cg_y_minus1);
        }

        p_aec->i_bits_to_follow++;      /* last_cg0_flag */

        if (cg_last_x || cg_last_y) {
            /* last_cg_x */
            biari_encode_tu_vrdo(p_aec, cg_last_x, num_cg_x_minus1, NULL);

            /* last_cg_y or last_cg_y_minus1 */
            count = (cg_last_x == 0);  // 若cg_last_x为零，则cg_last_y可少写一个零（两者至少有一个非零）
            biari_encode_tu_vrdo(p_aec, cg_last_y - count, num_cg_y_minus1 - count, NULL);
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
void aec_write_last_coeff_pos(aec_t *p_aec, int isLastCG,
                              int pos, 
                              int b_dc_diag)
{
    static const ALIGN32(int8_t tab_bits_pos[4][16]) = {
        { 4, 5, 4, 3, 5, 6, 6, 6, 4, 2, 3, 5, 6, 5, 4, 4 },
        { 2, 3, 3, 4, 4, 4, 4, 5, 5, 4, 5, 6, 5, 6, 6, 6 },
        { 6, 6, 6, 5, 6, 5, 4, 5, 5, 4, 4, 4, 4, 3, 3, 2 },
        { 6, 6, 6, 5, 6, 5, 4, 5, 5, 4, 4, 4, 4, 3, 3, 2 },
    };
    // int scan_pos = tab_1d_scan_4x4[15 - pos];
    p_aec->i_bits_to_follow += tab_bits_pos[(isLastCG << 1) + b_dc_diag][pos];
}

/* ---------------------------------------------------------------------------
 */
static
int aec_write_run_level_luma_vrdo(aec_t *p_aec, int b_dc_diag,
                                 runlevel_t *runlevel, xavs2_t *h, int maxvalue)
{
    const int16_t(*p_tab_cg_scan)[2] = runlevel->tab_cg_scan;
    runlevel_pair_t *p_runlevel = runlevel->runlevels_cg;
    int rank          = 0;
    int num_cg        = runlevel->num_cg;
    int org_bits      = rdo_get_written_bits(p_aec);
    int i_cg;
    int cur_bits;
    UNUSED_PARAMETER(h);

    /* write coefficients in CG */
    for (i_cg = num_cg - 1; i_cg >= 0; i_cg--) {
        DECLARE_CONTEXT(context_t *p_ctx = NULL);
        int CGx = 0;
        int CGy = 0;
        DECLARE_CONTEXT(uint32_t Level_sign = 0);
        int pos;
        int num_pairs;
        int pairs;

        /* 1. 检查当前CG是否包含有非零系数 */
        coeff_t *quant_coeff = runlevel->quant_coeff;
        const int b_hor = runlevel->b_hor;
        quant_coeff += ((p_tab_cg_scan[i_cg][!b_hor] << runlevel->i_stride_shift) + p_tab_cg_scan[i_cg][b_hor]) << 2;
        // number of pairs in CG
        num_pairs = tu_get_cg_run_level_info(runlevel, quant_coeff, runlevel->i_stride_shift, runlevel->b_hor);

        /* 2, Sig CG Flag, "nonzero_cg_flag" */
        if (rank > 0) {
            biari_encode_symbol_vrdo(p_aec, !!num_pairs, p_ctx);
            if (!num_pairs) {
                continue;       // 无非零系数，结束当前CG编码
            }
            CGx = p_tab_cg_scan[i_cg][0];
            CGy = p_tab_cg_scan[i_cg][1];
        } else if (num_pairs > 0) {
            if (num_cg > 1) {   // for TB > 4x4, need to write
                int num_cg_x = p_tab_cg_scan[num_cg - 1][0];
                int num_cg_y = p_tab_cg_scan[num_cg - 1][1];

                CGx = p_tab_cg_scan[i_cg][0];
                CGy = p_tab_cg_scan[i_cg][1];
                aec_write_last_cg_pos(p_aec, 1, b_dc_diag, i_cg, CGx, CGy, num_cg, num_cg_x, num_cg_y);
            }
        } else {
            continue;
        }

        /* early terminate? */
        CHECK_EARLY_RETURN_RUNLEVEL(p_aec);

        /* 3, (Run, Level) */

        /* 3.1, LAST IN CG, "last_coeff_pos_x", "last_coeff_pos_y" */
        pos   = runlevel->last_pos_cg;
        pairs = num_pairs - 1;

        {
            aec_write_last_coeff_pos(p_aec, rank == 0, 
                                     pos, b_dc_diag);
        }

        for (; pairs >= 0 && pos < NUM_OF_COEFFS_IN_CG; pairs--) {
            DECLARE_CONTEXT(int offset = 0);
            int Level = p_runlevel[pairs].level;
            int Run   = p_runlevel[pairs].run;
            int absLevel = XAVS2_ABS(Level);
            int symbol = absLevel - 1;

            DECLARE_CONTEXT(Level_sign |= (Level < 0) << (pairs + 1));      // record Sign

            /* 3.2, level, "coeff_level_minus1_band[i]", "coeff_level_minus1_pos_in_band[i]" */
            if (symbol > 31) {
                int exp_golomb_order = 0;

                biari_encode_symbol_final_vrdo(p_aec, 1);  // "coeff_level_minus1_band[i]", > 32

                /* coeff_level_minus1_pos_in_band[i] */
                symbol -= 32;
                while (symbol >= (1 << exp_golomb_order)) {
                    symbol -= (1 << exp_golomb_order);
                    exp_golomb_order++;
                }
                biari_encode_symbols_eq_prob_vrdo(p_aec, 1, exp_golomb_order + 1);  // Exp-Golomb: prefix and 1
                biari_encode_symbols_eq_prob_vrdo(p_aec, symbol, exp_golomb_order); // Exp-Golomb: suffix
            } else {
                biari_encode_symbol_final_vrdo(p_aec, 0);  // "coeff_level_minus1_band[i]", <= 32

                /* coeff_level_minus1_pos_in_band[i] */
                biari_encode_tu_vrdo(p_aec, symbol, 31, p_ctx);
            }

            /* 3.3, run, "coeff_run[i]" */
            p_aec->i_bits_to_follow += Run;

            pos += (Run + 1);   // update position
        }   // run-level loop

        p_aec->i_bits_to_follow += (num_pairs - 1);
        rank = 1;

        /* 4, sign of coefficient */
        biari_encode_symbols_eq_prob_vrdo(p_aec, Level_sign >> 1, num_pairs);

        /* early terminate? */
        CHECK_EARLY_RETURN_RUNLEVEL(p_aec);
    }   // for (; i_cg >= 0; i_cg--) 

    /* get the number of written bits */
    org_bits = rdo_get_written_bits(p_aec) - org_bits;

#ifdef DEBUG
    if (rank == 0) {
        xavs2_log(h, XAVS2_LOG_ERROR, "no non-zero run-level luma, POC[%d]: p_cu: (%d, %d), level %d, cu_type %d\n",
            h->fdec->i_poc, runlevel->p_cu_info->i_scu_x, runlevel->p_cu_info->i_scu_y, runlevel->p_cu_info->i_level, 
            runlevel->p_cu_info->i_mode);
    }
#endif
    assert(rank > 0);  // 当有非零系数时，rank的值应大于零

    /* return the number of written bits */
    return org_bits;
}


/* ---------------------------------------------------------------------------
 */
static
int aec_write_run_level_chroma_vrdo(aec_t *p_aec, runlevel_t *runlevel, xavs2_t *h, int maxvalue)
{
    const int16_t(*p_tab_cg_scan)[2] = runlevel->tab_cg_scan;
    runlevel_pair_t *p_runlevel = runlevel->runlevels_cg;
    int rank          = 0;
    int num_cg        = runlevel->num_cg;
    int org_bits      = rdo_get_written_bits(p_aec);
    int i_cg;
    int cur_bits;
    UNUSED_PARAMETER(h);

    /* write coefficients in CG */
    for (i_cg = num_cg - 1; i_cg >= 0; i_cg--) {
        DECLARE_CONTEXT(context_t *p_ctx);
        int CGx = 0;
        int CGy = 0;
        DECLARE_CONTEXT(uint32_t Level_sign = 0);
        int pos;
        int num_pairs;
        int pairs;

        /* 1. 检查当前CG是否包含有非零系数 */
        coeff_t *quant_coeff = runlevel->quant_coeff;
        const int b_hor = 0; // runlevel->b_hor;
        quant_coeff += ((p_tab_cg_scan[i_cg][!b_hor] << runlevel->i_stride_shift) + p_tab_cg_scan[i_cg][b_hor]) << 2;
        // number of pairs in CG
        num_pairs = tu_get_cg_run_level_info(runlevel, quant_coeff, runlevel->i_stride_shift, b_hor);

        /* 2, Sig CG Flag, "nonzero_cg_flag" */
        if (rank > 0) {
            biari_encode_symbol_vrdo(p_aec, !!num_pairs, p_ctx);
            if (!num_pairs) {
                continue;       // 无非零系数，结束当前CG编码
            }
            CGx = p_tab_cg_scan[i_cg][0];
            CGy = p_tab_cg_scan[i_cg][1];
        } else if (num_pairs > 0) {
            if (num_cg > 1) {
                int num_cg_x = p_tab_cg_scan[num_cg - 1][0];
                int num_cg_y = p_tab_cg_scan[num_cg - 1][1];

                CGx = p_tab_cg_scan[i_cg][0];
                CGy = p_tab_cg_scan[i_cg][1];
                aec_write_last_cg_pos(p_aec, 0, 1, i_cg, CGx, CGy, num_cg, num_cg_x, num_cg_y);
            }
        } else {
            continue;  // 未找到第一个包含非零系数的CG
        }

        /* early terminate? */
        CHECK_EARLY_RETURN_RUNLEVEL(p_aec);

        /* 3, (Run, Level) */

        /* 3.1, LAST IN CG, "last_coeff_pos_x", "last_coeff_pos_y" */
        pos   = runlevel->last_pos_cg;
        pairs = num_pairs - 1;

        {
            aec_write_last_coeff_pos(p_aec, rank == 0,
                                     pos, 1);
        }

        for (; pairs >= 0 && pos < NUM_OF_COEFFS_IN_CG; pairs--) {
            DECLARE_CONTEXT(int offset = 0);
            int Level = p_runlevel[pairs].level;
            int Run   = p_runlevel[pairs].run;
            int absLevel = XAVS2_ABS(Level);
            int symbol = absLevel - 1;

            DECLARE_CONTEXT(Level_sign |= (Level < 0) << (pairs + 1));      // record Sign

            /* 3.2, level, "coeff_level_minus1_band[i]", "coeff_level_minus1_pos_in_band[i]" */
            if (symbol > 31) {
                int exp_golomb_order = 0;

                biari_encode_symbol_final_vrdo(p_aec, 1);  // "coeff_level_minus1_band[i]", > 32

                /* coeff_level_minus1_pos_in_band[i] */
                symbol -= 32;
                while (symbol >= (1 << exp_golomb_order)) {
                    symbol -= (1 << exp_golomb_order);
                    exp_golomb_order++;
                }
                biari_encode_symbols_eq_prob_vrdo(p_aec, 1, exp_golomb_order + 1);  // Exp-Golomb: prefix and 1
                biari_encode_symbols_eq_prob_vrdo(p_aec, symbol, exp_golomb_order); // Exp-Golomb: suffix
            } else {
                biari_encode_symbol_final_vrdo(p_aec, 0);  // "coeff_level_minus1_band[i]", <= 32

                /* coeff_level_minus1_pos_in_band[i] */
                biari_encode_tu_vrdo(p_aec, symbol, 31, p_ctx);
            }

            /* 3.3, run, "coeff_run[i]" */
            p_aec->i_bits_to_follow += Run;

            pos += (Run + 1);   // update position
        }   // run-level loop

        p_aec->i_bits_to_follow += (num_pairs - 1);
        rank = 1;   // update rank

        /* 4, sign of coefficient */
        biari_encode_symbols_eq_prob_vrdo(p_aec, Level_sign >> 1, num_pairs);

        /* early terminate? */
        CHECK_EARLY_RETURN_RUNLEVEL(p_aec);
    }   // for (; i_cg >= 0; i_cg--) 

    /* get the number of written bits */
    org_bits = rdo_get_written_bits(p_aec) - org_bits;

#ifdef DEBUG
    if (rank == 0) {
        xavs2_log(h, XAVS2_LOG_ERROR, "no non-zero run-level chroma, p_cu: (%d, %d), level %d, cu_type %d\n",
            runlevel->p_cu_info->i_scu_x, runlevel->p_cu_info->i_scu_y, runlevel->p_cu_info->i_level, 
            runlevel->p_cu_info->i_mode);
    }
#endif
    assert(rank > 0);  // 当有非零系数时，rank的值应大于零

    /* return the number of written bits */
    return org_bits;
}

/* ---------------------------------------------------------------------------
 */
int aec_write_split_flag_vrdo(aec_t *p_aec, int i_cu_split, int i_cu_level)
{
    int org_bits = rdo_get_written_bits(p_aec);

    UNUSED_PARAMETER(i_cu_level);
    UNUSED_PARAMETER(i_cu_split);
    p_aec->i_bits_to_follow++;

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}


/* ---------------------------------------------------------------------------
 */
int write_sao_mergeflag_vrdo(aec_t *p_aec, int avail_left, int avail_up, SAOBlkParam *p_sao_param)
{
    int b_merge_left = 0;
    int b_merge_up;
    int val = 0;
    DECLARE_CONTEXT(context_t *p_ctx = p_aec->p_ctx_set->sao_merge_type_index);
    int org_bits = rdo_get_written_bits(p_aec);
    int ctx_offset = avail_left + avail_up;

    if (avail_left) {
        b_merge_left = (p_sao_param->mergeIdx == SAO_MERGE_LEFT);
        val = b_merge_left ? 1 : 0;
    }

    if (avail_up && !b_merge_left) {
        b_merge_up = (p_sao_param->mergeIdx == SAO_MERGE_ABOVE);
        val = b_merge_up ? (1 + avail_left) : 0;
    }

    if (ctx_offset == 1) {
        assert(val <= 1);
        biari_encode_symbol_vrdo(p_aec, (uint8_t)val, p_ctx + 0);
    } else if (ctx_offset == 2) {
        assert(val <= 2);
        biari_encode_symbol_vrdo(p_aec, val & 0x01, p_ctx + 1);
        if (val != 1) {
            biari_encode_symbol_vrdo(p_aec, (val >> 1) & 0x01, p_ctx + 2);
        }
    }

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}

/* ---------------------------------------------------------------------------
 */
int write_sao_mode_vrdo(aec_t *p_aec, SAOBlkParam *saoBlkParam)
{
    DECLARE_CONTEXT(context_t *p_ctx = p_aec->p_ctx_set->sao_mode);
    int org_bits = rdo_get_written_bits(p_aec);
    int sao_type = saoBlkParam->typeIdc;

    if (sao_type == SAO_TYPE_OFF) {
        biari_encode_symbol_vrdo(p_aec, 1, p_ctx);
    } else if (sao_type == SAO_TYPE_BO) {
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx);
        biari_encode_symbol_eq_prob_vrdo(p_aec, 1);
    } else {  // SAO_TYPE_EO (0~3)
        biari_encode_symbol_vrdo(p_aec, 0, p_ctx);
        biari_encode_symbol_eq_prob_vrdo(p_aec, 0);
    }

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}

/* ---------------------------------------------------------------------------
 */
static int aec_write_sao_offset_vrdo(aec_t *p_aec, int val, int offset_type)
{
    /* ---------------------------------------------------------------------------
     */
    static const int EO_OFFSET_MAP[8] = {
        3, 1, 0, 2, 4, 5, 6, 7
    };

    DECLARE_CONTEXT(context_t *p_ctx = p_aec->p_ctx_set->sao_interval_offset_abs);
    int org_bits = rdo_get_written_bits(p_aec);
    int act_sym;

    assert(offset_type != SAO_CLASS_EO_PLAIN);
    if (offset_type == SAO_CLASS_EO_FULL_VALLEY) {
        act_sym = EO_OFFSET_MAP[val + 1];
    } else if (offset_type == SAO_CLASS_EO_FULL_PEAK) {
        act_sym = EO_OFFSET_MAP[-val + 1];
    } else {
        act_sym = XAVS2_ABS(val);
    }

    if (act_sym == 0) {
        if (offset_type == SAO_CLASS_BO) {
            biari_encode_symbol_vrdo(p_aec, 1, p_ctx);
        } else {
            biari_encode_symbol_eq_prob_vrdo(p_aec, 1);
        }
    } else {
        int maxvalue = tab_saoclip[offset_type][2];
        int temp = act_sym;
        while (temp != 0) {
            if (offset_type == SAO_CLASS_BO && temp == act_sym) {
                biari_encode_symbol_vrdo(p_aec, 0, p_ctx);
            } else {
                biari_encode_symbol_eq_prob_vrdo(p_aec, 0);
            }

            temp--;
        }
        if (act_sym < maxvalue) {
            biari_encode_symbol_eq_prob_vrdo(p_aec, 1);
        }
    }

    if (offset_type == SAO_CLASS_BO && act_sym) {
        // write sign symbol
        biari_encode_symbol_eq_prob_vrdo(p_aec, (uint8_t)(val >= 0 ? 0 : 1));
    }

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}

/* ---------------------------------------------------------------------------
 */
int write_sao_offset_vrdo(aec_t *p_aec, SAOBlkParam *saoBlkParam)
{
    int rate = 0;

    assert(saoBlkParam->typeIdc != SAO_TYPE_OFF);
    if (saoBlkParam->typeIdc == SAO_TYPE_BO) {
        int bandIdxBO[4];

        bandIdxBO[0] = saoBlkParam->startBand;
        bandIdxBO[1] = bandIdxBO[0] + 1;
        bandIdxBO[2] = (saoBlkParam->startBand + saoBlkParam->deltaBand) & 31;
        bandIdxBO[3] = bandIdxBO[2] + 1;

        rate += aec_write_sao_offset_vrdo(p_aec, saoBlkParam->offset[bandIdxBO[0]], SAO_CLASS_BO);
        rate += aec_write_sao_offset_vrdo(p_aec, saoBlkParam->offset[bandIdxBO[1]], SAO_CLASS_BO);
        rate += aec_write_sao_offset_vrdo(p_aec, saoBlkParam->offset[bandIdxBO[2]], SAO_CLASS_BO);
        rate += aec_write_sao_offset_vrdo(p_aec, saoBlkParam->offset[bandIdxBO[3]], SAO_CLASS_BO);
    } else {
        assert(saoBlkParam->typeIdc >= SAO_TYPE_EO_0 && saoBlkParam->typeIdc <= SAO_TYPE_EO_45);

        rate += aec_write_sao_offset_vrdo(p_aec, saoBlkParam->offset[SAO_CLASS_EO_FULL_VALLEY], SAO_CLASS_EO_FULL_VALLEY);
        rate += aec_write_sao_offset_vrdo(p_aec, saoBlkParam->offset[SAO_CLASS_EO_HALF_VALLEY], SAO_CLASS_EO_HALF_VALLEY);
        rate += aec_write_sao_offset_vrdo(p_aec, saoBlkParam->offset[SAO_CLASS_EO_HALF_PEAK], SAO_CLASS_EO_HALF_PEAK);
        rate += aec_write_sao_offset_vrdo(p_aec, saoBlkParam->offset[SAO_CLASS_EO_FULL_PEAK], SAO_CLASS_EO_FULL_PEAK);
    }

    return rate;
}

/* ---------------------------------------------------------------------------
 */
int write_sao_type_vrdo(aec_t *p_aec, SAOBlkParam *saoBlkParam)
{
    int rate = 0;
    int val;

    assert(saoBlkParam->typeIdc != SAO_TYPE_OFF);
    if (saoBlkParam->typeIdc == SAO_TYPE_BO) {
        int exp_golomb_order = 1;

        /* start band */
        val = saoBlkParam->startBand;
        biari_encode_symbol_eq_prob_vrdo(p_aec, val & 0x01);
        biari_encode_symbol_eq_prob_vrdo(p_aec, (val >> 1) & 0x01);
        biari_encode_symbol_eq_prob_vrdo(p_aec, (val >> 2) & 0x01);
        biari_encode_symbol_eq_prob_vrdo(p_aec, (val >> 3) & 0x01);
        biari_encode_symbol_eq_prob_vrdo(p_aec, (val >> 4) & 0x01);

        /* delta band */
        assert(saoBlkParam->deltaBand >= 2);
        val = saoBlkParam->deltaBand - 2;

        while (val >= (1 << exp_golomb_order)) {
            biari_encode_symbol_eq_prob_vrdo(p_aec, 0);
            val -= (1 << exp_golomb_order);
            exp_golomb_order++;
        }
        if (exp_golomb_order == 4) {
            exp_golomb_order = 0;
        } else {
            biari_encode_symbol_eq_prob_vrdo(p_aec, 1);
        }
        while (exp_golomb_order--) { // next binary part
            biari_encode_symbol_eq_prob_vrdo(p_aec, (uint8_t)((val >> exp_golomb_order) & 1));
        }
    } else {
        assert(saoBlkParam->typeIdc >= SAO_TYPE_EO_0 && saoBlkParam->typeIdc <= SAO_TYPE_EO_45);
        val = saoBlkParam->typeIdc;

        biari_encode_symbol_eq_prob_vrdo(p_aec, val & 0x01);
        biari_encode_symbol_eq_prob_vrdo(p_aec, (val >> 1) & 0x01);
    }

    return rate;
}


/* ---------------------------------------------------------------------------
 */
int aec_write_alf_lcu_ctrl_vrdo(aec_t *p_aec, uint8_t iflag)
{
    int org_bits = rdo_get_written_bits(p_aec);
    DECLARE_CONTEXT(context_t *p_ctx =  &(p_aec->p_ctx_set->alf_cu_enable_scmodel[0][0]));
    UNUSED_PARAMETER(iflag);

    biari_encode_symbol_vrdo(p_aec, iflag, p_ctx);

    /* return the number of written bits */
    return rdo_get_written_bits(p_aec) - org_bits;
}


/* ---------------------------------------------------------------------------
 * codes cu header
 */
static
int write_cu_header_vrdo(xavs2_t *h, aec_t *p_aec, cu_t *p_cu)
{
    int rate = 0;
    int level = p_cu->cu_info.i_level;
    int mode  = p_cu->cu_info.i_mode;
    int i;

    // write bits for inter cu type
    if (h->i_type != SLICE_TYPE_I) {
        rate += aec_write_cutype_vrdo(p_aec, mode, level, p_cu->cu_info.i_cbp, h->param->enable_amp);

        if (h->i_type == SLICE_TYPE_B && (mode >= PRED_2Nx2N && mode <= PRED_nRx2N)) {
            rate += aec_write_pdir_vrdo(p_aec, mode, level, p_cu->cu_info.b8pdir[0], p_cu->cu_info.b8pdir[1]);
        } else if (h->i_type == SLICE_TYPE_F && h->param->enable_dhp && (h->i_ref > 1) &&
            ((mode >= PRED_2Nx2N && mode <= PRED_nRx2N && level > B8X8_IN_BIT) ||
             (mode == PRED_2Nx2N                       && level == B8X8_IN_BIT))) {
            rate += aec_write_pdir_dhp_vrdo(p_aec, mode, p_cu->cu_info.b8pdir[0], p_cu->cu_info.b8pdir[1]);
        }

        /* write bits for F slice skip/direct mode */
        if (IS_SKIP_MODE(mode)) {
            int b_write_spatial_skip = 0;

            if (h->i_type == SLICE_TYPE_F) {
                int weighted_skip_mode = p_cu->cu_info.directskip_wsm_idx;
                /* write weighted skip mode */
                if (h->param->enable_wsm && h->i_ref > 1) {
                    rate += aec_write_wpm_vrdo(p_aec, weighted_skip_mode, h->i_ref);
                }

                /* write bits for F-spatial-skip mode */
                b_write_spatial_skip = (h->param->enable_mhp_skip && (weighted_skip_mode == 0));
            }

            b_write_spatial_skip = b_write_spatial_skip || (SLICE_TYPE_B == h->i_type);
            /* write bits for b-direct-skip mode */
            if (b_write_spatial_skip) {
                rate += aec_write_spatial_skip_mode_vrdo(p_aec, p_cu->cu_info.directskip_mhp_idx + 1);
            }
        }
    }

    // write bits for intra modes
    if (IS_INTRA_MODE(mode)) {
        int num_of_intra_block = mode != PRED_I_2Nx2N ? 4 : 1;

        /* write "transform_split_flag" and cu_type for SDIP */
        rate += aec_write_intra_cutype_vrdo(p_aec, mode, level, p_cu->cu_info.i_tu_split, h->param->enable_sdip);

        /* write intra pred mode */
        for (i = 0; i < num_of_intra_block; i++) {
            rate += aec_write_intra_pred_mode_vrdo(p_aec, p_cu->cu_info.pred_intra_modes[i]);
        }

        if (h->param->chroma_format != CHROMA_400) {
            int i_left_cmode = DM_PRED_C;
            /* check left */
            if (p_cu->p_left_cu != NULL) {
                i_left_cmode = p_cu->p_left_cu->i_intra_mode_c;
            }
            rate += aec_write_intra_pred_cmode_vrdo(p_aec, &p_cu->cu_info, i_left_cmode);
        }
    }

    return rate;
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE int write_mvd_vrdo(aec_t *p_aec, cu_t *p_cu, int k, int bwd_flag)
{
    int curr_mvd_x = p_cu->cu_info.mvd[bwd_flag][k].x;
    int curr_mvd_y = p_cu->cu_info.mvd[bwd_flag][k].y;
    int rate;

    rate  = aec_write_mvd_vrdo(p_aec, curr_mvd_x, 0);
    rate += aec_write_mvd_vrdo(p_aec, curr_mvd_y, 1);

    return rate;
}

/* ---------------------------------------------------------------------------
 */
static
int write_cu_refs_mvds_vrdo(xavs2_t *h, aec_t *p_aec, cu_t *p_cu)
{
    int mode = p_cu->cu_info.i_mode;
    int rate   = 0;
    int k, refframe;
    int pdir;
    int dmh_mode;

    /* When CU is intra or skip mode, no need to code ref_idx and mvd */
    if (IS_INTRA_MODE(mode) || IS_SKIP_MODE(mode)) {
        return 0;
    }

    /* only one frame on each direction, no need to code ref_idx */
    // forward reference
    if (h->i_type != SLICE_TYPE_B && h->i_ref > 1) {
        for (k = 0; k < p_cu->cu_info.num_pu; k++) {
            if (p_cu->cu_info.b8pdir[k] == PDIR_FWD || p_cu->cu_info.b8pdir[k] == PDIR_DUAL) {
                refframe = p_cu->cu_info.ref_idx_1st[k];
                rate += aec_write_ref_vrdo(h, p_aec, refframe);
            }
        }
    }


    /* write backward reference indexes of this CU, no need for current AVS2 */

    /* write DMH mode, "dir_multi_hypothesis_mode" */
    if (h->i_type == SLICE_TYPE_F /*&& h->param->enable_dmh*/ 
        && p_cu->cu_info.b8pdir[0] == PDIR_FWD && p_cu->cu_info.b8pdir[1] == PDIR_FWD 
        && p_cu->cu_info.b8pdir[2] == PDIR_FWD && p_cu->cu_info.b8pdir[3] == PDIR_FWD) {
        if (!(p_cu->cu_info.i_level == B8X8_IN_BIT && p_cu->cu_info.i_mode >= PRED_2NxN && p_cu->cu_info.i_mode <= PRED_nRx2N)) {
            dmh_mode = p_cu->cu_info.dmh_mode;
            rate += aec_write_dmh_mode_vrdo(p_aec, p_cu->cu_info.i_level, dmh_mode);
        }
    }

    /* write forward MVD */
    for (k = 0; k < p_cu->cu_info.num_pu; k++) {
        pdir = p_cu->cu_info.b8pdir[k];
        if (pdir != PDIR_BWD) {
            rate += write_mvd_vrdo(p_aec, p_cu, k, 0);
        }
    }

    /* write backward MVD */
    if (h->i_type == SLICE_TYPE_B) {
        for (k = 0; k < p_cu->cu_info.num_pu; k++) {
            pdir = p_cu->cu_info.b8pdir[k];
            if (pdir == PDIR_BWD || pdir == PDIR_BID) {   //has backward vector
                rate += write_mvd_vrdo(p_aec, p_cu, k, 1);
            }
        }
    }

    return rate;
}

#if ENABLE_RATE_CONTROL_CU
/* ---------------------------------------------------------------------------
 */
int write_cu_cbp_dqp_vrdo(xavs2_t *h, aec_t *p_aec, cu_info_t *p_cu_info, int slice_index_cur_cu, int *last_dqp)
{
    int rate = aec_write_cu_cbp_vrdo(p_aec, p_cu_info, slice_index_cur_cu, h);

    if (!p_cu_info->i_cbp) {
        *last_dqp = 0;
    }

    if (p_cu_info->i_cbp != 0 && h->param->i_rc_method == XAVS2_RC_CBR_SCU) {
        rate += aec_write_dqp_vrdo(p_aec, cu_get_qp(h, p_cu_info), *last_dqp);

#if ENABLE_RATE_CONTROL_CU
        *last_dqp = p_cu_info->i_delta_qp;
#else
        *last_dqp = 0;
#endif
    }

    return rate;
}
#endif


/* ---------------------------------------------------------------------------
 */
static
int write_luma_block_coeff_vrdo(xavs2_t *h, aec_t *p_aec, cu_t *p_cu,
                                coeff_t *quant_coeff, runlevel_t *runlevel, 
                                int i_level, int i_stride_shift, int is_intra, 
                                int intra_mode, int max_bits)
{
    const int16_t(*cg_scan)[2] = NULL;
    int b_ver = p_cu->cu_info.i_tu_split == TU_SPLIT_VER;
    int b_hor = p_cu->cu_info.i_tu_split == TU_SPLIT_HOR;
    int intra_pred_class = INTRA_PRED_DC_DIAG;
    int num_cg;

    if (max_bits < 1) {
        return 1;   ///< 编码run_level至少需要1比特（sign为bypass模式）
    }

    if (b_hor) {
        cg_scan = tab_cg_scan_list_hor[i_level - 2];
    } else if (b_ver) {
        cg_scan = tab_cg_scan_list_ver[i_level - 2];
    } else {
        cg_scan = tab_cg_scan_list_nxn[i_level - 2];
    }

    // reset b_hor and b_ver
    b_hor = (is_intra && tab_intra_mode_scan_type[intra_mode] == INTRA_PRED_HOR && p_cu->cu_info.i_mode != PRED_I_2Nxn && p_cu->cu_info.i_mode != PRED_I_nx2N);
    b_ver = !b_hor;

    num_cg = 1 << (i_level + i_level - 4);     // number of CGs
    if (IS_ALG_ENABLE(OPT_BIT_EST_PSZT) && num_cg == 64 && !h->lcu.b_2nd_rdcost_pass) {  // 32x32 TB
        num_cg = 25;
    }

    /* 初始化RunLevel结构体 */
    runlevel->tab_cg_scan    = cg_scan;
    runlevel->num_cg         = num_cg;
    runlevel->i_stride_shift = i_stride_shift;
    runlevel->b_hor          = b_hor;
    runlevel->quant_coeff    = quant_coeff;
    runlevel->p_cu_info      = &p_cu->cu_info;

    /* return bit rate */
    if (IS_INTRA_MODE(p_cu->cu_info.i_mode)) {
        assert(intra_mode < NUM_INTRA_MODE);
        intra_pred_class = tab_intra_mode_scan_type[intra_mode];
    }
    return aec_write_run_level_luma_vrdo(p_aec, intra_pred_class == INTRA_PRED_DC_DIAG, runlevel, h, max_bits);
}

/* ---------------------------------------------------------------------------
 */
static
int write_chroma_block_coeff_vrdo(xavs2_t *h, aec_t *p_aec, cu_t *p_cu, coeff_t *quant_coeff, runlevel_t *runlevel, int i_level, int max_bits)
{
    int num_cg = 1 << (i_level + i_level - 4);

    if (max_bits < 1) {
        return 1;   ///< 编码run_level至少需要1比特（sign为bypass模式）
    }

    if (IS_ALG_ENABLE(OPT_BIT_EST_PSZT) && num_cg == 64 && !h->lcu.b_2nd_rdcost_pass) {  // 32x32 TB
        num_cg = 25;
    }

    /* 初始化RunLevel结构体 */
    runlevel->tab_cg_scan    = tab_cg_scan_list_nxn[i_level - 2];
    runlevel->num_cg         = num_cg;
    runlevel->i_stride_shift = i_level;
    runlevel->b_hor          = 0;
    runlevel->quant_coeff    = quant_coeff;
    runlevel->p_cu_info      = &p_cu->cu_info;

    return aec_write_run_level_chroma_vrdo(p_aec, runlevel, h, max_bits);
}

/**
 * ===========================================================================
 * function handler
 * ===========================================================================
 */
binary_t gf_aec_vrdo = {
    /* syntax elements */
    .write_intra_pred_mode     = aec_write_intra_pred_mode_vrdo,
    .write_ctu_split_flag      = aec_write_split_flag_vrdo,
    .est_cu_header             = write_cu_header_vrdo,
    .est_cu_refs_mvds          = write_cu_refs_mvds_vrdo,
    .est_luma_block_coeff      = write_luma_block_coeff_vrdo,
    .est_chroma_block_coeff    = write_chroma_block_coeff_vrdo,
    
#if ENABLE_RATE_CONTROL_CU
    .write_cu_cbp_dqp          = write_cu_cbp_dqp_vrdo,
#else
    .write_cu_cbp              = aec_write_cu_cbp_vrdo,
#endif

    .write_sao_mergeflag       = write_sao_mergeflag_vrdo,
    .write_sao_mode            = write_sao_mode_vrdo,
    .write_sao_offset          = write_sao_offset_vrdo,
    .write_sao_type            = write_sao_type_vrdo,
    .write_alf_lcu_ctrl        = aec_write_alf_lcu_ctrl_vrdo,
};
