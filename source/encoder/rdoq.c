/*
 * rdoq.c
 *
 * Description of this file:
 *    RDOQ functions definition of the xavs2 library
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
#include "transform.h"
#include "rdoq.h"
#include "wquant.h"
#include "aec.h"
#include "cudata.h"


/**
 * ===========================================================================
 * type defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
enum node_type_e {
    LAST_POS,
    LAST_RUN,
    RUN_LEVEL_PAIR
};


/* ---------------------------------------------------------------------------
 * pair cost for RDOQ
 */
typedef struct pair_cost_t {
    rdcost_t    levelCost;
    rdcost_t    runCost;
    rdcost_t    uncodedCost;
    int16_t     posBlockX;
    int16_t     posBlockY;
    int32_t     scanPos;
} pair_cost_t;


/* ---------------------------------------------------------------------------
 * cost state for RDOQ
 */
typedef struct cost_state_t {
    pair_cost_t pairCost[16 + 1];    // OLD: +1
    rdcost_t    sigCGFlagCost;
    rdcost_t    sigCGFlagCost0;
    rdcost_t    lastRunCost;
    int         pairNum;
} cost_state_t;


/* ---------------------------------------------------------------------------
 * level info for RDOQ
 */
typedef struct level_info_t {
    int         pos_scan;             /* position in transform block zig-zag scan order */
    int         pos_xy;               /* position in block */
    int         num_level;            /* number of candidate levels */
    coeff_t     level[3];             /* candidate levels */
    coeff_t     coeff;                /* coefficient before quantization */
    double      errLevel[3];          /* quantization errors of each candidate */
} level_info_t;


/* ---------------------------------------------------------------------------
 */
typedef struct node_t node_t;
struct node_t {
    node_t       *prev;
    node_t       *next;
    level_info_t *level_info;
    int           attrib;             // node_type_e, 0: last pos; 1: last run; 2: (Run, Level) pair
    int           level;
    int           run;
    int           pos;                // scan position in CG， CoeffPosInCG = ScanCoeffInCG[LastCoeffPosX][LastCoefPosY]
    rdcost_t      cost;
};


/* ---------------------------------------------------------------------------
 */
typedef struct node_list_t {
    ALIGN16(node_t nodeBuf[16 + 4]);
    node_t     *head;
    node_t     *tail;
    int         i_size;               /* number of nodes in the list */
} node_list_t;


/**
 * ===========================================================================
 * local & global variable defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * AC ENGINE PARAMETERS
 */
static const int16_t tab_LPSbits[256] = {
    2184,2184,1928,1779,1673,1591,1525,1468,1419,1376,1338,1303,1272,1243,1216,1191,
    1167,1145,1125,1105,1087,1069,1053,1037,1022,1007, 993, 980, 967, 954, 942, 930,
    919, 908, 898, 888, 878, 868, 859, 850, 841, 832, 824, 816, 808, 800, 792, 785,
    777, 770, 763, 756, 750, 743, 737, 730, 724, 718, 712, 707, 701, 695, 690, 684,
    679, 674, 669, 663, 658, 654, 649, 644, 639, 635, 630, 626, 621, 617, 613, 608,
    604, 600, 596, 592, 588, 584, 580, 577, 573, 569, 566, 562, 558, 555, 551, 548,
    545, 541, 538, 535, 531, 528, 525, 522, 519, 516, 513, 510, 507, 504, 501, 498,
    495, 492, 490, 487, 484, 482, 479, 476, 474, 471, 468, 466, 463, 461, 458, 456,
    454, 451, 449, 446, 444, 442, 439, 437, 435, 433, 430, 428, 426, 424, 422, 420,
    418, 415, 413, 411, 409, 407, 405, 403, 401, 399, 397, 395, 394, 392, 390, 388,
    386, 384, 382, 381, 379, 377, 375, 373, 372, 370, 368, 367, 365, 363, 362, 360,
    358, 357, 355, 353, 352, 350, 349, 347, 346, 344, 342, 341, 339, 338, 336, 335,
    333, 332, 331, 329, 328, 326, 325, 323, 322, 321, 319, 318, 317, 315, 314, 313,
    311, 310, 309, 307, 306, 305, 303, 302, 301, 300, 298, 297, 296, 295, 293, 292,
    291, 290, 289, 287, 286, 285, 284, 283, 282, 281, 279, 278, 277, 276, 275, 274,
    273, 272, 271, 269, 268, 267, 266, 265, 264, 263, 262, 261, 260, 259, 258, 257
};

extern const int tab_intra_mode_scan_type[NUM_INTRA_MODE];

static const int8_t tab_rank[6] = {0, 1, 2, 3, 3, 4/*, 4 ...*/};

/**
 * ===========================================================================
 * local function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void list_init(node_list_t *list)
{
    list->head = NULL;
    list->tail = NULL;
    list->i_size = 0;
}

/* ---------------------------------------------------------------------------
 * create a new node and append it to list
 */
static ALWAYS_INLINE node_t *
create_and_append_node(node_list_t *list, level_info_t *level_info, int attrib, int pos)
{
    node_t *p_node = list->nodeBuf + list->i_size;

    /* 1, create a new node */
    p_node->attrib     = attrib;
    p_node->level_info = level_info;
    p_node->run        = 0;
    p_node->pos        = pos;
    p_node->prev       = NULL;
    p_node->next       = NULL;

    list->i_size++;

    /* 2, append the new node to list */
    if (list->head == NULL) {
        /* for empty list */
        list->head = p_node;
    } else {
        /* append tail */
        list->tail->next = p_node;
        p_node->prev = list->tail;
    }
    list->tail = p_node;

    return p_node;
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void remove_node(node_t *p_node)
{
    p_node->prev->next = p_node->next;
    if (p_node->next != NULL) {
        p_node->next->prev = p_node->prev;
    }
}

/* ---------------------------------------------------------------------------
 * actually arithmetic encoding of one binary symbol by using the probability
 * estimate of its associated context model
 */
static ALWAYS_INLINE
int biari_encode_symbol_est(uint8_t symbol, context_t *ctx)
{
    const int lg_pmps = ctx->LG_PMPS >> 2;

    return (symbol == ctx->MPS) ? lg_pmps : tab_LPSbits[lg_pmps];
}

/* ---------------------------------------------------------------------------
 * actually arithmetic encoding of one binary symbol by using the probability
 * estimate of its associated context model
 * firstly encode 0 and then encode 1
 */
static ALWAYS_INLINE
int biari_encode_symbol_est_0_then_1(context_t *ctx)
{
    const int lg_pmps = ctx->LG_PMPS >> 2;
    return lg_pmps + tab_LPSbits[lg_pmps];
}

/* ---------------------------------------------------------------------------
 * 用于连续编码len个symbol值，由编码单个符号可知上下文不更新，因而会简化成单个符号的整数倍
 */
static ALWAYS_INLINE int
biari_encode_symbols_est(uint8_t symbol, int len, context_t *ctx)
{
    return len * biari_encode_symbol_est(symbol, ctx);
}

/* ---------------------------------------------------------------------------
 */
#define biari_encode_symbol_eq_prob_est(p)  (256)

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE int
biari_encode_symbol_final_est(uint8_t symbol)
{
    // context_t ctx = { 1 << LG_PMPS_SHIFTNO, 0, 0 };
    // return biari_encode_symbol_est(symbol, &ctx);
    /* symbol != MPS ? tab_LPSbits[lg_pmps] : lg_pmps */
    return symbol ? tab_LPSbits[1] : 1;
}

/* ---------------------------------------------------------------------------
 * set all coefficients to zeros
 */
static ALWAYS_INLINE void
rdoq_memset_zero_coeffs(coeff_t *ncur_blk, int pos_start, int pos_end)
{
    memset(ncur_blk + pos_start, 0, (pos_end - pos_start) * sizeof(coeff_t));
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE int
get_block_size_id_wq(int i_tu_split, int i_tu_level, int b_luma)
{
    int wqm_size_id = 0;

    if (b_luma && i_tu_split == TU_SPLIT_HOR) {
        if (i_tu_level == B8X8_IN_BIT) {
            wqm_size_id = 2;
        } else if (i_tu_level == B16X16_IN_BIT || i_tu_level == B32X32_IN_BIT) {
            wqm_size_id = 3;
        }
    } else if (b_luma && i_tu_split == TU_SPLIT_VER) {
        if (i_tu_level == B8X8_IN_BIT) {
            wqm_size_id = 2;
        } else if (i_tu_level == B16X16_IN_BIT || i_tu_level == B32X32_IN_BIT) {
            wqm_size_id = 3;
        }
    } else {
        wqm_size_id = i_tu_level - B4X4_IN_BIT;
    }

    return wqm_size_id;
}

/* ---------------------------------------------------------------------------
 */
static int est_rate_last_cg_pos(rdoq_t *p_rdoq, int iCG, int *cg_x, int *cg_y)
{
    int rate = 0;

    if (p_rdoq->i_tu_level == B4X4_IN_BIT) {
        *cg_x = 0;
        *cg_y = 0;
    } else {
        context_t *p_ctx = p_rdoq->p_ctx_last_cg;
        int i_cg_last_x = p_rdoq->p_scan_cg[iCG][0];
        int i_cg_last_y = p_rdoq->p_scan_cg[iCG][1];

        *cg_x = i_cg_last_x;
        *cg_y = i_cg_last_y;

        if (p_rdoq->i_tu_level == B8X8_IN_BIT) {
            switch (iCG) {
            case 0:
                rate += biari_encode_symbol_est(1, p_ctx + 0);
                break;
            case 1:
                rate += biari_encode_symbol_est(0, p_ctx + 0);
                rate += biari_encode_symbol_est(1, p_ctx + 1);
                break;
            case 2:
                rate += biari_encode_symbol_est(0, p_ctx + 0);
                rate += biari_encode_symbol_est(0, p_ctx + 1);
                rate += biari_encode_symbol_est(1, p_ctx + 2);
                break;
            default:  // case 3:
                rate += biari_encode_symbol_est(0, p_ctx + 0);
                rate += biari_encode_symbol_est(0, p_ctx + 1);
                rate += biari_encode_symbol_est(0, p_ctx + 2);
                break;
            }
        } else {
            const int b_luma = p_rdoq->b_luma;
            int num_cg_x = p_rdoq->num_cg_x - 1; // (number - 1) of CG in x direction
            int num_cg_y = p_rdoq->num_cg_y - 1; // (number - 1) of CG in y direction

            if (b_luma && p_rdoq->b_dc_diag) {
                XAVS2_SWAP(i_cg_last_x, i_cg_last_y);
            }

            if (i_cg_last_x == 0 && i_cg_last_y == 0) {
                rate += biari_encode_symbol_est(0, p_ctx + 3);  /* last_cg0_flag */
            } else {
                rate += biari_encode_symbol_est(1, p_ctx + 3);  /* last_cg0_flag */
                /* last_cg_x */
                rate += biari_encode_symbols_est(0, i_cg_last_x, p_ctx + 4);
                if (i_cg_last_x < num_cg_x) {
                    rate += biari_encode_symbol_est(1, p_ctx + 4);
                }
                /* last_cg_y */
                rate += biari_encode_symbols_est(0, i_cg_last_y - (i_cg_last_x == 0), p_ctx + 5);
                if (i_cg_last_y < num_cg_y) {
                    rate += biari_encode_symbol_est(1, p_ctx + 5);
                }
            }
        }
    }

    return rate;
}

/* ---------------------------------------------------------------------------
 * estimate rate of coding "significant_cg_flag"
 */
static ALWAYS_INLINE int
est_rate_nonzero_cg_flag(rdoq_t *p_rdoq, int sig_cg_flag, int ctx)
{
    return biari_encode_symbol_est((uint8_t)sig_cg_flag, p_rdoq->p_ctx_sign_cg + ctx);
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE int
est_rate_last_coeff_pos(rdoq_t *p_rdoq, int last_coeff_pos_x, int last_coeff_pos_y, int isLastCG, int cg_x, int cg_y)
{
    context_t *p_ctx = p_rdoq->p_ctx_last_pos;
    const int b_dc_diag = p_rdoq->b_dc_diag;
    const int b_one_cg = (p_rdoq->i_tu_level == B4X4_IN_BIT);
    const int b_luma = p_rdoq->b_luma;
    int rate = 0;
    int offset = 0;

    if (!isLastCG) {
        last_coeff_pos_x = 3 - last_coeff_pos_x;
        if (b_dc_diag) {
            last_coeff_pos_y = 3 - last_coeff_pos_y;
        }
    }

    if (cg_x == 0 && cg_y > 0 && b_dc_diag) {
        XAVS2_SWAP(last_coeff_pos_x, last_coeff_pos_y);
    }

    /* AVS2-P2国标: 8.3.3.2.14   确定last_coeff_pos_x 或last_coeff_pos_y 的ctxIdxInc */
    if (b_luma == 0) {                    // 色度分量共占用12个上下文
        offset = b_one_cg ? 0 : 4 + isLastCG * 4;
    } else if (b_one_cg) {                // Log2TransformSize 为 2，占用8个上下文
        offset = 40 + (b_dc_diag) * 4;
    } else if (cg_x != 0 && cg_y != 0) {  // cg_x 和 cg_y 均不为零，占用8个上下文
        offset = 32 + isLastCG * 4;
    } else {                              // 其他亮度位置占用40个上下文
        offset = (4 * isLastCG + 2 * (cg_x == 0 && cg_y == 0) + (b_dc_diag)) * 4;
    }

    p_ctx += offset;
    switch (last_coeff_pos_x) {
    case 0:
        rate += biari_encode_symbol_est(1, p_ctx + 0);
        break;
    case 1:
        rate += biari_encode_symbol_est(0, p_ctx + 0);
        rate += biari_encode_symbol_est(1, p_ctx + 1);
        break;
    case 2:
        rate += biari_encode_symbol_est(0, p_ctx + 0);
        rate += biari_encode_symbol_est_0_then_1(p_ctx + 1);
        break;
    default: // case 3:
        rate += biari_encode_symbol_est(0, p_ctx + 0);
        rate += biari_encode_symbol_est(0, p_ctx + 1) << 1;
        break;
    }

    p_ctx += 2;
    switch (last_coeff_pos_y) {
    case 0:
        rate += biari_encode_symbol_est(1, p_ctx + 0);
        break;
    case 1:
        rate += biari_encode_symbol_est(0, p_ctx + 0);
        rate += biari_encode_symbol_est(1, p_ctx + 1);
        break;
    case 2:
        rate += biari_encode_symbol_est(0, p_ctx + 0);
        rate += biari_encode_symbol_est_0_then_1(p_ctx + 1);
        break;
    default: // case 3:
        rate += biari_encode_symbol_est(0, p_ctx + 0);
        rate += biari_encode_symbol_est(0, p_ctx + 1) << 1;
        break;
    }

    return rate;
}

/* ---------------------------------------------------------------------------
 */
static int
est_rate_level(context_t *p_ctx, int rank, int absLevel, int pairsInCG, int iCG, int pos, int b_luma)
{
    int rate = 0;
    int symbol = absLevel - 1;

    if (symbol > 31) {
        int exp_golomb_order = 0;

        symbol -= 32;
        rate += biari_encode_symbol_final_est(1);
        while (symbol >= (1 << exp_golomb_order)) {
            symbol -= (1 << exp_golomb_order);
            exp_golomb_order++;
        }
        rate += (2 * exp_golomb_order + 1) * biari_encode_symbol_eq_prob_est(symbol);
    } else {
        int pairsInCGIdx = XAVS2_MIN(2, ((pairsInCG + 1) >> 1));
        p_ctx += 10 * (iCG == 0 && pos > 12) + XAVS2_MIN(rank, pairsInCGIdx + 2) + ((5 * pairsInCGIdx) >> 1);
        // chroma
        if (!b_luma) {
            p_ctx += 20;
        }

        rate += biari_encode_symbol_final_est(0);
        rate += biari_encode_symbols_est(0, symbol, p_ctx);
        if (symbol < 31) {
            rate += biari_encode_symbol_est(1, p_ctx);
        }
    }

    return rate;
}

/* ---------------------------------------------------------------------------
 */
static int est_rate_run(rdoq_t *p_rdoq, context_t *p_ctx, int run, int pos, int iCG, int remaining_pos)
{
    static const int8_t tab_run_rate_ctx_offset[16] = {
        2, 2, 2, 2, 2,          // pos <= 4
        1, 1, 1, 1, 1, 1, 1,    // pos > 4 && pos <= 11
        0, 0, 0, 0              // pos > 11
    };
    const int b_luma = p_rdoq->b_luma;
    const int b_dc_diag = p_rdoq->b_dc_diag;
    int symbol = run;
    int offset = 0;
    int rate = 0;
    int pos2 = 15 - pos - 1;
    int off2 = (p_rdoq->i_tu_level == B4X4_IN_BIT) ? 0 : (b_luma ? 4 : 3);
    int moddiv;
    int y_div2;

    if (pos < 15) {
        if (b_luma) {
            y_div2 = (tab_scan_4x4[pos2][1] + 1) >> 1;
            moddiv = b_dc_diag ? tab_run_rate_ctx_offset[pos] : y_div2;
            offset = ((iCG == 0) ? (pos == 14 ? 0 : (1 + moddiv)) : (4 + moddiv)) + off2;  // 0,...,9 （10个）
        } else {
            moddiv = (pos <= 9);
            offset = ((iCG == 0) ? (pos == 14 ? 0 : (1 + moddiv)) : 2) + off2;   // 0,1,2, +4  (8个)
        }
    }

    if (iCG == 0) {
        if (b_luma) {
            while (symbol-- > 0) {
                assert(offset >= 0 && offset < NUM_MAP_CTX);
                rate += biari_encode_symbol_est(0, p_ctx + offset);
                pos++;
                if (--pos2 >= 0) {
                    y_div2 = (tab_scan_4x4[pos2][1] + 1) >> 1;
                    moddiv = b_dc_diag ? tab_run_rate_ctx_offset[pos] : y_div2;
                    offset = off2 + (pos == 14 ? 0 : (1 + moddiv));
                }
            }
        } else {
            while (symbol-- > 0) {
                assert(offset >= 0 && offset < NUM_MAP_CTX);
                rate += biari_encode_symbol_est(0, p_ctx + offset);
                pos++;
                if (--pos2 >= 0) {
                    moddiv = (pos <= 9);
                    offset = off2 + (pos == 14 ? 0 : (1 + moddiv));
                }
            }
        }
    } else {
        if (b_luma) {
            while (symbol-- > 0) {
                assert(offset >= 0 && offset < NUM_MAP_CTX);
                rate += biari_encode_symbol_est(0, p_ctx + offset);
                pos++;
                if (--pos2 >= 0) {
                    y_div2 = (tab_scan_4x4[pos2][1] + 1) >> 1;
                    moddiv = b_dc_diag ? tab_run_rate_ctx_offset[pos] : y_div2;
                    offset = off2 + 4 + moddiv;
                }
            }
        } else {
            offset = off2 + 2;
            assert(offset >= 0 && offset < NUM_MAP_CTX);
            rate += symbol * biari_encode_symbol_est(0, p_ctx + offset);
        }
    }

    if (run < remaining_pos) {
        assert(offset >= 0 && offset < NUM_MAP_CTX);
        rate += biari_encode_symbol_est(1, p_ctx + offset);
    }

    return rate;
}

/* ---------------------------------------------------------------------------
 * get number of non-zero CGs
 */
static ALWAYS_INLINE int
rdoq_get_last_cg_pos(coeff_t *ncur_blk, int num_coeff, const int thres_lower_int)
{
    int idx_coeff;

    for (idx_coeff = num_coeff - 1; idx_coeff >= 0; idx_coeff--) {
        if (ncur_blk[idx_coeff] <= thres_lower_int) {
            ncur_blk[idx_coeff] = 0;
        } else {
            break;              /* last none zero coeff is found */
        }
    }

    return ((idx_coeff + 16) >> 4);
}

#if ENABLE_WQUANT
/* ---------------------------------------------------------------------------
 * rdoq中预先计算一个系数量化后可能取的值（加权量化）
 */
static int rdoq_est_coeff_level_wq(xavs2_t *h, level_info_t *p_level_info,
                                   wq_data_t *wq, int wqm_size_id, int wqm_size, int xx, int yy,
                                   coeff_t coeff, int qp, int shift_bit)
{
    const double f_err_level_mult = 256.0 / (1 << (shift_bit * 2));
    const int thres_lower_int = (int)((16384 << shift_bit) / (double)(tab_Q_TAB[qp]));
    const int scale = tab_IQ_TAB[qp];
    const int shift = tab_IQ_SHIFT[qp] - shift_bit;
    int wqm_shift   = (h->param->PicWQDataIndex == 1) ? 3 : 0;
    int wqm_coef    = 1;
    int rec, err;
    int level;
    int b_lower;
    int stride;

    if ((wqm_size_id == 0) || (wqm_size_id == 1)) {
        stride   = wqm_size;
        wqm_coef = wq->cur_wq_matrix[wqm_size_id][(yy & (stride - 1)) * stride + (xx & (stride - 1))];
    } else if (wqm_size_id == 2) {
        stride   = wqm_size >> 1;
        wqm_coef = wq->cur_wq_matrix[wqm_size_id][((yy >> 1) & (stride - 1)) * stride + ((xx >> 1) & (stride - 1))];
    } else if (wqm_size_id == 3) {
        stride   = wqm_size >> 2;
        wqm_coef = wq->cur_wq_matrix[wqm_size_id][((yy >> 2) & (stride - 1)) * stride + ((xx >> 2) & (stride - 1))];
    }

    level   = (int)(coeff * tab_Q_TAB[qp] >> (15 + shift_bit));
    level   = XAVS2_CLIP3((-((1 << 18) / wqm_coef)), (((1 << 18) / wqm_coef) - 1), level);
    rec     = (((((coeff * wqm_coef) >> 3) * scale) >> 4) + (1 << (shift - 1))) >> shift;
    b_lower = (coeff - rec) <= thres_lower_int;

#define GET_ERROR_LEVEL_WQ(i, cur_level) \
    {\
        rec = ((((((int)(cur_level) * wqm_coef) >> wqm_shift) * scale) >> 4) + (1 << (shift - 1))) >> shift;\
        err = coeff - rec;\
        p_level_info->errLevel[i] = err * err * f_err_level_mult;\
        p_level_info->level[i]    = (coeff_t)(cur_level);\
    }

    /* 1, generate levels of one coefficient [xx, yy] */
    p_level_info->coeff = coeff;
    GET_ERROR_LEVEL_WQ(0, 0);
    if (level == 0) {
        if (b_lower) {
            p_level_info->num_level = 1;
            return 0;
        } else {
            GET_ERROR_LEVEL_WQ(1, 1);
            p_level_info->num_level = 2;
            return 1;
        }
    } else {
        if (b_lower) {
            GET_ERROR_LEVEL_WQ(1, level);
            p_level_info->num_level = 2;
        } else {
            GET_ERROR_LEVEL_WQ(1, level);
            GET_ERROR_LEVEL_WQ(2, level + 1);
            p_level_info->num_level = 3;
        }

        return level;
    }

#undef GET_ERROR_LEVEL_WQ
}
#endif

/* ---------------------------------------------------------------------------
 * rdoq中预先计算一个系数量化后可能取的值
 */
static ALWAYS_INLINE int
rdoq_est_coeff_level(level_info_t *p_level_info, coeff_t coeff, int qp, int shift_bit,
                     const double f_err_level_mult, const int thres_lower_int)
{
    const int scale = tab_IQ_TAB[qp];
    const int shift = tab_IQ_SHIFT[qp] - shift_bit;
    const int shift_offset = 1 << (shift - 1);
    int rec, err;
    int level;
    int b_lower;

#define GET_ERROR_LEVEL(i, cur_level) \
    {\
        rec = ((cur_level) * scale + shift_offset) >> shift;\
        err = coeff - rec;\
        p_level_info->errLevel[i] = err * err * f_err_level_mult;\
        p_level_info->level[i] = (coeff_t)(cur_level);\
    }

    level   = (int)(coeff * tab_Q_TAB[qp] >> (15 + shift_bit));
    rec     = (level * scale + shift_offset) >> shift;
    b_lower = (coeff - rec) <= thres_lower_int;

    p_level_info->coeff = coeff;
    GET_ERROR_LEVEL(0, 0);
    if (level == 0) {
        if (b_lower) {
            p_level_info->num_level = 1;
            return 0;
        } else {
            GET_ERROR_LEVEL(1, 1);
            p_level_info->num_level = 2;
            return 1;
        }
    } else {
        if (b_lower) {
            GET_ERROR_LEVEL(1, level);
            p_level_info->num_level = 2;
        } else {
            GET_ERROR_LEVEL(1, level);
            GET_ERROR_LEVEL(2, level + 1);
            p_level_info->num_level = 3;
        }

        return level;
    }

#undef GET_ERROR_LEVEL
}

/* ---------------------------------------------------------------------------
 * 计算指定区间内的系数的累加和
 */
static ALWAYS_INLINE int
rdoq_get_sum_abs_coeff(level_info_t *p_level_info, const coeff_t *p_ncoeff, int pos_start, int pos_end)
{
    int sum = 0;
    int pos;

    for (pos = pos_start; pos <= pos_end; pos++) {
        sum += p_ncoeff[p_level_info[pos].pos_scan];
    }

    return sum;
}

/* ---------------------------------------------------------------------------
 * 迭代一个CG内的系数的各个level，求当前CG的最优
 */
static int rdoq_est_cg(xavs2_t *h, rdoq_t *p_rdoq, level_info_t *p_level_info, cost_state_t *p_cost_stat, node_t *node,
                       coeff_t *ncur_blk, int8_t *p_sig_cg_flag, int iCG, int rank_pre)
{
    static const int T_Chr[5] = {0, 1, 2, 4, INT_MAX};
    pair_cost_t *p_pair_cost = &p_cost_stat->pairCost[0];
    context_t(*p_ctx_primary)[NUM_MAP_CTX] = p_rdoq->p_ctx_primary;
    context_t *p_ctx;
    const rdcost_t lambda_rdoq = h->f_lambda_rdoq;
    rdcost_t lagrUncoded = 0;
    rdcost_t lagrAcc     = 0;
    int pairsInCG = 0;
    int isSigCG   = 0;
    int isLastCG  = 0;
    int level_max = T_Chr[rank_pre];
    int rank = rank_pre;
    int CGx  = p_rdoq->p_scan_cg[iCG][0];
    int CGy  = p_rdoq->p_scan_cg[iCG][1];
    int w_shift_x = p_rdoq->bit_size_shift_x;
    int w_shift_y = p_rdoq->bit_size_shift_y;

    // rdoq for this CG
    if (node != NULL && node->attrib == LAST_POS) {
        isLastCG = 1;
        node = node->next;
    }

    while (node != NULL) {
        if (node->attrib == LAST_RUN) { // this is not the last CG
            if (node->run != 16) {
                int scan_pos = tab_1d_scan_4x4[15 - node->run];
                p_cost_stat->lastRunCost = lambda_rdoq *
                                           est_rate_last_coeff_pos(p_rdoq, scan_pos & 3, (scan_pos >> 2), 0, CGx, CGy);
            } else {
                p_cost_stat->lastRunCost = 0;
            }
            lagrAcc += p_cost_stat->lastRunCost;
        } else { // a (level, run) pair
            // try level, level-1 first, then compare to level=0 case
            int levelNo;
            int absSum5;
            int absLevel;
            int rateRunMerged   = 0;
            int best_state      = 0;
            rdcost_t minlagr    = MAX_COST;
            rdcost_t lagrDelta  = 0;
            rdcost_t lagrDelta0 = 0;
            rdcost_t lagr;
            int xx_yy;

            isSigCG = 1;
            for (levelNo = 1; levelNo < node->level_info->num_level; levelNo++) {
                int rateLevel;
                int rateRunCurr;
                int rateRunPrev;
                int pos_end = XAVS2_MIN(node->pos + 6, 15);

                // rate: Level
                absLevel  = node->level_info->level[levelNo];
                p_ctx     = p_rdoq->p_ctx_coeff_level;
                rateLevel = est_rate_level(p_ctx, rank, absLevel, pairsInCG, iCG, 15 - node->pos, p_rdoq->b_luma);

                // rate: Sign
                rateLevel += biari_encode_symbol_eq_prob_est(absLevel < 0);

                // rate: Run[i]
                absSum5     = absLevel + rdoq_get_sum_abs_coeff(p_level_info, ncur_blk, node->pos + 1, pos_end);
                p_ctx       = p_ctx_primary[XAVS2_MIN((absSum5 >> 1), 2)];
                rateRunCurr = est_rate_run(p_rdoq, p_ctx, node->run, 15 - node->pos, iCG, node->pos);

                // rate: Run[i+1]
                // node->prev always exists
                if (node->prev->attrib == LAST_POS) {
                    rateRunPrev = 0;
                } else if (node->prev->attrib == LAST_RUN) {
                    if (node->prev->run != 16) {
                        int scan_pos = tab_1d_scan_4x4[15 - node->prev->run];
                        rateRunPrev = est_rate_last_coeff_pos(p_rdoq, scan_pos & 3, scan_pos >> 2, 0, CGx, CGy);
                    } else {
                        rateRunPrev = 0;
                    }
                    p_cost_stat->lastRunCost = lambda_rdoq * rateRunPrev;
                } else { // RUN_LEVEL_PAIR
                    pos_end     = XAVS2_MIN(node->prev->pos + 6, 15);
                    absSum5     = rdoq_get_sum_abs_coeff(p_level_info, ncur_blk, node->prev->pos, pos_end);
                    p_ctx       = p_ctx_primary[XAVS2_MIN((absSum5 >> 1), 2)];
                    rateRunPrev = est_rate_run(p_rdoq, p_ctx, node->prev->run, 15 - node->prev->pos, iCG, node->prev->pos);
                }

                // cost for the current (Level, Run) pair
                p_pair_cost->levelCost = (rdcost_t)(node->level_info->errLevel[levelNo] + lambda_rdoq * rateLevel);
                p_pair_cost->runCost   = lambda_rdoq * rateRunCurr;
                p_pair_cost->scanPos   = node->level_info->pos_scan;

                // calculate cost: distLevel[i] + rateLevel[i] + rateRun[i] + rateRun[i+1]
                lagr = (rdcost_t)(node->level_info->errLevel[levelNo] + lambda_rdoq * (rateLevel + rateRunCurr + rateRunPrev));
                if (lagr < minlagr) {
                    minlagr = lagr;
                    best_state = levelNo;
                }

                lagrDelta = lambda_rdoq * rateRunPrev;
            }
            p_pair_cost->uncodedCost = (rdcost_t)(node->level_info->errLevel[0]);

            // compare cost of level or level-1 with uncoded case (level=0)
            // Run[i]
            if (node->prev->attrib != LAST_POS && (node->prev->attrib != LAST_RUN || node->next != NULL)) {
                if (node->prev->attrib == RUN_LEVEL_PAIR) {
                    int pos_start = node->prev->pos;
                    int pos_end   = XAVS2_MIN(pos_start + 6, 15);

                    absSum5       = rdoq_get_sum_abs_coeff(p_level_info, ncur_blk, pos_start, pos_end);
                    p_ctx         = p_ctx_primary[XAVS2_MIN((absSum5 >> 1), 2)];
                    rateRunMerged = est_rate_run(p_rdoq, p_ctx, node->prev->run + 1 + node->run, 15 - pos_start, iCG, pos_start);
                    lagrDelta0    = p_cost_stat->pairCost[pairsInCG - 1].runCost;
                } else { /*if (node->next != NULL)*/     // LAST_RUN
                    // only try 0 when there's more than 1 pair in the CG
                    lagrDelta0    = p_cost_stat->lastRunCost;
                    if (node->prev->run != 16) {
                        int scan_pos = tab_1d_scan_4x4[15 - (node->prev->run + 1 + node->run)];
                        rateRunMerged = est_rate_last_coeff_pos(p_rdoq, scan_pos & 3, scan_pos >> 2, 0, CGx, CGy);
                    } else {
                        rateRunMerged = 0;
                    }
                }

                // calculate cost: distLevel[i][0] + rate(Run[i] + Run[i+1] + 1)
                lagr = (rdcost_t)(node->level_info->errLevel[0] + lambda_rdoq * rateRunMerged);

                if (lagr < minlagr) {
                    minlagr    = lagr;
                    lagrDelta  = lagrDelta0;
                    best_state = 0;
                }
            }

            // set SDQ results
            xx_yy = p_level_info[node->pos].pos_xy;
            absLevel = node->level = node->level_info->level[best_state];
            ncur_blk[p_level_info[node->pos].pos_scan] = (coeff_t)absLevel;

            lagrAcc     += minlagr - lagrDelta;
            lagrUncoded += (rdcost_t)(node->level_info->errLevel[0]);

            p_pair_cost->posBlockX = (int16_t)((xx_yy >> w_shift_x) & 0x3);
            p_pair_cost->posBlockY = (int16_t)((xx_yy >> w_shift_y) & 0x3);

            p_pair_cost->scanPos = p_level_info[node->pos].pos_scan;
            if (best_state == 0) {
                // adjust the run of the previous node and remove the current node
                node->prev->run += node->run + 1;
                if (node->prev->attrib == LAST_RUN) {
                    p_cost_stat->lastRunCost = lambda_rdoq * rateRunMerged;
                } else {
                    p_cost_stat->pairCost[pairsInCG - 1].runCost = lambda_rdoq * rateRunMerged;
                }
                remove_node(node);
            } else {
                pairsInCG++;
                p_pair_cost++;
            }

            // update rank
            level_max = XAVS2_MAX(level_max, absLevel);
            rank = tab_rank[XAVS2_MIN(5, level_max)];
        }

        node = node->next;
    }

    if (!isLastCG) {
        int sig_cg_ctx = p_rdoq->b_luma && (iCG != 0);

        if (isSigCG) {
            lagrAcc     += (p_cost_stat->sigCGFlagCost  = lambda_rdoq * est_rate_nonzero_cg_flag(p_rdoq, 1, sig_cg_ctx));
            lagrUncoded += (p_cost_stat->sigCGFlagCost0 = lambda_rdoq * est_rate_nonzero_cg_flag(p_rdoq, 0, sig_cg_ctx));

            // try to turn CG to all-zero here. don't do this to last CG
            if (lagrUncoded < lagrAcc) {
                rdoq_memset_zero_coeffs(ncur_blk, iCG << 4, (iCG + 1) << 4);
                p_sig_cg_flag[iCG] = 0;

                p_cost_stat->sigCGFlagCost = (iCG == 0) ? 0 : p_cost_stat->sigCGFlagCost0;
                p_cost_stat->lastRunCost = 0;
                pairsInCG = 0;
                rank      = rank_pre;
            }
        } else {
            p_cost_stat->sigCGFlagCost = lambda_rdoq * est_rate_nonzero_cg_flag(p_rdoq, 0, sig_cg_ctx);
        }
    }

    p_cost_stat->pairNum = pairsInCG;

    return rank;
}

/* ---------------------------------------------------------------------------
 */
static
int rdoq_cg(xavs2_t *h, rdoq_t *p_rdoq, cu_t *p_cu, coeff_t *ncur_blk, const int num_coeff, int qp)
{
    ALIGN16(cost_state_t    cg_cost_stat [64]);
    ALIGN16(level_info_t    cg_level_data[16]);   // level data in a CG
    int8_t *p_sig_cg_flag = p_rdoq->sig_cg_flag;
    node_list_t list_run_level;
    cost_state_t *p_cost_stat;
    const int16_t *p_tab_coeff_scan1d = p_rdoq->p_scan_tab_1d;
    const int i_tu_level = p_rdoq->i_tu_level;
    const int shift_bit = 16 - (h->param->sample_bit_depth + 1) - i_tu_level;
    const double f_err_level_mult = 256.0 / (1 << (shift_bit * 2));
    const int thres_lower_int = (int)((16384 << shift_bit) / (double)(tab_Q_TAB[qp]));
    const rdcost_t lambda_rdoq = h->f_lambda_rdoq;
    int last_pos = -1;
    int rank = 0;
    int num_cg;
    int i_cg;
    int num_nonzero = 0;  // number of non-zero coefficients

#if ENABLE_WQUANT
    wq_data_t *wq = &h->wq_data;
    int wqm_size_id = 0;
    int wqm_size = 0;

    /* init weighted quant block size */
    if (h->WeightQuantEnable) {
        wqm_size_id = get_block_size_id_wq(p_cu->cu_info.i_tu_split, i_tu_level, p_rdoq->b_luma);
        wqm_size    = 1 << (wqm_size_id + 2);
    }
#else
    UNUSED_PARAMETER(p_cu);
#endif

    /* init */
    list_init(&list_run_level);
    memset(p_sig_cg_flag, 0, sizeof(p_rdoq->sig_cg_flag));

    /* 跳过尾部的全零系数cg */
    num_cg = rdoq_get_last_cg_pos(ncur_blk, num_coeff, thres_lower_int);

    for (i_cg = num_cg - 1; i_cg >= 0; i_cg--) {
        node_t *p_node = NULL;
        int idx_coeff_in_cg = 15;
        int idx_coeff = (i_cg << 4) + idx_coeff_in_cg;

        p_cost_stat = &cg_cost_stat[i_cg];
        for (; idx_coeff_in_cg >= 0; idx_coeff_in_cg--, idx_coeff--) {
            level_info_t *p_level_info = &cg_level_data[idx_coeff_in_cg];
            int xx_yy;

            // quant_init
            xx_yy = p_tab_coeff_scan1d[idx_coeff];

            /* 1, generate levels of one coefficient [xx, yy] */
#if ENABLE_WQUANT
            if (h->WeightQuantEnable) {
                ncur_blk[idx_coeff] = (coeff_t)rdoq_est_coeff_level_wq(h, p_level_info,
                                      wq, wqm_size_id, wqm_size, xx, yy, ncur_blk[idx_coeff],
                                      qp, shift_bit);
            } else {
                ncur_blk[idx_coeff] = (coeff_t)rdoq_est_coeff_level(p_level_info, ncur_blk[idx_coeff],
                                      qp, shift_bit, f_err_level_mult, thres_lower_int);
            }
#else
            ncur_blk[idx_coeff] = (coeff_t)rdoq_est_coeff_level(p_level_info, ncur_blk[idx_coeff],
                                  qp, shift_bit, f_err_level_mult, thres_lower_int);
#endif

            p_level_info->pos_xy   = xx_yy;
            p_level_info->pos_scan = idx_coeff;

            /* 2, build (Level, Run) pair linked list */
            if (last_pos == -1) { // last is not found yet
                if (p_level_info->num_level > 1) {
                    list_init(&list_run_level);
                    // found last position in last CG
                    last_pos = idx_coeff_in_cg;
                    // first node in the list is last position
                    p_node = create_and_append_node(&list_run_level, NULL, LAST_POS, idx_coeff_in_cg);

                    // the second node is the (run, pair) pair
                    p_node = create_and_append_node(&list_run_level, p_level_info, RUN_LEVEL_PAIR, idx_coeff_in_cg);

                    num_cg = i_cg + 1;
                    p_sig_cg_flag[i_cg] = 1; // this is the last CG
                }
            } else { // last is found
                // first node is last run
                if (idx_coeff_in_cg == 15) { // a new CG begins
                    list_init(&list_run_level);
                    // the position of the last run is always initialized to 15
                    p_node = create_and_append_node(&list_run_level, NULL, LAST_RUN, idx_coeff_in_cg);
                }

                // starting from the 2nd node, it is (level, run) node
                if (p_level_info->num_level > 1) {
                    p_node = create_and_append_node(&list_run_level, p_level_info, RUN_LEVEL_PAIR, idx_coeff_in_cg);
                    p_sig_cg_flag[i_cg] = 1;
                    // get the real position of last run
                    if (p_node->prev->attrib == LAST_RUN) {
                        p_node->prev->pos = idx_coeff_in_cg;
                    }
                } else {
                    p_node->run++;
                }
            }
        }

        /* 3, estimate costs */
        if (last_pos != -1) { // a CG just ended
            rank = rdoq_est_cg(h, p_rdoq, cg_level_data, p_cost_stat, list_run_level.head, ncur_blk, p_sig_cg_flag, i_cg, rank);
            num_nonzero += p_cost_stat->pairNum;
        }
    }

    if (!num_nonzero) {
        return 0;
    }

    /* 4, estimate last */
    i_cg = num_cg - 1;
    if (last_pos != -1) {
        int CGx, CGy;
        int xx, yy;
        int pos_last_scan = last_pos + (i_cg << 4); // get scan position of the last
        rdcost_t cost_uncoded_block = 0;
        rdcost_t cost_prev_last_cg, cost_prev_last_pos;
        rdcost_t cost_prev_level = 0, cost_prev_run = 0, cost_prev_uncoded = 0;
        rdcost_t cost_best, cost_curr;

        cost_prev_last_cg = lambda_rdoq * est_rate_last_cg_pos(p_rdoq, num_cg - 1, &CGx, &CGy);
        p_cost_stat = &cg_cost_stat[i_cg];
        xx = p_cost_stat->pairCost[0].posBlockX;
        yy = p_cost_stat->pairCost[0].posBlockY;
        cost_prev_last_pos = lambda_rdoq * est_rate_last_coeff_pos(p_rdoq, xx, yy, 1, CGx, CGy);

        // init cost_best
        cost_best = cost_prev_last_cg + cost_prev_last_pos + cost_prev_level + cost_prev_run;
        cost_curr = cost_best;

        for (; i_cg >= 0; i_cg--, p_cost_stat--) {
            int pairNo = 0;
            rdcost_t cost_curr_last_cg = lambda_rdoq * est_rate_last_cg_pos(p_rdoq, i_cg, &CGx, &CGy);
            rdcost_t cost_curr_last_pos;
            pair_cost_t *p_pair_cost = &p_cost_stat->pairCost[pairNo];

            if (i_cg != num_cg - 1) { // last run
                cost_best += p_cost_stat->lastRunCost;
                if (i_cg > 0) {
                    cost_best += p_cost_stat->sigCGFlagCost;
                }
            }
            cost_curr += cost_curr_last_cg - cost_prev_last_cg;

            for (; pairNo < p_cost_stat->pairNum; pairNo++, p_pair_cost++) { // when pairNo == 0, it is the last pair in CG
                // last position in last CG
                xx = p_pair_cost->posBlockX;
                yy = p_pair_cost->posBlockY;

                cost_curr_last_pos = lambda_rdoq * est_rate_last_coeff_pos(p_rdoq, xx, yy, 1, CGx, CGy);
                cost_curr += cost_curr_last_pos - cost_prev_last_pos + cost_prev_uncoded
                             +  p_pair_cost->levelCost - cost_prev_level
                             +  p_pair_cost->runCost   - cost_prev_run;

                cost_best += p_pair_cost->levelCost + p_pair_cost->runCost;

                cost_uncoded_block += p_pair_cost->uncodedCost;

                cost_prev_uncoded  = p_pair_cost->uncodedCost;
                cost_prev_level    = p_pair_cost->levelCost;
                cost_prev_run      = p_pair_cost->runCost;
                cost_prev_last_pos = cost_curr_last_pos;

                if (cost_curr <= cost_best) {
                    cost_best = cost_curr;
                    rdoq_memset_zero_coeffs(ncur_blk, p_pair_cost->scanPos + 1, pos_last_scan + 1);
                    pos_last_scan = p_pair_cost->scanPos;
                }
            }
            cost_prev_last_cg = cost_curr_last_cg;
        }

        // cost_uncoded_block is the total uncoded distortion
        // cost_best is the summation of Best LastPos and Best lastCG and CGSign and lastrun and (run,level)s
        if (cost_uncoded_block < cost_best) {
            rdoq_memset_zero_coeffs(ncur_blk, 0, pos_last_scan + 1);
            return 0;
        }

        i_cg = num_cg - 2;
        p_cost_stat = &cg_cost_stat[i_cg];
        // estimate last for each non-last CG
        for (; i_cg >= 0; i_cg--, p_cost_stat--) {
            pair_cost_t *p_pair_cost = &p_cost_stat->pairCost[0];
            int lastScanPosInCG;
            int pairNo;
            rdcost_t cost_curr_last_pos;

            if (p_sig_cg_flag[i_cg] == 0) {
                continue;
            }
            lastScanPosInCG = p_pair_cost->scanPos;

            // Last Position in current CG
            xx = p_pair_cost->posBlockX;
            yy = p_pair_cost->posBlockY;
            cost_prev_last_pos = lambda_rdoq * est_rate_last_coeff_pos(p_rdoq, xx, yy, 0, CGx, CGy);
            cost_prev_level    = 0;
            cost_prev_run      = 0;
            cost_prev_uncoded  = 0;
            cost_curr          = cost_prev_last_pos;
            cost_best          = cost_curr;

            for (pairNo = 0; pairNo < p_cost_stat->pairNum; pairNo++, p_pair_cost++) {
                // Last Position in current CG
                xx = p_pair_cost->posBlockX;
                yy = p_pair_cost->posBlockY;

                cost_curr_last_pos = lambda_rdoq * est_rate_last_coeff_pos(p_rdoq, xx, yy, 0, CGx, CGy);
                cost_curr += cost_curr_last_pos - cost_prev_last_pos + cost_prev_uncoded
                             + p_pair_cost->levelCost - cost_prev_level
                             + p_pair_cost->runCost   - cost_prev_run;

                if (pairNo == p_cost_stat->pairNum - 1) {
                    cost_curr += p_cost_stat->sigCGFlagCost0 - p_cost_stat->sigCGFlagCost;
                }

                cost_best += p_pair_cost->levelCost + p_pair_cost->runCost;

                cost_prev_uncoded  = p_pair_cost->uncodedCost;
                cost_prev_level    = p_pair_cost->levelCost;
                cost_prev_run      = p_pair_cost->runCost;
                cost_prev_last_pos = cost_curr_last_pos;

                if (cost_curr <= cost_best) {
                    cost_best = cost_curr;
                    rdoq_memset_zero_coeffs(ncur_blk, p_pair_cost->scanPos + 1, lastScanPosInCG + 1);
                    lastScanPosInCG = p_pair_cost->scanPos;
                }
            }
        }
    }

    return num_nonzero;
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void
rdoq_init(rdoq_t *p_rdoq, aec_t *p_aec, cu_t *p_cu, int bsx, int bsy, int i_tu_level, int b_luma, int intra_mode)
{
    int scan_level = XAVS2_MIN(2, i_tu_level - 2);
    int b_swap_xy;

    b_swap_xy = (b_luma && tab_intra_mode_scan_type[intra_mode] == INTRA_PRED_HOR && p_cu->cu_info.i_mode != PRED_I_2Nxn && p_cu->cu_info.i_mode != PRED_I_nx2N);

    p_rdoq->num_cg_x    = bsx >> 2;
    p_rdoq->num_cg_y    = bsy >> 2;
    p_rdoq->i_tu_level  = i_tu_level;
    p_rdoq->b_luma      = b_luma;
    p_rdoq->b_dc_diag   = (b_luma && tab_intra_mode_scan_type[intra_mode] != INTRA_PRED_DC_DIAG) ? 0 : 1;

    if (b_swap_xy) {
        p_rdoq->bit_size_shift_x = xavs2_log2u(bsx);
        p_rdoq->bit_size_shift_y = 0;
    } else {
        p_rdoq->bit_size_shift_x = 0;
        p_rdoq->bit_size_shift_y = xavs2_log2u(bsx);
    }

    if (b_luma && p_cu->cu_info.i_tu_split == TU_SPLIT_HOR) {
        p_rdoq->p_scan_cg     = tab_cg_scan_list_hor  [scan_level];
        p_rdoq->p_scan_tab_1d = tab_coef_scan1_list_hor[scan_level];
    } else if (b_luma && p_cu->cu_info.i_tu_split == TU_SPLIT_VER) {
        p_rdoq->p_scan_cg     = tab_cg_scan_list_ver  [scan_level];
        p_rdoq->p_scan_tab_1d = tab_coef_scan1_list_ver[scan_level];
    } else {
        scan_level = XAVS2_MIN(3, i_tu_level - 2);
        p_rdoq->p_scan_cg     = tab_cg_scan_list_nxn[scan_level];
        p_rdoq->p_scan_tab_1d = tab_coef_scan1_list_nxn[b_swap_xy][scan_level];
    }

    // initialize contexts
    if (b_luma) {
        p_rdoq->p_ctx_primary  = p_aec->p_ctx_set->coeff_run[0];
        p_rdoq->p_ctx_sign_cg  = p_aec->p_ctx_set->nonzero_cg_flag;
        p_rdoq->p_ctx_last_cg  = p_aec->p_ctx_set->last_cg_contexts;
        p_rdoq->p_ctx_last_pos = p_aec->p_ctx_set->last_pos_contexts;
    } else {
        p_rdoq->p_ctx_primary  = p_aec->p_ctx_set->coeff_run[1];
        p_rdoq->p_ctx_sign_cg  = p_aec->p_ctx_set->nonzero_cg_flag  + NUM_SIGN_CG_CTX_LUMA;
        p_rdoq->p_ctx_last_cg  = p_aec->p_ctx_set->last_cg_contexts  + NUM_LAST_CG_CTX_LUMA;
        p_rdoq->p_ctx_last_pos = p_aec->p_ctx_set->last_pos_contexts + NUM_LAST_POS_CTX_LUMA;
    }

    p_rdoq->p_ctx_coeff_level = p_aec->p_ctx_set->coeff_level;
}


/**
 * ===========================================================================
 * interface function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
int rdoq_block(xavs2_t *h, aec_t *p_aec, cu_t *p_cu, coeff_t *cur_blk, int bsx, int bsy, int i_tu_level, int qp, int b_luma, int intra_mode)
{
    cu_parallel_t *p_enc = cu_get_enc_context(h, p_cu->cu_info.i_level);
    rdoq_t  *p_rdoq  = &p_enc->rdoq_info;
    coeff_t *p_coeff = p_rdoq->coeff_buff;
    coeff_t *ncur_blk = p_rdoq->ncur_blk;
    const int coeff_num = bsx * bsy;
    int num_non_zero = 0;
    int i;
    const int16_t *p_tab_coeff_scan_1d;

    rdoq_init(p_rdoq, p_aec, p_cu, bsx, bsy, i_tu_level, b_luma, intra_mode);

    g_funcs.dctf.abs_coeff(p_coeff, cur_blk, coeff_num);

    /* scan the coeffs */
    p_tab_coeff_scan_1d = p_rdoq->p_scan_tab_1d;

    for (i = 0; i < coeff_num; i++) {
        ncur_blk[i] = p_coeff[p_tab_coeff_scan_1d[i]];
    }

    num_non_zero = rdoq_cg(h, p_rdoq, p_cu, ncur_blk, coeff_num, qp);

    /* inverse scan the coeffs */
    if (num_non_zero) {
        for (i = 0; i < coeff_num; i++) {
            p_coeff[p_tab_coeff_scan_1d[i]] = ncur_blk[i];
        }

        num_non_zero = g_funcs.dctf.add_sign(cur_blk, p_coeff, coeff_num);
    } else {
        cur_blk[0] = 0;
    }

    return num_non_zero;
}
