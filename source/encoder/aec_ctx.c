/*
 * aec_ctx.c
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
#include "cudata.h"
#include "aec.h"
#include "bitstream.h"
#include "block_info.h"


/* ---------------------------------------------------------------------------
 * 0: INTRA_PRED_VER
 * 1: INTRA_PRED_HOR
 * 2: INTRA_PRED_DC_DIAG
 */
const int tab_intra_mode_scan_type[NUM_INTRA_MODE] = {
    2, 2, 2, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0
};

const int8_t tab_intra_mode_luma2chroma[NUM_INTRA_MODE] = {
    DC_PRED_C,   -1, BI_PRED_C, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    VERT_PRED_C, -1,        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    HOR_PRED_C,  -1,        -1, -1, -1, -1, -1, -1, -1
};

#if CTRL_OPT_AEC
context_t g_tab_ctx_mps[4096 * 5];    /* [2 * lg_pmps + mps + cycno * 4096] */
context_t g_tab_ctx_lps[4096 * 5];    /* [2 * lg_pmps + mps + cycno * 4096] */

static const uint8_t tab_cwr_shift[] = {
    3, 3, 4, 5, 5, 5, 5 /* 5, 5, 5, 5 */
};

static const uint16_t tab_lg_pmps_offset[6] = {
    0, 0, 0, 197, 95, 46 /* 5, 5, 5, 5 */
};
#endif

/* ---------------------------------------------------------------------------
 * 向码流文件中输出所有剩余bits，结束码流
 */
static INLINE
void bitstr_end_stream(aec_t *p_aec)
{
    if (p_aec->num_left_flush_bits == NUM_FLUSH_BITS) {
        return;
    }

    switch (NUM_FLUSH_BITS - p_aec->num_left_flush_bits) {
    case 24:
        p_aec->p[0] = (uint8_t)(p_aec->reg_flush_bits >> (NUM_FLUSH_BITS - 8));
        p_aec->p[1] = (uint8_t)(p_aec->reg_flush_bits >> (NUM_FLUSH_BITS - 16));
        p_aec->p[2] = (uint8_t)(p_aec->reg_flush_bits >> (NUM_FLUSH_BITS - 24));
        p_aec->p += 3;
        break;
    case 16:
        p_aec->p[0] = (uint8_t)(p_aec->reg_flush_bits >> (NUM_FLUSH_BITS - 8));
        p_aec->p[1] = (uint8_t)(p_aec->reg_flush_bits >> (NUM_FLUSH_BITS - 16));
        p_aec->p += 2;
        break;
    case 8:
        p_aec->p[0] = (uint8_t)(p_aec->reg_flush_bits >> (NUM_FLUSH_BITS - 8));
        p_aec->p += 1;
        break;
    default:
        fprintf(stderr, "Un-aligned tail bits %d\n", p_aec->num_left_flush_bits);
        assert(0);
        break;
    }

    p_aec->num_left_flush_bits = NUM_FLUSH_BITS;
}




/* ---------------------------------------------------------------------------
 * 向码流文件中输出one bit和剩余的位数
 */
static INLINE
void bitstt_put_one_bit_and_remainder(aec_t *p_aec, const int b)
{
    uint32_t N = 1 + p_aec->i_bits_to_follow;   // 总共输出的比特数

    if (N > p_aec->num_left_flush_bits) {   /* 编码的比特数超过当前码流字节中剩余的比特数 */
        int header_bits = p_aec->num_left_flush_bits;   // 当前码流最后一个字节剩余位的数量
        uint32_t header_byte = (1 << (header_bits - 1)) - (!b);  // 剩余位的填充值
        int num_left_bytes = (N - header_bits) >> 3;            // 除开当前字节外，剩余应该填充的整字节数
        int num_left_bits = N - header_bits - (num_left_bytes << 3);   // 多余的比特数

        p_aec->reg_flush_bits |= header_byte;
        bitstr_flush_bits(p_aec);
        p_aec->num_left_flush_bits = NUM_FLUSH_BITS - num_left_bits;

        if (b == 0) {
            /* b 为零时中间的bits全部填充 1 */
            while (num_left_bytes != 0) {
                *(p_aec->p) = 0xff;
                p_aec->p++;
                num_left_bytes--;
            }
            /* 最后填充 num_left_bits 位到 reg_flush_bits 的最高位 */
            p_aec->reg_flush_bits = 0xffu >> (8 - num_left_bits) << p_aec->num_left_flush_bits;
        } else {
            p_aec->p += num_left_bytes;
        }
    } else  {  /* 当前需要输出的bit数量小于码流中写入字节剩余的bit数量 */
        uint32_t bits = (1 << p_aec->i_bits_to_follow) - (!b);  // 输出的比特组成的二进制值

        p_aec->reg_flush_bits |= bits << (p_aec->num_left_flush_bits - N);
        p_aec->num_left_flush_bits -= N;
        if (p_aec->num_left_flush_bits == 0) {
            bitstr_flush_bits(p_aec);
            p_aec->reg_flush_bits      = 0;
            p_aec->num_left_flush_bits = NUM_FLUSH_BITS;
        }
    }
    p_aec->i_bits_to_follow = 0;

}


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
static ALWAYS_INLINE void aec_set_function_handles(xavs2_t *h, binary_t *fh, int b_writing)
{
    if (b_writing) {
        // write bitstream to buffer
        memcpy(fh, &gf_aec_default, sizeof(binary_t));
    } else {
        // estimate bit rate without writing (during RDO)
        switch (h->param->rdo_bit_est_method) {
        case 1:
            memcpy(fh, &gf_aec_fastrdo, sizeof(binary_t));
            break;
        case 2:
            memcpy(fh, &gf_aec_vrdo, sizeof(binary_t));
            break;
        default:
            memcpy(fh, &gf_aec_rdo, sizeof(binary_t));
            break;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
void init_contexts(aec_t *p_aec)
{
    const uint16_t lg_pmps = ((QUARTER << LG_PMPS_SHIFTNO) - 1);
    uint16_t  v = MAKE_CONTEXT(lg_pmps, 0, 0);
    uint16_t *d = (uint16_t *)&p_aec->ctx_set;
    int ctx_cnt = sizeof(ctx_set_t) / sizeof(uint16_t);

    while (ctx_cnt-- != 0) {
        *d++ = v;
    }
    p_aec->p_ctx_set = &p_aec->ctx_set;
}

#if CTRL_OPT_AEC
/* ---------------------------------------------------------------------------
 */
void init_aec_context_tab(void)
{
    context_t ctx_i;
    context_t ctx_o;
    int cycno;
    int mps;

    /* init context table */
    ctx_i.v = 0;
    ctx_o.v = 0;

    /* mps */
    for (cycno = 0; cycno < 4; cycno++) {
        uint32_t cwr = tab_cwr_shift[cycno];
        ctx_i.cycno = cycno;
        ctx_o.cycno = (uint8_t)XAVS2_MAX(cycno, 1);

        for (mps = 0; mps < 2; mps++) {
            ctx_i.MPS = (uint8_t)mps;
            ctx_o.MPS = (uint8_t)mps;
            for (ctx_i.LG_PMPS = 0; ctx_i.LG_PMPS <= 1024; ctx_i.LG_PMPS++) {
                uint32_t lg_pmps = ctx_i.LG_PMPS;
                lg_pmps -= (lg_pmps >> cwr) + (lg_pmps >> (cwr + 2));
                ctx_o.LG_PMPS = (uint16_t)lg_pmps;
                g_tab_ctx_mps[ctx_i.v].v = ctx_o.v;
            }
        }
    }

    /* lps */
    for (cycno = 0; cycno < 4; cycno++) {
        uint32_t cwr = tab_cwr_shift[cycno];
        ctx_i.cycno = cycno;
        ctx_o.cycno = (uint8_t)XAVS2_MIN(cycno + 1, 3);

        for (mps = 0; mps < 2; mps++) {
            ctx_i.MPS = (uint8_t)mps;
            ctx_o.MPS = (uint8_t)mps;
            for (ctx_i.LG_PMPS = 0; ctx_i.LG_PMPS <= 1024; ctx_i.LG_PMPS++) {
                uint32_t lg_pmps = ctx_i.LG_PMPS + tab_lg_pmps_offset[cwr];
                if (lg_pmps >= (256 << LG_PMPS_SHIFTNO)) {
                    lg_pmps = (512 << LG_PMPS_SHIFTNO) - 1 - lg_pmps;
                    ctx_o.MPS = !mps;
                }
                ctx_o.LG_PMPS = (uint16_t)lg_pmps;
                g_tab_ctx_lps[ctx_i.v].v = ctx_o.v;
            }
        }
    }
}
#endif

/* ---------------------------------------------------------------------------
 * initializes the aec_t for the arithmetic coder
 */
void aec_start(xavs2_t *h, aec_t *p_aec, uint8_t *p_bs_start, uint8_t *p_bs_end, int b_writing)
{
    p_aec->p_start          = p_bs_start;
    p_aec->p                = p_bs_start;
    p_aec->p_end            = p_bs_end;
    p_aec->i_low            = 0;
    p_aec->i_t1             = 0xFF;
    p_aec->i_bits_to_follow = 0;
    p_aec->b_writting       = 0;

    p_aec->num_left_flush_bits = NUM_FLUSH_BITS + 1;      // to swallow first redundant bit
    p_aec->reg_flush_bits      = 0;
    if (b_writing) {
        memset(p_aec->p_start, 0, p_bs_end - p_bs_start);
    }

    /* int function handles */
    aec_set_function_handles(h, &p_aec->binary, b_writing);

    /* init contexts */
    init_contexts(p_aec);
}

/* ---------------------------------------------------------------------------
 * terminates the arithmetic codeword, writes stop bit and stuffing bytes (if any)
 */
void aec_done(aec_t *p_aec)
{
    int i;
    uint8_t bit_out_standing = (uint8_t)((p_aec->i_low >> (B_BITS - 1)) & 1);
    uint8_t bit_ending;

    bitstt_put_one_bit_and_remainder(p_aec, bit_out_standing);

    bit_ending = (uint8_t)((p_aec->i_low >> (B_BITS - 2)) & 1);
    bitstr_put_one_bit(p_aec, bit_ending);

    /* end of AEC */
    bitstr_put_one_bit(p_aec, 1);
    for (i = 0; i < 7; i++) {
        bitstr_put_one_bit(p_aec, 0);
    }

    /* write stuffing pattern */
    bitstr_put_one_bit(p_aec, 1);
    if (p_aec->num_left_flush_bits != NUM_FLUSH_BITS) {
        for (i = p_aec->num_left_flush_bits & 7; i > 0; i--) {
            bitstr_put_one_bit(p_aec, 0);
        }
    }

    /* end bitstream */
    bitstr_end_stream(p_aec);
}

/* ---------------------------------------------------------------------------
 * create structure for storing coding state
 */
void aec_init_coding_state(aec_t *p_aec)
{
    if (p_aec == NULL) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "aec_create_coding_state: p_aec");
    } else {
        memset(p_aec, 0, sizeof(aec_t));
        p_aec->p_ctx_set = &p_aec->ctx_set;
    }
}
