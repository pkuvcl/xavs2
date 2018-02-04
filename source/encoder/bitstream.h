/*
 * bitstream.h
 *
 * Description of this file:
 *    Bitstream Processing functions definition of the xavs2 library
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

#ifndef XAVS2_BITSTREAM_H
#define XAVS2_BITSTREAM_H


/**
 * ===========================================================================
 * global variables
 * ===========================================================================
 */

#if XAVS2_TRACE
extern int g_sym_count;         /* global symbol count for trace */
extern int g_bit_count;         /* global bit    count for trace */
#endif


/**
 * ===========================================================================
 * function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void xavs2_bs_init(bs_t *bs, void *p_data, int i_data)
{
    bs->p_start = (uint8_t *)p_data;
    bs->p       = (uint8_t *)p_data;
    bs->p_end   = bs->p_start + i_data;
    bs->i_left  = 8;
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE int xavs2_bs_pos(bs_t *bs)
{
    return ((int)(8 * (bs->p - bs->p_start) + 8 - bs->i_left));
}

/* ---------------------------------------------------------------------------
 * writes UVLC code to the bitstream buffer
 */
static ALWAYS_INLINE void xavs2_bs_write(bs_t *bs, uint32_t code, int len)
{
    assert(bs->p < bs->p_end);

    while (len > 0) {
        if (len < 32) {
            code &= (1 << len) - 1;
        }
        if (len < bs->i_left) {
            (*bs->p) = (uint8_t)(((*bs->p) << len) | code);
            bs->i_left -= len;
            break;
        } else {
            (*bs->p) = (uint8_t)(((*bs->p) << bs->i_left) | (code >> (len - bs->i_left)));
            bs->p++;
            len -= bs->i_left;
            bs->i_left = 8;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void xavs2_bs_write1(bs_t *bs, uint8_t i_bit)
{
    if (bs->p < bs->p_end) {
        (*bs->p) <<= 1;
        (*bs->p) |= i_bit;
        if (--bs->i_left == 0) {
            bs->i_left = 8;
            bs->p++;
        }
    }
}

/* ---------------------------------------------------------------------------
 * one bit "1" is added to the end of stream, then some bits "0" are added
 * to byte aligned position.
 */
static ALWAYS_INLINE void xavs2_bs_stuff_bits(bs_t *bs)
{
    if (bs->i_left != 8) {
        xavs2_bs_write1(bs, 1);
        xavs2_bs_write(bs, 0, bs->i_left & 7);
    } else {
        xavs2_bs_write(bs, 0x80, 8);
    }
}
#define bs_stuff_bits   xavs2_bs_stuff_bits

/* ---------------------------------------------------------------------------
 * one bit "1" is added to the end of stream, then some bits "0" are added
 * to byte aligned position.
 */
static ALWAYS_INLINE int xavs2_bs_byte_align(bs_t *bs)
{
    if (bs->i_left != 8) {
        int bits = bs->i_left;
        xavs2_bs_write1(bs, 1);
        xavs2_bs_write(bs, 0, bs->i_left & 7);
        return bits;
    }

    return 0;
}
#define bs_byte_align   xavs2_bs_byte_align

/* ---------------------------------------------------------------------------
 * write out a trace string to the trace file
 */
#if XAVS2_TRACE
static void write_trace_info(char *trace_string, int bit_pattern, int value, int len)
{
    int i, chars;

    xavs2_trace("@");
    chars = xavs2_trace("%i", g_bit_count);

    while (chars++ < 6) {
        xavs2_trace(" ");
    }

    chars += xavs2_trace("%s", trace_string);

    while (chars++ < 55) {
        xavs2_trace(" ");
    }

    // align bit-pattern
    if (len < 15) {
        for (i = 0; i < 15 - len; i++) {
            xavs2_trace(" ");
        }
    }

    // print bit-pattern
    g_bit_count += len;
    for (i = 1; i <= len; i++) {
        if ((bit_pattern >> (len - i)) & 0x1) {
            xavs2_trace("1");
        } else {
            xavs2_trace("0");
        }
    }

    xavs2_trace(" (%3d) \n", value);
}
#endif

/* ---------------------------------------------------------------------------
 * write out a trace string to the trace file
 */
#if XAVS2_TRACE
static void write_trace_info2(char *trace_string, int value, int len)
{
    int i, chars;

    xavs2_trace("@");
    chars = xavs2_trace("%i", g_bit_count);

    while (chars++ < 6) {
        xavs2_trace(" ");
    }

    chars += xavs2_trace("%s", trace_string);

    while (chars++ < 55) {
        xavs2_trace(" ");
    }

    // align bit-pattern
    if (len < 15) {
        for (i = 0; i < 15 - len; i++) {
            xavs2_trace(" ");
        }
    }

    g_bit_count += len;
    while (len >= 32) {
        for (i = 0; i < 8; i++) {
            xavs2_trace("0");
        }

        len -= 8;
    }

    // print bit-pattern
    for (i = 0; i < len; i++) {
        if (0x01 & (value >> (len - i - 1))) {
            xavs2_trace("1");
        } else {
            xavs2_trace("0");
        }
    }

    xavs2_trace(" (%3d) \n", value);
}
#endif

/**
 * ---------------------------------------------------------------------------
 * Function   : writes an ue(v) syntax element, returns the length in bits
 * Parameters :
 *      [in ] : trace_string - the string for the trace file
 *            : value        - the value to be coded
 *      [out] : bs           - the bs_t the value should be coded into
 * Return     : number of bits used by the coded syntax element
 * ---------------------------------------------------------------------------
 */
#if XAVS2_TRACE
static ALWAYS_INLINE int xavs2_bs_write_ue(bs_t *bs, char *trace_string, int value)
#else
static ALWAYS_INLINE int xavs2_bs_write_ue(bs_t *bs, int value)
#endif
{
    int i, nn, len, inf, suffix_len, bit_pattern;

    // generates UVLC code and passes the codeword to the buffer
    nn = (value + 1) >> 1;
    for (i = 0; i < 16 && nn != 0; i++) {
        nn /= 2;
    }

    len = 2 * i + 1;
    inf = value + 1 - (int)pow(2, i);

    suffix_len = len >> 1;
    bit_pattern = (1 << suffix_len) | (inf & ((1 << suffix_len) - 1));

    xavs2_bs_write(bs, bit_pattern, len);

#if XAVS2_TRACE
    write_trace_info(trace_string, bit_pattern, value, len);
#endif

    return len;
}

/**
 * ---------------------------------------------------------------------------
 * Function   : writes an se(v) syntax element, returns the length in bits
 * Parameters :
 *      [in ] : trace_string - the string for the trace file
 *            : value        - the value to be coded
 *      [out] : bs           - the bs_t the value should be coded into
 * Return     : number of bits used by the coded syntax element
 * ---------------------------------------------------------------------------
 */
#if XAVS2_TRACE
static ALWAYS_INLINE int xavs2_bs_write_se(bs_t *bs, char *trace_string, int value)
#else
static ALWAYS_INLINE int xavs2_bs_write_se(bs_t *bs, int value)
#endif
{
#if XAVS2_TRACE
    return xavs2_bs_write_ue(bs, trace_string, value <= 0 ? -value * 2 : value * 2 - 1);
#else
    return xavs2_bs_write_ue(bs, value <= 0 ? -value * 2 : value * 2 - 1);
#endif
}

/**
 * ---------------------------------------------------------------------------
 * Function   : writes n bit fixed length syntax element, returns the length in bits
 * Parameters :
 *      [in ] : len          - number of bits write out
 *            : trace_string - the string for the trace file
 *            : value        - the value to be coded
 *      [out] : bs           - the bs_t the value should be coded into
 * Return     : number of bits used by the coded syntax element
 * ---------------------------------------------------------------------------
 */
#if XAVS2_TRACE
static ALWAYS_INLINE int xavs2_bs_write_uv(bs_t *bs, int len, char *trace_string, int value, int b_trace)
#else
static ALWAYS_INLINE int xavs2_bs_write_uv(bs_t *bs, int len, int value)
#endif
{
    xavs2_bs_write(bs, value, len);

#if XAVS2_TRACE
    if (b_trace) {
        write_trace_info(trace_string, value, value, len);
    }
#endif

    return len;
}

#if XAVS2_TRACE
#define ue_v(bs, value, trace_string)       xavs2_bs_write_ue(bs, trace_string, value)
#define se_v(bs, value, trace_string)       xavs2_bs_write_se(bs, trace_string, value)
#define u_v(bs, len, value, trace_string)   xavs2_bs_write_uv(bs, len, trace_string, value, 1)
#define u_0(bs, len, value, trace_string)   xavs2_bs_write_uv(bs, len, trace_string, value, 0)    /* no trace */

#else
#define ue_v(bs, value, trace_string)       xavs2_bs_write_ue(bs, value)
#define se_v(bs, value, trace_string)       xavs2_bs_write_se(bs, value)
#define u_v(bs, len, value, trace_string)   xavs2_bs_write_uv(bs, len, value)
#define u_0(bs, len, value, trace_string)   xavs2_bs_write_uv(bs, len, value)
#endif

#endif  // XAVS2_BITSTREAM_H
