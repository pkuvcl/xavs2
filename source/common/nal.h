/*
 * nal.h
 *
 * Description of this file:
 *    NAL functions definition of the xavs2 library
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


#ifndef XAVS2_NAL_H
#define XAVS2_NAL_H

#include "bitstream.h"

/**
 * ===========================================================================
 * nal function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void nal_start(xavs2_t *h, int i_type, int i_ref_idc)
{
    nal_t *nal = &h->p_nal[h->i_nal];

    nal->i_ref_idc = i_ref_idc;
    nal->i_type    = i_type;
    nal->i_payload = 0;
    nal->p_payload = &h->p_bs_buf_header[xavs2_bs_pos(&h->header_bs) >> 3];
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void nal_end(xavs2_t *h)
{
    nal_t *nal = &h->p_nal[h->i_nal];
    uint8_t *end = &h->p_bs_buf_header[xavs2_bs_pos(&h->header_bs) >> 3];

    nal->i_payload = (int)(end - nal->p_payload);
    h->i_nal++;
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void
nal_merge_slice(xavs2_t *h, uint8_t *p_bs_buffer, int i_bs_len, int i_type, int i_ref_idc)
{
    nal_t *nal = &h->p_nal[h->i_nal];

    assert(i_bs_len > 8);

    // update the current nal
    nal->i_ref_idc = i_ref_idc;
    nal->i_type    = i_type;
    nal->i_payload = i_bs_len;
    nal->p_payload = p_bs_buffer;

    // next nal
    h->i_nal++;
}

/* ---------------------------------------------------------------------------
 */
static INLINE
uint8_t *nal_escape_c(uint8_t *dst, uint8_t *src, uint8_t *end)
{
    int left_bits = 8;
    uint8_t tmp = 0;

    /* check pseudo start code */
    while (src < end) {
        tmp |= (uint8_t)(*src >> (8 - left_bits));
        if (tmp <= 0x03 && !dst[-2] && !dst[-1]) {
            *dst++ = 0x02;      /* insert '10' */
            tmp <<= 6;
            if (left_bits >= 2) {
                tmp |= (uint8_t)((*src++) << (left_bits - 2));
                left_bits = left_bits - 2;
            } else {
                tmp |= (uint8_t)((*src) >> (2 - left_bits));
                *dst++ = tmp;
                tmp = (uint8_t)((*src++) << (6 + left_bits));
                left_bits = 6 + left_bits;
            }
            continue;
        }
        *dst++ = tmp;
        tmp = (uint8_t)((*src++) << left_bits);
    }

    /* rest bits */
    if (left_bits != 8 && tmp != 0) {
        *dst++ = tmp;
    }

    return dst;
}

/* ---------------------------------------------------------------------------
 */
static INLINE
intptr_t encoder_encapsulate_nals(xavs2_t *h, xavs2_frame_t *frm, int start)
{
    uint8_t *nal_buffer;
    int previous_nal_size = 0;
    int nal_size = 0;
    int i;

    for (i = 0; i < start; i++) {
        previous_nal_size += h->p_nal[i].i_payload;
    }

    for (i = start; i < h->i_nal; i++) {
        nal_size += h->p_nal[i].i_payload;
    }

    /* NOTE: frame->i_bs_buf is big enough, no need to reallocate memory */
    // assert(previous_nal_size + nal_size <= frame->i_bs_buf);

    /* copy new nals */
    nal_buffer = frm->p_bs_buf + previous_nal_size;
    nal_size   = h->i_nal;      /* number of all nals */
    for (i = start; i < nal_size; i++) {
        nal_t *nal = &h->p_nal[i];
        memcpy(nal_buffer, nal->p_payload, nal->i_payload);
        nal_buffer += nal->i_payload;
    }

    return nal_buffer - (frm->p_bs_buf + previous_nal_size);
}


#endif // XAVS2_NAL_H
