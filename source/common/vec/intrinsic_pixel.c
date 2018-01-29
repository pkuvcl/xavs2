/*
 * intrinsic_pixel.c
 *
 * Description of this file:
 *    SSE assembly functions of Pixel-Processing module of the xavs2 library
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

#include "../basic_types.h"
#include "intrinsic.h"

#include <mmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>


void xavs2_pixel_average_sse128(pel_t *dst, int i_dst, pel_t *src1, int i_src1, pel_t *src2, int i_src2, int width, int height)
{
    int i, j;
    __m128i S1, S2, D;

    if (width & 15) {
        __m128i mask = _mm_load_si128((const __m128i*)intrinsic_mask[(width & 15) - 1]);

        for (i = 0; i < height; i++) {
            for (j = 0; j < width - 15; j += 16) {
                S1 = _mm_loadu_si128((const __m128i*)(src1 + j));
                S2 = _mm_loadu_si128((const __m128i*)(src2 + j));
                D = _mm_avg_epu8(S1, S2);
                _mm_storeu_si128((__m128i*)(dst + j), D);
            }

            S1 = _mm_loadu_si128((const __m128i*)(src1 + j));
            S2 = _mm_loadu_si128((const __m128i*)(src2 + j));
            D = _mm_avg_epu8(S1, S2);
            _mm_maskmoveu_si128(D, mask, (char*)&dst[j]);

            src1 += i_src1;
            src2 += i_src2;
            dst += i_dst;
        }
    } else {
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j += 16) {
                S1 = _mm_loadu_si128((const __m128i*)(src1 + j));
                S2 = _mm_loadu_si128((const __m128i*)(src2 + j));
                D = _mm_avg_epu8(S1, S2);
                _mm_storeu_si128((__m128i*)(dst + j), D);
            }
            src1 += i_src1;
            src2 += i_src2;
            dst += i_dst;
        }
    }

}

/* ---------------------------------------------------------------------------
 */
void *xavs2_memzero_aligned_c_sse2(void *dst, size_t n)
{
    __m128i *p_dst = (__m128i *)dst;
    __m128i m0 = _mm_setzero_si128();
    int i = (int)(n >> 4);

    for (; i != 0; i--) {
        _mm_store_si128(p_dst, m0);
        p_dst++;
    }

    return dst;
}

/* ---------------------------------------------------------------------------
 */
void xavs2_mem_repeat_i_c_sse2(void *dst, int val, size_t count)
{
    __m128i *p_dst = (__m128i *)dst;
    __m128i m0 = _mm_set1_epi32(val);
    int i = (int)((count + 3) >> 2);

    for (; i != 0; i--) {
        _mm_store_si128(p_dst, m0);
        p_dst++;
    }
}

/* ---------------------------------------------------------------------------
 */
void *xavs2_memcpy_aligned_c_sse2(void *dst, const void *src, size_t n)
{
    __m128i *p_dst = (__m128i *)dst;
    const __m128i *p_src = (const __m128i *)src;
    int i = (int)(n >> 4);

    for (; i != 0; i--) {
        _mm_store_si128(p_dst, _mm_load_si128(p_src));
        p_src++;
        p_dst++;
    }

    return dst;
}

