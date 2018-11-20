/*
 * intrinsic_pixel_avx.c
 *
 * Description of this file:
 *    AVX2 assembly functions of Pixel-Processing module of the xavs2 library
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

#include <mmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <string.h>

#include "../basic_types.h"
#include "../avs2_defs.h"
#include "intrinsic.h"

/* ---------------------------------------------------------------------------
 */
void *xavs2_memzero_aligned_c_avx(void *dst, size_t n)
{
    __m256i *p_dst = (__m256i *)dst;
    __m256i m0 = _mm256_setzero_si256();
    int i = (int)(n >> 5);

    for (; i != 0; i--) {
        _mm256_store_si256(p_dst, m0);
        p_dst++;
    }

    return dst;
}


/* ---------------------------------------------------------------------------
 */
void xavs2_mem_repeat_i_c_avx(void *dst, int val, size_t count)
{
    __m256i *p_dst = (__m256i *)dst;
    __m256i m0 = _mm256_set1_epi32(val);
    int i = (int)((count + 7) >> 3);

    for (; i != 0; i--) {
        _mm256_store_si256(p_dst, m0);
        p_dst++;
    }
}

void padding_rows_sse256_10bit(pel_t *src, int i_src, int width, int height, int start, int rows, int pad)
{
    int i, j;
    pel_t *p, *p1, *p2;
    int pad_lr = pad + 16 - (pad & 0xF);
    start = XAVS2_MAX(start, 0);

    if (start + rows > height) {
        rows = height - start;
    }

    p = src + start * i_src;

    // left & right
    for (i = 0; i < rows; i++) {
        __m256i Val1 = _mm256_set1_epi16((int16_t)p[0]);
        __m256i Val2 = _mm256_set1_epi16((int16_t)p[width - 1]);
        p1 = p - pad_lr;
        p2 = p + width;
        for (j = 0; j < pad_lr; j += 16) {
            _mm256_storeu_si256((__m256i *)(p1 + j), Val1);
            _mm256_storeu_si256((__m256i *)(p2 + j), Val2);
        }

        p += i_src;
    }

    if (start == 0) {
        p = src - pad;
        for (i = 1; i <= pad; i++) {
            memcpy(p - i_src * i, p, (width + 2 * pad) * sizeof(pel_t));
        }
    }

    if (start + rows == height) {
        p = src + i_src * (height - 1) - pad;
        for (i = 1; i <= pad; i++) {
            memcpy(p + i_src * i, p, (width + 2 * pad) * sizeof(pel_t));
        }
    }
}

void padding_rows_lr_sse256_10bit(pel_t *src, int i_src, int width, int height, int start, int rows, int pad)
{
    int i, j;
    pel_t *p, *p1, *p2;
    int pad_lr = pad + 16 - (pad & 0xF);
    start = XAVS2_MAX(start, 0);

    if (start + rows > height) {
        rows = height - start;
    }

    p = src + start * i_src;

    // left & right
    for (i = 0; i < rows; i++) {
        __m256i Val1 = _mm256_set1_epi16((int16_t)p[0]);
        __m256i Val2 = _mm256_set1_epi16((int16_t)p[width - 1]);
        p1 = p - pad_lr;
        p2 = p + width;
        for (j = 0; j < pad_lr; j += 16) {
            _mm256_storeu_si256((__m256i *)(p1 + j), Val1);
            _mm256_storeu_si256((__m256i *)(p2 + j), Val2);
        }

        p += i_src;
    }
}


void add_pel_clip_sse256(const pel_t *src1, int i_src1, const coeff_t *src2, int i_src2, pel_t *dst, int i_dst,
                         int width, int height)
{
    int i, j;
    __m256i mask;
    __m128i mask1;

    if (width >= 32) {
        __m256i S, R1, R2, S1, S2, D;
        __m256i zero = _mm256_setzero_si256();
        mask = _mm256_load_si256((const __m256i *)intrinsic_mask32[(width & 31)]);
        for (i = 0; i < height; i++) {
            S = _mm256_loadu_si256((const __m256i *)(src1));
            R1 = _mm256_loadu_si256((const __m256i *)(src2));
            R2 = _mm256_loadu_si256((const __m256i *)(src2 + 16));
            S = _mm256_permute4x64_epi64(S, 0xd8);
            S1 = _mm256_unpacklo_epi8(S, zero);
            S2 = _mm256_unpackhi_epi8(S, zero);
            S1 = _mm256_add_epi16(R1, S1);
            S2 = _mm256_add_epi16(R2, S2);
            D = _mm256_packus_epi16(S1, S2);
            D = _mm256_permute4x64_epi64(D, 0xd8);
            _mm256_storeu_si256((__m256i *)(dst), D);

            if (width > 32) {
                S = _mm256_loadu_si256((const __m256i *)(src1 + 32));
                R1 = _mm256_loadu_si256((const __m256i *)(src2 + 32));
                R2 = _mm256_loadu_si256((const __m256i *)(src2 + 48));
                S = _mm256_permute4x64_epi64(S, 0xd8);
                S1 = _mm256_unpacklo_epi8(S, zero);
                S2 = _mm256_unpackhi_epi8(S, zero);
                S1 = _mm256_add_epi16(R1, S1);
                S2 = _mm256_add_epi16(R2, S2);
                D = _mm256_packus_epi16(S1, S2);
                D = _mm256_permute4x64_epi64(D, 0xd8);
                _mm256_maskstore_epi32((int *)(dst + 32), mask, D);
            }
            src1 += i_src1;
            src2 += i_src2;
            dst += i_dst;
        }
    } else {
        __m128i zero = _mm_setzero_si128();
        __m128i S, S1, S2, R1, R2, D;
        if (width & 15) {
            mask1 = _mm_load_si128((const __m128i *)intrinsic_mask[(width & 15) - 1]);

            for (i = 0; i < height; i++) {
                for (j = 0; j < width - 15; j += 16) {
                    S = _mm_load_si128((const __m128i *)(src1 + j));
                    R1 = _mm_load_si128((const __m128i *)(src2 + j));
                    R2 = _mm_load_si128((const __m128i *)(src2 + j + 8));
                    S1 = _mm_unpacklo_epi8(S, zero);
                    S2 = _mm_unpackhi_epi8(S, zero);
                    S1 = _mm_add_epi16(R1, S1);
                    S2 = _mm_add_epi16(R2, S2);
                    D = _mm_packus_epi16(S1, S2);
                    _mm_store_si128((__m128i *)(dst + j), D);
                }

                S = _mm_loadu_si128((const __m128i *)(src1 + j));
                R1 = _mm_loadu_si128((const __m128i *)(src2 + j));
                R2 = _mm_loadu_si128((const __m128i *)(src2 + j + 8));
                S1 = _mm_unpacklo_epi8(S, zero);
                S2 = _mm_unpackhi_epi8(S, zero);
                S1 = _mm_add_epi16(R1, S1);
                S2 = _mm_add_epi16(R2, S2);
                D = _mm_packus_epi16(S1, S2);
                _mm_maskmoveu_si128(D, mask1, (char *)&dst[j]);

                src1 += i_src1;
                src2 += i_src2;
                dst += i_dst;
            }
        } else {
            for (i = 0; i < height; i++) {
                for (j = 0; j < width; j += 16) {
                    S = _mm_load_si128((const __m128i *)(src1 + j));
                    R1 = _mm_load_si128((const __m128i *)(src2 + j));
                    R2 = _mm_load_si128((const __m128i *)(src2 + j + 8));
                    S1 = _mm_unpacklo_epi8(S, zero);
                    S2 = _mm_unpackhi_epi8(S, zero);
                    S1 = _mm_add_epi16(R1, S1);
                    S2 = _mm_add_epi16(R2, S2);
                    D = _mm_packus_epi16(S1, S2);
                    _mm_store_si128((__m128i *)(dst + j), D);
                }
                src1 += i_src1;
                src2 += i_src2;
                dst += i_dst;
            }
        }
    }

}

void xavs2_pixel_average_avx(pel_t *dst, int i_dst, pel_t *src1, int i_src1, pel_t *src2, int i_src2, int width, int height)
{
    int i;

    if (width >= 32) {
        __m256i mask, S1, S2, D;

        mask = _mm256_load_si256((const __m256i *)intrinsic_mask32[(width & 31)]);
        for (i = 0; i < height; i++) {
            S1 = _mm256_loadu_si256((const __m256i *)(src1));
            S2 = _mm256_loadu_si256((const __m256i *)(src2));
            D = _mm256_avg_epu8(S1, S2);
            _mm256_storeu_si256((__m256i *)(dst), D);

            if (32 < width) {
                S1 = _mm256_loadu_si256((const __m256i *)(src1 + 32));
                S2 = _mm256_loadu_si256((const __m256i *)(src2 + 32));
                D = _mm256_avg_epu8(S1, S2);
                _mm256_maskstore_epi32((int *)(dst + 32), mask, D);
            }
            src1 += i_src1;
            src2 += i_src2;
            dst += i_dst;
        }
    } else {
        int  j;
        __m128i S1, S2, D;

        if (width & 15) {
            __m128i mask = _mm_load_si128((const __m128i *)intrinsic_mask[(width & 15) - 1]);

            for (i = 0; i < height; i++) {
                for (j = 0; j < width - 15; j += 16) {
                    S1 = _mm_loadu_si128((const __m128i *)(src1 + j));
                    S2 = _mm_load_si128((const __m128i *)(src2 + j));
                    D = _mm_avg_epu8(S1, S2);
                    _mm_storeu_si128((__m128i *)(dst + j), D);
                }

                S1 = _mm_loadu_si128((const __m128i *)(src1 + j));
                S2 = _mm_loadu_si128((const __m128i *)(src2 + j));
                D = _mm_avg_epu8(S1, S2);
                _mm_maskmoveu_si128(D, mask, (char *)&dst[j]);

                src1 += i_src1;
                src2 += i_src2;
                dst += i_dst;
            }
        } else {
            for (i = 0; i < height; i++) {
                for (j = 0; j < width; j += 16) {
                    S1 = _mm_loadu_si128((const __m128i *)(src1 + j));
                    S2 = _mm_load_si128((const __m128i *)(src2 + j));
                    D = _mm_avg_epu8(S1, S2);
                    _mm_storeu_si128((__m128i *)(dst + j), D);
                }
                src1 += i_src1;
                src2 += i_src2;
                dst += i_dst;
            }
        }
    }

}

void padding_rows_lr_sse256(pel_t *src, int i_src, int width, int height, int start, int rows, int pad)
{
    int i, j;
    pel_t *p, *p1, *p2;

    start = XAVS2_MAX(start, 0);

    if (start + rows > height) {
        rows = height - start;
    }

    p = src + start * i_src;

    pad = pad + 16 - (pad & 0xF);
    if (pad & 0x1f) {
        __m256i mask = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0);
        for (i = 0; i < rows; i++) {
            __m256i Val1 = _mm256_set1_epi8((char)p[0]);
            __m256i Val2 = _mm256_set1_epi8((char)p[width - 1]);
            p1 = p - pad;
            p2 = p + width;
            for (j = 0; j < pad - 31; j += 32) {
                _mm256_storeu_si256((__m256i *)(p1 + j), Val1);
                _mm256_storeu_si256((__m256i *)(p2 + j), Val2);
            }
            _mm256_maskstore_epi32((int *)(p1 + j), mask, Val1);
            _mm256_maskstore_epi32((int *)(p2 + j), mask, Val2);
            p += i_src;
        }
    } else {
        __m256i Val1 = _mm256_set1_epi8((char)p[0]);
        __m256i Val2 = _mm256_set1_epi8((char)p[width - 1]);
        p1 = p - pad;
        p2 = p + width;
        for (j = 0; j < pad; j += 32) {
            _mm256_storeu_si256((__m256i *)(p1 + j), Val1);
            _mm256_storeu_si256((__m256i *)(p2 + j), Val2);
        }
        p += i_src;
    }
}

void padding_rows_sse256(pel_t *src, int i_src, int width, int height, int start, int rows, int pad)
{
    int i, j;
    pel_t *p, *p1, *p2;

    start = XAVS2_MAX(start, 0);

    if (start + rows > height) {
        rows = height - start;
    }

    p = src + start * i_src;

    pad = pad + 16 - (pad & 0xF);
    if (pad & 0x1f) {
        __m256i mask = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0);
        for (i = 0; i < rows; i++) {
            __m256i Val1 = _mm256_set1_epi8((char)p[0]);
            __m256i Val2 = _mm256_set1_epi8((char)p[width - 1]);
            p1 = p - pad;
            p2 = p + width;
            for (j = 0; j < pad - 31; j += 32) {
                _mm256_storeu_si256((__m256i *)(p1 + j), Val1);
                _mm256_storeu_si256((__m256i *)(p2 + j), Val2);
            }
            _mm256_maskstore_epi32((int *)(p1 + j), mask, Val1);
            _mm256_maskstore_epi32((int *)(p2 + j), mask, Val2);
            p += i_src;
        }
    } else {
        __m256i Val1 = _mm256_set1_epi8((char)p[0]);
        __m256i Val2 = _mm256_set1_epi8((char)p[width - 1]);
        p1 = p - pad;
        p2 = p + width;
        for (j = 0; j < pad; j += 32) {
            _mm256_storeu_si256((__m256i *)(p1 + j), Val1);
            _mm256_storeu_si256((__m256i *)(p2 + j), Val2);
        }
        p += i_src;
    }

    if (start == 0) {
        p = src - pad;
        for (i = 1; i <= pad; i++) {
            memcpy(p - i_src * i, p, (width + 2 * pad) * sizeof(pel_t));
        }
    }

    if (start + rows == height) {
        p = src + i_src * (height - 1) - pad;
        for (i = 1; i <= pad; i++) {
            memcpy(p + i_src * i, p, (width + 2 * pad) * sizeof(pel_t));
        }
    }
}

