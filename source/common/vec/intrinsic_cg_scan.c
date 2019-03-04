/*
 * intrinsic_cg_scan.c
 *
 * Description of this file:
 *    SSE assembly functions of CG-Scanning module of the xavs2 library
 *
 * --------------------------------------------------------------------------
 *
 *    xavs2 - video encoder of AVS2/IEEE1857.4 video coding standard
 *    Copyright (C) 2018~ VCL, NELVT, Peking University
 *
 *    Authors: Falei LUO <falei.luo@gmail.com>
 *             Jiaqi ZHANG <zhangjiaqi.cs@gmail.com>
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

#include "../common.h"
#include "intrinsic.h"

#include <mmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>


/* ---------------------------------------------------------------------------
 */
void coeff_scan_4x4_xy_sse128(coeff_t *dst, const coeff_t *src, int i_src_shift)
{
    __m128i row0, row1, row2, row3;
    __m128i dst1, dst2;
    __m128i order1, order2;
    int int1, int2;

    order1 = _mm_setr_epi16(0x0100, 0x0302, 0x0908, 0x0F0E, 0x0B0A, 0x0504, 0x0706, 0x0D0C);
    order2 = _mm_setr_epi16(0x0302, 0x0908, 0x0B0A, 0x0504, 0x0100, 0x0706, 0x0D0C, 0x0F0E);

    row0 = _mm_loadl_epi64((const __m128i*)&src[0 << i_src_shift]);
    row1 = _mm_loadl_epi64((const __m128i*)&src[(int64_t)(1 << i_src_shift)]);
    row2 = _mm_loadl_epi64((const __m128i*)&src[2 << i_src_shift]);
    row3 = _mm_loadl_epi64((const __m128i*)&src[3 << i_src_shift]);

    dst1 = _mm_unpacklo_epi64(row0, row1);    //0 1 2 3 4 5 6 7
    dst2 = _mm_unpacklo_epi64(row2, row3);    //8 9 10 11 12 13 14 15

    int1 = _mm_extract_epi16(dst1, 7);
    int2 = _mm_extract_epi16(dst2, 0);

    dst1 = _mm_insert_epi16(dst1, int2, 7);    //0 1 2 3 4 5 6 8
    dst2 = _mm_insert_epi16(dst2, int1, 0);    //7 9 10 11 12 13 14 15

    //0 1 2 3 4 5 6 8    ------->    0 1 4 8 5 2 3 6
    dst1 = _mm_shuffle_epi8(dst1, order1);
    //0 1  2  3  4  5  6  7
    //7 9 10 11 12 13 14 15    -------->    9 12 13 10 7 11 14 15
    dst2 = _mm_shuffle_epi8(dst2, order2);

    _mm_store_si128((__m128i*)(dst + 0), dst1);
    _mm_store_si128((__m128i*)(dst + 8), dst2);
}

/* ---------------------------------------------------------------------------
 */
void coeff_scan_4x4_yx_sse128(coeff_t *dst, const coeff_t *src, int i_src_shift)
{
    __m128i row0, row1, row2, row3;
    __m128i dst1, dst2;
    __m128i order1, order2;
    int int1, int2;

    order1 = _mm_setr_epi16(0x0100, 0x0908, 0x0302, 0x0504, 0x0B0A, 0x0D0C, 0x0706, 0x0F0E);
    order2 = _mm_setr_epi16(0x0100, 0x0908, 0x0302, 0x0504, 0x0B0A, 0x0D0C, 0x0706, 0x0F0E);

    row0 = _mm_loadl_epi64((const __m128i*)&src[0 << i_src_shift]);    // 0  1  2  3
    row1 = _mm_loadl_epi64((const __m128i*)&src[(int64_t)1 << i_src_shift]);    // 4  5  6  7
    row2 = _mm_loadl_epi64((const __m128i*)&src[2 << i_src_shift]);    // 8  9 10 11
    row3 = _mm_loadl_epi64((const __m128i*)&src[3 << i_src_shift]);    //12 13 14 15

    dst1 = _mm_unpacklo_epi64(row0, row1);    //0 1 2 3 4 5 6 7
    dst2 = _mm_unpacklo_epi64(row2, row3);    //8 9 10 11 12 13 14 15

    int1 = _mm_extract_epi32(dst1, 3);
    int2 = _mm_extract_epi32(dst2, 0);

    dst1 = _mm_insert_epi32(dst1, int2, 3);    //0 1 2 3 4 5 8 9
    dst2 = _mm_insert_epi32(dst2, int1, 0);    //6 7 10 11 12 13 14 15

    int1 = _mm_extract_epi16(dst1, 3);
    int2 = _mm_extract_epi16(dst2, 4);

    dst1 = _mm_insert_epi16(dst1, int2, 3);    //0 1 2 12 4 5 8 9
    dst2 = _mm_insert_epi16(dst2, int1, 4);    //6 7 10 11 3 13 14 15

    //0 1 2  3 4 5 6 7
    //0 1 2 12 4 5 8 9    ------->    0 4 1 2 5 8 12 9
    dst1 = _mm_shuffle_epi8(dst1, order1);
    //0 1  2  3 4  5  6  7
    //6 7 10 11 3 13 14 15    -------->    6 3 7 10 13 14 11 15
    dst2 = _mm_shuffle_epi8(dst2, order2);

    _mm_store_si128((__m128i*)(dst + 0), dst1);
    _mm_store_si128((__m128i*)(dst + 8), dst2);
}

#if ARCH_X86_64
/* ---------------------------------------------------------------------------
 */
void coeff_scan4_xy_sse128(coeff_t *dst, uint64_t r1, uint64_t r2, uint64_t r3, uint64_t r4)
{
    __m128i dst1, dst2;
    __m128i order1, order2;
    int int1, int2;

    order1 = _mm_setr_epi16(0x0100, 0x0302, 0x0908, 0x0F0E, 0x0B0A, 0x0504, 0x0706, 0x0D0C);
    order2 = _mm_setr_epi16(0x0302, 0x0908, 0x0B0A, 0x0504, 0x0100, 0x0706, 0x0D0C, 0x0F0E);

    dst1 = _mm_set_epi64x((int64_t)r2, (int64_t)r1);    //0 1 2 3 4 5 6 7
    dst2 = _mm_set_epi64x((int64_t)r4, (int64_t)r3);    //8 9 10 11 12 13 14 15

    int1 = _mm_extract_epi16(dst1, 7);
    int2 = _mm_extract_epi16(dst2, 0);

    dst1 = _mm_insert_epi16(dst1, int2, 7);    //0 1 2 3 4 5 6 8
    dst2 = _mm_insert_epi16(dst2, int1, 0);    //7 9 10 11 12 13 14 15

    //0 1 2 3 4 5 6 8    ------->    0 1 4 8 5 2 3 6
    dst1 = _mm_shuffle_epi8(dst1, order1);
    //0 1  2  3  4  5  6  7
    //7 9 10 11 12 13 14 15    -------->    9 12 13 10 7 11 14 15
    dst2 = _mm_shuffle_epi8(dst2, order2);

    _mm_store_si128((__m128i*)(dst + 0), dst1);
    _mm_store_si128((__m128i*)(dst + 8), dst2);
}

/* ---------------------------------------------------------------------------
 */
void coeff_scan4_yx_sse128(coeff_t *dst, uint64_t r1, uint64_t r2, uint64_t r3, uint64_t r4)
{
    __m128i dst1, dst2;
    __m128i order1, order2;
    int int1, int2;

    order1 = _mm_setr_epi16(0x0100, 0x0908, 0x0302, 0x0504, 0x0B0A, 0x0D0C, 0x0706, 0x0F0E);
    order2 = _mm_setr_epi16(0x0100, 0x0908, 0x0302, 0x0504, 0x0B0A, 0x0D0C, 0x0706, 0x0F0E);

    dst1 = _mm_set_epi64x((int64_t)r2, (int64_t)r1);    //0 1 2 3 4 5 6 7
    dst2 = _mm_set_epi64x((int64_t)r4, (int64_t)r3);    //8 9 10 11 12 13 14 15

    int1 = _mm_extract_epi32(dst1, 3);
    int2 = _mm_extract_epi32(dst2, 0);

    dst1 = _mm_insert_epi32(dst1, int2, 3);    //0 1 2 3 4 5 8 9
    dst2 = _mm_insert_epi32(dst2, int1, 0);    //6 7 10 11 12 13 14 15

    int1 = _mm_extract_epi16(dst1, 3);
    int2 = _mm_extract_epi16(dst2, 4);

    dst1 = _mm_insert_epi16(dst1, int2, 3);    //0 1 2 12 4 5 8 9
    dst2 = _mm_insert_epi16(dst2, int1, 4);    //6 7 10 11 3 13 14 15

    //0 1 2  3 4 5 6 7
    //0 1 2 12 4 5 8 9    ------->    0 4 1 2 5 8 12 9
    dst1 = _mm_shuffle_epi8(dst1, order1);
    //0 1  2  3 4  5  6  7
    //6 7 10 11 3 13 14 15    -------->    6 3 7 10 13 14 11 15
    dst2 = _mm_shuffle_epi8(dst2, order2);

    _mm_store_si128((__m128i*)(dst + 0), dst1);
    _mm_store_si128((__m128i*)(dst + 8), dst2);
}
#endif  // ARCH_X86_64
