/*
 * intrinsic_cg_scan_avx.c
 *
 * Description of this file:
 *    AVX2 assembly functions of CG-Scanning module of the xavs2 library
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

#include <mmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#include "../common.h"
#include "intrinsic.h"

#if ARCH_X86_64
/* ---------------------------------------------------------------------------
 */
void coeff_scan4_xy_avx(coeff_t *dst, uint64_t r1, uint64_t r2, uint64_t r3, uint64_t r4)
{
    __m256i order1;
    __m256i m_in;
    int16_t int1, int2;

    order1 = _mm256_setr_epi16(0x0100, 0x0302, 0x0908, 0x0F0E, 0x0B0A, 0x0504, 0x0706, 0x0D0C,
                               0x0302, 0x0908, 0x0B0A, 0x0504, 0x0100, 0x0706, 0x0D0C, 0x0F0E);

    m_in = _mm256_setr_epi64x(r1, r2, r3, r4);    // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

    int1 = _mm256_extract_epi16(m_in, 7);
    int2 = _mm256_extract_epi16(m_in, 8);

    _mm256_insert_epi16(m_in, int2, 7);    //0 1 2 3 4 5 6 8
    _mm256_insert_epi16(m_in, int1, 8);    //7 9 10 11 12 13 14 15

    //0 1  2  3  4  5  6  8    -------->    0  1  4  8 5  2  3  6
    //7 9 10 11 12 13 14 15    -------->    9 12 13 10 7 11 14 15
    m_in = _mm256_shuffle_epi8(m_in, order1);

    _mm256_storeu_si256((__m256i*)dst, m_in);
}

/* ---------------------------------------------------------------------------
 */
void coeff_scan4_yx_avx(coeff_t *dst, uint64_t r1, uint64_t r2, uint64_t r3, uint64_t r4)
{
    __m256i order1;
    __m256i m_in;
    int32_t int1, int2;

    order1 = _mm256_setr_epi16(0x0100, 0x0908, 0x0302, 0x0504, 0x0B0A, 0x0D0C, 0x0706, 0x0F0E,
                               0x0100, 0x0908, 0x0302, 0x0504, 0x0B0A, 0x0D0C, 0x0706, 0x0F0E);

    m_in = _mm256_setr_epi64x(r1, r2, r3, r4);    // 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

    int1 = _mm256_extract_epi32(m_in, 3);
    int2 = _mm256_extract_epi32(m_in, 4);

    _mm256_insert_epi32(m_in, int2, 3);    //0 1 2 3 4 5 8 9
    _mm256_insert_epi32(m_in, int1, 4);    //6 7 10 11 12 13 14 15

    int1 = _mm256_extract_epi16(m_in, 3);
    int2 = _mm256_extract_epi16(m_in, 12);

    _mm256_insert_epi16(m_in, (int16_t)int2, 3);    //0 1 2 12 4 5 8 9
    _mm256_insert_epi16(m_in, (int16_t)int1, 12);    //6 7 10 11 3 13 14 15

    //0 1  2 12 4  5  8  9    -------->    0 4 1  2  5  8 12  9
    //6 7 10 11 3 13 14 15    -------->    6 3 7 10 13 14 11 15
    m_in= _mm256_shuffle_epi8(m_in, order1);

    _mm256_storeu_si256((__m256i*)dst, m_in);
}
#endif  // ARCH_X86_64
