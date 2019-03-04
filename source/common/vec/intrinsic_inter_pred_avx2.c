/*
 * intrinsic_inter-pred_avx2.c
 *
 * Description of this file:
 *    AVX2 assembly functions of Inter-Prediction module of the xavs2 library
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

#include "../basic_types.h"
#include "intrinsic.h"

#pragma warning(disable:4127)  // warning C4127: 条件表达式是常量

/* ---------------------------------------------------------------------------
 */
void intpl_luma_block_hor_w16_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff)
{
    int row, col;
    const int offset = 32;
    const int shift = 6;
    const __m256i mAddOffset = _mm256_set1_epi16((short)offset);
    const __m256i mask16 = _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0);
    const __m256i mSwitch1 = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i mSwitch2 = _mm256_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10);
    const __m256i mSwitch3 = _mm256_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12, 4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12);
    const __m256i mSwitch4 = _mm256_setr_epi8(6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14, 6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14);
    __m256i mCoef;
    src -= 3;

#if ARCH_X86_64
    mCoef = _mm256_set1_epi64x(*(long long*)coeff);
#else
    mCoef = _mm256_loadu_si256((__m256i*)coeff);
    mCoef = _mm256_permute4x64_epi64(mCoef, 0x0);
#endif

    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col += 16) {
            __m256i S = _mm256_loadu_si256((__m256i*)(src + col));
            __m256i S0 = _mm256_permute4x64_epi64(S, 0x94);
            __m256i T0, T1, T2, T3;
            __m256i sum;

            T0 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch1), mCoef);
            T1 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch2), mCoef);
            T2 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch3), mCoef);
            T3 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch4), mCoef);

            T0 = _mm256_hadd_epi16(T0, T1);
            T1 = _mm256_hadd_epi16(T2, T3);
            sum = _mm256_hadd_epi16(T0, T1);

            sum = _mm256_srai_epi16(_mm256_add_epi16(sum, mAddOffset), shift);

            sum = _mm256_packus_epi16(sum, sum);
            sum = _mm256_permute4x64_epi64(sum, 0xd8);

            _mm256_maskstore_epi32((int*)(dst + col), mask16, sum);
        }
        src += i_src;
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_luma_block_hor_w24_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff)
{
    int row;
    const int offset = 32;
    const int shift = 6;
    const __m256i mAddOffset = _mm256_set1_epi16((short)offset);
    const __m256i index = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i mask24 = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0);
    const __m256i mSwitch1 = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10);
    const __m256i mSwitch2 = _mm256_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12, 6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14);
    const __m256i mSwitch3 = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i mSwitch4 = _mm256_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10);
    const __m256i mSwitch5 = _mm256_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12, 4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12);
    const __m256i mSwitch6 = _mm256_setr_epi8(6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14, 6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14);
    __m256i mCoef;

    UNUSED_PARAMETER(width);

    src -= 3;

#if ARCH_X86_64
    mCoef = _mm256_set1_epi64x(*(long long*)coeff);
#else
    mCoef = _mm256_loadu_si256((__m256i*)coeff);
    mCoef = _mm256_permute4x64_epi64(mCoef, 0x0);
#endif

    for (row = 0; row < height; row++) {
        __m256i S0 = _mm256_loadu_si256((__m256i*)(src));
        __m256i S1 = _mm256_permute4x64_epi64(S0, 0x99);
        __m256i T0, T1, T2, T3, T4, T5;
        __m256i sum1, sum2;

        T0 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S1, mSwitch1), mCoef);
        T1 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S1, mSwitch2), mCoef);
        T2 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch3), mCoef);
        T3 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch4), mCoef);
        T4 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch5), mCoef);
        T5 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch6), mCoef);

        T0 = _mm256_hadd_epi16(T0, T1);
        sum1 = _mm256_hadd_epi16(_mm256_hadd_epi16(T2, T3), _mm256_hadd_epi16(T4, T5));
        sum2 = _mm256_hadd_epi16(T0, T0);

        sum1 = _mm256_srai_epi16(_mm256_add_epi16(sum1, mAddOffset), shift);
        sum2 = _mm256_srai_epi16(_mm256_add_epi16(sum2, mAddOffset), shift);

        sum2 = _mm256_permutevar8x32_epi32(sum2, index);
        sum1 = _mm256_packus_epi16(sum1, sum2);

        _mm256_maskstore_epi32((int*)(dst), mask24, sum1);
        src += i_src;
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_luma_block_ver_w32_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff)
{
    int row;
    const int offset = 32;
    const int shift = 6;
    int bsym = (coeff[1] == coeff[6]);
    const int i_src2 = i_src * 2;
    const int i_src3 = i_src * 3;
    const int i_src4 = i_src * 4;
    const int i_src5 = i_src * 5;
    const int i_src6 = i_src * 6;
    const int i_src7 = i_src * 7;
    __m256i mAddOffset = _mm256_set1_epi16((short)offset);

    UNUSED_PARAMETER(width);

    src -= 3 * i_src;

    if (bsym) {
        __m256i coeff0 = _mm256_set1_epi8(coeff[0]);
        __m256i coeff1 = _mm256_set1_epi8(coeff[1]);
        __m256i coeff2 = _mm256_set1_epi8(coeff[2]);
        __m256i coeff3 = _mm256_set1_epi8(coeff[3]);
        __m256i T0, T1, T2, T3, T4, T5, T6, T7, mVal1, mVal2;

        for (row = 0; row < height; row++) {
            __m256i S0 = _mm256_loadu_si256((__m256i*)(src));
            __m256i S1 = _mm256_loadu_si256((__m256i*)(src + i_src));
            __m256i S2 = _mm256_loadu_si256((__m256i*)(src + i_src2));
            __m256i S3 = _mm256_loadu_si256((__m256i*)(src + i_src3));
            __m256i S4 = _mm256_loadu_si256((__m256i*)(src + i_src4));
            __m256i S5 = _mm256_loadu_si256((__m256i*)(src + i_src5));
            __m256i S6 = _mm256_loadu_si256((__m256i*)(src + i_src6));
            __m256i S7 = _mm256_loadu_si256((__m256i*)(src + i_src7));

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S0, S7), coeff0);
            T4 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S0, S7), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S1, S6), coeff1);
            T5 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S1, S6), coeff1);
            T2 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S2, S5), coeff2);
            T6 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S2, S5), coeff2);
            T3 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S3, S4), coeff3);
            T7 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S3, S4), coeff3);

            mVal1 = _mm256_add_epi16(_mm256_add_epi16(T0, T1), _mm256_add_epi16(T2, T3));
            mVal2 = _mm256_add_epi16(_mm256_add_epi16(T4, T5), _mm256_add_epi16(T6, T7));

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(mVal1, mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(mVal2, mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);

            _mm256_storeu_si256((__m256i*)(dst), mVal1);

            src += i_src;
            dst += i_dst;
        }
    } else {
        __m256i coeff0 = _mm256_set1_epi16(*(short*)coeff);
        __m256i coeff1 = _mm256_set1_epi16(*(short*)(coeff + 2));
        __m256i coeff2 = _mm256_set1_epi16(*(short*)(coeff + 4));
        __m256i coeff3 = _mm256_set1_epi16(*(short*)(coeff + 6));
        __m256i T0, T1, T2, T3, T4, T5, T6, T7, mVal1, mVal2;

        for (row = 0; row < height; row++) {
            __m256i S0 = _mm256_loadu_si256((__m256i*)(src));
            __m256i S1 = _mm256_loadu_si256((__m256i*)(src + i_src));
            __m256i S2 = _mm256_loadu_si256((__m256i*)(src + i_src2));
            __m256i S3 = _mm256_loadu_si256((__m256i*)(src + i_src3));
            __m256i S4 = _mm256_loadu_si256((__m256i*)(src + i_src4));
            __m256i S5 = _mm256_loadu_si256((__m256i*)(src + i_src5));
            __m256i S6 = _mm256_loadu_si256((__m256i*)(src + i_src6));
            __m256i S7 = _mm256_loadu_si256((__m256i*)(src + i_src7));

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S0, S1), coeff0);
            T4 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S0, S1), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S2, S3), coeff1);
            T5 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S2, S3), coeff1);
            T2 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S4, S5), coeff2);
            T6 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S4, S5), coeff2);
            T3 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S6, S7), coeff3);
            T7 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S6, S7), coeff3);

            mVal1 = _mm256_add_epi16(_mm256_add_epi16(T0, T1), _mm256_add_epi16(T2, T3));
            mVal2 = _mm256_add_epi16(_mm256_add_epi16(T4, T5), _mm256_add_epi16(T6, T7));

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(mVal1, mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(mVal2, mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);

            _mm256_storeu_si256((__m256i*)(dst), mVal1);

            src += i_src;
            dst += i_dst;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_luma_block_ver_w64_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff)
{
    int row;
    const int offset = 32;
    const int shift = 6;
    int bsym = (coeff[1] == coeff[6]);
    const int i_src2 = i_src * 2;
    const int i_src3 = i_src * 3;
    const int i_src4 = i_src * 4;
    const int i_src5 = i_src * 5;
    const int i_src6 = i_src * 6;
    const int i_src7 = i_src * 7;
    __m256i mAddOffset = _mm256_set1_epi16((short)offset);

    UNUSED_PARAMETER(width);

    src -= 3 * i_src;

    if (bsym) {
        __m256i coeff0 = _mm256_set1_epi8(coeff[0]);
        __m256i coeff1 = _mm256_set1_epi8(coeff[1]);
        __m256i coeff2 = _mm256_set1_epi8(coeff[2]);
        __m256i coeff3 = _mm256_set1_epi8(coeff[3]);

        for (row = 0; row < height; row++) {
            const pel_t *p = src + 32;
            __m256i S0 = _mm256_loadu_si256((__m256i*)(src));
            __m256i S1 = _mm256_loadu_si256((__m256i*)(src + i_src));
            __m256i S2 = _mm256_loadu_si256((__m256i*)(src + i_src2));
            __m256i S3 = _mm256_loadu_si256((__m256i*)(src + i_src3));
            __m256i S4 = _mm256_loadu_si256((__m256i*)(src + i_src4));
            __m256i S5 = _mm256_loadu_si256((__m256i*)(src + i_src5));
            __m256i S6 = _mm256_loadu_si256((__m256i*)(src + i_src6));
            __m256i S7 = _mm256_loadu_si256((__m256i*)(src + i_src7));
            __m256i T0, T1, T2, T3, T4, T5, T6, T7, mVal1, mVal2;

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S0, S7), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S1, S6), coeff1);
            T2 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S2, S5), coeff2);
            T3 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S3, S4), coeff3);
            T4 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S0, S7), coeff0);
            T5 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S1, S6), coeff1);
            T6 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S2, S5), coeff2);
            T7 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S3, S4), coeff3);

            mVal1 = _mm256_add_epi16(_mm256_add_epi16(T0, T1), _mm256_add_epi16(T2, T3));
            mVal2 = _mm256_add_epi16(_mm256_add_epi16(T4, T5), _mm256_add_epi16(T6, T7));

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(mVal1, mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(mVal2, mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);

            _mm256_storeu_si256((__m256i*)(dst), mVal1);

            S0 = _mm256_loadu_si256((__m256i*)(p));
            S1 = _mm256_loadu_si256((__m256i*)(p + i_src));
            S2 = _mm256_loadu_si256((__m256i*)(p + i_src2));
            S3 = _mm256_loadu_si256((__m256i*)(p + i_src3));
            S4 = _mm256_loadu_si256((__m256i*)(p + i_src4));
            S5 = _mm256_loadu_si256((__m256i*)(p + i_src5));
            S6 = _mm256_loadu_si256((__m256i*)(p + i_src6));
            S7 = _mm256_loadu_si256((__m256i*)(p + i_src7));

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S0, S7), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S1, S6), coeff1);
            T2 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S2, S5), coeff2);
            T3 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S3, S4), coeff3);
            T4 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S0, S7), coeff0);
            T5 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S1, S6), coeff1);
            T6 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S2, S5), coeff2);
            T7 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S3, S4), coeff3);

            mVal1 = _mm256_add_epi16(_mm256_add_epi16(T0, T1), _mm256_add_epi16(T2, T3));
            mVal2 = _mm256_add_epi16(_mm256_add_epi16(T4, T5), _mm256_add_epi16(T6, T7));

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(mVal1, mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(mVal2, mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);

            _mm256_storeu_si256((__m256i*)(dst + 32), mVal1);

            src += i_src;
            dst += i_dst;
        }
    } else {
        __m256i coeff0 = _mm256_set1_epi16(*(short*)coeff);
        __m256i coeff1 = _mm256_set1_epi16(*(short*)(coeff + 2));
        __m256i coeff2 = _mm256_set1_epi16(*(short*)(coeff + 4));
        __m256i coeff3 = _mm256_set1_epi16(*(short*)(coeff + 6));

        for (row = 0; row < height; row++) {
            __m256i T0, T1, T2, T3, T4, T5, T6, T7, mVal1, mVal2;
            const pel_t *p = src + 32;
            __m256i S0 = _mm256_loadu_si256((__m256i*)(src));
            __m256i S1 = _mm256_loadu_si256((__m256i*)(src + i_src));
            __m256i S2 = _mm256_loadu_si256((__m256i*)(src + i_src2));
            __m256i S3 = _mm256_loadu_si256((__m256i*)(src + i_src3));
            __m256i S4 = _mm256_loadu_si256((__m256i*)(src + i_src4));
            __m256i S5 = _mm256_loadu_si256((__m256i*)(src + i_src5));
            __m256i S6 = _mm256_loadu_si256((__m256i*)(src + i_src6));
            __m256i S7 = _mm256_loadu_si256((__m256i*)(src + i_src7));

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S0, S1), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S2, S3), coeff1);
            T2 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S4, S5), coeff2);
            T3 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S6, S7), coeff3);
            T4 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S0, S1), coeff0);
            T5 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S2, S3), coeff1);
            T6 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S4, S5), coeff2);
            T7 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S6, S7), coeff3);

            mVal1 = _mm256_add_epi16(_mm256_add_epi16(T0, T1), _mm256_add_epi16(T2, T3));
            mVal2 = _mm256_add_epi16(_mm256_add_epi16(T4, T5), _mm256_add_epi16(T6, T7));

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(mVal1, mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(mVal2, mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);

            _mm256_storeu_si256((__m256i*)(dst), mVal1);

            S0 = _mm256_loadu_si256((__m256i*)(p));
            S1 = _mm256_loadu_si256((__m256i*)(p + i_src));
            S2 = _mm256_loadu_si256((__m256i*)(p + i_src2));
            S3 = _mm256_loadu_si256((__m256i*)(p + i_src3));
            S4 = _mm256_loadu_si256((__m256i*)(p + i_src4));
            S5 = _mm256_loadu_si256((__m256i*)(p + i_src5));
            S6 = _mm256_loadu_si256((__m256i*)(p + i_src6));
            S7 = _mm256_loadu_si256((__m256i*)(p + i_src7));

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S0, S1), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S2, S3), coeff1);
            T2 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S4, S5), coeff2);
            T3 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S6, S7), coeff3);
            T4 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S0, S1), coeff0);
            T5 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S2, S3), coeff1);
            T6 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S4, S5), coeff2);
            T7 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S6, S7), coeff3);

            mVal1 = _mm256_add_epi16(_mm256_add_epi16(T0, T1), _mm256_add_epi16(T2, T3));
            mVal2 = _mm256_add_epi16(_mm256_add_epi16(T4, T5), _mm256_add_epi16(T6, T7));

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(mVal1, mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(mVal2, mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);

            _mm256_storeu_si256((__m256i*)(dst + 32), mVal1);

            src += i_src;
            dst += i_dst;
        }
    }
}


/* ---------------------------------------------------------------------------
 */
void intpl_luma_block_ver_w16_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff)
{
    int row;
    const int offset = 32;
    const int shift = 6;
    int bsym = (coeff[1] == coeff[6]);
    const int i_src2 = i_src * 2;
    const int i_src3 = i_src * 3;
    const int i_src4 = i_src * 4;
    const int i_src5 = i_src * 5;
    const int i_src6 = i_src * 6;
    const int i_src7 = i_src * 7;
    const int i_src8 = i_src * 8;
    __m256i mAddOffset = _mm256_set1_epi16((short)offset);

    src -= 3 * i_src;
    UNUSED_PARAMETER(width);

    if (bsym) {
        __m256i coeff0 = _mm256_set1_epi8(coeff[0]);
        __m256i coeff1 = _mm256_set1_epi8(coeff[1]);
        __m256i coeff2 = _mm256_set1_epi8(coeff[2]);
        __m256i coeff3 = _mm256_set1_epi8(coeff[3]);

        for (row = 0; row < height; row += 2) {
            __m128i S0 = _mm_loadu_si128((__m128i*)(src));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src + i_src));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + i_src2));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src + i_src3));
            __m128i S4 = _mm_loadu_si128((__m128i*)(src + i_src4));
            __m128i S5 = _mm_loadu_si128((__m128i*)(src + i_src5));
            __m128i S6 = _mm_loadu_si128((__m128i*)(src + i_src6));
            __m128i S7 = _mm_loadu_si128((__m128i*)(src + i_src7));
            __m128i S8 = _mm_loadu_si128((__m128i*)(src + i_src8));

            __m256i T0, T1, T2, T3, T4, T5, T6, T7, mVal1, mVal2;
            __m256i R0, R1, R2, R3, R4, R5, R6, R7;

            R0 = _mm256_set_m128i(S0, S1);
            R1 = _mm256_set_m128i(S1, S2);
            R2 = _mm256_set_m128i(S2, S3);
            R3 = _mm256_set_m128i(S3, S4);
            R4 = _mm256_set_m128i(S4, S5);
            R5 = _mm256_set_m128i(S5, S6);
            R6 = _mm256_set_m128i(S6, S7);
            R7 = _mm256_set_m128i(S7, S8);

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(R0, R7), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(R0, R7), coeff0);
            T2 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(R1, R6), coeff1);
            T3 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(R1, R6), coeff1);
            T4 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(R2, R5), coeff2);
            T5 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(R2, R5), coeff2);
            T6 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(R3, R4), coeff3);
            T7 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(R3, R4), coeff3);

            mVal1 = _mm256_add_epi16(_mm256_add_epi16(T0, T2), _mm256_add_epi16(T4, T6));
            mVal2 = _mm256_add_epi16(_mm256_add_epi16(T1, T3), _mm256_add_epi16(T5, T7));

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(mVal1, mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(mVal2, mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);

            _mm256_storeu2_m128i((__m128i*)dst, (__m128i*)(dst + i_dst), mVal1);
            src += 2 * i_src;
            dst += 2 * i_dst;
        }
    } else {
        __m256i coeff0 = _mm256_set1_epi16(*(int16_t*)(coeff + 0));
        __m256i coeff1 = _mm256_set1_epi16(*(int16_t*)(coeff + 2));
        __m256i coeff2 = _mm256_set1_epi16(*(int16_t*)(coeff + 4));
        __m256i coeff3 = _mm256_set1_epi16(*(int16_t*)(coeff + 6));

        for (row = 0; row < height; row += 2) {
            __m128i S0 = _mm_loadu_si128((__m128i*)(src));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src + i_src));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + i_src2));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src + i_src3));
            __m128i S4 = _mm_loadu_si128((__m128i*)(src + i_src4));
            __m128i S5 = _mm_loadu_si128((__m128i*)(src + i_src5));
            __m128i S6 = _mm_loadu_si128((__m128i*)(src + i_src6));
            __m128i S7 = _mm_loadu_si128((__m128i*)(src + i_src7));
            __m128i S8 = _mm_loadu_si128((__m128i*)(src + i_src8));

            __m256i T0, T1, T2, T3, T4, T5, T6, T7, mVal1, mVal2;
            __m256i R0, R1, R2, R3, R4, R5, R6, R7;

            R0 = _mm256_set_m128i(S0, S1);
            R1 = _mm256_set_m128i(S1, S2);
            R2 = _mm256_set_m128i(S2, S3);
            R3 = _mm256_set_m128i(S3, S4);
            R4 = _mm256_set_m128i(S4, S5);
            R5 = _mm256_set_m128i(S5, S6);
            R6 = _mm256_set_m128i(S6, S7);
            R7 = _mm256_set_m128i(S7, S8);

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(R0, R1), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(R0, R1), coeff0);
            T2 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(R2, R3), coeff1);
            T3 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(R2, R3), coeff1);
            T4 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(R4, R5), coeff2);
            T5 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(R4, R5), coeff2);
            T6 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(R6, R7), coeff3);
            T7 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(R6, R7), coeff3);

            mVal1 = _mm256_add_epi16(_mm256_add_epi16(T0, T2), _mm256_add_epi16(T4, T6));
            mVal2 = _mm256_add_epi16(_mm256_add_epi16(T1, T3), _mm256_add_epi16(T5, T7));

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(mVal1, mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(mVal2, mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);

            _mm256_storeu2_m128i((__m128i*)dst, (__m128i*)(dst + i_dst), mVal1);
            src += 2 * i_src;
            dst += 2 * i_dst;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_luma_block_ver_w24_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff)
{
    int row;
    const int offset = 32;
    const int shift = 6;
    int bsym = (coeff[1] == coeff[6]);
    __m256i mask24 = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0);
    const int i_src2 = i_src * 2;
    const int i_src3 = i_src * 3;
    const int i_src4 = i_src * 4;
    const int i_src5 = i_src * 5;
    const int i_src6 = i_src * 6;
    const int i_src7 = i_src * 7;
    __m256i mAddOffset = _mm256_set1_epi16((short)offset);

    UNUSED_PARAMETER(width);
    src -= 3 * i_src;

    if (bsym) {
        __m256i coeff0 = _mm256_set1_epi8(coeff[0]);
        __m256i coeff1 = _mm256_set1_epi8(coeff[1]);
        __m256i coeff2 = _mm256_set1_epi8(coeff[2]);
        __m256i coeff3 = _mm256_set1_epi8(coeff[3]);
        __m256i T0, T1, T2, T3, T4, T5, T6, T7, mVal1, mVal2;

        for (row = 0; row < height; row++) {
            __m256i S0 = _mm256_loadu_si256((__m256i*)(src));
            __m256i S1 = _mm256_loadu_si256((__m256i*)(src + i_src));
            __m256i S2 = _mm256_loadu_si256((__m256i*)(src + i_src2));
            __m256i S3 = _mm256_loadu_si256((__m256i*)(src + i_src3));
            __m256i S4 = _mm256_loadu_si256((__m256i*)(src + i_src4));
            __m256i S5 = _mm256_loadu_si256((__m256i*)(src + i_src5));
            __m256i S6 = _mm256_loadu_si256((__m256i*)(src + i_src6));
            __m256i S7 = _mm256_loadu_si256((__m256i*)(src + i_src7));

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S0, S7), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S1, S6), coeff1);
            T2 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S2, S5), coeff2);
            T3 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S3, S4), coeff3);
            T4 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S0, S7), coeff0);
            T5 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S1, S6), coeff1);
            T6 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S2, S5), coeff2);
            T7 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S3, S4), coeff3);

            mVal1 = _mm256_add_epi16(_mm256_add_epi16(T0, T1), _mm256_add_epi16(T2, T3));
            mVal2 = _mm256_add_epi16(_mm256_add_epi16(T4, T5), _mm256_add_epi16(T6, T7));

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(mVal1, mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(mVal2, mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);

            _mm256_maskstore_epi32((int*)(dst), mask24, mVal1);

            src += i_src;
            dst += i_dst;
        }
    } else {
        __m256i coeff0 = _mm256_set1_epi16(*(short*)coeff);
        __m256i coeff1 = _mm256_set1_epi16(*(short*)(coeff + 2));
        __m256i coeff2 = _mm256_set1_epi16(*(short*)(coeff + 4));
        __m256i coeff3 = _mm256_set1_epi16(*(short*)(coeff + 6));
        __m256i T0, T1, T2, T3, T4, T5, T6, T7, mVal1, mVal2;

        for (row = 0; row < height; row++) {
            __m256i S0 = _mm256_loadu_si256((__m256i*)(src));
            __m256i S1 = _mm256_loadu_si256((__m256i*)(src + i_src));
            __m256i S2 = _mm256_loadu_si256((__m256i*)(src + i_src2));
            __m256i S3 = _mm256_loadu_si256((__m256i*)(src + i_src3));
            __m256i S4 = _mm256_loadu_si256((__m256i*)(src + i_src4));
            __m256i S5 = _mm256_loadu_si256((__m256i*)(src + i_src5));
            __m256i S6 = _mm256_loadu_si256((__m256i*)(src + i_src6));
            __m256i S7 = _mm256_loadu_si256((__m256i*)(src + i_src7));

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S0, S1), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S2, S3), coeff1);
            T2 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S4, S5), coeff2);
            T3 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S6, S7), coeff3);
            T4 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S0, S1), coeff0);
            T5 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S2, S3), coeff1);
            T6 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S4, S5), coeff2);
            T7 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S6, S7), coeff3);

            mVal1 = _mm256_add_epi16(_mm256_add_epi16(T0, T1), _mm256_add_epi16(T2, T3));
            mVal2 = _mm256_add_epi16(_mm256_add_epi16(T4, T5), _mm256_add_epi16(T6, T7));

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(mVal1, mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(mVal2, mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);

            _mm256_maskstore_epi32((int*)(dst), mask24, mVal1);

            src += i_src;
            dst += i_dst;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_luma_block_ver_w48_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff)
{
    const int shift = 6;
    const int offset = (1 << shift) >> 1;
    const int i_src2 = i_src * 2;
    const int i_src3 = i_src * 3;
    const int i_src4 = i_src * 4;
    const int i_src5 = i_src * 5;
    const int i_src6 = i_src * 6;
    const int i_src7 = i_src * 7;
    const __m256i mask16 = _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0);
    int bsym = (coeff[1] == coeff[6]);
    int row;

    src -= 3 * i_src;
    UNUSED_PARAMETER(width);

    if (bsym) {
        __m256i mAddOffset = _mm256_set1_epi16((short)offset);
        __m256i coeff0 = _mm256_set1_epi8(coeff[0]);
        __m256i coeff1 = _mm256_set1_epi8(coeff[1]);
        __m256i coeff2 = _mm256_set1_epi8(coeff[2]);
        __m256i coeff3 = _mm256_set1_epi8(coeff[3]);
        __m256i T0, T1, T2, T3, T4, T5, T6, T7, mVal1, mVal2;

        for (row = 0; row < height; row++) {
            const pel_t *p = src + 32;
            __m256i S0 = _mm256_loadu_si256((__m256i*)(src));
            __m256i S1 = _mm256_loadu_si256((__m256i*)(src + i_src));
            __m256i S2 = _mm256_loadu_si256((__m256i*)(src + i_src2));
            __m256i S3 = _mm256_loadu_si256((__m256i*)(src + i_src3));
            __m256i S4 = _mm256_loadu_si256((__m256i*)(src + i_src4));
            __m256i S5 = _mm256_loadu_si256((__m256i*)(src + i_src5));
            __m256i S6 = _mm256_loadu_si256((__m256i*)(src + i_src6));
            __m256i S7 = _mm256_loadu_si256((__m256i*)(src + i_src7));

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S0, S7), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S1, S6), coeff1);
            T2 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S2, S5), coeff2);
            T3 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S3, S4), coeff3);
            T4 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S0, S7), coeff0);
            T5 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S1, S6), coeff1);
            T6 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S2, S5), coeff2);
            T7 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S3, S4), coeff3);

            mVal1 = _mm256_add_epi16(_mm256_add_epi16(T0, T1), _mm256_add_epi16(T2, T3));
            mVal2 = _mm256_add_epi16(_mm256_add_epi16(T4, T5), _mm256_add_epi16(T6, T7));

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(mVal1, mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(mVal2, mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);

            _mm256_storeu_si256((__m256i*)(dst), mVal1);

            S0 = _mm256_loadu_si256((__m256i*)(p));
            S1 = _mm256_loadu_si256((__m256i*)(p + i_src));
            S2 = _mm256_loadu_si256((__m256i*)(p + i_src2));
            S3 = _mm256_loadu_si256((__m256i*)(p + i_src3));
            S4 = _mm256_loadu_si256((__m256i*)(p + i_src4));
            S5 = _mm256_loadu_si256((__m256i*)(p + i_src5));
            S6 = _mm256_loadu_si256((__m256i*)(p + i_src6));
            S7 = _mm256_loadu_si256((__m256i*)(p + i_src7));

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S0, S7), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S1, S6), coeff1);
            T2 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S2, S5), coeff2);
            T3 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S3, S4), coeff3);
            T4 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S0, S7), coeff0);
            T5 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S1, S6), coeff1);
            T6 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S2, S5), coeff2);
            T7 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S3, S4), coeff3);

            mVal1 = _mm256_add_epi16(_mm256_add_epi16(T0, T1), _mm256_add_epi16(T2, T3));
            mVal2 = _mm256_add_epi16(_mm256_add_epi16(T4, T5), _mm256_add_epi16(T6, T7));

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(mVal1, mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(mVal2, mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);

            _mm256_maskstore_epi32((int*)(dst + 32), mask16, mVal1);

            src += i_src;
            dst += i_dst;
        }
    } else {
        __m256i mAddOffset = _mm256_set1_epi16((short)offset);
        __m256i coeff0 = _mm256_set1_epi16(*(short*)coeff);
        __m256i coeff1 = _mm256_set1_epi16(*(short*)(coeff + 2));
        __m256i coeff2 = _mm256_set1_epi16(*(short*)(coeff + 4));
        __m256i coeff3 = _mm256_set1_epi16(*(short*)(coeff + 6));
        __m256i T0, T1, T2, T3, T4, T5, T6, T7, mVal1, mVal2;

        for (row = 0; row < height; row++) {
            const pel_t *p = src + 32;
            __m256i S0 = _mm256_loadu_si256((__m256i*)(src));
            __m256i S1 = _mm256_loadu_si256((__m256i*)(src + i_src));
            __m256i S2 = _mm256_loadu_si256((__m256i*)(src + i_src2));
            __m256i S3 = _mm256_loadu_si256((__m256i*)(src + i_src3));
            __m256i S4 = _mm256_loadu_si256((__m256i*)(src + i_src4));
            __m256i S5 = _mm256_loadu_si256((__m256i*)(src + i_src5));
            __m256i S6 = _mm256_loadu_si256((__m256i*)(src + i_src6));
            __m256i S7 = _mm256_loadu_si256((__m256i*)(src + i_src7));

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S0, S1), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S2, S3), coeff1);
            T2 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S4, S5), coeff2);
            T3 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S6, S7), coeff3);
            T4 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S0, S1), coeff0);
            T5 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S2, S3), coeff1);
            T6 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S4, S5), coeff2);
            T7 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S6, S7), coeff3);

            mVal1 = _mm256_add_epi16(_mm256_add_epi16(T0, T1), _mm256_add_epi16(T2, T3));
            mVal2 = _mm256_add_epi16(_mm256_add_epi16(T4, T5), _mm256_add_epi16(T6, T7));

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(mVal1, mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(mVal2, mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);

            _mm256_storeu_si256((__m256i*)(dst), mVal1);

            S0 = _mm256_loadu_si256((__m256i*)(p));
            S1 = _mm256_loadu_si256((__m256i*)(p + i_src));
            S2 = _mm256_loadu_si256((__m256i*)(p + i_src2));
            S3 = _mm256_loadu_si256((__m256i*)(p + i_src3));
            S4 = _mm256_loadu_si256((__m256i*)(p + i_src4));
            S5 = _mm256_loadu_si256((__m256i*)(p + i_src5));
            S6 = _mm256_loadu_si256((__m256i*)(p + i_src6));
            S7 = _mm256_loadu_si256((__m256i*)(p + i_src7));

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S0, S1), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S2, S3), coeff1);
            T2 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S4, S5), coeff2);
            T3 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S6, S7), coeff3);
            T4 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S0, S1), coeff0);
            T5 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S2, S3), coeff1);
            T6 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S4, S5), coeff2);
            T7 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S6, S7), coeff3);

            mVal1 = _mm256_add_epi16(_mm256_add_epi16(T0, T1), _mm256_add_epi16(T2, T3));
            mVal2 = _mm256_add_epi16(_mm256_add_epi16(T4, T5), _mm256_add_epi16(T6, T7));

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(mVal1, mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(mVal2, mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);

            _mm256_maskstore_epi32((int*)(dst + 32), mask16, mVal1);

            src += i_src;
            dst += i_dst;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_luma_block_ext_w16_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coef_x, const int8_t *coef_y)
{
    ALIGN32(int16_t tmp_res[(64 + 7) * 64]);
    int16_t *tmp = tmp_res;
    const int i_tmp = 64;
    const int i_tmp2 = 2 * i_tmp;
    const int i_tmp3 = 3 * i_tmp;
    const int i_tmp4 = 4 * i_tmp;
    const int i_tmp5 = 5 * i_tmp;
    const int i_tmp6 = 6 * i_tmp;
    const int i_tmp7 = 7 * i_tmp;
    const int shift = 12;
    const __m256i mAddOffset = _mm256_set1_epi32((1 << shift) >> 1);

    int row, col;
    __m256i mSwitch1 = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7,       1, 2, 3, 4, 5, 6, 7, 8,  // 前 8 个点
                                        0, 1, 2, 3, 4, 5, 6, 7,       1, 2, 3, 4, 5, 6, 7, 8); // 后 8 个点
    __m256i mSwitch2 = _mm256_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9,       3, 4, 5, 6, 7, 8, 9, 10,
                                        2, 3, 4, 5, 6, 7, 8, 9,       3, 4, 5, 6, 7, 8, 9, 10);
    __m256i mSwitch3 = _mm256_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11,     5, 6, 7, 8, 9, 10, 11, 12,
                                        4, 5, 6, 7, 8, 9, 10, 11,     5, 6, 7, 8, 9, 10, 11, 12);
    __m256i mSwitch4 = _mm256_setr_epi8(6, 7, 8, 9, 10, 11, 12, 13,   7, 8, 9, 10, 11, 12, 13, 14,
                                        6, 7, 8, 9, 10, 11, 12, 13,   7, 8, 9, 10, 11, 12, 13, 14);
    __m256i mCoef;

    src = src - 3 * i_src - 3;

    //HOR
#if ARCH_X86_64
    mCoef = _mm256_set1_epi64x(*(long long*)coef_x);
#else
    mCoef = _mm256_loadu_si256((__m256i*)coef_x);
    mCoef = _mm256_permute4x64_epi64(mCoef, 0x0);
#endif

    for (row = -3; row < height + 4; row++) {
        for (col = 0; col < width; col += 16) {
            __m256i T0, T1, sum, T2, T3;
            __m256i S = _mm256_loadu_si256((__m256i*)(src + col));
            // 把前8个点插值依赖的像素点和后8个点插值依赖的点分别载入到前后各128位
            __m256i S0 = _mm256_permute4x64_epi64(S, 0x94);

            T0 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch1), mCoef);
            T1 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch2), mCoef);
            T2 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch3), mCoef);
            T3 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch4), mCoef);

            sum = _mm256_hadd_epi16(_mm256_hadd_epi16(T0, T1), _mm256_hadd_epi16(T2, T3));

            _mm256_store_si256((__m256i*)(tmp + col), sum);
        }
        src += i_src;
        tmp += i_tmp;
    }

    // VER
    tmp = tmp_res;

    __m256i mCoefy1 = _mm256_cvtepi8_epi16(_mm_set1_epi16(*(int16_t*)coef_y));
    __m256i mCoefy2 = _mm256_cvtepi8_epi16(_mm_set1_epi16(*(int16_t*)(coef_y + 2)));
    __m256i mCoefy3 = _mm256_cvtepi8_epi16(_mm_set1_epi16(*(int16_t*)(coef_y + 4)));
    __m256i mCoefy4 = _mm256_cvtepi8_epi16(_mm_set1_epi16(*(int16_t*)(coef_y + 6)));

    // 同时插值2行/4行，减少重复load
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col += 16) {
            __m256i T0, T1, T2, T3, T4, T5, T6, T7;
            __m256i mVal1, mVal2;
            __m256i S0 = _mm256_load_si256((__m256i*)(tmp + col));
            __m256i S1 = _mm256_load_si256((__m256i*)(tmp + col + i_tmp));
            __m256i S2 = _mm256_load_si256((__m256i*)(tmp + col + i_tmp2));
            __m256i S3 = _mm256_load_si256((__m256i*)(tmp + col + i_tmp3));
            __m256i S4 = _mm256_load_si256((__m256i*)(tmp + col + i_tmp4));
            __m256i S5 = _mm256_load_si256((__m256i*)(tmp + col + i_tmp5));
            __m256i S6 = _mm256_load_si256((__m256i*)(tmp + col + i_tmp6));
            __m256i S7 = _mm256_load_si256((__m256i*)(tmp + col + i_tmp7));

            T0 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S0, S1), mCoefy1);
            T1 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S2, S3), mCoefy2);
            T2 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S4, S5), mCoefy3);
            T3 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S6, S7), mCoefy4);
            T4 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S0, S1), mCoefy1);
            T5 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S2, S3), mCoefy2);
            T6 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S4, S5), mCoefy3);
            T7 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S6, S7), mCoefy4);

            mVal1 = _mm256_add_epi32(_mm256_add_epi32(T0, T1), _mm256_add_epi32(T2, T3));
            mVal2 = _mm256_add_epi32(_mm256_add_epi32(T4, T5), _mm256_add_epi32(T6, T7));

            mVal1 = _mm256_srai_epi32(_mm256_add_epi32(mVal1, mAddOffset), shift);
            mVal2 = _mm256_srai_epi32(_mm256_add_epi32(mVal2, mAddOffset), shift);

            mVal1 = _mm256_packs_epi32(mVal1, mVal2);
            mVal1 = _mm256_packus_epi16(mVal1, mVal1);

            mVal1 = _mm256_permute4x64_epi64(mVal1, 0xd8);
            _mm_storeu_si128((__m128i*)(dst + col), _mm256_castsi256_si128(mVal1));
        }
        tmp += i_tmp;
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_luma_block_ext_w24_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int height, const int8_t *coef_x, const int8_t *coef_y)
{
    ALIGN32(int16_t tmp_res[(64 + 7) * 64]);
    int16_t *tmp = tmp_res;
    const int i_tmp  = 32;
    const int i_tmp2 = 2 * i_tmp;
    const int i_tmp3 = 3 * i_tmp;
    const int i_tmp4 = 4 * i_tmp;
    const int i_tmp5 = 5 * i_tmp;
    const int i_tmp6 = 6 * i_tmp;
    const int i_tmp7 = 7 * i_tmp;

    int row;
    int bsymy = (coef_y[1] == coef_y[6]);
    int shift = 12;
    __m256i mAddOffset = _mm256_set1_epi32(1 << 11);
    __m256i mCoef;
    __m256i mask24 = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0);

    // HOR
    __m256i mSwitch1 = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8, 4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12);
    __m256i mSwitch2 = _mm256_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10, 6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14);
    __m256i mSwitch3 = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8);
    __m256i mSwitch4 = _mm256_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10);
    __m256i mSwitch5 = _mm256_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12, 4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12);
    __m256i mSwitch6 = _mm256_setr_epi8(6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14, 6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14);

    src -= (3 * i_src + 3);
#if ARCH_X86_64
    mCoef = _mm256_set1_epi64x(*(long long*)coef_x);
#else
    mCoef = _mm256_loadu_si256((__m256i*)coef_x);
    mCoef = _mm256_permute4x64_epi64(mCoef, 0x0);
#endif

    for (row = -3; row < height + 4; row++) {
        __m256i T0, T1, T2, T3, T4, T5, sum1, sum2;
        __m256i S0 = _mm256_loadu_si256((__m256i*)(src));
        __m256i S1 = _mm256_permute4x64_epi64(S0, 0x99);

        T0 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S1, mSwitch1), mCoef);
        T1 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S1, mSwitch2), mCoef);
        T0 = _mm256_hadd_epi16(T0, T1);

        T2 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch3), mCoef);
        T3 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch4), mCoef);
        T4 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch5), mCoef);
        T5 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch6), mCoef);

        sum1 = _mm256_hadd_epi16(_mm256_hadd_epi16(T2, T3), _mm256_hadd_epi16(T4, T5));
        sum2 = _mm256_hadd_epi16(T0, T0);

        sum2 = _mm256_permute4x64_epi64(sum2, 0xd8);
        sum2 = _mm256_permute2x128_si256(sum1, sum2, 0x13);
        _mm_storeu_si128((__m128i*)(tmp), _mm256_castsi256_si128(sum1));
        _mm256_storeu_si256((__m256i*)(tmp + 8), sum2);

        src += i_src;
        tmp += i_tmp;
    }

    // VER
    tmp = tmp_res;
    if (bsymy) {
        __m256i mCoefy1 = _mm256_set1_epi16(coef_y[0]);
        __m256i mCoefy2 = _mm256_set1_epi16(coef_y[1]);
        __m256i mCoefy3 = _mm256_set1_epi16(coef_y[2]);
        __m256i mCoefy4 = _mm256_set1_epi16(coef_y[3]);

        for (row = 0; row < height; row++) {
            __m256i mVal1, mVal2, mVal, mVal3, mVal4;
            __m256i T0, T1, T2, T3, S0, S1, S2, S3;
            __m256i T4, T5, T6, T7, S4, S5, S6, S7;
            __m256i T00, T11, T22, T33, S00, S11, S22, S33;
            __m256i T44, T55, T66, T77, S44, S55, S66, S77;

            S0 = _mm256_loadu_si256((__m256i*)(tmp));
            S1 = _mm256_loadu_si256((__m256i*)(tmp + i_tmp));
            S2 = _mm256_loadu_si256((__m256i*)(tmp + i_tmp2));
            S3 = _mm256_loadu_si256((__m256i*)(tmp + i_tmp3));
            S4 = _mm256_loadu_si256((__m256i*)(tmp + i_tmp4));
            S5 = _mm256_loadu_si256((__m256i*)(tmp + i_tmp5));
            S6 = _mm256_loadu_si256((__m256i*)(tmp + i_tmp6));
            S7 = _mm256_loadu_si256((__m256i*)(tmp + i_tmp7));

            S00 = _mm256_loadu_si256((__m256i*)(tmp + 16));
            S11 = _mm256_loadu_si256((__m256i*)(tmp + 16 + i_tmp));
            S22 = _mm256_loadu_si256((__m256i*)(tmp + 16 + i_tmp2));
            S33 = _mm256_loadu_si256((__m256i*)(tmp + 16 + i_tmp3));
            S44 = _mm256_loadu_si256((__m256i*)(tmp + 16 + i_tmp4));
            S55 = _mm256_loadu_si256((__m256i*)(tmp + 16 + i_tmp5));
            S66 = _mm256_loadu_si256((__m256i*)(tmp + 16 + i_tmp6));
            S77 = _mm256_loadu_si256((__m256i*)(tmp + 16 + i_tmp7));

            T0 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S0, S7), mCoefy1);
            T1 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S1, S6), mCoefy2);
            T2 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S2, S5), mCoefy3);
            T3 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S3, S4), mCoefy4);
            T4 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S0, S7), mCoefy1);
            T5 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S1, S6), mCoefy2);
            T6 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S2, S5), mCoefy3);
            T7 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S3, S4), mCoefy4);

            T00 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S00, S77), mCoefy1);
            T11 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S11, S66), mCoefy2);
            T22 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S22, S55), mCoefy3);
            T33 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S33, S44), mCoefy4);
            T44 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S00, S77), mCoefy1);
            T55 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S11, S66), mCoefy2);
            T66 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S22, S55), mCoefy3);
            T77 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S33, S44), mCoefy4);

            mVal1 = _mm256_add_epi32(_mm256_add_epi32(T0, T1), _mm256_add_epi32(T2, T3));
            mVal2 = _mm256_add_epi32(_mm256_add_epi32(T4, T5), _mm256_add_epi32(T6, T7));

            mVal3 = _mm256_add_epi32(_mm256_add_epi32(T00, T11), _mm256_add_epi32(T22, T33));
            mVal4 = _mm256_add_epi32(_mm256_add_epi32(T44, T55), _mm256_add_epi32(T66, T77));

            mVal1 = _mm256_srai_epi32(_mm256_add_epi32(mVal1, mAddOffset), shift);
            mVal2 = _mm256_srai_epi32(_mm256_add_epi32(mVal2, mAddOffset), shift);
            mVal3 = _mm256_srai_epi32(_mm256_add_epi32(mVal3, mAddOffset), shift);
            mVal4 = _mm256_srai_epi32(_mm256_add_epi32(mVal4, mAddOffset), shift);

            mVal = _mm256_packus_epi16(_mm256_packs_epi32(mVal1, mVal2), _mm256_packs_epi32(mVal3, mVal4));

            mVal = _mm256_permute4x64_epi64(mVal, 0xd8);
            _mm256_maskstore_epi32((int*)(dst), mask24, mVal);

            tmp += i_tmp;
            dst += i_dst;
        }
    } else {
        __m256i mCoefy1 = _mm256_cvtepi8_epi16(_mm_set1_epi16(*(int16_t*)(coef_y)));
        __m256i mCoefy2 = _mm256_cvtepi8_epi16(_mm_set1_epi16(*(int16_t*)(coef_y + 2)));
        __m256i mCoefy3 = _mm256_cvtepi8_epi16(_mm_set1_epi16(*(int16_t*)(coef_y + 4)));
        __m256i mCoefy4 = _mm256_cvtepi8_epi16(_mm_set1_epi16(*(int16_t*)(coef_y + 6)));

        for (row = 0; row < height; row++) {
            __m256i mVal1, mVal2, mVal, mVal3, mVal4;
            __m256i T0, T1, T2, T3, S0, S1, S2, S3;
            __m256i T4, T5, T6, T7, S4, S5, S6, S7;
            __m256i T00, T11, T22, T33, S00, S11, S22, S33;
            __m256i T44, T55, T66, T77, S44, S55, S66, S77;

            S0 = _mm256_loadu_si256((__m256i*)(tmp));
            S1 = _mm256_loadu_si256((__m256i*)(tmp + i_tmp));
            S2 = _mm256_loadu_si256((__m256i*)(tmp + i_tmp2));
            S3 = _mm256_loadu_si256((__m256i*)(tmp + i_tmp3));
            S4 = _mm256_loadu_si256((__m256i*)(tmp + i_tmp4));
            S5 = _mm256_loadu_si256((__m256i*)(tmp + i_tmp5));
            S6 = _mm256_loadu_si256((__m256i*)(tmp + i_tmp6));
            S7 = _mm256_loadu_si256((__m256i*)(tmp + i_tmp7));

            S00 = _mm256_loadu_si256((__m256i*)(tmp + 16));
            S11 = _mm256_loadu_si256((__m256i*)(tmp + 16 + i_tmp));
            S22 = _mm256_loadu_si256((__m256i*)(tmp + 16 + i_tmp2));
            S33 = _mm256_loadu_si256((__m256i*)(tmp + 16 + i_tmp3));
            S44 = _mm256_loadu_si256((__m256i*)(tmp + 16 + i_tmp4));
            S55 = _mm256_loadu_si256((__m256i*)(tmp + 16 + i_tmp5));
            S66 = _mm256_loadu_si256((__m256i*)(tmp + 16 + i_tmp6));
            S77 = _mm256_loadu_si256((__m256i*)(tmp + 16 + i_tmp7));

            T0 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S0, S1), mCoefy1);
            T1 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S2, S3), mCoefy2);
            T2 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S4, S5), mCoefy3);
            T3 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S6, S7), mCoefy4);
            T4 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S0, S1), mCoefy1);
            T5 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S2, S3), mCoefy2);
            T6 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S4, S5), mCoefy3);
            T7 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S6, S7), mCoefy4);

            T00 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S00, S11), mCoefy1);
            T11 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S22, S33), mCoefy2);
            T22 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S44, S55), mCoefy3);
            T33 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S66, S77), mCoefy4);
            T44 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S00, S11), mCoefy1);
            T55 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S22, S33), mCoefy2);
            T66 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S44, S55), mCoefy3);
            T77 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S66, S77), mCoefy4);

            mVal1 = _mm256_add_epi32(_mm256_add_epi32(T0, T1), _mm256_add_epi32(T2, T3));
            mVal2 = _mm256_add_epi32(_mm256_add_epi32(T4, T5), _mm256_add_epi32(T6, T7));

            mVal3 = _mm256_add_epi32(_mm256_add_epi32(T00, T11), _mm256_add_epi32(T22, T33));
            mVal4 = _mm256_add_epi32(_mm256_add_epi32(T44, T55), _mm256_add_epi32(T66, T77));

            mVal1 = _mm256_srai_epi32(_mm256_add_epi32(mVal1, mAddOffset), shift);
            mVal2 = _mm256_srai_epi32(_mm256_add_epi32(mVal2, mAddOffset), shift);
            mVal3 = _mm256_srai_epi32(_mm256_add_epi32(mVal3, mAddOffset), shift);
            mVal4 = _mm256_srai_epi32(_mm256_add_epi32(mVal4, mAddOffset), shift);

            mVal = _mm256_packus_epi16(_mm256_packs_epi32(mVal1, mVal2), _mm256_packs_epi32(mVal3, mVal4));

            mVal = _mm256_permute4x64_epi64(mVal, 0xd8);
            _mm256_maskstore_epi32((int*)(dst), mask24, mVal);

            tmp += i_tmp;
            dst += i_dst;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_chroma_block_hor_w16_avx2(pel_t *dst, int i_dst, const pel_t *src, int i_src, int width, int height, const int8_t *coeff)
{
    int row, col;
    const int offset = 32;
    const int shift = 6;

    __m256i mCoef = _mm256_set1_epi32(*(int32_t*)coeff);
    __m256i mSwitch1 = _mm256_setr_epi8(0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6);
    __m256i mSwitch2 = _mm256_setr_epi8(4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10);
    __m256i mAddOffset = _mm256_set1_epi16((short)offset);
    __m256i mask16 = _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0);
    src -= 1;

    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col += 16) {
            __m256i T0, T1, sum;
            __m256i S  = _mm256_loadu_si256((__m256i*)(src + col));
            __m256i S0 = _mm256_permute4x64_epi64(S, 0x94);

            T0 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch1), mCoef);
            T1 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch2), mCoef);

            sum = _mm256_srai_epi16(_mm256_add_epi16(_mm256_hadd_epi16(T0, T1), mAddOffset), shift);
            sum = _mm256_packus_epi16(sum, sum);
            sum = _mm256_permute4x64_epi64(sum, 0xd8);

            _mm256_maskstore_epi32((int*)(dst + col), mask16, sum);
        }
        src += i_src;
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_chroma_block_hor_w24_avx2(pel_t *dst, int i_dst, const pel_t *src, int i_src, int height, const int8_t *coeff)
{
    const int offset = 32;
    const int shift = 6;

    const __m256i mCoef = _mm256_set1_epi32(*(int32_t*)coeff);
    const __m256i mSwitch = _mm256_setr_epi8(0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10);
    const __m256i mSwitch1 = _mm256_setr_epi8(0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6);
    const __m256i mSwitch2 = _mm256_setr_epi8(4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10);
    const __m256i mask24 = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0);
    const __m256i mAddOffset = _mm256_set1_epi16((short)offset);
    const __m256i index = _mm256_setr_epi32(0, 1, 2, 6, 4, 5, 3, 7);

    int row;
    src -= 1;

    for (row = 0; row < height; row++) {
        __m256i T0, T1, T2, sum1, sum2;
        __m256i S  = _mm256_loadu_si256((__m256i*)(src));
        __m256i S0 = _mm256_permute4x64_epi64(S, 0x99);

        T0 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S, mSwitch1), mCoef);
        T1 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S, mSwitch2), mCoef);
        T2 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch), mCoef);

        sum1 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_hadd_epi16(T0, T1), mAddOffset), shift);
        sum2 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_hadd_epi16(T2, T2), mAddOffset), shift);

        sum1 = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(sum1, sum2), index);

        _mm256_maskstore_epi32((int*)(dst), mask24, sum1);

        src += i_src;
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_chroma_block_ver_w32_avx2(pel_t *dst, int i_dst, const pel_t *src, int i_src, int height, const int8_t *coeff)
{
    int row;
    const int offset = 32;
    const int shift = 6;
    int bsym = (coeff[1] == coeff[2]);
    __m256i mAddOffset = _mm256_set1_epi16((short)offset);
    const int i_src2 = i_src * 2;
    const int i_src3 = i_src * 3;

    src -= i_src;

    if (bsym) {
        __m256i coeff0 = _mm256_set1_epi8(coeff[0]);
        __m256i coeff1 = _mm256_set1_epi8(coeff[1]);

        for (row = 0; row < height; row++) {
            __m256i S0, S1, S2, S3;
            __m256i T0, T1, T2, T3, mVal1, mVal2;
            S0 = _mm256_loadu_si256((__m256i*)(src));
            S1 = _mm256_loadu_si256((__m256i*)(src + i_src));
            S2 = _mm256_loadu_si256((__m256i*)(src + i_src2));
            S3 = _mm256_loadu_si256((__m256i*)(src + i_src3));

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S0, S3), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S1, S2), coeff1);
            T2 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S0, S3), coeff0);
            T3 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S1, S2), coeff1);

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T0, T1), mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T2, T3), mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);
            _mm256_storeu_si256((__m256i*)(dst), mVal1);
            src += i_src;
            dst += i_dst;
        }
    } else {
        __m256i coeff0 = _mm256_set1_epi16(*(int16_t*)coeff);
        __m256i coeff1 = _mm256_set1_epi16(*(int16_t*)(coeff + 2));

        for (row = 0; row < height; row++) {
            __m256i T0, T1, T2, T3, mVal1, mVal2;
            __m256i S0 = _mm256_loadu_si256((__m256i*)(src));
            __m256i S1 = _mm256_loadu_si256((__m256i*)(src + i_src));
            __m256i S2 = _mm256_loadu_si256((__m256i*)(src + i_src2));
            __m256i S3 = _mm256_loadu_si256((__m256i*)(src + i_src3));

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S0, S1), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S2, S3), coeff1);
            T2 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S0, S1), coeff0);
            T3 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S2, S3), coeff1);

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T0, T1), mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T2, T3), mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);
            _mm256_storeu_si256((__m256i*)(dst), mVal1);

            src += i_src;
            dst += i_dst;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_chroma_block_ver_w24_avx2(pel_t *dst, int i_dst, const pel_t *src, int i_src, int height, const int8_t *coeff)
{
    int row;
    const int offset = 32;
    const int shift = 6;
    int bsym = (coeff[1] == coeff[2]);
    __m256i mAddOffset = _mm256_set1_epi16((short)offset);
    __m256i mask24 = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0);
    const int i_src2 = i_src * 2;
    const int i_src3 = i_src * 3;

    src -= i_src;

    if (bsym) {
        __m256i coeff0 = _mm256_set1_epi8(coeff[0]);
        __m256i coeff1 = _mm256_set1_epi8(coeff[1]);

        for (row = 0; row < height; row++) {
            __m256i T0, T1, T2, T3, mVal1, mVal2;
            __m256i S0 = _mm256_loadu_si256((__m256i*)(src));
            __m256i S1 = _mm256_loadu_si256((__m256i*)(src + i_src));
            __m256i S2 = _mm256_loadu_si256((__m256i*)(src + i_src2));
            __m256i S3 = _mm256_loadu_si256((__m256i*)(src + i_src3));

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S0, S3), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S1, S2), coeff1);
            T2 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S0, S3), coeff0);
            T3 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S1, S2), coeff1);

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T0, T1), mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T2, T3), mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);
            _mm256_maskstore_epi32((int*)(dst), mask24, mVal1);
            src += i_src;
            dst += i_dst;
        }
    } else {
        __m256i coeff0 = _mm256_set1_epi16(*(int16_t*)coeff);
        __m256i coeff1 = _mm256_set1_epi16(*(int16_t*)(coeff + 2));

        for (row = 0; row < height; row++) {
            __m256i T0, T1, T2, T3, mVal1, mVal2;
            __m256i S0 = _mm256_loadu_si256((__m256i*)(src));
            __m256i S1 = _mm256_loadu_si256((__m256i*)(src + i_src));
            __m256i S2 = _mm256_loadu_si256((__m256i*)(src + i_src2));
            __m256i S3 = _mm256_loadu_si256((__m256i*)(src + i_src3));

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S0, S1), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(S2, S3), coeff1);
            T2 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S0, S1), coeff0);
            T3 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(S2, S3), coeff1);

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T0, T1), mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T2, T3), mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);
            _mm256_maskstore_epi32((int*)(dst), mask24, mVal1);

            src += i_src;
            dst += i_dst;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_chroma_block_ver_w16_avx2(pel_t *dst, int i_dst, const pel_t *src, int i_src, int height, const int8_t *coeff)
{
    int row;
    const int offset = 32;
    const int shift = 6;
    int bsym = (coeff[1] == coeff[2]);
    __m256i mAddOffset = _mm256_set1_epi16((short)offset);
    const int i_src2 = i_src * 2;
    const int i_src3 = i_src * 3;
    const int i_src4 = i_src * 4;

    src -= i_src;

    if (bsym) {
        __m256i coeff0 = _mm256_set1_epi8(coeff[0]);
        __m256i coeff1 = _mm256_set1_epi8(coeff[1]);

        for (row = 0; row < height; row = row + 2) {
            __m256i T0, T1, T2, T3, mVal1, mVal2;
            __m256i R0, R1, R2, R3;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src + i_src));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + i_src2));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src + i_src3));
            __m128i S4 = _mm_loadu_si128((__m128i*)(src + i_src4));

            R0 = _mm256_set_m128i(S0, S1);
            R1 = _mm256_set_m128i(S1, S2);
            R2 = _mm256_set_m128i(S2, S3);
            R3 = _mm256_set_m128i(S3, S4);

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(R0, R3), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(R0, R3), coeff0);
            T2 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(R1, R2), coeff1);
            T3 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(R1, R2), coeff1);

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T0, T2), mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T1, T3), mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);

            _mm256_storeu2_m128i((__m128i*)dst, (__m128i*)(dst + i_dst), mVal1);

            src += 2 * i_src;
            dst += 2 * i_dst;
        }
    } else {
        __m256i coeff0 = _mm256_set1_epi16(*(int16_t*)coeff);
        __m256i coeff1 = _mm256_set1_epi16(*(int16_t*)(coeff + 2));

        for (row = 0; row < height; row = row + 2) {
            __m256i T0, T1, T2, T3, mVal1, mVal2;
            __m256i R0, R1, R2, R3;

            __m128i S0 = _mm_loadu_si128((__m128i*)(src));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src + i_src));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + i_src2));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src + i_src3));
            __m128i S4 = _mm_loadu_si128((__m128i*)(src + i_src4));

            R0 = _mm256_set_m128i(S0, S1);
            R1 = _mm256_set_m128i(S1, S2);
            R2 = _mm256_set_m128i(S2, S3);
            R3 = _mm256_set_m128i(S3, S4);

            T0 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(R0, R1), coeff0);
            T1 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(R0, R1), coeff0);
            T2 = _mm256_maddubs_epi16(_mm256_unpacklo_epi8(R2, R3), coeff1);
            T3 = _mm256_maddubs_epi16(_mm256_unpackhi_epi8(R2, R3), coeff1);

            mVal1 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T0, T2), mAddOffset), shift);
            mVal2 = _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T1, T3), mAddOffset), shift);
            mVal1 = _mm256_packus_epi16(mVal1, mVal2);

            _mm256_storeu2_m128i((__m128i*)dst, (__m128i*)(dst + i_dst), mVal1);

            src += 2 * i_src;
            dst += 2 * i_dst;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_chroma_block_ext_w16_avx2(pel_t *dst, int i_dst, const pel_t *src, int i_src, int width, int height, const int8_t *coef_x, const int8_t *coef_y)
{
    ALIGN32(int16_t tmp_res[(32 + 3) * 32]);
    int16_t *tmp = tmp_res;
    const int i_tmp = 32;
    const int i_tmp2 = 2 * i_tmp;
    const int i_tmp3 = 3 * i_tmp;
    const int shift = 12;

    int row, col;
    int bsymy = (coef_y[1] == coef_y[6]);
    __m256i mAddOffset = _mm256_set1_epi32(1 << (shift - 1));
    __m256i mCoef = _mm256_set1_epi32(*(int32_t*)coef_x);
    __m256i mSwitch1 = _mm256_setr_epi8(0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6);
    __m256i mSwitch2 = _mm256_setr_epi8(4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10);

    // HOR
    src -= (i_src + 1);

    for (row = -1; row < height + 2; row++) {
        for (col = 0; col < width; col += 16) {
            __m256i T0, T1, S, S0, sum;
            S = _mm256_loadu_si256((__m256i*)(src + col));
            S0 = _mm256_permute4x64_epi64(S, 0x94);

            T0 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch1), mCoef);
            T1 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch2), mCoef);
            sum = _mm256_hadd_epi16(T0, T1);

            _mm256_storeu_si256((__m256i*)(tmp + col), sum);
        }
        src += i_src;
        tmp += i_tmp;
    }

    // VER
    tmp = tmp_res;
    if (bsymy) {
        __m256i mCoefy1 = _mm256_set1_epi16(coef_y[0]);
        __m256i mCoefy2 = _mm256_set1_epi16(coef_y[1]);

        for (row = 0; row < height; row++) {
            for (col = 0; col < width; col += 16) {
                __m256i mVal1, mVal2, mVal;
                __m256i T0, T1, T2, T3, S0, S1, S2, S3;
                S0 = _mm256_load_si256((__m256i*)(tmp + col));
                S1 = _mm256_load_si256((__m256i*)(tmp + col + i_tmp));
                S2 = _mm256_load_si256((__m256i*)(tmp + col + i_tmp2));
                S3 = _mm256_load_si256((__m256i*)(tmp + col + i_tmp3));

                T0 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S0, S3), mCoefy1);
                T1 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S1, S2), mCoefy2);
                T2 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S0, S3), mCoefy1);
                T3 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S1, S2), mCoefy2);

                mVal1 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(T0, T1), mAddOffset), shift);
                mVal2 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(T2, T3), mAddOffset), shift);

                mVal = _mm256_packus_epi16(_mm256_packs_epi32(mVal1, mVal2), /*no-use*/mVal1);

                mVal = _mm256_permute4x64_epi64(mVal, 0xd8);
                _mm_storeu_si128((__m128i*)(dst + col), _mm256_castsi256_si128(mVal));
            }
            tmp += i_tmp;
            dst += i_dst;
        }
    } else {
        __m256i mCoefy1 = _mm256_cvtepi8_epi16(_mm_set1_epi16(*(int16_t*)coef_y));
        __m256i mCoefy2 = _mm256_cvtepi8_epi16(_mm_set1_epi16(*(int16_t*)(coef_y + 2)));

        for (row = 0; row < height; row++) {
            for (col = 0; col < width; col += 16) {
                __m256i mVal1, mVal2, mVal;
                __m256i T0, T1, T2, T3, S0, S1, S2, S3;
                S0 = _mm256_load_si256((__m256i*)(tmp + col));
                S1 = _mm256_load_si256((__m256i*)(tmp + col + i_tmp));
                S2 = _mm256_load_si256((__m256i*)(tmp + col + i_tmp2));
                S3 = _mm256_load_si256((__m256i*)(tmp + col + i_tmp3));

                T0 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S0, S1), mCoefy1);
                T1 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S2, S3), mCoefy2);
                T2 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S0, S1), mCoefy1);
                T3 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S2, S3), mCoefy2);

                mVal1 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(T0, T1), mAddOffset), shift);
                mVal2 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(T2, T3), mAddOffset), shift);

                mVal = _mm256_packus_epi16(_mm256_packs_epi32(mVal1, mVal2), /*no-use*/mVal1);

                mVal = _mm256_permute4x64_epi64(mVal, 0xd8);
                _mm_storeu_si128((__m128i*)(dst + col), _mm256_castsi256_si128(mVal));
            }
            tmp += i_tmp;
            dst += i_dst;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_chroma_block_ext_w24_avx2(pel_t *dst, int i_dst, const pel_t *src, int i_src, int width, int height, const int8_t *coef_x, const int8_t *coef_y)
{
    ALIGN32(int16_t tmp_res[(32 + 3) * 32]);
    int16_t *tmp = tmp_res;
    const int i_tmp = 32;
    const int i_tmp2 = 2 * i_tmp;
    const int i_tmp3 = 3 * i_tmp;

    int row;
    int bsymy = (coef_y[1] == coef_y[6]);
    const int shift = 12;
    __m256i mAddOffset = _mm256_set1_epi32(1 << (shift - 1));
    __m256i mCoef = _mm256_set1_epi32(*(int32_t*)coef_x);
    __m256i mSwitch = _mm256_setr_epi8(0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10);
    __m256i mSwitch1 = _mm256_setr_epi8(0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6);
    __m256i mSwitch2 = _mm256_setr_epi8(4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10);
    __m256i mask24 = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0);
    //HOR
    src = src - i_src - 1;
    UNUSED_PARAMETER(width);

    for (row = -1; row < height + 2; row++) {
        __m256i T0, T1, T2, S, S0;
        S = _mm256_loadu_si256((__m256i*)(src));
        S0 = _mm256_permute4x64_epi64(S, 0x99);

        T0 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S, mSwitch1), mCoef);
        T1 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S, mSwitch2), mCoef);
        T2 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(S0, mSwitch), mCoef);
        T0 = _mm256_hadd_epi16(T0, T1);
        T2 = _mm256_hadd_epi16(T2, T2);

        T2 = _mm256_permute4x64_epi64(T2, 0xd8);
        T2 = _mm256_permute2x128_si256(T0, T2, 0x13);
        _mm_storeu_si128((__m128i*)(tmp), _mm256_castsi256_si128(T0));
        _mm256_storeu_si256((__m256i*)(tmp + 8), T2);
        src += i_src;
        tmp += i_tmp;
    }

    // VER
    tmp = tmp_res;
    if (bsymy) {
        __m256i mCoefy1 = _mm256_set1_epi16(coef_y[0]);
        __m256i mCoefy2 = _mm256_set1_epi16(coef_y[1]);

        for (row = 0; row < height; row++) {
            __m256i mVal1, mVal2, mVal3, mVal4, mVal;
            __m256i S0, S1, S2, S3, S4, S5, S6, S7;
            __m256i T0, T1, T2, T3, T4, T5, T6, T7;

            S0 = _mm256_load_si256((__m256i*)(tmp));
            S1 = _mm256_load_si256((__m256i*)(tmp + i_tmp));
            S2 = _mm256_load_si256((__m256i*)(tmp + i_tmp2));
            S3 = _mm256_load_si256((__m256i*)(tmp + i_tmp3));

            S4 = _mm256_load_si256((__m256i*)(tmp + 16));
            S5 = _mm256_load_si256((__m256i*)(tmp + 16 + i_tmp));
            S6 = _mm256_load_si256((__m256i*)(tmp + 16 + i_tmp2));
            S7 = _mm256_load_si256((__m256i*)(tmp + 16 + i_tmp3));

            T0 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S0, S3), mCoefy1);
            T1 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S1, S2), mCoefy2);
            T2 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S0, S3), mCoefy1);
            T3 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S1, S2), mCoefy2);
            T4 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S4, S7), mCoefy1);
            T5 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S5, S6), mCoefy2);
            T6 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S4, S7), mCoefy1);
            T7 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S5, S6), mCoefy2);

            mVal1 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(T0, T1), mAddOffset), shift);
            mVal2 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(T2, T3), mAddOffset), shift);
            mVal3 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(T4, T5), mAddOffset), shift);
            mVal4 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(T6, T7), mAddOffset), shift);

            mVal = _mm256_packus_epi16(_mm256_packs_epi32(mVal1, mVal2), _mm256_packs_epi32(mVal3, mVal4));

            mVal = _mm256_permute4x64_epi64(mVal, 0xd8);
            _mm256_maskstore_epi32((int*)(dst), mask24, mVal);

            tmp += i_tmp;
            dst += i_dst;
        }
    } else {
        __m256i mCoefy1 = _mm256_cvtepi8_epi16(_mm_set1_epi16(*(int16_t*)coef_y));
        __m256i mCoefy2 = _mm256_cvtepi8_epi16(_mm_set1_epi16(*(int16_t*)(coef_y + 2)));

        for (row = 0; row < height; row++) {
            __m256i mVal1, mVal2, mVal3, mVal4, mVal;
            __m256i S0, S1, S2, S3, S4, S5, S6, S7;
            __m256i T0, T1, T2, T3, T4, T5, T6, T7;

            S0 = _mm256_load_si256((__m256i*)(tmp));
            S1 = _mm256_load_si256((__m256i*)(tmp + i_tmp));
            S2 = _mm256_load_si256((__m256i*)(tmp + i_tmp2));
            S3 = _mm256_load_si256((__m256i*)(tmp + i_tmp3));

            S4 = _mm256_load_si256((__m256i*)(tmp + 16));
            S5 = _mm256_load_si256((__m256i*)(tmp + 16 + i_tmp));
            S6 = _mm256_load_si256((__m256i*)(tmp + 16 + i_tmp2));
            S7 = _mm256_load_si256((__m256i*)(tmp + 16 + i_tmp3));

            T0 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S0, S1), mCoefy1);
            T1 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S2, S3), mCoefy2);
            T2 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S0, S1), mCoefy1);
            T3 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S2, S3), mCoefy2);
            T4 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S4, S5), mCoefy1);
            T5 = _mm256_madd_epi16(_mm256_unpacklo_epi16(S6, S7), mCoefy2);
            T6 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S4, S5), mCoefy1);
            T7 = _mm256_madd_epi16(_mm256_unpackhi_epi16(S6, S7), mCoefy2);

            mVal1 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(T0, T1), mAddOffset), shift);
            mVal2 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(T2, T3), mAddOffset), shift);
            mVal3 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(T4, T5), mAddOffset), shift);
            mVal4 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(T6, T7), mAddOffset), shift);

            mVal = _mm256_packus_epi16(_mm256_packs_epi32(mVal1, mVal2), _mm256_packs_epi32(mVal3, mVal4));

            mVal = _mm256_permute4x64_epi64(mVal, 0xd8);
            _mm256_maskstore_epi32((int*)(dst), mask24, mVal);

            tmp += i_tmp;
            dst += i_dst;
        }
    }
}

/*--------------------------------------- 插值函数 ------------------------------------------------------*/

/* ---------------------------------------------------------------------------
 */
void intpl_luma_block_hor_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff)
{
    switch (width / 4 - 1) {
    case 3:
    case 7:
    case 11:
    case 15:
        intpl_luma_block_hor_w16_avx2(dst, i_dst, src, i_src, width, height, coeff);
        break;
    case 5:
        intpl_luma_block_hor_w24_avx2(dst, i_dst, src, i_src, width, height, coeff);
        break;
    default:
        intpl_luma_block_hor_sse128(dst, i_dst, src, i_src, width, height, coeff);
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_luma_block_ver_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff)
{
    switch (width / 4 - 1) {
    case 3:
        intpl_luma_block_ver_w16_avx2(dst, i_dst, src, i_src, width, height, coeff);
        break;
    case 5:
        intpl_luma_block_ver_w24_avx2(dst, i_dst, src, i_src, width, height, coeff);
        break;
    case 7:
        intpl_luma_block_ver_w32_avx2(dst, i_dst, src, i_src, width, height, coeff);
        break;
    case 11:
        intpl_luma_block_ver_w48_avx2(dst, i_dst, src, i_src, width, height, coeff);
        break;
    case 15:
        intpl_luma_block_ver_w64_avx2(dst, i_dst, src, i_src, width, height, coeff);
        break;
    default:
        intpl_luma_block_ver_sse128(dst, i_dst, src, i_src, width, height, coeff);
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_luma_block_ext_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coef_x, const int8_t *coef_y)
{
    switch (width / 4 - 1) {
    case 3:
    case 7:
    case 11:
    case 15:
        intpl_luma_block_ext_w16_avx2(dst, i_dst, src, i_src, width, height, coef_x, coef_y);
        break;
    case 5:
        intpl_luma_block_ext_w24_avx2(dst, i_dst, src, i_src, height, coef_x, coef_y);
        break;
    default:
        intpl_luma_block_ext_sse128(dst, i_dst, src, i_src, width, height, coef_x, coef_y);
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_chroma_block_hor_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff)
{
    switch (width / 2 - 1) {
    case 7:
    case 15:
        intpl_chroma_block_hor_w16_avx2(dst, i_dst, src, i_src, width, height, coeff);
        break;
    case 11:
        intpl_chroma_block_hor_w24_avx2(dst, i_dst, src, i_src, height, coeff);
        break;
    default:
        intpl_chroma_block_hor_sse128(dst, i_dst, src, i_src, width, height, coeff);
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_chroma_block_ver_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff)
{
    switch (width / 2 - 1) {
    case 7:
        intpl_chroma_block_ver_w16_avx2(dst, i_dst, src, i_src, height, coeff);
        break;
    case 11:
        intpl_chroma_block_ver_w24_avx2(dst, i_dst, src, i_src, height, coeff);
        break;
    case 15:
        intpl_chroma_block_ver_w32_avx2(dst, i_dst, src, i_src, height, coeff);
        break;
    default:
        intpl_chroma_block_ver_sse128(dst, i_dst, src, i_src, width, height, coeff);
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_chroma_block_ext_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coef_x, const int8_t *coef_y)
{
    switch (width / 2 - 1) {
    case 7:
    case 15:
        intpl_chroma_block_ext_w16_avx2(dst, i_dst, src, i_src, width, height, coef_x, coef_y);
        break;
    case 11:
        intpl_chroma_block_ext_w24_avx2(dst, i_dst, src, i_src, width, height, coef_x, coef_y);
        break;
    default:
        intpl_chroma_block_ext_sse128(dst, i_dst, src, i_src, width, height, coef_x, coef_y);
    }
}

/* ---------------------------------------------------------------------------
 */
#define INTPL_LUMA_EXT_COMPUT(W0,W1,W2,W3,W4,W5,W6,W7,result)          \
    T0 = _mm256_madd_epi16(_mm256_unpacklo_epi16(W0, W1), mCoefy01);   \
    T1 = _mm256_madd_epi16(_mm256_unpacklo_epi16(W2, W3), mCoefy23);   \
    T2 = _mm256_madd_epi16(_mm256_unpacklo_epi16(W4, W5), mCoefy45);   \
    T3 = _mm256_madd_epi16(_mm256_unpacklo_epi16(W6, W7), mCoefy67);   \
    T4 = _mm256_madd_epi16(_mm256_unpackhi_epi16(W0, W1), mCoefy01);   \
    T5 = _mm256_madd_epi16(_mm256_unpackhi_epi16(W2, W3), mCoefy23);   \
    T6 = _mm256_madd_epi16(_mm256_unpackhi_epi16(W4, W5), mCoefy45);   \
    T7 = _mm256_madd_epi16(_mm256_unpackhi_epi16(W6, W7), mCoefy67);   \
    \
    mVal1 = _mm256_add_epi32(_mm256_add_epi32(T0, T1), _mm256_add_epi32(T2, T3));  \
    mVal2 = _mm256_add_epi32(_mm256_add_epi32(T4, T5), _mm256_add_epi32(T6, T7));  \
    \
    mVal1 = _mm256_srai_epi32(_mm256_add_epi32(mVal1, mAddOffset), shift);         \
    mVal2 = _mm256_srai_epi32(_mm256_add_epi32(mVal2, mAddOffset), shift);         \
    result = _mm256_packs_epi32(mVal1, mVal2);

#define INTPL_LUMA_EXT_STORE(a, b, c)                      \
    mVal = _mm256_permute4x64_epi64(_mm256_packus_epi16(a, b), 216);            \
    _mm256_storeu_si256((__m256i*)(c), mVal);


/* ---------------------------------------------------------------------------
 */
void intpl_luma_ext_avx2(pel_t *dst, int i_dst, int16_t *tmp, int i_tmp, int width, int height, const int8_t *coeff)
{
    const int shift = 12;
    int row, col;
    int16_t const *p;

    __m256i mAddOffset = _mm256_set1_epi32(1 << (shift - 1));

    __m256i mCoefy01 = _mm256_cvtepi8_epi16(_mm_set1_epi16(*(int16_t*)(coeff + 0)));
    __m256i mCoefy23 = _mm256_cvtepi8_epi16(_mm_set1_epi16(*(int16_t*)(coeff + 2)));
    __m256i mCoefy45 = _mm256_cvtepi8_epi16(_mm_set1_epi16(*(int16_t*)(coeff + 4)));
    __m256i mCoefy67 = _mm256_cvtepi8_epi16(_mm_set1_epi16(*(int16_t*)(coeff + 6)));

    tmp -= 3 * i_tmp;

    for (row = 0; row < height; row = row + 4) {
        __m256i T00, T10, T20, T30, T40, T50, T60, T70, T80, T90, Ta0;
        __m256i T0, T1, T2, T3, T4, T5, T6, T7;
        __m256i U0, U1, U2, U3;
        __m256i V0, V1, V2, V3;
        __m256i mVal1, mVal2, mVal;

        p = tmp;
        for (col = 0; col < width - 31; col += 32) {

            T00 = _mm256_loadu_si256((__m256i*)(p));
            T10 = _mm256_loadu_si256((__m256i*)(p + i_tmp));
            T20 = _mm256_loadu_si256((__m256i*)(p + 2 * i_tmp));
            T30 = _mm256_loadu_si256((__m256i*)(p + 3 * i_tmp));
            T40 = _mm256_loadu_si256((__m256i*)(p + 4 * i_tmp));
            T50 = _mm256_loadu_si256((__m256i*)(p + 5 * i_tmp));
            T60 = _mm256_loadu_si256((__m256i*)(p + 6 * i_tmp));
            T70 = _mm256_loadu_si256((__m256i*)(p + 7 * i_tmp));
            T80 = _mm256_loadu_si256((__m256i*)(p + 8 * i_tmp));
            T90 = _mm256_loadu_si256((__m256i*)(p + 9 * i_tmp));
            Ta0 = _mm256_loadu_si256((__m256i*)(p + 10 * i_tmp));

            INTPL_LUMA_EXT_COMPUT(T00, T10, T20, T30, T40, T50, T60, T70, U0);
            INTPL_LUMA_EXT_COMPUT(T10, T20, T30, T40, T50, T60, T70, T80, U1);
            INTPL_LUMA_EXT_COMPUT(T20, T30, T40, T50, T60, T70, T80, T90, U2);
            INTPL_LUMA_EXT_COMPUT(T30, T40, T50, T60, T70, T80, T90, Ta0, U3);

            //col + 16
            T00 = _mm256_loadu_si256((__m256i*)(p + 16));
            T10 = _mm256_loadu_si256((__m256i*)(p + 16 + i_tmp));
            T20 = _mm256_loadu_si256((__m256i*)(p + 16 + 2 * i_tmp));
            T30 = _mm256_loadu_si256((__m256i*)(p + 16 + 3 * i_tmp));
            T40 = _mm256_loadu_si256((__m256i*)(p + 16 + 4 * i_tmp));
            T50 = _mm256_loadu_si256((__m256i*)(p + 16 + 5 * i_tmp));
            T60 = _mm256_loadu_si256((__m256i*)(p + 16 + 6 * i_tmp));
            T70 = _mm256_loadu_si256((__m256i*)(p + 16 + 7 * i_tmp));
            T80 = _mm256_loadu_si256((__m256i*)(p + 16 + 8 * i_tmp));
            T90 = _mm256_loadu_si256((__m256i*)(p + 16 + 9 * i_tmp));
            Ta0 = _mm256_loadu_si256((__m256i*)(p + 16 + 10 * i_tmp));

            INTPL_LUMA_EXT_COMPUT(T00, T10, T20, T30, T40, T50, T60, T70, V0);
            INTPL_LUMA_EXT_COMPUT(T10, T20, T30, T40, T50, T60, T70, T80, V1);
            INTPL_LUMA_EXT_COMPUT(T20, T30, T40, T50, T60, T70, T80, T90, V2);
            INTPL_LUMA_EXT_COMPUT(T30, T40, T50, T60, T70, T80, T90, Ta0, V3);

            INTPL_LUMA_EXT_STORE(U0, V0, dst + col);
            INTPL_LUMA_EXT_STORE(U1, V1, dst + i_dst + col);
            INTPL_LUMA_EXT_STORE(U2, V2, dst + 2 * i_dst + col);
            INTPL_LUMA_EXT_STORE(U3, V3, dst + 3 * i_dst + col);

            p += 32;
        }

        if (col < width - 16) {
            T00 = _mm256_loadu_si256((__m256i*)(p));
            T10 = _mm256_loadu_si256((__m256i*)(p + i_tmp));
            T20 = _mm256_loadu_si256((__m256i*)(p + 2 * i_tmp));
            T30 = _mm256_loadu_si256((__m256i*)(p + 3 * i_tmp));
            T40 = _mm256_loadu_si256((__m256i*)(p + 4 * i_tmp));
            T50 = _mm256_loadu_si256((__m256i*)(p + 5 * i_tmp));
            T60 = _mm256_loadu_si256((__m256i*)(p + 6 * i_tmp));
            T70 = _mm256_loadu_si256((__m256i*)(p + 7 * i_tmp));
            T80 = _mm256_loadu_si256((__m256i*)(p + 8 * i_tmp));
            T90 = _mm256_loadu_si256((__m256i*)(p + 9 * i_tmp));
            Ta0 = _mm256_loadu_si256((__m256i*)(p + 10 * i_tmp));

            INTPL_LUMA_EXT_COMPUT(T00, T10, T20, T30, T40, T50, T60, T70, U0);
            INTPL_LUMA_EXT_COMPUT(T10, T20, T30, T40, T50, T60, T70, T80, U1);
            INTPL_LUMA_EXT_COMPUT(T20, T30, T40, T50, T60, T70, T80, T90, U2);
            INTPL_LUMA_EXT_COMPUT(T30, T40, T50, T60, T70, T80, T90, Ta0, U3);

            //col + 16
            T00 = _mm256_loadu_si256((__m256i*)(p + 16));
            T10 = _mm256_loadu_si256((__m256i*)(p + 16 + i_tmp));
            T20 = _mm256_loadu_si256((__m256i*)(p + 16 + 2 * i_tmp));
            T30 = _mm256_loadu_si256((__m256i*)(p + 16 + 3 * i_tmp));
            T40 = _mm256_loadu_si256((__m256i*)(p + 16 + 4 * i_tmp));
            T50 = _mm256_loadu_si256((__m256i*)(p + 16 + 5 * i_tmp));
            T60 = _mm256_loadu_si256((__m256i*)(p + 16 + 6 * i_tmp));
            T70 = _mm256_loadu_si256((__m256i*)(p + 16 + 7 * i_tmp));
            T80 = _mm256_loadu_si256((__m256i*)(p + 16 + 8 * i_tmp));
            T90 = _mm256_loadu_si256((__m256i*)(p + 16 + 9 * i_tmp));
            Ta0 = _mm256_loadu_si256((__m256i*)(p + 16 + 10 * i_tmp));

            INTPL_LUMA_EXT_COMPUT(T00, T10, T20, T30, T40, T50, T60, T70, V0);
            INTPL_LUMA_EXT_COMPUT(T10, T20, T30, T40, T50, T60, T70, T80, V1);
            INTPL_LUMA_EXT_COMPUT(T20, T30, T40, T50, T60, T70, T80, T90, V2);
            INTPL_LUMA_EXT_COMPUT(T30, T40, T50, T60, T70, T80, T90, Ta0, V3);

            INTPL_LUMA_EXT_STORE(U0, V0, dst + col);
            INTPL_LUMA_EXT_STORE(U1, V1, dst + i_dst + col);
            INTPL_LUMA_EXT_STORE(U2, V2, dst + 2 * i_dst + col);
            INTPL_LUMA_EXT_STORE(U3, V3, dst + 3 * i_dst + col);

            p += 32;
            col += 32;
        }

        if (col < width) {
            T00 = _mm256_loadu_si256((__m256i*)(p));
            T10 = _mm256_loadu_si256((__m256i*)(p + i_tmp));
            T20 = _mm256_loadu_si256((__m256i*)(p + 2 * i_tmp));
            T30 = _mm256_loadu_si256((__m256i*)(p + 3 * i_tmp));
            T40 = _mm256_loadu_si256((__m256i*)(p + 4 * i_tmp));
            T50 = _mm256_loadu_si256((__m256i*)(p + 5 * i_tmp));
            T60 = _mm256_loadu_si256((__m256i*)(p + 6 * i_tmp));
            T70 = _mm256_loadu_si256((__m256i*)(p + 7 * i_tmp));
            T80 = _mm256_loadu_si256((__m256i*)(p + 8 * i_tmp));
            T90 = _mm256_loadu_si256((__m256i*)(p + 9 * i_tmp));
            Ta0 = _mm256_loadu_si256((__m256i*)(p + 10 * i_tmp));

            INTPL_LUMA_EXT_COMPUT(T00, T10, T20, T30, T40, T50, T60, T70, U0);
            INTPL_LUMA_EXT_COMPUT(T10, T20, T30, T40, T50, T60, T70, T80, U1);
            INTPL_LUMA_EXT_COMPUT(T20, T30, T40, T50, T60, T70, T80, T90, U2);
            INTPL_LUMA_EXT_COMPUT(T30, T40, T50, T60, T70, T80, T90, Ta0, U3);

            INTPL_LUMA_EXT_STORE(U0, U0, dst + col);
            INTPL_LUMA_EXT_STORE(U1, U1, dst + i_dst + col);
            INTPL_LUMA_EXT_STORE(U2, U2, dst + 2 * i_dst + col);
            INTPL_LUMA_EXT_STORE(U3, U3, dst + 3 * i_dst + col);

            p += 16;
            col += 16;
        }

        tmp += i_tmp * 4;
        dst += i_dst * 4;
    }
}


/* ---------------------------------------------------------------------------
 */
void intpl_luma_ext_x3_avx2(pel_t *const dst[3], int i_dst, int16_t *tmp, int i_tmp, int width, int height, const int8_t **coeff)
{
    intpl_luma_ext_avx2(dst[0], i_dst, tmp, i_tmp, width, height, coeff[0]);
    intpl_luma_ext_avx2(dst[1], i_dst, tmp, i_tmp, width, height, coeff[1]);
    intpl_luma_ext_avx2(dst[2], i_dst, tmp, i_tmp, width, height, coeff[2]);
}

/* ---------------------------------------------------------------------------
 */
void intpl_luma_hor_avx2(pel_t *dst, int i_dst, int16_t *tmp, int i_tmp, pel_t *src, int i_src, int width, int height, int8_t const *coeff)
{
    int row, col = 0;
    const short offset = 32;
    const int shift = 6;

    __m256i mAddOffset = _mm256_set1_epi16(offset);

    __m256i mSwitch1 = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8);
    __m256i mSwitch2 = _mm256_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10);
    __m256i mSwitch3 = _mm256_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12, 4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12);
    __m256i mSwitch4 = _mm256_setr_epi8(6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14, 6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14);
    __m256i mask4 = _mm256_setr_epi16(-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i mask8 = _mm256_setr_epi16(-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i mask16 = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0);

#if ARCH_X86_64
    __m256i mCoef = _mm256_set1_epi64x(*(int64_t*)coeff);
#else
    __m256i mCoef = _mm256_loadu_si256((__m256i*)coeff);
    mCoef = _mm256_permute4x64_epi64(mCoef, 0x0);
#endif

    src -= 3;
    for (row = 0; row < height; row++) {
        __m256i srcCoeff1, srcCoeff2;
        __m256i T20, T40, T60, T80;
        __m256i sum10, sum20;

        for (col = 0; col < width - 31; col += 32) {
            srcCoeff1 = _mm256_loadu_si256((__m256i*)(src + col));
            srcCoeff2 = _mm256_loadu_si256((__m256i*)(src + col + 8));

            T20 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff1, mSwitch1), mCoef);
            T40 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff1, mSwitch2), mCoef);
            T60 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff1, mSwitch3), mCoef);
            T80 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff1, mSwitch4), mCoef);

            sum10 = _mm256_hadd_epi16(_mm256_hadd_epi16(T20, T40), _mm256_hadd_epi16(T60, T80));

            T20 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff2, mSwitch1), mCoef);
            T40 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff2, mSwitch2), mCoef);
            T60 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff2, mSwitch3), mCoef);
            T80 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff2, mSwitch4), mCoef);

            sum20 = _mm256_hadd_epi16(_mm256_hadd_epi16(T20, T40), _mm256_hadd_epi16(T60, T80));

            // store 32bit
            _mm256_storeu_si256((__m256i*)&tmp[col],      _mm256_permute2x128_si256(sum10, sum20, 32));
            _mm256_storeu_si256((__m256i*)&tmp[col + 16], _mm256_permute2x128_si256(sum10, sum20, 49));

            // store 16bit
            sum10 = _mm256_srai_epi16(_mm256_add_epi16(sum10, mAddOffset), shift);
            sum20 = _mm256_srai_epi16(_mm256_add_epi16(sum20, mAddOffset), shift);

            _mm256_storeu_si256((__m256i*)&dst[col], _mm256_packus_epi16(sum10, sum20));
        }

        // width 16
        if (col < width - 15) {
            srcCoeff1 = _mm256_loadu_si256((__m256i*)(src + col));
            srcCoeff2 = _mm256_loadu_si256((__m256i*)(src + col + 8));

            srcCoeff1 = _mm256_permute2x128_si256(srcCoeff1, srcCoeff2, 32);

            T20 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff1, mSwitch1), mCoef);
            T40 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff1, mSwitch2), mCoef);
            T60 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff1, mSwitch3), mCoef);
            T80 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff1, mSwitch4), mCoef);

            sum10 = _mm256_hadd_epi16(_mm256_hadd_epi16(T20, T40), _mm256_hadd_epi16(T60, T80));

            // store 32bit
            _mm256_storeu_si256((__m256i*)&tmp[col], sum10);

            // store 16bit
            sum10 = _mm256_srai_epi16(_mm256_add_epi16(sum10, mAddOffset), shift);
            sum10 = _mm256_permute4x64_epi64(_mm256_packus_epi16(sum10, sum10), 8);
            _mm256_maskstore_epi32((int*)&dst[col], mask16, sum10);
            col += 16;
        }

        // width 8
        if (col < width - 7) {
            srcCoeff1 = _mm256_loadu_si256((__m256i*)(src + col));

            T20 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff1, mSwitch1), mCoef);
            T40 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff1, mSwitch2), mCoef);
            T60 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff1, mSwitch3), mCoef);
            T80 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff1, mSwitch4), mCoef);

            sum10 = _mm256_hadd_epi16(_mm256_hadd_epi16(T20, T40), _mm256_hadd_epi16(T60, T80));

            // store 16bit
            _mm256_maskstore_epi32((int*)&tmp[col], mask16, sum10);

            // store 8bit
            sum10 = _mm256_srai_epi16(_mm256_add_epi16(sum10, mAddOffset), shift);
            sum10 = _mm256_packus_epi16(sum10, sum10);

            _mm256_maskstore_epi32((int*)&dst[col], mask8, sum10);
            col += 8;
        }

        if (col < width - 3) {
            srcCoeff1 = _mm256_loadu_si256((__m256i*)(src + col));

            T20 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff1, mSwitch1), mCoef);
            T40 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff1, mSwitch2), mCoef);
            T60 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff1, mSwitch3), mCoef);
            T80 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(srcCoeff1, mSwitch4), mCoef);

            sum10 = _mm256_hadd_epi16(_mm256_hadd_epi16(T20, T40), _mm256_hadd_epi16(T60, T80));

            // store 8bit
            _mm256_maskstore_epi32((int*)&tmp[col], mask8, sum10);

            // store 4bit
            sum10 = _mm256_srai_epi16(_mm256_add_epi16(sum10, mAddOffset), shift);
            sum10 = _mm256_packus_epi16(sum10, sum10);

            _mm256_maskstore_epi32((int*)&dst[col], mask4, sum10);
        }
        src += i_src;
        tmp += i_tmp;
        dst += i_dst;
    }
}


/* ---------------------------------------------------------------------------
 */
void intpl_luma_hor_x3_avx2(pel_t *const dst[3], int i_dst, mct_t *const tmp[3], int i_tmp, pel_t *src, int i_src, int width, int height, const int8_t **coeff)
{
    int row, col = 0;
    const short offset = 32;
    const int shift = 6;

    __m256i mOffset = _mm256_set1_epi16(offset);

    __m256i mSwitch1 = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8);
    __m256i mSwitch2 = _mm256_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10);
    __m256i mSwitch3 = _mm256_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12, 4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12);
    __m256i mSwitch4 = _mm256_setr_epi8(6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14, 6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14);
    __m256i mCoef0, mCoef1, mCoef2;
    mct_t *tmp0 = tmp[0];
    mct_t *tmp1 = tmp[1];
    mct_t *tmp2 = tmp[2];
    pel_t *dst0 = dst[0];
    pel_t *dst1 = dst[1];
    pel_t *dst2 = dst[2];

    __m256i mask4 = _mm256_setr_epi16(-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i mask8 = _mm256_setr_epi16(-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i mask16 = _mm256_setr_epi16(-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0);
#if ARCH_X86_64
    mCoef0 = _mm256_set1_epi64x(*(int64_t*)coeff[0]);
    mCoef1 = _mm256_set1_epi64x(*(int64_t*)coeff[1]);
    mCoef2 = _mm256_set1_epi64x(*(int64_t*)coeff[2]);
#else
    mCoef0 = _mm256_permute4x64_epi64(_mm256_loadu_si256((__m256i*)coeff[0]), 0x0);
    mCoef1 = _mm256_permute4x64_epi64(_mm256_loadu_si256((__m256i*)coeff[1]), 0x0);
    mCoef2 = _mm256_permute4x64_epi64(_mm256_loadu_si256((__m256i*)coeff[2]), 0x0);
#endif

    src -= 3;
    for (row = 0; row < height; row++) {
        __m256i srcCoeff1, srcCoeff2;
        __m256i S11, S12, S13, S14;
        __m256i S21, S22, S23, S24;
        __m256i sum10, sum20;

        for (col = 0; col < width - 31; col += 32) {
            srcCoeff1 = _mm256_loadu_si256((__m256i*)(src + col));
            srcCoeff2 = _mm256_loadu_si256((__m256i*)(src + col + 8));

            S11 = _mm256_shuffle_epi8(srcCoeff1, mSwitch1);
            S12 = _mm256_shuffle_epi8(srcCoeff1, mSwitch2);
            S13 = _mm256_shuffle_epi8(srcCoeff1, mSwitch3);
            S14 = _mm256_shuffle_epi8(srcCoeff1, mSwitch4);

            S21 = _mm256_shuffle_epi8(srcCoeff2, mSwitch1);
            S22 = _mm256_shuffle_epi8(srcCoeff2, mSwitch2);
            S23 = _mm256_shuffle_epi8(srcCoeff2, mSwitch3);
            S24 = _mm256_shuffle_epi8(srcCoeff2, mSwitch4);

#define INTPL_HOR_FLT(Coef, S1, S2, S3, S4, Res)   do { \
                __m256i T0 = _mm256_maddubs_epi16(S1, Coef); \
                __m256i T1 = _mm256_maddubs_epi16(S2, Coef); \
                __m256i T2 = _mm256_maddubs_epi16(S3, Coef); \
                __m256i T3 = _mm256_maddubs_epi16(S4, Coef); \
                Res = _mm256_hadd_epi16(_mm256_hadd_epi16(T0, T1), _mm256_hadd_epi16(T2, T3)); \
            } while (0)

            /* 1st */
            INTPL_HOR_FLT(mCoef0, S11, S12, S13, S14, sum10);
            INTPL_HOR_FLT(mCoef0, S21, S22, S23, S24, sum20);

            // store 16bit
            _mm256_storeu_si256((__m256i*)&tmp0[col],      _mm256_permute2x128_si256(sum10, sum20, 32));
            _mm256_storeu_si256((__m256i*)&tmp0[col + 16], _mm256_permute2x128_si256(sum10, sum20, 49));

            // store 8bit
            sum10 = _mm256_srai_epi16(_mm256_add_epi16(sum10, mOffset), shift);
            sum20 = _mm256_srai_epi16(_mm256_add_epi16(sum20, mOffset), shift);

            _mm256_storeu_si256((__m256i*)&dst0[col], _mm256_packus_epi16(sum10, sum20));

            /* 2nd */
            INTPL_HOR_FLT(mCoef1, S11, S12, S13, S14, sum10);
            INTPL_HOR_FLT(mCoef1, S21, S22, S23, S24, sum20);

            // store 16bit
            _mm256_storeu_si256((__m256i*)&tmp1[col], _mm256_permute2x128_si256(sum10, sum20, 32));
            _mm256_storeu_si256((__m256i*)&tmp1[col + 16], _mm256_permute2x128_si256(sum10, sum20, 49));

            // store 8bit
            sum10 = _mm256_srai_epi16(_mm256_add_epi16(sum10, mOffset), shift);
            sum20 = _mm256_srai_epi16(_mm256_add_epi16(sum20, mOffset), shift);

            _mm256_storeu_si256((__m256i*)&dst1[col], _mm256_packus_epi16(sum10, sum20));

            /* 3rd */
            INTPL_HOR_FLT(mCoef2, S11, S12, S13, S14, sum10);
            INTPL_HOR_FLT(mCoef2, S21, S22, S23, S24, sum20);

            // store 16bit
            _mm256_storeu_si256((__m256i*)&tmp2[col], _mm256_permute2x128_si256(sum10, sum20, 32));
            _mm256_storeu_si256((__m256i*)&tmp2[col + 16], _mm256_permute2x128_si256(sum10, sum20, 49));

            // store 8bit
            sum10 = _mm256_srai_epi16(_mm256_add_epi16(sum10, mOffset), shift);
            sum20 = _mm256_srai_epi16(_mm256_add_epi16(sum20, mOffset), shift);

            _mm256_storeu_si256((__m256i*)&dst2[col], _mm256_packus_epi16(sum10, sum20));
        }

        // width 16
        if (col < width - 15) {
            srcCoeff1 = _mm256_loadu_si256((__m256i*)(src + col));
            srcCoeff2 = _mm256_loadu_si256((__m256i*)(src + col + 8));

            srcCoeff1 = _mm256_permute2x128_si256(srcCoeff1, srcCoeff2, 32);
            S11 = _mm256_shuffle_epi8(srcCoeff1, mSwitch1);
            S12 = _mm256_shuffle_epi8(srcCoeff1, mSwitch2);
            S13 = _mm256_shuffle_epi8(srcCoeff1, mSwitch3);
            S14 = _mm256_shuffle_epi8(srcCoeff1, mSwitch4);

            /* 1st */
            INTPL_HOR_FLT(mCoef0, S11, S12, S13, S14, sum10);

            // store 16bit
            _mm256_storeu_si256((__m256i*)&tmp0[col], sum10);

            // store 8bit
            sum10 = _mm256_srai_epi16(_mm256_add_epi16(sum10, mOffset), shift);
            sum10 = _mm256_permute4x64_epi64(_mm256_packus_epi16(sum10, sum10), 8);
            //_mm256_storeu_si256((__m256i*)&dst0[col], sum10);
            _mm256_maskstore_epi32((int*)&dst0[col], mask16, sum10);

            /* 2nd */
            INTPL_HOR_FLT(mCoef1, S11, S12, S13, S14, sum10);

            // store 16bit
            _mm256_storeu_si256((__m256i*)&tmp1[col], sum10);

            // store 8bit
            sum10 = _mm256_srai_epi16(_mm256_add_epi16(sum10, mOffset), shift);
            sum10 = _mm256_permute4x64_epi64(_mm256_packus_epi16(sum10, sum10), 8);
            //_mm256_storeu_si256((__m256i*)&dst1[col], sum10);
            _mm256_maskstore_epi32((int*)&dst1[col], mask16, sum10);

            /* 3rd */
            INTPL_HOR_FLT(mCoef2, S11, S12, S13, S14, sum10);

            // store 16bit
            _mm256_storeu_si256((__m256i*)&tmp2[col], sum10);

            // store 8bit
            sum10 = _mm256_srai_epi16(_mm256_add_epi16(sum10, mOffset), shift);
            sum10 = _mm256_permute4x64_epi64(_mm256_packus_epi16(sum10, sum10), 8);
            //_mm256_storeu_si256((__m256i*)&dst2[col], sum10);
            _mm256_maskstore_epi32((int*)&dst2[col], mask16, sum10);
            col += 16;
        }

        // width 8
        if (col < width - 7) {
            srcCoeff1 = _mm256_loadu_si256((__m256i*)(src + col));
            S11 = _mm256_shuffle_epi8(srcCoeff1, mSwitch1);
            S12 = _mm256_shuffle_epi8(srcCoeff1, mSwitch2);
            S13 = _mm256_shuffle_epi8(srcCoeff1, mSwitch3);
            S14 = _mm256_shuffle_epi8(srcCoeff1, mSwitch4);

            /* 1st */
            INTPL_HOR_FLT(mCoef0, S11, S12, S13, S14, sum10);

            // store 16bit
            //_mm256_storeu_si256((__m256i*)&tmp0[col], sum10);
            _mm256_maskstore_epi32((int*)&tmp0[col], mask16, sum10);

            // store 8bit
            sum10 = _mm256_srai_epi16(_mm256_add_epi16(sum10, mOffset), shift);
            sum10 = _mm256_packus_epi16(sum10, sum10);

            //_mm256_storeu_si256((__m256i*)&dst0[col], sum10);
            _mm256_maskstore_epi32((int*)&dst0[col], mask8, sum10);

            /* 2nd */
            INTPL_HOR_FLT(mCoef1, S11, S12, S13, S14, sum10);

            // store 16bit
            //_mm256_storeu_si256((__m256i*)&tmp1[col], sum10);
            _mm256_maskstore_epi32((int*)&tmp1[col], mask16, sum10);

            // store 8bit
            sum10 = _mm256_srai_epi16(_mm256_add_epi16(sum10, mOffset), shift);
            sum10 = _mm256_packus_epi16(sum10, sum10);

            //_mm256_storeu_si256((__m256i*)&dst1[col], sum10);
            _mm256_maskstore_epi32((int*)&dst1[col], mask8, sum10);

            /* 3rd */
            INTPL_HOR_FLT(mCoef2, S11, S12, S13, S14, sum10);

            // store 16bit
            //_mm256_storeu_si256((__m256i*)&tmp2[col], sum10);
            _mm256_maskstore_epi32((int*)&tmp2[col], mask16, sum10);

            // store 8bit
            sum10 = _mm256_srai_epi16(_mm256_add_epi16(sum10, mOffset), shift);
            sum10 = _mm256_packus_epi16(sum10, sum10);

            //_mm256_storeu_si256((__m256i*)&dst2[col], sum10);
            _mm256_maskstore_epi32((int*)&dst2[col], mask8, sum10);
            col += 8;
        }

        // width 4
        if (col < width - 3) {
            srcCoeff1 = _mm256_loadu_si256((__m256i*)(src + col));
            S11 = _mm256_shuffle_epi8(srcCoeff1, mSwitch1);
            S12 = _mm256_shuffle_epi8(srcCoeff1, mSwitch2);
            S13 = _mm256_shuffle_epi8(srcCoeff1, mSwitch3);
            S14 = _mm256_shuffle_epi8(srcCoeff1, mSwitch4);

            /* 1st */
            INTPL_HOR_FLT(mCoef0, S11, S12, S13, S14, sum10);

            // store 8bit
            //_mm256_storeu_si256((__m256i*)&tmp0[col], sum10);
            _mm256_maskstore_epi32((int*)&tmp0[col], mask8, sum10);

            // store 4bit
            sum10 = _mm256_srai_epi16(_mm256_add_epi16(sum10, mOffset), shift);
            sum10 = _mm256_packus_epi16(sum10, sum10);

            //_mm256_storeu_si256((__m256i*)&dst0[col], sum10);
            _mm256_maskstore_epi32((int*)&dst0[col], mask4, sum10);

            /* 2nd */
            INTPL_HOR_FLT(mCoef1, S11, S12, S13, S14, sum10);

            // store 16bit
            //_mm256_storeu_si256((__m256i*)&tmp1[col], sum10);
            _mm256_maskstore_epi32((int*)&tmp1[col], mask8, sum10);

            // store 8bit
            sum10 = _mm256_srai_epi16(_mm256_add_epi16(sum10, mOffset), shift);
            sum10 = _mm256_packus_epi16(sum10, sum10);

            //_mm256_storeu_si256((__m256i*)&dst1[col], sum10);
            _mm256_maskstore_epi32((int*)&dst1[col], mask4, sum10);

            /* 3rd */
            INTPL_HOR_FLT(mCoef2, S11, S12, S13, S14, sum10);

            // store 16bit
            //_mm256_storeu_si256((__m256i*)&tmp2[col], sum10);
            _mm256_maskstore_epi32((int*)&tmp2[col], mask8, sum10);

            // store 8bit
            sum10 = _mm256_srai_epi16(_mm256_add_epi16(sum10, mOffset), shift);
            sum10 = _mm256_packus_epi16(sum10, sum10);

            //_mm256_storeu_si256((__m256i*)&dst2[col], sum10);
            _mm256_maskstore_epi32((int*)&dst2[col], mask4, sum10);
        }

        src    += i_src;
        tmp0 += i_tmp;
        tmp1 += i_tmp;
        tmp2 += i_tmp;
        dst0 += i_dst;
        dst1 += i_dst;
        dst2 += i_dst;
    }
#undef INTPL_HOR_FLT

}

/* ---------------------------------------------------------------------------
 */
#define INTPL_LUMA_VER_COMPUT(W0,W1,W2,W3,W4,W5,W6,W7,result)      \
    T0 = _mm256_maddubs_epi16(D0, W0);                  \
    T1 = _mm256_maddubs_epi16(D1, W1);                  \
    T2 = _mm256_maddubs_epi16(D2, W2);                  \
    T3 = _mm256_maddubs_epi16(D3, W3);                  \
    T4 = _mm256_maddubs_epi16(D4, W4);                  \
    T5 = _mm256_maddubs_epi16(D5, W5);                  \
    T6 = _mm256_maddubs_epi16(D6, W6);                  \
    T7 = _mm256_maddubs_epi16(D7, W7);                  \
    \
    mVal1 = _mm256_add_epi16(_mm256_add_epi16(T0, T1), _mm256_add_epi16(T2, T3));       \
    mVal2 = _mm256_add_epi16(_mm256_add_epi16(T4, T5), _mm256_add_epi16(T6, T7));       \
    \
    mVal1 = _mm256_srai_epi16(_mm256_add_epi16(mVal1, mAddOffset), shift);              \
    mVal2 = _mm256_srai_epi16(_mm256_add_epi16(mVal2, mAddOffset), shift);              \
    result = _mm256_packus_epi16(mVal1, mVal2);

#define INTPL_LUMA_VER_STORE(a, b)                         \
    _mm256_storeu_si256((__m256i*)(b), a);

#define INTPL_LUMA_VER_COMPUT_LOW(W0,W1,W2,W3,W4,W5,W6,W7,result)      \
    T0 = _mm256_maddubs_epi16(D0, W0);                  \
    T1 = _mm256_maddubs_epi16(D1, W1);                  \
    T2 = _mm256_maddubs_epi16(D2, W2);                  \
    T3 = _mm256_maddubs_epi16(D3, W3);                  \
    \
    mVal1 = _mm256_add_epi16(_mm256_add_epi16(T0, T1), _mm256_add_epi16(T2, T3));       \
    \
    mVal1 = _mm256_srai_epi16(_mm256_add_epi16(mVal1, mAddOffset), shift);              \
    result = _mm256_packus_epi16(mVal1, mVal1);

/* ---------------------------------------------------------------------------
 */
void intpl_luma_ver_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, int8_t const *coeff)
{
    int row, col;
    const short offset = 32;
    const int shift = 6;

    __m256i mAddOffset = _mm256_set1_epi16(offset);

    pel_t const *p;

    src -= 3 * i_src;

    __m256i mVal1, mVal2;

    int8_t coeff_tmp[2];
    coeff_tmp[0] = coeff[7], coeff_tmp[1] = coeff[0];
    __m256i mCoefy70 = _mm256_set1_epi16(*(short*)coeff_tmp);
    __m256i mCoefy12 = _mm256_set1_epi16(*(short*)(coeff + 1));
    __m256i mCoefy34 = _mm256_set1_epi16(*(short*)(coeff + 3));
    __m256i mCoefy56 = _mm256_set1_epi16(*(short*)(coeff + 5));

    __m256i mCoefy01 = _mm256_set1_epi16(*(short*)coeff);
    __m256i mCoefy23 = _mm256_set1_epi16(*(short*)(coeff + 2));
    __m256i mCoefy45 = _mm256_set1_epi16(*(short*)(coeff + 4));
    __m256i mCoefy67 = _mm256_set1_epi16(*(short*)(coeff + 6));

    __m256i T00, T10, T20, T30, T40, T50, T60, T70, T80, T90, Ta0;
    __m256i T0, T1, T2, T3, T4, T5, T6, T7;
    __m256i D0, D1, D2, D3, D4, D5, D6, D7;
    __m256i U0, U1, U2, U3;
    for (row = 0; row < height; row = row + 4) {
        p = src;
        for (col = 0; col < width; col += 32) {
            T00 = _mm256_loadu_si256((__m256i*)(p));
            T10 = _mm256_loadu_si256((__m256i*)(p + i_src));
            T20 = _mm256_loadu_si256((__m256i*)(p + 2 * i_src));
            T30 = _mm256_loadu_si256((__m256i*)(p + 3 * i_src));
            T40 = _mm256_loadu_si256((__m256i*)(p + 4 * i_src));
            T50 = _mm256_loadu_si256((__m256i*)(p + 5 * i_src));
            T60 = _mm256_loadu_si256((__m256i*)(p + 6 * i_src));
            T70 = _mm256_loadu_si256((__m256i*)(p + 7 * i_src));
            T80 = _mm256_loadu_si256((__m256i*)(p + 8 * i_src));
            T90 = _mm256_loadu_si256((__m256i*)(p + 9 * i_src));
            Ta0 = _mm256_loadu_si256((__m256i*)(p + 10 * i_src));

            D0 = _mm256_unpacklo_epi8(T00, T10);
            D1 = _mm256_unpacklo_epi8(T20, T30);
            D2 = _mm256_unpacklo_epi8(T40, T50);
            D3 = _mm256_unpacklo_epi8(T60, T70);
            D4 = _mm256_unpackhi_epi8(T00, T10);
            D5 = _mm256_unpackhi_epi8(T20, T30);
            D6 = _mm256_unpackhi_epi8(T40, T50);
            D7 = _mm256_unpackhi_epi8(T60, T70);


            INTPL_LUMA_VER_COMPUT(mCoefy01, mCoefy23, mCoefy45, mCoefy67, mCoefy01, mCoefy23, mCoefy45, mCoefy67, U0);
            INTPL_LUMA_VER_STORE(U0, dst + col);

            D0 = _mm256_unpacklo_epi8(T80, T10);
            D4 = _mm256_unpackhi_epi8(T80, T10);

            INTPL_LUMA_VER_COMPUT(mCoefy70, mCoefy12, mCoefy34, mCoefy56, mCoefy70, mCoefy12, mCoefy34, mCoefy56, U1);
            INTPL_LUMA_VER_STORE(U1, dst + i_dst + col);

            D0 = _mm256_unpacklo_epi8(T80, T90);
            D4 = _mm256_unpackhi_epi8(T80, T90);

            INTPL_LUMA_VER_COMPUT(mCoefy67, mCoefy01, mCoefy23, mCoefy45, mCoefy67, mCoefy01, mCoefy23, mCoefy45, U2);
            INTPL_LUMA_VER_STORE(U2, dst + 2 * i_dst + col);

            D1 = _mm256_unpacklo_epi8(Ta0, T30);
            D5 = _mm256_unpackhi_epi8(Ta0, T30);

            INTPL_LUMA_VER_COMPUT(mCoefy56, mCoefy70, mCoefy12, mCoefy34, mCoefy56, mCoefy70, mCoefy12, mCoefy34, U3);
            INTPL_LUMA_VER_STORE(U3, dst + 3 * i_dst + col);

            p += 32;
        }
        src += 4 * i_src;
        dst += 4 * i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_luma_ver_x3_avx2(pel_t *const dst[3], int i_dst, pel_t *src, int i_src, int width, int height, const int8_t **coeff)
{
    intpl_luma_ver_avx2(dst[0], i_dst, src, i_src, width, height, coeff[0]);
    intpl_luma_ver_avx2(dst[1], i_dst, src, i_src, width, height, coeff[1]);
    intpl_luma_ver_avx2(dst[2], i_dst, src, i_src, width, height, coeff[2]);
}

