/*
 * intrinsic_inter-pred.c
 *
 * Description of this file:
 *    SSE assembly functions of Inter-Prediction module of the xavs2 library
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
#include "avs2_defs.h"

/* ---------------------------------------------------------------------------
 */
void intpl_chroma_block_hor_sse128(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff)
{
    const int16_t offset = 32;
    const int shift = 6;
    int row, col;
    const __m128i mAddOffset = _mm_set1_epi16(offset);
    const __m128i mSwitch1   = _mm_setr_epi8(0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6);
    const __m128i mSwitch2   = _mm_setr_epi8(4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10);
    const __m128i mCoef      = _mm_set1_epi32(*(int*)coeff);
    const __m128i mask       = _mm_loadu_si128((__m128i*)(intrinsic_mask[(width & 7) - 1]));

    src -= 1;

    for (row = 0; row < height; row++) {
        __m128i mSrc, mT20, mT40, mVal;

        for (col = 0; col < width - 7; col += 8) {
            mSrc = _mm_loadu_si128((__m128i*)(src + col));

            mT20 = _mm_maddubs_epi16(_mm_shuffle_epi8(mSrc, mSwitch1), mCoef);
            mT40 = _mm_maddubs_epi16(_mm_shuffle_epi8(mSrc, mSwitch2), mCoef);

            mVal = _mm_hadd_epi16(mT20, mT40);
            mVal = _mm_srai_epi16(_mm_add_epi16(mVal, mAddOffset), shift);
            mVal = _mm_packus_epi16(mVal, mVal);

            _mm_storel_epi64((__m128i*)&dst[col], mVal);
        }

        if (col < width) {
            mSrc = _mm_loadu_si128((__m128i*)(src + col));

            mT20 = _mm_maddubs_epi16(_mm_shuffle_epi8(mSrc, mSwitch1), mCoef);
            mT40 = _mm_maddubs_epi16(_mm_shuffle_epi8(mSrc, mSwitch2), mCoef);

            mVal = _mm_hadd_epi16(mT20, mT40);
            mVal = _mm_srai_epi16(_mm_add_epi16(mVal, mAddOffset), shift);
            mVal = _mm_packus_epi16(mVal, mVal);

            _mm_maskmoveu_si128(mVal, mask, (char *)&dst[col]);
        }

        src += i_src;
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_luma_block_hor_sse128(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff)
{
    int row, col = 0;
    const short offset = 32;
    const int shift = 6;

    __m128i mAddOffset = _mm_set1_epi16(offset);

    __m128i mSwitch1 = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8);
    __m128i mSwitch2 = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10);
    __m128i mSwitch3 = _mm_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12);
    __m128i mSwitch4 = _mm_setr_epi8(6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14);

    __m128i mCoef = _mm_loadl_epi64((__m128i*)coeff);

    __m128i mask = _mm_loadu_si128((__m128i*)(intrinsic_mask[(width & 7) - 1]));
    mCoef = _mm_unpacklo_epi64(mCoef, mCoef);

    src -= 3;
    for (row = 0; row < height; row++) {
        __m128i srcCoeff, T20, T40, T60, T80, sum;

        for (col = 0; col < width - 7; col += 8) {
            srcCoeff = _mm_loadu_si128((__m128i*)(src + col));

            T20 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff, mSwitch1), mCoef);
            T40 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff, mSwitch2), mCoef);
            T60 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff, mSwitch3), mCoef);
            T80 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff, mSwitch4), mCoef);

            sum = _mm_hadd_epi16(_mm_hadd_epi16(T20, T40), _mm_hadd_epi16(T60, T80));
            sum = _mm_srai_epi16(_mm_add_epi16(sum, mAddOffset), shift);
            sum = _mm_packus_epi16(sum, sum);

            _mm_storel_epi64((__m128i*)&dst[col], sum);
        }

        if (col < width) {
            srcCoeff = _mm_loadu_si128((__m128i*)(src + col));

            T20 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff, mSwitch1), mCoef);
            T40 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff, mSwitch2), mCoef);
            T60 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff, mSwitch3), mCoef);
            T80 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff, mSwitch4), mCoef);

            sum = _mm_hadd_epi16(_mm_hadd_epi16(T20, T40), _mm_hadd_epi16(T60, T80));
            sum = _mm_srai_epi16(_mm_add_epi16(sum, mAddOffset), shift);
            sum = _mm_packus_epi16(sum, sum);

            _mm_maskmoveu_si128(sum, mask, (char *)&dst[col]);
        }

        src += i_src;
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_luma_hor_sse128(pel_t *dst, int i_dst, mct_t *tmp, int i_tmp, pel_t *src, int i_src, int width, int height, int8_t const *coeff)
{
    int row, col = 0;
    const short offset = 32;
    const int shift = 6;

    __m128i mAddOffset = _mm_set1_epi16(offset);

    __m128i mSwitch1 = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7,     1, 2, 3, 4, 5, 6, 7, 8);
    __m128i mSwitch2 = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9,     3, 4, 5, 6, 7, 8, 9, 10);
    __m128i mSwitch3 = _mm_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11,   5, 6, 7, 8, 9, 10, 11, 12);
    __m128i mSwitch4 = _mm_setr_epi8(6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14);

    __m128i mCoef = _mm_loadl_epi64((__m128i*)coeff);

    mCoef = _mm_unpacklo_epi64(mCoef, mCoef);

    __m128i T01, T23, T45, T67, T89, Tab, Tcd, Tef;
    __m128i S1, S2, S3, S4;
    __m128i U0, U1;
    __m128i Val1, Val2, Val;

    __m128i mask8 = _mm_loadu_si128((__m128i*)(intrinsic_mask[(width & 8) - 1]));
    __m128i mask4 = _mm_loadu_si128((__m128i*)(intrinsic_mask[(width & 4) - 1]));
    __m128i maskx = _mm_loadu_si128((__m128i*)(intrinsic_mask[((width & 7) << 1) - 1]));

    src -= 3;
    for (row = 0; row < height; row++) {
        for (col = 0; col < width - 15; col += 16) {
            __m128i srcCoeff1 = _mm_loadu_si128((__m128i*)(src + col));
            __m128i srcCoeff2 = _mm_loadu_si128((__m128i*)(src + col + 8));

            T01 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff1, mSwitch1), mCoef);
            T23 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff1, mSwitch2), mCoef);
            T45 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff1, mSwitch3), mCoef);
            T67 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff1, mSwitch4), mCoef);

            S1 = _mm_hadd_epi16(T01, T23);
            S2 = _mm_hadd_epi16(T45, T67);
            U0 = _mm_hadd_epi16(S1, S2);

            _mm_storeu_si128((__m128i*)&tmp[col], U0);

            T89 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff2, mSwitch1), mCoef);
            Tab = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff2, mSwitch2), mCoef);
            Tcd = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff2, mSwitch3), mCoef);
            Tef = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff2, mSwitch4), mCoef);

            S3 = _mm_hadd_epi16(T89, Tab);
            S4 = _mm_hadd_epi16(Tcd, Tef);
            U1 = _mm_hadd_epi16(S3, S4);

            _mm_storeu_si128((__m128i*)&tmp[col + 8], U1);


            Val1 = _mm_add_epi16(U0, mAddOffset);
            Val2 = _mm_add_epi16(U1, mAddOffset);

            Val1 = _mm_srai_epi16(Val1, shift);
            Val2 = _mm_srai_epi16(Val2, shift);

            Val = _mm_packus_epi16(Val1, Val2);

            _mm_storeu_si128((__m128i*)&dst[col], Val);
        }

        if (col < width - 7) {
            __m128i srcCoeff1 = _mm_loadu_si128((__m128i*)(src + col));

            T01 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff1, mSwitch1), mCoef);
            T23 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff1, mSwitch2), mCoef);
            T45 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff1, mSwitch3), mCoef);
            T67 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff1, mSwitch4), mCoef);

            S1 = _mm_hadd_epi16(T01, T23);
            S2 = _mm_hadd_epi16(T45, T67);
            U0 = _mm_hadd_epi16(S1, S2);

            _mm_storeu_si128((__m128i*)&tmp[col], U0);

            Val1 = _mm_add_epi16(U0, mAddOffset);
            Val1 = _mm_srai_epi16(Val1, shift);

            Val = _mm_packus_epi16(Val1, Val1);
            _mm_maskmoveu_si128(Val, mask8, (char *)&dst[col]);
            col += 8;
        }

        if (col < width - 3) {
            __m128i srcCoeff1 = _mm_loadu_si128((__m128i*)(src + col));

            T01 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff1, mSwitch1), mCoef);
            T23 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff1, mSwitch2), mCoef);
            T45 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff1, mSwitch3), mCoef);
            T67 = _mm_maddubs_epi16(_mm_shuffle_epi8(srcCoeff1, mSwitch4), mCoef);

            S1 = _mm_hadd_epi16(T01, T23);
            S2 = _mm_hadd_epi16(T45, T67);
            U0 = _mm_hadd_epi16(S1, S2);

            //_mm_store_si128((__m128i*)&tmp[col], U0);
            _mm_maskmoveu_si128(U0, maskx, (char *)&tmp[col]);

            Val1 = _mm_add_epi16(U0, mAddOffset);
            Val1 = _mm_srai_epi16(Val1, shift);

            Val = _mm_packus_epi16(Val1, Val1);
            _mm_maskmoveu_si128(Val, mask4, (char *)&dst[col]);
        }
        src += i_src;
        tmp += i_tmp;
        dst += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 * TODO: @luofl 20170827 按照 intpl_luma_hor_sse128() 改写，依次插值16列
 */
void intpl_luma_hor_x3_sse128(pel_t *const dst[3], int i_dst, mct_t *const tmp[3], int i_tmp, pel_t *src, int i_src, int width, int height, const int8_t **coeff)
{
    int row, col = 0;
    const short offset = 32;
    const int shift = 6;

    __m128i mAddOffset = _mm_set1_epi16(offset);

    __m128i mSwitch1 = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8);
    __m128i mSwitch2 = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10);
    __m128i mSwitch3 = _mm_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12);
    __m128i mSwitch4 = _mm_setr_epi8(6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14);

    __m128i mCoef0 = _mm_loadl_epi64((__m128i*)coeff[0]);
    __m128i mCoef1 = _mm_loadl_epi64((__m128i*)coeff[1]);
    __m128i mCoef2 = _mm_loadl_epi64((__m128i*)coeff[2]);
    mct_t *tmp0 = tmp[0];
    mct_t *tmp1 = tmp[1];
    mct_t *tmp2 = tmp[2];
    pel_t *dst0 = dst[0];
    pel_t *dst1 = dst[1];
    pel_t *dst2 = dst[2];

    __m128i mask8 = _mm_loadu_si128((__m128i*)(intrinsic_mask[(width & 8) - 1]));
    __m128i mask4 = _mm_loadu_si128((__m128i*)(intrinsic_mask[(width & 4) - 1]));
    __m128i maskx = _mm_loadu_si128((__m128i*)(intrinsic_mask[((width & 7) << 1) - 1]));
    mCoef0 = _mm_unpacklo_epi64(mCoef0, mCoef0);
    mCoef1 = _mm_unpacklo_epi64(mCoef1, mCoef1);
    mCoef2 = _mm_unpacklo_epi64(mCoef2, mCoef2);

    src -= 3;
    for (row = 0; row < height; row++) {
        __m128i TC1, TC2, TC3, TC4, TC5, TC6, TC7, TC8;
        __m128i T20, T40, T60, T80;
        __m128i sum1, sum2, val1, val2, val;
        __m128i srcCoeff1, srcCoeff2;
        for (col = 0; col < width - 15; col += 16) {
            srcCoeff1 = _mm_loadu_si128((__m128i*)(src + col));
            srcCoeff2 = _mm_loadu_si128((__m128i*)(src + col + 8));

            TC1 = _mm_shuffle_epi8(srcCoeff1, mSwitch1);
            TC2 = _mm_shuffle_epi8(srcCoeff1, mSwitch2);
            TC3 = _mm_shuffle_epi8(srcCoeff1, mSwitch3);
            TC4 = _mm_shuffle_epi8(srcCoeff1, mSwitch4);

            TC5 = _mm_shuffle_epi8(srcCoeff2, mSwitch1);
            TC6 = _mm_shuffle_epi8(srcCoeff2, mSwitch2);
            TC7 = _mm_shuffle_epi8(srcCoeff2, mSwitch3);
            TC8 = _mm_shuffle_epi8(srcCoeff2, mSwitch4);
            // First
            T20 = _mm_maddubs_epi16(TC1, mCoef0);
            T40 = _mm_maddubs_epi16(TC2, mCoef0);
            T60 = _mm_maddubs_epi16(TC3, mCoef0);
            T80 = _mm_maddubs_epi16(TC4, mCoef0);

            sum1 = _mm_hadd_epi16(_mm_hadd_epi16(T20, T40), _mm_hadd_epi16(T60, T80));

            _mm_storeu_si128((__m128i*)(&tmp0[col]), sum1);

            T20 = _mm_maddubs_epi16(TC5, mCoef0);
            T40 = _mm_maddubs_epi16(TC6, mCoef0);
            T60 = _mm_maddubs_epi16(TC7, mCoef0);
            T80 = _mm_maddubs_epi16(TC8, mCoef0);

            sum2 = _mm_hadd_epi16(_mm_hadd_epi16(T20, T40), _mm_hadd_epi16(T60, T80));

            _mm_storeu_si128((__m128i*)(&tmp0[col + 8]), sum2);

            val1 = _mm_srai_epi16(_mm_add_epi16(sum1, mAddOffset), shift);
            val2 = _mm_srai_epi16(_mm_add_epi16(sum2, mAddOffset), shift);
            val = _mm_packus_epi16(val1, val2);

            _mm_storeu_si128((__m128i*)&dst0[col], val);

            // Second
            T20 = _mm_maddubs_epi16(TC1, mCoef1);
            T40 = _mm_maddubs_epi16(TC2, mCoef1);
            T60 = _mm_maddubs_epi16(TC3, mCoef1);
            T80 = _mm_maddubs_epi16(TC4, mCoef1);

            sum1 = _mm_hadd_epi16(_mm_hadd_epi16(T20, T40), _mm_hadd_epi16(T60, T80));

            _mm_storeu_si128((__m128i*)(&tmp1[col]), sum1);

            T20 = _mm_maddubs_epi16(TC5, mCoef1);
            T40 = _mm_maddubs_epi16(TC6, mCoef1);
            T60 = _mm_maddubs_epi16(TC7, mCoef1);
            T80 = _mm_maddubs_epi16(TC8, mCoef1);

            sum2 = _mm_hadd_epi16(_mm_hadd_epi16(T20, T40), _mm_hadd_epi16(T60, T80));

            _mm_storeu_si128((__m128i*)(&tmp1[col + 8]), sum2);

            val1 = _mm_srai_epi16(_mm_add_epi16(sum1, mAddOffset), shift);
            val2 = _mm_srai_epi16(_mm_add_epi16(sum2, mAddOffset), shift);
            val = _mm_packus_epi16(val1, val2);

            _mm_storeu_si128((__m128i*)&dst1[col], val);

            // Third
            T20 = _mm_maddubs_epi16(TC1, mCoef2);
            T40 = _mm_maddubs_epi16(TC2, mCoef2);
            T60 = _mm_maddubs_epi16(TC3, mCoef2);
            T80 = _mm_maddubs_epi16(TC4, mCoef2);

            sum1 = _mm_hadd_epi16(_mm_hadd_epi16(T20, T40), _mm_hadd_epi16(T60, T80));

            _mm_storeu_si128((__m128i*)(&tmp2[col]), sum1);

            T20 = _mm_maddubs_epi16(TC5, mCoef2);
            T40 = _mm_maddubs_epi16(TC6, mCoef2);
            T60 = _mm_maddubs_epi16(TC7, mCoef2);
            T80 = _mm_maddubs_epi16(TC8, mCoef2);

            sum2 = _mm_hadd_epi16(_mm_hadd_epi16(T20, T40), _mm_hadd_epi16(T60, T80));

            _mm_storeu_si128((__m128i*)(&tmp2[col + 8]), sum2);

            val1 = _mm_srai_epi16(_mm_add_epi16(sum1, mAddOffset), shift);
            val2 = _mm_srai_epi16(_mm_add_epi16(sum2, mAddOffset), shift);
            val = _mm_packus_epi16(val1, val2);

            _mm_storeu_si128((__m128i*)&dst2[col], val);
        }

        if (col < width - 7) {
            srcCoeff1 = _mm_loadu_si128((__m128i*)(src + col));

            TC1 = _mm_shuffle_epi8(srcCoeff1, mSwitch1);
            TC2 = _mm_shuffle_epi8(srcCoeff1, mSwitch2);
            TC3 = _mm_shuffle_epi8(srcCoeff1, mSwitch3);
            TC4 = _mm_shuffle_epi8(srcCoeff1, mSwitch4);

            // First
            T20 = _mm_maddubs_epi16(TC1, mCoef0);
            T40 = _mm_maddubs_epi16(TC2, mCoef0);
            T60 = _mm_maddubs_epi16(TC3, mCoef0);
            T80 = _mm_maddubs_epi16(TC4, mCoef0);

            sum1 = _mm_hadd_epi16(_mm_hadd_epi16(T20, T40), _mm_hadd_epi16(T60, T80));

            _mm_storeu_si128((__m128i*)&tmp0[col], sum1);

            val = _mm_srai_epi16(_mm_add_epi16(sum1, mAddOffset), shift);
            val = _mm_packus_epi16(val, val);

            _mm_maskmoveu_si128(val, mask8, (char *)&dst0[col]);

            // Second
            T20 = _mm_maddubs_epi16(TC1, mCoef1);
            T40 = _mm_maddubs_epi16(TC2, mCoef1);
            T60 = _mm_maddubs_epi16(TC3, mCoef1);
            T80 = _mm_maddubs_epi16(TC4, mCoef1);

            sum1 = _mm_hadd_epi16(_mm_hadd_epi16(T20, T40), _mm_hadd_epi16(T60, T80));

            _mm_storeu_si128((__m128i*)&tmp1[col], sum1);

            val = _mm_srai_epi16(_mm_add_epi16(sum1, mAddOffset), shift);
            val = _mm_packus_epi16(val, val);

            _mm_maskmoveu_si128(val, mask8, (char *)&dst1[col]);

            // Third
            T20 = _mm_maddubs_epi16(TC1, mCoef2);
            T40 = _mm_maddubs_epi16(TC2, mCoef2);
            T60 = _mm_maddubs_epi16(TC3, mCoef2);
            T80 = _mm_maddubs_epi16(TC4, mCoef2);

            sum1 = _mm_hadd_epi16(_mm_hadd_epi16(T20, T40), _mm_hadd_epi16(T60, T80));

            _mm_storeu_si128((__m128i*)&tmp2[col], sum1);

            val = _mm_srai_epi16(_mm_add_epi16(sum1, mAddOffset), shift);
            val = _mm_packus_epi16(val, val);

            _mm_maskmoveu_si128(val, mask8, (char *)&dst2[col]);
            col += 8;
        }
        if (col < (width - 3)) {
            srcCoeff1 = _mm_loadu_si128((__m128i*)(src + col));

            TC1 = _mm_shuffle_epi8(srcCoeff1, mSwitch1);
            TC2 = _mm_shuffle_epi8(srcCoeff1, mSwitch2);
            TC3 = _mm_shuffle_epi8(srcCoeff1, mSwitch3);
            TC4 = _mm_shuffle_epi8(srcCoeff1, mSwitch4);

            // First
            T20 = _mm_maddubs_epi16(TC1, mCoef0);
            T40 = _mm_maddubs_epi16(TC2, mCoef0);
            T60 = _mm_maddubs_epi16(TC3, mCoef0);
            T80 = _mm_maddubs_epi16(TC4, mCoef0);

            sum1 = _mm_hadd_epi16(_mm_hadd_epi16(T20, T40), _mm_hadd_epi16(T60, T80));

            _mm_maskmoveu_si128(sum1, maskx, (char *)&tmp0[col]);

            val = _mm_srai_epi16(_mm_add_epi16(sum1, mAddOffset), shift);
            val = _mm_packus_epi16(val, val);

            _mm_maskmoveu_si128(val, mask4, (char *)&dst0[col]);

            // Second
            T20 = _mm_maddubs_epi16(TC1, mCoef1);
            T40 = _mm_maddubs_epi16(TC2, mCoef1);
            T60 = _mm_maddubs_epi16(TC3, mCoef1);
            T80 = _mm_maddubs_epi16(TC4, mCoef1);

            sum1 = _mm_hadd_epi16(_mm_hadd_epi16(T20, T40), _mm_hadd_epi16(T60, T80));

            _mm_maskmoveu_si128(sum1, maskx, (char *)&tmp1[col]);

            val = _mm_srai_epi16(_mm_add_epi16(sum1, mAddOffset), shift);
            val = _mm_packus_epi16(val, val);

            _mm_maskmoveu_si128(val, mask4, (char *)&dst1[col]);

            // Third
            T20 = _mm_maddubs_epi16(TC1, mCoef2);
            T40 = _mm_maddubs_epi16(TC2, mCoef2);
            T60 = _mm_maddubs_epi16(TC3, mCoef2);
            T80 = _mm_maddubs_epi16(TC4, mCoef2);

            sum1 = _mm_hadd_epi16(_mm_hadd_epi16(T20, T40), _mm_hadd_epi16(T60, T80));

            _mm_maskmoveu_si128(sum1, maskx, (char *)&tmp2[col]);

            val = _mm_srai_epi16(_mm_add_epi16(sum1, mAddOffset), shift);
            val = _mm_packus_epi16(val, val);

            _mm_maskmoveu_si128(val, mask4, (char *)&dst2[col]);
        }
        src += i_src;
        tmp0 += i_tmp;
        tmp1 += i_tmp;
        tmp2 += i_tmp;
        dst0 += i_dst;
        dst1 += i_dst;
        dst2 += i_dst;
    }
}
/* ---------------------------------------------------------------------------
 */
#define INTPL_LUMA_VER_SSE128_COMPUT(W0,W1,W2,W3,W4,W5,W6,W7,result)      \
    T0 = _mm_maddubs_epi16(D0, W0);                                \
    T1 = _mm_maddubs_epi16(D1, W1);                                \
    T2 = _mm_maddubs_epi16(D2, W2);                                \
    T3 = _mm_maddubs_epi16(D3, W3);                                \
    T4 = _mm_maddubs_epi16(D4, W4);                                \
    T5 = _mm_maddubs_epi16(D5, W5);                                \
    T6 = _mm_maddubs_epi16(D6, W6);                                \
    T7 = _mm_maddubs_epi16(D7, W7);                                \
                                                                   \
    mVal1 = _mm_add_epi16(T0, T1);                                 \
    mVal1 = _mm_add_epi16(mVal1, T2);                              \
    mVal1 = _mm_add_epi16(mVal1, T3);                              \
                                                                   \
    mVal2 = _mm_add_epi16(T4, T5);                                 \
    mVal2 = _mm_add_epi16(mVal2, T6);                              \
    mVal2 = _mm_add_epi16(mVal2, T7);                              \
                                                                   \
    mVal1 = _mm_add_epi16(mVal1, mAddOffset);                      \
    mVal2 = _mm_add_epi16(mVal2, mAddOffset);                      \
    mVal1 = _mm_srai_epi16(mVal1, shift);                          \
    mVal2 = _mm_srai_epi16(mVal2, shift);                          \
    result = _mm_packus_epi16(mVal1, mVal2);

#define INTPL_LUMA_VER_SSE128_STORE(result, store_dst)             \
    _mm_storeu_si128((__m128i*)&(store_dst)[col], result);

#define INTPL_LUMA_VER_SSE128_COMPUT_LO(W0,W1,W2,W3,result)        \
    T0 = _mm_maddubs_epi16(D0, W0);                                \
    T1 = _mm_maddubs_epi16(D1, W1);                                \
    T2 = _mm_maddubs_epi16(D2, W2);                                \
    T3 = _mm_maddubs_epi16(D3, W3);                                \
                                                                   \
    mVal1 = _mm_add_epi16(T0, T1);                                 \
    mVal1 = _mm_add_epi16(mVal1, T2);                              \
    mVal1 = _mm_add_epi16(mVal1, T3);                              \
                                                                   \
    mVal1 = _mm_add_epi16(mVal1, mAddOffset);                      \
    mVal1 = _mm_srai_epi16(mVal1, shift);                          \
    result = _mm_packus_epi16(mVal1, mVal1);


void intpl_luma_ver_sse128(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, int8_t const *coeff)
{
    int row, col;
    const short offset = 32;
    const int shift = 6;

    __m128i mAddOffset = _mm_set1_epi16(offset);

    pel_t const *p;

    src -= 3 * i_src;

    int8_t coeff_tmp[2];
    coeff_tmp[0] = coeff[7],coeff_tmp[1] = coeff[0];
    __m128i coeff70 = _mm_set1_epi16(*(short*)coeff_tmp);
    __m128i coeff12 = _mm_set1_epi16(*(short*)(coeff + 1));
    __m128i coeff34 = _mm_set1_epi16(*(short*)(coeff + 3));
    __m128i coeff56 = _mm_set1_epi16(*(short*)(coeff + 5));

    __m128i coeff01 = _mm_set1_epi16(*(short*)coeff);
    __m128i coeff23 = _mm_set1_epi16(*(short*)(coeff + 2));
    __m128i coeff45 = _mm_set1_epi16(*(short*)(coeff + 4));
    __m128i coeff67 = _mm_set1_epi16(*(short*)(coeff + 6));
    __m128i mVal1, mVal2;

    __m128i T00, T10, T20, T30, T40, T50, T60, T70, T80, T90, Ta0;
    __m128i T0, T1, T2, T3, T4, T5, T6, T7;
    __m128i D0, D1, D2, D3, D4, D5, D6, D7;
    __m128i U0, U1, U2, U3;
    for (row = 0; row < height; row = row + 4) {
        p = src;
        for (col = 0; col < width - 8; col += 16) {
            T00 = _mm_loadu_si128((__m128i*)(p));
            T10 = _mm_loadu_si128((__m128i*)(p + i_src));
            T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_src));
            T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_src));
            T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_src));
            T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_src));
            T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_src));
            T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_src));
            T80 = _mm_loadu_si128((__m128i*)(p + 8 * i_src));
            T90 = _mm_loadu_si128((__m128i*)(p + 9 * i_src));
            Ta0 = _mm_loadu_si128((__m128i*)(p + 10 * i_src));

            //0
            D0 = _mm_unpacklo_epi8(T00, T10);
            D1 = _mm_unpacklo_epi8(T20, T30);
            D2 = _mm_unpacklo_epi8(T40, T50);
            D3 = _mm_unpacklo_epi8(T60, T70);
            D4 = _mm_unpackhi_epi8(T00, T10);
            D5 = _mm_unpackhi_epi8(T20, T30);
            D6 = _mm_unpackhi_epi8(T40, T50);
            D7 = _mm_unpackhi_epi8(T60, T70);

            INTPL_LUMA_VER_SSE128_COMPUT(coeff01, coeff23, coeff45, coeff67, coeff01, coeff23, coeff45, coeff67, U0);
            INTPL_LUMA_VER_SSE128_STORE(U0, dst);

            //1
            D0 = _mm_unpacklo_epi8(T80, T10);
            D4 = _mm_unpackhi_epi8(T80, T10);

            INTPL_LUMA_VER_SSE128_COMPUT(coeff70, coeff12, coeff34, coeff56, coeff70, coeff12, coeff34, coeff56, U1);
            INTPL_LUMA_VER_SSE128_STORE(U1, dst + i_dst);

            //2
            D0 = _mm_unpacklo_epi8(T80, T90);
            D4 = _mm_unpackhi_epi8(T80, T90);

            INTPL_LUMA_VER_SSE128_COMPUT(coeff67, coeff01, coeff23, coeff45, coeff67, coeff01, coeff23, coeff45, U2);
            INTPL_LUMA_VER_SSE128_STORE(U2, dst + 2 * i_dst);

            //3
            D1 = _mm_unpacklo_epi8(Ta0, T30);
            D5 = _mm_unpackhi_epi8(Ta0, T30);

            INTPL_LUMA_VER_SSE128_COMPUT(coeff56, coeff70, coeff12, coeff34, coeff56, coeff70, coeff12, coeff34, U3);
            INTPL_LUMA_VER_SSE128_STORE(U3, dst + 3 * i_dst);

            p += 16;
        }

        //<=8bit
        if (col < width) {
            T00 = _mm_loadu_si128((__m128i*)(p));
            T10 = _mm_loadu_si128((__m128i*)(p + i_src));
            T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_src));
            T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_src));
            T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_src));
            T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_src));
            T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_src));
            T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_src));
            T80 = _mm_loadu_si128((__m128i*)(p + 8 * i_src));
            T90 = _mm_loadu_si128((__m128i*)(p + 9 * i_src));
            Ta0 = _mm_loadu_si128((__m128i*)(p + 10 * i_src));

            //0
            D0 = _mm_unpacklo_epi8(T00, T10);
            D1 = _mm_unpacklo_epi8(T20, T30);
            D2 = _mm_unpacklo_epi8(T40, T50);
            D3 = _mm_unpacklo_epi8(T60, T70);

            INTPL_LUMA_VER_SSE128_COMPUT_LO(coeff01, coeff23, coeff45, coeff67, U0);
            INTPL_LUMA_VER_SSE128_STORE(U0, dst);

            //1
            D0 = _mm_unpacklo_epi8(T80, T10);

            INTPL_LUMA_VER_SSE128_COMPUT_LO(coeff70, coeff12, coeff34, coeff56, U1);
            INTPL_LUMA_VER_SSE128_STORE(U1, dst + i_dst);

            //2
            D0 = _mm_unpacklo_epi8(T80, T90);

            INTPL_LUMA_VER_SSE128_COMPUT_LO(coeff67, coeff01, coeff23, coeff45, U2);
            INTPL_LUMA_VER_SSE128_STORE(U2, dst + 2 * i_dst);

            //3
            D1 = _mm_unpacklo_epi8(Ta0, T30);

            INTPL_LUMA_VER_SSE128_COMPUT_LO(coeff56, coeff70, coeff12, coeff34, U3);
            INTPL_LUMA_VER_SSE128_STORE(U3, dst + 3 * i_dst);

            p += 8;
            col += 8;
        }

        src += i_src * 4;
        dst += i_dst * 4;
    }
}

/* ---------------------------------------------------------------------------
 *
 */
void intpl_luma_ver_x3_sse128(pel_t *const dst[3], int i_dst, pel_t *src, int i_src, int width, int height, int8_t const **coeff)
{
    /*
    intpl_luma_ver_sse128(dst0, i_dst, src, i_src, width, height, coeff[0]);
    intpl_luma_ver_sse128(dst1, i_dst, src, i_src, width, height, coeff[1]);
    intpl_luma_ver_sse128(dst2, i_dst, src, i_src, width, height, coeff[2]);
    */
    int row, col;
    const short offset = 32;
    const int shift = 6;
    int bsymFirst = (coeff[0][1] == coeff[0][6]);
    int bsymSecond = (coeff[1][1] == coeff[1][6]);
    int bsymThird = (coeff[2][1] == coeff[2][6]);

    __m128i mAddOffset = _mm_set1_epi16(offset);

    pel_t const *p;

    __m128i mask = _mm_loadu_si128((__m128i*)(intrinsic_mask[(width & 7) - 1]));

    src -= 3 * i_src;

    __m128i coeffFirst0, coeffFirst1, coeffFirst2, coeffFirst3;
    __m128i coeffSecond0, coeffSecond1, coeffSecond2, coeffSecond3;
    __m128i coeffThird0, coeffThird1, coeffThird2, coeffThird3;
    __m128i tempT00, tempT10, tempT20, tempT30;
    __m128i mVal;

    pel_t *dst0 = dst[0];
    pel_t *dst1 = dst[1];
    pel_t *dst2 = dst[2];

    //load Coefficient
    if (bsymFirst) {
        coeffFirst0 = _mm_set1_epi8(coeff[0][0]);
        coeffFirst1 = _mm_set1_epi8(coeff[0][1]);
        coeffFirst2 = _mm_set1_epi8(coeff[0][2]);
        coeffFirst3 = _mm_set1_epi8(coeff[0][3]);
    } else {
        coeffFirst0 = _mm_set1_epi16(*(short*)coeff[0]);
        coeffFirst1 = _mm_set1_epi16(*(short*)(coeff[0] + 2));
        coeffFirst2 = _mm_set1_epi16(*(short*)(coeff[0] + 4));
        coeffFirst3 = _mm_set1_epi16(*(short*)(coeff[0] + 6));
    }
    if (bsymSecond) {
        coeffSecond0 = _mm_set1_epi8(coeff[1][0]);
        coeffSecond1 = _mm_set1_epi8(coeff[1][1]);
        coeffSecond2 = _mm_set1_epi8(coeff[1][2]);
        coeffSecond3 = _mm_set1_epi8(coeff[1][3]);
    } else {
        coeffSecond0 = _mm_set1_epi16(*(short*)coeff[1]);
        coeffSecond1 = _mm_set1_epi16(*(short*)(coeff[1] + 2));
        coeffSecond2 = _mm_set1_epi16(*(short*)(coeff[1] + 4));
        coeffSecond3 = _mm_set1_epi16(*(short*)(coeff[1] + 6));
    }
    if (bsymThird) {
        coeffThird0 = _mm_set1_epi8(coeff[2][0]);
        coeffThird1 = _mm_set1_epi8(coeff[2][1]);
        coeffThird2 = _mm_set1_epi8(coeff[2][2]);
        coeffThird3 = _mm_set1_epi8(coeff[2][3]);
    } else {
        coeffThird0 = _mm_set1_epi16(*(short*)coeff[2]);
        coeffThird1 = _mm_set1_epi16(*(short*)(coeff[2] + 2));
        coeffThird2 = _mm_set1_epi16(*(short*)(coeff[2] + 4));
        coeffThird3 = _mm_set1_epi16(*(short*)(coeff[2] + 6));
    }

    //Double For
    for (row = 0; row < height; row++) {
        p = src;
        for (col = 0; col < width - 7; col += 8) {
            __m128i T00 = _mm_loadu_si128((__m128i*)(p));
            __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_src));
            __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_src));
            __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_src));
            __m128i T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_src));
            __m128i T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_src));
            __m128i T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_src));
            __m128i T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_src));

            //First
            if (bsymFirst) {
                tempT00 = _mm_unpacklo_epi8(T00, T70);
                tempT10 = _mm_unpacklo_epi8(T10, T60);
                tempT20 = _mm_unpacklo_epi8(T20, T50);
                tempT30 = _mm_unpacklo_epi8(T30, T40);
            } else {
                tempT00 = _mm_unpacklo_epi8(T00, T10);
                tempT10 = _mm_unpacklo_epi8(T20, T30);
                tempT20 = _mm_unpacklo_epi8(T40, T50);
                tempT30 = _mm_unpacklo_epi8(T60, T70);
            }
            tempT00 = _mm_maddubs_epi16(tempT00, coeffFirst0);
            tempT10 = _mm_maddubs_epi16(tempT10, coeffFirst1);
            tempT20 = _mm_maddubs_epi16(tempT20, coeffFirst2);
            tempT30 = _mm_maddubs_epi16(tempT30, coeffFirst3);

            mVal = _mm_add_epi16(tempT00, tempT10);
            mVal = _mm_add_epi16(mVal, tempT20);
            mVal = _mm_add_epi16(mVal, tempT30);

            mVal = _mm_add_epi16(mVal, mAddOffset);
            mVal = _mm_srai_epi16(mVal, shift);
            mVal = _mm_packus_epi16(mVal, mVal);

            _mm_storel_epi64((__m128i*)&dst0[col], mVal);

            //Second
            if (bsymSecond) {
                tempT00 = _mm_unpacklo_epi8(T00, T70);
                tempT10 = _mm_unpacklo_epi8(T10, T60);
                tempT20 = _mm_unpacklo_epi8(T20, T50);
                tempT30 = _mm_unpacklo_epi8(T30, T40);
            } else {
                tempT00 = _mm_unpacklo_epi8(T00, T10);
                tempT10 = _mm_unpacklo_epi8(T20, T30);
                tempT20 = _mm_unpacklo_epi8(T40, T50);
                tempT30 = _mm_unpacklo_epi8(T60, T70);
            }
            tempT00 = _mm_maddubs_epi16(tempT00, coeffSecond0);
            tempT10 = _mm_maddubs_epi16(tempT10, coeffSecond1);
            tempT20 = _mm_maddubs_epi16(tempT20, coeffSecond2);
            tempT30 = _mm_maddubs_epi16(tempT30, coeffSecond3);

            mVal = _mm_add_epi16(tempT00, tempT10);
            mVal = _mm_add_epi16(mVal, tempT20);
            mVal = _mm_add_epi16(mVal, tempT30);

            mVal = _mm_add_epi16(mVal, mAddOffset);
            mVal = _mm_srai_epi16(mVal, shift);
            mVal = _mm_packus_epi16(mVal, mVal);

            _mm_storel_epi64((__m128i*)&dst1[col], mVal);

            //Third
            if (bsymThird) {
                tempT00 = _mm_unpacklo_epi8(T00, T70);
                tempT10 = _mm_unpacklo_epi8(T10, T60);
                tempT20 = _mm_unpacklo_epi8(T20, T50);
                tempT30 = _mm_unpacklo_epi8(T30, T40);
            } else {
                tempT00 = _mm_unpacklo_epi8(T00, T10);
                tempT10 = _mm_unpacklo_epi8(T20, T30);
                tempT20 = _mm_unpacklo_epi8(T40, T50);
                tempT30 = _mm_unpacklo_epi8(T60, T70);
            }
            tempT00 = _mm_maddubs_epi16(tempT00, coeffThird0);
            tempT10 = _mm_maddubs_epi16(tempT10, coeffThird1);
            tempT20 = _mm_maddubs_epi16(tempT20, coeffThird2);
            tempT30 = _mm_maddubs_epi16(tempT30, coeffThird3);

            mVal = _mm_add_epi16(tempT00, tempT10);
            mVal = _mm_add_epi16(mVal, tempT20);
            mVal = _mm_add_epi16(mVal, tempT30);

            mVal = _mm_add_epi16(mVal, mAddOffset);
            mVal = _mm_srai_epi16(mVal, shift);
            mVal = _mm_packus_epi16(mVal, mVal);

            _mm_storel_epi64((__m128i*)&dst2[col], mVal);

            p += 8;
        }

        if (col < width) { // store either 1, 2, 3, 4, 5, 6, or 7 8-bit results in dst
            __m128i T00 = _mm_loadu_si128((__m128i*)(p));
            __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_src));
            __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_src));
            __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_src));
            __m128i T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_src));
            __m128i T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_src));
            __m128i T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_src));
            __m128i T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_src));

            //First
            if (bsymFirst) {
                tempT00 = _mm_unpacklo_epi8(T00, T70);
                tempT10 = _mm_unpacklo_epi8(T10, T60);
                tempT20 = _mm_unpacklo_epi8(T20, T50);
                tempT30 = _mm_unpacklo_epi8(T30, T40);
            } else {
                tempT00 = _mm_unpacklo_epi8(T00, T10);
                tempT10 = _mm_unpacklo_epi8(T20, T30);
                tempT20 = _mm_unpacklo_epi8(T40, T50);
                tempT30 = _mm_unpacklo_epi8(T60, T70);
            }
            tempT00 = _mm_maddubs_epi16(tempT00, coeffFirst0);
            tempT10 = _mm_maddubs_epi16(tempT10, coeffFirst1);
            tempT20 = _mm_maddubs_epi16(tempT20, coeffFirst2);
            tempT30 = _mm_maddubs_epi16(tempT30, coeffFirst3);

            mVal = _mm_add_epi16(tempT00, tempT10);
            mVal = _mm_add_epi16(mVal, tempT20);
            mVal = _mm_add_epi16(mVal, tempT30);

            mVal = _mm_add_epi16(mVal, mAddOffset);
            mVal = _mm_srai_epi16(mVal, shift);
            mVal = _mm_packus_epi16(mVal, mVal);

            _mm_maskmoveu_si128(mVal, mask, (char *)&dst0[col]);

            //Second
            if (bsymSecond) {
                tempT00 = _mm_unpacklo_epi8(T00, T70);
                tempT10 = _mm_unpacklo_epi8(T10, T60);
                tempT20 = _mm_unpacklo_epi8(T20, T50);
                tempT30 = _mm_unpacklo_epi8(T30, T40);
            } else {
                tempT00 = _mm_unpacklo_epi8(T00, T10);
                tempT10 = _mm_unpacklo_epi8(T20, T30);
                tempT20 = _mm_unpacklo_epi8(T40, T50);
                tempT30 = _mm_unpacklo_epi8(T60, T70);
            }
            tempT00 = _mm_maddubs_epi16(tempT00, coeffSecond0);
            tempT10 = _mm_maddubs_epi16(tempT10, coeffSecond1);
            tempT20 = _mm_maddubs_epi16(tempT20, coeffSecond2);
            tempT30 = _mm_maddubs_epi16(tempT30, coeffSecond3);

            mVal = _mm_add_epi16(tempT00, tempT10);
            mVal = _mm_add_epi16(mVal, tempT20);
            mVal = _mm_add_epi16(mVal, tempT30);

            mVal = _mm_add_epi16(mVal, mAddOffset);
            mVal = _mm_srai_epi16(mVal, shift);
            mVal = _mm_packus_epi16(mVal, mVal);

            _mm_maskmoveu_si128(mVal, mask, (char *)&dst1[col]);

            //Third
            if (bsymThird) {
                tempT00 = _mm_unpacklo_epi8(T00, T70);
                tempT10 = _mm_unpacklo_epi8(T10, T60);
                tempT20 = _mm_unpacklo_epi8(T20, T50);
                tempT30 = _mm_unpacklo_epi8(T30, T40);
            } else {
                tempT00 = _mm_unpacklo_epi8(T00, T10);
                tempT10 = _mm_unpacklo_epi8(T20, T30);
                tempT20 = _mm_unpacklo_epi8(T40, T50);
                tempT30 = _mm_unpacklo_epi8(T60, T70);
            }
            tempT00 = _mm_maddubs_epi16(tempT00, coeffThird0);
            tempT10 = _mm_maddubs_epi16(tempT10, coeffThird1);
            tempT20 = _mm_maddubs_epi16(tempT20, coeffThird2);
            tempT30 = _mm_maddubs_epi16(tempT30, coeffThird3);

            mVal = _mm_add_epi16(tempT00, tempT10);
            mVal = _mm_add_epi16(mVal, tempT20);
            mVal = _mm_add_epi16(mVal, tempT30);

            mVal = _mm_add_epi16(mVal, mAddOffset);
            mVal = _mm_srai_epi16(mVal, shift);
            mVal = _mm_packus_epi16(mVal, mVal);

            _mm_maskmoveu_si128(mVal, mask, (char *)&dst2[col]);
        }

        src += i_src;
        dst0 += i_dst;
        dst1 += i_dst;
        dst2 += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_luma_ext_sse128(pel_t *dst, int i_dst, mct_t *tmp, int i_tmp, int width, int height, const int8_t *coeff)
{
    int row, col;
    int shift;
    int16_t const *p;
    int bsymy = (coeff[1] == coeff[6]);

    __m128i mAddOffset;
    __m128i mask = _mm_loadu_si128((__m128i*)(intrinsic_mask[(width & 7) - 1]));

    // VER
    shift = 12;
    mAddOffset = _mm_set1_epi32(1 << (shift - 1));
    tmp = tmp - 3 * i_tmp;
    if (bsymy) {
        __m128i mCoefy1 = _mm_set1_epi16(coeff[0]);
        __m128i mCoefy2 = _mm_set1_epi16(coeff[1]);
        __m128i mCoefy3 = _mm_set1_epi16(coeff[2]);
        __m128i mCoefy4 = _mm_set1_epi16(coeff[3]);
        __m128i mVal1, mVal2, mVal;

        for (row = 0; row < height; row++) {
            p = tmp;
            for (col = 0; col < width - 7; col += 8) {
                __m128i T00 = _mm_loadu_si128((__m128i*)(p));
                __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_tmp));
                __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_tmp));
                __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_tmp));
                __m128i T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_tmp));
                __m128i T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_tmp));
                __m128i T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_tmp));
                __m128i T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_tmp));

                __m128i T0 = _mm_unpacklo_epi16(T00, T70);
                __m128i T1 = _mm_unpacklo_epi16(T10, T60);
                __m128i T2 = _mm_unpacklo_epi16(T20, T50);
                __m128i T3 = _mm_unpacklo_epi16(T30, T40);
                __m128i T4 = _mm_unpackhi_epi16(T00, T70);
                __m128i T5 = _mm_unpackhi_epi16(T10, T60);
                __m128i T6 = _mm_unpackhi_epi16(T20, T50);
                __m128i T7 = _mm_unpackhi_epi16(T30, T40);

                T0 = _mm_madd_epi16(T0, mCoefy1);
                T1 = _mm_madd_epi16(T1, mCoefy2);
                T2 = _mm_madd_epi16(T2, mCoefy3);
                T3 = _mm_madd_epi16(T3, mCoefy4);
                T4 = _mm_madd_epi16(T4, mCoefy1);
                T5 = _mm_madd_epi16(T5, mCoefy2);
                T6 = _mm_madd_epi16(T6, mCoefy3);
                T7 = _mm_madd_epi16(T7, mCoefy4);

                mVal1 = _mm_add_epi32(T0, T1);
                mVal1 = _mm_add_epi32(mVal1, T2);
                mVal1 = _mm_add_epi32(mVal1, T3);

                mVal2 = _mm_add_epi32(T4, T5);
                mVal2 = _mm_add_epi32(mVal2, T6);
                mVal2 = _mm_add_epi32(mVal2, T7);

                mVal1 = _mm_add_epi32(mVal1, mAddOffset);
                mVal2 = _mm_add_epi32(mVal2, mAddOffset);
                mVal1 = _mm_srai_epi32(mVal1, shift);
                mVal2 = _mm_srai_epi32(mVal2, shift);
                mVal = _mm_packs_epi32(mVal1, mVal2);
                mVal = _mm_packus_epi16(mVal, mVal);

                _mm_storel_epi64((__m128i*)&dst[col], mVal);

                p += 8;
            }

            if (col < width) { // store either 1, 2, 3, 4, 5, 6, or 7 8-bit results in dst
                __m128i T00 = _mm_loadu_si128((__m128i*)(p));
                __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_tmp));
                __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_tmp));
                __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_tmp));
                __m128i T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_tmp));
                __m128i T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_tmp));
                __m128i T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_tmp));
                __m128i T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_tmp));

                __m128i T0 = _mm_unpacklo_epi16(T00, T70);
                __m128i T1 = _mm_unpacklo_epi16(T10, T60);
                __m128i T2 = _mm_unpacklo_epi16(T20, T50);
                __m128i T3 = _mm_unpacklo_epi16(T30, T40);
                __m128i T4 = _mm_unpackhi_epi16(T00, T70);
                __m128i T5 = _mm_unpackhi_epi16(T10, T60);
                __m128i T6 = _mm_unpackhi_epi16(T20, T50);
                __m128i T7 = _mm_unpackhi_epi16(T30, T40);

                T0 = _mm_madd_epi16(T0, mCoefy1);
                T1 = _mm_madd_epi16(T1, mCoefy2);
                T2 = _mm_madd_epi16(T2, mCoefy3);
                T3 = _mm_madd_epi16(T3, mCoefy4);
                T4 = _mm_madd_epi16(T4, mCoefy1);
                T5 = _mm_madd_epi16(T5, mCoefy2);
                T6 = _mm_madd_epi16(T6, mCoefy3);
                T7 = _mm_madd_epi16(T7, mCoefy4);

                mVal1 = _mm_add_epi32(T0, T1);
                mVal1 = _mm_add_epi32(mVal1, T2);
                mVal1 = _mm_add_epi32(mVal1, T3);

                mVal2 = _mm_add_epi32(T4, T5);
                mVal2 = _mm_add_epi32(mVal2, T6);
                mVal2 = _mm_add_epi32(mVal2, T7);

                mVal1 = _mm_add_epi32(mVal1, mAddOffset);
                mVal2 = _mm_add_epi32(mVal2, mAddOffset);
                mVal1 = _mm_srai_epi32(mVal1, shift);
                mVal2 = _mm_srai_epi32(mVal2, shift);
                mVal = _mm_packs_epi32(mVal1, mVal2);
                mVal = _mm_packus_epi16(mVal, mVal);

                _mm_maskmoveu_si128(mVal, mask, (char *)&dst[col]);
            }
            tmp += i_tmp;
            dst += i_dst;
        }
    } else {
        __m128i mCoefy1 = _mm_set1_epi16(*(int16_t*)(coeff + 0));
        __m128i mCoefy2 = _mm_set1_epi16(*(int16_t*)(coeff + 2));
        __m128i mCoefy3 = _mm_set1_epi16(*(int16_t*)(coeff + 4));
        __m128i mCoefy4 = _mm_set1_epi16(*(int16_t*)(coeff + 6));
        __m128i mVal1, mVal2, mVal;
        mCoefy1 = _mm_cvtepi8_epi16(mCoefy1);
        mCoefy2 = _mm_cvtepi8_epi16(mCoefy2);
        mCoefy3 = _mm_cvtepi8_epi16(mCoefy3);
        mCoefy4 = _mm_cvtepi8_epi16(mCoefy4);

        for (row = 0; row < height; row++) {
            p = tmp;
            for (col = 0; col < width - 7; col += 8) {
                __m128i T00 = _mm_loadu_si128((__m128i*)(p));
                __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_tmp));
                __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_tmp));
                __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_tmp));
                __m128i T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_tmp));
                __m128i T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_tmp));
                __m128i T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_tmp));
                __m128i T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_tmp));

                __m128i T0 = _mm_unpacklo_epi16(T00, T10);
                __m128i T1 = _mm_unpacklo_epi16(T20, T30);
                __m128i T2 = _mm_unpacklo_epi16(T40, T50);
                __m128i T3 = _mm_unpacklo_epi16(T60, T70);
                __m128i T4 = _mm_unpackhi_epi16(T00, T10);
                __m128i T5 = _mm_unpackhi_epi16(T20, T30);
                __m128i T6 = _mm_unpackhi_epi16(T40, T50);
                __m128i T7 = _mm_unpackhi_epi16(T60, T70);

                T0 = _mm_madd_epi16(T0, mCoefy1);
                T1 = _mm_madd_epi16(T1, mCoefy2);
                T2 = _mm_madd_epi16(T2, mCoefy3);
                T3 = _mm_madd_epi16(T3, mCoefy4);
                T4 = _mm_madd_epi16(T4, mCoefy1);
                T5 = _mm_madd_epi16(T5, mCoefy2);
                T6 = _mm_madd_epi16(T6, mCoefy3);
                T7 = _mm_madd_epi16(T7, mCoefy4);

                mVal1 = _mm_add_epi32(T0, T1);
                mVal1 = _mm_add_epi32(mVal1, T2);
                mVal1 = _mm_add_epi32(mVal1, T3);

                mVal2 = _mm_add_epi32(T4, T5);
                mVal2 = _mm_add_epi32(mVal2, T6);
                mVal2 = _mm_add_epi32(mVal2, T7);

                mVal1 = _mm_add_epi32(mVal1, mAddOffset);
                mVal2 = _mm_add_epi32(mVal2, mAddOffset);
                mVal1 = _mm_srai_epi32(mVal1, shift);
                mVal2 = _mm_srai_epi32(mVal2, shift);
                mVal = _mm_packs_epi32(mVal1, mVal2);
                mVal = _mm_packus_epi16(mVal, mVal);

                _mm_storel_epi64((__m128i*)&dst[col], mVal);

                p += 8;
            }

            if (col < width) {
                __m128i T00 = _mm_loadu_si128((__m128i*)(p));
                __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_tmp));
                __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_tmp));
                __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_tmp));
                __m128i T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_tmp));
                __m128i T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_tmp));
                __m128i T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_tmp));
                __m128i T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_tmp));

                __m128i T0 = _mm_unpacklo_epi16(T00, T10);
                __m128i T1 = _mm_unpacklo_epi16(T20, T30);
                __m128i T2 = _mm_unpacklo_epi16(T40, T50);
                __m128i T3 = _mm_unpacklo_epi16(T60, T70);
                __m128i T4 = _mm_unpackhi_epi16(T00, T10);
                __m128i T5 = _mm_unpackhi_epi16(T20, T30);
                __m128i T6 = _mm_unpackhi_epi16(T40, T50);
                __m128i T7 = _mm_unpackhi_epi16(T60, T70);

                T0 = _mm_madd_epi16(T0, mCoefy1);
                T1 = _mm_madd_epi16(T1, mCoefy2);
                T2 = _mm_madd_epi16(T2, mCoefy3);
                T3 = _mm_madd_epi16(T3, mCoefy4);
                T4 = _mm_madd_epi16(T4, mCoefy1);
                T5 = _mm_madd_epi16(T5, mCoefy2);
                T6 = _mm_madd_epi16(T6, mCoefy3);
                T7 = _mm_madd_epi16(T7, mCoefy4);

                mVal1 = _mm_add_epi32(T0, T1);
                mVal1 = _mm_add_epi32(mVal1, T2);
                mVal1 = _mm_add_epi32(mVal1, T3);

                mVal2 = _mm_add_epi32(T4, T5);
                mVal2 = _mm_add_epi32(mVal2, T6);
                mVal2 = _mm_add_epi32(mVal2, T7);

                mVal1 = _mm_add_epi32(mVal1, mAddOffset);
                mVal2 = _mm_add_epi32(mVal2, mAddOffset);
                mVal1 = _mm_srai_epi32(mVal1, shift);
                mVal2 = _mm_srai_epi32(mVal2, shift);
                mVal = _mm_packs_epi32(mVal1, mVal2);
                mVal = _mm_packus_epi16(mVal, mVal);

                _mm_maskmoveu_si128(mVal, mask, (char *)&dst[col]);
            }

            tmp += i_tmp;
            dst += i_dst;
        }
    }
}

void intpl_luma_ext_x3_sse128(pel_t *const dst[3], int i_dst, mct_t *tmp, int i_tmp, int width, int height, const int8_t **coeff)
{
    /*
    intpl_luma_ext_sse128(dst0, i_dst, tmp, i_tmp, width, height, coeff[0]);
    intpl_luma_ext_sse128(dst1, i_dst, tmp, i_tmp, width, height, coeff[1]);
    intpl_luma_ext_sse128(dst2, i_dst, tmp, i_tmp, width, height, coeff[2]);
    */
    int row, col;
    int shift;
    int16_t const *p;
    int bsymyFirst = (coeff[0][1] == coeff[0][6]);
    int bsymySecond = (coeff[1][1] == coeff[1][6]);
    int bsymyThird = (coeff[2][1] == coeff[2][6]);

    __m128i mAddOffset;
    __m128i mask = _mm_loadu_si128((__m128i*)(intrinsic_mask[(width & 7) - 1]));

    // VER
    shift = 12;
    mAddOffset = _mm_set1_epi32(1 << (shift - 1));
    tmp = tmp - 3 * i_tmp;

    __m128i mCoefy1First,mCoefy2First,mCoefy3First,mCoefy4First;
    __m128i mCoefy1Second,mCoefy2Second,mCoefy3Second,mCoefy4Second;
    __m128i mCoefy1Third,mCoefy2Third,mCoefy3Third,mCoefy4Third;

    pel_t *dst0 = dst[0];
    pel_t *dst1 = dst[1];
    pel_t *dst2 = dst[2];

    if(bsymyFirst) {
        mCoefy1First = _mm_set1_epi16(coeff[0][0]);
        mCoefy2First = _mm_set1_epi16(coeff[0][1]);
        mCoefy3First = _mm_set1_epi16(coeff[0][2]);
        mCoefy4First = _mm_set1_epi16(coeff[0][3]);
    } else {
        mCoefy1First = _mm_set1_epi16(*(int16_t*)coeff[0]);
        mCoefy2First = _mm_set1_epi16(*(int16_t*)(coeff[0] + 2));
        mCoefy3First = _mm_set1_epi16(*(int16_t*)(coeff[0] + 4));
        mCoefy4First = _mm_set1_epi16(*(int16_t*)(coeff[0] + 6));
        mCoefy1First = _mm_cvtepi8_epi16(mCoefy1First);
        mCoefy2First = _mm_cvtepi8_epi16(mCoefy2First);
        mCoefy3First = _mm_cvtepi8_epi16(mCoefy3First);
        mCoefy4First = _mm_cvtepi8_epi16(mCoefy4First);
    }

    if(bsymySecond) {
        mCoefy1Second = _mm_set1_epi16(coeff[1][0]);
        mCoefy2Second = _mm_set1_epi16(coeff[1][1]);
        mCoefy3Second = _mm_set1_epi16(coeff[1][2]);
        mCoefy4Second = _mm_set1_epi16(coeff[1][3]);
    } else {
        mCoefy1Second = _mm_set1_epi16(*(int16_t*)coeff[1]);
        mCoefy2Second = _mm_set1_epi16(*(int16_t*)(coeff[1] + 2));
        mCoefy3Second = _mm_set1_epi16(*(int16_t*)(coeff[1] + 4));
        mCoefy4Second = _mm_set1_epi16(*(int16_t*)(coeff[1] + 6));
        mCoefy1Second = _mm_cvtepi8_epi16(mCoefy1Second);
        mCoefy2Second = _mm_cvtepi8_epi16(mCoefy2Second);
        mCoefy3Second = _mm_cvtepi8_epi16(mCoefy3Second);
        mCoefy4Second = _mm_cvtepi8_epi16(mCoefy4Second);
    }

    if(bsymyThird) {
        mCoefy1Third = _mm_set1_epi16(coeff[2][0]);
        mCoefy2Third = _mm_set1_epi16(coeff[2][1]);
        mCoefy3Third = _mm_set1_epi16(coeff[2][2]);
        mCoefy4Third = _mm_set1_epi16(coeff[2][3]);
    } else {
        mCoefy1Third = _mm_set1_epi16(*(int16_t*)coeff[2]);
        mCoefy2Third = _mm_set1_epi16(*(int16_t*)(coeff[2] + 2));
        mCoefy3Third = _mm_set1_epi16(*(int16_t*)(coeff[2] + 4));
        mCoefy4Third = _mm_set1_epi16(*(int16_t*)(coeff[2] + 6));
        mCoefy1Third = _mm_cvtepi8_epi16(mCoefy1Third);
        mCoefy2Third = _mm_cvtepi8_epi16(mCoefy2Third);
        mCoefy3Third = _mm_cvtepi8_epi16(mCoefy3Third);
        mCoefy4Third = _mm_cvtepi8_epi16(mCoefy4Third);
    }

    __m128i T00, T10, T20, T30, T40, T50, T60, T70;
    __m128i T0, T1, T2, T3, T4, T5, T6, T7;
    __m128i mVal1, mVal2, mVal;
    //
    for (row = 0; row < height; row++) {
        p = tmp;
        for (col = 0; col < width - 7; col += 8) {
            T00 = _mm_loadu_si128((__m128i*)(p));
            T10 = _mm_loadu_si128((__m128i*)(p + i_tmp));
            T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_tmp));
            T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_tmp));
            T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_tmp));
            T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_tmp));
            T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_tmp));
            T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_tmp));

            //First
            if (bsymyFirst) {
                T0 = _mm_unpacklo_epi16(T00, T70);
                T1 = _mm_unpacklo_epi16(T10, T60);
                T2 = _mm_unpacklo_epi16(T20, T50);
                T3 = _mm_unpacklo_epi16(T30, T40);
                T4 = _mm_unpackhi_epi16(T00, T70);
                T5 = _mm_unpackhi_epi16(T10, T60);
                T6 = _mm_unpackhi_epi16(T20, T50);
                T7 = _mm_unpackhi_epi16(T30, T40);
            } else {
                T0 = _mm_unpacklo_epi16(T00, T10);
                T1 = _mm_unpacklo_epi16(T20, T30);
                T2 = _mm_unpacklo_epi16(T40, T50);
                T3 = _mm_unpacklo_epi16(T60, T70);
                T4 = _mm_unpackhi_epi16(T00, T10);
                T5 = _mm_unpackhi_epi16(T20, T30);
                T6 = _mm_unpackhi_epi16(T40, T50);
                T7 = _mm_unpackhi_epi16(T60, T70);
            }
            T0 = _mm_madd_epi16(T0, mCoefy1First);
            T1 = _mm_madd_epi16(T1, mCoefy2First);
            T2 = _mm_madd_epi16(T2, mCoefy3First);
            T3 = _mm_madd_epi16(T3, mCoefy4First);
            T4 = _mm_madd_epi16(T4, mCoefy1First);
            T5 = _mm_madd_epi16(T5, mCoefy2First);
            T6 = _mm_madd_epi16(T6, mCoefy3First);
            T7 = _mm_madd_epi16(T7, mCoefy4First);

            mVal1 = _mm_add_epi32(T0, T1);
            mVal1 = _mm_add_epi32(mVal1, T2);
            mVal1 = _mm_add_epi32(mVal1, T3);

            mVal2 = _mm_add_epi32(T4, T5);
            mVal2 = _mm_add_epi32(mVal2, T6);
            mVal2 = _mm_add_epi32(mVal2, T7);

            mVal1 = _mm_add_epi32(mVal1, mAddOffset);
            mVal2 = _mm_add_epi32(mVal2, mAddOffset);
            mVal1 = _mm_srai_epi32(mVal1, shift);
            mVal2 = _mm_srai_epi32(mVal2, shift);
            mVal = _mm_packs_epi32(mVal1, mVal2);
            mVal = _mm_packus_epi16(mVal, mVal);

            _mm_storel_epi64((__m128i*)&dst0[col], mVal);

            //Second
            if (bsymySecond) {
                T0 = _mm_unpacklo_epi16(T00, T70);
                T1 = _mm_unpacklo_epi16(T10, T60);
                T2 = _mm_unpacklo_epi16(T20, T50);
                T3 = _mm_unpacklo_epi16(T30, T40);
                T4 = _mm_unpackhi_epi16(T00, T70);
                T5 = _mm_unpackhi_epi16(T10, T60);
                T6 = _mm_unpackhi_epi16(T20, T50);
                T7 = _mm_unpackhi_epi16(T30, T40);
            } else {
                T0 = _mm_unpacklo_epi16(T00, T10);
                T1 = _mm_unpacklo_epi16(T20, T30);
                T2 = _mm_unpacklo_epi16(T40, T50);
                T3 = _mm_unpacklo_epi16(T60, T70);
                T4 = _mm_unpackhi_epi16(T00, T10);
                T5 = _mm_unpackhi_epi16(T20, T30);
                T6 = _mm_unpackhi_epi16(T40, T50);
                T7 = _mm_unpackhi_epi16(T60, T70);
            }
            T0 = _mm_madd_epi16(T0, mCoefy1Second);
            T1 = _mm_madd_epi16(T1, mCoefy2Second);
            T2 = _mm_madd_epi16(T2, mCoefy3Second);
            T3 = _mm_madd_epi16(T3, mCoefy4Second);
            T4 = _mm_madd_epi16(T4, mCoefy1Second);
            T5 = _mm_madd_epi16(T5, mCoefy2Second);
            T6 = _mm_madd_epi16(T6, mCoefy3Second);
            T7 = _mm_madd_epi16(T7, mCoefy4Second);

            mVal1 = _mm_add_epi32(T0, T1);
            mVal1 = _mm_add_epi32(mVal1, T2);
            mVal1 = _mm_add_epi32(mVal1, T3);

            mVal2 = _mm_add_epi32(T4, T5);
            mVal2 = _mm_add_epi32(mVal2, T6);
            mVal2 = _mm_add_epi32(mVal2, T7);

            mVal1 = _mm_add_epi32(mVal1, mAddOffset);
            mVal2 = _mm_add_epi32(mVal2, mAddOffset);
            mVal1 = _mm_srai_epi32(mVal1, shift);
            mVal2 = _mm_srai_epi32(mVal2, shift);
            mVal = _mm_packs_epi32(mVal1, mVal2);
            mVal = _mm_packus_epi16(mVal, mVal);

            _mm_storel_epi64((__m128i*)&dst1[col], mVal);

            //Third
            if (bsymyThird) {
                T0 = _mm_unpacklo_epi16(T00, T70);
                T1 = _mm_unpacklo_epi16(T10, T60);
                T2 = _mm_unpacklo_epi16(T20, T50);
                T3 = _mm_unpacklo_epi16(T30, T40);
                T4 = _mm_unpackhi_epi16(T00, T70);
                T5 = _mm_unpackhi_epi16(T10, T60);
                T6 = _mm_unpackhi_epi16(T20, T50);
                T7 = _mm_unpackhi_epi16(T30, T40);
            } else {
                T0 = _mm_unpacklo_epi16(T00, T10);
                T1 = _mm_unpacklo_epi16(T20, T30);
                T2 = _mm_unpacklo_epi16(T40, T50);
                T3 = _mm_unpacklo_epi16(T60, T70);
                T4 = _mm_unpackhi_epi16(T00, T10);
                T5 = _mm_unpackhi_epi16(T20, T30);
                T6 = _mm_unpackhi_epi16(T40, T50);
                T7 = _mm_unpackhi_epi16(T60, T70);
            }
            T0 = _mm_madd_epi16(T0, mCoefy1Third);
            T1 = _mm_madd_epi16(T1, mCoefy2Third);
            T2 = _mm_madd_epi16(T2, mCoefy3Third);
            T3 = _mm_madd_epi16(T3, mCoefy4Third);
            T4 = _mm_madd_epi16(T4, mCoefy1Third);
            T5 = _mm_madd_epi16(T5, mCoefy2Third);
            T6 = _mm_madd_epi16(T6, mCoefy3Third);
            T7 = _mm_madd_epi16(T7, mCoefy4Third);

            mVal1 = _mm_add_epi32(T0, T1);
            mVal1 = _mm_add_epi32(mVal1, T2);
            mVal1 = _mm_add_epi32(mVal1, T3);

            mVal2 = _mm_add_epi32(T4, T5);
            mVal2 = _mm_add_epi32(mVal2, T6);
            mVal2 = _mm_add_epi32(mVal2, T7);

            mVal1 = _mm_add_epi32(mVal1, mAddOffset);
            mVal2 = _mm_add_epi32(mVal2, mAddOffset);
            mVal1 = _mm_srai_epi32(mVal1, shift);
            mVal2 = _mm_srai_epi32(mVal2, shift);
            mVal = _mm_packs_epi32(mVal1, mVal2);
            mVal = _mm_packus_epi16(mVal, mVal);

            _mm_storel_epi64((__m128i*)&dst2[col], mVal);

            p += 8;
        }

        if (col < width) {
            T00 = _mm_loadu_si128((__m128i*)(p));
            T10 = _mm_loadu_si128((__m128i*)(p + i_tmp));
            T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_tmp));
            T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_tmp));
            T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_tmp));
            T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_tmp));
            T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_tmp));
            T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_tmp));

            //First
            if (bsymyFirst) {
                T0 = _mm_unpacklo_epi16(T00, T70);
                T1 = _mm_unpacklo_epi16(T10, T60);
                T2 = _mm_unpacklo_epi16(T20, T50);
                T3 = _mm_unpacklo_epi16(T30, T40);
                T4 = _mm_unpackhi_epi16(T00, T70);
                T5 = _mm_unpackhi_epi16(T10, T60);
                T6 = _mm_unpackhi_epi16(T20, T50);
                T7 = _mm_unpackhi_epi16(T30, T40);
            } else {
                T0 = _mm_unpacklo_epi16(T00, T10);
                T1 = _mm_unpacklo_epi16(T20, T30);
                T2 = _mm_unpacklo_epi16(T40, T50);
                T3 = _mm_unpacklo_epi16(T60, T70);
                T4 = _mm_unpackhi_epi16(T00, T10);
                T5 = _mm_unpackhi_epi16(T20, T30);
                T6 = _mm_unpackhi_epi16(T40, T50);
                T7 = _mm_unpackhi_epi16(T60, T70);
            }
            T0 = _mm_madd_epi16(T0, mCoefy1First);
            T1 = _mm_madd_epi16(T1, mCoefy2First);
            T2 = _mm_madd_epi16(T2, mCoefy3First);
            T3 = _mm_madd_epi16(T3, mCoefy4First);
            T4 = _mm_madd_epi16(T4, mCoefy1First);
            T5 = _mm_madd_epi16(T5, mCoefy2First);
            T6 = _mm_madd_epi16(T6, mCoefy3First);
            T7 = _mm_madd_epi16(T7, mCoefy4First);

            mVal1 = _mm_add_epi32(T0, T1);
            mVal1 = _mm_add_epi32(mVal1, T2);
            mVal1 = _mm_add_epi32(mVal1, T3);

            mVal2 = _mm_add_epi32(T4, T5);
            mVal2 = _mm_add_epi32(mVal2, T6);
            mVal2 = _mm_add_epi32(mVal2, T7);

            mVal1 = _mm_add_epi32(mVal1, mAddOffset);
            mVal2 = _mm_add_epi32(mVal2, mAddOffset);
            mVal1 = _mm_srai_epi32(mVal1, shift);
            mVal2 = _mm_srai_epi32(mVal2, shift);
            mVal = _mm_packs_epi32(mVal1, mVal2);
            mVal = _mm_packus_epi16(mVal, mVal);

            _mm_maskmoveu_si128(mVal, mask, (char *)&dst0[col]);

            //Second
            if (bsymySecond) {
                T0 = _mm_unpacklo_epi16(T00, T70);
                T1 = _mm_unpacklo_epi16(T10, T60);
                T2 = _mm_unpacklo_epi16(T20, T50);
                T3 = _mm_unpacklo_epi16(T30, T40);
                T4 = _mm_unpackhi_epi16(T00, T70);
                T5 = _mm_unpackhi_epi16(T10, T60);
                T6 = _mm_unpackhi_epi16(T20, T50);
                T7 = _mm_unpackhi_epi16(T30, T40);
            } else {
                T0 = _mm_unpacklo_epi16(T00, T10);
                T1 = _mm_unpacklo_epi16(T20, T30);
                T2 = _mm_unpacklo_epi16(T40, T50);
                T3 = _mm_unpacklo_epi16(T60, T70);
                T4 = _mm_unpackhi_epi16(T00, T10);
                T5 = _mm_unpackhi_epi16(T20, T30);
                T6 = _mm_unpackhi_epi16(T40, T50);
                T7 = _mm_unpackhi_epi16(T60, T70);
            }
            T0 = _mm_madd_epi16(T0, mCoefy1Second);
            T1 = _mm_madd_epi16(T1, mCoefy2Second);
            T2 = _mm_madd_epi16(T2, mCoefy3Second);
            T3 = _mm_madd_epi16(T3, mCoefy4Second);
            T4 = _mm_madd_epi16(T4, mCoefy1Second);
            T5 = _mm_madd_epi16(T5, mCoefy2Second);
            T6 = _mm_madd_epi16(T6, mCoefy3Second);
            T7 = _mm_madd_epi16(T7, mCoefy4Second);

            mVal1 = _mm_add_epi32(T0, T1);
            mVal1 = _mm_add_epi32(mVal1, T2);
            mVal1 = _mm_add_epi32(mVal1, T3);

            mVal2 = _mm_add_epi32(T4, T5);
            mVal2 = _mm_add_epi32(mVal2, T6);
            mVal2 = _mm_add_epi32(mVal2, T7);

            mVal1 = _mm_add_epi32(mVal1, mAddOffset);
            mVal2 = _mm_add_epi32(mVal2, mAddOffset);
            mVal1 = _mm_srai_epi32(mVal1, shift);
            mVal2 = _mm_srai_epi32(mVal2, shift);
            mVal = _mm_packs_epi32(mVal1, mVal2);
            mVal = _mm_packus_epi16(mVal, mVal);

            _mm_maskmoveu_si128(mVal, mask, (char *)&dst1[col]);

            //Third
            if (bsymyThird) {
                T0 = _mm_unpacklo_epi16(T00, T70);
                T1 = _mm_unpacklo_epi16(T10, T60);
                T2 = _mm_unpacklo_epi16(T20, T50);
                T3 = _mm_unpacklo_epi16(T30, T40);
                T4 = _mm_unpackhi_epi16(T00, T70);
                T5 = _mm_unpackhi_epi16(T10, T60);
                T6 = _mm_unpackhi_epi16(T20, T50);
                T7 = _mm_unpackhi_epi16(T30, T40);
            } else {
                T0 = _mm_unpacklo_epi16(T00, T10);
                T1 = _mm_unpacklo_epi16(T20, T30);
                T2 = _mm_unpacklo_epi16(T40, T50);
                T3 = _mm_unpacklo_epi16(T60, T70);
                T4 = _mm_unpackhi_epi16(T00, T10);
                T5 = _mm_unpackhi_epi16(T20, T30);
                T6 = _mm_unpackhi_epi16(T40, T50);
                T7 = _mm_unpackhi_epi16(T60, T70);
            }
            T0 = _mm_madd_epi16(T0, mCoefy1Third);
            T1 = _mm_madd_epi16(T1, mCoefy2Third);
            T2 = _mm_madd_epi16(T2, mCoefy3Third);
            T3 = _mm_madd_epi16(T3, mCoefy4Third);
            T4 = _mm_madd_epi16(T4, mCoefy1Third);
            T5 = _mm_madd_epi16(T5, mCoefy2Third);
            T6 = _mm_madd_epi16(T6, mCoefy3Third);
            T7 = _mm_madd_epi16(T7, mCoefy4Third);

            mVal1 = _mm_add_epi32(T0, T1);
            mVal1 = _mm_add_epi32(mVal1, T2);
            mVal1 = _mm_add_epi32(mVal1, T3);

            mVal2 = _mm_add_epi32(T4, T5);
            mVal2 = _mm_add_epi32(mVal2, T6);
            mVal2 = _mm_add_epi32(mVal2, T7);

            mVal1 = _mm_add_epi32(mVal1, mAddOffset);
            mVal2 = _mm_add_epi32(mVal2, mAddOffset);
            mVal1 = _mm_srai_epi32(mVal1, shift);
            mVal2 = _mm_srai_epi32(mVal2, shift);
            mVal = _mm_packs_epi32(mVal1, mVal2);
            mVal = _mm_packus_epi16(mVal, mVal);

            _mm_maskmoveu_si128(mVal, mask, (char *)&dst2[col]);
        }

        tmp += i_tmp;
        dst0 += i_dst;
        dst1 += i_dst;
        dst2 += i_dst;
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_chroma_block_ver_sse128(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff)
{
    int row, col;
    const short offset = 32;
    const int shift = 6;
    int bsym = (coeff[1] == coeff[2]);
    __m128i mAddOffset = _mm_set1_epi16(offset);
    pel_t const *p;
    __m128i mask = _mm_loadu_si128((__m128i*)(intrinsic_mask[(width & 7) - 1]));

    src -= i_src;
    if (bsym) {
        __m128i coeff0 = _mm_set1_epi8(coeff[0]);
        __m128i coeff1 = _mm_set1_epi8(coeff[1]);
        __m128i mVal;

        for (row = 0; row < height; row++) {
            p = src;
            for (col = 0; col < width - 7; col += 8) {
                __m128i T00 = _mm_loadu_si128((__m128i*)(p));
                __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_src));
                __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_src));
                __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_src));

                T00 = _mm_unpacklo_epi8(T00, T30);
                T10 = _mm_unpacklo_epi8(T10, T20);

                T00 = _mm_maddubs_epi16(T00, coeff0);
                T10 = _mm_maddubs_epi16(T10, coeff1);

                mVal = _mm_add_epi16(T00, T10);

                mVal = _mm_add_epi16(mVal, mAddOffset);
                mVal = _mm_srai_epi16(mVal, shift);
                mVal = _mm_packus_epi16(mVal, mVal);

                _mm_storel_epi64((__m128i*)&dst[col], mVal);

                p += 8;
            }

            if (col < width) { // store either 1, 2, 3, 4, 5, 6, or 7 8-bit results in dst
                __m128i T00 = _mm_loadu_si128((__m128i*)(p));
                __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_src));
                __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_src));
                __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_src));

                T00 = _mm_unpacklo_epi8(T00, T30);
                T10 = _mm_unpacklo_epi8(T10, T20);

                T00 = _mm_maddubs_epi16(T00, coeff0);
                T10 = _mm_maddubs_epi16(T10, coeff1);

                mVal = _mm_add_epi16(T00, T10);

                mVal = _mm_add_epi16(mVal, mAddOffset);
                mVal = _mm_srai_epi16(mVal, shift);
                mVal = _mm_packus_epi16(mVal, mVal);

                _mm_maskmoveu_si128(mVal, mask, (char *)&dst[col]);
            }

            src += i_src;
            dst += i_dst;
        }
    } else {
        __m128i coeff0 = _mm_set1_epi16(*(short*)coeff);
        __m128i coeff1 = _mm_set1_epi16(*(short*)(coeff + 2));
        __m128i mVal;
        for (row = 0; row < height; row++) {
            p = src;
            for (col = 0; col < width - 7; col += 8) {
                __m128i T00 = _mm_loadu_si128((__m128i*)(p));
                __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_src));
                __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_src));
                __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_src));

                T00 = _mm_unpacklo_epi8(T00, T10);
                T10 = _mm_unpacklo_epi8(T20, T30);

                T00 = _mm_maddubs_epi16(T00, coeff0);
                T10 = _mm_maddubs_epi16(T10, coeff1);

                mVal = _mm_add_epi16(T00, T10);

                mVal = _mm_add_epi16(mVal, mAddOffset);
                mVal = _mm_srai_epi16(mVal, shift);
                mVal = _mm_packus_epi16(mVal, mVal);

                _mm_storel_epi64((__m128i*)&dst[col], mVal);

                p += 8;
            }

            if (col < width) { // store either 1, 2, 3, 4, 5, 6, or 7 8-bit results in dst
                __m128i T00 = _mm_loadu_si128((__m128i*)(p));
                __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_src));
                __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_src));
                __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_src));

                T00 = _mm_unpacklo_epi8(T00, T10);
                T10 = _mm_unpacklo_epi8(T20, T30);

                T00 = _mm_maddubs_epi16(T00, coeff0);
                T10 = _mm_maddubs_epi16(T10, coeff1);

                mVal = _mm_add_epi16(T00, T10);

                mVal = _mm_add_epi16(mVal, mAddOffset);
                mVal = _mm_srai_epi16(mVal, shift);
                mVal = _mm_packus_epi16(mVal, mVal);

                _mm_maskmoveu_si128(mVal, mask, (char *)&dst[col]);
            }

            src += i_src;
            dst += i_dst;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_luma_block_ver_sse128(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff)
{
    const short offset = 32;
    const int shift = 6;
    int row, col;
    int bsym = (coeff[1] == coeff[6]);

    __m128i mAddOffset = _mm_set1_epi16(offset);

    pel_t const *p;

    __m128i mask = _mm_loadu_si128((__m128i*)(intrinsic_mask[(width & 7) - 1]));

    src -= 3 * i_src;

    if (bsym) {
        __m128i coeff0 = _mm_set1_epi8(coeff[0]);
        __m128i coeff1 = _mm_set1_epi8(coeff[1]);
        __m128i coeff2 = _mm_set1_epi8(coeff[2]);
        __m128i coeff3 = _mm_set1_epi8(coeff[3]);

        for (row = 0; row < height; row++) {
            __m128i mVal;
            p = src;
            for (col = 0; col < width - 7; col += 8) {
                __m128i T00 = _mm_loadu_si128((__m128i*)(p));
                __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_src));
                __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_src));
                __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_src));
                __m128i T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_src));
                __m128i T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_src));
                __m128i T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_src));
                __m128i T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_src));

                T00 = _mm_maddubs_epi16(_mm_unpacklo_epi8(T00, T70), coeff0);
                T10 = _mm_maddubs_epi16(_mm_unpacklo_epi8(T10, T60), coeff1);
                T20 = _mm_maddubs_epi16(_mm_unpacklo_epi8(T20, T50), coeff2);
                T30 = _mm_maddubs_epi16(_mm_unpacklo_epi8(T30, T40), coeff3);

                mVal = _mm_add_epi16(_mm_add_epi16(T00, T10), _mm_add_epi16(T20, T30));
                mVal = _mm_srai_epi16(_mm_add_epi16(mVal, mAddOffset), shift);
                mVal = _mm_packus_epi16(mVal, mVal);

                _mm_storel_epi64((__m128i*)&dst[col], mVal);

                p += 8;
            }

            if (col < width) { // store either 1, 2, 3, 4, 5, 6, or 7 8-bit results in dst
                __m128i T00 = _mm_loadu_si128((__m128i*)(p));
                __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_src));
                __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_src));
                __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_src));
                __m128i T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_src));
                __m128i T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_src));
                __m128i T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_src));
                __m128i T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_src));

                T00 = _mm_maddubs_epi16(_mm_unpacklo_epi8(T00, T70), coeff0);
                T10 = _mm_maddubs_epi16(_mm_unpacklo_epi8(T10, T60), coeff1);
                T20 = _mm_maddubs_epi16(_mm_unpacklo_epi8(T20, T50), coeff2);
                T30 = _mm_maddubs_epi16(_mm_unpacklo_epi8(T30, T40), coeff3);

                mVal = _mm_add_epi16(_mm_add_epi16(T00, T10), _mm_add_epi16(T20, T30));
                mVal = _mm_srai_epi16(_mm_add_epi16(mVal, mAddOffset), shift);
                mVal = _mm_packus_epi16(mVal, mVal);

                _mm_maskmoveu_si128(mVal, mask, (char *)&dst[col]);
            }

            src += i_src;
            dst += i_dst;
        }
    } else {
        __m128i coeff0 = _mm_set1_epi16(*(short*)coeff);
        __m128i coeff1 = _mm_set1_epi16(*(short*)(coeff + 2));
        __m128i coeff2 = _mm_set1_epi16(*(short*)(coeff + 4));
        __m128i coeff3 = _mm_set1_epi16(*(short*)(coeff + 6));
        for (row = 0; row < height; row++) {
            __m128i mVal;
            p = src;
            for (col = 0; col < width - 7; col += 8) {
                __m128i T00 = _mm_loadu_si128((__m128i*)(p));
                __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_src));
                __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_src));
                __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_src));
                __m128i T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_src));
                __m128i T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_src));
                __m128i T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_src));
                __m128i T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_src));

                T00 = _mm_maddubs_epi16(_mm_unpacklo_epi8(T00, T10), coeff0);
                T10 = _mm_maddubs_epi16(_mm_unpacklo_epi8(T20, T30), coeff1);
                T20 = _mm_maddubs_epi16(_mm_unpacklo_epi8(T40, T50), coeff2);
                T30 = _mm_maddubs_epi16(_mm_unpacklo_epi8(T60, T70), coeff3);

                mVal = _mm_add_epi16(_mm_add_epi16(T00, T10), _mm_add_epi16(T20, T30));
                mVal = _mm_srai_epi16(_mm_add_epi16(mVal, mAddOffset), shift);
                mVal = _mm_packus_epi16(mVal, mVal);

                _mm_storel_epi64((__m128i*)&dst[col], mVal);

                p += 8;
            }

            if (col < width) { // store either 1, 2, 3, 4, 5, 6, or 7 8-bit results in dst
                __m128i T00 = _mm_loadu_si128((__m128i*)(p));
                __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_src));
                __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_src));
                __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_src));
                __m128i T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_src));
                __m128i T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_src));
                __m128i T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_src));
                __m128i T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_src));

                T00 = _mm_maddubs_epi16(_mm_unpacklo_epi8(T00, T10), coeff0);
                T10 = _mm_maddubs_epi16(_mm_unpacklo_epi8(T20, T30), coeff1);
                T20 = _mm_maddubs_epi16(_mm_unpacklo_epi8(T40, T50), coeff2);
                T30 = _mm_maddubs_epi16(_mm_unpacklo_epi8(T60, T70), coeff3);

                mVal = _mm_add_epi16(_mm_add_epi16(T00, T10), _mm_add_epi16(T20, T30));
                mVal = _mm_srai_epi16(_mm_add_epi16(mVal, mAddOffset), shift);
                mVal = _mm_packus_epi16(mVal, mVal);

                _mm_maskmoveu_si128(mVal, mask, (char *)&dst[col]);
            }

            src += i_src;
            dst += i_dst;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_chroma_block_ext_sse128(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coef_x, const int8_t *coef_y)
{
    ALIGN16(int16_t tmp_res[(32 + 3) * 32]);
    int16_t *tmp = tmp_res;
    const int i_tmp = 32;
    int row, col;
    int shift;
    int16_t const *p;

    int bsymy = (coef_y[1] == coef_y[6]);

    __m128i mAddOffset;

    __m128i mSwitch1 = _mm_setr_epi8(0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6);
    __m128i mSwitch2 = _mm_setr_epi8(4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10);

    __m128i mCoefx = _mm_set1_epi32(*(int*)coef_x);

    __m128i mask = _mm_loadu_si128((__m128i*)(intrinsic_mask[(width & 7) - 1]));

    // HOR
    src = src - 1 * i_src - 1;

    if (width > 4) {
        for (row = -1; row < height + 2; row++) {
            __m128i mT0, mT1, mV01;
            for (col = 0; col < width; col += 8) {
                __m128i mSrc = _mm_loadu_si128((__m128i*)(src + col));
                mT0 = _mm_maddubs_epi16(_mm_shuffle_epi8(mSrc, mSwitch1), mCoefx);
                mT1 = _mm_maddubs_epi16(_mm_shuffle_epi8(mSrc, mSwitch2), mCoefx);

                mV01 = _mm_hadd_epi16(mT0, mT1);
                _mm_store_si128((__m128i*)&tmp[col], mV01);
            }
            src += i_src;
            tmp += i_tmp;
        }
    } else {
        for (row = -1; row < height + 2; row++) {
            __m128i mSrc = _mm_loadu_si128((__m128i*)src);
            __m128i mT0 = _mm_maddubs_epi16(_mm_shuffle_epi8(mSrc, mSwitch1), mCoefx);
            __m128i mV01 = _mm_hadd_epi16(mT0, mT0);
            _mm_storel_epi64((__m128i*)tmp, mV01);
            src += i_src;
            tmp += i_tmp;
        }
    }


    // VER
    shift = 12;
    mAddOffset = _mm_set1_epi32(1 << 11);

    tmp = tmp_res;
    if (bsymy) {
        __m128i mCoefy1 = _mm_set1_epi16(coef_y[0]);
        __m128i mCoefy2 = _mm_set1_epi16(coef_y[1]);

        for (row = 0; row < height; row += 2) {
            p = tmp;
            for (col = 0; col < width - 7; col += 8) {
                __m128i mV01, mV02;
                __m128i mV11, mV12;
                __m128i T0 = _mm_loadu_si128((__m128i*)(p));
                __m128i T1 = _mm_loadu_si128((__m128i*)(p + i_tmp));
                __m128i T2 = _mm_loadu_si128((__m128i*)(p + 2 * i_tmp));
                __m128i T3 = _mm_loadu_si128((__m128i*)(p + 3 * i_tmp));
                __m128i T4 = _mm_loadu_si128((__m128i*)(p + 4 * i_tmp));

                __m128i M00 = _mm_unpacklo_epi16(T0, T3);
                __m128i M01 = _mm_unpacklo_epi16(T1, T2);
                __m128i M02 = _mm_unpackhi_epi16(T0, T3);
                __m128i M03 = _mm_unpackhi_epi16(T1, T2);

                __m128i M10 = _mm_unpacklo_epi16(T1, T4);
                __m128i M11 = _mm_unpacklo_epi16(T2, T3);
                __m128i M12 = _mm_unpackhi_epi16(T1, T4);
                __m128i M13 = _mm_unpackhi_epi16(T2, T3);

                mV01 = _mm_add_epi32(_mm_madd_epi16(M00, mCoefy1), _mm_madd_epi16(M01, mCoefy2));
                mV02 = _mm_add_epi32(_mm_madd_epi16(M02, mCoefy1), _mm_madd_epi16(M03, mCoefy2));
                mV11 = _mm_add_epi32(_mm_madd_epi16(M10, mCoefy1), _mm_madd_epi16(M11, mCoefy2));
                mV12 = _mm_add_epi32(_mm_madd_epi16(M12, mCoefy1), _mm_madd_epi16(M13, mCoefy2));

                mV01 = _mm_srai_epi32(_mm_add_epi32(mV01, mAddOffset), shift);
                mV02 = _mm_srai_epi32(_mm_add_epi32(mV02, mAddOffset), shift);
                mV11 = _mm_srai_epi32(_mm_add_epi32(mV11, mAddOffset), shift);
                mV12 = _mm_srai_epi32(_mm_add_epi32(mV12, mAddOffset), shift);

                mV01 = _mm_packs_epi32 (mV01, mV02);
                mV01 = _mm_packus_epi16(mV01, mV01);
                mV11 = _mm_packs_epi32 (mV11, mV12);
                mV11 = _mm_packus_epi16(mV11, mV11);

                _mm_storel_epi64((__m128i*)&dst[col],         mV01);
                _mm_storel_epi64((__m128i*)&dst[col + i_dst], mV11);

                p += 8;
            }

            if (col < width) { // store either 1, 2, 3, 4, 5, 6, or 7 8-bit results in dst
                __m128i mV01, mV02;
                __m128i mV11, mV12;
                __m128i T0 = _mm_loadu_si128((__m128i*)(p));
                __m128i T1 = _mm_loadu_si128((__m128i*)(p + i_tmp));
                __m128i T2 = _mm_loadu_si128((__m128i*)(p + 2 * i_tmp));
                __m128i T3 = _mm_loadu_si128((__m128i*)(p + 3 * i_tmp));
                __m128i T4 = _mm_loadu_si128((__m128i*)(p + 4 * i_tmp));

                __m128i M00 = _mm_unpacklo_epi16(T0, T3);
                __m128i M01 = _mm_unpacklo_epi16(T1, T2);
                __m128i M02 = _mm_unpackhi_epi16(T0, T3);
                __m128i M03 = _mm_unpackhi_epi16(T1, T2);

                __m128i M10 = _mm_unpacklo_epi16(T1, T4);
                __m128i M11 = _mm_unpacklo_epi16(T2, T3);
                __m128i M12 = _mm_unpackhi_epi16(T1, T4);
                __m128i M13 = _mm_unpackhi_epi16(T2, T3);

                mV01 = _mm_add_epi32(_mm_madd_epi16(M00, mCoefy1), _mm_madd_epi16(M01, mCoefy2));
                mV02 = _mm_add_epi32(_mm_madd_epi16(M02, mCoefy1), _mm_madd_epi16(M03, mCoefy2));
                mV11 = _mm_add_epi32(_mm_madd_epi16(M10, mCoefy1), _mm_madd_epi16(M11, mCoefy2));
                mV12 = _mm_add_epi32(_mm_madd_epi16(M12, mCoefy1), _mm_madd_epi16(M13, mCoefy2));

                mV01 = _mm_srai_epi32(_mm_add_epi32(mV01, mAddOffset), shift);
                mV02 = _mm_srai_epi32(_mm_add_epi32(mV02, mAddOffset), shift);
                mV11 = _mm_srai_epi32(_mm_add_epi32(mV11, mAddOffset), shift);
                mV12 = _mm_srai_epi32(_mm_add_epi32(mV12, mAddOffset), shift);

                mV01 = _mm_packs_epi32 (mV01, mV02);
                mV01 = _mm_packus_epi16(mV01, mV01);
                mV11 = _mm_packs_epi32 (mV11, mV12);
                mV11 = _mm_packus_epi16(mV11, mV11);

                _mm_maskmoveu_si128(mV01, mask, (char *)&dst[col]);
                _mm_maskmoveu_si128(mV01, mask, (char *)&dst[col + i_dst]);
            }

            tmp += i_tmp * 2;
            dst += i_dst * 2;
        }
    } else {
        __m128i coeff0 = _mm_set1_epi16(*(short*)coef_y);
        __m128i coeff1 = _mm_set1_epi16(*(short*)(coef_y + 2));
        coeff0 = _mm_cvtepi8_epi16(coeff0);
        coeff1 = _mm_cvtepi8_epi16(coeff1);

        for (row = 0; row < height; row += 2) {
            p = tmp;
            for (col = 0; col < width - 7; col += 8) {
                __m128i mV01, mV02;
                __m128i mV11, mV12;
                __m128i T0 = _mm_loadu_si128((__m128i*)(p));
                __m128i T1 = _mm_loadu_si128((__m128i*)(p + i_tmp));
                __m128i T2 = _mm_loadu_si128((__m128i*)(p + 2 * i_tmp));
                __m128i T3 = _mm_loadu_si128((__m128i*)(p + 3 * i_tmp));
                __m128i T4 = _mm_loadu_si128((__m128i*)(p + 4 * i_tmp));

                __m128i M00 = _mm_unpacklo_epi16(T0, T1);
                __m128i M01 = _mm_unpacklo_epi16(T2, T3);
                __m128i M02 = _mm_unpackhi_epi16(T0, T1);
                __m128i M03 = _mm_unpackhi_epi16(T2, T3);

                __m128i M10 = _mm_unpacklo_epi16(T1, T2);
                __m128i M11 = _mm_unpacklo_epi16(T3, T4);
                __m128i M12 = _mm_unpackhi_epi16(T1, T2);
                __m128i M13 = _mm_unpackhi_epi16(T3, T4);

                mV01 = _mm_add_epi32(_mm_madd_epi16(M00, coeff0), _mm_madd_epi16(M01, coeff1));
                mV02 = _mm_add_epi32(_mm_madd_epi16(M02, coeff0), _mm_madd_epi16(M03, coeff1));
                mV11 = _mm_add_epi32(_mm_madd_epi16(M10, coeff0), _mm_madd_epi16(M11, coeff1));
                mV12 = _mm_add_epi32(_mm_madd_epi16(M12, coeff0), _mm_madd_epi16(M13, coeff1));

                mV01 = _mm_srai_epi32(_mm_add_epi32(mV01, mAddOffset), shift);
                mV02 = _mm_srai_epi32(_mm_add_epi32(mV02, mAddOffset), shift);
                mV11 = _mm_srai_epi32(_mm_add_epi32(mV11, mAddOffset), shift);
                mV12 = _mm_srai_epi32(_mm_add_epi32(mV12, mAddOffset), shift);

                mV01 = _mm_packs_epi32 (mV01, mV02);
                mV01 = _mm_packus_epi16(mV01, mV01);
                mV11 = _mm_packs_epi32 (mV11, mV12);
                mV11 = _mm_packus_epi16(mV11, mV11);

                _mm_storel_epi64((__m128i*)&dst[col],         mV01);
                _mm_storel_epi64((__m128i*)&dst[col + i_dst], mV11);

                p += 8;
            }

            if (col < width) { // store either 1, 2, 3, 4, 5, 6, or 7 8-bit results in dst
                __m128i mV01, mV02;
                __m128i mV11, mV12;
                __m128i T0 = _mm_loadu_si128((__m128i*)(p));
                __m128i T1 = _mm_loadu_si128((__m128i*)(p + i_tmp));
                __m128i T2 = _mm_loadu_si128((__m128i*)(p + 2 * i_tmp));
                __m128i T3 = _mm_loadu_si128((__m128i*)(p + 3 * i_tmp));
                __m128i T4 = _mm_loadu_si128((__m128i*)(p + 4 * i_tmp));

                __m128i M00 = _mm_unpacklo_epi16(T0, T1);
                __m128i M01 = _mm_unpacklo_epi16(T2, T3);
                __m128i M02 = _mm_unpackhi_epi16(T0, T1);
                __m128i M03 = _mm_unpackhi_epi16(T2, T3);

                __m128i M10 = _mm_unpacklo_epi16(T1, T2);
                __m128i M11 = _mm_unpacklo_epi16(T3, T4);
                __m128i M12 = _mm_unpackhi_epi16(T1, T2);
                __m128i M13 = _mm_unpackhi_epi16(T3, T4);

                mV01 = _mm_add_epi32(_mm_madd_epi16(M00, coeff0), _mm_madd_epi16(M01, coeff1));
                mV02 = _mm_add_epi32(_mm_madd_epi16(M02, coeff0), _mm_madd_epi16(M03, coeff1));
                mV11 = _mm_add_epi32(_mm_madd_epi16(M10, coeff0), _mm_madd_epi16(M11, coeff1));
                mV12 = _mm_add_epi32(_mm_madd_epi16(M12, coeff0), _mm_madd_epi16(M13, coeff1));

                mV01 = _mm_srai_epi32(_mm_add_epi32(mV01, mAddOffset), shift);
                mV02 = _mm_srai_epi32(_mm_add_epi32(mV02, mAddOffset), shift);
                mV11 = _mm_srai_epi32(_mm_add_epi32(mV11, mAddOffset), shift);
                mV12 = _mm_srai_epi32(_mm_add_epi32(mV12, mAddOffset), shift);

                mV01 = _mm_packs_epi32 (mV01, mV02);
                mV01 = _mm_packus_epi16(mV01, mV01);
                mV11 = _mm_packs_epi32 (mV11, mV12);
                mV11 = _mm_packus_epi16(mV11, mV11);

                _mm_maskmoveu_si128(mV01, mask, (char *)&dst[col]);
                _mm_maskmoveu_si128(mV11, mask, (char *)&dst[col + i_dst]);
            }

            tmp += i_tmp * 2;
            dst += i_dst * 2;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void intpl_luma_block_ext_sse128(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coef_x, const int8_t *coef_y)
{
    ALIGN16(int16_t tmp_res[(64 + 7) * 64]);
    int16_t *tmp = tmp_res;
    const int i_tmp = 64;
    int row, col;
    int shift = 12;
    int16_t const *p;

    int bsymy = (coef_y[1] == coef_y[6]);

    __m128i mAddOffset = _mm_set1_epi32(1 << (shift - 1));

    __m128i mSwitch1 = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8);
    __m128i mSwitch2 = _mm_setr_epi8(2, 3, 4, 5, 6, 7, 8, 9, 3, 4, 5, 6, 7, 8, 9, 10);
    __m128i mSwitch3 = _mm_setr_epi8(4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9, 10, 11, 12);
    __m128i mSwitch4 = _mm_setr_epi8(6, 7, 8, 9, 10, 11, 12, 13, 7, 8, 9, 10, 11, 12, 13, 14);

    __m128i mCoefx = _mm_loadl_epi64((__m128i*)coef_x);
    __m128i mask = _mm_loadu_si128((__m128i*)(intrinsic_mask[(width & 7) - 1]));

    mCoefx = _mm_unpacklo_epi64(mCoefx, mCoefx);

    // HOR
    src -= (3 * i_src + 3);

    for (row = -3; row < height + 4; row++) {
        for (col = 0; col < width; col += 8) {
            __m128i mSrc = _mm_loadu_si128((__m128i*)(src + col));
            __m128i mT0  = _mm_maddubs_epi16(_mm_shuffle_epi8(mSrc, mSwitch1), mCoefx);
            __m128i mT1  = _mm_maddubs_epi16(_mm_shuffle_epi8(mSrc, mSwitch2), mCoefx);
            __m128i mT2  = _mm_maddubs_epi16(_mm_shuffle_epi8(mSrc, mSwitch3), mCoefx);
            __m128i mT3  = _mm_maddubs_epi16(_mm_shuffle_epi8(mSrc, mSwitch4), mCoefx);
            __m128i mVal = _mm_hadd_epi16(_mm_hadd_epi16(mT0, mT1), _mm_hadd_epi16(mT2, mT3));

            _mm_store_si128((__m128i*)&tmp[col], mVal);
        }

        src += i_src;
        tmp += i_tmp;
    }

    // VER
    tmp = tmp_res;

    if (bsymy) {
        __m128i mCoefy1 = _mm_set1_epi16(coef_y[0]);
        __m128i mCoefy2 = _mm_set1_epi16(coef_y[1]);
        __m128i mCoefy3 = _mm_set1_epi16(coef_y[2]);
        __m128i mCoefy4 = _mm_set1_epi16(coef_y[3]);

        for (row = 0; row < height; row++) {
            p = tmp;
            for (col = 0; col < width - 7; col += 8) {
                __m128i T00 = _mm_loadu_si128((__m128i*)(p));
                __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_tmp));
                __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_tmp));
                __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_tmp));
                __m128i T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_tmp));
                __m128i T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_tmp));
                __m128i T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_tmp));
                __m128i T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_tmp));

                __m128i T0 = _mm_unpacklo_epi16(T00, T70);
                __m128i T1 = _mm_unpacklo_epi16(T10, T60);
                __m128i T2 = _mm_unpacklo_epi16(T20, T50);
                __m128i T3 = _mm_unpacklo_epi16(T30, T40);
                __m128i T4 = _mm_unpackhi_epi16(T00, T70);
                __m128i T5 = _mm_unpackhi_epi16(T10, T60);
                __m128i T6 = _mm_unpackhi_epi16(T20, T50);
                __m128i T7 = _mm_unpackhi_epi16(T30, T40);
                __m128i mVal1, mVal2, mVal;

                T0 = _mm_madd_epi16(T0, mCoefy1);
                T1 = _mm_madd_epi16(T1, mCoefy2);
                T2 = _mm_madd_epi16(T2, mCoefy3);
                T3 = _mm_madd_epi16(T3, mCoefy4);
                T4 = _mm_madd_epi16(T4, mCoefy1);
                T5 = _mm_madd_epi16(T5, mCoefy2);
                T6 = _mm_madd_epi16(T6, mCoefy3);
                T7 = _mm_madd_epi16(T7, mCoefy4);

                mVal1 = _mm_add_epi32(_mm_add_epi32(T0, T1), _mm_add_epi32(T2, T3));
                mVal2 = _mm_add_epi32(_mm_add_epi32(T4, T5), _mm_add_epi32(T6, T7));

                mVal1 = _mm_srai_epi32(_mm_add_epi32(mVal1, mAddOffset), shift);
                mVal2 = _mm_srai_epi32(_mm_add_epi32(mVal2, mAddOffset), shift);

                mVal = _mm_packs_epi32(mVal1, mVal2);
                mVal = _mm_packus_epi16(mVal, mVal);

                _mm_storel_epi64((__m128i*)&dst[col], mVal);

                p += 8;
            }

            if (col < width) { // store either 1, 2, 3, 4, 5, 6, or 7 8-bit results in dst
                __m128i T00 = _mm_loadu_si128((__m128i*)(p));
                __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_tmp));
                __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_tmp));
                __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_tmp));
                __m128i T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_tmp));
                __m128i T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_tmp));
                __m128i T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_tmp));
                __m128i T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_tmp));

                __m128i T0 = _mm_unpacklo_epi16(T00, T70);
                __m128i T1 = _mm_unpacklo_epi16(T10, T60);
                __m128i T2 = _mm_unpacklo_epi16(T20, T50);
                __m128i T3 = _mm_unpacklo_epi16(T30, T40);
                __m128i T4 = _mm_unpackhi_epi16(T00, T70);
                __m128i T5 = _mm_unpackhi_epi16(T10, T60);
                __m128i T6 = _mm_unpackhi_epi16(T20, T50);
                __m128i T7 = _mm_unpackhi_epi16(T30, T40);
                __m128i mVal1, mVal2, mVal;

                T0 = _mm_madd_epi16(T0, mCoefy1);
                T1 = _mm_madd_epi16(T1, mCoefy2);
                T2 = _mm_madd_epi16(T2, mCoefy3);
                T3 = _mm_madd_epi16(T3, mCoefy4);
                T4 = _mm_madd_epi16(T4, mCoefy1);
                T5 = _mm_madd_epi16(T5, mCoefy2);
                T6 = _mm_madd_epi16(T6, mCoefy3);
                T7 = _mm_madd_epi16(T7, mCoefy4);

                mVal1 = _mm_add_epi32(_mm_add_epi32(T0, T1), _mm_add_epi32(T2, T3));
                mVal2 = _mm_add_epi32(_mm_add_epi32(T4, T5), _mm_add_epi32(T6, T7));

                mVal1 = _mm_srai_epi32(_mm_add_epi32(mVal1, mAddOffset), shift);
                mVal2 = _mm_srai_epi32(_mm_add_epi32(mVal2, mAddOffset), shift);

                mVal = _mm_packs_epi32(mVal1, mVal2);
                mVal = _mm_packus_epi16(mVal, mVal);

                _mm_maskmoveu_si128(mVal, mask, (char *)&dst[col]);
            }
            tmp += i_tmp;
            dst += i_dst;
        }
    } else {
        __m128i mCoefy1 = _mm_set1_epi16(*(int16_t*)coef_y);
        __m128i mCoefy2 = _mm_set1_epi16(*(int16_t*)(coef_y + 2));
        __m128i mCoefy3 = _mm_set1_epi16(*(int16_t*)(coef_y + 4));
        __m128i mCoefy4 = _mm_set1_epi16(*(int16_t*)(coef_y + 6));
        mCoefy1 = _mm_cvtepi8_epi16(mCoefy1);
        mCoefy2 = _mm_cvtepi8_epi16(mCoefy2);
        mCoefy3 = _mm_cvtepi8_epi16(mCoefy3);
        mCoefy4 = _mm_cvtepi8_epi16(mCoefy4);

        for (row = 0; row < height; row++) {
            p = tmp;
            for (col = 0; col < width - 7; col += 8) {
                __m128i T00 = _mm_loadu_si128((__m128i*)(p));
                __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_tmp));
                __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_tmp));
                __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_tmp));
                __m128i T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_tmp));
                __m128i T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_tmp));
                __m128i T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_tmp));
                __m128i T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_tmp));

                __m128i T0 = _mm_unpacklo_epi16(T00, T10);
                __m128i T1 = _mm_unpacklo_epi16(T20, T30);
                __m128i T2 = _mm_unpacklo_epi16(T40, T50);
                __m128i T3 = _mm_unpacklo_epi16(T60, T70);
                __m128i T4 = _mm_unpackhi_epi16(T00, T10);
                __m128i T5 = _mm_unpackhi_epi16(T20, T30);
                __m128i T6 = _mm_unpackhi_epi16(T40, T50);
                __m128i T7 = _mm_unpackhi_epi16(T60, T70);
                __m128i mVal1, mVal2, mVal;

                T0 = _mm_madd_epi16(T0, mCoefy1);
                T1 = _mm_madd_epi16(T1, mCoefy2);
                T2 = _mm_madd_epi16(T2, mCoefy3);
                T3 = _mm_madd_epi16(T3, mCoefy4);
                T4 = _mm_madd_epi16(T4, mCoefy1);
                T5 = _mm_madd_epi16(T5, mCoefy2);
                T6 = _mm_madd_epi16(T6, mCoefy3);
                T7 = _mm_madd_epi16(T7, mCoefy4);

                mVal1 = _mm_add_epi32(_mm_add_epi32(T0, T1), _mm_add_epi32(T2, T3));
                mVal2 = _mm_add_epi32(_mm_add_epi32(T4, T5), _mm_add_epi32(T6, T7));

                mVal1 = _mm_srai_epi32(_mm_add_epi32(mVal1, mAddOffset), shift);
                mVal2 = _mm_srai_epi32(_mm_add_epi32(mVal2, mAddOffset), shift);
                mVal = _mm_packs_epi32(mVal1, mVal2);
                mVal = _mm_packus_epi16(mVal, mVal);

                _mm_storel_epi64((__m128i*)&dst[col], mVal);

                p += 8;
            }

            if (col < width) {
                __m128i T00 = _mm_loadu_si128((__m128i*)(p));
                __m128i T10 = _mm_loadu_si128((__m128i*)(p + i_tmp));
                __m128i T20 = _mm_loadu_si128((__m128i*)(p + 2 * i_tmp));
                __m128i T30 = _mm_loadu_si128((__m128i*)(p + 3 * i_tmp));
                __m128i T40 = _mm_loadu_si128((__m128i*)(p + 4 * i_tmp));
                __m128i T50 = _mm_loadu_si128((__m128i*)(p + 5 * i_tmp));
                __m128i T60 = _mm_loadu_si128((__m128i*)(p + 6 * i_tmp));
                __m128i T70 = _mm_loadu_si128((__m128i*)(p + 7 * i_tmp));

                __m128i T0 = _mm_unpacklo_epi16(T00, T10);
                __m128i T1 = _mm_unpacklo_epi16(T20, T30);
                __m128i T2 = _mm_unpacklo_epi16(T40, T50);
                __m128i T3 = _mm_unpacklo_epi16(T60, T70);
                __m128i T4 = _mm_unpackhi_epi16(T00, T10);
                __m128i T5 = _mm_unpackhi_epi16(T20, T30);
                __m128i T6 = _mm_unpackhi_epi16(T40, T50);
                __m128i T7 = _mm_unpackhi_epi16(T60, T70);
                __m128i mVal1, mVal2, mVal;

                T0 = _mm_madd_epi16(T0, mCoefy1);
                T1 = _mm_madd_epi16(T1, mCoefy2);
                T2 = _mm_madd_epi16(T2, mCoefy3);
                T3 = _mm_madd_epi16(T3, mCoefy4);
                T4 = _mm_madd_epi16(T4, mCoefy1);
                T5 = _mm_madd_epi16(T5, mCoefy2);
                T6 = _mm_madd_epi16(T6, mCoefy3);
                T7 = _mm_madd_epi16(T7, mCoefy4);

                mVal1 = _mm_add_epi32(_mm_add_epi32(T0, T1), _mm_add_epi32(T2, T3));
                mVal2 = _mm_add_epi32(_mm_add_epi32(T4, T5), _mm_add_epi32(T6, T7));

                mVal1 = _mm_srai_epi32(_mm_add_epi32(mVal1, mAddOffset), shift);
                mVal2 = _mm_srai_epi32(_mm_add_epi32(mVal2, mAddOffset), shift);
                mVal = _mm_packs_epi32(mVal1, mVal2);
                mVal = _mm_packus_epi16(mVal, mVal);

                _mm_maskmoveu_si128(mVal, mask, (char *)&dst[col]);
            }

            tmp += i_tmp;
            dst += i_dst;
        }
    }
}

