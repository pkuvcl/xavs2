/*
 * intrinsic_quant.c
 *
 * Description of this file:
 *    SSE assembly functions of QUANT module of the xavs2 library
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
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE3
#include <tmmintrin.h>  // SSSE3
#include <smmintrin.h>

int quant_c_sse128(coeff_t *coef, const int i_coef, const int scale, const int shift, const int add)
{
    __m128i mScale, mAdd;
    __m128i data0, data1;
    __m128i T0, T1;
    __m128i mZero, mCount;
    int i;

    mScale = _mm_set1_epi32(scale);
    mAdd = _mm_set1_epi32(add);
    mZero = _mm_setzero_si128();
    mCount = _mm_setzero_si128();

    for (i = 0; i < i_coef; i += 16) {
        data0 = _mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i *)(coef + i)));
        data1 = _mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i *)(coef + i + 4)));

        T0 = _mm_abs_epi32(data0);
        T1 = _mm_abs_epi32(data1);

        T0 = _mm_mullo_epi32(T0, mScale);
        T1 = _mm_mullo_epi32(T1, mScale);

        T0 = _mm_add_epi32(T0, mAdd);
        T1 = _mm_add_epi32(T1, mAdd);

        T0 = _mm_srai_epi32(T0, shift);
        T1 = _mm_srai_epi32(T1, shift);

        T0 = _mm_sign_epi32(T0, data0);
        T1 = _mm_sign_epi32(T1, data1);

        T0 = _mm_packs_epi32(T0, T1);

        _mm_store_si128((__m128i *)(coef + i), T0);
        mCount = _mm_sub_epi16(mCount, _mm_cmpeq_epi16(T0, mZero));

        data0 = _mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i *)(coef + i + 8)));
        data1 = _mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i *)(coef + i + 12)));

        T0 = _mm_abs_epi32(data0);
        T1 = _mm_abs_epi32(data1);

        T0 = _mm_mullo_epi32(T0, mScale);
        T1 = _mm_mullo_epi32(T1, mScale);

        T0 = _mm_add_epi32(T0, mAdd);
        T1 = _mm_add_epi32(T1, mAdd);

        T0 = _mm_srai_epi32(T0, shift);
        T1 = _mm_srai_epi32(T1, shift);

        T0 = _mm_sign_epi32(T0, data0);
        T1 = _mm_sign_epi32(T1, data1);

        T0 = _mm_packs_epi32(T0, T1);

        _mm_store_si128((__m128i *)(coef + i + 8), T0);
        mCount = _mm_sub_epi16(mCount, _mm_cmpeq_epi16(T0, mZero));
    }
    mCount = _mm_packus_epi16(mCount, mCount);
    mCount = _mm_sad_epu8(mCount, mZero); // get the total number of 0

    return i_coef - _mm_extract_epi16(mCount, 0);
}

void dequant_c_sse128(coeff_t *coef, const int i_coef, const int scale, const int shift, const int add)
{
    __m128i mScale, mAdd;
    __m128i data0, data1;
    int i;

    mScale = _mm_set1_epi32(scale);
    mAdd = _mm_set1_epi32(add);

    for (i = 0; i < i_coef; i += 16) {
        data0 = _mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i *)(coef + i)));
        data1 = _mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i *)(coef + i + 4)));

        data0 = _mm_mullo_epi32(data0, mScale);
        data1 = _mm_mullo_epi32(data1, mScale);

        data0 = _mm_add_epi32(data0, mAdd);
        data1 = _mm_add_epi32(data1, mAdd);

        data0 = _mm_srai_epi32(data0, shift);
        data1 = _mm_srai_epi32(data1, shift);

        _mm_store_si128((__m128i *)(coef + i), _mm_packs_epi32(data0, data1));

        data0 = _mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i *)(coef + i + 8)));
        data1 = _mm_cvtepi16_epi32(_mm_loadl_epi64((__m128i *)(coef + i + 12)));

        data0 = _mm_mullo_epi32(data0, mScale);
        data1 = _mm_mullo_epi32(data1, mScale);

        data0 = _mm_add_epi32(data0, mAdd);
        data1 = _mm_add_epi32(data1, mAdd);

        data0 = _mm_srai_epi32(data0, shift);
        data1 = _mm_srai_epi32(data1, shift);

        _mm_store_si128((__m128i *)(coef + i + 8), _mm_packs_epi32(data0, data1));
    }
}

void abs_coeff_sse128(coeff_t *dst, const coeff_t *src, const int i_coef)
{
    int i;

    for (i = 0; i < i_coef; i += 16) {
        _mm_store_si128((__m128i *)(dst + i), _mm_abs_epi16(_mm_load_si128((__m128i *)(src + i))));
        _mm_store_si128((__m128i *)(dst + i + 8), _mm_abs_epi16(_mm_load_si128((__m128i *)(src + i + 8))));
    }
}

int add_sign_sse128(coeff_t *dst, const coeff_t *abs_val, const int i_coef)
{
    __m128i mDst, mAbs;
    __m128i mZero, mCount;
    int i;

    mZero = _mm_setzero_si128();
    mCount = _mm_setzero_si128();

    for (i = 0; i < i_coef; i += 16) {
        mDst = _mm_load_si128((__m128i *)(dst + i));
        mAbs = _mm_load_si128((__m128i *)(abs_val + i));
        mDst = _mm_sign_epi16(mAbs, mDst);
        _mm_store_si128((__m128i *)(dst + i), mDst);
        mCount = _mm_sub_epi16(mCount, _mm_cmpeq_epi16(mAbs, mZero));

        mDst = _mm_load_si128((__m128i *)(dst + i + 8));
        mAbs = _mm_load_si128((__m128i *)(abs_val + i + 8));
        mDst = _mm_sign_epi16(mAbs, mDst);
        _mm_store_si128((__m128i *)(dst + i + 8), mDst);
        mCount = _mm_sub_epi16(mCount, _mm_cmpeq_epi16(mAbs, mZero));
    }
    mCount = _mm_packus_epi16(mCount, mCount);
    mCount = _mm_sad_epu8(mCount, mZero);

    return i_coef - _mm_extract_epi16(mCount, 0);
}
