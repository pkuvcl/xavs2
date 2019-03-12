/*
 * intrinsic_quant_avx2.c
 *
 * Description of this file:
 *    AVX2 assembly functions of QUANT module of the xavs2 library
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

#include "../basic_types.h"
#include "intrinsic.h"

int quant_c_avx2(coeff_t *coef, const int i_coef, const int scale, const int shift, const int add)
{
    __m256i mScale, mAdd;
    __m256i data0, data1;
    __m256i T0, T1;
    __m256i mZero, mCount, mCmp;
    int i;

    mScale = _mm256_set1_epi32(scale);
    mAdd = _mm256_set1_epi32(add);
    mZero = _mm256_setzero_si256();
    mCount = _mm256_setzero_si256();

    if (i_coef == 16) {
        data1 = _mm256_load_si256((__m256i *) coef);
        data0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(data1));
        data1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data1, 0x1));

        T0 = _mm256_abs_epi32(data0);
        T1 = _mm256_abs_epi32(data1);
        T0 = _mm256_mullo_epi32(T0, mScale);
        T1 = _mm256_mullo_epi32(T1, mScale);
        T0 = _mm256_add_epi32(T0, mAdd);
        T1 = _mm256_add_epi32(T1, mAdd);
        T0 = _mm256_srai_epi32(T0, shift);
        T1 = _mm256_srai_epi32(T1, shift);
        T0 = _mm256_sign_epi32(T0, data0);
        T1 = _mm256_sign_epi32(T1, data1);

        T0 = _mm256_packs_epi32(T0, T1);
        T0 = _mm256_permute4x64_epi64(T0, 0xD8);

        mCmp = _mm256_cmpeq_epi16(T0, mZero); // for i from 1 to 8, if coeff0[i] == zero, cmp[i] = -1(0xFFFF)
        mCount = _mm256_sub_epi16(mCount, mCmp);

        _mm256_store_si256((__m256i *) coef, T0);
    } else {
        for (i = 0; i < i_coef; i += 64) {
            // 0 ~ 15
            data1 = _mm256_load_si256((__m256i *)(coef + i));
            data0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(data1));
            data1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data1, 0x1));

            T0 = _mm256_abs_epi32(data0);
            T1 = _mm256_abs_epi32(data1);
            T0 = _mm256_mullo_epi32(T0, mScale);
            T1 = _mm256_mullo_epi32(T1, mScale);
            T0 = _mm256_add_epi32(T0, mAdd);
            T1 = _mm256_add_epi32(T1, mAdd);
            T0 = _mm256_srai_epi32(T0, shift);
            T1 = _mm256_srai_epi32(T1, shift);
            T0 = _mm256_sign_epi32(T0, data0);
            T1 = _mm256_sign_epi32(T1, data1);

            T0 = _mm256_packs_epi32(T0, T1);
            T0 = _mm256_permute4x64_epi64(T0, 0xD8);

            mCmp = _mm256_cmpeq_epi16(T0, mZero);
            mCount = _mm256_sub_epi16(mCount, mCmp);

            _mm256_store_si256((__m256i *)(coef + i), T0);

            // 16 ~ 31
            data1 = _mm256_load_si256((__m256i *)(coef + i + 16));
            data0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(data1));
            data1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data1, 0x1));

            T0 = _mm256_abs_epi32(data0);
            T1 = _mm256_abs_epi32(data1);
            T0 = _mm256_mullo_epi32(T0, mScale);
            T1 = _mm256_mullo_epi32(T1, mScale);
            T0 = _mm256_add_epi32(T0, mAdd);
            T1 = _mm256_add_epi32(T1, mAdd);
            T0 = _mm256_srai_epi32(T0, shift);
            T1 = _mm256_srai_epi32(T1, shift);
            T0 = _mm256_sign_epi32(T0, data0);
            T1 = _mm256_sign_epi32(T1, data1);

            T0 = _mm256_packs_epi32(T0, T1);
            T0 = _mm256_permute4x64_epi64(T0, 0xD8);

            mCmp = _mm256_cmpeq_epi16(T0, mZero);
            mCount = _mm256_sub_epi16(mCount, mCmp);

            _mm256_store_si256((__m256i *)(coef + i + 16), T0);

            // 32 ~ 47
            data1 = _mm256_load_si256((__m256i *)(coef + i + 32));
            data0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(data1));
            data1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data1, 0x1));

            T0 = _mm256_abs_epi32(data0);
            T1 = _mm256_abs_epi32(data1);
            T0 = _mm256_mullo_epi32(T0, mScale);
            T1 = _mm256_mullo_epi32(T1, mScale);
            T0 = _mm256_add_epi32(T0, mAdd);
            T1 = _mm256_add_epi32(T1, mAdd);
            T0 = _mm256_srai_epi32(T0, shift);
            T1 = _mm256_srai_epi32(T1, shift);
            T0 = _mm256_sign_epi32(T0, data0);
            T1 = _mm256_sign_epi32(T1, data1);

            T0 = _mm256_packs_epi32(T0, T1);
            T0 = _mm256_permute4x64_epi64(T0, 0xD8);

            mCmp = _mm256_cmpeq_epi16(T0, mZero);
            mCount = _mm256_sub_epi16(mCount, mCmp);

            _mm256_store_si256((__m256i *)(coef + i + 32), T0);

            // 48 ~ 63
            data1 = _mm256_load_si256((__m256i *)(coef + i + 48));
            data0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(data1));
            data1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data1, 0x1));

            T0 = _mm256_abs_epi32(data0);
            T1 = _mm256_abs_epi32(data1);
            T0 = _mm256_mullo_epi32(T0, mScale);
            T1 = _mm256_mullo_epi32(T1, mScale);
            T0 = _mm256_add_epi32(T0, mAdd);
            T1 = _mm256_add_epi32(T1, mAdd);
            T0 = _mm256_srai_epi32(T0, shift);
            T1 = _mm256_srai_epi32(T1, shift);
            T0 = _mm256_sign_epi32(T0, data0);
            T1 = _mm256_sign_epi32(T1, data1);

            T0 = _mm256_packs_epi32(T0, T1);
            T0 = _mm256_permute4x64_epi64(T0, 0xD8);

            mCmp = _mm256_cmpeq_epi16(T0, mZero);
            mCount = _mm256_sub_epi16(mCount, mCmp);

            _mm256_store_si256((__m256i *)(coef + i + 48), T0);
        }
    }

    mCount = _mm256_packus_epi16(mCount, mCount);
    mCount = _mm256_permute4x64_epi64(mCount, 0xD8);
    mCount = _mm256_sad_epu8(mCount, mZero); // get the total number of 0

    return i_coef - _mm256_extract_epi16(mCount, 0) - _mm256_extract_epi16(mCount, 4);
}

void dequant_c_avx2(coeff_t *coef, const int i_coef, const int scale, const int shift)
{
    __m256i mScale, mAdd;
    __m256i data0, data1;
    __m256i T0, T1;
    int i;

    mScale = _mm256_set1_epi32(scale);
    mAdd = _mm256_set1_epi32(1 << (shift - 1));

    if (i_coef == 16) {
        data1 = _mm256_load_si256((__m256i *) coef);
        data0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(data1));
        data1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data1, 0x1));

        T0 = _mm256_mullo_epi32(data0, mScale);
        T1 = _mm256_mullo_epi32(data1, mScale);
        T0 = _mm256_add_epi32(T0, mAdd);
        T1 = _mm256_add_epi32(T1, mAdd);
        T0 = _mm256_srai_epi32(T0, shift);
        T1 = _mm256_srai_epi32(T1, shift);

        T0 = _mm256_permute4x64_epi64(_mm256_packs_epi32(T0, T1), 0xD8);
        _mm256_store_si256((__m256i *) coef, T0);
    } else {
        for (i = 0; i < i_coef; i += 64) {
            // 0 ~ 15
            data1 = _mm256_load_si256((__m256i *)(coef + i));
            data0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(data1));
            data1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data1, 0x1));

            T0 = _mm256_mullo_epi32(data0, mScale);
            T1 = _mm256_mullo_epi32(data1, mScale);
            T0 = _mm256_add_epi32(T0, mAdd);
            T1 = _mm256_add_epi32(T1, mAdd);
            T0 = _mm256_srai_epi32(T0, shift);
            T1 = _mm256_srai_epi32(T1, shift);

            T0 = _mm256_permute4x64_epi64(_mm256_packs_epi32(T0, T1), 0xD8);
            _mm256_store_si256((__m256i *)(coef + i), T0);

            // 16 ~ 31
            data1 = _mm256_load_si256((__m256i *)(coef + i + 16));
            data0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(data1));
            data1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data1, 0x1));

            T0 = _mm256_mullo_epi32(data0, mScale);
            T1 = _mm256_mullo_epi32(data1, mScale);
            T0 = _mm256_add_epi32(T0, mAdd);
            T1 = _mm256_add_epi32(T1, mAdd);
            T0 = _mm256_srai_epi32(T0, shift);
            T1 = _mm256_srai_epi32(T1, shift);

            T0 = _mm256_permute4x64_epi64(_mm256_packs_epi32(T0, T1), 0xD8);
            _mm256_store_si256((__m256i *)(coef + i + 16), T0);

            // 32 ~ 47
            data1 = _mm256_load_si256((__m256i *)(coef + i + 32));
            data0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(data1));
            data1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data1, 0x1));

            T0 = _mm256_mullo_epi32(data0, mScale);
            T1 = _mm256_mullo_epi32(data1, mScale);
            T0 = _mm256_add_epi32(T0, mAdd);
            T1 = _mm256_add_epi32(T1, mAdd);
            T0 = _mm256_srai_epi32(T0, shift);
            T1 = _mm256_srai_epi32(T1, shift);

            T0 = _mm256_permute4x64_epi64(_mm256_packs_epi32(T0, T1), 0xD8);
            _mm256_store_si256((__m256i *)(coef + i + 32), T0);

            // 48 ~ 63
            data1 = _mm256_load_si256((__m256i *)(coef + i + 48));
            data0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(data1));
            data1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data1, 0x1));

            T0 = _mm256_mullo_epi32(data0, mScale);
            T1 = _mm256_mullo_epi32(data1, mScale);
            T0 = _mm256_add_epi32(T0, mAdd);
            T1 = _mm256_add_epi32(T1, mAdd);
            T0 = _mm256_srai_epi32(T0, shift);
            T1 = _mm256_srai_epi32(T1, shift);

            T0 = _mm256_permute4x64_epi64(_mm256_packs_epi32(T0, T1), 0xD8);
            _mm256_store_si256((__m256i *)(coef + i + 48), T0);
        }
    }
}

void abs_coeff_avx2(coeff_t *dst, const coeff_t *src, const int i_coef)
{
    int i;

    if (i_coef == 16) {
        _mm256_store_si256((__m256i *) dst, _mm256_abs_epi16(_mm256_load_si256((__m256i *) src)));
    } else {
        for (i = 0; i < i_coef; i += 64) {
            _mm256_store_si256((__m256i *)(dst + i), _mm256_abs_epi16(_mm256_load_si256((__m256i *)(src + i))));
            _mm256_store_si256((__m256i *)(dst + i + 16), _mm256_abs_epi16(_mm256_load_si256((__m256i *)(src + i + 16))));
            _mm256_store_si256((__m256i *)(dst + i + 32), _mm256_abs_epi16(_mm256_load_si256((__m256i *)(src + i + 32))));
            _mm256_store_si256((__m256i *)(dst + i + 48), _mm256_abs_epi16(_mm256_load_si256((__m256i *)(src + i + 48))));
        }
    }
}

int add_sign_avx2(coeff_t *dst, const coeff_t *abs_val, const int i_coef)
{
    __m256i mDst, mAbs;
    __m256i mZero, mCount;
    int i;

    mZero = _mm256_setzero_si256();
    mCount = _mm256_setzero_si256();

    if (i_coef == 16) {
        mDst = _mm256_load_si256((__m256i *) dst);
        mAbs = _mm256_load_si256((__m256i *) abs_val);

        mDst = _mm256_sign_epi16(mAbs, mDst);
        mCount = _mm256_sub_epi16(mCount, _mm256_cmpeq_epi16(mAbs, mZero));

        _mm256_store_si256((__m256i *) dst, mDst);
    } else {
        for (i = 0; i < i_coef; i += 64) {
            // 0 ~ 15
            mDst = _mm256_load_si256((__m256i *)(dst + i));
            mAbs = _mm256_load_si256((__m256i *)(abs_val + i));

            mDst = _mm256_sign_epi16(mAbs, mDst);
            mCount = _mm256_sub_epi16(mCount, _mm256_cmpeq_epi16(mAbs, mZero));

            _mm256_store_si256((__m256i *)(dst + i), mDst);

            // 16 ~ 31
            mDst = _mm256_load_si256((__m256i *)(dst + i + 16));
            mAbs = _mm256_load_si256((__m256i *)(abs_val + i + 16));

            mDst = _mm256_sign_epi16(mAbs, mDst);
            mCount = _mm256_sub_epi16(mCount, _mm256_cmpeq_epi16(mAbs, mZero));

            _mm256_store_si256((__m256i *)(dst + i + 16), mDst);

            // 32 ~ 47
            mDst = _mm256_load_si256((__m256i *)(dst + i + 32));
            mAbs = _mm256_load_si256((__m256i *)(abs_val + i + 32));

            mDst = _mm256_sign_epi16(mAbs, mDst);
            mCount = _mm256_sub_epi16(mCount, _mm256_cmpeq_epi16(mAbs, mZero));

            _mm256_store_si256((__m256i *)(dst + i + 32), mDst);

            // 48 ~ 63
            mDst = _mm256_load_si256((__m256i *)(dst + i + 48));
            mAbs = _mm256_load_si256((__m256i *)(abs_val + i + 48));

            mDst = _mm256_sign_epi16(mAbs, mDst);
            mCount = _mm256_sub_epi16(mCount, _mm256_cmpeq_epi16(mAbs, mZero));

            _mm256_store_si256((__m256i *)(dst + i + 48), mDst);
        }
    }

    mCount = _mm256_permute4x64_epi64(_mm256_packus_epi16(mCount, mCount), 0xD8);
    mCount = _mm256_sad_epu8(mCount, mZero);

    return i_coef - _mm256_extract_epi16(mCount, 0) - _mm256_extract_epi16(mCount, 4);
}
