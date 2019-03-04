/*
 * intrinsic_deblock_avx2.c
 *
 * Description of this file:
 *    AVX2 assembly functions of Deblock module of the xavs2 library
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

void deblock_edge_ver_avx2(pel_t *SrcPtr, int stride, int Alpha, int Beta, uint8_t *flt_flag)
{
    pel_t *pTmp = SrcPtr - 4;
    int flag0 = flt_flag[0] ? -1 : 0;
    int flag1 = flt_flag[1] ? -1 : 0;
    __m128i TL0, TL1, TL2, TL3;
    __m128i TR0, TR1, TR2, TR3;
    __m128i T0, T1, T2, T3, T4, T5, T6, T7;
    __m128i M0, M1;
    __m128i FLT, FS;
    __m128i FS3, FS4, FS56;
    __m256i TLR0, TLR1, TLR2; // store TL* and TR*
    __m256i TRL0, TRL1, TRL2; // store TR* and TL*
    __m256i T0_256, T1_256, T2_256;
    __m256i FLT_LR;
    __m256i TLR0w, TLR1w;
    __m256i FS_256;

    __m128i ALPHA = _mm_set1_epi16((pel_t)Alpha);
    __m128i BETA = _mm_set1_epi16((pel_t)Beta);
    __m128i c_0 = _mm_set1_epi16(0);
    __m256i c_1_256 = _mm256_set1_epi16(1);
    __m256i c_2_256 = _mm256_set1_epi16(2);
    __m256i c_3_256 = _mm256_set1_epi16(3);
    __m256i c_4_256 = _mm256_set1_epi16(4);
    __m256i c_8_256 = _mm256_set1_epi16(8);
    __m256i c_16_256 = _mm256_set1_epi16(16);
    __m256i BETA_256 = _mm256_set1_epi16((short)Beta);

    T0 = _mm_loadl_epi64((__m128i*)(pTmp));
    T1 = _mm_loadl_epi64((__m128i*)(pTmp + stride));
    T2 = _mm_loadl_epi64((__m128i*)(pTmp + stride * 2));
    T3 = _mm_loadl_epi64((__m128i*)(pTmp + stride * 3));
    T4 = _mm_loadl_epi64((__m128i*)(pTmp + stride * 4));
    T5 = _mm_loadl_epi64((__m128i*)(pTmp + stride * 5));
    T6 = _mm_loadl_epi64((__m128i*)(pTmp + stride * 6));
    T7 = _mm_loadl_epi64((__m128i*)(pTmp + stride * 7));

    //--------------- transpose -------------------------------
    T0 = _mm_unpacklo_epi8(T0, T1);
    T1 = _mm_unpacklo_epi8(T2, T3);
    T2 = _mm_unpacklo_epi8(T4, T5);
    T3 = _mm_unpacklo_epi8(T6, T7);

    T4 = _mm_unpacklo_epi16(T0, T1);
    T5 = _mm_unpacklo_epi16(T2, T3);
    T6 = _mm_unpackhi_epi16(T0, T1);
    T7 = _mm_unpackhi_epi16(T2, T3);

    /*
    TLR0 = _mm256_inserti128_si256(_mm256_castsi128_si256(T4), T6, 1);
    TLR1 = _mm256_inserti128_si256(_mm256_castsi128_si256(T5), T7, 1);

    TLR0w = _mm256_unpacklo_epi32(TLR0, TLR1);      //T0 T2
    TLR1w = _mm256_unpackhi_epi32(TLR0, TLR1);      //T1 T3

    TLR3 = _mm256_unpacklo_epi8(TLR0w, c_0_256);    //TL3 TR0
    TLR2 = _mm256_unpackhi_epi8(TLR0w, c_0_256);    //TL2 TR1
    TLR1 = _mm256_unpacklo_epi8(TLR1w, c_0_256);    //TL1 TR2
    TLR0 = _mm256_unpackhi_epi8(TLR1w, c_0_256);    //TL0 TR3

    TR0 = _mm256_extracti128_si256(TLR3, 0x01);
    TR1 = _mm256_extracti128_si256(TLR2, 0x01);
    TR2 = _mm256_extracti128_si256(TLR1, 0x01);
    TR3 = _mm256_extracti128_si256(TLR0, 0x01);

    TLR0 = _mm256_inserti128_si256(TLR0, TR0, 1);
    TLR1 = _mm256_inserti128_si256(TLR1, TR1, 1);
    TLR2 = _mm256_inserti128_si256(TLR2, TR2, 1);
    TRL0 = _mm256_inserti128_si256(_mm256_castsi128_si256(TR0), _mm256_castsi256_si128(TLR0), 1);
    TRL1 = _mm256_inserti128_si256(_mm256_castsi128_si256(TR1), _mm256_castsi256_si128(TLR1), 1);
    */

    T0 = _mm_unpacklo_epi32(T4, T5);
    T1 = _mm_unpackhi_epi32(T4, T5);
    T2 = _mm_unpacklo_epi32(T6, T7);
    T3 = _mm_unpackhi_epi32(T6, T7);

    TL3 = _mm_unpacklo_epi8(T0, c_0);
    TL2 = _mm_unpackhi_epi8(T0, c_0);
    TL1 = _mm_unpacklo_epi8(T1, c_0);
    TL0 = _mm_unpackhi_epi8(T1, c_0);

    TR0 = _mm_unpacklo_epi8(T2, c_0);
    TR1 = _mm_unpackhi_epi8(T2, c_0);
    TR2 = _mm_unpacklo_epi8(T3, c_0);
    TR3 = _mm_unpackhi_epi8(T3, c_0);

    TLR0 = _mm256_inserti128_si256(_mm256_castsi128_si256(TL0), TR0, 1);
    TLR1 = _mm256_inserti128_si256(_mm256_castsi128_si256(TL1), TR1, 1);
    TLR2 = _mm256_inserti128_si256(_mm256_castsi128_si256(TL2), TR2, 1);
    TRL0 = _mm256_inserti128_si256(_mm256_castsi128_si256(TR0), TL0, 1);
    TRL1 = _mm256_inserti128_si256(_mm256_castsi128_si256(TR1), TL1, 1);

    T0 = _mm_abs_epi16(_mm_subs_epi16(TL0, TR0));
    T1 = _mm_cmpgt_epi16(T0, _mm256_castsi256_si128(c_1_256));
    T2 = _mm_cmpgt_epi16(ALPHA, T0);

    M0 = _mm_set_epi32(flag1, flag1, flag0, flag0);
    M0 = _mm_and_si128(M0, _mm_and_si128(T1, T2)); // mask1

    T0_256 = _mm256_abs_epi16(_mm256_subs_epi16(TLR1, TLR0));
    FLT_LR = _mm256_and_si256(_mm256_cmpgt_epi16(BETA_256, T0_256), c_2_256);

    T1_256 = _mm256_abs_epi16(_mm256_subs_epi16(TLR2, TLR0));
    T2_256 = _mm256_cmpgt_epi16(BETA_256, T1_256);

    FLT_LR = _mm256_add_epi16(_mm256_and_si256(T2_256, c_1_256), FLT_LR);
    FLT = _mm_add_epi16(_mm256_castsi256_si128(FLT_LR), _mm256_extracti128_si256(FLT_LR, 0x01));

    T0_256 = _mm256_cmpeq_epi16(TLR1, TLR0);
    M1 = _mm_and_si128(_mm256_castsi256_si128(T0_256), _mm256_extracti128_si256(T0_256, 0x01));
    T0 = _mm_subs_epi16(FLT, _mm256_castsi256_si128(c_2_256));
    T1 = _mm_subs_epi16(FLT, _mm256_castsi256_si128(c_3_256));

    T2 = _mm_abs_epi16(_mm_subs_epi16(TL1, TR1));

    FS56 = _mm_blendv_epi8(T1, T0, M1);
    FS4 = _mm_blendv_epi8(_mm256_castsi256_si128(c_1_256), _mm256_castsi256_si128(c_2_256), _mm_cmpeq_epi16(_mm256_castsi256_si128(FLT_LR), _mm256_castsi256_si128(c_2_256)));
    FS3 = _mm_blendv_epi8(c_0, _mm256_castsi256_si128(c_1_256), _mm_cmpgt_epi16(BETA, T2));

    FS = _mm_blendv_epi8(c_0, FS56, _mm_cmpgt_epi16(FLT, _mm256_castsi256_si128(c_4_256)));
    FS = _mm_blendv_epi8(FS, FS4, _mm_cmpeq_epi16(FLT, _mm256_castsi256_si128(c_4_256)));
    FS = _mm_blendv_epi8(FS, FS3, _mm_cmpeq_epi16(FLT, _mm256_castsi256_si128(c_3_256)));

    FS = _mm_and_si128(FS, M0);
    FS_256 = _mm256_inserti128_si256(_mm256_castsi128_si256(FS), FS, 1);


    TLR0w = TLR0;
    TLR1w = TLR1;
    /* fs == 1 */
    T2 = _mm_add_epi16(_mm_add_epi16(TL0, TR0), _mm256_castsi256_si128(c_2_256)); // L0 + R0 + 2
    T2_256 = _mm256_castsi128_si256(T2);
    T2_256 = _mm256_inserti128_si256(T2_256, T2, 1); // save
    T1_256 = _mm256_srli_epi16(_mm256_add_epi16(_mm256_slli_epi16(TLR0, 1), T2_256), 2);
    TLR0w = _mm256_blendv_epi8(TLR0, T1_256, _mm256_cmpeq_epi16(FS_256, c_1_256));

    /* fs == 2 */
    T2_256 = _mm256_slli_epi16(T2_256, 1);
    T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR1, 1), _mm256_add_epi16(TLR1, TRL0));
    T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR0, 3), _mm256_add_epi16(T0_256, T2_256));
    T1_256 = _mm256_srli_epi16(_mm256_add_epi16(T0_256, c_4_256), 4);
    TLR0w = _mm256_blendv_epi8(TLR0w, T1_256, _mm256_cmpeq_epi16(FS_256, c_2_256));

    /* fs == 3 */
    T2_256 = _mm256_slli_epi16(T2_256, 1); // (L0 << 2) + (R0 << 2) + 8
    T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR1, 2), _mm256_add_epi16(TLR2, TRL1));
    T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR0, 1), _mm256_add_epi16(T0_256, T2_256));
    T1_256 = _mm256_srli_epi16(T0_256, 4);
    TLR0w = _mm256_blendv_epi8(TLR0w, T1_256, _mm256_cmpeq_epi16(FS_256, c_3_256));

    T0_256 = _mm256_add_epi16(_mm256_add_epi16(TLR2, TRL0), _mm256_slli_epi16(TLR2, 1));
    T0_256 = _mm256_add_epi16(T0_256, _mm256_slli_epi16(TLR1, 3));
    T0_256 = _mm256_add_epi16(T0_256, _mm256_slli_epi16(TLR0, 2));
    T1_256 = _mm256_srli_epi16(_mm256_add_epi16(T0_256, c_8_256), 4);

    TLR1w = _mm256_blendv_epi8(TLR1w, T1_256, _mm256_cmpeq_epi16(FS_256, c_3_256));

    FS = _mm_cmpeq_epi16(FS, _mm256_castsi256_si128(c_4_256));

    if (_mm_extract_epi64(FS, 0) || _mm_extract_epi64(FS, 1)) { /* fs == 4 */
        TRL2 = _mm256_inserti128_si256(_mm256_castsi128_si256(TR2), TL2, 1);
        FS_256 = _mm256_inserti128_si256(_mm256_castsi128_si256(FS), FS, 1);

        /* cal L0/R0 */
        T0_256 = _mm256_slli_epi16(_mm256_add_epi16(_mm256_add_epi16(TLR0, TLR2), TRL0), 3);
        T0_256 = _mm256_add_epi16(_mm256_add_epi16(T0_256, c_16_256), _mm256_add_epi16(TLR0, TLR2));
        T2_256 = _mm256_add_epi16(_mm256_slli_epi16(TRL2, 1), _mm256_slli_epi16(TRL2, 2));
        T1_256 = _mm256_srli_epi16(_mm256_add_epi16(T0_256, T2_256), 5);

        TLR0w = _mm256_blendv_epi8(TLR0w, T1_256, FS_256);

        /* cal L1/R1 */
        T0_256 = _mm256_slli_epi16(_mm256_add_epi16(TLR2, TRL0), 1);
        T0_256 = _mm256_add_epi16(T0_256, _mm256_sub_epi16(_mm256_slli_epi16(TLR0, 3), TLR0));
        T2_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR2, 2), _mm256_add_epi16(TRL0, c_8_256));
        T1_256 = _mm256_srli_epi16(_mm256_add_epi16(T0_256, T2_256), 4);

        TLR1w = _mm256_blendv_epi8(TLR1w, T1_256, FS_256);

        /* cal L2/R2 */
        T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR2, 1), TLR2);
        T2_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR0, 2), TRL0);
        T1_256 = _mm256_srli_epi16(_mm256_add_epi16(T0_256, _mm256_add_epi16(T2_256, c_4_256)), 3);

        TLR2 = _mm256_blendv_epi8(TLR2, T1_256, FS_256);

    }

    /* stroe result */
    T4 = _mm_packus_epi16(TL3, _mm256_extracti128_si256(TLR0w, 0x01));
    T5 = _mm_packus_epi16(_mm256_castsi256_si128(TLR2), _mm256_extracti128_si256(TLR1w, 0x01));
    T6 = _mm_packus_epi16(_mm256_castsi256_si128(TLR1w), _mm256_extracti128_si256(TLR2, 0x01));
    T7 = _mm_packus_epi16(_mm256_castsi256_si128(TLR0w), TR3);

    T0 = _mm_unpacklo_epi8(T4, T5);
    T1 = _mm_unpacklo_epi8(T6, T7);
    T2 = _mm_unpackhi_epi8(T4, T5);
    T3 = _mm_unpackhi_epi8(T6, T7);

    T4 = _mm_unpacklo_epi16(T0, T1);
    T5 = _mm_unpacklo_epi16(T2, T3);
    T6 = _mm_unpackhi_epi16(T0, T1);
    T7 = _mm_unpackhi_epi16(T2, T3);

    T0 = _mm_unpacklo_epi32(T4, T5);
    T1 = _mm_unpackhi_epi32(T4, T5);
    T2 = _mm_unpacklo_epi32(T6, T7);
    T3 = _mm_unpackhi_epi32(T6, T7);

    pTmp = SrcPtr - 4;
    _mm_storel_epi64((__m128i*)(pTmp), T0);
    pTmp += stride;
    _mm_storel_epi64((__m128i*)(pTmp), _mm_srli_si128(T0, 8));
    pTmp += stride;
    _mm_storel_epi64((__m128i*)(pTmp), T1);
    pTmp += stride;
    _mm_storel_epi64((__m128i*)(pTmp), _mm_srli_si128(T1, 8));
    pTmp += stride;
    _mm_storel_epi64((__m128i*)(pTmp), T2);
    pTmp += stride;
    _mm_storel_epi64((__m128i*)(pTmp), _mm_srli_si128(T2, 8));
    pTmp += stride;
    _mm_storel_epi64((__m128i*)(pTmp), T3);
    pTmp += stride;
    _mm_storel_epi64((__m128i*)(pTmp), _mm_srli_si128(T3, 8));
}


void deblock_edge_ver_c_avx2(pel_t *SrcPtrU, pel_t *SrcPtrV, int stride, int Alpha, int Beta, uint8_t *flt_flag)
{
    pel_t *pTmp;
    int flag0 = flt_flag[0] ? -1 : 0;
    int flag1 = flt_flag[1] ? -1 : 0;

    __m128i TL0, TL1, TL2, TL3;
    __m128i TR0, TR1, TR2, TR3;
    __m128i T0, T1, T2, T3, T4, T5, T6, T7;
    __m128i M0, M1;
    __m128i FLT, FS;
    __m128i FS4, FS56;
    __m256i TLR0, TLR1, TLR2; // store TL* and TR*
    __m256i TRL0, TRL1; // store TR* and TL*
    __m256i T0_256, T1_256, T2_256;
    __m256i FLT_X;
    __m256i TLR0w, TLR1w;
    __m256i FS_256;

    __m128i ALPHA = _mm_set1_epi16((pel_t)Alpha);
    __m128i c_0 = _mm_set1_epi16(0);
    __m256i c_1_256 = _mm256_set1_epi16(1);
    __m256i c_2_256 = _mm256_set1_epi16(2);
    __m256i c_3_256 = _mm256_set1_epi16(3);
    __m256i c_4_256 = _mm256_set1_epi16(4);
    __m256i c_8_256 = _mm256_set1_epi16(8);
    __m256i BETA_256 = _mm256_set1_epi16((short)Beta);

    pTmp = SrcPtrU - 4;
    T0 = _mm_loadl_epi64((__m128i*)(pTmp));
    T1 = _mm_loadl_epi64((__m128i*)(pTmp + stride));
    T2 = _mm_loadl_epi64((__m128i*)(pTmp + stride * 2));
    T3 = _mm_loadl_epi64((__m128i*)(pTmp + stride * 3));

    pTmp = SrcPtrV - 4;
    T4 = _mm_loadl_epi64((__m128i*)(pTmp));
    T5 = _mm_loadl_epi64((__m128i*)(pTmp + stride));
    T6 = _mm_loadl_epi64((__m128i*)(pTmp + stride * 2));
    T7 = _mm_loadl_epi64((__m128i*)(pTmp + stride * 3));

    T0 = _mm_unpacklo_epi8(T0, T1);
    T1 = _mm_unpacklo_epi8(T2, T3);
    T2 = _mm_unpacklo_epi8(T4, T5);
    T3 = _mm_unpacklo_epi8(T6, T7);

    T4 = _mm_unpacklo_epi16(T0, T1);
    T5 = _mm_unpacklo_epi16(T2, T3);
    T6 = _mm_unpackhi_epi16(T0, T1);
    T7 = _mm_unpackhi_epi16(T2, T3);

    T0 = _mm_unpacklo_epi32(T4, T5);
    T1 = _mm_unpackhi_epi32(T4, T5);
    T2 = _mm_unpacklo_epi32(T6, T7);
    T3 = _mm_unpackhi_epi32(T6, T7);

    TL3 = _mm_unpacklo_epi8(T0, c_0);
    TL2 = _mm_unpackhi_epi8(T0, c_0);
    TL1 = _mm_unpacklo_epi8(T1, c_0);
    TL0 = _mm_unpackhi_epi8(T1, c_0);

    TR0 = _mm_unpacklo_epi8(T2, c_0);
    TR1 = _mm_unpackhi_epi8(T2, c_0);
    TR2 = _mm_unpacklo_epi8(T3, c_0);
    TR3 = _mm_unpackhi_epi8(T3, c_0);

    TLR0 = _mm256_inserti128_si256(_mm256_castsi128_si256(TL0), TR0, 1);
    TLR1 = _mm256_inserti128_si256(_mm256_castsi128_si256(TL1), TR1, 1);
    TLR2 = _mm256_inserti128_si256(_mm256_castsi128_si256(TL2), TR2, 1);
    TRL0 = _mm256_inserti128_si256(_mm256_castsi128_si256(TR0), TL0, 1);
    TRL1 = _mm256_inserti128_si256(_mm256_castsi128_si256(TR1), TL1, 1);

    T0 = _mm_abs_epi16(_mm_subs_epi16(_mm256_castsi256_si128(TLR0), _mm256_castsi256_si128(TRL0)));
    T1 = _mm_cmpgt_epi16(T0, _mm256_castsi256_si128(c_1_256));
    T2 = _mm_cmpgt_epi16(ALPHA, T0);

    M0 = _mm_set_epi32(flag1, flag0, flag1, flag0);
    M0 = _mm_and_si128(M0, _mm_and_si128(T1, T2)); // mask1

    T0_256 = _mm256_abs_epi16(_mm256_subs_epi16(TLR1, TLR0));

    FLT_X = _mm256_and_si256(_mm256_cmpgt_epi16(BETA_256, T0_256), c_2_256);

    T0_256 = _mm256_abs_epi16(_mm256_subs_epi16(TLR2, TLR0));
    T1_256 = _mm256_and_si256(_mm256_cmpgt_epi16(BETA_256, T0_256), c_1_256);

    FLT_X = _mm256_add_epi16(T1_256, FLT_X);
    FLT = _mm_add_epi16(_mm256_castsi256_si128(FLT_X), _mm256_extracti128_si256(FLT_X, 0x01));

    T0_256 = _mm256_cmpeq_epi16(TLR1, TLR0);
    M1 = _mm_and_si128(_mm256_castsi256_si128(T0_256), _mm256_extracti128_si256(T0_256, 0x01));
    T0 = _mm_subs_epi16(FLT, _mm256_castsi256_si128(c_3_256));
    T1 = _mm_subs_epi16(FLT, _mm256_castsi256_si128(c_4_256));

    FS56 = _mm_blendv_epi8(T1, T0, M1);
    FS4 = _mm_blendv_epi8(c_0, _mm256_castsi256_si128(c_1_256), _mm_cmpeq_epi16(_mm256_castsi256_si128(FLT_X), _mm256_castsi256_si128(c_2_256)));

    FS = _mm_blendv_epi8(c_0, FS56, _mm_cmpgt_epi16(FLT, _mm256_castsi256_si128(c_4_256)));
    FS = _mm_blendv_epi8(FS, FS4, _mm_cmpeq_epi16(FLT, _mm256_castsi256_si128(c_4_256)));

    FS = _mm_and_si128(FS, M0);
    FS_256 = _mm256_inserti128_si256(_mm256_castsi128_si256(FS), FS, 1);

    TLR0w = TLR0;
    TLR1w = TLR1;
    /* fs == 1 */
    T2 = _mm_add_epi16(_mm_add_epi16(_mm256_castsi256_si128(TLR0), _mm256_castsi256_si128(TRL0)), _mm256_castsi256_si128(c_2_256)); // L0 + R0 + 2
    T2_256 = _mm256_castsi128_si256(T2);
    T2_256 = _mm256_inserti128_si256(T2_256, T2, 1); // save
    T1_256 = _mm256_srli_epi16(_mm256_add_epi16(_mm256_slli_epi16(TLR0, 1), T2_256), 2);
    TLR0w = _mm256_blendv_epi8(TLR0, T1_256, _mm256_cmpeq_epi16(FS_256, c_1_256));

    /* fs == 2 */
    T2_256 = _mm256_slli_epi16(T2_256, 1);
    T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR1, 1), _mm256_add_epi16(TLR1, TRL0));
    T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR0, 3), _mm256_add_epi16(T0_256, T2_256));
    T1_256 = _mm256_srli_epi16(_mm256_add_epi16(T0_256, c_4_256), 4);
    TLR0w = _mm256_blendv_epi8(TLR0w, T1_256, _mm256_cmpeq_epi16(FS_256, c_2_256));

    /* fs == 3 */
    T2_256 = _mm256_slli_epi16(T2_256, 1); // (L0 << 2) + (R0 << 2) + 8
    T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR1, 2), _mm256_add_epi16(TLR2, TRL1));
    T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR0, 1), _mm256_add_epi16(T0_256, T2_256));
    T1_256 = _mm256_srli_epi16(T0_256, 4);
    TLR0w = _mm256_blendv_epi8(TLR0w, T1_256, _mm256_cmpeq_epi16(FS_256, c_3_256));

    T0_256 = _mm256_add_epi16(_mm256_add_epi16(TLR2, TRL0), _mm256_slli_epi16(TLR2, 1));
    T0_256 = _mm256_add_epi16(T0_256, _mm256_slli_epi16(TLR1, 3));
    T0_256 = _mm256_add_epi16(T0_256, _mm256_slli_epi16(TLR0, 2));
    T1_256 = _mm256_srli_epi16(_mm256_add_epi16(T0_256, c_8_256), 4);

    TLR1w = _mm256_blendv_epi8(TLR1w, T1_256, _mm256_cmpeq_epi16(FS_256, c_3_256));

    /* stroe result */
    T4 = _mm_packus_epi16(TL3, _mm256_extracti128_si256(TLR0w, 0x01));
    T5 = _mm_packus_epi16(TL2, _mm256_extracti128_si256(TLR1w, 0x01));
    T6 = _mm_packus_epi16(_mm256_castsi256_si128(TLR1w), TR2);
    T7 = _mm_packus_epi16(_mm256_castsi256_si128(TLR0w), TR3);

    T0 = _mm_unpacklo_epi8(T4, T5);
    T1 = _mm_unpacklo_epi8(T6, T7);
    T2 = _mm_unpackhi_epi8(T4, T5);
    T3 = _mm_unpackhi_epi8(T6, T7);

    T4 = _mm_unpacklo_epi16(T0, T1);
    T5 = _mm_unpacklo_epi16(T2, T3);
    T6 = _mm_unpackhi_epi16(T0, T1);
    T7 = _mm_unpackhi_epi16(T2, T3);

    T0 = _mm_unpacklo_epi32(T4, T5);
    T1 = _mm_unpackhi_epi32(T4, T5);
    T2 = _mm_unpacklo_epi32(T6, T7);
    T3 = _mm_unpackhi_epi32(T6, T7);

    pTmp = SrcPtrU - 4;
    _mm_storel_epi64((__m128i*)(pTmp), T0);
    _mm_storel_epi64((__m128i*)(pTmp + stride), _mm_srli_si128(T0, 8));
    _mm_storel_epi64((__m128i*)(pTmp + (stride << 1)), T1);
    _mm_storel_epi64((__m128i*)(pTmp + stride * 3), _mm_srli_si128(T1, 8));

    pTmp = SrcPtrV - 4;
    _mm_storel_epi64((__m128i*)(pTmp), T2);
    _mm_storel_epi64((__m128i*)(pTmp + stride), _mm_srli_si128(T2, 8));
    _mm_storel_epi64((__m128i*)(pTmp + (stride << 1)), T3);
    _mm_storel_epi64((__m128i*)(pTmp + stride * 3), _mm_srli_si128(T3, 8));

}


void deblock_edge_hor_avx2(pel_t *SrcPtr, int stride, int Alpha, int Beta, uint8_t *flt_flag)
{

    int inc = stride;
    int inc2 = inc << 1;
    int inc3 = inc + inc2;
    int flag0 = flt_flag[0] ? -1 : 0;
    int flag1 = flt_flag[1] ? -1 : 0;

    __m128i TL0, TL1, TL2;
    __m128i TR0, TR1, TR2;
    __m128i T0, T1, T2;
    __m128i M0, M1;
    __m128i FLT, FS;
    __m128i FS3, FS4, FS56;
    __m256i TLR0, TLR1, TLR2; // store TL* and TR*
    __m256i TRL0, TRL1, TRL2; // store TR* and TL*
    __m256i T0_256, T1_256, T2_256;
    __m256i FLT_X;
    __m256i TLR0w, TLR1w;
    __m256i FS_256;

    __m128i ALPHA = _mm_set1_epi16((short)Alpha);
    __m128i BETA = _mm_set1_epi16((short)Beta);
    __m128i c_0 = _mm_set1_epi16(0);
    __m256i c_0_256 = _mm256_setzero_si256();
    __m256i c_1_256 = _mm256_set1_epi16(1);
    __m256i c_2_256 = _mm256_set1_epi16(2);
    __m256i c_3_256 = _mm256_set1_epi16(3);
    __m256i c_4_256 = _mm256_set1_epi16(4);
    __m256i c_8_256 = _mm256_set1_epi16(8);
    __m256i c_16_256 = _mm256_set1_epi16(16);
    __m256i BETA_256 = _mm256_set1_epi16((short)Beta);

    TL2 = _mm_loadl_epi64((__m128i*)(SrcPtr - inc3));
    TL1 = _mm_loadl_epi64((__m128i*)(SrcPtr - inc2));
    TL0 = _mm_loadl_epi64((__m128i*)(SrcPtr - inc));
    TR0 = _mm_loadl_epi64((__m128i*)(SrcPtr + 0));
    TR1 = _mm_loadl_epi64((__m128i*)(SrcPtr + inc));
    TR2 = _mm_loadl_epi64((__m128i*)(SrcPtr + inc2));

    TL2 = _mm_unpacklo_epi8(TL2, c_0);
    TL1 = _mm_unpacklo_epi8(TL1, c_0);
    TL0 = _mm_unpacklo_epi8(TL0, c_0);
    TR0 = _mm_unpacklo_epi8(TR0, c_0);
    TR1 = _mm_unpacklo_epi8(TR1, c_0);
    TR2 = _mm_unpacklo_epi8(TR2, c_0);

    TLR0 = _mm256_inserti128_si256(_mm256_castsi128_si256(TL0), TR0, 1);
    TLR1 = _mm256_inserti128_si256(_mm256_castsi128_si256(TL1), TR1, 1);
    TLR2 = _mm256_inserti128_si256(_mm256_castsi128_si256(TL2), TR2, 1);
    TRL0 = _mm256_inserti128_si256(_mm256_castsi128_si256(TR0), TL0, 1);
    TRL1 = _mm256_inserti128_si256(_mm256_castsi128_si256(TR1), TL1, 1);

    T0 = _mm_abs_epi16(_mm_subs_epi16(TL0, TR0));
    T1 = _mm_cmpgt_epi16(T0, _mm256_castsi256_si128(c_1_256));
    T2 = _mm_cmpgt_epi16(ALPHA, T0);

    M0 = _mm_set_epi32(flag1, flag1, flag0, flag0);
    M0 = _mm_and_si128(M0, _mm_and_si128(T1, T2)); // mask1

    T0_256 = _mm256_abs_epi16(_mm256_subs_epi16(TLR1, TLR0));

    FLT_X = _mm256_and_si256(_mm256_cmpgt_epi16(BETA_256, T0_256), c_2_256);

    T0_256 = _mm256_abs_epi16(_mm256_subs_epi16(TLR2, TLR0));
    T1_256 = _mm256_and_si256(_mm256_cmpgt_epi16(BETA_256, T0_256), c_1_256);

    FLT_X = _mm256_add_epi16(T1_256, FLT_X);
    FLT = _mm_add_epi16(_mm256_castsi256_si128(FLT_X), _mm256_extracti128_si256(FLT_X, 0x01));

    T0_256 = _mm256_cmpeq_epi16(TLR1, TLR0);
    M1 = _mm_and_si128(_mm256_castsi256_si128(T0_256), _mm256_extracti128_si256(T0_256, 0x01));
    T0 = _mm_subs_epi16(FLT, _mm256_castsi256_si128(c_2_256));
    T1 = _mm_subs_epi16(FLT, _mm256_castsi256_si128(c_3_256));

    T2 = _mm_abs_epi16(_mm_subs_epi16(TL1, TR1));

    FS56 = _mm_blendv_epi8(T1, T0, M1);
    FS4 = _mm_blendv_epi8(_mm256_castsi256_si128(c_1_256), _mm256_castsi256_si128(c_2_256), _mm_cmpeq_epi16(_mm256_castsi256_si128(FLT_X), _mm256_castsi256_si128(c_2_256)));
    FS3 = _mm_blendv_epi8(c_0, _mm256_castsi256_si128(c_1_256), _mm_cmpgt_epi16(BETA, T2));

    FS = _mm_blendv_epi8(c_0, FS56, _mm_cmpgt_epi16(FLT, _mm256_castsi256_si128(c_4_256)));
    FS = _mm_blendv_epi8(FS, FS4, _mm_cmpeq_epi16(FLT, _mm256_castsi256_si128(c_4_256)));
    FS = _mm_blendv_epi8(FS, FS3, _mm_cmpeq_epi16(FLT, _mm256_castsi256_si128(c_3_256)));

    FS = _mm_and_si128(FS, M0);
    FS_256 = _mm256_inserti128_si256(_mm256_castsi128_si256(FS), FS, 1);

    TLR0w = TLR0;
    TLR1w = TLR1;
    /* fs == 1 */
    T2 = _mm_add_epi16(_mm_add_epi16(TL0, TR0), _mm256_castsi256_si128(c_2_256)); // L0 + R0 + 2
    T2_256 = _mm256_castsi128_si256(T2);
    T2_256 = _mm256_inserti128_si256(T2_256, T2, 1); // save
    T1_256 = _mm256_srli_epi16(_mm256_add_epi16(_mm256_slli_epi16(TLR0, 1), T2_256), 2);
    TLR0w = _mm256_blendv_epi8(TLR0, T1_256, _mm256_cmpeq_epi16(FS_256, c_1_256));

    /* fs == 2 */
    T2_256 = _mm256_slli_epi16(T2_256, 1);
    T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR1, 1), _mm256_add_epi16(TLR1, TRL0));
    T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR0, 3), _mm256_add_epi16(T0_256, T2_256));
    T1_256 = _mm256_srli_epi16(_mm256_add_epi16(T0_256, c_4_256), 4);
    TLR0w = _mm256_blendv_epi8(TLR0w, T1_256, _mm256_cmpeq_epi16(FS_256, c_2_256));

    /* fs == 3 */
    T2_256 = _mm256_slli_epi16(T2_256, 1); // (L0 << 2) + (R0 << 2) + 8
    T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR1, 2), _mm256_add_epi16(TLR2, TRL1));
    T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR0, 1), _mm256_add_epi16(T0_256, T2_256));
    T1_256 = _mm256_srli_epi16(T0_256, 4);
    TLR0w = _mm256_blendv_epi8(TLR0w, T1_256, _mm256_cmpeq_epi16(FS_256, c_3_256));

    T0_256 = _mm256_add_epi16(_mm256_add_epi16(TLR2, TRL0), _mm256_slli_epi16(TLR2, 1));
    T0_256 = _mm256_add_epi16(T0_256, _mm256_slli_epi16(TLR1, 3));
    T0_256 = _mm256_add_epi16(T0_256, _mm256_slli_epi16(TLR0, 2));
    T1_256 = _mm256_srli_epi16(_mm256_add_epi16(T0_256, c_8_256), 4);

    TLR1w = _mm256_blendv_epi8(TLR1w, T1_256, _mm256_cmpeq_epi16(FS_256, c_3_256));

    FS = _mm_cmpeq_epi16(FS, _mm256_castsi256_si128(c_4_256));

    if (_mm_extract_epi64(FS, 0) || _mm_extract_epi64(FS, 1)) { /* fs == 4 */
        TRL2 = _mm256_inserti128_si256(_mm256_castsi128_si256(TR2), TL2, 1);
        FS_256 = _mm256_inserti128_si256(_mm256_castsi128_si256(FS), FS, 1);

        /* cal L0/R0 */
        T0_256 = _mm256_slli_epi16(_mm256_add_epi16(_mm256_add_epi16(TLR0, TLR2), TRL0), 3);
        T0_256 = _mm256_add_epi16(_mm256_add_epi16(T0_256, c_16_256), _mm256_add_epi16(TLR0, TLR2));
        T2_256 = _mm256_add_epi16(_mm256_slli_epi16(TRL2, 1), _mm256_slli_epi16(TRL2, 2));
        T1_256 = _mm256_srli_epi16(_mm256_add_epi16(T0_256, T2_256), 5);

        TLR0w = _mm256_blendv_epi8(TLR0w, T1_256, FS_256);

        /* cal L1/R1 */
        T0_256 = _mm256_slli_epi16(_mm256_add_epi16(TLR2, TRL0), 1);
        T0_256 = _mm256_add_epi16(T0_256, _mm256_sub_epi16(_mm256_slli_epi16(TLR0, 3), TLR0));
        T2_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR2, 2), _mm256_add_epi16(TRL0, c_8_256));
        T1_256 = _mm256_srli_epi16(_mm256_add_epi16(T0_256, T2_256), 4);

        TLR1w = _mm256_blendv_epi8(TLR1w, T1_256, FS_256);

        /* cal L2/R2 */
        T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR2, 1), TLR2);
        T2_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR0, 2), TRL0);
        T1_256 = _mm256_srli_epi16(_mm256_add_epi16(T0_256, _mm256_add_epi16(T2_256, c_4_256)), 3);

        TLR2 = _mm256_blendv_epi8(TLR2, T1_256, FS_256);

        TLR0w = _mm256_packus_epi16(TLR0w, c_0_256);
        TLR1w = _mm256_packus_epi16(TLR1w, c_0_256);
        TLR2 = _mm256_packus_epi16(TLR2, c_0_256);
        /* stroe result */
        _mm_storel_epi64((__m128i*)(SrcPtr - inc), _mm256_castsi256_si128(TLR0w));
        _mm_storel_epi64((__m128i*)(SrcPtr - 0), _mm256_extracti128_si256(TLR0w, 0x01));

        _mm_storel_epi64((__m128i*)(SrcPtr - inc2), _mm256_castsi256_si128(TLR1w));
        _mm_storel_epi64((__m128i*)(SrcPtr + inc), _mm256_extracti128_si256(TLR1w, 0x01));

        _mm_storel_epi64((__m128i*)(SrcPtr - inc3), _mm256_castsi256_si128(TLR2));
        _mm_storel_epi64((__m128i*)(SrcPtr + inc2), _mm256_extracti128_si256(TLR2, 0x01));
    } else {
        /* stroe result */
        TLR0w = _mm256_packus_epi16(TLR0w, c_0_256);
        TLR1w = _mm256_packus_epi16(TLR1w, c_0_256);
        _mm_storel_epi64((__m128i*)(SrcPtr - inc), _mm256_castsi256_si128(TLR0w));
        _mm_storel_epi64((__m128i*)(SrcPtr - 0), _mm256_extracti128_si256(TLR0w, 0x01));

        _mm_storel_epi64((__m128i*)(SrcPtr - inc2), _mm256_castsi256_si128(TLR1w));
        _mm_storel_epi64((__m128i*)(SrcPtr + inc), _mm256_extracti128_si256(TLR1w, 0x01));
    }

}

//需要修改变量  修改变量   i32s_t为int32_t;（signed int）
void deblock_edge_hor_c_avx2(pel_t *SrcPtrU, pel_t *SrcPtrV, int stride, int Alpha, int Beta, uint8_t *flt_flag)
{
    int inc = stride;
    int inc2 = inc << 1;
    int inc3 = inc + inc2;
    int flag0 = flt_flag[0] ? -1 : 0;
    int flag1 = flt_flag[1] ? -1 : 0;

    __m128i T0, T1, T2;
    __m128i M0, M1;
    __m128i FLT, FS;
    __m128i FS4, FS56;

    __m256i TLR0, TLR1, TLR2; // store TL* and TR*
    __m256i TRL0, TRL1; // store TR* and TL*
    __m256i T0_256, T1_256, T2_256;
    __m256i FLT_X;
    __m256i TLR0w, TLR1w;
    __m256i FS_256;

    __m128i ALPHA = _mm_set1_epi16((short)Alpha);
    __m128i c_0 = _mm_set1_epi16(0);
    __m256i c_0_256 = _mm256_setzero_si256();
    __m256i c_1_256 = _mm256_set1_epi16(1);
    __m256i c_2_256 = _mm256_set1_epi16(2);
    __m256i c_3_256 = _mm256_set1_epi16(3);
    __m256i c_4_256 = _mm256_set1_epi16(4);
    __m256i c_8_256 = _mm256_set1_epi16(8);
    __m256i BETA_256 = _mm256_set1_epi16((short)Beta);

    TLR0 = _mm256_set_epi32(0, 0, ((int32_t*)(SrcPtrV))[0], ((int32_t*)(SrcPtrU))[0], 0, 0, ((int32_t*)(SrcPtrV - inc))[0], ((int32_t*)(SrcPtrU - inc))[0]);
    TLR1 = _mm256_set_epi32(0, 0, ((int32_t*)(SrcPtrV + inc))[0], ((int32_t*)(SrcPtrU + inc))[0], 0, 0, ((int32_t*)(SrcPtrV - inc2))[0], ((int32_t*)(SrcPtrU - inc2))[0]);
    TLR2 = _mm256_set_epi32(0, 0, ((int32_t*)(SrcPtrV + inc2))[0], ((int32_t*)(SrcPtrU + inc2))[0], 0, 0, ((int32_t*)(SrcPtrV - inc3))[0], ((int32_t*)(SrcPtrU - inc3))[0]);

    TLR0 = _mm256_unpacklo_epi8(TLR0, c_0_256);
    TLR1 = _mm256_unpacklo_epi8(TLR1, c_0_256);
    TLR2 = _mm256_unpacklo_epi8(TLR2, c_0_256);

    TRL0 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm256_extracti128_si256(TLR0, 0x01)), _mm256_castsi256_si128(TLR0), 1);
    TRL1 = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm256_extracti128_si256(TLR1, 0x01)), _mm256_castsi256_si128(TLR1), 1);

    T0 = _mm_abs_epi16(_mm_subs_epi16(_mm256_castsi256_si128(TLR0), _mm256_castsi256_si128(TRL0)));
    T1 = _mm_cmpgt_epi16(T0, _mm256_castsi256_si128(c_1_256));
    T2 = _mm_cmpgt_epi16(ALPHA, T0);

    M0 = _mm_set_epi32(flag1, flag0, flag1, flag0);
    M0 = _mm_and_si128(M0, _mm_and_si128(T1, T2)); // mask1

    T0_256 = _mm256_abs_epi16(_mm256_subs_epi16(TLR1, TLR0));

    FLT_X = _mm256_and_si256(_mm256_cmpgt_epi16(BETA_256, T0_256), c_2_256);

    T0_256 = _mm256_abs_epi16(_mm256_subs_epi16(TLR2, TLR0));
    T1_256 = _mm256_and_si256(_mm256_cmpgt_epi16(BETA_256, T0_256), c_1_256);

    FLT_X = _mm256_add_epi16(T1_256, FLT_X);
    FLT = _mm_add_epi16(_mm256_castsi256_si128(FLT_X), _mm256_extracti128_si256(FLT_X, 0x01));

    T0_256 = _mm256_cmpeq_epi16(TLR1, TLR0);
    M1 = _mm_and_si128(_mm256_castsi256_si128(T0_256), _mm256_extracti128_si256(T0_256, 0x01));
    T0 = _mm_subs_epi16(FLT, _mm256_castsi256_si128(c_3_256));
    T1 = _mm_subs_epi16(FLT, _mm256_castsi256_si128(c_4_256));

    FS56 = _mm_blendv_epi8(T1, T0, M1);
    FS4 = _mm_blendv_epi8(c_0, _mm256_castsi256_si128(c_1_256), _mm_cmpeq_epi16(_mm256_castsi256_si128(FLT_X), _mm256_castsi256_si128(c_2_256)));

    FS = _mm_blendv_epi8(c_0, FS56, _mm_cmpgt_epi16(FLT, _mm256_castsi256_si128(c_4_256)));
    FS = _mm_blendv_epi8(FS, FS4, _mm_cmpeq_epi16(FLT, _mm256_castsi256_si128(c_4_256)));

    FS = _mm_and_si128(FS, M0);
    FS_256 = _mm256_inserti128_si256(_mm256_castsi128_si256(FS), FS, 1);

    TLR0w = TLR0;
    TLR1w = TLR1;
    /* fs == 1 */
    T2 = _mm_add_epi16(_mm_add_epi16(_mm256_castsi256_si128(TLR0), _mm256_castsi256_si128(TRL0)), _mm256_castsi256_si128(c_2_256)); // L0 + R0 + 2
    T2_256 = _mm256_castsi128_si256(T2);
    T2_256 = _mm256_inserti128_si256(T2_256, T2, 1); // save
    T1_256 = _mm256_srli_epi16(_mm256_add_epi16(_mm256_slli_epi16(TLR0, 1), T2_256), 2);
    TLR0w = _mm256_blendv_epi8(TLR0, T1_256, _mm256_cmpeq_epi16(FS_256, c_1_256));

    /* fs == 2 */
    T2_256 = _mm256_slli_epi16(T2_256, 1);
    T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR1, 1), _mm256_add_epi16(TLR1, TRL0));
    T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR0, 3), _mm256_add_epi16(T0_256, T2_256));
    T1_256 = _mm256_srli_epi16(_mm256_add_epi16(T0_256, c_4_256), 4);
    TLR0w = _mm256_blendv_epi8(TLR0w, T1_256, _mm256_cmpeq_epi16(FS_256, c_2_256));

    /* fs == 3 */
    T2_256 = _mm256_slli_epi16(T2_256, 1); // (L0 << 2) + (R0 << 2) + 8
    T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR1, 2), _mm256_add_epi16(TLR2, TRL1));
    T0_256 = _mm256_add_epi16(_mm256_slli_epi16(TLR0, 1), _mm256_add_epi16(T0_256, T2_256));
    T1_256 = _mm256_srli_epi16(T0_256, 4);
    TLR0w = _mm256_blendv_epi8(TLR0w, T1_256, _mm256_cmpeq_epi16(FS_256, c_3_256));

    T0_256 = _mm256_add_epi16(_mm256_add_epi16(TLR2, TRL0), _mm256_slli_epi16(TLR2, 1));
    T0_256 = _mm256_add_epi16(T0_256, _mm256_slli_epi16(TLR1, 3));
    T0_256 = _mm256_add_epi16(T0_256, _mm256_slli_epi16(TLR0, 2));
    T1_256 = _mm256_srli_epi16(_mm256_add_epi16(T0_256, c_8_256), 4);

    TLR1w = _mm256_blendv_epi8(TLR1w, T1_256, _mm256_cmpeq_epi16(FS_256, c_3_256));

    /* stroe result */
    TLR0w = _mm256_packus_epi16(TLR0w, c_0_256);
    TLR1w = _mm256_packus_epi16(TLR1w, c_0_256);

    ((int32_t*)(SrcPtrU - inc))[0] = _mm256_extract_epi32(TLR0w, 0);
    ((int32_t*)(SrcPtrU))[0] = _mm256_extract_epi32(TLR0w, 4);
    ((int32_t*)(SrcPtrU - inc2))[0] = _mm256_extract_epi32(TLR1w, 0);
    ((int32_t*)(SrcPtrU + inc))[0] = _mm256_extract_epi32(TLR1w, 4);
    ((int32_t*)(SrcPtrV - inc))[0] = _mm256_extract_epi32(TLR0w, 1);
    ((int32_t*)(SrcPtrV))[0] = _mm256_extract_epi32(TLR0w, 5);
    ((int32_t*)(SrcPtrV - inc2))[0] = _mm256_extract_epi32(TLR1w, 1);
    ((int32_t*)(SrcPtrV + inc))[0] = _mm256_extract_epi32(TLR1w, 5);

}

