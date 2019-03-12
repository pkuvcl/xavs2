/*
 * intrinsic_deblock.c
 *
 * Description of this file:
 *    SSE assembly functions of Deblock module of the xavs2 library
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

void deblock_edge_ver_sse128(pel_t *SrcPtr, int stride, int Alpha, int Beta, uint8_t *flt_flag)
{
    pel_t *pTmp = SrcPtr - 4;
    int flag0 = flt_flag[0] ? -1 : 0;
    int flag1 = flt_flag[1] ? -1 : 0;
    __m128i TL0, TL1, TL2, TL3;
    __m128i TR0, TR1, TR2, TR3;
    __m128i TL0l, TL1l;
    __m128i TR0l, TR1l;
    __m128i V0, V1, V2, V3, V4, V5;
    __m128i T0, T1, T2, T3, T4, T5, T6, T7;
    __m128i M0, M1, M2;
    __m128i FLT_L, FLT_R, FLT, FS;
    __m128i FS3, FS4, FS56;

    __m128i ALPHA = _mm_set1_epi16((pel_t)Alpha);
    __m128i BETA = _mm_set1_epi16((pel_t)Beta);
    __m128i c_0 = _mm_set1_epi16(0);
    __m128i c_1 = _mm_set1_epi16(1);
    __m128i c_2 = _mm_set1_epi16(2);
    __m128i c_3 = _mm_set1_epi16(3);
    __m128i c_4 = _mm_set1_epi16(4);
    __m128i c_8 = _mm_set1_epi16(8);
    __m128i c_16 = _mm_set1_epi16(16);

    T0 = _mm_loadl_epi64((__m128i*)(pTmp));
    T1 = _mm_loadl_epi64((__m128i*)(pTmp + stride));
    T2 = _mm_loadl_epi64((__m128i*)(pTmp + stride * 2));
    T3 = _mm_loadl_epi64((__m128i*)(pTmp + stride * 3));
    T4 = _mm_loadl_epi64((__m128i*)(pTmp + stride * 4));
    T5 = _mm_loadl_epi64((__m128i*)(pTmp + stride * 5));
    T6 = _mm_loadl_epi64((__m128i*)(pTmp + stride * 6));
    T7 = _mm_loadl_epi64((__m128i*)(pTmp + stride * 7));

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

#define _mm_subabs_epu16(a, b) _mm_abs_epi16(_mm_subs_epi16(a, b))

    T0 = _mm_subabs_epu16(TL0, TR0);
    T1 = _mm_cmpgt_epi16(T0, c_1);
    T2 = _mm_cmpgt_epi16(ALPHA, T0);

    M0 = _mm_set_epi32(flag1, flag1, flag0, flag0);
    M0 = _mm_and_si128(M0, _mm_and_si128(T1, T2)); // mask1

    T0 = _mm_subabs_epu16(TL1, TL0);
    T1 = _mm_subabs_epu16(TR1, TR0);
    FLT_L = _mm_and_si128(_mm_cmpgt_epi16(BETA, T0), c_2);
    FLT_R = _mm_and_si128(_mm_cmpgt_epi16(BETA, T1), c_2);

    T0 = _mm_subabs_epu16(TL2, TL0);
    T1 = _mm_subabs_epu16(TR2, TR0);
    M1 = _mm_cmpgt_epi16(BETA, T0);
    M2 = _mm_cmpgt_epi16(BETA, T1);
    FLT_L = _mm_add_epi16(_mm_and_si128(M1, c_1), FLT_L);
    FLT_R = _mm_add_epi16(_mm_and_si128(M2, c_1), FLT_R);
    FLT = _mm_add_epi16(FLT_L, FLT_R);

    M1 = _mm_and_si128(_mm_cmpeq_epi16(TR0, TR1), _mm_cmpeq_epi16(TL0, TL1));
    T0 = _mm_sub_epi16(FLT, c_2);
    T1 = _mm_sub_epi16(FLT, c_3);
    T2 = _mm_subabs_epu16(TL1, TR1);

    FS56 = _mm_blendv_epi8(T1, T0, M1);
    FS4 = _mm_blendv_epi8(c_1, c_2, _mm_cmpeq_epi16(FLT_L, c_2));
    FS3 = _mm_blendv_epi8(c_0, c_1, _mm_cmpgt_epi16(BETA, T2));

    FS = _mm_blendv_epi8(c_0, FS56, _mm_cmpgt_epi16(FLT, c_4));
    FS = _mm_blendv_epi8(FS, FS4, _mm_cmpeq_epi16(FLT, c_4));
    FS = _mm_blendv_epi8(FS, FS3, _mm_cmpeq_epi16(FLT, c_3));

    FS = _mm_and_si128(FS, M0);

#undef _mm_subabs_epu16

    TL0l = TL0;
    TL1l = TL1;
    TR0l = TR0;
    TR1l = TR1;

    /* fs == 1 */
    T2 = _mm_add_epi16(_mm_add_epi16(TL0l, TR0l), c_2); // L0 + R0 + 2

    V0 = _mm_srli_epi16(_mm_add_epi16(_mm_slli_epi16(TL0l, 1), T2), 2);

    V1 = _mm_srli_epi16(_mm_add_epi16(_mm_slli_epi16(TR0l, 1), T2), 2);

    TL0 = _mm_blendv_epi8(TL0, V0, _mm_cmpeq_epi16(FS, c_1));
    TR0 = _mm_blendv_epi8(TR0, V1, _mm_cmpeq_epi16(FS, c_1));

    /* fs == 2 */
    T2 = _mm_slli_epi16(T2, 1); // (L0 << 1) + (R0 << 1) + 4
    T3 = _mm_slli_epi16(T3, 1);

    T0 = _mm_add_epi16(_mm_slli_epi16(TL1l, 1), _mm_add_epi16(TL1l, TR0l));

    T0 = _mm_add_epi16(_mm_slli_epi16(TL0l, 3), _mm_add_epi16(T0, T2));

    V0 = _mm_srli_epi16(_mm_add_epi16(T0, c_4), 4);

    T0 = _mm_add_epi16(_mm_slli_epi16(TR1l, 1), _mm_add_epi16(TR1l, TL0l));

    T0 = _mm_add_epi16(_mm_slli_epi16(TR0l, 3), _mm_add_epi16(T0, T2));

    V1 = _mm_srli_epi16(_mm_add_epi16(T0, c_4), 4);

    TL0 = _mm_blendv_epi8(TL0, V0, _mm_cmpeq_epi16(FS, c_2));
    TR0 = _mm_blendv_epi8(TR0, V1, _mm_cmpeq_epi16(FS, c_2));

    /* fs == 3 */
    T2 = _mm_slli_epi16(T2, 1); // (L0 << 2) + (R0 << 2) + 8
    T3 = _mm_slli_epi16(T3, 1);

    T0 = _mm_add_epi16(_mm_slli_epi16(TL1l, 2), _mm_add_epi16(TL2, TR1l));

    T0 = _mm_add_epi16(_mm_slli_epi16(TL0l, 1), _mm_add_epi16(T0, T2));

    V0 = _mm_srli_epi16(T0, 4);

    T0 = _mm_add_epi16(_mm_slli_epi16(TR1l, 2), _mm_add_epi16(TR2, TL1l));

    T0 = _mm_add_epi16(_mm_slli_epi16(TR0l, 1), _mm_add_epi16(T0, T2));

    V1 = _mm_srli_epi16(T0, 4);

    TL0 = _mm_blendv_epi8(TL0, V0, _mm_cmpeq_epi16(FS, c_3));
    TR0 = _mm_blendv_epi8(TR0, V1, _mm_cmpeq_epi16(FS, c_3));

    T0 = _mm_add_epi16(_mm_add_epi16(TL2, TR0l), _mm_slli_epi16(TL2, 1));
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(TL1l, 3));
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(TL0l, 2));
    V2 = _mm_srli_epi16(_mm_add_epi16(T0, c_8), 4);

    T0 = _mm_add_epi16(_mm_add_epi16(TR2, TL0l), _mm_slli_epi16(TR2, 1));
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(TR1l, 3));
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(TR0l, 2));
    V3 = _mm_srli_epi16(_mm_add_epi16(T0, c_8), 4);

    TL1 = _mm_blendv_epi8(TL1, V2, _mm_cmpeq_epi16(FS, c_3));
    TR1 = _mm_blendv_epi8(TR1, V3, _mm_cmpeq_epi16(FS, c_3));

    FS = _mm_cmpeq_epi16(FS, c_4);

    if (!_mm_testz_si128(FS, _mm_set1_epi16(-1))) { /* fs == 4 */
        /* cal L0/R0 */
        T0 = _mm_slli_epi16(_mm_add_epi16(_mm_add_epi16(TL0l, TL2), TR0l), 3);
        T0 = _mm_add_epi16(_mm_add_epi16(T0, c_16), _mm_add_epi16(TL0l, TL2));
        T2 = _mm_add_epi16(_mm_slli_epi16(TR2, 1), _mm_slli_epi16(TR2, 2));
        V0 = _mm_srli_epi16(_mm_add_epi16(T0, T2), 5);

        T0 = _mm_slli_epi16(_mm_add_epi16(_mm_add_epi16(TR0l, TR2), TL0l), 3);
        T0 = _mm_add_epi16(_mm_add_epi16(T0, c_16), _mm_add_epi16(TR0l, TR2));
        T2 = _mm_add_epi16(_mm_slli_epi16(TL2, 1), _mm_slli_epi16(TL2, 2));
        V1 = _mm_srli_epi16(_mm_add_epi16(T0, T2), 5);

        TL0 = _mm_blendv_epi8(TL0, V0, FS);
        TR0 = _mm_blendv_epi8(TR0, V1, FS);

        /* cal L1/R1 */
        T0 = _mm_slli_epi16(_mm_add_epi16(TL2, TR0l), 1);
        T0 = _mm_add_epi16(T0, _mm_sub_epi16(_mm_slli_epi16(TL0l, 3), TL0l));
        T2 = _mm_add_epi16(_mm_slli_epi16(TL2, 2), _mm_add_epi16(TR0l, c_8));
        V2 = _mm_srli_epi16(_mm_add_epi16(T0, T2), 4);

        T0 = _mm_slli_epi16(_mm_add_epi16(TR2, TL0l), 1);
        T0 = _mm_add_epi16(T0, _mm_sub_epi16(_mm_slli_epi16(TR0l, 3), TR0l));
        T2 = _mm_add_epi16(_mm_slli_epi16(TR2, 2), _mm_add_epi16(TL0l, c_8));
        V3 = _mm_srli_epi16(_mm_add_epi16(T0, T2), 4);

        TL1 = _mm_blendv_epi8(TL1, V2, FS);
        TR1 = _mm_blendv_epi8(TR1, V3, FS);

        /* cal L2/R2 */
        T0 = _mm_add_epi16(_mm_slli_epi16(TL2, 1), TL2);
        T2 = _mm_add_epi16(_mm_slli_epi16(TL0l, 2), TR0l);
        V4 = _mm_srli_epi16(_mm_add_epi16(T0, _mm_add_epi16(T2, c_4)), 3);

        T0 = _mm_add_epi16(_mm_slli_epi16(TR2, 1), TR2);
        T2 = _mm_add_epi16(_mm_slli_epi16(TR0l, 2), TL0l);
        V5 = _mm_srli_epi16(_mm_add_epi16(T0, _mm_add_epi16(T2, c_4)), 3);

        TL2 = _mm_blendv_epi8(TL2, V4, FS);
        TR2 = _mm_blendv_epi8(TR2, V5, FS);
    }

    /* store result */
    T0 = _mm_packus_epi16(TL3, TR0);
    T1 = _mm_packus_epi16(TL2, TR1);
    T2 = _mm_packus_epi16(TL1, TR2);
    T3 = _mm_packus_epi16(TL0, TR3);

    T4 = _mm_unpacklo_epi8(T0, T1);
    T5 = _mm_unpacklo_epi8(T2, T3);
    T6 = _mm_unpackhi_epi8(T0, T1);
    T7 = _mm_unpackhi_epi8(T2, T3);

    V0 = _mm_unpacklo_epi16(T4, T5);
    V1 = _mm_unpacklo_epi16(T6, T7);
    V2 = _mm_unpackhi_epi16(T4, T5);
    V3 = _mm_unpackhi_epi16(T6, T7);

    T0 = _mm_unpacklo_epi32(V0, V1);
    T1 = _mm_unpackhi_epi32(V0, V1);
    T2 = _mm_unpacklo_epi32(V2, V3);
    T3 = _mm_unpackhi_epi32(V2, V3);

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

void deblock_edge_ver_c_sse128(pel_t *SrcPtrU, pel_t *SrcPtrV, int stride, int Alpha, int Beta, uint8_t *flt_flag)
{
    pel_t *pTmp;
    int flag0 = flt_flag[0] ? -1 : 0;
    int flag1 = flt_flag[1] ? -1 : 0;

    __m128i UVL0, UVL1, UVR0, UVR1;
    __m128i TL0, TL1, TL2, TL3;
    __m128i TR0, TR1, TR2, TR3;
    __m128i T0, T1, T2, T3, T4, T5, T6, T7;
    __m128i P0, P1, P2, P3, P4, P5, P6, P7;
    __m128i V0, V1, V2, V3;
    __m128i M0, M1, M2;
    __m128i FLT_L, FLT_R, FLT, FS;
    __m128i FS4, FS56;

    __m128i ALPHA = _mm_set1_epi16((pel_t)Alpha);
    __m128i BETA = _mm_set1_epi16((pel_t)Beta);
    __m128i c_0 = _mm_set1_epi16(0);
    __m128i c_1 = _mm_set1_epi16(1);
    __m128i c_2 = _mm_set1_epi16(2);
    __m128i c_3 = _mm_set1_epi16(3);
    __m128i c_4 = _mm_set1_epi16(4);
    __m128i c_8 = _mm_set1_epi16(8);

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

    P0 = _mm_unpacklo_epi8(T0, T1);
    P1 = _mm_unpacklo_epi8(T2, T3);
    P2 = _mm_unpacklo_epi8(T4, T5);
    P3 = _mm_unpacklo_epi8(T6, T7);

    P4 = _mm_unpacklo_epi16(P0, P1);
    P5 = _mm_unpacklo_epi16(P2, P3);
    P6 = _mm_unpackhi_epi16(P0, P1);
    P7 = _mm_unpackhi_epi16(P2, P3);

    T0 = _mm_unpacklo_epi32(P4, P5);
    T1 = _mm_unpackhi_epi32(P4, P5);
    T2 = _mm_unpacklo_epi32(P6, P7);
    T3 = _mm_unpackhi_epi32(P6, P7);

    TL3 = _mm_unpacklo_epi8(T0, c_0);
    TL2 = _mm_unpackhi_epi8(T0, c_0);
    TL1 = _mm_unpacklo_epi8(T1, c_0);
    TL0 = _mm_unpackhi_epi8(T1, c_0);

    TR0 = _mm_unpacklo_epi8(T2, c_0);
    TR1 = _mm_unpackhi_epi8(T2, c_0);
    TR2 = _mm_unpacklo_epi8(T3, c_0);
    TR3 = _mm_unpackhi_epi8(T3, c_0);

#define _mm_subabs_epu16(a, b) _mm_abs_epi16(_mm_subs_epi16(a, b))

    T0 = _mm_subabs_epu16(TL0, TR0);
    T1 = _mm_cmpgt_epi16(T0, c_1);
    T2 = _mm_cmpgt_epi16(ALPHA, T0);
    M0 = _mm_set_epi32(flag1, flag0, flag1, flag0);
    M0 = _mm_and_si128(M0, _mm_and_si128(T1, T2)); // mask1

    T0 = _mm_subabs_epu16(TL1, TL0);
    T1 = _mm_subabs_epu16(TR1, TR0);
    FLT_L = _mm_and_si128(_mm_cmpgt_epi16(BETA, T0), c_2);
    FLT_R = _mm_and_si128(_mm_cmpgt_epi16(BETA, T1), c_2);

    T0 = _mm_subabs_epu16(TL2, TL0);
    T1 = _mm_subabs_epu16(TR2, TR0);
    M1 = _mm_cmpgt_epi16(BETA, T0);
    M2 = _mm_cmpgt_epi16(BETA, T1);
    FLT_L = _mm_add_epi16(_mm_and_si128(M1, c_1), FLT_L);
    FLT_R = _mm_add_epi16(_mm_and_si128(M2, c_1), FLT_R);
    FLT = _mm_add_epi16(FLT_L, FLT_R);

    M1 = _mm_and_si128(_mm_cmpeq_epi16(TR0, TR1), _mm_cmpeq_epi16(TL0, TL1));
    T0 = _mm_sub_epi16(FLT, c_3);
    T1 = _mm_sub_epi16(FLT, c_4);
    T2 = _mm_subabs_epu16(TL1, TR1);

    FS56 = _mm_blendv_epi8(T1, T0, M1);
    FS4 = _mm_blendv_epi8(c_0, c_1, _mm_cmpeq_epi16(FLT_L, c_2));

    FS = _mm_blendv_epi8(c_0, FS56, _mm_cmpgt_epi16(FLT, c_4));
    FS = _mm_blendv_epi8(FS, FS4, _mm_cmpeq_epi16(FLT, c_4));

    FS = _mm_and_si128(FS, M0);

#undef _mm_subabs_epu16

    UVL0 = TL0;
    UVL1 = TL1;
    UVR0 = TR0;
    UVR1 = TR1;

    /* fs == 1 */
    T2 = _mm_add_epi16(_mm_add_epi16(UVL0, UVR0), c_2); // L0 + R0 + 2

    V0 = _mm_srli_epi16(_mm_add_epi16(_mm_slli_epi16(UVL0, 1), T2), 2);

    V1 = _mm_srli_epi16(_mm_add_epi16(_mm_slli_epi16(UVR0, 1), T2), 2);

    TL0 = _mm_blendv_epi8(TL0, V0, _mm_cmpeq_epi16(FS, c_1));
    TR0 = _mm_blendv_epi8(TR0, V1, _mm_cmpeq_epi16(FS, c_1));

    /* fs == 2 */
    T2 = _mm_slli_epi16(T2, 1); // (L0 << 1) + (R0 << 1) + 4
    T3 = _mm_slli_epi16(T3, 1);

    T0 = _mm_add_epi16(_mm_slli_epi16(UVL1, 1), _mm_add_epi16(UVL1, UVR0));
    T0 = _mm_add_epi16(_mm_slli_epi16(UVL0, 3), _mm_add_epi16(T0, T2));
    V0 = _mm_srli_epi16(_mm_add_epi16(T0, c_4), 4);
    T0 = _mm_add_epi16(_mm_slli_epi16(UVR1, 1), _mm_add_epi16(UVR1, UVL0));
    T0 = _mm_add_epi16(_mm_slli_epi16(UVR0, 3), _mm_add_epi16(T0, T2));
    V1 = _mm_srli_epi16(_mm_add_epi16(T0, c_4), 4);

    TL0 = _mm_blendv_epi8(TL0, V0, _mm_cmpeq_epi16(FS, c_2));
    TR0 = _mm_blendv_epi8(TR0, V1, _mm_cmpeq_epi16(FS, c_2));

    /* fs == 3 */
    T2 = _mm_slli_epi16(T2, 1); // (L0 << 2) + (R0 << 2) + 8
    T3 = _mm_slli_epi16(T3, 1);

    T0 = _mm_add_epi16(_mm_slli_epi16(UVL1, 2), _mm_add_epi16(TL2, UVR1));
    T0 = _mm_add_epi16(_mm_slli_epi16(UVL0, 1), _mm_add_epi16(T0, T2));
    V0 = _mm_srli_epi16(T0, 4);
    T0 = _mm_add_epi16(_mm_slli_epi16(UVR1, 2), _mm_add_epi16(TR2, UVL1));
    T0 = _mm_add_epi16(_mm_slli_epi16(UVR0, 1), _mm_add_epi16(T0, T2));
    V1 = _mm_srli_epi16(T0, 4);

    TL0 = _mm_blendv_epi8(TL0, V0, _mm_cmpeq_epi16(FS, c_3));
    TR0 = _mm_blendv_epi8(TR0, V1, _mm_cmpeq_epi16(FS, c_3));

    T0 = _mm_add_epi16(_mm_add_epi16(TL2, UVR0), _mm_slli_epi16(TL2, 1));
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(UVL1, 3));
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(UVL0, 2));
    V2 = _mm_srli_epi16(_mm_add_epi16(T0, c_8), 4);

    T0 = _mm_add_epi16(_mm_add_epi16(TR2, UVL0), _mm_slli_epi16(TR2, 1));
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(UVR1, 3));
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(UVR0, 2));
    V3 = _mm_srli_epi16(_mm_add_epi16(T0, c_8), 4);

    TL1 = _mm_blendv_epi8(TL1, V2, _mm_cmpeq_epi16(FS, c_3));
    TR1 = _mm_blendv_epi8(TR1, V3, _mm_cmpeq_epi16(FS, c_3));

    /* store result */
    T0 = _mm_packus_epi16(TL3, TR0);
    T1 = _mm_packus_epi16(TL2, TR1);
    T2 = _mm_packus_epi16(TL1, TR2);
    T3 = _mm_packus_epi16(TL0, TR3);

    P0 = _mm_unpacklo_epi8(T0, T1);
    P1 = _mm_unpacklo_epi8(T2, T3);
    P2 = _mm_unpackhi_epi8(T0, T1);
    P3 = _mm_unpackhi_epi8(T2, T3);

    P4 = _mm_unpacklo_epi16(P0, P1);
    P5 = _mm_unpacklo_epi16(P2, P3);
    P6 = _mm_unpackhi_epi16(P0, P1);
    P7 = _mm_unpackhi_epi16(P2, P3);

    T0 = _mm_unpacklo_epi32(P4, P5);
    T1 = _mm_unpackhi_epi32(P4, P5);
    T2 = _mm_unpacklo_epi32(P6, P7);
    T3 = _mm_unpackhi_epi32(P6, P7);

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

void deblock_edge_hor_sse128(pel_t *SrcPtr, int stride, int Alpha, int Beta, uint8_t *flt_flag)
{
    int inc = stride;
    int inc2 = inc << 1;
    int inc3 = inc + inc2;
    int flag0 = flt_flag[0] ? -1 : 0;
    int flag1 = flt_flag[1] ? -1 : 0;
    __m128i TL0, TL1, TL2;
    __m128i TR0, TR1, TR2;
    __m128i TL0w, TL1w, TL2w, TR0w, TR1w, TR2w; //for write
    __m128i V0, V1, V2, V3, V4, V5;
    __m128i T0, T1, T2;
    __m128i M0, M1, M2;
    __m128i FLT_L, FLT_R, FLT, FS;
    __m128i FS3, FS4, FS56;

    __m128i ALPHA = _mm_set1_epi16((int16_t)Alpha);
    __m128i BETA = _mm_set1_epi16((int16_t)Beta);
    __m128i c_0 = _mm_set1_epi16(0);
    __m128i c_1 = _mm_set1_epi16(1);
    __m128i c_2 = _mm_set1_epi16(2);
    __m128i c_3 = _mm_set1_epi16(3);
    __m128i c_4 = _mm_set1_epi16(4);
    __m128i c_8 = _mm_set1_epi16(8);
    __m128i c_16 = _mm_set1_epi16(16);

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

#define _mm_subabs_epu16(a, b) _mm_abs_epi16(_mm_subs_epi16(a, b))

    T0 = _mm_subabs_epu16(TL0, TR0);
    T1 = _mm_cmpgt_epi16(T0, c_1);
    T2 = _mm_cmpgt_epi16(ALPHA, T0);
    M0 = _mm_set_epi32(flag1, flag1, flag0, flag0);
    M0 = _mm_and_si128(M0, _mm_and_si128(T1, T2)); // mask1

    T0 = _mm_subabs_epu16(TL1, TL0);
    T1 = _mm_subabs_epu16(TR1, TR0);
    FLT_L = _mm_and_si128(_mm_cmpgt_epi16(BETA, T0), c_2);
    FLT_R = _mm_and_si128(_mm_cmpgt_epi16(BETA, T1), c_2);

    T0 = _mm_subabs_epu16(TL2, TL0);
    T1 = _mm_subabs_epu16(TR2, TR0);
    M1 = _mm_cmpgt_epi16(BETA, T0);
    M2 = _mm_cmpgt_epi16(BETA, T1);
    FLT_L = _mm_add_epi16(_mm_and_si128(M1, c_1), FLT_L);
    FLT_R = _mm_add_epi16(_mm_and_si128(M2, c_1), FLT_R);
    FLT = _mm_add_epi16(FLT_L, FLT_R);

    M1 = _mm_and_si128(_mm_cmpeq_epi16(TR0, TR1), _mm_cmpeq_epi16(TL0, TL1));
    T0 = _mm_subs_epi16(FLT, c_2);
    T1 = _mm_subs_epi16(FLT, c_3);
    T2 = _mm_subabs_epu16(TL1, TR1);

    FS56 = _mm_blendv_epi8(T1, T0, M1);
    FS4 = _mm_blendv_epi8(c_1, c_2, _mm_cmpeq_epi16(FLT_L, c_2));
    FS3 = _mm_blendv_epi8(c_0, c_1, _mm_cmpgt_epi16(BETA, T2));

    FS = _mm_blendv_epi8(c_0, FS56, _mm_cmpgt_epi16(FLT, c_4));
    FS = _mm_blendv_epi8(FS, FS4, _mm_cmpeq_epi16(FLT, c_4));
    FS = _mm_blendv_epi8(FS, FS3, _mm_cmpeq_epi16(FLT, c_3));

    FS = _mm_and_si128(FS, M0);

#undef _mm_subabs_epu16

    TR0w = TR0;
    TR1w = TR1;
    TL0w = TL0;
    TL1w = TL1;

    /* fs == 1 */
    T2 = _mm_add_epi16(_mm_add_epi16(TL0, TR0), c_2); // L0 + R0 + 2
    V0 = _mm_srli_epi16(_mm_add_epi16(_mm_slli_epi16(TL0, 1), T2), 2);
    V1 = _mm_srli_epi16(_mm_add_epi16(_mm_slli_epi16(TR0, 1), T2), 2);

    TL0w = _mm_blendv_epi8(TL0, V0, _mm_cmpeq_epi16(FS, c_1));
    TR0w = _mm_blendv_epi8(TR0, V1, _mm_cmpeq_epi16(FS, c_1));

    /* fs == 2 */
    T2 = _mm_slli_epi16(T2, 1); // (L0 << 1) + (R0 << 1) + 4
    T0 = _mm_add_epi16(_mm_slli_epi16(TL1, 1), _mm_add_epi16(TL1, TR0));
    T0 = _mm_add_epi16(_mm_slli_epi16(TL0, 3), _mm_add_epi16(T0, T2));
    V0 = _mm_srli_epi16(_mm_add_epi16(T0, c_4), 4);
    T0 = _mm_add_epi16(_mm_slli_epi16(TR1, 1), _mm_add_epi16(TR1, TL0));
    T0 = _mm_add_epi16(_mm_slli_epi16(TR0, 3), _mm_add_epi16(T0, T2));
    V1 = _mm_srli_epi16(_mm_add_epi16(T0, c_4), 4);

    TL0w = _mm_blendv_epi8(TL0w, V0, _mm_cmpeq_epi16(FS, c_2));
    TR0w = _mm_blendv_epi8(TR0w, V1, _mm_cmpeq_epi16(FS, c_2));

    /* fs == 3 */
    T2 = _mm_slli_epi16(T2, 1); // (L0 << 2) + (R0 << 2) + 8
    T0 = _mm_add_epi16(_mm_slli_epi16(TL1, 2), _mm_add_epi16(TL2, TR1));
    T0 = _mm_add_epi16(_mm_slli_epi16(TL0, 1), _mm_add_epi16(T0, T2));
    V0 = _mm_srli_epi16(T0, 4);
    T0 = _mm_add_epi16(_mm_slli_epi16(TR1, 2), _mm_add_epi16(TR2, TL1));
    T0 = _mm_add_epi16(_mm_slli_epi16(TR0, 1), _mm_add_epi16(T0, T2));
    V1 = _mm_srli_epi16(T0, 4);

    TL0w = _mm_blendv_epi8(TL0w, V0, _mm_cmpeq_epi16(FS, c_3));
    TR0w = _mm_blendv_epi8(TR0w, V1, _mm_cmpeq_epi16(FS, c_3));

    T0 = _mm_add_epi16(_mm_add_epi16(TL2, TR0), _mm_slli_epi16(TL2, 1));
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(TL1, 3));
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(TL0, 2));
    V2 = _mm_srli_epi16(_mm_add_epi16(T0, c_8), 4);

    T0 = _mm_add_epi16(_mm_add_epi16(TR2, TL0), _mm_slli_epi16(TR2, 1));
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(TR1, 3));
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(TR0, 2));
    V3 = _mm_srli_epi16(_mm_add_epi16(T0, c_8), 4);

    TL1w = _mm_blendv_epi8(TL1w, V2, _mm_cmpeq_epi16(FS, c_3));
    TR1w = _mm_blendv_epi8(TR1w, V3, _mm_cmpeq_epi16(FS, c_3));

    FS = _mm_cmpeq_epi16(FS, c_4);

    if (M128_U64(FS, 0) || M128_U64(FS, 1)) { /* fs == 4 */
        /* cal L0/R0 */
        T0 = _mm_slli_epi16(_mm_add_epi16(_mm_add_epi16(TL0, TL2), TR0), 3);
        T0 = _mm_add_epi16(_mm_add_epi16(T0, c_16), _mm_add_epi16(TL0, TL2));
        T2 = _mm_add_epi16(_mm_slli_epi16(TR2, 1), _mm_slli_epi16(TR2, 2));
        V0 = _mm_srli_epi16(_mm_add_epi16(T0, T2), 5);

        T0 = _mm_slli_epi16(_mm_add_epi16(_mm_add_epi16(TR0, TR2), TL0), 3);
        T0 = _mm_add_epi16(_mm_add_epi16(T0, c_16), _mm_add_epi16(TR0, TR2));
        T2 = _mm_add_epi16(_mm_slli_epi16(TL2, 1), _mm_slli_epi16(TL2, 2));
        V1 = _mm_srli_epi16(_mm_add_epi16(T0, T2), 5);

        TL0w = _mm_blendv_epi8(TL0w, V0, FS);
        TR0w = _mm_blendv_epi8(TR0w, V1, FS);

        /* cal L1/R1 */
        T0 = _mm_slli_epi16(_mm_add_epi16(TL2, TR0), 1);
        T0 = _mm_add_epi16(T0, _mm_sub_epi16(_mm_slli_epi16(TL0, 3), TL0));
        T2 = _mm_add_epi16(_mm_slli_epi16(TL2, 2), _mm_add_epi16(TR0, c_8));
        V2 = _mm_srli_epi16(_mm_add_epi16(T0, T2), 4);

        T0 = _mm_slli_epi16(_mm_add_epi16(TR2, TL0), 1);
        T0 = _mm_add_epi16(T0, _mm_sub_epi16(_mm_slli_epi16(TR0, 3), TR0));
        T2 = _mm_add_epi16(_mm_slli_epi16(TR2, 2), _mm_add_epi16(TL0, c_8));
        V3 = _mm_srli_epi16(_mm_add_epi16(T0, T2), 4);

        TL1w = _mm_blendv_epi8(TL1w, V2, FS);
        TR1w = _mm_blendv_epi8(TR1w, V3, FS);

        /* cal L2/R2 */
        T0 = _mm_add_epi16(_mm_slli_epi16(TL2, 1), TL2);
        T2 = _mm_add_epi16(_mm_slli_epi16(TL0, 2), TR0);
        V4 = _mm_srli_epi16(_mm_add_epi16(T0, _mm_add_epi16(T2, c_4)), 3);

        T0 = _mm_add_epi16(_mm_slli_epi16(TR2, 1), TR2);
        T2 = _mm_add_epi16(_mm_slli_epi16(TR0, 2), TL0);
        V5 = _mm_srli_epi16(_mm_add_epi16(T0, _mm_add_epi16(T2, c_4)), 3);

        TL2w = _mm_blendv_epi8(TL2, V4, FS);
        TR2w = _mm_blendv_epi8(TR2, V5, FS);

        /* store result */
        _mm_storel_epi64((__m128i*)(SrcPtr - inc ), _mm_packus_epi16(TL0w, c_0));
        _mm_storel_epi64((__m128i*)(SrcPtr       ), _mm_packus_epi16(TR0w, c_0));
        _mm_storel_epi64((__m128i*)(SrcPtr - inc2), _mm_packus_epi16(TL1w, c_0));
        _mm_storel_epi64((__m128i*)(SrcPtr + inc ), _mm_packus_epi16(TR1w, c_0));
        _mm_storel_epi64((__m128i*)(SrcPtr - inc3), _mm_packus_epi16(TL2w, c_0));
        _mm_storel_epi64((__m128i*)(SrcPtr + inc2), _mm_packus_epi16(TR2w, c_0));
    } else {
        /* store result */
        _mm_storel_epi64((__m128i*)(SrcPtr - inc ), _mm_packus_epi16(TL0w, c_0));
        _mm_storel_epi64((__m128i*)(SrcPtr       ), _mm_packus_epi16(TR0w, c_0));
        _mm_storel_epi64((__m128i*)(SrcPtr - inc2), _mm_packus_epi16(TL1w, c_0));
        _mm_storel_epi64((__m128i*)(SrcPtr + inc ), _mm_packus_epi16(TR1w, c_0));
    }
}

void deblock_edge_hor_c_sse128(pel_t *SrcPtrU, pel_t *SrcPtrV, int stride, int Alpha, int Beta, uint8_t *flt_flag)
{
    int inc = stride;
    int inc2 = inc << 1;
    int inc3 = inc + inc2;
    int flag0 = flt_flag[0] ? -1 : 0;
    int flag1 = flt_flag[1] ? -1 : 0;

    __m128i UL0, UL1, UR0, UR1;
    __m128i TL0, TL1, TL2;
    __m128i TR0, TR1, TR2;
    __m128i T0, T1, T2;
    __m128i V0, V1, V2, V3;
    __m128i M0, M1, M2;
    __m128i FLT_L, FLT_R, FLT, FS;
    __m128i FS4, FS56;

    __m128i ALPHA = _mm_set1_epi16((pel_t)Alpha);
    __m128i BETA = _mm_set1_epi16((pel_t)Beta);
    __m128i c_0 = _mm_set1_epi16(0);
    __m128i c_1 = _mm_set1_epi16(1);
    __m128i c_2 = _mm_set1_epi16(2);
    __m128i c_3 = _mm_set1_epi16(3);
    __m128i c_4 = _mm_set1_epi16(4);
    __m128i c_8 = _mm_set1_epi16(8);

    TL0 = _mm_set_epi32(0, 0, ((int32_t*)(SrcPtrV - inc))[0], ((int32_t*)(SrcPtrU - inc))[0]);
    TL1 = _mm_set_epi32(0, 0, ((int32_t*)(SrcPtrV - inc2))[0], ((int32_t*)(SrcPtrU - inc2))[0]);
    TL2 = _mm_set_epi32(0, 0, ((int32_t*)(SrcPtrV - inc3))[0], ((int32_t*)(SrcPtrU - inc3))[0]);
    TR0 = _mm_set_epi32(0, 0, ((int32_t*)(SrcPtrV))[0], ((int32_t*)(SrcPtrU))[0]);
    TR1 = _mm_set_epi32(0, 0, ((int32_t*)(SrcPtrV + inc))[0], ((int32_t*)(SrcPtrU + inc))[0]);
    TR2 = _mm_set_epi32(0, 0, ((int32_t*)(SrcPtrV + inc2))[0], ((int32_t*)(SrcPtrU + inc2))[0]);

    TL0 = _mm_unpacklo_epi8(TL0, c_0);
    TL1 = _mm_unpacklo_epi8(TL1, c_0);
    TL2 = _mm_unpacklo_epi8(TL2, c_0);
    TR0 = _mm_unpacklo_epi8(TR0, c_0);
    TR1 = _mm_unpacklo_epi8(TR1, c_0);
    TR2 = _mm_unpacklo_epi8(TR2, c_0);

#define _mm_subabs_epu16(a, b) _mm_abs_epi16(_mm_subs_epi16(a, b))

    T0 = _mm_subabs_epu16(TL0, TR0);
    T1 = _mm_cmpgt_epi16(T0, c_1);
    T2 = _mm_cmpgt_epi16(ALPHA, T0);

    M0 = _mm_set_epi32(flag1, flag0, flag1, flag0);
    M0 = _mm_and_si128(M0, _mm_and_si128(T1, T2)); // mask1

    T0 = _mm_subabs_epu16(TL1, TL0);
    T1 = _mm_subabs_epu16(TR1, TR0);
    FLT_L = _mm_and_si128(_mm_cmpgt_epi16(BETA, T0), c_2);
    FLT_R = _mm_and_si128(_mm_cmpgt_epi16(BETA, T1), c_2);

    T0 = _mm_subabs_epu16(TL2, TL0);
    T1 = _mm_subabs_epu16(TR2, TR0);
    M1 = _mm_cmpgt_epi16(BETA, T0);
    M2 = _mm_cmpgt_epi16(BETA, T1);
    FLT_L = _mm_add_epi16(_mm_and_si128(M1, c_1), FLT_L);
    FLT_R = _mm_add_epi16(_mm_and_si128(M2, c_1), FLT_R);
    FLT = _mm_add_epi16(FLT_L, FLT_R);

    M1 = _mm_and_si128(_mm_cmpeq_epi16(TR0, TR1), _mm_cmpeq_epi16(TL0, TL1));
    T0 = _mm_subs_epi16(FLT, c_3);
    T1 = _mm_subs_epi16(FLT, c_4);

    FS56 = _mm_blendv_epi8(T1, T0, M1);
    FS4 = _mm_blendv_epi8(c_0, c_1, _mm_cmpeq_epi16(FLT_L, c_2));

    FS = _mm_blendv_epi8(c_0, FS56, _mm_cmpgt_epi16(FLT, c_4));
    FS = _mm_blendv_epi8(FS, FS4, _mm_cmpeq_epi16(FLT, c_4));

    FS = _mm_and_si128(FS, M0);

#undef _mm_subabs_epu16

    UR0 = TR0;  //UR0 TR0 to store
    UR1 = TR1;
    UL0 = TL0;
    UL1 = TL1;

    /* fs == 1 */
    T2 = _mm_add_epi16(_mm_add_epi16(TL0, TR0), c_2); // L0 + R0 + 2
    V0 = _mm_srli_epi16(_mm_add_epi16(_mm_slli_epi16(TL0, 1), T2), 2);
    V1 = _mm_srli_epi16(_mm_add_epi16(_mm_slli_epi16(TR0, 1), T2), 2);

    UL0 = _mm_blendv_epi8(TL0, V0, _mm_cmpeq_epi16(FS, c_1));
    UR0 = _mm_blendv_epi8(TR0, V1, _mm_cmpeq_epi16(FS, c_1));

    /* fs == 2 */
    T2 = _mm_slli_epi16(T2, 1); // (L0 << 1) + (R0 << 1) + 4
    T0 = _mm_add_epi16(_mm_slli_epi16(TL1, 1), _mm_add_epi16(TL1, TR0));
    T0 = _mm_add_epi16(_mm_slli_epi16(TL0, 3), _mm_add_epi16(T0, T2));
    V0 = _mm_srli_epi16(_mm_add_epi16(T0, c_4), 4);
    T0 = _mm_add_epi16(_mm_slli_epi16(TR1, 1), _mm_add_epi16(TR1, TL0));
    T0 = _mm_add_epi16(_mm_slli_epi16(TR0, 3), _mm_add_epi16(T0, T2));
    V1 = _mm_srli_epi16(_mm_add_epi16(T0, c_4), 4);

    UL0 = _mm_blendv_epi8(UL0, V0, _mm_cmpeq_epi16(FS, c_2));
    UR0 = _mm_blendv_epi8(UR0, V1, _mm_cmpeq_epi16(FS, c_2));

    /* fs == 3 */
    T2 = _mm_slli_epi16(T2, 1); // (L0 << 2) + (R0 << 2) + 8
    T0 = _mm_add_epi16(_mm_slli_epi16(TL1, 2), _mm_add_epi16(TL2, TR1));
    T0 = _mm_add_epi16(_mm_slli_epi16(TL0, 1), _mm_add_epi16(T0, T2));
    V0 = _mm_srli_epi16(T0, 4);
    T0 = _mm_add_epi16(_mm_slli_epi16(TR1, 2), _mm_add_epi16(TR2, TL1));
    T0 = _mm_add_epi16(_mm_slli_epi16(TR0, 1), _mm_add_epi16(T0, T2));
    V1 = _mm_srli_epi16(T0, 4);

    UL0 = _mm_blendv_epi8(UL0, V0, _mm_cmpeq_epi16(FS, c_3));
    UR0 = _mm_blendv_epi8(UR0, V1, _mm_cmpeq_epi16(FS, c_3));

    T0 = _mm_add_epi16(_mm_add_epi16(TL2, TR0), _mm_slli_epi16(TL2, 1));
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(TL1, 3));
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(TL0, 2));
    V2 = _mm_srli_epi16(_mm_add_epi16(T0, c_8), 4);

    T0 = _mm_add_epi16(_mm_add_epi16(TR2, TL0), _mm_slli_epi16(TR2, 1));
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(TR1, 3));
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(TR0, 2));
    V3 = _mm_srli_epi16(_mm_add_epi16(T0, c_8), 4);

    UL1 = _mm_blendv_epi8(UL1, V2, _mm_cmpeq_epi16(FS, c_3));
    UR1 = _mm_blendv_epi8(UR1, V3, _mm_cmpeq_epi16(FS, c_3));

    /* store result */
    UL0 = _mm_packus_epi16(UL0, c_0);
    UL1 = _mm_packus_epi16(UL1, c_0);
    UR0 = _mm_packus_epi16(UR0, c_0);
    UR1 = _mm_packus_epi16(UR1, c_0);

    ((int32_t*)(SrcPtrU - inc ))[0] = M128_I32(UL0, 0);
    ((int32_t*)(SrcPtrU       ))[0] = M128_I32(UR0, 0);
    ((int32_t*)(SrcPtrU - inc2))[0] = M128_I32(UL1, 0);
    ((int32_t*)(SrcPtrU + inc ))[0] = M128_I32(UR1, 0);
    ((int32_t*)(SrcPtrV - inc ))[0] = M128_I32(UL0, 1);
    ((int32_t*)(SrcPtrV       ))[0] = M128_I32(UR0, 1);
    ((int32_t*)(SrcPtrV - inc2))[0] = M128_I32(UL1, 1);
    ((int32_t*)(SrcPtrV + inc ))[0] = M128_I32(UR1, 1);
}
