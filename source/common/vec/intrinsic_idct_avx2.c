/*
 * intrinsic_idct_avx2.c
 *
 * Description of this file:
 *    AVX2 assembly functions of IDCT module of the xavs2 library
 *
 * --------------------------------------------------------------------------
 *
 *    xavs2 - video encoder of AVS2/IEEE1857.4 video coding standard
 *    Copyright (C) 2018~ VCL, NELVT, Peking University
 *
 *    Authors: Falei LUO <falei.luo@gmail.com>
               Jiaqi ZHANG <zhangjiaqi.cs@gmail.com>
               Tianliang FU <futl@pku.edu.cn>
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
#include "../avs2_defs.h"
#include "intrinsic.h"

/* disable warnings */
#pragma warning(disable:4127)  // warning C4127: 条件表达式是常量


ALIGN32(static const coeff_t tab_idct_8x8_256[12][16]) = {
    { 44, 38, 44, 38, 44, 38, 44, 38, 44, 38, 44, 38, 44, 38, 44, 38 },
    { 25, 9, 25, 9, 25, 9, 25, 9, 25, 9, 25, 9, 25, 9, 25, 9 },
    { 38, -9, 38, -9, 38, -9, 38, -9, 38, -9, 38, -9, 38, -9, 38, -9 },
    { -44, -25, -44, -25, -44, -25, -44, -25, -44, -25, -44, -25, -44, -25, -44, -25 },
    { 25, -44, 25, -44, 25, -44, 25, -44, 25, -44, 25, -44, 25, -44, 25, -44 },
    { 9, 38, 9, 38, 9, 38, 9, 38, 9, 38, 9, 38, 9, 38, 9, 38 },
    { 9, -25, 9, -25, 9, -25, 9, -25, 9, -25, 9, -25, 9, -25, 9, -25 },
    { 38, -44, 38, -44, 38, -44, 38, -44, 38, -44, 38, -44, 38, -44, 38, -44 },
    { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 },
    { 32, -32, 32, -32, 32, -32, 32, -32, 32, -32, 32, -32, 32, -32, 32, -32 },
    { 42, 17, 42, 17, 42, 17, 42, 17, 42, 17, 42, 17, 42, 17, 42, 17 },
    { 17, -42, 17, -42, 17, -42, 17, -42, 17, -42, 17, -42, 17, -42, 17, -42 }
};

void idct_c_8x8_avx2(const coeff_t *src, coeff_t *dst, int i_dst)
{
    const int SHIFT1 = 5;
    // const int CLIP1 = LIMIT_BIT;
    const int SHIFT2 = 20 - g_bit_depth;
    const int CLIP2 = g_bit_depth + 1;

    __m256i mAdd;
    __m256i S1S5, S3S7;
    __m256i T0, T1, T2, T3;
    __m256i E0, E1, E2, E3, O0, O1, O2, O3;
    __m256i EE0, EE1, EO0, EO1;
    __m256i S0, S1, S2, S3, S4, S5, S6, S7;
    __m256i C00, C01, C02, C03, C04, C05, C06, C07;
    __m256i max_val, min_val;

    UNUSED_PARAMETER(i_dst);
    S1S5 = _mm256_loadu2_m128i((__m128i*)&src[40], (__m128i*)&src[ 8]);
    S3S7 = _mm256_loadu2_m128i((__m128i*)&src[56], (__m128i*)&src[24]);

    T0 = _mm256_unpacklo_epi16(S1S5, S3S7);
    T1 = _mm256_unpackhi_epi16(S1S5, S3S7);

    T2 = _mm256_permute2x128_si256(T0, T1, 0x20);
    T3 = _mm256_permute2x128_si256(T0, T1, 0x31);

    O0 = _mm256_add_epi32(_mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[0]))),
                          _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[1]))));
    O1 = _mm256_add_epi32(_mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[2]))),
                          _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[3]))));
    O2 = _mm256_add_epi32(_mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[4]))),
                          _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[5]))));
    O3 = _mm256_add_epi32(_mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[6]))),
                          _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[7]))));

    /*    -------     */
    S1S5 = _mm256_loadu2_m128i((__m128i*)&src[16], (__m128i*)&src[0]);
    S3S7 = _mm256_loadu2_m128i((__m128i*)&src[48], (__m128i*)&src[32]);

    T0 = _mm256_unpacklo_epi16(S1S5, S3S7);
    T1 = _mm256_unpackhi_epi16(S1S5, S3S7);

    T2 = _mm256_permute2x128_si256(T0, T1, 0x20);
    T3 = _mm256_permute2x128_si256(T0, T1, 0x31);

    EE0 = _mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[8])));
    EE1 = _mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[9])));
    EO0 = _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[10])));
    EO1 = _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[11])));

    /*    -------     */
    mAdd = _mm256_set1_epi32((1 << (SHIFT1 - 1)));                               // 首次反变换四舍五入的数字

    E0 = _mm256_add_epi32(EE0, EO0);
    E1 = _mm256_add_epi32(EE1, EO1);
    E3 = _mm256_sub_epi32(EE0, EO0);
    E2 = _mm256_sub_epi32(EE1, EO1);
    E0 = _mm256_add_epi32(E0, mAdd);
    E1 = _mm256_add_epi32(E1, mAdd);
    E2 = _mm256_add_epi32(E2, mAdd);
    E3 = _mm256_add_epi32(E3, mAdd);

    S0 = _mm256_srai_epi32(_mm256_add_epi32(E0, O0), SHIFT1);
    S7 = _mm256_srai_epi32(_mm256_sub_epi32(E0, O0), SHIFT1);
    S1 = _mm256_srai_epi32(_mm256_add_epi32(E1, O1), SHIFT1);
    S6 = _mm256_srai_epi32(_mm256_sub_epi32(E1, O1), SHIFT1);
    S2 = _mm256_srai_epi32(_mm256_add_epi32(E2, O2), SHIFT1);
    S5 = _mm256_srai_epi32(_mm256_sub_epi32(E2, O2), SHIFT1);
    S3 = _mm256_srai_epi32(_mm256_add_epi32(E3, O3), SHIFT1);
    S4 = _mm256_srai_epi32(_mm256_sub_epi32(E3, O3), SHIFT1);

    C00 = _mm256_permute2x128_si256(S0, S4, 0x20);
    C01 = _mm256_permute2x128_si256(S0, S4, 0x31);

    C02 = _mm256_permute2x128_si256(S1, S5, 0x20);
    C03 = _mm256_permute2x128_si256(S1, S5, 0x31);

    C04 = _mm256_permute2x128_si256(S2, S6, 0x20);
    C05 = _mm256_permute2x128_si256(S2, S6, 0x31);

    C06 = _mm256_permute2x128_si256(S3, S7, 0x20);
    C07 = _mm256_permute2x128_si256(S3, S7, 0x31);

    S0 = _mm256_packs_epi32(C00, C01);
    S1 = _mm256_packs_epi32(C02, C03);
    S2 = _mm256_packs_epi32(C04, C05);
    S3 = _mm256_packs_epi32(C06, C07);

    S4 = _mm256_unpacklo_epi16(S0, S1);
    S5 = _mm256_unpacklo_epi16(S2, S3);
    S6 = _mm256_unpackhi_epi16(S0, S1);
    S7 = _mm256_unpackhi_epi16(S2, S3);

    C00 = _mm256_unpacklo_epi32(S4, S5);
    C01 = _mm256_unpacklo_epi32(S6, S7);
    C02 = _mm256_unpackhi_epi32(S4, S5);
    C03 = _mm256_unpackhi_epi32(S6, S7);

    C04 = _mm256_permute2x128_si256(C00, C02, 0x20);
    C05 = _mm256_permute2x128_si256(C00, C02, 0x31);
    C06 = _mm256_permute2x128_si256(C01, C03, 0x20);
    C07 = _mm256_permute2x128_si256(C01, C03, 0x31);

    S0 = _mm256_unpacklo_epi64(C04, C05);
    S1 = _mm256_unpacklo_epi64(C06, C07);

    S2 = _mm256_unpackhi_epi64(C04, C05);
    S3 = _mm256_unpackhi_epi64(C06, C07);

    S4 = _mm256_permute2x128_si256(S2, S3, 0x20);
    S5 = _mm256_permute2x128_si256(S2, S3, 0x31);


    T0 = _mm256_unpacklo_epi16(S4, S5);
    T1 = _mm256_unpackhi_epi16(S4, S5);

    T2 = _mm256_permute2x128_si256(T0, T1, 0x20);
    T3 = _mm256_permute2x128_si256(T0, T1, 0x31);

    O0 = _mm256_add_epi32(_mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[0]))),
                          _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[1]))));
    O1 = _mm256_add_epi32(_mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[2]))),
                          _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[3]))));
    O2 = _mm256_add_epi32(_mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[4]))),
                          _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[5]))));
    O3 = _mm256_add_epi32(_mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[6]))),
                          _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[7]))));

    /*    -------     */
    T0 = _mm256_unpacklo_epi16(S0, S1);
    T1 = _mm256_unpackhi_epi16(S0, S1);

    T2 = _mm256_permute2x128_si256(T0, T1, 0x20);
    T3 = _mm256_permute2x128_si256(T0, T1, 0x31);

    EE0 = _mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[8])));
    EE1 = _mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[9])));
    EO0 = _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[10])));
    EO1 = _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[11])));

    /*    -------     */
    mAdd = _mm256_set1_epi32(SHIFT2 ? (1 << (SHIFT2 - 1)) : 0);                       //设置四舍五入

    E0 = _mm256_add_epi32(EE0, EO0);
    E1 = _mm256_add_epi32(EE1, EO1);
    E3 = _mm256_sub_epi32(EE0, EO0);
    E2 = _mm256_sub_epi32(EE1, EO1);
    E0 = _mm256_add_epi32(E0, mAdd);
    E1 = _mm256_add_epi32(E1, mAdd);
    E2 = _mm256_add_epi32(E2, mAdd);
    E3 = _mm256_add_epi32(E3, mAdd);

    S0 = _mm256_srai_epi32(_mm256_add_epi32(E0, O0), SHIFT2);
    S7 = _mm256_srai_epi32(_mm256_sub_epi32(E0, O0), SHIFT2);
    S1 = _mm256_srai_epi32(_mm256_add_epi32(E1, O1), SHIFT2);
    S6 = _mm256_srai_epi32(_mm256_sub_epi32(E1, O1), SHIFT2);
    S2 = _mm256_srai_epi32(_mm256_add_epi32(E2, O2), SHIFT2);
    S5 = _mm256_srai_epi32(_mm256_sub_epi32(E2, O2), SHIFT2);
    S3 = _mm256_srai_epi32(_mm256_add_epi32(E3, O3), SHIFT2);
    S4 = _mm256_srai_epi32(_mm256_sub_epi32(E3, O3), SHIFT2);

    C00 = _mm256_permute2x128_si256(S0, S4, 0x20);
    C01 = _mm256_permute2x128_si256(S0, S4, 0x31);

    C02 = _mm256_permute2x128_si256(S1, S5, 0x20);
    C03 = _mm256_permute2x128_si256(S1, S5, 0x31);

    C04 = _mm256_permute2x128_si256(S2, S6, 0x20);
    C05 = _mm256_permute2x128_si256(S2, S6, 0x31);

    C06 = _mm256_permute2x128_si256(S3, S7, 0x20);
    C07 = _mm256_permute2x128_si256(S3, S7, 0x31);

    S0 = _mm256_packs_epi32(C00, C01);
    S1 = _mm256_packs_epi32(C02, C03);
    S2 = _mm256_packs_epi32(C04, C05);
    S3 = _mm256_packs_epi32(C06, C07);

    S4 = _mm256_unpacklo_epi16(S0, S1);
    S5 = _mm256_unpacklo_epi16(S2, S3);
    S6 = _mm256_unpackhi_epi16(S0, S1);
    S7 = _mm256_unpackhi_epi16(S2, S3);

    C00 = _mm256_unpacklo_epi32(S4, S5);
    C01 = _mm256_unpacklo_epi32(S6, S7);
    C02 = _mm256_unpackhi_epi32(S4, S5);
    C03 = _mm256_unpackhi_epi32(S6, S7);

    C04 = _mm256_permute2x128_si256(C00, C02, 0x20);
    C05 = _mm256_permute2x128_si256(C00, C02, 0x31);
    C06 = _mm256_permute2x128_si256(C01, C03, 0x20);
    C07 = _mm256_permute2x128_si256(C01, C03, 0x31);

    S0 = _mm256_unpacklo_epi64(C04, C05);
    S1 = _mm256_unpacklo_epi64(C06, C07);
    S2 = _mm256_unpackhi_epi64(C04, C05);
    S3 = _mm256_unpackhi_epi64(C06, C07);

    // CLIP2
    max_val = _mm256_set1_epi16((1 << (CLIP2 - 1)) - 1);
    min_val = _mm256_set1_epi16(-(1 << (CLIP2 - 1)));

    S0 = _mm256_max_epi16(_mm256_min_epi16(S0, max_val), min_val);
    S1 = _mm256_max_epi16(_mm256_min_epi16(S1, max_val), min_val);
    S2 = _mm256_max_epi16(_mm256_min_epi16(S2, max_val), min_val);
    S3 = _mm256_max_epi16(_mm256_min_epi16(S3, max_val), min_val);

    // store
    _mm256_storeu2_m128i((__m128i*)&dst[16], (__m128i*)&dst[ 0], S0);
    _mm256_storeu2_m128i((__m128i*)&dst[48], (__m128i*)&dst[32], S1);
    _mm256_storeu2_m128i((__m128i*)&dst[24], (__m128i*)&dst[ 8], S2);
    _mm256_storeu2_m128i((__m128i*)&dst[56], (__m128i*)&dst[40], S3);
}


void idct_c_16x16_avx2(const coeff_t *src, coeff_t *dst, int i_dst)
{
    const int shift = 20-g_bit_depth;
    const int clip = g_bit_depth + 1;

    const __m256i c16_p43_p45 = _mm256_set1_epi32(0x002B002D);      //row0 87high - 90low address
    const __m256i c16_p35_p40 = _mm256_set1_epi32(0x00230028);
    const __m256i c16_p21_p29 = _mm256_set1_epi32(0x0015001D);
    const __m256i c16_p04_p13 = _mm256_set1_epi32(0x0004000D);
    const __m256i c16_p29_p43 = _mm256_set1_epi32(0x001D002B);      //row1
    const __m256i c16_n21_p04 = _mm256_set1_epi32(0xFFEB0004);
    const __m256i c16_n45_n40 = _mm256_set1_epi32(0xFFD3FFD8);
    const __m256i c16_n13_n35 = _mm256_set1_epi32(0xFFF3FFDD);
    const __m256i c16_p04_p40 = _mm256_set1_epi32(0x00040028);      //row2
    const __m256i c16_n43_n35 = _mm256_set1_epi32(0xFFD5FFDD);
    const __m256i c16_p29_n13 = _mm256_set1_epi32(0x001DFFF3);
    const __m256i c16_p21_p45 = _mm256_set1_epi32(0x0015002D);
    const __m256i c16_n21_p35 = _mm256_set1_epi32(0xFFEB0023);      //row3
    const __m256i c16_p04_n43 = _mm256_set1_epi32(0x0004FFD5);
    const __m256i c16_p13_p45 = _mm256_set1_epi32(0x000D002D);
    const __m256i c16_n29_n40 = _mm256_set1_epi32(0xFFE3FFD8);
    const __m256i c16_n40_p29 = _mm256_set1_epi32(0xFFD8001D);      //row4
    const __m256i c16_p45_n13 = _mm256_set1_epi32(0x002DFFF3);
    const __m256i c16_n43_n04 = _mm256_set1_epi32(0xFFD5FFFC);
    const __m256i c16_p35_p21 = _mm256_set1_epi32(0x00230015);
    const __m256i c16_n45_p21 = _mm256_set1_epi32(0xFFD30015);      //row5
    const __m256i c16_p13_p29 = _mm256_set1_epi32(0x000D001D);
    const __m256i c16_p35_n43 = _mm256_set1_epi32(0x0023FFD5);
    const __m256i c16_n40_p04 = _mm256_set1_epi32(0xFFD80004);
    const __m256i c16_n35_p13 = _mm256_set1_epi32(0xFFDD000D);      //row6
    const __m256i c16_n40_p45 = _mm256_set1_epi32(0xFFD8002D);
    const __m256i c16_p04_p21 = _mm256_set1_epi32(0x00040015);
    const __m256i c16_p43_n29 = _mm256_set1_epi32(0x002BFFE3);
    const __m256i c16_n13_p04 = _mm256_set1_epi32(0xFFF30004);      //row7
    const __m256i c16_n29_p21 = _mm256_set1_epi32(0xFFE30015);
    const __m256i c16_n40_p35 = _mm256_set1_epi32(0xFFD80023);
    const __m256i c16_n45_p43 = _mm256_set1_epi32(0xFFD3002B);

    const __m256i c16_p38_p44 = _mm256_set1_epi32(0x0026002C);
    const __m256i c16_p09_p25 = _mm256_set1_epi32(0x00090019);
    const __m256i c16_n09_p38 = _mm256_set1_epi32(0xFFF70026);
    const __m256i c16_n25_n44 = _mm256_set1_epi32(0xFFE7FFD4);
    const __m256i c16_n44_p25 = _mm256_set1_epi32(0xFFD40019);
    const __m256i c16_p38_p09 = _mm256_set1_epi32(0x00260009);
    const __m256i c16_n25_p09 = _mm256_set1_epi32(0xFFE70009);
    const __m256i c16_n44_p38 = _mm256_set1_epi32(0xFFD40026);

    const __m256i c16_p17_p42 = _mm256_set1_epi32(0x0011002A);
    const __m256i c16_n42_p17 = _mm256_set1_epi32(0xFFD60011);

    const __m256i c16_n32_p32 = _mm256_set1_epi32(0xFFE00020);
    const __m256i c16_p32_p32 = _mm256_set1_epi32(0x00200020);

    __m256i max_val, min_val;
    __m256i c32_rnd = _mm256_set1_epi32(16);                                    // 第一次四舍五入

    int nShift = 5;
    int pass;

    __m256i in00, in01, in02, in03, in04, in05, in06, in07;
    __m256i in08, in09, in10, in11, in12, in13, in14, in15;
    __m256i res00, res01, res02, res03, res04, res05, res06, res07;
    __m256i res08, res09, res10, res11, res12, res13, res14, res15;


    UNUSED_PARAMETER(i_dst);

    in00 = _mm256_lddqu_si256((const __m256i*)&src[0 * 16]);    // [07 06 05 04 03 02 01 00]
    in01 = _mm256_lddqu_si256((const __m256i*)&src[1 * 16]);    // [17 16 15 14 13 12 11 10]
    in02 = _mm256_lddqu_si256((const __m256i*)&src[2 * 16]);    // [27 26 25 24 23 22 21 20]
    in03 = _mm256_lddqu_si256((const __m256i*)&src[3 * 16]);    // [37 36 35 34 33 32 31 30]
    in04 = _mm256_lddqu_si256((const __m256i*)&src[4 * 16]);    // [47 46 45 44 43 42 41 40]
    in05 = _mm256_lddqu_si256((const __m256i*)&src[5 * 16]);    // [57 56 55 54 53 52 51 50]
    in06 = _mm256_lddqu_si256((const __m256i*)&src[6 * 16]);    // [67 66 65 64 63 62 61 60]
    in07 = _mm256_lddqu_si256((const __m256i*)&src[7 * 16]);    // [77 76 75 74 73 72 71 70]
    in08 = _mm256_lddqu_si256((const __m256i*)&src[8 * 16]);
    in09 = _mm256_lddqu_si256((const __m256i*)&src[9 * 16]);
    in10 = _mm256_lddqu_si256((const __m256i*)&src[10 * 16]);
    in11 = _mm256_lddqu_si256((const __m256i*)&src[11 * 16]);
    in12 = _mm256_lddqu_si256((const __m256i*)&src[12 * 16]);
    in13 = _mm256_lddqu_si256((const __m256i*)&src[13 * 16]);
    in14 = _mm256_lddqu_si256((const __m256i*)&src[14 * 16]);
    in15 = _mm256_lddqu_si256((const __m256i*)&src[15 * 16]);


    for (pass = 0; pass < 2; pass++) {
        const __m256i T_00_00A = _mm256_unpacklo_epi16(in01, in03);       // [33 13 32 12 31 11 30 10]
        const __m256i T_00_00B = _mm256_unpackhi_epi16(in01, in03);       // [37 17 36 16 35 15 34 14]
        const __m256i T_00_01A = _mm256_unpacklo_epi16(in05, in07);       // [ ]
        const __m256i T_00_01B = _mm256_unpackhi_epi16(in05, in07);       // [ ]
        const __m256i T_00_02A = _mm256_unpacklo_epi16(in09, in11);       // [ ]
        const __m256i T_00_02B = _mm256_unpackhi_epi16(in09, in11);       // [ ]
        const __m256i T_00_03A = _mm256_unpacklo_epi16(in13, in15);       // [ ]
        const __m256i T_00_03B = _mm256_unpackhi_epi16(in13, in15);       // [ ]
        const __m256i T_00_04A = _mm256_unpacklo_epi16(in02, in06);       // [ ]
        const __m256i T_00_04B = _mm256_unpackhi_epi16(in02, in06);       // [ ]
        const __m256i T_00_05A = _mm256_unpacklo_epi16(in10, in14);       // [ ]
        const __m256i T_00_05B = _mm256_unpackhi_epi16(in10, in14);       // [ ]
        const __m256i T_00_06A = _mm256_unpacklo_epi16(in04, in12);       // [ ]row
        const __m256i T_00_06B = _mm256_unpackhi_epi16(in04, in12);       // [ ]
        const __m256i T_00_07A = _mm256_unpacklo_epi16(in00, in08);       // [83 03 82 02 81 01 81 00] row08 row00
        const __m256i T_00_07B = _mm256_unpackhi_epi16(in00, in08);       // [87 07 86 06 85 05 84 04]

        __m256i O0A, O1A, O2A, O3A, O4A, O5A, O6A, O7A;
        __m256i O0B, O1B, O2B, O3B, O4B, O5B, O6B, O7B;
        __m256i EO0A, EO1A, EO2A, EO3A;
        __m256i EO0B, EO1B, EO2B, EO3B;
        __m256i EEO0A, EEO1A;
        __m256i EEO0B, EEO1B;
        __m256i EEE0A, EEE1A;
        __m256i EEE0B, EEE1B;

        {
            __m256i T00, T01;
#define COMPUTE_ROW(row0103, row0507, row0911, row1315, c0103, c0507, c0911, c1315, row) \
    T00 = _mm256_add_epi32(_mm256_madd_epi16(row0103, c0103), _mm256_madd_epi16(row0507, c0507)); \
    T01 = _mm256_add_epi32(_mm256_madd_epi16(row0911, c0911), _mm256_madd_epi16(row1315, c1315)); \
    row = _mm256_add_epi32(T00, T01);

            COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_p43_p45, c16_p35_p40, c16_p21_p29, c16_p04_p13, O0A)
            COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_p29_p43, c16_n21_p04, c16_n45_n40, c16_n13_n35, O1A)
            COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_p04_p40, c16_n43_n35, c16_p29_n13, c16_p21_p45, O2A)
            COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n21_p35, c16_p04_n43, c16_p13_p45, c16_n29_n40, O3A)
            COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n40_p29, c16_p45_n13, c16_n43_n04, c16_p35_p21, O4A)
            COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n45_p21, c16_p13_p29, c16_p35_n43, c16_n40_p04, O5A)
            COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n35_p13, c16_n40_p45, c16_p04_p21, c16_p43_n29, O6A)
            COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n13_p04, c16_n29_p21, c16_n40_p35, c16_n45_p43, O7A)

            COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, c16_p43_p45, c16_p35_p40, c16_p21_p29, c16_p04_p13, O0B)
            COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, c16_p29_p43, c16_n21_p04, c16_n45_n40, c16_n13_n35, O1B)
            COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, c16_p04_p40, c16_n43_n35, c16_p29_n13, c16_p21_p45, O2B)
            COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, c16_n21_p35, c16_p04_n43, c16_p13_p45, c16_n29_n40, O3B)
            COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, c16_n40_p29, c16_p45_n13, c16_n43_n04, c16_p35_p21, O4B)
            COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, c16_n45_p21, c16_p13_p29, c16_p35_n43, c16_n40_p04, O5B)
            COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, c16_n35_p13, c16_n40_p45, c16_p04_p21, c16_p43_n29, O6B)
            COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, c16_n13_p04, c16_n29_p21, c16_n40_p35, c16_n45_p43, O7B)
#undef COMPUTE_ROW
        }

        EO0A = _mm256_add_epi32(_mm256_madd_epi16(T_00_04A, c16_p38_p44), _mm256_madd_epi16(T_00_05A, c16_p09_p25)); // EO0
        EO0B = _mm256_add_epi32(_mm256_madd_epi16(T_00_04B, c16_p38_p44), _mm256_madd_epi16(T_00_05B, c16_p09_p25));
        EO1A = _mm256_add_epi32(_mm256_madd_epi16(T_00_04A, c16_n09_p38), _mm256_madd_epi16(T_00_05A, c16_n25_n44)); // EO1
        EO1B = _mm256_add_epi32(_mm256_madd_epi16(T_00_04B, c16_n09_p38), _mm256_madd_epi16(T_00_05B, c16_n25_n44));
        EO2A = _mm256_add_epi32(_mm256_madd_epi16(T_00_04A, c16_n44_p25), _mm256_madd_epi16(T_00_05A, c16_p38_p09)); // EO2
        EO2B = _mm256_add_epi32(_mm256_madd_epi16(T_00_04B, c16_n44_p25), _mm256_madd_epi16(T_00_05B, c16_p38_p09));
        EO3A = _mm256_add_epi32(_mm256_madd_epi16(T_00_04A, c16_n25_p09), _mm256_madd_epi16(T_00_05A, c16_n44_p38)); // EO3
        EO3B = _mm256_add_epi32(_mm256_madd_epi16(T_00_04B, c16_n25_p09), _mm256_madd_epi16(T_00_05B, c16_n44_p38));

        EEO0A = _mm256_madd_epi16(T_00_06A, c16_p17_p42);
        EEO0B = _mm256_madd_epi16(T_00_06B, c16_p17_p42);
        EEO1A = _mm256_madd_epi16(T_00_06A, c16_n42_p17);
        EEO1B = _mm256_madd_epi16(T_00_06B, c16_n42_p17);

        EEE0A = _mm256_madd_epi16(T_00_07A, c16_p32_p32);
        EEE0B = _mm256_madd_epi16(T_00_07B, c16_p32_p32);
        EEE1A = _mm256_madd_epi16(T_00_07A, c16_n32_p32);
        EEE1B = _mm256_madd_epi16(T_00_07B, c16_n32_p32);
        {
            const __m256i EE0A = _mm256_add_epi32(EEE0A, EEO0A);          // EE0 = EEE0 + EEO0
            const __m256i EE0B = _mm256_add_epi32(EEE0B, EEO0B);
            const __m256i EE1A = _mm256_add_epi32(EEE1A, EEO1A);          // EE1 = EEE1 + EEO1
            const __m256i EE1B = _mm256_add_epi32(EEE1B, EEO1B);
            const __m256i EE3A = _mm256_sub_epi32(EEE0A, EEO0A);          // EE2 = EEE0 - EEO0
            const __m256i EE3B = _mm256_sub_epi32(EEE0B, EEO0B);
            const __m256i EE2A = _mm256_sub_epi32(EEE1A, EEO1A);          // EE3 = EEE1 - EEO1
            const __m256i EE2B = _mm256_sub_epi32(EEE1B, EEO1B);

            const __m256i E0A = _mm256_add_epi32(EE0A, EO0A);          // E0 = EE0 + EO0
            const __m256i E0B = _mm256_add_epi32(EE0B, EO0B);
            const __m256i E1A = _mm256_add_epi32(EE1A, EO1A);          // E1 = EE1 + EO1
            const __m256i E1B = _mm256_add_epi32(EE1B, EO1B);
            const __m256i E2A = _mm256_add_epi32(EE2A, EO2A);          // E2 = EE2 + EO2
            const __m256i E2B = _mm256_add_epi32(EE2B, EO2B);
            const __m256i E3A = _mm256_add_epi32(EE3A, EO3A);          // E3 = EE3 + EO3
            const __m256i E3B = _mm256_add_epi32(EE3B, EO3B);
            const __m256i E7A = _mm256_sub_epi32(EE0A, EO0A);          // E7 = EE0 - EO0
            const __m256i E7B = _mm256_sub_epi32(EE0B, EO0B);
            const __m256i E6A = _mm256_sub_epi32(EE1A, EO1A);          // E6 = EE1 - EO1
            const __m256i E6B = _mm256_sub_epi32(EE1B, EO1B);
            const __m256i E5A = _mm256_sub_epi32(EE2A, EO2A);          // E5 = EE2 - EO2
            const __m256i E5B = _mm256_sub_epi32(EE2B, EO2B);
            const __m256i E4A = _mm256_sub_epi32(EE3A, EO3A);          // E4 = EE3 - EO3
            const __m256i E4B = _mm256_sub_epi32(EE3B, EO3B);

            const __m256i T10A = _mm256_add_epi32(E0A, c32_rnd);         // E0 + rnd
            const __m256i T10B = _mm256_add_epi32(E0B, c32_rnd);
            const __m256i T11A = _mm256_add_epi32(E1A, c32_rnd);         // E1 + rnd
            const __m256i T11B = _mm256_add_epi32(E1B, c32_rnd);
            const __m256i T12A = _mm256_add_epi32(E2A, c32_rnd);         // E2 + rnd
            const __m256i T12B = _mm256_add_epi32(E2B, c32_rnd);
            const __m256i T13A = _mm256_add_epi32(E3A, c32_rnd);         // E3 + rnd
            const __m256i T13B = _mm256_add_epi32(E3B, c32_rnd);
            const __m256i T14A = _mm256_add_epi32(E4A, c32_rnd);         // E4 + rnd
            const __m256i T14B = _mm256_add_epi32(E4B, c32_rnd);
            const __m256i T15A = _mm256_add_epi32(E5A, c32_rnd);         // E5 + rnd
            const __m256i T15B = _mm256_add_epi32(E5B, c32_rnd);
            const __m256i T16A = _mm256_add_epi32(E6A, c32_rnd);         // E6 + rnd
            const __m256i T16B = _mm256_add_epi32(E6B, c32_rnd);
            const __m256i T17A = _mm256_add_epi32(E7A, c32_rnd);         // E7 + rnd
            const __m256i T17B = _mm256_add_epi32(E7B, c32_rnd);

            const __m256i T20A = _mm256_add_epi32(T10A, O0A);          // E0 + O0 + rnd
            const __m256i T20B = _mm256_add_epi32(T10B, O0B);
            const __m256i T21A = _mm256_add_epi32(T11A, O1A);          // E1 + O1 + rnd
            const __m256i T21B = _mm256_add_epi32(T11B, O1B);
            const __m256i T22A = _mm256_add_epi32(T12A, O2A);          // E2 + O2 + rnd
            const __m256i T22B = _mm256_add_epi32(T12B, O2B);
            const __m256i T23A = _mm256_add_epi32(T13A, O3A);          // E3 + O3 + rnd
            const __m256i T23B = _mm256_add_epi32(T13B, O3B);
            const __m256i T24A = _mm256_add_epi32(T14A, O4A);          // E4
            const __m256i T24B = _mm256_add_epi32(T14B, O4B);
            const __m256i T25A = _mm256_add_epi32(T15A, O5A);          // E5
            const __m256i T25B = _mm256_add_epi32(T15B, O5B);
            const __m256i T26A = _mm256_add_epi32(T16A, O6A);          // E6
            const __m256i T26B = _mm256_add_epi32(T16B, O6B);
            const __m256i T27A = _mm256_add_epi32(T17A, O7A);          // E7
            const __m256i T27B = _mm256_add_epi32(T17B, O7B);
            const __m256i T2FA = _mm256_sub_epi32(T10A, O0A);          // E0 - O0 + rnd
            const __m256i T2FB = _mm256_sub_epi32(T10B, O0B);
            const __m256i T2EA = _mm256_sub_epi32(T11A, O1A);          // E1 - O1 + rnd
            const __m256i T2EB = _mm256_sub_epi32(T11B, O1B);
            const __m256i T2DA = _mm256_sub_epi32(T12A, O2A);          // E2 - O2 + rnd
            const __m256i T2DB = _mm256_sub_epi32(T12B, O2B);
            const __m256i T2CA = _mm256_sub_epi32(T13A, O3A);          // E3 - O3 + rnd
            const __m256i T2CB = _mm256_sub_epi32(T13B, O3B);
            const __m256i T2BA = _mm256_sub_epi32(T14A, O4A);          // E4
            const __m256i T2BB = _mm256_sub_epi32(T14B, O4B);
            const __m256i T2AA = _mm256_sub_epi32(T15A, O5A);          // E5
            const __m256i T2AB = _mm256_sub_epi32(T15B, O5B);
            const __m256i T29A = _mm256_sub_epi32(T16A, O6A);          // E6
            const __m256i T29B = _mm256_sub_epi32(T16B, O6B);
            const __m256i T28A = _mm256_sub_epi32(T17A, O7A);          // E7
            const __m256i T28B = _mm256_sub_epi32(T17B, O7B);

            const __m256i T30A = _mm256_srai_epi32(T20A, nShift);             // [30 20 10 00] // This operation make it much slower than 128
            const __m256i T30B = _mm256_srai_epi32(T20B, nShift);             // [70 60 50 40] // This operation make it much slower than 128
            const __m256i T31A = _mm256_srai_epi32(T21A, nShift);             // [31 21 11 01] // This operation make it much slower than 128
            const __m256i T31B = _mm256_srai_epi32(T21B, nShift);             // [71 61 51 41] // This operation make it much slower than 128
            const __m256i T32A = _mm256_srai_epi32(T22A, nShift);             // [32 22 12 02] // This operation make it much slower than 128
            const __m256i T32B = _mm256_srai_epi32(T22B, nShift);             // [72 62 52 42] // This operation make it much slower than 128
            const __m256i T33A = _mm256_srai_epi32(T23A, nShift);             // [33 23 13 03] // This operation make it much slower than 128
            const __m256i T33B = _mm256_srai_epi32(T23B, nShift);             // [73 63 53 43] // This operation make it much slower than 128
            const __m256i T34A = _mm256_srai_epi32(T24A, nShift);             // [33 24 14 04] // This operation make it much slower than 128
            const __m256i T34B = _mm256_srai_epi32(T24B, nShift);             // [74 64 54 44] // This operation make it much slower than 128
            const __m256i T35A = _mm256_srai_epi32(T25A, nShift);             // [35 25 15 05] // This operation make it much slower than 128
            const __m256i T35B = _mm256_srai_epi32(T25B, nShift);             // [75 65 55 45] // This operation make it much slower than 128
            const __m256i T36A = _mm256_srai_epi32(T26A, nShift);             // [36 26 16 06] // This operation make it much slower than 128
            const __m256i T36B = _mm256_srai_epi32(T26B, nShift);             // [76 66 56 46] // This operation make it much slower than 128
            const __m256i T37A = _mm256_srai_epi32(T27A, nShift);             // [37 27 17 07] // This operation make it much slower than 128
            const __m256i T37B = _mm256_srai_epi32(T27B, nShift);             // [77 67 57 47] // This operation make it much slower than 128

            const __m256i T38A = _mm256_srai_epi32(T28A, nShift);             // [30 20 10 00] x8 // This operation make it much slower than 128
            const __m256i T38B = _mm256_srai_epi32(T28B, nShift);             // [70 60 50 40]
            const __m256i T39A = _mm256_srai_epi32(T29A, nShift);             // [31 21 11 01] x9 // This operation make it much slower than 128
            const __m256i T39B = _mm256_srai_epi32(T29B, nShift);             // [71 61 51 41]
            const __m256i T3AA = _mm256_srai_epi32(T2AA, nShift);             // [32 22 12 02] xA // This operation make it much slower than 128
            const __m256i T3AB = _mm256_srai_epi32(T2AB, nShift);             // [72 62 52 42]
            const __m256i T3BA = _mm256_srai_epi32(T2BA, nShift);             // [33 23 13 03] xB // This operation make it much slower than 128
            const __m256i T3BB = _mm256_srai_epi32(T2BB, nShift);             // [73 63 53 43]
            const __m256i T3CA = _mm256_srai_epi32(T2CA, nShift);             // [33 24 14 04] xC // This operation make it much slower than 128
            const __m256i T3CB = _mm256_srai_epi32(T2CB, nShift);             // [74 64 54 44]
            const __m256i T3DA = _mm256_srai_epi32(T2DA, nShift);             // [35 25 15 05] xD // This operation make it much slower than 128
            const __m256i T3DB = _mm256_srai_epi32(T2DB, nShift);             // [75 65 55 45]
            const __m256i T3EA = _mm256_srai_epi32(T2EA, nShift);             // [36 26 16 06] xE // This operation make it much slower than 128
            const __m256i T3EB = _mm256_srai_epi32(T2EB, nShift);             // [76 66 56 46]
            const __m256i T3FA = _mm256_srai_epi32(T2FA, nShift);             // [37 27 17 07] xF // This operation make it much slower than 128
            const __m256i T3FB = _mm256_srai_epi32(T2FB, nShift);             // [77 67 57 47]

            res00 = _mm256_packs_epi32(T30A, T30B);        // [70 60 50 40 30 20 10 00]
            res01 = _mm256_packs_epi32(T31A, T31B);        // [71 61 51 41 31 21 11 01]
            res02 = _mm256_packs_epi32(T32A, T32B);        // [72 62 52 42 32 22 12 02]
            res03 = _mm256_packs_epi32(T33A, T33B);        // [73 63 53 43 33 23 13 03]
            res04 = _mm256_packs_epi32(T34A, T34B);        // [74 64 54 44 34 24 14 04]
            res05 = _mm256_packs_epi32(T35A, T35B);        // [75 65 55 45 35 25 15 05]
            res06 = _mm256_packs_epi32(T36A, T36B);        // [76 66 56 46 36 26 16 06]
            res07 = _mm256_packs_epi32(T37A, T37B);        // [77 67 57 47 37 27 17 07]

            res08 = _mm256_packs_epi32(T38A, T38B);        // [A0 ... 80]
            res09 = _mm256_packs_epi32(T39A, T39B);        // [A1 ... 81]
            res10 = _mm256_packs_epi32(T3AA, T3AB);        // [A2 ... 82]
            res11 = _mm256_packs_epi32(T3BA, T3BB);        // [A3 ... 83]
            res12 = _mm256_packs_epi32(T3CA, T3CB);        // [A4 ... 84]
            res13 = _mm256_packs_epi32(T3DA, T3DB);        // [A5 ... 85]
            res14 = _mm256_packs_epi32(T3EA, T3EB);        // [A6 ... 86]
            res15 = _mm256_packs_epi32(T3FA, T3FB);        // [A7 ... 87]
        }

        //transpose matrix 16x16 16bit.
        {
            __m256i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7, tr0_8, tr0_9, tr0_10, tr0_11, tr0_12, tr0_13, tr0_14, tr0_15;
#define TRANSPOSE_16x16_16BIT(I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14, I15, O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14, O15) \
        tr0_0 = _mm256_unpacklo_epi16(I0, I1); \
        tr0_1 = _mm256_unpacklo_epi16(I2, I3); \
        tr0_2 = _mm256_unpacklo_epi16(I4, I5); \
        tr0_3 = _mm256_unpacklo_epi16(I6, I7); \
        tr0_4 = _mm256_unpacklo_epi16(I8, I9); \
        tr0_5 = _mm256_unpacklo_epi16(I10, I11); \
        tr0_6 = _mm256_unpacklo_epi16(I12, I13); \
        tr0_7 = _mm256_unpacklo_epi16(I14, I15); \
        tr0_8 = _mm256_unpackhi_epi16(I0, I1); \
        tr0_9 = _mm256_unpackhi_epi16(I2, I3); \
        tr0_10 = _mm256_unpackhi_epi16(I4, I5); \
        tr0_11 = _mm256_unpackhi_epi16(I6, I7); \
        tr0_12 = _mm256_unpackhi_epi16(I8, I9); \
        tr0_13 = _mm256_unpackhi_epi16(I10, I11); \
        tr0_14 = _mm256_unpackhi_epi16(I12, I13); \
        tr0_15 = _mm256_unpackhi_epi16(I14, I15); \
        O0 = _mm256_unpacklo_epi32(tr0_0, tr0_1); \
        O1 = _mm256_unpacklo_epi32(tr0_2, tr0_3); \
        O2 = _mm256_unpacklo_epi32(tr0_4, tr0_5); \
        O3 = _mm256_unpacklo_epi32(tr0_6, tr0_7); \
        O4 = _mm256_unpackhi_epi32(tr0_0, tr0_1); \
        O5 = _mm256_unpackhi_epi32(tr0_2, tr0_3); \
        O6 = _mm256_unpackhi_epi32(tr0_4, tr0_5); \
        O7 = _mm256_unpackhi_epi32(tr0_6, tr0_7); \
        O8 = _mm256_unpacklo_epi32(tr0_8, tr0_9); \
        O9 = _mm256_unpacklo_epi32(tr0_10, tr0_11); \
        O10 = _mm256_unpacklo_epi32(tr0_12, tr0_13); \
        O11 = _mm256_unpacklo_epi32(tr0_14, tr0_15); \
        O12 = _mm256_unpackhi_epi32(tr0_8, tr0_9); \
        O13 = _mm256_unpackhi_epi32(tr0_10, tr0_11); \
        O14 = _mm256_unpackhi_epi32(tr0_12, tr0_13); \
        O15 = _mm256_unpackhi_epi32(tr0_14, tr0_15); \
        tr0_0 = _mm256_unpacklo_epi64(O0, O1); \
        tr0_1 = _mm256_unpacklo_epi64(O2, O3); \
        tr0_2 = _mm256_unpackhi_epi64(O0, O1); \
        tr0_3 = _mm256_unpackhi_epi64(O2, O3); \
        tr0_4 = _mm256_unpacklo_epi64(O4, O5); \
        tr0_5 = _mm256_unpacklo_epi64(O6, O7); \
        tr0_6 = _mm256_unpackhi_epi64(O4, O5); \
        tr0_7 = _mm256_unpackhi_epi64(O6, O7); \
        tr0_8 = _mm256_unpacklo_epi64(O8, O9); \
        tr0_9 = _mm256_unpacklo_epi64(O10, O11); \
        tr0_10 = _mm256_unpackhi_epi64(O8, O9); \
        tr0_11 = _mm256_unpackhi_epi64(O10, O11); \
        tr0_12 = _mm256_unpacklo_epi64(O12, O13); \
        tr0_13 = _mm256_unpacklo_epi64(O14, O15); \
        tr0_14 = _mm256_unpackhi_epi64(O12, O13); \
        tr0_15 = _mm256_unpackhi_epi64(O14, O15); \
        O0 = _mm256_permute2x128_si256(tr0_0, tr0_1, 0x20); \
        O1 = _mm256_permute2x128_si256(tr0_2, tr0_3, 0x20); \
        O2 = _mm256_permute2x128_si256(tr0_4, tr0_5, 0x20); \
        O3 = _mm256_permute2x128_si256(tr0_6, tr0_7, 0x20); \
        O4 = _mm256_permute2x128_si256(tr0_8, tr0_9, 0x20); \
        O5 = _mm256_permute2x128_si256(tr0_10, tr0_11, 0x20); \
        O6 = _mm256_permute2x128_si256(tr0_12, tr0_13, 0x20); \
        O7 = _mm256_permute2x128_si256(tr0_14, tr0_15, 0x20); \
        O8 = _mm256_permute2x128_si256(tr0_0, tr0_1, 0x31); \
        O9 = _mm256_permute2x128_si256(tr0_2, tr0_3, 0x31); \
        O10 = _mm256_permute2x128_si256(tr0_4, tr0_5, 0x31); \
        O11 = _mm256_permute2x128_si256(tr0_6, tr0_7, 0x31); \
        O12 = _mm256_permute2x128_si256(tr0_8, tr0_9, 0x31); \
        O13 = _mm256_permute2x128_si256(tr0_10, tr0_11, 0x31); \
        O14 = _mm256_permute2x128_si256(tr0_12, tr0_13, 0x31); \
        O15 = _mm256_permute2x128_si256(tr0_14, tr0_15, 0x31); \
 
            TRANSPOSE_16x16_16BIT(res00, res01, res02, res03, res04, res05, res06, res07, res08, res09, res10, res11, res12, res13, res14, res15, in00, in01, in02, in03, in04, in05, in06, in07, in08, in09, in10, in11, in12, in13, in14, in15)
#undef TRANSPOSE_16x16_16BIT
        }

        nShift = shift;
        c32_rnd = _mm256_set1_epi32(shift ? (1 << (shift - 1)) : 0);                // pass == 1 第二次四舍五入
    }

    // clip
    max_val = _mm256_set1_epi16((1 << (clip - 1)) - 1);
    min_val = _mm256_set1_epi16(-(1 << (clip - 1)));

    in00 = _mm256_max_epi16(_mm256_min_epi16(in00, max_val), min_val);
    in01 = _mm256_max_epi16(_mm256_min_epi16(in01, max_val), min_val);
    in02 = _mm256_max_epi16(_mm256_min_epi16(in02, max_val), min_val);
    in03 = _mm256_max_epi16(_mm256_min_epi16(in03, max_val), min_val);
    in04 = _mm256_max_epi16(_mm256_min_epi16(in04, max_val), min_val);
    in05 = _mm256_max_epi16(_mm256_min_epi16(in05, max_val), min_val);
    in06 = _mm256_max_epi16(_mm256_min_epi16(in06, max_val), min_val);
    in07 = _mm256_max_epi16(_mm256_min_epi16(in07, max_val), min_val);
    in08 = _mm256_max_epi16(_mm256_min_epi16(in08, max_val), min_val);
    in09 = _mm256_max_epi16(_mm256_min_epi16(in09, max_val), min_val);
    in10 = _mm256_max_epi16(_mm256_min_epi16(in10, max_val), min_val);
    in11 = _mm256_max_epi16(_mm256_min_epi16(in11, max_val), min_val);
    in12 = _mm256_max_epi16(_mm256_min_epi16(in12, max_val), min_val);
    in13 = _mm256_max_epi16(_mm256_min_epi16(in13, max_val), min_val);
    in14 = _mm256_max_epi16(_mm256_min_epi16(in14, max_val), min_val);
    in15 = _mm256_max_epi16(_mm256_min_epi16(in15, max_val), min_val);

    // store
    _mm256_storeu_si256((__m256i*)&dst[0 * 16 + 0], in00);
    _mm256_storeu_si256((__m256i*)&dst[1 * 16 + 0], in01);
    _mm256_storeu_si256((__m256i*)&dst[2 * 16 + 0], in02);
    _mm256_storeu_si256((__m256i*)&dst[3 * 16 + 0], in03);
    _mm256_storeu_si256((__m256i*)&dst[4 * 16 + 0], in04);
    _mm256_storeu_si256((__m256i*)&dst[5 * 16 + 0], in05);
    _mm256_storeu_si256((__m256i*)&dst[6 * 16 + 0], in06);
    _mm256_storeu_si256((__m256i*)&dst[7 * 16 + 0], in07);
    _mm256_storeu_si256((__m256i*)&dst[8 * 16 + 0], in08);
    _mm256_storeu_si256((__m256i*)&dst[9 * 16 + 0], in09);
    _mm256_storeu_si256((__m256i*)&dst[10 * 16 + 0], in10);
    _mm256_storeu_si256((__m256i*)&dst[11 * 16 + 0], in11);
    _mm256_storeu_si256((__m256i*)&dst[12 * 16 + 0], in12);
    _mm256_storeu_si256((__m256i*)&dst[13 * 16 + 0], in13);
    _mm256_storeu_si256((__m256i*)&dst[14 * 16 + 0], in14);
    _mm256_storeu_si256((__m256i*)&dst[15 * 16 + 0], in15);
}


void idct_c_32x32_avx2(const coeff_t *src, coeff_t *dst, int i_dst)
{
    int shift = 20 - g_bit_depth - (i_dst & 0x01);
    int clip = g_bit_depth + 1 + (i_dst & 0x01);
    int k, i;
    __m256i max_val, min_val;
    __m256i EEO0A, EEO1A, EEO2A, EEO3A, EEO0B, EEO1B, EEO2B, EEO3B;
    __m256i EEEO0A, EEEO0B, EEEO1A, EEEO1B;
    __m256i EEEE0A, EEEE0B, EEEE1A, EEEE1B;
    __m256i EEE0A, EEE0B, EEE1A, EEE1B, EEE3A, EEE3B, EEE2A, EEE2B;
    __m256i EE0A, EE0B, EE1A, EE1B, EE2A, EE2B, EE3A, EE3B, EE7A, EE7B, EE6A, EE6B, EE5A, EE5B, EE4A, EE4B;
    __m256i E0A, E0B, E1A, E1B, E2A, E2B, E3A, E3B, E4A, E4B, E5A, E5B, E6A, E6B, E7A, E7B, EFA, EFB, EEA, EEB, EDA, EDB, ECA, ECB, EBA, EBB, EAA, EAB, E9A, E9B, E8A, E8B;
    __m256i T10A, T10B, T11A, T11B, T12A, T12B, T13A, T13B, T14A, T14B, T15A, T15B, T16A, T16B, T17A, T17B, T18A, T18B, T19A, T19B, T1AA, T1AB, T1BA, T1BB, T1CA, T1CB, T1DA, T1DB, T1EA, T1EB, T1FA, T1FB;
    __m256i T2_00A, T2_00B, T2_01A, T2_01B, T2_02A, T2_02B, T2_03A, T2_03B, T2_04A, T2_04B, T2_05A, T2_05B, T2_06A, T2_06B, T2_07A, T2_07B, T2_08A, T2_08B, T2_09A, T2_09B, T2_10A, T2_10B, T2_11A, T2_11B, T2_12A, T2_12B, T2_13A, T2_13B, T2_14A, T2_14B, T2_15A, T2_15B, T2_31A, T2_31B, T2_30A, T2_30B, T2_29A, T2_29B, T2_28A, T2_28B, T2_27A, T2_27B, T2_26A, T2_26B, T2_25A, T2_25B, T2_24A, T2_24B, T2_23A, T2_23B, T2_22A, T2_22B, T2_21A, T2_21B, T2_20A, T2_20B, T2_19A, T2_19B, T2_18A, T2_18B, T2_17A, T2_17B, T2_16A, T2_16B;
    __m256i T3_00A, T3_00B, T3_01A, T3_01B, T3_02A, T3_02B, T3_03A, T3_03B, T3_04A, T3_04B, T3_05A, T3_05B, T3_06A, T3_06B, T3_07A, T3_07B, T3_08A, T3_08B, T3_09A, T3_09B, T3_10A, T3_10B, T3_11A, T3_11B, T3_12A, T3_12B, T3_13A, T3_13B, T3_14A, T3_14B, T3_15A, T3_15B;
    __m256i T3_16A, T3_16B, T3_17A, T3_17B, T3_18A, T3_18B, T3_19A, T3_19B, T3_20A, T3_20B, T3_21A, T3_21B, T3_22A, T3_22B, T3_23A, T3_23B, T3_24A, T3_24B, T3_25A, T3_25B, T3_26A, T3_26B, T3_27A, T3_27B, T3_28A, T3_28B, T3_29A, T3_29B, T3_30A, T3_30B, T3_31A, T3_31B;
    const __m256i c16_p45_p45 = _mm256_set1_epi32(0x002D002D);
    const __m256i c16_p43_p44 = _mm256_set1_epi32(0x002B002C);
    const __m256i c16_p39_p41 = _mm256_set1_epi32(0x00270029);
    const __m256i c16_p34_p36 = _mm256_set1_epi32(0x00220024);
    const __m256i c16_p27_p30 = _mm256_set1_epi32(0x001B001E);
    const __m256i c16_p19_p23 = _mm256_set1_epi32(0x00130017);
    const __m256i c16_p11_p15 = _mm256_set1_epi32(0x000B000F);
    const __m256i c16_p02_p07 = _mm256_set1_epi32(0x00020007);
    const __m256i c16_p41_p45 = _mm256_set1_epi32(0x0029002D);
    const __m256i c16_p23_p34 = _mm256_set1_epi32(0x00170022);
    const __m256i c16_n02_p11 = _mm256_set1_epi32(0xFFFE000B);
    const __m256i c16_n27_n15 = _mm256_set1_epi32(0xFFE5FFF1);
    const __m256i c16_n43_n36 = _mm256_set1_epi32(0xFFD5FFDC);
    const __m256i c16_n44_n45 = _mm256_set1_epi32(0xFFD4FFD3);
    const __m256i c16_n30_n39 = _mm256_set1_epi32(0xFFE2FFD9);
    const __m256i c16_n07_n19 = _mm256_set1_epi32(0xFFF9FFED);
    const __m256i c16_p34_p44 = _mm256_set1_epi32(0x0022002C);
    const __m256i c16_n07_p15 = _mm256_set1_epi32(0xFFF9000F);
    const __m256i c16_n41_n27 = _mm256_set1_epi32(0xFFD7FFE5);
    const __m256i c16_n39_n45 = _mm256_set1_epi32(0xFFD9FFD3);
    const __m256i c16_n02_n23 = _mm256_set1_epi32(0xFFFEFFE9);
    const __m256i c16_p36_p19 = _mm256_set1_epi32(0x00240013);
    const __m256i c16_p43_p45 = _mm256_set1_epi32(0x002B002D);
    const __m256i c16_p11_p30 = _mm256_set1_epi32(0x000B001E);
    const __m256i c16_p23_p43 = _mm256_set1_epi32(0x0017002B);
    const __m256i c16_n34_n07 = _mm256_set1_epi32(0xFFDEFFF9);
    const __m256i c16_n36_n45 = _mm256_set1_epi32(0xFFDCFFD3);
    const __m256i c16_p19_n11 = _mm256_set1_epi32(0x0013FFF5);
    const __m256i c16_p44_p41 = _mm256_set1_epi32(0x002C0029);
    const __m256i c16_n02_p27 = _mm256_set1_epi32(0xFFFE001B);
    const __m256i c16_n45_n30 = _mm256_set1_epi32(0xFFD3FFE2);
    const __m256i c16_n15_n39 = _mm256_set1_epi32(0xFFF1FFD9);
    const __m256i c16_p11_p41 = _mm256_set1_epi32(0x000B0029);
    const __m256i c16_n45_n27 = _mm256_set1_epi32(0xFFD3FFE5);
    const __m256i c16_p07_n30 = _mm256_set1_epi32(0x0007FFE2);
    const __m256i c16_p43_p39 = _mm256_set1_epi32(0x002B0027);
    const __m256i c16_n23_p15 = _mm256_set1_epi32(0xFFE9000F);
    const __m256i c16_n34_n45 = _mm256_set1_epi32(0xFFDEFFD3);
    const __m256i c16_p36_p02 = _mm256_set1_epi32(0x00240002);
    const __m256i c16_p19_p44 = _mm256_set1_epi32(0x0013002C);
    const __m256i c16_n02_p39 = _mm256_set1_epi32(0xFFFE0027);
    const __m256i c16_n36_n41 = _mm256_set1_epi32(0xFFDCFFD7);
    const __m256i c16_p43_p07 = _mm256_set1_epi32(0x002B0007);
    const __m256i c16_n11_p34 = _mm256_set1_epi32(0xFFF50022);
    const __m256i c16_n30_n44 = _mm256_set1_epi32(0xFFE2FFD4);
    const __m256i c16_p45_p15 = _mm256_set1_epi32(0x002D000F);
    const __m256i c16_n19_p27 = _mm256_set1_epi32(0xFFED001B);
    const __m256i c16_n23_n45 = _mm256_set1_epi32(0xFFE9FFD3);
    const __m256i c16_n15_p36 = _mm256_set1_epi32(0xFFF10024);
    const __m256i c16_n11_n45 = _mm256_set1_epi32(0xFFF5FFD3);
    const __m256i c16_p34_p39 = _mm256_set1_epi32(0x00220027);
    const __m256i c16_n45_n19 = _mm256_set1_epi32(0xFFD3FFED);
    const __m256i c16_p41_n07 = _mm256_set1_epi32(0x0029FFF9);
    const __m256i c16_n23_p30 = _mm256_set1_epi32(0xFFE9001E);
    const __m256i c16_n02_n44 = _mm256_set1_epi32(0xFFFEFFD4);
    const __m256i c16_p27_p43 = _mm256_set1_epi32(0x001B002B);
    const __m256i c16_n27_p34 = _mm256_set1_epi32(0xFFE50022);
    const __m256i c16_p19_n39 = _mm256_set1_epi32(0x0013FFD9);
    const __m256i c16_n11_p43 = _mm256_set1_epi32(0xFFF5002B);
    const __m256i c16_p02_n45 = _mm256_set1_epi32(0x0002FFD3);
    const __m256i c16_p07_p45 = _mm256_set1_epi32(0x0007002D);
    const __m256i c16_n15_n44 = _mm256_set1_epi32(0xFFF1FFD4);
    const __m256i c16_p23_p41 = _mm256_set1_epi32(0x00170029);
    const __m256i c16_n30_n36 = _mm256_set1_epi32(0xFFE2FFDC);
    const __m256i c16_n36_p30 = _mm256_set1_epi32(0xFFDC001E);
    const __m256i c16_p41_n23 = _mm256_set1_epi32(0x0029FFE9);
    const __m256i c16_n44_p15 = _mm256_set1_epi32(0xFFD4000F);
    const __m256i c16_p45_n07 = _mm256_set1_epi32(0x002DFFF9);
    const __m256i c16_n45_n02 = _mm256_set1_epi32(0xFFD3FFFE);
    const __m256i c16_p43_p11 = _mm256_set1_epi32(0x002B000B);
    const __m256i c16_n39_n19 = _mm256_set1_epi32(0xFFD9FFED);
    const __m256i c16_p34_p27 = _mm256_set1_epi32(0x0022001B);
    const __m256i c16_n43_p27 = _mm256_set1_epi32(0xFFD5001B);
    const __m256i c16_p44_n02 = _mm256_set1_epi32(0x002CFFFE);
    const __m256i c16_n30_n23 = _mm256_set1_epi32(0xFFE2FFE9);
    const __m256i c16_p07_p41 = _mm256_set1_epi32(0x00070029);
    const __m256i c16_p19_n45 = _mm256_set1_epi32(0x0013FFD3);
    const __m256i c16_n39_p34 = _mm256_set1_epi32(0xFFD90022);
    const __m256i c16_p45_n11 = _mm256_set1_epi32(0x002DFFF5);
    const __m256i c16_n36_n15 = _mm256_set1_epi32(0xFFDCFFF1);
    const __m256i c16_n45_p23 = _mm256_set1_epi32(0xFFD30017);
    const __m256i c16_p27_p19 = _mm256_set1_epi32(0x001B0013);
    const __m256i c16_p15_n45 = _mm256_set1_epi32(0x000FFFD3);
    const __m256i c16_n44_p30 = _mm256_set1_epi32(0xFFD4001E);
    const __m256i c16_p34_p11 = _mm256_set1_epi32(0x0022000B);
    const __m256i c16_p07_n43 = _mm256_set1_epi32(0x0007FFD5);
    const __m256i c16_n41_p36 = _mm256_set1_epi32(0xFFD70024);
    const __m256i c16_p39_p02 = _mm256_set1_epi32(0x00270002);
    const __m256i c16_n44_p19 = _mm256_set1_epi32(0xFFD40013);
    const __m256i c16_n02_p36 = _mm256_set1_epi32(0xFFFE0024);
    const __m256i c16_p45_n34 = _mm256_set1_epi32(0x002DFFDE);
    const __m256i c16_n15_n23 = _mm256_set1_epi32(0xFFF1FFE9);
    const __m256i c16_n39_p43 = _mm256_set1_epi32(0xFFD9002B);
    const __m256i c16_p30_p07 = _mm256_set1_epi32(0x001E0007);
    const __m256i c16_p27_n45 = _mm256_set1_epi32(0x001BFFD3);
    const __m256i c16_n41_p11 = _mm256_set1_epi32(0xFFD7000B);
    const __m256i c16_n39_p15 = _mm256_set1_epi32(0xFFD9000F);
    const __m256i c16_n30_p45 = _mm256_set1_epi32(0xFFE2002D);
    const __m256i c16_p27_p02 = _mm256_set1_epi32(0x001B0002);
    const __m256i c16_p41_n44 = _mm256_set1_epi32(0x0029FFD4);
    const __m256i c16_n11_n19 = _mm256_set1_epi32(0xFFF5FFED);
    const __m256i c16_n45_p36 = _mm256_set1_epi32(0xFFD30024);
    const __m256i c16_n07_p34 = _mm256_set1_epi32(0xFFF90022);
    const __m256i c16_p43_n23 = _mm256_set1_epi32(0x002BFFE9);
    const __m256i c16_n30_p11 = _mm256_set1_epi32(0xFFE2000B);
    const __m256i c16_n45_p43 = _mm256_set1_epi32(0xFFD3002B);
    const __m256i c16_n19_p36 = _mm256_set1_epi32(0xFFED0024);
    const __m256i c16_p23_n02 = _mm256_set1_epi32(0x0017FFFE);
    const __m256i c16_p45_n39 = _mm256_set1_epi32(0x002DFFD9);
    const __m256i c16_p27_n41 = _mm256_set1_epi32(0x001BFFD7);
    const __m256i c16_n15_n07 = _mm256_set1_epi32(0xFFF1FFF9);
    const __m256i c16_n44_p34 = _mm256_set1_epi32(0xFFD40022);
    const __m256i c16_n19_p07 = _mm256_set1_epi32(0xFFED0007);
    const __m256i c16_n39_p30 = _mm256_set1_epi32(0xFFD9001E);
    const __m256i c16_n45_p44 = _mm256_set1_epi32(0xFFD3002C);
    const __m256i c16_n36_p43 = _mm256_set1_epi32(0xFFDC002B);
    const __m256i c16_n15_p27 = _mm256_set1_epi32(0xFFF1001B);
    const __m256i c16_p11_p02 = _mm256_set1_epi32(0x000B0002);
    const __m256i c16_p34_n23 = _mm256_set1_epi32(0x0022FFE9);
    const __m256i c16_p45_n41 = _mm256_set1_epi32(0x002DFFD7);
    const __m256i c16_n07_p02 = _mm256_set1_epi32(0xFFF90002);
    const __m256i c16_n15_p11 = _mm256_set1_epi32(0xFFF1000B);
    const __m256i c16_n23_p19 = _mm256_set1_epi32(0xFFE90013);
    const __m256i c16_n30_p27 = _mm256_set1_epi32(0xFFE2001B);
    const __m256i c16_n36_p34 = _mm256_set1_epi32(0xFFDC0022);
    const __m256i c16_n41_p39 = _mm256_set1_epi32(0xFFD70027);
    const __m256i c16_n44_p43 = _mm256_set1_epi32(0xFFD4002B);
    const __m256i c16_n45_p45 = _mm256_set1_epi32(0xFFD3002D);

    //  const __m256i c16_p43_p45 = _mm256_set1_epi32(0x002B002D);
    const __m256i c16_p35_p40 = _mm256_set1_epi32(0x00230028);
    const __m256i c16_p21_p29 = _mm256_set1_epi32(0x0015001D);
    const __m256i c16_p04_p13 = _mm256_set1_epi32(0x0004000D);
    const __m256i c16_p29_p43 = _mm256_set1_epi32(0x001D002B);
    const __m256i c16_n21_p04 = _mm256_set1_epi32(0xFFEB0004);
    const __m256i c16_n45_n40 = _mm256_set1_epi32(0xFFD3FFD8);
    const __m256i c16_n13_n35 = _mm256_set1_epi32(0xFFF3FFDD);
    const __m256i c16_p04_p40 = _mm256_set1_epi32(0x00040028);
    const __m256i c16_n43_n35 = _mm256_set1_epi32(0xFFD5FFDD);
    const __m256i c16_p29_n13 = _mm256_set1_epi32(0x001DFFF3);
    const __m256i c16_p21_p45 = _mm256_set1_epi32(0x0015002D);
    const __m256i c16_n21_p35 = _mm256_set1_epi32(0xFFEB0023);
    const __m256i c16_p04_n43 = _mm256_set1_epi32(0x0004FFD5);
    const __m256i c16_p13_p45 = _mm256_set1_epi32(0x000D002D);
    const __m256i c16_n29_n40 = _mm256_set1_epi32(0xFFE3FFD8);
    const __m256i c16_n40_p29 = _mm256_set1_epi32(0xFFD8001D);
    const __m256i c16_p45_n13 = _mm256_set1_epi32(0x002DFFF3);
    const __m256i c16_n43_n04 = _mm256_set1_epi32(0xFFD5FFFC);
    const __m256i c16_p35_p21 = _mm256_set1_epi32(0x00230015);
    const __m256i c16_n45_p21 = _mm256_set1_epi32(0xFFD30015);
    const __m256i c16_p13_p29 = _mm256_set1_epi32(0x000D001D);
    const __m256i c16_p35_n43 = _mm256_set1_epi32(0x0023FFD5);
    const __m256i c16_n40_p04 = _mm256_set1_epi32(0xFFD80004);
    const __m256i c16_n35_p13 = _mm256_set1_epi32(0xFFDD000D);
    const __m256i c16_n40_p45 = _mm256_set1_epi32(0xFFD8002D);
    const __m256i c16_p04_p21 = _mm256_set1_epi32(0x00040015);
    const __m256i c16_p43_n29 = _mm256_set1_epi32(0x002BFFE3);
    const __m256i c16_n13_p04 = _mm256_set1_epi32(0xFFF30004);
    const __m256i c16_n29_p21 = _mm256_set1_epi32(0xFFE30015);
    const __m256i c16_n40_p35 = _mm256_set1_epi32(0xFFD80023);
    //const __m256i c16_n45_p43 = _mm256_set1_epi32(0xFFD3002B);

    const __m256i c16_p38_p44 = _mm256_set1_epi32(0x0026002C);
    const __m256i c16_p09_p25 = _mm256_set1_epi32(0x00090019);
    const __m256i c16_n09_p38 = _mm256_set1_epi32(0xFFF70026);
    const __m256i c16_n25_n44 = _mm256_set1_epi32(0xFFE7FFD4);

    const __m256i c16_n44_p25 = _mm256_set1_epi32(0xFFD40019);
    const __m256i c16_p38_p09 = _mm256_set1_epi32(0x00260009);
    const __m256i c16_n25_p09 = _mm256_set1_epi32(0xFFE70009);
    const __m256i c16_n44_p38 = _mm256_set1_epi32(0xFFD40026);

    const __m256i c16_p17_p42 = _mm256_set1_epi32(0x0011002A);
    const __m256i c16_n42_p17 = _mm256_set1_epi32(0xFFD60011);

    const __m256i c16_p32_p32 = _mm256_set1_epi32(0x00200020);
    const __m256i c16_n32_p32 = _mm256_set1_epi32(0xFFE00020);

    __m256i c32_rnd = _mm256_set1_epi32(16);
    int nShift = 5;

    // DCT1
    __m256i in00[2], in01[2], in02[2], in03[2], in04[2], in05[2], in06[2], in07[2], in08[2], in09[2], in10[2], in11[2], in12[2], in13[2], in14[2], in15[2];
    __m256i in16[2], in17[2], in18[2], in19[2], in20[2], in21[2], in22[2], in23[2], in24[2], in25[2], in26[2], in27[2], in28[2], in29[2], in30[2], in31[2];
    __m256i res00[2], res01[2], res02[2], res03[2], res04[2], res05[2], res06[2], res07[2], res08[2], res09[2], res10[2], res11[2], res12[2], res13[2], res14[2], res15[2];
    __m256i res16[2], res17[2], res18[2], res19[2], res20[2], res21[2], res22[2], res23[2], res24[2], res25[2], res26[2], res27[2], res28[2], res29[2], res30[2], res31[2];

    int pass, part;

    UNUSED_PARAMETER(i_dst);

    for (i = 0; i < 2; i++) {
        const int offset = (i << 4);
        in00[i] = _mm256_lddqu_si256((const __m256i*)&src[0 * 32 + offset]);
        in01[i] = _mm256_lddqu_si256((const __m256i*)&src[1 * 32 + offset]);
        in02[i] = _mm256_lddqu_si256((const __m256i*)&src[2 * 32 + offset]);
        in03[i] = _mm256_lddqu_si256((const __m256i*)&src[3 * 32 + offset]);
        in04[i] = _mm256_lddqu_si256((const __m256i*)&src[4 * 32 + offset]);
        in05[i] = _mm256_lddqu_si256((const __m256i*)&src[5 * 32 + offset]);
        in06[i] = _mm256_lddqu_si256((const __m256i*)&src[6 * 32 + offset]);
        in07[i] = _mm256_lddqu_si256((const __m256i*)&src[7 * 32 + offset]);
        in08[i] = _mm256_lddqu_si256((const __m256i*)&src[8 * 32 + offset]);
        in09[i] = _mm256_lddqu_si256((const __m256i*)&src[9 * 32 + offset]);
        in10[i] = _mm256_lddqu_si256((const __m256i*)&src[10 * 32 + offset]);
        in11[i] = _mm256_lddqu_si256((const __m256i*)&src[11 * 32 + offset]);
        in12[i] = _mm256_lddqu_si256((const __m256i*)&src[12 * 32 + offset]);
        in13[i] = _mm256_lddqu_si256((const __m256i*)&src[13 * 32 + offset]);
        in14[i] = _mm256_lddqu_si256((const __m256i*)&src[14 * 32 + offset]);
        in15[i] = _mm256_lddqu_si256((const __m256i*)&src[15 * 32 + offset]);
        in16[i] = _mm256_lddqu_si256((const __m256i*)&src[16 * 32 + offset]);
        in17[i] = _mm256_lddqu_si256((const __m256i*)&src[17 * 32 + offset]);
        in18[i] = _mm256_lddqu_si256((const __m256i*)&src[18 * 32 + offset]);
        in19[i] = _mm256_lddqu_si256((const __m256i*)&src[19 * 32 + offset]);
        in20[i] = _mm256_lddqu_si256((const __m256i*)&src[20 * 32 + offset]);
        in21[i] = _mm256_lddqu_si256((const __m256i*)&src[21 * 32 + offset]);
        in22[i] = _mm256_lddqu_si256((const __m256i*)&src[22 * 32 + offset]);
        in23[i] = _mm256_lddqu_si256((const __m256i*)&src[23 * 32 + offset]);
        in24[i] = _mm256_lddqu_si256((const __m256i*)&src[24 * 32 + offset]);
        in25[i] = _mm256_lddqu_si256((const __m256i*)&src[25 * 32 + offset]);
        in26[i] = _mm256_lddqu_si256((const __m256i*)&src[26 * 32 + offset]);
        in27[i] = _mm256_lddqu_si256((const __m256i*)&src[27 * 32 + offset]);
        in28[i] = _mm256_lddqu_si256((const __m256i*)&src[28 * 32 + offset]);
        in29[i] = _mm256_lddqu_si256((const __m256i*)&src[29 * 32 + offset]);
        in30[i] = _mm256_lddqu_si256((const __m256i*)&src[30 * 32 + offset]);
        in31[i] = _mm256_lddqu_si256((const __m256i*)&src[31 * 32 + offset]);
    }

    for (pass = 0; pass < 2; pass++) {
        for (part = 0; part < 2; part++) {
            const __m256i T_00_00A = _mm256_unpacklo_epi16(in01[part], in03[part]);       // [33 13 32 12 31 11 30 10]
            const __m256i T_00_00B = _mm256_unpackhi_epi16(in01[part], in03[part]);       // [37 17 36 16 35 15 34 14]
            const __m256i T_00_01A = _mm256_unpacklo_epi16(in05[part], in07[part]);       // [ ]
            const __m256i T_00_01B = _mm256_unpackhi_epi16(in05[part], in07[part]);       // [ ]
            const __m256i T_00_02A = _mm256_unpacklo_epi16(in09[part], in11[part]);       // [ ]
            const __m256i T_00_02B = _mm256_unpackhi_epi16(in09[part], in11[part]);       // [ ]
            const __m256i T_00_03A = _mm256_unpacklo_epi16(in13[part], in15[part]);       // [ ]
            const __m256i T_00_03B = _mm256_unpackhi_epi16(in13[part], in15[part]);       // [ ]
            const __m256i T_00_04A = _mm256_unpacklo_epi16(in17[part], in19[part]);       // [ ]
            const __m256i T_00_04B = _mm256_unpackhi_epi16(in17[part], in19[part]);       // [ ]
            const __m256i T_00_05A = _mm256_unpacklo_epi16(in21[part], in23[part]);       // [ ]
            const __m256i T_00_05B = _mm256_unpackhi_epi16(in21[part], in23[part]);       // [ ]
            const __m256i T_00_06A = _mm256_unpacklo_epi16(in25[part], in27[part]);       // [ ]
            const __m256i T_00_06B = _mm256_unpackhi_epi16(in25[part], in27[part]);       // [ ]
            const __m256i T_00_07A = _mm256_unpacklo_epi16(in29[part], in31[part]);       //
            const __m256i T_00_07B = _mm256_unpackhi_epi16(in29[part], in31[part]);       // [ ]

            const __m256i T_00_08A = _mm256_unpacklo_epi16(in02[part], in06[part]);       // [ ]
            const __m256i T_00_08B = _mm256_unpackhi_epi16(in02[part], in06[part]);       // [ ]
            const __m256i T_00_09A = _mm256_unpacklo_epi16(in10[part], in14[part]);       // [ ]
            const __m256i T_00_09B = _mm256_unpackhi_epi16(in10[part], in14[part]);       // [ ]
            const __m256i T_00_10A = _mm256_unpacklo_epi16(in18[part], in22[part]);       // [ ]
            const __m256i T_00_10B = _mm256_unpackhi_epi16(in18[part], in22[part]);       // [ ]
            const __m256i T_00_11A = _mm256_unpacklo_epi16(in26[part], in30[part]);       // [ ]
            const __m256i T_00_11B = _mm256_unpackhi_epi16(in26[part], in30[part]);       // [ ]

            const __m256i T_00_12A = _mm256_unpacklo_epi16(in04[part], in12[part]);       // [ ]
            const __m256i T_00_12B = _mm256_unpackhi_epi16(in04[part], in12[part]);       // [ ]
            const __m256i T_00_13A = _mm256_unpacklo_epi16(in20[part], in28[part]);       // [ ]
            const __m256i T_00_13B = _mm256_unpackhi_epi16(in20[part], in28[part]);       // [ ]

            const __m256i T_00_14A = _mm256_unpacklo_epi16(in08[part], in24[part]);       //
            const __m256i T_00_14B = _mm256_unpackhi_epi16(in08[part], in24[part]);       // [ ]
            const __m256i T_00_15A = _mm256_unpacklo_epi16(in00[part], in16[part]);       //
            const __m256i T_00_15B = _mm256_unpackhi_epi16(in00[part], in16[part]);       // [ ]

            __m256i O00A, O01A, O02A, O03A, O04A, O05A, O06A, O07A, O08A, O09A, O10A, O11A, O12A, O13A, O14A, O15A;
            __m256i O00B, O01B, O02B, O03B, O04B, O05B, O06B, O07B, O08B, O09B, O10B, O11B, O12B, O13B, O14B, O15B;
            __m256i EO0A, EO1A, EO2A, EO3A, EO4A, EO5A, EO6A, EO7A;
            __m256i EO0B, EO1B, EO2B, EO3B, EO4B, EO5B, EO6B, EO7B;
            {
                __m256i T00, T01, T02, T03;
#define     COMPUTE_ROW(r0103, r0507, r0911, r1315, r1719, r2123, r2527, r2931, c0103, c0507, c0911, c1315, c1719, c2123, c2527, c2931, row) \
            T00 = _mm256_add_epi32(_mm256_madd_epi16(r0103, c0103), _mm256_madd_epi16(r0507, c0507)); \
            T01 = _mm256_add_epi32(_mm256_madd_epi16(r0911, c0911), _mm256_madd_epi16(r1315, c1315)); \
            T02 = _mm256_add_epi32(_mm256_madd_epi16(r1719, c1719), _mm256_madd_epi16(r2123, c2123)); \
            T03 = _mm256_add_epi32(_mm256_madd_epi16(r2527, c2527), _mm256_madd_epi16(r2931, c2931)); \
            row = _mm256_add_epi32(_mm256_add_epi32(T00, T01), _mm256_add_epi32(T02, T03));

                COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                            c16_p45_p45, c16_p43_p44, c16_p39_p41, c16_p34_p36, c16_p27_p30, c16_p19_p23, c16_p11_p15, c16_p02_p07, O00A)
                COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                            c16_p41_p45, c16_p23_p34, c16_n02_p11, c16_n27_n15, c16_n43_n36, c16_n44_n45, c16_n30_n39, c16_n07_n19, O01A)
                COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                            c16_p34_p44, c16_n07_p15, c16_n41_n27, c16_n39_n45, c16_n02_n23, c16_p36_p19, c16_p43_p45, c16_p11_p30, O02A)
                COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                            c16_p23_p43, c16_n34_n07, c16_n36_n45, c16_p19_n11, c16_p44_p41, c16_n02_p27, c16_n45_n30, c16_n15_n39, O03A)
                COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                            c16_p11_p41, c16_n45_n27, c16_p07_n30, c16_p43_p39, c16_n23_p15, c16_n34_n45, c16_p36_p02, c16_p19_p44, O04A)
                COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                            c16_n02_p39, c16_n36_n41, c16_p43_p07, c16_n11_p34, c16_n30_n44, c16_p45_p15, c16_n19_p27, c16_n23_n45, O05A)
                COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                            c16_n15_p36, c16_n11_n45, c16_p34_p39, c16_n45_n19, c16_p41_n07, c16_n23_p30, c16_n02_n44, c16_p27_p43, O06A)
                COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                            c16_n27_p34, c16_p19_n39, c16_n11_p43, c16_p02_n45, c16_p07_p45, c16_n15_n44, c16_p23_p41, c16_n30_n36, O07A)
                COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                            c16_n36_p30, c16_p41_n23, c16_n44_p15, c16_p45_n07, c16_n45_n02, c16_p43_p11, c16_n39_n19, c16_p34_p27, O08A)
                COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                            c16_n43_p27, c16_p44_n02, c16_n30_n23, c16_p07_p41, c16_p19_n45, c16_n39_p34, c16_p45_n11, c16_n36_n15, O09A)
                COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                            c16_n45_p23, c16_p27_p19, c16_p15_n45, c16_n44_p30, c16_p34_p11, c16_p07_n43, c16_n41_p36, c16_p39_p02, O10A)
                COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                            c16_n44_p19, c16_n02_p36, c16_p45_n34, c16_n15_n23, c16_n39_p43, c16_p30_p07, c16_p27_n45, c16_n41_p11, O11A)
                COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                            c16_n39_p15, c16_n30_p45, c16_p27_p02, c16_p41_n44, c16_n11_n19, c16_n45_p36, c16_n07_p34, c16_p43_n23, O12A)
                COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                            c16_n30_p11, c16_n45_p43, c16_n19_p36, c16_p23_n02, c16_p45_n39, c16_p27_n41, c16_n15_n07, c16_n44_p34, O13A)
                COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                            c16_n19_p07, c16_n39_p30, c16_n45_p44, c16_n36_p43, c16_n15_p27, c16_p11_p02, c16_p34_n23, c16_p45_n41, O14A)
                COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                            c16_n07_p02, c16_n15_p11, c16_n23_p19, c16_n30_p27, c16_n36_p34, c16_n41_p39, c16_n44_p43, c16_n45_p45, O15A)

                COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                            c16_p45_p45, c16_p43_p44, c16_p39_p41, c16_p34_p36, c16_p27_p30, c16_p19_p23, c16_p11_p15, c16_p02_p07, O00B)
                COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                            c16_p41_p45, c16_p23_p34, c16_n02_p11, c16_n27_n15, c16_n43_n36, c16_n44_n45, c16_n30_n39, c16_n07_n19, O01B)
                COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                            c16_p34_p44, c16_n07_p15, c16_n41_n27, c16_n39_n45, c16_n02_n23, c16_p36_p19, c16_p43_p45, c16_p11_p30, O02B)
                COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                            c16_p23_p43, c16_n34_n07, c16_n36_n45, c16_p19_n11, c16_p44_p41, c16_n02_p27, c16_n45_n30, c16_n15_n39, O03B)
                COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                            c16_p11_p41, c16_n45_n27, c16_p07_n30, c16_p43_p39, c16_n23_p15, c16_n34_n45, c16_p36_p02, c16_p19_p44, O04B)
                COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                            c16_n02_p39, c16_n36_n41, c16_p43_p07, c16_n11_p34, c16_n30_n44, c16_p45_p15, c16_n19_p27, c16_n23_n45, O05B)
                COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                            c16_n15_p36, c16_n11_n45, c16_p34_p39, c16_n45_n19, c16_p41_n07, c16_n23_p30, c16_n02_n44, c16_p27_p43, O06B)
                COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                            c16_n27_p34, c16_p19_n39, c16_n11_p43, c16_p02_n45, c16_p07_p45, c16_n15_n44, c16_p23_p41, c16_n30_n36, O07B)
                COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                            c16_n36_p30, c16_p41_n23, c16_n44_p15, c16_p45_n07, c16_n45_n02, c16_p43_p11, c16_n39_n19, c16_p34_p27, O08B)
                COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                            c16_n43_p27, c16_p44_n02, c16_n30_n23, c16_p07_p41, c16_p19_n45, c16_n39_p34, c16_p45_n11, c16_n36_n15, O09B)
                COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                            c16_n45_p23, c16_p27_p19, c16_p15_n45, c16_n44_p30, c16_p34_p11, c16_p07_n43, c16_n41_p36, c16_p39_p02, O10B)
                COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                            c16_n44_p19, c16_n02_p36, c16_p45_n34, c16_n15_n23, c16_n39_p43, c16_p30_p07, c16_p27_n45, c16_n41_p11, O11B)
                COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                            c16_n39_p15, c16_n30_p45, c16_p27_p02, c16_p41_n44, c16_n11_n19, c16_n45_p36, c16_n07_p34, c16_p43_n23, O12B)
                COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                            c16_n30_p11, c16_n45_p43, c16_n19_p36, c16_p23_n02, c16_p45_n39, c16_p27_n41, c16_n15_n07, c16_n44_p34, O13B)
                COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                            c16_n19_p07, c16_n39_p30, c16_n45_p44, c16_n36_p43, c16_n15_p27, c16_p11_p02, c16_p34_n23, c16_p45_n41, O14B)
                COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                            c16_n07_p02, c16_n15_p11, c16_n23_p19, c16_n30_p27, c16_n36_p34, c16_n41_p39, c16_n44_p43, c16_n45_p45, O15B)

#undef      COMPUTE_ROW
            }


            {
                __m256i T00, T01;
#define     COMPUTE_ROW(row0206, row1014, row1822, row2630, c0206, c1014, c1822, c2630, row) \
            T00 = _mm256_add_epi32(_mm256_madd_epi16(row0206, c0206), _mm256_madd_epi16(row1014, c1014)); \
            T01 = _mm256_add_epi32(_mm256_madd_epi16(row1822, c1822), _mm256_madd_epi16(row2630, c2630)); \
            row = _mm256_add_epi32(T00, T01);

                COMPUTE_ROW(T_00_08A, T_00_09A, T_00_10A, T_00_11A, c16_p43_p45, c16_p35_p40, c16_p21_p29, c16_p04_p13, EO0A)
                COMPUTE_ROW(T_00_08A, T_00_09A, T_00_10A, T_00_11A, c16_p29_p43, c16_n21_p04, c16_n45_n40, c16_n13_n35, EO1A)
                COMPUTE_ROW(T_00_08A, T_00_09A, T_00_10A, T_00_11A, c16_p04_p40, c16_n43_n35, c16_p29_n13, c16_p21_p45, EO2A)
                COMPUTE_ROW(T_00_08A, T_00_09A, T_00_10A, T_00_11A, c16_n21_p35, c16_p04_n43, c16_p13_p45, c16_n29_n40, EO3A)
                COMPUTE_ROW(T_00_08A, T_00_09A, T_00_10A, T_00_11A, c16_n40_p29, c16_p45_n13, c16_n43_n04, c16_p35_p21, EO4A)
                COMPUTE_ROW(T_00_08A, T_00_09A, T_00_10A, T_00_11A, c16_n45_p21, c16_p13_p29, c16_p35_n43, c16_n40_p04, EO5A)
                COMPUTE_ROW(T_00_08A, T_00_09A, T_00_10A, T_00_11A, c16_n35_p13, c16_n40_p45, c16_p04_p21, c16_p43_n29, EO6A)
                COMPUTE_ROW(T_00_08A, T_00_09A, T_00_10A, T_00_11A, c16_n13_p04, c16_n29_p21, c16_n40_p35, c16_n45_p43, EO7A)

                COMPUTE_ROW(T_00_08B, T_00_09B, T_00_10B, T_00_11B, c16_p43_p45, c16_p35_p40, c16_p21_p29, c16_p04_p13, EO0B)
                COMPUTE_ROW(T_00_08B, T_00_09B, T_00_10B, T_00_11B, c16_p29_p43, c16_n21_p04, c16_n45_n40, c16_n13_n35, EO1B)
                COMPUTE_ROW(T_00_08B, T_00_09B, T_00_10B, T_00_11B, c16_p04_p40, c16_n43_n35, c16_p29_n13, c16_p21_p45, EO2B)
                COMPUTE_ROW(T_00_08B, T_00_09B, T_00_10B, T_00_11B, c16_n21_p35, c16_p04_n43, c16_p13_p45, c16_n29_n40, EO3B)
                COMPUTE_ROW(T_00_08B, T_00_09B, T_00_10B, T_00_11B, c16_n40_p29, c16_p45_n13, c16_n43_n04, c16_p35_p21, EO4B)
                COMPUTE_ROW(T_00_08B, T_00_09B, T_00_10B, T_00_11B, c16_n45_p21, c16_p13_p29, c16_p35_n43, c16_n40_p04, EO5B)
                COMPUTE_ROW(T_00_08B, T_00_09B, T_00_10B, T_00_11B, c16_n35_p13, c16_n40_p45, c16_p04_p21, c16_p43_n29, EO6B)
                COMPUTE_ROW(T_00_08B, T_00_09B, T_00_10B, T_00_11B, c16_n13_p04, c16_n29_p21, c16_n40_p35, c16_n45_p43, EO7B)
#undef      COMPUTE_ROW
            }

            EEO0A = _mm256_add_epi32(_mm256_madd_epi16(T_00_12A, c16_p38_p44), _mm256_madd_epi16(T_00_13A, c16_p09_p25));
            EEO1A = _mm256_add_epi32(_mm256_madd_epi16(T_00_12A, c16_n09_p38), _mm256_madd_epi16(T_00_13A, c16_n25_n44));
            EEO2A = _mm256_add_epi32(_mm256_madd_epi16(T_00_12A, c16_n44_p25), _mm256_madd_epi16(T_00_13A, c16_p38_p09));
            EEO3A = _mm256_add_epi32(_mm256_madd_epi16(T_00_12A, c16_n25_p09), _mm256_madd_epi16(T_00_13A, c16_n44_p38));
            EEO0B = _mm256_add_epi32(_mm256_madd_epi16(T_00_12B, c16_p38_p44), _mm256_madd_epi16(T_00_13B, c16_p09_p25));
            EEO1B = _mm256_add_epi32(_mm256_madd_epi16(T_00_12B, c16_n09_p38), _mm256_madd_epi16(T_00_13B, c16_n25_n44));
            EEO2B = _mm256_add_epi32(_mm256_madd_epi16(T_00_12B, c16_n44_p25), _mm256_madd_epi16(T_00_13B, c16_p38_p09));
            EEO3B = _mm256_add_epi32(_mm256_madd_epi16(T_00_12B, c16_n25_p09), _mm256_madd_epi16(T_00_13B, c16_n44_p38));

            EEEO0A = _mm256_madd_epi16(T_00_14A, c16_p17_p42);
            EEEO0B = _mm256_madd_epi16(T_00_14B, c16_p17_p42);
            EEEO1A = _mm256_madd_epi16(T_00_14A, c16_n42_p17);
            EEEO1B = _mm256_madd_epi16(T_00_14B, c16_n42_p17);

            EEEE0A = _mm256_madd_epi16(T_00_15A, c16_p32_p32);
            EEEE0B = _mm256_madd_epi16(T_00_15B, c16_p32_p32);
            EEEE1A = _mm256_madd_epi16(T_00_15A, c16_n32_p32);
            EEEE1B = _mm256_madd_epi16(T_00_15B, c16_n32_p32);

            EEE0A = _mm256_add_epi32(EEEE0A, EEEO0A);          // EEE0 = EEEE0 + EEEO0
            EEE0B = _mm256_add_epi32(EEEE0B, EEEO0B);
            EEE1A = _mm256_add_epi32(EEEE1A, EEEO1A);          // EEE1 = EEEE1 + EEEO1
            EEE1B = _mm256_add_epi32(EEEE1B, EEEO1B);
            EEE3A = _mm256_sub_epi32(EEEE0A, EEEO0A);          // EEE2 = EEEE0 - EEEO0
            EEE3B = _mm256_sub_epi32(EEEE0B, EEEO0B);
            EEE2A = _mm256_sub_epi32(EEEE1A, EEEO1A);          // EEE3 = EEEE1 - EEEO1
            EEE2B = _mm256_sub_epi32(EEEE1B, EEEO1B);

            EE0A = _mm256_add_epi32(EEE0A, EEO0A);          // EE0 = EEE0 + EEO0
            EE0B = _mm256_add_epi32(EEE0B, EEO0B);
            EE1A = _mm256_add_epi32(EEE1A, EEO1A);          // EE1 = EEE1 + EEO1
            EE1B = _mm256_add_epi32(EEE1B, EEO1B);
            EE2A = _mm256_add_epi32(EEE2A, EEO2A);          // EE2 = EEE0 + EEO0
            EE2B = _mm256_add_epi32(EEE2B, EEO2B);
            EE3A = _mm256_add_epi32(EEE3A, EEO3A);          // EE3 = EEE1 + EEO1
            EE3B = _mm256_add_epi32(EEE3B, EEO3B);
            EE7A = _mm256_sub_epi32(EEE0A, EEO0A);          // EE7 = EEE0 - EEO0
            EE7B = _mm256_sub_epi32(EEE0B, EEO0B);
            EE6A = _mm256_sub_epi32(EEE1A, EEO1A);          // EE6 = EEE1 - EEO1
            EE6B = _mm256_sub_epi32(EEE1B, EEO1B);
            EE5A = _mm256_sub_epi32(EEE2A, EEO2A);          // EE5 = EEE0 - EEO0
            EE5B = _mm256_sub_epi32(EEE2B, EEO2B);
            EE4A = _mm256_sub_epi32(EEE3A, EEO3A);          // EE4 = EEE1 - EEO1
            EE4B = _mm256_sub_epi32(EEE3B, EEO3B);

            E0A = _mm256_add_epi32(EE0A, EO0A);          // E0 = EE0 + EO0
            E0B = _mm256_add_epi32(EE0B, EO0B);
            E1A = _mm256_add_epi32(EE1A, EO1A);          // E1 = EE1 + EO1
            E1B = _mm256_add_epi32(EE1B, EO1B);
            E2A = _mm256_add_epi32(EE2A, EO2A);          // E2 = EE2 + EO2
            E2B = _mm256_add_epi32(EE2B, EO2B);
            E3A = _mm256_add_epi32(EE3A, EO3A);          // E3 = EE3 + EO3
            E3B = _mm256_add_epi32(EE3B, EO3B);
            E4A = _mm256_add_epi32(EE4A, EO4A);          // E4 =
            E4B = _mm256_add_epi32(EE4B, EO4B);
            E5A = _mm256_add_epi32(EE5A, EO5A);          // E5 =
            E5B = _mm256_add_epi32(EE5B, EO5B);
            E6A = _mm256_add_epi32(EE6A, EO6A);          // E6 =
            E6B = _mm256_add_epi32(EE6B, EO6B);
            E7A = _mm256_add_epi32(EE7A, EO7A);          // E7 =
            E7B = _mm256_add_epi32(EE7B, EO7B);
            EFA = _mm256_sub_epi32(EE0A, EO0A);          // EF = EE0 - EO0
            EFB = _mm256_sub_epi32(EE0B, EO0B);
            EEA = _mm256_sub_epi32(EE1A, EO1A);          // EE = EE1 - EO1
            EEB = _mm256_sub_epi32(EE1B, EO1B);
            EDA = _mm256_sub_epi32(EE2A, EO2A);          // ED = EE2 - EO2
            EDB = _mm256_sub_epi32(EE2B, EO2B);
            ECA = _mm256_sub_epi32(EE3A, EO3A);          // EC = EE3 - EO3
            ECB = _mm256_sub_epi32(EE3B, EO3B);
            EBA = _mm256_sub_epi32(EE4A, EO4A);          // EB =
            EBB = _mm256_sub_epi32(EE4B, EO4B);
            EAA = _mm256_sub_epi32(EE5A, EO5A);          // EA =
            EAB = _mm256_sub_epi32(EE5B, EO5B);
            E9A = _mm256_sub_epi32(EE6A, EO6A);          // E9 =
            E9B = _mm256_sub_epi32(EE6B, EO6B);
            E8A = _mm256_sub_epi32(EE7A, EO7A);          // E8 =
            E8B = _mm256_sub_epi32(EE7B, EO7B);

            T10A = _mm256_add_epi32(E0A, c32_rnd);         // E0 + rnd
            T10B = _mm256_add_epi32(E0B, c32_rnd);
            T11A = _mm256_add_epi32(E1A, c32_rnd);         // E1 + rnd
            T11B = _mm256_add_epi32(E1B, c32_rnd);
            T12A = _mm256_add_epi32(E2A, c32_rnd);         // E2 + rnd
            T12B = _mm256_add_epi32(E2B, c32_rnd);
            T13A = _mm256_add_epi32(E3A, c32_rnd);         // E3 + rnd
            T13B = _mm256_add_epi32(E3B, c32_rnd);
            T14A = _mm256_add_epi32(E4A, c32_rnd);         // E4 + rnd
            T14B = _mm256_add_epi32(E4B, c32_rnd);
            T15A = _mm256_add_epi32(E5A, c32_rnd);         // E5 + rnd
            T15B = _mm256_add_epi32(E5B, c32_rnd);
            T16A = _mm256_add_epi32(E6A, c32_rnd);         // E6 + rnd
            T16B = _mm256_add_epi32(E6B, c32_rnd);
            T17A = _mm256_add_epi32(E7A, c32_rnd);         // E7 + rnd
            T17B = _mm256_add_epi32(E7B, c32_rnd);
            T18A = _mm256_add_epi32(E8A, c32_rnd);         // E8 + rnd
            T18B = _mm256_add_epi32(E8B, c32_rnd);
            T19A = _mm256_add_epi32(E9A, c32_rnd);         // E9 + rnd
            T19B = _mm256_add_epi32(E9B, c32_rnd);
            T1AA = _mm256_add_epi32(EAA, c32_rnd);         // E10 + rnd
            T1AB = _mm256_add_epi32(EAB, c32_rnd);
            T1BA = _mm256_add_epi32(EBA, c32_rnd);         // E11 + rnd
            T1BB = _mm256_add_epi32(EBB, c32_rnd);
            T1CA = _mm256_add_epi32(ECA, c32_rnd);         // E12 + rnd
            T1CB = _mm256_add_epi32(ECB, c32_rnd);
            T1DA = _mm256_add_epi32(EDA, c32_rnd);         // E13 + rnd
            T1DB = _mm256_add_epi32(EDB, c32_rnd);
            T1EA = _mm256_add_epi32(EEA, c32_rnd);         // E14 + rnd
            T1EB = _mm256_add_epi32(EEB, c32_rnd);
            T1FA = _mm256_add_epi32(EFA, c32_rnd);         // E15 + rnd
            T1FB = _mm256_add_epi32(EFB, c32_rnd);

            T2_00A = _mm256_add_epi32(T10A, O00A);          // E0 + O0 + rnd
            T2_00B = _mm256_add_epi32(T10B, O00B);
            T2_01A = _mm256_add_epi32(T11A, O01A);          // E1 + O1 + rnd
            T2_01B = _mm256_add_epi32(T11B, O01B);
            T2_02A = _mm256_add_epi32(T12A, O02A);          // E2 + O2 + rnd
            T2_02B = _mm256_add_epi32(T12B, O02B);
            T2_03A = _mm256_add_epi32(T13A, O03A);          // E3 + O3 + rnd
            T2_03B = _mm256_add_epi32(T13B, O03B);
            T2_04A = _mm256_add_epi32(T14A, O04A);          // E4
            T2_04B = _mm256_add_epi32(T14B, O04B);
            T2_05A = _mm256_add_epi32(T15A, O05A);          // E5
            T2_05B = _mm256_add_epi32(T15B, O05B);
            T2_06A = _mm256_add_epi32(T16A, O06A);          // E6
            T2_06B = _mm256_add_epi32(T16B, O06B);
            T2_07A = _mm256_add_epi32(T17A, O07A);          // E7
            T2_07B = _mm256_add_epi32(T17B, O07B);
            T2_08A = _mm256_add_epi32(T18A, O08A);          // E8
            T2_08B = _mm256_add_epi32(T18B, O08B);
            T2_09A = _mm256_add_epi32(T19A, O09A);          // E9
            T2_09B = _mm256_add_epi32(T19B, O09B);
            T2_10A = _mm256_add_epi32(T1AA, O10A);          // E10
            T2_10B = _mm256_add_epi32(T1AB, O10B);
            T2_11A = _mm256_add_epi32(T1BA, O11A);          // E11
            T2_11B = _mm256_add_epi32(T1BB, O11B);
            T2_12A = _mm256_add_epi32(T1CA, O12A);          // E12
            T2_12B = _mm256_add_epi32(T1CB, O12B);
            T2_13A = _mm256_add_epi32(T1DA, O13A);          // E13
            T2_13B = _mm256_add_epi32(T1DB, O13B);
            T2_14A = _mm256_add_epi32(T1EA, O14A);          // E14
            T2_14B = _mm256_add_epi32(T1EB, O14B);
            T2_15A = _mm256_add_epi32(T1FA, O15A);          // E15
            T2_15B = _mm256_add_epi32(T1FB, O15B);
            T2_31A = _mm256_sub_epi32(T10A, O00A);          // E0 - O0 + rnd
            T2_31B = _mm256_sub_epi32(T10B, O00B);
            T2_30A = _mm256_sub_epi32(T11A, O01A);          // E1 - O1 + rnd
            T2_30B = _mm256_sub_epi32(T11B, O01B);
            T2_29A = _mm256_sub_epi32(T12A, O02A);          // E2 - O2 + rnd
            T2_29B = _mm256_sub_epi32(T12B, O02B);
            T2_28A = _mm256_sub_epi32(T13A, O03A);          // E3 - O3 + rnd
            T2_28B = _mm256_sub_epi32(T13B, O03B);
            T2_27A = _mm256_sub_epi32(T14A, O04A);          // E4
            T2_27B = _mm256_sub_epi32(T14B, O04B);
            T2_26A = _mm256_sub_epi32(T15A, O05A);          // E5
            T2_26B = _mm256_sub_epi32(T15B, O05B);
            T2_25A = _mm256_sub_epi32(T16A, O06A);          // E6
            T2_25B = _mm256_sub_epi32(T16B, O06B);
            T2_24A = _mm256_sub_epi32(T17A, O07A);          // E7
            T2_24B = _mm256_sub_epi32(T17B, O07B);
            T2_23A = _mm256_sub_epi32(T18A, O08A);          //
            T2_23B = _mm256_sub_epi32(T18B, O08B);
            T2_22A = _mm256_sub_epi32(T19A, O09A);          //
            T2_22B = _mm256_sub_epi32(T19B, O09B);
            T2_21A = _mm256_sub_epi32(T1AA, O10A);          //
            T2_21B = _mm256_sub_epi32(T1AB, O10B);
            T2_20A = _mm256_sub_epi32(T1BA, O11A);          //
            T2_20B = _mm256_sub_epi32(T1BB, O11B);
            T2_19A = _mm256_sub_epi32(T1CA, O12A);          //
            T2_19B = _mm256_sub_epi32(T1CB, O12B);
            T2_18A = _mm256_sub_epi32(T1DA, O13A);          //
            T2_18B = _mm256_sub_epi32(T1DB, O13B);
            T2_17A = _mm256_sub_epi32(T1EA, O14A);          //
            T2_17B = _mm256_sub_epi32(T1EB, O14B);
            T2_16A = _mm256_sub_epi32(T1FA, O15A);          //
            T2_16B = _mm256_sub_epi32(T1FB, O15B);

            T3_00A = _mm256_srai_epi32(T2_00A, nShift);             // [30 20 10 00] // This operation make it much slower than 128
            T3_00B = _mm256_srai_epi32(T2_00B, nShift);             // [70 60 50 40] // This operation make it much slower than 128
            T3_01A = _mm256_srai_epi32(T2_01A, nShift);             // [31 21 11 01] // This operation make it much slower than 128
            T3_01B = _mm256_srai_epi32(T2_01B, nShift);             // [71 61 51 41] // This operation make it much slower than 128
            T3_02A = _mm256_srai_epi32(T2_02A, nShift);             // [32 22 12 02] // This operation make it much slower than 128
            T3_02B = _mm256_srai_epi32(T2_02B, nShift);             // [72 62 52 42]
            T3_03A = _mm256_srai_epi32(T2_03A, nShift);             // [33 23 13 03]
            T3_03B = _mm256_srai_epi32(T2_03B, nShift);             // [73 63 53 43]
            T3_04A = _mm256_srai_epi32(T2_04A, nShift);             // [33 24 14 04]
            T3_04B = _mm256_srai_epi32(T2_04B, nShift);             // [74 64 54 44]
            T3_05A = _mm256_srai_epi32(T2_05A, nShift);             // [35 25 15 05]
            T3_05B = _mm256_srai_epi32(T2_05B, nShift);             // [75 65 55 45]
            T3_06A = _mm256_srai_epi32(T2_06A, nShift);             // [36 26 16 06]
            T3_06B = _mm256_srai_epi32(T2_06B, nShift);             // [76 66 56 46]
            T3_07A = _mm256_srai_epi32(T2_07A, nShift);             // [37 27 17 07]
            T3_07B = _mm256_srai_epi32(T2_07B, nShift);             // [77 67 57 47]
            T3_08A = _mm256_srai_epi32(T2_08A, nShift);             // [30 20 10 00] x8
            T3_08B = _mm256_srai_epi32(T2_08B, nShift);             // [70 60 50 40]
            T3_09A = _mm256_srai_epi32(T2_09A, nShift);             // [31 21 11 01] x9
            T3_09B = _mm256_srai_epi32(T2_09B, nShift);             // [71 61 51 41]
            T3_10A = _mm256_srai_epi32(T2_10A, nShift);             // [32 22 12 02] xA
            T3_10B = _mm256_srai_epi32(T2_10B, nShift);             // [72 62 52 42]
            T3_11A = _mm256_srai_epi32(T2_11A, nShift);             // [33 23 13 03] xB
            T3_11B = _mm256_srai_epi32(T2_11B, nShift);             // [73 63 53 43]
            T3_12A = _mm256_srai_epi32(T2_12A, nShift);             // [33 24 14 04] xC
            T3_12B = _mm256_srai_epi32(T2_12B, nShift);             // [74 64 54 44]
            T3_13A = _mm256_srai_epi32(T2_13A, nShift);             // [35 25 15 05] xD
            T3_13B = _mm256_srai_epi32(T2_13B, nShift);             // [75 65 55 45]
            T3_14A = _mm256_srai_epi32(T2_14A, nShift);             // [36 26 16 06] xE
            T3_14B = _mm256_srai_epi32(T2_14B, nShift);             // [76 66 56 46]
            T3_15A = _mm256_srai_epi32(T2_15A, nShift);             // [37 27 17 07] xF
            T3_15B = _mm256_srai_epi32(T2_15B, nShift);             // [77 67 57 47]

            T3_16A = _mm256_srai_epi32(T2_16A, nShift);             // [30 20 10 00] // This operation make it much slower than 128
            T3_16B = _mm256_srai_epi32(T2_16B, nShift);             // [70 60 50 40] // This operation make it much slower than 128
            T3_17A = _mm256_srai_epi32(T2_17A, nShift);             // [31 21 11 01] // This operation make it much slower than 128
            T3_17B = _mm256_srai_epi32(T2_17B, nShift);             // [71 61 51 41]
            T3_18A = _mm256_srai_epi32(T2_18A, nShift);             // [32 22 12 02]
            T3_18B = _mm256_srai_epi32(T2_18B, nShift);             // [72 62 52 42]
            T3_19A = _mm256_srai_epi32(T2_19A, nShift);             // [33 23 13 03]
            T3_19B = _mm256_srai_epi32(T2_19B, nShift);             // [73 63 53 43]
            T3_20A = _mm256_srai_epi32(T2_20A, nShift);             // [33 24 14 04]
            T3_20B = _mm256_srai_epi32(T2_20B, nShift);             // [74 64 54 44]
            T3_21A = _mm256_srai_epi32(T2_21A, nShift);             // [35 25 15 05]
            T3_21B = _mm256_srai_epi32(T2_21B, nShift);             // [75 65 55 45]
            T3_22A = _mm256_srai_epi32(T2_22A, nShift);             // [36 26 16 06]
            T3_22B = _mm256_srai_epi32(T2_22B, nShift);             // [76 66 56 46]
            T3_23A = _mm256_srai_epi32(T2_23A, nShift);             // [37 27 17 07]
            T3_23B = _mm256_srai_epi32(T2_23B, nShift);             // [77 67 57 47]
            T3_24A = _mm256_srai_epi32(T2_24A, nShift);             // [30 20 10 00] x8
            T3_24B = _mm256_srai_epi32(T2_24B, nShift);             // [70 60 50 40]
            T3_25A = _mm256_srai_epi32(T2_25A, nShift);             // [31 21 11 01] x9
            T3_25B = _mm256_srai_epi32(T2_25B, nShift);             // [71 61 51 41]
            T3_26A = _mm256_srai_epi32(T2_26A, nShift);             // [32 22 12 02] xA
            T3_26B = _mm256_srai_epi32(T2_26B, nShift);             // [72 62 52 42]
            T3_27A = _mm256_srai_epi32(T2_27A, nShift);             // [33 23 13 03] xB
            T3_27B = _mm256_srai_epi32(T2_27B, nShift);             // [73 63 53 43]
            T3_28A = _mm256_srai_epi32(T2_28A, nShift);             // [33 24 14 04] xC
            T3_28B = _mm256_srai_epi32(T2_28B, nShift);             // [74 64 54 44]
            T3_29A = _mm256_srai_epi32(T2_29A, nShift);             // [35 25 15 05] xD
            T3_29B = _mm256_srai_epi32(T2_29B, nShift);             // [75 65 55 45]
            T3_30A = _mm256_srai_epi32(T2_30A, nShift);             // [36 26 16 06] xE
            T3_30B = _mm256_srai_epi32(T2_30B, nShift);             // [76 66 56 46]
            T3_31A = _mm256_srai_epi32(T2_31A, nShift);             // [37 27 17 07] xF
            T3_31B = _mm256_srai_epi32(T2_31B, nShift);             // [77 67 57 47]

            res00[part] = _mm256_packs_epi32(T3_00A, T3_00B);        // [70 60 50 40 30 20 10 00]
            res01[part] = _mm256_packs_epi32(T3_01A, T3_01B);        // [71 61 51 41 31 21 11 01]
            res02[part] = _mm256_packs_epi32(T3_02A, T3_02B);        // [72 62 52 42 32 22 12 02]
            res03[part] = _mm256_packs_epi32(T3_03A, T3_03B);        // [73 63 53 43 33 23 13 03]
            res04[part] = _mm256_packs_epi32(T3_04A, T3_04B);        // [74 64 54 44 34 24 14 04]
            res05[part] = _mm256_packs_epi32(T3_05A, T3_05B);        // [75 65 55 45 35 25 15 05]
            res06[part] = _mm256_packs_epi32(T3_06A, T3_06B);        // [76 66 56 46 36 26 16 06]
            res07[part] = _mm256_packs_epi32(T3_07A, T3_07B);        // [77 67 57 47 37 27 17 07]
            res08[part] = _mm256_packs_epi32(T3_08A, T3_08B);        // [A0 ... 80]
            res09[part] = _mm256_packs_epi32(T3_09A, T3_09B);        // [A1 ... 81]
            res10[part] = _mm256_packs_epi32(T3_10A, T3_10B);        // [A2 ... 82]
            res11[part] = _mm256_packs_epi32(T3_11A, T3_11B);        // [A3 ... 83]
            res12[part] = _mm256_packs_epi32(T3_12A, T3_12B);        // [A4 ... 84]
            res13[part] = _mm256_packs_epi32(T3_13A, T3_13B);        // [A5 ... 85]
            res14[part] = _mm256_packs_epi32(T3_14A, T3_14B);        // [A6 ... 86]
            res15[part] = _mm256_packs_epi32(T3_15A, T3_15B);        // [A7 ... 87]
            res16[part] = _mm256_packs_epi32(T3_16A, T3_16B);
            res17[part] = _mm256_packs_epi32(T3_17A, T3_17B);
            res18[part] = _mm256_packs_epi32(T3_18A, T3_18B);
            res19[part] = _mm256_packs_epi32(T3_19A, T3_19B);
            res20[part] = _mm256_packs_epi32(T3_20A, T3_20B);
            res21[part] = _mm256_packs_epi32(T3_21A, T3_21B);
            res22[part] = _mm256_packs_epi32(T3_22A, T3_22B);
            res23[part] = _mm256_packs_epi32(T3_23A, T3_23B);
            res24[part] = _mm256_packs_epi32(T3_24A, T3_24B);
            res25[part] = _mm256_packs_epi32(T3_25A, T3_25B);
            res26[part] = _mm256_packs_epi32(T3_26A, T3_26B);
            res27[part] = _mm256_packs_epi32(T3_27A, T3_27B);
            res28[part] = _mm256_packs_epi32(T3_28A, T3_28B);
            res29[part] = _mm256_packs_epi32(T3_29A, T3_29B);
            res30[part] = _mm256_packs_epi32(T3_30A, T3_30B);
            res31[part] = _mm256_packs_epi32(T3_31A, T3_31B);

        }

        //transpose 32x32 matrix
        {
            __m256i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7, tr0_8, tr0_9, tr0_10, tr0_11, tr0_12, tr0_13, tr0_14, tr0_15;
#define TRANSPOSE_16x16_16BIT(I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14, I15, O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14, O15) \
        tr0_0 = _mm256_unpacklo_epi16(I0, I1); \
        tr0_1 = _mm256_unpacklo_epi16(I2, I3); \
        tr0_2 = _mm256_unpacklo_epi16(I4, I5); \
        tr0_3 = _mm256_unpacklo_epi16(I6, I7); \
        tr0_4 = _mm256_unpacklo_epi16(I8, I9); \
        tr0_5 = _mm256_unpacklo_epi16(I10, I11); \
        tr0_6 = _mm256_unpacklo_epi16(I12, I13); \
        tr0_7 = _mm256_unpacklo_epi16(I14, I15); \
        tr0_8 = _mm256_unpackhi_epi16(I0, I1); \
        tr0_9 = _mm256_unpackhi_epi16(I2, I3); \
        tr0_10 = _mm256_unpackhi_epi16(I4, I5); \
        tr0_11 = _mm256_unpackhi_epi16(I6, I7); \
        tr0_12 = _mm256_unpackhi_epi16(I8, I9); \
        tr0_13 = _mm256_unpackhi_epi16(I10, I11); \
        tr0_14 = _mm256_unpackhi_epi16(I12, I13); \
        tr0_15 = _mm256_unpackhi_epi16(I14, I15); \
        O0 = _mm256_unpacklo_epi32(tr0_0, tr0_1); \
        O1 = _mm256_unpacklo_epi32(tr0_2, tr0_3); \
        O2 = _mm256_unpacklo_epi32(tr0_4, tr0_5); \
        O3 = _mm256_unpacklo_epi32(tr0_6, tr0_7); \
        O4 = _mm256_unpackhi_epi32(tr0_0, tr0_1); \
        O5 = _mm256_unpackhi_epi32(tr0_2, tr0_3); \
        O6 = _mm256_unpackhi_epi32(tr0_4, tr0_5); \
        O7 = _mm256_unpackhi_epi32(tr0_6, tr0_7); \
        O8 = _mm256_unpacklo_epi32(tr0_8, tr0_9); \
        O9 = _mm256_unpacklo_epi32(tr0_10, tr0_11); \
        O10 = _mm256_unpacklo_epi32(tr0_12, tr0_13); \
        O11 = _mm256_unpacklo_epi32(tr0_14, tr0_15); \
        O12 = _mm256_unpackhi_epi32(tr0_8, tr0_9); \
        O13 = _mm256_unpackhi_epi32(tr0_10, tr0_11); \
        O14 = _mm256_unpackhi_epi32(tr0_12, tr0_13); \
        O15 = _mm256_unpackhi_epi32(tr0_14, tr0_15); \
        tr0_0 = _mm256_unpacklo_epi64(O0, O1); \
        tr0_1 = _mm256_unpacklo_epi64(O2, O3); \
        tr0_2 = _mm256_unpackhi_epi64(O0, O1); \
        tr0_3 = _mm256_unpackhi_epi64(O2, O3); \
        tr0_4 = _mm256_unpacklo_epi64(O4, O5); \
        tr0_5 = _mm256_unpacklo_epi64(O6, O7); \
        tr0_6 = _mm256_unpackhi_epi64(O4, O5); \
        tr0_7 = _mm256_unpackhi_epi64(O6, O7); \
        tr0_8 = _mm256_unpacklo_epi64(O8, O9); \
        tr0_9 = _mm256_unpacklo_epi64(O10, O11); \
        tr0_10 = _mm256_unpackhi_epi64(O8, O9); \
        tr0_11 = _mm256_unpackhi_epi64(O10, O11); \
        tr0_12 = _mm256_unpacklo_epi64(O12, O13); \
        tr0_13 = _mm256_unpacklo_epi64(O14, O15); \
        tr0_14 = _mm256_unpackhi_epi64(O12, O13); \
        tr0_15 = _mm256_unpackhi_epi64(O14, O15); \
        O0 = _mm256_permute2x128_si256(tr0_0, tr0_1, 0x20); \
        O1 = _mm256_permute2x128_si256(tr0_2, tr0_3, 0x20); \
        O2 = _mm256_permute2x128_si256(tr0_4, tr0_5, 0x20); \
        O3 = _mm256_permute2x128_si256(tr0_6, tr0_7, 0x20); \
        O4 = _mm256_permute2x128_si256(tr0_8, tr0_9, 0x20); \
        O5 = _mm256_permute2x128_si256(tr0_10, tr0_11, 0x20); \
        O6 = _mm256_permute2x128_si256(tr0_12, tr0_13, 0x20); \
        O7 = _mm256_permute2x128_si256(tr0_14, tr0_15, 0x20); \
        O8 = _mm256_permute2x128_si256(tr0_0, tr0_1, 0x31); \
        O9 = _mm256_permute2x128_si256(tr0_2, tr0_3, 0x31); \
        O10 = _mm256_permute2x128_si256(tr0_4, tr0_5, 0x31); \
        O11 = _mm256_permute2x128_si256(tr0_6, tr0_7, 0x31); \
        O12 = _mm256_permute2x128_si256(tr0_8, tr0_9, 0x31); \
        O13 = _mm256_permute2x128_si256(tr0_10, tr0_11, 0x31); \
        O14 = _mm256_permute2x128_si256(tr0_12, tr0_13, 0x31); \
        O15 = _mm256_permute2x128_si256(tr0_14, tr0_15, 0x31); \
 
            TRANSPOSE_16x16_16BIT(res00[0], res01[0], res02[0], res03[0], res04[0], res05[0], res06[0], res07[0], res08[0], res09[0], res10[0], res11[0], res12[0], res13[0], res14[0], res15[0], in00[0], in01[0], in02[0], in03[0], in04[0], in05[0], in06[0], in07[0], in08[0], in09[0], in10[0], in11[0], in12[0], in13[0], in14[0], in15[0])
            TRANSPOSE_16x16_16BIT(res16[0], res17[0], res18[0], res19[0], res20[0], res21[0], res22[0], res23[0], res24[0], res25[0], res26[0], res27[0], res28[0], res29[0], res30[0], res31[0], in00[1], in01[1], in02[1], in03[1], in04[1], in05[1], in06[1], in07[1], in08[1], in09[1], in10[1], in11[1], in12[1], in13[1], in14[1], in15[1]);
            TRANSPOSE_16x16_16BIT(res00[1], res01[1], res02[1], res03[1], res04[1], res05[1], res06[1], res07[1], res08[1], res09[1], res10[1], res11[1], res12[1], res13[1], res14[1], res15[1], in16[0], in17[0], in18[0], in19[0], in20[0], in21[0], in22[0], in23[0], in24[0], in25[0], in26[0], in27[0], in28[0], in29[0], in30[0], in31[0]);
            TRANSPOSE_16x16_16BIT(res16[1], res17[1], res18[1], res19[1], res20[1], res21[1], res22[1], res23[1], res24[1], res25[1], res26[1], res27[1], res28[1], res29[1], res30[1], res31[1], in16[1], in17[1], in18[1], in19[1], in20[1], in21[1], in22[1], in23[1], in24[1], in25[1], in26[1], in27[1], in28[1], in29[1], in30[1], in31[1]);

#undef  TRANSPOSE_16x16_16BIT

        }

        c32_rnd = _mm256_set1_epi32(shift ? (1 << (shift - 1)) : 0);                    // pass == 1 第二次四舍五入
        nShift = shift;
    }

    // clip
    max_val = _mm256_set1_epi16((1 << (clip - 1)) - 1);
    min_val = _mm256_set1_epi16(-(1 << (clip - 1)));

    for (k = 0; k < 2; k++) {
        in00[k] = _mm256_max_epi16(_mm256_min_epi16(in00[k], max_val), min_val);
        in01[k] = _mm256_max_epi16(_mm256_min_epi16(in01[k], max_val), min_val);
        in02[k] = _mm256_max_epi16(_mm256_min_epi16(in02[k], max_val), min_val);
        in03[k] = _mm256_max_epi16(_mm256_min_epi16(in03[k], max_val), min_val);
        in04[k] = _mm256_max_epi16(_mm256_min_epi16(in04[k], max_val), min_val);
        in05[k] = _mm256_max_epi16(_mm256_min_epi16(in05[k], max_val), min_val);
        in06[k] = _mm256_max_epi16(_mm256_min_epi16(in06[k], max_val), min_val);
        in07[k] = _mm256_max_epi16(_mm256_min_epi16(in07[k], max_val), min_val);
        in08[k] = _mm256_max_epi16(_mm256_min_epi16(in08[k], max_val), min_val);
        in09[k] = _mm256_max_epi16(_mm256_min_epi16(in09[k], max_val), min_val);
        in10[k] = _mm256_max_epi16(_mm256_min_epi16(in10[k], max_val), min_val);
        in11[k] = _mm256_max_epi16(_mm256_min_epi16(in11[k], max_val), min_val);
        in12[k] = _mm256_max_epi16(_mm256_min_epi16(in12[k], max_val), min_val);
        in13[k] = _mm256_max_epi16(_mm256_min_epi16(in13[k], max_val), min_val);
        in14[k] = _mm256_max_epi16(_mm256_min_epi16(in14[k], max_val), min_val);
        in15[k] = _mm256_max_epi16(_mm256_min_epi16(in15[k], max_val), min_val);
        in16[k] = _mm256_max_epi16(_mm256_min_epi16(in16[k], max_val), min_val);
        in17[k] = _mm256_max_epi16(_mm256_min_epi16(in17[k], max_val), min_val);
        in18[k] = _mm256_max_epi16(_mm256_min_epi16(in18[k], max_val), min_val);
        in19[k] = _mm256_max_epi16(_mm256_min_epi16(in19[k], max_val), min_val);
        in20[k] = _mm256_max_epi16(_mm256_min_epi16(in20[k], max_val), min_val);
        in21[k] = _mm256_max_epi16(_mm256_min_epi16(in21[k], max_val), min_val);
        in22[k] = _mm256_max_epi16(_mm256_min_epi16(in22[k], max_val), min_val);
        in23[k] = _mm256_max_epi16(_mm256_min_epi16(in23[k], max_val), min_val);
        in24[k] = _mm256_max_epi16(_mm256_min_epi16(in24[k], max_val), min_val);
        in25[k] = _mm256_max_epi16(_mm256_min_epi16(in25[k], max_val), min_val);
        in26[k] = _mm256_max_epi16(_mm256_min_epi16(in26[k], max_val), min_val);
        in27[k] = _mm256_max_epi16(_mm256_min_epi16(in27[k], max_val), min_val);
        in28[k] = _mm256_max_epi16(_mm256_min_epi16(in28[k], max_val), min_val);
        in29[k] = _mm256_max_epi16(_mm256_min_epi16(in29[k], max_val), min_val);
        in30[k] = _mm256_max_epi16(_mm256_min_epi16(in30[k], max_val), min_val);
        in31[k] = _mm256_max_epi16(_mm256_min_epi16(in31[k], max_val), min_val);
    }


    // Store
    for (i = 0; i < 2; i++) {
        const int offset = (i << 4);
        _mm256_storeu_si256((__m256i*)&dst[0 * 32 + offset], in00[i]);
        _mm256_storeu_si256((__m256i*)&dst[1 * 32 + offset], in01[i]);
        _mm256_storeu_si256((__m256i*)&dst[2 * 32 + offset], in02[i]);
        _mm256_storeu_si256((__m256i*)&dst[3 * 32 + offset], in03[i]);
        _mm256_storeu_si256((__m256i*)&dst[4 * 32 + offset], in04[i]);
        _mm256_storeu_si256((__m256i*)&dst[5 * 32 + offset], in05[i]);
        _mm256_storeu_si256((__m256i*)&dst[6 * 32 + offset], in06[i]);
        _mm256_storeu_si256((__m256i*)&dst[7 * 32 + offset], in07[i]);
        _mm256_storeu_si256((__m256i*)&dst[8 * 32 + offset], in08[i]);
        _mm256_storeu_si256((__m256i*)&dst[9 * 32 + offset], in09[i]);
        _mm256_storeu_si256((__m256i*)&dst[10 * 32 + offset], in10[i]);
        _mm256_storeu_si256((__m256i*)&dst[11 * 32 + offset], in11[i]);
        _mm256_storeu_si256((__m256i*)&dst[12 * 32 + offset], in12[i]);
        _mm256_storeu_si256((__m256i*)&dst[13 * 32 + offset], in13[i]);
        _mm256_storeu_si256((__m256i*)&dst[14 * 32 + offset], in14[i]);
        _mm256_storeu_si256((__m256i*)&dst[15 * 32 + offset], in15[i]);
        _mm256_storeu_si256((__m256i*)&dst[16 * 32 + offset], in16[i]);
        _mm256_storeu_si256((__m256i*)&dst[17 * 32 + offset], in17[i]);
        _mm256_storeu_si256((__m256i*)&dst[18 * 32 + offset], in18[i]);
        _mm256_storeu_si256((__m256i*)&dst[19 * 32 + offset], in19[i]);
        _mm256_storeu_si256((__m256i*)&dst[20 * 32 + offset], in20[i]);
        _mm256_storeu_si256((__m256i*)&dst[21 * 32 + offset], in21[i]);
        _mm256_storeu_si256((__m256i*)&dst[22 * 32 + offset], in22[i]);
        _mm256_storeu_si256((__m256i*)&dst[23 * 32 + offset], in23[i]);
        _mm256_storeu_si256((__m256i*)&dst[24 * 32 + offset], in24[i]);
        _mm256_storeu_si256((__m256i*)&dst[25 * 32 + offset], in25[i]);
        _mm256_storeu_si256((__m256i*)&dst[26 * 32 + offset], in26[i]);
        _mm256_storeu_si256((__m256i*)&dst[27 * 32 + offset], in27[i]);
        _mm256_storeu_si256((__m256i*)&dst[28 * 32 + offset], in28[i]);
        _mm256_storeu_si256((__m256i*)&dst[29 * 32 + offset], in29[i]);
        _mm256_storeu_si256((__m256i*)&dst[30 * 32 + offset], in30[i]);
        _mm256_storeu_si256((__m256i*)&dst[31 * 32 + offset], in31[i]);
    }

}



#define TRANSPOSE_8x8_16BIT_m256i(I0, I1, I2, I3, I4, I5, I6, I7, O0, O1, O2, O3, O4, O5, O6, O7) \
        tr0_0 = _mm256_unpacklo_epi16(I0, I1); \
        tr0_1 = _mm256_unpacklo_epi16(I2, I3); \
        tr0_2 = _mm256_unpackhi_epi16(I0, I1); \
        tr0_3 = _mm256_unpackhi_epi16(I2, I3); \
        tr0_4 = _mm256_unpacklo_epi16(I4, I5); \
        tr0_5 = _mm256_unpacklo_epi16(I6, I7); \
        tr0_6 = _mm256_unpackhi_epi16(I4, I5); \
        tr0_7 = _mm256_unpackhi_epi16(I6, I7); \
        tr1_0 = _mm256_unpacklo_epi32(tr0_0, tr0_1); \
        tr1_1 = _mm256_unpacklo_epi32(tr0_2, tr0_3); \
        tr1_2 = _mm256_unpackhi_epi32(tr0_0, tr0_1); \
        tr1_3 = _mm256_unpackhi_epi32(tr0_2, tr0_3); \
        tr1_4 = _mm256_unpacklo_epi32(tr0_4, tr0_5); \
        tr1_5 = _mm256_unpacklo_epi32(tr0_6, tr0_7); \
        tr1_6 = _mm256_unpackhi_epi32(tr0_4, tr0_5); \
        tr1_7 = _mm256_unpackhi_epi32(tr0_6, tr0_7); \
        O0 = _mm256_unpacklo_epi64(tr1_0, tr1_4); \
        O1 = _mm256_unpackhi_epi64(tr1_0, tr1_4); \
        O2 = _mm256_unpacklo_epi64(tr1_2, tr1_6); \
        O3 = _mm256_unpackhi_epi64(tr1_2, tr1_6); \
        O4 = _mm256_unpacklo_epi64(tr1_1, tr1_5); \
        O5 = _mm256_unpackhi_epi64(tr1_1, tr1_5); \
        O6 = _mm256_unpacklo_epi64(tr1_3, tr1_7); \
        O7 = _mm256_unpackhi_epi64(tr1_3, tr1_7);

#define TRANSPOSE_16x16_16BIT_m256i(I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14, I15, O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14, O15) \
        TRANSPOSE_8x8_16BIT_m256i(I0, I1, I2, I3, I4, I5, I6, I7, t0, t1, t2, t3, t4, t5, t6, t7); \
        TRANSPOSE_8x8_16BIT_m256i(I8, I9, I10, I11, I12, I13, I14, I15, t8, t9, t10, t11, t12, t13, t14, t15); \
        O0 = _mm256_permute2x128_si256(t0, t8, 0x20); \
        O1 = _mm256_permute2x128_si256(t1, t9, 0x20); \
        O2 = _mm256_permute2x128_si256(t2, t10, 0x20); \
        O3 = _mm256_permute2x128_si256(t3, t11, 0x20); \
        O4 = _mm256_permute2x128_si256(t4, t12, 0x20); \
        O5 = _mm256_permute2x128_si256(t5, t13, 0x20); \
        O6 = _mm256_permute2x128_si256(t6, t14, 0x20); \
        O7 = _mm256_permute2x128_si256(t7, t15, 0x20); \
        O8 = _mm256_permute2x128_si256(t0, t8, 0x31); \
        O9 = _mm256_permute2x128_si256(t1, t9, 0x31); \
        O10 = _mm256_permute2x128_si256(t2, t10, 0x31); \
        O11 = _mm256_permute2x128_si256(t3, t11, 0x31); \
        O12 = _mm256_permute2x128_si256(t4, t12, 0x31); \
        O13 = _mm256_permute2x128_si256(t5, t13, 0x31); \
        O14 = _mm256_permute2x128_si256(t6, t14, 0x31); \
        O15 = _mm256_permute2x128_si256(t7, t15, 0x31);

//inv_wavelet_64x16_sse128
static void inv_wavelet_64x16_avx2(coeff_t *coeff)
{
    int i;

    __m256i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
    __m256i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;
    __m256i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

    //按行 64*16
    __m256i T00[4], T01[4], T02[4], T03[4], T04[4], T05[4], T06[4], T07[4], T08[4], T09[4], T10[4], T11[4], T12[4], T13[4], T14[4], T15[4];

    //按列 16*64
    __m256i V00, V01, V02, V03, V04, V05, V06, V07, V08, V09, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29, V30, V31, V32, V33, V34, V35, V36, V37, V38, V39, V40, V41, V42, V43, V44, V45, V46, V47, V48, V49, V50, V51, V52, V53, V54, V55, V56, V57, V58, V59, V60, V61, V62, V63;

    /*--vertical transform--*/
    //32*8, LOAD AND SHIFT
    T00[0] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[0 + 32 * 0]), 1);
    T01[0] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[0 + 32 * 1]), 1);
    T02[0] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[0 + 32 * 2]), 1);
    T03[0] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[0 + 32 * 3]), 1);
    T04[0] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[0 + 32 * 4]), 1);
    T05[0] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[0 + 32 * 5]), 1);
    T06[0] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[0 + 32 * 6]), 1);
    T07[0] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[0 + 32 * 7]), 1);

    T00[1] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 + 32 * 0]), 1);
    T01[1] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 + 32 * 1]), 1);
    T02[1] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 + 32 * 2]), 1);
    T03[1] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 + 32 * 3]), 1);
    T04[1] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 + 32 * 4]), 1);
    T05[1] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 + 32 * 5]), 1);
    T06[1] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 + 32 * 6]), 1);
    T07[1] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 + 32 * 7]), 1);

    //filter (odd pixel/row)
    T08[0] = _mm256_srai_epi16(_mm256_add_epi16(T00[0], T01[0]), 1);
    T09[0] = _mm256_srai_epi16(_mm256_add_epi16(T01[0], T02[0]), 1);
    T10[0] = _mm256_srai_epi16(_mm256_add_epi16(T02[0], T03[0]), 1);
    T11[0] = _mm256_srai_epi16(_mm256_add_epi16(T03[0], T04[0]), 1);
    T12[0] = _mm256_srai_epi16(_mm256_add_epi16(T04[0], T05[0]), 1);
    T13[0] = _mm256_srai_epi16(_mm256_add_epi16(T05[0], T06[0]), 1);
    T14[0] = _mm256_srai_epi16(_mm256_add_epi16(T06[0], T07[0]), 1);
    T15[0] = _mm256_srai_epi16(_mm256_add_epi16(T07[0], T07[0]), 1);

    T08[1] = _mm256_srai_epi16(_mm256_add_epi16(T00[1], T01[1]), 1);
    T09[1] = _mm256_srai_epi16(_mm256_add_epi16(T01[1], T02[1]), 1);
    T10[1] = _mm256_srai_epi16(_mm256_add_epi16(T02[1], T03[1]), 1);
    T11[1] = _mm256_srai_epi16(_mm256_add_epi16(T03[1], T04[1]), 1);
    T12[1] = _mm256_srai_epi16(_mm256_add_epi16(T04[1], T05[1]), 1);
    T13[1] = _mm256_srai_epi16(_mm256_add_epi16(T05[1], T06[1]), 1);
    T14[1] = _mm256_srai_epi16(_mm256_add_epi16(T06[1], T07[1]), 1);
    T15[1] = _mm256_srai_epi16(_mm256_add_epi16(T07[1], T07[1]), 1);

    /*--transposition--*/
    //32x16 -> 16x32
    TRANSPOSE_16x16_16BIT_m256i(T00[0], T08[0], T01[0], T09[0], T02[0], T10[0], T03[0], T11[0], T04[0], T12[0], T05[0], T13[0], T06[0], T14[0], T07[0], T15[0], V00, V01, V02, V03, V04, V05, V06, V07, V08, V09, V10, V11, V12, V13, V14, V15);
    TRANSPOSE_16x16_16BIT_m256i(T00[1], T08[1], T01[1], T09[1], T02[1], T10[1], T03[1], T11[1], T04[1], T12[1], T05[1], T13[1], T06[1], T14[1], T07[1], T15[1], V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29, V30, V31);

    /*--horizontal transform--*/
    //filter (odd pixel/column)
    V32 = _mm256_srai_epi16(_mm256_add_epi16(V00, V01), 1);
    V33 = _mm256_srai_epi16(_mm256_add_epi16(V01, V02), 1);
    V34 = _mm256_srai_epi16(_mm256_add_epi16(V02, V03), 1);
    V35 = _mm256_srai_epi16(_mm256_add_epi16(V03, V04), 1);
    V36 = _mm256_srai_epi16(_mm256_add_epi16(V04, V05), 1);
    V37 = _mm256_srai_epi16(_mm256_add_epi16(V05, V06), 1);
    V38 = _mm256_srai_epi16(_mm256_add_epi16(V06, V07), 1);
    V39 = _mm256_srai_epi16(_mm256_add_epi16(V07, V08), 1);
    V40 = _mm256_srai_epi16(_mm256_add_epi16(V08, V09), 1);
    V41 = _mm256_srai_epi16(_mm256_add_epi16(V09, V10), 1);
    V42 = _mm256_srai_epi16(_mm256_add_epi16(V10, V11), 1);
    V43 = _mm256_srai_epi16(_mm256_add_epi16(V11, V12), 1);
    V44 = _mm256_srai_epi16(_mm256_add_epi16(V12, V13), 1);
    V45 = _mm256_srai_epi16(_mm256_add_epi16(V13, V14), 1);
    V46 = _mm256_srai_epi16(_mm256_add_epi16(V14, V15), 1);
    V47 = _mm256_srai_epi16(_mm256_add_epi16(V15, V16), 1);

    V48 = _mm256_srai_epi16(_mm256_add_epi16(V16, V17), 1);
    V49 = _mm256_srai_epi16(_mm256_add_epi16(V17, V18), 1);
    V50 = _mm256_srai_epi16(_mm256_add_epi16(V18, V19), 1);
    V51 = _mm256_srai_epi16(_mm256_add_epi16(V19, V20), 1);
    V52 = _mm256_srai_epi16(_mm256_add_epi16(V20, V21), 1);
    V53 = _mm256_srai_epi16(_mm256_add_epi16(V21, V22), 1);
    V54 = _mm256_srai_epi16(_mm256_add_epi16(V22, V23), 1);
    V55 = _mm256_srai_epi16(_mm256_add_epi16(V23, V24), 1);
    V56 = _mm256_srai_epi16(_mm256_add_epi16(V24, V25), 1);
    V57 = _mm256_srai_epi16(_mm256_add_epi16(V25, V26), 1);
    V58 = _mm256_srai_epi16(_mm256_add_epi16(V26, V27), 1);
    V59 = _mm256_srai_epi16(_mm256_add_epi16(V27, V28), 1);
    V60 = _mm256_srai_epi16(_mm256_add_epi16(V28, V29), 1);
    V61 = _mm256_srai_epi16(_mm256_add_epi16(V29, V30), 1);
    V62 = _mm256_srai_epi16(_mm256_add_epi16(V30, V31), 1);
    V63 = _mm256_srai_epi16(_mm256_add_epi16(V31, V31), 1);

    /*--transposition & Store--*/
    //16x64 -> 64x16
    TRANSPOSE_16x16_16BIT_m256i(V00, V32, V01, V33, V02, V34, V03, V35, V04, V36, V05, V37, V06, V38, V07, V39, T00[0], T01[0], T02[0], T03[0], T04[0], T05[0], T06[0], T07[0], T08[0], T09[0], T10[0], T11[0], T12[0], T13[0], T14[0], T15[0]);
    TRANSPOSE_16x16_16BIT_m256i(V08, V40, V09, V41, V10, V42, V11, V43, V12, V44, V13, V45, V14, V46, V15, V47, T00[1], T01[1], T02[1], T03[1], T04[1], T05[1], T06[1], T07[1], T08[1], T09[1], T10[1], T11[1], T12[1], T13[1], T14[1], T15[1]);
    TRANSPOSE_16x16_16BIT_m256i(V16, V48, V17, V49, V18, V50, V19, V51, V20, V52, V21, V53, V22, V54, V23, V55, T00[2], T01[2], T02[2], T03[2], T04[2], T05[2], T06[2], T07[2], T08[2], T09[2], T10[2], T11[2], T12[2], T13[2], T14[2], T15[2]);
    TRANSPOSE_16x16_16BIT_m256i(V24, V56, V25, V57, V26, V58, V27, V59, V28, V60, V29, V61, V30, V62, V31, V63, T00[3], T01[3], T02[3], T03[3], T04[3], T05[3], T06[3], T07[3], T08[3], T09[3], T10[3], T11[3], T12[3], T13[3], T14[3], T15[3]);

    //store
    for (i = 0; i < 4; i++) {
        _mm256_storeu_si256((__m256i*)&coeff[16 * i], T00[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64], T01[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 2], T02[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 3], T03[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 4], T04[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 5], T05[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 6], T06[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 7], T07[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 8], T08[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 9], T09[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 10], T10[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 11], T11[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 12], T12[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 13], T13[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 14], T14[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 15], T15[i]);
    }
}

static void inv_wavelet_16x64_avx2(coeff_t *coeff)
{
    //src blk 8*32

    __m256i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
    __m256i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;
    __m256i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

    __m256i S00, S01, S02, S03, S04, S05, S06, S07, S08, S09, S10, S11, S12, S13, S14, S15, S16, S17, S18, S19, S20, S21, S22, S23, S24, S25, S26, S27, S28, S29, S30, S31;
    __m256i S32, S33, S34, S35, S36, S37, S38, S39, S40, S41, S42, S43, S44, S45, S46, S47, S48, S49, S50, S51, S52, S53, S54, S55, S56, S57, S58, S59, S60, S61, S62, S63;

    //按行 64*16
    __m256i TT00[8], TT01[8], TT02[8], TT03[8], TT04[8], TT05[8], TT06[8], TT07[8];
    __m256i T00[4], T01[4], T02[4], T03[4], T04[4], T05[4], T06[4], T07[4], T08[4], T09[4], T10[4], T11[4], T12[4], T13[4], T14[4], T15[4];

    //按列 16*64
    __m256i V00, V01, V02, V03, V04, V05, V06, V07, V08, V09, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29, V30, V31, V32, V33, V34, V35, V36, V37, V38, V39, V40, V41, V42, V43, V44, V45, V46, V47, V48, V49, V50, V51, V52, V53, V54, V55, V56, V57, V58, V59, V60, V61, V62, V63;

    int i;
    /*--load & shift--*/
    //8*32
    S00 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 0]), 1);
    S01 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 1]), 1);
    S02 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 2]), 1);
    S03 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 3]), 1);
    S04 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 4]), 1);
    S05 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 5]), 1);
    S06 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 6]), 1);
    S07 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 7]), 1);
    S08 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 8]), 1);
    S09 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 9]), 1);
    S10 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 10]), 1);
    S11 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 11]), 1);
    S12 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 12]), 1);
    S13 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 13]), 1);
    S14 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 14]), 1);
    S15 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 15]), 1);
    S16 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 16]), 1);
    S17 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 17]), 1);
    S18 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 18]), 1);
    S19 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 19]), 1);
    S20 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 20]), 1);
    S21 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 21]), 1);
    S22 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 22]), 1);
    S23 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 23]), 1);
    S24 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 24]), 1);
    S25 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 25]), 1);
    S26 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 26]), 1);
    S27 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 27]), 1);
    S28 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 28]), 1);
    S29 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 29]), 1);
    S30 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 30]), 1);
    S31 = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[8 * 31]), 1);

    /*--vertical transform--*/
    S32 = _mm256_srai_epi16(_mm256_add_epi16(S00, S01), 1);
    S33 = _mm256_srai_epi16(_mm256_add_epi16(S01, S02), 1);
    S34 = _mm256_srai_epi16(_mm256_add_epi16(S02, S03), 1);
    S35 = _mm256_srai_epi16(_mm256_add_epi16(S03, S04), 1);
    S36 = _mm256_srai_epi16(_mm256_add_epi16(S04, S05), 1);
    S37 = _mm256_srai_epi16(_mm256_add_epi16(S05, S06), 1);
    S38 = _mm256_srai_epi16(_mm256_add_epi16(S06, S07), 1);
    S39 = _mm256_srai_epi16(_mm256_add_epi16(S07, S08), 1);
    S40 = _mm256_srai_epi16(_mm256_add_epi16(S08, S09), 1);
    S41 = _mm256_srai_epi16(_mm256_add_epi16(S09, S10), 1);
    S42 = _mm256_srai_epi16(_mm256_add_epi16(S10, S11), 1);
    S43 = _mm256_srai_epi16(_mm256_add_epi16(S11, S12), 1);
    S44 = _mm256_srai_epi16(_mm256_add_epi16(S12, S13), 1);
    S45 = _mm256_srai_epi16(_mm256_add_epi16(S13, S14), 1);
    S46 = _mm256_srai_epi16(_mm256_add_epi16(S14, S15), 1);
    S47 = _mm256_srai_epi16(_mm256_add_epi16(S15, S16), 1);
    S48 = _mm256_srai_epi16(_mm256_add_epi16(S16, S17), 1);
    S49 = _mm256_srai_epi16(_mm256_add_epi16(S17, S18), 1);
    S50 = _mm256_srai_epi16(_mm256_add_epi16(S18, S19), 1);
    S51 = _mm256_srai_epi16(_mm256_add_epi16(S19, S20), 1);
    S52 = _mm256_srai_epi16(_mm256_add_epi16(S20, S21), 1);
    S53 = _mm256_srai_epi16(_mm256_add_epi16(S21, S22), 1);
    S54 = _mm256_srai_epi16(_mm256_add_epi16(S22, S23), 1);
    S55 = _mm256_srai_epi16(_mm256_add_epi16(S23, S24), 1);
    S56 = _mm256_srai_epi16(_mm256_add_epi16(S24, S25), 1);
    S57 = _mm256_srai_epi16(_mm256_add_epi16(S25, S26), 1);
    S58 = _mm256_srai_epi16(_mm256_add_epi16(S26, S27), 1);
    S59 = _mm256_srai_epi16(_mm256_add_epi16(S27, S28), 1);
    S60 = _mm256_srai_epi16(_mm256_add_epi16(S28, S29), 1);
    S61 = _mm256_srai_epi16(_mm256_add_epi16(S29, S30), 1);
    S62 = _mm256_srai_epi16(_mm256_add_epi16(S30, S31), 1);
    S63 = _mm256_srai_epi16(_mm256_add_epi16(S31, S31), 1);

    /*--transposition--*/
    //8x64 -> 64x8
    TRANSPOSE_8x8_16BIT_m256i(S00, S32, S01, S33, S02, S34, S03, S35, TT00[0], TT01[0], TT02[0], TT03[0], TT04[0], TT05[0], TT06[0], TT07[0]);
    TRANSPOSE_8x8_16BIT_m256i(S04, S36, S05, S37, S06, S38, S07, S39, TT00[1], TT01[1], TT02[1], TT03[1], TT04[1], TT05[1], TT06[1], TT07[1]);
    TRANSPOSE_8x8_16BIT_m256i(S08, S40, S09, S41, S10, S42, S11, S43, TT00[2], TT01[2], TT02[2], TT03[2], TT04[2], TT05[2], TT06[2], TT07[2]);
    TRANSPOSE_8x8_16BIT_m256i(S12, S44, S13, S45, S14, S46, S15, S47, TT00[3], TT01[3], TT02[3], TT03[3], TT04[3], TT05[3], TT06[3], TT07[3]);
    TRANSPOSE_8x8_16BIT_m256i(S16, S48, S17, S49, S18, S50, S19, S51, TT00[4], TT01[4], TT02[4], TT03[4], TT04[4], TT05[4], TT06[4], TT07[4]);
    TRANSPOSE_8x8_16BIT_m256i(S20, S52, S21, S53, S22, S54, S23, S55, TT00[5], TT01[5], TT02[5], TT03[5], TT04[5], TT05[5], TT06[5], TT07[5]);
    TRANSPOSE_8x8_16BIT_m256i(S24, S56, S25, S57, S26, S58, S27, S59, TT00[6], TT01[6], TT02[6], TT03[6], TT04[6], TT05[6], TT06[6], TT07[6]);
    TRANSPOSE_8x8_16BIT_m256i(S28, S60, S29, S61, S30, S62, S31, S63, TT00[7], TT01[7], TT02[7], TT03[7], TT04[7], TT05[7], TT06[7], TT07[7]);

    T00[0] = _mm256_permute2x128_si256(TT00[0], TT00[1], 0x20);
    T00[1] = _mm256_permute2x128_si256(TT00[2], TT00[3], 0x20);
    T00[2] = _mm256_permute2x128_si256(TT00[4], TT00[5], 0x20);
    T00[3] = _mm256_permute2x128_si256(TT00[6], TT00[7], 0x20);
    T01[0] = _mm256_permute2x128_si256(TT01[0], TT01[1], 0x20);
    T01[1] = _mm256_permute2x128_si256(TT01[2], TT01[3], 0x20);
    T01[2] = _mm256_permute2x128_si256(TT01[4], TT01[5], 0x20);
    T01[3] = _mm256_permute2x128_si256(TT01[6], TT01[7], 0x20);
    T02[0] = _mm256_permute2x128_si256(TT02[0], TT02[1], 0x20);
    T02[1] = _mm256_permute2x128_si256(TT02[2], TT02[3], 0x20);
    T02[2] = _mm256_permute2x128_si256(TT02[4], TT02[5], 0x20);
    T02[3] = _mm256_permute2x128_si256(TT02[6], TT02[7], 0x20);
    T03[0] = _mm256_permute2x128_si256(TT03[0], TT03[1], 0x20);
    T03[1] = _mm256_permute2x128_si256(TT03[2], TT03[3], 0x20);
    T03[2] = _mm256_permute2x128_si256(TT03[4], TT03[5], 0x20);
    T03[3] = _mm256_permute2x128_si256(TT03[6], TT03[7], 0x20);

    T04[0] = _mm256_permute2x128_si256(TT04[0], TT04[1], 0x20);
    T04[1] = _mm256_permute2x128_si256(TT04[2], TT04[3], 0x20);
    T04[2] = _mm256_permute2x128_si256(TT04[4], TT04[5], 0x20);
    T04[3] = _mm256_permute2x128_si256(TT04[6], TT04[7], 0x20);
    T05[0] = _mm256_permute2x128_si256(TT05[0], TT05[1], 0x20);
    T05[1] = _mm256_permute2x128_si256(TT05[2], TT05[3], 0x20);
    T05[2] = _mm256_permute2x128_si256(TT05[4], TT05[5], 0x20);
    T05[3] = _mm256_permute2x128_si256(TT05[6], TT05[7], 0x20);
    T06[0] = _mm256_permute2x128_si256(TT06[0], TT06[1], 0x20);
    T06[1] = _mm256_permute2x128_si256(TT06[2], TT06[3], 0x20);
    T06[2] = _mm256_permute2x128_si256(TT06[4], TT06[5], 0x20);
    T06[3] = _mm256_permute2x128_si256(TT06[6], TT06[7], 0x20);
    T07[0] = _mm256_permute2x128_si256(TT07[0], TT07[1], 0x20);
    T07[1] = _mm256_permute2x128_si256(TT07[2], TT07[3], 0x20);
    T07[2] = _mm256_permute2x128_si256(TT07[4], TT07[5], 0x20);
    T07[3] = _mm256_permute2x128_si256(TT07[6], TT07[7], 0x20);

    /*--horizontal transform--*/
    for (i = 0; i < 4; i++) {
        T08[i] = _mm256_srai_epi16(_mm256_add_epi16(T00[i], T01[i]), 1);
        T09[i] = _mm256_srai_epi16(_mm256_add_epi16(T01[i], T02[i]), 1);
        T10[i] = _mm256_srai_epi16(_mm256_add_epi16(T02[i], T03[i]), 1);
        T11[i] = _mm256_srai_epi16(_mm256_add_epi16(T03[i], T04[i]), 1);
        T12[i] = _mm256_srai_epi16(_mm256_add_epi16(T04[i], T05[i]), 1);
        T13[i] = _mm256_srai_epi16(_mm256_add_epi16(T05[i], T06[i]), 1);
        T14[i] = _mm256_srai_epi16(_mm256_add_epi16(T06[i], T07[i]), 1);
        T15[i] = _mm256_srai_epi16(_mm256_add_epi16(T07[i], T07[i]), 1);
    }

    /*--transposition--*/
    //64x16 -> 16x64
    TRANSPOSE_16x16_16BIT_m256i(T00[0], T08[0], T01[0], T09[0], T02[0], T10[0], T03[0], T11[0], T04[0], T12[0], T05[0], T13[0], T06[0], T14[0], T07[0], T15[0], V00, V01, V02, V03, V04, V05, V06, V07, V08, V09, V10, V11, V12, V13, V14, V15);
    TRANSPOSE_16x16_16BIT_m256i(T00[1], T08[1], T01[1], T09[1], T02[1], T10[1], T03[1], T11[1], T04[1], T12[1], T05[1], T13[1], T06[1], T14[1], T07[1], T15[1], V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29, V30, V31);
    TRANSPOSE_16x16_16BIT_m256i(T00[2], T08[2], T01[2], T09[2], T02[2], T10[2], T03[2], T11[2], T04[2], T12[2], T05[2], T13[2], T06[2], T14[2], T07[2], T15[2], V32, V33, V34, V35, V36, V37, V38, V39, V40, V41, V42, V43, V44, V45, V46, V47);
    TRANSPOSE_16x16_16BIT_m256i(T00[3], T08[3], T01[3], T09[3], T02[3], T10[3], T03[3], T11[3], T04[3], T12[3], T05[3], T13[3], T06[3], T14[3], T07[3], T15[3], V48, V49, V50, V51, V52, V53, V54, V55, V56, V57, V58, V59, V60, V61, V62, V63);

    /*--Store--*/
    //16x64
    _mm256_storeu_si256((__m256i*)&coeff[16 * 0], V00);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 1], V01);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 2], V02);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 3], V03);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 4], V04);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 5], V05);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 6], V06);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 7], V07);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 8], V08);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 9], V09);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 10], V10);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 11], V11);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 12], V12);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 13], V13);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 14], V14);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 15], V15);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 16], V16);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 17], V17);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 18], V18);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 19], V19);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 20], V20);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 21], V21);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 22], V22);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 23], V23);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 24], V24);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 25], V25);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 26], V26);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 27], V27);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 28], V28);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 29], V29);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 30], V30);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 31], V31);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 32], V32);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 33], V33);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 34], V34);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 35], V35);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 36], V36);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 37], V37);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 38], V38);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 39], V39);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 40], V40);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 41], V41);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 42], V42);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 43], V43);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 44], V44);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 45], V45);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 46], V46);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 47], V47);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 48], V48);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 49], V49);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 50], V50);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 51], V51);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 52], V52);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 53], V53);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 54], V54);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 55], V55);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 56], V56);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 57], V57);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 58], V58);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 59], V59);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 60], V60);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 61], V61);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 62], V62);
    _mm256_storeu_si256((__m256i*)&coeff[16 * 63], V63);
}

static void inv_wavelet_64x64_avx2(coeff_t *coeff)
{
    int i;

    __m256i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
    __m256i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;
    __m256i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

    //按行 64*64
    __m256i T00[4], T01[4], T02[4], T03[4], T04[4], T05[4], T06[4], T07[4], T08[4], T09[4], T10[4], T11[4], T12[4], T13[4], T14[4], T15[4], T16[4], T17[4], T18[4], T19[4], T20[4], T21[4], T22[4], T23[4], T24[4], T25[4], T26[4], T27[4], T28[4], T29[4], T30[4], T31[4], T32[4], T33[4], T34[4], T35[4], T36[4], T37[4], T38[4], T39[4], T40[4], T41[4], T42[4], T43[4], T44[4], T45[4], T46[4], T47[4], T48[4], T49[4], T50[4], T51[4], T52[4], T53[4], T54[4], T55[4], T56[4], T57[4], T58[4], T59[4], T60[4], T61[4], T62[4], T63[4];

    //按列 64*64
    __m256i V00[4], V01[4], V02[4], V03[4], V04[4], V05[4], V06[4], V07[4], V08[4], V09[4], V10[4], V11[4], V12[4], V13[4], V14[4], V15[4], V16[4], V17[4], V18[4], V19[4], V20[4], V21[4], V22[4], V23[4], V24[4], V25[4], V26[4], V27[4], V28[4], V29[4], V30[4], V31[4], V32[4], V33[4], V34[4], V35[4], V36[4], V37[4], V38[4], V39[4], V40[4], V41[4], V42[4], V43[4], V44[4], V45[4], V46[4], V47[4], V48[4], V49[4], V50[4], V51[4], V52[4], V53[4], V54[4], V55[4], V56[4], V57[4], V58[4], V59[4], V60[4], V61[4], V62[4], V63[4];

    /*--vertical transform--*/
    //32*32, LOAD AND SHIFT
    for (i = 0; i < 2; i++) {
        T00[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 0]), 1);
        T01[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 1]), 1);
        T02[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 2]), 1);
        T03[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 3]), 1);
        T04[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 4]), 1);
        T05[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 5]), 1);
        T06[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 6]), 1);
        T07[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 7]), 1);

        T08[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 8]), 1);
        T09[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 9]), 1);
        T10[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 10]), 1);
        T11[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 11]), 1);
        T12[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 12]), 1);
        T13[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 13]), 1);
        T14[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 14]), 1);
        T15[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 15]), 1);

        T16[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 16]), 1);
        T17[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 17]), 1);
        T18[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 18]), 1);
        T19[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 19]), 1);
        T20[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 20]), 1);
        T21[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 21]), 1);
        T22[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 22]), 1);
        T23[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 23]), 1);

        T24[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 24]), 1);
        T25[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 25]), 1);
        T26[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 26]), 1);
        T27[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 27]), 1);
        T28[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 28]), 1);
        T29[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 29]), 1);
        T30[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 30]), 1);
        T31[i] = _mm256_srai_epi16(_mm256_loadu_si256((__m256i*)&coeff[16 * i + 32 * 31]), 1);
    }

    //filter (odd pixel/row)
    for (i = 0; i < 4; i++) {
        T32[i] = _mm256_srai_epi16(_mm256_add_epi16(T00[i], T01[i]), 1);
        T33[i] = _mm256_srai_epi16(_mm256_add_epi16(T01[i], T02[i]), 1);
        T34[i] = _mm256_srai_epi16(_mm256_add_epi16(T02[i], T03[i]), 1);
        T35[i] = _mm256_srai_epi16(_mm256_add_epi16(T03[i], T04[i]), 1);
        T36[i] = _mm256_srai_epi16(_mm256_add_epi16(T04[i], T05[i]), 1);
        T37[i] = _mm256_srai_epi16(_mm256_add_epi16(T05[i], T06[i]), 1);
        T38[i] = _mm256_srai_epi16(_mm256_add_epi16(T06[i], T07[i]), 1);
        T39[i] = _mm256_srai_epi16(_mm256_add_epi16(T07[i], T08[i]), 1);

        T40[i] = _mm256_srai_epi16(_mm256_add_epi16(T08[i], T09[i]), 1);
        T41[i] = _mm256_srai_epi16(_mm256_add_epi16(T09[i], T10[i]), 1);
        T42[i] = _mm256_srai_epi16(_mm256_add_epi16(T10[i], T11[i]), 1);
        T43[i] = _mm256_srai_epi16(_mm256_add_epi16(T11[i], T12[i]), 1);
        T44[i] = _mm256_srai_epi16(_mm256_add_epi16(T12[i], T13[i]), 1);
        T45[i] = _mm256_srai_epi16(_mm256_add_epi16(T13[i], T14[i]), 1);
        T46[i] = _mm256_srai_epi16(_mm256_add_epi16(T14[i], T15[i]), 1);
        T47[i] = _mm256_srai_epi16(_mm256_add_epi16(T15[i], T16[i]), 1);

        T48[i] = _mm256_srai_epi16(_mm256_add_epi16(T16[i], T17[i]), 1);
        T49[i] = _mm256_srai_epi16(_mm256_add_epi16(T17[i], T18[i]), 1);
        T50[i] = _mm256_srai_epi16(_mm256_add_epi16(T18[i], T19[i]), 1);
        T51[i] = _mm256_srai_epi16(_mm256_add_epi16(T19[i], T20[i]), 1);
        T52[i] = _mm256_srai_epi16(_mm256_add_epi16(T20[i], T21[i]), 1);
        T53[i] = _mm256_srai_epi16(_mm256_add_epi16(T21[i], T22[i]), 1);
        T54[i] = _mm256_srai_epi16(_mm256_add_epi16(T22[i], T23[i]), 1);
        T55[i] = _mm256_srai_epi16(_mm256_add_epi16(T23[i], T24[i]), 1);

        T56[i] = _mm256_srai_epi16(_mm256_add_epi16(T24[i], T25[i]), 1);
        T57[i] = _mm256_srai_epi16(_mm256_add_epi16(T25[i], T26[i]), 1);
        T58[i] = _mm256_srai_epi16(_mm256_add_epi16(T26[i], T27[i]), 1);
        T59[i] = _mm256_srai_epi16(_mm256_add_epi16(T27[i], T28[i]), 1);
        T60[i] = _mm256_srai_epi16(_mm256_add_epi16(T28[i], T29[i]), 1);
        T61[i] = _mm256_srai_epi16(_mm256_add_epi16(T29[i], T30[i]), 1);
        T62[i] = _mm256_srai_epi16(_mm256_add_epi16(T30[i], T31[i]), 1);
        T63[i] = _mm256_srai_epi16(_mm256_add_epi16(T31[i], T31[i]), 1);
    }

    /*--transposition--*/
    //32x64 -> 64x32
    TRANSPOSE_16x16_16BIT_m256i(T00[0], T32[0], T01[0], T33[0], T02[0], T34[0], T03[0], T35[0], T04[0], T36[0], T05[0], T37[0], T06[0], T38[0], T07[0], T39[0], V00[0], V01[0], V02[0], V03[0], V04[0], V05[0], V06[0], V07[0], V08[0], V09[0], V10[0], V11[0], V12[0], V13[0], V14[0], V15[0]);
    TRANSPOSE_16x16_16BIT_m256i(T08[0], T40[0], T09[0], T41[0], T10[0], T42[0], T11[0], T43[0], T12[0], T44[0], T13[0], T45[0], T14[0], T46[0], T15[0], T47[0], V00[1], V01[1], V02[1], V03[1], V04[1], V05[1], V06[1], V07[1], V08[1], V09[1], V10[1], V11[1], V12[1], V13[1], V14[1], V15[1]);
    TRANSPOSE_16x16_16BIT_m256i(T16[0], T48[0], T17[0], T49[0], T18[0], T50[0], T19[0], T51[0], T20[0], T52[0], T21[0], T53[0], T22[0], T54[0], T23[0], T55[0], V00[2], V01[2], V02[2], V03[2], V04[2], V05[2], V06[2], V07[2], V08[2], V09[2], V10[2], V11[2], V12[2], V13[2], V14[2], V15[2]);
    TRANSPOSE_16x16_16BIT_m256i(T24[0], T56[0], T25[0], T57[0], T26[0], T58[0], T27[0], T59[0], T28[0], T60[0], T29[0], T61[0], T30[0], T62[0], T31[0], T63[0], V00[3], V01[3], V02[3], V03[3], V04[3], V05[3], V06[3], V07[3], V08[3], V09[3], V10[3], V11[3], V12[3], V13[3], V14[3], V15[3]);

    TRANSPOSE_16x16_16BIT_m256i(T00[1], T32[1], T01[1], T33[1], T02[1], T34[1], T03[1], T35[1], T04[1], T36[1], T05[1], T37[1], T06[1], T38[1], T07[1], T39[1], V16[0], V17[0], V18[0], V19[0], V20[0], V21[0], V22[0], V23[0], V24[0], V25[0], V26[0], V27[0], V28[0], V29[0], V30[0], V31[0]);
    TRANSPOSE_16x16_16BIT_m256i(T08[1], T40[1], T09[1], T41[1], T10[1], T42[1], T11[1], T43[1], T12[1], T44[1], T13[1], T45[1], T14[1], T46[1], T15[1], T47[1], V16[1], V17[1], V18[1], V19[1], V20[1], V21[1], V22[1], V23[1], V24[1], V25[1], V26[1], V27[1], V28[1], V29[1], V30[1], V31[1]);
    TRANSPOSE_16x16_16BIT_m256i(T16[1], T48[1], T17[1], T49[1], T18[1], T50[1], T19[1], T51[1], T20[1], T52[1], T21[1], T53[1], T22[1], T54[1], T23[1], T55[1], V16[2], V17[2], V18[2], V19[2], V20[2], V21[2], V22[2], V23[2], V24[2], V25[2], V26[2], V27[2], V28[2], V29[2], V30[2], V31[2]);
    TRANSPOSE_16x16_16BIT_m256i(T24[1], T56[1], T25[1], T57[1], T26[1], T58[1], T27[1], T59[1], T28[1], T60[1], T29[1], T61[1], T30[1], T62[1], T31[1], T63[1], V16[3], V17[3], V18[3], V19[3], V20[3], V21[3], V22[3], V23[3], V24[3], V25[3], V26[3], V27[3], V28[3], V29[3], V30[3], V31[3]);

    /*--horizontal transform--*/
    //filter (odd pixel/column)
    for (i = 0; i < 4; i++) {
        V32[i] = _mm256_srai_epi16(_mm256_add_epi16(V00[i], V01[i]), 1);
        V33[i] = _mm256_srai_epi16(_mm256_add_epi16(V01[i], V02[i]), 1);
        V34[i] = _mm256_srai_epi16(_mm256_add_epi16(V02[i], V03[i]), 1);
        V35[i] = _mm256_srai_epi16(_mm256_add_epi16(V03[i], V04[i]), 1);
        V36[i] = _mm256_srai_epi16(_mm256_add_epi16(V04[i], V05[i]), 1);
        V37[i] = _mm256_srai_epi16(_mm256_add_epi16(V05[i], V06[i]), 1);
        V38[i] = _mm256_srai_epi16(_mm256_add_epi16(V06[i], V07[i]), 1);
        V39[i] = _mm256_srai_epi16(_mm256_add_epi16(V07[i], V08[i]), 1);
        V40[i] = _mm256_srai_epi16(_mm256_add_epi16(V08[i], V09[i]), 1);
        V41[i] = _mm256_srai_epi16(_mm256_add_epi16(V09[i], V10[i]), 1);
        V42[i] = _mm256_srai_epi16(_mm256_add_epi16(V10[i], V11[i]), 1);
        V43[i] = _mm256_srai_epi16(_mm256_add_epi16(V11[i], V12[i]), 1);
        V44[i] = _mm256_srai_epi16(_mm256_add_epi16(V12[i], V13[i]), 1);
        V45[i] = _mm256_srai_epi16(_mm256_add_epi16(V13[i], V14[i]), 1);
        V46[i] = _mm256_srai_epi16(_mm256_add_epi16(V14[i], V15[i]), 1);
        V47[i] = _mm256_srai_epi16(_mm256_add_epi16(V15[i], V16[i]), 1);

        V48[i] = _mm256_srai_epi16(_mm256_add_epi16(V16[i], V17[i]), 1);
        V49[i] = _mm256_srai_epi16(_mm256_add_epi16(V17[i], V18[i]), 1);
        V50[i] = _mm256_srai_epi16(_mm256_add_epi16(V18[i], V19[i]), 1);
        V51[i] = _mm256_srai_epi16(_mm256_add_epi16(V19[i], V20[i]), 1);
        V52[i] = _mm256_srai_epi16(_mm256_add_epi16(V20[i], V21[i]), 1);
        V53[i] = _mm256_srai_epi16(_mm256_add_epi16(V21[i], V22[i]), 1);
        V54[i] = _mm256_srai_epi16(_mm256_add_epi16(V22[i], V23[i]), 1);
        V55[i] = _mm256_srai_epi16(_mm256_add_epi16(V23[i], V24[i]), 1);
        V56[i] = _mm256_srai_epi16(_mm256_add_epi16(V24[i], V25[i]), 1);
        V57[i] = _mm256_srai_epi16(_mm256_add_epi16(V25[i], V26[i]), 1);
        V58[i] = _mm256_srai_epi16(_mm256_add_epi16(V26[i], V27[i]), 1);
        V59[i] = _mm256_srai_epi16(_mm256_add_epi16(V27[i], V28[i]), 1);
        V60[i] = _mm256_srai_epi16(_mm256_add_epi16(V28[i], V29[i]), 1);
        V61[i] = _mm256_srai_epi16(_mm256_add_epi16(V29[i], V30[i]), 1);
        V62[i] = _mm256_srai_epi16(_mm256_add_epi16(V30[i], V31[i]), 1);
        V63[i] = _mm256_srai_epi16(_mm256_add_epi16(V31[i], V31[i]), 1);
    }

    /*--transposition & Store--*/
    //64x64
    TRANSPOSE_16x16_16BIT_m256i(V00[0], V32[0], V01[0], V33[0], V02[0], V34[0], V03[0], V35[0], V04[0], V36[0], V05[0], V37[0], V06[0], V38[0], V07[0], V39[0], T00[0], T01[0], T02[0], T03[0], T04[0], T05[0], T06[0], T07[0], T08[0], T09[0], T10[0], T11[0], T12[0], T13[0], T14[0], T15[0]);
    TRANSPOSE_16x16_16BIT_m256i(V00[1], V32[1], V01[1], V33[1], V02[1], V34[1], V03[1], V35[1], V04[1], V36[1], V05[1], V37[1], V06[1], V38[1], V07[1], V39[1], T16[0], T17[0], T18[0], T19[0], T20[0], T21[0], T22[0], T23[0], T24[0], T25[0], T26[0], T27[0], T28[0], T29[0], T30[0], T31[0]);
    TRANSPOSE_16x16_16BIT_m256i(V00[2], V32[2], V01[2], V33[2], V02[2], V34[2], V03[2], V35[2], V04[2], V36[2], V05[2], V37[2], V06[2], V38[2], V07[2], V39[2], T32[0], T33[0], T34[0], T35[0], T36[0], T37[0], T38[0], T39[0], T40[0], T41[0], T42[0], T43[0], T44[0], T45[0], T46[0], T47[0]);
    TRANSPOSE_16x16_16BIT_m256i(V00[3], V32[3], V01[3], V33[3], V02[3], V34[3], V03[3], V35[3], V04[3], V36[3], V05[3], V37[3], V06[3], V38[3], V07[3], V39[3], T48[0], T49[0], T50[0], T51[0], T52[0], T53[0], T54[0], T55[0], T56[0], T57[0], T58[0], T59[0], T60[0], T61[0], T62[0], T63[0]);

    TRANSPOSE_16x16_16BIT_m256i(V08[0], V40[0], V09[0], V41[0], V10[0], V42[0], V11[0], V43[0], V12[0], V44[0], V13[0], V45[0], V14[0], V46[0], V15[0], V47[0], T00[1], T01[1], T02[1], T03[1], T04[1], T05[1], T06[1], T07[1], T08[1], T09[1], T10[1], T11[1], T12[1], T13[1], T14[1], T15[1]);
    TRANSPOSE_16x16_16BIT_m256i(V08[1], V40[1], V09[1], V41[1], V10[1], V42[1], V11[1], V43[1], V12[1], V44[1], V13[1], V45[1], V14[1], V46[1], V15[1], V47[1], T16[1], T17[1], T18[1], T19[1], T20[1], T21[1], T22[1], T23[1], T24[1], T25[1], T26[1], T27[1], T28[1], T29[1], T30[1], T31[1]);
    TRANSPOSE_16x16_16BIT_m256i(V08[2], V40[2], V09[2], V41[2], V10[2], V42[2], V11[2], V43[2], V12[2], V44[2], V13[2], V45[2], V14[2], V46[2], V15[2], V47[2], T32[1], T33[1], T34[1], T35[1], T36[1], T37[1], T38[1], T39[1], T40[1], T41[1], T42[1], T43[1], T44[1], T45[1], T46[1], T47[1]);
    TRANSPOSE_16x16_16BIT_m256i(V08[3], V40[3], V09[3], V41[3], V10[3], V42[3], V11[3], V43[3], V12[3], V44[3], V13[3], V45[3], V14[3], V46[3], V15[3], V47[3], T48[1], T49[1], T50[1], T51[1], T52[1], T53[1], T54[1], T55[1], T56[1], T57[1], T58[1], T59[1], T60[1], T61[1], T62[1], T63[1]);

    TRANSPOSE_16x16_16BIT_m256i(V16[0], V48[0], V17[0], V49[0], V18[0], V50[0], V19[0], V51[0], V20[0], V52[0], V21[0], V53[0], V22[0], V54[0], V23[0], V55[0], T00[2], T01[2], T02[2], T03[2], T04[2], T05[2], T06[2], T07[2], T08[2], T09[2], T10[2], T11[2], T12[2], T13[2], T14[2], T15[2]);
    TRANSPOSE_16x16_16BIT_m256i(V16[1], V48[1], V17[1], V49[1], V18[1], V50[1], V19[1], V51[1], V20[1], V52[1], V21[1], V53[1], V22[1], V54[1], V23[1], V55[1], T16[2], T17[2], T18[2], T19[2], T20[2], T21[2], T22[2], T23[2], T24[2], T25[2], T26[2], T27[2], T28[2], T29[2], T30[2], T31[2]);
    TRANSPOSE_16x16_16BIT_m256i(V16[2], V48[2], V17[2], V49[2], V18[2], V50[2], V19[2], V51[2], V20[2], V52[2], V21[2], V53[2], V22[2], V54[2], V23[2], V55[2], T32[2], T33[2], T34[2], T35[2], T36[2], T37[2], T38[2], T39[2], T40[2], T41[2], T42[2], T43[2], T44[2], T45[2], T46[2], T47[2]);
    TRANSPOSE_16x16_16BIT_m256i(V16[3], V48[3], V17[3], V49[3], V18[3], V50[3], V19[3], V51[3], V20[3], V52[3], V21[3], V53[3], V22[3], V54[3], V23[3], V55[3], T48[2], T49[2], T50[2], T51[2], T52[2], T53[2], T54[2], T55[2], T56[2], T57[2], T58[2], T59[2], T60[2], T61[2], T62[2], T63[2]);

    TRANSPOSE_16x16_16BIT_m256i(V24[0], V56[0], V25[0], V57[0], V26[0], V58[0], V27[0], V59[0], V28[0], V60[0], V29[0], V61[0], V30[0], V62[0], V31[0], V63[0], T00[3], T01[3], T02[3], T03[3], T04[3], T05[3], T06[3], T07[3], T08[3], T09[3], T10[3], T11[3], T12[3], T13[3], T14[3], T15[3]);
    TRANSPOSE_16x16_16BIT_m256i(V24[1], V56[1], V25[1], V57[1], V26[1], V58[1], V27[1], V59[1], V28[1], V60[1], V29[1], V61[1], V30[1], V62[1], V31[1], V63[1], T16[3], T17[3], T18[3], T19[3], T20[3], T21[3], T22[3], T23[3], T24[3], T25[3], T26[3], T27[3], T28[3], T29[3], T30[3], T31[3]);
    TRANSPOSE_16x16_16BIT_m256i(V24[2], V56[2], V25[2], V57[2], V26[2], V58[2], V27[2], V59[2], V28[2], V60[2], V29[2], V61[2], V30[2], V62[2], V31[2], V63[2], T32[3], T33[3], T34[3], T35[3], T36[3], T37[3], T38[3], T39[3], T40[3], T41[3], T42[3], T43[3], T44[3], T45[3], T46[3], T47[3]);
    TRANSPOSE_16x16_16BIT_m256i(V24[3], V56[3], V25[3], V57[3], V26[3], V58[3], V27[3], V59[3], V28[3], V60[3], V29[3], V61[3], V30[3], V62[3], V31[3], V63[3], T48[3], T49[3], T50[3], T51[3], T52[3], T53[3], T54[3], T55[3], T56[3], T57[3], T58[3], T59[3], T60[3], T61[3], T62[3], T63[3]);

    //store
    for (i = 0; i < 4; i++) {
        _mm256_storeu_si256((__m256i*)&coeff[16 * i], T00[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64], T01[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 2], T02[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 3], T03[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 4], T04[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 5], T05[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 6], T06[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 7], T07[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 8], T08[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 9], T09[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 10], T10[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 11], T11[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 12], T12[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 13], T13[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 14], T14[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 15], T15[i]);

        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 16], T16[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 17], T17[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 18], T18[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 19], T19[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 20], T20[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 21], T21[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 22], T22[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 23], T23[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 24], T24[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 25], T25[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 26], T26[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 27], T27[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 28], T28[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 29], T29[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 30], T30[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 31], T31[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 32], T32[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 33], T33[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 34], T34[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 35], T35[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 36], T36[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 37], T37[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 38], T38[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 39], T39[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 40], T40[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 41], T41[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 42], T42[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 43], T43[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 44], T44[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 45], T45[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 46], T46[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 47], T47[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 48], T48[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 49], T49[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 50], T50[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 51], T51[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 52], T52[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 53], T53[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 54], T54[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 55], T55[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 56], T56[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 57], T57[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 58], T58[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 59], T59[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 60], T60[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 61], T61[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 62], T62[i]);
        _mm256_storeu_si256((__m256i*)&coeff[16 * i + 64 * 63], T63[i]);
    }
}


/* ---------------------------------------------------------------------------
*/
void idct_c_64x64_avx2(const coeff_t *src, coeff_t *dst, int i_dst)
{
    UNUSED_PARAMETER(i_dst);
    idct_c_32x32_avx2(src, dst, 32 | 0x01);
    inv_wavelet_64x64_avx2(dst);
}

/* ---------------------------------------------------------------------------
*/
void idct_c_64x16_avx2(const coeff_t *src, coeff_t *dst, int i_dst)
{
    UNUSED_PARAMETER(i_dst);
    idct_c_32x8_sse128(src, dst, 32 | 0x01);
    inv_wavelet_64x16_avx2(dst);
}

/* ---------------------------------------------------------------------------
*/
void idct_c_16x64_avx2(const coeff_t *src, coeff_t *dst, int i_dst)
{
    UNUSED_PARAMETER(i_dst);
    idct_c_8x32_sse128(src, dst, 8 | 0x01);
    inv_wavelet_16x64_avx2(dst);
}

