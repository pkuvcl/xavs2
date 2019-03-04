/*
 * intrinsic_idct.c
 *
 * Description of this file:
 *    SSE assembly functions of IDCT module of the xavs2 library
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
#include "../avs2_defs.h"
#include "intrinsic.h"

#include <mmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>


ALIGN32(static const coeff_t tab_idct_8x8[12][8]) = {
    {  44,  38,  44,  38,  44,  38,  44,  38 },
    {  25,   9,  25,   9,  25,   9,  25,   9 },
    {  38,  -9,  38,  -9,  38,  -9,  38,  -9 },
    { -44, -25, -44, -25, -44, -25, -44, -25 },
    {  25, -44,  25, -44,  25, -44,  25, -44 },
    {   9,  38,   9,  38,   9,  38,   9,  38 },
    {   9, -25,   9, -25,   9, -25,   9, -25 },
    {  38, -44,  38, -44,  38, -44,  38, -44 },
    {  32,  32,  32,  32,  32,  32,  32,  32 },
    {  32, -32,  32, -32,  32, -32,  32, -32 },
    {  42,  17,  42,  17,  42,  17,  42,  17 },
    {  17, -42,  17, -42,  17, -42,  17, -42 }
};

extern ALIGN16(const int16_t g_2T  [SEC_TR_SIZE * SEC_TR_SIZE]);
extern ALIGN16(const int16_t g_2T_C[SEC_TR_SIZE * SEC_TR_SIZE]);


/* ---------------------------------------------------------------------------
 */
void idct_c_4x4_sse128(const coeff_t *src, coeff_t *dst, int i_dst)
{
    const int shift1 = 5;
    const int shift2 = 20 - g_bit_depth;
    // const int clip_depth1 = LIMIT_BIT;
    const int clip_depth2 = g_bit_depth + 1;

    const __m128i c16_p17_p42 = _mm_set1_epi32(0x0011002A);
    const __m128i c16_n42_p17 = _mm_set1_epi32(0xFFD60011);
    const __m128i c16_n32_p32 = _mm_set1_epi32(0xFFE00020);
    const __m128i c16_p32_p32 = _mm_set1_epi32(0x00200020);

    __m128i c32_rnd = _mm_set1_epi32(1 << (shift1 - 1));    // add1
    __m128i S0, S1;
    __m128i T0, T1;
    __m128i E0, E1, O0, O1;

    S0  = _mm_loadu_si128((__m128i*)(src   ));
    S1  = _mm_loadu_si128((__m128i*)(src+ 8));

    T0 = _mm_unpacklo_epi16(S0, S1);
    E0 = _mm_add_epi32(_mm_madd_epi16(T0, c16_p32_p32), c32_rnd);
    E1 = _mm_add_epi32(_mm_madd_epi16(T0, c16_n32_p32), c32_rnd);

    T1 = _mm_unpackhi_epi16(S0, S1);
    O0 = _mm_madd_epi16(T1, c16_p17_p42);
    O1 = _mm_madd_epi16(T1, c16_n42_p17);

    S0 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E0, O0), shift1), _mm_srai_epi32(_mm_sub_epi32(E1, O1), shift1));
    S1 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E1, O1), shift1), _mm_srai_epi32(_mm_sub_epi32(E0, O0), shift1));

    /* inverse */
    T0 = _mm_unpacklo_epi16(S0, S1);
    T1 = _mm_unpackhi_epi16(S0, S1);
    S0 = _mm_unpacklo_epi32(T0, T1);
    S1 = _mm_unpackhi_epi32(T0, T1);

    /* second pass -------------------------------------------------
     */
    c32_rnd  = _mm_set1_epi32(1 << (shift2 - 1));    // add2

    T0 = _mm_unpacklo_epi16(S0, S1);
    E0 = _mm_add_epi32(_mm_madd_epi16(T0, c16_p32_p32), c32_rnd);
    E1 = _mm_add_epi32(_mm_madd_epi16(T0, c16_n32_p32), c32_rnd);

    T1 = _mm_unpackhi_epi16(S0, S1);
    O0 = _mm_madd_epi16(T1, c16_p17_p42);
    O1 = _mm_madd_epi16(T1, c16_n42_p17);

    S0  = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E0, O0), shift2), _mm_srai_epi32(_mm_sub_epi32(E1, O1), shift2));
    S1  = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E1, O1), shift2), _mm_srai_epi32(_mm_sub_epi32(E0, O0), shift2));

    T0 = _mm_unpacklo_epi16(S0, S1);
    T1 = _mm_unpackhi_epi16(S0, S1);
    S0 = _mm_unpacklo_epi32(T0, T1);
    S1 = _mm_unpackhi_epi32(T0, T1);

    // clip
    {
        const __m128i max_val = _mm_set1_epi16((1 << (clip_depth2 - 1)) - 1);
        const __m128i min_val = _mm_set1_epi16(-(1 << (clip_depth2 - 1)));

        S0 = _mm_max_epi16(_mm_min_epi16(S0, max_val), min_val);
        S1 = _mm_max_epi16(_mm_min_epi16(S1, max_val), min_val);
    }

    // store
    if (i_dst == 4) {
        _mm_store_si128((__m128i*)(dst + 0), S0);
        _mm_store_si128((__m128i*)(dst + 8), S1);
    } else {
        _mm_storel_epi64((__m128i*)(dst + 0 * i_dst), S0);
        _mm_storeh_pi((__m64  *)(dst + 1 * i_dst), _mm_castsi128_ps(S0));
        _mm_storel_epi64((__m128i*)(dst + 2 * i_dst), S1);
        _mm_storeh_pi((__m64  *)(dst + 3 * i_dst), _mm_castsi128_ps(S1));
    }
}

/* ---------------------------------------------------------------------------
 */
void idct_c_4x16_sse128(const coeff_t *src, coeff_t *dst, int i_dst)
{
    const int shift1 = 5;
    const int shift2 = 20 - g_bit_depth;
    // const int clip_depth1 = LIMIT_BIT;
    const int clip_depth2 = g_bit_depth + 1;

    const __m128i c16_p43_p45 = _mm_set1_epi32(0x002B002D);   //row0 87high - 90low address
    const __m128i c16_p35_p40 = _mm_set1_epi32(0x00230028);
    const __m128i c16_p21_p29 = _mm_set1_epi32(0x0015001D);
    const __m128i c16_p04_p13 = _mm_set1_epi32(0x0004000D);
    const __m128i c16_p29_p43 = _mm_set1_epi32(0x001D002B);   //row1
    const __m128i c16_n21_p04 = _mm_set1_epi32(0xFFEB0004);
    const __m128i c16_n45_n40 = _mm_set1_epi32(0xFFD3FFD8);
    const __m128i c16_n13_n35 = _mm_set1_epi32(0xFFF3FFDD);
    const __m128i c16_p04_p40 = _mm_set1_epi32(0x00040028);   //row2
    const __m128i c16_n43_n35 = _mm_set1_epi32(0xFFD5FFDD);
    const __m128i c16_p29_n13 = _mm_set1_epi32(0x001DFFF3);
    const __m128i c16_p21_p45 = _mm_set1_epi32(0x0015002D);
    const __m128i c16_n21_p35 = _mm_set1_epi32(0xFFEB0023);   //row3
    const __m128i c16_p04_n43 = _mm_set1_epi32(0x0004FFD5);
    const __m128i c16_p13_p45 = _mm_set1_epi32(0x000D002D);
    const __m128i c16_n29_n40 = _mm_set1_epi32(0xFFE3FFD8);
    const __m128i c16_n40_p29 = _mm_set1_epi32(0xFFD8001D);   //row4
    const __m128i c16_p45_n13 = _mm_set1_epi32(0x002DFFF3);
    const __m128i c16_n43_n04 = _mm_set1_epi32(0xFFD5FFFC);
    const __m128i c16_p35_p21 = _mm_set1_epi32(0x00230015);
    const __m128i c16_n45_p21 = _mm_set1_epi32(0xFFD30015);   //row5
    const __m128i c16_p13_p29 = _mm_set1_epi32(0x000D001D);
    const __m128i c16_p35_n43 = _mm_set1_epi32(0x0023FFD5);
    const __m128i c16_n40_p04 = _mm_set1_epi32(0xFFD80004);
    const __m128i c16_n35_p13 = _mm_set1_epi32(0xFFDD000D);   //row6
    const __m128i c16_n40_p45 = _mm_set1_epi32(0xFFD8002D);
    const __m128i c16_p04_p21 = _mm_set1_epi32(0x00040015);
    const __m128i c16_p43_n29 = _mm_set1_epi32(0x002BFFE3);
    const __m128i c16_n13_p04 = _mm_set1_epi32(0xFFF30004);   //row7
    const __m128i c16_n29_p21 = _mm_set1_epi32(0xFFE30015);
    const __m128i c16_n40_p35 = _mm_set1_epi32(0xFFD80023);
    const __m128i c16_n45_p43 = _mm_set1_epi32(0xFFD3002B);

    const __m128i c16_p38_p44 = _mm_set1_epi32(0x0026002C);
    const __m128i c16_p09_p25 = _mm_set1_epi32(0x00090019);
    const __m128i c16_n09_p38 = _mm_set1_epi32(0xFFF70026);
    const __m128i c16_n25_n44 = _mm_set1_epi32(0xFFE7FFD4);
    const __m128i c16_n44_p25 = _mm_set1_epi32(0xFFD40019);
    const __m128i c16_p38_p09 = _mm_set1_epi32(0x00260009);
    const __m128i c16_n25_p09 = _mm_set1_epi32(0xFFE70009);
    const __m128i c16_n44_p38 = _mm_set1_epi32(0xFFD40026);

    const __m128i c16_p17_p42 = _mm_set1_epi32(0x0011002A);
    const __m128i c16_n42_p17 = _mm_set1_epi32(0xFFD60011);

    const __m128i c16_n32_p32 = _mm_set1_epi32(0xFFE00020);
    const __m128i c16_p32_p32 = _mm_set1_epi32(0x00200020);

    __m128i c32_rnd = _mm_set1_epi32(1 << (shift1 - 1));            // add1

    // DCT1
    __m128i in00, in01, in02, in03, in04, in05, in06, in07;
    __m128i res00, res01, res02, res03, res04, res05, res06, res07;

    in00 = _mm_loadu_si128((const __m128i*)&src[ 0 * 4]);           // [07 06 05 04 03 02 01 00]
    in01 = _mm_loadu_si128((const __m128i*)&src[ 2 * 4]);           // [27 26 25 24 23 22 21 20]
    in02 = _mm_loadu_si128((const __m128i*)&src[ 4 * 4]);           // [47 46 45 44 43 42 41 40]
    in03 = _mm_loadu_si128((const __m128i*)&src[ 6 * 4]);           // [67 66 65 64 63 62 61 60]
    in04 = _mm_loadu_si128((const __m128i*)&src[ 8 * 4]);
    in05 = _mm_loadu_si128((const __m128i*)&src[10 * 4]);
    in06 = _mm_loadu_si128((const __m128i*)&src[12 * 4]);
    in07 = _mm_loadu_si128((const __m128i*)&src[14 * 4]);

    {
        const __m128i T_00_00A = _mm_unpackhi_epi16(in00, in01);    // [33 13 32 12 31 11 30 10]
        const __m128i T_00_01A = _mm_unpackhi_epi16(in02, in03);    // [ ]
        const __m128i T_00_02A = _mm_unpackhi_epi16(in04, in05);    // [ ]
        const __m128i T_00_03A = _mm_unpackhi_epi16(in06, in07);    // [ ]
        const __m128i T_00_04A = _mm_unpacklo_epi16(in01, in03);    // [ ]
        const __m128i T_00_05A = _mm_unpacklo_epi16(in05, in07);    // [ ]
        const __m128i T_00_06A = _mm_unpacklo_epi16(in02, in06);    // [ ]row
        const __m128i T_00_07A = _mm_unpacklo_epi16(in00, in04);    // [83 03 82 02 81 01 81 00] row08 row00

        __m128i O0A, O1A, O2A, O3A, O4A, O5A, O6A, O7A;
        __m128i EO0A, EO1A, EO2A, EO3A;
        __m128i EEO0A, EEO1A;
        __m128i EEE0A, EEE1A;

#define COMPUTE_ROW(row0103, row0507, row0911, row1315, c0103, c0507, c0911, c1315, row) \
    row = _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(row0103, c0103), _mm_madd_epi16(row0507, c0507)), \
                        _mm_add_epi32(_mm_madd_epi16(row0911, c0911), _mm_madd_epi16(row1315, c1315)));

        COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_p43_p45, c16_p35_p40, c16_p21_p29, c16_p04_p13, O0A)
        COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_p29_p43, c16_n21_p04, c16_n45_n40, c16_n13_n35, O1A)
        COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_p04_p40, c16_n43_n35, c16_p29_n13, c16_p21_p45, O2A)
        COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n21_p35, c16_p04_n43, c16_p13_p45, c16_n29_n40, O3A)
        COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n40_p29, c16_p45_n13, c16_n43_n04, c16_p35_p21, O4A)
        COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n45_p21, c16_p13_p29, c16_p35_n43, c16_n40_p04, O5A)
        COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n35_p13, c16_n40_p45, c16_p04_p21, c16_p43_n29, O6A)
        COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n13_p04, c16_n29_p21, c16_n40_p35, c16_n45_p43, O7A)
#undef COMPUTE_ROW

        EO0A = _mm_add_epi32(_mm_madd_epi16(T_00_04A, c16_p38_p44), _mm_madd_epi16(T_00_05A, c16_p09_p25)); // EO0
        EO1A = _mm_add_epi32(_mm_madd_epi16(T_00_04A, c16_n09_p38), _mm_madd_epi16(T_00_05A, c16_n25_n44)); // EO1
        EO2A = _mm_add_epi32(_mm_madd_epi16(T_00_04A, c16_n44_p25), _mm_madd_epi16(T_00_05A, c16_p38_p09)); // EO2
        EO3A = _mm_add_epi32(_mm_madd_epi16(T_00_04A, c16_n25_p09), _mm_madd_epi16(T_00_05A, c16_n44_p38)); // EO3

        EEO0A = _mm_madd_epi16(T_00_06A, c16_p17_p42);
        EEO1A = _mm_madd_epi16(T_00_06A, c16_n42_p17);

        EEE0A = _mm_madd_epi16(T_00_07A, c16_p32_p32);
        EEE1A = _mm_madd_epi16(T_00_07A, c16_n32_p32);
        {
            const __m128i EE0A = _mm_add_epi32(EEE0A, EEO0A);   // EE0 = EEE0 + EEO0
            const __m128i EE1A = _mm_add_epi32(EEE1A, EEO1A);   // EE1 = EEE1 + EEO1
            const __m128i EE3A = _mm_sub_epi32(EEE0A, EEO0A);   // EE2 = EEE0 - EEO0
            const __m128i EE2A = _mm_sub_epi32(EEE1A, EEO1A);   // EE3 = EEE1 - EEO1

            const __m128i T10A = _mm_add_epi32(_mm_add_epi32(EE0A, EO0A), c32_rnd);   // E0 (= EE0 + EO0) + rnd
            const __m128i T11A = _mm_add_epi32(_mm_add_epi32(EE1A, EO1A), c32_rnd);   // E1 (= EE1 + EO1) + rnd
            const __m128i T12A = _mm_add_epi32(_mm_add_epi32(EE2A, EO2A), c32_rnd);   // E2 (= EE2 + EO2) + rnd
            const __m128i T13A = _mm_add_epi32(_mm_add_epi32(EE3A, EO3A), c32_rnd);   // E3 (= EE3 + EO3) + rnd
            const __m128i T14A = _mm_add_epi32(_mm_sub_epi32(EE3A, EO3A), c32_rnd);   // E4 (= EE3 - EO3) + rnd
            const __m128i T15A = _mm_add_epi32(_mm_sub_epi32(EE2A, EO2A), c32_rnd);   // E5 (= EE2 - EO2) + rnd
            const __m128i T16A = _mm_add_epi32(_mm_sub_epi32(EE1A, EO1A), c32_rnd);   // E6 (= EE1 - EO1) + rnd
            const __m128i T17A = _mm_add_epi32(_mm_sub_epi32(EE0A, EO0A), c32_rnd);   // E7 (= EE0 - EO0) + rnd


            const __m128i T30A = _mm_srai_epi32(_mm_add_epi32(T10A, O0A), shift1);  // E0 + O0 + rnd   [30 20 10 00]
            const __m128i T31A = _mm_srai_epi32(_mm_add_epi32(T11A, O1A), shift1);  // E1 + O1 + rnd   [31 21 11 01]
            const __m128i T32A = _mm_srai_epi32(_mm_add_epi32(T12A, O2A), shift1);  // E2 + O2 + rnd   [32 22 12 02]
            const __m128i T33A = _mm_srai_epi32(_mm_add_epi32(T13A, O3A), shift1);  // E3 + O3 + rnd   [33 23 13 03]
            const __m128i T34A = _mm_srai_epi32(_mm_add_epi32(T14A, O4A), shift1);  // E4              [33 24 14 04]
            const __m128i T35A = _mm_srai_epi32(_mm_add_epi32(T15A, O5A), shift1);  // E5              [35 25 15 05]
            const __m128i T36A = _mm_srai_epi32(_mm_add_epi32(T16A, O6A), shift1);  // E6              [36 26 16 06]
            const __m128i T37A = _mm_srai_epi32(_mm_add_epi32(T17A, O7A), shift1);  // E7              [37 27 17 07]

            const __m128i T38A = _mm_srai_epi32(_mm_sub_epi32(T17A, O7A), shift1);  // E7             [30 20 10 00] x8
            const __m128i T39A = _mm_srai_epi32(_mm_sub_epi32(T16A, O6A), shift1);  // E6             [31 21 11 01] x9
            const __m128i T3AA = _mm_srai_epi32(_mm_sub_epi32(T15A, O5A), shift1);  // E5             [32 22 12 02] xA
            const __m128i T3BA = _mm_srai_epi32(_mm_sub_epi32(T14A, O4A), shift1);  // E4             [33 23 13 03] xB
            const __m128i T3CA = _mm_srai_epi32(_mm_sub_epi32(T13A, O3A), shift1);  // E3 - O3 + rnd  [33 24 14 04] xC
            const __m128i T3DA = _mm_srai_epi32(_mm_sub_epi32(T12A, O2A), shift1);  // E2 - O2 + rnd  [35 25 15 05] xD
            const __m128i T3EA = _mm_srai_epi32(_mm_sub_epi32(T11A, O1A), shift1);  // E1 - O1 + rnd  [36 26 16 06] xE
            const __m128i T3FA = _mm_srai_epi32(_mm_sub_epi32(T10A, O0A), shift1);  // E0 - O0 + rnd  [37 27 17 07] xF

            res00 = _mm_packs_epi32(T30A, T38A);
            res01 = _mm_packs_epi32(T31A, T39A);
            res02 = _mm_packs_epi32(T32A, T3AA);
            res03 = _mm_packs_epi32(T33A, T3BA);

            res04 = _mm_packs_epi32(T34A, T3CA);
            res05 = _mm_packs_epi32(T35A, T3DA);
            res06 = _mm_packs_epi32(T36A, T3EA);
            res07 = _mm_packs_epi32(T37A, T3FA);
        }
    }

    // transpose matrix
    {
        __m128i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
        __m128i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;
        __m128i E01, E02, E03, E04, E11, E12, E13, E14;
        __m128i O01, O02, O03, O04, O11, O12, O13, O14;
        __m128i m128Tmp0, m128Tmp1, m128Tmp2, m128Tmp3;

        tr0_0 = _mm_unpacklo_epi16(res00, res01);
        tr0_1 = _mm_unpackhi_epi16(res00, res01);
        tr0_2 = _mm_unpacklo_epi16(res02, res03);
        tr0_3 = _mm_unpackhi_epi16(res02, res03);
        tr0_4 = _mm_unpacklo_epi16(res04, res05);
        tr0_5 = _mm_unpackhi_epi16(res04, res05);
        tr0_6 = _mm_unpacklo_epi16(res06, res07);
        tr0_7 = _mm_unpackhi_epi16(res06, res07);

        tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_2);
        tr1_1 = _mm_unpackhi_epi32(tr0_0, tr0_2);
        tr1_2 = _mm_unpacklo_epi32(tr0_1, tr0_3);
        tr1_3 = _mm_unpackhi_epi32(tr0_1, tr0_3);
        tr1_4 = _mm_unpacklo_epi32(tr0_4, tr0_6);
        tr1_5 = _mm_unpackhi_epi32(tr0_4, tr0_6);
        tr1_6 = _mm_unpacklo_epi32(tr0_5, tr0_7);
        tr1_7 = _mm_unpackhi_epi32(tr0_5, tr0_7);

        res00 = _mm_unpacklo_epi64(tr1_0, tr1_4);
        res02 = _mm_unpackhi_epi64(tr1_0, tr1_4);
        res04 = _mm_unpacklo_epi64(tr1_1, tr1_5);
        res06 = _mm_unpackhi_epi64(tr1_1, tr1_5);
        res01 = _mm_unpacklo_epi64(tr1_2, tr1_6);
        res03 = _mm_unpackhi_epi64(tr1_2, tr1_6);
        res05 = _mm_unpacklo_epi64(tr1_3, tr1_7);
        res07 = _mm_unpackhi_epi64(tr1_3, tr1_7);

        c32_rnd = _mm_set1_epi32(1 << (shift2 - 1));    // add2

        m128Tmp0 = _mm_unpacklo_epi16(res00, res04);
        E01 = _mm_add_epi32(_mm_madd_epi16(m128Tmp0, c16_p32_p32), c32_rnd);
        E11 = _mm_add_epi32(_mm_madd_epi16(m128Tmp0, c16_n32_p32), c32_rnd);

        m128Tmp1 = _mm_unpackhi_epi16(res00, res04);
        E02 = _mm_add_epi32(_mm_madd_epi16(m128Tmp1, c16_p32_p32), c32_rnd);
        E12 = _mm_add_epi32(_mm_madd_epi16(m128Tmp1, c16_n32_p32), c32_rnd);

        m128Tmp0 = _mm_unpacklo_epi16(res01, res05);
        E03 = _mm_add_epi32(_mm_madd_epi16(m128Tmp0, c16_p32_p32), c32_rnd);
        E13 = _mm_add_epi32(_mm_madd_epi16(m128Tmp0, c16_n32_p32), c32_rnd);

        m128Tmp1 = _mm_unpackhi_epi16(res01, res05);
        E04 = _mm_add_epi32(_mm_madd_epi16(m128Tmp1, c16_p32_p32), c32_rnd);
        E14 = _mm_add_epi32(_mm_madd_epi16(m128Tmp1, c16_n32_p32), c32_rnd);

        m128Tmp0 = _mm_unpacklo_epi16(res02, res06);
        O01 = _mm_madd_epi16(m128Tmp0, c16_p17_p42);
        O11 = _mm_madd_epi16(m128Tmp0, c16_n42_p17);

        m128Tmp1 = _mm_unpackhi_epi16(res02, res06);
        O02 = _mm_madd_epi16(m128Tmp1, c16_p17_p42);
        O12 = _mm_madd_epi16(m128Tmp1, c16_n42_p17);

        m128Tmp0 = _mm_unpacklo_epi16(res03, res07);
        O03 = _mm_madd_epi16(m128Tmp0, c16_p17_p42);
        O13 = _mm_madd_epi16(m128Tmp0, c16_n42_p17);

        m128Tmp1 = _mm_unpackhi_epi16(res03, res07);
        O04 = _mm_madd_epi16(m128Tmp1, c16_p17_p42);
        O14 = _mm_madd_epi16(m128Tmp1, c16_n42_p17);

        res00 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E01, O01), shift2), _mm_srai_epi32(_mm_add_epi32(E02, O02), shift2));
        res01 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E03, O03), shift2), _mm_srai_epi32(_mm_add_epi32(E04, O04), shift2));

        res06 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E01, O01), shift2), _mm_srai_epi32(_mm_sub_epi32(E02, O02), shift2));
        res07 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E03, O03), shift2), _mm_srai_epi32(_mm_sub_epi32(E04, O04), shift2));

        res02 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E11, O11), shift2), _mm_srai_epi32(_mm_add_epi32(E12, O12), shift2));
        res03 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E13, O13), shift2), _mm_srai_epi32(_mm_add_epi32(E14, O14), shift2));

        res04 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E11, O11), shift2), _mm_srai_epi32(_mm_sub_epi32(E12, O12), shift2));
        res05 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E13, O13), shift2), _mm_srai_epi32(_mm_sub_epi32(E14, O14), shift2));

        m128Tmp0 = _mm_unpacklo_epi16(res00, res02);
        m128Tmp1 = _mm_unpackhi_epi16(res00, res02);
        m128Tmp2 = _mm_unpacklo_epi16(res04, res06);
        m128Tmp3 = _mm_unpackhi_epi16(res04, res06);

        res00 = _mm_unpacklo_epi32(m128Tmp0, m128Tmp2);
        res02 = _mm_unpackhi_epi32(m128Tmp0, m128Tmp2);
        res04 = _mm_unpacklo_epi32(m128Tmp1, m128Tmp3);
        res06 = _mm_unpackhi_epi32(m128Tmp1, m128Tmp3);

        m128Tmp0 = _mm_unpacklo_epi16(res01, res03);
        m128Tmp1 = _mm_unpackhi_epi16(res01, res03);
        m128Tmp2 = _mm_unpacklo_epi16(res05, res07);
        m128Tmp3 = _mm_unpackhi_epi16(res05, res07);

        res01 = _mm_unpacklo_epi32(m128Tmp0, m128Tmp2);
        res03 = _mm_unpackhi_epi32(m128Tmp0, m128Tmp2);
        res05 = _mm_unpacklo_epi32(m128Tmp1, m128Tmp3);
        res07 = _mm_unpackhi_epi32(m128Tmp1, m128Tmp3);
    }

    // clip
    {
        const __m128i max_val = _mm_set1_epi16((1 << (clip_depth2 - 1)) - 1);
        const __m128i min_val = _mm_set1_epi16(-(1 << (clip_depth2 - 1)));

        res00 = _mm_max_epi16(_mm_min_epi16(res00, max_val), min_val);
        res02 = _mm_max_epi16(_mm_min_epi16(res02, max_val), min_val);
        res04 = _mm_max_epi16(_mm_min_epi16(res04, max_val), min_val);
        res06 = _mm_max_epi16(_mm_min_epi16(res06, max_val), min_val);
        res01 = _mm_max_epi16(_mm_min_epi16(res01, max_val), min_val);
        res03 = _mm_max_epi16(_mm_min_epi16(res03, max_val), min_val);
        res05 = _mm_max_epi16(_mm_min_epi16(res05, max_val), min_val);
        res07 = _mm_max_epi16(_mm_min_epi16(res07, max_val), min_val);
    }

    // store
    if (i_dst == 4) {
        _mm_store_si128((__m128i*)(dst +  0 * 4), res00);
        _mm_store_si128((__m128i*)(dst +  2 * 4), res02);
        _mm_store_si128((__m128i*)(dst +  4 * 4), res04);
        _mm_store_si128((__m128i*)(dst +  6 * 4), res06);
        _mm_store_si128((__m128i*)(dst +  8 * 4), res01);
        _mm_store_si128((__m128i*)(dst + 10 * 4), res03);
        _mm_store_si128((__m128i*)(dst + 12 * 4), res05);
        _mm_store_si128((__m128i*)(dst + 14 * 4), res07);
    } else {
        _mm_storel_epi64((__m128i*)(dst +  0 * i_dst), res00);
        _mm_storeh_pi   ((__m64  *)(dst +  1 * i_dst), _mm_castsi128_ps(res00));
        _mm_storel_epi64((__m128i*)(dst +  2 * i_dst), res02);
        _mm_storeh_pi   ((__m64  *)(dst +  3 * i_dst), _mm_castsi128_ps(res02));
        _mm_storel_epi64((__m128i*)(dst +  4 * i_dst), res04);
        _mm_storeh_pi   ((__m64  *)(dst +  5 * i_dst), _mm_castsi128_ps(res04));
        _mm_storel_epi64((__m128i*)(dst +  6 * i_dst), res06);
        _mm_storeh_pi   ((__m64  *)(dst +  7 * i_dst), _mm_castsi128_ps(res06));
        _mm_storel_epi64((__m128i*)(dst +  8 * i_dst), res01);
        _mm_storeh_pi   ((__m64  *)(dst +  9 * i_dst), _mm_castsi128_ps(res01));
        _mm_storel_epi64((__m128i*)(dst + 10 * i_dst), res03);
        _mm_storeh_pi   ((__m64  *)(dst + 11 * i_dst), _mm_castsi128_ps(res03));
        _mm_storel_epi64((__m128i*)(dst + 12 * i_dst), res05);
        _mm_storeh_pi   ((__m64  *)(dst + 13 * i_dst), _mm_castsi128_ps(res05));
        _mm_storel_epi64((__m128i*)(dst + 14 * i_dst), res07);
        _mm_storeh_pi   ((__m64  *)(dst + 15 * i_dst), _mm_castsi128_ps(res07));
    }
}

/* ---------------------------------------------------------------------------
 */
void idct_c_16x4_sse128(const coeff_t *src, coeff_t *dst, int i_dst)
{
    const int shift1 = 5;
    const int shift2 = 20 - g_bit_depth;
    // const int clip_depth1 = LIMIT_BIT;
    const int clip_depth2 = g_bit_depth + 1;

    const __m128i c16_p43_p45 = _mm_set1_epi32(0x002B002D);   //row0 87high - 90low address
    const __m128i c16_p35_p40 = _mm_set1_epi32(0x00230028);
    const __m128i c16_p21_p29 = _mm_set1_epi32(0x0015001D);
    const __m128i c16_p04_p13 = _mm_set1_epi32(0x0004000D);
    const __m128i c16_p29_p43 = _mm_set1_epi32(0x001D002B);   //row1
    const __m128i c16_n21_p04 = _mm_set1_epi32(0xFFEB0004);
    const __m128i c16_n45_n40 = _mm_set1_epi32(0xFFD3FFD8);
    const __m128i c16_n13_n35 = _mm_set1_epi32(0xFFF3FFDD);
    const __m128i c16_p04_p40 = _mm_set1_epi32(0x00040028);   //row2
    const __m128i c16_n43_n35 = _mm_set1_epi32(0xFFD5FFDD);
    const __m128i c16_p29_n13 = _mm_set1_epi32(0x001DFFF3);
    const __m128i c16_p21_p45 = _mm_set1_epi32(0x0015002D);
    const __m128i c16_n21_p35 = _mm_set1_epi32(0xFFEB0023);   //row3
    const __m128i c16_p04_n43 = _mm_set1_epi32(0x0004FFD5);
    const __m128i c16_p13_p45 = _mm_set1_epi32(0x000D002D);
    const __m128i c16_n29_n40 = _mm_set1_epi32(0xFFE3FFD8);
    const __m128i c16_n40_p29 = _mm_set1_epi32(0xFFD8001D);   //row4
    const __m128i c16_p45_n13 = _mm_set1_epi32(0x002DFFF3);
    const __m128i c16_n43_n04 = _mm_set1_epi32(0xFFD5FFFC);
    const __m128i c16_p35_p21 = _mm_set1_epi32(0x00230015);
    const __m128i c16_n45_p21 = _mm_set1_epi32(0xFFD30015);   //row5
    const __m128i c16_p13_p29 = _mm_set1_epi32(0x000D001D);
    const __m128i c16_p35_n43 = _mm_set1_epi32(0x0023FFD5);
    const __m128i c16_n40_p04 = _mm_set1_epi32(0xFFD80004);
    const __m128i c16_n35_p13 = _mm_set1_epi32(0xFFDD000D);   //row6
    const __m128i c16_n40_p45 = _mm_set1_epi32(0xFFD8002D);
    const __m128i c16_p04_p21 = _mm_set1_epi32(0x00040015);
    const __m128i c16_p43_n29 = _mm_set1_epi32(0x002BFFE3);
    const __m128i c16_n13_p04 = _mm_set1_epi32(0xFFF30004);   //row7
    const __m128i c16_n29_p21 = _mm_set1_epi32(0xFFE30015);
    const __m128i c16_n40_p35 = _mm_set1_epi32(0xFFD80023);
    const __m128i c16_n45_p43 = _mm_set1_epi32(0xFFD3002B);

    const __m128i c16_p38_p44 = _mm_set1_epi32(0x0026002C);
    const __m128i c16_p09_p25 = _mm_set1_epi32(0x00090019);
    const __m128i c16_n09_p38 = _mm_set1_epi32(0xFFF70026);
    const __m128i c16_n25_n44 = _mm_set1_epi32(0xFFE7FFD4);
    const __m128i c16_n44_p25 = _mm_set1_epi32(0xFFD40019);
    const __m128i c16_p38_p09 = _mm_set1_epi32(0x00260009);
    const __m128i c16_n25_p09 = _mm_set1_epi32(0xFFE70009);
    const __m128i c16_n44_p38 = _mm_set1_epi32(0xFFD40026);

    const __m128i c16_p17_p42 = _mm_set1_epi32(0x0011002A);
    const __m128i c16_n42_p17 = _mm_set1_epi32(0xFFD60011);

    const __m128i c16_n32_p32 = _mm_set1_epi32(0xFFE00020);
    const __m128i c16_p32_p32 = _mm_set1_epi32(0x00200020);

    __m128i c32_rnd = _mm_set1_epi32(1 << (shift1 - 1));        // add1

    // DCT1
    __m128i in00[2], in01[2], in02[2], in03[2];
    __m128i res00[2], res01[2], res02[2], res03[2];
    int i, part;

    for (i = 0; i < 2; i++) {
        const int offset = (i << 3);
        in00[i] = _mm_loadu_si128((const __m128i*)&src[0 * 16 + offset]);   // [07 06 05 04 03 02 01 00]
        in01[i] = _mm_loadu_si128((const __m128i*)&src[1 * 16 + offset]);   // [17 16 15 14 13 12 11 10]
        in02[i] = _mm_loadu_si128((const __m128i*)&src[2 * 16 + offset]);   // [27 26 25 24 23 22 21 20]
        in03[i] = _mm_loadu_si128((const __m128i*)&src[3 * 16 + offset]);   // [37 36 35 34 33 32 31 30]
    }

    for (part = 0; part < 2; part++) {
        const __m128i T_00_00A = _mm_unpacklo_epi16(in01[part], in03[part]);
        const __m128i T_00_00B = _mm_unpackhi_epi16(in01[part], in03[part]);
        const __m128i T_00_01A = _mm_unpacklo_epi16(in00[part], in02[part]);
        const __m128i T_00_01B = _mm_unpackhi_epi16(in00[part], in02[part]);

        __m128i E0A, E0B, E1A, E1B, O0A, O0B, O1A, O1B;

        E0A = _mm_add_epi32(_mm_madd_epi16(T_00_01A, c16_p32_p32), c32_rnd);
        E1A = _mm_add_epi32(_mm_madd_epi16(T_00_01A, c16_n32_p32), c32_rnd);

        E0B = _mm_add_epi32(_mm_madd_epi16(T_00_01B, c16_p32_p32), c32_rnd);
        E1B = _mm_add_epi32(_mm_madd_epi16(T_00_01B, c16_n32_p32), c32_rnd);

        O0A = _mm_madd_epi16(T_00_00A, c16_p17_p42);
        O1A = _mm_madd_epi16(T_00_00A, c16_n42_p17);

        O0B = _mm_madd_epi16(T_00_00B, c16_p17_p42);
        O1B = _mm_madd_epi16(T_00_00B, c16_n42_p17);

        res00[part] = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E0A, O0A), 5), _mm_srai_epi32(_mm_add_epi32(E0B, O0B), 5));
        res03[part] = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E0A, O0A), 5), _mm_srai_epi32(_mm_sub_epi32(E0B, O0B), 5));
        res01[part] = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E1A, O1A), 5), _mm_srai_epi32(_mm_add_epi32(E1B, O1B), 5));
        res02[part] = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E1A, O1A), 5), _mm_srai_epi32(_mm_sub_epi32(E1B, O1B), 5));
    }

    // transpose matrix
    {
        __m128i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
        __m128i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;

        tr0_0 = _mm_unpacklo_epi16(res00[0], res01[0]);
        tr0_1 = _mm_unpacklo_epi16(res02[0], res03[0]);

        tr0_2 = _mm_unpackhi_epi16(res00[0], res01[0]);
        tr0_3 = _mm_unpackhi_epi16(res02[0], res03[0]);

        tr0_4 = _mm_unpacklo_epi16(res00[1], res01[1]);
        tr0_5 = _mm_unpacklo_epi16(res02[1], res03[1]);

        tr0_6 = _mm_unpackhi_epi16(res00[1], res01[1]);
        tr0_7 = _mm_unpackhi_epi16(res02[1], res03[1]);

        tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1);
        tr1_1 = _mm_unpacklo_epi32(tr0_2, tr0_3);

        tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1);
        tr1_3 = _mm_unpackhi_epi32(tr0_2, tr0_3);

        tr1_4 = _mm_unpacklo_epi32(tr0_4, tr0_5);
        tr1_5 = _mm_unpacklo_epi32(tr0_6, tr0_7);

        tr1_6 = _mm_unpackhi_epi32(tr0_4, tr0_5);
        tr1_7 = _mm_unpackhi_epi32(tr0_6, tr0_7);

        // second fft
        c32_rnd = _mm_set1_epi32(1 << (shift2 - 1));                    // add2
        {
            const __m128i T_00_00A = _mm_unpackhi_epi16(tr1_0, tr1_2);  // [33 13 32 12 31 11 30 10]
            const __m128i T_00_01A = _mm_unpackhi_epi16(tr1_1, tr1_3);  // [ ]
            const __m128i T_00_02A = _mm_unpackhi_epi16(tr1_4, tr1_6);  // [ ]
            const __m128i T_00_03A = _mm_unpackhi_epi16(tr1_5, tr1_7);  // [ ]
            const __m128i T_00_04A = _mm_unpacklo_epi16(tr1_2, tr1_3);  // [ ]
            const __m128i T_00_05A = _mm_unpacklo_epi16(tr1_6, tr1_7);  // [ ]
            const __m128i T_00_06A = _mm_unpacklo_epi16(tr1_1, tr1_5);  // [ ]row
            const __m128i T_00_07A = _mm_unpacklo_epi16(tr1_0, tr1_4);  // [83 03 82 02 81 01 81 00] row08 row00

            __m128i O0A, O1A, O2A, O3A, O4A, O5A, O6A, O7A;

            __m128i EO0A, EO1A, EO2A, EO3A;
            __m128i EEO0A, EEO1A;
            __m128i EEE0A, EEE1A;
#define COMPUTE_ROW(row0103, row0507, row0911, row1315, c0103, c0507, c0911, c1315, row) \
    row = _mm_add_epi32(_mm_add_epi32(_mm_madd_epi16(row0103, c0103), _mm_madd_epi16(row0507, c0507)), \
                        _mm_add_epi32(_mm_madd_epi16(row0911, c0911), _mm_madd_epi16(row1315, c1315)));

            COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_p43_p45, c16_p35_p40, c16_p21_p29, c16_p04_p13, O0A)
            COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_p29_p43, c16_n21_p04, c16_n45_n40, c16_n13_n35, O1A)
            COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_p04_p40, c16_n43_n35, c16_p29_n13, c16_p21_p45, O2A)
            COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n21_p35, c16_p04_n43, c16_p13_p45, c16_n29_n40, O3A)
            COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n40_p29, c16_p45_n13, c16_n43_n04, c16_p35_p21, O4A)
            COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n45_p21, c16_p13_p29, c16_p35_n43, c16_n40_p04, O5A)
            COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n35_p13, c16_n40_p45, c16_p04_p21, c16_p43_n29, O6A)
            COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n13_p04, c16_n29_p21, c16_n40_p35, c16_n45_p43, O7A)

#undef COMPUTE_ROW

            EO0A = _mm_add_epi32(_mm_madd_epi16(T_00_04A, c16_p38_p44), _mm_madd_epi16(T_00_05A, c16_p09_p25)); // EO0
            EO1A = _mm_add_epi32(_mm_madd_epi16(T_00_04A, c16_n09_p38), _mm_madd_epi16(T_00_05A, c16_n25_n44)); // EO1
            EO2A = _mm_add_epi32(_mm_madd_epi16(T_00_04A, c16_n44_p25), _mm_madd_epi16(T_00_05A, c16_p38_p09)); // EO2
            EO3A = _mm_add_epi32(_mm_madd_epi16(T_00_04A, c16_n25_p09), _mm_madd_epi16(T_00_05A, c16_n44_p38)); // EO3

            EEO0A = _mm_madd_epi16(T_00_06A, c16_p17_p42);
            EEO1A = _mm_madd_epi16(T_00_06A, c16_n42_p17);

            EEE0A = _mm_madd_epi16(T_00_07A, c16_p32_p32);
            EEE1A = _mm_madd_epi16(T_00_07A, c16_n32_p32);
            {
                const __m128i EE0A = _mm_add_epi32(EEE0A, EEO0A);   // EE0 = EEE0 + EEO0
                const __m128i EE1A = _mm_add_epi32(EEE1A, EEO1A);   // EE1 = EEE1 + EEO1
                const __m128i EE3A = _mm_sub_epi32(EEE0A, EEO0A);   // EE2 = EEE0 - EEO0
                const __m128i EE2A = _mm_sub_epi32(EEE1A, EEO1A);   // EE3 = EEE1 - EEO1

                const __m128i T10A = _mm_add_epi32(_mm_add_epi32(EE0A, EO0A), c32_rnd);   // E0 (= EE0 + EO0) + rnd
                const __m128i T11A = _mm_add_epi32(_mm_add_epi32(EE1A, EO1A), c32_rnd);   // E1 (= EE1 + EO1) + rnd
                const __m128i T12A = _mm_add_epi32(_mm_add_epi32(EE2A, EO2A), c32_rnd);   // E2 (= EE2 + EO2) + rnd
                const __m128i T13A = _mm_add_epi32(_mm_add_epi32(EE3A, EO3A), c32_rnd);   // E3 (= EE3 + EO3) + rnd
                const __m128i T14A = _mm_add_epi32(_mm_sub_epi32(EE3A, EO3A), c32_rnd);   // E4 (= EE3 - EO3) + rnd
                const __m128i T15A = _mm_add_epi32(_mm_sub_epi32(EE2A, EO2A), c32_rnd);   // E5 (= EE2 - EO2) + rnd
                const __m128i T16A = _mm_add_epi32(_mm_sub_epi32(EE1A, EO1A), c32_rnd);   // E6 (= EE1 - EO1) + rnd
                const __m128i T17A = _mm_add_epi32(_mm_sub_epi32(EE0A, EO0A), c32_rnd);   // E7 (= EE0 - EO0) + rnd

                const __m128i T30A = _mm_srai_epi32(_mm_add_epi32(T10A, O0A), shift2);  // E0 + O0 + rnd [30 20 10 00]
                const __m128i T31A = _mm_srai_epi32(_mm_add_epi32(T11A, O1A), shift2);  // E1 + O1 + rnd [31 21 11 01]
                const __m128i T32A = _mm_srai_epi32(_mm_add_epi32(T12A, O2A), shift2);  // E2 + O2 + rnd [32 22 12 02]
                const __m128i T33A = _mm_srai_epi32(_mm_add_epi32(T13A, O3A), shift2);  // E3 + O3 + rnd [33 23 13 03]
                const __m128i T34A = _mm_srai_epi32(_mm_add_epi32(T14A, O4A), shift2);  // E4            [33 24 14 04]
                const __m128i T35A = _mm_srai_epi32(_mm_add_epi32(T15A, O5A), shift2);  // E5            [35 25 15 05]
                const __m128i T36A = _mm_srai_epi32(_mm_add_epi32(T16A, O6A), shift2);  // E6            [36 26 16 06]
                const __m128i T37A = _mm_srai_epi32(_mm_add_epi32(T17A, O7A), shift2);  // E7            [37 27 17 07]
                const __m128i T38A = _mm_srai_epi32(_mm_sub_epi32(T17A, O7A), shift2);  // E7            [30 20 10 00] x8
                const __m128i T39A = _mm_srai_epi32(_mm_sub_epi32(T16A, O6A), shift2);  // E6            [31 21 11 01] x9
                const __m128i T3AA = _mm_srai_epi32(_mm_sub_epi32(T15A, O5A), shift2);  // E5            [32 22 12 02] xA
                const __m128i T3BA = _mm_srai_epi32(_mm_sub_epi32(T14A, O4A), shift2);  // E4            [33 23 13 03] xB
                const __m128i T3CA = _mm_srai_epi32(_mm_sub_epi32(T13A, O3A), shift2);  // E3 - O3 + rnd [33 24 14 04] xC
                const __m128i T3DA = _mm_srai_epi32(_mm_sub_epi32(T12A, O2A), shift2);  // E2 - O2 + rnd [35 25 15 05] xD
                const __m128i T3EA = _mm_srai_epi32(_mm_sub_epi32(T11A, O1A), shift2);  // E1 - O1 + rnd [36 26 16 06] xE
                const __m128i T3FA = _mm_srai_epi32(_mm_sub_epi32(T10A, O0A), shift2);  // E0 - O0 + rnd [37 27 17 07] xF

                res00[0] = _mm_packs_epi32(T30A, T38A);
                res01[0] = _mm_packs_epi32(T31A, T39A);
                res02[0] = _mm_packs_epi32(T32A, T3AA);
                res03[0] = _mm_packs_epi32(T33A, T3BA);
                res00[1] = _mm_packs_epi32(T34A, T3CA);
                res01[1] = _mm_packs_epi32(T35A, T3DA);
                res02[1] = _mm_packs_epi32(T36A, T3EA);
                res03[1] = _mm_packs_epi32(T37A, T3FA);
            }
        }

        // transpose matrix
        tr0_0 = _mm_unpacklo_epi16(res00[0], res01[0]);
        tr0_1 = _mm_unpacklo_epi16(res02[0], res03[0]);
        tr0_2 = _mm_unpackhi_epi16(res00[0], res01[0]);
        tr0_3 = _mm_unpackhi_epi16(res02[0], res03[0]);
        tr0_4 = _mm_unpacklo_epi16(res00[1], res01[1]);
        tr0_5 = _mm_unpacklo_epi16(res02[1], res03[1]);
        tr0_6 = _mm_unpackhi_epi16(res00[1], res01[1]);
        tr0_7 = _mm_unpackhi_epi16(res02[1], res03[1]);

        tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1);
        tr1_1 = _mm_unpacklo_epi32(tr0_2, tr0_3);
        tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1);
        tr1_3 = _mm_unpackhi_epi32(tr0_2, tr0_3);
        tr1_4 = _mm_unpacklo_epi32(tr0_4, tr0_5);
        tr1_5 = _mm_unpacklo_epi32(tr0_6, tr0_7);
        tr1_6 = _mm_unpackhi_epi32(tr0_4, tr0_5);
        tr1_7 = _mm_unpackhi_epi32(tr0_6, tr0_7);

        res00[0] = _mm_unpacklo_epi64(tr1_0, tr1_4);
        res01[0] = _mm_unpackhi_epi64(tr1_0, tr1_4);
        res02[0] = _mm_unpacklo_epi64(tr1_2, tr1_6);
        res03[0] = _mm_unpackhi_epi64(tr1_2, tr1_6);
        res00[1] = _mm_unpacklo_epi64(tr1_1, tr1_5);
        res01[1] = _mm_unpackhi_epi64(tr1_1, tr1_5);
        res02[1] = _mm_unpacklo_epi64(tr1_3, tr1_7);
        res03[1] = _mm_unpackhi_epi64(tr1_3, tr1_7);

        // clip
        {
            const __m128i max_val = _mm_set1_epi16((1 << (clip_depth2 - 1)) - 1);
            const __m128i min_val = _mm_set1_epi16(-(1 << (clip_depth2 - 1)));

            res00[0] = _mm_max_epi16(_mm_min_epi16(res00[0], max_val), min_val);
            res01[0] = _mm_max_epi16(_mm_min_epi16(res01[0], max_val), min_val);
            res02[0] = _mm_max_epi16(_mm_min_epi16(res02[0], max_val), min_val);
            res03[0] = _mm_max_epi16(_mm_min_epi16(res03[0], max_val), min_val);

            res00[1] = _mm_max_epi16(_mm_min_epi16(res00[1], max_val), min_val);
            res01[1] = _mm_max_epi16(_mm_min_epi16(res01[1], max_val), min_val);
            res02[1] = _mm_max_epi16(_mm_min_epi16(res02[1], max_val), min_val);
            res03[1] = _mm_max_epi16(_mm_min_epi16(res03[1], max_val), min_val);
        }
    }

    _mm_storeu_si128((__m128i*)(dst + 0 * i_dst    ), res00[0]);
    _mm_storeu_si128((__m128i*)(dst + 0 * i_dst + 8), res00[1]);
    _mm_storeu_si128((__m128i*)(dst + 1 * i_dst    ), res01[0]);
    _mm_storeu_si128((__m128i*)(dst + 1 * i_dst + 8), res01[1]);
    _mm_storeu_si128((__m128i*)(dst + 2 * i_dst    ), res02[0]);
    _mm_storeu_si128((__m128i*)(dst + 2 * i_dst + 8), res02[1]);
    _mm_storeu_si128((__m128i*)(dst + 3 * i_dst    ), res03[0]);
    _mm_storeu_si128((__m128i*)(dst + 3 * i_dst + 8), res03[1]);
}


/* ---------------------------------------------------------------------------
 */
void idct_c_8x8_sse128(const coeff_t *src, coeff_t *dst, int i_dst)
{
    // const int shift1 = 5;
    const int shift2 = 20 - g_bit_depth;
    // const int clip_depth1 = LIMIT_BIT;
    const int clip_depth2 = g_bit_depth + 1;

    __m128i S0, S1, S2, S3, S4, S5, S6, S7;
    __m128i mAdd, T0, T1, T2, T3;
    __m128i E0h, E1h, E2h, E3h, E0l, E1l, E2l, E3l;
    __m128i O0h, O1h, O2h, O3h, O0l, O1l, O2l, O3l;
    __m128i EE0l, EE1l, E00l, E01l, EE0h, EE1h, E00h, E01h;
    __m128i T00, T01, T02, T03, T04, T05, T06, T07;

    mAdd = _mm_set1_epi32(16);                // add1

    S1 = _mm_load_si128((__m128i*)&src[8]);
    S3 = _mm_load_si128((__m128i*)&src[24]);

    T0  = _mm_unpacklo_epi16(S1, S3);
    E1l = _mm_madd_epi16(T0, _mm_load_si128((__m128i*)(tab_idct_8x8[0])));
    T1  = _mm_unpackhi_epi16(S1, S3);
    E1h = _mm_madd_epi16(T1, _mm_load_si128((__m128i*)(tab_idct_8x8[0])));

    S5  = _mm_load_si128((__m128i*)&src[40]);
    S7  = _mm_load_si128((__m128i*)&src[56]);

    T2  = _mm_unpacklo_epi16(S5, S7);
    E2l = _mm_madd_epi16(T2, _mm_load_si128((__m128i*)(tab_idct_8x8[1])));
    T3  = _mm_unpackhi_epi16(S5, S7);
    E2h = _mm_madd_epi16(T3, _mm_load_si128((__m128i*)(tab_idct_8x8[1])));

    O0l = _mm_add_epi32(E1l, E2l);
    O0h = _mm_add_epi32(E1h, E2h);

    E1l = _mm_madd_epi16(T0, _mm_load_si128((__m128i*)(tab_idct_8x8[2])));
    E1h = _mm_madd_epi16(T1, _mm_load_si128((__m128i*)(tab_idct_8x8[2])));
    E2l = _mm_madd_epi16(T2, _mm_load_si128((__m128i*)(tab_idct_8x8[3])));
    E2h = _mm_madd_epi16(T3, _mm_load_si128((__m128i*)(tab_idct_8x8[3])));

    O1l = _mm_add_epi32(E1l, E2l);
    O1h = _mm_add_epi32(E1h, E2h);

    E1l = _mm_madd_epi16(T0, _mm_load_si128((__m128i*)(tab_idct_8x8[4])));
    E1h = _mm_madd_epi16(T1, _mm_load_si128((__m128i*)(tab_idct_8x8[4])));
    E2l = _mm_madd_epi16(T2, _mm_load_si128((__m128i*)(tab_idct_8x8[5])));
    E2h = _mm_madd_epi16(T3, _mm_load_si128((__m128i*)(tab_idct_8x8[5])));
    O2l = _mm_add_epi32(E1l, E2l);
    O2h = _mm_add_epi32(E1h, E2h);

    E1l = _mm_madd_epi16(T0, _mm_load_si128((__m128i*)(tab_idct_8x8[6])));
    E1h = _mm_madd_epi16(T1, _mm_load_si128((__m128i*)(tab_idct_8x8[6])));
    E2l = _mm_madd_epi16(T2, _mm_load_si128((__m128i*)(tab_idct_8x8[7])));
    E2h = _mm_madd_epi16(T3, _mm_load_si128((__m128i*)(tab_idct_8x8[7])));
    O3h = _mm_add_epi32(E1h, E2h);
    O3l = _mm_add_epi32(E1l, E2l);

    /*    -------     */

    S0 = _mm_load_si128((__m128i*)&src[0]);
    S4 = _mm_load_si128((__m128i*)&src[32]);

    T0   = _mm_unpacklo_epi16(S0, S4);
    EE0l = _mm_madd_epi16(T0, _mm_load_si128((__m128i*)(tab_idct_8x8[8])));
    T1   = _mm_unpackhi_epi16(S0, S4);
    EE0h = _mm_madd_epi16(T1, _mm_load_si128((__m128i*)(tab_idct_8x8[8])));

    EE1l = _mm_madd_epi16(T0, _mm_load_si128((__m128i*)(tab_idct_8x8[9])));
    EE1h = _mm_madd_epi16(T1, _mm_load_si128((__m128i*)(tab_idct_8x8[9])));

    /*    -------     */

    S2 = _mm_load_si128((__m128i*)&src[16]);
    S6 = _mm_load_si128((__m128i*)&src[48]);

    T0   = _mm_unpacklo_epi16(S2, S6);
    E00l = _mm_madd_epi16(T0, _mm_load_si128((__m128i*)(tab_idct_8x8[10])));
    T1   = _mm_unpackhi_epi16(S2, S6);
    E00h = _mm_madd_epi16(T1, _mm_load_si128((__m128i*)(tab_idct_8x8[10])));
    E01l = _mm_madd_epi16(T0, _mm_load_si128((__m128i*)(tab_idct_8x8[11])));
    E01h = _mm_madd_epi16(T1, _mm_load_si128((__m128i*)(tab_idct_8x8[11])));
    E0l = _mm_add_epi32(EE0l, E00l);
    E0l = _mm_add_epi32(E0l, mAdd);
    E0h = _mm_add_epi32(EE0h, E00h);
    E0h = _mm_add_epi32(E0h, mAdd);
    E3l = _mm_sub_epi32(EE0l, E00l);
    E3l = _mm_add_epi32(E3l, mAdd);
    E3h = _mm_sub_epi32(EE0h, E00h);
    E3h = _mm_add_epi32(E3h, mAdd);

    E1l = _mm_add_epi32(EE1l, E01l);
    E1l = _mm_add_epi32(E1l, mAdd);
    E1h = _mm_add_epi32(EE1h, E01h);
    E1h = _mm_add_epi32(E1h, mAdd);
    E2l = _mm_sub_epi32(EE1l, E01l);
    E2l = _mm_add_epi32(E2l, mAdd);
    E2h = _mm_sub_epi32(EE1h, E01h);
    E2h = _mm_add_epi32(E2h, mAdd);
    S0 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E0l, O0l), 5), _mm_srai_epi32(_mm_add_epi32(E0h, O0h), 5));  // 首次反变换移位数
    S7 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E0l, O0l), 5), _mm_srai_epi32(_mm_sub_epi32(E0h, O0h), 5));
    S1 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E1l, O1l), 5), _mm_srai_epi32(_mm_add_epi32(E1h, O1h), 5));
    S6 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E1l, O1l), 5), _mm_srai_epi32(_mm_sub_epi32(E1h, O1h), 5));
    S2 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E2l, O2l), 5), _mm_srai_epi32(_mm_add_epi32(E2h, O2h), 5));
    S5 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E2l, O2l), 5), _mm_srai_epi32(_mm_sub_epi32(E2h, O2h), 5));
    S3 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E3l, O3l), 5), _mm_srai_epi32(_mm_add_epi32(E3h, O3h), 5));
    S4 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E3l, O3l), 5), _mm_srai_epi32(_mm_sub_epi32(E3h, O3h), 5));

    /*  Inverse matrix   */

    E0l = _mm_unpacklo_epi16(S0, S4);
    E1l = _mm_unpacklo_epi16(S1, S5);
    E2l = _mm_unpacklo_epi16(S2, S6);
    E3l = _mm_unpacklo_epi16(S3, S7);
    O0l = _mm_unpackhi_epi16(S0, S4);
    O1l = _mm_unpackhi_epi16(S1, S5);
    O2l = _mm_unpackhi_epi16(S2, S6);
    O3l = _mm_unpackhi_epi16(S3, S7);

    T0  = _mm_unpacklo_epi16(E0l, E2l);
    T1  = _mm_unpacklo_epi16(E1l, E3l);
    S0  = _mm_unpacklo_epi16(T0, T1);
    S1  = _mm_unpackhi_epi16(T0, T1);

    T2  = _mm_unpackhi_epi16(E0l, E2l);
    T3  = _mm_unpackhi_epi16(E1l, E3l);
    S2  = _mm_unpacklo_epi16(T2, T3);
    S3  = _mm_unpackhi_epi16(T2, T3);

    T0  = _mm_unpacklo_epi16(O0l, O2l);
    T1  = _mm_unpacklo_epi16(O1l, O3l);
    S4  = _mm_unpacklo_epi16(T0, T1);
    S5  = _mm_unpackhi_epi16(T0, T1);

    T2  = _mm_unpackhi_epi16(O0l, O2l);
    T3  = _mm_unpackhi_epi16(O1l, O3l);
    S6  = _mm_unpacklo_epi16(T2, T3);
    S7  = _mm_unpackhi_epi16(T2, T3);

    mAdd = _mm_set1_epi32(1 << (shift2 - 1));   // add2

    T0  = _mm_unpacklo_epi16(S1, S3);
    E1l = _mm_madd_epi16(T0, _mm_load_si128((__m128i*)(tab_idct_8x8[0])));
    T1  = _mm_unpackhi_epi16(S1, S3);
    E1h = _mm_madd_epi16(T1, _mm_load_si128((__m128i*)(tab_idct_8x8[0])));
    T2  = _mm_unpacklo_epi16(S5, S7);
    E2l = _mm_madd_epi16(T2, _mm_load_si128((__m128i*)(tab_idct_8x8[1])));
    T3  = _mm_unpackhi_epi16(S5, S7);
    E2h = _mm_madd_epi16(T3, _mm_load_si128((__m128i*)(tab_idct_8x8[1])));

    O0l = _mm_add_epi32(E1l, E2l);
    O0h = _mm_add_epi32(E1h, E2h);
    E1l = _mm_madd_epi16(T0, _mm_load_si128((__m128i*)(tab_idct_8x8[2])));
    E1h = _mm_madd_epi16(T1, _mm_load_si128((__m128i*)(tab_idct_8x8[2])));
    E2l = _mm_madd_epi16(T2, _mm_load_si128((__m128i*)(tab_idct_8x8[3])));
    E2h = _mm_madd_epi16(T3, _mm_load_si128((__m128i*)(tab_idct_8x8[3])));
    O1l = _mm_add_epi32(E1l, E2l);
    O1h = _mm_add_epi32(E1h, E2h);
    E1l = _mm_madd_epi16(T0, _mm_load_si128((__m128i*)(tab_idct_8x8[4])));
    E1h = _mm_madd_epi16(T1, _mm_load_si128((__m128i*)(tab_idct_8x8[4])));
    E2l = _mm_madd_epi16(T2, _mm_load_si128((__m128i*)(tab_idct_8x8[5])));
    E2h = _mm_madd_epi16(T3, _mm_load_si128((__m128i*)(tab_idct_8x8[5])));
    O2l = _mm_add_epi32(E1l, E2l);
    O2h = _mm_add_epi32(E1h, E2h);
    E1l = _mm_madd_epi16(T0, _mm_load_si128((__m128i*)(tab_idct_8x8[6])));
    E1h = _mm_madd_epi16(T1, _mm_load_si128((__m128i*)(tab_idct_8x8[6])));
    E2l = _mm_madd_epi16(T2, _mm_load_si128((__m128i*)(tab_idct_8x8[7])));
    E2h = _mm_madd_epi16(T3, _mm_load_si128((__m128i*)(tab_idct_8x8[7])));
    O3h = _mm_add_epi32(E1h, E2h);
    O3l = _mm_add_epi32(E1l, E2l);

    T0   = _mm_unpacklo_epi16(S0, S4);
    T1   = _mm_unpackhi_epi16(S0, S4);
    EE0l = _mm_madd_epi16(T0, _mm_load_si128((__m128i*)(tab_idct_8x8[8])));
    EE0h = _mm_madd_epi16(T1, _mm_load_si128((__m128i*)(tab_idct_8x8[8])));
    EE1l = _mm_madd_epi16(T0, _mm_load_si128((__m128i*)(tab_idct_8x8[9])));
    EE1h = _mm_madd_epi16(T1, _mm_load_si128((__m128i*)(tab_idct_8x8[9])));

    T0   = _mm_unpacklo_epi16(S2, S6);
    T1   = _mm_unpackhi_epi16(S2, S6);
    E00l = _mm_madd_epi16(T0, _mm_load_si128((__m128i*)(tab_idct_8x8[10])));
    E00h = _mm_madd_epi16(T1, _mm_load_si128((__m128i*)(tab_idct_8x8[10])));
    E01l = _mm_madd_epi16(T0, _mm_load_si128((__m128i*)(tab_idct_8x8[11])));
    E01h = _mm_madd_epi16(T1, _mm_load_si128((__m128i*)(tab_idct_8x8[11])));
    E0l = _mm_add_epi32(EE0l, E00l);
    E0l = _mm_add_epi32(E0l, mAdd);
    E0h = _mm_add_epi32(EE0h, E00h);
    E0h = _mm_add_epi32(E0h, mAdd);
    E3l = _mm_sub_epi32(EE0l, E00l);
    E3l = _mm_add_epi32(E3l, mAdd);
    E3h = _mm_sub_epi32(EE0h, E00h);
    E3h = _mm_add_epi32(E3h, mAdd);
    E1l = _mm_add_epi32(EE1l, E01l);
    E1l = _mm_add_epi32(E1l, mAdd);
    E1h = _mm_add_epi32(EE1h, E01h);
    E1h = _mm_add_epi32(E1h, mAdd);
    E2l = _mm_sub_epi32(EE1l, E01l);
    E2l = _mm_add_epi32(E2l, mAdd);
    E2h = _mm_sub_epi32(EE1h, E01h);
    E2h = _mm_add_epi32(E2h, mAdd);

    S0 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E0l, O0l), shift2), _mm_srai_epi32(_mm_add_epi32(E0h, O0h), shift2));
    S7 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E0l, O0l), shift2), _mm_srai_epi32(_mm_sub_epi32(E0h, O0h), shift2));
    S1 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E1l, O1l), shift2), _mm_srai_epi32(_mm_add_epi32(E1h, O1h), shift2));
    S6 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E1l, O1l), shift2), _mm_srai_epi32(_mm_sub_epi32(E1h, O1h), shift2));
    S2 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E2l, O2l), shift2), _mm_srai_epi32(_mm_add_epi32(E2h, O2h), shift2));
    S5 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E2l, O2l), shift2), _mm_srai_epi32(_mm_sub_epi32(E2h, O2h), shift2));
    S3 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E3l, O3l), shift2), _mm_srai_epi32(_mm_add_epi32(E3h, O3h), shift2));
    S4 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E3l, O3l), shift2), _mm_srai_epi32(_mm_sub_epi32(E3h, O3h), shift2));

    // [07 06 05 04 03 02 01 00]
    // [17 16 15 14 13 12 11 10]
    // [27 26 25 24 23 22 21 20]
    // [37 36 35 34 33 32 31 30]
    // [47 46 45 44 43 42 41 40]
    // [57 56 55 54 53 52 51 50]
    // [67 66 65 64 63 62 61 60]
    // [77 76 75 74 73 72 71 70]

    T00 = _mm_unpacklo_epi16(S0, S1);     // [13 03 12 02 11 01 10 00]
    T01 = _mm_unpackhi_epi16(S0, S1);     // [17 07 16 06 15 05 14 04]
    T02 = _mm_unpacklo_epi16(S2, S3);     // [33 23 32 22 31 21 30 20]
    T03 = _mm_unpackhi_epi16(S2, S3);     // [37 27 36 26 35 25 34 24]
    T04 = _mm_unpacklo_epi16(S4, S5);     // [53 43 52 42 51 41 50 40]
    T05 = _mm_unpackhi_epi16(S4, S5);     // [57 47 56 46 55 45 54 44]
    T06 = _mm_unpacklo_epi16(S6, S7);     // [73 63 72 62 71 61 70 60]
    T07 = _mm_unpackhi_epi16(S6, S7);     // [77 67 76 66 75 65 74 64]

    // clip
    {
        const __m128i max_val = _mm_set1_epi16((1 << (clip_depth2 - 1)) - 1);
        const __m128i min_val = _mm_set1_epi16(-(1 << (clip_depth2 - 1)));

        T00 = _mm_max_epi16(_mm_min_epi16(T00, max_val), min_val);
        T01 = _mm_max_epi16(_mm_min_epi16(T01, max_val), min_val);
        T02 = _mm_max_epi16(_mm_min_epi16(T02, max_val), min_val);
        T03 = _mm_max_epi16(_mm_min_epi16(T03, max_val), min_val);
        T04 = _mm_max_epi16(_mm_min_epi16(T04, max_val), min_val);
        T05 = _mm_max_epi16(_mm_min_epi16(T05, max_val), min_val);
        T06 = _mm_max_epi16(_mm_min_epi16(T06, max_val), min_val);
        T07 = _mm_max_epi16(_mm_min_epi16(T07, max_val), min_val);
    }

    {
        __m128i T10, T11, T12, T13;

        T10 = _mm_unpacklo_epi32(T00, T02);     // [31 21 11 01 30 20 10 00]
        T11 = _mm_unpackhi_epi32(T00, T02);     // [33 23 13 03 32 22 12 02]
        T12 = _mm_unpacklo_epi32(T04, T06);     // [71 61 51 41 70 60 50 40]
        T13 = _mm_unpackhi_epi32(T04, T06);     // [73 63 53 43 72 62 52 42]

        _mm_store_si128((__m128i*)(dst + 0 * i_dst), _mm_unpacklo_epi64(T10, T12));  // [70 60 50 40 30 20 10 00]
        _mm_store_si128((__m128i*)(dst + 1 * i_dst), _mm_unpackhi_epi64(T10, T12));  // [71 61 51 41 31 21 11 01]
        _mm_store_si128((__m128i*)(dst + 2 * i_dst), _mm_unpacklo_epi64(T11, T13));  // [72 62 52 42 32 22 12 02]
        _mm_store_si128((__m128i*)(dst + 3 * i_dst), _mm_unpackhi_epi64(T11, T13));  // [73 63 53 43 33 23 13 03]

        T10 = _mm_unpacklo_epi32(T01, T03);     // [35 25 15 05 34 24 14 04]
        T12 = _mm_unpacklo_epi32(T05, T07);     // [75 65 55 45 74 64 54 44]
        T11 = _mm_unpackhi_epi32(T01, T03);     // [37 27 17 07 36 26 16 06]
        T13 = _mm_unpackhi_epi32(T05, T07);     // [77 67 57 47 76 56 46 36]

        _mm_store_si128((__m128i*)(dst + 4 * i_dst), _mm_unpacklo_epi64(T10, T12));  // [74 64 54 44 34 24 14 04]
        _mm_store_si128((__m128i*)(dst + 5 * i_dst), _mm_unpackhi_epi64(T10, T12));  // [75 65 55 45 35 25 15 05]
        _mm_store_si128((__m128i*)(dst + 6 * i_dst), _mm_unpacklo_epi64(T11, T13));  // [76 66 56 46 36 26 16 06]
        _mm_store_si128((__m128i*)(dst + 7 * i_dst), _mm_unpackhi_epi64(T11, T13));  // [77 67 57 47 37 27 17 07]
    }
}


/* ---------------------------------------------------------------------------
 */
void idct_c_16x16_sse128(const coeff_t *src, coeff_t *dst, int i_dst)
{
    const int shift1 = 5;
    const int shift2 = 20 - g_bit_depth;
    //const int clip_depth1 = LIMIT_BIT;
    const int clip_depth2 = g_bit_depth + 1;

    const __m128i c16_p43_p45 = _mm_set1_epi32(0x002B002D);   //row0 87high - 90low address
    const __m128i c16_p35_p40 = _mm_set1_epi32(0x00230028);
    const __m128i c16_p21_p29 = _mm_set1_epi32(0x0015001D);
    const __m128i c16_p04_p13 = _mm_set1_epi32(0x0004000D);
    const __m128i c16_p29_p43 = _mm_set1_epi32(0x001D002B);   //row1
    const __m128i c16_n21_p04 = _mm_set1_epi32(0xFFEB0004);
    const __m128i c16_n45_n40 = _mm_set1_epi32(0xFFD3FFD8);
    const __m128i c16_n13_n35 = _mm_set1_epi32(0xFFF3FFDD);
    const __m128i c16_p04_p40 = _mm_set1_epi32(0x00040028);   //row2
    const __m128i c16_n43_n35 = _mm_set1_epi32(0xFFD5FFDD);
    const __m128i c16_p29_n13 = _mm_set1_epi32(0x001DFFF3);
    const __m128i c16_p21_p45 = _mm_set1_epi32(0x0015002D);
    const __m128i c16_n21_p35 = _mm_set1_epi32(0xFFEB0023);   //row3
    const __m128i c16_p04_n43 = _mm_set1_epi32(0x0004FFD5);
    const __m128i c16_p13_p45 = _mm_set1_epi32(0x000D002D);
    const __m128i c16_n29_n40 = _mm_set1_epi32(0xFFE3FFD8);
    const __m128i c16_n40_p29 = _mm_set1_epi32(0xFFD8001D);   //row4
    const __m128i c16_p45_n13 = _mm_set1_epi32(0x002DFFF3);
    const __m128i c16_n43_n04 = _mm_set1_epi32(0xFFD5FFFC);
    const __m128i c16_p35_p21 = _mm_set1_epi32(0x00230015);
    const __m128i c16_n45_p21 = _mm_set1_epi32(0xFFD30015);   //row5
    const __m128i c16_p13_p29 = _mm_set1_epi32(0x000D001D);
    const __m128i c16_p35_n43 = _mm_set1_epi32(0x0023FFD5);
    const __m128i c16_n40_p04 = _mm_set1_epi32(0xFFD80004);
    const __m128i c16_n35_p13 = _mm_set1_epi32(0xFFDD000D);   //row6
    const __m128i c16_n40_p45 = _mm_set1_epi32(0xFFD8002D);
    const __m128i c16_p04_p21 = _mm_set1_epi32(0x00040015);
    const __m128i c16_p43_n29 = _mm_set1_epi32(0x002BFFE3);
    const __m128i c16_n13_p04 = _mm_set1_epi32(0xFFF30004);   //row7
    const __m128i c16_n29_p21 = _mm_set1_epi32(0xFFE30015);
    const __m128i c16_n40_p35 = _mm_set1_epi32(0xFFD80023);
    const __m128i c16_n45_p43 = _mm_set1_epi32(0xFFD3002B);

    const __m128i c16_p38_p44 = _mm_set1_epi32(0x0026002C);
    const __m128i c16_p09_p25 = _mm_set1_epi32(0x00090019);
    const __m128i c16_n09_p38 = _mm_set1_epi32(0xFFF70026);
    const __m128i c16_n25_n44 = _mm_set1_epi32(0xFFE7FFD4);
    const __m128i c16_n44_p25 = _mm_set1_epi32(0xFFD40019);
    const __m128i c16_p38_p09 = _mm_set1_epi32(0x00260009);
    const __m128i c16_n25_p09 = _mm_set1_epi32(0xFFE70009);
    const __m128i c16_n44_p38 = _mm_set1_epi32(0xFFD40026);

    const __m128i c16_p17_p42 = _mm_set1_epi32(0x0011002A);
    const __m128i c16_n42_p17 = _mm_set1_epi32(0xFFD60011);

    const __m128i c16_n32_p32 = _mm_set1_epi32(0xFFE00020);
    const __m128i c16_p32_p32 = _mm_set1_epi32(0x00200020);

    int i, pass, part;

    int nShift = shift1;
    __m128i c32_rnd = _mm_set1_epi32((1 << shift1) >> 1);               // add1

    // DCT1
    __m128i in00[2], in01[2], in02[2], in03[2], in04[2], in05[2], in06[2], in07[2];
    __m128i in08[2], in09[2], in10[2], in11[2], in12[2], in13[2], in14[2], in15[2];
    __m128i res00[2], res01[2], res02[2], res03[2], res04[2], res05[2], res06[2], res07[2];
    __m128i res08[2], res09[2], res10[2], res11[2], res12[2], res13[2], res14[2], res15[2];

    for (i = 0; i < 2; i++) {
        const int offset = (i << 3);

        in00[i] = _mm_load_si128((const __m128i*)&src[ 0 * 16 + offset]);   // [07 06 05 04 03 02 01 00]
        in01[i] = _mm_load_si128((const __m128i*)&src[ 1 * 16 + offset]);   // [17 16 15 14 13 12 11 10]
        in02[i] = _mm_load_si128((const __m128i*)&src[ 2 * 16 + offset]);   // [27 26 25 24 23 22 21 20]
        in03[i] = _mm_load_si128((const __m128i*)&src[ 3 * 16 + offset]);   // [37 36 35 34 33 32 31 30]
        in04[i] = _mm_load_si128((const __m128i*)&src[ 4 * 16 + offset]);   // [47 46 45 44 43 42 41 40]
        in05[i] = _mm_load_si128((const __m128i*)&src[ 5 * 16 + offset]);   // [57 56 55 54 53 52 51 50]
        in06[i] = _mm_load_si128((const __m128i*)&src[ 6 * 16 + offset]);   // [67 66 65 64 63 62 61 60]
        in07[i] = _mm_load_si128((const __m128i*)&src[ 7 * 16 + offset]);   // [77 76 75 74 73 72 71 70]
        in08[i] = _mm_load_si128((const __m128i*)&src[ 8 * 16 + offset]);
        in09[i] = _mm_load_si128((const __m128i*)&src[ 9 * 16 + offset]);
        in10[i] = _mm_load_si128((const __m128i*)&src[10 * 16 + offset]);
        in11[i] = _mm_load_si128((const __m128i*)&src[11 * 16 + offset]);
        in12[i] = _mm_load_si128((const __m128i*)&src[12 * 16 + offset]);
        in13[i] = _mm_load_si128((const __m128i*)&src[13 * 16 + offset]);
        in14[i] = _mm_load_si128((const __m128i*)&src[14 * 16 + offset]);
        in15[i] = _mm_load_si128((const __m128i*)&src[15 * 16 + offset]);
    }

    for (pass = 0; pass < 2; pass++) {
        for (part = 0; part < 2; part++) {
            const __m128i T_00_00A = _mm_unpacklo_epi16(in01[part], in03[part]);    // [33 13 32 12 31 11 30 10]
            const __m128i T_00_00B = _mm_unpackhi_epi16(in01[part], in03[part]);    // [37 17 36 16 35 15 34 14]
            const __m128i T_00_01A = _mm_unpacklo_epi16(in05[part], in07[part]);    // [ ]
            const __m128i T_00_01B = _mm_unpackhi_epi16(in05[part], in07[part]);    // [ ]
            const __m128i T_00_02A = _mm_unpacklo_epi16(in09[part], in11[part]);    // [ ]
            const __m128i T_00_02B = _mm_unpackhi_epi16(in09[part], in11[part]);    // [ ]
            const __m128i T_00_03A = _mm_unpacklo_epi16(in13[part], in15[part]);    // [ ]
            const __m128i T_00_03B = _mm_unpackhi_epi16(in13[part], in15[part]);    // [ ]
            const __m128i T_00_04A = _mm_unpacklo_epi16(in02[part], in06[part]);    // [ ]
            const __m128i T_00_04B = _mm_unpackhi_epi16(in02[part], in06[part]);    // [ ]
            const __m128i T_00_05A = _mm_unpacklo_epi16(in10[part], in14[part]);    // [ ]
            const __m128i T_00_05B = _mm_unpackhi_epi16(in10[part], in14[part]);    // [ ]
            const __m128i T_00_06A = _mm_unpacklo_epi16(in04[part], in12[part]);    // [ ]row
            const __m128i T_00_06B = _mm_unpackhi_epi16(in04[part], in12[part]);    // [ ]
            const __m128i T_00_07A = _mm_unpacklo_epi16(in00[part], in08[part]);    // [83 03 82 02 81 01 81 00] row08 row00
            const __m128i T_00_07B = _mm_unpackhi_epi16(in00[part], in08[part]);    // [87 07 86 06 85 05 84 04]

            __m128i O0A, O1A, O2A, O3A, O4A, O5A, O6A, O7A;
            __m128i O0B, O1B, O2B, O3B, O4B, O5B, O6B, O7B;
            __m128i EO0A, EO1A, EO2A, EO3A;
            __m128i EO0B, EO1B, EO2B, EO3B;
            __m128i EEO0A, EEO1A;
            __m128i EEO0B, EEO1B;
            __m128i EEE0A, EEE1A;
            __m128i EEE0B, EEE1B;
            __m128i T00, T01;

#define COMPUTE_ROW(row0103, row0507, row0911, row1315, c0103, c0507, c0911, c1315, row) \
    T00 = _mm_add_epi32(_mm_madd_epi16(row0103, c0103), _mm_madd_epi16(row0507, c0507)); \
    T01 = _mm_add_epi32(_mm_madd_epi16(row0911, c0911), _mm_madd_epi16(row1315, c1315)); \
    row = _mm_add_epi32(T00, T01);

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


            EO0A = _mm_add_epi32(_mm_madd_epi16(T_00_04A, c16_p38_p44), _mm_madd_epi16(T_00_05A, c16_p09_p25)); // EO0
            EO0B = _mm_add_epi32(_mm_madd_epi16(T_00_04B, c16_p38_p44), _mm_madd_epi16(T_00_05B, c16_p09_p25));
            EO1A = _mm_add_epi32(_mm_madd_epi16(T_00_04A, c16_n09_p38), _mm_madd_epi16(T_00_05A, c16_n25_n44)); // EO1
            EO1B = _mm_add_epi32(_mm_madd_epi16(T_00_04B, c16_n09_p38), _mm_madd_epi16(T_00_05B, c16_n25_n44));
            EO2A = _mm_add_epi32(_mm_madd_epi16(T_00_04A, c16_n44_p25), _mm_madd_epi16(T_00_05A, c16_p38_p09)); // EO2
            EO2B = _mm_add_epi32(_mm_madd_epi16(T_00_04B, c16_n44_p25), _mm_madd_epi16(T_00_05B, c16_p38_p09));
            EO3A = _mm_add_epi32(_mm_madd_epi16(T_00_04A, c16_n25_p09), _mm_madd_epi16(T_00_05A, c16_n44_p38)); // EO3
            EO3B = _mm_add_epi32(_mm_madd_epi16(T_00_04B, c16_n25_p09), _mm_madd_epi16(T_00_05B, c16_n44_p38));


            EEO0A = _mm_madd_epi16(T_00_06A, c16_p17_p42);
            EEO0B = _mm_madd_epi16(T_00_06B, c16_p17_p42);
            EEO1A = _mm_madd_epi16(T_00_06A, c16_n42_p17);
            EEO1B = _mm_madd_epi16(T_00_06B, c16_n42_p17);


            EEE0A = _mm_madd_epi16(T_00_07A, c16_p32_p32);
            EEE0B = _mm_madd_epi16(T_00_07B, c16_p32_p32);
            EEE1A = _mm_madd_epi16(T_00_07A, c16_n32_p32);
            EEE1B = _mm_madd_epi16(T_00_07B, c16_n32_p32);

            {
                const __m128i EE0A = _mm_add_epi32(EEE0A, EEO0A);       // EE0 = EEE0 + EEO0
                const __m128i EE0B = _mm_add_epi32(EEE0B, EEO0B);
                const __m128i EE1A = _mm_add_epi32(EEE1A, EEO1A);       // EE1 = EEE1 + EEO1
                const __m128i EE1B = _mm_add_epi32(EEE1B, EEO1B);
                const __m128i EE3A = _mm_sub_epi32(EEE0A, EEO0A);       // EE2 = EEE0 - EEO0
                const __m128i EE3B = _mm_sub_epi32(EEE0B, EEO0B);
                const __m128i EE2A = _mm_sub_epi32(EEE1A, EEO1A);       // EE3 = EEE1 - EEO1
                const __m128i EE2B = _mm_sub_epi32(EEE1B, EEO1B);

                const __m128i E0A = _mm_add_epi32(EE0A, EO0A);          // E0 = EE0 + EO0
                const __m128i E0B = _mm_add_epi32(EE0B, EO0B);
                const __m128i E1A = _mm_add_epi32(EE1A, EO1A);          // E1 = EE1 + EO1
                const __m128i E1B = _mm_add_epi32(EE1B, EO1B);
                const __m128i E2A = _mm_add_epi32(EE2A, EO2A);          // E2 = EE2 + EO2
                const __m128i E2B = _mm_add_epi32(EE2B, EO2B);
                const __m128i E3A = _mm_add_epi32(EE3A, EO3A);          // E3 = EE3 + EO3
                const __m128i E3B = _mm_add_epi32(EE3B, EO3B);
                const __m128i E7A = _mm_sub_epi32(EE0A, EO0A);          // E0 = EE0 - EO0
                const __m128i E7B = _mm_sub_epi32(EE0B, EO0B);
                const __m128i E6A = _mm_sub_epi32(EE1A, EO1A);          // E1 = EE1 - EO1
                const __m128i E6B = _mm_sub_epi32(EE1B, EO1B);
                const __m128i E5A = _mm_sub_epi32(EE2A, EO2A);          // E2 = EE2 - EO2
                const __m128i E5B = _mm_sub_epi32(EE2B, EO2B);
                const __m128i E4A = _mm_sub_epi32(EE3A, EO3A);          // E3 = EE3 - EO3
                const __m128i E4B = _mm_sub_epi32(EE3B, EO3B);

                const __m128i T10A = _mm_add_epi32(E0A, c32_rnd);       // E0 + rnd
                const __m128i T10B = _mm_add_epi32(E0B, c32_rnd);
                const __m128i T11A = _mm_add_epi32(E1A, c32_rnd);       // E1 + rnd
                const __m128i T11B = _mm_add_epi32(E1B, c32_rnd);
                const __m128i T12A = _mm_add_epi32(E2A, c32_rnd);       // E2 + rnd
                const __m128i T12B = _mm_add_epi32(E2B, c32_rnd);
                const __m128i T13A = _mm_add_epi32(E3A, c32_rnd);       // E3 + rnd
                const __m128i T13B = _mm_add_epi32(E3B, c32_rnd);
                const __m128i T14A = _mm_add_epi32(E4A, c32_rnd);       // E4 + rnd
                const __m128i T14B = _mm_add_epi32(E4B, c32_rnd);
                const __m128i T15A = _mm_add_epi32(E5A, c32_rnd);       // E5 + rnd
                const __m128i T15B = _mm_add_epi32(E5B, c32_rnd);
                const __m128i T16A = _mm_add_epi32(E6A, c32_rnd);       // E6 + rnd
                const __m128i T16B = _mm_add_epi32(E6B, c32_rnd);
                const __m128i T17A = _mm_add_epi32(E7A, c32_rnd);       // E7 + rnd
                const __m128i T17B = _mm_add_epi32(E7B, c32_rnd);

                const __m128i T20A = _mm_add_epi32(T10A, O0A);          // E0 + O0 + rnd
                const __m128i T20B = _mm_add_epi32(T10B, O0B);
                const __m128i T21A = _mm_add_epi32(T11A, O1A);          // E1 + O1 + rnd
                const __m128i T21B = _mm_add_epi32(T11B, O1B);
                const __m128i T22A = _mm_add_epi32(T12A, O2A);          // E2 + O2 + rnd
                const __m128i T22B = _mm_add_epi32(T12B, O2B);
                const __m128i T23A = _mm_add_epi32(T13A, O3A);          // E3 + O3 + rnd
                const __m128i T23B = _mm_add_epi32(T13B, O3B);
                const __m128i T24A = _mm_add_epi32(T14A, O4A);          // E4
                const __m128i T24B = _mm_add_epi32(T14B, O4B);
                const __m128i T25A = _mm_add_epi32(T15A, O5A);          // E5
                const __m128i T25B = _mm_add_epi32(T15B, O5B);
                const __m128i T26A = _mm_add_epi32(T16A, O6A);          // E6
                const __m128i T26B = _mm_add_epi32(T16B, O6B);
                const __m128i T27A = _mm_add_epi32(T17A, O7A);          // E7
                const __m128i T27B = _mm_add_epi32(T17B, O7B);

                const __m128i T2FA = _mm_sub_epi32(T10A, O0A);          // E0 - O0 + rnd
                const __m128i T2FB = _mm_sub_epi32(T10B, O0B);
                const __m128i T2EA = _mm_sub_epi32(T11A, O1A);          // E1 - O1 + rnd
                const __m128i T2EB = _mm_sub_epi32(T11B, O1B);
                const __m128i T2DA = _mm_sub_epi32(T12A, O2A);          // E2 - O2 + rnd
                const __m128i T2DB = _mm_sub_epi32(T12B, O2B);
                const __m128i T2CA = _mm_sub_epi32(T13A, O3A);          // E3 - O3 + rnd
                const __m128i T2CB = _mm_sub_epi32(T13B, O3B);
                const __m128i T2BA = _mm_sub_epi32(T14A, O4A);          // E4
                const __m128i T2BB = _mm_sub_epi32(T14B, O4B);
                const __m128i T2AA = _mm_sub_epi32(T15A, O5A);          // E5
                const __m128i T2AB = _mm_sub_epi32(T15B, O5B);
                const __m128i T29A = _mm_sub_epi32(T16A, O6A);          // E6
                const __m128i T29B = _mm_sub_epi32(T16B, O6B);
                const __m128i T28A = _mm_sub_epi32(T17A, O7A);          // E7
                const __m128i T28B = _mm_sub_epi32(T17B, O7B);

                const __m128i T30A = _mm_srai_epi32(T20A, nShift);      // [30 20 10 00]
                const __m128i T30B = _mm_srai_epi32(T20B, nShift);      // [70 60 50 40]
                const __m128i T31A = _mm_srai_epi32(T21A, nShift);      // [31 21 11 01]
                const __m128i T31B = _mm_srai_epi32(T21B, nShift);      // [71 61 51 41]
                const __m128i T32A = _mm_srai_epi32(T22A, nShift);      // [32 22 12 02]
                const __m128i T32B = _mm_srai_epi32(T22B, nShift);      // [72 62 52 42]
                const __m128i T33A = _mm_srai_epi32(T23A, nShift);      // [33 23 13 03]
                const __m128i T33B = _mm_srai_epi32(T23B, nShift);      // [73 63 53 43]
                const __m128i T34A = _mm_srai_epi32(T24A, nShift);      // [33 24 14 04]
                const __m128i T34B = _mm_srai_epi32(T24B, nShift);      // [74 64 54 44]
                const __m128i T35A = _mm_srai_epi32(T25A, nShift);      // [35 25 15 05]
                const __m128i T35B = _mm_srai_epi32(T25B, nShift);      // [75 65 55 45]
                const __m128i T36A = _mm_srai_epi32(T26A, nShift);      // [36 26 16 06]
                const __m128i T36B = _mm_srai_epi32(T26B, nShift);      // [76 66 56 46]
                const __m128i T37A = _mm_srai_epi32(T27A, nShift);      // [37 27 17 07]
                const __m128i T37B = _mm_srai_epi32(T27B, nShift);      // [77 67 57 47]

                const __m128i T38A = _mm_srai_epi32(T28A, nShift);      // [30 20 10 00] x8
                const __m128i T38B = _mm_srai_epi32(T28B, nShift);      // [70 60 50 40]
                const __m128i T39A = _mm_srai_epi32(T29A, nShift);      // [31 21 11 01] x9
                const __m128i T39B = _mm_srai_epi32(T29B, nShift);      // [71 61 51 41]
                const __m128i T3AA = _mm_srai_epi32(T2AA, nShift);      // [32 22 12 02] xA
                const __m128i T3AB = _mm_srai_epi32(T2AB, nShift);      // [72 62 52 42]
                const __m128i T3BA = _mm_srai_epi32(T2BA, nShift);      // [33 23 13 03] xB
                const __m128i T3BB = _mm_srai_epi32(T2BB, nShift);      // [73 63 53 43]
                const __m128i T3CA = _mm_srai_epi32(T2CA, nShift);      // [33 24 14 04] xC
                const __m128i T3CB = _mm_srai_epi32(T2CB, nShift);      // [74 64 54 44]
                const __m128i T3DA = _mm_srai_epi32(T2DA, nShift);      // [35 25 15 05] xD
                const __m128i T3DB = _mm_srai_epi32(T2DB, nShift);      // [75 65 55 45]
                const __m128i T3EA = _mm_srai_epi32(T2EA, nShift);      // [36 26 16 06] xE
                const __m128i T3EB = _mm_srai_epi32(T2EB, nShift);      // [76 66 56 46]
                const __m128i T3FA = _mm_srai_epi32(T2FA, nShift);      // [37 27 17 07] xF
                const __m128i T3FB = _mm_srai_epi32(T2FB, nShift);      // [77 67 57 47]

                res00[part] = _mm_packs_epi32(T30A, T30B);              // [70 60 50 40 30 20 10 00]
                res01[part] = _mm_packs_epi32(T31A, T31B);              // [71 61 51 41 31 21 11 01]
                res02[part] = _mm_packs_epi32(T32A, T32B);              // [72 62 52 42 32 22 12 02]
                res03[part] = _mm_packs_epi32(T33A, T33B);              // [73 63 53 43 33 23 13 03]
                res04[part] = _mm_packs_epi32(T34A, T34B);              // [74 64 54 44 34 24 14 04]
                res05[part] = _mm_packs_epi32(T35A, T35B);              // [75 65 55 45 35 25 15 05]
                res06[part] = _mm_packs_epi32(T36A, T36B);              // [76 66 56 46 36 26 16 06]
                res07[part] = _mm_packs_epi32(T37A, T37B);              // [77 67 57 47 37 27 17 07]

                res08[part] = _mm_packs_epi32(T38A, T38B);              // [A0 ... 80]
                res09[part] = _mm_packs_epi32(T39A, T39B);              // [A1 ... 81]
                res10[part] = _mm_packs_epi32(T3AA, T3AB);              // [A2 ... 82]
                res11[part] = _mm_packs_epi32(T3BA, T3BB);              // [A3 ... 83]
                res12[part] = _mm_packs_epi32(T3CA, T3CB);              // [A4 ... 84]
                res13[part] = _mm_packs_epi32(T3DA, T3DB);              // [A5 ... 85]
                res14[part] = _mm_packs_epi32(T3EA, T3EB);              // [A6 ... 86]
                res15[part] = _mm_packs_epi32(T3FA, T3FB);              // [A7 ... 87]
            }
        }

        // transpose matrix 8x8 16bit
        {
            __m128i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
            __m128i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;

#define TRANSPOSE_8x8_16BIT(I0, I1, I2, I3, I4, I5, I6, I7, O0, O1, O2, O3, O4, O5, O6, O7) \
    tr0_0 = _mm_unpacklo_epi16(I0, I1); \
    tr0_1 = _mm_unpacklo_epi16(I2, I3); \
    tr0_2 = _mm_unpackhi_epi16(I0, I1); \
    tr0_3 = _mm_unpackhi_epi16(I2, I3); \
    tr0_4 = _mm_unpacklo_epi16(I4, I5); \
    tr0_5 = _mm_unpacklo_epi16(I6, I7); \
    tr0_6 = _mm_unpackhi_epi16(I4, I5); \
    tr0_7 = _mm_unpackhi_epi16(I6, I7); \
    tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1); \
    tr1_1 = _mm_unpacklo_epi32(tr0_2, tr0_3); \
    tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1); \
    tr1_3 = _mm_unpackhi_epi32(tr0_2, tr0_3); \
    tr1_4 = _mm_unpacklo_epi32(tr0_4, tr0_5); \
    tr1_5 = _mm_unpacklo_epi32(tr0_6, tr0_7); \
    tr1_6 = _mm_unpackhi_epi32(tr0_4, tr0_5); \
    tr1_7 = _mm_unpackhi_epi32(tr0_6, tr0_7); \
    O0 = _mm_unpacklo_epi64(tr1_0, tr1_4); \
    O1 = _mm_unpackhi_epi64(tr1_0, tr1_4); \
    O2 = _mm_unpacklo_epi64(tr1_2, tr1_6); \
    O3 = _mm_unpackhi_epi64(tr1_2, tr1_6); \
    O4 = _mm_unpacklo_epi64(tr1_1, tr1_5); \
    O5 = _mm_unpackhi_epi64(tr1_1, tr1_5); \
    O6 = _mm_unpacklo_epi64(tr1_3, tr1_7); \
    O7 = _mm_unpackhi_epi64(tr1_3, tr1_7); \
 
            TRANSPOSE_8x8_16BIT(res00[0], res01[0], res02[0], res03[0], res04[0], res05[0], res06[0], res07[0], in00[0], in01[0], in02[0], in03[0], in04[0], in05[0], in06[0], in07[0])
            TRANSPOSE_8x8_16BIT(res08[0], res09[0], res10[0], res11[0], res12[0], res13[0], res14[0], res15[0], in00[1], in01[1], in02[1], in03[1], in04[1], in05[1], in06[1], in07[1])
            TRANSPOSE_8x8_16BIT(res00[1], res01[1], res02[1], res03[1], res04[1], res05[1], res06[1], res07[1], in08[0], in09[0], in10[0], in11[0], in12[0], in13[0], in14[0], in15[0])
            TRANSPOSE_8x8_16BIT(res08[1], res09[1], res10[1], res11[1], res12[1], res13[1], res14[1], res15[1], in08[1], in09[1], in10[1], in11[1], in12[1], in13[1], in14[1], in15[1])

#undef TRANSPOSE_8x8_16BIT
        }

        nShift = shift2;
        c32_rnd = _mm_set1_epi32(1 << (shift2 - 1));    // add2
    }

    // clip
    {
        const __m128i max_val = _mm_set1_epi16((1 << (clip_depth2 - 1)) - 1);
        const __m128i min_val = _mm_set1_epi16(-(1 << (clip_depth2 - 1)));

        in00[0] = _mm_max_epi16(_mm_min_epi16(in00[0], max_val), min_val);
        in00[1] = _mm_max_epi16(_mm_min_epi16(in00[1], max_val), min_val);

        in01[0] = _mm_max_epi16(_mm_min_epi16(in01[0], max_val), min_val);
        in01[1] = _mm_max_epi16(_mm_min_epi16(in01[1], max_val), min_val);

        in02[0] = _mm_max_epi16(_mm_min_epi16(in02[0], max_val), min_val);
        in02[1] = _mm_max_epi16(_mm_min_epi16(in02[1], max_val), min_val);

        in03[0] = _mm_max_epi16(_mm_min_epi16(in03[0], max_val), min_val);
        in03[1] = _mm_max_epi16(_mm_min_epi16(in03[1], max_val), min_val);

        in04[0] = _mm_max_epi16(_mm_min_epi16(in04[0], max_val), min_val);
        in04[1] = _mm_max_epi16(_mm_min_epi16(in04[1], max_val), min_val);

        in05[0] = _mm_max_epi16(_mm_min_epi16(in05[0], max_val), min_val);
        in05[1] = _mm_max_epi16(_mm_min_epi16(in05[1], max_val), min_val);

        in06[0] = _mm_max_epi16(_mm_min_epi16(in06[0], max_val), min_val);
        in06[1] = _mm_max_epi16(_mm_min_epi16(in06[1], max_val), min_val);

        in07[0] = _mm_max_epi16(_mm_min_epi16(in07[0], max_val), min_val);
        in07[1] = _mm_max_epi16(_mm_min_epi16(in07[1], max_val), min_val);

        in08[0] = _mm_max_epi16(_mm_min_epi16(in08[0], max_val), min_val);
        in08[1] = _mm_max_epi16(_mm_min_epi16(in08[1], max_val), min_val);

        in09[0] = _mm_max_epi16(_mm_min_epi16(in09[0], max_val), min_val);
        in09[1] = _mm_max_epi16(_mm_min_epi16(in09[1], max_val), min_val);

        in10[0] = _mm_max_epi16(_mm_min_epi16(in10[0], max_val), min_val);
        in10[1] = _mm_max_epi16(_mm_min_epi16(in10[1], max_val), min_val);

        in11[0] = _mm_max_epi16(_mm_min_epi16(in11[0], max_val), min_val);
        in11[1] = _mm_max_epi16(_mm_min_epi16(in11[1], max_val), min_val);

        in12[0] = _mm_max_epi16(_mm_min_epi16(in12[0], max_val), min_val);
        in12[1] = _mm_max_epi16(_mm_min_epi16(in12[1], max_val), min_val);

        in13[0] = _mm_max_epi16(_mm_min_epi16(in13[0], max_val), min_val);
        in13[1] = _mm_max_epi16(_mm_min_epi16(in13[1], max_val), min_val);

        in14[0] = _mm_max_epi16(_mm_min_epi16(in14[0], max_val), min_val);
        in14[1] = _mm_max_epi16(_mm_min_epi16(in14[1], max_val), min_val);

        in15[0] = _mm_max_epi16(_mm_min_epi16(in15[0], max_val), min_val);
        in15[1] = _mm_max_epi16(_mm_min_epi16(in15[1], max_val), min_val);
    }

    // store
    _mm_store_si128((__m128i*)(dst +  0 * i_dst + 0), in00[0]);
    _mm_store_si128((__m128i*)(dst +  0 * i_dst + 8), in00[1]);
    _mm_store_si128((__m128i*)(dst +  1 * i_dst + 0), in01[0]);
    _mm_store_si128((__m128i*)(dst +  1 * i_dst + 8), in01[1]);
    _mm_store_si128((__m128i*)(dst +  2 * i_dst + 0), in02[0]);
    _mm_store_si128((__m128i*)(dst +  2 * i_dst + 8), in02[1]);
    _mm_store_si128((__m128i*)(dst +  3 * i_dst + 0), in03[0]);
    _mm_store_si128((__m128i*)(dst +  3 * i_dst + 8), in03[1]);
    _mm_store_si128((__m128i*)(dst +  4 * i_dst + 0), in04[0]);
    _mm_store_si128((__m128i*)(dst +  4 * i_dst + 8), in04[1]);
    _mm_store_si128((__m128i*)(dst +  5 * i_dst + 0), in05[0]);
    _mm_store_si128((__m128i*)(dst +  5 * i_dst + 8), in05[1]);
    _mm_store_si128((__m128i*)(dst +  6 * i_dst + 0), in06[0]);
    _mm_store_si128((__m128i*)(dst +  6 * i_dst + 8), in06[1]);
    _mm_store_si128((__m128i*)(dst +  7 * i_dst + 0), in07[0]);
    _mm_store_si128((__m128i*)(dst +  7 * i_dst + 8), in07[1]);
    _mm_store_si128((__m128i*)(dst +  8 * i_dst + 0), in08[0]);
    _mm_store_si128((__m128i*)(dst +  8 * i_dst + 8), in08[1]);
    _mm_store_si128((__m128i*)(dst +  9 * i_dst + 0), in09[0]);
    _mm_store_si128((__m128i*)(dst +  9 * i_dst + 8), in09[1]);
    _mm_store_si128((__m128i*)(dst + 10 * i_dst + 0), in10[0]);
    _mm_store_si128((__m128i*)(dst + 10 * i_dst + 8), in10[1]);
    _mm_store_si128((__m128i*)(dst + 11 * i_dst + 0), in11[0]);
    _mm_store_si128((__m128i*)(dst + 11 * i_dst + 8), in11[1]);
    _mm_store_si128((__m128i*)(dst + 12 * i_dst + 0), in12[0]);
    _mm_store_si128((__m128i*)(dst + 12 * i_dst + 8), in12[1]);
    _mm_store_si128((__m128i*)(dst + 13 * i_dst + 0), in13[0]);
    _mm_store_si128((__m128i*)(dst + 13 * i_dst + 8), in13[1]);
    _mm_store_si128((__m128i*)(dst + 14 * i_dst + 0), in14[0]);
    _mm_store_si128((__m128i*)(dst + 14 * i_dst + 8), in14[1]);
    _mm_store_si128((__m128i*)(dst + 15 * i_dst + 0), in15[0]);
    _mm_store_si128((__m128i*)(dst + 15 * i_dst + 8), in15[1]);
}

/* ---------------------------------------------------------------------------
 */
void idct_c_32x32_sse128(const coeff_t *src, coeff_t *dst, int i_dst)
{
    int a_flag = i_dst & 0x01;
    //int shift1 = 5;
    int shift2 = 20 - g_bit_depth - a_flag;
    //int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = g_bit_depth + 1 + a_flag;

    const __m128i c16_p45_p45 = _mm_set1_epi32(0x002D002D);
    const __m128i c16_p43_p44 = _mm_set1_epi32(0x002B002C);
    const __m128i c16_p39_p41 = _mm_set1_epi32(0x00270029);
    const __m128i c16_p34_p36 = _mm_set1_epi32(0x00220024);
    const __m128i c16_p27_p30 = _mm_set1_epi32(0x001B001E);
    const __m128i c16_p19_p23 = _mm_set1_epi32(0x00130017);
    const __m128i c16_p11_p15 = _mm_set1_epi32(0x000B000F);
    const __m128i c16_p02_p07 = _mm_set1_epi32(0x00020007);
    const __m128i c16_p41_p45 = _mm_set1_epi32(0x0029002D);
    const __m128i c16_p23_p34 = _mm_set1_epi32(0x00170022);
    const __m128i c16_n02_p11 = _mm_set1_epi32(0xFFFE000B);
    const __m128i c16_n27_n15 = _mm_set1_epi32(0xFFE5FFF1);
    const __m128i c16_n43_n36 = _mm_set1_epi32(0xFFD5FFDC);
    const __m128i c16_n44_n45 = _mm_set1_epi32(0xFFD4FFD3);
    const __m128i c16_n30_n39 = _mm_set1_epi32(0xFFE2FFD9);
    const __m128i c16_n07_n19 = _mm_set1_epi32(0xFFF9FFED);
    const __m128i c16_p34_p44 = _mm_set1_epi32(0x0022002C);
    const __m128i c16_n07_p15 = _mm_set1_epi32(0xFFF9000F);
    const __m128i c16_n41_n27 = _mm_set1_epi32(0xFFD7FFE5);
    const __m128i c16_n39_n45 = _mm_set1_epi32(0xFFD9FFD3);
    const __m128i c16_n02_n23 = _mm_set1_epi32(0xFFFEFFE9);
    const __m128i c16_p36_p19 = _mm_set1_epi32(0x00240013);
    const __m128i c16_p43_p45 = _mm_set1_epi32(0x002B002D);
    const __m128i c16_p11_p30 = _mm_set1_epi32(0x000B001E);
    const __m128i c16_p23_p43 = _mm_set1_epi32(0x0017002B);
    const __m128i c16_n34_n07 = _mm_set1_epi32(0xFFDEFFF9);
    const __m128i c16_n36_n45 = _mm_set1_epi32(0xFFDCFFD3);
    const __m128i c16_p19_n11 = _mm_set1_epi32(0x0013FFF5);
    const __m128i c16_p44_p41 = _mm_set1_epi32(0x002C0029);
    const __m128i c16_n02_p27 = _mm_set1_epi32(0xFFFE001B);
    const __m128i c16_n45_n30 = _mm_set1_epi32(0xFFD3FFE2);
    const __m128i c16_n15_n39 = _mm_set1_epi32(0xFFF1FFD9);
    const __m128i c16_p11_p41 = _mm_set1_epi32(0x000B0029);
    const __m128i c16_n45_n27 = _mm_set1_epi32(0xFFD3FFE5);
    const __m128i c16_p07_n30 = _mm_set1_epi32(0x0007FFE2);
    const __m128i c16_p43_p39 = _mm_set1_epi32(0x002B0027);
    const __m128i c16_n23_p15 = _mm_set1_epi32(0xFFE9000F);
    const __m128i c16_n34_n45 = _mm_set1_epi32(0xFFDEFFD3);
    const __m128i c16_p36_p02 = _mm_set1_epi32(0x00240002);
    const __m128i c16_p19_p44 = _mm_set1_epi32(0x0013002C);
    const __m128i c16_n02_p39 = _mm_set1_epi32(0xFFFE0027);
    const __m128i c16_n36_n41 = _mm_set1_epi32(0xFFDCFFD7);
    const __m128i c16_p43_p07 = _mm_set1_epi32(0x002B0007);
    const __m128i c16_n11_p34 = _mm_set1_epi32(0xFFF50022);
    const __m128i c16_n30_n44 = _mm_set1_epi32(0xFFE2FFD4);
    const __m128i c16_p45_p15 = _mm_set1_epi32(0x002D000F);
    const __m128i c16_n19_p27 = _mm_set1_epi32(0xFFED001B);
    const __m128i c16_n23_n45 = _mm_set1_epi32(0xFFE9FFD3);
    const __m128i c16_n15_p36 = _mm_set1_epi32(0xFFF10024);
    const __m128i c16_n11_n45 = _mm_set1_epi32(0xFFF5FFD3);
    const __m128i c16_p34_p39 = _mm_set1_epi32(0x00220027);
    const __m128i c16_n45_n19 = _mm_set1_epi32(0xFFD3FFED);
    const __m128i c16_p41_n07 = _mm_set1_epi32(0x0029FFF9);
    const __m128i c16_n23_p30 = _mm_set1_epi32(0xFFE9001E);
    const __m128i c16_n02_n44 = _mm_set1_epi32(0xFFFEFFD4);
    const __m128i c16_p27_p43 = _mm_set1_epi32(0x001B002B);
    const __m128i c16_n27_p34 = _mm_set1_epi32(0xFFE50022);
    const __m128i c16_p19_n39 = _mm_set1_epi32(0x0013FFD9);
    const __m128i c16_n11_p43 = _mm_set1_epi32(0xFFF5002B);
    const __m128i c16_p02_n45 = _mm_set1_epi32(0x0002FFD3);
    const __m128i c16_p07_p45 = _mm_set1_epi32(0x0007002D);
    const __m128i c16_n15_n44 = _mm_set1_epi32(0xFFF1FFD4);
    const __m128i c16_p23_p41 = _mm_set1_epi32(0x00170029);
    const __m128i c16_n30_n36 = _mm_set1_epi32(0xFFE2FFDC);
    const __m128i c16_n36_p30 = _mm_set1_epi32(0xFFDC001E);
    const __m128i c16_p41_n23 = _mm_set1_epi32(0x0029FFE9);
    const __m128i c16_n44_p15 = _mm_set1_epi32(0xFFD4000F);
    const __m128i c16_p45_n07 = _mm_set1_epi32(0x002DFFF9);
    const __m128i c16_n45_n02 = _mm_set1_epi32(0xFFD3FFFE);
    const __m128i c16_p43_p11 = _mm_set1_epi32(0x002B000B);
    const __m128i c16_n39_n19 = _mm_set1_epi32(0xFFD9FFED);
    const __m128i c16_p34_p27 = _mm_set1_epi32(0x0022001B);
    const __m128i c16_n43_p27 = _mm_set1_epi32(0xFFD5001B);
    const __m128i c16_p44_n02 = _mm_set1_epi32(0x002CFFFE);
    const __m128i c16_n30_n23 = _mm_set1_epi32(0xFFE2FFE9);
    const __m128i c16_p07_p41 = _mm_set1_epi32(0x00070029);
    const __m128i c16_p19_n45 = _mm_set1_epi32(0x0013FFD3);
    const __m128i c16_n39_p34 = _mm_set1_epi32(0xFFD90022);
    const __m128i c16_p45_n11 = _mm_set1_epi32(0x002DFFF5);
    const __m128i c16_n36_n15 = _mm_set1_epi32(0xFFDCFFF1);
    const __m128i c16_n45_p23 = _mm_set1_epi32(0xFFD30017);
    const __m128i c16_p27_p19 = _mm_set1_epi32(0x001B0013);
    const __m128i c16_p15_n45 = _mm_set1_epi32(0x000FFFD3);
    const __m128i c16_n44_p30 = _mm_set1_epi32(0xFFD4001E);
    const __m128i c16_p34_p11 = _mm_set1_epi32(0x0022000B);
    const __m128i c16_p07_n43 = _mm_set1_epi32(0x0007FFD5);
    const __m128i c16_n41_p36 = _mm_set1_epi32(0xFFD70024);
    const __m128i c16_p39_p02 = _mm_set1_epi32(0x00270002);
    const __m128i c16_n44_p19 = _mm_set1_epi32(0xFFD40013);
    const __m128i c16_n02_p36 = _mm_set1_epi32(0xFFFE0024);
    const __m128i c16_p45_n34 = _mm_set1_epi32(0x002DFFDE);
    const __m128i c16_n15_n23 = _mm_set1_epi32(0xFFF1FFE9);
    const __m128i c16_n39_p43 = _mm_set1_epi32(0xFFD9002B);
    const __m128i c16_p30_p07 = _mm_set1_epi32(0x001E0007);
    const __m128i c16_p27_n45 = _mm_set1_epi32(0x001BFFD3);
    const __m128i c16_n41_p11 = _mm_set1_epi32(0xFFD7000B);
    const __m128i c16_n39_p15 = _mm_set1_epi32(0xFFD9000F);
    const __m128i c16_n30_p45 = _mm_set1_epi32(0xFFE2002D);
    const __m128i c16_p27_p02 = _mm_set1_epi32(0x001B0002);
    const __m128i c16_p41_n44 = _mm_set1_epi32(0x0029FFD4);
    const __m128i c16_n11_n19 = _mm_set1_epi32(0xFFF5FFED);
    const __m128i c16_n45_p36 = _mm_set1_epi32(0xFFD30024);
    const __m128i c16_n07_p34 = _mm_set1_epi32(0xFFF90022);
    const __m128i c16_p43_n23 = _mm_set1_epi32(0x002BFFE9);
    const __m128i c16_n30_p11 = _mm_set1_epi32(0xFFE2000B);
    const __m128i c16_n45_p43 = _mm_set1_epi32(0xFFD3002B);
    const __m128i c16_n19_p36 = _mm_set1_epi32(0xFFED0024);
    const __m128i c16_p23_n02 = _mm_set1_epi32(0x0017FFFE);
    const __m128i c16_p45_n39 = _mm_set1_epi32(0x002DFFD9);
    const __m128i c16_p27_n41 = _mm_set1_epi32(0x001BFFD7);
    const __m128i c16_n15_n07 = _mm_set1_epi32(0xFFF1FFF9);
    const __m128i c16_n44_p34 = _mm_set1_epi32(0xFFD40022);
    const __m128i c16_n19_p07 = _mm_set1_epi32(0xFFED0007);
    const __m128i c16_n39_p30 = _mm_set1_epi32(0xFFD9001E);
    const __m128i c16_n45_p44 = _mm_set1_epi32(0xFFD3002C);
    const __m128i c16_n36_p43 = _mm_set1_epi32(0xFFDC002B);
    const __m128i c16_n15_p27 = _mm_set1_epi32(0xFFF1001B);
    const __m128i c16_p11_p02 = _mm_set1_epi32(0x000B0002);
    const __m128i c16_p34_n23 = _mm_set1_epi32(0x0022FFE9);
    const __m128i c16_p45_n41 = _mm_set1_epi32(0x002DFFD7);
    const __m128i c16_n07_p02 = _mm_set1_epi32(0xFFF90002);
    const __m128i c16_n15_p11 = _mm_set1_epi32(0xFFF1000B);
    const __m128i c16_n23_p19 = _mm_set1_epi32(0xFFE90013);
    const __m128i c16_n30_p27 = _mm_set1_epi32(0xFFE2001B);
    const __m128i c16_n36_p34 = _mm_set1_epi32(0xFFDC0022);
    const __m128i c16_n41_p39 = _mm_set1_epi32(0xFFD70027);
    const __m128i c16_n44_p43 = _mm_set1_epi32(0xFFD4002B);
    const __m128i c16_n45_p45 = _mm_set1_epi32(0xFFD3002D);

    //  const __m128i c16_p43_p45 = _mm_set1_epi32(0x002B002D);
    const __m128i c16_p35_p40 = _mm_set1_epi32(0x00230028);
    const __m128i c16_p21_p29 = _mm_set1_epi32(0x0015001D);
    const __m128i c16_p04_p13 = _mm_set1_epi32(0x0004000D);
    const __m128i c16_p29_p43 = _mm_set1_epi32(0x001D002B);
    const __m128i c16_n21_p04 = _mm_set1_epi32(0xFFEB0004);
    const __m128i c16_n45_n40 = _mm_set1_epi32(0xFFD3FFD8);
    const __m128i c16_n13_n35 = _mm_set1_epi32(0xFFF3FFDD);
    const __m128i c16_p04_p40 = _mm_set1_epi32(0x00040028);
    const __m128i c16_n43_n35 = _mm_set1_epi32(0xFFD5FFDD);
    const __m128i c16_p29_n13 = _mm_set1_epi32(0x001DFFF3);
    const __m128i c16_p21_p45 = _mm_set1_epi32(0x0015002D);
    const __m128i c16_n21_p35 = _mm_set1_epi32(0xFFEB0023);
    const __m128i c16_p04_n43 = _mm_set1_epi32(0x0004FFD5);
    const __m128i c16_p13_p45 = _mm_set1_epi32(0x000D002D);
    const __m128i c16_n29_n40 = _mm_set1_epi32(0xFFE3FFD8);
    const __m128i c16_n40_p29 = _mm_set1_epi32(0xFFD8001D);
    const __m128i c16_p45_n13 = _mm_set1_epi32(0x002DFFF3);
    const __m128i c16_n43_n04 = _mm_set1_epi32(0xFFD5FFFC);
    const __m128i c16_p35_p21 = _mm_set1_epi32(0x00230015);
    const __m128i c16_n45_p21 = _mm_set1_epi32(0xFFD30015);
    const __m128i c16_p13_p29 = _mm_set1_epi32(0x000D001D);
    const __m128i c16_p35_n43 = _mm_set1_epi32(0x0023FFD5);
    const __m128i c16_n40_p04 = _mm_set1_epi32(0xFFD80004);
    const __m128i c16_n35_p13 = _mm_set1_epi32(0xFFDD000D);
    const __m128i c16_n40_p45 = _mm_set1_epi32(0xFFD8002D);
    const __m128i c16_p04_p21 = _mm_set1_epi32(0x00040015);
    const __m128i c16_p43_n29 = _mm_set1_epi32(0x002BFFE3);
    const __m128i c16_n13_p04 = _mm_set1_epi32(0xFFF30004);
    const __m128i c16_n29_p21 = _mm_set1_epi32(0xFFE30015);
    const __m128i c16_n40_p35 = _mm_set1_epi32(0xFFD80023);
    //  const __m128i c16_n45_p43 = _mm_set1_epi32(0xFFD3002B);

    const __m128i c16_p38_p44 = _mm_set1_epi32(0x0026002C);
    const __m128i c16_p09_p25 = _mm_set1_epi32(0x00090019);
    const __m128i c16_n09_p38 = _mm_set1_epi32(0xFFF70026);
    const __m128i c16_n25_n44 = _mm_set1_epi32(0xFFE7FFD4);

    const __m128i c16_n44_p25 = _mm_set1_epi32(0xFFD40019);
    const __m128i c16_p38_p09 = _mm_set1_epi32(0x00260009);
    const __m128i c16_n25_p09 = _mm_set1_epi32(0xFFE70009);
    const __m128i c16_n44_p38 = _mm_set1_epi32(0xFFD40026);

    const __m128i c16_p17_p42 = _mm_set1_epi32(0x0011002A);
    const __m128i c16_n42_p17 = _mm_set1_epi32(0xFFD60011);

    const __m128i c16_p32_p32 = _mm_set1_epi32(0x00200020);
    const __m128i c16_n32_p32 = _mm_set1_epi32(0xFFE00020);

    __m128i c32_rnd = _mm_set1_epi32(16);   // add1

    int nShift = 5;
    int i, pass, part;

    // DCT1
    __m128i in00[4], in01[4], in02[4], in03[4], in04[4], in05[4], in06[4], in07[4], in08[4], in09[4], in10[4], in11[4], in12[4], in13[4], in14[4], in15[4];
    __m128i in16[4], in17[4], in18[4], in19[4], in20[4], in21[4], in22[4], in23[4], in24[4], in25[4], in26[4], in27[4], in28[4], in29[4], in30[4], in31[4];
    __m128i res00[4], res01[4], res02[4], res03[4], res04[4], res05[4], res06[4], res07[4], res08[4], res09[4], res10[4], res11[4], res12[4], res13[4], res14[4], res15[4];
    __m128i res16[4], res17[4], res18[4], res19[4], res20[4], res21[4], res22[4], res23[4], res24[4], res25[4], res26[4], res27[4], res28[4], res29[4], res30[4], res31[4];

    i_dst &= 0xFE;    /* remember to remove the flag bit */

    for (i = 0; i < 4; i++) {
        const int offset = (i << 3);

        in00[i] = _mm_loadu_si128((const __m128i*)&src[ 0 * 32 + offset]);
        in01[i] = _mm_loadu_si128((const __m128i*)&src[ 1 * 32 + offset]);
        in02[i] = _mm_loadu_si128((const __m128i*)&src[ 2 * 32 + offset]);
        in03[i] = _mm_loadu_si128((const __m128i*)&src[ 3 * 32 + offset]);
        in04[i] = _mm_loadu_si128((const __m128i*)&src[ 4 * 32 + offset]);
        in05[i] = _mm_loadu_si128((const __m128i*)&src[ 5 * 32 + offset]);
        in06[i] = _mm_loadu_si128((const __m128i*)&src[ 6 * 32 + offset]);
        in07[i] = _mm_loadu_si128((const __m128i*)&src[ 7 * 32 + offset]);
        in08[i] = _mm_loadu_si128((const __m128i*)&src[ 8 * 32 + offset]);
        in09[i] = _mm_loadu_si128((const __m128i*)&src[ 9 * 32 + offset]);
        in10[i] = _mm_loadu_si128((const __m128i*)&src[10 * 32 + offset]);
        in11[i] = _mm_loadu_si128((const __m128i*)&src[11 * 32 + offset]);
        in12[i] = _mm_loadu_si128((const __m128i*)&src[12 * 32 + offset]);
        in13[i] = _mm_loadu_si128((const __m128i*)&src[13 * 32 + offset]);
        in14[i] = _mm_loadu_si128((const __m128i*)&src[14 * 32 + offset]);
        in15[i] = _mm_loadu_si128((const __m128i*)&src[15 * 32 + offset]);
        in16[i] = _mm_loadu_si128((const __m128i*)&src[16 * 32 + offset]);
        in17[i] = _mm_loadu_si128((const __m128i*)&src[17 * 32 + offset]);
        in18[i] = _mm_loadu_si128((const __m128i*)&src[18 * 32 + offset]);
        in19[i] = _mm_loadu_si128((const __m128i*)&src[19 * 32 + offset]);
        in20[i] = _mm_loadu_si128((const __m128i*)&src[20 * 32 + offset]);
        in21[i] = _mm_loadu_si128((const __m128i*)&src[21 * 32 + offset]);
        in22[i] = _mm_loadu_si128((const __m128i*)&src[22 * 32 + offset]);
        in23[i] = _mm_loadu_si128((const __m128i*)&src[23 * 32 + offset]);
        in24[i] = _mm_loadu_si128((const __m128i*)&src[24 * 32 + offset]);
        in25[i] = _mm_loadu_si128((const __m128i*)&src[25 * 32 + offset]);
        in26[i] = _mm_loadu_si128((const __m128i*)&src[26 * 32 + offset]);
        in27[i] = _mm_loadu_si128((const __m128i*)&src[27 * 32 + offset]);
        in28[i] = _mm_loadu_si128((const __m128i*)&src[28 * 32 + offset]);
        in29[i] = _mm_loadu_si128((const __m128i*)&src[29 * 32 + offset]);
        in30[i] = _mm_loadu_si128((const __m128i*)&src[30 * 32 + offset]);
        in31[i] = _mm_loadu_si128((const __m128i*)&src[31 * 32 + offset]);
    }

    for (pass = 0; pass < 2; pass++) {
        if (pass == 1) {
            c32_rnd = _mm_set1_epi32(1 << (shift2 - 1));    // add2
            nShift = shift2;
        }

        for (part = 0; part < 4; part++) {
            const __m128i T_00_00A = _mm_unpacklo_epi16(in01[part], in03[part]);    // [33 13 32 12 31 11 30 10]
            const __m128i T_00_00B = _mm_unpackhi_epi16(in01[part], in03[part]);    // [37 17 36 16 35 15 34 14]
            const __m128i T_00_01A = _mm_unpacklo_epi16(in05[part], in07[part]);    // [ ]
            const __m128i T_00_01B = _mm_unpackhi_epi16(in05[part], in07[part]);    // [ ]
            const __m128i T_00_02A = _mm_unpacklo_epi16(in09[part], in11[part]);    // [ ]
            const __m128i T_00_02B = _mm_unpackhi_epi16(in09[part], in11[part]);    // [ ]
            const __m128i T_00_03A = _mm_unpacklo_epi16(in13[part], in15[part]);    // [ ]
            const __m128i T_00_03B = _mm_unpackhi_epi16(in13[part], in15[part]);    // [ ]
            const __m128i T_00_04A = _mm_unpacklo_epi16(in17[part], in19[part]);    // [ ]
            const __m128i T_00_04B = _mm_unpackhi_epi16(in17[part], in19[part]);    // [ ]
            const __m128i T_00_05A = _mm_unpacklo_epi16(in21[part], in23[part]);    // [ ]
            const __m128i T_00_05B = _mm_unpackhi_epi16(in21[part], in23[part]);    // [ ]
            const __m128i T_00_06A = _mm_unpacklo_epi16(in25[part], in27[part]);    // [ ]
            const __m128i T_00_06B = _mm_unpackhi_epi16(in25[part], in27[part]);    // [ ]
            const __m128i T_00_07A = _mm_unpacklo_epi16(in29[part], in31[part]);    //
            const __m128i T_00_07B = _mm_unpackhi_epi16(in29[part], in31[part]);    // [ ]

            const __m128i T_00_08A = _mm_unpacklo_epi16(in02[part], in06[part]);    // [ ]
            const __m128i T_00_08B = _mm_unpackhi_epi16(in02[part], in06[part]);    // [ ]
            const __m128i T_00_09A = _mm_unpacklo_epi16(in10[part], in14[part]);    // [ ]
            const __m128i T_00_09B = _mm_unpackhi_epi16(in10[part], in14[part]);    // [ ]
            const __m128i T_00_10A = _mm_unpacklo_epi16(in18[part], in22[part]);    // [ ]
            const __m128i T_00_10B = _mm_unpackhi_epi16(in18[part], in22[part]);    // [ ]
            const __m128i T_00_11A = _mm_unpacklo_epi16(in26[part], in30[part]);    // [ ]
            const __m128i T_00_11B = _mm_unpackhi_epi16(in26[part], in30[part]);    // [ ]

            const __m128i T_00_12A = _mm_unpacklo_epi16(in04[part], in12[part]);    // [ ]
            const __m128i T_00_12B = _mm_unpackhi_epi16(in04[part], in12[part]);    // [ ]
            const __m128i T_00_13A = _mm_unpacklo_epi16(in20[part], in28[part]);    // [ ]
            const __m128i T_00_13B = _mm_unpackhi_epi16(in20[part], in28[part]);    // [ ]

            const __m128i T_00_14A = _mm_unpacklo_epi16(in08[part], in24[part]);    //
            const __m128i T_00_14B = _mm_unpackhi_epi16(in08[part], in24[part]);    // [ ]
            const __m128i T_00_15A = _mm_unpacklo_epi16(in00[part], in16[part]);    //
            const __m128i T_00_15B = _mm_unpackhi_epi16(in00[part], in16[part]);    // [ ]

            __m128i O00A, O01A, O02A, O03A, O04A, O05A, O06A, O07A, O08A, O09A, O10A, O11A, O12A, O13A, O14A, O15A;
            __m128i O00B, O01B, O02B, O03B, O04B, O05B, O06B, O07B, O08B, O09B, O10B, O11B, O12B, O13B, O14B, O15B;
            __m128i EO0A, EO1A, EO2A, EO3A, EO4A, EO5A, EO6A, EO7A;
            __m128i EO0B, EO1B, EO2B, EO3B, EO4B, EO5B, EO6B, EO7B;
            {
                __m128i T00, T01, T02, T03;
#define COMPUTE_ROW(r0103, r0507, r0911, r1315, r1719, r2123, r2527, r2931, c0103, c0507, c0911, c1315, c1719, c2123, c2527, c2931, row) \
    T00 = _mm_add_epi32(_mm_madd_epi16(r0103, c0103), _mm_madd_epi16(r0507, c0507)); \
    T01 = _mm_add_epi32(_mm_madd_epi16(r0911, c0911), _mm_madd_epi16(r1315, c1315)); \
    T02 = _mm_add_epi32(_mm_madd_epi16(r1719, c1719), _mm_madd_epi16(r2123, c2123)); \
    T03 = _mm_add_epi32(_mm_madd_epi16(r2527, c2527), _mm_madd_epi16(r2931, c2931)); \
    row = _mm_add_epi32(_mm_add_epi32(T00, T01), _mm_add_epi32(T02, T03));

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
#undef COMPUTE_ROW
            }

            {
                __m128i T00, T01;
#define COMPUTE_ROW(row0206, row1014, row1822, row2630, c0206, c1014, c1822, c2630, row) \
    T00 = _mm_add_epi32(_mm_madd_epi16(row0206, c0206), _mm_madd_epi16(row1014, c1014)); \
    T01 = _mm_add_epi32(_mm_madd_epi16(row1822, c1822), _mm_madd_epi16(row2630, c2630)); \
    row = _mm_add_epi32(T00, T01);

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
#undef COMPUTE_ROW
            }
            {
                const __m128i EEO0A = _mm_add_epi32(_mm_madd_epi16(T_00_12A, c16_p38_p44), _mm_madd_epi16(T_00_13A, c16_p09_p25));
                const __m128i EEO1A = _mm_add_epi32(_mm_madd_epi16(T_00_12A, c16_n09_p38), _mm_madd_epi16(T_00_13A, c16_n25_n44));
                const __m128i EEO2A = _mm_add_epi32(_mm_madd_epi16(T_00_12A, c16_n44_p25), _mm_madd_epi16(T_00_13A, c16_p38_p09));
                const __m128i EEO3A = _mm_add_epi32(_mm_madd_epi16(T_00_12A, c16_n25_p09), _mm_madd_epi16(T_00_13A, c16_n44_p38));
                const __m128i EEO0B = _mm_add_epi32(_mm_madd_epi16(T_00_12B, c16_p38_p44), _mm_madd_epi16(T_00_13B, c16_p09_p25));
                const __m128i EEO1B = _mm_add_epi32(_mm_madd_epi16(T_00_12B, c16_n09_p38), _mm_madd_epi16(T_00_13B, c16_n25_n44));
                const __m128i EEO2B = _mm_add_epi32(_mm_madd_epi16(T_00_12B, c16_n44_p25), _mm_madd_epi16(T_00_13B, c16_p38_p09));
                const __m128i EEO3B = _mm_add_epi32(_mm_madd_epi16(T_00_12B, c16_n25_p09), _mm_madd_epi16(T_00_13B, c16_n44_p38));

                const __m128i EEEO0A = _mm_madd_epi16(T_00_14A, c16_p17_p42);
                const __m128i EEEO0B = _mm_madd_epi16(T_00_14B, c16_p17_p42);
                const __m128i EEEO1A = _mm_madd_epi16(T_00_14A, c16_n42_p17);
                const __m128i EEEO1B = _mm_madd_epi16(T_00_14B, c16_n42_p17);

                const __m128i EEEE0A = _mm_madd_epi16(T_00_15A, c16_p32_p32);
                const __m128i EEEE0B = _mm_madd_epi16(T_00_15B, c16_p32_p32);
                const __m128i EEEE1A = _mm_madd_epi16(T_00_15A, c16_n32_p32);
                const __m128i EEEE1B = _mm_madd_epi16(T_00_15B, c16_n32_p32);

                const __m128i EEE0A = _mm_add_epi32(EEEE0A, EEEO0A);    // EEE0 = EEEE0 + EEEO0
                const __m128i EEE0B = _mm_add_epi32(EEEE0B, EEEO0B);
                const __m128i EEE1A = _mm_add_epi32(EEEE1A, EEEO1A);    // EEE1 = EEEE1 + EEEO1
                const __m128i EEE1B = _mm_add_epi32(EEEE1B, EEEO1B);
                const __m128i EEE3A = _mm_sub_epi32(EEEE0A, EEEO0A);    // EEE2 = EEEE0 - EEEO0
                const __m128i EEE3B = _mm_sub_epi32(EEEE0B, EEEO0B);
                const __m128i EEE2A = _mm_sub_epi32(EEEE1A, EEEO1A);    // EEE3 = EEEE1 - EEEO1
                const __m128i EEE2B = _mm_sub_epi32(EEEE1B, EEEO1B);

                const __m128i EE0A = _mm_add_epi32(EEE0A, EEO0A);       // EE0 = EEE0 + EEO0
                const __m128i EE0B = _mm_add_epi32(EEE0B, EEO0B);
                const __m128i EE1A = _mm_add_epi32(EEE1A, EEO1A);       // EE1 = EEE1 + EEO1
                const __m128i EE1B = _mm_add_epi32(EEE1B, EEO1B);
                const __m128i EE2A = _mm_add_epi32(EEE2A, EEO2A);       // EE2 = EEE0 + EEO0
                const __m128i EE2B = _mm_add_epi32(EEE2B, EEO2B);
                const __m128i EE3A = _mm_add_epi32(EEE3A, EEO3A);       // EE3 = EEE1 + EEO1
                const __m128i EE3B = _mm_add_epi32(EEE3B, EEO3B);
                const __m128i EE7A = _mm_sub_epi32(EEE0A, EEO0A);       // EE7 = EEE0 - EEO0
                const __m128i EE7B = _mm_sub_epi32(EEE0B, EEO0B);
                const __m128i EE6A = _mm_sub_epi32(EEE1A, EEO1A);       // EE6 = EEE1 - EEO1
                const __m128i EE6B = _mm_sub_epi32(EEE1B, EEO1B);
                const __m128i EE5A = _mm_sub_epi32(EEE2A, EEO2A);       // EE5 = EEE0 - EEO0
                const __m128i EE5B = _mm_sub_epi32(EEE2B, EEO2B);
                const __m128i EE4A = _mm_sub_epi32(EEE3A, EEO3A);       // EE4 = EEE1 - EEO1
                const __m128i EE4B = _mm_sub_epi32(EEE3B, EEO3B);

                const __m128i E0A = _mm_add_epi32(EE0A, EO0A);          // E0 = EE0 + EO0
                const __m128i E0B = _mm_add_epi32(EE0B, EO0B);
                const __m128i E1A = _mm_add_epi32(EE1A, EO1A);          // E1 = EE1 + EO1
                const __m128i E1B = _mm_add_epi32(EE1B, EO1B);
                const __m128i E2A = _mm_add_epi32(EE2A, EO2A);          // E2 = EE2 + EO2
                const __m128i E2B = _mm_add_epi32(EE2B, EO2B);
                const __m128i E3A = _mm_add_epi32(EE3A, EO3A);          // E3 = EE3 + EO3
                const __m128i E3B = _mm_add_epi32(EE3B, EO3B);
                const __m128i E4A = _mm_add_epi32(EE4A, EO4A);          // E4 =
                const __m128i E4B = _mm_add_epi32(EE4B, EO4B);
                const __m128i E5A = _mm_add_epi32(EE5A, EO5A);          // E5 =
                const __m128i E5B = _mm_add_epi32(EE5B, EO5B);
                const __m128i E6A = _mm_add_epi32(EE6A, EO6A);          // E6 =
                const __m128i E6B = _mm_add_epi32(EE6B, EO6B);
                const __m128i E7A = _mm_add_epi32(EE7A, EO7A);          // E7 =
                const __m128i E7B = _mm_add_epi32(EE7B, EO7B);
                const __m128i EFA = _mm_sub_epi32(EE0A, EO0A);          // EF = EE0 - EO0
                const __m128i EFB = _mm_sub_epi32(EE0B, EO0B);
                const __m128i EEA = _mm_sub_epi32(EE1A, EO1A);          // EE = EE1 - EO1
                const __m128i EEB = _mm_sub_epi32(EE1B, EO1B);
                const __m128i EDA = _mm_sub_epi32(EE2A, EO2A);          // ED = EE2 - EO2
                const __m128i EDB = _mm_sub_epi32(EE2B, EO2B);
                const __m128i ECA = _mm_sub_epi32(EE3A, EO3A);          // EC = EE3 - EO3
                const __m128i ECB = _mm_sub_epi32(EE3B, EO3B);
                const __m128i EBA = _mm_sub_epi32(EE4A, EO4A);          // EB =
                const __m128i EBB = _mm_sub_epi32(EE4B, EO4B);
                const __m128i EAA = _mm_sub_epi32(EE5A, EO5A);          // EA =
                const __m128i EAB = _mm_sub_epi32(EE5B, EO5B);
                const __m128i E9A = _mm_sub_epi32(EE6A, EO6A);          // E9 =
                const __m128i E9B = _mm_sub_epi32(EE6B, EO6B);
                const __m128i E8A = _mm_sub_epi32(EE7A, EO7A);          // E8 =
                const __m128i E8B = _mm_sub_epi32(EE7B, EO7B);

                const __m128i T10A = _mm_add_epi32(E0A, c32_rnd);       // E0 + rnd
                const __m128i T10B = _mm_add_epi32(E0B, c32_rnd);
                const __m128i T11A = _mm_add_epi32(E1A, c32_rnd);       // E1 + rnd
                const __m128i T11B = _mm_add_epi32(E1B, c32_rnd);
                const __m128i T12A = _mm_add_epi32(E2A, c32_rnd);       // E2 + rnd
                const __m128i T12B = _mm_add_epi32(E2B, c32_rnd);
                const __m128i T13A = _mm_add_epi32(E3A, c32_rnd);       // E3 + rnd
                const __m128i T13B = _mm_add_epi32(E3B, c32_rnd);
                const __m128i T14A = _mm_add_epi32(E4A, c32_rnd);       // E4 + rnd
                const __m128i T14B = _mm_add_epi32(E4B, c32_rnd);
                const __m128i T15A = _mm_add_epi32(E5A, c32_rnd);       // E5 + rnd
                const __m128i T15B = _mm_add_epi32(E5B, c32_rnd);
                const __m128i T16A = _mm_add_epi32(E6A, c32_rnd);       // E6 + rnd
                const __m128i T16B = _mm_add_epi32(E6B, c32_rnd);
                const __m128i T17A = _mm_add_epi32(E7A, c32_rnd);       // E7 + rnd
                const __m128i T17B = _mm_add_epi32(E7B, c32_rnd);
                const __m128i T18A = _mm_add_epi32(E8A, c32_rnd);       // E8 + rnd
                const __m128i T18B = _mm_add_epi32(E8B, c32_rnd);
                const __m128i T19A = _mm_add_epi32(E9A, c32_rnd);       // E9 + rnd
                const __m128i T19B = _mm_add_epi32(E9B, c32_rnd);
                const __m128i T1AA = _mm_add_epi32(EAA, c32_rnd);       // E10 + rnd
                const __m128i T1AB = _mm_add_epi32(EAB, c32_rnd);
                const __m128i T1BA = _mm_add_epi32(EBA, c32_rnd);       // E11 + rnd
                const __m128i T1BB = _mm_add_epi32(EBB, c32_rnd);
                const __m128i T1CA = _mm_add_epi32(ECA, c32_rnd);       // E12 + rnd
                const __m128i T1CB = _mm_add_epi32(ECB, c32_rnd);
                const __m128i T1DA = _mm_add_epi32(EDA, c32_rnd);       // E13 + rnd
                const __m128i T1DB = _mm_add_epi32(EDB, c32_rnd);
                const __m128i T1EA = _mm_add_epi32(EEA, c32_rnd);       // E14 + rnd
                const __m128i T1EB = _mm_add_epi32(EEB, c32_rnd);
                const __m128i T1FA = _mm_add_epi32(EFA, c32_rnd);       // E15 + rnd
                const __m128i T1FB = _mm_add_epi32(EFB, c32_rnd);

                const __m128i T2_00A = _mm_add_epi32(T10A, O00A);       // E0 + O0 + rnd
                const __m128i T2_00B = _mm_add_epi32(T10B, O00B);
                const __m128i T2_01A = _mm_add_epi32(T11A, O01A);       // E1 + O1 + rnd
                const __m128i T2_01B = _mm_add_epi32(T11B, O01B);
                const __m128i T2_02A = _mm_add_epi32(T12A, O02A);       // E2 + O2 + rnd
                const __m128i T2_02B = _mm_add_epi32(T12B, O02B);
                const __m128i T2_03A = _mm_add_epi32(T13A, O03A);       // E3 + O3 + rnd
                const __m128i T2_03B = _mm_add_epi32(T13B, O03B);
                const __m128i T2_04A = _mm_add_epi32(T14A, O04A);       // E4
                const __m128i T2_04B = _mm_add_epi32(T14B, O04B);
                const __m128i T2_05A = _mm_add_epi32(T15A, O05A);       // E5
                const __m128i T2_05B = _mm_add_epi32(T15B, O05B);
                const __m128i T2_06A = _mm_add_epi32(T16A, O06A);       // E6
                const __m128i T2_06B = _mm_add_epi32(T16B, O06B);
                const __m128i T2_07A = _mm_add_epi32(T17A, O07A);       // E7
                const __m128i T2_07B = _mm_add_epi32(T17B, O07B);
                const __m128i T2_08A = _mm_add_epi32(T18A, O08A);       // E8
                const __m128i T2_08B = _mm_add_epi32(T18B, O08B);
                const __m128i T2_09A = _mm_add_epi32(T19A, O09A);       // E9
                const __m128i T2_09B = _mm_add_epi32(T19B, O09B);
                const __m128i T2_10A = _mm_add_epi32(T1AA, O10A);       // E10
                const __m128i T2_10B = _mm_add_epi32(T1AB, O10B);
                const __m128i T2_11A = _mm_add_epi32(T1BA, O11A);       // E11
                const __m128i T2_11B = _mm_add_epi32(T1BB, O11B);
                const __m128i T2_12A = _mm_add_epi32(T1CA, O12A);       // E12
                const __m128i T2_12B = _mm_add_epi32(T1CB, O12B);
                const __m128i T2_13A = _mm_add_epi32(T1DA, O13A);       // E13
                const __m128i T2_13B = _mm_add_epi32(T1DB, O13B);
                const __m128i T2_14A = _mm_add_epi32(T1EA, O14A);       // E14
                const __m128i T2_14B = _mm_add_epi32(T1EB, O14B);
                const __m128i T2_15A = _mm_add_epi32(T1FA, O15A);       // E15
                const __m128i T2_15B = _mm_add_epi32(T1FB, O15B);
                const __m128i T2_31A = _mm_sub_epi32(T10A, O00A);       // E0 - O0 + rnd
                const __m128i T2_31B = _mm_sub_epi32(T10B, O00B);
                const __m128i T2_30A = _mm_sub_epi32(T11A, O01A);       // E1 - O1 + rnd
                const __m128i T2_30B = _mm_sub_epi32(T11B, O01B);
                const __m128i T2_29A = _mm_sub_epi32(T12A, O02A);       // E2 - O2 + rnd
                const __m128i T2_29B = _mm_sub_epi32(T12B, O02B);
                const __m128i T2_28A = _mm_sub_epi32(T13A, O03A);       // E3 - O3 + rnd
                const __m128i T2_28B = _mm_sub_epi32(T13B, O03B);
                const __m128i T2_27A = _mm_sub_epi32(T14A, O04A);       // E4
                const __m128i T2_27B = _mm_sub_epi32(T14B, O04B);
                const __m128i T2_26A = _mm_sub_epi32(T15A, O05A);       // E5
                const __m128i T2_26B = _mm_sub_epi32(T15B, O05B);
                const __m128i T2_25A = _mm_sub_epi32(T16A, O06A);       // E6
                const __m128i T2_25B = _mm_sub_epi32(T16B, O06B);
                const __m128i T2_24A = _mm_sub_epi32(T17A, O07A);       // E7
                const __m128i T2_24B = _mm_sub_epi32(T17B, O07B);
                const __m128i T2_23A = _mm_sub_epi32(T18A, O08A);       //
                const __m128i T2_23B = _mm_sub_epi32(T18B, O08B);
                const __m128i T2_22A = _mm_sub_epi32(T19A, O09A);       //
                const __m128i T2_22B = _mm_sub_epi32(T19B, O09B);
                const __m128i T2_21A = _mm_sub_epi32(T1AA, O10A);       //
                const __m128i T2_21B = _mm_sub_epi32(T1AB, O10B);
                const __m128i T2_20A = _mm_sub_epi32(T1BA, O11A);       //
                const __m128i T2_20B = _mm_sub_epi32(T1BB, O11B);
                const __m128i T2_19A = _mm_sub_epi32(T1CA, O12A);       //
                const __m128i T2_19B = _mm_sub_epi32(T1CB, O12B);
                const __m128i T2_18A = _mm_sub_epi32(T1DA, O13A);       //
                const __m128i T2_18B = _mm_sub_epi32(T1DB, O13B);
                const __m128i T2_17A = _mm_sub_epi32(T1EA, O14A);       //
                const __m128i T2_17B = _mm_sub_epi32(T1EB, O14B);
                const __m128i T2_16A = _mm_sub_epi32(T1FA, O15A);       //
                const __m128i T2_16B = _mm_sub_epi32(T1FB, O15B);

                const __m128i T3_00A = _mm_srai_epi32(T2_00A, nShift);  // [30 20 10 00]
                const __m128i T3_00B = _mm_srai_epi32(T2_00B, nShift);  // [70 60 50 40]
                const __m128i T3_01A = _mm_srai_epi32(T2_01A, nShift);  // [31 21 11 01]
                const __m128i T3_01B = _mm_srai_epi32(T2_01B, nShift);  // [71 61 51 41]
                const __m128i T3_02A = _mm_srai_epi32(T2_02A, nShift);  // [32 22 12 02]
                const __m128i T3_02B = _mm_srai_epi32(T2_02B, nShift);  // [72 62 52 42]
                const __m128i T3_03A = _mm_srai_epi32(T2_03A, nShift);  // [33 23 13 03]
                const __m128i T3_03B = _mm_srai_epi32(T2_03B, nShift);  // [73 63 53 43]
                const __m128i T3_04A = _mm_srai_epi32(T2_04A, nShift);  // [33 24 14 04]
                const __m128i T3_04B = _mm_srai_epi32(T2_04B, nShift);  // [74 64 54 44]
                const __m128i T3_05A = _mm_srai_epi32(T2_05A, nShift);  // [35 25 15 05]
                const __m128i T3_05B = _mm_srai_epi32(T2_05B, nShift);  // [75 65 55 45]
                const __m128i T3_06A = _mm_srai_epi32(T2_06A, nShift);  // [36 26 16 06]
                const __m128i T3_06B = _mm_srai_epi32(T2_06B, nShift);  // [76 66 56 46]
                const __m128i T3_07A = _mm_srai_epi32(T2_07A, nShift);  // [37 27 17 07]
                const __m128i T3_07B = _mm_srai_epi32(T2_07B, nShift);  // [77 67 57 47]
                const __m128i T3_08A = _mm_srai_epi32(T2_08A, nShift);  // [30 20 10 00] x8
                const __m128i T3_08B = _mm_srai_epi32(T2_08B, nShift);  // [70 60 50 40]
                const __m128i T3_09A = _mm_srai_epi32(T2_09A, nShift);  // [31 21 11 01] x9
                const __m128i T3_09B = _mm_srai_epi32(T2_09B, nShift);  // [71 61 51 41]
                const __m128i T3_10A = _mm_srai_epi32(T2_10A, nShift);  // [32 22 12 02] xA
                const __m128i T3_10B = _mm_srai_epi32(T2_10B, nShift);  // [72 62 52 42]
                const __m128i T3_11A = _mm_srai_epi32(T2_11A, nShift);  // [33 23 13 03] xB
                const __m128i T3_11B = _mm_srai_epi32(T2_11B, nShift);  // [73 63 53 43]
                const __m128i T3_12A = _mm_srai_epi32(T2_12A, nShift);  // [33 24 14 04] xC
                const __m128i T3_12B = _mm_srai_epi32(T2_12B, nShift);  // [74 64 54 44]
                const __m128i T3_13A = _mm_srai_epi32(T2_13A, nShift);  // [35 25 15 05] xD
                const __m128i T3_13B = _mm_srai_epi32(T2_13B, nShift);  // [75 65 55 45]
                const __m128i T3_14A = _mm_srai_epi32(T2_14A, nShift);  // [36 26 16 06] xE
                const __m128i T3_14B = _mm_srai_epi32(T2_14B, nShift);  // [76 66 56 46]
                const __m128i T3_15A = _mm_srai_epi32(T2_15A, nShift);  // [37 27 17 07] xF
                const __m128i T3_15B = _mm_srai_epi32(T2_15B, nShift);  // [77 67 57 47]

                const __m128i T3_16A = _mm_srai_epi32(T2_16A, nShift);  // [30 20 10 00]
                const __m128i T3_16B = _mm_srai_epi32(T2_16B, nShift);  // [70 60 50 40]
                const __m128i T3_17A = _mm_srai_epi32(T2_17A, nShift);  // [31 21 11 01]
                const __m128i T3_17B = _mm_srai_epi32(T2_17B, nShift);  // [71 61 51 41]
                const __m128i T3_18A = _mm_srai_epi32(T2_18A, nShift);  // [32 22 12 02]
                const __m128i T3_18B = _mm_srai_epi32(T2_18B, nShift);  // [72 62 52 42]
                const __m128i T3_19A = _mm_srai_epi32(T2_19A, nShift);  // [33 23 13 03]
                const __m128i T3_19B = _mm_srai_epi32(T2_19B, nShift);  // [73 63 53 43]
                const __m128i T3_20A = _mm_srai_epi32(T2_20A, nShift);  // [33 24 14 04]
                const __m128i T3_20B = _mm_srai_epi32(T2_20B, nShift);  // [74 64 54 44]
                const __m128i T3_21A = _mm_srai_epi32(T2_21A, nShift);  // [35 25 15 05]
                const __m128i T3_21B = _mm_srai_epi32(T2_21B, nShift);  // [75 65 55 45]
                const __m128i T3_22A = _mm_srai_epi32(T2_22A, nShift);  // [36 26 16 06]
                const __m128i T3_22B = _mm_srai_epi32(T2_22B, nShift);  // [76 66 56 46]
                const __m128i T3_23A = _mm_srai_epi32(T2_23A, nShift);  // [37 27 17 07]
                const __m128i T3_23B = _mm_srai_epi32(T2_23B, nShift);  // [77 67 57 47]
                const __m128i T3_24A = _mm_srai_epi32(T2_24A, nShift);  // [30 20 10 00] x8
                const __m128i T3_24B = _mm_srai_epi32(T2_24B, nShift);  // [70 60 50 40]
                const __m128i T3_25A = _mm_srai_epi32(T2_25A, nShift);  // [31 21 11 01] x9
                const __m128i T3_25B = _mm_srai_epi32(T2_25B, nShift);  // [71 61 51 41]
                const __m128i T3_26A = _mm_srai_epi32(T2_26A, nShift);  // [32 22 12 02] xA
                const __m128i T3_26B = _mm_srai_epi32(T2_26B, nShift);  // [72 62 52 42]
                const __m128i T3_27A = _mm_srai_epi32(T2_27A, nShift);  // [33 23 13 03] xB
                const __m128i T3_27B = _mm_srai_epi32(T2_27B, nShift);  // [73 63 53 43]
                const __m128i T3_28A = _mm_srai_epi32(T2_28A, nShift);  // [33 24 14 04] xC
                const __m128i T3_28B = _mm_srai_epi32(T2_28B, nShift);  // [74 64 54 44]
                const __m128i T3_29A = _mm_srai_epi32(T2_29A, nShift);  // [35 25 15 05] xD
                const __m128i T3_29B = _mm_srai_epi32(T2_29B, nShift);  // [75 65 55 45]
                const __m128i T3_30A = _mm_srai_epi32(T2_30A, nShift);  // [36 26 16 06] xE
                const __m128i T3_30B = _mm_srai_epi32(T2_30B, nShift);  // [76 66 56 46]
                const __m128i T3_31A = _mm_srai_epi32(T2_31A, nShift);  // [37 27 17 07] xF
                const __m128i T3_31B = _mm_srai_epi32(T2_31B, nShift);  // [77 67 57 47]

                res00[part] = _mm_packs_epi32(T3_00A, T3_00B);          // [70 60 50 40 30 20 10 00]
                res01[part] = _mm_packs_epi32(T3_01A, T3_01B);          // [71 61 51 41 31 21 11 01]
                res02[part] = _mm_packs_epi32(T3_02A, T3_02B);          // [72 62 52 42 32 22 12 02]
                res03[part] = _mm_packs_epi32(T3_03A, T3_03B);          // [73 63 53 43 33 23 13 03]
                res04[part] = _mm_packs_epi32(T3_04A, T3_04B);          // [74 64 54 44 34 24 14 04]
                res05[part] = _mm_packs_epi32(T3_05A, T3_05B);          // [75 65 55 45 35 25 15 05]
                res06[part] = _mm_packs_epi32(T3_06A, T3_06B);          // [76 66 56 46 36 26 16 06]
                res07[part] = _mm_packs_epi32(T3_07A, T3_07B);          // [77 67 57 47 37 27 17 07]
                res08[part] = _mm_packs_epi32(T3_08A, T3_08B);          // [A0 ... 80]
                res09[part] = _mm_packs_epi32(T3_09A, T3_09B);          // [A1 ... 81]
                res10[part] = _mm_packs_epi32(T3_10A, T3_10B);          // [A2 ... 82]
                res11[part] = _mm_packs_epi32(T3_11A, T3_11B);          // [A3 ... 83]
                res12[part] = _mm_packs_epi32(T3_12A, T3_12B);          // [A4 ... 84]
                res13[part] = _mm_packs_epi32(T3_13A, T3_13B);          // [A5 ... 85]
                res14[part] = _mm_packs_epi32(T3_14A, T3_14B);          // [A6 ... 86]
                res15[part] = _mm_packs_epi32(T3_15A, T3_15B);          // [A7 ... 87]
                res16[part] = _mm_packs_epi32(T3_16A, T3_16B);
                res17[part] = _mm_packs_epi32(T3_17A, T3_17B);
                res18[part] = _mm_packs_epi32(T3_18A, T3_18B);
                res19[part] = _mm_packs_epi32(T3_19A, T3_19B);
                res20[part] = _mm_packs_epi32(T3_20A, T3_20B);
                res21[part] = _mm_packs_epi32(T3_21A, T3_21B);
                res22[part] = _mm_packs_epi32(T3_22A, T3_22B);
                res23[part] = _mm_packs_epi32(T3_23A, T3_23B);
                res24[part] = _mm_packs_epi32(T3_24A, T3_24B);
                res25[part] = _mm_packs_epi32(T3_25A, T3_25B);
                res26[part] = _mm_packs_epi32(T3_26A, T3_26B);
                res27[part] = _mm_packs_epi32(T3_27A, T3_27B);
                res28[part] = _mm_packs_epi32(T3_28A, T3_28B);
                res29[part] = _mm_packs_epi32(T3_29A, T3_29B);
                res30[part] = _mm_packs_epi32(T3_30A, T3_30B);
                res31[part] = _mm_packs_epi32(T3_31A, T3_31B);
            }
        }

        //transpose matrix 8x8 16bit.
        {
            __m128i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
            __m128i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;
#define TRANSPOSE_8x8_16BIT(I0, I1, I2, I3, I4, I5, I6, I7, O0, O1, O2, O3, O4, O5, O6, O7) \
    tr0_0 = _mm_unpacklo_epi16(I0, I1); \
    tr0_1 = _mm_unpacklo_epi16(I2, I3); \
    tr0_2 = _mm_unpackhi_epi16(I0, I1); \
    tr0_3 = _mm_unpackhi_epi16(I2, I3); \
    tr0_4 = _mm_unpacklo_epi16(I4, I5); \
    tr0_5 = _mm_unpacklo_epi16(I6, I7); \
    tr0_6 = _mm_unpackhi_epi16(I4, I5); \
    tr0_7 = _mm_unpackhi_epi16(I6, I7); \
    tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1); \
    tr1_1 = _mm_unpacklo_epi32(tr0_2, tr0_3); \
    tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1); \
    tr1_3 = _mm_unpackhi_epi32(tr0_2, tr0_3); \
    tr1_4 = _mm_unpacklo_epi32(tr0_4, tr0_5); \
    tr1_5 = _mm_unpacklo_epi32(tr0_6, tr0_7); \
    tr1_6 = _mm_unpackhi_epi32(tr0_4, tr0_5); \
    tr1_7 = _mm_unpackhi_epi32(tr0_6, tr0_7); \
    O0 = _mm_unpacklo_epi64(tr1_0, tr1_4); \
    O1 = _mm_unpackhi_epi64(tr1_0, tr1_4); \
    O2 = _mm_unpacklo_epi64(tr1_2, tr1_6); \
    O3 = _mm_unpackhi_epi64(tr1_2, tr1_6); \
    O4 = _mm_unpacklo_epi64(tr1_1, tr1_5); \
    O5 = _mm_unpackhi_epi64(tr1_1, tr1_5); \
    O6 = _mm_unpacklo_epi64(tr1_3, tr1_7); \
    O7 = _mm_unpackhi_epi64(tr1_3, tr1_7);

            TRANSPOSE_8x8_16BIT(res00[0], res01[0], res02[0], res03[0], res04[0], res05[0], res06[0], res07[0], in00[0], in01[0], in02[0], in03[0], in04[0], in05[0], in06[0], in07[0])
            TRANSPOSE_8x8_16BIT(res00[1], res01[1], res02[1], res03[1], res04[1], res05[1], res06[1], res07[1], in08[0], in09[0], in10[0], in11[0], in12[0], in13[0], in14[0], in15[0])
            TRANSPOSE_8x8_16BIT(res00[2], res01[2], res02[2], res03[2], res04[2], res05[2], res06[2], res07[2], in16[0], in17[0], in18[0], in19[0], in20[0], in21[0], in22[0], in23[0])
            TRANSPOSE_8x8_16BIT(res00[3], res01[3], res02[3], res03[3], res04[3], res05[3], res06[3], res07[3], in24[0], in25[0], in26[0], in27[0], in28[0], in29[0], in30[0], in31[0])

            TRANSPOSE_8x8_16BIT(res08[0], res09[0], res10[0], res11[0], res12[0], res13[0], res14[0], res15[0], in00[1], in01[1], in02[1], in03[1], in04[1], in05[1], in06[1], in07[1])
            TRANSPOSE_8x8_16BIT(res08[1], res09[1], res10[1], res11[1], res12[1], res13[1], res14[1], res15[1], in08[1], in09[1], in10[1], in11[1], in12[1], in13[1], in14[1], in15[1])
            TRANSPOSE_8x8_16BIT(res08[2], res09[2], res10[2], res11[2], res12[2], res13[2], res14[2], res15[2], in16[1], in17[1], in18[1], in19[1], in20[1], in21[1], in22[1], in23[1])
            TRANSPOSE_8x8_16BIT(res08[3], res09[3], res10[3], res11[3], res12[3], res13[3], res14[3], res15[3], in24[1], in25[1], in26[1], in27[1], in28[1], in29[1], in30[1], in31[1])

            TRANSPOSE_8x8_16BIT(res16[0], res17[0], res18[0], res19[0], res20[0], res21[0], res22[0], res23[0], in00[2], in01[2], in02[2], in03[2], in04[2], in05[2], in06[2], in07[2])
            TRANSPOSE_8x8_16BIT(res16[1], res17[1], res18[1], res19[1], res20[1], res21[1], res22[1], res23[1], in08[2], in09[2], in10[2], in11[2], in12[2], in13[2], in14[2], in15[2])
            TRANSPOSE_8x8_16BIT(res16[2], res17[2], res18[2], res19[2], res20[2], res21[2], res22[2], res23[2], in16[2], in17[2], in18[2], in19[2], in20[2], in21[2], in22[2], in23[2])
            TRANSPOSE_8x8_16BIT(res16[3], res17[3], res18[3], res19[3], res20[3], res21[3], res22[3], res23[3], in24[2], in25[2], in26[2], in27[2], in28[2], in29[2], in30[2], in31[2])

            TRANSPOSE_8x8_16BIT(res24[0], res25[0], res26[0], res27[0], res28[0], res29[0], res30[0], res31[0], in00[3], in01[3], in02[3], in03[3], in04[3], in05[3], in06[3], in07[3])
            TRANSPOSE_8x8_16BIT(res24[1], res25[1], res26[1], res27[1], res28[1], res29[1], res30[1], res31[1], in08[3], in09[3], in10[3], in11[3], in12[3], in13[3], in14[3], in15[3])
            TRANSPOSE_8x8_16BIT(res24[2], res25[2], res26[2], res27[2], res28[2], res29[2], res30[2], res31[2], in16[3], in17[3], in18[3], in19[3], in20[3], in21[3], in22[3], in23[3])
            TRANSPOSE_8x8_16BIT(res24[3], res25[3], res26[3], res27[3], res28[3], res29[3], res30[3], res31[3], in24[3], in25[3], in26[3], in27[3], in28[3], in29[3], in30[3], in31[3])
#undef TRANSPOSE_8x8_16BIT
        }
    }

    //clip
    {
        __m128i max_val = _mm_set1_epi16((1 << (clip_depth2 - 1)) - 1);
        __m128i min_val = _mm_set1_epi16(-(1 << (clip_depth2 - 1)));
        int k;

        for (k = 0; k < 4; k++) {
            in00[k] = _mm_max_epi16(_mm_min_epi16(in00[k], max_val), min_val);
            in01[k] = _mm_max_epi16(_mm_min_epi16(in01[k], max_val), min_val);
            in02[k] = _mm_max_epi16(_mm_min_epi16(in02[k], max_val), min_val);
            in03[k] = _mm_max_epi16(_mm_min_epi16(in03[k], max_val), min_val);
            in04[k] = _mm_max_epi16(_mm_min_epi16(in04[k], max_val), min_val);
            in05[k] = _mm_max_epi16(_mm_min_epi16(in05[k], max_val), min_val);
            in06[k] = _mm_max_epi16(_mm_min_epi16(in06[k], max_val), min_val);
            in07[k] = _mm_max_epi16(_mm_min_epi16(in07[k], max_val), min_val);
            in08[k] = _mm_max_epi16(_mm_min_epi16(in08[k], max_val), min_val);
            in09[k] = _mm_max_epi16(_mm_min_epi16(in09[k], max_val), min_val);
            in10[k] = _mm_max_epi16(_mm_min_epi16(in10[k], max_val), min_val);
            in11[k] = _mm_max_epi16(_mm_min_epi16(in11[k], max_val), min_val);
            in12[k] = _mm_max_epi16(_mm_min_epi16(in12[k], max_val), min_val);
            in13[k] = _mm_max_epi16(_mm_min_epi16(in13[k], max_val), min_val);
            in14[k] = _mm_max_epi16(_mm_min_epi16(in14[k], max_val), min_val);
            in15[k] = _mm_max_epi16(_mm_min_epi16(in15[k], max_val), min_val);
            in16[k] = _mm_max_epi16(_mm_min_epi16(in16[k], max_val), min_val);
            in17[k] = _mm_max_epi16(_mm_min_epi16(in17[k], max_val), min_val);
            in18[k] = _mm_max_epi16(_mm_min_epi16(in18[k], max_val), min_val);
            in19[k] = _mm_max_epi16(_mm_min_epi16(in19[k], max_val), min_val);
            in20[k] = _mm_max_epi16(_mm_min_epi16(in20[k], max_val), min_val);
            in21[k] = _mm_max_epi16(_mm_min_epi16(in21[k], max_val), min_val);
            in22[k] = _mm_max_epi16(_mm_min_epi16(in22[k], max_val), min_val);
            in23[k] = _mm_max_epi16(_mm_min_epi16(in23[k], max_val), min_val);
            in24[k] = _mm_max_epi16(_mm_min_epi16(in24[k], max_val), min_val);
            in25[k] = _mm_max_epi16(_mm_min_epi16(in25[k], max_val), min_val);
            in26[k] = _mm_max_epi16(_mm_min_epi16(in26[k], max_val), min_val);
            in27[k] = _mm_max_epi16(_mm_min_epi16(in27[k], max_val), min_val);
            in28[k] = _mm_max_epi16(_mm_min_epi16(in28[k], max_val), min_val);
            in29[k] = _mm_max_epi16(_mm_min_epi16(in29[k], max_val), min_val);
            in30[k] = _mm_max_epi16(_mm_min_epi16(in30[k], max_val), min_val);
            in31[k] = _mm_max_epi16(_mm_min_epi16(in31[k], max_val), min_val);
        }
    }

    // Add
    for (i = 0; i < 2; i++) {
#define STORE_LINE(L0, L1, L2, L3, L4, L5, L6, L7, H0, H1, H2, H3, H4, H5, H6, H7, offsetV, offsetH) \
    _mm_storeu_si128((__m128i*)(dst + (0 + (offsetV)) * i_dst + (offsetH)+0), L0); \
    _mm_storeu_si128((__m128i*)(dst + (0 + (offsetV)) * i_dst + (offsetH)+8), H0); \
    _mm_storeu_si128((__m128i*)(dst + (1 + (offsetV)) * i_dst + (offsetH)+0), L1); \
    _mm_storeu_si128((__m128i*)(dst + (1 + (offsetV)) * i_dst + (offsetH)+8), H1); \
    _mm_storeu_si128((__m128i*)(dst + (2 + (offsetV)) * i_dst + (offsetH)+0), L2); \
    _mm_storeu_si128((__m128i*)(dst + (2 + (offsetV)) * i_dst + (offsetH)+8), H2); \
    _mm_storeu_si128((__m128i*)(dst + (3 + (offsetV)) * i_dst + (offsetH)+0), L3); \
    _mm_storeu_si128((__m128i*)(dst + (3 + (offsetV)) * i_dst + (offsetH)+8), H3); \
    _mm_storeu_si128((__m128i*)(dst + (4 + (offsetV)) * i_dst + (offsetH)+0), L4); \
    _mm_storeu_si128((__m128i*)(dst + (4 + (offsetV)) * i_dst + (offsetH)+8), H4); \
    _mm_storeu_si128((__m128i*)(dst + (5 + (offsetV)) * i_dst + (offsetH)+0), L5); \
    _mm_storeu_si128((__m128i*)(dst + (5 + (offsetV)) * i_dst + (offsetH)+8), H5); \
    _mm_storeu_si128((__m128i*)(dst + (6 + (offsetV)) * i_dst + (offsetH)+0), L6); \
    _mm_storeu_si128((__m128i*)(dst + (6 + (offsetV)) * i_dst + (offsetH)+8), H6); \
    _mm_storeu_si128((__m128i*)(dst + (7 + (offsetV)) * i_dst + (offsetH)+0), L7); \
    _mm_storeu_si128((__m128i*)(dst + (7 + (offsetV)) * i_dst + (offsetH)+8), H7);

        const int k = i * 2;
        STORE_LINE(in00[k], in01[k], in02[k], in03[k], in04[k], in05[k], in06[k], in07[k], in00[k + 1], in01[k + 1], in02[k + 1], in03[k + 1], in04[k + 1], in05[k + 1], in06[k + 1], in07[k + 1], 0,  i * 16)
        STORE_LINE(in08[k], in09[k], in10[k], in11[k], in12[k], in13[k], in14[k], in15[k], in08[k + 1], in09[k + 1], in10[k + 1], in11[k + 1], in12[k + 1], in13[k + 1], in14[k + 1], in15[k + 1], 8,  i * 16)
        STORE_LINE(in16[k], in17[k], in18[k], in19[k], in20[k], in21[k], in22[k], in23[k], in16[k + 1], in17[k + 1], in18[k + 1], in19[k + 1], in20[k + 1], in21[k + 1], in22[k + 1], in23[k + 1], 16, i * 16)
        STORE_LINE(in24[k], in25[k], in26[k], in27[k], in28[k], in29[k], in30[k], in31[k], in24[k + 1], in25[k + 1], in26[k + 1], in27[k + 1], in28[k + 1], in29[k + 1], in30[k + 1], in31[k + 1], 24, i * 16)
#undef STORE_LINE
    }
}


/* ---------------------------------------------------------------------------
 */
void idct_c_32x8_sse128(const coeff_t *src, coeff_t *dst, int i_dst)
{
    __m128i m128iS0[4], m128iS1[4], m128iS2[4], m128iS3[4], m128iS4[4], m128iS5[4], m128iS6[4], m128iS7[4];
    __m128i m128iAdd, m128Tmp0, m128Tmp1, m128Tmp2, m128Tmp3;
    __m128i E0h, E1h, E2h, E3h, E0l, E1l, E2l, E3l;
    __m128i O0h, O1h, O2h, O3h, O0l, O1l, O2l, O3l;
    __m128i EE0l, EE1l, E00l, E01l, EE0h, EE1h, E00h, E01h;
    //int shift1 = 5;
    int shift2 = 20 - g_bit_depth - (i_dst & 0x01);
    //int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = g_bit_depth + 1 + (i_dst & 0x01);
    int i, pass;

    i_dst &= 0xFE;    /* remember to remove the flag bit */
    m128iAdd = _mm_set1_epi32(16);      // add1

    for (pass = 0; pass < 4; pass++) {
        m128iS1[pass] = _mm_load_si128((__m128i*)&src[pass * 8 + 1 * 32]);
        m128iS3[pass] = _mm_load_si128((__m128i*)&src[pass * 8 + 3 * 32]);

        m128Tmp0 = _mm_unpacklo_epi16(m128iS1[pass], m128iS3[pass]);
        E1l      = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8[0])));
        m128Tmp1 = _mm_unpackhi_epi16(m128iS1[pass], m128iS3[pass]);
        E1h      = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8[0])));

        m128iS5[pass] = _mm_load_si128((__m128i*)&src[pass * 8 + 5 * 32]);
        m128iS7[pass] = _mm_load_si128((__m128i*)&src[pass * 8 + 7 * 32]);

        m128Tmp2 = _mm_unpacklo_epi16(m128iS5[pass], m128iS7[pass]);
        E2l      = _mm_madd_epi16(m128Tmp2, _mm_load_si128((__m128i*)(tab_idct_8x8[1])));
        m128Tmp3 = _mm_unpackhi_epi16(m128iS5[pass], m128iS7[pass]);
        E2h = _mm_madd_epi16(m128Tmp3, _mm_load_si128((__m128i*)(tab_idct_8x8[1])));
        O0l = _mm_add_epi32(E1l, E2l);
        O0h = _mm_add_epi32(E1h, E2h);

        E1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8[2])));
        E1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8[2])));
        E2l = _mm_madd_epi16(m128Tmp2, _mm_load_si128((__m128i*)(tab_idct_8x8[3])));
        E2h = _mm_madd_epi16(m128Tmp3, _mm_load_si128((__m128i*)(tab_idct_8x8[3])));

        O1l = _mm_add_epi32(E1l, E2l);
        O1h = _mm_add_epi32(E1h, E2h);

        E1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8[4])));
        E1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8[4])));
        E2l = _mm_madd_epi16(m128Tmp2, _mm_load_si128((__m128i*)(tab_idct_8x8[5])));
        E2h = _mm_madd_epi16(m128Tmp3, _mm_load_si128((__m128i*)(tab_idct_8x8[5])));
        O2l = _mm_add_epi32(E1l, E2l);
        O2h = _mm_add_epi32(E1h, E2h);

        E1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8[6])));
        E1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8[6])));
        E2l = _mm_madd_epi16(m128Tmp2, _mm_load_si128((__m128i*)(tab_idct_8x8[7])));
        E2h = _mm_madd_epi16(m128Tmp3, _mm_load_si128((__m128i*)(tab_idct_8x8[7])));
        O3h = _mm_add_epi32(E1h, E2h);
        O3l = _mm_add_epi32(E1l, E2l);

        /*    -------     */

        m128iS0[pass] = _mm_load_si128((__m128i*)&src[pass * 8 + 0 * 32]);
        m128iS4[pass] = _mm_load_si128((__m128i*)&src[pass * 8 + 4 * 32]);

        m128Tmp0 = _mm_unpacklo_epi16(m128iS0[pass], m128iS4[pass]);
        EE0l     = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8[8])));
        m128Tmp1 = _mm_unpackhi_epi16(m128iS0[pass], m128iS4[pass]);
        EE0h     = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8[8])));

        EE1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8[9])));
        EE1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8[9])));

        /*    -------     */

        m128iS2[pass] = _mm_load_si128((__m128i*)&src[pass * 8 + 2 * 32]);
        m128iS6[pass] = _mm_load_si128((__m128i*)&src[pass * 8 + 6 * 32]);

        m128Tmp0 = _mm_unpacklo_epi16(m128iS2[pass], m128iS6[pass]);
        E00l     = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8[10])));
        m128Tmp1 = _mm_unpackhi_epi16(m128iS2[pass], m128iS6[pass]);
        E00h     = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8[10])));
        E01l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8[11])));
        E01h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8[11])));
        E0l = _mm_add_epi32(EE0l, E00l);
        E0l = _mm_add_epi32(E0l,  m128iAdd);
        E0h = _mm_add_epi32(EE0h, E00h);
        E0h = _mm_add_epi32(E0h,  m128iAdd);
        E3l = _mm_sub_epi32(EE0l, E00l);
        E3l = _mm_add_epi32(E3l,  m128iAdd);
        E3h = _mm_sub_epi32(EE0h, E00h);
        E3h = _mm_add_epi32(E3h,  m128iAdd);

        E1l = _mm_add_epi32(EE1l, E01l);
        E1l = _mm_add_epi32(E1l,  m128iAdd);
        E1h = _mm_add_epi32(EE1h, E01h);
        E1h = _mm_add_epi32(E1h,  m128iAdd);
        E2l = _mm_sub_epi32(EE1l, E01l);
        E2l = _mm_add_epi32(E2l,  m128iAdd);
        E2h = _mm_sub_epi32(EE1h, E01h);
        E2h = _mm_add_epi32(E2h,  m128iAdd);

        m128iS0[pass] = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E0l, O0l), 5), _mm_srai_epi32(_mm_add_epi32(E0h, O0h), 5));    // 首次反变换移位数
        m128iS7[pass] = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E0l, O0l), 5), _mm_srai_epi32(_mm_sub_epi32(E0h, O0h), 5));
        m128iS1[pass] = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E1l, O1l), 5), _mm_srai_epi32(_mm_add_epi32(E1h, O1h), 5));
        m128iS6[pass] = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E1l, O1l), 5), _mm_srai_epi32(_mm_sub_epi32(E1h, O1h), 5));
        m128iS2[pass] = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E2l, O2l), 5), _mm_srai_epi32(_mm_add_epi32(E2h, O2h), 5));
        m128iS5[pass] = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E2l, O2l), 5), _mm_srai_epi32(_mm_sub_epi32(E2h, O2h), 5));
        m128iS3[pass] = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E3l, O3l), 5), _mm_srai_epi32(_mm_add_epi32(E3h, O3h), 5));
        m128iS4[pass] = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E3l, O3l), 5), _mm_srai_epi32(_mm_sub_epi32(E3h, O3h), 5));

        /*  Inverts matrix   */
        E0l = _mm_unpacklo_epi16(m128iS0[pass], m128iS4[pass]);
        E1l = _mm_unpacklo_epi16(m128iS1[pass], m128iS5[pass]);
        E2l = _mm_unpacklo_epi16(m128iS2[pass], m128iS6[pass]);
        E3l = _mm_unpacklo_epi16(m128iS3[pass], m128iS7[pass]);
        O0l = _mm_unpackhi_epi16(m128iS0[pass], m128iS4[pass]);
        O1l = _mm_unpackhi_epi16(m128iS1[pass], m128iS5[pass]);
        O2l = _mm_unpackhi_epi16(m128iS2[pass], m128iS6[pass]);
        O3l = _mm_unpackhi_epi16(m128iS3[pass], m128iS7[pass]);
        m128Tmp0      = _mm_unpacklo_epi16(E0l, E2l);
        m128Tmp1      = _mm_unpacklo_epi16(E1l, E3l);
        m128iS0[pass] = _mm_unpacklo_epi16(m128Tmp0, m128Tmp1);
        m128iS1[pass] = _mm_unpackhi_epi16(m128Tmp0, m128Tmp1);
        m128Tmp2      = _mm_unpackhi_epi16(E0l, E2l);
        m128Tmp3      = _mm_unpackhi_epi16(E1l, E3l);
        m128iS2[pass] = _mm_unpacklo_epi16(m128Tmp2, m128Tmp3);
        m128iS3[pass] = _mm_unpackhi_epi16(m128Tmp2, m128Tmp3);
        m128Tmp0      = _mm_unpacklo_epi16(O0l, O2l);
        m128Tmp1      = _mm_unpacklo_epi16(O1l, O3l);
        m128iS4[pass] = _mm_unpacklo_epi16(m128Tmp0, m128Tmp1);
        m128iS5[pass] = _mm_unpackhi_epi16(m128Tmp0, m128Tmp1);
        m128Tmp2      = _mm_unpackhi_epi16(O0l, O2l);
        m128Tmp3      = _mm_unpackhi_epi16(O1l, O3l);
        m128iS6[pass] = _mm_unpacklo_epi16(m128Tmp2, m128Tmp3);
        m128iS7[pass] = _mm_unpackhi_epi16(m128Tmp2, m128Tmp3);
    }

    {
        const __m128i c16_p45_p45 = _mm_set1_epi32(0x002D002D);
        const __m128i c16_p43_p44 = _mm_set1_epi32(0x002B002C);
        const __m128i c16_p39_p41 = _mm_set1_epi32(0x00270029);
        const __m128i c16_p34_p36 = _mm_set1_epi32(0x00220024);
        const __m128i c16_p27_p30 = _mm_set1_epi32(0x001B001E);
        const __m128i c16_p19_p23 = _mm_set1_epi32(0x00130017);
        const __m128i c16_p11_p15 = _mm_set1_epi32(0x000B000F);
        const __m128i c16_p02_p07 = _mm_set1_epi32(0x00020007);
        const __m128i c16_p41_p45 = _mm_set1_epi32(0x0029002D);
        const __m128i c16_p23_p34 = _mm_set1_epi32(0x00170022);
        const __m128i c16_n02_p11 = _mm_set1_epi32(0xFFFE000B);
        const __m128i c16_n27_n15 = _mm_set1_epi32(0xFFE5FFF1);
        const __m128i c16_n43_n36 = _mm_set1_epi32(0xFFD5FFDC);
        const __m128i c16_n44_n45 = _mm_set1_epi32(0xFFD4FFD3);
        const __m128i c16_n30_n39 = _mm_set1_epi32(0xFFE2FFD9);
        const __m128i c16_n07_n19 = _mm_set1_epi32(0xFFF9FFED);
        const __m128i c16_p34_p44 = _mm_set1_epi32(0x0022002C);
        const __m128i c16_n07_p15 = _mm_set1_epi32(0xFFF9000F);
        const __m128i c16_n41_n27 = _mm_set1_epi32(0xFFD7FFE5);
        const __m128i c16_n39_n45 = _mm_set1_epi32(0xFFD9FFD3);
        const __m128i c16_n02_n23 = _mm_set1_epi32(0xFFFEFFE9);
        const __m128i c16_p36_p19 = _mm_set1_epi32(0x00240013);
        const __m128i c16_p43_p45 = _mm_set1_epi32(0x002B002D);
        const __m128i c16_p11_p30 = _mm_set1_epi32(0x000B001E);
        const __m128i c16_p23_p43 = _mm_set1_epi32(0x0017002B);
        const __m128i c16_n34_n07 = _mm_set1_epi32(0xFFDEFFF9);
        const __m128i c16_n36_n45 = _mm_set1_epi32(0xFFDCFFD3);
        const __m128i c16_p19_n11 = _mm_set1_epi32(0x0013FFF5);
        const __m128i c16_p44_p41 = _mm_set1_epi32(0x002C0029);
        const __m128i c16_n02_p27 = _mm_set1_epi32(0xFFFE001B);
        const __m128i c16_n45_n30 = _mm_set1_epi32(0xFFD3FFE2);
        const __m128i c16_n15_n39 = _mm_set1_epi32(0xFFF1FFD9);
        const __m128i c16_p11_p41 = _mm_set1_epi32(0x000B0029);
        const __m128i c16_n45_n27 = _mm_set1_epi32(0xFFD3FFE5);
        const __m128i c16_p07_n30 = _mm_set1_epi32(0x0007FFE2);
        const __m128i c16_p43_p39 = _mm_set1_epi32(0x002B0027);
        const __m128i c16_n23_p15 = _mm_set1_epi32(0xFFE9000F);
        const __m128i c16_n34_n45 = _mm_set1_epi32(0xFFDEFFD3);
        const __m128i c16_p36_p02 = _mm_set1_epi32(0x00240002);
        const __m128i c16_p19_p44 = _mm_set1_epi32(0x0013002C);
        const __m128i c16_n02_p39 = _mm_set1_epi32(0xFFFE0027);
        const __m128i c16_n36_n41 = _mm_set1_epi32(0xFFDCFFD7);
        const __m128i c16_p43_p07 = _mm_set1_epi32(0x002B0007);
        const __m128i c16_n11_p34 = _mm_set1_epi32(0xFFF50022);
        const __m128i c16_n30_n44 = _mm_set1_epi32(0xFFE2FFD4);
        const __m128i c16_p45_p15 = _mm_set1_epi32(0x002D000F);
        const __m128i c16_n19_p27 = _mm_set1_epi32(0xFFED001B);
        const __m128i c16_n23_n45 = _mm_set1_epi32(0xFFE9FFD3);
        const __m128i c16_n15_p36 = _mm_set1_epi32(0xFFF10024);
        const __m128i c16_n11_n45 = _mm_set1_epi32(0xFFF5FFD3);
        const __m128i c16_p34_p39 = _mm_set1_epi32(0x00220027);
        const __m128i c16_n45_n19 = _mm_set1_epi32(0xFFD3FFED);
        const __m128i c16_p41_n07 = _mm_set1_epi32(0x0029FFF9);
        const __m128i c16_n23_p30 = _mm_set1_epi32(0xFFE9001E);
        const __m128i c16_n02_n44 = _mm_set1_epi32(0xFFFEFFD4);
        const __m128i c16_p27_p43 = _mm_set1_epi32(0x001B002B);
        const __m128i c16_n27_p34 = _mm_set1_epi32(0xFFE50022);
        const __m128i c16_p19_n39 = _mm_set1_epi32(0x0013FFD9);
        const __m128i c16_n11_p43 = _mm_set1_epi32(0xFFF5002B);
        const __m128i c16_p02_n45 = _mm_set1_epi32(0x0002FFD3);
        const __m128i c16_p07_p45 = _mm_set1_epi32(0x0007002D);
        const __m128i c16_n15_n44 = _mm_set1_epi32(0xFFF1FFD4);
        const __m128i c16_p23_p41 = _mm_set1_epi32(0x00170029);
        const __m128i c16_n30_n36 = _mm_set1_epi32(0xFFE2FFDC);
        const __m128i c16_n36_p30 = _mm_set1_epi32(0xFFDC001E);
        const __m128i c16_p41_n23 = _mm_set1_epi32(0x0029FFE9);
        const __m128i c16_n44_p15 = _mm_set1_epi32(0xFFD4000F);
        const __m128i c16_p45_n07 = _mm_set1_epi32(0x002DFFF9);
        const __m128i c16_n45_n02 = _mm_set1_epi32(0xFFD3FFFE);
        const __m128i c16_p43_p11 = _mm_set1_epi32(0x002B000B);
        const __m128i c16_n39_n19 = _mm_set1_epi32(0xFFD9FFED);
        const __m128i c16_p34_p27 = _mm_set1_epi32(0x0022001B);
        const __m128i c16_n43_p27 = _mm_set1_epi32(0xFFD5001B);
        const __m128i c16_p44_n02 = _mm_set1_epi32(0x002CFFFE);
        const __m128i c16_n30_n23 = _mm_set1_epi32(0xFFE2FFE9);
        const __m128i c16_p07_p41 = _mm_set1_epi32(0x00070029);
        const __m128i c16_p19_n45 = _mm_set1_epi32(0x0013FFD3);
        const __m128i c16_n39_p34 = _mm_set1_epi32(0xFFD90022);
        const __m128i c16_p45_n11 = _mm_set1_epi32(0x002DFFF5);
        const __m128i c16_n36_n15 = _mm_set1_epi32(0xFFDCFFF1);
        const __m128i c16_n45_p23 = _mm_set1_epi32(0xFFD30017);
        const __m128i c16_p27_p19 = _mm_set1_epi32(0x001B0013);
        const __m128i c16_p15_n45 = _mm_set1_epi32(0x000FFFD3);
        const __m128i c16_n44_p30 = _mm_set1_epi32(0xFFD4001E);
        const __m128i c16_p34_p11 = _mm_set1_epi32(0x0022000B);
        const __m128i c16_p07_n43 = _mm_set1_epi32(0x0007FFD5);
        const __m128i c16_n41_p36 = _mm_set1_epi32(0xFFD70024);
        const __m128i c16_p39_p02 = _mm_set1_epi32(0x00270002);
        const __m128i c16_n44_p19 = _mm_set1_epi32(0xFFD40013);
        const __m128i c16_n02_p36 = _mm_set1_epi32(0xFFFE0024);
        const __m128i c16_p45_n34 = _mm_set1_epi32(0x002DFFDE);
        const __m128i c16_n15_n23 = _mm_set1_epi32(0xFFF1FFE9);
        const __m128i c16_n39_p43 = _mm_set1_epi32(0xFFD9002B);
        const __m128i c16_p30_p07 = _mm_set1_epi32(0x001E0007);
        const __m128i c16_p27_n45 = _mm_set1_epi32(0x001BFFD3);
        const __m128i c16_n41_p11 = _mm_set1_epi32(0xFFD7000B);
        const __m128i c16_n39_p15 = _mm_set1_epi32(0xFFD9000F);
        const __m128i c16_n30_p45 = _mm_set1_epi32(0xFFE2002D);
        const __m128i c16_p27_p02 = _mm_set1_epi32(0x001B0002);
        const __m128i c16_p41_n44 = _mm_set1_epi32(0x0029FFD4);
        const __m128i c16_n11_n19 = _mm_set1_epi32(0xFFF5FFED);
        const __m128i c16_n45_p36 = _mm_set1_epi32(0xFFD30024);
        const __m128i c16_n07_p34 = _mm_set1_epi32(0xFFF90022);
        const __m128i c16_p43_n23 = _mm_set1_epi32(0x002BFFE9);
        const __m128i c16_n30_p11 = _mm_set1_epi32(0xFFE2000B);
        const __m128i c16_n45_p43 = _mm_set1_epi32(0xFFD3002B);
        const __m128i c16_n19_p36 = _mm_set1_epi32(0xFFED0024);
        const __m128i c16_p23_n02 = _mm_set1_epi32(0x0017FFFE);
        const __m128i c16_p45_n39 = _mm_set1_epi32(0x002DFFD9);
        const __m128i c16_p27_n41 = _mm_set1_epi32(0x001BFFD7);
        const __m128i c16_n15_n07 = _mm_set1_epi32(0xFFF1FFF9);
        const __m128i c16_n44_p34 = _mm_set1_epi32(0xFFD40022);
        const __m128i c16_n19_p07 = _mm_set1_epi32(0xFFED0007);
        const __m128i c16_n39_p30 = _mm_set1_epi32(0xFFD9001E);
        const __m128i c16_n45_p44 = _mm_set1_epi32(0xFFD3002C);
        const __m128i c16_n36_p43 = _mm_set1_epi32(0xFFDC002B);
        const __m128i c16_n15_p27 = _mm_set1_epi32(0xFFF1001B);
        const __m128i c16_p11_p02 = _mm_set1_epi32(0x000B0002);
        const __m128i c16_p34_n23 = _mm_set1_epi32(0x0022FFE9);
        const __m128i c16_p45_n41 = _mm_set1_epi32(0x002DFFD7);
        const __m128i c16_n07_p02 = _mm_set1_epi32(0xFFF90002);
        const __m128i c16_n15_p11 = _mm_set1_epi32(0xFFF1000B);
        const __m128i c16_n23_p19 = _mm_set1_epi32(0xFFE90013);
        const __m128i c16_n30_p27 = _mm_set1_epi32(0xFFE2001B);
        const __m128i c16_n36_p34 = _mm_set1_epi32(0xFFDC0022);
        const __m128i c16_n41_p39 = _mm_set1_epi32(0xFFD70027);
        const __m128i c16_n44_p43 = _mm_set1_epi32(0xFFD4002B);
        const __m128i c16_n45_p45 = _mm_set1_epi32(0xFFD3002D);

        //  const __m128i c16_p43_p45 = _mm_set1_epi32(0x002B002D);
        const __m128i c16_p35_p40 = _mm_set1_epi32(0x00230028);
        const __m128i c16_p21_p29 = _mm_set1_epi32(0x0015001D);
        const __m128i c16_p04_p13 = _mm_set1_epi32(0x0004000D);
        const __m128i c16_p29_p43 = _mm_set1_epi32(0x001D002B);
        const __m128i c16_n21_p04 = _mm_set1_epi32(0xFFEB0004);
        const __m128i c16_n45_n40 = _mm_set1_epi32(0xFFD3FFD8);
        const __m128i c16_n13_n35 = _mm_set1_epi32(0xFFF3FFDD);
        const __m128i c16_p04_p40 = _mm_set1_epi32(0x00040028);
        const __m128i c16_n43_n35 = _mm_set1_epi32(0xFFD5FFDD);
        const __m128i c16_p29_n13 = _mm_set1_epi32(0x001DFFF3);
        const __m128i c16_p21_p45 = _mm_set1_epi32(0x0015002D);
        const __m128i c16_n21_p35 = _mm_set1_epi32(0xFFEB0023);
        const __m128i c16_p04_n43 = _mm_set1_epi32(0x0004FFD5);
        const __m128i c16_p13_p45 = _mm_set1_epi32(0x000D002D);
        const __m128i c16_n29_n40 = _mm_set1_epi32(0xFFE3FFD8);
        const __m128i c16_n40_p29 = _mm_set1_epi32(0xFFD8001D);
        const __m128i c16_p45_n13 = _mm_set1_epi32(0x002DFFF3);
        const __m128i c16_n43_n04 = _mm_set1_epi32(0xFFD5FFFC);
        const __m128i c16_p35_p21 = _mm_set1_epi32(0x00230015);
        const __m128i c16_n45_p21 = _mm_set1_epi32(0xFFD30015);
        const __m128i c16_p13_p29 = _mm_set1_epi32(0x000D001D);
        const __m128i c16_p35_n43 = _mm_set1_epi32(0x0023FFD5);
        const __m128i c16_n40_p04 = _mm_set1_epi32(0xFFD80004);
        const __m128i c16_n35_p13 = _mm_set1_epi32(0xFFDD000D);
        const __m128i c16_n40_p45 = _mm_set1_epi32(0xFFD8002D);
        const __m128i c16_p04_p21 = _mm_set1_epi32(0x00040015);
        const __m128i c16_p43_n29 = _mm_set1_epi32(0x002BFFE3);
        const __m128i c16_n13_p04 = _mm_set1_epi32(0xFFF30004);
        const __m128i c16_n29_p21 = _mm_set1_epi32(0xFFE30015);
        const __m128i c16_n40_p35 = _mm_set1_epi32(0xFFD80023);
        //  const __m128i c16_n45_p43 = _mm_set1_epi32(0xFFD3002B);

        const __m128i c16_p38_p44 = _mm_set1_epi32(0x0026002C);
        const __m128i c16_p09_p25 = _mm_set1_epi32(0x00090019);
        const __m128i c16_n09_p38 = _mm_set1_epi32(0xFFF70026);
        const __m128i c16_n25_n44 = _mm_set1_epi32(0xFFE7FFD4);

        const __m128i c16_n44_p25 = _mm_set1_epi32(0xFFD40019);
        const __m128i c16_p38_p09 = _mm_set1_epi32(0x00260009);
        const __m128i c16_n25_p09 = _mm_set1_epi32(0xFFE70009);
        const __m128i c16_n44_p38 = _mm_set1_epi32(0xFFD40026);

        const __m128i c16_p17_p42 = _mm_set1_epi32(0x0011002A);
        const __m128i c16_n42_p17 = _mm_set1_epi32(0xFFD60011);

        const __m128i c16_p32_p32 = _mm_set1_epi32(0x00200020);
        const __m128i c16_n32_p32 = _mm_set1_epi32(0xFFE00020);


        __m128i c32_rnd = _mm_set1_epi32(1 << (shift2 - 1));    // add2
        int nShift = shift2;

        // DCT1

        __m128i res00[4], res01[4], res02[4], res03[4], res04[4], res05[4], res06[4], res07[4], res08[4], res09[4], res10[4], res11[4], res12[4], res13[4], res14[4], res15[4];
        __m128i res16[4], res17[4], res18[4], res19[4], res20[4], res21[4], res22[4], res23[4], res24[4], res25[4], res26[4], res27[4], res28[4], res29[4], res30[4], res31[4];

        const __m128i T_00_00A = _mm_unpacklo_epi16(m128iS1[0], m128iS3[0]);    // [33 13 32 12 31 11 30 10]
        const __m128i T_00_00B = _mm_unpackhi_epi16(m128iS1[0], m128iS3[0]);    // [37 17 36 16 35 15 34 14]
        const __m128i T_00_01A = _mm_unpacklo_epi16(m128iS5[0], m128iS7[0]);    // [ ]
        const __m128i T_00_01B = _mm_unpackhi_epi16(m128iS5[0], m128iS7[0]);    // [ ]
        const __m128i T_00_02A = _mm_unpacklo_epi16(m128iS1[1], m128iS3[1]);    // [ ]
        const __m128i T_00_02B = _mm_unpackhi_epi16(m128iS1[1], m128iS3[1]);    // [ ]
        const __m128i T_00_03A = _mm_unpacklo_epi16(m128iS5[1], m128iS7[1]);    // [ ]
        const __m128i T_00_03B = _mm_unpackhi_epi16(m128iS5[1], m128iS7[1]);    // [ ]
        const __m128i T_00_04A = _mm_unpacklo_epi16(m128iS1[2], m128iS3[2]);    // [ ]
        const __m128i T_00_04B = _mm_unpackhi_epi16(m128iS1[2], m128iS3[2]);    // [ ]
        const __m128i T_00_05A = _mm_unpacklo_epi16(m128iS5[2], m128iS7[2]);    // [ ]
        const __m128i T_00_05B = _mm_unpackhi_epi16(m128iS5[2], m128iS7[2]);    // [ ]
        const __m128i T_00_06A = _mm_unpacklo_epi16(m128iS1[3], m128iS3[3]);    // [ ]
        const __m128i T_00_06B = _mm_unpackhi_epi16(m128iS1[3], m128iS3[3]);    // [ ]
        const __m128i T_00_07A = _mm_unpacklo_epi16(m128iS5[3], m128iS7[3]);    //
        const __m128i T_00_07B = _mm_unpackhi_epi16(m128iS5[3], m128iS7[3]);    // [ ]

        const __m128i T_00_08A = _mm_unpacklo_epi16(m128iS2[0], m128iS6[0]);    // [ ]
        const __m128i T_00_08B = _mm_unpackhi_epi16(m128iS2[0], m128iS6[0]);    // [ ]
        const __m128i T_00_09A = _mm_unpacklo_epi16(m128iS2[1], m128iS6[1]);    // [ ]
        const __m128i T_00_09B = _mm_unpackhi_epi16(m128iS2[1], m128iS6[1]);    // [ ]
        const __m128i T_00_10A = _mm_unpacklo_epi16(m128iS2[2], m128iS6[2]);    // [ ]
        const __m128i T_00_10B = _mm_unpackhi_epi16(m128iS2[2], m128iS6[2]);    // [ ]
        const __m128i T_00_11A = _mm_unpacklo_epi16(m128iS2[3], m128iS6[3]);    // [ ]
        const __m128i T_00_11B = _mm_unpackhi_epi16(m128iS2[3], m128iS6[3]);    // [ ]

        const __m128i T_00_12A = _mm_unpacklo_epi16(m128iS4[0], m128iS4[1]);    // [ ]
        const __m128i T_00_12B = _mm_unpackhi_epi16(m128iS4[0], m128iS4[1]);    // [ ]
        const __m128i T_00_13A = _mm_unpacklo_epi16(m128iS4[2], m128iS4[3]);    // [ ]
        const __m128i T_00_13B = _mm_unpackhi_epi16(m128iS4[2], m128iS4[3]);    // [ ]

        const __m128i T_00_14A = _mm_unpacklo_epi16(m128iS0[1], m128iS0[3]);    //
        const __m128i T_00_14B = _mm_unpackhi_epi16(m128iS0[1], m128iS0[3]);    // [ ]
        const __m128i T_00_15A = _mm_unpacklo_epi16(m128iS0[0], m128iS0[2]);    //
        const __m128i T_00_15B = _mm_unpackhi_epi16(m128iS0[0], m128iS0[2]);    // [ ]

        __m128i O00A, O01A, O02A, O03A, O04A, O05A, O06A, O07A, O08A, O09A, O10A, O11A, O12A, O13A, O14A, O15A;
        __m128i O00B, O01B, O02B, O03B, O04B, O05B, O06B, O07B, O08B, O09B, O10B, O11B, O12B, O13B, O14B, O15B;
        __m128i EO0A, EO1A, EO2A, EO3A, EO4A, EO5A, EO6A, EO7A;
        __m128i EO0B, EO1B, EO2B, EO3B, EO4B, EO5B, EO6B, EO7B;
        __m128i T00, T01, T02, T03;

#define COMPUTE_ROW(r0103, r0507, r0911, r1315, r1719, r2123, r2527, r2931, c0103, c0507, c0911, c1315, c1719, c2123, c2527, c2931, row) \
    T00 = _mm_add_epi32(_mm_madd_epi16(r0103, c0103), _mm_madd_epi16(r0507, c0507)); \
    T01 = _mm_add_epi32(_mm_madd_epi16(r0911, c0911), _mm_madd_epi16(r1315, c1315)); \
    T02 = _mm_add_epi32(_mm_madd_epi16(r1719, c1719), _mm_madd_epi16(r2123, c2123)); \
    T03 = _mm_add_epi32(_mm_madd_epi16(r2527, c2527), _mm_madd_epi16(r2931, c2931)); \
    row = _mm_add_epi32(_mm_add_epi32(T00, T01), _mm_add_epi32(T02, T03));

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

#undef COMPUTE_ROW

        {
#define COMPUTE_ROW(row0206, row1014, row1822, row2630, c0206, c1014, c1822, c2630, row) \
    T00 = _mm_add_epi32(_mm_madd_epi16(row0206, c0206), _mm_madd_epi16(row1014, c1014)); \
    T01 = _mm_add_epi32(_mm_madd_epi16(row1822, c1822), _mm_madd_epi16(row2630, c2630)); \
    row = _mm_add_epi32(T00, T01);

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
#undef COMPUTE_ROW
        }

        {
            const __m128i EEO0A = _mm_add_epi32(_mm_madd_epi16(T_00_12A, c16_p38_p44), _mm_madd_epi16(T_00_13A, c16_p09_p25));
            const __m128i EEO1A = _mm_add_epi32(_mm_madd_epi16(T_00_12A, c16_n09_p38), _mm_madd_epi16(T_00_13A, c16_n25_n44));
            const __m128i EEO2A = _mm_add_epi32(_mm_madd_epi16(T_00_12A, c16_n44_p25), _mm_madd_epi16(T_00_13A, c16_p38_p09));
            const __m128i EEO3A = _mm_add_epi32(_mm_madd_epi16(T_00_12A, c16_n25_p09), _mm_madd_epi16(T_00_13A, c16_n44_p38));
            const __m128i EEO0B = _mm_add_epi32(_mm_madd_epi16(T_00_12B, c16_p38_p44), _mm_madd_epi16(T_00_13B, c16_p09_p25));
            const __m128i EEO1B = _mm_add_epi32(_mm_madd_epi16(T_00_12B, c16_n09_p38), _mm_madd_epi16(T_00_13B, c16_n25_n44));
            const __m128i EEO2B = _mm_add_epi32(_mm_madd_epi16(T_00_12B, c16_n44_p25), _mm_madd_epi16(T_00_13B, c16_p38_p09));
            const __m128i EEO3B = _mm_add_epi32(_mm_madd_epi16(T_00_12B, c16_n25_p09), _mm_madd_epi16(T_00_13B, c16_n44_p38));

            const __m128i EEEO0A = _mm_madd_epi16(T_00_14A, c16_p17_p42);
            const __m128i EEEO0B = _mm_madd_epi16(T_00_14B, c16_p17_p42);
            const __m128i EEEO1A = _mm_madd_epi16(T_00_14A, c16_n42_p17);
            const __m128i EEEO1B = _mm_madd_epi16(T_00_14B, c16_n42_p17);

            const __m128i EEEE0A = _mm_madd_epi16(T_00_15A, c16_p32_p32);
            const __m128i EEEE0B = _mm_madd_epi16(T_00_15B, c16_p32_p32);
            const __m128i EEEE1A = _mm_madd_epi16(T_00_15A, c16_n32_p32);
            const __m128i EEEE1B = _mm_madd_epi16(T_00_15B, c16_n32_p32);

            const __m128i EEE0A = _mm_add_epi32(EEEE0A, EEEO0A);    // EEE0 = EEEE0 + EEEO0
            const __m128i EEE0B = _mm_add_epi32(EEEE0B, EEEO0B);
            const __m128i EEE1A = _mm_add_epi32(EEEE1A, EEEO1A);    // EEE1 = EEEE1 + EEEO1
            const __m128i EEE1B = _mm_add_epi32(EEEE1B, EEEO1B);
            const __m128i EEE3A = _mm_sub_epi32(EEEE0A, EEEO0A);    // EEE2 = EEEE0 - EEEO0
            const __m128i EEE3B = _mm_sub_epi32(EEEE0B, EEEO0B);
            const __m128i EEE2A = _mm_sub_epi32(EEEE1A, EEEO1A);    // EEE3 = EEEE1 - EEEO1
            const __m128i EEE2B = _mm_sub_epi32(EEEE1B, EEEO1B);

            const __m128i EE0A = _mm_add_epi32(EEE0A, EEO0A);       // EE0 = EEE0 + EEO0
            const __m128i EE0B = _mm_add_epi32(EEE0B, EEO0B);
            const __m128i EE1A = _mm_add_epi32(EEE1A, EEO1A);       // EE1 = EEE1 + EEO1
            const __m128i EE1B = _mm_add_epi32(EEE1B, EEO1B);
            const __m128i EE2A = _mm_add_epi32(EEE2A, EEO2A);       // EE2 = EEE0 + EEO0
            const __m128i EE2B = _mm_add_epi32(EEE2B, EEO2B);
            const __m128i EE3A = _mm_add_epi32(EEE3A, EEO3A);       // EE3 = EEE1 + EEO1
            const __m128i EE3B = _mm_add_epi32(EEE3B, EEO3B);
            const __m128i EE7A = _mm_sub_epi32(EEE0A, EEO0A);       // EE7 = EEE0 - EEO0
            const __m128i EE7B = _mm_sub_epi32(EEE0B, EEO0B);
            const __m128i EE6A = _mm_sub_epi32(EEE1A, EEO1A);       // EE6 = EEE1 - EEO1
            const __m128i EE6B = _mm_sub_epi32(EEE1B, EEO1B);
            const __m128i EE5A = _mm_sub_epi32(EEE2A, EEO2A);       // EE5 = EEE0 - EEO0
            const __m128i EE5B = _mm_sub_epi32(EEE2B, EEO2B);
            const __m128i EE4A = _mm_sub_epi32(EEE3A, EEO3A);       // EE4 = EEE1 - EEO1
            const __m128i EE4B = _mm_sub_epi32(EEE3B, EEO3B);

            const __m128i E0A = _mm_add_epi32(EE0A, EO0A);          // E0 = EE0 + EO0
            const __m128i E0B = _mm_add_epi32(EE0B, EO0B);
            const __m128i E1A = _mm_add_epi32(EE1A, EO1A);          // E1 = EE1 + EO1
            const __m128i E1B = _mm_add_epi32(EE1B, EO1B);
            const __m128i E2A = _mm_add_epi32(EE2A, EO2A);          // E2 = EE2 + EO2
            const __m128i E2B = _mm_add_epi32(EE2B, EO2B);
            const __m128i E3A = _mm_add_epi32(EE3A, EO3A);          // E3 = EE3 + EO3
            const __m128i E3B = _mm_add_epi32(EE3B, EO3B);
            const __m128i E4A = _mm_add_epi32(EE4A, EO4A);          // E4 =
            const __m128i E4B = _mm_add_epi32(EE4B, EO4B);
            const __m128i E5A = _mm_add_epi32(EE5A, EO5A);          // E5 =
            const __m128i E5B = _mm_add_epi32(EE5B, EO5B);
            const __m128i E6A = _mm_add_epi32(EE6A, EO6A);          // E6 =
            const __m128i E6B = _mm_add_epi32(EE6B, EO6B);
            const __m128i E7A = _mm_add_epi32(EE7A, EO7A);          // E7 =
            const __m128i E7B = _mm_add_epi32(EE7B, EO7B);
            const __m128i EFA = _mm_sub_epi32(EE0A, EO0A);          // EF = EE0 - EO0
            const __m128i EFB = _mm_sub_epi32(EE0B, EO0B);
            const __m128i EEA = _mm_sub_epi32(EE1A, EO1A);          // EE = EE1 - EO1
            const __m128i EEB = _mm_sub_epi32(EE1B, EO1B);
            const __m128i EDA = _mm_sub_epi32(EE2A, EO2A);          // ED = EE2 - EO2
            const __m128i EDB = _mm_sub_epi32(EE2B, EO2B);
            const __m128i ECA = _mm_sub_epi32(EE3A, EO3A);          // EC = EE3 - EO3
            const __m128i ECB = _mm_sub_epi32(EE3B, EO3B);
            const __m128i EBA = _mm_sub_epi32(EE4A, EO4A);          // EB =
            const __m128i EBB = _mm_sub_epi32(EE4B, EO4B);
            const __m128i EAA = _mm_sub_epi32(EE5A, EO5A);          // EA =
            const __m128i EAB = _mm_sub_epi32(EE5B, EO5B);
            const __m128i E9A = _mm_sub_epi32(EE6A, EO6A);          // E9 =
            const __m128i E9B = _mm_sub_epi32(EE6B, EO6B);
            const __m128i E8A = _mm_sub_epi32(EE7A, EO7A);          // E8 =
            const __m128i E8B = _mm_sub_epi32(EE7B, EO7B);

            const __m128i T10A = _mm_add_epi32(E0A, c32_rnd);       // E0 + rnd
            const __m128i T10B = _mm_add_epi32(E0B, c32_rnd);
            const __m128i T11A = _mm_add_epi32(E1A, c32_rnd);       // E1 + rnd
            const __m128i T11B = _mm_add_epi32(E1B, c32_rnd);
            const __m128i T12A = _mm_add_epi32(E2A, c32_rnd);       // E2 + rnd
            const __m128i T12B = _mm_add_epi32(E2B, c32_rnd);
            const __m128i T13A = _mm_add_epi32(E3A, c32_rnd);       // E3 + rnd
            const __m128i T13B = _mm_add_epi32(E3B, c32_rnd);
            const __m128i T14A = _mm_add_epi32(E4A, c32_rnd);       // E4 + rnd
            const __m128i T14B = _mm_add_epi32(E4B, c32_rnd);
            const __m128i T15A = _mm_add_epi32(E5A, c32_rnd);       // E5 + rnd
            const __m128i T15B = _mm_add_epi32(E5B, c32_rnd);
            const __m128i T16A = _mm_add_epi32(E6A, c32_rnd);       // E6 + rnd
            const __m128i T16B = _mm_add_epi32(E6B, c32_rnd);
            const __m128i T17A = _mm_add_epi32(E7A, c32_rnd);       // E7 + rnd
            const __m128i T17B = _mm_add_epi32(E7B, c32_rnd);
            const __m128i T18A = _mm_add_epi32(E8A, c32_rnd);       // E8 + rnd
            const __m128i T18B = _mm_add_epi32(E8B, c32_rnd);
            const __m128i T19A = _mm_add_epi32(E9A, c32_rnd);       // E9 + rnd
            const __m128i T19B = _mm_add_epi32(E9B, c32_rnd);
            const __m128i T1AA = _mm_add_epi32(EAA, c32_rnd);       // E10 + rnd
            const __m128i T1AB = _mm_add_epi32(EAB, c32_rnd);
            const __m128i T1BA = _mm_add_epi32(EBA, c32_rnd);       // E11 + rnd
            const __m128i T1BB = _mm_add_epi32(EBB, c32_rnd);
            const __m128i T1CA = _mm_add_epi32(ECA, c32_rnd);       // E12 + rnd
            const __m128i T1CB = _mm_add_epi32(ECB, c32_rnd);
            const __m128i T1DA = _mm_add_epi32(EDA, c32_rnd);       // E13 + rnd
            const __m128i T1DB = _mm_add_epi32(EDB, c32_rnd);
            const __m128i T1EA = _mm_add_epi32(EEA, c32_rnd);       // E14 + rnd
            const __m128i T1EB = _mm_add_epi32(EEB, c32_rnd);
            const __m128i T1FA = _mm_add_epi32(EFA, c32_rnd);       // E15 + rnd
            const __m128i T1FB = _mm_add_epi32(EFB, c32_rnd);

            const __m128i T2_00A = _mm_add_epi32(T10A, O00A);       // E0 + O0 + rnd
            const __m128i T2_00B = _mm_add_epi32(T10B, O00B);
            const __m128i T2_01A = _mm_add_epi32(T11A, O01A);       // E1 + O1 + rnd
            const __m128i T2_01B = _mm_add_epi32(T11B, O01B);
            const __m128i T2_02A = _mm_add_epi32(T12A, O02A);       // E2 + O2 + rnd
            const __m128i T2_02B = _mm_add_epi32(T12B, O02B);
            const __m128i T2_03A = _mm_add_epi32(T13A, O03A);       // E3 + O3 + rnd
            const __m128i T2_03B = _mm_add_epi32(T13B, O03B);
            const __m128i T2_04A = _mm_add_epi32(T14A, O04A);       // E4
            const __m128i T2_04B = _mm_add_epi32(T14B, O04B);
            const __m128i T2_05A = _mm_add_epi32(T15A, O05A);       // E5
            const __m128i T2_05B = _mm_add_epi32(T15B, O05B);
            const __m128i T2_06A = _mm_add_epi32(T16A, O06A);       // E6
            const __m128i T2_06B = _mm_add_epi32(T16B, O06B);
            const __m128i T2_07A = _mm_add_epi32(T17A, O07A);       // E7
            const __m128i T2_07B = _mm_add_epi32(T17B, O07B);
            const __m128i T2_08A = _mm_add_epi32(T18A, O08A);       // E8
            const __m128i T2_08B = _mm_add_epi32(T18B, O08B);
            const __m128i T2_09A = _mm_add_epi32(T19A, O09A);       // E9
            const __m128i T2_09B = _mm_add_epi32(T19B, O09B);
            const __m128i T2_10A = _mm_add_epi32(T1AA, O10A);       // E10
            const __m128i T2_10B = _mm_add_epi32(T1AB, O10B);
            const __m128i T2_11A = _mm_add_epi32(T1BA, O11A);       // E11
            const __m128i T2_11B = _mm_add_epi32(T1BB, O11B);
            const __m128i T2_12A = _mm_add_epi32(T1CA, O12A);       // E12
            const __m128i T2_12B = _mm_add_epi32(T1CB, O12B);
            const __m128i T2_13A = _mm_add_epi32(T1DA, O13A);       // E13
            const __m128i T2_13B = _mm_add_epi32(T1DB, O13B);
            const __m128i T2_14A = _mm_add_epi32(T1EA, O14A);       // E14
            const __m128i T2_14B = _mm_add_epi32(T1EB, O14B);
            const __m128i T2_15A = _mm_add_epi32(T1FA, O15A);       // E15
            const __m128i T2_15B = _mm_add_epi32(T1FB, O15B);
            const __m128i T2_31A = _mm_sub_epi32(T10A, O00A);       // E0 - O0 + rnd
            const __m128i T2_31B = _mm_sub_epi32(T10B, O00B);
            const __m128i T2_30A = _mm_sub_epi32(T11A, O01A);       // E1 - O1 + rnd
            const __m128i T2_30B = _mm_sub_epi32(T11B, O01B);
            const __m128i T2_29A = _mm_sub_epi32(T12A, O02A);       // E2 - O2 + rnd
            const __m128i T2_29B = _mm_sub_epi32(T12B, O02B);
            const __m128i T2_28A = _mm_sub_epi32(T13A, O03A);       // E3 - O3 + rnd
            const __m128i T2_28B = _mm_sub_epi32(T13B, O03B);
            const __m128i T2_27A = _mm_sub_epi32(T14A, O04A);       // E4
            const __m128i T2_27B = _mm_sub_epi32(T14B, O04B);
            const __m128i T2_26A = _mm_sub_epi32(T15A, O05A);       // E5
            const __m128i T2_26B = _mm_sub_epi32(T15B, O05B);
            const __m128i T2_25A = _mm_sub_epi32(T16A, O06A);       // E6
            const __m128i T2_25B = _mm_sub_epi32(T16B, O06B);
            const __m128i T2_24A = _mm_sub_epi32(T17A, O07A);       // E7
            const __m128i T2_24B = _mm_sub_epi32(T17B, O07B);
            const __m128i T2_23A = _mm_sub_epi32(T18A, O08A);       //
            const __m128i T2_23B = _mm_sub_epi32(T18B, O08B);
            const __m128i T2_22A = _mm_sub_epi32(T19A, O09A);       //
            const __m128i T2_22B = _mm_sub_epi32(T19B, O09B);
            const __m128i T2_21A = _mm_sub_epi32(T1AA, O10A);       //
            const __m128i T2_21B = _mm_sub_epi32(T1AB, O10B);
            const __m128i T2_20A = _mm_sub_epi32(T1BA, O11A);       //
            const __m128i T2_20B = _mm_sub_epi32(T1BB, O11B);
            const __m128i T2_19A = _mm_sub_epi32(T1CA, O12A);       //
            const __m128i T2_19B = _mm_sub_epi32(T1CB, O12B);
            const __m128i T2_18A = _mm_sub_epi32(T1DA, O13A);       //
            const __m128i T2_18B = _mm_sub_epi32(T1DB, O13B);
            const __m128i T2_17A = _mm_sub_epi32(T1EA, O14A);       //
            const __m128i T2_17B = _mm_sub_epi32(T1EB, O14B);
            const __m128i T2_16A = _mm_sub_epi32(T1FA, O15A);       //
            const __m128i T2_16B = _mm_sub_epi32(T1FB, O15B);

            const __m128i T3_00A = _mm_srai_epi32(T2_00A, nShift);  // [30 20 10 00]
            const __m128i T3_00B = _mm_srai_epi32(T2_00B, nShift);  // [70 60 50 40]
            const __m128i T3_01A = _mm_srai_epi32(T2_01A, nShift);  // [31 21 11 01]
            const __m128i T3_01B = _mm_srai_epi32(T2_01B, nShift);  // [71 61 51 41]
            const __m128i T3_02A = _mm_srai_epi32(T2_02A, nShift);  // [32 22 12 02]
            const __m128i T3_02B = _mm_srai_epi32(T2_02B, nShift);  // [72 62 52 42]
            const __m128i T3_03A = _mm_srai_epi32(T2_03A, nShift);  // [33 23 13 03]
            const __m128i T3_03B = _mm_srai_epi32(T2_03B, nShift);  // [73 63 53 43]
            const __m128i T3_04A = _mm_srai_epi32(T2_04A, nShift);  // [33 24 14 04]
            const __m128i T3_04B = _mm_srai_epi32(T2_04B, nShift);  // [74 64 54 44]
            const __m128i T3_05A = _mm_srai_epi32(T2_05A, nShift);  // [35 25 15 05]
            const __m128i T3_05B = _mm_srai_epi32(T2_05B, nShift);  // [75 65 55 45]
            const __m128i T3_06A = _mm_srai_epi32(T2_06A, nShift);  // [36 26 16 06]
            const __m128i T3_06B = _mm_srai_epi32(T2_06B, nShift);  // [76 66 56 46]
            const __m128i T3_07A = _mm_srai_epi32(T2_07A, nShift);  // [37 27 17 07]
            const __m128i T3_07B = _mm_srai_epi32(T2_07B, nShift);  // [77 67 57 47]
            const __m128i T3_08A = _mm_srai_epi32(T2_08A, nShift);  // [30 20 10 00] x8
            const __m128i T3_08B = _mm_srai_epi32(T2_08B, nShift);  // [70 60 50 40]
            const __m128i T3_09A = _mm_srai_epi32(T2_09A, nShift);  // [31 21 11 01] x9
            const __m128i T3_09B = _mm_srai_epi32(T2_09B, nShift);  // [71 61 51 41]
            const __m128i T3_10A = _mm_srai_epi32(T2_10A, nShift);  // [32 22 12 02] xA
            const __m128i T3_10B = _mm_srai_epi32(T2_10B, nShift);  // [72 62 52 42]
            const __m128i T3_11A = _mm_srai_epi32(T2_11A, nShift);  // [33 23 13 03] xB
            const __m128i T3_11B = _mm_srai_epi32(T2_11B, nShift);  // [73 63 53 43]
            const __m128i T3_12A = _mm_srai_epi32(T2_12A, nShift);  // [33 24 14 04] xC
            const __m128i T3_12B = _mm_srai_epi32(T2_12B, nShift);  // [74 64 54 44]
            const __m128i T3_13A = _mm_srai_epi32(T2_13A, nShift);  // [35 25 15 05] xD
            const __m128i T3_13B = _mm_srai_epi32(T2_13B, nShift);  // [75 65 55 45]
            const __m128i T3_14A = _mm_srai_epi32(T2_14A, nShift);  // [36 26 16 06] xE
            const __m128i T3_14B = _mm_srai_epi32(T2_14B, nShift);  // [76 66 56 46]
            const __m128i T3_15A = _mm_srai_epi32(T2_15A, nShift);  // [37 27 17 07] xF
            const __m128i T3_15B = _mm_srai_epi32(T2_15B, nShift);  // [77 67 57 47]

            const __m128i T3_16A = _mm_srai_epi32(T2_16A, nShift);  // [30 20 10 00]
            const __m128i T3_16B = _mm_srai_epi32(T2_16B, nShift);  // [70 60 50 40]
            const __m128i T3_17A = _mm_srai_epi32(T2_17A, nShift);  // [31 21 11 01]
            const __m128i T3_17B = _mm_srai_epi32(T2_17B, nShift);  // [71 61 51 41]
            const __m128i T3_18A = _mm_srai_epi32(T2_18A, nShift);  // [32 22 12 02]
            const __m128i T3_18B = _mm_srai_epi32(T2_18B, nShift);  // [72 62 52 42]
            const __m128i T3_19A = _mm_srai_epi32(T2_19A, nShift);  // [33 23 13 03]
            const __m128i T3_19B = _mm_srai_epi32(T2_19B, nShift);  // [73 63 53 43]
            const __m128i T3_20A = _mm_srai_epi32(T2_20A, nShift);  // [33 24 14 04]
            const __m128i T3_20B = _mm_srai_epi32(T2_20B, nShift);  // [74 64 54 44]
            const __m128i T3_21A = _mm_srai_epi32(T2_21A, nShift);  // [35 25 15 05]
            const __m128i T3_21B = _mm_srai_epi32(T2_21B, nShift);  // [75 65 55 45]
            const __m128i T3_22A = _mm_srai_epi32(T2_22A, nShift);  // [36 26 16 06]
            const __m128i T3_22B = _mm_srai_epi32(T2_22B, nShift);  // [76 66 56 46]
            const __m128i T3_23A = _mm_srai_epi32(T2_23A, nShift);  // [37 27 17 07]
            const __m128i T3_23B = _mm_srai_epi32(T2_23B, nShift);  // [77 67 57 47]
            const __m128i T3_24A = _mm_srai_epi32(T2_24A, nShift);  // [30 20 10 00] x8
            const __m128i T3_24B = _mm_srai_epi32(T2_24B, nShift);  // [70 60 50 40]
            const __m128i T3_25A = _mm_srai_epi32(T2_25A, nShift);  // [31 21 11 01] x9
            const __m128i T3_25B = _mm_srai_epi32(T2_25B, nShift);  // [71 61 51 41]
            const __m128i T3_26A = _mm_srai_epi32(T2_26A, nShift);  // [32 22 12 02] xA
            const __m128i T3_26B = _mm_srai_epi32(T2_26B, nShift);  // [72 62 52 42]
            const __m128i T3_27A = _mm_srai_epi32(T2_27A, nShift);  // [33 23 13 03] xB
            const __m128i T3_27B = _mm_srai_epi32(T2_27B, nShift);  // [73 63 53 43]
            const __m128i T3_28A = _mm_srai_epi32(T2_28A, nShift);  // [33 24 14 04] xC
            const __m128i T3_28B = _mm_srai_epi32(T2_28B, nShift);  // [74 64 54 44]
            const __m128i T3_29A = _mm_srai_epi32(T2_29A, nShift);  // [35 25 15 05] xD
            const __m128i T3_29B = _mm_srai_epi32(T2_29B, nShift);  // [75 65 55 45]
            const __m128i T3_30A = _mm_srai_epi32(T2_30A, nShift);  // [36 26 16 06] xE
            const __m128i T3_30B = _mm_srai_epi32(T2_30B, nShift);  // [76 66 56 46]
            const __m128i T3_31A = _mm_srai_epi32(T2_31A, nShift);  // [37 27 17 07] xF
            const __m128i T3_31B = _mm_srai_epi32(T2_31B, nShift);  // [77 67 57 47]

            res00[0] = _mm_packs_epi32(T3_00A, T3_00B);             // [70 60 50 40 30 20 10 00]
            res01[0] = _mm_packs_epi32(T3_01A, T3_01B);             // [71 61 51 41 31 21 11 01]
            res02[0] = _mm_packs_epi32(T3_02A, T3_02B);             // [72 62 52 42 32 22 12 02]
            res03[0] = _mm_packs_epi32(T3_03A, T3_03B);             // [73 63 53 43 33 23 13 03]
            res04[0] = _mm_packs_epi32(T3_04A, T3_04B);             // [74 64 54 44 34 24 14 04]
            res05[0] = _mm_packs_epi32(T3_05A, T3_05B);             // [75 65 55 45 35 25 15 05]
            res06[0] = _mm_packs_epi32(T3_06A, T3_06B);             // [76 66 56 46 36 26 16 06]
            res07[0] = _mm_packs_epi32(T3_07A, T3_07B);             // [77 67 57 47 37 27 17 07]
            res08[0] = _mm_packs_epi32(T3_08A, T3_08B);             // [A0 ... 80]
            res09[0] = _mm_packs_epi32(T3_09A, T3_09B);             // [A1 ... 81]
            res10[0] = _mm_packs_epi32(T3_10A, T3_10B);             // [A2 ... 82]
            res11[0] = _mm_packs_epi32(T3_11A, T3_11B);             // [A3 ... 83]
            res12[0] = _mm_packs_epi32(T3_12A, T3_12B);             // [A4 ... 84]
            res13[0] = _mm_packs_epi32(T3_13A, T3_13B);             // [A5 ... 85]
            res14[0] = _mm_packs_epi32(T3_14A, T3_14B);             // [A6 ... 86]
            res15[0] = _mm_packs_epi32(T3_15A, T3_15B);             // [A7 ... 87]
            res16[0] = _mm_packs_epi32(T3_16A, T3_16B);
            res17[0] = _mm_packs_epi32(T3_17A, T3_17B);
            res18[0] = _mm_packs_epi32(T3_18A, T3_18B);
            res19[0] = _mm_packs_epi32(T3_19A, T3_19B);
            res20[0] = _mm_packs_epi32(T3_20A, T3_20B);
            res21[0] = _mm_packs_epi32(T3_21A, T3_21B);
            res22[0] = _mm_packs_epi32(T3_22A, T3_22B);
            res23[0] = _mm_packs_epi32(T3_23A, T3_23B);
            res24[0] = _mm_packs_epi32(T3_24A, T3_24B);
            res25[0] = _mm_packs_epi32(T3_25A, T3_25B);
            res26[0] = _mm_packs_epi32(T3_26A, T3_26B);
            res27[0] = _mm_packs_epi32(T3_27A, T3_27B);
            res28[0] = _mm_packs_epi32(T3_28A, T3_28B);
            res29[0] = _mm_packs_epi32(T3_29A, T3_29B);
            res30[0] = _mm_packs_epi32(T3_30A, T3_30B);
            res31[0] = _mm_packs_epi32(T3_31A, T3_31B);
        }

        //transpose matrix 8x8 16bit.
        {
            __m128i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
            __m128i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;
#define TRANSPOSE_8x8_16BIT(I0, I1, I2, I3, I4, I5, I6, I7, O0, O1, O2, O3, O4, O5, O6, O7) \
    tr0_0 = _mm_unpacklo_epi16(I0, I1); \
    tr0_1 = _mm_unpacklo_epi16(I2, I3); \
    tr0_2 = _mm_unpackhi_epi16(I0, I1); \
    tr0_3 = _mm_unpackhi_epi16(I2, I3); \
    tr0_4 = _mm_unpacklo_epi16(I4, I5); \
    tr0_5 = _mm_unpacklo_epi16(I6, I7); \
    tr0_6 = _mm_unpackhi_epi16(I4, I5); \
    tr0_7 = _mm_unpackhi_epi16(I6, I7); \
    tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1); \
    tr1_1 = _mm_unpacklo_epi32(tr0_2, tr0_3); \
    tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1); \
    tr1_3 = _mm_unpackhi_epi32(tr0_2, tr0_3); \
    tr1_4 = _mm_unpacklo_epi32(tr0_4, tr0_5); \
    tr1_5 = _mm_unpacklo_epi32(tr0_6, tr0_7); \
    tr1_6 = _mm_unpackhi_epi32(tr0_4, tr0_5); \
    tr1_7 = _mm_unpackhi_epi32(tr0_6, tr0_7); \
    O0 = _mm_unpacklo_epi64(tr1_0, tr1_4); \
    O1 = _mm_unpackhi_epi64(tr1_0, tr1_4); \
    O2 = _mm_unpacklo_epi64(tr1_2, tr1_6); \
    O3 = _mm_unpackhi_epi64(tr1_2, tr1_6); \
    O4 = _mm_unpacklo_epi64(tr1_1, tr1_5); \
    O5 = _mm_unpackhi_epi64(tr1_1, tr1_5); \
    O6 = _mm_unpacklo_epi64(tr1_3, tr1_7); \
    O7 = _mm_unpackhi_epi64(tr1_3, tr1_7);

            TRANSPOSE_8x8_16BIT(res00[0], res01[0], res02[0], res03[0], res04[0], res05[0], res06[0], res07[0], m128iS0[0], m128iS1[0], m128iS2[0], m128iS3[0], m128iS4[0], m128iS5[0], m128iS6[0], m128iS7[0])
            TRANSPOSE_8x8_16BIT(res08[0], res09[0], res10[0], res11[0], res12[0], res13[0], res14[0], res15[0], m128iS0[1], m128iS1[1], m128iS2[1], m128iS3[1], m128iS4[1], m128iS5[1], m128iS6[1], m128iS7[1])
            TRANSPOSE_8x8_16BIT(res16[0], res17[0], res18[0], res19[0], res20[0], res21[0], res22[0], res23[0], m128iS0[2], m128iS1[2], m128iS2[2], m128iS3[2], m128iS4[2], m128iS5[2], m128iS6[2], m128iS7[2])
            TRANSPOSE_8x8_16BIT(res24[0], res25[0], res26[0], res27[0], res28[0], res29[0], res30[0], res31[0], m128iS0[3], m128iS1[3], m128iS2[3], m128iS3[3], m128iS4[3], m128iS5[3], m128iS6[3], m128iS7[3])

#undef TRANSPOSE_8x8_16BIT
        }
    }

    //clip
    {
        __m128i max_val = _mm_set1_epi16((1 << (clip_depth2 - 1)) - 1);
        __m128i min_val = _mm_set1_epi16(-(1 << (clip_depth2 - 1)));

        for (i = 0; i < 4; i++) {
            m128iS0[i] = _mm_min_epi16(m128iS0[i], max_val);
            m128iS0[i] = _mm_max_epi16(m128iS0[i], min_val);

            m128iS1[i] = _mm_min_epi16(m128iS1[i], max_val);
            m128iS1[i] = _mm_max_epi16(m128iS1[i], min_val);

            m128iS2[i] = _mm_min_epi16(m128iS2[i], max_val);
            m128iS2[i] = _mm_max_epi16(m128iS2[i], min_val);

            m128iS3[i] = _mm_min_epi16(m128iS3[i], max_val);
            m128iS3[i] = _mm_max_epi16(m128iS3[i], min_val);

            m128iS4[i] = _mm_min_epi16(m128iS4[i], max_val);
            m128iS4[i] = _mm_max_epi16(m128iS4[i], min_val);

            m128iS5[i] = _mm_min_epi16(m128iS5[i], max_val);
            m128iS5[i] = _mm_max_epi16(m128iS5[i], min_val);

            m128iS6[i] = _mm_min_epi16(m128iS6[i], max_val);
            m128iS6[i] = _mm_max_epi16(m128iS6[i], min_val);

            m128iS7[i] = _mm_min_epi16(m128iS7[i], max_val);
            m128iS7[i] = _mm_max_epi16(m128iS7[i], min_val);
        }
    }
    //  coeff_t blk2[32 * 8];

    // Add
    for (i = 0; i < 2; i++) {
#define STORE_LINE(L0, L1, L2, L3, offsetV) \
    _mm_store_si128((__m128i*)(dst + offsetV * i_dst +  0), L0); \
    _mm_store_si128((__m128i*)(dst + offsetV * i_dst +  8), L1); \
    _mm_store_si128((__m128i*)(dst + offsetV * i_dst + 16), L2); \
    _mm_store_si128((__m128i*)(dst + offsetV * i_dst + 24), L3);

        STORE_LINE(m128iS0[0], m128iS0[1], m128iS0[2], m128iS0[3], 0)
        STORE_LINE(m128iS1[0], m128iS1[1], m128iS1[2], m128iS1[3], 1)
        STORE_LINE(m128iS2[0], m128iS2[1], m128iS2[2], m128iS2[3], 2)
        STORE_LINE(m128iS3[0], m128iS3[1], m128iS3[2], m128iS3[3], 3)
        STORE_LINE(m128iS4[0], m128iS4[1], m128iS4[2], m128iS4[3], 4)
        STORE_LINE(m128iS5[0], m128iS5[1], m128iS5[2], m128iS5[3], 5)
        STORE_LINE(m128iS6[0], m128iS6[1], m128iS6[2], m128iS6[3], 6)
        STORE_LINE(m128iS7[0], m128iS7[1], m128iS7[2], m128iS7[3], 7)

#undef STORE_LINE
    }
}

/* ---------------------------------------------------------------------------
 */
void idct_c_8x32_sse128(const coeff_t *src, coeff_t *dst, int i_dst)
{
    const __m128i c16_p45_p45 = _mm_set1_epi32(0x002D002D);
    const __m128i c16_p43_p44 = _mm_set1_epi32(0x002B002C);
    const __m128i c16_p39_p41 = _mm_set1_epi32(0x00270029);
    const __m128i c16_p34_p36 = _mm_set1_epi32(0x00220024);
    const __m128i c16_p27_p30 = _mm_set1_epi32(0x001B001E);
    const __m128i c16_p19_p23 = _mm_set1_epi32(0x00130017);
    const __m128i c16_p11_p15 = _mm_set1_epi32(0x000B000F);
    const __m128i c16_p02_p07 = _mm_set1_epi32(0x00020007);
    const __m128i c16_p41_p45 = _mm_set1_epi32(0x0029002D);
    const __m128i c16_p23_p34 = _mm_set1_epi32(0x00170022);
    const __m128i c16_n02_p11 = _mm_set1_epi32(0xFFFE000B);
    const __m128i c16_n27_n15 = _mm_set1_epi32(0xFFE5FFF1);
    const __m128i c16_n43_n36 = _mm_set1_epi32(0xFFD5FFDC);
    const __m128i c16_n44_n45 = _mm_set1_epi32(0xFFD4FFD3);
    const __m128i c16_n30_n39 = _mm_set1_epi32(0xFFE2FFD9);
    const __m128i c16_n07_n19 = _mm_set1_epi32(0xFFF9FFED);
    const __m128i c16_p34_p44 = _mm_set1_epi32(0x0022002C);
    const __m128i c16_n07_p15 = _mm_set1_epi32(0xFFF9000F);
    const __m128i c16_n41_n27 = _mm_set1_epi32(0xFFD7FFE5);
    const __m128i c16_n39_n45 = _mm_set1_epi32(0xFFD9FFD3);
    const __m128i c16_n02_n23 = _mm_set1_epi32(0xFFFEFFE9);
    const __m128i c16_p36_p19 = _mm_set1_epi32(0x00240013);
    const __m128i c16_p43_p45 = _mm_set1_epi32(0x002B002D);
    const __m128i c16_p11_p30 = _mm_set1_epi32(0x000B001E);
    const __m128i c16_p23_p43 = _mm_set1_epi32(0x0017002B);
    const __m128i c16_n34_n07 = _mm_set1_epi32(0xFFDEFFF9);
    const __m128i c16_n36_n45 = _mm_set1_epi32(0xFFDCFFD3);
    const __m128i c16_p19_n11 = _mm_set1_epi32(0x0013FFF5);
    const __m128i c16_p44_p41 = _mm_set1_epi32(0x002C0029);
    const __m128i c16_n02_p27 = _mm_set1_epi32(0xFFFE001B);
    const __m128i c16_n45_n30 = _mm_set1_epi32(0xFFD3FFE2);
    const __m128i c16_n15_n39 = _mm_set1_epi32(0xFFF1FFD9);
    const __m128i c16_p11_p41 = _mm_set1_epi32(0x000B0029);
    const __m128i c16_n45_n27 = _mm_set1_epi32(0xFFD3FFE5);
    const __m128i c16_p07_n30 = _mm_set1_epi32(0x0007FFE2);
    const __m128i c16_p43_p39 = _mm_set1_epi32(0x002B0027);
    const __m128i c16_n23_p15 = _mm_set1_epi32(0xFFE9000F);
    const __m128i c16_n34_n45 = _mm_set1_epi32(0xFFDEFFD3);
    const __m128i c16_p36_p02 = _mm_set1_epi32(0x00240002);
    const __m128i c16_p19_p44 = _mm_set1_epi32(0x0013002C);
    const __m128i c16_n02_p39 = _mm_set1_epi32(0xFFFE0027);
    const __m128i c16_n36_n41 = _mm_set1_epi32(0xFFDCFFD7);
    const __m128i c16_p43_p07 = _mm_set1_epi32(0x002B0007);
    const __m128i c16_n11_p34 = _mm_set1_epi32(0xFFF50022);
    const __m128i c16_n30_n44 = _mm_set1_epi32(0xFFE2FFD4);
    const __m128i c16_p45_p15 = _mm_set1_epi32(0x002D000F);
    const __m128i c16_n19_p27 = _mm_set1_epi32(0xFFED001B);
    const __m128i c16_n23_n45 = _mm_set1_epi32(0xFFE9FFD3);
    const __m128i c16_n15_p36 = _mm_set1_epi32(0xFFF10024);
    const __m128i c16_n11_n45 = _mm_set1_epi32(0xFFF5FFD3);
    const __m128i c16_p34_p39 = _mm_set1_epi32(0x00220027);
    const __m128i c16_n45_n19 = _mm_set1_epi32(0xFFD3FFED);
    const __m128i c16_p41_n07 = _mm_set1_epi32(0x0029FFF9);
    const __m128i c16_n23_p30 = _mm_set1_epi32(0xFFE9001E);
    const __m128i c16_n02_n44 = _mm_set1_epi32(0xFFFEFFD4);
    const __m128i c16_p27_p43 = _mm_set1_epi32(0x001B002B);
    const __m128i c16_n27_p34 = _mm_set1_epi32(0xFFE50022);
    const __m128i c16_p19_n39 = _mm_set1_epi32(0x0013FFD9);
    const __m128i c16_n11_p43 = _mm_set1_epi32(0xFFF5002B);
    const __m128i c16_p02_n45 = _mm_set1_epi32(0x0002FFD3);
    const __m128i c16_p07_p45 = _mm_set1_epi32(0x0007002D);
    const __m128i c16_n15_n44 = _mm_set1_epi32(0xFFF1FFD4);
    const __m128i c16_p23_p41 = _mm_set1_epi32(0x00170029);
    const __m128i c16_n30_n36 = _mm_set1_epi32(0xFFE2FFDC);
    const __m128i c16_n36_p30 = _mm_set1_epi32(0xFFDC001E);
    const __m128i c16_p41_n23 = _mm_set1_epi32(0x0029FFE9);
    const __m128i c16_n44_p15 = _mm_set1_epi32(0xFFD4000F);
    const __m128i c16_p45_n07 = _mm_set1_epi32(0x002DFFF9);
    const __m128i c16_n45_n02 = _mm_set1_epi32(0xFFD3FFFE);
    const __m128i c16_p43_p11 = _mm_set1_epi32(0x002B000B);
    const __m128i c16_n39_n19 = _mm_set1_epi32(0xFFD9FFED);
    const __m128i c16_p34_p27 = _mm_set1_epi32(0x0022001B);
    const __m128i c16_n43_p27 = _mm_set1_epi32(0xFFD5001B);
    const __m128i c16_p44_n02 = _mm_set1_epi32(0x002CFFFE);
    const __m128i c16_n30_n23 = _mm_set1_epi32(0xFFE2FFE9);
    const __m128i c16_p07_p41 = _mm_set1_epi32(0x00070029);
    const __m128i c16_p19_n45 = _mm_set1_epi32(0x0013FFD3);
    const __m128i c16_n39_p34 = _mm_set1_epi32(0xFFD90022);
    const __m128i c16_p45_n11 = _mm_set1_epi32(0x002DFFF5);
    const __m128i c16_n36_n15 = _mm_set1_epi32(0xFFDCFFF1);
    const __m128i c16_n45_p23 = _mm_set1_epi32(0xFFD30017);
    const __m128i c16_p27_p19 = _mm_set1_epi32(0x001B0013);
    const __m128i c16_p15_n45 = _mm_set1_epi32(0x000FFFD3);
    const __m128i c16_n44_p30 = _mm_set1_epi32(0xFFD4001E);
    const __m128i c16_p34_p11 = _mm_set1_epi32(0x0022000B);
    const __m128i c16_p07_n43 = _mm_set1_epi32(0x0007FFD5);
    const __m128i c16_n41_p36 = _mm_set1_epi32(0xFFD70024);
    const __m128i c16_p39_p02 = _mm_set1_epi32(0x00270002);
    const __m128i c16_n44_p19 = _mm_set1_epi32(0xFFD40013);
    const __m128i c16_n02_p36 = _mm_set1_epi32(0xFFFE0024);
    const __m128i c16_p45_n34 = _mm_set1_epi32(0x002DFFDE);
    const __m128i c16_n15_n23 = _mm_set1_epi32(0xFFF1FFE9);
    const __m128i c16_n39_p43 = _mm_set1_epi32(0xFFD9002B);
    const __m128i c16_p30_p07 = _mm_set1_epi32(0x001E0007);
    const __m128i c16_p27_n45 = _mm_set1_epi32(0x001BFFD3);
    const __m128i c16_n41_p11 = _mm_set1_epi32(0xFFD7000B);
    const __m128i c16_n39_p15 = _mm_set1_epi32(0xFFD9000F);
    const __m128i c16_n30_p45 = _mm_set1_epi32(0xFFE2002D);
    const __m128i c16_p27_p02 = _mm_set1_epi32(0x001B0002);
    const __m128i c16_p41_n44 = _mm_set1_epi32(0x0029FFD4);
    const __m128i c16_n11_n19 = _mm_set1_epi32(0xFFF5FFED);
    const __m128i c16_n45_p36 = _mm_set1_epi32(0xFFD30024);
    const __m128i c16_n07_p34 = _mm_set1_epi32(0xFFF90022);
    const __m128i c16_p43_n23 = _mm_set1_epi32(0x002BFFE9);
    const __m128i c16_n30_p11 = _mm_set1_epi32(0xFFE2000B);
    const __m128i c16_n45_p43 = _mm_set1_epi32(0xFFD3002B);
    const __m128i c16_n19_p36 = _mm_set1_epi32(0xFFED0024);
    const __m128i c16_p23_n02 = _mm_set1_epi32(0x0017FFFE);
    const __m128i c16_p45_n39 = _mm_set1_epi32(0x002DFFD9);
    const __m128i c16_p27_n41 = _mm_set1_epi32(0x001BFFD7);
    const __m128i c16_n15_n07 = _mm_set1_epi32(0xFFF1FFF9);
    const __m128i c16_n44_p34 = _mm_set1_epi32(0xFFD40022);
    const __m128i c16_n19_p07 = _mm_set1_epi32(0xFFED0007);
    const __m128i c16_n39_p30 = _mm_set1_epi32(0xFFD9001E);
    const __m128i c16_n45_p44 = _mm_set1_epi32(0xFFD3002C);
    const __m128i c16_n36_p43 = _mm_set1_epi32(0xFFDC002B);
    const __m128i c16_n15_p27 = _mm_set1_epi32(0xFFF1001B);
    const __m128i c16_p11_p02 = _mm_set1_epi32(0x000B0002);
    const __m128i c16_p34_n23 = _mm_set1_epi32(0x0022FFE9);
    const __m128i c16_p45_n41 = _mm_set1_epi32(0x002DFFD7);
    const __m128i c16_n07_p02 = _mm_set1_epi32(0xFFF90002);
    const __m128i c16_n15_p11 = _mm_set1_epi32(0xFFF1000B);
    const __m128i c16_n23_p19 = _mm_set1_epi32(0xFFE90013);
    const __m128i c16_n30_p27 = _mm_set1_epi32(0xFFE2001B);
    const __m128i c16_n36_p34 = _mm_set1_epi32(0xFFDC0022);
    const __m128i c16_n41_p39 = _mm_set1_epi32(0xFFD70027);
    const __m128i c16_n44_p43 = _mm_set1_epi32(0xFFD4002B);
    const __m128i c16_n45_p45 = _mm_set1_epi32(0xFFD3002D);

    //  const __m128i c16_p43_p45 = _mm_set1_epi32(0x002B002D);
    const __m128i c16_p35_p40 = _mm_set1_epi32(0x00230028);
    const __m128i c16_p21_p29 = _mm_set1_epi32(0x0015001D);
    const __m128i c16_p04_p13 = _mm_set1_epi32(0x0004000D);
    const __m128i c16_p29_p43 = _mm_set1_epi32(0x001D002B);
    const __m128i c16_n21_p04 = _mm_set1_epi32(0xFFEB0004);
    const __m128i c16_n45_n40 = _mm_set1_epi32(0xFFD3FFD8);
    const __m128i c16_n13_n35 = _mm_set1_epi32(0xFFF3FFDD);
    const __m128i c16_p04_p40 = _mm_set1_epi32(0x00040028);
    const __m128i c16_n43_n35 = _mm_set1_epi32(0xFFD5FFDD);
    const __m128i c16_p29_n13 = _mm_set1_epi32(0x001DFFF3);
    const __m128i c16_p21_p45 = _mm_set1_epi32(0x0015002D);
    const __m128i c16_n21_p35 = _mm_set1_epi32(0xFFEB0023);
    const __m128i c16_p04_n43 = _mm_set1_epi32(0x0004FFD5);
    const __m128i c16_p13_p45 = _mm_set1_epi32(0x000D002D);
    const __m128i c16_n29_n40 = _mm_set1_epi32(0xFFE3FFD8);
    const __m128i c16_n40_p29 = _mm_set1_epi32(0xFFD8001D);
    const __m128i c16_p45_n13 = _mm_set1_epi32(0x002DFFF3);
    const __m128i c16_n43_n04 = _mm_set1_epi32(0xFFD5FFFC);
    const __m128i c16_p35_p21 = _mm_set1_epi32(0x00230015);
    const __m128i c16_n45_p21 = _mm_set1_epi32(0xFFD30015);
    const __m128i c16_p13_p29 = _mm_set1_epi32(0x000D001D);
    const __m128i c16_p35_n43 = _mm_set1_epi32(0x0023FFD5);
    const __m128i c16_n40_p04 = _mm_set1_epi32(0xFFD80004);
    const __m128i c16_n35_p13 = _mm_set1_epi32(0xFFDD000D);
    const __m128i c16_n40_p45 = _mm_set1_epi32(0xFFD8002D);
    const __m128i c16_p04_p21 = _mm_set1_epi32(0x00040015);
    const __m128i c16_p43_n29 = _mm_set1_epi32(0x002BFFE3);
    const __m128i c16_n13_p04 = _mm_set1_epi32(0xFFF30004);
    const __m128i c16_n29_p21 = _mm_set1_epi32(0xFFE30015);
    const __m128i c16_n40_p35 = _mm_set1_epi32(0xFFD80023);
    //  const __m128i c16_n45_p43 = _mm_set1_epi32(0xFFD3002B);

    const __m128i c16_p38_p44 = _mm_set1_epi32(0x0026002C);
    const __m128i c16_p09_p25 = _mm_set1_epi32(0x00090019);
    const __m128i c16_n09_p38 = _mm_set1_epi32(0xFFF70026);
    const __m128i c16_n25_n44 = _mm_set1_epi32(0xFFE7FFD4);

    const __m128i c16_n44_p25 = _mm_set1_epi32(0xFFD40019);
    const __m128i c16_p38_p09 = _mm_set1_epi32(0x00260009);
    const __m128i c16_n25_p09 = _mm_set1_epi32(0xFFE70009);
    const __m128i c16_n44_p38 = _mm_set1_epi32(0xFFD40026);

    const __m128i c16_p17_p42 = _mm_set1_epi32(0x0011002A);
    const __m128i c16_n42_p17 = _mm_set1_epi32(0xFFD60011);

    const __m128i c16_p32_p32 = _mm_set1_epi32(0x00200020);
    const __m128i c16_n32_p32 = _mm_set1_epi32(0xFFE00020);

    __m128i c32_rnd = _mm_set1_epi32(16);

    int nShift = 5, pass;
    //int shift1 = 5;
    int shift2 = 20 - g_bit_depth - (i_dst & 0x01);
    //int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = g_bit_depth + 1 + (i_dst & 0x01);

    // DCT1
    __m128i in00, in01, in02, in03, in04, in05, in06, in07, in08, in09, in10, in11, in12, in13, in14, in15;
    __m128i in16, in17, in18, in19, in20, in21, in22, in23, in24, in25, in26, in27, in28, in29, in30, in31;
    __m128i res00[4], res01[4], res02[4], res03[4], res04[4], res05[4], res06[4], res07[4];

    i_dst &= 0xFE;

    in00 = _mm_load_si128((const __m128i*)&src[0 * 8]);
    in01 = _mm_load_si128((const __m128i*)&src[ 1 * 8]);
    in02 = _mm_load_si128((const __m128i*)&src[ 2 * 8]);
    in03 = _mm_load_si128((const __m128i*)&src[ 3 * 8]);
    in04 = _mm_load_si128((const __m128i*)&src[ 4 * 8]);
    in05 = _mm_load_si128((const __m128i*)&src[ 5 * 8]);
    in06 = _mm_load_si128((const __m128i*)&src[ 6 * 8]);
    in07 = _mm_load_si128((const __m128i*)&src[ 7 * 8]);
    in08 = _mm_load_si128((const __m128i*)&src[ 8 * 8]);
    in09 = _mm_load_si128((const __m128i*)&src[ 9 * 8]);
    in10 = _mm_load_si128((const __m128i*)&src[10 * 8]);
    in11 = _mm_load_si128((const __m128i*)&src[11 * 8]);
    in12 = _mm_load_si128((const __m128i*)&src[12 * 8]);
    in13 = _mm_load_si128((const __m128i*)&src[13 * 8]);
    in14 = _mm_load_si128((const __m128i*)&src[14 * 8]);
    in15 = _mm_load_si128((const __m128i*)&src[15 * 8]);
    in16 = _mm_load_si128((const __m128i*)&src[16 * 8]);
    in17 = _mm_load_si128((const __m128i*)&src[17 * 8]);
    in18 = _mm_load_si128((const __m128i*)&src[18 * 8]);
    in19 = _mm_load_si128((const __m128i*)&src[19 * 8]);
    in20 = _mm_load_si128((const __m128i*)&src[20 * 8]);
    in21 = _mm_load_si128((const __m128i*)&src[21 * 8]);
    in22 = _mm_load_si128((const __m128i*)&src[22 * 8]);
    in23 = _mm_load_si128((const __m128i*)&src[23 * 8]);
    in24 = _mm_load_si128((const __m128i*)&src[24 * 8]);
    in25 = _mm_load_si128((const __m128i*)&src[25 * 8]);
    in26 = _mm_load_si128((const __m128i*)&src[26 * 8]);
    in27 = _mm_load_si128((const __m128i*)&src[27 * 8]);
    in28 = _mm_load_si128((const __m128i*)&src[28 * 8]);
    in29 = _mm_load_si128((const __m128i*)&src[29 * 8]);
    in30 = _mm_load_si128((const __m128i*)&src[30 * 8]);
    in31 = _mm_load_si128((const __m128i*)&src[31 * 8]);

    {
        const __m128i T_00_00A = _mm_unpacklo_epi16(in01, in03);    // [33 13 32 12 31 11 30 10]
        const __m128i T_00_00B = _mm_unpackhi_epi16(in01, in03);    // [37 17 36 16 35 15 34 14]
        const __m128i T_00_01A = _mm_unpacklo_epi16(in05, in07);    // [ ]
        const __m128i T_00_01B = _mm_unpackhi_epi16(in05, in07);    // [ ]
        const __m128i T_00_02A = _mm_unpacklo_epi16(in09, in11);    // [ ]
        const __m128i T_00_02B = _mm_unpackhi_epi16(in09, in11);    // [ ]
        const __m128i T_00_03A = _mm_unpacklo_epi16(in13, in15);    // [ ]
        const __m128i T_00_03B = _mm_unpackhi_epi16(in13, in15);    // [ ]
        const __m128i T_00_04A = _mm_unpacklo_epi16(in17, in19);    // [ ]
        const __m128i T_00_04B = _mm_unpackhi_epi16(in17, in19);    // [ ]
        const __m128i T_00_05A = _mm_unpacklo_epi16(in21, in23);    // [ ]
        const __m128i T_00_05B = _mm_unpackhi_epi16(in21, in23);    // [ ]
        const __m128i T_00_06A = _mm_unpacklo_epi16(in25, in27);    // [ ]
        const __m128i T_00_06B = _mm_unpackhi_epi16(in25, in27);    // [ ]
        const __m128i T_00_07A = _mm_unpacklo_epi16(in29, in31);    //
        const __m128i T_00_07B = _mm_unpackhi_epi16(in29, in31);    // [ ]

        const __m128i T_00_08A = _mm_unpacklo_epi16(in02, in06);    // [ ]
        const __m128i T_00_08B = _mm_unpackhi_epi16(in02, in06);    // [ ]
        const __m128i T_00_09A = _mm_unpacklo_epi16(in10, in14);    // [ ]
        const __m128i T_00_09B = _mm_unpackhi_epi16(in10, in14);    // [ ]
        const __m128i T_00_10A = _mm_unpacklo_epi16(in18, in22);    // [ ]
        const __m128i T_00_10B = _mm_unpackhi_epi16(in18, in22);    // [ ]
        const __m128i T_00_11A = _mm_unpacklo_epi16(in26, in30);    // [ ]
        const __m128i T_00_11B = _mm_unpackhi_epi16(in26, in30);    // [ ]

        const __m128i T_00_12A = _mm_unpacklo_epi16(in04, in12);    // [ ]
        const __m128i T_00_12B = _mm_unpackhi_epi16(in04, in12);    // [ ]
        const __m128i T_00_13A = _mm_unpacklo_epi16(in20, in28);    // [ ]
        const __m128i T_00_13B = _mm_unpackhi_epi16(in20, in28);    // [ ]

        const __m128i T_00_14A = _mm_unpacklo_epi16(in08, in24);    //
        const __m128i T_00_14B = _mm_unpackhi_epi16(in08, in24);    // [ ]
        const __m128i T_00_15A = _mm_unpacklo_epi16(in00, in16);    //
        const __m128i T_00_15B = _mm_unpackhi_epi16(in00, in16);    // [ ]

        __m128i O00A, O01A, O02A, O03A, O04A, O05A, O06A, O07A, O08A, O09A, O10A, O11A, O12A, O13A, O14A, O15A;
        __m128i O00B, O01B, O02B, O03B, O04B, O05B, O06B, O07B, O08B, O09B, O10B, O11B, O12B, O13B, O14B, O15B;
        __m128i EO0A, EO1A, EO2A, EO3A, EO4A, EO5A, EO6A, EO7A;
        __m128i EO0B, EO1B, EO2B, EO3B, EO4B, EO5B, EO6B, EO7B;
        {
            __m128i T00, T01, T02, T03;
#define COMPUTE_ROW(r0103, r0507, r0911, r1315, r1719, r2123, r2527, r2931, c0103, c0507, c0911, c1315, c1719, c2123, c2527, c2931, row) \
    T00 = _mm_add_epi32(_mm_madd_epi16(r0103, c0103), _mm_madd_epi16(r0507, c0507)); \
    T01 = _mm_add_epi32(_mm_madd_epi16(r0911, c0911), _mm_madd_epi16(r1315, c1315)); \
    T02 = _mm_add_epi32(_mm_madd_epi16(r1719, c1719), _mm_madd_epi16(r2123, c2123)); \
    T03 = _mm_add_epi32(_mm_madd_epi16(r2527, c2527), _mm_madd_epi16(r2931, c2931)); \
    row = _mm_add_epi32(_mm_add_epi32(T00, T01), _mm_add_epi32(T02, T03));

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

#undef COMPUTE_ROW
        }

        {
            __m128i T00, T01;
#define COMPUTE_ROW(row0206, row1014, row1822, row2630, c0206, c1014, c1822, c2630, row) \
    T00 = _mm_add_epi32(_mm_madd_epi16(row0206, c0206), _mm_madd_epi16(row1014, c1014)); \
    T01 = _mm_add_epi32(_mm_madd_epi16(row1822, c1822), _mm_madd_epi16(row2630, c2630)); \
    row = _mm_add_epi32(T00, T01);

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
#undef COMPUTE_ROW
        }

        {
            const __m128i EEO0A = _mm_add_epi32(_mm_madd_epi16(T_00_12A, c16_p38_p44), _mm_madd_epi16(T_00_13A, c16_p09_p25));
            const __m128i EEO1A = _mm_add_epi32(_mm_madd_epi16(T_00_12A, c16_n09_p38), _mm_madd_epi16(T_00_13A, c16_n25_n44));
            const __m128i EEO2A = _mm_add_epi32(_mm_madd_epi16(T_00_12A, c16_n44_p25), _mm_madd_epi16(T_00_13A, c16_p38_p09));
            const __m128i EEO3A = _mm_add_epi32(_mm_madd_epi16(T_00_12A, c16_n25_p09), _mm_madd_epi16(T_00_13A, c16_n44_p38));
            const __m128i EEO0B = _mm_add_epi32(_mm_madd_epi16(T_00_12B, c16_p38_p44), _mm_madd_epi16(T_00_13B, c16_p09_p25));
            const __m128i EEO1B = _mm_add_epi32(_mm_madd_epi16(T_00_12B, c16_n09_p38), _mm_madd_epi16(T_00_13B, c16_n25_n44));
            const __m128i EEO2B = _mm_add_epi32(_mm_madd_epi16(T_00_12B, c16_n44_p25), _mm_madd_epi16(T_00_13B, c16_p38_p09));
            const __m128i EEO3B = _mm_add_epi32(_mm_madd_epi16(T_00_12B, c16_n25_p09), _mm_madd_epi16(T_00_13B, c16_n44_p38));

            const __m128i EEEO0A = _mm_madd_epi16(T_00_14A, c16_p17_p42);
            const __m128i EEEO0B = _mm_madd_epi16(T_00_14B, c16_p17_p42);
            const __m128i EEEO1A = _mm_madd_epi16(T_00_14A, c16_n42_p17);
            const __m128i EEEO1B = _mm_madd_epi16(T_00_14B, c16_n42_p17);

            const __m128i EEEE0A = _mm_madd_epi16(T_00_15A, c16_p32_p32);
            const __m128i EEEE0B = _mm_madd_epi16(T_00_15B, c16_p32_p32);
            const __m128i EEEE1A = _mm_madd_epi16(T_00_15A, c16_n32_p32);
            const __m128i EEEE1B = _mm_madd_epi16(T_00_15B, c16_n32_p32);

            const __m128i EEE0A = _mm_add_epi32(EEEE0A, EEEO0A);    // EEE0 = EEEE0 + EEEO0
            const __m128i EEE0B = _mm_add_epi32(EEEE0B, EEEO0B);
            const __m128i EEE1A = _mm_add_epi32(EEEE1A, EEEO1A);    // EEE1 = EEEE1 + EEEO1
            const __m128i EEE1B = _mm_add_epi32(EEEE1B, EEEO1B);
            const __m128i EEE3A = _mm_sub_epi32(EEEE0A, EEEO0A);    // EEE2 = EEEE0 - EEEO0
            const __m128i EEE3B = _mm_sub_epi32(EEEE0B, EEEO0B);
            const __m128i EEE2A = _mm_sub_epi32(EEEE1A, EEEO1A);    // EEE3 = EEEE1 - EEEO1
            const __m128i EEE2B = _mm_sub_epi32(EEEE1B, EEEO1B);

            const __m128i EE0A = _mm_add_epi32(EEE0A, EEO0A);       // EE0 = EEE0 + EEO0
            const __m128i EE0B = _mm_add_epi32(EEE0B, EEO0B);
            const __m128i EE1A = _mm_add_epi32(EEE1A, EEO1A);       // EE1 = EEE1 + EEO1
            const __m128i EE1B = _mm_add_epi32(EEE1B, EEO1B);
            const __m128i EE2A = _mm_add_epi32(EEE2A, EEO2A);       // EE2 = EEE0 + EEO0
            const __m128i EE2B = _mm_add_epi32(EEE2B, EEO2B);
            const __m128i EE3A = _mm_add_epi32(EEE3A, EEO3A);       // EE3 = EEE1 + EEO1
            const __m128i EE3B = _mm_add_epi32(EEE3B, EEO3B);
            const __m128i EE7A = _mm_sub_epi32(EEE0A, EEO0A);       // EE7 = EEE0 - EEO0
            const __m128i EE7B = _mm_sub_epi32(EEE0B, EEO0B);
            const __m128i EE6A = _mm_sub_epi32(EEE1A, EEO1A);       // EE6 = EEE1 - EEO1
            const __m128i EE6B = _mm_sub_epi32(EEE1B, EEO1B);
            const __m128i EE5A = _mm_sub_epi32(EEE2A, EEO2A);       // EE5 = EEE0 - EEO0
            const __m128i EE5B = _mm_sub_epi32(EEE2B, EEO2B);
            const __m128i EE4A = _mm_sub_epi32(EEE3A, EEO3A);       // EE4 = EEE1 - EEO1
            const __m128i EE4B = _mm_sub_epi32(EEE3B, EEO3B);

            const __m128i E0A = _mm_add_epi32(EE0A, EO0A);          // E0 = EE0 + EO0
            const __m128i E0B = _mm_add_epi32(EE0B, EO0B);
            const __m128i E1A = _mm_add_epi32(EE1A, EO1A);          // E1 = EE1 + EO1
            const __m128i E1B = _mm_add_epi32(EE1B, EO1B);
            const __m128i E2A = _mm_add_epi32(EE2A, EO2A);          // E2 = EE2 + EO2
            const __m128i E2B = _mm_add_epi32(EE2B, EO2B);
            const __m128i E3A = _mm_add_epi32(EE3A, EO3A);          // E3 = EE3 + EO3
            const __m128i E3B = _mm_add_epi32(EE3B, EO3B);
            const __m128i E4A = _mm_add_epi32(EE4A, EO4A);          // E4 =
            const __m128i E4B = _mm_add_epi32(EE4B, EO4B);
            const __m128i E5A = _mm_add_epi32(EE5A, EO5A);          // E5 =
            const __m128i E5B = _mm_add_epi32(EE5B, EO5B);
            const __m128i E6A = _mm_add_epi32(EE6A, EO6A);          // E6 =
            const __m128i E6B = _mm_add_epi32(EE6B, EO6B);
            const __m128i E7A = _mm_add_epi32(EE7A, EO7A);          // E7 =
            const __m128i E7B = _mm_add_epi32(EE7B, EO7B);
            const __m128i EFA = _mm_sub_epi32(EE0A, EO0A);          // EF = EE0 - EO0
            const __m128i EFB = _mm_sub_epi32(EE0B, EO0B);
            const __m128i EEA = _mm_sub_epi32(EE1A, EO1A);          // EE = EE1 - EO1
            const __m128i EEB = _mm_sub_epi32(EE1B, EO1B);
            const __m128i EDA = _mm_sub_epi32(EE2A, EO2A);          // ED = EE2 - EO2
            const __m128i EDB = _mm_sub_epi32(EE2B, EO2B);
            const __m128i ECA = _mm_sub_epi32(EE3A, EO3A);          // EC = EE3 - EO3
            const __m128i ECB = _mm_sub_epi32(EE3B, EO3B);
            const __m128i EBA = _mm_sub_epi32(EE4A, EO4A);          // EB =
            const __m128i EBB = _mm_sub_epi32(EE4B, EO4B);
            const __m128i EAA = _mm_sub_epi32(EE5A, EO5A);          // EA =
            const __m128i EAB = _mm_sub_epi32(EE5B, EO5B);
            const __m128i E9A = _mm_sub_epi32(EE6A, EO6A);          // E9 =
            const __m128i E9B = _mm_sub_epi32(EE6B, EO6B);
            const __m128i E8A = _mm_sub_epi32(EE7A, EO7A);          // E8 =
            const __m128i E8B = _mm_sub_epi32(EE7B, EO7B);

            const __m128i T10A = _mm_add_epi32(E0A, c32_rnd);       // E0 + rnd
            const __m128i T10B = _mm_add_epi32(E0B, c32_rnd);
            const __m128i T11A = _mm_add_epi32(E1A, c32_rnd);       // E1 + rnd
            const __m128i T11B = _mm_add_epi32(E1B, c32_rnd);
            const __m128i T12A = _mm_add_epi32(E2A, c32_rnd);       // E2 + rnd
            const __m128i T12B = _mm_add_epi32(E2B, c32_rnd);
            const __m128i T13A = _mm_add_epi32(E3A, c32_rnd);       // E3 + rnd
            const __m128i T13B = _mm_add_epi32(E3B, c32_rnd);
            const __m128i T14A = _mm_add_epi32(E4A, c32_rnd);       // E4 + rnd
            const __m128i T14B = _mm_add_epi32(E4B, c32_rnd);
            const __m128i T15A = _mm_add_epi32(E5A, c32_rnd);       // E5 + rnd
            const __m128i T15B = _mm_add_epi32(E5B, c32_rnd);
            const __m128i T16A = _mm_add_epi32(E6A, c32_rnd);       // E6 + rnd
            const __m128i T16B = _mm_add_epi32(E6B, c32_rnd);
            const __m128i T17A = _mm_add_epi32(E7A, c32_rnd);       // E7 + rnd
            const __m128i T17B = _mm_add_epi32(E7B, c32_rnd);
            const __m128i T18A = _mm_add_epi32(E8A, c32_rnd);       // E8 + rnd
            const __m128i T18B = _mm_add_epi32(E8B, c32_rnd);
            const __m128i T19A = _mm_add_epi32(E9A, c32_rnd);       // E9 + rnd
            const __m128i T19B = _mm_add_epi32(E9B, c32_rnd);
            const __m128i T1AA = _mm_add_epi32(EAA, c32_rnd);       // E10 + rnd
            const __m128i T1AB = _mm_add_epi32(EAB, c32_rnd);
            const __m128i T1BA = _mm_add_epi32(EBA, c32_rnd);       // E11 + rnd
            const __m128i T1BB = _mm_add_epi32(EBB, c32_rnd);
            const __m128i T1CA = _mm_add_epi32(ECA, c32_rnd);       // E12 + rnd
            const __m128i T1CB = _mm_add_epi32(ECB, c32_rnd);
            const __m128i T1DA = _mm_add_epi32(EDA, c32_rnd);       // E13 + rnd
            const __m128i T1DB = _mm_add_epi32(EDB, c32_rnd);
            const __m128i T1EA = _mm_add_epi32(EEA, c32_rnd);       // E14 + rnd
            const __m128i T1EB = _mm_add_epi32(EEB, c32_rnd);
            const __m128i T1FA = _mm_add_epi32(EFA, c32_rnd);       // E15 + rnd
            const __m128i T1FB = _mm_add_epi32(EFB, c32_rnd);

            const __m128i T2_00A = _mm_add_epi32(T10A, O00A);       // E0 + O0 + rnd
            const __m128i T2_00B = _mm_add_epi32(T10B, O00B);
            const __m128i T2_01A = _mm_add_epi32(T11A, O01A);       // E1 + O1 + rnd
            const __m128i T2_01B = _mm_add_epi32(T11B, O01B);
            const __m128i T2_02A = _mm_add_epi32(T12A, O02A);       // E2 + O2 + rnd
            const __m128i T2_02B = _mm_add_epi32(T12B, O02B);
            const __m128i T2_03A = _mm_add_epi32(T13A, O03A);       // E3 + O3 + rnd
            const __m128i T2_03B = _mm_add_epi32(T13B, O03B);
            const __m128i T2_04A = _mm_add_epi32(T14A, O04A);       // E4
            const __m128i T2_04B = _mm_add_epi32(T14B, O04B);
            const __m128i T2_05A = _mm_add_epi32(T15A, O05A);       // E5
            const __m128i T2_05B = _mm_add_epi32(T15B, O05B);
            const __m128i T2_06A = _mm_add_epi32(T16A, O06A);       // E6
            const __m128i T2_06B = _mm_add_epi32(T16B, O06B);
            const __m128i T2_07A = _mm_add_epi32(T17A, O07A);       // E7
            const __m128i T2_07B = _mm_add_epi32(T17B, O07B);
            const __m128i T2_08A = _mm_add_epi32(T18A, O08A);       // E8
            const __m128i T2_08B = _mm_add_epi32(T18B, O08B);
            const __m128i T2_09A = _mm_add_epi32(T19A, O09A);       // E9
            const __m128i T2_09B = _mm_add_epi32(T19B, O09B);
            const __m128i T2_10A = _mm_add_epi32(T1AA, O10A);       // E10
            const __m128i T2_10B = _mm_add_epi32(T1AB, O10B);
            const __m128i T2_11A = _mm_add_epi32(T1BA, O11A);       // E11
            const __m128i T2_11B = _mm_add_epi32(T1BB, O11B);
            const __m128i T2_12A = _mm_add_epi32(T1CA, O12A);       // E12
            const __m128i T2_12B = _mm_add_epi32(T1CB, O12B);
            const __m128i T2_13A = _mm_add_epi32(T1DA, O13A);       // E13
            const __m128i T2_13B = _mm_add_epi32(T1DB, O13B);
            const __m128i T2_14A = _mm_add_epi32(T1EA, O14A);       // E14
            const __m128i T2_14B = _mm_add_epi32(T1EB, O14B);
            const __m128i T2_15A = _mm_add_epi32(T1FA, O15A);       // E15
            const __m128i T2_15B = _mm_add_epi32(T1FB, O15B);
            const __m128i T2_31A = _mm_sub_epi32(T10A, O00A);       // E0 - O0 + rnd
            const __m128i T2_31B = _mm_sub_epi32(T10B, O00B);
            const __m128i T2_30A = _mm_sub_epi32(T11A, O01A);       // E1 - O1 + rnd
            const __m128i T2_30B = _mm_sub_epi32(T11B, O01B);
            const __m128i T2_29A = _mm_sub_epi32(T12A, O02A);       // E2 - O2 + rnd
            const __m128i T2_29B = _mm_sub_epi32(T12B, O02B);
            const __m128i T2_28A = _mm_sub_epi32(T13A, O03A);       // E3 - O3 + rnd
            const __m128i T2_28B = _mm_sub_epi32(T13B, O03B);
            const __m128i T2_27A = _mm_sub_epi32(T14A, O04A);       // E4
            const __m128i T2_27B = _mm_sub_epi32(T14B, O04B);
            const __m128i T2_26A = _mm_sub_epi32(T15A, O05A);       // E5
            const __m128i T2_26B = _mm_sub_epi32(T15B, O05B);
            const __m128i T2_25A = _mm_sub_epi32(T16A, O06A);       // E6
            const __m128i T2_25B = _mm_sub_epi32(T16B, O06B);
            const __m128i T2_24A = _mm_sub_epi32(T17A, O07A);       // E7
            const __m128i T2_24B = _mm_sub_epi32(T17B, O07B);
            const __m128i T2_23A = _mm_sub_epi32(T18A, O08A);       //
            const __m128i T2_23B = _mm_sub_epi32(T18B, O08B);
            const __m128i T2_22A = _mm_sub_epi32(T19A, O09A);       //
            const __m128i T2_22B = _mm_sub_epi32(T19B, O09B);
            const __m128i T2_21A = _mm_sub_epi32(T1AA, O10A);       //
            const __m128i T2_21B = _mm_sub_epi32(T1AB, O10B);
            const __m128i T2_20A = _mm_sub_epi32(T1BA, O11A);       //
            const __m128i T2_20B = _mm_sub_epi32(T1BB, O11B);
            const __m128i T2_19A = _mm_sub_epi32(T1CA, O12A);       //
            const __m128i T2_19B = _mm_sub_epi32(T1CB, O12B);
            const __m128i T2_18A = _mm_sub_epi32(T1DA, O13A);       //
            const __m128i T2_18B = _mm_sub_epi32(T1DB, O13B);
            const __m128i T2_17A = _mm_sub_epi32(T1EA, O14A);       //
            const __m128i T2_17B = _mm_sub_epi32(T1EB, O14B);
            const __m128i T2_16A = _mm_sub_epi32(T1FA, O15A);       //
            const __m128i T2_16B = _mm_sub_epi32(T1FB, O15B);

            const __m128i T3_00A = _mm_srai_epi32(T2_00A, nShift);  // [30 20 10 00]
            const __m128i T3_00B = _mm_srai_epi32(T2_00B, nShift);  // [70 60 50 40]
            const __m128i T3_01A = _mm_srai_epi32(T2_01A, nShift);  // [31 21 11 01]
            const __m128i T3_01B = _mm_srai_epi32(T2_01B, nShift);  // [71 61 51 41]
            const __m128i T3_02A = _mm_srai_epi32(T2_02A, nShift);  // [32 22 12 02]
            const __m128i T3_02B = _mm_srai_epi32(T2_02B, nShift);  // [72 62 52 42]
            const __m128i T3_03A = _mm_srai_epi32(T2_03A, nShift);  // [33 23 13 03]
            const __m128i T3_03B = _mm_srai_epi32(T2_03B, nShift);  // [73 63 53 43]
            const __m128i T3_04A = _mm_srai_epi32(T2_04A, nShift);  // [33 24 14 04]
            const __m128i T3_04B = _mm_srai_epi32(T2_04B, nShift);  // [74 64 54 44]
            const __m128i T3_05A = _mm_srai_epi32(T2_05A, nShift);  // [35 25 15 05]
            const __m128i T3_05B = _mm_srai_epi32(T2_05B, nShift);  // [75 65 55 45]
            const __m128i T3_06A = _mm_srai_epi32(T2_06A, nShift);  // [36 26 16 06]
            const __m128i T3_06B = _mm_srai_epi32(T2_06B, nShift);  // [76 66 56 46]
            const __m128i T3_07A = _mm_srai_epi32(T2_07A, nShift);  // [37 27 17 07]
            const __m128i T3_07B = _mm_srai_epi32(T2_07B, nShift);  // [77 67 57 47]
            const __m128i T3_08A = _mm_srai_epi32(T2_08A, nShift);  // [30 20 10 00] x8
            const __m128i T3_08B = _mm_srai_epi32(T2_08B, nShift);  // [70 60 50 40]
            const __m128i T3_09A = _mm_srai_epi32(T2_09A, nShift);  // [31 21 11 01] x9
            const __m128i T3_09B = _mm_srai_epi32(T2_09B, nShift);  // [71 61 51 41]
            const __m128i T3_10A = _mm_srai_epi32(T2_10A, nShift);  // [32 22 12 02] xA
            const __m128i T3_10B = _mm_srai_epi32(T2_10B, nShift);  // [72 62 52 42]
            const __m128i T3_11A = _mm_srai_epi32(T2_11A, nShift);  // [33 23 13 03] xB
            const __m128i T3_11B = _mm_srai_epi32(T2_11B, nShift);  // [73 63 53 43]
            const __m128i T3_12A = _mm_srai_epi32(T2_12A, nShift);  // [33 24 14 04] xC
            const __m128i T3_12B = _mm_srai_epi32(T2_12B, nShift);  // [74 64 54 44]
            const __m128i T3_13A = _mm_srai_epi32(T2_13A, nShift);  // [35 25 15 05] xD
            const __m128i T3_13B = _mm_srai_epi32(T2_13B, nShift);  // [75 65 55 45]
            const __m128i T3_14A = _mm_srai_epi32(T2_14A, nShift);  // [36 26 16 06] xE
            const __m128i T3_14B = _mm_srai_epi32(T2_14B, nShift);  // [76 66 56 46]
            const __m128i T3_15A = _mm_srai_epi32(T2_15A, nShift);  // [37 27 17 07] xF
            const __m128i T3_15B = _mm_srai_epi32(T2_15B, nShift);  // [77 67 57 47]

            const __m128i T3_16A = _mm_srai_epi32(T2_16A, nShift);  // [30 20 10 00]
            const __m128i T3_16B = _mm_srai_epi32(T2_16B, nShift);  // [70 60 50 40]
            const __m128i T3_17A = _mm_srai_epi32(T2_17A, nShift);  // [31 21 11 01]
            const __m128i T3_17B = _mm_srai_epi32(T2_17B, nShift);  // [71 61 51 41]
            const __m128i T3_18A = _mm_srai_epi32(T2_18A, nShift);  // [32 22 12 02]
            const __m128i T3_18B = _mm_srai_epi32(T2_18B, nShift);  // [72 62 52 42]
            const __m128i T3_19A = _mm_srai_epi32(T2_19A, nShift);  // [33 23 13 03]
            const __m128i T3_19B = _mm_srai_epi32(T2_19B, nShift);  // [73 63 53 43]
            const __m128i T3_20A = _mm_srai_epi32(T2_20A, nShift);  // [33 24 14 04]
            const __m128i T3_20B = _mm_srai_epi32(T2_20B, nShift);  // [74 64 54 44]
            const __m128i T3_21A = _mm_srai_epi32(T2_21A, nShift);  // [35 25 15 05]
            const __m128i T3_21B = _mm_srai_epi32(T2_21B, nShift);  // [75 65 55 45]
            const __m128i T3_22A = _mm_srai_epi32(T2_22A, nShift);  // [36 26 16 06]
            const __m128i T3_22B = _mm_srai_epi32(T2_22B, nShift);  // [76 66 56 46]
            const __m128i T3_23A = _mm_srai_epi32(T2_23A, nShift);  // [37 27 17 07]
            const __m128i T3_23B = _mm_srai_epi32(T2_23B, nShift);  // [77 67 57 47]
            const __m128i T3_24A = _mm_srai_epi32(T2_24A, nShift);  // [30 20 10 00] x8
            const __m128i T3_24B = _mm_srai_epi32(T2_24B, nShift);  // [70 60 50 40]
            const __m128i T3_25A = _mm_srai_epi32(T2_25A, nShift);  // [31 21 11 01] x9
            const __m128i T3_25B = _mm_srai_epi32(T2_25B, nShift);  // [71 61 51 41]
            const __m128i T3_26A = _mm_srai_epi32(T2_26A, nShift);  // [32 22 12 02] xA
            const __m128i T3_26B = _mm_srai_epi32(T2_26B, nShift);  // [72 62 52 42]
            const __m128i T3_27A = _mm_srai_epi32(T2_27A, nShift);  // [33 23 13 03] xB
            const __m128i T3_27B = _mm_srai_epi32(T2_27B, nShift);  // [73 63 53 43]
            const __m128i T3_28A = _mm_srai_epi32(T2_28A, nShift);  // [33 24 14 04] xC
            const __m128i T3_28B = _mm_srai_epi32(T2_28B, nShift);  // [74 64 54 44]
            const __m128i T3_29A = _mm_srai_epi32(T2_29A, nShift);  // [35 25 15 05] xD
            const __m128i T3_29B = _mm_srai_epi32(T2_29B, nShift);  // [75 65 55 45]
            const __m128i T3_30A = _mm_srai_epi32(T2_30A, nShift);  // [36 26 16 06] xE
            const __m128i T3_30B = _mm_srai_epi32(T2_30B, nShift);  // [76 66 56 46]
            const __m128i T3_31A = _mm_srai_epi32(T2_31A, nShift);  // [37 27 17 07] xF
            const __m128i T3_31B = _mm_srai_epi32(T2_31B, nShift);  // [77 67 57 47]

            res00[0] = _mm_packs_epi32(T3_00A, T3_00B);             // [70 60 50 40 30 20 10 00]
            res01[0] = _mm_packs_epi32(T3_01A, T3_01B);             // [71 61 51 41 31 21 11 01]
            res02[0] = _mm_packs_epi32(T3_02A, T3_02B);             // [72 62 52 42 32 22 12 02]
            res03[0] = _mm_packs_epi32(T3_03A, T3_03B);             // [73 63 53 43 33 23 13 03]
            res04[0] = _mm_packs_epi32(T3_04A, T3_04B);             // [74 64 54 44 34 24 14 04]
            res05[0] = _mm_packs_epi32(T3_05A, T3_05B);             // [75 65 55 45 35 25 15 05]
            res06[0] = _mm_packs_epi32(T3_06A, T3_06B);             // [76 66 56 46 36 26 16 06]
            res07[0] = _mm_packs_epi32(T3_07A, T3_07B);             // [77 67 57 47 37 27 17 07]
            res00[1] = _mm_packs_epi32(T3_08A, T3_08B);             // [A0 ... 80]
            res01[1] = _mm_packs_epi32(T3_09A, T3_09B);             // [A1 ... 81]
            res02[1] = _mm_packs_epi32(T3_10A, T3_10B);             // [A2 ... 82]
            res03[1] = _mm_packs_epi32(T3_11A, T3_11B);             // [A3 ... 83]
            res04[1] = _mm_packs_epi32(T3_12A, T3_12B);             // [A4 ... 84]
            res05[1] = _mm_packs_epi32(T3_13A, T3_13B);             // [A5 ... 85]
            res06[1] = _mm_packs_epi32(T3_14A, T3_14B);             // [A6 ... 86]
            res07[1] = _mm_packs_epi32(T3_15A, T3_15B);             // [A7 ... 87]
            res00[2] = _mm_packs_epi32(T3_16A, T3_16B);
            res01[2] = _mm_packs_epi32(T3_17A, T3_17B);
            res02[2] = _mm_packs_epi32(T3_18A, T3_18B);
            res03[2] = _mm_packs_epi32(T3_19A, T3_19B);
            res04[2] = _mm_packs_epi32(T3_20A, T3_20B);
            res05[2] = _mm_packs_epi32(T3_21A, T3_21B);
            res06[2] = _mm_packs_epi32(T3_22A, T3_22B);
            res07[2] = _mm_packs_epi32(T3_23A, T3_23B);
            res00[3] = _mm_packs_epi32(T3_24A, T3_24B);
            res01[3] = _mm_packs_epi32(T3_25A, T3_25B);
            res02[3] = _mm_packs_epi32(T3_26A, T3_26B);
            res03[3] = _mm_packs_epi32(T3_27A, T3_27B);
            res04[3] = _mm_packs_epi32(T3_28A, T3_28B);
            res05[3] = _mm_packs_epi32(T3_29A, T3_29B);
            res06[3] = _mm_packs_epi32(T3_30A, T3_30B);
            res07[3] = _mm_packs_epi32(T3_31A, T3_31B);
        }

    }

#define TRANSPOSE_8x8_16BIT(I0, I1, I2, I3, I4, I5, I6, I7, O0, O1, O2, O3, O4, O5, O6, O7) \
    tr0_0 = _mm_unpacklo_epi16(I0, I1); \
    tr0_1 = _mm_unpacklo_epi16(I2, I3); \
    tr0_2 = _mm_unpackhi_epi16(I0, I1); \
    tr0_3 = _mm_unpackhi_epi16(I2, I3); \
    tr0_4 = _mm_unpacklo_epi16(I4, I5); \
    tr0_5 = _mm_unpacklo_epi16(I6, I7); \
    tr0_6 = _mm_unpackhi_epi16(I4, I5); \
    tr0_7 = _mm_unpackhi_epi16(I6, I7); \
    tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1); \
    tr1_1 = _mm_unpacklo_epi32(tr0_2, tr0_3); \
    tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1); \
    tr1_3 = _mm_unpackhi_epi32(tr0_2, tr0_3); \
    tr1_4 = _mm_unpacklo_epi32(tr0_4, tr0_5); \
    tr1_5 = _mm_unpacklo_epi32(tr0_6, tr0_7); \
    tr1_6 = _mm_unpackhi_epi32(tr0_4, tr0_5); \
    tr1_7 = _mm_unpackhi_epi32(tr0_6, tr0_7); \
    O0 = _mm_unpacklo_epi64(tr1_0, tr1_4); \
    O1 = _mm_unpackhi_epi64(tr1_0, tr1_4); \
    O2 = _mm_unpacklo_epi64(tr1_2, tr1_6); \
    O3 = _mm_unpackhi_epi64(tr1_2, tr1_6); \
    O4 = _mm_unpacklo_epi64(tr1_1, tr1_5); \
    O5 = _mm_unpackhi_epi64(tr1_1, tr1_5); \
    O6 = _mm_unpacklo_epi64(tr1_3, tr1_7); \
    O7 = _mm_unpackhi_epi64(tr1_3, tr1_7);

    //clip
    {
        __m128i max_val = _mm_set1_epi16((1 << (clip_depth2 - 1)) - 1);
        __m128i min_val = _mm_set1_epi16(-(1 << (clip_depth2 - 1)));

        c32_rnd = _mm_set1_epi32(1 << (shift2 - 1));    // add2
        nShift = shift2;

        for (pass = 0; pass < 4; pass++) {
            __m128i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
            __m128i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;
            __m128i m128Tmp0, m128Tmp1, m128Tmp2, m128Tmp3, E0h, E1h, E2h, E3h, E0l, E1l, E2l, E3l, O0h, O1h, O2h, O3h, O0l, O1l, O2l, O3l, EE0l, EE1l, E00l, E01l, EE0h, EE1h, E00h, E01h;

            TRANSPOSE_8x8_16BIT(res00[pass], res01[pass], res02[pass], res03[pass], res04[pass], res05[pass], res06[pass], res07[pass], in00, in01, in02, in03, in04, in05, in06, in07)

            m128Tmp0 = _mm_unpacklo_epi16(in01, in03);
            E1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8[0])));
            m128Tmp1 = _mm_unpackhi_epi16(in01, in03);
            E1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8[0])));

            m128Tmp2 = _mm_unpacklo_epi16(in05, in07);
            E2l = _mm_madd_epi16(m128Tmp2, _mm_load_si128((__m128i*)(tab_idct_8x8[1])));
            m128Tmp3 = _mm_unpackhi_epi16(in05, in07);
            E2h = _mm_madd_epi16(m128Tmp3, _mm_load_si128((__m128i*)(tab_idct_8x8[1])));
            O0l = _mm_add_epi32(E1l, E2l);
            O0h = _mm_add_epi32(E1h, E2h);

            E1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8[2])));
            E1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8[2])));
            E2l = _mm_madd_epi16(m128Tmp2, _mm_load_si128((__m128i*)(tab_idct_8x8[3])));
            E2h = _mm_madd_epi16(m128Tmp3, _mm_load_si128((__m128i*)(tab_idct_8x8[3])));

            O1l = _mm_add_epi32(E1l, E2l);
            O1h = _mm_add_epi32(E1h, E2h);

            E1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8[4])));
            E1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8[4])));
            E2l = _mm_madd_epi16(m128Tmp2, _mm_load_si128((__m128i*)(tab_idct_8x8[5])));
            E2h = _mm_madd_epi16(m128Tmp3, _mm_load_si128((__m128i*)(tab_idct_8x8[5])));
            O2l = _mm_add_epi32(E1l, E2l);
            O2h = _mm_add_epi32(E1h, E2h);

            E1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8[6])));
            E1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8[6])));
            E2l = _mm_madd_epi16(m128Tmp2, _mm_load_si128((__m128i*)(tab_idct_8x8[7])));
            E2h = _mm_madd_epi16(m128Tmp3, _mm_load_si128((__m128i*)(tab_idct_8x8[7])));
            O3h = _mm_add_epi32(E1h, E2h);
            O3l = _mm_add_epi32(E1l, E2l);

            /*    -------     */
            m128Tmp0 = _mm_unpacklo_epi16(in00, in04);
            EE0l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8[8])));
            m128Tmp1 = _mm_unpackhi_epi16(in00, in04);
            EE0h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8[8])));

            EE1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8[9])));
            EE1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8[9])));

            /*    -------     */
            m128Tmp0 = _mm_unpacklo_epi16(in02, in06);
            E00l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8[10])));
            m128Tmp1 = _mm_unpackhi_epi16(in02, in06);
            E00h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8[10])));
            E01l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8[11])));
            E01h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8[11])));
            E0l = _mm_add_epi32(EE0l, E00l);
            E0l = _mm_add_epi32(E0l, c32_rnd);
            E0h = _mm_add_epi32(EE0h, E00h);
            E0h = _mm_add_epi32(E0h, c32_rnd);
            E3l = _mm_sub_epi32(EE0l, E00l);
            E3l = _mm_add_epi32(E3l, c32_rnd);
            E3h = _mm_sub_epi32(EE0h, E00h);
            E3h = _mm_add_epi32(E3h, c32_rnd);

            E1l = _mm_add_epi32(EE1l, E01l);
            E1l = _mm_add_epi32(E1l, c32_rnd);
            E1h = _mm_add_epi32(EE1h, E01h);
            E1h = _mm_add_epi32(E1h, c32_rnd);
            E2l = _mm_sub_epi32(EE1l, E01l);
            E2l = _mm_add_epi32(E2l, c32_rnd);
            E2h = _mm_sub_epi32(EE1h, E01h);
            E2h = _mm_add_epi32(E2h, c32_rnd);
            in00 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E0l, O0l), nShift), _mm_srai_epi32(_mm_add_epi32(E0h, O0h), nShift));     // 首次反变换移位数
            in07 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E0l, O0l), nShift), _mm_srai_epi32(_mm_sub_epi32(E0h, O0h), nShift));
            in01 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E1l, O1l), nShift), _mm_srai_epi32(_mm_add_epi32(E1h, O1h), nShift));
            in06 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E1l, O1l), nShift), _mm_srai_epi32(_mm_sub_epi32(E1h, O1h), nShift));
            in02 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E2l, O2l), nShift), _mm_srai_epi32(_mm_add_epi32(E2h, O2h), nShift));
            in05 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E2l, O2l), nShift), _mm_srai_epi32(_mm_sub_epi32(E2h, O2h), nShift));
            in03 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E3l, O3l), nShift), _mm_srai_epi32(_mm_add_epi32(E3h, O3h), nShift));
            in04 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E3l, O3l), nShift), _mm_srai_epi32(_mm_sub_epi32(E3h, O3h), nShift));

            /*  Invers matrix   */
            E0l = _mm_unpacklo_epi16(in00, in04);
            E1l = _mm_unpacklo_epi16(in01, in05);
            E2l = _mm_unpacklo_epi16(in02, in06);
            E3l = _mm_unpacklo_epi16(in03, in07);
            O0l = _mm_unpackhi_epi16(in00, in04);
            O1l = _mm_unpackhi_epi16(in01, in05);
            O2l = _mm_unpackhi_epi16(in02, in06);
            O3l = _mm_unpackhi_epi16(in03, in07);

            m128Tmp0 = _mm_unpacklo_epi16(E0l, E2l);
            m128Tmp1 = _mm_unpacklo_epi16(E1l, E3l);
            in00 = _mm_unpacklo_epi16(m128Tmp0, m128Tmp1);
            in00 = _mm_min_epi16(in00, max_val);
            in00 = _mm_max_epi16(in00, min_val);
            _mm_store_si128((__m128i*)&dst[pass * 8 * i_dst + 0 * 8], in00);
            in01 = _mm_unpackhi_epi16(m128Tmp0, m128Tmp1);
            in01 = _mm_min_epi16(in01, max_val);
            in01 = _mm_max_epi16(in01, min_val);
            _mm_store_si128((__m128i*)&dst[pass * 8 * i_dst + 1 * 8], in01);

            m128Tmp2 = _mm_unpackhi_epi16(E0l, E2l);
            m128Tmp3 = _mm_unpackhi_epi16(E1l, E3l);
            in02 = _mm_unpacklo_epi16(m128Tmp2, m128Tmp3);
            in02 = _mm_min_epi16(in02, max_val);
            in02 = _mm_max_epi16(in02, min_val);
            _mm_store_si128((__m128i*)&dst[pass * 8 * i_dst + 2 * 8], in02);
            in03 = _mm_unpackhi_epi16(m128Tmp2, m128Tmp3);
            in03 = _mm_min_epi16(in03, max_val);
            in03 = _mm_max_epi16(in03, min_val);
            _mm_store_si128((__m128i*)&dst[pass * 8 * i_dst + 3 * 8], in03);

            m128Tmp0 = _mm_unpacklo_epi16(O0l, O2l);
            m128Tmp1 = _mm_unpacklo_epi16(O1l, O3l);
            in04 = _mm_unpacklo_epi16(m128Tmp0, m128Tmp1);
            in04 = _mm_min_epi16(in04, max_val);
            in04 = _mm_max_epi16(in04, min_val);
            _mm_store_si128((__m128i*)&dst[pass * 8 * i_dst + 4 * 8], in04);
            in05 = _mm_unpackhi_epi16(m128Tmp0, m128Tmp1);
            in05 = _mm_min_epi16(in05, max_val);
            in05 = _mm_max_epi16(in05, min_val);
            _mm_store_si128((__m128i*)&dst[pass * 8 * i_dst + 5 * 8], in05);

            m128Tmp2 = _mm_unpackhi_epi16(O0l, O2l);
            m128Tmp3 = _mm_unpackhi_epi16(O1l, O3l);
            in06 = _mm_unpacklo_epi16(m128Tmp2, m128Tmp3);
            in06 = _mm_min_epi16(in06, max_val);
            in06 = _mm_max_epi16(in06, min_val);
            _mm_store_si128((__m128i*)&dst[pass * 8 * i_dst + 6 * 8], in06);
            in07 = _mm_unpackhi_epi16(m128Tmp2, m128Tmp3);
            in07 = _mm_min_epi16(in07, max_val);
            in07 = _mm_max_epi16(in07, min_val);
            _mm_store_si128((__m128i*)&dst[pass * 8 * i_dst + 7 * 8], in07);
        }
    }
#undef TRANSPOSE_8x8_16BIT
}


/* ---------------------------------------------------------------------------
 */
static void inv_2nd_trans_hor_sse128(coeff_t *coeff, int i_coeff, int i_shift, const int16_t *tc)
{
    int rnd_factor = 1 << (i_shift - 1);
    int j;

    __m128i factor = _mm_set1_epi32(rnd_factor);
    __m128i tmpZero = _mm_setzero_si128();                      // 0 elements

    // load tc data, a matrix of 4x4
    __m128i tmpLoad0 = _mm_loadu_si128((__m128i*)&tc[0 * SEC_TR_SIZE + 0]);  // tc[0][] & tc[1][]
    __m128i tmpLoad1 = _mm_loadu_si128((__m128i*)&tc[2 * SEC_TR_SIZE + 0]);  // tc[2][] & tc[3][]
    __m128i tmpCoef0 = _mm_unpacklo_epi16(tmpLoad0, tmpZero);   // tc[0][]
    __m128i tmpCoef1 = _mm_unpackhi_epi16(tmpLoad0, tmpZero);   // tc[1][]
    __m128i tmpCoef2 = _mm_unpacklo_epi16(tmpLoad1, tmpZero);   // tc[2][]
    __m128i tmpCoef3 = _mm_unpackhi_epi16(tmpLoad1, tmpZero);   // tc[3][]

    for (j = 0; j < 4; j++) {
        // multiple & add
        __m128i tmpProduct0 = _mm_madd_epi16(tmpCoef0, _mm_set1_epi32(coeff[0]));
        __m128i tmpProduct1 = _mm_madd_epi16(tmpCoef1, _mm_set1_epi32(coeff[1]));
        __m128i tmpProduct2 = _mm_madd_epi16(tmpCoef2, _mm_set1_epi32(coeff[2]));
        __m128i tmpProduct3 = _mm_madd_epi16(tmpCoef3, _mm_set1_epi32(coeff[3]));

        // add operation
        __m128i tmpDst0 = _mm_add_epi32(_mm_add_epi32(tmpProduct0, tmpProduct1), _mm_add_epi32(tmpProduct2, tmpProduct3));

        // shift operation
        tmpDst0 = _mm_srai_epi32(_mm_add_epi32(tmpDst0, factor), i_shift);
        // clip3 operation
        tmpDst0 = _mm_packs_epi32(tmpDst0, tmpZero);    // only low 64bits (4xSHORT) are valid!

        _mm_storel_epi64((__m128i*)coeff, tmpDst0); // store from &coeff[0]
        coeff += i_coeff;
    }
}

/* ---------------------------------------------------------------------------
 */
static void inv_2nd_trans_ver_sse128(coeff_t *coeff, int i_coeff, int i_shift, const int16_t *tc)
{
    const int rnd_factor = 1 << (i_shift - 1);
    __m128i factor = _mm_set1_epi32(rnd_factor);
    __m128i tmpZero = _mm_setzero_si128();                // 0 elements

    // load coeff data
    __m128i tmpLoad0 = _mm_loadu_si128((__m128i*)&coeff[0        ]);
    __m128i tmpLoad1 = _mm_loadu_si128((__m128i*)&coeff[1 * i_coeff]);
    __m128i tmpLoad2 = _mm_loadu_si128((__m128i*)&coeff[2 * i_coeff]);
    __m128i tmpLoad3 = _mm_loadu_si128((__m128i*)&coeff[3 * i_coeff]);
    __m128i tmpSrc0 = _mm_unpacklo_epi16(tmpLoad0, tmpZero);    // tmpSrc[0][]
    __m128i tmpSrc1 = _mm_unpacklo_epi16(tmpLoad1, tmpZero);    // tmpSrc[1][]
    __m128i tmpSrc2 = _mm_unpacklo_epi16(tmpLoad2, tmpZero);    // tmpSrc[2][]
    __m128i tmpSrc3 = _mm_unpacklo_epi16(tmpLoad3, tmpZero);    // tmpSrc[3][]
    int i;

    for (i = 0; i < 4; i++) {
        // multiple & add
        __m128i tmpProduct0 = _mm_madd_epi16(_mm_set1_epi32(tc[0 * SEC_TR_SIZE + i]), tmpSrc0);
        __m128i tmpProduct1 = _mm_madd_epi16(_mm_set1_epi32(tc[1 * SEC_TR_SIZE + i]), tmpSrc1);
        __m128i tmpProduct2 = _mm_madd_epi16(_mm_set1_epi32(tc[2 * SEC_TR_SIZE + i]), tmpSrc2);
        __m128i tmpProduct3 = _mm_madd_epi16(_mm_set1_epi32(tc[3 * SEC_TR_SIZE + i]), tmpSrc3);
        // add operation
        __m128i tmpDst0 = _mm_add_epi32(_mm_add_epi32(tmpProduct0, tmpProduct1), _mm_add_epi32(tmpProduct2, tmpProduct3));
        // shift operation
        tmpDst0 = _mm_srai_epi32(_mm_add_epi32(tmpDst0, factor), i_shift);
        // clip3 operation
        tmpDst0 = _mm_packs_epi32(tmpDst0, tmpZero);        // only low 64bits (4xSHORT) are valid!

        // store from &coeff[0]
        _mm_storel_epi64((__m128i*)&coeff[0 * i_coeff + 0], tmpDst0);
        coeff += i_coeff;
    }
}

/* ---------------------------------------------------------------------------
*/
void inv_transform_2nd_sse128(coeff_t *coeff, int i_coeff, int i_mode, int b_top, int b_left)
{
    int vt = (i_mode >=  0 && i_mode <= 23);
    int ht = (i_mode >= 13 && i_mode <= 32) || (i_mode >= 0 && i_mode <= 2);

    if (ht && b_left) {
        inv_2nd_trans_hor_sse128(coeff, i_coeff, 7, g_2T);
    }
    if (vt && b_top) {
        inv_2nd_trans_ver_sse128(coeff, i_coeff, 7, g_2T);
    }
}

/* ---------------------------------------------------------------------------
 */
void inv_transform_4x4_2nd_sse128(coeff_t *coeff, int i_coeff)
{
    const int shift1 = 5;
    const int shift2 = 20 - g_bit_depth + 2;
    const int clip_depth2 = g_bit_depth + 1;

    /*---vertical transform first---*/
    __m128i factor = _mm_set1_epi32(1 << (shift1 - 1));         // add1
    __m128i tmpZero = _mm_setzero_si128();                      // 0 elements

    // load coeff data
    __m128i tmpLoad0 = _mm_loadu_si128((__m128i*)&coeff[0          ]);
    __m128i tmpLoad1 = _mm_loadu_si128((__m128i*)&coeff[1 * i_coeff]);
    __m128i tmpLoad2 = _mm_loadu_si128((__m128i*)&coeff[2 * i_coeff]);
    __m128i tmpLoad3 = _mm_loadu_si128((__m128i*)&coeff[3 * i_coeff]);
    __m128i tmpSrc0 = _mm_unpacklo_epi16(tmpLoad0, tmpZero);    // tmpSrc[0][]
    __m128i tmpSrc1 = _mm_unpacklo_epi16(tmpLoad1, tmpZero);    // tmpSrc[1][]
    __m128i tmpSrc2 = _mm_unpacklo_epi16(tmpLoad2, tmpZero);    // tmpSrc[2][]
    __m128i tmpSrc3 = _mm_unpacklo_epi16(tmpLoad3, tmpZero);    // tmpSrc[3][]
    int i;

    for (i = 0; i < 4; i++) {
        // multiple & add
        __m128i tmpProduct0 = _mm_madd_epi16(_mm_set1_epi32(g_2T_C[0 * SEC_TR_SIZE + i]), tmpSrc0);
        __m128i tmpProduct1 = _mm_madd_epi16(_mm_set1_epi32(g_2T_C[1 * SEC_TR_SIZE + i]), tmpSrc1);
        __m128i tmpProduct2 = _mm_madd_epi16(_mm_set1_epi32(g_2T_C[2 * SEC_TR_SIZE + i]), tmpSrc2);
        __m128i tmpProduct3 = _mm_madd_epi16(_mm_set1_epi32(g_2T_C[3 * SEC_TR_SIZE + i]), tmpSrc3);
        // add operation
        __m128i tmpDst0 = _mm_add_epi32(_mm_add_epi32(tmpProduct0, tmpProduct1), _mm_add_epi32(tmpProduct2, tmpProduct3));
        // shift operation
        tmpDst0 = _mm_srai_epi32(_mm_add_epi32(tmpDst0, factor), shift1);
        // clip3 operation
        tmpDst0 = _mm_packs_epi32(tmpDst0, tmpZero);        // only low 64bits (4xSHORT) are valid!

        _mm_storel_epi64((__m128i*)&coeff[i * i_coeff + 0], tmpDst0); // store from &coeff[0]
    }

    /*---hor transform---*/
    factor = _mm_set1_epi32(1 << (shift2 - 1));
    const __m128i vmax_val = _mm_set1_epi32((1 << (clip_depth2 - 1)) - 1);
    const __m128i vmin_val = _mm_set1_epi32(-(1 << (clip_depth2 - 1)));

    //load coef data, a matrix of 4x4
    tmpLoad0 = _mm_loadu_si128((__m128i*)&g_2T_C[0 * SEC_TR_SIZE + 0]);  // coef[0][] & coef[1][]
    tmpLoad1 = _mm_loadu_si128((__m128i*)&g_2T_C[2 * SEC_TR_SIZE + 0]);  // coef[2][] & coef[3][]
    const __m128i tmpCoef0 = _mm_unpacklo_epi16(tmpLoad0, tmpZero);   // coef[0][]
    const __m128i tmpCoef1 = _mm_unpackhi_epi16(tmpLoad0, tmpZero);   // coef[1][]
    const __m128i tmpCoef2 = _mm_unpacklo_epi16(tmpLoad1, tmpZero);   // coef[2][]
    const __m128i tmpCoef3 = _mm_unpackhi_epi16(tmpLoad1, tmpZero);   // coef[3][]

    for (i = 0; i < 4; i++) {
        // multiple & add
        __m128i tmpProduct0 = _mm_madd_epi16(tmpCoef0, _mm_set1_epi32(coeff[0]));
        __m128i tmpProduct1 = _mm_madd_epi16(tmpCoef1, _mm_set1_epi32(coeff[1]));
        __m128i tmpProduct2 = _mm_madd_epi16(tmpCoef2, _mm_set1_epi32(coeff[2]));
        __m128i tmpProduct3 = _mm_madd_epi16(tmpCoef3, _mm_set1_epi32(coeff[3]));
        // add operation
        __m128i tmpDst0 = _mm_add_epi32(_mm_add_epi32(tmpProduct0, tmpProduct1), _mm_add_epi32(tmpProduct2, tmpProduct3));
        // shift operation
        tmpDst0 = _mm_srai_epi32(_mm_add_epi32(tmpDst0, factor), shift2);
        // clip3 operation
        tmpDst0 = _mm_max_epi32(_mm_min_epi32(tmpDst0, vmax_val), vmin_val);

        tmpDst0 = _mm_packs_epi32(tmpDst0, tmpZero);        // only low 64bits (4xSHORT) are valid!
        _mm_storel_epi64((__m128i*)coeff, tmpDst0); // store from &coeff[0]
        coeff += i_coeff;
    }
}


// transpose 8x8 & transpose 16x16
#define TRANSPOSE_8x8_16BIT(I0, I1, I2, I3, I4, I5, I6, I7, O0, O1, O2, O3, O4, O5, O6, O7) \
    tr0_0 = _mm_unpacklo_epi16(I0, I1); \
    tr0_1 = _mm_unpacklo_epi16(I2, I3); \
    tr0_2 = _mm_unpackhi_epi16(I0, I1); \
    tr0_3 = _mm_unpackhi_epi16(I2, I3); \
    tr0_4 = _mm_unpacklo_epi16(I4, I5); \
    tr0_5 = _mm_unpacklo_epi16(I6, I7); \
    tr0_6 = _mm_unpackhi_epi16(I4, I5); \
    tr0_7 = _mm_unpackhi_epi16(I6, I7); \
    tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1); \
    tr1_1 = _mm_unpacklo_epi32(tr0_2, tr0_3); \
    tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1); \
    tr1_3 = _mm_unpackhi_epi32(tr0_2, tr0_3); \
    tr1_4 = _mm_unpacklo_epi32(tr0_4, tr0_5); \
    tr1_5 = _mm_unpacklo_epi32(tr0_6, tr0_7); \
    tr1_6 = _mm_unpackhi_epi32(tr0_4, tr0_5); \
    tr1_7 = _mm_unpackhi_epi32(tr0_6, tr0_7); \
    O0 = _mm_unpacklo_epi64(tr1_0, tr1_4); \
    O1 = _mm_unpackhi_epi64(tr1_0, tr1_4); \
    O2 = _mm_unpacklo_epi64(tr1_2, tr1_6); \
    O3 = _mm_unpackhi_epi64(tr1_2, tr1_6); \
    O4 = _mm_unpacklo_epi64(tr1_1, tr1_5); \
    O5 = _mm_unpackhi_epi64(tr1_1, tr1_5); \
    O6 = _mm_unpacklo_epi64(tr1_3, tr1_7); \
    O7 = _mm_unpackhi_epi64(tr1_3, tr1_7); \
 
#define TRANSPOSE_16x16_16BIT(A0_0, A1_0, A2_0, A3_0, A4_0, A5_0, A6_0, A7_0, A8_0, A9_0, A10_0, A11_0, A12_0, A13_0, A14_0, A15_0, A0_1, A1_1, A2_1, A3_1, A4_1, A5_1, A6_1, A7_1, A8_1, A9_1, A10_1, A11_1, A12_1, A13_1, A14_1, A15_1, B0_0, B1_0, B2_0, B3_0, B4_0, B5_0, B6_0, B7_0, B8_0, B9_0, B10_0, B11_0, B12_0, B13_0, B14_0, B15_0, B0_1, B1_1, B2_1, B3_1, B4_1, B5_1, B6_1, B7_1, B8_1, B9_1, B10_1, B11_1, B12_1, B13_1, B14_1, B15_1) \
    TRANSPOSE_8x8_16BIT(A0_0, A1_0, A2_0, A3_0, A4_0, A5_0, A6_0, A7_0, B0_0, B1_0, B2_0, B3_0, B4_0, B5_0, B6_0, B7_0); \
    TRANSPOSE_8x8_16BIT(A8_0, A9_0, A10_0, A11_0, A12_0, A13_0, A14_0, A15_0, B0_1, B1_1, B2_1, B3_1, B4_1, B5_1, B6_1, B7_1); \
    TRANSPOSE_8x8_16BIT(A0_1, A1_1, A2_1, A3_1, A4_1, A5_1, A6_1, A7_1, B8_0, B9_0, B10_0, B11_0, B12_0, B13_0, B14_0, B15_0); \
    TRANSPOSE_8x8_16BIT(A8_1, A9_1, A10_1, A11_1, A12_1, A13_1, A14_1, A15_1, B8_1, B9_1, B10_1, B11_1, B12_1, B13_1, B14_1, B15_1); \
 

/* ---------------------------------------------------------------------------
 */
static void inv_wavelet_64x64_sse128(coeff_t *coeff)
{
    int i;
    //按行 64*64
    __m128i T00[8], T01[8], T02[8], T03[8], T04[8], T05[8], T06[8], T07[8], T08[8], T09[8], T10[8], T11[8], T12[8], T13[8], T14[8], T15[8], T16[8], T17[8], T18[8], T19[8], T20[8], T21[8], T22[8], T23[8], T24[8], T25[8], T26[8], T27[8], T28[8], T29[8], T30[8], T31[8], T32[8], T33[8], T34[8], T35[8], T36[8], T37[8], T38[8], T39[8], T40[8], T41[8], T42[8], T43[8], T44[8], T45[8], T46[8], T47[8], T48[8], T49[8], T50[8], T51[8], T52[8], T53[8], T54[8], T55[8], T56[8], T57[8], T58[8], T59[8], T60[8], T61[8], T62[8], T63[8];

    //按列 16*64
    __m128i V00[8], V01[8], V02[8], V03[8], V04[8], V05[8], V06[8], V07[8], V08[8], V09[8], V10[8], V11[8], V12[8], V13[8], V14[8], V15[8], V16[8], V17[8], V18[8], V19[8], V20[8], V21[8], V22[8], V23[8], V24[8], V25[8], V26[8], V27[8], V28[8], V29[8], V30[8], V31[8], V32[8], V33[8], V34[8], V35[8], V36[8], V37[8], V38[8], V39[8], V40[8], V41[8], V42[8], V43[8], V44[8], V45[8], V46[8], V47[8], V48[8], V49[8], V50[8], V51[8], V52[8], V53[8], V54[8], V55[8], V56[8], V57[8], V58[8], V59[8], V60[8], V61[8], V62[8], V63[8];

    __m128i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
    __m128i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;

    /*--vertical transform--*/
    //32*32, LOAD AND SHIFT
    for (i = 0; i < 4; i++) {
        T00[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 *  0]), 1);
        T01[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 *  1]), 1);
        T02[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 *  2]), 1);
        T03[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 *  3]), 1);
        T04[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 *  4]), 1);
        T05[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 *  5]), 1);
        T06[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 *  6]), 1);
        T07[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 *  7]), 1);

        T08[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 *  8]), 1);
        T09[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 *  9]), 1);
        T10[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 10]), 1);
        T11[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 11]), 1);
        T12[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 12]), 1);
        T13[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 13]), 1);
        T14[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 14]), 1);
        T15[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 15]), 1);

        T16[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 16]), 1);
        T17[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 17]), 1);
        T18[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 18]), 1);
        T19[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 19]), 1);
        T20[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 20]), 1);
        T21[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 21]), 1);
        T22[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 22]), 1);
        T23[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 23]), 1);

        T24[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 24]), 1);
        T25[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 25]), 1);
        T26[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 26]), 1);
        T27[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 27]), 1);
        T28[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 28]), 1);
        T29[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 29]), 1);
        T30[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 30]), 1);
        T31[i] = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * i + 32 * 31]), 1);
    }

    //filter (odd pixel/row)
    for (i = 0; i < 4; i++) {
        T32[i] = _mm_srai_epi16(_mm_add_epi16(T00[i], T01[i]), 1);
        T33[i] = _mm_srai_epi16(_mm_add_epi16(T01[i], T02[i]), 1);
        T34[i] = _mm_srai_epi16(_mm_add_epi16(T02[i], T03[i]), 1);
        T35[i] = _mm_srai_epi16(_mm_add_epi16(T03[i], T04[i]), 1);
        T36[i] = _mm_srai_epi16(_mm_add_epi16(T04[i], T05[i]), 1);
        T37[i] = _mm_srai_epi16(_mm_add_epi16(T05[i], T06[i]), 1);
        T38[i] = _mm_srai_epi16(_mm_add_epi16(T06[i], T07[i]), 1);
        T39[i] = _mm_srai_epi16(_mm_add_epi16(T07[i], T08[i]), 1);

        T40[i] = _mm_srai_epi16(_mm_add_epi16(T08[i], T09[i]), 1);
        T41[i] = _mm_srai_epi16(_mm_add_epi16(T09[i], T10[i]), 1);
        T42[i] = _mm_srai_epi16(_mm_add_epi16(T10[i], T11[i]), 1);
        T43[i] = _mm_srai_epi16(_mm_add_epi16(T11[i], T12[i]), 1);
        T44[i] = _mm_srai_epi16(_mm_add_epi16(T12[i], T13[i]), 1);
        T45[i] = _mm_srai_epi16(_mm_add_epi16(T13[i], T14[i]), 1);
        T46[i] = _mm_srai_epi16(_mm_add_epi16(T14[i], T15[i]), 1);
        T47[i] = _mm_srai_epi16(_mm_add_epi16(T15[i], T16[i]), 1);

        T48[i] = _mm_srai_epi16(_mm_add_epi16(T16[i], T17[i]), 1);
        T49[i] = _mm_srai_epi16(_mm_add_epi16(T17[i], T18[i]), 1);
        T50[i] = _mm_srai_epi16(_mm_add_epi16(T18[i], T19[i]), 1);
        T51[i] = _mm_srai_epi16(_mm_add_epi16(T19[i], T20[i]), 1);
        T52[i] = _mm_srai_epi16(_mm_add_epi16(T20[i], T21[i]), 1);
        T53[i] = _mm_srai_epi16(_mm_add_epi16(T21[i], T22[i]), 1);
        T54[i] = _mm_srai_epi16(_mm_add_epi16(T22[i], T23[i]), 1);
        T55[i] = _mm_srai_epi16(_mm_add_epi16(T23[i], T24[i]), 1);

        T56[i] = _mm_srai_epi16(_mm_add_epi16(T24[i], T25[i]), 1);
        T57[i] = _mm_srai_epi16(_mm_add_epi16(T25[i], T26[i]), 1);
        T58[i] = _mm_srai_epi16(_mm_add_epi16(T26[i], T27[i]), 1);
        T59[i] = _mm_srai_epi16(_mm_add_epi16(T27[i], T28[i]), 1);
        T60[i] = _mm_srai_epi16(_mm_add_epi16(T28[i], T29[i]), 1);
        T61[i] = _mm_srai_epi16(_mm_add_epi16(T29[i], T30[i]), 1);
        T62[i] = _mm_srai_epi16(_mm_add_epi16(T30[i], T31[i]), 1);
        T63[i] = _mm_srai_epi16(_mm_add_epi16(T31[i], T31[i]), 1);
    }

    /*--transposition--*/
    //32x64 -> 64x32
    TRANSPOSE_16x16_16BIT(
        T00[0], T32[0], T01[0], T33[0], T02[0], T34[0], T03[0], T35[0], T04[0], T36[0], T05[0], T37[0], T06[0], T38[0], T07[0], T39[0], T00[1], T32[1], T01[1], T33[1], T02[1], T34[1], T03[1], T35[1], T04[1], T36[1], T05[1], T37[1], T06[1], T38[1], T07[1], T39[1],
        V00[0], V01[0], V02[0], V03[0], V04[0], V05[0], V06[0], V07[0], V08[0], V09[0], V10[0], V11[0], V12[0], V13[0], V14[0], V15[0], V00[1], V01[1], V02[1], V03[1], V04[1], V05[1], V06[1], V07[1], V08[1], V09[1], V10[1], V11[1], V12[1], V13[1], V14[1], V15[1]);
    TRANSPOSE_16x16_16BIT(
        T00[2], T32[2], T01[2], T33[2], T02[2], T34[2], T03[2], T35[2], T04[2], T36[2], T05[2], T37[2], T06[2], T38[2], T07[2], T39[2], T00[3], T32[3], T01[3], T33[3], T02[3], T34[3], T03[3], T35[3], T04[3], T36[3], T05[3], T37[3], T06[3], T38[3], T07[3], T39[3],
        V16[0], V17[0], V18[0], V19[0], V20[0], V21[0], V22[0], V23[0], V24[0], V25[0], V26[0], V27[0], V28[0], V29[0], V30[0], V31[0], V16[1], V17[1], V18[1], V19[1], V20[1], V21[1], V22[1], V23[1], V24[1], V25[1], V26[1], V27[1], V28[1], V29[1], V30[1], V31[1]);

    TRANSPOSE_16x16_16BIT(
        T08[0], T40[0], T09[0], T41[0], T10[0], T42[0], T11[0], T43[0], T12[0], T44[0], T13[0], T45[0], T14[0], T46[0], T15[0], T47[0], T08[1], T40[1], T09[1], T41[1], T10[1], T42[1], T11[1], T43[1], T12[1], T44[1], T13[1], T45[1], T14[1], T46[1], T15[1], T47[1],
        V00[2], V01[2], V02[2], V03[2], V04[2], V05[2], V06[2], V07[2], V08[2], V09[2], V10[2], V11[2], V12[2], V13[2], V14[2], V15[2], V00[3], V01[3], V02[3], V03[3], V04[3], V05[3], V06[3], V07[3], V08[3], V09[3], V10[3], V11[3], V12[3], V13[3], V14[3], V15[3]);
    TRANSPOSE_16x16_16BIT(
        T08[2], T40[2], T09[2], T41[2], T10[2], T42[2], T11[2], T43[2], T12[2], T44[2], T13[2], T45[2], T14[2], T46[2], T15[2], T47[2], T08[3], T40[3], T09[3], T41[3], T10[3], T42[3], T11[3], T43[3], T12[3], T44[3], T13[3], T45[3], T14[3], T46[3], T15[3], T47[3],
        V16[2], V17[2], V18[2], V19[2], V20[2], V21[2], V22[2], V23[2], V24[2], V25[2], V26[2], V27[2], V28[2], V29[2], V30[2], V31[2], V16[3], V17[3], V18[3], V19[3], V20[3], V21[3], V22[3], V23[3], V24[3], V25[3], V26[3], V27[3], V28[3], V29[3], V30[3], V31[3]);

    TRANSPOSE_16x16_16BIT(
        T16[0], T48[0], T17[0], T49[0], T18[0], T50[0], T19[0], T51[0], T20[0], T52[0], T21[0], T53[0], T22[0], T54[0], T23[0], T55[0], T16[1], T48[1], T17[1], T49[1], T18[1], T50[1], T19[1], T51[1], T20[1], T52[1], T21[1], T53[1], T22[1], T54[1], T23[1], T55[1],
        V00[4], V01[4], V02[4], V03[4], V04[4], V05[4], V06[4], V07[4], V08[4], V09[4], V10[4], V11[4], V12[4], V13[4], V14[4], V15[4], V00[5], V01[5], V02[5], V03[5], V04[5], V05[5], V06[5], V07[5], V08[5], V09[5], V10[5], V11[5], V12[5], V13[5], V14[5], V15[5]);
    TRANSPOSE_16x16_16BIT(
        T16[2], T48[2], T17[2], T49[2], T18[2], T50[2], T19[2], T51[2], T20[2], T52[2], T21[2], T53[2], T22[2], T54[2], T23[2], T55[2], T16[3], T48[3], T17[3], T49[3], T18[3], T50[3], T19[3], T51[3], T20[3], T52[3], T21[3], T53[3], T22[3], T54[3], T23[3], T55[3],
        V16[4], V17[4], V18[4], V19[4], V20[4], V21[4], V22[4], V23[4], V24[4], V25[4], V26[4], V27[4], V28[4], V29[4], V30[4], V31[4], V16[5], V17[5], V18[5], V19[5], V20[5], V21[5], V22[5], V23[5], V24[5], V25[5], V26[5], V27[5], V28[5], V29[5], V30[5], V31[5]);

    TRANSPOSE_16x16_16BIT(
        T24[0], T56[0], T25[0], T57[0], T26[0], T58[0], T27[0], T59[0], T28[0], T60[0], T29[0], T61[0], T30[0], T62[0], T31[0], T63[0], T24[1], T56[1], T25[1], T57[1], T26[1], T58[1], T27[1], T59[1], T28[1], T60[1], T29[1], T61[1], T30[1], T62[1], T31[1], T63[1],
        V00[6], V01[6], V02[6], V03[6], V04[6], V05[6], V06[6], V07[6], V08[6], V09[6], V10[6], V11[6], V12[6], V13[6], V14[6], V15[6], V00[7], V01[7], V02[7], V03[7], V04[7], V05[7], V06[7], V07[7], V08[7], V09[7], V10[7], V11[7], V12[7], V13[7], V14[7], V15[7]);
    TRANSPOSE_16x16_16BIT(
        T24[2], T56[2], T25[2], T57[2], T26[2], T58[2], T27[2], T59[2], T28[2], T60[2], T29[2], T61[2], T30[2], T62[2], T31[2], T63[2], T24[3], T56[3], T25[3], T57[3], T26[3], T58[3], T27[3], T59[3], T28[3], T60[3], T29[3], T61[3], T30[3], T62[3], T31[3], T63[3],
        V16[6], V17[6], V18[6], V19[6], V20[6], V21[6], V22[6], V23[6], V24[6], V25[6], V26[6], V27[6], V28[6], V29[6], V30[6], V31[6], V16[7], V17[7], V18[7], V19[7], V20[7], V21[7], V22[7], V23[7], V24[7], V25[7], V26[7], V27[7], V28[7], V29[7], V30[7], V31[7]);

    /*--horizontal transform--*/
    //filter (odd pixel/column)
    for (i = 0; i < 8; i++) {
        V32[i] = _mm_srai_epi16(_mm_add_epi16(V00[i], V01[i]), 1);
        V33[i] = _mm_srai_epi16(_mm_add_epi16(V01[i], V02[i]), 1);
        V34[i] = _mm_srai_epi16(_mm_add_epi16(V02[i], V03[i]), 1);
        V35[i] = _mm_srai_epi16(_mm_add_epi16(V03[i], V04[i]), 1);
        V36[i] = _mm_srai_epi16(_mm_add_epi16(V04[i], V05[i]), 1);
        V37[i] = _mm_srai_epi16(_mm_add_epi16(V05[i], V06[i]), 1);
        V38[i] = _mm_srai_epi16(_mm_add_epi16(V06[i], V07[i]), 1);
        V39[i] = _mm_srai_epi16(_mm_add_epi16(V07[i], V08[i]), 1);
        V40[i] = _mm_srai_epi16(_mm_add_epi16(V08[i], V09[i]), 1);
        V41[i] = _mm_srai_epi16(_mm_add_epi16(V09[i], V10[i]), 1);
        V42[i] = _mm_srai_epi16(_mm_add_epi16(V10[i], V11[i]), 1);
        V43[i] = _mm_srai_epi16(_mm_add_epi16(V11[i], V12[i]), 1);
        V44[i] = _mm_srai_epi16(_mm_add_epi16(V12[i], V13[i]), 1);
        V45[i] = _mm_srai_epi16(_mm_add_epi16(V13[i], V14[i]), 1);
        V46[i] = _mm_srai_epi16(_mm_add_epi16(V14[i], V15[i]), 1);
        V47[i] = _mm_srai_epi16(_mm_add_epi16(V15[i], V16[i]), 1);

        V48[i] = _mm_srai_epi16(_mm_add_epi16(V16[i], V17[i]), 1);
        V49[i] = _mm_srai_epi16(_mm_add_epi16(V17[i], V18[i]), 1);
        V50[i] = _mm_srai_epi16(_mm_add_epi16(V18[i], V19[i]), 1);
        V51[i] = _mm_srai_epi16(_mm_add_epi16(V19[i], V20[i]), 1);
        V52[i] = _mm_srai_epi16(_mm_add_epi16(V20[i], V21[i]), 1);
        V53[i] = _mm_srai_epi16(_mm_add_epi16(V21[i], V22[i]), 1);
        V54[i] = _mm_srai_epi16(_mm_add_epi16(V22[i], V23[i]), 1);
        V55[i] = _mm_srai_epi16(_mm_add_epi16(V23[i], V24[i]), 1);
        V56[i] = _mm_srai_epi16(_mm_add_epi16(V24[i], V25[i]), 1);
        V57[i] = _mm_srai_epi16(_mm_add_epi16(V25[i], V26[i]), 1);
        V58[i] = _mm_srai_epi16(_mm_add_epi16(V26[i], V27[i]), 1);
        V59[i] = _mm_srai_epi16(_mm_add_epi16(V27[i], V28[i]), 1);
        V60[i] = _mm_srai_epi16(_mm_add_epi16(V28[i], V29[i]), 1);
        V61[i] = _mm_srai_epi16(_mm_add_epi16(V29[i], V30[i]), 1);
        V62[i] = _mm_srai_epi16(_mm_add_epi16(V30[i], V31[i]), 1);
        V63[i] = _mm_srai_epi16(_mm_add_epi16(V31[i], V31[i]), 1);
    }

    /*--transposition & Store--*/
    //64x64
    TRANSPOSE_16x16_16BIT(
        V00[0], V32[0], V01[0], V33[0], V02[0], V34[0], V03[0], V35[0], V04[0], V36[0], V05[0], V37[0], V06[0], V38[0], V07[0], V39[0], V00[1], V32[1], V01[1], V33[1], V02[1], V34[1], V03[1], V35[1], V04[1], V36[1], V05[1], V37[1], V06[1], V38[1], V07[1], V39[1],
        T00[0], T01[0], T02[0], T03[0], T04[0], T05[0], T06[0], T07[0], T08[0], T09[0], T10[0], T11[0], T12[0], T13[0], T14[0], T15[0], T00[1], T01[1], T02[1], T03[1], T04[1], T05[1], T06[1], T07[1], T08[1], T09[1], T10[1], T11[1], T12[1], T13[1], T14[1], T15[1]);
    TRANSPOSE_16x16_16BIT(
        V00[2], V32[2], V01[2], V33[2], V02[2], V34[2], V03[2], V35[2], V04[2], V36[2], V05[2], V37[2], V06[2], V38[2], V07[2], V39[2], V00[3], V32[3], V01[3], V33[3], V02[3], V34[3], V03[3], V35[3], V04[3], V36[3], V05[3], V37[3], V06[3], V38[3], V07[3], V39[3],
        T16[0], T17[0], T18[0], T19[0], T20[0], T21[0], T22[0], T23[0], T24[0], T25[0], T26[0], T27[0], T28[0], T29[0], T30[0], T31[0], T16[1], T17[1], T18[1], T19[1], T20[1], T21[1], T22[1], T23[1], T24[1], T25[1], T26[1], T27[1], T28[1], T29[1], T30[1], T31[1]);
    TRANSPOSE_16x16_16BIT(V00[4], V32[4], V01[4], V33[4], V02[4], V34[4], V03[4], V35[4], V04[4], V36[4], V05[4], V37[4], V06[4], V38[4], V07[4], V39[4], V00[5], V32[5], V01[5], V33[5], V02[5], V34[5], V03[5], V35[5], V04[5], V36[5], V05[5], V37[5], V06[5], V38[5], V07[5], V39[5], T32[0], T33[0], T34[0], T35[0], T36[0], T37[0], T38[0], T39[0], T40[0], T41[0], T42[0], T43[0], T44[0], T45[0], T46[0], T47[0], T32[1], T33[1], T34[1], T35[1], T36[1], T37[1], T38[1], T39[1], T40[1], T41[1], T42[1], T43[1], T44[1], T45[1], T46[1], T47[1]);
    TRANSPOSE_16x16_16BIT(V00[6], V32[6], V01[6], V33[6], V02[6], V34[6], V03[6], V35[6], V04[6], V36[6], V05[6], V37[6], V06[6], V38[6], V07[6], V39[6], V00[7], V32[7], V01[7], V33[7], V02[7], V34[7], V03[7], V35[7], V04[7], V36[7], V05[7], V37[7], V06[7], V38[7], V07[7], V39[7], T48[0], T49[0], T50[0], T51[0], T52[0], T53[0], T54[0], T55[0], T56[0], T57[0], T58[0], T59[0], T60[0], T61[0], T62[0], T63[0], T48[1], T49[1], T50[1], T51[1], T52[1], T53[1], T54[1], T55[1], T56[1], T57[1], T58[1], T59[1], T60[1], T61[1], T62[1], T63[1]);

    TRANSPOSE_16x16_16BIT(
        V08[0], V40[0], V09[0], V41[0], V10[0], V42[0], V11[0], V43[0], V12[0], V44[0], V13[0], V45[0], V14[0], V46[0], V15[0], V47[0], V08[1], V40[1], V09[1], V41[1], V10[1], V42[1], V11[1], V43[1], V12[1], V44[1], V13[1], V45[1], V14[1], V46[1], V15[1], V47[1],
        T00[2], T01[2], T02[2], T03[2], T04[2], T05[2], T06[2], T07[2], T08[2], T09[2], T10[2], T11[2], T12[2], T13[2], T14[2], T15[2], T00[3], T01[3], T02[3], T03[3], T04[3], T05[3], T06[3], T07[3], T08[3], T09[3], T10[3], T11[3], T12[3], T13[3], T14[3], T15[3]);
    TRANSPOSE_16x16_16BIT(
        V08[2], V40[2], V09[2], V41[2], V10[2], V42[2], V11[2], V43[2], V12[2], V44[2], V13[2], V45[2], V14[2], V46[2], V15[2], V47[2], V08[3], V40[3], V09[3], V41[3], V10[3], V42[3], V11[3], V43[3], V12[3], V44[3], V13[3], V45[3], V14[3], V46[3], V15[3], V47[3],
        T16[2], T17[2], T18[2], T19[2], T20[2], T21[2], T22[2], T23[2], T24[2], T25[2], T26[2], T27[2], T28[2], T29[2], T30[2], T31[2], T16[3], T17[3], T18[3], T19[3], T20[3], T21[3], T22[3], T23[3], T24[3], T25[3], T26[3], T27[3], T28[3], T29[3], T30[3], T31[3]);
    TRANSPOSE_16x16_16BIT(
        V08[4], V40[4], V09[4], V41[4], V10[4], V42[4], V11[4], V43[4], V12[4], V44[4], V13[4], V45[4], V14[4], V46[4], V15[4], V47[4], V08[5], V40[5], V09[5], V41[5], V10[5], V42[5], V11[5], V43[5], V12[5], V44[5], V13[5], V45[5], V14[5], V46[5], V15[5], V47[5],
        T32[2], T33[2], T34[2], T35[2], T36[2], T37[2], T38[2], T39[2], T40[2], T41[2], T42[2], T43[2], T44[2], T45[2], T46[2], T47[2], T32[3], T33[3], T34[3], T35[3], T36[3], T37[3], T38[3], T39[3], T40[3], T41[3], T42[3], T43[3], T44[3], T45[3], T46[3], T47[3]);
    TRANSPOSE_16x16_16BIT(
        V08[6], V40[6], V09[6], V41[6], V10[6], V42[6], V11[6], V43[6], V12[6], V44[6], V13[6], V45[6], V14[6], V46[6], V15[6], V47[6], V08[7], V40[7], V09[7], V41[7], V10[7], V42[7], V11[7], V43[7], V12[7], V44[7], V13[7], V45[7], V14[7], V46[7], V15[7], V47[7],
        T48[2], T49[2], T50[2], T51[2], T52[2], T53[2], T54[2], T55[2], T56[2], T57[2], T58[2], T59[2], T60[2], T61[2], T62[2], T63[2], T48[3], T49[3], T50[3], T51[3], T52[3], T53[3], T54[3], T55[3], T56[3], T57[3], T58[3], T59[3], T60[3], T61[3], T62[3], T63[3]);

    TRANSPOSE_16x16_16BIT(
        V16[0], V48[0], V17[0], V49[0], V18[0], V50[0], V19[0], V51[0], V20[0], V52[0], V21[0], V53[0], V22[0], V54[0], V23[0], V55[0], V16[1], V48[1], V17[1], V49[1], V18[1], V50[1], V19[1], V51[1], V20[1], V52[1], V21[1], V53[1], V22[1], V54[1], V23[1], V55[1],
        T00[4], T01[4], T02[4], T03[4], T04[4], T05[4], T06[4], T07[4], T08[4], T09[4], T10[4], T11[4], T12[4], T13[4], T14[4], T15[4], T00[5], T01[5], T02[5], T03[5], T04[5], T05[5], T06[5], T07[5], T08[5], T09[5], T10[5], T11[5], T12[5], T13[5], T14[5], T15[5]);
    TRANSPOSE_16x16_16BIT(
        V16[2], V48[2], V17[2], V49[2], V18[2], V50[2], V19[2], V51[2], V20[2], V52[2], V21[2], V53[2], V22[2], V54[2], V23[2], V55[2], V16[3], V48[3], V17[3], V49[3], V18[3], V50[3], V19[3], V51[3], V20[3], V52[3], V21[3], V53[3], V22[3], V54[3], V23[3], V55[3],
        T16[4], T17[4], T18[4], T19[4], T20[4], T21[4], T22[4], T23[4], T24[4], T25[4], T26[4], T27[4], T28[4], T29[4], T30[4], T31[4], T16[5], T17[5], T18[5], T19[5], T20[5], T21[5], T22[5], T23[5], T24[5], T25[5], T26[5], T27[5], T28[5], T29[5], T30[5], T31[5]);
    TRANSPOSE_16x16_16BIT(
        V16[4], V48[4], V17[4], V49[4], V18[4], V50[4], V19[4], V51[4], V20[4], V52[4], V21[4], V53[4], V22[4], V54[4], V23[4], V55[4], V16[5], V48[5], V17[5], V49[5], V18[5], V50[5], V19[5], V51[5], V20[5], V52[5], V21[5], V53[5], V22[5], V54[5], V23[5], V55[5],
        T32[4], T33[4], T34[4], T35[4], T36[4], T37[4], T38[4], T39[4], T40[4], T41[4], T42[4], T43[4], T44[4], T45[4], T46[4], T47[4], T32[5], T33[5], T34[5], T35[5], T36[5], T37[5], T38[5], T39[5], T40[5], T41[5], T42[5], T43[5], T44[5], T45[5], T46[5], T47[5]);
    TRANSPOSE_16x16_16BIT(
        V16[6], V48[6], V17[6], V49[6], V18[6], V50[6], V19[6], V51[6], V20[6], V52[6], V21[6], V53[6], V22[6], V54[6], V23[6], V55[6], V16[7], V48[7], V17[7], V49[7], V18[7], V50[7], V19[7], V51[7], V20[7], V52[7], V21[7], V53[7], V22[7], V54[7], V23[7], V55[7],
        T48[4], T49[4], T50[4], T51[4], T52[4], T53[4], T54[4], T55[4], T56[4], T57[4], T58[4], T59[4], T60[4], T61[4], T62[4], T63[4], T48[5], T49[5], T50[5], T51[5], T52[5], T53[5], T54[5], T55[5], T56[5], T57[5], T58[5], T59[5], T60[5], T61[5], T62[5], T63[5]);

    TRANSPOSE_16x16_16BIT(
        V24[0], V56[0], V25[0], V57[0], V26[0], V58[0], V27[0], V59[0], V28[0], V60[0], V29[0], V61[0], V30[0], V62[0], V31[0], V63[0], V24[1], V56[1], V25[1], V57[1], V26[1], V58[1], V27[1], V59[1], V28[1], V60[1], V29[1], V61[1], V30[1], V62[1], V31[1], V63[1],
        T00[6], T01[6], T02[6], T03[6], T04[6], T05[6], T06[6], T07[6], T08[6], T09[6], T10[6], T11[6], T12[6], T13[6], T14[6], T15[6], T00[7], T01[7], T02[7], T03[7], T04[7], T05[7], T06[7], T07[7], T08[7], T09[7], T10[7], T11[7], T12[7], T13[7], T14[7], T15[7]);
    TRANSPOSE_16x16_16BIT(
        V24[2], V56[2], V25[2], V57[2], V26[2], V58[2], V27[2], V59[2], V28[2], V60[2], V29[2], V61[2], V30[2], V62[2], V31[2], V63[2], V24[3], V56[3], V25[3], V57[3], V26[3], V58[3], V27[3], V59[3], V28[3], V60[3], V29[3], V61[3], V30[3], V62[3], V31[3], V63[3],
        T16[6], T17[6], T18[6], T19[6], T20[6], T21[6], T22[6], T23[6], T24[6], T25[6], T26[6], T27[6], T28[6], T29[6], T30[6], T31[6], T16[7], T17[7], T18[7], T19[7], T20[7], T21[7], T22[7], T23[7], T24[7], T25[7], T26[7], T27[7], T28[7], T29[7], T30[7], T31[7]);
    TRANSPOSE_16x16_16BIT(
        V24[4], V56[4], V25[4], V57[4], V26[4], V58[4], V27[4], V59[4], V28[4], V60[4], V29[4], V61[4], V30[4], V62[4], V31[4], V63[4], V24[5], V56[5], V25[5], V57[5], V26[5], V58[5], V27[5], V59[5], V28[5], V60[5], V29[5], V61[5], V30[5], V62[5], V31[5], V63[5],
        T32[6], T33[6], T34[6], T35[6], T36[6], T37[6], T38[6], T39[6], T40[6], T41[6], T42[6], T43[6], T44[6], T45[6], T46[6], T47[6], T32[7], T33[7], T34[7], T35[7], T36[7], T37[7], T38[7], T39[7], T40[7], T41[7], T42[7], T43[7], T44[7], T45[7], T46[7], T47[7]);
    TRANSPOSE_16x16_16BIT(
        V24[6], V56[6], V25[6], V57[6], V26[6], V58[6], V27[6], V59[6], V28[6], V60[6], V29[6], V61[6], V30[6], V62[6], V31[6], V63[6], V24[7], V56[7], V25[7], V57[7], V26[7], V58[7], V27[7], V59[7], V28[7], V60[7], V29[7], V61[7], V30[7], V62[7], V31[7], V63[7],
        T48[6], T49[6], T50[6], T51[6], T52[6], T53[6], T54[6], T55[6], T56[6], T57[6], T58[6], T59[6], T60[6], T61[6], T62[6], T63[6], T48[7], T49[7], T50[7], T51[7], T52[7], T53[7], T54[7], T55[7], T56[7], T57[7], T58[7], T59[7], T60[7], T61[7], T62[7], T63[7]);

    //store
    for (i = 0; i < 8; i++) {
        _mm_storeu_si128((__m128i*)&coeff[8 * i          ], T00[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64     ], T01[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 *  2], T02[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 *  3], T03[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 *  4], T04[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 *  5], T05[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 *  6], T06[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 *  7], T07[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 *  8], T08[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 *  9], T09[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 10], T10[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 11], T11[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 12], T12[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 13], T13[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 14], T14[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 15], T15[i]);

        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 16], T16[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 17], T17[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 18], T18[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 19], T19[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 20], T20[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 21], T21[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 22], T22[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 23], T23[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 24], T24[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 25], T25[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 26], T26[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 27], T27[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 28], T28[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 29], T29[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 30], T30[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 31], T31[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 32], T32[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 33], T33[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 34], T34[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 35], T35[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 36], T36[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 37], T37[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 38], T38[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 39], T39[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 40], T40[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 41], T41[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 42], T42[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 43], T43[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 44], T44[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 45], T45[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 46], T46[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 47], T47[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 48], T48[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 49], T49[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 50], T50[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 51], T51[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 52], T52[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 53], T53[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 54], T54[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 55], T55[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 56], T56[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 57], T57[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 58], T58[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 59], T59[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 60], T60[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 61], T61[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 62], T62[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 64 * 63], T63[i]);
    }
}


/* ---------------------------------------------------------------------------
 */
static void inv_wavelet_64x16_sse128(coeff_t *coeff)
{
    int i;
    //按行 64*16
    __m128i T00[8], T01[8], T02[8], T03[8], T04[8], T05[8], T06[8], T07[8], T08[8], T09[8], T10[8], T11[8], T12[8], T13[8], T14[8], T15[8];

    //按列 16*64
    __m128i V00[2], V01[2], V02[2], V03[2], V04[2], V05[2], V06[2], V07[2], V08[2], V09[2], V10[2], V11[2], V12[2], V13[2], V14[2], V15[2], V16[2], V17[2], V18[2], V19[2], V20[2], V21[2], V22[2], V23[2], V24[2], V25[2], V26[2], V27[2], V28[2], V29[2], V30[2], V31[2], V32[2], V33[2], V34[2], V35[2], V36[2], V37[2], V38[2], V39[2], V40[2], V41[2], V42[2], V43[2], V44[2], V45[2], V46[2], V47[2], V48[2], V49[2], V50[2], V51[2], V52[2], V53[2], V54[2], V55[2], V56[2], V57[2], V58[2], V59[2], V60[2], V61[2], V62[2], V63[2];

    __m128i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
    __m128i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;

    /*--vertical transform--*/
    //32*8, LOAD AND SHIFT
    T00[0] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[ 0 + 32 * 0]), 1);
    T01[0] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[ 0 + 32 * 1]), 1);
    T02[0] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[ 0 + 32 * 2]), 1);
    T03[0] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[ 0 + 32 * 3]), 1);
    T04[0] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[ 0 + 32 * 4]), 1);
    T05[0] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[ 0 + 32 * 5]), 1);
    T06[0] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[ 0 + 32 * 6]), 1);
    T07[0] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[ 0 + 32 * 7]), 1);

    T00[1] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[ 8 + 32 * 0]), 1);
    T01[1] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[ 8 + 32 * 1]), 1);
    T02[1] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[ 8 + 32 * 2]), 1);
    T03[1] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[ 8 + 32 * 3]), 1);
    T04[1] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[ 8 + 32 * 4]), 1);
    T05[1] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[ 8 + 32 * 5]), 1);
    T06[1] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[ 8 + 32 * 6]), 1);
    T07[1] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[ 8 + 32 * 7]), 1);

    T00[2] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[16 + 32 * 0]), 1);
    T01[2] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[16 + 32 * 1]), 1);
    T02[2] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[16 + 32 * 2]), 1);
    T03[2] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[16 + 32 * 3]), 1);
    T04[2] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[16 + 32 * 4]), 1);
    T05[2] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[16 + 32 * 5]), 1);
    T06[2] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[16 + 32 * 6]), 1);
    T07[2] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[16 + 32 * 7]), 1);

    T00[3] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[24 + 32 * 0]), 1);
    T01[3] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[24 + 32 * 1]), 1);
    T02[3] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[24 + 32 * 2]), 1);
    T03[3] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[24 + 32 * 3]), 1);
    T04[3] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[24 + 32 * 4]), 1);
    T05[3] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[24 + 32 * 5]), 1);
    T06[3] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[24 + 32 * 6]), 1);
    T07[3] = _mm_srai_epi16(_mm_load_si128((__m128i*)&coeff[24 + 32 * 7]), 1);

    //filter (odd pixel/row)
    T08[0] = _mm_srai_epi16(_mm_add_epi16(T00[0], T01[0]), 1);
    T09[0] = _mm_srai_epi16(_mm_add_epi16(T01[0], T02[0]), 1);
    T10[0] = _mm_srai_epi16(_mm_add_epi16(T02[0], T03[0]), 1);
    T11[0] = _mm_srai_epi16(_mm_add_epi16(T03[0], T04[0]), 1);
    T12[0] = _mm_srai_epi16(_mm_add_epi16(T04[0], T05[0]), 1);
    T13[0] = _mm_srai_epi16(_mm_add_epi16(T05[0], T06[0]), 1);
    T14[0] = _mm_srai_epi16(_mm_add_epi16(T06[0], T07[0]), 1);
    T15[0] = _mm_srai_epi16(_mm_add_epi16(T07[0], T07[0]), 1);

    T08[1] = _mm_srai_epi16(_mm_add_epi16(T00[1], T01[1]), 1);
    T09[1] = _mm_srai_epi16(_mm_add_epi16(T01[1], T02[1]), 1);
    T10[1] = _mm_srai_epi16(_mm_add_epi16(T02[1], T03[1]), 1);
    T11[1] = _mm_srai_epi16(_mm_add_epi16(T03[1], T04[1]), 1);
    T12[1] = _mm_srai_epi16(_mm_add_epi16(T04[1], T05[1]), 1);
    T13[1] = _mm_srai_epi16(_mm_add_epi16(T05[1], T06[1]), 1);
    T14[1] = _mm_srai_epi16(_mm_add_epi16(T06[1], T07[1]), 1);
    T15[1] = _mm_srai_epi16(_mm_add_epi16(T07[1], T07[1]), 1);

    T08[2] = _mm_srai_epi16(_mm_add_epi16(T00[2], T01[2]), 1);
    T09[2] = _mm_srai_epi16(_mm_add_epi16(T01[2], T02[2]), 1);
    T10[2] = _mm_srai_epi16(_mm_add_epi16(T02[2], T03[2]), 1);
    T11[2] = _mm_srai_epi16(_mm_add_epi16(T03[2], T04[2]), 1);
    T12[2] = _mm_srai_epi16(_mm_add_epi16(T04[2], T05[2]), 1);
    T13[2] = _mm_srai_epi16(_mm_add_epi16(T05[2], T06[2]), 1);
    T14[2] = _mm_srai_epi16(_mm_add_epi16(T06[2], T07[2]), 1);
    T15[2] = _mm_srai_epi16(_mm_add_epi16(T07[2], T07[2]), 1);

    T08[3] = _mm_srai_epi16(_mm_add_epi16(T00[3], T01[3]), 1);
    T09[3] = _mm_srai_epi16(_mm_add_epi16(T01[3], T02[3]), 1);
    T10[3] = _mm_srai_epi16(_mm_add_epi16(T02[3], T03[3]), 1);
    T11[3] = _mm_srai_epi16(_mm_add_epi16(T03[3], T04[3]), 1);
    T12[3] = _mm_srai_epi16(_mm_add_epi16(T04[3], T05[3]), 1);
    T13[3] = _mm_srai_epi16(_mm_add_epi16(T05[3], T06[3]), 1);
    T14[3] = _mm_srai_epi16(_mm_add_epi16(T06[3], T07[3]), 1);
    T15[3] = _mm_srai_epi16(_mm_add_epi16(T07[3], T07[3]), 1);

    /*--transposition--*/
    //32x16 -> 16x32
    TRANSPOSE_8x8_16BIT(T00[0], T08[0], T01[0], T09[0], T02[0], T10[0], T03[0], T11[0], V00[0], V01[0], V02[0], V03[0], V04[0], V05[0], V06[0], V07[0]);
    TRANSPOSE_8x8_16BIT(T00[1], T08[1], T01[1], T09[1], T02[1], T10[1], T03[1], T11[1], V08[0], V09[0], V10[0], V11[0], V12[0], V13[0], V14[0], V15[0]);
    TRANSPOSE_8x8_16BIT(T00[2], T08[2], T01[2], T09[2], T02[2], T10[2], T03[2], T11[2], V16[0], V17[0], V18[0], V19[0], V20[0], V21[0], V22[0], V23[0]);
    TRANSPOSE_8x8_16BIT(T00[3], T08[3], T01[3], T09[3], T02[3], T10[3], T03[3], T11[3], V24[0], V25[0], V26[0], V27[0], V28[0], V29[0], V30[0], V31[0]);

    TRANSPOSE_8x8_16BIT(T04[0], T12[0], T05[0], T13[0], T06[0], T14[0], T07[0], T15[0], V00[1], V01[1], V02[1], V03[1], V04[1], V05[1], V06[1], V07[1]);
    TRANSPOSE_8x8_16BIT(T04[1], T12[1], T05[1], T13[1], T06[1], T14[1], T07[1], T15[1], V08[1], V09[1], V10[1], V11[1], V12[1], V13[1], V14[1], V15[1]);
    TRANSPOSE_8x8_16BIT(T04[2], T12[2], T05[2], T13[2], T06[2], T14[2], T07[2], T15[2], V16[1], V17[1], V18[1], V19[1], V20[1], V21[1], V22[1], V23[1]);
    TRANSPOSE_8x8_16BIT(T04[3], T12[3], T05[3], T13[3], T06[3], T14[3], T07[3], T15[3], V24[1], V25[1], V26[1], V27[1], V28[1], V29[1], V30[1], V31[1]);

    /*--horizontal transform--*/
    //filter (odd pixel/column)
    V32[0] = _mm_srai_epi16(_mm_add_epi16(V00[0], V01[0]), 1);
    V33[0] = _mm_srai_epi16(_mm_add_epi16(V01[0], V02[0]), 1);
    V34[0] = _mm_srai_epi16(_mm_add_epi16(V02[0], V03[0]), 1);
    V35[0] = _mm_srai_epi16(_mm_add_epi16(V03[0], V04[0]), 1);
    V36[0] = _mm_srai_epi16(_mm_add_epi16(V04[0], V05[0]), 1);
    V37[0] = _mm_srai_epi16(_mm_add_epi16(V05[0], V06[0]), 1);
    V38[0] = _mm_srai_epi16(_mm_add_epi16(V06[0], V07[0]), 1);
    V39[0] = _mm_srai_epi16(_mm_add_epi16(V07[0], V08[0]), 1);
    V40[0] = _mm_srai_epi16(_mm_add_epi16(V08[0], V09[0]), 1);
    V41[0] = _mm_srai_epi16(_mm_add_epi16(V09[0], V10[0]), 1);
    V42[0] = _mm_srai_epi16(_mm_add_epi16(V10[0], V11[0]), 1);
    V43[0] = _mm_srai_epi16(_mm_add_epi16(V11[0], V12[0]), 1);
    V44[0] = _mm_srai_epi16(_mm_add_epi16(V12[0], V13[0]), 1);
    V45[0] = _mm_srai_epi16(_mm_add_epi16(V13[0], V14[0]), 1);
    V46[0] = _mm_srai_epi16(_mm_add_epi16(V14[0], V15[0]), 1);
    V47[0] = _mm_srai_epi16(_mm_add_epi16(V15[0], V16[0]), 1);

    V48[0] = _mm_srai_epi16(_mm_add_epi16(V16[0], V17[0]), 1);
    V49[0] = _mm_srai_epi16(_mm_add_epi16(V17[0], V18[0]), 1);
    V50[0] = _mm_srai_epi16(_mm_add_epi16(V18[0], V19[0]), 1);
    V51[0] = _mm_srai_epi16(_mm_add_epi16(V19[0], V20[0]), 1);
    V52[0] = _mm_srai_epi16(_mm_add_epi16(V20[0], V21[0]), 1);
    V53[0] = _mm_srai_epi16(_mm_add_epi16(V21[0], V22[0]), 1);
    V54[0] = _mm_srai_epi16(_mm_add_epi16(V22[0], V23[0]), 1);
    V55[0] = _mm_srai_epi16(_mm_add_epi16(V23[0], V24[0]), 1);
    V56[0] = _mm_srai_epi16(_mm_add_epi16(V24[0], V25[0]), 1);
    V57[0] = _mm_srai_epi16(_mm_add_epi16(V25[0], V26[0]), 1);
    V58[0] = _mm_srai_epi16(_mm_add_epi16(V26[0], V27[0]), 1);
    V59[0] = _mm_srai_epi16(_mm_add_epi16(V27[0], V28[0]), 1);
    V60[0] = _mm_srai_epi16(_mm_add_epi16(V28[0], V29[0]), 1);
    V61[0] = _mm_srai_epi16(_mm_add_epi16(V29[0], V30[0]), 1);
    V62[0] = _mm_srai_epi16(_mm_add_epi16(V30[0], V31[0]), 1);
    V63[0] = _mm_srai_epi16(_mm_add_epi16(V31[0], V31[0]), 1);

    V32[1] = _mm_srai_epi16(_mm_add_epi16(V00[1], V01[1]), 1);
    V33[1] = _mm_srai_epi16(_mm_add_epi16(V01[1], V02[1]), 1);
    V34[1] = _mm_srai_epi16(_mm_add_epi16(V02[1], V03[1]), 1);
    V35[1] = _mm_srai_epi16(_mm_add_epi16(V03[1], V04[1]), 1);
    V36[1] = _mm_srai_epi16(_mm_add_epi16(V04[1], V05[1]), 1);
    V37[1] = _mm_srai_epi16(_mm_add_epi16(V05[1], V06[1]), 1);
    V38[1] = _mm_srai_epi16(_mm_add_epi16(V06[1], V07[1]), 1);
    V39[1] = _mm_srai_epi16(_mm_add_epi16(V07[1], V08[1]), 1);
    V40[1] = _mm_srai_epi16(_mm_add_epi16(V08[1], V09[1]), 1);
    V41[1] = _mm_srai_epi16(_mm_add_epi16(V09[1], V10[1]), 1);
    V42[1] = _mm_srai_epi16(_mm_add_epi16(V10[1], V11[1]), 1);
    V43[1] = _mm_srai_epi16(_mm_add_epi16(V11[1], V12[1]), 1);
    V44[1] = _mm_srai_epi16(_mm_add_epi16(V12[1], V13[1]), 1);
    V45[1] = _mm_srai_epi16(_mm_add_epi16(V13[1], V14[1]), 1);
    V46[1] = _mm_srai_epi16(_mm_add_epi16(V14[1], V15[1]), 1);
    V47[1] = _mm_srai_epi16(_mm_add_epi16(V15[1], V16[1]), 1);

    V48[1] = _mm_srai_epi16(_mm_add_epi16(V16[1], V17[1]), 1);
    V49[1] = _mm_srai_epi16(_mm_add_epi16(V17[1], V18[1]), 1);
    V50[1] = _mm_srai_epi16(_mm_add_epi16(V18[1], V19[1]), 1);
    V51[1] = _mm_srai_epi16(_mm_add_epi16(V19[1], V20[1]), 1);
    V52[1] = _mm_srai_epi16(_mm_add_epi16(V20[1], V21[1]), 1);
    V53[1] = _mm_srai_epi16(_mm_add_epi16(V21[1], V22[1]), 1);
    V54[1] = _mm_srai_epi16(_mm_add_epi16(V22[1], V23[1]), 1);
    V55[1] = _mm_srai_epi16(_mm_add_epi16(V23[1], V24[1]), 1);
    V56[1] = _mm_srai_epi16(_mm_add_epi16(V24[1], V25[1]), 1);
    V57[1] = _mm_srai_epi16(_mm_add_epi16(V25[1], V26[1]), 1);
    V58[1] = _mm_srai_epi16(_mm_add_epi16(V26[1], V27[1]), 1);
    V59[1] = _mm_srai_epi16(_mm_add_epi16(V27[1], V28[1]), 1);
    V60[1] = _mm_srai_epi16(_mm_add_epi16(V28[1], V29[1]), 1);
    V61[1] = _mm_srai_epi16(_mm_add_epi16(V29[1], V30[1]), 1);
    V62[1] = _mm_srai_epi16(_mm_add_epi16(V30[1], V31[1]), 1);
    V63[1] = _mm_srai_epi16(_mm_add_epi16(V31[1], V31[1]), 1);

    /*--transposition & Store--*/
    //16x64 -> 64x16
    TRANSPOSE_8x8_16BIT(V00[0], V32[0], V01[0], V33[0], V02[0], V34[0], V03[0], V35[0], T00[0], T01[0], T02[0], T03[0], T04[0], T05[0], T06[0], T07[0]);
    TRANSPOSE_8x8_16BIT(V04[0], V36[0], V05[0], V37[0], V06[0], V38[0], V07[0], V39[0], T00[1], T01[1], T02[1], T03[1], T04[1], T05[1], T06[1], T07[1]);
    TRANSPOSE_8x8_16BIT(V08[0], V40[0], V09[0], V41[0], V10[0], V42[0], V11[0], V43[0], T00[2], T01[2], T02[2], T03[2], T04[2], T05[2], T06[2], T07[2]);
    TRANSPOSE_8x8_16BIT(V12[0], V44[0], V13[0], V45[0], V14[0], V46[0], V15[0], V47[0], T00[3], T01[3], T02[3], T03[3], T04[3], T05[3], T06[3], T07[3]);
    TRANSPOSE_8x8_16BIT(V16[0], V48[0], V17[0], V49[0], V18[0], V50[0], V19[0], V51[0], T00[4], T01[4], T02[4], T03[4], T04[4], T05[4], T06[4], T07[4]);
    TRANSPOSE_8x8_16BIT(V20[0], V52[0], V21[0], V53[0], V22[0], V54[0], V23[0], V55[0], T00[5], T01[5], T02[5], T03[5], T04[5], T05[5], T06[5], T07[5]);
    TRANSPOSE_8x8_16BIT(V24[0], V56[0], V25[0], V57[0], V26[0], V58[0], V27[0], V59[0], T00[6], T01[6], T02[6], T03[6], T04[6], T05[6], T06[6], T07[6]);
    TRANSPOSE_8x8_16BIT(V28[0], V60[0], V29[0], V61[0], V30[0], V62[0], V31[0], V63[0], T00[7], T01[7], T02[7], T03[7], T04[7], T05[7], T06[7], T07[7]);

    TRANSPOSE_8x8_16BIT(V00[1], V32[1], V01[1], V33[1], V02[1], V34[1], V03[1], V35[1], T08[0], T09[0], T10[0], T11[0], T12[0], T13[0], T14[0], T15[0]);
    TRANSPOSE_8x8_16BIT(V04[1], V36[1], V05[1], V37[1], V06[1], V38[1], V07[1], V39[1], T08[1], T09[1], T10[1], T11[1], T12[1], T13[1], T14[1], T15[1]);
    TRANSPOSE_8x8_16BIT(V08[1], V40[1], V09[1], V41[1], V10[1], V42[1], V11[1], V43[1], T08[2], T09[2], T10[2], T11[2], T12[2], T13[2], T14[2], T15[2]);
    TRANSPOSE_8x8_16BIT(V12[1], V44[1], V13[1], V45[1], V14[1], V46[1], V15[1], V47[1], T08[3], T09[3], T10[3], T11[3], T12[3], T13[3], T14[3], T15[3]);
    TRANSPOSE_8x8_16BIT(V16[1], V48[1], V17[1], V49[1], V18[1], V50[1], V19[1], V51[1], T08[4], T09[4], T10[4], T11[4], T12[4], T13[4], T14[4], T15[4]);
    TRANSPOSE_8x8_16BIT(V20[1], V52[1], V21[1], V53[1], V22[1], V54[1], V23[1], V55[1], T08[5], T09[5], T10[5], T11[5], T12[5], T13[5], T14[5], T15[5]);
    TRANSPOSE_8x8_16BIT(V24[1], V56[1], V25[1], V57[1], V26[1], V58[1], V27[1], V59[1], T08[6], T09[6], T10[6], T11[6], T12[6], T13[6], T14[6], T15[6]);
    TRANSPOSE_8x8_16BIT(V28[1], V60[1], V29[1], V61[1], V30[1], V62[1], V31[1], V63[1], T08[7], T09[7], T10[7], T11[7], T12[7], T13[7], T14[7], T15[7]);

    //store
    for (i = 0; i < 8; i++) {
        _mm_store_si128((__m128i*)&coeff[8 * i          ], T00[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 64     ], T01[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 64 *  2], T02[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 64 *  3], T03[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 64 *  4], T04[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 64 *  5], T05[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 64 *  6], T06[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 64 *  7], T07[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 64 *  8], T08[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 64 *  9], T09[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 64 * 10], T10[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 64 * 11], T11[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 64 * 12], T12[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 64 * 13], T13[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 64 * 14], T14[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 64 * 15], T15[i]);
    }
}


/* ---------------------------------------------------------------------------
 */
static void inv_wavelet_16x64_sse128(coeff_t *coeff)
{
    //src coeff 8*32
    __m128i S00, S01, S02, S03, S04, S05, S06, S07, S08, S09, S10, S11, S12, S13, S14, S15, S16, S17, S18, S19, S20, S21, S22, S23, S24, S25, S26, S27, S28, S29, S30, S31;
    __m128i S32, S33, S34, S35, S36, S37, S38, S39, S40, S41, S42, S43, S44, S45, S46, S47, S48, S49, S50, S51, S52, S53, S54, S55, S56, S57, S58, S59, S60, S61, S62, S63;

    //按行 64*16
    __m128i T00[8], T01[8], T02[8], T03[8], T04[8], T05[8], T06[8], T07[8], T08[8], T09[8], T10[8], T11[8], T12[8], T13[8], T14[8], T15[8];

    //按列 16*64
    __m128i V00[2], V01[2], V02[2], V03[2], V04[2], V05[2], V06[2], V07[2], V08[2], V09[2], V10[2], V11[2], V12[2], V13[2], V14[2], V15[2], V16[2], V17[2], V18[2], V19[2], V20[2], V21[2], V22[2], V23[2], V24[2], V25[2], V26[2], V27[2], V28[2], V29[2], V30[2], V31[2], V32[2], V33[2], V34[2], V35[2], V36[2], V37[2], V38[2], V39[2], V40[2], V41[2], V42[2], V43[2], V44[2], V45[2], V46[2], V47[2], V48[2], V49[2], V50[2], V51[2], V52[2], V53[2], V54[2], V55[2], V56[2], V57[2], V58[2], V59[2], V60[2], V61[2], V62[2], V63[2];

    __m128i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
    __m128i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;

    int i;
    /*--load & shift--*/
    //8*32
    S00 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 *  0]), 1);
    S01 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 *  1]), 1);
    S02 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 *  2]), 1);
    S03 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 *  3]), 1);
    S04 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 *  4]), 1);
    S05 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 *  5]), 1);
    S06 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 *  6]), 1);
    S07 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 *  7]), 1);
    S08 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 *  8]), 1);
    S09 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 *  9]), 1);
    S10 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 10]), 1);
    S11 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 11]), 1);
    S12 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 12]), 1);
    S13 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 13]), 1);
    S14 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 14]), 1);
    S15 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 15]), 1);
    S16 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 16]), 1);
    S17 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 17]), 1);
    S18 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 18]), 1);
    S19 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 19]), 1);
    S20 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 20]), 1);
    S21 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 21]), 1);
    S22 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 22]), 1);
    S23 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 23]), 1);
    S24 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 24]), 1);
    S25 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 25]), 1);
    S26 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 26]), 1);
    S27 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 27]), 1);
    S28 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 28]), 1);
    S29 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 29]), 1);
    S30 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 30]), 1);
    S31 = _mm_srai_epi16(_mm_loadu_si128((__m128i*)&coeff[8 * 31]), 1);

    /*--vertical transform--*/
    S32 = _mm_srai_epi16(_mm_add_epi16(S00, S01), 1);
    S33 = _mm_srai_epi16(_mm_add_epi16(S01, S02), 1);
    S34 = _mm_srai_epi16(_mm_add_epi16(S02, S03), 1);
    S35 = _mm_srai_epi16(_mm_add_epi16(S03, S04), 1);
    S36 = _mm_srai_epi16(_mm_add_epi16(S04, S05), 1);
    S37 = _mm_srai_epi16(_mm_add_epi16(S05, S06), 1);
    S38 = _mm_srai_epi16(_mm_add_epi16(S06, S07), 1);
    S39 = _mm_srai_epi16(_mm_add_epi16(S07, S08), 1);
    S40 = _mm_srai_epi16(_mm_add_epi16(S08, S09), 1);
    S41 = _mm_srai_epi16(_mm_add_epi16(S09, S10), 1);
    S42 = _mm_srai_epi16(_mm_add_epi16(S10, S11), 1);
    S43 = _mm_srai_epi16(_mm_add_epi16(S11, S12), 1);
    S44 = _mm_srai_epi16(_mm_add_epi16(S12, S13), 1);
    S45 = _mm_srai_epi16(_mm_add_epi16(S13, S14), 1);
    S46 = _mm_srai_epi16(_mm_add_epi16(S14, S15), 1);
    S47 = _mm_srai_epi16(_mm_add_epi16(S15, S16), 1);
    S48 = _mm_srai_epi16(_mm_add_epi16(S16, S17), 1);
    S49 = _mm_srai_epi16(_mm_add_epi16(S17, S18), 1);
    S50 = _mm_srai_epi16(_mm_add_epi16(S18, S19), 1);
    S51 = _mm_srai_epi16(_mm_add_epi16(S19, S20), 1);
    S52 = _mm_srai_epi16(_mm_add_epi16(S20, S21), 1);
    S53 = _mm_srai_epi16(_mm_add_epi16(S21, S22), 1);
    S54 = _mm_srai_epi16(_mm_add_epi16(S22, S23), 1);
    S55 = _mm_srai_epi16(_mm_add_epi16(S23, S24), 1);
    S56 = _mm_srai_epi16(_mm_add_epi16(S24, S25), 1);
    S57 = _mm_srai_epi16(_mm_add_epi16(S25, S26), 1);
    S58 = _mm_srai_epi16(_mm_add_epi16(S26, S27), 1);
    S59 = _mm_srai_epi16(_mm_add_epi16(S27, S28), 1);
    S60 = _mm_srai_epi16(_mm_add_epi16(S28, S29), 1);
    S61 = _mm_srai_epi16(_mm_add_epi16(S29, S30), 1);
    S62 = _mm_srai_epi16(_mm_add_epi16(S30, S31), 1);
    S63 = _mm_srai_epi16(_mm_add_epi16(S31, S31), 1);

    /*--transposition--*/
    //8x64 -> 64x8
    TRANSPOSE_8x8_16BIT(S00, S32, S01, S33, S02, S34, S03, S35, T00[0], T01[0], T02[0], T03[0], T04[0], T05[0], T06[0], T07[0]);
    TRANSPOSE_8x8_16BIT(S04, S36, S05, S37, S06, S38, S07, S39, T00[1], T01[1], T02[1], T03[1], T04[1], T05[1], T06[1], T07[1]);
    TRANSPOSE_8x8_16BIT(S08, S40, S09, S41, S10, S42, S11, S43, T00[2], T01[2], T02[2], T03[2], T04[2], T05[2], T06[2], T07[2]);
    TRANSPOSE_8x8_16BIT(S12, S44, S13, S45, S14, S46, S15, S47, T00[3], T01[3], T02[3], T03[3], T04[3], T05[3], T06[3], T07[3]);
    TRANSPOSE_8x8_16BIT(S16, S48, S17, S49, S18, S50, S19, S51, T00[4], T01[4], T02[4], T03[4], T04[4], T05[4], T06[4], T07[4]);
    TRANSPOSE_8x8_16BIT(S20, S52, S21, S53, S22, S54, S23, S55, T00[5], T01[5], T02[5], T03[5], T04[5], T05[5], T06[5], T07[5]);
    TRANSPOSE_8x8_16BIT(S24, S56, S25, S57, S26, S58, S27, S59, T00[6], T01[6], T02[6], T03[6], T04[6], T05[6], T06[6], T07[6]);
    TRANSPOSE_8x8_16BIT(S28, S60, S29, S61, S30, S62, S31, S63, T00[7], T01[7], T02[7], T03[7], T04[7], T05[7], T06[7], T07[7]);

    /*--horizontal transform--*/
    for (i = 0; i < 8; i++) {
        T08[i] = _mm_srai_epi16(_mm_add_epi16(T00[i], T01[i]), 1);
        T09[i] = _mm_srai_epi16(_mm_add_epi16(T01[i], T02[i]), 1);
        T10[i] = _mm_srai_epi16(_mm_add_epi16(T02[i], T03[i]), 1);
        T11[i] = _mm_srai_epi16(_mm_add_epi16(T03[i], T04[i]), 1);
        T12[i] = _mm_srai_epi16(_mm_add_epi16(T04[i], T05[i]), 1);
        T13[i] = _mm_srai_epi16(_mm_add_epi16(T05[i], T06[i]), 1);
        T14[i] = _mm_srai_epi16(_mm_add_epi16(T06[i], T07[i]), 1);
        T15[i] = _mm_srai_epi16(_mm_add_epi16(T07[i], T07[i]), 1);
    }

    /*--transposition--*/
    //64x16 -> 16x64
    TRANSPOSE_8x8_16BIT(T00[0], T08[0], T01[0], T09[0], T02[0], T10[0], T03[0], T11[0], V00[0], V01[0], V02[0], V03[0], V04[0], V05[0], V06[0], V07[0]);
    TRANSPOSE_8x8_16BIT(T00[1], T08[1], T01[1], T09[1], T02[1], T10[1], T03[1], T11[1], V08[0], V09[0], V10[0], V11[0], V12[0], V13[0], V14[0], V15[0]);
    TRANSPOSE_8x8_16BIT(T00[2], T08[2], T01[2], T09[2], T02[2], T10[2], T03[2], T11[2], V16[0], V17[0], V18[0], V19[0], V20[0], V21[0], V22[0], V23[0]);
    TRANSPOSE_8x8_16BIT(T00[3], T08[3], T01[3], T09[3], T02[3], T10[3], T03[3], T11[3], V24[0], V25[0], V26[0], V27[0], V28[0], V29[0], V30[0], V31[0]);
    TRANSPOSE_8x8_16BIT(T00[4], T08[4], T01[4], T09[4], T02[4], T10[4], T03[4], T11[4], V32[0], V33[0], V34[0], V35[0], V36[0], V37[0], V38[0], V39[0]);
    TRANSPOSE_8x8_16BIT(T00[5], T08[5], T01[5], T09[5], T02[5], T10[5], T03[5], T11[5], V40[0], V41[0], V42[0], V43[0], V44[0], V45[0], V46[0], V47[0]);
    TRANSPOSE_8x8_16BIT(T00[6], T08[6], T01[6], T09[6], T02[6], T10[6], T03[6], T11[6], V48[0], V49[0], V50[0], V51[0], V52[0], V53[0], V54[0], V55[0]);
    TRANSPOSE_8x8_16BIT(T00[7], T08[7], T01[7], T09[7], T02[7], T10[7], T03[7], T11[7], V56[0], V57[0], V58[0], V59[0], V60[0], V61[0], V62[0], V63[0]);

    TRANSPOSE_8x8_16BIT(T04[0], T12[0], T05[0], T13[0], T06[0], T14[0], T07[0], T15[0], V00[1], V01[1], V02[1], V03[1], V04[1], V05[1], V06[1], V07[1]);
    TRANSPOSE_8x8_16BIT(T04[1], T12[1], T05[1], T13[1], T06[1], T14[1], T07[1], T15[1], V08[1], V09[1], V10[1], V11[1], V12[1], V13[1], V14[1], V15[1]);
    TRANSPOSE_8x8_16BIT(T04[2], T12[2], T05[2], T13[2], T06[2], T14[2], T07[2], T15[2], V16[1], V17[1], V18[1], V19[1], V20[1], V21[1], V22[1], V23[1]);
    TRANSPOSE_8x8_16BIT(T04[3], T12[3], T05[3], T13[3], T06[3], T14[3], T07[3], T15[3], V24[1], V25[1], V26[1], V27[1], V28[1], V29[1], V30[1], V31[1]);
    TRANSPOSE_8x8_16BIT(T04[4], T12[4], T05[4], T13[4], T06[4], T14[4], T07[4], T15[4], V32[1], V33[1], V34[1], V35[1], V36[1], V37[1], V38[1], V39[1]);
    TRANSPOSE_8x8_16BIT(T04[5], T12[5], T05[5], T13[5], T06[5], T14[5], T07[5], T15[5], V40[1], V41[1], V42[1], V43[1], V44[1], V45[1], V46[1], V47[1]);
    TRANSPOSE_8x8_16BIT(T04[6], T12[6], T05[6], T13[6], T06[6], T14[6], T07[6], T15[6], V48[1], V49[1], V50[1], V51[1], V52[1], V53[1], V54[1], V55[1]);
    TRANSPOSE_8x8_16BIT(T04[7], T12[7], T05[7], T13[7], T06[7], T14[7], T07[7], T15[7], V56[1], V57[1], V58[1], V59[1], V60[1], V61[1], V62[1], V63[1]);

    /*--Store--*/
    //16x64
    for (i = 0; i < 2; i++) {
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 *  0], V00[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 *  1], V01[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 *  2], V02[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 *  3], V03[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 *  4], V04[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 *  5], V05[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 *  6], V06[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 *  7], V07[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 *  8], V08[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 *  9], V09[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 10], V10[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 11], V11[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 12], V12[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 13], V13[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 14], V14[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 15], V15[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 16], V16[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 17], V17[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 18], V18[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 19], V19[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 20], V20[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 21], V21[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 22], V22[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 23], V23[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 24], V24[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 25], V25[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 26], V26[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 27], V27[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 28], V28[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 29], V29[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 30], V30[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 31], V31[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 32], V32[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 33], V33[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 34], V34[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 35], V35[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 36], V36[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 37], V37[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 38], V38[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 39], V39[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 40], V40[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 41], V41[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 42], V42[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 43], V43[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 44], V44[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 45], V45[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 46], V46[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 47], V47[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 48], V48[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 49], V49[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 50], V50[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 51], V51[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 52], V52[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 53], V53[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 54], V54[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 55], V55[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 56], V56[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 57], V57[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 58], V58[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 59], V59[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 60], V60[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 61], V61[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 62], V62[i]);
        _mm_storeu_si128((__m128i*)&coeff[8 * i + 16 * 63], V63[i]);
    }
}

/* ---------------------------------------------------------------------------
 */
void idct_c_64x64_sse128(const coeff_t *src, coeff_t *dst, int i_dst)
{
    UNUSED_PARAMETER(i_dst);
    idct_c_32x32_sse128(src, dst, 32 | 0x01); /* 32x32 idct */
    inv_wavelet_64x64_sse128(dst);
}

/* ---------------------------------------------------------------------------
 */
void idct_c_64x16_sse128(const coeff_t *src, coeff_t *dst, int i_dst)
{
    UNUSED_PARAMETER(i_dst);
    idct_c_32x8_sse128(src, dst, 32 | 0x01);
    inv_wavelet_64x16_sse128(dst);
}

/* ---------------------------------------------------------------------------
 */
void idct_c_16x64_sse128(const coeff_t *src, coeff_t *dst, int i_dst)
{
    UNUSED_PARAMETER(i_dst);
    idct_c_8x32_sse128(src, dst, 8 | 0x01);
    inv_wavelet_16x64_sse128(dst);
}
