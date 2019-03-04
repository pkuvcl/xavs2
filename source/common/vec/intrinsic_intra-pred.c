/*
 * intrinsic_intra-pred.c
 *
 * Description of this file:
 *    SSE assembly functions of Intra-Prediction module of the xavs2 library
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

#include "../avs2_defs.h"
#include "../basic_types.h"
#include "intrinsic.h"
#include <string.h>
#include <mmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>


static ALIGN16(int8_t tab_coeff_mode_5[8][16]) = {
    { 20, 52, 44, 12, 20, 52, 44, 12, 20, 52, 44, 12, 20, 52, 44, 12 },
    { 8, 40, 56, 24, 8, 40, 56, 24, 8, 40, 56, 24, 8, 40, 56, 24 },
    { 28, 60, 36, 4, 28, 60, 36, 4, 28, 60, 36, 4, 28, 60, 36, 4 },
    { 16, 48, 48, 16, 16, 48, 48, 16, 16, 48, 48, 16, 16, 48, 48, 16 },
    { 4, 36, 60, 28, 4, 36, 60, 28, 4, 36, 60, 28, 4, 36, 60, 28 },
    { 24, 56, 40, 8, 24, 56, 40, 8, 24, 56, 40, 8, 24, 56, 40, 8 },
    { 12, 44, 52, 20, 12, 44, 52, 20, 12, 44, 52, 20, 12, 44, 52, 20 },
    { 32, 64, 32, 0, 32, 64, 32, 0, 32, 64, 32, 0, 32, 64, 32, 0 }
};
static uint8_t tab_idx_mode_5[64] = {
    1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 22, 23, 24, 26, 27, 28, 30, 31,
    33, 34, 35, 37, 38, 39, 41, 42, 44, 45, 46, 48, 49, 50, 52, 53, 55, 56, 57, 59, 60,
    61, 63, 64, 66, 67, 68, 70, 71, 72, 74, 75, 77, 78, 79, 81, 82, 83, 85, 86, 88
};


/* ---------------------------------------------------------------------------
 */
void intra_pred_ver_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int y;
    pel_t *rpSrc = src + 1;
    __m128i T1, T2, T3, T4;

    UNUSED_PARAMETER(dir_mode);

    switch (bsx) {
    case 4:
        for (y = 0; y < bsy; y += 2) {
            CP32(dst, rpSrc);
            CP32(dst + i_dst, rpSrc);
            dst += i_dst << 1;
        }
        break;
    case 8:
        for (y = 0; y < bsy; y += 2) {
            CP64(dst, rpSrc);
            CP64(dst + i_dst, rpSrc);
            dst += i_dst << 1;
        }
        break;
    case 16:
        T1 = _mm_loadu_si128((__m128i*)rpSrc);
        for (y = 0; y < bsy; y++) {
            _mm_storeu_si128((__m128i*)(dst), T1);
            dst += i_dst;
        }
        break;
    case 32:
        T1 = _mm_loadu_si128((__m128i*)(rpSrc + 0));
        T2 = _mm_loadu_si128((__m128i*)(rpSrc + 16));
        for (y = 0; y < bsy; y++) {
            _mm_storeu_si128((__m128i*)(dst + 0), T1);
            _mm_storeu_si128((__m128i*)(dst + 16), T2);
            dst += i_dst;
        }
        break;
    case 64:
        T1 = _mm_loadu_si128((__m128i*)(rpSrc + 0));
        T2 = _mm_loadu_si128((__m128i*)(rpSrc + 16));
        T3 = _mm_loadu_si128((__m128i*)(rpSrc + 32));
        T4 = _mm_loadu_si128((__m128i*)(rpSrc + 48));
        for (y = 0; y < bsy; y++) {
            _mm_storeu_si128((__m128i*)(dst + 0), T1);
            _mm_storeu_si128((__m128i*)(dst + 16), T2);
            _mm_storeu_si128((__m128i*)(dst + 32), T3);
            _mm_storeu_si128((__m128i*)(dst + 48), T4);
            dst += i_dst;
        }
        break;
    default:
        assert(0);
        break;
    }
}


/* ---------------------------------------------------------------------------
 */
void intra_pred_hor_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int y;
    pel_t *rpSrc = src - 1;
    __m128i T;

    UNUSED_PARAMETER(dir_mode);

    switch (bsx) {
    case 4:
        for (y = 0; y < bsy; y++) {
            M32(dst) = 0x01010101 * rpSrc[-y];
            dst += i_dst;
        }
        break;
    case 8:
        for (y = 0; y < bsy; y++) {
            M64(dst) = 0x0101010101010101 * rpSrc[-y];
            dst += i_dst;
        }
        break;
    case 16:
        for (y = 0; y < bsy; y++) {
            T = _mm_set1_epi8((char)rpSrc[-y]);
            _mm_storeu_si128((__m128i*)(dst), T);
            dst += i_dst;
        }
        break;
    case 32:
        for (y = 0; y < bsy; y++) {
            T = _mm_set1_epi8((char)rpSrc[-y]);
            _mm_storeu_si128((__m128i*)(dst + 0), T);
            _mm_storeu_si128((__m128i*)(dst + 16), T);
            dst += i_dst;
        }
        break;
    case 64:
        for (y = 0; y < bsy; y++) {
            T = _mm_set1_epi8((char)rpSrc[-y]);
            _mm_storeu_si128((__m128i*)(dst + 0), T);
            _mm_storeu_si128((__m128i*)(dst + 16), T);
            _mm_storeu_si128((__m128i*)(dst + 32), T);
            _mm_storeu_si128((__m128i*)(dst + 48), T);
            dst += i_dst;
        }
        break;
    default:
        assert(0);
        break;
    }

}

/* ---------------------------------------------------------------------------
 */
void intra_pred_dc_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int avail_above = dir_mode >> 8;
    int avail_left = dir_mode & 0xFF;
    int dc_value;
    int sum_above = 0;
    int sum_left = 0;
    int x, y;
    pel_t *p_src;

    __m128i zero = _mm_setzero_si128();
    __m128i S0;
    __m128i p00, p10, p20, p30;

    /* sum of left samples */
    // for (y = 0; y < bsy; y++)  dc_value += p_src[-y];
    p_src = src - bsy;
    if (bsy == 4) {
        sum_left += p_src[0] + p_src[1] + p_src[2] + p_src[3];
    } else if (bsy == 8) {
        S0 = _mm_loadu_si128((__m128i*)(p_src));
        p00 = _mm_unpacklo_epi8(S0, zero);
        p10 = _mm_srli_si128(p00, 8);
        p00 = _mm_add_epi16(p00, p10);
        sum_left += M128_U16(p00, 0) + M128_U16(p00, 1) + M128_U16(p00, 2) + M128_U16(p00, 3);
    } else {
        p30 = zero;
        for (y = 0; y < bsy - 8; y += 16, p_src += 16) {
            S0 = _mm_loadu_si128((__m128i*)(p_src));
            p00 = _mm_unpacklo_epi8(S0, zero);
            p10 = _mm_unpackhi_epi8(S0, zero);
            p20 = _mm_add_epi16(p00, p10);
            p30 = _mm_add_epi16(p30, p20);
        }
        p00 = _mm_srli_si128(p30, 8);
        p00 = _mm_add_epi16(p30, p00);
        sum_left += M128_U16(p00, 0) + M128_U16(p00, 1) + M128_U16(p00, 2) + M128_U16(p00, 3);
    }

    /* sum of above samples */
    //for (x = 0; x < bsx; x++)  dc_value += p_src[x];
    p_src = src + 1;
    if (bsx == 4) {
        sum_above += p_src[0] + p_src[1] + p_src[2] + p_src[3];
    } else if (bsx == 8) {
        S0 = _mm_loadu_si128((__m128i*)(p_src));
        p00 = _mm_unpacklo_epi8(S0, zero);
        p10 = _mm_srli_si128(p00, 8);
        p00 = _mm_add_epi16(p00, p10);
        sum_above += M128_U16(p00, 0) + M128_U16(p00, 1) + M128_U16(p00, 2) + M128_U16(p00, 3);
    } else {
        p30 = zero;
        for (x = 0; x < bsx - 8; x += 16, p_src += 16) {
            S0 = _mm_loadu_si128((__m128i*)(p_src));
            p00 = _mm_unpacklo_epi8(S0, zero);
            p10 = _mm_unpackhi_epi8(S0, zero);
            p20 = _mm_add_epi16(p00, p10);
            p30 = _mm_add_epi16(p30, p20);
        }
        p00 = _mm_srli_si128(p30, 8);
        p00 = _mm_add_epi16(p30, p00);
        sum_above += M128_U16(p00, 0) + M128_U16(p00, 1) + M128_U16(p00, 2) + M128_U16(p00, 3);
    }

    if (avail_left && avail_above) {
        x = bsx + bsy;
        dc_value = ((sum_above + sum_left + (x >> 1)) * (512 / x)) >> 9;
    } else if (avail_left) {
        dc_value = (sum_left + (bsy >> 1)) >> xavs2_log2u(bsy);
    } else if (avail_above) {
        dc_value = (sum_above + (bsx >> 1)) >> xavs2_log2u(bsx);
    } else {
        dc_value = g_dc_value;
    }

    p00 = _mm_set1_epi8((pel_t)dc_value);
    for (y = 0; y < bsy; y++) {
        if (bsx == 8) {
            _mm_storel_epi64((__m128i*)dst, p00);
        } else if (bsx == 4) {
            *(int*)(dst) = _mm_cvtsi128_si32(p00);
        } else {
            for (x = 0; x < bsx - 8; x += 16) {
                _mm_storeu_si128((__m128i*)(dst + x), p00);
            }
        }
        dst += i_dst;
    }

}

/* ---------------------------------------------------------------------------
*/
void intra_pred_plane_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    pel_t  *rpSrc;
    int iH = 0;
    int iV = 0;
    int iA, iB, iC;
    int x, y;
    int iW2 = bsx >> 1;
    int iH2 = bsy >> 1;
    int ib_mult[5] = { 13, 17, 5, 11, 23 };
    int ib_shift[5] = { 7, 10, 11, 15, 19 };
    int im_h = ib_mult[tab_log2[bsx] - 2];
    int is_h = ib_shift[tab_log2[bsx] - 2];
    int im_v = ib_mult[tab_log2[bsy] - 2];
    int is_v = ib_shift[tab_log2[bsy] - 2];

    int iTmp;

    UNUSED_PARAMETER(dir_mode);

    rpSrc = src + iW2;
    for (x = 1; x < iW2 + 1; x++) {
        iH += x * (rpSrc[x] - rpSrc[-x]);
    }

    rpSrc = src - iH2;
    for (y = 1; y < iH2 + 1; y++) {
        iV += y * (rpSrc[-y] - rpSrc[y]);
    }

    iA = (src[-1 - (bsy - 1)] + src[1 + bsx - 1]) << 4;
    iB = ((iH << 5) * im_h + (1 << (is_h - 1))) >> is_h;
    iC = ((iV << 5) * im_v + (1 << (is_v - 1))) >> is_v;

    iTmp = iA - (iH2 - 1) * iC - (iW2 - 1) * iB + 16;

    __m128i TC, TB, TA, T_Start, T, D, D1;
    TA = _mm_set1_epi16((int16_t)iTmp);
    TB = _mm_set1_epi16((int16_t)iB);
    TC = _mm_set1_epi16((int16_t)iC);

    T_Start = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
    T_Start = _mm_mullo_epi16(TB, T_Start);
    T_Start = _mm_add_epi16(T_Start, TA);

    TB = _mm_mullo_epi16(TB, _mm_set1_epi16(8));

    if (bsx == 4) {
        for (y = 0; y < bsy; y++) {
            D = _mm_srai_epi16(T_Start, 5);
            D = _mm_packus_epi16(D, D);
            // extract low 32 bits from the packed result , and put it into a integer . (Redundant operation?)
            _mm_stream_si32((int *)dst, _mm_extract_epi32(D, 0));
            T_Start = _mm_add_epi16(T_Start, TC);
            dst += i_dst;
        }
    } else if (bsx == 8) {
        for (y = 0; y < bsy; y++) {
            D = _mm_srai_epi16(T_Start, 5);
            D = _mm_packus_epi16(D, D);
            _mm_storel_epi64((__m128i*)dst, D);
            T_Start = _mm_add_epi16(T_Start, TC);
            dst += i_dst;
        }
    } else {
        for (y = 0; y < bsy; y++) {
            T = T_Start;
            for (x = 0; x < bsx; x += 16) {
                D = _mm_srai_epi16(T, 5);
                T = _mm_add_epi16(T, TB);
                D1 = _mm_srai_epi16(T, 5);
                T = _mm_add_epi16(T, TB);
                D = _mm_packus_epi16(D, D1);
                _mm_storeu_si128((__m128i*)(dst + x), D);
            }
            T_Start = _mm_add_epi16(T_Start, TC);
            dst += i_dst;
        }
    }

}

/* ---------------------------------------------------------------------------
*/
void intra_pred_bilinear_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int x, y;
    int ishift_x = tab_log2[bsx];
    int ishift_y = tab_log2[bsy];
    int ishift = XAVS2_MIN(ishift_x, ishift_y);
    int ishift_xy = ishift_x + ishift_y + 1;
    int offset = 1 << (ishift_x + ishift_y);
    int a, b, c, w, val;
    pel_t *p;
    __m128i T, T1, T2, T3, C1, C2, ADD;
    __m128i ZERO = _mm_setzero_si128();

    ALIGN32(itr_t pTop [MAX_CU_SIZE + 32]);
    ALIGN32(itr_t pLeft[MAX_CU_SIZE + 32]);
    ALIGN32(itr_t pT   [MAX_CU_SIZE + 32]);
    ALIGN32(itr_t pL   [MAX_CU_SIZE + 32]);
    ALIGN32(itr_t wy   [MAX_CU_SIZE + 32]);

    UNUSED_PARAMETER(dir_mode);

    p = src + 1;
    for (x = 0; x < bsx; x += 16) {
        T = _mm_loadu_si128((__m128i*)(p + x));
        T1 = _mm_unpacklo_epi8(T, ZERO);
        T2 = _mm_unpackhi_epi8(T, ZERO);
        _mm_store_si128((__m128i*)(pTop + x), T1);
        _mm_store_si128((__m128i*)(pTop + x + 8), T2);
    }
    for (y = 0; y < bsy; y++) {
        pLeft[y] = src[-1 - y];
    }

    a = pTop[bsx - 1];
    b = pLeft[bsy - 1];

    if (bsx == bsy) {
        c = (a + b + 1) >> 1;
    } else {
        c = (((a << ishift_x) + (b << ishift_y)) * 13 + (1 << (ishift + 5))) >> (ishift + 6);
    }

    w = (c << 1) - a - b;

    T = _mm_set1_epi16((int16_t)b);
    for (x = 0; x < bsx; x += 8) {
        T1 = _mm_load_si128((__m128i*)(pTop + x));
        T2 = _mm_sub_epi16(T, T1);
        T1 = _mm_slli_epi16(T1, ishift_y);
        _mm_store_si128((__m128i*)(pT + x), T2);
        _mm_store_si128((__m128i*)(pTop + x), T1);
    }

    T = _mm_set1_epi16((int16_t)a);
    for (y = 0; y < bsy; y += 8) {
        T1 = _mm_load_si128((__m128i*)(pLeft + y));
        T2 = _mm_sub_epi16(T, T1);
        T1 = _mm_slli_epi16(T1, ishift_x);
        _mm_store_si128((__m128i*)(pL + y), T2);
        _mm_store_si128((__m128i*)(pLeft + y), T1);
    }

    T = _mm_set1_epi16((int16_t)w);
    T = _mm_mullo_epi16(T, _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0));
    T1 = _mm_set1_epi16((int16_t)(8 * w));

    for (y = 0; y < bsy; y += 8) {
        _mm_store_si128((__m128i*)(wy + y), T);
        T = _mm_add_epi16(T, T1);
    }

    C1 = _mm_set_epi32(3, 2, 1, 0);
    C2 = _mm_set1_epi32(4);

    if (bsx == 4) {
        __m128i pTT = _mm_loadl_epi64((__m128i*)pT);
        T = _mm_loadl_epi64((__m128i*)pTop);
        for (y = 0; y < bsy; y++) {
            int add = (pL[y] << ishift_y) + wy[y];
            ADD = _mm_set1_epi32(add);
            ADD = _mm_mullo_epi32(C1, ADD);

            val = (pLeft[y] << ishift_y) + offset + (pL[y] << ishift_y);

            ADD = _mm_add_epi32(ADD, _mm_set1_epi32(val));
            T = _mm_add_epi16(T, pTT);

            T1 = _mm_cvtepi16_epi32(T);
            T1 = _mm_slli_epi32(T1, ishift_x);

            T1 = _mm_add_epi32(T1, ADD);
            T1 = _mm_srai_epi32(T1, ishift_xy);

            T1 = _mm_packus_epi32(T1, T1);
            T1 = _mm_packus_epi16(T1, T1);

            M32(dst) = _mm_cvtsi128_si32(T1);

            dst += i_dst;
        }
    } else if (bsx == 8) {
        __m128i pTT = _mm_load_si128((__m128i*)pT);
        T = _mm_load_si128((__m128i*)pTop);
        for (y = 0; y < bsy; y++) {
            int add = (pL[y] << ishift_y) + wy[y];
            ADD = _mm_set1_epi32(add);
            T3 = _mm_mullo_epi32(C2, ADD);
            ADD = _mm_mullo_epi32(C1, ADD);

            val = (pLeft[y] << ishift_y) + offset + (pL[y] << ishift_y);

            ADD = _mm_add_epi32(ADD, _mm_set1_epi32(val));

            T = _mm_add_epi16(T, pTT);

            T1 = _mm_cvtepi16_epi32(T);
            T2 = _mm_cvtepi16_epi32(_mm_srli_si128(T, 8));
            T1 = _mm_slli_epi32(T1, ishift_x);
            T2 = _mm_slli_epi32(T2, ishift_x);

            T1 = _mm_add_epi32(T1, ADD);
            T1 = _mm_srai_epi32(T1, ishift_xy);
            ADD = _mm_add_epi32(ADD, T3);

            T2 = _mm_add_epi32(T2, ADD);
            T2 = _mm_srai_epi32(T2, ishift_xy);
            ADD = _mm_add_epi32(ADD, T3);

            T1 = _mm_packus_epi32(T1, T2);
            T1 = _mm_packus_epi16(T1, T1);

            _mm_storel_epi64((__m128i*)dst, T1);

            dst += i_dst;
        }
    } else {
        __m128i TT[16];
        __m128i PTT[16];
        for (x = 0; x < bsx; x += 8) {
            int idx = x >> 2;
            __m128i M0 = _mm_load_si128((__m128i*)(pTop + x));
            __m128i M1 = _mm_load_si128((__m128i*)(pT + x));
            TT[idx] = _mm_unpacklo_epi16(M0, ZERO);
            TT[idx + 1] = _mm_unpackhi_epi16(M0, ZERO);
            PTT[idx] = _mm_cvtepi16_epi32(M1);
            PTT[idx + 1] = _mm_cvtepi16_epi32(_mm_srli_si128(M1, 8));
        }
        for (y = 0; y < bsy; y++) {
            int add = (pL[y] << ishift_y) + wy[y];
            ADD = _mm_set1_epi32(add);
            T3 = _mm_mullo_epi32(C2, ADD);
            ADD = _mm_mullo_epi32(C1, ADD);

            val = (pLeft[y] << ishift_y) + offset + (pL[y] << ishift_y);

            ADD = _mm_add_epi32(ADD, _mm_set1_epi32(val));

            for (x = 0; x < bsx; x += 8) {
                int idx = x >> 2;
                TT[idx] = _mm_add_epi32(TT[idx], PTT[idx]);
                TT[idx + 1] = _mm_add_epi32(TT[idx + 1], PTT[idx + 1]);

                T1 = _mm_slli_epi32(TT[idx], ishift_x);
                T2 = _mm_slli_epi32(TT[idx + 1], ishift_x);

                T1 = _mm_add_epi32(T1, ADD);
                T1 = _mm_srai_epi32(T1, ishift_xy);
                ADD = _mm_add_epi32(ADD, T3);

                T2 = _mm_add_epi32(T2, ADD);
                T2 = _mm_srai_epi32(T2, ishift_xy);
                ADD = _mm_add_epi32(ADD, T3);

                T1 = _mm_packus_epi32(T1, T2);
                T1 = _mm_packus_epi16(T1, T1);

                _mm_storel_epi64((__m128i*)(dst + x), T1);
            }
            dst += i_dst;
        }
    }

}


/* ---------------------------------------------------------------------------
 */
void intra_pred_ang_x_3_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    __m128i zero = _mm_setzero_si128();
    __m128i coeff2 = _mm_set1_epi16(2);
    __m128i coeff3 = _mm_set1_epi16(3);
    __m128i coeff4 = _mm_set1_epi16(4);
    __m128i coeff5 = _mm_set1_epi16(5);
    __m128i coeff7 = _mm_set1_epi16(7);
    __m128i coeff8 = _mm_set1_epi16(8);

    pel_t *dst1 = dst;
    pel_t *dst2 = dst1 + i_dst;
    pel_t *dst3 = dst2 + i_dst;
    pel_t *dst4 = dst3 + i_dst;

    UNUSED_PARAMETER(dir_mode);

    if ((bsy > 4) && (bsx > 8)) {
        ALIGN16(pel_t first_line[(64 + 176 + 16) << 2]);
        int line_size = bsx + (((bsy - 4) * 11) >> 2);
        int aligned_line_size = 64 + 176 + 16;
        int i;
        pel_t *pfirst[4];

        pfirst[0] = first_line;
        pfirst[1] = pfirst[0] + aligned_line_size;
        pfirst[2] = pfirst[1] + aligned_line_size;
        pfirst[3] = pfirst[2] + aligned_line_size;

        for (i = 0; i < line_size - 8; i += 16, src += 16) {
            __m128i p00, p10, p20, p30;
            __m128i p01, p11, p21, p31;

            __m128i SS2 = _mm_loadu_si128((__m128i*)(src + 2));
            __m128i L2 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L3 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L4 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L5 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L6 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L7 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L8 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L9 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L10 = _mm_unpacklo_epi8(SS2, zero);
            __m128i H2 = L10;

            __m128i SS11 = _mm_loadu_si128((__m128i*)(src + 11));
            __m128i L11 = _mm_unpacklo_epi8(SS11, zero);
            __m128i H3 = L11;
            SS11 = _mm_srli_si128(SS11, 1);
            __m128i L12 = _mm_unpacklo_epi8(SS11, zero);
            __m128i H4 = L12;
            SS11 = _mm_srli_si128(SS11, 1);
            __m128i L13 = _mm_unpacklo_epi8(SS11, zero);
            __m128i H5 = L13;

            SS11 = _mm_srli_si128(SS11, 1);
            __m128i H6 = _mm_unpacklo_epi8(SS11, zero);
            SS11 = _mm_srli_si128(SS11, 1);
            __m128i H7 = _mm_unpacklo_epi8(SS11, zero);
            SS11 = _mm_srli_si128(SS11, 1);
            __m128i H8 = _mm_unpacklo_epi8(SS11, zero);
            SS11 = _mm_srli_si128(SS11, 1);
            __m128i H9 = _mm_unpacklo_epi8(SS11, zero);
            SS11 = _mm_srli_si128(SS11, 1);
            __m128i H10 = _mm_unpacklo_epi8(SS11, zero);
            SS11 = _mm_srli_si128(SS11, 1);
            __m128i H11 = _mm_unpacklo_epi8(SS11, zero);

            __m128i SS20 = _mm_loadu_si128((__m128i*)(src + 20));
            __m128i H12 = _mm_unpacklo_epi8(SS20, zero);
            SS20 = _mm_srli_si128(SS20, 1);
            __m128i H13 = _mm_unpacklo_epi8(SS20, zero);

            p00 = _mm_add_epi16(L2, coeff8);
            p10 = _mm_mullo_epi16(L3, coeff5);
            p20 = _mm_mullo_epi16(L4, coeff7);
            p30 = _mm_mullo_epi16(L5, coeff3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p01 = _mm_add_epi16(H2, coeff8);
            p11 = _mm_mullo_epi16(H3, coeff5);
            p21 = _mm_mullo_epi16(H4, coeff7);
            p31 = _mm_mullo_epi16(H5, coeff3);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)&pfirst[0][i], p00);

            p00 = _mm_add_epi16(L5, L8);
            p10 = _mm_add_epi16(L6, L7);
            p10 = _mm_mullo_epi16(p10, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_srli_epi16(p00, 3);

            p01 = _mm_add_epi16(H5, H8);
            p11 = _mm_add_epi16(H6, H7);
            p11 = _mm_mullo_epi16(p11, coeff3);
            p01 = _mm_add_epi16(p01, coeff4);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_srli_epi16(p01, 3);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)&pfirst[1][i], p00);

            p00 = _mm_mullo_epi16(L8, coeff3);
            p10 = _mm_mullo_epi16(L9, coeff7);
            p20 = _mm_mullo_epi16(L10, coeff5);
            p30 = _mm_add_epi16(L11, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p01 = _mm_mullo_epi16(H8, coeff3);
            p11 = _mm_mullo_epi16(H9, coeff7);
            p21 = _mm_mullo_epi16(H10, coeff5);
            p31 = _mm_add_epi16(H11, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)&pfirst[2][i], p00);

            p00 = _mm_add_epi16(L11, L13);
            p10 = _mm_mullo_epi16(L12, coeff2);
            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_srli_epi16(p00, 2);

            p01 = _mm_add_epi16(H11, H13);
            p11 = _mm_mullo_epi16(H12, coeff2);
            p01 = _mm_add_epi16(p01, coeff2);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_srli_epi16(p01, 2);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)&pfirst[3][i], p00);
        }
        if (i < line_size) {
            __m128i p00, p10, p20, p30;
            __m128i SS2 = _mm_loadu_si128((__m128i*)(src + 2));
            __m128i L2 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L3 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L4 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L5 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L6 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L7 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L8 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L9 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L10 = _mm_unpacklo_epi8(SS2, zero);

            __m128i SS11 = _mm_loadu_si128((__m128i*)(src + 11));
            __m128i L11 = _mm_unpacklo_epi8(SS11, zero);
            SS11 = _mm_srli_si128(SS11, 1);
            __m128i L12 = _mm_unpacklo_epi8(SS11, zero);
            SS11 = _mm_srli_si128(SS11, 1);
            __m128i L13 = _mm_unpacklo_epi8(SS11, zero);

            p00 = _mm_add_epi16(L2, coeff8);
            p10 = _mm_mullo_epi16(L3, coeff5);
            p20 = _mm_mullo_epi16(L4, coeff7);
            p30 = _mm_mullo_epi16(L5, coeff3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[0][i], p00);

            p00 = _mm_add_epi16(L5, L8);
            p10 = _mm_add_epi16(L6, L7);
            p10 = _mm_mullo_epi16(p10, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_srli_epi16(p00, 3);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[1][i], p00);

            p00 = _mm_mullo_epi16(L8, coeff3);
            p10 = _mm_mullo_epi16(L9, coeff7);
            p20 = _mm_mullo_epi16(L10, coeff5);
            p30 = _mm_add_epi16(L11, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[2][i], p00);

            p00 = _mm_add_epi16(L11, L13);
            p10 = _mm_mullo_epi16(L12, coeff2);
            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_srli_epi16(p00, 2);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[3][i], p00);
        }

        bsy >>= 2;
        for (i = 0; i < bsy; i++) {
            memcpy(dst1, pfirst[0] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst2, pfirst[1] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst3, pfirst[2] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst4, pfirst[3] + i * 11, bsx * sizeof(pel_t));
            dst1 = dst4 + i_dst;
            dst2 = dst1 + i_dst;
            dst3 = dst2 + i_dst;
            dst4 = dst3 + i_dst;
        }
    } else if (bsx == 16) {
        __m128i p00, p10, p20, p30;
        __m128i p01, p11, p21, p31;

        __m128i SS2 = _mm_loadu_si128((__m128i*)(src + 2));
        __m128i L2 = _mm_unpacklo_epi8(SS2, zero);
        SS2 = _mm_srli_si128(SS2, 1);
        __m128i L3 = _mm_unpacklo_epi8(SS2, zero);
        SS2 = _mm_srli_si128(SS2, 1);
        __m128i L4 = _mm_unpacklo_epi8(SS2, zero);
        SS2 = _mm_srli_si128(SS2, 1);
        __m128i L5 = _mm_unpacklo_epi8(SS2, zero);
        SS2 = _mm_srli_si128(SS2, 1);
        __m128i L6 = _mm_unpacklo_epi8(SS2, zero);
        SS2 = _mm_srli_si128(SS2, 1);
        __m128i L7 = _mm_unpacklo_epi8(SS2, zero);
        SS2 = _mm_srli_si128(SS2, 1);
        __m128i L8 = _mm_unpacklo_epi8(SS2, zero);
        SS2 = _mm_srli_si128(SS2, 1);
        __m128i L9 = _mm_unpacklo_epi8(SS2, zero);
        SS2 = _mm_srli_si128(SS2, 1);
        __m128i L10 = _mm_unpacklo_epi8(SS2, zero);
        __m128i H2 = L10;

        __m128i SS11 = _mm_loadu_si128((__m128i*)(src + 11));
        __m128i L11 = _mm_unpacklo_epi8(SS11, zero);
        __m128i H3 = L11;
        SS11 = _mm_srli_si128(SS11, 1);
        __m128i L12 = _mm_unpacklo_epi8(SS11, zero);
        __m128i H4 = L12;
        SS11 = _mm_srli_si128(SS11, 1);
        __m128i L13 = _mm_unpacklo_epi8(SS11, zero);
        __m128i H5 = L13;

        SS11 = _mm_srli_si128(SS11, 1);
        __m128i H6 = _mm_unpacklo_epi8(SS11, zero);
        SS11 = _mm_srli_si128(SS11, 1);
        __m128i H7 = _mm_unpacklo_epi8(SS11, zero);
        SS11 = _mm_srli_si128(SS11, 1);
        __m128i H8 = _mm_unpacklo_epi8(SS11, zero);
        SS11 = _mm_srli_si128(SS11, 1);
        __m128i H9 = _mm_unpacklo_epi8(SS11, zero);
        SS11 = _mm_srli_si128(SS11, 1);
        __m128i H10 = _mm_unpacklo_epi8(SS11, zero);
        SS11 = _mm_srli_si128(SS11, 1);
        __m128i H11 = _mm_unpacklo_epi8(SS11, zero);

        __m128i SS20 = _mm_loadu_si128((__m128i*)(src + 20));
        __m128i H12 = _mm_unpacklo_epi8(SS20, zero);
        SS20 = _mm_srli_si128(SS20, 1);
        __m128i H13 = _mm_unpacklo_epi8(SS20, zero);

        p00 = _mm_add_epi16(L2, coeff8);
        p10 = _mm_mullo_epi16(L3, coeff5);
        p20 = _mm_mullo_epi16(L4, coeff7);
        p30 = _mm_mullo_epi16(L5, coeff3);
        p00 = _mm_add_epi16(p00, p10);
        p00 = _mm_add_epi16(p00, p20);
        p00 = _mm_add_epi16(p00, p30);
        p00 = _mm_srli_epi16(p00, 4);

        p01 = _mm_add_epi16(H2, coeff8);
        p11 = _mm_mullo_epi16(H3, coeff5);
        p21 = _mm_mullo_epi16(H4, coeff7);
        p31 = _mm_mullo_epi16(H5, coeff3);
        p01 = _mm_add_epi16(p01, p11);
        p01 = _mm_add_epi16(p01, p21);
        p01 = _mm_add_epi16(p01, p31);
        p01 = _mm_srli_epi16(p01, 4);

        p00 = _mm_packus_epi16(p00, p01);
        _mm_store_si128((__m128i*)dst1, p00);

        p00 = _mm_add_epi16(L5, L8);
        p10 = _mm_add_epi16(L6, L7);
        p10 = _mm_mullo_epi16(p10, coeff3);
        p00 = _mm_add_epi16(p00, coeff4);
        p00 = _mm_add_epi16(p00, p10);
        p00 = _mm_srli_epi16(p00, 3);

        p01 = _mm_add_epi16(H5, H8);
        p11 = _mm_add_epi16(H6, H7);
        p11 = _mm_mullo_epi16(p11, coeff3);
        p01 = _mm_add_epi16(p01, coeff4);
        p01 = _mm_add_epi16(p01, p11);
        p01 = _mm_srli_epi16(p01, 3);

        p00 = _mm_packus_epi16(p00, p01);
        _mm_store_si128((__m128i*)dst2, p00);

        p00 = _mm_mullo_epi16(L8, coeff3);
        p10 = _mm_mullo_epi16(L9, coeff7);
        p20 = _mm_mullo_epi16(L10, coeff5);
        p30 = _mm_add_epi16(L11, coeff8);
        p00 = _mm_add_epi16(p00, p10);
        p00 = _mm_add_epi16(p00, p20);
        p00 = _mm_add_epi16(p00, p30);
        p00 = _mm_srli_epi16(p00, 4);

        p01 = _mm_mullo_epi16(H8, coeff3);
        p11 = _mm_mullo_epi16(H9, coeff7);
        p21 = _mm_mullo_epi16(H10, coeff5);
        p31 = _mm_add_epi16(H11, coeff8);
        p01 = _mm_add_epi16(p01, p11);
        p01 = _mm_add_epi16(p01, p21);
        p01 = _mm_add_epi16(p01, p31);
        p01 = _mm_srli_epi16(p01, 4);

        p00 = _mm_packus_epi16(p00, p01);
        _mm_store_si128((__m128i*)dst3, p00);

        p00 = _mm_add_epi16(L11, L13);
        p10 = _mm_mullo_epi16(L12, coeff2);
        p00 = _mm_add_epi16(p00, coeff2);
        p00 = _mm_add_epi16(p00, p10);
        p00 = _mm_srli_epi16(p00, 2);

        p01 = _mm_add_epi16(H11, H13);
        p11 = _mm_mullo_epi16(H12, coeff2);
        p01 = _mm_add_epi16(p01, coeff2);
        p01 = _mm_add_epi16(p01, p11);
        p01 = _mm_srli_epi16(p01, 2);

        p00 = _mm_packus_epi16(p00, p01);
        _mm_store_si128((__m128i*)dst4, p00);
    } else if (bsx == 8) {
        __m128i p00, p10, p20, p30;

        __m128i SS2 = _mm_loadu_si128((__m128i*)(src + 2));
        __m128i L2 = _mm_unpacklo_epi8(SS2, zero);
        SS2 = _mm_srli_si128(SS2, 1);
        __m128i L3 = _mm_unpacklo_epi8(SS2, zero);
        SS2 = _mm_srli_si128(SS2, 1);
        __m128i L4 = _mm_unpacklo_epi8(SS2, zero);
        SS2 = _mm_srli_si128(SS2, 1);
        __m128i L5 = _mm_unpacklo_epi8(SS2, zero);
        SS2 = _mm_srli_si128(SS2, 1);
        __m128i L6 = _mm_unpacklo_epi8(SS2, zero);
        SS2 = _mm_srli_si128(SS2, 1);
        __m128i L7 = _mm_unpacklo_epi8(SS2, zero);
        SS2 = _mm_srli_si128(SS2, 1);
        __m128i L8 = _mm_unpacklo_epi8(SS2, zero);
        SS2 = _mm_srli_si128(SS2, 1);
        __m128i L9 = _mm_unpacklo_epi8(SS2, zero);
        SS2 = _mm_srli_si128(SS2, 1);
        __m128i L10 = _mm_unpacklo_epi8(SS2, zero);

        __m128i SS11 = _mm_loadu_si128((__m128i*)(src + 11));
        __m128i L11 = _mm_unpacklo_epi8(SS11, zero);
        SS11 = _mm_srli_si128(SS11, 1);
        __m128i L12 = _mm_unpacklo_epi8(SS11, zero);
        SS11 = _mm_srli_si128(SS11, 1);
        __m128i L13 = _mm_unpacklo_epi8(SS11, zero);

        p00 = _mm_add_epi16(L2, coeff8);
        p10 = _mm_mullo_epi16(L3, coeff5);
        p20 = _mm_mullo_epi16(L4, coeff7);
        p30 = _mm_mullo_epi16(L5, coeff3);
        p00 = _mm_add_epi16(p00, p10);
        p00 = _mm_add_epi16(p00, p20);
        p00 = _mm_add_epi16(p00, p30);
        p00 = _mm_srli_epi16(p00, 4);

        p00 = _mm_packus_epi16(p00, p00);
        _mm_storel_epi64((__m128i*)dst1, p00);

        p00 = _mm_add_epi16(L5, L8);
        p10 = _mm_add_epi16(L6, L7);
        p10 = _mm_mullo_epi16(p10, coeff3);
        p00 = _mm_add_epi16(p00, coeff4);
        p00 = _mm_add_epi16(p00, p10);
        p00 = _mm_srli_epi16(p00, 3);

        p00 = _mm_packus_epi16(p00, p00);
        _mm_storel_epi64((__m128i*)dst2, p00);

        p00 = _mm_mullo_epi16(L8, coeff3);
        p10 = _mm_mullo_epi16(L9, coeff7);
        p20 = _mm_mullo_epi16(L10, coeff5);
        p30 = _mm_add_epi16(L11, coeff8);
        p00 = _mm_add_epi16(p00, p10);
        p00 = _mm_add_epi16(p00, p20);
        p00 = _mm_add_epi16(p00, p30);
        p00 = _mm_srli_epi16(p00, 4);

        p00 = _mm_packus_epi16(p00, p00);
        _mm_storel_epi64((__m128i*)dst3, p00);

        p00 = _mm_add_epi16(L11, L13);
        p10 = _mm_mullo_epi16(L12, coeff2);
        p00 = _mm_add_epi16(p00, coeff2);
        p00 = _mm_add_epi16(p00, p10);
        p00 = _mm_srli_epi16(p00, 2);

        p00 = _mm_packus_epi16(p00, p00);
        _mm_storel_epi64((__m128i*)dst4, p00);
        __m128i pad1 = _mm_set1_epi8(src[16]);

        dst1 = dst4 + i_dst;
        dst2 = dst1 + i_dst;
        dst3 = dst2 + i_dst;
        dst4 = dst3 + i_dst;

        _mm_storel_epi64((__m128i*)dst1, pad1);
        _mm_storel_epi64((__m128i*)dst2, pad1);
        _mm_storel_epi64((__m128i*)dst3, pad1);
        _mm_storel_epi64((__m128i*)dst4, pad1);

        dst1[0] = (pel_t)((src[13] + 5 * src[14] + 7 * src[15] + 3 * src[16] + 8) >> 4);
        dst1[1] = (pel_t)((src[14] + 5 * src[15] + 7 * src[16] + 3 * src[17] + 8) >> 4);
        dst1[2] = (pel_t)((src[15] + 5 * src[16] + 7 * src[17] + 3 * src[18] + 8) >> 4);

        if (bsy == 32) {
            for (int i = 0; i < 6; i++) {
                dst1 = dst4 + i_dst;
                dst2 = dst1 + i_dst;
                dst3 = dst2 + i_dst;
                dst4 = dst3 + i_dst;

                _mm_storel_epi64((__m128i*)dst1, pad1);
                _mm_storel_epi64((__m128i*)dst2, pad1);
                _mm_storel_epi64((__m128i*)dst3, pad1);
                _mm_storel_epi64((__m128i*)dst4, pad1);
            }
        }
    } else {
        if (bsy == 16) {
            __m128i p00, p10, p20, p30;

            __m128i SS2 = _mm_loadu_si128((__m128i*)(src + 2));
            __m128i L2 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L3 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L4 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L5 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L6 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L7 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L8 = _mm_unpacklo_epi8(SS2, zero);

            p00 = _mm_add_epi16(L2, coeff8);
            p10 = _mm_mullo_epi16(L3, coeff5);
            p20 = _mm_mullo_epi16(L4, coeff7);
            p30 = _mm_mullo_epi16(L5, coeff3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            *((int*)(dst1)) = _mm_cvtsi128_si32(p00);

            p00 = _mm_add_epi16(L5, L8);
            p10 = _mm_add_epi16(L6, L7);
            p10 = _mm_mullo_epi16(p10, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_srli_epi16(p00, 3);

            p00 = _mm_packus_epi16(p00, p00);
            *((int*)(dst2)) = _mm_cvtsi128_si32(p00);
            __m128i pad1 = _mm_set1_epi8(src[8]);
            *((int*)(dst3)) = _mm_cvtsi128_si32(pad1);
            *((int*)(dst4)) = _mm_cvtsi128_si32(pad1);

            for (int i = 0; i < 3; i++) {
                dst1 = dst4 + i_dst;
                dst2 = dst1 + i_dst;
                dst3 = dst2 + i_dst;
                dst4 = dst3 + i_dst;

                *((int*)(dst1)) = _mm_cvtsi128_si32(pad1);
                *((int*)(dst2)) = _mm_cvtsi128_si32(pad1);
                *((int*)(dst3)) = _mm_cvtsi128_si32(pad1);
                *((int*)(dst4)) = _mm_cvtsi128_si32(pad1);
            }
        } else {
            __m128i p00, p10, p20, p30;
            __m128i SS2 = _mm_loadu_si128((__m128i*)(src + 2));
            __m128i L2 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L3 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L4 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L5 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L6 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L7 = _mm_unpacklo_epi8(SS2, zero);
            SS2 = _mm_srli_si128(SS2, 1);
            __m128i L8 = _mm_unpacklo_epi8(SS2, zero);

            p00 = _mm_add_epi16(L2, coeff8);
            p10 = _mm_mullo_epi16(L3, coeff5);
            p20 = _mm_mullo_epi16(L4, coeff7);
            p30 = _mm_mullo_epi16(L5, coeff3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            *((int*)(dst1)) = _mm_cvtsi128_si32(p00);

            p00 = _mm_add_epi16(L5, L8);
            p10 = _mm_add_epi16(L6, L7);
            p10 = _mm_mullo_epi16(p10, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_srli_epi16(p00, 3);

            p00 = _mm_packus_epi16(p00, p00);
            *((int*)(dst2)) = _mm_cvtsi128_si32(p00);

            __m128i pad1 = _mm_set1_epi8(src[8]);
            *((int*)(dst3)) = _mm_cvtsi128_si32(pad1);
            *((int*)(dst4)) = _mm_cvtsi128_si32(pad1);
        }
    }

}


/* ---------------------------------------------------------------------------
 */
void intra_pred_ang_x_4_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{

    ALIGN16(pel_t first_line[64 + 128]);
    int line_size = bsx + ((bsy - 1) << 1);
    int iHeight2 = bsy << 1;
    int i;
    __m128i zero = _mm_setzero_si128();
    __m128i offset = _mm_set1_epi16(2);

    UNUSED_PARAMETER(dir_mode);

    src += 3;
    for (i = 0; i < line_size - 8; i += 16, src += 16) {
        __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);

        __m128i H0 = _mm_unpackhi_epi8(S0, zero);
        __m128i H1 = _mm_unpackhi_epi8(S1, zero);
        __m128i H2 = _mm_unpackhi_epi8(S2, zero);

        __m128i sum1 = _mm_add_epi16(L0, L1);
        __m128i sum2 = _mm_add_epi16(L1, L2);
        __m128i sum3 = _mm_add_epi16(H0, H1);
        __m128i sum4 = _mm_add_epi16(H1, H2);

        sum1 = _mm_add_epi16(sum1, sum2);
        sum3 = _mm_add_epi16(sum3, sum4);

        sum1 = _mm_add_epi16(sum1, offset);
        sum3 = _mm_add_epi16(sum3, offset);

        sum1 = _mm_srli_epi16(sum1, 2);
        sum3 = _mm_srli_epi16(sum3, 2);

        sum1 = _mm_packus_epi16(sum1, sum3);

        _mm_store_si128((__m128i*)&first_line[i], sum1);
    }

    if (i < line_size) {
        __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);

        __m128i sum1 = _mm_add_epi16(L0, L1);
        __m128i sum2 = _mm_add_epi16(L1, L2);

        sum1 = _mm_add_epi16(sum1, sum2);
        sum1 = _mm_add_epi16(sum1, offset);
        sum1 = _mm_srli_epi16(sum1, 2);

        sum1 = _mm_packus_epi16(sum1, sum1);
        _mm_storel_epi64((__m128i*)&first_line[i], sum1);
    }


    if (bsx == bsy || bsx > 16) {
        for (i = 0; i < iHeight2; i += 2) {
            memcpy(dst, first_line + i, bsx * sizeof(pel_t));
            dst += i_dst;
        }
    } else if (bsx == 16) {
        pel_t *dst1 = dst;
        __m128i M = _mm_loadu_si128((__m128i*)&first_line[0]);
        _mm_storel_epi64((__m128i*)dst, M);
        dst += i_dst;
        M = _mm_srli_si128(M, 2);
        _mm_storel_epi64((__m128i*)dst, M);
        dst += i_dst;
        M = _mm_srli_si128(M, 2);
        _mm_storel_epi64((__m128i*)dst, M);
        dst += i_dst;
        M = _mm_srli_si128(M, 2);
        _mm_storel_epi64((__m128i*)dst, M);
        dst = dst1 + 8;
        M = _mm_loadu_si128((__m128i*)&first_line[8]);
        _mm_storel_epi64((__m128i*)dst, M);
        dst += i_dst;
        M = _mm_srli_si128(M, 2);
        _mm_storel_epi64((__m128i*)dst, M);
        dst += i_dst;
        M = _mm_srli_si128(M, 2);
        _mm_storel_epi64((__m128i*)dst, M);
        dst += i_dst;
        M = _mm_srli_si128(M, 2);
        _mm_storel_epi64((__m128i*)dst, M);
    } else if (bsx == 8) {
        for (i = 0; i < iHeight2; i += 8) {
            __m128i M = _mm_loadu_si128((__m128i*)&first_line[i]);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
            M = _mm_srli_si128(M, 2);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
            M = _mm_srli_si128(M, 2);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
            M = _mm_srli_si128(M, 2);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
        }
    } else {
        for (i = 0; i < iHeight2; i += 8) {
            __m128i M = _mm_loadu_si128((__m128i*)&first_line[i]);
            ((int*)(dst))[0] = _mm_cvtsi128_si32(M);
            dst += i_dst;
            M = _mm_srli_si128(M, 2);
            ((int*)(dst))[0] = _mm_cvtsi128_si32(M);
            dst += i_dst;
            M = _mm_srli_si128(M, 2);
            ((int*)(dst))[0] = _mm_cvtsi128_si32(M);
            dst += i_dst;
            M = _mm_srli_si128(M, 2);
            ((int*)(dst))[0] = _mm_cvtsi128_si32(M);
            dst += i_dst;
        }
    }

}


/* ---------------------------------------------------------------------------
 */
void intra_pred_ang_x_5_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    __m128i zero = _mm_setzero_si128();
    __m128i coeff2 = _mm_set1_epi16(2);
    __m128i coeff3 = _mm_set1_epi16(3);
    __m128i coeff4 = _mm_set1_epi16(4);
    __m128i coeff5 = _mm_set1_epi16(5);
    __m128i coeff7 = _mm_set1_epi16(7);
    __m128i coeff8 = _mm_set1_epi16(8);
    __m128i coeff9 = _mm_set1_epi16(9);
    __m128i coeff11 = _mm_set1_epi16(11);
    __m128i coeff13 = _mm_set1_epi16(13);
    __m128i coeff15 = _mm_set1_epi16(15);
    __m128i coeff16 = _mm_set1_epi16(16);

    UNUSED_PARAMETER(dir_mode);

    int i;
    if (((bsy > 4) && (bsx > 8))) {
        ALIGN16(pel_t first_line[(64 + 80 + 16) << 3]);
        int line_size = bsx + ((bsy - 8) >> 3) * 11;
        int aligned_line_size = (((line_size + 15) >> 4) << 4) + 16;
        pel_t *pfirst[8];

        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        pel_t *dst5 = dst4 + i_dst;
        pel_t *dst6 = dst5 + i_dst;
        pel_t *dst7 = dst6 + i_dst;
        pel_t *dst8 = dst7 + i_dst;

        pfirst[0] = first_line;
        pfirst[1] = pfirst[0] + aligned_line_size;
        pfirst[2] = pfirst[1] + aligned_line_size;
        pfirst[3] = pfirst[2] + aligned_line_size;
        pfirst[4] = pfirst[3] + aligned_line_size;
        pfirst[5] = pfirst[4] + aligned_line_size;
        pfirst[6] = pfirst[5] + aligned_line_size;
        pfirst[7] = pfirst[6] + aligned_line_size;

        __m128i p00, p10, p20, p30;
        __m128i p01, p11, p21, p31;
        for (i = 0; i < line_size - 8; i += 16, src += 16) {
            __m128i SS1 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i L1 = _mm_unpacklo_epi8(SS1, zero);
            SS1 = _mm_srli_si128(SS1, 1);
            __m128i L2 = _mm_unpacklo_epi8(SS1, zero);
            SS1 = _mm_srli_si128(SS1, 1);
            __m128i L3 = _mm_unpacklo_epi8(SS1, zero);
            SS1 = _mm_srli_si128(SS1, 1);
            __m128i L4 = _mm_unpacklo_epi8(SS1, zero);
            SS1 = _mm_srli_si128(SS1, 1);
            __m128i L5 = _mm_unpacklo_epi8(SS1, zero);
            SS1 = _mm_srli_si128(SS1, 1);
            __m128i L6 = _mm_unpacklo_epi8(SS1, zero);
            SS1 = _mm_srli_si128(SS1, 1);
            __m128i L7 = _mm_unpacklo_epi8(SS1, zero);
            SS1 = _mm_srli_si128(SS1, 1);
            __m128i L8 = _mm_unpacklo_epi8(SS1, zero);
            SS1 = _mm_srli_si128(SS1, 1);
            __m128i L9 = _mm_unpacklo_epi8(SS1, zero);
            __m128i H1 = L9;

            __m128i SS10 = _mm_loadu_si128((__m128i*)(src + 10));
            __m128i L10 = _mm_unpacklo_epi8(SS10, zero);
            __m128i H2 = L10;
            SS10 = _mm_srli_si128(SS10, 1);
            __m128i L11 = _mm_unpacklo_epi8(SS10, zero);
            __m128i H3 = L11;
            SS10 = _mm_srli_si128(SS10, 1);
            __m128i L12 = _mm_unpacklo_epi8(SS10, zero);
            __m128i H4 = L12;
            SS10 = _mm_srli_si128(SS10, 1);
            __m128i L13 = _mm_unpacklo_epi8(SS10, zero);
            __m128i H5 = L13;

            SS10 = _mm_srli_si128(SS10, 1);
            __m128i H6 = _mm_unpacklo_epi8(SS10, zero);
            SS10 = _mm_srli_si128(SS10, 1);
            __m128i H7 = _mm_unpacklo_epi8(SS10, zero);
            SS10 = _mm_srli_si128(SS10, 1);
            __m128i H8 = _mm_unpacklo_epi8(SS10, zero);
            SS10 = _mm_srli_si128(SS10, 1);
            __m128i H9 = _mm_unpacklo_epi8(SS10, zero);
            SS10 = _mm_srli_si128(SS10, 1);
            __m128i H10 = _mm_unpacklo_epi8(SS10, zero);

            __m128i SS19 = _mm_loadu_si128((__m128i*)(src + 19));
            __m128i H11 = _mm_unpacklo_epi8(SS19, zero);
            SS19 = _mm_srli_si128(SS19, 1);
            __m128i H12 = _mm_unpacklo_epi8(SS19, zero);
            SS19 = _mm_srli_si128(SS19, 1);
            __m128i H13 = _mm_unpacklo_epi8(SS19, zero);

            p00 = _mm_mullo_epi16(L1, coeff5);
            p10 = _mm_mullo_epi16(L2, coeff13);
            p20 = _mm_mullo_epi16(L3, coeff11);
            p30 = _mm_mullo_epi16(L4, coeff3);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H1, coeff5);
            p11 = _mm_mullo_epi16(H2, coeff13);
            p21 = _mm_mullo_epi16(H3, coeff11);
            p31 = _mm_mullo_epi16(H4, coeff3);
            p01 = _mm_add_epi16(p01, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)&pfirst[0][i], p00);

            p10 = _mm_mullo_epi16(L3, coeff5);
            p20 = _mm_mullo_epi16(L4, coeff7);
            p30 = _mm_mullo_epi16(L5, coeff3);
            p00 = _mm_add_epi16(L2, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p11 = _mm_mullo_epi16(H3, coeff5);
            p21 = _mm_mullo_epi16(H4, coeff7);
            p31 = _mm_mullo_epi16(H5, coeff3);
            p01 = _mm_add_epi16(H2, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)&pfirst[1][i], p00);

            p00 = _mm_mullo_epi16(L4, coeff7);
            p10 = _mm_mullo_epi16(L5, coeff15);
            p20 = _mm_mullo_epi16(L6, coeff9);
            p30 = _mm_add_epi16(L7, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H4, coeff7);
            p11 = _mm_mullo_epi16(H5, coeff15);
            p21 = _mm_mullo_epi16(H6, coeff9);
            p31 = _mm_add_epi16(H7, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)&pfirst[2][i], p00);

            p00 = _mm_add_epi16(L5, L8);
            p10 = _mm_add_epi16(L6, L7);
            p10 = _mm_mullo_epi16(p10, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_srli_epi16(p00, 3);

            p01 = _mm_add_epi16(H5, H8);
            p11 = _mm_add_epi16(H6, H7);
            p11 = _mm_mullo_epi16(p11, coeff3);
            p01 = _mm_add_epi16(p01, coeff4);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_srli_epi16(p01, 3);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)&pfirst[3][i], p00);

            p00 = _mm_add_epi16(L6, coeff16);
            p10 = _mm_mullo_epi16(L7, coeff9);
            p20 = _mm_mullo_epi16(L8, coeff15);
            p30 = _mm_mullo_epi16(L9, coeff7);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p01 = _mm_add_epi16(H6, coeff16);
            p11 = _mm_mullo_epi16(H7, coeff9);
            p21 = _mm_mullo_epi16(H8, coeff15);
            p31 = _mm_mullo_epi16(H9, coeff7);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)&pfirst[4][i], p00);

            p00 = _mm_mullo_epi16(L8, coeff3);
            p10 = _mm_mullo_epi16(L9, coeff7);
            p20 = _mm_mullo_epi16(L10, coeff5);
            p30 = _mm_add_epi16(L11, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p01 = _mm_mullo_epi16(H8, coeff3);
            p11 = _mm_mullo_epi16(H9, coeff7);
            p21 = _mm_mullo_epi16(H10, coeff5);
            p31 = _mm_add_epi16(H11, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)&pfirst[5][i], p00);

            p00 = _mm_mullo_epi16(L9, coeff3);
            p10 = _mm_mullo_epi16(L10, coeff11);
            p20 = _mm_mullo_epi16(L11, coeff13);
            p30 = _mm_mullo_epi16(L12, coeff5);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H9, coeff3);
            p11 = _mm_mullo_epi16(H10, coeff11);
            p21 = _mm_mullo_epi16(H11, coeff13);
            p31 = _mm_mullo_epi16(H12, coeff5);
            p01 = _mm_add_epi16(p01, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)&pfirst[6][i], p00);

            p00 = _mm_add_epi16(L11, L13);
            p10 = _mm_add_epi16(L12, L12);
            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_srli_epi16(p00, 2);

            p01 = _mm_add_epi16(H11, H13);
            p11 = _mm_add_epi16(H12, H12);
            p01 = _mm_add_epi16(p01, coeff2);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_srli_epi16(p01, 2);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)&pfirst[7][i], p00);
        }
        if (i < line_size) {
            __m128i SS1 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i L1 = _mm_unpacklo_epi8(SS1, zero);
            SS1 = _mm_srli_si128(SS1, 1);
            __m128i L2 = _mm_unpacklo_epi8(SS1, zero);
            SS1 = _mm_srli_si128(SS1, 1);
            __m128i L3 = _mm_unpacklo_epi8(SS1, zero);
            SS1 = _mm_srli_si128(SS1, 1);
            __m128i L4 = _mm_unpacklo_epi8(SS1, zero);
            SS1 = _mm_srli_si128(SS1, 1);
            __m128i L5 = _mm_unpacklo_epi8(SS1, zero);
            SS1 = _mm_srli_si128(SS1, 1);
            __m128i L6 = _mm_unpacklo_epi8(SS1, zero);
            SS1 = _mm_srli_si128(SS1, 1);
            __m128i L7 = _mm_unpacklo_epi8(SS1, zero);
            SS1 = _mm_srli_si128(SS1, 1);
            __m128i L8 = _mm_unpacklo_epi8(SS1, zero);
            SS1 = _mm_srli_si128(SS1, 1);
            __m128i L9 = _mm_unpacklo_epi8(SS1, zero);

            __m128i SS10 = _mm_loadu_si128((__m128i*)(src + 10));
            __m128i L10 = _mm_unpacklo_epi8(SS10, zero);
            SS10 = _mm_srli_si128(SS10, 1);
            __m128i L11 = _mm_unpacklo_epi8(SS10, zero);
            SS10 = _mm_srli_si128(SS10, 1);
            __m128i L12 = _mm_unpacklo_epi8(SS10, zero);
            SS10 = _mm_srli_si128(SS10, 1);
            __m128i L13 = _mm_unpacklo_epi8(SS10, zero);

            p00 = _mm_mullo_epi16(L1, coeff5);
            p10 = _mm_mullo_epi16(L2, coeff13);
            p20 = _mm_mullo_epi16(L3, coeff11);
            p30 = _mm_mullo_epi16(L4, coeff3);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[0][i], p00);

            p10 = _mm_mullo_epi16(L3, coeff5);
            p20 = _mm_mullo_epi16(L4, coeff7);
            p30 = _mm_mullo_epi16(L5, coeff3);
            p00 = _mm_add_epi16(L2, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[1][i], p00);

            p00 = _mm_mullo_epi16(L4, coeff7);
            p10 = _mm_mullo_epi16(L5, coeff15);
            p20 = _mm_mullo_epi16(L6, coeff9);
            p30 = _mm_add_epi16(L7, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[2][i], p00);

            p00 = _mm_add_epi16(L5, L8);
            p10 = _mm_add_epi16(L6, L7);
            p10 = _mm_mullo_epi16(p10, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_srli_epi16(p00, 3);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[3][i], p00);

            p00 = _mm_add_epi16(L6, coeff16);
            p10 = _mm_mullo_epi16(L7, coeff9);
            p20 = _mm_mullo_epi16(L8, coeff15);
            p30 = _mm_mullo_epi16(L9, coeff7);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[4][i], p00);

            p00 = _mm_mullo_epi16(L8, coeff3);
            p10 = _mm_mullo_epi16(L9, coeff7);
            p20 = _mm_mullo_epi16(L10, coeff5);
            p30 = _mm_add_epi16(L11, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[5][i], p00);

            p00 = _mm_mullo_epi16(L9, coeff3);
            p10 = _mm_mullo_epi16(L10, coeff11);
            p20 = _mm_mullo_epi16(L11, coeff13);
            p30 = _mm_mullo_epi16(L12, coeff5);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[6][i], p00);

            p00 = _mm_add_epi16(L11, L13);
            p10 = _mm_add_epi16(L12, L12);
            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_srli_epi16(p00, 2);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[7][i], p00);
        }

        bsy >>= 3;
        for (i = 0; i < bsy; i++) {
            memcpy(dst1, pfirst[0] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst2, pfirst[1] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst3, pfirst[2] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst4, pfirst[3] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst5, pfirst[4] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst6, pfirst[5] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst7, pfirst[6] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst8, pfirst[7] + i * 11, bsx * sizeof(pel_t));

            dst1 = dst8 + i_dst;
            dst2 = dst1 + i_dst;
            dst3 = dst2 + i_dst;
            dst4 = dst3 + i_dst;
            dst5 = dst4 + i_dst;
            dst6 = dst5 + i_dst;
            dst7 = dst6 + i_dst;
            dst8 = dst7 + i_dst;
        }
    } else if (bsx == 16) {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;

        __m128i p00, p10, p20, p30;
        __m128i p01, p11, p21, p31;

        __m128i SS1 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i L1 = _mm_unpacklo_epi8(SS1, zero);
        SS1 = _mm_srli_si128(SS1, 1);
        __m128i L2 = _mm_unpacklo_epi8(SS1, zero);
        SS1 = _mm_srli_si128(SS1, 1);
        __m128i L3 = _mm_unpacklo_epi8(SS1, zero);
        SS1 = _mm_srli_si128(SS1, 1);
        __m128i L4 = _mm_unpacklo_epi8(SS1, zero);
        SS1 = _mm_srli_si128(SS1, 1);
        __m128i L5 = _mm_unpacklo_epi8(SS1, zero);
        SS1 = _mm_srli_si128(SS1, 1);
        __m128i L6 = _mm_unpacklo_epi8(SS1, zero);
        SS1 = _mm_srli_si128(SS1, 1);
        __m128i L7 = _mm_unpacklo_epi8(SS1, zero);
        SS1 = _mm_srli_si128(SS1, 1);
        __m128i L8 = _mm_unpacklo_epi8(SS1, zero);
        SS1 = _mm_srli_si128(SS1, 1);
        __m128i H1 = _mm_unpacklo_epi8(SS1, zero);

        __m128i SS10 = _mm_loadu_si128((__m128i*)(src + 10));
        __m128i H2 = _mm_unpacklo_epi8(SS10, zero);
        SS10 = _mm_srli_si128(SS10, 1);
        __m128i H3 = _mm_unpacklo_epi8(SS10, zero);
        SS10 = _mm_srli_si128(SS10, 1);
        __m128i H4 = _mm_unpacklo_epi8(SS10, zero);
        SS10 = _mm_srli_si128(SS10, 1);
        __m128i H5 = _mm_unpacklo_epi8(SS10, zero);

        SS10 = _mm_srli_si128(SS10, 1);
        __m128i H6 = _mm_unpacklo_epi8(SS10, zero);
        SS10 = _mm_srli_si128(SS10, 1);
        __m128i H7 = _mm_unpacklo_epi8(SS10, zero);
        SS10 = _mm_srli_si128(SS10, 1);
        __m128i H8 = _mm_unpacklo_epi8(SS10, zero);

        p00 = _mm_mullo_epi16(L1, coeff5);
        p10 = _mm_mullo_epi16(L2, coeff13);
        p20 = _mm_mullo_epi16(L3, coeff11);
        p30 = _mm_mullo_epi16(L4, coeff3);
        p00 = _mm_add_epi16(p00, coeff16);
        p00 = _mm_add_epi16(p00, p10);
        p00 = _mm_add_epi16(p00, p20);
        p00 = _mm_add_epi16(p00, p30);
        p00 = _mm_srli_epi16(p00, 5);

        p01 = _mm_mullo_epi16(H1, coeff5);
        p11 = _mm_mullo_epi16(H2, coeff13);
        p21 = _mm_mullo_epi16(H3, coeff11);
        p31 = _mm_mullo_epi16(H4, coeff3);
        p01 = _mm_add_epi16(p01, coeff16);
        p01 = _mm_add_epi16(p01, p11);
        p01 = _mm_add_epi16(p01, p21);
        p01 = _mm_add_epi16(p01, p31);
        p01 = _mm_srli_epi16(p01, 5);

        p00 = _mm_packus_epi16(p00, p01);
        _mm_store_si128((__m128i*)dst1, p00);

        p10 = _mm_mullo_epi16(L3, coeff5);
        p20 = _mm_mullo_epi16(L4, coeff7);
        p30 = _mm_mullo_epi16(L5, coeff3);
        p00 = _mm_add_epi16(L2, coeff8);
        p00 = _mm_add_epi16(p00, p10);
        p00 = _mm_add_epi16(p00, p20);
        p00 = _mm_add_epi16(p00, p30);
        p00 = _mm_srli_epi16(p00, 4);

        p11 = _mm_mullo_epi16(H3, coeff5);
        p21 = _mm_mullo_epi16(H4, coeff7);
        p31 = _mm_mullo_epi16(H5, coeff3);
        p01 = _mm_add_epi16(H2, coeff8);
        p01 = _mm_add_epi16(p01, p11);
        p01 = _mm_add_epi16(p01, p21);
        p01 = _mm_add_epi16(p01, p31);
        p01 = _mm_srli_epi16(p01, 4);

        p00 = _mm_packus_epi16(p00, p01);
        _mm_store_si128((__m128i*)dst2, p00);

        p00 = _mm_mullo_epi16(L4, coeff7);
        p10 = _mm_mullo_epi16(L5, coeff15);
        p20 = _mm_mullo_epi16(L6, coeff9);
        p30 = _mm_add_epi16(L7, coeff16);
        p00 = _mm_add_epi16(p00, p10);
        p00 = _mm_add_epi16(p00, p20);
        p00 = _mm_add_epi16(p00, p30);
        p00 = _mm_srli_epi16(p00, 5);

        p01 = _mm_mullo_epi16(H4, coeff7);
        p11 = _mm_mullo_epi16(H5, coeff15);
        p21 = _mm_mullo_epi16(H6, coeff9);
        p31 = _mm_add_epi16(H7, coeff16);
        p01 = _mm_add_epi16(p01, p11);
        p01 = _mm_add_epi16(p01, p21);
        p01 = _mm_add_epi16(p01, p31);
        p01 = _mm_srli_epi16(p01, 5);

        p00 = _mm_packus_epi16(p00, p01);
        _mm_store_si128((__m128i*)dst3, p00);

        p00 = _mm_add_epi16(L5, L8);
        p10 = _mm_add_epi16(L6, L7);
        p10 = _mm_mullo_epi16(p10, coeff3);
        p00 = _mm_add_epi16(p00, coeff4);
        p00 = _mm_add_epi16(p00, p10);
        p00 = _mm_srli_epi16(p00, 3);

        p01 = _mm_add_epi16(H5, H8);
        p11 = _mm_add_epi16(H6, H7);
        p11 = _mm_mullo_epi16(p11, coeff3);
        p01 = _mm_add_epi16(p01, coeff4);
        p01 = _mm_add_epi16(p01, p11);
        p01 = _mm_srli_epi16(p01, 3);

        p00 = _mm_packus_epi16(p00, p01);
        _mm_store_si128((__m128i*)dst4, p00);
    } else if (bsx == 8) {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        pel_t *dst5 = dst4 + i_dst;
        pel_t *dst6 = dst5 + i_dst;
        pel_t *dst7 = dst6 + i_dst;
        pel_t *dst8 = dst7 + i_dst;

        for (i = 0; i < 8; src++, i++) {
            dst1[i] = (pel_t)((5 * src[1] + 13 * src[2] + 11 * src[3] + 3 * src[4] + 16) >> 5);
            dst2[i] = (pel_t)((src[2] + 5 * src[3] + 7 * src[4] + 3 * src[5] + 8) >> 4);
            dst3[i] = (pel_t)((7 * src[4] + 15 * src[5] + 9 * src[6] + 1 * src[7] + 16) >> 5);
            dst4[i] = (pel_t)((src[5] + 3 * src[6] + 3 * src[7] + 1 * src[8] + 4) >> 3);

            dst5[i] = (pel_t)((src[6] + 9 * src[7] + 15 * src[8] + 7 * src[9] + 16) >> 5);
            dst6[i] = (pel_t)((3 * src[8] + 7 * src[9] + 5 * src[10] + src[11] + 8) >> 4);
            dst7[i] = (pel_t)((3 * src[9] + 11 * src[10] + 13 * src[11] + 5 * src[12] + 16) >> 5);
            dst8[i] = (pel_t)((src[11] + 2 * src[12] + src[13] + 2) >> 2);
        }
        if (bsy == 32) {
            //src -> 8,src[7] -> 15
            __m128i pad1 = _mm_set1_epi8(src[8]);

            dst1 = dst8 + i_dst;
            dst2 = dst1 + i_dst;
            dst3 = dst2 + i_dst;
            dst4 = dst3 + i_dst;
            dst5 = dst4 + i_dst;
            dst6 = dst5 + i_dst;
            dst7 = dst6 + i_dst;
            dst8 = dst7 + i_dst;

            _mm_storel_epi64((__m128i*)dst1, pad1);
            _mm_storel_epi64((__m128i*)dst2, pad1);
            _mm_storel_epi64((__m128i*)dst3, pad1);
            _mm_storel_epi64((__m128i*)dst4, pad1);

            _mm_storel_epi64((__m128i*)dst5, pad1);
            _mm_storel_epi64((__m128i*)dst6, pad1);
            _mm_storel_epi64((__m128i*)dst7, pad1);
            _mm_storel_epi64((__m128i*)dst8, pad1);

            src += 4;
            dst1[0] = (pel_t)((5 * src[0] + 13 * src[1] + 11 * src[2] + 3 * src[3] + 16) >> 5);
            dst1[1] = (pel_t)((5 * src[1] + 13 * src[2] + 11 * src[3] + 3 * src[4] + 16) >> 5);
            dst1[2] = (pel_t)((5 * src[2] + 13 * src[3] + 11 * src[4] + 3 * src[5] + 16) >> 5);
            dst1[3] = (pel_t)((5 * src[3] + 13 * src[4] + 11 * src[5] + 3 * src[6] + 16) >> 5);
            dst2[0] = (pel_t)((src[1] + 5 * src[2] + 7 * src[3] + 3 * src[4] + 8) >> 4);
            dst2[1] = (pel_t)((src[2] + 5 * src[3] + 7 * src[4] + 3 * src[5] + 8) >> 4);
            dst2[2] = (pel_t)((src[3] + 5 * src[4] + 7 * src[5] + 3 * src[6] + 8) >> 4);
            dst3[0] = (pel_t)((7 * src[3] + 15 * src[4] + 9 * src[5] + src[6] + 16) >> 5);

            dst1 = dst8 + i_dst;
            dst2 = dst1 + i_dst;
            dst3 = dst2 + i_dst;
            dst4 = dst3 + i_dst;
            dst5 = dst4 + i_dst;
            dst6 = dst5 + i_dst;
            dst7 = dst6 + i_dst;
            dst8 = dst7 + i_dst;

            _mm_storel_epi64((__m128i*)dst1, pad1);
            _mm_storel_epi64((__m128i*)dst2, pad1);
            _mm_storel_epi64((__m128i*)dst3, pad1);
            _mm_storel_epi64((__m128i*)dst4, pad1);

            _mm_storel_epi64((__m128i*)dst5, pad1);
            _mm_storel_epi64((__m128i*)dst6, pad1);
            _mm_storel_epi64((__m128i*)dst7, pad1);
            _mm_storel_epi64((__m128i*)dst8, pad1);

            dst1 = dst8 + i_dst;
            dst2 = dst1 + i_dst;
            dst3 = dst2 + i_dst;
            dst4 = dst3 + i_dst;
            dst5 = dst4 + i_dst;
            dst6 = dst5 + i_dst;
            dst7 = dst6 + i_dst;
            dst8 = dst7 + i_dst;

            _mm_storel_epi64((__m128i*)dst1, pad1);
            _mm_storel_epi64((__m128i*)dst2, pad1);
            _mm_storel_epi64((__m128i*)dst3, pad1);
            _mm_storel_epi64((__m128i*)dst4, pad1);

            _mm_storel_epi64((__m128i*)dst5, pad1);
            _mm_storel_epi64((__m128i*)dst6, pad1);
            _mm_storel_epi64((__m128i*)dst7, pad1);
            _mm_storel_epi64((__m128i*)dst8, pad1);
        }
    } else {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;

        __m128i p00, p10, p20, p30;

        __m128i SS1 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i L1 = _mm_unpacklo_epi8(SS1, zero);
        SS1 = _mm_srli_si128(SS1, 1);
        __m128i L2 = _mm_unpacklo_epi8(SS1, zero);
        SS1 = _mm_srli_si128(SS1, 1);
        __m128i L3 = _mm_unpacklo_epi8(SS1, zero);
        SS1 = _mm_srli_si128(SS1, 1);
        __m128i L4 = _mm_unpacklo_epi8(SS1, zero);
        SS1 = _mm_srli_si128(SS1, 1);
        __m128i L5 = _mm_unpacklo_epi8(SS1, zero);
        SS1 = _mm_srli_si128(SS1, 1);
        __m128i L6 = _mm_unpacklo_epi8(SS1, zero);
        SS1 = _mm_srli_si128(SS1, 1);
        __m128i L7 = _mm_unpacklo_epi8(SS1, zero);
        SS1 = _mm_srli_si128(SS1, 1);
        __m128i L8 = _mm_unpacklo_epi8(SS1, zero);

        p00 = _mm_mullo_epi16(L1, coeff5);
        p10 = _mm_mullo_epi16(L2, coeff13);
        p20 = _mm_mullo_epi16(L3, coeff11);
        p30 = _mm_mullo_epi16(L4, coeff3);
        p00 = _mm_add_epi16(p00, coeff16);
        p00 = _mm_add_epi16(p00, p10);
        p00 = _mm_add_epi16(p00, p20);
        p00 = _mm_add_epi16(p00, p30);
        p00 = _mm_srli_epi16(p00, 5);

        p00 = _mm_packus_epi16(p00, p00);
        *((int*)(dst1)) = _mm_cvtsi128_si32(p00);

        p10 = _mm_mullo_epi16(L3, coeff5);
        p20 = _mm_mullo_epi16(L4, coeff7);
        p30 = _mm_mullo_epi16(L5, coeff3);
        p00 = _mm_add_epi16(L2, coeff8);
        p00 = _mm_add_epi16(p00, p10);
        p00 = _mm_add_epi16(p00, p20);
        p00 = _mm_add_epi16(p00, p30);
        p00 = _mm_srli_epi16(p00, 4);

        p00 = _mm_packus_epi16(p00, p00);
        *((int*)(dst2)) = _mm_cvtsi128_si32(p00);

        p00 = _mm_mullo_epi16(L4, coeff7);
        p10 = _mm_mullo_epi16(L5, coeff15);
        p20 = _mm_mullo_epi16(L6, coeff9);
        p30 = _mm_add_epi16(L7, coeff16);
        p00 = _mm_add_epi16(p00, p10);
        p00 = _mm_add_epi16(p00, p20);
        p00 = _mm_add_epi16(p00, p30);
        p00 = _mm_srli_epi16(p00, 5);

        p00 = _mm_packus_epi16(p00, p00);
        *((int*)(dst3)) = _mm_cvtsi128_si32(p00);

        p00 = _mm_add_epi16(L5, L8);
        p10 = _mm_add_epi16(L6, L7);
        p10 = _mm_mullo_epi16(p10, coeff3);
        p00 = _mm_add_epi16(p00, coeff4);
        p00 = _mm_add_epi16(p00, p10);
        p00 = _mm_srli_epi16(p00, 3);

        p00 = _mm_packus_epi16(p00, p00);
        *((int*)(dst4)) = _mm_cvtsi128_si32(p00);

        if (bsy == 16) {
            pel_t *dst5 = dst4 + i_dst;
            pel_t *dst6 = dst5 + i_dst;
            pel_t *dst7 = dst6 + i_dst;
            pel_t *dst8 = dst7 + i_dst;

            src += 8;
            __m128i pad1 = _mm_set1_epi8(src[0]);

            *(int*)(dst5) = _mm_cvtsi128_si32(pad1);
            *(int*)(dst6) = _mm_cvtsi128_si32(pad1);
            *(int*)(dst7) = _mm_cvtsi128_si32(pad1);
            *(int*)(dst8) = _mm_cvtsi128_si32(pad1);

            dst5[0] = (pel_t)((src[-2] + 9 * src[-1] + 15 * src[0] + 7 * src[1] + 16) >> 5);
            dst5[1] = (pel_t)((src[-1] + 9 * src[0] + 15 * src[1] + 7 * src[2] + 16) >> 5);

            dst1 = dst8 + i_dst;
            dst2 = dst1 + i_dst;
            dst3 = dst2 + i_dst;
            dst4 = dst3 + i_dst;
            dst5 = dst4 + i_dst;
            dst6 = dst5 + i_dst;
            dst7 = dst6 + i_dst;
            dst8 = dst7 + i_dst;

            *(int*)(dst1) = _mm_cvtsi128_si32(pad1);
            *(int*)(dst2) = _mm_cvtsi128_si32(pad1);
            *(int*)(dst3) = _mm_cvtsi128_si32(pad1);
            *(int*)(dst4) = _mm_cvtsi128_si32(pad1);

            *(int*)(dst5) = _mm_cvtsi128_si32(pad1);
            *(int*)(dst6) = _mm_cvtsi128_si32(pad1);
            *(int*)(dst7) = _mm_cvtsi128_si32(pad1);
            *(int*)(dst8) = _mm_cvtsi128_si32(pad1);
        }
    }

}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_x_6_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[64 + 64]);
    int line_size = bsx + bsy - 1;
    int i;
    __m128i zero = _mm_setzero_si128();
    __m128i offset = _mm_set1_epi16(2);

    UNUSED_PARAMETER(dir_mode);

    src += 2;
    for (i = 0; i < line_size - 8; i += 16, src += 16) {
        __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);

        __m128i H0 = _mm_unpackhi_epi8(S0, zero);
        __m128i H1 = _mm_unpackhi_epi8(S1, zero);
        __m128i H2 = _mm_unpackhi_epi8(S2, zero);

        __m128i sum1 = _mm_add_epi16(L0, L1);
        __m128i sum2 = _mm_add_epi16(L1, L2);
        __m128i sum3 = _mm_add_epi16(H0, H1);
        __m128i sum4 = _mm_add_epi16(H1, H2);

        sum1 = _mm_add_epi16(sum1, sum2);
        sum3 = _mm_add_epi16(sum3, sum4);

        sum1 = _mm_add_epi16(sum1, offset);
        sum3 = _mm_add_epi16(sum3, offset);

        sum1 = _mm_srli_epi16(sum1, 2);
        sum3 = _mm_srli_epi16(sum3, 2);

        sum1 = _mm_packus_epi16(sum1, sum3);

        _mm_store_si128((__m128i*)&first_line[i], sum1);
    }

    if (i < line_size) {
        __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);

        __m128i sum1 = _mm_add_epi16(L0, L1);
        __m128i sum2 = _mm_add_epi16(L1, L2);

        sum1 = _mm_add_epi16(sum1, sum2);
        sum1 = _mm_add_epi16(sum1, offset);
        sum1 = _mm_srli_epi16(sum1, 2);

        sum1 = _mm_packus_epi16(sum1, sum1);
        _mm_storel_epi64((__m128i*)&first_line[i], sum1);
    }

    if (bsx > 16 || bsx == 4) {
        for (i = 0; i < bsy; i++) {
            memcpy(dst, first_line + i, bsx * sizeof(pel_t));
            dst += i_dst;
        }
    } else if (bsx == 16) {
        pel_t *dst1 = dst;
        pel_t *dst2;
        if (bsy == 4) {
            __m128i M = _mm_loadu_si128((__m128i*)&first_line[0]);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 = dst + 8;
            M = _mm_loadu_si128((__m128i*)&first_line[8]);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
        } else {
            __m128i M = _mm_loadu_si128((__m128i*)&first_line[0]);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst2 = dst1 + i_dst;
            dst1 = dst + 8;
            M = _mm_loadu_si128((__m128i*)&first_line[8]);
            _mm_storel_epi64((__m128i*)dst1, M);
            _mm_storel_epi64((__m128i*)dst2, M);
            dst1 += i_dst;
            dst2 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            _mm_storel_epi64((__m128i*)dst2, M);
            dst1 += i_dst;
            dst2 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            _mm_storel_epi64((__m128i*)dst2, M);
            dst1 += i_dst;
            dst2 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            _mm_storel_epi64((__m128i*)dst2, M);
            dst1 += i_dst;
            dst2 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            _mm_storel_epi64((__m128i*)dst2, M);
            dst1 += i_dst;
            dst2 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            _mm_storel_epi64((__m128i*)dst2, M);
            dst1 += i_dst;
            dst2 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            _mm_storel_epi64((__m128i*)dst2, M);
            dst1 += i_dst;
            dst2 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            _mm_storel_epi64((__m128i*)dst2, M);
            dst1 += i_dst;
            M = _mm_loadu_si128((__m128i*)&first_line[16]);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
        }
    } else {
        for (i = 0; i < bsy; i += 8) {
            __m128i M = _mm_loadu_si128((__m128i*)&first_line[i]);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
        }
    }
}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_x_7_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i, j;
    int iWidth2 = bsx << 1;
    __m128i zero = _mm_setzero_si128();
    __m128i S0, S1, S2, S3;
    __m128i t0, t1, t2, t3;
    __m128i off = _mm_set1_epi16(64);
    __m128i c0;

    UNUSED_PARAMETER(dir_mode);

    if (bsx >= bsy) {
        if (bsx & 0x07) {
            __m128i D0;
            int i_dst2 = i_dst << 1;

            for (j = 0; j < bsy; j += 2) {
                int idx = tab_idx_mode_7[j];
                c0 = _mm_load_si128((__m128i*)tab_coeff_mode_7[j]);

                S0 = _mm_loadl_epi64((__m128i*)(src + idx));
                S1 = _mm_srli_si128(S0, 1);
                S2 = _mm_srli_si128(S0, 2);
                S3 = _mm_srli_si128(S0, 3);

                t0 = _mm_unpacklo_epi8(S0, S1);
                t1 = _mm_unpacklo_epi8(S2, S3);
                t2 = _mm_unpacklo_epi16(t0, t1);

                t0 = _mm_maddubs_epi16(t2, c0);

                idx = tab_idx_mode_7[j + 1];
                c0 = _mm_load_si128((__m128i*)tab_coeff_mode_7[j + 1]);
                S0 = _mm_loadl_epi64((__m128i*)(src + idx));
                S1 = _mm_srli_si128(S0, 1);
                S2 = _mm_srli_si128(S0, 2);
                S3 = _mm_srli_si128(S0, 3);

                t1 = _mm_unpacklo_epi8(S0, S1);
                t2 = _mm_unpacklo_epi8(S2, S3);
                t1 = _mm_unpacklo_epi16(t1, t2);

                t1 = _mm_maddubs_epi16(t1, c0);

                D0 = _mm_hadds_epi16(t0, t1);
                D0 = _mm_add_epi16(D0, off);
                D0 = _mm_srli_epi16(D0, 7);
                D0 = _mm_packus_epi16(D0, zero);

                ((uint32_t*)(dst))[0] = _mm_cvtsi128_si32(D0);
                D0= _mm_srli_si128(D0, 4);
                ((uint32_t*)(dst + i_dst))[0] = _mm_cvtsi128_si32(D0);
                //dst[i] = (pel_t)((src[idx] * c1 + src[idx + 1] * c2 + src[idx + 2] * c3 + src[idx + 3] * c4 + 64) >> 7);
                dst += i_dst2;
            }
        } else if (bsx & 0x0f) {
            __m128i D0;

            for (j = 0; j < bsy; j++) {
                int idx = tab_idx_mode_7[j];
                c0 = _mm_load_si128((__m128i*)tab_coeff_mode_7[j]);

                S0 = _mm_loadu_si128((__m128i*)(src + idx));
                S1 = _mm_srli_si128(S0, 1);
                S2 = _mm_srli_si128(S0, 2);
                S3 = _mm_srli_si128(S0, 3);

                t0 = _mm_unpacklo_epi8(S0, S1);
                t1 = _mm_unpacklo_epi8(S2, S3);
                t2 = _mm_unpacklo_epi16(t0, t1);
                t3 = _mm_unpackhi_epi16(t0, t1);

                t0 = _mm_maddubs_epi16(t2, c0);
                t1 = _mm_maddubs_epi16(t3, c0);

                D0 = _mm_hadds_epi16(t0, t1);
                D0 = _mm_add_epi16(D0, off);
                D0 = _mm_srli_epi16(D0, 7);

                D0 = _mm_packus_epi16(D0, _mm_setzero_si128());

                _mm_storel_epi64((__m128i*)(dst), D0);
                //dst[i] = (pel_t)((src[idx] * c1 + src[idx + 1] * c2 + src[idx + 2] * c3 + src[idx + 3] * c4 + 64) >> 7);

                dst += i_dst;
            }
        } else {
            for (j = 0; j < bsy; j++) {
                __m128i D0, D1;

                int idx = tab_idx_mode_7[j];
                c0 = _mm_load_si128((__m128i*)tab_coeff_mode_7[j]);

                for (i = 0; i < bsx; i += 16, idx += 16) {
                    S0 = _mm_loadu_si128((__m128i*)(src + idx));
                    S1 = _mm_loadu_si128((__m128i*)(src + idx + 1));
                    S2 = _mm_loadu_si128((__m128i*)(src + idx + 2));
                    S3 = _mm_loadu_si128((__m128i*)(src + idx + 3));

                    t0 = _mm_unpacklo_epi8(S0, S1);
                    t1 = _mm_unpacklo_epi8(S2, S3);
                    t2 = _mm_unpacklo_epi16(t0, t1);
                    t3 = _mm_unpackhi_epi16(t0, t1);

                    t0 = _mm_maddubs_epi16(t2, c0);
                    t1 = _mm_maddubs_epi16(t3, c0);

                    D0 = _mm_hadds_epi16(t0, t1);
                    D0 = _mm_add_epi16(D0, off);
                    D0 = _mm_srli_epi16(D0, 7);

                    t0 = _mm_unpackhi_epi8(S0, S1);
                    t1 = _mm_unpackhi_epi8(S2, S3);
                    t2 = _mm_unpacklo_epi16(t0, t1);
                    t3 = _mm_unpackhi_epi16(t0, t1);

                    t0 = _mm_maddubs_epi16(t2, c0);
                    t1 = _mm_maddubs_epi16(t3, c0);

                    D1 = _mm_hadds_epi16(t0, t1);
                    D1 = _mm_add_epi16(D1, off);
                    D1 = _mm_srli_epi16(D1, 7);

                    D0 = _mm_packus_epi16(D0, D1);

                    _mm_storeu_si128((__m128i*)(dst + i), D0);
                    //dst[i] = (pel_t)((src[idx] * c1 + src[idx + 1] * c2 + src[idx + 2] * c3 + src[idx + 3] * c4 + 64) >> 7);
                }

                dst += i_dst;
            }
        }
    } else {
        if (bsx & 0x07) {
            for (j = 0; j < bsy; j++) {
                int real_width;
                int idx = tab_idx_mode_7[j];

                real_width = XAVS2_MIN(bsx, iWidth2 - idx + 1);

                if (real_width <= 0) {
                    pel_t val = (pel_t)((src[iWidth2] * tab_coeff_mode_7[j][0] + src[iWidth2 + 1] * tab_coeff_mode_7[j][1] + src[iWidth2 + 2] * tab_coeff_mode_7[j][2] + src[iWidth2 + 3] * tab_coeff_mode_7[j][3] + 64) >> 7);
                    __m128i D0 = _mm_set1_epi8((char)val);
                    _mm_storel_epi64((__m128i*)(dst), D0);
                    dst += i_dst;
                    j++;

                    for (; j < bsy; j++) {
                        val = (pel_t)((src[iWidth2] * tab_coeff_mode_7[j][0] + src[iWidth2 + 1] * tab_coeff_mode_7[j][1] + src[iWidth2 + 2] * tab_coeff_mode_7[j][2] + src[iWidth2 + 3] * tab_coeff_mode_7[j][3] + 64) >> 7);
                        D0 = _mm_set1_epi8((char)val);
                        _mm_storel_epi64((__m128i*)(dst), D0);
                        dst += i_dst;
                    }
                    break;
                } else {
                    __m128i D0;
                    c0 = _mm_load_si128((__m128i*)tab_coeff_mode_7[j]);

                    S0 = _mm_loadl_epi64((__m128i*)(src + idx));
                    S1 = _mm_srli_si128(S0, 1);
                    S2 = _mm_srli_si128(S0, 2);
                    S3 = _mm_srli_si128(S0, 3);

                    t0 = _mm_unpacklo_epi8(S0, S1);
                    t1 = _mm_unpacklo_epi8(S2, S3);
                    t2 = _mm_unpacklo_epi16(t0, t1);

                    t0 = _mm_maddubs_epi16(t2, c0);

                    D0 = _mm_hadds_epi16(t0, zero);
                    D0 = _mm_add_epi16(D0, off);
                    D0 = _mm_srli_epi16(D0, 7);

                    D0 = _mm_packus_epi16(D0, zero);

                    _mm_storel_epi64((__m128i*)(dst), D0);

                    if (real_width < bsx) {
                        D0 = _mm_set1_epi8((char)dst[real_width - 1]);
                        _mm_storel_epi64((__m128i*)(dst + real_width), D0);
                    }
                }
                dst += i_dst;
            }
        } else if (bsx & 0x0f) {
            for (j = 0; j < bsy; j++) {
                int real_width;
                int idx = tab_idx_mode_7[j];

                real_width = XAVS2_MIN(bsx, iWidth2 - idx + 1);

                if (real_width <= 0) {
                    pel_t val = (pel_t)((src[iWidth2] * tab_coeff_mode_7[j][0] + src[iWidth2 + 1] * tab_coeff_mode_7[j][1] + src[iWidth2 + 2] * tab_coeff_mode_7[j][2] + src[iWidth2 + 3] * tab_coeff_mode_7[j][3] + 64) >> 7);
                    __m128i D0 = _mm_set1_epi8((char)val);
                    _mm_storel_epi64((__m128i*)(dst), D0);
                    dst += i_dst;
                    j++;

                    for (; j < bsy; j++) {
                        val = (pel_t)((src[iWidth2] * tab_coeff_mode_7[j][0] + src[iWidth2 + 1] * tab_coeff_mode_7[j][1] + src[iWidth2 + 2] * tab_coeff_mode_7[j][2] + src[iWidth2 + 3] * tab_coeff_mode_7[j][3] + 64) >> 7);
                        D0 = _mm_set1_epi8((char)val);
                        _mm_storel_epi64((__m128i*)(dst), D0);
                        dst += i_dst;
                    }
                    break;
                } else {
                    __m128i D0;
                    c0 = _mm_load_si128((__m128i*)tab_coeff_mode_7[j]);

                    S0 = _mm_loadu_si128((__m128i*)(src + idx));
                    S1 = _mm_srli_si128(S0, 1);
                    S2 = _mm_srli_si128(S0, 2);
                    S3 = _mm_srli_si128(S0, 3);

                    t0 = _mm_unpacklo_epi8(S0, S1);
                    t1 = _mm_unpacklo_epi8(S2, S3);
                    t2 = _mm_unpacklo_epi16(t0, t1);
                    t3 = _mm_unpackhi_epi16(t0, t1);

                    t0 = _mm_maddubs_epi16(t2, c0);
                    t1 = _mm_maddubs_epi16(t3, c0);

                    D0 = _mm_hadds_epi16(t0, t1);
                    D0 = _mm_add_epi16(D0, off);
                    D0 = _mm_srli_epi16(D0, 7);

                    D0 = _mm_packus_epi16(D0, zero);

                    _mm_storel_epi64((__m128i*)(dst), D0);
                    //dst[i] = (pel_t)((src[idx] * c1 + src[idx + 1] * c2 + src[idx + 2] * c3 + src[idx + 3] * c4 + 64) >> 7);

                    if (real_width < bsx) {
                        D0 = _mm_set1_epi8((char)dst[real_width - 1]);
                        _mm_storel_epi64((__m128i*)(dst + real_width), D0);
                    }

                }

                dst += i_dst;
            }
        } else {
            for (j = 0; j < bsy; j++) {
                int real_width;
                int idx = tab_idx_mode_7[j];

                real_width = XAVS2_MIN(bsx, iWidth2 - idx + 1);

                if (real_width <= 0) {
                    pel_t val = (pel_t)((src[iWidth2] * tab_coeff_mode_7[j][0] + src[iWidth2 + 1] * tab_coeff_mode_7[j][1] + src[iWidth2 + 2] * tab_coeff_mode_7[j][2] + src[iWidth2 + 3] * tab_coeff_mode_7[j][3] + 64) >> 7);
                    __m128i D0 = _mm_set1_epi8((char)val);

                    for (i = 0; i < bsx; i += 16) {
                        _mm_storeu_si128((__m128i*)(dst + i), D0);
                    }
                    dst += i_dst;
                    j++;

                    for (; j < bsy; j++) {
                        val = (pel_t)((src[iWidth2] * tab_coeff_mode_7[j][0] + src[iWidth2 + 1] * tab_coeff_mode_7[j][1] + src[iWidth2 + 2] * tab_coeff_mode_7[j][2] + src[iWidth2 + 3] * tab_coeff_mode_7[j][3] + 64) >> 7);
                        D0 = _mm_set1_epi8((char)val);
                        for (i = 0; i < bsx; i += 16) {
                            _mm_storeu_si128((__m128i*)(dst + i), D0);
                        }
                        dst += i_dst;
                    }
                    break;
                } else {
                    __m128i D0, D1;

                    c0 = _mm_load_si128((__m128i*)tab_coeff_mode_7[j]);
                    for (i = 0; i < real_width; i += 16, idx += 16) {
                        S0 = _mm_loadu_si128((__m128i*)(src + idx));
                        S1 = _mm_loadu_si128((__m128i*)(src + idx + 1));
                        S2 = _mm_loadu_si128((__m128i*)(src + idx + 2));
                        S3 = _mm_loadu_si128((__m128i*)(src + idx + 3));

                        t0 = _mm_unpacklo_epi8(S0, S1);
                        t1 = _mm_unpacklo_epi8(S2, S3);
                        t2 = _mm_unpacklo_epi16(t0, t1);
                        t3 = _mm_unpackhi_epi16(t0, t1);

                        t0 = _mm_maddubs_epi16(t2, c0);
                        t1 = _mm_maddubs_epi16(t3, c0);

                        D0 = _mm_hadds_epi16(t0, t1);
                        D0 = _mm_add_epi16(D0, off);
                        D0 = _mm_srli_epi16(D0, 7);

                        t0 = _mm_unpackhi_epi8(S0, S1);
                        t1 = _mm_unpackhi_epi8(S2, S3);
                        t2 = _mm_unpacklo_epi16(t0, t1);
                        t3 = _mm_unpackhi_epi16(t0, t1);

                        t0 = _mm_maddubs_epi16(t2, c0);
                        t1 = _mm_maddubs_epi16(t3, c0);

                        D1 = _mm_hadds_epi16(t0, t1);
                        D1 = _mm_add_epi16(D1, off);
                        D1 = _mm_srli_epi16(D1, 7);

                        D0 = _mm_packus_epi16(D0, D1);

                        _mm_store_si128((__m128i*)(dst + i), D0);
                        //dst[i] = (pel_t)((src[idx] * c1 + src[idx + 1] * c2 + src[idx + 2] * c3 + src[idx + 3] * c4 + 64) >> 7);
                    }

                    if (real_width < bsx) {
                        D0 = _mm_set1_epi8((char)dst[real_width - 1]);
                        for (i = real_width; i < bsx; i += 16) {
                            _mm_storeu_si128((__m128i*)(dst + i), D0);
                            //dst[i] = dst[real_width - 1];
                        }
                    }

                }

                dst += i_dst;
            }
        }
    }

}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_x_8_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[2 * (64 + 48)]);
    int line_size = bsx + (bsy >> 1) - 1;
    int i;
    int aligned_line_size = ((line_size + 31) >> 4) << 4;
    pel_t *pfirst[2];
    __m128i zero = _mm_setzero_si128();
    __m128i coeff = _mm_set1_epi16(3);
    __m128i offset1 = _mm_set1_epi16(4);
    __m128i offset2 = _mm_set1_epi16(2);
    int i_dst2 = i_dst * 2;

    UNUSED_PARAMETER(dir_mode);

    pfirst[0] = first_line;
    pfirst[1] = first_line + aligned_line_size;

    for (i = 0; i < line_size - 8; i += 16, src += 16) {
        __m128i p01, p02, p11, p12;
        __m128i S0 = _mm_loadu_si128((__m128i*)(src));
        __m128i S3 = _mm_loadu_si128((__m128i*)(src + 3));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 2));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);
        __m128i L3 = _mm_unpacklo_epi8(S3, zero);

        __m128i H0 = _mm_unpackhi_epi8(S0, zero);
        __m128i H1 = _mm_unpackhi_epi8(S1, zero);
        __m128i H2 = _mm_unpackhi_epi8(S2, zero);
        __m128i H3 = _mm_unpackhi_epi8(S3, zero);

        p01 = _mm_add_epi16(L1, L2);
        p01 = _mm_mullo_epi16(p01, coeff);
        p02 = _mm_add_epi16(L0, L3);
        p02 = _mm_add_epi16(p02, offset1);
        p01 = _mm_add_epi16(p01, p02);
        p01 = _mm_srli_epi16(p01, 3);

        p11 = _mm_add_epi16(H1, H2);
        p11 = _mm_mullo_epi16(p11, coeff);
        p12 = _mm_add_epi16(H0, H3);
        p12 = _mm_add_epi16(p12, offset1);
        p11 = _mm_add_epi16(p11, p12);
        p11 = _mm_srli_epi16(p11, 3);

        p01 = _mm_packus_epi16(p01, p11);
        _mm_store_si128((__m128i*)&pfirst[0][i], p01);

        p01 = _mm_add_epi16(L1, L2);
        p02 = _mm_add_epi16(L2, L3);
        p11 = _mm_add_epi16(H1, H2);
        p12 = _mm_add_epi16(H2, H3);

        p01 = _mm_add_epi16(p01, p02);
        p11 = _mm_add_epi16(p11, p12);

        p01 = _mm_add_epi16(p01, offset2);
        p11 = _mm_add_epi16(p11, offset2);

        p01 = _mm_srli_epi16(p01, 2);
        p11 = _mm_srli_epi16(p11, 2);

        p01 = _mm_packus_epi16(p01, p11);
        _mm_store_si128((__m128i*)&pfirst[1][i], p01);
    }

    if (i < line_size) {
        __m128i p01, p02;
        __m128i S0 = _mm_loadu_si128((__m128i*)(src));
        __m128i S3 = _mm_loadu_si128((__m128i*)(src + 3));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 2));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);
        __m128i L3 = _mm_unpacklo_epi8(S3, zero);

        p01 = _mm_add_epi16(L1, L2);
        p01 = _mm_mullo_epi16(p01, coeff);
        p02 = _mm_add_epi16(L0, L3);
        p02 = _mm_add_epi16(p02, offset1);
        p01 = _mm_add_epi16(p01, p02);
        p01 = _mm_srli_epi16(p01, 3);

        p01 = _mm_packus_epi16(p01, p01);
        _mm_storel_epi64((__m128i*)&pfirst[0][i], p01);

        p01 = _mm_add_epi16(L1, L2);
        p02 = _mm_add_epi16(L2, L3);

        p01 = _mm_add_epi16(p01, p02);
        p01 = _mm_add_epi16(p01, offset2);
        p01 = _mm_srli_epi16(p01, 2);

        p01 = _mm_packus_epi16(p01, p01);
        _mm_storel_epi64((__m128i*)&pfirst[1][i], p01);
    }

    bsy >>= 1;

    if (bsx != 8) {
        for (i = 0; i < bsy; i++) {
            memcpy(dst, pfirst[0] + i, bsx * sizeof(pel_t));
            memcpy(dst + i_dst, pfirst[1] + i, bsx * sizeof(pel_t));
            dst += i_dst2;
        }
    } else if (bsy == 4) {
        __m128i M1 = _mm_loadu_si128((__m128i*)&pfirst[0][0]);
        __m128i M2 = _mm_loadu_si128((__m128i*)&pfirst[1][0]);
        _mm_storel_epi64((__m128i*)dst, M1);
        _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
        dst += i_dst2;
        M1 = _mm_srli_si128(M1, 1);
        M2 = _mm_srli_si128(M2, 1);
        _mm_storel_epi64((__m128i*)dst, M1);
        _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
        dst += i_dst2;
        M1 = _mm_srli_si128(M1, 1);
        M2 = _mm_srli_si128(M2, 1);
        _mm_storel_epi64((__m128i*)dst, M1);
        _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
        dst += i_dst2;
        M1 = _mm_srli_si128(M1, 1);
        M2 = _mm_srli_si128(M2, 1);
        _mm_storel_epi64((__m128i*)dst, M1);
        _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
    } else {
        for (i = 0; i < 16; i = i + 8) {
            __m128i M1 = _mm_loadu_si128((__m128i*)&pfirst[0][i]);
            __m128i M2 = _mm_loadu_si128((__m128i*)&pfirst[1][i]);
            _mm_storel_epi64((__m128i*)dst, M1);
            _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
            dst += i_dst2;
            M1 = _mm_srli_si128(M1, 1);
            M2 = _mm_srli_si128(M2, 1);
            _mm_storel_epi64((__m128i*)dst, M1);
            _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
            dst += i_dst2;
            M1 = _mm_srli_si128(M1, 1);
            M2 = _mm_srli_si128(M2, 1);
            _mm_storel_epi64((__m128i*)dst, M1);
            _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
            dst += i_dst2;
            M1 = _mm_srli_si128(M1, 1);
            M2 = _mm_srli_si128(M2, 1);
            _mm_storel_epi64((__m128i*)dst, M1);
            _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
            dst += i_dst2;
            M1 = _mm_srli_si128(M1, 1);
            M2 = _mm_srli_si128(M2, 1);
            _mm_storel_epi64((__m128i*)dst, M1);
            _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
            dst += i_dst2;
            M1 = _mm_srli_si128(M1, 1);
            M2 = _mm_srli_si128(M2, 1);
            _mm_storel_epi64((__m128i*)dst, M1);
            _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
            dst += i_dst2;
            M1 = _mm_srli_si128(M1, 1);
            M2 = _mm_srli_si128(M2, 1);
            _mm_storel_epi64((__m128i*)dst, M1);
            _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
            dst += i_dst2;
            M1 = _mm_srli_si128(M1, 1);
            M2 = _mm_srli_si128(M2, 1);
            _mm_storel_epi64((__m128i*)dst, M1);
            _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
            dst += i_dst2;
        }
    }

}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_x_9_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i, j;
    int iWidth2 = bsx << 1;
    __m128i zero = _mm_setzero_si128();
    __m128i S0, S1, S2, S3;
    __m128i t0, t1, t2, t3;
    __m128i off = _mm_set1_epi16(64);
    __m128i c0;

    UNUSED_PARAMETER(dir_mode);

    if (bsx >= bsy) {
        if (bsx & 0x07) {
            __m128i D0;
            int i_dst2 = i_dst << 1;

            for (j = 0; j < bsy; j += 2) {
                int idx = tab_idx_mode_9[j];
                c0 = _mm_load_si128((__m128i*)tab_coeff_mode_9[j]);

                S0 = _mm_loadl_epi64((__m128i*)(src + idx));
                S1 = _mm_srli_si128(S0, 1);
                S2 = _mm_srli_si128(S0, 2);
                S3 = _mm_srli_si128(S0, 3);

                t0 = _mm_unpacklo_epi8(S0, S1);
                t1 = _mm_unpacklo_epi8(S2, S3);
                t2 = _mm_unpacklo_epi16(t0, t1);

                t0 = _mm_maddubs_epi16(t2, c0);

                idx = tab_idx_mode_9[j + 1];
                c0 = _mm_load_si128((__m128i*)tab_coeff_mode_9[j + 1]);
                S0 = _mm_loadl_epi64((__m128i*)(src + idx));
                S1 = _mm_srli_si128(S0, 1);
                S2 = _mm_srli_si128(S0, 2);
                S3 = _mm_srli_si128(S0, 3);

                t1 = _mm_unpacklo_epi8(S0, S1);
                t2 = _mm_unpacklo_epi8(S2, S3);
                t1 = _mm_unpacklo_epi16(t1, t2);

                t1 = _mm_maddubs_epi16(t1, c0);

                D0 = _mm_hadds_epi16(t0, t1);
                D0 = _mm_add_epi16(D0, off);
                D0 = _mm_srli_epi16(D0, 7);
                D0 = _mm_packus_epi16(D0, zero);

                ((uint32_t*)(dst))[0] = _mm_cvtsi128_si32(D0);
                D0 = _mm_srli_si128(D0, 4);
                ((uint32_t*)(dst + i_dst))[0] = _mm_cvtsi128_si32(D0);
                //_mm_maskmoveu_si128(D0, mask, (char*)(dst + i_dst));
                //dst[i] = (pel_t)((src[idx] * c1 + src[idx + 1] * c2 + src[idx + 2] * c3 + src[idx + 3] * c4 + 64) >> 7);
                dst += i_dst2;
            }
        } else if (bsx & 0x0f) {
            __m128i D0;

            for (j = 0; j < bsy; j++) {
                int idx = tab_idx_mode_9[j];
                c0 = _mm_load_si128((__m128i*)tab_coeff_mode_9[j]);

                S0 = _mm_loadu_si128((__m128i*)(src + idx));
                S1 = _mm_srli_si128(S0, 1);
                S2 = _mm_srli_si128(S0, 2);
                S3 = _mm_srli_si128(S0, 3);

                t0 = _mm_unpacklo_epi8(S0, S1);
                t1 = _mm_unpacklo_epi8(S2, S3);
                t2 = _mm_unpacklo_epi16(t0, t1);
                t3 = _mm_unpackhi_epi16(t0, t1);

                t0 = _mm_maddubs_epi16(t2, c0);
                t1 = _mm_maddubs_epi16(t3, c0);

                D0 = _mm_hadds_epi16(t0, t1);
                D0 = _mm_add_epi16(D0, off);
                D0 = _mm_srli_epi16(D0, 7);

                D0 = _mm_packus_epi16(D0, _mm_setzero_si128());

                _mm_storel_epi64((__m128i*)(dst), D0);
                //dst[i] = (pel_t)((src[idx] * c1 + src[idx + 1] * c2 + src[idx + 2] * c3 + src[idx + 3] * c4 + 64) >> 7);

                dst += i_dst;
            }
        } else {
            for (j = 0; j < bsy; j++) {
                __m128i D0, D1;

                int idx = tab_idx_mode_9[j];
                c0 = _mm_load_si128((__m128i*)tab_coeff_mode_9[j]);

                for (i = 0; i < bsx; i += 16, idx += 16) {
                    S0 = _mm_loadu_si128((__m128i*)(src + idx));
                    S1 = _mm_loadu_si128((__m128i*)(src + idx + 1));
                    S2 = _mm_loadu_si128((__m128i*)(src + idx + 2));
                    S3 = _mm_loadu_si128((__m128i*)(src + idx + 3));

                    t0 = _mm_unpacklo_epi8(S0, S1);
                    t1 = _mm_unpacklo_epi8(S2, S3);
                    t2 = _mm_unpacklo_epi16(t0, t1);
                    t3 = _mm_unpackhi_epi16(t0, t1);

                    t0 = _mm_maddubs_epi16(t2, c0);
                    t1 = _mm_maddubs_epi16(t3, c0);

                    D0 = _mm_hadds_epi16(t0, t1);
                    D0 = _mm_add_epi16(D0, off);
                    D0 = _mm_srli_epi16(D0, 7);

                    t0 = _mm_unpackhi_epi8(S0, S1);
                    t1 = _mm_unpackhi_epi8(S2, S3);
                    t2 = _mm_unpacklo_epi16(t0, t1);
                    t3 = _mm_unpackhi_epi16(t0, t1);

                    t0 = _mm_maddubs_epi16(t2, c0);
                    t1 = _mm_maddubs_epi16(t3, c0);

                    D1 = _mm_hadds_epi16(t0, t1);
                    D1 = _mm_add_epi16(D1, off);
                    D1 = _mm_srli_epi16(D1, 7);

                    D0 = _mm_packus_epi16(D0, D1);

                    _mm_storeu_si128((__m128i*)(dst + i), D0);
                    //dst[i] = (pel_t)((src[idx] * c1 + src[idx + 1] * c2 + src[idx + 2] * c3 + src[idx + 3] * c4 + 64) >> 7);
                }

                dst += i_dst;
            }
        }
    } else {
        if (bsx & 0x07) {
            for (j = 0; j < bsy; j++) {
                int real_width;
                int idx = tab_idx_mode_9[j];

                real_width = XAVS2_MIN(bsx, iWidth2 - idx + 1);

                if (real_width <= 0) {
                    pel_t val = (pel_t)((src[iWidth2] * tab_coeff_mode_9[j][0] + src[iWidth2 + 1] * tab_coeff_mode_9[j][1] + src[iWidth2 + 2] * tab_coeff_mode_9[j][2] + src[iWidth2 + 3] * tab_coeff_mode_9[j][3] + 64) >> 7);
                    __m128i D0 = _mm_set1_epi8((char)val);
                    _mm_storel_epi64((__m128i*)(dst), D0);
                    dst += i_dst;
                    j++;

                    for (; j < bsy; j++) {
                        val = (pel_t)((src[iWidth2] * tab_coeff_mode_9[j][0] + src[iWidth2 + 1] * tab_coeff_mode_9[j][1] + src[iWidth2 + 2] * tab_coeff_mode_9[j][2] + src[iWidth2 + 3] * tab_coeff_mode_9[j][3] + 64) >> 7);
                        D0 = _mm_set1_epi8((char)val);
                        _mm_storel_epi64((__m128i*)(dst), D0);
                        dst += i_dst;
                    }
                    break;
                } else {
                    __m128i D0;
                    c0 = _mm_load_si128((__m128i*)tab_coeff_mode_9[j]);

                    S0 = _mm_loadl_epi64((__m128i*)(src + idx));
                    S1 = _mm_srli_si128(S0, 1);
                    S2 = _mm_srli_si128(S0, 2);
                    S3 = _mm_srli_si128(S0, 3);

                    t0 = _mm_unpacklo_epi8(S0, S1);
                    t1 = _mm_unpacklo_epi8(S2, S3);
                    t2 = _mm_unpacklo_epi16(t0, t1);

                    t0 = _mm_maddubs_epi16(t2, c0);

                    D0 = _mm_hadds_epi16(t0, zero);
                    D0 = _mm_add_epi16(D0, off);
                    D0 = _mm_srli_epi16(D0, 7);

                    D0 = _mm_packus_epi16(D0, zero);

                    _mm_storel_epi64((__m128i*)(dst), D0);

                    if (real_width < bsx) {
                        D0 = _mm_set1_epi8((char)dst[real_width - 1]);
                        _mm_storel_epi64((__m128i*)(dst + real_width), D0);
                    }
                }
                dst += i_dst;
            }
        } else if (bsx & 0x0f) {
            for (j = 0; j < bsy; j++) {
                int real_width;
                int idx = tab_idx_mode_9[j];

                real_width = XAVS2_MIN(bsx, iWidth2 - idx + 1);

                if (real_width <= 0) {
                    pel_t val = (pel_t)((src[iWidth2] * tab_coeff_mode_9[j][0] + src[iWidth2 + 1] * tab_coeff_mode_9[j][1] + src[iWidth2 + 2] * tab_coeff_mode_9[j][2] + src[iWidth2 + 3] * tab_coeff_mode_9[j][3] + 64) >> 7);
                    __m128i D0 = _mm_set1_epi8((char)val);
                    _mm_storel_epi64((__m128i*)(dst), D0);
                    dst += i_dst;
                    j++;

                    for (; j < bsy; j++) {
                        val = (pel_t)((src[iWidth2] * tab_coeff_mode_9[j][0] + src[iWidth2 + 1] * tab_coeff_mode_9[j][1] + src[iWidth2 + 2] * tab_coeff_mode_9[j][2] + src[iWidth2 + 3] * tab_coeff_mode_9[j][3] + 64) >> 7);
                        D0 = _mm_set1_epi8((char)val);
                        _mm_storel_epi64((__m128i*)(dst), D0);
                        dst += i_dst;
                    }
                    break;
                } else {
                    __m128i D0;
                    c0 = _mm_load_si128((__m128i*)tab_coeff_mode_9[j]);

                    S0 = _mm_loadu_si128((__m128i*)(src + idx));
                    S1 = _mm_srli_si128(S0, 1);
                    S2 = _mm_srli_si128(S0, 2);
                    S3 = _mm_srli_si128(S0, 3);

                    t0 = _mm_unpacklo_epi8(S0, S1);
                    t1 = _mm_unpacklo_epi8(S2, S3);
                    t2 = _mm_unpacklo_epi16(t0, t1);
                    t3 = _mm_unpackhi_epi16(t0, t1);

                    t0 = _mm_maddubs_epi16(t2, c0);
                    t1 = _mm_maddubs_epi16(t3, c0);

                    D0 = _mm_hadds_epi16(t0, t1);
                    D0 = _mm_add_epi16(D0, off);
                    D0 = _mm_srli_epi16(D0, 7);

                    D0 = _mm_packus_epi16(D0, zero);

                    _mm_storel_epi64((__m128i*)(dst), D0);
                    //dst[i] = (pel_t)((src[idx] * c1 + src[idx + 1] * c2 + src[idx + 2] * c3 + src[idx + 3] * c4 + 64) >> 7);

                    if (real_width < bsx) {
                        D0 = _mm_set1_epi8((char)dst[real_width - 1]);
                        _mm_storel_epi64((__m128i*)(dst + real_width), D0);
                    }

                }

                dst += i_dst;
            }
        } else {
            for (j = 0; j < bsy; j++) {
                int real_width;
                int idx = tab_idx_mode_9[j];

                real_width = XAVS2_MIN(bsx, iWidth2 - idx + 1);

                if (real_width <= 0) {
                    pel_t val = (pel_t)((src[iWidth2] * tab_coeff_mode_9[j][0] + src[iWidth2 + 1] * tab_coeff_mode_9[j][1] + src[iWidth2 + 2] * tab_coeff_mode_9[j][2] + src[iWidth2 + 3] * tab_coeff_mode_9[j][3] + 64) >> 7);
                    __m128i D0 = _mm_set1_epi8((char)val);

                    for (i = 0; i < bsx; i += 16) {
                        _mm_storeu_si128((__m128i*)(dst + i), D0);
                    }
                    dst += i_dst;
                    j++;

                    for (; j < bsy; j++) {
                        val = (pel_t)((src[iWidth2] * tab_coeff_mode_9[j][0] + src[iWidth2 + 1] * tab_coeff_mode_9[j][1] + src[iWidth2 + 2] * tab_coeff_mode_9[j][2] + src[iWidth2 + 3] * tab_coeff_mode_9[j][3] + 64) >> 7);
                        D0 = _mm_set1_epi8((char)val);
                        for (i = 0; i < bsx; i += 16) {
                            _mm_storeu_si128((__m128i*)(dst + i), D0);
                        }
                        dst += i_dst;
                    }
                    break;
                } else {
                    __m128i D0, D1;

                    c0 = _mm_load_si128((__m128i*)tab_coeff_mode_9[j]);
                    for (i = 0; i < real_width; i += 16, idx += 16) {
                        S0 = _mm_loadu_si128((__m128i*)(src + idx));
                        S1 = _mm_loadu_si128((__m128i*)(src + idx + 1));
                        S2 = _mm_loadu_si128((__m128i*)(src + idx + 2));
                        S3 = _mm_loadu_si128((__m128i*)(src + idx + 3));

                        t0 = _mm_unpacklo_epi8(S0, S1);
                        t1 = _mm_unpacklo_epi8(S2, S3);
                        t2 = _mm_unpacklo_epi16(t0, t1);
                        t3 = _mm_unpackhi_epi16(t0, t1);

                        t0 = _mm_maddubs_epi16(t2, c0);
                        t1 = _mm_maddubs_epi16(t3, c0);

                        D0 = _mm_hadds_epi16(t0, t1);
                        D0 = _mm_add_epi16(D0, off);
                        D0 = _mm_srli_epi16(D0, 7);

                        t0 = _mm_unpackhi_epi8(S0, S1);
                        t1 = _mm_unpackhi_epi8(S2, S3);
                        t2 = _mm_unpacklo_epi16(t0, t1);
                        t3 = _mm_unpackhi_epi16(t0, t1);

                        t0 = _mm_maddubs_epi16(t2, c0);
                        t1 = _mm_maddubs_epi16(t3, c0);

                        D1 = _mm_hadds_epi16(t0, t1);
                        D1 = _mm_add_epi16(D1, off);
                        D1 = _mm_srli_epi16(D1, 7);

                        D0 = _mm_packus_epi16(D0, D1);

                        _mm_store_si128((__m128i*)(dst + i), D0);
                        //dst[i] = (pel_t)((src[idx] * c1 + src[idx + 1] * c2 + src[idx + 2] * c3 + src[idx + 3] * c4 + 64) >> 7);
                    }

                    if (real_width < bsx) {
                        D0 = _mm_set1_epi8((char)dst[real_width - 1]);
                        for (i = real_width; i < bsx; i += 16) {
                            _mm_storeu_si128((__m128i*)(dst + i), D0);
                            //dst[i] = dst[real_width - 1];
                        }
                    }

                }

                dst += i_dst;
            }
        }
    }

}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_x_10_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;
    pel_t *dst1 = dst;
    pel_t *dst2 = dst1 + i_dst;
    pel_t *dst3 = dst2 + i_dst;
    pel_t *dst4 = dst3 + i_dst;
    __m128i zero = _mm_setzero_si128();
    __m128i coeff2 = _mm_set1_epi16(2);
    __m128i coeff3 = _mm_set1_epi16(3);
    __m128i coeff4 = _mm_set1_epi16(4);
    __m128i coeff5 = _mm_set1_epi16(5);
    __m128i coeff7 = _mm_set1_epi16(7);
    __m128i coeff8 = _mm_set1_epi16(8);

    UNUSED_PARAMETER(dir_mode);

    if (bsy != 4) {
        ALIGN16(pel_t first_line[4 * (64 + 32)]);
        int line_size = bsx + bsy / 4 - 1;
        int aligned_line_size = ((line_size + 31) >> 4) << 4;
        pel_t *pfirst[4];

        pfirst[0] = first_line;
        pfirst[1] = first_line + aligned_line_size;
        pfirst[2] = first_line + aligned_line_size * 2;
        pfirst[3] = first_line + aligned_line_size * 3;

        for (i = 0; i < line_size - 8; i += 16, src += 16) {
            __m128i p00, p10, p20, p30;
            __m128i p01, p11, p21, p31;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src + 3));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + 2));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);
            __m128i H3 = _mm_unpackhi_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_srli_epi16(p00, 4);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff7);
            p21 = _mm_mullo_epi16(H2, coeff5);
            p31 = _mm_add_epi16(H3, coeff8);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)&pfirst[0][i], p00);

            p00 = _mm_add_epi16(L1, L2);
            p00 = _mm_mullo_epi16(p00, coeff3);
            p10 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(p10, coeff4);
            p00 = _mm_add_epi16(p10, p00);
            p00 = _mm_srli_epi16(p00, 3);

            p01 = _mm_add_epi16(H1, H2);
            p01 = _mm_mullo_epi16(p01, coeff3);
            p11 = _mm_add_epi16(H0, H3);
            p11 = _mm_add_epi16(p11, coeff4);
            p01 = _mm_add_epi16(p11, p01);
            p01 = _mm_srli_epi16(p01, 3);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)&pfirst[1][i], p00);

            p10 = _mm_mullo_epi16(L1, coeff5);
            p20 = _mm_mullo_epi16(L2, coeff7);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(L0, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p11 = _mm_mullo_epi16(H1, coeff5);
            p21 = _mm_mullo_epi16(H2, coeff7);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(H0, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)&pfirst[2][i], p00);

            p00 = _mm_add_epi16(L1, L2);
            p10 = _mm_add_epi16(L2, L3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_srli_epi16(p00, 2);

            p01 = _mm_add_epi16(H1, H2);
            p11 = _mm_add_epi16(H2, H3);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, coeff2);
            p01 = _mm_srli_epi16(p01, 2);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)&pfirst[3][i], p00);
        }

        if (i < line_size) {
            __m128i p00, p10, p20, p30;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src + 3));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + 2));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[0][i], p00);

            p00 = _mm_add_epi16(L1, L2);
            p00 = _mm_mullo_epi16(p00, coeff3);
            p10 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(p10, coeff4);
            p00 = _mm_add_epi16(p10, p00);
            p00 = _mm_srli_epi16(p00, 3);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[1][i], p00);

            p10 = _mm_mullo_epi16(L1, coeff5);
            p20 = _mm_mullo_epi16(L2, coeff7);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(L0, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[2][i], p00);

            p00 = _mm_add_epi16(L1, L2);
            p10 = _mm_add_epi16(L2, L3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_srli_epi16(p00, 2);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[3][i], p00);
        }

        bsy >>= 2;

        if (bsx != 8) {
            int i_dstx4 = i_dst << 2;
            switch (bsx) {
            case 4:
                for (i = 0; i < bsy; i++) {
                    CP32(dst1, pfirst[0] + i);
                    dst1 += i_dstx4;
                    CP32(dst2, pfirst[1] + i);
                    dst2 += i_dstx4;
                    CP32(dst3, pfirst[2] + i);
                    dst3 += i_dstx4;
                    CP32(dst4, pfirst[3] + i);
                    dst4 += i_dstx4;
                }
                break;
            case 16:
                for (i = 0; i < bsy; i++) {
                    memcpy(dst1, pfirst[0] + i, 16 * sizeof(pel_t));
                    dst1 += i_dstx4;
                    memcpy(dst2, pfirst[1] + i, 16 * sizeof(pel_t));
                    dst2 += i_dstx4;
                    memcpy(dst3, pfirst[2] + i, 16 * sizeof(pel_t));
                    dst3 += i_dstx4;
                    memcpy(dst4, pfirst[3] + i, 16 * sizeof(pel_t));
                    dst4 += i_dstx4;
                }
                break;
            case 32:
                for (i = 0; i < bsy; i++) {
                    memcpy(dst1, pfirst[0] + i, 32 * sizeof(pel_t));
                    dst1 += i_dstx4;
                    memcpy(dst2, pfirst[1] + i, 32 * sizeof(pel_t));
                    dst2 += i_dstx4;
                    memcpy(dst3, pfirst[2] + i, 32 * sizeof(pel_t));
                    dst3 += i_dstx4;
                    memcpy(dst4, pfirst[3] + i, 32 * sizeof(pel_t));
                    dst4 += i_dstx4;
                }
                break;
            case 64:
                for (i = 0; i < bsy; i++) {
                    memcpy(dst1, pfirst[0] + i, 64 * sizeof(pel_t));
                    dst1 += i_dstx4;
                    memcpy(dst2, pfirst[1] + i, 64 * sizeof(pel_t));
                    dst2 += i_dstx4;
                    memcpy(dst3, pfirst[2] + i, 64 * sizeof(pel_t));
                    dst3 += i_dstx4;
                    memcpy(dst4, pfirst[3] + i, 64 * sizeof(pel_t));
                    dst4 += i_dstx4;
                }
                break;
            default:
                assert(0);
                break;
            }

        } else {
            if (bsy == 2) {
                for (i = 0; i < bsy; i++) {
                    CP64(dst1, pfirst[0] + i);
                    CP64(dst2, pfirst[1] + i);
                    CP64(dst3, pfirst[2] + i);
                    CP64(dst4, pfirst[3] + i);
                    dst1 = dst4 + i_dst;
                    dst2 = dst1 + i_dst;
                    dst3 = dst2 + i_dst;
                    dst4 = dst3 + i_dst;
                }
            } else {
                __m128i M1 = _mm_loadu_si128((__m128i*)&pfirst[0][0]);
                __m128i M2 = _mm_loadu_si128((__m128i*)&pfirst[1][0]);
                __m128i M3 = _mm_loadu_si128((__m128i*)&pfirst[2][0]);
                __m128i M4 = _mm_loadu_si128((__m128i*)&pfirst[3][0]);
                _mm_storel_epi64((__m128i*)dst1, M1);
                _mm_storel_epi64((__m128i*)dst2, M2);
                _mm_storel_epi64((__m128i*)dst3, M3);
                _mm_storel_epi64((__m128i*)dst4, M4);
                dst1 = dst4 + i_dst;
                dst2 = dst1 + i_dst;
                dst3 = dst2 + i_dst;
                dst4 = dst3 + i_dst;
                M1 = _mm_srli_si128(M1, 1);
                M2 = _mm_srli_si128(M2, 1);
                M3 = _mm_srli_si128(M3, 1);
                M4 = _mm_srli_si128(M4, 1);
                _mm_storel_epi64((__m128i*)dst1, M1);
                _mm_storel_epi64((__m128i*)dst2, M2);
                _mm_storel_epi64((__m128i*)dst3, M3);
                _mm_storel_epi64((__m128i*)dst4, M4);
                dst1 = dst4 + i_dst;
                dst2 = dst1 + i_dst;
                dst3 = dst2 + i_dst;
                dst4 = dst3 + i_dst;
                M1 = _mm_srli_si128(M1, 1);
                M2 = _mm_srli_si128(M2, 1);
                M3 = _mm_srli_si128(M3, 1);
                M4 = _mm_srli_si128(M4, 1);
                _mm_storel_epi64((__m128i*)dst1, M1);
                _mm_storel_epi64((__m128i*)dst2, M2);
                _mm_storel_epi64((__m128i*)dst3, M3);
                _mm_storel_epi64((__m128i*)dst4, M4);
                dst1 = dst4 + i_dst;
                dst2 = dst1 + i_dst;
                dst3 = dst2 + i_dst;
                dst4 = dst3 + i_dst;
                M1 = _mm_srli_si128(M1, 1);
                M2 = _mm_srli_si128(M2, 1);
                M3 = _mm_srli_si128(M3, 1);
                M4 = _mm_srli_si128(M4, 1);
                _mm_storel_epi64((__m128i*)dst1, M1);
                _mm_storel_epi64((__m128i*)dst2, M2);
                _mm_storel_epi64((__m128i*)dst3, M3);
                _mm_storel_epi64((__m128i*)dst4, M4);
                dst1 = dst4 + i_dst;
                dst2 = dst1 + i_dst;
                dst3 = dst2 + i_dst;
                dst4 = dst3 + i_dst;
                M1 = _mm_srli_si128(M1, 1);
                M2 = _mm_srli_si128(M2, 1);
                M3 = _mm_srli_si128(M3, 1);
                M4 = _mm_srli_si128(M4, 1);
                _mm_storel_epi64((__m128i*)dst1, M1);
                _mm_storel_epi64((__m128i*)dst2, M2);
                _mm_storel_epi64((__m128i*)dst3, M3);
                _mm_storel_epi64((__m128i*)dst4, M4);
                dst1 = dst4 + i_dst;
                dst2 = dst1 + i_dst;
                dst3 = dst2 + i_dst;
                dst4 = dst3 + i_dst;
                M1 = _mm_srli_si128(M1, 1);
                M2 = _mm_srli_si128(M2, 1);
                M3 = _mm_srli_si128(M3, 1);
                M4 = _mm_srli_si128(M4, 1);
                _mm_storel_epi64((__m128i*)dst1, M1);
                _mm_storel_epi64((__m128i*)dst2, M2);
                _mm_storel_epi64((__m128i*)dst3, M3);
                _mm_storel_epi64((__m128i*)dst4, M4);
                dst1 = dst4 + i_dst;
                dst2 = dst1 + i_dst;
                dst3 = dst2 + i_dst;
                dst4 = dst3 + i_dst;
                M1 = _mm_srli_si128(M1, 1);
                M2 = _mm_srli_si128(M2, 1);
                M3 = _mm_srli_si128(M3, 1);
                M4 = _mm_srli_si128(M4, 1);
                _mm_storel_epi64((__m128i*)dst1, M1);
                _mm_storel_epi64((__m128i*)dst2, M2);
                _mm_storel_epi64((__m128i*)dst3, M3);
                _mm_storel_epi64((__m128i*)dst4, M4);
                dst1 = dst4 + i_dst;
                dst2 = dst1 + i_dst;
                dst3 = dst2 + i_dst;
                dst4 = dst3 + i_dst;
                M1 = _mm_srli_si128(M1, 1);
                M2 = _mm_srli_si128(M2, 1);
                M3 = _mm_srli_si128(M3, 1);
                M4 = _mm_srli_si128(M4, 1);
                _mm_storel_epi64((__m128i*)dst1, M1);
                _mm_storel_epi64((__m128i*)dst2, M2);
                _mm_storel_epi64((__m128i*)dst3, M3);
                _mm_storel_epi64((__m128i*)dst4, M4);
            }
        }
    } else {
        if (bsx == 16) {
            __m128i p00, p10, p20, p30;
            __m128i p01, p11, p21, p31;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src + 3));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + 2));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);
            __m128i H3 = _mm_unpackhi_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_srli_epi16(p00, 4);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff7);
            p21 = _mm_mullo_epi16(H2, coeff5);
            p31 = _mm_add_epi16(H3, coeff8);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst1, p00);

            p00 = _mm_add_epi16(L1, L2);
            p00 = _mm_mullo_epi16(p00, coeff3);
            p10 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(p10, coeff4);
            p00 = _mm_add_epi16(p10, p00);
            p00 = _mm_srli_epi16(p00, 3);

            p01 = _mm_add_epi16(H1, H2);
            p01 = _mm_mullo_epi16(p01, coeff3);
            p11 = _mm_add_epi16(H0, H3);
            p11 = _mm_add_epi16(p11, coeff4);
            p01 = _mm_add_epi16(p11, p01);
            p01 = _mm_srli_epi16(p01, 3);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst2, p00);

            p10 = _mm_mullo_epi16(L1, coeff5);
            p20 = _mm_mullo_epi16(L2, coeff7);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(L0, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p11 = _mm_mullo_epi16(H1, coeff5);
            p21 = _mm_mullo_epi16(H2, coeff7);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(H0, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst3, p00);

            p00 = _mm_add_epi16(L1, L2);
            p10 = _mm_add_epi16(L2, L3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_srli_epi16(p00, 2);

            p01 = _mm_add_epi16(H1, H2);
            p11 = _mm_add_epi16(H2, H3);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, coeff2);
            p01 = _mm_srli_epi16(p01, 2);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst4, p00);
        } else {
            __m128i p00, p10, p20, p30;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src + 3));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + 2));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            ((int*)(dst1))[0] = _mm_cvtsi128_si32(p00);

            p00 = _mm_add_epi16(L1, L2);
            p00 = _mm_mullo_epi16(p00, coeff3);
            p10 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(p10, coeff4);
            p00 = _mm_add_epi16(p10, p00);
            p00 = _mm_srli_epi16(p00, 3);

            p00 = _mm_packus_epi16(p00, p00);
            ((int*)(dst2))[0] = _mm_cvtsi128_si32(p00);

            p10 = _mm_mullo_epi16(L1, coeff5);
            p20 = _mm_mullo_epi16(L2, coeff7);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(L0, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            ((int*)(dst3))[0] = _mm_cvtsi128_si32(p00);

            p00 = _mm_add_epi16(L1, L2);
            p10 = _mm_add_epi16(L2, L3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_srli_epi16(p00, 2);

            p00 = _mm_packus_epi16(p00, p00);
            ((int*)(dst4))[0] = _mm_cvtsi128_si32(p00);
        }
    }

}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_x_11_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i, j, idx;
    __m128i zero = _mm_setzero_si128();
    __m128i S0, S1, S2, S3;
    __m128i t0, t1, t2, t3;
    __m128i off = _mm_set1_epi16(64);
    __m128i c0;

    UNUSED_PARAMETER(dir_mode);

    if (bsx & 0x07) {
        __m128i D0;
        int i_dst2 = i_dst << 1;

        for (j = 0; j < bsy; j += 2) {
            idx = (j + 1) >> 3;
            c0 = _mm_load_si128((__m128i*)tab_coeff_mode_11[j & 0x07]);

            S0 = _mm_loadl_epi64((__m128i*)(src + idx));
            S1 = _mm_srli_si128(S0, 1);
            S2 = _mm_srli_si128(S0, 2);
            S3 = _mm_srli_si128(S0, 3);

            t0 = _mm_unpacklo_epi8(S0, S1);
            t1 = _mm_unpacklo_epi8(S2, S3);
            t2 = _mm_unpacklo_epi16(t0, t1);

            t0 = _mm_maddubs_epi16(t2, c0);

            idx = (j + 2) >> 3;
            c0 = _mm_load_si128((__m128i*)tab_coeff_mode_11[(j + 1) & 0x07]);
            S0 = _mm_loadl_epi64((__m128i*)(src + idx));
            S1 = _mm_srli_si128(S0, 1);
            S2 = _mm_srli_si128(S0, 2);
            S3 = _mm_srli_si128(S0, 3);

            t1 = _mm_unpacklo_epi8(S0, S1);
            t2 = _mm_unpacklo_epi8(S2, S3);
            t1 = _mm_unpacklo_epi16(t1, t2);

            t1 = _mm_maddubs_epi16(t1, c0);

            D0 = _mm_hadds_epi16(t0, t1);
            D0 = _mm_add_epi16(D0, off);
            D0 = _mm_srli_epi16(D0, 7);
            D0 = _mm_packus_epi16(D0, zero);

            ((uint32_t*)(dst))[0] = _mm_cvtsi128_si32(D0);
            D0 = _mm_srli_si128(D0, 4);
            ((uint32_t*)(dst + i_dst))[0] = _mm_cvtsi128_si32(D0);
            //dst[i] = (pel_t)((src[idx] * c1 + src[idx + 1] * c2 + src[idx + 2] * c3 + src[idx + 3] * c4 + 64) >> 7);
            dst += i_dst2;
        }
    } else if (bsx & 0x0f) {
        __m128i D0;

        for (j = 0; j < bsy; j++) {
            idx = (j + 1) >> 3;
            c0 = _mm_load_si128((__m128i*)tab_coeff_mode_11[j & 0x07]);

            S0 = _mm_loadu_si128((__m128i*)(src + idx));
            S1 = _mm_srli_si128(S0, 1);
            S2 = _mm_srli_si128(S0, 2);
            S3 = _mm_srli_si128(S0, 3);

            t0 = _mm_unpacklo_epi8(S0, S1);
            t1 = _mm_unpacklo_epi8(S2, S3);
            t2 = _mm_unpacklo_epi16(t0, t1);
            t3 = _mm_unpackhi_epi16(t0, t1);

            t0 = _mm_maddubs_epi16(t2, c0);
            t1 = _mm_maddubs_epi16(t3, c0);

            D0 = _mm_hadds_epi16(t0, t1);
            D0 = _mm_add_epi16(D0, off);
            D0 = _mm_srli_epi16(D0, 7);

            D0 = _mm_packus_epi16(D0, _mm_setzero_si128());

            _mm_storel_epi64((__m128i*)(dst), D0);
            //dst[i] = (pel_t)((src[idx] * c1 + src[idx + 1] * c2 + src[idx + 2] * c3 + src[idx + 3] * c4 + 64) >> 7);

            dst += i_dst;
        }
    } else {
        for (j = 0; j < bsy; j++) {
            __m128i D0, D1;

            idx = (j + 1) >> 3;
            c0 = _mm_load_si128((__m128i*)tab_coeff_mode_11[j & 0x07]);

            for (i = 0; i < bsx; i += 16, idx += 16) {
                S0 = _mm_loadu_si128((__m128i*)(src + idx));
                S1 = _mm_loadu_si128((__m128i*)(src + idx + 1));
                S2 = _mm_loadu_si128((__m128i*)(src + idx + 2));
                S3 = _mm_loadu_si128((__m128i*)(src + idx + 3));

                t0 = _mm_unpacklo_epi8(S0, S1);
                t1 = _mm_unpacklo_epi8(S2, S3);
                t2 = _mm_unpacklo_epi16(t0, t1);
                t3 = _mm_unpackhi_epi16(t0, t1);

                t0 = _mm_maddubs_epi16(t2, c0);
                t1 = _mm_maddubs_epi16(t3, c0);

                D0 = _mm_hadds_epi16(t0, t1);
                D0 = _mm_add_epi16(D0, off);
                D0 = _mm_srli_epi16(D0, 7);

                t0 = _mm_unpackhi_epi8(S0, S1);
                t1 = _mm_unpackhi_epi8(S2, S3);
                t2 = _mm_unpacklo_epi16(t0, t1);
                t3 = _mm_unpackhi_epi16(t0, t1);

                t0 = _mm_maddubs_epi16(t2, c0);
                t1 = _mm_maddubs_epi16(t3, c0);

                D1 = _mm_hadds_epi16(t0, t1);
                D1 = _mm_add_epi16(D1, off);
                D1 = _mm_srli_epi16(D1, 7);

                D0 = _mm_packus_epi16(D0, D1);

                _mm_storeu_si128((__m128i*)(dst + i), D0);
                //dst[i] = (pel_t)((src[idx] * c1 + src[idx + 1] * c2 + src[idx + 2] * c3 + src[idx + 3] * c4 + 64) >> 7);
            }

            dst += i_dst;
        }
    }

}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_y_25_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;

    UNUSED_PARAMETER(dir_mode);

    if (bsx > 8) {
        ALIGN16(pel_t first_line[64 + (64 << 3)]);
        int line_size = bsx + ((bsy - 1) << 3);
        int iHeight8 = bsy << 3;
        pel_t *pfirst = first_line;

        __m128i coeff0 = _mm_setr_epi16(7, 3, 5, 1, 3, 1, 1, 0);
        __m128i coeff1 = _mm_setr_epi16(15, 7, 13, 3, 11, 5, 9, 1);
        __m128i coeff2 = _mm_setr_epi16(9, 5, 11, 3, 13, 7, 15, 2);
        __m128i coeff3 = _mm_setr_epi16(1, 1, 3, 1, 5, 3, 7, 1);
        __m128i coeff4 = _mm_setr_epi16(16, 8, 16, 4, 16, 8, 16, 2);
        __m128i coeff5 = _mm_setr_epi16(1, 2, 1, 4, 1, 2, 1, 8);

        __m128i p00, p10, p20, p30;

        __m128i L0 = _mm_set1_epi16(src[0]);
        __m128i L1 = _mm_set1_epi16(src[-1]);
        __m128i L2 = _mm_set1_epi16(src[-2]);
        __m128i L3 = _mm_set1_epi16(src[-3]);

        src -= 4;

        for (i = 0; i < line_size - 24; i += 32, src -= 4) {
            p00 = _mm_mullo_epi16(L0, coeff0);
            p10 = _mm_mullo_epi16(L1, coeff1);
            p20 = _mm_mullo_epi16(L2, coeff2);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)pfirst, p00);

            pfirst += 8;
            L0 = _mm_set1_epi16(src[0]);

            p00 = _mm_mullo_epi16(L1, coeff0);
            p10 = _mm_mullo_epi16(L2, coeff1);
            p20 = _mm_mullo_epi16(L3, coeff2);
            p30 = _mm_mullo_epi16(L0, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)pfirst, p00);

            pfirst += 8;
            L1 = _mm_set1_epi16(src[-1]);

            p00 = _mm_mullo_epi16(L2, coeff0);
            p10 = _mm_mullo_epi16(L3, coeff1);
            p20 = _mm_mullo_epi16(L0, coeff2);
            p30 = _mm_mullo_epi16(L1, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)pfirst, p00);

            pfirst += 8;
            L2 = _mm_set1_epi16(src[-2]);

            p00 = _mm_mullo_epi16(L3, coeff0);
            p10 = _mm_mullo_epi16(L0, coeff1);
            p20 = _mm_mullo_epi16(L1, coeff2);
            p30 = _mm_mullo_epi16(L2, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)pfirst, p00);

            pfirst += 8;
            L3 = _mm_set1_epi16(src[-3]);
        }

        if (bsx == 16) {
            p00 = _mm_mullo_epi16(L0, coeff0);
            p10 = _mm_mullo_epi16(L1, coeff1);
            p20 = _mm_mullo_epi16(L2, coeff2);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)pfirst, p00);
        } else {
            p00 = _mm_mullo_epi16(L0, coeff0);
            p10 = _mm_mullo_epi16(L1, coeff1);
            p20 = _mm_mullo_epi16(L2, coeff2);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)pfirst, p00);

            pfirst += 8;
            L0 = _mm_set1_epi16(src[0]);

            p00 = _mm_mullo_epi16(L1, coeff0);
            p10 = _mm_mullo_epi16(L2, coeff1);
            p20 = _mm_mullo_epi16(L3, coeff2);
            p30 = _mm_mullo_epi16(L0, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)pfirst, p00);

            pfirst += 8;
            L1 = _mm_set1_epi16(src[-1]);

            p00 = _mm_mullo_epi16(L2, coeff0);
            p10 = _mm_mullo_epi16(L3, coeff1);
            p20 = _mm_mullo_epi16(L0, coeff2);
            p30 = _mm_mullo_epi16(L1, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)pfirst, p00);
        }

        for (i = 0; i < iHeight8; i += 8) {
            memcpy(dst, first_line + i, bsx * sizeof(pel_t));
            dst += i_dst;
        }
    } else if (bsx == 8) {
        __m128i coeff0 = _mm_setr_epi16(7, 3, 5, 1, 3, 1, 1, 0);
        __m128i coeff1 = _mm_setr_epi16(15, 7, 13, 3, 11, 5, 9, 1);
        __m128i coeff2 = _mm_setr_epi16(9, 5, 11, 3, 13, 7, 15, 2);
        __m128i coeff3 = _mm_setr_epi16(1, 1, 3, 1, 5, 3, 7, 1);
        __m128i coeff4 = _mm_setr_epi16(16, 8, 16, 4, 16, 8, 16, 2);
        __m128i coeff5 = _mm_setr_epi16(1, 2, 1, 4, 1, 2, 1, 8);

        __m128i p00, p10, p20, p30;

        __m128i L0 = _mm_set1_epi16(src[0]);
        __m128i L1 = _mm_set1_epi16(src[-1]);
        __m128i L2 = _mm_set1_epi16(src[-2]);
        __m128i L3 = _mm_set1_epi16(src[-3]);
        src -= 4;

        bsy >>= 2;
        for (i = 0; i < bsy; i++, src -= 4) {
            p00 = _mm_mullo_epi16(L0, coeff0);
            p10 = _mm_mullo_epi16(L1, coeff1);
            p20 = _mm_mullo_epi16(L2, coeff2);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)dst, p00);

            dst += i_dst;
            L0 = _mm_set1_epi16(src[0]);

            p00 = _mm_mullo_epi16(L1, coeff0);
            p10 = _mm_mullo_epi16(L2, coeff1);
            p20 = _mm_mullo_epi16(L3, coeff2);
            p30 = _mm_mullo_epi16(L0, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)dst, p00);

            dst += i_dst;
            L1 = _mm_set1_epi16(src[-1]);

            p00 = _mm_mullo_epi16(L2, coeff0);
            p10 = _mm_mullo_epi16(L3, coeff1);
            p20 = _mm_mullo_epi16(L0, coeff2);
            p30 = _mm_mullo_epi16(L1, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)dst, p00);

            dst += i_dst;
            L2 = _mm_set1_epi16(src[-2]);

            p00 = _mm_mullo_epi16(L3, coeff0);
            p10 = _mm_mullo_epi16(L0, coeff1);
            p20 = _mm_mullo_epi16(L1, coeff2);
            p30 = _mm_mullo_epi16(L2, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)dst, p00);

            dst += i_dst;
            L3 = _mm_set1_epi16(src[-3]);
        }
    } else {
        __m128i zero = _mm_setzero_si128();
        __m128i coeff3 = _mm_set1_epi16(3);
        __m128i coeff4 = _mm_set1_epi16(4);
        __m128i coeff5 = _mm_set1_epi16(5);
        __m128i coeff7 = _mm_set1_epi16(7);
        __m128i coeff8 = _mm_set1_epi16(8);
        __m128i coeff9 = _mm_set1_epi16(9);
        __m128i coeff11 = _mm_set1_epi16(11);
        __m128i coeff13 = _mm_set1_epi16(13);
        __m128i coeff15 = _mm_set1_epi16(15);
        __m128i coeff16 = _mm_set1_epi16(16);
        __m128i shuffle = _mm_setr_epi8(7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 0, 8);

        if (bsy == 4) {
            src -= 15;
            __m128i p01, p11, p21, p31;
            __m128i M2, M4, M6, M8;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src - 3));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src - 2));

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);
            __m128i H3 = _mm_unpackhi_epi8(S3, zero);

            p01 = _mm_mullo_epi16(H0, coeff7);
            p11 = _mm_mullo_epi16(H1, coeff15);
            p21 = _mm_mullo_epi16(H2, coeff9);
            p31 = _mm_add_epi16(H3, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            M2 = _mm_srli_epi16(p01, 5);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff7);
            p21 = _mm_mullo_epi16(H2, coeff5);
            p31 = _mm_add_epi16(H3, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            M4 = _mm_srli_epi16(p01, 4);

            p01 = _mm_mullo_epi16(H0, coeff5);
            p11 = _mm_mullo_epi16(H1, coeff13);
            p21 = _mm_mullo_epi16(H2, coeff11);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(p01, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            M6 = _mm_srli_epi16(p01, 5);

            p01 = _mm_add_epi16(H0, H3);
            p11 = _mm_add_epi16(H1, H2);
            p11 = _mm_mullo_epi16(p11, coeff3);
            p01 = _mm_add_epi16(p01, coeff4);
            p01 = _mm_add_epi16(p01, p11);
            M8 = _mm_srli_epi16(p01, 3);

            M2 = _mm_packus_epi16(M2, M4);
            M6 = _mm_packus_epi16(M6, M8);
            M2 = _mm_shuffle_epi8(M2, shuffle);
            M6 = _mm_shuffle_epi8(M6, shuffle);

            M4 = _mm_unpacklo_epi16(M2, M6);

            *((int*)dst) = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M4);
        } else {
            src -= 15;

            __m128i p00, p10, p20, p30;
            __m128i p01, p11, p21, p31;
            __m128i M1, M2, M3, M4, M5, M6, M7, M8;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src - 3));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src - 2));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);
            __m128i H3 = _mm_unpackhi_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff7);
            p10 = _mm_mullo_epi16(L1, coeff15);
            p20 = _mm_mullo_epi16(L2, coeff9);
            p30 = _mm_add_epi16(L3, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            M1 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H0, coeff7);
            p11 = _mm_mullo_epi16(H1, coeff15);
            p21 = _mm_mullo_epi16(H2, coeff9);
            p31 = _mm_add_epi16(H3, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            M2 = _mm_srli_epi16(p01, 5);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            M3 = _mm_srli_epi16(p00, 4);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff7);
            p21 = _mm_mullo_epi16(H2, coeff5);
            p31 = _mm_add_epi16(H3, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            M4 = _mm_srli_epi16(p01, 4);

            p00 = _mm_mullo_epi16(L0, coeff5);
            p10 = _mm_mullo_epi16(L1, coeff13);
            p20 = _mm_mullo_epi16(L2, coeff11);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            M5 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H0, coeff5);
            p11 = _mm_mullo_epi16(H1, coeff13);
            p21 = _mm_mullo_epi16(H2, coeff11);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(p01, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            M6 = _mm_srli_epi16(p01, 5);

            p00 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(L1, L2);
            p10 = _mm_mullo_epi16(p10, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            M7 = _mm_srli_epi16(p00, 3);

            p01 = _mm_add_epi16(H0, H3);
            p11 = _mm_add_epi16(H1, H2);
            p11 = _mm_mullo_epi16(p11, coeff3);
            p01 = _mm_add_epi16(p01, coeff4);
            p01 = _mm_add_epi16(p01, p11);
            M8 = _mm_srli_epi16(p01, 3);

            M1 = _mm_packus_epi16(M1, M3);
            M5 = _mm_packus_epi16(M5, M7);
            M1 = _mm_shuffle_epi8(M1, shuffle);
            M5 = _mm_shuffle_epi8(M5, shuffle);

            M2 = _mm_packus_epi16(M2, M4);
            M6 = _mm_packus_epi16(M6, M8);
            M2 = _mm_shuffle_epi8(M2, shuffle);
            M6 = _mm_shuffle_epi8(M6, shuffle);

            M3 = _mm_unpacklo_epi16(M1, M5);
            M7 = _mm_unpackhi_epi16(M1, M5);
            M4 = _mm_unpacklo_epi16(M2, M6);
            M8 = _mm_unpackhi_epi16(M2, M6);

            *((int*)dst) = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            *((int*)dst) = _mm_cvtsi128_si32(M8);
            dst += i_dst;
            M8 = _mm_srli_si128(M8, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M8);
            dst += i_dst;
            M8 = _mm_srli_si128(M8, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M8);
            dst += i_dst;
            M8 = _mm_srli_si128(M8, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M8);
            dst += i_dst;
            *((int*)dst) = _mm_cvtsi128_si32(M3);
            dst += i_dst;
            M3 = _mm_srli_si128(M3, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M3);
            dst += i_dst;
            M3 = _mm_srli_si128(M3, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M3);
            dst += i_dst;
            M3 = _mm_srli_si128(M3, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M3);
            dst += i_dst;
            *((int*)dst) = _mm_cvtsi128_si32(M7);
            dst += i_dst;
            M7 = _mm_srli_si128(M7, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M7);
            dst += i_dst;
            M7 = _mm_srli_si128(M7, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M7);
            dst += i_dst;
            M7 = _mm_srli_si128(M7, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M7);
        }
    }

}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_y_26_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;

    UNUSED_PARAMETER(dir_mode);
    if (bsx != 4) {
        __m128i zero = _mm_setzero_si128();
        __m128i coeff2 = _mm_set1_epi16(2);
        __m128i coeff3 = _mm_set1_epi16(3);
        __m128i coeff4 = _mm_set1_epi16(4);
        __m128i coeff5 = _mm_set1_epi16(5);
        __m128i coeff7 = _mm_set1_epi16(7);
        __m128i coeff8 = _mm_set1_epi16(8);
        __m128i shuffle = _mm_setr_epi8(7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 0, 8);

        ALIGN16(pel_t first_line[64 + 256]);
        int line_size = bsx + (bsy - 1) * 4;
        int iHeight4 = bsy << 2;

        src -= 15;

        for (i = 0; i < line_size - 32; i += 64, src -= 16) {
            __m128i p00, p10, p20, p30;
            __m128i p01, p11, p21, p31;
            __m128i M1, M2, M3, M4, M5, M6, M7, M8;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src - 3));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src - 2));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);
            __m128i H3 = _mm_unpackhi_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            M1 = _mm_srli_epi16(p00, 4);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff7);
            p21 = _mm_mullo_epi16(H2, coeff5);
            p31 = _mm_add_epi16(H3, coeff8);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            M2 = _mm_srli_epi16(p01, 4);


            p00 = _mm_add_epi16(L1, L2);
            p00 = _mm_mullo_epi16(p00, coeff3);
            p10 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(p10, coeff4);
            p00 = _mm_add_epi16(p10, p00);
            M3 = _mm_srli_epi16(p00, 3);

            p01 = _mm_add_epi16(H1, H2);
            p01 = _mm_mullo_epi16(p01, coeff3);
            p11 = _mm_add_epi16(H0, H3);
            p11 = _mm_add_epi16(p11, coeff4);
            p01 = _mm_add_epi16(p11, p01);
            M4 = _mm_srli_epi16(p01, 3);


            p10 = _mm_mullo_epi16(L1, coeff5);
            p20 = _mm_mullo_epi16(L2, coeff7);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(L0, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            M5 = _mm_srli_epi16(p00, 4);

            p11 = _mm_mullo_epi16(H1, coeff5);
            p21 = _mm_mullo_epi16(H2, coeff7);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(H0, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            M6 = _mm_srli_epi16(p01, 4);


            p00 = _mm_add_epi16(L1, L2);
            p10 = _mm_add_epi16(L2, L3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff2);
            M7 = _mm_srli_epi16(p00, 2);

            p01 = _mm_add_epi16(H1, H2);
            p11 = _mm_add_epi16(H2, H3);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, coeff2);
            M8 = _mm_srli_epi16(p01, 2);

            M1 = _mm_packus_epi16(M1, M3);
            M5 = _mm_packus_epi16(M5, M7);
            M1 = _mm_shuffle_epi8(M1, shuffle);
            M5 = _mm_shuffle_epi8(M5, shuffle);

            M2 = _mm_packus_epi16(M2, M4);
            M6 = _mm_packus_epi16(M6, M8);
            M2 = _mm_shuffle_epi8(M2, shuffle);
            M6 = _mm_shuffle_epi8(M6, shuffle);

            M3 = _mm_unpacklo_epi16(M1, M5);
            M7 = _mm_unpackhi_epi16(M1, M5);
            M4 = _mm_unpacklo_epi16(M2, M6);
            M8 = _mm_unpackhi_epi16(M2, M6);

            _mm_store_si128((__m128i*)&first_line[i], M4);
            _mm_store_si128((__m128i*)&first_line[16 + i], M8);
            _mm_store_si128((__m128i*)&first_line[32 + i], M3);
            _mm_store_si128((__m128i*)&first_line[48 + i], M7);
        }

        if (i < line_size) {
            __m128i p01, p11, p21, p31;
            __m128i M2, M4, M6, M8;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src - 3));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src - 2));

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);
            __m128i H3 = _mm_unpackhi_epi8(S3, zero);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff7);
            p21 = _mm_mullo_epi16(H2, coeff5);
            p31 = _mm_add_epi16(H3, coeff8);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            M2 = _mm_srli_epi16(p01, 4);

            p01 = _mm_add_epi16(H1, H2);
            p01 = _mm_mullo_epi16(p01, coeff3);
            p11 = _mm_add_epi16(H0, H3);
            p11 = _mm_add_epi16(p11, coeff4);
            p01 = _mm_add_epi16(p11, p01);
            M4 = _mm_srli_epi16(p01, 3);

            p11 = _mm_mullo_epi16(H1, coeff5);
            p21 = _mm_mullo_epi16(H2, coeff7);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(H0, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            M6 = _mm_srli_epi16(p01, 4);

            p01 = _mm_add_epi16(H1, H2);
            p11 = _mm_add_epi16(H2, H3);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, coeff2);
            M8 = _mm_srli_epi16(p01, 2);

            M2 = _mm_packus_epi16(M2, M4);
            M6 = _mm_packus_epi16(M6, M8);
            M2 = _mm_shuffle_epi8(M2, shuffle);
            M6 = _mm_shuffle_epi8(M6, shuffle);

            M4 = _mm_unpacklo_epi16(M2, M6);
            M8 = _mm_unpackhi_epi16(M2, M6);

            _mm_store_si128((__m128i*)&first_line[i], M4);
            _mm_store_si128((__m128i*)&first_line[16 + i], M8);
        }

        switch (bsx) {
        case 4:
            for (i = 0; i < iHeight4; i += 4) {
                CP32(dst, first_line + i);
                dst += i_dst;
            }
            break;
        case 8:
            for (i = 0; i < iHeight4; i += 4) {
                CP64(dst, first_line + i);
                dst += i_dst;
            }
            break;
        default:
            for (i = 0; i < iHeight4; i += 4) {
                memcpy(dst, first_line + i, bsx * sizeof(pel_t));
                dst += i_dst;
            }
            break;
        }
    } else {
        __m128i zero = _mm_setzero_si128();
        __m128i coeff2 = _mm_set1_epi16(2);
        __m128i coeff3 = _mm_set1_epi16(3);
        __m128i coeff4 = _mm_set1_epi16(4);
        __m128i coeff5 = _mm_set1_epi16(5);
        __m128i coeff7 = _mm_set1_epi16(7);
        __m128i coeff8 = _mm_set1_epi16(8);
        __m128i shuffle = _mm_setr_epi8(7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 0, 8);
        src -= 15;

        if (bsy == 4) {
            __m128i p01, p11, p21, p31;
            __m128i M2, M4, M6, M8;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src - 3));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src - 2));

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);
            __m128i H3 = _mm_unpackhi_epi8(S3, zero);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff7);
            p21 = _mm_mullo_epi16(H2, coeff5);
            p31 = _mm_add_epi16(H3, coeff8);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            M2 = _mm_srli_epi16(p01, 4);

            p01 = _mm_add_epi16(H1, H2);
            p01 = _mm_mullo_epi16(p01, coeff3);
            p11 = _mm_add_epi16(H0, H3);
            p11 = _mm_add_epi16(p11, coeff4);
            p01 = _mm_add_epi16(p11, p01);
            M4 = _mm_srli_epi16(p01, 3);

            p11 = _mm_mullo_epi16(H1, coeff5);
            p21 = _mm_mullo_epi16(H2, coeff7);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(H0, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            M6 = _mm_srli_epi16(p01, 4);

            p01 = _mm_add_epi16(H1, H2);
            p11 = _mm_add_epi16(H2, H3);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, coeff2);
            M8 = _mm_srli_epi16(p01, 2);

            M2 = _mm_packus_epi16(M2, M4);
            M6 = _mm_packus_epi16(M6, M8);
            M2 = _mm_shuffle_epi8(M2, shuffle);
            M6 = _mm_shuffle_epi8(M6, shuffle);

            M4 = _mm_unpacklo_epi16(M2, M6);

            ((int*)dst)[0] = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            ((int*)dst)[0] = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            ((int*)dst)[0] = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            ((int*)dst)[0] = _mm_cvtsi128_si32(M4);
        } else {
            __m128i p00, p10, p20, p30;
            __m128i p01, p11, p21, p31;
            __m128i M1, M2, M3, M4, M5, M6, M7, M8;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src - 3));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src - 2));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);
            __m128i H3 = _mm_unpackhi_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            M1 = _mm_srli_epi16(p00, 4);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff7);
            p21 = _mm_mullo_epi16(H2, coeff5);
            p31 = _mm_add_epi16(H3, coeff8);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            M2 = _mm_srli_epi16(p01, 4);


            p00 = _mm_add_epi16(L1, L2);
            p00 = _mm_mullo_epi16(p00, coeff3);
            p10 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(p10, coeff4);
            p00 = _mm_add_epi16(p10, p00);
            M3 = _mm_srli_epi16(p00, 3);

            p01 = _mm_add_epi16(H1, H2);
            p01 = _mm_mullo_epi16(p01, coeff3);
            p11 = _mm_add_epi16(H0, H3);
            p11 = _mm_add_epi16(p11, coeff4);
            p01 = _mm_add_epi16(p11, p01);
            M4 = _mm_srli_epi16(p01, 3);


            p10 = _mm_mullo_epi16(L1, coeff5);
            p20 = _mm_mullo_epi16(L2, coeff7);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(L0, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            M5 = _mm_srli_epi16(p00, 4);

            p11 = _mm_mullo_epi16(H1, coeff5);
            p21 = _mm_mullo_epi16(H2, coeff7);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(H0, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            M6 = _mm_srli_epi16(p01, 4);


            p00 = _mm_add_epi16(L1, L2);
            p10 = _mm_add_epi16(L2, L3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff2);
            M7 = _mm_srli_epi16(p00, 2);

            p01 = _mm_add_epi16(H1, H2);
            p11 = _mm_add_epi16(H2, H3);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, coeff2);
            M8 = _mm_srli_epi16(p01, 2);

            M1 = _mm_packus_epi16(M1, M3);
            M5 = _mm_packus_epi16(M5, M7);
            M1 = _mm_shuffle_epi8(M1, shuffle);
            M5 = _mm_shuffle_epi8(M5, shuffle);

            M2 = _mm_packus_epi16(M2, M4);
            M6 = _mm_packus_epi16(M6, M8);
            M2 = _mm_shuffle_epi8(M2, shuffle);
            M6 = _mm_shuffle_epi8(M6, shuffle);

            M3 = _mm_unpacklo_epi16(M1, M5);
            M7 = _mm_unpackhi_epi16(M1, M5);
            M4 = _mm_unpacklo_epi16(M2, M6);
            M8 = _mm_unpackhi_epi16(M2, M6);

            ((int*)dst)[0] = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            ((int*)dst)[0] = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            ((int*)dst)[0] = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            ((int*)dst)[0] = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            ((int*)dst)[0] = _mm_cvtsi128_si32(M8);
            dst += i_dst;
            M8 = _mm_srli_si128(M8, 4);
            ((int*)dst)[0] = _mm_cvtsi128_si32(M8);
            dst += i_dst;
            M8 = _mm_srli_si128(M8, 4);
            ((int*)dst)[0] = _mm_cvtsi128_si32(M8);
            dst += i_dst;
            M8 = _mm_srli_si128(M8, 4);
            ((int*)dst)[0] = _mm_cvtsi128_si32(M8);
            dst += i_dst;
            ((int*)dst)[0] = _mm_cvtsi128_si32(M3);
            dst += i_dst;
            M3 = _mm_srli_si128(M3, 4);
            ((int*)dst)[0] = _mm_cvtsi128_si32(M3);
            dst += i_dst;
            M3 = _mm_srli_si128(M3, 4);
            ((int*)dst)[0] = _mm_cvtsi128_si32(M3);
            dst += i_dst;
            M3 = _mm_srli_si128(M3, 4);
            ((int*)dst)[0] = _mm_cvtsi128_si32(M3);
            dst += i_dst;
            ((int*)dst)[0] = _mm_cvtsi128_si32(M7);
            dst += i_dst;
            M7 = _mm_srli_si128(M7, 4);
            ((int*)dst)[0] = _mm_cvtsi128_si32(M7);
            dst += i_dst;
            M7 = _mm_srli_si128(M7, 4);
            ((int*)dst)[0] = _mm_cvtsi128_si32(M7);
            dst += i_dst;
            M7 = _mm_srli_si128(M7, 4);
            ((int*)dst)[0] = _mm_cvtsi128_si32(M7);
        }
    }

}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_y_28_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[64 + 128]);
    int line_size = bsx + (bsy - 1) * 2;
    int i;
    int iHeight2 = bsy << 1;
    __m128i coeff2 = _mm_set1_epi16(2);
    __m128i coeff3 = _mm_set1_epi16(3);
    __m128i coeff4 = _mm_set1_epi16(4);
    __m128i shuffle = _mm_setr_epi8(7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 0, 8);
    __m128i zero = _mm_setzero_si128();

    UNUSED_PARAMETER(dir_mode);

    src -= 15;
    for (i = 0; i < line_size - 16; i += 32, src -= 16) {
        __m128i p00, p10, p01, p11;
        __m128i S0 = _mm_loadu_si128((__m128i*)(src));
        __m128i S3 = _mm_loadu_si128((__m128i*)(src - 3));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src - 2));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);
        __m128i L3 = _mm_unpacklo_epi8(S3, zero);

        __m128i H0 = _mm_unpackhi_epi8(S0, zero);
        __m128i H1 = _mm_unpackhi_epi8(S1, zero);
        __m128i H2 = _mm_unpackhi_epi8(S2, zero);
        __m128i H3 = _mm_unpackhi_epi8(S3, zero);

        p00 = _mm_adds_epi16(L1, L2);
        p01 = _mm_add_epi16(L1, L2);
        p00 = _mm_mullo_epi16(p00, coeff3);
        p10 = _mm_adds_epi16(L0, L3);
        p11 = _mm_add_epi16(L2, L3);
        p10 = _mm_adds_epi16(p10, coeff4);
        p00 = _mm_adds_epi16(p00, p10);
        p01 = _mm_add_epi16(p01, p11);
        p01 = _mm_add_epi16(p01, coeff2);

        p00 = _mm_srli_epi16(p00, 3);
        p01 = _mm_srli_epi16(p01, 2);

        p00 = _mm_packus_epi16(p00, p01);
        p00 = _mm_shuffle_epi8(p00, shuffle);

        _mm_store_si128((__m128i*)&first_line[i + 16], p00);

        p00 = _mm_adds_epi16(H1, H2);
        p01 = _mm_add_epi16(H1, H2);
        p00 = _mm_mullo_epi16(p00, coeff3);
        p10 = _mm_adds_epi16(H0, H3);
        p11 = _mm_add_epi16(H2, H3);
        p10 = _mm_adds_epi16(p10, coeff4);
        p00 = _mm_adds_epi16(p00, p10);
        p01 = _mm_add_epi16(p01, p11);
        p01 = _mm_add_epi16(p01, coeff2);

        p00 = _mm_srli_epi16(p00, 3);
        p01 = _mm_srli_epi16(p01, 2);

        p00 = _mm_packus_epi16(p00, p01);
        p00 = _mm_shuffle_epi8(p00, shuffle);

        _mm_store_si128((__m128i*)&first_line[i], p00);
    }

    if (i < line_size) {
        __m128i p00, p10, p01, p11;
        __m128i S0 = _mm_loadu_si128((__m128i*)(src));
        __m128i S3 = _mm_loadu_si128((__m128i*)(src - 3));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src - 2));

        __m128i H0 = _mm_unpackhi_epi8(S0, zero);
        __m128i H1 = _mm_unpackhi_epi8(S1, zero);
        __m128i H2 = _mm_unpackhi_epi8(S2, zero);
        __m128i H3 = _mm_unpackhi_epi8(S3, zero);

        p00 = _mm_adds_epi16(H1, H2);
        p01 = _mm_add_epi16(H1, H2);
        p00 = _mm_mullo_epi16(p00, coeff3);
        p10 = _mm_adds_epi16(H0, H3);
        p11 = _mm_add_epi16(H2, H3);
        p10 = _mm_adds_epi16(p10, coeff4);
        p00 = _mm_adds_epi16(p00, p10);
        p01 = _mm_add_epi16(p01, p11);
        p01 = _mm_add_epi16(p01, coeff2);

        p00 = _mm_srli_epi16(p00, 3);
        p01 = _mm_srli_epi16(p01, 2);

        p00 = _mm_packus_epi16(p00, p01);
        p00 = _mm_shuffle_epi8(p00, shuffle);

        _mm_store_si128((__m128i*)&first_line[i], p00);
    }

    if (bsx >= 16) {
        for (i = 0; i < iHeight2; i += 2) {
            memcpy(dst, first_line + i, bsx * sizeof(pel_t));
            dst += i_dst;
        }
    } else if (bsx == 8) {
        for (i = 0; i < iHeight2; i += 8) {
            __m128i M = _mm_loadu_si128((__m128i*)&first_line[i]);
            _mm_storel_epi64((__m128i*)(dst), M);
            dst += i_dst;
            M = _mm_srli_si128(M, 2);
            _mm_storel_epi64((__m128i*)(dst), M);
            dst += i_dst;
            M = _mm_srli_si128(M, 2);
            _mm_storel_epi64((__m128i*)(dst), M);
            dst += i_dst;
            M = _mm_srli_si128(M, 2);
            _mm_storel_epi64((__m128i*)(dst), M);
            dst += i_dst;
        }
    } else {
        for (i = 0; i < iHeight2; i += 8) {
            __m128i M = _mm_loadu_si128((__m128i*)&first_line[i]);
            ((int*)(dst))[0] = _mm_cvtsi128_si32(M);
            dst += i_dst;
            M = _mm_srli_si128(M, 2);
            ((int*)(dst))[0] = _mm_cvtsi128_si32(M);
            dst += i_dst;
            M = _mm_srli_si128(M, 2);
            ((int*)(dst))[0] = _mm_cvtsi128_si32(M);
            dst += i_dst;
            M = _mm_srli_si128(M, 2);
            ((int*)(dst))[0] = _mm_cvtsi128_si32(M);
            dst += i_dst;
        }
    }
}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_y_30_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[64 + 64]);
    int line_size = bsx + bsy - 1;
    int i;
    __m128i coeff2 = _mm_set1_epi16(2);
    __m128i shuffle = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m128i zero = _mm_setzero_si128();

    UNUSED_PARAMETER(dir_mode);

    src -= 17;
    for (i = 0; i < line_size - 8; i += 16, src -= 16) {
        __m128i p00, p10, p01, p11;
        __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);

        __m128i H0 = _mm_unpackhi_epi8(S0, zero);
        __m128i H1 = _mm_unpackhi_epi8(S1, zero);
        __m128i H2 = _mm_unpackhi_epi8(S2, zero);

        p00 = _mm_add_epi16(L0, L1);
        p10 = _mm_add_epi16(L1, L2);
        p01 = _mm_add_epi16(H0, H1);
        p11 = _mm_add_epi16(H1, H2);

        p00 = _mm_add_epi16(p00, p10);
        p01 = _mm_add_epi16(p01, p11);
        p00 = _mm_add_epi16(p00, coeff2);
        p01 = _mm_add_epi16(p01, coeff2);

        p00 = _mm_srli_epi16(p00, 2);
        p01 = _mm_srli_epi16(p01, 2);

        p00 = _mm_packus_epi16(p00, p01);
        p00 = _mm_shuffle_epi8(p00, shuffle);

        _mm_store_si128((__m128i*)&first_line[i], p00);
    }

    if (i < line_size) {
        __m128i p01, p11;
        __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src));

        __m128i H0 = _mm_unpackhi_epi8(S0, zero);
        __m128i H1 = _mm_unpackhi_epi8(S1, zero);
        __m128i H2 = _mm_unpackhi_epi8(S2, zero);

        p01 = _mm_add_epi16(H0, H1);
        p11 = _mm_add_epi16(H1, H2);

        p01 = _mm_add_epi16(p01, p11);
        p01 = _mm_add_epi16(p01, coeff2);

        p01 = _mm_srli_epi16(p01, 2);

        p01 = _mm_packus_epi16(p01, p01);
        p01 = _mm_shuffle_epi8(p01, shuffle);

        _mm_store_si128((__m128i*)&first_line[i], p01);
    }

    if (bsx > 16) {
        for (i = 0; i < bsy; i++) {
            memcpy(dst, first_line + i, bsx * sizeof(pel_t));
            dst += i_dst;
        }
    } else if (bsx == 16) {
        pel_t *dst1 = dst;
        pel_t *dst2;
        if (bsy == 4) {
            __m128i M = _mm_loadu_si128((__m128i*)&first_line[0]);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 = dst + 8;
            M = _mm_loadu_si128((__m128i*)&first_line[8]);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
        } else {
            __m128i M = _mm_loadu_si128((__m128i*)&first_line[0]);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst2 = dst1 + i_dst;
            dst1 = dst + 8;
            M = _mm_loadu_si128((__m128i*)&first_line[8]);
            _mm_storel_epi64((__m128i*)dst1, M);
            _mm_storel_epi64((__m128i*)dst2, M);
            dst1 += i_dst;
            dst2 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            _mm_storel_epi64((__m128i*)dst2, M);
            dst1 += i_dst;
            dst2 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            _mm_storel_epi64((__m128i*)dst2, M);
            dst1 += i_dst;
            dst2 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            _mm_storel_epi64((__m128i*)dst2, M);
            dst1 += i_dst;
            dst2 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            _mm_storel_epi64((__m128i*)dst2, M);
            dst1 += i_dst;
            dst2 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            _mm_storel_epi64((__m128i*)dst2, M);
            dst1 += i_dst;
            dst2 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            _mm_storel_epi64((__m128i*)dst2, M);
            dst1 += i_dst;
            dst2 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            _mm_storel_epi64((__m128i*)dst2, M);
            dst1 += i_dst;
            M = _mm_loadu_si128((__m128i*)&first_line[16]);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
            dst1 += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst1, M);
        }
    } else if (bsx == 8) {
        for (i = 0; i < bsy; i += 8) {
            __m128i M = _mm_loadu_si128((__m128i*)&first_line[i]);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
            M = _mm_srli_si128(M, 1);
            _mm_storel_epi64((__m128i*)dst, M);
            dst += i_dst;
        }
    } else {
        for (i = 0; i < bsy; i += 4) {
            __m128i M = _mm_loadu_si128((__m128i*)&first_line[i]);
            ((int*)(dst))[0] = _mm_cvtsi128_si32(M);
            dst += i_dst;
            M = _mm_srli_si128(M, 1);
            ((int*)(dst))[0] = _mm_cvtsi128_si32(M);
            dst += i_dst;
            M = _mm_srli_si128(M, 1);
            ((int*)(dst))[0] = _mm_cvtsi128_si32(M);
            dst += i_dst;
            M = _mm_srli_si128(M, 1);
            ((int*)(dst))[0] = _mm_cvtsi128_si32(M);
            dst += i_dst;
        }
    }

}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_y_31_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t dst_tran[64 * 80]);
    ALIGN16(pel_t src_tran[64 * 8]);
    int i_dst2 = (((bsy + 15) >> 4) << 4) + 16;
    int i;

    UNUSED_PARAMETER(dir_mode);

    //transposition
    for (i = 0; i < (bsy + bsx * 11 / 8 + 3); i++) {
        src_tran[i] = src[-i];
    }

    intra_pred_ang_x_5_sse128(src_tran, dst_tran, i_dst2, 5, bsy, bsx);

    if ((bsy > 4) && (bsx > 4)) {
        pel_t *pDst_128[64];
        pel_t *pTra_128[64];

        int iSize_x = bsx >> 3;
        int iSize_y = bsy >> 3;
        int iSize = iSize_x * iSize_y;

        for (int y = 0; y < iSize_y; y++) {
            for (int x = 0; x < iSize_x; x++) {
                pDst_128[x + y * iSize_x] = dst      + x * 8 + y * 8 * i_dst;
                pTra_128[x + y * iSize_x] = dst_tran + y * 8 + x * 8 * i_dst2;
            }
        }

        for (i = 0; i < iSize; i++) {
            pel_t *dst_tran_org = pTra_128[i];

            pel_t *dst1 = pDst_128[i];
            pel_t *dst2 = dst1 + i_dst;
            pel_t *dst3 = dst2 + i_dst;
            pel_t *dst4 = dst3 + i_dst;
            pel_t *dst5 = dst4 + i_dst;
            pel_t *dst6 = dst5 + i_dst;
            pel_t *dst7 = dst6 + i_dst;
            pel_t *dst8 = dst7 + i_dst;
            __m128i Org_8_0, Org_8_1, Org_8_2, Org_8_3, Org_8_4, Org_8_5, Org_8_6, Org_8_7;
            __m128i p00, p10, p20, p30;
            __m128i t00, t10, t20, t30;
            Org_8_0 = _mm_loadu_si128((__m128i*)dst_tran_org);
            dst_tran_org += i_dst2;
            Org_8_1 = _mm_loadu_si128((__m128i*)dst_tran_org);
            dst_tran_org += i_dst2;
            Org_8_2 = _mm_loadu_si128((__m128i*)dst_tran_org);
            dst_tran_org += i_dst2;
            Org_8_3 = _mm_loadu_si128((__m128i*)dst_tran_org);
            dst_tran_org += i_dst2;
            Org_8_4 = _mm_loadu_si128((__m128i*)dst_tran_org);
            dst_tran_org += i_dst2;
            Org_8_5 = _mm_loadu_si128((__m128i*)dst_tran_org);
            dst_tran_org += i_dst2;
            Org_8_6 = _mm_loadu_si128((__m128i*)dst_tran_org);
            dst_tran_org += i_dst2;
            Org_8_7 = _mm_loadu_si128((__m128i*)dst_tran_org);
            dst_tran_org += i_dst2;

            p00 = _mm_unpacklo_epi8(Org_8_0, Org_8_1);
            p10 = _mm_unpacklo_epi8(Org_8_2, Org_8_3);
            p20 = _mm_unpacklo_epi8(Org_8_4, Org_8_5);
            p30 = _mm_unpacklo_epi8(Org_8_6, Org_8_7);

            t00 = _mm_unpacklo_epi16(p00, p10);
            t20 = _mm_unpacklo_epi16(p20, p30);
            t10 = _mm_unpackhi_epi16(p00, p10);
            t30 = _mm_unpackhi_epi16(p20, p30);

            p00 = _mm_unpacklo_epi32(t00, t20);
            p10 = _mm_unpackhi_epi32(t00, t20);
            p20 = _mm_unpacklo_epi32(t10, t30);
            p30 = _mm_unpackhi_epi32(t10, t30);

            _mm_storel_epi64((__m128i*)dst1, p00);
            p00 = _mm_srli_si128(p00, 8);
            _mm_storel_epi64((__m128i*)dst2, p00);

            _mm_storel_epi64((__m128i*)dst3, p10);
            p10 = _mm_srli_si128(p10, 8);
            _mm_storel_epi64((__m128i*)dst4, p10);

            _mm_storel_epi64((__m128i*)dst5, p20);
            p20 = _mm_srli_si128(p20, 8);
            _mm_storel_epi64((__m128i*)dst6, p20);

            _mm_storel_epi64((__m128i*)dst7, p30);
            p30 = _mm_srli_si128(p30, 8);
            _mm_storel_epi64((__m128i*)dst8, p30);
        }
    } else if (bsx == 16) {
        for (i = 0; i < 2; i++) {
            pel_t *dst_tran_org = dst_tran + i * 8 * i_dst2;

            pel_t *dst1 = dst + i * 8;
            pel_t *dst2 = dst1 + i_dst;
            pel_t *dst3 = dst2 + i_dst;
            pel_t *dst4 = dst3 + i_dst;

            __m128i Org_8_0, Org_8_1, Org_8_2, Org_8_3, Org_8_4, Org_8_5, Org_8_6, Org_8_7;
            __m128i p00, p10, p20, p30;
            __m128i t00, t20;
            Org_8_0 = _mm_loadu_si128((__m128i*)dst_tran_org);
            dst_tran_org += i_dst2;
            Org_8_1 = _mm_loadu_si128((__m128i*)dst_tran_org);
            dst_tran_org += i_dst2;
            Org_8_2 = _mm_loadu_si128((__m128i*)dst_tran_org);
            dst_tran_org += i_dst2;
            Org_8_3 = _mm_loadu_si128((__m128i*)dst_tran_org);
            dst_tran_org += i_dst2;
            Org_8_4 = _mm_loadu_si128((__m128i*)dst_tran_org);
            dst_tran_org += i_dst2;
            Org_8_5 = _mm_loadu_si128((__m128i*)dst_tran_org);
            dst_tran_org += i_dst2;
            Org_8_6 = _mm_loadu_si128((__m128i*)dst_tran_org);
            dst_tran_org += i_dst2;
            Org_8_7 = _mm_loadu_si128((__m128i*)dst_tran_org);
            dst_tran_org += i_dst2;

            p00 = _mm_unpacklo_epi8(Org_8_0, Org_8_1);
            p10 = _mm_unpacklo_epi8(Org_8_2, Org_8_3);
            p20 = _mm_unpacklo_epi8(Org_8_4, Org_8_5);
            p30 = _mm_unpacklo_epi8(Org_8_6, Org_8_7);

            t00 = _mm_unpacklo_epi16(p00, p10);
            t20 = _mm_unpacklo_epi16(p20, p30);

            p00 = _mm_unpacklo_epi32(t00, t20);
            p10 = _mm_unpackhi_epi32(t00, t20);

            _mm_storel_epi64((__m128i*)dst1, p00);
            p00 = _mm_srli_si128(p00, 8);
            _mm_storel_epi64((__m128i*)dst2, p00);

            _mm_storel_epi64((__m128i*)dst3, p10);
            p10 = _mm_srli_si128(p10, 8);
            _mm_storel_epi64((__m128i*)dst4, p10);
        }
    } else if (bsy == 16) {//bsx == 4
        pel_t *dst_tran_org = dst_tran;

        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        pel_t *dst5 = dst4 + i_dst;
        pel_t *dst6 = dst5 + i_dst;
        pel_t *dst7 = dst6 + i_dst;
        pel_t *dst8 = dst7 + i_dst;

        __m128i Org_8_0, Org_8_1, Org_8_2, Org_8_3;
        __m128i p00, p10;
        __m128i t00, t10;
        Org_8_0 = _mm_loadu_si128((__m128i*)dst_tran_org);
        dst_tran_org += i_dst2;
        Org_8_1 = _mm_loadu_si128((__m128i*)dst_tran_org);
        dst_tran_org += i_dst2;
        Org_8_2 = _mm_loadu_si128((__m128i*)dst_tran_org);
        dst_tran_org += i_dst2;
        Org_8_3 = _mm_loadu_si128((__m128i*)dst_tran_org);
        dst_tran_org += i_dst2;

        p00 = _mm_unpacklo_epi8(Org_8_0, Org_8_1);
        p10 = _mm_unpacklo_epi8(Org_8_2, Org_8_3);

        t00 = _mm_unpacklo_epi16(p00, p10);
        t10 = _mm_unpackhi_epi16(p00, p10);

        *((int*)(dst1)) = _mm_cvtsi128_si32(t00);
        t00 = _mm_srli_si128(t00, 4);
        *((int*)(dst2)) = _mm_cvtsi128_si32(t00);
        t00 = _mm_srli_si128(t00, 4);
        *((int*)(dst3)) = _mm_cvtsi128_si32(t00);
        t00 = _mm_srli_si128(t00, 4);
        *((int*)(dst4)) = _mm_cvtsi128_si32(t00);

        *((int*)(dst5)) = _mm_cvtsi128_si32(t10);
        t10 = _mm_srli_si128(t10, 4);
        *((int*)(dst6)) = _mm_cvtsi128_si32(t10);
        t10 = _mm_srli_si128(t10, 4);
        *((int*)(dst7)) = _mm_cvtsi128_si32(t10);
        t10 = _mm_srli_si128(t10, 4);
        *((int*)(dst8)) = _mm_cvtsi128_si32(t10);

        dst1 = dst8 + i_dst;
        dst2 = dst1 + i_dst;
        dst3 = dst2 + i_dst;
        dst4 = dst3 + i_dst;
        dst5 = dst4 + i_dst;
        dst6 = dst5 + i_dst;
        dst7 = dst6 + i_dst;
        dst8 = dst7 + i_dst;

        p00 = _mm_unpackhi_epi8(Org_8_0, Org_8_1);
        p10 = _mm_unpackhi_epi8(Org_8_2, Org_8_3);

        t00 = _mm_unpacklo_epi16(p00, p10);
        t10 = _mm_unpackhi_epi16(p00, p10);

        *((int*)(dst1)) = _mm_cvtsi128_si32(t00);
        t00 = _mm_srli_si128(t00, 4);
        *((int*)(dst2)) = _mm_cvtsi128_si32(t00);
        t00 = _mm_srli_si128(t00, 4);
        *((int*)(dst3)) = _mm_cvtsi128_si32(t00);
        t00 = _mm_srli_si128(t00, 4);
        *((int*)(dst4)) = _mm_cvtsi128_si32(t00);

        *((int*)(dst5)) = _mm_cvtsi128_si32(t10);
        t10 = _mm_srli_si128(t10, 4);
        *((int*)(dst6)) = _mm_cvtsi128_si32(t10);
        t10 = _mm_srli_si128(t10, 4);
        *((int*)(dst7)) = _mm_cvtsi128_si32(t10);
        t10 = _mm_srli_si128(t10, 4);
        *((int*)(dst8)) = _mm_cvtsi128_si32(t10);
    } else {// bsx == 4 bsy ==4
        pel_t *dst_tran_org = dst_tran;

        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;

        __m128i Org_8_0, Org_8_1, Org_8_2, Org_8_3;
        __m128i p00, p10;
        __m128i t00;
        Org_8_0 = _mm_loadu_si128((__m128i*)dst_tran_org);
        dst_tran_org += i_dst2;
        Org_8_1 = _mm_loadu_si128((__m128i*)dst_tran_org);
        dst_tran_org += i_dst2;
        Org_8_2 = _mm_loadu_si128((__m128i*)dst_tran_org);
        dst_tran_org += i_dst2;
        Org_8_3 = _mm_loadu_si128((__m128i*)dst_tran_org);
        dst_tran_org += i_dst2;

        p00 = _mm_unpacklo_epi8(Org_8_0, Org_8_1);
        p10 = _mm_unpacklo_epi8(Org_8_2, Org_8_3);

        t00 = _mm_unpacklo_epi16(p00, p10);

        *((int*)(dst1)) = _mm_cvtsi128_si32(t00);
        t00 = _mm_srli_si128(t00, 4);
        *((int*)(dst2)) = _mm_cvtsi128_si32(t00);
        t00 = _mm_srli_si128(t00, 4);
        *((int*)(dst3)) = _mm_cvtsi128_si32(t00);
        t00 = _mm_srli_si128(t00, 4);
        *((int*)(dst4)) = _mm_cvtsi128_si32(t00);
    }

}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_y_32_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[2 * (64 + 64)]);
    int line_size = (bsy >> 1) + bsx - 1;
    int i;
    int aligned_line_size = ((line_size + 63) >> 4) << 4;
    pel_t *pfirst[2];

    __m128i coeff2 = _mm_set1_epi16(2);
    __m128i zero = _mm_setzero_si128();
    __m128i shuffle1 = _mm_setr_epi8(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
    __m128i shuffle2 = _mm_setr_epi8(14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1);
    int i_dst2 = i_dst * 2;

    UNUSED_PARAMETER(dir_mode);

    pfirst[0] = first_line;
    pfirst[1] = first_line + aligned_line_size;

    src -= 18;
    for (i = 0; i < line_size - 4; i += 8, src -= 16) {
        __m128i p00, p01, p10, p11;
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);

        __m128i H0 = _mm_unpackhi_epi8(S0, zero);
        __m128i H1 = _mm_unpackhi_epi8(S1, zero);
        __m128i H2 = _mm_unpackhi_epi8(S2, zero);

        p00 = _mm_add_epi16(L0, L1);
        p01 = _mm_add_epi16(L1, L2);
        p00 = _mm_add_epi16(p00, coeff2);
        p00 = _mm_add_epi16(p00, p01);
        p00 = _mm_srli_epi16(p00, 2);

        p10 = _mm_add_epi16(H0, H1);
        p11 = _mm_add_epi16(H1, H2);
        p10 = _mm_add_epi16(p10, coeff2);
        p10 = _mm_add_epi16(p10, p11);
        p10 = _mm_srli_epi16(p10, 2);

        p00 = _mm_packus_epi16(p00, p10);
        p10 = _mm_shuffle_epi8(p00, shuffle2);
        p00 = _mm_shuffle_epi8(p00, shuffle1);
        _mm_storel_epi64((__m128i*)&pfirst[0][i], p00);
        _mm_storel_epi64((__m128i*)&pfirst[1][i], p10);
    }

    if (i < line_size) {
        __m128i p10, p11;
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src));

        __m128i H0 = _mm_unpackhi_epi8(S0, zero);
        __m128i H1 = _mm_unpackhi_epi8(S1, zero);
        __m128i H2 = _mm_unpackhi_epi8(S2, zero);

        p10 = _mm_add_epi16(H0, H1);
        p11 = _mm_add_epi16(H1, H2);
        p10 = _mm_add_epi16(p10, coeff2);
        p10 = _mm_add_epi16(p10, p11);
        p10 = _mm_srli_epi16(p10, 2);

        p11 = _mm_packus_epi16(p10, p10);
        p10 = _mm_shuffle_epi8(p11, shuffle2);
        p11 = _mm_shuffle_epi8(p11, shuffle1);
        ((int*)&pfirst[0][i])[0] = _mm_cvtsi128_si32(p11);
        ((int*)&pfirst[1][i])[0] = _mm_cvtsi128_si32(p10);
    }

    bsy >>= 1;

    if (bsx >= 16 || bsx == 4) {
        for (i = 0; i < bsy; i++) {
            memcpy(dst, pfirst[0] + i, bsx * sizeof(pel_t));
            memcpy(dst + i_dst, pfirst[1] + i, bsx * sizeof(pel_t));
            dst += i_dst2;
        }
    } else {
        if (bsy == 4) {
            __m128i M1 = _mm_loadu_si128((__m128i*)&pfirst[0][0]);
            __m128i M2 = _mm_loadu_si128((__m128i*)&pfirst[1][0]);
            _mm_storel_epi64((__m128i*)dst, M1);
            _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
            dst += i_dst2;
            M1 = _mm_srli_si128(M1, 1);
            M2 = _mm_srli_si128(M2, 1);
            _mm_storel_epi64((__m128i*)dst, M1);
            _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
            dst += i_dst2;
            M1 = _mm_srli_si128(M1, 1);
            M2 = _mm_srli_si128(M2, 1);
            _mm_storel_epi64((__m128i*)dst, M1);
            _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
            dst += i_dst2;
            M1 = _mm_srli_si128(M1, 1);
            M2 = _mm_srli_si128(M2, 1);
            _mm_storel_epi64((__m128i*)dst, M1);
            _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
        } else {
            for (i = 0; i < 16; i = i + 8) {
                __m128i M1 = _mm_loadu_si128((__m128i*)&pfirst[0][i]);
                __m128i M2 = _mm_loadu_si128((__m128i*)&pfirst[1][i]);
                _mm_storel_epi64((__m128i*)dst, M1);
                _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
                dst += i_dst2;
                M1 = _mm_srli_si128(M1, 1);
                M2 = _mm_srli_si128(M2, 1);
                _mm_storel_epi64((__m128i*)dst, M1);
                _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
                dst += i_dst2;
                M1 = _mm_srli_si128(M1, 1);
                M2 = _mm_srli_si128(M2, 1);
                _mm_storel_epi64((__m128i*)dst, M1);
                _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
                dst += i_dst2;
                M1 = _mm_srli_si128(M1, 1);
                M2 = _mm_srli_si128(M2, 1);
                _mm_storel_epi64((__m128i*)dst, M1);
                _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
                dst += i_dst2;
                M1 = _mm_srli_si128(M1, 1);
                M2 = _mm_srli_si128(M2, 1);
                _mm_storel_epi64((__m128i*)dst, M1);
                _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
                dst += i_dst2;
                M1 = _mm_srli_si128(M1, 1);
                M2 = _mm_srli_si128(M2, 1);
                _mm_storel_epi64((__m128i*)dst, M1);
                _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
                dst += i_dst2;
                M1 = _mm_srli_si128(M1, 1);
                M2 = _mm_srli_si128(M2, 1);
                _mm_storel_epi64((__m128i*)dst, M1);
                _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
                dst += i_dst2;
                M1 = _mm_srli_si128(M1, 1);
                M2 = _mm_srli_si128(M2, 1);
                _mm_storel_epi64((__m128i*)dst, M1);
                _mm_storel_epi64((__m128i*)(dst + i_dst), M2);
                dst += i_dst2;
            }
        }
    }
}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_xy_13_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    __m128i zero = _mm_setzero_si128();
    __m128i coeff2 = _mm_set1_epi16(2);
    __m128i coeff3 = _mm_set1_epi16(3);
    __m128i coeff4 = _mm_set1_epi16(4);
    __m128i coeff5 = _mm_set1_epi16(5);
    __m128i coeff7 = _mm_set1_epi16(7);
    __m128i coeff8 = _mm_set1_epi16(8);
    __m128i coeff9 = _mm_set1_epi16(9);
    __m128i coeff11 = _mm_set1_epi16(11);
    __m128i coeff13 = _mm_set1_epi16(13);
    __m128i coeff15 = _mm_set1_epi16(15);
    __m128i coeff16 = _mm_set1_epi16(16);

    UNUSED_PARAMETER(dir_mode);

    int i;
    if (bsy > 8) {
        ALIGN16(pel_t first_line[(64 + 16) << 3]);
        int line_size = bsx + (bsy >> 3) - 1;
        int left_size = line_size - bsx;
        int aligned_line_size = ((line_size + 15) >> 4) << 4;
        pel_t *pfirst[8];

        pfirst[0] = first_line;
        pfirst[1] = pfirst[0] + aligned_line_size;
        pfirst[2] = pfirst[1] + aligned_line_size;
        pfirst[3] = pfirst[2] + aligned_line_size;
        pfirst[4] = pfirst[3] + aligned_line_size;
        pfirst[5] = pfirst[4] + aligned_line_size;
        pfirst[6] = pfirst[5] + aligned_line_size;
        pfirst[7] = pfirst[6] + aligned_line_size;

        src -= bsy - 8;
        for (i = 0; i < left_size; i++, src += 8) {
            pfirst[0][i] = (pel_t)((src[6] + (src[7] << 1) + src[8] + 2) >> 2);
            pfirst[1][i] = (pel_t)((src[5] + (src[6] << 1) + src[7] + 2) >> 2);
            pfirst[2][i] = (pel_t)((src[4] + (src[5] << 1) + src[6] + 2) >> 2);
            pfirst[3][i] = (pel_t)((src[3] + (src[4] << 1) + src[5] + 2) >> 2);

            pfirst[4][i] = (pel_t)((src[2] + (src[3] << 1) + src[4] + 2) >> 2);
            pfirst[5][i] = (pel_t)((src[1] + (src[2] << 1) + src[3] + 2) >> 2);
            pfirst[6][i] = (pel_t)((src[0] + (src[1] << 1) + src[2] + 2) >> 2);
            pfirst[7][i] = (pel_t)((src[-1] + (src[0] << 1) + src[1] + 2) >> 2);
        }

        for (; i < line_size - 8; i += 16, src += 16) {
            __m128i p00, p10, p20, p30;
            __m128i p01, p11, p21, p31;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src + 2));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src - 1));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);
            __m128i H3 = _mm_unpackhi_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff7);
            p10 = _mm_mullo_epi16(L1, coeff15);
            p20 = _mm_mullo_epi16(L2, coeff9);
            p30 = _mm_add_epi16(L3, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H0, coeff7);
            p11 = _mm_mullo_epi16(H1, coeff15);
            p21 = _mm_mullo_epi16(H2, coeff9);
            p31 = _mm_add_epi16(H3, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_storeu_si128((__m128i*)&pfirst[0][i], p00);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff7);
            p21 = _mm_mullo_epi16(H2, coeff5);
            p31 = _mm_add_epi16(H3, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_storeu_si128((__m128i*)&pfirst[1][i], p00);

            p00 = _mm_mullo_epi16(L0, coeff5);
            p10 = _mm_mullo_epi16(L1, coeff13);
            p20 = _mm_mullo_epi16(L2, coeff11);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H0, coeff5);
            p11 = _mm_mullo_epi16(H1, coeff13);
            p21 = _mm_mullo_epi16(H2, coeff11);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(p01, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_storeu_si128((__m128i*)&pfirst[2][i], p00);

            p00 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(L1, L2);
            p10 = _mm_mullo_epi16(p10, coeff3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_srli_epi16(p00, 3);

            p01 = _mm_add_epi16(H0, H3);
            p11 = _mm_add_epi16(H1, H2);
            p11 = _mm_mullo_epi16(p11, coeff3);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, coeff4);
            p01 = _mm_srli_epi16(p01, 3);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_storeu_si128((__m128i*)&pfirst[3][i], p00);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff11);
            p20 = _mm_mullo_epi16(L2, coeff13);
            p30 = _mm_mullo_epi16(L3, coeff5);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff11);
            p21 = _mm_mullo_epi16(H2, coeff13);
            p31 = _mm_mullo_epi16(H3, coeff5);
            p01 = _mm_add_epi16(p01, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_storeu_si128((__m128i*)&pfirst[4][i], p00);

            p10 = _mm_mullo_epi16(L1, coeff5);
            p20 = _mm_mullo_epi16(L2, coeff7);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(L0, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p11 = _mm_mullo_epi16(H1, coeff5);
            p21 = _mm_mullo_epi16(H2, coeff7);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(H0, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_storeu_si128((__m128i*)&pfirst[5][i], p00);

            p10 = _mm_mullo_epi16(L1, coeff9);
            p20 = _mm_mullo_epi16(L2, coeff15);
            p30 = _mm_mullo_epi16(L3, coeff7);
            p00 = _mm_add_epi16(L0, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p11 = _mm_mullo_epi16(H1, coeff9);
            p21 = _mm_mullo_epi16(H2, coeff15);
            p31 = _mm_mullo_epi16(H3, coeff7);
            p01 = _mm_add_epi16(H0, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_storeu_si128((__m128i*)&pfirst[6][i], p00);


            p10 = _mm_mullo_epi16(L2, coeff2);
            p00 = _mm_add_epi16(L1, L3);
            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_srli_epi16(p00, 2);

            p11 = _mm_mullo_epi16(H2, coeff2);
            p01 = _mm_add_epi16(H1, H3);
            p01 = _mm_add_epi16(p01, coeff2);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_srli_epi16(p01, 2);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_storeu_si128((__m128i*)&pfirst[7][i], p00);
        }

        if (i < line_size) {
            __m128i p00, p10, p20, p30;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src + 2));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src - 1));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff7);
            p10 = _mm_mullo_epi16(L1, coeff15);
            p20 = _mm_mullo_epi16(L2, coeff9);
            p30 = _mm_add_epi16(L3, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[0][i], p00);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[1][i], p00);

            p00 = _mm_mullo_epi16(L0, coeff5);
            p10 = _mm_mullo_epi16(L1, coeff13);
            p20 = _mm_mullo_epi16(L2, coeff11);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[2][i], p00);

            p00 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(L1, L2);
            p10 = _mm_mullo_epi16(p10, coeff3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_srli_epi16(p00, 3);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[3][i], p00);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff11);
            p20 = _mm_mullo_epi16(L2, coeff13);
            p30 = _mm_mullo_epi16(L3, coeff5);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[4][i], p00);

            p10 = _mm_mullo_epi16(L1, coeff5);
            p20 = _mm_mullo_epi16(L2, coeff7);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(L0, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[5][i], p00);

            p10 = _mm_mullo_epi16(L1, coeff9);
            p20 = _mm_mullo_epi16(L2, coeff15);
            p30 = _mm_mullo_epi16(L3, coeff7);
            p00 = _mm_add_epi16(L0, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[6][i], p00);

            p10 = _mm_mullo_epi16(L2, coeff2);
            p00 = _mm_add_epi16(L1, L3);
            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_srli_epi16(p00, 2);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)&pfirst[7][i], p00);
        }

        pfirst[0] += left_size;
        pfirst[1] += left_size;
        pfirst[2] += left_size;
        pfirst[3] += left_size;
        pfirst[4] += left_size;
        pfirst[5] += left_size;
        pfirst[6] += left_size;
        pfirst[7] += left_size;

        bsy >>= 3;
        for (i = 0; i < bsy; i++) {
            memcpy(dst, pfirst[0] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[1] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[2] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[3] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[4] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[5] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[6] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[7] - i, bsx * sizeof(pel_t));
            dst += i_dst;
        }
    } else if (bsy == 8) {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        pel_t *dst5 = dst4 + i_dst;
        pel_t *dst6 = dst5 + i_dst;
        pel_t *dst7 = dst6 + i_dst;
        pel_t *dst8 = dst7 + i_dst;
        if (bsx == 32) {
            __m128i p00, p10, p20, p30;
            __m128i p01, p11, p21, p31;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src + 2));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src - 1));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);
            __m128i H3 = _mm_unpackhi_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff7);
            p10 = _mm_mullo_epi16(L1, coeff15);
            p20 = _mm_mullo_epi16(L2, coeff9);
            p30 = _mm_add_epi16(L3, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H0, coeff7);
            p11 = _mm_mullo_epi16(H1, coeff15);
            p21 = _mm_mullo_epi16(H2, coeff9);
            p31 = _mm_add_epi16(H3, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst1, p00);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff7);
            p21 = _mm_mullo_epi16(H2, coeff5);
            p31 = _mm_add_epi16(H3, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst2, p00);

            p00 = _mm_mullo_epi16(L0, coeff5);
            p10 = _mm_mullo_epi16(L1, coeff13);
            p20 = _mm_mullo_epi16(L2, coeff11);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H0, coeff5);
            p11 = _mm_mullo_epi16(H1, coeff13);
            p21 = _mm_mullo_epi16(H2, coeff11);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(p01, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst3, p00);

            p00 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(L1, L2);
            p10 = _mm_mullo_epi16(p10, coeff3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_srli_epi16(p00, 3);

            p01 = _mm_add_epi16(H0, H3);
            p11 = _mm_add_epi16(H1, H2);
            p11 = _mm_mullo_epi16(p11, coeff3);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, coeff4);
            p01 = _mm_srli_epi16(p01, 3);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst4, p00);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff11);
            p20 = _mm_mullo_epi16(L2, coeff13);
            p30 = _mm_mullo_epi16(L3, coeff5);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff11);
            p21 = _mm_mullo_epi16(H2, coeff13);
            p31 = _mm_mullo_epi16(H3, coeff5);
            p01 = _mm_add_epi16(p01, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst5, p00);

            p10 = _mm_mullo_epi16(L1, coeff5);
            p20 = _mm_mullo_epi16(L2, coeff7);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(L0, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p11 = _mm_mullo_epi16(H1, coeff5);
            p21 = _mm_mullo_epi16(H2, coeff7);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(H0, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst6, p00);

            p10 = _mm_mullo_epi16(L1, coeff9);
            p20 = _mm_mullo_epi16(L2, coeff15);
            p30 = _mm_mullo_epi16(L3, coeff7);
            p00 = _mm_add_epi16(L0, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p11 = _mm_mullo_epi16(H1, coeff9);
            p21 = _mm_mullo_epi16(H2, coeff15);
            p31 = _mm_mullo_epi16(H3, coeff7);
            p01 = _mm_add_epi16(H0, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst7, p00);


            p10 = _mm_mullo_epi16(L2, coeff2);
            p00 = _mm_add_epi16(L1, L3);
            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_srli_epi16(p00, 2);

            p11 = _mm_mullo_epi16(H2, coeff2);
            p01 = _mm_add_epi16(H1, H3);
            p01 = _mm_add_epi16(p01, coeff2);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_srli_epi16(p01, 2);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst8, p00);

            src += 16;
            dst1 += 16;
            dst2 += 16;
            dst3 += 16;
            dst4 += 16;
            dst5 += 16;
            dst6 += 16;
            dst7 += 16;
            dst8 += 16;

            S0 = _mm_loadu_si128((__m128i*)(src + 2));
            S1 = _mm_loadu_si128((__m128i*)(src + 1));
            S2 = _mm_loadu_si128((__m128i*)(src));
            S3 = _mm_loadu_si128((__m128i*)(src - 1));

            L0 = _mm_unpacklo_epi8(S0, zero);
            L1 = _mm_unpacklo_epi8(S1, zero);
            L2 = _mm_unpacklo_epi8(S2, zero);
            L3 = _mm_unpacklo_epi8(S3, zero);

            H0 = _mm_unpackhi_epi8(S0, zero);
            H1 = _mm_unpackhi_epi8(S1, zero);
            H2 = _mm_unpackhi_epi8(S2, zero);
            H3 = _mm_unpackhi_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff7);
            p10 = _mm_mullo_epi16(L1, coeff15);
            p20 = _mm_mullo_epi16(L2, coeff9);
            p30 = _mm_add_epi16(L3, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H0, coeff7);
            p11 = _mm_mullo_epi16(H1, coeff15);
            p21 = _mm_mullo_epi16(H2, coeff9);
            p31 = _mm_add_epi16(H3, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst1, p00);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff7);
            p21 = _mm_mullo_epi16(H2, coeff5);
            p31 = _mm_add_epi16(H3, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst2, p00);

            p00 = _mm_mullo_epi16(L0, coeff5);
            p10 = _mm_mullo_epi16(L1, coeff13);
            p20 = _mm_mullo_epi16(L2, coeff11);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H0, coeff5);
            p11 = _mm_mullo_epi16(H1, coeff13);
            p21 = _mm_mullo_epi16(H2, coeff11);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(p01, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst3, p00);

            p00 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(L1, L2);
            p10 = _mm_mullo_epi16(p10, coeff3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_srli_epi16(p00, 3);

            p01 = _mm_add_epi16(H0, H3);
            p11 = _mm_add_epi16(H1, H2);
            p11 = _mm_mullo_epi16(p11, coeff3);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, coeff4);
            p01 = _mm_srli_epi16(p01, 3);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst4, p00);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff11);
            p20 = _mm_mullo_epi16(L2, coeff13);
            p30 = _mm_mullo_epi16(L3, coeff5);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff11);
            p21 = _mm_mullo_epi16(H2, coeff13);
            p31 = _mm_mullo_epi16(H3, coeff5);
            p01 = _mm_add_epi16(p01, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst5, p00);

            p10 = _mm_mullo_epi16(L1, coeff5);
            p20 = _mm_mullo_epi16(L2, coeff7);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(L0, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p11 = _mm_mullo_epi16(H1, coeff5);
            p21 = _mm_mullo_epi16(H2, coeff7);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(H0, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst6, p00);

            p10 = _mm_mullo_epi16(L1, coeff9);
            p20 = _mm_mullo_epi16(L2, coeff15);
            p30 = _mm_mullo_epi16(L3, coeff7);
            p00 = _mm_add_epi16(L0, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p11 = _mm_mullo_epi16(H1, coeff9);
            p21 = _mm_mullo_epi16(H2, coeff15);
            p31 = _mm_mullo_epi16(H3, coeff7);
            p01 = _mm_add_epi16(H0, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst7, p00);


            p10 = _mm_mullo_epi16(L2, coeff2);
            p00 = _mm_add_epi16(L1, L3);
            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_srli_epi16(p00, 2);

            p11 = _mm_mullo_epi16(H2, coeff2);
            p01 = _mm_add_epi16(H1, H3);
            p01 = _mm_add_epi16(p01, coeff2);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_srli_epi16(p01, 2);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst8, p00);
        } else {
            __m128i p00, p10, p20, p30;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src + 2));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src - 1));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff7);
            p10 = _mm_mullo_epi16(L1, coeff15);
            p20 = _mm_mullo_epi16(L2, coeff9);
            p30 = _mm_add_epi16(L3, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)dst1, p00);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)dst2, p00);

            p00 = _mm_mullo_epi16(L0, coeff5);
            p10 = _mm_mullo_epi16(L1, coeff13);
            p20 = _mm_mullo_epi16(L2, coeff11);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)dst3, p00);

            p00 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(L1, L2);
            p10 = _mm_mullo_epi16(p10, coeff3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_srli_epi16(p00, 3);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)dst4, p00);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff11);
            p20 = _mm_mullo_epi16(L2, coeff13);
            p30 = _mm_mullo_epi16(L3, coeff5);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)dst5, p00);

            p10 = _mm_mullo_epi16(L1, coeff5);
            p20 = _mm_mullo_epi16(L2, coeff7);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(L0, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)dst6, p00);

            p10 = _mm_mullo_epi16(L1, coeff9);
            p20 = _mm_mullo_epi16(L2, coeff15);
            p30 = _mm_mullo_epi16(L3, coeff7);
            p00 = _mm_add_epi16(L0, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)dst7, p00);


            p10 = _mm_mullo_epi16(L2, coeff2);
            p00 = _mm_add_epi16(L1, L3);
            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_srli_epi16(p00, 2);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)dst8, p00);
        }
    } else {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;

        if (bsx == 16) {
            __m128i p00, p10, p20, p30;
            __m128i p01, p11, p21, p31;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src + 2));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src - 1));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);
            __m128i H3 = _mm_unpackhi_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff7);
            p10 = _mm_mullo_epi16(L1, coeff15);
            p20 = _mm_mullo_epi16(L2, coeff9);
            p30 = _mm_add_epi16(L3, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H0, coeff7);
            p11 = _mm_mullo_epi16(H1, coeff15);
            p21 = _mm_mullo_epi16(H2, coeff9);
            p31 = _mm_add_epi16(H3, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst1, p00);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff7);
            p21 = _mm_mullo_epi16(H2, coeff5);
            p31 = _mm_add_epi16(H3, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst2, p00);

            p00 = _mm_mullo_epi16(L0, coeff5);
            p10 = _mm_mullo_epi16(L1, coeff13);
            p20 = _mm_mullo_epi16(L2, coeff11);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H0, coeff5);
            p11 = _mm_mullo_epi16(H1, coeff13);
            p21 = _mm_mullo_epi16(H2, coeff11);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(p01, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 5);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst3, p00);

            p00 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(L1, L2);
            p10 = _mm_mullo_epi16(p10, coeff3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_srli_epi16(p00, 3);

            p01 = _mm_add_epi16(H0, H3);
            p11 = _mm_add_epi16(H1, H2);
            p11 = _mm_mullo_epi16(p11, coeff3);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, coeff4);
            p01 = _mm_srli_epi16(p01, 3);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)dst4, p00);
        } else {
            __m128i p00, p10, p20, p30;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src + 2));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src - 1));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff7);
            p10 = _mm_mullo_epi16(L1, coeff15);
            p20 = _mm_mullo_epi16(L2, coeff9);
            p30 = _mm_add_epi16(L3, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            ((int*)(dst1))[0] = _mm_cvtsi128_si32(p00);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            ((int*)(dst2))[0] = _mm_cvtsi128_si32(p00);

            p00 = _mm_mullo_epi16(L0, coeff5);
            p10 = _mm_mullo_epi16(L1, coeff13);
            p20 = _mm_mullo_epi16(L2, coeff11);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            ((int*)(dst3))[0] = _mm_cvtsi128_si32(p00);

            p00 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(L1, L2);
            p10 = _mm_mullo_epi16(p10, coeff3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_srli_epi16(p00, 3);

            p00 = _mm_packus_epi16(p00, p00);
            ((int*)(dst4))[0] = _mm_cvtsi128_si32(p00);
        }
    }
}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_xy_14_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;
    __m128i coeff2 = _mm_set1_epi16(2);
    __m128i coeff3 = _mm_set1_epi16(3);
    __m128i coeff4 = _mm_set1_epi16(4);
    __m128i coeff5 = _mm_set1_epi16(5);
    __m128i coeff7 = _mm_set1_epi16(7);
    __m128i coeff8 = _mm_set1_epi16(8);
    __m128i zero = _mm_setzero_si128();

    UNUSED_PARAMETER(dir_mode);

    if (bsy != 4) {
        ALIGN16(pel_t first_line[4 * (64 + 32)]);
        int line_size = bsx + bsy / 4 - 1;
        int left_size = line_size - bsx;
        int aligned_line_size = ((line_size + 31) >> 4) << 4;
        pel_t *pfirst[4];
        __m128i shuffle1 = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        __m128i shuffle2 = _mm_setr_epi8(1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12);
        __m128i shuffle3 = _mm_setr_epi8(2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13);
        __m128i shuffle4 = _mm_setr_epi8(3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14);
        pel_t *pSrc1 = src;

        pfirst[0] = first_line;
        pfirst[1] = first_line + aligned_line_size;
        pfirst[2] = first_line + aligned_line_size * 2;
        pfirst[3] = first_line + aligned_line_size * 3;
        src -= bsy - 4;
        for (i = 0; i < left_size - 1; i += 4, src += 16) {
            __m128i p00, p01, p10, p11;
            __m128i p20, p30;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);

            p00 = _mm_add_epi16(L0, L1);
            p01 = _mm_add_epi16(L1, L2);
            p10 = _mm_add_epi16(H0, H1);
            p11 = _mm_add_epi16(H1, H2);

            p00 = _mm_add_epi16(p00, coeff2);
            p10 = _mm_add_epi16(p10, coeff2);
            p00 = _mm_add_epi16(p00, p01);
            p10 = _mm_add_epi16(p10, p11);

            p00 = _mm_srli_epi16(p00, 2);
            p10 = _mm_srli_epi16(p10, 2);

            p00 = _mm_packus_epi16(p00, p10);
            p10 = _mm_shuffle_epi8(p00, shuffle2);
            p20 = _mm_shuffle_epi8(p00, shuffle3);
            p30 = _mm_shuffle_epi8(p00, shuffle4);
            p00 = _mm_shuffle_epi8(p00, shuffle1);

            ((int*)&pfirst[0][i])[0] = _mm_cvtsi128_si32(p30);
            ((int*)&pfirst[1][i])[0] = _mm_cvtsi128_si32(p20);
            ((int*)&pfirst[2][i])[0] = _mm_cvtsi128_si32(p10);
            ((int*)&pfirst[3][i])[0] = _mm_cvtsi128_si32(p00);
        }

        if (i < left_size) { //使用c语言可能会更优
            __m128i p00, p01, p10;
            __m128i p20, p30;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);

            p00 = _mm_add_epi16(L0, L1);
            p01 = _mm_add_epi16(L1, L2);

            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_add_epi16(p00, p01);

            p00 = _mm_srli_epi16(p00, 2);

            p00 = _mm_packus_epi16(p00, p00);
            p10 = _mm_shuffle_epi8(p00, shuffle2);
            p20 = _mm_shuffle_epi8(p00, shuffle3);
            p30 = _mm_shuffle_epi8(p00, shuffle4);
            p00 = _mm_shuffle_epi8(p00, shuffle1);

            ((int*)&pfirst[0][i])[0] = _mm_cvtsi128_si32(p30);
            ((int*)&pfirst[1][i])[0] = _mm_cvtsi128_si32(p20);
            ((int*)&pfirst[2][i])[0] = _mm_cvtsi128_si32(p10);
            ((int*)&pfirst[3][i])[0] = _mm_cvtsi128_si32(p00);
        }

        src = pSrc1;

        for (i = left_size; i < line_size; i++, src++) {
            __m128i p00, p10, p20, p30;
            __m128i p01, p11, p21, p31;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src + 2));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);
            __m128i H3 = _mm_unpackhi_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_srli_epi16(p00, 4);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff7);
            p21 = _mm_mullo_epi16(H2, coeff5);
            p31 = _mm_add_epi16(H3, coeff8);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_storeu_si128((__m128i*)&pfirst[2][i], p00);

            p00 = _mm_add_epi16(L1, L2);
            p00 = _mm_mullo_epi16(p00, coeff3);
            p10 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(p10, coeff4);
            p00 = _mm_add_epi16(p10, p00);
            p00 = _mm_srli_epi16(p00, 3);

            p01 = _mm_add_epi16(H1, H2);
            p01 = _mm_mullo_epi16(p01, coeff3);
            p11 = _mm_add_epi16(H0, H3);
            p11 = _mm_add_epi16(p11, coeff4);
            p01 = _mm_add_epi16(p11, p01);
            p01 = _mm_srli_epi16(p01, 3);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_storeu_si128((__m128i*)&pfirst[1][i], p00);

            p10 = _mm_mullo_epi16(L1, coeff5);
            p20 = _mm_mullo_epi16(L2, coeff7);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(L0, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p11 = _mm_mullo_epi16(H1, coeff5);
            p21 = _mm_mullo_epi16(H2, coeff7);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(H0, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_storeu_si128((__m128i*)&pfirst[0][i], p00);

            p00 = _mm_add_epi16(L0, L1);
            p10 = _mm_add_epi16(L1, L2);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_srli_epi16(p00, 2);

            p01 = _mm_add_epi16(H0, H1);
            p11 = _mm_add_epi16(H1, H2);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, coeff2);
            p01 = _mm_srli_epi16(p01, 2);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_storeu_si128((__m128i*)&pfirst[3][i], p00);
        }

        pfirst[0] += left_size;
        pfirst[1] += left_size;
        pfirst[2] += left_size;
        pfirst[3] += left_size;

        bsy >>= 2;

        for (i = 0; i < bsy; i++) {
            memcpy(dst, pfirst[0] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[1] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[2] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[3] - i, bsx * sizeof(pel_t));
            dst += i_dst;
        }
    } else {
        if (bsx == 16) {
            pel_t *dst2 = dst + i_dst;
            pel_t *dst3 = dst2 + i_dst;
            pel_t *dst4 = dst3 + i_dst;
            __m128i p00, p10, p20, p30;
            __m128i p01, p11, p21, p31;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src + 2));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);
            __m128i H3 = _mm_unpackhi_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_srli_epi16(p00, 4);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff7);
            p21 = _mm_mullo_epi16(H2, coeff5);
            p31 = _mm_add_epi16(H3, coeff8);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_storeu_si128((__m128i*)dst3, p00);

            p00 = _mm_add_epi16(L1, L2);
            p00 = _mm_mullo_epi16(p00, coeff3);
            p10 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(p10, coeff4);
            p00 = _mm_add_epi16(p10, p00);
            p00 = _mm_srli_epi16(p00, 3);

            p01 = _mm_add_epi16(H1, H2);
            p01 = _mm_mullo_epi16(p01, coeff3);
            p11 = _mm_add_epi16(H0, H3);
            p11 = _mm_add_epi16(p11, coeff4);
            p01 = _mm_add_epi16(p11, p01);
            p01 = _mm_srli_epi16(p01, 3);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_storeu_si128((__m128i*)dst2, p00);

            p10 = _mm_mullo_epi16(L1, coeff5);
            p20 = _mm_mullo_epi16(L2, coeff7);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(L0, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p11 = _mm_mullo_epi16(H1, coeff5);
            p21 = _mm_mullo_epi16(H2, coeff7);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(H0, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_srli_epi16(p01, 4);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_storeu_si128((__m128i*)dst, p00);

            p00 = _mm_add_epi16(L0, L1);
            p10 = _mm_add_epi16(L1, L2);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_srli_epi16(p00, 2);

            p01 = _mm_add_epi16(H0, H1);
            p11 = _mm_add_epi16(H1, H2);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, coeff2);
            p01 = _mm_srli_epi16(p01, 2);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_storeu_si128((__m128i*)dst4, p00);
        } else {
            pel_t *dst2 = dst + i_dst;
            pel_t *dst3 = dst2 + i_dst;
            pel_t *dst4 = dst3 + i_dst;
            __m128i p00, p10, p20, p30;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src + 2));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            ((int*)dst3)[0] = _mm_cvtsi128_si32(p00);

            p00 = _mm_add_epi16(L1, L2);
            p00 = _mm_mullo_epi16(p00, coeff3);
            p10 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(p10, coeff4);
            p00 = _mm_add_epi16(p10, p00);
            p00 = _mm_srli_epi16(p00, 3);

            p00 = _mm_packus_epi16(p00, p00);
            ((int*)dst2)[0] = _mm_cvtsi128_si32(p00);

            p10 = _mm_mullo_epi16(L1, coeff5);
            p20 = _mm_mullo_epi16(L2, coeff7);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(L0, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            ((int*)dst)[0] = _mm_cvtsi128_si32(p00);

            p00 = _mm_add_epi16(L0, L1);
            p10 = _mm_add_epi16(L1, L2);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_srli_epi16(p00, 2);

            p00 = _mm_packus_epi16(p00, p00);
            ((int*)dst4)[0] = _mm_cvtsi128_si32(p00);
        }
    }
}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_xy_16_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[2 * (64 + 48)]);
    int line_size = bsx + bsy / 2 - 1;
    int left_size = line_size - bsx;
    int aligned_line_size = ((line_size + 31) >> 4) << 4;
    pel_t *pfirst[2];
    __m128i zero = _mm_setzero_si128();
    __m128i coeff2 = _mm_set1_epi16(2);
    __m128i coeff3 = _mm_set1_epi16(3);
    __m128i coeff4 = _mm_set1_epi16(4);
    __m128i shuffle1 = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    __m128i shuffle2 = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14);
    int i;
    pel_t *pSrc1;

    UNUSED_PARAMETER(dir_mode);

    pfirst[0] = first_line;
    pfirst[1] = first_line + aligned_line_size;

    src -= bsy - 2;

    pSrc1 = src;

    for (i = 0; i < left_size - 4; i += 8, src += 16) {
        __m128i p00, p01, p10, p11;
        __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);

        __m128i H0 = _mm_unpackhi_epi8(S0, zero);
        __m128i H1 = _mm_unpackhi_epi8(S1, zero);
        __m128i H2 = _mm_unpackhi_epi8(S2, zero);

        p00 = _mm_add_epi16(L0, L1);
        p01 = _mm_add_epi16(L1, L2);
        p10 = _mm_add_epi16(H0, H1);
        p11 = _mm_add_epi16(H1, H2);

        p00 = _mm_add_epi16(p00, coeff2);
        p10 = _mm_add_epi16(p10, coeff2);

        p00 = _mm_add_epi16(p00, p01);
        p10 = _mm_add_epi16(p10, p11);

        p00 = _mm_srli_epi16(p00, 2);
        p10 = _mm_srli_epi16(p10, 2);
        p00 = _mm_packus_epi16(p00, p10);

        p10 = _mm_shuffle_epi8(p00, shuffle2);
        p00 = _mm_shuffle_epi8(p00, shuffle1);
        _mm_storel_epi64((__m128i*)&pfirst[1][i], p00);
        _mm_storel_epi64((__m128i*)&pfirst[0][i], p10);
    }

    if (i < left_size) {
        __m128i p00, p01;
        __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);

        p00 = _mm_add_epi16(L0, L1);
        p01 = _mm_add_epi16(L1, L2);
        p00 = _mm_add_epi16(p00, coeff2);
        p00 = _mm_add_epi16(p00, p01);
        p00 = _mm_srli_epi16(p00, 2);
        p00 = _mm_packus_epi16(p00, p00);

        p01 = _mm_shuffle_epi8(p00, shuffle2);
        p00 = _mm_shuffle_epi8(p00, shuffle1);
        ((int*)&pfirst[1][i])[0] = _mm_cvtsi128_si32(p00);
        ((int*)&pfirst[0][i])[0] = _mm_cvtsi128_si32(p01);
    }

    src = pSrc1 + left_size + left_size;

    for (i = left_size; i < line_size; i += 16, src += 16) {
        __m128i p00, p01, p10, p11;
        __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S3 = _mm_loadu_si128((__m128i*)(src + 2));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);
        __m128i L3 = _mm_unpacklo_epi8(S3, zero);

        __m128i H0 = _mm_unpackhi_epi8(S0, zero);
        __m128i H1 = _mm_unpackhi_epi8(S1, zero);
        __m128i H2 = _mm_unpackhi_epi8(S2, zero);
        __m128i H3 = _mm_unpackhi_epi8(S3, zero);

        p00 = _mm_add_epi16(L1, L2);
        p10 = _mm_add_epi16(H1, H2);
        p00 = _mm_mullo_epi16(p00, coeff3);
        p10 = _mm_mullo_epi16(p10, coeff3);

        p01 = _mm_add_epi16(L0, L3);
        p11 = _mm_add_epi16(H0, H3);
        p00 = _mm_add_epi16(p00, coeff4);
        p10 = _mm_add_epi16(p10, coeff4);
        p00 = _mm_add_epi16(p00, p01);
        p10 = _mm_add_epi16(p10, p11);

        p00 = _mm_srli_epi16(p00, 3);
        p10 = _mm_srli_epi16(p10, 3);

        p00 = _mm_packus_epi16(p00, p10);
        _mm_storeu_si128((__m128i*)&pfirst[0][i], p00);

        p00 = _mm_add_epi16(L0, L1);
        p01 = _mm_add_epi16(L1, L2);
        p10 = _mm_add_epi16(H0, H1);
        p11 = _mm_add_epi16(H1, H2);

        p00 = _mm_add_epi16(p00, coeff2);
        p10 = _mm_add_epi16(p10, coeff2);

        p00 = _mm_add_epi16(p00, p01);
        p10 = _mm_add_epi16(p10, p11);

        p00 = _mm_srli_epi16(p00, 2);
        p10 = _mm_srli_epi16(p10, 2);

        p00 = _mm_packus_epi16(p00, p10);
        _mm_storeu_si128((__m128i*)&pfirst[1][i], p00);
    }

    pfirst[0] += left_size;
    pfirst[1] += left_size;

    bsy >>= 1;

    switch (bsx) {
    case 4:
        for (i = 0; i < bsy; i++) {
            CP32(dst, pfirst[0] - i);
            CP32(dst + i_dst, pfirst[1] - i);
            dst += (i_dst << 1);
        }
        break;
    case 8:
        for (i = 0; i < bsy; i++) {
            CP64(dst, pfirst[0] - i);
            CP64(dst + i_dst, pfirst[1] - i);
            dst += (i_dst << 1);
        }
        break;
    default:
        for (i = 0; i < bsy; i++) {
            memcpy(dst, pfirst[0] - i, bsx * sizeof(pel_t));
            memcpy(dst + i_dst, pfirst[1] - i, bsx * sizeof(pel_t));
            dst += (i_dst << 1);
        }
        break;
    }
}

/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_xy_18_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[64 + 64]);
    int line_size = bsx + bsy - 1;
    int i;
    pel_t *pfirst = first_line + bsy - 1;
    __m128i coeff2 = _mm_set1_epi16(2);
    __m128i zero = _mm_setzero_si128();

    UNUSED_PARAMETER(dir_mode);

    src -= bsy - 1;

    for (i = 0; i < line_size - 8; i += 16, src += 16) {
        __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);

        __m128i H0 = _mm_unpackhi_epi8(S0, zero);
        __m128i H1 = _mm_unpackhi_epi8(S1, zero);
        __m128i H2 = _mm_unpackhi_epi8(S2, zero);

        __m128i sum1 = _mm_add_epi16(L0, L1);
        __m128i sum2 = _mm_add_epi16(L1, L2);
        __m128i sum3 = _mm_add_epi16(H0, H1);
        __m128i sum4 = _mm_add_epi16(H1, H2);

        sum1 = _mm_add_epi16(sum1, sum2);
        sum3 = _mm_add_epi16(sum3, sum4);

        sum1 = _mm_add_epi16(sum1, coeff2);
        sum3 = _mm_add_epi16(sum3, coeff2);

        sum1 = _mm_srli_epi16(sum1, 2);
        sum3 = _mm_srli_epi16(sum3, 2);

        sum1 = _mm_packus_epi16(sum1, sum3);

        _mm_store_si128((__m128i*)&first_line[i], sum1);
    }

    if (i < line_size) {
        __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);

        __m128i sum1 = _mm_add_epi16(L0, L1);
        __m128i sum2 = _mm_add_epi16(L1, L2);

        sum1 = _mm_add_epi16(sum1, sum2);
        sum1 = _mm_add_epi16(sum1, coeff2);
        sum1 = _mm_srli_epi16(sum1, 2);

        sum1 = _mm_packus_epi16(sum1, sum1);
        _mm_storel_epi64((__m128i*)&first_line[i], sum1);
    }

    switch (bsx) {
    case 4:
        for (i = 0; i < bsy; i++) {
            CP32(dst, pfirst--);
            dst += i_dst;
        }
        break;
    case 8:
        for (i = 0; i < bsy; i++) {
            CP64(dst, pfirst--);
            dst += i_dst;
        }
        break;
    default:
        for (i = 0; i < bsy; i++) {
            memcpy(dst, pfirst--, bsx * sizeof(pel_t));
            dst += i_dst;
        }
        break;
        break;
    }
}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_xy_20_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN16(pel_t first_line[64 + 128]);
    int left_size = (bsy - 1) * 2 + 1;
    int top_size = bsx - 1;
    int line_size = left_size + top_size;
    int i;
    pel_t *pfirst = first_line + left_size - 1;
    __m128i zero = _mm_setzero_si128();
    __m128i coeff2 = _mm_set1_epi16(2);
    __m128i coeff3 = _mm_set1_epi16(3);
    __m128i coeff4 = _mm_set1_epi16(4);
    __m128i shuffle = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
    pel_t *pSrc1 = src;

    UNUSED_PARAMETER(dir_mode);

    src -= bsy;

    for (i = 0; i < left_size - 16; i += 32, src += 16) {
        __m128i p00, p01, p10, p11;
        __m128i p20, p21, p30, p31;
        __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S3 = _mm_loadu_si128((__m128i*)(src + 2));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);
        __m128i L3 = _mm_unpacklo_epi8(S3, zero);

        __m128i H0 = _mm_unpackhi_epi8(S0, zero);
        __m128i H1 = _mm_unpackhi_epi8(S1, zero);
        __m128i H2 = _mm_unpackhi_epi8(S2, zero);
        __m128i H3 = _mm_unpackhi_epi8(S3, zero);

        p00 = _mm_add_epi16(L1, L2);
        p10 = _mm_add_epi16(H1, H2);
        p00 = _mm_mullo_epi16(p00, coeff3);
        p10 = _mm_mullo_epi16(p10, coeff3);

        p01 = _mm_add_epi16(L0, L3);
        p11 = _mm_add_epi16(H0, H3);
        p00 = _mm_add_epi16(p00, coeff4);
        p10 = _mm_add_epi16(p10, coeff4);
        p00 = _mm_add_epi16(p00, p01);
        p10 = _mm_add_epi16(p10, p11);

        p00 = _mm_srli_epi16(p00, 3);
        p10 = _mm_srli_epi16(p10, 3);

        p20 = _mm_add_epi16(L1, L2);
        p30 = _mm_add_epi16(H1, H2);
        p21 = _mm_add_epi16(L2, L3);
        p31 = _mm_add_epi16(H2, H3);
        p20 = _mm_add_epi16(p20, coeff2);
        p30 = _mm_add_epi16(p30, coeff2);
        p20 = _mm_add_epi16(p20, p21);
        p30 = _mm_add_epi16(p30, p31);

        p20 = _mm_srli_epi16(p20, 2);
        p30 = _mm_srli_epi16(p30, 2);

        p00 = _mm_packus_epi16(p00, p20);
        p10 = _mm_packus_epi16(p10, p30);

        p00 = _mm_shuffle_epi8(p00, shuffle);
        p10 = _mm_shuffle_epi8(p10, shuffle);
        _mm_store_si128((__m128i*)&first_line[i], p00);
        _mm_store_si128((__m128i*)&first_line[i + 16], p10);
    }

    if (i < left_size) {
        __m128i p00, p01;
        __m128i p20, p21;
        __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S3 = _mm_loadu_si128((__m128i*)(src + 2));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);
        __m128i L3 = _mm_unpacklo_epi8(S3, zero);

        p00 = _mm_add_epi16(L1, L2);
        p00 = _mm_mullo_epi16(p00, coeff3);

        p01 = _mm_add_epi16(L0, L3);
        p00 = _mm_add_epi16(p00, coeff4);
        p00 = _mm_add_epi16(p00, p01);

        p00 = _mm_srli_epi16(p00, 3);

        p20 = _mm_add_epi16(L1, L2);
        p21 = _mm_add_epi16(L2, L3);
        p20 = _mm_add_epi16(p20, coeff2);
        p20 = _mm_add_epi16(p20, p21);

        p20 = _mm_srli_epi16(p20, 2);

        p00 = _mm_packus_epi16(p00, p20);

        p00 = _mm_shuffle_epi8(p00, shuffle);
        _mm_store_si128((__m128i*)&first_line[i], p00);
    }

    src = pSrc1;

    for (i = left_size; i < line_size - 8; i += 16, src += 16) {
        __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);

        __m128i H0 = _mm_unpackhi_epi8(S0, zero);
        __m128i H1 = _mm_unpackhi_epi8(S1, zero);
        __m128i H2 = _mm_unpackhi_epi8(S2, zero);

        __m128i sum1 = _mm_add_epi16(L0, L1);
        __m128i sum2 = _mm_add_epi16(L1, L2);
        __m128i sum3 = _mm_add_epi16(H0, H1);
        __m128i sum4 = _mm_add_epi16(H1, H2);

        sum1 = _mm_add_epi16(sum1, sum2);
        sum3 = _mm_add_epi16(sum3, sum4);

        sum1 = _mm_add_epi16(sum1, coeff2);
        sum3 = _mm_add_epi16(sum3, coeff2);

        sum1 = _mm_srli_epi16(sum1, 2);
        sum3 = _mm_srli_epi16(sum3, 2);

        sum1 = _mm_packus_epi16(sum1, sum3);

        _mm_storeu_si128((__m128i*)&first_line[i], sum1);
    }

    if (i < line_size) {
        __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
        __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
        __m128i S1 = _mm_loadu_si128((__m128i*)(src));

        __m128i L0 = _mm_unpacklo_epi8(S0, zero);
        __m128i L1 = _mm_unpacklo_epi8(S1, zero);
        __m128i L2 = _mm_unpacklo_epi8(S2, zero);

        __m128i sum1 = _mm_add_epi16(L0, L1);
        __m128i sum2 = _mm_add_epi16(L1, L2);

        sum1 = _mm_add_epi16(sum1, sum2);
        sum1 = _mm_add_epi16(sum1, coeff2);
        sum1 = _mm_srli_epi16(sum1, 2);

        sum1 = _mm_packus_epi16(sum1, sum1);
        _mm_storel_epi64((__m128i*)&first_line[i], sum1);
    }

    for (i = 0; i < bsy; i++) {
        memcpy(dst, pfirst, bsx * sizeof(pel_t));
        pfirst -= 2;
        dst += i_dst;
    }
}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_xy_22_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;

    UNUSED_PARAMETER(dir_mode);

    src -= bsy;
    if (bsx != 4) {
        ALIGN16(pel_t first_line[64 + 256]);
        int left_size = (bsy - 1) * 4 + 3;
        int top_size = bsx - 3;
        int line_size = left_size + top_size;
        pel_t *pfirst = first_line + left_size - 3;
        pel_t *pSrc1 = src;

        __m128i zero = _mm_setzero_si128();
        __m128i coeff2 = _mm_set1_epi16(2);
        __m128i coeff3 = _mm_set1_epi16(3);
        __m128i coeff4 = _mm_set1_epi16(4);
        __m128i coeff5 = _mm_set1_epi16(5);
        __m128i coeff7 = _mm_set1_epi16(7);
        __m128i coeff8 = _mm_set1_epi16(8);
        __m128i shuffle = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);

        for (i = 0; i < line_size - 32; i += 64, src += 16) {
            __m128i p00, p10, p20, p30;
            __m128i p01, p11, p21, p31;
            __m128i M1, M2, M3, M4, M5, M6, M7, M8;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src + 2));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);
            __m128i H3 = _mm_unpackhi_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            M1 = _mm_srli_epi16(p00, 4);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff7);
            p21 = _mm_mullo_epi16(H2, coeff5);
            p31 = _mm_add_epi16(H3, coeff8);
            p01 = _mm_add_epi16(p01, p31);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            M2 = _mm_srli_epi16(p01, 4);


            p00 = _mm_add_epi16(L1, L2);
            p00 = _mm_mullo_epi16(p00, coeff3);
            p10 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(p10, coeff4);
            p00 = _mm_add_epi16(p10, p00);
            M3 = _mm_srli_epi16(p00, 3);

            p01 = _mm_add_epi16(H1, H2);
            p01 = _mm_mullo_epi16(p01, coeff3);
            p11 = _mm_add_epi16(H0, H3);
            p11 = _mm_add_epi16(p11, coeff4);
            p01 = _mm_add_epi16(p11, p01);
            M4 = _mm_srli_epi16(p01, 3);


            p10 = _mm_mullo_epi16(L1, coeff5);
            p20 = _mm_mullo_epi16(L2, coeff7);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(L0, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            M5 = _mm_srli_epi16(p00, 4);

            p11 = _mm_mullo_epi16(H1, coeff5);
            p21 = _mm_mullo_epi16(H2, coeff7);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(H0, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            M6 = _mm_srli_epi16(p01, 4);


            p00 = _mm_add_epi16(L1, L2);
            p10 = _mm_add_epi16(L2, L3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff2);
            M7 = _mm_srli_epi16(p00, 2);

            p01 = _mm_add_epi16(H1, H2);
            p11 = _mm_add_epi16(H2, H3);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, coeff2);
            M8 = _mm_srli_epi16(p01, 2);

            M1 = _mm_packus_epi16(M1, M3);
            M5 = _mm_packus_epi16(M5, M7);
            M1 = _mm_shuffle_epi8(M1, shuffle);
            M5 = _mm_shuffle_epi8(M5, shuffle);

            M2 = _mm_packus_epi16(M2, M4);
            M6 = _mm_packus_epi16(M6, M8);
            M2 = _mm_shuffle_epi8(M2, shuffle);
            M6 = _mm_shuffle_epi8(M6, shuffle);

            M3 = _mm_unpacklo_epi16(M1, M5);
            M7 = _mm_unpackhi_epi16(M1, M5);
            M4 = _mm_unpacklo_epi16(M2, M6);
            M8 = _mm_unpackhi_epi16(M2, M6);

            _mm_store_si128((__m128i*)&first_line[i], M3);
            _mm_store_si128((__m128i*)&first_line[16 + i], M7);
            _mm_store_si128((__m128i*)&first_line[32 + i], M4);
            _mm_store_si128((__m128i*)&first_line[48 + i], M8);
        }

        if (i < left_size) {
            __m128i p00, p10, p20, p30;
            __m128i M1, M3, M5, M7;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src + 2));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            M1 = _mm_srli_epi16(p00, 4);

            p00 = _mm_add_epi16(L1, L2);
            p00 = _mm_mullo_epi16(p00, coeff3);
            p10 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(p10, coeff4);
            p00 = _mm_add_epi16(p10, p00);
            M3 = _mm_srli_epi16(p00, 3);

            p10 = _mm_mullo_epi16(L1, coeff5);
            p20 = _mm_mullo_epi16(L2, coeff7);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(L0, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            M5 = _mm_srli_epi16(p00, 4);

            p00 = _mm_add_epi16(L1, L2);
            p10 = _mm_add_epi16(L2, L3);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff2);
            M7 = _mm_srli_epi16(p00, 2);

            M1 = _mm_packus_epi16(M1, M3);
            M5 = _mm_packus_epi16(M5, M7);
            M1 = _mm_shuffle_epi8(M1, shuffle);
            M5 = _mm_shuffle_epi8(M5, shuffle);

            M3 = _mm_unpacklo_epi16(M1, M5);
            M7 = _mm_unpackhi_epi16(M1, M5);

            _mm_store_si128((__m128i*)&first_line[i], M3);
            _mm_store_si128((__m128i*)&first_line[16 + i], M7);
        }

        src = pSrc1 + bsy;

        for (i = left_size; i < line_size - 8; i += 16, src += 16) {
            __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);

            __m128i sum1 = _mm_add_epi16(L0, L1);
            __m128i sum2 = _mm_add_epi16(L1, L2);
            __m128i sum3 = _mm_add_epi16(H0, H1);
            __m128i sum4 = _mm_add_epi16(H1, H2);

            sum1 = _mm_add_epi16(sum1, sum2);
            sum3 = _mm_add_epi16(sum3, sum4);

            sum1 = _mm_add_epi16(sum1, coeff2);
            sum3 = _mm_add_epi16(sum3, coeff2);

            sum1 = _mm_srli_epi16(sum1, 2);
            sum3 = _mm_srli_epi16(sum3, 2);

            sum1 = _mm_packus_epi16(sum1, sum3);

            _mm_storeu_si128((__m128i*)&first_line[i], sum1);
        }

        if (i < line_size) {
            __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);

            __m128i sum1 = _mm_add_epi16(L0, L1);
            __m128i sum2 = _mm_add_epi16(L1, L2);

            sum1 = _mm_add_epi16(sum1, sum2);
            sum1 = _mm_add_epi16(sum1, coeff2);
            sum1 = _mm_srli_epi16(sum1, 2);

            sum1 = _mm_packus_epi16(sum1, sum1);
            _mm_storel_epi64((__m128i*)&first_line[i], sum1);
        }

        switch (bsx) {
        case 8:
            while (bsy--) {
                CP64(dst, pfirst);
                dst += i_dst;
                pfirst -= 4;
            }
            break;
        case 16:
        case 32:
        case 64:
            while (bsy--) {
                memcpy(dst, pfirst, bsx * sizeof(pel_t));
                dst += i_dst;
                pfirst -= 4;
            }
            break;
        default:
            assert(0);
            break;
        }
    } else {
        dst += (bsy - 1) * i_dst;
        for (i = 0; i < bsy; i++, src++) {
            dst[0] = (src[-1] * 3 + src[0] * 7 + src[1] * 5 + src[2] + 8) >> 4;
            dst[1] = (src[-1] + (src[0] + src[1]) * 3 + src[2] + 4) >> 3;
            dst[2] = (src[-1] + src[0] * 5 + src[1] * 7 + src[2] * 3 + 8) >> 4;
            dst[3] = (src[0] + src[1] * 2 + src[2] + 2) >> 2;
            dst -= i_dst;
        }
    }

}


/* ---------------------------------------------------------------------------
*/
void intra_pred_ang_xy_23_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{

    int i;

    UNUSED_PARAMETER(dir_mode);

    if (bsx > 8) {
        ALIGN16(pel_t first_line[64 + 512]);
        int left_size = (bsy << 3) - 1;
        int top_size = bsx - 7;
        int line_size = left_size + top_size;
        pel_t *pfirst = first_line + left_size - 7;
        pel_t *pfirst1 = first_line;
        pel_t *src_org = src;

        src -= bsy;

        __m128i zero = _mm_setzero_si128();
        __m128i coeff0 = _mm_setr_epi16(7, 3, 5, 1, 3, 1, 1, 0);
        __m128i coeff1 = _mm_setr_epi16(15, 7, 13, 3, 11, 5, 9, 1);
        __m128i coeff2 = _mm_setr_epi16(9, 5, 11, 3, 13, 7, 15, 2);
        __m128i coeff3 = _mm_setr_epi16(1, 1, 3, 1, 5, 3, 7, 1);
        __m128i coeff4 = _mm_setr_epi16(16, 8, 16, 4, 16, 8, 16, 2);
        __m128i coeff5 = _mm_setr_epi16(1, 2, 1, 4, 1, 2, 1, 8);

        __m128i p00, p10, p20, p30;

        __m128i L0 = _mm_set1_epi16(src[-1]);
        __m128i L1 = _mm_set1_epi16(src[0]);
        __m128i L2 = _mm_set1_epi16(src[1]);
        __m128i L3 = _mm_set1_epi16(src[2]);

        src += 4;

        for (i = 0; i < left_size + 1; i += 32, src += 4) {
            p00 = _mm_mullo_epi16(L0, coeff0);
            p10 = _mm_mullo_epi16(L1, coeff1);
            p20 = _mm_mullo_epi16(L2, coeff2);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)pfirst1, p00);

            pfirst1 += 8;
            L0 = _mm_set1_epi16(src[-1]);

            p00 = _mm_mullo_epi16(L1, coeff0);
            p10 = _mm_mullo_epi16(L2, coeff1);
            p20 = _mm_mullo_epi16(L3, coeff2);
            p30 = _mm_mullo_epi16(L0, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)pfirst1, p00);

            pfirst1 += 8;
            L1 = _mm_set1_epi16(src[0]);

            p00 = _mm_mullo_epi16(L2, coeff0);
            p10 = _mm_mullo_epi16(L3, coeff1);
            p20 = _mm_mullo_epi16(L0, coeff2);
            p30 = _mm_mullo_epi16(L1, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)pfirst1, p00);

            pfirst1 += 8;
            L2 = _mm_set1_epi16(src[1]);

            p00 = _mm_mullo_epi16(L3, coeff0);
            p10 = _mm_mullo_epi16(L0, coeff1);
            p20 = _mm_mullo_epi16(L1, coeff2);
            p30 = _mm_mullo_epi16(L2, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)pfirst1, p00);

            pfirst1 += 8;
            L3 = _mm_set1_epi16(src[2]);
        }

        src = src_org + 1;
        for (; i < line_size; i += 16, src += 16) {
            coeff2 = _mm_set1_epi16(2);


            __m128i p01, p11;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src - 1));

            L0 = _mm_unpacklo_epi8(S0, zero);
            L1 = _mm_unpacklo_epi8(S1, zero);
            L2 = _mm_unpacklo_epi8(S2, zero);

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);

            p00 = _mm_mullo_epi16(L0, coeff2);
            p10 = _mm_add_epi16(L1, L2);
            p00 = _mm_add_epi16(p00, coeff2);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_srli_epi16(p00, 2);

            p01 = _mm_mullo_epi16(H0, coeff2);
            p11 = _mm_add_epi16(H1, H2);
            p01 = _mm_add_epi16(p01, coeff2);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_srli_epi16(p01, 2);

            p00 = _mm_packus_epi16(p00, p01);
            _mm_store_si128((__m128i*)&first_line[i], p00);
        }

        for (i = 0; i < bsy; i++) {
            memcpy(dst, pfirst, bsx * sizeof(pel_t));
            dst += i_dst;
            pfirst -= 8;
        }
    } else if (bsx == 8) {
        __m128i coeff0 = _mm_setr_epi16(7, 3, 5, 1, 3, 1, 1, 0);
        __m128i coeff1 = _mm_setr_epi16(15, 7, 13, 3, 11, 5, 9, 1);
        __m128i coeff2 = _mm_setr_epi16(9, 5, 11, 3, 13, 7, 15, 2);
        __m128i coeff3 = _mm_setr_epi16(1, 1, 3, 1, 5, 3, 7, 1);
        __m128i coeff4 = _mm_setr_epi16(16, 8, 16, 4, 16, 8, 16, 2);
        __m128i coeff5 = _mm_setr_epi16(1, 2, 1, 4, 1, 2, 1, 8);

        __m128i p00, p10, p20, p30;

        __m128i L0 = _mm_set1_epi16(src[-2]);
        __m128i L1 = _mm_set1_epi16(src[-1]);
        __m128i L2 = _mm_set1_epi16(src[0]);
        __m128i L3 = _mm_set1_epi16(src[1]);
        src -= 4;

        bsy >>= 2;
        for (i = 0; i < bsy; i++, src -= 4) {
            p00 = _mm_mullo_epi16(L0, coeff0);
            p10 = _mm_mullo_epi16(L1, coeff1);
            p20 = _mm_mullo_epi16(L2, coeff2);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)dst, p00);

            dst += i_dst;
            L3 = _mm_set1_epi16(src[1]);

            p00 = _mm_mullo_epi16(L3, coeff0);
            p10 = _mm_mullo_epi16(L0, coeff1);
            p20 = _mm_mullo_epi16(L1, coeff2);
            p30 = _mm_mullo_epi16(L2, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)dst, p00);

            dst += i_dst;
            L2 = _mm_set1_epi16(src[0]);

            p00 = _mm_mullo_epi16(L2, coeff0);
            p10 = _mm_mullo_epi16(L3, coeff1);
            p20 = _mm_mullo_epi16(L0, coeff2);
            p30 = _mm_mullo_epi16(L1, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)dst, p00);

            dst += i_dst;
            L1 = _mm_set1_epi16(src[-1]);

            p00 = _mm_mullo_epi16(L1, coeff0);
            p10 = _mm_mullo_epi16(L2, coeff1);
            p20 = _mm_mullo_epi16(L3, coeff2);
            p30 = _mm_mullo_epi16(L0, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_mullo_epi16(p00, coeff5);
            p00 = _mm_srli_epi16(p00, 5);

            p00 = _mm_packus_epi16(p00, p00);
            _mm_storel_epi64((__m128i*)dst, p00);

            dst += i_dst;
            L0 = _mm_set1_epi16(src[-2]);
        }
    } else {
        __m128i zero = _mm_setzero_si128();
        __m128i coeff3 = _mm_set1_epi16(3);
        __m128i coeff4 = _mm_set1_epi16(4);
        __m128i coeff5 = _mm_set1_epi16(5);
        __m128i coeff7 = _mm_set1_epi16(7);
        __m128i coeff8 = _mm_set1_epi16(8);
        __m128i coeff9 = _mm_set1_epi16(9);
        __m128i coeff11 = _mm_set1_epi16(11);
        __m128i coeff13 = _mm_set1_epi16(13);
        __m128i coeff15 = _mm_set1_epi16(15);
        __m128i coeff16 = _mm_set1_epi16(16);
        __m128i shuffle = _mm_setr_epi8(7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 0, 8);
        if (bsy == 4) {
            src -= 15;
            __m128i p01, p11, p21, p31;
            __m128i M2, M4, M6, M8;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src - 2));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src));

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);
            __m128i H3 = _mm_unpackhi_epi8(S3, zero);

            p01 = _mm_mullo_epi16(H0, coeff7);
            p11 = _mm_mullo_epi16(H1, coeff15);
            p21 = _mm_mullo_epi16(H2, coeff9);
            p31 = _mm_add_epi16(H3, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            M2 = _mm_srli_epi16(p01, 5);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff7);
            p21 = _mm_mullo_epi16(H2, coeff5);
            p31 = _mm_add_epi16(H3, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            M4 = _mm_srli_epi16(p01, 4);

            p01 = _mm_mullo_epi16(H0, coeff5);
            p11 = _mm_mullo_epi16(H1, coeff13);
            p21 = _mm_mullo_epi16(H2, coeff11);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(p01, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            M6 = _mm_srli_epi16(p01, 5);

            p01 = _mm_add_epi16(H0, H3);
            p11 = _mm_add_epi16(H1, H2);
            p11 = _mm_mullo_epi16(p11, coeff3);
            p01 = _mm_add_epi16(p01, coeff4);
            p01 = _mm_add_epi16(p01, p11);
            M8 = _mm_srli_epi16(p01, 3);

            M2 = _mm_packus_epi16(M2, M4);
            M6 = _mm_packus_epi16(M6, M8);
            M2 = _mm_shuffle_epi8(M2, shuffle);
            M6 = _mm_shuffle_epi8(M6, shuffle);

            M4 = _mm_unpacklo_epi16(M2, M6);

            *((int*)dst) = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M4);
        } else {
            src -= 15;

            __m128i p00, p10, p20, p30;
            __m128i p01, p11, p21, p31;
            __m128i M1, M2, M3, M4, M5, M6, M7, M8;
            __m128i S0 = _mm_loadu_si128((__m128i*)(src - 2));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            __m128i H0 = _mm_unpackhi_epi8(S0, zero);
            __m128i H1 = _mm_unpackhi_epi8(S1, zero);
            __m128i H2 = _mm_unpackhi_epi8(S2, zero);
            __m128i H3 = _mm_unpackhi_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff7);
            p10 = _mm_mullo_epi16(L1, coeff15);
            p20 = _mm_mullo_epi16(L2, coeff9);
            p30 = _mm_add_epi16(L3, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            M1 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H0, coeff7);
            p11 = _mm_mullo_epi16(H1, coeff15);
            p21 = _mm_mullo_epi16(H2, coeff9);
            p31 = _mm_add_epi16(H3, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            M2 = _mm_srli_epi16(p01, 5);

            p00 = _mm_mullo_epi16(L0, coeff3);
            p10 = _mm_mullo_epi16(L1, coeff7);
            p20 = _mm_mullo_epi16(L2, coeff5);
            p30 = _mm_add_epi16(L3, coeff8);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            M3 = _mm_srli_epi16(p00, 4);

            p01 = _mm_mullo_epi16(H0, coeff3);
            p11 = _mm_mullo_epi16(H1, coeff7);
            p21 = _mm_mullo_epi16(H2, coeff5);
            p31 = _mm_add_epi16(H3, coeff8);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            M4 = _mm_srli_epi16(p01, 4);

            p00 = _mm_mullo_epi16(L0, coeff5);
            p10 = _mm_mullo_epi16(L1, coeff13);
            p20 = _mm_mullo_epi16(L2, coeff11);
            p30 = _mm_mullo_epi16(L3, coeff3);
            p00 = _mm_add_epi16(p00, coeff16);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            M5 = _mm_srli_epi16(p00, 5);

            p01 = _mm_mullo_epi16(H0, coeff5);
            p11 = _mm_mullo_epi16(H1, coeff13);
            p21 = _mm_mullo_epi16(H2, coeff11);
            p31 = _mm_mullo_epi16(H3, coeff3);
            p01 = _mm_add_epi16(p01, coeff16);
            p01 = _mm_add_epi16(p01, p11);
            p01 = _mm_add_epi16(p01, p21);
            p01 = _mm_add_epi16(p01, p31);
            M6 = _mm_srli_epi16(p01, 5);

            p00 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(L1, L2);
            p10 = _mm_mullo_epi16(p10, coeff3);
            p00 = _mm_add_epi16(p00, coeff4);
            p00 = _mm_add_epi16(p00, p10);
            M7 = _mm_srli_epi16(p00, 3);

            p01 = _mm_add_epi16(H0, H3);
            p11 = _mm_add_epi16(H1, H2);
            p11 = _mm_mullo_epi16(p11, coeff3);
            p01 = _mm_add_epi16(p01, coeff4);
            p01 = _mm_add_epi16(p01, p11);
            M8 = _mm_srli_epi16(p01, 3);

            M1 = _mm_packus_epi16(M1, M3);
            M5 = _mm_packus_epi16(M5, M7);
            M1 = _mm_shuffle_epi8(M1, shuffle);
            M5 = _mm_shuffle_epi8(M5, shuffle);

            M2 = _mm_packus_epi16(M2, M4);
            M6 = _mm_packus_epi16(M6, M8);
            M2 = _mm_shuffle_epi8(M2, shuffle);
            M6 = _mm_shuffle_epi8(M6, shuffle);

            M3 = _mm_unpacklo_epi16(M1, M5);
            M7 = _mm_unpackhi_epi16(M1, M5);
            M4 = _mm_unpacklo_epi16(M2, M6);
            M8 = _mm_unpackhi_epi16(M2, M6);

            *((int*)dst) = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            M4 = _mm_srli_si128(M4, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M4);
            dst += i_dst;
            *((int*)dst) = _mm_cvtsi128_si32(M8);
            dst += i_dst;
            M8 = _mm_srli_si128(M8, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M8);
            dst += i_dst;
            M8 = _mm_srli_si128(M8, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M8);
            dst += i_dst;
            M8 = _mm_srli_si128(M8, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M8);
            dst += i_dst;
            *((int*)dst) = _mm_cvtsi128_si32(M3);
            dst += i_dst;
            M3 = _mm_srli_si128(M3, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M3);
            dst += i_dst;
            M3 = _mm_srli_si128(M3, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M3);
            dst += i_dst;
            M3 = _mm_srli_si128(M3, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M3);
            dst += i_dst;
            *((int*)dst) = _mm_cvtsi128_si32(M7);
            dst += i_dst;
            M7 = _mm_srli_si128(M7, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M7);
            dst += i_dst;
            M7 = _mm_srli_si128(M7, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M7);
            dst += i_dst;
            M7 = _mm_srli_si128(M7, 4);
            *((int*)dst) = _mm_cvtsi128_si32(M7);
        }
    }

}


