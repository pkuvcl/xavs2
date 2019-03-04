/*
 * intrinsic_intra-pred_avx2.c
 *
 * Description of this file:
 *    AVX2 assembly functions of Intra-Prediction module of the xavs2 library
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
#include "avs2_defs.h"
#include "intrinsic.h"

#ifndef _MSC_VER
#define __int64 int64_t
#endif

void intra_pred_ver_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    if (bsx <= 8 && bsy <= 8) {
        // 当block_size小于8时avx2比sse慢
        intra_pred_ver_sse128(src, dst, i_dst, dir_mode, bsx, bsy);
        return;
    }
    pel_t *rsrc = src + 1;
    int i;

    __m256i S1;
    if (bsx >= 32) {
        for (i = 0; i < bsy; i++) {
            S1 = _mm256_loadu_si256((const __m256i*)(rsrc));//32
            _mm256_storeu_si256((__m256i*)(dst), S1);

            if (32 < bsx) {
                S1 = _mm256_loadu_si256((const __m256i*)(rsrc + 32));//64
                _mm256_storeu_si256((__m256i*)(dst + 32), S1);
            }
            dst += i_dst;
        }
    } else {
        int j;
        __m128i S1_;
        if (bsx & 15) {//4/8
            __m128i mask = _mm_load_si128((const __m128i*)intrinsic_mask[(bsx & 15) - 1]);
            for (i = 0; i < bsy; i++) {
                for (j = 0; j < bsx - 15; j += 16) {
                    S1_ = _mm_loadu_si128((const __m128i*)(rsrc + j));
                    _mm_storeu_si128((__m128i*)(dst + j), S1_);
                }
                S1_ = _mm_loadu_si128((const __m128i*)(rsrc + j));
                _mm_maskmoveu_si128(S1_, mask, (char *)&dst[j]);
                dst += i_dst;
            }
        }
        /*{//4/8
        for (i = 0; i < bsy; i++) {
        for (j = 0; j < bsx; j += 4) {
        S1 = _mm_loadu_si128((const __m128i*)(rsrc + j));
        _mm_storeu_si128((__m128i*)(dst + j), S1);
        }
        dst += i_dst;
        }
        }*/
        else {
            for (i = 0; i < bsy; i++) {//16
                S1_ = _mm_loadu_si128((const __m128i*)rsrc);
                _mm_storeu_si128((__m128i*)dst, S1_);
                dst += i_dst;
            }
        }
    }

}

void intra_pred_hor_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    if (bsx <= 8 && bsy <= 8) {
        // 当block_size小于8时avx2比sse慢
        intra_pred_hor_sse128(src, dst, i_dst, dir_mode, bsx, bsy);
        return;
    }
    int i;
    pel_t *rsrc = src - 1;
    __m256i S1;

    if (bsx >= 32) {
        for (i = 0; i < bsy; i++) {
            S1 = _mm256_set1_epi8((char)rsrc[-i]);//32
            _mm256_storeu_si256((__m256i*)(dst), S1);

            if (32 < bsx) {//64
                _mm256_storeu_si256((__m256i*)(dst + 32), S1);
            }
            dst += i_dst;
        }
    } else {
        int j;
        __m128i S1_;
        if (bsx & 15) {//4/8
            __m128i mask = _mm_load_si128((const __m128i*)intrinsic_mask[(bsx & 15) - 1]);
            for (i = 0; i < bsy; i++) {
                for (j = 0; j < bsx - 15; j += 16) {
                    S1_ = _mm_set1_epi8((char)rsrc[-i]);
                    _mm_storeu_si128((__m128i*)(dst + j), S1_);
                }
                S1_ = _mm_set1_epi8((char)rsrc[-i]);
                _mm_maskmoveu_si128(S1_, mask, (char*)&dst[j]);
                dst += i_dst;
            }
        } else {
            for (i = 0; i < bsy; i++) {//16
                S1_ = _mm_set1_epi8((char)rsrc[-i]);
                _mm_storeu_si128((__m128i*)dst, S1_);
                dst += i_dst;
            }
        }
    }
}

void intra_pred_dc_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    if (bsx <= 8 && bsy <= 8) {
        // 当block_size小于8时avx2比sse慢
        intra_pred_dc_sse128(src, dst, i_dst, dir_mode, bsx, bsy);
        return;
    }
    int bAboveAvail = dir_mode >> 8;
    int bLeftAvail = dir_mode & 0xFF;
    int   x, y;
    int   iDCValue = 0;
    pel_t  *rsrc = src - 1;
    __m256i S1;
    int i;
    if (bLeftAvail) {
        for (y = 0; y < bsy; y++) {
            iDCValue += rsrc[-y];
        }

        rsrc = src + 1;
        if (bAboveAvail) {
            for (x = 0; x < bsx; x++) {
                iDCValue += rsrc[x];
            }

            iDCValue += ((bsx + bsy) >> 1);
            iDCValue = (iDCValue * (512 / (bsx + bsy))) >> 9;
        } else {
            iDCValue += bsy / 2;
            iDCValue /= bsy;
        }
    } else {
        rsrc = src + 1;
        if (bAboveAvail) {
            for (x = 0; x < bsx; x++) {
                iDCValue += rsrc[x];
            }

            iDCValue += bsx / 2;
            iDCValue /= bsx;
        } else {
            iDCValue = g_dc_value;
        }
    }
    /*
    for (y = 0; y < bsy; y++) {
    for (x = 0; x < bsx; x++) {
    dst[x] = iDCValue;
    }
    dst += i_dst;
    }
    */

    S1 = _mm256_set1_epi8((char)iDCValue);
    if (bsx >= 32) {
        for (i = 0; i < bsy; i++) {
            _mm256_storeu_si256((__m256i*)(dst), S1);//32
            if (32 < bsx) {//64
                _mm256_storeu_si256((__m256i*)(dst + 32), S1);
            }
            dst += i_dst;
        }
    } else {
        __m128i S1_;
        int j;
        S1_ = _mm_set1_epi8((char)iDCValue);
        if (bsx & 15) {//4/8
            __m128i mask = _mm_load_si128((const __m128i*)intrinsic_mask[(bsx & 15) - 1]);
            for (i = 0; i < bsy; i++) {
                for (j = 0; j < bsx - 15; j += 16) {
                    _mm_storeu_si128((__m128i*)(dst + j), S1_);
                }
                _mm_maskmoveu_si128(S1_, mask, (char*)&dst[j]);
                dst += i_dst;
            }
        } else {
            for (i = 0; i < bsy; i++) {//16
                _mm_storeu_si128((__m128i*)dst, S1_);
                dst += i_dst;
            }
        }
    }

}

void intra_pred_plane_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
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
    int im_h = ib_mult [tab_log2[bsx] - 2];
    int is_h = ib_shift[tab_log2[bsx] - 2];
    int im_v = ib_mult [tab_log2[bsy] - 2];
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

    __m256i TC, TB, TA, T_Start, T, D, D1;
    __m256i mask ;

    TA = _mm256_set1_epi16((int16_t)iTmp);
    TB = _mm256_set1_epi16((int16_t)iB);
    TC = _mm256_set1_epi16((int16_t)iC);

    T_Start = _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    T_Start = _mm256_mullo_epi16(TB, T_Start);
    T_Start = _mm256_add_epi16(T_Start, TA);

    TB = _mm256_mullo_epi16(TB, _mm256_set1_epi16(16));

    if (bsx == 4) {
        mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[3]);
        for (y = 0; y < bsy; y++) {
            D = _mm256_srai_epi16(T_Start, 5);
            D = _mm256_packus_epi16(D, D);
            _mm256_maskstore_epi32((int*)dst, mask, D);
            T_Start = _mm256_add_epi16(T_Start, TC);
            dst += i_dst;
        }
    } else if (bsx == 8) {
        mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[7]);
        for (y = 0; y < bsy; y++) {
            D = _mm256_srai_epi16(T_Start, 5);
            D = _mm256_packus_epi16(D, D);
            _mm256_maskstore_epi64((__int64*)dst, mask, D);
            T_Start = _mm256_add_epi16(T_Start, TC);
            dst += i_dst;
        }
    } else if (bsx == 16) {
        mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[15]);
        for (y = 0; y < bsy; y++) {
            D = _mm256_srai_epi16(T_Start, 5);
            D = _mm256_packus_epi16(D, D);
            D = _mm256_permute4x64_epi64(D, 8);
            _mm256_maskstore_epi64((__int64*)dst, mask, D);
            T_Start = _mm256_add_epi16(T_Start, TC);
            dst += i_dst;
        }
    } else { //32 64
        for (y = 0; y < bsy; y++) {
            T = T_Start;
            for (x = 0; x < bsx; x += 32) {
                D = _mm256_srai_epi16(T, 5);
                T = _mm256_add_epi16(T, TB);
                D1 = _mm256_srai_epi16(T, 5);
                D = _mm256_packus_epi16(D, D1);
                D = _mm256_permute4x64_epi64(D, 0x00D8);
                _mm256_storeu_si256((__m256i*)(dst + x), D);

                T = _mm256_add_epi16(T, TB);
            }
            T_Start = _mm256_add_epi16(T_Start, TC);
            dst += i_dst;
        }
    }


}

void intra_pred_bilinear_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int x, y;
    int ishift_x = tab_log2[bsx];
    int ishift_y = tab_log2[bsy];
    int ishift = XAVS2_MIN(ishift_x, ishift_y);
    int ishift_xy = ishift_x + ishift_y + 1;
    int offset = 1 << (ishift_x + ishift_y);
    int a, b, c, t, val;
    pel_t *p;


    __m256i T, T1, T2, T3, C1, C2, ADD;

    ALIGN32(itr_t pTop[MAX_CU_SIZE + 32]);
    ALIGN32(itr_t pLeft[MAX_CU_SIZE + 32]);
    ALIGN32(itr_t pT[MAX_CU_SIZE + 32]);
    ALIGN32(itr_t pL[MAX_CU_SIZE + 32]);
    ALIGN32(itr_t wy[MAX_CU_SIZE + 32]);

    UNUSED_PARAMETER(dir_mode);

    p = src + 1;
    __m256i ZERO = _mm256_setzero_si256();
    for (x = 0; x < bsx; x += 32) {
        T = _mm256_loadu_si256((__m256i*)(p + x));//8bit 32个
        T1 = _mm256_unpacklo_epi8(T, ZERO); //0 2
        T2 = _mm256_unpackhi_epi8(T, ZERO); //1 3
        T = _mm256_permute2x128_si256(T1, T2, 0x0020);
        _mm256_store_si256((__m256i*)(pTop + x), T);
        T = _mm256_permute2x128_si256(T1, T2, 0x0031);
        _mm256_store_si256((__m256i*)(pTop + x + 16), T);
    }
    for (y = 0; y < bsy; y++) {
        pLeft[y] = src[-1 - y];
    }


    //p = src + 1;
    //for (x = 0; x < bsx; x++) {
    //    pTop[x] = p[x];
    //}
    //p = src - 1;
    //for (y = 0; y < bsy; y++) {
    //    pLeft[y] = p[-y];
    //}


    a = pTop[bsx - 1];
    b = pLeft[bsy - 1];

    if (bsx == bsy) {
        c = (a + b + 1) >> 1;
    } else {
        c = (((a << ishift_x) + (b << ishift_y)) * 13 + (1 << (ishift + 5))) >> (ishift + 6);
    }

    t = (c << 1) - a - b;

    T = _mm256_set1_epi16((int16_t)b);
    for (x = 0; x < bsx; x += 16) {
        T1 = _mm256_loadu_si256((__m256i*)(pTop + x));
        T2 = _mm256_sub_epi16(T, T1);
        T1 = _mm256_slli_epi16(T1, ishift_y);
        _mm256_store_si256((__m256i*)(pT + x), T2);
        _mm256_store_si256((__m256i*)(pTop + x), T1);
    }

    T = _mm256_set1_epi16((int16_t)a);
    for (y = 0; y < bsy; y += 16) {
        T1 = _mm256_loadu_si256((__m256i*)(pLeft + y));
        T2 = _mm256_sub_epi16(T, T1);
        T1 = _mm256_slli_epi16(T1, ishift_x);
        _mm256_store_si256((__m256i*)(pL + y), T2);
        _mm256_store_si256((__m256i*)(pLeft + y), T1);
    }

    T = _mm256_set1_epi16((int16_t)t);
    T = _mm256_mullo_epi16(T, _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
    T1 = _mm256_set1_epi16((int16_t)(16 * t));

    for (y = 0; y < bsy; y += 16) {
        _mm256_store_si256((__m256i*)(wy + y), T);
        T = _mm256_add_epi16(T, T1);
    }

    C1 = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    C2 = _mm256_set1_epi32(8);

    if (bsx == 4) {
        __m256i pTT = _mm256_loadu_si256((__m256i*)pT);
        T = _mm256_loadu_si256((__m256i*)pTop);
        __m256i mask = _mm256_loadu_si256((__m256i*)intrinsic_mask_256_8bit[3]);
        for (y = 0; y < bsy; y++) {
            int add = (pL[y] << ishift_y) + wy[y];
            ADD = _mm256_set1_epi32(add);
            ADD = _mm256_mullo_epi32(C1, ADD);
            val = (pLeft[y] << ishift_y) + offset + (pL[y] << ishift_y);
            ADD = _mm256_add_epi32(ADD, _mm256_set1_epi32(val));

            T = _mm256_add_epi16(T, pTT);
            T1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(T, 0));
            T1 = _mm256_slli_epi32(T1, ishift_x);

            T1 = _mm256_add_epi32(T1, ADD);
            T1 = _mm256_srai_epi32(T1, ishift_xy);

            T1 = _mm256_packus_epi32(T1, T1);
            T1 = _mm256_packus_epi16(T1, T1);

            _mm256_maskstore_epi32((int*)dst, mask, T1);

            dst += i_dst;
        }
    } else if (bsx == 8) {
        __m256i pTT = _mm256_load_si256((__m256i*)pT);
        T = _mm256_load_si256((__m256i*)pTop);
        __m256i mask = _mm256_loadu_si256((__m256i*)intrinsic_mask_256_8bit[7]);
        for (y = 0; y < bsy; y++) {
            int add = (pL[y] << ishift_y) + wy[y];
            ADD = _mm256_set1_epi32(add);
            ADD = _mm256_mullo_epi32(C1, ADD);
            val = (pLeft[y] << ishift_y) + offset + (pL[y] << ishift_y);
            ADD = _mm256_add_epi32(ADD, _mm256_set1_epi32(val));

            T = _mm256_add_epi16(T, pTT);
            T1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(T, 0));
            T1 = _mm256_slli_epi32(T1, ishift_x);

            T1 = _mm256_add_epi32(T1, ADD);
            T1 = _mm256_srai_epi32(T1, ishift_xy);

            //mask

            //T1 is the result
            T1 = _mm256_packus_epi32(T1, T1); //1 2 3 4 1 2 3 4 5 6 7 8 5 6 7 8
            T1 = _mm256_permute4x64_epi64(T1, 0x0008);
            T1 = _mm256_packus_epi16(T1, T1);

            _mm256_maskstore_epi64((__int64*)dst, mask, T1);

            dst += i_dst;
        }
    } else {
        __m256i TT[8];
        __m256i PTT[8];
        __m256i temp1, temp2;
        __m256i mask1 = _mm256_set_epi32(3, 2, 1, 0, 5, 1, 4, 0);
        __m256i mask2 = _mm256_loadu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
        for (x = 0; x < bsx; x += 16) {
            int idx = x >> 3;
            __m256i M0 = _mm256_loadu_si256((__m256i*)(pTop + x)); //0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
            __m256i M1 = _mm256_loadu_si256((__m256i*)(pT + x));
            temp1 = _mm256_unpacklo_epi16(M0, ZERO); //0 1 2 3   8  9 10 11
            temp2 = _mm256_unpackhi_epi16(M0, ZERO); //4 5 6 7  12 13 14 15
            TT[idx]      = _mm256_permute2x128_si256(temp1, temp2, 0x0020); //0 1 2 3 4 5 6 7
            TT[idx + 1]  = _mm256_permute2x128_si256(temp1, temp2, 0x0031); //8 9 10 11 12 13 14 15

            PTT[idx]     = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(M1, 0));
            PTT[idx + 1] = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(M1, 1));
        }
        for (y = 0; y < bsy; y++) {
            int add = (pL[y] << ishift_y) + wy[y];
            ADD = _mm256_set1_epi32(add);
            T3 = _mm256_mullo_epi32(C2, ADD);
            ADD = _mm256_mullo_epi32(C1, ADD);

            val = (pLeft[y] << ishift_y) + offset + (pL[y] << ishift_y);

            ADD = _mm256_add_epi32(ADD, _mm256_set1_epi32(val));

            for (x = 0; x < bsx; x += 16) {
                int idx = x >> 3;
                TT[idx] = _mm256_add_epi32(TT[idx], PTT[idx]); //0 1 2 3 4 5 6 7
                TT[idx + 1] = _mm256_add_epi32(TT[idx + 1], PTT[idx + 1]); //8 9 10 11 12 13 14 15

                T1 = _mm256_slli_epi32(TT[idx], ishift_x);
                T2 = _mm256_slli_epi32(TT[idx + 1], ishift_x);

                T1 = _mm256_add_epi32(T1, ADD);
                T1 = _mm256_srai_epi32(T1, ishift_xy);//0 1 2 3 4 5 6 7

                ADD = _mm256_add_epi32(ADD, T3);
                T2 = _mm256_add_epi32(T2, ADD);
                T2 = _mm256_srai_epi32(T2, ishift_xy);//8 9 10 11 12 13 14 15

                //T1 T2 is the result
                T1 = _mm256_packus_epi32(T1, T2); //0 1 2 3 8 9 10 11 4 5 6 7 12 13 14 15
                T1 = _mm256_packus_epi16(T1, T1); //0 1 2 3 8 9 10 11 0 1 2 3 8 9 10 11     4 5 6 7 12 13 14 15 4 5 6 7 12 13 14 15
                T1 = _mm256_permutevar8x32_epi32(T1, mask1);

                //store 128 bits
                _mm256_maskstore_epi64((__int64*)(dst + x), mask2, T1);

                ADD = _mm256_add_epi32(ADD, T3);
            }
            dst += i_dst;
        }
    }

}

void intra_pred_ang_x_3_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{

    pel_t *dst1 = dst;
    pel_t *dst2 = dst1 + i_dst;
    pel_t *dst3 = dst2 + i_dst;
    pel_t *dst4 = dst3 + i_dst;

    UNUSED_PARAMETER(dir_mode);

    if ((bsy > 4) && (bsx > 8)) {

        __m256i coeff2 = _mm256_set1_epi16(2);
        __m256i coeff3 = _mm256_set1_epi16(3);
        __m256i coeff4 = _mm256_set1_epi16(4);
        __m256i coeff5 = _mm256_set1_epi16(5);
        __m256i coeff7 = _mm256_set1_epi16(7);
        __m256i coeff8 = _mm256_set1_epi16(8);

        ALIGN32(pel_t first_line[(64 + 176 + 16) << 2]);
        int line_size = bsx + (((bsy - 4) * 11) >> 2);
        int aligned_line_size = 64 + 176 + 16;
        int i;
        pel_t *pfirst[4];

        pfirst[0] = first_line;
        pfirst[1] = pfirst[0] + aligned_line_size;
        pfirst[2] = pfirst[1] + aligned_line_size;
        pfirst[3] = pfirst[2] + aligned_line_size;

        __m256i p00, p10, p20, p30;
        __m256i p01, p11, p21, p31;

        __m256i SS2, SS11;
        __m256i L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, L13;
        __m256i H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13;

        for (i = 0; i < line_size - 16; i += 32, src += 32) {

            SS2 = _mm256_loadu_si256((__m256i*)(src + 2));//2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
            L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));//2...17
            H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 1));//18...34
            SS2 = _mm256_loadu_si256((__m256i*)(src + 3));//3 4 5 6 7 8 9 10 11 12 13 14 15
            L3  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));//3...18
            H3  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 1));//19...35
            SS2 = _mm256_loadu_si256((__m256i*)(src + 4));//4 5 6 7 8 9 10 11 12 13 14 15
            L4  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));//4
            H4  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 1));//20
            SS2  = _mm256_loadu_si256((__m256i*)(src + 5));//5 6 7 8 9 10 11 12 13 14 15
            L5  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));//5
            H5  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 1));//21
            SS2 = _mm256_loadu_si256((__m256i*)(src + 6));//6 7 8 9 10 11 12 13 14 15
            L6  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));//6
            H6  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 1));//22
            SS2 = _mm256_loadu_si256((__m256i*)(src + 7));//7 8 9 10 11 12 13 14 15
            L7  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
            H7  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 1));
            SS2 = _mm256_loadu_si256((__m256i*)(src + 8));//8 9 10 11 12 13 14 15
            L8  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
            H8  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 1));
            SS2 = _mm256_loadu_si256((__m256i*)(src + 9));//9 10 11 12 13 14 15
            L9  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
            H9  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 1));
            SS2 = _mm256_loadu_si256((__m256i*)(src + 10));//10 11 12 13 14 15
            L10 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
            H10 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 1));
            SS2 = _mm256_loadu_si256((__m256i*)(src + 11));//11 12 13 14 15 16 17 18 19 20 21 22 23
            L11 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
            H11 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 1));
            SS2 = _mm256_loadu_si256((__m256i*)(src + 12));//12 13 14 15 16 17 18 19 20...
            L12 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
            H12 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 1));
            SS2 = _mm256_loadu_si256((__m256i*)(src + 13));//13 ...28 29...44
            L13 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
            H13 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 1));

            p00 = _mm256_add_epi16(L2, coeff8);//2 ...17
            p10 = _mm256_mullo_epi16(L3, coeff5);
            p20 = _mm256_mullo_epi16(L4, coeff7);
            p30 = _mm256_mullo_epi16(L5, coeff3);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 4);

            p01 = _mm256_add_epi16(H2, coeff8);
            p11 = _mm256_mullo_epi16(H3, coeff5);
            p21 = _mm256_mullo_epi16(H4, coeff7);
            p31 = _mm256_mullo_epi16(H5, coeff3);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_srli_epi16(p01, 4);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[0][i], p00);

            p00 = _mm256_add_epi16(L5, L8);
            p10 = _mm256_add_epi16(L6, L7);
            p10 = _mm256_mullo_epi16(p10, coeff3);
            p00 = _mm256_add_epi16(p00, coeff4);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_srli_epi16(p00, 3);

            p01 = _mm256_add_epi16(H5, H8);
            p11 = _mm256_add_epi16(H6, H7);
            p11 = _mm256_mullo_epi16(p11, coeff3);
            p01 = _mm256_add_epi16(p01, coeff4);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_srli_epi16(p01, 3);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[1][i], p00);

            p00 = _mm256_mullo_epi16(L8, coeff3);
            p10 = _mm256_mullo_epi16(L9, coeff7);
            p20 = _mm256_mullo_epi16(L10, coeff5);
            p30 = _mm256_add_epi16(L11, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 4);

            p01 = _mm256_mullo_epi16(H8, coeff3);
            p11 = _mm256_mullo_epi16(H9, coeff7);
            p21 = _mm256_mullo_epi16(H10, coeff5);
            p31 = _mm256_add_epi16(H11, coeff8);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_srli_epi16(p01, 4);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[2][i], p00);

            p00 = _mm256_add_epi16(L11, L13);
            p10 = _mm256_mullo_epi16(L12, coeff2);
            p00 = _mm256_add_epi16(p00, coeff2);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_srli_epi16(p00, 2);

            p01 = _mm256_add_epi16(H11, H13);
            p11 = _mm256_mullo_epi16(H12, coeff2);
            p01 = _mm256_add_epi16(p01, coeff2);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_srli_epi16(p01, 2);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[3][i], p00);
        }

        if (i < line_size) {
            SS2 = _mm256_loadu_si256((__m256i*)(src + 2));
            L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
            SS2 = _mm256_loadu_si256((__m256i*)(src + 3));
            L3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
            SS2 = _mm256_loadu_si256((__m256i*)(src + 4));
            L4 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
            SS2 = _mm256_loadu_si256((__m256i*)(src + 5));
            L5 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
            SS2 = _mm256_loadu_si256((__m256i*)(src + 6));
            L6 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
            SS2 = _mm256_loadu_si256((__m256i*)(src + 7));
            L7 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
            SS2 = _mm256_loadu_si256((__m256i*)(src + 8));
            L8 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
            SS2 = _mm256_loadu_si256((__m256i*)(src + 9));
            L9 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
            SS2 = _mm256_loadu_si256((__m256i*)(src + 10));
            L10 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));

            SS11 = _mm256_loadu_si256((__m256i*)(src + 11));
            L11 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS11, 0));
            SS11 = _mm256_loadu_si256((__m256i*)(src + 12));
            L12 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS11, 0));
            SS11 = _mm256_loadu_si256((__m256i*)(src + 13));
            L13 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS11, 0));

            p00 = _mm256_add_epi16(L2, coeff8);
            p10 = _mm256_mullo_epi16(L3, coeff5);
            p20 = _mm256_mullo_epi16(L4, coeff7);
            p30 = _mm256_mullo_epi16(L5, coeff3);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 4);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            __m256i mask = _mm256_loadu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
            _mm256_maskstore_epi64((__int64*)&pfirst[0][i], mask, p00);

            p00 = _mm256_add_epi16(L5, L8);
            p10 = _mm256_add_epi16(L6, L7);
            p10 = _mm256_mullo_epi16(p10, coeff3);
            p00 = _mm256_add_epi16(p00, coeff4);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_srli_epi16(p00, 3);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            _mm256_maskstore_epi64((__int64*)&pfirst[1][i], mask, p00);

            p00 = _mm256_mullo_epi16(L8, coeff3);
            p10 = _mm256_mullo_epi16(L9, coeff7);
            p20 = _mm256_mullo_epi16(L10, coeff5);
            p30 = _mm256_add_epi16(L11, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 4);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            _mm256_maskstore_epi64((__int64*)&pfirst[2][i], mask, p00);

            p00 = _mm256_add_epi16(L11, L13);
            p10 = _mm256_mullo_epi16(L12, coeff2);
            p00 = _mm256_add_epi16(p00, coeff2);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_srli_epi16(p00, 2);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            _mm256_maskstore_epi64((__int64*)&pfirst[3][i], mask, p00);
        }

        bsy >>= 2;
        __m256i M;
        if (bsx == 64) {
            for (i = 0; i < bsy; i++) {
                M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i * 11));
                _mm256_storeu_si256((__m256i*)dst1, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i * 11 + 32));
                _mm256_storeu_si256((__m256i*)(dst1 + 32), M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i * 11));
                _mm256_storeu_si256((__m256i*)dst2, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i * 11 + 32));
                _mm256_storeu_si256((__m256i*)(dst2 + 32), M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] + i * 11));
                _mm256_storeu_si256((__m256i*)dst3, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] + i * 11 + 32));
                _mm256_storeu_si256((__m256i*)(dst3 + 32), M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] + i * 11));
                _mm256_storeu_si256((__m256i*)dst4, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] + i * 11 + 32));
                _mm256_storeu_si256((__m256i*)(dst4 + 32), M);

                dst1 = dst4 + i_dst;
                dst2 = dst1 + i_dst;
                dst3 = dst2 + i_dst;
                dst4 = dst3 + i_dst;
            }
        } else if (bsx == 32) {
            for (i = 0; i < bsy; i++) {
                M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i * 11));
                _mm256_storeu_si256((__m256i*)dst1, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i * 11));
                _mm256_storeu_si256((__m256i*)dst2, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] + i * 11));
                _mm256_storeu_si256((__m256i*)dst3, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] + i * 11));
                _mm256_storeu_si256((__m256i*)dst4, M);

                dst1 = dst4 + i_dst;
                dst2 = dst1 + i_dst;
                dst3 = dst2 + i_dst;
                dst4 = dst3 + i_dst;
            }
        } else {
            __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
            for (i = 0; i < bsy; i++) {
                M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i * 11));
                _mm256_maskstore_epi64((__int64*)dst1, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i * 11));
                _mm256_maskstore_epi64((__int64*)dst2, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] + i * 11));
                _mm256_maskstore_epi64((__int64*)dst3, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] + i * 11));
                _mm256_maskstore_epi64((__int64*)dst4, mask, M);

                dst1 = dst4 + i_dst;
                dst2 = dst1 + i_dst;
                dst3 = dst2 + i_dst;
                dst4 = dst3 + i_dst;
            }
        }

        /*for (i = 0; i < bsy; i++) {
            memcpy(dst1, pfirst[0] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst2, pfirst[1] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst3, pfirst[2] + i * 11, bsx * sizeof(pel_t));
            memcpy(dst4, pfirst[3] + i * 11, bsx * sizeof(pel_t));
            dst1 = dst4 + i_dst;
            dst2 = dst1 + i_dst;
            dst3 = dst2 + i_dst;
            dst4 = dst3 + i_dst;
        }*/
    } else if (bsx == 16) {

        __m256i coeff2 = _mm256_set1_epi16(2);
        __m256i coeff3 = _mm256_set1_epi16(3);
        __m256i coeff4 = _mm256_set1_epi16(4);
        __m256i coeff5 = _mm256_set1_epi16(5);
        __m256i coeff7 = _mm256_set1_epi16(7);
        __m256i coeff8 = _mm256_set1_epi16(8);

        __m256i p00, p10, p20, p30;
        __m256i SS2 = _mm256_loadu_si256((__m256i*)(src + 2));
        __m256i L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
        SS2 = _mm256_loadu_si256((__m256i*)(src + 3));
        __m256i L3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
        SS2 = _mm256_loadu_si256((__m256i*)(src + 4));
        __m256i L4 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
        SS2 = _mm256_loadu_si256((__m256i*)(src + 5));
        __m256i L5 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
        SS2 = _mm256_loadu_si256((__m256i*)(src + 6));
        __m256i L6 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
        SS2 = _mm256_loadu_si256((__m256i*)(src + 7));
        __m256i L7 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
        SS2 = _mm256_loadu_si256((__m256i*)(src + 8));
        __m256i L8 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
        SS2 = _mm256_loadu_si256((__m256i*)(src + 9));
        __m256i L9 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));
        SS2 = _mm256_loadu_si256((__m256i*)(src + 10));
        __m256i L10 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS2, 0));

        __m256i SS11 = _mm256_loadu_si256((__m256i*)(src + 11));
        __m256i L11 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS11, 0));
        SS11 = _mm256_loadu_si256((__m256i*)(src + 12));
        __m256i L12 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS11, 0));
        SS11 = _mm256_loadu_si256((__m256i*)(src + 13));
        __m256i L13 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS11, 0));

        p00 = _mm256_add_epi16(L2, coeff8);
        p10 = _mm256_mullo_epi16(L3, coeff5);
        p20 = _mm256_mullo_epi16(L4, coeff7);
        p30 = _mm256_mullo_epi16(L5, coeff3);
        p00 = _mm256_add_epi16(p00, p10);
        p00 = _mm256_add_epi16(p00, p20);
        p00 = _mm256_add_epi16(p00, p30);
        p00 = _mm256_srli_epi16(p00, 4);

        p00 = _mm256_packus_epi16(p00, p00);
        p00 = _mm256_permute4x64_epi64(p00, 0x0008);
        __m256i mask = _mm256_loadu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
        _mm256_maskstore_epi64((__int64*)dst1, mask, p00);

        p00 = _mm256_add_epi16(L5, L8);
        p10 = _mm256_add_epi16(L6, L7);
        p10 = _mm256_mullo_epi16(p10, coeff3);
        p00 = _mm256_add_epi16(p00, coeff4);
        p00 = _mm256_add_epi16(p00, p10);
        p00 = _mm256_srli_epi16(p00, 3);

        p00 = _mm256_packus_epi16(p00, p00);
        p00 = _mm256_permute4x64_epi64(p00, 0x0008);
        _mm256_maskstore_epi64((__int64*)dst2, mask, p00);

        p00 = _mm256_mullo_epi16(L8, coeff3);
        p10 = _mm256_mullo_epi16(L9, coeff7);
        p20 = _mm256_mullo_epi16(L10, coeff5);
        p30 = _mm256_add_epi16(L11, coeff8);
        p00 = _mm256_add_epi16(p00, p10);
        p00 = _mm256_add_epi16(p00, p20);
        p00 = _mm256_add_epi16(p00, p30);
        p00 = _mm256_srli_epi16(p00, 4);

        p00 = _mm256_packus_epi16(p00, p00);
        p00 = _mm256_permute4x64_epi64(p00, 0x0008);
        _mm256_maskstore_epi64((__int64*)dst3, mask, p00);

        p00 = _mm256_add_epi16(L11, L13);
        p10 = _mm256_mullo_epi16(L12, coeff2);
        p00 = _mm256_add_epi16(p00, coeff2);
        p00 = _mm256_add_epi16(p00, p10);
        p00 = _mm256_srli_epi16(p00, 2);

        p00 = _mm256_packus_epi16(p00, p00);
        p00 = _mm256_permute4x64_epi64(p00, 0x0008);
        _mm256_maskstore_epi64((__int64*)dst4, mask, p00);

    } else { //8x8 8x32 4x16 4x4

        intra_pred_ang_x_3_sse128(src, dst, i_dst, dir_mode, bsx, bsy);

    }

}

void intra_pred_ang_x_4_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    if (bsx != bsy && bsx < bsy) {
        intra_pred_ang_x_4_sse128(src, dst, i_dst, dir_mode, bsx, bsy);
        return;
    }
    ALIGN32(pel_t first_line[64 + 128]);
    int line_size = bsx + ((bsy - 1) << 1);

    int iHeight2 = bsy << 1;
    int i;
    __m256i zero = _mm256_setzero_si256();
    __m256i offset = _mm256_set1_epi16(2);

    UNUSED_PARAMETER(dir_mode);
    src += 3;

    for (i = 0; i < line_size - 16; i += 32, src += 32) {
        //0 1 2 3 .... 12 13 14 15    16 17 18 19 .... 28 29 30 21
        __m256i S0 = _mm256_loadu_si256((__m256i*)(src - 1));
        __m256i S1 = _mm256_loadu_si256((__m256i*)(src));
        __m256i S2 = _mm256_loadu_si256((__m256i*)(src + 1));

        __m256i L0 = _mm256_unpacklo_epi8(S0, zero);//0 1 2 3 4 5 6 7     16 17 18 19 20 21 22 23
        __m256i L1 = _mm256_unpacklo_epi8(S1, zero);
        __m256i L2 = _mm256_unpacklo_epi8(S2, zero);

        __m256i H0 = _mm256_unpackhi_epi8(S0, zero);//8 9 10 11 12 13 14 15     24 25 26 27 28 29 30 31
        __m256i H1 = _mm256_unpackhi_epi8(S1, zero);
        __m256i H2 = _mm256_unpackhi_epi8(S2, zero);

        __m256i tmp0 = _mm256_permute2x128_si256(L0, H0, 0x0020);//0 1 2 3 4 5 6 7   8 9 10 11 12 13 14 15
        __m256i tmp1 = _mm256_permute2x128_si256(L1, H1, 0x0020);
        __m256i tmp2 = _mm256_permute2x128_si256(L2, H2, 0x0020);
        __m256i sum1 = _mm256_add_epi16(tmp0, tmp1);
        __m256i sum2 = _mm256_add_epi16(tmp1, tmp2);


        tmp0 = _mm256_permute2x128_si256(L0, H0, 0x0031);//16 17...24 25...
        tmp1 = _mm256_permute2x128_si256(L1, H1, 0x0031);
        tmp2 = _mm256_permute2x128_si256(L2, H2, 0x0031);
        __m256i sum3 = _mm256_add_epi16(tmp0, tmp1);
        __m256i sum4 = _mm256_add_epi16(tmp1, tmp2);

        sum1 = _mm256_add_epi16(sum1, sum2);
        sum3 = _mm256_add_epi16(sum3, sum4);

        sum1 = _mm256_add_epi16(sum1, offset);
        sum3 = _mm256_add_epi16(sum3, offset);

        sum1 = _mm256_srli_epi16(sum1, 2);
        sum3 = _mm256_srli_epi16(sum3, 2);

        sum1 = _mm256_packus_epi16(sum1, sum3);//0 2 1 3
        sum1 = _mm256_permute4x64_epi64(sum1, 0x00D8);
        _mm256_storeu_si256((__m256i*)&first_line[i], sum1);
    }

    if (i < line_size) {
        //0 1 2 3 .... 12 13 14 15    16 17 18 19 .... 28 29 30 21
        __m256i S0 = _mm256_loadu_si256((__m256i*)(src - 1));
        __m256i S2 = _mm256_loadu_si256((__m256i*)(src + 1));
        __m256i S1 = _mm256_loadu_si256((__m256i*)(src));
        S0 = _mm256_permute4x64_epi64(S0, 0x00D8);
        S2 = _mm256_permute4x64_epi64(S2, 0x00D8);
        S1 = _mm256_permute4x64_epi64(S1, 0x00D8);

        __m256i L0 = _mm256_unpacklo_epi8(S0, zero);
        __m256i L1 = _mm256_unpacklo_epi8(S1, zero);
        __m256i L2 = _mm256_unpacklo_epi8(S2, zero);

        __m256i sum1 = _mm256_add_epi16(L0, L1);
        __m256i sum2 = _mm256_add_epi16(L1, L2);

        sum1 = _mm256_add_epi16(sum1, sum2);
        sum1 = _mm256_add_epi16(sum1, offset);
        sum1 = _mm256_srli_epi16(sum1, 2);

        sum1 = _mm256_packus_epi16(sum1, sum1);
        sum1 = _mm256_permute4x64_epi64(sum1, 0x0008);
        //store 128 bit
        __m256i mask2 = _mm256_loadu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
        _mm256_maskstore_epi64((__int64*)(first_line + i), mask2, sum1);

        //_mm_storel_epi64((__m128i*)&first_line[i], sum1);
    }

    if (bsx == 64) {

        for (i = 0; i < iHeight2; i += 8) {
            __m256i M = _mm256_lddqu_si256((__m256i*)&first_line[i]);
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(&first_line[i]+32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 2]);
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(&first_line[i + 2] + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 4]);
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(&first_line[i + 4] + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 6]);
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(&first_line[i + 6] + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
        }
    } else if (bsx == 32) {
        for (i = 0; i < iHeight2; i += 8) {
            __m256i M = _mm256_lddqu_si256((__m256i*)&first_line[i]);
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 2]);
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 4]);
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 6]);
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
        }
    } else if (bsx == 16) {
        __m256i mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < iHeight2; i += 8) {
            __m256i M = _mm256_loadu_si256((__m256i*)&first_line[i]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[i + 2]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[i + 4]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[i + 6]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
        }
    } else if (bsx == 8) {
        __m256i mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < iHeight2; i += 8) {
            __m256i M = _mm256_loadu_si256((__m256i*)&first_line[i]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_srli_si256(M, 2);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_srli_si256(M, 2);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_srli_si256(M, 2);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
        }
    } else {
        __m256i mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < iHeight2; i += 8) {
            __m256i M = _mm256_loadu_si256((__m256i*)&first_line[i]);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_srli_si256(M, 2);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_srli_si256(M, 2);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_srli_si256(M, 2);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
        }
    }

    /*if (bsx == bsy || bsx >= 16) {
        for (i = 0; i < iHeight2; i += 2) {
            memcpy(dst, first_line + i, bsx * sizeof(pel_t));
            dst += i_dst;
        }
    } else {
        __m256i mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < iHeight2; i += 8) {
            __m256i M = _mm256_loadu_si256((__m256i*)&first_line[i]);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_srli_si256(M, 2);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_srli_si256(M, 2);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_srli_si256(M, 2);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
        }
    }*/

}

void intra_pred_ang_x_5_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    __m256i coeff2  = _mm256_set1_epi16(2);
    __m256i coeff3  = _mm256_set1_epi16(3);
    __m256i coeff4  = _mm256_set1_epi16(4);
    __m256i coeff5  = _mm256_set1_epi16(5);
    __m256i coeff7  = _mm256_set1_epi16(7);
    __m256i coeff8  = _mm256_set1_epi16(8);
    __m256i coeff9  = _mm256_set1_epi16(9);
    __m256i coeff11 = _mm256_set1_epi16(11);
    __m256i coeff13 = _mm256_set1_epi16(13);
    __m256i coeff15 = _mm256_set1_epi16(15);
    __m256i coeff16 = _mm256_set1_epi16(16);

    UNUSED_PARAMETER(dir_mode);

    int i;
    if (((bsy > 4) && (bsx > 8))) {
        ALIGN32(pel_t first_line[(64 + 80 + 16) << 3]);
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

        __m256i p00, p10, p20, p30;
        __m256i p01, p11, p21, p31;

        __m256i SS1;
        __m256i L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, L13;
        __m256i H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13;


        for (i = 0; i < line_size - 16; i += 32, src += 32) {
            SS1 = _mm256_loadu_si256((__m256i*)(src + 1));//1...8 9...16 17..24 25..32
            L1  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));//1
            H1  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 1));//17
            SS1 = _mm256_loadu_si256((__m256i*)(src + 2));
            L2  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));//2
            H2  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 1));//18
            SS1 = _mm256_loadu_si256((__m256i*)(src + 3));
            L3  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));//3
            H3  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 1));//19
            SS1 = _mm256_loadu_si256((__m256i*)(src + 4));
            L4  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));//4
            H4  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 1));//20
            SS1 = _mm256_loadu_si256((__m256i*)(src + 5));
            L5  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
            H5  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 1));
            SS1 = _mm256_loadu_si256((__m256i*)(src + 6));
            L6  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
            H6  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 1));
            SS1 = _mm256_loadu_si256((__m256i*)(src + 7));
            L7  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
            H7  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 1));
            SS1 = _mm256_loadu_si256((__m256i*)(src + 8));
            L8  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
            H8  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 1));
            SS1 = _mm256_loadu_si256((__m256i*)(src + 9));
            L9  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
            H9  = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 1));
            SS1 = _mm256_loadu_si256((__m256i*)(src + 10));
            L10 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
            H10 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 1));
            SS1 = _mm256_loadu_si256((__m256i*)(src + 11));
            L11 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
            H11 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 1));
            SS1 = _mm256_loadu_si256((__m256i*)(src + 12));
            L12 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
            H12 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 1));
            SS1 = _mm256_loadu_si256((__m256i*)(src + 13));
            L13 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
            H13 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 1));

            p00 = _mm256_mullo_epi16(L1, coeff5);
            p10 = _mm256_mullo_epi16(L2, coeff13);
            p20 = _mm256_mullo_epi16(L3, coeff11);
            p30 = _mm256_mullo_epi16(L4, coeff3);
            p00 = _mm256_add_epi16(p00, coeff16);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 5);

            p01 = _mm256_mullo_epi16(H1, coeff5);
            p11 = _mm256_mullo_epi16(H2, coeff13);
            p21 = _mm256_mullo_epi16(H3, coeff11);
            p31 = _mm256_mullo_epi16(H4, coeff3);
            p01 = _mm256_add_epi16(p01, coeff16);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_srli_epi16(p01, 5);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[0][i], p00);

            p10 = _mm256_mullo_epi16(L3, coeff5);
            p20 = _mm256_mullo_epi16(L4, coeff7);
            p30 = _mm256_mullo_epi16(L5, coeff3);
            p00 = _mm256_add_epi16(L2, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 4);

            p11 = _mm256_mullo_epi16(H3, coeff5);
            p21 = _mm256_mullo_epi16(H4, coeff7);
            p31 = _mm256_mullo_epi16(H5, coeff3);
            p01 = _mm256_add_epi16(H2, coeff8);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_srli_epi16(p01, 4);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[1][i], p00);

            p00 = _mm256_mullo_epi16(L4, coeff7);
            p10 = _mm256_mullo_epi16(L5, coeff15);
            p20 = _mm256_mullo_epi16(L6, coeff9);
            p30 = _mm256_add_epi16(L7, coeff16);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 5);

            p01 = _mm256_mullo_epi16(H4, coeff7);
            p11 = _mm256_mullo_epi16(H5, coeff15);
            p21 = _mm256_mullo_epi16(H6, coeff9);
            p31 = _mm256_add_epi16(H7, coeff16);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_srli_epi16(p01, 5);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[2][i], p00);

            p00 = _mm256_add_epi16(L5, L8);
            p10 = _mm256_add_epi16(L6, L7);
            p10 = _mm256_mullo_epi16(p10, coeff3);
            p00 = _mm256_add_epi16(p00, coeff4);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_srli_epi16(p00, 3);

            p01 = _mm256_add_epi16(H5, H8);
            p11 = _mm256_add_epi16(H6, H7);
            p11 = _mm256_mullo_epi16(p11, coeff3);
            p01 = _mm256_add_epi16(p01, coeff4);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_srli_epi16(p01, 3);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[3][i], p00);

            p00 = _mm256_add_epi16(L6, coeff16);
            p10 = _mm256_mullo_epi16(L7, coeff9);
            p20 = _mm256_mullo_epi16(L8, coeff15);
            p30 = _mm256_mullo_epi16(L9, coeff7);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 5);

            p01 = _mm256_add_epi16(H6, coeff16);
            p11 = _mm256_mullo_epi16(H7, coeff9);
            p21 = _mm256_mullo_epi16(H8, coeff15);
            p31 = _mm256_mullo_epi16(H9, coeff7);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_srli_epi16(p01, 5);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[4][i], p00);

            p00 = _mm256_mullo_epi16(L8, coeff3);
            p10 = _mm256_mullo_epi16(L9, coeff7);
            p20 = _mm256_mullo_epi16(L10, coeff5);
            p30 = _mm256_add_epi16(L11, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 4);

            p01 = _mm256_mullo_epi16(H8, coeff3);
            p11 = _mm256_mullo_epi16(H9, coeff7);
            p21 = _mm256_mullo_epi16(H10, coeff5);
            p31 = _mm256_add_epi16(H11, coeff8);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_srli_epi16(p01, 4);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[5][i], p00);

            p00 = _mm256_mullo_epi16(L9, coeff3);
            p10 = _mm256_mullo_epi16(L10, coeff11);
            p20 = _mm256_mullo_epi16(L11, coeff13);
            p30 = _mm256_mullo_epi16(L12, coeff5);
            p00 = _mm256_add_epi16(p00, coeff16);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 5);

            p01 = _mm256_mullo_epi16(H9, coeff3);
            p11 = _mm256_mullo_epi16(H10, coeff11);
            p21 = _mm256_mullo_epi16(H11, coeff13);
            p31 = _mm256_mullo_epi16(H12, coeff5);
            p01 = _mm256_add_epi16(p01, coeff16);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_srli_epi16(p01, 5);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[6][i], p00);

            p00 = _mm256_add_epi16(L11, L13);
            p10 = _mm256_add_epi16(L12, L12);
            p00 = _mm256_add_epi16(p00, coeff2);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_srli_epi16(p00, 2);

            p01 = _mm256_add_epi16(H11, H13);
            p11 = _mm256_add_epi16(H12, H12);
            p01 = _mm256_add_epi16(p01, coeff2);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_srli_epi16(p01, 2);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[7][i], p00);
        }
        if (i < line_size) {
            SS1 = _mm256_loadu_si256((__m256i*)(src + 1));//1...8 9...16 17..24 25..32
            L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));//1
            SS1 = _mm256_loadu_si256((__m256i*)(src + 2));
            L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));//2
            SS1 = _mm256_loadu_si256((__m256i*)(src + 3));
            L3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));//3
            SS1 = _mm256_loadu_si256((__m256i*)(src + 4));
            L4 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));//4
            SS1 = _mm256_loadu_si256((__m256i*)(src + 5));
            L5 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
            SS1 = _mm256_loadu_si256((__m256i*)(src + 6));
            L6 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
            SS1 = _mm256_loadu_si256((__m256i*)(src + 7));
            L7 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
            SS1 = _mm256_loadu_si256((__m256i*)(src + 8));
            L8 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
            SS1 = _mm256_loadu_si256((__m256i*)(src + 9));
            L9 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
            SS1 = _mm256_loadu_si256((__m256i*)(src + 10));
            L10 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
            SS1 = _mm256_loadu_si256((__m256i*)(src + 11));
            L11 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
            SS1 = _mm256_loadu_si256((__m256i*)(src + 12));
            L12 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
            SS1 = _mm256_loadu_si256((__m256i*)(src + 13));
            L13 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));

            p00 = _mm256_mullo_epi16(L1, coeff5);
            p10 = _mm256_mullo_epi16(L2, coeff13);
            p20 = _mm256_mullo_epi16(L3, coeff11);
            p30 = _mm256_mullo_epi16(L4, coeff3);
            p00 = _mm256_add_epi16(p00, coeff16);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 5);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            __m256i mask = _mm256_loadu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
            _mm256_maskstore_epi64((__int64*)&pfirst[0][i], mask, p00);

            p10 = _mm256_mullo_epi16(L3, coeff5);
            p20 = _mm256_mullo_epi16(L4, coeff7);
            p30 = _mm256_mullo_epi16(L5, coeff3);
            p00 = _mm256_add_epi16(L2, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 4);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            _mm256_maskstore_epi64((__int64*)&pfirst[1][i], mask, p00);

            p00 = _mm256_mullo_epi16(L4, coeff7);
            p10 = _mm256_mullo_epi16(L5, coeff15);
            p20 = _mm256_mullo_epi16(L6, coeff9);
            p30 = _mm256_add_epi16(L7, coeff16);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 5);

            p00 = _mm256_packus_epi16(p00, p00);
            _mm256_maskstore_epi64((__int64*)&pfirst[2][i], mask, p00);

            p00 = _mm256_add_epi16(L5, L8);
            p10 = _mm256_add_epi16(L6, L7);
            p10 = _mm256_mullo_epi16(p10, coeff3);
            p00 = _mm256_add_epi16(p00, coeff4);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_srli_epi16(p00, 3);

            p00 = _mm256_packus_epi16(p00, p00);
            _mm256_maskstore_epi64((__int64*)&pfirst[3][i], mask, p00);

            p00 = _mm256_add_epi16(L6, coeff16);
            p10 = _mm256_mullo_epi16(L7, coeff9);
            p20 = _mm256_mullo_epi16(L8, coeff15);
            p30 = _mm256_mullo_epi16(L9, coeff7);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 5);

            p00 = _mm256_packus_epi16(p00, p00);
            _mm256_maskstore_epi64((__int64*)&pfirst[4][i], mask, p00);

            p00 = _mm256_mullo_epi16(L8, coeff3);
            p10 = _mm256_mullo_epi16(L9, coeff7);
            p20 = _mm256_mullo_epi16(L10, coeff5);
            p30 = _mm256_add_epi16(L11, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 4);

            p00 = _mm256_packus_epi16(p00, p00);
            _mm256_maskstore_epi64((__int64*)&pfirst[5][i], mask, p00);

            p00 = _mm256_mullo_epi16(L9, coeff3);
            p10 = _mm256_mullo_epi16(L10, coeff11);
            p20 = _mm256_mullo_epi16(L11, coeff13);
            p30 = _mm256_mullo_epi16(L12, coeff5);
            p00 = _mm256_add_epi16(p00, coeff16);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 5);

            p00 = _mm256_packus_epi16(p00, p00);
            _mm256_maskstore_epi64((__int64*)&pfirst[6][i], mask, p00);

            p00 = _mm256_add_epi16(L11, L13);
            p10 = _mm256_add_epi16(L12, L12);
            p00 = _mm256_add_epi16(p00, coeff2);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_srli_epi16(p00, 2);

            p00 = _mm256_packus_epi16(p00, p00);
            _mm256_maskstore_epi64((__int64*)&pfirst[7][i], mask, p00);
        }

        bsy >>= 3;

        __m256i M;
        if (bsx == 64) {
            for (i = 0; i < bsy; i++) {
                M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i * 11));
                _mm256_storeu_si256((__m256i*)dst1, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i * 11 + 32));
                _mm256_storeu_si256((__m256i*)(dst1 + 32), M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i * 11));
                _mm256_storeu_si256((__m256i*)dst2, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i * 11 + 32));
                _mm256_storeu_si256((__m256i*)(dst2 + 32), M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] + i * 11));
                _mm256_storeu_si256((__m256i*)dst3, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] + i * 11 + 32));
                _mm256_storeu_si256((__m256i*)(dst3 + 32), M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] + i * 11));
                _mm256_storeu_si256((__m256i*)dst4, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] + i * 11 + 32));
                _mm256_storeu_si256((__m256i*)(dst4 + 32), M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[4] + i * 11));
                _mm256_storeu_si256((__m256i*)dst5, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[4] + i * 11 + 32));
                _mm256_storeu_si256((__m256i*)(dst5 + 32), M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[5] + i * 11));
                _mm256_storeu_si256((__m256i*)dst6, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[5] + i * 11 + 32));
                _mm256_storeu_si256((__m256i*)(dst6 + 32), M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[6] + i * 11));
                _mm256_storeu_si256((__m256i*)dst7, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[6] + i * 11 + 32));
                _mm256_storeu_si256((__m256i*)(dst7 + 32), M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[7] + i * 11));
                _mm256_storeu_si256((__m256i*)dst8, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[7] + i * 11 + 32));
                _mm256_storeu_si256((__m256i*)(dst8 + 32), M);

                dst1 = dst8 + i_dst;
                dst2 = dst1 + i_dst;
                dst3 = dst2 + i_dst;
                dst4 = dst3 + i_dst;
                dst5 = dst4 + i_dst;
                dst6 = dst5 + i_dst;
                dst7 = dst6 + i_dst;
                dst8 = dst7 + i_dst;
            }
        } else if (bsx == 32) {
            for (i = 0; i < bsy; i++) {
                M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i * 11));
                _mm256_storeu_si256((__m256i*)dst1, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i * 11));
                _mm256_storeu_si256((__m256i*)dst2, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] + i * 11));
                _mm256_storeu_si256((__m256i*)dst3, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] + i * 11));
                _mm256_storeu_si256((__m256i*)dst4, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[4] + i * 11));
                _mm256_storeu_si256((__m256i*)dst5, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[5] + i * 11));
                _mm256_storeu_si256((__m256i*)dst6, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[6] + i * 11));
                _mm256_storeu_si256((__m256i*)dst7, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[7] + i * 11));
                _mm256_storeu_si256((__m256i*)dst8, M);

                dst1 = dst8 + i_dst;
                dst2 = dst1 + i_dst;
                dst3 = dst2 + i_dst;
                dst4 = dst3 + i_dst;
                dst5 = dst4 + i_dst;
                dst6 = dst5 + i_dst;
                dst7 = dst6 + i_dst;
                dst8 = dst7 + i_dst;
            }
        } else {
            __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
            for (i = 0; i < bsy; i++) {
                M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i * 11));
                _mm256_maskstore_epi64((__int64*)dst1, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i * 11));
                _mm256_maskstore_epi64((__int64*)dst2, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] + i * 11));
                _mm256_maskstore_epi64((__int64*)dst3, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] + i * 11));
                _mm256_maskstore_epi64((__int64*)dst4, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[4] + i * 11));
                _mm256_maskstore_epi64((__int64*)dst5, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[5] + i * 11));
                _mm256_maskstore_epi64((__int64*)dst6, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[6] + i * 11));
                _mm256_maskstore_epi64((__int64*)dst7, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[7] + i * 11));
                _mm256_maskstore_epi64((__int64*)dst8, mask, M);

                dst1 = dst8 + i_dst;
                dst2 = dst1 + i_dst;
                dst3 = dst2 + i_dst;
                dst4 = dst3 + i_dst;
                dst5 = dst4 + i_dst;
                dst6 = dst5 + i_dst;
                dst7 = dst6 + i_dst;
                dst8 = dst7 + i_dst;
            }
        }




        /*for (i = 0; i < bsy; i++) {
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
        }*/
    } else if (bsx == 16) {

        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;

        __m256i p00, p10, p20, p30;

        __m256i SS1;
        __m256i L1, L2, L3, L4, L5, L6, L7, L8;

        SS1 = _mm256_loadu_si256((__m256i*)(src + 1));//1...8 9...16 17..24 25..32
        L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));//1
        SS1 = _mm256_loadu_si256((__m256i*)(src + 2));
        L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));//2
        SS1 = _mm256_loadu_si256((__m256i*)(src + 3));
        L3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));//3
        SS1 = _mm256_loadu_si256((__m256i*)(src + 4));
        L4 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));//4
        SS1 = _mm256_loadu_si256((__m256i*)(src + 5));
        L5 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
        SS1 = _mm256_loadu_si256((__m256i*)(src + 6));
        L6 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
        SS1 = _mm256_loadu_si256((__m256i*)(src + 7));
        L7 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));
        SS1 = _mm256_loadu_si256((__m256i*)(src + 8));
        L8 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(SS1, 0));

        p00 = _mm256_mullo_epi16(L1, coeff5);
        p10 = _mm256_mullo_epi16(L2, coeff13);
        p20 = _mm256_mullo_epi16(L3, coeff11);
        p30 = _mm256_mullo_epi16(L4, coeff3);
        p00 = _mm256_add_epi16(p00, coeff16);
        p00 = _mm256_add_epi16(p00, p10);
        p00 = _mm256_add_epi16(p00, p20);
        p00 = _mm256_add_epi16(p00, p30);
        p00 = _mm256_srli_epi16(p00, 5);

        p00 = _mm256_packus_epi16(p00, p00);
        p00 = _mm256_permute4x64_epi64(p00, 0x0008);
        __m256i mask = _mm256_loadu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
        _mm256_maskstore_epi64((__int64*)dst1, mask, p00);

        p10 = _mm256_mullo_epi16(L3, coeff5);
        p20 = _mm256_mullo_epi16(L4, coeff7);
        p30 = _mm256_mullo_epi16(L5, coeff3);
        p00 = _mm256_add_epi16(L2, coeff8);
        p00 = _mm256_add_epi16(p00, p10);
        p00 = _mm256_add_epi16(p00, p20);
        p00 = _mm256_add_epi16(p00, p30);
        p00 = _mm256_srli_epi16(p00, 4);

        p00 = _mm256_packus_epi16(p00, p00);
        p00 = _mm256_permute4x64_epi64(p00, 0x0008);
        _mm256_maskstore_epi64((__int64*)dst2, mask, p00);

        p00 = _mm256_mullo_epi16(L4, coeff7);
        p10 = _mm256_mullo_epi16(L5, coeff15);
        p20 = _mm256_mullo_epi16(L6, coeff9);
        p30 = _mm256_add_epi16(L7, coeff16);
        p00 = _mm256_add_epi16(p00, p10);
        p00 = _mm256_add_epi16(p00, p20);
        p00 = _mm256_add_epi16(p00, p30);
        p00 = _mm256_srli_epi16(p00, 5);

        p00 = _mm256_packus_epi16(p00, p00);
        p00 = _mm256_permute4x64_epi64(p00, 0x0008);
        _mm256_maskstore_epi64((__int64*)dst3, mask, p00);

        p00 = _mm256_add_epi16(L5, L8);
        p10 = _mm256_add_epi16(L6, L7);
        p10 = _mm256_mullo_epi16(p10, coeff3);
        p00 = _mm256_add_epi16(p00, coeff4);
        p00 = _mm256_add_epi16(p00, p10);
        p00 = _mm256_srli_epi16(p00, 3);

        p00 = _mm256_packus_epi16(p00, p00);
        p00 = _mm256_permute4x64_epi64(p00, 0x0008);
        _mm256_maskstore_epi64((__int64*)dst4, mask, p00);


    } else { //8x8 8x32 4x4 4x16

        intra_pred_ang_x_5_sse128(src, dst, i_dst, dir_mode, bsx, bsy);

    }

}

void intra_pred_ang_x_6_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN32(pel_t first_line[64 + 64]);
    int line_size = bsx + bsy - 1;

    int i;
    __m256i zero = _mm256_setzero_si256();
    __m256i offset = _mm256_set1_epi16(2);

    UNUSED_PARAMETER(dir_mode);
    src += 2;

    for (i = 0; i < line_size - 16; i += 32, src += 32) {
        //0 1 2 3 .... 12 13 14 15    16 17 18 19 .... 28 29 30 21
        __m256i S0 = _mm256_loadu_si256((__m256i*)(src - 1));
        __m256i S1 = _mm256_loadu_si256((__m256i*)(src));
        __m256i S2 = _mm256_loadu_si256((__m256i*)(src + 1));

        __m256i L0 = _mm256_unpacklo_epi8(S0, zero);//0 1 2 3 4 5 6 7     16 17 18 19 20 21 22 23
        __m256i L1 = _mm256_unpacklo_epi8(S1, zero);
        __m256i L2 = _mm256_unpacklo_epi8(S2, zero);

        __m256i H0 = _mm256_unpackhi_epi8(S0, zero);//8 9 10 11 12 13 14 15     24 25 26 27 28 29 30 31
        __m256i H1 = _mm256_unpackhi_epi8(S1, zero);
        __m256i H2 = _mm256_unpackhi_epi8(S2, zero);

        __m256i tmp0 = _mm256_permute2x128_si256(L0, H0, 0x0020);//0 1 2 3 4 5 6 7   8 9 10 11 12 13 14 15
        __m256i tmp1 = _mm256_permute2x128_si256(L1, H1, 0x0020);
        __m256i tmp2 = _mm256_permute2x128_si256(L2, H2, 0x0020);
        __m256i sum1 = _mm256_add_epi16(tmp0, tmp1);
        __m256i sum2 = _mm256_add_epi16(tmp1, tmp2);


        tmp0 = _mm256_permute2x128_si256(L0, H0, 0x0031);//16 17...24 25...
        tmp1 = _mm256_permute2x128_si256(L1, H1, 0x0031);
        tmp2 = _mm256_permute2x128_si256(L2, H2, 0x0031);
        __m256i sum3 = _mm256_add_epi16(tmp0, tmp1);
        __m256i sum4 = _mm256_add_epi16(tmp1, tmp2);

        sum1 = _mm256_add_epi16(sum1, sum2);
        sum3 = _mm256_add_epi16(sum3, sum4);

        sum1 = _mm256_add_epi16(sum1, offset);
        sum3 = _mm256_add_epi16(sum3, offset);

        sum1 = _mm256_srli_epi16(sum1, 2);
        sum3 = _mm256_srli_epi16(sum3, 2);

        sum1 = _mm256_packus_epi16(sum1, sum3);//0 2 1 3
        sum1 = _mm256_permute4x64_epi64(sum1, 0x00D8);
        _mm256_storeu_si256((__m256i*)&first_line[i], sum1);
    }

    if (i < line_size) {
        //0 1 2 3 .... 12 13 14 15    16 17 18 19 .... 28 29 30 21
        __m256i S0 = _mm256_loadu_si256((__m256i*)(src - 1));
        __m256i S2 = _mm256_loadu_si256((__m256i*)(src + 1));
        __m256i S1 = _mm256_loadu_si256((__m256i*)(src));
        S0 = _mm256_permute4x64_epi64(S0, 0x00D8);
        S2 = _mm256_permute4x64_epi64(S2, 0x00D8);
        S1 = _mm256_permute4x64_epi64(S1, 0x00D8);

        __m256i L0 = _mm256_unpacklo_epi8(S0, zero);
        __m256i L1 = _mm256_unpacklo_epi8(S1, zero);
        __m256i L2 = _mm256_unpacklo_epi8(S2, zero);

        __m256i sum1 = _mm256_add_epi16(L0, L1);
        __m256i sum2 = _mm256_add_epi16(L1, L2);

        sum1 = _mm256_add_epi16(sum1, sum2);
        sum1 = _mm256_add_epi16(sum1, offset);
        sum1 = _mm256_srli_epi16(sum1, 2);

        sum1 = _mm256_packus_epi16(sum1, sum1);
        sum1 = _mm256_permute4x64_epi64(sum1, 0x0008);
        //store 128 bit
        __m256i mask2 = _mm256_loadu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
        _mm256_maskstore_epi64((__int64*)(first_line + i), mask2, sum1);

        //_mm_storel_epi64((__m128i*)&first_line[i], sum1);
    }

    if (bsx == 64) {
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)&first_line[i]);
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(&first_line[i] + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 1]);
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(&first_line[i + 1] + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 2]);
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(&first_line[i + 2] + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 3]);
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(&first_line[i + 3] + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
        }
    } else if (bsx == 32) {
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)&first_line[i]);
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 1]);
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 2]);
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 3]);
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
        }
    } else if (bsx == 16) {
        __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)&first_line[i]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 1]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 2]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 3]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

        }
    } else if (bsx == 8) {
        __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[7]);
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)&first_line[i]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 1]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 2]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 3]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

        }
    } else {
        __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[3]);
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)&first_line[i]);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 1]);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 2]);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 3]);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;

        }
    }



    /*
    if (bsx == bsy || bsx >= 16) {
        for (i = 0; i < bsy; i++) {
            memcpy(dst, first_line + i, bsx * sizeof(pel_t));
            dst += i_dst;
        }
    } else {//8x32 4x16

        __m256i mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_loadu_si256((__m256i*)&first_line[i]);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_srli_si256(M, 1);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_srli_si256(M, 1);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_srli_si256(M, 1);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
        }
    }*/

}

void intra_pred_ang_x_7_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i, j;

    UNUSED_PARAMETER(dir_mode);

    if (bsx >= bsy) {
        if (bsx <= 8) {//4x4 8x8

            intra_pred_ang_x_7_sse128(src, dst, i_dst, dir_mode, bsx, bsy);

        } else if (bsx & 16) { //16

            __m256i S0, S1, S2, S3;
            __m256i t0, t1, t2, t3;
            __m256i c0;
            __m256i D0;
            __m256i off = _mm256_set1_epi16(64);

            __m256i mask = _mm256_loadu_si256((__m256i*)intrinsic_mask_256_8bit[15]);

            for (j = 0; j < bsy; j++) {
                int idx = tab_idx_mode_7[j];
                c0 = _mm256_loadu_si256((__m256i*)tab_coeff_mode_7_avx[j]);

                S0 = _mm256_loadu_si256((__m256i*)(src + idx));    //0...7 8...15 16...23 24...31
                S1 = _mm256_loadu_si256((__m256i*)(src + idx + 1));//1.. 8 9...16 17...24 25...32
                S2 = _mm256_loadu_si256((__m256i*)(src + idx + 2));//2...9 10...17
                S3 = _mm256_loadu_si256((__m256i*)(src + idx + 3));//3...10 11...18

                S0 = _mm256_permute4x64_epi64(S0, 0x00D8);//0...7 16...23 8...15 24...31
                S1 = _mm256_permute4x64_epi64(S1, 0x00D8);//1...8 17...24 9...16 25...32
                S2 = _mm256_permute4x64_epi64(S2, 0x00D8);
                S3 = _mm256_permute4x64_epi64(S3, 0x00D8);

                t0 = _mm256_unpacklo_epi8(S0, S1);//0 1 1 2 2 3 3 4  4 5 5 6 6 7 7  8     8  9  9 10 10 11 11 12  12 13 13 14 14 15 15 16
                t1 = _mm256_unpacklo_epi8(S2, S3);//2 3 3 4 4 5 5 6  6 7 7 8 8 9 9 10    10 11 11 12 12 13 13 14  14 15 15 16 16 17 17 18
                t0 = _mm256_permute4x64_epi64(t0, 0x00D8);
                t1 = _mm256_permute4x64_epi64(t1, 0x00D8);
                t2 = _mm256_unpacklo_epi16(t0, t1);//0...7
                t3 = _mm256_unpackhi_epi16(t0, t1);//8...15

                t0 = _mm256_maddubs_epi16(t2, c0);
                t1 = _mm256_maddubs_epi16(t3, c0);

                D0 = _mm256_hadds_epi16(t0, t1);//0 1 2 3 8 9 10 11    4 5 6 7 12 13 14 15
                D0 = _mm256_permute4x64_epi64(D0, 0x00D8);
                D0 = _mm256_add_epi16(D0, off);
                D0 = _mm256_srli_epi16(D0, 7);

                D0 = _mm256_packus_epi16(D0, D0);
                D0 = _mm256_permute4x64_epi64(D0, 0x00D8);
                _mm256_maskstore_epi64((__int64*)dst, mask, D0);

                dst += i_dst;
            }

        } else {//32 64

            __m256i S0, S1, S2, S3;
            __m256i t0, t1, t2, t3;
            __m256i c0;
            __m256i D0, D1;
            __m256i off = _mm256_set1_epi16(64);

            for (j = 0; j < bsy; j++) {
                int idx = tab_idx_mode_7[j];
                c0 = _mm256_loadu_si256((__m256i*)tab_coeff_mode_7_avx[j]);
                for (i = 0; i < bsx; i += 32, idx += 32) {
                    S0 = _mm256_loadu_si256((__m256i*)(src + idx));    //0...7 8...15 16...23 24...31
                    S1 = _mm256_loadu_si256((__m256i*)(src + idx + 1));//1.. 8 9...16 17...24 25...32
                    S2 = _mm256_loadu_si256((__m256i*)(src + idx + 2));//2...9 10...17 18
                    S3 = _mm256_loadu_si256((__m256i*)(src + idx + 3));//3...10 11...18 19

                    S0 = _mm256_permute4x64_epi64(S0, 0x00D8);//0...7 16...23 8...15 24...31
                    S1 = _mm256_permute4x64_epi64(S1, 0x00D8);//1...8 17...24 9...16 25...32
                    S2 = _mm256_permute4x64_epi64(S2, 0x00D8);
                    S3 = _mm256_permute4x64_epi64(S3, 0x00D8);

                    t0 = _mm256_unpacklo_epi8(S0, S1);//0 1 1 2 2 3 3 4  4 5 5 6 6 7 7  8     8  9  9 10 10 11 11 12  12 13 13 14 14 15 15 16
                    t1 = _mm256_unpacklo_epi8(S2, S3);//2 3 3 4 4 5 5 6  6 7 7 8 8 9 9 10    10 11 11 12 12 13 13 14  14 15 15 16 16 17 17 18
                    t0 = _mm256_permute4x64_epi64(t0, 0x00D8);
                    t1 = _mm256_permute4x64_epi64(t1, 0x00D8);
                    t2 = _mm256_unpacklo_epi16(t0, t1);//
                    t3 = _mm256_unpackhi_epi16(t0, t1);//........15 16 17 18

                    t0 = _mm256_maddubs_epi16(t2, c0);
                    t1 = _mm256_maddubs_epi16(t3, c0);

                    D0 = _mm256_hadds_epi16(t0, t1);//0 1 2 3 8 9 10 11    4 5 6 7 12 13 14 15
                    D0 = _mm256_permute4x64_epi64(D0, 0x00D8);
                    D0 = _mm256_add_epi16(D0, off);
                    D0 = _mm256_srli_epi16(D0, 7);

                    t0 = _mm256_unpackhi_epi8(S0, S1);//16 17 17 18  18 19 19 20  20 21 21 22 22 23 23 24...24 25 25..
                    t1 = _mm256_unpackhi_epi8(S2, S3);//18 19 19 20  .....
                    t0 = _mm256_permute4x64_epi64(t0, 0x00D8);
                    t1 = _mm256_permute4x64_epi64(t1, 0x00D8);
                    t2 = _mm256_unpacklo_epi16(t0, t1);//16 17 18 19...
                    t3 = _mm256_unpackhi_epi16(t0, t1);//24 25 26 27...

                    t0 = _mm256_maddubs_epi16(t2, c0);
                    t1 = _mm256_maddubs_epi16(t3, c0);

                    D1 = _mm256_hadds_epi16(t0, t1);//16 17 18 19 24 25 26 27    20 21 22 23 28 29 30 31
                    D1 = _mm256_permute4x64_epi64(D1, 0x00D8);
                    D1 = _mm256_add_epi16(D1, off);
                    D1 = _mm256_srli_epi16(D1, 7);

                    D0 = _mm256_packus_epi16(D0, D1);
                    D0 = _mm256_permute4x64_epi64(D0, 0x00D8);
                    _mm256_storeu_si256((__m256i*)(dst + i), D0);

                }
                dst += i_dst;
            }
        }
    } else {
        intra_pred_ang_x_7_sse128(src, dst, i_dst, dir_mode, bsx, bsy);
    }

}

void intra_pred_ang_x_8_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{

    ALIGN32(pel_t first_line[2 * (64 + 48)]);
    int line_size = bsx + (bsy >> 1) - 1;
    int i;
    int aligned_line_size = ((line_size + 31) >> 4) << 4;
    pel_t *pfirst[2];
    __m256i zero = _mm256_setzero_si256();

    __m256i coeff   = _mm256_set1_epi16(3); //16个
    __m256i offset1 = _mm256_set1_epi16(4);
    __m256i offset2 = _mm256_set1_epi16(2);

    UNUSED_PARAMETER(dir_mode);

    pfirst[0] = first_line;
    pfirst[1] = first_line + aligned_line_size;

    __m256i p01, p02, p11, p12;
    __m256i p21, p22, p31, p32;
    __m256i tmp0, tmp1, tmp2, tmp3;
    for (i = 0; i < line_size - 16; i += 32, src += 32) {
        //0 1 2 3 .... 12 13 14 15    16 17 18 19 .... 28 29 30 21
        __m256i S0 = _mm256_loadu_si256((__m256i*)(src));
        __m256i S3 = _mm256_loadu_si256((__m256i*)(src + 3));
        __m256i S1 = _mm256_loadu_si256((__m256i*)(src + 1));
        __m256i S2 = _mm256_loadu_si256((__m256i*)(src + 2));

        __m256i L0 = _mm256_unpacklo_epi8(S0, zero);//0 1 2 3 4 5 6 7     16 17 18 19 20 21 22 23
        __m256i L1 = _mm256_unpacklo_epi8(S1, zero);
        __m256i L2 = _mm256_unpacklo_epi8(S2, zero);
        __m256i L3 = _mm256_unpacklo_epi8(S3, zero);

        __m256i H0 = _mm256_unpackhi_epi8(S0, zero);//8 9 10 11 12 13 14 15     24 25 26 27 28 29 30 31
        __m256i H1 = _mm256_unpackhi_epi8(S1, zero);
        __m256i H2 = _mm256_unpackhi_epi8(S2, zero);
        __m256i H3 = _mm256_unpackhi_epi8(S3, zero);

        tmp0 = _mm256_permute2x128_si256(L0, H0, 0x0020);//0 1 2 3 4 5 6 7   8 9 10 11 12 13 14 15
        tmp1 = _mm256_permute2x128_si256(L1, H1, 0x0020);
        tmp2 = _mm256_permute2x128_si256(L2, H2, 0x0020);
        tmp3 = _mm256_permute2x128_si256(L3, H3, 0x0020);

        p01 = _mm256_add_epi16(tmp1, tmp2);
        p01 = _mm256_mullo_epi16(p01, coeff);
        p02 = _mm256_add_epi16(tmp0, tmp3);
        p02 = _mm256_add_epi16(p02, offset1);
        p01 = _mm256_add_epi16(p01, p02);
        p01 = _mm256_srli_epi16(p01, 3); //

        //prepare for next line
        p21 = _mm256_add_epi16(tmp1, tmp2);
        p22 = _mm256_add_epi16(tmp2, tmp3);

        tmp0 = _mm256_permute2x128_si256(L0, H0, 0x0031);//16 17....24 25....
        tmp1 = _mm256_permute2x128_si256(L1, H1, 0x0031);
        tmp2 = _mm256_permute2x128_si256(L2, H2, 0x0031);
        tmp3 = _mm256_permute2x128_si256(L3, H3, 0x0031);

        p11 = _mm256_add_epi16(tmp1, tmp2);
        p11 = _mm256_mullo_epi16(p11, coeff);
        p12 = _mm256_add_epi16(tmp0, tmp3);
        p12 = _mm256_add_epi16(p12, offset1);
        p11 = _mm256_add_epi16(p11, p12);
        p11 = _mm256_srli_epi16(p11, 3);

        //prepare for next line
        p31 = _mm256_add_epi16(tmp1, tmp2);
        p32 = _mm256_add_epi16(tmp2, tmp3);

        p01 = _mm256_packus_epi16(p01, p11);
        p01 = _mm256_permute4x64_epi64(p01, 0x00D8);
        _mm256_storeu_si256((__m256i*)&pfirst[0][i], p01);

        p21 = _mm256_add_epi16(p21, p22);
        p31 = _mm256_add_epi16(p31, p32);

        p21 = _mm256_add_epi16(p21, offset2);
        p31 = _mm256_add_epi16(p31, offset2);

        p21 = _mm256_srli_epi16(p21, 2);
        p31 = _mm256_srli_epi16(p31, 2);

        p21 = _mm256_packus_epi16(p21, p31);
        p21 = _mm256_permute4x64_epi64(p21, 0x00D8);
        _mm256_storeu_si256((__m256i*)&pfirst[1][i], p21);
    }

    if (i < line_size) {
        __m256i S0 = _mm256_loadu_si256((__m256i*)(src));
        __m256i S3 = _mm256_loadu_si256((__m256i*)(src + 3));
        __m256i S1 = _mm256_loadu_si256((__m256i*)(src + 1));
        __m256i S2 = _mm256_loadu_si256((__m256i*)(src + 2));

        S0 = _mm256_permute4x64_epi64(S0, 0x00D8);
        S3 = _mm256_permute4x64_epi64(S3, 0x00D8);
        S1 = _mm256_permute4x64_epi64(S1, 0x00D8);
        S2 = _mm256_permute4x64_epi64(S2, 0x00D8);

        __m256i L0 = _mm256_unpacklo_epi8(S0, zero);
        __m256i L1 = _mm256_unpacklo_epi8(S1, zero);
        __m256i L2 = _mm256_unpacklo_epi8(S2, zero);
        __m256i L3 = _mm256_unpacklo_epi8(S3, zero);

        p01 = _mm256_add_epi16(L1, L2);
        p01 = _mm256_mullo_epi16(p01, coeff);
        p02 = _mm256_add_epi16(L0, L3);
        p02 = _mm256_add_epi16(p02, offset1);
        p01 = _mm256_add_epi16(p01, p02);
        p01 = _mm256_srli_epi16(p01, 3);

        p01 = _mm256_packus_epi16(p01, p01);
        p01 = _mm256_permute4x64_epi64(p01, 0x0008);
        __m256i mask = _mm256_loadu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
        _mm256_maskstore_epi64((__int64*)&pfirst[0][i], mask, p01);

        p01 = _mm256_add_epi16(L1, L2);
        p02 = _mm256_add_epi16(L2, L3);

        p01 = _mm256_add_epi16(p01, p02);
        p01 = _mm256_add_epi16(p01, offset2);
        p01 = _mm256_srli_epi16(p01, 2);

        p01 = _mm256_packus_epi16(p01, p01);
        p01=_mm256_permute4x64_epi64(p01,0x0008);
        _mm256_maskstore_epi64((__int64*)&pfirst[1][i], mask, p01);
    }

    bsy >>= 1;

    if (bsx == 64) {

        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 1));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 1 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 1));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 1 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 2));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 2 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 2));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 2 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 3));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 3 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 3));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 3 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

        }
    } else if (bsx == 32) {
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 1));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 1));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 2));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 2));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 3));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 3));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
        }
    } else if (bsx == 16) {
        __m256i mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 1));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 1));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 2));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 2));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 3));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 3));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
        }
    } else if (bsx == 8) {
        __m256i mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 1));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 1));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 2));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 2));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 3));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 3));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
        }
    } else {
        __m256i mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 1));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 1));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 2));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 2));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 3));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 3));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
        }
    }

    /*if (bsx != 8) {
        for (i = 0; i < bsy; i++) {
            memcpy(dst, pfirst[0] + i, bsx * sizeof(pel_t));
            memcpy(dst + i_dst, pfirst[1] + i, bsx * sizeof(pel_t));
            dst += i_dst2;
        }
    } else if (bsy == 4) {//8x8
        __m256i mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);

        __m256i M1 = _mm256_loadu_si256((__m256i*)&pfirst[0][0]);
        __m256i M2 = _mm256_loadu_si256((__m256i*)&pfirst[1][0]);
        _mm256_maskstore_epi64((__int64*)dst, mask, M1);
        _mm256_maskstore_epi64((__int64*)(dst + i_dst), mask, M2);
        dst += i_dst2;
        M1 = _mm256_srli_si256(M1, 1);
        M2 = _mm256_srli_si256(M2, 1);
        _mm256_maskstore_epi64((__int64*)dst, mask, M1);
        _mm256_maskstore_epi64((__int64*)(dst + i_dst), mask, M2);
        dst += i_dst2;
        M1 = _mm256_srli_si256(M1, 1);
        M2 = _mm256_srli_si256(M2, 1);
        _mm256_maskstore_epi64((__int64*)dst, mask, M1);
        _mm256_maskstore_epi64((__int64*)(dst + i_dst), mask, M2);
        dst += i_dst2;
        M1 = _mm256_srli_si256(M1, 1);
        M2 = _mm256_srli_si256(M2, 1);
        _mm256_maskstore_epi64((__int64*)dst, mask, M1);
        _mm256_maskstore_epi64((__int64*)(dst + i_dst), mask, M2);
    } else { //8x32
        __m256i mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < 16; i = i + 4) {
            __m256i M1 = _mm256_loadu_si256((__m256i*)&pfirst[0][i]);
            __m256i M2 = _mm256_loadu_si256((__m256i*)&pfirst[1][i]);

            _mm256_maskstore_epi64((__int64*)dst, mask, M1);
            _mm256_maskstore_epi64((__int64*)(dst + i_dst), mask, M2);
            dst += i_dst2;
            M1 = _mm256_srli_si256(M1, 1);
            M2 = _mm256_srli_si256(M2, 1);
            _mm256_maskstore_epi64((__int64*)dst, mask, M1);
            _mm256_maskstore_epi64((__int64*)(dst + i_dst), mask, M2);
            dst += i_dst2;
            M1 = _mm256_srli_si256(M1, 1);
            M2 = _mm256_srli_si256(M2, 1);
            _mm256_maskstore_epi64((__int64*)dst, mask, M1);
            _mm256_maskstore_epi64((__int64*)(dst + i_dst), mask, M2);
            dst += i_dst2;
            M1 = _mm256_srli_si256(M1, 1);
            M2 = _mm256_srli_si256(M2, 1);
            _mm256_maskstore_epi64((__int64*)dst, mask, M1);
            _mm256_maskstore_epi64((__int64*)(dst + i_dst), mask, M2);
            dst += i_dst2;
            //M1 = _mm256_srli_si256(M1, 1);
            //M2 = _mm256_srli_si256(M2, 1);
            //_mm256_maskstore_epi64((__m256i*)dst, mask, M1);
            //_mm256_maskstore_epi64((__m256i*)(dst + i_dst), mask, M2);
            //dst += i_dst2;
            //M1 = _mm256_srli_si256(M1, 1);
            //M2 = _mm256_srli_si256(M2, 1);
            //_mm256_maskstore_epi64((__m256i*)dst, mask, M1);
            //_mm256_maskstore_epi64((__m256i*)(dst + i_dst), mask, M2);
            //dst += i_dst2;
            //M1 = _mm256_srli_si256(M1, 1);
            //M2 = _mm256_srli_si256(M2, 1);
            //_mm256_maskstore_epi64((__m256i*)dst, mask, M1);
            //_mm256_maskstore_epi64((__m256i*)(dst + i_dst), mask, M2);
            //dst += i_dst2;
            //M1 = _mm256_srli_si256(M1, 1);
            //M2 = _mm256_srli_si256(M2, 1);
            //_mm256_maskstore_epi64((__m256i*)dst, mask, M1);
            //_mm256_maskstore_epi64((__m256i*)(dst + i_dst), mask, M2);
            //dst += i_dst2;
        }
    }*/

}

void intra_pred_ang_x_9_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i, j;

    UNUSED_PARAMETER(dir_mode);

    if (bsx >= bsy) {
        if (bsx & 0x07) {//4
            intra_pred_ang_x_9_sse128(src, dst, i_dst, dir_mode, bsx, bsy);

        } else if (bsx & 0x0f) {//8
            intra_pred_ang_x_9_sse128(src, dst, i_dst, dir_mode, bsx, bsy);

        } else if (bsx & 16) { //16

            __m256i S0, S1, S2, S3;
            __m256i t0, t1, t2, t3;
            __m256i c0;
            __m256i D0;
            __m256i off = _mm256_set1_epi16(64);

            __m256i mask = _mm256_loadu_si256((__m256i*)intrinsic_mask_256_8bit[15]);

            for (j = 0; j < bsy; j++) {

                int idx = tab_idx_mode_9[j];
                c0 = _mm256_set1_epi32(((int*)(tab_coeff_mode_9[j]))[0]);

                S0 = _mm256_loadu_si256((__m256i*)(src + idx));    //0...7 8...15 16...23 24...31
                S1 = _mm256_loadu_si256((__m256i*)(src + idx + 1));//1.. 8 9...16 17...24 25...32
                S2 = _mm256_loadu_si256((__m256i*)(src + idx + 2));//2...9 10...17
                S3 = _mm256_loadu_si256((__m256i*)(src + idx + 3));//3...10 11...18

                S0 = _mm256_permute4x64_epi64(S0, 0x00D8);//0...7 16...23 8...15 24...31
                S1 = _mm256_permute4x64_epi64(S1, 0x00D8);//1...8 17...24 9...16 25...32
                S2 = _mm256_permute4x64_epi64(S2, 0x00D8);
                S3 = _mm256_permute4x64_epi64(S3, 0x00D8);

                t0 = _mm256_unpacklo_epi8(S0, S1);//0 1 1 2 2 3 3 4  4 5 5 6 6 7 7  8     8  9  9 10 10 11 11 12  12 13 13 14 14 15 15 16
                t1 = _mm256_unpacklo_epi8(S2, S3);//2 3 3 4 4 5 5 6  6 7 7 8 8 9 9 10    10 11 11 12 12 13 13 14  14 15 15 16 16 17 17 18
                t0 = _mm256_permute4x64_epi64(t0, 0x00D8);
                t1 = _mm256_permute4x64_epi64(t1, 0x00D8);
                t2 = _mm256_unpacklo_epi16(t0, t1);//0...7
                t3 = _mm256_unpackhi_epi16(t0, t1);//8...15

                t0 = _mm256_maddubs_epi16(t2, c0);
                t1 = _mm256_maddubs_epi16(t3, c0);

                D0 = _mm256_hadds_epi16(t0, t1);//0 1 2 3 8 9 10 11    4 5 6 7 12 13 14 15
                D0 = _mm256_permute4x64_epi64(D0, 0x00D8);
                D0 = _mm256_add_epi16(D0, off);
                D0 = _mm256_srli_epi16(D0, 7);

                D0 = _mm256_packus_epi16(D0, D0);
                D0 = _mm256_permute4x64_epi64(D0, 0x00D8);
                _mm256_maskstore_epi64((__int64*)dst, mask, D0);

                dst += i_dst;
            }

        } else {//32 64

            __m256i S0, S1, S2, S3;
            __m256i t0, t1, t2, t3;
            __m256i c0;
            __m256i D0, D1;
            __m256i off = _mm256_set1_epi16(64);

            for (j = 0; j < bsy; j++) {
                int idx = tab_idx_mode_9[j];
                c0 = _mm256_set1_epi32(((int*)tab_coeff_mode_9[j])[0]);
                for (i = 0; i < bsx; i += 32, idx += 32) {
                    S0 = _mm256_loadu_si256((__m256i*)(src + idx));    //0...7 8...15 16...23 24...31
                    S1 = _mm256_loadu_si256((__m256i*)(src + idx + 1));//1.. 8 9...16 17...24 25...32
                    S2 = _mm256_loadu_si256((__m256i*)(src + idx + 2));//2...9 10...17 18
                    S3 = _mm256_loadu_si256((__m256i*)(src + idx + 3));//3...10 11...18 19

                    S0 = _mm256_permute4x64_epi64(S0, 0x00D8);//0...7 16...23 8...15 24...31
                    S1 = _mm256_permute4x64_epi64(S1, 0x00D8);//1...8 17...24 9...16 25...32
                    S2 = _mm256_permute4x64_epi64(S2, 0x00D8);
                    S3 = _mm256_permute4x64_epi64(S3, 0x00D8);

                    t0 = _mm256_unpacklo_epi8(S0, S1);//0 1 1 2 2 3 3 4  4 5 5 6 6 7 7  8     8  9  9 10 10 11 11 12  12 13 13 14 14 15 15 16
                    t1 = _mm256_unpacklo_epi8(S2, S3);//2 3 3 4 4 5 5 6  6 7 7 8 8 9 9 10    10 11 11 12 12 13 13 14  14 15 15 16 16 17 17 18
                    t0 = _mm256_permute4x64_epi64(t0, 0x00D8);
                    t1 = _mm256_permute4x64_epi64(t1, 0x00D8);
                    t2 = _mm256_unpacklo_epi16(t0, t1);//
                    t3 = _mm256_unpackhi_epi16(t0, t1);//........15 16 17 18

                    t0 = _mm256_maddubs_epi16(t2, c0);
                    t1 = _mm256_maddubs_epi16(t3, c0);

                    D0 = _mm256_hadds_epi16(t0, t1);//0 1 2 3 8 9 10 11    4 5 6 7 12 13 14 15
                    D0 = _mm256_permute4x64_epi64(D0, 0x00D8);
                    D0 = _mm256_add_epi16(D0, off);
                    D0 = _mm256_srli_epi16(D0, 7);

                    t0 = _mm256_unpackhi_epi8(S0, S1);//16 17 17 18  18 19 19 20  20 21 21 22 22 23 23 24...24 25 25..
                    t1 = _mm256_unpackhi_epi8(S2, S3);//18 19 19 20  .....
                    t0 = _mm256_permute4x64_epi64(t0, 0x00D8);
                    t1 = _mm256_permute4x64_epi64(t1, 0x00D8);
                    t2 = _mm256_unpacklo_epi16(t0, t1);//16 17 18 19...
                    t3 = _mm256_unpackhi_epi16(t0, t1);//24 25 26 27...

                    t0 = _mm256_maddubs_epi16(t2, c0);
                    t1 = _mm256_maddubs_epi16(t3, c0);

                    D1 = _mm256_hadds_epi16(t0, t1);//16 17 18 19 24 25 26 27    20 21 22 23 28 29 30 31
                    D1 = _mm256_permute4x64_epi64(D1, 0x00D8);
                    D1 = _mm256_add_epi16(D1, off);
                    D1 = _mm256_srli_epi16(D1, 7);

                    D0 = _mm256_packus_epi16(D0, D1);
                    D0 = _mm256_permute4x64_epi64(D0, 0x00D8);
                    _mm256_storeu_si256((__m256i*)(dst + i), D0);

                }
                dst += i_dst;
            }
        }
    } else {//4x16 8x32
        intra_pred_ang_x_9_sse128(src, dst, i_dst, dir_mode, bsx, bsy);
    }

}

void intra_pred_ang_x_10_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    if (bsy == 4) {
        intra_pred_ang_x_10_sse128(src, dst, i_dst, dir_mode, bsx, bsy);
        return;
    }
    int i;
    pel_t *dst1 = dst;
    pel_t *dst2 = dst1 + i_dst;
    pel_t *dst3 = dst2 + i_dst;
    pel_t *dst4 = dst3 + i_dst;
    UNUSED_PARAMETER(dir_mode);

    if (bsy != 4) {

        __m256i zero = _mm256_setzero_si256();

        __m256i coeff2 = _mm256_set1_epi16(2);
        __m256i coeff3 = _mm256_set1_epi16(3);
        __m256i coeff4 = _mm256_set1_epi16(4);
        __m256i coeff5 = _mm256_set1_epi16(5);
        __m256i coeff7 = _mm256_set1_epi16(7);
        __m256i coeff8 = _mm256_set1_epi16(8);

        ALIGN32(pel_t first_line[4 * (64 + 32)]);
        int line_size = bsx + bsy / 4 - 1;
        int aligned_line_size = ((line_size + 31) >> 4) << 4;
        pel_t *pfirst[4];

        pfirst[0] = first_line;
        pfirst[1] = first_line + aligned_line_size;
        pfirst[2] = first_line + aligned_line_size * 2;
        pfirst[3] = first_line + aligned_line_size * 3;

        for (i = 0; i < line_size - 16; i += 32, src += 32) {
            __m256i p00, p10, p20, p30;
            __m256i p01, p11, p21, p31;
            //0 1 2 3 .... 12 13 14 15    16 17 18 19 .... 28 29 30 21
            __m256i S0 = _mm256_loadu_si256((__m256i*)(src));
            __m256i S3 = _mm256_loadu_si256((__m256i*)(src + 3));
            __m256i S1 = _mm256_loadu_si256((__m256i*)(src + 1));
            __m256i S2 = _mm256_loadu_si256((__m256i*)(src + 2));

            __m256i L0 = _mm256_unpacklo_epi8(S0, zero);//0 1 2 3 4 5 6 7     16 17 18 19 20 21 22 23
            __m256i L1 = _mm256_unpacklo_epi8(S1, zero);
            __m256i L2 = _mm256_unpacklo_epi8(S2, zero);
            __m256i L3 = _mm256_unpacklo_epi8(S3, zero);

            __m256i H0 = _mm256_unpackhi_epi8(S0, zero);// 8 9 10 11 12 13 14 15     24 25 26 27 28 29 30 31
            __m256i H1 = _mm256_unpackhi_epi8(S1, zero);
            __m256i H2 = _mm256_unpackhi_epi8(S2, zero);
            __m256i H3 = _mm256_unpackhi_epi8(S3, zero);

            __m256i tmpL0 = _mm256_permute2x128_si256(L0, H0, 0x0020);//0 1 2 3 4 5 6 7   8 9 10 11 12 13 14 15
            __m256i tmpL1 = _mm256_permute2x128_si256(L1, H1, 0x0020);
            __m256i tmpL2 = _mm256_permute2x128_si256(L2, H2, 0x0020);
            __m256i tmpL3 = _mm256_permute2x128_si256(L3, H3, 0x0020);

            __m256i tmpH0 = _mm256_permute2x128_si256(L0, H0, 0x0031);//16 17...24 25...
            __m256i tmpH1 = _mm256_permute2x128_si256(L1, H1, 0x0031);
            __m256i tmpH2 = _mm256_permute2x128_si256(L2, H2, 0x0031);
            __m256i tmpH3 = _mm256_permute2x128_si256(L3, H3, 0x0031);

            p00 = _mm256_mullo_epi16(tmpL0, coeff3);//0 1 2 3 4 5 6 7   8 9 10 11 12 13 14 15
            p10 = _mm256_mullo_epi16(tmpL1, coeff7);
            p20 = _mm256_mullo_epi16(tmpL2, coeff5);
            p30 = _mm256_add_epi16(tmpL3, coeff8);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_srli_epi16(p00, 4);

            p01 = _mm256_mullo_epi16(tmpH0, coeff3);//16 17...24 25...
            p11 = _mm256_mullo_epi16(tmpH1, coeff7);
            p21 = _mm256_mullo_epi16(tmpH2, coeff5);
            p31 = _mm256_add_epi16(tmpH3, coeff8);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_srli_epi16(p01, 4);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[0][i], p00);

            p00 = _mm256_add_epi16(tmpL1, tmpL2);
            p00 = _mm256_mullo_epi16(p00, coeff3);
            p10 = _mm256_add_epi16(tmpL0, tmpL3);
            p10 = _mm256_add_epi16(p10, coeff4);
            p00 = _mm256_add_epi16(p10, p00);
            p00 = _mm256_srli_epi16(p00, 3);

            p01 = _mm256_add_epi16(tmpH1, tmpH2);
            p01 = _mm256_mullo_epi16(p01, coeff3);
            p11 = _mm256_add_epi16(tmpH0, tmpH3);
            p11 = _mm256_add_epi16(p11, coeff4);
            p01 = _mm256_add_epi16(p11, p01);
            p01 = _mm256_srli_epi16(p01, 3);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[1][i], p00);

            p10 = _mm256_mullo_epi16(tmpL1, coeff5);
            p20 = _mm256_mullo_epi16(tmpL2, coeff7);
            p30 = _mm256_mullo_epi16(tmpL3, coeff3);
            p00 = _mm256_add_epi16(tmpL0, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 4);

            p11 = _mm256_mullo_epi16(tmpH1, coeff5);
            p21 = _mm256_mullo_epi16(tmpH2, coeff7);
            p31 = _mm256_mullo_epi16(tmpH3, coeff3);
            p01 = _mm256_add_epi16(tmpH0, coeff8);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_srli_epi16(p01, 4);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[2][i], p00);

            p00 = _mm256_add_epi16(tmpL1, tmpL2);
            p10 = _mm256_add_epi16(tmpL2, tmpL3);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, coeff2);
            p00 = _mm256_srli_epi16(p00, 2);

            p01 = _mm256_add_epi16(tmpH1, tmpH2);
            p11 = _mm256_add_epi16(tmpH2, tmpH3);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, coeff2);
            p01 = _mm256_srli_epi16(p01, 2);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[3][i], p00);
        }

        if (i < line_size) {
            __m256i p00, p10, p20, p30;
            __m256i S0 = _mm256_loadu_si256((__m256i*)(src));
            __m256i S3 = _mm256_loadu_si256((__m256i*)(src + 3));
            __m256i S1 = _mm256_loadu_si256((__m256i*)(src + 1));
            __m256i S2 = _mm256_loadu_si256((__m256i*)(src + 2));

            S0 = _mm256_permute4x64_epi64(S0, 0x00D8);
            S3 = _mm256_permute4x64_epi64(S3, 0x00D8);
            S2 = _mm256_permute4x64_epi64(S2, 0x00D8);
            S1 = _mm256_permute4x64_epi64(S1, 0x00D8);

            __m256i L0 = _mm256_unpacklo_epi8(S0, zero);
            __m256i L1 = _mm256_unpacklo_epi8(S1, zero);
            __m256i L2 = _mm256_unpacklo_epi8(S2, zero);
            __m256i L3 = _mm256_unpacklo_epi8(S3, zero);

            p00 = _mm256_mullo_epi16(L0, coeff3);
            p10 = _mm256_mullo_epi16(L1, coeff7);
            p20 = _mm256_mullo_epi16(L2, coeff5);
            p30 = _mm256_add_epi16(L3, coeff8);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_srli_epi16(p00, 4);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            __m256i mask = _mm256_loadu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
            _mm256_maskstore_epi64((__int64*)&pfirst[0][i], mask, p00);

            p00 = _mm256_add_epi16(L1, L2);
            p00 = _mm256_mullo_epi16(p00, coeff3);
            p10 = _mm256_add_epi16(L0, L3);
            p10 = _mm256_add_epi16(p10, coeff4);
            p00 = _mm256_add_epi16(p10, p00);
            p00 = _mm256_srli_epi16(p00, 3);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            _mm256_maskstore_epi64((__int64*)&pfirst[1][i], mask, p00);

            p10 = _mm256_mullo_epi16(L1, coeff5);
            p20 = _mm256_mullo_epi16(L2, coeff7);
            p30 = _mm256_mullo_epi16(L3, coeff3);
            p00 = _mm256_add_epi16(L0, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 4);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            _mm256_maskstore_epi64((__int64*)&pfirst[2][i], mask, p00);

            p00 = _mm256_add_epi16(L1, L2);
            p10 = _mm256_add_epi16(L2, L3);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, coeff2);
            p00 = _mm256_srli_epi16(p00, 2);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            _mm256_maskstore_epi64((__int64*)&pfirst[3][i], mask, p00);
        }

        bsy >>= 2;
        int i_dstx4 = i_dst << 2;
        if (bsx == 64) {

            for (i = 0; i < bsy; i++) {
                __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i));
                _mm256_storeu_si256((__m256i*)dst1, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 32));
                _mm256_storeu_si256((__m256i*)(dst1 + 32), M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i));
                _mm256_storeu_si256((__m256i*)dst2, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 32));
                _mm256_storeu_si256((__m256i*)(dst2 + 32), M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] + i));
                _mm256_storeu_si256((__m256i*)dst3, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] + i + 32));
                _mm256_storeu_si256((__m256i*)(dst3 + 32), M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] + i));
                _mm256_storeu_si256((__m256i*)dst4, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] + i + 32));
                _mm256_storeu_si256((__m256i*)(dst4 + 32), M);

                dst1 += i_dstx4;
                dst2 += i_dstx4;
                dst3 += i_dstx4;
                dst4 += i_dstx4;
            }

        } else if (bsx == 32) {

            for (i = 0; i < bsy; i++) {
                __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i));
                _mm256_storeu_si256((__m256i*)dst1, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i));
                _mm256_storeu_si256((__m256i*)dst2, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] + i));
                _mm256_storeu_si256((__m256i*)dst3, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] + i));
                _mm256_storeu_si256((__m256i*)dst4, M);

                dst1 += i_dstx4;
                dst2 += i_dstx4;
                dst3 += i_dstx4;
                dst4 += i_dstx4;
            }

        } else if (bsx == 16) {
            __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
            for (i = 0; i < bsy; i++) {
                __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i));
                _mm256_maskstore_epi64((__int64*)dst1, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i));
                _mm256_maskstore_epi64((__int64*)dst2, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] + i));
                _mm256_maskstore_epi64((__int64*)dst3, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] + i));
                _mm256_maskstore_epi64((__int64*)dst4, mask, M);

                dst1 += i_dstx4;
                dst2 += i_dstx4;
                dst3 += i_dstx4;
                dst4 += i_dstx4;
            }
        } else if (bsx == 8) {
            __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[7]);
            for (i = 0; i < bsy; i++) {
                __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i));
                _mm256_maskstore_epi64((__int64*)dst1, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i));
                _mm256_maskstore_epi64((__int64*)dst2, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] + i));
                _mm256_maskstore_epi64((__int64*)dst3, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] + i));
                _mm256_maskstore_epi64((__int64*)dst4, mask, M);

                dst1 += i_dstx4;
                dst2 += i_dstx4;
                dst3 += i_dstx4;
                dst4 += i_dstx4;
            }
        } else {
            __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[3]);
            for (i = 0; i < bsy; i++) {
                __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i));
                _mm256_maskstore_epi32((int*)dst1, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i));
                _mm256_maskstore_epi32((int*)dst2, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] + i));
                _mm256_maskstore_epi32((int*)dst3, mask, M);

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] + i));
                _mm256_maskstore_epi32((int*)dst4, mask, M);

                dst1 += i_dstx4;
                dst2 += i_dstx4;
                dst3 += i_dstx4;
                dst4 += i_dstx4;
            }

        }

        /*
        if (bsx != 8) {
            switch (bsx) {
            case 4:
                for (i = 0; i < bsy; i++) {
                    CP32(dst1, pfirst[0] + i); dst1 += i_dstx4;
                    CP32(dst2, pfirst[1] + i); dst2 += i_dstx4;
                    CP32(dst3, pfirst[2] + i); dst3 += i_dstx4;
                    CP32(dst4, pfirst[3] + i); dst4 += i_dstx4;
                }
                break;
            case 16:
                for (i = 0; i < bsy; i++) {
                    memcpy(dst1, pfirst[0] + i, 16 * sizeof(pel_t)); dst1 += i_dstx4;
                    memcpy(dst2, pfirst[1] + i, 16 * sizeof(pel_t)); dst2 += i_dstx4;
                    memcpy(dst3, pfirst[2] + i, 16 * sizeof(pel_t)); dst3 += i_dstx4;
                    memcpy(dst4, pfirst[3] + i, 16 * sizeof(pel_t)); dst4 += i_dstx4;
                }
                break;
            case 32:
                for (i = 0; i < bsy; i++) {
                    memcpy(dst1, pfirst[0] + i, 32 * sizeof(pel_t)); dst1 += i_dstx4;
                    memcpy(dst2, pfirst[1] + i, 32 * sizeof(pel_t)); dst2 += i_dstx4;
                    memcpy(dst3, pfirst[2] + i, 32 * sizeof(pel_t)); dst3 += i_dstx4;
                    memcpy(dst4, pfirst[3] + i, 32 * sizeof(pel_t)); dst4 += i_dstx4;
                }
                break;
            case 64:
                for (i = 0; i < bsy; i++) {
                    memcpy(dst1, pfirst[0] + i, 64 * sizeof(pel_t)); dst1 += i_dstx4;
                    memcpy(dst2, pfirst[1] + i, 64 * sizeof(pel_t)); dst2 += i_dstx4;
                    memcpy(dst3, pfirst[2] + i, 64 * sizeof(pel_t)); dst3 += i_dstx4;
                    memcpy(dst4, pfirst[3] + i, 64 * sizeof(pel_t)); dst4 += i_dstx4;
                }
                break;
            default:
                assert(0);
                break;
            }

        } else {
            if (bsy == 2) { //8x8
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
            } else {//8x32
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
        }*/
    }
}

void intra_pred_ang_x_11_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i, j;

    UNUSED_PARAMETER(dir_mode);

    if (bsx & 0x07) {
        intra_pred_ang_x_11_sse128(src, dst, i_dst, dir_mode, bsx, bsy);
    } else if (bsx & 0x0f) {
        intra_pred_ang_x_11_sse128(src, dst, i_dst, dir_mode, bsx, bsy);
    } else if (bsx & 16) {

        __m256i S0, S1, S2, S3;
        __m256i t0, t1, t2, t3;
        __m256i c0;
        __m256i D0;
        __m256i off = _mm256_set1_epi16(64);

        __m256i mask = _mm256_loadu_si256((__m256i*)intrinsic_mask_256_8bit[15]);

        for (j = 0; j < bsy; j++) {

            int idx = (j + 1) >> 3;
            c0 = _mm256_set1_epi32(((int*)(tab_coeff_mode_11[j & 0x07]))[0]);

            S0 = _mm256_loadu_si256((__m256i*)(src + idx));    //0...7 8...15 16...23 24...31
            S1 = _mm256_loadu_si256((__m256i*)(src + idx + 1));//1.. 8 9...16 17...24 25...32
            S2 = _mm256_loadu_si256((__m256i*)(src + idx + 2));//2...9 10...17
            S3 = _mm256_loadu_si256((__m256i*)(src + idx + 3));//3...10 11...18

            S0 = _mm256_permute4x64_epi64(S0, 0x00D8);//0...7 16...23 8...15 24...31
            S1 = _mm256_permute4x64_epi64(S1, 0x00D8);//1...8 17...24 9...16 25...32
            S2 = _mm256_permute4x64_epi64(S2, 0x00D8);
            S3 = _mm256_permute4x64_epi64(S3, 0x00D8);

            t0 = _mm256_unpacklo_epi8(S0, S1);//0 1 1 2 2 3 3 4  4 5 5 6 6 7 7  8     8  9  9 10 10 11 11 12  12 13 13 14 14 15 15 16
            t1 = _mm256_unpacklo_epi8(S2, S3);//2 3 3 4 4 5 5 6  6 7 7 8 8 9 9 10    10 11 11 12 12 13 13 14  14 15 15 16 16 17 17 18
            t0 = _mm256_permute4x64_epi64(t0, 0x00D8);
            t1 = _mm256_permute4x64_epi64(t1, 0x00D8);
            t2 = _mm256_unpacklo_epi16(t0, t1);//0...7
            t3 = _mm256_unpackhi_epi16(t0, t1);//8...15

            t0 = _mm256_maddubs_epi16(t2, c0);
            t1 = _mm256_maddubs_epi16(t3, c0);

            D0 = _mm256_hadds_epi16(t0, t1);//0 1 2 3 8 9 10 11    4 5 6 7 12 13 14 15
            D0 = _mm256_permute4x64_epi64(D0, 0x00D8);
            D0 = _mm256_add_epi16(D0, off);
            D0 = _mm256_srli_epi16(D0, 7);

            D0 = _mm256_packus_epi16(D0, D0);
            D0 = _mm256_permute4x64_epi64(D0, 0x00D8);
            _mm256_maskstore_epi64((__int64*)dst, mask, D0);

            dst += i_dst;
        }

    } else {

        __m256i S0, S1, S2, S3;
        __m256i t0, t1, t2, t3;
        __m256i c0;
        __m256i D0, D1;
        __m256i off = _mm256_set1_epi16(64);

        for (j = 0; j < bsy; j++) {
            int idx = (j + 1) >> 3;
            c0 = _mm256_set1_epi32(((int*)tab_coeff_mode_11[j & 0x07])[0]);
            for (i = 0; i < bsx; i += 32, idx += 32) {
                S0 = _mm256_loadu_si256((__m256i*)(src + idx));    //0...7 8...15 16...23 24...31
                S1 = _mm256_loadu_si256((__m256i*)(src + idx + 1));//1.. 8 9...16 17...24 25...32
                S2 = _mm256_loadu_si256((__m256i*)(src + idx + 2));//2...9 10...17 18
                S3 = _mm256_loadu_si256((__m256i*)(src + idx + 3));//3...10 11...18 19

                S0 = _mm256_permute4x64_epi64(S0, 0x00D8);//0...7 16...23 8...15 24...31
                S1 = _mm256_permute4x64_epi64(S1, 0x00D8);//1...8 17...24 9...16 25...32
                S2 = _mm256_permute4x64_epi64(S2, 0x00D8);
                S3 = _mm256_permute4x64_epi64(S3, 0x00D8);

                t0 = _mm256_unpacklo_epi8(S0, S1);//0 1 1 2 2 3 3 4  4 5 5 6 6 7 7  8     8  9  9 10 10 11 11 12  12 13 13 14 14 15 15 16
                t1 = _mm256_unpacklo_epi8(S2, S3);//2 3 3 4 4 5 5 6  6 7 7 8 8 9 9 10    10 11 11 12 12 13 13 14  14 15 15 16 16 17 17 18
                t0 = _mm256_permute4x64_epi64(t0, 0x00D8);
                t1 = _mm256_permute4x64_epi64(t1, 0x00D8);
                t2 = _mm256_unpacklo_epi16(t0, t1);//
                t3 = _mm256_unpackhi_epi16(t0, t1);//........15 16 17 18

                t0 = _mm256_maddubs_epi16(t2, c0);
                t1 = _mm256_maddubs_epi16(t3, c0);

                D0 = _mm256_hadds_epi16(t0, t1);//0 1 2 3 8 9 10 11    4 5 6 7 12 13 14 15
                D0 = _mm256_permute4x64_epi64(D0, 0x00D8);
                D0 = _mm256_add_epi16(D0, off);
                D0 = _mm256_srli_epi16(D0, 7);

                t0 = _mm256_unpackhi_epi8(S0, S1);//16 17 17 18  18 19 19 20  20 21 21 22 22 23 23 24...24 25 25..
                t1 = _mm256_unpackhi_epi8(S2, S3);//18 19 19 20  .....
                t0 = _mm256_permute4x64_epi64(t0, 0x00D8);
                t1 = _mm256_permute4x64_epi64(t1, 0x00D8);
                t2 = _mm256_unpacklo_epi16(t0, t1);//16 17 18 19...
                t3 = _mm256_unpackhi_epi16(t0, t1);//24 25 26 27...

                t0 = _mm256_maddubs_epi16(t2, c0);
                t1 = _mm256_maddubs_epi16(t3, c0);

                D1 = _mm256_hadds_epi16(t0, t1);//16 17 18 19 24 25 26 27    20 21 22 23 28 29 30 31
                D1 = _mm256_permute4x64_epi64(D1, 0x00D8);
                D1 = _mm256_add_epi16(D1, off);
                D1 = _mm256_srli_epi16(D1, 7);

                D0 = _mm256_packus_epi16(D0, D1);
                D0 = _mm256_permute4x64_epi64(D0, 0x00D8);
                _mm256_storeu_si256((__m256i*)(dst + i), D0);

            }
            dst += i_dst;
        }

    }

}

void intra_pred_ang_y_25_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    UNUSED_PARAMETER(dir_mode);
    int i;

    if (bsx > 8) {

        ALIGN32(pel_t first_line[64 + (64 << 3)]);
        int line_size = bsx + ((bsy - 1) << 3);
        int iHeight8 = bsy << 3;
        pel_t *pfirst = first_line;

        __m256i coeff0 = _mm256_setr_epi16( 7,  3,  5,  1,  3,  1,  1,  0,    7,  3,  5,  1,  3,  1,  1,  0);
        __m256i coeff1 = _mm256_setr_epi16(15,  7, 13,  3, 11,  5,  9,  1,   15,  7, 13,  3, 11,  5,  9,  1);
        __m256i coeff2 = _mm256_setr_epi16( 9,  5, 11,  3, 13,  7, 15,  2,    9,  5, 11,  3, 13,  7, 15,  2);
        __m256i coeff3 = _mm256_setr_epi16( 1,  1,  3,  1,  5,  3,  7,  1,    1,  1,  3,  1,  5,  3,  7,  1);
        __m256i coeff4 = _mm256_setr_epi16(16,  8, 16,  4, 16,  8, 16,  2,   16,  8, 16,  4, 16,  8, 16,  2);
        __m256i coeff5 = _mm256_setr_epi16( 1,  2,  1,  4,  1,  2,  1,  8,    1,  2,  1,  4,  1,  2,  1,  8);

        __m256i p00, p10, p20, p30;
        __m256i p01, p11, p21, p31;
        __m256i res1, res2;
        __m256i L0 = _mm256_setr_epi16(src[0],  src[0],  src[0],  src[0],  src[0],  src[0],  src[0],  src[0],
                                       src[-4], src[-4], src[-4], src[-4], src[-4], src[-4], src[-4], src[-4]);

        __m256i L1 = _mm256_setr_epi16(src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1],
                                       src[-5], src[-5], src[-5], src[-5], src[-5], src[-5], src[-5], src[-5]);

        __m256i L2 = _mm256_setr_epi16(src[-2], src[-2], src[-2], src[-2], src[-2], src[-2], src[-2], src[-2],
                                       src[-6], src[-6], src[-6], src[-6], src[-6], src[-6], src[-6], src[-6]);

        __m256i L3 = _mm256_setr_epi16(src[-3], src[-3], src[-3], src[-3], src[-3], src[-3], src[-3], src[-3],
                                       src[-7], src[-7], src[-7], src[-7], src[-7], src[-7], src[-7], src[-7]);

        src -= 4;

        for (i = 0; i < line_size; i += 64, src -= 4) {
            p00 = _mm256_mullo_epi16(L0, coeff0);//0...4...
            p10 = _mm256_mullo_epi16(L1, coeff1);//1...5...
            p20 = _mm256_mullo_epi16(L2, coeff2);//2...6...
            p30 = _mm256_mullo_epi16(L3, coeff3);//3...7...
            p00 = _mm256_add_epi16(p00, coeff4);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_mullo_epi16(p00, coeff5);
            p00 = _mm256_srli_epi16(p00, 5);

            L0 = _mm256_setr_epi16(src[0], src[0], src[0], src[0], src[0], src[0], src[0], src[0],
                                   src[-4], src[-4], src[-4], src[-4], src[-4], src[-4], src[-4], src[-4]);//4 8

            p01 = _mm256_mullo_epi16(L1, coeff0);//1...5...
            p11 = _mm256_mullo_epi16(L2, coeff1);//2...6...
            p21 = _mm256_mullo_epi16(L3, coeff2);//3...7...
            p31 = _mm256_mullo_epi16(L0, coeff3);//4...8...
            p01 = _mm256_add_epi16(p01, coeff4);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_mullo_epi16(p01, coeff5);
            p01 = _mm256_srli_epi16(p01, 5);

            res1 = _mm256_packus_epi16(p00, p01);


            L1 = _mm256_setr_epi16(src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1],
                                   src[-5], src[-5], src[-5], src[-5], src[-5], src[-5], src[-5], src[-5]);//5 9

            p00 = _mm256_mullo_epi16(L2, coeff0);//2...6...
            p10 = _mm256_mullo_epi16(L3, coeff1);//3...7...
            p20 = _mm256_mullo_epi16(L0, coeff2);//4...8...
            p30 = _mm256_mullo_epi16(L1, coeff3);//5...9...
            p00 = _mm256_add_epi16(p00, coeff4);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_mullo_epi16(p00, coeff5);
            p00 = _mm256_srli_epi16(p00, 5);

            L2 = _mm256_setr_epi16(src[-2], src[-2], src[-2], src[-2], src[-2], src[-2], src[-2], src[-2],
                                   src[-6], src[-6], src[-6], src[-6], src[-6], src[-6], src[-6], src[-6]);//6 10

            p01 = _mm256_mullo_epi16(L3, coeff0);//3...7...
            p11 = _mm256_mullo_epi16(L0, coeff1);//4...8...
            p21 = _mm256_mullo_epi16(L1, coeff2);//5...9...
            p31 = _mm256_mullo_epi16(L2, coeff3);//6...10...
            p01 = _mm256_add_epi16(p01, coeff4);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_mullo_epi16(p01, coeff5);
            p01 = _mm256_srli_epi16(p01, 5);

            res2 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute2x128_si256(res1, res2, 0x0020);
            _mm256_storeu_si256((__m256i*)pfirst, p00);
            pfirst += 32;

            p00 = _mm256_permute2x128_si256(res1, res2, 0x0031);
            _mm256_storeu_si256((__m256i*)pfirst, p00);

            pfirst += 32;

            src -= 4;
            L0 = _mm256_setr_epi16(src[0], src[0], src[0], src[0], src[0], src[0], src[0], src[0],
                                   src[-4], src[-4], src[-4], src[-4], src[-4], src[-4], src[-4], src[-4]);//8 12

            L1 = _mm256_setr_epi16(src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1],
                                   src[-5], src[-5], src[-5], src[-5], src[-5], src[-5], src[-5], src[-5]);//9 13

            L2 = _mm256_setr_epi16(src[-2], src[-2], src[-2], src[-2], src[-2], src[-2], src[-2], src[-2],
                                   src[-6], src[-6], src[-6], src[-6], src[-6], src[-6], src[-6], src[-6]);//10 14

            L3 = _mm256_setr_epi16(src[-3], src[-3], src[-3], src[-3], src[-3], src[-3], src[-3], src[-3],
                                   src[-7], src[-7], src[-7], src[-7], src[-7], src[-7], src[-7], src[-7]);//11 15

        }

        //if (bsx == 16) {// 8个
        //    __m256i mask = _mm256_loadu_si256((__m256i*)intrinsic_mask_256_8bit[7]);
        //    p00 = _mm256_mullo_epi16(L0, coeff0);
        //    p10 = _mm256_mullo_epi16(L1, coeff1);
        //    p20 = _mm256_mullo_epi16(L2, coeff2);
        //    p30 = _mm256_mullo_epi16(L3, coeff3);
        //    p00 = _mm256_add_epi16(p00, coeff4);
        //    p00 = _mm256_add_epi16(p00, p10);
        //    p00 = _mm256_add_epi16(p00, p20);
        //    p00 = _mm256_add_epi16(p00, p30);
        //    p00 = _mm256_mullo_epi16(p00, coeff5);
        //    p00 = _mm256_srli_epi16(p00, 5);
        //
        //    p00 = _mm256_packus_epi16(p00, p00);
        //    p00 = _mm256_permute4x64_epi64(p00, 0x0008);
        //    _mm256_maskstore_epi64((__m256i*)pfirst, mask, p00);
        //} else if(bsx == 32){
        //    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
        //    p00 = _mm256_mullo_epi16(L0, coeff0);
        //    p10 = _mm256_mullo_epi16(L1, coeff1);
        //    p20 = _mm256_mullo_epi16(L2, coeff2);
        //    p30 = _mm256_mullo_epi16(L3, coeff3);
        //    p00 = _mm256_add_epi16(p00, coeff4);
        //    p00 = _mm256_add_epi16(p00, p10);
        //    p00 = _mm256_add_epi16(p00, p20);
        //    p00 = _mm256_add_epi16(p00, p30);
        //    p00 = _mm256_mullo_epi16(p00, coeff5);
        //    p00 = _mm256_srli_epi16(p00, 5);
        //
        //    L0 = _mm256_setr_epi16(src[0], src[0], src[0], src[0], src[0], src[0], src[0], src[0],
        //        src[-4], src[-4], src[-4], src[-4], src[-4], src[-4], src[-4], src[-4]);
        //
        //    p01 = _mm256_mullo_epi16(L1, coeff0);
        //    p11 = _mm256_mullo_epi16(L2, coeff1);
        //    p21 = _mm256_mullo_epi16(L3, coeff2);
        //    p31 = _mm256_mullo_epi16(L0, coeff3);
        //    p01 = _mm256_add_epi16(p01, coeff4);
        //    p01 = _mm256_add_epi16(p01, p11);
        //    p01 = _mm256_add_epi16(p01, p21);
        //    p01 = _mm256_add_epi16(p01, p31);
        //    p01 = _mm256_mullo_epi16(p01, coeff5);
        //    p01 = _mm256_srli_epi16(p01, 5);
        //
        //    p00 = _mm256_packus_epi16(p00, p01);
        //    p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
        //    _mm256_maskstore_epi64((__int64*)pfirst, mask, p00);
        //
        //} else {
        //    __m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
        //    p00 = _mm256_mullo_epi16(L0, coeff0);
        //    p10 = _mm256_mullo_epi16(L1, coeff1);
        //    p20 = _mm256_mullo_epi16(L2, coeff2);
        //    p30 = _mm256_mullo_epi16(L3, coeff3);
        //    p00 = _mm256_add_epi16(p00, coeff4);
        //    p00 = _mm256_add_epi16(p00, p10);
        //    p00 = _mm256_add_epi16(p00, p20);
        //    p00 = _mm256_add_epi16(p00, p30);
        //    p00 = _mm256_mullo_epi16(p00, coeff5);
        //    p00 = _mm256_srli_epi16(p00, 5);
        //
        //    L0 = _mm256_setr_epi16(src[0], src[0], src[0], src[0], src[0], src[0], src[0], src[0],
        //        src[-4], src[-4], src[-4], src[-4], src[-4], src[-4], src[-4], src[-4]);
        //
        //    p01 = _mm256_mullo_epi16(L1, coeff0);
        //    p11 = _mm256_mullo_epi16(L2, coeff1);
        //    p21 = _mm256_mullo_epi16(L3, coeff2);
        //    p31 = _mm256_mullo_epi16(L0, coeff3);
        //    p01 = _mm256_add_epi16(p01, coeff4);
        //    p01 = _mm256_add_epi16(p01, p11);
        //    p01 = _mm256_add_epi16(p01, p21);
        //    p01 = _mm256_add_epi16(p01, p31);
        //    p01 = _mm256_mullo_epi16(p01, coeff5);
        //    p01 = _mm256_srli_epi16(p01, 5);
        //
        //    p00 = _mm256_packus_epi16(p00, p01);
        //    p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
        //    _mm256_storeu_si256((__m256*)pfirst, p00);
        //
        //    pfirst += 32;
        //
        //    L1 = _mm256_setr_epi16(src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1],
        //        src[-5], src[-5], src[-5], src[-5], src[-5], src[-5], src[-5], src[-5]);
        //
        //    p00 = _mm256_mullo_epi16(L2, coeff0);
        //    p10 = _mm256_mullo_epi16(L3, coeff1);
        //    p20 = _mm256_mullo_epi16(L0, coeff2);
        //    p30 = _mm256_mullo_epi16(L1, coeff3);
        //    p00 = _mm256_add_epi16(p00, coeff4);
        //    p00 = _mm256_add_epi16(p00, p10);
        //    p00 = _mm256_add_epi16(p00, p20);
        //    p00 = _mm256_add_epi16(p00, p30);
        //    p00 = _mm256_mullo_epi16(p00, coeff5);
        //    p00 = _mm256_srli_epi16(p00, 5);
        //
        //    p00 = _mm256_packus_epi16(p00, p00);
        //    p00 = _mm256_permute4x64_epi64(p00, 0x0008);
        //    _mm256_maskstore_epi64((__int64*)pfirst, mask, p00);
        //
        //}

        __m256i M;

        if (bsx == 64) {
            for (i = 0; i < iHeight8; i += 32) {
                M = _mm256_lddqu_si256((__m256i*)(first_line + i));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 8));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(first_line + i + +8 + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 16));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 16 + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 24));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 24 + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;
            }
        } else if (bsx == 32) {
            for (i = 0; i < iHeight8; i += 32) {
                M = _mm256_lddqu_si256((__m256i*)(first_line + i));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 8));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 16));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 24));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;
            }
        } else if (bsx == 16) {
            __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
            for (i = 0; i < iHeight8; i += 32) {
                M = _mm256_lddqu_si256((__m256i*)(first_line + i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 8));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 16));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 24));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
            }
        }

        /*for (i = 0; i < iHeight8; i += 8) {
            memcpy(dst, first_line + i, bsx * sizeof(pel_t));
            dst += i_dst;
        }*/
    } else {//8x8 8x32 4x4 4x16
        intra_pred_ang_y_25_sse128(src, dst, i_dst, dir_mode, bsx, bsy);
        return ;
    }

}

void intra_pred_ang_y_26_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;

    UNUSED_PARAMETER(dir_mode);

    if (bsx != 4) {
        __m256i coeff2 = _mm256_set1_epi16(2);
        __m256i coeff3 = _mm256_set1_epi16(3);
        __m256i coeff4 = _mm256_set1_epi16(4);
        __m256i coeff5 = _mm256_set1_epi16(5);
        __m256i coeff7 = _mm256_set1_epi16(7);
        __m256i coeff8 = _mm256_set1_epi16(8);
        __m256i shuffle = _mm256_setr_epi8(7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 0, 8,
                                           7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 0, 8);

        ALIGN32(pel_t first_line[64 + 256]);
        int line_size = bsx + (bsy - 1) * 4;
        int iHeight4 = bsy << 2;

        src -= 31;
        __m256i p00, p10, p20, p30;
        __m256i p01, p11, p21, p31;
        __m256i M1, M2, M3, M4, M5, M6, M7, M8;
        __m256i S0, S1, S2, S3;
        __m256i L0, L1, L2, L3;
        __m256i H0, H1, H2, H3;


        for (i = 0; i < line_size - 64; i += 128, src -= 32) {

            S0 = _mm256_loadu_si256((__m256i*)(src));    //15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
            S1 = _mm256_loadu_si256((__m256i*)(src - 1));//16 15 14...
            S2 = _mm256_loadu_si256((__m256i*)(src - 2));//17 16 15...
            S3 = _mm256_loadu_si256((__m256i*)(src - 3));//18 17 16...

            L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));//15 14 13 12 11 10 9 8
            L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));//16 15 14...
            L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));//17 16 15...
            L3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 0));//18 17 16...

            H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));//7 6 5 4 3 2 1 0
            H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));//8 7 6..
            H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));//9 8 7...
            H3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 1));//10 9 8...

            p00 = _mm256_mullo_epi16(L0, coeff3);
            p10 = _mm256_mullo_epi16(L1, coeff7);
            p20 = _mm256_mullo_epi16(L2, coeff5);
            p30 = _mm256_add_epi16(L3, coeff8);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            M1  = _mm256_srli_epi16(p00, 4);//31...16

            p01 = _mm256_mullo_epi16(H0, coeff3);
            p11 = _mm256_mullo_epi16(H1, coeff7);
            p21 = _mm256_mullo_epi16(H2, coeff5);
            p31 = _mm256_add_epi16(H3, coeff8);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            M2  = _mm256_srli_epi16(p01, 4);//15...0

            p00 = _mm256_add_epi16(L1, L2);
            p00 = _mm256_mullo_epi16(p00, coeff3);
            p10 = _mm256_add_epi16(L0, L3);
            p10 = _mm256_add_epi16(p10, coeff4);
            p00 = _mm256_add_epi16(p10, p00);
            M3  = _mm256_srli_epi16(p00, 3);

            p01 = _mm256_add_epi16(H1, H2);
            p01 = _mm256_mullo_epi16(p01, coeff3);
            p11 = _mm256_add_epi16(H0, H3);
            p11 = _mm256_add_epi16(p11, coeff4);
            p01 = _mm256_add_epi16(p11, p01);
            M4  = _mm256_srli_epi16(p01, 3);

            p10 = _mm256_mullo_epi16(L1, coeff5);
            p20 = _mm256_mullo_epi16(L2, coeff7);
            p30 = _mm256_mullo_epi16(L3, coeff3);
            p00 = _mm256_add_epi16(L0, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            M5  = _mm256_srli_epi16(p00, 4);//31...16

            p11 = _mm256_mullo_epi16(H1, coeff5);
            p21 = _mm256_mullo_epi16(H2, coeff7);
            p31 = _mm256_mullo_epi16(H3, coeff3);
            p01 = _mm256_add_epi16(H0, coeff8);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            M6  = _mm256_srli_epi16(p01, 4);//15...0

            p00 = _mm256_add_epi16(L1, L2);
            p10 = _mm256_add_epi16(L2, L3);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, coeff2);
            M7  = _mm256_srli_epi16(p00, 2);

            p01 = _mm256_add_epi16(H1, H2);
            p11 = _mm256_add_epi16(H2, H3);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, coeff2);
            M8  = _mm256_srli_epi16(p01, 2);

            M1 = _mm256_packus_epi16(M1, M3);
            M5 = _mm256_packus_epi16(M5, M7);
            M1 = _mm256_shuffle_epi8(M1, shuffle);
            M5 = _mm256_shuffle_epi8(M5, shuffle);

            M2 = _mm256_packus_epi16(M2, M4);
            M6 = _mm256_packus_epi16(M6, M8);
            M2 = _mm256_shuffle_epi8(M2, shuffle);
            M6 = _mm256_shuffle_epi8(M6, shuffle);

            //M1 = _mm256_permute4x64_epi64(M1, 0x4E);
            //M5 = _mm256_permute4x64_epi64(M5, 0x4E);
            //M2 = _mm256_permute4x64_epi64(M2, 0x4E);
            //M6 = _mm256_permute4x64_epi64(M6, 0x4E);

            M1 = _mm256_permute4x64_epi64(M1, 0x72);
            M5 = _mm256_permute4x64_epi64(M5, 0x72);
            M2 = _mm256_permute4x64_epi64(M2, 0x72);
            M6 = _mm256_permute4x64_epi64(M6, 0x72);

            M3 = _mm256_unpacklo_epi16(M1, M5);
            M7 = _mm256_unpackhi_epi16(M1, M5);
            M4 = _mm256_unpacklo_epi16(M2, M6);
            M8 = _mm256_unpackhi_epi16(M2, M6);

            _mm256_storeu_si256((__m256i*)&first_line[i], M4);
            _mm256_storeu_si256((__m256i*)&first_line[32 + i], M8);
            _mm256_storeu_si256((__m256i*)&first_line[64 + i], M3);
            _mm256_storeu_si256((__m256i*)&first_line[96 + i], M7);
        }

        if (i < line_size) {
            S0 = _mm256_loadu_si256((__m256i*)(src));    //15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
            S1 = _mm256_loadu_si256((__m256i*)(src - 1));//16 15 14...
            S2 = _mm256_loadu_si256((__m256i*)(src - 2));//17 16 15...
            S3 = _mm256_loadu_si256((__m256i*)(src - 3));//18 17 16...

            H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));//7 6 5 4 3 2 1 0
            H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));//8 7 6..
            H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));//9 8 7...
            H3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 1));//10 9 8...

            p01 = _mm256_mullo_epi16(H0, coeff3);
            p11 = _mm256_mullo_epi16(H1, coeff7);
            p21 = _mm256_mullo_epi16(H2, coeff5);
            p31 = _mm256_add_epi16(H3, coeff8);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            M2 = _mm256_srli_epi16(p01, 4);//15...0

            p01 = _mm256_add_epi16(H1, H2);
            p01 = _mm256_mullo_epi16(p01, coeff3);
            p11 = _mm256_add_epi16(H0, H3);
            p11 = _mm256_add_epi16(p11, coeff4);
            p01 = _mm256_add_epi16(p11, p01);
            M4 = _mm256_srli_epi16(p01, 3);

            p11 = _mm256_mullo_epi16(H1, coeff5);
            p21 = _mm256_mullo_epi16(H2, coeff7);
            p31 = _mm256_mullo_epi16(H3, coeff3);
            p01 = _mm256_add_epi16(H0, coeff8);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            M6 = _mm256_srli_epi16(p01, 4);//15...0

            p01 = _mm256_add_epi16(H1, H2);
            p11 = _mm256_add_epi16(H2, H3);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, coeff2);
            M8 = _mm256_srli_epi16(p01, 2);

            M2 = _mm256_packus_epi16(M2, M4);
            M6 = _mm256_packus_epi16(M6, M8);
            M2 = _mm256_shuffle_epi8(M2, shuffle);
            M6 = _mm256_shuffle_epi8(M6, shuffle);

            //M2 = _mm256_permute4x64_epi64(M2, 0x4E);
            //M6 = _mm256_permute4x64_epi64(M6, 0x4E);

            M2 = _mm256_permute4x64_epi64(M2, 0x72);
            M6 = _mm256_permute4x64_epi64(M6, 0x72);

            M4 = _mm256_unpacklo_epi16(M2, M6);
            M8 = _mm256_unpackhi_epi16(M2, M6);

            _mm256_storeu_si256((__m256i*)&first_line[i], M4);
            _mm256_storeu_si256((__m256i*)&first_line[32 + i], M8);
        }

        __m256i M;
        if (bsx == 64) {
            for (i = 0; i < iHeight4; i += 16) {
                M = _mm256_lddqu_si256((__m256i*)(first_line + i));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 4));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 32 + 4));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 8));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 32 + 8));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 12));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 32 + 12));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;
            }
        } else if (bsx == 32) {
            for (i = 0; i < iHeight4; i += 16) {
                M = _mm256_lddqu_si256((__m256i*)(first_line + i));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 4));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 8));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 12));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;
            }
        } else if (bsx == 16) {
            __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
            for (i = 0; i < iHeight4; i += 16) {

                M = _mm256_lddqu_si256((__m256i*)(first_line + i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 4));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 8));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 12));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

            }
        } else {
            __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[7]);
            for (i = 0; i < iHeight4; i += 16) {
                M = _mm256_lddqu_si256((__m256i*)(first_line + i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 4));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 8));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(first_line + i + 12));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
            }
        }

        /*switch (bsx) {
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
        }*/
    } else { //4x4 4x16
        intra_pred_ang_y_26_sse128(src, dst, i_dst, dir_mode, bsx, bsy);
        return;
    }

}

void intra_pred_ang_y_28_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN32(pel_t first_line[64 + 128]);
    int line_size = bsx + (bsy - 1) * 2;

    int i;
    int iHeight2 = bsy << 1;
    UNUSED_PARAMETER(dir_mode);

    __m256i coeff2 = _mm256_set1_epi16(2);
    __m256i coeff3 = _mm256_set1_epi16(3);
    __m256i coeff4 = _mm256_set1_epi16(4);
    __m256i shuffle = _mm256_setr_epi8(7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 0, 8,
                                       7, 15, 6, 14, 5, 13, 4, 12, 3, 11, 2, 10, 1, 9, 0, 8);

    src -= 31;
    __m256i p00, p10;
    __m256i p01, p11;
    __m256i S0, S1, S2, S3;
    __m256i L0, L1, L2, L3;
    __m256i H0, H1, H2, H3;
    for (i = 0; i < line_size - 32; i += 64, src -= 32) {
        S0 = _mm256_loadu_si256((__m256i*)(src));
        S3 = _mm256_loadu_si256((__m256i*)(src - 3));
        S1 = _mm256_loadu_si256((__m256i*)(src - 1));
        S2 = _mm256_loadu_si256((__m256i*)(src - 2));

        L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));//15 14 13 12 11 10 9 8
        L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));//16 15 14...
        L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));//17 16 15...
        L3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 0));//18 17 16...

        H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));//7 6 5 4 3 2 1 0
        H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));//8 7 6..
        H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));//9 8 7...
        H3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 1));//10 9 8...

        p00 = _mm256_adds_epi16(L1, L2);
        p00 = _mm256_mullo_epi16(p00, coeff3);
        p10 = _mm256_add_epi16(L0, L3);
        p10 = _mm256_add_epi16(p10, coeff4);
        p00 = _mm256_add_epi16(p00, p10);
        p00 = _mm256_srli_epi16(p00, 3);//031...016

        p01 = _mm256_add_epi16(L1, L2);
        p11 = _mm256_add_epi16(L2, L3);
        p01 = _mm256_add_epi16(p01, p11);
        p01 = _mm256_add_epi16(p01, coeff2);
        p01 = _mm256_srli_epi16(p01, 2);//131...116

        p00 = _mm256_packus_epi16(p00, p01);//
        p00 = _mm256_shuffle_epi8(p00, shuffle);
        p00 = _mm256_permute4x64_epi64(p00, 0x4E);
        _mm256_storeu_si256((__m256i*)&first_line[i + 32], p00);

        p00 = _mm256_adds_epi16(H1, H2);
        p00 = _mm256_mullo_epi16(p00, coeff3);
        p10 = _mm256_adds_epi16(H0, H3);
        p10 = _mm256_adds_epi16(p10, coeff4);
        p00 = _mm256_adds_epi16(p00, p10);
        p00 = _mm256_srli_epi16(p00, 3);

        p01 = _mm256_add_epi16(H1, H2);
        p11 = _mm256_add_epi16(H2, H3);
        p01 = _mm256_add_epi16(p01, p11);
        p01 = _mm256_add_epi16(p01, coeff2);
        p01 = _mm256_srli_epi16(p01, 2);

        p00 = _mm256_packus_epi16(p00, p01);
        p00 = _mm256_shuffle_epi8(p00, shuffle);
        p00 = _mm256_permute4x64_epi64(p00, 0x4E);
        _mm256_storeu_si256((__m256i*)&first_line[i], p00);
    }

    if (i < line_size) {
        S0 = _mm256_loadu_si256((__m256i*)(src));
        S3 = _mm256_loadu_si256((__m256i*)(src - 3));
        S1 = _mm256_loadu_si256((__m256i*)(src - 1));
        S2 = _mm256_loadu_si256((__m256i*)(src - 2));

        H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));//7 6 5 4 3 2 1 0
        H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));//8 7 6..
        H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));//9 8 7...
        H3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 1));//10 9 8...

        p00 = _mm256_adds_epi16(H1, H2);
        p00 = _mm256_mullo_epi16(p00, coeff3);
        p10 = _mm256_adds_epi16(H0, H3);
        p10 = _mm256_adds_epi16(p10, coeff4);
        p00 = _mm256_adds_epi16(p00, p10);
        p00 = _mm256_srli_epi16(p00, 3);

        p01 = _mm256_add_epi16(H1, H2);
        p11 = _mm256_add_epi16(H2, H3);
        p01 = _mm256_add_epi16(p01, p11);
        p01 = _mm256_add_epi16(p01, coeff2);
        p01 = _mm256_srli_epi16(p01, 2);

        p00 = _mm256_packus_epi16(p00, p01);
        p00 = _mm256_shuffle_epi8(p00, shuffle);
        p00 = _mm256_permute4x64_epi64(p00, 0x4E);
        _mm256_storeu_si256((__m256i*)&first_line[i], p00);
    }

    if (bsx == 64) {
        for (i = 0; i < iHeight2; i += 8) {
            __m256i M = _mm256_lddqu_si256((__m256i*)&first_line[i]);
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(&first_line[i] + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 2]);
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(&first_line[i + 2] + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 4]);
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(&first_line[i + 4] + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 6]);
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(&first_line[i + 6] + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
        }

    } else if (bsx == 32) {
        for (i = 0; i < iHeight2; i += 8) {
            __m256i M = _mm256_lddqu_si256((__m256i*)&first_line[i]);
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 2]);
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 4]);
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 6]);
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
        }
    } else if (bsx == 16) {
        __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
        for (i = 0; i < iHeight2; i += 8) {
            __m256i M = _mm256_lddqu_si256((__m256i*)&first_line[i]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 2]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 4]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 6]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

        }
    } else if (bsx == 8) {
        __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[7]);
        for (i = 0; i < iHeight2; i += 8) {
            __m256i M = _mm256_lddqu_si256((__m256i*)&first_line[i]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 2]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 4]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 6]);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

        }
    } else {
        __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[3]);
        for (i = 0; i < iHeight2; i += 8) {
            __m256i M = _mm256_lddqu_si256((__m256i*)&first_line[i]);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 2]);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 4]);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)&first_line[i + 6]);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;

        }
    }



    /*if (bsx >= 16) {

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
    }*/
}

void intra_pred_ang_y_30_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN32(pel_t first_line[64 + 64]);
    int line_size = bsx + bsy - 1;
    UNUSED_PARAMETER(dir_mode);
    int i;

    __m256i coeff2 = _mm256_set1_epi16(2);
    __m256i shuffle = _mm256_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                       15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    __m256i p00, p10;
    __m256i p01, p11;
    __m256i S0, S1, S2;
    __m256i L0, L1, L2;
    __m256i H0, H1, H2;

    src -= 33;

    for (i = 0; i < line_size - 16; i += 32, src -= 32) {

        S0 = _mm256_loadu_si256((__m256i*)(src - 1));
        S1 = _mm256_loadu_si256((__m256i*)(src));
        S2 = _mm256_loadu_si256((__m256i*)(src + 1));

        L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));//35 34 33...
        L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));//34 33 32...
        L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));

        H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));//20 19 18...
        H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));//19 18 17...
        H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));

        p00 = _mm256_add_epi16(L0, L1);
        p10 = _mm256_add_epi16(L1, L2);
        p00 = _mm256_add_epi16(p00, p10);
        p00 = _mm256_add_epi16(p00, coeff2);
        p00 = _mm256_srli_epi16(p00, 2);//31...24 23...16

        p01 = _mm256_add_epi16(H0, H1);
        p11 = _mm256_add_epi16(H1, H2);
        p01 = _mm256_add_epi16(p01, p11);
        p01 = _mm256_add_epi16(p01, coeff2);
        p01 = _mm256_srli_epi16(p01, 2);//15..8 7...0

        p00 = _mm256_packus_epi16(p00, p01);//32...24 15...8 23...16 7...0
        p00 = _mm256_permute4x64_epi64(p00, 0x8D);
        p00 = _mm256_shuffle_epi8(p00, shuffle);

        _mm256_storeu_si256((__m256i*)&first_line[i], p00);

    }

    __m256i mask = _mm256_load_si256((__m256i*)intrinsic_mask_256_8bit[15]);

    if (i < line_size) {
        S0 = _mm256_loadu_si256((__m256i*)(src - 1));
        S1 = _mm256_loadu_si256((__m256i*)(src));
        S2 = _mm256_loadu_si256((__m256i*)(src + 1));

        H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));//20 19 18...
        H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));//19 18 17...
        H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));//18

        p01 = _mm256_add_epi16(H0, H1);
        p11 = _mm256_add_epi16(H1, H2);
        p01 = _mm256_add_epi16(p01, p11);
        p01 = _mm256_add_epi16(p01, coeff2);
        p01 = _mm256_srli_epi16(p01, 2);//15...8..7..0

        p01 = _mm256_packus_epi16(p01, p01);//15...8 15...8 7...0 7...0
        p01 = _mm256_permute4x64_epi64(p01, 0x0008);
        p01 = _mm256_shuffle_epi8(p01, shuffle);

        _mm256_maskstore_epi64((__int64*)&first_line[i], mask, p01);
    }

    __m256i M;
    if (bsx == 64) {
        for (i = 0; i < bsy; i += 4) {
            M = _mm256_lddqu_si256((__m256i*)(first_line + i));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 1));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 32 + 1));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 2));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 32 + 2));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 3));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 32 + 3));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
        }
    } else if (bsx == 32) {
        for (i = 0; i < bsy; i += 4) {
            M = _mm256_lddqu_si256((__m256i*)(first_line + i));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 1));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 2));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 3));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
        }
    } else if (bsx == 16) {
        mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
        for (i = 0; i < bsy; i += 4) {

            M = _mm256_lddqu_si256((__m256i*)(first_line + i));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 1));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 2));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 3));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

        }
    } else if (bsx == 8) {
        mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[7]);
        for (i = 0; i < bsy; i += 4) {
            M = _mm256_lddqu_si256((__m256i*)(first_line + i));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 1));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 2));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 3));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
        }
    } else {
        mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[3]);
        for (i = 0; i < bsy; i += 4) {
            M = _mm256_lddqu_si256((__m256i*)(first_line + i));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 1));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 2));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(first_line + i + 3));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
        }
    }



    /*if (bsx > 16) {

        for (i = 0; i < bsy; i++) {
            memcpy(dst, first_line + i, bsx * sizeof(pel_t));
            dst += i_dst;
        }
    } else if (bsx == 16) {

        pel_t *dst1 = dst;

        if (bsy == 4) {
            __m256i M = _mm256_loadu_si256((__m256i*)&first_line[0]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[1]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[2]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[3]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);

        } else {
            __m256i M = _mm256_loadu_si256((__m256i*)&first_line[0]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[1]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[2]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[3]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[4]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[5]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[6]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[7]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[8]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[9]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[10]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[11]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[12]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[13]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[14]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
            dst1 += i_dst;
            M = _mm256_loadu_si256((__m256i*)&first_line[15]);
            _mm256_maskstore_epi64((__int64*)dst1, mask, M);
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
    }*/

}

void intra_pred_ang_y_31_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;
    UNUSED_PARAMETER(dir_mode);

    if (bsx >= bsy) {
        ALIGN32(pel_t dst_tran[MAX_CU_SIZE * MAX_CU_SIZE]);
        ALIGN32(pel_t src_tran[MAX_CU_SIZE << 3]);

        for (i = 0; i < (bsy + bsx * 11 / 8 + 3); i++) {
            src_tran[i] = src[-i];
        }
        intra_pred_ang_x_5_avx(src_tran, dst_tran, bsy, 5, bsy, bsx);
        for (i = 0; i < bsy; i++) {
            for (int j = 0; j < bsx; j++) {
                dst[j + i_dst * i] = dst_tran[i + bsy * j];
            }
        }
    } else if (bsx == 8) {

        __m128i coeff0 = _mm_setr_epi16( 5, 1,  7, 1,  1, 3,  3, 1);
        __m128i coeff1 = _mm_setr_epi16(13, 5, 15, 3,  9, 7, 11, 2);
        __m128i coeff2 = _mm_setr_epi16(11, 7,  9, 3, 15, 5, 13, 1);
        __m128i coeff3 = _mm_setr_epi16( 3, 3,  1, 1,  7, 1,  5, 0);
        __m128i coeff4 = _mm_setr_epi16(16, 8, 16, 4, 16, 8, 16, 2);
        __m128i coeff5 = _mm_setr_epi16( 1, 2,  1, 4,  1, 2,  1, 8);

        __m128i L0, L1, L2, L3;
        __m128i p00, p10, p20, p30;

        for (i = 0; i < bsy; i++,src--) {
            L0 = _mm_setr_epi16(src[-1], src[-2], src[-4], src[-5], src[-6], src[ -8], src[ -9], src[-11]);
            L1 = _mm_setr_epi16(src[-2], src[-3], src[-5], src[-6], src[-7], src[ -9], src[-10], src[-12]);
            L2 = _mm_setr_epi16(src[-3], src[-4], src[-6], src[-7], src[-8], src[-10], src[-11], src[-13]);
            L3 = _mm_setr_epi16(src[-4], src[-5], src[-7], src[-8], src[-9], src[-11], src[-12], src[-14]);

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
        }

    } else {
        intra_pred_ang_y_31_sse128(src, dst, i_dst, dir_mode, bsx, bsy);
    }


}

void intra_pred_ang_y_32_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN32(pel_t first_line[2 * (64 + 64)]);
    int line_size = (bsy >> 1) + bsx - 1;

    int i;
    int aligned_line_size = ((line_size + 63) >> 4) << 4;
    pel_t *pfirst[2];
    UNUSED_PARAMETER(dir_mode);

    __m256i coeff2 = _mm256_set1_epi16(2);
    __m256i shuffle = _mm256_setr_epi8(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0,
                                       15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);

    pfirst[0] = first_line;
    pfirst[1] = first_line + aligned_line_size;

    src -= 34;

    __m256i S0, S1, S2;
    __m256i L0, L1, L2;
    __m256i H0, H1, H2;
    __m256i p00, p01, p10, p11;

    __m256i mask = _mm256_load_si256((__m256i*)intrinsic_mask_256_8bit[15]);
    for (i = 0; i < line_size - 8; i += 16, src -= 32) {

        S0 = _mm256_loadu_si256((__m256i*)(src - 1));//19 18 17 16 15 14 13 12  11 10 9 8 7 6 5 4
        S1 = _mm256_loadu_si256((__m256i*)(src));    //18 17 16 15 14 13 12 11  10  9 8 7 6 5 4 3
        S2 = _mm256_loadu_si256((__m256i*)(src + 1));//17 16 15 14 13 12 11 10   9  8 7 6 5 4 3 2

        L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));//19 18 17 16 15 14 13 12
        L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));//18 17 16 15 14 13 12 11
        L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));//17 16 15 14 13 12 11 10

        H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));//11 10 9 8 7 6 5 4
        H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));//10  9 8 7 6 5 4 3
        H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));// 9  8 7 6 5 4 3 2

        p00 = _mm256_add_epi16(L0, L1);
        p01 = _mm256_add_epi16(L1, L2);
        p00 = _mm256_add_epi16(p00, coeff2);
        p00 = _mm256_add_epi16(p00, p01);
        p00 = _mm256_srli_epi16(p00, 2);//19...12(31...16)

        p10 = _mm256_add_epi16(H0, H1);
        p11 = _mm256_add_epi16(H1, H2);
        p10 = _mm256_add_epi16(p10, coeff2);
        p10 = _mm256_add_epi16(p10, p11);
        p10 = _mm256_srli_epi16(p10, 2);//11...4(15...0)

        //31...24 15...8 23...16 7...0
        p00 = _mm256_packus_epi16(p00, p10);     //19 18 17 16 15 14 13 12   11 10 9 8 7 6 5 4
        p00 = _mm256_permute4x64_epi64(p00, 0x8D);//31...16 15..0
        //0 2 4 6 8 10 12 14  1 3 5 7 9 11 13 15  16....
        p00 = _mm256_shuffle_epi8(p00, shuffle);
        p10 = _mm256_permute4x64_epi64(p00, 0x0D);
        p00 = _mm256_permute4x64_epi64(p00, 0x08);

        _mm256_maskstore_epi64((__int64*)&pfirst[0][i], mask, p00);
        _mm256_maskstore_epi64((__int64*)&pfirst[1][i], mask, p10);
    }

    mask = _mm256_load_si256((__m256i*)intrinsic_mask_256_8bit[8]);
    if (i < line_size) {
        S0 = _mm256_loadu_si256((__m256i*)(src - 1));//19 18 17 16 15 14 13 12  11 10 9 8 7 6 5 4
        S1 = _mm256_loadu_si256((__m256i*)(src));    //18 17 16 15 14 13 12 11  10  9 8 7 6 5 4 3
        S2 = _mm256_loadu_si256((__m256i*)(src + 1));//17 16 15 14 13 12 11 10   9  8 7 6 5 4 3 2

        H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));//11 10 9 8 7 6 5 4
        H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));//10  9 8 7 6 5 4 3
        H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));// 9  8 7 6 5 4 3 2

        p10 = _mm256_add_epi16(H0, H1);
        p11 = _mm256_add_epi16(H1, H2);
        p10 = _mm256_add_epi16(p10, coeff2);
        p10 = _mm256_add_epi16(p10, p11);
        p10 = _mm256_srli_epi16(p10, 2);

        //15...8 15...8 7...0 7...0
        p00 = _mm256_packus_epi16(p10, p10);     //19 18 17 16 15 14 13 12   11 10 9 8 7 6 5 4
        p00 = _mm256_permute4x64_epi64(p00, 0x8D);//15...0 15...0
        //0 2 4 6 8 10 12 14  1 3 5 7 1 3 5 7 8....
        p00 = _mm256_shuffle_epi8(p00, shuffle);
        p10 = _mm256_permute4x64_epi64(p00, 0x0D);
        p00 = _mm256_permute4x64_epi64(p00, 0x08);

        _mm256_maskstore_epi64((__int64*)&pfirst[0][i], mask, p00);
        _mm256_maskstore_epi64((__int64*)&pfirst[1][i], mask, p10);
        ;
    }
    bsy >>= 1;

    if (bsx == 64) {

        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 1));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 1 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 1));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 1 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 2));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 2 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 2));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 2 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 3));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 3 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 3));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 3 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

        }
    } else if (bsx == 32) {

        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 1));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 1));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 2));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 2));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 3));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 3));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
        }
    } else if (bsx == 16) {
        mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 1));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 1));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 2));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 2));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 3));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 3));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
        }
    } else if (bsx == 8) {
        mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 1));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 1));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 2));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 2));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 3));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 3));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
        }
    } else {
        mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 1));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 1));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 2));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 2));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] + i + 3));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] + i + 3));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
        }
    }


    /*if (bsx >= 16 || bsx == 4) {
        for (i = 0; i < bsy; i++) {
            memcpy(dst, pfirst[0] + i, bsx * sizeof(pel_t));
            memcpy(dst + i_dst, pfirst[1] + i, bsx * sizeof(pel_t));
            dst += i_dst2;
        }
    } else {

        if (bsy == 4) {//8x8
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
        } else {//8x32
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
    }*/
}

void intra_pred_ang_xy_13_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;
    UNUSED_PARAMETER(dir_mode);

    if (bsy > 4) {

        __m256i coeff2 = _mm256_set1_epi16(2);
        __m256i coeff3 = _mm256_set1_epi16(3);
        __m256i coeff4 = _mm256_set1_epi16(4);
        __m256i coeff5 = _mm256_set1_epi16(5);
        __m256i coeff7 = _mm256_set1_epi16(7);
        __m256i coeff8 = _mm256_set1_epi16(8);
        __m256i coeff9 = _mm256_set1_epi16(9);
        __m256i coeff11 = _mm256_set1_epi16(11);
        __m256i coeff13 = _mm256_set1_epi16(13);
        __m256i coeff15 = _mm256_set1_epi16(15);
        __m256i coeff16 = _mm256_set1_epi16(16);

        ALIGN32(pel_t first_line[(64 + 16) << 3]);
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
        for (i = 0; i < left_size; i++, src += 8) {//left size`s value is small ,there is no need to use intrinsic assmble
            pfirst[0][i] = (pel_t)((src[6] + (src[7] << 1) + src[8] + 2) >> 2);
            pfirst[1][i] = (pel_t)((src[5] + (src[6] << 1) + src[7] + 2) >> 2);
            pfirst[2][i] = (pel_t)((src[4] + (src[5] << 1) + src[6] + 2) >> 2);
            pfirst[3][i] = (pel_t)((src[3] + (src[4] << 1) + src[5] + 2) >> 2);

            pfirst[4][i] = (pel_t)((src[2] + (src[3] << 1) + src[4] + 2) >> 2);
            pfirst[5][i] = (pel_t)((src[1] + (src[2] << 1) + src[3] + 2) >> 2);
            pfirst[6][i] = (pel_t)((src[0] + (src[1] << 1) + src[2] + 2) >> 2);
            pfirst[7][i] = (pel_t)((src[-1] + (src[0] << 1) + src[1] + 2) >> 2);
        }

        __m256i p00, p10, p20, p30;
        __m256i p01, p11, p21, p31;
        __m256i S0, S1, S2, S3;
        __m256i L0, L1, L2, L3;
        __m256i H0, H1, H2, H3;

        for (; i < line_size - 16; i += 32, src += 32) {

            S0 = _mm256_loadu_si256((__m256i*)(src + 2));
            S1 = _mm256_loadu_si256((__m256i*)(src + 1));
            S2 = _mm256_loadu_si256((__m256i*)(src));
            S3 = _mm256_loadu_si256((__m256i*)(src - 1));

            L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
            L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
            L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));
            L3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 0));

            H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));
            H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));
            H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));
            H3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 1));

            p00 = _mm256_mullo_epi16(L0, coeff7);
            p10 = _mm256_mullo_epi16(L1, coeff15);
            p20 = _mm256_mullo_epi16(L2, coeff9);
            p30 = _mm256_add_epi16(L3, coeff16);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 5);

            p01 = _mm256_mullo_epi16(H0, coeff7);
            p11 = _mm256_mullo_epi16(H1, coeff15);
            p21 = _mm256_mullo_epi16(H2, coeff9);
            p31 = _mm256_add_epi16(H3, coeff16);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_srli_epi16(p01, 5);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);

            _mm256_storeu_si256((__m256i*)&pfirst[0][i], p00);

            p00 = _mm256_mullo_epi16(L0, coeff3);
            p10 = _mm256_mullo_epi16(L1, coeff7);
            p20 = _mm256_mullo_epi16(L2, coeff5);
            p30 = _mm256_add_epi16(L3, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 4);

            p01 = _mm256_mullo_epi16(H0, coeff3);
            p11 = _mm256_mullo_epi16(H1, coeff7);
            p21 = _mm256_mullo_epi16(H2, coeff5);
            p31 = _mm256_add_epi16(H3, coeff8);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_srli_epi16(p01, 4);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[1][i], p00);


            p00 = _mm256_mullo_epi16(L0, coeff5);
            p10 = _mm256_mullo_epi16(L1, coeff13);
            p20 = _mm256_mullo_epi16(L2, coeff11);
            p30 = _mm256_mullo_epi16(L3, coeff3);
            p00 = _mm256_add_epi16(p00, coeff16);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 5);

            p01 = _mm256_mullo_epi16(H0, coeff5);
            p11 = _mm256_mullo_epi16(H1, coeff13);
            p21 = _mm256_mullo_epi16(H2, coeff11);
            p31 = _mm256_mullo_epi16(H3, coeff3);
            p01 = _mm256_add_epi16(p01, coeff16);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_srli_epi16(p01, 5);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[2][i], p00);

            p00 = _mm256_add_epi16(L0, L3);
            p10 = _mm256_add_epi16(L1, L2);
            p10 = _mm256_mullo_epi16(p10, coeff3);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, coeff4);
            p00 = _mm256_srli_epi16(p00, 3);

            p01 = _mm256_add_epi16(H0, H3);
            p11 = _mm256_add_epi16(H1, H2);
            p11 = _mm256_mullo_epi16(p11, coeff3);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, coeff4);
            p01 = _mm256_srli_epi16(p01, 3);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[3][i], p00);

            p00 = _mm256_mullo_epi16(L0, coeff3);
            p10 = _mm256_mullo_epi16(L1, coeff11);
            p20 = _mm256_mullo_epi16(L2, coeff13);
            p30 = _mm256_mullo_epi16(L3, coeff5);
            p00 = _mm256_add_epi16(p00, coeff16);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 5);

            p01 = _mm256_mullo_epi16(H0, coeff3);
            p11 = _mm256_mullo_epi16(H1, coeff11);
            p21 = _mm256_mullo_epi16(H2, coeff13);
            p31 = _mm256_mullo_epi16(H3, coeff5);
            p01 = _mm256_add_epi16(p01, coeff16);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_srli_epi16(p01, 5);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[4][i], p00);

            p10 = _mm256_mullo_epi16(L1, coeff5);
            p20 = _mm256_mullo_epi16(L2, coeff7);
            p30 = _mm256_mullo_epi16(L3, coeff3);
            p00 = _mm256_add_epi16(L0, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 4);

            p11 = _mm256_mullo_epi16(H1, coeff5);
            p21 = _mm256_mullo_epi16(H2, coeff7);
            p31 = _mm256_mullo_epi16(H3, coeff3);
            p01 = _mm256_add_epi16(H0, coeff8);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_srli_epi16(p01, 4);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[5][i], p00);

            p10 = _mm256_mullo_epi16(L1, coeff9);
            p20 = _mm256_mullo_epi16(L2, coeff15);
            p30 = _mm256_mullo_epi16(L3, coeff7);
            p00 = _mm256_add_epi16(L0, coeff16);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 5);

            p11 = _mm256_mullo_epi16(H1, coeff9);
            p21 = _mm256_mullo_epi16(H2, coeff15);
            p31 = _mm256_mullo_epi16(H3, coeff7);
            p01 = _mm256_add_epi16(H0, coeff16);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_srli_epi16(p01, 5);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[6][i], p00);

            p10 = _mm256_mullo_epi16(L2, coeff2);
            p00 = _mm256_add_epi16(L1, L3);
            p00 = _mm256_add_epi16(p00, coeff2);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_srli_epi16(p00, 2);

            p11 = _mm256_mullo_epi16(H2, coeff2);
            p01 = _mm256_add_epi16(H1, H3);
            p01 = _mm256_add_epi16(p01, coeff2);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_srli_epi16(p01, 2);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[7][i], p00);

        }
        __m256i mask = _mm256_load_si256((__m256i*)intrinsic_mask_256_8bit[bsx - 1]);

        if (i < line_size) {

            S0 = _mm256_loadu_si256((__m256i*)(src + 2));
            S1 = _mm256_loadu_si256((__m256i*)(src + 1));
            S2 = _mm256_loadu_si256((__m256i*)(src));
            S3 = _mm256_loadu_si256((__m256i*)(src - 1));

            L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
            L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
            L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));
            L3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 0));

            p00 = _mm256_mullo_epi16(L0, coeff7);
            p10 = _mm256_mullo_epi16(L1, coeff15);
            p20 = _mm256_mullo_epi16(L2, coeff9);
            p30 = _mm256_add_epi16(L3, coeff16);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 5);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_maskstore_epi32((int*)&pfirst[0][i], mask, p00);

            p00 = _mm256_mullo_epi16(L0, coeff3);
            p10 = _mm256_mullo_epi16(L1, coeff7);
            p20 = _mm256_mullo_epi16(L2, coeff5);
            p30 = _mm256_add_epi16(L3, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 4);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_maskstore_epi32((int*)&pfirst[1][i], mask, p00);

            p00 = _mm256_mullo_epi16(L0, coeff5);
            p10 = _mm256_mullo_epi16(L1, coeff13);
            p20 = _mm256_mullo_epi16(L2, coeff11);
            p30 = _mm256_mullo_epi16(L3, coeff3);
            p00 = _mm256_add_epi16(p00, coeff16);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 5);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_maskstore_epi32((int*)&pfirst[2][i], mask, p00);

            p00 = _mm256_add_epi16(L0, L3);
            p10 = _mm256_add_epi16(L1, L2);
            p10 = _mm256_mullo_epi16(p10, coeff3);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, coeff4);
            p00 = _mm256_srli_epi16(p00, 3);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_maskstore_epi32((int*)&pfirst[3][i], mask, p00);

            p00 = _mm256_mullo_epi16(L0, coeff3);
            p10 = _mm256_mullo_epi16(L1, coeff11);
            p20 = _mm256_mullo_epi16(L2, coeff13);
            p30 = _mm256_mullo_epi16(L3, coeff5);
            p00 = _mm256_add_epi16(p00, coeff16);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 5);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_maskstore_epi32((int*)&pfirst[4][i], mask, p00);

            p10 = _mm256_mullo_epi16(L1, coeff5);
            p20 = _mm256_mullo_epi16(L2, coeff7);
            p30 = _mm256_mullo_epi16(L3, coeff3);
            p00 = _mm256_add_epi16(L0, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 4);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_maskstore_epi32((int*)&pfirst[5][i], mask, p00);

            p10 = _mm256_mullo_epi16(L1, coeff9);
            p20 = _mm256_mullo_epi16(L2, coeff15);
            p30 = _mm256_mullo_epi16(L3, coeff7);
            p00 = _mm256_add_epi16(L0, coeff16);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 5);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_maskstore_epi32((int*)&pfirst[6][i], mask, p00);

            p10 = _mm256_mullo_epi16(L2, coeff2);
            p00 = _mm256_add_epi16(L1, L3);
            p00 = _mm256_add_epi16(p00, coeff2);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_srli_epi16(p00, 2);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_maskstore_epi32((int*)&pfirst[7][i], mask, p00);
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
        __m256i M;
        if (bsx == 64) {
            for (i = 0; i < bsy; i++) {
                M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] - i + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] - i + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[4] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[4] - i + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[5] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[5] - i + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[6] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[6] - i + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[7] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[7] - i + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;
            }
        } else if (bsx == 32) {
            for (i = 0; i < bsy; i++) {
                M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[4] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[5] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[6] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[7] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;
            }
        } else if (bsx == 16) {
            mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
            for (i = 0; i < bsy; i++) {
                M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[4] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[5] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[6] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[7] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
            }
        } else if (bsx == 8) {
            mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[7]);
            for (i = 0; i < bsy; i++) {
                M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[4] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[5] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[6] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[7] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
            }
        } else {
            mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[3]);
            for (i = 0; i < bsy; i++) {
                M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i));
                _mm256_maskstore_epi32((int*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i));
                _mm256_maskstore_epi32((int*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] - i));
                _mm256_maskstore_epi32((int*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] - i));
                _mm256_maskstore_epi32((int*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[4] - i));
                _mm256_maskstore_epi32((int*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[5] - i));
                _mm256_maskstore_epi32((int*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[6] - i));
                _mm256_maskstore_epi32((int*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[7] - i));
                _mm256_maskstore_epi32((int*)dst, mask, M);
                dst += i_dst;
            }
        }

        /*for (i = 0; i < bsy; i++) {
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
        }*/
    } else {
        intra_pred_ang_xy_13_sse128(src, dst, i_dst, dir_mode, bsx, bsy);
        return;
    }

}

void intra_pred_ang_xy_14_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;
    UNUSED_PARAMETER(dir_mode);

    __m256i coeff2 = _mm256_set1_epi16(2);
    __m256i coeff3 = _mm256_set1_epi16(3);
    __m256i coeff4 = _mm256_set1_epi16(4);
    __m256i coeff5 = _mm256_set1_epi16(5);
    __m256i coeff7 = _mm256_set1_epi16(7);
    __m256i coeff8 = _mm256_set1_epi16(8);

    if (bsy != 4) {
        ALIGN32(pel_t first_line[4 * (64 + 32)]);
        int line_size = bsx + bsy / 4 - 1;
        int left_size = line_size - bsx;
        int aligned_line_size = ((line_size + 31) >> 4) << 4;
        pel_t *pfirst[4];
        __m256i shuffle = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                           0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

        __m256i index = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        pel_t *pSrc1 = src;

        pfirst[0] = first_line;
        pfirst[1] = first_line + aligned_line_size;
        pfirst[2] = first_line + aligned_line_size * 2;
        pfirst[3] = first_line + aligned_line_size * 3;
        src -= bsy - 4;

        __m256i p00, p01, p10, p11;
        __m256i p20, p30, p21, p31;
        __m256i S0, S1, S2, S3;
        __m256i L0, L1, L2, L3;
        __m256i H0, H1, H2, H3;

        for (i = 0; i < left_size - 1; i += 8, src += 32) {

            S0 = _mm256_loadu_si256((__m256i*)(src - 1));//0 1 2 3 4 5 6 7 8...15
            S1 = _mm256_loadu_si256((__m256i*)(src));
            S2 = _mm256_loadu_si256((__m256i*)(src + 1));

            L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));//0...15
            L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
            L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));

            H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));//16...31
            H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));
            H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));

            p00 = _mm256_add_epi16(L0, L1);
            p01 = _mm256_add_epi16(L1, L2);
            p10 = _mm256_add_epi16(H0, H1);
            p11 = _mm256_add_epi16(H1, H2);

            p00 = _mm256_add_epi16(p00, coeff2);
            p10 = _mm256_add_epi16(p10, coeff2);
            p00 = _mm256_add_epi16(p00, p01);
            p10 = _mm256_add_epi16(p10, p11);

            p00 = _mm256_srli_epi16(p00, 2);//0...7 8...15
            p10 = _mm256_srli_epi16(p10, 2);//16...23 24...31

            p00 = _mm256_packus_epi16(p00, p10);//0...7 16...23 8...15 24...31

            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            //0 4 8 12 1 5 9 13 2 6 10 14 3 7 11 15     16 20 24 28 17 21...
            p10 = _mm256_shuffle_epi8(p00, shuffle);
            //0 4 8 12 16 20 24 28    1 5 9 13 17 21 25 29
            p10 = _mm256_permutevar8x32_epi32(p10, index);

            ((__int64*)&pfirst[0][i])[0] = _mm256_extract_epi64(p10, 3);
            ((__int64*)&pfirst[1][i])[0] = _mm256_extract_epi64(p10, 2);
            ((__int64*)&pfirst[2][i])[0] = _mm256_extract_epi64(p10, 1);
            ((__int64*)&pfirst[3][i])[0] = _mm256_extract_epi64(p10, 0);

        }

        if (i < left_size) { //sse版本比avx快，处理的数据较少
            __m128i shuffle1 = _mm_setr_epi8(0, 4, 1, 5, 2, 6, 3, 7, 0, 4, 1, 5, 2, 6, 3, 7);
            __m128i coeff2_ = _mm_set1_epi16(2);
            __m128i zero = _mm_setzero_si128();

            __m128i S0_ = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S2_ = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i S1_ = _mm_loadu_si128((__m128i*)(src));

            __m128i L0_ = _mm_unpacklo_epi8(S0_, zero);//0 1 2 3 4 5 6 7
            __m128i L1_ = _mm_unpacklo_epi8(S1_, zero);
            __m128i L2_ = _mm_unpacklo_epi8(S2_, zero);

            __m128i p00_ = _mm_add_epi16(L0_, L1_);
            __m128i p01_ = _mm_add_epi16(L1_, L2_);

            p00_ = _mm_add_epi16(p00_, coeff2_);
            p00_ = _mm_add_epi16(p00_, p01_);

            p00_ = _mm_srli_epi16(p00_, 2);

            p00_ = _mm_packus_epi16(p00_, p00_);//0 1 2 3 4 5 6 7

            p00_ = _mm_shuffle_epi8(p00_, shuffle1);//0 4 1 5 2 6 3 7

            ((int*)&pfirst[0][i])[0] = _mm_extract_epi16(p00_, 3);
            ((int*)&pfirst[1][i])[0] = _mm_extract_epi16(p00_, 2);
            ((int*)&pfirst[2][i])[0] = _mm_extract_epi16(p00_, 1);
            ((int*)&pfirst[3][i])[0] = _mm_extract_epi16(p00_, 0);
        }

        src = pSrc1;

        for (i = left_size; i < line_size - 16; i += 32, src += 32) {

            S0 = _mm256_loadu_si256((__m256i*)(src - 1));
            S1 = _mm256_loadu_si256((__m256i*)(src));
            S2 = _mm256_loadu_si256((__m256i*)(src + 1));
            S3 = _mm256_loadu_si256((__m256i*)(src + 2));

            L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
            L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
            L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));
            L3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 0));

            H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));
            H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));
            H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));
            H3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 1));

            p00 = _mm256_mullo_epi16(L0, coeff3);
            p10 = _mm256_mullo_epi16(L1, coeff7);
            p20 = _mm256_mullo_epi16(L2, coeff5);
            p30 = _mm256_add_epi16(L3, coeff8);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_srli_epi16(p00, 4);

            p01 = _mm256_mullo_epi16(H0, coeff3);
            p11 = _mm256_mullo_epi16(H1, coeff7);
            p21 = _mm256_mullo_epi16(H2, coeff5);
            p31 = _mm256_add_epi16(H3, coeff8);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_srli_epi16(p01, 4);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[2][i], p00);

            p00 = _mm256_add_epi16(L1, L2);
            p00 = _mm256_mullo_epi16(p00, coeff3);
            p10 = _mm256_add_epi16(L0, L3);
            p10 = _mm256_add_epi16(p10, coeff4);
            p00 = _mm256_add_epi16(p10, p00);
            p00 = _mm256_srli_epi16(p00, 3);

            p01 = _mm256_add_epi16(H1, H2);
            p01 = _mm256_mullo_epi16(p01, coeff3);
            p11 = _mm256_add_epi16(H0, H3);
            p11 = _mm256_add_epi16(p11, coeff4);
            p01 = _mm256_add_epi16(p11, p01);
            p01 = _mm256_srli_epi16(p01, 3);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[1][i], p00);

            p10 = _mm256_mullo_epi16(L1, coeff5);
            p20 = _mm256_mullo_epi16(L2, coeff7);
            p30 = _mm256_mullo_epi16(L3, coeff3);
            p00 = _mm256_add_epi16(L0, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 4);

            p11 = _mm256_mullo_epi16(H1, coeff5);
            p21 = _mm256_mullo_epi16(H2, coeff7);
            p31 = _mm256_mullo_epi16(H3, coeff3);
            p01 = _mm256_add_epi16(H0, coeff8);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_srli_epi16(p01, 4);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[0][i], p00);

            p00 = _mm256_add_epi16(L0, L1);
            p10 = _mm256_add_epi16(L1, L2);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, coeff2);
            p00 = _mm256_srli_epi16(p00, 2);

            p01 = _mm256_add_epi16(H0, H1);
            p11 = _mm256_add_epi16(H1, H2);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, coeff2);
            p01 = _mm256_srli_epi16(p01, 2);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&pfirst[3][i], p00);
        }

        if (i  < line_size) {
            __m256i mask = _mm256_load_si256((__m256i*)intrinsic_mask_256_8bit[15]);

            S0 = _mm256_loadu_si256((__m256i*)(src - 1));
            S1 = _mm256_loadu_si256((__m256i*)(src));
            S2 = _mm256_loadu_si256((__m256i*)(src + 1));
            S3 = _mm256_loadu_si256((__m256i*)(src + 2));

            L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
            L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
            L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));
            L3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 0));

            p00 = _mm256_mullo_epi16(L0, coeff3);
            p10 = _mm256_mullo_epi16(L1, coeff7);
            p20 = _mm256_mullo_epi16(L2, coeff5);
            p30 = _mm256_add_epi16(L3, coeff8);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_srli_epi16(p00, 4);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            _mm256_maskstore_epi64((__int64*)&pfirst[2][i], mask, p00);

            p00 = _mm256_add_epi16(L1, L2);
            p00 = _mm256_mullo_epi16(p00, coeff3);
            p10 = _mm256_add_epi16(L0, L3);
            p10 = _mm256_add_epi16(p10, coeff4);
            p00 = _mm256_add_epi16(p10, p00);
            p00 = _mm256_srli_epi16(p00, 3);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            _mm256_maskstore_epi64((__int64*)&pfirst[1][i], mask, p00);

            p10 = _mm256_mullo_epi16(L1, coeff5);
            p20 = _mm256_mullo_epi16(L2, coeff7);
            p30 = _mm256_mullo_epi16(L3, coeff3);
            p00 = _mm256_add_epi16(L0, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 4);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            _mm256_maskstore_epi64((__int64*)&pfirst[0][i], mask, p00);

            p00 = _mm256_add_epi16(L0, L1);
            p10 = _mm256_add_epi16(L1, L2);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, coeff2);
            p00 = _mm256_srli_epi16(p00, 2);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            _mm256_maskstore_epi64((__int64*)&pfirst[3][i], mask, p00);
        }

        pfirst[0] += left_size;
        pfirst[1] += left_size;
        pfirst[2] += left_size;
        pfirst[3] += left_size;

        bsy >>= 2;


        if (bsx == 64) {
            for (i = 0; i < bsy; i++) {
                __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] - i + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] - i + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;
            }
        } else if (bsx == 32) {
            for (i = 0; i < bsy; i++) {
                __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] - i));
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;
            }

        } else if (bsx == 16) {
            __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
            for (i = 0; i < bsy; i++) {
                __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
            }
        } else if (bsx == 8) {
            __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[7]);
            for (i = 0; i < bsy; i++) {
                __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] - i));
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
            }
        } else {
            __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[3]);
            for (i = 0; i < bsy; i++) {
                __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i));
                _mm256_maskstore_epi32((int*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i));
                _mm256_maskstore_epi32((int*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[2] - i));
                _mm256_maskstore_epi32((int*)dst, mask, M);
                dst += i_dst;

                M = _mm256_lddqu_si256((__m256i*)(pfirst[3] - i));
                _mm256_maskstore_epi32((int*)dst, mask, M);
                dst += i_dst;
            }
        }


        /*for (i = 0; i < bsy; i++) {
            memcpy(dst, pfirst[0] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[1] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[2] - i, bsx * sizeof(pel_t));
            dst += i_dst;
            memcpy(dst, pfirst[3] - i, bsx * sizeof(pel_t));
            dst += i_dst;
        }*/
    } else {
        if (bsx == 16) {
            __m256i mask = _mm256_load_si256((__m256i*)intrinsic_mask_256_8bit[15]);
            pel_t *dst2 = dst + i_dst;
            pel_t *dst3 = dst2 + i_dst;
            pel_t *dst4 = dst3 + i_dst;

            __m256i p00, p10, p20, p30;
            __m256i L0, L1, L2, L3;
            __m256i S0 = _mm256_loadu_si256((__m256i*)(src - 1));
            __m256i S3 = _mm256_loadu_si256((__m256i*)(src + 2));
            __m256i S1 = _mm256_loadu_si256((__m256i*)(src));
            __m256i S2 = _mm256_loadu_si256((__m256i*)(src + 1));

            L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
            L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
            L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));
            L3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 0));

            p00 = _mm256_mullo_epi16(L0, coeff3);
            p10 = _mm256_mullo_epi16(L1, coeff7);
            p20 = _mm256_mullo_epi16(L2, coeff5);
            p30 = _mm256_add_epi16(L3, coeff8);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_srli_epi16(p00, 4);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            _mm256_maskstore_epi64((__int64*)dst3, mask, p00);

            p00 = _mm256_add_epi16(L1, L2);
            p00 = _mm256_mullo_epi16(p00, coeff3);
            p10 = _mm256_add_epi16(L0, L3);
            p10 = _mm256_add_epi16(p10, coeff4);
            p00 = _mm256_add_epi16(p10, p00);
            p00 = _mm256_srli_epi16(p00, 3);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            _mm256_maskstore_epi64((__int64*)dst2, mask, p00);


            p10 = _mm256_mullo_epi16(L1, coeff5);
            p20 = _mm256_mullo_epi16(L2, coeff7);
            p30 = _mm256_mullo_epi16(L3, coeff3);
            p00 = _mm256_add_epi16(L0, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_srli_epi16(p00, 4);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            _mm256_maskstore_epi64((__int64*)dst, mask, p00);

            p00 = _mm256_add_epi16(L0, L1);
            p10 = _mm256_add_epi16(L1, L2);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, coeff2);
            p00 = _mm256_srli_epi16(p00, 2);

            p00 = _mm256_packus_epi16(p00, p00);
            p00 = _mm256_permute4x64_epi64(p00, 0x0008);
            _mm256_maskstore_epi64((__int64*)dst4, mask, p00);
        } else {//4x4
            pel_t *dst2 = dst + i_dst;
            pel_t *dst3 = dst2 + i_dst;
            pel_t *dst4 = dst3 + i_dst;
            __m128i p00, p10, p20, p30;
            __m128i coeff2_ = _mm_set1_epi16(2);
            __m128i coeff3_ = _mm_set1_epi16(3);
            __m128i coeff4_ = _mm_set1_epi16(4);
            __m128i coeff5_ = _mm_set1_epi16(5);
            __m128i coeff7_ = _mm_set1_epi16(7);
            __m128i coeff8_ = _mm_set1_epi16(8);
            __m128i zero = _mm_setzero_si128();

            __m128i S0 = _mm_loadu_si128((__m128i*)(src - 1));
            __m128i S3 = _mm_loadu_si128((__m128i*)(src + 2));
            __m128i S1 = _mm_loadu_si128((__m128i*)(src));
            __m128i S2 = _mm_loadu_si128((__m128i*)(src + 1));

            __m128i L0 = _mm_unpacklo_epi8(S0, zero);
            __m128i L1 = _mm_unpacklo_epi8(S1, zero);
            __m128i L2 = _mm_unpacklo_epi8(S2, zero);
            __m128i L3 = _mm_unpacklo_epi8(S3, zero);

            p00 = _mm_mullo_epi16(L0, coeff3_);
            p10 = _mm_mullo_epi16(L1, coeff7_);
            p20 = _mm_mullo_epi16(L2, coeff5_);
            p30 = _mm_add_epi16(L3, coeff8_);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            ((int*)dst3)[0] = _mm_cvtsi128_si32(p00);

            p00 = _mm_add_epi16(L1, L2);
            p00 = _mm_mullo_epi16(p00, coeff3_);
            p10 = _mm_add_epi16(L0, L3);
            p10 = _mm_add_epi16(p10, coeff4_);
            p00 = _mm_add_epi16(p10, p00);
            p00 = _mm_srli_epi16(p00, 3);

            p00 = _mm_packus_epi16(p00, p00);
            ((int*)dst2)[0] = _mm_cvtsi128_si32(p00);

            p10 = _mm_mullo_epi16(L1, coeff5_);
            p20 = _mm_mullo_epi16(L2, coeff7_);
            p30 = _mm_mullo_epi16(L3, coeff3_);
            p00 = _mm_add_epi16(L0, coeff8_);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, p20);
            p00 = _mm_add_epi16(p00, p30);
            p00 = _mm_srli_epi16(p00, 4);

            p00 = _mm_packus_epi16(p00, p00);
            ((int*)dst)[0] = _mm_cvtsi128_si32(p00);

            p00 = _mm_add_epi16(L0, L1);
            p10 = _mm_add_epi16(L1, L2);
            p00 = _mm_add_epi16(p00, p10);
            p00 = _mm_add_epi16(p00, coeff2_);
            p00 = _mm_srli_epi16(p00, 2);

            p00 = _mm_packus_epi16(p00, p00);
            ((int*)dst4)[0] = _mm_cvtsi128_si32(p00);
        }
    }

}

void intra_pred_ang_xy_16_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN32(pel_t first_line[2 * (64 + 48)]);
    int line_size = bsx + bsy / 2 - 1;
    int left_size = line_size - bsx;
    int aligned_line_size = ((line_size + 31) >> 4) << 4;
    pel_t *pfirst[2];
    UNUSED_PARAMETER(dir_mode);
    __m256i coeff2   = _mm256_set1_epi16(2);
    __m256i coeff3   = _mm256_set1_epi16(3);
    __m256i coeff4   = _mm256_set1_epi16(4);
    __m256i shuffle = _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
                                       0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);

    int i;
    pel_t *pSrc1;

    pfirst[0] = first_line;
    pfirst[1] = first_line + aligned_line_size;

    src -= bsy - 2;
    pSrc1 = src;

    __m256i p00, p01, p10, p11;
    __m256i S0, S1, S2, S3;
    __m256i L0, L1, L2, L3;
    __m256i H0, H1, H2, H3;

    __m256i mask1 = _mm256_load_si256((__m256i*)intrinsic_mask_256_8bit[15]);
    for (i = 0; i < left_size - 8; i += 16, src += 32) {

        S0 = _mm256_loadu_si256((__m256i*)(src - 1));//
        S2 = _mm256_loadu_si256((__m256i*)(src + 1));//
        S1 = _mm256_loadu_si256((__m256i*)(src));//

        L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
        L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
        L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));

        H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));
        H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));
        H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));

        p00 = _mm256_add_epi16(L0, L1);
        p01 = _mm256_add_epi16(L1, L2);
        p10 = _mm256_add_epi16(H0, H1);
        p11 = _mm256_add_epi16(H1, H2);

        p00 = _mm256_add_epi16(p00, coeff2);
        p10 = _mm256_add_epi16(p10, coeff2);

        p00 = _mm256_add_epi16(p00, p01);
        p10 = _mm256_add_epi16(p10, p11);

        p00 = _mm256_srli_epi16(p00, 2);//0 1 2 3 4 5 6 7....15
        p10 = _mm256_srli_epi16(p10, 2);//16 17 18....31

        //0...7 16...23 8...15 24...31
        p00 = _mm256_packus_epi16(p00, p10);
        p00 = _mm256_permute4x64_epi64(p00, 0x00D8);//31...16 15..0

        //0 1 2 3
        p00 = _mm256_shuffle_epi8(p00, shuffle);

        p10 = _mm256_permute4x64_epi64(p00, 0x08);//0 2
        p00 = _mm256_permute4x64_epi64(p00, 0x0D);//1 3

        _mm256_maskstore_epi64((__int64*)&pfirst[0][i], mask1, p00);
        _mm256_maskstore_epi64((__int64*)&pfirst[1][i], mask1, p10);

    }

    __m256i mask2 = _mm256_load_si256((__m256i*)intrinsic_mask_256_8bit[7]);
    if (i < left_size) {
        S0 = _mm256_loadu_si256((__m256i*)(src - 1));
        S2 = _mm256_loadu_si256((__m256i*)(src + 1));
        S1 = _mm256_loadu_si256((__m256i*)(src));

        L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
        L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
        L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));

        p00 = _mm256_add_epi16(L0, L1);
        p01 = _mm256_add_epi16(L1, L2);
        p00 = _mm256_add_epi16(p00, coeff2);
        p00 = _mm256_add_epi16(p00, p01);
        p00 = _mm256_srli_epi16(p00, 2);

        //0...7 0...7 8...15 8...15
        p00 = _mm256_packus_epi16(p00, p00);
        p00 = _mm256_permute4x64_epi64(p00, 0x0008);//0...15 0...15

        p01 = _mm256_shuffle_epi8(p00, shuffle);//0 2 4 6 7 8 10 12 14    1 3 5 7 9 11 13 15

        p10 = _mm256_permute4x64_epi64(p01, 0x01);

        _mm256_maskstore_epi64((__int64*)&pfirst[0][i], mask2, p10);
        _mm256_maskstore_epi64((__int64*)&pfirst[1][i], mask2, p01);
    }

    src = pSrc1 + left_size + left_size;

    for (i = left_size; i < line_size - 16; i += 32, src += 32) {

        S0 = _mm256_loadu_si256((__m256i*)(src - 1));
        S3 = _mm256_loadu_si256((__m256i*)(src + 2));
        S1 = _mm256_loadu_si256((__m256i*)(src));
        S2 = _mm256_loadu_si256((__m256i*)(src + 1));

        L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
        L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
        L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));
        L3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 0));

        H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));
        H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));
        H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));
        H3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 1));

        p00 = _mm256_add_epi16(L1, L2);
        p01 = _mm256_add_epi16(L0, L3);
        p00 = _mm256_mullo_epi16(p00, coeff3);
        p00 = _mm256_add_epi16(p00, coeff4);
        p00 = _mm256_add_epi16(p00, p01);
        p00 = _mm256_srli_epi16(p00, 3);

        p10 = _mm256_add_epi16(H1, H2);
        p11 = _mm256_add_epi16(H0, H3);
        p10 = _mm256_mullo_epi16(p10, coeff3);
        p10 = _mm256_add_epi16(p10, coeff4);
        p10 = _mm256_add_epi16(p10, p11);
        p10 = _mm256_srli_epi16(p10, 3);

        p00 = _mm256_packus_epi16(p00, p10);
        p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
        _mm256_storeu_si256((__m256i*)&pfirst[0][i], p00);

        p00 = _mm256_add_epi16(L0, L1);
        p01 = _mm256_add_epi16(L1, L2);
        p10 = _mm256_add_epi16(H0, H1);
        p11 = _mm256_add_epi16(H1, H2);

        p00 = _mm256_add_epi16(p00, coeff2);
        p10 = _mm256_add_epi16(p10, coeff2);

        p00 = _mm256_add_epi16(p00, p01);
        p10 = _mm256_add_epi16(p10, p11);

        p00 = _mm256_srli_epi16(p00, 2);
        p10 = _mm256_srli_epi16(p10, 2);

        p00 = _mm256_packus_epi16(p00, p10);
        p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
        _mm256_storeu_si256((__m256i*)&pfirst[1][i], p00);

    }

    if (i < line_size) {

        S0 = _mm256_loadu_si256((__m256i*)(src - 1));
        S3 = _mm256_loadu_si256((__m256i*)(src + 2));
        S1 = _mm256_loadu_si256((__m256i*)(src));
        S2 = _mm256_loadu_si256((__m256i*)(src + 1));

        L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
        L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
        L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));
        L3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 0));

        p00 = _mm256_add_epi16(L1, L2);
        p01 = _mm256_add_epi16(L0, L3);
        p00 = _mm256_mullo_epi16(p00, coeff3);
        p00 = _mm256_add_epi16(p00, coeff4);
        p00 = _mm256_add_epi16(p00, p01);
        p00 = _mm256_srli_epi16(p00, 3);

        p00 = _mm256_packus_epi16(p00, p00);
        p00 = _mm256_permute4x64_epi64(p00, 0x0008);
        _mm256_maskstore_epi64((__int64*)&pfirst[0][i], mask1, p00);

        p00 = _mm256_add_epi16(L0, L1);
        p01 = _mm256_add_epi16(L1, L2);
        p00 = _mm256_add_epi16(p00, coeff2);
        p00 = _mm256_add_epi16(p00, p01);
        p00 = _mm256_srli_epi16(p00, 2);

        p00 = _mm256_packus_epi16(p00, p00);
        p00 = _mm256_permute4x64_epi64(p00, 0x0008);
        _mm256_maskstore_epi64((__int64*)&pfirst[1][i], mask1, p00);

    }

    pfirst[0] += left_size;
    pfirst[1] += left_size;

    bsy >>= 1;

    if (bsx == 64) {

        for (i = 0; i < bsy; i += 4) {

            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 1));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 1 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 1));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 1 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 2));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 2 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 2));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 2 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 3));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 3 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 3));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 3 + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;

        }
    } else if (bsx == 32) {
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 1));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 1));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 2));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 2));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 3));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 3));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
        }
    } else if (bsx == 16) {
        __m256i mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 1));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 1));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 2));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 2));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 3));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 3));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
        }
    } else if (bsx == 8) {
        __m256i mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 1));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 1));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 2));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 2));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 3));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 3));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
        }
    } else {
        __m256i mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 1));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 1));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 2));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 2));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;

            M = _mm256_lddqu_si256((__m256i*)(pfirst[0] - i - 3));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            M = _mm256_lddqu_si256((__m256i*)(pfirst[1] - i - 3));
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
        }
    }

    /*switch (bsx) {
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
    }*/

}

void intra_pred_ang_xy_18_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN32(pel_t first_line[64 + 64]);
    int line_size = bsx + bsy - 1;
    int i;
    pel_t *pfirst = first_line + bsy - 1;
    UNUSED_PARAMETER(dir_mode);
    __m256i coeff2 = _mm256_set1_epi16(2);

    src -= bsy - 1;

    __m256i S0, S1, S2;
    __m256i L0, L1, L2;
    __m256i H0, H1, H2;
    __m256i sum1, sum2, sum3, sum4;

    for (i = 0; i < line_size - 16; i += 32, src += 32) {
        S0 = _mm256_loadu_si256((__m256i*)(src - 1));
        S2 = _mm256_loadu_si256((__m256i*)(src + 1));
        S1 = _mm256_loadu_si256((__m256i*)(src));

        L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
        L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
        L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));

        H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));
        H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));
        H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));

        sum1 = _mm256_add_epi16(L0, L1);
        sum2 = _mm256_add_epi16(L1, L2);
        sum3 = _mm256_add_epi16(H0, H1);
        sum4 = _mm256_add_epi16(H1, H2);

        sum1 = _mm256_add_epi16(sum1, sum2);
        sum3 = _mm256_add_epi16(sum3, sum4);

        sum1 = _mm256_add_epi16(sum1, coeff2);
        sum3 = _mm256_add_epi16(sum3, coeff2);

        sum1 = _mm256_srli_epi16(sum1, 2);
        sum3 = _mm256_srli_epi16(sum3, 2);

        sum1 = _mm256_packus_epi16(sum1, sum3);
        sum1 = _mm256_permute4x64_epi64(sum1, 0x00D8);
        _mm256_storeu_si256((__m256i*)&first_line[i], sum1);
    }

    if (i < line_size) {
        __m256i mask = _mm256_load_si256((__m256i*)intrinsic_mask_256_8bit[15]);
        S0 = _mm256_loadu_si256((__m256i*)(src - 1));
        S2 = _mm256_loadu_si256((__m256i*)(src + 1));
        S1 = _mm256_loadu_si256((__m256i*)(src));

        L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
        L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
        L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));

        sum1 = _mm256_add_epi16(L0, L1);
        sum2 = _mm256_add_epi16(L1, L2);

        sum1 = _mm256_add_epi16(sum1, sum2);
        sum1 = _mm256_add_epi16(sum1, coeff2);
        sum1 = _mm256_srli_epi16(sum1, 2);

        sum1 = _mm256_packus_epi16(sum1, sum1);
        sum1 = _mm256_permute4x64_epi64(sum1, 0x00D8);

        _mm256_maskstore_epi64((__int64*)&first_line[i], mask, sum1);
    }

    __m256i M;
    if (bsx == 64) {
        for (i = 0; i < bsy; i += 4) {
            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            pfirst--;

            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            pfirst--;

            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            pfirst--;

            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            pfirst--;
        }
    } else if (bsx == 32) {
        for (i = 0; i < bsy; i += 4) {
            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            pfirst--;

            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            pfirst--;

            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            pfirst--;

            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            pfirst--;
        }
    } else if (bsx == 16) {
        __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
        for (i = 0; i < bsy; i += 4) {
            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst--;

            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst--;

            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst--;

            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst--;
        }
    } else if (bsx == 8) {
        __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[7]);
        for (i = 0; i < bsy; i += 4) {
            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst--;

            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst--;

            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst--;

            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst--;
        }
    } else {
        __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[3]);
        for (i = 0; i < bsy; i += 4) {
            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            pfirst--;

            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            pfirst--;

            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            pfirst--;

            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            pfirst--;
        }
    }


    /*switch (bsx) {
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
    }*/

}

void intra_pred_ang_xy_20_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    ALIGN32(pel_t first_line[64 + 128]);
    int left_size = (bsy - 1) * 2 + 1;
    int top_size = bsx - 1;
    int line_size = left_size + top_size;
    int i;
    pel_t *pfirst = first_line + left_size - 1;

    __m256i coeff2 = _mm256_set1_epi16(2);
    __m256i coeff3 = _mm256_set1_epi16(3);
    __m256i coeff4 = _mm256_set1_epi16(4);
    __m256i shuffle = _mm256_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15,
                                       0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
    pel_t *pSrc1 = src;

    UNUSED_PARAMETER(dir_mode);

    src -= bsy;

    __m256i p00, p01, p10, p11;
    __m256i p20, p21, p30, p31;

    __m256i S0, S1, S2, S3;
    __m256i L0, L1, L2, L3;
    __m256i H0, H1, H2, H3;

    for (i = 0; i < left_size - 32; i += 64, src += 32) {

        S0 = _mm256_loadu_si256((__m256i*)(src - 1));//0...7 8...15
        S1 = _mm256_loadu_si256((__m256i*)(src));
        S2 = _mm256_loadu_si256((__m256i*)(src + 1));
        S3 = _mm256_loadu_si256((__m256i*)(src + 2));

        L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));//0...7
        L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
        L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));
        L3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 0));

        H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));
        H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));
        H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));
        H3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 1));

        p00 = _mm256_add_epi16(L1, L2);
        p01 = _mm256_add_epi16(L0, L3);
        p00 = _mm256_mullo_epi16(p00, coeff3);
        p00 = _mm256_add_epi16(p00, coeff4);
        p00 = _mm256_add_epi16(p00, p01);
        p00 = _mm256_srli_epi16(p00, 3);//0...15

        p10 = _mm256_add_epi16(H1, H2);
        p11 = _mm256_add_epi16(H0, H3);
        p10 = _mm256_mullo_epi16(p10, coeff3);
        p10 = _mm256_add_epi16(p10, coeff4);
        p10 = _mm256_add_epi16(p10, p11);
        p10 = _mm256_srli_epi16(p10, 3);//16..31

        p20 = _mm256_add_epi16(L1, L2);
        p21 = _mm256_add_epi16(L2, L3);
        p20 = _mm256_add_epi16(p20, coeff2);
        p20 = _mm256_add_epi16(p20, p21);
        p20 = _mm256_srli_epi16(p20, 2);//0...15

        p30 = _mm256_add_epi16(H1, H2);
        p31 = _mm256_add_epi16(H2, H3);
        p30 = _mm256_add_epi16(p30, coeff2);
        p30 = _mm256_add_epi16(p30, p31);
        p30 = _mm256_srli_epi16(p30, 2);//16...31

        //00...07 10...17 08...015 18...115
        p00 = _mm256_packus_epi16(p00, p20);
        p10 = _mm256_packus_epi16(p10, p30);

        p00 = _mm256_shuffle_epi8(p00, shuffle);
        p10 = _mm256_shuffle_epi8(p10, shuffle);

        _mm256_storeu_si256((__m256i*)&first_line[i], p00);
        _mm256_storeu_si256((__m256i*)&first_line[i + 32], p10);
    }

    if (i < left_size) {

        S0 = _mm256_loadu_si256((__m256i*)(src - 1));//0...7 8...15
        S1 = _mm256_loadu_si256((__m256i*)(src));
        S2 = _mm256_loadu_si256((__m256i*)(src + 1));
        S3 = _mm256_loadu_si256((__m256i*)(src + 2));

        L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));//0...7
        L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
        L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));
        L3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 0));

        p00 = _mm256_add_epi16(L1, L2);
        p00 = _mm256_mullo_epi16(p00, coeff3);
        p01 = _mm256_add_epi16(L0, L3);
        p00 = _mm256_add_epi16(p00, coeff4);
        p00 = _mm256_add_epi16(p00, p01);
        p00 = _mm256_srli_epi16(p00, 3);//0...15

        p20 = _mm256_add_epi16(L1, L2);
        p21 = _mm256_add_epi16(L2, L3);
        p20 = _mm256_add_epi16(p20, coeff2);
        p20 = _mm256_add_epi16(p20, p21);
        p20 = _mm256_srli_epi16(p20, 2);//0...15

        p00 = _mm256_packus_epi16(p00, p20);
        p00 = _mm256_shuffle_epi8(p00, shuffle);
        _mm256_storeu_si256((__m256i*)&first_line[i], p00);
    }

    src = pSrc1;

    __m256i sum1, sum2, sum3, sum4;
    for (i = left_size; i < line_size - 16; i += 32, src += 32) {
        S2 = _mm256_loadu_si256((__m256i*)(src + 1));
        S0 = _mm256_loadu_si256((__m256i*)(src - 1));
        S1 = _mm256_loadu_si256((__m256i*)(src));

        L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
        L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
        L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));

        H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));
        H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));
        H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));

        sum1 = _mm256_add_epi16(L0, L1);
        sum2 = _mm256_add_epi16(L1, L2);
        sum3 = _mm256_add_epi16(H0, H1);
        sum4 = _mm256_add_epi16(H1, H2);

        sum1 = _mm256_add_epi16(sum1, sum2);
        sum3 = _mm256_add_epi16(sum3, sum4);

        sum1 = _mm256_add_epi16(sum1, coeff2);
        sum3 = _mm256_add_epi16(sum3, coeff2);

        sum1 = _mm256_srli_epi16(sum1, 2);
        sum3 = _mm256_srli_epi16(sum3, 2);

        sum1 = _mm256_packus_epi16(sum1, sum3);
        sum1 = _mm256_permute4x64_epi64(sum1, 0x00D8);
        _mm256_storeu_si256((__m256i*)&first_line[i], sum1);
    }
    __m256i mask = _mm256_load_si256((__m256i*)intrinsic_mask_256_8bit[15]);
    if (i < line_size) {
        S0 = _mm256_loadu_si256((__m256i*)(src - 1));
        S2 = _mm256_loadu_si256((__m256i*)(src + 1));
        S1 = _mm256_loadu_si256((__m256i*)(src));

        L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
        L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
        L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));

        sum1 = _mm256_add_epi16(L0, L1);
        sum2 = _mm256_add_epi16(L1, L2);

        sum1 = _mm256_add_epi16(sum1, sum2);
        sum1 = _mm256_add_epi16(sum1, coeff2);
        sum1 = _mm256_srli_epi16(sum1, 2);

        sum1 = _mm256_packus_epi16(sum1, sum1);
        sum1 = _mm256_permute4x64_epi64(sum1, 0x00D8);
        _mm256_maskstore_epi64((__int64*)&first_line[i], mask, sum1);
    }

    if (bsx == 64) {

        for (i = 0; i < bsy; i += 8) {
            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_storeu_si256((__m256i*)dst, M);
            M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
            _mm256_storeu_si256((__m256i*)(dst + 32), M);
            dst += i_dst;
            pfirst -= 2;
        }
    } else if (bsx == 32) {
        for (i = 0; i < bsy; i += 8) {
            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_storeu_si256((__m256i*)dst, M);
            dst += i_dst;
            pfirst -= 2;
        }
    } else if (bsx == 16) {

        mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)(pfirst));
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst -= 2;

        }
    } else if (bsx == 8) {

        mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < bsy; i += 8) {
            __m256i M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi64((__int64*)dst, mask, M);
            dst += i_dst;
            pfirst -= 2;
        }
    } else {

        mask = _mm256_loadu_si256((const __m256i*)intrinsic_mask_256_8bit[bsx - 1]);
        for (i = 0; i < bsy; i += 4) {
            __m256i M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            pfirst -= 2;
            M = _mm256_lddqu_si256((__m256i*)pfirst);
            _mm256_maskstore_epi32((int*)dst, mask, M);
            dst += i_dst;
            pfirst -= 2;
        }
    }



    /*for (i = 0; i < bsy; i++) {
        memcpy(dst, pfirst, bsx * sizeof(pel_t));
        pfirst -= 2;
        dst += i_dst;
    }*/

}

void intra_pred_ang_xy_22_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;
    UNUSED_PARAMETER(dir_mode);

    if (bsx != 4) {
        src -= bsy;
        ALIGN32(pel_t first_line[64 + 256]);
        int left_size = (bsy - 1) * 4 + 3;
        int top_size = bsx - 3;
        int line_size = left_size + top_size;
        pel_t *pfirst = first_line + left_size - 3;
        pel_t *pSrc1 = src;

        __m256i coeff2 = _mm256_set1_epi16(2);
        __m256i coeff3 = _mm256_set1_epi16(3);
        __m256i coeff4 = _mm256_set1_epi16(4);
        __m256i coeff5 = _mm256_set1_epi16(5);
        __m256i coeff7 = _mm256_set1_epi16(7);
        __m256i coeff8 = _mm256_set1_epi16(8);
        __m256i shuffle = _mm256_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15,
                                           0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);

        __m256i p00, p10, p20, p30;
        __m256i p01, p11, p21, p31;
        __m256i M1, M2, M3, M4, M5, M6, M7, M8;
        __m256i S0, S1, S2, S3;
        __m256i L0, L1, L2, L3;
        __m256i H0, H1, H2, H3;

        for (i = 0; i < line_size - 64; i += 128, src += 32) {

            S0 = _mm256_loadu_si256((__m256i*)(src - 1));
            S3 = _mm256_loadu_si256((__m256i*)(src + 2));
            S1 = _mm256_loadu_si256((__m256i*)(src));
            S2 = _mm256_loadu_si256((__m256i*)(src + 1));

            L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
            L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
            L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));
            L3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 0));

            H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));
            H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));
            H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));
            H3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 1));

            p00 = _mm256_mullo_epi16(L0, coeff3);
            p10 = _mm256_mullo_epi16(L1, coeff7);
            p20 = _mm256_mullo_epi16(L2, coeff5);
            p30 = _mm256_add_epi16(L3, coeff8);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            M1  = _mm256_srli_epi16(p00, 4);//0...15

            p01 = _mm256_mullo_epi16(H0, coeff3);
            p11 = _mm256_mullo_epi16(H1, coeff7);
            p21 = _mm256_mullo_epi16(H2, coeff5);
            p31 = _mm256_add_epi16(H3, coeff8);
            p01 = _mm256_add_epi16(p01, p31);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            M2  = _mm256_srli_epi16(p01, 4);//16...31

            p00 = _mm256_add_epi16(L1, L2);
            p00 = _mm256_mullo_epi16(p00, coeff3);
            p10 = _mm256_add_epi16(L0, L3);
            p10 = _mm256_add_epi16(p10, coeff4);
            p00 = _mm256_add_epi16(p10, p00);
            M3  = _mm256_srli_epi16(p00, 3);

            p01 = _mm256_add_epi16(H1, H2);
            p01 = _mm256_mullo_epi16(p01, coeff3);
            p11 = _mm256_add_epi16(H0, H3);
            p11 = _mm256_add_epi16(p11, coeff4);
            p01 = _mm256_add_epi16(p11, p01);
            M4  = _mm256_srli_epi16(p01, 3);


            p10 = _mm256_mullo_epi16(L1, coeff5);
            p20 = _mm256_mullo_epi16(L2, coeff7);
            p30 = _mm256_mullo_epi16(L3, coeff3);
            p00 = _mm256_add_epi16(L0, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            M5  = _mm256_srli_epi16(p00, 4);

            p11 = _mm256_mullo_epi16(H1, coeff5);
            p21 = _mm256_mullo_epi16(H2, coeff7);
            p31 = _mm256_mullo_epi16(H3, coeff3);
            p01 = _mm256_add_epi16(H0, coeff8);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, p21);
            p01 = _mm256_add_epi16(p01, p31);
            M6  = _mm256_srli_epi16(p01, 4);


            p00 = _mm256_add_epi16(L1, L2);
            p10 = _mm256_add_epi16(L2, L3);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, coeff2);
            M7  = _mm256_srli_epi16(p00, 2);

            p01 = _mm256_add_epi16(H1, H2);
            p11 = _mm256_add_epi16(H2, H3);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_add_epi16(p01, coeff2);
            M8  = _mm256_srli_epi16(p01, 2);

            M1 = _mm256_packus_epi16(M1, M3);//00...08 10...18
            M5 = _mm256_packus_epi16(M5, M7);
            M1 = _mm256_shuffle_epi8(M1, shuffle);//00 10 01 11 02 12...
            M5 = _mm256_shuffle_epi8(M5, shuffle);

            M2 = _mm256_packus_epi16(M2, M4);
            M6 = _mm256_packus_epi16(M6, M8);
            M2 = _mm256_shuffle_epi8(M2, shuffle);
            M6 = _mm256_shuffle_epi8(M6, shuffle);

            M1 = _mm256_permute4x64_epi64(M1, 0x00D8);
            M5 = _mm256_permute4x64_epi64(M5, 0x00D8);
            M2 = _mm256_permute4x64_epi64(M2, 0x00D8);
            M6 = _mm256_permute4x64_epi64(M6, 0x00D8);

            M3 = _mm256_unpacklo_epi16(M1, M5);
            M7 = _mm256_unpackhi_epi16(M1, M5);
            M4 = _mm256_unpacklo_epi16(M2, M6);
            M8 = _mm256_unpackhi_epi16(M2, M6);

            _mm256_storeu_si256((__m256i*)&first_line[i], M3);
            _mm256_storeu_si256((__m256i*)&first_line[32 + i], M7);
            _mm256_storeu_si256((__m256i*)&first_line[64 + i], M4);
            _mm256_storeu_si256((__m256i*)&first_line[96 + i], M8);
        }

        if (i < left_size) {

            S0 = _mm256_loadu_si256((__m256i*)(src - 1));
            S3 = _mm256_loadu_si256((__m256i*)(src + 2));
            S1 = _mm256_loadu_si256((__m256i*)(src));
            S2 = _mm256_loadu_si256((__m256i*)(src + 1));

            L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
            L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
            L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));
            L3 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S3, 0));

            p00 = _mm256_mullo_epi16(L0, coeff3);
            p10 = _mm256_mullo_epi16(L1, coeff7);
            p20 = _mm256_mullo_epi16(L2, coeff5);
            p30 = _mm256_add_epi16(L3, coeff8);
            p00 = _mm256_add_epi16(p00, p30);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            M1  = _mm256_srli_epi16(p00, 4);

            p00 = _mm256_add_epi16(L1, L2);
            p00 = _mm256_mullo_epi16(p00, coeff3);
            p10 = _mm256_add_epi16(L0, L3);
            p10 = _mm256_add_epi16(p10, coeff4);
            p00 = _mm256_add_epi16(p10, p00);
            M3  = _mm256_srli_epi16(p00, 3);

            p10 = _mm256_mullo_epi16(L1, coeff5);
            p20 = _mm256_mullo_epi16(L2, coeff7);
            p30 = _mm256_mullo_epi16(L3, coeff3);
            p00 = _mm256_add_epi16(L0, coeff8);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, p20);
            p00 = _mm256_add_epi16(p00, p30);
            M5  = _mm256_srli_epi16(p00, 4);

            p10 = _mm256_add_epi16(L2, L3);
            p00 = _mm256_add_epi16(L1, L2);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_add_epi16(p00, coeff2);
            M7  = _mm256_srli_epi16(p00, 2);

            M1 = _mm256_packus_epi16(M1, M3);
            M5 = _mm256_packus_epi16(M5, M7);
            M1 = _mm256_shuffle_epi8(M1, shuffle);
            M5 = _mm256_shuffle_epi8(M5, shuffle);

            M1 = _mm256_permute4x64_epi64(M1, 0x00D8);
            M5 = _mm256_permute4x64_epi64(M5, 0x00D8);

            M3 = _mm256_unpacklo_epi16(M1, M5);
            M7 = _mm256_unpackhi_epi16(M1, M5);

            _mm256_store_si256((__m256i*)&first_line[i], M3);
            _mm256_store_si256((__m256i*)&first_line[32 + i], M7);
        }

        src = pSrc1 + bsy;

        __m256i sum1, sum2, sum3, sum4;
        for (i = left_size; i < line_size - 16; i += 32, src += 32) {

            S0 = _mm256_loadu_si256((__m256i*)(src - 1));
            S2 = _mm256_loadu_si256((__m256i*)(src + 1));
            S1 = _mm256_loadu_si256((__m256i*)(src));

            L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
            L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
            L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));

            H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));
            H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));
            H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));

            sum1 = _mm256_add_epi16(L0, L1);
            sum2 = _mm256_add_epi16(L1, L2);
            sum3 = _mm256_add_epi16(H0, H1);
            sum4 = _mm256_add_epi16(H1, H2);

            sum1 = _mm256_add_epi16(sum1, sum2);
            sum3 = _mm256_add_epi16(sum3, sum4);

            sum1 = _mm256_add_epi16(sum1, coeff2);
            sum3 = _mm256_add_epi16(sum3, coeff2);

            sum1 = _mm256_srli_epi16(sum1, 2);
            sum3 = _mm256_srli_epi16(sum3, 2);

            sum1 = _mm256_packus_epi16(sum1, sum3);
            sum1 = _mm256_permute4x64_epi64(sum1, 0x00D8);

            _mm256_storeu_si256((__m256i*)&first_line[i], sum1);
        }

        if (i < line_size) {

            __m256i mask = _mm256_load_si256((__m256i*)intrinsic_mask_256_8bit[15]);
            S0 = _mm256_loadu_si256((__m256i*)(src - 1));
            S2 = _mm256_loadu_si256((__m256i*)(src + 1));
            S1 = _mm256_loadu_si256((__m256i*)(src));

            L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
            L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
            L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));

            sum1 = _mm256_add_epi16(L0, L1);
            sum2 = _mm256_add_epi16(L1, L2);

            sum1 = _mm256_add_epi16(sum1, sum2);
            sum1 = _mm256_add_epi16(sum1, coeff2);
            sum1 = _mm256_srli_epi16(sum1, 2);

            sum1 = _mm256_packus_epi16(sum1, sum1);
            sum1 = _mm256_permute4x64_epi64(sum1, 0x00D8);

            _mm256_maskstore_epi64((__int64*)&first_line[i], mask, sum1);


        }

        __m256i M;
        if (bsx == 64) {
            for (i = 0; i < bsy; i += 4) {
                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;
                pfirst -= 4;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;
                pfirst -= 4;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;
                pfirst -= 4;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;
                pfirst -= 4;
            }
        } else if (bsx == 32) {
            for (i = 0; i < bsy; i += 4) {
                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;
                pfirst -= 4;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;
                pfirst -= 4;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;
                pfirst -= 4;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;
                pfirst -= 4;
            }
        } else if (bsx == 16) {
            __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
            for (i = 0; i < bsy; i += 4) {
                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
                pfirst -= 4;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
                pfirst -= 4;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
                pfirst -= 4;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
                pfirst -= 4;
            }
        } else {
            __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[7]);
            for (i = 0; i < bsy; i += 4) {
                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
                pfirst -= 4;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
                pfirst -= 4;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
                pfirst -= 4;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
                pfirst -= 4;
            }
        }



        /*
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
        }*/
    } else {//4x4 4x16
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((src[-2] * 3 +  src[-1] * 7 + src[0]  * 5 + src[1]     + 8) >> 4);
            dst[1] = (pel_t)((src[-2]     + (src[-1]     + src[0]) * 3 + src[1]     + 4) >> 3);
            dst[2] = (pel_t)((src[-2]     +  src[-1] * 5 + src[0]  * 7 + src[1] * 3 + 8) >> 4);
            dst[3] = (pel_t)((               src[-1]     + src[0]  * 2 + src[1]     + 2) >> 2);
            dst += i_dst;
        }
    }

}

void intra_pred_ang_xy_23_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy)
{
    int i;
    UNUSED_PARAMETER(dir_mode);

    if (bsx > 8) {
        ALIGN32(pel_t first_line[64 + 512]);
        int left_size = (bsy << 3) - 1;
        int top_size = bsx - 7;
        int line_size = left_size + top_size;
        pel_t *pfirst = first_line + left_size - 7;
        pel_t *pfirst1 = first_line;
        pel_t *src_org = src;

        src -= bsy;

        __m256i coeff0 = _mm256_setr_epi16(7, 3, 5, 1, 3, 1, 1, 0, 7, 3, 5, 1, 3, 1, 1, 0);
        __m256i coeff1 = _mm256_setr_epi16(15, 7, 13, 3, 11, 5, 9, 1, 15, 7, 13, 3, 11, 5, 9, 1);
        __m256i coeff2 = _mm256_setr_epi16(9, 5, 11, 3, 13, 7, 15, 2, 9, 5, 11, 3, 13, 7, 15, 2);
        __m256i coeff3 = _mm256_setr_epi16(1, 1, 3, 1, 5, 3, 7, 1, 1, 1, 3, 1, 5, 3, 7, 1);
        __m256i coeff4 = _mm256_setr_epi16(16, 8, 16, 4, 16, 8, 16, 2, 16, 8, 16, 4, 16, 8, 16, 2);
        __m256i coeff5 = _mm256_setr_epi16(1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1, 8);

        __m256i p00, p10, p20, p30;
        __m256i p01, p11, p21, p31;
        __m256i res1, res2;
        __m256i L0, L1, L2, L3;


        __m256i H0, H1, H2;

        if (bsy == 4) {
            L0 = _mm256_setr_epi16(src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1],
                                   src[1], src[1], src[1], src[1], src[1], src[1], src[1], src[1]);//-1 3

            L1 = _mm256_setr_epi16(src[0], src[0], src[0], src[0], src[0], src[0], src[0], src[0],
                                   src[2], src[2], src[2], src[2], src[2], src[2], src[2], src[2]);//0 4

            L2 = _mm256_setr_epi16(src[1], src[1], src[1], src[1], src[1], src[1], src[1], src[1],
                                   src[3], src[3], src[3], src[3], src[3], src[3], src[3], src[3]);//1 5

            L3 = _mm256_setr_epi16(src[2], src[2], src[2], src[2], src[2], src[2], src[2], src[2],
                                   src[4], src[4], src[4], src[4], src[4], src[4], src[4], src[4]);//2 6

            src += 4;

            for (i = 0; i < left_size + 1; i += 32) {
                p00 = _mm256_mullo_epi16(L0, coeff0);//-1
                p10 = _mm256_mullo_epi16(L1, coeff1);//0
                p20 = _mm256_mullo_epi16(L2, coeff2);//1
                p30 = _mm256_mullo_epi16(L3, coeff3);//2
                p00 = _mm256_add_epi16(p00, coeff4);
                p00 = _mm256_add_epi16(p00, p10);
                p00 = _mm256_add_epi16(p00, p20);
                p00 = _mm256_add_epi16(p00, p30);
                p00 = _mm256_mullo_epi16(p00, coeff5);
                p00 = _mm256_srli_epi16(p00, 5);

                L0 = _mm256_setr_epi16(src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1],
                                       src[1], src[1], src[1], src[1], src[1], src[1], src[1], src[1]);//-1 3

                p01 = _mm256_mullo_epi16(L1, coeff0);//0
                p11 = _mm256_mullo_epi16(L2, coeff1);//1
                p21 = _mm256_mullo_epi16(L3, coeff2);//2
                p31 = _mm256_mullo_epi16(L0, coeff3);//3
                p01 = _mm256_add_epi16(p01, coeff4);
                p01 = _mm256_add_epi16(p01, p11);
                p01 = _mm256_add_epi16(p01, p21);
                p01 = _mm256_add_epi16(p01, p31);
                p01 = _mm256_mullo_epi16(p01, coeff5);
                p01 = _mm256_srli_epi16(p01, 5);

                res1 = _mm256_packus_epi16(p00, p01);
                _mm256_storeu_si256((__m256i*)pfirst1, res1);

            }

        } else {

            L0 = _mm256_setr_epi16(src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1],
                                   src[3], src[3], src[3], src[3], src[3], src[3], src[3], src[3]);//-1 3

            L1 = _mm256_setr_epi16(src[0], src[0], src[0], src[0], src[0], src[0], src[0], src[0],
                                   src[4], src[4], src[4], src[4], src[4], src[4], src[4], src[4]);//0 4

            L2 = _mm256_setr_epi16(src[1], src[1], src[1], src[1], src[1], src[1], src[1], src[1],
                                   src[5], src[5], src[5], src[5], src[5], src[5], src[5], src[5]);//1 5

            L3 = _mm256_setr_epi16(src[2], src[2], src[2], src[2], src[2], src[2], src[2], src[2],
                                   src[6], src[6], src[6], src[6], src[6], src[6], src[6], src[6]);//2 6

            src += 4;

            for (i = 0; i < left_size + 1; i += 64, src += 4) {
                p00 = _mm256_mullo_epi16(L0, coeff0);//-1 3
                p10 = _mm256_mullo_epi16(L1, coeff1);// 0 4
                p20 = _mm256_mullo_epi16(L2, coeff2);// 1 5
                p30 = _mm256_mullo_epi16(L3, coeff3);// 2 6
                p00 = _mm256_add_epi16(p00, coeff4);
                p00 = _mm256_add_epi16(p00, p10);
                p00 = _mm256_add_epi16(p00, p20);
                p00 = _mm256_add_epi16(p00, p30);
                p00 = _mm256_mullo_epi16(p00, coeff5);
                p00 = _mm256_srli_epi16(p00, 5);

                L0 = _mm256_setr_epi16(src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1],
                                       src[3], src[3], src[3], src[3], src[3], src[3], src[3], src[3]);//3 7

                p01 = _mm256_mullo_epi16(L1, coeff0);//0 4
                p11 = _mm256_mullo_epi16(L2, coeff1);//1 5
                p21 = _mm256_mullo_epi16(L3, coeff2);//2 6
                p31 = _mm256_mullo_epi16(L0, coeff3);//3 7
                p01 = _mm256_add_epi16(p01, coeff4);
                p01 = _mm256_add_epi16(p01, p11);
                p01 = _mm256_add_epi16(p01, p21);
                p01 = _mm256_add_epi16(p01, p31);
                p01 = _mm256_mullo_epi16(p01, coeff5);
                p01 = _mm256_srli_epi16(p01, 5);

                res1 = _mm256_packus_epi16(p00, p01);

                L1 = _mm256_setr_epi16(src[0], src[0], src[0], src[0], src[0], src[0], src[0], src[0],
                                       src[4], src[4], src[4], src[4], src[4], src[4], src[4], src[4]);//4 8

                p00 = _mm256_mullo_epi16(L2, coeff0);//1 5
                p10 = _mm256_mullo_epi16(L3, coeff1);//2 6
                p20 = _mm256_mullo_epi16(L0, coeff2);//3 7
                p30 = _mm256_mullo_epi16(L1, coeff3);//4 8
                p00 = _mm256_add_epi16(p00, coeff4);
                p00 = _mm256_add_epi16(p00, p10);
                p00 = _mm256_add_epi16(p00, p20);
                p00 = _mm256_add_epi16(p00, p30);
                p00 = _mm256_mullo_epi16(p00, coeff5);
                p00 = _mm256_srli_epi16(p00, 5);

                L2 = _mm256_setr_epi16(src[1], src[1], src[1], src[1], src[1], src[1], src[1], src[1],
                                       src[5], src[5], src[5], src[5], src[5], src[5], src[5], src[5]);//5 9

                p01 = _mm256_mullo_epi16(L3, coeff0);//2 6
                p11 = _mm256_mullo_epi16(L0, coeff1);//3 7
                p21 = _mm256_mullo_epi16(L1, coeff2);//4 8
                p31 = _mm256_mullo_epi16(L2, coeff3);//5 9
                p01 = _mm256_add_epi16(p01, coeff4);
                p01 = _mm256_add_epi16(p01, p11);
                p01 = _mm256_add_epi16(p01, p21);
                p01 = _mm256_add_epi16(p01, p31);
                p01 = _mm256_mullo_epi16(p01, coeff5);
                p01 = _mm256_srli_epi16(p01, 5);

                res2 = _mm256_packus_epi16(p00, p01);
                p00 = _mm256_permute2x128_si256(res1, res2, 0x0020);
                _mm256_storeu_si256((__m256i*)pfirst1, p00);
                pfirst1 += 32;

                p00 = _mm256_permute2x128_si256(res1, res2, 0x0031);
                _mm256_storeu_si256((__m256i*)pfirst1, p00);

                pfirst1 += 32;

                src += 4;

                L0 = _mm256_setr_epi16(src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1], src[-1],
                                       src[3], src[3], src[3], src[3], src[3], src[3], src[3], src[3]);

                L1 = _mm256_setr_epi16(src[0], src[0], src[0], src[0], src[0], src[0], src[0], src[0],
                                       src[4], src[4], src[4], src[4], src[4], src[4], src[4], src[4]);

                L2 = _mm256_setr_epi16(src[1], src[1], src[1], src[1], src[1], src[1], src[1], src[1],
                                       src[5], src[5], src[5], src[5], src[5], src[5], src[5], src[5]);

                L3 = _mm256_setr_epi16(src[2], src[2], src[2], src[2], src[2], src[2], src[2], src[2],
                                       src[6], src[6], src[6], src[6], src[6], src[6], src[6], src[6]);
            }
        }

        src = src_org + 1;
        __m256i S0, S1, S2;
        coeff2 = _mm256_set1_epi16(2);
        for (; i < line_size; i += 32, src += 32) {

            S0 = _mm256_loadu_si256((__m256i*)(src));
            S1 = _mm256_loadu_si256((__m256i*)(src + 1));
            S2 = _mm256_loadu_si256((__m256i*)(src - 1));

            L0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 0));
            L1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 0));
            L2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 0));

            H0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S0, 1));
            H1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S1, 1));
            H2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(S2, 1));

            p00 = _mm256_mullo_epi16(L0, coeff2);
            p10 = _mm256_add_epi16(L1, L2);
            p00 = _mm256_add_epi16(p00, coeff2);
            p00 = _mm256_add_epi16(p00, p10);
            p00 = _mm256_srli_epi16(p00, 2);

            p01 = _mm256_mullo_epi16(H0, coeff2);
            p11 = _mm256_add_epi16(H1, H2);
            p01 = _mm256_add_epi16(p01, coeff2);
            p01 = _mm256_add_epi16(p01, p11);
            p01 = _mm256_srli_epi16(p01, 2);

            p00 = _mm256_packus_epi16(p00, p01);
            p00 = _mm256_permute4x64_epi64(p00, 0x00D8);
            _mm256_storeu_si256((__m256i*)&first_line[i], p00);
        }

        __m256i M;
        if (bsx == 64) {
            for (i = 0; i < bsy; i += 4) {
                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;
                pfirst -= 8;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;
                pfirst -= 8;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;
                pfirst -= 8;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_storeu_si256((__m256i*)dst, M);
                M = _mm256_lddqu_si256((__m256i*)(pfirst + 32));
                _mm256_storeu_si256((__m256i*)(dst + 32), M);
                dst += i_dst;
                pfirst -= 8;
            }
        } else if (bsx == 32) {
            for (i = 0; i < bsy; i += 4) {
                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;
                pfirst -= 8;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;
                pfirst -= 8;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;
                pfirst -= 8;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_storeu_si256((__m256i*)dst, M);
                dst += i_dst;
                pfirst -= 8;
            }
        } else if (bsx == 16) {
            __m256i mask = _mm256_lddqu_si256((__m256i*)intrinsic_mask_256_8bit[15]);
            for (i = 0; i < bsy; i += 4) {
                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
                pfirst -= 8;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
                pfirst -= 8;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
                pfirst -= 8;

                M = _mm256_lddqu_si256((__m256i*)pfirst);
                _mm256_maskstore_epi64((__int64*)dst, mask, M);
                dst += i_dst;
                pfirst -= 8;
            }
        }

        /*for (i = 0; i < bsy; i++) {
            memcpy(dst, pfirst, bsx * sizeof(pel_t));
            dst += i_dst;
            pfirst -= 8;
        }*/
    } else {//8x8 8x32 4x4 4x16------128bit is enough
        intra_pred_ang_xy_23_sse128(src, dst, i_dst, dir_mode, bsx, bsy);
        return;
    }
}

