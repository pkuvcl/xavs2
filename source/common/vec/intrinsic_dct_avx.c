/*
 * intrinsic_dct_avx.c
 *
 * Description of this file:
 *    AVX2 assembly functions of DCT module of the xavs2 library
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

#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE3
#include <tmmintrin.h>  // SSSE3
#include <immintrin.h>  // AVX and AVX2

#include "../basic_types.h"
#include "intrinsic.h"
#include "../avs2_defs.h"

/* disable warnings */
#ifdef _MSC_VER
#pragma warning(disable:4127)  // warning C4127: 条件表达式是常量
#endif

#define pair_set_epi16(a, b) \
    _mm_set_epi16(b, a, b, a, b, a, b, a)

/* ---------------------------------------------------------------------------
 * functions defined in this file:
 * dct16, dct32
 */

ALIGN32(static const int16_t tab_dct_4[][8]) = {
    { 32,  32,  32,  32,  32,  32,  32,  32 },
    { 42,  17,  42,  17,  42,  17,  42,  17 },
    { 32, -32,  32, -32,  32, -32,  32, -32 },
    { 17, -42,  17, -42,  17, -42,  17, -42 },
};


ALIGN32(int16_t tab_dct_16_avx2[][16][16]) = {
    {
        // order is 0 7 3 4 1 6 2 5, for dct 1
        { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 }, // 0
        { 45, 4, 35, 29, 43, 13, 40, 21, 45, 4, 35, 29, 43, 13, 40, 21 }, // 1
        { 44, 9, 38, 25, 44, 9, 38, 25, 44, 9, 38, 25, 44, 9, 38, 25 }, // 2
        { 43, -13, -21, -40, 29, -35, 4, -45, 43, -13, -21, -40, 29, -35, 4, -45 }, // 3
        { 42, 17, 42, 17, 42, 17, 42, 17, 42, 17, 42, 17, 42, 17, 42, 17 }, // 4
        { 40, 21, -43, -13, 4, 45, -35, 29, 40, 21, -43, -13, 4, 45, -35, 29 }, // 5
        { 38, -25, -9, -44, 38, -25, -9, -44, 38, -25, -9, -44, 38, -25, -9, -44 }, // 6
        { 35, -29, 4, 45, -21, -40, -43, 13, 35, -29, 4, 45, -21, -40, -43, 13 }, // 7
        { 32, -32, 32, -32, 32, -32, 32, -32, 32, -32, 32, -32, 32, -32, 32, -32 }, // 8
        { 29, 35, 45, -4, -40, 21, -13, -43, 29, 35, 45, -4, -40, 21, -13, -43 }, // 9
        { 25, 38, -44, 9, 25, 38, -44, 9, 25, 38, -44, 9, 25, 38, -44, 9 }, // 10
        { 21, -40, 13, -43, -45, 4, 29, 35, 21, -40, 13, -43, -45, 4, 29, 35 }, // 11
        { 17, -42, 17, -42, 17, -42, 17, -42, 17, -42, 17, -42, 17, -42, 17, -42 }, // 12
        { 13, 43, -40, 21, -35, -29, 45, 4, 13, 43, -40, 21, -35, -29, 45, 4 }, // 13
        { 9, -44, -25, 38, 9, -44, -25, 38, 9, -44, -25, 38, 9, -44, -25, 38 }, // 14
        { 4, -45, -29, 35, -13, 43, 21, -40, 4, -45, -29, 35, -13, 43, 21, -40 }  // 15
    },
    {
        { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 }, // 0
        { 45, -45, 35, -35, 43, -43, 40, -40, 45, -45, 35, -35, 43, -43, 40, -40 }, // 1
        { 44, 44, 9, 9, 38, 38, 25, 25, 44, 44, 9, 9, 38, 38, 25, 25 }, // 2
        { 43, -43, -21, 21, 29, -29, 4, -4, 43, -43, -21, 21, 29, -29, 4, -4 }, // 3
        { 42, 42, -42, -42, 17, 17, -17, -17, 42, 42, -42, -42, 17, 17, -17, -17 }, // 4
        { 40, -40, -43, 43, 4, -4, -35, 35, 40, -40, -43, 43, 4, -4, -35, 35 }, // 5
        { 38, 38, -25, -25, -9, -9, -44, -44, 38, 38, -25, -25, -9, -9, -44, -44 }, // 6
        { 35, -35, 4, -4, -21, 21, -43, 43, 35, -35, 4, -4, -21, 21, -43, 43 }, // 7
        { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 }, // 8
        { 29, -29, 45, -45, -40, 40, -13, 13, 29, -29, 45, -45, -40, 40, -13, 13 }, // 9
        { 25, 25, 38, 38, -44, -44, 9, 9, 25, 25, 38, 38, -44, -44, 9, 9 }, // 10
        { 21, -21, 13, -13, -45, 45, 29, -29, 21, -21, 13, -13, -45, 45, 29, -29 }, // 11
        { 17, 17, -17, -17, -42, -42, 42, 42, 17, 17, -17, -17, -42, -42, 42, 42 }, // 12
        { 13, -13, -40, 40, -35, 35, 45, -45, 13, -13, -40, 40, -35, 35, 45, -45 }, // 13
        { 9, 9, -44, -44, -25, -25, 38, 38, 9, 9, -44, -44, -25, -25, 38, 38 }, // 14
        { 4, -4, -29, 29, -13, 13, 21, -21, 4, -4, -29, 29, -13, 13, 21, -21 }  // 15
    },
    {
        { 4, -4, 29, -29, 13, -13, 21, -21, 4, -4, 29, -29, 13, -13, 21, -21 }, // 0
        { -13, 13, -40, 40, -35, 35, -45, 45, -13, 13, -40, 40, -35, 35, -45, 45 }, // 1
        { 21, -21, -13, 13, 45, -45, 29, -29, 21, -21, -13, 13, 45, -45, 29, -29 }, // 2
        { -29, 29, 45, -45, -40, 40, 13, -13, -29, 29, 45, -45, -40, 40, 13, -13 }, // 3
        { 35, -35, -4, 4, 21, -21, -43, 43, 35, -35, -4, 4, 21, -21, -43, 43 }, // 4
        { -40, 40, -43, 43, 4, -4, 35, -35, -40, 40, -43, 43, 4, -4, 35, -35 }, // 5
        { 43, -43, 21, -21, -29, 29, 4, -4, 43, -43, 21, -21, -29, 29, 4, -4 }, // 6
        { -45, 45, 35, -35, 43, -43, -40, 40, -45, 45, 35, -35, 43, -43, -40, 40 }  // 7
    }
};

ALIGN32(int16_t tab_dct_16_shuffle_avx2[][16]) = {
    {
        0x0100, 0x0F0E, 0x0706, 0x0908, 0x0302, 0x0D0C, 0x0504, 0x0B0A,
        0x0100, 0x0F0E, 0x0706, 0x0908, 0x0302, 0x0D0C, 0x0504, 0x0B0A
    },
    {
        0x0F0E, 0x0100, 0x0908, 0x0706, 0x0D0C, 0x0302, 0x0B0A, 0x0504,
        0x0F0E, 0x0100, 0x0908, 0x0706, 0x0D0C, 0x0302, 0x0B0A, 0x0504,
    },
    {
        0x0100, 0x0706, 0x0302, 0x0504, 0x0F0E, 0x0908, 0x0D0C, 0x0B0A,
        0x0100, 0x0706, 0x0302, 0x0504, 0x0F0E, 0x0908, 0x0D0C, 0x0B0A,
    },
    {
        0x0F0E, 0x0908, 0x0D0C, 0x0B0A, 0x0100, 0x0706, 0x0302, 0x0504,
        0x0F0E, 0x0908, 0x0D0C, 0x0B0A, 0x0100, 0x0706, 0x0302, 0x0504
    }
};
ALIGN32(static const int16_t tab_dct_8x32_avx2[][16]) = {
    { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 }, //0
    { 44, 9, 38, 25, 44, 9, 38, 25, 44, 9, 38, 25, 44, 9, 38, 25 }, //1
    { 42, 17, 42, 17, 42, 17, 42, 17, 42, 17, 42, 17, 42, 17, 42, 17 }, //2
    { 38, -25, -9, -44, 38, -25, -9, -44, 38, -25, -9, -44, 38, -25, -9, -44 }, //3
    { 32, -32, 32, -32, 32, -32, 32, -32, 32, -32, 32, -32, 32, -32, 32, -32 }, //4
    { 25, 38, -44, 9, 25, 38, -44, 9, 25, 38, -44, 9, 25, 38, -44, 9 }, //5
    { 17, -42, 17, -42, 17, -42, 17, -42, 17, -42, 17, -42, 17, -42, 17, -42 }, //6
    { 9, -44, -25, 38, 9, -44, -25, 38, 9, -44, -25, 38, 9, -44, -25, 38 }  //7
};
ALIGN32(static const int32_t tab_dct_8x8_avx2[][8] )= {
    { 32, 32, 32, 32, 32, 32, 32, 32 },//     0
    { 44, 38, 44, 38, 25, 9, 25, 9 },//
    { 42, 17, 42, 17, -17, -42, -17, -42 },//     2
    { 38, -9, 38, -9, -44, -25, -44, -25 },//
    { 32, -32, 32, -32, -32, 32, -32, 32 },//     4
    { 25, -44, 25, -44, 9, 38, 9, 38 },//
    { 17, -42, 17, -42, 42, -17, 42, -17 },//     6
    { 9, -25, 9, -25, 38, -44, 38, -44 } //
};

ALIGN32(static const int16_t tab_dct1_4[][8]) = {
    { 32,  32,  32,  32, 32,  32,  32,  32 },
    { 42,  17, -17, -42, 42,  17, -17, -42 },
    { 32, -32, -32,  32, 32, -32, -32,  32 },
    { 17, -42,  42, -17, 17, -42,  42, -17 }
};

ALIGN32(static const int16_t tab_dct_8[][8]) = {
    { 0x0100, 0x0F0E, 0x0706, 0x0908, 0x0302, 0x0D0C, 0x0504, 0x0B0A },

    { 32,  32,  32,  32,  32,  32,  32,  32 },
    { 32, -32,  32, -32,  32, -32,  32, -32 },
    { 42,  17,  42,  17,  42,  17,  42,  17 },
    { 17, -42,  17, -42,  17, -42,  17, -42 },
    { 44,   9,  38,  25,  44,   9,  38,  25 },
    { 38, -25,  -9, -44,  38, -25,  -9, -44 },
    { 25,  38, -44,   9,  25,  38, -44,   9 },
    {  9, -44, -25,  38,   9, -44, -25,  38 },

    { 42,  42, -42, -42,  17,  17, -17, -17 },
    { 17,  17, -17, -17, -42, -42,  42,  42 },
    { 44, -44,   9,  -9,  38, -38,  25, -25 },
    { 38, -38, -25,  25,  -9,   9, -44,  44 },
    { 25, -25,  38, -38, -44,  44,   9,  -9 },
    {  9,  -9, -44,  44, -25,  25,  38, -38 }
};


ALIGN32(static const int16_t tab_dct_8_1[][8]) = {
    { 32,  32,  32,  32,  32,  32,  32,  32 },
    { 44,  38,  25,   9, - 9, -25, -38, -44 },
    { 42,  17, -17, -42, -42, -17,  17,  42 },
    { 38, - 9, -44, -25,  25,  44,   9, -38 },
    { 32, -32, -32,  32,  32, -32, -32,  32 },
    { 25, -44,   9,  38, -38, - 9,  44, -25 },
    { 17, -42,  42, -17, -17,  42, -42,  17 },
    {  9, -25,  38, -44,  44, -38,  25, - 9 }
};

ALIGN32(static const int16_t tab_dct_16_0[][8]) = {
    { 0x0F0E, 0x0D0C, 0x0B0A, 0x0908, 0x0706, 0x0504, 0x0302, 0x0100 },  // 0
    { 0x0100, 0x0F0E, 0x0706, 0x0908, 0x0302, 0x0D0C, 0x0504, 0x0B0A },  // 1
    { 0x0100, 0x0706, 0x0302, 0x0504, 0x0F0E, 0x0908, 0x0D0C, 0x0B0A },  // 2
    { 0x0F0E, 0x0908, 0x0D0C, 0x0B0A, 0x0100, 0x0706, 0x0302, 0x0504 }   // 3
};

ALIGN32(static const int16_t tab_dct_16_1[][8]) = {
    { 45,  43,  40,  35,  29,  21,  13,   4 },  //  0
    { 43,  29,   4, -21, -40, -45, -35, -13 },  //  1
    { 40,   4, -35, -43, -13,  29,  45,  21 },  //  2
    { 35, -21, -43,   4,  45,  13, -40, -29 },  //  3
    { 29, -40, -13,  45,  -4, -43,  21,  35 },  //  4
    { 21, -45,  29,  13, -43,  35,   4, -40 },  //  5
    { 13, -35,  45, -40,  21,   4, -29,  43 },  //  6
    {  4, -13,  21, -29,  35, -40,  43, -45 },  //  7
    { 42,  42, -42, -42,  17,  17, -17, -17 },  //  8
    { 17,  17, -17, -17, -42, -42,  42,  42 },  //  9
    { 44,  44,   9,   9,  38,  38,  25,  25 },  // 10
    { 38,  38, -25, -25,  -9,  -9, -44, -44 },  // 11
    { 25,  25,  38,  38, -44, -44,   9,   9 },  // 12
    {  9,   9, -44, -44, -25, -25,  38,  38 },  // 13

#define MAKE_COEF(a0, a1, a2, a3, a4, a5, a6, a7) \
    { (a0), -(a0), (a3), -(a3), (a1), -(a1), (a2), -(a2) }, \
    { (a7), -(a7), (a4), -(a4), (a6), -(a6), (a5), -(a5) },

    MAKE_COEF(45,  43,  40,  35,  29,  21,  13,   4)
    MAKE_COEF(43,  29,   4, -21, -40, -45, -35, -13)
    MAKE_COEF(40,   4, -35, -43, -13,  29,  45,  21)
    MAKE_COEF(35, -21, -43,   4,  45,  13, -40, -29)
    MAKE_COEF(29, -40, -13,  45,  -4, -43,  21,  35)
    MAKE_COEF(21, -45,  29,  13, -43,  35,   4, -40)
    MAKE_COEF(13, -35,  45, -40,  21,   4, -29,  43)
    MAKE_COEF( 4, -13,  21, -29,  35, -40,  43, -45)
#undef MAKE_COEF
};

ALIGN32(static const int16_t tab_dct_32_0[][8]) = {
    { 0x0F0E, 0x0100, 0x0908, 0x0706, 0x0D0C, 0x0302, 0x0B0A, 0x0504 },  // 0
};

ALIGN32(static const int16_t tab_dct_32_1[][8]) = {
    { 44, -44,   9,  -9,  38, -38,  25, -25 },          //  0
    { 38, -38, -25,  25,  -9,   9, -44,  44 },          //  1
    { 25, -25,  38, -38, -44,  44,   9,  -9 },          //  2
    {  9,  -9, -44,  44, -25,  25,  38, -38 },          //  3

#define MAKE_COEF8(a0, a1, a2, a3, a4, a5, a6, a7) \
    { (a0), (a7), (a3), (a4), (a1), (a6), (a2), (a5) },

    MAKE_COEF8(45,  43,  40,  35,  29,  21,  13,   4)   //  4
    MAKE_COEF8(43,  29,   4, -21, -40, -45, -35, -13)   //  5
    MAKE_COEF8(40,   4, -35, -43, -13,  29,  45,  21)   //  6
    MAKE_COEF8(35, -21, -43,   4,  45,  13, -40, -29)   //  7
    MAKE_COEF8(29, -40, -13,  45,  -4, -43,  21,  35)   //  8
    MAKE_COEF8(21, -45,  29,  13, -43,  35,   4, -40)   //  9
    MAKE_COEF8(13, -35,  45, -40,  21,   4, -29,  43)   // 10
    MAKE_COEF8( 4, -13,  21, -29,  35, -40,  43, -45)   // 11
#undef MAKE_COEF8

#define MAKE_COEF16(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15) \
    { (a00), (a07), (a03), (a04), (a01), (a06), (a02), (a05) }, \
    { (a15), (a08), (a12), (a11), (a14), (a09), (a13), (a10) },

    MAKE_COEF16(45,  45,  44,  43,  41,  39,  36,  34,  30,  27,  23,  19,  15,  11,   7,   2)  // 12
    MAKE_COEF16(45,  41,  34,  23,  11,  -2, -15, -27, -36, -43, -45, -44, -39, -30, -19,  -7)  // 14
    MAKE_COEF16(44,  34,  15,  -7, -27, -41, -45, -39, -23,  -2,  19,  36,  45,  43,  30,  11)  // 16
    MAKE_COEF16(43,  23,  -7, -34, -45, -36, -11,  19,  41,  44,  27,  -2, -30, -45, -39, -15)  // 18
    MAKE_COEF16(41,  11, -27, -45, -30,   7,  39,  43,  15, -23, -45, -34,   2,  36,  44,  19)  // 20
    MAKE_COEF16(39,  -2, -41, -36,   7,  43,  34, -11, -44, -30,  15,  45,  27, -19, -45, -23)  // 22
    MAKE_COEF16(36, -15, -45, -11,  39,  34, -19, -45,  -7,  41,  30, -23, -44,  -2,  43,  27)  // 24
    MAKE_COEF16(34, -27, -39,  19,  43, -11, -45,   2,  45,   7, -44, -15,  41,  23, -36, -30)  // 26
    MAKE_COEF16(30, -36, -23,  41,  15, -44,  -7,  45,  -2, -45,  11,  43, -19, -39,  27,  34)  // 28
    MAKE_COEF16(27, -43,  -2,  44, -23, -30,  41,   7, -45,  19,  34, -39, -11,  45, -15, -36)  // 30
    MAKE_COEF16(23, -45,  19,  27, -45,  15,  30, -44,  11,  34, -43,   7,  36, -41,   2,  39)  // 32
    MAKE_COEF16(19, -44,  36,  -2, -34,  45, -23, -15,  43, -39,   7,  30, -45,  27,  11, -41)  // 34
    MAKE_COEF16(15, -39,  45, -30,   2,  27, -44,  41, -19, -11,  36, -45,  34,  -7, -23,  43)  // 36
    MAKE_COEF16(11, -30,  43, -45,  36, -19,  -2,  23, -39,  45, -41,  27,  -7, -15,  34, -44)  // 38
    MAKE_COEF16( 7, -19,  30, -39,  44, -45,  43, -36,  27, -15,   2,  11, -23,  34, -41,  45)  // 40
    MAKE_COEF16( 2,  -7,  11, -15,  19, -23,  27, -30,  34, -36,  39, -41,  43, -44,  45, -45)  // 42
#undef MAKE_COEF16

    {  32,  32,  32,  32,  32,  32,  32,  32 }, // 44
    {  32,  32, -32, -32, -32, -32,  32,  32 }, // 45
    {  42,  42,  17,  17, -17, -17, -42, -42 }, // 46
    { -42, -42, -17, -17,  17,  17,  42,  42 }, // 47
    {  17,  17, -42, -42,  42,  42, -17, -17 }, // 48
    { -17, -17,  42,  42, -42, -42,  17,  17 }, // 49

#define MAKE_COEF16(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15) \
    { (a00), (a00), (a01), (a01), (a02), (a02), (a03), (a03) }, \
    { (a04), (a04), (a05), (a05), (a06), (a06), (a07), (a07) }, \
    { (a08), (a08), (a09), (a09), (a10), (a10), (a11), (a11) }, \
    { (a12), (a12), (a13), (a13), (a14), (a14), (a15), (a15) },

    MAKE_COEF16(44, 38, 25, 9, -9, -25, -38, -44, -44, -38, -25, -9, 9, 25, 38, 44) // 50
    MAKE_COEF16(38, -9, -44, -25, 25, 44, 9, -38, -38, 9, 44, 25, -25, -44, -9, 38) // 54

    // TODO: convert below table here
#undef MAKE_COEF16

    {  25,  25, -44, -44,   9,   9,  38,  38 }, // 58
    { -38, -38,  -9,  -9,  44,  44, -25, -25 }, // 59
    { -25, -25,  44,  44,  -9,  -9, -38, -38 }, // 60
    {  38,  38,   9,   9, -44, -44,  25,  25 }, // 61
    {   9,   9, -25, -25,  38,  38, -44, -44 }, // 62
    {  44,  44, -38, -38,  25,  25,  -9,  -9 }, // 63
    {  -9,  -9,  25,  25, -38, -38,  44,  44 }, // 64
    { -44, -44,  38,  38, -25, -25,   9,   9 }, // 65
    {  45,  45,  43,  43,  40,  40,  35,  35 }, // 66
    {  29,  29,  21,  21,  13,  13,   4,   4 }, // 67
    {  -4,  -4, -13, -13, -21, -21, -29, -29 }, // 68
    { -35, -35, -40, -40, -43, -43, -45, -45 }, // 69
    {  43,  43,  29,  29,   4,   4, -21, -21 }, // 70
    { -40, -40, -45, -45, -35, -35, -13, -13 }, // 71
    {  13,  13,  35,  35,  45,  45,  40,  40 }, // 72
    {  21,  21,  -4,  -4, -29, -29, -43, -43 }, // 73
    {  40,  40,   4,   4, -35, -35, -43, -43 }, // 74
    { -13, -13,  29,  29,  45,  45,  21,  21 }, // 75
    { -21, -21, -45, -45, -29, -29,  13,  13 }, // 76
    {  43,  43,  35,  35,  -4,  -4, -40, -40 }, // 77
    {  35,  35, -21, -21, -43, -43,   4,   4 }, // 78
    {  45,  45,  13,  13, -40, -40, -29, -29 }, // 79
    {  29,  29,  40,  40, -13, -13, -45, -45 }, // 80
    {  -4,  -4,  43,  43,  21,  21, -35, -35 }, // 81
    {  29,  29, -40, -40, -13, -13,  45,  45 }, // 82
    {  -4,  -4, -43, -43,  21,  21,  35,  35 }, // 83
    { -35, -35, -21, -21,  43,  43,   4,   4 }, // 84
    { -45, -45,  13,  13,  40,  40, -29, -29 }, // 85
    {  21,  21, -45, -45,  29,  29,  13,  13 }, // 86
    { -43, -43,  35,  35,   4,   4, -40, -40 }, // 87
    {  40,  40,  -4,  -4, -35, -35,  43,  43 }, // 88
    { -13, -13, -29, -29,  45,  45, -21, -21 }, // 89
    {  13,  13, -35, -35,  45,  45, -40, -40 }, // 90
    {  21,  21,   4,   4, -29, -29,  43,  43 }, // 91
    { -43, -43,  29,  29,  -4,  -4, -21, -21 }, // 92
    {  40,  40, -45, -45,  35,  35, -13, -13 }, // 93
    {   4,   4, -13, -13,  21,  21, -29, -29 }, // 94
    {  35,  35, -40, -40,  43,  43, -45, -45 }, // 95
    {  45,  45, -43, -43,  40,  40, -35, -35 }, // 96
    {  29,  29, -21, -21,  13,  13,  -4,  -4 }, // 97

#define MAKE_COEF16(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15) \
    { (a00), -(a00), (a01), -(a01), (a02), -(a02), (a03), -(a03) }, \
    { (a04), -(a04), (a05), -(a05), (a06), -(a06), (a07), -(a07) }, \
    { (a08), -(a08), (a09), -(a09), (a10), -(a10), (a11), -(a11) }, \
    { (a12), -(a12), (a13), -(a13), (a14), -(a14), (a15), -(a15) },

    MAKE_COEF16(45,  45,  44,  43,  41,  39,  36,  34,  30,  27,  23,  19,  15,  11,   7,   2)  // 98
    MAKE_COEF16(45,  41,  34,  23,  11,  -2, -15, -27, -36, -43, -45, -44, -39, -30, -19,  -7)  //102
    MAKE_COEF16(44,  34,  15,  -7, -27, -41, -45, -39, -23,  -2,  19,  36,  45,  43,  30,  11)  //106
    MAKE_COEF16(43,  23,  -7, -34, -45, -36, -11,  19,  41,  44,  27,  -2, -30, -45, -39, -15)  //110
    MAKE_COEF16(41,  11, -27, -45, -30,   7,  39,  43,  15, -23, -45, -34,   2,  36,  44,  19)  //114
    MAKE_COEF16(39,  -2, -41, -36,   7,  43,  34, -11, -44, -30,  15,  45,  27, -19, -45, -23)  //118
    MAKE_COEF16(36, -15, -45, -11,  39,  34, -19, -45,  -7,  41,  30, -23, -44,  -2,  43,  27)  //122
    MAKE_COEF16(34, -27, -39,  19,  43, -11, -45,   2,  45,   7, -44, -15,  41,  23, -36, -30)  //126
    MAKE_COEF16(30, -36, -23,  41,  15, -44,  -7,  45,  -2, -45,  11,  43, -19, -39,  27,  34)  //130
    MAKE_COEF16(27, -43,  -2,  44, -23, -30,  41,   7, -45,  19,  34, -39, -11,  45, -15, -36)  //134
    MAKE_COEF16(23, -45,  19,  27, -45,  15,  30, -44,  11,  34, -43,   7,  36, -41,   2,  39)  //138
    MAKE_COEF16(19, -44,  36,  -2, -34,  45, -23, -15,  43, -39,   7,  30, -45,  27,  11, -41)  //142
    MAKE_COEF16(15, -39,  45, -30,   2,  27, -44,  41, -19, -11,  36, -45,  34,  -7, -23,  43)  //146
    MAKE_COEF16(11, -30,  43, -45,  36, -19,  -2,  23, -39,  45, -41,  27,  -7, -15,  34, -44)  //150
    MAKE_COEF16( 7, -19,  30, -39,  44, -45,  43, -36,  27, -15,   2,  11, -23,  34, -41,  45)  //154
    MAKE_COEF16( 2,  -7,  11, -15,  19, -23,  27, -30,  34, -36,  39, -41,  43, -44,  45, -45)  //158

#undef MAKE_COEF16
};

/* ---------------------------------------------------------------------------
 * secondary transform
 */
//ALIGN16(const int16_t g_2T[SEC_TR_SIZE * SEC_TR_SIZE]) = {
//    123,  -35,  -8,  -3,    // e0 e1 e2 e3
//    -32, -120,  30,  10,    // f0 f1 f2 f3
//     14,   25, 123, -22,    // g0 g1 g2 g3
//      8,   13,  19, 126     // h0 h1 h2 h3
//};
ALIGN16(static const int16_t g_2T_H[2 * (2 * SEC_TR_SIZE)]) = {
    123, -35, -32, -120,  14,  25,  8,  13, // e0 e1 f0 f1 g0 g1 h0 h1
     -8,  -3,  30,   10, 123, -22, 19, 126  // e2 e3 f2 f3 g2 g3 h2 h3
};

ALIGN16(static const int16_t g_2T_V[8 * (2 * SEC_TR_SIZE)]) = {
    123,  -35, 123,  -35, 123,  -35, 123,  -35, // e0 e1 e0 e1 e0 e1 e0 e1
     -8,   -3,  -8,   -3,  -8,   -3,  -8,   -3, // e2 e3 e2 e3 e2 e3 e2 e3
    -32, -120, -32, -120, -32, -120, -32, -120, // f0 f1 f0 f1 f0 f1 f0 f1
     30,   10,  30,   10,  30,   10,  30,   10, // f2 f3 f2 f3 f2 f3 f2 f3
     14,   25,  14,   25,  14,   25,  14,   25, // g0 g1 g0 g1 g0 g1 g0 g1
    123,  -22, 123,  -22, 123,  -22, 123,  -22, // g2 g3 g2 g3 g2 g3 g2 g3
      8,   13,   8,   13,   8,   13,   8,   13, // h0 h1 h0 h1 h0 h1 h0 h1
     19,  126,  19,  126,  19,  126,  19,  126, // h2 h3 h2 h3 h2 h3 h2 h3
};

/* ---------------------------------------------------------------------------
 * secondary transform (only for 4x4)
 */
//ALIGN16(const int16_t g_2T_C[SEC_TR_SIZE * SEC_TR_SIZE]) = {
//    34,  58,  72,  81,    // e0 e1 e2 e3
//    77,  69,  -7, -75,    // f0 f1 f2 f3
//    79, -33, -75,  58,    // g0 g1 g2 g3
//    55, -84,  73, -28     // h0 h1 h2 h3
//};
ALIGN16(static const int16_t g_2TC_H[2 * (2 * SEC_TR_SIZE)]) = {
    34, 58, 77,  69,  79, -33, 55, -84, // e0 e1 f0 f1 g0 g1 h0 h1
    72, 81, -7, -75, -75,  58, 73, -28  // e2 e3 f2 f3 g2 g3 h2 h3
};

ALIGN16(static const int16_t g_2TC_V[8 * (2 * SEC_TR_SIZE)]) = {
     34,  58,  34,  58,  34,  58,  34,  58, // e0 e1 e0 e1 e0 e1 e0 e1
     72,  81,  72,  81,  72,  81,  72,  81, // e2 e3 e2 e3 e2 e3 e2 e3
     77,  69,  77,  69,  77,  69,  77,  69, // f0 f1 f0 f1 f0 f1 f0 f1
     -7, -75,  -7, -75,  -7, -75,  -7, -75, // f2 f3 f2 f3 f2 f3 f2 f3
     79, -33,  79, -33,  79, -33,  79, -33, // g0 g1 g0 g1 g0 g1 g0 g1
    -75,  58, -75,  58, -75,  58, -75,  58, // g2 g3 g2 g3 g2 g3 g2 g3
     55, -84,  55, -84,  55, -84,  55, -84, // h0 h1 h0 h1 h0 h1 h0 h1
     73, -28,  73, -28,  73, -28,  73, -28, // h2 h3 h2 h3 h2 h3 h2 h3
};

//**************************************************************
//futl
//**************************************************************
ALIGN32(static const int16_t tab_dct_4_avx2_1[][16]) = {
    { 32, 32, 32, 32, 32, 32, 32, 32, 32, -32, 32, -32, 32, -32, 32, -32 },
    { 42, 17, 42, 17, 42, 17, 42, 17, 17, -42, 17, -42, 17, -42, 17, -42 },
};

ALIGN32(static const int16_t tab_dct_4_avx2[][16]) = {
    { 32, 32, 32, 32, 42, 17, 42, 17, 32, 32, 32, 32, 42, 17, 42, 17 },
    { 32, -32, 32, -32, 17, -42, 17, -42, 32, -32, 32, -32, 17, -42, 17, -42 },
};

/* ---------------------------------------------------------------------------
 */
void dct_c_4x4_avx2(const coeff_t *src, coeff_t *dst, int i_src)
{
#define ADD1    0
#define ADD2    64
#define SHIFT1  0
#define SHIFT2  7
    __m256i T20;
    __m256i T30, T31, T32, T40, T41, T42, T50, T51, T60, T70, T71;

    __m256i c_add2 = _mm256_set1_epi32(ADD2);

    __m256i Tab0, Tab1;
    Tab0 = _mm256_load_si256((__m256i*)tab_dct_4_avx2_1[0]);
    Tab1 = _mm256_load_si256((__m256i*)tab_dct_4_avx2_1[1]);

    T20 = _mm256_loadu_si256((__m256i*)(src + 0 * i_src));

    T30 = _mm256_shufflehi_epi16(T20, 0x9C);    //0 1 2 3 4 7 5 6   8 11 9 10 12 15 13 14......
    T31 = _mm256_shufflelo_epi16(T30, 0x9C);    //0 3 1 2 4 7 5 6   8 11 9 10 12 15 13 14......
    T32 = _mm256_permute4x64_epi64(T31, 0xB4);  //0 3 1 2 4 7 5 6   12 15 13 14 8 11 9 10 ......

    T40 = _mm256_hadd_epi16(T32, T32);
    T41 = _mm256_hsub_epi16(T32, T32);
    T42 = _mm256_unpacklo_epi64(T40, T41);

    T50 = _mm256_madd_epi16(T42, _mm256_load_si256((__m256i*)tab_dct_4_avx2[0]));
    T51 = _mm256_madd_epi16(T42, _mm256_load_si256((__m256i*)tab_dct_4_avx2[1]));

    T60 = _mm256_packs_epi32(T50, T51);
    T70 = _mm256_permute2f128_si256(T60, T60, 0x20);
    T71 = _mm256_permute2f128_si256(T60, T60, 0x31);


    T30 = _mm256_madd_epi16(T70, Tab0);
    T31 = _mm256_madd_epi16(T71, Tab0);
    T40 = _mm256_add_epi32(T30, T31);
    T50 = _mm256_srai_epi32(_mm256_add_epi32(T40, c_add2), SHIFT2);

    T30 = _mm256_madd_epi16(T70, Tab1);
    T31 = _mm256_madd_epi16(T71, Tab1);
    T41 = _mm256_sub_epi32(T30, T31);
    T51 = _mm256_srai_epi32(_mm256_add_epi32(T41, c_add2), SHIFT2);

    T60 = _mm256_packs_epi32(T50, T51);
    __m256i mask64 = _mm256_set1_epi8(0xff);
    _mm256_maskstore_epi64((long long *)(dst), mask64, T60);
#undef SHIFT1
#undef ADD1
#undef SHIFT2
#undef ADD2
}


ALIGN32(static const int16_t tab_dct_8_avx2[][16]) = {
    { 0x0100, 0x0F0E, 0x0706, 0x0908, 0x0302, 0x0D0C, 0x0504, 0x0B0A, 0x0100, 0x0F0E, 0x0706, 0x0908, 0x0302, 0x0D0C, 0x0504, 0x0B0A },

    { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 },
    { 32, -32, 32, -32, 32, -32, 32, -32, 32, -32, 32, -32, 32, -32, 32, -32 },
    { 42, 17, 42, 17, 42, 17, 42, 17, 42, 17, 42, 17, 42, 17, 42, 17 },
    { 17, -42, 17, -42, 17, -42, 17, -42, 17, -42, 17, -42, 17, -42, 17, -42 },
    { 44, 9, 38, 25, 44, 9, 38, 25, 44, 9, 38, 25, 44, 9, 38, 25 },
    { 38, -25, -9, -44, 38, -25, -9, -44, 38, -25, -9, -44, 38, -25, -9, -44 },
    { 25, 38, -44, 9, 25, 38, -44, 9, 25, 38, -44, 9, 25, 38, -44, 9 },
    { 9, -44, -25, 38, 9, -44, -25, 38, 9, -44, -25, 38, 9, -44, -25, 38 },

    { 42, 42, -42, -42, 17, 17, -17, -17, 42, 42, -42, -42, 17, 17, -17, -17 },
    { 17, 17, -17, -17, -42, -42, 42, 42, 17, 17, -17, -17, -42, -42, 42, 42 },
    { 44, -44, 9, -9, 38, -38, 25, -25, 44, -44, 9, -9, 38, -38, 25, -25 },
    { 38, -38, -25, 25, -9, 9, -44, 44, 38, -38, -25, 25, -9, 9, -44, 44 },
    { 25, -25, 38, -38, -44, 44, 9, -9, 25, -25, 38, -38, -44, 44, 9, -9 },
    { 9, -9, -44, 44, -25, 25, 38, -38, 9, -9, -44, 44, -25, 25, 38, -38 }
};



/* ---------------------------------------------------------------------------
 */
void dct_c_8x8_avx2(const coeff_t *src, coeff_t *dst, int i_src)
{
#define ADD1    1
#define ADD2    128
#define SHIFT1  1
#define SHIFT2  8

    // Const
    __m256i c_add1 = _mm256_set1_epi32(ADD1);              // add1 = 1
    __m256i c_add2 = _mm256_set1_epi32(ADD2);              // add2 = 128

    // DCT1
    __m256i T00, T01, T02, T03;
    __m256i T10, T11, T12, T13;
    __m256i T20, T21, T22, T23;
    __m256i T30, T31;
    __m256i T40, T41, T42, T43;
    __m256i T50, T51, T52, T53;
    __m256i T60, T61;
    __m256i T70, T71, T72, T73, T74, T75;
    __m256i T80, T81, T82, T83;

    __m256i Tab;

    T00 = _mm256_loadu_si256((__m256i*)(src + 0 * i_src));
    T01 = _mm256_loadu_si256((__m256i*)(src + 2 * i_src));
    T02 = _mm256_loadu_si256((__m256i*)(src + 4 * i_src));
    T03 = _mm256_loadu_si256((__m256i*)(src + 6 * i_src));

    Tab = _mm256_loadu_si256((__m256i*)tab_dct_8_avx2[0]);
    T10 = _mm256_shuffle_epi8(T00, Tab);
    T11 = _mm256_shuffle_epi8(T01, Tab);
    T12 = _mm256_shuffle_epi8(T02, Tab);
    T13 = _mm256_shuffle_epi8(T03, Tab);

    T20 = _mm256_hadd_epi16(T10, T11);
    T21 = _mm256_hadd_epi16(T12, T13);

    T22 = _mm256_hsub_epi16(T10, T11);
    T23 = _mm256_hsub_epi16(T12, T13);

    T30 = _mm256_hadd_epi16(T20, T21);
    T31 = _mm256_hsub_epi16(T20, T21);

    T40 = _mm256_madd_epi16(T30, _mm256_load_si256((__m256i*)tab_dct_8_avx2[1]));
    T40 = _mm256_srai_epi32(_mm256_add_epi32(T40, c_add1), SHIFT1);
    T41 = _mm256_madd_epi16(T30, _mm256_load_si256((__m256i*)tab_dct_8_avx2[2]));
    T41 = _mm256_srai_epi32(_mm256_add_epi32(T41, c_add1), SHIFT1);
    T42 = _mm256_madd_epi16(T31, _mm256_load_si256((__m256i*)tab_dct_8_avx2[3]));
    T42 = _mm256_srai_epi32(_mm256_add_epi32(T42, c_add1), SHIFT1);
    T43 = _mm256_madd_epi16(T31, _mm256_load_si256((__m256i*)tab_dct_8_avx2[4]));
    T43 = _mm256_srai_epi32(_mm256_add_epi32(T43, c_add1), SHIFT1);

    T50 = _mm256_packs_epi32(T40, T42);
    T52 = _mm256_packs_epi32(T41, T43);

    Tab = _mm256_load_si256((__m256i*)tab_dct_8_avx2[5]);
    T40 = _mm256_madd_epi16(T22, Tab);
    T41 = _mm256_madd_epi16(T23, Tab);
    T40 = _mm256_hadd_epi32(T40, T41);
    T40 = _mm256_srai_epi32(_mm256_add_epi32(T40, c_add1), SHIFT1);
    Tab = _mm256_load_si256((__m256i*)tab_dct_8_avx2[6]);
    T42 = _mm256_madd_epi16(T22, Tab);
    T43 = _mm256_madd_epi16(T23, Tab);
    T42 = _mm256_hadd_epi32(T42, T43);
    T42 = _mm256_srai_epi32(_mm256_add_epi32(T42, c_add1), SHIFT1);

    T51 = _mm256_packs_epi32(T40, T42);

    Tab = _mm256_load_si256((__m256i*)tab_dct_8_avx2[7]);
    T40 = _mm256_madd_epi16(T22, Tab);
    T41 = _mm256_madd_epi16(T23, Tab);
    T40 = _mm256_hadd_epi32(T40, T41);
    T40 = _mm256_srai_epi32(_mm256_add_epi32(T40, c_add1), SHIFT1);
    Tab = _mm256_load_si256((__m256i*)tab_dct_8_avx2[8]);
    T42 = _mm256_madd_epi16(T22, Tab);
    T43 = _mm256_madd_epi16(T23, Tab);
    T42 = _mm256_hadd_epi32(T42, T43);
    T42 = _mm256_srai_epi32(_mm256_add_epi32(T42, c_add1), SHIFT1);

    T53 = _mm256_packs_epi32(T40, T42);

    T60 = _mm256_permute4x64_epi64(T50, 0xD8);
    T61 = _mm256_permute4x64_epi64(T50, 0x72);
    T50 = _mm256_unpacklo_epi16(T60, T61);

    T60 = _mm256_permute4x64_epi64(T51, 0xD8);
    T61 = _mm256_permute4x64_epi64(T51, 0x72);
    T51 = _mm256_unpacklo_epi16(T60, T61);

    T60 = _mm256_permute4x64_epi64(T52, 0xD8);
    T61 = _mm256_permute4x64_epi64(T52, 0x72);
    T52 = _mm256_unpacklo_epi16(T60, T61);

    T60 = _mm256_permute4x64_epi64(T53, 0xD8);
    T61 = _mm256_permute4x64_epi64(T53, 0x72);
    T53 = _mm256_unpacklo_epi16(T60, T61);

    Tab = _mm256_load_si256((__m256i*)tab_dct_8_avx2[0]);
    T10 = _mm256_shuffle_epi8(T50, Tab);
    T11 = _mm256_shuffle_epi8(T51, Tab);
    T12 = _mm256_shuffle_epi8(T52, Tab);
    T13 = _mm256_shuffle_epi8(T53, Tab);

    // DCT2
    Tab = _mm256_load_si256((__m256i*)tab_dct_8_avx2[1]);
    T20 = _mm256_madd_epi16(T10, Tab);
    T21 = _mm256_madd_epi16(T11, Tab);
    T22 = _mm256_madd_epi16(T12, Tab);
    T23 = _mm256_madd_epi16(T13, Tab);

    T30 = _mm256_hadd_epi32(T20, T21);
    T31 = _mm256_hadd_epi32(T22, T23);

    T40 = _mm256_hadd_epi32(T30, T31);
    T41 = _mm256_hsub_epi32(T30, T31);

    T50 = _mm256_srai_epi32(_mm256_add_epi32(T40, c_add2), SHIFT2);
    T51 = _mm256_srai_epi32(_mm256_add_epi32(T41, c_add2), SHIFT2);

    T70 = _mm256_packs_epi32(T50, T51);
    T70 = _mm256_permute4x64_epi64(T70, 0xD8);
    T70 = _mm256_shuffle_epi32(T70, 0xD8);

#define MAKE_ODD(tab, TT0) \
    Tab = _mm256_load_si256((__m256i*)tab_dct_8_avx2[tab]); \
    T20 = _mm256_madd_epi16(T10, Tab); \
    T21 = _mm256_madd_epi16(T11, Tab); \
    T22 = _mm256_madd_epi16(T12, Tab); \
    T23 = _mm256_madd_epi16(T13, Tab); \
    T30 = _mm256_hadd_epi32(T20, T21); \
    T31 = _mm256_hadd_epi32(T22, T23); \
    T40 = _mm256_hadd_epi32(T30, T31); \
    T50 = _mm256_srai_epi32(_mm256_add_epi32(T40, c_add2), SHIFT2); \
    Tab = _mm256_load_si256((__m256i*)tab_dct_8_avx2[tab + 1]); \
    T20 = _mm256_madd_epi16(T10, Tab); \
    T21 = _mm256_madd_epi16(T11, Tab); \
    T22 = _mm256_madd_epi16(T12, Tab); \
    T23 = _mm256_madd_epi16(T13, Tab); \
    T30 = _mm256_hadd_epi32(T20, T21); \
    T31 = _mm256_hadd_epi32(T22, T23); \
    T40 = _mm256_hadd_epi32(T30, T31); \
    T51 = _mm256_srai_epi32(_mm256_add_epi32(T40, c_add2), SHIFT2); \
    TT0 = _mm256_packs_epi32(T50, T51); \
    TT0 = _mm256_permute4x64_epi64(TT0, 0xD8); \
    TT0 = _mm256_shuffle_epi32(TT0, 0xD8);


    MAKE_ODD(9, T71);
    MAKE_ODD(11, T72);
    MAKE_ODD(13, T73);

    T74 = _mm256_permute2f128_si256(T70, T71, 0x20);//0 2
    T75 = _mm256_permute2f128_si256(T70, T71, 0x31);//4 6
    T80 = _mm256_permute2f128_si256(T74, T72, 0x20);
    T81 = _mm256_permute2f128_si256(T74, T72, 0x31);
    T82 = _mm256_permute2f128_si256(T75, T73, 0x20);
    T83 = _mm256_permute2f128_si256(T75, T73, 0x31);


    __m256i mask64 = _mm256_set1_epi8(0xff);
    _mm256_maskstore_epi64((long long *)(dst + 0 * i_src), mask64, T80);
    _mm256_maskstore_epi64((long long *)(dst + 2 * i_src), mask64, T81);
    _mm256_maskstore_epi64((long long *)(dst + 4 * i_src), mask64, T82);
    _mm256_maskstore_epi64((long long *)(dst + 6 * i_src), mask64, T83);

#undef MAKE_ODD
#undef SHIFT1
#undef ADD1
#undef SHIFT2
#undef ADD2
}


/* ---------------------------------------------------------------------------
 */
void dct_c_16x16_avx2(const coeff_t * src, coeff_t * dst, int i_src)
{
    const int SHIFT1 = B16X16_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT;
    const int ADD1 = 1 << (SHIFT1 - 1);

    const int SHIFT2 = B16X16_IN_BIT + FACTO_BIT;
    const int ADD2 = 1 << (SHIFT2 - 1);


    __m256i data0, data1, data2, data3, data4, data5, data6, data7, data8, data9, dataA, dataB, dataC, dataD, dataE, dataF;
    __m256i s0, s1, s2, s3, s4, s5, s6, s7, d0, d1, d2, d3, d4, d5, d6, d7;
    __m256i ss0, ss1, ss2, ss3, sd0, sd1, sd2, sd3;
    __m256i sss0, sss1, ssd0, ssd1;
    __m256i shuffle0;
    __m256i shuffle1;
    __m256i coeff0, coeff1, coeff2, coeff3, coeff4, coeff5, coeff6, coeff7;
    __m256i coeff8, coeff9, coeffA, coeffB, coeffC, coeffD, coeffE, coeffF;
    __m256i coeff_0, coeff_1, coeff_2, coeff_3, coeff_4, coeff_5, coeff_6, coeff_7;
    __m256i add1, add2;
    __m256i temp, temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
    __m256i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rA, rB, rC, rD, rE, rF;

    int i;

    add1 = _mm256_set1_epi32(ADD1);
    shuffle0 = _mm256_load_si256((__m256i *)tab_dct_16_shuffle_avx2[0]);
    shuffle1 = _mm256_load_si256((__m256i *)tab_dct_16_shuffle_avx2[1]);
#define load_coeff(var, line_no) var = _mm256_load_si256((__m256i *) tab_dct_16_avx2[0][line_no])
    load_coeff(coeff0, 0);
    load_coeff(coeff1, 1);
    load_coeff(coeff2, 2);
    load_coeff(coeff3, 3);
    load_coeff(coeff4, 4);
    load_coeff(coeff5, 5);
    load_coeff(coeff6, 6);
    load_coeff(coeff7, 7);
    load_coeff(coeff8, 8);
    load_coeff(coeff9, 9);
    load_coeff(coeffA, 10);
    load_coeff(coeffB, 11);
    load_coeff(coeffC, 12);
    load_coeff(coeffD, 13);
    load_coeff(coeffE, 14);
    load_coeff(coeffF, 15);
#undef load_coeff

    // load data from src
    data0 = _mm256_loadu2_m128i((__m128i *)(src + 8 * i_src + 0), (__m128i *)(src + 0 * i_src + 0)); // [00 01 02 03 04 05 06 07 80 81 82 83 84 85 86 87]
    data1 = _mm256_loadu2_m128i((__m128i *)(src + 8 * i_src + 8), (__m128i *)(src + 0 * i_src + 8)); // [08 09 0A 0B 0C 0D 0E 0F 88 89 8A 8B 8C 8D 8E 8F]
    data2 = _mm256_loadu2_m128i((__m128i *)(src + 9 * i_src + 0), (__m128i *)(src + 1 * i_src + 0));
    data3 = _mm256_loadu2_m128i((__m128i *)(src + 9 * i_src + 8), (__m128i *)(src + 1 * i_src + 8));
    data4 = _mm256_loadu2_m128i((__m128i *)(src + 10 * i_src + 0), (__m128i *)(src + 2 * i_src + 0));
    data5 = _mm256_loadu2_m128i((__m128i *)(src + 10 * i_src + 8), (__m128i *)(src + 2 * i_src + 8));
    data6 = _mm256_loadu2_m128i((__m128i *)(src + 11 * i_src + 0), (__m128i *)(src + 3 * i_src + 0));
    data7 = _mm256_loadu2_m128i((__m128i *)(src + 11 * i_src + 8), (__m128i *)(src + 3 * i_src + 8));
    data8 = _mm256_loadu2_m128i((__m128i *)(src + 12 * i_src + 0), (__m128i *)(src + 4 * i_src + 0));
    data9 = _mm256_loadu2_m128i((__m128i *)(src + 12 * i_src + 8), (__m128i *)(src + 4 * i_src + 8));
    dataA = _mm256_loadu2_m128i((__m128i *)(src + 13 * i_src + 0), (__m128i *)(src + 5 * i_src + 0));
    dataB = _mm256_loadu2_m128i((__m128i *)(src + 13 * i_src + 8), (__m128i *)(src + 5 * i_src + 8));
    dataC = _mm256_loadu2_m128i((__m128i *)(src + 14 * i_src + 0), (__m128i *)(src + 6 * i_src + 0));
    dataD = _mm256_loadu2_m128i((__m128i *)(src + 14 * i_src + 8), (__m128i *)(src + 6 * i_src + 8));
    dataE = _mm256_loadu2_m128i((__m128i *)(src + 15 * i_src + 0), (__m128i *)(src + 7 * i_src + 0));
    dataF = _mm256_loadu2_m128i((__m128i *)(src + 15 * i_src + 8), (__m128i *)(src + 7 * i_src + 8));

    // reoder the data
    data0 = _mm256_shuffle_epi8(data0, shuffle0); // [00 07 03 04 01 06 02 05 80 87 83 84 81 86 82 85]
    data2 = _mm256_shuffle_epi8(data2, shuffle0);
    data4 = _mm256_shuffle_epi8(data4, shuffle0);
    data6 = _mm256_shuffle_epi8(data6, shuffle0);
    data8 = _mm256_shuffle_epi8(data8, shuffle0);
    dataA = _mm256_shuffle_epi8(dataA, shuffle0);
    dataC = _mm256_shuffle_epi8(dataC, shuffle0);
    dataE = _mm256_shuffle_epi8(dataE, shuffle0);
    data1 = _mm256_shuffle_epi8(data1, shuffle1); // [0F 08 0B 0C 0E 09 0D 0A 8F 88 8B 8C 8E 89 8D 8A]
    data3 = _mm256_shuffle_epi8(data3, shuffle1);
    data5 = _mm256_shuffle_epi8(data5, shuffle1);
    data7 = _mm256_shuffle_epi8(data7, shuffle1);
    data9 = _mm256_shuffle_epi8(data9, shuffle1);
    dataB = _mm256_shuffle_epi8(dataB, shuffle1);
    dataD = _mm256_shuffle_epi8(dataD, shuffle1);
    dataF = _mm256_shuffle_epi8(dataF, shuffle1);

    s0 = _mm256_add_epi16(data0, data1); // [s00 s07 s03 s04 s01 s06 s02 s05 s80 s87 s83 s84 s81 s86 s82 s85]
    s1 = _mm256_add_epi16(data2, data3); // [s10 s17 s13 s14 s11 s16 s12 s15 s90 s97 s93 s94 s91 s96 s92 s95]
    s2 = _mm256_add_epi16(data4, data5);
    s3 = _mm256_add_epi16(data6, data7);
    s4 = _mm256_add_epi16(data8, data9);
    s5 = _mm256_add_epi16(dataA, dataB);
    s6 = _mm256_add_epi16(dataC, dataD);
    s7 = _mm256_add_epi16(dataE, dataF);

    d0 = _mm256_sub_epi16(data0, data1); // [d00 d07 d03 d04 d01 d06 d02 d05 d80 d87 d83 d84 d81 d86 d82 d85]
    d1 = _mm256_sub_epi16(data2, data3);
    d2 = _mm256_sub_epi16(data4, data5);
    d3 = _mm256_sub_epi16(data6, data7);
    d4 = _mm256_sub_epi16(data8, data9);
    d5 = _mm256_sub_epi16(dataA, dataB);
    d6 = _mm256_sub_epi16(dataC, dataD);
    d7 = _mm256_sub_epi16(dataE, dataF);

    ss0 = _mm256_hadd_epi16(s0, s1); // [ss00 ss03 ss01 ss02 ss10 ss13 ss11 ss12 ss80 ss83 ss81 ss82 ss90 ss93 ss91 ss92]
    ss1 = _mm256_hadd_epi16(s2, s3);
    ss2 = _mm256_hadd_epi16(s4, s5);
    ss3 = _mm256_hadd_epi16(s6, s7);

    sd0 = _mm256_hsub_epi16(s0, s1); // [sd00 sd03 sd01 sd02 sd80 sd10 sd13 sd11 sd12 sd83 sd81 sd82 sd90 sd93 sd91 sd92]
    sd1 = _mm256_hsub_epi16(s2, s3);
    sd2 = _mm256_hsub_epi16(s4, s5);
    sd3 = _mm256_hsub_epi16(s6, s7);

    sss0 = _mm256_hadd_epi16(ss0, ss1); // [sss00 sss01 sss10 sss11 sss21 sss22 sss30 sss31 sss80 sss81 sss90 sss91 sssA0 sssA1 sssB0 sssB1]
    sss1 = _mm256_hadd_epi16(ss2, ss3);

    ssd0 = _mm256_hsub_epi16(ss0, ss1); // [ssd00 ssd01 ssd10 ssd11 ssd20 ssd21 ssd31 ssd32 ssd80 ssd81 ssd90 ssd9S ssdA0 ssdA1 ssdB0 ssdB1]
    ssd1 = _mm256_hsub_epi16(ss2, ss3);

    temp0 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(sss0, coeff0), add1), SHIFT1); // [00 10 20 30 80 90 A0 B0]
    temp1 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(sss1, coeff0), add1), SHIFT1); // [40 50 60 70 C0 D0 E0 F0]
    data0 = _mm256_packs_epi32(temp0, temp1); // [00 10 20 30 40 50 60 70 80 90 A0 B0 C0 D0 E0 F0]

    temp0 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(sss0, coeff8), add1), SHIFT1);
    temp1 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(sss1, coeff8), add1), SHIFT1);
    data8 = _mm256_packs_epi32(temp0, temp1); // [08 18 28 38 48 58 68 78 88 98 A8 B8 C8 D8 E8 F8]

    temp0 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(ssd0, coeff4), add1), SHIFT1);
    temp1 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(ssd1, coeff4), add1), SHIFT1);
    data4 = _mm256_packs_epi32(temp0, temp1); // [04 14 24 34 44 54 64 74 84 94 A4 B4 C4 D4 E4 F4]

    temp0 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(ssd0, coeffC), add1), SHIFT1);
    temp1 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(ssd1, coeffC), add1), SHIFT1);
    dataC = _mm256_packs_epi32(temp0, temp1); // [0C 1C 2C 3C 4C 5C 6C 7C 8C 9C AC BC CC DC EC FC]

#define CALC_4x(data, coeff) \
    temp0 = _mm256_hadd_epi32(_mm256_madd_epi16(sd0, coeff), _mm256_madd_epi16(sd1, coeff)); \
    temp1 = _mm256_hadd_epi32(_mm256_madd_epi16(sd2, coeff), _mm256_madd_epi16(sd3, coeff)); \
    temp0 = _mm256_srai_epi32(_mm256_add_epi32(temp0, add1), SHIFT1); \
    temp1 = _mm256_srai_epi32(_mm256_add_epi32(temp1, add1), SHIFT1); \
    data = _mm256_packs_epi32(temp0, temp1); // [0X 1X 2X 3X 4X 5X 6X 7X 8X 9X AX BX CX DX EX FX] -> X = 2 + 4x

    CALC_4x(data2, coeff2);
    CALC_4x(data6, coeff6);
    CALC_4x(dataA, coeffA);
    CALC_4x(dataE, coeffE);
#undef CALC_4x

#define CALC_2x(data, coeff) \
    temp0 = _mm256_hadd_epi32(_mm256_madd_epi16(d0, coeff), _mm256_madd_epi16(d1, coeff)); \
    temp1 = _mm256_hadd_epi32(_mm256_madd_epi16(d2, coeff), _mm256_madd_epi16(d3, coeff)); \
    temp2 = _mm256_hadd_epi32(_mm256_madd_epi16(d4, coeff), _mm256_madd_epi16(d5, coeff)); \
    temp3 = _mm256_hadd_epi32(_mm256_madd_epi16(d6, coeff), _mm256_madd_epi16(d7, coeff)); \
    temp0 = _mm256_hadd_epi32(temp0, temp1); \
    temp1 = _mm256_hadd_epi32(temp2, temp3); \
    temp0 = _mm256_srai_epi32(_mm256_add_epi32(temp0, add1), SHIFT1); \
    temp1 = _mm256_srai_epi32(_mm256_add_epi32(temp1, add1), SHIFT1); \
    data = _mm256_packs_epi32(temp0, temp1);  // [0X 1X 2X 3X 4X 5X 6X 7X 8X 9X AX BX CX DX EX FX] -> X = 1 + 2x

    CALC_2x(data1, coeff1);
    CALC_2x(data3, coeff3);
    CALC_2x(data5, coeff5);
    CALC_2x(data7, coeff7);
    CALC_2x(data9, coeff9);
    CALC_2x(dataB, coeffB);
    CALC_2x(dataD, coeffD);
    CALC_2x(dataF, coeffF);
#undef CALC_2x

    /*-------------------------------------------------------*/
    // dct 2
    add2 = _mm256_set1_epi32(ADD2);
    shuffle0 = _mm256_load_si256((__m256i *)tab_dct_16_shuffle_avx2[2]);
    shuffle1 = _mm256_load_si256((__m256i *)tab_dct_16_shuffle_avx2[3]);
#define load_coeff(var, line_no) var = _mm256_load_si256((__m256i *) tab_dct_16_avx2[1][line_no])
    load_coeff(coeff0, 0);
    load_coeff(coeff1, 1);
    load_coeff(coeff2, 2);
    load_coeff(coeff3, 3);
    load_coeff(coeff4, 4);
    load_coeff(coeff5, 5);
    load_coeff(coeff6, 6);
    load_coeff(coeff7, 7);
    load_coeff(coeff8, 8);
    load_coeff(coeff9, 9);
    load_coeff(coeffA, 10);
    load_coeff(coeffB, 11);
    load_coeff(coeffC, 12);
    load_coeff(coeffD, 13);
    load_coeff(coeffE, 14);
    load_coeff(coeffF, 15);
#undef load_coeff

#define load_coeff(var, line_no) var = _mm256_load_si256((__m256i *) tab_dct_16_avx2[2][line_no])
    load_coeff(coeff_0, 0);
    load_coeff(coeff_1, 1);
    load_coeff(coeff_2, 2);
    load_coeff(coeff_3, 3);
    load_coeff(coeff_4, 4);
    load_coeff(coeff_5, 5);
    load_coeff(coeff_6, 6);
    load_coeff(coeff_7, 7);
#undef load_coeff
    // now data0 ~ dataF store all of the data like [00 01 02 03 04 05...]
    for (i = 0; i < 16; i += 8) {
        r0 = _mm256_permute2x128_si256(data4, data0, 0x02); // [00 01 02 03 04 05 06 07 40 41 42 43 44 45 46 47]
        r1 = _mm256_permute2x128_si256(data4, data0, 0x13); // [08 09 0A 0B 0C 0D 0E 0F 48 49 4A 4B 4C 4D 4E 4F]
        r2 = _mm256_permute2x128_si256(data5, data1, 0x02);
        r3 = _mm256_permute2x128_si256(data5, data1, 0x13);
        r4 = _mm256_permute2x128_si256(data6, data2, 0x02);
        r5 = _mm256_permute2x128_si256(data6, data2, 0x13);
        r6 = _mm256_permute2x128_si256(data7, data3, 0x02);
        r7 = _mm256_permute2x128_si256(data7, data3, 0x13);

        r0 = _mm256_shuffle_epi8(r0, shuffle0); // [00 03 01 02 07 04 06 05 40 43 41 42 47 44 46 45]
        r1 = _mm256_shuffle_epi8(r1, shuffle1); // [0F 0C 0E 0D 08 0B 09 0A 4F 4C 4E 4D 48 4B 49 4A]
        r2 = _mm256_shuffle_epi8(r2, shuffle0);
        r3 = _mm256_shuffle_epi8(r3, shuffle1);
        r4 = _mm256_shuffle_epi8(r4, shuffle0);
        r5 = _mm256_shuffle_epi8(r5, shuffle1);
        r6 = _mm256_shuffle_epi8(r6, shuffle0);
        r7 = _mm256_shuffle_epi8(r7, shuffle1);

        temp0 = _mm256_unpacklo_epi16(r0, r1); // [00 0F 03 0C 01 0E 02 0D 40 4F 43 4C 41 4E 42 4D]
        temp1 = _mm256_unpackhi_epi16(r0, r1); // [07 08 04 0B 06 09 05 0A 47 48 44 4B 46 49 45 4A]
        temp2 = _mm256_unpacklo_epi16(r2, r3);
        temp3 = _mm256_unpackhi_epi16(r2, r3);
        temp4 = _mm256_unpacklo_epi16(r4, r5);
        temp5 = _mm256_unpackhi_epi16(r4, r5);
        temp6 = _mm256_unpacklo_epi16(r6, r7);
        temp7 = _mm256_unpackhi_epi16(r6, r7);

#define CALC_DATA(data, coeff) \
    s0 = _mm256_madd_epi16(temp0, coeff); /* [32*s00 32*s03 32*s01 32*s02 32*s40 32*s43 32*s41 32*s42] */ \
    s1 = _mm256_madd_epi16(temp1, coeff); /* [32*s07 32*s04 32*s06 32*s05 32*s47 32*s44 32*s46 32*s45] */ \
    s2 = _mm256_madd_epi16(temp2, coeff); \
    s3 = _mm256_madd_epi16(temp3, coeff); \
    s4 = _mm256_madd_epi16(temp4, coeff); \
    s5 = _mm256_madd_epi16(temp5, coeff); \
    s6 = _mm256_madd_epi16(temp6, coeff); \
    s7 = _mm256_madd_epi16(temp7, coeff); \
    \
    ss0 = _mm256_add_epi32(s0, s1); /* [32*ss02 32*ss01 32*ss03 32*ss00 32*ss42 32*ss41 32*ss43 32*ss40] */ \
    ss1 = _mm256_add_epi32(s2, s3); /* [32*ss12 32*ss11 32*ss13 32*ss10 32*ss52 32*ss51 32*ss53 32*ss50] */ \
    ss2 = _mm256_add_epi32(s4, s5); \
    ss3 = _mm256_add_epi32(s6, s7); \
    \
    sss0 = _mm256_hadd_epi32(ss0, ss1); /* [32*sss01 32*sss00 32*sss11 32*sss10 32*sss41 32*sss40 32*sss51 32*sss50] */ \
    sss1 = _mm256_hadd_epi32(ss2, ss3); /* [32*sss21 32*sss20 32*sss31 32*sss30 32*sss61 32*sss60 32*sss71 32*sss70] */ \
    data = _mm256_srai_epi32(_mm256_add_epi32(_mm256_hadd_epi32(sss0, sss1), add2), SHIFT2) // [00 01 02 03 04 05 06 07]

        CALC_DATA(r0, coeff0);
        r8 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_hsub_epi32(sss0, sss1), add2), SHIFT2); // [80 81 82 83 84 85 86 87]
        temp = _mm256_permute4x64_epi64(_mm256_packs_epi32(r0, r8), 0xD8);
        _mm256_storeu2_m128i((__m128i *)(dst + 8 * 16 + i), (__m128i *)(dst + 0 * 16 + i), temp);

        CALC_DATA(r4, coeff4);
        CALC_DATA(rC, coeffC);
        temp = _mm256_permute4x64_epi64(_mm256_packs_epi32(r4, rC), 0xD8);
        _mm256_storeu2_m128i((__m128i *)(dst + 12 * 16 + i), (__m128i *)(dst + 4 * 16 + i), temp);
#undef CALC_DATA

#define CALC_DATA(data, coeff) \
    s0 = _mm256_madd_epi16(temp0, coeff); /* [32*s00 32*s03 32*s01 32*s02 32*s40 32*s43 32*s41 32*s42] */ \
    s1 = _mm256_madd_epi16(temp1, coeff); /* [32*s07 32*s04 32*s06 32*s05 32*s47 32*s44 32*s46 32*s45] */ \
    s2 = _mm256_madd_epi16(temp2, coeff); \
    s3 = _mm256_madd_epi16(temp3, coeff); \
    s4 = _mm256_madd_epi16(temp4, coeff); \
    s5 = _mm256_madd_epi16(temp5, coeff); \
    s6 = _mm256_madd_epi16(temp6, coeff); \
    s7 = _mm256_madd_epi16(temp7, coeff); \
    \
    ss0 = _mm256_sub_epi32(s0, s1); /* [32*ss02 32*ss01 32*ss03 32*ss00 32*ss42 32*ss41 32*ss43 32*ss40] */ \
    ss1 = _mm256_sub_epi32(s2, s3); /* [32*ss12 32*ss11 32*ss13 32*ss10 32*ss52 32*ss51 32*ss53 32*ss50] */ \
    ss2 = _mm256_sub_epi32(s4, s5); \
    ss3 = _mm256_sub_epi32(s6, s7); \
    \
    sss0 = _mm256_hadd_epi32(ss0, ss1); /* [32*sss01 32*sss00 32*sss11 32*sss10 32*sss41 32*sss40 32*sss51 32*sss50] */ \
    sss1 = _mm256_hadd_epi32(ss2, ss3); /* [32*sss21 32*sss20 32*sss31 32*sss30 32*sss61 32*sss60 32*sss71 32*sss70] */ \
    data = _mm256_srai_epi32(_mm256_add_epi32(_mm256_hadd_epi32(sss0, sss1), add2), SHIFT2)

        CALC_DATA(r2, coeff2);
        CALC_DATA(r6, coeff6);
        temp = _mm256_permute4x64_epi64(_mm256_packs_epi32(r2, r6), 0xD8);
        _mm256_storeu2_m128i((__m128i *)(dst + 6 * 16 + i), (__m128i *)(dst + 2 * 16 + i), temp);

        CALC_DATA(rA, coeffA);
        CALC_DATA(rE, coeffE);
        temp = _mm256_permute4x64_epi64(_mm256_packs_epi32(rA, rE), 0xD8);
        _mm256_storeu2_m128i((__m128i *)(dst + 14 * 16 + i), (__m128i *)(dst + 10 * 16 + i), temp);
#undef CALC_DATA

#define CALC_DATA(data, c0, c1) \
    s0 = _mm256_madd_epi16(temp0, c0); \
    s1 = _mm256_madd_epi16(temp1, c1); \
    s2 = _mm256_madd_epi16(temp2, c0); \
    s3 = _mm256_madd_epi16(temp3, c1); \
    s4 = _mm256_madd_epi16(temp4, c0); \
    s5 = _mm256_madd_epi16(temp5, c1); \
    s6 = _mm256_madd_epi16(temp6, c0); \
    s7 = _mm256_madd_epi16(temp7, c1); \
    \
    ss0 = _mm256_add_epi32(s0, s1); \
    ss1 = _mm256_add_epi32(s2, s3); \
    ss2 = _mm256_add_epi32(s4, s5); \
    ss3 = _mm256_add_epi32(s6, s7); \
    \
    sss0 = _mm256_hadd_epi32(ss0, ss1); \
    sss1 = _mm256_hadd_epi32(ss2, ss3); \
    \
    data = _mm256_srai_epi32(_mm256_add_epi32(_mm256_hadd_epi32(sss0, sss1), add2), SHIFT2)

        CALC_DATA(r1, coeff1, coeff_0);
        CALC_DATA(r3, coeff3, coeff_1);
        temp = _mm256_permute4x64_epi64(_mm256_packs_epi32(r1, r3), 0xD8);
        _mm256_storeu2_m128i((__m128i *)(dst + 3 * 16 + i), (__m128i *)(dst + 1 * 16 + i), temp);

        CALC_DATA(r5, coeff5, coeff_2);
        CALC_DATA(r7, coeff7, coeff_3);
        temp = _mm256_permute4x64_epi64(_mm256_packs_epi32(r5, r7), 0xD8);
        _mm256_storeu2_m128i((__m128i *)(dst + 7 * 16 + i), (__m128i *)(dst + 5 * 16 + i), temp);

        CALC_DATA(r9, coeff9, coeff_4);
        CALC_DATA(rB, coeffB, coeff_5);
        temp = _mm256_permute4x64_epi64(_mm256_packs_epi32(r9, rB), 0xD8);
        _mm256_storeu2_m128i((__m128i *)(dst + 11 * 16 + i), (__m128i *)(dst + 9 * 16 + i), temp);

        CALC_DATA(rD, coeffD, coeff_6);
        CALC_DATA(rF, coeffF, coeff_7);
        temp = _mm256_permute4x64_epi64(_mm256_packs_epi32(rD, rF), 0xD8);
        _mm256_storeu2_m128i((__m128i *)(dst + 15 * 16 + i), (__m128i *)(dst + 13 * 16 + i), temp);

#undef CALC_DATA
        data0 = data8;
        data1 = data9;
        data2 = dataA;
        data3 = dataB;
        data4 = dataC;
        data5 = dataD;
        data6 = dataE;
        data7 = dataF;
    }
}


/* ---------------------------------------------------------------------------
 */
ALIGN32(static const int16_t tab_dct_32x32_avx2[][16]) = {

    { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 },   //order 0        //    0 / 16
    { 42, 42, -42, -42, 42, 42, -42, -42, 17, 17, -17, -17, 17, 17, -17, -17 },   //oeder 1        //    8
    { 17, 17, -17, -17, 17, 17, -17, -17, -42, -42, 42, 42, -42, -42, 42, 42 },     //order 2        //    24
    { 44, -44, 9, -9, 44, -44, 9, -9, 38, -38, 25, -25, 38, -38, 25, -25 },    //order 3         //4
    { 38, -38, -25, 25, 38, -38, -25, 25, -9, 9, -44, 44, -9, 9, -44, 44 },    //order 4         //12
    { 25, -25, 38, -38, 25, -25, 38, -38, -44, 44, 9, -9, -44, 44, 9, -9 },    //order 5         //20
    { 9, -9, -44, 44, 9, -9, -44, 44, -25, 25, 38, -38, -25, 25, 38, -38 },    //order 6         //28

#define MAKE_COE16(a0, a1, a2, a3, a4, a5, a6, a7)\
    { (a0), (a7), (a3), (a4), (a0), (a7), (a3), (a4), (a1), (a6), (a2), (a5), (a1), (a6), (a2), (a5) },

    MAKE_COE16(45, 43, 40, 35, 29, 21, 13, 4)        //  7
    MAKE_COE16(43, 29, 4, -21, -40, -45, -35, -13)   //  8
    MAKE_COE16(40, 4, -35, -43, -13, 29, 45, 21)     //  9
    MAKE_COE16(35, -21, -43, 4, 45, 13, -40, -29)    //  10
    MAKE_COE16(29, -40, -13, 45, -4, -43, 21, 35)    //  11
    MAKE_COE16(21, -45, 29, 13, -43, 35, 4, -40)     //  12
    MAKE_COE16(13, -35, 45, -40, 21, 4, -29, 43)     //  13
    MAKE_COE16(4, -13, 21, -29, 35, -40, 43, -45)    //  14
#undef MAKE_COE16

#define MAKE_COEF16(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15) \
    {(a00), (a07), (a03), (a04), (a01), (a06), (a02), (a05), (a08), (a15), (a11), (a12), (a09), (a14), (a10), (a13)},

    MAKE_COEF16(45, 45, 44, 43, 41, 39, 36, 34, 30, 27, 23, 19, 15, 11, 7, 2)  // 15
    MAKE_COEF16(45, 41, 34, 23, 11, -2, -15, -27, -36, -43, -45, -44, -39, -30, -19, -7)  // 16
    MAKE_COEF16(44, 34, 15, -7, -27, -41, -45, -39, -23, -2, 19, 36, 45, 43, 30, 11)  // 17
    MAKE_COEF16(43, 23, -7, -34, -45, -36, -11, 19, 41, 44, 27, -2, -30, -45, -39, -15)  // 18
    MAKE_COEF16(41, 11, -27, -45, -30, 7, 39, 43, 15, -23, -45, -34, 2, 36, 44, 19)  // 19
    MAKE_COEF16(39, -2, -41, -36, 7, 43, 34, -11, -44, -30, 15, 45, 27, -19, -45, -23)  // 20
    MAKE_COEF16(36, -15, -45, -11, 39, 34, -19, -45, -7, 41, 30, -23, -44, -2, 43, 27)  // 21
    MAKE_COEF16(34, -27, -39, 19, 43, -11, -45, 2, 45, 7, -44, -15, 41, 23, -36, -30)  // 22
    MAKE_COEF16(30, -36, -23, 41, 15, -44, -7, 45, -2, -45, 11, 43, -19, -39, 27, 34)  // 23
    MAKE_COEF16(27, -43, -2, 44, -23, -30, 41, 7, -45, 19, 34, -39, -11, 45, -15, -36)  // 24
    MAKE_COEF16(23, -45, 19, 27, -45, 15, 30, -44, 11, 34, -43, 7, 36, -41, 2, 39)  // 25
    MAKE_COEF16(19, -44, 36, -2, -34, 45, -23, -15, 43, -39, 7, 30, -45, 27, 11, -41)  // 26
    MAKE_COEF16(15, -39, 45, -30, 2, 27, -44, 41, -19, -11, 36, -45, 34, -7, -23, 43)  // 27
    MAKE_COEF16(11, -30, 43, -45, 36, -19, -2, 23, -39, 45, -41, 27, -7, -15, 34, -44)  // 28
    MAKE_COEF16(7, -19, 30, -39, 44, -45, 43, -36, 27, -15, 2, 11, -23, 34, -41, 45)  // 29
    MAKE_COEF16(2, -7, 11, -15, 19, -23, 27, -30, 34, -36, 39, -41, 43, -44, 45, -45)  // 30

#undef MAKE_COE16

};

ALIGN32(static const int32_t tab_dct2_32x32_avx2[][8]) = {

#define MAKE_COE8(a0, a1, a2, a3, a4, a5, a6, a7) \
    { a0, a1, a2, a3, a4, a5, a6, a7 },

    MAKE_COE8(32, 32, 32, 32, 32, 32, 32, 32)          //order 0     //      0
    MAKE_COE8(42, 17, -17, -42, -42, -17, 17, 42)          //order 1     //      8
    MAKE_COE8(32, -32, -32, 32, 32, -32, -32, 32)          //order 2     //      16
    MAKE_COE8(17, -42, 42, -17, -17, 42, -42, 17)          //order 3     //      24
    MAKE_COE8(44, 38, 25, 9, -9, -25, -38, -44)          //order 4     //      4
    MAKE_COE8(38, -9, -44, -25, 25, 44, 9, -38)          //order 5     //      12
    MAKE_COE8(25, -44, 9, 38, -38, -9, 44, -25)          //order 6     //      20
    MAKE_COE8(9, -25, 38, -44, 44, -38, 25, -9)          //order 7     //      28



    MAKE_COE8(45, 43, 40, 35, 29, 21, 13, 4)             //order 8      //      2
    MAKE_COE8(43, 29, 4, -21, -40, -45, -35, -13)             //order 9      //      6
    MAKE_COE8(40, 4, -35, -43, -13, 29, 45, 21)             //order 10     //     10
    MAKE_COE8(35, -21, -43, 4, 45, 13, -40, -29)             //order 11     //     14
    MAKE_COE8(29, -40, -13, 45, -4, -43, 21, 35)             //order 12     //     18
    MAKE_COE8(21, -45, 29, 13, -43, 35, 4, -40)             //order 13     //     22
    MAKE_COE8(13, -35, 45, -40, 21, 4, -29, 43)             //order 14     //     26
    MAKE_COE8(4, -13, 21, -29, 35, -40, 43, -45)          //order 15     //     30


#undef MAKE_COE8


#define MAKE_COE16(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15) \
    { a00, a01, a02, a03, a04, a05, a06, a07}, \
    { a15, a14, a13, a12, a11, a10, a09, a08 },

    MAKE_COE16(45, 45, 44, 43, 41, 39, 36, 34, 30, 27, 23, 19, 15, 11, 7, 2)               //order 16    // 1
    MAKE_COE16(45, 41, 34, 23, 11, -2, -15, -27, -36, -43, -45, -44, -39, -30, -19, -7)               //order 18    // 3
    MAKE_COE16(44, 34, 15, -7, -27, -41, -45, -39, -23, -2, 19, 36, 45, 43, 30, 11)               //order 20    // 5
    MAKE_COE16(43, 23, -7, -34, -45, -36, -11, 19, 41, 44, 27, -2, -30, -45, -39, -15)               //order 22    // 7
    MAKE_COE16(41, 11, -27, -45, -30, 7, 39, 43, 15, -23, -45, -34, 2, 36, 44, 19)               //order 24    // 9
    MAKE_COE16(39, -2, -41, -36, 7, 43, 34, -11, -44, -30, 15, 45, 27, -19, -45, -23)               //order 26    //11
    MAKE_COE16(36, -15, -45, -11, 39, 34, -19, -45, -7, 41, 30, -23, -44, -2, 43, 27)               //order 28    //13
    MAKE_COE16(34, -27, -39, 19, 43, -11, -45, 2, 45, 7, -44, -15, 41, 23, -36, -30)               //order 30    //15
    MAKE_COE16(30, -36, -23, 41, 15, -44, -7, 45, -2, -45, 11, 43, -19, -39, 27, 34)               //order 32    //17
    MAKE_COE16(27, -43, -2, 44, -23, -30, 41, 7, -45, 19, 34, -39, -11, 45, -15, -36)               //order 34    //19
    MAKE_COE16(23, -45, 19, 27, -45, 15, 30, -44, 11, 34, -43, 7, 36, -41, 2, 39)               //order 36    //21
    MAKE_COE16(19, -44, 36, -2, -34, 45, -23, -15, 43, -39, 7, 30, -45, 27, 11, -41)               //order 38    //23
    MAKE_COE16(15, -39, 45, -30, 2, 27, -44, 41, -19, -11, 36, -45, 34, -7, -23, 43)               //order 40    //25
    MAKE_COE16(11, -30, 43, -45, 36, -19, -2, 23, -39, 45, -41, 27, -7, -15, 34, -44)               //order 42    //27
    MAKE_COE16(7, -19, 30, -39, 44, -45, 43, -36, 27, -15, 2, 11, -23, 34, -41, 45)               //order 44    //29
    MAKE_COE16(2, -7, 11, -15, 19, -23, 27, -30, 34, -36, 39, -41, 43, -44, 45, -45)               //order 46    //31


#undef MAKE_COE16

};

/* ---------------------------------------------------------------------------
 */
void dct_c_32x32_avx2(const coeff_t *src, coeff_t *dst, int i_src)
{
    const int shift1 = B32X32_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT + (i_src & 0x01);
    const int ADD1 = 1 << (shift1 - 1);
    const int SHIFT2 = B32X32_IN_BIT + FACTO_BIT;
    const int ADD2 = 1 << (SHIFT2 - 1);

    __m256i c_add1 = _mm256_set1_epi32(ADD1);
    __m256i c_add2 = _mm256_set1_epi32(ADD2);
    //R---row  C-column
    __m256i R0C0, R0C1, R1C0, R1C1, R2C0, R2C1, R3C0, R3C1, R4C0, R4C1, R5C0, R5C1, R6C0, R6C1, R7C0, R7C1;
    __m256i R8C0, R8C1, R9C0, R9C1, R10C0, R10C1, R11C0, R11C1, R12C0, R12C1, R13C0, R13C1, R14C0, R14C1, R15C0, R15C1;
    //store anser
    __m256i A0C0, A0C1, A1C0, A1C1, A2C0, A2C1, A3C0, A3C1, A4C0, A4C1, A5C0, A5C1, A6C0, A6C1, A7C0, A7C1;
    __m256i R0R1, R2R3, R4R5, R6R7, R8R9, R10R11, R12R13, R14R15;
    __m256i COE0, COE1, COE2, COE3;
    __m256i COE_RESULT;
    __m256i im[32][2];

    __m256i  R0_ODD, R1_ODD, R2_ODD, R3_ODD, R4_ODD, R5_ODD, R6_ODD, R7_ODD;
    __m256i  R8_ODD, R9_ODD, R10_ODD, R11_ODD, R12_ODD, R13_ODD, R14_ODD, R15_ODD;
    __m256i  tab_t, tab_t1;
    coeff_t * addr;

    i_src &= 0xFE;    /* remember to remove the flag bit */

    int i;
    // DCT1
    for (i = 0; i < 32 / 16; i++) {
        R0C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  0) * i_src +  0));  //[15 14 13 12 11 10... 03 02 01 00]
        R0C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  0) * i_src + 16));  //[31 30 29 28 11 10... 19 18 17 16]
        R1C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  1) * i_src +  0));
        R1C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  1) * i_src + 16));
        R2C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  2) * i_src +  0));
        R2C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  2) * i_src + 16));
        R3C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  3) * i_src +  0));
        R3C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  3) * i_src + 16));
        R4C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  4) * i_src +  0));
        R4C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  4) * i_src + 16));
        R5C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  5) * i_src +  0));
        R5C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  5) * i_src + 16));
        R6C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  6) * i_src +  0));
        R6C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  6) * i_src + 16));
        R7C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  7) * i_src +  0));
        R7C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  7) * i_src + 16));
        R8C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  8) * i_src +  0));
        R8C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  8) * i_src + 16));
        R9C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  9) * i_src +  0));
        R9C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  9) * i_src + 16));
        R10C0 = _mm256_load_si256((__m256i*)(src + (i * 16 + 10) * i_src +  0));
        R10C1 = _mm256_load_si256((__m256i*)(src + (i * 16 + 10) * i_src + 16));
        R11C0 = _mm256_load_si256((__m256i*)(src + (i * 16 + 11) * i_src +  0));
        R11C1 = _mm256_load_si256((__m256i*)(src + (i * 16 + 11) * i_src + 16));
        R12C0 = _mm256_load_si256((__m256i*)(src + (i * 16 + 12) * i_src +  0));
        R12C1 = _mm256_load_si256((__m256i*)(src + (i * 16 + 12) * i_src + 16));
        R13C0 = _mm256_load_si256((__m256i*)(src + (i * 16 + 13) * i_src +  0));
        R13C1 = _mm256_load_si256((__m256i*)(src + (i * 16 + 13) * i_src + 16));
        R14C0 = _mm256_load_si256((__m256i*)(src + (i * 16 + 14) * i_src +  0));
        R14C1 = _mm256_load_si256((__m256i*)(src + (i * 16 + 14) * i_src + 16));
        R15C0 = _mm256_load_si256((__m256i*)(src + (i * 16 + 15) * i_src +  0));
        R15C1 = _mm256_load_si256((__m256i*)(src + (i * 16 + 15) * i_src + 16));

        //notice that different set / setr low dizhi butong
        __m256i tab_shuffle = _mm256_setr_epi16(0x0100, 0x0F0E, 0x0706, 0x0908, 0x0302, 0x0D0C, 0x0504, 0x0B0A,
                                                0x0100, 0x0F0E, 0x0706, 0x0908, 0x0302, 0x0D0C, 0x0504, 0x0B0A);

        __m256i tab_shuffle_1 = _mm256_setr_epi16(0x0100, 0x0B0A, 0x0302, 0x0908, 0x0504, 0x0F0E, 0x0706, 0x0D0C,
                                0x0100, 0x0B0A, 0x0302, 0x0908, 0x0504, 0x0F0E, 0x0706, 0x0D0C);

        __m256i tab_shuffle_2 = _mm256_setr_epi16(0x0302, 0x0100, 0x0706, 0x0504, 0x0B0A, 0x0908, 0x0F0E, 0x0D0C,
                                0x0302, 0x0100, 0x0706, 0x0504, 0x0B0A, 0x0908, 0x0F0E, 0x0D0C);

        //[13 10 14 09 12 11 15 08 05 02 06 01 04 03 07 00]
        //[29 26 30 25 28 27 31 24 21 18 22 17 20 19 23 16z
        R0C0 = _mm256_shuffle_epi8(R0C0, tab_shuffle);
        R0C1 = _mm256_shuffle_epi8(R0C1, tab_shuffle);
        R0C1 = _mm256_permute2x128_si256(R0C1, R0C1, 0x0003);//permute [21 18 22 17 20 19 23 16 / 29 26 30 25 28 27 31 24]
        R0C1 = _mm256_shuffle_epi8(R0C1, tab_shuffle_2);  // [18 21 17 22 19 20 16 23 / 26 29 25 30 27 28 24 31]
        //[13 10 14 09 12 11 15 08 / 05 02 06 01 04 03 07 00]
        //[18 21 17 22 19 20 16 23 / 26 29 25 30 27 28 24 31]
        R0_ODD = _mm256_sub_epi16(R0C0, R0C1);
        R0C0 = _mm256_add_epi16(R0C0, R0C1);//[13 10 14 09 12 11 15 08 / 05 02 06 01 04 03 07 00]
        R0C0 = _mm256_permute4x64_epi64(R0C0, 0x00D8);//[13 10 14 09 05 02 06 01 / 12 11 15 08 04 03 07 00]
        R0C0 = _mm256_shuffle_epi8(R0C0, tab_shuffle_1);//[10 05 13 02 09 06 14 01 / 11 04 12 03 08 07 15 00]

        R1C0 = _mm256_shuffle_epi8(R1C0, tab_shuffle);
        R1C1 = _mm256_shuffle_epi8(R1C1, tab_shuffle);
        R1C1 = _mm256_permute2x128_si256(R1C1, R1C1, 0x0003);
        R1C1 = _mm256_shuffle_epi8(R1C1, tab_shuffle_2);
        R1_ODD = _mm256_sub_epi16(R1C0, R1C1);

        R1C0 = _mm256_add_epi16(R1C0, R1C1);
        R1C0 = _mm256_permute4x64_epi64(R1C0, 0x00D8);
        R1C0 = _mm256_shuffle_epi8(R1C0, tab_shuffle_1);

        R2C0 = _mm256_shuffle_epi8(R2C0, tab_shuffle);
        R2C1 = _mm256_shuffle_epi8(R2C1, tab_shuffle);
        R2C1 = _mm256_permute2x128_si256(R2C1, R2C1, 0x0003);
        R2C1 = _mm256_shuffle_epi8(R2C1, tab_shuffle_2);
        R2_ODD = _mm256_sub_epi16(R2C0, R2C1);
        R2C0 = _mm256_add_epi16(R2C0, R2C1);
        R2C0 = _mm256_permute4x64_epi64(R2C0, 0x00D8);
        R2C0 = _mm256_shuffle_epi8(R2C0, tab_shuffle_1);

        R3C0 = _mm256_shuffle_epi8(R3C0, tab_shuffle);
        R3C1 = _mm256_shuffle_epi8(R3C1, tab_shuffle);
        R3C1 = _mm256_permute2x128_si256(R3C1, R3C1, 0x0003);
        R3C1 = _mm256_shuffle_epi8(R3C1, tab_shuffle_2);
        R3_ODD = _mm256_sub_epi16(R3C0, R3C1);
        R3C0 = _mm256_add_epi16(R3C0, R3C1);
        R3C0 = _mm256_permute4x64_epi64(R3C0, 0x00D8);
        R3C0 = _mm256_shuffle_epi8(R3C0, tab_shuffle_1);

        R4C0 = _mm256_shuffle_epi8(R4C0, tab_shuffle);
        R4C1 = _mm256_shuffle_epi8(R4C1, tab_shuffle);
        R4C1 = _mm256_permute2x128_si256(R4C1, R4C1, 0x0003);
        R4C1 = _mm256_shuffle_epi8(R4C1, tab_shuffle_2);
        R4_ODD = _mm256_sub_epi16(R4C0, R4C1);
        R4C0 = _mm256_add_epi16(R4C0, R4C1);
        R4C0 = _mm256_permute4x64_epi64(R4C0, 0x00D8);
        R4C0 = _mm256_shuffle_epi8(R4C0, tab_shuffle_1);


        R5C0 = _mm256_shuffle_epi8(R5C0, tab_shuffle);
        R5C1 = _mm256_shuffle_epi8(R5C1, tab_shuffle);
        R5C1 = _mm256_permute2x128_si256(R5C1, R5C1, 0x0003);
        R5C1 = _mm256_shuffle_epi8(R5C1, tab_shuffle_2);
        R5_ODD = _mm256_sub_epi16(R5C0, R5C1);
        R5C0 = _mm256_add_epi16(R5C0, R5C1);
        R5C0 = _mm256_permute4x64_epi64(R5C0, 0x00D8);
        R5C0 = _mm256_shuffle_epi8(R5C0, tab_shuffle_1);

        R6C0 = _mm256_shuffle_epi8(R6C0, tab_shuffle);
        R6C1 = _mm256_shuffle_epi8(R6C1, tab_shuffle);
        R6C1 = _mm256_permute2x128_si256(R6C1, R6C1, 0x0003);
        R6C1 = _mm256_shuffle_epi8(R6C1, tab_shuffle_2);
        R6_ODD = _mm256_sub_epi16(R6C0, R6C1);
        R6C0 = _mm256_add_epi16(R6C0, R6C1);
        R6C0 = _mm256_permute4x64_epi64(R6C0, 0x00D8);
        R6C0 = _mm256_shuffle_epi8(R6C0, tab_shuffle_1);

        R7C0 = _mm256_shuffle_epi8(R7C0, tab_shuffle);
        R7C1 = _mm256_shuffle_epi8(R7C1, tab_shuffle);
        R7C1 = _mm256_permute2x128_si256(R7C1, R7C1, 0x0003);
        R7C1 = _mm256_shuffle_epi8(R7C1, tab_shuffle_2);
        R7_ODD = _mm256_sub_epi16(R7C0, R7C1);
        R7C0 = _mm256_add_epi16(R7C0, R7C1);
        R7C0 = _mm256_permute4x64_epi64(R7C0, 0x00D8);
        R7C0 = _mm256_shuffle_epi8(R7C0, tab_shuffle_1);


        R8C0 = _mm256_shuffle_epi8(R8C0, tab_shuffle);
        R8C1 = _mm256_shuffle_epi8(R8C1, tab_shuffle);
        R8C1 = _mm256_permute2x128_si256(R8C1, R8C1, 0x0003);
        R8C1 = _mm256_shuffle_epi8(R8C1, tab_shuffle_2);
        R8_ODD = _mm256_sub_epi16(R8C0, R8C1);
        R8C0 = _mm256_add_epi16(R8C0, R8C1);
        R8C0 = _mm256_permute4x64_epi64(R8C0, 0x00D8);
        R8C0 = _mm256_shuffle_epi8(R8C0, tab_shuffle_1);

        R9C0 = _mm256_shuffle_epi8(R9C0, tab_shuffle);
        R9C1 = _mm256_shuffle_epi8(R9C1, tab_shuffle);
        R9C1 = _mm256_permute2x128_si256(R9C1, R9C1, 0x0003);
        R9C1 = _mm256_shuffle_epi8(R9C1, tab_shuffle_2);
        R9_ODD = _mm256_sub_epi16(R9C0, R9C1);
        R9C0 = _mm256_add_epi16(R9C0, R9C1);
        R9C0 = _mm256_permute4x64_epi64(R9C0, 0x00D8);
        R9C0 = _mm256_shuffle_epi8(R9C0, tab_shuffle_1);

        R10C0 = _mm256_shuffle_epi8(R10C0, tab_shuffle);
        R10C1 = _mm256_shuffle_epi8(R10C1, tab_shuffle);
        R10C1 = _mm256_permute2x128_si256(R10C1, R10C1, 0x0003);
        R10C1 = _mm256_shuffle_epi8(R10C1, tab_shuffle_2);
        R10_ODD = _mm256_sub_epi16(R10C0, R10C1);
        R10C0 = _mm256_add_epi16(R10C0, R10C1);
        R10C0 = _mm256_permute4x64_epi64(R10C0, 0x00D8);
        R10C0 = _mm256_shuffle_epi8(R10C0, tab_shuffle_1);

        R11C0 = _mm256_shuffle_epi8(R11C0, tab_shuffle);
        R11C1 = _mm256_shuffle_epi8(R11C1, tab_shuffle);
        R11C1 = _mm256_permute2x128_si256(R11C1, R11C1, 0x0003);
        R11C1 = _mm256_shuffle_epi8(R11C1, tab_shuffle_2);
        R11_ODD = _mm256_sub_epi16(R11C0, R11C1);
        R11C0 = _mm256_add_epi16(R11C0, R11C1);
        R11C0 = _mm256_permute4x64_epi64(R11C0, 0x00D8);
        R11C0 = _mm256_shuffle_epi8(R11C0, tab_shuffle_1);

        R12C0 = _mm256_shuffle_epi8(R12C0, tab_shuffle);
        R12C1 = _mm256_shuffle_epi8(R12C1, tab_shuffle);
        R12C1 = _mm256_permute2x128_si256(R12C1, R12C1, 0x0003);
        R12C1 = _mm256_shuffle_epi8(R12C1, tab_shuffle_2);
        R12_ODD = _mm256_sub_epi16(R12C0, R12C1);
        R12C0 = _mm256_add_epi16(R12C0, R12C1);
        R12C0 = _mm256_permute4x64_epi64(R12C0, 0x00D8);
        R12C0 = _mm256_shuffle_epi8(R12C0, tab_shuffle_1);

        R13C0 = _mm256_shuffle_epi8(R13C0, tab_shuffle);
        R13C1 = _mm256_shuffle_epi8(R13C1, tab_shuffle);
        R13C1 = _mm256_permute2x128_si256(R13C1, R13C1, 0x0003);
        R13C1 = _mm256_shuffle_epi8(R13C1, tab_shuffle_2);
        R13_ODD = _mm256_sub_epi16(R13C0, R13C1);
        R13C0 = _mm256_add_epi16(R13C0, R13C1);
        R13C0 = _mm256_permute4x64_epi64(R13C0, 0x00D8);
        R13C0 = _mm256_shuffle_epi8(R13C0, tab_shuffle_1);

        R14C0 = _mm256_shuffle_epi8(R14C0, tab_shuffle);
        R14C1 = _mm256_shuffle_epi8(R14C1, tab_shuffle);
        R14C1 = _mm256_permute2x128_si256(R14C1, R14C1, 0x0003);
        R14C1 = _mm256_shuffle_epi8(R14C1, tab_shuffle_2);
        R14_ODD = _mm256_sub_epi16(R14C0, R14C1);
        R14C0 = _mm256_add_epi16(R14C0, R14C1);
        R14C0 = _mm256_permute4x64_epi64(R14C0, 0x00D8);
        R14C0 = _mm256_shuffle_epi8(R14C0, tab_shuffle_1);

        R15C0 = _mm256_shuffle_epi8(R15C0, tab_shuffle);
        R15C1 = _mm256_shuffle_epi8(R15C1, tab_shuffle);
        R15C1 = _mm256_permute2x128_si256(R15C1, R15C1, 0x0003);
        R15C1 = _mm256_shuffle_epi8(R15C1, tab_shuffle_2);
        R15_ODD = _mm256_sub_epi16(R15C0, R15C1);
        R15C0 = _mm256_add_epi16(R15C0, R15C1);
        R15C0 = _mm256_permute4x64_epi64(R15C0, 0x00D8);
        R15C0 = _mm256_shuffle_epi8(R15C0, tab_shuffle_1);


        R0R1 = _mm256_hadd_epi16(R0C0, R1C0);//[105 102 106 101 005 002 006 001 / 104 103 107 100 004 003 007 000]
        R2R3 = _mm256_hadd_epi16(R2C0, R3C0);
        R4R5 = _mm256_hadd_epi16(R4C0, R5C0);
        R6R7 = _mm256_hadd_epi16(R6C0, R7C0);
        R8R9 = _mm256_hadd_epi16(R8C0, R9C0);//[905 902 906 901 805 802 806 801 / 904 903 907 900 804 803 807 800]
        R10R11 = _mm256_hadd_epi16(R10C0, R11C0);
        R12R13 = _mm256_hadd_epi16(R12C0, R13C0);
        R14R15 = _mm256_hadd_epi16(R14C0, R15C0);

        // mul the coefficient
        //0th row ,1th row   [105+102 106+101 005+002 006+001 / 104+103 107+100 004+003 007+000]
        tab_t = _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[0]);
        A0C0 = _mm256_madd_epi16(R0R1,   tab_t);
        A1C0 = _mm256_madd_epi16(R2R3,   tab_t);// 2   3
        A2C0 = _mm256_madd_epi16(R4R5,   tab_t);// 4   5
        A3C0 = _mm256_madd_epi16(R6R7,   tab_t);// 6   7
        A4C0 = _mm256_madd_epi16(R8R9,   tab_t);// 8   9
        A5C0 = _mm256_madd_epi16(R10R11, tab_t);//10  11
        A6C0 = _mm256_madd_epi16(R12R13, tab_t);//12  13
        A7C0 = _mm256_madd_epi16(R14R15, tab_t);//14  15

        A0C0 = _mm256_hadd_epi32(A0C0, A1C0); //[3B 2B 1B 0B(05+02+06+01) / 3A 2A 1A 0A(04+03+07+00)]
        A1C0 = _mm256_permute2f128_si256(A0C0, A0C0, 0x0001); //[3A 2A 1A 0A / 3B 2B 1B 0B]

        A2C0 = _mm256_hadd_epi32(A2C0, A3C0); //[7B 6B 5B 4B / 7A 6A 5A 4A]
        A3C0 = _mm256_permute2f128_si256(A2C0, A2C0, 0x0001);//[7A 6A 5A 4A / 7B 6B 5B 4B]

        A4C0 = _mm256_hadd_epi32(A4C0, A5C0); //[11B 10B 9B 8B / 11A 10A 9A 8A]
        A5C0 = _mm256_permute2f128_si256(A4C0, A4C0, 0x0001);//[11A 10A 9A 8A / 11B 10B 9B 8B]

        A6C0 = _mm256_hadd_epi32(A6C0, A7C0); //[15B 14B 13B 12B / 15A 14A 13A 12A]
        A7C0 = _mm256_permute2f128_si256(A6C0, A6C0, 0x0001);//[15A 14A 13A 12A / 15B 14B 13B 12B]


        COE0 = _mm256_add_epi32(A0C0, A1C0); //the same line`s data add to low 128 bit
        COE1 = _mm256_add_epi32(A2C0, A3C0);
        COE2 = _mm256_add_epi32(A4C0, A5C0);
        COE3 = _mm256_add_epi32(A6C0, A7C0);
        //low 128 bit is 0 1 2 3 rows data ,the high 128 bit is 8 9 10 11 rows data
        COE0 = _mm256_blend_epi32(COE0, COE2, 0x00F0);//[11 10 9 8 3 2 1 0]
        COE1 = _mm256_blend_epi32(COE1, COE3, 0x00F0);//[15 14 13 12 7 6 5 4]

        COE0 = _mm256_srai_epi32(_mm256_add_epi32(COE0, c_add1), shift1);
        COE1 = _mm256_srai_epi32(_mm256_add_epi32(COE1, c_add1), shift1);

        COE_RESULT = _mm256_packs_epi32(COE0, COE1);//[15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0]

        im[0][i] = COE_RESULT;

        COE0 = _mm256_sub_epi32(A0C0, A1C0);
        COE1 = _mm256_sub_epi32(A2C0, A3C0);
        COE2 = _mm256_sub_epi32(A4C0, A5C0);
        COE3 = _mm256_sub_epi32(A6C0, A7C0);

        COE0 = _mm256_permute2f128_si256(COE0, COE2, 0x0020);
        COE1 = _mm256_permute2f128_si256(COE1, COE3, 0x0020);

        COE0 = _mm256_srai_epi32(_mm256_add_epi32(COE0, c_add1), shift1);
        COE1 = _mm256_srai_epi32(_mm256_add_epi32(COE1, c_add1), shift1);

        COE_RESULT = _mm256_packs_epi32(COE0, COE1);//[15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0]

        im[16][i] = COE_RESULT;

#define MAKE_ODD(tab,dstPos) \
    tab_t = _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab]);  \
    A0C0 = _mm256_madd_epi16(R0R1,   tab_t); \
    A1C0 = _mm256_madd_epi16(R2R3,   tab_t); \
    A2C0 = _mm256_madd_epi16(R4R5,   tab_t); \
    A3C0 = _mm256_madd_epi16(R6R7,   tab_t); \
    A4C0 = _mm256_madd_epi16(R8R9,   tab_t); \
    A5C0 = _mm256_madd_epi16(R10R11, tab_t); \
    A6C0 = _mm256_madd_epi16(R12R13, tab_t); \
    A7C0 = _mm256_madd_epi16(R14R15, tab_t); \
    \
    A0C0 = _mm256_hadd_epi32(A0C0, A1C0); \
    A1C0 = _mm256_permute2f128_si256(A0C0, A0C0, 0x0001); \
    \
    A2C0 = _mm256_hadd_epi32(A2C0, A3C0); \
    A3C0 = _mm256_permute2f128_si256(A2C0, A2C0, 0x0001); \
    \
    A4C0 = _mm256_hadd_epi32(A4C0, A5C0); \
    A5C0 = _mm256_permute2f128_si256(A4C0, A4C0, 0x0001); \
    \
    A6C0 = _mm256_hadd_epi32(A6C0, A7C0); \
    A7C0 = _mm256_permute2f128_si256(A6C0, A6C0, 0x0001); \
    \
    COE0 = _mm256_add_epi32(A0C0, A1C0); \
    COE1 = _mm256_add_epi32(A2C0, A3C0); \
    COE2 = _mm256_add_epi32(A4C0, A5C0); \
    COE3 = _mm256_add_epi32(A6C0, A7C0); \
    \
    COE0 = _mm256_blend_epi32(COE0, COE2, 0x00F0); \
    COE1 = _mm256_blend_epi32(COE1, COE3, 0x00F0); \
    \
    COE0 = _mm256_srai_epi32(_mm256_add_epi32(COE0, c_add1), shift1); \
    COE1 = _mm256_srai_epi32(_mm256_add_epi32(COE1, c_add1), shift1); \
    \
    COE_RESULT = _mm256_packs_epi32(COE0, COE1); \
    im[dstPos][i] = COE_RESULT;

        MAKE_ODD(1, 8);
        MAKE_ODD(2, 24);

#undef MAKE_ODD


#define MAKE_ODD(tab,dstPos) \
    tab_t = _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab]);  \
    A0C0 = _mm256_madd_epi16(R0R1,   tab_t); \
    A1C0 = _mm256_madd_epi16(R2R3,   tab_t); \
    A2C0 = _mm256_madd_epi16(R4R5,   tab_t); \
    A3C0 = _mm256_madd_epi16(R6R7,   tab_t); \
    A4C0 = _mm256_madd_epi16(R8R9,   tab_t); \
    A5C0 = _mm256_madd_epi16(R10R11, tab_t); \
    A6C0 = _mm256_madd_epi16(R12R13, tab_t); \
    A7C0 = _mm256_madd_epi16(R14R15, tab_t); \
    \
    A0C0 = _mm256_hadd_epi32(A0C0, A1C0); \
    A1C0 = _mm256_permute2f128_si256(A0C0, A0C0, 0x0001); \
    \
    A2C0 = _mm256_hadd_epi32(A2C0, A3C0); \
    A3C0 = _mm256_permute2f128_si256(A2C0, A2C0, 0x0001); \
    \
    A4C0 = _mm256_hadd_epi32(A4C0, A5C0); \
    A5C0 = _mm256_permute2f128_si256(A4C0, A4C0, 0x0001); \
    \
    A6C0 = _mm256_hadd_epi32(A6C0, A7C0); \
    A7C0 = _mm256_permute2f128_si256(A6C0, A6C0, 0x0001); \
    \
    COE0 = _mm256_add_epi32(A0C0, A1C0); \
    COE1 = _mm256_add_epi32(A2C0, A3C0); \
    COE2 = _mm256_add_epi32(A4C0, A5C0); \
    COE3 = _mm256_add_epi32(A6C0, A7C0); \
    \
    COE0 = _mm256_blend_epi32(COE0, COE2, 0x00F0); \
    COE1 = _mm256_blend_epi32(COE1, COE3, 0x00F0); \
    \
    COE0 = _mm256_srai_epi32(_mm256_add_epi32(COE0, c_add1), shift1); \
    COE1 = _mm256_srai_epi32(_mm256_add_epi32(COE1, c_add1), shift1); \
    \
    COE_RESULT = _mm256_packs_epi32(COE0, COE1); \
    im[dstPos][i] = COE_RESULT;

        MAKE_ODD(3, 4);
        MAKE_ODD(4, 12);
        MAKE_ODD(5, 20);
        MAKE_ODD(6, 28);

        R0R1 = _mm256_hsub_epi16(R0C0, R1C0);//[105 102 106 101 005 002 006 001 / 104 103 107 100 004 003 007 000]
        R2R3 = _mm256_hsub_epi16(R2C0, R3C0);
        R4R5 = _mm256_hsub_epi16(R4C0, R5C0);
        R6R7 = _mm256_hsub_epi16(R6C0, R7C0);
        R8R9 = _mm256_hsub_epi16(R8C0, R9C0);//[905 902 906 901 805 802 806 801 / 904 903 907 900 804 803 807 800]
        R10R11 = _mm256_hsub_epi16(R10C0, R11C0);
        R12R13 = _mm256_hsub_epi16(R12C0, R13C0);
        R14R15 = _mm256_hsub_epi16(R14C0, R15C0);

        MAKE_ODD(7, 2);
        MAKE_ODD(8, 6);
        MAKE_ODD(9, 10);
        MAKE_ODD(10, 14);
        MAKE_ODD(11, 18);
        MAKE_ODD(12, 22);
        MAKE_ODD(13, 26);
        MAKE_ODD(14, 30);

#undef MAKE_ODD


#define MAKE_ODD(tab,dstPos) \
    tab_t = _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab]);  \
    A0C0 = _mm256_madd_epi16(R0_ODD,  tab_t); \
    A0C1 = _mm256_madd_epi16(R1_ODD,  tab_t); \
    A1C0 = _mm256_madd_epi16(R2_ODD,  tab_t); \
    A1C1 = _mm256_madd_epi16(R3_ODD,  tab_t); \
    A2C0 = _mm256_madd_epi16(R4_ODD,  tab_t); \
    A2C1 = _mm256_madd_epi16(R5_ODD,  tab_t); \
    A3C0 = _mm256_madd_epi16(R6_ODD,  tab_t); \
    A3C1 = _mm256_madd_epi16(R7_ODD,  tab_t); \
    A4C0 = _mm256_madd_epi16(R8_ODD,  tab_t); \
    A4C1 = _mm256_madd_epi16(R9_ODD,  tab_t); \
    A5C0 = _mm256_madd_epi16(R10_ODD, tab_t); \
    A5C1 = _mm256_madd_epi16(R11_ODD, tab_t); \
    A6C0 = _mm256_madd_epi16(R12_ODD, tab_t); \
    A6C1 = _mm256_madd_epi16(R13_ODD, tab_t); \
    A7C0 = _mm256_madd_epi16(R14_ODD, tab_t); \
    A7C1 = _mm256_madd_epi16(R15_ODD, tab_t); \
    \
    A0C0 = _mm256_hadd_epi32(A0C0, A0C1); \
    A1C0 = _mm256_hadd_epi32(A1C0, A1C1); \
    A2C0 = _mm256_hadd_epi32(A2C0, A2C1); \
    A3C0 = _mm256_hadd_epi32(A3C0, A3C1); \
    A4C0 = _mm256_hadd_epi32(A4C0, A4C1); \
    A5C0 = _mm256_hadd_epi32(A5C0, A5C1); \
    A6C0 = _mm256_hadd_epi32(A6C0, A6C1); \
    A7C0 = _mm256_hadd_epi32(A7C0, A7C1); \
    \
    A0C0 = _mm256_hadd_epi32(A0C0, A1C0); \
    A1C0 = _mm256_hadd_epi32(A2C0, A3C0); \
    A2C0 = _mm256_hadd_epi32(A4C0, A5C0); \
    A3C0 = _mm256_hadd_epi32(A6C0, A7C0); \
    \
    A0C1 = _mm256_permute2f128_si256(A0C0, A2C0, 0x0020); \
    A1C1 = _mm256_permute2f128_si256(A0C0, A2C0, 0x0031); \
    A2C1 = _mm256_permute2f128_si256(A1C0, A3C0, 0x0020); \
    A3C1 = _mm256_permute2f128_si256(A1C0, A3C0, 0x0031); \
    \
    COE0 = _mm256_add_epi32(A0C1, A1C1); \
    COE1 = _mm256_add_epi32(A2C1, A3C1); \
    COE0 = _mm256_srai_epi32(_mm256_add_epi32(COE0, c_add1), shift1); \
    COE1 = _mm256_srai_epi32(_mm256_add_epi32(COE1, c_add1), shift1); \
    \
    COE_RESULT = _mm256_packs_epi32(COE0, COE1); \
    im[dstPos][i] = COE_RESULT;

        MAKE_ODD(15, 1);
        MAKE_ODD(16, 3);
        MAKE_ODD(17, 5);
        MAKE_ODD(18, 7);
        MAKE_ODD(19, 9);
        MAKE_ODD(20, 11);
        MAKE_ODD(21, 13);
        MAKE_ODD(22, 15);
        MAKE_ODD(23, 17);
        MAKE_ODD(24, 19);
        MAKE_ODD(25, 21);
        MAKE_ODD(26, 23);
        MAKE_ODD(27, 25);
        MAKE_ODD(28, 27);
        MAKE_ODD(29, 29);
        MAKE_ODD(30, 31);

#undef MAKE_ODD

    }

    __m128i mask = _mm_set1_epi16(0xffff);
    //DCT2
    for (i = 0; i < 32 / 8; i++) {
        R0C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 0) * i_src) +  0), mask));
        R0C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 0) * i_src) +  8), mask));
        R1C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 0) * i_src) + 16), mask));
        R1C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 0) * i_src) + 24), mask));

        R2C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 1) * i_src) +  0), mask));
        R2C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 1) * i_src) +  8), mask));
        R3C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 1) * i_src) + 16), mask));
        R3C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 1) * i_src) + 24), mask));

        R4C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 2) * i_src) +  0), mask));
        R4C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 2) * i_src) +  8), mask));
        R5C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 2) * i_src) + 16), mask));
        R5C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 2) * i_src) + 24), mask));

        R6C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 3) * i_src) +  0), mask));
        R6C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 3) * i_src) +  8), mask));
        R7C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 3) * i_src) + 16), mask));
        R7C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 3) * i_src) + 24), mask));

        R8C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 4) * i_src) +  0), mask));
        R8C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 4) * i_src) +  8), mask));
        R9C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 4) * i_src) + 16), mask));
        R9C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 4) * i_src) + 24), mask));

        R10C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 5) * i_src) +  0), mask));
        R10C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 5) * i_src) +  8), mask));
        R11C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 5) * i_src) + 16), mask));
        R11C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 5) * i_src) + 24), mask));

        R12C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 6) * i_src) +  0), mask));
        R12C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 6) * i_src) +  8), mask));
        R13C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 6) * i_src) + 16), mask));
        R13C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 6) * i_src) + 24), mask));

        R14C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 7) * i_src) +  0), mask));
        R14C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 7) * i_src) +  8), mask));
        R15C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 7) * i_src) + 16), mask));
        R15C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 7) * i_src) + 24), mask));


        // inverse _m256i per 32 bit
        __m256i tab_inv = _mm256_setr_epi32(0x0007, 0x0006, 0x0005, 0x0004, 0x0003, 0x0002, 0x0001, 0x0000);


        R0C1 = _mm256_permutevar8x32_epi32(R0C1, tab_inv); //[8 9 10 11 / 12 13 14 15]
        R1C1 = _mm256_permutevar8x32_epi32(R1C1, tab_inv); //[24 25 26 27 / 28 29 30 31]
        R0_ODD = _mm256_sub_epi32(R0C0, R1C1); //[7-24 6-25  5-26  4-27 /  3-28  2-29  1-30  0-31]
        R15_ODD = _mm256_sub_epi32(R0C1, R1C0); //[8-23 9-22 10-21 11-20 / 12-19 13-18 14-17 15-16]
        R0C0 = _mm256_add_epi32(R0C0, R1C1);    //[7 6  5  4 /  3  2  1  0]
        R0C1 = _mm256_add_epi32(R0C1, R1C0);    //[8 9 10 11 / 12 13 14 15]
        A0C0 = _mm256_add_epi32(R0C0, R0C1);    //[7 6 5 4 / 3 2 1 0]



        R2C1 = _mm256_permutevar8x32_epi32(R2C1, tab_inv);
        R3C1 = _mm256_permutevar8x32_epi32(R3C1, tab_inv);
        R1_ODD = _mm256_sub_epi32(R2C0, R3C1);
        R14_ODD = _mm256_sub_epi32(R2C1, R3C0);
        R1C0 = _mm256_add_epi32(R2C0, R3C1);
        R1C1 = _mm256_add_epi32(R2C1, R3C0);
        A1C0 = _mm256_add_epi32(R1C0, R1C1);

        R4C1 = _mm256_permutevar8x32_epi32(R4C1, tab_inv);
        R5C1 = _mm256_permutevar8x32_epi32(R5C1, tab_inv);
        R2_ODD = _mm256_sub_epi32(R4C0, R5C1);
        R13_ODD = _mm256_sub_epi32(R4C1, R5C0);
        R2C0 = _mm256_add_epi32(R4C0, R5C1);
        R2C1 = _mm256_add_epi32(R4C1, R5C0);
        A2C0 = _mm256_add_epi32(R2C0, R2C1);

        R6C1 = _mm256_permutevar8x32_epi32(R6C1, tab_inv);
        R7C1 = _mm256_permutevar8x32_epi32(R7C1, tab_inv);
        R3_ODD = _mm256_sub_epi32(R6C0, R7C1);
        R12_ODD = _mm256_sub_epi32(R6C1, R7C0);
        R3C0 = _mm256_add_epi32(R6C0, R7C1);
        R3C1 = _mm256_add_epi32(R6C1, R7C0);
        A3C0 = _mm256_add_epi32(R3C0, R3C1);

        R8C1 = _mm256_permutevar8x32_epi32(R8C1, tab_inv);
        R9C1 = _mm256_permutevar8x32_epi32(R9C1, tab_inv);
        R4_ODD = _mm256_sub_epi32(R8C0, R9C1);
        R11_ODD = _mm256_sub_epi32(R8C1, R9C0);
        R4C0 = _mm256_add_epi32(R8C0, R9C1);
        R4C1 = _mm256_add_epi32(R8C1, R9C0);
        A4C0 = _mm256_add_epi32(R4C0, R4C1);


        R10C1 = _mm256_permutevar8x32_epi32(R10C1, tab_inv);
        R11C1 = _mm256_permutevar8x32_epi32(R11C1, tab_inv);
        R5_ODD = _mm256_sub_epi32(R10C0, R11C1);
        R10_ODD = _mm256_sub_epi32(R10C1, R11C0);
        R5C0 = _mm256_add_epi32(R10C0, R11C1);
        R5C1 = _mm256_add_epi32(R10C1, R11C0);
        A5C0 = _mm256_add_epi32(R5C0, R5C1);


        R12C1 = _mm256_permutevar8x32_epi32(R12C1, tab_inv);
        R13C1 = _mm256_permutevar8x32_epi32(R13C1, tab_inv);
        R6_ODD = _mm256_sub_epi32(R12C0, R13C1);
        R9_ODD = _mm256_sub_epi32(R12C1, R13C0);
        R6C0 = _mm256_add_epi32(R12C0, R13C1);
        R6C1 = _mm256_add_epi32(R12C1, R13C0);
        A6C0 = _mm256_add_epi32(R6C0, R6C1);



        R14C1 = _mm256_permutevar8x32_epi32(R14C1, tab_inv);
        R15C1 = _mm256_permutevar8x32_epi32(R15C1, tab_inv);
        R7_ODD = _mm256_sub_epi32(R14C0, R15C1);//[7-24 6-25  5-26  4-27 /  3-28  2-29  1-30  0-31]
        R8_ODD = _mm256_sub_epi32(R14C1, R15C0);  //[8-23 9-22 10-21 11-20 / 12-19 13-18 14-17 15-16]
        R7C0 = _mm256_add_epi32(R14C0, R15C1);    //[7 6  5  4 /  3  2  1  0]
        R7C1 = _mm256_add_epi32(R14C1, R15C0);    //[8 9 10 11 / 12 13 14 15]
        A7C0 = _mm256_add_epi32(R7C0, R7C1);      //[7 6 5 4 / 3 2 1 0]


        __m256i result_mask = _mm256_setr_epi32(0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0, 0, 0, 0);
#define MAKE_ODD(tab,dstPos) \
    tab_t = _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab]);  \
    A0C1 = _mm256_mullo_epi32(A0C0, tab_t); \
    A1C1 = _mm256_mullo_epi32(A1C0, tab_t); \
    A2C1 = _mm256_mullo_epi32(A2C0, tab_t); \
    A3C1 = _mm256_mullo_epi32(A3C0, tab_t); \
    A4C1 = _mm256_mullo_epi32(A4C0, tab_t); \
    A5C1 = _mm256_mullo_epi32(A5C0, tab_t); \
    A6C1 = _mm256_mullo_epi32(A6C0, tab_t); \
    A7C1 = _mm256_mullo_epi32(A7C0, tab_t); \
    \
    COE0 = _mm256_hadd_epi32(A0C1, A1C1); /* [107+106 105+104 007+006 005+004 / 103+102 101+100 003+002 001+000] */\
    COE1 = _mm256_hadd_epi32(A2C1, A3C1); \
    COE2 = _mm256_hadd_epi32(A4C1, A5C1); \
    COE3 = _mm256_hadd_epi32(A6C1, A7C1); \
    \
    COE0 = _mm256_hadd_epi32(COE0, COE1); /* [3A 2A 1A 0A / 3B 2B 1B 0B] */\
    COE1 = _mm256_hadd_epi32(COE2, COE3); /* [7A 6A 5A 4A / 7B 6B 5B 4B] */\
    \
    COE2 = _mm256_permute2f128_si256(COE0, COE1, 0x0020); /*[7B 6B 5B 4B / 3B 2B 1B 0B]*/\
    COE3 = _mm256_permute2f128_si256(COE0, COE1, 0x0031); /*[7A 6A 5A 4A / 3A 2A 1A 0A]*/\
    \
    COE_RESULT = _mm256_add_epi32(COE2, COE3); /* [7 6 5 4 / 3 2 1 0] */\
    COE_RESULT = _mm256_srai_epi32(_mm256_add_epi32(COE_RESULT, c_add2), SHIFT2); \
    COE0 = _mm256_permute2f128_si256(COE_RESULT, COE_RESULT, 0x0001);/* [3 2 1 0 / 7 6 5 4] */ \
    COE_RESULT = _mm256_packs_epi32(COE_RESULT, COE0); /*[3 2 1 0 7 6 5 4 / 7 6 5 4 3 2 1 0]*/\
    addr = (dst + (dstPos * 32) + (i * 8)); \
    \
    _mm256_maskstore_epi32((int*)addr, result_mask, COE_RESULT);
        //_mm256_storeu2_m128i(addr, &COE3, COE_RESULT);

        /*_mm256_storeu_si256(addr, COE_RESULT);*/
        MAKE_ODD(0, 0);
        MAKE_ODD(1, 8);
        MAKE_ODD(2, 16);
        MAKE_ODD(3, 24);
        MAKE_ODD(4, 4);
        MAKE_ODD(5, 12);
        MAKE_ODD(6, 20);
        MAKE_ODD(7, 28);


        A0C0 = _mm256_sub_epi32(R0C0, R0C1);
        A1C0 = _mm256_sub_epi32(R1C0, R1C1);
        A2C0 = _mm256_sub_epi32(R2C0, R2C1);
        A3C0 = _mm256_sub_epi32(R3C0, R3C1);
        A4C0 = _mm256_sub_epi32(R4C0, R4C1);
        A5C0 = _mm256_sub_epi32(R5C0, R5C1);
        A6C0 = _mm256_sub_epi32(R6C0, R6C1);
        A7C0 = _mm256_sub_epi32(R7C0, R7C1);

        MAKE_ODD(8, 2);
        MAKE_ODD(9, 6);
        MAKE_ODD(10, 10);
        MAKE_ODD(11, 14);
        MAKE_ODD(12, 18);
        MAKE_ODD(13, 22);
        MAKE_ODD(14, 26);
        MAKE_ODD(15, 30);


#undef MAKE_ODD


#define MAKE_ODD(tab,dstPos) \
    tab_t  = _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab]);  \
    tab_t1 = _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab + 1]);  \
    A0C1 = _mm256_add_epi32(_mm256_mullo_epi32(R0_ODD, tab_t), _mm256_mullo_epi32(R15_ODD, tab_t1)); \
    A1C1 = _mm256_add_epi32(_mm256_mullo_epi32(R1_ODD, tab_t), _mm256_mullo_epi32(R14_ODD, tab_t1)); \
    A2C1 = _mm256_add_epi32(_mm256_mullo_epi32(R2_ODD, tab_t), _mm256_mullo_epi32(R13_ODD, tab_t1)); \
    A3C1 = _mm256_add_epi32(_mm256_mullo_epi32(R3_ODD, tab_t), _mm256_mullo_epi32(R12_ODD, tab_t1)); \
    A4C1 = _mm256_add_epi32(_mm256_mullo_epi32(R4_ODD, tab_t), _mm256_mullo_epi32(R11_ODD, tab_t1)); \
    A5C1 = _mm256_add_epi32(_mm256_mullo_epi32(R5_ODD, tab_t), _mm256_mullo_epi32(R10_ODD, tab_t1)); \
    A6C1 = _mm256_add_epi32(_mm256_mullo_epi32(R6_ODD, tab_t), _mm256_mullo_epi32(R9_ODD,  tab_t1)); \
    A7C1 = _mm256_add_epi32(_mm256_mullo_epi32(R7_ODD, tab_t), _mm256_mullo_epi32(R8_ODD,  tab_t1)); \
    \
    COE0 = _mm256_hadd_epi32(A0C1, A1C1); \
    COE1 = _mm256_hadd_epi32(A2C1, A3C1); \
    COE2 = _mm256_hadd_epi32(A4C1, A5C1); \
    COE3 = _mm256_hadd_epi32(A6C1, A7C1); \
    \
    COE0 = _mm256_hadd_epi32(COE0, COE1); \
    COE1 = _mm256_hadd_epi32(COE2, COE3); \
    \
    COE2 = _mm256_permute2f128_si256(COE0, COE1, 0x0020); \
    COE3 = _mm256_permute2f128_si256(COE0, COE1, 0x0031); \
    \
    COE_RESULT = _mm256_add_epi32(COE2, COE3); \
    COE_RESULT = _mm256_srai_epi32(_mm256_add_epi32(COE_RESULT, c_add2), SHIFT2); \
    COE0 = _mm256_permute2f128_si256(COE_RESULT, COE_RESULT, 0x0001); \
    COE_RESULT = _mm256_packs_epi32(COE_RESULT, COE0); \
    addr = (dst + (dstPos * 32) + (i * 8)); \
    \
    _mm256_maskstore_epi32((int*)addr, result_mask, COE_RESULT);

        //_mm256_storeu2_m128i(addr, &COE3, COE_RESULT);

        //_mm256_storeu_si256(addr, COE_RESULT);

        MAKE_ODD(16, 1);
        MAKE_ODD(18, 3);
        MAKE_ODD(20, 5);
        MAKE_ODD(22, 7);
        MAKE_ODD(24, 9);
        MAKE_ODD(26, 11);
        MAKE_ODD(28, 13);
        MAKE_ODD(30, 15);
        MAKE_ODD(32, 17);
        MAKE_ODD(34, 19);
        MAKE_ODD(36, 21);
        MAKE_ODD(38, 23);
        MAKE_ODD(40, 25);
        MAKE_ODD(42, 27);
        MAKE_ODD(44, 29);
        MAKE_ODD(46, 31);

#undef MAKE_ODD

    }

}


/* ---------------------------------------------------------------------------
 */
void dct_c_32x32_half_avx2(const coeff_t *src, coeff_t *dst, int i_src)
{
    const int shift1 = B32X32_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT + (i_src & 0x01);
    const int ADD1 = 1 << (shift1 - 1);
    const int SHIFT2 = B32X32_IN_BIT + FACTO_BIT;
    const int ADD2 = 1 << (SHIFT2 - 1);

    __m256i c_add1 = _mm256_set1_epi32(ADD1);
    __m256i c_add2 = _mm256_set1_epi32(ADD2);
    //R---row  C-column
    __m256i R0C0, R0C1, R1C0, R1C1, R2C0, R2C1, R3C0, R3C1, R4C0, R4C1, R5C0, R5C1, R6C0, R6C1, R7C0, R7C1;
    __m256i R8C0, R8C1, R9C0, R9C1, R10C0, R10C1, R11C0, R11C1, R12C0, R12C1, R13C0, R13C1, R14C0, R14C1, R15C0, R15C1;
    //store anser
    __m256i A0C0, A0C1, A1C0, A1C1, A2C0, A2C1, A3C0, A3C1, A4C0, A4C1, A5C0, A5C1, A6C0, A6C1, A7C0, A7C1;
    __m256i R0R1, R2R3, R4R5, R6R7, R8R9, R10R11, R12R13, R14R15;
    __m256i COE0, COE1, COE2, COE3;
    __m256i COE_RESULT;
    __m256i im[16][2];

    __m256i  R0_ODD, R1_ODD, R2_ODD, R3_ODD, R4_ODD, R5_ODD, R6_ODD, R7_ODD;
    __m256i  R8_ODD, R9_ODD, R10_ODD, R11_ODD, R12_ODD, R13_ODD, R14_ODD, R15_ODD;
    __m256i  tab_t, tab_t1;
    coeff_t * addr;

    i_src &= 0xFE;    /* remember to remove the flag bit */

    int i;
    // DCT1
    for (i = 0; i < 32 / 16; i++) {
        R0C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  0) * i_src +  0));  //[15 14 13 12 11 10... 03 02 01 00]
        R0C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  0) * i_src + 16));  //[31 30 29 28 11 10... 19 18 17 16]
        R1C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  1) * i_src +  0));
        R1C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  1) * i_src + 16));
        R2C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  2) * i_src +  0));
        R2C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  2) * i_src + 16));
        R3C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  3) * i_src +  0));
        R3C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  3) * i_src + 16));
        R4C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  4) * i_src +  0));
        R4C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  4) * i_src + 16));
        R5C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  5) * i_src +  0));
        R5C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  5) * i_src + 16));
        R6C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  6) * i_src +  0));
        R6C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  6) * i_src + 16));
        R7C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  7) * i_src +  0));
        R7C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  7) * i_src + 16));
        R8C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  8) * i_src +  0));
        R8C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  8) * i_src + 16));
        R9C0  = _mm256_load_si256((__m256i*)(src + (i * 16 +  9) * i_src +  0));
        R9C1  = _mm256_load_si256((__m256i*)(src + (i * 16 +  9) * i_src + 16));
        R10C0 = _mm256_load_si256((__m256i*)(src + (i * 16 + 10) * i_src +  0));
        R10C1 = _mm256_load_si256((__m256i*)(src + (i * 16 + 10) * i_src + 16));
        R11C0 = _mm256_load_si256((__m256i*)(src + (i * 16 + 11) * i_src +  0));
        R11C1 = _mm256_load_si256((__m256i*)(src + (i * 16 + 11) * i_src + 16));
        R12C0 = _mm256_load_si256((__m256i*)(src + (i * 16 + 12) * i_src +  0));
        R12C1 = _mm256_load_si256((__m256i*)(src + (i * 16 + 12) * i_src + 16));
        R13C0 = _mm256_load_si256((__m256i*)(src + (i * 16 + 13) * i_src +  0));
        R13C1 = _mm256_load_si256((__m256i*)(src + (i * 16 + 13) * i_src + 16));
        R14C0 = _mm256_load_si256((__m256i*)(src + (i * 16 + 14) * i_src +  0));
        R14C1 = _mm256_load_si256((__m256i*)(src + (i * 16 + 14) * i_src + 16));
        R15C0 = _mm256_load_si256((__m256i*)(src + (i * 16 + 15) * i_src +  0));
        R15C1 = _mm256_load_si256((__m256i*)(src + (i * 16 + 15) * i_src + 16));

        //notice that different set / setr low dizhi butong
        __m256i tab_shuffle = _mm256_setr_epi16(0x0100, 0x0F0E, 0x0706, 0x0908, 0x0302, 0x0D0C, 0x0504, 0x0B0A,
                                                0x0100, 0x0F0E, 0x0706, 0x0908, 0x0302, 0x0D0C, 0x0504, 0x0B0A);

        __m256i tab_shuffle_1 = _mm256_setr_epi16(0x0100, 0x0B0A, 0x0302, 0x0908, 0x0504, 0x0F0E, 0x0706, 0x0D0C,
                                0x0100, 0x0B0A, 0x0302, 0x0908, 0x0504, 0x0F0E, 0x0706, 0x0D0C);

        __m256i tab_shuffle_2 = _mm256_setr_epi16(0x0302, 0x0100, 0x0706, 0x0504, 0x0B0A, 0x0908, 0x0F0E, 0x0D0C,
                                0x0302, 0x0100, 0x0706, 0x0504, 0x0B0A, 0x0908, 0x0F0E, 0x0D0C);

        //[13 10 14 09 12 11 15 08 05 02 06 01 04 03 07 00]
        //[29 26 30 25 28 27 31 24 21 18 22 17 20 19 23 16z
        R0C0 = _mm256_shuffle_epi8(R0C0, tab_shuffle);
        R0C1 = _mm256_shuffle_epi8(R0C1, tab_shuffle);
        R0C1 = _mm256_permute2x128_si256(R0C1, R0C1, 0x0003);//permute [21 18 22 17 20 19 23 16 / 29 26 30 25 28 27 31 24]
        R0C1 = _mm256_shuffle_epi8(R0C1, tab_shuffle_2);  // [18 21 17 22 19 20 16 23 / 26 29 25 30 27 28 24 31]
        //[13 10 14 09 12 11 15 08 / 05 02 06 01 04 03 07 00]
        //[18 21 17 22 19 20 16 23 / 26 29 25 30 27 28 24 31]
        R0_ODD = _mm256_sub_epi16(R0C0, R0C1);
        R0C0 = _mm256_add_epi16(R0C0, R0C1);//[13 10 14 09 12 11 15 08 / 05 02 06 01 04 03 07 00]
        R0C0 = _mm256_permute4x64_epi64(R0C0, 0x00D8);//[13 10 14 09 05 02 06 01 / 12 11 15 08 04 03 07 00]
        R0C0 = _mm256_shuffle_epi8(R0C0, tab_shuffle_1);//[10 05 13 02 09 06 14 01 / 11 04 12 03 08 07 15 00]

        R1C0 = _mm256_shuffle_epi8(R1C0, tab_shuffle);
        R1C1 = _mm256_shuffle_epi8(R1C1, tab_shuffle);
        R1C1 = _mm256_permute2x128_si256(R1C1, R1C1, 0x0003);
        R1C1 = _mm256_shuffle_epi8(R1C1, tab_shuffle_2);
        R1_ODD = _mm256_sub_epi16(R1C0, R1C1);

        R1C0 = _mm256_add_epi16(R1C0, R1C1);
        R1C0 = _mm256_permute4x64_epi64(R1C0, 0x00D8);
        R1C0 = _mm256_shuffle_epi8(R1C0, tab_shuffle_1);

        R2C0 = _mm256_shuffle_epi8(R2C0, tab_shuffle);
        R2C1 = _mm256_shuffle_epi8(R2C1, tab_shuffle);
        R2C1 = _mm256_permute2x128_si256(R2C1, R2C1, 0x0003);
        R2C1 = _mm256_shuffle_epi8(R2C1, tab_shuffle_2);
        R2_ODD = _mm256_sub_epi16(R2C0, R2C1);
        R2C0 = _mm256_add_epi16(R2C0, R2C1);
        R2C0 = _mm256_permute4x64_epi64(R2C0, 0x00D8);
        R2C0 = _mm256_shuffle_epi8(R2C0, tab_shuffle_1);

        R3C0 = _mm256_shuffle_epi8(R3C0, tab_shuffle);
        R3C1 = _mm256_shuffle_epi8(R3C1, tab_shuffle);
        R3C1 = _mm256_permute2x128_si256(R3C1, R3C1, 0x0003);
        R3C1 = _mm256_shuffle_epi8(R3C1, tab_shuffle_2);
        R3_ODD = _mm256_sub_epi16(R3C0, R3C1);
        R3C0 = _mm256_add_epi16(R3C0, R3C1);
        R3C0 = _mm256_permute4x64_epi64(R3C0, 0x00D8);
        R3C0 = _mm256_shuffle_epi8(R3C0, tab_shuffle_1);

        R4C0 = _mm256_shuffle_epi8(R4C0, tab_shuffle);
        R4C1 = _mm256_shuffle_epi8(R4C1, tab_shuffle);
        R4C1 = _mm256_permute2x128_si256(R4C1, R4C1, 0x0003);
        R4C1 = _mm256_shuffle_epi8(R4C1, tab_shuffle_2);
        R4_ODD = _mm256_sub_epi16(R4C0, R4C1);
        R4C0 = _mm256_add_epi16(R4C0, R4C1);
        R4C0 = _mm256_permute4x64_epi64(R4C0, 0x00D8);
        R4C0 = _mm256_shuffle_epi8(R4C0, tab_shuffle_1);


        R5C0 = _mm256_shuffle_epi8(R5C0, tab_shuffle);
        R5C1 = _mm256_shuffle_epi8(R5C1, tab_shuffle);
        R5C1 = _mm256_permute2x128_si256(R5C1, R5C1, 0x0003);
        R5C1 = _mm256_shuffle_epi8(R5C1, tab_shuffle_2);
        R5_ODD = _mm256_sub_epi16(R5C0, R5C1);
        R5C0 = _mm256_add_epi16(R5C0, R5C1);
        R5C0 = _mm256_permute4x64_epi64(R5C0, 0x00D8);
        R5C0 = _mm256_shuffle_epi8(R5C0, tab_shuffle_1);

        R6C0 = _mm256_shuffle_epi8(R6C0, tab_shuffle);
        R6C1 = _mm256_shuffle_epi8(R6C1, tab_shuffle);
        R6C1 = _mm256_permute2x128_si256(R6C1, R6C1, 0x0003);
        R6C1 = _mm256_shuffle_epi8(R6C1, tab_shuffle_2);
        R6_ODD = _mm256_sub_epi16(R6C0, R6C1);
        R6C0 = _mm256_add_epi16(R6C0, R6C1);
        R6C0 = _mm256_permute4x64_epi64(R6C0, 0x00D8);
        R6C0 = _mm256_shuffle_epi8(R6C0, tab_shuffle_1);

        R7C0 = _mm256_shuffle_epi8(R7C0, tab_shuffle);
        R7C1 = _mm256_shuffle_epi8(R7C1, tab_shuffle);
        R7C1 = _mm256_permute2x128_si256(R7C1, R7C1, 0x0003);
        R7C1 = _mm256_shuffle_epi8(R7C1, tab_shuffle_2);
        R7_ODD = _mm256_sub_epi16(R7C0, R7C1);
        R7C0 = _mm256_add_epi16(R7C0, R7C1);
        R7C0 = _mm256_permute4x64_epi64(R7C0, 0x00D8);
        R7C0 = _mm256_shuffle_epi8(R7C0, tab_shuffle_1);


        R8C0 = _mm256_shuffle_epi8(R8C0, tab_shuffle);
        R8C1 = _mm256_shuffle_epi8(R8C1, tab_shuffle);
        R8C1 = _mm256_permute2x128_si256(R8C1, R8C1, 0x0003);
        R8C1 = _mm256_shuffle_epi8(R8C1, tab_shuffle_2);
        R8_ODD = _mm256_sub_epi16(R8C0, R8C1);
        R8C0 = _mm256_add_epi16(R8C0, R8C1);
        R8C0 = _mm256_permute4x64_epi64(R8C0, 0x00D8);
        R8C0 = _mm256_shuffle_epi8(R8C0, tab_shuffle_1);

        R9C0 = _mm256_shuffle_epi8(R9C0, tab_shuffle);
        R9C1 = _mm256_shuffle_epi8(R9C1, tab_shuffle);
        R9C1 = _mm256_permute2x128_si256(R9C1, R9C1, 0x0003);
        R9C1 = _mm256_shuffle_epi8(R9C1, tab_shuffle_2);
        R9_ODD = _mm256_sub_epi16(R9C0, R9C1);
        R9C0 = _mm256_add_epi16(R9C0, R9C1);
        R9C0 = _mm256_permute4x64_epi64(R9C0, 0x00D8);
        R9C0 = _mm256_shuffle_epi8(R9C0, tab_shuffle_1);

        R10C0 = _mm256_shuffle_epi8(R10C0, tab_shuffle);
        R10C1 = _mm256_shuffle_epi8(R10C1, tab_shuffle);
        R10C1 = _mm256_permute2x128_si256(R10C1, R10C1, 0x0003);
        R10C1 = _mm256_shuffle_epi8(R10C1, tab_shuffle_2);
        R10_ODD = _mm256_sub_epi16(R10C0, R10C1);
        R10C0 = _mm256_add_epi16(R10C0, R10C1);
        R10C0 = _mm256_permute4x64_epi64(R10C0, 0x00D8);
        R10C0 = _mm256_shuffle_epi8(R10C0, tab_shuffle_1);

        R11C0 = _mm256_shuffle_epi8(R11C0, tab_shuffle);
        R11C1 = _mm256_shuffle_epi8(R11C1, tab_shuffle);
        R11C1 = _mm256_permute2x128_si256(R11C1, R11C1, 0x0003);
        R11C1 = _mm256_shuffle_epi8(R11C1, tab_shuffle_2);
        R11_ODD = _mm256_sub_epi16(R11C0, R11C1);
        R11C0 = _mm256_add_epi16(R11C0, R11C1);
        R11C0 = _mm256_permute4x64_epi64(R11C0, 0x00D8);
        R11C0 = _mm256_shuffle_epi8(R11C0, tab_shuffle_1);

        R12C0 = _mm256_shuffle_epi8(R12C0, tab_shuffle);
        R12C1 = _mm256_shuffle_epi8(R12C1, tab_shuffle);
        R12C1 = _mm256_permute2x128_si256(R12C1, R12C1, 0x0003);
        R12C1 = _mm256_shuffle_epi8(R12C1, tab_shuffle_2);
        R12_ODD = _mm256_sub_epi16(R12C0, R12C1);
        R12C0 = _mm256_add_epi16(R12C0, R12C1);
        R12C0 = _mm256_permute4x64_epi64(R12C0, 0x00D8);
        R12C0 = _mm256_shuffle_epi8(R12C0, tab_shuffle_1);

        R13C0 = _mm256_shuffle_epi8(R13C0, tab_shuffle);
        R13C1 = _mm256_shuffle_epi8(R13C1, tab_shuffle);
        R13C1 = _mm256_permute2x128_si256(R13C1, R13C1, 0x0003);
        R13C1 = _mm256_shuffle_epi8(R13C1, tab_shuffle_2);
        R13_ODD = _mm256_sub_epi16(R13C0, R13C1);
        R13C0 = _mm256_add_epi16(R13C0, R13C1);
        R13C0 = _mm256_permute4x64_epi64(R13C0, 0x00D8);
        R13C0 = _mm256_shuffle_epi8(R13C0, tab_shuffle_1);

        R14C0 = _mm256_shuffle_epi8(R14C0, tab_shuffle);
        R14C1 = _mm256_shuffle_epi8(R14C1, tab_shuffle);
        R14C1 = _mm256_permute2x128_si256(R14C1, R14C1, 0x0003);
        R14C1 = _mm256_shuffle_epi8(R14C1, tab_shuffle_2);
        R14_ODD = _mm256_sub_epi16(R14C0, R14C1);
        R14C0 = _mm256_add_epi16(R14C0, R14C1);
        R14C0 = _mm256_permute4x64_epi64(R14C0, 0x00D8);
        R14C0 = _mm256_shuffle_epi8(R14C0, tab_shuffle_1);

        R15C0 = _mm256_shuffle_epi8(R15C0, tab_shuffle);
        R15C1 = _mm256_shuffle_epi8(R15C1, tab_shuffle);
        R15C1 = _mm256_permute2x128_si256(R15C1, R15C1, 0x0003);
        R15C1 = _mm256_shuffle_epi8(R15C1, tab_shuffle_2);
        R15_ODD = _mm256_sub_epi16(R15C0, R15C1);
        R15C0 = _mm256_add_epi16(R15C0, R15C1);
        R15C0 = _mm256_permute4x64_epi64(R15C0, 0x00D8);
        R15C0 = _mm256_shuffle_epi8(R15C0, tab_shuffle_1);


        R0R1 = _mm256_hadd_epi16(R0C0, R1C0);//[105 102 106 101 005 002 006 001 / 104 103 107 100 004 003 007 000]
        R2R3 = _mm256_hadd_epi16(R2C0, R3C0);
        R4R5 = _mm256_hadd_epi16(R4C0, R5C0);
        R6R7 = _mm256_hadd_epi16(R6C0, R7C0);
        R8R9 = _mm256_hadd_epi16(R8C0, R9C0);//[905 902 906 901 805 802 806 801 / 904 903 907 900 804 803 807 800]
        R10R11 = _mm256_hadd_epi16(R10C0, R11C0);
        R12R13 = _mm256_hadd_epi16(R12C0, R13C0);
        R14R15 = _mm256_hadd_epi16(R14C0, R15C0);

        // mul the coefficient
        //0th row ,1th row   [105+102 106+101 005+002 006+001 / 104+103 107+100 004+003 007+000]
        tab_t = _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[0]);
        A0C0 = _mm256_madd_epi16(R0R1,   tab_t);
        A1C0 = _mm256_madd_epi16(R2R3,   tab_t);// 2   3
        A2C0 = _mm256_madd_epi16(R4R5,   tab_t);// 4   5
        A3C0 = _mm256_madd_epi16(R6R7,   tab_t);// 6   7
        A4C0 = _mm256_madd_epi16(R8R9,   tab_t);// 8   9
        A5C0 = _mm256_madd_epi16(R10R11, tab_t);//10  11
        A6C0 = _mm256_madd_epi16(R12R13, tab_t);//12  13
        A7C0 = _mm256_madd_epi16(R14R15, tab_t);//14  15

        A0C0 = _mm256_hadd_epi32(A0C0, A1C0); //[3B 2B 1B 0B(05+02+06+01) / 3A 2A 1A 0A(04+03+07+00)]
        A1C0 = _mm256_permute2f128_si256(A0C0, A0C0, 0x0001); //[3A 2A 1A 0A / 3B 2B 1B 0B]

        A2C0 = _mm256_hadd_epi32(A2C0, A3C0); //[7B 6B 5B 4B / 7A 6A 5A 4A]
        A3C0 = _mm256_permute2f128_si256(A2C0, A2C0, 0x0001);//[7A 6A 5A 4A / 7B 6B 5B 4B]

        A4C0 = _mm256_hadd_epi32(A4C0, A5C0); //[11B 10B 9B 8B / 11A 10A 9A 8A]
        A5C0 = _mm256_permute2f128_si256(A4C0, A4C0, 0x0001);//[11A 10A 9A 8A / 11B 10B 9B 8B]

        A6C0 = _mm256_hadd_epi32(A6C0, A7C0); //[15B 14B 13B 12B / 15A 14A 13A 12A]
        A7C0 = _mm256_permute2f128_si256(A6C0, A6C0, 0x0001);//[15A 14A 13A 12A / 15B 14B 13B 12B]


        COE0 = _mm256_add_epi32(A0C0, A1C0); //the same line`s data add to low 128 bit
        COE1 = _mm256_add_epi32(A2C0, A3C0);
        COE2 = _mm256_add_epi32(A4C0, A5C0);
        COE3 = _mm256_add_epi32(A6C0, A7C0);
        //low 128 bit is 0 1 2 3 rows data ,the high 128 bit is 8 9 10 11 rows data
        COE0 = _mm256_blend_epi32(COE0, COE2, 0x00F0);//[11 10 9 8 3 2 1 0]
        COE1 = _mm256_blend_epi32(COE1, COE3, 0x00F0);//[15 14 13 12 7 6 5 4]

        COE0 = _mm256_srai_epi32(_mm256_add_epi32(COE0, c_add1), shift1);
        COE1 = _mm256_srai_epi32(_mm256_add_epi32(COE1, c_add1), shift1);

        COE_RESULT = _mm256_packs_epi32(COE0, COE1);//[15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0]

        im[0][i] = COE_RESULT;

#define MAKE_ODD(tab,dstPos) \
    tab_t = _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab]);  \
    A0C0 = _mm256_madd_epi16(R0R1,   tab_t); \
    A1C0 = _mm256_madd_epi16(R2R3,   tab_t); \
    A2C0 = _mm256_madd_epi16(R4R5,   tab_t); \
    A3C0 = _mm256_madd_epi16(R6R7,   tab_t); \
    A4C0 = _mm256_madd_epi16(R8R9,   tab_t); \
    A5C0 = _mm256_madd_epi16(R10R11, tab_t); \
    A6C0 = _mm256_madd_epi16(R12R13, tab_t); \
    A7C0 = _mm256_madd_epi16(R14R15, tab_t); \
    \
    A0C0 = _mm256_hadd_epi32(A0C0, A1C0); \
    A1C0 = _mm256_permute2f128_si256(A0C0, A0C0, 0x0001); \
    \
    A2C0 = _mm256_hadd_epi32(A2C0, A3C0); \
    A3C0 = _mm256_permute2f128_si256(A2C0, A2C0, 0x0001); \
    \
    A4C0 = _mm256_hadd_epi32(A4C0, A5C0); \
    A5C0 = _mm256_permute2f128_si256(A4C0, A4C0, 0x0001); \
    \
    A6C0 = _mm256_hadd_epi32(A6C0, A7C0); \
    A7C0 = _mm256_permute2f128_si256(A6C0, A6C0, 0x0001); \
    \
    COE0 = _mm256_add_epi32(A0C0, A1C0); \
    COE1 = _mm256_add_epi32(A2C0, A3C0); \
    COE2 = _mm256_add_epi32(A4C0, A5C0); \
    COE3 = _mm256_add_epi32(A6C0, A7C0); \
    \
    COE0 = _mm256_blend_epi32(COE0, COE2, 0x00F0); \
    COE1 = _mm256_blend_epi32(COE1, COE3, 0x00F0); \
    \
    COE0 = _mm256_srai_epi32(_mm256_add_epi32(COE0, c_add1), shift1); \
    COE1 = _mm256_srai_epi32(_mm256_add_epi32(COE1, c_add1), shift1); \
    \
    COE_RESULT = _mm256_packs_epi32(COE0, COE1); \
    im[dstPos][i] = COE_RESULT;

        MAKE_ODD(1, 8);

#undef MAKE_ODD


#define MAKE_ODD(tab,dstPos) \
    tab_t = _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab]);  \
    A0C0 = _mm256_madd_epi16(R0R1,   tab_t); \
    A1C0 = _mm256_madd_epi16(R2R3,   tab_t); \
    A2C0 = _mm256_madd_epi16(R4R5,   tab_t); \
    A3C0 = _mm256_madd_epi16(R6R7,   tab_t); \
    A4C0 = _mm256_madd_epi16(R8R9,   tab_t); \
    A5C0 = _mm256_madd_epi16(R10R11, tab_t); \
    A6C0 = _mm256_madd_epi16(R12R13, tab_t); \
    A7C0 = _mm256_madd_epi16(R14R15, tab_t); \
    \
    A0C0 = _mm256_hadd_epi32(A0C0, A1C0); \
    A1C0 = _mm256_permute2f128_si256(A0C0, A0C0, 0x0001); \
    \
    A2C0 = _mm256_hadd_epi32(A2C0, A3C0); \
    A3C0 = _mm256_permute2f128_si256(A2C0, A2C0, 0x0001); \
    \
    A4C0 = _mm256_hadd_epi32(A4C0, A5C0); \
    A5C0 = _mm256_permute2f128_si256(A4C0, A4C0, 0x0001); \
    \
    A6C0 = _mm256_hadd_epi32(A6C0, A7C0); \
    A7C0 = _mm256_permute2f128_si256(A6C0, A6C0, 0x0001); \
    \
    COE0 = _mm256_add_epi32(A0C0, A1C0); \
    COE1 = _mm256_add_epi32(A2C0, A3C0); \
    COE2 = _mm256_add_epi32(A4C0, A5C0); \
    COE3 = _mm256_add_epi32(A6C0, A7C0); \
    \
    COE0 = _mm256_blend_epi32(COE0, COE2, 0x00F0); \
    COE1 = _mm256_blend_epi32(COE1, COE3, 0x00F0); \
    \
    COE0 = _mm256_srai_epi32(_mm256_add_epi32(COE0, c_add1), shift1); \
    COE1 = _mm256_srai_epi32(_mm256_add_epi32(COE1, c_add1), shift1); \
    \
    COE_RESULT = _mm256_packs_epi32(COE0, COE1); \
    im[dstPos][i] = COE_RESULT;

        MAKE_ODD(3, 4);
        MAKE_ODD(4, 12);

        R0R1 = _mm256_hsub_epi16(R0C0, R1C0);//[105 102 106 101 005 002 006 001 / 104 103 107 100 004 003 007 000]
        R2R3 = _mm256_hsub_epi16(R2C0, R3C0);
        R4R5 = _mm256_hsub_epi16(R4C0, R5C0);
        R6R7 = _mm256_hsub_epi16(R6C0, R7C0);
        R8R9 = _mm256_hsub_epi16(R8C0, R9C0);//[905 902 906 901 805 802 806 801 / 904 903 907 900 804 803 807 800]
        R10R11 = _mm256_hsub_epi16(R10C0, R11C0);
        R12R13 = _mm256_hsub_epi16(R12C0, R13C0);
        R14R15 = _mm256_hsub_epi16(R14C0, R15C0);

        MAKE_ODD(7, 2);
        MAKE_ODD(8, 6);
        MAKE_ODD(9, 10);
        MAKE_ODD(10, 14);
#undef MAKE_ODD


#define MAKE_ODD(tab,dstPos) \
    tab_t = _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab]);  \
    A0C0 = _mm256_madd_epi16(R0_ODD,  tab_t); \
    A0C1 = _mm256_madd_epi16(R1_ODD,  tab_t); \
    A1C0 = _mm256_madd_epi16(R2_ODD,  tab_t); \
    A1C1 = _mm256_madd_epi16(R3_ODD,  tab_t); \
    A2C0 = _mm256_madd_epi16(R4_ODD,  tab_t); \
    A2C1 = _mm256_madd_epi16(R5_ODD,  tab_t); \
    A3C0 = _mm256_madd_epi16(R6_ODD,  tab_t); \
    A3C1 = _mm256_madd_epi16(R7_ODD,  tab_t); \
    A4C0 = _mm256_madd_epi16(R8_ODD,  tab_t); \
    A4C1 = _mm256_madd_epi16(R9_ODD,  tab_t); \
    A5C0 = _mm256_madd_epi16(R10_ODD, tab_t); \
    A5C1 = _mm256_madd_epi16(R11_ODD, tab_t); \
    A6C0 = _mm256_madd_epi16(R12_ODD, tab_t); \
    A6C1 = _mm256_madd_epi16(R13_ODD, tab_t); \
    A7C0 = _mm256_madd_epi16(R14_ODD, tab_t); \
    A7C1 = _mm256_madd_epi16(R15_ODD, tab_t); \
    \
    A0C0 = _mm256_hadd_epi32(A0C0, A0C1); \
    A1C0 = _mm256_hadd_epi32(A1C0, A1C1); \
    A2C0 = _mm256_hadd_epi32(A2C0, A2C1); \
    A3C0 = _mm256_hadd_epi32(A3C0, A3C1); \
    A4C0 = _mm256_hadd_epi32(A4C0, A4C1); \
    A5C0 = _mm256_hadd_epi32(A5C0, A5C1); \
    A6C0 = _mm256_hadd_epi32(A6C0, A6C1); \
    A7C0 = _mm256_hadd_epi32(A7C0, A7C1); \
    \
    A0C0 = _mm256_hadd_epi32(A0C0, A1C0); \
    A1C0 = _mm256_hadd_epi32(A2C0, A3C0); \
    A2C0 = _mm256_hadd_epi32(A4C0, A5C0); \
    A3C0 = _mm256_hadd_epi32(A6C0, A7C0); \
    \
    A0C1 = _mm256_permute2f128_si256(A0C0, A2C0, 0x0020); \
    A1C1 = _mm256_permute2f128_si256(A0C0, A2C0, 0x0031); \
    A2C1 = _mm256_permute2f128_si256(A1C0, A3C0, 0x0020); \
    A3C1 = _mm256_permute2f128_si256(A1C0, A3C0, 0x0031); \
    \
    COE0 = _mm256_add_epi32(A0C1, A1C1); \
    COE1 = _mm256_add_epi32(A2C1, A3C1); \
    COE0 = _mm256_srai_epi32(_mm256_add_epi32(COE0, c_add1), shift1); \
    COE1 = _mm256_srai_epi32(_mm256_add_epi32(COE1, c_add1), shift1); \
    \
    COE_RESULT = _mm256_packs_epi32(COE0, COE1); \
    im[dstPos][i] = COE_RESULT;

        MAKE_ODD(15, 1);
        MAKE_ODD(16, 3);
        MAKE_ODD(17, 5);
        MAKE_ODD(18, 7);
        MAKE_ODD(19, 9);
        MAKE_ODD(20, 11);
        MAKE_ODD(21, 13);
        MAKE_ODD(22, 15);
#undef MAKE_ODD
    }

    /* clear result buffer */
    xavs2_memzero_aligned_c_avx(dst, 32 * 32 * sizeof(coeff_t));

    __m128i mask = _mm_set1_epi16(0xffff);
    // DCT2， 只保留前16行和16列
    for (i = 0; i < 16 / 8; i++) {
        R0C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 0) * i_src) +  0), mask));
        R0C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 0) * i_src) +  8), mask));
        R1C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 0) * i_src) + 16), mask));
        R1C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 0) * i_src) + 24), mask));

        R2C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 1) * i_src) +  0), mask));
        R2C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 1) * i_src) +  8), mask));
        R3C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 1) * i_src) + 16), mask));
        R3C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 1) * i_src) + 24), mask));

        R4C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 2) * i_src) +  0), mask));
        R4C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 2) * i_src) +  8), mask));
        R5C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 2) * i_src) + 16), mask));
        R5C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 2) * i_src) + 24), mask));

        R6C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 3) * i_src) +  0), mask));
        R6C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 3) * i_src) +  8), mask));
        R7C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 3) * i_src) + 16), mask));
        R7C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 3) * i_src) + 24), mask));

        R8C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 4) * i_src) +  0), mask));
        R8C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 4) * i_src) +  8), mask));
        R9C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 4) * i_src) + 16), mask));
        R9C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 4) * i_src) + 24), mask));

        R10C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 5) * i_src) +  0), mask));
        R10C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 5) * i_src) +  8), mask));
        R11C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 5) * i_src) + 16), mask));
        R11C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 5) * i_src) + 24), mask));

        R12C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 6) * i_src) +  0), mask));
        R12C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 6) * i_src) +  8), mask));
        R13C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 6) * i_src) + 16), mask));
        R13C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 6) * i_src) + 24), mask));

        R14C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 7) * i_src) +  0), mask));
        R14C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 7) * i_src) +  8), mask));
        R15C0 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 7) * i_src) + 16), mask));
        R15C1 = _mm256_cvtepi16_epi32(_mm_maskload_epi32((int const*)((int16_t*)(im)+((i * 8 + 7) * i_src) + 24), mask));


        // inverse _m256i per 32 bit
        __m256i tab_inv = _mm256_setr_epi32(0x0007, 0x0006, 0x0005, 0x0004, 0x0003, 0x0002, 0x0001, 0x0000);


        R0C1 = _mm256_permutevar8x32_epi32(R0C1, tab_inv); //[8 9 10 11 / 12 13 14 15]
        R1C1 = _mm256_permutevar8x32_epi32(R1C1, tab_inv); //[24 25 26 27 / 28 29 30 31]
        R0_ODD = _mm256_sub_epi32(R0C0, R1C1); //[7-24 6-25  5-26  4-27 /  3-28  2-29  1-30  0-31]
        R15_ODD = _mm256_sub_epi32(R0C1, R1C0); //[8-23 9-22 10-21 11-20 / 12-19 13-18 14-17 15-16]
        R0C0 = _mm256_add_epi32(R0C0, R1C1);    //[7 6  5  4 /  3  2  1  0]
        R0C1 = _mm256_add_epi32(R0C1, R1C0);    //[8 9 10 11 / 12 13 14 15]
        A0C0 = _mm256_add_epi32(R0C0, R0C1);    //[7 6 5 4 / 3 2 1 0]



        R2C1 = _mm256_permutevar8x32_epi32(R2C1, tab_inv);
        R3C1 = _mm256_permutevar8x32_epi32(R3C1, tab_inv);
        R1_ODD = _mm256_sub_epi32(R2C0, R3C1);
        R14_ODD = _mm256_sub_epi32(R2C1, R3C0);
        R1C0 = _mm256_add_epi32(R2C0, R3C1);
        R1C1 = _mm256_add_epi32(R2C1, R3C0);
        A1C0 = _mm256_add_epi32(R1C0, R1C1);

        R4C1 = _mm256_permutevar8x32_epi32(R4C1, tab_inv);
        R5C1 = _mm256_permutevar8x32_epi32(R5C1, tab_inv);
        R2_ODD = _mm256_sub_epi32(R4C0, R5C1);
        R13_ODD = _mm256_sub_epi32(R4C1, R5C0);
        R2C0 = _mm256_add_epi32(R4C0, R5C1);
        R2C1 = _mm256_add_epi32(R4C1, R5C0);
        A2C0 = _mm256_add_epi32(R2C0, R2C1);

        R6C1 = _mm256_permutevar8x32_epi32(R6C1, tab_inv);
        R7C1 = _mm256_permutevar8x32_epi32(R7C1, tab_inv);
        R3_ODD = _mm256_sub_epi32(R6C0, R7C1);
        R12_ODD = _mm256_sub_epi32(R6C1, R7C0);
        R3C0 = _mm256_add_epi32(R6C0, R7C1);
        R3C1 = _mm256_add_epi32(R6C1, R7C0);
        A3C0 = _mm256_add_epi32(R3C0, R3C1);

        R8C1 = _mm256_permutevar8x32_epi32(R8C1, tab_inv);
        R9C1 = _mm256_permutevar8x32_epi32(R9C1, tab_inv);
        R4_ODD = _mm256_sub_epi32(R8C0, R9C1);
        R11_ODD = _mm256_sub_epi32(R8C1, R9C0);
        R4C0 = _mm256_add_epi32(R8C0, R9C1);
        R4C1 = _mm256_add_epi32(R8C1, R9C0);
        A4C0 = _mm256_add_epi32(R4C0, R4C1);


        R10C1 = _mm256_permutevar8x32_epi32(R10C1, tab_inv);
        R11C1 = _mm256_permutevar8x32_epi32(R11C1, tab_inv);
        R5_ODD = _mm256_sub_epi32(R10C0, R11C1);
        R10_ODD = _mm256_sub_epi32(R10C1, R11C0);
        R5C0 = _mm256_add_epi32(R10C0, R11C1);
        R5C1 = _mm256_add_epi32(R10C1, R11C0);
        A5C0 = _mm256_add_epi32(R5C0, R5C1);


        R12C1 = _mm256_permutevar8x32_epi32(R12C1, tab_inv);
        R13C1 = _mm256_permutevar8x32_epi32(R13C1, tab_inv);
        R6_ODD = _mm256_sub_epi32(R12C0, R13C1);
        R9_ODD = _mm256_sub_epi32(R12C1, R13C0);
        R6C0 = _mm256_add_epi32(R12C0, R13C1);
        R6C1 = _mm256_add_epi32(R12C1, R13C0);
        A6C0 = _mm256_add_epi32(R6C0, R6C1);



        R14C1 = _mm256_permutevar8x32_epi32(R14C1, tab_inv);
        R15C1 = _mm256_permutevar8x32_epi32(R15C1, tab_inv);
        R7_ODD = _mm256_sub_epi32(R14C0, R15C1);//[7-24 6-25  5-26  4-27 /  3-28  2-29  1-30  0-31]
        R8_ODD = _mm256_sub_epi32(R14C1, R15C0);  //[8-23 9-22 10-21 11-20 / 12-19 13-18 14-17 15-16]
        R7C0 = _mm256_add_epi32(R14C0, R15C1);    //[7 6  5  4 /  3  2  1  0]
        R7C1 = _mm256_add_epi32(R14C1, R15C0);    //[8 9 10 11 / 12 13 14 15]
        A7C0 = _mm256_add_epi32(R7C0, R7C1);      //[7 6 5 4 / 3 2 1 0]


        __m256i result_mask = _mm256_setr_epi32(0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0, 0, 0, 0);
#define MAKE_ODD(tab,dstPos) \
    tab_t = _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab]);  \
    A0C1 = _mm256_mullo_epi32(A0C0, tab_t); \
    A1C1 = _mm256_mullo_epi32(A1C0, tab_t); \
    A2C1 = _mm256_mullo_epi32(A2C0, tab_t); \
    A3C1 = _mm256_mullo_epi32(A3C0, tab_t); \
    A4C1 = _mm256_mullo_epi32(A4C0, tab_t); \
    A5C1 = _mm256_mullo_epi32(A5C0, tab_t); \
    A6C1 = _mm256_mullo_epi32(A6C0, tab_t); \
    A7C1 = _mm256_mullo_epi32(A7C0, tab_t); \
    \
    COE0 = _mm256_hadd_epi32(A0C1, A1C1); /* [107+106 105+104 007+006 005+004 / 103+102 101+100 003+002 001+000] */\
    COE1 = _mm256_hadd_epi32(A2C1, A3C1); \
    COE2 = _mm256_hadd_epi32(A4C1, A5C1); \
    COE3 = _mm256_hadd_epi32(A6C1, A7C1); \
    \
    COE0 = _mm256_hadd_epi32(COE0, COE1); /* [3A 2A 1A 0A / 3B 2B 1B 0B] */\
    COE1 = _mm256_hadd_epi32(COE2, COE3); /* [7A 6A 5A 4A / 7B 6B 5B 4B] */\
    \
    COE2 = _mm256_permute2f128_si256(COE0, COE1, 0x0020); /*[7B 6B 5B 4B / 3B 2B 1B 0B]*/\
    COE3 = _mm256_permute2f128_si256(COE0, COE1, 0x0031); /*[7A 6A 5A 4A / 3A 2A 1A 0A]*/\
    \
    COE_RESULT = _mm256_add_epi32(COE2, COE3); /* [7 6 5 4 / 3 2 1 0] */\
    COE_RESULT = _mm256_srai_epi32(_mm256_add_epi32(COE_RESULT, c_add2), SHIFT2); \
    COE0 = _mm256_permute2f128_si256(COE_RESULT, COE_RESULT, 0x0001);/* [3 2 1 0 / 7 6 5 4] */ \
    COE_RESULT = _mm256_packs_epi32(COE_RESULT, COE0); /*[3 2 1 0 7 6 5 4 / 7 6 5 4 3 2 1 0]*/\
    addr = (dst + (dstPos * 32) + (i * 8)); \
    \
    _mm256_maskstore_epi32((int*)addr, result_mask, COE_RESULT);
        //_mm256_storeu2_m128i(addr, &COE3, COE_RESULT);

        /*_mm256_storeu_si256(addr, COE_RESULT);*/
        MAKE_ODD(0, 0);
        MAKE_ODD(1, 8);
        MAKE_ODD(4, 4);
        MAKE_ODD(5, 12);


        A0C0 = _mm256_sub_epi32(R0C0, R0C1);
        A1C0 = _mm256_sub_epi32(R1C0, R1C1);
        A2C0 = _mm256_sub_epi32(R2C0, R2C1);
        A3C0 = _mm256_sub_epi32(R3C0, R3C1);
        A4C0 = _mm256_sub_epi32(R4C0, R4C1);
        A5C0 = _mm256_sub_epi32(R5C0, R5C1);
        A6C0 = _mm256_sub_epi32(R6C0, R6C1);
        A7C0 = _mm256_sub_epi32(R7C0, R7C1);

        MAKE_ODD(8, 2);
        MAKE_ODD(9, 6);
        MAKE_ODD(10, 10);
        MAKE_ODD(11, 14);


#undef MAKE_ODD


#define MAKE_ODD(tab,dstPos) \
    tab_t  = _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab]);  \
    tab_t1 = _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab + 1]);  \
    A0C1 = _mm256_add_epi32(_mm256_mullo_epi32(R0_ODD, tab_t), _mm256_mullo_epi32(R15_ODD, tab_t1)); \
    A1C1 = _mm256_add_epi32(_mm256_mullo_epi32(R1_ODD, tab_t), _mm256_mullo_epi32(R14_ODD, tab_t1)); \
    A2C1 = _mm256_add_epi32(_mm256_mullo_epi32(R2_ODD, tab_t), _mm256_mullo_epi32(R13_ODD, tab_t1)); \
    A3C1 = _mm256_add_epi32(_mm256_mullo_epi32(R3_ODD, tab_t), _mm256_mullo_epi32(R12_ODD, tab_t1)); \
    A4C1 = _mm256_add_epi32(_mm256_mullo_epi32(R4_ODD, tab_t), _mm256_mullo_epi32(R11_ODD, tab_t1)); \
    A5C1 = _mm256_add_epi32(_mm256_mullo_epi32(R5_ODD, tab_t), _mm256_mullo_epi32(R10_ODD, tab_t1)); \
    A6C1 = _mm256_add_epi32(_mm256_mullo_epi32(R6_ODD, tab_t), _mm256_mullo_epi32(R9_ODD,  tab_t1)); \
    A7C1 = _mm256_add_epi32(_mm256_mullo_epi32(R7_ODD, tab_t), _mm256_mullo_epi32(R8_ODD,  tab_t1)); \
    \
    COE0 = _mm256_hadd_epi32(A0C1, A1C1); \
    COE1 = _mm256_hadd_epi32(A2C1, A3C1); \
    COE2 = _mm256_hadd_epi32(A4C1, A5C1); \
    COE3 = _mm256_hadd_epi32(A6C1, A7C1); \
    \
    COE0 = _mm256_hadd_epi32(COE0, COE1); \
    COE1 = _mm256_hadd_epi32(COE2, COE3); \
    \
    COE2 = _mm256_permute2f128_si256(COE0, COE1, 0x0020); \
    COE3 = _mm256_permute2f128_si256(COE0, COE1, 0x0031); \
    \
    COE_RESULT = _mm256_add_epi32(COE2, COE3); \
    COE_RESULT = _mm256_srai_epi32(_mm256_add_epi32(COE_RESULT, c_add2), SHIFT2); \
    COE0 = _mm256_permute2f128_si256(COE_RESULT, COE_RESULT, 0x0001); \
    COE_RESULT = _mm256_packs_epi32(COE_RESULT, COE0); \
    addr = (dst + (dstPos * 32) + (i * 8)); \
    \
    _mm256_maskstore_epi32((int*)addr, result_mask, COE_RESULT);

        //_mm256_storeu2_m128i(addr, &COE3, COE_RESULT);

        //_mm256_storeu_si256(addr, COE_RESULT);

        MAKE_ODD(16, 1);
        MAKE_ODD(18, 3);
        MAKE_ODD(20, 5);
        MAKE_ODD(22, 7);
        MAKE_ODD(24, 9);
        MAKE_ODD(26, 11);
        MAKE_ODD(28, 13);
        MAKE_ODD(30, 15);

#undef MAKE_ODD

    }
}

/* ---------------------------------------------------------------------------
 */
void dct_c_8x32_avx2(const coeff_t *src, coeff_t *dst, int i_src)
{
    __m256i line00, line10, line20, line30, line40, line50, line60, line70;
    __m256i line01, line11, line21, line31, line41, line51, line61, line71;
    __m256i line02, line12, line22, line32, line42, line52, line62, line72;
    __m256i line03, line13, line23, line33, line43, line53, line63, line73;
    __m256i e0, e1;
    __m256i  o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15;
    __m256i ee0, eo0;
    __m256i add1, add2;
    __m256i im[32][4];
    ALIGN32(static const int16_t shuffle[]) = {
        0x0100, 0x0F0E, 0x0706, 0x0908, 0x0302, 0x0D0C, 0x0504, 0x0B0A,
        0x0100, 0x0F0E, 0x0706, 0x0908, 0x0302, 0x0D0C, 0x0504, 0x0B0A,
    };

    __m256i A0C0, A0C1, A1C0, A1C1, A2C0, A2C1, A3C0, A3C1, A4C0, A4C1, A5C0, A5C1, A6C0, A6C1, A7C0, A7C1;
    __m256i COE0, COE1, COE2, COE3;
    __m256i COE_RESULT;

    int shift1, shift2;
    int i;

    shift1 = B32X32_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT - 2 + (i_src & 0x01);
    shift2 = B32X32_IN_BIT + FACTO_BIT;

    const int ADD11 = (1 << shift1) >> 1;
    const int ADD21 = (1 << shift2) >> 1;
    add1 = _mm256_set1_epi32(ADD11);        // TODO: shift1 = 2
    add2 = _mm256_set1_epi32(ADD21);

    i_src &= 0xFE;    /* remember to remove the flag bit */

    // dct1
    for (i = 0; i < 32 / 8; i++) {
        line00 = _mm256_loadu2_m128i((__m128i *)(src + (i * 8 + 4) * i_src), (__m128i *)(src + (i * 8 + 0) * i_src)); // [00~07 40~47]
        line10 = _mm256_loadu2_m128i((__m128i *)(src + (i * 8 + 5) * i_src), (__m128i *)(src + (i * 8 + 1) * i_src)); // [10~17 50~57]
        line20 = _mm256_loadu2_m128i((__m128i *)(src + (i * 8 + 6) * i_src), (__m128i *)(src + (i * 8 + 2) * i_src)); // [20~27 60~67]
        line30 = _mm256_loadu2_m128i((__m128i *)(src + (i * 8 + 7) * i_src), (__m128i *)(src + (i * 8 + 3) * i_src)); // [30~37 70~77]

        line00 = _mm256_shuffle_epi8(line00, _mm256_load_si256((const __m256i *)shuffle)); // [00 07 03 04 01 06 02 05 40 47 43 44 41 46 42 45]
        line10 = _mm256_shuffle_epi8(line10, _mm256_load_si256((const __m256i *)shuffle));
        line20 = _mm256_shuffle_epi8(line20, _mm256_load_si256((const __m256i *)shuffle));
        line30 = _mm256_shuffle_epi8(line30, _mm256_load_si256((const __m256i *)shuffle));

        e0 = _mm256_hadd_epi16(line00, line10); // [e00 e03 e01 e02 e10 e13 e11 e12 e40 e43 e41 e42 e50 e53 e51 e52]
        e1 = _mm256_hadd_epi16(line20, line30);
        o0 = _mm256_hsub_epi16(line00, line10); // [o00 o03 o01 o02 o10 o13 o11 o12 o40 o43 o41 o42 o50 o53 o51 o52]
        o1 = _mm256_hsub_epi16(line20, line30);

        ee0 = _mm256_hadd_epi16(e0, e1); // [ee00 ee01 ee10 ee11 ee20 ee21 ee30 ee31 ee40 ee41 ee50 ee51 ee60 ee61 ee70 ee71]
        eo0 = _mm256_hsub_epi16(e0, e1);

        line00 = _mm256_madd_epi16(ee0, _mm256_load_si256((const __m256i *)tab_dct_8x32_avx2[0]));
        line40 = _mm256_madd_epi16(ee0, _mm256_load_si256((const __m256i *)tab_dct_8x32_avx2[4]));
        line20 = _mm256_madd_epi16(eo0, _mm256_load_si256((const __m256i *)tab_dct_8x32_avx2[2]));
        line60 = _mm256_madd_epi16(eo0, _mm256_load_si256((const __m256i *)tab_dct_8x32_avx2[6]));

#define CALC_DATA(line, tab) \
    line = _mm256_hadd_epi32(\
    _mm256_madd_epi16(o0, _mm256_load_si256((const __m256i *)tab_dct_8x32_avx2[tab])), \
    _mm256_madd_epi16(o1, _mm256_load_si256((const __m256i *)tab_dct_8x32_avx2[tab]))  \
    );
        CALC_DATA(line10, 1);
        CALC_DATA(line30, 3);
        CALC_DATA(line50, 5);
        CALC_DATA(line70, 7);
#undef CALC_DATA

        _mm256_storeu_si256(&im[0][i], _mm256_srai_epi32(_mm256_add_epi32(line00, add1), shift1));
        _mm256_storeu_si256(&im[1][i], _mm256_srai_epi32(_mm256_add_epi32(line10, add1), shift1));
        _mm256_storeu_si256(&im[2][i], _mm256_srai_epi32(_mm256_add_epi32(line20, add1), shift1));
        _mm256_storeu_si256(&im[3][i], _mm256_srai_epi32(_mm256_add_epi32(line30, add1), shift1));
        _mm256_storeu_si256(&im[4][i], _mm256_srai_epi32(_mm256_add_epi32(line40, add1), shift1));
        _mm256_storeu_si256(&im[5][i], _mm256_srai_epi32(_mm256_add_epi32(line50, add1), shift1));
        _mm256_storeu_si256(&im[6][i], _mm256_srai_epi32(_mm256_add_epi32(line60, add1), shift1));
        _mm256_storeu_si256(&im[7][i], _mm256_srai_epi32(_mm256_add_epi32(line70, add1), shift1));
    }

    //DCT2
#define load_one_line(x) \
    line##x##0 = _mm256_load_si256(&im[x][0]); \
    line##x##1 = _mm256_load_si256(&im[x][1]); \
    line##x##2 = _mm256_load_si256(&im[x][2]); \
    line##x##3 = _mm256_load_si256(&im[x][3])  \
 
    load_one_line(0);
    load_one_line(1);
    load_one_line(2);
    load_one_line(3);
    load_one_line(4);
    load_one_line(5);
    load_one_line(6);
    load_one_line(7);
#undef load_one_line

    //inverse _m256i per 32 bit
    __m256i tab_inv = _mm256_setr_epi32(0x0007, 0x0006, 0x0005, 0x0004, 0x0003, 0x0002, 0x0001, 0x0000);

    line01 = _mm256_permutevar8x32_epi32(line01, tab_inv); //[8 9 10 11 / 12 13 14 15]
    line03 = _mm256_permutevar8x32_epi32(line03, tab_inv); //[24 25 26 27 / 28 29 30 31]
    o0 = _mm256_sub_epi32(line00, line03); //[7-24 6-25  5-26  4-27 /  3-28  2-29  1-30  0-31]
    o15 = _mm256_sub_epi32(line01, line02); //[8-23 9-22 10-21 11-20 / 12-19 13-18 14-17 15-16]
    line00 = _mm256_add_epi32(line00, line03);    //[7 6  5  4 /  3  2  1  0]
    line01 = _mm256_add_epi32(line01, line02);    //[8 9 10 11 / 12 13 14 15]
    A0C0 = _mm256_add_epi32(line00, line01);    //[7 6 5 4 / 3 2 1 0]

    line11 = _mm256_permutevar8x32_epi32(line11, tab_inv);
    line13 = _mm256_permutevar8x32_epi32(line13, tab_inv);
    o1 = _mm256_sub_epi32(line10, line13);
    o14 = _mm256_sub_epi32(line11, line12);
    line02 = _mm256_add_epi32(line10, line13);
    line03 = _mm256_add_epi32(line11, line12);
    A1C0 = _mm256_add_epi32(line02, line03);

    line21 = _mm256_permutevar8x32_epi32(line21, tab_inv);
    line23 = _mm256_permutevar8x32_epi32(line23, tab_inv);
    o2 = _mm256_sub_epi32(line20, line23);
    o13 = _mm256_sub_epi32(line21, line22);
    line10 = _mm256_add_epi32(line20, line23);
    line11 = _mm256_add_epi32(line21, line22);
    A2C0 = _mm256_add_epi32(line10, line11);

    line31 = _mm256_permutevar8x32_epi32(line31, tab_inv);
    line33 = _mm256_permutevar8x32_epi32(line33, tab_inv);
    o3 = _mm256_sub_epi32(line30, line33);
    o12 = _mm256_sub_epi32(line31, line32);
    line12 = _mm256_add_epi32(line30, line33);
    line13 = _mm256_add_epi32(line31, line32);
    A3C0 = _mm256_add_epi32(line12, line13);

    line41 = _mm256_permutevar8x32_epi32(line41, tab_inv);
    line43 = _mm256_permutevar8x32_epi32(line43, tab_inv);
    o4 = _mm256_sub_epi32(line40, line43);
    o11 = _mm256_sub_epi32(line41, line42);
    line20 = _mm256_add_epi32(line40, line43);
    line21 = _mm256_add_epi32(line41, line42);
    A4C0 = _mm256_add_epi32(line20, line21);

    line51 = _mm256_permutevar8x32_epi32(line51, tab_inv);
    line53 = _mm256_permutevar8x32_epi32(line53, tab_inv);
    o5 = _mm256_sub_epi32(line50, line53);
    o10 = _mm256_sub_epi32(line51, line52);
    line22 = _mm256_add_epi32(line50, line53);
    line23 = _mm256_add_epi32(line51, line52);
    A5C0 = _mm256_add_epi32(line22, line23);

    line61 = _mm256_permutevar8x32_epi32(line61, tab_inv);
    line63 = _mm256_permutevar8x32_epi32(line63, tab_inv);
    o6 = _mm256_sub_epi32(line60, line63);
    o9 = _mm256_sub_epi32(line61, line62);
    line30 = _mm256_add_epi32(line60, line63);
    line31 = _mm256_add_epi32(line61, line62);
    A6C0 = _mm256_add_epi32(line30, line31);

    line71 = _mm256_permutevar8x32_epi32(line71, tab_inv);
    line73 = _mm256_permutevar8x32_epi32(line73, tab_inv);
    o7 = _mm256_sub_epi32(line70, line73);//[7-24 6-25  5-26  4-27 /  3-28  2-29  1-30  0-31]
    o8 = _mm256_sub_epi32(line71, line72);  //[8-23 9-22 10-21 11-20 / 12-19 13-18 14-17 15-16]
    line32 = _mm256_add_epi32(line70, line73);    //[7 6  5  4 /  3  2  1  0]
    line33 = _mm256_add_epi32(line71, line72);    //[8 9 10 11 / 12 13 14 15]
    A7C0 = _mm256_add_epi32(line32, line33);      //[7 6 5 4 / 3 2 1 0]

    __m256i result_mask = _mm256_setr_epi32(0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0, 0, 0, 0);
#define MAKE_ODD(tab,dstPos)\
    A0C1 = _mm256_mullo_epi32(A0C0, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab])); \
    A1C1 = _mm256_mullo_epi32(A1C0, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab])); \
    A2C1 = _mm256_mullo_epi32(A2C0, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab])); \
    A3C1 = _mm256_mullo_epi32(A3C0, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab])); \
    A4C1 = _mm256_mullo_epi32(A4C0, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab])); \
    A5C1 = _mm256_mullo_epi32(A5C0, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab])); \
    A6C1 = _mm256_mullo_epi32(A6C0, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab])); \
    A7C1 = _mm256_mullo_epi32(A7C0, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab])); \
    \
    COE0 = _mm256_hadd_epi32(A0C1, A1C1); /* [107+106 105+104 007+006 005+004 / 103+102 101+100 003+002 001+000] */\
    COE1 = _mm256_hadd_epi32(A2C1, A3C1); \
    COE2 = _mm256_hadd_epi32(A4C1, A5C1); \
    COE3 = _mm256_hadd_epi32(A6C1, A7C1); \
    \
    COE0 = _mm256_hadd_epi32(COE0, COE1); /* [3A 2A 1A 0A / 3B 2B 1B 0B] */\
    COE1 = _mm256_hadd_epi32(COE2, COE3); /* [7A 6A 5A 4A / 7B 6B 5B 4B] */\
    \
    COE2 = _mm256_permute2f128_si256(COE0, COE1, 0x0020); /*[7B 6B 5B 4B / 3B 2B 1B 0B]*/\
    COE3 = _mm256_permute2f128_si256(COE0, COE1, 0x0031); /*[7A 6A 5A 4A / 3A 2A 1A 0A]*/\
    \
    COE_RESULT = _mm256_add_epi32(COE2, COE3); /* [7 6 5 4 / 3 2 1 0] */\
    COE_RESULT = _mm256_srai_epi32(_mm256_add_epi32(COE_RESULT, add2), shift2); \
    COE0 = _mm256_permute2f128_si256(COE_RESULT, COE_RESULT, 0x0001);/* [3 2 1 0 / 7 6 5 4] */ \
    COE_RESULT = _mm256_packs_epi32(COE_RESULT, COE0); /*[3 2 1 0 7 6 5 4 / 7 6 5 4 3 2 1 0]*/\
    _mm256_maskstore_epi32((int *)(dst + dstPos * 8), result_mask, COE_RESULT);

    MAKE_ODD(0, 0);
    MAKE_ODD(1, 8);
    MAKE_ODD(2, 16);
    MAKE_ODD(3, 24);
    MAKE_ODD(4, 4);
    MAKE_ODD(5, 12);
    MAKE_ODD(6, 20);
    MAKE_ODD(7, 28);

    A0C0 = _mm256_sub_epi32(line00, line01);
    A1C0 = _mm256_sub_epi32(line02, line03);
    A2C0 = _mm256_sub_epi32(line10, line11);
    A3C0 = _mm256_sub_epi32(line12, line13);
    A4C0 = _mm256_sub_epi32(line20, line21);
    A5C0 = _mm256_sub_epi32(line22, line23);
    A6C0 = _mm256_sub_epi32(line30, line31);
    A7C0 = _mm256_sub_epi32(line32, line33);

    MAKE_ODD(8, 2);
    MAKE_ODD(9, 6);
    MAKE_ODD(10, 10);
    MAKE_ODD(11, 14);
    MAKE_ODD(12, 18);
    MAKE_ODD(13, 22);
    MAKE_ODD(14, 26);
    MAKE_ODD(15, 30);
#undef MAKE_ODD

#define MAKE_ODD(tab,dstPos) \
    A0C1 = _mm256_add_epi32(_mm256_mullo_epi32(o0, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab])), _mm256_mullo_epi32(o15, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab + 1]))); \
    A1C1 = _mm256_add_epi32(_mm256_mullo_epi32(o1, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab])), _mm256_mullo_epi32(o14, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab + 1]))); \
    A2C1 = _mm256_add_epi32(_mm256_mullo_epi32(o2, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab])), _mm256_mullo_epi32(o13, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab + 1]))); \
    A3C1 = _mm256_add_epi32(_mm256_mullo_epi32(o3, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab])), _mm256_mullo_epi32(o12, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab + 1]))); \
    A4C1 = _mm256_add_epi32(_mm256_mullo_epi32(o4, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab])), _mm256_mullo_epi32(o11, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab + 1]))); \
    A5C1 = _mm256_add_epi32(_mm256_mullo_epi32(o5, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab])), _mm256_mullo_epi32(o10, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab + 1]))); \
    A6C1 = _mm256_add_epi32(_mm256_mullo_epi32(o6, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab])), _mm256_mullo_epi32(o9, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab + 1]))); \
    A7C1 = _mm256_add_epi32(_mm256_mullo_epi32(o7, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab])), _mm256_mullo_epi32(o8, _mm256_load_si256((__m256i*)tab_dct2_32x32_avx2[tab + 1]))); \
    \
    COE0 = _mm256_hadd_epi32(A0C1, A1C1); \
    COE1 = _mm256_hadd_epi32(A2C1, A3C1); \
    COE2 = _mm256_hadd_epi32(A4C1, A5C1); \
    COE3 = _mm256_hadd_epi32(A6C1, A7C1); \
    \
    COE0 = _mm256_hadd_epi32(COE0, COE1); \
    COE1 = _mm256_hadd_epi32(COE2, COE3); \
    \
    COE2 = _mm256_permute2f128_si256(COE0, COE1, 0x0020); \
    COE3 = _mm256_permute2f128_si256(COE0, COE1, 0x0031); \
    \
    COE_RESULT = _mm256_add_epi32(COE2, COE3); \
    COE_RESULT = _mm256_srai_epi32(_mm256_add_epi32(COE_RESULT, add2), shift2); \
    COE0 = _mm256_permute2f128_si256(COE_RESULT, COE_RESULT, 0x0001); \
    COE_RESULT = _mm256_packs_epi32(COE_RESULT, COE0); \
    _mm256_maskstore_epi32((int *)(dst + dstPos * 8), result_mask, COE_RESULT);

    MAKE_ODD(16, 1);
    MAKE_ODD(18, 3);
    MAKE_ODD(20, 5);
    MAKE_ODD(22, 7);
    MAKE_ODD(24, 9);
    MAKE_ODD(26, 11);
    MAKE_ODD(28, 13);
    MAKE_ODD(30, 15);
    MAKE_ODD(32, 17);
    MAKE_ODD(34, 19);
    MAKE_ODD(36, 21);
    MAKE_ODD(38, 23);
    MAKE_ODD(40, 25);
    MAKE_ODD(42, 27);
    MAKE_ODD(44, 29);
    MAKE_ODD(46, 31);
#undef MAKE_ODD
}


/* ---------------------------------------------------------------------------
 */
void dct_c_32x8_avx2(const coeff_t *src, coeff_t *dst, int i_src)
{

    //const int shift1 = SHIFT1 + (i_src & 0x01);
    int shift1 = B32X32_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT;
    int shift2 = B32X32_IN_BIT + FACTO_BIT - 2 - (i_src & 0x01);
    const int ADD1 = (1 << shift1) >> 1;
    const int ADD2 = (1 << shift2) >> 1;
    const __m256i c_add1 = _mm256_set1_epi32(ADD1);        // TODO: shift1 = 2
    const __m256i c_add2 = _mm256_set1_epi32(ADD2);

    //R---row  C-column
    __m256i R0C0, R0C1, R1C0, R1C1, R2C0, R2C1, R3C0, R3C1, R4C0, R4C1, R5C0, R5C1, R6C0, R6C1, R7C0, R7C1;
    //store anser
    __m256i A0C0, A0C1, A1C0, A1C1, A2C0, A2C1, A3C0, A3C1;
    __m256i R0R1, R2R3, R4R5, R6R7;
    __m256i COE0, COE1;
    __m256i COE_RESULT;
    __m256i im[32];

    __m256i  R0_ODD, R1_ODD, R2_ODD, R3_ODD, R4_ODD, R5_ODD, R6_ODD, R7_ODD;

    int i;
    coeff_t* addr;

    i_src &= 0xFE;

    R0C0 = _mm256_load_si256((__m256i*)(src + 0 * i_src + 0));  //[15 14 13 12 11 10... 03 02 01 00]
    R0C1 = _mm256_load_si256((__m256i*)(src + 0 * i_src + 16));  //[31 30 29 28 11 10... 19 18 17 16]
    R1C0 = _mm256_load_si256((__m256i*)(src + 1 * i_src + 0));
    R1C1 = _mm256_load_si256((__m256i*)(src + 1 * i_src + 16));
    R2C0 = _mm256_load_si256((__m256i*)(src + 2 * i_src + 0));
    R2C1 = _mm256_load_si256((__m256i*)(src + 2 * i_src + 16));
    R3C0 = _mm256_load_si256((__m256i*)(src + 3 * i_src + 0));
    R3C1 = _mm256_load_si256((__m256i*)(src + 3 * i_src + 16));
    R4C0 = _mm256_load_si256((__m256i*)(src + 4 * i_src + 0));
    R4C1 = _mm256_load_si256((__m256i*)(src + 4 * i_src + 16));
    R5C0 = _mm256_load_si256((__m256i*)(src + 5 * i_src + 0));
    R5C1 = _mm256_load_si256((__m256i*)(src + 5 * i_src + 16));
    R6C0 = _mm256_load_si256((__m256i*)(src + 6 * i_src + 0));
    R6C1 = _mm256_load_si256((__m256i*)(src + 6 * i_src + 16));
    R7C0 = _mm256_load_si256((__m256i*)(src + 7 * i_src + 0));
    R7C1 = _mm256_load_si256((__m256i*)(src + 7 * i_src + 16));

    //notice that different set / setr low dizhi butong
    __m256i tab_shuffle =
        _mm256_setr_epi16(0x0100, 0x0F0E, 0x0706, 0x0908, 0x0302, 0x0D0C, 0x0504, 0x0B0A,
                          0x0100, 0x0F0E, 0x0706, 0x0908, 0x0302, 0x0D0C, 0x0504, 0x0B0A);

    __m256i tab_shuffle_1 =
        _mm256_setr_epi16(0x0100, 0x0B0A, 0x0302, 0x0908, 0x0504, 0x0F0E, 0x0706, 0x0D0C,
                          0x0100, 0x0B0A, 0x0302, 0x0908, 0x0504, 0x0F0E, 0x0706, 0x0D0C);

    __m256i tab_shuffle_2 =
        _mm256_setr_epi16(0x0302, 0x0100, 0x0706, 0x0504, 0x0B0A, 0x0908, 0x0F0E, 0x0D0C,
                          0x0302, 0x0100, 0x0706, 0x0504, 0x0B0A, 0x0908, 0x0F0E, 0x0D0C);


    //[13 10 14 09 12 11 15 08 05 02 06 01 04 03 07 00]
    //[29 26 30 25 28 27 31 24 21 18 22 17 20 19 23 16]
    R0C0 = _mm256_shuffle_epi8(R0C0, tab_shuffle);
    R0C1 = _mm256_shuffle_epi8(R0C1, tab_shuffle);
    R0C1 = _mm256_permute2x128_si256(R0C1, R0C1, 0x0003);//permute [21 18 22 17 20 19 23 16 / 29 26 30 25 28 27 31 24]
    R0C1 = _mm256_shuffle_epi8(R0C1, tab_shuffle_2);  // [18 21 17 22 19 20 16 23 / 26 29 25 30 27 28 24 31]
    //[13 10 14 09 12 11 15 08 / 05 02 06 01 04 03 07 00]
    //[18 21 17 22 19 20 16 23 / 26 29 25 30 27 28 24 31]
    R0_ODD = _mm256_sub_epi16(R0C0, R0C1);
    R0C0 = _mm256_add_epi16(R0C0, R0C1);//[13 10 14 09 12 11 15 08 / 05 02 06 01 04 03 07 00]
    R0C0 = _mm256_permute4x64_epi64(R0C0, 0x00D8);//[13 10 14 09 05 02 06 01 / 12 11 15 08 04 03 07 00]
    R0C0 = _mm256_shuffle_epi8(R0C0, tab_shuffle_1);//[10 05 13 02 09 06 14 01 / 11 04 12 03 08 07 15 00]


    R1C0 = _mm256_shuffle_epi8(R1C0, tab_shuffle);
    R1C1 = _mm256_shuffle_epi8(R1C1, tab_shuffle);
    R1C1 = _mm256_permute2x128_si256(R1C1, R1C1, 0x0003);
    R1C1 = _mm256_shuffle_epi8(R1C1, tab_shuffle_2);
    R1_ODD = _mm256_sub_epi16(R1C0, R1C1);
    R1C0 = _mm256_add_epi16(R1C0, R1C1);
    R1C0 = _mm256_permute4x64_epi64(R1C0, 0x00D8);
    R1C0 = _mm256_shuffle_epi8(R1C0, tab_shuffle_1);

    R2C0 = _mm256_shuffle_epi8(R2C0, tab_shuffle);
    R2C1 = _mm256_shuffle_epi8(R2C1, tab_shuffle);
    R2C1 = _mm256_permute2x128_si256(R2C1, R2C1, 0x0003);
    R2C1 = _mm256_shuffle_epi8(R2C1, tab_shuffle_2);
    R2_ODD = _mm256_sub_epi16(R2C0, R2C1);
    R2C0 = _mm256_add_epi16(R2C0, R2C1);
    R2C0 = _mm256_permute4x64_epi64(R2C0, 0x00D8);
    R2C0 = _mm256_shuffle_epi8(R2C0, tab_shuffle_1);

    R3C0 = _mm256_shuffle_epi8(R3C0, tab_shuffle);
    R3C1 = _mm256_shuffle_epi8(R3C1, tab_shuffle);
    R3C1 = _mm256_permute2x128_si256(R3C1, R3C1, 0x0003);
    R3C1 = _mm256_shuffle_epi8(R3C1, tab_shuffle_2);
    R3_ODD = _mm256_sub_epi16(R3C0, R3C1);
    R3C0 = _mm256_add_epi16(R3C0, R3C1);
    R3C0 = _mm256_permute4x64_epi64(R3C0, 0x00D8);
    R3C0 = _mm256_shuffle_epi8(R3C0, tab_shuffle_1);

    R4C0 = _mm256_shuffle_epi8(R4C0, tab_shuffle);
    R4C1 = _mm256_shuffle_epi8(R4C1, tab_shuffle);
    R4C1 = _mm256_permute2x128_si256(R4C1, R4C1, 0x0003);
    R4C1 = _mm256_shuffle_epi8(R4C1, tab_shuffle_2);
    R4_ODD = _mm256_sub_epi16(R4C0, R4C1);
    R4C0 = _mm256_add_epi16(R4C0, R4C1);
    R4C0 = _mm256_permute4x64_epi64(R4C0, 0x00D8);
    R4C0 = _mm256_shuffle_epi8(R4C0, tab_shuffle_1);

    R5C0 = _mm256_shuffle_epi8(R5C0, tab_shuffle);
    R5C1 = _mm256_shuffle_epi8(R5C1, tab_shuffle);
    R5C1 = _mm256_permute2x128_si256(R5C1, R5C1, 0x0003);
    R5C1 = _mm256_shuffle_epi8(R5C1, tab_shuffle_2);
    R5_ODD = _mm256_sub_epi16(R5C0, R5C1);
    R5C0 = _mm256_add_epi16(R5C0, R5C1);
    R5C0 = _mm256_permute4x64_epi64(R5C0, 0x00D8);
    R5C0 = _mm256_shuffle_epi8(R5C0, tab_shuffle_1);

    R6C0 = _mm256_shuffle_epi8(R6C0, tab_shuffle);
    R6C1 = _mm256_shuffle_epi8(R6C1, tab_shuffle);
    R6C1 = _mm256_permute2x128_si256(R6C1, R6C1, 0x0003);
    R6C1 = _mm256_shuffle_epi8(R6C1, tab_shuffle_2);
    R6_ODD = _mm256_sub_epi16(R6C0, R6C1);
    R6C0 = _mm256_add_epi16(R6C0, R6C1);
    R6C0 = _mm256_permute4x64_epi64(R6C0, 0x00D8);
    R6C0 = _mm256_shuffle_epi8(R6C0, tab_shuffle_1);

    R7C0 = _mm256_shuffle_epi8(R7C0, tab_shuffle);
    R7C1 = _mm256_shuffle_epi8(R7C1, tab_shuffle);
    R7C1 = _mm256_permute2x128_si256(R7C1, R7C1, 0x0003);
    R7C1 = _mm256_shuffle_epi8(R7C1, tab_shuffle_2);
    R7_ODD = _mm256_sub_epi16(R7C0, R7C1);
    R7C0 = _mm256_add_epi16(R7C0, R7C1);
    R7C0 = _mm256_permute4x64_epi64(R7C0, 0x00D8);
    R7C0 = _mm256_shuffle_epi8(R7C0, tab_shuffle_1);

    R0R1 = _mm256_hadd_epi16(R0C0, R1C0);//[105 102 106 101 005 002 006 001 / 104 103 107 100 004 003 007 000]
    R2R3 = _mm256_hadd_epi16(R2C0, R3C0);
    R4R5 = _mm256_hadd_epi16(R4C0, R5C0);
    R6R7 = _mm256_hadd_epi16(R6C0, R7C0);

    // mul the coefficient
    //0th row ,1th row   [105+102 106+101 005+002 006+001 / 104+103 107+100 004+003 007+000]
    A0C0 = _mm256_madd_epi16(R0R1, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[0]));
    A1C0 = _mm256_madd_epi16(R2R3, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[0]));// 2   3
    A2C0 = _mm256_madd_epi16(R4R5, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[0]));// 4   5
    A3C0 = _mm256_madd_epi16(R6R7, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[0]));// 6   7

    A0C0 = _mm256_hadd_epi32(A0C0, A1C0); //[3B 2B 1B 0B(05+02+06+01) / 3A 2A 1A 0A(04+03+07+00)]
    A1C0 = _mm256_permute2f128_si256(A0C0, A0C0, 0x0001); //[3A 2A 1A 0A / 3B 2B 1B 0B]

    A2C0 = _mm256_hadd_epi32(A2C0, A3C0); //[7B 6B 5B 4B / 7A 6A 5A 4A]
    A3C0 = _mm256_permute2f128_si256(A2C0, A2C0, 0x0001);//[7A 6A 5A 4A / 7B 6B 5B 4B]

    COE0 = _mm256_add_epi32(A0C0, A1C0); //the same line`s data add to low 128 bit (3 2 1 0)
    COE1 = _mm256_add_epi32(A2C0, A3C0);

    COE0 = _mm256_srai_epi32(_mm256_add_epi32(COE0, c_add1), shift1);
    COE1 = _mm256_srai_epi32(_mm256_add_epi32(COE1, c_add1), shift1);

    COE_RESULT = _mm256_packs_epi32(COE0, COE1);//[15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0]

    im[0] = COE_RESULT;

    COE0 = _mm256_sub_epi32(A0C0, A1C0);
    COE1 = _mm256_sub_epi32(A2C0, A3C0);

    COE0 = _mm256_srai_epi32(_mm256_add_epi32(COE0, c_add1), shift1);
    COE1 = _mm256_srai_epi32(_mm256_add_epi32(COE1, c_add1), shift1);

    COE_RESULT = _mm256_packs_epi32(COE0, COE1);//[15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0]

    im[16] = COE_RESULT;



#define MAKE_ODD(tab,dstPos) \
    A0C0 = _mm256_madd_epi16(R0R1, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab])); \
    A1C0 = _mm256_madd_epi16(R2R3, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab])); \
    A2C0 = _mm256_madd_epi16(R4R5, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab])); \
    A3C0 = _mm256_madd_epi16(R6R7, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab])); \
    \
    A0C0 = _mm256_hadd_epi32(A0C0, A1C0); \
    A1C0 = _mm256_permute2f128_si256(A0C0, A0C0, 0x0001); \
    \
    A2C0 = _mm256_hadd_epi32(A2C0, A3C0); \
    A3C0 = _mm256_permute2f128_si256(A2C0, A2C0, 0x0001); \
    \
    COE0 = _mm256_add_epi32(A0C0, A1C0); \
    COE1 = _mm256_add_epi32(A2C0, A3C0); \
    \
    COE0 = _mm256_srai_epi32(_mm256_add_epi32(COE0, c_add1), shift1); \
    COE1 = _mm256_srai_epi32(_mm256_add_epi32(COE1, c_add1), shift1); \
    \
    COE_RESULT = _mm256_packs_epi32(COE0, COE1); \
    im[dstPos] = COE_RESULT;

    MAKE_ODD(1, 8);
    MAKE_ODD(2, 24);

#undef MAKE_ODD


#define MAKE_ODD(tab,dstPos) \
    A0C0 = _mm256_madd_epi16(R0R1, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab])); \
    A1C0 = _mm256_madd_epi16(R2R3, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab])); \
    A2C0 = _mm256_madd_epi16(R4R5, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab])); \
    A3C0 = _mm256_madd_epi16(R6R7, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab])); \
    \
    A0C0 = _mm256_hadd_epi32(A0C0, A1C0); \
    A1C0 = _mm256_permute2f128_si256(A0C0, A0C0, 0x0001); \
    \
    A2C0 = _mm256_hadd_epi32(A2C0, A3C0); \
    A3C0 = _mm256_permute2f128_si256(A2C0, A2C0, 0x0001); \
    \
    COE0 = _mm256_add_epi32(A0C0, A1C0); \
    COE1 = _mm256_add_epi32(A2C0, A3C0); \
    \
    COE0 = _mm256_srai_epi32(_mm256_add_epi32(COE0, c_add1), shift1); \
    COE1 = _mm256_srai_epi32(_mm256_add_epi32(COE1, c_add1), shift1); \
    \
    COE_RESULT = _mm256_packs_epi32(COE0, COE1); \
    \
    im[dstPos] = COE_RESULT;


    MAKE_ODD(3, 4);
    MAKE_ODD(4, 12);
    MAKE_ODD(5, 20);
    MAKE_ODD(6, 28);

    R0R1 = _mm256_hsub_epi16(R0C0, R1C0);//[105 102 106 101 005 002 006 001 / 104 103 107 100 004 003 007 000]
    R2R3 = _mm256_hsub_epi16(R2C0, R3C0);
    R4R5 = _mm256_hsub_epi16(R4C0, R5C0);
    R6R7 = _mm256_hsub_epi16(R6C0, R7C0);

    MAKE_ODD(7, 2);
    MAKE_ODD(8, 6);
    MAKE_ODD(9, 10);
    MAKE_ODD(10, 14);
    MAKE_ODD(11, 18);
    MAKE_ODD(12, 22);
    MAKE_ODD(13, 26);
    MAKE_ODD(14, 30);


#undef MAKE_ODD


#define MAKE_ODD(tab,dstPos) \
    A0C0 = _mm256_madd_epi16(R0_ODD, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab])); \
    A0C1 = _mm256_madd_epi16(R1_ODD, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab])); \
    A1C0 = _mm256_madd_epi16(R2_ODD, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab])); \
    A1C1 = _mm256_madd_epi16(R3_ODD, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab])); \
    A2C0 = _mm256_madd_epi16(R4_ODD, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab])); \
    A2C1 = _mm256_madd_epi16(R5_ODD, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab])); \
    A3C0 = _mm256_madd_epi16(R6_ODD, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab])); \
    A3C1 = _mm256_madd_epi16(R7_ODD, _mm256_load_si256((__m256i*)tab_dct_32x32_avx2[tab])); \
    \
    A0C0 = _mm256_hadd_epi32(A0C0, A0C1); \
    A1C0 = _mm256_hadd_epi32(A1C0, A1C1); \
    A2C0 = _mm256_hadd_epi32(A2C0, A2C1); \
    A3C0 = _mm256_hadd_epi32(A3C0, A3C1); \
    \
    A0C0 = _mm256_hadd_epi32(A0C0, A1C0)/*[3B 2B 1B 0B / 3A 2A 1A 0A]*/; \
    A1C0 = _mm256_hadd_epi32(A2C0, A3C0)/*[7B 6B 5B 4B / 7A 6A 5A 4A]*/; \
    \
    A0C1 = _mm256_permute2f128_si256(A0C0, A0C0, 0x0001)/*[3A 2A 1A 0A / 3B 2B 1B 0B]*/; \
    A1C1 = _mm256_permute2f128_si256(A1C0, A1C0, 0x0001)/*[7A 6A 5A 4A / 7B 6B 5B 4B]*/; \
    \
    COE0 = _mm256_add_epi32(A0C0, A0C1); \
    COE1 = _mm256_add_epi32(A1C0, A1C1); \
    COE0 = _mm256_srai_epi32(_mm256_add_epi32(COE0, c_add1), shift1); \
    COE1 = _mm256_srai_epi32(_mm256_add_epi32(COE1, c_add1), shift1); \
    \
    COE_RESULT = _mm256_packs_epi32(COE0, COE1); \
    \
    im[dstPos] = COE_RESULT;

    MAKE_ODD(15, 1);
    MAKE_ODD(16, 3);
    MAKE_ODD(17, 5);
    MAKE_ODD(18, 7);
    MAKE_ODD(19, 9);
    MAKE_ODD(20, 11);
    MAKE_ODD(21, 13);
    MAKE_ODD(22, 15);
    MAKE_ODD(23, 17);
    MAKE_ODD(24, 19);
    MAKE_ODD(25, 21);
    MAKE_ODD(26, 23);
    MAKE_ODD(27, 25);
    MAKE_ODD(28, 27);
    MAKE_ODD(29, 29);
    MAKE_ODD(30, 31);

#undef MAKE_ODD


    __m256i table_shuffle = _mm256_setr_epi16(0x0100, 0x0F0E, 0x0302, 0x0D0C, 0x0504, 0x0B0A, 0x0706, 0x0908,
                            0x0100, 0x0F0E, 0x0302, 0x0D0C, 0x0504, 0x0B0A, 0x0706, 0x0908);

    //__m256i im[32]

    for (i = 0; i < 32 / 8; i++) {

        R0C0 = _mm256_load_si256((__m256i const *)im + (i * 8 + 0));//[0 1 2 3 4 5 6 7 / *********]
        R1C0 = _mm256_load_si256((__m256i const *)im + (i * 8 + 1));
        R2C0 = _mm256_load_si256((__m256i const *)im + (i * 8 + 2));
        R3C0 = _mm256_load_si256((__m256i const *)im + (i * 8 + 3));
        R4C0 = _mm256_load_si256((__m256i const *)im + (i * 8 + 4));
        R5C0 = _mm256_load_si256((__m256i const *)im + (i * 8 + 5));
        R6C0 = _mm256_load_si256((__m256i const *)im + (i * 8 + 6));
        R7C0 = _mm256_load_si256((__m256i const *)im + (i * 8 + 7));

        R0C0 = _mm256_shuffle_epi8(R0C0, table_shuffle); //[00 07 01 06 02 05 03 04 / *********]
        R1C0 = _mm256_shuffle_epi8(R1C0, table_shuffle);
        R2C0 = _mm256_shuffle_epi8(R2C0, table_shuffle);
        R3C0 = _mm256_shuffle_epi8(R3C0, table_shuffle);
        R4C0 = _mm256_shuffle_epi8(R4C0, table_shuffle);
        R5C0 = _mm256_shuffle_epi8(R5C0, table_shuffle);
        R6C0 = _mm256_shuffle_epi8(R6C0, table_shuffle);
        R7C0 = _mm256_shuffle_epi8(R7C0, table_shuffle);


        R0C0 = _mm256_cvtepi16_epi32(*(__m128i*)&R0C0); //[00 07 01 06 / 02 05 03 04]
        R1C0 = _mm256_cvtepi16_epi32(*(__m128i*)&R1C0);
        R2C0 = _mm256_cvtepi16_epi32(*(__m128i*)&R2C0);
        R3C0 = _mm256_cvtepi16_epi32(*(__m128i*)&R3C0);
        R4C0 = _mm256_cvtepi16_epi32(*(__m128i*)&R4C0);
        R5C0 = _mm256_cvtepi16_epi32(*(__m128i*)&R5C0);
        R6C0 = _mm256_cvtepi16_epi32(*(__m128i*)&R6C0);
        R7C0 = _mm256_cvtepi16_epi32(*(__m128i*)&R7C0);



        R0R1 = _mm256_hadd_epi32(R0C0, R1C0);//[00+07 01+06 10+17 11+16 / 02+05 03+06 12+15 13+16]------------0/1
        R2R3 = _mm256_hadd_epi32(R2C0, R3C0);
        R4R5 = _mm256_hadd_epi32(R4C0, R5C0);
        R6R7 = _mm256_hadd_epi32(R6C0, R7C0);

        __m256i result_mask = _mm256_setr_epi32(0xf0000000, 0xf0000000, 0xf0000000, 0xf0000000, 0, 0, 0, 0);

#define MAKE_ODD(tab,dstPos)\
    R0C1 = _mm256_mullo_epi32(R0R1, _mm256_load_si256((__m256i *)tab_dct_8x8_avx2[tab])); \
    R1C1 = _mm256_mullo_epi32(R2R3, _mm256_load_si256((__m256i *)tab_dct_8x8_avx2[tab])); \
    R2C1 = _mm256_mullo_epi32(R4R5, _mm256_load_si256((__m256i *)tab_dct_8x8_avx2[tab])); \
    R3C1 = _mm256_mullo_epi32(R6R7, _mm256_load_si256((__m256i *)tab_dct_8x8_avx2[tab])); \
    \
    R0C1 = _mm256_hadd_epi32(R0C1, R1C1/*[00+07+01+06 10+17+11+16 20+27+21+26 30+27+41+36 / 02+05+03+06 12+15+13+16 22+25+23+26 32+35+33+36]*/); \
    R1C1 = _mm256_hadd_epi32(R2C1, R3C1); \
    \
    R2C1 = _mm256_permute2f128_si256(R0C1, R1C1, 0x0020/*[0A 1A 2A 3A / 4A 5A 6A 7A]*/); \
    R3C1 = _mm256_permute2f128_si256(R0C1, R1C1, 0x0031/*[0B 1B 2B 3B / 4B 5B 6B 7B]*/); \
    \
    COE0 = _mm256_add_epi32(R2C1, R3C1/*[0 1 2 3 / 4 5 6 7]*/); \
    COE1 = _mm256_permute2f128_si256(COE0, COE0, 0x33/*[4 5 6 7 / 4 5 6 7]*/); \
    \
    COE0 = _mm256_srai_epi32(_mm256_add_epi32(COE0, c_add2), shift2); \
    COE1 = _mm256_srai_epi32(_mm256_add_epi32(COE1, c_add2), shift2); \
    \
    COE_RESULT = _mm256_packs_epi32(COE0, COE1); \
    \
    addr = (dst + (dstPos * 32) + (i * 8)); \
    _mm256_maskstore_epi32((int*)addr, result_mask, COE_RESULT); \
 
        MAKE_ODD(0, 0);
        MAKE_ODD(2, 2);
        MAKE_ODD(4, 4);
        MAKE_ODD(6, 6);


#undef MAKE_ODD


        R0R1 = _mm256_hsub_epi32(R0C0, R1C0);//[00-07 01-06 10-17 11-16 / 02-05 03-06 12-15 13-16]------------0/1
        R2R3 = _mm256_hsub_epi32(R2C0, R3C0);
        R4R5 = _mm256_hsub_epi32(R4C0, R5C0);
        R6R7 = _mm256_hsub_epi32(R6C0, R7C0);

#define MAKE_ODD(tab,dstPos)\
    R0C1 = _mm256_mullo_epi32(R0R1, _mm256_load_si256((__m256i *)tab_dct_8x8_avx2[tab])); \
    R1C1 = _mm256_mullo_epi32(R2R3, _mm256_load_si256((__m256i *)tab_dct_8x8_avx2[tab])); \
    R2C1 = _mm256_mullo_epi32(R4R5, _mm256_load_si256((__m256i *)tab_dct_8x8_avx2[tab])); \
    R3C1 = _mm256_mullo_epi32(R6R7, _mm256_load_si256((__m256i *)tab_dct_8x8_avx2[tab])); \
    \
    R0C1 = _mm256_hadd_epi32(R0C1, R1C1/*[00+07+01+06 10+17+11+16 20+27+21+26 30+27+41+36 / 02+05+03+06 12+15+13+16 22+25+23+26 32+35+33+36]*/); \
    R1C1 = _mm256_hadd_epi32(R2C1, R3C1); \
    \
    R2C1 = _mm256_permute2f128_si256(R0C1, R1C1, 0x0020/*[0A 1A 2A 3A / 4A 5A 6A 7A]*/); \
    R3C1 = _mm256_permute2f128_si256(R0C1, R1C1, 0x0031/*[0B 1B 2B 3B / 4B 5B 6B 7B]*/); \
    \
    COE0 = _mm256_add_epi32(R2C1, R3C1/*[0 1 2 3 / 4 5 6 7]*/); \
    COE1 = _mm256_permute2f128_si256(COE0, COE0, 0x33/*[4 5 6 7 / 4 5 6 7]*/); \
    \
    COE0 = _mm256_srai_epi32(_mm256_add_epi32(COE0, c_add2), shift2); \
    COE1 = _mm256_srai_epi32(_mm256_add_epi32(COE1, c_add2), shift2); \
    \
    COE_RESULT = _mm256_packs_epi32(COE0, COE1); \
    \
    addr = (dst + (dstPos * 32) + (i * 8)); \
    _mm256_maskstore_epi32((int*)addr, result_mask, COE_RESULT); \
 
        MAKE_ODD(1, 1);
        MAKE_ODD(3, 3);
        MAKE_ODD(5, 5);
        MAKE_ODD(7, 7);


#undef MAKE_ODD

    }


}


/* ---------------------------------------------------------------------------
 * transpose 16x16(矩阵转置)
 */
#define TRANSPOSE_16x16_16BIT(A00, A01, A02, A03, A04, A05, A06, A07, A08, A09, A10, A11, A12, A13, A14, A15, H00, H01,H02, H03, H04, H05, H06, H07, H08, H09, H10, H11, H12, H13, H14, H15) \
    tr0_00 = _mm256_unpacklo_epi16(A00, A01); \
    tr0_01 = _mm256_unpacklo_epi16(A02, A03); \
    tr0_02 = _mm256_unpackhi_epi16(A00, A01); \
    tr0_03 = _mm256_unpackhi_epi16(A02, A03); \
    tr0_04 = _mm256_unpacklo_epi16(A04, A05); \
    tr0_05 = _mm256_unpacklo_epi16(A06, A07); \
    tr0_06 = _mm256_unpackhi_epi16(A04, A05); \
    tr0_07 = _mm256_unpackhi_epi16(A06, A07); \
    tr0_08 = _mm256_unpacklo_epi16(A08, A09); \
    tr0_09 = _mm256_unpacklo_epi16(A10, A11); \
    tr0_10 = _mm256_unpackhi_epi16(A08, A09); \
    tr0_11 = _mm256_unpackhi_epi16(A10, A11); \
    tr0_12 = _mm256_unpacklo_epi16(A12, A13); \
    tr0_13 = _mm256_unpacklo_epi16(A14, A15); \
    tr0_14 = _mm256_unpackhi_epi16(A12, A13); \
    tr0_15 = _mm256_unpackhi_epi16(A14, A15); \
    tr1_00 = _mm256_unpacklo_epi32(tr0_00, tr0_01); \
    tr1_01 = _mm256_unpacklo_epi32(tr0_02, tr0_03); \
    tr1_02 = _mm256_unpackhi_epi32(tr0_00, tr0_01); \
    tr1_03 = _mm256_unpackhi_epi32(tr0_02, tr0_03); \
    tr1_04 = _mm256_unpacklo_epi32(tr0_04, tr0_05); \
    tr1_05 = _mm256_unpacklo_epi32(tr0_06, tr0_07); \
    tr1_06 = _mm256_unpackhi_epi32(tr0_04, tr0_05); \
    tr1_07 = _mm256_unpackhi_epi32(tr0_06, tr0_07); \
    tr1_08 = _mm256_unpacklo_epi32(tr0_08, tr0_09); \
    tr1_09 = _mm256_unpacklo_epi32(tr0_10, tr0_11); \
    tr1_10 = _mm256_unpackhi_epi32(tr0_08, tr0_09); \
    tr1_11 = _mm256_unpackhi_epi32(tr0_10, tr0_11); \
    tr1_12 = _mm256_unpacklo_epi32(tr0_12, tr0_13); \
    tr1_13 = _mm256_unpacklo_epi32(tr0_14, tr0_15); \
    tr1_14 = _mm256_unpackhi_epi32(tr0_12, tr0_13); \
    tr1_15 = _mm256_unpackhi_epi32(tr0_14, tr0_15); \
    tr0_00 = _mm256_unpacklo_epi64(tr1_00, tr1_04); \
    tr0_01 = _mm256_unpackhi_epi64(tr1_00, tr1_04); \
    tr0_02 = _mm256_unpacklo_epi64(tr1_02, tr1_06); \
    tr0_03 = _mm256_unpackhi_epi64(tr1_02, tr1_06); \
    tr0_04 = _mm256_unpacklo_epi64(tr1_01, tr1_05); \
    tr0_05 = _mm256_unpackhi_epi64(tr1_01, tr1_05); \
    tr0_06 = _mm256_unpacklo_epi64(tr1_03, tr1_07); \
    tr0_07 = _mm256_unpackhi_epi64(tr1_03, tr1_07); \
    tr0_08 = _mm256_unpacklo_epi64(tr1_08, tr1_12); \
    tr0_09 = _mm256_unpackhi_epi64(tr1_08, tr1_12); \
    tr0_10 = _mm256_unpacklo_epi64(tr1_10, tr1_14); \
    tr0_11 = _mm256_unpackhi_epi64(tr1_10, tr1_14); \
    tr0_12 = _mm256_unpacklo_epi64(tr1_09, tr1_13); \
    tr0_13 = _mm256_unpackhi_epi64(tr1_09, tr1_13); \
    tr0_14 = _mm256_unpacklo_epi64(tr1_11, tr1_15); \
    tr0_15 = _mm256_unpackhi_epi64(tr1_11, tr1_15); \
    H00 = _mm256_permute2x128_si256(tr0_00, tr0_08, 0x20);\
    H01 = _mm256_permute2x128_si256(tr0_01, tr0_09, 0x20);\
    H02 = _mm256_permute2x128_si256(tr0_02, tr0_10, 0x20);\
    H03 = _mm256_permute2x128_si256(tr0_03, tr0_11, 0x20);\
    H04 = _mm256_permute2x128_si256(tr0_04, tr0_12, 0x20);\
    H05 = _mm256_permute2x128_si256(tr0_05, tr0_13, 0x20);\
    H06 = _mm256_permute2x128_si256(tr0_06, tr0_14, 0x20);\
    H07 = _mm256_permute2x128_si256(tr0_07, tr0_15, 0x20);\
    H08 = _mm256_permute2x128_si256(tr0_00, tr0_08, 0x31);\
    H09 = _mm256_permute2x128_si256(tr0_01, tr0_09, 0x31);\
    H10 = _mm256_permute2x128_si256(tr0_02, tr0_10, 0x31);\
    H11 = _mm256_permute2x128_si256(tr0_03, tr0_11, 0x31);\
    H12 = _mm256_permute2x128_si256(tr0_04, tr0_12, 0x31);\
    H13 = _mm256_permute2x128_si256(tr0_05, tr0_13, 0x31);\
    H14 = _mm256_permute2x128_si256(tr0_06, tr0_14, 0x31);\
    H15 = _mm256_permute2x128_si256(tr0_07, tr0_15, 0x31);\
 


/* ---------------------------------------------------------------------------
 * transpose 8x16(矩阵转置)
 */
#define TRANSPOSE_8x16_16BIT(A_00, A_01, A_02, A_03, A_04, A_05, A_06, A_07, H_00, H_01,H_02, H_03, H_04, H_05, H_06, H_07, H_08, H_09, H_10, H_11, H_12, H_13, H_14, H_15) \
    tr0_00 = _mm256_unpacklo_epi16(A_00, A_01); \
    tr0_01 = _mm256_unpacklo_epi16(A_02, A_03); \
    tr0_02 = _mm256_unpackhi_epi16(A_00, A_01); \
    tr0_03 = _mm256_unpackhi_epi16(A_02, A_03); \
    tr0_04 = _mm256_unpacklo_epi16(A_04, A_05); \
    tr0_05 = _mm256_unpacklo_epi16(A_06, A_07); \
    tr0_06 = _mm256_unpackhi_epi16(A_04, A_05); \
    tr0_07 = _mm256_unpackhi_epi16(A_06, A_07); \
    tr1_00 = _mm256_unpacklo_epi32(tr0_00, tr0_01); \
    tr1_01 = _mm256_unpacklo_epi32(tr0_02, tr0_03); \
    tr1_02 = _mm256_unpackhi_epi32(tr0_00, tr0_01); \
    tr1_03 = _mm256_unpackhi_epi32(tr0_02, tr0_03); \
    tr1_04 = _mm256_unpacklo_epi32(tr0_04, tr0_05); \
    tr1_05 = _mm256_unpacklo_epi32(tr0_06, tr0_07); \
    tr1_06 = _mm256_unpackhi_epi32(tr0_04, tr0_05); \
    tr1_07 = _mm256_unpackhi_epi32(tr0_06, tr0_07); \
    tr0_00 = _mm256_unpacklo_epi64(tr1_00, tr1_04); \
    tr0_01 = _mm256_unpackhi_epi64(tr1_00, tr1_04); \
    tr0_02 = _mm256_unpacklo_epi64(tr1_02, tr1_06); \
    tr0_03 = _mm256_unpackhi_epi64(tr1_02, tr1_06); \
    tr0_04 = _mm256_unpacklo_epi64(tr1_01, tr1_05); \
    tr0_05 = _mm256_unpackhi_epi64(tr1_01, tr1_05); \
    tr0_06 = _mm256_unpacklo_epi64(tr1_03, tr1_07); \
    tr0_07 = _mm256_unpackhi_epi64(tr1_03, tr1_07); \
    H_00 = _mm256_extracti128_si256(tr0_00, 0);\
    H_01 = _mm256_extracti128_si256(tr0_01, 0);\
    H_02 = _mm256_extracti128_si256(tr0_02, 0);\
    H_03 = _mm256_extracti128_si256(tr0_03, 0);\
    H_04 = _mm256_extracti128_si256(tr0_04, 0);\
    H_05 = _mm256_extracti128_si256(tr0_05, 0);\
    H_06 = _mm256_extracti128_si256(tr0_06, 0);\
    H_07 = _mm256_extracti128_si256(tr0_07, 0);\
    H_08 = _mm256_extracti128_si256(tr0_00, 1);\
    H_09 = _mm256_extracti128_si256(tr0_01, 1);\
    H_10 = _mm256_extracti128_si256(tr0_02, 1);\
    H_11 = _mm256_extracti128_si256(tr0_03, 1);\
    H_12 = _mm256_extracti128_si256(tr0_04, 1);\
    H_13 = _mm256_extracti128_si256(tr0_05, 1);\
    H_14 = _mm256_extracti128_si256(tr0_06, 1);\
    H_15 = _mm256_extracti128_si256(tr0_07, 1);\
 
/* ---------------------------------------------------------------------------
 */
static void wavelet_16x64_avx2(coeff_t *coeff)
{
    //按列 16*64
    __m256i V00, V01, V02, V03, V04, V05, V06, V07, V08, V09, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29, V30, V31, V32, V33, V34, V35, V36, V37, V38, V39, V40, V41, V42, V43, V44, V45, V46, V47, V48, V49, V50, V51, V52, V53, V54, V55, V56, V57, V58, V59, V60, V61, V62, V63;

    //按行 64*16
    __m256i T00[4], T01[4], T02[4], T03[4], T04[4], T05[4], T06[4], T07[4], T08[4], T09[4], T10[4], T11[4], T12[4], T13[4], T14[4], T15[4];

    //临时
    __m128i B00, B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20, B21, B22, B23, B24, B25, B26, B27, B28, B29, B30, B31;
    __m128i B32, B33, B34, B35, B36, B37, B38, B39, B40, B41, B42, B43, B44, B45, B46, B47, B48, B49, B50, B51, B52, B53, B54, B55, B56, B57, B58, B59, B60, B61, B62, B63;

    __m256i tr0_00, tr0_01, tr0_02, tr0_03, tr0_04, tr0_05, tr0_06, tr0_07, tr0_08, tr0_09, tr0_10, tr0_11, tr0_12, tr0_13, tr0_14, tr0_15;
    __m256i tr1_00, tr1_01, tr1_02, tr1_03, tr1_04, tr1_05, tr1_06, tr1_07, tr1_08, tr1_09, tr1_10, tr1_11, tr1_12, tr1_13, tr1_14, tr1_15;
    int i;
    __m128i mAddOffset1 = _mm_set1_epi16(1);
    __m256i mAddOffset2 = _mm256_set1_epi16(2);

    V00 = _mm256_load_si256((__m256i*)&coeff[16 * 0]);
    V01 = _mm256_load_si256((__m256i*)&coeff[16 * 1]);
    V02 = _mm256_load_si256((__m256i*)&coeff[16 * 2]);
    V03 = _mm256_load_si256((__m256i*)&coeff[16 * 3]);
    V04 = _mm256_load_si256((__m256i*)&coeff[16 * 4]);
    V05 = _mm256_load_si256((__m256i*)&coeff[16 * 5]);
    V06 = _mm256_load_si256((__m256i*)&coeff[16 * 6]);
    V07 = _mm256_load_si256((__m256i*)&coeff[16 * 7]);
    V08 = _mm256_load_si256((__m256i*)&coeff[16 * 8]);
    V09 = _mm256_load_si256((__m256i*)&coeff[16 * 9]);
    V10 = _mm256_load_si256((__m256i*)&coeff[16 * 10]);
    V11 = _mm256_load_si256((__m256i*)&coeff[16 * 11]);
    V12 = _mm256_load_si256((__m256i*)&coeff[16 * 12]);
    V13 = _mm256_load_si256((__m256i*)&coeff[16 * 13]);
    V14 = _mm256_load_si256((__m256i*)&coeff[16 * 14]);
    V15 = _mm256_load_si256((__m256i*)&coeff[16 * 15]);
    V16 = _mm256_load_si256((__m256i*)&coeff[16 * 16]);
    V17 = _mm256_load_si256((__m256i*)&coeff[16 * 17]);
    V18 = _mm256_load_si256((__m256i*)&coeff[16 * 18]);
    V19 = _mm256_load_si256((__m256i*)&coeff[16 * 19]);
    V20 = _mm256_load_si256((__m256i*)&coeff[16 * 20]);
    V21 = _mm256_load_si256((__m256i*)&coeff[16 * 21]);
    V22 = _mm256_load_si256((__m256i*)&coeff[16 * 22]);
    V23 = _mm256_load_si256((__m256i*)&coeff[16 * 23]);
    V24 = _mm256_load_si256((__m256i*)&coeff[16 * 24]);
    V25 = _mm256_load_si256((__m256i*)&coeff[16 * 25]);
    V26 = _mm256_load_si256((__m256i*)&coeff[16 * 26]);
    V27 = _mm256_load_si256((__m256i*)&coeff[16 * 27]);
    V28 = _mm256_load_si256((__m256i*)&coeff[16 * 28]);
    V29 = _mm256_load_si256((__m256i*)&coeff[16 * 29]);
    V30 = _mm256_load_si256((__m256i*)&coeff[16 * 30]);
    V31 = _mm256_load_si256((__m256i*)&coeff[16 * 31]);
    V32 = _mm256_load_si256((__m256i*)&coeff[16 * 32]);
    V33 = _mm256_load_si256((__m256i*)&coeff[16 * 33]);
    V34 = _mm256_load_si256((__m256i*)&coeff[16 * 34]);
    V35 = _mm256_load_si256((__m256i*)&coeff[16 * 35]);
    V36 = _mm256_load_si256((__m256i*)&coeff[16 * 36]);
    V37 = _mm256_load_si256((__m256i*)&coeff[16 * 37]);
    V38 = _mm256_load_si256((__m256i*)&coeff[16 * 38]);
    V39 = _mm256_load_si256((__m256i*)&coeff[16 * 39]);
    V40 = _mm256_load_si256((__m256i*)&coeff[16 * 40]);
    V41 = _mm256_load_si256((__m256i*)&coeff[16 * 41]);
    V42 = _mm256_load_si256((__m256i*)&coeff[16 * 42]);
    V43 = _mm256_load_si256((__m256i*)&coeff[16 * 43]);
    V44 = _mm256_load_si256((__m256i*)&coeff[16 * 44]);
    V45 = _mm256_load_si256((__m256i*)&coeff[16 * 45]);
    V46 = _mm256_load_si256((__m256i*)&coeff[16 * 46]);
    V47 = _mm256_load_si256((__m256i*)&coeff[16 * 47]);
    V48 = _mm256_load_si256((__m256i*)&coeff[16 * 48]);
    V49 = _mm256_load_si256((__m256i*)&coeff[16 * 49]);
    V50 = _mm256_load_si256((__m256i*)&coeff[16 * 50]);
    V51 = _mm256_load_si256((__m256i*)&coeff[16 * 51]);
    V52 = _mm256_load_si256((__m256i*)&coeff[16 * 52]);
    V53 = _mm256_load_si256((__m256i*)&coeff[16 * 53]);
    V54 = _mm256_load_si256((__m256i*)&coeff[16 * 54]);
    V55 = _mm256_load_si256((__m256i*)&coeff[16 * 55]);
    V56 = _mm256_load_si256((__m256i*)&coeff[16 * 56]);
    V57 = _mm256_load_si256((__m256i*)&coeff[16 * 57]);
    V58 = _mm256_load_si256((__m256i*)&coeff[16 * 58]);
    V59 = _mm256_load_si256((__m256i*)&coeff[16 * 59]);
    V60 = _mm256_load_si256((__m256i*)&coeff[16 * 60]);
    V61 = _mm256_load_si256((__m256i*)&coeff[16 * 61]);
    V62 = _mm256_load_si256((__m256i*)&coeff[16 * 62]);
    V63 = _mm256_load_si256((__m256i*)&coeff[16 * 63]);

    TRANSPOSE_16x16_16BIT(V00, V01, V02, V03, V04, V05, V06, V07, V08, V09, V10, V11, V12, V13, V14, V15, T00[0], T01[0], T02[0], T03[0], T04[0], T05[0], T06[0], T07[0], T08[0], T09[0], T10[0], T11[0], T12[0], T13[0], T14[0], T15[0]);
    TRANSPOSE_16x16_16BIT(V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29, V30, V31, T00[1], T01[1], T02[1], T03[1], T04[1], T05[1], T06[1], T07[1], T08[1], T09[1], T10[1], T11[1], T12[1], T13[1], T14[1], T15[1]);
    TRANSPOSE_16x16_16BIT(V32, V33, V34, V35, V36, V37, V38, V39, V40, V41, V42, V43, V44, V45, V46, V47, T00[2], T01[2], T02[2], T03[2], T04[2], T05[2], T06[2], T07[2], T08[2], T09[2], T10[2], T11[2], T12[2], T13[2], T14[2], T15[2]);
    TRANSPOSE_16x16_16BIT(V48, V49, V50, V51, V52, V53, V54, V55, V56, V57, V58, V59, V60, V61, V62, V63, T00[3], T01[3], T02[3], T03[3], T04[3], T05[3], T06[3], T07[3], T08[3], T09[3], T10[3], T11[3], T12[3], T13[3], T14[3], T15[3]);


    /* step 1: horizontal transform */

    // pExt[x] -= (pExt[x - 1] + pExt[x + 1]) >> 1;

    for (i = 0; i < 4; i++) {
        T01[i] = _mm256_sub_epi16(T01[i], _mm256_srai_epi16(_mm256_add_epi16(T00[i], T02[i]), 1));
        T03[i] = _mm256_sub_epi16(T03[i], _mm256_srai_epi16(_mm256_add_epi16(T02[i], T04[i]), 1));
        T05[i] = _mm256_sub_epi16(T05[i], _mm256_srai_epi16(_mm256_add_epi16(T04[i], T06[i]), 1));
        T07[i] = _mm256_sub_epi16(T07[i], _mm256_srai_epi16(_mm256_add_epi16(T06[i], T08[i]), 1));

        T09[i] = _mm256_sub_epi16(T09[i], _mm256_srai_epi16(_mm256_add_epi16(T08[i], T10[i]), 1));
        T11[i] = _mm256_sub_epi16(T11[i], _mm256_srai_epi16(_mm256_add_epi16(T10[i], T12[i]), 1));
        T13[i] = _mm256_sub_epi16(T13[i], _mm256_srai_epi16(_mm256_add_epi16(T12[i], T14[i]), 1));
        T15[i] = _mm256_sub_epi16(T15[i], _mm256_srai_epi16(_mm256_add_epi16(T14[i], T14[i]), 1));
    }

    for (i = 0; i < 4; i++) {
        T00[i] = _mm256_add_epi16(T00[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T01[i], T01[i]), mAddOffset2), 2));
        T02[i] = _mm256_add_epi16(T02[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T01[i], T03[i]), mAddOffset2), 2));
        T04[i] = _mm256_add_epi16(T04[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T03[i], T05[i]), mAddOffset2), 2));
        T06[i] = _mm256_add_epi16(T06[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T05[i], T07[i]), mAddOffset2), 2));

        T08[i] = _mm256_add_epi16(T08[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T07[i], T09[i]), mAddOffset2), 2));
        T10[i] = _mm256_add_epi16(T10[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T09[i], T11[i]), mAddOffset2), 2));
        T12[i] = _mm256_add_epi16(T12[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T11[i], T13[i]), mAddOffset2), 2));
        T14[i] = _mm256_add_epi16(T14[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(T13[i], T15[i]), mAddOffset2), 2));
    }

    /* step 2: vertical transform */
    /* copy 转置*/
    TRANSPOSE_8x16_16BIT(T00[0], T02[0], T04[0], T06[0], T08[0], T10[0], T12[0], T14[0], B00, B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B13, B14, B15);
    TRANSPOSE_8x16_16BIT(T00[1], T02[1], T04[1], T06[1], T08[1], T10[1], T12[1], T14[1], B16, B17, B18, B19, B20, B21, B22, B23, B24, B25, B26, B27, B28, B29, B30, B31);
    TRANSPOSE_8x16_16BIT(T00[2], T02[2], T04[2], T06[2], T08[2], T10[2], T12[2], T14[2], B32, B33, B34, B35, B36, B37, B38, B39, B40, B41, B42, B43, B44, B45, B46, B47);
    TRANSPOSE_8x16_16BIT(T00[3], T02[3], T04[3], T06[3], T08[3], T10[3], T12[3], T14[3], B48, B49, B50, B51, B52, B53, B54, B55, B56, B57, B58, B59, B60, B61, B62, B63);

    //pExt[y] -= (pExt[y - 1] + pExt[y + 1]) >> 1;

    B01 = _mm_sub_epi16(B01, _mm_srai_epi16(_mm_add_epi16(B00, B02), 1));
    B03 = _mm_sub_epi16(B03, _mm_srai_epi16(_mm_add_epi16(B02, B04), 1));
    B05 = _mm_sub_epi16(B05, _mm_srai_epi16(_mm_add_epi16(B04, B06), 1));
    B07 = _mm_sub_epi16(B07, _mm_srai_epi16(_mm_add_epi16(B06, B08), 1));
    B09 = _mm_sub_epi16(B09, _mm_srai_epi16(_mm_add_epi16(B08, B10), 1));
    B11 = _mm_sub_epi16(B11, _mm_srai_epi16(_mm_add_epi16(B10, B12), 1));
    B13 = _mm_sub_epi16(B13, _mm_srai_epi16(_mm_add_epi16(B12, B14), 1));
    B15 = _mm_sub_epi16(B15, _mm_srai_epi16(_mm_add_epi16(B14, B16), 1));
    B17 = _mm_sub_epi16(B17, _mm_srai_epi16(_mm_add_epi16(B16, B18), 1));
    B19 = _mm_sub_epi16(B19, _mm_srai_epi16(_mm_add_epi16(B18, B20), 1));
    B21 = _mm_sub_epi16(B21, _mm_srai_epi16(_mm_add_epi16(B20, B22), 1));
    B23 = _mm_sub_epi16(B23, _mm_srai_epi16(_mm_add_epi16(B22, B24), 1));
    B25 = _mm_sub_epi16(B25, _mm_srai_epi16(_mm_add_epi16(B24, B26), 1));
    B27 = _mm_sub_epi16(B27, _mm_srai_epi16(_mm_add_epi16(B26, B28), 1));
    B29 = _mm_sub_epi16(B29, _mm_srai_epi16(_mm_add_epi16(B28, B30), 1));
    B31 = _mm_sub_epi16(B31, _mm_srai_epi16(_mm_add_epi16(B30, B32), 1));

    B33 = _mm_sub_epi16(B33, _mm_srai_epi16(_mm_add_epi16(B32, B34), 1));
    B35 = _mm_sub_epi16(B35, _mm_srai_epi16(_mm_add_epi16(B34, B36), 1));
    B37 = _mm_sub_epi16(B37, _mm_srai_epi16(_mm_add_epi16(B36, B38), 1));
    B39 = _mm_sub_epi16(B39, _mm_srai_epi16(_mm_add_epi16(B38, B40), 1));
    B41 = _mm_sub_epi16(B41, _mm_srai_epi16(_mm_add_epi16(B40, B42), 1));
    B43 = _mm_sub_epi16(B43, _mm_srai_epi16(_mm_add_epi16(B42, B44), 1));
    B45 = _mm_sub_epi16(B45, _mm_srai_epi16(_mm_add_epi16(B44, B46), 1));
    B47 = _mm_sub_epi16(B47, _mm_srai_epi16(_mm_add_epi16(B46, B48), 1));
    B49 = _mm_sub_epi16(B49, _mm_srai_epi16(_mm_add_epi16(B48, B50), 1));
    B51 = _mm_sub_epi16(B51, _mm_srai_epi16(_mm_add_epi16(B50, B52), 1));
    B53 = _mm_sub_epi16(B53, _mm_srai_epi16(_mm_add_epi16(B52, B54), 1));
    B55 = _mm_sub_epi16(B55, _mm_srai_epi16(_mm_add_epi16(B54, B56), 1));
    B57 = _mm_sub_epi16(B57, _mm_srai_epi16(_mm_add_epi16(B56, B58), 1));
    B59 = _mm_sub_epi16(B59, _mm_srai_epi16(_mm_add_epi16(B58, B60), 1));
    B61 = _mm_sub_epi16(B61, _mm_srai_epi16(_mm_add_epi16(B60, B62), 1));
    B63 = _mm_sub_epi16(B63, _mm_srai_epi16(_mm_add_epi16(B62, B62), 1));

    //pExt[y] = (pExt[y] << 1) + ((pExt[y - 1] + pExt[y + 1] + 1) >> 1);

    B00 = _mm_add_epi16(_mm_slli_epi16(B00, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B01, B01), mAddOffset1), 1));
    B02 = _mm_add_epi16(_mm_slli_epi16(B02, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B01, B03), mAddOffset1), 1));
    B04 = _mm_add_epi16(_mm_slli_epi16(B04, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B03, B05), mAddOffset1), 1));
    B06 = _mm_add_epi16(_mm_slli_epi16(B06, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B05, B07), mAddOffset1), 1));
    B08 = _mm_add_epi16(_mm_slli_epi16(B08, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B07, B09), mAddOffset1), 1));
    B10 = _mm_add_epi16(_mm_slli_epi16(B10, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B09, B11), mAddOffset1), 1));
    B12 = _mm_add_epi16(_mm_slli_epi16(B12, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B11, B13), mAddOffset1), 1));
    B14 = _mm_add_epi16(_mm_slli_epi16(B14, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B13, B15), mAddOffset1), 1));
    B16 = _mm_add_epi16(_mm_slli_epi16(B16, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B15, B17), mAddOffset1), 1));
    B18 = _mm_add_epi16(_mm_slli_epi16(B18, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B17, B19), mAddOffset1), 1));
    B20 = _mm_add_epi16(_mm_slli_epi16(B20, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B19, B21), mAddOffset1), 1));
    B22 = _mm_add_epi16(_mm_slli_epi16(B22, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B21, B23), mAddOffset1), 1));
    B24 = _mm_add_epi16(_mm_slli_epi16(B24, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B23, B25), mAddOffset1), 1));
    B26 = _mm_add_epi16(_mm_slli_epi16(B26, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B25, B27), mAddOffset1), 1));
    B28 = _mm_add_epi16(_mm_slli_epi16(B28, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B27, B29), mAddOffset1), 1));
    B30 = _mm_add_epi16(_mm_slli_epi16(B30, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B29, B31), mAddOffset1), 1));

    B32 = _mm_add_epi16(_mm_slli_epi16(B32, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B31, B33), mAddOffset1), 1));
    B34 = _mm_add_epi16(_mm_slli_epi16(B34, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B33, B35), mAddOffset1), 1));
    B36 = _mm_add_epi16(_mm_slli_epi16(B36, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B35, B37), mAddOffset1), 1));
    B38 = _mm_add_epi16(_mm_slli_epi16(B38, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B37, B39), mAddOffset1), 1));
    B40 = _mm_add_epi16(_mm_slli_epi16(B40, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B39, B41), mAddOffset1), 1));
    B42 = _mm_add_epi16(_mm_slli_epi16(B42, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B41, B43), mAddOffset1), 1));
    B44 = _mm_add_epi16(_mm_slli_epi16(B44, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B43, B45), mAddOffset1), 1));
    B46 = _mm_add_epi16(_mm_slli_epi16(B46, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B45, B47), mAddOffset1), 1));
    B48 = _mm_add_epi16(_mm_slli_epi16(B48, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B47, B49), mAddOffset1), 1));
    B50 = _mm_add_epi16(_mm_slli_epi16(B50, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B49, B51), mAddOffset1), 1));
    B52 = _mm_add_epi16(_mm_slli_epi16(B52, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B51, B53), mAddOffset1), 1));
    B54 = _mm_add_epi16(_mm_slli_epi16(B54, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B53, B55), mAddOffset1), 1));
    B56 = _mm_add_epi16(_mm_slli_epi16(B56, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B55, B57), mAddOffset1), 1));
    B58 = _mm_add_epi16(_mm_slli_epi16(B58, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B57, B59), mAddOffset1), 1));
    B60 = _mm_add_epi16(_mm_slli_epi16(B60, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B59, B61), mAddOffset1), 1));
    B62 = _mm_add_epi16(_mm_slli_epi16(B62, 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(B61, B63), mAddOffset1), 1));

    //STORE
    _mm_store_si128((__m128i*)&coeff[8 * 0], B00);
    _mm_store_si128((__m128i*)&coeff[8 * 1], B02);
    _mm_store_si128((__m128i*)&coeff[8 * 2], B04);
    _mm_store_si128((__m128i*)&coeff[8 * 3], B06);
    _mm_store_si128((__m128i*)&coeff[8 * 4], B08);
    _mm_store_si128((__m128i*)&coeff[8 * 5], B10);
    _mm_store_si128((__m128i*)&coeff[8 * 6], B12);
    _mm_store_si128((__m128i*)&coeff[8 * 7], B14);

    _mm_store_si128((__m128i*)&coeff[8 * 8], B16);
    _mm_store_si128((__m128i*)&coeff[8 * 9], B18);
    _mm_store_si128((__m128i*)&coeff[8 * 10], B20);
    _mm_store_si128((__m128i*)&coeff[8 * 11], B22);
    _mm_store_si128((__m128i*)&coeff[8 * 12], B24);
    _mm_store_si128((__m128i*)&coeff[8 * 13], B26);
    _mm_store_si128((__m128i*)&coeff[8 * 14], B28);
    _mm_store_si128((__m128i*)&coeff[8 * 15], B30);

    _mm_store_si128((__m128i*)&coeff[8 * 16], B32);
    _mm_store_si128((__m128i*)&coeff[8 * 17], B34);
    _mm_store_si128((__m128i*)&coeff[8 * 18], B36);
    _mm_store_si128((__m128i*)&coeff[8 * 19], B38);
    _mm_store_si128((__m128i*)&coeff[8 * 20], B40);
    _mm_store_si128((__m128i*)&coeff[8 * 21], B42);
    _mm_store_si128((__m128i*)&coeff[8 * 22], B44);
    _mm_store_si128((__m128i*)&coeff[8 * 23], B46);

    _mm_store_si128((__m128i*)&coeff[8 * 24], B48);
    _mm_store_si128((__m128i*)&coeff[8 * 25], B50);
    _mm_store_si128((__m128i*)&coeff[8 * 26], B52);
    _mm_store_si128((__m128i*)&coeff[8 * 27], B54);
    _mm_store_si128((__m128i*)&coeff[8 * 28], B56);
    _mm_store_si128((__m128i*)&coeff[8 * 29], B58);
    _mm_store_si128((__m128i*)&coeff[8 * 30], B60);
    _mm_store_si128((__m128i*)&coeff[8 * 31], B62);
}

/* ---------------------------------------------------------------------------
 */
static void wavelet_64x16_avx2(coeff_t *coeff)
{
    //按列 16*64
    __m256i V00, V01, V02, V03, V04, V05, V06, V07, V08, V09, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29, V30, V31, V32, V33, V34, V35, V36, V37, V38, V39, V40, V41, V42, V43, V44, V45, V46, V47, V48, V49, V50, V51, V52, V53, V54, V55, V56, V57, V58, V59, V60, V61, V62, V63;

    //按行 64*16
    __m256i T00[4], T01[4], T02[4], T03[4], T04[4], T05[4], T06[4], T07[4], T08[4], T09[4], T10[4], T11[4], T12[4], T13[4], T14[4], T15[4];

    //临时 64*16
    __m256i A00[2], A01[2], A02[2], A03[2], A04[2], A05[2], A06[2], A07[2], A08[2], A09[2], A10[2], A11[2], A12[2], A13[2], A14[2], A15[2];

    __m256i tr0_00, tr0_01, tr0_02, tr0_03, tr0_04, tr0_05, tr0_06, tr0_07, tr0_08, tr0_09, tr0_10, tr0_11, tr0_12, tr0_13, tr0_14, tr0_15;
    __m256i tr1_00, tr1_01, tr1_02, tr1_03, tr1_04, tr1_05, tr1_06, tr1_07, tr1_08, tr1_09, tr1_10, tr1_11, tr1_12, tr1_13, tr1_14, tr1_15;
    int i;
    __m256i mAddOffset1 = _mm256_set1_epi16(1);
    __m256i mAddOffset2 = _mm256_set1_epi16(2);
    //load

    for (i = 0; i < 4; i++) {
        T00[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 0]));
        T01[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 1]));
        T02[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 2]));
        T03[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 3]));
        T04[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 4]));
        T05[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 5]));
        T06[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 6]));
        T07[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 7]));
        T08[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 8]));
        T09[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 9]));
        T10[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 10]));
        T11[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 11]));
        T12[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 12]));
        T13[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 13]));
        T14[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 14]));
        T15[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 15]));
    }

    TRANSPOSE_16x16_16BIT(T00[0], T01[0], T02[0], T03[0], T04[0], T05[0], T06[0], T07[0], T08[0], T09[0], T10[0], T11[0], T12[0], T13[0], T14[0], T15[0], V00, V01, V02, V03, V04, V05, V06, V07, V08, V09, V10, V11, V12, V13, V14, V15);
    TRANSPOSE_16x16_16BIT(T00[1], T01[1], T02[1], T03[1], T04[1], T05[1], T06[1], T07[1], T08[1], T09[1], T10[1], T11[1], T12[1], T13[1], T14[1], T15[1], V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29, V30, V31);
    TRANSPOSE_16x16_16BIT(T00[2], T01[2], T02[2], T03[2], T04[2], T05[2], T06[2], T07[2], T08[2], T09[2], T10[2], T11[2], T12[2], T13[2], T14[2], T15[2], V32, V33, V34, V35, V36, V37, V38, V39, V40, V41, V42, V43, V44, V45, V46, V47);
    TRANSPOSE_16x16_16BIT(T00[3], T01[3], T02[3], T03[3], T04[3], T05[3], T06[3], T07[3], T08[3], T09[3], T10[3], T11[3], T12[3], T13[3], T14[3], T15[3], V48, V49, V50, V51, V52, V53, V54, V55, V56, V57, V58, V59, V60, V61, V62, V63);

    //pExt[x] -= (pExt[x - 1] + pExt[x + 1]) >> 1;

    V01 = _mm256_sub_epi16(V01, _mm256_srai_epi16(_mm256_add_epi16(V00, V02), 1));
    V03 = _mm256_sub_epi16(V03, _mm256_srai_epi16(_mm256_add_epi16(V02, V04), 1));
    V05 = _mm256_sub_epi16(V05, _mm256_srai_epi16(_mm256_add_epi16(V04, V06), 1));
    V07 = _mm256_sub_epi16(V07, _mm256_srai_epi16(_mm256_add_epi16(V06, V08), 1));
    V09 = _mm256_sub_epi16(V09, _mm256_srai_epi16(_mm256_add_epi16(V08, V10), 1));
    V11 = _mm256_sub_epi16(V11, _mm256_srai_epi16(_mm256_add_epi16(V10, V12), 1));
    V13 = _mm256_sub_epi16(V13, _mm256_srai_epi16(_mm256_add_epi16(V12, V14), 1));
    V15 = _mm256_sub_epi16(V15, _mm256_srai_epi16(_mm256_add_epi16(V14, V16), 1));

    V17 = _mm256_sub_epi16(V17, _mm256_srai_epi16(_mm256_add_epi16(V16, V18), 1));
    V19 = _mm256_sub_epi16(V19, _mm256_srai_epi16(_mm256_add_epi16(V18, V20), 1));
    V21 = _mm256_sub_epi16(V21, _mm256_srai_epi16(_mm256_add_epi16(V20, V22), 1));
    V23 = _mm256_sub_epi16(V23, _mm256_srai_epi16(_mm256_add_epi16(V22, V24), 1));
    V25 = _mm256_sub_epi16(V25, _mm256_srai_epi16(_mm256_add_epi16(V24, V26), 1));
    V27 = _mm256_sub_epi16(V27, _mm256_srai_epi16(_mm256_add_epi16(V26, V28), 1));
    V29 = _mm256_sub_epi16(V29, _mm256_srai_epi16(_mm256_add_epi16(V28, V30), 1));
    V31 = _mm256_sub_epi16(V31, _mm256_srai_epi16(_mm256_add_epi16(V30, V32), 1));

    V33 = _mm256_sub_epi16(V33, _mm256_srai_epi16(_mm256_add_epi16(V32, V34), 1));
    V35 = _mm256_sub_epi16(V35, _mm256_srai_epi16(_mm256_add_epi16(V34, V36), 1));
    V37 = _mm256_sub_epi16(V37, _mm256_srai_epi16(_mm256_add_epi16(V36, V38), 1));
    V39 = _mm256_sub_epi16(V39, _mm256_srai_epi16(_mm256_add_epi16(V38, V40), 1));
    V41 = _mm256_sub_epi16(V41, _mm256_srai_epi16(_mm256_add_epi16(V40, V42), 1));
    V43 = _mm256_sub_epi16(V43, _mm256_srai_epi16(_mm256_add_epi16(V42, V44), 1));
    V45 = _mm256_sub_epi16(V45, _mm256_srai_epi16(_mm256_add_epi16(V44, V46), 1));
    V47 = _mm256_sub_epi16(V47, _mm256_srai_epi16(_mm256_add_epi16(V46, V48), 1));

    V49 = _mm256_sub_epi16(V49, _mm256_srai_epi16(_mm256_add_epi16(V48, V50), 1));
    V51 = _mm256_sub_epi16(V51, _mm256_srai_epi16(_mm256_add_epi16(V50, V52), 1));
    V53 = _mm256_sub_epi16(V53, _mm256_srai_epi16(_mm256_add_epi16(V52, V54), 1));
    V55 = _mm256_sub_epi16(V55, _mm256_srai_epi16(_mm256_add_epi16(V54, V56), 1));
    V57 = _mm256_sub_epi16(V57, _mm256_srai_epi16(_mm256_add_epi16(V56, V58), 1));
    V59 = _mm256_sub_epi16(V59, _mm256_srai_epi16(_mm256_add_epi16(V58, V60), 1));
    V61 = _mm256_sub_epi16(V61, _mm256_srai_epi16(_mm256_add_epi16(V60, V62), 1));
    V63 = _mm256_sub_epi16(V63, _mm256_srai_epi16(_mm256_add_epi16(V62, V62), 1));

    //pExt[x] += (pExt[x - 1] + pExt[x + 1] + 2) >> 2;

    V00 = _mm256_add_epi16(V00, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V01, V01), mAddOffset2), 2));
    V02 = _mm256_add_epi16(V02, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V01, V03), mAddOffset2), 2));
    V04 = _mm256_add_epi16(V04, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V03, V05), mAddOffset2), 2));
    V06 = _mm256_add_epi16(V06, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V05, V07), mAddOffset2), 2));
    V08 = _mm256_add_epi16(V08, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V07, V09), mAddOffset2), 2));
    V10 = _mm256_add_epi16(V10, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V09, V11), mAddOffset2), 2));
    V12 = _mm256_add_epi16(V12, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V11, V13), mAddOffset2), 2));
    V14 = _mm256_add_epi16(V14, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V13, V15), mAddOffset2), 2));

    V16 = _mm256_add_epi16(V16, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V15, V17), mAddOffset2), 2));
    V18 = _mm256_add_epi16(V18, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V17, V19), mAddOffset2), 2));
    V20 = _mm256_add_epi16(V20, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V19, V21), mAddOffset2), 2));
    V22 = _mm256_add_epi16(V22, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V21, V23), mAddOffset2), 2));
    V24 = _mm256_add_epi16(V24, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V23, V25), mAddOffset2), 2));
    V26 = _mm256_add_epi16(V26, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V25, V27), mAddOffset2), 2));
    V28 = _mm256_add_epi16(V28, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V27, V29), mAddOffset2), 2));
    V30 = _mm256_add_epi16(V30, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V29, V31), mAddOffset2), 2));

    V32 = _mm256_add_epi16(V32, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V31, V33), mAddOffset2), 2));
    V34 = _mm256_add_epi16(V34, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V33, V35), mAddOffset2), 2));
    V36 = _mm256_add_epi16(V36, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V35, V37), mAddOffset2), 2));
    V38 = _mm256_add_epi16(V38, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V37, V39), mAddOffset2), 2));
    V40 = _mm256_add_epi16(V40, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V39, V41), mAddOffset2), 2));
    V42 = _mm256_add_epi16(V42, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V41, V43), mAddOffset2), 2));
    V44 = _mm256_add_epi16(V44, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V43, V45), mAddOffset2), 2));
    V46 = _mm256_add_epi16(V46, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V45, V47), mAddOffset2), 2));

    V48 = _mm256_add_epi16(V48, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V47, V49), mAddOffset2), 2));
    V50 = _mm256_add_epi16(V50, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V49, V51), mAddOffset2), 2));
    V52 = _mm256_add_epi16(V52, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V51, V53), mAddOffset2), 2));
    V54 = _mm256_add_epi16(V54, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V53, V55), mAddOffset2), 2));
    V56 = _mm256_add_epi16(V56, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V55, V57), mAddOffset2), 2));
    V58 = _mm256_add_epi16(V58, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V57, V59), mAddOffset2), 2));
    V60 = _mm256_add_epi16(V60, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V59, V61), mAddOffset2), 2));
    V62 = _mm256_add_epi16(V62, _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V61, V63), mAddOffset2), 2));

    TRANSPOSE_16x16_16BIT(V00, V02, V04, V06, V08, V10, V12, V14, V16, V18, V20, V22, V24, V26, V28, V30, A00[0], A01[0], A02[0], A03[0], A04[0], A05[0], A06[0], A07[0], A08[0], A09[0], A10[0], A11[0], A12[0], A13[0], A14[0], A15[0]);
    TRANSPOSE_16x16_16BIT(V32, V34, V36, V38, V40, V42, V44, V46, V48, V50, V52, V54, V56, V58, V60, V62, A00[1], A01[1], A02[1], A03[1], A04[1], A05[1], A06[1], A07[1], A08[1], A09[1], A10[1], A11[1], A12[1], A13[1], A14[1], A15[1]);

    //pExt[y] -= (pExt[y - 1] + pExt[y + 1]) >> 1;
    for (i = 0; i < 2; i++) {
        A01[i] = _mm256_sub_epi16(A01[i], _mm256_srai_epi16(_mm256_add_epi16(A00[i], A02[i]), 1));
        A03[i] = _mm256_sub_epi16(A03[i], _mm256_srai_epi16(_mm256_add_epi16(A02[i], A04[i]), 1));
        A05[i] = _mm256_sub_epi16(A05[i], _mm256_srai_epi16(_mm256_add_epi16(A04[i], A06[i]), 1));
        A07[i] = _mm256_sub_epi16(A07[i], _mm256_srai_epi16(_mm256_add_epi16(A06[i], A08[i]), 1));
        A09[i] = _mm256_sub_epi16(A09[i], _mm256_srai_epi16(_mm256_add_epi16(A08[i], A10[i]), 1));
        A11[i] = _mm256_sub_epi16(A11[i], _mm256_srai_epi16(_mm256_add_epi16(A10[i], A12[i]), 1));
        A13[i] = _mm256_sub_epi16(A13[i], _mm256_srai_epi16(_mm256_add_epi16(A12[i], A14[i]), 1));
        A15[i] = _mm256_sub_epi16(A15[i], _mm256_srai_epi16(_mm256_add_epi16(A14[i], A14[i]), 1));
    }

    //pExt[y] = (pExt[y] << 1) + ((pExt[y - 1] + pExt[y + 1] + 1) >> 1);
    for (i = 0; i < 2; i++) {
        A00[i] = _mm256_add_epi16(_mm256_slli_epi16(A00[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A01[i], A01[i]), mAddOffset1), 1));
        A02[i] = _mm256_add_epi16(_mm256_slli_epi16(A02[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A01[i], A03[i]), mAddOffset1), 1));
        A04[i] = _mm256_add_epi16(_mm256_slli_epi16(A04[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A03[i], A05[i]), mAddOffset1), 1));
        A06[i] = _mm256_add_epi16(_mm256_slli_epi16(A06[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A05[i], A07[i]), mAddOffset1), 1));
        A08[i] = _mm256_add_epi16(_mm256_slli_epi16(A08[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A07[i], A09[i]), mAddOffset1), 1));
        A10[i] = _mm256_add_epi16(_mm256_slli_epi16(A10[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A09[i], A11[i]), mAddOffset1), 1));
        A12[i] = _mm256_add_epi16(_mm256_slli_epi16(A12[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A11[i], A13[i]), mAddOffset1), 1));
        A14[i] = _mm256_add_epi16(_mm256_slli_epi16(A14[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A13[i], A15[i]), mAddOffset1), 1));
    }

    //Store
    for (i = 0; i < 2; i++) {
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 0], A00[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 1], A02[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 2], A04[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 3], A06[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 4], A08[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 5], A10[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 6], A12[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 7], A14[i]);
    }
}


/* ---------------------------------------------------------------------------
 */
static void wavelet_64x64_avx2(coeff_t *coeff)
{
    //按列 16*64
    __m256i V00[4], V01[4], V02[4], V03[4], V04[4], V05[4], V06[4], V07[4], V08[4], V09[4], V10[4], V11[4], V12[4], V13[4], V14[4], V15[4], V16[4], V17[4], V18[4], V19[4], V20[4], V21[4], V22[4], V23[4], V24[4], V25[4], V26[4], V27[4], V28[4], V29[4], V30[4], V31[4], V32[4], V33[4], V34[4], V35[4], V36[4], V37[4], V38[4], V39[4], V40[4], V41[4], V42[4], V43[4], V44[4], V45[4], V46[4], V47[4], V48[4], V49[4], V50[4], V51[4], V52[4], V53[4], V54[4], V55[4], V56[4], V57[4], V58[4], V59[4], V60[4], V61[4], V62[4], V63[4];

    //按行 64*64
    __m256i T00[4], T01[4], T02[4], T03[4], T04[4], T05[4], T06[4], T07[4], T08[4], T09[4], T10[4], T11[4], T12[4], T13[4], T14[4], T15[4], T16[4], T17[4], T18[4], T19[4], T20[4], T21[4], T22[4], T23[4], T24[4], T25[4], T26[4], T27[4], T28[4], T29[4], T30[4], T31[4], T32[4], T33[4], T34[4], T35[4], T36[4], T37[4], T38[4], T39[4], T40[4], T41[4], T42[4], T43[4], T44[4], T45[4], T46[4], T47[4], T48[4], T49[4], T50[4], T51[4], T52[4], T53[4], T54[4], T55[4], T56[4], T57[4], T58[4], T59[4], T60[4], T61[4], T62[4], T63[4];

    //临时 32*64
    __m256i A00[2], A01[2], A02[2], A03[2], A04[2], A05[2], A06[2], A07[2], A08[2], A09[2], A10[2], A11[2], A12[2], A13[2], A14[2], A15[2], A16[2], A17[2], A18[2], A19[2], A20[2], A21[2], A22[2], A23[2], A24[2], A25[2], A26[2], A27[2], A28[2], A29[2], A30[2], A31[2], A32[2], A33[2], A34[2], A35[2], A36[2], A37[2], A38[2], A39[2], A40[2], A41[2], A42[2], A43[2], A44[2], A45[2], A46[2], A47[2], A48[2], A49[2], A50[2], A51[2], A52[2], A53[2], A54[2], A55[2], A56[2], A57[2], A58[2], A59[2], A60[2], A61[2], A62[2], A63[2];

    __m256i tr0_00, tr0_01, tr0_02, tr0_03, tr0_04, tr0_05, tr0_06, tr0_07, tr0_08, tr0_09, tr0_10, tr0_11, tr0_12, tr0_13, tr0_14, tr0_15;
    __m256i tr1_00, tr1_01, tr1_02, tr1_03, tr1_04, tr1_05, tr1_06, tr1_07, tr1_08, tr1_09, tr1_10, tr1_11, tr1_12, tr1_13, tr1_14, tr1_15;
    int i;
    __m256i mAddOffset1 = _mm256_set1_epi16(1);
    __m256i mAddOffset2 = _mm256_set1_epi16(2);
    //load

    for (i = 0; i < 4; i++) {
        T00[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 0]));
        T01[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 1]));
        T02[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 2]));
        T03[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 3]));
        T04[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 4]));
        T05[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 5]));
        T06[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 6]));
        T07[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 7]));

        T08[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 8]));
        T09[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 9]));
        T10[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 10]));
        T11[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 11]));
        T12[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 12]));
        T13[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 13]));
        T14[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 14]));
        T15[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 15]));

        T16[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 16]));
        T17[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 17]));
        T18[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 18]));
        T19[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 19]));
        T20[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 20]));
        T21[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 21]));
        T22[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 22]));
        T23[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 23]));

        T24[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 24]));
        T25[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 25]));
        T26[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 26]));
        T27[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 27]));
        T28[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 28]));
        T29[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 29]));
        T30[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 30]));
        T31[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 31]));

        T32[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 32]));
        T33[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 33]));
        T34[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 34]));
        T35[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 35]));
        T36[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 36]));
        T37[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 37]));
        T38[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 38]));
        T39[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 39]));

        T40[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 40]));
        T41[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 41]));
        T42[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 42]));
        T43[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 43]));
        T44[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 44]));
        T45[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 45]));
        T46[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 46]));
        T47[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 47]));

        T48[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 48]));
        T49[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 49]));
        T50[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 50]));
        T51[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 51]));
        T52[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 52]));
        T53[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 53]));
        T54[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 54]));
        T55[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 55]));

        T56[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 56]));
        T57[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 57]));
        T58[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 58]));
        T59[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 59]));
        T60[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 60]));
        T61[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 61]));
        T62[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 62]));
        T63[i] = _mm256_load_si256((__m256i const *)((__m128i*)&coeff[16 * i + 64 * 63]));
    }

    //0-15行转置
    TRANSPOSE_16x16_16BIT(T00[0], T01[0], T02[0], T03[0], T04[0], T05[0], T06[0], T07[0], T08[0], T09[0], T10[0], T11[0], T12[0], T13[0], T14[0], T15[0], V00[0], V01[0], V02[0], V03[0], V04[0], V05[0], V06[0], V07[0], V08[0], V09[0], V10[0], V11[0], V12[0], V13[0], V14[0], V15[0]);
    TRANSPOSE_16x16_16BIT(T00[1], T01[1], T02[1], T03[1], T04[1], T05[1], T06[1], T07[1], T08[1], T09[1], T10[1], T11[1], T12[1], T13[1], T14[1], T15[1], V16[0], V17[0], V18[0], V19[0], V20[0], V21[0], V22[0], V23[0], V24[0], V25[0], V26[0], V27[0], V28[0], V29[0], V30[0], V31[0]);
    TRANSPOSE_16x16_16BIT(T00[2], T01[2], T02[2], T03[2], T04[2], T05[2], T06[2], T07[2], T08[2], T09[2], T10[2], T11[2], T12[2], T13[2], T14[2], T15[2], V32[0], V33[0], V34[0], V35[0], V36[0], V37[0], V38[0], V39[0], V40[0], V41[0], V42[0], V43[0], V44[0], V45[0], V46[0], V47[0]);
    TRANSPOSE_16x16_16BIT(T00[3], T01[3], T02[3], T03[3], T04[3], T05[3], T06[3], T07[3], T08[3], T09[3], T10[3], T11[3], T12[3], T13[3], T14[3], T15[3], V48[0], V49[0], V50[0], V51[0], V52[0], V53[0], V54[0], V55[0], V56[0], V57[0], V58[0], V59[0], V60[0], V61[0], V62[0], V63[0]);

    //16-31行转置
    TRANSPOSE_16x16_16BIT(T16[0], T17[0], T18[0], T19[0], T20[0], T21[0], T22[0], T23[0], T24[0], T25[0], T26[0], T27[0], T28[0], T29[0], T30[0], T31[0], V00[1], V01[1], V02[1], V03[1], V04[1], V05[1], V06[1], V07[1], V08[1], V09[1], V10[1], V11[1], V12[1], V13[1], V14[1], V15[1]);
    TRANSPOSE_16x16_16BIT(T16[1], T17[1], T18[1], T19[1], T20[1], T21[1], T22[1], T23[1], T24[1], T25[1], T26[1], T27[1], T28[1], T29[1], T30[1], T31[1], V16[1], V17[1], V18[1], V19[1], V20[1], V21[1], V22[1], V23[1], V24[1], V25[1], V26[1], V27[1], V28[1], V29[1], V30[1], V31[1]);
    TRANSPOSE_16x16_16BIT(T16[2], T17[2], T18[2], T19[2], T20[2], T21[2], T22[2], T23[2], T24[2], T25[2], T26[2], T27[2], T28[2], T29[2], T30[2], T31[2], V32[1], V33[1], V34[1], V35[1], V36[1], V37[1], V38[1], V39[1], V40[1], V41[1], V42[1], V43[1], V44[1], V45[1], V46[1], V47[1]);
    TRANSPOSE_16x16_16BIT(T16[3], T17[3], T18[3], T19[3], T20[3], T21[3], T22[3], T23[3], T24[3], T25[3], T26[3], T27[3], T28[3], T29[3], T30[3], T31[3], V48[1], V49[1], V50[1], V51[1], V52[1], V53[1], V54[1], V55[1], V56[1], V57[1], V58[1], V59[1], V60[1], V61[1], V62[1], V63[1]);

    //32-47行转置
    TRANSPOSE_16x16_16BIT(T32[0], T33[0], T34[0], T35[0], T36[0], T37[0], T38[0], T39[0], T40[0], T41[0], T42[0], T43[0], T44[0], T45[0], T46[0], T47[0], V00[2], V01[2], V02[2], V03[2], V04[2], V05[2], V06[2], V07[2], V08[2], V09[2], V10[2], V11[2], V12[2], V13[2], V14[2], V15[2]);
    TRANSPOSE_16x16_16BIT(T32[1], T33[1], T34[1], T35[1], T36[1], T37[1], T38[1], T39[1], T40[1], T41[1], T42[1], T43[1], T44[1], T45[1], T46[1], T47[1], V16[2], V17[2], V18[2], V19[2], V20[2], V21[2], V22[2], V23[2], V24[2], V25[2], V26[2], V27[2], V28[2], V29[2], V30[2], V31[2]);
    TRANSPOSE_16x16_16BIT(T32[2], T33[2], T34[2], T35[2], T36[2], T37[2], T38[2], T39[2], T40[2], T41[2], T42[2], T43[2], T44[2], T45[2], T46[2], T47[2], V32[2], V33[2], V34[2], V35[2], V36[2], V37[2], V38[2], V39[2], V40[2], V41[2], V42[2], V43[2], V44[2], V45[2], V46[2], V47[2]);
    TRANSPOSE_16x16_16BIT(T32[3], T33[3], T34[3], T35[3], T36[3], T37[3], T38[3], T39[3], T40[3], T41[3], T42[3], T43[3], T44[3], T45[3], T46[3], T47[3], V48[2], V49[2], V50[2], V51[2], V52[2], V53[2], V54[2], V55[2], V56[2], V57[2], V58[2], V59[2], V60[2], V61[2], V62[2], V63[2]);

    //48-63行转置
    TRANSPOSE_16x16_16BIT(T48[0], T49[0], T50[0], T51[0], T52[0], T53[0], T54[0], T55[0], T56[0], T57[0], T58[0], T59[0], T60[0], T61[0], T62[0], T63[0], V00[3], V01[3], V02[3], V03[3], V04[3], V05[3], V06[3], V07[3], V08[3], V09[3], V10[3], V11[3], V12[3], V13[3], V14[3], V15[3]);
    TRANSPOSE_16x16_16BIT(T48[1], T49[1], T50[1], T51[1], T52[1], T53[1], T54[1], T55[1], T56[1], T57[1], T58[1], T59[1], T60[1], T61[1], T62[1], T63[1], V16[3], V17[3], V18[3], V19[3], V20[3], V21[3], V22[3], V23[3], V24[3], V25[3], V26[3], V27[3], V28[3], V29[3], V30[3], V31[3]);
    TRANSPOSE_16x16_16BIT(T48[2], T49[2], T50[2], T51[2], T52[2], T53[2], T54[2], T55[2], T56[2], T57[2], T58[2], T59[2], T60[2], T61[2], T62[2], T63[2], V32[3], V33[3], V34[3], V35[3], V36[3], V37[3], V38[3], V39[3], V40[3], V41[3], V42[3], V43[3], V44[3], V45[3], V46[3], V47[3]);
    TRANSPOSE_16x16_16BIT(T48[3], T49[3], T50[3], T51[3], T52[3], T53[3], T54[3], T55[3], T56[3], T57[3], T58[3], T59[3], T60[3], T61[3], T62[3], T63[3], V48[3], V49[3], V50[3], V51[3], V52[3], V53[3], V54[3], V55[3], V56[3], V57[3], V58[3], V59[3], V60[3], V61[3], V62[3], V63[3]);


    //pExt[x] -= (pExt[x - 1] + pExt[x + 1]) >> 1;
    for (i = 0; i < 4; i++) {
        V01[i] = _mm256_sub_epi16(V01[i], _mm256_srai_epi16(_mm256_add_epi16(V00[i], V02[i]), 1));
        V03[i] = _mm256_sub_epi16(V03[i], _mm256_srai_epi16(_mm256_add_epi16(V02[i], V04[i]), 1));
        V05[i] = _mm256_sub_epi16(V05[i], _mm256_srai_epi16(_mm256_add_epi16(V04[i], V06[i]), 1));
        V07[i] = _mm256_sub_epi16(V07[i], _mm256_srai_epi16(_mm256_add_epi16(V06[i], V08[i]), 1));
        V09[i] = _mm256_sub_epi16(V09[i], _mm256_srai_epi16(_mm256_add_epi16(V08[i], V10[i]), 1));
        V11[i] = _mm256_sub_epi16(V11[i], _mm256_srai_epi16(_mm256_add_epi16(V10[i], V12[i]), 1));
        V13[i] = _mm256_sub_epi16(V13[i], _mm256_srai_epi16(_mm256_add_epi16(V12[i], V14[i]), 1));
        V15[i] = _mm256_sub_epi16(V15[i], _mm256_srai_epi16(_mm256_add_epi16(V14[i], V16[i]), 1));

        V17[i] = _mm256_sub_epi16(V17[i], _mm256_srai_epi16(_mm256_add_epi16(V16[i], V18[i]), 1));
        V19[i] = _mm256_sub_epi16(V19[i], _mm256_srai_epi16(_mm256_add_epi16(V18[i], V20[i]), 1));
        V21[i] = _mm256_sub_epi16(V21[i], _mm256_srai_epi16(_mm256_add_epi16(V20[i], V22[i]), 1));
        V23[i] = _mm256_sub_epi16(V23[i], _mm256_srai_epi16(_mm256_add_epi16(V22[i], V24[i]), 1));
        V25[i] = _mm256_sub_epi16(V25[i], _mm256_srai_epi16(_mm256_add_epi16(V24[i], V26[i]), 1));
        V27[i] = _mm256_sub_epi16(V27[i], _mm256_srai_epi16(_mm256_add_epi16(V26[i], V28[i]), 1));
        V29[i] = _mm256_sub_epi16(V29[i], _mm256_srai_epi16(_mm256_add_epi16(V28[i], V30[i]), 1));
        V31[i] = _mm256_sub_epi16(V31[i], _mm256_srai_epi16(_mm256_add_epi16(V30[i], V32[i]), 1));

        V33[i] = _mm256_sub_epi16(V33[i], _mm256_srai_epi16(_mm256_add_epi16(V32[i], V34[i]), 1));
        V35[i] = _mm256_sub_epi16(V35[i], _mm256_srai_epi16(_mm256_add_epi16(V34[i], V36[i]), 1));
        V37[i] = _mm256_sub_epi16(V37[i], _mm256_srai_epi16(_mm256_add_epi16(V36[i], V38[i]), 1));
        V39[i] = _mm256_sub_epi16(V39[i], _mm256_srai_epi16(_mm256_add_epi16(V38[i], V40[i]), 1));
        V41[i] = _mm256_sub_epi16(V41[i], _mm256_srai_epi16(_mm256_add_epi16(V40[i], V42[i]), 1));
        V43[i] = _mm256_sub_epi16(V43[i], _mm256_srai_epi16(_mm256_add_epi16(V42[i], V44[i]), 1));
        V45[i] = _mm256_sub_epi16(V45[i], _mm256_srai_epi16(_mm256_add_epi16(V44[i], V46[i]), 1));
        V47[i] = _mm256_sub_epi16(V47[i], _mm256_srai_epi16(_mm256_add_epi16(V46[i], V48[i]), 1));

        V49[i] = _mm256_sub_epi16(V49[i], _mm256_srai_epi16(_mm256_add_epi16(V48[i], V50[i]), 1));
        V51[i] = _mm256_sub_epi16(V51[i], _mm256_srai_epi16(_mm256_add_epi16(V50[i], V52[i]), 1));
        V53[i] = _mm256_sub_epi16(V53[i], _mm256_srai_epi16(_mm256_add_epi16(V52[i], V54[i]), 1));
        V55[i] = _mm256_sub_epi16(V55[i], _mm256_srai_epi16(_mm256_add_epi16(V54[i], V56[i]), 1));
        V57[i] = _mm256_sub_epi16(V57[i], _mm256_srai_epi16(_mm256_add_epi16(V56[i], V58[i]), 1));
        V59[i] = _mm256_sub_epi16(V59[i], _mm256_srai_epi16(_mm256_add_epi16(V58[i], V60[i]), 1));
        V61[i] = _mm256_sub_epi16(V61[i], _mm256_srai_epi16(_mm256_add_epi16(V60[i], V62[i]), 1));
        V63[i] = _mm256_sub_epi16(V63[i], _mm256_srai_epi16(_mm256_add_epi16(V62[i], V62[i]), 1));
    }

    //pExt[x] += (pExt[x - 1] + pExt[x + 1] + 2) >> 2;
    for (i = 0; i < 4; i++) {
        V00[i] = _mm256_add_epi16(V00[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V01[i], V01[i]), mAddOffset2), 2));
        V02[i] = _mm256_add_epi16(V02[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V01[i], V03[i]), mAddOffset2), 2));
        V04[i] = _mm256_add_epi16(V04[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V03[i], V05[i]), mAddOffset2), 2));
        V06[i] = _mm256_add_epi16(V06[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V05[i], V07[i]), mAddOffset2), 2));
        V08[i] = _mm256_add_epi16(V08[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V07[i], V09[i]), mAddOffset2), 2));
        V10[i] = _mm256_add_epi16(V10[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V09[i], V11[i]), mAddOffset2), 2));
        V12[i] = _mm256_add_epi16(V12[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V11[i], V13[i]), mAddOffset2), 2));
        V14[i] = _mm256_add_epi16(V14[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V13[i], V15[i]), mAddOffset2), 2));


        V16[i] = _mm256_add_epi16(V16[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V15[i], V17[i]), mAddOffset2), 2));
        V18[i] = _mm256_add_epi16(V18[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V17[i], V19[i]), mAddOffset2), 2));
        V20[i] = _mm256_add_epi16(V20[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V19[i], V21[i]), mAddOffset2), 2));
        V22[i] = _mm256_add_epi16(V22[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V21[i], V23[i]), mAddOffset2), 2));
        V24[i] = _mm256_add_epi16(V24[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V23[i], V25[i]), mAddOffset2), 2));
        V26[i] = _mm256_add_epi16(V26[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V25[i], V27[i]), mAddOffset2), 2));
        V28[i] = _mm256_add_epi16(V28[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V27[i], V29[i]), mAddOffset2), 2));
        V30[i] = _mm256_add_epi16(V30[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V29[i], V31[i]), mAddOffset2), 2));

        V32[i] = _mm256_add_epi16(V32[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V31[i], V33[i]), mAddOffset2), 2));
        V34[i] = _mm256_add_epi16(V34[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V33[i], V35[i]), mAddOffset2), 2));
        V36[i] = _mm256_add_epi16(V36[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V35[i], V37[i]), mAddOffset2), 2));
        V38[i] = _mm256_add_epi16(V38[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V37[i], V39[i]), mAddOffset2), 2));
        V40[i] = _mm256_add_epi16(V40[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V39[i], V41[i]), mAddOffset2), 2));
        V42[i] = _mm256_add_epi16(V42[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V41[i], V43[i]), mAddOffset2), 2));
        V44[i] = _mm256_add_epi16(V44[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V43[i], V45[i]), mAddOffset2), 2));
        V46[i] = _mm256_add_epi16(V46[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V45[i], V47[i]), mAddOffset2), 2));

        V48[i] = _mm256_add_epi16(V48[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V47[i], V49[i]), mAddOffset2), 2));
        V50[i] = _mm256_add_epi16(V50[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V49[i], V51[i]), mAddOffset2), 2));
        V52[i] = _mm256_add_epi16(V52[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V51[i], V53[i]), mAddOffset2), 2));
        V54[i] = _mm256_add_epi16(V54[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V53[i], V55[i]), mAddOffset2), 2));
        V56[i] = _mm256_add_epi16(V56[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V55[i], V57[i]), mAddOffset2), 2));
        V58[i] = _mm256_add_epi16(V58[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V57[i], V59[i]), mAddOffset2), 2));
        V60[i] = _mm256_add_epi16(V60[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V59[i], V61[i]), mAddOffset2), 2));
        V62[i] = _mm256_add_epi16(V62[i], _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(V61[i], V63[i]), mAddOffset2), 2));
    }

    TRANSPOSE_16x16_16BIT(V00[0], V02[0], V04[0], V06[0], V08[0], V10[0], V12[0], V14[0], V16[0], V18[0], V20[0], V22[0], V24[0], V26[0], V28[0], V30[0], A00[0], A01[0], A02[0], A03[0], A04[0], A05[0], A06[0], A07[0], A08[0], A09[0], A10[0], A11[0], A12[0], A13[0], A14[0], A15[0]);
    TRANSPOSE_16x16_16BIT(V32[0], V34[0], V36[0], V38[0], V40[0], V42[0], V44[0], V46[0], V48[0], V50[0], V52[0], V54[0], V56[0], V58[0], V60[0], V62[0], A00[1], A01[1], A02[1], A03[1], A04[1], A05[1], A06[1], A07[1], A08[1], A09[1], A10[1], A11[1], A12[1], A13[1], A14[1], A15[1]);

    TRANSPOSE_16x16_16BIT(V00[1], V02[1], V04[1], V06[1], V08[1], V10[1], V12[1], V14[1], V16[1], V18[1], V20[1], V22[1], V24[1], V26[1], V28[1], V30[1], A16[0], A17[0], A18[0], A19[0], A20[0], A21[0], A22[0], A23[0], A24[0], A25[0], A26[0], A27[0], A28[0], A29[0], A30[0], A31[0]);
    TRANSPOSE_16x16_16BIT(V32[1], V34[1], V36[1], V38[1], V40[1], V42[1], V44[1], V46[1], V48[1], V50[1], V52[1], V54[1], V56[1], V58[1], V60[1], V62[1], A16[1], A17[1], A18[1], A19[1], A20[1], A21[1], A22[1], A23[1], A24[1], A25[1], A26[1], A27[1], A28[1], A29[1], A30[1], A31[1]);

    TRANSPOSE_16x16_16BIT(V00[2], V02[2], V04[2], V06[2], V08[2], V10[2], V12[2], V14[2], V16[2], V18[2], V20[2], V22[2], V24[2], V26[2], V28[2], V30[2], A32[0], A33[0], A34[0], A35[0], A36[0], A37[0], A38[0], A39[0], A40[0], A41[0], A42[0], A43[0], A44[0], A45[0], A46[0], A47[0]);
    TRANSPOSE_16x16_16BIT(V32[2], V34[2], V36[2], V38[2], V40[2], V42[2], V44[2], V46[2], V48[2], V50[2], V52[2], V54[2], V56[2], V58[2], V60[2], V62[2], A32[1], A33[1], A34[1], A35[1], A36[1], A37[1], A38[1], A39[1], A40[1], A41[1], A42[1], A43[1], A44[1], A45[1], A46[1], A47[1]);

    TRANSPOSE_16x16_16BIT(V00[3], V02[3], V04[3], V06[3], V08[3], V10[3], V12[3], V14[3], V16[3], V18[3], V20[3], V22[3], V24[3], V26[3], V28[3], V30[3], A48[0], A49[0], A50[0], A51[0], A52[0], A53[0], A54[0], A55[0], A56[0], A57[0], A58[0], A59[0], A60[0], A61[0], A62[0], A63[0]);
    TRANSPOSE_16x16_16BIT(V32[3], V34[3], V36[3], V38[3], V40[3], V42[3], V44[3], V46[3], V48[3], V50[3], V52[3], V54[3], V56[3], V58[3], V60[3], V62[3], A48[1], A49[1], A50[1], A51[1], A52[1], A53[1], A54[1], A55[1], A56[1], A57[1], A58[1], A59[1], A60[1], A61[1], A62[1], A63[1]);

    //pExt[y] -= (pExt[y - 1] + pExt[y + 1]) >> 1;
    for (i = 0; i < 2; i++) {
        A01[i] = _mm256_sub_epi16(A01[i], _mm256_srai_epi16(_mm256_add_epi16(A00[i], A02[i]), 1));
        A03[i] = _mm256_sub_epi16(A03[i], _mm256_srai_epi16(_mm256_add_epi16(A02[i], A04[i]), 1));
        A05[i] = _mm256_sub_epi16(A05[i], _mm256_srai_epi16(_mm256_add_epi16(A04[i], A06[i]), 1));
        A07[i] = _mm256_sub_epi16(A07[i], _mm256_srai_epi16(_mm256_add_epi16(A06[i], A08[i]), 1));
        A09[i] = _mm256_sub_epi16(A09[i], _mm256_srai_epi16(_mm256_add_epi16(A08[i], A10[i]), 1));
        A11[i] = _mm256_sub_epi16(A11[i], _mm256_srai_epi16(_mm256_add_epi16(A10[i], A12[i]), 1));
        A13[i] = _mm256_sub_epi16(A13[i], _mm256_srai_epi16(_mm256_add_epi16(A12[i], A14[i]), 1));
        A15[i] = _mm256_sub_epi16(A15[i], _mm256_srai_epi16(_mm256_add_epi16(A14[i], A16[i]), 1));

        A17[i] = _mm256_sub_epi16(A17[i], _mm256_srai_epi16(_mm256_add_epi16(A16[i], A18[i]), 1));
        A19[i] = _mm256_sub_epi16(A19[i], _mm256_srai_epi16(_mm256_add_epi16(A18[i], A20[i]), 1));
        A21[i] = _mm256_sub_epi16(A21[i], _mm256_srai_epi16(_mm256_add_epi16(A20[i], A22[i]), 1));
        A23[i] = _mm256_sub_epi16(A23[i], _mm256_srai_epi16(_mm256_add_epi16(A22[i], A24[i]), 1));
        A25[i] = _mm256_sub_epi16(A25[i], _mm256_srai_epi16(_mm256_add_epi16(A24[i], A26[i]), 1));
        A27[i] = _mm256_sub_epi16(A27[i], _mm256_srai_epi16(_mm256_add_epi16(A26[i], A28[i]), 1));
        A29[i] = _mm256_sub_epi16(A29[i], _mm256_srai_epi16(_mm256_add_epi16(A28[i], A30[i]), 1));
        A31[i] = _mm256_sub_epi16(A31[i], _mm256_srai_epi16(_mm256_add_epi16(A30[i], A32[i]), 1));

        A33[i] = _mm256_sub_epi16(A33[i], _mm256_srai_epi16(_mm256_add_epi16(A32[i], A34[i]), 1));
        A35[i] = _mm256_sub_epi16(A35[i], _mm256_srai_epi16(_mm256_add_epi16(A34[i], A36[i]), 1));
        A37[i] = _mm256_sub_epi16(A37[i], _mm256_srai_epi16(_mm256_add_epi16(A36[i], A38[i]), 1));
        A39[i] = _mm256_sub_epi16(A39[i], _mm256_srai_epi16(_mm256_add_epi16(A38[i], A40[i]), 1));
        A41[i] = _mm256_sub_epi16(A41[i], _mm256_srai_epi16(_mm256_add_epi16(A40[i], A42[i]), 1));
        A43[i] = _mm256_sub_epi16(A43[i], _mm256_srai_epi16(_mm256_add_epi16(A42[i], A44[i]), 1));
        A45[i] = _mm256_sub_epi16(A45[i], _mm256_srai_epi16(_mm256_add_epi16(A44[i], A46[i]), 1));
        A47[i] = _mm256_sub_epi16(A47[i], _mm256_srai_epi16(_mm256_add_epi16(A46[i], A48[i]), 1));

        A49[i] = _mm256_sub_epi16(A49[i], _mm256_srai_epi16(_mm256_add_epi16(A48[i], A50[i]), 1));
        A51[i] = _mm256_sub_epi16(A51[i], _mm256_srai_epi16(_mm256_add_epi16(A50[i], A52[i]), 1));
        A53[i] = _mm256_sub_epi16(A53[i], _mm256_srai_epi16(_mm256_add_epi16(A52[i], A54[i]), 1));
        A55[i] = _mm256_sub_epi16(A55[i], _mm256_srai_epi16(_mm256_add_epi16(A54[i], A56[i]), 1));
        A57[i] = _mm256_sub_epi16(A57[i], _mm256_srai_epi16(_mm256_add_epi16(A56[i], A58[i]), 1));
        A59[i] = _mm256_sub_epi16(A59[i], _mm256_srai_epi16(_mm256_add_epi16(A58[i], A60[i]), 1));
        A61[i] = _mm256_sub_epi16(A61[i], _mm256_srai_epi16(_mm256_add_epi16(A60[i], A62[i]), 1));
        A63[i] = _mm256_sub_epi16(A63[i], _mm256_srai_epi16(_mm256_add_epi16(A62[i], A62[i]), 1));
    }

    //pExt[y] = (pExt[y] << 1) + ((pExt[y - 1] + pExt[y + 1] + 1) >> 1);
    for (i = 0; i < 2; i++) {
        A00[i] = _mm256_add_epi16(_mm256_slli_epi16(A00[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A01[i], A01[i]), mAddOffset1), 1));
        A02[i] = _mm256_add_epi16(_mm256_slli_epi16(A02[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A01[i], A03[i]), mAddOffset1), 1));
        A04[i] = _mm256_add_epi16(_mm256_slli_epi16(A04[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A03[i], A05[i]), mAddOffset1), 1));
        A06[i] = _mm256_add_epi16(_mm256_slli_epi16(A06[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A05[i], A07[i]), mAddOffset1), 1));
        A08[i] = _mm256_add_epi16(_mm256_slli_epi16(A08[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A07[i], A09[i]), mAddOffset1), 1));
        A10[i] = _mm256_add_epi16(_mm256_slli_epi16(A10[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A09[i], A11[i]), mAddOffset1), 1));
        A12[i] = _mm256_add_epi16(_mm256_slli_epi16(A12[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A11[i], A13[i]), mAddOffset1), 1));
        A14[i] = _mm256_add_epi16(_mm256_slli_epi16(A14[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A13[i], A15[i]), mAddOffset1), 1));

        A16[i] = _mm256_add_epi16(_mm256_slli_epi16(A16[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A15[i], A17[i]), mAddOffset1), 1));
        A18[i] = _mm256_add_epi16(_mm256_slli_epi16(A18[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A17[i], A19[i]), mAddOffset1), 1));
        A20[i] = _mm256_add_epi16(_mm256_slli_epi16(A20[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A19[i], A21[i]), mAddOffset1), 1));
        A22[i] = _mm256_add_epi16(_mm256_slli_epi16(A22[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A21[i], A23[i]), mAddOffset1), 1));
        A24[i] = _mm256_add_epi16(_mm256_slli_epi16(A24[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A23[i], A25[i]), mAddOffset1), 1));
        A26[i] = _mm256_add_epi16(_mm256_slli_epi16(A26[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A25[i], A27[i]), mAddOffset1), 1));
        A28[i] = _mm256_add_epi16(_mm256_slli_epi16(A28[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A27[i], A29[i]), mAddOffset1), 1));
        A30[i] = _mm256_add_epi16(_mm256_slli_epi16(A30[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A29[i], A31[i]), mAddOffset1), 1));

        A32[i] = _mm256_add_epi16(_mm256_slli_epi16(A32[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A31[i], A33[i]), mAddOffset1), 1));
        A34[i] = _mm256_add_epi16(_mm256_slli_epi16(A34[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A33[i], A35[i]), mAddOffset1), 1));
        A36[i] = _mm256_add_epi16(_mm256_slli_epi16(A36[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A35[i], A37[i]), mAddOffset1), 1));
        A38[i] = _mm256_add_epi16(_mm256_slli_epi16(A38[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A37[i], A39[i]), mAddOffset1), 1));
        A40[i] = _mm256_add_epi16(_mm256_slli_epi16(A40[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A39[i], A41[i]), mAddOffset1), 1));
        A42[i] = _mm256_add_epi16(_mm256_slli_epi16(A42[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A41[i], A43[i]), mAddOffset1), 1));
        A44[i] = _mm256_add_epi16(_mm256_slli_epi16(A44[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A43[i], A45[i]), mAddOffset1), 1));
        A46[i] = _mm256_add_epi16(_mm256_slli_epi16(A46[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A45[i], A47[i]), mAddOffset1), 1));

        A48[i] = _mm256_add_epi16(_mm256_slli_epi16(A48[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A47[i], A49[i]), mAddOffset1), 1));
        A50[i] = _mm256_add_epi16(_mm256_slli_epi16(A50[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A49[i], A51[i]), mAddOffset1), 1));
        A52[i] = _mm256_add_epi16(_mm256_slli_epi16(A52[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A51[i], A53[i]), mAddOffset1), 1));
        A54[i] = _mm256_add_epi16(_mm256_slli_epi16(A54[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A53[i], A55[i]), mAddOffset1), 1));
        A56[i] = _mm256_add_epi16(_mm256_slli_epi16(A56[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A55[i], A57[i]), mAddOffset1), 1));
        A58[i] = _mm256_add_epi16(_mm256_slli_epi16(A58[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A57[i], A59[i]), mAddOffset1), 1));
        A60[i] = _mm256_add_epi16(_mm256_slli_epi16(A60[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A59[i], A61[i]), mAddOffset1), 1));
        A62[i] = _mm256_add_epi16(_mm256_slli_epi16(A62[i], 1), _mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(A61[i], A63[i]), mAddOffset1), 1));
    }

    //Store
    for (i = 0; i < 2; i++) {
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 0], A00[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 1], A02[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 2], A04[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 3], A06[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 4], A08[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 5], A10[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 6], A12[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 7], A14[i]);

        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 8], A16[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 9], A18[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 10], A20[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 11], A22[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 12], A24[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 13], A26[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 14], A28[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 15], A30[i]);

        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 16], A32[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 17], A34[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 18], A36[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 19], A38[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 20], A40[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 21], A42[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 22], A44[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 23], A46[i]);

        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 24], A48[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 25], A50[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 26], A52[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 27], A54[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 28], A56[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 29], A58[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 30], A60[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * i + 32 * 31], A62[i]);
    }
}


/* ---------------------------------------------------------------------------
 */
void dct_c_64x64_avx2(const coeff_t *src, coeff_t *dst, int i_src)
{
    UNUSED_PARAMETER(src);
    UNUSED_PARAMETER(i_src);
    wavelet_64x64_avx2(dst);
    dct_c_32x32_avx2(dst, dst, 32 | 1);
}

/* ---------------------------------------------------------------------------
 */
void dct_c_64x64_half_avx2(const coeff_t *src, coeff_t *dst, int i_src)
{
    UNUSED_PARAMETER(src);
    UNUSED_PARAMETER(i_src);
    wavelet_64x64_avx2(dst);
    dct_c_32x32_half_avx2(dst, dst, 32 | 1);
}

/* ---------------------------------------------------------------------------
 */
void dct_c_64x16_avx2(const coeff_t *src, coeff_t *dst, int i_src)
{
    UNUSED_PARAMETER(src);
    UNUSED_PARAMETER(i_src);
    wavelet_64x16_avx2(dst);
    dct_c_32x8_avx2(dst, dst, 32 | 0x01);
}

/* ---------------------------------------------------------------------------
 */
void dct_c_16x64_avx2(const coeff_t *src, coeff_t *dst, int i_src)
{
    UNUSED_PARAMETER(src);
    UNUSED_PARAMETER(i_src);
    wavelet_16x64_avx2(dst);
    dct_c_8x32_avx2(dst, dst, 8 | 0x01);
}



/* ---------------------------------------------------------------------------
 */
ALIGN32(static const int16_t tab_dct_16_0_avx[][16]) = {
    { 0x0F0E, 0x0D0C, 0x0B0A, 0x0908, 0x0706, 0x0504, 0x0302, 0x0100, 0x0F0E, 0x0D0C, 0x0B0A, 0x0908, 0x0706, 0x0504, 0x0302, 0x0100 },  // 0
    { 0x0100, 0x0F0E, 0x0706, 0x0908, 0x0302, 0x0D0C, 0x0504, 0x0B0A, 0x0100, 0x0F0E, 0x0706, 0x0908, 0x0302, 0x0D0C, 0x0504, 0x0B0A },  // 1
    { 0x0100, 0x0706, 0x0302, 0x0504, 0x0F0E, 0x0908, 0x0D0C, 0x0B0A, 0x0100, 0x0706, 0x0302, 0x0504, 0x0F0E, 0x0908, 0x0D0C, 0x0B0A },  // 2
    { 0x0F0E, 0x0908, 0x0D0C, 0x0B0A, 0x0100, 0x0706, 0x0302, 0x0504, 0x0F0E, 0x0908, 0x0D0C, 0x0B0A, 0x0100, 0x0706, 0x0302, 0x0504 }   // 3
};

/* ---------------------------------------------------------------------------
 */
ALIGN32(static const int16_t tab_dct_16_1_avx[][16]) = {
    { 45, 43, 40, 35, 29, 21, 13, 4, 45, 43, 40, 35, 29, 21, 13, 4 },  //  0
    { 43, 29, 4, -21, -40, -45, -35, -13, 43, 29, 4, -21, -40, -45, -35, -13 },  //  1
    { 40, 4, -35, -43, -13, 29, 45, 21, 40, 4, -35, -43, -13, 29, 45, 21 },  //  2
    { 35, -21, -43, 4, 45, 13, -40, -29, 35, -21, -43, 4, 45, 13, -40, -29 },  //  3
    { 29, -40, -13, 45, -4, -43, 21, 35, 29, -40, -13, 45, -4, -43, 21, 35 },  //  4
    { 21, -45, 29, 13, -43, 35, 4, -40, 21, -45, 29, 13, -43, 35, 4, -40 },  //  5
    { 13, -35, 45, -40, 21, 4, -29, 43, 13, -35, 45, -40, 21, 4, -29, 43 },  //  6
    { 4, -13, 21, -29, 35, -40, 43, -45, 4, -13, 21, -29, 35, -40, 43, -45 },  //  7
    { 42, 42, -42, -42, 17, 17, -17, -17, 42, 42, -42, -42, 17, 17, -17, -17 },  //  8
    { 17, 17, -17, -17, -42, -42, 42, 42, 17, 17, -17, -17, -42, -42, 42, 42 },  //  9
    { 44, 44, 9, 9, 38, 38, 25, 25, 44, 44, 9, 9, 38, 38, 25, 25 },  // 10
    { 38, 38, -25, -25, -9, -9, -44, -44, 38, 38, -25, -25, -9, -9, -44, -44 },  // 11
    { 25, 25, 38, 38, -44, -44, 9, 9, 25, 25, 38, 38, -44, -44, 9, 9 },  // 12
    { 9, 9, -44, -44, -25, -25, 38, 38, 9, 9, -44, -44, -25, -25, 38, 38 },  // 13

    /* ---------------------------------------------------------------------------
     */
#define MAKE_COEF(a0, a1, a2, a3, a4, a5, a6, a7) \
    { (a0), -(a0), (a3), -(a3), (a1), -(a1), (a2), -(a2), (a0), -(a0), (a3), -(a3), (a1), -(a1), (a2), -(a2) }, \
    { (a7), -(a7), (a4), -(a4), (a6), -(a6), (a5), -(a5), (a7), -(a7), (a4), -(a4), (a6), -(a6), (a5), -(a5) },

    MAKE_COEF(45, 43, 40, 35, 29, 21, 13, 4)
    MAKE_COEF(43, 29, 4, -21, -40, -45, -35, -13)
    MAKE_COEF(40, 4, -35, -43, -13, 29, 45, 21)
    MAKE_COEF(35, -21, -43, 4, 45, 13, -40, -29)
    MAKE_COEF(29, -40, -13, 45, -4, -43, 21, 35)
    MAKE_COEF(21, -45, 29, 13, -43, 35, 4, -40)
    MAKE_COEF(13, -35, 45, -40, 21, 4, -29, 43)
    MAKE_COEF(4, -13, 21, -29, 35, -40, 43, -45)
#undef MAKE_COEF
    {
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32
    },
    { 32, 32, 32, 32, -32, -32, -32, -32, 32, 32, 32, 32, -32, -32, -32, -32 },

};

/* ---------------------------------------------------------------------------
 */
#define pair256_set_epi16(a, b) \
    _mm256_set_epi16(b, a, b, a, b, a, b, a, b, a, b, a, b, a, b, a)

/* ---------------------------------------------------------------------------
 */
void dct_c_4x16_avx2(const coeff_t *src, coeff_t *dst, int i_src)
{
    const __m256i k_p32_p32 = _mm256_set1_epi16(32);
    const __m256i k_p32_m32 = pair256_set_epi16(32, -32);
    const __m256i k_p42_p17 = pair256_set_epi16(42, 17);
    const __m256i k_p17_m42 = pair256_set_epi16(17, -42);
    __m256i in[16];
    __m256i  tr00, tr01;
    __m256i r0, r1,  r4, r5, t0, t2, u0, u1, u2, u3;

    __m256i T00A, T01A,  T00B, T01B;
    __m256i T10, T11, T12, T13;
    __m256i T20, T21, T22, T23, T24, T25, T26, T27;
    __m256i T30, T31, T32, T33;
    __m256i T40, T41;
    __m256i T70;

    int shift2 = 9;
    const __m256i c_256 = _mm256_set1_epi32(256);

    __m256i tab_dct_16_02 = _mm256_loadu_si256((__m256i*)tab_dct_16_0_avx[2]);
    __m256i tab_dct_16_03 = _mm256_loadu_si256((__m256i*)tab_dct_16_0_avx[3]);
    __m256i tab_dct_16_8 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[8]);
    __m256i tab_dct_16_9 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[9]);
    __m256i tab_dct_16_10 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[10]);
    __m256i tab_dct_16_11 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[11]);
    __m256i tab_dct_16_12 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[12]);
    __m256i tab_dct_16_13 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[13]);
    __m256i tab_dct_16_14 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[14]);
    __m256i tab_dct_16_15 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[15]);
    __m256i tab_dct_16_16 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[16]);
    __m256i tab_dct_16_17 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[17]);
    __m256i tab_dct_16_18 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[18]);
    __m256i tab_dct_16_19 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[19]);
    __m256i tab_dct_16_20 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[20]);
    __m256i tab_dct_16_21 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[21]);
    __m256i tab_dct_16_22 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[22]);
    __m256i tab_dct_16_23 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[23]);
    __m256i tab_dct_16_24 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[24]);
    __m256i tab_dct_16_25 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[25]);
    __m256i tab_dct_16_26 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[26]);
    __m256i tab_dct_16_27 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[27]);
    __m256i tab_dct_16_28 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[28]);
    __m256i tab_dct_16_29 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[29]);
    __m256i tab_dct_16_30 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[30]);
    __m256i tab_dct_16_31 = _mm256_loadu_si256((__m256i*)tab_dct_16_1_avx[31]);

    ///// DCT1 4x16->16x4//////
    //transpose input data
    in[0] = _mm256_load_si256((const __m256i *)(src + 0 * i_src));
    in[4] = _mm256_load_si256((const __m256i *)(src + 4 * i_src));
    in[8] = _mm256_load_si256((const __m256i *)(src + 8 * i_src));
    in[12] = _mm256_load_si256((const __m256i *)(src + 12 * i_src));

    tr00 = _mm256_shuffle_epi32(in[0], 0xD8);//00 01 04 05 02 03 06 07 / 08 09 12 13 10 11 14 15
    tr01 = _mm256_shuffle_epi32(in[4], 0xD8);
    r0 = _mm256_shufflehi_epi16(tr00, 0xB1);//00 01 04 05 03 02 07 06 / 08 09 12 13 11 10 15 14
    r4 = _mm256_shufflehi_epi16(tr01, 0xB1);

    r1 = _mm256_unpacklo_epi64(r0, r4);//0, 1, 4, 5, 16, 17, 20, 21, 8, 9, 12, 13, 24, 25, 28, 29
    r5 = _mm256_unpackhi_epi64(r0, r4);//3, 2, 7, 6, 19, 18, 23, 22, 11, 10, 15, 14, 27, 26, 31, 30

    t0 = _mm256_add_epi16(r1, r5);//3, 3, 11, 11, 35, 35, 43, 43, 19, 19, 27, 27, 51, 51, 59, 59
    t2 = _mm256_sub_epi16(r1, r5);//-3, -1, -3, -1, -3, -1, -3, -1, -3, -1, -3, -1, -3, -1, -3, -1

    u0 = _mm256_madd_epi16(t0, k_p32_p32);
    u2 = _mm256_madd_epi16(t0, k_p32_m32);
    u1 = _mm256_madd_epi16(t2, k_p42_p17);
    u3 = _mm256_madd_epi16(t2, k_p17_m42);

    T00A = _mm256_packs_epi32(u0, u1);
    T00B = _mm256_packs_epi32(u2, u3);

    T00A = _mm256_permute4x64_epi64(T00A, 0xD8);
    T00A = _mm256_shuffle_epi32(T00A, 0xD8);//out[0]   out[4]    out[1]    out[5]
    T00B = _mm256_permute4x64_epi64(T00B, 0xD8);
    T00B = _mm256_shuffle_epi32(T00B, 0xD8);//out[2]   out[6]    out[3]    out[7]

    tr00 = _mm256_shuffle_epi32(in[8], 0xD8);//00 01 04 05 02 03 06 07 / 08 09 12 13 10 11 14 15
    tr01 = _mm256_shuffle_epi32(in[12], 0xD8);
    r0 = _mm256_shufflehi_epi16(tr00, 0xB1);//00 01 04 05 03 02 07 06 / 08 09 12 13 11 10 15 14
    r4 = _mm256_shufflehi_epi16(tr01, 0xB1);

    r1 = _mm256_unpacklo_epi64(r0, r4);
    r5 = _mm256_unpackhi_epi64(r0, r4);

    t0 = _mm256_add_epi16(r1, r5);
    t2 = _mm256_sub_epi16(r1, r5);

    u0 = _mm256_madd_epi16(t0, k_p32_p32);
    u2 = _mm256_madd_epi16(t0, k_p32_m32);
    u1 = _mm256_madd_epi16(t2, k_p42_p17);
    u3 = _mm256_madd_epi16(t2, k_p17_m42);

    T01A = _mm256_packs_epi32(u0, u1);
    T01B = _mm256_packs_epi32(u2, u3);

    T01A = _mm256_permute4x64_epi64(T01A, 0xD8);
    T01A = _mm256_shuffle_epi32(T01A, 0xD8);//out[8]   out[12]    out[9]    out[13]
    T01B = _mm256_permute4x64_epi64(T01B, 0xD8);
    T01B = _mm256_shuffle_epi32(T01B, 0xD8);//out[10]  out[14]    out[11]   out[15]

    T00A = _mm256_shuffle_epi8(T00A, tab_dct_16_02);
    T01A = _mm256_shuffle_epi8(T01A, tab_dct_16_03);
    T00B = _mm256_shuffle_epi8(T00B, tab_dct_16_02);
    T01B = _mm256_shuffle_epi8(T01B, tab_dct_16_03);

    T10 = _mm256_unpacklo_epi16(T00A, T01A);//T10 T12
    T11 = _mm256_unpackhi_epi16(T00A, T01A);//T11 T13
    T12 = _mm256_unpacklo_epi16(T00B, T01B);//T14 T16
    T13 = _mm256_unpackhi_epi16(T00B, T01B);//T15 T17


    T20 = _mm256_madd_epi16(T10, tab_dct_16_30);
    T21 = _mm256_madd_epi16(T11, tab_dct_16_30);
    T22 = _mm256_madd_epi16(T12, tab_dct_16_30);
    T23 = _mm256_madd_epi16(T13, tab_dct_16_30);
    T24 = _mm256_madd_epi16(T10, tab_dct_16_10);
    T25 = _mm256_madd_epi16(T11, tab_dct_16_10);
    T26 = _mm256_madd_epi16(T12, tab_dct_16_10);
    T27 = _mm256_madd_epi16(T13, tab_dct_16_10);

    T30 = _mm256_add_epi32(T20, T21);
    T31 = _mm256_add_epi32(T22, T23);
    T32 = _mm256_sub_epi32(T24, T25);
    T33 = _mm256_sub_epi32(T26, T27);

    T30 = _mm256_hadd_epi32(T30, T31);
    T32 = _mm256_hadd_epi32(T32, T33);

    T40 = _mm256_hadd_epi32(T30, T32);
    T40 = _mm256_srai_epi32(_mm256_add_epi32(T40, c_256), shift2);
    T40 = _mm256_permute4x64_epi64(T40, 0xD8);//0 2

    T20 = _mm256_madd_epi16(T10, tab_dct_16_14);
    T21 = _mm256_madd_epi16(T11, tab_dct_16_15);
    T22 = _mm256_madd_epi16(T12, tab_dct_16_14);
    T23 = _mm256_madd_epi16(T13, tab_dct_16_15);
    T24 = _mm256_madd_epi16(T10, tab_dct_16_16);
    T25 = _mm256_madd_epi16(T11, tab_dct_16_17);
    T26 = _mm256_madd_epi16(T12, tab_dct_16_16);
    T27 = _mm256_madd_epi16(T13, tab_dct_16_17);

    T30 = _mm256_add_epi32(T20, T21);
    T31 = _mm256_add_epi32(T22, T23);
    T32 = _mm256_add_epi32(T24, T25);
    T33 = _mm256_add_epi32(T26, T27);

    T30 = _mm256_hadd_epi32(T30, T31);
    T32 = _mm256_hadd_epi32(T32, T33);

    T41 = _mm256_hadd_epi32(T30, T32);
    T41 = _mm256_srai_epi32(_mm256_add_epi32(T41, c_256), shift2);
    T41 = _mm256_permute4x64_epi64(T41, 0xD8);//1 3

    T70 = _mm256_packs_epi32(T40, T41);
    T70 = _mm256_shufflehi_epi16(T70, 0xD8);
    T70 = _mm256_shufflelo_epi16(T70, 0xD8);

    _mm256_storeu2_m128i((__m128i*)(dst + 2 * 4), (__m128i*)(dst + 0 * 4), T70);


    T20 = _mm256_madd_epi16(T10, tab_dct_16_8);
    T21 = _mm256_madd_epi16(T11, tab_dct_16_8);
    T22 = _mm256_madd_epi16(T12, tab_dct_16_8);
    T23 = _mm256_madd_epi16(T13, tab_dct_16_8);
    T24 = _mm256_madd_epi16(T10, tab_dct_16_11);
    T25 = _mm256_madd_epi16(T11, tab_dct_16_11);
    T26 = _mm256_madd_epi16(T12, tab_dct_16_11);
    T27 = _mm256_madd_epi16(T13, tab_dct_16_11);

    T30 = _mm256_add_epi32(T20, T21);
    T31 = _mm256_add_epi32(T22, T23);
    T32 = _mm256_sub_epi32(T24, T25);
    T33 = _mm256_sub_epi32(T26, T27);

    T30 = _mm256_hadd_epi32(T30, T31);
    T32 = _mm256_hadd_epi32(T32, T33);

    T40 = _mm256_hadd_epi32(T30, T32);
    T40 = _mm256_srai_epi32(_mm256_add_epi32(T40, c_256), shift2);
    T40 = _mm256_permute4x64_epi64(T40, 0xD8);//4 6

    T20 = _mm256_madd_epi16(T10, tab_dct_16_18);
    T21 = _mm256_madd_epi16(T11, tab_dct_16_19);
    T22 = _mm256_madd_epi16(T12, tab_dct_16_18);
    T23 = _mm256_madd_epi16(T13, tab_dct_16_19);
    T24 = _mm256_madd_epi16(T10, tab_dct_16_20);
    T25 = _mm256_madd_epi16(T11, tab_dct_16_21);
    T26 = _mm256_madd_epi16(T12, tab_dct_16_20);
    T27 = _mm256_madd_epi16(T13, tab_dct_16_21);

    T30 = _mm256_add_epi32(T20, T21);
    T31 = _mm256_add_epi32(T22, T23);
    T32 = _mm256_add_epi32(T24, T25);
    T33 = _mm256_add_epi32(T26, T27);

    T30 = _mm256_hadd_epi32(T30, T31);
    T32 = _mm256_hadd_epi32(T32, T33);

    T41 = _mm256_hadd_epi32(T30, T32);
    T41 = _mm256_srai_epi32(_mm256_add_epi32(T41, c_256), shift2);
    T41 = _mm256_permute4x64_epi64(T41, 0xD8);//5 7



    T70 = _mm256_packs_epi32(T40, T41);
    T70 = _mm256_shufflehi_epi16(T70, 0xD8);
    T70 = _mm256_shufflelo_epi16(T70, 0xD8);

    _mm256_storeu2_m128i((__m128i*)(dst + 6 * 4), (__m128i*)(dst + 4 * 4), T70);



    T20 = _mm256_madd_epi16(T10, tab_dct_16_31);
    T21 = _mm256_madd_epi16(T11, tab_dct_16_31);
    T22 = _mm256_madd_epi16(T12, tab_dct_16_31);
    T23 = _mm256_madd_epi16(T13, tab_dct_16_31);
    T24 = _mm256_madd_epi16(T10, tab_dct_16_12);
    T25 = _mm256_madd_epi16(T11, tab_dct_16_12);
    T26 = _mm256_madd_epi16(T12, tab_dct_16_12);
    T27 = _mm256_madd_epi16(T13, tab_dct_16_12);

    T30 = _mm256_add_epi32(T20, T21);
    T31 = _mm256_add_epi32(T22, T23);
    T32 = _mm256_sub_epi32(T24, T25);
    T33 = _mm256_sub_epi32(T26, T27);

    T30 = _mm256_hadd_epi32(T30, T31);
    T32 = _mm256_hadd_epi32(T32, T33);

    T40 = _mm256_hadd_epi32(T30, T32);
    T40 = _mm256_srai_epi32(_mm256_add_epi32(T40, c_256), shift2);
    T40 = _mm256_permute4x64_epi64(T40, 0xD8);//8 10

    T20 = _mm256_madd_epi16(T10, tab_dct_16_22);
    T21 = _mm256_madd_epi16(T11, tab_dct_16_23);
    T22 = _mm256_madd_epi16(T12, tab_dct_16_22);
    T23 = _mm256_madd_epi16(T13, tab_dct_16_23);
    T24 = _mm256_madd_epi16(T10, tab_dct_16_24);
    T25 = _mm256_madd_epi16(T11, tab_dct_16_25);
    T26 = _mm256_madd_epi16(T12, tab_dct_16_24);
    T27 = _mm256_madd_epi16(T13, tab_dct_16_25);

    T30 = _mm256_add_epi32(T20, T21);
    T31 = _mm256_add_epi32(T22, T23);
    T32 = _mm256_add_epi32(T24, T25);
    T33 = _mm256_add_epi32(T26, T27);

    T30 = _mm256_hadd_epi32(T30, T31);
    T32 = _mm256_hadd_epi32(T32, T33);

    T41 = _mm256_hadd_epi32(T30, T32);
    T41 = _mm256_srai_epi32(_mm256_add_epi32(T41, c_256), shift2);
    T41 = _mm256_permute4x64_epi64(T41, 0xD8);//9 11



    T70 = _mm256_packs_epi32(T40, T41);
    T70 = _mm256_shufflehi_epi16(T70, 0xD8);
    T70 = _mm256_shufflelo_epi16(T70, 0xD8);

    _mm256_storeu2_m128i((__m128i*)(dst + 10 * 4), (__m128i*)(dst + 8 * 4), T70);



    T20 = _mm256_madd_epi16(T10, tab_dct_16_9);
    T21 = _mm256_madd_epi16(T11, tab_dct_16_9);
    T22 = _mm256_madd_epi16(T12, tab_dct_16_9);
    T23 = _mm256_madd_epi16(T13, tab_dct_16_9);
    T24 = _mm256_madd_epi16(T10, tab_dct_16_13);
    T25 = _mm256_madd_epi16(T11, tab_dct_16_13);
    T26 = _mm256_madd_epi16(T12, tab_dct_16_13);
    T27 = _mm256_madd_epi16(T13, tab_dct_16_13);

    T30 = _mm256_add_epi32(T20, T21);
    T31 = _mm256_add_epi32(T22, T23);
    T32 = _mm256_sub_epi32(T24, T25);
    T33 = _mm256_sub_epi32(T26, T27);

    T30 = _mm256_hadd_epi32(T30, T31);
    T32 = _mm256_hadd_epi32(T32, T33);

    T40 = _mm256_hadd_epi32(T30, T32);
    T40 = _mm256_srai_epi32(_mm256_add_epi32(T40, c_256), shift2);
    T40 = _mm256_permute4x64_epi64(T40, 0xD8);//12 14

    T20 = _mm256_madd_epi16(T10, tab_dct_16_26);
    T21 = _mm256_madd_epi16(T11, tab_dct_16_27);
    T22 = _mm256_madd_epi16(T12, tab_dct_16_26);
    T23 = _mm256_madd_epi16(T13, tab_dct_16_27);
    T24 = _mm256_madd_epi16(T10, tab_dct_16_28);
    T25 = _mm256_madd_epi16(T11, tab_dct_16_29);
    T26 = _mm256_madd_epi16(T12, tab_dct_16_28);
    T27 = _mm256_madd_epi16(T13, tab_dct_16_29);

    T30 = _mm256_add_epi32(T20, T21);
    T31 = _mm256_add_epi32(T22, T23);
    T32 = _mm256_add_epi32(T24, T25);
    T33 = _mm256_add_epi32(T26, T27);

    T30 = _mm256_hadd_epi32(T30, T31);
    T32 = _mm256_hadd_epi32(T32, T33);

    T41 = _mm256_hadd_epi32(T30, T32);
    T41 = _mm256_srai_epi32(_mm256_add_epi32(T41, c_256), shift2);
    T41 = _mm256_permute4x64_epi64(T41, 0xD8);//13 15



    T70 = _mm256_packs_epi32(T40, T41);
    T70 = _mm256_shufflehi_epi16(T70, 0xD8);
    T70 = _mm256_shufflelo_epi16(T70, 0xD8);

    _mm256_storeu2_m128i((__m128i*)(dst + 14 * 4), (__m128i*)(dst + 12 * 4), T70);

}


/* ---------------------------------------------------------------------------
 */
ALIGN32(static const int16_t tab_dct_16x4_avx2[][16]) = {
    { 0x0F0E, 0x0D0C, 0x0B0A, 0x0908, 0x0706, 0x0504, 0x0302, 0x0100, 0x0F0E, 0x0D0C, 0x0B0A, 0x0908, 0x0706, 0x0504, 0x0302, 0x0100 },
    { 32, 32, 32, 32, 32, 32, 32, 32, 32, -32, 32, -32, 32, -32, 32, -32 },//0  8
    { 42, 17, 42, 17, 42, 17, 42, 17, 17, -42, 17, -42, 17, -42, 17, -42 },//4  12
    { 44, 9, 38, 25, 44, 9, 38, 25, 44, 9, 38, 25, 44, 9, 38, 25 },//2
    { 38, -25, -9, -44, 38, -25, -9, -44, 38, -25, -9, -44, 38, -25, -9, -44 },//6
    { 25, 38, -44, 9, 25, 38, -44, 9, 25, 38, -44, 9, 25, 38, -44, 9 },//10
    { 9, -44, -25, 38, 9, -44, -25, 38, 9, -44, -25, 38, 9, -44, -25, 38 } //14
};

/* ---------------------------------------------------------------------------
 */
ALIGN32(static const int16_t tab_dct_16x4_1_avx2[][16]) = {
    { 45, 43, 40, 35, 29, 21, 13, 4, 45, 43, 40, 35, 29, 21, 13, 4 },//  0
    { 43, 29, 4, -21, -40, -45, -35, -13, 43, 29, 4, -21, -40, -45, -35, -13 },//  1
    { 40, 4, -35, -43, -13, 29, 45, 21, 40, 4, -35, -43, -13, 29, 45, 21 },//  2
    { 35, -21, -43, 4, 45, 13, -40, -29, 35, -21, -43, 4, 45, 13, -40, -29 },//  3
    { 29, -40, -13, 45, -4, -43, 21, 35, 29, -40, -13, 45, -4, -43, 21, 35 },//  4
    { 21, -45, 29, 13, -43, 35, 4, -40, 21, -45, 29, 13, -43, 35, 4, -40 },//  5
    { 13, -35, 45, -40, 21, 4, -29, 43, 13, -35, 45, -40, 21, 4, -29, 43 },//  6
    { 4, -13, 21, -29, 35, -40, 43, -45, 4, -13, 21, -29, 35, -40, 43, -45 } //  7
};

/* ---------------------------------------------------------------------------
 */
ALIGN32(static const int16_t tab_dct1_4_avx2[][16]) = {
    { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 },
    { 42, 17, -17, -42, 42, 17, -17, -42, 42, 17, -17, -42, 42, 17, -17, -42 },
    { 32, -32, -32, 32, 32, -32, -32, 32, 32, -32, -32, 32, 32, -32, -32, 32 },
    { 17, -42, 42, -17, 17, -42, 42, -17, 17, -42, 42, -17, 17, -42, 42, -17 }
};

/* ---------------------------------------------------------------------------
 */
void dct_c_16x4_avx2(const coeff_t * src, coeff_t * dst, int i_src)
{
    int shift1 = B16X16_IN_BIT + FACTO_BIT +g_bit_depth + 1 - LIMIT_BIT;
    int shift2 = B16X16_IN_BIT + FACTO_BIT - 2;
    const int ADD1 = (1 << shift1) >> 1;
    const int ADD2 = (1 << shift2) >> 1;
    const __m256i c_2 = _mm256_set1_epi32(ADD1);        // TODO: shift1 = 2
    const __m256i k_ROUNDING2 = _mm256_set1_epi32(ADD2);

    __m256i T00, T01, T02, T03, T04, T05, T06, T07;
    __m256i T10, T11;
    __m256i T20, T21;
    __m256i T30, T31;
    __m256i T40, T41;
    __m256i T50, T51;
    __m256i T60, T61;
    __m256i im1[2];
    __m256i temp[4];
    __m256i r0, r1,  u0, u1, u2, u3, v0, v1, v2, v3, w0, w1, w2, w3;
    __m256i res0, res1, res2, res3;
    __m256i d0, d1, d2, d3;

    __m256i im[4];

    //////// DCT1 16x4->4x16 ///////
    //input data
    T00 = _mm256_loadu2_m128i((__m128i *)(src + 2 * i_src + 0), (__m128i *)(src + 0 * i_src + 0)); // [00 01 02 03 04 05 06 07 20 21 22 23 24 25 26 27]
    T01 = _mm256_loadu2_m128i((__m128i *)(src + 2 * i_src + 8), (__m128i *)(src + 0 * i_src + 8)); // [08 09 0A 0B 0C 0D 0E 0F 28 29 2A 2B 2C 2D 2E 2F]
    T02 = _mm256_loadu2_m128i((__m128i *)(src + 3 * i_src + 0), (__m128i *)(src + 1 * i_src + 0));
    T03 = _mm256_loadu2_m128i((__m128i *)(src + 3 * i_src + 8), (__m128i *)(src + 1 * i_src + 8));

    //shuffle
    T04 = _mm256_shuffle_epi8(T01, _mm256_load_si256((__m256i *)tab_dct_16x4_avx2[0]));
    T05 = _mm256_shuffle_epi8(T03, _mm256_load_si256((__m256i *)tab_dct_16x4_avx2[0]));
    T06 = _mm256_sub_epi16(T00, T04);
    T07 = _mm256_sub_epi16(T02, T05);

    T00 = _mm256_shuffle_epi8(T00, _mm256_load_si256((__m256i *)tab_dct_16_shuffle_avx2[0])); // [00 07 03 04 01 06 02 05 20 27 23 24 21 26 22 25]
    T02 = _mm256_shuffle_epi8(T02, _mm256_load_si256((__m256i *)tab_dct_16_shuffle_avx2[0])); // [0F 08 0B 0C 0E 09 0D 0A 2F 28 2B 2C 2E 29 2D 2A]
    T01 = _mm256_shuffle_epi8(T01, _mm256_load_si256((__m256i *)tab_dct_16_shuffle_avx2[1]));
    T03 = _mm256_shuffle_epi8(T03, _mm256_load_si256((__m256i *)tab_dct_16_shuffle_avx2[1]));

    T10 = _mm256_add_epi16(T00, T01); // [00 07 03 04 01 06 02 05 20 27 23 24 21 26 22 25]
    T11 = _mm256_add_epi16(T02, T03);

    T20 = _mm256_hadd_epi16(T10, T11);// [00 03 01 02 10 13 11 12 20 23 21 22 30 33 31 32]
    T21 = _mm256_hsub_epi16(T10, T11);

    T30 = _mm256_hadd_epi16(T20, T20);// [00 01 10 11 00 01 10 11 20 21 30 31 20 21 30 31]
    T31 = _mm256_hsub_epi16(T20, T20);

    T30 = _mm256_permute4x64_epi64(T30, 0xd8);
    T31 = _mm256_permute4x64_epi64(T31, 0xd8);

    T40 = _mm256_madd_epi16(T30, _mm256_load_si256((__m256i*)tab_dct_16x4_avx2[1]));//0/8
    T41 = _mm256_madd_epi16(T31, _mm256_load_si256((__m256i*)tab_dct_16x4_avx2[2]));//4/12
    T50 = _mm256_srai_epi32(_mm256_add_epi32(T40, c_2), shift1);
    T51 = _mm256_srai_epi32(_mm256_add_epi32(T41, c_2), shift1);
    T60 = _mm256_packs_epi32(T50, T51);//0 4 8 12
    T60 = _mm256_permute4x64_epi64(T60, 0xd8);//0 8 4 12
    im[0] = T60;

    T40 = _mm256_madd_epi16(T21, _mm256_load_si256((__m256i*)tab_dct_16x4_avx2[3]));//2
    T41 = _mm256_madd_epi16(T21, _mm256_load_si256((__m256i*)tab_dct_16x4_avx2[4]));///6
    T50 = _mm256_hadd_epi32(T40, T41);
    T50 = _mm256_permute4x64_epi64(T50, 0xd8);

    T40 = _mm256_madd_epi16(T21, _mm256_load_si256((__m256i*)tab_dct_16x4_avx2[5]));//10
    T41 = _mm256_madd_epi16(T21, _mm256_load_si256((__m256i*)tab_dct_16x4_avx2[6]));//14
    T51 = _mm256_hadd_epi32(T40, T41);
    T51 = _mm256_permute4x64_epi64(T51, 0xd8);

    T50 = _mm256_srai_epi32(_mm256_add_epi32(T50, c_2), shift1);
    T51 = _mm256_srai_epi32(_mm256_add_epi32(T51, c_2), shift1);
    T60 = _mm256_packs_epi32(T50, T51);//2 10 6 14
    im[1] = T60;

    T40 = _mm256_madd_epi16(T06, _mm256_load_si256((__m256i*)tab_dct_16x4_1_avx2[(0)]));//1
    T41 = _mm256_madd_epi16(T07, _mm256_load_si256((__m256i*)tab_dct_16x4_1_avx2[(0)]));
    T50 = _mm256_hadd_epi32(T40, T41);
    T40 = _mm256_madd_epi16(T06, _mm256_load_si256((__m256i*)tab_dct_16x4_1_avx2[(1)]));//3
    T41 = _mm256_madd_epi16(T07, _mm256_load_si256((__m256i*)tab_dct_16x4_1_avx2[(1)]));
    T51 = _mm256_hadd_epi32(T40, T41);
    T60 = _mm256_hadd_epi32(T50, T51);
    T60 = _mm256_permute4x64_epi64(T60, 0xd8);
    T60 = _mm256_srai_epi32(_mm256_add_epi32(T60, c_2), shift1);

    T40 = _mm256_madd_epi16(T06, _mm256_load_si256((__m256i*)tab_dct_16x4_1_avx2[(2)]));//9
    T41 = _mm256_madd_epi16(T07, _mm256_load_si256((__m256i*)tab_dct_16x4_1_avx2[(2)]));
    T50 = _mm256_hadd_epi32(T40, T41);
    T40 = _mm256_madd_epi16(T06, _mm256_load_si256((__m256i*)tab_dct_16x4_1_avx2[(3)]));//11
    T41 = _mm256_madd_epi16(T07, _mm256_load_si256((__m256i*)tab_dct_16x4_1_avx2[(3)]));
    T51 = _mm256_hadd_epi32(T40, T41);
    T61 = _mm256_hadd_epi32(T50, T51);
    T61 = _mm256_permute4x64_epi64(T61, 0xd8);
    T61 = _mm256_srai_epi32(_mm256_add_epi32(T61, c_2), shift1);
    T60 = _mm256_packs_epi32(T60, T61);//1 5 3 7
    T60 = _mm256_permute4x64_epi64(T60, 0xd8);//1 3 5 7
    im[2] = T60;

    T40 = _mm256_madd_epi16(T06, _mm256_load_si256((__m256i*)tab_dct_16x4_1_avx2[(4)]));//9
    T41 = _mm256_madd_epi16(T07, _mm256_load_si256((__m256i*)tab_dct_16x4_1_avx2[(4)]));
    T50 = _mm256_hadd_epi32(T40, T41);
    T40 = _mm256_madd_epi16(T06, _mm256_load_si256((__m256i*)tab_dct_16x4_1_avx2[(5)]));//11
    T41 = _mm256_madd_epi16(T07, _mm256_load_si256((__m256i*)tab_dct_16x4_1_avx2[(5)]));
    T51 = _mm256_hadd_epi32(T40, T41);
    T60 = _mm256_hadd_epi32(T50, T51);
    T60 = _mm256_permute4x64_epi64(T60, 0xd8);
    T60 = _mm256_srai_epi32(_mm256_add_epi32(T60, c_2), shift1);

    T40 = _mm256_madd_epi16(T06, _mm256_load_si256((__m256i*)tab_dct_16x4_1_avx2[(6)]));//13
    T41 = _mm256_madd_epi16(T07, _mm256_load_si256((__m256i*)tab_dct_16x4_1_avx2[(6)]));
    T50 = _mm256_hadd_epi32(T40, T41);
    T40 = _mm256_madd_epi16(T06, _mm256_load_si256((__m256i*)tab_dct_16x4_1_avx2[(7)]));//15
    T41 = _mm256_madd_epi16(T07, _mm256_load_si256((__m256i*)tab_dct_16x4_1_avx2[(7)]));
    T51 = _mm256_hadd_epi32(T40, T41);
    T61 = _mm256_hadd_epi32(T50, T51);
    T61 = _mm256_permute4x64_epi64(T61, 0xd8);
    T61 = _mm256_srai_epi32(_mm256_add_epi32(T61, c_2), shift1);
    T60 = _mm256_packs_epi32(T60, T61);//9 13 11 15
    T60 = _mm256_permute4x64_epi64(T60, 0xd8);//9 11 13 15
    im[3] = T60;

    im1[0] = _mm256_unpacklo_epi64(im[0], im[1]);//0 2 4 6
    im1[1] = _mm256_unpackhi_epi64(im[0], im[1]);//8 10 12 14

    temp[0] = _mm256_unpacklo_epi64(im1[0], im[2]);//0 1 4 5
    temp[1] = _mm256_unpackhi_epi64(im1[0], im[2]);//2 3 6 7
    temp[2] = _mm256_unpacklo_epi64(im1[1], im[3]);//8 9 12 13
    temp[3] = _mm256_unpackhi_epi64(im1[1], im[3]);//10 11 14 15

    //////// DCT2 16x4->4x16 ///////
    //1st 4x4
    r0 = _mm256_madd_epi16(temp[0], _mm256_load_si256((__m256i*)tab_dct1_4_avx2[0]));
    r1 = _mm256_madd_epi16(temp[1], _mm256_load_si256((__m256i*)tab_dct1_4_avx2[0]));
    u0 = _mm256_hadd_epi32(r0, r1);

    r0 = _mm256_madd_epi16(temp[0], _mm256_load_si256((__m256i*)tab_dct1_4_avx2[1]));
    r1 = _mm256_madd_epi16(temp[1], _mm256_load_si256((__m256i*)tab_dct1_4_avx2[1]));
    u1 = _mm256_hadd_epi32(r0, r1);

    r0 = _mm256_madd_epi16(temp[0], _mm256_load_si256((__m256i*)tab_dct1_4_avx2[2]));
    r1 = _mm256_madd_epi16(temp[1], _mm256_load_si256((__m256i*)tab_dct1_4_avx2[2]));
    u2 = _mm256_hadd_epi32(r0, r1);

    r0 = _mm256_madd_epi16(temp[0], _mm256_load_si256((__m256i*)tab_dct1_4_avx2[3]));
    r1 = _mm256_madd_epi16(temp[1], _mm256_load_si256((__m256i*)tab_dct1_4_avx2[3]));
    u3 = _mm256_hadd_epi32(r0, r1);

    v0 = _mm256_add_epi32(u0, k_ROUNDING2);
    v1 = _mm256_add_epi32(u1, k_ROUNDING2);
    v2 = _mm256_add_epi32(u2, k_ROUNDING2);
    v3 = _mm256_add_epi32(u3, k_ROUNDING2);
    w0 = _mm256_srai_epi32(v0, shift2);
    w1 = _mm256_srai_epi32(v1, shift2);
    w2 = _mm256_srai_epi32(v2, shift2);
    w3 = _mm256_srai_epi32(v3, shift2);
    res0 = _mm256_packs_epi32(w0, w1);//0 2
    res1 = _mm256_packs_epi32(w2, w3);//1 3

    //2st 4x4
    r0 = _mm256_madd_epi16(temp[2], _mm256_load_si256((__m256i*)tab_dct1_4_avx2[0]));
    r1 = _mm256_madd_epi16(temp[3], _mm256_load_si256((__m256i*)tab_dct1_4_avx2[0]));
    u0 = _mm256_hadd_epi32(r0, r1);

    r0 = _mm256_madd_epi16(temp[2], _mm256_load_si256((__m256i*)tab_dct1_4_avx2[1]));
    r1 = _mm256_madd_epi16(temp[3], _mm256_load_si256((__m256i*)tab_dct1_4_avx2[1]));
    u1 = _mm256_hadd_epi32(r0, r1);

    r0 = _mm256_madd_epi16(temp[2], _mm256_load_si256((__m256i*)tab_dct1_4_avx2[2]));
    r1 = _mm256_madd_epi16(temp[3], _mm256_load_si256((__m256i*)tab_dct1_4_avx2[2]));
    u2 = _mm256_hadd_epi32(r0, r1);

    r0 = _mm256_madd_epi16(temp[2], _mm256_load_si256((__m256i*)tab_dct1_4_avx2[3]));
    r1 = _mm256_madd_epi16(temp[3], _mm256_load_si256((__m256i*)tab_dct1_4_avx2[3]));
    u3 = _mm256_hadd_epi32(r0, r1);

    v0 = _mm256_add_epi32(u0, k_ROUNDING2);
    v1 = _mm256_add_epi32(u1, k_ROUNDING2);
    v2 = _mm256_add_epi32(u2, k_ROUNDING2);
    v3 = _mm256_add_epi32(u3, k_ROUNDING2);
    w0 = _mm256_srai_epi32(v0, shift2);
    w1 = _mm256_srai_epi32(v1, shift2);
    w2 = _mm256_srai_epi32(v2, shift2);
    w3 = _mm256_srai_epi32(v3, shift2);
    res2 = _mm256_packs_epi32(w0, w1);//4 6
    res3 = _mm256_packs_epi32(w2, w3);//5 7

    res0 = _mm256_permute4x64_epi64(res0, 0xd8);
    res1 = _mm256_permute4x64_epi64(res1, 0xd8);
    res2 = _mm256_permute4x64_epi64(res2, 0xd8);
    res3 = _mm256_permute4x64_epi64(res3, 0xd8);

    d0 = _mm256_permute2x128_si256(res0, res2, 0x20);
    d1 = _mm256_permute2x128_si256(res0, res2, 0x31);
    d2 = _mm256_permute2x128_si256(res1, res3, 0x20);
    d3 = _mm256_permute2x128_si256(res1, res3, 0x31);

    //store
    _mm256_store_si256((__m256i *)(dst + 0), d0);
    _mm256_store_si256((__m256i *)(dst + 16), d1);
    _mm256_store_si256((__m256i *)(dst + 32), d2);
    _mm256_store_si256((__m256i *)(dst + 48), d3);
}

