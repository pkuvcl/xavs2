/*
 * intrinsic_dct.c
 *
 * Description of this file:
 *    SSE assembly functions of DCT module of the xavs2 library
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
#include "../avs2_defs.h"
#include "intrinsic.h"

void *xavs2_fast_memzero_mmx(void *dst, size_t n);

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
    { 0x0100, 0x0706, 0x0302, 0x0504, 0x0F0E, 0x0908, 0x0D0C, 0x0B0A },  // 2  [0 3 1 2 7 4 6 5]
    { 0x0F0E, 0x0908, 0x0D0C, 0x0B0A, 0x0100, 0x0706, 0x0302, 0x0504 }   // 3  [7 4 6 5 0 3 1 2]
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

ALIGN32(static const int32_t tab_dct_16_new_coeff[][4]) = {
    { 32, 32, 32, 32 }, // 0
    { 44, 9, 38, 25 }, // 2
    { 42, 17, 42, 17 }, // 4
    { 38, -25, -9, -44 }, // 6
    { 32, 32, 32, 32 }, // 8
    { 25, 38, -44, 9 }, // 10
    { 17, -42, 17, -42 }, // 12
    { 9, -44, -25, 38 }, // 14
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



ALIGN32(static const int32_t tab_dct_32_zhangjiaqi[][4]) = {
    { 32, 32, 32, 32 },        // order:0   //   0 / 16
    { 42, -42, 17, -17 },        // order:1   //   8
    { 17, -17, -42, 42 },        // order:2   //   24

#define MAKE_COEF8(a0, a1, a2, a3, a4, a5, a6, a7) \
    { (a0), (a1), (a2), (a3) }, \
    { (a4), (a5), (a6), (a7) },

    MAKE_COEF8(44, -44, 9, -9, 38, -38, 25, -25)          // order:3/4   //   4
    MAKE_COEF8(38, -38, -25, 25, -9, 9, -44, 44)          // order:5/6   //   12
    MAKE_COEF8(25, -25, 38, -38, -44, 44, 9, -9)          // order:7/8   //   20
    MAKE_COEF8(9, -9, -44, 44, -25, 25, 38, -38)          // order:9/10   //  28
#undef MAKE_COEF8


#define MAKE_COEF8(a0, a1, a2, a3, a4, a5, a6, a7) \
    {(a0), (a7), (a3), (a4)}, \
    { (a1), (a6), (a2), (a5) },

    MAKE_COEF8(45, 43, 40, 35, 29, 21, 13, 4)          // order:11/12   //   2
    MAKE_COEF8(43, 29, 4, -21, -40, -45, -35, -13)          // order:13/14   //   6
    MAKE_COEF8(40, 4, -35, -43, -13, 29, 45, 21)          // order:15/16   //  10
    MAKE_COEF8(35, -21, -43, 4, 45, 13, -40, -29)          // order:17/18   //  14
    MAKE_COEF8(29, -40, -13, 45, -4, -43, 21, 35)          // order:19/20   //  18
    MAKE_COEF8(21, -45, 29, 13, -43, 35, 4, -40)          // order:21/22   //  22
    MAKE_COEF8(13, -35, 45, -40, 21, 4, -29, 43)          // order:23/24   //  26
    MAKE_COEF8(4, -13, 21, -29, 35, -40, 43, -45)          // order:25/26   //  30

#undef MAKE_COEF8

#define MAKE_COEF16(a00, a01, a02, a03, a04, a05, a06, a07, a08, a09, a10, a11, a12, a13, a14, a15) \
    { (a00), (a07), (a03), (a04) }, \
    { (a01), (a06), (a02), (a05) }, \
    { (a15), (a08), (a12), (a11) }, \
    { (a14), (a09), (a13), (a10) },

    MAKE_COEF16(45, 45, 44, 43, 41, 39, 36, 34, 30, 27, 23, 19, 15, 11, 7, 2)          // order:27   //   1
    MAKE_COEF16(45, 41, 34, 23, 11, -2, -15, -27, -36, -43, -45, -44, -39, -30, -19, -7)          // order:31   //   3
    MAKE_COEF16(44, 34, 15, -7, -27, -41, -45, -39, -23, -2, 19, 36, 45, 43, 30, 11)          // order:35   //   5
    MAKE_COEF16(43, 23, -7, -34, -45, -36, -11, 19, 41, 44, 27, -2, -30, -45, -39, -15)          // order:39   //   7
    MAKE_COEF16(41, 11, -27, -45, -30, 7, 39, 43, 15, -23, -45, -34, 2, 36, 44, 19)          // order:43   //   9
    MAKE_COEF16(39, -2, -41, -36, 7, 43, 34, -11, -44, -30, 15, 45, 27, -19, -45, -23)          // order:47   //  11
    MAKE_COEF16(36, -15, -45, -11, 39, 34, -19, -45, -7, 41, 30, -23, -44, -2, 43, 27)          // order:51   //  13
    MAKE_COEF16(34, -27, -39, 19, 43, -11, -45, 2, 45, 7, -44, -15, 41, 23, -36, -30)          // order:55   //  15
    MAKE_COEF16(30, -36, -23, 41, 15, -44, -7, 45, -2, -45, 11, 43, -19, -39, 27, 34)          // order:59   //  17
    MAKE_COEF16(27, -43, -2, 44, -23, -30, 41, 7, -45, 19, 34, -39, -11, 45, -15, -36)          // order:63   //  19
    MAKE_COEF16(23, -45, 19, 27, -45, 15, 30, -44, 11, 34, -43, 7, 36, -41, 2, 39)          // order:67   //  21
    MAKE_COEF16(19, -44, 36, -2, -34, 45, -23, -15, 43, -39, 7, 30, -45, 27, 11, -41)          // order:71   //  23
    MAKE_COEF16(15, -39, 45, -30, 2, 27, -44, 41, -19, -11, 36, -45, 34, -7, -23, 43)          // order:75   //  25
    MAKE_COEF16(11, -30, 43, -45, 36, -19, -2, 23, -39, 45, -41, 27, -7, -15, 34, -44)          // order:79   //  27
    MAKE_COEF16(7, -19, 30, -39, 44, -45, 43, -36, 27, -15, 2, 11, -23, 34, -41, 45)          // order:83   //  29
    MAKE_COEF16(2, -7, 11, -15, 19, -23, 27, -30, 34, -36, 39, -41, 43, -44, 45, -45)          // order:87   //  31

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


/* ---------------------------------------------------------------------------
futl change 2016.12.19*/
void dct_c_4x4_sse128(const coeff_t *src, coeff_t *dst, int i_src)
{
    const int SHIFT1 = B4X4_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT;
    const int SHIFT2 = B4X4_IN_BIT + FACTO_BIT;
    const int ADD1 = (1 << SHIFT1) >> 1;
    const int ADD2 = (1 << SHIFT2) >> 1;

    // Const
    __m128i c_add1 = _mm_set1_epi32(ADD1);
    __m128i c_add2 = _mm_set1_epi32(ADD2);

    __m128i T20, T21;
    __m128i T30, T31, T32, T33;
    __m128i T40, T41, T50, T51, T60, T61, T62, T63, T70, T71, T72, T73;
    __m128i T50_, T51_;

    __m128i Tab0, Tab1, Tab2, Tab3;
    Tab0 = _mm_load_si128((__m128i*)tab_dct_4[0]);
    Tab1 = _mm_load_si128((__m128i*)tab_dct_4[1]);
    Tab2 = _mm_load_si128((__m128i*)tab_dct_4[2]);
    Tab3 = _mm_load_si128((__m128i*)tab_dct_4[3]);

    T20 = _mm_loadu_si128((__m128i*)(src + 0 * i_src));
    T21 = _mm_loadu_si128((__m128i*)(src + 2 * i_src));

    // DCT1
    T30 = _mm_shuffle_epi32(T20, 0xD8);         // [13 12 03 02 11 10 01 00]
    T31 = _mm_shuffle_epi32(T21, 0xD8);         // [33 32 23 22 31 30 21 20]
    T32 = _mm_shufflehi_epi16(T30, 0xB1);      // [12 13 02 03 11 10 01 00]
    T33 = _mm_shufflehi_epi16(T31, 0xB1);      // [32 33 22 23 31 30 21 20]

    T40 = _mm_unpacklo_epi64(T32, T33);        // [31 30 21 20 11 10 01 00]
    T41 = _mm_unpackhi_epi64(T32, T33);        // [32 33 22 23 12 13 02 03]
    T50 = _mm_add_epi16(T40, T41);             // [1+2 0+3]
    T51 = _mm_sub_epi16(T40, T41);             // [1-2 0-3]
    T60 = _mm_madd_epi16(T50, Tab0); // [ 32*s12 + 32*s03] = [03 02 01 00]
    T61 = _mm_madd_epi16(T51, Tab1); // [ 17*d12 + 42*d03] = [13 12 11 10]
    T62 = _mm_madd_epi16(T50, Tab2); // [-32*s12 + 32*s03] = [23 22 21 20]
    T63 = _mm_madd_epi16(T51, Tab3); // [-42*d12 + 17*d03] = [33 32 31 30]

    T60 = _mm_srai_epi32(_mm_add_epi32(T60, c_add1), SHIFT1);  // [30 20 10 00]
    T61 = _mm_srai_epi32(_mm_add_epi32(T61, c_add1), SHIFT1);  // [31 21 11 01]
    T62 = _mm_srai_epi32(_mm_add_epi32(T62, c_add1), SHIFT1);  // [32 22 12 02]
    T63 = _mm_srai_epi32(_mm_add_epi32(T63, c_add1), SHIFT1);  // [33 23 13 03]

    // Transpose
    T20 = _mm_packs_epi32(T60, T61);            // [13 12 11 10 03 02 01 00]
    T21 = _mm_packs_epi32(T62, T63);            // [33 32 31 30 23 22 21 20]

    T30 = _mm_shuffle_epi32(T20, 0xD8);       // [13 12 03 02 11 10 01 00]
    T31 = _mm_shuffle_epi32(T21, 0xD8);       // [33 32 23 22 31 30 21 20]
    T32 = _mm_shufflehi_epi16(T30, 0xB1);       // [12 13 02 03 11 10 01 00]
    T33 = _mm_shufflehi_epi16(T31, 0xB1);       // [32 33 22 23 31 30 21 20]

    T40 = _mm_unpacklo_epi64(T32, T33);         // [31 30 21 20 11 10 01 00]
    T41 = _mm_unpackhi_epi64(T32, T33);         // [32 33 22 23 12 13 02 03]

    T50_ = _mm_madd_epi16(T40, Tab0);
    T51_ = _mm_madd_epi16(T41, Tab0);
    T60 = _mm_add_epi32(T50_, T51_);
    T50_ = _mm_madd_epi16(T40, Tab1);
    T51_ = _mm_madd_epi16(T41, Tab1);
    T61 = _mm_sub_epi32(T50_, T51_);
    T50_ = _mm_madd_epi16(T40, Tab2);
    T51_ = _mm_madd_epi16(T41, Tab2);
    T62 = _mm_add_epi32(T50_, T51_);
    T50_ = _mm_madd_epi16(T40, Tab3);
    T51_ = _mm_madd_epi16(T41, Tab3);
    T63 = _mm_sub_epi32(T50_, T51_);

    T70 = _mm_srai_epi32(_mm_add_epi32(T60, c_add2), SHIFT2);  // [30 20 10 00]
    T71 = _mm_srai_epi32(_mm_add_epi32(T61, c_add2), SHIFT2);  // [31 21 11 01]
    T72 = _mm_srai_epi32(_mm_add_epi32(T62, c_add2), SHIFT2);  // [32 22 12 02]
    T73 = _mm_srai_epi32(_mm_add_epi32(T63, c_add2), SHIFT2);  // [33 23 13 03]

    T70 = _mm_packs_epi32(T70, T70);
    T71 = _mm_packs_epi32(T71, T71);
    T72 = _mm_packs_epi32(T72, T72);
    T73 = _mm_packs_epi32(T73, T73);

    _mm_storel_epi64((__m128i*)(dst + 0 * 4), T70);
    _mm_storel_epi64((__m128i*)(dst + 1 * 4), T71);
    _mm_storel_epi64((__m128i*)(dst + 2 * 4), T72);
    _mm_storel_epi64((__m128i*)(dst + 3 * 4), T73);
}

/* ---------------------------------------------------------------------------
futl change 2016.12.19*/
void dct_c_8x8_sse128(const coeff_t *src, coeff_t *dst, int i_src)
{
    const int SHIFT1 = B8X8_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT;
    const int SHIFT2 = B8X8_IN_BIT + FACTO_BIT;
    const int ADD1 = (1 << SHIFT1) >> 1;
    const int ADD2 = (1 << SHIFT2) >> 1;

    // Const
    __m128i c_add1 = _mm_set1_epi32(ADD1);              // add1 = 1
    __m128i c_add2 = _mm_set1_epi32(ADD2);              // add2 = 128

    // DCT1
    __m128i T00, T01, T02, T03, T04, T05, T06, T07;
    __m128i T10, T11, T12, T13, T14, T15, T16, T17;
    __m128i T20, T21, T22, T23, T24, T25, T26, T27;
    __m128i T30, T31, T32, T33;
    __m128i T40, T41, T42, T43, T44, T45, T46, T47;
    __m128i T50, T51, T52, T53, T54, T55, T56, T57;

    __m128i Tab;

    T00 = _mm_load_si128((__m128i*)(src + 0 * i_src));    // [07 06 05 04 03 02 01 00]
    T01 = _mm_load_si128((__m128i*)(src + 1 * i_src));    // [17 16 15 14 13 12 11 10]
    T02 = _mm_load_si128((__m128i*)(src + 2 * i_src));    // [27 26 25 24 23 22 21 20]
    T03 = _mm_load_si128((__m128i*)(src + 3 * i_src));    // [37 36 35 34 33 32 31 30]
    T04 = _mm_load_si128((__m128i*)(src + 4 * i_src));    // [47 46 45 44 43 42 41 40]
    T05 = _mm_load_si128((__m128i*)(src + 5 * i_src));    // [57 56 55 54 53 52 51 50]
    T06 = _mm_load_si128((__m128i*)(src + 6 * i_src));    // [67 66 65 64 63 62 61 60]
    T07 = _mm_load_si128((__m128i*)(src + 7 * i_src));    // [77 76 75 74 73 72 71 70]

    Tab = _mm_load_si128((__m128i*)tab_dct_8[0]);
    T10 = _mm_shuffle_epi8(T00, Tab);  // [05 02 06 01 04 03 07 00]
    T11 = _mm_shuffle_epi8(T01, Tab);
    T12 = _mm_shuffle_epi8(T02, Tab);
    T13 = _mm_shuffle_epi8(T03, Tab);
    T14 = _mm_shuffle_epi8(T04, Tab);
    T15 = _mm_shuffle_epi8(T05, Tab);
    T16 = _mm_shuffle_epi8(T06, Tab);
    T17 = _mm_shuffle_epi8(T07, Tab);

    T20 = _mm_hadd_epi16(T10, T11);     // [s25_1 s16_1 s34_1 s07_1 s25_0 s16_0 s34_0 s07_0]
    T21 = _mm_hadd_epi16(T12, T13);     // [s25_3 s16_3 s34_3 s07_3 s25_2 s16_2 s34_2 s07_2]
    T22 = _mm_hadd_epi16(T14, T15);     // [s25_5 s16_5 s34_5 s07_5 s25_4 s16_4 s34_4 s07_4]
    T23 = _mm_hadd_epi16(T16, T17);     // [s25_7 s16_7 s34_7 s07_7 s25_6 s16_6 s34_6 s07_6]

    T24 = _mm_hsub_epi16(T10, T11);     // [d25_1 d16_1 d34_1 d07_1 d25_0 d16_0 d34_0 d07_0]
    T25 = _mm_hsub_epi16(T12, T13);     // [d25_3 d16_3 d34_3 d07_3 d25_2 d16_2 d34_2 d07_2]
    T26 = _mm_hsub_epi16(T14, T15);     // [d25_5 d16_5 d34_5 d07_5 d25_4 d16_4 d34_4 d07_4]
    T27 = _mm_hsub_epi16(T16, T17);     // [d25_7 d16_7 d34_7 d07_7 d25_6 d16_6 d34_6 d07_6]

    T30 = _mm_hadd_epi16(T20, T21);     // [EE1_3 EE0_3 EE1_2 EE0_2 EE1_1 EE0_1 EE1_0 EE0_0]
    T31 = _mm_hadd_epi16(T22, T23);     // [EE1_7 EE0_7 EE1_6 EE0_6 EE1_5 EE0_5 EE1_4 EE0_4]
    T32 = _mm_hsub_epi16(T20, T21);     // [EO1_3 EO0_3 EO1_2 EO0_2 EO1_1 EO0_1 EO1_0 EO0_0]
    T33 = _mm_hsub_epi16(T22, T23);     // [EO1_7 EO0_7 EO1_6 EO0_6 EO1_5 EO0_5 EO1_4 EO0_4]

    Tab = _mm_load_si128((__m128i*)tab_dct_8[1]);
    T40 = _mm_madd_epi16(T30, Tab);
    T41 = _mm_madd_epi16(T31, Tab);
    T40 = _mm_srai_epi32(_mm_add_epi32(T40, c_add1), SHIFT1);
    T41 = _mm_srai_epi32(_mm_add_epi32(T41, c_add1), SHIFT1);
    T50 = _mm_packs_epi32(T40, T41);

    Tab = _mm_load_si128((__m128i*)tab_dct_8[2]);
    T42 = _mm_madd_epi16(T30, Tab);
    T43 = _mm_madd_epi16(T31, Tab);
    T42 = _mm_srai_epi32(_mm_add_epi32(T42, c_add1), SHIFT1);
    T43 = _mm_srai_epi32(_mm_add_epi32(T43, c_add1), SHIFT1);
    T54 = _mm_packs_epi32(T42, T43);

    Tab = _mm_load_si128((__m128i*)tab_dct_8[3]);
    T44 = _mm_madd_epi16(T32, Tab);
    T45 = _mm_madd_epi16(T33, Tab);
    T44 = _mm_srai_epi32(_mm_add_epi32(T44, c_add1), SHIFT1);
    T45 = _mm_srai_epi32(_mm_add_epi32(T45, c_add1), SHIFT1);
    T52 = _mm_packs_epi32(T44, T45);

    Tab = _mm_load_si128((__m128i*)tab_dct_8[4]);
    T46 = _mm_madd_epi16(T32, Tab);
    T47 = _mm_madd_epi16(T33, Tab);
    T46 = _mm_srai_epi32(_mm_add_epi32(T46, c_add1), SHIFT1);
    T47 = _mm_srai_epi32(_mm_add_epi32(T47, c_add1), SHIFT1);
    T56 = _mm_packs_epi32(T46, T47);

    Tab = _mm_load_si128((__m128i*)tab_dct_8[5]);
    T40 = _mm_madd_epi16(T24, Tab);
    T41 = _mm_madd_epi16(T25, Tab);
    T42 = _mm_madd_epi16(T26, Tab);
    T43 = _mm_madd_epi16(T27, Tab);
    T40 = _mm_hadd_epi32(T40, T41);
    T42 = _mm_hadd_epi32(T42, T43);
    T40 = _mm_srai_epi32(_mm_add_epi32(T40, c_add1), SHIFT1);
    T42 = _mm_srai_epi32(_mm_add_epi32(T42, c_add1), SHIFT1);
    T51 = _mm_packs_epi32(T40, T42);

    Tab = _mm_load_si128((__m128i*)tab_dct_8[6]);
    T40 = _mm_madd_epi16(T24, Tab);
    T41 = _mm_madd_epi16(T25, Tab);
    T42 = _mm_madd_epi16(T26, Tab);
    T43 = _mm_madd_epi16(T27, Tab);
    T40 = _mm_hadd_epi32(T40, T41);
    T42 = _mm_hadd_epi32(T42, T43);
    T40 = _mm_srai_epi32(_mm_add_epi32(T40, c_add1), SHIFT1);
    T42 = _mm_srai_epi32(_mm_add_epi32(T42, c_add1), SHIFT1);
    T53 = _mm_packs_epi32(T40, T42);

    Tab = _mm_load_si128((__m128i*)tab_dct_8[7]);
    T40 = _mm_madd_epi16(T24, Tab);
    T41 = _mm_madd_epi16(T25, Tab);
    T42 = _mm_madd_epi16(T26, Tab);
    T43 = _mm_madd_epi16(T27, Tab);
    T40 = _mm_hadd_epi32(T40, T41);
    T42 = _mm_hadd_epi32(T42, T43);
    T40 = _mm_srai_epi32(_mm_add_epi32(T40, c_add1), SHIFT1);
    T42 = _mm_srai_epi32(_mm_add_epi32(T42, c_add1), SHIFT1);
    T55 = _mm_packs_epi32(T40, T42);

    Tab = _mm_load_si128((__m128i*)tab_dct_8[8]);
    T40 = _mm_madd_epi16(T24, Tab);
    T41 = _mm_madd_epi16(T25, Tab);
    T42 = _mm_madd_epi16(T26, Tab);
    T43 = _mm_madd_epi16(T27, Tab);
    T40 = _mm_hadd_epi32(T40, T41);
    T42 = _mm_hadd_epi32(T42, T43);
    T40 = _mm_srai_epi32(_mm_add_epi32(T40, c_add1), SHIFT1);
    T42 = _mm_srai_epi32(_mm_add_epi32(T42, c_add1), SHIFT1);
    T57 = _mm_packs_epi32(T40, T42);

    Tab = _mm_load_si128((__m128i*)tab_dct_8[0]);
    T10 = _mm_shuffle_epi8(T50, Tab);    // [05 02 06 01 04 03 07 00]
    T11 = _mm_shuffle_epi8(T51, Tab);
    T12 = _mm_shuffle_epi8(T52, Tab);
    T13 = _mm_shuffle_epi8(T53, Tab);
    T14 = _mm_shuffle_epi8(T54, Tab);
    T15 = _mm_shuffle_epi8(T55, Tab);
    T16 = _mm_shuffle_epi8(T56, Tab);
    T17 = _mm_shuffle_epi8(T57, Tab);

    // DCT2
    Tab = _mm_load_si128((__m128i*)tab_dct_8[1]);
    T20 = _mm_madd_epi16(T10, Tab);      // [64*s25_0 64*s16_0 64*s34_0 64*s07_0]
    T21 = _mm_madd_epi16(T11, Tab);      // [64*s25_1 64*s16_1 64*s34_1 64*s07_1]
    T22 = _mm_madd_epi16(T12, Tab);      // [64*s25_2 64*s16_2 64*s34_2 64*s07_2]
    T23 = _mm_madd_epi16(T13, Tab);      // [64*s25_3 64*s16_3 64*s34_3 64*s07_3]
    T24 = _mm_madd_epi16(T14, Tab);      // [64*s25_4 64*s16_4 64*s34_4 64*s07_4]
    T25 = _mm_madd_epi16(T15, Tab);      // [64*s25_5 64*s16_5 64*s34_5 64*s07_5]
    T26 = _mm_madd_epi16(T16, Tab);      // [64*s25_6 64*s16_6 64*s34_6 64*s07_6]
    T27 = _mm_madd_epi16(T17, Tab);      // [64*s25_7 64*s16_7 64*s34_7 64*s07_7]

    T30 = _mm_hadd_epi32(T20, T21); // [64*(s16+s25)_1 64*(s07+s34)_1 64*(s16+s25)_0 64*(s07+s34)_0]
    T31 = _mm_hadd_epi32(T22, T23); // [64*(s16+s25)_3 64*(s07+s34)_3 64*(s16+s25)_2 64*(s07+s34)_2]
    T32 = _mm_hadd_epi32(T24, T25); // [64*(s16+s25)_5 64*(s07+s34)_5 64*(s16+s25)_4 64*(s07+s34)_4]
    T33 = _mm_hadd_epi32(T26, T27); // [64*(s16+s25)_7 64*(s07+s34)_7 64*(s16+s25)_6 64*(s07+s34)_6]

    T40 = _mm_hadd_epi32(T30, T31); // [64*((s07+s34)+(s16+s25))_3 64*((s07+s34)+(s16+s25))_2 64*((s07+s34)+(s16+s25))_1 64*((s07+s34)+(s16+s25))_0]
    T41 = _mm_hadd_epi32(T32, T33); // [64*((s07+s34)+(s16+s25))_7 64*((s07+s34)+(s16+s25))_6 64*((s07+s34)+(s16+s25))_5 64*((s07+s34)+(s16+s25))_4]
    T42 = _mm_hsub_epi32(T30, T31); // [64*((s07+s34)-(s16+s25))_3 64*((s07+s34)-(s16+s25))_2 64*((s07+s34)-(s16+s25))_1 64*((s07+s34)-(s16+s25))_0]
    T43 = _mm_hsub_epi32(T32, T33); // [64*((s07+s34)-(s16+s25))_7 64*((s07+s34)-(s16+s25))_6 64*((s07+s34)-(s16+s25))_5 64*((s07+s34)-(s16+s25))_4]

    T50 = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2);
    T51 = _mm_srai_epi32(_mm_add_epi32(T41, c_add2), SHIFT2);
    T52 = _mm_srai_epi32(_mm_add_epi32(T42, c_add2), SHIFT2);
    T53 = _mm_srai_epi32(_mm_add_epi32(T43, c_add2), SHIFT2);

    T50 = _mm_packs_epi32(T50, T51);
    T52 = _mm_packs_epi32(T52, T53);
    _mm_store_si128((__m128i*)(dst + 0 * 8), T50);
    _mm_store_si128((__m128i*)(dst + 4 * 8), T52);

#define MAKE_ODD(tab, dstPos) \
    Tab = _mm_load_si128((__m128i const*)tab_dct_8[(tab)]); \
    T20 = _mm_madd_epi16(T10, Tab); \
    T21 = _mm_madd_epi16(T11, Tab); \
    T22 = _mm_madd_epi16(T12, Tab); \
    T23 = _mm_madd_epi16(T13, Tab); \
    T24 = _mm_madd_epi16(T14, Tab); \
    T25 = _mm_madd_epi16(T15, Tab); \
    T26 = _mm_madd_epi16(T16, Tab); \
    T27 = _mm_madd_epi16(T17, Tab); \
    T30 = _mm_hadd_epi32(T20, T21); \
    T31 = _mm_hadd_epi32(T22, T23); \
    T32 = _mm_hadd_epi32(T24, T25); \
    T33 = _mm_hadd_epi32(T26, T27); \
    T40 = _mm_hadd_epi32(T30, T31); \
    T41 = _mm_hadd_epi32(T32, T33); \
    T50 = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2); \
    T51 = _mm_srai_epi32(_mm_add_epi32(T41, c_add2), SHIFT2); \
    T50 = _mm_packs_epi32(T50, T51); \
    _mm_store_si128((__m128i*)(dst + (dstPos)* 8), T50);

    MAKE_ODD(9, 2);
    MAKE_ODD(10, 6);
    MAKE_ODD(11, 1);
    MAKE_ODD(12, 3);
    MAKE_ODD(13, 5);
    MAKE_ODD(14, 7);
#undef MAKE_ODD
}


/* ---------------------------------------------------------------------------
 */
void dct_c_16x4_sse128(const coeff_t *src, coeff_t *dst, int i_src)
{
    const int shift1 = B16X16_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT;
    const int shift2 = B16X16_IN_BIT + FACTO_BIT - 2;
    const int ADD1 = (1 << shift1) >> 1;
    const int ADD2 = (1 << shift2) >> 1;
    const __m128i c_2 = _mm_set1_epi32(ADD1);        // TODO: shift1 = 2
    const __m128i k_ROUNDING2 = _mm_set1_epi32(ADD2);

    __m128i T00A, T01A, T02A, T03A, T00B, T01B, T02B, T03B;
    __m128i T10, T11, T12, T13;
    __m128i T20, T21, T22, T23;
    __m128i T30, T31, T32, T33;
    __m128i T40, T41, T44, T45;
    __m128i T50, T52;
    __m128i T60, T61, T62, T63;
    __m128i T70;
    __m128i r0, r1, t0, t1, u0, u1, u2, u3, v0, v1, v2, v3, w0, w1, w2, w3;
    __m128i res0, res1, res2, res3, res4, res5, res6, res7;
    __m128i d0, d1, d2, d3, d4, d5, d6, d7;

    __m128i im[16];
    __m128i tmpZero = _mm_setzero_si128();

    //////// DCT1 16x4->4x16 ///////
    //input data
    T00A = _mm_loadu_si128((__m128i*)&src[0 * i_src + 0]);    // [07 06 05 04 03 02 01 00]
    T00B = _mm_loadu_si128((__m128i*)&src[0 * i_src + 8]);    // [0F 0E 0D 0C 0B 0A 09 08]
    T01A = _mm_loadu_si128((__m128i*)&src[1 * i_src + 0]);    // [17 16 15 14 13 12 11 10]
    T01B = _mm_loadu_si128((__m128i*)&src[1 * i_src + 8]);    // [1F 1E 1D 1C 1B 1A 19 18]
    T02A = _mm_loadu_si128((__m128i*)&src[2 * i_src + 0]);    // [27 26 25 24 23 22 21 20]
    T02B = _mm_loadu_si128((__m128i*)&src[2 * i_src + 8]);    // [2F 2E 2D 2C 2B 2A 29 28]
    T03A = _mm_loadu_si128((__m128i*)&src[3 * i_src + 0]);    // [37 36 35 34 33 32 31 30]
    T03B = _mm_loadu_si128((__m128i*)&src[3 * i_src + 8]);    // [3F 3E 3D 3C 3B 3A 39 38]

    //shuffle
    T00B = _mm_shuffle_epi8(T00B, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
    T01B = _mm_shuffle_epi8(T01B, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
    T02B = _mm_shuffle_epi8(T02B, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
    T03B = _mm_shuffle_epi8(T03B, _mm_load_si128((__m128i*)tab_dct_16_0[0]));

    T10 = _mm_add_epi16(T00A, T00B);
    T11 = _mm_add_epi16(T01A, T01B);
    T12 = _mm_add_epi16(T02A, T02B);
    T13 = _mm_add_epi16(T03A, T03B);
    T20 = _mm_sub_epi16(T00A, T00B);
    T21 = _mm_sub_epi16(T01A, T01B);
    T22 = _mm_sub_epi16(T02A, T02B);
    T23 = _mm_sub_epi16(T03A, T03B);

    T30 = _mm_shuffle_epi8(T10, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
    T31 = _mm_shuffle_epi8(T11, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
    T32 = _mm_shuffle_epi8(T12, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
    T33 = _mm_shuffle_epi8(T13, _mm_load_si128((__m128i*)tab_dct_16_0[1]));

    T40 = _mm_hadd_epi16(T30, T31);
    T41 = _mm_hadd_epi16(T32, T33);
    T44 = _mm_hsub_epi16(T30, T31);
    T45 = _mm_hsub_epi16(T32, T33);

    T50 = _mm_hadd_epi16(T40, T41);
    T52 = _mm_hsub_epi16(T40, T41);

    T60 = _mm_madd_epi16(T50, _mm_load_si128((__m128i*)tab_dct_8[1]));
    T60 = _mm_srai_epi32(_mm_add_epi32(T60, c_2), shift1);
    T70 = _mm_packs_epi32(T60, tmpZero);
    im[0] = T70;
    //_mm_storel_epi64((__m128i*)&dst[0 * 4], T70);

    T60 = _mm_madd_epi16(T50, _mm_load_si128((__m128i*)tab_dct_8[2]));
    T60 = _mm_srai_epi32(_mm_add_epi32(T60, c_2), shift1);
    T70 = _mm_packs_epi32(T60, tmpZero);
    im[8] = T70;
    //_mm_storel_epi64((__m128i*)&dst[8 * 4], T70);

    T60 = _mm_madd_epi16(T52, _mm_load_si128((__m128i*)tab_dct_8[3]));
    T60 = _mm_srai_epi32(_mm_add_epi32(T60, c_2), shift1);
    T70 = _mm_packs_epi32(T60, tmpZero);
    im[4] = T70;
    //_mm_storel_epi64((__m128i*)&dst[4 * 4], T70);

    T60 = _mm_madd_epi16(T52, _mm_load_si128((__m128i*)tab_dct_8[4]));
    T60 = _mm_srai_epi32(_mm_add_epi32(T60, c_2), shift1);
    T70 = _mm_packs_epi32(T60, tmpZero);
    im[12] = T70;
    //_mm_storel_epi64((__m128i*)&dst[12 * 4], T70);

    T60 = _mm_madd_epi16(T44, _mm_load_si128((__m128i*)tab_dct_8[5]));
    T61 = _mm_madd_epi16(T45, _mm_load_si128((__m128i*)tab_dct_8[5]));
    T60 = _mm_hadd_epi32(T60, T61);
    T60 = _mm_srai_epi32(_mm_add_epi32(T60, c_2), shift1);
    T70 = _mm_packs_epi32(T60, tmpZero);
    im[2] = T70;
    //_mm_storel_epi64((__m128i*)&dst[2 * 4], T70);

    T60 = _mm_madd_epi16(T44, _mm_load_si128((__m128i*)tab_dct_8[6]));
    T61 = _mm_madd_epi16(T45, _mm_load_si128((__m128i*)tab_dct_8[6]));
    T60 = _mm_hadd_epi32(T60, T61);
    T60 = _mm_srai_epi32(_mm_add_epi32(T60, c_2), shift1);
    T70 = _mm_packs_epi32(T60, tmpZero);
    im[6] = T70;
    //_mm_storel_epi64((__m128i*)&dst[6 * 4], T70);

    T60 = _mm_madd_epi16(T44, _mm_load_si128((__m128i*)tab_dct_8[7]));
    T61 = _mm_madd_epi16(T45, _mm_load_si128((__m128i*)tab_dct_8[7]));
    T60 = _mm_hadd_epi32(T60, T61);
    T60 = _mm_srai_epi32(_mm_add_epi32(T60, c_2), shift1);
    T70 = _mm_packs_epi32(T60, tmpZero);
    im[10] = T70;
    //_mm_storel_epi64((__m128i*)&dst[10 * 4], T70);

    T60 = _mm_madd_epi16(T44, _mm_load_si128((__m128i*)tab_dct_8[8]));
    T61 = _mm_madd_epi16(T45, _mm_load_si128((__m128i*)tab_dct_8[8]));
    T60 = _mm_hadd_epi32(T60, T61);
    T60 = _mm_srai_epi32(_mm_add_epi32(T60, c_2), shift1);
    T70 = _mm_packs_epi32(T60, T61);
    im[14] = T70;
    //_mm_storel_epi64((__m128i*)&dst[14 * 4], T70);

#define MAKE_ODD(tab, dstPos) \
    T60 = _mm_madd_epi16(T20, _mm_load_si128((__m128i*)tab_dct_16_1[(tab)])); \
    T61 = _mm_madd_epi16(T21, _mm_load_si128((__m128i*)tab_dct_16_1[(tab)])); \
    T62 = _mm_madd_epi16(T22, _mm_load_si128((__m128i*)tab_dct_16_1[(tab)])); \
    T63 = _mm_madd_epi16(T23, _mm_load_si128((__m128i*)tab_dct_16_1[(tab)])); \
    T60 = _mm_hadd_epi32(T60, T61); \
    T61 = _mm_hadd_epi32(T62, T63); \
    T60 = _mm_hadd_epi32(T60, T61); \
    T60 = _mm_srai_epi32(_mm_add_epi32(T60, c_2), shift1); \
    T70 = _mm_packs_epi32(T60, tmpZero); \
    im[dstPos] = T70;
    //_mm_storel_epi64((__m128i*)&dst[(dstPos)* 4], T70);

    MAKE_ODD(0, 1);
    MAKE_ODD(1, 3);
    MAKE_ODD(2, 5);
    MAKE_ODD(3, 7);
    MAKE_ODD(4, 9);
    MAKE_ODD(5, 11);
    MAKE_ODD(6, 13);
    MAKE_ODD(7, 15);
#undef MAKE_ODD

    //////// DCT2 16x4->4x16 ///////
    //1st 4x4
    t0 = _mm_unpacklo_epi64(im[0], im[1]);
    t1 = _mm_unpacklo_epi64(im[2], im[3]);

    r0 = _mm_madd_epi16(t0, _mm_load_si128((__m128i*)tab_dct1_4[0]));
    r1 = _mm_madd_epi16(t1, _mm_load_si128((__m128i*)tab_dct1_4[0]));
    u0 = _mm_hadd_epi32(r0, r1);

    r0 = _mm_madd_epi16(t0, _mm_load_si128((__m128i*)tab_dct1_4[1]));
    r1 = _mm_madd_epi16(t1, _mm_load_si128((__m128i*)tab_dct1_4[1]));
    u1 = _mm_hadd_epi32(r0, r1);

    r0 = _mm_madd_epi16(t0, _mm_load_si128((__m128i*)tab_dct1_4[2]));
    r1 = _mm_madd_epi16(t1, _mm_load_si128((__m128i*)tab_dct1_4[2]));
    u2 = _mm_hadd_epi32(r0, r1);

    r0 = _mm_madd_epi16(t0, _mm_load_si128((__m128i*)tab_dct1_4[3]));
    r1 = _mm_madd_epi16(t1, _mm_load_si128((__m128i*)tab_dct1_4[3]));
    u3 = _mm_hadd_epi32(r0, r1);

    v0 = _mm_add_epi32(u0, k_ROUNDING2);
    v1 = _mm_add_epi32(u1, k_ROUNDING2);
    v2 = _mm_add_epi32(u2, k_ROUNDING2);
    v3 = _mm_add_epi32(u3, k_ROUNDING2);
    w0 = _mm_srai_epi32(v0, shift2);
    w1 = _mm_srai_epi32(v1, shift2);
    w2 = _mm_srai_epi32(v2, shift2);
    w3 = _mm_srai_epi32(v3, shift2);
    res0 = _mm_packs_epi32(w0, w1);
    res1 = _mm_packs_epi32(w2, w3);

    //2nd 4x4
    t0 = _mm_unpacklo_epi64(im[4], im[5]);
    t1 = _mm_unpacklo_epi64(im[6], im[7]);

    r0 = _mm_madd_epi16(t0, _mm_load_si128((__m128i*)tab_dct1_4[0]));
    r1 = _mm_madd_epi16(t1, _mm_load_si128((__m128i*)tab_dct1_4[0]));
    u0 = _mm_hadd_epi32(r0, r1);

    r0 = _mm_madd_epi16(t0, _mm_load_si128((__m128i*)tab_dct1_4[1]));
    r1 = _mm_madd_epi16(t1, _mm_load_si128((__m128i*)tab_dct1_4[1]));
    u1 = _mm_hadd_epi32(r0, r1);

    r0 = _mm_madd_epi16(t0, _mm_load_si128((__m128i*)tab_dct1_4[2]));
    r1 = _mm_madd_epi16(t1, _mm_load_si128((__m128i*)tab_dct1_4[2]));
    u2 = _mm_hadd_epi32(r0, r1);

    r0 = _mm_madd_epi16(t0, _mm_load_si128((__m128i*)tab_dct1_4[3]));
    r1 = _mm_madd_epi16(t1, _mm_load_si128((__m128i*)tab_dct1_4[3]));
    u3 = _mm_hadd_epi32(r0, r1);

    v0 = _mm_add_epi32(u0, k_ROUNDING2);
    v1 = _mm_add_epi32(u1, k_ROUNDING2);
    v2 = _mm_add_epi32(u2, k_ROUNDING2);
    v3 = _mm_add_epi32(u3, k_ROUNDING2);
    w0 = _mm_srai_epi32(v0, shift2);
    w1 = _mm_srai_epi32(v1, shift2);
    w2 = _mm_srai_epi32(v2, shift2);
    w3 = _mm_srai_epi32(v3, shift2);
    res2 = _mm_packs_epi32(w0, w1);
    res3 = _mm_packs_epi32(w2, w3);

    //3rd 4x4
    t0 = _mm_unpacklo_epi64(im[8], im[9]);
    t1 = _mm_unpacklo_epi64(im[10], im[11]);

    r0 = _mm_madd_epi16(t0, _mm_load_si128((__m128i*)tab_dct1_4[0]));
    r1 = _mm_madd_epi16(t1, _mm_load_si128((__m128i*)tab_dct1_4[0]));
    u0 = _mm_hadd_epi32(r0, r1);

    r0 = _mm_madd_epi16(t0, _mm_load_si128((__m128i*)tab_dct1_4[1]));
    r1 = _mm_madd_epi16(t1, _mm_load_si128((__m128i*)tab_dct1_4[1]));
    u1 = _mm_hadd_epi32(r0, r1);

    r0 = _mm_madd_epi16(t0, _mm_load_si128((__m128i*)tab_dct1_4[2]));
    r1 = _mm_madd_epi16(t1, _mm_load_si128((__m128i*)tab_dct1_4[2]));
    u2 = _mm_hadd_epi32(r0, r1);

    r0 = _mm_madd_epi16(t0, _mm_load_si128((__m128i*)tab_dct1_4[3]));
    r1 = _mm_madd_epi16(t1, _mm_load_si128((__m128i*)tab_dct1_4[3]));
    u3 = _mm_hadd_epi32(r0, r1);

    v0 = _mm_add_epi32(u0, k_ROUNDING2);
    v1 = _mm_add_epi32(u1, k_ROUNDING2);
    v2 = _mm_add_epi32(u2, k_ROUNDING2);
    v3 = _mm_add_epi32(u3, k_ROUNDING2);
    w0 = _mm_srai_epi32(v0, shift2);
    w1 = _mm_srai_epi32(v1, shift2);
    w2 = _mm_srai_epi32(v2, shift2);
    w3 = _mm_srai_epi32(v3, shift2);
    res4 = _mm_packs_epi32(w0, w1);
    res5 = _mm_packs_epi32(w2, w3);

    //4th 4x4
    t0 = _mm_unpacklo_epi64(im[12], im[13]);
    t1 = _mm_unpacklo_epi64(im[14], im[15]);

    r0 = _mm_madd_epi16(t0, _mm_load_si128((__m128i*)tab_dct1_4[0]));
    r1 = _mm_madd_epi16(t1, _mm_load_si128((__m128i*)tab_dct1_4[0]));
    u0 = _mm_hadd_epi32(r0, r1);

    r0 = _mm_madd_epi16(t0, _mm_load_si128((__m128i*)tab_dct1_4[1]));
    r1 = _mm_madd_epi16(t1, _mm_load_si128((__m128i*)tab_dct1_4[1]));
    u1 = _mm_hadd_epi32(r0, r1);

    r0 = _mm_madd_epi16(t0, _mm_load_si128((__m128i*)tab_dct1_4[2]));
    r1 = _mm_madd_epi16(t1, _mm_load_si128((__m128i*)tab_dct1_4[2]));
    u2 = _mm_hadd_epi32(r0, r1);

    r0 = _mm_madd_epi16(t0, _mm_load_si128((__m128i*)tab_dct1_4[3]));
    r1 = _mm_madd_epi16(t1, _mm_load_si128((__m128i*)tab_dct1_4[3]));
    u3 = _mm_hadd_epi32(r0, r1);

    v0 = _mm_add_epi32(u0, k_ROUNDING2);
    v1 = _mm_add_epi32(u1, k_ROUNDING2);
    v2 = _mm_add_epi32(u2, k_ROUNDING2);
    v3 = _mm_add_epi32(u3, k_ROUNDING2);
    w0 = _mm_srai_epi32(v0, shift2);
    w1 = _mm_srai_epi32(v1, shift2);
    w2 = _mm_srai_epi32(v2, shift2);
    w3 = _mm_srai_epi32(v3, shift2);
    res6 = _mm_packs_epi32(w0, w1);
    res7 = _mm_packs_epi32(w2, w3);

    //store
    d0 = _mm_unpacklo_epi64(res0, res2);
    d1 = _mm_unpacklo_epi64(res4, res6);
    d2 = _mm_unpackhi_epi64(res0, res2);
    d3 = _mm_unpackhi_epi64(res4, res6);
    d4 = _mm_unpacklo_epi64(res1, res3);
    d5 = _mm_unpacklo_epi64(res5, res7);
    d6 = _mm_unpackhi_epi64(res1, res3);
    d7 = _mm_unpackhi_epi64(res5, res7);
    _mm_storeu_si128((__m128i *)(dst + 0), d0);
    _mm_storeu_si128((__m128i *)(dst + 8), d1);
    _mm_storeu_si128((__m128i *)(dst + 16), d2);
    _mm_storeu_si128((__m128i *)(dst + 24), d3);
    _mm_storeu_si128((__m128i *)(dst + 32), d4);
    _mm_storeu_si128((__m128i *)(dst + 40), d5);
    _mm_storeu_si128((__m128i *)(dst + 48), d6);
    _mm_storeu_si128((__m128i *)(dst + 56), d7);
}

/* ---------------------------------------------------------------------------
 */
void dct_c_4x16_sse128(const coeff_t *src, coeff_t *dst, int i_src)
{
    const int SHIFT1 = B16X16_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT - 2;
    const int ADD1 = (1 << SHIFT1) >> 1;
    const int SHIFT2 = B16X16_IN_BIT + FACTO_BIT;
    const int ADD2 = (1 << SHIFT2) >> 1;
    const __m128i c_add1 = _mm_set1_epi32(ADD1);
    const __m128i c_add2 = _mm_set1_epi32(ADD2);

    const __m128i k_p32_p32 = _mm_set1_epi16(32);
    const __m128i k_p32_m32 = pair_set_epi16(32, -32);
    const __m128i k_p17_p42 = pair_set_epi16(17, 42);
    const __m128i k_m42_p17 = pair_set_epi16(-42, 17);
    __m128i in[16];
    __m128i tr00, tr01;
    __m128i r0, r1, r2, r3, t0, t2;
    __m128i u10, u11, u12, u13, u20, u21, u22, u23;
    __m128i T00A, T01A, T02A, T03A, T00B, T01B, T02B, T03B;
    __m128i T10, T11, T12, T13, T14, T15, T16, T17;
    __m128i T20, T21, T22, T23, T24, T25, T26, T27;
    __m128i T30, T31, T32, T33;
    __m128i T40, T41;
    __m128i T70;
    __m128i tmpZero = _mm_setzero_si128();

    __m128i tab_dct_16_02 = _mm_loadu_si128((__m128i*)tab_dct_16_0[2]);
    __m128i tab_dct_16_03 = _mm_loadu_si128((__m128i*)tab_dct_16_0[3]);
    __m128i tab_dct_8_1_ = _mm_loadu_si128((__m128i*)tab_dct_8[1]);
    __m128i tab_dct_16_18 = _mm_loadu_si128((__m128i*)tab_dct_16_1[8]);
    __m128i tab_dct_16_19 = _mm_loadu_si128((__m128i*)tab_dct_16_1[9]);
    __m128i tab_dct_16_110 = _mm_loadu_si128((__m128i*)tab_dct_16_1[10]);
    __m128i tab_dct_16_111 = _mm_loadu_si128((__m128i*)tab_dct_16_1[11]);
    __m128i tab_dct_16_112 = _mm_loadu_si128((__m128i*)tab_dct_16_1[12]);
    __m128i tab_dct_16_113 = _mm_loadu_si128((__m128i*)tab_dct_16_1[13]);


    ///// DCT1 4x16->16x4//////
    in[0] = _mm_load_si128((const __m128i *)(src + 0 * i_src));
    in[1] = _mm_load_si128((const __m128i *)(src + 2 * i_src));
    in[4] = _mm_load_si128((const __m128i *)(src + 4 * i_src));
    in[5] = _mm_load_si128((const __m128i *)(src + 6 * i_src));
    in[8] = _mm_load_si128((const __m128i *)(src + 8 * i_src));
    in[9] = _mm_load_si128((const __m128i *)(src + 10 * i_src));
    in[12] = _mm_load_si128((const __m128i *)(src + 12 * i_src));
    in[13] = _mm_load_si128((const __m128i *)(src + 14 * i_src));

    //transpose input data
    //1st 4x4
    tr00 = _mm_shuffle_epi32(in[0], 0xD8);//00 01 04 05 02 03 06 07
    tr01 = _mm_shuffle_epi32(in[1], 0xD8);//08 09 12 13 10 11 14 15
    r0 = _mm_unpacklo_epi64(tr00, tr01);//00 01 04 05 08 09 12 13
    r1 = _mm_unpackhi_epi64(tr00, tr01);//02 03 06 07 10 11 14 15
    r2 = _mm_shufflehi_epi16(r0, 0xB1);
    r2 = _mm_shufflelo_epi16(r2, 0xB1);//01 00 05 04 09 08 13 12
    r3 = _mm_shufflehi_epi16(r1, 0xB1);
    r3 = _mm_shufflelo_epi16(r3, 0xB1);//03 02 07 06 11 10 15 14
    t0 = _mm_add_epi16(r0, r3);//00+03 01+02 04+07 05+06 08+11 09+10 12+15 13+14
    t2 = _mm_sub_epi16(r2, r1);

    u10 = _mm_madd_epi16(t0, k_p32_p32);//(00+03)*32+(01+02)*32  (04+07)*32+(05+06)*32  (08+11)*32+(09+10)*32 (12+15)*32+(13+14)*32
    u12 = _mm_madd_epi16(t0, k_p32_m32);//(00+03)*32-(01+02)*32  (04+07)*32-(05+06)*32  (08+11)*32-(09+10)*32 (12+15)*32-(13+14)*32
    u11 = _mm_madd_epi16(t2, k_p17_p42);
    u13 = _mm_madd_epi16(t2, k_m42_p17);

    //ÒÆÎ»²¹³¥
    u10 = _mm_srai_epi32(_mm_add_epi32(u10, c_add1), SHIFT1);
    u11 = _mm_srai_epi32(_mm_add_epi32(u11, c_add1), SHIFT1);
    u12 = _mm_srai_epi32(_mm_add_epi32(u12, c_add1), SHIFT1);
    u13 = _mm_srai_epi32(_mm_add_epi32(u13, c_add1), SHIFT1);


    //2nd 4x4
    tr00 = _mm_shuffle_epi32(in[4], 0xD8);
    tr01 = _mm_shuffle_epi32(in[5], 0xD8);
    r0 = _mm_unpacklo_epi64(tr00, tr01);
    r1 = _mm_unpackhi_epi64(tr00, tr01);
    r2 = _mm_shufflehi_epi16(r0, 0xB1);
    r2 = _mm_shufflelo_epi16(r2, 0xB1);
    r3 = _mm_shufflehi_epi16(r1, 0xB1);
    r3 = _mm_shufflelo_epi16(r3, 0xB1);
    t0 = _mm_add_epi16(r0, r3);
    t2 = _mm_sub_epi16(r2, r1);

    u20 = _mm_madd_epi16(t0, k_p32_p32);
    u22 = _mm_madd_epi16(t0, k_p32_m32);
    u21 = _mm_madd_epi16(t2, k_p17_p42);
    u23 = _mm_madd_epi16(t2, k_m42_p17);

    //ÒÆÎ»²¹³¥
    u20 = _mm_srai_epi32(_mm_add_epi32(u20, c_add1), SHIFT1);
    u21 = _mm_srai_epi32(_mm_add_epi32(u21, c_add1), SHIFT1);
    u22 = _mm_srai_epi32(_mm_add_epi32(u22, c_add1), SHIFT1);
    u23 = _mm_srai_epi32(_mm_add_epi32(u23, c_add1), SHIFT1);

    T00A = _mm_packs_epi32(u10, u20);
    T01A = _mm_packs_epi32(u11, u21);
    T02A = _mm_packs_epi32(u12, u22);
    T03A = _mm_packs_epi32(u13, u23);

    //3rd 4x4
    tr00 = _mm_shuffle_epi32(in[8], 0xD8);
    tr01 = _mm_shuffle_epi32(in[9], 0xD8);
    r0 = _mm_unpacklo_epi64(tr00, tr01);
    r1 = _mm_unpackhi_epi64(tr00, tr01);
    r2 = _mm_shufflehi_epi16(r0, 0xB1);
    r2 = _mm_shufflelo_epi16(r2, 0xB1);
    r3 = _mm_shufflehi_epi16(r1, 0xB1);
    r3 = _mm_shufflelo_epi16(r3, 0xB1);
    t0 = _mm_add_epi16(r0, r3);
    t2 = _mm_sub_epi16(r2, r1);

    u10 = _mm_madd_epi16(t0, k_p32_p32);
    u12 = _mm_madd_epi16(t0, k_p32_m32);
    u11 = _mm_madd_epi16(t2, k_p17_p42);
    u13 = _mm_madd_epi16(t2, k_m42_p17);

    //ÒÆÎ»²¹³¥
    u10 = _mm_srai_epi32(_mm_add_epi32(u10, c_add1), SHIFT1);
    u11 = _mm_srai_epi32(_mm_add_epi32(u11, c_add1), SHIFT1);
    u12 = _mm_srai_epi32(_mm_add_epi32(u12, c_add1), SHIFT1);
    u13 = _mm_srai_epi32(_mm_add_epi32(u13, c_add1), SHIFT1);

    //4th 4x4
    tr00 = _mm_shuffle_epi32(in[12], 0xD8);
    tr01 = _mm_shuffle_epi32(in[13], 0xD8);
    r0 = _mm_unpacklo_epi64(tr00, tr01);
    r1 = _mm_unpackhi_epi64(tr00, tr01);
    r2 = _mm_shufflehi_epi16(r0, 0xB1);
    r2 = _mm_shufflelo_epi16(r2, 0xB1);
    r3 = _mm_shufflehi_epi16(r1, 0xB1);
    r3 = _mm_shufflelo_epi16(r3, 0xB1);
    t0 = _mm_add_epi16(r0, r3);
    t2 = _mm_sub_epi16(r2, r1);

    u20 = _mm_madd_epi16(t0, k_p32_p32);
    u22 = _mm_madd_epi16(t0, k_p32_m32);
    u21 = _mm_madd_epi16(t2, k_p17_p42);
    u23 = _mm_madd_epi16(t2, k_m42_p17);
    //ÒÆÎ»²¹³¥
    u20 = _mm_srai_epi32(_mm_add_epi32(u20, c_add1), SHIFT1);
    u21 = _mm_srai_epi32(_mm_add_epi32(u21, c_add1), SHIFT1);
    u22 = _mm_srai_epi32(_mm_add_epi32(u22, c_add1), SHIFT1);
    u23 = _mm_srai_epi32(_mm_add_epi32(u23, c_add1), SHIFT1);


    T00B = _mm_packs_epi32(u10, u20);
    T01B = _mm_packs_epi32(u11, u21);
    T02B = _mm_packs_epi32(u12, u22);
    T03B = _mm_packs_epi32(u13, u23);

    ///// DCT2 16x4->4x16//////

    T00A = _mm_shuffle_epi8(T00A, tab_dct_16_02);//00 03 01 02 07 04 06 05
    T00B = _mm_shuffle_epi8(T00B, tab_dct_16_03);//17 14 16 15 10 13 11 12
    T01A = _mm_shuffle_epi8(T01A, tab_dct_16_02);
    T01B = _mm_shuffle_epi8(T01B, tab_dct_16_03);
    T02A = _mm_shuffle_epi8(T02A, tab_dct_16_02);
    T02B = _mm_shuffle_epi8(T02B, tab_dct_16_03);
    T03A = _mm_shuffle_epi8(T03A, tab_dct_16_02);
    T03B = _mm_shuffle_epi8(T03B, tab_dct_16_03);

    T10 = _mm_unpacklo_epi16(T00A, T00B);//00 17 03 14 01 16 02 15
    T11 = _mm_unpackhi_epi16(T00A, T00B);//07 10 04 13 06 11 05 12
    T12 = _mm_unpacklo_epi16(T01A, T01B);
    T13 = _mm_unpackhi_epi16(T01A, T01B);
    T14 = _mm_unpacklo_epi16(T02A, T02B);
    T15 = _mm_unpackhi_epi16(T02A, T02B);
    T16 = _mm_unpacklo_epi16(T03A, T03B);
    T17 = _mm_unpackhi_epi16(T03A, T03B);

    T20 = _mm_madd_epi16(T10, tab_dct_8_1_);//00+17 03+14 01+16 02+15 *32
    T21 = _mm_madd_epi16(T11, tab_dct_8_1_);//07+10 04+13 06+11 05+12 *32
    T22 = _mm_madd_epi16(T12, tab_dct_8_1_);
    T23 = _mm_madd_epi16(T13, tab_dct_8_1_);
    T24 = _mm_madd_epi16(T14, tab_dct_8_1_);
    T25 = _mm_madd_epi16(T15, tab_dct_8_1_);
    T26 = _mm_madd_epi16(T16, tab_dct_8_1_);
    T27 = _mm_madd_epi16(T17, tab_dct_8_1_);

    T30 = _mm_add_epi32(T20, T21);//00+17 + 07+10    03+14 + 04+13    01+16 + 06+11    02+15 + 05+12
    T31 = _mm_add_epi32(T22, T23);
    T32 = _mm_add_epi32(T24, T25);
    T33 = _mm_add_epi32(T26, T27);

    T30 = _mm_hadd_epi32(T30, T31);
    T31 = _mm_hadd_epi32(T32, T33);

    T40 = _mm_hadd_epi32(T30, T31);
    T41 = _mm_hsub_epi32(T30, T31);
    T40 = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2);
    T41 = _mm_srai_epi32(_mm_add_epi32(T41, c_add2), SHIFT2);

    T70 = _mm_packs_epi32(T40, tmpZero);
    _mm_storel_epi64((__m128i*)&dst[0 * 4], T70);
    T70 = _mm_packs_epi32(T41, tmpZero);
    _mm_storel_epi64((__m128i*)&dst[8 * 4], T70);

    T20 = _mm_madd_epi16(T10, tab_dct_16_18);
    T21 = _mm_madd_epi16(T11, tab_dct_16_18);
    T22 = _mm_madd_epi16(T12, tab_dct_16_18);
    T23 = _mm_madd_epi16(T13, tab_dct_16_18);
    T24 = _mm_madd_epi16(T14, tab_dct_16_18);
    T25 = _mm_madd_epi16(T15, tab_dct_16_18);
    T26 = _mm_madd_epi16(T16, tab_dct_16_18);
    T27 = _mm_madd_epi16(T17, tab_dct_16_18);

    T30 = _mm_add_epi32(T20, T21);
    T31 = _mm_add_epi32(T22, T23);
    T32 = _mm_add_epi32(T24, T25);
    T33 = _mm_add_epi32(T26, T27);

    T30 = _mm_hadd_epi32(T30, T31);
    T31 = _mm_hadd_epi32(T32, T33);

    T40 = _mm_hadd_epi32(T30, T31);
    T40 = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2);
    T70 = _mm_packs_epi32(T40, tmpZero);
    _mm_storel_epi64((__m128i*)&dst[4 * 4], T70);

    T20 = _mm_madd_epi16(T10, tab_dct_16_19);
    T21 = _mm_madd_epi16(T11, tab_dct_16_19);
    T22 = _mm_madd_epi16(T12, tab_dct_16_19);
    T23 = _mm_madd_epi16(T13, tab_dct_16_19);
    T24 = _mm_madd_epi16(T14, tab_dct_16_19);
    T25 = _mm_madd_epi16(T15, tab_dct_16_19);
    T26 = _mm_madd_epi16(T16, tab_dct_16_19);
    T27 = _mm_madd_epi16(T17, tab_dct_16_19);

    T30 = _mm_add_epi32(T20, T21);
    T31 = _mm_add_epi32(T22, T23);
    T32 = _mm_add_epi32(T24, T25);
    T33 = _mm_add_epi32(T26, T27);

    T30 = _mm_hadd_epi32(T30, T31);
    T31 = _mm_hadd_epi32(T32, T33);

    T40 = _mm_hadd_epi32(T30, T31);
    T40 = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2);
    T70 = _mm_packs_epi32(T40, tmpZero);
    _mm_storel_epi64((__m128i*)&dst[12 * 4], T70);

    T20 = _mm_madd_epi16(T10, tab_dct_16_110);
    T21 = _mm_madd_epi16(T11, tab_dct_16_110);
    T22 = _mm_madd_epi16(T12, tab_dct_16_110);
    T23 = _mm_madd_epi16(T13, tab_dct_16_110);
    T24 = _mm_madd_epi16(T14, tab_dct_16_110);
    T25 = _mm_madd_epi16(T15, tab_dct_16_110);
    T26 = _mm_madd_epi16(T16, tab_dct_16_110);
    T27 = _mm_madd_epi16(T17, tab_dct_16_110);

    T30 = _mm_sub_epi32(T20, T21);
    T31 = _mm_sub_epi32(T22, T23);
    T32 = _mm_sub_epi32(T24, T25);
    T33 = _mm_sub_epi32(T26, T27);

    T30 = _mm_hadd_epi32(T30, T31);
    T31 = _mm_hadd_epi32(T32, T33);

    T40 = _mm_hadd_epi32(T30, T31);
    T40 = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2);
    T70 = _mm_packs_epi32(T40, tmpZero);
    _mm_storel_epi64((__m128i*)&dst[2 * 4], T70);

    T20 = _mm_madd_epi16(T10, tab_dct_16_111);
    T21 = _mm_madd_epi16(T11, tab_dct_16_111);
    T22 = _mm_madd_epi16(T12, tab_dct_16_111);
    T23 = _mm_madd_epi16(T13, tab_dct_16_111);
    T24 = _mm_madd_epi16(T14, tab_dct_16_111);
    T25 = _mm_madd_epi16(T15, tab_dct_16_111);
    T26 = _mm_madd_epi16(T16, tab_dct_16_111);
    T27 = _mm_madd_epi16(T17, tab_dct_16_111);

    T30 = _mm_sub_epi32(T20, T21);
    T31 = _mm_sub_epi32(T22, T23);
    T32 = _mm_sub_epi32(T24, T25);
    T33 = _mm_sub_epi32(T26, T27);

    T30 = _mm_hadd_epi32(T30, T31);
    T31 = _mm_hadd_epi32(T32, T33);

    T40 = _mm_hadd_epi32(T30, T31);
    T40 = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2);
    T70 = _mm_packs_epi32(T40, tmpZero);
    _mm_storel_epi64((__m128i*)&dst[6 * 4], T70);

    T20 = _mm_madd_epi16(T10, tab_dct_16_112);
    T21 = _mm_madd_epi16(T11, tab_dct_16_112);
    T22 = _mm_madd_epi16(T12, tab_dct_16_112);
    T23 = _mm_madd_epi16(T13, tab_dct_16_112);
    T24 = _mm_madd_epi16(T14, tab_dct_16_112);
    T25 = _mm_madd_epi16(T15, tab_dct_16_112);
    T26 = _mm_madd_epi16(T16, tab_dct_16_112);
    T27 = _mm_madd_epi16(T17, tab_dct_16_112);

    T30 = _mm_sub_epi32(T20, T21);
    T31 = _mm_sub_epi32(T22, T23);
    T32 = _mm_sub_epi32(T24, T25);
    T33 = _mm_sub_epi32(T26, T27);

    T30 = _mm_hadd_epi32(T30, T31);
    T31 = _mm_hadd_epi32(T32, T33);

    T40 = _mm_hadd_epi32(T30, T31);
    T40 = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2);
    T70 = _mm_packs_epi32(T40, tmpZero);
    _mm_storel_epi64((__m128i*)&dst[10 * 4], T70);

    T20 = _mm_madd_epi16(T10, tab_dct_16_113);
    T21 = _mm_madd_epi16(T11, tab_dct_16_113);
    T22 = _mm_madd_epi16(T12, tab_dct_16_113);
    T23 = _mm_madd_epi16(T13, tab_dct_16_113);
    T24 = _mm_madd_epi16(T14, tab_dct_16_113);
    T25 = _mm_madd_epi16(T15, tab_dct_16_113);
    T26 = _mm_madd_epi16(T16, tab_dct_16_113);
    T27 = _mm_madd_epi16(T17, tab_dct_16_113);

    T30 = _mm_sub_epi32(T20, T21);
    T31 = _mm_sub_epi32(T22, T23);
    T32 = _mm_sub_epi32(T24, T25);
    T33 = _mm_sub_epi32(T26, T27);

    T30 = _mm_hadd_epi32(T30, T31);
    T31 = _mm_hadd_epi32(T32, T33);

    T40 = _mm_hadd_epi32(T30, T31);
    T40 = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2);
    T70 = _mm_packs_epi32(T40, tmpZero);
    _mm_storel_epi64((__m128i*)&dst[14 * 4], T70);

    __m128i tab_dct_16_1_tab;
    __m128i tab_dct_16_1_tab1;
#define MAKE_ODD(tab, dstPos) \
    tab_dct_16_1_tab = _mm_loadu_si128((__m128i const*)tab_dct_16_1[(tab)]); \
    tab_dct_16_1_tab1 = _mm_loadu_si128((__m128i const*)tab_dct_16_1[(tab + 1)]); \
    T20 = _mm_madd_epi16(T10, tab_dct_16_1_tab);       /* [*O2_0 *O1_0 *O3_0 *O0_0] */ \
    T21 = _mm_madd_epi16(T11, tab_dct_16_1_tab1);   /* [*O5_0 *O6_0 *O4_0 *O7_0] */ \
    T22 = _mm_madd_epi16(T12, tab_dct_16_1_tab); \
    T23 = _mm_madd_epi16(T13, tab_dct_16_1_tab1); \
    T24 = _mm_madd_epi16(T14, tab_dct_16_1_tab); \
    T25 = _mm_madd_epi16(T15, tab_dct_16_1_tab1); \
    T26 = _mm_madd_epi16(T16, tab_dct_16_1_tab); \
    T27 = _mm_madd_epi16(T17, tab_dct_16_1_tab1); \
    \
    T30 = _mm_add_epi32(T20, T21); \
    T31 = _mm_add_epi32(T22, T23); \
    T32 = _mm_add_epi32(T24, T25); \
    T33 = _mm_add_epi32(T26, T27); \
    \
    T30 = _mm_hadd_epi32(T30, T31); \
    T31 = _mm_hadd_epi32(T32, T33); \
    \
    T40 = _mm_hadd_epi32(T30, T31); \
    T40 = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2); \
    \
    T70 = _mm_packs_epi32(T40, tmpZero); \
    _mm_storel_epi64((__m128i*)&dst[(dstPos)* 4], T70);

    MAKE_ODD(14, 1);
    MAKE_ODD(16, 3);
    MAKE_ODD(18, 5);
    MAKE_ODD(20, 7);
    MAKE_ODD(22, 9);
    MAKE_ODD(24, 11);
    MAKE_ODD(26, 13);
    MAKE_ODD(28, 15);
#undef MAKE_ODD
}

/* ---------------------------------------------------------------------------
 */
void dct_c_16x16_sse128(const coeff_t *src, coeff_t *dst, int i_src)
{
    const int SHIFT1 = B16X16_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT;
    const int SHIFT2 = B16X16_IN_BIT + FACTO_BIT;
    const int ADD1 = (1 << SHIFT1) >> 1;
    const int ADD2 = (1 << SHIFT2) >> 1;

    ALIGN32(int16_t tmp[16 * 16]);

    // Const
    __m128i c_add1  = _mm_set1_epi32(ADD1);
    __m128i c_add2  = _mm_set1_epi32(ADD2);

    __m128i T00A, T01A, T02A, T03A, T04A, T05A, T06A, T07A;
    __m128i T00B, T01B, T02B, T03B, T04B, T05B, T06B, T07B;
    __m128i T10, T11, T12, T13, T14, T15, T16, T17;
    __m128i T20, T21, T22, T23, T24, T25, T26, T27;
    __m128i T30, T31, T32, T33, T34, T35, T36, T37;
    __m128i T40, T41, T42, T43, T44, T45, T46, T47;
    __m128i T50, T51, T52, T53;
    __m128i T60, T61, T62, T63, T64, T65, T66, T67;
    __m128i T70;

    int i;

    // DCT1
    for (i = 0; i < 16; i += 8) {
        T00A = _mm_load_si128((__m128i*)(src + (i + 0) * i_src + 0)); // [07 06 05 04 03 02 01 00]
        T00B = _mm_load_si128((__m128i*)(src + (i + 0) * i_src + 8)); // [0F 0E 0D 0C 0B 0A 09 08]
        T01A = _mm_load_si128((__m128i*)(src + (i + 1) * i_src + 0)); // [17 16 15 14 13 12 11 10]
        T01B = _mm_load_si128((__m128i*)(src + (i + 1) * i_src + 8)); // [1F 1E 1D 1C 1B 1A 19 18]
        T02A = _mm_load_si128((__m128i*)(src + (i + 2) * i_src + 0)); // [27 26 25 24 23 22 21 20]
        T02B = _mm_load_si128((__m128i*)(src + (i + 2) * i_src + 8)); // [2F 2E 2D 2C 2B 2A 29 28]
        T03A = _mm_load_si128((__m128i*)(src + (i + 3) * i_src + 0)); // [37 36 35 34 33 32 31 30]
        T03B = _mm_load_si128((__m128i*)(src + (i + 3) * i_src + 8)); // [3F 3E 3D 3C 3B 3A 39 38]
        T04A = _mm_load_si128((__m128i*)(src + (i + 4) * i_src + 0)); // [47 46 45 44 43 42 41 40]
        T04B = _mm_load_si128((__m128i*)(src + (i + 4) * i_src + 8)); // [4F 4E 4D 4C 4B 4A 49 48]
        T05A = _mm_load_si128((__m128i*)(src + (i + 5) * i_src + 0)); // [57 56 55 54 53 52 51 50]
        T05B = _mm_load_si128((__m128i*)(src + (i + 5) * i_src + 8)); // [5F 5E 5D 5C 5B 5A 59 58]
        T06A = _mm_load_si128((__m128i*)(src + (i + 6) * i_src + 0)); // [67 66 65 64 63 62 61 60]
        T06B = _mm_load_si128((__m128i*)(src + (i + 6) * i_src + 8)); // [6F 6E 6D 6C 6B 6A 69 68]
        T07A = _mm_load_si128((__m128i*)(src + (i + 7) * i_src + 0)); // [77 76 75 74 73 72 71 70]
        T07B = _mm_load_si128((__m128i*)(src + (i + 7) * i_src + 8)); // [7F 7E 7D 7C 7B 7A 79 78]

        T00B = _mm_shuffle_epi8(T00B, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
        T01B = _mm_shuffle_epi8(T01B, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
        T02B = _mm_shuffle_epi8(T02B, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
        T03B = _mm_shuffle_epi8(T03B, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
        T04B = _mm_shuffle_epi8(T04B, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
        T05B = _mm_shuffle_epi8(T05B, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
        T06B = _mm_shuffle_epi8(T06B, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
        T07B = _mm_shuffle_epi8(T07B, _mm_load_si128((__m128i*)tab_dct_16_0[0]));

        T10  = _mm_add_epi16(T00A, T00B);
        T11  = _mm_add_epi16(T01A, T01B);
        T12  = _mm_add_epi16(T02A, T02B);
        T13  = _mm_add_epi16(T03A, T03B);
        T14  = _mm_add_epi16(T04A, T04B);
        T15  = _mm_add_epi16(T05A, T05B);
        T16  = _mm_add_epi16(T06A, T06B);
        T17  = _mm_add_epi16(T07A, T07B);

        T20  = _mm_sub_epi16(T00A, T00B);
        T21  = _mm_sub_epi16(T01A, T01B);
        T22  = _mm_sub_epi16(T02A, T02B);
        T23  = _mm_sub_epi16(T03A, T03B);
        T24  = _mm_sub_epi16(T04A, T04B);
        T25  = _mm_sub_epi16(T05A, T05B);
        T26  = _mm_sub_epi16(T06A, T06B);
        T27  = _mm_sub_epi16(T07A, T07B);

        T30  = _mm_shuffle_epi8(T10, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T31  = _mm_shuffle_epi8(T11, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T32  = _mm_shuffle_epi8(T12, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T33  = _mm_shuffle_epi8(T13, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T34  = _mm_shuffle_epi8(T14, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T35  = _mm_shuffle_epi8(T15, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T36  = _mm_shuffle_epi8(T16, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T37  = _mm_shuffle_epi8(T17, _mm_load_si128((__m128i*)tab_dct_16_0[1]));

        T40  = _mm_hadd_epi16(T30, T31);
        T41  = _mm_hadd_epi16(T32, T33);
        T42  = _mm_hadd_epi16(T34, T35);
        T43  = _mm_hadd_epi16(T36, T37);
        T44  = _mm_hsub_epi16(T30, T31);
        T45  = _mm_hsub_epi16(T32, T33);
        T46  = _mm_hsub_epi16(T34, T35);
        T47  = _mm_hsub_epi16(T36, T37);

        T50  = _mm_hadd_epi16(T40, T41);
        T51  = _mm_hadd_epi16(T42, T43);
        T52  = _mm_hsub_epi16(T40, T41);
        T53  = _mm_hsub_epi16(T42, T43);

        T60  = _mm_madd_epi16(T50, _mm_load_si128((__m128i*)tab_dct_8[1]));
        T61  = _mm_madd_epi16(T51, _mm_load_si128((__m128i*)tab_dct_8[1]));
        T60  = _mm_srai_epi32(_mm_add_epi32(T60, c_add1), SHIFT1);
        T61  = _mm_srai_epi32(_mm_add_epi32(T61, c_add1), SHIFT1);
        T70  = _mm_packs_epi32(T60, T61);
        _mm_store_si128((__m128i*)(tmp + 0 * 16 + i), T70);

        T60  = _mm_madd_epi16(T50, _mm_load_si128((__m128i*)tab_dct_8[2]));
        T61  = _mm_madd_epi16(T51, _mm_load_si128((__m128i*)tab_dct_8[2]));
        T60  = _mm_srai_epi32(_mm_add_epi32(T60, c_add1), SHIFT1);
        T61  = _mm_srai_epi32(_mm_add_epi32(T61, c_add1), SHIFT1);
        T70  = _mm_packs_epi32(T60, T61);
        _mm_store_si128((__m128i*)(tmp + 8 * 16 + i), T70);

        T60  = _mm_madd_epi16(T52, _mm_load_si128((__m128i*)tab_dct_8[3]));
        T61  = _mm_madd_epi16(T53, _mm_load_si128((__m128i*)tab_dct_8[3]));
        T60  = _mm_srai_epi32(_mm_add_epi32(T60, c_add1), SHIFT1);
        T61  = _mm_srai_epi32(_mm_add_epi32(T61, c_add1), SHIFT1);
        T70  = _mm_packs_epi32(T60, T61);
        _mm_store_si128((__m128i*)(tmp + 4 * 16 + i), T70);

        T60  = _mm_madd_epi16(T52, _mm_load_si128((__m128i*)tab_dct_8[4]));
        T61  = _mm_madd_epi16(T53, _mm_load_si128((__m128i*)tab_dct_8[4]));
        T60  = _mm_srai_epi32(_mm_add_epi32(T60, c_add1), SHIFT1);
        T61  = _mm_srai_epi32(_mm_add_epi32(T61, c_add1), SHIFT1);
        T70  = _mm_packs_epi32(T60, T61);
        _mm_store_si128((__m128i*)(tmp + 12 * 16 + i), T70);

        T60  = _mm_madd_epi16(T44, _mm_load_si128((__m128i*)tab_dct_8[5]));
        T61  = _mm_madd_epi16(T45, _mm_load_si128((__m128i*)tab_dct_8[5]));
        T62  = _mm_madd_epi16(T46, _mm_load_si128((__m128i*)tab_dct_8[5]));
        T63  = _mm_madd_epi16(T47, _mm_load_si128((__m128i*)tab_dct_8[5]));
        T60  = _mm_hadd_epi32(T60, T61);
        T61  = _mm_hadd_epi32(T62, T63);
        T60  = _mm_srai_epi32(_mm_add_epi32(T60, c_add1), SHIFT1);
        T61  = _mm_srai_epi32(_mm_add_epi32(T61, c_add1), SHIFT1);
        T70  = _mm_packs_epi32(T60, T61);
        _mm_store_si128((__m128i*)(tmp + 2 * 16 + i), T70);

        T60  = _mm_madd_epi16(T44, _mm_load_si128((__m128i*)tab_dct_8[6]));
        T61  = _mm_madd_epi16(T45, _mm_load_si128((__m128i*)tab_dct_8[6]));
        T62  = _mm_madd_epi16(T46, _mm_load_si128((__m128i*)tab_dct_8[6]));
        T63  = _mm_madd_epi16(T47, _mm_load_si128((__m128i*)tab_dct_8[6]));
        T60  = _mm_hadd_epi32(T60, T61);
        T61  = _mm_hadd_epi32(T62, T63);
        T60  = _mm_srai_epi32(_mm_add_epi32(T60, c_add1), SHIFT1);
        T61  = _mm_srai_epi32(_mm_add_epi32(T61, c_add1), SHIFT1);
        T70  = _mm_packs_epi32(T60, T61);
        _mm_store_si128((__m128i*)(tmp + 6 * 16 + i), T70);

        T60  = _mm_madd_epi16(T44, _mm_load_si128((__m128i*)tab_dct_8[7]));
        T61  = _mm_madd_epi16(T45, _mm_load_si128((__m128i*)tab_dct_8[7]));
        T62  = _mm_madd_epi16(T46, _mm_load_si128((__m128i*)tab_dct_8[7]));
        T63  = _mm_madd_epi16(T47, _mm_load_si128((__m128i*)tab_dct_8[7]));
        T60  = _mm_hadd_epi32(T60, T61);
        T61  = _mm_hadd_epi32(T62, T63);
        T60  = _mm_srai_epi32(_mm_add_epi32(T60, c_add1), SHIFT1);
        T61  = _mm_srai_epi32(_mm_add_epi32(T61, c_add1), SHIFT1);
        T70  = _mm_packs_epi32(T60, T61);
        _mm_store_si128((__m128i*)(tmp + 10 * 16 + i), T70);

        T60  = _mm_madd_epi16(T44, _mm_load_si128((__m128i*)tab_dct_8[8]));
        T61  = _mm_madd_epi16(T45, _mm_load_si128((__m128i*)tab_dct_8[8]));
        T62  = _mm_madd_epi16(T46, _mm_load_si128((__m128i*)tab_dct_8[8]));
        T63  = _mm_madd_epi16(T47, _mm_load_si128((__m128i*)tab_dct_8[8]));
        T60  = _mm_hadd_epi32(T60, T61);
        T61  = _mm_hadd_epi32(T62, T63);
        T60  = _mm_srai_epi32(_mm_add_epi32(T60, c_add1), SHIFT1);
        T61  = _mm_srai_epi32(_mm_add_epi32(T61, c_add1), SHIFT1);
        T70  = _mm_packs_epi32(T60, T61);
        _mm_store_si128((__m128i*)(tmp + 14 * 16 + i), T70);

#define MAKE_ODD(tab, dstPos) \
    T60  = _mm_madd_epi16(T20, _mm_load_si128((__m128i*)tab_dct_16_1[(tab)])); \
    T61  = _mm_madd_epi16(T21, _mm_load_si128((__m128i*)tab_dct_16_1[(tab)])); \
    T62  = _mm_madd_epi16(T22, _mm_load_si128((__m128i*)tab_dct_16_1[(tab)])); \
    T63  = _mm_madd_epi16(T23, _mm_load_si128((__m128i*)tab_dct_16_1[(tab)])); \
    T64  = _mm_madd_epi16(T24, _mm_load_si128((__m128i*)tab_dct_16_1[(tab)])); \
    T65  = _mm_madd_epi16(T25, _mm_load_si128((__m128i*)tab_dct_16_1[(tab)])); \
    T66  = _mm_madd_epi16(T26, _mm_load_si128((__m128i*)tab_dct_16_1[(tab)])); \
    T67  = _mm_madd_epi16(T27, _mm_load_si128((__m128i*)tab_dct_16_1[(tab)])); \
    T60  = _mm_hadd_epi32(T60, T61); \
    T61  = _mm_hadd_epi32(T62, T63); \
    T62  = _mm_hadd_epi32(T64, T65); \
    T63  = _mm_hadd_epi32(T66, T67); \
    T60  = _mm_hadd_epi32(T60, T61); \
    T61  = _mm_hadd_epi32(T62, T63); \
    T60  = _mm_srai_epi32(_mm_add_epi32(T60, c_add1), SHIFT1); \
    T61  = _mm_srai_epi32(_mm_add_epi32(T61, c_add1), SHIFT1); \
    T70  = _mm_packs_epi32(T60, T61); \
    _mm_store_si128((__m128i*)(tmp + (dstPos) * 16 + i), T70);

        MAKE_ODD(0, 1);
        MAKE_ODD(1, 3);
        MAKE_ODD(2, 5);
        MAKE_ODD(3, 7);
        MAKE_ODD(4, 9);
        MAKE_ODD(5, 11);
        MAKE_ODD(6, 13);
        MAKE_ODD(7, 15);
#undef MAKE_ODD
    }

    // DCT2
    for (i = 0; i < 16; i += 4) {
        T00A = _mm_load_si128((__m128i*)(tmp + (i + 0) * 16 + 0));    // [07 06 05 04 03 02 01 00]
        T00B = _mm_load_si128((__m128i*)(tmp + (i + 0) * 16 + 8));    // [0F 0E 0D 0C 0B 0A 09 08]
        T01A = _mm_load_si128((__m128i*)(tmp + (i + 1) * 16 + 0));    // [17 16 15 14 13 12 11 10]
        T01B = _mm_load_si128((__m128i*)(tmp + (i + 1) * 16 + 8));    // [1F 1E 1D 1C 1B 1A 19 18]
        T02A = _mm_load_si128((__m128i*)(tmp + (i + 2) * 16 + 0));    // [27 26 25 24 23 22 21 20]
        T02B = _mm_load_si128((__m128i*)(tmp + (i + 2) * 16 + 8));    // [2F 2E 2D 2C 2B 2A 29 28]
        T03A = _mm_load_si128((__m128i*)(tmp + (i + 3) * 16 + 0));    // [37 36 35 34 33 32 31 30]
        T03B = _mm_load_si128((__m128i*)(tmp + (i + 3) * 16 + 8));    // [3F 3E 3D 3C 3B 3A 39 38]

        T00A = _mm_shuffle_epi8(T00A, _mm_load_si128((__m128i*)tab_dct_16_0[2]));
        T00B = _mm_shuffle_epi8(T00B, _mm_load_si128((__m128i*)tab_dct_16_0[3]));
        T01A = _mm_shuffle_epi8(T01A, _mm_load_si128((__m128i*)tab_dct_16_0[2]));
        T01B = _mm_shuffle_epi8(T01B, _mm_load_si128((__m128i*)tab_dct_16_0[3]));
        T02A = _mm_shuffle_epi8(T02A, _mm_load_si128((__m128i*)tab_dct_16_0[2]));
        T02B = _mm_shuffle_epi8(T02B, _mm_load_si128((__m128i*)tab_dct_16_0[3]));
        T03A = _mm_shuffle_epi8(T03A, _mm_load_si128((__m128i*)tab_dct_16_0[2]));
        T03B = _mm_shuffle_epi8(T03B, _mm_load_si128((__m128i*)tab_dct_16_0[3]));

        T10  = _mm_unpacklo_epi16(T00A, T00B);
        T11  = _mm_unpackhi_epi16(T00A, T00B);
        T12  = _mm_unpacklo_epi16(T01A, T01B);
        T13  = _mm_unpackhi_epi16(T01A, T01B);
        T14  = _mm_unpacklo_epi16(T02A, T02B);
        T15  = _mm_unpackhi_epi16(T02A, T02B);
        T16  = _mm_unpacklo_epi16(T03A, T03B);
        T17  = _mm_unpackhi_epi16(T03A, T03B);

        T20  = _mm_madd_epi16(T10, _mm_load_si128((__m128i*)tab_dct_8[1]));
        T21  = _mm_madd_epi16(T11, _mm_load_si128((__m128i*)tab_dct_8[1]));
        T22  = _mm_madd_epi16(T12, _mm_load_si128((__m128i*)tab_dct_8[1]));
        T23  = _mm_madd_epi16(T13, _mm_load_si128((__m128i*)tab_dct_8[1]));
        T24  = _mm_madd_epi16(T14, _mm_load_si128((__m128i*)tab_dct_8[1]));
        T25  = _mm_madd_epi16(T15, _mm_load_si128((__m128i*)tab_dct_8[1]));
        T26  = _mm_madd_epi16(T16, _mm_load_si128((__m128i*)tab_dct_8[1]));
        T27  = _mm_madd_epi16(T17, _mm_load_si128((__m128i*)tab_dct_8[1]));

        T30  = _mm_add_epi32(T20, T21);
        T31  = _mm_add_epi32(T22, T23);
        T32  = _mm_add_epi32(T24, T25);
        T33  = _mm_add_epi32(T26, T27);

        T30  = _mm_hadd_epi32(T30, T31);
        T31  = _mm_hadd_epi32(T32, T33);

        T40  = _mm_hadd_epi32(T30, T31);
        T41  = _mm_hsub_epi32(T30, T31);
        T40  = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2);
        T41  = _mm_srai_epi32(_mm_add_epi32(T41, c_add2), SHIFT2);
        T40  = _mm_packs_epi32(T40, T40);
        T41  = _mm_packs_epi32(T41, T41);
        _mm_storel_epi64((__m128i*)(dst + 0 * 16 + i), T40);
        _mm_storel_epi64((__m128i*)(dst + 8 * 16 + i), T41);

        T20  = _mm_madd_epi16(T10, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
        T21  = _mm_madd_epi16(T11, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
        T22  = _mm_madd_epi16(T12, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
        T23  = _mm_madd_epi16(T13, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
        T24  = _mm_madd_epi16(T14, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
        T25  = _mm_madd_epi16(T15, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
        T26  = _mm_madd_epi16(T16, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
        T27  = _mm_madd_epi16(T17, _mm_load_si128((__m128i*)tab_dct_16_1[8]));

        T30  = _mm_add_epi32(T20, T21);
        T31  = _mm_add_epi32(T22, T23);
        T32  = _mm_add_epi32(T24, T25);
        T33  = _mm_add_epi32(T26, T27);

        T30  = _mm_hadd_epi32(T30, T31);
        T31  = _mm_hadd_epi32(T32, T33);

        T40  = _mm_hadd_epi32(T30, T31);
        T40  = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2);
        T40  = _mm_packs_epi32(T40, T40);
        _mm_storel_epi64((__m128i*)(dst + 4 * 16 + i), T40);

        T20  = _mm_madd_epi16(T10, _mm_load_si128((__m128i*)tab_dct_16_1[9]));
        T21  = _mm_madd_epi16(T11, _mm_load_si128((__m128i*)tab_dct_16_1[9]));
        T22  = _mm_madd_epi16(T12, _mm_load_si128((__m128i*)tab_dct_16_1[9]));
        T23  = _mm_madd_epi16(T13, _mm_load_si128((__m128i*)tab_dct_16_1[9]));
        T24  = _mm_madd_epi16(T14, _mm_load_si128((__m128i*)tab_dct_16_1[9]));
        T25  = _mm_madd_epi16(T15, _mm_load_si128((__m128i*)tab_dct_16_1[9]));
        T26  = _mm_madd_epi16(T16, _mm_load_si128((__m128i*)tab_dct_16_1[9]));
        T27  = _mm_madd_epi16(T17, _mm_load_si128((__m128i*)tab_dct_16_1[9]));

        T30  = _mm_add_epi32(T20, T21);
        T31  = _mm_add_epi32(T22, T23);
        T32  = _mm_add_epi32(T24, T25);
        T33  = _mm_add_epi32(T26, T27);

        T30  = _mm_hadd_epi32(T30, T31);
        T31  = _mm_hadd_epi32(T32, T33);

        T40  = _mm_hadd_epi32(T30, T31);
        T40  = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2);
        T40  = _mm_packs_epi32(T40, T40);
        _mm_storel_epi64((__m128i*)(dst + 12 * 16 + i), T40);

        T20  = _mm_madd_epi16(T10, _mm_load_si128((__m128i*)tab_dct_16_1[10]));
        T21  = _mm_madd_epi16(T11, _mm_load_si128((__m128i*)tab_dct_16_1[10]));
        T22  = _mm_madd_epi16(T12, _mm_load_si128((__m128i*)tab_dct_16_1[10]));
        T23  = _mm_madd_epi16(T13, _mm_load_si128((__m128i*)tab_dct_16_1[10]));
        T24  = _mm_madd_epi16(T14, _mm_load_si128((__m128i*)tab_dct_16_1[10]));
        T25  = _mm_madd_epi16(T15, _mm_load_si128((__m128i*)tab_dct_16_1[10]));
        T26  = _mm_madd_epi16(T16, _mm_load_si128((__m128i*)tab_dct_16_1[10]));
        T27  = _mm_madd_epi16(T17, _mm_load_si128((__m128i*)tab_dct_16_1[10]));

        T30  = _mm_sub_epi32(T20, T21);
        T31  = _mm_sub_epi32(T22, T23);
        T32  = _mm_sub_epi32(T24, T25);
        T33  = _mm_sub_epi32(T26, T27);

        T30  = _mm_hadd_epi32(T30, T31);
        T31  = _mm_hadd_epi32(T32, T33);

        T40  = _mm_hadd_epi32(T30, T31);
        T40  = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2);
        T40  = _mm_packs_epi32(T40, T40);
        _mm_storel_epi64((__m128i*)(dst + 2 * 16 + i), T40);

        T20  = _mm_madd_epi16(T10, _mm_load_si128((__m128i*)tab_dct_16_1[11]));
        T21  = _mm_madd_epi16(T11, _mm_load_si128((__m128i*)tab_dct_16_1[11]));
        T22  = _mm_madd_epi16(T12, _mm_load_si128((__m128i*)tab_dct_16_1[11]));
        T23  = _mm_madd_epi16(T13, _mm_load_si128((__m128i*)tab_dct_16_1[11]));
        T24  = _mm_madd_epi16(T14, _mm_load_si128((__m128i*)tab_dct_16_1[11]));
        T25  = _mm_madd_epi16(T15, _mm_load_si128((__m128i*)tab_dct_16_1[11]));
        T26  = _mm_madd_epi16(T16, _mm_load_si128((__m128i*)tab_dct_16_1[11]));
        T27  = _mm_madd_epi16(T17, _mm_load_si128((__m128i*)tab_dct_16_1[11]));

        T30  = _mm_sub_epi32(T20, T21);
        T31  = _mm_sub_epi32(T22, T23);
        T32  = _mm_sub_epi32(T24, T25);
        T33  = _mm_sub_epi32(T26, T27);

        T30  = _mm_hadd_epi32(T30, T31);
        T31  = _mm_hadd_epi32(T32, T33);

        T40  = _mm_hadd_epi32(T30, T31);
        T40  = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2);
        T40  = _mm_packs_epi32(T40, T40);
        _mm_storel_epi64((__m128i*)(dst + 6 * 16 + i), T40);

        T20  = _mm_madd_epi16(T10, _mm_load_si128((__m128i*)tab_dct_16_1[12]));
        T21  = _mm_madd_epi16(T11, _mm_load_si128((__m128i*)tab_dct_16_1[12]));
        T22  = _mm_madd_epi16(T12, _mm_load_si128((__m128i*)tab_dct_16_1[12]));
        T23  = _mm_madd_epi16(T13, _mm_load_si128((__m128i*)tab_dct_16_1[12]));
        T24  = _mm_madd_epi16(T14, _mm_load_si128((__m128i*)tab_dct_16_1[12]));
        T25  = _mm_madd_epi16(T15, _mm_load_si128((__m128i*)tab_dct_16_1[12]));
        T26  = _mm_madd_epi16(T16, _mm_load_si128((__m128i*)tab_dct_16_1[12]));
        T27  = _mm_madd_epi16(T17, _mm_load_si128((__m128i*)tab_dct_16_1[12]));

        T30  = _mm_sub_epi32(T20, T21);
        T31  = _mm_sub_epi32(T22, T23);
        T32  = _mm_sub_epi32(T24, T25);
        T33  = _mm_sub_epi32(T26, T27);

        T30  = _mm_hadd_epi32(T30, T31);
        T31  = _mm_hadd_epi32(T32, T33);

        T40  = _mm_hadd_epi32(T30, T31);
        T40  = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2);
        T40  = _mm_packs_epi32(T40, T40);
        _mm_storel_epi64((__m128i*)(dst + 10 * 16 + i), T40);

        T20  = _mm_madd_epi16(T10, _mm_load_si128((__m128i*)tab_dct_16_1[13]));
        T21  = _mm_madd_epi16(T11, _mm_load_si128((__m128i*)tab_dct_16_1[13]));
        T22  = _mm_madd_epi16(T12, _mm_load_si128((__m128i*)tab_dct_16_1[13]));
        T23  = _mm_madd_epi16(T13, _mm_load_si128((__m128i*)tab_dct_16_1[13]));
        T24  = _mm_madd_epi16(T14, _mm_load_si128((__m128i*)tab_dct_16_1[13]));
        T25  = _mm_madd_epi16(T15, _mm_load_si128((__m128i*)tab_dct_16_1[13]));
        T26  = _mm_madd_epi16(T16, _mm_load_si128((__m128i*)tab_dct_16_1[13]));
        T27  = _mm_madd_epi16(T17, _mm_load_si128((__m128i*)tab_dct_16_1[13]));

        T30  = _mm_sub_epi32(T20, T21);
        T31  = _mm_sub_epi32(T22, T23);
        T32  = _mm_sub_epi32(T24, T25);
        T33  = _mm_sub_epi32(T26, T27);

        T30  = _mm_hadd_epi32(T30, T31);
        T31  = _mm_hadd_epi32(T32, T33);

        T40  = _mm_hadd_epi32(T30, T31);
        T40  = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2);
        T40  = _mm_packs_epi32(T40, T40);
        _mm_storel_epi64((__m128i*)(dst + 14 * 16 + i), T40);

#define MAKE_ODD(tab, dstPos) \
    T20  = _mm_madd_epi16(T10, _mm_load_si128((__m128i*)tab_dct_16_1[(tab)    ])); /* [*O2_0 *O1_0 *O3_0 *O0_0] */ \
    T21  = _mm_madd_epi16(T11, _mm_load_si128((__m128i*)tab_dct_16_1[(tab) + 1])); /* [*O5_0 *O6_0 *O4_0 *O7_0] */ \
    T22  = _mm_madd_epi16(T12, _mm_load_si128((__m128i*)tab_dct_16_1[(tab)    ])); \
    T23  = _mm_madd_epi16(T13, _mm_load_si128((__m128i*)tab_dct_16_1[(tab) + 1])); \
    T24  = _mm_madd_epi16(T14, _mm_load_si128((__m128i*)tab_dct_16_1[(tab)    ])); \
    T25  = _mm_madd_epi16(T15, _mm_load_si128((__m128i*)tab_dct_16_1[(tab) + 1])); \
    T26  = _mm_madd_epi16(T16, _mm_load_si128((__m128i*)tab_dct_16_1[(tab)    ])); \
    T27  = _mm_madd_epi16(T17, _mm_load_si128((__m128i*)tab_dct_16_1[(tab) + 1])); \
    \
    T30  = _mm_add_epi32(T20, T21); \
    T31  = _mm_add_epi32(T22, T23); \
    T32  = _mm_add_epi32(T24, T25); \
    T33  = _mm_add_epi32(T26, T27); \
    \
    T30  = _mm_hadd_epi32(T30, T31); \
    T31  = _mm_hadd_epi32(T32, T33); \
    \
    T40  = _mm_hadd_epi32(T30, T31); \
    T40  = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2); \
    T40  = _mm_packs_epi32(T40, T40); \
    _mm_storel_epi64((__m128i*)(dst + (dstPos) * 16 + i), T40);

        MAKE_ODD(14,  1);
        MAKE_ODD(16,  3);
        MAKE_ODD(18,  5);
        MAKE_ODD(20,  7);
        MAKE_ODD(22,  9);
        MAKE_ODD(24, 11);
        MAKE_ODD(26, 13);
        MAKE_ODD(28, 15);
#undef MAKE_ODD
    }
}


/* ---------------------------------------------------------------------------
 */
void dct_c_8x32_sse128(const coeff_t *src, coeff_t *dst, int i_src)
{
    int i;
    int shift1 = B32X32_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT - 2 + (i_src & 0x01);
    int shift2 = B32X32_IN_BIT + FACTO_BIT;
    const int ADD1 = (1 << shift1) >> 1;
    const int ADD2 = (1 << shift2) >> 1;
    const __m128i c_512 = _mm_set1_epi32(ADD2);        // TODO: shift1 = 2
    const __m128i k_ROUNDING1 = _mm_set1_epi32(ADD1);

    __m128i T00A, T01A, T02A, T03A;
    __m128i T00B, T01B, T02B, T03B;
    __m128i T00C, T01C, T02C, T03C;
    __m128i T00D, T01D, T02D, T03D;
    __m128i T10A, T11A, T12A, T13A;
    __m128i T10B, T11B, T12B, T13B;
    __m128i T20, T21, T22, T23, T24, T25, T26, T27;
    __m128i T30, T31, T32, T33, T34, T35, T36, T37;
    __m128i T60, T61, T62, T63, T64, T65, T66, T67;

    __m128i TT00A, TT01A, TT02A, TT03A;
    __m128i TT00B, TT01B, TT02B, TT03B;
    __m128i TT00C, TT01C, TT02C, TT03C;
    __m128i TT00D, TT01D, TT02D, TT03D;
    __m128i TT10A, TT11A, TT12A, TT13A;
    __m128i TT10B, TT11B, TT12B, TT13B;
    __m128i TT20, TT21, TT22, TT23, TT24, TT25, TT26, TT27;
    __m128i TT30, TT31, TT32, TT33, TT34, TT35, TT36, TT37;
    __m128i TT60, TT61, TT62, TT63, TT64, TT65, TT66, TT67;
    __m128i tResult;
    __m128i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
    __m128i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;
    __m128i in0, in1, in2, in3, in4, in5, in6, in7;
    __m128i res0[4], res1[4], res2[4], res3[4], res4[4], res5[4], res6[4], res7[4];
    __m128i r0, r1, r2, r3, t0, t1, t2, t3;
    __m128i q0, q1, q2, q3, q4, q5, q6, q7, u0, u1, u2, u3, u4, u5, u6, u7, v0, v1, v2, v3, v4, v5, v6, v7, w0, w1, w2, w3, w4, w5, w6, w7;
    const __m128i k_p32_p32 = _mm_set1_epi16(32);
    const __m128i k_p32_m32 = pair_set_epi16(32, -32);
    const __m128i k_p42_p17 = pair_set_epi16(42, 17);
    const __m128i k_p17_m42 = pair_set_epi16(17, -42);
    const __m128i k_p44_p38 = pair_set_epi16(44, 38);
    const __m128i k_p25_p9  = pair_set_epi16(25, 9);
    const __m128i k_p38_m9  = pair_set_epi16(38, -9);
    const __m128i k_m44_m25 = pair_set_epi16(-44, -25);
    const __m128i k_p25_m44 = pair_set_epi16(25, -44);
    const __m128i k_p9_p38  = pair_set_epi16(9, 38);
    const __m128i k_p9_m25  = pair_set_epi16(9, -25);
    const __m128i k_p38_m44 = pair_set_epi16(38, -44);

    i_src &= 0xFE;

    for (i = 0; i < 32 / 8; i++) {
        //load data
        in0 = _mm_loadu_si128((__m128i*)&src[(0 + i * 8) * i_src]);
        in1 = _mm_loadu_si128((__m128i*)&src[(1 + i * 8) * i_src]);
        in2 = _mm_loadu_si128((__m128i*)&src[(2 + i * 8) * i_src]);
        in3 = _mm_loadu_si128((__m128i*)&src[(3 + i * 8) * i_src]);
        in4 = _mm_loadu_si128((__m128i*)&src[(4 + i * 8) * i_src]);
        in5 = _mm_loadu_si128((__m128i*)&src[(5 + i * 8) * i_src]);
        in6 = _mm_loadu_si128((__m128i*)&src[(6 + i * 8) * i_src]);
        in7 = _mm_loadu_si128((__m128i*)&src[(7 + i * 8) * i_src]);

        //DCT1
#define TRANSPOSE_8x8(I0, I1, I2, I3, I4, I5, I6, I7) \
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
    I0 = _mm_unpacklo_epi64(tr1_0, tr1_4); \
    I1 = _mm_unpackhi_epi64(tr1_0, tr1_4); \
    I2 = _mm_unpacklo_epi64(tr1_2, tr1_6); \
    I3 = _mm_unpackhi_epi64(tr1_2, tr1_6); \
    I4 = _mm_unpacklo_epi64(tr1_1, tr1_5); \
    I5 = _mm_unpackhi_epi64(tr1_1, tr1_5); \
    I6 = _mm_unpacklo_epi64(tr1_3, tr1_7); \
    I7 = _mm_unpackhi_epi64(tr1_3, tr1_7); \
 
        TRANSPOSE_8x8(in0, in1, in2, in3, in4, in5, in6, in7)
#undef TRANSPOSE_8x8

        q0 = _mm_add_epi16(in0, in7);    //E0
        q1 = _mm_add_epi16(in1, in6);    //E1
        q2 = _mm_add_epi16(in2, in5);    //E2
        q3 = _mm_add_epi16(in3, in4);    //E3
        q4 = _mm_sub_epi16(in0, in7);    //O0
        q5 = _mm_sub_epi16(in1, in6);    //O1
        q6 = _mm_sub_epi16(in2, in5);    //O2
        q7 = _mm_sub_epi16(in3, in4);    //O3

        //even lines
        r0 = _mm_add_epi16(q0, q3);    //EE0
        r1 = _mm_add_epi16(q1, q2);    //EE1
        r2 = _mm_sub_epi16(q0, q3);    //EO0
        r3 = _mm_sub_epi16(q1, q2);    //EO1

        t0 = _mm_unpacklo_epi16(r0, r1);    //interleave EE0 & EE1
        t1 = _mm_unpackhi_epi16(r0, r1);
        t2 = _mm_unpacklo_epi16(r2, r3);    //interleave EO0 & EO1
        t3 = _mm_unpackhi_epi16(r2, r3);
        u0 = _mm_madd_epi16(t0, k_p32_p32);
        u1 = _mm_madd_epi16(t1, k_p32_p32);
        u2 = _mm_madd_epi16(t0, k_p32_m32);
        u3 = _mm_madd_epi16(t1, k_p32_m32);
        u4 = _mm_madd_epi16(t2, k_p42_p17);
        u5 = _mm_madd_epi16(t3, k_p42_p17);
        u6 = _mm_madd_epi16(t2, k_p17_m42);
        u7 = _mm_madd_epi16(t3, k_p17_m42);

        v0 = _mm_add_epi32(u0, k_ROUNDING1);
        v1 = _mm_add_epi32(u1, k_ROUNDING1);
        v2 = _mm_add_epi32(u2, k_ROUNDING1);
        v3 = _mm_add_epi32(u3, k_ROUNDING1);
        v4 = _mm_add_epi32(u4, k_ROUNDING1);
        v5 = _mm_add_epi32(u5, k_ROUNDING1);
        v6 = _mm_add_epi32(u6, k_ROUNDING1);
        v7 = _mm_add_epi32(u7, k_ROUNDING1);
        w0 = _mm_srai_epi32(v0, shift1);
        w1 = _mm_srai_epi32(v1, shift1);
        w2 = _mm_srai_epi32(v2, shift1);
        w3 = _mm_srai_epi32(v3, shift1);
        w4 = _mm_srai_epi32(v4, shift1);
        w5 = _mm_srai_epi32(v5, shift1);
        w6 = _mm_srai_epi32(v6, shift1);
        w7 = _mm_srai_epi32(v7, shift1);

        res0[i] = _mm_packs_epi32(w0, w1);
        res4[i] = _mm_packs_epi32(w2, w3);
        res2[i] = _mm_packs_epi32(w4, w5);
        res6[i] = _mm_packs_epi32(w6, w7);

        // odd lines
        t0 = _mm_unpacklo_epi16(q4, q5);    //interleave O0 & O1
        t1 = _mm_unpackhi_epi16(q4, q5);
        t2 = _mm_unpacklo_epi16(q6, q7);    //interleave O2 & O3
        t3 = _mm_unpackhi_epi16(q6, q7);

        //line 1
        u0 = _mm_madd_epi16(t0, k_p44_p38);
        u1 = _mm_madd_epi16(t1, k_p44_p38);
        u2 = _mm_madd_epi16(t2, k_p25_p9);
        u3 = _mm_madd_epi16(t3, k_p25_p9);
        v0 = _mm_add_epi32(u0, u2);
        v1 = _mm_add_epi32(u1, u3);
        v0 = _mm_add_epi32(v0, k_ROUNDING1);
        v1 = _mm_add_epi32(v1, k_ROUNDING1);
        w0 = _mm_srai_epi32(v0, shift1);
        w1 = _mm_srai_epi32(v1, shift1);
        res1[i] = _mm_packs_epi32(w0, w1);

        //line 3
        u0 = _mm_madd_epi16(t0, k_p38_m9);
        u1 = _mm_madd_epi16(t1, k_p38_m9);
        u2 = _mm_madd_epi16(t2, k_m44_m25);
        u3 = _mm_madd_epi16(t3, k_m44_m25);
        v0 = _mm_add_epi32(u0, u2);
        v1 = _mm_add_epi32(u1, u3);
        v0 = _mm_add_epi32(v0, k_ROUNDING1);
        v1 = _mm_add_epi32(v1, k_ROUNDING1);
        w0 = _mm_srai_epi32(v0, shift1);
        w1 = _mm_srai_epi32(v1, shift1);
        res3[i] = _mm_packs_epi32(w0, w1);

        //line 5
        u0 = _mm_madd_epi16(t0, k_p25_m44);
        u1 = _mm_madd_epi16(t1, k_p25_m44);
        u2 = _mm_madd_epi16(t2, k_p9_p38);
        u3 = _mm_madd_epi16(t3, k_p9_p38);
        v0 = _mm_add_epi32(u0, u2);
        v1 = _mm_add_epi32(u1, u3);
        v0 = _mm_add_epi32(v0, k_ROUNDING1);
        v1 = _mm_add_epi32(v1, k_ROUNDING1);
        w0 = _mm_srai_epi32(v0, shift1);
        w1 = _mm_srai_epi32(v1, shift1);
        res5[i] = _mm_packs_epi32(w0, w1);

        //line 7
        u0 = _mm_madd_epi16(t0, k_p9_m25);
        u1 = _mm_madd_epi16(t1, k_p9_m25);
        u2 = _mm_madd_epi16(t2, k_p38_m44);
        u3 = _mm_madd_epi16(t3, k_p38_m44);
        v0 = _mm_add_epi32(u0, u2);
        v1 = _mm_add_epi32(u1, u3);
        v0 = _mm_add_epi32(v0, k_ROUNDING1);
        v1 = _mm_add_epi32(v1, k_ROUNDING1);
        w0 = _mm_srai_epi32(v0, shift1);
        w1 = _mm_srai_epi32(v1, shift1);
        res7[i] = _mm_packs_epi32(w0, w1);
    }

    //DCT2
    T00A = res0[0];    // [07 06 05 04 03 02 01 00]
    T00B = res0[1];    // [15 14 13 12 11 10 09 08]
    T00C = res0[2];    // [23 22 21 20 19 18 17 16]
    T00D = res0[3];    // [31 30 29 28 27 26 25 24]
    T01A = res1[0];
    T01B = res1[1];
    T01C = res1[2];
    T01D = res1[3];
    T02A = res2[0];
    T02B = res2[1];
    T02C = res2[2];
    T02D = res2[3];
    T03A = res3[0];
    T03B = res3[1];
    T03C = res3[2];
    T03D = res3[3];
    TT00A = res4[0];
    TT00B = res4[1];
    TT00C = res4[2];
    TT00D = res4[3];
    TT01A = res5[0];
    TT01B = res5[1];
    TT01C = res5[2];
    TT01D = res5[3];
    TT02A = res6[0];
    TT02B = res6[1];
    TT02C = res6[2];
    TT02D = res6[3];
    TT03A = res7[0];
    TT03B = res7[1];
    TT03C = res7[2];
    TT03D = res7[3];

    T00C = _mm_shuffle_epi8(T00C, _mm_load_si128((__m128i*)tab_dct_16_0[0]));    // [16 17 18 19 20 21 22 23]
    T00D = _mm_shuffle_epi8(T00D, _mm_load_si128((__m128i*)tab_dct_16_0[0]));    // [24 25 26 27 28 29 30 31]
    T01C = _mm_shuffle_epi8(T01C, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
    T01D = _mm_shuffle_epi8(T01D, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
    T02C = _mm_shuffle_epi8(T02C, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
    T02D = _mm_shuffle_epi8(T02D, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
    T03C = _mm_shuffle_epi8(T03C, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
    T03D = _mm_shuffle_epi8(T03D, _mm_load_si128((__m128i*)tab_dct_16_0[0]));

    TT00C = _mm_shuffle_epi8(TT00C, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
    TT00D = _mm_shuffle_epi8(TT00D, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
    TT01C = _mm_shuffle_epi8(TT01C, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
    TT01D = _mm_shuffle_epi8(TT01D, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
    TT02C = _mm_shuffle_epi8(TT02C, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
    TT02D = _mm_shuffle_epi8(TT02D, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
    TT03C = _mm_shuffle_epi8(TT03C, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
    TT03D = _mm_shuffle_epi8(TT03D, _mm_load_si128((__m128i*)tab_dct_16_0[0]));

    T10A = _mm_unpacklo_epi16(T00A, T00D);  // [28 03 29 02 30 01 31 00]
    T10B = _mm_unpackhi_epi16(T00A, T00D);  // [24 07 25 06 26 05 27 04]
    T00A = _mm_unpacklo_epi16(T00B, T00C);  // [20 11 21 10 22 09 23 08]
    T00B = _mm_unpackhi_epi16(T00B, T00C);  // [16 15 17 14 18 13 19 12]
    T11A = _mm_unpacklo_epi16(T01A, T01D);
    T11B = _mm_unpackhi_epi16(T01A, T01D);
    T01A = _mm_unpacklo_epi16(T01B, T01C);
    T01B = _mm_unpackhi_epi16(T01B, T01C);
    T12A = _mm_unpacklo_epi16(T02A, T02D);
    T12B = _mm_unpackhi_epi16(T02A, T02D);
    T02A = _mm_unpacklo_epi16(T02B, T02C);
    T02B = _mm_unpackhi_epi16(T02B, T02C);
    T13A = _mm_unpacklo_epi16(T03A, T03D);
    T13B = _mm_unpackhi_epi16(T03A, T03D);
    T03A = _mm_unpacklo_epi16(T03B, T03C);
    T03B = _mm_unpackhi_epi16(T03B, T03C);

    TT10A = _mm_unpacklo_epi16(TT00A, TT00D);
    TT10B = _mm_unpackhi_epi16(TT00A, TT00D);
    TT00A = _mm_unpacklo_epi16(TT00B, TT00C);
    TT00B = _mm_unpackhi_epi16(TT00B, TT00C);
    TT11A = _mm_unpacklo_epi16(TT01A, TT01D);
    TT11B = _mm_unpackhi_epi16(TT01A, TT01D);
    TT01A = _mm_unpacklo_epi16(TT01B, TT01C);
    TT01B = _mm_unpackhi_epi16(TT01B, TT01C);
    TT12A = _mm_unpacklo_epi16(TT02A, TT02D);
    TT12B = _mm_unpackhi_epi16(TT02A, TT02D);
    TT02A = _mm_unpacklo_epi16(TT02B, TT02C);
    TT02B = _mm_unpackhi_epi16(TT02B, TT02C);
    TT13A = _mm_unpacklo_epi16(TT03A, TT03D);
    TT13B = _mm_unpackhi_epi16(TT03A, TT03D);
    TT03A = _mm_unpacklo_epi16(TT03B, TT03C);
    TT03B = _mm_unpackhi_epi16(TT03B, TT03C);

#define MAKE_ODD(tab0, tab1, tab2, tab3, dstPos) \
    T20 = _mm_madd_epi16(T10A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab0)])); \
    T21 = _mm_madd_epi16(T10B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab1)])); \
    T22 = _mm_madd_epi16(T00A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab2)])); \
    T23 = _mm_madd_epi16(T00B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab3)])); \
    T24 = _mm_madd_epi16(T11A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab0)])); \
    T25 = _mm_madd_epi16(T11B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab1)])); \
    T26 = _mm_madd_epi16(T01A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab2)])); \
    T27 = _mm_madd_epi16(T01B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab3)])); \
    T30 = _mm_madd_epi16(T12A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab0)])); \
    T31 = _mm_madd_epi16(T12B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab1)])); \
    T32 = _mm_madd_epi16(T02A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab2)])); \
    T33 = _mm_madd_epi16(T02B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab3)])); \
    T34 = _mm_madd_epi16(T13A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab0)])); \
    T35 = _mm_madd_epi16(T13B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab1)])); \
    T36 = _mm_madd_epi16(T03A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab2)])); \
    T37 = _mm_madd_epi16(T03B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab3)])); \
    \
    T60 = _mm_hadd_epi32(T20, T21); \
    T61 = _mm_hadd_epi32(T22, T23); \
    T62 = _mm_hadd_epi32(T24, T25); \
    T63 = _mm_hadd_epi32(T26, T27); \
    T64 = _mm_hadd_epi32(T30, T31); \
    T65 = _mm_hadd_epi32(T32, T33); \
    T66 = _mm_hadd_epi32(T34, T35); \
    T67 = _mm_hadd_epi32(T36, T37); \
    \
    T60 = _mm_hadd_epi32(T60, T61); \
    T61 = _mm_hadd_epi32(T62, T63); \
    T62 = _mm_hadd_epi32(T64, T65); \
    T63 = _mm_hadd_epi32(T66, T67); \
    \
    T60 = _mm_hadd_epi32(T60, T61); \
    T61 = _mm_hadd_epi32(T62, T63); \
    \
    T60 = _mm_hadd_epi32(T60, T61); \
    \
    T60 = _mm_srai_epi32(_mm_add_epi32(T60, c_512), shift2); \
    \
    TT20 = _mm_madd_epi16(TT10A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab0)])); \
    TT21 = _mm_madd_epi16(TT10B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab1)])); \
    TT22 = _mm_madd_epi16(TT00A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab2)])); \
    TT23 = _mm_madd_epi16(TT00B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab3)])); \
    TT24 = _mm_madd_epi16(TT11A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab0)])); \
    TT25 = _mm_madd_epi16(TT11B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab1)])); \
    TT26 = _mm_madd_epi16(TT01A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab2)])); \
    TT27 = _mm_madd_epi16(TT01B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab3)])); \
    TT30 = _mm_madd_epi16(TT12A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab0)])); \
    TT31 = _mm_madd_epi16(TT12B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab1)])); \
    TT32 = _mm_madd_epi16(TT02A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab2)])); \
    TT33 = _mm_madd_epi16(TT02B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab3)])); \
    TT34 = _mm_madd_epi16(TT13A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab0)])); \
    TT35 = _mm_madd_epi16(TT13B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab1)])); \
    TT36 = _mm_madd_epi16(TT03A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab2)])); \
    TT37 = _mm_madd_epi16(TT03B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab3)])); \
    \
    TT60 = _mm_hadd_epi32(TT20, TT21); \
    TT61 = _mm_hadd_epi32(TT22, TT23); \
    TT62 = _mm_hadd_epi32(TT24, TT25); \
    TT63 = _mm_hadd_epi32(TT26, TT27); \
    TT64 = _mm_hadd_epi32(TT30, TT31); \
    TT65 = _mm_hadd_epi32(TT32, TT33); \
    TT66 = _mm_hadd_epi32(TT34, TT35); \
    TT67 = _mm_hadd_epi32(TT36, TT37); \
    \
    TT60 = _mm_hadd_epi32(TT60, TT61); \
    TT61 = _mm_hadd_epi32(TT62, TT63); \
    TT62 = _mm_hadd_epi32(TT64, TT65); \
    TT63 = _mm_hadd_epi32(TT66, TT67); \
    \
    TT60 = _mm_hadd_epi32(TT60, TT61); \
    TT61 = _mm_hadd_epi32(TT62, TT63); \
    \
    TT60 = _mm_hadd_epi32(TT60, TT61); \
    \
    TT60 = _mm_srai_epi32(_mm_add_epi32(TT60, c_512), shift2); \
    \
    tResult = _mm_packs_epi32(T60, TT60); \
    _mm_storeu_si128((__m128i*)&dst[(dstPos)* 8], tResult); \
 
    MAKE_ODD(44, 44, 44, 44, 0);
    MAKE_ODD(45, 45, 45, 45, 16);
    MAKE_ODD(46, 47, 46, 47, 8);
    MAKE_ODD(48, 49, 48, 49, 24);

    MAKE_ODD(50, 51, 52, 53, 4);
    MAKE_ODD(54, 55, 56, 57, 12);
    MAKE_ODD(58, 59, 60, 61, 20);
    MAKE_ODD(62, 63, 64, 65, 28);

    MAKE_ODD(66, 67, 68, 69, 2);
    MAKE_ODD(70, 71, 72, 73, 6);
    MAKE_ODD(74, 75, 76, 77, 10);
    MAKE_ODD(78, 79, 80, 81, 14);

    MAKE_ODD(82, 83, 84, 85, 18);
    MAKE_ODD(86, 87, 88, 89, 22);
    MAKE_ODD(90, 91, 92, 93, 26);
    MAKE_ODD(94, 95, 96, 97, 30);

    MAKE_ODD(98, 99, 100, 101, 1);
    MAKE_ODD(102, 103, 104, 105, 3);
    MAKE_ODD(106, 107, 108, 109, 5);
    MAKE_ODD(110, 111, 112, 113, 7);
    MAKE_ODD(114, 115, 116, 117, 9);
    MAKE_ODD(118, 119, 120, 121, 11);
    MAKE_ODD(122, 123, 124, 125, 13);
    MAKE_ODD(126, 127, 128, 129, 15);
    MAKE_ODD(130, 131, 132, 133, 17);
    MAKE_ODD(134, 135, 136, 137, 19);
    MAKE_ODD(138, 139, 140, 141, 21);
    MAKE_ODD(142, 143, 144, 145, 23);
    MAKE_ODD(146, 147, 148, 149, 25);
    MAKE_ODD(150, 151, 152, 153, 27);
    MAKE_ODD(154, 155, 156, 157, 29);
    MAKE_ODD(158, 159, 160, 161, 31);
#undef MAKE_ODD
}

/* ---------------------------------------------------------------------------
 */
void dct_c_32x8_sse128(const coeff_t *src, coeff_t *dst, int i_src)
{
    int i;
    int shift1 = B32X32_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT;
    int shift2 = B32X32_IN_BIT + FACTO_BIT - 2 - (i_src & 0x01);
    const int ADD1 = (1 << shift1) >> 1;
    const int ADD2 = (1 << shift2) >> 1;
    const __m128i c_4 = _mm_set1_epi32(ADD1);        // TODO: shift1 = 2
    const __m128i k_ROUNDING2 = _mm_set1_epi32(ADD2);

    __m128i r0, r1, r2, r3, t0, q0, q1, q2, q3, q4, q5, q6, q7, v0, v1, w0, w1;
    __m128i T00A, T01A, T02A, T03A, T04A, T05A, T06A, T07A;
    __m128i T00B, T01B, T02B, T03B, T04B, T05B, T06B, T07B;
    __m128i T00C, T01C, T02C, T03C, T04C, T05C, T06C, T07C;
    __m128i T00D, T01D, T02D, T03D, T04D, T05D, T06D, T07D;
    __m128i T10A, T11A, T12A, T13A, T14A, T15A, T16A, T17A;
    __m128i T10B, T11B, T12B, T13B, T14B, T15B, T16B, T17B;
    __m128i T20, T21, T22, T23, T24, T25, T26, T27;
    __m128i T30, T31, T32, T33, T34, T35, T36, T37;
    __m128i T40, T41, T42, T43, T44, T45, T46, T47;
    __m128i T50, T51, T52, T53;
    __m128i T60;
    __m128i im[32];

    i_src &= 0xFE;
    T00A = _mm_loadu_si128((__m128i*)&src[0 * i_src + 0]);    // [07 06 05 04 03 02 01 00]
    T00B = _mm_loadu_si128((__m128i*)&src[0 * i_src + 8]);    // [15 14 13 12 11 10 09 08]
    T00C = _mm_loadu_si128((__m128i*)&src[0 * i_src + 16]);    // [23 22 21 20 19 18 17 16]
    T00D = _mm_loadu_si128((__m128i*)&src[0 * i_src + 24]);    // [31 30 29 28 27 26 25 24]
    T01A = _mm_loadu_si128((__m128i*)&src[1 * i_src + 0]);
    T01B = _mm_loadu_si128((__m128i*)&src[1 * i_src + 8]);
    T01C = _mm_loadu_si128((__m128i*)&src[1 * i_src + 16]);
    T01D = _mm_loadu_si128((__m128i*)&src[1 * i_src + 24]);
    T02A = _mm_loadu_si128((__m128i*)&src[2 * i_src + 0]);
    T02B = _mm_loadu_si128((__m128i*)&src[2 * i_src + 8]);
    T02C = _mm_loadu_si128((__m128i*)&src[2 * i_src + 16]);
    T02D = _mm_loadu_si128((__m128i*)&src[2 * i_src + 24]);
    T03A = _mm_loadu_si128((__m128i*)&src[3 * i_src + 0]);
    T03B = _mm_loadu_si128((__m128i*)&src[3 * i_src + 8]);
    T03C = _mm_loadu_si128((__m128i*)&src[3 * i_src + 16]);
    T03D = _mm_loadu_si128((__m128i*)&src[3 * i_src + 24]);
    T04A = _mm_loadu_si128((__m128i*)&src[4 * i_src + 0]);
    T04B = _mm_loadu_si128((__m128i*)&src[4 * i_src + 8]);
    T04C = _mm_loadu_si128((__m128i*)&src[4 * i_src + 16]);
    T04D = _mm_loadu_si128((__m128i*)&src[4 * i_src + 24]);
    T05A = _mm_loadu_si128((__m128i*)&src[5 * i_src + 0]);
    T05B = _mm_loadu_si128((__m128i*)&src[5 * i_src + 8]);
    T05C = _mm_loadu_si128((__m128i*)&src[5 * i_src + 16]);
    T05D = _mm_loadu_si128((__m128i*)&src[5 * i_src + 24]);
    T06A = _mm_loadu_si128((__m128i*)&src[6 * i_src + 0]);
    T06B = _mm_loadu_si128((__m128i*)&src[6 * i_src + 8]);
    T06C = _mm_loadu_si128((__m128i*)&src[6 * i_src + 16]);
    T06D = _mm_loadu_si128((__m128i*)&src[6 * i_src + 24]);
    T07A = _mm_loadu_si128((__m128i*)&src[7 * i_src + 0]);
    T07B = _mm_loadu_si128((__m128i*)&src[7 * i_src + 8]);
    T07C = _mm_loadu_si128((__m128i*)&src[7 * i_src + 16]);
    T07D = _mm_loadu_si128((__m128i*)&src[7 * i_src + 24]);

    // DCT1
    T00A = _mm_shuffle_epi8(T00A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));    // [05 02 06 01 04 03 07 00]
    T00B = _mm_shuffle_epi8(T00B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));    // [10 13 09 14 11 12 08 15]
    T00C = _mm_shuffle_epi8(T00C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));    // [21 18 22 17 20 19 23 16]
    T00D = _mm_shuffle_epi8(T00D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));    // [26 29 25 30 27 28 24 31]
    T01A = _mm_shuffle_epi8(T01A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
    T01B = _mm_shuffle_epi8(T01B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
    T01C = _mm_shuffle_epi8(T01C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
    T01D = _mm_shuffle_epi8(T01D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
    T02A = _mm_shuffle_epi8(T02A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
    T02B = _mm_shuffle_epi8(T02B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
    T02C = _mm_shuffle_epi8(T02C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
    T02D = _mm_shuffle_epi8(T02D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
    T03A = _mm_shuffle_epi8(T03A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
    T03B = _mm_shuffle_epi8(T03B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
    T03C = _mm_shuffle_epi8(T03C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
    T03D = _mm_shuffle_epi8(T03D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
    T04A = _mm_shuffle_epi8(T04A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
    T04B = _mm_shuffle_epi8(T04B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
    T04C = _mm_shuffle_epi8(T04C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
    T04D = _mm_shuffle_epi8(T04D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
    T05A = _mm_shuffle_epi8(T05A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
    T05B = _mm_shuffle_epi8(T05B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
    T05C = _mm_shuffle_epi8(T05C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
    T05D = _mm_shuffle_epi8(T05D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
    T06A = _mm_shuffle_epi8(T06A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
    T06B = _mm_shuffle_epi8(T06B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
    T06C = _mm_shuffle_epi8(T06C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
    T06D = _mm_shuffle_epi8(T06D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
    T07A = _mm_shuffle_epi8(T07A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
    T07B = _mm_shuffle_epi8(T07B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
    T07C = _mm_shuffle_epi8(T07C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
    T07D = _mm_shuffle_epi8(T07D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));

    T10A = _mm_add_epi16(T00A, T00D);   // [E05 E02 E06 E01 E04 E03 E07 E00]
    T10B = _mm_add_epi16(T00B, T00C);   // [E10 E13 E09 E14 E11 E12 E08 E15]
    T11A = _mm_add_epi16(T01A, T01D);
    T11B = _mm_add_epi16(T01B, T01C);
    T12A = _mm_add_epi16(T02A, T02D);
    T12B = _mm_add_epi16(T02B, T02C);
    T13A = _mm_add_epi16(T03A, T03D);
    T13B = _mm_add_epi16(T03B, T03C);
    T14A = _mm_add_epi16(T04A, T04D);
    T14B = _mm_add_epi16(T04B, T04C);
    T15A = _mm_add_epi16(T05A, T05D);
    T15B = _mm_add_epi16(T05B, T05C);
    T16A = _mm_add_epi16(T06A, T06D);
    T16B = _mm_add_epi16(T06B, T06C);
    T17A = _mm_add_epi16(T07A, T07D);
    T17B = _mm_add_epi16(T07B, T07C);

    T00A = _mm_sub_epi16(T00A, T00D);   // [O05 O02 O06 O01 O04 O03 O07 O00]
    T00B = _mm_sub_epi16(T00B, T00C);   // [O10 O13 O09 O14 O11 O12 O08 O15]
    T01A = _mm_sub_epi16(T01A, T01D);
    T01B = _mm_sub_epi16(T01B, T01C);
    T02A = _mm_sub_epi16(T02A, T02D);
    T02B = _mm_sub_epi16(T02B, T02C);
    T03A = _mm_sub_epi16(T03A, T03D);
    T03B = _mm_sub_epi16(T03B, T03C);
    T04A = _mm_sub_epi16(T04A, T04D);
    T04B = _mm_sub_epi16(T04B, T04C);
    T05A = _mm_sub_epi16(T05A, T05D);
    T05B = _mm_sub_epi16(T05B, T05C);
    T06A = _mm_sub_epi16(T06A, T06D);
    T06B = _mm_sub_epi16(T06B, T06C);
    T07A = _mm_sub_epi16(T07A, T07D);
    T07B = _mm_sub_epi16(T07B, T07C);

    T20 = _mm_add_epi16(T10A, T10B);   // [EE5 EE2 EE6 EE1 EE4 EE3 EE7 EE0]
    T21 = _mm_add_epi16(T11A, T11B);
    T22 = _mm_add_epi16(T12A, T12B);
    T23 = _mm_add_epi16(T13A, T13B);
    T24 = _mm_add_epi16(T14A, T14B);
    T25 = _mm_add_epi16(T15A, T15B);
    T26 = _mm_add_epi16(T16A, T16B);
    T27 = _mm_add_epi16(T17A, T17B);

    T30 = _mm_madd_epi16(T20, _mm_load_si128((__m128i*)tab_dct_8[1]));
    T31 = _mm_madd_epi16(T21, _mm_load_si128((__m128i*)tab_dct_8[1]));
    T32 = _mm_madd_epi16(T22, _mm_load_si128((__m128i*)tab_dct_8[1]));
    T33 = _mm_madd_epi16(T23, _mm_load_si128((__m128i*)tab_dct_8[1]));
    T34 = _mm_madd_epi16(T24, _mm_load_si128((__m128i*)tab_dct_8[1]));
    T35 = _mm_madd_epi16(T25, _mm_load_si128((__m128i*)tab_dct_8[1]));
    T36 = _mm_madd_epi16(T26, _mm_load_si128((__m128i*)tab_dct_8[1]));
    T37 = _mm_madd_epi16(T27, _mm_load_si128((__m128i*)tab_dct_8[1]));

    T40 = _mm_hadd_epi32(T30, T31);
    T41 = _mm_hadd_epi32(T32, T33);
    T42 = _mm_hadd_epi32(T34, T35);
    T43 = _mm_hadd_epi32(T36, T37);

    T50 = _mm_hadd_epi32(T40, T41);
    T51 = _mm_hadd_epi32(T42, T43);
    T50 = _mm_srai_epi32(_mm_add_epi32(T50, c_4), shift1);
    T51 = _mm_srai_epi32(_mm_add_epi32(T51, c_4), shift1);
    T60 = _mm_packs_epi32(T50, T51);
    im[0] = T60;
    //_mm_storeu_si128((__m128i*)&dst[0 * 8], T60);

    T50 = _mm_hsub_epi32(T40, T41);
    T51 = _mm_hsub_epi32(T42, T43);
    T50 = _mm_srai_epi32(_mm_add_epi32(T50, c_4), shift1);
    T51 = _mm_srai_epi32(_mm_add_epi32(T51, c_4), shift1);
    T60 = _mm_packs_epi32(T50, T51);
    im[16] = T60;
    //_mm_storeu_si128((__m128i*)&dst[16 * 8], T60);

    T30 = _mm_madd_epi16(T20, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
    T31 = _mm_madd_epi16(T21, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
    T32 = _mm_madd_epi16(T22, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
    T33 = _mm_madd_epi16(T23, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
    T34 = _mm_madd_epi16(T24, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
    T35 = _mm_madd_epi16(T25, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
    T36 = _mm_madd_epi16(T26, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
    T37 = _mm_madd_epi16(T27, _mm_load_si128((__m128i*)tab_dct_16_1[8]));

    T40 = _mm_hadd_epi32(T30, T31);
    T41 = _mm_hadd_epi32(T32, T33);
    T42 = _mm_hadd_epi32(T34, T35);
    T43 = _mm_hadd_epi32(T36, T37);

    T50 = _mm_hadd_epi32(T40, T41);
    T51 = _mm_hadd_epi32(T42, T43);
    T50 = _mm_srai_epi32(_mm_add_epi32(T50, c_4), shift1);
    T51 = _mm_srai_epi32(_mm_add_epi32(T51, c_4), shift1);
    T60 = _mm_packs_epi32(T50, T51);
    im[8] = T60;
    //_mm_storeu_si128((__m128i*)&dst[8 * 8], T60);

    T30 = _mm_madd_epi16(T20, _mm_load_si128((__m128i*)tab_dct_16_1[9]));
    T31 = _mm_madd_epi16(T21, _mm_load_si128((__m128i*)tab_dct_16_1[9]));
    T32 = _mm_madd_epi16(T22, _mm_load_si128((__m128i*)tab_dct_16_1[9]));
    T33 = _mm_madd_epi16(T23, _mm_load_si128((__m128i*)tab_dct_16_1[9]));
    T34 = _mm_madd_epi16(T24, _mm_load_si128((__m128i*)tab_dct_16_1[9]));
    T35 = _mm_madd_epi16(T25, _mm_load_si128((__m128i*)tab_dct_16_1[9]));
    T36 = _mm_madd_epi16(T26, _mm_load_si128((__m128i*)tab_dct_16_1[9]));
    T37 = _mm_madd_epi16(T27, _mm_load_si128((__m128i*)tab_dct_16_1[9]));

    T40 = _mm_hadd_epi32(T30, T31);
    T41 = _mm_hadd_epi32(T32, T33);
    T42 = _mm_hadd_epi32(T34, T35);
    T43 = _mm_hadd_epi32(T36, T37);

    T50 = _mm_hadd_epi32(T40, T41);
    T51 = _mm_hadd_epi32(T42, T43);
    T50 = _mm_srai_epi32(_mm_add_epi32(T50, c_4), shift1);
    T51 = _mm_srai_epi32(_mm_add_epi32(T51, c_4), shift1);
    T60 = _mm_packs_epi32(T50, T51);
    im[24] = T60;
    //_mm_storeu_si128((__m128i*)&dst[24 * 8], T60);

#define MAKE_ODD(tab, dstPos) \
    T30 = _mm_madd_epi16(T20, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T31 = _mm_madd_epi16(T21, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T32 = _mm_madd_epi16(T22, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T33 = _mm_madd_epi16(T23, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T34 = _mm_madd_epi16(T24, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T35 = _mm_madd_epi16(T25, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T36 = _mm_madd_epi16(T26, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T37 = _mm_madd_epi16(T27, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    \
    T40 = _mm_hadd_epi32(T30, T31); \
    T41 = _mm_hadd_epi32(T32, T33); \
    T42 = _mm_hadd_epi32(T34, T35); \
    T43 = _mm_hadd_epi32(T36, T37); \
    \
    T50 = _mm_hadd_epi32(T40, T41); \
    T51 = _mm_hadd_epi32(T42, T43); \
    T50 = _mm_srai_epi32(_mm_add_epi32(T50, c_4), shift1); \
    T51 = _mm_srai_epi32(_mm_add_epi32(T51, c_4), shift1); \
    T60 = _mm_packs_epi32(T50, T51); \
    im[(dstPos)] = T60;
    //_mm_storeu_si128((__m128i*)&dst[dstPos * 8], T60);

    MAKE_ODD(0, 4);
    MAKE_ODD(1, 12);
    MAKE_ODD(2, 20);
    MAKE_ODD(3, 28);

    T20 = _mm_sub_epi16(T10A, T10B);   // [EO5 EO2 EO6 EO1 EO4 EO3 EO7 EO0]
    T21 = _mm_sub_epi16(T11A, T11B);
    T22 = _mm_sub_epi16(T12A, T12B);
    T23 = _mm_sub_epi16(T13A, T13B);
    T24 = _mm_sub_epi16(T14A, T14B);
    T25 = _mm_sub_epi16(T15A, T15B);
    T26 = _mm_sub_epi16(T16A, T16B);
    T27 = _mm_sub_epi16(T17A, T17B);

    MAKE_ODD(4, 2);
    MAKE_ODD(5, 6);
    MAKE_ODD(6, 10);
    MAKE_ODD(7, 14);
    MAKE_ODD(8, 18);
    MAKE_ODD(9, 22);
    MAKE_ODD(10, 26);
    MAKE_ODD(11, 30);
#undef MAKE_ODD

#define MAKE_ODD(tab, dstPos) \
    T20 = _mm_madd_epi16(T00A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T21 = _mm_madd_epi16(T00B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)+1])); \
    T22 = _mm_madd_epi16(T01A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T23 = _mm_madd_epi16(T01B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)+1])); \
    T24 = _mm_madd_epi16(T02A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T25 = _mm_madd_epi16(T02B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)+1])); \
    T26 = _mm_madd_epi16(T03A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T27 = _mm_madd_epi16(T03B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)+1])); \
    T30 = _mm_madd_epi16(T04A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T31 = _mm_madd_epi16(T04B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)+1])); \
    T32 = _mm_madd_epi16(T05A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T33 = _mm_madd_epi16(T05B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)+1])); \
    T34 = _mm_madd_epi16(T06A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T35 = _mm_madd_epi16(T06B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)+1])); \
    T36 = _mm_madd_epi16(T07A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T37 = _mm_madd_epi16(T07B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)+1])); \
    \
    T40 = _mm_hadd_epi32(T20, T21); \
    T41 = _mm_hadd_epi32(T22, T23); \
    T42 = _mm_hadd_epi32(T24, T25); \
    T43 = _mm_hadd_epi32(T26, T27); \
    T44 = _mm_hadd_epi32(T30, T31); \
    T45 = _mm_hadd_epi32(T32, T33); \
    T46 = _mm_hadd_epi32(T34, T35); \
    T47 = _mm_hadd_epi32(T36, T37); \
    \
    T50 = _mm_hadd_epi32(T40, T41); \
    T51 = _mm_hadd_epi32(T42, T43); \
    T52 = _mm_hadd_epi32(T44, T45); \
    T53 = _mm_hadd_epi32(T46, T47); \
    \
    T50 = _mm_hadd_epi32(T50, T51); \
    T51 = _mm_hadd_epi32(T52, T53); \
    T50 = _mm_srai_epi32(_mm_add_epi32(T50, c_4), shift1); \
    T51 = _mm_srai_epi32(_mm_add_epi32(T51, c_4), shift1); \
    T60 = _mm_packs_epi32(T50, T51); \
    im[(dstPos)] = T60;
    //_mm_storeu_si128((__m128i*)&dst[dstPos * 8], T60);

    MAKE_ODD(12, 1);
    MAKE_ODD(14, 3);
    MAKE_ODD(16, 5);
    MAKE_ODD(18, 7);
    MAKE_ODD(20, 9);
    MAKE_ODD(22, 11);
    MAKE_ODD(24, 13);
    MAKE_ODD(26, 15);
    MAKE_ODD(28, 17);
    MAKE_ODD(30, 19);
    MAKE_ODD(32, 21);
    MAKE_ODD(34, 23);
    MAKE_ODD(36, 25);
    MAKE_ODD(38, 27);
    MAKE_ODD(40, 29);
    MAKE_ODD(42, 31);

#undef MAKE_ODD

    //DCT2
    for (i = 0; i < 32 / 8; i++) {
        /*in0 = _mm_loadu_si128((const __m128i *)(src + (0 + i * 8) * 8));
        in1 = _mm_loadu_si128((const __m128i *)(src + (1 + i * 8) * 8));
        in2 = _mm_loadu_si128((const __m128i *)(src + (2 + i * 8) * 8));
        in3 = _mm_loadu_si128((const __m128i *)(src + (3 + i * 8) * 8));
        in4 = _mm_loadu_si128((const __m128i *)(src + (4 + i * 8) * 8));
        in5 = _mm_loadu_si128((const __m128i *)(src + (5 + i * 8) * 8));
        in6 = _mm_loadu_si128((const __m128i *)(src + (6 + i * 8) * 8));
        in7 = _mm_loadu_si128((const __m128i *)(src + (7 + i * 8) * 8));*/

#define MAKE_ODD(tab)\
    q0 = _mm_madd_epi16(im[8 * i + 0], _mm_load_si128((__m128i*)tab_dct_8_1[tab])); \
    q1 = _mm_madd_epi16(im[8 * i + 1], _mm_load_si128((__m128i*)tab_dct_8_1[tab])); \
    q2 = _mm_madd_epi16(im[8 * i + 2], _mm_load_si128((__m128i*)tab_dct_8_1[tab])); \
    q3 = _mm_madd_epi16(im[8 * i + 3], _mm_load_si128((__m128i*)tab_dct_8_1[tab])); \
    q4 = _mm_madd_epi16(im[8 * i + 4], _mm_load_si128((__m128i*)tab_dct_8_1[tab])); \
    q5 = _mm_madd_epi16(im[8 * i + 5], _mm_load_si128((__m128i*)tab_dct_8_1[tab])); \
    q6 = _mm_madd_epi16(im[8 * i + 6], _mm_load_si128((__m128i*)tab_dct_8_1[tab])); \
    q7 = _mm_madd_epi16(im[8 * i + 7], _mm_load_si128((__m128i*)tab_dct_8_1[tab])); \
    r0 = _mm_hadd_epi32(q0, q1); \
    r1 = _mm_hadd_epi32(q2, q3); \
    r2 = _mm_hadd_epi32(q4, q5); \
    r3 = _mm_hadd_epi32(q6, q7); \
    v0 = _mm_hadd_epi32(r0, r1); \
    v1 = _mm_hadd_epi32(r2, r3); \
    v0 = _mm_add_epi32(v0, k_ROUNDING2); \
    v1 = _mm_add_epi32(v1, k_ROUNDING2); \
    w0 = _mm_srai_epi32(v0, shift2); \
    w1 = _mm_srai_epi32(v1, shift2); \
    t0 = _mm_packs_epi32(w0, w1); \
    _mm_storeu_si128((__m128i *)(dst + 32 * tab + i * 8), t0);

        MAKE_ODD(0);
        MAKE_ODD(1);
        MAKE_ODD(2);
        MAKE_ODD(3);
        MAKE_ODD(4);
        MAKE_ODD(5);
        MAKE_ODD(6);
        MAKE_ODD(7);
#undef MAKE_ODD
    }
}

/* ---------------------------------------------------------------------------
 */
void dct_c_32x32_half_sse128(const coeff_t *src, coeff_t *dst, int i_src)
{
    const int shift1 = B32X32_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT + (i_src & 0x01);
    const int SHIFT2 = B32X32_IN_BIT + FACTO_BIT;
    const int ADD1 = (1 << shift1) >> 1;
    const int ADD2 = (1 << SHIFT2) >> 1;

    // Const
    __m128i c_add1  = _mm_set1_epi32(ADD1);
    __m128i c_add2  = _mm_set1_epi32(ADD2);

    __m128i T00A, T01A, T02A, T03A, T04A, T05A, T06A, T07A;
    __m128i T00B, T01B, T02B, T03B, T04B, T05B, T06B, T07B;
    __m128i T00C, T01C, T02C, T03C, T04C, T05C, T06C, T07C;
    __m128i T00D, T01D, T02D, T03D, T04D, T05D, T06D, T07D;
    __m128i T10A, T11A, T12A, T13A, T14A, T15A, T16A, T17A;
    __m128i T10B, T11B, T12B, T13B, T14B, T15B, T16B, T17B;
    __m128i T20, T21, T22, T23, T24, T25, T26, T27;
    __m128i T30, T31, T32, T33, T34, T35, T36, T37;
    __m128i T40, T41, T42, T43, T44, T45, T46, T47;
    __m128i T50, T51, T52, T53;
    __m128i T60, T61, T62, T63, T64, T65, T66, T67;
    __m128i im[16][4];
    int i;

    i_src &= 0xFE;    /* remember to remove the flag bit */

    // DCT1
    for (i = 0; i < 32 / 8; i++) {
        T00A = _mm_load_si128((__m128i*)(src + (i * 8 + 0) * i_src +  0));    // [07 06 05 04 03 02 01 00]
        T00B = _mm_load_si128((__m128i*)(src + (i * 8 + 0) * i_src +  8));    // [15 14 13 12 11 10 09 08]
        T00C = _mm_load_si128((__m128i*)(src + (i * 8 + 0) * i_src + 16));    // [23 22 21 20 19 18 17 16]
        T00D = _mm_load_si128((__m128i*)(src + (i * 8 + 0) * i_src + 24));    // [31 30 29 28 27 26 25 24]
        T01A = _mm_load_si128((__m128i*)(src + (i * 8 + 1) * i_src +  0));
        T01B = _mm_load_si128((__m128i*)(src + (i * 8 + 1) * i_src +  8));
        T01C = _mm_load_si128((__m128i*)(src + (i * 8 + 1) * i_src + 16));
        T01D = _mm_load_si128((__m128i*)(src + (i * 8 + 1) * i_src + 24));
        T02A = _mm_load_si128((__m128i*)(src + (i * 8 + 2) * i_src +  0));
        T02B = _mm_load_si128((__m128i*)(src + (i * 8 + 2) * i_src +  8));
        T02C = _mm_load_si128((__m128i*)(src + (i * 8 + 2) * i_src + 16));
        T02D = _mm_load_si128((__m128i*)(src + (i * 8 + 2) * i_src + 24));
        T03A = _mm_load_si128((__m128i*)(src + (i * 8 + 3) * i_src +  0));
        T03B = _mm_load_si128((__m128i*)(src + (i * 8 + 3) * i_src +  8));
        T03C = _mm_load_si128((__m128i*)(src + (i * 8 + 3) * i_src + 16));
        T03D = _mm_load_si128((__m128i*)(src + (i * 8 + 3) * i_src + 24));
        T04A = _mm_load_si128((__m128i*)(src + (i * 8 + 4) * i_src +  0));
        T04B = _mm_load_si128((__m128i*)(src + (i * 8 + 4) * i_src +  8));
        T04C = _mm_load_si128((__m128i*)(src + (i * 8 + 4) * i_src + 16));
        T04D = _mm_load_si128((__m128i*)(src + (i * 8 + 4) * i_src + 24));
        T05A = _mm_load_si128((__m128i*)(src + (i * 8 + 5) * i_src +  0));
        T05B = _mm_load_si128((__m128i*)(src + (i * 8 + 5) * i_src +  8));
        T05C = _mm_load_si128((__m128i*)(src + (i * 8 + 5) * i_src + 16));
        T05D = _mm_load_si128((__m128i*)(src + (i * 8 + 5) * i_src + 24));
        T06A = _mm_load_si128((__m128i*)(src + (i * 8 + 6) * i_src +  0));
        T06B = _mm_load_si128((__m128i*)(src + (i * 8 + 6) * i_src +  8));
        T06C = _mm_load_si128((__m128i*)(src + (i * 8 + 6) * i_src + 16));
        T06D = _mm_load_si128((__m128i*)(src + (i * 8 + 6) * i_src + 24));
        T07A = _mm_load_si128((__m128i*)(src + (i * 8 + 7) * i_src +  0));
        T07B = _mm_load_si128((__m128i*)(src + (i * 8 + 7) * i_src +  8));
        T07C = _mm_load_si128((__m128i*)(src + (i * 8 + 7) * i_src + 16));
        T07D = _mm_load_si128((__m128i*)(src + (i * 8 + 7) * i_src + 24));

        T00A = _mm_shuffle_epi8(T00A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));   // [05 02 06 01 04 03 07 00]
        T00B = _mm_shuffle_epi8(T00B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));   // [10 13 09 14 11 12 08 15]
        T00C = _mm_shuffle_epi8(T00C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));   // [21 18 22 17 20 19 23 16]
        T00D = _mm_shuffle_epi8(T00D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));   // [26 29 25 30 27 28 24 31]
        T01A = _mm_shuffle_epi8(T01A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T01B = _mm_shuffle_epi8(T01B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T01C = _mm_shuffle_epi8(T01C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T01D = _mm_shuffle_epi8(T01D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T02A = _mm_shuffle_epi8(T02A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T02B = _mm_shuffle_epi8(T02B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T02C = _mm_shuffle_epi8(T02C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T02D = _mm_shuffle_epi8(T02D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T03A = _mm_shuffle_epi8(T03A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T03B = _mm_shuffle_epi8(T03B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T03C = _mm_shuffle_epi8(T03C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T03D = _mm_shuffle_epi8(T03D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T04A = _mm_shuffle_epi8(T04A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T04B = _mm_shuffle_epi8(T04B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T04C = _mm_shuffle_epi8(T04C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T04D = _mm_shuffle_epi8(T04D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T05A = _mm_shuffle_epi8(T05A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T05B = _mm_shuffle_epi8(T05B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T05C = _mm_shuffle_epi8(T05C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T05D = _mm_shuffle_epi8(T05D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T06A = _mm_shuffle_epi8(T06A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T06B = _mm_shuffle_epi8(T06B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T06C = _mm_shuffle_epi8(T06C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T06D = _mm_shuffle_epi8(T06D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T07A = _mm_shuffle_epi8(T07A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T07B = _mm_shuffle_epi8(T07B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T07C = _mm_shuffle_epi8(T07C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T07D = _mm_shuffle_epi8(T07D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));

        T10A = _mm_add_epi16(T00A, T00D);   // [E05 E02 E06 E01 E04 E03 E07 E00]
        T10B = _mm_add_epi16(T00B, T00C);   // [E10 E13 E09 E14 E11 E12 E08 E15]
        T11A = _mm_add_epi16(T01A, T01D);
        T11B = _mm_add_epi16(T01B, T01C);
        T12A = _mm_add_epi16(T02A, T02D);
        T12B = _mm_add_epi16(T02B, T02C);
        T13A = _mm_add_epi16(T03A, T03D);
        T13B = _mm_add_epi16(T03B, T03C);
        T14A = _mm_add_epi16(T04A, T04D);
        T14B = _mm_add_epi16(T04B, T04C);
        T15A = _mm_add_epi16(T05A, T05D);
        T15B = _mm_add_epi16(T05B, T05C);
        T16A = _mm_add_epi16(T06A, T06D);
        T16B = _mm_add_epi16(T06B, T06C);
        T17A = _mm_add_epi16(T07A, T07D);
        T17B = _mm_add_epi16(T07B, T07C);

        T00A = _mm_sub_epi16(T00A, T00D);   // [O05 O02 O06 O01 O04 O03 O07 O00]
        T00B = _mm_sub_epi16(T00B, T00C);   // [O10 O13 O09 O14 O11 O12 O08 O15]
        T01A = _mm_sub_epi16(T01A, T01D);
        T01B = _mm_sub_epi16(T01B, T01C);
        T02A = _mm_sub_epi16(T02A, T02D);
        T02B = _mm_sub_epi16(T02B, T02C);
        T03A = _mm_sub_epi16(T03A, T03D);
        T03B = _mm_sub_epi16(T03B, T03C);
        T04A = _mm_sub_epi16(T04A, T04D);
        T04B = _mm_sub_epi16(T04B, T04C);
        T05A = _mm_sub_epi16(T05A, T05D);
        T05B = _mm_sub_epi16(T05B, T05C);
        T06A = _mm_sub_epi16(T06A, T06D);
        T06B = _mm_sub_epi16(T06B, T06C);
        T07A = _mm_sub_epi16(T07A, T07D);
        T07B = _mm_sub_epi16(T07B, T07C);

        T20  = _mm_add_epi16(T10A, T10B);   // [EE5 EE2 EE6 EE1 EE4 EE3 EE7 EE0]
        T21  = _mm_add_epi16(T11A, T11B);
        T22  = _mm_add_epi16(T12A, T12B);
        T23  = _mm_add_epi16(T13A, T13B);
        T24  = _mm_add_epi16(T14A, T14B);
        T25  = _mm_add_epi16(T15A, T15B);
        T26  = _mm_add_epi16(T16A, T16B);
        T27  = _mm_add_epi16(T17A, T17B);

        T30  = _mm_madd_epi16(T20, _mm_load_si128((__m128i*)tab_dct_8[1]));//  [ 05+02 06+01 04+03 07+00]
        T31  = _mm_madd_epi16(T21, _mm_load_si128((__m128i*)tab_dct_8[1]));
        T32  = _mm_madd_epi16(T22, _mm_load_si128((__m128i*)tab_dct_8[1]));
        T33  = _mm_madd_epi16(T23, _mm_load_si128((__m128i*)tab_dct_8[1]));
        T34  = _mm_madd_epi16(T24, _mm_load_si128((__m128i*)tab_dct_8[1]));
        T35  = _mm_madd_epi16(T25, _mm_load_si128((__m128i*)tab_dct_8[1]));
        T36  = _mm_madd_epi16(T26, _mm_load_si128((__m128i*)tab_dct_8[1]));
        T37  = _mm_madd_epi16(T27, _mm_load_si128((__m128i*)tab_dct_8[1]));

        T40  = _mm_hadd_epi32(T30, T31);//[05+02+06+01 04+03+07+00]
        T41  = _mm_hadd_epi32(T32, T33);
        T42  = _mm_hadd_epi32(T34, T35);
        T43  = _mm_hadd_epi32(T36, T37);

        T50  = _mm_hadd_epi32(T40, T41);
        T51  = _mm_hadd_epi32(T42, T43);
        T50  = _mm_srai_epi32(_mm_add_epi32(T50, c_add1), shift1);
        T51  = _mm_srai_epi32(_mm_add_epi32(T51, c_add1), shift1);
        T60  = _mm_packs_epi32(T50, T51);
        im[0][i] = T60;

        T30  = _mm_madd_epi16(T20, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
        T31  = _mm_madd_epi16(T21, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
        T32  = _mm_madd_epi16(T22, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
        T33  = _mm_madd_epi16(T23, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
        T34  = _mm_madd_epi16(T24, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
        T35  = _mm_madd_epi16(T25, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
        T36  = _mm_madd_epi16(T26, _mm_load_si128((__m128i*)tab_dct_16_1[8]));
        T37  = _mm_madd_epi16(T27, _mm_load_si128((__m128i*)tab_dct_16_1[8]));

        T40  = _mm_hadd_epi32(T30, T31);
        T41  = _mm_hadd_epi32(T32, T33);
        T42  = _mm_hadd_epi32(T34, T35);
        T43  = _mm_hadd_epi32(T36, T37);

        T50  = _mm_hadd_epi32(T40, T41);
        T51  = _mm_hadd_epi32(T42, T43);
        T50  = _mm_srai_epi32(_mm_add_epi32(T50, c_add1), shift1);
        T51  = _mm_srai_epi32(_mm_add_epi32(T51, c_add1), shift1);
        T60  = _mm_packs_epi32(T50, T51);
        im[8][i] = T60;

#define MAKE_ODD(tab, dstPos) \
    T30  = _mm_madd_epi16(T20, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T31  = _mm_madd_epi16(T21, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T32  = _mm_madd_epi16(T22, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T33  = _mm_madd_epi16(T23, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T34  = _mm_madd_epi16(T24, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T35  = _mm_madd_epi16(T25, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T36  = _mm_madd_epi16(T26, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    T37  = _mm_madd_epi16(T27, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)])); \
    \
    T40  = _mm_hadd_epi32(T30, T31); \
    T41  = _mm_hadd_epi32(T32, T33); \
    T42  = _mm_hadd_epi32(T34, T35); \
    T43  = _mm_hadd_epi32(T36, T37); \
    \
    T50  = _mm_hadd_epi32(T40, T41); \
    T51  = _mm_hadd_epi32(T42, T43); \
    T50  = _mm_srai_epi32(_mm_add_epi32(T50, c_add1), shift1); \
    T51  = _mm_srai_epi32(_mm_add_epi32(T51, c_add1), shift1); \
    T60  = _mm_packs_epi32(T50, T51); \
    im[(dstPos)][i] = T60;

        MAKE_ODD(0, 4);
        MAKE_ODD(1, 12);

        T20  = _mm_sub_epi16(T10A, T10B);   // [EO5 EO2 EO6 EO1 EO4 EO3 EO7 EO0]
        T21  = _mm_sub_epi16(T11A, T11B);
        T22  = _mm_sub_epi16(T12A, T12B);
        T23  = _mm_sub_epi16(T13A, T13B);
        T24  = _mm_sub_epi16(T14A, T14B);
        T25  = _mm_sub_epi16(T15A, T15B);
        T26  = _mm_sub_epi16(T16A, T16B);
        T27  = _mm_sub_epi16(T17A, T17B);

        MAKE_ODD( 4,  2);
        MAKE_ODD( 5,  6);
        MAKE_ODD( 6, 10);
        MAKE_ODD( 7, 14);
#undef MAKE_ODD

#define MAKE_ODD(tab, dstPos) \
    T20  = _mm_madd_epi16(T00A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)    ])); \
    T21  = _mm_madd_epi16(T00B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab) + 1])); \
    T22  = _mm_madd_epi16(T01A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)    ])); \
    T23  = _mm_madd_epi16(T01B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab) + 1])); \
    T24  = _mm_madd_epi16(T02A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)    ])); \
    T25  = _mm_madd_epi16(T02B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab) + 1])); \
    T26  = _mm_madd_epi16(T03A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)    ])); \
    T27  = _mm_madd_epi16(T03B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab) + 1])); \
    T30  = _mm_madd_epi16(T04A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)    ])); \
    T31  = _mm_madd_epi16(T04B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab) + 1])); \
    T32  = _mm_madd_epi16(T05A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)    ])); \
    T33  = _mm_madd_epi16(T05B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab) + 1])); \
    T34  = _mm_madd_epi16(T06A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)    ])); \
    T35  = _mm_madd_epi16(T06B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab) + 1])); \
    T36  = _mm_madd_epi16(T07A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab)    ])); \
    T37  = _mm_madd_epi16(T07B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab) + 1])); \
    \
    T40  = _mm_hadd_epi32(T20, T21); \
    T41  = _mm_hadd_epi32(T22, T23); \
    T42  = _mm_hadd_epi32(T24, T25); \
    T43  = _mm_hadd_epi32(T26, T27); \
    T44  = _mm_hadd_epi32(T30, T31); \
    T45  = _mm_hadd_epi32(T32, T33); \
    T46  = _mm_hadd_epi32(T34, T35); \
    T47  = _mm_hadd_epi32(T36, T37); \
    \
    T50  = _mm_hadd_epi32(T40, T41); \
    T51  = _mm_hadd_epi32(T42, T43); \
    T52  = _mm_hadd_epi32(T44, T45); \
    T53  = _mm_hadd_epi32(T46, T47); \
    \
    T50  = _mm_hadd_epi32(T50, T51); \
    T51  = _mm_hadd_epi32(T52, T53); \
    T50  = _mm_srai_epi32(_mm_add_epi32(T50, c_add1), shift1); \
    T51  = _mm_srai_epi32(_mm_add_epi32(T51, c_add1), shift1); \
    T60  = _mm_packs_epi32(T50, T51); \
    im[(dstPos)][i] = T60;

        MAKE_ODD(12,  1);
        MAKE_ODD(14,  3);
        MAKE_ODD(16,  5);
        MAKE_ODD(18,  7);
        MAKE_ODD(20,  9);
        MAKE_ODD(22, 11);
        MAKE_ODD(24, 13);
        MAKE_ODD(26, 15);

#undef MAKE_ODD
    }

    /* clear result buffer */
    xavs2_fast_memzero_mmx(dst, 32 * 32 * sizeof(coeff_t));

    // DCT2, Ö»±£ÁôÇ°16ÐÐºÍÇ°16ÁÐ
    for (i = 0; i < 16 / 4; i++) {
        // OPT_ME: to avoid register spill, I use matrix multiply, have other way?
        T00A = im[i * 4 + 0][0];    // [07 06 05 04 03 02 01 00]
        T00B = im[i * 4 + 0][1];    // [15 14 13 12 11 10 09 08]
        T00C = im[i * 4 + 0][2];    // [23 22 21 20 19 18 17 16]
        T00D = im[i * 4 + 0][3];    // [31 30 29 28 27 26 25 24]
        T01A = im[i * 4 + 1][0];
        T01B = im[i * 4 + 1][1];
        T01C = im[i * 4 + 1][2];
        T01D = im[i * 4 + 1][3];
        T02A = im[i * 4 + 2][0];
        T02B = im[i * 4 + 2][1];
        T02C = im[i * 4 + 2][2];
        T02D = im[i * 4 + 2][3];
        T03A = im[i * 4 + 3][0];
        T03B = im[i * 4 + 3][1];
        T03C = im[i * 4 + 3][2];
        T03D = im[i * 4 + 3][3];

        T00C = _mm_shuffle_epi8(T00C, _mm_load_si128((__m128i*)tab_dct_16_0[0]));    // [16 17 18 19 20 21 22 23]
        T00D = _mm_shuffle_epi8(T00D, _mm_load_si128((__m128i*)tab_dct_16_0[0]));    // [24 25 26 27 28 29 30 31]
        T01C = _mm_shuffle_epi8(T01C, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
        T01D = _mm_shuffle_epi8(T01D, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
        T02C = _mm_shuffle_epi8(T02C, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
        T02D = _mm_shuffle_epi8(T02D, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
        T03C = _mm_shuffle_epi8(T03C, _mm_load_si128((__m128i*)tab_dct_16_0[0]));
        T03D = _mm_shuffle_epi8(T03D, _mm_load_si128((__m128i*)tab_dct_16_0[0]));

        T10A = _mm_unpacklo_epi16(T00A, T00D);  // [28 03 29 02 30 01 31 00]
        T10B = _mm_unpackhi_epi16(T00A, T00D);  // [24 07 25 06 26 05 27 04]
        T00A = _mm_unpacklo_epi16(T00B, T00C);  // [20 11 21 10 22 09 23 08]
        T00B = _mm_unpackhi_epi16(T00B, T00C);  // [16 15 17 14 18 13 19 12]
        T11A = _mm_unpacklo_epi16(T01A, T01D);
        T11B = _mm_unpackhi_epi16(T01A, T01D);
        T01A = _mm_unpacklo_epi16(T01B, T01C);
        T01B = _mm_unpackhi_epi16(T01B, T01C);
        T12A = _mm_unpacklo_epi16(T02A, T02D);
        T12B = _mm_unpackhi_epi16(T02A, T02D);
        T02A = _mm_unpacklo_epi16(T02B, T02C);
        T02B = _mm_unpackhi_epi16(T02B, T02C);
        T13A = _mm_unpacklo_epi16(T03A, T03D);
        T13B = _mm_unpackhi_epi16(T03A, T03D);
        T03A = _mm_unpacklo_epi16(T03B, T03C);
        T03B = _mm_unpackhi_epi16(T03B, T03C);

#define MAKE_ODD(tab0, tab1, tab2, tab3, dstPos) \
    T20  = _mm_madd_epi16(T10A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab0)])); \
    T21  = _mm_madd_epi16(T10B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab1)])); \
    T22  = _mm_madd_epi16(T00A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab2)])); \
    T23  = _mm_madd_epi16(T00B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab3)])); \
    T24  = _mm_madd_epi16(T11A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab0)])); \
    T25  = _mm_madd_epi16(T11B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab1)])); \
    T26  = _mm_madd_epi16(T01A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab2)])); \
    T27  = _mm_madd_epi16(T01B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab3)])); \
    T30  = _mm_madd_epi16(T12A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab0)])); \
    T31  = _mm_madd_epi16(T12B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab1)])); \
    T32  = _mm_madd_epi16(T02A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab2)])); \
    T33  = _mm_madd_epi16(T02B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab3)])); \
    T34  = _mm_madd_epi16(T13A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab0)])); \
    T35  = _mm_madd_epi16(T13B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab1)])); \
    T36  = _mm_madd_epi16(T03A, _mm_load_si128((__m128i*)tab_dct_32_1[(tab2)])); \
    T37  = _mm_madd_epi16(T03B, _mm_load_si128((__m128i*)tab_dct_32_1[(tab3)])); \
    \
    T60  = _mm_hadd_epi32(T20, T21); \
    T61  = _mm_hadd_epi32(T22, T23); \
    T62  = _mm_hadd_epi32(T24, T25); \
    T63  = _mm_hadd_epi32(T26, T27); \
    T64  = _mm_hadd_epi32(T30, T31); \
    T65  = _mm_hadd_epi32(T32, T33); \
    T66  = _mm_hadd_epi32(T34, T35); \
    T67  = _mm_hadd_epi32(T36, T37); \
    \
    T60  = _mm_hadd_epi32(T60, T61); \
    T61  = _mm_hadd_epi32(T62, T63); \
    T62  = _mm_hadd_epi32(T64, T65); \
    T63  = _mm_hadd_epi32(T66, T67); \
    \
    T60  = _mm_hadd_epi32(T60, T61); \
    T61  = _mm_hadd_epi32(T62, T63); \
    \
    T60  = _mm_hadd_epi32(T60, T61); \
    \
    T60  = _mm_srai_epi32(_mm_add_epi32(T60, c_add2), SHIFT2); \
    T60  = _mm_packs_epi32(T60, T60); \
    _mm_storel_epi64((__m128i*)(dst + (dstPos) * 32 + (i * 4) + 0), T60);

        MAKE_ODD(44, 44, 44, 44,  0);
        MAKE_ODD(46, 47, 46, 47,  8);

        MAKE_ODD(50, 51, 52, 53,  4);
        MAKE_ODD(54, 55, 56, 57, 12);

        MAKE_ODD(66, 67, 68, 69,  2);
        MAKE_ODD(70, 71, 72, 73,  6);
        MAKE_ODD(74, 75, 76, 77, 10);
        MAKE_ODD(78, 79, 80, 81, 14);

        MAKE_ODD( 98,  99, 100, 101,  1);
        MAKE_ODD(102, 103, 104, 105,  3);
        MAKE_ODD(106, 107, 108, 109,  5);
        MAKE_ODD(110, 111, 112, 113,  7);
        MAKE_ODD(114, 115, 116, 117,  9);
        MAKE_ODD(118, 119, 120, 121, 11);
        MAKE_ODD(122, 123, 124, 125, 13);
        MAKE_ODD(126, 127, 128, 129, 15);
#undef MAKE_ODD
    }
}

//optimize 32x32 size transform
void dct_c_32x32_sse128(const coeff_t *src, coeff_t *dst, int i_src)
{
    const int shift1 = B32X32_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT + (i_src & 0x01);
    const int SHIFT2 = B32X32_IN_BIT + FACTO_BIT;
    const int ADD1 = (1 << shift1) >> 1;
    const int ADD2 = (1 << SHIFT2) >> 1;

    // Const
    __m128i c_add1 = _mm_set1_epi32(ADD1);
    __m128i c_add2 = _mm_set1_epi32(ADD2);

    __m128i T00A, T01A, T02A, T03A, T04A, T05A, T06A, T07A;
    __m128i T00B, T01B, T02B, T03B, T04B, T05B, T06B, T07B;
    __m128i T00C, T01C, T02C, T03C, T04C, T05C, T06C, T07C;
    __m128i T00D, T01D, T02D, T03D, T04D, T05D, T06D, T07D;
    __m128i T10A, T11A, T12A, T13A, T14A, T15A, T16A, T17A;
    __m128i T10B, T11B, T12B, T13B, T14B, T15B, T16B, T17B;
    __m128i T20, T21, T22, T23, T24, T25, T26, T27;
    __m128i T30, T31, T32, T33, T34, T35, T36, T37;
    __m128i T40, T41, T42, T43, T44, T45, T46, T47;
    __m128i T50, T51, T52, T53;
    __m128i T60;
    __m128i im[32][4];
    int i;

    i_src &= 0xFE;    /* remember to remove the flag bit */

    // DCT1
    for (i = 0; i < 32 / 8; i++) {
        T00A = _mm_load_si128((__m128i*)(src + 0));    // [07 06 05 04 03 02 01 00]
        T00B = _mm_load_si128((__m128i*)(src + 8));    // [15 14 13 12 11 10 09 08]
        T00C = _mm_load_si128((__m128i*)(src + 16));    // [23 22 21 20 19 18 17 16]
        T00D = _mm_load_si128((__m128i*)(src + 24));    // [31 30 29 28 27 26 25 24]
        src += i_src;
        T01A = _mm_load_si128((__m128i*)(src + 0));
        T01B = _mm_load_si128((__m128i*)(src + 8));
        T01C = _mm_load_si128((__m128i*)(src + 16));
        T01D = _mm_load_si128((__m128i*)(src + 24));
        src += i_src;
        T02A = _mm_load_si128((__m128i*)(src + 0));
        T02B = _mm_load_si128((__m128i*)(src + 8));
        T02C = _mm_load_si128((__m128i*)(src + 16));
        T02D = _mm_load_si128((__m128i*)(src + 24));
        src += i_src;
        T03A = _mm_load_si128((__m128i*)(src + 0));
        T03B = _mm_load_si128((__m128i*)(src + 8));
        T03C = _mm_load_si128((__m128i*)(src + 16));
        T03D = _mm_load_si128((__m128i*)(src + 24));
        src += i_src;
        T04A = _mm_load_si128((__m128i*)(src + 0));
        T04B = _mm_load_si128((__m128i*)(src + 8));
        T04C = _mm_load_si128((__m128i*)(src + 16));
        T04D = _mm_load_si128((__m128i*)(src + 24));
        src += i_src;
        T05A = _mm_load_si128((__m128i*)(src + 0));
        T05B = _mm_load_si128((__m128i*)(src + 8));
        T05C = _mm_load_si128((__m128i*)(src + 16));
        T05D = _mm_load_si128((__m128i*)(src + 24));
        src += i_src;
        T06A = _mm_load_si128((__m128i*)(src + 0));
        T06B = _mm_load_si128((__m128i*)(src + 8));
        T06C = _mm_load_si128((__m128i*)(src + 16));
        T06D = _mm_load_si128((__m128i*)(src + 24));
        src += i_src;
        T07A = _mm_load_si128((__m128i*)(src + 0));
        T07B = _mm_load_si128((__m128i*)(src + 8));
        T07C = _mm_load_si128((__m128i*)(src + 16));
        T07D = _mm_load_si128((__m128i*)(src + 24));
        src += i_src;

        //_mm_load_si128((__m128i)tab_dct_16_0[1]) »»³É *((__m128i*)tab_dct_16_0[1])
        T00A = _mm_shuffle_epi8(T00A, *((__m128i*)tab_dct_16_0[1]));   // [05 02 06 01 04 03 07 00]
        T00B = _mm_shuffle_epi8(T00B, *((__m128i*)tab_dct_32_0[0]));   // [10 13 09 14 11 12 08 15]
        T00C = _mm_shuffle_epi8(T00C, *((__m128i*)tab_dct_16_0[1]));   // [21 18 22 17 20 19 23 16]
        T00D = _mm_shuffle_epi8(T00D, *((__m128i*)tab_dct_32_0[0]));   // [26 29 25 30 27 28 24 31]
        T01A = _mm_shuffle_epi8(T01A, *((__m128i*)tab_dct_16_0[1]));
        T01B = _mm_shuffle_epi8(T01B, *((__m128i*)tab_dct_32_0[0]));
        T01C = _mm_shuffle_epi8(T01C, *((__m128i*)tab_dct_16_0[1]));
        T01D = _mm_shuffle_epi8(T01D, *((__m128i*)tab_dct_32_0[0]));
        T02A = _mm_shuffle_epi8(T02A, *((__m128i*)tab_dct_16_0[1]));
        T02B = _mm_shuffle_epi8(T02B, *((__m128i*)tab_dct_32_0[0]));
        T02C = _mm_shuffle_epi8(T02C, *((__m128i*)tab_dct_16_0[1]));
        T02D = _mm_shuffle_epi8(T02D, *((__m128i*)tab_dct_32_0[0]));
        T03A = _mm_shuffle_epi8(T03A, *((__m128i*)tab_dct_16_0[1]));
        T03B = _mm_shuffle_epi8(T03B, *((__m128i*)tab_dct_32_0[0]));
        T03C = _mm_shuffle_epi8(T03C, *((__m128i*)tab_dct_16_0[1]));
        T03D = _mm_shuffle_epi8(T03D, *((__m128i*)tab_dct_32_0[0]));
        T04A = _mm_shuffle_epi8(T04A, *((__m128i*)tab_dct_16_0[1]));
        T04B = _mm_shuffle_epi8(T04B, *((__m128i*)tab_dct_32_0[0]));
        T04C = _mm_shuffle_epi8(T04C, *((__m128i*)tab_dct_16_0[1]));
        T04D = _mm_shuffle_epi8(T04D, *((__m128i*)tab_dct_32_0[0]));
        T05A = _mm_shuffle_epi8(T05A, *((__m128i*)tab_dct_16_0[1]));
        T05B = _mm_shuffle_epi8(T05B, *((__m128i*)tab_dct_32_0[0]));
        T05C = _mm_shuffle_epi8(T05C, *((__m128i*)tab_dct_16_0[1]));
        T05D = _mm_shuffle_epi8(T05D, *((__m128i*)tab_dct_32_0[0]));
        T06A = _mm_shuffle_epi8(T06A, *((__m128i*)tab_dct_16_0[1]));
        T06B = _mm_shuffle_epi8(T06B, *((__m128i*)tab_dct_32_0[0]));
        T06C = _mm_shuffle_epi8(T06C, *((__m128i*)tab_dct_16_0[1]));
        T06D = _mm_shuffle_epi8(T06D, *((__m128i*)tab_dct_32_0[0]));
        T07A = _mm_shuffle_epi8(T07A, *((__m128i*)tab_dct_16_0[1]));
        T07B = _mm_shuffle_epi8(T07B, *((__m128i*)tab_dct_32_0[0]));
        T07C = _mm_shuffle_epi8(T07C, *((__m128i*)tab_dct_16_0[1]));
        T07D = _mm_shuffle_epi8(T07D, *((__m128i*)tab_dct_32_0[0]));

        T10A = _mm_add_epi16(T00A, T00D);   // [E05 E02 E06 E01 E04 E03 E07 E00]
        T10B = _mm_add_epi16(T00B, T00C);   // [E10 E13 E09 E14 E11 E12 E08 E15]
        T11A = _mm_add_epi16(T01A, T01D);
        T11B = _mm_add_epi16(T01B, T01C);
        T12A = _mm_add_epi16(T02A, T02D);
        T12B = _mm_add_epi16(T02B, T02C);
        T13A = _mm_add_epi16(T03A, T03D);
        T13B = _mm_add_epi16(T03B, T03C);
        T14A = _mm_add_epi16(T04A, T04D);
        T14B = _mm_add_epi16(T04B, T04C);
        T15A = _mm_add_epi16(T05A, T05D);
        T15B = _mm_add_epi16(T05B, T05C);
        T16A = _mm_add_epi16(T06A, T06D);
        T16B = _mm_add_epi16(T06B, T06C);
        T17A = _mm_add_epi16(T07A, T07D);
        T17B = _mm_add_epi16(T07B, T07C);

        T00A = _mm_sub_epi16(T00A, T00D);   // [O05 O02 O06 O01 O04 O03 O07 O00]
        T00B = _mm_sub_epi16(T00B, T00C);   // [O10 O13 O09 O14 O11 O12 O08 O15]
        T01A = _mm_sub_epi16(T01A, T01D);
        T01B = _mm_sub_epi16(T01B, T01C);
        T02A = _mm_sub_epi16(T02A, T02D);
        T02B = _mm_sub_epi16(T02B, T02C);
        T03A = _mm_sub_epi16(T03A, T03D);
        T03B = _mm_sub_epi16(T03B, T03C);
        T04A = _mm_sub_epi16(T04A, T04D);
        T04B = _mm_sub_epi16(T04B, T04C);
        T05A = _mm_sub_epi16(T05A, T05D);
        T05B = _mm_sub_epi16(T05B, T05C);
        T06A = _mm_sub_epi16(T06A, T06D);
        T06B = _mm_sub_epi16(T06B, T06C);
        T07A = _mm_sub_epi16(T07A, T07D);
        T07B = _mm_sub_epi16(T07B, T07C);

        T20 = _mm_add_epi16(T10A, T10B);   // [EE5 EE2 EE6 EE1 EE4 EE3 EE7 EE0]
        T21 = _mm_add_epi16(T11A, T11B);
        T22 = _mm_add_epi16(T12A, T12B);
        T23 = _mm_add_epi16(T13A, T13B);
        T24 = _mm_add_epi16(T14A, T14B);
        T25 = _mm_add_epi16(T15A, T15B);
        T26 = _mm_add_epi16(T16A, T16B);
        T27 = _mm_add_epi16(T17A, T17B);

        //_mm_load_si128((__m128i*)tab_dct_8[1]) ->*((__m128i*)tab_dct_8[1])
        T30 = _mm_madd_epi16(T20, *((__m128i*)tab_dct_8[1]));
        T31 = _mm_madd_epi16(T21, *((__m128i*)tab_dct_8[1]));
        T32 = _mm_madd_epi16(T22, *((__m128i*)tab_dct_8[1]));
        T33 = _mm_madd_epi16(T23, *((__m128i*)tab_dct_8[1]));
        T34 = _mm_madd_epi16(T24, *((__m128i*)tab_dct_8[1]));
        T35 = _mm_madd_epi16(T25, *((__m128i*)tab_dct_8[1]));
        T36 = _mm_madd_epi16(T26, *((__m128i*)tab_dct_8[1]));
        T37 = _mm_madd_epi16(T27, *((__m128i*)tab_dct_8[1]));

        T40 = _mm_hadd_epi32(T30, T31);
        T41 = _mm_hadd_epi32(T32, T33);
        T42 = _mm_hadd_epi32(T34, T35);
        T43 = _mm_hadd_epi32(T36, T37);

        T50 = _mm_hadd_epi32(T40, T41);
        T51 = _mm_hadd_epi32(T42, T43);
        T50 = _mm_srai_epi32(_mm_add_epi32(T50, c_add1), shift1);
        T51 = _mm_srai_epi32(_mm_add_epi32(T51, c_add1), shift1);
        T60 = _mm_packs_epi32(T50, T51);
        im[0][i] = T60;//16¸ö0µ½8ÐÐ¼ÆËã³öÀ´µÄ±ä»»ÏµÊý(16 bit per bit width)

        T50 = _mm_hsub_epi32(T40, T41);
        T51 = _mm_hsub_epi32(T42, T43);
        T50 = _mm_srai_epi32(_mm_add_epi32(T50, c_add1), shift1);
        T51 = _mm_srai_epi32(_mm_add_epi32(T51, c_add1), shift1);
        T60 = _mm_packs_epi32(T50, T51);
        im[16][i] = T60;

        //_mm_load_si128((__m128i*)tab_dct_16_1[8]) ->
        T30 = _mm_madd_epi16(T20, *((__m128i*)tab_dct_16_1[8]));
        T31 = _mm_madd_epi16(T21, *((__m128i*)tab_dct_16_1[8]));
        T32 = _mm_madd_epi16(T22, *((__m128i*)tab_dct_16_1[8]));
        T33 = _mm_madd_epi16(T23, *((__m128i*)tab_dct_16_1[8]));
        T34 = _mm_madd_epi16(T24, *((__m128i*)tab_dct_16_1[8]));
        T35 = _mm_madd_epi16(T25, *((__m128i*)tab_dct_16_1[8]));
        T36 = _mm_madd_epi16(T26, *((__m128i*)tab_dct_16_1[8]));
        T37 = _mm_madd_epi16(T27, *((__m128i*)tab_dct_16_1[8]));

        T40 = _mm_hadd_epi32(T30, T31);
        T41 = _mm_hadd_epi32(T32, T33);
        T42 = _mm_hadd_epi32(T34, T35);
        T43 = _mm_hadd_epi32(T36, T37);

        T50 = _mm_hadd_epi32(T40, T41);
        T51 = _mm_hadd_epi32(T42, T43);
        T50 = _mm_srai_epi32(_mm_add_epi32(T50, c_add1), shift1);
        T51 = _mm_srai_epi32(_mm_add_epi32(T51, c_add1), shift1);
        T60 = _mm_packs_epi32(T50, T51);
        im[8][i] = T60;

        //
        T30 = _mm_madd_epi16(T20, *((__m128i*)tab_dct_16_1[9]));
        T31 = _mm_madd_epi16(T21, *((__m128i*)tab_dct_16_1[9]));
        T32 = _mm_madd_epi16(T22, *((__m128i*)tab_dct_16_1[9]));
        T33 = _mm_madd_epi16(T23, *((__m128i*)tab_dct_16_1[9]));
        T34 = _mm_madd_epi16(T24, *((__m128i*)tab_dct_16_1[9]));
        T35 = _mm_madd_epi16(T25, *((__m128i*)tab_dct_16_1[9]));
        T36 = _mm_madd_epi16(T26, *((__m128i*)tab_dct_16_1[9]));
        T37 = _mm_madd_epi16(T27, *((__m128i*)tab_dct_16_1[9]));

        T40 = _mm_hadd_epi32(T30, T31);
        T41 = _mm_hadd_epi32(T32, T33);
        T42 = _mm_hadd_epi32(T34, T35);
        T43 = _mm_hadd_epi32(T36, T37);

        T50 = _mm_hadd_epi32(T40, T41);
        T51 = _mm_hadd_epi32(T42, T43);
        T50 = _mm_srai_epi32(_mm_add_epi32(T50, c_add1), shift1);
        T51 = _mm_srai_epi32(_mm_add_epi32(T51, c_add1), shift1);
        T60 = _mm_packs_epi32(T50, T51);
        im[24][i] = T60;

#define MAKE_ODD(tab, dstPos) \
    T30 = _mm_madd_epi16(T20, *((__m128i*)tab_dct_32_1[(tab)])); \
    T31 = _mm_madd_epi16(T21, *((__m128i*)tab_dct_32_1[(tab)])); \
    T32 = _mm_madd_epi16(T22, *((__m128i*)tab_dct_32_1[(tab)])); \
    T33 = _mm_madd_epi16(T23, *((__m128i*)tab_dct_32_1[(tab)])); \
    T34 = _mm_madd_epi16(T24, *((__m128i*)tab_dct_32_1[(tab)])); \
    T35 = _mm_madd_epi16(T25, *((__m128i*)tab_dct_32_1[(tab)])); \
    T36 = _mm_madd_epi16(T26, *((__m128i*)tab_dct_32_1[(tab)])); \
    T37 = _mm_madd_epi16(T27, *((__m128i*)tab_dct_32_1[(tab)])); \
    \
    T40 = _mm_hadd_epi32(T30, T31); \
    T41 = _mm_hadd_epi32(T32, T33); \
    T42 = _mm_hadd_epi32(T34, T35); \
    T43 = _mm_hadd_epi32(T36, T37); \
    \
    T50 = _mm_hadd_epi32(T40, T41); \
    T51 = _mm_hadd_epi32(T42, T43); \
    T50 = _mm_srai_epi32(_mm_add_epi32(T50, c_add1), shift1); \
    T51 = _mm_srai_epi32(_mm_add_epi32(T51, c_add1), shift1); \
    T60 = _mm_packs_epi32(T50, T51); \
    im[(dstPos)][i] = T60;

        MAKE_ODD(0, 4);
        MAKE_ODD(1, 12);
        MAKE_ODD(2, 20);
        MAKE_ODD(3, 28);

        T20 = _mm_sub_epi16(T10A, T10B);   // [EO5 EO2 EO6 EO1 EO4 EO3 EO7 EO0]
        T21 = _mm_sub_epi16(T11A, T11B);
        T22 = _mm_sub_epi16(T12A, T12B);
        T23 = _mm_sub_epi16(T13A, T13B);
        T24 = _mm_sub_epi16(T14A, T14B);
        T25 = _mm_sub_epi16(T15A, T15B);
        T26 = _mm_sub_epi16(T16A, T16B);
        T27 = _mm_sub_epi16(T17A, T17B);

        MAKE_ODD(4, 2);
        MAKE_ODD(5, 6);
        MAKE_ODD(6, 10);
        MAKE_ODD(7, 14);
        MAKE_ODD(8, 18);
        MAKE_ODD(9, 22);
        MAKE_ODD(10, 26);
        MAKE_ODD(11, 30);
#undef MAKE_ODD

        // _mm_load_si128((__m128i*)tab_dct_32_1[(tab)    ]) ->  *((__m128i*)tab_dct_32_1[(tab)    ])
#define MAKE_ODD(tab, dstPos) \
    T20 = _mm_madd_epi16(T00A, *((__m128i*)tab_dct_32_1[(tab)])); \
    T21 = _mm_madd_epi16(T00B, *((__m128i*)tab_dct_32_1[(tab)+1])); \
    T22 = _mm_madd_epi16(T01A, *((__m128i*)tab_dct_32_1[(tab)])); \
    T23 = _mm_madd_epi16(T01B, *((__m128i*)tab_dct_32_1[(tab)+1])); \
    T24 = _mm_madd_epi16(T02A, *((__m128i*)tab_dct_32_1[(tab)])); \
    T25 = _mm_madd_epi16(T02B, *((__m128i*)tab_dct_32_1[(tab)+1])); \
    T26 = _mm_madd_epi16(T03A, *((__m128i*)tab_dct_32_1[(tab)])); \
    T27 = _mm_madd_epi16(T03B, *((__m128i*)tab_dct_32_1[(tab)+1])); \
    T30 = _mm_madd_epi16(T04A, *((__m128i*)tab_dct_32_1[(tab)])); \
    T31 = _mm_madd_epi16(T04B, *((__m128i*)tab_dct_32_1[(tab)+1])); \
    T32 = _mm_madd_epi16(T05A, *((__m128i*)tab_dct_32_1[(tab)])); \
    T33 = _mm_madd_epi16(T05B, *((__m128i*)tab_dct_32_1[(tab)+1])); \
    T34 = _mm_madd_epi16(T06A, *((__m128i*)tab_dct_32_1[(tab)])); \
    T35 = _mm_madd_epi16(T06B, *((__m128i*)tab_dct_32_1[(tab)+1])); \
    T36 = _mm_madd_epi16(T07A, *((__m128i*)tab_dct_32_1[(tab)])); \
    T37 = _mm_madd_epi16(T07B, *((__m128i*)tab_dct_32_1[(tab)+1])); \
    \
    T40 = _mm_hadd_epi32(T20, T21); \
    T41 = _mm_hadd_epi32(T22, T23); \
    T42 = _mm_hadd_epi32(T24, T25); \
    T43 = _mm_hadd_epi32(T26, T27); \
    T44 = _mm_hadd_epi32(T30, T31); \
    T45 = _mm_hadd_epi32(T32, T33); \
    T46 = _mm_hadd_epi32(T34, T35); \
    T47 = _mm_hadd_epi32(T36, T37); \
    \
    T50 = _mm_hadd_epi32(T40, T41); \
    T51 = _mm_hadd_epi32(T42, T43); \
    T52 = _mm_hadd_epi32(T44, T45); \
    T53 = _mm_hadd_epi32(T46, T47); \
    \
    T50 = _mm_hadd_epi32(T50, T51); \
    T51 = _mm_hadd_epi32(T52, T53); \
    T50 = _mm_srai_epi32(_mm_add_epi32(T50, c_add1), shift1); \
    T51 = _mm_srai_epi32(_mm_add_epi32(T51, c_add1), shift1); \
    T60 = _mm_packs_epi32(T50, T51); \
    im[(dstPos)][i] = T60;

        MAKE_ODD(12, 1);
        MAKE_ODD(14, 3);
        MAKE_ODD(16, 5);
        MAKE_ODD(18, 7);
        MAKE_ODD(20, 9);
        MAKE_ODD(22, 11);
        MAKE_ODD(24, 13);
        MAKE_ODD(26, 15);
        MAKE_ODD(28, 17);
        MAKE_ODD(30, 19);
        MAKE_ODD(32, 21);
        MAKE_ODD(34, 23);
        MAKE_ODD(36, 25);
        MAKE_ODD(38, 27);
        MAKE_ODD(40, 29);
        MAKE_ODD(42, 31);

#undef MAKE_ODD
    }

    // DCT2

    __m128i R0C0, R0C1, R0C2, R0C3, R0C4, R0C5, R0C6;
    __m128i R1C0, R1C1, R1C2, R1C3, R1C4, R1C5, R1C6;
    __m128i R2C0, R2C1, R2C2, R2C3, R2C4, R2C5, R2C6;
    __m128i R3C0, R3C1, R3C2, R3C3, R3C4, R3C5, R3C6;
    __m128i R4C0, R4C1, R4C2, R4C3, R4C4, R4C5, R4C6;
    __m128i R5C0, R5C1, R5C2, R5C3, R5C4, R5C5, R5C6;
    __m128i R6C0, R6C1, R6C2, R6C3, R6C4, R6C5, R6C6;
    __m128i R7C0, R7C1, R7C2, R7C3, R7C4, R7C5, R7C6;

    __m128i R0C0_origin, R0C1_origin, R0C2_origin, R0C3_origin, R0C4_origin, R0C5_origin, R0C6_origin, R0C7_origin;
    __m128i R1C0_origin, R1C1_origin, R1C2_origin, R1C3_origin, R1C4_origin, R1C5_origin, R1C6_origin, R1C7_origin;
    __m128i R2C0_origin, R2C1_origin, R2C2_origin, R2C3_origin, R2C4_origin, R2C5_origin, R2C6_origin, R2C7_origin;
    __m128i R3C0_origin, R3C1_origin, R3C2_origin, R3C3_origin, R3C4_origin, R3C5_origin, R3C6_origin, R3C7_origin;
    __m128i R4C0_origin, R4C1_origin, R4C2_origin, R4C3_origin, R4C4_origin, R4C5_origin, R4C6_origin, R4C7_origin;
    __m128i R5C0_origin, R5C1_origin, R5C2_origin, R5C3_origin, R5C4_origin, R5C5_origin, R5C6_origin, R5C7_origin;
    __m128i R6C0_origin, R6C1_origin, R6C2_origin, R6C3_origin, R6C4_origin, R6C5_origin, R6C6_origin, R6C7_origin;
    __m128i R7C0_origin, R7C1_origin, R7C2_origin, R7C3_origin, R7C4_origin, R7C5_origin, R7C6_origin, R7C7_origin;

    __m128i COE0, COE1, COE2, COE3, COE4, COE5, COE6, COE7;
    __m128i COE_re_0, COE_re_1;
    __m128i COE_result;

    for (i = 0; i < 32 / 8; i++) {

        T00A = im[i * 8 + 0][0];    // [07 06 05 04 03 02 01 00]
        T00B = im[i * 8 + 0][1];    // [15 14 13 12 11 10 09 08]
        T00C = im[i * 8 + 0][2];    // [23 22 21 20 19 18 17 16]
        T00D = im[i * 8 + 0][3];    // [31 30 29 28 27 26 25 24]
        T01A = im[i * 8 + 1][0];
        T01B = im[i * 8 + 1][1];
        T01C = im[i * 8 + 1][2];
        T01D = im[i * 8 + 1][3];
        T02A = im[i * 8 + 2][0];
        T02B = im[i * 8 + 2][1];
        T02C = im[i * 8 + 2][2];
        T02D = im[i * 8 + 2][3];
        T03A = im[i * 8 + 3][0];
        T03B = im[i * 8 + 3][1];
        T03C = im[i * 8 + 3][2];
        T03D = im[i * 8 + 3][3];
        T04A = im[i * 8 + 4][0];
        T04B = im[i * 8 + 4][1];
        T04C = im[i * 8 + 4][2];
        T04D = im[i * 8 + 4][3];
        T05A = im[i * 8 + 5][0];
        T05B = im[i * 8 + 5][1];
        T05C = im[i * 8 + 5][2];
        T05D = im[i * 8 + 5][3];
        T06A = im[i * 8 + 6][0];
        T06B = im[i * 8 + 6][1];
        T06C = im[i * 8 + 6][2];
        T06D = im[i * 8 + 6][3];
        T07A = im[i * 8 + 7][0];
        T07B = im[i * 8 + 7][1];
        T07C = im[i * 8 + 7][2];
        T07D = im[i * 8 + 7][3];


        T00A = _mm_shuffle_epi8(T00A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));   // [05 02 06 01 04 03 07 00]
        T00B = _mm_shuffle_epi8(T00B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));   // [10 13 09 14 11 12 08 15]
        T00C = _mm_shuffle_epi8(T00C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));   // [21 18 22 17 20 19 23 16]
        T00D = _mm_shuffle_epi8(T00D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));   // [26 29 25 30 27 28 24 31]
        T01A = _mm_shuffle_epi8(T01A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T01B = _mm_shuffle_epi8(T01B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T01C = _mm_shuffle_epi8(T01C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T01D = _mm_shuffle_epi8(T01D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T02A = _mm_shuffle_epi8(T02A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T02B = _mm_shuffle_epi8(T02B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T02C = _mm_shuffle_epi8(T02C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T02D = _mm_shuffle_epi8(T02D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T03A = _mm_shuffle_epi8(T03A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T03B = _mm_shuffle_epi8(T03B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T03C = _mm_shuffle_epi8(T03C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T03D = _mm_shuffle_epi8(T03D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T04A = _mm_shuffle_epi8(T04A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T04B = _mm_shuffle_epi8(T04B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T04C = _mm_shuffle_epi8(T04C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T04D = _mm_shuffle_epi8(T04D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T05A = _mm_shuffle_epi8(T05A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T05B = _mm_shuffle_epi8(T05B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T05C = _mm_shuffle_epi8(T05C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T05D = _mm_shuffle_epi8(T05D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T06A = _mm_shuffle_epi8(T06A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T06B = _mm_shuffle_epi8(T06B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T06C = _mm_shuffle_epi8(T06C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T06D = _mm_shuffle_epi8(T06D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T07A = _mm_shuffle_epi8(T07A, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T07B = _mm_shuffle_epi8(T07B, _mm_load_si128((__m128i*)tab_dct_32_0[0]));
        T07C = _mm_shuffle_epi8(T07C, _mm_load_si128((__m128i*)tab_dct_16_0[1]));
        T07D = _mm_shuffle_epi8(T07D, _mm_load_si128((__m128i*)tab_dct_32_0[0]));


        // (i*8)+0
        R0C0_origin = _mm_cvtepi16_epi32(T00A);// 04 03 07 00
        T00A = _mm_srli_si128(T00A, 8);
        R0C1_origin = _mm_cvtepi16_epi32(T00A);// 05 02 06 01
        R0C2_origin = _mm_cvtepi16_epi32(T00B);// 11 12 08 15
        T00B = _mm_srli_si128(T00B, 8);
        R0C3_origin = _mm_cvtepi16_epi32(T00B);// 10 13 09 14
        R0C4_origin = _mm_cvtepi16_epi32(T00C);// 20 19 23 16
        T00C = _mm_srli_si128(T00C, 8);
        R0C5_origin = _mm_cvtepi16_epi32(T00C);// 21 18 22 17
        R0C6_origin = _mm_cvtepi16_epi32(T00D);// 27 28 24 31
        T00D = _mm_srli_si128(T00D, 8);
        R0C7_origin = _mm_cvtepi16_epi32(T00D);// 26 29 25 30
        //add 32bit
        R0C0 = _mm_add_epi32(R0C0_origin, R0C6_origin);// [04+27] [03+28] [07+24] [00+31]
        R0C1 = _mm_add_epi32(R0C1_origin, R0C7_origin);// [05+26] [02+29] [06+25] [01+30]
        R0C2 = _mm_add_epi32(R0C2_origin, R0C4_origin);// [11+20] [12+19] [08+23] [15+16]
        R0C3 = _mm_add_epi32(R0C3_origin, R0C5_origin);// [10+21] [13+18] [09+22] [14+17]

        R0C4 = _mm_add_epi32(R0C0, R0C2);// [04+27 + 11+20] [03+28 + 12+29] [07+24 + 08+23] [00+31 + 15+16]----4 3 7 0
        R0C5 = _mm_add_epi32(R0C1, R0C3);// [05+26 + 10+21] [02+29 + 13+18] [06+25 + 09+22] [01+30 + 14+17]----5 2 6 1
        R0C6 = _mm_hadd_epi32(R0C4, R0C5);// [] [] [] [] 2 1 3 0

        // (i*8)+1

        R1C0_origin = _mm_cvtepi16_epi32(T01A);// 04 03 07 00
        T01A = _mm_srli_si128(T01A, 8);
        R1C1_origin = _mm_cvtepi16_epi32(T01A);// 05 02 06 01
        R1C2_origin = _mm_cvtepi16_epi32(T01B);// 11 12 08 15
        T01B = _mm_srli_si128(T01B, 8);
        R1C3_origin = _mm_cvtepi16_epi32(T01B);// 10 13 09 14
        R1C4_origin = _mm_cvtepi16_epi32(T01C);// 20 19 23 16
        T01C = _mm_srli_si128(T01C, 8);
        R1C5_origin = _mm_cvtepi16_epi32(T01C);// 21 18 22 17
        R1C6_origin = _mm_cvtepi16_epi32(T01D);// 27 28 24 31
        T01D = _mm_srli_si128(T01D, 8);
        R1C7_origin = _mm_cvtepi16_epi32(T01D);// 26 29 25 30

        R1C0 = _mm_add_epi32(R1C0_origin, R1C6_origin);
        R1C1 = _mm_add_epi32(R1C1_origin, R1C7_origin);
        R1C2 = _mm_add_epi32(R1C2_origin, R1C4_origin);
        R1C3 = _mm_add_epi32(R1C3_origin, R1C5_origin);

        R1C4 = _mm_add_epi32(R1C0, R1C2);// [04+27 + 11+20] [03+28 + 12+29] [07+24 + 08+23] [00+31 + 15+16]----4 3 7 0
        R1C5 = _mm_add_epi32(R1C1, R1C3);// [05+26 + 10+21] [02+29 + 13+18] [06+25 + 09+22] [01+30 + 14+17]----5 2 6 1
        R1C6 = _mm_hadd_epi32(R1C4, R1C5);// [] [] [] [] 2 1 3 0

        // (i*8)+2

        R2C0_origin = _mm_cvtepi16_epi32(T02A);    // 04 03 07 00
        T02A = _mm_srli_si128(T02A, 8);
        R2C1_origin = _mm_cvtepi16_epi32(T02A);// 05 02 06 01
        R2C2_origin = _mm_cvtepi16_epi32(T02B);// 11 12 08 15
        T02B = _mm_srli_si128(T02B, 8);
        R2C3_origin = _mm_cvtepi16_epi32(T02B);// 10 13 09 14
        R2C4_origin = _mm_cvtepi16_epi32(T02C);// 20 19 23 16
        T02C = _mm_srli_si128(T02C, 8);
        R2C5_origin = _mm_cvtepi16_epi32(T02C);// 21 18 22 17
        R2C6_origin = _mm_cvtepi16_epi32(T02D);// 27 28 24 31
        T02D = _mm_srli_si128(T02D, 8);
        R2C7_origin = _mm_cvtepi16_epi32(T02D);// 26 29 25 30

        R2C0 = _mm_add_epi32(R2C0_origin, R2C6_origin);// [04+27] [03+28] [07+24] [00+31]
        R2C1 = _mm_add_epi32(R2C1_origin, R2C7_origin);// [05+26] [02+29] [06+25] [01+30]
        R2C2 = _mm_add_epi32(R2C2_origin, R2C4_origin);// [11+20] [12+19] [08+23] [15+16]
        R2C3 = _mm_add_epi32(R2C3_origin, R2C5_origin);// [10+21] [13+18] [09+22] [14+17]

        R2C4 = _mm_add_epi32(R2C0, R2C2);// [04+27 + 11+20] [03+28 + 12+29] [07+24 + 08+23] [00+31 + 15+16]----4 3 7 0
        R2C5 = _mm_add_epi32(R2C1, R2C3);// [05+26 + 10+21] [02+29 + 13+18] [06+25 + 09+22] [01+30 + 14+17]----5 2 6 1
        R2C6 = _mm_hadd_epi32(R2C4, R2C5);// [] [] [] [] 2 1 3 0

        // (i*8)+3
        R3C0_origin = _mm_cvtepi16_epi32(T03A);    // 04 03 07 00
        T03A = _mm_srli_si128(T03A, 8);
        R3C1_origin = _mm_cvtepi16_epi32(T03A);// 05 02 06 01
        R3C2_origin = _mm_cvtepi16_epi32(T03B);// 11 12 08 15
        T03B = _mm_srli_si128(T03B, 8);
        R3C3_origin = _mm_cvtepi16_epi32(T03B);// 10 13 09 14
        R3C4_origin = _mm_cvtepi16_epi32(T03C);// 20 19 23 16
        T03C = _mm_srli_si128(T03C, 8);
        R3C5_origin = _mm_cvtepi16_epi32(T03C);// 21 18 22 17
        R3C6_origin = _mm_cvtepi16_epi32(T03D);// 27 28 24 31
        T03D = _mm_srli_si128(T03D, 8);
        R3C7_origin = _mm_cvtepi16_epi32(T03D);// 26 29 25 30

        R3C0 = _mm_add_epi32(R3C0_origin, R3C6_origin);
        R3C1 = _mm_add_epi32(R3C1_origin, R3C7_origin);
        R3C2 = _mm_add_epi32(R3C2_origin, R3C4_origin);
        R3C3 = _mm_add_epi32(R3C3_origin, R3C5_origin);

        R3C4 = _mm_add_epi32(R3C0, R3C2);// [04+27 + 11+20] [03+28 + 12+29] [07+24 + 08+23] [00+31 + 15+16]----4 3 7 0
        R3C5 = _mm_add_epi32(R3C1, R3C3);// [05+26 + 10+21] [02+29 + 13+18] [06+25 + 09+22] [01+30 + 14+17]----5 2 6 1
        R3C6 = _mm_hadd_epi32(R3C4, R3C5);// [] [] [] [] 2 1 3 0

        // (i*8)+4
        R4C0_origin = _mm_cvtepi16_epi32(T04A);    // 04 03 07 00
        T04A = _mm_srli_si128(T04A, 8);
        R4C1_origin = _mm_cvtepi16_epi32(T04A);// 05 02 06 01
        R4C2_origin = _mm_cvtepi16_epi32(T04B);// 11 12 08 15
        T04B = _mm_srli_si128(T04B, 8);
        R4C3_origin = _mm_cvtepi16_epi32(T04B);// 10 13 09 14
        R4C4_origin = _mm_cvtepi16_epi32(T04C);// 20 19 23 16
        T04C = _mm_srli_si128(T04C, 8);
        R4C5_origin = _mm_cvtepi16_epi32(T04C);// 21 18 22 17
        R4C6_origin = _mm_cvtepi16_epi32(T04D);// 27 28 24 31
        T04D = _mm_srli_si128(T04D, 8);
        R4C7_origin = _mm_cvtepi16_epi32(T04D);// 26 29 25 30

        R4C0 = _mm_add_epi32(R4C0_origin, R4C6_origin);// [04+27] [03+28] [07+24] [00+31]
        R4C1 = _mm_add_epi32(R4C1_origin, R4C7_origin);// [05+26] [02+29] [06+25] [01+30]
        R4C2 = _mm_add_epi32(R4C2_origin, R4C4_origin);// [11+20] [12+19] [08+23] [15+16]
        R4C3 = _mm_add_epi32(R4C3_origin, R4C5_origin);// [10+21] [13+18] [09+22] [14+17]

        R4C4 = _mm_add_epi32(R4C0, R4C2);// [04+27 + 11+20] [03+28 + 12+29] [07+24 + 08+23] [00+31 + 15+16]----4 3 7 0
        R4C5 = _mm_add_epi32(R4C1, R4C3);// [05+26 + 10+21] [02+29 + 13+18] [06+25 + 09+22] [01+30 + 14+17]----5 2 6 1
        R4C6 = _mm_hadd_epi32(R4C4, R4C5);// [] [] [] [] 2 1 3 0

        // (i*8)+5
        R5C0_origin = _mm_cvtepi16_epi32(T05A);    // 04 03 07 00
        T05A = _mm_srli_si128(T05A, 8);
        R5C1_origin = _mm_cvtepi16_epi32(T05A);// 05 02 06 01
        R5C2_origin = _mm_cvtepi16_epi32(T05B);// 11 12 08 15
        T05B = _mm_srli_si128(T05B, 8);
        R5C3_origin = _mm_cvtepi16_epi32(T05B);// 10 13 09 14
        R5C4_origin = _mm_cvtepi16_epi32(T05C);// 20 19 23 16
        T05C = _mm_srli_si128(T05C, 8);
        R5C5_origin = _mm_cvtepi16_epi32(T05C);// 21 18 22 17
        R5C6_origin = _mm_cvtepi16_epi32(T05D);// 27 28 24 31
        T05D = _mm_srli_si128(T05D, 8);
        R5C7_origin = _mm_cvtepi16_epi32(T05D);// 26 29 25 30

        R5C0 = _mm_add_epi32(R5C0_origin, R5C6_origin);
        R5C1 = _mm_add_epi32(R5C1_origin, R5C7_origin);
        R5C2 = _mm_add_epi32(R5C2_origin, R5C4_origin);
        R5C3 = _mm_add_epi32(R5C3_origin, R5C5_origin);

        R5C4 = _mm_add_epi32(R5C0, R5C2);// [04+27 + 11+20] [03+28 + 12+29] [07+24 + 08+23] [00+31 + 15+16]----4 3 7 0
        R5C5 = _mm_add_epi32(R5C1, R5C3);// [05+26 + 10+21] [02+29 + 13+18] [06+25 + 09+22] [01+30 + 14+17]----5 2 6 1
        R5C6 = _mm_hadd_epi32(R5C4, R5C5);// [] [] [] [] 2 1 3 0

        // (i*8)+6
        R6C0_origin = _mm_cvtepi16_epi32(T06A);    // 04 03 07 00
        T06A = _mm_srli_si128(T06A, 8);
        R6C1_origin = _mm_cvtepi16_epi32(T06A);// 05 02 06 01
        R6C2_origin = _mm_cvtepi16_epi32(T06B);// 11 12 08 15
        T06B = _mm_srli_si128(T06B, 8);
        R6C3_origin = _mm_cvtepi16_epi32(T06B);// 10 13 09 14
        R6C4_origin = _mm_cvtepi16_epi32(T06C);// 20 19 23 16
        T06C = _mm_srli_si128(T06C, 8);
        R6C5_origin = _mm_cvtepi16_epi32(T06C);// 21 18 22 17
        R6C6_origin = _mm_cvtepi16_epi32(T06D);// 27 28 24 31
        T06D = _mm_srli_si128(T06D, 8);
        R6C7_origin = _mm_cvtepi16_epi32(T06D);// 26 29 25 30

        R6C0 = _mm_add_epi32(R6C0_origin, R6C6_origin);// [04+27] [03+28] [07+24] [00+31]
        R6C1 = _mm_add_epi32(R6C1_origin, R6C7_origin);// [05+26] [02+29] [06+25] [01+30]
        R6C2 = _mm_add_epi32(R6C2_origin, R6C4_origin);// [11+20] [12+19] [08+23] [15+16]
        R6C3 = _mm_add_epi32(R6C3_origin, R6C5_origin);// [10+21] [13+18] [09+22] [14+17]

        R6C4 = _mm_add_epi32(R6C0, R6C2);// [04+27 + 11+20] [03+28 + 12+29] [07+24 + 08+23] [00+31 + 15+16]----4 3 7 0
        R6C5 = _mm_add_epi32(R6C1, R6C3);// [05+26 + 10+21] [02+29 + 13+18] [06+25 + 09+22] [01+30 + 14+17]----5 2 6 1
        R6C6 = _mm_hadd_epi32(R6C4, R6C5);// [] [] [] [] 2 1 3 0

        // (i*8)+7
        R7C0_origin = _mm_cvtepi16_epi32(T07A);// 04 03 07 00
        T07A = _mm_srli_si128(T07A, 8);
        R7C1_origin = _mm_cvtepi16_epi32(T07A);// 05 02 06 01
        R7C2_origin = _mm_cvtepi16_epi32(T07B);// 11 12 08 15
        T07B = _mm_srli_si128(T07B, 8);
        R7C3_origin = _mm_cvtepi16_epi32(T07B);// 10 13 09 14
        R7C4_origin = _mm_cvtepi16_epi32(T07C);// 20 19 23 16
        T07C = _mm_srli_si128(T07C, 8);
        R7C5_origin = _mm_cvtepi16_epi32(T07C);// 21 18 22 17
        R7C6_origin = _mm_cvtepi16_epi32(T07D);// 27 28 24 31
        T07D = _mm_srli_si128(T07D, 8);
        R7C7_origin = _mm_cvtepi16_epi32(T07D);// 26 29 25 30

        R7C0 = _mm_add_epi32(R7C0_origin, R7C6_origin);
        R7C1 = _mm_add_epi32(R7C1_origin, R7C7_origin);
        R7C2 = _mm_add_epi32(R7C2_origin, R7C4_origin);
        R7C3 = _mm_add_epi32(R7C3_origin, R7C5_origin);

        R7C4 = _mm_add_epi32(R7C0, R7C2);// [04+27 + 11+20] [03+28 + 12+29] [07+24 + 08+23] [00+31 + 15+16]----4 3 7 0
        R7C5 = _mm_add_epi32(R7C1, R7C3);// [05+26 + 10+21] [02+29 + 13+18] [06+25 + 09+22] [01+30 + 14+17]----5 2 6 1
        R7C6 = _mm_hadd_epi32(R7C4, R7C5);// [] [] [] [] 2 1 3 0

        //coefficient

        //compute coefficient
        COE0 = _mm_mullo_epi32(R0C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[0]));
        COE1 = _mm_mullo_epi32(R1C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[0]));
        COE2 = _mm_mullo_epi32(R2C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[0]));
        COE3 = _mm_mullo_epi32(R3C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[0]));
        COE4 = _mm_mullo_epi32(R4C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[0]));
        COE5 = _mm_mullo_epi32(R5C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[0]));
        COE6 = _mm_mullo_epi32(R6C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[0]));
        COE7 = _mm_mullo_epi32(R7C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[0]));

        COE0 = _mm_hadd_epi32(COE0, COE1);//low 64 bit 1th row  ,high 64 bit 2th coefficient
        COE1 = _mm_hadd_epi32(COE2, COE3);
        COE2 = _mm_hadd_epi32(COE4, COE5);
        COE3 = _mm_hadd_epi32(COE6, COE7);

        COE_re_0 = _mm_hadd_epi32(COE0, COE1);//[127-96] 3th row  [95-64] 2th row  [63-32] 1th row  [31-0] 0 row
        COE_re_1 = _mm_hadd_epi32(COE2, COE3);
        COE_re_0 = _mm_srai_epi32(_mm_add_epi32(COE_re_0, c_add2), SHIFT2);
        COE_re_1 = _mm_srai_epi32(_mm_add_epi32(COE_re_1, c_add2), SHIFT2);
        COE_result = _mm_packs_epi32(COE_re_0, COE_re_1);

        _mm_store_si128((__m128i*)(dst + (i * 8) + 0), COE_result);

        COE_re_0 = _mm_hsub_epi32(COE0, COE1);
        COE_re_1 = _mm_hsub_epi32(COE2, COE3);
        COE_re_0 = _mm_srai_epi32(_mm_add_epi32(COE_re_0, c_add2), SHIFT2);
        COE_re_1 = _mm_srai_epi32(_mm_add_epi32(COE_re_1, c_add2), SHIFT2);
        COE_result = _mm_packs_epi32(COE_re_0, COE_re_1);

        _mm_store_si128((__m128i*)(dst + (16 * i_src) + (i * 8) + 0), COE_result);


        COE0 = _mm_mullo_epi32(R0C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[1]));
        COE1 = _mm_mullo_epi32(R1C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[1]));
        COE2 = _mm_mullo_epi32(R2C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[1]));
        COE3 = _mm_mullo_epi32(R3C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[1]));
        COE4 = _mm_mullo_epi32(R4C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[1]));
        COE5 = _mm_mullo_epi32(R5C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[1]));
        COE6 = _mm_mullo_epi32(R6C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[1]));
        COE7 = _mm_mullo_epi32(R7C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[1]));

        COE0 = _mm_hadd_epi32(COE0, COE1);//low 64 bit 1th row  ,high 64 bit 2th coefficient
        COE1 = _mm_hadd_epi32(COE2, COE3);
        COE2 = _mm_hadd_epi32(COE4, COE5);
        COE3 = _mm_hadd_epi32(COE6, COE7);

        COE_re_0 = _mm_hadd_epi32(COE0, COE1);//[127-96] 3th row  [95-64] 2th row  [63-32] 1th row  [31-0] 0 row
        COE_re_1 = _mm_hadd_epi32(COE2, COE3);
        COE_re_0 = _mm_srai_epi32(_mm_add_epi32(COE_re_0, c_add2), SHIFT2);
        COE_re_1 = _mm_srai_epi32(_mm_add_epi32(COE_re_1, c_add2), SHIFT2);
        COE_result = _mm_packs_epi32(COE_re_0, COE_re_1);

        _mm_store_si128((__m128i*)(dst + (8 * i_src) + (i * 8) + 0), COE_result);

        COE0 = _mm_mullo_epi32(R0C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[2]));
        COE1 = _mm_mullo_epi32(R1C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[2]));
        COE2 = _mm_mullo_epi32(R2C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[2]));
        COE3 = _mm_mullo_epi32(R3C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[2]));
        COE4 = _mm_mullo_epi32(R4C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[2]));
        COE5 = _mm_mullo_epi32(R5C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[2]));
        COE6 = _mm_mullo_epi32(R6C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[2]));
        COE7 = _mm_mullo_epi32(R7C6, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[2]));

        COE0 = _mm_hadd_epi32(COE0, COE1);//low 64 bit 1th row  ,high 64 bit 2th coefficient
        COE1 = _mm_hadd_epi32(COE2, COE3);
        COE2 = _mm_hadd_epi32(COE4, COE5);
        COE3 = _mm_hadd_epi32(COE6, COE7);

        COE_re_0 = _mm_hadd_epi32(COE0, COE1);//[127-96] 3th row  [95-64] 2th row  [63-32] 1th row  [31-0] 0 row
        COE_re_1 = _mm_hadd_epi32(COE2, COE3);
        COE_re_0 = _mm_srai_epi32(_mm_add_epi32(COE_re_0, c_add2), SHIFT2);
        COE_re_1 = _mm_srai_epi32(_mm_add_epi32(COE_re_1, c_add2), SHIFT2);
        COE_result = _mm_packs_epi32(COE_re_0, COE_re_1);

        _mm_store_si128((__m128i*)(dst + (24 * i_src) + (i * 8) + 0), COE_result);




#define MAKE_ODD(tab,dstPos) \
    COE0 = _mm_add_epi32(_mm_mullo_epi32(R0C4, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(R0C5, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))); \
    COE1 = _mm_add_epi32(_mm_mullo_epi32(R1C4, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(R1C5, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))); \
    COE2 = _mm_add_epi32(_mm_mullo_epi32(R2C4, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(R2C5, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))); \
    COE3 = _mm_add_epi32(_mm_mullo_epi32(R3C4, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(R3C5, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))); \
    COE4 = _mm_add_epi32(_mm_mullo_epi32(R4C4, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(R4C5, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))); \
    COE5 = _mm_add_epi32(_mm_mullo_epi32(R5C4, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(R5C5, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))); \
    COE6 = _mm_add_epi32(_mm_mullo_epi32(R6C4, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(R6C5, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))); \
    COE7 = _mm_add_epi32(_mm_mullo_epi32(R7C4, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(R7C5, _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))); \
    \
    COE0 = _mm_hadd_epi32(COE0, COE1); \
    COE1 = _mm_hadd_epi32(COE2, COE3); \
    COE2 = _mm_hadd_epi32(COE4, COE5); \
    COE3 = _mm_hadd_epi32(COE6, COE7); \
    \
    COE_re_0 = _mm_hadd_epi32(COE0, COE1); \
    COE_re_1 = _mm_hadd_epi32(COE2, COE3); \
    COE_re_0 = _mm_srai_epi32(_mm_add_epi32(COE_re_0, c_add2), SHIFT2); \
    COE_re_1 = _mm_srai_epi32(_mm_add_epi32(COE_re_1, c_add2), SHIFT2); \
    COE_result = _mm_packs_epi32(COE_re_0, COE_re_1); \
    _mm_store_si128((__m128i*)(dst + (dstPos * i_src) + (i * 8) + 0), COE_result);


        MAKE_ODD(3, 4);
        MAKE_ODD(5, 12);
        MAKE_ODD(7, 20);
        MAKE_ODD(9, 28);


#undef MAKE_ODD

#define MAKE_ODD(tab,dstPos) \
    COE0 = _mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R0C0, R0C2), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(_mm_sub_epi32(R0C1, R0C3), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))); \
    COE1 = _mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R1C0, R1C2), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(_mm_sub_epi32(R1C1, R1C3), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))); \
    COE2 = _mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R2C0, R2C2), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(_mm_sub_epi32(R2C1, R2C3), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))); \
    COE3 = _mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R3C0, R3C2), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(_mm_sub_epi32(R3C1, R3C3), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))); \
    COE4 = _mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R4C0, R4C2), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(_mm_sub_epi32(R4C1, R4C3), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))); \
    COE5 = _mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R5C0, R5C2), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(_mm_sub_epi32(R5C1, R5C3), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))); \
    COE6 = _mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R6C0, R6C2), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(_mm_sub_epi32(R6C1, R6C3), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))); \
    COE7 = _mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R7C0, R7C2), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(_mm_sub_epi32(R7C1, R7C3), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))); \
    \
    COE0 = _mm_hadd_epi32(COE0, COE1); \
    COE1 = _mm_hadd_epi32(COE2, COE3); \
    COE2 = _mm_hadd_epi32(COE4, COE5); \
    COE3 = _mm_hadd_epi32(COE6, COE7); \
    \
    COE_re_0 = _mm_hadd_epi32(COE0, COE1); \
    COE_re_1 = _mm_hadd_epi32(COE2, COE3); \
    COE_re_0 = _mm_srai_epi32(_mm_add_epi32(COE_re_0, c_add2), SHIFT2); \
    COE_re_1 = _mm_srai_epi32(_mm_add_epi32(COE_re_1, c_add2), SHIFT2); \
    COE_result = _mm_packs_epi32(COE_re_0, COE_re_1); \
    _mm_store_si128((__m128i*)(dst + (dstPos * i_src) + (i * 8) + 0), COE_result);


        MAKE_ODD(11, 2);
        MAKE_ODD(13, 6);
        MAKE_ODD(15, 10);
        MAKE_ODD(17, 14);
        MAKE_ODD(19, 18);
        MAKE_ODD(21, 22);
        MAKE_ODD(23, 26);
        MAKE_ODD(25, 30);


#undef MAKE_ODD


        /*COE0 = _mm_add_epi32(
        _mm_add_epi32(
        _mm_mullo_epi32(_mm_sub_epi32(R0C0_origin, R0C6_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[1  ]) )
        ,
        _mm_mullo_epi32(_mm_sub_epi32(R0C1_origin, R0C7_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[1]  ) )
        )
        ,
        _mm_add_epi32(
        _mm_mullo_epi32(_mm_sub_epi32(R0C2_origin, R0C4_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[1]  ) )
        ,
        _mm_mullo_epi32(_mm_sub_epi32(R0C3_origin, R0C5_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[1] )   )
        )
        ); */

        //compute 1 3 5 7 9 11 ....29 31
        //dct coefficient matrix is symmetry .So according to this property . we can compute [0-31] [7-24]....
        //then add corresponding bit .we can get the result.-------zhangjiaqi  2016-12-10

#define MAKE_ODD(tab,dstPos) \
    COE0 = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R0C0_origin, R0C6_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(_mm_sub_epi32(R0C1_origin, R0C7_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))), _mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R0C2_origin, R0C4_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 2])), _mm_mullo_epi32(_mm_sub_epi32(R0C3_origin, R0C5_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 3])))); \
    COE1 = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R1C0_origin, R1C6_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(_mm_sub_epi32(R1C1_origin, R1C7_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))), _mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R1C2_origin, R1C4_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 2])), _mm_mullo_epi32(_mm_sub_epi32(R1C3_origin, R1C5_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 3])))); \
    COE2 = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R2C0_origin, R2C6_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(_mm_sub_epi32(R2C1_origin, R2C7_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))), _mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R2C2_origin, R2C4_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 2])), _mm_mullo_epi32(_mm_sub_epi32(R2C3_origin, R2C5_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 3])))); \
    COE3 = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R3C0_origin, R3C6_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(_mm_sub_epi32(R3C1_origin, R3C7_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))), _mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R3C2_origin, R3C4_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 2])), _mm_mullo_epi32(_mm_sub_epi32(R3C3_origin, R3C5_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 3])))); \
    COE4 = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R4C0_origin, R4C6_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(_mm_sub_epi32(R4C1_origin, R4C7_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))), _mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R4C2_origin, R4C4_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 2])), _mm_mullo_epi32(_mm_sub_epi32(R4C3_origin, R4C5_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 3])))); \
    COE5 = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R5C0_origin, R5C6_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(_mm_sub_epi32(R5C1_origin, R5C7_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))), _mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R5C2_origin, R5C4_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 2])), _mm_mullo_epi32(_mm_sub_epi32(R5C3_origin, R5C5_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 3])))); \
    COE6 = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R6C0_origin, R6C6_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(_mm_sub_epi32(R6C1_origin, R6C7_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))), _mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R6C2_origin, R6C4_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 2])), _mm_mullo_epi32(_mm_sub_epi32(R6C3_origin, R6C5_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 3])))); \
    COE7 = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R7C0_origin, R7C6_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab])), _mm_mullo_epi32(_mm_sub_epi32(R7C1_origin, R7C7_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 1]))), _mm_add_epi32(_mm_mullo_epi32(_mm_sub_epi32(R7C2_origin, R7C4_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 2])), _mm_mullo_epi32(_mm_sub_epi32(R7C3_origin, R7C5_origin), _mm_load_si128((__m128i*)tab_dct_32_zhangjiaqi[tab + 3])))); \
    \
    COE0 = _mm_hadd_epi32(COE0, COE1); \
    COE1 = _mm_hadd_epi32(COE2, COE3); \
    COE2 = _mm_hadd_epi32(COE4, COE5); \
    COE3 = _mm_hadd_epi32(COE6, COE7); \
    \
    COE_re_0 = _mm_hadd_epi32(COE0, COE1); \
    COE_re_1 = _mm_hadd_epi32(COE2, COE3); \
    COE_re_0 = _mm_srai_epi32(_mm_add_epi32(COE_re_0, c_add2), SHIFT2); \
    COE_re_1 = _mm_srai_epi32(_mm_add_epi32(COE_re_1, c_add2), SHIFT2); \
    COE_result = _mm_packs_epi32(COE_re_0, COE_re_1); \
    \
    _mm_store_si128((__m128i*)(dst + (dstPos * i_src) + (i * 8) + 0), COE_result);

        MAKE_ODD(27, 1);
        MAKE_ODD(31, 3);
        MAKE_ODD(35, 5);
        MAKE_ODD(39, 7);
        MAKE_ODD(43, 9);
        MAKE_ODD(47, 11);
        MAKE_ODD(51, 13);
        MAKE_ODD(55, 15);
        MAKE_ODD(59, 17);
        MAKE_ODD(63, 19);
        MAKE_ODD(67, 21);
        MAKE_ODD(71, 23);
        MAKE_ODD(75, 25);
        MAKE_ODD(79, 27);
        MAKE_ODD(83, 29);
        MAKE_ODD(87, 31);


#undef MAKE_ODD

    }
#undef SHIFT1
#undef ADD1
#undef SHIFT2
#undef ADD2


}



/* ---------------------------------------------------------------------------
 */
static void trans_2nd_hor_sse128(coeff_t *coeff, int i_coeff)
{
#define SHIFT   7
#define ADD     64
    // const
    __m128i c_add = _mm_set1_epi32(ADD);

    // load 4x4 coeffs
    __m128i T10 = _mm_loadl_epi64((__m128i*)(coeff + 0 * i_coeff));   // [0 0 0 0 a3 a2 a1 a0]
    __m128i T11 = _mm_loadl_epi64((__m128i*)(coeff + 1 * i_coeff));   // [0 0 0 0 b3 b2 b1 b0]
    __m128i T12 = _mm_loadl_epi64((__m128i*)(coeff + 2 * i_coeff));   // [0 0 0 0 c3 c2 c1 c0]
    __m128i T13 = _mm_loadl_epi64((__m128i*)(coeff + 3 * i_coeff));   // [0 0 0 0 d3 d2 d1 d0]

    __m128i T20 = _mm_shuffle_epi32(T10, 0x00); // [a1 a0 a1 a0 a1 a0 a1 a0]
    __m128i T21 = _mm_shuffle_epi32(T10, 0x55); // [a3 a2 a3 a2 a3 a2 a3 a2]
    __m128i T22 = _mm_shuffle_epi32(T11, 0x00);
    __m128i T23 = _mm_shuffle_epi32(T11, 0x55);
    __m128i T24 = _mm_shuffle_epi32(T12, 0x00);
    __m128i T25 = _mm_shuffle_epi32(T12, 0x55);
    __m128i T26 = _mm_shuffle_epi32(T13, 0x00);
    __m128i T27 = _mm_shuffle_epi32(T13, 0x55);

    // load g_2T_H transform matrix
    __m128i C10 = _mm_load_si128((__m128i*)(g_2T_H + 0 * 2 * SEC_TR_SIZE)); // [h1 h0 g1 g0 f1 f0 e1 e0]
    __m128i C11 = _mm_load_si128((__m128i*)(g_2T_H + 1 * 2 * SEC_TR_SIZE)); // [h3 h2 g3 g2 f3 f2 e3 e2]

    // transform
    __m128i T30 = _mm_madd_epi16(T20, C10);     // [a0*h0+a1*h1, a0*g0+a1*g1, a0*f0+a1*f1, a0*e0+a1*e1]
    __m128i T31 = _mm_madd_epi16(T21, C11);     // [a2*h2+a3*h3, a2*g2+a3*g3, a2*f2+a3*f3, a2*e2+a3*e3]
    __m128i T32 = _mm_madd_epi16(T22, C10);
    __m128i T33 = _mm_madd_epi16(T23, C11);
    __m128i T34 = _mm_madd_epi16(T24, C10);
    __m128i T35 = _mm_madd_epi16(T25, C11);
    __m128i T36 = _mm_madd_epi16(T26, C10);
    __m128i T37 = _mm_madd_epi16(T27, C11);

    __m128i T40 = _mm_add_epi32(T30, T31);
    __m128i T41 = _mm_add_epi32(T32, T33);
    __m128i T42 = _mm_add_epi32(T34, T35);
    __m128i T43 = _mm_add_epi32(T36, T37);

    T40 = _mm_srai_epi32(_mm_add_epi32(T40, c_add), SHIFT);
    T41 = _mm_srai_epi32(_mm_add_epi32(T41, c_add), SHIFT);
    T42 = _mm_srai_epi32(_mm_add_epi32(T42, c_add), SHIFT);
    T43 = _mm_srai_epi32(_mm_add_epi32(T43, c_add), SHIFT);

    // store
    T40 = _mm_packs_epi32(T40, T40);
    T41 = _mm_packs_epi32(T41, T41);
    T42 = _mm_packs_epi32(T42, T42);
    T43 = _mm_packs_epi32(T43, T43);

    _mm_storel_epi64((__m128i*)(coeff + 0 * i_coeff), T40);
    _mm_storel_epi64((__m128i*)(coeff + 1 * i_coeff), T41);
    _mm_storel_epi64((__m128i*)(coeff + 2 * i_coeff), T42);
    _mm_storel_epi64((__m128i*)(coeff + 3 * i_coeff), T43);
#undef SHIFT
#undef ADD
}

/* ---------------------------------------------------------------------------
 */
static void trans_2nd_ver_sse128(coeff_t *coeff, int i_coeff)
{
#define SHIFT   7
#define ADD     64
    // const
    __m128i c_add = _mm_set1_epi32(ADD);

    // load 4x4 coeffs
    __m128i T10 = _mm_loadl_epi64((__m128i*)(coeff + 0 * i_coeff));   // [0 0 0 0 a3 a2 a1 a0]
    __m128i T11 = _mm_loadl_epi64((__m128i*)(coeff + 1 * i_coeff));   // [0 0 0 0 b3 b2 b1 b0]
    __m128i T12 = _mm_loadl_epi64((__m128i*)(coeff + 2 * i_coeff));   // [0 0 0 0 c3 c2 c1 c0]
    __m128i T13 = _mm_loadl_epi64((__m128i*)(coeff + 3 * i_coeff));   // [0 0 0 0 d3 d2 d1 d0]

    __m128i T20 = _mm_unpacklo_epi16(T10, T11);     // [b3 a3 b2 a2 b1 a1 b0 a0]
    __m128i T21 = _mm_unpacklo_epi16(T12, T13);     // [d3 c3 d2 c2 d1 c1 d0 c0]

    // load g_2T_V transform matrix
    __m128i C10 = _mm_load_si128((__m128i*)(g_2T_V + 0 * 2 * SEC_TR_SIZE)); // [e1 e0 e1 e0 e1 e0 e1 e0]
    __m128i C11 = _mm_load_si128((__m128i*)(g_2T_V + 1 * 2 * SEC_TR_SIZE)); // [e3 e2 e3 e2 e3 e2 e3 e2]
    __m128i C12 = _mm_load_si128((__m128i*)(g_2T_V + 2 * 2 * SEC_TR_SIZE)); // [f1 f0 f1 f0 f1 f0 f1 f0]
    __m128i C13 = _mm_load_si128((__m128i*)(g_2T_V + 3 * 2 * SEC_TR_SIZE)); // [f3 f2 f3 f2 f3 f2 f3 f2]
    __m128i C14 = _mm_load_si128((__m128i*)(g_2T_V + 4 * 2 * SEC_TR_SIZE)); // [g1 g0 g1 g0 g1 g0 g1 g0]
    __m128i C15 = _mm_load_si128((__m128i*)(g_2T_V + 5 * 2 * SEC_TR_SIZE)); // [g3 g2 g3 g2 g3 g2 g3 g2]
    __m128i C16 = _mm_load_si128((__m128i*)(g_2T_V + 6 * 2 * SEC_TR_SIZE)); // [h1 h0 h1 h0 h1 h0 h1 h0]
    __m128i C17 = _mm_load_si128((__m128i*)(g_2T_V + 7 * 2 * SEC_TR_SIZE)); // [h3 h2 h3 h2 h3 h2 h3 h2]

    // transform
    __m128i T30 = _mm_madd_epi16(T20, C10);     // [a3*e0+b3*e1, a2*e0+b2*e1, a1*e0+b1*e1, a0*e0+b0*e1]
    __m128i T31 = _mm_madd_epi16(T21, C11);     // [c3*e2+d3*e3, c2*e2+d2*e3, c1*e2+d1*e3, c0*e2+d0*e3]
    __m128i T32 = _mm_madd_epi16(T20, C12);
    __m128i T33 = _mm_madd_epi16(T21, C13);
    __m128i T34 = _mm_madd_epi16(T20, C14);
    __m128i T35 = _mm_madd_epi16(T21, C15);
    __m128i T36 = _mm_madd_epi16(T20, C16);
    __m128i T37 = _mm_madd_epi16(T21, C17);

    __m128i T40 = _mm_add_epi32(T30, T31);
    __m128i T41 = _mm_add_epi32(T32, T33);
    __m128i T42 = _mm_add_epi32(T34, T35);
    __m128i T43 = _mm_add_epi32(T36, T37);

    T40 = _mm_srai_epi32(_mm_add_epi32(T40, c_add), SHIFT);
    T41 = _mm_srai_epi32(_mm_add_epi32(T41, c_add), SHIFT);
    T42 = _mm_srai_epi32(_mm_add_epi32(T42, c_add), SHIFT);
    T43 = _mm_srai_epi32(_mm_add_epi32(T43, c_add), SHIFT);

    // store
    T40 = _mm_packs_epi32(T40, T40);
    T41 = _mm_packs_epi32(T41, T41);
    T42 = _mm_packs_epi32(T42, T42);
    T43 = _mm_packs_epi32(T43, T43);

    _mm_storel_epi64((__m128i*)(coeff + 0 * i_coeff), T40);
    _mm_storel_epi64((__m128i*)(coeff + 1 * i_coeff), T41);
    _mm_storel_epi64((__m128i*)(coeff + 2 * i_coeff), T42);
    _mm_storel_epi64((__m128i*)(coeff + 3 * i_coeff), T43);
#undef SHIFT
#undef ADD
}

/* ---------------------------------------------------------------------------
 * i_mode - real intra mode (luma)
 * b_top  - block top available?
 * b_left - block left available?
 */
void transform_2nd_sse128(coeff_t *coeff, int i_coeff, int i_mode, int b_top, int b_left)
{
    int vt = (i_mode >=  0 && i_mode <= 23);
    int ht = (i_mode >= 13 && i_mode <= 32) || (i_mode >= 0 && i_mode <= 2);

    if (vt && b_top) {
        trans_2nd_ver_sse128(coeff, i_coeff);
    }
    if (ht && b_left) {
        trans_2nd_hor_sse128(coeff, i_coeff);
    }
}

/* ---------------------------------------------------------------------------
 */
void transform_4x4_2nd_sse128(coeff_t *coeff, int i_coeff)
{
    const int SHIFT1 = B4X4_IN_BIT + FACTO_BIT + g_bit_depth + 1 - LIMIT_BIT + 1;
    const int SHIFT2 = B4X4_IN_BIT + FACTO_BIT + 1;
    const int ADD1 = 1 << (SHIFT1 - 1);
    const int ADD2 = 1 << (SHIFT2 - 1);
    __m128i C12, C13, C14, C15, C16, C17;

    // const
    __m128i c_add1 = _mm_set1_epi32(ADD1);
    __m128i c_add2 = _mm_set1_epi32(ADD2);

    // hor ---------------------------------------------------------

    // load 4x4 coeffs
    __m128i T10 = _mm_loadl_epi64((__m128i*)(coeff + 0 * i_coeff));   // [0 0 0 0 a3 a2 a1 a0]
    __m128i T11 = _mm_loadl_epi64((__m128i*)(coeff + 1 * i_coeff));   // [0 0 0 0 b3 b2 b1 b0]
    __m128i T12 = _mm_loadl_epi64((__m128i*)(coeff + 2 * i_coeff));   // [0 0 0 0 c3 c2 c1 c0]
    __m128i T13 = _mm_loadl_epi64((__m128i*)(coeff + 3 * i_coeff));   // [0 0 0 0 d3 d2 d1 d0]

    __m128i T20 = _mm_shuffle_epi32(T10, 0x00); // [a1 a0 a1 a0 a1 a0 a1 a0]
    __m128i T21 = _mm_shuffle_epi32(T10, 0x55); // [a3 a2 a3 a2 a3 a2 a3 a2]
    __m128i T22 = _mm_shuffle_epi32(T11, 0x00);
    __m128i T23 = _mm_shuffle_epi32(T11, 0x55);
    __m128i T24 = _mm_shuffle_epi32(T12, 0x00);
    __m128i T25 = _mm_shuffle_epi32(T12, 0x55);
    __m128i T26 = _mm_shuffle_epi32(T13, 0x00);
    __m128i T27 = _mm_shuffle_epi32(T13, 0x55);

    // load g_2TC_H transform matrix
    __m128i C10 = _mm_load_si128((__m128i*)(g_2TC_H + 0 * 2 * SEC_TR_SIZE));    // [h1 h0 g1 g0 f1 f0 e1 e0]
    __m128i C11 = _mm_load_si128((__m128i*)(g_2TC_H + 1 * 2 * SEC_TR_SIZE));    // [h3 h2 g3 g2 f3 f2 e3 e2]

    // transform
    __m128i T30 = _mm_madd_epi16(T20, C10);     // [a0*h0+a1*h1, a0*g0+a1*g1, a0*f0+a1*f1, a0*e0+a1*e1]
    __m128i T31 = _mm_madd_epi16(T21, C11);     // [a2*h2+a3*h3, a2*g2+a3*g3, a2*f2+a3*f3, a2*e2+a3*e3]
    __m128i T32 = _mm_madd_epi16(T22, C10);
    __m128i T33 = _mm_madd_epi16(T23, C11);
    __m128i T34 = _mm_madd_epi16(T24, C10);
    __m128i T35 = _mm_madd_epi16(T25, C11);
    __m128i T36 = _mm_madd_epi16(T26, C10);
    __m128i T37 = _mm_madd_epi16(T27, C11);

    __m128i T40 = _mm_add_epi32(T30, T31);
    __m128i T41 = _mm_add_epi32(T32, T33);
    __m128i T42 = _mm_add_epi32(T34, T35);
    __m128i T43 = _mm_add_epi32(T36, T37);

    T40 = _mm_srai_epi32(_mm_add_epi32(T40, c_add1), SHIFT1);
    T41 = _mm_srai_epi32(_mm_add_epi32(T41, c_add1), SHIFT1);
    T42 = _mm_srai_epi32(_mm_add_epi32(T42, c_add1), SHIFT1);
    T43 = _mm_srai_epi32(_mm_add_epi32(T43, c_add1), SHIFT1);

    // result of hor transform
    T40 = _mm_packs_epi32(T40, T40);        // [? ? ? ? a3 a2 a1 a0]
    T41 = _mm_packs_epi32(T41, T41);        // [? ? ? ? b3 b2 b1 b0]
    T42 = _mm_packs_epi32(T42, T42);        // [? ? ? ? c3 c2 c1 c0]
    T43 = _mm_packs_epi32(T43, T43);        // [? ? ? ? d3 d2 d1 d0]

    // ver ---------------------------------------------------------

    T20 = _mm_unpacklo_epi16(T40, T41);     // [b3 a3 b2 a2 b1 a1 b0 a0]
    T21 = _mm_unpacklo_epi16(T42, T43);     // [d3 c3 d2 c2 d1 c1 d0 c0]

    // load g_2TC_V transform matrix
    C10 = _mm_load_si128((__m128i*)(g_2TC_V + 0 * 2 * SEC_TR_SIZE));    // [e1 e0 e1 e0 e1 e0 e1 e0]
    C11 = _mm_load_si128((__m128i*)(g_2TC_V + 1 * 2 * SEC_TR_SIZE));    // [e3 e2 e3 e2 e3 e2 e3 e2]
    C12 = _mm_load_si128((__m128i*)(g_2TC_V + 2 * 2 * SEC_TR_SIZE));    // [f1 f0 f1 f0 f1 f0 f1 f0]
    C13 = _mm_load_si128((__m128i*)(g_2TC_V + 3 * 2 * SEC_TR_SIZE));    // [f3 f2 f3 f2 f3 f2 f3 f2]
    C14 = _mm_load_si128((__m128i*)(g_2TC_V + 4 * 2 * SEC_TR_SIZE));    // [g1 g0 g1 g0 g1 g0 g1 g0]
    C15 = _mm_load_si128((__m128i*)(g_2TC_V + 5 * 2 * SEC_TR_SIZE));    // [g3 g2 g3 g2 g3 g2 g3 g2]
    C16 = _mm_load_si128((__m128i*)(g_2TC_V + 6 * 2 * SEC_TR_SIZE));    // [h1 h0 h1 h0 h1 h0 h1 h0]
    C17 = _mm_load_si128((__m128i*)(g_2TC_V + 7 * 2 * SEC_TR_SIZE));    // [h3 h2 h3 h2 h3 h2 h3 h2]

    // transform
    T30 = _mm_madd_epi16(T20, C10);         // [a3*e0+b3*e1, a2*e0+b2*e1, a1*e0+b1*e1, a0*e0+b0*e1]
    T31 = _mm_madd_epi16(T21, C11);         // [c3*e2+d3*e3, c2*e2+d2*e3, c1*e2+d1*e3, c0*e2+d0*e3]
    T32 = _mm_madd_epi16(T20, C12);
    T33 = _mm_madd_epi16(T21, C13);
    T34 = _mm_madd_epi16(T20, C14);
    T35 = _mm_madd_epi16(T21, C15);
    T36 = _mm_madd_epi16(T20, C16);
    T37 = _mm_madd_epi16(T21, C17);

    T40 = _mm_add_epi32(T30, T31);
    T41 = _mm_add_epi32(T32, T33);
    T42 = _mm_add_epi32(T34, T35);
    T43 = _mm_add_epi32(T36, T37);

    T40 = _mm_srai_epi32(_mm_add_epi32(T40, c_add2), SHIFT2);
    T41 = _mm_srai_epi32(_mm_add_epi32(T41, c_add2), SHIFT2);
    T42 = _mm_srai_epi32(_mm_add_epi32(T42, c_add2), SHIFT2);
    T43 = _mm_srai_epi32(_mm_add_epi32(T43, c_add2), SHIFT2);

    // store
    T40 = _mm_packs_epi32(T40, T40);
    T41 = _mm_packs_epi32(T41, T41);
    T42 = _mm_packs_epi32(T42, T42);
    T43 = _mm_packs_epi32(T43, T43);

    _mm_storel_epi64((__m128i*)(coeff + 0 * i_coeff), T40);
    _mm_storel_epi64((__m128i*)(coeff + 1 * i_coeff), T41);
    _mm_storel_epi64((__m128i*)(coeff + 2 * i_coeff), T42);
    _mm_storel_epi64((__m128i*)(coeff + 3 * i_coeff), T43);
}



// transpose 8x8 & transpose 16x16(¾ØÕó×ªÖÃ)
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
 
void wavelet_16x64_sse128(coeff_t *coeff)
{
    //ï¿½ï¿½ï¿½ï¿½ 16*64
    __m128i V00[2], V01[2], V02[2], V03[2], V04[2], V05[2], V06[2], V07[2], V08[2], V09[2], V10[2], V11[2], V12[2], V13[2], V14[2], V15[2], V16[2], V17[2], V18[2], V19[2], V20[2], V21[2], V22[2], V23[2], V24[2], V25[2], V26[2], V27[2], V28[2], V29[2], V30[2], V31[2], V32[2], V33[2], V34[2], V35[2], V36[2], V37[2], V38[2], V39[2], V40[2], V41[2], V42[2], V43[2], V44[2], V45[2], V46[2], V47[2], V48[2], V49[2], V50[2], V51[2], V52[2], V53[2], V54[2], V55[2], V56[2], V57[2], V58[2], V59[2], V60[2], V61[2], V62[2], V63[2];

    //ï¿½ï¿½ï¿½ï¿½ 64*16
    __m128i T00[8], T01[8], T02[8], T03[8], T04[8], T05[8], T06[8], T07[8], T08[8], T09[8], T10[8], T11[8], T12[8], T13[8], T14[8], T15[8];

    //ï¿½ï¿½Ê±
    __m128i B00, B01, B02, B03, B04, B05, B06, B07, B08, B09, B10, B11, B12, B13, B14, B15, B16, B17, B18, B19, B20, B21, B22, B23, B24, B25, B26, B27, B28, B29, B30, B31;
    __m128i B32, B33, B34, B35, B36, B37, B38, B39, B40, B41, B42, B43, B44, B45, B46, B47, B48, B49, B50, B51, B52, B53, B54, B55, B56, B57, B58, B59, B60, B61, B62, B63;

    __m128i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
    __m128i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;
    int i;
    __m128i mAddOffset1 = _mm_set1_epi16(1);
    __m128i mAddOffset2 = _mm_set1_epi16(2);

    //load

    for (i = 0; i < 2; i++) {
        V00[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 0]);
        V01[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 1]);
        V02[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 2]);
        V03[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 3]);
        V04[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 4]);
        V05[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 5]);
        V06[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 6]);
        V07[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 7]);
        V08[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 8]);
        V09[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 9]);
        V10[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 10]);
        V11[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 11]);
        V12[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 12]);
        V13[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 13]);
        V14[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 14]);
        V15[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 15]);
        V16[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 16]);
        V17[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 17]);
        V18[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 18]);
        V19[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 19]);
        V20[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 20]);
        V21[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 21]);
        V22[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 22]);
        V23[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 23]);
        V24[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 24]);
        V25[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 25]);
        V26[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 26]);
        V27[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 27]);
        V28[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 28]);
        V29[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 29]);
        V30[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 30]);
        V31[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 31]);
        V32[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 32]);
        V33[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 33]);
        V34[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 34]);
        V35[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 35]);
        V36[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 36]);
        V37[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 37]);
        V38[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 38]);
        V39[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 39]);
        V40[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 40]);
        V41[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 41]);
        V42[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 42]);
        V43[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 43]);
        V44[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 44]);
        V45[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 45]);
        V46[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 46]);
        V47[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 47]);
        V48[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 48]);
        V49[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 49]);
        V50[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 50]);
        V51[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 51]);
        V52[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 52]);
        V53[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 53]);
        V54[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 54]);
        V55[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 55]);
        V56[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 56]);
        V57[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 57]);
        V58[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 58]);
        V59[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 59]);
        V60[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 60]);
        V61[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 61]);
        V62[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 62]);
        V63[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 16 * 63]);
    }

    TRANSPOSE_8x8_16BIT(V00[0], V01[0], V02[0], V03[0], V04[0], V05[0], V06[0], V07[0], T00[0], T01[0], T02[0], T03[0], T04[0], T05[0], T06[0], T07[0]);
    TRANSPOSE_8x8_16BIT(V08[0], V09[0], V10[0], V11[0], V12[0], V13[0], V14[0], V15[0], T00[1], T01[1], T02[1], T03[1], T04[1], T05[1], T06[1], T07[1]);
    TRANSPOSE_8x8_16BIT(V16[0], V17[0], V18[0], V19[0], V20[0], V21[0], V22[0], V23[0], T00[2], T01[2], T02[2], T03[2], T04[2], T05[2], T06[2], T07[2]);
    TRANSPOSE_8x8_16BIT(V24[0], V25[0], V26[0], V27[0], V28[0], V29[0], V30[0], V31[0], T00[3], T01[3], T02[3], T03[3], T04[3], T05[3], T06[3], T07[3]);
    TRANSPOSE_8x8_16BIT(V32[0], V33[0], V34[0], V35[0], V36[0], V37[0], V38[0], V39[0], T00[4], T01[4], T02[4], T03[4], T04[4], T05[4], T06[4], T07[4]);
    TRANSPOSE_8x8_16BIT(V40[0], V41[0], V42[0], V43[0], V44[0], V45[0], V46[0], V47[0], T00[5], T01[5], T02[5], T03[5], T04[5], T05[5], T06[5], T07[5]);
    TRANSPOSE_8x8_16BIT(V48[0], V49[0], V50[0], V51[0], V52[0], V53[0], V54[0], V55[0], T00[6], T01[6], T02[6], T03[6], T04[6], T05[6], T06[6], T07[6]);
    TRANSPOSE_8x8_16BIT(V56[0], V57[0], V58[0], V59[0], V60[0], V61[0], V62[0], V63[0], T00[7], T01[7], T02[7], T03[7], T04[7], T05[7], T06[7], T07[7]);

    TRANSPOSE_8x8_16BIT(V00[1], V01[1], V02[1], V03[1], V04[1], V05[1], V06[1], V07[1], T08[0], T09[0], T10[0], T11[0], T12[0], T13[0], T14[0], T15[0]);
    TRANSPOSE_8x8_16BIT(V08[1], V09[1], V10[1], V11[1], V12[1], V13[1], V14[1], V15[1], T08[1], T09[1], T10[1], T11[1], T12[1], T13[1], T14[1], T15[1]);
    TRANSPOSE_8x8_16BIT(V16[1], V17[1], V18[1], V19[1], V20[1], V21[1], V22[1], V23[1], T08[2], T09[2], T10[2], T11[2], T12[2], T13[2], T14[2], T15[2]);
    TRANSPOSE_8x8_16BIT(V24[1], V25[1], V26[1], V27[1], V28[1], V29[1], V30[1], V31[1], T08[3], T09[3], T10[3], T11[3], T12[3], T13[3], T14[3], T15[3]);
    TRANSPOSE_8x8_16BIT(V32[1], V33[1], V34[1], V35[1], V36[1], V37[1], V38[1], V39[1], T08[4], T09[4], T10[4], T11[4], T12[4], T13[4], T14[4], T15[4]);
    TRANSPOSE_8x8_16BIT(V40[1], V41[1], V42[1], V43[1], V44[1], V45[1], V46[1], V47[1], T08[5], T09[5], T10[5], T11[5], T12[5], T13[5], T14[5], T15[5]);
    TRANSPOSE_8x8_16BIT(V48[1], V49[1], V50[1], V51[1], V52[1], V53[1], V54[1], V55[1], T08[6], T09[6], T10[6], T11[6], T12[6], T13[6], T14[6], T15[6]);
    TRANSPOSE_8x8_16BIT(V56[1], V57[1], V58[1], V59[1], V60[1], V61[1], V62[1], V63[1], T08[7], T09[7], T10[7], T11[7], T12[7], T13[7], T14[7], T15[7]);


    /* step 1: horizontal transform */

    // pExt[x] -= (pExt[x - 1] + pExt[x + 1]) >> 1;

    for (i = 0; i < 8; i++) {
        T01[i] = _mm_sub_epi16(T01[i], _mm_srai_epi16(_mm_add_epi16(T00[i], T02[i]), 1));
        T03[i] = _mm_sub_epi16(T03[i], _mm_srai_epi16(_mm_add_epi16(T02[i], T04[i]), 1));
        T05[i] = _mm_sub_epi16(T05[i], _mm_srai_epi16(_mm_add_epi16(T04[i], T06[i]), 1));
        T07[i] = _mm_sub_epi16(T07[i], _mm_srai_epi16(_mm_add_epi16(T06[i], T08[i]), 1));

        T09[i] = _mm_sub_epi16(T09[i], _mm_srai_epi16(_mm_add_epi16(T08[i], T10[i]), 1));
        T11[i] = _mm_sub_epi16(T11[i], _mm_srai_epi16(_mm_add_epi16(T10[i], T12[i]), 1));
        T13[i] = _mm_sub_epi16(T13[i], _mm_srai_epi16(_mm_add_epi16(T12[i], T14[i]), 1));
        T15[i] = _mm_sub_epi16(T15[i], _mm_srai_epi16(_mm_add_epi16(T14[i], T14[i]), 1));
    }

    for (i = 0; i < 8; i++) {
        T00[i] = _mm_add_epi16(T00[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(T01[i], T01[i]), mAddOffset2), 2));
        T02[i] = _mm_add_epi16(T02[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(T01[i], T03[i]), mAddOffset2), 2));
        T04[i] = _mm_add_epi16(T04[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(T03[i], T05[i]), mAddOffset2), 2));
        T06[i] = _mm_add_epi16(T06[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(T05[i], T07[i]), mAddOffset2), 2));

        T08[i] = _mm_add_epi16(T08[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(T07[i], T09[i]), mAddOffset2), 2));
        T10[i] = _mm_add_epi16(T10[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(T09[i], T11[i]), mAddOffset2), 2));
        T12[i] = _mm_add_epi16(T12[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(T11[i], T13[i]), mAddOffset2), 2));
        T14[i] = _mm_add_epi16(T14[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(T13[i], T15[i]), mAddOffset2), 2));
    }

    /* step 2: vertical transform */
    /* copy ×ªï¿½ï¿½*/
    TRANSPOSE_8x8_16BIT(T00[0], T02[0], T04[0], T06[0], T08[0], T10[0], T12[0], T14[0], B00, B01, B02, B03, B04, B05, B06, B07);
    TRANSPOSE_8x8_16BIT(T00[1], T02[1], T04[1], T06[1], T08[1], T10[1], T12[1], T14[1], B08, B09, B10, B11, B12, B13, B14, B15);
    TRANSPOSE_8x8_16BIT(T00[2], T02[2], T04[2], T06[2], T08[2], T10[2], T12[2], T14[2], B16, B17, B18, B19, B20, B21, B22, B23);
    TRANSPOSE_8x8_16BIT(T00[3], T02[3], T04[3], T06[3], T08[3], T10[3], T12[3], T14[3], B24, B25, B26, B27, B28, B29, B30, B31);

    TRANSPOSE_8x8_16BIT(T00[4], T02[4], T04[4], T06[4], T08[4], T10[4], T12[4], T14[4], B32, B33, B34, B35, B36, B37, B38, B39);
    TRANSPOSE_8x8_16BIT(T00[5], T02[5], T04[5], T06[5], T08[5], T10[5], T12[5], T14[5], B40, B41, B42, B43, B44, B45, B46, B47);
    TRANSPOSE_8x8_16BIT(T00[6], T02[6], T04[6], T06[6], T08[6], T10[6], T12[6], T14[6], B48, B49, B50, B51, B52, B53, B54, B55);
    TRANSPOSE_8x8_16BIT(T00[7], T02[7], T04[7], T06[7], T08[7], T10[7], T12[7], T14[7], B56, B57, B58, B59, B60, B61, B62, B63);

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

void wavelet_64x16_sse128(coeff_t *coeff)
{
    //ï¿½ï¿½ï¿½ï¿½ 16*64
    __m128i V00[2], V01[2], V02[2], V03[2], V04[2], V05[2], V06[2], V07[2], V08[2], V09[2], V10[2], V11[2], V12[2], V13[2], V14[2], V15[2], V16[2], V17[2], V18[2], V19[2], V20[2], V21[2], V22[2], V23[2], V24[2], V25[2], V26[2], V27[2], V28[2], V29[2], V30[2], V31[2], V32[2], V33[2], V34[2], V35[2], V36[2], V37[2], V38[2], V39[2], V40[2], V41[2], V42[2], V43[2], V44[2], V45[2], V46[2], V47[2], V48[2], V49[2], V50[2], V51[2], V52[2], V53[2], V54[2], V55[2], V56[2], V57[2], V58[2], V59[2], V60[2], V61[2], V62[2], V63[2];

    //ï¿½ï¿½ï¿½ï¿½ 64*16
    __m128i T00[8], T01[8], T02[8], T03[8], T04[8], T05[8], T06[8], T07[8], T08[8], T09[8], T10[8], T11[8], T12[8], T13[8], T14[8], T15[8];

    //ï¿½ï¿½Ê± 64*16
    __m128i A00[4], A01[4], A02[4], A03[4], A04[4], A05[4], A06[4], A07[4], A08[4], A09[4], A10[4], A11[4], A12[4], A13[4], A14[4], A15[4];

    //ÁÙÊ±
    __m128i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
    __m128i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;
    int i;

    __m128i mAddOffset1 = _mm_set1_epi16(1);
    __m128i mAddOffset2 = _mm_set1_epi16(2);
    //load

    for (i = 0; i < 8; i++) {
        T00[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 0]);
        T01[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 1]);
        T02[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 2]);
        T03[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 3]);
        T04[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 4]);
        T05[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 5]);
        T06[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 6]);
        T07[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 7]);
        T08[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 8]);
        T09[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 9]);
        T10[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 10]);
        T11[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 11]);
        T12[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 12]);
        T13[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 13]);
        T14[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 14]);
        T15[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 15]);
    }

    TRANSPOSE_8x8_16BIT(T00[0], T01[0], T02[0], T03[0], T04[0], T05[0], T06[0], T07[0], V00[0], V01[0], V02[0], V03[0], V04[0], V05[0], V06[0], V07[0]);
    TRANSPOSE_8x8_16BIT(T00[1], T01[1], T02[1], T03[1], T04[1], T05[1], T06[1], T07[1], V08[0], V09[0], V10[0], V11[0], V12[0], V13[0], V14[0], V15[0]);
    TRANSPOSE_8x8_16BIT(T00[2], T01[2], T02[2], T03[2], T04[2], T05[2], T06[2], T07[2], V16[0], V17[0], V18[0], V19[0], V20[0], V21[0], V22[0], V23[0]);
    TRANSPOSE_8x8_16BIT(T00[3], T01[3], T02[3], T03[3], T04[3], T05[3], T06[3], T07[3], V24[0], V25[0], V26[0], V27[0], V28[0], V29[0], V30[0], V31[0]);

    TRANSPOSE_8x8_16BIT(T00[4], T01[4], T02[4], T03[4], T04[4], T05[4], T06[4], T07[4], V32[0], V33[0], V34[0], V35[0], V36[0], V37[0], V38[0], V39[0]);
    TRANSPOSE_8x8_16BIT(T00[5], T01[5], T02[5], T03[5], T04[5], T05[5], T06[5], T07[5], V40[0], V41[0], V42[0], V43[0], V44[0], V45[0], V46[0], V47[0]);
    TRANSPOSE_8x8_16BIT(T00[6], T01[6], T02[6], T03[6], T04[6], T05[6], T06[6], T07[6], V48[0], V49[0], V50[0], V51[0], V52[0], V53[0], V54[0], V55[0]);
    TRANSPOSE_8x8_16BIT(T00[7], T01[7], T02[7], T03[7], T04[7], T05[7], T06[7], T07[7], V56[0], V57[0], V58[0], V59[0], V60[0], V61[0], V62[0], V63[0]);

    TRANSPOSE_8x8_16BIT(T08[0], T09[0], T10[0], T11[0], T12[0], T13[0], T14[0], T15[0], V00[1], V01[1], V02[1], V03[1], V04[1], V05[1], V06[1], V07[1]);
    TRANSPOSE_8x8_16BIT(T08[1], T09[1], T10[1], T11[1], T12[1], T13[1], T14[1], T15[1], V08[1], V09[1], V10[1], V11[1], V12[1], V13[1], V14[1], V15[1]);
    TRANSPOSE_8x8_16BIT(T08[2], T09[2], T10[2], T11[2], T12[2], T13[2], T14[2], T15[2], V16[1], V17[1], V18[1], V19[1], V20[1], V21[1], V22[1], V23[1]);
    TRANSPOSE_8x8_16BIT(T08[3], T09[3], T10[3], T11[3], T12[3], T13[3], T14[3], T15[3], V24[1], V25[1], V26[1], V27[1], V28[1], V29[1], V30[1], V31[1]);

    TRANSPOSE_8x8_16BIT(T08[4], T09[4], T10[4], T11[4], T12[4], T13[4], T14[4], T15[4], V32[1], V33[1], V34[1], V35[1], V36[1], V37[1], V38[1], V39[1]);
    TRANSPOSE_8x8_16BIT(T08[5], T09[5], T10[5], T11[5], T12[5], T13[5], T14[5], T15[5], V40[1], V41[1], V42[1], V43[1], V44[1], V45[1], V46[1], V47[1]);
    TRANSPOSE_8x8_16BIT(T08[6], T09[6], T10[6], T11[6], T12[6], T13[6], T14[6], T15[6], V48[1], V49[1], V50[1], V51[1], V52[1], V53[1], V54[1], V55[1]);
    TRANSPOSE_8x8_16BIT(T08[7], T09[7], T10[7], T11[7], T12[7], T13[7], T14[7], T15[7], V56[1], V57[1], V58[1], V59[1], V60[1], V61[1], V62[1], V63[1]);

    //pExt[x] -= (pExt[x - 1] + pExt[x + 1]) >> 1;

    V01[0] = _mm_sub_epi16(V01[0], _mm_srai_epi16(_mm_add_epi16(V00[0], V02[0]), 1));
    V03[0] = _mm_sub_epi16(V03[0], _mm_srai_epi16(_mm_add_epi16(V02[0], V04[0]), 1));
    V05[0] = _mm_sub_epi16(V05[0], _mm_srai_epi16(_mm_add_epi16(V04[0], V06[0]), 1));
    V07[0] = _mm_sub_epi16(V07[0], _mm_srai_epi16(_mm_add_epi16(V06[0], V08[0]), 1));
    V09[0] = _mm_sub_epi16(V09[0], _mm_srai_epi16(_mm_add_epi16(V08[0], V10[0]), 1));
    V11[0] = _mm_sub_epi16(V11[0], _mm_srai_epi16(_mm_add_epi16(V10[0], V12[0]), 1));
    V13[0] = _mm_sub_epi16(V13[0], _mm_srai_epi16(_mm_add_epi16(V12[0], V14[0]), 1));
    V15[0] = _mm_sub_epi16(V15[0], _mm_srai_epi16(_mm_add_epi16(V14[0], V16[0]), 1));

    V17[0] = _mm_sub_epi16(V17[0], _mm_srai_epi16(_mm_add_epi16(V16[0], V18[0]), 1));
    V19[0] = _mm_sub_epi16(V19[0], _mm_srai_epi16(_mm_add_epi16(V18[0], V20[0]), 1));
    V21[0] = _mm_sub_epi16(V21[0], _mm_srai_epi16(_mm_add_epi16(V20[0], V22[0]), 1));
    V23[0] = _mm_sub_epi16(V23[0], _mm_srai_epi16(_mm_add_epi16(V22[0], V24[0]), 1));
    V25[0] = _mm_sub_epi16(V25[0], _mm_srai_epi16(_mm_add_epi16(V24[0], V26[0]), 1));
    V27[0] = _mm_sub_epi16(V27[0], _mm_srai_epi16(_mm_add_epi16(V26[0], V28[0]), 1));
    V29[0] = _mm_sub_epi16(V29[0], _mm_srai_epi16(_mm_add_epi16(V28[0], V30[0]), 1));
    V31[0] = _mm_sub_epi16(V31[0], _mm_srai_epi16(_mm_add_epi16(V30[0], V32[0]), 1));

    V33[0] = _mm_sub_epi16(V33[0], _mm_srai_epi16(_mm_add_epi16(V32[0], V34[0]), 1));
    V35[0] = _mm_sub_epi16(V35[0], _mm_srai_epi16(_mm_add_epi16(V34[0], V36[0]), 1));
    V37[0] = _mm_sub_epi16(V37[0], _mm_srai_epi16(_mm_add_epi16(V36[0], V38[0]), 1));
    V39[0] = _mm_sub_epi16(V39[0], _mm_srai_epi16(_mm_add_epi16(V38[0], V40[0]), 1));
    V41[0] = _mm_sub_epi16(V41[0], _mm_srai_epi16(_mm_add_epi16(V40[0], V42[0]), 1));
    V43[0] = _mm_sub_epi16(V43[0], _mm_srai_epi16(_mm_add_epi16(V42[0], V44[0]), 1));
    V45[0] = _mm_sub_epi16(V45[0], _mm_srai_epi16(_mm_add_epi16(V44[0], V46[0]), 1));
    V47[0] = _mm_sub_epi16(V47[0], _mm_srai_epi16(_mm_add_epi16(V46[0], V48[0]), 1));

    V49[0] = _mm_sub_epi16(V49[0], _mm_srai_epi16(_mm_add_epi16(V48[0], V50[0]), 1));
    V51[0] = _mm_sub_epi16(V51[0], _mm_srai_epi16(_mm_add_epi16(V50[0], V52[0]), 1));
    V53[0] = _mm_sub_epi16(V53[0], _mm_srai_epi16(_mm_add_epi16(V52[0], V54[0]), 1));
    V55[0] = _mm_sub_epi16(V55[0], _mm_srai_epi16(_mm_add_epi16(V54[0], V56[0]), 1));
    V57[0] = _mm_sub_epi16(V57[0], _mm_srai_epi16(_mm_add_epi16(V56[0], V58[0]), 1));
    V59[0] = _mm_sub_epi16(V59[0], _mm_srai_epi16(_mm_add_epi16(V58[0], V60[0]), 1));
    V61[0] = _mm_sub_epi16(V61[0], _mm_srai_epi16(_mm_add_epi16(V60[0], V62[0]), 1));
    V63[0] = _mm_sub_epi16(V63[0], _mm_srai_epi16(_mm_add_epi16(V62[0], V62[0]), 1));

    V01[1] = _mm_sub_epi16(V01[1], _mm_srai_epi16(_mm_add_epi16(V00[1], V02[1]), 1));
    V03[1] = _mm_sub_epi16(V03[1], _mm_srai_epi16(_mm_add_epi16(V02[1], V04[1]), 1));
    V05[1] = _mm_sub_epi16(V05[1], _mm_srai_epi16(_mm_add_epi16(V04[1], V06[1]), 1));
    V07[1] = _mm_sub_epi16(V07[1], _mm_srai_epi16(_mm_add_epi16(V06[1], V08[1]), 1));
    V09[1] = _mm_sub_epi16(V09[1], _mm_srai_epi16(_mm_add_epi16(V08[1], V10[1]), 1));
    V11[1] = _mm_sub_epi16(V11[1], _mm_srai_epi16(_mm_add_epi16(V10[1], V12[1]), 1));
    V13[1] = _mm_sub_epi16(V13[1], _mm_srai_epi16(_mm_add_epi16(V12[1], V14[1]), 1));
    V15[1] = _mm_sub_epi16(V15[1], _mm_srai_epi16(_mm_add_epi16(V14[1], V16[1]), 1));

    V17[1] = _mm_sub_epi16(V17[1], _mm_srai_epi16(_mm_add_epi16(V16[1], V18[1]), 1));
    V19[1] = _mm_sub_epi16(V19[1], _mm_srai_epi16(_mm_add_epi16(V18[1], V20[1]), 1));
    V21[1] = _mm_sub_epi16(V21[1], _mm_srai_epi16(_mm_add_epi16(V20[1], V22[1]), 1));
    V23[1] = _mm_sub_epi16(V23[1], _mm_srai_epi16(_mm_add_epi16(V22[1], V24[1]), 1));
    V25[1] = _mm_sub_epi16(V25[1], _mm_srai_epi16(_mm_add_epi16(V24[1], V26[1]), 1));
    V27[1] = _mm_sub_epi16(V27[1], _mm_srai_epi16(_mm_add_epi16(V26[1], V28[1]), 1));
    V29[1] = _mm_sub_epi16(V29[1], _mm_srai_epi16(_mm_add_epi16(V28[1], V30[1]), 1));
    V31[1] = _mm_sub_epi16(V31[1], _mm_srai_epi16(_mm_add_epi16(V30[1], V32[1]), 1));

    V33[1] = _mm_sub_epi16(V33[1], _mm_srai_epi16(_mm_add_epi16(V32[1], V34[1]), 1));
    V35[1] = _mm_sub_epi16(V35[1], _mm_srai_epi16(_mm_add_epi16(V34[1], V36[1]), 1));
    V37[1] = _mm_sub_epi16(V37[1], _mm_srai_epi16(_mm_add_epi16(V36[1], V38[1]), 1));
    V39[1] = _mm_sub_epi16(V39[1], _mm_srai_epi16(_mm_add_epi16(V38[1], V40[1]), 1));
    V41[1] = _mm_sub_epi16(V41[1], _mm_srai_epi16(_mm_add_epi16(V40[1], V42[1]), 1));
    V43[1] = _mm_sub_epi16(V43[1], _mm_srai_epi16(_mm_add_epi16(V42[1], V44[1]), 1));
    V45[1] = _mm_sub_epi16(V45[1], _mm_srai_epi16(_mm_add_epi16(V44[1], V46[1]), 1));
    V47[1] = _mm_sub_epi16(V47[1], _mm_srai_epi16(_mm_add_epi16(V46[1], V48[1]), 1));

    V49[1] = _mm_sub_epi16(V49[1], _mm_srai_epi16(_mm_add_epi16(V48[1], V50[1]), 1));
    V51[1] = _mm_sub_epi16(V51[1], _mm_srai_epi16(_mm_add_epi16(V50[1], V52[1]), 1));
    V53[1] = _mm_sub_epi16(V53[1], _mm_srai_epi16(_mm_add_epi16(V52[1], V54[1]), 1));
    V55[1] = _mm_sub_epi16(V55[1], _mm_srai_epi16(_mm_add_epi16(V54[1], V56[1]), 1));
    V57[1] = _mm_sub_epi16(V57[1], _mm_srai_epi16(_mm_add_epi16(V56[1], V58[1]), 1));
    V59[1] = _mm_sub_epi16(V59[1], _mm_srai_epi16(_mm_add_epi16(V58[1], V60[1]), 1));
    V61[1] = _mm_sub_epi16(V61[1], _mm_srai_epi16(_mm_add_epi16(V60[1], V62[1]), 1));
    V63[1] = _mm_sub_epi16(V63[1], _mm_srai_epi16(_mm_add_epi16(V62[1], V62[1]), 1));


    //pExt[x] += (pExt[x - 1] + pExt[x + 1] + 2) >> 2;

    V00[0] = _mm_add_epi16(V00[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V01[0], V01[0]), mAddOffset2), 2));
    V02[0] = _mm_add_epi16(V02[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V01[0], V03[0]), mAddOffset2), 2));
    V04[0] = _mm_add_epi16(V04[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V03[0], V05[0]), mAddOffset2), 2));
    V06[0] = _mm_add_epi16(V06[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V05[0], V07[0]), mAddOffset2), 2));
    V08[0] = _mm_add_epi16(V08[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V07[0], V09[0]), mAddOffset2), 2));
    V10[0] = _mm_add_epi16(V10[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V09[0], V11[0]), mAddOffset2), 2));
    V12[0] = _mm_add_epi16(V12[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V11[0], V13[0]), mAddOffset2), 2));
    V14[0] = _mm_add_epi16(V14[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V13[0], V15[0]), mAddOffset2), 2));

    V16[0] = _mm_add_epi16(V16[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V15[0], V17[0]), mAddOffset2), 2));
    V18[0] = _mm_add_epi16(V18[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V17[0], V19[0]), mAddOffset2), 2));
    V20[0] = _mm_add_epi16(V20[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V19[0], V21[0]), mAddOffset2), 2));
    V22[0] = _mm_add_epi16(V22[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V21[0], V23[0]), mAddOffset2), 2));
    V24[0] = _mm_add_epi16(V24[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V23[0], V25[0]), mAddOffset2), 2));
    V26[0] = _mm_add_epi16(V26[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V25[0], V27[0]), mAddOffset2), 2));
    V28[0] = _mm_add_epi16(V28[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V27[0], V29[0]), mAddOffset2), 2));
    V30[0] = _mm_add_epi16(V30[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V29[0], V31[0]), mAddOffset2), 2));

    V32[0] = _mm_add_epi16(V32[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V31[0], V33[0]), mAddOffset2), 2));
    V34[0] = _mm_add_epi16(V34[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V33[0], V35[0]), mAddOffset2), 2));
    V36[0] = _mm_add_epi16(V36[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V35[0], V37[0]), mAddOffset2), 2));
    V38[0] = _mm_add_epi16(V38[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V37[0], V39[0]), mAddOffset2), 2));
    V40[0] = _mm_add_epi16(V40[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V39[0], V41[0]), mAddOffset2), 2));
    V42[0] = _mm_add_epi16(V42[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V41[0], V43[0]), mAddOffset2), 2));
    V44[0] = _mm_add_epi16(V44[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V43[0], V45[0]), mAddOffset2), 2));
    V46[0] = _mm_add_epi16(V46[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V45[0], V47[0]), mAddOffset2), 2));

    V48[0] = _mm_add_epi16(V48[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V47[0], V49[0]), mAddOffset2), 2));
    V50[0] = _mm_add_epi16(V50[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V49[0], V51[0]), mAddOffset2), 2));
    V52[0] = _mm_add_epi16(V52[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V51[0], V53[0]), mAddOffset2), 2));
    V54[0] = _mm_add_epi16(V54[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V53[0], V55[0]), mAddOffset2), 2));
    V56[0] = _mm_add_epi16(V56[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V55[0], V57[0]), mAddOffset2), 2));
    V58[0] = _mm_add_epi16(V58[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V57[0], V59[0]), mAddOffset2), 2));
    V60[0] = _mm_add_epi16(V60[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V59[0], V61[0]), mAddOffset2), 2));
    V62[0] = _mm_add_epi16(V62[0], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V61[0], V63[0]), mAddOffset2), 2));

    V00[1] = _mm_add_epi16(V00[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V01[1], V01[1]), mAddOffset2), 2));
    V02[1] = _mm_add_epi16(V02[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V01[1], V03[1]), mAddOffset2), 2));
    V04[1] = _mm_add_epi16(V04[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V03[1], V05[1]), mAddOffset2), 2));
    V06[1] = _mm_add_epi16(V06[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V05[1], V07[1]), mAddOffset2), 2));
    V08[1] = _mm_add_epi16(V08[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V07[1], V09[1]), mAddOffset2), 2));
    V10[1] = _mm_add_epi16(V10[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V09[1], V11[1]), mAddOffset2), 2));
    V12[1] = _mm_add_epi16(V12[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V11[1], V13[1]), mAddOffset2), 2));
    V14[1] = _mm_add_epi16(V14[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V13[1], V15[1]), mAddOffset2), 2));

    V16[1] = _mm_add_epi16(V16[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V15[1], V17[1]), mAddOffset2), 2));
    V18[1] = _mm_add_epi16(V18[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V17[1], V19[1]), mAddOffset2), 2));
    V20[1] = _mm_add_epi16(V20[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V19[1], V21[1]), mAddOffset2), 2));
    V22[1] = _mm_add_epi16(V22[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V21[1], V23[1]), mAddOffset2), 2));
    V24[1] = _mm_add_epi16(V24[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V23[1], V25[1]), mAddOffset2), 2));
    V26[1] = _mm_add_epi16(V26[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V25[1], V27[1]), mAddOffset2), 2));
    V28[1] = _mm_add_epi16(V28[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V27[1], V29[1]), mAddOffset2), 2));
    V30[1] = _mm_add_epi16(V30[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V29[1], V31[1]), mAddOffset2), 2));

    V32[1] = _mm_add_epi16(V32[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V31[1], V33[1]), mAddOffset2), 2));
    V34[1] = _mm_add_epi16(V34[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V33[1], V35[1]), mAddOffset2), 2));
    V36[1] = _mm_add_epi16(V36[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V35[1], V37[1]), mAddOffset2), 2));
    V38[1] = _mm_add_epi16(V38[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V37[1], V39[1]), mAddOffset2), 2));
    V40[1] = _mm_add_epi16(V40[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V39[1], V41[1]), mAddOffset2), 2));
    V42[1] = _mm_add_epi16(V42[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V41[1], V43[1]), mAddOffset2), 2));
    V44[1] = _mm_add_epi16(V44[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V43[1], V45[1]), mAddOffset2), 2));
    V46[1] = _mm_add_epi16(V46[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V45[1], V47[1]), mAddOffset2), 2));

    V48[1] = _mm_add_epi16(V48[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V47[1], V49[1]), mAddOffset2), 2));
    V50[1] = _mm_add_epi16(V50[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V49[1], V51[1]), mAddOffset2), 2));
    V52[1] = _mm_add_epi16(V52[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V51[1], V53[1]), mAddOffset2), 2));
    V54[1] = _mm_add_epi16(V54[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V53[1], V55[1]), mAddOffset2), 2));
    V56[1] = _mm_add_epi16(V56[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V55[1], V57[1]), mAddOffset2), 2));
    V58[1] = _mm_add_epi16(V58[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V57[1], V59[1]), mAddOffset2), 2));
    V60[1] = _mm_add_epi16(V60[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V59[1], V61[1]), mAddOffset2), 2));
    V62[1] = _mm_add_epi16(V62[1], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V61[1], V63[1]), mAddOffset2), 2));

    /* step 2: vertical transform */
    //×ªï¿½ï¿½
    TRANSPOSE_8x8_16BIT(V00[0], V02[0], V04[0], V06[0], V08[0], V10[0], V12[0], V14[0], A00[0], A01[0], A02[0], A03[0], A04[0], A05[0], A06[0], A07[0]);
    TRANSPOSE_8x8_16BIT(V16[0], V18[0], V20[0], V22[0], V24[0], V26[0], V28[0], V30[0], A00[1], A01[1], A02[1], A03[1], A04[1], A05[1], A06[1], A07[1]);
    TRANSPOSE_8x8_16BIT(V32[0], V34[0], V36[0], V38[0], V40[0], V42[0], V44[0], V46[0], A00[2], A01[2], A02[2], A03[2], A04[2], A05[2], A06[2], A07[2]);
    TRANSPOSE_8x8_16BIT(V48[0], V50[0], V52[0], V54[0], V56[0], V58[0], V60[0], V62[0], A00[3], A01[3], A02[3], A03[3], A04[3], A05[3], A06[3], A07[3]);

    TRANSPOSE_8x8_16BIT(V00[1], V02[1], V04[1], V06[1], V08[1], V10[1], V12[1], V14[1], A08[0], A09[0], A10[0], A11[0], A12[0], A13[0], A14[0], A15[0]);
    TRANSPOSE_8x8_16BIT(V16[1], V18[1], V20[1], V22[1], V24[1], V26[1], V28[1], V30[1], A08[1], A09[1], A10[1], A11[1], A12[1], A13[1], A14[1], A15[1]);
    TRANSPOSE_8x8_16BIT(V32[1], V34[1], V36[1], V38[1], V40[1], V42[1], V44[1], V46[1], A08[2], A09[2], A10[2], A11[2], A12[2], A13[2], A14[2], A15[2]);
    TRANSPOSE_8x8_16BIT(V48[1], V50[1], V52[1], V54[1], V56[1], V58[1], V60[1], V62[1], A08[3], A09[3], A10[3], A11[3], A12[3], A13[3], A14[3], A15[3]);

    //pExt[y] -= (pExt[y - 1] + pExt[y + 1]) >> 1;
    for (i = 0; i < 4; i++) {
        A01[i] = _mm_sub_epi16(A01[i], _mm_srai_epi16(_mm_add_epi16(A00[i], A02[i]), 1));
        A03[i] = _mm_sub_epi16(A03[i], _mm_srai_epi16(_mm_add_epi16(A02[i], A04[i]), 1));
        A05[i] = _mm_sub_epi16(A05[i], _mm_srai_epi16(_mm_add_epi16(A04[i], A06[i]), 1));
        A07[i] = _mm_sub_epi16(A07[i], _mm_srai_epi16(_mm_add_epi16(A06[i], A08[i]), 1));
        A09[i] = _mm_sub_epi16(A09[i], _mm_srai_epi16(_mm_add_epi16(A08[i], A10[i]), 1));
        A11[i] = _mm_sub_epi16(A11[i], _mm_srai_epi16(_mm_add_epi16(A10[i], A12[i]), 1));
        A13[i] = _mm_sub_epi16(A13[i], _mm_srai_epi16(_mm_add_epi16(A12[i], A14[i]), 1));
        A15[i] = _mm_sub_epi16(A15[i], _mm_srai_epi16(_mm_add_epi16(A14[i], A14[i]), 1));
    }

    //pExt[y] = (pExt[y] << 1) + ((pExt[y - 1] + pExt[y + 1] + 1) >> 1);
    for (i = 0; i < 4; i++) {
        A00[i] = _mm_add_epi16(_mm_slli_epi16(A00[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A01[i], A01[i]), mAddOffset1), 1));
        A02[i] = _mm_add_epi16(_mm_slli_epi16(A02[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A01[i], A03[i]), mAddOffset1), 1));
        A04[i] = _mm_add_epi16(_mm_slli_epi16(A04[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A03[i], A05[i]), mAddOffset1), 1));
        A06[i] = _mm_add_epi16(_mm_slli_epi16(A06[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A05[i], A07[i]), mAddOffset1), 1));
        A08[i] = _mm_add_epi16(_mm_slli_epi16(A08[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A07[i], A09[i]), mAddOffset1), 1));
        A10[i] = _mm_add_epi16(_mm_slli_epi16(A10[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A09[i], A11[i]), mAddOffset1), 1));
        A12[i] = _mm_add_epi16(_mm_slli_epi16(A12[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A11[i], A13[i]), mAddOffset1), 1));
        A14[i] = _mm_add_epi16(_mm_slli_epi16(A14[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A13[i], A15[i]), mAddOffset1), 1));
    }

    //Store
    for (i = 0; i < 4; i++) {
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 0], A00[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 1], A02[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 2], A04[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 3], A06[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 4], A08[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 5], A10[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 6], A12[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 7], A14[i]);
    }
}

void wavelet_64x64_sse128(coeff_t *coeff)
{
    //ï¿½ï¿½ï¿½ï¿½ 16*64
    __m128i V00[8], V01[8], V02[8], V03[8], V04[8], V05[8], V06[8], V07[8], V08[8], V09[8], V10[8], V11[8], V12[8], V13[8], V14[8], V15[8], V16[8], V17[8], V18[8], V19[8], V20[8], V21[8], V22[8], V23[8], V24[8], V25[8], V26[8], V27[8], V28[8], V29[8], V30[8], V31[8], V32[8], V33[8], V34[8], V35[8], V36[8], V37[8], V38[8], V39[8], V40[8], V41[8], V42[8], V43[8], V44[8], V45[8], V46[8], V47[8], V48[8], V49[8], V50[8], V51[8], V52[8], V53[8], V54[8], V55[8], V56[8], V57[8], V58[8], V59[8], V60[8], V61[8], V62[8], V63[8];

    //ï¿½ï¿½ï¿½ï¿½ 64*64
    __m128i T00[8], T01[8], T02[8], T03[8], T04[8], T05[8], T06[8], T07[8], T08[8], T09[8], T10[8], T11[8], T12[8], T13[8], T14[8], T15[8], T16[8], T17[8], T18[8], T19[8], T20[8], T21[8], T22[8], T23[8], T24[8], T25[8], T26[8], T27[8], T28[8], T29[8], T30[8], T31[8], T32[8], T33[8], T34[8], T35[8], T36[8], T37[8], T38[8], T39[8], T40[8], T41[8], T42[8], T43[8], T44[8], T45[8], T46[8], T47[8], T48[8], T49[8], T50[8], T51[8], T52[8], T53[8], T54[8], T55[8], T56[8], T57[8], T58[8], T59[8], T60[8], T61[8], T62[8], T63[8];

    //ÁÙÊ± 32*64
    __m128i A00[4], A01[4], A02[4], A03[4], A04[4], A05[4], A06[4], A07[4], A08[4], A09[4], A10[4], A11[4], A12[4], A13[4], A14[4], A15[4], A16[4], A17[4], A18[4], A19[4], A20[4], A21[4], A22[4], A23[4], A24[4], A25[4], A26[4], A27[4], A28[4], A29[4], A30[4], A31[4], A32[4], A33[4], A34[4], A35[4], A36[4], A37[4], A38[4], A39[4], A40[4], A41[4], A42[4], A43[4], A44[4], A45[4], A46[4], A47[4], A48[4], A49[4], A50[4], A51[4], A52[4], A53[4], A54[4], A55[4], A56[4], A57[4], A58[4], A59[4], A60[4], A61[4], A62[4], A63[4];

    __m128i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
    __m128i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;

    __m128i mAddOffset1 = _mm_set1_epi16(1);
    __m128i mAddOffset2 = _mm_set1_epi16(2);
    int i;

    for (i = 0; i < 8; i++) {
        T00[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 0]);
        T01[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 1]);
        T02[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 2]);
        T03[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 3]);
        T04[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 4]);
        T05[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 5]);
        T06[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 6]);
        T07[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 7]);

        T08[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 8]);
        T09[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 9]);
        T10[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 10]);
        T11[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 11]);
        T12[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 12]);
        T13[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 13]);
        T14[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 14]);
        T15[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 15]);

        T16[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 16]);
        T17[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 17]);
        T18[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 18]);
        T19[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 19]);
        T20[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 20]);
        T21[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 21]);
        T22[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 22]);
        T23[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 23]);

        T24[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 24]);
        T25[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 25]);
        T26[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 26]);
        T27[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 27]);
        T28[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 28]);
        T29[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 29]);
        T30[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 30]);
        T31[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 31]);

        T32[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 32]);
        T33[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 33]);
        T34[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 34]);
        T35[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 35]);
        T36[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 36]);
        T37[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 37]);
        T38[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 38]);
        T39[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 39]);

        T40[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 40]);
        T41[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 41]);
        T42[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 42]);
        T43[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 43]);
        T44[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 44]);
        T45[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 45]);
        T46[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 46]);
        T47[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 47]);

        T48[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 48]);
        T49[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 49]);
        T50[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 50]);
        T51[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 51]);
        T52[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 52]);
        T53[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 53]);
        T54[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 54]);
        T55[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 55]);

        T56[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 56]);
        T57[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 57]);
        T58[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 58]);
        T59[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 59]);
        T60[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 60]);
        T61[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 61]);
        T62[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 62]);
        T63[i] = _mm_load_si128((__m128i*)&coeff[8 * i + 64 * 63]);
    }
    //0-15ï¿½ï¿½×ªï¿½ï¿½
    TRANSPOSE_16x16_16BIT(
        T00[0], T01[0], T02[0], T03[0], T04[0], T05[0], T06[0], T07[0], T08[0], T09[0], T10[0], T11[0], T12[0], T13[0], T14[0], T15[0],
        T00[1], T01[1], T02[1], T03[1], T04[1], T05[1], T06[1], T07[1], T08[1], T09[1], T10[1], T11[1], T12[1], T13[1], T14[1], T15[1],
        V00[0], V01[0], V02[0], V03[0], V04[0], V05[0], V06[0], V07[0], V08[0], V09[0], V10[0], V11[0], V12[0], V13[0], V14[0], V15[0],
        V00[1], V01[1], V02[1], V03[1], V04[1], V05[1], V06[1], V07[1], V08[1], V09[1], V10[1], V11[1], V12[1], V13[1], V14[1], V15[1]
    );
    TRANSPOSE_16x16_16BIT(
        T00[2], T01[2], T02[2], T03[2], T04[2], T05[2], T06[2], T07[2], T08[2], T09[2], T10[2], T11[2], T12[2], T13[2], T14[2], T15[2],
        T00[3], T01[3], T02[3], T03[3], T04[3], T05[3], T06[3], T07[3], T08[3], T09[3], T10[3], T11[3], T12[3], T13[3], T14[3], T15[3],
        V16[0], V17[0], V18[0], V19[0], V20[0], V21[0], V22[0], V23[0], V24[0], V25[0], V26[0], V27[0], V28[0], V29[0], V30[0], V31[0],
        V16[1], V17[1], V18[1], V19[1], V20[1], V21[1], V22[1], V23[1], V24[1], V25[1], V26[1], V27[1], V28[1], V29[1], V30[1], V31[1]
    );
    TRANSPOSE_16x16_16BIT(
        T00[4], T01[4], T02[4], T03[4], T04[4], T05[4], T06[4], T07[4], T08[4], T09[4], T10[4], T11[4], T12[4], T13[4], T14[4], T15[4],
        T00[5], T01[5], T02[5], T03[5], T04[5], T05[5], T06[5], T07[5], T08[5], T09[5], T10[5], T11[5], T12[5], T13[5], T14[5], T15[5],
        V32[0], V33[0], V34[0], V35[0], V36[0], V37[0], V38[0], V39[0], V40[0], V41[0], V42[0], V43[0], V44[0], V45[0], V46[0], V47[0],
        V32[1], V33[1], V34[1], V35[1], V36[1], V37[1], V38[1], V39[1], V40[1], V41[1], V42[1], V43[1], V44[1], V45[1], V46[1], V47[1]
    );
    TRANSPOSE_16x16_16BIT(
        T00[6], T01[6], T02[6], T03[6], T04[6], T05[6], T06[6], T07[6], T08[6], T09[6], T10[6], T11[6], T12[6], T13[6], T14[6], T15[6],
        T00[7], T01[7], T02[7], T03[7], T04[7], T05[7], T06[7], T07[7], T08[7], T09[7], T10[7], T11[7], T12[7], T13[7], T14[7], T15[7],
        V48[0], V49[0], V50[0], V51[0], V52[0], V53[0], V54[0], V55[0], V56[0], V57[0], V58[0], V59[0], V60[0], V61[0], V62[0], V63[0],
        V48[1], V49[1], V50[1], V51[1], V52[1], V53[1], V54[1], V55[1], V56[1], V57[1], V58[1], V59[1], V60[1], V61[1], V62[1], V63[1]
    );
    //16-31ï¿½ï¿½×ªï¿½ï¿½
    TRANSPOSE_16x16_16BIT(
        T16[0], T17[0], T18[0], T19[0], T20[0], T21[0], T22[0], T23[0], T24[0], T25[0], T26[0], T27[0], T28[0], T29[0], T30[0], T31[0],
        T16[1], T17[1], T18[1], T19[1], T20[1], T21[1], T22[1], T23[1], T24[1], T25[1], T26[1], T27[1], T28[1], T29[1], T30[1], T31[1],
        V00[2], V01[2], V02[2], V03[2], V04[2], V05[2], V06[2], V07[2], V08[2], V09[2], V10[2], V11[2], V12[2], V13[2], V14[2], V15[2],
        V00[3], V01[3], V02[3], V03[3], V04[3], V05[3], V06[3], V07[3], V08[3], V09[3], V10[3], V11[3], V12[3], V13[3], V14[3], V15[3]
    );
    TRANSPOSE_16x16_16BIT(
        T16[2], T17[2], T18[2], T19[2], T20[2], T21[2], T22[2], T23[2], T24[2], T25[2], T26[2], T27[2], T28[2], T29[2], T30[2], T31[2],
        T16[3], T17[3], T18[3], T19[3], T20[3], T21[3], T22[3], T23[3], T24[3], T25[3], T26[3], T27[3], T28[3], T29[3], T30[3], T31[3],
        V16[2], V17[2], V18[2], V19[2], V20[2], V21[2], V22[2], V23[2], V24[2], V25[2], V26[2], V27[2], V28[2], V29[2], V30[2], V31[2],
        V16[3], V17[3], V18[3], V19[3], V20[3], V21[3], V22[3], V23[3], V24[3], V25[3], V26[3], V27[3], V28[3], V29[3], V30[3], V31[3]
    );
    TRANSPOSE_16x16_16BIT(
        T16[4], T17[4], T18[4], T19[4], T20[4], T21[4], T22[4], T23[4], T24[4], T25[4], T26[4], T27[4], T28[4], T29[4], T30[4], T31[4],
        T16[5], T17[5], T18[5], T19[5], T20[5], T21[5], T22[5], T23[5], T24[5], T25[5], T26[5], T27[5], T28[5], T29[5], T30[5], T31[5],
        V32[2], V33[2], V34[2], V35[2], V36[2], V37[2], V38[2], V39[2], V40[2], V41[2], V42[2], V43[2], V44[2], V45[2], V46[2], V47[2],
        V32[3], V33[3], V34[3], V35[3], V36[3], V37[3], V38[3], V39[3], V40[3], V41[3], V42[3], V43[3], V44[3], V45[3], V46[3], V47[3]
    );
    TRANSPOSE_16x16_16BIT(
        T16[6], T17[6], T18[6], T19[6], T20[6], T21[6], T22[6], T23[6], T24[6], T25[6], T26[6], T27[6], T28[6], T29[6], T30[6], T31[6],
        T16[7], T17[7], T18[7], T19[7], T20[7], T21[7], T22[7], T23[7], T24[7], T25[7], T26[7], T27[7], T28[7], T29[7], T30[7], T31[7],
        V48[2], V49[2], V50[2], V51[2], V52[2], V53[2], V54[2], V55[2], V56[2], V57[2], V58[2], V59[2], V60[2], V61[2], V62[2], V63[2],
        V48[3], V49[3], V50[3], V51[3], V52[3], V53[3], V54[3], V55[3], V56[3], V57[3], V58[3], V59[3], V60[3], V61[3], V62[3], V63[3]
    );
    //32-47ï¿½ï¿½×ªï¿½ï¿½
    TRANSPOSE_16x16_16BIT(
        T32[0], T33[0], T34[0], T35[0], T36[0], T37[0], T38[0], T39[0], T40[0], T41[0], T42[0], T43[0], T44[0], T45[0], T46[0], T47[0],
        T32[1], T33[1], T34[1], T35[1], T36[1], T37[1], T38[1], T39[1], T40[1], T41[1], T42[1], T43[1], T44[1], T45[1], T46[1], T47[1],
        V00[4], V01[4], V02[4], V03[4], V04[4], V05[4], V06[4], V07[4], V08[4], V09[4], V10[4], V11[4], V12[4], V13[4], V14[4], V15[4],
        V00[5], V01[5], V02[5], V03[5], V04[5], V05[5], V06[5], V07[5], V08[5], V09[5], V10[5], V11[5], V12[5], V13[5], V14[5], V15[5]
    );
    TRANSPOSE_16x16_16BIT(
        T32[2], T33[2], T34[2], T35[2], T36[2], T37[2], T38[2], T39[2], T40[2], T41[2], T42[2], T43[2], T44[2], T45[2], T46[2], T47[2],
        T32[3], T33[3], T34[3], T35[3], T36[3], T37[3], T38[3], T39[3], T40[3], T41[3], T42[3], T43[3], T44[3], T45[3], T46[3], T47[3],
        V16[4], V17[4], V18[4], V19[4], V20[4], V21[4], V22[4], V23[4], V24[4], V25[4], V26[4], V27[4], V28[4], V29[4], V30[4], V31[4],
        V16[5], V17[5], V18[5], V19[5], V20[5], V21[5], V22[5], V23[5], V24[5], V25[5], V26[5], V27[5], V28[5], V29[5], V30[5], V31[5]
    );
    TRANSPOSE_16x16_16BIT(
        T32[4], T33[4], T34[4], T35[4], T36[4], T37[4], T38[4], T39[4], T40[4], T41[4], T42[4], T43[4], T44[4], T45[4], T46[4], T47[4],
        T32[5], T33[5], T34[5], T35[5], T36[5], T37[5], T38[5], T39[5], T40[5], T41[5], T42[5], T43[5], T44[5], T45[5], T46[5], T47[5],
        V32[4], V33[4], V34[4], V35[4], V36[4], V37[4], V38[4], V39[4], V40[4], V41[4], V42[4], V43[4], V44[4], V45[4], V46[4], V47[4],
        V32[5], V33[5], V34[5], V35[5], V36[5], V37[5], V38[5], V39[5], V40[5], V41[5], V42[5], V43[5], V44[5], V45[5], V46[5], V47[5]
    );
    TRANSPOSE_16x16_16BIT(
        T32[6], T33[6], T34[6], T35[6], T36[6], T37[6], T38[6], T39[6], T40[6], T41[6], T42[6], T43[6], T44[6], T45[6], T46[6], T47[6],
        T32[7], T33[7], T34[7], T35[7], T36[7], T37[7], T38[7], T39[7], T40[7], T41[7], T42[7], T43[7], T44[7], T45[7], T46[7], T47[7],
        V48[4], V49[4], V50[4], V51[4], V52[4], V53[4], V54[4], V55[4], V56[4], V57[4], V58[4], V59[4], V60[4], V61[4], V62[4], V63[4],
        V48[5], V49[5], V50[5], V51[5], V52[5], V53[5], V54[5], V55[5], V56[5], V57[5], V58[5], V59[5], V60[5], V61[5], V62[5], V63[5]
    );
    //48-63ï¿½ï¿½×ªï¿½ï¿½
    TRANSPOSE_16x16_16BIT(
        T48[0], T49[0], T50[0], T51[0], T52[0], T53[0], T54[0], T55[0], T56[0], T57[0], T58[0], T59[0], T60[0], T61[0], T62[0], T63[0],
        T48[1], T49[1], T50[1], T51[1], T52[1], T53[1], T54[1], T55[1], T56[1], T57[1], T58[1], T59[1], T60[1], T61[1], T62[1], T63[1],
        V00[6], V01[6], V02[6], V03[6], V04[6], V05[6], V06[6], V07[6], V08[6], V09[6], V10[6], V11[6], V12[6], V13[6], V14[6], V15[6],
        V00[7], V01[7], V02[7], V03[7], V04[7], V05[7], V06[7], V07[7], V08[7], V09[7], V10[7], V11[7], V12[7], V13[7], V14[7], V15[7]
    );
    TRANSPOSE_16x16_16BIT(
        T48[2], T49[2], T50[2], T51[2], T52[2], T53[2], T54[2], T55[2], T56[2], T57[2], T58[2], T59[2], T60[2], T61[2], T62[2], T63[2],
        T48[3], T49[3], T50[3], T51[3], T52[3], T53[3], T54[3], T55[3], T56[3], T57[3], T58[3], T59[3], T60[3], T61[3], T62[3], T63[3],
        V16[6], V17[6], V18[6], V19[6], V20[6], V21[6], V22[6], V23[6], V24[6], V25[6], V26[6], V27[6], V28[6], V29[6], V30[6], V31[6],
        V16[7], V17[7], V18[7], V19[7], V20[7], V21[7], V22[7], V23[7], V24[7], V25[7], V26[7], V27[7], V28[7], V29[7], V30[7], V31[7]
    );
    TRANSPOSE_16x16_16BIT(
        T48[4], T49[4], T50[4], T51[4], T52[4], T53[4], T54[4], T55[4], T56[4], T57[4], T58[4], T59[4], T60[4], T61[4], T62[4], T63[4],
        T48[5], T49[5], T50[5], T51[5], T52[5], T53[5], T54[5], T55[5], T56[5], T57[5], T58[5], T59[5], T60[5], T61[5], T62[5], T63[5],
        V32[6], V33[6], V34[6], V35[6], V36[6], V37[6], V38[6], V39[6], V40[6], V41[6], V42[6], V43[6], V44[6], V45[6], V46[6], V47[6],
        V32[7], V33[7], V34[7], V35[7], V36[7], V37[7], V38[7], V39[7], V40[7], V41[7], V42[7], V43[7], V44[7], V45[7], V46[7], V47[7]
    );
    TRANSPOSE_16x16_16BIT(
        T48[6], T49[6], T50[6], T51[6], T52[6], T53[6], T54[6], T55[6], T56[6], T57[6], T58[6], T59[6], T60[6], T61[6], T62[6], T63[6],
        T48[7], T49[7], T50[7], T51[7], T52[7], T53[7], T54[7], T55[7], T56[7], T57[7], T58[7], T59[7], T60[7], T61[7], T62[7], T63[7],
        V48[6], V49[6], V50[6], V51[6], V52[6], V53[6], V54[6], V55[6], V56[6], V57[6], V58[6], V59[6], V60[6], V61[6], V62[6], V63[6],
        V48[7], V49[7], V50[7], V51[7], V52[7], V53[7], V54[7], V55[7], V56[7], V57[7], V58[7], V59[7], V60[7], V61[7], V62[7], V63[7]
    );

    //pExt[x] -= (pExt[x - 1] + pExt[x + 1]) >> 1;
    for (i = 0; i < 8; i++) {
        V01[i] = _mm_sub_epi16(V01[i], _mm_srai_epi16(_mm_add_epi16(V00[i], V02[i]), 1));
        V03[i] = _mm_sub_epi16(V03[i], _mm_srai_epi16(_mm_add_epi16(V02[i], V04[i]), 1));
        V05[i] = _mm_sub_epi16(V05[i], _mm_srai_epi16(_mm_add_epi16(V04[i], V06[i]), 1));
        V07[i] = _mm_sub_epi16(V07[i], _mm_srai_epi16(_mm_add_epi16(V06[i], V08[i]), 1));
        V09[i] = _mm_sub_epi16(V09[i], _mm_srai_epi16(_mm_add_epi16(V08[i], V10[i]), 1));
        V11[i] = _mm_sub_epi16(V11[i], _mm_srai_epi16(_mm_add_epi16(V10[i], V12[i]), 1));
        V13[i] = _mm_sub_epi16(V13[i], _mm_srai_epi16(_mm_add_epi16(V12[i], V14[i]), 1));
        V15[i] = _mm_sub_epi16(V15[i], _mm_srai_epi16(_mm_add_epi16(V14[i], V16[i]), 1));

        V17[i] = _mm_sub_epi16(V17[i], _mm_srai_epi16(_mm_add_epi16(V16[i], V18[i]), 1));
        V19[i] = _mm_sub_epi16(V19[i], _mm_srai_epi16(_mm_add_epi16(V18[i], V20[i]), 1));
        V21[i] = _mm_sub_epi16(V21[i], _mm_srai_epi16(_mm_add_epi16(V20[i], V22[i]), 1));
        V23[i] = _mm_sub_epi16(V23[i], _mm_srai_epi16(_mm_add_epi16(V22[i], V24[i]), 1));
        V25[i] = _mm_sub_epi16(V25[i], _mm_srai_epi16(_mm_add_epi16(V24[i], V26[i]), 1));
        V27[i] = _mm_sub_epi16(V27[i], _mm_srai_epi16(_mm_add_epi16(V26[i], V28[i]), 1));
        V29[i] = _mm_sub_epi16(V29[i], _mm_srai_epi16(_mm_add_epi16(V28[i], V30[i]), 1));
        V31[i] = _mm_sub_epi16(V31[i], _mm_srai_epi16(_mm_add_epi16(V30[i], V32[i]), 1));

        V33[i] = _mm_sub_epi16(V33[i], _mm_srai_epi16(_mm_add_epi16(V32[i], V34[i]), 1));
        V35[i] = _mm_sub_epi16(V35[i], _mm_srai_epi16(_mm_add_epi16(V34[i], V36[i]), 1));
        V37[i] = _mm_sub_epi16(V37[i], _mm_srai_epi16(_mm_add_epi16(V36[i], V38[i]), 1));
        V39[i] = _mm_sub_epi16(V39[i], _mm_srai_epi16(_mm_add_epi16(V38[i], V40[i]), 1));
        V41[i] = _mm_sub_epi16(V41[i], _mm_srai_epi16(_mm_add_epi16(V40[i], V42[i]), 1));
        V43[i] = _mm_sub_epi16(V43[i], _mm_srai_epi16(_mm_add_epi16(V42[i], V44[i]), 1));
        V45[i] = _mm_sub_epi16(V45[i], _mm_srai_epi16(_mm_add_epi16(V44[i], V46[i]), 1));
        V47[i] = _mm_sub_epi16(V47[i], _mm_srai_epi16(_mm_add_epi16(V46[i], V48[i]), 1));

        V49[i] = _mm_sub_epi16(V49[i], _mm_srai_epi16(_mm_add_epi16(V48[i], V50[i]), 1));
        V51[i] = _mm_sub_epi16(V51[i], _mm_srai_epi16(_mm_add_epi16(V50[i], V52[i]), 1));
        V53[i] = _mm_sub_epi16(V53[i], _mm_srai_epi16(_mm_add_epi16(V52[i], V54[i]), 1));
        V55[i] = _mm_sub_epi16(V55[i], _mm_srai_epi16(_mm_add_epi16(V54[i], V56[i]), 1));
        V57[i] = _mm_sub_epi16(V57[i], _mm_srai_epi16(_mm_add_epi16(V56[i], V58[i]), 1));
        V59[i] = _mm_sub_epi16(V59[i], _mm_srai_epi16(_mm_add_epi16(V58[i], V60[i]), 1));
        V61[i] = _mm_sub_epi16(V61[i], _mm_srai_epi16(_mm_add_epi16(V60[i], V62[i]), 1));
        V63[i] = _mm_sub_epi16(V63[i], _mm_srai_epi16(_mm_add_epi16(V62[i], V62[i]), 1));
    }

    //pExt[x] += (pExt[x - 1] + pExt[x + 1] + 2) >> 2;
    for (i = 0; i < 8; i++) {
        V00[i] = _mm_add_epi16(V00[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V01[i], V01[i]), mAddOffset2), 2));
        V02[i] = _mm_add_epi16(V02[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V01[i], V03[i]), mAddOffset2), 2));
        V04[i] = _mm_add_epi16(V04[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V03[i], V05[i]), mAddOffset2), 2));
        V06[i] = _mm_add_epi16(V06[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V05[i], V07[i]), mAddOffset2), 2));
        V08[i] = _mm_add_epi16(V08[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V07[i], V09[i]), mAddOffset2), 2));
        V10[i] = _mm_add_epi16(V10[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V09[i], V11[i]), mAddOffset2), 2));
        V12[i] = _mm_add_epi16(V12[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V11[i], V13[i]), mAddOffset2), 2));
        V14[i] = _mm_add_epi16(V14[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V13[i], V15[i]), mAddOffset2), 2));


        V16[i] = _mm_add_epi16(V16[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V15[i], V17[i]), mAddOffset2), 2));
        V18[i] = _mm_add_epi16(V18[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V17[i], V19[i]), mAddOffset2), 2));
        V20[i] = _mm_add_epi16(V20[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V19[i], V21[i]), mAddOffset2), 2));
        V22[i] = _mm_add_epi16(V22[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V21[i], V23[i]), mAddOffset2), 2));
        V24[i] = _mm_add_epi16(V24[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V23[i], V25[i]), mAddOffset2), 2));
        V26[i] = _mm_add_epi16(V26[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V25[i], V27[i]), mAddOffset2), 2));
        V28[i] = _mm_add_epi16(V28[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V27[i], V29[i]), mAddOffset2), 2));
        V30[i] = _mm_add_epi16(V30[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V29[i], V31[i]), mAddOffset2), 2));

        V32[i] = _mm_add_epi16(V32[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V31[i], V33[i]), mAddOffset2), 2));
        V34[i] = _mm_add_epi16(V34[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V33[i], V35[i]), mAddOffset2), 2));
        V36[i] = _mm_add_epi16(V36[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V35[i], V37[i]), mAddOffset2), 2));
        V38[i] = _mm_add_epi16(V38[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V37[i], V39[i]), mAddOffset2), 2));
        V40[i] = _mm_add_epi16(V40[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V39[i], V41[i]), mAddOffset2), 2));
        V42[i] = _mm_add_epi16(V42[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V41[i], V43[i]), mAddOffset2), 2));
        V44[i] = _mm_add_epi16(V44[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V43[i], V45[i]), mAddOffset2), 2));
        V46[i] = _mm_add_epi16(V46[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V45[i], V47[i]), mAddOffset2), 2));

        V48[i] = _mm_add_epi16(V48[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V47[i], V49[i]), mAddOffset2), 2));
        V50[i] = _mm_add_epi16(V50[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V49[i], V51[i]), mAddOffset2), 2));
        V52[i] = _mm_add_epi16(V52[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V51[i], V53[i]), mAddOffset2), 2));
        V54[i] = _mm_add_epi16(V54[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V53[i], V55[i]), mAddOffset2), 2));
        V56[i] = _mm_add_epi16(V56[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V55[i], V57[i]), mAddOffset2), 2));
        V58[i] = _mm_add_epi16(V58[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V57[i], V59[i]), mAddOffset2), 2));
        V60[i] = _mm_add_epi16(V60[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V59[i], V61[i]), mAddOffset2), 2));
        V62[i] = _mm_add_epi16(V62[i], _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(V61[i], V63[i]), mAddOffset2), 2));
    }
    //1-15
    TRANSPOSE_16x16_16BIT(
        V00[0], V02[0], V04[0], V06[0], V08[0], V10[0], V12[0], V14[0], V16[0], V18[0], V20[0], V22[0], V24[0], V26[0], V28[0], V30[0],
        V00[1], V02[1], V04[1], V06[1], V08[1], V10[1], V12[1], V14[1], V16[1], V18[1], V20[1], V22[1], V24[1], V26[1], V28[1], V30[1],
        A00[0], A01[0], A02[0], A03[0], A04[0], A05[0], A06[0], A07[0], A08[0], A09[0], A10[0], A11[0], A12[0], A13[0], A14[0], A15[0],
        A00[1], A01[1], A02[1], A03[1], A04[1], A05[1], A06[1], A07[1], A08[1], A09[1], A10[1], A11[1], A12[1], A13[1], A14[1], A15[1]
    );
    TRANSPOSE_16x16_16BIT(
        V00[2], V02[2], V04[2], V06[2], V08[2], V10[2], V12[2], V14[2], V16[2], V18[2], V20[2], V22[2], V24[2], V26[2], V28[2], V30[2],
        V00[3], V02[3], V04[3], V06[3], V08[3], V10[3], V12[3], V14[3], V16[3], V18[3], V20[3], V22[3], V24[3], V26[3], V28[3], V30[3],
        A16[0], A17[0], A18[0], A19[0], A20[0], A21[0], A22[0], A23[0], A24[0], A25[0], A26[0], A27[0], A28[0], A29[0], A30[0], A31[0],
        A16[1], A17[1], A18[1], A19[1], A20[1], A21[1], A22[1], A23[1], A24[1], A25[1], A26[1], A27[1], A28[1], A29[1], A30[1], A31[1]
    );
    TRANSPOSE_16x16_16BIT(
        V00[4], V02[4], V04[4], V06[4], V08[4], V10[4], V12[4], V14[4], V16[4], V18[4], V20[4], V22[4], V24[4], V26[4], V28[4], V30[4],
        V00[5], V02[5], V04[5], V06[5], V08[5], V10[5], V12[5], V14[5], V16[5], V18[5], V20[5], V22[5], V24[5], V26[5], V28[5], V30[5],
        A32[0], A33[0], A34[0], A35[0], A36[0], A37[0], A38[0], A39[0], A40[0], A41[0], A42[0], A43[0], A44[0], A45[0], A46[0], A47[0],
        A32[1], A33[1], A34[1], A35[1], A36[1], A37[1], A38[1], A39[1], A40[1], A41[1], A42[1], A43[1], A44[1], A45[1], A46[1], A47[1]
    );
    TRANSPOSE_16x16_16BIT(
        V00[6], V02[6], V04[6], V06[6], V08[6], V10[6], V12[6], V14[6], V16[6], V18[6], V20[6], V22[6], V24[6], V26[6], V28[6], V30[6],
        V00[7], V02[7], V04[7], V06[7], V08[7], V10[7], V12[7], V14[7], V16[7], V18[7], V20[7], V22[7], V24[7], V26[7], V28[7], V30[7],
        A48[0], A49[0], A50[0], A51[0], A52[0], A53[0], A54[0], A55[0], A56[0], A57[0], A58[0], A59[0], A60[0], A61[0], A62[0], A63[0],
        A48[1], A49[1], A50[1], A51[1], A52[1], A53[1], A54[1], A55[1], A56[1], A57[1], A58[1], A59[1], A60[1], A61[1], A62[1], A63[1]
    );
    //16-31ï¿½ï¿½
    TRANSPOSE_16x16_16BIT(
        V32[0], V34[0], V36[0], V38[0], V40[0], V42[0], V44[0], V46[0], V48[0], V50[0], V52[0], V54[0], V56[0], V58[0], V60[0], V62[0],
        V32[1], V34[1], V36[1], V38[1], V40[1], V42[1], V44[1], V46[1], V48[1], V50[1], V52[1], V54[1], V56[1], V58[1], V60[1], V62[1],
        A00[2], A01[2], A02[2], A03[2], A04[2], A05[2], A06[2], A07[2], A08[2], A09[2], A10[2], A11[2], A12[2], A13[2], A14[2], A15[2],
        A00[3], A01[3], A02[3], A03[3], A04[3], A05[3], A06[3], A07[3], A08[3], A09[3], A10[3], A11[3], A12[3], A13[3], A14[3], A15[3]
    );
    TRANSPOSE_16x16_16BIT(
        V32[2], V34[2], V36[2], V38[2], V40[2], V42[2], V44[2], V46[2], V48[2], V50[2], V52[2], V54[2], V56[2], V58[2], V60[2], V62[2],
        V32[3], V34[3], V36[3], V38[3], V40[3], V42[3], V44[3], V46[3], V48[3], V50[3], V52[3], V54[3], V56[3], V58[3], V60[3], V62[3],
        A16[2], A17[2], A18[2], A19[2], A20[2], A21[2], A22[2], A23[2], A24[2], A25[2], A26[2], A27[2], A28[2], A29[2], A30[2], A31[2],
        A16[3], A17[3], A18[3], A19[3], A20[3], A21[3], A22[3], A23[3], A24[3], A25[3], A26[3], A27[3], A28[3], A29[3], A30[3], A31[3]
    );
    TRANSPOSE_16x16_16BIT(
        V32[4], V34[4], V36[4], V38[4], V40[4], V42[4], V44[4], V46[4], V48[4], V50[4], V52[4], V54[4], V56[4], V58[4], V60[4], V62[4],
        V32[5], V34[5], V36[5], V38[5], V40[5], V42[5], V44[5], V46[5], V48[5], V50[5], V52[5], V54[5], V56[5], V58[5], V60[5], V62[5],
        A32[2], A33[2], A34[2], A35[2], A36[2], A37[2], A38[2], A39[2], A40[2], A41[2], A42[2], A43[2], A44[2], A45[2], A46[2], A47[2],
        A32[3], A33[3], A34[3], A35[3], A36[3], A37[3], A38[3], A39[3], A40[3], A41[3], A42[3], A43[3], A44[3], A45[3], A46[3], A47[3]
    );
    TRANSPOSE_16x16_16BIT(
        V32[6], V34[6], V36[6], V38[6], V40[6], V42[6], V44[6], V46[6], V48[6], V50[6], V52[6], V54[6], V56[6], V58[6], V60[6], V62[6],
        V32[7], V34[7], V36[7], V38[7], V40[7], V42[7], V44[7], V46[7], V48[7], V50[7], V52[7], V54[7], V56[7], V58[7], V60[7], V62[7],
        A48[2], A49[2], A50[2], A51[2], A52[2], A53[2], A54[2], A55[2], A56[2], A57[2], A58[2], A59[2], A60[2], A61[2], A62[2], A63[2],
        A48[3], A49[3], A50[3], A51[3], A52[3], A53[3], A54[3], A55[3], A56[3], A57[3], A58[3], A59[3], A60[3], A61[3], A62[3], A63[3]
    );

    //pExt[y] -= (pExt[y - 1] + pExt[y + 1]) >> 1;
    for (i = 0; i < 4; i++) {
        A01[i] = _mm_sub_epi16(A01[i], _mm_srai_epi16(_mm_add_epi16(A00[i], A02[i]), 1));
        A03[i] = _mm_sub_epi16(A03[i], _mm_srai_epi16(_mm_add_epi16(A02[i], A04[i]), 1));
        A05[i] = _mm_sub_epi16(A05[i], _mm_srai_epi16(_mm_add_epi16(A04[i], A06[i]), 1));
        A07[i] = _mm_sub_epi16(A07[i], _mm_srai_epi16(_mm_add_epi16(A06[i], A08[i]), 1));
        A09[i] = _mm_sub_epi16(A09[i], _mm_srai_epi16(_mm_add_epi16(A08[i], A10[i]), 1));
        A11[i] = _mm_sub_epi16(A11[i], _mm_srai_epi16(_mm_add_epi16(A10[i], A12[i]), 1));
        A13[i] = _mm_sub_epi16(A13[i], _mm_srai_epi16(_mm_add_epi16(A12[i], A14[i]), 1));
        A15[i] = _mm_sub_epi16(A15[i], _mm_srai_epi16(_mm_add_epi16(A14[i], A16[i]), 1));

        A17[i] = _mm_sub_epi16(A17[i], _mm_srai_epi16(_mm_add_epi16(A16[i], A18[i]), 1));
        A19[i] = _mm_sub_epi16(A19[i], _mm_srai_epi16(_mm_add_epi16(A18[i], A20[i]), 1));
        A21[i] = _mm_sub_epi16(A21[i], _mm_srai_epi16(_mm_add_epi16(A20[i], A22[i]), 1));
        A23[i] = _mm_sub_epi16(A23[i], _mm_srai_epi16(_mm_add_epi16(A22[i], A24[i]), 1));
        A25[i] = _mm_sub_epi16(A25[i], _mm_srai_epi16(_mm_add_epi16(A24[i], A26[i]), 1));
        A27[i] = _mm_sub_epi16(A27[i], _mm_srai_epi16(_mm_add_epi16(A26[i], A28[i]), 1));
        A29[i] = _mm_sub_epi16(A29[i], _mm_srai_epi16(_mm_add_epi16(A28[i], A30[i]), 1));
        A31[i] = _mm_sub_epi16(A31[i], _mm_srai_epi16(_mm_add_epi16(A30[i], A32[i]), 1));

        A33[i] = _mm_sub_epi16(A33[i], _mm_srai_epi16(_mm_add_epi16(A32[i], A34[i]), 1));
        A35[i] = _mm_sub_epi16(A35[i], _mm_srai_epi16(_mm_add_epi16(A34[i], A36[i]), 1));
        A37[i] = _mm_sub_epi16(A37[i], _mm_srai_epi16(_mm_add_epi16(A36[i], A38[i]), 1));
        A39[i] = _mm_sub_epi16(A39[i], _mm_srai_epi16(_mm_add_epi16(A38[i], A40[i]), 1));
        A41[i] = _mm_sub_epi16(A41[i], _mm_srai_epi16(_mm_add_epi16(A40[i], A42[i]), 1));
        A43[i] = _mm_sub_epi16(A43[i], _mm_srai_epi16(_mm_add_epi16(A42[i], A44[i]), 1));
        A45[i] = _mm_sub_epi16(A45[i], _mm_srai_epi16(_mm_add_epi16(A44[i], A46[i]), 1));
        A47[i] = _mm_sub_epi16(A47[i], _mm_srai_epi16(_mm_add_epi16(A46[i], A48[i]), 1));

        A49[i] = _mm_sub_epi16(A49[i], _mm_srai_epi16(_mm_add_epi16(A48[i], A50[i]), 1));
        A51[i] = _mm_sub_epi16(A51[i], _mm_srai_epi16(_mm_add_epi16(A50[i], A52[i]), 1));
        A53[i] = _mm_sub_epi16(A53[i], _mm_srai_epi16(_mm_add_epi16(A52[i], A54[i]), 1));
        A55[i] = _mm_sub_epi16(A55[i], _mm_srai_epi16(_mm_add_epi16(A54[i], A56[i]), 1));
        A57[i] = _mm_sub_epi16(A57[i], _mm_srai_epi16(_mm_add_epi16(A56[i], A58[i]), 1));
        A59[i] = _mm_sub_epi16(A59[i], _mm_srai_epi16(_mm_add_epi16(A58[i], A60[i]), 1));
        A61[i] = _mm_sub_epi16(A61[i], _mm_srai_epi16(_mm_add_epi16(A60[i], A62[i]), 1));
        A63[i] = _mm_sub_epi16(A63[i], _mm_srai_epi16(_mm_add_epi16(A62[i], A62[i]), 1));
    }

    //pExt[y] = (pExt[y] << 1) + ((pExt[y - 1] + pExt[y + 1] + 1) >> 1);
    for (i = 0; i < 4; i++) {
        A00[i] = _mm_add_epi16(_mm_slli_epi16(A00[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A01[i], A01[i]), mAddOffset1), 1));
        A02[i] = _mm_add_epi16(_mm_slli_epi16(A02[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A01[i], A03[i]), mAddOffset1), 1));
        A04[i] = _mm_add_epi16(_mm_slli_epi16(A04[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A03[i], A05[i]), mAddOffset1), 1));
        A06[i] = _mm_add_epi16(_mm_slli_epi16(A06[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A05[i], A07[i]), mAddOffset1), 1));
        A08[i] = _mm_add_epi16(_mm_slli_epi16(A08[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A07[i], A09[i]), mAddOffset1), 1));
        A10[i] = _mm_add_epi16(_mm_slli_epi16(A10[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A09[i], A11[i]), mAddOffset1), 1));
        A12[i] = _mm_add_epi16(_mm_slli_epi16(A12[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A11[i], A13[i]), mAddOffset1), 1));
        A14[i] = _mm_add_epi16(_mm_slli_epi16(A14[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A13[i], A15[i]), mAddOffset1), 1));

        A16[i] = _mm_add_epi16(_mm_slli_epi16(A16[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A15[i], A17[i]), mAddOffset1), 1));
        A18[i] = _mm_add_epi16(_mm_slli_epi16(A18[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A17[i], A19[i]), mAddOffset1), 1));
        A20[i] = _mm_add_epi16(_mm_slli_epi16(A20[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A19[i], A21[i]), mAddOffset1), 1));
        A22[i] = _mm_add_epi16(_mm_slli_epi16(A22[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A21[i], A23[i]), mAddOffset1), 1));
        A24[i] = _mm_add_epi16(_mm_slli_epi16(A24[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A23[i], A25[i]), mAddOffset1), 1));
        A26[i] = _mm_add_epi16(_mm_slli_epi16(A26[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A25[i], A27[i]), mAddOffset1), 1));
        A28[i] = _mm_add_epi16(_mm_slli_epi16(A28[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A27[i], A29[i]), mAddOffset1), 1));
        A30[i] = _mm_add_epi16(_mm_slli_epi16(A30[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A29[i], A31[i]), mAddOffset1), 1));

        A32[i] = _mm_add_epi16(_mm_slli_epi16(A32[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A31[i], A33[i]), mAddOffset1), 1));
        A34[i] = _mm_add_epi16(_mm_slli_epi16(A34[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A33[i], A35[i]), mAddOffset1), 1));
        A36[i] = _mm_add_epi16(_mm_slli_epi16(A36[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A35[i], A37[i]), mAddOffset1), 1));
        A38[i] = _mm_add_epi16(_mm_slli_epi16(A38[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A37[i], A39[i]), mAddOffset1), 1));
        A40[i] = _mm_add_epi16(_mm_slli_epi16(A40[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A39[i], A41[i]), mAddOffset1), 1));
        A42[i] = _mm_add_epi16(_mm_slli_epi16(A42[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A41[i], A43[i]), mAddOffset1), 1));
        A44[i] = _mm_add_epi16(_mm_slli_epi16(A44[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A43[i], A45[i]), mAddOffset1), 1));
        A46[i] = _mm_add_epi16(_mm_slli_epi16(A46[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A45[i], A47[i]), mAddOffset1), 1));

        A48[i] = _mm_add_epi16(_mm_slli_epi16(A48[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A47[i], A49[i]), mAddOffset1), 1));
        A50[i] = _mm_add_epi16(_mm_slli_epi16(A50[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A49[i], A51[i]), mAddOffset1), 1));
        A52[i] = _mm_add_epi16(_mm_slli_epi16(A52[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A51[i], A53[i]), mAddOffset1), 1));
        A54[i] = _mm_add_epi16(_mm_slli_epi16(A54[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A53[i], A55[i]), mAddOffset1), 1));
        A56[i] = _mm_add_epi16(_mm_slli_epi16(A56[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A55[i], A57[i]), mAddOffset1), 1));
        A58[i] = _mm_add_epi16(_mm_slli_epi16(A58[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A57[i], A59[i]), mAddOffset1), 1));
        A60[i] = _mm_add_epi16(_mm_slli_epi16(A60[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A59[i], A61[i]), mAddOffset1), 1));
        A62[i] = _mm_add_epi16(_mm_slli_epi16(A62[i], 1), _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(A61[i], A63[i]), mAddOffset1), 1));
    }

    //STORE
    for (i = 0; i < 4; i++) {
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 0], A00[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 1], A02[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 2], A04[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 3], A06[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 4], A08[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 5], A10[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 6], A12[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 7], A14[i]);

        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 8], A16[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 9], A18[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 10], A20[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 11], A22[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 12], A24[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 13], A26[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 14], A28[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 15], A30[i]);

        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 16], A32[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 17], A34[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 18], A36[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 19], A38[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 20], A40[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 21], A42[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 22], A44[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 23], A46[i]);

        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 24], A48[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 25], A50[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 26], A52[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 27], A54[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 28], A56[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 29], A58[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 30], A60[i]);
        _mm_store_si128((__m128i*)&coeff[8 * i + 32 * 31], A62[i]);
    }
}

/* ---------------------------------------------------------------------------
 */
void dct_c_64x64_sse128(const coeff_t *src, coeff_t *dst, int i_src)
{
    UNUSED_PARAMETER(src);
    UNUSED_PARAMETER(i_src);
    wavelet_64x64_sse128(dst);
    dct_c_32x32_sse128(dst, dst, 32 | 1);
}

/* ---------------------------------------------------------------------------
 */
void dct_c_64x64_half_sse128(const coeff_t *src, coeff_t *dst, int i_src)
{
    UNUSED_PARAMETER(src);
    UNUSED_PARAMETER(i_src);
    wavelet_64x64_sse128(dst);
    dct_c_32x32_half_sse128(dst, dst, 32 | 1);
}

/* ---------------------------------------------------------------------------
 */
void dct_c_64x16_sse128(const coeff_t *src, coeff_t *dst, int i_src)
{
    UNUSED_PARAMETER(src);
    UNUSED_PARAMETER(i_src);
    wavelet_64x16_sse128(dst);
    dct_c_32x8_sse128(dst, dst, 32 | 0x01);
}

/* ---------------------------------------------------------------------------
 */
void dct_c_16x64_sse128(const coeff_t *src, coeff_t *dst, int i_src)
{
    UNUSED_PARAMETER(src);
    UNUSED_PARAMETER(i_src);
    wavelet_16x64_sse128(dst);
    dct_c_8x32_sse128(dst, dst, 8 | 0x01);
}
