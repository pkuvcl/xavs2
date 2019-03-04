/*
 * avs2_defs.h
 *
 * Description of this file:
 *    Struct definition of the xavs2 library
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


#ifndef XAVS2_AVS2_DEFINITIONS_H
#define XAVS2_AVS2_DEFINITIONS_H

#include <assert.h>
#include <stdint.h>
#include "defines.h"
#include "osdep.h"
#include "basic_types.h"
#if (ARCH_X86 || ARCH_X86_64)
#include <xmmintrin.h>
#endif

/* ---------------------------------------------------------------------------
 */
enum intra_avail_e {
    MD_I_LEFT      = 0,
    MD_I_TOP       = 1,
    MD_I_LEFT_DOWN = 2,
    MD_I_TOP_RIGHT = 3,
    MD_I_TOP_LEFT  = 4,
    MD_I_NUM       = 5,
#define IS_NEIGHBOR_AVAIL(i_avai, md)    ((i_avai) & (1 << (md)))
};

enum transform_scan_direction_e {
    INTRA_PRED_VER = 0,
    INTRA_PRED_HOR,
    INTRA_PRED_DC_DIAG
};

/* ---------------------------------------------------------------------------
 * luma intra prediction modes
 */
enum intra_pred_mode_e {
    /* non-angular mode */
    DC_PRED         = 0 ,                /* prediction mode: DC */
    PLANE_PRED      = 1 ,                /* prediction mode: PLANE */
    BI_PRED         = 2 ,                /* prediction mode: BI */

    /* vertical angular mode */
    INTRA_ANG_X_3   =  3, INTRA_ANG_X_4   =  4, INTRA_ANG_X_5   =  5,
    INTRA_ANG_X_6   =  6, INTRA_ANG_X_7   =  7, INTRA_ANG_X_8   =  8,
    INTRA_ANG_X_9   =  9, INTRA_ANG_X_10  = 10, INTRA_ANG_X_11  = 11,
    INTRA_ANG_X_12  = 12,
    VERT_PRED       = INTRA_ANG_X_12,    /* prediction mode: VERT */

    /* vertical + horizontal angular mode */
    INTRA_ANG_XY_13 = 13, INTRA_ANG_XY_14 = 14, INTRA_ANG_XY_15 = 15,
    INTRA_ANG_XY_16 = 16, INTRA_ANG_XY_17 = 17, INTRA_ANG_XY_18 = 18,
    INTRA_ANG_XY_19 = 19, INTRA_ANG_XY_20 = 20, INTRA_ANG_XY_21 = 21,
    INTRA_ANG_XY_22 = 22, INTRA_ANG_XY_23 = 23,

    /* horizontal angular mode */
    INTRA_ANG_Y_24  = 24, INTRA_ANG_Y_25  = 25, INTRA_ANG_Y_26 = 26,
    INTRA_ANG_Y_27  = 27, INTRA_ANG_Y_28  = 28, INTRA_ANG_Y_29 = 29,
    INTRA_ANG_Y_30  = 30, INTRA_ANG_Y_31  = 31, INTRA_ANG_Y_32 = 32,
    HOR_PRED        = INTRA_ANG_Y_24,    /* prediction mode: HOR */
    NUM_INTRA_MODE  = 33,                /* number of luma intra prediction modes */
};

/* ---------------------------------------------------------------------------
 * chroma intra prediction modes
 */
enum intra_chroma_pred_mode_e {
    /* chroma intra prediction modes */
    DM_PRED_C             = 0,     /* prediction mode: DM */
    DC_PRED_C             = 1,     /* prediction mode: DC */
    HOR_PRED_C            = 2,     /* prediction mode: HOR */
    VERT_PRED_C           = 3,     /* prediction mode: VERT */
    BI_PRED_C             = 4,     /* prediction mode: BI */
    NUM_INTRA_MODE_CHROMA = 5,     /* number of chroma intra prediction modes */
};

/* ---------------------------------------------------------------------------
 */
enum mvp_e {
    MVP_MEDIAN      = 0,        /* mv pred type: median */
    MVP_LEFT        = 1,        /*             : left */
    MVP_TOP         = 2,        /*             : top */
    MVP_TR          = 3         /*             : top-right */
};


/* ---------------------------------------------------------------------------
 */
enum inter_pred_direction_e {
    PDIR_FWD        = 0,        /* pred direction: forward */
    PDIR_BWD        = 1,        /*               : backward */
    PDIR_SYM        = 2,        /*               : symmetric */
    PDIR_BID        = 3,        /*               : bidirectional */
    PDIR_DUAL       = 4,        /*               : dual */
    PDIR_INVALID    =-1         /*               : invalid */
};


/* ---------------------------------------------------------------------------
 * reference index
 */
enum inter_pred_index_e {
    INVALID_REF     = -1,       /* invalid reference index */
    B_BWD           = 0,        /* backward reference index for B frame: h->fref[0], used for ref_idx derivation */
    B_FWD           = 1         /* forward  reference index for B frame: h->fref[1], used for ref_idx derivation */
};


/* ---------------------------------------------------------------------------
 */
enum direct_skip_mode_e {
    DS_NONE         = -1,       /* no spatial direct/skip mode */

    /* spatial direct/skip mode for B frame */
    DS_B_BID        = 0,        /* skip/direct mode: bi-direction */
    DS_B_BWD        = 1,        /*                 : backward direction */
    DS_B_SYM        = 2,        /*                 : symmetrical direction */
    DS_B_FWD        = 3,        /*                 : forward direction */

    /* spatial direct/skip mode for F frame */
    DS_DUAL_1ST     = 0,        /* skip/direct mode: dual 1st */
    DS_DUAL_2ND     = 1,        /*                 : dual 2nd */
    DS_SINGLE_1ST   = 2,        /*                 : single 1st */
    DS_SINGLE_2ND   = 3,        /*                 : single 2st */

    /* max number */
    DS_MAX_NUM      = 4         /* max spatial direct/skip mode number of B or F frames */
};


/* ---------------------------------------------------------------------------
 * neighbor position used in inter coding (MVP) or intra prediction
 */
enum neighbor_block_pos_e {
    BLK_TOPLEFT     = 0,        /* D: top-left   block: (x     - 1, y     - 1) */
    BLK_TOP         = 1,        /* B: top        block: (x        , y     - 1) */
    BLK_LEFT        = 2,        /* A: left       block: (x     - 1, y        ) */
    BLK_TOPRIGHT    = 3,        /* C: top-right  block: (x + W    , y     - 1) */
    BLK_TOP2        = 4,        /* G: top        block: (x + W - 1, y     - 1) */
    BLK_LEFT2       = 5,        /* F: left       block: (x     - 1, y + H - 1) */
    BLK_COL         = 6,        /* Z: collocated block of temporal neighbor    */
};

/* ---------------------------------------------------------------------------
 * level for RDO
 */
enum rdo_level_e {
    RDO_OFF       = 0,          /* disable RDO */
    RDO_CU_LEVEL1 = 1,          /* conduct RDO only for best 1 partition mode of CU */
    RDO_CU_LEVEL2 = 2,          /* conduct RDO only for best 2 partition mode of CU,
                                 * including 1 skip/direct mode and 1 normal partition mode */
    RDO_ALL       = 3           /* conduct for all partition modes */
};


/* ---------------------------------------------------------------------------
 * level for RDOQ
 */
enum rdoq_level_e {
    RDOQ_OFF      = 0,          /* disable RDOQ */
    RDOQ_CU_LEVEL = 1,          /* conduct RDOQ only for best partition mode of CU */
    RDOQ_ALL      = 2           /* conduct for all modes */
};

/* ---------------------------------------------------------------------------
 */
enum sao_component_index_e {
    SAO_Y           = 0,
    SAO_Cb,
    SAO_Cr,
    NUM_SAO_COMPONENTS
};


/* ---------------------------------------------------------------------------
 */
enum sao_mode_merge_type_e {
    SAO_MERGE_LEFT      = 0,
    SAO_MERGE_ABOVE,
    SAO_MERGE_NONE,
    NUM_SAO_MERGE_TYPES = 2
};


/* ---------------------------------------------------------------------------
 */
enum sao_mode_type_e {
    SAO_TYPE_OFF        = -1,
    SAO_TYPE_EO_0,
    SAO_TYPE_EO_90,
    SAO_TYPE_EO_135,
    SAO_TYPE_EO_45,
    SAO_TYPE_BO,
    NUM_SAO_NEW_TYPES
};


/* ---------------------------------------------------------------------------
 * EO Groups, the assignments depended on how you implement the edgeType calculation
 */
enum sao_class_e {
    SAO_CLASS_EO_FULL_VALLEY = 0,
    SAO_CLASS_EO_HALF_VALLEY = 1,
    SAO_CLASS_EO_PLAIN       = 2,
    SAO_CLASS_EO_HALF_PEAK   = 3,
    SAO_CLASS_EO_FULL_PEAK   = 4,
    SAO_CLASS_BO             = 5,
    NUM_SAO_EO_CLASSES       = SAO_CLASS_BO,
    NUM_SAO_OFFSET
};



/*
 * ===========================================================================
 * macros
 * ===========================================================================
 */

#define XAVS2_MIN(a, b)       ((a) < (b)? (a) : (b))
#define XAVS2_MAX(a, b)       ((a) > (b)? (a) : (b))
#define XAVS2_MIN3(a, b, c)   XAVS2_MIN((a), XAVS2_MIN((b),(c)))
#define XAVS2_MAX3(a, b, c)   XAVS2_MAX((a), XAVS2_MAX((b),(c)))

#define XAVS2_CLIP1(a)        ((a) > max_pel_value ? max_pel_value : ((a) < 0 ? 0 : (a)))
#define XAVS2_CLIP3F(L, H, v) (((v) < (L)) ? (L) : (((v) > (H)) ? (H) : (v)))
#define XAVS2_CLIP3(L, H, v)  xavs2_clip3(L, H, v)
#define XAVS2_ABS(A)          ((A) < 0 ? (-(A)) : (A))    // abs macro, faster than procedure

#define XAVS2_SWAP(x, y)      {(y) = (y) ^ (x); (x) = (y) ^ (x); (y) = (x) ^ (y);}
#ifdef __cplusplus
template <class T>
static void XAVS2_SWAP_PTR(T *&x, T *&y)
{
    T *t = x;
    x = y;
    y = x;
}
#else
#define XAVS2_SWAP_PTR(x, y)  {void *_t = (void *)(x); (x) = (y); (y) = _t;}
#endif
#define XAVS2_ALIGN(x, a)     (((x) + ((a) - 1)) & (~((a) - 1)))
#define MAKEDWORD(mx, my)     (((my) << 16) | ((mx) & 0xFFFF))

/**
 * ===========================================================================
 * global variables
 * ===========================================================================
 */
static const int g_bit_depth   = BIT_DEPTH;
static const int max_pel_value = (1 << BIT_DEPTH) - 1;
static const int g_dc_value    = (1 << BIT_DEPTH) >> 1;

/**
 * ===========================================================================
 * inline function defines
 * ===========================================================================
 */

static ALWAYS_INLINE pel_t xavs2_clip_pixel(int x)
{
    return (pel_t)((x & ~max_pel_value) ? (-x) >> 31 & max_pel_value : x);
}

static ALWAYS_INLINE int xavs2_clip3(int i_min, int i_max, int v)
{
    return ((v < i_min) ? i_min : (v > i_max) ? i_max : v);
}

static ALWAYS_INLINE double xavs2_clip3f(double f_min, double f_max, double v)
{
    return ((v < f_min) ? f_min : (v > f_max) ? f_max : v);
}

static ALWAYS_INLINE float xavs2_clip3ff(float f_min, float f_max, float v)
{
    return ((v < f_min) ? f_min : (v > f_max) ? f_max : v);
}

static ALWAYS_INLINE int xavs2_median(int a, int b, int c)
{
    int t = (a - b) & ((a - b) >> 31);

    a -= t;
    b += t;
    b -= (b - c) & ((b - c) >> 31);
    b += (a - b) & ((a - b) >> 31);

    return b;
}

// 返回数值的符号位，负数返回-1，否则返回1
static ALWAYS_INLINE int xavs2_sign2(int val)
{
    return ((val >> 31) << 1) + 1;
}

// 返回数值的符号位，负数返回-1，0值返回0，正数返回1
static ALWAYS_INLINE int xavs2_sign3(int val)
{
    return (val >> 31) | (int)(((uint32_t)-val) >> 31u);
}

// 计算正整数的log2值，0和1时返回0，其他返回log2(val)
#define xavs2_log2u(val)  xavs2_ctz(val)


/* ---------------------------------------------------------------------------
* unions for type-punning.
* Mn: load or store n bits, aligned, native-endian
* CPn: copy n bits, aligned, native-endian
* we don't use memcpy for CPn because memcpy's args aren't assumed
* to be aligned */
typedef union {
    uint16_t    i;
    uint8_t     c[2];
} MAY_ALIAS xavs2_union16_t;

typedef union {
    uint32_t    i;
    uint16_t    b[2];
    uint8_t     c[4];
} MAY_ALIAS xavs2_union32_t;

typedef union {
    uint64_t    i;
    uint32_t    a[2];
    uint16_t    b[4];
    uint8_t     c[8];
} MAY_ALIAS xavs2_union64_t;

#define M16(src)                (((xavs2_union16_t *)(src))->i)
#define M32(src)                (((xavs2_union32_t *)(src))->i)
#define M64(src)                (((xavs2_union64_t *)(src))->i)
#define CP16(dst,src)           M16(dst)  = M16(src)
#define CP32(dst,src)           M32(dst)  = M32(src)
#define CP64(dst,src)           M64(dst)  = M64(src)
#define CP128(dst,src)          M128(dst) = M128(src)

#if defined(_MSC_VER) || defined(__ICL)
#define M128(src)               (*(__m128*)(src))
#define M128_ZERO               _mm_setzero_ps()
#else

typedef struct {
    uint64_t    i[2];
} xavs2_uint128_t;

typedef union {
    xavs2_uint128_t i;
    uint64_t    a[2];
    uint32_t    b[4];
    uint16_t    c[8];
    uint8_t     d[16];
} MAY_ALIAS xavs2_union128_t;

#define M128(src)               (((xavs2_union128_t*)(src))->i)

#if (ARCH_X86 || ARCH_X86_64) && defined(__SSE__)
#define M128_ZERO ((__m128){0,0,0,0})
#define xavs2_union128_t xavs2_union128_sse_t
typedef union {
    __m128      i;
    uint64_t    a[2];
    uint32_t    b[4];
    uint16_t    c[8];
    uint8_t     d[16];
} MAY_ALIAS xavs2_union128_sse_t;
#else

#define M128_ZERO               ((xavs2_uint128_t){{0,0}})
#endif // (ARCH_X86 || ARCH_X86_64) && defined(__SSE__)
#endif // defined(_MSC_VER) || defined(__ICL)


#endif  // XAVS2_BASIC_TYPES_H
