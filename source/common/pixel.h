/*
 * pixel.h
 *
 * Description of this file:
 *    Pixel processing functions definition of the xavs2 library
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

#ifndef XAVS2_PIXEL_H
#define XAVS2_PIXEL_H


/**
 * ===========================================================================
 * type defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * Luma PU partition
 */
enum LumaPU {
    /* square (the first 5 PUs match the block sizes) */
    LUMA_4x4,   LUMA_8x8, LUMA_16x16, LUMA_32x32, LUMA_64x64,
    /* rectangular */
    LUMA_8x4,   LUMA_4x8,
    LUMA_16x8,  LUMA_8x16,
    LUMA_32x16, LUMA_16x32,
    LUMA_64x32, LUMA_32x64,
    /* asymmetrical (0.75, 0.25) */
    LUMA_16x12, LUMA_12x16, LUMA_16x4,  LUMA_4x16,
    LUMA_32x24, LUMA_24x32, LUMA_32x8,  LUMA_8x32,
    LUMA_64x48, LUMA_48x64, LUMA_64x16, LUMA_16x64,
    /* number */
    NUM_PU_SIZES,                /* total number of PU sizes */
    LUMA_INVALID = 255
};

/* ---------------------------------------------------------------------------
 * Luma CU sizes, can be indexed using log2n(width)-2
 */
enum LumaCU {
    BLOCK_4x4,
    BLOCK_8x8,
    BLOCK_16x16,
    BLOCK_32x32,
    BLOCK_64x64,
    NUM_CU_SIZES                /* total number of CU sizes */
};

/* ---------------------------------------------------------------------------
 * TU sizes
 */
enum TransUnit {
    /* square */
    TU_4x4, TU_8x8, TU_16x16, TU_32x32, TU_64x64,
    /* asymmetrical */
    TU_16x4,  TU_4x16,
    TU_32x8,  TU_8x32,
    TU_64x16, TU_16x64,
    /* number */
    NUM_TU_SIZES                /* total number of TU sizes */
};

/* ---------------------------------------------------------------------------
 * Chroma (only for 4:2:0) partition sizes.
 * These enum are only a convenience for indexing into the chroma primitive
 * arrays when instantiating macros or templates. The chroma function tables
 * should always be indexed by a LumaPU enum when used.
 */
enum ChromaPU {
    /* square */
    CHROMA_2x2, CHROMA_4x4, CHROMA_8x8, CHROMA_16x16, CHROMA_32x32,
    /* rectangular */
    CHROMA_4x2,   CHROMA_2x4,
    CHROMA_8x4,   CHROMA_4x8,
    CHROMA_16x8,  CHROMA_8x16,
    CHROMA_32x16, CHROMA_16x32,
    /* asymmetrical (0.75, 0.25) */
    CHROMA_8x6,   CHROMA_6x8,   CHROMA_8x2,  CHROMA_2x8,
    CHROMA_16x12, CHROMA_12x16, CHROMA_16x4, CHROMA_4x16,
    CHROMA_32x24, CHROMA_24x32, CHROMA_32x8, CHROMA_8x32,
};

/* ---------------------------------------------------------------------------
 */
enum ChromaCU {
    BLOCK_C_2x2,
    BLOCK_C_4x4,
    BLOCK_C_8x8,
    BLOCK_C_16x16,
    BLOCK_C_32x32
};


typedef cmp_dist_t(*pixel_cmp_t)(const pel_t *pix1, intptr_t i_pix1, const pel_t *pix2, intptr_t i_pix2);
typedef dist_t(*pixel_ssd_t)(const pel_t *pix1, intptr_t i_pix1, const pel_t *pix2, intptr_t i_pix2);
typedef dist_t(*pixel_ssd2_t)(const pel_t *pix1, intptr_t i_pix1, const pel_t *pix2, intptr_t i_pix2, int width, int height);
typedef void(*pixel_cmp_x3_t)(const pel_t *fenc, const pel_t *pix0, const pel_t *pix1, const pel_t *pix2,                    intptr_t i_stride, int scores[3]);
typedef void(*pixel_cmp_x4_t)(const pel_t *fenc, const pel_t *pix0, const pel_t *pix1, const pel_t *pix2, const pel_t *pix3, intptr_t i_stride, int scores[4]);

typedef void(*copy_pp_t)(pel_t* dst, intptr_t dstStride, const pel_t* src, intptr_t srcStride); // dst is aligned
typedef void(*copy_sp_t)(pel_t* dst, intptr_t dstStride, const coeff_t* src, intptr_t srcStride);
typedef void(*copy_ps_t)(coeff_t* dst, intptr_t dstStride, const pel_t* src, intptr_t srcStride);
typedef void(*copy_ss_t)(coeff_t* dst, intptr_t dstStride, const coeff_t* src, intptr_t srcStride);

typedef void(*pixel_sub_ps_t)(coeff_t* dst, intptr_t dstride, const pel_t* src0, const pel_t* src1, intptr_t sstride0, intptr_t sstride1);
typedef void(*pixel_add_ps_t)(pel_t* a, intptr_t dstride, const pel_t* b0, const coeff_t* b1, intptr_t sstride0, intptr_t sstride1);
typedef void(*pixel_avg_pp_t)(pel_t* dst, intptr_t dstride, const pel_t* src0, intptr_t sstride0, const pel_t* src1, intptr_t sstride1, int weight);

typedef int(*mad_funcs_t)(pel_t *p_src, int i_src, int cu_size);

typedef struct {

    pixel_cmp_t     sad    [NUM_PU_SIZES];
    pixel_cmp_t     satd   [NUM_PU_SIZES];
    pixel_cmp_t     sa8d   [NUM_PU_SIZES];
    pixel_ssd_t     ssd    [NUM_PU_SIZES];
    pixel_cmp_x3_t  sad_x3 [NUM_PU_SIZES];
    pixel_cmp_x4_t  sad_x4 [NUM_PU_SIZES];

    pixel_sub_ps_t  sub_ps [NUM_PU_SIZES];
    pixel_add_ps_t  add_ps [NUM_PU_SIZES];
    copy_sp_t       copy_sp[NUM_PU_SIZES];
    copy_ps_t       copy_ps[NUM_PU_SIZES];
    copy_ss_t       copy_ss[NUM_PU_SIZES];
    copy_pp_t       copy_pp[NUM_PU_SIZES];
    pixel_avg_pp_t  avg    [NUM_PU_SIZES];

    pixel_cmp_t    *intra_cmp;  /* either satd or sad for intra mode prediction */
    pixel_cmp_t    *fpel_cmp;   /* either satd or sad for fractional pixel comparison in ME */

    mad_funcs_t     madf[CTU_DEPTH];

    pixel_ssd2_t    ssd_block;
    /* block average */
    void (*average)(pel_t *dst, int i_dst, pel_t *src1, int i_src1, pel_t *src2, int i_src2, int width, int height);
} pixel_funcs_t;


/**
 * ===========================================================================
 * global variables
 * ===========================================================================
 */

/* get partition index for the given size */
#define g_partition_map_tab FPFX(g_partition_map_tab)
extern const uint8_t g_partition_map_tab[];
#define PART_INDEX(w, h)    (g_partition_map_tab[((((w) >> 2) - 1) << 4) + ((h) >> 2) - 1])


/**
 * ===========================================================================
 * function declares
 * ===========================================================================
 */

#define xavs2_pixel_init FPFX(pixel_init)
void xavs2_pixel_init(uint32_t cpu, pixel_funcs_t* pixf);

#define xavs2_pixel_ssd_wxh FPFX(xpixel_ssd_wxh)
uint64_t xavs2_pixel_ssd_wxh(pixel_funcs_t *pf,
                             pel_t *p_pix1, intptr_t i_pix1,
                             pel_t *p_pix2, intptr_t i_pix2,
                             int i_width, int i_height,
                             int inout_shift);


#define xavs2_mad_init FPFX(mad_init)
void xavs2_mad_init(uint32_t cpu, mad_funcs_t *madf);

#endif  // XAVS2_PIXEL_H
