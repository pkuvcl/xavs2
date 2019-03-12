/*
 * intrinsic.h
 *
 * Description of this file:
 *    SIMD assembly functions definition of the xavs2 library
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

#ifndef XAVS2_INTRINSIC_H
#define XAVS2_INTRINSIC_H

/* ---------------------------------------------------------------------------
 * macros used for quick access of __m128i
 */
#define M128_U64(mx, idx)  _mm_extract_epi64(mx, idx)
#define M128_U32(mx, idx)  _mm_extract_epi32(mx, idx)
#define M128_I32(mx, idx)  _mm_extract_epi32(mx, idx)
#define M128_U16(mx, idx)  _mm_extract_epi16(mx, idx)
#define M128_I16(mx, idx)  _mm_extract_epi16(mx, idx)


#if _MSC_VER // 解决vs下immintrin.h中没有定义这些函数的问题
#define _mm256_extract_epi64(a, i) (a.m256i_i64[i])
#define _mm256_extract_epi32(a, i) (a.m256i_i32[i])
#define _mm256_extract_epi16(a, i) (a.m256i_i16[i])
#define _mm256_extract_epi8(a, i)  (a.m256i_i8 [i])
#define _mm256_insert_epi64(a, v, i) (a.m256i_i64[i] = v)
#define _mm_extract_epi64(r, i) r.m128i_i64[i]

//insert integrate to dst
#define _mm256_insert_epi64(a, value, index) (a.m256i_i64[index] = value)
#define _mm256_insert_epi32(a, value, index) (a.m256i_i32[index] = value)
#define _mm256_insert_epi16(a, value, index) (a.m256i_i16[index] = value)
#define _mm256_insert_epi8 (a, value, index) (a.m256i_i8 [index] = value)
#else
// 添加部分gcc中缺少的avx函数定义
#define _mm256_set_m128i(/* __m128i */ hi, /* __m128i */ lo) \
            _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 0x1)
#define _mm256_loadu2_m128i(/* __m128i const* */ hiaddr, \
                            /* __m128i const* */ loaddr) \
            _mm256_set_m128i(_mm_loadu_si128(hiaddr), _mm_loadu_si128(loaddr))
#define _mm256_storeu2_m128i(/* __m128i* */ hiaddr, /* __m128i* */ loaddr, \
                             /* __m256i */ a) \
    do { \
        __m256i _a = (a); /* reference a only once in macro body */ \
        _mm_storeu_si128((loaddr), _mm256_castsi256_si128(_a)); \
        _mm_storeu_si128((hiaddr), _mm256_extractf128_si256(_a, 0x1)); \
    } while (0)
#endif

/* ---------------------------------------------------------------------------
 * global variables
 */
ALIGN32(extern const int8_t  intrinsic_mask[15][16]);
ALIGN32(extern const int8_t  intrinsic_mask_256_8bit[16][32]);
ALIGN32(extern const int8_t  intrinsic_mask32[32][32]);
ALIGN32(extern const int16_t intrinsic_mask_10bit[15][16]);
ALIGN32(extern const int8_t tab_log2[65]);
ALIGN16(extern const pel_t tab_coeff_mode_7[64][16]);
ALIGN32(extern const uint8_t tab_idx_mode_7[64]);
ALIGN32(extern const pel_t tab_coeff_mode_7_avx[64][32]);
ALIGN16(extern const int8_t tab_coeff_mode_9[64][16]);

extern const uint8_t tab_idx_mode_9[64];
ALIGN16(extern const int8_t tab_coeff_mode_11[64][16]);

/* ---------------------------------------------------------------------------
 * functions
 */
#define intpl_copy_block_sse128 FPFX(intpl_copy_block_sse128)
void intpl_copy_block_sse128      (pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height);
#define intpl_luma_block_hor_sse128 FPFX(intpl_luma_block_hor_sse128)
void intpl_luma_block_hor_sse128  (pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff);
#define intpl_luma_block_ver_sse128 FPFX(intpl_luma_block_ver_sse128)
void intpl_luma_block_ver_sse128  (pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff);
#define intpl_luma_block_ext_sse128 FPFX(intpl_luma_block_ext_sse128)
void intpl_luma_block_ext_sse128  (pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff_h, const int8_t *coeff_v);

#define intpl_luma_hor_sse128 FPFX(intpl_luma_hor_sse128)
void intpl_luma_hor_sse128(pel_t *dst, int i_dst, mct_t *tmp, int i_tmp, pel_t *src, int i_src, int width, int height, const int8_t *coeff);
#define intpl_luma_ver_sse128 FPFX(intpl_luma_ver_sse128)
void intpl_luma_ver_sse128(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff);
#define intpl_luma_ext_sse128 FPFX(intpl_luma_ext_sse128)
void intpl_luma_ext_sse128(pel_t *dst, int i_dst, mct_t *tmp, int i_tmp, int width, int height, const int8_t *coeff);

#define intpl_luma_hor_avx2 FPFX(intpl_luma_hor_avx2)
void intpl_luma_hor_avx2(pel_t *dst, int i_dst, mct_t *tmp, int i_tmp, pel_t *src, int i_src, int width, int height, int8_t const *coeff);
#define intpl_luma_ver_avx2 FPFX(intpl_luma_ver_avx2)
void intpl_luma_ver_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, int8_t const *coeff);
#define intpl_luma_ext_avx2 FPFX(intpl_luma_ext_avx2)
void intpl_luma_ext_avx2(pel_t *dst, int i_dst, mct_t *tmp, int i_tmp, int width, int height, const int8_t *coeff);

#define intpl_luma_hor_x3_sse128 FPFX(intpl_luma_hor_x3_sse128)
void intpl_luma_hor_x3_sse128(pel_t *const dst[3], int i_dst, mct_t *const tmp[3], int i_tmp, pel_t *src, int i_src, int width, int height, const int8_t **coeff);
#define intpl_luma_ver_x3_sse128 FPFX(intpl_luma_ver_x3_sse128)
void intpl_luma_ver_x3_sse128(pel_t *const dst[3], int i_dst, pel_t *src, int i_src, int width, int height, const int8_t **coeff);
#define intpl_luma_ext_x3_sse128 FPFX(intpl_luma_ext_x3_sse128)
void intpl_luma_ext_x3_sse128(pel_t *const dst[3], int i_dst, mct_t *tmp, int i_tmp, int width, int height, const int8_t **coeff);

#define intpl_chroma_block_hor_sse128 FPFX(intpl_chroma_block_hor_sse128)
void intpl_chroma_block_hor_sse128(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff);
#define intpl_chroma_block_ver_sse128 FPFX(intpl_chroma_block_ver_sse128)
void intpl_chroma_block_ver_sse128(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff);
#define intpl_chroma_block_ext_sse128 FPFX(intpl_chroma_block_ext_sse128)
void intpl_chroma_block_ext_sse128(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff_h, const int8_t *coeff_v);

#define intpl_luma_block_hor_avx2 FPFX(intpl_luma_block_hor_avx2)
void intpl_luma_block_hor_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff);
#define intpl_luma_block_ver_avx2 FPFX(intpl_luma_block_ver_avx2)
void intpl_luma_block_ver_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff);
#define intpl_luma_block_ext_avx2 FPFX(intpl_luma_block_ext_avx2)
void intpl_luma_block_ext_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff_h, const int8_t *coeff_v);

#define intpl_chroma_block_hor_avx2 FPFX(intpl_chroma_block_hor_avx2)
void intpl_chroma_block_hor_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff);
#define intpl_chroma_block_ver_avx2 FPFX(intpl_chroma_block_ver_avx2)
void intpl_chroma_block_ver_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff);
#define intpl_chroma_block_ext_avx2 FPFX(intpl_chroma_block_ext_avx2)
void intpl_chroma_block_ext_avx2(pel_t *dst, int i_dst, pel_t *src, int i_src, int width, int height, const int8_t *coeff_h, const int8_t *coeff_v);

#define intpl_luma_hor_x3_avx2 FPFX(intpl_luma_hor_x3_avx2)
void intpl_luma_hor_x3_avx2(pel_t *const dst[3], int i_dst, mct_t *const tmp[3], int i_tmp, pel_t *src, int i_src, int width, int height, const int8_t **coeff);
#define intpl_luma_ver_x3_avx2 FPFX(intpl_luma_ver_x3_avx2)
void intpl_luma_ver_x3_avx2(pel_t *const dst[3], int i_dst, pel_t *src, int i_src, int width, int height, const int8_t **coeff);
#define intpl_luma_ext_x3_avx2 FPFX(intpl_luma_ext_x3_avx2)
void intpl_luma_ext_x3_avx2(pel_t *const dst[3], int i_dst, mct_t *tmp, int i_tmp, int width, int height, const int8_t **coeff);

/* memory operation */
#define cpy_pel_I420_to_uchar_YUY2_sse128 FPFX(cpy_pel_I420_to_uchar_YUY2_sse128)
void cpy_pel_I420_to_uchar_YUY2_sse128(const pel_t *srcy, const pel_t *srcu, const pel_t *srcv, int i_src, int i_srcc, unsigned char *dst, int i_dst, int width, int height, int bit_size);
#define add_pel_clip_sse128 FPFX(add_pel_clip_sse128)
void add_pel_clip_sse128(const pel_t *src1, int i_src1, const int16_t *src2, int i_src2, pel_t *dst, int i_dst, int width, int height, int bit_depth);
#define xavs2_pixel_average_sse128 FPFX(pixel_average_sse128)
void xavs2_pixel_average_sse128(pel_t *dst, int i_dst, pel_t *src1, int i_src1, pel_t *src2, int i_src2, int width, int height);
#define xavs2_pixel_average_avx FPFX(pixel_average_avx)
void xavs2_pixel_average_avx   (pel_t *dst, int i_dst, pel_t *src1, int i_src1, pel_t *src2, int i_src2, int width, int height);
#define padding_rows_sse128 FPFX(padding_rows_sse128)
void padding_rows_sse128   (pel_t *src, int i_src, int width, int height, int start, int rows, int pad);
#define padding_rows_lr_sse128 FPFX(padding_rows_lr_sse128)
void padding_rows_lr_sse128(pel_t *src, int i_src, int width, int height, int start, int rows, int pad);
#define padding_rows_sse256 FPFX(padding_rows_sse256)
void padding_rows_sse256(pel_t *src, int i_src, int width, int height, int start, int rows, int pad);
#define padding_rows_sse256_10bit FPFX(padding_rows_sse256_10bit)
void padding_rows_sse256_10bit(pel_t *src, int i_src, int width, int height, int start, int rows, int pad);
#define padding_rows_lr_sse256 FPFX(padding_rows_lr_sse256_10bit)
void padding_rows_lr_sse256(pel_t *src, int i_src, int width, int height, int start, int rows, int pad);
#define padding_rows_lr_sse256_10bit FPFX(padding_rows_lr_sse256)
void padding_rows_lr_sse256_10bit(pel_t *src, int i_src, int width, int height, int start, int rows, int pad);

#define xavs2_memzero_aligned_c_sse2 FPFX(memzero_aligned_c_sse2)
void *xavs2_memzero_aligned_c_sse2(void *dst, size_t n);
#define xavs2_memzero_aligned_c_avx FPFX(memzero_aligned_c_avx)
void *xavs2_memzero_aligned_c_avx (void *dst, size_t n);
#define xavs2_mem_repeat_i_c_sse2 FPFX(mem_repeat_i_c_sse2)
void  xavs2_mem_repeat_i_c_sse2   (void *dst, int val, size_t count);
#define xavs2_mem_repeat_i_c_avx FPFX(mem_repeat_i_c_avx)
void  xavs2_mem_repeat_i_c_avx    (void *dst, int val, size_t count);
#define xavs2_memcpy_aligned_c_sse2 FPFX(memcpy_aligned_c_sse2)
void *xavs2_memcpy_aligned_c_sse2 (void *dst, const void *src, size_t n);

#define deblock_edge_ver_sse128 FPFX(deblock_edge_ver_sse128)
void deblock_edge_ver_sse128  (pel_t *SrcPtr, int stride, int Alpha, int Beta, unsigned char *flt_flag);
#define deblock_edge_hor_sse128 FPFX(deblock_edge_hor_sse128)
void deblock_edge_hor_sse128  (pel_t *SrcPtr, int stride, int Alpha, int Beta, unsigned char *flt_flag);
#define deblock_edge_ver_c_sse128 FPFX(deblock_edge_ver_c_sse128)
void deblock_edge_ver_c_sse128(pel_t *SrcPtrU, pel_t *SrcPtrV, int stride, int Alpha, int Beta, unsigned char *flt_flag);
#define deblock_edge_hor_c_sse128 FPFX(deblock_edge_hor_c_sse128)
void deblock_edge_hor_c_sse128(pel_t *SrcPtrU, pel_t *SrcPtrV, int stride, int Alpha, int Beta, unsigned char *flt_flag);

//--------avx2--------    add by zhangjiaqi    2016-12-02
#define deblock_edge_hor_avx2 FPFX(deblock_edge_hor_avx2)
void deblock_edge_hor_avx2(pel_t *SrcPtr, int stride, int Alpha, int Beta, uint8_t *flt_flag);
#define deblock_edge_ver_avx2 FPFX(deblock_edge_ver_avx2)
void deblock_edge_ver_avx2(pel_t *SrcPtr, int stride, int Alpha, int Beta, uint8_t *flt_flag);
#define deblock_edge_hor_c_avx2 FPFX(deblock_edge_hor_c_avx2)
void deblock_edge_hor_c_avx2(pel_t *SrcPtrU, pel_t *SrcPtrV, int stride, int Alpha, int Beta, uint8_t *flt_flag);
#define deblock_edge_ver_c_avx2 FPFX(deblock_edge_ver_c_avx2)
void deblock_edge_ver_c_avx2(pel_t *SrcPtrU, pel_t *SrcPtrV, int stride, int Alpha, int Beta, uint8_t *flt_flag);

#define dct_c_4x4_sse128 FPFX(dct_c_4x4_sse128)
void dct_c_4x4_sse128  (const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_8x8_sse128 FPFX(dct_c_8x8_sse128)
void dct_c_8x8_sse128  (const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_16x16_sse128 FPFX(dct_c_16x16_sse128)
void dct_c_16x16_sse128(const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_32x32_sse128 FPFX(dct_c_32x32_sse128)
void dct_c_32x32_sse128(const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_64x64_sse128 FPFX(dct_c_64x64_sse128)
void dct_c_64x64_sse128(const coeff_t *src, coeff_t *dst, int i_src);

#define dct_c_4x16_sse128 FPFX(dct_c_4x16_sse128)
void dct_c_4x16_sse128 (const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_8x32_sse128 FPFX(dct_c_8x32_sse128)
void dct_c_8x32_sse128 (const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_16x4_sse128 FPFX(dct_c_16x4_sse128)
void dct_c_16x4_sse128 (const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_32x8_sse128 FPFX(dct_c_32x8_sse128)
void dct_c_32x8_sse128 (const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_64x16_sse128 FPFX(dct_c_64x16_sse128)
void dct_c_64x16_sse128(const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_16x64_sse128 FPFX(dct_c_16x64_sse128)
void dct_c_16x64_sse128(const coeff_t *src, coeff_t *dst, int i_src);

//futl
#define dct_c_4x4_avx2 FPFX(dct_c_4x4_avx2)
void dct_c_4x4_avx2(const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_8x8_avx2 FPFX(dct_c_8x8_avx2)
void dct_c_8x8_avx2(const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_4x16_avx2 FPFX(dct_c_4x16_avx2)
void dct_c_4x16_avx2(const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_16x4_avx2 FPFX(dct_c_16x4_avx2)
void dct_c_16x4_avx2(const coeff_t *src, coeff_t *dst, int i_src);

#define dct_c_16x16_avx2 FPFX(dct_c_16x16_avx2)
void dct_c_16x16_avx2(const coeff_t * src, coeff_t * dst, int i_src);
#define dct_c_8x32_avx2 FPFX(dct_c_8x32_avx2)
void dct_c_8x32_avx2(const coeff_t *src, coeff_t *dst, int i_src);
//avx2 function  -zhangjiaqi
#define dct_c_32x32_avx2 FPFX(dct_c_32x32_avx2)
void dct_c_32x32_avx2(const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_32x8_avx2 FPFX(dct_c_32x8_avx2)
void dct_c_32x8_avx2(const coeff_t *src, coeff_t *dst, int i_src);

#define dct_c_64x64_avx2 FPFX(dct_c_64x64_avx2)
void dct_c_64x64_avx2(const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_64x16_avx2 FPFX(dct_c_64x16_avx2)
void dct_c_64x16_avx2(const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_16x64_avx2 FPFX(dct_c_16x64_avx2)
void dct_c_16x64_avx2(const coeff_t *src, coeff_t *dst, int i_src);

/* half DCT, only keep low frequency coefficients */
#define dct_c_32x32_half_sse128 FPFX(dct_c_32x32_half_sse128)
void dct_c_32x32_half_sse128(const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_64x64_half_sse128 FPFX(dct_c_64x64_half_sse128)
void dct_c_64x64_half_sse128(const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_32x32_half_avx2 FPFX(dct_c_32x32_half_avx2)
void dct_c_32x32_half_avx2(const coeff_t *src, coeff_t *dst, int i_src);
#define dct_c_64x64_half_avx2 FPFX(dct_c_64x64_half_avx2)
void dct_c_64x64_half_avx2(const coeff_t *src, coeff_t *dst, int i_src);

#define transform_4x4_2nd_sse128 FPFX(transform_4x4_2nd_sse128)
void transform_4x4_2nd_sse128(coeff_t *coeff, int i_coeff);
#define transform_2nd_sse128 FPFX(transform_2nd_sse128)
void transform_2nd_sse128    (coeff_t *coeff, int i_coeff, int i_mode, int b_top, int b_left);

#define idct_c_4x4_sse128 FPFX(idct_c_4x4_sse128)
void idct_c_4x4_sse128  (const coeff_t *src, coeff_t *dst, int i_dst);
#define idct_c_8x8_sse128 FPFX(idct_c_8x8_sse128)
void idct_c_8x8_sse128  (const coeff_t *src, coeff_t *dst, int i_dst);
#define idct_c_16x16_sse128 FPFX(idct_c_16x16_sse128)
void idct_c_16x16_sse128(const coeff_t *src, coeff_t *dst, int i_dst);
#define idct_c_32x32_sse128 FPFX(idct_c_32x32_sse128)
void idct_c_32x32_sse128(const coeff_t *src, coeff_t *dst, int i_dst);
#define idct_c_64x64_sse128 FPFX(idct_c_64x64_sse128)
void idct_c_64x64_sse128(const coeff_t *src, coeff_t *dst, int i_dst);

#define idct_c_16x4_sse128 FPFX(idct_c_16x4_sse128)
void idct_c_16x4_sse128 (const coeff_t *src, coeff_t *dst, int i_dst);
#define idct_c_32x8_sse128 FPFX(idct_c_32x8_sse128)
void idct_c_32x8_sse128 (const coeff_t *src, coeff_t *dst, int i_dst);
#define idct_c_64x16_sse128 FPFX(idct_c_64x16_sse128)
void idct_c_64x16_sse128(const coeff_t *src, coeff_t *dst, int i_dst);

#define idct_c_4x16_sse128 FPFX(idct_c_4x16_sse128)
void idct_c_4x16_sse128 (const coeff_t *src, coeff_t *dst, int i_dst);
#define idct_c_8x32_sse128 FPFX(idct_c_8x32_sse128)
void idct_c_8x32_sse128 (const coeff_t *src, coeff_t *dst, int i_dst);
#define idct_c_16x64_sse128 FPFX(idct_c_16x64_sse128)
void idct_c_16x64_sse128(const coeff_t *src, coeff_t *dst, int i_dst);

#define inv_transform_4x4_2nd_sse128 FPFX(inv_transform_4x4_2nd_sse128)
void inv_transform_4x4_2nd_sse128(coeff_t *coeff, int i_coeff);
#define inv_transform_2nd_sse128 FPFX(inv_transform_2nd_sse128)
void inv_transform_2nd_sse128    (coeff_t *coeff, int i_coeff, int i_mode, int b_top, int b_left);


//zhangjiaqi add 2016.11.30    avx2
#define idct_c_8x8_avx2 FPFX(idct_c_8x8_avx2)
void idct_c_8x8_avx2  (const coeff_t *src, coeff_t *dst, int i_dst);
#define idct_c_16x16_avx2 FPFX(idct_c_16x16_avx2)
void idct_c_16x16_avx2(const coeff_t *src, coeff_t *dst, int i_dst);
#define idct_c_32x32_avx2 FPFX(idct_c_32x32_avx2)
void idct_c_32x32_avx2(const coeff_t *src, coeff_t *dst, int i_dst);
#define idct_c_64x64_avx2 FPFX(idct_c_64x64_avx2)
void idct_c_64x64_avx2(const coeff_t *src, coeff_t *dst, int i_dst);
#define idct_c_64x16_avx2 FPFX(idct_c_64x16_avx2)
void idct_c_64x16_avx2(const coeff_t *src, coeff_t *dst, int i_dst);
#define idct_c_16x64_avx2 FPFX(idct_c_16x64_avx2)
void idct_c_16x64_avx2(const coeff_t *src, coeff_t *dst, int i_dst);

// scan the cg coefficient
#define coeff_scan_4x4_xy_sse128 FPFX(coeff_scan_4x4_xy_sse128)
void coeff_scan_4x4_xy_sse128(coeff_t *dst, const coeff_t *src, int i_src_shift);
#define coeff_scan_4x4_yx_sse128 FPFX(coeff_scan_4x4_yx_sse128)
void coeff_scan_4x4_yx_sse128(coeff_t *dst, const coeff_t *src, int i_src_shift);

#define coeff_scan4_xy_sse128 FPFX(coeff_scan4_xy_sse128)
void coeff_scan4_xy_sse128(coeff_t *dst, uint64_t r1, uint64_t r2, uint64_t r3, uint64_t r4);
#define coeff_scan4_yx_sse128 FPFX(coeff_scan4_yx_sse128)
void coeff_scan4_yx_sse128(coeff_t *dst, uint64_t r1, uint64_t r2, uint64_t r3, uint64_t r4);
#define coeff_scan4_xy_avx FPFX(coeff_scan4_xy_avx)
void coeff_scan4_xy_avx(coeff_t *dst, uint64_t r1, uint64_t r2, uint64_t r3, uint64_t r4);
#define coeff_scan4_yx_avx FPFX(coeff_scan4_yx_avx)
void coeff_scan4_yx_avx(coeff_t *dst, uint64_t r1, uint64_t r2, uint64_t r3, uint64_t r4);

#define abs_coeff_sse128 FPFX(abs_coeff_sse128)
void abs_coeff_sse128(coeff_t *dst, const coeff_t *src, const int i_coef);
#define add_sign_sse128 FPFX(add_sign_sse128)
int add_sign_sse128(coeff_t *dst, const coeff_t *abs_val, const int i_coef);

#define quant_c_avx2 FPFX(quant_c_avx2)
int quant_c_avx2(coeff_t *coef, const int i_coef, const int scale, const int shift, const int add);
#define dequant_c_avx2 FPFX(dequant_c_avx2)
void dequant_c_avx2(coeff_t *coef, const int i_coef, const int scale, const int shift);
#define quant_c_sse128 FPFX(quant_c_avx2)
int quant_c_sse128(coeff_t *coef, const int i_coef, const int scale, const int shift, const int add);
#define dequant_c_sse128 FPFX(dequant_c_sse128)
void dequant_c_sse128(coeff_t *coef, const int i_coef, const int scale, const int shift, const int add);
#define abs_coeff_avx2 FPFX(abs_coeff_avx2)
void abs_coeff_avx2(coeff_t *dst, const coeff_t *src, const int i_coef);
#define add_sign_avx2 FPFX(add_sign_avx2)
int add_sign_avx2(coeff_t *dst, const coeff_t *abs_val, const int i_coef);

#define SAO_on_block_sse128 FPFX(SAO_on_block_sse128)
void SAO_on_block_sse128(pel_t *p_dst, int i_dst, pel_t *p_src,
                         int i_src, int i_block_w, int i_block_h,
                         int *lcu_avail, SAOBlkParam *sao_param);
#define SAO_on_block_sse256 FPFX(SAO_on_block_sse256)
void SAO_on_block_sse256(pel_t *p_dst, int i_dst, pel_t *p_src,
                         int i_src,int i_block_w, int i_block_h,
                         int *lcu_avail, SAOBlkParam *sao_param);

#define alf_flt_one_block_sse128 FPFX(alf_flt_one_block_sse128)
void alf_flt_one_block_sse128(pel_t *p_dst, int i_dst, pel_t *p_src, int i_src,
                              int lcu_pix_x, int lcu_pix_y, int lcu_width, int lcu_height,
                              int *alf_coeff, int b_top_avail, int b_down_avail);

#define intra_pred_dc_sse128 FPFX(intra_pred_dc_sse128)
void intra_pred_dc_sse128       (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_plane_sse128 FPFX(intra_pred_plane_sse128)
void intra_pred_plane_sse128    (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_bilinear_sse128 FPFX(intra_pred_bilinear_sse128)
void intra_pred_bilinear_sse128 (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_hor_sse128 FPFX(intra_pred_hor_sse128)
void intra_pred_hor_sse128      (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ver_sse128 FPFX(intra_pred_ver_sse128)
void intra_pred_ver_sse128      (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);

#define intra_pred_ang_x_3_sse128 FPFX(intra_pred_ang_x_3_sse128)
void intra_pred_ang_x_3_sse128  (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_x_4_sse128 FPFX(intra_pred_ang_x_4_sse128)
void intra_pred_ang_x_4_sse128  (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_x_5_sse128 FPFX(intra_pred_ang_x_5_sse128)
void intra_pred_ang_x_5_sse128  (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_x_6_sse128 FPFX(intra_pred_ang_x_6_sse128)
void intra_pred_ang_x_6_sse128  (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_x_7_sse128 FPFX(intra_pred_ang_x_7_sse128)
void intra_pred_ang_x_7_sse128  (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_x_8_sse128 FPFX(intra_pred_ang_x_8_sse128)
void intra_pred_ang_x_8_sse128  (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_x_9_sse128 FPFX(intra_pred_ang_x_9_sse128)
void intra_pred_ang_x_9_sse128  (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_x_10_sse128 FPFX(intra_pred_ang_x_10_sse128)
void intra_pred_ang_x_10_sse128 (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_x_11_sse128 FPFX(intra_pred_ang_x_11_sse128)
void intra_pred_ang_x_11_sse128 (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);

#define intra_pred_ang_y_25_sse128 FPFX(intra_pred_ang_y_25_sse128)
void intra_pred_ang_y_25_sse128 (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_y_26_sse128 FPFX(intra_pred_ang_y_26_sse128)
void intra_pred_ang_y_26_sse128 (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_y_28_sse128 FPFX(intra_pred_ang_y_28_sse128)
void intra_pred_ang_y_28_sse128 (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_y_30_sse128 FPFX(intra_pred_ang_y_30_sse128)
void intra_pred_ang_y_30_sse128 (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_y_31_sse128 FPFX(intra_pred_ang_y_31_sse128)
void intra_pred_ang_y_31_sse128 (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_y_32_sse128 FPFX(intra_pred_ang_y_32_sse128)
void intra_pred_ang_y_32_sse128 (pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);

#define intra_pred_ang_xy_13_sse128 FPFX(intra_pred_ang_xy_13_sse128)
void intra_pred_ang_xy_13_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_xy_14_sse128 FPFX(intra_pred_ang_xy_14_sse128)
void intra_pred_ang_xy_14_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_xy_16_sse128 FPFX(intra_pred_ang_xy_16_sse128)
void intra_pred_ang_xy_16_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_xy_18_sse128 FPFX(intra_pred_ang_xy_18_sse128)
void intra_pred_ang_xy_18_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_xy_20_sse128 FPFX(intra_pred_ang_xy_20_sse128)
void intra_pred_ang_xy_20_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_xy_22_sse128 FPFX(intra_pred_ang_xy_22_sse128)
void intra_pred_ang_xy_22_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_xy_23_sse128 FPFX(intra_pred_ang_xy_23_sse128)
void intra_pred_ang_xy_23_sse128(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);

#define fill_edge_samples_0_sse128 FPFX(fill_edge_samples_0_sse128)
void fill_edge_samples_0_sse128 (const pel_t *pTL, int i_TL, const pel_t *pLcuEP, pel_t *EP, uint32_t i_avai, int bsx, int bsy);
#define fill_edge_samples_x_sse128 FPFX(fill_edge_samples_x_sse128)
void fill_edge_samples_x_sse128 (const pel_t *pTL, int i_TL, const pel_t *pLcuEP, pel_t *EP, uint32_t i_avai, int bsx, int bsy);
#define fill_edge_samples_y_sse128 FPFX(fill_edge_samples_y_sse128)
void fill_edge_samples_y_sse128 (const pel_t *pTL, int i_TL, const pel_t *pLcuEP, pel_t *EP, uint32_t i_avai, int bsx, int bsy);
#define fill_edge_samples_xy_sse128 FPFX(fill_edge_samples_xy_sse128)
void fill_edge_samples_xy_sse128(const pel_t *pTL, int i_TL, const pel_t *pLcuEP, pel_t *EP, uint32_t i_avai, int bsx, int bsy);

//intra prediction avx functions
#define intra_pred_ver_avx FPFX(intra_pred_ver_avx)
void intra_pred_ver_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_hor_avx FPFX(intra_pred_hor_avx)
void intra_pred_hor_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_dc_avx FPFX(intra_pred_dc_avx)
void intra_pred_dc_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_plane_avx FPFX(intra_pred_plane_avx)
void intra_pred_plane_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_bilinear_avx FPFX(intra_pred_bilinear_avx)
void intra_pred_bilinear_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_x_3_avx FPFX(intra_pred_ang_x_3_avx)
void intra_pred_ang_x_3_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_x_4_avx FPFX(intra_pred_ang_x_4_avx)
void intra_pred_ang_x_4_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_x_5_avx FPFX(intra_pred_ang_x_5_avx)
void intra_pred_ang_x_5_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_x_6_avx FPFX(intra_pred_ang_x_6_avx)
void intra_pred_ang_x_6_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_x_7_avx FPFX(intra_pred_ang_x_7_avx)
void intra_pred_ang_x_7_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_x_8_avx FPFX(intra_pred_ang_x_8_avx)
void intra_pred_ang_x_8_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_x_9_avx FPFX(intra_pred_ang_x_9_avx)
void intra_pred_ang_x_9_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_x_10_avx FPFX(intra_pred_ang_x_10_avx)
void intra_pred_ang_x_10_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_x_11_avx FPFX(intra_pred_ang_x_11_avx)
void intra_pred_ang_x_11_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);

#define intra_pred_ang_xy_13_avx FPFX(intra_pred_ang_xy_13_avx)
void intra_pred_ang_xy_13_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_xy_14_avx FPFX(intra_pred_ang_xy_14_avx)
void intra_pred_ang_xy_14_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_xy_16_avx FPFX(intra_pred_ang_xy_16_avx)
void intra_pred_ang_xy_16_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_xy_18_avx FPFX(intra_pred_ang_xy_18_avx)
void intra_pred_ang_xy_18_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_xy_20_avx FPFX(intra_pred_ang_xy_20_avx)
void intra_pred_ang_xy_20_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_xy_22_avx FPFX(intra_pred_ang_xy_22_avx)
void intra_pred_ang_xy_22_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_xy_23_avx FPFX(intra_pred_ang_xy_23_avx)
void intra_pred_ang_xy_23_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);

#define intra_pred_ang_y_25_avx FPFX(intra_pred_ang_y_25_avx)
void intra_pred_ang_y_25_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_y_26_avx FPFX(intra_pred_ang_y_26_avx)
void intra_pred_ang_y_26_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_y_28_avx FPFX(intra_pred_ang_y_28_avx)
void intra_pred_ang_y_28_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_y_30_avx FPFX(intra_pred_ang_y_30_avx)
void intra_pred_ang_y_30_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_y_31_avx FPFX(intra_pred_ang_y_31_avx)
void intra_pred_ang_y_31_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);
#define intra_pred_ang_y_32_avx FPFX(intra_pred_ang_y_32_avx)
void intra_pred_ang_y_32_avx(pel_t *src, pel_t *dst, int i_dst, int dir_mode, int bsx, int bsy);



#define mad_16x16_sse128 FPFX(mad_16x16_sse128)
int mad_16x16_sse128(pel_t *p_src, int i_src, int cu_size);
#define mad_32x32_sse128 FPFX(mad_32x32_sse128)
int mad_32x32_sse128(pel_t *p_src, int i_src, int cu_size);
#define mad_64x64_sse128 FPFX(mad_64x64_sse128)
int mad_64x64_sse128(pel_t *p_src, int i_src, int cu_size);


#endif // #ifndef XAVS2_INTRINSIC_H
