/*****************************************************************************
 * Copyright (C) 2013-2017 MulticoreWare, Inc
 *
 * Authors: Steve Borho <steve@borho.org>
;*          Min Chen <chenm003@163.com>
 *          Jiaqi Zhang <zhangjiaqi.cs@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at license @ x265.com.
 *****************************************************************************/

#ifndef XAVS2_PIXEL_UTIL_H
#define XAVS2_PIXEL_UTIL_H

#define xavs2_getResidual4_sse2 FPFX(getResidual4_sse2)
void xavs2_getResidual4_sse2(const pel_t *fenc, const pel_t *pred, int16_t *residual, intptr_t stride);
#define xavs2_getResidual8_sse2 FPFX(getResidual8_sse2)
void xavs2_getResidual8_sse2(const pel_t *fenc, const pel_t *pred, int16_t *residual, intptr_t stride);
#define xavs2_getResidual16_sse2 FPFX(getResidual16_sse2)
void xavs2_getResidual16_sse2(const pel_t *fenc, const pel_t *pred, int16_t *residual, intptr_t stride);
#define xavs2_getResidual16_sse4 FPFX(getResidual16_sse4)
void xavs2_getResidual16_sse4(const pel_t *fenc, const pel_t *pred, int16_t *residual, intptr_t stride);
#define xavs2_getResidual32_sse2 FPFX(getResidual32_sse2)
void xavs2_getResidual32_sse2(const pel_t *fenc, const pel_t *pred, int16_t *residual, intptr_t stride);
#define xavs2_getResidual32_sse4 FPFX(getResidual32_sse4)
void xavs2_getResidual32_sse4(const pel_t *fenc, const pel_t *pred, int16_t *residual, intptr_t stride);
#define xavs2_getResidual16_avx2 FPFX(getResidual16_avx2)
void xavs2_getResidual16_avx2(const pel_t *fenc, const pel_t *pred, int16_t *residual, intptr_t stride);
#define xavs2_getResidual32_avx2 FPFX(getResidual32_avx2)
void xavs2_getResidual32_avx2(const pel_t *fenc, const pel_t *pred, int16_t *residual, intptr_t stride);

#define xavs2_transpose4_sse2 FPFX(transpose4_sse2)
void xavs2_transpose4_sse2(pel_t *dst, const pel_t *src, intptr_t stride);
#define xavs2_transpose8_sse2 FPFX(transpose8_sse2)
void xavs2_transpose8_sse2(pel_t *dst, const pel_t *src, intptr_t stride);
#define xavs2_transpose16_sse2 FPFX(transpose16_sse2)
void xavs2_transpose16_sse2(pel_t *dst, const pel_t *src, intptr_t stride);
#define xavs2_transpose32_sse2 FPFX(transpose32_sse2)
void xavs2_transpose32_sse2(pel_t *dst, const pel_t *src, intptr_t stride);
#define xavs2_transpose64_sse2 FPFX(transpose64_sse2)
void xavs2_transpose64_sse2(pel_t *dst, const pel_t *src, intptr_t stride);

#define xavs2_transpose8_avx2 FPFX(transpose8_avx2)
void xavs2_transpose8_avx2(pel_t *dst, const pel_t *src, intptr_t stride);
#define xavs2_transpose16_avx2 FPFX(transpose16_avx2)
void xavs2_transpose16_avx2(pel_t *dst, const pel_t *src, intptr_t stride);
#define xavs2_transpose32_avx2 FPFX(transpose32_avx2)
void xavs2_transpose32_avx2(pel_t *dst, const pel_t *src, intptr_t stride);
#define xavs2_transpose64_avx2 FPFX(transpose64_avx2)
void xavs2_transpose64_avx2(pel_t *dst, const pel_t *src, intptr_t stride);

#define xavs2_count_nonzero_4x4_ssse3 FPFX(count_nonzero_4x4_ssse3)
int xavs2_count_nonzero_4x4_ssse3(const int16_t *quantCoeff);
#define xavs2_count_nonzero_8x8_ssse3 FPFX(count_nonzero_8x8_ssse3)
int xavs2_count_nonzero_8x8_ssse3(const int16_t *quantCoeff);
#define xavs2_count_nonzero_16x16_ssse3 FPFX(count_nonzero_16x16_ssse3)
int xavs2_count_nonzero_16x16_ssse3(const int16_t *quantCoeff);
#define xavs2_count_nonzero_32x32_ssse3 FPFX(count_nonzero_32x32_ssse3)
int xavs2_count_nonzero_32x32_ssse3(const int16_t *quantCoeff);
#define xavs2_count_nonzero_4x4_avx2 FPFX(count_nonzero_4x4_avx2)
int xavs2_count_nonzero_4x4_avx2(const int16_t *quantCoeff);
#define xavs2_count_nonzero_8x8_avx2 FPFX(count_nonzero_8x8_avx2)
int xavs2_count_nonzero_8x8_avx2(const int16_t *quantCoeff);
#define xavs2_count_nonzero_16x16_avx2 FPFX(count_nonzero_16x16_avx2)
int xavs2_count_nonzero_16x16_avx2(const int16_t *quantCoeff);
#define xavs2_count_nonzero_32x32_avx2 FPFX(count_nonzero_32x32_avx2)
int xavs2_count_nonzero_32x32_avx2(const int16_t *quantCoeff);

#define xavs2_weight_pp_sse4 FPFX(weight_pp_sse4)
void xavs2_weight_pp_sse4(const pel_t *src, pel_t *dst, intptr_t stride, int width, int height, int w0, int round, int shift, int offset);
#define xavs2_weight_pp_avx2 FPFX(weight_pp_avx2)
void xavs2_weight_pp_avx2(const pel_t *src, pel_t *dst, intptr_t stride, int width, int height, int w0, int round, int shift, int offset);
#define xavs2_weight_sp_sse4 FPFX(weight_sp_sse4)
void xavs2_weight_sp_sse4(const int16_t *src, pel_t *dst, intptr_t srcStride, intptr_t dstStride, int width, int height, int w0, int round, int shift, int offset);

#define xavs2_pixel_ssim_4x4x2_core_mmx2 FPFX(pixel_ssim_4x4x2_core_mmx2)
void xavs2_pixel_ssim_4x4x2_core_mmx2(const pel_t *pix1, intptr_t stride1,
                                      const pel_t *pix2, intptr_t stride2, int sums[2][4]);
#define xavs2_pixel_ssim_4x4x2_core_sse2 FPFX(pixel_ssim_4x4x2_core_sse2)
void xavs2_pixel_ssim_4x4x2_core_sse2(const pel_t *pix1, intptr_t stride1,
                                      const pel_t *pix2, intptr_t stride2, int sums[2][4]);
#define xavs2_pixel_ssim_4x4x2_core_avx FPFX(pixel_ssim_4x4x2_core_avx)
void xavs2_pixel_ssim_4x4x2_core_avx(const pel_t *pix1, intptr_t stride1,
                                     const pel_t *pix2, intptr_t stride2, int sums[2][4]);
#define xavs2_pixel_ssim_end4_sse2 FPFX(pixel_ssim_end4_sse2)
float xavs2_pixel_ssim_end4_sse2(int sum0[5][4], int sum1[5][4], int width);
#define xavs2_pixel_ssim_end4_avx FPFX(pixel_ssim_end4_avx)
float xavs2_pixel_ssim_end4_avx(int sum0[5][4], int sum1[5][4], int width);

#define xavs2_scale1D_128to64_ssse3 FPFX(scale1D_128to64_ssse3)
void xavs2_scale1D_128to64_ssse3(pel_t*, const pel_t*);
#define xavs2_scale1D_128to64_avx2 FPFX(scale1D_128to64_avx2)
void xavs2_scale1D_128to64_avx2(pel_t*, const pel_t*);
#define xavs2_scale2D_64to32_ssse3 FPFX(scale2D_64to32_ssse3)
void xavs2_scale2D_64to32_ssse3(pel_t*, const pel_t*, intptr_t);
#define xavs2_scale2D_64to32_avx2 FPFX(scale2D_64to32_avx2)
void xavs2_scale2D_64to32_avx2(pel_t*, const pel_t*, intptr_t);

#define xavs2_scanPosLast_x64 FPFX(scanPosLast_x64)
int xavs2_scanPosLast_x64(const uint16_t *scan, const coeff_t *coeff, uint16_t *coeffSign, uint16_t *coeffFlag, uint8_t *coeffNum, int numSig, const uint16_t *scanCG4x4, const int trSize);
#define xavs2_scanPosLast_avx2_bmi2 FPFX(scanPosLast_avx2_bmi2)
int xavs2_scanPosLast_avx2_bmi2(const uint16_t *scan, const coeff_t *coeff, uint16_t *coeffSign, uint16_t *coeffFlag, uint8_t *coeffNum, int numSig, const uint16_t *scanCG4x4, const int trSize);
#define xavs2_findPosFirstLast_ssse3 FPFX(findPosFirstLast_ssse3)
uint32_t xavs2_findPosFirstLast_ssse3(const int16_t *dstCoeff, const intptr_t trSize, const uint16_t scanTbl[16]);

#define xavs2_costCoeffNxN_sse4 FPFX(costCoeffNxN_sse4)
uint32_t xavs2_costCoeffNxN_sse4(const uint16_t *scan, const coeff_t *coeff, intptr_t trSize, uint16_t *absCoeff, const uint8_t *tabSigCtx, uint32_t scanFlagMask, uint8_t *baseCtx, int offset, int scanPosSigOff, int subPosBase);


#define SETUP_CHROMA_PIXELSUB_PS_FUNC(W, H, cpu) \
    void xavs2_pixel_sub_ps_ ## W ## x ## H ## cpu(coeff_t *dst, intptr_t destride, const pel_t *src0, const pel_t *src1, intptr_t srcstride0, intptr_t srcstride1); \
    void xavs2_pixel_add_ps_ ## W ## x ## H ## cpu(pel_t *dst, intptr_t destride, const pel_t *src0, const coeff_t * src1, intptr_t srcStride0, intptr_t srcStride1);

#define CHROMA_PIXELSUB_DEF(cpu) \
    SETUP_CHROMA_PIXELSUB_PS_FUNC(4, 4, cpu); \
    SETUP_CHROMA_PIXELSUB_PS_FUNC(8, 8, cpu); \
    SETUP_CHROMA_PIXELSUB_PS_FUNC(16, 16, cpu); \
    SETUP_CHROMA_PIXELSUB_PS_FUNC(32, 32, cpu);

#define CHROMA_422_PIXELSUB_DEF(cpu) \
    SETUP_CHROMA_PIXELSUB_PS_FUNC(4, 8, cpu); \
    SETUP_CHROMA_PIXELSUB_PS_FUNC(8, 16, cpu); \
    SETUP_CHROMA_PIXELSUB_PS_FUNC(16, 32, cpu); \
    SETUP_CHROMA_PIXELSUB_PS_FUNC(32, 64, cpu);

#define SETUP_LUMA_PIXELSUB_PS_FUNC(W, H, cpu) \
    void xavs2_pixel_sub_ps_ ## W ## x ## H ## cpu(coeff_t *dst, intptr_t destride, const pel_t *src0, const pel_t *src1, intptr_t srcstride0, intptr_t srcstride1); \
    void xavs2_pixel_add_ps_ ## W ## x ## H ## cpu(pel_t *dst, intptr_t destride, const pel_t *src0, const coeff_t * src1, intptr_t srcStride0, intptr_t srcStride1);

#define LUMA_PIXELSUB_DEF(cpu) \
    SETUP_LUMA_PIXELSUB_PS_FUNC(8,   8, cpu); \
    SETUP_LUMA_PIXELSUB_PS_FUNC(16, 16, cpu); \
    SETUP_LUMA_PIXELSUB_PS_FUNC(32, 32, cpu); \
    SETUP_LUMA_PIXELSUB_PS_FUNC(64, 64, cpu);

LUMA_PIXELSUB_DEF(_sse2);
CHROMA_PIXELSUB_DEF(_sse2);
CHROMA_422_PIXELSUB_DEF(_sse2);

LUMA_PIXELSUB_DEF(_sse4);
CHROMA_PIXELSUB_DEF(_sse4);
CHROMA_422_PIXELSUB_DEF(_sse4);

#define SETUP_LUMA_PIXELVAR_FUNC(W, H, cpu) \
    uint64_t xavs2_pixel_var_ ## W ## x ## H ## cpu(const pel_t *pix, intptr_t pixstride);

#define LUMA_PIXELVAR_DEF(cpu) \
    SETUP_LUMA_PIXELVAR_FUNC(8,   8, cpu); \
    SETUP_LUMA_PIXELVAR_FUNC(16, 16, cpu); \
    SETUP_LUMA_PIXELVAR_FUNC(32, 32, cpu); \
    SETUP_LUMA_PIXELVAR_FUNC(64, 64, cpu);

LUMA_PIXELVAR_DEF(_sse2);
LUMA_PIXELVAR_DEF(_xop);
LUMA_PIXELVAR_DEF(_avx);

#undef CHROMA_PIXELSUB_DEF
#undef CHROMA_422_PIXELSUB_DEF
#undef LUMA_PIXELSUB_DEF
#undef LUMA_PIXELVAR_DEF
#undef SETUP_CHROMA_PIXELSUB_PS_FUNC
#undef SETUP_LUMA_PIXELSUB_PS_FUNC
#undef SETUP_LUMA_PIXELVAR_FUNC

#endif // ifndef XAVS2_PIXEL_UTIL_H
