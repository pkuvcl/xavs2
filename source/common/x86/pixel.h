/*****************************************************************************
 * pixel.h: x86 pixel metrics
 *****************************************************************************
 * Copyright (C) 2003-2013 x264 project
 * Copyright (C) 2013-2017 MulticoreWare, Inc
 *
 * Authors: Laurent Aimar <fenrir@via.ecp.fr>
 *          Loren Merritt <lorenm@u.washington.edu>
 *          Fiona Glaser <fiona@x264.com>
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

#ifndef XAVS2_X86_PIXEL_H
#define XAVS2_X86_PIXEL_H


/**
 * ===========================================================================
 * function declares
 * ===========================================================================
 */

#define FUNCDEF_TU(ret, name, cpu, ...) \
    ret FPFX(name ## _4x4_   ## cpu(__VA_ARGS__));\
    ret FPFX(name ## _8x8_   ## cpu(__VA_ARGS__));\
    ret FPFX(name ## _16x16_ ## cpu(__VA_ARGS__));\
    ret FPFX(name ## _32x32_ ## cpu(__VA_ARGS__));\
    ret FPFX(name ## _64x64_ ## cpu(__VA_ARGS__))

#define FUNCDEF_TU_S(ret, name, cpu, ...) \
    ret FPFX(name ## _4_  ## cpu(__VA_ARGS__));\
    ret FPFX(name ## _8_  ## cpu(__VA_ARGS__));\
    ret FPFX(name ## _16_ ## cpu(__VA_ARGS__));\
    ret FPFX(name ## _32_ ## cpu(__VA_ARGS__));\
    ret FPFX(name ## _64_ ## cpu(__VA_ARGS__))

#define FUNCDEF_PU(ret, name, cpu, ...) \
    ret FPFX(name ## _4x4_   ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _8x8_   ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _16x16_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _32x32_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _64x64_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _8x4_   ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _4x8_   ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _16x8_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _8x16_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _16x32_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _32x16_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _64x32_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _32x64_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _16x12_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _12x16_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _16x4_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _4x16_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _32x24_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _24x32_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _32x8_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _8x32_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _64x48_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _48x64_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _64x16_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _16x64_ ## cpu)(__VA_ARGS__)

#define FUNCDEF_CHROMA_PU(ret, name, cpu, ...) \
    FUNCDEF_PU(ret, name, cpu, __VA_ARGS__);\
    ret FPFX(name ## _4x2_   ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _2x4_   ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _8x2_   ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _2x8_   ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _8x6_   ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _6x8_   ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _8x12_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _12x8_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _6x16_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _16x6_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _2x16_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _16x2_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _4x12_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _12x4_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _32x12_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _12x32_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _32x4_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _4x32_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _32x48_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _48x32_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _16x24_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _24x16_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _8x64_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _64x8_  ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _64x24_ ## cpu)(__VA_ARGS__);\
    ret FPFX(name ## _24x64_ ## cpu)(__VA_ARGS__);

#define DECL_PIXELS(cpu) \
    FUNCDEF_PU(int,         pixel_ssd,    cpu, const pel_t*, intptr_t, const pel_t*, intptr_t);\
    FUNCDEF_PU(int,         pixel_sa8d,   cpu, const pel_t*, intptr_t, const pel_t*, intptr_t);\
    FUNCDEF_PU(void,        pixel_sad_x3, cpu, const pel_t*, const pel_t*, const pel_t*, const pel_t*,               intptr_t, int32_t*);\
    FUNCDEF_PU(void,        pixel_sad_x4, cpu, const pel_t*, const pel_t*, const pel_t*, const pel_t*, const pel_t*, intptr_t, int32_t*);\
    FUNCDEF_PU(void,        pixel_avg,    cpu, pel_t* dst, intptr_t dstride, const pel_t* src0, intptr_t sstride0, const pel_t* src1, intptr_t sstride1, int);\
    FUNCDEF_PU(void,        pixel_add_ps, cpu, pel_t* a,   intptr_t dstride, const pel_t* b0, const coeff_t* b1, intptr_t sstride0, intptr_t sstride1);\
    FUNCDEF_PU(void,        pixel_sub_ps, cpu, coeff_t* a, intptr_t dstride, const pel_t* b0, const pel_t*   b1, intptr_t sstride0, intptr_t sstride1);\
    FUNCDEF_PU(int,         pixel_satd,   cpu, const pel_t*, intptr_t, const pel_t*, intptr_t);\
    FUNCDEF_PU(int,         pixel_sad,    cpu, const pel_t*, intptr_t, const pel_t*, intptr_t);\
    FUNCDEF_PU(int,         pixel_ssd_ss, cpu, const int16_t*, intptr_t, const int16_t*, intptr_t);\
    FUNCDEF_PU(void,        addAvg,       cpu, const int16_t*, const int16_t*, pel_t*, intptr_t, intptr_t, intptr_t);\
    FUNCDEF_PU(int,         pixel_ssd_s,  cpu, const int16_t*, intptr_t);\
    FUNCDEF_TU_S(int,       pixel_ssd_s,  cpu, const int16_t*, intptr_t);\
    FUNCDEF_TU(uint64_t,    pixel_var,    cpu, const pel_t*, intptr_t);\
    FUNCDEF_TU(int,         psyCost_pp,   cpu, const pel_t*   source, intptr_t sstride, const pel_t*   recon, intptr_t rstride);\
    FUNCDEF_TU(int,         psyCost_ss,   cpu, const int16_t* source, intptr_t sstride, const int16_t* recon, intptr_t rstride)

DECL_PIXELS(mmx);
DECL_PIXELS(mmx2);
DECL_PIXELS(sse2);
DECL_PIXELS(sse3);
DECL_PIXELS(sse4);
DECL_PIXELS(ssse3);
DECL_PIXELS(avx);
DECL_PIXELS(xop);
DECL_PIXELS(avx2);

#undef DECL_PIXELS

#endif  // XAVS2_X86_PIXEL_H
