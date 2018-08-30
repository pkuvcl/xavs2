/*****************************************************************************
 * Copyright (C) 2013-2017 MulticoreWare, Inc
 *
 * Authors: Steve Borho <steve@borho.org>
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

#ifndef XAVS2_I386_MC_H
#define XAVS2_I386_MC_H

#define xavs2_plane_copy_core_mmx2 FPFX(plane_copy_core_mmx2)
void xavs2_plane_copy_core_mmx2(pel_t *dst, intptr_t i_dst, pel_t *src, intptr_t i_src, int w, int h);
#define xavs2_plane_copy_deinterleave_mmx FPFX(plane_copy_deinterleave_mmx)
void xavs2_plane_copy_deinterleave_mmx(pel_t *dstu, intptr_t i_dstu, pel_t *dstv, intptr_t i_dstv, pel_t *src, intptr_t i_src, int w, int h);

#define xavs2_memcpy_aligned_mmx FPFX(memcpy_aligned_mmx)
void *xavs2_memcpy_aligned_mmx(void *dst, const void *src, size_t n);
#define xavs2_memcpy_aligned_sse FPFX(memcpy_aligned_sse)
void *xavs2_memcpy_aligned_sse(void *dst, const void *src, size_t n);

#define xavs2_fast_memcpy_mmx FPFX(fast_memcpy_mmx)
void *xavs2_fast_memcpy_mmx    (void *dst, const void *src, size_t n);

#define xavs2_fast_memset_mmx FPFX(fast_memset_mmx)
void *xavs2_fast_memset_mmx    (void *dst, int val, size_t n);

#define xavs2_memzero_aligned_mmx FPFX(memzero_aligned_mmx)
void *xavs2_memzero_aligned_mmx(void *dst, size_t n);
#define xavs2_memzero_aligned_sse FPFX(memzero_aligned_sse)
void *xavs2_memzero_aligned_sse(void *dst, size_t n);
#define xavs2_memzero_aligned_avx FPFX(memzero_aligned_avx)
void *xavs2_memzero_aligned_avx(void *dst, size_t n);

#define xavs2_fast_memzero_mmx FPFX(fast_memzero_mmx)
void *xavs2_fast_memzero_mmx   (void *dst, size_t n);

#define xavs2_lowres_filter_core_mmx2 FPFX(lowres_filter_core_mmx2)
void xavs2_lowres_filter_core_mmx2 (pel_t *src, int i_src, pel_t *dst, int i_dst, int width, int height);
#define xavs2_lowres_filter_core_sse2 FPFX(lowres_filter_core_sse2)
void xavs2_lowres_filter_core_sse2 (pel_t *src, int i_src, pel_t *dst, int i_dst, int width, int height);
#define xavs2_lowres_filter_core_ssse3 FPFX(lowres_filter_core_ssse3)
void xavs2_lowres_filter_core_ssse3(pel_t *src, int i_src, pel_t *dst, int i_dst, int width, int height);
#define xavs2_lowres_filter_core_avx FPFX(lowres_filter_core_avx)
void xavs2_lowres_filter_core_avx  (pel_t *src, int i_src, pel_t *dst, int i_dst, int width, int height);

#endif  // XAVS2_I386_MC_H
