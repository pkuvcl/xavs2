/*
 * configw.h
 *
 * Description of this file:
 *    compiling configuration for windows platform
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

#ifndef XAVS2_CONFIGW_H
#define XAVS2_CONFIGW_H

#if defined(__ICL) || defined(_MSC_VER)

/* arch */
#define ARCH_X86                1
#define ARCH_PPC                0
#define ARCH_ARM                0
#define ARCH_UltraSPARC         0

/* system */
#define SYS_WINDOWS             1
#define SYS_LINUX               0
#define SYS_MACOSX              0
#define SYS_BEOS                0
#define SYS_FREEBSD             0
#define SYS_OPENBSD             0

/* cpu */
#ifndef __SSE__
#define __SSE__
#endif
#define HAVE_MMX                1     /* X86     */
#define HAVE_ALTIVEC            0     /* ALTIVEC */
#define HAVE_ALTIVEC_H          0
#define HAVE_ARMV6              0
#define HAVE_ARMV6T2            0

/* thread */
#define HAVE_THREAD             1
#define HAVE_WIN32THREAD        1
#define HAVE_PTHREAD            0
#define HAVE_BEOSTHREAD         0
#define HAVE_POSIXTHREAD        0
#define PTW32_STATIC_LIB        0

/* interlace support */
#define HAVE_INTERLACED         1

/* malloc */
#define HAVE_MALLOC_H           0

/* big-endian */
#define WORDS_BIGENDIAN         0

/* others */
#define HAVE_STDINT_H           1
#define HAVE_VECTOREXT          0
#define HAVE_LOG2F              0
#define HAVE_SWSCALE            0
#define HAVE_LAVF               0
#define HAVE_FFMS               0
#define HAVE_GPAC               0
#define HAVE_GF_MALLOC          0
#define HAVE_AVS                0

#endif
#endif // XAVS2_CONFIGW_H
