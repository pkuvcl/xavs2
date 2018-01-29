/*
 * common_arm.c
 *
 * Description of this file:
 *    common tables definition of the xavs2 library
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

#ifdef __ARM_ARCH_7A__

#include "common_arm.h"
#include "common.h"

//ARM_PART_4x2=10, ARM_PART_2x4 = 12, ARM_PART_8x2= 18, ARM_PART_2x8=24,ARM_PART_8x6=54,ARM_PART_6x8=56
const unsigned char g_arm_partition_map_tab[] =
{
        ARM_PLANE_COPY_W88, 255, ARM_PLANE_COPY_W160, ARM_PLANE_COPY_W176, 255, 255, 255, ARM_PLANE_COPY_W320, ARM_PLANE_COPY_W352, ARM_PLANE_COPY_W360, ARM_PART_4x2, 255, ARM_PART_2x4, ARM_PLANE_COPY_W512, 255, 255,
        255, ARM_PLANE_COPY_W640, ARM_PART_8x2, ARM_PLANE_COPY_W704, ARM_PLANE_COPY_W720, 255, 255, 255, ARM_PART_2x8, 255, 255, ARM_PLANE_COPY_W960, 255, ARM_PLANE_COPY_W1024, 255, 255,
        255, 255, 255, 255, 255, ARM_PLANE_COPY_W1280, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, ARM_PART_8x6, 255, ARM_PART_6x8, ARM_PLANE_COPY_W1920, 255, 255, 255, 255, 255, 255
};


/*
        g_T4[0][0] * g_T4[0][0], g_T4[0][0] * g_T4[1][0], g_T4[0][0] * g_T4[2][0], g_T4[0][0] * g_T4[3][0],
        g_T4[0][0] * g_T4[0][1], g_T4[0][0] * g_T4[1][1], g_T4[0][0] * g_T4[2][1], g_T4[0][0] * g_T4[3][1],
        g_T4[0][1] * g_T4[0][0], g_T4[0][1] * g_T4[1][0], g_T4[0][1] * g_T4[2][0], g_T4[0][1] * g_T4[3][0],
        g_T4[0][1] * g_T4[0][1], g_T4[0][1] * g_T4[1][1], g_T4[0][1] * g_T4[2][1], g_T4[0][1] * g_T4[3][1],

        g_T4[1][0] * g_T4[0][0], g_T4[1][0] * g_T4[1][0], g_T4[1][0] * g_T4[2][0], g_T4[1][0] * g_T4[3][0],
        g_T4[1][0] * g_T4[0][1], g_T4[1][0] * g_T4[1][1], g_T4[1][0] * g_T4[2][1], g_T4[1][0] * g_T4[3][1],
        g_T4[1][1] * g_T4[0][0], g_T4[1][1] * g_T4[1][0], g_T4[1][1] * g_T4[2][0], g_T4[1][1] * g_T4[3][0],
        g_T4[1][1] * g_T4[0][1], g_T4[1][1] * g_T4[1][1], g_T4[1][1] * g_T4[2][1], g_T4[1][1] * g_T4[3][1],

        g_T4[2][0] * g_T4[0][0], g_T4[2][0] * g_T4[1][0], g_T4[2][0] * g_T4[2][0], g_T4[2][0] * g_T4[3][0],
        g_T4[2][0] * g_T4[0][1], g_T4[2][0] * g_T4[1][1], g_T4[2][0] * g_T4[2][1], g_T4[2][0] * g_T4[3][1],
        g_T4[2][1] * g_T4[0][0], g_T4[2][1] * g_T4[1][0], g_T4[2][1] * g_T4[2][0], g_T4[2][1] * g_T4[3][0],
        g_T4[2][1] * g_T4[0][1], g_T4[2][1] * g_T4[1][1], g_T4[2][1] * g_T4[2][1], g_T4[2][1] * g_T4[3][1],

        g_T4[3][0] * g_T4[0][0], g_T4[3][0] * g_T4[1][0], g_T4[3][0] * g_T4[2][0], g_T4[3][0] * g_T4[3][0],
        g_T4[3][0] * g_T4[0][1], g_T4[3][0] * g_T4[1][1], g_T4[3][0] * g_T4[2][1], g_T4[3][0] * g_T4[3][1],
        g_T4[3][1] * g_T4[0][0], g_T4[3][1] * g_T4[1][0], g_T4[3][1] * g_T4[2][0], g_T4[3][1] * g_T4[3][0],
        g_T4[3][1] * g_T4[0][1], g_T4[3][1] * g_T4[1][1], g_T4[3][1] * g_T4[2][1], g_T4[3][1] * g_T4[3][1]
 */

ALIGN32(short dct4x4_const_table[64]) = {
            32 * 32,    32 * 42,    32 *   32,      32 *   17,
            32 * 32,    32 * 17,    32 * (-32),     32 * (-42),
            32 * 32,    32 * 42,    32 *   32,      32 *   17,
            32 * 32,    32 * 17,    32 * (-32),     32 * (-42),

            42 * 32,    42 * 42,    42 *   32,      42 *   17,
            42 * 32,    42 * 17,    42 * (-32),     42 * (-42),
            17 * 32,    17 * 42,    17 *   32,      17 *   17,
            17 * 32,    17 * 17,    17 * (-32),     17 * (-42),

            32 * 32,    32 * 42,    32 *   32,      32 *   17,
            32 * 32,    32 * 17,    32 * (-32),     32 * (-42),
         (-32) * 32, (-32) * 42, (-32) *   32,   (-32) *   17,
         (-32) * 32, (-32) * 17, (-32) * (-32),  (-32) * (-42),

            17 * 32,    17 * 42,    17 *   32,    	17 *   17,
            17 * 32,    17 * 17,    17 * (-32),     17 * (-42),
         (-42) * 32, (-42) * 42, (-42) *   32,   (-42) *   17,
         (-42) * 32, (-42) * 17, (-42) * (-32),  (-42) * (-42)
};
ALIGN32(short g_dct_temp_buf[1024]) = {0};
#endif  //__ARM_ARCH_7A__
