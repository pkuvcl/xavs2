/*
 * common_arm.h
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

#ifndef COMMON_ARM_H_
#define COMMON_ARM_H_

#ifdef __ARM_ARCH_7A__

enum ARM_MC_PART {
    /*mc_copy idx*/
    ARM_PART_2x4, ARM_PART_2x8,
    ARM_PART_4x2, ARM_PART_6x8,
    ARM_PART_8x2, ARM_PART_8x6
};

enum ARM_PLANE_COPY_PART {
    /*plane_copy idx*/
    ARM_PLANE_COPY_W88, ARM_PLANE_COPY_W160,
    ARM_PLANE_COPY_W176, ARM_PLANE_COPY_W320,
    ARM_PLANE_COPY_W352, ARM_PLANE_COPY_W360,
    ARM_PLANE_COPY_W512, ARM_PLANE_COPY_W640,
    ARM_PLANE_COPY_W704, ARM_PLANE_COPY_W720,
    ARM_PLANE_COPY_W960, ARM_PLANE_COPY_W1024,
    ARM_PLANE_COPY_W1280, ARM_PLANE_COPY_W1920
};
extern const unsigned char g_arm_partition_map_tab[];
//Wxh= 2x4, 2x8, 6x8, 8x6
#define ARM_MC_PART_INDEX(w, h)         g_arm_partition_map_tab[(w + 1) * h]
#define ARM_PLANE_COPY_INDEX(w)         g_arm_partition_map_tab[(((w + 8) >> 4) - 5) >> 1]

extern short dct4x4_const_table[64];
#endif /* __ARM_ARCH_7A__ */

#endif /* COMMON_ARM_H_ */
