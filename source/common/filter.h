/*
 * filter.h
 *
 * Description of this file:
 *    Filter functions definition of the xavs2 library
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


#ifndef XAVS2_IN_LOOP_FILTERS_H
#define XAVS2_IN_LOOP_FILTERS_H

/**
 * ===========================================================================
 * global/local variables
 * ===========================================================================
 */

static const int tab_saoclip[NUM_SAO_OFFSET][3] = {
    // EO
    { -1, 6, 7 },   // low bound, upper bound, threshold
    {  0, 1, 1 },
    {  0, 0, 0 },
    { -1, 0, 1 },
    { -6, 1, 7 },
    { -7, 7, 7 }    // BO
};
/* ---------------------------------------------------------------------------
* lcu neighbor
*/
enum lcu_neighbor_e {
    SAO_T = 0,    /* top        */
    SAO_D = 1,    /* down       */
    SAO_L = 2,    /* left       */
    SAO_R = 3,    /* right      */
    SAO_TL = 4,    /* top-left   */
    SAO_TR = 5,    /* top-right  */
    SAO_DL = 6,    /* down-left  */
    SAO_DR = 7     /* down-right */
};

typedef struct sao_region_t {
    int    pix_x[NUM_SAO_COMPONENTS];       /* start pixel position in x */
    int    pix_y[NUM_SAO_COMPONENTS];       /* start pixel position in y */
    int    width[NUM_SAO_COMPONENTS];       /*  */
    int    height[NUM_SAO_COMPONENTS];      /*  */

    /* availabilities of neighboring blocks */
    int8_t b_left;
    int8_t b_top_left;
    int8_t b_top;
    int8_t b_top_right;
    int8_t b_right;
    int8_t b_right_down;
    int8_t b_down;
    int8_t b_down_left;
} sao_region_t;

#define xavs2_lcu_deblock FPFX(lcu_deblock)
void xavs2_lcu_deblock(xavs2_t *h, xavs2_frame_t *frm);

#endif  // XAVS2_IN_LOOP_FILTERS_H
