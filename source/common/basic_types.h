/*
 * basic_types.h
 *
 * Description of this file:
 *    basic types definition of the xavs2 library
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


#ifndef XAVS2_BASIC_TYPES_H
#define XAVS2_BASIC_TYPES_H

#include <stdint.h>
#include "defines.h"
#include "osdep.h"

/*
 * ===========================================================================
 * basic types
 * ===========================================================================
 */
typedef uint8_t     pel_t;      /* type for pixel */
typedef int16_t     itr_t;      /* intra prediction temp */
typedef uint16_t    sum_t;
typedef uint32_t    sum2_t;
typedef uint32_t    pixel4;
typedef int32_t     ssum2_t;    /* Signed sum */
typedef int32_t     dist_t;

typedef int8_t      bool_t;     /* Bool type, true or false */
typedef int16_t     mct_t;      /* motion compensation temp */
typedef int16_t     coeff_t;    /* type for transform coefficient */
typedef int32_t     cmp_dist_t; /* distortion type */
typedef double      rdcost_t;   /* type for RDcost calculation, can also be int64_t */

/*
 * ===========================================================================
 * structure types
 * ===========================================================================
 */
typedef struct xavs2_handler_t  xavs2_handler_t;   /* top handler of the encoder */
typedef struct xavs2_log_t      xavs2_log_t;       /* log module */
typedef struct xavs2_t          xavs2_t;           /* main encoder context for one thread */
typedef struct xavs2_frame_t    xavs2_frame_t;
typedef struct xavs2_frame_buffer_t xavs2_frame_buffer_t;
typedef struct ratectrl_t       ratectrl_t;
typedef struct cu_size_ctrl_t   cu_size_ctrl_t;
typedef struct td_rdo_t         td_rdo_t;
typedef struct aec_t            aec_t;
typedef struct cu_t             cu_t;
typedef union  mv_t             mv_t;
typedef struct cu_info_t        cu_info_t;

typedef struct outputframe_t    outputframe_t;


/* ---------------------------------------------------------------------------
 * SAOStatData
 */
typedef struct SAOBlkParam {
    int         mergeIdx;             // 0: merge_left, 1: merge_up, 2 not merge (new parameter)
    int         typeIdc;              // OFF(-1), EO_0, EO_90, EO_135, EO_45, BO
    int         startBand;            // BO: starting band index
    int         deltaBand;            // BO: third starting band distance
    int         offset[MAX_NUM_SAO_CLASSES];
} SAOBlkParam;


#endif  // XAVS2_BASIC_TYPES_H
