/*
 * primitives.c
 *
 * Description of this file:
 *    function handles initialize functions definition of the xavs2 library
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

#include "common.h"
#include "primitives.h"
#include "cpu.h"
#include "intra.h"
#include "mc.h"
#include "transform.h"
#include "filter.h"
#include "sao.h"

/* ---------------------------------------------------------------------------
 * global function handle
 */
intrinsic_func_t g_funcs;

/* ---------------------------------------------------------------------------
 */
void xavs2_init_all_primitives(xavs2_param_t* param, intrinsic_func_t *p_funcs)
{
    uint32_t cpuid = p_funcs->cpuid;

    if (param != NULL) {
        if (param->sample_bit_depth != g_bit_depth) {
            xavs2_log(NULL, XAVS2_LOG_ERROR, "init primitives error: only %d bit-depth is supported\n", g_bit_depth);
        }
    }

    /* init memory operation function handlers */
    xavs2_mem_oper_init  (cpuid, p_funcs);

    /* init function handles */
    xavs2_intra_pred_init(cpuid, p_funcs);
    xavs2_mc_init        (cpuid, p_funcs);
    xavs2_pixel_init     (cpuid, &p_funcs->pixf);
    xavs2_deblock_init   (cpuid, p_funcs);
    xavs2_dct_init       (cpuid, &p_funcs->dctf);
    xavs2_quant_init     (cpuid, &p_funcs->dctf);
    xavs2_cg_scan_init   (cpuid, p_funcs);
    xavs2_mad_init       (cpuid, p_funcs->pixf.madf);

    xavs2_sao_init       (cpuid, p_funcs);
    xavs2_alf_init       (cpuid, p_funcs);

    xavs2_rdo_init       (cpuid, p_funcs);
}
