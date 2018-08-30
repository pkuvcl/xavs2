/*
 * presets.h
 *
 * Description of this file:
 *    parse preset level functions definition of the xavs2 library
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


#ifndef XAVS2_PRESETS_H
#define XAVS2_PRESETS_H

#define parse_preset_level FPFX(parse_preset_level)
void parse_preset_level(xavs2_param_t *p_param, int i_preset_level);
#define encoder_set_fast_algorithms FPFX(encoder_set_fast_algorithms)
void encoder_set_fast_algorithms(xavs2_t *h);
#define decide_ultimate_paramters FPFX(decide_ultimate_paramters)
void decide_ultimate_paramters(xavs2_param_t *p_param);

#endif  // XAVS2_PRESET_LEVELS_H
