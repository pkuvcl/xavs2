/*
 * xavs2_api.c
 *
 * Description of this file:
 *    API wrapper for multi bit-depth
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
#include "wrapper.h"
#include "encoder.h"
#include "version.h"

/* ---------------------------------------------------------------------------
 * Macros
 */
#if SYS_WINDOWS
#define ext_dyn_lib "dll"
#elif SYS_MACOSX
#include <dlfcn.h>
#define ext_dyn_lib "dylib"
#else
#include <dlfcn.h>
#define ext_dyn_lib "so"
#endif

/* ---------------------------------------------------------------------------
 */
static xavs2_api_t api_default = {
    XVERSION_STR,
    VER_MAJOR * 10 + VER_MINOR,
    BIT_DEPTH,
    xavs2_encoder_opt_help,
    xavs2_encoder_opt_alloc,
    xavs2_encoder_opt_set,
    xavs2_encoder_opt_set2,
    xavs2_encoder_opt_get,
    xavs2_encoder_opt_destroy,
    xavs2_encoder_get_buffer,
    xavs2_encoder_create,
    xavs2_encoder_destroy,
    xavs2_encoder_encode,
    xavs2_encoder_packet_unref,
};

typedef const xavs2_api_t *(*xavs2_api_get_t)(int bit_depth);

/* ---------------------------------------------------------------------------
 */
static
const xavs2_api_t *xavs2_load_new_module(const char *dll_path, const char *methofd_name, int bit_depth)
{
    /* TODO: 在使用错误的库时, 会出现递归调用此函数最终导致崩溃 */
#if _WIN32
    HMODULE h = LoadLibraryA(dll_path);
    if (h) {
        xavs2_api_get_t get = (xavs2_api_get_t)GetProcAddress(h, methofd_name);
        if (get) {
            return get(bit_depth);
        }
    }
#else
    void* h = dlopen(dll_path, RTLD_LAZY | RTLD_LOCAL);
    if (h) {
        xavs2_api_get_t get = (xavs2_api_get_t)dlsym(h, methofd_name);
        if (get) {
            return get(bit_depth);
        }
    }
#endif
    xavs2_log(NULL, XAVS2_LOG_ERROR, "Failed to load library: %s, %d-bit \n", dll_path, bit_depth);
    return NULL;
}


/**
 * ---------------------------------------------------------------------------
 * Function   : get xavs2 APi handler
 * Parameters :
 *      [in ] : bit_depth - required bit-depth for encoding
 * Return     : NULL when failure
 * ---------------------------------------------------------------------------
 */
XAVS2_API const xavs2_api_t *
xavs2_api_get(int bit_depth)
{
    char s_lib_name[64];
    const char* method_name = "xavs2_api_get";

    switch (bit_depth) {
    case BIT_DEPTH:
        return &api_default;
    default:
        sprintf(s_lib_name, "libxavs2-%d-%dbit.%s", VER_MAJOR * 10 + VER_MINOR, bit_depth, ext_dyn_lib);
        return xavs2_load_new_module(s_lib_name, method_name, bit_depth);
    }
}
