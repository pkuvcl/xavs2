/*
 * threadpool.h
 *
 * Description of this file:
 *    thread pooling functions definition of the xavs2 library
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

#ifndef XAVS2_THREADPOOL_H
#define XAVS2_THREADPOOL_H

typedef struct xavs2_threadpool_t xavs2_threadpool_t;

#define xavs2_threadpool_init FPFX(threadpool_init)
int   xavs2_threadpool_init  (xavs2_threadpool_t **p_pool, int threads,
                              xavs2_tfunc_t init_func, void *init_arg);
#define xavs2_threadpool_run FPFX(threadpool_run)
void  xavs2_threadpool_run   (xavs2_threadpool_t *pool, void *(*func)(void *), void *arg, int wait_sign);
#define xavs2_threadpool_wait FPFX(threadpool_wait)
void *xavs2_threadpool_wait  (xavs2_threadpool_t *pool, void *arg);
#define xavs2_threadpool_delete FPFX(threadpool_delete)
void  xavs2_threadpool_delete(xavs2_threadpool_t *pool);

#endif  // XAVS2_THREADPOOL_H
