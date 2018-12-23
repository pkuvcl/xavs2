/*
 * win32thread.h
 *
 * Description of this file:
 *    windows threading of the xavs2 library
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

#ifndef XAVS2_WIN32THREAD_H
#define XAVS2_WIN32THREAD_H

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
/* the following macro is used within xavs2 encoder */
#undef ERROR

typedef struct {
    void *handle;
    void *(*func)(void *arg);
    void *arg;
    void *ret;
} xavs2_thread_t;
#define xavs2_thread_attr_t int

/* the conditional variable api for windows 6.0+ uses critical sections and not mutexes */
typedef CRITICAL_SECTION xavs2_thread_mutex_t;
#define XAVS2_PTHREAD_MUTEX_INITIALIZER {0}
#define xavs2_thread_mutexattr_t int
#define pthread_exit(a)

/* This is the CONDITIONAL_VARIABLE typedef for using Window's native conditional variables on kernels 6.0+.
 * MinGW does not currently have this typedef. */
typedef struct {
    xavs2_thread_mutex_t mtx_broadcast;
    xavs2_thread_mutex_t mtx_waiter_count;
    int waiter_count;
    HANDLE semaphore;
    HANDLE waiters_done;
    int is_broadcast;
} xavs2_win32_cond_t, xavs2_thread_cond_t;

#define xavs2_thread_condattr_t int

#define xavs2_thread_create FPFX(thread_create)
int xavs2_thread_create(xavs2_thread_t *thread, const xavs2_thread_attr_t *attr,
                        void *(*start_routine)(void *), void *arg);
#define xavs2_thread_join FPFX(thread_join)
int xavs2_thread_join(xavs2_thread_t thread, void **value_ptr);

#define xavs2_thread_mutex_init FPFX(thread_mutex_init)
int xavs2_thread_mutex_init(xavs2_thread_mutex_t *mutex, const xavs2_thread_mutexattr_t *attr);
#define xavs2_thread_mutex_destroy FPFX(thread_mutex_destroy)
int xavs2_thread_mutex_destroy(xavs2_thread_mutex_t *mutex);
#define xavs2_thread_mutex_lock FPFX(thread_mutex_lock)
int xavs2_thread_mutex_lock(xavs2_thread_mutex_t *mutex);
#define xavs2_thread_mutex_unlock FPFX(thread_mutex_unlock)
int xavs2_thread_mutex_unlock(xavs2_thread_mutex_t *mutex);

#define xavs2_thread_cond_init FPFX(thread_cond_init)
int xavs2_thread_cond_init(xavs2_thread_cond_t *cond, const xavs2_thread_condattr_t *attr);
#define xavs2_thread_cond_destroy FPFX(thread_cond_destroy)
int xavs2_thread_cond_destroy(xavs2_thread_cond_t *cond);
#define xavs2_thread_cond_broadcast FPFX(thread_cond_broadcast)
int xavs2_thread_cond_broadcast(xavs2_thread_cond_t *cond);
#define xavs2_thread_cond_wait FPFX(thread_cond_wait)
int xavs2_thread_cond_wait(xavs2_thread_cond_t *cond, xavs2_thread_mutex_t *mutex);
#define xavs2_thread_cond_signal FPFX(thread_cond_signal)
int xavs2_thread_cond_signal(xavs2_thread_cond_t *cond);

#define xavs2_thread_attr_init(a) 0
#define xavs2_thread_attr_destroy(a) 0

#define xavs2_win32_threading_init FPFX(win32_threading_init)
int  xavs2_win32_threading_init(void);
#define xavs2_win32_threading_destroy FPFX(win32_threading_destroy)
void xavs2_win32_threading_destroy(void);

#define xavs2_thread_num_processors_np FPFX(thread_num_processors_np)
int xavs2_thread_num_processors_np(void);

#endif  // XAVS2_WIN32THREAD_H
