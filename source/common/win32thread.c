/*
 * win32thread.c
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

/* Microsoft's way of supporting systems with >64 logical cpus can be found at
 * http://www.microsoft.com/whdc/system/Sysinternals/MoreThan64proc.mspx */

/* Based on the agreed standing that xavs2 encoder does not need to utilize >64 logical cpus,
 * this API does not detect nor utilize more than 64 cpus for systems that have them. */


#include "common.h"

#if HAVE_WIN32THREAD
#include <process.h>

/**
 * ===========================================================================
 * type defines
 * ===========================================================================
 */

/* number of times to spin a thread about to block on a locked mutex before retrying and sleeping if still locked */
#define XAVS2_SPIN_COUNT 0

/* GROUP_AFFINITY struct */
typedef struct {
    ULONG_PTR   mask;           // KAFFINITY = ULONG_PTR
    USHORT      group;
    USHORT      reserved[3];
} xavs2_group_affinity_t;

typedef void (WINAPI *cond_func_t)(xavs2_thread_cond_t *cond);
typedef BOOL (WINAPI *cond_wait_t)(xavs2_thread_cond_t *cond, xavs2_thread_mutex_t *mutex, DWORD milliseconds);

typedef struct {
    /* global mutex for replacing MUTEX_INITIALIZER instances */
    xavs2_thread_mutex_t static_mutex;

    /* function pointers to conditional variable API on windows 6.0+ kernels */
    cond_func_t cond_broadcast;
    cond_func_t cond_init;
    cond_func_t cond_signal;
    cond_wait_t cond_wait;
} xavs2_win32thread_control_t;

static xavs2_win32thread_control_t thread_control;


/**
 * ===========================================================================
 * function defines
 * ===========================================================================
 */

/* _beginthreadex requires that the start routine is __stdcall */
static unsigned __stdcall xavs2_win32thread_worker(void *arg)
{
    xavs2_thread_t *h = arg;

    h->ret = h->func(h->arg);

    return 0;
}

int xavs2_thread_create(xavs2_thread_t *thread, const xavs2_thread_attr_t *attr,
                        void *(*start_routine)(void *), void *arg)
{
    UNUSED_PARAMETER(attr);

    thread->func   = start_routine;
    thread->arg    = arg;
    thread->handle = (void *)_beginthreadex(NULL, 0, xavs2_win32thread_worker, thread, 0, NULL);
    return !thread->handle;
}

int xavs2_thread_join(xavs2_thread_t thread, void **value_ptr)
{
    DWORD ret = WaitForSingleObject(thread.handle, INFINITE);

    if (ret != WAIT_OBJECT_0) {
        return -1;
    }
    if (value_ptr) {
        *value_ptr = thread.ret;
    }
    CloseHandle(thread.handle);

    return 0;
}

int xavs2_thread_mutex_init(xavs2_thread_mutex_t *mutex, const xavs2_thread_mutexattr_t *attr)
{
    UNUSED_PARAMETER(attr);
    return !InitializeCriticalSectionAndSpinCount(mutex, XAVS2_SPIN_COUNT);
}

int xavs2_thread_mutex_destroy(xavs2_thread_mutex_t *mutex)
{
    DeleteCriticalSection(mutex);
    return 0;
}

int xavs2_thread_mutex_lock(xavs2_thread_mutex_t *mutex)
{
    static xavs2_thread_mutex_t init = XAVS2_PTHREAD_MUTEX_INITIALIZER;

    if (!memcmp(mutex, &init, sizeof(xavs2_thread_mutex_t))) {
        *mutex = thread_control.static_mutex;
    }
    EnterCriticalSection(mutex);

    return 0;
}

int xavs2_thread_mutex_unlock(xavs2_thread_mutex_t *mutex)
{
    LeaveCriticalSection(mutex);
    return 0;
}

int xavs2_thread_cond_init(xavs2_thread_cond_t *cond, const xavs2_thread_condattr_t *attr)
{
    xavs2_win32_cond_t *win32_cond = cond;

    UNUSED_PARAMETER(attr);

    if (thread_control.cond_init) {
        thread_control.cond_init(cond);
        return 0;
    }

    /* non native condition variables */
    memset(win32_cond, 0, sizeof(xavs2_win32_cond_t));
    win32_cond->semaphore = CreateSemaphore(NULL, 0, 0x7fffffff, NULL);
    if (!win32_cond->semaphore) {
        return -1;
    }

    if (xavs2_thread_mutex_init(&win32_cond->mtx_waiter_count, NULL)) {
        return -1;
    }
    if (xavs2_thread_mutex_init(&win32_cond->mtx_broadcast, NULL)) {
        return -1;
    }

    win32_cond->waiters_done = CreateEvent(NULL, FALSE, FALSE, NULL);
    if (!win32_cond->waiters_done) {
        return -1;
    }

    return 0;
}

int xavs2_thread_cond_destroy(xavs2_thread_cond_t *cond)
{
    xavs2_win32_cond_t *win32_cond = cond;

    /* native condition variables do not destroy */
    if (thread_control.cond_init) {
        return 0;
    }

    /* non native condition variables */
    CloseHandle(win32_cond->semaphore);
    CloseHandle(win32_cond->waiters_done);
    xavs2_thread_mutex_destroy(&win32_cond->mtx_broadcast);
    xavs2_thread_mutex_destroy(&win32_cond->mtx_waiter_count);

    return 0;
}

int xavs2_thread_cond_broadcast(xavs2_thread_cond_t *cond)
{
    xavs2_win32_cond_t *win32_cond = cond;
    int have_waiter = 0;

    if (thread_control.cond_broadcast) {
        thread_control.cond_broadcast(cond);
        return 0;
    }

    /* non native condition variables */
    xavs2_thread_mutex_lock(&win32_cond->mtx_broadcast);
    xavs2_thread_mutex_lock(&win32_cond->mtx_waiter_count);

    if (win32_cond->waiter_count) {
        win32_cond->is_broadcast = 1;
        have_waiter = 1;
    }

    if (have_waiter) {
        ReleaseSemaphore(win32_cond->semaphore, win32_cond->waiter_count, NULL);
        xavs2_thread_mutex_unlock(&win32_cond->mtx_waiter_count);
        WaitForSingleObject(win32_cond->waiters_done, INFINITE);
        win32_cond->is_broadcast = 0;
    } else {
        xavs2_thread_mutex_unlock(&win32_cond->mtx_waiter_count);
    }

    return xavs2_thread_mutex_unlock(&win32_cond->mtx_broadcast);
}

int xavs2_thread_cond_signal(xavs2_thread_cond_t *cond)
{
    xavs2_win32_cond_t *win32_cond = cond;
    int have_waiter;

    if (thread_control.cond_signal) {
        thread_control.cond_signal(cond);
        return 0;
    }

    /* non-native condition variables */
    xavs2_thread_mutex_lock(&win32_cond->mtx_broadcast);
    xavs2_thread_mutex_lock(&win32_cond->mtx_waiter_count);
    have_waiter = win32_cond->waiter_count;
    xavs2_thread_mutex_unlock(&win32_cond->mtx_waiter_count);

    if (have_waiter) {
        ReleaseSemaphore(win32_cond->semaphore, 1, NULL);
    }

    return xavs2_thread_mutex_unlock(&win32_cond->mtx_broadcast);
}

int xavs2_thread_cond_wait(xavs2_thread_cond_t *cond, xavs2_thread_mutex_t *mutex)
{
    xavs2_win32_cond_t *win32_cond = cond;
    int last_waiter;

    if (thread_control.cond_wait) {
        return !thread_control.cond_wait(cond, mutex, INFINITE);
    }

    /* non native condition variables */
    xavs2_thread_mutex_lock(&win32_cond->mtx_broadcast);
    xavs2_thread_mutex_lock(&win32_cond->mtx_waiter_count);
    win32_cond->waiter_count++;
    xavs2_thread_mutex_unlock(&win32_cond->mtx_waiter_count);
    xavs2_thread_mutex_unlock(&win32_cond->mtx_broadcast);

    // unlock the external mutex
    xavs2_thread_mutex_unlock(mutex);
    WaitForSingleObject(win32_cond->semaphore, INFINITE);

    xavs2_thread_mutex_lock(&win32_cond->mtx_waiter_count);
    win32_cond->waiter_count--;
    last_waiter = !win32_cond->waiter_count || !win32_cond->is_broadcast;
    xavs2_thread_mutex_unlock(&win32_cond->mtx_waiter_count);

    if (last_waiter) {
        SetEvent(win32_cond->waiters_done);
    }

    // lock the external mutex
    return xavs2_thread_mutex_lock(mutex);
}

int xavs2_win32_threading_init(void)
{
    /* find function pointers to API functions, if they exist */
    HMODULE kernel_dll = GetModuleHandle(TEXT("kernel32"));

    thread_control.cond_init = (cond_func_t)GetProcAddress(kernel_dll, "InitializeConditionVariable");
    if (thread_control.cond_init) {
        /* we're on a windows 6.0+ kernel, acquire the rest of the functions */
        thread_control.cond_broadcast = (cond_func_t)GetProcAddress(kernel_dll, "WakeAllConditionVariable");
        thread_control.cond_signal = (cond_func_t)GetProcAddress(kernel_dll, "WakeConditionVariable");
        thread_control.cond_wait = (cond_wait_t)GetProcAddress(kernel_dll, "SleepConditionVariableCS");
    }
    return xavs2_thread_mutex_init(&thread_control.static_mutex, NULL);
}

void xavs2_win32_threading_destroy(void)
{
    xavs2_thread_mutex_destroy(&thread_control.static_mutex);
    memset(&thread_control, 0, sizeof(xavs2_win32thread_control_t));
}

int xavs2_thread_num_processors_np()
{
    DWORD_PTR system_cpus, process_cpus = 0;
    int cpus = 0;
    DWORD_PTR bit;

    /* GetProcessAffinityMask returns affinities of 0 when the process has threads in multiple processor groups.
     * On platforms that support processor grouping, use GetThreadGroupAffinity to get the current thread's affinity instead. */
#if ARCH_X86_64
    /* find function pointers to API functions specific to x86_64 platforms, if they exist.
     * BOOL GetThreadGroupAffinity(_In_  HANDLE hThread, _Out_ PGROUP_AFFINITY GroupAffinity); */
    typedef BOOL(*get_thread_affinity_t)(HANDLE thread, xavs2_group_affinity_t *group_affinity);
    HANDLE kernel_dll = GetModuleHandle(TEXT("kernel32.dll"));
    get_thread_affinity_t get_thread_affinity = (get_thread_affinity_t)GetProcAddress(kernel_dll, "GetThreadGroupAffinity");

    if (get_thread_affinity) {
        /* running on a platform that supports >64 logical cpus */
        xavs2_group_affinity_t thread_affinity;
        if (get_thread_affinity(GetCurrentThread(), &thread_affinity)) {
            process_cpus = thread_affinity.mask;
        }
    }
#endif
    if (!process_cpus) {
        GetProcessAffinityMask(GetCurrentProcess(), &process_cpus, &system_cpus);
    }

    for (bit = 1; bit; bit <<= 1) {
        cpus += !!(process_cpus & bit);
    }

    return cpus ? cpus : 1;
}

#endif // #if HAVE_WIN32THREAD
