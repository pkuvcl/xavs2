/*
 * threadpool.c
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

#include "common.h"
#include "threadpool.h"
#include "cpu.h"


/**
 * ===========================================================================
 * type defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * job
 */
typedef struct threadpool_job_t {
    xavs2_tfunc_t   func;
    void           *arg;
    void           *ret;
    int             wait;
} threadpool_job_t;

/* ---------------------------------------------------------------------------
 * synchronized job list
 */
typedef struct xavs2_sync_job_list_t {
    xavs2_thread_mutex_t mutex;
    xavs2_thread_cond_t  cv_fill;  /* event signaling that the list became fuller */
    xavs2_thread_cond_t  cv_empty; /* event signaling that the list became emptier */
    int                   i_max_size;
    int                   i_size;
    threadpool_job_t     *list[XAVS2_THREAD_MAX + 1];
} xavs2_sync_job_list_t;

/* ---------------------------------------------------------------------------
 * thread pool
 */
struct xavs2_threadpool_t {
    int                   i_exit;       /* exit flag */
    int                   i_threads;    /* thread number in pool */
    xavs2_tfunc_t         init_func;
    void                 *init_arg;

    /* requires a synchronized list structure and associated methods,
       so use what is already implemented for jobs */
    xavs2_sync_job_list_t uninit;       /* list of jobs that are awaiting use */
    xavs2_sync_job_list_t run;          /* list of jobs that are queued for processing by the pool */
    xavs2_sync_job_list_t done;         /* list of jobs that have finished processing */

    /* handler of threads */
    xavs2_thread_t       thread_handle[XAVS2_THREAD_MAX];
    uint8_t               cpu_core_used[64];
};

/**
 * ===========================================================================
 * thread properties
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static INLINE
int  xavs2_thread_set_cpu(int idx_core)
{
#if HAVE_POSIXTHREAD && (SYS_WINDOWS || SYS_LINUX) && !__MINGW32__
    cpu_set_t mask;
    CPU_ZERO(&mask);

    CPU_SET(idx_core, &mask);

    if (-1 == pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask)) {
        return -1;
    }
    return 0;
#else
    return 0;
#endif
}

/* ---------------------------------------------------------------------------
 */
static INLINE
int  xavs2_thread_is_on_cpu(int idx_core)
{
#if HAVE_POSIXTHREAD && (SYS_WINDOWS || SYS_LINUX) && !__MINGW32__
    cpu_set_t get;

    CPU_ZERO(&get);
    if (pthread_getaffinity_np(pthread_self(), sizeof(get), &get) < 0) {
        fprintf(stderr, "get thread affinity failed\n");
    }
    return (CPU_ISSET(idx_core, &get));
#else
    return 0;
#endif
}

/**
 * ===========================================================================
 * list operators
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static threadpool_job_t *xavs2_job_shift(threadpool_job_t **list)
{
    threadpool_job_t *job = list[0];
    int i;

    for (i = 0; list[i]; i++) {
        list[i] = list[i + 1];
    }
    assert(job);

    return job;
}

/**
 * ===========================================================================
 * list operators
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static int xavs2_sync_job_list_init(xavs2_sync_job_list_t *slist, int i_max_size)
{
    if (i_max_size < 0 || i_max_size > XAVS2_THREAD_MAX) {
        return -1;
    }

    slist->i_max_size = i_max_size;
    slist->i_size     = 0;

    if (xavs2_thread_mutex_init(&slist->mutex,   NULL) ||
        xavs2_thread_cond_init(&slist->cv_fill,  NULL) ||
        xavs2_thread_cond_init(&slist->cv_empty, NULL)) {
        return -1;
    }

    return 0;
}

/* ---------------------------------------------------------------------------
 */
static void xavs2_sync_job_list_delete(xavs2_sync_job_list_t *slist)
{
    xavs2_thread_mutex_destroy(&slist->mutex);
    xavs2_thread_cond_destroy(&slist->cv_fill);
    xavs2_thread_cond_destroy(&slist->cv_empty);
}

/* ---------------------------------------------------------------------------
 */
static void xavs2_sync_job_list_push(xavs2_sync_job_list_t *slist, threadpool_job_t *job)
{
    xavs2_thread_mutex_lock(&slist->mutex);      /* lock */
    while (slist->i_size == slist->i_max_size) {
        xavs2_thread_cond_wait(&slist->cv_empty, &slist->mutex);
    }
    slist->list[slist->i_size++] = job;
    xavs2_thread_mutex_unlock(&slist->mutex);    /* unlock */

    xavs2_thread_cond_broadcast(&slist->cv_fill);
}

/* ---------------------------------------------------------------------------
 */
static threadpool_job_t *xavs2_sync_job_list_pop(xavs2_sync_job_list_t *slist)
{
    threadpool_job_t *job;

    xavs2_thread_mutex_lock(&slist->mutex);      /* lock */
    while (!slist->i_size) {
        xavs2_thread_cond_wait(&slist->cv_fill, &slist->mutex);
    }
    job = slist->list[--slist->i_size];
    slist->list[slist->i_size] = NULL;
    xavs2_thread_cond_broadcast(&slist->cv_empty);
    xavs2_thread_mutex_unlock(&slist->mutex);    /* unlock */

    return job;
}


/**
 * ===========================================================================
 * thread pool operators
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static
void *proc_xavs2_threadpool_thread(xavs2_threadpool_t *pool)
{
    /* init */
    if (pool->init_func) {
        pool->init_func(pool->init_arg);
    }

    /* loop until exit flag is set */
    while (pool->i_exit != XAVS2_EXIT_THREAD) {
        threadpool_job_t *job = NULL;

        /* fetch a job */
        xavs2_thread_mutex_lock(&pool->run.mutex);   /* lock */
        while (pool->i_exit != XAVS2_EXIT_THREAD && !pool->run.i_size) {
            xavs2_thread_cond_wait(&pool->run.cv_fill, &pool->run.mutex);
        }
        if (pool->run.i_size) {
            job = xavs2_job_shift(pool->run.list);
            pool->run.i_size--;
        }
        xavs2_thread_mutex_unlock(&pool->run.mutex); /* unlock */

        /* do the job */
        if (!job) {
            continue;
        }
        job->ret = job->func(job->arg); /* execute the function */

        /* the job is done */
        if (job->wait) {
            xavs2_sync_job_list_push(&pool->done, job);
        } else {
            xavs2_sync_job_list_push(&pool->uninit, job);
        }
    }

    return NULL;
}

/* ---------------------------------------------------------------------------
 */
int xavs2_threadpool_init(xavs2_threadpool_t **p_pool, int threads, xavs2_tfunc_t init_func, void *init_arg)
{
    xavs2_threadpool_t *pool;
    uint8_t *mem_ptr = NULL;
    int size_mem = 0;
    int i;

    if (threads <= 0) {
        return -1;
    }

    threads = XAVS2_MIN(threads, XAVS2_THREAD_MAX);
    size_mem = sizeof(xavs2_threadpool_t)  +
               threads * sizeof(threadpool_job_t) +
               CACHE_LINE_SIZE * XAVS2_THREAD_MAX * 2;

    CHECKED_MALLOCZERO(mem_ptr, uint8_t *, size_mem);
    pool          = (xavs2_threadpool_t *)mem_ptr;
    mem_ptr      += sizeof(xavs2_threadpool_t);
    ALIGN_POINTER(mem_ptr);

    *p_pool = pool;

    pool->init_func = init_func;
    pool->init_arg  = init_arg;
    pool->i_threads = threads;

    if (xavs2_sync_job_list_init(&pool->uninit, pool->i_threads) ||
        xavs2_sync_job_list_init(&pool->run,    pool->i_threads) ||
        xavs2_sync_job_list_init(&pool->done,   pool->i_threads)) {
        goto fail;
    }

    for (i = 0; i < pool->i_threads; i++) {
        threadpool_job_t *job = (threadpool_job_t *)mem_ptr;
        mem_ptr += sizeof(threadpool_job_t);
        ALIGN_POINTER(mem_ptr);

        xavs2_sync_job_list_push(&pool->uninit, job);
    }

    for (i = 0; i < pool->i_threads; i++) {
        if (xavs2_create_thread(pool->thread_handle + i, (xavs2_tfunc_t)proc_xavs2_threadpool_thread, pool)) {
            goto fail;
        }
    }

    return 0;

fail:
    return -1;
}

/* ---------------------------------------------------------------------------
 */
void xavs2_threadpool_run(xavs2_threadpool_t *pool, void *(*func)(void *), void *arg, int wait_sign)
{
    threadpool_job_t *job = xavs2_sync_job_list_pop(&pool->uninit);

    job->func = func;
    job->arg  = arg;
    job->wait = wait_sign;
    xavs2_sync_job_list_push(&pool->run, job);
}

/* ---------------------------------------------------------------------------
 */
void *xavs2_threadpool_wait(xavs2_threadpool_t *pool, void *arg)
{
    threadpool_job_t *job = NULL;
    void *ret;
    int i;

    xavs2_thread_mutex_lock(&pool->done.mutex);      /* lock */
    while (!job) {
        for (i = 0; i < pool->done.i_size; i++) {
            threadpool_job_t *t = pool->done.list[i];
            if (t->arg == arg) {
                job = xavs2_job_shift(pool->done.list + i);
                pool->done.i_size--;
                break;          /* found the job according to arg */
            }
        }
        if (!job) {
            xavs2_thread_cond_wait(&pool->done.cv_fill, &pool->done.mutex);
        }
    }
    xavs2_thread_mutex_unlock(&pool->done.mutex);    /* unlock */

    ret = job->ret;
    xavs2_sync_job_list_push(&pool->uninit, job);

    return ret;
}

/* ---------------------------------------------------------------------------
 */
void xavs2_threadpool_delete(xavs2_threadpool_t *pool)
{
    int i;

    xavs2_thread_mutex_lock(&pool->run.mutex);   /* lock */
    pool->i_exit = XAVS2_EXIT_THREAD;
    xavs2_thread_cond_broadcast(&pool->run.cv_fill);
    xavs2_thread_mutex_unlock(&pool->run.mutex); /* unlock */

    for (i = 0; i < pool->i_threads; i++) {
        xavs2_thread_join(pool->thread_handle[i], NULL);
    }

    xavs2_sync_job_list_delete(&pool->uninit);
    xavs2_sync_job_list_delete(&pool->run);
    xavs2_sync_job_list_delete(&pool->done);

    xavs2_free(pool);
}
