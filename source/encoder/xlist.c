/*
 * xlist.c
 *
 * Description of this file:
 *    list structure with multi-thread support of the xavs2 library
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
#include "xlist.h"

#if !defined(_MSC_VER)
#include <errno.h>
#include <pthread.h>
#endif

/**
 * ===========================================================================
 * semaphore
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
int create_semaphore(semaphore_t *sem, void *attributes, int init_count, int max_count, const char *name)
{
#if (defined(__ICL) || defined(_MSC_VER)) && defined(_WIN32)
    *sem = CreateSemaphore((LPSECURITY_ATTRIBUTES)attributes, init_count, max_count, name);

    if (*sem == 0) {
        return GetLastError();
    } else {
        return 0;
    }

#else
    return sem_init(sem, 0, init_count);
#endif
}

/* ---------------------------------------------------------------------------
 */
int release_semaphore(semaphore_t *sem)
{
#if (defined(__ICL) || defined(_MSC_VER)) && defined(_WIN32)
    if (ReleaseSemaphore(*sem, 1, NULL) == TRUE) {
        return 0;
    } else {
        return errno;
    }

#else
    return sem_post(sem);
#endif
}

/* ---------------------------------------------------------------------------
 */
int close_semaphore(semaphore_t *sem)
{
#if (defined(__ICL) || defined(_MSC_VER)) && defined(_WIN32)
    if (CloseHandle(*sem) == TRUE) {
        return 0;
    } else {
        return errno;
    }

#else
    return sem_destroy(sem);
#endif
}

/**
 * ---------------------------------------------------------------------------
 * Function   : wait for a semaphore object
 * Parameters :
 *      [ in] : sem - handle of the semaphore
 *      [out] : none
 * Return     : value indicates the event that caused the function to return
 * ---------------------------------------------------------------------------
 */
int xavs2_wait_for_object(semaphore_t *sem)
{
#if (defined(__ICL) || defined(_MSC_VER)) && defined(_WIN32)
    return WaitForSingleObject((HANDLE)(*sem), INFINITE);
#else
    int ret;
    while ((ret = sem_wait((sem_t *)sem)) == -1 && errno == EINTR) {
        continue;
    }
    return ret;
#endif
}


/**
 * ===========================================================================
 * xlist
 * ===========================================================================
 */

/**
 * ---------------------------------------------------------------------------
 * Function   : initialize a list
 * Parameters :
 *      [in ] : xlist    - pointer to the node list
 *      [out] : none
 * Return     : zero for success, otherwise failed
 * Remarks    : also create 2 synchronous objects, but without any node
 * ---------------------------------------------------------------------------
 */
int xl_init(xlist_t *const xlist)
{
    if (xlist == NULL) {
        return -1;
    }

    /* set list empty */
    xlist->p_list_head = NULL;
    xlist->p_list_tail = NULL;

    /* set node number */
    xlist->i_node_num = 0;

    /* create semaphore */
    create_semaphore(&(xlist->list_sem), NULL, 0, INT_MAX, NULL);

    /* init list lock */
    SPIN_INIT(xlist->list_lock);

    return 0;
}

/**
 * ---------------------------------------------------------------------------
 * Function   : destroy a list
 * Parameters :
 *      [in ] : xlist - the list, pointer to struct xlist_t
 *      [out] : none
 * Return     : none
 * ---------------------------------------------------------------------------
 */
void xl_destroy(xlist_t *const xlist)
{
    if (xlist == NULL) {
        return;
    }

    /* destroy the spin lock */
    SPIN_DESTROY(xlist->list_lock);

    /* close handles */
    close_semaphore(&(xlist->list_sem));

    /* clear */
    memset(xlist, 0, sizeof(xlist_t));
}

/**
 * ---------------------------------------------------------------------------
 * Function   : append data to the tail of a list
 * Parameters :
 *      [in ] : xlist - the node list, pointer to struct xlist
 *            : data  - the data to append
 *      [out] : none
 * Return     : none
 * ---------------------------------------------------------------------------
 */
void xl_append(xlist_t *const xlist, void *node)
{
    node_t *new_node = (node_t *)node;

    if (xlist == NULL || new_node == NULL) {
        return;                       /* error */
    }

    new_node->next = NULL;            /* set NULL */

    /* append this node */
    SPIN_LOCK(xlist->list_lock);

    if (xlist->p_list_tail != NULL) {
        /* append this node at tail */
        xlist->p_list_tail->next = new_node;
    } else {
        xlist->p_list_head = new_node;
    }

    xlist->p_list_tail = new_node;    /* point to the tail node */
    xlist->i_node_num++;              /* increase the node number */
    SPIN_UNLOCK(xlist->list_lock);

    /* all is done, release a semaphore */
    release_semaphore(&(xlist->list_sem));
}

/**
 * ---------------------------------------------------------------------------
 * Function   : remove one node from the list's head position
 * Parameters :
 *      [in ] : xlist - the node list, pointer to struct xlist_t
 *            : wait  - wait the semaphore?
 *      [out] : none
 * Return     : node pointer for success, or NULL for failure
 * ---------------------------------------------------------------------------
 */
void *xl_remove_head(xlist_t *const xlist, const int wait)
{
    node_t *node = NULL;

    if (xlist == NULL) {
        return NULL;                  /* error */
    }

    if (wait) {
        xavs2_wait_for_object(&(xlist->list_sem));
    }

    SPIN_LOCK(xlist->list_lock);

    /* remove the header node */
    if (xlist->i_node_num > 0) {
        node = xlist->p_list_head;    /* point to the header node */

        /* modify the list */
        if (node != NULL) {
            xlist->p_list_head = node->next;
        }

        if (xlist->p_list_head == NULL) {
            /* there are no any node in this list, reset the tail pointer */
            xlist->p_list_tail = NULL;
        }

        xlist->i_node_num--;          /* decrease the number */
    }

    SPIN_UNLOCK(xlist->list_lock);

    return node;
}
