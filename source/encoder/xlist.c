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

    /* create lock and conditions */
    if (xavs2_thread_mutex_init(&xlist->list_mutex, NULL) < 0 ||
        xavs2_thread_cond_init(&xlist->list_cond, NULL) < 0) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Failed to init lock for xl_init()");
        return -1;
    }

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

    /* destroy lock and conditions */
    xavs2_thread_mutex_destroy(&xlist->list_mutex);
    xavs2_thread_cond_destroy(&xlist->list_cond);

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

    xavs2_thread_mutex_lock(&xlist->list_mutex);   /* lock */

    /* append this node */
    if (xlist->p_list_tail != NULL) {
        /* append this node at tail */
        xlist->p_list_tail->next = new_node;
    } else {
        xlist->p_list_head = new_node;
    }

    xlist->p_list_tail = new_node;    /* point to the tail node */
    xlist->i_node_num++;              /* increase the node number */

    xavs2_thread_mutex_unlock(&xlist->list_mutex);  /* unlock */

    /* all is done, notify one waiting thread to work */
    xavs2_thread_cond_signal(&xlist->list_cond);
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

    xavs2_thread_mutex_lock(&xlist->list_mutex);

    while (wait && !xlist->i_node_num) {
        xavs2_thread_cond_wait(&xlist->list_cond, &xlist->list_mutex);
    }

    /* remove the header node */
    if (xlist->i_node_num > 0) {
        node = xlist->p_list_head;    /* point to the header node */

        /* modify the list */
        xlist->p_list_head = node->next;

        if (xlist->p_list_head == NULL) {
            /* there are no any node in this list, reset the tail pointer */
            xlist->p_list_tail = NULL;
        }

        xlist->i_node_num--;          /* decrease the number */
    }

    xavs2_thread_mutex_unlock(&xlist->list_mutex);

    return node;
}
