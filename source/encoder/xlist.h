/*
 * xlist.h
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

#ifndef XAVS2_XLIST_H
#define XAVS2_XLIST_H


/**
 * ===========================================================================
 * type defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * node
 */
typedef struct node_t   node_t;
struct node_t {
    node_t      *next;                /* pointer to next node */
};

/* ---------------------------------------------------------------------------
 * xlist_t
 */
typedef struct xlist_t {
    node_t               *p_list_head;     /* pointer to head of node list */
    node_t               *p_list_tail;     /* pointer to tail of node list */
    xavs2_thread_cond_t  list_cond;       /* list condition variable */
    xavs2_thread_mutex_t list_mutex;      /* list mutex lock */
    int                   i_node_num;      /* node number in the list */
} xlist_t;


/**
 * ===========================================================================
 * interface function declares
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * xlist
 */
#define xl_init FPFX(xl_init)
int   xl_init(xlist_t *const xlist);
#define xl_destroy FPFX(xl_destroy)
void  xl_destroy(xlist_t *const xlist);
#define xl_append FPFX(xl_append)
void  xl_append(xlist_t *const xlist, void *node);
#define xl_remove_head FPFX(xl_remove_head)
void *xl_remove_head(xlist_t *const xlist, const int wait);

#endif  // XAVS2_XLIST_H
