/*
 * wrapper.h
 *
 * Description of this file:
 *    encoder wrapper functions definition of the xavs2 library
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

#include "xlist.h"
#include "threadpool.h"

#ifndef __XAVS2_WRAPPER_H__
#define __XAVS2_WRAPPER_H__


/**
 * ===========================================================================
 * type defines
 * ===========================================================================
 */

// function type
typedef void(*vpp_ipred_t)(pel_t *p_pred, pel_t *p_top, pel_t *p_left);

/* ---------------------------------------------------------------------------
 * lookahead_t
 */
typedef struct lookahead_t {
    int         start;
    int         pframes;
    int         bpframes;
} lookahead_t;


/* ---------------------------------------------------------------------------
 * low resolution of frame (luma plane)
 */
typedef struct frm_lowres_t {
    int         i_width;              /* width  for luma plane */
    int         i_lines;              /* height for luma plane */
    int         i_stride;             /* stride for luma plane */
    pel_t      *filtered;             /* half-size copy of input frame (luma only) */
} frm_lowres_t;

/* ---------------------------------------------------------------------------
 * video pre-processing motion estimation 
 */
typedef struct vpp_me_t {
    int             mv_min[2];        /* full pel MV range for motion search (min) */
    int             mv_max[2];        /* full pel MV range for motion search (max) */
    mv_t            bmv;              /* [OUT] best motion vector */
    mv_t            pmv;              /* pred motion vector for the current block */
    uint16_t       *mvbits;           /* used for getting the mv bits */
    pixel_cmp_t     sad_8x8;          /* function handle for cal sad of 8x8 block */
    pixel_cmp_x3_t  sad_8x8_x3;       /* function handle for cal sad of 8x8 block (X3) */
    pixel_cmp_x4_t  sad_8x8_x4;       /* function handle for cal sad of 8x8 block (X4) */
} vpp_me_t;

/* ---------------------------------------------------------------------------
 * frame buffer manager
 */
struct xavs2_frame_buffer_t {
    xavs2_frame_t   *frames[FREF_BUF_SIZE];  /* all managed pictures */
    int              num_frames;             /* number of managed pictures */
    int              COI;                    /* Coding Order Index */
    int              COI_IDR;                /* COI of current IDR frame */
    int              POC_IDR;                /* POC of current IDR frame */
    int              ip_pic_idx;           /* encoded I/P/F-picture index (to be REMOVED) */
    int              i_frame_b;            /* number of encoded B-picture in a GOP */

    /* frames to be removed before next frame encoding */
    int         num_frames_to_remove; /* number of frames to be removed */
    int         coi_remove_frame[8];  /* COI of frames to be removed */
};

/* ---------------------------------------------------------------------------
 * xavs2_handler_t
 */
struct xavs2_handler_t {
    /* encoder engines */
    xavs2_t    *p_coder;                            /* point to the xavs2 video encoder */
    xavs2_t    *frm_contexts[MAX_PARALLEL_FRAMES];  /* frame task contexts */
    xavs2_t    *row_contexts;                       /* row   task contexts */

    /* frame buffers */
    xavs2_frame_buffer_t ipb;         /* input picture buffer */
    xavs2_frame_buffer_t dpb;         /* decoding picture buffer */

    /* properties */
    int64_t     max_out_pts;          /* max output pts */
    int64_t     max_out_dts;          /* max output dts */

    /* number of frames */
    int         num_input;            /* number of frames: input into the encoder */
    int         num_encode;           /* number of frames: sent into encoding queue */
    int         num_output;           /* number of frames: outputted */
    int         b_seq_end;            /* has all frames been output */

    /* output frame index, use get_next_frame_id() to get next output index */
    int         i_input;              /* index  of frames: input  already accepted, used for frame output () */
    int         i_output;             /* index  of frames: output already encoded , used for frame output () */

    /* index of frames, [0, i_frm_threads), to determine frame order */
    int         i_frame_in;           /* frame order [0, i_frm_threads): next input  */
    int         i_frame_aec;          /* frame order [0, i_frm_threads): current AEC */

    /* threads & synchronization */
    volatile int          i_exit_flag;        /* app signal to exit */
    int                   i_frm_threads;      /* real number of thread in frame   level parallel */
    int                   i_row_threads;      /* real number of thread in LCU-row level parallel */
    int                   num_pool_threads;   /* number of threads allocated in threadpool */
    int                   num_row_contexts;   /* number of row contexts */
    xavs2_threadpool_t   *threadpool_rdo;     /* the thread pool (for parallel encoding) */
    xavs2_threadpool_t   *threadpool_aec;     /* the thread pool for aec encoding */
    xavs2_pthread_t       thread_wrapper;     /* thread for wrapper proceeding */

    xavs2_pthread_cond_t  cond[SIG_COUNT];
    xavs2_pthread_mutex_t mutex;              /* mutex */


    /* frames and lists */
    xlist_t         list_frames_free;         /* list[0]: frames which are free to use */
    xlist_t         list_frames_ready;        /* list[1]: frames which are ready for encoding (slice type configured) */
#if XAVS2_API_VERSION >= 2
    xlist_t         list_frames_output;       /* list[2]: frames which are ready for output */
#endif

    /* lookahead and slice type decision */
    xavs2_frame_t  *blocked_frm_set[XAVS2_MAX_GOP_SIZE + 4];
    int64_t         blocked_pts_set[XAVS2_MAX_GOP_SIZE + 4];
    int64_t         prev_reordered_pts_set[XAVS2_MAX_GOP_SIZE + 4];
    int             num_encoded_frames_for_dts;
    lookahead_t     lookahead;                /* lookahead */
    int             index_in_gop;

    /* rate-control */
    ratectrl_t     *rate_control;            /* rate control */
    td_rdo_t       *td_rdo;

#if XAVS2_STAT
    xavs2_stat_t      stat;           /* stat total */
    FILE             *fp_trace;       /* for trace output */
#endif

#if XAVS2_API_VERSION < 2
    xavs2_dump_func_t dump_func;      /* handle of dump function, called inside */
#endif
    void             *user_data;      /* handle of user data */
    int64_t           create_time;    /* time of encoder creation, used for encoding speed test */

#if XAVS2_DUMP_REC
    FILE             *h_rec_file;     /* file handle to output reconstructed frame data */
#endif
};

/**
 * ===========================================================================
 * inline functions
 * ===========================================================================
 */
static ALWAYS_INLINE
int get_next_frame_id(int idx_cur)
{
    idx_cur = (idx_cur + 1);
    if (idx_cur > MAX_FRAME_INDEX) {
        return 0;
    } else {
        return idx_cur;
    }
}


/**
 * ===========================================================================
 * interface function declares
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * frame buffer operation
 */
void frame_buffer_init(xavs2_handler_t *h_mgr, uint8_t **mem_base, xavs2_frame_buffer_t *frm_buf, int num_frm, int frm_type);
void frame_buffer_destroy(xavs2_handler_t *h_mgr, xavs2_frame_buffer_t *frm_buf);

void frame_buffer_update(xavs2_t *h, xavs2_frame_buffer_t *frm_buf, xavs2_frame_t *frm);

/* ---------------------------------------------------------------------------
 * wrapper
 */
void destroy_all_lists(xavs2_handler_t *h_mgr);

void encoder_task_manager_free(xavs2_handler_t *h_mgr);

void *proc_wrapper_thread(void *args);


#endif  // __XAVS2_WRAPPER_H__
