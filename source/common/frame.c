/*
 * frame.c
 *
 * Description of this file:
 *    Frame handling functions definition of the xavs2 library
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
#include "frame.h"

/**
 * ===========================================================================
 * macro defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * pointer */
#define get_plane_ptr(...) \
    MULTI_LINE_MACRO_BEGIN\
    if (get_plane_pointer(__VA_ARGS__) < 0) {\
        return -1;\
    }\
    MULTI_LINE_MACRO_END


/**
 * ===========================================================================
 * memory handling
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE int
align_stride(int x, int align, int disalign)
{
    x = XAVS2_ALIGN(x, align);
    if (!(x & (disalign - 1))) {
        x += align;
    }
    return x;
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE int
align_plane_size(int x, int disalign)
{
    if (!(x & (disalign - 1))) {
        x += 128;
    }
    return x;
}

/* ---------------------------------------------------------------------------
 */
size_t xavs2_frame_buffer_size(const xavs2_param_t *param, int alloc_type)
{
    int img_w_l  = ((param->org_width  + MIN_CU_SIZE - 1) >> MIN_CU_SIZE_IN_BIT) << MIN_CU_SIZE_IN_BIT;
    int img_h_l  = ((param->org_height + MIN_CU_SIZE - 1) >> MIN_CU_SIZE_IN_BIT) << MIN_CU_SIZE_IN_BIT;
    int img_w_c  = img_w_l >> (param->chroma_format <= CHROMA_420 ? 1 : 0);
    int img_h_c  = img_h_l >> (param->chroma_format <= CHROMA_420 ? 1 : 0);
    int align    = 32;
    int disalign = 1 << 16;
    int stride_l, stride_c;
    int size_l, size_c;         /* size of luma and chroma plane */
    int i_nal_info_size = 0;
    int mem_size;               /* total memory size */
    int planes_size;
    int bs_size  = 0;           /* reuse the YUV plane space */
    int cmp_size = 0;           /* size of frame complexity buffer */
    int cmp_buf_size = 0;       /* complexity buffer size */
#if SAVE_CU_INFO
    int frame_size_in_mincu = 0;
#endif
    int frame_size_in_mvstore = 0;  /* reference information size */

    /* compute stride and the plane size */
    switch (alloc_type) {
    case FT_DEC:
        /* +PAD for extra data for me */
        stride_l = align_stride(img_w_l + ((XAVS2_PAD << 1)     ), align, disalign);
        stride_c = align_stride(img_w_c + ((XAVS2_PAD >> 1) << 1), align, disalign);
        size_l   = align_plane_size(stride_l * (img_h_l + ((XAVS2_PAD << 1)     ) + 1), disalign);
        size_c   = align_plane_size(stride_c * (img_h_c + ((XAVS2_PAD >> 1) << 1) + 1), disalign);
#if SAVE_CU_INFO
        frame_size_in_mincu = (img_w_l >> MIN_CU_SIZE_IN_BIT) * (img_h_l >> MIN_CU_SIZE_IN_BIT);
#endif
        frame_size_in_mvstore = (((img_w_l >> MIN_PU_SIZE_IN_BIT) + 3) >> 2) * (((img_h_l >> MIN_PU_SIZE_IN_BIT) + 3) >> 2);
        planes_size = size_l + size_c * 2;
#if ENABLE_FRAME_SUBPEL_INTPL
        planes_size += size_l * 15;
#endif
        break;
    case FT_TEMP:
        /* +PAD for extra data for me */
        stride_l = align_stride(img_w_l + ((XAVS2_PAD << 1)     ), align, disalign);
        stride_c = align_stride(img_w_c + ((XAVS2_PAD >> 1) << 1), align, disalign);
        size_l   = align_plane_size(stride_l * (img_h_l + ((XAVS2_PAD << 1)     ) + 1), disalign);
        size_c   = align_plane_size(stride_c * (img_h_c + ((XAVS2_PAD >> 1) << 1) + 1), disalign);
        planes_size = size_l + size_c * 2;
        break;
    default:
        stride_l = align_stride(img_w_l, align, disalign);
        stride_c = align_stride(img_w_c, align, disalign);
        size_l   = align_plane_size(stride_l * img_h_l, disalign);
        size_c   = align_plane_size(stride_c * img_h_c, disalign);
        planes_size = size_l + size_c * 2;
    }

    if (alloc_type == FT_ENC) {
#if XAVS2_ADAPT_LAYER
        i_nal_info_size = (param->slice_num + 6) * sizeof(xavs2_nal_info_t);
#endif
        bs_size         = size_l * sizeof(uint8_t);    /* let the PSNR compute correctly */
    }

    /* compute space size and alloc memory */
    mem_size = sizeof(xavs2_frame_t)                + /* M0, size of frame handle */
               i_nal_info_size                             + /* M1, size of nal_info buffer */
               cmp_size + cmp_buf_size                     + /* M2, size of frame complexity buffer */
               bs_size                                     + /* M3, size of bitstream buffer */
               planes_size * sizeof(pel_t)                 + /* M4, size of planes buffer: Y+U+V */
               frame_size_in_mvstore * sizeof(int8_t)      + /* M5, size of pu reference index buffer */
               frame_size_in_mvstore * sizeof(mv_t)        + /* M6, size of pu motion vector buffer */
#if SAVE_CU_INFO
               frame_size_in_mincu * sizeof(int8_t) * 3    + /* M7, size of cu mode/cbp/level buffers */
#endif
               (img_h_l >> MIN_CU_SIZE_IN_BIT) * sizeof(int)+ /* M8, line status array */
               CACHE_LINE_SIZE * 10;

    /* align to CACHE_LINE_SIZE */
    mem_size = (mem_size + CACHE_LINE_SIZE - 1) & (~(uint32_t)(CACHE_LINE_SIZE - 1));

    return mem_size;
}

/* ---------------------------------------------------------------------------
 */
xavs2_frame_t *xavs2_frame_new(xavs2_t *h, uint8_t **mem_base, int alloc_type)
{
    xavs2_frame_t *frame;
    int img_w_l  = h->i_width;
    int img_h_l  = h->i_height;
    int img_w_c  = img_w_l >> (h->param->chroma_format <= CHROMA_420 ? 1 : 0);
    int img_h_c  = img_h_l >> (h->param->chroma_format <= CHROMA_420 ? 1 : 0);
    int align    = 32;
    int disalign = 1 << 16;
    int stride_l, stride_c;
    int size_l, size_c;         /* size of luma and chroma plane */
    int i_nal_info_size = 0;
    int mem_size;               /* total memory size */
    int planes_size, i;
    int bs_size  = 0;           /* reuse the YUV plane space */
    int cmp_size = 0;           /* size of frame complexity buffer */
    int cmp_buf_size = 0;       /* complexity buffer size */
#if SAVE_CU_INFO
    int frame_size_in_mincu = 0;
#endif
    int frame_size_in_mvstore = 0;  /* reference information size */
    uint8_t *mem_ptr;

    /* compute stride and the plane size */
    switch (alloc_type) {
    case FT_DEC:
        /* +PAD for extra data for me */
        stride_l = align_stride(img_w_l + ((XAVS2_PAD << 1)     ), align, disalign);
        stride_c = align_stride(img_w_c + ((XAVS2_PAD >> 1) << 1), align, disalign);
        size_l   = align_plane_size(stride_l * (img_h_l + ((XAVS2_PAD << 1)     ) + 1), disalign);
        size_c   = align_plane_size(stride_c * (img_h_c + ((XAVS2_PAD >> 1) << 1) + 1), disalign);
#if SAVE_CU_INFO
        frame_size_in_mincu = h->i_width_in_mincu * h->i_height_in_mincu;
#endif
        frame_size_in_mvstore = ((h->i_width_in_minpu + 3) >> 2) * ((h->i_height_in_minpu + 3) >> 2);
        planes_size = size_l + size_c * 2;
#if ENABLE_FRAME_SUBPEL_INTPL
        if (h->use_fractional_me == 1) {
            planes_size += size_l * 3;
        } else if (h->use_fractional_me == 2) {
            planes_size += size_l * 15;
        }
#endif
        break;
    case FT_TEMP:  /* for SAO and ALF */
        /* +PAD for extra data for me */
        stride_l = align_stride(img_w_l + ((XAVS2_PAD << 1)     ), align, disalign);
        stride_c = align_stride(img_w_c + ((XAVS2_PAD >> 1) << 1), align, disalign);
        size_l   = align_plane_size(stride_l * (img_h_l + ((XAVS2_PAD << 1)     ) + 1), disalign);
        size_c   = align_plane_size(stride_c * (img_h_c + ((XAVS2_PAD >> 1) << 1) + 1), disalign);
        planes_size = size_l + size_c * 2;
        break;
    default:
        stride_l = align_stride(img_w_l, align, disalign);
        stride_c = align_stride(img_w_c, align, disalign);
        size_l   = align_plane_size(stride_l * img_h_l, disalign);
        size_c   = align_plane_size(stride_c * img_h_c, disalign);
        planes_size = size_l + size_c * 2;
    }

    if (alloc_type == FT_ENC) {
#if XAVS2_ADAPT_LAYER
        i_nal_info_size = (h->param->slice_num + 6) * sizeof(xavs2_nal_info_t);
#endif
        bs_size         = size_l * sizeof(uint8_t);    /* let the PSNR compute correctly */
    }

    /* compute space size and alloc memory */
    mem_size = sizeof(xavs2_frame_t)                + /* M0, size of frame handle */
               i_nal_info_size                             + /* M1, size of nal_info buffer */
               cmp_size + cmp_buf_size                     + /* M2, size of frame complexity buffer */
               bs_size                                     + /* M3, size of bitstream buffer */
               planes_size * sizeof(pel_t)                 + /* M4, size of planes buffer: Y+U+V */
               frame_size_in_mvstore * sizeof(int8_t)      + /* M5, size of pu reference index buffer */
               frame_size_in_mvstore * sizeof(mv_t)        + /* M6, size of pu motion vector buffer */
#if SAVE_CU_INFO
               frame_size_in_mincu * sizeof(int8_t) * 3    + /* M7, size of cu mode/cbp/level buffers */
#endif
               h->i_height_in_lcu * sizeof(int)            + /* M8, line status array */
               CACHE_LINE_SIZE * 10;

    /* align to CACHE_LINE_SIZE */
    mem_size = (mem_size + CACHE_LINE_SIZE - 1) & (~(uint32_t)(CACHE_LINE_SIZE - 1));

    if (mem_base == NULL) {
        CHECKED_MALLOC(mem_ptr, uint8_t *, mem_size);
    } else {
        mem_ptr = *mem_base;
    }

    /* M0, frame handle */
    frame    = (xavs2_frame_t *)mem_ptr;
    mem_ptr += sizeof(xavs2_frame_t);
    ALIGN_POINTER(mem_ptr);

    /* set frame properties */
    frame->i_plane     = 3;           /* planes: Y+U+V */
    frame->i_width [0] = img_w_l;
    frame->i_lines [0] = img_h_l;
    frame->i_stride[0] = stride_l;
    frame->i_width [1] = frame->i_width [2] = img_w_c;
    frame->i_lines [1] = frame->i_lines [2] = img_h_c;
    frame->i_stride[1] = frame->i_stride[2] = stride_c;

    /* the default setting of a frame */
    frame->i_frame   = -1;
    frame->i_frm_coi = -1;
    frame->i_gop_idr_coi = -1;

    if (h->param->chroma_format == CHROMA_400) {
        frame->i_plane = 1;
    }

    frame->i_frm_type = XAVS2_TYPE_AUTO;
    frame->i_pts  = -1;
    frame->i_dts  = -1;
    frame->b_enable_intra = (h->param->enable_intra);

    /* buffer for fenc */
    if (alloc_type == FT_ENC) {
#if XAVS2_ADAPT_LAYER
        /* M1, nal_info buffer */
        frame->nal_info = (xavs2_nal_info_t *)mem_ptr;
        frame->i_nal    = 0;
        mem_ptr        += i_nal_info_size;
        ALIGN_POINTER(mem_ptr);
#endif

        /* M2, set the bit stream buffer pointer and length
         * NOTE: the size of bitstream buffer is big enough, no need to reallocate
         *       memory in function encoder_encapsulate_nals */
        frame->p_bs_buf = mem_ptr;
        frame->i_bs_buf = bs_size;     /* the length is long enough */
        mem_ptr        += bs_size;
    }

    /* M3, buffer for planes: Y+U+V */
    frame->plane_buf = (pel_t *)mem_ptr;
    frame->size_plane_buf = (size_l + 2 * size_c) * sizeof(pel_t);

    frame->planes[0] = (pel_t *)mem_ptr;
    frame->planes[1] = frame->planes[0] + size_l;
    frame->planes[2] = frame->planes[1] + size_c;
    mem_ptr         += (size_l + size_c * 2) * sizeof(pel_t);

    if (alloc_type == FT_DEC || alloc_type == FT_TEMP) {
        uint8_t *p_align;
        /* point to plane data area */
        frame->planes[0] += frame->i_stride[0] * (XAVS2_PAD    ) + (XAVS2_PAD    );
        frame->planes[1] += frame->i_stride[1] * (XAVS2_PAD / 2) + (XAVS2_PAD / 2);
        frame->planes[2] += frame->i_stride[2] * (XAVS2_PAD / 2) + (XAVS2_PAD / 2);

        /* make sure the pointers are aligned */
        p_align = (uint8_t *)frame->planes[0];
        ALIGN_POINTER(p_align);
        frame->planes[0] = (pel_t *)p_align;
        p_align = (uint8_t *)frame->planes[1];
        ALIGN_POINTER(p_align);
        frame->planes[1] = (pel_t *)p_align;
        p_align = (uint8_t *)frame->planes[2];
        ALIGN_POINTER(p_align);
        frame->planes[2] = (pel_t *)p_align;
    }

    if (alloc_type == FT_DEC) {
        /* buffer for luma interpolated planes */
        frame->filtered[0] = frame->planes[0];  // full pel plane, reused
        for (i = 1; i < 16; i++) {
            frame->filtered[i] = NULL;
        }
#if ENABLE_FRAME_SUBPEL_INTPL
        switch (h->use_fractional_me) {
        case 1:
            frame->filtered[2]  = (pel_t *)mem_ptr;
            mem_ptr            += size_l * sizeof(pel_t);
            frame->filtered[8]  = (pel_t *)mem_ptr;
            mem_ptr            += size_l * sizeof(pel_t);
            frame->filtered[10] = (pel_t *)mem_ptr;
            mem_ptr            += size_l * sizeof(pel_t);

            break;
        case 2:
            for (i = 1; i < 16; i++) {
                frame->filtered[i] = (pel_t *)mem_ptr;
                mem_ptr           += size_l * sizeof(pel_t);
            }
            break;
        default:
            break;
        }
#endif
        /* point to plane data area */
        for (i = 1; i < 16; i++) {
            if (frame->filtered[i] != NULL) {
                frame->filtered[i] += frame->i_stride[0] * XAVS2_PAD + XAVS2_PAD;
            }
        }
        ALIGN_POINTER(mem_ptr);

        /* M4, reference index buffer */
        frame->pu_ref = (int8_t *)mem_ptr;
        mem_ptr      += frame_size_in_mvstore * sizeof(int8_t);
        ALIGN_POINTER(mem_ptr);

        /* M5, pu motion vector buffer */
        frame->pu_mv  = (mv_t *)mem_ptr;
        mem_ptr += frame_size_in_mvstore * sizeof(mv_t);
        ALIGN_POINTER(mem_ptr);

#if SAVE_CU_INFO
        /* M6, cu mode/cbp/level buffers */
        frame->cu_mode  = (int8_t *)mem_ptr;
        mem_ptr        += frame_size_in_mincu * sizeof(int8_t);
        ALIGN_POINTER(mem_ptr);
        frame->cu_cbp   = (int8_t *)mem_ptr;
        mem_ptr        += frame_size_in_mincu * sizeof(int8_t);
        ALIGN_POINTER(mem_ptr);
        frame->cu_level = (int8_t *)mem_ptr;
        mem_ptr        += frame_size_in_mincu * sizeof(int8_t);
        ALIGN_POINTER(mem_ptr);
#endif

        /* M7, line status array */
        frame->num_lcu_coded_in_row = (int *)mem_ptr;
        mem_ptr                    += h->i_height_in_lcu * sizeof(int);
        ALIGN_POINTER(mem_ptr);

        memset(frame->num_lcu_sao_off, 0, sizeof(frame->num_lcu_sao_off));
    }

    if (mem_ptr - (uint8_t *)frame > mem_size) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Failed to alloc one frame, type %d\n", alloc_type);
        goto fail;
    }

    /* update mem_base */
    if (mem_base != NULL) {
        *mem_base = mem_ptr;
    }

    /* initialize default value */
    frame->i_qpplus1     = 0;
    frame->cnt_refered   = 0;

    /* initialize signals */
    if (xavs2_thread_mutex_init(&frame->mutex, NULL)) {
        goto fail;
    }
    if (xavs2_thread_cond_init(&frame->cond, NULL)) {
        goto fail;
    }

    return frame;

fail:
    xavs2_free(mem_ptr);
    return NULL;
}

/* ---------------------------------------------------------------------------
 */
void xavs2_frame_delete(xavs2_handler_t *h_mgr, xavs2_frame_t *frame)
{
    if (frame == NULL) {
        return;
    }

    UNUSED_PARAMETER(h_mgr);

    xavs2_thread_cond_destroy(&frame->cond);

    xavs2_thread_mutex_destroy(&frame->mutex);

    /* free the frame itself */
    xavs2_free(frame);
}

/* ---------------------------------------------------------------------------
 */
void xavs2_frame_destroy_objects(xavs2_handler_t *h_mgr, xavs2_frame_t *frame)
{
    if (frame == NULL) {
        return;
    }

    UNUSED_PARAMETER(h_mgr);

    xavs2_thread_cond_destroy(&frame->cond);
    xavs2_thread_mutex_destroy(&frame->mutex);
}

/**
 * ===========================================================================
 * border expanding
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
void
plane_expand_border(pel_t *p_pix, int i_stride, int i_width, int i_height,
                    int i_padh, int i_padv, int b_pad_top, int b_pad_bottom)
{
    pel_t *pix = p_pix;
    pel_t *row;
    int y;

    /* --- horizontal ----------------------------------------------
     */
    for (y = 0; y < i_height; y++) {
        g_funcs.mem_repeat_p(pix - i_padh,  pix[0          ], i_padh);    /* left  band */
        g_funcs.mem_repeat_p(pix + i_width, pix[i_width - 1], i_padh);    /* right band */
        pix += i_stride;
    }

    /* --- vertical ------------------------------------------------
     */
    i_width += (i_padh << 1);

    /* upper band */
    if (b_pad_top) {
        pix = row = p_pix - i_padh;   /* start row position */
        for (y = 0; y < i_padv; y++) {
            pix -= i_stride;
            memcpy(pix, row, i_width * sizeof(pel_t));
        }
    }

    /* lower band */
    if (b_pad_bottom) {
        pix = row = p_pix + (i_height - 1) * i_stride - i_padh;
        for (y = 0; y < i_padv; y++) {
            pix += i_stride;
            memcpy(pix, row, i_width * sizeof(pel_t));
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void xavs2_frame_expand_border_frame(xavs2_t *h, xavs2_frame_t *frame)
{
    int slice_start_y = 0;
    int slice_height  = frame->i_lines[0];
    int b_frame_start = 1;
    int b_frame_end   = 1;
    int i;
    pel_t *pix;

    UNUSED_PARAMETER(h);

    for (i = 0; i < frame->i_plane; i++) {
        int chroma = !!i;
        int stride = frame->i_stride[i];
        int width  = frame->i_width[i];
        int height = slice_height >> chroma;
        int pad_h  = XAVS2_PAD >> chroma;
        int pad_v  = XAVS2_PAD >> chroma;

        pix = frame->planes[i] + (slice_start_y >> chroma) * stride;
        plane_expand_border(pix, stride, width, height, pad_h, pad_v, b_frame_start, b_frame_end);
    }
}

/* ---------------------------------------------------------------------------
 */
void xavs2_frame_expand_border_lcurow(xavs2_t *h, xavs2_frame_t *frame, int i_lcu_y)
{
    static const int UP_SHIFT = 4;
    int i_lcu_level = h->i_lcu_level;
    int b_start     = !i_lcu_y;
    int b_end       = (i_lcu_y == h->i_height_in_lcu - 1);
    int i;

    assert(h->param->slice_num == 1 || !h->param->b_cross_slice_loop_filter);

    for (i = 0; i < frame->i_plane; i++) {
        int chroma_shift = !!i;
        int stride  = frame->i_stride[i];
        int width   = frame->i_width[i];
        int padh    = XAVS2_PAD >> chroma_shift;
        int padv    = XAVS2_PAD >> chroma_shift;
        int y_start = ((i_lcu_y + 0) << (i_lcu_level - chroma_shift));
        int y_end   = ((i_lcu_y + 1) << (i_lcu_level - chroma_shift));
        int height;
        pel_t *pix;

        if (i_lcu_y != h->slices[h->i_slice_index]->i_first_lcu_y) {
            y_start -= UP_SHIFT;
        }
        if (i_lcu_y != h->slices[h->i_slice_index]->i_last_lcu_y) {
            y_end -= UP_SHIFT;
        }

        y_end = XAVS2_MIN(frame->i_lines[i], y_end);
        height = y_end - y_start;
        // if (i == 0) {
        //     xavs2_log(NULL, XAVS2_LOG_DEBUG, "Pad   POC [%3d], Slice %2d, Row %2d, [%3d, %3d)\n",
        //               h->fenc->i_frame, h->i_slice_index, i_lcu_y, y_start, y_end);
        // }

        pix = frame->planes[i] + y_start * stride;
        plane_expand_border(pix, stride, width, height, padh, padv, b_start, b_end);
    }
}

/* ---------------------------------------------------------------------------
 */
void xavs2_frame_expand_border_mod8(xavs2_t *h, xavs2_frame_t *frame)
{
    int i, y;

    for (i = 0; i < frame->i_plane; i++) {
        int i_scale  = !!i;
        int i_width  = h->param->org_width  >> i_scale;
        int i_height = h->param->org_height >> i_scale;
        int i_padx   = (h->i_width  - h->param->org_width ) >> i_scale;
        int i_pady   = (h->i_height - h->param->org_height) >> i_scale;
        int i_stride = frame->i_stride[i];

        /* expand right border */
        if (i_padx) {
            pel_t *pix = frame->planes[i] + i_width;
            for (y = 0; y < i_height; y++) {
                memset(pix, pix[-1], i_padx);
                pix += i_stride;
            }
        }

        /* expand bottom border */
        if (i_pady) {
            int rowlen = (i_width + i_padx) * sizeof(pel_t);
            pel_t *row = frame->planes[i] + (i_height - 1) * i_stride;
            pel_t *pix = frame->planes[i] + (i_height    ) * i_stride;
            for (y = i_height; y < i_height + i_pady; y++) {
                memcpy(pix, row, rowlen);
                pix += i_stride;
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * FIXME: 还需要考虑padding区域的拷贝
 */
void xavs2_frame_copy_planes(xavs2_t *h, xavs2_frame_t *dst, xavs2_frame_t *src)
{
    int k;

    UNUSED_PARAMETER(h);
    if (dst->size_plane_buf == src->size_plane_buf && dst->i_width[0] == src->i_width[0]) {
        g_funcs.fast_memcpy(dst->plane_buf, src->plane_buf, src->size_plane_buf);
    } else {
        for (k = 0; k < dst->i_plane; k++) {
            g_funcs.plane_copy(dst->planes[k], dst->i_stride[k],
                               src->planes[k], src->i_stride[k],
                               src->i_width[k], src->i_lines[k]);
        }
    }
}

