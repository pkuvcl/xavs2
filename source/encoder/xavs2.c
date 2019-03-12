/*
 * xavs2.c
 *
 * Description of this file:
 *    API functions definition of the xavs2 library
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
#include "encoder.h"
#include "cpu.h"
#include "ratecontrol.h"
#include "tdrdo.h"
#include "presets.h"
#include "rps.h"

/* ---------------------------------------------------------------------------
 */
static INLINE
int get_num_frame_threads(xavs2_param_t *param, int num_frame_threads, int num_row_threads)
{
    int a = ((param->search_range + (1 << param->lcu_bit_level) - 1) >> param->lcu_bit_level) + 1;
    int i;

    if (num_frame_threads > 0 && num_frame_threads < XAVS2_THREAD_MAX) {
        return num_frame_threads;
    }

    for (i = 2; i < XAVS2_THREAD_MAX; i++) {
        int n_row_threads_need = ((a * (i + 1) - 4) * i) >> 1;
        if (n_row_threads_need > num_row_threads) {
            break;
        }
    }

    return i - 1;
}


/**
 * ===========================================================================
 * interface function defines (xavs2 encoder library APIs for AVS2 video encoder)
 * ===========================================================================
 */

/**
 * ---------------------------------------------------------------------------
 * Function   : initialize default parameters for the xavs2 video encoder
 * Parameters :
 *      [in ] : param - pointer to struct xavs2_param_t
 *      [out] : none
 * Return     : none
 * ---------------------------------------------------------------------------
 */
xavs2_param_t *xavs2_encoder_opt_alloc(void)
{
    xavs2_param_t *param = (xavs2_param_t *)xavs2_malloc(sizeof(xavs2_param_t));

    if (param == NULL) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Failed to malloc space for xavs2_param_t with %d bytes\n",
                  sizeof(xavs2_param_t));
        return NULL;
    }

    memset(param, 0, sizeof(xavs2_param_t));

    /* --- sequence --------------------------------------------- */
    param->profile_id                 = MAIN_PROFILE;
    param->level_id                   = 66;
    param->progressive_sequence       = 1;
    param->chroma_format              = CHROMA_420;
    param->input_sample_bit_depth     = 8;
    param->sample_bit_depth           = 8;
    param->sample_precision           = 1;
    param->aspect_ratio_information   = 1;
    param->frame_rate                 = 25.0f;
    param->lcu_bit_level              = MAX_CU_SIZE_IN_BIT;
    param->scu_bit_level              = MIN_CU_SIZE_IN_BIT;
    param->org_width                  = 1920;
    param->org_height                 = 1080;
    strcpy(param->psz_in_file,        "input.yuv");
    strcpy(param->psz_bs_file,        "test.avs");
    strcpy(param->psz_dump_yuv,       "");
#if XAVS2_TRACE
    strcpy(param->psz_trace_file,     "trace_enc.txt");
#endif

    /* --- stream structure ------------------------------------- */
    param->enable_f_frame             = TRUE;
    param->InterlaceCodingOption      = 0;
    param->b_open_gop                 = 1;
    param->i_gop_size                 = -8;
    param->num_bframes                = 7;
    param->intra_period_max           = -1;
    param->intra_period_min           = -1;

    /* --- picture ---------------------------------------------- */
    param->progressive_frame          = 1;
    param->time_code_flag             = 0;
    param->top_field_first            = 0;
    param->repeat_first_field         = 0;
    param->fixed_picture_qp           = TRUE;
    param->i_initial_qp               = 32;

    /* --- slice ------------------------------------------------ */
    param->slice_num                  = 1;

    /* --- analysis options ------------------------------------- */
    param->enable_hadamard            = TRUE;
    param->me_method                  = XAVS2_ME_UMH;
    param->search_range               = 64;
    param->num_max_ref                = XAVS2_MAX_REFS;
    param->inter_2pu                  = TRUE;
    param->enable_amp                 = TRUE;
    param->enable_intra               = TRUE;
    param->i_rd_level                 = RDO_ALL;
    param->preset_level               = 5;
    param->is_preset_configured       = FALSE;
    param->rdo_bit_est_method         = 0;

    /* encoding tools ------------------------------------------- */
    param->enable_mhp_skip            = FALSE;
    param->enable_dhp                 = TRUE;
    param->enable_wsm                 = TRUE;
    param->enable_nsqt                = TRUE;
    param->enable_sdip                = TRUE;
    param->enable_secT                = TRUE;
    param->enable_sao                 = TRUE;
    param->b_sao_before_deblock       = FALSE;
    param->enable_alf                 = TRUE;
    param->alf_LowLatencyEncoding     = FALSE;
    param->enable_pmvr                = TRUE;
    param->b_cross_slice_loop_filter  = FALSE;    // 影响帧级并行编解码的速度，默认禁用
    param->enable_dmh                 = TRUE;
    param->b_fast_2lelvel_tu          = FALSE;

    /* RDOQ */
    param->i_rdoq_level               = RDOQ_ALL;
    param->lambda_factor_rdoq         = 75;
    param->lambda_factor_rdoq_p       = 120;
    param->lambda_factor_rdoq_b       = 100;

    param->enable_refine_qp           = TRUE;
    param->enable_tdrdo               = FALSE;

    /* loop filter */
    param->loop_filter_disable        = FALSE;
    param->loop_filter_parameter_flag = 0;
    param->alpha_c_offset             = 0;
    param->beta_offset                = 0;

    /* weight quant */
    param->enable_wquant              = FALSE;

#if ENABLE_WQUANT
    param->SeqWQM                     = 0;

    param->PicWQEnable                = FALSE;
    param->PicWQDataIndex             = 0;
    param->MBAdaptQuant               = 0;
    param->chroma_quant_param_delta_u = 0;
    param->chroma_quant_param_delta_v = 0;
    param->WQParam                    = 2;
    param->WQModel                    = 1;
#endif

    /* --- rate control ----------------------------------------- */
    param->i_rc_method                = XAVS2_RC_CQP;
    param->i_min_qp                   = 20;
    param->i_max_qp                   = MAX_QP;
    param->i_target_bitrate           = 1000000;

    /* --- parallel --------------------------------------------- */
    param->num_parallel_gop           = 1;
    param->i_frame_threads            = 0;
    param->i_lcurow_threads           = 0;
    param->enable_aec_thread          = 1;

    /* --- log -------------------------------------------------- */
    param->i_log_level                = 3;
    param->enable_psnr                = 1;
    param->enable_ssim                = 0;

    /* --- input/output for testing ----------------------------- */
    param->infile_header              = 0;
    param->output_merged_picture      = 0;

    parse_preset_level(param, param->preset_level);

    return param;
}

/**
 * ---------------------------------------------------------------------------
 * Function   : free memory of parameter
 * Parameters :
 *      [in ] : none
 *      [out] : parameter handler, can be further configured
 * Return     : none
 * ---------------------------------------------------------------------------
 */
void xavs2_encoder_opt_destroy(xavs2_param_t *param)
{
    if (param != NULL) {
        xavs2_free(param);
    }
}

/**
 * ---------------------------------------------------------------------------
 * Function   : create and initialize the xavs2 video encoder
 * Parameters :
 *      [in ] : param     - pointer to struct xavs2_param_t
 *      [out] : handle of xavs2 encoder wrapper
 * Return     : handle of xavs2 encoder wrapper, none zero for success, otherwise false
 * ---------------------------------------------------------------------------
 */
void *xavs2_encoder_create(xavs2_param_t *param)
{
    xavs2_handler_t *h_mgr   = NULL;
    xavs2_frame_t   *frm     = NULL;
    uint8_t         *mem_ptr = NULL;
    size_t size_ratecontrol;      /* size for rate control module */
    size_t size_tdrdo;
    size_t mem_size;
    int i;

    if (param == NULL) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Null input parameters for encoder creation\n");
        return NULL;
    }

    /* confirm the input parameters (log_level)  */
    if (param->i_log_level < XAVS2_LOG_NONE ||
        param->i_log_level > XAVS2_LOG_DEBUG) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Invalid parameter: log_level %d\n",
                  param->i_log_level);
        return NULL;
    }
    g_xavs2_default_log.i_log_level = param->i_log_level;

    /* init all function handlers */
    memset(&g_funcs, 0, sizeof(g_funcs));
#if HAVE_MMX
    g_funcs.cpuid = xavs2_cpu_detect();
#endif
    xavs2_init_all_primitives(param, &g_funcs);

    /* check parameters */
    if (encoder_check_parameters(param) < 0) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "error encoder parameters\n");
        goto fail;
    }

    size_ratecontrol = xavs2_rc_get_buffer_size(param);      /* rate control */
    size_tdrdo       = tdrdo_get_buffer_size(param);

    /* compute the memory size */
    mem_size = sizeof(xavs2_handler_t)                           +   /* M0, size of the encoder wrapper */
               xavs2_frame_buffer_size(param, FT_ENC) * XAVS2_INPUT_NUM     +   /* M4, size of buffered input frames */
               size_ratecontrol                                             +   /* M5, rate control information */
               size_tdrdo                                                   +   /* M6, TDRDO */
               CACHE_LINE_SIZE * (XAVS2_INPUT_NUM + 4);

    /* alloc memory for the encoder wrapper */
    CHECKED_MALLOC(mem_ptr, uint8_t *, mem_size);

    /* M0: assign the wrapper */
    h_mgr = (xavs2_handler_t *)mem_ptr;
    memset(h_mgr, 0, sizeof(xavs2_handler_t));
    mem_ptr  += sizeof(xavs2_handler_t);
    ALIGN_POINTER(mem_ptr);

    /* init log module */
    h_mgr->module_log.i_log_level = param->i_log_level;
    sprintf(h_mgr->module_log.module_name, "Manager %06llx", (uintptr_t)(h_mgr));

    /* counter: number of frames */
    h_mgr->num_input  = 0;
    h_mgr->num_encode = 0;
    h_mgr->num_output = 0;

    /* counters for encoding */
    h_mgr->i_exit_flag = 0;
    h_mgr->i_input     = 0;
    h_mgr->i_output    = -1;
    h_mgr->i_frame_in  = 0;
    h_mgr->i_frame_aec = 0;
    h_mgr->b_seq_end   = 0;
    h_mgr->max_out_dts = 0;
    h_mgr->max_out_pts = 0;
    h_mgr->create_time = xavs2_mdate();
    srand((uint32_t)h_mgr->create_time);

#if XAVS2_DUMP_REC
    if (strlen(param->psz_dump_yuv) > 0) {
        /* open dump file */
        if ((h_mgr->h_rec_file = fopen(param->psz_dump_yuv, "wb")) == NULL) {
            xavs2_log(h_mgr, XAVS2_LOG_ERROR, "Error open file %s\n", param->psz_dump_yuv);
        }
    }
#endif

    if (xavs2_thread_mutex_init(&h_mgr->mutex, NULL)) {
        goto fail;
    }

    for (i = 0; i < SIG_COUNT; i++) {
        if (xavs2_thread_cond_init(&h_mgr->cond[i], NULL)) {
            goto fail;
        }
    }

    /* decide all thread numbers */
    h_mgr->i_row_threads = param->i_lcurow_threads == 0 ? xavs2_cpu_num_processors() : param->i_lcurow_threads;
    h_mgr->i_frm_threads = get_num_frame_threads(param, param->i_frame_threads, h_mgr->i_row_threads);
    h_mgr->num_pool_threads = 0;
    h_mgr->num_row_contexts = 0;
    param->i_lcurow_threads = h_mgr->i_row_threads;
    param->i_frame_threads  = h_mgr->i_frm_threads;

    /* create RDO thread pool */
    if (h_mgr->i_frm_threads > 1 || h_mgr->i_row_threads > 1) {
        int thread_num = h_mgr->i_frm_threads + h_mgr->i_row_threads;   /* total threads */

        h_mgr->num_row_contexts = thread_num + h_mgr->i_frm_threads;

        /* create the thread pool */
        if (xavs2_threadpool_init(&h_mgr->threadpool_rdo, thread_num, NULL, NULL)) {
            xavs2_log(h_mgr, XAVS2_LOG_ERROR, "Error init thread pool RDO. %d", thread_num);
            goto fail;
        }
        h_mgr->num_pool_threads = thread_num;
    }

    /* create AEC thread pool */
    h_mgr->threadpool_aec = NULL;
    if (param->enable_aec_thread) {
        xavs2_threadpool_init(&h_mgr->threadpool_aec, h_mgr->i_frm_threads, NULL, NULL);
    }

    /* init all lists */
    if (xl_init(&h_mgr->list_frames_free)  != 0 ||
        xl_init(&h_mgr->list_frames_output) != 0 ||
        xl_init(&h_mgr->list_frames_ready) != 0) {
        goto fail;
    }

    /* init rate-control buffer */
    ALIGN_POINTER(mem_ptr);
    h_mgr->rate_control = (ratectrl_t *)mem_ptr;
    mem_ptr            += size_ratecontrol;
    ALIGN_POINTER(mem_ptr);

    if (xavs2_rc_init(h_mgr->rate_control, param) < 0) {
        xavs2_log(h_mgr, XAVS2_LOG_ERROR, "create rate control fail\n");
        goto fail;

    }

    /* TD-RDO */
    if (param->enable_tdrdo) {
        h_mgr->td_rdo = (td_rdo_t *)mem_ptr;
        mem_ptr      += size_tdrdo;
        ALIGN_POINTER(mem_ptr);

        if (tdrdo_init(h_mgr->td_rdo, param) != 0) {
            xavs2_log(h_mgr, XAVS2_LOG_ERROR, "init td-rdo fail\n");
            goto fail;
        }
    }

    /* create an encoder handler */
    h_mgr->p_coder = encoder_open(param, h_mgr);
    if (h_mgr->p_coder == NULL) {
        goto fail;
    }

    /* create encoder handlers for multi-thread */
    if (h_mgr->i_frm_threads > 1 || h_mgr->i_row_threads > 1) {
        if (encoder_contexts_init(h_mgr->p_coder, h_mgr) < 0) {
            goto fail;
        }
    }

    /* M4: alloc memory for each node and append to image idle list */
    frame_buffer_init(h_mgr, &mem_ptr, &h_mgr->ipb,
                      XAVS2_INPUT_NUM, FT_ENC);
    for (i = 0; i < XAVS2_INPUT_NUM; i++) {
        frm = h_mgr->ipb.frames[i];
        if (frm) {
            xl_append(&h_mgr->list_frames_free, frm);
        } else {
            goto fail;
        }
    }

    /* allocate DPB */
    frame_buffer_init(h_mgr, NULL, &h_mgr->dpb,
                      XAVS2_MIN(FREF_BUF_SIZE, MAX_REFS + h_mgr->i_frm_threads * 4), FT_DEC);

    /* memory check */
    if ((uintptr_t)(h_mgr) + mem_size < (uintptr_t)mem_ptr) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Failed to create input frame buffer.\n");
        goto fail;
    }

    /* init lookahead in the encoder wrapper */
    h_mgr->lookahead.bpframes = param->i_gop_size;
    h_mgr->lookahead.start    = 0;
    memset(h_mgr->blocked_frm_set, 0, sizeof(h_mgr->blocked_frm_set));
    memset(h_mgr->blocked_pts_set, 0, sizeof(h_mgr->blocked_pts_set));
    h_mgr->num_blocked_frames = 0;

    h_mgr->fp_trace = NULL;

    /* create wrapper thread */
    if (xavs2_create_thread(&h_mgr->thread_wrapper, proc_wrapper_thread, h_mgr)) {
        xavs2_log(h_mgr, XAVS2_LOG_ERROR, "create encoding thread\n");
        goto fail;
    }

    return h_mgr;

fail:
    if (mem_ptr && h_mgr) {
        xavs2_encoder_destroy(h_mgr);
    }

    return NULL;
}

/**
 * ---------------------------------------------------------------------------
 * Function   : destroy the xavs2 video encoder
 * Parameters :
 *      [in ] : coder - pointer to wrapper of the xavs2 encoder
 *      [out] : none
 * Return     : none
 * ---------------------------------------------------------------------------
 */
void xavs2_encoder_destroy(void *coder)
{
    xavs2_handler_t *h_mgr = (xavs2_handler_t *)coder;
    xavs2_frame_t frm_flush = { 0 };
    xavs2_frame_t frm_exit  = { 0 };

    /* destroy all threads: lookahead and wrapper threads */
    if (h_mgr->p_coder != NULL) {
        frm_flush.i_state = XAVS2_FLUSH;        /* signal to flush encoder */
        frm_exit.i_state  = XAVS2_EXIT_THREAD;  /* signal to exit */
        send_frame_to_enc_queue(h_mgr, &frm_flush);
        send_frame_to_enc_queue(h_mgr, &frm_exit);

        /* wait until the RDO process exit, then memory can be released */
        xavs2_thread_join(h_mgr->thread_wrapper, NULL);
    }

    /* close the encoder */
    encoder_close(h_mgr);

    xavs2_log(h_mgr, XAVS2_LOG_DEBUG, "Encoded %d frames, %.3f secs\n",
              h_mgr->num_input, 0.000001 * (xavs2_mdate() - h_mgr->create_time));

    if (h_mgr->fp_trace) {
        fclose(h_mgr->fp_trace);
    }

    /* free memory of encoder wrapper */
    memset(h_mgr, 0, sizeof(xavs2_handler_t));
    xavs2_free(h_mgr);

}

/**
 * ---------------------------------------------------------------------------
 * Function   : get buffer for the encoder caller
 * Parameters :
 *      [in ] : coder - pointer to wrapper of the xavs2 encoder
 *            : pic   - pointer to struct xavs2_picture_t
 *      [out] : pic   - memory would be allocated for the image planes
 * Return     : zero for success, otherwise failed
 * ---------------------------------------------------------------------------
 */
int xavs2_encoder_get_buffer(void *coder, xavs2_picture_t *pic)
{
    xavs2_handler_t *h_mgr   = (xavs2_handler_t *)coder;
    xavs2_t         *p_coder = h_mgr->p_coder;
    const xavs2_param_t *param = p_coder->param;
    xavs2_frame_t   *frame;

    assert(h_mgr != NULL && pic != NULL);
    if (h_mgr == NULL || pic == NULL) {
        return -1;
    }

    memset(pic, 0, sizeof(xavs2_picture_t));

    /* fetch an empty node from unused list */
    frame = frame_buffer_get_free_frame_ipb(h_mgr);

    /* set properties */
    pic->img.in_sample_size  = param->input_sample_bit_depth == 8 ? 1 : 2;
    pic->img.enc_sample_size = sizeof(pel_t);
    pic->img.i_width[0]      = param->org_width;
    pic->img.i_width[1]      = param->org_width >> 1;
    pic->img.i_width[2]      = param->org_width >> 1;
    pic->img.i_lines[0]      = param->org_height;
    pic->img.i_lines[1]      = param->org_height >> (param->chroma_format <= CHROMA_420 ? 1 : 0);
    pic->img.i_lines[2]      = param->org_height >> (param->chroma_format <= CHROMA_420 ? 1 : 0);
    pic->img.i_csp           = XAVS2_CSP_I420;
    pic->img.i_plane         = frame->i_plane;
    pic->img.i_stride[0]     = frame->i_stride[0] * sizeof(pel_t);
    pic->img.i_stride[1]     = frame->i_stride[1] * sizeof(pel_t);
    pic->img.i_stride[2]     = frame->i_stride[2] * sizeof(pel_t);
    pic->img.img_planes[0]   = (uint8_t *)frame->planes[0];
    pic->img.img_planes[1]   = (uint8_t *)frame->planes[1];
    pic->img.img_planes[2]   = (uint8_t *)frame->planes[2];
    pic->priv                = frame;   /* keep trace of this frame */

    return 0;
}

/**
 * ---------------------------------------------------------------------------
 * Function   : label a packet to be recycled
 * Parameters :
 *      [in ] : coder    - pointer to handle of xavs2 encoder (return by `encoder_create()`)
 *            : packet   - pointer to struct xavs2_outpacket_t, whose bit-stream buffer would be recycled
 *      [out] : none
 * Return     : zero for success, otherwise failed
 * ---------------------------------------------------------------------------
 */
int xavs2_encoder_packet_unref(void *coder, xavs2_outpacket_t *packet)
{
    if (coder == NULL || packet == NULL) {
        return 0;
    }

    if (packet->private_data != NULL) {
        xavs2_handler_t *h_mgr = (xavs2_handler_t *)coder;
        xl_append(&h_mgr->list_frames_free, packet->private_data);
    }

    return 0;
}

/**
 * ---------------------------------------------------------------------------
 * Function   : write (send) data to the xavs2 encoder
 * Parameters :
 *      [in ] : coder - pointer to wrapper of the xavs2 encoder
 *            : pic   - pointer to struct xavs2_picture_t
 *      [out] : packet- output bit-stream
 * Return     : zero for success, otherwise failed
 * ---------------------------------------------------------------------------
 */
int xavs2_encoder_encode(void *coder, xavs2_picture_t *pic, xavs2_outpacket_t *packet)
{
    xavs2_handler_t *h_mgr = (xavs2_handler_t *)coder;
    xavs2_frame_t *frame = NULL;

    assert(h_mgr != NULL);

    if (pic != NULL) {
        xavs2_t *h = NULL;

        /* is this our own frame buffer ? */
        assert(pic->priv != NULL);
        if (pic->priv == NULL) {
            return -1;
        }

        frame = (xavs2_frame_t *)pic->priv;

        if (pic->i_state != XAVS2_STATE_NO_DATA) {
            /* copy frame properties */
            frame->i_frm_type = pic->i_type;
            frame->i_qpplus1  = pic->i_qpplus1;
            frame->i_pts      = pic->i_pts;
            frame->b_keyframe = pic->b_keyframe;

            /* set encoder handle */
            h = h_mgr->p_coder;

            /* expand border if need */
            if (h->param->org_width != h->i_width || h->param->org_height != h->i_height) {
                xavs2_frame_expand_border_mod8(h, frame);
            }

            /* set frame number here */
            frame->i_frame = h_mgr->i_input;
            h_mgr->i_input = get_next_frame_id(h_mgr->i_input);

            /* set flag */
            frame->i_state = 0;

            /* counter number of input frames */
            h_mgr->num_input++;
        } else {
            /* recycle space for the pic handler */
            xl_append(&h_mgr->list_frames_free, frame);
            frame = NULL;
        }
    } else {
        /* fetch an empty node from unused list */
        frame = frame_buffer_get_free_frame_ipb(h_mgr);

        /* set flag to flush delayed frames */
        frame->i_state = XAVS2_FLUSH;
    }

    /* decide slice type and send frames into encoding queue */
    if (frame != NULL) {
        send_frame_to_enc_queue(h_mgr, frame);
    }

    /* fetch a frame */
    encoder_fetch_one_encoded_frame(h_mgr, packet, pic == NULL);

    return 0;
}
