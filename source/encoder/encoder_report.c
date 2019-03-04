/*
 * encoder_report.c
 *
 * Description of this file:
 *    Encoder Reporting functions definition of the xavs2 library
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
#include "mc.h"
#include "wrapper.h"
#include "encoder.h"
#include "version.h"
#include "cpu.h"

#if defined(__ARM_ARCH_7A__) || SYS_LINUX || !defined(_MSC_VER)
#define sprintf_s snprintf
#endif

#if XAVS2_STAT
/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
double get_psnr_with_ssd(double f_max, uint64_t diff)
{
    if (diff > 0) {
        return 10.0 * log10(f_max / diff);
    } else {
        return 99.999;
    }
}


/* ---------------------------------------------------------------------------
 * calculate PSNR for all three components (Y, U and V)
 */
void encoder_cal_psnr(xavs2_t *h, double *psnr_y, double *psnr_u, double *psnr_v)
{
    int i_width = h->param->org_width;
    int i_height = h->param->org_height;
    int i_size = i_width * i_height;
    int uvformat = h->param->chroma_format == CHROMA_420 ? 4 : 2;
    const double f_max_signal = (double)(255 * 255) * i_size;
    const int inout_shift     = 0;
    uint64_t diff_y, diff_u, diff_v;

    /* luma */
    diff_y = xavs2_pixel_ssd_wxh(&g_funcs.pixf,
                                 h->fenc->planes[0], h->fenc->i_stride[0],
                                 h->fdec->planes[0], h->fdec->i_stride[0], i_width, i_height, inout_shift);

    /* chroma */
    if (h->param->chroma_format != CHROMA_400) {
        i_width  >>= 1;
        i_height >>= 1;
        diff_u = xavs2_pixel_ssd_wxh(&g_funcs.pixf,
                                     h->fenc->planes[1], h->fenc->i_stride[1],
                                     h->fdec->planes[1], h->fdec->i_stride[1], i_width, i_height, inout_shift);
        diff_v = xavs2_pixel_ssd_wxh(&g_funcs.pixf,
                                     h->fenc->planes[2], h->fenc->i_stride[2],
                                     h->fdec->planes[2], h->fdec->i_stride[2], i_width, i_height, inout_shift);
    } else {
        diff_u = 0;
        diff_v = 0;
    }

    xavs2_emms();     /* call before using float instructions */

    /* get the PSNR for current frame */
    *psnr_y = get_psnr_with_ssd(f_max_signal, diff_y);
    *psnr_u = get_psnr_with_ssd(f_max_signal, diff_u * uvformat);
    *psnr_v = get_psnr_with_ssd(f_max_signal, diff_v * uvformat);
}

/* ---------------------------------------------------------------------------
 * calculate SSIM
 */
double ssim_calculate_plane(xavs2_t *h, int comp_id)
{
    uint32_t g_uiBitDepth = 8;    // base bit-depth
    uint32_t g_uiBitIncrement = 0;    // increments

    double k_ssim_1 = 0.01;
    double k_ssim_2 = 0.03;

    int m_uiWidth = h->param->org_width;
    int m_uiHeight = h->param->org_height;

    int iStride1 = h->fenc->i_stride[comp_id];
    int iStride2 = h->fdec->i_stride[comp_id];

    // xavs2_log(h, XAVS2_LOG_INFO, "iStride: %d",iStride);
    uint32_t uiWinWidth = 8;
    uint32_t uiWinHeight= 8;

    uint32_t uiWidth  = comp_id ? m_uiWidth >> 1 : m_uiWidth;
    uint32_t uiHeight = comp_id ? m_uiHeight >> 1 : m_uiHeight;

    double dLocSSIM, dLocMeanRef, dLocMeanRec, dLocVarRef, dLocVarRec, dLocCovar, Num1, Num2, Den1, Den2, dMSSIM = 0;
    uint32_t i, j, x, y;
    // xavs2_log(h, XAVS2_LOG_INFO, "uiHeight: %d uiWinHeight: %d uiWidth: %d uiWinWidth:%d\n",uiHeight,uiWinHeight,uiWidth,uiWinWidth);

    uint32_t uiNumWin = (uiHeight - uiWinHeight + 1)*(uiWidth - uiWinWidth + 1);
    uint32_t iWinPixel = uiWinWidth * uiWinHeight;
    uint32_t uiMaxval = 255 * (1 << (g_uiBitDepth + g_uiBitIncrement - 8));
    // xavs2_log(h, XAVS2_LOG_INFO, "uiNumWin : %d uiMaxval : %d\n",uiNumWin,uiMaxval);

    double C1 = k_ssim_1 * k_ssim_1 * uiMaxval * uiMaxval;
    double C2 = k_ssim_2 * k_ssim_2 * uiMaxval * uiMaxval;

    pel_t*  pOrg = h->fenc->planes[comp_id];
    pel_t*  pRec = h->fdec->planes[comp_id];
    // xavs2_log(h, XAVS2_LOG_INFO, "pOrg : %p pRec : %p\n",pOrg,pRec);

    pel_t*  pOrgPel = pOrg;
    pel_t*  pRecPel = pRec;

    for (j = 0; j <= uiHeight - uiWinHeight; j++) {
        for (i = 0; i <= uiWidth - uiWinWidth; i++) {
            dLocMeanRef = 0;
            dLocMeanRec = 0;
            dLocVarRef = 0;
            dLocVarRec = 0;
            dLocCovar = 0;
            pOrgPel = pOrg + i + iStride1*j;
            pRecPel = pRec + i + iStride2*j;
            // xavs2_log(h, XAVS2_LOG_INFO, "pOrgPel[0] : %d pRecPel[0] : %d\n",pOrgPel[0],pRecPel[0]);
            // xavs2_log(h, XAVS2_LOG_INFO, "uiWinWidth : %d uiWinHeight : %d\n",uiWinWidth,uiWinHeight);

            for (y = 0; y < uiWinHeight; y++) {
                for (x = 0; x < uiWinWidth; x++) {
                    // xavs2_log(h, XAVS2_LOG_INFO, "pOrgPel[%d] : %d pRecPel[%d] : %d\n",x,pOrgPel[x],x,pRecPel[x]);

                    dLocMeanRef += pOrgPel[x];
                    dLocMeanRec += pRecPel[x];
                    dLocVarRef += pOrgPel[x] * pOrgPel[x];
                    dLocVarRec += pRecPel[x] * pRecPel[x];
                    dLocCovar += pOrgPel[x] * pRecPel[x];

                }
                pOrgPel += iStride1;
                pRecPel += iStride2;
            }

            dLocMeanRef /= iWinPixel;
            dLocMeanRec /= iWinPixel;
            // xavs2_log(h, XAVS2_LOG_INFO, "dLocMeanRef : %7.4f dLocMeanRec : %7.4f \n",dLocMeanRef,dLocMeanRec);

            dLocVarRef = (dLocVarRef - dLocMeanRef * dLocMeanRef * iWinPixel) / iWinPixel;
            dLocVarRec = (dLocVarRec - dLocMeanRec * dLocMeanRec * iWinPixel) / iWinPixel;
            dLocCovar = (dLocCovar - dLocMeanRef * dLocMeanRec * iWinPixel) / iWinPixel;

            Num1 = 2.0 * dLocMeanRef * dLocMeanRec + C1;
            Num2 = 2.0 * dLocCovar + C2;
            Den1 = dLocMeanRef * dLocMeanRef + dLocMeanRec * dLocMeanRec + C1;
            Den2 = dLocVarRef + dLocVarRec + C2;

            dLocSSIM = (Num1 * Num2) / (Den1 * Den2);

            dMSSIM += dLocSSIM;
        }
    }

    // xavs2_log(h, XAVS2_LOG_INFO,"ssim: %7.4f \n ", dMSSIM / (double)uiNumWin);
    return dMSSIM / (double)uiNumWin;
}

/* ---------------------------------------------------------------------------
 * calculate SSIM for all three components (Y, U and V)
 */
void encoder_cal_ssim(xavs2_t *h, double *ssim_y, double *ssim_u, double *ssim_v)
{
    *ssim_y = ssim_calculate_plane(h, 0);
    *ssim_u = ssim_calculate_plane(h, 1);
    *ssim_v = ssim_calculate_plane(h, 2);
}

/* ---------------------------------------------------------------------------
 */
void encoder_report_stat_info(xavs2_t *h)
{
    xavs2_stat_t *p_stat = &h->h_top->stat;
    float f_bitrate;

    double f_psnr_y = p_stat->stat_total.f_psnr[0];
    double f_psnr_u = p_stat->stat_total.f_psnr[1];
    double f_psnr_v = p_stat->stat_total.f_psnr[2];

    double ssim_y = p_stat->stat_total.f_ssim[0];
    double ssim_u = p_stat->stat_total.f_ssim[1];
    double ssim_v = p_stat->stat_total.f_ssim[2];

    int64_t i_total_bits = p_stat->stat_total.i_frame_size;
    int num_total_frames = p_stat->stat_total.num_frames;

    if (num_total_frames == 0) {
        xavs2_log(NULL, XAVS2_LOG_WARNING, "------------------------------------------------------------------\n");
        xavs2_log(NULL, XAVS2_LOG_WARNING, "TOTAL TIME: %8.3f sec, NO FRAMES CODED\n",
                  (double)(p_stat->i_end_time - p_stat->i_start_time) / 1000000.0);
        return;
    }

    xavs2_log(h, XAVS2_LOG_INFO, "---------------------------------------------------------------------\n");

    // FIXME: cause "Segmentation fault (core dumped)" in Linux, print directly (gcc 4.7)
    if (h->param->enable_psnr) {
        xavs2_log(h, XAVS2_LOG_INFO, "AVERAGE SEQ PSNR:      %7.4f %7.4f %7.4f\n",
                  f_psnr_y / num_total_frames, f_psnr_u / num_total_frames, f_psnr_v / num_total_frames);
    }
    if (h->param->enable_ssim) {
        xavs2_log(h, XAVS2_LOG_INFO, "AVERAGE SEQ SSIM:      %7.5f %7.5f %7.5f\n",
                  ssim_y / num_total_frames, ssim_u / num_total_frames, ssim_v / num_total_frames);
    }

    // BITRATE
    f_bitrate = (i_total_bits * (8.0f / 1000.0f) * h->framerate) / ((float)num_total_frames);
    xavs2_log(h, XAVS2_LOG_INFO, "         BITRATE: %6.2f kb/s @ %4.1f Hz, %d frames, xavs2 p%d \n",
              f_bitrate, h->framerate, num_total_frames, h->param->preset_level);

    // TOTAL BITS
    xavs2_log(h, XAVS2_LOG_INFO, "      TOTAL BITS: %lld (I: %lld, B: %lld, P/F: %lld)\n",
              i_total_bits * 8,
              p_stat->stat_i_frame.i_frame_size * 8,
              p_stat->stat_b_frame.i_frame_size * 8,
              p_stat->stat_p_frame.i_frame_size * 8);

    // TOTAL TIME
    xavs2_log(h, XAVS2_LOG_DEBUG, "      TOTAL TIME: %8.3f sec, total %d frames, speed: %5.2f fps \n",
              (double)(p_stat->i_end_time - p_stat->i_start_time) / 1000000.0,
              num_total_frames,
              (double)num_total_frames / ((p_stat->i_end_time - p_stat->i_start_time) / 1000000.0));
    // Time Distribution
    xavs2_log(h, XAVS2_LOG_DEBUG, "      Frame Time:   I: %6.2f%%;   B: %6.2f%%;   P/F: %6.2f%%\n",
              (double)(p_stat->stat_i_frame.i_time_duration * 100.0) / p_stat->stat_total.i_time_duration,
              (double)(p_stat->stat_b_frame.i_time_duration * 100.0) / p_stat->stat_total.i_time_duration,
              (double)(p_stat->stat_p_frame.i_time_duration * 100.0) / p_stat->stat_total.i_time_duration);
    xavs2_log(h, XAVS2_LOG_INFO, "      Frame Num :   I: %6.2f%%;   B: %6.2f%%;   P/F: %6.2f%%\n",
              p_stat->stat_i_frame.num_frames * 100.0 / num_total_frames,
              p_stat->stat_b_frame.num_frames * 100.0 / num_total_frames,
              p_stat->stat_p_frame.num_frames * 100.0 / num_total_frames);
    xavs2_log(h, XAVS2_LOG_INFO, "---------------------------------------------------------------------\n");
}

/* ---------------------------------------------------------------------------
 */
static INLINE
void stat_add_frame_info(com_stat_t *sum_stat, com_stat_t *frm_stat, int frm_bs_len)
{
    sum_stat->num_frames++;
    sum_stat->i_frame_size    += frm_bs_len;
    sum_stat->i_time_duration += frm_stat->i_time_duration;

    sum_stat->f_psnr[0] += frm_stat->f_psnr[0];
    sum_stat->f_psnr[1] += frm_stat->f_psnr[1];
    sum_stat->f_psnr[2] += frm_stat->f_psnr[2];

    sum_stat->f_ssim[0] += frm_stat->f_ssim[0];
    sum_stat->f_ssim[1] += frm_stat->f_ssim[1];
    sum_stat->f_ssim[2] += frm_stat->f_ssim[2];
}

/* ---------------------------------------------------------------------------
 * get reference list string
 */
static ALWAYS_INLINE
void get_reference_list_str(char *s_ref_list, int *p_poc, int num_ref)
{
    if (num_ref > 1) {
        char str_tmp[16];
        int i;

        sprintf(s_ref_list, "[%3d ", p_poc[0]);
        for (i = 1; i < num_ref - 1; i++) {
            sprintf(str_tmp, "%3d ", p_poc[i]);
            strcat(s_ref_list, str_tmp);
        }
        sprintf(str_tmp, "%3d]", p_poc[i]);
        strcat(s_ref_list, str_tmp);
    } else if (num_ref == 1) {
        sprintf(s_ref_list, "[%3d]", p_poc[0]);
    }
}


/* ---------------------------------------------------------------------------
 */
void encoder_report_one_frame(xavs2_t *h, outputframe_t *frame)
{
    static const char frm_type[4] = {'I', 'P', 'B', 'F'};
    char s_out_base[256];
    char s_psnr[64] = "";
    char s_ref_list[32] = "";
    xavs2_stat_t *p_stat  = &h->h_top->stat;
    frame_stat_t *frmstat = &frame->out_frm_stat;
    int frm_bs_len = frame->frm_enc->i_bs_len;

    if (p_stat->i_start_time == 0) {
        p_stat->i_start_time = frame->frm_enc->i_time_start;
    }

    p_stat->i_end_time = frame->frm_enc->i_time_end;
    frmstat->stat_frm.i_time_duration = frame->frm_enc->i_time_end - frame->frm_enc->i_time_start;

    /* frame info */
    xavs2_thread_mutex_lock(&h->h_top->mutex);
    switch (frmstat->i_type) {
    case 0:
        stat_add_frame_info(&p_stat->stat_i_frame, &frmstat->stat_frm, frm_bs_len);
        break;
    case 2:
        stat_add_frame_info(&p_stat->stat_b_frame, &frmstat->stat_frm, frm_bs_len);
        break;
    default:
        stat_add_frame_info(&p_stat->stat_p_frame, &frmstat->stat_frm, frm_bs_len);
        break;
    }

    stat_add_frame_info(&p_stat->stat_total, &frmstat->stat_frm, frm_bs_len);
    xavs2_thread_mutex_unlock(&h->h_top->mutex);

    if (h->param->enable_psnr && h->param->enable_ssim) {
        sprintf_s(s_psnr, 64, " %7.4f %7.4f %7.4f %7.5f %7.5f %7.5f",
                  frmstat->stat_frm.f_psnr[0],
                  frmstat->stat_frm.f_psnr[1],
                  frmstat->stat_frm.f_psnr[2],
                  frmstat->stat_frm.f_ssim[0],
                  frmstat->stat_frm.f_ssim[1],
                  frmstat->stat_frm.f_ssim[2]);
    } else if (h->param->enable_psnr) {
        sprintf_s(s_psnr, 64, " %7.4f %7.4f %7.4f",
                  frmstat->stat_frm.f_psnr[0],
                  frmstat->stat_frm.f_psnr[1],
                  frmstat->stat_frm.f_psnr[2]);
    }

    sprintf_s(s_out_base, 256, "%4d (%c) %2d  %8d %s %5d",
              frmstat->i_frame,
              frm_type[frmstat->i_type],
              frmstat->i_qp,
              // frmstat->stat_frm.f_lambda_frm,  // %7.2f
              frame->frm_enc->i_bs_len * 8,
              s_psnr,
              (int)((frmstat->stat_frm.i_time_duration) / 1000));

    get_reference_list_str(s_ref_list, frmstat->ref_poc_set, frmstat->i_ref);

    xavs2_log(h, XAVS2_LOG_DEBUG, "%s  %s\n", s_out_base, s_ref_list);
}

/* ---------------------------------------------------------------------------
 */
void encoder_show_head_info(xavs2_param_t *param)
{
    const char *s_gop_param = param->b_open_gop ? "Open" : "Closed";
    char buf_cpu[120] = "";
    char s_threads_row  [16] = "auto";
    char s_threads_frame[16] = "auto";

    /* init temp string */
    if (param->i_lcurow_threads != 0) {
        sprintf(s_threads_row, "%d", param->i_lcurow_threads);
    }
    if (param->i_frame_threads != 0) {
        sprintf(s_threads_frame, "%d", param->i_frame_threads);
    }

    /* algorithms and controls in the encoder */
    if (param->enable_refine_qp) {
        xavs2_log(NULL, XAVS2_LOG_DEBUG, " RefinedQp is on, the input QP might be changed;\n");
    }
    /* input/output properties */
    xavs2_log(NULL, XAVS2_LOG_DEBUG, " xavs2enc version : %s  %s\n",
              XVERSION_STR, XBUILD_TIME);
    xavs2_log(NULL, XAVS2_LOG_DEBUG, " Input YUV file   : %s \n", param->psz_in_file);
    xavs2_log(NULL, XAVS2_LOG_DEBUG, " Output bitstream : %s \n", param->psz_bs_file);
    xavs2_log(NULL, XAVS2_LOG_DEBUG, " Recon YUV file   : %s \n", param->psz_dump_yuv);
    xavs2_log(NULL, XAVS2_LOG_DEBUG, " Total Frames     : %d \n", param->num_frames);
    /* basic parameters */
    xavs2_log(NULL, XAVS2_LOG_INFO, "--------------------------------------------------------------------------------\n");
    xavs2_log(NULL, XAVS2_LOG_INFO, " Profile & Level  : 0x%02X-0x%02X, BitDepth: %d/%d, size(pel): %d \n",
              param->profile_id, param->level_id, param->input_sample_bit_depth, param->sample_bit_depth, sizeof(pel_t));
    xavs2_log(NULL, XAVS2_LOG_INFO, " Video Property   : %dx%d, %.3f Hz (FrameRateCode: %d)\n",
              param->org_width, param->org_height, param->frame_rate, param->frame_rate_code);

    /* CPU capacities */
    xavs2_get_simd_capabilities(buf_cpu, g_funcs.cpuid);
    xavs2_log(NULL, XAVS2_LOG_INFO, " CPU Capabilities : %s\n", buf_cpu);

    xavs2_log(NULL, XAVS2_LOG_INFO, " Preset Level     : %d,  %s \n", param->preset_level, xavs2_preset_names[param->preset_level]);
    xavs2_log(NULL, XAVS2_LOG_INFO, " Ref Structure    : BFrames: %d; %s GOP; IntraPeriod: %d~%d\n",
              param->num_bframes, s_gop_param, param->intra_period_min, param->intra_period_max);
    xavs2_log(NULL, XAVS2_LOG_INFO, " Rate Control     : %d; QP: %d, [%2d, %2d]; %.3f Mbps\n",
              param->i_rc_method, param->i_initial_qp, param->i_min_qp, param->i_max_qp, 0.000001f * param->i_target_bitrate);
    xavs2_log(NULL, XAVS2_LOG_INFO, " Threads (Row/Frm): %s / %s, cpu cores %d \n", s_threads_row, s_threads_frame, xavs2_cpu_num_processors());
}

/* ---------------------------------------------------------------------------
 */
void encoder_show_frame_info_tab(xavs2_t *h, xavs2_handler_t *mgr)
{
    const xavs2_param_t *param = h->param;
    size_t space_alloc = xavs2_get_total_malloc_space();
    space_alloc = (space_alloc + (1 << 20) - 1) >> 20;

    xavs2_log(NULL, XAVS2_LOG_INFO, " Threads (Alloc)  : %d / %d, threadpool %d, RowContexts %d \n",
              mgr->i_row_threads, mgr->i_frm_threads, mgr->num_pool_threads, mgr->num_row_contexts);
    xavs2_log(NULL, XAVS2_LOG_INFO, " Memory  (Alloc)  : %d MB \n", (int)(space_alloc));
    xavs2_log(NULL, XAVS2_LOG_INFO, " Enabled Tools    : LCU %d, 2NxN/Nx2N:%d, AMP:%d, IntraInInter:%d, SDIP:%d,\n"\
              "                    FFrame %d, DHP:%d, DMH:%d, MHP:%d, WSM:%d,\n"\
              "                    NSQT:%d, Fast2LevelTu:%d, 2ndTrans:%d,\n"\
              "                    ME:%d, SearchRange:%d,\n"\
              "                    RefinedQP:%d, TDRDO:%d, Algorithm: %8llx\n"\
              "                    RdLevel:%d, RdoqLevel:%d, SAO:%d, ALF:%d.\n",
              1 << param->lcu_bit_level, param->inter_2pu, param->enable_amp, param->enable_intra, param->enable_sdip,
              param->enable_f_frame, param->enable_dhp, param->enable_dmh, param->enable_mhp_skip, param->enable_wsm,
              param->enable_nsqt, param->b_fast_2lelvel_tu, param->enable_secT,
              param->me_method, param->search_range,
              param->enable_refine_qp, param->enable_tdrdo, h->i_fast_algs,
              param->i_rd_level, param->i_rdoq_level, param->enable_sao, param->enable_alf);
    /* table header */
    xavs2_log(NULL, XAVS2_LOG_INFO, "--------------------------------------------------------------------------------\n");
    if (param->enable_psnr && param->enable_ssim) {
        xavs2_log(NULL, XAVS2_LOG_DEBUG, "POC Type QP +   Bits    PsnrY   PsnrU   PsnrV   SsimY   SsimU   SsimV   Time  [ RefList ]\n");
    } else if (param->enable_psnr) {
        xavs2_log(NULL, XAVS2_LOG_DEBUG, "POC Type QP +   Bits    PsnrY   PsnrU   PsnrV   Time  [ RefList ]\n");
    } else {
        xavs2_log(NULL, XAVS2_LOG_DEBUG, "POC Type QP +   Bits     Time  [ RefList ]\n");
    }
}

#endif  // #if XAVS2_STAT
