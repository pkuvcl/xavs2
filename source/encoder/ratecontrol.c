/*
 * ratecontrol.c
 *
 * Description of this file:
 *    Ratecontrol functions definition of the xavs2 library
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
#include "ratecontrol.h"
#include "cpu.h"

#include "defines.h"
/**
 * ===========================================================================
 * const defines
 * ===========================================================================
 */
static const double PI               = (3.14159265358979);
static const int    RC_MAX_INT       = 1024;    // max frame number, used to refresh encoder when frame number is not known
static const double RC_MAX_DELTA_QP  = 3.5;     // max delta QP between current key frame and its previous key frame

#define RC_LCU_LEVEL            0       // 1 - enable LCU level rate control, 0 - disable
#define RC_AUTO_ADJUST          0       // 1 - enable auto adjust the qp

// #define RC_MODEL_HISTORY        2
// #define RC_MAX_TEMPORAL_LEVELS  5


/**
* ===========================================================================
* type defines
* ===========================================================================
*/

#if RC_LCU_LEVEL
/* ---------------------------------------------------------------------------
*/
typedef struct RCLCU {
    int         GlobeLCUNum;
    int         CurBits;
    int         PreBits;
    double      PreBpp;
    double      CurBpp;
    double      LCUTBL;
    double      LCUTBLdelta;
    double      LCUBuffer;
    double      LCUBufferError;
    double      LCUBufferDifError;
    double      LCUPreBufferError;
    int         LCUPreLCUQP;
    int         LCUCurLCUQP;
    int         LCUdeltaQP;
    double      LCUCurLambda;
} RCLCU;
#endif

/* ---------------------------------------------------------------------------
*           |<---                 WIN                   --->|
* . . . . . I B B . . . F B B . . . F B B . . . F B B . . . I B B . . .
*           |<-- GOP -->|<-- GOP -->|<-- GOP -->|<-- GOP -->| B B . . .
*/

struct ratectrl_t {    //EKIN_MARK
    /* const */
    int         i_total_frames;       // total frames to be encoded (=0, forever)
    int         i_intra_period;       // period of I-frames (=0, only first)
    int         i_frame_size;         // frame size in pixel
    int         b_open_gop;           // open GOP? 1: open, 0: close

    /* qp */
    double      f_delta_qp;           // delta qp
    int         i_last_qp;            // qp for the last KEY frame
    int         i_base_qp;            // initial and base qp
    int         i_min_qp;             /* min QP */
    int         i_max_qp;             /* max QP */

    /* count */
    int         i_coded_frames;       // number of encoded frames

    /* gop/win */
    int         i_gop_flag;           // flag (index of first frame in GOP )
    int         i_win_size;           // size of WIN
    int         i_win_cnt;            // count of KEY frames  in current WIN (window length = a period of I-frames)
    int         i_win_qp;             // sum of KEY frame QP  in current WIN
    double      f_win_bpp;            // sum of KEY frame BPP in current WIN
    double      f_gop_bpp;            // sum of frame BPP in current GOP

    /* bpp */
    double      f_target_bpp;         // average target BBP (bit per pixel) for each frame
    double      f_intra_bpp;          // BPP of intra KEY frame (used only for i_intra_period = 0/1)
    double      f_inter_bpp;          // BPP of inter KEY frame (used only for i_intra_period = 0/1)

    /* buffer */
    double      f_buf_curr;           // current buffer size in BPP (bits per pixel)
    double      f_buf_error;          // buffer error
    double      f_buf_error_diff;     // different buffer error
    double      f_buf_error_prev;     // previous  buffer error

#if RC_AUTO_ADJUST
    /* level */
    double      f_first_buf_level;    // first buffer size level
    double      f_target_buf_level;   // target buffer level
    double      f_delta_buf_level;    // delta value of buffer level
#endif

#if RC_LCU_LEVEL
    /* LCU RC */
    int         RcMBQP;               //
    int         SumMBQP;              //
    int         NumMB;                //
    int         LCUbaseQP;            //
    RCLCU       rc_lcu;               //
#endif
    xavs2_thread_mutex_t rc_mutex;
};


/**
* ===========================================================================
* local/global variables
* ===========================================================================
*/

/* ---------------------------------------------------------------------------
*/
static const double tab_fuzzy_initial[13][13] = {
    {-4.80, -4.80, -4.80, -4.80, -3.57, -3.57, -3.17, -3.17, -2.00, -2.00, -0.25, -0.25, 0.00},
    {-4.80, -4.80, -4.80, -4.80, -3.57, -3.57, -3.17, -3.17, -2.00, -2.00, -0.25, -0.25, 0.00},
    {-4.80, -4.80, -3.57, -3.57, -3.57, -3.57, -2.00, -2.00, -1.10, -1.10,  0.00,  0.00, 0.25},
    {-4.80, -4.80, -3.57, -3.57, -3.57, -3.57, -2.00, -2.00, -1.10, -1.10,  0.00,  0.00, 0.25},
    {-3.57, -3.57, -3.57, -3.57, -2.00, -2.00, -1.10, -1.10,  0.00,  0.00,  1.10,  1.10, 2.00},
    {-3.57, -3.57, -3.57, -3.57, -2.00, -2.00, -1.10, -1.10,  0.00,  0.00,  1.10,  1.10, 2.00},
    {-3.17, -3.17, -2.00, -2.00, -1.10, -1.10,  0.00,  0.00,  1.10,  1.10,  2.00,  2.00, 3.17},
    {-3.17, -3.17, -2.00, -2.00, -1.10, -1.10,  0.00,  0.00,  1.10,  1.10,  2.00,  2.00, 3.17},
    {-2.00, -2.00, -1.10, -1.10,  0.00,  0.00,  1.10,  1.10,  2.00,  2.00,  3.57,  3.57, 3.57},
    {-2.00, -2.00, -1.10, -1.10,  0.00,  0.00,  1.10,  1.10,  2.00,  2.00,  3.57,  3.57, 3.57},
    {-0.25, -0.25,  0.00,  0.00,  1.10,  1.10,  2.44,  2.44,  3.57,  3.57,  3.57,  3.57, 4.80},
    {-0.25, -0.25,  0.00,  0.00,  1.10,  1.10,  2.44,  2.44,  3.57,  3.57,  3.57,  3.57, 4.80},
    { 0.00,  0.00,  0.25,  0.25,  2.00,  2.00,  3.57,  3.57,  3.86,  3.86,  4.80,  4.80, 4.80},
};

/* ---------------------------------------------------------------------------
*/
static double tab_fuzzy_qp_query[13][13];

#if ENABLE_AUTO_INIT_QP
/* ---------------------------------------------------------------------------
* table for getting initial qp via gpp */
static const double tab_qp_gpp[3][3] = {
    {5.656359783, 1.029364114, 0.120057248},
    {6.520734830, 1.191140657, 0.089733000},
    {5.494096438, 0.954657540, 0.111765010}
};
#endif


#if ENABLE_AUTO_INIT_QP
/* ---------------------------------------------------------------------------
* compute the gradient per pixel
*/
static double cal_frame_gradient(xavs2_frame_t *frm)
{
    double grad_per_pixel = 0;        // gradient per pixel
    pel_t *src = frm->planes[IMG_Y];// pointer to luma component
    int width = frm->i_width[IMG_Y];
    int height = frm->i_lines[IMG_Y];
    int stride = frm->i_stride[IMG_Y];
    int size = width * height;
    int i, j;

    width--;
    height--;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            int dx = src[j] - src[j + 1];
            int dy = src[j] - src[j + stride];

            if (dx || dy) {
                grad_per_pixel += sqrt((double)(dx * dx + dy * dy));
            }
        }
        src += stride;
    }

    return grad_per_pixel / size;
}
#endif

/* ---------------------------------------------------------------------------
*/
static void init_fuzzy_controller(double f_scale_factor)
{
    int i, j;

    for (i = 0; i < 13; i++) {
        for (j = 0; j < 13; j++) {
            tab_fuzzy_qp_query[i][j] = tab_fuzzy_initial[i][j] * f_scale_factor;
        }
    }
}

/* ---------------------------------------------------------------------------
*/
static
double fuzzy_get_delta_qp(double f_actual_val,
                          double f_delta_val,
                          double max_a, double min_a,
                          double max_b, double min_b)
{
    double dFuzAct = (12.0 / (max_a - min_a)) * (f_actual_val - (max_a + min_a) / 2.0);
    double dFuzDel = (12.0 / (max_b - min_b)) * (f_delta_val  - (max_b + min_b) / 2.0);
    int iFuzAct, iFuzDel;

    dFuzAct = XAVS2_CLIP3F(-6.0, 6.0, dFuzAct);
    dFuzDel = XAVS2_CLIP3F(-6.0, 6.0, dFuzDel);

    iFuzAct = (int)((dFuzAct < 0 ? floor(dFuzAct + 0.5) : ceil(dFuzAct - 0.5)) + 6);
    iFuzDel = (int)((dFuzDel < 0 ? floor(dFuzDel + 0.5) : ceil(dFuzDel - 0.5)) + 6);

    return tab_fuzzy_qp_query[iFuzAct][iFuzDel];
}

/* ---------------------------------------------------------------------------
*/
static double rc_calculate_gop_delta_qp(ratectrl_t *rc, int frm_type, int gop_len)
{
    double buf_range;
    double buf_range_delta;
    double tmp_bpp;

    /* get ERROR */
#if RC_AUTO_ADJUST
    rc->f_buf_error = rc->f_buf_curr - rc->f_target_buf_level;
#else
    rc->f_buf_error = rc->f_buf_curr;
#endif
    if ((rc->i_coded_frames % gop_len == 1) || (rc->i_intra_period == 1)) {
        rc->f_buf_error_diff = rc->f_buf_error - rc->f_buf_error_prev;
        rc->f_buf_error_prev = rc->f_buf_error;
    }

    /* get BPP */
    if (rc->i_intra_period > 1) {
        if (frm_type == XAVS2_TYPE_I) {
            tmp_bpp = rc->f_win_bpp;
        } else {
            tmp_bpp = rc->f_gop_bpp;
        }
    } else {
        if (rc->i_intra_period == 1) {
            tmp_bpp = rc->f_intra_bpp;
        } else { /*if (rc->i_intra_period == 0)*/
            tmp_bpp = rc->f_inter_bpp;
        }
    }

    /* get RANGE */
    buf_range = rc->i_coded_frames < 2 ? (tmp_bpp * 4) : (tmp_bpp / 4);
    buf_range = XAVS2_MAX(buf_range, 0.0001);

    if (rc->i_intra_period <= 1 || frm_type == XAVS2_TYPE_I) {
        buf_range_delta = buf_range * 2;
    } else {
        buf_range_delta = buf_range / 2;
    }

    return fuzzy_get_delta_qp(rc->f_buf_error, rc->f_buf_error_diff, buf_range, -buf_range, buf_range_delta, -buf_range_delta);
}

/**
* ---------------------------------------------------------------------------
* Function   : calculate the key frame QP
* Parameters :
*      [in ] : h        - handle of the xavs2 video encoder
*            : frm_idx  - frame index
*            : frm_type - frame type
*            : force_qp - specified qp for encoding current frame
*      [out] : none
* Return     : the key frame QP
* ---------------------------------------------------------------------------
*/
static int rc_calculate_frame_qp(xavs2_t *h, int frm_idx, int frm_type, int force_qp)
{
    ratectrl_t *rc = h->rc;
    const int max_qp = rc->i_max_qp + (h->param->sample_bit_depth - 8) * 8;
    const int remain_frames = rc->i_total_frames - rc->i_coded_frames;
    double qp;

    assert(h->param->i_rc_method != XAVS2_RC_CQP);

    /* call before floating point arithmetic */
    xavs2_emms();

    /* the initial frame QP */
    if (h->param->enable_refine_qp && h->param->intra_period_max > 1) {
        qp = 5.661 * log(h->f_lambda_mode) + 13.131;
    } else {
        qp = h->i_qp;
    }

    /* Update Frame count when I frame is found */
    if (frm_type == XAVS2_TYPE_I && rc->i_coded_frames > (RC_MAX_INT >> 1)) {
        rc->i_total_frames = RC_MAX_INT;
        rc->i_coded_frames = 0;
    }
    /* size is changed from the 2nd WIN for OPEN GOP */
    if (frm_type == XAVS2_TYPE_I && frm_idx > 0 && rc->b_open_gop) {
        rc->i_win_size = h->i_gop_size * rc->i_intra_period;
    }

#if ENABLE_AUTO_INIT_QP
    /* compute the initial qp */
    if (frm_idx == 0) {
        double bit = log(1000 * rc->f_target_bpp);
        double gpp = log(cal_frame_gradient(h->fenc));
        int    idx = XAVS2_MIN(2, rc->i_intra_period);
        int    max_i_qp = 63 + (h->param->sample_bit_depth - 8) * 8 - 10;

        qp = (tab_qp_gpp[idx][0] + tab_qp_gpp[idx][1] * gpp - bit) / tab_qp_gpp[idx][2];
        qp = XAVS2_CLIP3F(20, max_i_qp, qp);
        rc->i_base_qp = (int)(qp + 0.5);    // reset the QP in encoder parameters
    }
#endif

    /* compute the delta qp */
    if (rc->i_intra_period == 0) {
        if (frm_idx % h->i_gop_size == 1) {
            rc->f_delta_qp = rc_calculate_gop_delta_qp(rc, frm_type, h->i_gop_size);
        }
    } else if (rc->i_intra_period == 1) {
        if ((frm_idx % h->i_gop_size == 0) && (frm_idx != 0)) {
            rc->f_delta_qp = rc_calculate_gop_delta_qp(rc, frm_type, h->i_gop_size);
        }
    } else {
        if ((frm_type == XAVS2_TYPE_I) && (remain_frames <= (2 * rc->i_win_size))) {
            init_fuzzy_controller(0.50);    // enhance adjusting strength of the last WIN
        }
        if ((frm_idx % h->i_gop_size == 0) && (frm_idx != 0)) {
            rc->f_delta_qp = rc_calculate_gop_delta_qp(rc, frm_type, h->i_gop_size);
        } else if (remain_frames == (rc->i_total_frames - 1) % h->i_gop_size) {
            rc->f_delta_qp = rc_calculate_gop_delta_qp(rc, frm_type, h->i_gop_size);
        }
    }

    if ((frm_idx && frm_idx % h->i_gop_size == 0) || (rc->i_intra_period == 1)) {
        if (rc->i_intra_period > 1 && frm_type == XAVS2_TYPE_I) {
#if RC_AUTO_ADJUST
            double remain_gop_num;

            /* adjust the delta QP according to the final WIN */
            if (remain_frames > rc->i_win_size + h->i_gop_size) {
                remain_gop_num = rc->i_intra_period;
            } else {
                remain_gop_num = ceil((double)remain_frames / (double)h->i_gop_size);
            }

            if (remain_frames <= rc->i_win_size + h->i_gop_size) {
                remain_gop_num /= rc->i_intra_period;
                if (remain_gop_num < 1.0 / 3) {
                    rc->f_delta_qp += 4.2;  // as bitrate halve
                } else if (remain_gop_num < 1.0 / 2) {
                    rc->f_delta_qp += 3.4;
                } else if (remain_gop_num < 2.0 / 3) {
                    rc->f_delta_qp += 2.6;
                } else if (remain_gop_num < 3.0 / 4) {
                    rc->f_delta_qp += 1.8;
                } else if (remain_gop_num < 4.0 / 4) {
                    rc->f_delta_qp += 1.0;
                }
            }
#endif
            /* calculate the average QP of all KEY frames in last WIN */
            qp = (double)rc->i_win_qp / rc->i_win_cnt + rc->f_delta_qp;
            rc->i_base_qp = (int)(qp + 0.5);    // reset the QP in encoder parameters
        } else {
            /* handle middle GOPs */
            qp += rc->f_delta_qp;
            rc->i_base_qp += (int)(rc->f_delta_qp + 0.5);   // also fix the QP in encoder parameters
        }

        qp = XAVS2_CLIP3F(rc->i_min_qp, max_qp, qp);
        rc->i_base_qp = XAVS2_CLIP3F(rc->i_min_qp, max_qp, rc->i_base_qp);
    }

    if (force_qp != XAVS2_QP_AUTO) {
        qp = force_qp - 1;
    }

    // check the QP
    if (rc->i_coded_frames > 0 && frm_type != XAVS2_TYPE_B) {
        qp = XAVS2_CLIP3F(rc->i_last_qp - RC_MAX_DELTA_QP, rc->i_last_qp + RC_MAX_DELTA_QP, qp);
    }

    return XAVS2_CLIP3F(rc->i_min_qp, max_qp, (int)(qp + 0.5));
}

/* ---------------------------------------------------------------------------
*/
#if RC_LCU_LEVEL
static void Init_LCURateControl(ratectrl_t *rc, int NumUnitsLCU)
{
    rc->rc_lcu.LCUTBL = rc->f_target_buf_level;
    rc->rc_lcu.LCUBuffer = rc->f_buf_curr;
    rc->rc_lcu.LCUTBLdelta = rc->f_delta_buf_level / NumUnitsLCU;
    rc->rc_lcu.LCUBufferError = 0;
    rc->rc_lcu.LCUBufferDifError = 0;
    rc->rc_lcu.LCUPreBufferError = 0;
}

/* ---------------------------------------------------------------------------
*/
static int CalculateLCUDeltaQP(ratectrl_t *rc)
{
    if (rc->i_intra_period <= 1) { // lcu level RC does not support RA now.
        double belta = 0.12;
        double tmp_bpp = rc->f_target_bpp;
        double buf_range = tmp_bpp * belta * 2;
        double buf_range_delta;

        buf_range = buf_range < 0.0001 ? 0.0001 : buf_range;
        buf_range_delta = buf_range * 2;

        return fuzzy_get_delta_qp(rc->rc_lcu.LCUBufferError, rc->rc_lcu.LCUBufferDifError, buf_range, -buf_range, buf_range_delta, -buf_range_delta);
    }

    return 0;
}

/* ---------------------------------------------------------------------------
*/
static void UpdataLCURateControl(ratectrl_t *rc, int qp, double lambda, int bits, int NumLCU)
{
    rc->rc_lcu.PreBpp = rc->rc_lcu.CurBpp;
    rc->rc_lcu.CurBpp = ((double)bits) / rc->i_frame_size;
    rc->rc_lcu.LCUTBL -= rc->rc_lcu.LCUTBLdelta;
    rc->rc_lcu.LCUBuffer = rc->rc_lcu.LCUBuffer + rc->rc_lcu.CurBpp - rc->f_target_bpp / NumLCU;
    rc->rc_lcu.LCUBufferError = rc->rc_lcu.LCUBuffer - rc->rc_lcu.LCUTBL;
    rc->rc_lcu.LCUBufferDifError = rc->rc_lcu.LCUBufferError - rc->rc_lcu.LCUPreBufferError;
    rc->rc_lcu.LCUPreBufferError = rc->rc_lcu.LCUBufferError;
    rc->rc_lcu.LCUPreLCUQP = rc->rc_lcu.LCUCurLCUQP;
    rc->rc_lcu.LCUCurLCUQP = qp;
    rc->rc_lcu.LCUCurLambda = lambda;
}
#endif


/**
* ===========================================================================
* interface function defines
* ===========================================================================
*/

/**
 * ---------------------------------------------------------------------------
 * Function   : get buffer size for rate control module
 * Parameters :
 *      [in ] : param - handle of the xavs2 encoder
 *      [out] : none
 * Return     : return > 0 on success, 0/-1 on failure
 * ---------------------------------------------------------------------------
 */
int xavs2_rc_get_buffer_size(xavs2_param_t *param)
{
    UNUSED_PARAMETER(param);
    return sizeof(ratectrl_t);
}

/**
 * ---------------------------------------------------------------------------
 * Function   : create and init the rate control module
 * Parameters :
 *      [in ] : h - handle of the xavs2 encoder
 *      [out] : none
 * Return     : return 0 on success, -1 on failure
 * ---------------------------------------------------------------------------
 */
int xavs2_rc_init(ratectrl_t *rc, xavs2_param_t *param)
{
    /* clear memory for rate control handle */
    memset(rc, 0, sizeof(ratectrl_t));

    if (param->i_rc_method == XAVS2_RC_CBR_SCU && param->intra_period_max > 1) {
        param->i_rc_method = XAVS2_RC_CBR_FRM;
        xavs2_log(NULL, XAVS2_LOG_WARNING, "LCU Rate Control does not support RA. Using Frame RC. \n");
    }

    // init
    rc->i_total_frames = param->num_frames == 0 ? RC_MAX_INT : param->num_frames;
    rc->i_total_frames = XAVS2_MIN(RC_MAX_INT, param->num_frames);
    rc->i_coded_frames = 0;
    rc->i_intra_period = param->intra_period_max;
    rc->i_frame_size = param->org_width * param->org_height;
    rc->b_open_gop = param->b_open_gop;

    rc->f_delta_qp = 0.0;
    rc->i_base_qp = param->i_initial_qp;
    rc->i_last_qp = 0;
    rc->i_min_qp = param->i_min_qp;
    rc->i_max_qp = param->i_max_qp;

    rc->i_coded_frames = 0;

    rc->i_gop_flag = -1024;
    rc->i_win_cnt = 0;
    rc->i_win_qp = 0;
    rc->f_win_bpp = 0.0;
    rc->f_gop_bpp = 0.0;

    rc->f_target_bpp = param->i_target_bitrate / (param->frame_rate * rc->i_frame_size);
    rc->f_intra_bpp = 0.0;
    rc->f_inter_bpp = 0.0;

    rc->f_buf_curr = 0.0;
    rc->f_buf_error = 0.0;
    rc->f_buf_error_diff = 0.0;
    rc->f_buf_error_prev = 0.0;

#if RC_AUTO_ADJUST
    rc->f_first_buf_level = 0.0;
    rc->f_target_buf_level = 0.0;
    rc->f_delta_buf_level = 0.0;
#endif

    // set size of WIN (intra period)
    rc->i_win_size = param->i_gop_size * (rc->i_intra_period - 1) + 1;

    // init table of fuzzy controller
    if (rc->i_intra_period == 1) {
        init_fuzzy_controller(0.85);
    } else {
        init_fuzzy_controller(0.75);
    }

    if (xavs2_thread_mutex_init(&rc->rc_mutex, NULL)) {
        return -1;
    }

    return 0;
}

/**
* ---------------------------------------------------------------------------
* Function   : get base qp of the encoder
* Parameters :
*      [in ] : h - handle of the xavs2 video encoder
*      [out] : none
* Return     : none
* ---------------------------------------------------------------------------
*/
int xavs2_rc_get_base_qp(xavs2_t *h)
{
    return h->rc->i_base_qp;       // return the base qp directly
}

/**
* ---------------------------------------------------------------------------
* Function   : get frame qp
* Parameters :
*      [in ] : h - handle of the xavs2 video encoder
*      [out] : none
* Return     : none
* ---------------------------------------------------------------------------
*/
int xavs2_rc_get_frame_qp(xavs2_t *h, int frm_idx, int frm_type, int force_qp)
{
    /* get QP for current frame */
    if (h->param->i_rc_method != XAVS2_RC_CQP && frm_type != XAVS2_TYPE_B) {
        int i_qp;
        xavs2_thread_mutex_lock(&h->rc->rc_mutex);
        i_qp = rc_calculate_frame_qp(h, frm_idx, frm_type, force_qp);
        xavs2_thread_mutex_unlock(&h->rc->rc_mutex);
        return i_qp;
    } else {
        return h->i_qp;         // return the old value directly
    }
}

/**
* ---------------------------------------------------------------------------
* Function   : get qp for one lcu
* Parameters :
*      [in ] : h       - handle of the xavs2 video encoder
*            : frm_idx - frame index
*            : qp      - basic QP of the LCU to be encoded
*      [out] : none
* Return     : adjusted qp of the LCU
* ---------------------------------------------------------------------------
*/
int xavs2_rc_get_lcu_qp(xavs2_t *h, int frm_idx, int qp)
{
    UNUSED_PARAMETER(h);
    UNUSED_PARAMETER(frm_idx);

    //if (h->param->i_rc_method == XAVS2_RC_CBR_SCU && img->current_mb_nr == 0) {
    //    Init_LCURateControl(rc, num_of_orgMB);
    //}

#if RC_LCU_LEVEL
    if (h->param->i_rc_method == XAVS2_RC_CBR_SCU) {
        ratectrl_t *rc = h->rc;
        double lambda_mode = 0.5;
        int current_mb_nr = 0;  /* FIX: current LCU index */

        if (current_mb_nr == 0) {
            rc->SumMBQP = rc->NumMB = rc->LCUbaseQP = 0;
        }

        rc->RcMBQP = rc->LCUbaseQP = qp;

        if (rc->i_intra_period == 1) {
            if (current_mb_nr == 0) {
                rc->RcMBQP = qp;
            } else {
                rc->rc_lcu.LCUdeltaQP = CalculateLCUDeltaQP(rc);
                rc->RcMBQP = qp + rc->rc_lcu.LCUdeltaQP;
                rc->RcMBQP = XAVS2_MAX(qp - 3, XAVS2_MIN(rc->RcMBQP, qp + 3));
            }
        }

        if (rc->i_intra_period == 0) {
            if (frm_idx == 0) {
                rc->RcMBQP = qp;
            } else {
                rc->rc_lcu.LCUdeltaQP = CalculateLCUDeltaQP(rc);
                rc->RcMBQP = qp + rc->rc_lcu.LCUdeltaQP;
                rc->RcMBQP = XAVS2_MAX(qp - 5, XAVS2_MIN(rc->RcMBQP, qp + 5));
            }
        }

        if (rc->i_intra_period > 1) {
            if (frm_idx == 0) {
                rc->RcMBQP = qp;
            } else {
                rc->rc_lcu.LCUdeltaQP = CalculateLCUDeltaQP(rc);
                rc->RcMBQP = qp + rc->rc_lcu.LCUdeltaQP;
                rc->RcMBQP = XAVS2_MAX(qp - 5, XAVS2_MIN(rc->RcMBQP, qp + 5));
            }
        }

        lambda_mode *= pow(2, (rc->RcMBQP - qp) / 4.0);
        rc->RcMBQP = XAVS2_MAX(rc->i_min_qp, XAVS2_MIN(rc->RcMBQP, rc->i_max_qp + (h->param->sample_bit_depth - 8) * 8));

        rc->SumMBQP += rc->RcMBQP;
        rc->NumMB++;
    }
#endif

    return qp;
}


/**
* ---------------------------------------------------------------------------
* Function   : save stats and update rate control state after encoding a LCU block
* Parameters :
*      [in ] : h       - handle of the xavs2 video encoder
*            : frm_idx - frame index
*            : qp      - QP of the encoded LCU
*            : bits    - number of bits of the encoded LCU
* Return     : none
* ---------------------------------------------------------------------------
*/
void xavs2_rc_update_after_lcu_coded(xavs2_t *h, int frm_idx, int qp)
{
    UNUSED_PARAMETER(h);
    UNUSED_PARAMETER(frm_idx);
    UNUSED_PARAMETER(qp);

#if RC_LCU_LEVEL
    if (h->param->i_rc_method == XAVS2_RC_CBR_SCU) {
        ratectrl_t *rc = h->rc;
        int LCUbits;

        if (img->current_mb_nr == 0) {
            rc->rc_lcu.PreBits = 0;
        }

        rc->rc_lcu.CurBits = currBitStream->byte_pos * 8;
        LCUbits = rc->rc_lcu.CurBits - rc->rc_lcu.PreBits;
        rc->rc_lcu.PreBits = rc->rc_lcu.CurBits;

        UpdataLCURateControl(rc, rc->RcMBQP, lambda_mode, LCUbits, numLCUInPicWidth * numLCUInPicHeight);
    }
#endif
}

/**
* ---------------------------------------------------------------------------
* Function   : save stats and update rate control state after encoding a frame
* Parameters :
*      [in ] : h        - handle of the xavs2 video encoder
*            : frm_bits - number of bits of the encoded frame
*            : frm_qp   - average QP of the encoded frame
*            : frm_type - frame type
*            : frm_idx  - frame index
* Return     : none
* ---------------------------------------------------------------------------
*/
void xavs2_rc_update_after_frame_coded(xavs2_t *h, int frm_bits, int frm_qp, int frm_type, int frm_idx)
{
    ratectrl_t *rc = h->rc;
    double frm_bpp = (double)frm_bits / rc->i_frame_size;   // bits per pixel

    if (h->param->i_rc_method == XAVS2_RC_CQP) {
        return;                 /* no need to update */
    }

    xavs2_thread_mutex_lock(&rc->rc_mutex);   // lock

#if RC_LCU_LEVEL
    if (h->param->i_rc_method == XAVS2_RC_CBR_SCU) {
        frm_qp = (int)((0.5 + rc->SumMBQP) / rc->NumMB);
    }
#endif

    /* update */
    rc->i_coded_frames++;       /* sum up number of encoded frames */
    rc->f_buf_curr += frm_bpp - rc->f_target_bpp;   /* sum up buffer ERROR */
    if (frm_type != XAVS2_TYPE_B) {
        if (frm_type == XAVS2_TYPE_I) {
            /* reset for the WIN */
            rc->f_intra_bpp = frm_bpp;
            rc->i_win_qp = frm_qp;
            rc->f_win_bpp = frm_bpp;
            rc->i_win_cnt = 1;
        } else {
            /* sum up in the WIN */
            rc->f_inter_bpp = frm_bpp;
            rc->i_win_qp += frm_qp;
            rc->f_win_bpp += frm_bpp;
            rc->i_win_cnt++;
        }
        rc->i_last_qp = frm_qp;
        rc->f_gop_bpp = frm_bpp;      /* reset for a GOP */
    } else {
        rc->f_gop_bpp += frm_bpp;     /* sum up in a GOP */
    }

#if RC_AUTO_ADJUST
    /* adjust */
    if (rc->i_intra_period == 1) {
        rc->f_target_buf_level = rc->f_delta_buf_level = 0.0;
    } else if (rc->i_intra_period == 0) {
        if (frm_type == XAVS2_TYPE_I) {
            rc->f_target_buf_level = rc->f_buf_curr;
            rc->f_delta_buf_level = rc->f_target_buf_level / (rc->i_total_frames + XAVS2_MAX(0, XAVS2_MIN(150, (rc->i_total_frames - 150) / 3)));
        } else {
            rc->f_target_buf_level = rc->f_target_buf_level - rc->f_delta_buf_level;
        }
    } else if (rc->i_intra_period == 2 || rc->i_intra_period == 3) {
        if (frm_type == XAVS2_TYPE_I && rc->i_coded_frames < 2) {
            rc->f_first_buf_level = rc->f_buf_curr;
            rc->f_target_buf_level = rc->f_buf_curr;
        } else {
            rc->f_target_buf_level = rc->f_first_buf_level * cos(PI / 2 * rc->i_coded_frames / rc->i_total_frames);
        }
    } else {
        const int remain_frames = rc->i_total_frames - rc->i_coded_frames;
        int LevelLength;

        if (frm_type == XAVS2_TYPE_I && rc->i_coded_frames < 2) {
            /* in the first WIN, after encoding the frame I */
            LevelLength = XAVS2_MIN(rc->i_win_size - 1, rc->i_total_frames - frm_idx);
            rc->f_target_buf_level = rc->f_buf_curr;
            rc->f_delta_buf_level = rc->f_buf_curr / LevelLength;
        } else if (frm_type == XAVS2_TYPE_I && rc->i_coded_frames >= 2 && (remain_frames > rc->i_win_size + h->i_gop_size)) {
            /* in the middle WIN, after encoding the frame I */
            rc->i_gop_flag = rc->i_coded_frames;            /* store the position */
            rc->f_target_buf_level = rc->f_delta_buf_level = 0.0;   /* not adjust the buffer level */
        } else if (frm_type == XAVS2_TYPE_I && rc->i_coded_frames >= 2 && (remain_frames <= rc->i_win_size + h->i_gop_size)) {
            /* in the final WIN, after encoding the frame I */
            rc->f_target_buf_level = rc->f_buf_curr;
            rc->f_delta_buf_level = rc->f_buf_curr / remain_frames;
        } else if (rc->i_gop_flag == rc->i_coded_frames - h->i_gop_size) {
            /* in the middle WIN, after encoding the first GOP */
            if (remain_frames <= rc->i_win_size) {
                LevelLength = remain_frames;
            } else {
                LevelLength = (rc->i_intra_period - 1) * h->i_gop_size;
            }
            rc->f_target_buf_level = rc->f_buf_curr;
            rc->f_delta_buf_level = rc->f_buf_curr / LevelLength;
            rc->i_gop_flag = -1024;
        } else {
            rc->f_target_buf_level = rc->f_target_buf_level - rc->f_delta_buf_level;
        }
    }
#else
    UNUSED_PARAMETER(frm_idx);
#endif

    xavs2_thread_mutex_unlock(&rc->rc_mutex);  // unlock
}

/**
* ---------------------------------------------------------------------------
* Function   : destroy the rate control
* Parameters :
*      [in ] : rc - handle of the ratecontrol handler
*      [out] : none
* Return     : none
* ---------------------------------------------------------------------------
*/
void xavs2_rc_destroy(ratectrl_t *rc)
{
    xavs2_thread_mutex_destroy(&rc->rc_mutex);
}
