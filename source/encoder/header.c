/*
 * header.c
 *
 * Description of this file:
 *    Header writing functions definition of the xavs2 library
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
#include "header.h"
#include "wquant.h"
#include "bitstream.h"
#include "aec.h"


/**
 * ===========================================================================
 * function defines
 * ===========================================================================
 */


/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE int is_valid_qp(xavs2_t *h, int i_qp)
{
    int max_qp = MAX_QP;
    UNUSED_PARAMETER(h);
    return i_qp >= 0 && i_qp <= max_qp;
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
int get_frame_coi_to_write(xavs2_t *h, xavs2_frame_t *frm)
{
    if (h->param->num_parallel_gop > 1) {
        return (frm->i_frm_coi - frm->i_gop_idr_coi) & 255;
    } else {
        return frm->i_frm_coi & 255;
    }
}

/* ---------------------------------------------------------------------------
 * write sequence header information
 */
int xavs2_sequence_write(xavs2_t *h, bs_t *p_bs)
{
    const xavs2_rps_t *p_seq_rps = h->param->cfg_ref_all;
    int bits = 0;
    int i, j;

    bits += u_0(p_bs, 32, 0x1b0,                                    "seqence_start_code");
    bits += u_v(p_bs,  8, h->param->profile_id,                      "profile_id");
    bits += u_v(p_bs,  8, h->param->level_id,                        "level_id");
    bits += u_v(p_bs,  1, h->param->progressive_sequence,            "progressive_sequence");
    bits += u_v(p_bs,  1, h->b_field_sequence,                      "field_coded_sequence");
    bits += u_v(p_bs, 14, h->param->org_width,                       "horizontal_size");
    bits += u_v(p_bs, 14, h->param->org_height,                      "vertical_size");
    bits += u_v(p_bs,  2, h->param->chroma_format,                   "chroma_format");

    bits += u_v(p_bs,  3, h->param->sample_precision,                "sample_precision");
    if (h->param->profile_id == MAIN10_PROFILE) { // MAIN10 profile
        bits += u_v(p_bs, 3, ((h->param->sample_bit_depth - 6) / 2), "encoding_precision");
    }

    bits += u_v(p_bs,  4, h->param->aspect_ratio_information,        "aspect_ratio_information");
    bits += u_v(p_bs,  4, h->param->frame_rate_code,                 "frame_rate_code");
    bits += u_v(p_bs, 18, h->param->bitrate_lower,                   "bit_rate_lower");
    bits += u_v(p_bs,  1, 1,                                        "marker bit");
    bits += u_v(p_bs, 12, h->param->bitrate_upper,                   "bit_rate_upper");
    bits += u_v(p_bs,  1, h->param->low_delay,                       "low_delay");
    bits += u_v(p_bs,  1, 1,                                        "marker bit");
    bits += u_v(p_bs,  1, h->param->temporal_id_exist_flag,          "temporal_id exist flag");
    bits += u_v(p_bs, 18, h->param->bbv_buffer_size,                 "bbv buffer size");
    bits += u_v(p_bs,  3, h->i_lcu_level,                           "Largest Coding Block Size");
    bits += u_v(p_bs,  1, h->param->enable_wquant,                   "weight_quant_enable");

#if ENABLE_WQUANT
    if (h->param->enable_wquant) {
        bits += u_v(p_bs, 1, h->param->SeqWQM,                       "load_seq_weight_quant_data_flag");
        if (h->param->SeqWQM) {
            int x, y, sizeId, iWqMSize;
            wq_data_t *wq = &h->wq_data;

            for (sizeId = 0; sizeId < 2; sizeId++) {
                iWqMSize = XAVS2_MIN(1 << (sizeId + 2), 8);
                for (y = 0; y < iWqMSize; y++) {
                    for (x = 0; x < iWqMSize; x++) {
                        bits += ue_v(p_bs, wq->seq_wq_matrix[sizeId][y * iWqMSize + x], "weight_quant_coeff");
                    }
                }
            }
        }
    }
#else
    assert(h->param->enable_wquant == 0);
#endif

    bits += u_v(p_bs, 1, 1,                                         "background_picture_disable");
    bits += u_v(p_bs, 1, h->param->enable_mhp_skip,                  "mhpskip enabled");
    bits += u_v(p_bs, 1, h->param->enable_dhp,                       "dhp enabled");
    bits += u_v(p_bs, 1, h->param->enable_wsm,                       "wsm enabled");
    bits += u_v(p_bs, 1, h->param->enable_amp,                       "Asymmetric Motion Partitions");
    bits += u_v(p_bs, 1, h->param->enable_nsqt,                      "enable_NSQT");
    bits += u_v(p_bs, 1, h->param->enable_sdip,                      "enable_SDIP");
    bits += u_v(p_bs, 1, h->param->enable_secT,                      "secT enabled");
    bits += u_v(p_bs, 1, h->param->enable_sao,                       "SAO Enable Flag");
    bits += u_v(p_bs, 1, h->param->enable_alf,                       "ALF Enable Flag");
    bits += u_v(p_bs, 1, h->param->enable_pmvr,                      "pmvr enabled");

    bits += u_v(p_bs, 1, 1,                                         "marker bit");

    bits += u_v(p_bs, 6, h->i_gop_size,                             "num_of_RPS");

    for (i = 0; i < h->i_gop_size; i++) {
        bits += u_v(p_bs, 1, p_seq_rps[i].referd_by_others,         "refered by others");
        bits += u_v(p_bs, 3, p_seq_rps[i].num_of_ref,               "num of reference picture");

        for (j = 0; j < p_seq_rps[i].num_of_ref; j++) {
            bits += u_v(p_bs, 6, p_seq_rps[i].ref_pic[j],           "delta COI of ref pic");
        }
        bits += u_v(p_bs, 3, p_seq_rps[i].num_to_rm,                "num of removed picture");
        for (j = 0; j < p_seq_rps[i].num_to_rm; j++) {
            bits += u_v(p_bs, 6, p_seq_rps[i].rm_pic[j],            "delta COI of removed pic");
        }

        bits += u_v(p_bs, 1, 1,                                     "marker bit");
    }

    if (!h->param->low_delay) {
        bits += u_v(p_bs, 5, h->picture_reorder_delay,              "output_reorder_delay");
    }

    bits += u_v(p_bs, 1, h->param->b_cross_slice_loop_filter,        "Cross Loop Filter Flag");
    bits += u_v(p_bs, 3, 0,                                         "reserved bits");

    /* byte align */
    bits += bs_byte_align(p_bs);

    //xavs2_log(h, XAVS2_LOG_INFO, "Sequence Header, inserted before frame %d, COI %d\n", h->fenc->i_frame, h->curr_coi);
    return bits;
}

/* ---------------------------------------------------------------------------
 * write user data
 */
int xavs2_user_data_write(bs_t *p_bs)
{
    const char *avs2_log = "xavs2 encoder";
    int bits;

    bits = u_0(p_bs, 32, 0x1b2,             "user data start code");
    while (*avs2_log) {
        bits += u_v(p_bs, 8, *avs2_log++,   "user data");
    }
    bits += bs_byte_align(p_bs);

    return bits;
}

/* ---------------------------------------------------------------------------
 */
int xavs2_intra_picture_header_write(xavs2_t *h, bs_t *p_bs)
{
    int bbv_delay = 0xFFFF;
    int display_delay = 0;
    int len;
    int i;

    len  = u_0(p_bs, 32, 0x1B3,                                     "I picture start code");
    len += u_v(p_bs, 32, bbv_delay,                                 "bbv_delay");
    len += u_v(p_bs, 1, h->param->time_code_flag,                    "time_code_flag");

    // if (h->param->time_code_flag) {
    //     tc = frametotc(h, frame, h->tc_reserve_bit);
    //     len += u_v(p_bs, 24, tc, "time_code");
    // }

    if (!h->param->low_delay) {
        display_delay = h->fdec->i_frame - h->fdec->i_frm_coi + h->picture_reorder_delay;
    }

    len += u_v(p_bs, 8, get_frame_coi_to_write(h, h->fdec),         "coding_order");

    if (h->param->temporal_id_exist_flag == 1) {
        len += u_v(p_bs, TEMPORAL_MAXLEVEL_BIT, h->i_layer,         "temporal_id");
    }

    if (!h->param->low_delay) {
        len += ue_v(p_bs, display_delay,                            "picture_output_delay");
    }

    len += u_v(p_bs, 1, h->fdec->rps.idx_in_gop >= 0,               "use RCS in SPS");
    if (h->fdec->rps.idx_in_gop >= 0) {
        len += u_v(p_bs, 5, h->fdec->rps.idx_in_gop,                "predict for RCS");
    } else {
        len += u_v(p_bs, 1, h->fdec->rps.referd_by_others,          "referenced by others");

        len += u_v(p_bs, 3, h->fdec->rps.num_of_ref,                "num of reference picture");
        for (i = 0; i < h->fdec->rps.num_of_ref; i++) {
            len += u_v(p_bs, 6, h->fdec->rps.ref_pic[i],            "delta COI of ref pic");
        }

        len += u_v(p_bs, 3, h->fdec->rps.num_to_rm,                 "num of removed picture");
        for (i = 0; i < h->fdec->rps.num_to_rm; i++) {
            len += u_v(p_bs, 6, h->fdec->rps.rm_pic[i],             "delta COI of removed pic");
        }
        len += u_v(p_bs, 1, 1, "marker bit");
    }

    if (h->param->low_delay) {
        len += ue_v(p_bs, 0,                                        "bbv check times");
    }

    len += u_v(p_bs, 1, h->b_progressive,                           "progressive_frame");
    if (!h->b_progressive) {
        len += u_v(p_bs, 1, 1,                                      "picture_structure");
    }

    len += u_v(p_bs, 1, h->param->top_field_first,                   "top_field_first");
    len += u_v(p_bs, 1, h->param->repeat_first_field,                "repeat_first_field");
    if (h->param->InterlaceCodingOption == FIELD_CODING) {
        len += u_v(p_bs, 1, h->b_top_field,                         "is top field");
        len += u_v(p_bs, 1, 1,                                      "reserved bit for interlace coding");
    }

    len += u_v(p_bs, 1, h->param->fixed_picture_qp,                  "fixed_picture_qp");
    len += u_v(p_bs, 7, h->i_qp,                                    "picture_qp");

    len += u_v(p_bs, 1, h->param->loop_filter_disable,               "loop_filter_disable");
    if (!h->param->loop_filter_disable) {
        len += u_v(p_bs, 1, h->param->loop_filter_parameter_flag,    "loop_filter_parameter_flag");
        if (h->param->loop_filter_parameter_flag) {
            len += se_v(p_bs, h->param->alpha_c_offset,              "alpha offset");
            len += se_v(p_bs, h->param->beta_offset,                 "beta offset");
        }
    }

#if ENABLE_WQUANT
    len += u_v(p_bs, 1, h->param->chroma_quant_param_disable, "chroma_quant_param_disable");
    if (!h->param->chroma_quant_param_disable) {
        len += se_v(p_bs, h->param->chroma_quant_param_delta_u, "chroma_quant_param_delta_cb");
        len += se_v(p_bs, h->param->chroma_quant_param_delta_v, "chroma_quant_param_delta_cr");
    } else {
        assert(h->param->chroma_quant_param_delta_u == 0);
        assert(h->param->chroma_quant_param_delta_v == 0);
    }
#else
    len += u_v(p_bs, 1, 1, "chroma_quant_param_disable");
#endif // ENABLE_WQUANT

    if (!is_valid_qp(h, h->i_qp)) {
        xavs2_log(h,XAVS2_LOG_ERROR,"Invalid I Picture QP: %d\n",h->i_qp);
    }

#if ENABLE_WQUANT
    // adaptive frequency weighting quantization
    if (h->param->enable_wquant) {
        len += u_v(p_bs, 1, h->param->PicWQEnable,                   "pic_weight_quant_enable");
        if (h->param->PicWQEnable) {
            len += u_v(p_bs, 2, h->param->PicWQDataIndex,            "pic_weight_quant_data_index");
            if (h->param->PicWQDataIndex == 1) {
                len += u_v(p_bs, 1, 0,                              "reserved_bits");

                len += u_v(p_bs, 2, h->param->WQParam,               "weighting_quant_param_index");
                len += u_v(p_bs, 2, h->param->WQModel,               "weighting_quant_model");

                if ((h->param->WQParam == 1) || ((h->param->MBAdaptQuant) && (h->param->WQParam == 3))) {
                    for (i = 0; i < 6; i++) {
                        len += se_v(p_bs, (int)(h->wq_data.wq_param[UNDETAILED][i] - tab_wq_param_default[UNDETAILED][i]), "quant_param_delta_u");
                    }
                }
                if ((h->param->WQParam == 2) || ((h->param->MBAdaptQuant) && (h->param->WQParam == 3))) {
                    for (i = 0; i < 6; i++) {
                        len += se_v(p_bs, (int)(h->wq_data.wq_param[DETAILED][i] - tab_wq_param_default[DETAILED][i]), "quant_param_delta_d");
                    }
                }
            } else if (h->param->PicWQDataIndex == 2) {
                int x, y, sizeId, iWqMSize;
                for (sizeId = 0; sizeId < 2; sizeId++) {
                    i = 0;
                    iWqMSize = XAVS2_MIN(1 << (sizeId + 2), 8);
                    for (y = 0; y < iWqMSize; y++) {
                        for (x = 0; x < iWqMSize; x++) {
                            len += ue_v(p_bs, h->wq_data.pic_user_wq_matrix[sizeId][i++], "weight_quant_coeff");
                        }
                    }
                }
            }
        }
    }
#endif

    return len;
}

/* ---------------------------------------------------------------------------
 */
int xavs2_inter_picture_header_write(xavs2_t *h, bs_t *p_bs)
{
    int bbv_delay = 0xFFFF;
    int picture_coding_type;
    int display_delay = 0;
    int len;
    int i;

    if (h->i_type == SLICE_TYPE_P) {
        picture_coding_type = 1;
    } else if (h->i_type == SLICE_TYPE_F) {
        picture_coding_type = 3;
    } else {
        picture_coding_type = 2;
    }

    len  = u_0(p_bs, 24, 1,                                         "start_code_prefix");
    len += u_0(p_bs,  8, 0xB6,                                      "picture start code");
    len += u_v(p_bs, 32, bbv_delay,                                 "bbv delay");
    len += u_v(p_bs,  2, picture_coding_type,                       "picture_coding_type");

    if (!h->param->low_delay) {
        display_delay = h->fenc->i_frame - h->fdec->i_frm_coi + h->picture_reorder_delay;
    }

    len += u_v(p_bs, 8, get_frame_coi_to_write(h, h->fdec),         "coding_order");
    if (h->param->temporal_id_exist_flag == 1) {
        len += u_v(p_bs, TEMPORAL_MAXLEVEL_BIT, h->i_layer,         "temporal_id");
    }

    if (!h->param->low_delay) {
        len += ue_v(p_bs, display_delay,                            "displaydelay");
    }

    len += u_v(p_bs, 1, h->fdec->rps.idx_in_gop >= 0,               "use RPS in SPS");
    if (h->fdec->rps.idx_in_gop >= 0) {
        len += u_v(p_bs, 5, h->fdec->rps.idx_in_gop,                "predict for RPS");
    } else {
        len += u_v(p_bs, 1, h->fdec->rps.referd_by_others,          "refered by others");

        len += u_v(p_bs, 3, h->fdec->rps.num_of_ref,                "num of reference picture");
        for (i = 0; i < h->fdec->rps.num_of_ref; i++) {
            len += u_v(p_bs, 6, h->fdec->rps.ref_pic[i],            "delta COI of ref pic");
        }

        len += u_v(p_bs, 3, h->fdec->rps.num_to_rm,                 "num of removed picture");
        for (i = 0; i < h->fdec->rps.num_to_rm; i++) {
            len += u_v(p_bs, 6, h->fdec->rps.rm_pic[i],             "delta COI of removed pic");
        }
        len += u_v(p_bs, 1, 1, "marker bit");
    }

    if (h->param->low_delay) {
        len += ue_v(p_bs, 0,                                        "bbv check times");
    }

    len += u_v(p_bs, 1, h->b_progressive,                           "progressive_frame");
    if (!h->b_progressive) {
        len += u_v(p_bs, 1, 1,                                      "picture_structure");
    }

    len += u_v(p_bs, 1, h->param->top_field_first,                   "top_field_first");
    len += u_v(p_bs, 1, h->param->repeat_first_field,                "repeat_first_field");
    if (h->param->InterlaceCodingOption == FIELD_CODING) {
        len += u_v(p_bs, 1, h->b_top_field,                         "is top field");
        len += u_v(p_bs, 1, 1,                                      "reserved bit for interlace coding");
    }

    len += u_v(p_bs, 1, h->param->fixed_picture_qp,                  "fixed_picture_qp");
    len += u_v(p_bs, 7, h->i_qp,                                    "picture_qp");

    if (picture_coding_type != 2) {
        len += u_v(p_bs, 1, 0,                                      "reserved_bit");
    }

    len += u_v(p_bs, 1, h->fenc->b_random_access_decodable,         "random_access_decodable_flag");

    len += u_v(p_bs, 1, h->param->loop_filter_disable,               "loop_filter_disable");

    if (!h->param->loop_filter_disable) {
        len += u_v(p_bs, 1, h->param->loop_filter_parameter_flag,    "loop_filter_parameter_flag");

        if (h->param->loop_filter_parameter_flag) {
            len += se_v(p_bs, h->param->alpha_c_offset,              "alpha offset");
            len += se_v(p_bs, h->param->beta_offset,                 "beta offset");
        }
    }

#if ENABLE_WQUANT
    len += u_v(p_bs, 1, h->param->chroma_quant_param_disable, "chroma_quant_param_disable");
    if (!h->param->chroma_quant_param_disable) {
        len += se_v(p_bs, h->param->chroma_quant_param_delta_u, "chroma_quant_param_delta_cb");
        len += se_v(p_bs, h->param->chroma_quant_param_delta_v, "chroma_quant_param_delta_cr");
    } else {
        assert(h->param->chroma_quant_param_delta_u == 0);
        assert(h->param->chroma_quant_param_delta_v == 0);
    }
#else
    len += u_v(p_bs, 1, 1, "chroma_quant_param_disable");
#endif // ENABLE_WQUANT

    if (!is_valid_qp(h, h->i_qp)) {
        xavs2_log(h, XAVS2_LOG_ERROR, "Invalid PB Picture QP: %d\n", h->i_qp);
    }

    // adaptive frequency weighting quantization
#if ENABLE_WQUANT
    if (h->param->enable_wquant) {
        len += u_v(p_bs, 1, h->param->PicWQEnable,                   "pic_weight_quant_enable");
        if (h->param->PicWQEnable) {
            len += u_v(p_bs, 2, h->param->PicWQDataIndex,            "pic_weight_quant_data_index");

            if (h->param->PicWQDataIndex == 1) {
                len += u_v(p_bs, 1, 0,                              "reserved_bits");

                len += u_v(p_bs, 2, h->param->WQParam,               "weighting_quant_param_index");
                len += u_v(p_bs, 2, h->param->WQModel,               "weighting_quant_model");

                if ((h->param->WQParam == 1) || ((h->param->MBAdaptQuant) && (h->param->WQParam == 3))) {
                    for (i = 0; i < 6; i++) {
                        len += se_v(p_bs, (int)(h->wq_data.wq_param[UNDETAILED][i] - tab_wq_param_default[UNDETAILED][i]), "quant_param_delta_u");
                    }
                }
                if ((h->param->WQParam == 2) || ((h->param->MBAdaptQuant) && (h->param->WQParam == 3))) {
                    for (i = 0; i < 6; i++) {
                        len += se_v(p_bs, (int)(h->wq_data.wq_param[DETAILED][i] - tab_wq_param_default[DETAILED][i]), "quant_param_delta_d");
                    }
                }
            } else if (h->param->PicWQDataIndex == 2) {
                int x, y, sizeId, iWqMSize;
                for (sizeId = 0; sizeId < 2; sizeId++) {
                    i = 0;
                    iWqMSize = XAVS2_MIN(1 << (sizeId + 2), 8);
                    for (y = 0; y < iWqMSize; y++) {
                        for (x = 0; x < iWqMSize; x++) {
                            len += ue_v(p_bs, h->wq_data.pic_user_wq_matrix[sizeId][i++], "weight_quant_coeff");
                        }
                    }
                }
            }
        }
    }
#endif

    return len;
}

static void writeAlfCoeff(ALFParam *Alfp, bs_t *p_bs, int componentID)
{
    int groupIdx[NO_VAR_BINS];
    int pos, i;
    int f = 0;

    switch (componentID) {
    case IMG_U:
    case IMG_V:
        for (pos = 0; pos < ALF_MAX_NUM_COEF; pos++) {
            se_v(p_bs, Alfp->coeffmulti[0][pos],                    "Chroma ALF coefficients");
        }
        break;
    case IMG_Y:
        ue_v(p_bs, Alfp->filters_per_group - 1,                     "ALF filter number");
        groupIdx[0] = 0;
        f++;
        if (Alfp->filters_per_group > 1) {
            for (i = 1; i < NO_VAR_BINS; i++) {
                if (Alfp->filterPattern[i] == 1) {
                    groupIdx[f] = i;
                    f++;
                }
            }
        }
        for (f = 0; f < Alfp->filters_per_group; f++) {
            if (f > 0 && Alfp->filters_per_group != 16) {
                ue_v(p_bs, (uint32_t)(groupIdx[f] - groupIdx[f - 1]), "Region distance");
            }
            for (pos = 0; pos < ALF_MAX_NUM_COEF; pos++) {
                se_v(p_bs, Alfp->coeffmulti[f][pos],                "Luma ALF coefficients");
            }
        }
        break;
    default:
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Not a legal component ID.\n");
        assert(0);
        exit(-1);
    }
}

/* ---------------------------------------------------------------------------
 */
void xavs2_picture_header_alf_write(xavs2_t *h, ALFParam *alfPictureParam, bs_t *p_bs)
{
    if (h->param->enable_alf) {
        u_v(p_bs, 1, h->pic_alf_on[0], "alf_pic_flag_Y");
        u_v(p_bs, 1, h->pic_alf_on[1], "alf_pic_flag_Cb");
        u_v(p_bs, 1, h->pic_alf_on[2], "alf_pic_flag_Cr");

        if (h->pic_alf_on[0]) {
            writeAlfCoeff(alfPictureParam + 0, p_bs, 0);
        }

        if (h->pic_alf_on[1]) {
            writeAlfCoeff(alfPictureParam + 1, p_bs, 1);
        }

        if (h->pic_alf_on[2]) {
            writeAlfCoeff(alfPictureParam + 2, p_bs, 2);
        }
    }
}


/* ---------------------------------------------------------------------------
 * slice header write, only full-row slices supported
 */
int xavs2_slice_header_write(xavs2_t *h, slice_t *p_slice)
{
    int len;
    bs_t *p_bs = &p_slice->bs;

    len  = u_0(p_bs, 24, 1,                                         "start code prefix");
    len += u_v(p_bs, 8, p_slice->i_first_lcu_y,                     "slice vertical position");

    if (h->i_height > (144 * (1 << h->i_lcu_level))) {
        int slice_vertical_position_extension = 0;      /* TODO: ? */
        len += u_v(p_bs, 3, slice_vertical_position_extension,      "slice vertical position extension");
    }

    len += u_v(p_bs, 8, 0,                                          "slice horizontal position");
    if (h->i_width > (255 * (1 << h->i_lcu_level))) {
        int slice_horizontal_position_extension = 0;      /* TODO: ? */
        len += u_v(p_bs, 2, slice_horizontal_position_extension,    "slice horizontal position extension");
    }

    if (!h->param->fixed_picture_qp) {
        len += u_v(p_bs, 1, h->param->i_rc_method != XAVS2_RC_CBR_SCU, "fixed_slice_qp");
        len += u_v(p_bs, 7, p_slice->i_qp,                          "slice_qp");
    }

    if (h->param->enable_sao) {
        len += u_v(p_bs, 1, h->slice_sao_on[0],                     "sao_slice_flag_Y");
        len += u_v(p_bs, 1, h->slice_sao_on[1],                     "sao_slice_flag_Cb");
        len += u_v(p_bs, 1, h->slice_sao_on[2],                     "sao_slice_flag_Cr");
    }

    if (!is_valid_qp(h, h->i_qp)) {
        xavs2_log(h, XAVS2_LOG_ERROR, "Invalid Slice QP: %d\n", h->i_qp);
    }

    return len;
}
