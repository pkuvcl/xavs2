/*
 * wquant.c
 *
 * Description of this file:
 *    Weighted Quant functions definition of the xavs2 library
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
#include "wquant.h"


#if ENABLE_WQUANT

/**
 * ===========================================================================
 * global/local tables
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
const short tab_wq_param_default[2][6] = {
    { 67, 71, 71, 80, 80, 106},
    { 64, 49, 53, 58, 58, 64 }
};

/* ---------------------------------------------------------------------------
 */
static const int tab_WqMDefault4x4[16] = {
    64, 64, 64, 68,
    64, 64, 68, 72,
    64, 68, 76, 80,
    72, 76, 84, 96
};

/* ---------------------------------------------------------------------------
 */
static const int tab_WqMDefault8x8[64] = {
    64,  64,  64,  64,  68,  68,  72,  76,
    64,  64,  64,  68,  72,  76,  84,  92,
    64,  64,  68,  72,  76,  80,  88,  100,
    64,  68,  72,  80,  84,  92,  100, 28,
    68,  72,  80,  84,  92,  104, 112, 128,
    76,  80,  84,  92,  104, 116, 132, 152,
    96,  100, 104, 116, 124, 140, 164, 188,
    104, 108, 116, 128, 152, 172, 192, 216
};

/* ---------------------------------------------------------------------------
 * weight quant model for
 */
static const uint8_t tab_WeightQuantModel[4][64] = {
    //   l a b c d h
    //   0 1 2 3 4 5
    {
        0, 0, 0, 4, 4, 4, 5, 5,       // Mode 0
        0, 0, 3, 3, 3, 3, 5, 5,
        0, 3, 2, 2, 1, 1, 5, 5,
        4, 3, 2, 2, 1, 5, 5, 5,
        4, 3, 1, 1, 5, 5, 5, 5,
        4, 3, 1, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5
    }, {
        0, 0, 0, 4, 4, 4, 5, 5,       // Mode 1
        0, 0, 4, 4, 4, 4, 5, 5,
        0, 3, 2, 2, 2, 1, 5, 5,
        3, 3, 2, 2, 1, 5, 5, 5,
        3, 3, 2, 1, 5, 5, 5, 5,
        3, 3, 1, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5
    }, {
        0, 0, 0, 4, 4, 3, 5, 5,       // Mode 2
        0, 0, 4, 4, 3, 2, 5, 5,
        0, 4, 4, 3, 2, 1, 5, 5,
        4, 4, 3, 2, 1, 5, 5, 5,
        4, 3, 2, 1, 5, 5, 5, 5,
        3, 2, 1, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5
    }, {
        0, 0, 0, 3, 2, 1, 5, 5,       // Mode 3
        0, 0, 4, 3, 2, 1, 5, 5,
        0, 4, 4, 3, 2, 1, 5, 5,
        3, 3, 3, 3, 2, 5, 5, 5,
        2, 2, 2, 2, 5, 5, 5, 5,
        1, 1, 1, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5
    }
};

/* ---------------------------------------------------------------------------
 */
static const uint8_t tab_WeightQuantModel4x4[4][16] = {
    //   l a b c d h
    //   0 1 2 3 4 5
    {
        0, 4, 3, 5,                   // Mode 0
        4, 2, 1, 5,
        3, 1, 1, 5,
        5, 5, 5, 5
    }, {
        0, 4, 4, 5,                   // Mode 1
        3, 2, 2, 5,
        3, 2, 1, 5,
        5, 5, 5, 5
    }, {
        0, 4, 3, 5,                   // Mode 2
        4, 3, 2, 5,
        3, 2, 1, 5,
        5, 5, 5, 5
    }, {
        0, 3, 1, 5,                   // Mode 3
        3, 4, 2, 5,
        1, 2, 2, 5,
        5, 5, 5, 5
    }
};

/* ---------------------------------------------------------------------------
 */
static const char *tab_WQMType[2] = {
    "WQM_4X4",
    "WQM_8X8",
};


/**
 * ===========================================================================
 * local function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE void wq_get_default_matrix(int wqm_idx, int *src)
{
    int wqm_size = 1 << (wqm_idx + 2);
    int i;

    if (wqm_idx == 0) {
        for (i = 0; i < wqm_size * wqm_size; i++) {
            src[i] = tab_WqMDefault4x4[i];
        }
    } else if (wqm_idx == 1) {
        for (i = 0; i < wqm_size * wqm_size; i++) {
            src[i] = tab_WqMDefault8x8[i];
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void wq_get_user_defined_matrix(char* wqm_file, int wqm_idx, int *src)
{
    char line[1024];
    char *ret;
    FILE *fp;
    int x, y, coef, wqm_size;

    if ((fp = fopen(wqm_file, "r")) == (FILE*)NULL) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Can't open file %s.\n %s.\n", wqm_file);
        exit(303);
    }

    fseek(fp, 0L, SEEK_SET);
    do {
        ret = fgets(line, 1024, fp);
        if ((ret == NULL) || (strstr(line, tab_WQMType[wqm_idx]) == NULL && feof(fp))) {
            xavs2_log(NULL, XAVS2_LOG_ERROR, "Error: can't read matrix %s.\n", tab_WQMType[wqm_idx]);
            exit(304);
        }
    } while (strstr(line, tab_WQMType[wqm_idx]) == NULL);

    wqm_size = 1 << (wqm_idx + 2);
    for (y = 0; y < wqm_size; y++) {
        for (x = 0; x < wqm_size; x++) {
            fscanf(fp, "%d,", &coef);
            if ((coef == 0) || coef > 255) {
                xavs2_log(NULL, XAVS2_LOG_ERROR, "QM coefficients %d is not in the range of [1, 255].\n", coef);
                exit(305);
            } else {
                src[y * wqm_size + x] = coef;
            }
        }
    }

    fclose(fp);
}

/* ---------------------------------------------------------------------------
 * calculate the level scale matrix from the current frequency weighting matrix
 * wqm_idx, 0: 4x4  1:8x8  2: 16x16  3:32x32
 */
static void wq_calculate_quant_param(xavs2_t *h, int wqm_idx)
{
    wq_data_t *wq = &h->wq_data;
    int *LevelScaleNxN[2] = { NULL, NULL };
    int block_size = 1 << (wqm_idx + 2);
    int i, j;

    LevelScaleNxN[0] = wq->levelScale[wqm_idx][0];
    LevelScaleNxN[1] = wq->levelScale[wqm_idx][1];

    if (h->WeightQuantEnable) {
        for (j = 0; j < block_size; j++) {
            for (i = 0; i < block_size; i++) {
                if ((wqm_idx == 0) || (wqm_idx == 1)) {
                    LevelScaleNxN[1][j * block_size + i] = (int)((float)(32768 << 7) / wq->cur_wq_matrix[wqm_idx][j * block_size + i]);
                } else if (wqm_idx == 2) {
                    LevelScaleNxN[1][j * block_size + i] = (int)((float)(32768 << 7) / wq->cur_wq_matrix[wqm_idx][(j >> 1) * (block_size >> 1) + (i >> 1)]);
                } else if (wqm_idx == 3) {
                    LevelScaleNxN[1][j * block_size + i] = (int)((float)(32768 << 7) / wq->cur_wq_matrix[wqm_idx][(j >> 2) * (block_size >> 2) + (i >> 2)]);
                }
            }
        }
    } else {
        for (j = 0; j < block_size; j++) {
            for (i = 0; i < block_size; i++) {
                LevelScaleNxN[0][j * block_size + i] = 32768;
                LevelScaleNxN[1][j * block_size + i] = 32768;
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * read user-defined frequency weighting parameters from configure file
 * Input:    str_param, input parameters string
 *           mode,      =0  load string to the UnDetailed parameters
 *                      =1  load string to the Detailed parameters
 */
static void wq_get_user_defined_param(xavs2_t *h, char *str_param, int mode)
{
    char str[WQMODEL_PARAM_SIZE];
    char *p;
    int param = 0;
    int num = 0;

    if (strlen(str_param) > WQMODEL_PARAM_SIZE) {
        xavs2_log(h, XAVS2_LOG_ERROR, "Cannot read the weight parameters in configuration file %s.\n", str_param);
        exit(301);
    }
    strcpy(str, str_param);

    p = str;
    for (;;) {
        if (*p == '[') {
            p++;
            param = 0;
            continue;
        } else if ((*p >= '0') && (*p <= '9')) {
            param = param * 10 + (*p - '0');
        } else if ((*p == ',') || (*p == ' ')) {
            h->wq_data.wq_param[mode][num] = (int16_t)param;
            num++;
            param = 0;
        }

        if (*p != ']') {
            p++;
        } else {
            h->wq_data.wq_param[mode][num] = (int16_t)param;
            num++;
            break;
        }
    }

    if (num != PARAM_NUM) {
        xavs2_log(h, XAVS2_LOG_ERROR,  "Not all of the weight parameters is loaded in configuration file %s.\n", str_param);
        exit(302);
    }
}


/**
 * ===========================================================================
 * interface function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * initializes the frequency weighting parameters for a new frame
 */
void xavs2_wq_init_seq_quant_param(xavs2_t *h)
{
    int seq_wqm[64];
    wq_data_t *wq = &h->wq_data;
    int wqm_index, wqm_idx;
    int block_size;
    int i;

    wq->levelScale[0][0] = wq->LevelScale4x4[0];
    wq->levelScale[0][1] = wq->LevelScale4x4[1];
    wq->levelScale[1][0] = wq->LevelScale8x8[0];
    wq->levelScale[1][1] = wq->LevelScale8x8[1];
    wq->levelScale[2][0] = wq->LevelScale16x16[0];
    wq->levelScale[2][1] = wq->LevelScale16x16[1];
    wq->levelScale[3][0] = wq->LevelScale32x32[0];
    wq->levelScale[3][1] = wq->LevelScale32x32[1];

    for (wqm_index = 0; wqm_index < 4; wqm_index++) {
        for (i = 0; i < 64; i++) {
            wq->cur_wq_matrix[wqm_index][i] = 1 << 4;
        }
    }

    for (wqm_index = 0; wqm_index < 2; wqm_index++) {
        block_size = XAVS2_MIN(1 << (wqm_index + 2), 8);
        wqm_idx = (wqm_index < 2) ? wqm_index : 1;
        if (h->param->SeqWQM == 0) {
            wq_get_default_matrix(wqm_idx, seq_wqm);
        } else if (h->param->SeqWQM == 1) {
            wq_get_user_defined_matrix(h->param->psz_seq_wq_file, wqm_idx, seq_wqm);
        }
        for (i = 0; i < (block_size * block_size); i++) {
            wq->seq_wq_matrix[wqm_index][i] = (int16_t)seq_wqm[i];
        }
    }
}

/* ---------------------------------------------------------------------------
 * initializes the frequency weighting parameters for a new picture
 */
void xavs2_wq_init_pic_quant_param(xavs2_t *h)
{
    int pic_wqm[64];
    wq_data_t *wq = &h->wq_data;
    int wqm_index, block_size, wqm_idx;
    int wq_model;
    int i, j, k;

    h->WeightQuantEnable = (h->param->enable_wquant && h->param->PicWQEnable);

    if (!h->WeightQuantEnable) {
        for (i = 0; i < 2; i++) {
            for (j = 0; j < 6; j++) {
                wq->wq_param[i][j] = 128;
            }
        }

        for (wqm_index = 0; wqm_index < 2; wqm_index++) {
            block_size = 1 << (wqm_index + 2);
            for (k = 0; k < 2; k++) {
                for (j = 0; j < block_size; j++) {
                    for (i = 0; i < block_size; i++) {
                        wq->wq_matrix[wqm_index][k][j * block_size + i] = 1 << 7;
                    }
                }
            }
        }
    } else {
        if (h->param->PicWQDataIndex == 1) {
            // patch the weighting parameters use default weighted parameters, input->WQParam==0
            for (i = 0; i < 2; i++) {
                for (j = 0; j < 6; j++) {
                    wq->wq_param[i][j] = 128;
                }
            }

            // if input->WQParam!=0, update wq_param
            if (h->param->WQParam == 0) {
                wq->cur_frame_wq_param = FRAME_WQ_DEFAULT;      // default param - detailed
                for (i = 0; i < 6; i++) {
                    wq->wq_param[DETAILED][i] = tab_wq_param_default[DETAILED][i];
                }
            } else if (h->param->WQParam == 1) {
                // load user defined weighted parameters
                wq->cur_frame_wq_param = USER_DEF_UNDETAILED;   // user defined undetailed param
                wq_get_user_defined_param(h, h->param->WeightParamUnDetailed, 0);
            } else if (h->param->WQParam == 2) {
                // load user defined weighted parameters
                wq->cur_frame_wq_param = USER_DEF_DETAILED;     // user defined detailed param
                wq_get_user_defined_param(h, h->param->WeightParamDetailed, 1);
            }

            // reconstruct the weighting matrix
            wq_model = h->param->WQModel;
            for (k = 0; k < 2; k++) {
                for (j = 0; j < 8; j++) {
                    for (i = 0; i < 8; i++) {
                        wq->wq_matrix[1][k][j * 8 + i] = (wq->wq_param[k][tab_WeightQuantModel[wq_model][j * 8 + i]]);
                    }
                }
            }
            for (k = 0; k < 2; k++) {
                for (j = 0; j < 4; j++) {
                    for (i = 0; i < 4; i++) {
                        wq->wq_matrix[0][k][j * 4 + i] = (wq->wq_param[k][tab_WeightQuantModel4x4[wq_model][j * 4 + i]]);
                    }
                }
            }
        } else if (h->param->PicWQDataIndex == 2) {
            for (wqm_index = 0; wqm_index < 2; wqm_index++) {
                block_size = XAVS2_MIN(1 << (wqm_index + 2), 8);
                wqm_idx = (wqm_index < 2) ? wqm_index : 1;
                wq_get_user_defined_matrix(h->param->psz_pic_wq_file, wqm_idx, pic_wqm);
                for (i = 0; i < (block_size * block_size); i++) {
                    wq->pic_user_wq_matrix[wqm_index][i] = (int16_t)pic_wqm[i];
                }
            }
        }
    }

    for (wqm_index = 0; wqm_index < 4; wqm_index++) {
        for (i = 0; i < 64; i++) {
            wq->cur_wq_matrix[wqm_index][i] = 1 << 7;
        }
    }
}

/* ---------------------------------------------------------------------------
 * update the frequency weighting matrix for current picture
 */
void xavs2_wq_update_pic_matrix(xavs2_t *h)
{
    wq_data_t *wq = &h->wq_data;
    int wqm_index, wqm_idx;
    int block_size;
    int i;

    if (h->WeightQuantEnable) {
        for (wqm_index = 0; wqm_index < 4; wqm_index++) {
            block_size = XAVS2_MIN(1 << (wqm_index + 2), 8);
            wqm_idx = (wqm_index < 2) ? wqm_index : 1;
            if (h->param->PicWQDataIndex == 0) {
                for (i = 0; i < (block_size * block_size); i++) {
                    wq->cur_wq_matrix[wqm_index][i] = wq->seq_wq_matrix[wqm_idx][i];
                }
            } else if (h->param->PicWQDataIndex == 1) {
                if (h->param->WQParam == 0) {
                    for (i = 0; i < (block_size * block_size); i++) {
                        wq->cur_wq_matrix[wqm_index][i] = wq->wq_matrix[wqm_idx][DETAILED][i];
                    }
                } else if (h->param->WQParam == 1) {
                    for (i = 0; i < (block_size * block_size); i++) {
                        wq->cur_wq_matrix[wqm_index][i] = wq->wq_matrix[wqm_idx][0][i];
                    }
                } else if (h->param->WQParam == 2) {
                    for (i = 0; i < (block_size * block_size); i++) {
                        wq->cur_wq_matrix[wqm_index][i] = wq->wq_matrix[wqm_idx][1][i];
                    }
                }
            } else if (h->param->PicWQDataIndex == 2) {
                for (i = 0; i < (block_size * block_size); i++) {
                    wq->cur_wq_matrix[wqm_index][i] = wq->pic_user_wq_matrix[wqm_idx][i];
                }
            }
        }
    }

    for (wqm_index = 0; wqm_index < 4; wqm_index++) {
        wq_calculate_quant_param(h, wqm_index);
    }
}

#endif // ENABLE_WQUANT
