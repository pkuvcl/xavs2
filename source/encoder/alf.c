/*
 * alf.c
 *
 * Description of this file:
 *    ALF functions definition of the xavs2 library
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
#include "aec.h"
#include "primitives.h"
#include "alf.h"
#include "header.h"
#include "cpu.h"
#include "cudata.h"


#define ROUND(a)  (((a) < 0)? (int)((a) - 0.5) : (int)((a) + 0.5))
#define REG              0.0001
#define REG_SQR          0.0000001

#define Clip_post(high,val) ((val > high)? high: val)


/**
 * ===========================================================================
 * global/local variables
 * ===========================================================================
 */

static const int tab_weightsShape1Sym[ALF_MAX_NUM_COEF + 1] = {
    2,
    2,
    2, 2, 2,
    2, 2, 2, 1,
    1
};

static const int svlc_bitrate_estimate[128] = {
    15,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    9, 9, 9, 9, 9, 9, 9, 9,
    7, 7, 7, 7,
    5, 5,
    3,
    1,
    3,
    5, 5,
    7, 7, 7, 7,
    9, 9, 9, 9, 9, 9, 9, 9,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13
};

static const int uvlc_bitrate_estimate[128] = {
    1,
    3, 3,
    5, 5, 5, 5,
    7, 7, 7, 7, 7, 7, 7, 7,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    15
};

typedef struct dh_nc {
    double dh;
    int nc;
} DhNc;

typedef struct {
    int64_t     m_autoCorr[NO_VAR_BINS][ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF];          // auto-correlation matrix
    double      m_crossCorr[NO_VAR_BINS][ALF_MAX_NUM_COEF];          // cross-correlation
    double      pixAcc[NO_VAR_BINS];
} AlfCorrData;

typedef struct {
    double      m_cross_merged[NO_VAR_BINS][ALF_MAX_NUM_COEF];
    int64_t     m_auto_merged[NO_VAR_BINS][ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF];
    double      m_cross_temp[ALF_MAX_NUM_COEF];
    double      m_pixAcc_merged[NO_VAR_BINS];
    int64_t     m_auto_temp[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF];

    int         m_coeffNoFilter[NO_VAR_BINS][ALF_MAX_NUM_COEF];
    int         m_filterCoeffSym[NO_VAR_BINS][ALF_MAX_NUM_COEF];
    int         m_varIndTab[NO_VAR_BINS];

    AlfCorrData m_pic_corr[IMG_CMPNTS];
    AlfCorrData     m_alfCorrMerged[IMG_CMPNTS];
    AlfCorrData    *m_alfCorr[IMG_CMPNTS];
    AlfCorrData    *m_alfNonSkippedCorr[IMG_CMPNTS];
    AlfCorrData    *m_alfPrevCorr;

    int         m_alfReDesignIteration;
    uint32_t    m_uiBitIncrement;

    ALFParam    m_alfPictureParam[32][IMG_CMPNTS];
    int        *m_numSlicesDataInOneLCU;
    int8_t     *tab_lcu_region;
} alf_ctx_t;

/* -------------------------------------------------------------
 */
static ALWAYS_INLINE
void init_alf_frame_param(ALFParam *p_alf)
{
    p_alf->alf_flag = 0;
    p_alf->num_coeff = ALF_MAX_NUM_COEF;
    p_alf->filters_per_group = 1;
}


/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
void reconstructCoefficients(ALFParam *alfParam, int filterCoeff[][ALF_MAX_NUM_COEF])
{
    int g, sum, i, coeffPred;

    for (g = 0; g < alfParam->filters_per_group; g++) {
        for (i = 0, sum = 0; i < alfParam->num_coeff - 1; i++) {
            sum += (2 * alfParam->coeffmulti[g][i]);
            filterCoeff[g][i] = alfParam->coeffmulti[g][i];
        }
        coeffPred = (1 << ALF_NUM_BIT_SHIFT) - sum;
        filterCoeff[g][alfParam->num_coeff - 1] = coeffPred + alfParam->coeffmulti[g][alfParam->num_coeff - 1];
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
void reconstructCoefInfo(int compIdx, ALFParam *alfParam, int filterCoeff[][ALF_MAX_NUM_COEF], int *varIndTab)
{
    int i;

    if (compIdx == IMG_Y) {
        memset(varIndTab, 0, NO_VAR_BINS * sizeof(int));
        if (alfParam->filters_per_group > 1) {
            for (i = 1; i < NO_VAR_BINS; ++i) {
                varIndTab[i] = varIndTab[i - 1];
                if (alfParam->filterPattern[i]) {
                    varIndTab[i] ++;
                }
            }
        }
    }

    reconstructCoefficients(alfParam, filterCoeff);
}


/* ---------------------------------------------------------------------------
 */
static INLINE
void checkFilterCoeffValue(int *filter, int filterLength)
{
    int maxValueNonCenter = 1 * (1 << ALF_NUM_BIT_SHIFT) - 1;
    int minValueNonCenter = 0 - 1 * (1 << ALF_NUM_BIT_SHIFT);
    int maxValueCenter = 2 * (1 << ALF_NUM_BIT_SHIFT) - 1;
    int minValueCenter = 0;
    int i;

    for (i = 0; i < filterLength - 1; i++) {
        filter[i] = XAVS2_CLIP3(minValueNonCenter, maxValueNonCenter, filter[i]);
    }

    filter[filterLength - 1] = XAVS2_CLIP3(minValueCenter, maxValueCenter, filter[filterLength - 1]);
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
void copyALFparam(ALFParam *dst, ALFParam *src, int componentID)
{
    int j;

    dst->alf_flag          = src->alf_flag;
    dst->filters_per_group = src->filters_per_group;
    dst->num_coeff         = src->num_coeff;

    switch (componentID) {
    case IMG_Y:
        for (j = 0; j < NO_VAR_BINS; j++) {
            memcpy(dst->coeffmulti[j], src->coeffmulti[j], ALF_MAX_NUM_COEF * sizeof(int));
        }
        memcpy(dst->filterPattern, src->filterPattern, NO_VAR_BINS * sizeof(int));
        break;
    case IMG_U:
    case IMG_V:
        memcpy(dst->coeffmulti[0], src->coeffmulti[0], ALF_MAX_NUM_COEF * sizeof(int));
        break;
    default:
        printf("Not a legal component ID\n");
        assert(0);
        exit(-1);
    }
}

/* ---------------------------------------------------------------------------
 * calculate the correlation matrix for Luma
 */
static
void calcCorrOneCompRegionLuma(xavs2_t *h, alf_ctx_t *Enc_ALF, pel_t *org, int i_org, pel_t *rec, int i_rec,
                               int yPos, int xPos, int height, int width,
                               int64_t m_autoCorr[][ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF],
                               double m_crossCorr[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF],
                               double *pixAcc,
                               int isLeftAvail, int isRightAvail, int isAboveAvail, int isBelowAvail)
{
    int xPosEnd = xPos + width;
    int N = ALF_MAX_NUM_COEF; //m_sqrFiltLengthTab[0];

    int startPosLuma = isAboveAvail ? (yPos - 4) : yPos;
    int endPosLuma   = isBelowAvail ? (yPos + height - 4) : (yPos + height);
    int xOffSetLeft  = isLeftAvail  ? -3 : 0;
    int xOffSetRight = isRightAvail ?  3 : 0;
    pel_t *imgPad = rec;
    pel_t *imgOrg = org;
    int yUp, yBottom;
    int xLeft, xRight;

    int ELocal[ALF_MAX_NUM_COEF];
    pel_t *imgPad1, *imgPad2, *imgPad3, *imgPad4, *imgPad5, *imgPad6;
    int i, j, k, l, yLocal, varInd;
    int64_t(*E)[9];
    double *yy;

    imgPad += startPosLuma * i_rec;
    imgOrg += startPosLuma * i_org;

    varInd = Enc_ALF->tab_lcu_region[(yPos >> h->i_lcu_level) * h->i_width_in_lcu + (xPos >> h->i_lcu_level)];
    int step = 1;
    if (IS_ALG_ENABLE(OPT_FAST_ALF)) {
        step = 2;
    }
    for (i = startPosLuma; i < endPosLuma; i += step) {
        yUp     = XAVS2_CLIP3(startPosLuma, endPosLuma - 1, i - 1);
        yBottom = XAVS2_CLIP3(startPosLuma, endPosLuma - 1, i + 1);
        imgPad1 = imgPad + (yBottom - i) * i_rec;
        imgPad2 = imgPad + (yUp - i) * i_rec;

        yUp     = XAVS2_CLIP3(startPosLuma, endPosLuma - 1, i - 2);
        yBottom = XAVS2_CLIP3(startPosLuma, endPosLuma - 1, i + 2);
        imgPad3 = imgPad + (yBottom - i) * i_rec;
        imgPad4 = imgPad + (yUp - i) * i_rec;

        yUp     = XAVS2_CLIP3(startPosLuma, endPosLuma - 1, i - 3);
        yBottom = XAVS2_CLIP3(startPosLuma, endPosLuma - 1, i + 3);
        imgPad5 = imgPad + (yBottom - i) * i_rec;
        imgPad6 = imgPad + (yUp - i) * i_rec;

        for (j = xPos; j < xPosEnd; j += step) {
            memset(ELocal, 0, N * sizeof(int));

            ELocal[0] = (imgPad5[j] + imgPad6[j]);
            ELocal[1] = (imgPad3[j] + imgPad4[j]);

            xLeft  = XAVS2_CLIP3(xPos + xOffSetLeft, xPosEnd - 1 + xOffSetRight, j - 1);
            xRight = XAVS2_CLIP3(xPos + xOffSetLeft, xPosEnd - 1 + xOffSetRight, j + 1);
            ELocal[2] = (imgPad1[xRight] + imgPad2[xLeft]);
            ELocal[3] = (imgPad1[j  ] + imgPad2[j  ]);
            ELocal[4] = (imgPad1[xLeft] + imgPad2[xRight]);
            ELocal[7] = (imgPad[xRight] + imgPad[xLeft]);

            xLeft  = XAVS2_CLIP3(xPos + xOffSetLeft, xPosEnd - 1 + xOffSetRight, j - 2);
            xRight = XAVS2_CLIP3(xPos + xOffSetLeft, xPosEnd - 1 + xOffSetRight, j + 2);
            ELocal[6] = (imgPad[xRight] + imgPad[xLeft]);

            xLeft  = XAVS2_CLIP3(xPos + xOffSetLeft, xPosEnd - 1 + xOffSetRight, j - 3);
            xRight = XAVS2_CLIP3(xPos + xOffSetLeft, xPosEnd - 1 + xOffSetRight, j + 3);
            ELocal[5] = (imgPad[xRight] + imgPad[xLeft]);
            ELocal[8] = (imgPad[j  ]);

            yLocal = imgOrg[j];
            pixAcc[varInd] += (yLocal * yLocal);
            E  = m_autoCorr[varInd];
            yy = m_crossCorr[varInd];

            for (k = 0; k < N; k++) {
                for (l = k; l < N; l++) {
                    E[k][l] += (ELocal[k] * ELocal[l]);
                }
                yy[k] += (double)(ELocal[k] * yLocal);
            }
        }

        imgPad += i_rec;
        imgOrg += i_org;
    }

    for (varInd = 0; varInd < NO_VAR_BINS; varInd++) {
        E = m_autoCorr[varInd];
        for (k = 1; k < N; k++) {
            for (l = 0; l < k; l++) {
                E[k][l] = E[l][k];
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * calculate the correlation matrix for Chroma
 */
static
void calcCorrOneCompRegionChma(xavs2_t *h, pel_t *org, int i_org, pel_t *rec, int i_rec, int yPos, int xPos, int height, int width,
                               int64_t m_autoCorr[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF], double *m_crossCorr,
                               int isLeftAvail, int isRightAvail, int isAboveAvail, int isBelowAvail)
{
    int xPosEnd = xPos + width;
    const int N = ALF_MAX_NUM_COEF; //m_sqrFiltLengthTab[0];

    int startPosChroma = isAboveAvail ? (yPos - 4) : yPos;
    int endPosChroma   = isBelowAvail ? (yPos + height - 4) : (yPos + height);
    int xOffSetLeft    = isLeftAvail  ? -3 : 0;
    int xOffSetRight   = isRightAvail ?  3 : 0;
    pel_t *imgPad = rec;
    pel_t *imgOrg = org;
    int yUp, yBottom;
    int xLeft, xRight;

    int ELocal[ALF_MAX_NUM_COEF];
    pel_t *imgPad1, *imgPad2, *imgPad3, *imgPad4, *imgPad5, *imgPad6;
    int i, j, k, l, yLocal;

    imgPad += startPosChroma * i_rec;
    imgOrg += startPosChroma * i_org;

    int step = 1;
    if (IS_ALG_ENABLE(OPT_FAST_ALF)) {
        step = 2;
    }
    for (i = startPosChroma; i < endPosChroma; i += step) {
        yUp     = XAVS2_CLIP3(startPosChroma, endPosChroma - 1, i - 1);
        yBottom = XAVS2_CLIP3(startPosChroma, endPosChroma - 1, i + 1);
        imgPad1 = imgPad + (yBottom - i) * i_rec;
        imgPad2 = imgPad + (yUp - i) * i_rec;

        yUp     = XAVS2_CLIP3(startPosChroma, endPosChroma - 1, i - 2);
        yBottom = XAVS2_CLIP3(startPosChroma, endPosChroma - 1, i + 2);
        imgPad3 = imgPad + (yBottom - i) * i_rec;
        imgPad4 = imgPad + (yUp - i) * i_rec;

        yUp     = XAVS2_CLIP3(startPosChroma, endPosChroma - 1, i - 3);
        yBottom = XAVS2_CLIP3(startPosChroma, endPosChroma - 1, i + 3);
        imgPad5 = imgPad + (yBottom - i) * i_rec;
        imgPad6 = imgPad + (yUp - i) * i_rec;

        for (j = xPos; j < xPosEnd; j += step) {
            memset(ELocal, 0, N * sizeof(int));

            ELocal[0] = (imgPad5[j] + imgPad6[j]);
            ELocal[1] = (imgPad3[j] + imgPad4[j]);

            xLeft  = XAVS2_CLIP3(xPos + xOffSetLeft, xPosEnd - 1 + xOffSetRight, j - 1);
            xRight = XAVS2_CLIP3(xPos + xOffSetLeft, xPosEnd - 1 + xOffSetRight, j + 1);
            ELocal[2] = (imgPad1[xRight] + imgPad2[xLeft]);
            ELocal[3] = (imgPad1[j  ] + imgPad2[j  ]);
            ELocal[4] = (imgPad1[xLeft] + imgPad2[xRight]);
            ELocal[7] = (imgPad[xRight] + imgPad[xLeft]);

            xLeft  = XAVS2_CLIP3(xPos + xOffSetLeft, xPosEnd - 1 + xOffSetRight, j - 2);
            xRight = XAVS2_CLIP3(xPos + xOffSetLeft, xPosEnd - 1 + xOffSetRight, j + 2);
            ELocal[6] = (imgPad[xRight] + imgPad[xLeft]);

            xLeft  = XAVS2_CLIP3(xPos + xOffSetLeft, xPosEnd - 1 + xOffSetRight, j - 3);
            xRight = XAVS2_CLIP3(xPos + xOffSetLeft, xPosEnd - 1 + xOffSetRight, j + 3);
            ELocal[5] = (imgPad[xRight] + imgPad[xLeft]);
            ELocal[8] = (imgPad[j  ]);

            yLocal = (int)imgOrg[j];

            for (k = 0; k < N; k++) {
                m_autoCorr[k][k] += ELocal[k] * ELocal[k];
                for (l = k + 1; l < N; l++) {
                    m_autoCorr[k][l] += ELocal[k] * ELocal[l];
                }

                m_crossCorr[k] += yLocal * ELocal[k];
            }
        }

        imgPad += i_rec;
        imgOrg += i_org;
    }

    for (j = 0; j < N - 1; j++) {
        for (i = j + 1; i < N; i++) {
            m_autoCorr[i][j] = m_autoCorr[j][i];
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static
void reset_alfCorr(AlfCorrData *alfCorr, int componentID)
{
    int numCoef = ALF_MAX_NUM_COEF;
    int maxNumGroups = NO_VAR_BINS;
    int g, j, i;
    int numGroups = (componentID == IMG_Y) ? (maxNumGroups) : (1);
    for (g = 0; g < numGroups; g++) {
        alfCorr->pixAcc[g] = 0;

        for (j = 0; j < numCoef; j++) {
            alfCorr->m_crossCorr[g][j] = 0;
            for (i = 0; i < numCoef; i++) {
                alfCorr->m_autoCorr[g][j][i] = 0;
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static
void deriveBoundaryAvail(xavs2_t *h, int pic_x, int pic_y,
                         int *isLeftAvail, int *isRightAvail, int *isAboveAvail, int *isBelowAvail)
{
    int size_lcu = 1 << h->i_lcu_level;
    int mb_x, mb_y;
    //int pic_mb_width = h->i_width_in_mincu;
    //cu_info_t *cuCurr, *cuLeft, *cuRight, *cuAbove, *cuBelow; 

    mb_x      = pic_x >> MIN_CU_SIZE_IN_BIT;
    mb_y      = pic_y >> MIN_CU_SIZE_IN_BIT;
    //mb_nr     = mb_y * pic_mb_width + mb_x;

    *isLeftAvail  = pic_x > 0;
    *isRightAvail = pic_x + size_lcu < h->i_width;
    *isAboveAvail = pic_y > 0;
    *isBelowAvail = pic_y + size_lcu < h->i_height;
    /*
    cuCurr  = &(h->cu_info[mb_nr]);
    cuLeft  = *isLeftAvail  ? &(h->cu_info[mb_nr - 1]) : NULL;
    cuRight = *isRightAvail ? &(h->cu_info[mb_nr + 1]) : NULL;
    cuAbove = *isAboveAvail ? &(h->cu_info[mb_nr - pic_mb_width]) : NULL;
    cuBelow = *isBelowAvail ? &(h->cu_info[mb_nr + pic_mb_width]) : NULL;
    */
    if (!h->param->b_cross_slice_loop_filter) {
        int curSliceNr = cu_get_slice_index(h, mb_x, mb_y);
        if (*isLeftAvail) {
            *isLeftAvail = cu_get_slice_index(h, mb_x - 1, mb_y) == curSliceNr;
        }
        if (*isRightAvail) {
            *isRightAvail = cu_get_slice_index(h, mb_x + (size_lcu >> MIN_CU_SIZE_IN_BIT), mb_y) == curSliceNr;
        }
        if (*isAboveAvail) {
            *isAboveAvail = cu_get_slice_index(h, mb_x, mb_y - 1) == curSliceNr;
        }
    }
}


/* ---------------------------------------------------------------------------
 * Function: Calculate the correlation matrix for each LCU
 * Input:
 *              h  : handler of encoder
 * (lcu_x, lcu_y)  : The LCU position
 *         p_org   : The original image
 *         p_rec   : The reconstruction image before ALF
 * Output:
 * Return:
 * ---------------------------------------------------------------------------
 */
void alf_get_statistics_lcu(xavs2_t *h, int lcu_x, int lcu_y,
                            xavs2_frame_t *p_org, xavs2_frame_t *p_rec)
{
    alf_ctx_t *Enc_ALF = (alf_ctx_t *)h->enc_alf;
    int ctu = lcu_y * h->i_width_in_lcu + lcu_x;
    int ctuYPos = lcu_y << h->i_lcu_level;
    int ctuXPos = lcu_x << h->i_lcu_level;
    int size_lcu = 1 << h->i_lcu_level;
    int ctuHeight = XAVS2_MIN(size_lcu, h->i_height - ctuYPos);
    int ctuWidth  = XAVS2_MIN(size_lcu, h->i_width  - ctuXPos);

    int formatShift;
    int compIdx = IMG_U;
    AlfCorrData *alfCorr = &Enc_ALF->m_alfCorr[compIdx][ctu];
    int isLeftAvail, isRightAvail, isAboveAvail, isBelowAvail;
    deriveBoundaryAvail(h, ctuXPos, ctuYPos,
                        &isLeftAvail, &isRightAvail, &isAboveAvail, &isBelowAvail);

    reset_alfCorr(alfCorr, compIdx);
    formatShift = 1;
    calcCorrOneCompRegionChma(h, p_org->planes[compIdx], p_org->i_stride[compIdx],
                              p_rec->planes[compIdx], p_rec->i_stride[compIdx],
                              ctuYPos >> formatShift, ctuXPos >> formatShift,
                              ctuHeight >> formatShift, ctuWidth >> formatShift,
                              alfCorr->m_autoCorr[0], alfCorr->m_crossCorr[0],
                              isLeftAvail, isRightAvail, isAboveAvail, isBelowAvail);

    compIdx = IMG_V;
    alfCorr = &Enc_ALF->m_alfCorr[compIdx][ctu];
    reset_alfCorr(alfCorr, compIdx);
    //V分量的ypos, xpos, height, width四个值与U分量一样，不需要修改
    calcCorrOneCompRegionChma(h, p_org->planes[compIdx], p_org->i_stride[compIdx],
                              p_rec->planes[compIdx], p_rec->i_stride[compIdx],
                              ctuYPos >> formatShift, ctuXPos >> formatShift,
                              ctuHeight >> formatShift, ctuWidth >> formatShift,
                              alfCorr->m_autoCorr[0], alfCorr->m_crossCorr[0],
                              isLeftAvail, isRightAvail, isAboveAvail, isBelowAvail);

    compIdx = IMG_Y;
    alfCorr = &Enc_ALF->m_alfCorr[compIdx][ctu];
    reset_alfCorr(alfCorr, compIdx);
    formatShift = 0;
    calcCorrOneCompRegionLuma(h, Enc_ALF, p_org->planes[compIdx], p_org->i_stride[compIdx],
                              p_rec->planes[compIdx], p_rec->i_stride[compIdx],
                              ctuYPos >> formatShift, ctuXPos >> formatShift,
                              ctuHeight >> formatShift, ctuWidth >> formatShift,
                              alfCorr->m_autoCorr, alfCorr->m_crossCorr, alfCorr->pixAcc,
                              isLeftAvail, isRightAvail, isAboveAvail, isBelowAvail);
}


/**
 * ---------------------------------------------------------------------------
 * Function: correlation matrix merge
 * Input:
 *                src: input correlation matrix
 *         mergeTable: merge table
 * Output:
 *                dst: output correlation matrix
 * Return:
 * ---------------------------------------------------------------------------
 */
static
void mergeFrom(AlfCorrData *dst, AlfCorrData *src, int *mergeTable, int doPixAccMerge, int componentID)
{
    int numCoef = ALF_MAX_NUM_COEF;
    int64_t (*srcE)[ALF_MAX_NUM_COEF], (*dstE)[ALF_MAX_NUM_COEF];
    double *srcy, *dsty;
    int maxFilterSetSize, j, i, varInd, filtIdx;

    //assert(dst->componentID == src->componentID);
    reset_alfCorr(dst, componentID);

    switch (componentID) {
    case IMG_U:
    case IMG_V:
        srcE = src->m_autoCorr[0];
        dstE = dst->m_autoCorr[0];
        srcy = src->m_crossCorr[0];
        dsty = dst->m_crossCorr[0];
        for (j = 0; j < numCoef; j++) {
            for (i = 0; i < numCoef; i++) {
                dstE[j][i] += srcE[j][i];
            }

            dsty[j] += srcy[j];
        }
        if (doPixAccMerge) {
            dst->pixAcc[0] = src->pixAcc[0];
        }
        break;
    case IMG_Y:
        maxFilterSetSize = (int)NO_VAR_BINS;
        for (varInd = 0; varInd < maxFilterSetSize; varInd++) {
            filtIdx = (mergeTable == NULL) ? (0) : (mergeTable[varInd]);
            srcE = src->m_autoCorr[varInd];
            dstE = dst->m_autoCorr[filtIdx];
            srcy = src->m_crossCorr[varInd];
            dsty = dst->m_crossCorr[filtIdx];
            for (j = 0; j < numCoef; j++) {
                for (i = 0; i < numCoef; i++) {
                    dstE[j][i] += srcE[j][i];
                }
                dsty[j] += srcy[j];
            }
            if (doPixAccMerge) {
                dst->pixAcc[filtIdx] += src->pixAcc[varInd];
            }
        }
        break;
    default:
        printf("not a legal component ID\n");
        assert(0);
        exit(-1);
    }
}

/* ---------------------------------------------------------------------------
 */
static uint32_t ALFParamBitrateEstimate(ALFParam *alfParam)
{
    uint32_t  bitrate = 0; //alf enabled flag
    int g, i;
    for (g = 0; g < alfParam->filters_per_group; g++) {
        for (i = 0; i < (int)ALF_MAX_NUM_COEF; i++) {
            bitrate += svlc_bitrate_estimate[64 + alfParam->coeffmulti[g][i]];
        }
    }

    return bitrate;
}

/* ---------------------------------------------------------------------------
 */
static uint32_t estimateALFBitrateInPicHeader(ALFParam *alfPicParam)
{
    //CXCTBD please help to check if the implementation is consistent with syntax coding
    uint32_t bitrate = 3; // pic_alf_enabled_flag[0,1,2]

    if (alfPicParam[0].alf_flag) {
        int noFilters = alfPicParam[0].filters_per_group - 1;
        bitrate += uvlc_bitrate_estimate[noFilters] + (4 * noFilters);
        bitrate += ALFParamBitrateEstimate(&alfPicParam[0]);
    }
    if (alfPicParam[1].alf_flag) {
        bitrate += ALFParamBitrateEstimate(&alfPicParam[1]);
    }
    if (alfPicParam[2].alf_flag) {
        bitrate += ALFParamBitrateEstimate(&alfPicParam[2]);
    }
    return bitrate;
}

/* ---------------------------------------------------------------------------
 */
static
long xFastFiltDistEstimation(alf_ctx_t *Enc_ALF,
                             int64_t ppdE[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF],
                             double *pdy, int *piCoeff, int iFiltLength)
{
    //static memory
    double pdcoeff[ALF_MAX_NUM_COEF];
    //variable
    int    i, j;
    long  iDist;
    double dDist = 0;
    uint32_t uiShift;

    for (i = 0; i < iFiltLength; i++) {
        pdcoeff[i] = (double)piCoeff[i] / (double)(1 << ((int)ALF_NUM_BIT_SHIFT));
    }

    for (i = 0, dDist = 0; i < iFiltLength; i++) {
        double dsum = ((double)ppdE[i][i]) * pdcoeff[i];
        for (j = i + 1; j < iFiltLength; j++) {
            dsum += (double)(2 * ppdE[i][j]) * pdcoeff[j];
        }

        dDist += ((dsum - 2.0 * pdy[i]) * pdcoeff[i]);
    }

    uiShift = Enc_ALF->m_uiBitIncrement << 1;
    if (dDist < 0) {
        iDist = -(((long)(-dDist + 0.5)) >> uiShift);
    } else { //dDist >=0
        iDist = ((long)(dDist + 0.5)) >> uiShift;
    }

    return iDist;
}

/* ---------------------------------------------------------------------------
 */
static
long estimateFilterDistortion(alf_ctx_t *Enc_ALF, int compIdx, AlfCorrData *alfCorr,
                              int coeffSet[][ALF_MAX_NUM_COEF], int filterSetSize,
                              int *mergeTable, int doPixAccMerge)
{
    AlfCorrData *alfMerged = &Enc_ALF->m_alfCorrMerged[compIdx];
    int       f;
    long      iDist = 0;

    mergeFrom(alfMerged, alfCorr, mergeTable, doPixAccMerge, compIdx);

    if (coeffSet == NULL) {
        coeffSet = Enc_ALF->m_coeffNoFilter;
    }
    for (f = 0; f < filterSetSize; f++) {
        iDist += xFastFiltDistEstimation(Enc_ALF, alfMerged->m_autoCorr[f], alfMerged->m_crossCorr[f], coeffSet[f], ALF_MAX_NUM_COEF);
    }

    return iDist;
}

/* ---------------------------------------------------------------------------
 */
static
dist_t calcAlfLCUDist(xavs2_t *h, alf_ctx_t *Enc_ALF, int compIdx,
                      int ypos, int xpos, int height, int width, int isAboveAvail,
                      pel_t *picSrc, int i_src, pel_t *picCmp, int i_cmp)
{
    dist_t dist = 0;
    pel_t *pelCmp = picCmp;
    pel_t *pelSrc = picSrc;

    int notSkipLinesRightVB = TRUE;
    int notSkipLinesBelowVB = TRUE;
    //int NumCUsInFrame, numLCUInPicWidth, numLCUInPicHeight;

    //numLCUInPicHeight  = h->i_height_in_lcu;
    //numLCUInPicWidth   = h->i_width_in_lcu;
    //NumCUsInFrame      = numLCUInPicHeight * numLCUInPicWidth;

    switch (compIdx) {
    case IMG_U:
    case IMG_V:
        if (!notSkipLinesBelowVB) {
            height = height - (int)(DF_CHANGED_SIZE >> 1) - (int)(ALF_FOOTPRINT_SIZE >> 1);
        }

        if (!notSkipLinesRightVB) {
            width = width - (int)(DF_CHANGED_SIZE >> 1) - (int)(ALF_FOOTPRINT_SIZE >> 1);
        }

        if (isAboveAvail) {
            pelSrc += ((ypos - 4) * i_src) + xpos;
            pelCmp += ((ypos - 4) * i_cmp) + xpos;
        } else {
            pelSrc += (ypos * i_src) + xpos;
            pelCmp += (ypos * i_cmp) + xpos;
        }
        break;
    default:
        // case IMG_Y:
        if (!notSkipLinesBelowVB) {
            height = height - (int)(DF_CHANGED_SIZE)-(int)(ALF_FOOTPRINT_SIZE >> 1);
        }

        if (!notSkipLinesRightVB) {
            width = width - (int)(DF_CHANGED_SIZE)-(int)(ALF_FOOTPRINT_SIZE >> 1);
        }

        pelCmp = picCmp + (ypos * i_cmp) + xpos;
        pelSrc = picSrc + (ypos * i_src) + xpos;
        break;
    }
    if (PART_INDEX(width, height) == LUMA_INVALID) {
        uint32_t uiShift = Enc_ALF->m_uiBitIncrement << 1;
        dist += g_funcs.pixf.ssd_block(pelSrc, i_src, pelCmp, i_cmp, width, height) >> uiShift;
    } else {
        dist += g_funcs.pixf.ssd[PART_INDEX(width, height)](pelSrc, i_src, pelCmp, i_cmp);
    }

    return dist;
}

/* ---------------------------------------------------------------------------
 * ALF filter on CTB
 */
static
void filterOneCTB(xavs2_t *h, alf_ctx_t *Enc_ALF, pel_t *p_dst, int i_dst, pel_t *p_src, int i_src,
                  int compIdx, ALFParam *alfParam, int ypos, int height, int xpos, int width,
                  int isAboveAvail, int isBelowAvail)
{
    int *coef;

    //reconstruct coefficients to m_filterCoeffSym and m_varIndTab
    reconstructCoefInfo(compIdx, alfParam, Enc_ALF->m_filterCoeffSym, Enc_ALF->m_varIndTab); //reconstruct ALF coefficients & related parameters

    //derive CTB start positions, width, and height. If the boundary is not available, skip boundary samples.

    if (compIdx == IMG_Y) {
        int var = Enc_ALF->tab_lcu_region[(ypos >> h->i_lcu_level) * h->i_width_in_lcu + (xpos >> h->i_lcu_level)];
        coef = Enc_ALF->m_filterCoeffSym[Enc_ALF->m_varIndTab[var]];
    } else {
        coef = Enc_ALF->m_filterCoeffSym[0];
    }


    g_funcs.alf_flt[0](p_dst, i_dst, p_src, i_src,
                       xpos, ypos, width, height, coef,
                       isAboveAvail, isBelowAvail);
    g_funcs.alf_flt[1](p_dst, i_dst, p_src, i_src,
                       xpos, ypos, width, height, coef,
                       isAboveAvail, isBelowAvail);
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE
void copyOneAlfBlk(pel_t *p_dst, int i_dst, pel_t *p_src, int i_src, int ypos, int xpos,
                   int height, int width, int isAboveAvail, int isBelowAvail)
{
    int startPos  = isAboveAvail ? (ypos          - 4) : ypos;
    int endPos    = isBelowAvail ? (ypos + height - 4) : ypos + height;
    p_dst += (startPos * i_dst) + xpos;
    p_src += (startPos * i_src) + xpos;

    g_funcs.plane_copy(p_dst, i_dst, p_src, i_src, width, endPos - startPos);
}

/* ---------------------------------------------------------------------------
 * ALF On/off decision for LCU and do RDO Estimation
 */
static
double executePicLCUOnOffDecisionRDOEstimate(xavs2_t *h, alf_ctx_t *Enc_ALF, aec_t *p_aec, ALFParam *alfPictureParam,
        double lambda, AlfCorrData * alfCorr)
{
    dist_t distEnc, distOff;
    double rateEnc, rateOff, costEnc, costOff, costAlfOn, costAlfOff;
    dist_t distBestPic[IMG_CMPNTS];
    double rateBestPic[IMG_CMPNTS];
    int compIdx, ctu;
    double lambda_luma, lambda_chroma;
    //int img_height, img_width;
    int NumCUsInFrame;
    double bestCost = 0;
    int rate, noFilters;

    h->copy_aec_state_rdo(p_aec, &h->cs_data.cs_alf_initial);
    h->copy_aec_state_rdo(&h->cs_data.cs_alf_cu_ctr, p_aec);

    //img_height = h->i_height;
    //img_width = h->i_width;
    NumCUsInFrame = h->i_height_in_lcu * h->i_width_in_lcu;

    lambda_luma = lambda; //VKTBD lambda is not correct
    lambda_chroma = LAMBDA_SCALE_CHROMA * lambda_luma;
    for (compIdx = 0; compIdx < IMG_CMPNTS; compIdx++) {
        distBestPic[compIdx] = 0;
        rateBestPic[compIdx] = 0;
    }

    for (ctu = 0; ctu < NumCUsInFrame; ctu++) {
        for (compIdx = 0; compIdx < IMG_CMPNTS; compIdx++) {
            //if slice-level enabled flag is 0, set CTB-level enabled flag 0
            if (alfPictureParam[compIdx].alf_flag == 0) {
                h->is_alf_lcu_on[ctu][compIdx] = FALSE;
                continue;
            }

            // ALF on
            reconstructCoefInfo(compIdx, &alfPictureParam[compIdx], Enc_ALF->m_filterCoeffSym, Enc_ALF->m_varIndTab);
            //distEnc is the estimated distortion reduction compared with filter-off case
            distEnc = estimateFilterDistortion(Enc_ALF, compIdx, alfCorr + (compIdx * NumCUsInFrame) + ctu, Enc_ALF->m_filterCoeffSym,
                                               alfPictureParam[compIdx].filters_per_group, Enc_ALF->m_varIndTab, FALSE)
                      - estimateFilterDistortion(Enc_ALF, compIdx, alfCorr + (compIdx * NumCUsInFrame) + ctu, NULL, 1, NULL, FALSE);

            h->copy_aec_state_rdo(p_aec, &h->cs_data.cs_alf_cu_ctr);

            rateEnc = p_aec->binary.write_alf_lcu_ctrl(p_aec, 1);

            costEnc = (double)distEnc + (compIdx == 0 ? lambda_luma : lambda_chroma) * rateEnc;

            // ALF off
            distOff = 0;
            // rateOff = 1;
            h->copy_aec_state_rdo(p_aec, &h->cs_data.cs_alf_cu_ctr);
            rateOff = p_aec->binary.write_alf_lcu_ctrl(p_aec, 0);

            costOff = (double)distOff + (compIdx == 0 ? lambda_luma : lambda_chroma) * rateOff;

            //set CTB-level on/off flag
            h->is_alf_lcu_on[ctu][compIdx] = (costEnc < costOff) ? TRUE : FALSE;

            //update CABAC status
            //cabacCoder->updateAlfCtrlFlagState(m_pcPic->getCU(ctu)->getAlfLCUEnabled(compIdx)?1:0);

            h->copy_aec_state_rdo(p_aec, &h->cs_data.cs_alf_cu_ctr);
            rateOff = p_aec->binary.write_alf_lcu_ctrl(p_aec, (h->is_alf_lcu_on[ctu][compIdx] ? 1 : 0));
            h->copy_aec_state_rdo(&h->cs_data.cs_alf_cu_ctr, p_aec);

            rateBestPic[compIdx] += (h->is_alf_lcu_on[ctu][compIdx] ? rateEnc : rateOff);
            distBestPic[compIdx] += (h->is_alf_lcu_on[ctu][compIdx] ? distEnc : distOff);
        } //CTB
    } //CTU

    for (compIdx = 0; compIdx < IMG_CMPNTS; compIdx++) {
        if (alfPictureParam[compIdx].alf_flag == 1) {
            double Lambda = (compIdx == 0 ? lambda_luma : lambda_chroma);
            rate = ALFParamBitrateEstimate(&alfPictureParam[compIdx]);
            if (compIdx == IMG_Y) {
                noFilters = alfPictureParam[0].filters_per_group - 1;
                rate += uvlc_bitrate_estimate[noFilters] + (4 * noFilters);
            }
            costAlfOn = (double)distBestPic[compIdx] + Lambda *
                        (rateBestPic[compIdx] + (double)(rate));

            costAlfOff = 0;

            if (costAlfOn >= costAlfOff) {
                alfPictureParam[compIdx].alf_flag = 0;
                for (ctu = 0; ctu < NumCUsInFrame; ctu++) {
                    h->is_alf_lcu_on[ctu][compIdx] = FALSE;
                }
            }
        }
    }

    bestCost = 0;
    for (compIdx = 0; compIdx < IMG_CMPNTS; compIdx++) {
        if (alfPictureParam[compIdx].alf_flag == 1) {
            bestCost += (double)distBestPic[compIdx] + (compIdx == 0 ? lambda_luma : lambda_chroma) * (rateBestPic[compIdx]);
        }
    }

    // return the block-level RD cost
    return bestCost;
}

/* ---------------------------------------------------------------------------
* ALF On/Off decision for LCU
*/
static
void executePicLCUOnOffDecision(xavs2_t *h, alf_ctx_t *Enc_ALF, aec_t *p_aec, ALFParam *alfPictureParam,
                                double lambda, xavs2_frame_t *p_org, xavs2_frame_t *p_rec, xavs2_frame_t *p_dst)
{
    dist_t distEnc, distOff;
    double rateEnc, rateOff, costEnc, costOff, costAlfOn, costAlfOff;
    int isLeftAvail, isRightAvail, isAboveAvail, isBelowAvail;
    dist_t distBestPic[IMG_CMPNTS];
    double rateBestPic[IMG_CMPNTS];
    int compIdx, ctu, ctuYPos, ctuXPos, ctuHeight, ctuWidth;
    int formatShift = 0;
    int i_org = 0;
    int i_rec_before = 0;
    int i_rec_after = 0;
    pel_t *p_org_pixel = NULL;
    pel_t *p_rec_before = NULL;
    pel_t *p_rec_after = NULL;
    double lambda_luma, lambda_chroma;
    int img_height, img_width;
    int size_lcu = 1 << h->i_lcu_level;
    int ctux, ctuy;
    int NumCUsInFrame, numLCUInPicWidth, numLCUInPicHeight;
    int rate, noFilters;

    h->copy_aec_state_rdo(p_aec, &h->cs_data.cs_alf_initial);
    h->copy_aec_state_rdo(&h->cs_data.cs_alf_cu_ctr, p_aec);

    img_height = h->i_height;
    img_width = h->i_width;
    numLCUInPicHeight = h->i_height_in_lcu;
    numLCUInPicWidth = h->i_width_in_lcu;
    NumCUsInFrame = numLCUInPicHeight * numLCUInPicWidth;

    lambda_luma = lambda; //VKTBD lambda is not correct
    lambda_chroma = LAMBDA_SCALE_CHROMA * lambda_luma;
    for (compIdx = 0; compIdx < IMG_CMPNTS; compIdx++) {
        distBestPic[compIdx] = 0;
        rateBestPic[compIdx] = 0;
    }

    for (ctuy = 0, ctu = 0; ctuy < numLCUInPicHeight; ctuy++) {
        //derive CTU height
        ctuYPos = ctuy * size_lcu;
        ctuHeight = XAVS2_MIN(img_height - ctuYPos, size_lcu);
        for (ctux = 0; ctux < numLCUInPicWidth; ctux++, ctu++) {
            //derive CTU width
            ctuXPos = ctux * size_lcu;
            ctuWidth = XAVS2_MIN(img_width - ctuXPos, size_lcu);

            //derive CTU boundary availabilities
            deriveBoundaryAvail(h, ctuXPos, ctuYPos,
                                &isLeftAvail, &isRightAvail, &isAboveAvail, &isBelowAvail);

            for (compIdx = 0; compIdx < IMG_CMPNTS; compIdx++) {
                //if slice-level enabled flag is 0, set CTB-level enabled flag 0
                if (alfPictureParam[compIdx].alf_flag == 0) {
                    h->is_alf_lcu_on[ctu][compIdx] = FALSE;
                    continue;
                }

                formatShift = (compIdx == IMG_Y) ? 0 : 1;
                p_org_pixel = p_org->planes[compIdx];
                i_org = p_org->i_stride[compIdx];
                p_rec_before = p_rec->planes[compIdx];
                i_rec_before = p_rec->i_stride[compIdx];
                p_rec_after = p_dst->planes[compIdx];
                i_rec_after = p_dst->i_stride[compIdx];

                // ALF on
                filterOneCTB(h, Enc_ALF, p_rec_after, i_rec_after, p_rec_before, i_rec_before, compIdx,
                             &alfPictureParam[compIdx], ctuYPos >> formatShift, ctuHeight >> formatShift,
                             ctuXPos >> formatShift, ctuWidth >> formatShift, isAboveAvail, isBelowAvail);
                distEnc = calcAlfLCUDist(h, Enc_ALF, compIdx, ctuYPos >> formatShift, ctuXPos >> formatShift,
                                         ctuHeight >> formatShift, ctuWidth >> formatShift, isAboveAvail, p_org_pixel, i_org, p_rec_after, i_rec_after);
                distEnc -= calcAlfLCUDist(h, Enc_ALF, compIdx, ctuYPos >> formatShift, ctuXPos >> formatShift,
                                          ctuHeight >> formatShift, ctuWidth >> formatShift, isAboveAvail, p_org_pixel, i_org, p_rec_before, i_rec_before);

                h->copy_aec_state_rdo(p_aec, &h->cs_data.cs_alf_cu_ctr);

                rateEnc = p_aec->binary.write_alf_lcu_ctrl(p_aec, 1);

                costEnc = (double)distEnc + (compIdx == 0 ? lambda_luma : lambda_chroma) * rateEnc;

                // ALF off
                distOff = 0;
                //rateOff = 1;
                h->copy_aec_state_rdo(p_aec, &h->cs_data.cs_alf_cu_ctr);
                rateOff = p_aec->binary.write_alf_lcu_ctrl(p_aec, 0);

                costOff = (double)distOff + (compIdx == 0 ? lambda_luma : lambda_chroma) * rateOff;

                //set CTB-level on/off flag
                h->is_alf_lcu_on[ctu][compIdx] = (costEnc < costOff) ? TRUE : FALSE;

                if (!h->is_alf_lcu_on[ctu][compIdx]) {
                    copyOneAlfBlk(p_rec_after, i_rec_after, p_rec_before, i_rec_before,
                                  ctuYPos >> formatShift, ctuXPos >> formatShift, ctuHeight >> formatShift, ctuWidth >> formatShift,
                                  isAboveAvail, isBelowAvail);
                }

                //update CABAC status
                //cabacCoder->updateAlfCtrlFlagState(m_pcPic->getCU(ctu)->getAlfLCUEnabled(compIdx)?1:0);

                h->copy_aec_state_rdo(p_aec, &h->cs_data.cs_alf_cu_ctr);
                rateOff = p_aec->binary.write_alf_lcu_ctrl(p_aec, (h->is_alf_lcu_on[ctu][compIdx] ? 1 : 0));
                h->copy_aec_state_rdo(&h->cs_data.cs_alf_cu_ctr, p_aec);

                rateBestPic[compIdx] += (h->is_alf_lcu_on[ctu][compIdx] ? rateEnc : rateOff);
                distBestPic[compIdx] += (h->is_alf_lcu_on[ctu][compIdx] ? distEnc : distOff);

            } //CTB
        }
    } //CTU

    for (compIdx = 0; compIdx < IMG_CMPNTS; compIdx++) {
        if (alfPictureParam[compIdx].alf_flag == 1) {
            double Lambda = (compIdx == 0 ? lambda_luma : lambda_chroma);
            rate = ALFParamBitrateEstimate(&alfPictureParam[compIdx]);
            if (compIdx == IMG_Y) {
                noFilters = alfPictureParam[0].filters_per_group - 1;
                rate += uvlc_bitrate_estimate[noFilters] + (4 * noFilters);
            }
            costAlfOn = (double)distBestPic[compIdx] + Lambda *
                        (rateBestPic[compIdx] + (double)(rate));

            costAlfOff = 0;

            if (costAlfOn >= costAlfOff) {
                alfPictureParam[compIdx].alf_flag = 0;
                for (ctu = 0; ctu < NumCUsInFrame; ctu++) {
                    h->is_alf_lcu_on[ctu][compIdx] = FALSE;
                }

                g_funcs.plane_copy(p_dst->planes[compIdx], p_dst->i_stride[compIdx],
                                   p_rec->planes[compIdx], p_rec->i_stride[compIdx],
                                   p_rec->i_width[compIdx], p_rec->i_lines[compIdx]);
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void ADD_AlfCorrData(AlfCorrData *A, AlfCorrData *B, AlfCorrData *C, int componentID)
{
    int numCoef = ALF_MAX_NUM_COEF;
    int maxNumGroups = NO_VAR_BINS;
    int numGroups;
    int g, j, i;

    numGroups = (componentID == IMG_Y) ? (maxNumGroups) : (1);
    for (g = 0; g < numGroups; g++) {
        C->pixAcc[g] = A->pixAcc[g] + B->pixAcc[g];

        for (j = 0; j < numCoef; j++) {
            C->m_crossCorr[g][j] = A->m_crossCorr[g][j] + B->m_crossCorr[g][j];
            for (i = 0; i < numCoef; i++) {
                C->m_autoCorr[g][j][i] = A->m_autoCorr[g][j][i] + B->m_autoCorr[g][j][i];
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static
void accumulateLCUCorrelations(xavs2_t *h, AlfCorrData **alfCorrAcc, AlfCorrData ** alfCorSrcLCU, int useAllLCUs)
{
    int compIdx, numStatLCU, addr;
    AlfCorrData *alfCorrAccComp;
    int NumCUsInFrame = h->i_width_in_lcu * h->i_height_in_lcu;

    for (compIdx = 0; compIdx < IMG_CMPNTS; compIdx++) {
        alfCorrAccComp = alfCorrAcc[compIdx];

        reset_alfCorr(alfCorrAccComp, compIdx);

        if (!useAllLCUs) {
            numStatLCU = 0;
            for (addr = 0; addr < NumCUsInFrame; addr++) {
                if (h->is_alf_lcu_on[addr][compIdx]) {
                    numStatLCU++;
                    break;
                }
            }
            useAllLCUs = (numStatLCU == 0) ? TRUE : useAllLCUs;
        }

        for (addr = 0; addr < (int)NumCUsInFrame; addr++) {
            if (useAllLCUs || h->is_alf_lcu_on[addr][compIdx]) {
                //*alfCorrAccComp += *(alfCorSrcLCU[compIdx][addr]);
                ADD_AlfCorrData(&alfCorSrcLCU[compIdx][addr], alfCorrAccComp, alfCorrAccComp, compIdx);
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void predictALFCoeff(int coeff[][ALF_MAX_NUM_COEF], int numCoef, int numFilters)
{
    int g, pred, sum, i;

    for (g = 0; g < numFilters; g++) {
        for (i = 0, sum = 0; i < numCoef - 1; i++) {
            sum += (2 * coeff[g][i]);
        }

        pred = (1 << ALF_NUM_BIT_SHIFT) - (sum);
        coeff[g][numCoef - 1] = coeff[g][numCoef - 1] - pred;
    }
}

/* ---------------------------------------------------------------------------
 */
static void xcodeFiltCoeff(int filterCoeff[][ALF_MAX_NUM_COEF], int *varIndTab, int numFilters, ALFParam *alfParam)
{
    int filterPattern[NO_VAR_BINS], i, g;
    memset(filterPattern, 0, NO_VAR_BINS * sizeof(int));

    alfParam->num_coeff = (int)ALF_MAX_NUM_COEF;
    alfParam->filters_per_group = numFilters;

    //merge table assignment
    if (alfParam->filters_per_group > 1) {
        for (i = 1; i < NO_VAR_BINS; ++i) {
            if (varIndTab[i] != varIndTab[i - 1]) {
                filterPattern[i] = 1;
                //startSecondFilter = i;
            }
        }
    }
    memcpy(alfParam->filterPattern, filterPattern, NO_VAR_BINS * sizeof(int));

    //coefficient prediction
    for (g = 0; g < alfParam->filters_per_group; g++) {
        for (i = 0; i < alfParam->num_coeff; i++) {
            alfParam->coeffmulti[g][i] = filterCoeff[g][i];
        }
    }

    predictALFCoeff(alfParam->coeffmulti, alfParam->num_coeff, alfParam->filters_per_group);
}

/* ---------------------------------------------------------------------------
 */
static void gnsTransposeBacksubstitution(double U[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF], double rhs[], double x[], int order)
{
    int i, j;       // Looping variables
    double sum;     // Holds backsubstitution from already handled rows

    // Backsubstitution starts
    x[0] = rhs[0] / U[0][0];        // first row of U
    for (i = 1; i < order; i++) {
        // for the rows 1..order-1
        for (j = 0, sum = 0.0; j < i; j++) {    // Backsubst already solved unknowns
            sum += x[j] * U[j][i];
        }
        x[i] = (rhs[i] - sum) / U[i][i];        // i'th component of solution vect
    }
}

/* ---------------------------------------------------------------------------
 */
static void gnsBacksubstitution(double R[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF], double z[ALF_MAX_NUM_COEF], int R_size, double A[ALF_MAX_NUM_COEF])
{
    int i, j;
    double sum;

    R_size--;
    A[R_size] = z[R_size] / R[R_size][R_size];
    for (i = R_size - 1; i >= 0; i--) {
        for (j = i + 1, sum = 0.0; j <= R_size; j++) {
            sum += R[i][j] * A[j];
        }

        A[i] = (z[i] - sum) / R[i][i];
    }
}

/* ---------------------------------------------------------------------------
 */
static int gnsCholeskyDec(int64_t inpMatr[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF], double outMatr[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF], int noEq)
{
    int i, j, k;        /* Looping Variables */
    double scale;       /* scaling factor for each row */
    double invDiag[ALF_MAX_NUM_COEF];  /* Vector of the inverse of diagonal entries of outMatr */

    // Cholesky decomposition starts
    for (i = 0; i < noEq; i++) {
        for (j = i; j < noEq; j++) {
            /* Compute the scaling factor */
            scale = (double)inpMatr[i][j];
            if (i > 0) {
                for (k = i - 1; k >= 0; k--) {
                    scale -= outMatr[k][j] * outMatr[k][i];
                }
            }
            /* Compute i'th row of outMatr */
            if (i == j) {
                if (scale <= REG_SQR) { // if(scale <= 0 )  /* If inpMatr is singular */
                    return 0;
                } else {
                    /* Normal operation */
                    invDiag[i] = 1.0 / (outMatr[i][i] = sqrt(scale));
                }
            } else {
                outMatr[i][j] = scale * invDiag[i]; /* Upper triangular part          */
                outMatr[j][i] = 0.0;              /* Lower triangular part set to 0 */
            }
        }
    }

    return 1; /* Signal that Cholesky factorization is successfully performed */
}

/* ---------------------------------------------------------------------------
 */
static int gnsSolveByChol(int64_t LHS[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF], double *rhs, double *x, int noEq)
{
    double aux[ALF_MAX_NUM_COEF];     /* Auxiliary vector */
    double U[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF];    /* Upper triangular Cholesky factor of LHS */
    int  i, singular;          /* Looping variable */
    assert(noEq > 0);

    /* The equation to be solved is LHSx = rhs */

    /* Compute upper triangular U such that U'*U = LHS */
    if (gnsCholeskyDec(LHS, U, noEq)) { /* If Cholesky decomposition has been successful */
        singular = 1;
        /* Now, the equation is  U'*U*x = rhs, where U is upper triangular
        * Solve U'*aux = rhs for aux
        */
        gnsTransposeBacksubstitution(U, rhs, aux, noEq);

        /* The equation is now U*x = aux, solve it for x (new motion coefficients) */
        gnsBacksubstitution(U, aux, noEq, x);
    } else { /* LHS was singular */
        singular = 0;

        /* Regularize LHS
        for (i = 0; i < noEq; i++) {
            LHS[i][i] += REG;
        }*/
        /* Compute upper triangular U such that U'*U = regularized LHS */
        singular = gnsCholeskyDec(LHS, U, noEq);
        if (singular == 1) {
            /* Solve  U'*aux = rhs for aux */
            gnsTransposeBacksubstitution(U, rhs, aux, noEq);

            /* Solve U*x = aux for x */
            gnsBacksubstitution(U, aux, noEq, x);
        } else {
            x[0] = 1.0;
            for (i = 1; i < noEq; i++) {
                x[i] = 0.0;
            }
        }
    }
    return singular;
}

/* ---------------------------------------------------------------------------
 */
static double calculateErrorAbs(int64_t A[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF], double *b, double y, int size)
{
    int i;
    double error, sum;
    double c[ALF_MAX_NUM_COEF];

    gnsSolveByChol(A, b, c, size);

    sum = 0;
    for (i = 0; i < size; i++) {
        sum += c[i] * b[i];
    }
    error = y - sum;

    return error;
}

/* ---------------------------------------------------------------------------
 */
static
double mergeFiltersGreedy(alf_ctx_t *Enc_ALF, double yGlobalSeq[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF], int64_t EGlobalSeq[][ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF],
                          double *pixAccGlobalSeq, int intervalBest[NO_VAR_BINS][2], int sqrFiltLength, int noIntervals)
{
    int first, ind, ind1, ind2, i, j, bestToMerge;
    double error, error1, error2, errorMin;
    static double pixAcc_temp, error_tab[NO_VAR_BINS], error_comb_tab[NO_VAR_BINS];
    static int indexList[NO_VAR_BINS], available[NO_VAR_BINS], noRemaining;

    if (noIntervals == NO_VAR_BINS) {
        noRemaining = NO_VAR_BINS;
        for (ind = 0; ind < NO_VAR_BINS; ind++) {
            indexList[ind] = ind;
            available[ind] = 1;
            Enc_ALF->m_pixAcc_merged[ind] = pixAccGlobalSeq[ind];
            memcpy(Enc_ALF->m_cross_merged[ind], yGlobalSeq[ind], sizeof(double)*sqrFiltLength);
            for (i = 0; i < sqrFiltLength; i++) {
                memcpy(Enc_ALF->m_auto_merged[ind][i], EGlobalSeq[ind][i], sizeof(int64_t)*sqrFiltLength);
            }
        }

        // try merging different matrices
        for (ind = 0; ind < NO_VAR_BINS; ind++) {
            error_tab[ind] = calculateErrorAbs(Enc_ALF->m_auto_merged[ind], Enc_ALF->m_cross_merged[ind], Enc_ALF->m_pixAcc_merged[ind], sqrFiltLength);
        }
        for (ind = 0; ind < NO_VAR_BINS - 1; ind++) {
            ind1 = indexList[ind];
            ind2 = indexList[ind + 1];

            error1 = error_tab[ind1];
            error2 = error_tab[ind2];

            pixAcc_temp = Enc_ALF->m_pixAcc_merged[ind1] + Enc_ALF->m_pixAcc_merged[ind2];
            for (i = 0; i < sqrFiltLength; i++) {
                Enc_ALF->m_cross_temp[i] = Enc_ALF->m_cross_merged[ind1][i] + Enc_ALF->m_cross_merged[ind2][i];
                for (j = 0; j < sqrFiltLength; j++) {
                    Enc_ALF->m_auto_temp[i][j] = Enc_ALF->m_auto_merged[ind1][i][j] + Enc_ALF->m_auto_merged[ind2][i][j];
                }
            }
            error_comb_tab[ind1] = calculateErrorAbs(Enc_ALF->m_auto_temp, Enc_ALF->m_cross_temp, pixAcc_temp, sqrFiltLength) - error1 - error2;
        }
    }

    while (noRemaining > noIntervals) {
        errorMin = 0;
        first = 1;
        bestToMerge = 0;
        for (ind = 0; ind < noRemaining - 1; ind++) {
            error = error_comb_tab[indexList[ind]];
            if ((error < errorMin || first == 1)) {
                errorMin = error;
                bestToMerge = ind;
                first = 0;
            }
        }

        ind1 = indexList[bestToMerge];
        ind2 = indexList[bestToMerge + 1];
        Enc_ALF->m_pixAcc_merged[ind1] += Enc_ALF->m_pixAcc_merged[ind2];
        for (i = 0; i < sqrFiltLength; i++) {
            Enc_ALF->m_cross_merged[ind1][i] += Enc_ALF->m_cross_merged[ind2][i];
            for (j = 0; j < sqrFiltLength; j++) {
                Enc_ALF->m_auto_merged[ind1][i][j] += Enc_ALF->m_auto_merged[ind2][i][j];
            }
        }
        available[ind2] = 0;

        // update error tables
        error_tab[ind1] = error_comb_tab[ind1] + error_tab[ind1] + error_tab[ind2];
        if (indexList[bestToMerge] > 0) {
            ind1 = indexList[bestToMerge - 1];
            ind2 = indexList[bestToMerge];
            error1 = error_tab[ind1];
            error2 = error_tab[ind2];
            pixAcc_temp = Enc_ALF->m_pixAcc_merged[ind1] + Enc_ALF->m_pixAcc_merged[ind2];
            for (i = 0; i < sqrFiltLength; i++) {
                Enc_ALF->m_cross_temp[i] = Enc_ALF->m_cross_merged[ind1][i] + Enc_ALF->m_cross_merged[ind2][i];
                for (j = 0; j < sqrFiltLength; j++) {
                    Enc_ALF->m_auto_temp[i][j] = Enc_ALF->m_auto_merged[ind1][i][j] + Enc_ALF->m_auto_merged[ind2][i][j];
                }
            }
            error_comb_tab[ind1] = calculateErrorAbs(Enc_ALF->m_auto_temp, Enc_ALF->m_cross_temp, pixAcc_temp, sqrFiltLength) - error1 - error2;
        }

        if (indexList[bestToMerge + 1] < NO_VAR_BINS - 1) {
            ind1 = indexList[bestToMerge];
            ind2 = indexList[bestToMerge + 2];
            error1 = error_tab[ind1];
            error2 = error_tab[ind2];
            pixAcc_temp = Enc_ALF->m_pixAcc_merged[ind1] + Enc_ALF->m_pixAcc_merged[ind2];
            for (i = 0; i < sqrFiltLength; i++) {
                Enc_ALF->m_cross_temp[i] = Enc_ALF->m_cross_merged[ind1][i] + Enc_ALF->m_cross_merged[ind2][i];
                for (j = 0; j < sqrFiltLength; j++) {
                    Enc_ALF->m_auto_temp[i][j] = Enc_ALF->m_auto_merged[ind1][i][j] + Enc_ALF->m_auto_merged[ind2][i][j];
                }
            }
            error_comb_tab[ind1] = calculateErrorAbs(Enc_ALF->m_auto_temp, Enc_ALF->m_cross_temp, pixAcc_temp, sqrFiltLength) - error1 - error2;
        }

        ind = 0;
        for (i = 0; i < NO_VAR_BINS; i++) {
            if (available[i] == 1) {
                indexList[ind] = i;
                ind++;
            }
        }
        noRemaining--;
    }

    errorMin = 0;
    for (ind = 0; ind < noIntervals; ind++) {
        errorMin += error_tab[indexList[ind]];
    }

    for (ind = 0; ind < noIntervals - 1; ind++) {
        intervalBest[ind][0] = indexList[ind];
        intervalBest[ind][1] = indexList[ind + 1] - 1;
    }

    intervalBest[noIntervals - 1][0] = indexList[noIntervals - 1];
    intervalBest[noIntervals - 1][1] = NO_VAR_BINS - 1;

    return (errorMin);
}

/* ---------------------------------------------------------------------------
 */
static double xfindBestCoeffCodMethod(int filterCoeffSymQuant[][ALF_MAX_NUM_COEF], int sqrFiltLength, int filters_per_fr, double errorForce0CoeffTab[NO_VAR_BINS][2], double lambda)
{
    int coeffBits, i;
    double error = 0, lagrangian;
    int coeffmulti[NO_VAR_BINS][ALF_MAX_NUM_COEF];
    int g;

    for (g = 0; g < filters_per_fr; g++) {
        for (i = 0; i < sqrFiltLength; i++) {
            coeffmulti[g][i] = filterCoeffSymQuant[g][i];
        }
    }
    predictALFCoeff(coeffmulti, sqrFiltLength, filters_per_fr);

    coeffBits = 0;
    for (g = 0; g < filters_per_fr; g++) {
        for (i = 0; i < (int)ALF_MAX_NUM_COEF; i++) {
            coeffBits += svlc_bitrate_estimate[64 + coeffmulti[g][i]];
        }
        error += errorForce0CoeffTab[g][1];
    }
    lagrangian = error + lambda * coeffBits;

    return (lagrangian);
}

/* ---------------------------------------------------------------------------
 */
static void add_A(int64_t Amerged[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF], int64_t A[][ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF], int start, int stop, int size)
{
    int i, j, ind;

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            Amerged[i][j] = 0;
            for (ind = start; ind <= stop; ind++) {
                Amerged[i][j] += A[ind][i][j];
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void add_b(double *bmerged, double b[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF], int start, int stop, int size)
{
    int i, ind;

    for (i = 0; i < size; i++) {
        bmerged[i] = 0;
        for (ind = start; ind <= stop; ind++) {
            bmerged[i] += b[ind][i];
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void roundFiltCoeff(int *FilterCoeffQuan, double *FilterCoeff, int sqrFiltLength, int factor)
{
    int i, diffInt, sign;
    double diff;

    for (i = 0; i < sqrFiltLength; i++) {
        sign = (FilterCoeff[i] > 0) ? 1 : -1;
        diff = FilterCoeff[i] * sign;
        diffInt = (int)(diff * (double)factor + 0.5);
        FilterCoeffQuan[i] = diffInt * sign;
    }
}

/* ---------------------------------------------------------------------------
 */
static double calculateErrorCoeffProvided(int64_t A[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF], double *b, double *c, int size)
{
    int i, j;
    double error = 0, sum;

    for (i = 0; i < size; i++) { // diagonal
        for (sum = 0, j = i + 1; j < size; j++) {
            sum += (A[j][i] + A[i][j]) * c[j];
        }
        error += (A[i][i] * c[i] + sum - 2 * b[i]) * c[i];
    }

    return error;
}

/* ---------------------------------------------------------------------------
 */
static double QuantizeIntegerFilterPP(double *filterCoeff, int *filterCoeffQuant, int64_t E[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF], double *y, int sqrFiltLength)
{
    double error;
    int filterCoeffQuantMod[ALF_MAX_NUM_COEF];
    int factor = (1 << ((int)ALF_NUM_BIT_SHIFT));
    int i;
    int quantCoeffSum, minInd, targetCoeffSumInt, k, diff;
    double targetCoeffSum, errMin;
    const int *weights = tab_weightsShape1Sym;

    gnsSolveByChol(E, y, filterCoeff, sqrFiltLength);
    targetCoeffSum = 0;
    quantCoeffSum = 0;
    roundFiltCoeff(filterCoeffQuant, filterCoeff, sqrFiltLength, factor);

    for (i = 0; i < sqrFiltLength; i++) {
        targetCoeffSum += (weights[i] * filterCoeff[i] * factor);
        quantCoeffSum += weights[i] * filterCoeffQuant[i];
    }
    targetCoeffSumInt = ROUND(targetCoeffSum);

    while (quantCoeffSum != targetCoeffSumInt) {
        diff = (quantCoeffSum - targetCoeffSumInt);
        diff = (diff < 0) ? (-diff) : diff;
        errMin = 0;
        minInd = -1;
        for (k = 0; k < sqrFiltLength; k++) {
            if (weights[k] <= diff) {
                for (i = 0; i < sqrFiltLength; i++) {
                    filterCoeffQuantMod[i] = filterCoeffQuant[i];
                }
                if (quantCoeffSum > targetCoeffSumInt) {
                    filterCoeffQuantMod[k]--;
                } else {
                    filterCoeffQuantMod[k]++;
                }
                for (i = 0; i < sqrFiltLength; i++) {
                    filterCoeff[i] = (double)filterCoeffQuantMod[i] / (double)factor;
                }
                error = calculateErrorCoeffProvided(E, y, filterCoeff, sqrFiltLength);
                if (error < errMin || minInd == -1) {
                    errMin = error;
                    minInd = k;
                }
            } // if (weights(k)<=diff)
        } // for (k=0; k<sqrFiltLength; k++)
        if (quantCoeffSum > targetCoeffSumInt) {
            filterCoeffQuant[minInd]--;
        } else {
            filterCoeffQuant[minInd]++;
        }

        quantCoeffSum = 0;
        for (i = 0; i < sqrFiltLength; i++) {
            quantCoeffSum += weights[i] * filterCoeffQuant[i];
        }
    }

    checkFilterCoeffValue(filterCoeffQuant, sqrFiltLength);

    for (i = 0; i < sqrFiltLength; i++) {
        filterCoeff[i] = (double)filterCoeffQuant[i] / (double)factor;
    }

    error = calculateErrorCoeffProvided(E, y, filterCoeff, sqrFiltLength);

    return (error);
}

/* ---------------------------------------------------------------------------
 */
static double findFilterCoeff(alf_ctx_t *Enc_ALF, int64_t EGlobalSeq[][ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF], double yGlobalSeq[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF],
                              double *pixAccGlobalSeq, int filterCoeffSeq[][ALF_MAX_NUM_COEF], int filterCoeffQuantSeq[][ALF_MAX_NUM_COEF], int intervalBest[NO_VAR_BINS][2],
                              int varIndTab[NO_VAR_BINS], int sqrFiltLength, int filters_per_fr, double errorTabForce0Coeff[NO_VAR_BINS][2])
{
    double pixAcc_temp;
    int filterCoeffQuant[ALF_MAX_NUM_COEF];
    double filterCoeff[ALF_MAX_NUM_COEF];
    double error;
    int k, filtNo;

    error = 0;
    for (filtNo = 0; filtNo < filters_per_fr; filtNo++) {
        add_A(Enc_ALF->m_auto_temp, EGlobalSeq, intervalBest[filtNo][0], intervalBest[filtNo][1], sqrFiltLength);
        add_b(Enc_ALF->m_cross_temp, yGlobalSeq, intervalBest[filtNo][0], intervalBest[filtNo][1], sqrFiltLength);

        pixAcc_temp = 0;
        for (k = intervalBest[filtNo][0]; k <= intervalBest[filtNo][1]; k++) {
            pixAcc_temp += pixAccGlobalSeq[k];
            varIndTab[k] = filtNo;
        }

        // find coefficients
        errorTabForce0Coeff[filtNo][1] = pixAcc_temp + QuantizeIntegerFilterPP(filterCoeff, filterCoeffQuant, Enc_ALF->m_auto_temp, Enc_ALF->m_cross_temp, sqrFiltLength);
        errorTabForce0Coeff[filtNo][0] = pixAcc_temp;
        error += errorTabForce0Coeff[filtNo][1];

        for (k = 0; k < sqrFiltLength; k++) {
            filterCoeffSeq[filtNo][k] = filterCoeffQuant[k];
            filterCoeffQuantSeq[filtNo][k] = filterCoeffQuant[k];
        }
    }
    return (error);
}

/* ---------------------------------------------------------------------------
 */
static
void xfindBestFilterVarPred(alf_ctx_t *Enc_ALF, double ySym[ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF], int64_t ESym[][ALF_MAX_NUM_COEF][ALF_MAX_NUM_COEF],
                            double *pixAcc, int filterCoeffSym[][ALF_MAX_NUM_COEF], int *filters_per_fr_best, int varIndTab[], double lambda_val, int numMaxFilters)
{
    int filterCoeffSymQuant[NO_VAR_BINS][ALF_MAX_NUM_COEF];
    int filters_per_fr, firstFilt, interval[NO_VAR_BINS][2], intervalBest[NO_VAR_BINS][2];
    double  lagrangian, lagrangianMin;
    int sqrFiltLength;
    double errorForce0CoeffTab[NO_VAR_BINS][2];

    sqrFiltLength = (int)ALF_MAX_NUM_COEF;

    // zero all variables
    memset(varIndTab, 0, sizeof(int)*NO_VAR_BINS);
    memset(filterCoeffSym, 0, sizeof(int)*ALF_MAX_NUM_COEF * NO_VAR_BINS);
    memset(filterCoeffSymQuant, 0, sizeof(int)*ALF_MAX_NUM_COEF * NO_VAR_BINS);

    firstFilt = 1;
    lagrangianMin = 0;
    filters_per_fr = NO_VAR_BINS;

    while (filters_per_fr >= 1) {
        mergeFiltersGreedy(Enc_ALF, ySym, ESym, pixAcc, interval, sqrFiltLength, filters_per_fr);
        findFilterCoeff(Enc_ALF, ESym, ySym, pixAcc, filterCoeffSym, filterCoeffSymQuant, interval,
                        varIndTab, sqrFiltLength, filters_per_fr, errorForce0CoeffTab);

        lagrangian = xfindBestCoeffCodMethod(filterCoeffSymQuant, sqrFiltLength, filters_per_fr, errorForce0CoeffTab, lambda_val);
        if (lagrangian < lagrangianMin || firstFilt == 1 || filters_per_fr == numMaxFilters) {
            firstFilt = 0;
            lagrangianMin = lagrangian;

            (*filters_per_fr_best) = filters_per_fr;
            memcpy(intervalBest, interval, NO_VAR_BINS * 2 * sizeof(int));
        }
        filters_per_fr--;
    }

    findFilterCoeff(Enc_ALF, ESym, ySym, pixAcc, filterCoeffSym, filterCoeffSymQuant, intervalBest,
                    varIndTab, sqrFiltLength, (*filters_per_fr_best), errorForce0CoeffTab);

    if (*filters_per_fr_best == 1) {
        memset(varIndTab, 0, sizeof(int)*NO_VAR_BINS);
    }
}

/* ---------------------------------------------------------------------------
 */
static
int compare_coef(const void *value1, const void *value2)
{
    DhNc *a = (DhNc*)value1;
    DhNc *b = (DhNc*)value2;
    double temp = (a->dh - b->dh);
    return temp > 0.0 ? 1 : (temp < 0.0 ? -1 : 0);
}

/* ---------------------------------------------------------------------------
 */
static
void xQuantFilterCoef(double *hh, int *qh)
{
    int i;
    const int N = (int)ALF_MAX_NUM_COEF;
    int max_value, min_value;
    double dbl_total_gain;
    int total_gain, q_total_gain;
    int upper, lower;
    DhNc dhnc[ALF_MAX_NUM_COEF];
    const int *pFiltMag = tab_weightsShape1Sym;

    max_value = (1 << (1 + ALF_NUM_BIT_SHIFT)) - 1;
    min_value = 0 - (1 << (1 + ALF_NUM_BIT_SHIFT));
    dbl_total_gain = 0.0;
    q_total_gain = 0;

    for (i = 0; i < N; i++) {
        if (hh[i] >= 0.0) {
            qh[i] = (int)(hh[i] * (1 << ALF_NUM_BIT_SHIFT) + 0.5);
        } else {
            qh[i] = -(int)(-hh[i] * (1 << ALF_NUM_BIT_SHIFT) + 0.5);
        }
        dhnc[i].dh = (double)qh[i] / (double)(1 << ALF_NUM_BIT_SHIFT) - hh[i];
        dhnc[i].dh *= pFiltMag[i];
        dbl_total_gain += hh[i] * pFiltMag[i];
        q_total_gain += qh[i] * pFiltMag[i];
        dhnc[i].nc = i;
    }

    // modification of quantized filter coefficients
    total_gain = (int)(dbl_total_gain * (1 << ALF_NUM_BIT_SHIFT) + 0.5);
    if (q_total_gain != total_gain) {
        qsort(dhnc, N, sizeof(struct dh_nc), compare_coef);
        if (q_total_gain > total_gain) {
            upper = N - 1;
            while (q_total_gain > total_gain + 1) {
                i = dhnc[upper % N].nc;
                qh[i]--;
                q_total_gain -= pFiltMag[i];
                upper--;
            }

            if (q_total_gain == total_gain + 1) {
                if (dhnc[N - 1].dh > 0) {
                    qh[N - 1]--;
                } else {
                    i = dhnc[upper % N].nc;
                    qh[i]--;
                    qh[N - 1]++;
                }
            }
        } else if (q_total_gain < total_gain) {
            lower = 0;
            while (q_total_gain < total_gain - 1) {
                i = dhnc[lower % N].nc;
                qh[i]++;
                q_total_gain += pFiltMag[i];
                lower++;
            }

            if (q_total_gain == total_gain - 1) {
                if (dhnc[N - 1].dh < 0) {
                    qh[N - 1]++;
                } else {
                    i = dhnc[lower % N].nc;
                    qh[i]++;
                    qh[N - 1]--;
                }
            }
        }
    }

    // set of filter coefficients
    for (i = 0; i < N; i++) {
        qh[i] = XAVS2_CLIP3(min_value, max_value, qh[i]);
    }

    checkFilterCoeffValue(qh, N);
}

/* ---------------------------------------------------------------------------
 */
static
void deriveFilterInfo(alf_ctx_t *Enc_ALF, ALFParam *alfPictureParam, AlfCorrData **alfCorr_ptr, int maxNumFilters, double lambda)
{
    int numCoeff = ALF_MAX_NUM_COEF;
    double coef[ALF_MAX_NUM_COEF];
    int compIdx, lambdaForMerge, numFilters;

    compIdx = IMG_Y;
    AlfCorrData *alfCorr = alfCorr_ptr[compIdx];
    ALFParam *alfFiltParam = &alfPictureParam[compIdx];
    alfFiltParam->alf_flag = 1;
    lambdaForMerge = ((int)lambda) * (1 << (2 * Enc_ALF->m_uiBitIncrement));
    memset(Enc_ALF->m_varIndTab, 0, sizeof(int)*NO_VAR_BINS);
    xfindBestFilterVarPred(Enc_ALF, alfCorr->m_crossCorr, alfCorr->m_autoCorr, alfCorr->pixAcc, Enc_ALF->m_filterCoeffSym, &numFilters, Enc_ALF->m_varIndTab, lambdaForMerge, maxNumFilters);
    xcodeFiltCoeff(Enc_ALF->m_filterCoeffSym, Enc_ALF->m_varIndTab, numFilters, alfFiltParam);

    compIdx = IMG_U;
    alfCorr = alfCorr_ptr[compIdx];
    alfFiltParam = &alfPictureParam[compIdx];
    alfFiltParam->alf_flag = 1;
    gnsSolveByChol(alfCorr->m_autoCorr[0], alfCorr->m_crossCorr[0], coef, numCoeff);
    xQuantFilterCoef(coef, Enc_ALF->m_filterCoeffSym[0]);
    memcpy(alfFiltParam->coeffmulti[0], Enc_ALF->m_filterCoeffSym[0], sizeof(int)*numCoeff);
    predictALFCoeff(alfFiltParam->coeffmulti, numCoeff, alfFiltParam->filters_per_group);

    compIdx = IMG_V;
    alfCorr = alfCorr_ptr[compIdx];
    alfFiltParam = &alfPictureParam[compIdx];
    alfFiltParam->alf_flag = 1;
    gnsSolveByChol(alfCorr->m_autoCorr[0], alfCorr->m_crossCorr[0], coef, numCoeff);
    xQuantFilterCoef(coef, Enc_ALF->m_filterCoeffSym[0]);
    memcpy(alfFiltParam->coeffmulti[0], Enc_ALF->m_filterCoeffSym[0], sizeof(int)*numCoeff);
    predictALFCoeff(alfFiltParam->coeffmulti, numCoeff, alfFiltParam->filters_per_group);
}

/**
 * ---------------------------------------------------------------------------
 * Function: ALF parameter selection
 * Input:
 *    alfPictureParam: The ALF parameter
 *              apsId: The ALF parameter index in the buffer
 *       isNewApsSent：The New flag index
 *       lambda      : The lambda value in the ALF-RD decision
 * Return:
 * ---------------------------------------------------------------------------
 */
static
void setCurAlfParam(xavs2_t *h, alf_ctx_t *Enc_ALF, aec_t *p_aec, ALFParam *alfPictureParam, double lambda)
{
    int compIdx, i;
    AlfCorrData *alfPicCorr[IMG_CMPNTS];
    double costMin, cost;
    ALFParam tempAlfParam[IMG_CMPNTS];
    int picHeaderBitrate = 0;
    costMin = MAX_DOUBLE;

    for (compIdx = 0; compIdx < IMG_CMPNTS; compIdx++) {
        init_alf_frame_param(&tempAlfParam[compIdx]);
        alfPicCorr[compIdx] = &Enc_ALF->m_pic_corr[compIdx];
    }

    for (i = 0; i < Enc_ALF->m_alfReDesignIteration; i++) {
        // redesign filter according to the last on off results, "!i" replace TRUE/FALSE to control design or redesign
        accumulateLCUCorrelations(h, alfPicCorr, Enc_ALF->m_alfCorr, !i);
        deriveFilterInfo(Enc_ALF, tempAlfParam, alfPicCorr, NO_VAR_BINS, lambda);

        // estimate cost
        cost = executePicLCUOnOffDecisionRDOEstimate(h, Enc_ALF, p_aec, tempAlfParam, lambda, Enc_ALF->m_alfCorr[0]);
        picHeaderBitrate = estimateALFBitrateInPicHeader(tempAlfParam);
        cost += (double)picHeaderBitrate * lambda;
        if (cost < costMin) {
            costMin = cost;
            for (compIdx = 0; compIdx < IMG_CMPNTS; compIdx++) {
                copyALFparam(&alfPictureParam[compIdx], &tempAlfParam[compIdx], compIdx);
            }
        }
    }

    alfPicCorr[0] = alfPicCorr[1] = alfPicCorr[2] = NULL;
}

/**
 * ===========================================================================
 * interface function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
int alf_get_buffer_size(const xavs2_param_t *param)
{
    int size_lcu      = 1 << param->lcu_bit_level;
    int width_in_lcu  = (param->org_width  + size_lcu - 1) >> param->lcu_bit_level;
    int height_in_lcu = (param->org_height + size_lcu - 1) >> param->lcu_bit_level;
    int num_lcu = height_in_lcu * width_in_lcu;
    int maxNumTemporalLayer = (int)(log10((float)(param->i_gop_size)) / log10(2.0) + 1);

    int mem_size = sizeof(alf_ctx_t)
                   + 2 * IMG_CMPNTS * num_lcu * sizeof(AlfCorrData)
                   + maxNumTemporalLayer * IMG_CMPNTS * num_lcu * sizeof(AlfCorrData)
                   + num_lcu * sizeof(int)      // m_numSlicesDataInOneLCU
                   + num_lcu * sizeof(int8_t)   // tab_lcu_region
                   + num_lcu * IMG_CMPNTS * sizeof(bool_t)  // is_alf_lcu_on[3]
                   + num_lcu * sizeof(AlfCorrData)  //for other function temp variable alfPicCorr
                   + CACHE_LINE_SIZE * 50;

    return mem_size;
}


/* ---------------------------------------------------------------------------
 */
void alf_init_buffer(xavs2_t *h, uint8_t *mem_base)
{
    // 希尔伯特扫描顺序
    static const uint8_t regionTable[NO_VAR_BINS] = {
        0, 1, 4, 5, 15, 2, 3, 6, 14, 11, 10, 7, 13, 12, 9, 8
    }
    ;
    int width_in_lcu  = h->i_width_in_lcu;
    int height_in_lcu = h->i_height_in_lcu;
    int quad_w_in_lcu = ((width_in_lcu  + 1) >> 2);
    int quad_h_in_lcu = ((height_in_lcu + 1) >> 2);
    int region_idx_x;
    int region_idx_y;
    int i, j;

    int num_lcu = height_in_lcu * width_in_lcu;
    int compIdx, n;
    int maxNumTemporalLayer = (int)(log10((float)(h->param->i_gop_size)) / log10(2.0) + 1);
    int mem_size;
    uint8_t *mem_ptr = mem_base;
    alf_ctx_t *Enc_ALF;

    mem_size = alf_get_buffer_size(h->param);
    memset(mem_ptr, 0, mem_size);

    Enc_ALF                          = (alf_ctx_t *)mem_ptr;
    mem_ptr                         += sizeof(alf_ctx_t);
    Enc_ALF->m_alfReDesignIteration  = 3;
    Enc_ALF->m_uiBitIncrement        = 0;

    for (compIdx = 0; compIdx < IMG_CMPNTS; compIdx++) {
        Enc_ALF->m_alfCorr[compIdx]  = (AlfCorrData *)mem_ptr;
        mem_ptr += (num_lcu * sizeof(AlfCorrData));
    }

    for (compIdx = 0; compIdx < IMG_CMPNTS; compIdx++) {
        Enc_ALF->m_alfNonSkippedCorr[compIdx] = (AlfCorrData *)mem_ptr;
        mem_ptr += (num_lcu * sizeof(AlfCorrData));
    }

    Enc_ALF->m_alfPrevCorr = (AlfCorrData *)mem_ptr;
    mem_ptr += maxNumTemporalLayer * IMG_CMPNTS * num_lcu * sizeof(AlfCorrData);
    for (n = 0; n < maxNumTemporalLayer; n++) {
        for (compIdx = 0; compIdx < IMG_CMPNTS; compIdx++) {
            init_alf_frame_param(&(Enc_ALF->m_alfPictureParam[n][compIdx]));
        }
    }

    for (n = 0; n < NO_VAR_BINS; n++) {
        Enc_ALF->m_coeffNoFilter[n][ALF_MAX_NUM_COEF - 1] = (1 << ALF_NUM_BIT_SHIFT);
    }

    Enc_ALF->m_numSlicesDataInOneLCU = (int *)mem_ptr;
    mem_ptr += (num_lcu * sizeof(int));

    Enc_ALF->tab_lcu_region = (int8_t *)mem_ptr;
    mem_ptr += (num_lcu * sizeof(int8_t));

    h->is_alf_lcu_on = (bool_t(*)[IMG_CMPNTS])mem_ptr;
    mem_ptr += (num_lcu * IMG_CMPNTS * sizeof(bool_t));

    for (j = 0; j < height_in_lcu; j++) {
        region_idx_y = (quad_h_in_lcu == 0) ? 3 : XAVS2_MIN(j / quad_h_in_lcu, 3);
        for (i = 0; i < width_in_lcu; i++) {
            region_idx_x = (quad_w_in_lcu == 0) ? 3 : XAVS2_MIN(i / quad_w_in_lcu, 3);
            Enc_ALF->tab_lcu_region[j * width_in_lcu + i] = regionTable[region_idx_y * 4 + region_idx_x];
        }
    }

    h->enc_alf = Enc_ALF;
    aec_init_coding_state(&h->cs_data.cs_alf_cu_ctr);
    aec_init_coding_state(&h->cs_data.cs_alf_initial);
}

/* ---------------------------------------------------------------------------
 */
void alf_filter_one_frame(xavs2_t *h)
{
    aec_t *p_aec = &h->aec;
    ALFParam *alfPictureParam = h->pic_alf_params;
    xavs2_frame_t *p_org = h->fenc;
    xavs2_frame_t *p_rec = h->img_alf;
    alf_ctx_t *Enc_ALF = (alf_ctx_t *)h->enc_alf;
    double lambda_mode = h->f_lambda_mode * LAMBDA_SCALE_LUMA;
    int i;

    h->copy_aec_state_rdo(&h->cs_data.cs_alf_initial, p_aec);

    // init ALF buffers
    for (i = 0; i < IMG_CMPNTS; i++) {
        init_alf_frame_param(&alfPictureParam[i]);
    }

    setCurAlfParam(h, Enc_ALF, p_aec, alfPictureParam, lambda_mode);
    executePicLCUOnOffDecision(h, Enc_ALF, p_aec, alfPictureParam, lambda_mode,
                               p_org, p_rec, h->fdec);

    // set ALF frame parameters
    for (i = 0; i < IMG_CMPNTS; i++) {
        h->pic_alf_on[i] = alfPictureParam[i].alf_flag;
    }
}


