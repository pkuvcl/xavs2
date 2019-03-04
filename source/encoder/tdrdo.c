/*
 * tdrdo.c
 *
 * Description of this file:
 *    TDRDO functions definition of the xavs2 library
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
#include "tdrdo.h"
#include "wrapper.h"
#include "frame.h"


#define WORKBLOCKSIZE 64
#define SEARCHRANGE   64


/**
 * ===========================================================================
 * type defines
 * ===========================================================================
 */

typedef struct Frame {
    uint32_t    FrameWidth;
    uint32_t    FrameHeight;
    uint32_t    nStrideY;
    pel_t      *Y_base;
} Frame;

typedef struct BlockDistortion {
    uint32_t    GlobalBlockNumber;
    uint16_t    BlockNumInHeight;
    uint16_t    BlockNumInWidth;
    uint16_t    BlockWidth;
    uint16_t    BlockHeight;
    uint16_t    OriginX;
    uint16_t    OriginY;
    uint16_t    SearchRange;
    short       MVx;
    short       MVy;
    double      MSE;
    double      MVL;
    short       BlockQP;
    double      BlockLambda;
    short       BlockType;
} BlockDistortion, BD;

typedef struct FrameDistortion {
    uint32_t    FrameNumber;
    uint32_t    BlockSize;
    uint32_t    CUSize;
    uint32_t    TotalNumOfBlocks;
    uint32_t    TotalBlockNumInHeight;
    uint32_t    TotalBlockNumInWidth;
    BD         *BlockDistortionArray;
    struct FrameDistortion *subFrameDistortionArray;
} FrameDistortion, FD;

typedef struct DistortionList {
    uint32_t    TotalFrameNumber;
    uint32_t    FrameWidth;
    uint32_t    FrameHeight;
    uint32_t    BlockSize;
    FD         *FrameDistortionArray;
} DistortionList, DL;

struct td_rdo_t {
    Frame       porgF;
    Frame       ppreF;
    Frame       precF;
    DL          OMCPDList;
    DL          RealDList;
    FD         *pOMCPFD, *pRealFD;

    int         StepLength;
    double     *KappaTable;
    double      GlobeLambdaRatio;
    int         GlobeFrameNumber;
    int         CurMBQP;
    int         QpOffset[32];
    int         globenumber;

    double     *D;
    double     *DMCP;
    double     *BetaTable;
    double     *MultiplyBetas;
};

typedef struct Block {
    uint32_t    BlockWidth;
    uint32_t    BlockHeight;
    uint32_t    OriginX;
    uint32_t    OriginY;
} Block;


/**
 * ===========================================================================
 * local function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
static DL *CreatDistortionList(DL *NewDL, uint32_t totalframenumber, uint32_t width, uint32_t height, uint32_t blocksize, uint32_t cusize)
{
    int tBlockNumInHeight, tBlockNumInWidth, tBlockNumber;
    uint32_t i;

    NewDL->TotalFrameNumber = totalframenumber;
    NewDL->FrameWidth = width;
    NewDL->FrameHeight = height;
    blocksize = blocksize < 4 ? 4 : (blocksize > 64 ? 64 : blocksize);
    NewDL->BlockSize = blocksize;

    tBlockNumInHeight = (int)ceil(1.0 * height / blocksize);
    tBlockNumInWidth = (int)ceil(1.0 * width / blocksize);
    tBlockNumber = tBlockNumInHeight * tBlockNumInWidth;
    for (i = 0; i < totalframenumber; i++) {
        NewDL->FrameDistortionArray[i].FrameNumber = i;
        NewDL->FrameDistortionArray[i].BlockSize = blocksize;
        NewDL->FrameDistortionArray[i].CUSize = cusize;
        NewDL->FrameDistortionArray[i].TotalNumOfBlocks = tBlockNumber;
        NewDL->FrameDistortionArray[i].TotalBlockNumInHeight = tBlockNumInHeight;
        NewDL->FrameDistortionArray[i].TotalBlockNumInWidth = tBlockNumInWidth;
        NewDL->FrameDistortionArray[i].BlockDistortionArray = NULL;
        NewDL->FrameDistortionArray[i].subFrameDistortionArray = NULL;
    }

    return NewDL;
}


/* ---------------------------------------------------------------------------
 */
static double CalculateBlockMSE(Frame *FA, Frame *FB, Block *A, Block *B)
{
    uint16_t x, y;
    int e, blockpixel = A->BlockHeight * A->BlockWidth;
    pel_t *YA, *YB;
    double dSSE = 0;

    YA = FA->Y_base + A->OriginY * FA->nStrideY + A->OriginX;
    YB = FB->Y_base + B->OriginY * FB->nStrideY + B->OriginX;
    for (y = 0; y < A->BlockHeight; y++) {
        for (x = 0; x < A->BlockWidth; x++) {
            e = YA[x] - YB[x];
            dSSE += e * e;
        }
        YA = YA + FA->nStrideY;
        YB = YB + FB->nStrideY;
    }
    return dSSE / blockpixel;
}

/* ---------------------------------------------------------------------------
 */
static void MotionDistortion(FD *currentFD, Frame *FA, Frame *FB, uint32_t searchrange)
{
    static int dlx[9] = {0, -2, -1,  0,  1, 2, 1, 0, -1};
    static int dly[9] = {0,  0, -1, -2, -1, 0, 1, 2,  1};
    static int dsx[5] = {0, -1,  0,  1,  0};
    static int dsy[5] = {0,  0, -1,  0,  1};
    double currentMSE, candidateMSE;
    BD *currentBD;
    Block BA, BB;
    Block *pBA, *pBB;
    uint32_t blocksize, TotalBlockNumInHeight, TotalBlockNumInWidth, nBH, nBW;
    int top, bottom, left, right;
    int *searchpatternx = NULL;
    int *searchpatterny = NULL;
    int patternsize = 0;
    int cx, cy;
    int l;
    int flag9p, flag5p;
    int nextcx = 0;
    int nextcy = 0;
    int x, y;

    pBA = &BA;
    pBB = &BB;

    blocksize = currentFD->BlockSize;
    TotalBlockNumInHeight = currentFD->TotalBlockNumInHeight;
    TotalBlockNumInWidth  = currentFD->TotalBlockNumInWidth;
    for (nBH = 0; nBH < TotalBlockNumInHeight; nBH++) {
        for (nBW = 0; nBW < TotalBlockNumInWidth; nBW++) {
            memset(pBA, 0, sizeof(BA));
            memset(pBB, 0, sizeof(BB));

            pBA->OriginX = blocksize * nBW;
            pBA->OriginY = blocksize * nBH;
            pBA->BlockHeight = blocksize * (nBH + 1) < FA->FrameHeight ? blocksize : FA->FrameHeight - blocksize * nBH;
            pBA->BlockWidth = blocksize * (nBW + 1) < FA->FrameWidth ? blocksize : FA->FrameWidth - blocksize * nBW;

            currentBD = &currentFD->BlockDistortionArray[nBH * TotalBlockNumInWidth + nBW];
            currentBD->GlobalBlockNumber = nBH * TotalBlockNumInWidth + nBW;
            currentBD->BlockNumInHeight = (uint16_t)nBH;
            currentBD->BlockNumInWidth = (uint16_t)nBW;
            currentBD->BlockWidth = (uint16_t)pBA->BlockWidth;
            currentBD->BlockHeight = (uint16_t)pBA->BlockHeight;
            currentBD->OriginX = (uint16_t)pBA->OriginX;
            currentBD->OriginY = (uint16_t)pBA->OriginY;
            currentBD->SearchRange = (uint16_t)searchrange;

            top    = pBA->OriginY - searchrange;
            bottom = pBA->OriginY + searchrange;
            left   = pBA->OriginX - searchrange;
            right  = pBA->OriginX + searchrange;
            top    = XAVS2_CLIP3(0, (int)(FB->FrameHeight - pBA->BlockHeight), top);
            bottom = XAVS2_CLIP3(0, (int)(FB->FrameHeight - pBA->BlockHeight), bottom);
            left   = XAVS2_CLIP3(0, (int)(FB->FrameWidth  - pBA->BlockWidth ), left);
            right  = XAVS2_CLIP3(0, (int)(FB->FrameWidth  - pBA->BlockWidth ), right);

            pBB->BlockHeight = pBA->BlockHeight;
            pBB->BlockWidth = pBA->BlockWidth;

            flag5p = 0;
            flag9p = 1;
            cy = pBA->OriginY;
            cx = pBA->OriginX;
            while (flag9p || flag5p) {
                candidateMSE = 1048576; // 1048576 = 1024 * 1024;
                if (flag9p) {
                    searchpatternx = dlx;
                    searchpatterny = dly;
                    patternsize = 9;
                } else if (flag5p) {
                    searchpatternx = dsx;
                    searchpatterny = dsy;
                    patternsize = 5;
                }

                for (l = 0; l < patternsize; l++) {
                    y = cy + searchpatterny[l];
                    x = cx + searchpatternx[l];
                    if (x >= left && x <= right && y >= top && y <= bottom) {
                        pBB->OriginX = x;
                        pBB->OriginY = y;
                        currentMSE = CalculateBlockMSE(FA, FB, pBA, pBB);
                        if (currentMSE < candidateMSE) {
                            candidateMSE = currentMSE;
                            currentBD->MSE = currentMSE;
                            nextcx = x;
                            nextcy = y;
                        }
                    }
                }
                if (cy == nextcy && cx == nextcx) {
                    flag9p = 0;
                    flag5p = 1 - flag5p;
                } else {
                    cy = nextcy;
                    cx = nextcx;
                }
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 */
static void StoreLCUInf(FD *curRealFD, int LeaderBlockNumber, int cuinwidth, int iqp, rdcost_t lambda, int curtype)
{
    BD *workBD;
    int LeaderNumber = ((LeaderBlockNumber % cuinwidth) / 8 + LeaderBlockNumber / cuinwidth / 8 * curRealFD->TotalBlockNumInWidth) * (curRealFD->CUSize / curRealFD->BlockSize);
    int workBlockNum;
    uint32_t x, y, top, left, bottom, right;

    top    = LeaderNumber / curRealFD->TotalBlockNumInWidth;
    left   = LeaderNumber % curRealFD->TotalBlockNumInWidth;
    bottom = top + curRealFD->CUSize / curRealFD->BlockSize;
    bottom = bottom <= curRealFD->TotalBlockNumInHeight ? bottom : curRealFD->TotalBlockNumInHeight;
    right  = left + curRealFD->CUSize / curRealFD->BlockSize;
    right  = right <= curRealFD->TotalBlockNumInWidth ? right : curRealFD->TotalBlockNumInWidth;

    workBlockNum = LeaderNumber;
    for (y = top; y < bottom; y++) {
        for (x = left; x < right; x++) {
            workBD = &curRealFD->BlockDistortionArray[workBlockNum + x - left];
            workBD->BlockQP     = (short)iqp;
            workBD->BlockLambda = (double)lambda;
            workBD->BlockType   = (short)curtype;
        }
        workBlockNum = workBlockNum + curRealFD->TotalBlockNumInWidth;
    }
}

/* ---------------------------------------------------------------------------
 */
static ALWAYS_INLINE double F(double invalue)
{
    double f;

    if (invalue < 0.5F) {
        f = 0.015F;
    } else if (invalue < 2.0F) {
        f = (54.852103 * invalue * invalue + 10.295705 * invalue - 3.667158) / 1000;
    } else if (invalue < 8.0F) {
        f = (-19.235059 * invalue * invalue + 311.129530 * invalue - 317.360050) / 1000 - 0.2280 + 0.2363;
    } else {
        f = 0.949F;
    }

    return XAVS2_CLIP3F(0.015F, 0.949F, f);
}

/* ---------------------------------------------------------------------------
 */
static void CaculateKappaTableLDP(xavs2_t *h, DL *omcplist, DL *realDlist, int framenum, int FrameQP)
{
    td_rdo_t *td_rdo = h->td_rdo;
    BD *p1stBD, *pcurBD;
    const double tdrdoAlpha = 0.94F;
    double fxvalue;
    double *D, *DMCP;
    double *BetaTable, *MultiplyBetas;
    double DsxKappa, Ds;
    int TotalBlocksInAframe = realDlist->FrameDistortionArray[0].TotalNumOfBlocks;
    int BetaLength;
    int PreFrameQP;
    int t, b;

    BetaLength = realDlist->TotalFrameNumber - 1  - framenum - 1;
    BetaLength = XAVS2_MIN(2, BetaLength);

    memset(td_rdo->KappaTable, 0, TotalBlocksInAframe * sizeof(double));
    if (framenum <= 0) {
        return;
    }

    D             = td_rdo->D;
    DMCP          = td_rdo->DMCP;
    BetaTable     = td_rdo->BetaTable;
    MultiplyBetas = td_rdo->MultiplyBetas;
    memset(D,             0, TotalBlocksInAframe * sizeof(double));
    memset(DMCP,          0, TotalBlocksInAframe * sizeof(double));
    memset(BetaTable,     0, TotalBlocksInAframe * sizeof(double));
    memset(MultiplyBetas, 0, TotalBlocksInAframe * sizeof(double));

    p1stBD = realDlist->FrameDistortionArray[framenum - 1].BlockDistortionArray;
    for (b = 0; b < TotalBlocksInAframe; b++) {
        D[b] = p1stBD[b].MSE;
        BetaTable[b] = 1.0F;
    }

    for (b = 0; b < TotalBlocksInAframe; b++) {
        MultiplyBetas[b] = 1.0;
    }

    pcurBD = omcplist->FrameDistortionArray[framenum - 1].BlockDistortionArray;
    for (t = 0; t <= BetaLength; t++) {
        PreFrameQP = FrameQP - td_rdo->QpOffset[framenum % h->i_gop_size] + td_rdo->QpOffset[(framenum + t) % h->i_gop_size];

        for (b = 0; b < TotalBlocksInAframe; b++) {
            DMCP[b] = tdrdoAlpha * (D[b] + pcurBD[b].MSE);
        }
        for (b = 0; b < TotalBlocksInAframe; b++) {
            fxvalue = (sqrt(2.0) * pow(2.0, (PreFrameQP) / 8.0)) / sqrt(DMCP[b]);
            D[b] = DMCP[b] * F(fxvalue);
            BetaTable[b] = tdrdoAlpha * F(fxvalue);
            if (t > 0) {
                MultiplyBetas[b] *= BetaTable[b];
                td_rdo->KappaTable[b] += MultiplyBetas[b];
            }
        }
    }

    DsxKappa = Ds = 0.0F;

    for (b = 0; b < TotalBlocksInAframe; b++) {
        t = framenum - 1;
        Ds += realDlist->FrameDistortionArray[t].BlockDistortionArray[b].MSE;
        DsxKappa += realDlist->FrameDistortionArray[t].BlockDistortionArray[b].MSE * (1.0F + td_rdo->KappaTable[b]);
    }

    td_rdo->GlobeLambdaRatio = DsxKappa / Ds;
}

/* ---------------------------------------------------------------------------
 */
static void AdjustLcuQPLambdaLDP(xavs2_t *h, FD *curOMCPFD, int LeaderBlockNumber, int cuinwidth, rdcost_t *plambda)
{
    td_rdo_t *td_rdo = h->td_rdo;
    double ArithmeticMean, HarmonicMean, GeometricMean;
    double SumOfMSE;
    double Kappa, LambdaRatio, dDeltaQP;
    uint32_t x, y, top, left, bottom, right;
    int LeaderNumber;
    int workBlockNum;
    int counter, iDeltaQP;

    if (curOMCPFD == NULL) {
        return;
    }

    if (td_rdo->KappaTable == NULL) {
        dDeltaQP = 0.0F;
        iDeltaQP = dDeltaQP > 0 ? (int)(dDeltaQP + 0.5) : -(int)(-dDeltaQP + 0.5);
        iDeltaQP = XAVS2_CLIP3F(-2, 2, iDeltaQP);
        return;
    }

    LeaderNumber = ((LeaderBlockNumber % cuinwidth) / 8 + LeaderBlockNumber / cuinwidth / 8 * curOMCPFD->TotalBlockNumInWidth) * (curOMCPFD->CUSize / curOMCPFD->BlockSize);
    top    = LeaderNumber / curOMCPFD->TotalBlockNumInWidth;
    left   = LeaderNumber % curOMCPFD->TotalBlockNumInWidth;
    bottom = top + curOMCPFD->CUSize / curOMCPFD->BlockSize;
    bottom = bottom <= curOMCPFD->TotalBlockNumInHeight ? bottom : curOMCPFD->TotalBlockNumInHeight;
    right  = left + curOMCPFD->CUSize / curOMCPFD->BlockSize;
    right  = right <= curOMCPFD->TotalBlockNumInWidth ? right : curOMCPFD->TotalBlockNumInWidth;

    ArithmeticMean = 0.0;
    HarmonicMean   = 0.0;
    GeometricMean  = 1.0;
    SumOfMSE       = 0.0;
    counter        = 0;
    workBlockNum   = LeaderNumber;
    for (y = top; y < bottom; y++) {
        for (x = left; x < right; x++) {
            SumOfMSE += curOMCPFD->BlockDistortionArray[workBlockNum + x - left].MSE;
            Kappa = td_rdo->KappaTable[workBlockNum + x - left];
            ArithmeticMean += Kappa;
            HarmonicMean += 1.0 / Kappa;
            GeometricMean *= Kappa;
            counter++;
        }
        workBlockNum = workBlockNum + curOMCPFD->TotalBlockNumInWidth;
    }

    if (counter == 0) {
        return;
    }

    Kappa = ArithmeticMean / counter;
    SumOfMSE = SumOfMSE / counter;

    LambdaRatio = td_rdo->GlobeLambdaRatio / (1.0F + Kappa);
    LambdaRatio = XAVS2_CLIP3F(pow(2.0, -3.0 / 4.0), pow(2.0, 3.0 / 4.0), LambdaRatio);
    dDeltaQP    = (4.0 / log(2.0F)) * log(LambdaRatio);
    iDeltaQP    = dDeltaQP > 0.0 ? (int)(dDeltaQP + 0.5) : -(int)(-dDeltaQP - 0.5);
    iDeltaQP    = XAVS2_CLIP3F(-3, 3, iDeltaQP);
    *plambda    = (rdcost_t)((*plambda) * LambdaRatio);
}

/* ---------------------------------------------------------------------------
 * i_frame_index: frame number in file
 */
static FD *SearchFrameDistortionArray(DL *omcplist, int i_frame_index, int StepLength, int IntraPeriod)
{
    FD *NewFD = NULL;
    int keyframenum = i_frame_index / StepLength;
    int subframenumIndex = i_frame_index % StepLength;
    int subframenum;
    int i;

    if (subframenumIndex == 0) {
        NewFD = &omcplist->FrameDistortionArray[keyframenum - 1];
    }
    if (subframenumIndex != 0 && IntraPeriod == 0) {
        NewFD = &omcplist->FrameDistortionArray[keyframenum].subFrameDistortionArray[subframenumIndex - 1];
    }
    if (subframenumIndex != 0 && IntraPeriod != 0) {
        for (i = 0; i < StepLength - 1; i++) {
            subframenum = omcplist->FrameDistortionArray[keyframenum].subFrameDistortionArray[i].FrameNumber;
            if (subframenum == i_frame_index) {
                NewFD = &omcplist->FrameDistortionArray[keyframenum].subFrameDistortionArray[i];
                break;
            }
        }
    }

    return NewFD;
}


/**
 * ===========================================================================
 * interface function defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 */
int tdrdo_get_buffer_size(xavs2_param_t *param)
{
    int StepLength = param->num_bframes == 0 ? 1 : param->i_gop_size;
    int num_frames = 0;
    int size_blocks;

    if (param->enable_tdrdo) {
        if (!param->num_bframes) {
            num_frames += (param->num_frames / StepLength + 1);
            num_frames += (param->num_frames / StepLength + 1);
        } else {
            num_frames += (param->num_frames - 1) / StepLength + 1;
            num_frames += param->num_frames + 1;
        }
    }

    size_blocks = 5 * sizeof(double) * (int)ceil(1.0 * param->org_width / WORKBLOCKSIZE) * (int)ceil(1.0 * param->org_height / WORKBLOCKSIZE);

    return sizeof(td_rdo_t) + num_frames * sizeof(FD) + size_blocks;
}

/* ---------------------------------------------------------------------------
 */
int tdrdo_init(td_rdo_t *td_rdo, xavs2_param_t *param)
{
    uint8_t *mem_ptr = (uint8_t *)td_rdo;
    uint8_t *mem_start = mem_ptr;
    int size_buffer = tdrdo_get_buffer_size(param);
    int num_blocks = (int)ceil(1.0 * param->org_width / WORKBLOCKSIZE) * (int)ceil(1.0 * param->org_height / WORKBLOCKSIZE);
    int i;

    if (param->num_bframes != 0) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "TDRDO cannot be used in RA configuration.\n");
        return -1;
    }

    /* memory alloc */
    memset(td_rdo, 0, size_buffer);
    mem_ptr += sizeof(td_rdo_t);

    td_rdo->KappaTable = (double *)mem_ptr;
    mem_ptr += sizeof(double) * num_blocks;
    td_rdo->StepLength = param->num_bframes == 0 ? 1 : param->i_gop_size;

    if (!param->num_bframes) {
        td_rdo->OMCPDList.FrameDistortionArray = (FD *)mem_ptr;
        mem_ptr += (param->num_frames / td_rdo->StepLength + 1) * sizeof(FD);
        CreatDistortionList(&td_rdo->OMCPDList, param->num_frames / td_rdo->StepLength + 1, param->org_width, param->org_height, WORKBLOCKSIZE, 1 << param->lcu_bit_level);
        td_rdo->RealDList.FrameDistortionArray = (FD *)mem_ptr;
        mem_ptr += (param->num_frames / td_rdo->StepLength + 1) * sizeof(FD);
        CreatDistortionList(&td_rdo->RealDList, param->num_frames / td_rdo->StepLength + 1, param->org_width, param->org_height, WORKBLOCKSIZE, 1 << param->lcu_bit_level);
    } else {
        td_rdo->OMCPDList.FrameDistortionArray = (FD *)mem_ptr;
        mem_ptr += ((param->num_frames - 1) / td_rdo->StepLength + 1) * sizeof(FD);
        CreatDistortionList(&td_rdo->OMCPDList, (param->num_frames - 1) / td_rdo->StepLength + 1, param->org_width, param->org_height, WORKBLOCKSIZE, 1 << param->lcu_bit_level);
        td_rdo->RealDList.FrameDistortionArray = (FD *)mem_ptr;
        mem_ptr += (param->num_frames + 1) * sizeof(FD);
        CreatDistortionList(&td_rdo->RealDList, param->num_frames + 1, param->org_width, param->org_height, WORKBLOCKSIZE, 1 << param->lcu_bit_level);
    }


    td_rdo->porgF.FrameWidth  = param->org_width;
    td_rdo->porgF.FrameHeight = param->org_height;
    memcpy(&td_rdo->ppreF, &td_rdo->porgF, sizeof(Frame));
    memcpy(&td_rdo->precF, &td_rdo->porgF, sizeof(Frame));

    /* copy of QP offset */
    for (i = 0; i < param->i_gop_size; i++) {
        td_rdo->QpOffset[i] = param->cfg_ref_all[i].qp_offset;
    }

    td_rdo->D  = (double *)mem_ptr;
    mem_ptr   += num_blocks * sizeof(double);
    td_rdo->DMCP = (double *)mem_ptr;
    mem_ptr   += num_blocks * sizeof(double);
    td_rdo->BetaTable = (double *)mem_ptr;
    mem_ptr   += num_blocks * sizeof(double);
    td_rdo->MultiplyBetas = (double *)mem_ptr;
    mem_ptr   += num_blocks * sizeof(double);

    if (mem_ptr - mem_start <= size_buffer) {
        return 0;
    } else {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "TDRDO init error detected.\n");
        return -1;
    }
}

/* ---------------------------------------------------------------------------
 */
void tdrdo_destroy(td_rdo_t *td_rdo)
{
    UNUSED_PARAMETER(td_rdo);
}

/* ---------------------------------------------------------------------------
 */
void tdrdo_frame_start(xavs2_t *h)
{
    td_rdo_t *td_rdo = h->td_rdo;
    assert(td_rdo != NULL);

    td_rdo->GlobeFrameNumber = h->ip_pic_idx;
    if (h->param->num_bframes) {
        td_rdo->pRealFD = &td_rdo->RealDList.FrameDistortionArray[td_rdo->GlobeFrameNumber];
    } else {
        td_rdo->pRealFD = &td_rdo->RealDList.FrameDistortionArray[td_rdo->globenumber];
    }
    td_rdo->pRealFD->BlockDistortionArray = (BD *)xavs2_calloc(td_rdo->pRealFD->TotalNumOfBlocks, sizeof(BD));
    if (td_rdo->GlobeFrameNumber % td_rdo->StepLength == 0) {
        if (h->fenc->i_frame == 0) {
            td_rdo->porgF.Y_base   = h->fenc->planes[IMG_Y];
            td_rdo->porgF.nStrideY = h->fenc->i_stride[IMG_Y];
            td_rdo->ppreF.Y_base   = h->img_luma_pre->planes[IMG_Y];
            td_rdo->ppreF.nStrideY = h->img_luma_pre->i_stride[IMG_Y];
            xavs2_frame_copy_planes(h, h->img_luma_pre, h->fenc);
        } else  if ((int)h->fenc->i_frame < h->param->num_frames) {
            td_rdo->pOMCPFD = &td_rdo->OMCPDList.FrameDistortionArray[td_rdo->GlobeFrameNumber - 1];
            td_rdo->pOMCPFD->BlockDistortionArray = (BD *)xavs2_calloc(td_rdo->pOMCPFD->TotalNumOfBlocks, sizeof(BD));
            td_rdo->porgF.Y_base = h->fenc->planes[IMG_Y];
            td_rdo->porgF.nStrideY = h->fenc->i_stride[IMG_Y];
            MotionDistortion(td_rdo->pOMCPFD, &td_rdo->ppreF, &td_rdo->porgF, SEARCHRANGE);
            xavs2_frame_copy_planes(h, h->img_luma_pre, h->fenc);
        }
        td_rdo->pOMCPFD = NULL;
    }

    if (td_rdo->GlobeFrameNumber % td_rdo->StepLength == 0 && td_rdo->GlobeFrameNumber < h->param->num_frames - 1) {
        CaculateKappaTableLDP(h, &td_rdo->OMCPDList, &td_rdo->RealDList, td_rdo->GlobeFrameNumber, h->i_qp);
    }
}

/* ---------------------------------------------------------------------------
 */
void tdrdo_frame_done(xavs2_t *h)
{
    FD *pDelFD;
    int DelFDNumber;
    td_rdo_t *td_rdo = h->td_rdo;
    assert(td_rdo != NULL);

    if ((h->fenc->i_frame % td_rdo->StepLength == 0 && !h->param->num_bframes) || h->param->num_bframes) {
        td_rdo->precF.Y_base = h->fdec->planes[IMG_Y];
        //td_rdo->precF.nStrideY = h->fdec->i_stride[IMG_Y];// fdec->stride[0] , bitrate rise ?
        td_rdo->precF.nStrideY = h->img_luma_pre->i_stride[IMG_Y];   //to check: fdec->stride[0] ? by lutao
        MotionDistortion(td_rdo->pRealFD, &td_rdo->porgF, &td_rdo->precF, 0);
    }
    td_rdo->pRealFD->FrameNumber = h->fenc->i_frame;
    td_rdo->globenumber++;

    DelFDNumber = td_rdo->globenumber - td_rdo->StepLength - 1;
    if (DelFDNumber >= 0) {
        pDelFD = &td_rdo->RealDList.FrameDistortionArray[DelFDNumber];
        if (pDelFD->BlockDistortionArray != NULL) {
            xavs2_free(pDelFD->BlockDistortionArray);
        }
        pDelFD->BlockDistortionArray = NULL;
    }
    if (h->fenc->i_frame % td_rdo->StepLength == 0) {
        DelFDNumber = h->fenc->i_frame / td_rdo->StepLength - 2;
        if (DelFDNumber >= 0) {
            pDelFD = &td_rdo->OMCPDList.FrameDistortionArray[DelFDNumber];
            if (pDelFD->BlockDistortionArray != NULL) {
                xavs2_free(pDelFD->BlockDistortionArray);
            }
            pDelFD->BlockDistortionArray = NULL;
            if (pDelFD->subFrameDistortionArray != NULL) {
                xavs2_free(pDelFD->subFrameDistortionArray);
            }
            pDelFD->subFrameDistortionArray = NULL;
        }
    }
}

/* ---------------------------------------------------------------------------
 */
void tdrdo_lcu_adjust_lambda(xavs2_t *h, rdcost_t *new_lambda)
{
    td_rdo_t *td_rdo = h->td_rdo;
    assert(td_rdo != NULL);

    td_rdo->CurMBQP = h->i_qp;
    if (td_rdo->GlobeFrameNumber < h->param->num_frames && h->i_type != SLICE_TYPE_I) {
        if (h->param->num_bframes && h->param->num_frames > 1 && td_rdo->GlobeFrameNumber <= ((int)((h->param->num_frames - 1) / td_rdo->StepLength))*td_rdo->StepLength) {
            td_rdo->pOMCPFD = SearchFrameDistortionArray(&td_rdo->OMCPDList, td_rdo->GlobeFrameNumber, td_rdo->StepLength, h->i_type);
        } else if (!h->param->num_bframes && h->param->num_frames > td_rdo->StepLength && td_rdo->GlobeFrameNumber % td_rdo->StepLength == 0) {
            td_rdo->pOMCPFD = &td_rdo->OMCPDList.FrameDistortionArray[(td_rdo->GlobeFrameNumber - 1) / td_rdo->StepLength];
        } else {
            td_rdo->pOMCPFD = NULL;
        }
    }

    // Just for LDP
    if (h->i_type != SLICE_TYPE_I && h->param->num_bframes == 0) {
        AdjustLcuQPLambdaLDP(h, td_rdo->pOMCPFD, h->lcu.i_scu_xy, h->i_width_in_mincu, new_lambda);
        td_rdo->CurMBQP = XAVS2_CLIP3F(MIN_QP, MAX_QP, td_rdo->CurMBQP);
    }
}

/* ---------------------------------------------------------------------------
 */
void tdrdo_lcu_update(xavs2_t *h)
{
    td_rdo_t *td_rdo = h->td_rdo;
    assert(td_rdo != NULL);

    if ((td_rdo->GlobeFrameNumber % td_rdo->StepLength == 0 && !h->param->num_bframes) || h->param->num_bframes) {
        // stores for key frame
        StoreLCUInf(td_rdo->pRealFD, h->lcu.i_scu_xy, h->param->org_width / MIN_CU_SIZE, td_rdo->CurMBQP, h->f_lambda_mode, h->i_type);
    }
}
