/*
 * defines.h
 *
 * Description of this file:
 *    const variable definition of the xavs2 library
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


#ifndef XAVS2_DEFINES_H
#define XAVS2_DEFINES_H


/**
 * ===========================================================================
 * build switch
 * ===========================================================================
 */
/* ---------------------------------------------------------------------------
 * debug */
#define XAVS2_DUMP_REC        1     /* dump reconstruction frames, 1: ON, 0: OFF */
#define XAVS2_TRACE           0     /* write trace file,    1: ON, 0: OFF */
#define XAVS2_STAT            1     /* stat encoder info,   1: On, 0: OFF */


/**
 * ===========================================================================
 * optimization
 * ===========================================================================
 */

/* 检查算法是否开启 */
#define IS_ALG_ENABLE(alg)  ((h->i_fast_algs >> alg) & 1)

/* ---------------------------------------------------------------------------
 * mask for fast algorithms
 */
enum xavs2_fast_algorithms_e {
    /* fast inter */
    OPT_EARLY_SKIP           ,        /* 基于时空相关性的快速SKIP决策 */
    OPT_PSC_MD               ,        /* 基于时空相关性的快速模式决策 (prediction size correlation based mode decision) */
    OPT_FAST_CBF_MODE        ,        /* 基于最优划分模式的CBF快速跳过剩余的划分模式 */
    OPT_FAST_PU_SEL          ,        /* OPT_FAST_CBF_MODE的简化算法，cbf=0时，若2Nx2N不优于SKIP，则跳过剩余帧间模式和帧内模式 */
    OPT_BYPASS_AMP           ,        /* 如果PRED_2NxN未获得最优，直接跳过相同划分方向的PRED_2NxnU/PRED_2NxnD; PRED_Nx2N同理 */
    OPT_DMH_CANDIDATE        ,        /* 用于精简DMH模式下的RDO次数 */
    OPT_BYPASS_MODE_FPIC     ,        /* F帧中的帧内模式与DMH模式跳过 */
    OPT_ADVANCE_CHROMA_AEC   ,        /* 提前色度块的变换系数编码过程 */
    OPT_ROUGH_MODE_SKIP      ,        /* */
    OPT_CMS_ETMD             ,        /* 条件跳过帧内划分方式：
                                       * （1）若I_2Nx2N不优于帧间预测模式，则不遍历帧内其他划分；
                                       * （2）帧间最优模式的CBP为零时跳过帧内划分方式。*/
    OPT_ROUGH_PU_SEL         ,        /* 粗略的PU划分模式搜索 */
    OPT_CBP_DIRECT           ,        /* 根据direct模式下残差是否为全零块，跳过PU划分和CU递归划分 */
    OPT_SKIP_DMH_THRES       ,        /* 通过Distortion的阈值决定跳过DMH模式的遍历 */
    OPT_ROUGH_SKIP_SEL       ,        /* 通过distortion对比只对个别skip/direct模式做RDO */

    /* fast intra */
    OPT_BYPASS_SDIP          ,        /* 如果PRED_I_2Nxn已获最优，直接跳过PRED_I_nx2N */
    OPT_FAST_INTRA_MODE      ,        /* 帧内模式快速决策 */
    OPT_FAST_RDO_INTRA_C     ,        /* 快速帧内Chroma预测模式优化，减少色度分量决策数量 */
    OPT_ET_RDO_INTRA_L       ,        /* Luma RDO过程提前退出策略 */
    OPT_ET_INTRA_DEPTH       ,        /* 基于MAD值的I帧depth划分提前终止 */
    OPT_BYPASS_INTRA_BPIC    ,        /* B帧中若帧间预测模式的CBP为零，则跳过帧内预测模式决策 */
    OPT_FAST_INTRA_IN_INTER  ,        /* 依据子CU的最优模式是否帧内及当前CU的帧间模式RDCost禁用帧间的帧内模式 */

    /* fast CU depth */
    OPT_ECU                  ,        /* HM中全零SKIP模式终止下层划分 */
    OPT_ET_HOMO_MV           ,        /* */
    OPT_CU_CSET              ,        /* CSET of uAVS2, Only for inter frames that are not referenced by others */
    OPT_CU_DEPTH_CTRL        ,        /* 基于时空相关性的Depth估计，依据上、左、左上、右上和时域参考块level调整DEPTH范围，全I帧也适用 */
    OPT_CU_QSFD              ,        /* CU splitting termination based on RD-Cost:
                                         Z. Wang, R. Wang, K. Fan, H. Sun, and W. Gao,
                                         “uAVS2—Fast encoder for the 2nd generation IEEE 1857 video coding standard,”
                                         Signal Process. Image Commun., vol. 53, no. October 2016, pp. 13–23, 2017. */

    /* fast transform and Quant */
    OPT_BYPASS_INTRA_RDOQ    ,        /* 跳过B帧帧间编码中的帧内模式的RDOQ */
    OPT_RDOQ_AZPC            ,        /* 通过对变换系数的阈值判断检测全零块进行RDOQ预处理，跳过色度分量的RDOQ过程*/

    /* others */
    OPT_FAST_ZBLOCK          ,        /* 快速零块估计 */
    OPT_TR_KEY_FRAME_MD      ,        /* 以更大概率跳过非关键帧的部分模式，能节省5%以上时间 */
    OPT_CODE_OPTIMZATION     ,        /* OPT_CU_SUBCU_COST: 先编码大CU，再编码小CU时若前几个小CU的RDCost超过大CU的一定比率则跳过后续CU
                                       * OPT_RDOQ_SKIP:     通过在RDOQ之前对变换系数的阈值判断检测全零块，跳过RDOQ过程
                                       */
    OPT_BIT_EST_PSZT         ,        /* 快速TU比特估计：对33x32的亮度TU假定只有低频的16x16部分有非零系数 */
    OPT_TU_LEVEL_DEC         ,        /* TU两层划分决策：对第一层TU划分选出最优，对最优做第二层TU划分，决策是否需要两层TU划分 */
    OPT_FAST_ALF             ,        /* ALF快速算法，在顶层B帧（不被其余帧参考）禁用ALF，在所有ALF的协方差矩阵计算时，进行step=2的下采样 */
    OPT_FAST_SAO             ,        /* SAO快速算法，在顶层B帧（不被其余帧参考）禁用SAO */
    OPT_SUBCU_SPLIT          ,        /* 根据划分子块的数目决策父块是否对非SKIP模式做RDO */
    OPT_PU_RMS               ,        /* 关闭小块（8x8,16x16)划分的预测单元，仅保留2Nx2N的帧内，帧间以及SKIP模式*/
    NUM_FAST_ALGS                     /* 总的快速算法数量 */
};


/* ---------------------------------------------------------------------------
 * const defines related with fast algorithms
 */
#define SAVE_CU_INFO            1     /* 保存参考帧队列里的每一帧的cu type和cu bitsize，用于获取时域的cu模式和cu尺寸 */
#define NUM_INTRA_C_FULL_RD     4

/* ---------------------------------------------------------------------------
 * switches for modules to be removed
 */
/* remove code for Weighted Quant */
#define ENABLE_WQUANT           0     /* 1: enable, 0: disable */

/* frame level interpolation */
#define ENABLE_FRAME_SUBPEL_INTPL         1

/* Entropy coding optimization for context update */
#define CTRL_OPT_AEC            1

/* ---------------------------------------------------------------------------
 * Rate Control
 */
#define ENABLE_RATE_CONTROL_CU  0     /* Enable Rate-Control on CU level: 1: enable, 0: disable */

#define ENABLE_AUTO_INIT_QP     1     /* 根据目标码率自动设置初始QP值 */


/**
 * ===========================================================================
 * const defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * const for bool type
 */
#ifndef FALSE
#define FALSE                   0
#endif
#ifndef TRUE
#define TRUE                    1
#endif


/* ---------------------------------------------------------------------------
 * profiles
 */
#define MAIN_PICTURE_PROFILE    0x12  /* profile: MAIN_PICTURE */
#define MAIN_PROFILE            0x20  /* profile: MAIN */
#define MAIN10_PROFILE          0x22  /* profile: MAIN10 */


/* ---------------------------------------------------------------------------
* chroma formats
*/
#define CHROMA_400              0
#define CHROMA_420              1
#define CHROMA_422              2
#define CHROMA_444              3

#define CHROMA_V_SHIFT          (h->i_chroma_v_shift)

/* ---------------------------------------------------------------------------
 * quantization parameter range
 */
#define MIN_QP                  0     /* min QP */
#define MAX_QP                  63    /* max QP */
#define SHIFT_QP                11    /* shift QP */


/* ---------------------------------------------------------------------------
 * cu size
 */
#define MAX_CU_SIZE             64    /* max CU size */
#define MAX_CU_SIZE_IN_BIT      6
#define MIN_CU_SIZE             8     /* min CU size */
#define MIN_CU_SIZE_IN_BIT      3
#define MIN_PU_SIZE             4     /* min PU size */
#define MIN_PU_SIZE_IN_BIT      2
#define BLOCK_MULTIPLE          (MIN_CU_SIZE / MIN_PU_SIZE)
#define CTU_DEPTH               (MAX_CU_SIZE_IN_BIT - MIN_CU_SIZE_IN_BIT + 1)

#define B4X4_IN_BIT             2     /* unit level: 2 */
#define B8X8_IN_BIT             3     /* unit level: 3 */
#define B16X16_IN_BIT           4     /* unit level: 4 */
#define B32X32_IN_BIT           5     /* unit level: 5 */
#define B64X64_IN_BIT           6     /* unit level: 6 */


/* ---------------------------------------------------------------------------
 * parameters for scale mv
 */
#define MULTIx2                 32768
#define MULTI                   16384
#define HALF_MULTI              8192
#define OFFSET                  14


/* ---------------------------------------------------------------------------
 * prediction techniques
 */
#define LAM_2Level_TU           0.8
#define DMH_MODE_NUM            5     /* number of DMH mode */
#define WPM_NUM                 3     /* number of WPM */
#define TH_PMVR                 2     /* PMVR中四分之一像素精度MV的可用范围 */


/* ---------------------------------------------------------------------------
 * coefficient coding
 */
#define MAX_TU_SIZE             32    /* 最大变换块大小，熵编码时的系数矩阵 */
#define MAX_TU_SIZE_IN_BIT      5     /* 最大变换块大小，熵编码时的系数矩阵 */
#define SIZE_CG                 4     /* CG 大小 4x4 */
#define SIZE_CG_IN_BIT          2     /* CG 大小 4x4 */
#define MAX_CG_NUM_IN_TU        (1 << ((MAX_TU_SIZE_IN_BIT - SIZE_CG_IN_BIT) << 1))

/* ---------------------------------------------------------------------------
 * temporal level (layer)
 */
#define TEMPORAL_MAXLEVEL       8     /* max number of temporal levels */
#define TEMPORAL_MAXLEVEL_BIT   3     /* bits of temporal level */



/* ---------------------------------------------------------------------------
 * SAO (Sample Adaptive Offset)
 */
#define NUM_BO_OFFSET                 32                            /*BO模式下offset数量，其中最多4个非零*/
#define MAX_NUM_SAO_CLASSES           32                            /*最大offset数量*/
#define NUM_SAO_BO_CLASSES_LOG2       5                             /**/
#define NUM_SAO_BO_CLASSES_IN_BIT     5                             /**/
#define NUM_SAO_BO_CLASSES           (1 << NUM_SAO_BO_CLASSES_LOG2) /*BO模式下startband数目*/
#define SAO_RATE_THR                  1.0                          /*亮度分量，用于RDO决策*/
#define SAO_RATE_CHROMA_THR           1.0                          /*色度分量，用于RDO决策*/
#define SAO_SHIFT_PIX_NUM             4                             /*SAO向左上偏移的像素点数*/


#define MAX_DOUBLE              1.7e+308



/* ---------------------------------------------------------------------------
 * ALF (Adaptive Loop Filter)
 */
#define ALF_MAX_NUM_COEF        9
#define NO_VAR_BINS             16
#define LOG2_VAR_SIZE_H         2
#define LOG2_VAR_SIZE_W         2
#define ALF_FOOTPRINT_SIZE      7
#define DF_CHANGED_SIZE         3
#define ALF_NUM_BIT_SHIFT       6

#define LAMBDA_SCALE_LUMA   (1.0)     /* scale for luma */
#define LAMBDA_SCALE_CHROMA (1.0)     /* scale for chroma */



/* ---------------------------------------------------------------------------
 * threshold values to zero out quantized transform coefficients
 */
#define LUMA_COEFF_COST         1     /* threshold for luma coefficients */
#define MAX_COEFF_QUASI_ZERO    8     /* threshold for quasi zero block detection with luma coefficients */


/* ---------------------------------------------------------------------------
 * number of luma intra modes for full RDO
 */
#define INTRA_MODE_NUM_FOR_RDO  9     /* number of luma intra modes for full RDO */

/* ---------------------------------------------------------------------------
 * max values
 */
#define MAX_DISTORTION     (1 << 30)  /* maximum distortion (1 << bitdepth)^2 * (MAX_CU_SIZE)^2 */
#define XAVS2_THREAD_MAX        128   /* max number of threads */
#define XAVS2_BS_HEAD_LEN       256   /* length of bitstream buffer for headers */
#define XAVS2_PAD          (64 + 16)  /* number of pixels padded around the reference frame */
#define MAX_COST         (1LL << 50)  /* used for start value for cost variables */
#define MAX_FRAME_INDEX  0x3FFFFF00   /* max frame index */
#define MAX_REFS     XAVS2_MAX_REFS   /* max number of reference frames */
#define MAX_SLICES                8   /* max number of slices in one picture */
#define MAX_PARALLEL_FRAMES       8   /* max number of parallel encoding frames */
#define MAX_COI_VALUE   ((1<<8) - 1)  /* max COI value (unsigned char) */
#define PIXEL_MAX ((1<<BIT_DEPTH)-1)  /* max value of a pixel */


/* ---------------------------------------------------------------------------
 * reference picture management
 */
#define XAVS2_INPUT_NUM      (4 * MAX_PARALLEL_FRAMES + 4)    /* number of buffered input frames */
#define FREF_BUF_SIZE (MAX_REFS + MAX_PARALLEL_FRAMES * 4)    /* number of reference + decoding frames to buffer */


/* ---------------------------------------------------------------------------
 * reserved memory space for check pseudo code */
#define PSEUDO_CODE_SIZE        1024  /* size of reserved memory space */

/* ---------------------------------------------------------------------------
 * transform
 */
#define SEC_TR_SIZE             4
#define SEC_TR_MIN_BITSIZE      3     /* apply secT to greater than or equal to 8x8 block */

#define LIMIT_BIT               16
#define FACTO_BIT               5


/* ---------------------------------------------------------------------------
 * frame list type
 */
enum frame_alloc_type_e {
    FT_ENC              =  0,       /* encoding frame */
    FT_DEC              =  1,       /* decoding frame */
    FT_TEMP             =  2,       /* temporary frame for SAO/ALF/TDRDO decision or other modules */
};

/* ---------------------------------------------------------------------------
 * variable section delimiter
 */
#define SYNC_VARS_1(delimiter)  int delimiter
#define SYNC_VARS_2(delimiter)  int delimiter


/* ---------------------------------------------------------------------------
 * all assembly and related C functions are prefixed with 'xavs2_' default
 */
#define PFXB(prefix, name)  prefix ## _ ## name
#define PFXA(prefix, name)  PFXB(prefix, name)
#define FPFX(name)          PFXA(xavs2,  name)


/* ---------------------------------------------------------------------------
 * flag
 */
#define XAVS2_EXIT_THREAD     (-1)  /* flag to terminate thread */



/* ---------------------------------------------------------------------------
 * others
 */
/* reference management */
#define XAVS2_MAX_REFS         4     /* max number of reference frames */
#define XAVS2_MAX_GOPS        16     /* max number of GOPs */
#define XAVS2_MAX_GOP_SIZE    16     /* max length of GOP */

/* adapt layer */
#define XAVS2_ADAPT_LAYER      1     /* output adapt layer? */
#define XAVS2_MAX_NAL_NUM     16     /* max number of NAL in bitstream of one frame */

/* weight quant */
#define WQMODEL_PARAM_SIZE    64     /* size of weight quant model param */

/* qp */
#define XAVS2_QP_AUTO          0     /* get qp automatically */

#endif /* #if XAVS2_DEFINES_H */
