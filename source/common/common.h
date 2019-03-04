/*
 * common.h
 *
 * Description of this file:
 *    misc common functionsdefinition of the xavs2 library
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


#ifndef XAVS2_COMMON_H
#define XAVS2_COMMON_H


/**
 * ===========================================================================
 * common include files
 * ===========================================================================
 */
#include "defines.h"
#include "osdep.h"
#include "avs2_defs.h"
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <limits.h>


/**
 * ===========================================================================
 * macros
 * ===========================================================================
 */


/* ---------------------------------------------------------------------------
 * predicate mode & cu type
 */
#define ALLOW_HOR_TU_PART(mode) (((1 << (mode)) & MASK_HOR_TU_MODES) != 0)
#define ALLOW_VER_TU_PART(mode) (((1 << (mode)) & MASK_VER_TU_MODES) != 0)
#define IS_HOR_PU_PART(mode)    (((1 << (mode)) & MASK_HOR_PU_MODES) != 0)
#define IS_VER_PU_PART(mode)    (((1 << (mode)) & MASK_VER_PU_MODES) != 0)
#define IS_INTRA_MODE(mode)     (((1 << (mode)) & MASK_INTRA_MODES ) != 0)
#define IS_INTER_MODE(mode)     (((1 << (mode)) & MASK_INTER_MODES ) != 0)
#define IS_INTER_MODE_NS(mode)  (((1 << (mode)) & MASK_INTER_NOSKIP) != 0) /* is inter mode (except SKIP)? */
#define IS_SKIP_MODE(mode)      ((mode) == PRED_SKIP)

#define IS_INTRA(cu)            IS_INTRA_MODE((cu)->i_mode)
#define IS_INTER(cu)            IS_INTER_MODE((cu)->i_mode)
#define IS_SKIP(cu)             IS_SKIP_MODE((cu)->i_mode)


/* ---------------------------------------------------------------------------
 * weight cost of mvd/ref
 */
#define LAMBDA_ACCURACY_BITS    16
#define LAMBDA_FACTOR(lambda)   ((int)((double)(1<<LAMBDA_ACCURACY_BITS)*lambda+0.5))
#define WEIGHTED_COST(f, bits)  (((f)*(bits))>>LAMBDA_ACCURACY_BITS)


/* ---------------------------------------------------------------------------
 * multi line macros
 */
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#define MULTI_LINE_MACRO_BEGIN  do {
#define MULTI_LINE_MACRO_END \
    __pragma(warning(push))\
    __pragma(warning(disable:4127))\
    } while (0)\
    __pragma(warning(pop))
#else
#define MULTI_LINE_MACRO_BEGIN {
#define MULTI_LINE_MACRO_END   }
#endif


/* ---------------------------------------------------------------------------
 * memory malloc
 */
#define CHECKED_MALLOC(var, type, size) \
    MULTI_LINE_MACRO_BEGIN\
    (var) = (type)xavs2_malloc(size);\
    if ((var) == NULL) {\
        goto fail;\
    }\
    MULTI_LINE_MACRO_END

#define CHECKED_MALLOCZERO(var, type, size) \
    MULTI_LINE_MACRO_BEGIN\
    size_t new_size = ((size + 31) >> 5) << 5; /* align the size to 32 bytes */ \
    CHECKED_MALLOC(var, type, new_size);\
    g_funcs.memzero_aligned(var, new_size); \
    MULTI_LINE_MACRO_END


/**
 * ===========================================================================
 * enum defines
 * ===========================================================================
 */


/* ---------------------------------------------------------------------------
 * rate control methods
 */
enum rc_method_e {
    XAVS2_RC_CQP = 0,           /* const QP */
    XAVS2_RC_CBR_FRM = 1,       /* const bit-rate, frame level */
    XAVS2_RC_CBR_SCU = 2        /* const bit-rate, SCU   level */
};


/* ---------------------------------------------------------------------------
 * ME methods
 */
enum me_mothod_e {
    XAVS2_ME_FS   = 0,        /* full search */
    XAVS2_ME_DIA  = 1,        /* diamond search */
    XAVS2_ME_HEX  = 2,        /* hexagon search */
    XAVS2_ME_UMH  = 3,        /* UMH search */
    XAVS2_ME_TZ   = 4         /* TZ search */
};


/* ---------------------------------------------------------------------------
 * slice types
 */
enum slice_type_e {
    SLICE_TYPE_I    = 0,        /* slice type: I */
    SLICE_TYPE_P    = 1,        /* slice type: P */
    SLICE_TYPE_B    = 2,        /* slice type: B */
    SLICE_TYPE_F    = 3,        /* slice type: F */
    SLICE_TYPE_NUM  = 4         /* slice type number */
};


/* ---------------------------------------------------------------------------
 * NAL unit type
 */
enum nal_unit_type_e {
    NAL_UNKNOWN     = 0,
    NAL_SLICE       = 1,
    NAL_SLICE_DPA   = 2,
    NAL_SLICE_DPB   = 3,
    NAL_SLICE_DPC   = 4,
    NAL_SLICE_IDR   = 5,        /* ref_idc != 0 */
    NAL_SEI         = 6,        /* ref_idc == 0 */
    NAL_SPS         = 7,
    NAL_PPS         = 8,
    NAL_AUD         = 9,
    NAL_FILLER      = 12
};


/* ---------------------------------------------------------------------------
 * NAL priority
 */
enum nal_priority_e {
    NAL_PRIORITY_DISPOSABLE = 0,
    NAL_PRIORITY_LOW        = 1,
    NAL_PRIORITY_HIGH       = 2,
    NAL_PRIORITY_HIGHEST    = 3
};


/* ---------------------------------------------------------------------------
 * all prediction modes (n = N/2)
 */
enum cu_pred_mode_e {
    PRED_INVLALID     = -1,     /* invalid mode, as initial value   */
    /* all inter modes: 8                                           */
    PRED_SKIP         = 0,      /*  skip/direct           block: 1  */
    PRED_2Nx2N        = 1,      /*  2N x 2N               block: 1  */
    PRED_2NxN         = 2,      /*  2N x  N               block: 2  */
    PRED_Nx2N         = 3,      /*   N x 2N               block: 2  */
    PRED_2NxnU        = 4,      /*  2N x  n  +  2N x 3n   block: 2  */
    PRED_2NxnD        = 5,      /*  2N x 3n  +  2N x  n   block: 2  */
    PRED_nLx2N        = 6,      /*   n x 2N  +  3n x 2N   block: 2  */
    PRED_nRx2N        = 7,      /*  3n x 2N  +   n x 2N   block: 2  */
    /* all intra modes: 4                                           */
    PRED_I_2Nx2N      = 8,      /*  2N x 2N               block: 1  */
    PRED_I_NxN        = 9,      /*   N x  N               block: 4  */
    PRED_I_2Nxn       = 10,     /*  2N x  n  (32x8, 16x4) block: 4  */
    PRED_I_nx2N       = 11,     /*   n x 2N  (8x32, 4x16) block: 4  */
    /* mode numbers                                                 */
    MAX_PRED_MODES    = 12,     /* total 12 pred modes, include:    */
    MAX_INTER_MODES   = 8,      /*       8 inter modes              */
    MAX_INTRA_MODES   = 4,      /*       4 intra modes              */
    /* masks                                                        */
    MASK_HOR_TU_MODES = 0x0430, /* mask for horizontal TU partition */
    MASK_VER_TU_MODES = 0x08C0, /* mask for vertical   TU partition */
    MASK_HOR_PU_MODES = 0x0434, /* mask for horizontal PU partition */
    MASK_VER_PU_MODES = 0x08C8, /* mask for vertical   PU partition */
    MASK_INTER_MODES  = 0x00FF, /* mask for inter modes             */
    MASK_INTER_NOSKIP = 0x00FE, /* mask for inter modes except skip */
    MASK_INTRA_MODES  = 0x0F00  /* mask for intra modes             */
};


/* ---------------------------------------------------------------------------
 * splitting type of transform unit
 */
enum tu_split_type_e {
    TU_SPLIT_INVALID  = -1,     /*      invalid split type          */
    TU_SPLIT_NON      = 0,      /*          not split               */
    TU_SPLIT_HOR      = 1,      /* horizontally split into 4 blocks */
    TU_SPLIT_VER      = 2,      /*   vertically split into 4 blocks */
    TU_SPLIT_CROSS    = 3,      /*        cross split into 4 blocks */
    TU_SPLIT_TYPE_NUM = 4       /*    number of split types         */
};


/* ---------------------------------------------------------------------------
 * image components
 */
enum image_component_type_e {
    IMG_Y       = 0,            /* image component: Y */
    IMG_U       = 1,            /* image component: Cb */
    IMG_V       = 2,            /* image component: Cr */
    IMG_CMPNTS  = 3             /* image component number */

};


/* ---------------------------------------------------------------------------
 */
enum coding_type_e {
    FRAME_CODING    = 0,
    FIELD_CODING    = 3
};


/* ---------------------------------------------------------------------------
 */
enum sequence_type_e {
    FIELD,
    FRAME
};

/* ---------------------------------------------------------------------------
 * task type
 */
typedef enum task_type_e {
    XAVS2_TASK_FRAME  = 0,          /* frame task */
    XAVS2_TASK_SLICE  = 1,          /* slice task */
    XAVS2_TASK_ROW    = 2           /* row task */
} task_type_e;


/* ---------------------------------------------------------------------------
 * task status
 */
typedef enum task_status_e {
    XAVS2_TASK_FREE       = 0,      /* task is free */
    XAVS2_TASK_BUSY       = 1,      /* task is alloted */
    XAVS2_TASK_RDO_DONE   = 2,      /* RDO is finished */
    XAVS2_TASK_AEC_DONE   = 3       /* AEC is finished */
} task_status_e;


/* ---------------------------------------------------------------------------
 * signals
 */
enum xavs2_signal_e {
    SIG_FRM_CONTEXT_ALLOCATED = 0,    /* one frame context is allocated */
    SIG_FRM_CONTEXT_RELEASED  = 1,    /* one frame context is released */
    SIG_FRM_AEC_COMPLETED     = 2,    /* one frame finishes AEC */
    SIG_FRM_AEC_DONE          = 3,    /* one frame finishes AEC */
    SIG_FRM_DELIVERED         = 4,    /* one frame is outputted */
    SIG_FRM_BUFFER_RELEASED   = 5,    /* one frame buffer is available */
    SIG_ROW_CONTEXT_RELEASED  = 6,    /* one row context is released */
    SIG_COUNT                 = 7
};


/**
 * ===========================================================================
 * type defines
 * ===========================================================================
 */
#include "basic_types.h"

/* ---------------------------------------------------------------------------
 * function handle types
 */

/* thread function: proceeding of one thread */
typedef void *(*xavs2_tfunc_t)(void *);
// typedef void *(__stdcall *xavs2_afunc_t)(void *);

/**
 * ===========================================================================
 * internal include files
 * ===========================================================================
 */
#include "xavs2.h"
#include "pixel.h"
#include "intra.h"
#include "transform.h"
#include "filter.h"

#if HAVE_MMX
#include "vec/intrinsic.h"
#endif
#include "primitives.h"




/**
 * ===========================================================================
 * struct type defines
 * ===========================================================================
 */

#if defined(_MSC_VER) || defined(__ICL)
#pragma warning(disable: 4201)        // non-standard extension used (nameless struct/union)
#endif


/* ---------------------------------------------------------------------------
 * motion vector
 */
union mv_t {
    uint32_t    v;                    // v = ((y << 16) | (x & 0xFFFF)), 32-bit
    struct {
        int16_t x;                    // x, low  16-bit
        int16_t y;                    // y, high 16-bit
    };
};


/* ---------------------------------------------------------------------------
 * bitstream
 */
typedef struct bitstream {
    uint8_t    *p_start;              /* actual buffer for written bytes */
    uint8_t    *p;                    /* pointer to byte written currently */
    uint8_t    *p_end;                /* end of the actual buffer */
    int         i_left;               /* current bit counter to go */
} bs_t;


/* ---------------------------------------------------------------------------
 * struct for context management
 */
typedef union context_t {
    struct {
        unsigned    MPS     : 1;      // 1  bit
        unsigned    LG_PMPS : 11;     // 11 bits
        unsigned    cycno   : 2;      // 2  bits
    };
    uint16_t        v;
} context_t;

typedef union runlevel_pair_t {
    struct {
        coeff_t level;
        int8_t  run;
    };
    uint32_t    v;
} runlevel_pair_t;

/* ---------------------------------------------------------------------------
 * run-level infos (CG: Coefficient Group)
 * 熵编码过程中最大的变换块为 32x32，最多 8*8 个CG
 */
typedef struct runlevel_t {
    ALIGN16(runlevel_pair_t runlevels_cg[16]);
    int             num_cg;
    int             last_pos_cg;                         /* Last Coeff Position in CG */
    int             b_hor;
    int             i_stride_shift;
    coeff_t        *quant_coeff;                         /* coefficients */
    coeff_t        *transposed_coeff;                    /* coefficients in CG scan order */
    const int16_t(*tab_cg_scan)[2];                      /* CG scan table */
    cu_info_t      *p_cu_info;
} runlevel_t;


/* ---------------------------------------------------------------------------
 * binary_t
 */
typedef struct binary_t {
    /* 语法元素编码用函数指针 */
    int (*write_intra_pred_mode)(aec_t *p_aec, int ipmode);
    int (*write_ctu_split_flag)(aec_t *p_aec, int i_cu_split, int i_cu_level);
    int (*est_cu_header)(xavs2_t *h, aec_t *p_aec, cu_t *p_cu);
    int (*est_cu_refs_mvds)(xavs2_t *h, aec_t *p_aec, cu_t *p_cu);

    int (*est_luma_block_coeff)(xavs2_t *h, aec_t *p_aec, cu_t *p_cu, coeff_t *quant_coeff, runlevel_t *runlevel,
                                int i_level, int i_stride_shift, int is_intra, int intra_mode, int max_bits);
    int (*est_chroma_block_coeff)(xavs2_t *h, aec_t *p_aec, cu_t *p_cu, coeff_t *quant_coeff, runlevel_t *runlevel,
                                  int i_level, int max_bits);

#if ENABLE_RATE_CONTROL_CU
    int (*write_cu_cbp_dqp)(xavs2_t *h, aec_t *p_aec, cu_info_t *p_cu_info, int slice_index_cur_cu, int *last_dqp);
#else
    int (*write_cu_cbp)(aec_t *p_aec, cu_info_t *p_cu_info, int slice_index_cur_cu, xavs2_t *h);
#endif

    int  (*write_sao_mergeflag)(aec_t *p_aec, int mergeleft_avail, int mergeup_avail, SAOBlkParam *saoBlkParam);
    int  (*write_sao_mode)(aec_t *p_aec, SAOBlkParam *saoBlkParam);
    int  (*write_sao_offset)(aec_t *p_aec, SAOBlkParam *saoBlkParam);
    int  (*write_sao_type)(aec_t *p_aec, SAOBlkParam *saoBlkParam);

    int  (*write_alf_lcu_ctrl)(aec_t *p_aec, uint8_t iflag);
} binary_t;


/* ---------------------------------------------------------------------------
 * const for syntax elements
 */
#define NUM_BLOCK_TYPES         3

#define NUM_CU_TYPE_CTX         5
#define NUM_INTRA_PU_TYPE_CTX   1
#define NUM_INTRA_MODE_CTX      7
#define NUM_INTRA_MODE_C_CTX    3

#define NUM_SPLIT_CTX           (CTU_DEPTH - 1)     // CU depth
#define NUM_TU_CTX              3

#define NUM_INTER_DIR_CTX       15
#define NUM_INTER_DIR_MIN_CTX   2

#define NUM_AMP_CTX             2
#define NUM_CBP_CTX             9
#define NUM_MVD_CTX             3
#define NUM_DMH_MODE_CTX        12    // (MAX_CU_SIZE_IN_BIT - MIN_CU_SIZE_IN_BIT + 1) * 4
#define NUM_REF_NO_CTX          3
#define NUM_DELTA_QP_CTX        4

#define NUM_LAST_CG_CTX_LUMA    6
#define NUM_LAST_CG_CTX_CHROMA  6
#define NUM_SIGN_CG_CTX_LUMA    2
#define NUM_SIGN_CG_CTX_CHROMA  1
#define NUM_LAST_POS_CTX_LUMA   48    /* last_coeff_pos_x 和 last_coeff_pos_y 共计有48个色度分量上下文 */
#define NUM_LAST_POS_CTX_CHROMA 12    /* last_coeff_pos_x 和 last_coeff_pos_y 共计有12个色度分量上下文 */

#define NUM_MAP_CTX             12
#define NUM_LAST_CG_CTX         (NUM_LAST_CG_CTX_LUMA  + NUM_LAST_CG_CTX_CHROMA)   /* last_cg_pos:6; + last_cg0_flag:2(IsChroma); last_cg_x:2; last_cg_y:2 */
#define NUM_SIGN_CG_CTX         (NUM_SIGN_CG_CTX_LUMA  + NUM_SIGN_CG_CTX_CHROMA)
#define NUM_LAST_POS_CTX        (NUM_LAST_POS_CTX_LUMA + NUM_LAST_POS_CTX_CHROMA)  /* last_coeff_pos_x: (30) + last_coeff_pos_y: (30) */
#define NUM_COEFF_LEVEL_CTX     40    /* CoeffLevelMinus1Band 为 0 时 coeff_level_minus1_pos_in_band */

#define NUM_SAO_MERGE_FLAG_CTX  3
#define NUM_SAO_MODE_CTX        1
#define NUM_SAO_OFFSET_CTX      1
#define NUM_ALF_LCU_CTX         4


/* ---------------------------------------------------------------------------
 * reference parameter set
 */
typedef struct xavs2_rps_t {
    int     idx_in_gop;               /* index within a GOP */
    int     poc;                      /* picture order count */
    int     qp_offset;                /* QP offset based on key frame */
    int     referd_by_others;         /* referenced by other pictures? */

    int     temporal_id;              /* temporal id */
    int     reserved;                 /* reserved (not used) */

    int     num_of_ref;               /* number of reference pictures */
    int     num_to_rm;                /* number of picture to be removed */
    int     ref_pic[XAVS2_MAX_REFS];/* delta COI of reference pictures */
    int     rm_pic [8];               /* delta COI of removed pictures */
} xavs2_rps_t;



/* ---------------------------------------------------------------------------
 * xavs2 encoder input parameters
 */
typedef struct xavs2_param_t {
    /* --- sequence --------------------------------------------- */
    int     profile_id;               /* profile id */
    int     level_id;                 /* level id */
    int     progressive_sequence;     /* progressive sequence? */
    int     chroma_format;            /* YUV format (0=4:0:0, 1=4:2:0, 2=4:2:2, 3=4:4:4,currently only 4:2:0 is supported) */
    int     input_sample_bit_depth;   /* input file bit depth */
    int     sample_bit_depth;         /* sample bit depth */
    int     sample_precision;         /* sample precision */
    int     aspect_ratio_information; /* aspect ratio information */
    int     frame_rate_code;          /* frame rate code */
    float   frame_rate;               /* frame rate */
    int     bitrate_lower;            /* bit rate lower */
    int     bitrate_upper;            /* bit rate upper */
    int     low_delay;                /* low delay */
    int     temporal_id_exist_flag;   /* temporal_id exist flag */
    int     bbv_buffer_size;          /* bbv buffer size */
    int     lcu_bit_level;            /* largest  coding unit size in bit, 3:8x8, 4:16x16, 5:32x32, 6:64x64 */
    int     scu_bit_level;            /* smallest coding unit size in bit, 3:8x8, 4:16x16, 5:32x32, 6:64x64 */
    int     org_width;                /* original source image width */
    int     org_height;               /* original source image height */

    // sequence display extension
    // int     video_format;             /* video format */
    // int     video_range;              /* video range */
    // int     color_description;        /* color description */
    // int     color_primaries;          /* color primaries */
    // int     transfer_characteristics; /* transfer characteristics */
    // int     matrix_coefficients;      /* matrix coefficients */
    // int     display_horizontal_size;  /* display horizontal size */
    // int     display_vertical_size;    /* display vertical size */
    // int     TD_mode;                  /* 3D mode */
    // int     view_packing_mode;        /* 3D packing mode */
    // int     view_reverse;             /* view reverse */

    /* --- stream structure ------------------------------------- */
    int     intra_period_max;         /* maximum intra-period, one I-frame mush appear in any NumMax of frames */
    int     intra_period_min;         /* minimum intra-period, only one I-frame can appear in at most NumMin of frames */
    int     b_open_gop;               /* open GOP? 1: open, 0: close */
    int     enable_f_frame;           /* enable F-frame */
    int     num_bframes;              /* number of B frames that will be used */
    int     InterlaceCodingOption;    /* coding type: frame coding? field coding? */

    /* --- picture ---------------------------------------------- */
    int     progressive_frame;        /* progressive frame */
    int     time_code_flag;           /* time code flag */
    int     top_field_first;          /* top field first */
    int     repeat_first_field;       /* repeat first field */
    int     fixed_picture_qp;         /* fixed picture qp */

    /* --- slice ------------------------------------------------ */
    int     slice_num;                /* slice number */

    /* --- analysis options ------------------------------------- */
    int     enable_hadamard;          /* 0: 'normal' SAD in 1/4 pixel search.  1: use 4x4 Haphazard transform and
                                       * Sum of absolute transform difference' in 1/4 pixel search */
    int     me_method;                /* Fast motion estimation method. 1: DIA, 2: HEX 3: UMH */
    int     search_range;             /* search range - integer pel search and 16x16 blocks.  The search window is
                                       * generally around the predicted vector. Max vector is 2xmcrange.  For 8x8
                                       * and 4x4 block sizes the search range is 1/2 of that for 16x16 blocks. */
    int     num_max_ref;              /* 1: prediction from the last frame only. 2: prediction from the last or
                                       * second last frame etc.  Maximum 5 frames (number of reference frames) */
    int     inter_2pu;                /* enable inter 2NxN or Nx2N or AMP mode */
    int     enable_amp;               /* enable Asymmetric Motion Partitions */
    int     enable_intra;             /* enable intra mode for inter frame */
    int     rdo_bit_est_method;       /* RDO bit estimation method:
                                       * 0: AEC with context updating; 1: AEC without context update
                                       * 2: VLC */
    int     preset_level;             /* preset level */
    int     is_preset_configured;     /* whether preset configuration is utilized */

    /* encoding tools ------------------------------------------- */
    int     enable_mhp_skip;          /* enable MHP-skip */
    int     enable_dhp;               /* enabled DHP */
    int     enable_wsm;               /* enable Weight Skip Mode */
    int     enable_nsqt;              /* use NSQT or not */
    int     enable_sdip;              /* use SDIP or not */
    int     enable_secT;              /* secT enabled */
    int     enable_sao;               /* SAO enable flag */
    int     enable_alf;               /* ALF enable flag */
    int     alf_LowLatencyEncoding;   /* ALF low latency encoding enable flag */
    int     enable_pmvr;              /* pmvr enabled */
    int     b_cross_slice_loop_filter;   /* cross loop filter flag */
    int     enable_dmh;               /* DMH mode enable, (always true) */
    int     i_rd_level;               /* RDO level,
                                       * 0: off,
                                       * 1: only for best partition mode of one CU,
                                       * 2: only for best 2 partition modes;
                                       * 3: All partition modes */
    bool_t  b_sao_before_deblock;     /* conduct SAO parameter decision before deblock totally finish */
    bool_t  b_fast_sao;               /* Fast SAO encoding decision */
    bool_t  b_fast_2lelvel_tu;        /* enable fast 2-level TU for inter */
    float   factor_zero_block;        /* threadhold factor for zero block detection */

    /* RDOQ */
    int     i_rdoq_level;             /* RDOQ level,
                                       * 0: off,
                                       * 1: only for best partition mode of one CU,
                                       * 2: for all modes */
    int     lambda_factor_rdoq;       /* */
    int     lambda_factor_rdoq_p;     /* */
    int     lambda_factor_rdoq_b;     /* */

    int     enable_refine_qp;         /* refine QP? */
    int     enable_tdrdo;             /* enable TDRDO? */

    /* loop filter */
    int     loop_filter_disable;      /* loop filter disable */
    int     loop_filter_parameter_flag; /* loop filter parameter flag */
    int     alpha_c_offset;           /* alpha offset */
    int     beta_offset;              /* beta offset */

    /* weight quant */
    int     enable_wquant;            /* enable weight quant */
#if ENABLE_WQUANT
    int     SeqWQM;                   /* load seq weight quant data flag */
    int     PicWQEnable;              /* weighting quant_flag */
    int     PicWQDataIndex;           /* Picture level WQ data index */
    char    WeightParamDetailed[WQMODEL_PARAM_SIZE];
    char    WeightParamUnDetailed[WQMODEL_PARAM_SIZE];
    int     MBAdaptQuant;
    int     WQParam;                  /* weight quant param index */
    int     WQModel;                  /* weight quant model */
#endif
    int     chroma_quant_param_disable; /* chroma quant param disable */
    int     chroma_quant_param_delta_u; /* chroma quant param delta cb */
    int     chroma_quant_param_delta_v; /* chroma quant param delta cr */

    /* --- rate control ----------------------------------------- */
    int     i_rc_method;              /* rate control method: 0: CQP, 1: CBR (frame level), 2: CBR (SCU level), 3: VBR */
    int     i_target_bitrate;         /* target bitrate (bps) */
    int     i_initial_qp;             /* initial QP */
    int     i_min_qp;                 /* min QP */
    int     i_max_qp;                 /* max QP */

    /* --- parallel --------------------------------------------- */
    int     num_parallel_gop;         /* number of parallel GOP */
    int     i_frame_threads;          /* number of thread in frame   level parallel */
    int     i_lcurow_threads;         /* number of thread in LCU-row level parallel */
    int     enable_aec_thread;        /* enable AEC threadpool or not */

    /* --- log -------------------------------------------------- */
    int     i_log_level;              /* log level */
    int     enable_psnr;              /* enable PSNR calculation or not */
    int     enable_ssim;              /* enable SSIM calculation or not */

    /* --- reference management --------------------------------- */
    int     i_gop_size;               /* sub GOP size */
    xavs2_rps_t cfg_ref_all[XAVS2_MAX_GOPS];  /* ref_man array */

    /* --- input/output for testing ----------------------------- */
    int     infile_header;            /* if input file has a header set this to the length of the header */
    int     output_merged_picture;
    int     num_frames;               /* number of frames to be encoded */

#define FN_LEN  128
    char    psz_in_file[FN_LEN];      /* YUV 4:2:0 input format */
    char    psz_bs_file[FN_LEN];      /* AVS compressed output bitstream */
    char    psz_dump_yuv[FN_LEN];     /* filename for reconstructed frames */
#if XAVS2_TRACE
    char    psz_trace_file[FN_LEN];   /* filename for trace information */
#endif
#if ENABLE_WQUANT
    char    psz_seq_wq_file[FN_LEN];
    char    psz_pic_wq_file[FN_LEN];
#endif
} xavs2_param_t;

/* ---------------------------------------------------------------------------
 * syntax element set
 */
typedef struct ctx_set_t {
    ALIGN16(context_t cu_type_contexts      [NUM_CU_TYPE_CTX       ]);
    context_t intra_pu_type_contexts        [NUM_INTRA_PU_TYPE_CTX ];
    context_t split_flag                    [NUM_SPLIT_CTX         ];
    context_t transform_split_flag          [NUM_TU_CTX            ];
    context_t shape_of_partition_index      [NUM_AMP_CTX           ];
    context_t pu_reference_index            [NUM_REF_NO_CTX        ];
    context_t cbp_contexts                  [NUM_CBP_CTX           ];
    context_t mvd_contexts               [2][NUM_MVD_CTX           ];
    /* 帧间预测 */
    context_t pu_type_index                 [NUM_INTER_DIR_CTX     ];    // b_pu_type_index[15] = f_pu_type_index[3] + dir_multi_hypothesis_mode[12]
    context_t b_pu_type_min_index           [NUM_INTER_DIR_MIN_CTX ];
    // b_pu_type_index2 // for B_NxN
    // f_pu_type_index2 // for F_NxN
    context_t cu_subtype_index              [DS_MAX_NUM            ];  // B_Skip/B_Direct, F_Skip/F_Direct 公用
    context_t weighted_skip_mode            [WPM_NUM               ];
    /* 帧内预测 */
    context_t intra_luma_pred_mode          [NUM_INTRA_MODE_CTX    ];
    context_t intra_chroma_pred_mode        [NUM_INTRA_MODE_C_CTX  ];
    /* CU 级别QP调整 */
#if ENABLE_RATE_CONTROL_CU
    context_t delta_qp_contexts             [NUM_DELTA_QP_CTX      ];
#endif
    /* 变换系数编码 */
    context_t coeff_run [2][NUM_BLOCK_TYPES][NUM_MAP_CTX           ];  // [0:Luma, 1:Chroma][rank][ctx_idx]
    context_t nonzero_cg_flag               [NUM_SIGN_CG_CTX       ];
    context_t last_cg_contexts              [NUM_LAST_CG_CTX       ];
    context_t last_pos_contexts             [NUM_LAST_POS_CTX      ];
    context_t coeff_level                   [NUM_COEFF_LEVEL_CTX   ];
    /* 后处理模块 */
    context_t sao_merge_type_index          [NUM_SAO_MERGE_FLAG_CTX];
    context_t sao_mode                      [NUM_SAO_MODE_CTX      ];
    context_t sao_interval_offset_abs       [NUM_SAO_OFFSET_CTX    ];
    context_t alf_cu_enable_scmodel      [3][NUM_ALF_LCU_CTX       ];
} ctx_set_t;


/* ---------------------------------------------------------------------------
 * struct to characterize the state of the arithmetic coding
 */
struct aec_t {
    ALIGN16(uint8_t *p_start);        /* actual buffer for written bytes */
    /* bitstream */
    uint8_t    *p;                    /* pointer to byte written currently */
    uint8_t    *p_end;                /* end of actual buffer for written bytes */
    uint32_t    reg_flush_bits;       /* register: flushing bits (not written into byte buffer) */
    uint32_t    num_left_flush_bits;  /* number of bits in \ref{reg_flush_bits} could be used */

    /* AEC codec */
    uint32_t    i_low;                /* low */
    uint32_t    i_t1;                 /* t1 */
    uint32_t    i_bits_to_follow;     /* current bit counter to follow */

    /* flag */
    uint32_t    b_writting;           /* write to bitstream buffer? */

    /* handle */
    binary_t    binary;               /* binary function handles */

    /* context */
    ctx_set_t  *p_ctx_set;            /* can reference other aec_t object */
    ctx_set_t   ctx_set;              /* context models for AEC (current object) */
};


/* ---------------------------------------------------------------------------
 * slice_t
 */
typedef struct slice_t {
    bs_t        bs;                   /* bitstream controller */

    /* bitstream buffer */
    int         len_slice_bs_buf;     /* length  of bitstream buffer */
    uint8_t    *p_slice_bs_buf;       /* pointer of bitstream buffer (start address) */

    /* slice buffers */
    pel_t      *slice_intra_border[3];    /* buffer for store decoded bottom pixels of the top lcu row (before filter) */
    uint8_t    *slice_deblock_flag[2];    /* buffer for edge filter flag (of one LCU row), [dir][(scu_y, scu_x)] */
    int8_t     *slice_ipredmode;          /* [(i_height_in_minpu + 1) * (i_width_in_minpu + 16)], prediction intra mode */

    /* slice properties */
    int         i_first_lcu_xy;       /* first LCU index (in scan order) */
    int         i_last_lcu_xy;        /* last  LCU index (in scan order) */
    int         i_first_lcu_y;        /* first LCU position y in this slice */
    int         i_last_lcu_y;         /* last  LCU position y in this slice */
    int         i_lcu_row_num;        /* number of LCU-row    in this slice */
    int         i_first_scu_y;        /* first SCU position y in this slice */
    int         i_qp;                 /* slice qp */
    int         index_slice;          /* index of current Slice */
} slice_t;


/* ---------------------------------------------------------------------------
 * prediction mode
 */
typedef struct neighbor_inter_t {
    int8_t      is_available;         /* is block available */
    int8_t      i_dir_pred;           /* prediction direction, -1 for intra or un-available */
    int8_t      ref_idx[2];           /* reference indexes of 1st and 2nd frame */
    mv_t        mv[2];                /* motion vectors */
} neighbor_inter_t;


/* ---------------------------------------------------------------------------
 * candidate node, used for intra coding
 */
struct intra_candidate_t {
    rdcost_t    cost;                 /* the cost of one mode */
    int         mode;                 /* the mode index */
    int8_t      padding_bytes[4];     /* padding byte number */
};


/* ---------------------------------------------------------------------------
 * coding block
 */
typedef union cb_t {
    struct {
        int8_t  x;                    /* start position (x, in pixel) within current CU */
        int8_t  y;                    /* start position (y, in pixel) within current CU */
        int8_t  w;                    /* block width  (in pixel) */
        int8_t  h;                    /* block height (in pixel) */
    };
    uint32_t    v;                    /* used for fast operation for all components */
} cb_t;


/* ---------------------------------------------------------------------------
 * coding unit data for storing
 */
struct cu_info_t {
    /* basic */
    int         i_scu_x;              /* horizontal position for the first SCU in CU */
    int         i_scu_y;              /* vertical   position for the first SCU in CU */

    pel_t      *p_rec[3];             /* reconstruction pixels for current cu [y/u/v] */
    coeff_t    *p_coeff[3];           /* residual coefficient  for current cu [y/u/v] */

    int8_t      i_level;              /* cu level, 3: 8x8, 4: 16x16, 5: 32x32, 6: 64x64 */

#if ENABLE_RATE_CONTROL_CU
    /* qp */
    int8_t      i_cu_qp;              /* qp of current CU */
    int8_t      i_delta_qp;           /* delta qp */
#endif   // ENABLE_RATE_CONTROL_CU

    /* mode */
    int8_t      i_mode;               /* cu type (partition into prediction units (PUs)) */
    int8_t      directskip_wsm_idx;   /* weighted skip mode */
    int8_t      directskip_mhp_idx;   /* direct skip mode index */
    int8_t      dmh_mode;             /* DMH mode */

    /* partition */
    int8_t      num_pu;               /* number of prediction units (PU) */
    /* trans size */
    int8_t      i_tu_split;           /* transform unit split flag, tu_split_type_e */

    /* cbp */
    int8_t      i_cbp;                /* Coding Block Pattern (CBP) or Coding Transform Pattern (CTP):
                                       *   Indicating whether transform block (TB) has nonzero coefficients
                                       *   When it is zero, it means all 6 TBs are zero block */

    /* intra predicated mode */
    int8_t      i_intra_mode_c;       /* real intra mode (chroma) */

    /* buffers */
    cb_t        cb[4];                /* coding blocks (2 for inter, 4 for intra) */

    /* intra buffers */
    int8_t      pred_intra_modes[4];  /* pred intra modes */
    int8_t      real_intra_modes[4];  /* real intra modes */

    /* inter buffers */
    mv_t        mvd[2][4];            /* [fwd,bwd][block_y][block_x] */
#if XAVS2_TRACE
    mv_t        mvp[2][4];            /* [fwd,bwd][block_y][block_x], used only for normal inter mode */
    mv_t        mv [2][4];            /* [fwd,bwd][block_y][block_x], used only for normal inter mode */
#endif
    int8_t      b8pdir[4];
    int8_t      ref_idx_1st[4];       /* reference index of 1st direction */
    int8_t      ref_idx_2nd[4];       /* reference index of 2nd direction */
};


/* ---------------------------------------------------------------------------
 * cu_mv_mode_t
 */
typedef struct cu_mv_mode_t {
    mv_t        all_sym_mv[1];              /* 对称模式的MV */
    mv_t        all_single_mv[MAX_REFS];

    /* mvp可以只对整个LCU只保留一份，无须按照深度分层 */
    mv_t        all_mvp[MAX_REFS];          /* 1st MVP of dual hypothesis prediction mode, or Foreword of BiPrediction */

    /* 双向MV也只需要保留一份 */
    mv_t        all_dual_mv_1st[MAX_REFS];
    mv_t        all_dual_mv_2nd[MAX_REFS];
} cu_mv_mode_t;


/* ---------------------------------------------------------------------------
 * MVs and references for motion compensation and references
 */
typedef struct cu_mc_param_t {
    mv_t        mv[4][2];             /* [blockidx][refidx 1st/2nd] */
} cu_mc_param_t;


/* ---------------------------------------------------------------------------
 * cu_mode_t
 */
typedef struct cu_mode_t {
    uint8_t       mv_padding1[16];          /* 避免越界，至少需2字节，此处为对齐补到16字节 */
    cu_mv_mode_t  mvs[MAX_INTER_MODES][4];  /* MVs for normal inter prediction */
    cu_mc_param_t best_mc;                  /* MVs to store */
    cu_mc_param_t best_mc_tmp;              /* 用于算法 OPT_ROUGH_PU_SEL 保存多个帧间划分模式的最佳参数（不一定是全局最优） */

    int8_t      ref_idx_single[4];          /* [block], preserved for DMH */

    mv_t        skip_mv_1st[DS_MAX_NUM];    /* MVs for spatial skip modes (only for F and B frames) */
    mv_t        skip_mv_2nd[DS_MAX_NUM];

    int8_t      skip_ref_1st[DS_MAX_NUM];   /* reference indexes */
    int8_t      skip_ref_2nd[DS_MAX_NUM];

    mv_t        tskip_mv[4][MAX_REFS];      /* MVs for temporal skip modes (Weighted skip and the default) */

    // int8_t      all_intra_mode[1 << ((MAX_CU_SIZE_IN_BIT - MIN_PU_SIZE_IN_BIT) * 2)];
    // int8_t      all_ctp_y[1 << ((MAX_CU_SIZE_IN_BIT - MIN_PU_SIZE_IN_BIT) * 2)];
} cu_mode_t;


/* ---------------------------------------------------------------------------
 * cu_feature_t, used for fast encoding decision
 */
typedef struct cu_feature_t {
    int64_t    intra_complexity;        /* intra complexity */
    int64_t    complexity;              /* minimum of intra and inter complexity */
    double     variance;                /* variance of current CU */
    double     avg_variance_sub_block;  /* average variance of 4 sub CUs */
    double     var_variance_sub_block;  /* variance of variance of 4 sub CUs */
    double     var_diff;                /* variance difference */
    rdcost_t   intra_had_cost;
    rdcost_t   rdcost;
    rdcost_t   rdcost_luma;
    /* 0: try both (not determined);
     * 1: only try split;
     * 2: only try current depth
     * --------------------------- */
    int        pred_split_type;         /* prediction of cu split type: 0: un-determined; 1: split; 2: not-split */
    rdcost_t   pred_costs[MAX_PRED_MODES];  /* 每种PU划分模式的 cost （基于预分析等获取） */
} cu_feature_t;


/* ---------------------------------------------------------------------------
 * coding unit
 */
struct cu_t {
    /* basic */
    int         i_size;               /* cu size */
    int         i_pos_x;              /* pixel position (x) within CTU */
    int         i_pos_y;              /* pixel position (y) within CTU */
    int         i_pix_x;              /* pixel position (x) within picture */
    int         i_pix_y;              /* pixel position (y) within picture */
    int         i_scu_xy;             /* CU position within picture */
    cu_info_t   cu_info;              /* information of CU */

    cu_mc_param_t mc;                 /* motion information for MC and neighboring prediction */

    /* pointer to neighbor CUs. NULL pointer identifies unavailable */
    cu_info_t  *p_left_cu;            /* pointer to left      neighbor cu */
    cu_info_t  *p_topA_cu;            /* pointer to top-above neighbor cu */
    cu_info_t  *p_topL_cu;            /* pointer to top-left  neighbor cu */
    cu_info_t  *p_topR_cu;            /* pointer to top-right neighbor cu */

    /* block available */
    uint8_t     intra_avail;          /* intra availability of current CU */
    uint8_t     block_avail;          /* intra availability of current intra PU */
    int8_t      in_lcu_edge;          /* 0: on top-left of LCU; 1: x on edge (y == 0); 2: y on edge (x == 0); 3: not on edge */
    int8_t      idx_zorder;           /* CU z-order index in CTU (basic: 8x8 CU) */
    int8_t      idx_cu_bfs;           /* index of CU in BFS scan order */

    /* TU size */
    bool_t      is_ctu_split;         /* whether CTU is split */
    bool_t      b_cbp_direct;         /* direct mode total distortion is zero*/
    bool_t      is_zero_block;        /* low residual in luma component */
    int         sum_satd;             /* satd sum for zero block detection */

    /* splitting */
    cu_t       *sub_cu[4];            /* pointer to 4 sub CUs split */

    /* RDO result of current depth */
    rdcost_t    rdcost;               /* RD-Cost of current CTU */
    dist_t      best_dist_total;      /* Total distortion for current CU, no split */

    cu_feature_t feature;             /* used for fast mode decision */

    dist_t      mvcost[4];            /* mvcost of every pu*/
};


/* ---------------------------------------------------------------------------
 * frame complexity
 */
typedef struct complex_t {
    int64_t     i_best_cost;          /* sum best  cost of whole frame */
    int64_t     i_inter_cost;         /* sum inter cost of whole frame */
    int64_t     i_intra_cost;         /* sum intra cost of whole frame */
    int         i_sum_intras;         /* number of intra blocks in frame */
    int         i_sum_blocks;         /* number of total blocks in frame */
    int         i_slice_type;         /* slice type of frame, or -1 for uncertain */
    int         b_valid;              /* indicates whether complexity estimation has conducted */
} complex_t;

#if XAVS2_ADAPT_LAYER
/* ---------------------------------------------------------------------------
 * nal info
 */
typedef struct xavs2_nal_info_t {
    short   i_priority;               /* nal_priority_e  */
    short   i_type;                   /* nal_unit_type_e */
    int     i_payload;                /* size of payload in bytes */
} xavs2_nal_info_t;
#endif

typedef struct com_stat_t {
    double      f_psnr[3];                    /* psnr for all components: Y, U, V */
    double      f_ssim[3];                    /* SSIM for all components: Y, U, V */
    double      f_lambda_frm;                 /* lambda of current frame */
//  int64_t     i_time_start;                 /* encoding start time */
//  int64_t     i_time_end;                   /* encoding end time */
    int64_t     i_time_duration;              /* encoding time */
    int64_t     i_frame_size;                 /* frame size  (bs len) */
    int         num_frames;
} com_stat_t;

/* ---------------------------------------------------------------------------
 * xavs2_frame_t
 */
struct xavs2_frame_t {
    /* magic number */
    ALIGN16(void *magic);             /* must be the 1st member variable. do not change it */

    /* properties */
    int         i_frm_type;           /* frame type: XAVS2_TYPE_* */
    int         i_state;              /* flag, -1 for exit flag in thread */
    int         b_keyframe;           /* key frame? */
    int64_t     i_pts;                /* user pts (Presentation Time Stamp) */
    int64_t     i_dts;                /* user dts (Decoding Time Stamp) */
    int64_t     i_reordered_pts;      /* reordered PTS (in coding order) */

    int         i_frame;              /* presentation frame number */
    int         i_frm_coi;            /* COI (coding  order index) */
    int         i_frm_poc;            /* POC (picture order count), used for MV scaling */
    int         i_gop_idr_coi;        /* COI of IDR frame in this gop */
    int         ref_dpoc[MAX_REFS];   /* POC difference of its reference frames */
    int         ref_dpoc_multi[MAX_REFS];   /* MULTI / ref_dpoc[x] */

    int         i_frm_qp;             /* QP of frame */
    int         i_frm_lambda_sad;     /* frame level lambda in SAD domain */
    double      f_frm_lambda_ssd;     /* frame level lambda in SSD domain */

    int         i_qpplus1;            /* qp + 1: used for rate control */

    xavs2_rps_t   rps;
    int           rps_index_in_gop;
    bool_t        b_random_access_decodable;  /* random_access_decodable_flag */

    /* YUV buffer */
    int         i_plane;              /* number of planes */
    int         i_stride[3];          /* stride for Y/U/V */
    int         i_width[3];           /* width  for Y/U/V */
    int         i_lines[3];           /* height for Y/U/V */
    pel_t      *planes[3];            /* pointers to Y/U/V data buffer */
    pel_t      *filtered[16];         /* pointers to interpolated luma data buffers */

    pel_t      *plane_buf;
    int         size_plane_buf;

    /* bit stream buffer */
    uint8_t    *p_bs_buf;             /* bit stream buffer for encoding this frame */
    int         i_bs_buf;             /* length of bit stream buffer */
    int         i_bs_len;             /* length of bit stream data */

    int         b_enable_intra;       /* enable intra coding in frame level */

    /* encoding parameters */
    int8_t     *pu_ref;               /* pu reference index (store in 16x16 block) */
    mv_t       *pu_mv;                /* pu motion vector   (store in 16x16 block) */
#if SAVE_CU_INFO
    int8_t     *cu_mode;              /* cu type        (store in SCU) */
    int8_t     *cu_cbp;               /* cu cbp         (store in SCU) */
    int8_t     *cu_level;             /* cu size in bit (store in SCU) */
#endif

    int         num_lcu_sao_off[NUM_SAO_COMPONENTS];

    /* */
    uint32_t    cnt_refered;          /* reference count for FT_DEC */

    int        *num_lcu_coded_in_row; /* 0, not ready, 1, ready */

    xavs2_thread_cond_t  cond;
    xavs2_thread_mutex_t mutex;

#if XAVS2_ADAPT_LAYER
    /* nal */
    int               i_nal;          /* number of nal */
    xavs2_nal_info_t *nal_info;       /* nal information */
#endif

#if XAVS2_STAT
    int64_t     i_time_start;         /* encoding start time */
    int64_t     i_time_end;           /* encoding end time */
#endif
};


/* ---------------------------------------------------------------------------
 * xavs2_me_t
 */
typedef struct xavs2_me_t {
    /* PU info */
    int16_t     i_ref_idx;            /* current reference index */
    int16_t     i_pixel;              /* partition index via the block width and height */
    int         i_bias;               /* offset of the current PU block in the frame */
    int         i_pix_x;              /* pixel position (x) in frame */
    int         i_pix_y;              /* pixel position (y) in frame */
    int         i_block_w;            /* width  of the current PU block */
    int         i_block_h;            /* height of the current PU block */
    bool_t      b_search_dmh;         /* is searching for DMH mode */

    /* pointers */
    pel_t         *p_fenc;            /* pointer to the current PU block in source CTU */
    xavs2_frame_t *p_fref_1st;        /* pointer to the current (1st) reference frame */
    xavs2_frame_t *p_fref_2nd;        /* pointer to the current  2nd  reference frame */

    int         i_distance_1st;       /* distance index for 1st reference frame */
    int         i_distance_2nd;       /* distance index for 2nd reference frame */

    /* thresholds for UMH */
    double      beta2;
    double      beta3;

    /* SAD prediction */
    dist_t      pred_sad_space;
    dist_t      pred_sad_ref;
    dist_t      pred_sad_uplayer;
    dist_t      pred_sad;

    /* mv range */
    int         mv_min[2];            /* allowed qpel MV range to stay within */
    int         mv_max[2];            /* the picture + emulated edge pixels   */
    int         mv_min_fpel[2];       /* full pel MV range for motion search  */
    int         mv_max_fpel[2];

    /* pred motion vector */
    mv_t        mvp;                  /* pred motion vector for the current block */
    mv_t        mvp1;                 /* MVP via space */
    mv_t        mvp2;                 /* MVP via temporal collocation (previous search result) */
    mv_t        mvp3;                 /* MVP via collocated frame */

    /* output */
    mv_t        bmv;                  /* best motion vector (subpel ) */
    mv_t        bmv2;                 /* best motion vector (fullpel) */
    dist_t      bcost;                /* best cost of subpel  motion search, satd + lambda * nbits */
    dist_t      bcost2;               /* best cost of fullpel motion search, sad  + lambda * nbits */

    dist_t      mvcost[5];            /* mv cost for every direction*/
    dist_t      bmvcost[5];           /* cost of best mv of all ref for every direction */

    mv_t        all_best_mv[MAX_INTER_MODES][4][MAX_REFS];  /* all best mv results generated in ME (single) */
    mv_t        all_best_imv[MAX_REFS];    /* best integer MV for current PU in current CU */
} xavs2_me_t;



/* ---------------------------------------------------------------------------
 * SAOStatData
 */
typedef struct SAOStatData {
    long        diff[MAX_NUM_SAO_CLASSES];
    long        count[MAX_NUM_SAO_CLASSES];
} SAOStatData;


/* ---------------------------------------------------------------------------
 * ALFParam
 */
typedef struct ALFParam {
    int         alf_flag;
    int         num_coeff;
    int         filters_per_group;
    int         filterPattern[NO_VAR_BINS];
    int         coeffmulti[NO_VAR_BINS][ALF_MAX_NUM_COEF];
} ALFParam;



/* ---------------------------------------------------------------------------
 * parameters and buffers for RDOQ
 */
typedef struct rdoq_t {
    /* buffers */
    ALIGN32(coeff_t coeff_buff[32 * 32]);
    ALIGN32(coeff_t ncur_blk  [32 * 32]);
    ALIGN32(int8_t  sig_cg_flag[64]);

    /* pointers */
    context_t  *p_ctx_coeff_run;
    context_t  *p_ctx_coeff_level;
    context_t (*p_ctx_primary)[NUM_MAP_CTX];
    context_t  *p_ctx_sign_cg;
    context_t  *p_ctx_last_cg;
    context_t  *p_ctx_last_pos;
    const int16_t *p_scan_tab_1d;     /* scan table */
    const int16_t (*p_scan_cg)[2];    /* scan table (CG) */

    /* properties */
    int         num_cg_x;             /* number of CG in x axis */
    int         num_cg_y;             /* number of CG in y axis */
    int         bit_size_shift_x;     /* log2 (block size x) */
    int         bit_size_shift_y;     /* log2 (block size x) */
    int         i_tu_level;           /* */
    int         b_luma;               /* is luma? */
    int         b_dc_diag;            /* is INTRA_PRED_DC_DIAG or not */
} rdoq_t;


#if ENABLE_WQUANT
/* ---------------------------------------------------------------------------
 * weighted quant data
 */
typedef struct wq_data_t {
    int16_t     wq_param          [2][ 6];
    int16_t     cur_wq_matrix     [4][64];      // [matrix_id][coef]
    int16_t     wq_matrix      [2][2][64];      // [matrix_id][detail/undetail][coef]
    int16_t     seq_wq_matrix     [2][64];      // [matrix_id][coef]
    int16_t     pic_user_wq_matrix[2][64];      // [matrix_id][coef]

    int         LevelScale4x4  [2][ 4 *  4];
    int         LevelScale8x8  [2][ 8 *  8];    // [intra/inter][j * stride + i]
    int         LevelScale16x16[2][16 * 16];
    int         LevelScale32x32[2][32 * 32];

    int        *levelScale[4][2];               // [bit_size][intra/inter]
    int         cur_frame_wq_param;             // weighting quant param
} wq_data_t;
#endif


/* ---------------------------------------------------------------------------
 * The data within the payload is already NAL-encapsulated; the ref_idc and
 * type are merely in the struct for easy access by the calling application.
 * All data returned in an nal_t, including the data in p_payload, is no
 * longer valid after the next call to xavs2_encoder_encode.
 */
typedef struct nal_t {
    int         i_ref_idc;            /* nal_priority_e */
    int         i_type;               /* nal_unit_type_e */
    int         i_payload;            /* size of payload in bytes */
    uint8_t    *p_payload;            /* payload */
} nal_t;


/* ---------------------------------------------------------------------------
 * lcu_info_t
 */
typedef struct lcu_info_t {
    ALIGN32(coeff_t coeffs_y[MAX_CU_SIZE * MAX_CU_SIZE]);          /* dct coefficients of Y component */
    ALIGN32(coeff_t coeffs_uv[2][MAX_CU_SIZE * MAX_CU_SIZE / 4]);  /* dct coefficients of U/V component */
    int         scu_xy;               /* index (scan order ) for the first SCU in lcu */
    int         pix_x;                /* horizontal position (in pixel) of lcu (luma) */
    int         pix_y;                /* vertical   position (in pixel) of lcu (luma) */
    int         slice_index;          /* slice index */
#if ENABLE_RATE_CONTROL_CU
    int         last_dqp;             /* last delta QP */
#endif
} lcu_info_t;


/* ---------------------------------------------------------------------------
 * row_info_t
 */
typedef struct row_info_t {
    int             row;              /* row index [0, xavs2_t::i_height_in_lcu) */
    int             b_top_slice_border;   /* whether top  slice border should be processed */
    int             b_down_slice_border;  /* whether down slice border should be processed */
    volatile int    coded;            /* position of latest coded LCU. [0, xavs2_t::i_width_in_lcu) */

    xavs2_t         *h;               /* context for the row */
    lcu_info_t      *lcus;            /* [LCUs] */

    xavs2_thread_cond_t  cond;       /* lcu cond */
    xavs2_thread_mutex_t mutex;

    aec_t           aec_set;          /* aec contexts of the 2nd LCU which will be
                                       * referenced by the next row on startup */
} row_info_t;

#if XAVS2_STAT
/* ---------------------------------------------------------------------------
 * struct for encoding statistics of one frame
 */
typedef struct frame_stat_t {
    int         i_type;               /* frame type */
    int         i_frame;              /* POC */
    int         i_qp;                 /* frame QP */
    int         i_ref;                /* number of reference frames */
    int         ref_poc_set[XAVS2_MAX_REFS];   /* POCs   of reference frames */
    com_stat_t  stat_frm;
} frame_stat_t;

/* ---------------------------------------------------------------------------
 * struct for encoding statistics of all frames
 */
typedef struct xavs2_stat_t {
    int64_t     i_start_time;         /* encoding start time */
    int64_t     i_end_time;           /* encoding end time */
    com_stat_t  stat_i_frame;
    com_stat_t  stat_p_frame;
    com_stat_t  stat_b_frame;
    com_stat_t  stat_total;
    int         num_frame_small_qp;   /* number of frames whose QP is too small */
} xavs2_stat_t;
#endif


/* ---------------------------------------------------------------------------
 * frame_info_t
 */
typedef struct frame_info_t {
    row_info_t     *rows;             /* all lcu rows */
#if XAVS2_STAT
    frame_stat_t    frame_stat;       /* encoding statistics */
#endif
} frame_info_t;


/* ---------------------------------------------------------------------------
 * outputframe_t
 */
struct outputframe_t {
    xavs2_frame_t   *frm_enc;         /* frame with nalus */
#if XAVS2_STAT
    frame_stat_t     out_frm_stat;    /* encoding statistics */
#endif
    outputframe_t   *next;            /* pointer to the next output frame */
};



/* ---------------------------------------------------------------------------
 * buffer data used for each cu layer in xavs2_t
 */
typedef struct cu_layer_t {
    rdcost_t         best_rdcost;
    /* best rd-cost of current CU */
    rdcost_t         mode_rdcost[MAX_PRED_MODES];   /* min rd-cost for each mode */
    int              mask_md_res_pred;              /* available mode mask */

    pel_t           *p_rec_tmp[3];    /* tmp pointers to ping-pong buffer for swapping */
    coeff_t         *p_coeff_tmp[3];  /* tmp pointers to ping-pong buffer for swapping */

    cu_info_t        cu_best;         /* best info for each cu depth */
    cu_mode_t        cu_mode;         /* mode info for each cu depth (TODO: simplification for motion info like x265 ?) */

    intra_candidate_t intra_candidates[INTRA_MODE_NUM_FOR_RDO + 1]; /* candidate list, reserving the cost */
    neighbor_inter_t  neighbor_inter[BLK_COL + 1];   /* neighboring inter modes of 4x4 blocks*/

    aec_t            cs_cu ;          /* coding state after encoding each cu partition mode for current CU level */
    aec_t            cs_rdo;          /* coding state for mode decision (rdo) */

    // uint8_t     padding_bytes[24];   /* padding bytes to make align */
#define FENC_STRIDE    (MAX_CU_SIZE)    /* stride for LCU enc buffer, Y component */
#define FDEC_STRIDE    (MAX_CU_SIZE)    /* stride for LCU dec buffer, Y component */
#define FREC_STRIDE    (p_cu->i_size)   /* stride for current CU, Y component */
#define FREC_CSTRIDE   (p_cu->i_size)   /* stride for current CU, UV component */
#define FENC_BUF_SIZE  (FENC_STRIDE * (MAX_CU_SIZE + MAX_CU_SIZE / 2))
#define FDEC_BUF_SIZE  (FDEC_STRIDE * (MAX_CU_SIZE + MAX_CU_SIZE / 2))
#define LCU_BUF_SIZE   (MAX_CU_SIZE * MAX_CU_SIZE)

    ALIGN32(pel_t   rec_buf_y     [3][LCU_BUF_SIZE]);       /* luma   reconstruction buffer     [cur/tmp/best][] */
    ALIGN32(coeff_t coef_buf_y    [3][LCU_BUF_SIZE]);       /* luma   coefficient    buffer     [cur/tmp/best][] */
    ALIGN32(pel_t   rec_buf_uv [2][3][LCU_BUF_SIZE >> 2]);  /* chroma reconstruction buffer [uv][cur/tmp/best][] */
    ALIGN32(coeff_t coef_buf_uv[2][3][LCU_BUF_SIZE >> 2]);  /* chroma coefficient    buffer [uv][cur/tmp/best][] */

    /* inter prediction buffer */
    ALIGN32(pel_t   buf_pred_inter_luma[2][LCU_BUF_SIZE]);  /* temporary decoding buffer for inter prediction (luma) */
    /* Ping-pong buffer for inter prediction */
    pel_t   *buf_pred_inter;        /* current inter prediction buffer */
    pel_t   *buf_pred_inter_best;   /* backup of best inter prediction */
} cu_layer_t;

/* ---------------------------------------------------------------------------
 * buffer data used for encode each CU layer
 */
typedef struct cu_parallel_t {
    /* dct coefficients buffers */
    ALIGN32(coeff_t coeff_blk[LCU_BUF_SIZE]);
    ALIGN32(coeff_t coeff_bak[LCU_BUF_SIZE]);

    /* buffers used for inter prediction */
    ALIGN32(pel_t   buf_pred_inter_c[LCU_BUF_SIZE >> 1]);   /* temporary decoding buffer for inter prediction (chroma) */
    ALIGN32(pel_t   buf_pixel_temp  [LCU_BUF_SIZE]);        /* temporary pixel buffer, used for bi/dual-prediction */

    /* predication buffers for all intra modes */
    ALIGN32(pel_t   intra_pred  [NUM_INTRA_MODE       ][LCU_BUF_SIZE]);         /* for all 33 luma prediction modes */
    ALIGN32(pel_t   intra_pred_c[NUM_INTRA_MODE_CHROMA][LCU_BUF_SIZE >> 1]);    /* for all chroma intra prediction modes */
    ALIGN32(pel_t   buf_edge_pixels[MAX_CU_SIZE << 3]);     /* reference pixels for intra luma/chroma prediction */

    runlevel_t       runlevel;         /* run level buffer for RDO */

    /* parameters for RDOQ */
    ALIGN16(rdoq_t  rdoq_info);

    aec_t  cs_tu;                   /* coding state after encoding cu with different TU partition, or PU partition in intra */
    aec_t  cs_pu_init;              /* coding state before encoding one CU partition */
} cu_parallel_t;


/* ---------------------------------------------------------------------------
 */
struct xavs2_log_t {
    int         i_log_level;          /* log level */
    char        module_name[60];      /* module name */
};

/* ---------------------------------------------------------------------------
 * xavs2_t
 */
struct xavs2_t {
    ALIGN32(xavs2_log_t    module_log);      /* log module */
    /* === BEGIN ===================================================
     * communal variables
     * 序列级（编码的所有帧）共享变量区域开始
     */

    ALIGN32(SYNC_VARS_1(communal_vars_1));
    const xavs2_param_t*   param;            /* input parameters */

    /* -------------------------------------------------------------
     * contexts synchronization control
     */
    xavs2_handler_t*h_top;            /* encoder top handler */
    task_type_e     task_type;        /* task type: frame/slice/row */
    task_status_e   task_status;      /* for frame tasks: task status */
    int             i_aec_frm;        /* for frame tasks(task order for aec): [0, i_frame_threads) */
    int             b_all_row_ctx_released;   /* is all row context released */

    /* -------------------------------------------------------------
     * encoder contexts
     */

    ratectrl_t     *rc;               /* rate control */
    td_rdo_t       *td_rdo;           /* pointer to struct td_rdo_t */

    uint32_t    valid_modes[SLICE_TYPE_NUM][CTU_DEPTH]; /* [frame_type][bit_size] : valid modes for mode decision */

    uint64_t    i_fast_algs;          /* all fast algorithms enabled */
    bool_t      b_progressive;
    bool_t      b_field_sequence;
    bool_t      use_fractional_me;    /* whether use fractional Motion Estimation
                                       * 0: 关闭分像素搜索；1: 开启1/2分像素搜索；2:开启1/4分像素搜索
                                       */
    bool_t      use_fast_sub_me;      /* whether use fast quarter Motion Estimation: skip half fractional search point (from futl) */
    bool_t      UMH_big_hex_level;     /* whether skip big hex pattern when using UMH (from futl)
                                          0 : skip this step
                                          1 :  8 points. 0.17% loss ~ 4% TimeSaving
                                          2 : 16 points */
    bool_t      enable_tu_2level;        /* enable 2-level TU for inter ,
                                          * 0: off,
                                          * 1: tu-2level only for best partition mode of one CU,
                                          * 2: tu-2level rdo ,
                                          * 3: tu - 2level rdoq */
    bool_t      skip_rough_improved;  /* whether use the improved SKIP_ROUGH_SEL (from leimeng) */
    float       framerate;
    int         i_gop_size;           /* sub GOP size */
    int         picture_reorder_delay;/* picture reorder delay */
    int         i_lcu_level;          /* level of largest  cu, 3: 8x8, 4: 16x16, 5: 32x32, 6: 64x64 */
    int         i_scu_level;          /* level of smallest cu, 3: 8x8, 4: 16x16, 5: 32x32, 6: 64x64 */
    int         i_width;              /* frame width  (number of pels,  8N, Luma) */
    int         i_height;             /* frame height (number of lines, 8N, Luma) */
    int         i_width_in_lcu;       /* frame width  in lcu */
    int         i_height_in_lcu;      /* frame height in lcu */
    int         i_width_in_mincu;     /* frame width  in 8x8-block */
    int         i_height_in_mincu;    /* frame height in 8x8-block */
    int         i_width_in_minpu;     /* frame width  in 4x4-block */
    int         i_height_in_minpu;    /* frame height in 4x4-block */
    int         i_chroma_v_shift;     /* chroma vertical shift bits */
    int         i_max_ref;            /* max number of reference frames */
    int         min_mv_range[2];      /* mv range (min) decided by the level id */
    int         max_mv_range[2];      /* mv range (max) decided by the level id */
    /* function pointers */
    int       (*get_intra_candidates_luma)(xavs2_t *h, cu_t *p_cu, intra_candidate_t *p_candidates,
                                           pel_t *p_fenc, int mpm[], int blockidx,
                                           int block_x, int block_y, int block_w, int block_h);
    int       (*get_intra_candidates_chroma)(xavs2_t *h, cu_t *p_cu, int i_level, int pix_y_c, int pix_x_c,
                                             intra_candidate_t *p_candidate_list);
    void      (*copy_aec_state_rdo)(aec_t *dst, aec_t *src);  /* pointer to copy aec_t */
    int         size_aec_rdo_copy;    /* number of bytes to copy in RDO for \function aec_copy_aec_state_rdo() */
    uint8_t    *tab_avail_TR;         /* pointers to array of available table, Top Right */
    uint8_t    *tab_avail_DL;         /* pointers to array of available table, Down Left */
    uint8_t     tab_num_intra_rdo[MAX_CU_SIZE_IN_BIT + 1];    /* pointers to array of table, indicate numbers of intra prediction modes for RDO */
    int8_t      num_intra_rmd_dist2;  /* 距离2的角度的搜索数量 */
    int8_t      num_intra_rmd_dist1;  /* 距离1的角度的搜索数量 */
    int8_t      num_rdo_intra_chroma; /* number of RDO modes for intra chroma prediction */

    SYNC_VARS_2(communal_vars_2);
    /* === END ===================================================== */

    /* === BEGIN ===================================================
     * row-dependent variables : values below need to be synchronized between rows
     * 帧级共享变量区域开始，每帧的多个行级线程之间访问相同内容
     */
    SYNC_VARS_1(row_vars_1);

    frame_info_t   *frameinfo;        /* pointer to the frame info buffer */
    int             i_type;           /* frame type: SLICE_TYPE_* */
    int             i_layer;          /* temporal index of coding frame */
    int             i_qp;             /* frame level QP */

    int             ip_pic_idx;       /* encoded I/P/F-picture index (to be REMOVED) */
    int             i_frame_b;        /* number of encoded B-picture in a GOP */

    int             b_top_field;      /* top field flag */

    rdcost_t        f_lambda_mode;    /* lambda for mode cost and motion cost */
    rdcost_t        f_lambda_rdoq;    /* lambda for RDOQ */
    int             i_lambda_factor;  /* factor for determining Lagrangian's motion cost */
    double          f_lambda_1th;     /* 1.0 / f_lambda_mode */

    xavs2_frame_t  *fenc;             /* current frame being encoded */
    xavs2_frame_t  *fdec;             /* current frame being reconstructed */

    int             i_ref;            /* current number of reference frames */
    xavs2_frame_t  *fref[MAX_REFS];   /* reference frame list */
    mct_t          *img4Y_tmp[3];     /* temporary buffer for 1/4 interpolation: a,1,b */
    xavs2_frame_t  *img_luma_pre;     /* buffer used for TDRDO, only luma */

    /* slices */
    slice_t    *slices[MAX_SLICES];   /* all slices */
    int         i_slice_index;        /* slice index for the current thread */

    /* 不同Slice不同的buffer */
    pel_t      *intra_border[3];      /* buffer for store decoded bottom pixels of the top lcu row (before filter) */
    uint8_t    *p_deblock_flag[2];    /* buffer for edge filter flag (of one LCU row), [dir][(scu_y, scu_x)] */
    int8_t     *ipredmode;            /* [(i_height_in_minpu + 1) * (i_width_in_minpu + 16)], prediction intra mode */

    /* 帧级唯一的buffer */
    int8_t     *lcu_slice_idx;        /* [i_height_in_lcu][i_width_in_lcu] */
    int8_t     *dir_pred;             /* [i_height_in_minpu][i_width_in_minpu], inter prediction direction */
    int8_t     *fwd_1st_ref;          /* [i_height_in_minpu][i_width_in_minpu] */
    int8_t     *bwd_2nd_ref;          /* [i_height_in_minpu][i_width_in_minpu] */
    mv_t       *fwd_1st_mv;           /* [i_height_in_minpu][i_width_in_minpu] */
    mv_t       *bwd_2nd_mv;           /* [i_height_in_minpu][i_width_in_minpu] */

    uint16_t   *mvbits;               /* used for getting the mv bits */
    dist_t    (*all_mincost)[MAX_INTER_MODES][MAX_REFS];   /* store the min SAD (in 4x4 PU) */
    double      umh_bsize[MAX_INTER_MODES];

    double      thres_qsfd_cu[2][CTU_DEPTH];  /* QSFD threshold for inter frame, [0:inter, 1:intra][log2_cu_size - 3] */

    xavs2_frame_t *img_sao;          /* reconstruction image for SAO */
    SAOStatData(*sao_stat_datas)[NUM_SAO_COMPONENTS][NUM_SAO_NEW_TYPES]; /* [lcu][comp][types], 可不用全局 */
    SAOBlkParam(*sao_blk_params)[NUM_SAO_COMPONENTS];   /* [lcu][comp] */
    int        (*num_sao_lcu_off)[NUM_SAO_COMPONENTS];  /* [lcu_row][comp] */
    bool_t       slice_sao_on   [NUM_SAO_COMPONENTS];

    xavs2_frame_t *img_alf;          /* reconstruction image for ALF */
    void          *enc_alf;          /* handler of ALF encoder */
    ALFParam       pic_alf_params[IMG_CMPNTS];
    bool_t       (*is_alf_lcu_on)[IMG_CMPNTS]; /* [lcu][comp] */
    int            pic_alf_on[IMG_CMPNTS];

#if ENABLE_WQUANT
    int         WeightQuantEnable;    /* enable weight quantization? */
    wq_data_t   wq_data;
#endif

    cu_info_t  *cu_info;              /* pointer to buffer of all SCUs in frame */

    SYNC_VARS_2(row_vars_2);
    /* === END ===================================================== */

    nal_t      *p_nal;                /* pointer to struct nal_t */
    int         i_nal;                /* current NAL index */
    int         i_nal_type;           /* NAL type */
    int         i_nal_ref_idc;        /* NAL priority */

    bs_t        header_bs;            /* bitstream controller for main thread */
    uint8_t    *p_bs_buf_header;      /* pointer to bitstream buffer for headers */
    uint8_t    *p_bs_buf_slice;       /* pointer to bitstream buffer for slices */
    int         i_bs_buf_header;      /* size    of bitstream buffer for headers */
    int         i_bs_buf_slice;       /* size    of bitstream buffer for slices */

    xavs2_me_t  me_state;             /* used for motion estimation */

    aec_t       aec;                  /* ac engine for RDO */

#if ENABLE_RATE_CONTROL_CU
    int        *last_dquant;
#endif

    struct lcu_t {
        /* variable properties when coding each LCU ----------------
         */
        ALIGN16(int16_t i_pix_width); /* actual width  (in pixel) for current lcu */
        int16_t i_pix_height;         /* actual height (in pixel) for current lcu */
        int     i_pix_x;              /* horizontal position (in pixel) of lcu (luma) */
        int     i_pix_y;              /* vertical   position (in pixel) of lcu (luma) */
        int     i_scu_x;              /* horizontal position (raster scan order in frame buffer) for the first SCU of lcu */
        int     i_scu_y;              /* vertical   position (raster scan order in frame buffer) for the first SCU of lcu */
        int     i_scu_xy;             /* SCU index (raster scan order in frame buffer) for the top-left SCU of current lcu */
        int     i_lcu_xy;             /* LCU index (raster scan order in frame buffer) for current lcu */

        bool_t  b_enable_rdoq;
        bool_t  bypass_all_dmh;
        bool_t  b_2nd_rdcost_pass;    /* 2nd pass for RDCost update */

        /* function pointers for RDO */
        int   (*get_intra_dir_for_rdo_luma)(xavs2_t *h, cu_t *p_cu, intra_candidate_t *p_candidates,
                                            pel_t *p_fenc, int mpm[], int blockidx,
                                            int block_x, int block_y, int block_w, int block_h);
        int   (*get_skip_mvs)(xavs2_t *h, cu_t *p_cu);  /* get MVs for skip/direct mode */

        /* buffer and status for RDO & ENC -------------------------
         */

        /* 1, coding tree */
        cu_t   *p_ctu;                /* pointer to the top of current CTU */

        /* 2, enc/dec/pred Y/U/V pointers */
        pel_t      *p_fdec[3];        /* [Y/U/V] pointer over lcu of the frame to be reconstructed */
        pel_t      *p_fenc[3];        /* [Y/U/V] pointer over lcu of the frame to be compressed */

        coeff_t    *lcu_coeff[3];     /* [Y/U/V] coefficients of LCU */

        // uint8_t     padding_bytes[24];/* padding bytes to make align */

        /* data used in each ctu layer */
#define PARALLEL_INSIDE_CTU                    0
        cu_layer_t      cu_layer[CTU_DEPTH];
#if PARALLEL_INSIDE_CTU
        cu_parallel_t   cu_enc  [CTU_DEPTH];
#else
        cu_parallel_t   cu_enc  [1];                /* 无CTU内的多线程时，只需要一个 */
#endif

        ALIGN32(pel_t   fenc_buf[FENC_BUF_SIZE]);   /* encoding buffer (source Y/U/V buffer) */
        ALIGN32(pel_t   fdec_buf[FDEC_BUF_SIZE]);   /* decoding buffer (Reconstruction Y/U/V buffer) */
        struct lcu_intra_border_t {
            ALIGN32(pel_t rec_left[MAX_CU_SIZE]);          /* Left border of current LCU */
            ALIGN32(pel_t rec_top[MAX_CU_SIZE * 2 + 32]);  /* top-left, top and top-right samples (Reconstruction) of current LCU */
        } ctu_border[IMG_CMPNTS];                   /* Y, U, V components */

        /* buffer for the coding tree units */
        ALIGN16(cu_t    all_cu[85]);                /* all cu: 1(64x64) + 4(32x32) + 16(16x16) + 64(8x8) = 85 */
        ALIGN16(cu_t   *p_cu_l[4][8][8]);           /* all CU pointers */

        /* only used for AEC */
        runlevel_t      run_level_write;            /* run-level buffer for encoding */
    } lcu;

    /* coding states in RDO, independent for each thread */
    struct coding_states {

        /* 只用于备份上下文状态，无需初始化 */
        aec_t  cs_sao_start;
        aec_t  cs_sao_best;
        aec_t  cs_sao_temp;

        aec_t  cs_alf_cu_ctr;
        aec_t  cs_alf_initial;
    } cs_data;
};


/**
 * ===========================================================================
 * general function declares
 * ===========================================================================
 */

/* time (us) */

#define xavs2_mdate FPFX(mdate)
int64_t xavs2_mdate(void);

/* trace */
#if XAVS2_TRACE
#define xavs2_trace_init FPFX(trace_init)
int  xavs2_trace_init(xavs2_param_t *param);
#define xavs2_trace_destroy FPFX(trace_destroy)
void xavs2_trace_destroy(void);
#define xavs2_trace FPFX(trace)
int  xavs2_trace(const char *psz_fmt, ...);
#endif

/* thread */
#if HAVE_WIN32THREAD || PTW32_STATIC_LIB
#define xavs2_threading_init FPFX(threading_init)
int xavs2_threading_init(void);
#else
#define xavs2_threading_init()            0
#endif

#define xavs2_create_thread FPFX(create_thread)
int xavs2_create_thread(xavs2_thread_t *tid, xavs2_tfunc_t start, void *arg);

#define xavs2_log FPFX(log)
void xavs2_log(void *p, int i_log_level, const char *psz_fmt, ...);

/* ---------------------------------------------------------------------------
 * memory alloc
 */

/* xavs2_malloc : will do or emulate a memalign
 * you have to use xavs2_free for buffers allocated with xavs2_malloc */
#define xavs2_malloc FPFX(malloc)
void *xavs2_malloc(size_t i_size);
#define xavs2_calloc FPFX(calloc)
void *xavs2_calloc(size_t count, size_t size);
#define xavs2_free FPFX(free)
void  xavs2_free(void *ptr);
#define xavs2_get_total_malloc_space FPFX(get_total_malloc_space)
size_t xavs2_get_total_malloc_space(void);


#define g_xavs2_default_log          FPFX(g_xavs2_default_log)
extern xavs2_log_t    g_xavs2_default_log;

/**
 * ===========================================================================
 * global const tables
 * ===========================================================================
 */
#include "avs2tab.h"

#endif  // XAVS2_COMMON_H
