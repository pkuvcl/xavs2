/*
 * parameters.c
 *
 * Description of this file:
 *    Parameters definition of the xavs2 library
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

/* ---------------------------------------------------------------------------
 * disable warning C4996: functions or variables may be unsafe. */
#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS
#endif

/* ---------------------------------------------------------------------------
 * include files */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#if defined(_MSC_VER)
#include <io.h>
#include <fcntl.h>
#endif

#include "xavs2.h"
#include "common.h"
#include "rps.h"
#include "encoder/presets.h"
#include "encoder/encoder.h"
#include "encoder/wrapper.h"

/**
 * ===========================================================================
 * defines and global variables
 * ===========================================================================
 */

#define MAP_TAB_SIZE    512     /* maximal size of mapping table */
#define MAX_ITEMS       1024    /* maximal number of items to parse */

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#define REMOVE_WARNING \
    __pragma(warning(push))\
    __pragma(warning(disable:4127))

#define RESTORE_WARNING \
    __pragma(warning(pop))
#else
#define REMOVE_WARNING
#define RESTORE_WARNING
#endif

#define xavs2_param_match(x,y) (!strcasecmp(x,y))

/* ---------------------------------------------------------------------------
 * map type
 */
enum MAP_TYPE {
    MAP_STR  = 1,   /* char * */
    MAP_NUM  = 2,   /* int data */
    MAP_FLAG = 3,   /* flag: 0/1 value */
    MAP_FLOAT = 4,  /* float data */
    MAP_END = 9
};

/* ---------------------------------------------------------------------------
 * mapping for config item
 */
typedef struct mapping_t {
    char    name[32];           /* name for configuration */
    void    *addr;              /* memory address to store parameter value */
    int     type;               /* type, string or number */
    const char *s_instruction;  /* instruction */
} mapping_t;

/* mapping table of supported parameters */
typedef struct xavs2_param_map_t {
    xavs2_param_t *param;
    mapping_t      map_tab[MAP_TAB_SIZE];
} xavs2_param_map_t;

static xavs2_param_map_t g_param_map = {
    NULL
};

/**
 * ---------------------------------------------------------------------------
 * Function   : set default values for encoding parameters
 * Parameters :
 *      [in ] : map_tab - mapping table
 *            : p       - pointer to struct xavs2_param_t
 *      [out] : none
 * Return     : none
 * ---------------------------------------------------------------------------
 */
static void
mapping_default(xavs2_param_map_t *p_map_tab, xavs2_param_t *p)
{
    mapping_t * map_tab = p_map_tab->map_tab;
    p_map_tab->param = p;
    /* token - token name
     * var   - store address
     * t     - type
     * instr - instruction of parameter
     * */
#define MAP(token, var, t, instr)\
    REMOVE_WARNING\
    if (strlen(token) > 0) {\
        strcpy(map_tab[item_idx].name, (token));\
        map_tab[item_idx].addr = (var);\
        map_tab[item_idx].type = (t);\
        map_tab[item_idx].s_instruction = (instr);\
        item_idx++;\
    } else {\
        map_tab[item_idx].addr = NULL;\
        map_tab[item_idx].type = MAP_END;\
        map_tab[item_idx].s_instruction = "";\
    }\
    RESTORE_WARNING

    int item_idx = 0;

    /* input */
    MAP("Width",                        &p->org_width,                  MAP_NUM, "Image width  in pixels");
    MAP("SourceWidth",                  &p->org_width,                  MAP_NUM, "  - Same as `Width`");
    MAP("Height",                       &p->org_height,                 MAP_NUM, "Image height in pixels");
    MAP("SourceHeight",                 &p->org_height,                 MAP_NUM, "  - Same as `Height`");
    MAP("Input",                        &p->psz_in_file,                MAP_STR, "Input sequence, YUV 4:2:0");
    MAP("InputFile",                    &p->psz_in_file,                MAP_STR, "  - Same as `Input`");
    MAP("InputHeaderLength",            &p->infile_header,              MAP_NUM, "If the inputfile has a header, state it's length in byte here ");
    MAP("FrameRate",                    &p->frame_rate_code,            MAP_NUM, "FramerateCode, 1: 24000/1001,2: 24,3: 25(default), 4: 30000/1001,5: 30,6: 50,7: 60000/1001,8: 60");
    MAP("fps",                          &p->frame_rate,                 MAP_FLOAT, "Framerate, AVS2 supported value: 23.976(24000/1001), 24.0, 25.0(default), 29.97(30000/1001), 30.0, 50.0, 59.94(60000/1001), 60.0");
    MAP("ChromaFormat",                 &p->chroma_format,              MAP_NUM, "YUV format, 1=4:2:0 (default, the only supported format for the standard), 0=4:0:0, 2=4:2:2");
    MAP("InputSampleBitDepth",          &p->input_sample_bit_depth,     MAP_NUM, "Sample Bitdepth of input file");
    MAP("Frames",                       &p->num_frames,                 MAP_NUM, "Number of frames to be coded");
    MAP("FramesToBeEncoded",            &p->num_frames,                 MAP_NUM, "  - Same as `Frames`");

    /* output */
    MAP("output",                       &p->psz_bs_file,                MAP_STR, "Output bistream file path");
    MAP("OutputFile",                   &p->psz_bs_file,                MAP_STR, "  - Same as `output`");
    MAP("Recon",                        &p->psz_dump_yuv,               MAP_STR, "Output reconstruction YUV file path");
    MAP("ReconFile",                    &p->psz_dump_yuv,               MAP_STR, "  - Same as `Recon`");

    /* encoder configurations */
    MAP("MaxSizeInBit",                 &p->lcu_bit_level,              MAP_NUM, "Maximum Coding Unit (CU) Size (4, 5, 6)");
    MAP("MinSizeInBit",                 &p->scu_bit_level,              MAP_NUM, "Minimum Coding Unit (CU) Size (3, 4, 5, 6)");
    MAP("ProfileID",                    &p->profile_id,                 MAP_NUM, "Profile ID (18: MAIN PICTURE profile, 32: MAIN profile, 34: MAIN10 profile)");
    MAP("LevelID",                      &p->level_id,                   MAP_NUM, "Level ID   (16: 2.0;  32: 4.0;  34: 4.2;  64: 6.0;  66: 6.2)");
    MAP("SampleBitDepth",               &p->sample_bit_depth,           MAP_NUM, "Encoding bit-depth");
    MAP("IntraPeriodMax",               &p->intra_period_max,           MAP_NUM, "maximum intra-period, one I-frame mush appear in any NumMax of frames");
    MAP("IntraPeriodMin",               &p->intra_period_min,           MAP_NUM, "minimum intra-period, only one I-frame can appear in at most NumMin of frames");
    MAP("OpenGOP",                      &p->b_open_gop,                 MAP_NUM, "Open GOP or Closed GOP, 1: Open(default), 0: Closed");
    MAP("UseHadamard",                  &p->enable_hadamard,            MAP_NUM, "Hadamard transform (0=not used, 1=used)");
    MAP("FME",                          &p->me_method,                  MAP_NUM, "Motion Estimation method: 0-Full Search, 1-DIA, 2-HEX, 3-UMH (default), 4-TZ");
    MAP("SearchRange",                  &p->search_range,               MAP_NUM, "Max search range");
    MAP("NumberReferenceFrames",        &p->num_max_ref,                MAP_NUM, "Number of previous frames used for inter motion search (1-5)");

#if XAVS2_TRACE
    MAP("TraceFile",                    &p->psz_trace_file,             MAP_STR, "Tracing file path");
#endif
    MAP("TemporalIdExistFlag",          &p->temporal_id_exist_flag,     MAP_NUM, "temporal ID");
    MAP("FFRAMEEnable",                 &p->enable_f_frame,             MAP_NUM, "Use F Frame or not (0: Don't use F frames  1:Use F frames instead of P frames)");
    MAP("DHPEnable",                    &p->enable_dhp,                 MAP_NUM, "(0: Don't use DHP,      1:Use DHP)");
    MAP("MHPSKIPEnable",                &p->enable_mhp_skip,            MAP_NUM, "(0: Don't use MH_PSKIP, 1:Use MH_PSKIP)");
    MAP("WSMEnable",                    &p->enable_wsm,                 MAP_NUM, "(0: Don't use WSM,      1:Use WSM)");
    MAP("NumberBFrames",                &p->num_bframes,          MAP_NUM, "Number of B frames inserted between I/P/F frames (0=not used)");
    MAP("Inter2PU" ,                    &p->inter_2pu,                  MAP_NUM, "inter partition mode 2NxN or Nx2N or AMP");
    MAP("InterAMP",                     &p->enable_amp,                 MAP_NUM, "inter partition mode AMP");
    MAP("IntraInInter",                 &p->enable_intra,               MAP_NUM, "intra partition in inter frame");
    MAP("RdoLevel",                     &p->i_rd_level,                 MAP_NUM, "RD-optimized mode decision (0:off, 1: only for best partition mode of one CU, 2: only for best 2 partition modes; 3: All partition modes)");
    MAP("LoopFilterDisable",            &p->loop_filter_disable,        MAP_NUM, "Disable loop filter in picture header (0=Filter, 1=No Filter)");
    MAP("LoopFilterParameter",          &p->loop_filter_parameter_flag, MAP_NUM, "Send loop filter parameter (0= No parameter, 1= Send Parameter)");
    MAP("LoopFilterAlphaOffset",        &p->alpha_c_offset,             MAP_NUM, "Aplha offset in loop filter");
    MAP("LoopFilterBetaOffset",         &p->beta_offset,                MAP_NUM, "Beta offset in loop filter");
    MAP("SAOEnable",                    &p->enable_sao,                 MAP_NUM, "Enable SAO or not (1: on, 0: off)");
    MAP("ALFEnable",                    &p->enable_alf,                 MAP_NUM, "Enable ALF or not (1: on, 0: off)");
    MAP("ALFLowLatencyEncodingEnable",  &p->alf_LowLatencyEncoding,     MAP_NUM, "Enable Low Latency ALF (1=Low Latency mode, 0=High Efficiency mode)");
    MAP("CrossSliceLoopFilter",         &p->b_cross_slice_loop_filter,  MAP_NUM, "Enable Cross Slice Boundary Filter (0=Disable, 1=Enable)");

    /* ³¡±àÂë²ÎÊý */
    // MAP("InterlaceCodingOption",        &p->InterlaceCodingOption,      MAP_NUM);
    // MAP("RepeatFirstField",             &p->repeat_first_field,         MAP_NUM);
    // MAP("TopFieldFirst",                &p->top_field_first,            MAP_NUM);
    // MAP("OutputMergedPicture",          &p->output_merged_picture,      MAP_NUM);
    // MAP("Progressive_sequence",         &p->progressive_sequence,       MAP_NUM);
    // MAP("Progressive_frame",            &p->progressive_frame,          MAP_NUM);

    /* extension configuration */
    // MAP("TDMode",                       &p->TD_mode,                    MAP_NUM);
    // MAP("ViewPackingMode",              &p->view_packing_mode,          MAP_NUM);
    // MAP("ViewReverse",                  &p->view_reverse,               MAP_NUM);

    MAP("WQEnable",                     &p->enable_wquant,              MAP_NUM, "Weighted quantization");
#if XAVS2_TRACE && ENABLE_WQUANT
    MAP("SeqWQM",                       &p->SeqWQM,                     MAP_NUM);
    MAP("SeqWQFile",                    &p->psz_seq_wq_file,            MAP_STR);
    MAP("PicWQEnable",                  &p->PicWQEnable,                MAP_NUM);
    MAP("WQParam",                      &p->WQParam,                    MAP_NUM);
    MAP("WQModel",                      &p->WQModel,                    MAP_NUM);
    MAP("WeightParamDetailed",          &p->WeightParamDetailed,        MAP_STR);
    MAP("WeightParamUnDetailed",        &p->WeightParamUnDetailed,      MAP_STR);
    MAP("ChromaDeltaQPDisable",         &p->chroma_quant_param_disable, MAP_NUM);
    MAP("ChromaDeltaU",                 &p->chroma_quant_param_delta_u, MAP_NUM);
    MAP("ChromaDeltaV",                 &p->chroma_quant_param_delta_v, MAP_NUM);
    MAP("PicWQDataIndex",               &p->PicWQDataIndex,             MAP_NUM);
    MAP("PicWQFile",                    &p->psz_pic_wq_file,            MAP_STR);
#endif

    MAP("RdoqLevel",                    &p->i_rdoq_level,               MAP_NUM, "Rdoq Level (0: off, 1: cu level, only for best partition mode, 2: all mode)");
    MAP("LambdaFactor",                 &p->lambda_factor_rdoq,         MAP_NUM, "default: 75,  Rdoq Lambda factor");
    MAP("LambdaFactorP",                &p->lambda_factor_rdoq_p,       MAP_NUM, "default: 120, Rdoq Lambda factor P/F frame");
    MAP("LambdaFactorB",                &p->lambda_factor_rdoq_b,       MAP_NUM, "default: 100, Rdoq Lambda factor B frame");

    MAP("PMVREnable",                   &p->enable_pmvr,                MAP_NUM, "PMVR");
    MAP("NSQT",                         &p->enable_nsqt,                MAP_NUM, "NSQT");
    MAP("SDIP",                         &p->enable_sdip,                MAP_NUM, "SDIP");
    MAP("SECTEnable",                   &p->enable_secT,                MAP_NUM, "Secondary Transform");
    MAP("TDRDOEnable",                  &p->enable_tdrdo,               MAP_NUM, "TDRDO, only for LDP configuration (without B frames)");
    MAP("RefineQP",                     &p->enable_refine_qp,           MAP_NUM, "Refined QP, only for RA configuration (with B frames)");

    MAP("RateControl",                  &p->i_rc_method,                MAP_NUM, "0: CQP, 1: CBR (frame level), 2: CBR (SCU level), 3: VBR");
    MAP("TargetBitRate",                &p->i_target_bitrate,           MAP_NUM, "target bitrate, in bps");
    MAP("QP",                           &p->i_initial_qp,               MAP_NUM, "initial qp for first frame (8bit: 0~63; 10bit: 0~79)");
    MAP("InitialQP",                    &p->i_initial_qp,               MAP_NUM, "  - Same as `QP`");
    MAP("QPIFrame",                     &p->i_initial_qp,               MAP_NUM, "  - Same as `QP`");
    MAP("MinQP",                        &p->i_min_qp,                   MAP_NUM, "min qp (8bit: 0~63; 10bit: 0~79)");
    MAP("MaxQP",                        &p->i_max_qp,                   MAP_NUM, "max qp (8bit: 0~63; 10bit: 0~79)");

    MAP("GopSize",                      &p->i_gop_size,                 MAP_NUM, "sub GOP size (negative numbers indicating an employ of default settings, which will invliadate the following settings.)");
    MAP("PresetLevel",                  &p->preset_level,               MAP_NUM, "preset level for tradeoff between speed and performance, ordered from fastest to slowest (0, ..., 9), default: 5");
    MAP("Preset",                       &p->preset_level,               MAP_NUM, "  - Same as `PresetLevel`");

    MAP("SliceNum",                     &p->slice_num,                  MAP_NUM, "Number of slices for each frame");

    MAP("NumParallelGop",               &p->num_parallel_gop,           MAP_NUM, "number of parallel GOPs (0,1: no GOP parallelization)");
    MAP("ThreadFrames",                 &p->i_frame_threads,            MAP_NUM, "number of parallel threads for frames ( 0: auto )");
    MAP("ThreadRows",                   &p->i_lcurow_threads,           MAP_NUM, "number of parallel threads for rows   ( 0: auto )");
    MAP("EnableAecThread",              &p->enable_aec_thread,          MAP_NUM, "Enable AEC thread or not (default: enabled)");

    MAP("LogLevel",                     &p->i_log_level,                MAP_NUM, "log level: -1: none, 0: error, 1: warning, 2: info, 3: debug");
    MAP("Log",                          &p->i_log_level,                MAP_NUM, "  - Same as `LogLevel`");
    MAP("EnablePSNR",                   &p->enable_psnr,                MAP_NUM, "Enable PSNR or not (default: Enable)");
    MAP("EnableSSIM",                   &p->enable_ssim,                MAP_NUM, "Enable SSIM or not (default: Disabled)");

    /* end mapping */
    MAP("", NULL, MAP_END, "")
}


/**
 * ===========================================================================
 * function defines
 * ===========================================================================
 */

/**
 * ---------------------------------------------------------------------------
 * Function : allocate memory buffer, and read contents from file.
 * Params   : cfg_file - name of the file to be read.
 * Return   : file content buffer, or NULL on error.
 * Remarks  : the content buffer size is two times of file size so more
 *          : additive config can be appended, and the file size must be
 *          : less than 2M bytes.
 * ---------------------------------------------------------------------------
 */
static
char *GetConfigFileContent(char *file_content, int file_buf_size, const char *cfg_file)
{
    FILE *f_cfg;
    int file_len;

    /* open file */
    if ((f_cfg = fopen(cfg_file, "rb")) == NULL) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Cannot open configuration file %s.\n", cfg_file);
        return NULL;
    }

    /* get the file size */
    if (fseek(f_cfg, 0, SEEK_END) != 0) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Cannot fseek in configuration file %s.\n", cfg_file);
        fclose(f_cfg);
        return NULL;
    }
    file_len = (int)ftell(f_cfg);
    if (file_len < 0 || file_len > 2 * 1024 * 1024) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Unreasonable file size (%d) reported by ftell for configuration file %s.\n", file_len, cfg_file);
        fclose(f_cfg);
        return NULL;
    }
    if (fseek(f_cfg, 0, SEEK_SET) != 0) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Cannot fseek in configuration file %s.\n", cfg_file);
        fclose(f_cfg);
        return NULL;
    }

    if (file_len + 16 >= file_buf_size) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Too large configuration file: \"%s\", size %d Bytes\n",
                  cfg_file, file_len);
        file_len = file_buf_size - 16;
    }

    /* read file to buffer
     * Note that ftell() gives us the file size as the file system sees it.
     * The actual file size, as reported by fread() below will be often smaller
     * due to CR/LF to CR conversion and/or control characters after the dos
     * EOF marker in the file. */
    file_len = (int)fread(file_content, 1, file_len, f_cfg);
    file_content[file_len++] = '\n';
    file_content[file_len++] = '\0';

    /* close file */
    fclose(f_cfg);

    return file_content;
}

/* ---------------------------------------------------------------------------
 */
static intptr_t ParseRefContent(xavs2_param_t *param, char **buf)
{
    char header[10] = { 'F', 'r', 'a', 'm', 'e', '\0', '\0', '\0', '\0', '\0' };
    char str[4];
    char *token;
    const char *colon = ":";
    char **p = buf;
    xavs2_rps_t *tmp;
    int i = 1;
    int j;
    int i_gop_size = 0;
    int predict;

    sprintf(str, "%d", i);
    strcat(header, str);
    strcat(header, colon);

    memset(param->cfg_ref_all, -1, XAVS2_MAX_GOPS * sizeof(xavs2_rps_t));

    while (0 == strcmp(header, *p++)) {
        i_gop_size++;
        tmp   = param->cfg_ref_all + i - 1;
        token = *p++;
        tmp->poc              = atoi(token);
        token = *p++;
        tmp->qp_offset        = atoi(token);
        token = *p++;
        tmp->num_of_ref       = atoi(token);
        token = *p++;
        tmp->referd_by_others = atoi(token);
        for (j = 0; j < tmp->num_of_ref; j++) {
            token = *p++;
            tmp->ref_pic[j]       = atoi(token);
        }

        token = *p++;
        predict               = atoi(token);
        if (predict != 0) {
            token = *p++;
            j /* delta_rps */     = atoi(token);    /* delta_rps, not used */
        }

        token = *p++;
        tmp->num_to_rm        = atoi(token);
        for (j = 0; j < tmp->num_to_rm; j++) {
            token = *p++;
            tmp->rm_pic[j]        = atoi(token);
        }

        if (param->temporal_id_exist_flag == 1) {
            token = *p++;
            tmp->temporal_id      = atoi(token);
        }

        header[5] = header[6] = header[7] = header[8] = header[9] = '\0';
        sprintf(str, "%d", ++i);
        strcat(header, str);
        strcat(header, colon);
    }

    if (param->i_gop_size > 0 && param->i_gop_size != i_gop_size) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "gop_size set error.\n");
    }

    if (i_gop_size > XAVS2_MAX_GOP_SIZE) {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "gop_size set error (must <= %d).\n", XAVS2_MAX_GOP_SIZE);
    }

    return (p - buf - 1);
}

/* ---------------------------------------------------------------------------
 */
static INLINE
int ParameterNameToMapIndex(xavs2_param_map_t *p_map_tab, const char *param_name)
{
    mapping_t *map_tab = p_map_tab->map_tab;
    int i = 0;

    while (map_tab[i].name[0] != '\0') {  // ÖÕÖ¹Î»ÖÃÊÇ¿Õ×Ö·û´®
        if (xavs2_param_match(map_tab[i].name, param_name)) {
            return i;
        } else {
            i++;
        }
    }

    return -1;
}

/* ---------------------------------------------------------------------------
 */
static INLINE
void get_param_name(char *name, const char *param_item)
{
    char *str;
    name[0] = '\0';

    str = strtok(param_item, "_");

    while (str) {
        strcat(name, (const char *)str);
        str = strtok(NULL, "_");
    }
}

/* ---------------------------------------------------------------------------
 */
static INLINE
int xavs2e_atoi(const char *str, int *b_error)
{
    char *end;
    int v = strtol(str, &end, 0);

    if (end == str || end == NULL) {
        *b_error = 1;
        return v;
    }

    switch (*end) {
    case 'k':
        v *= 1000;
        break;
    case 'm':
        v *= 1000000;
        break;
    case '\0':
        break;
    default:
        *b_error = 1;  // un-known charaters
        break;
    }

    return v;
}

/* ---------------------------------------------------------------------------
 */
static INLINE
float xavs2e_atof(const char* str, int *b_error)
{
    char *end;
    float v = strtof(str, &end);

    if (end == str || *end != '\0') {
        *b_error = 1;
    }
    return v;
}

/*---------------------------------------------------------------------------
 */
static
char *copy_parameter(char *dst, const char *src)
{
    while (*src != '\0') {
        if (*src == '=') {
            /* the parser expects whitespace before & after '=' */
            *dst++ = ' ';
            *dst++ = '=';
            *dst++ = ' ';
        } else {
            *dst++ = *src;
        }

        src++;
    }
    *dst++ = ' ';     // add a space to support multiple config items
    return dst;
}

/**
 * ---------------------------------------------------------------------------
 * Function : get contents from config file and command line
 * Params   : argc - argument counter
 *          : argv - argument viscera, an array of null-terminated strings
 * Return   : file content buffer, or NULL on error
 * Remarks  : the content buffer size is two times of file size so more
 *          : additive config can be appended, and the file size must be
 *          : less than 2M bytes
 * ---------------------------------------------------------------------------
 */
static char *xavs2_get_configs(int argc, const char * const *argv)
{
    const int size_file_max = 1 << 20; // 1MB
    char item[4096];
    char *dst;
    char *cfg_content  = (char *)xavs2_malloc(2 * size_file_max);
    char *file_content = (char *)xavs2_malloc(size_file_max);
    int  item_len;
    int  num;           /* number of parameters */
    int  i;

    /* config file is the first parameter */
    if (cfg_content == NULL || file_content == NULL) {
        return NULL;
    }
    cfg_content[0] = '\0';

    /* parse the rest command line */
    for (i = 1; i < argc;) {
        if (0 == strncmp(argv[i], "-f", 2)) {   /* a new configuration file */
            GetConfigFileContent(file_content, size_file_max, argv[i + 1]);
            strcat(cfg_content, file_content);
            i += 2;
        } else if (argv[i][0] == '-' && argv[i][1] == '-') { // "--Parameter=XXX" style
            dst = copy_parameter(item, argv[i] + 2);
            /* add \n for each item */
            *dst++ = '\n';
            *dst = '\0';
            xavs2_log(NULL, XAVS2_LOG_DEBUG, "Adding cmd-line string 1: %s", item);
            /* append this item to the cfg_content */
            strcat(cfg_content, item);
            i++;
        } else if (0 == strncmp(argv[i], "-p", 2)) {   /* a config change? */
            /* collect all data until next parameter (starting with -<x>
             * (x is any character)), and append it to cfg_content */
            i++;
            item_len = 0;

            /* determine the necessary size for current item */
            for (num = i; num < argc && argv[num][0] != '-'; num++) {
                /* calculate the length for all the strings of current item */
                item_len += (int)(strlen(argv[num]));
            }

            /* additional bytes for spaces and \0s */
            item_len = ((item_len + 128) >> 4) << 4;

            item[0] = '\0';
            dst = item;

            /* concatenate all parameters identified before */
            while (i < num) {
                dst = copy_parameter(dst, argv[i]);
                i++;
            }

            /* add \n for each item */
            *dst++ = '\n';
            *dst = '\0';

            xavs2_log(NULL, XAVS2_LOG_DEBUG, "Adding cmd-line string 0: %s", item);

            /* append this item to the cfg_content */
            strcat(cfg_content, item);
        } else {
            xavs2_log(NULL, XAVS2_LOG_WARNING, "Invalid parameter style, argc %d, around string '%s'\n", i, argv[i]);
            xavs2_free(cfg_content);
            cfg_content = NULL;
            break;
        }
    }

    xavs2_free(file_content);
    return cfg_content;
}

/**
 * ---------------------------------------------------------------------------
 * Function   : Parsing encoding parameters
 * Parameters :
 *      [in ] : param - pointer to struct xavs2_param_t
 *      [in ] : argc  - number of command line parameters
 *      [in ] : argv  - pointer to parameter strings
 *      [out] : int   - zero for success, otherwise failed
 * Return     : none
 * ---------------------------------------------------------------------------
 */
int
xavs2_encoder_opt_set(xavs2_param_t *param, int argc, char *argv[])
{
    char *items[MAX_ITEMS];
    char *contents;
    char *p;
    char *bufend;
    char  name[64];
    int   map_index;
    int   item = 0;
    int   in_string = 0;
    int   in_item = 0;
    int   i;

    if ((contents = xavs2_get_configs(argc, argv)) == NULL) {
        fprintf(stderr, "get contents from configure file error.");
        return -1;
    }

    p = contents;
    bufend = &contents[strlen(contents)];

    /* alloc memory for mapping table and initialize the table */
    memset(&g_param_map, 0, sizeof(g_param_map));
    mapping_default(&g_param_map, param);

    /* generate an argc/argv-type list in items[], without comments and whitespace.
     * this is context insensitive and could be done most easily with lex(1). */
    while (p < bufend) {
        switch (*p) {
        case '#':           // found comment
            *p = '\0';      // replace '#' with '\0' in case of comment immediately following integer or string
            while (*p != '\n' && p < bufend) {
                p++;        // skip till EOL or EOF, whichever comes first
            }
            in_string = 0;
            in_item = 0;
            break;
        case '\r':  // case 13
        case '\n':
            in_item = 0;
            in_string = 0;
            *p++ = '\0';
            break;
        case ' ':
        case '\t':          // skip whitespace, leave state unchanged
            if (in_string) {
                p++;
            } else {
                *p++ = '\0';// terminate non-strings once whitespace is found
                in_item = 0;
            }
            break;
        case '\'':
        case '\"':           // begin/end of string
            *p++ = '\0';
            if (!in_string) {
                items[item++] = p;
                in_item = ~in_item;
            } else {
                in_item = 0;
            }
            in_string = ~in_string;   // toggle
            break;
        default:
            if (!in_item) {
                items[item++] = p;
                in_item = ~in_item;
            }
            p++;
        }
    }

    for (i = 0; i < item; i += 3) {
        get_param_name(name, items[i]);

        if (0 == strcmp(name, "Frame1:")) {
            i += (int)ParseRefContent(param, &items[i]);
            get_param_name(name, items[i]);
        }

        if (i + 2 >= item) {
            xavs2_log(NULL, XAVS2_LOG_ERROR, "Parsing error in the last parameter: %s.\n", items[i]);
            break;
        }

        if (strcmp("=", items[i + 1])) {
            xavs2_log(NULL, XAVS2_LOG_ERROR, "Parsing error in config file: '=' expected as the second token in each line.\n");
            return -1;
        }

        if (xavs2_encoder_opt_set2(param, name, items[i + 2]) < 0) {
            xavs2_log(NULL, XAVS2_LOG_WARNING, "Parameter Name not recognized: '%s'.\n", items[i]);
            continue;   // do not exit, continue to parse
        }
    }

    fflush(stdout);
    fflush(stderr);
    xavs2_free(contents);

    return 0;
}



/**
 * ---------------------------------------------------------------------------
 * Function   : Parsing encoding parameters
 * Parameters :
 *      [in ] : param - pointer to struct xavs2_param_t
 *      [in ] : name  - name of parameter
 *      [in ] : value_string - parameter value
 *      [in ] : value_i      - when value_string is null, use this value
 * Return     : int   - zero for success, otherwise failed
 * ---------------------------------------------------------------------------
 */
int
xavs2_encoder_opt_set2(xavs2_param_t *param, const char *name, const char *value_string)
{
    int map_index;
    int b_error = 0;

    if (g_param_map.param != param) {
        /* alloc memory for mapping table and initialize the table */
        memset(&g_param_map, 0, sizeof(g_param_map));
        mapping_default(&g_param_map, param);
    }

    if ((map_index = ParameterNameToMapIndex(&g_param_map, name)) >= 0) {
        int item_value;
        float val_float;

        switch (g_param_map.map_tab[map_index].type) {
        case MAP_NUM:   // numerical
            item_value = xavs2e_atoi(value_string, &b_error);
            if (b_error) {
                xavs2_log(NULL, XAVS2_LOG_ERROR, " Parsing error: Expected numerical value for Parameter of %s, found '%s'.\n",
                          name, value_string);
                return -1;
            }
            *(int *)(g_param_map.map_tab[map_index].addr) = item_value;
            if (xavs2_param_match(name, "preset_level") || xavs2_param_match(name, "presetlevel") || xavs2_param_match(name, "preset")) {
                parse_preset_level(param, param->preset_level);
            }
            if (xavs2_param_match(name, "FrameRate")) {
                xavs2_log(NULL, XAVS2_LOG_ERROR, " deprecated parameter: %s = %s\n",
                          name, value_string);
                if (item_value > 8 || item_value < 1) {
                    xavs2_log(NULL, XAVS2_LOG_ERROR, "FrameRate should be in 1..8 (1: 24000/1001,2: 24,3: 25,4: 30000/1001,5: 30,6: 50,7: 60000/1001,8: 60)\n");
                    return -1;
                }
                param->frame_rate = FRAME_RATE[param->frame_rate_code - 1];
            }
            // fprintf(stdout, ".");
            break;
        case MAP_FLOAT:  // float
            val_float = xavs2e_atof(value_string, &b_error);
            if (b_error) {
                xavs2_log(NULL, XAVS2_LOG_ERROR, " Parsing error: Expected float value for Parameter of %s, found '%s'.\n",
                          name, value_string);
                return -1;
            }
            *(float *)(g_param_map.map_tab[map_index].addr) = val_float;
            break;
        case MAP_FLAG:
            item_value = xavs2e_atoi(value_string, &b_error);
            if (b_error) {
                xavs2_log(NULL, XAVS2_LOG_ERROR, " Parsing error: Expected numerical value for Parameter of %s, found '%s'.\n",
                          name, value_string);
                return -1;
            }
            *(bool_t *)(g_param_map.map_tab[map_index].addr) = (bool_t)(!!item_value);
            // fprintf(stdout, ".");
            break;
        case MAP_STR:   // string
            strcpy((char *)g_param_map.map_tab[map_index].addr, value_string);
            // fprintf(stdout, ".");
            break;
        default:
            xavs2_log(NULL, XAVS2_LOG_ERROR, "Unknown value type in the map definition of config file.\n");
            return -1;
            break;
        }
    } else if (xavs2_param_match(name, "threads")) {
        param->i_lcurow_threads = xavs2e_atoi(value_string, &b_error);
        param->i_frame_threads  = 0;
    } else if (xavs2_param_match(name, "bframes")) {
        int value_i = xavs2e_atoi(value_string, &b_error);
        if (value_i > 0) {
            param->i_gop_size = value_i < 4 ? -4 : -8;
            param->num_bframes = XAVS2_ABS(param->i_gop_size) - 1;
            param->b_open_gop = 0;
        } else {
            param->num_bframes = 0;
            param->i_gop_size = -4;
            param->b_open_gop = 0;
        }
    } else if (xavs2_param_match(name, "bitdepth")) {
        int value_i = xavs2e_atoi(value_string, &b_error);
        param->input_sample_bit_depth = value_i;
        param->sample_bit_depth       = value_i;
    }

    return 0;
}


/**
 * ---------------------------------------------------------------------------
 * Function   : get value of a specific parameter
 * Parameters :
 *      [in ] : param   - pointer to struct xavs2_param_t
 *      [in ] : name    - name of a parameter
 * Return     : const char *: value string
 * ---------------------------------------------------------------------------
 */
const char *
xavs2_encoder_opt_get(xavs2_param_t *param, const char *name)
{
    static char buf[64];

    if (xavs2_param_match(name, "input")) {
        return param->psz_in_file;
    } else if (xavs2_param_match(name, "output")) {
        return param->psz_bs_file;
    } else if (xavs2_param_match(name, "width")) {
        sprintf(buf, "%d", param->org_width);
        return buf;
    } else if (xavs2_param_match(name, "height")) {
        sprintf(buf, "%d", param->org_height);
        return buf;
    } else if (xavs2_param_match(name, "frames")) {
        sprintf(buf, "%d", param->num_frames);
        return buf;
    } else if (xavs2_param_match(name, "BitDepth")) {
        sprintf(buf, "%d", param->sample_bit_depth);
        return buf;
    } else if (xavs2_param_match(name, "SampleShift")) {
        sprintf(buf, "%d", param->sample_bit_depth - param->input_sample_bit_depth);
        return buf;
    }

    return NULL;
}

/**
 * ---------------------------------------------------------------------------
 * Function   : Output help parameters
 * Parameters :
 *      [in ] : param - pointer to struct xavs2_param_t
 *      [out] : none
 * Return     : none
 * ---------------------------------------------------------------------------
 */
void
xavs2_encoder_opt_help(void)
{
    mapping_t *p_map = NULL;
    xavs2_param_t param;
    xavs2_log(NULL, XAVS2_LOG_INFO, "Usage:\n\t [-f EncoderFile.cfg] [-p ParameterName=Value] [--ParameterName=value]\n");
    xavs2_log(NULL, XAVS2_LOG_INFO, "Supported parameters:\n");

    memset(&g_param_map, 0, sizeof(g_param_map));
    mapping_default(&g_param_map, &param);

    p_map = g_param_map.map_tab;
    while (p_map != NULL) {
        if (p_map->addr == NULL) {
            break;
        }

        xavs2_log(NULL, XAVS2_LOG_INFO, "    %-20s : %s\n", p_map->name, p_map->s_instruction);
        p_map++;
    }
}
