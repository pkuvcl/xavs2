/*
 * xavs2.h
 *
 * Description of this file:
 *    API interface of the xavs2 encoder library
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

#ifndef XAVS2_XAVS2_H
#define XAVS2_XAVS2_H

#ifdef __cplusplus
extern "C" {    // only need to export C interface if used by C++ source code
#endif

#if !defined(_STDINT_H) && !defined(_STDINT_H_) && !defined(_STDINT_H_INCLUDED) && !defined(_STDINT) &&\
    !defined(_SYS_STDINT_H_) && !defined(_INTTYPES_H) && !defined(_INTTYPES_H_) && !defined(_INTTYPES)
# ifdef _MSC_VER
#  pragma message("You must include stdint.h or inttypes.h before xavs2.h")
# else
#  warning You must include stdint.h or inttypes.h before xavs2.h
# endif
#endif

/* xavs2 encoder build version, means different API interface
 * (10 * VER_MAJOR + VER_MINOR) */
#define XAVS2_BUILD                10
/* xavs2 encoder API version, version 2 the non-caller version */
#define XAVS2_API_VERSION          1

/**
 * ===========================================================================
 * define XAVS2_API
 * ===========================================================================
 */
#ifdef XAVS2_EXPORTS
#  ifdef __GNUC__                     /* for Linux  */
#    if __GNUC__ >= 4
#      define XAVS2_API __attribute__((visibility("default")))
#    else
#      define XAVS2_API __attribute__((dllexport))
#    endif
#  else                               /* for windows */
#    define XAVS2_API __declspec(dllexport)
#  endif
#else
#  ifdef __GNUC__                     /* for Linux   */
#    define XAVS2_API
#  else                               /* for windows */
#    define XAVS2_API __declspec(dllimport)
#  endif
#endif


/**
 * ===========================================================================
 * const defines
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * state defines
 */
#define XAVS2_UNDEFINE        0
#define XAVS2_STATE_NO_DATA   1     /* no bitstream data */
#define XAVS2_STATE_ENCODED   2     /* one frame has been encoded */
#define XAVS2_STATE_FLUSH_END 9     /* flush end */
#define XAVS2_FLUSH           99    /* flush (fetch bitstream data only) */

/* ---------------------------------------------------------------------------
 * slice type
 */
#define XAVS2_TYPE_AUTO       0     /* Let xavs2 encoder choose the right type */
#define XAVS2_TYPE_IDR        1
#define XAVS2_TYPE_I          2
#define XAVS2_TYPE_P          3
#define XAVS2_TYPE_F          4
#define XAVS2_TYPE_B          5
#define XAVS2_TYPE_KEYFRAME   6     /* IDR or I depending on b_open_gop option */
#define XAVS2_TYPE_G          7
#define XAVS2_TYPE_GB         8

/* ---------------------------------------------------------------------------
 * color space type
 */
#define XAVS2_CSP_MASK        0x00ff  /* mask */
#define XAVS2_CSP_NONE        0x0000  /* invalid mode     */
#define XAVS2_CSP_I420        0x0001  /* yuv 4:2:0 planar */
#define XAVS2_CSP_YV12        0x0002  /* yvu 4:2:0 planar */
#define XAVS2_CSP_NV12        0x0003  /* yuv 4:2:0, with one y plane and one packed u+v */
#define XAVS2_CSP_MAX         0x0004  /* end of list */
#define XAVS2_CSP_VFLIP       0x1000  /* the csp is vertically flipped */
#define XAVS2_CSP_HIGH_DEPTH  0x2000  /* the csp has a depth of 16 bits per pixel component */

/* ---------------------------------------------------------------------------
 * log level
 */
enum log_level_e {
    XAVS2_LOG_NONE     = -1,  /* none */
    XAVS2_LOG_ERROR    = 0,   /* level 0 */
    XAVS2_LOG_WARNING  = 1,   /* level 1 */
    XAVS2_LOG_INFO     = 2,   /* level 2 */
    XAVS2_LOG_DEBUG    = 3,   /* level 3 */
    XAVS2_LOG_NOPREFIX = 0x80 /* special using: without prefix */
};

/* ---------------------------------------------------------------------------
 * others
 */


/**
 * ===========================================================================
 * interface struct type defines
 * ===========================================================================
 */

/* -----------------------------
 * xavs2 encoder input parameters
 *
 * For version safety you may use 
 *    xavs2_encoder_opt_alloc(), xavs2_encoder_opt_destroy()
 * to manage the allocation of xavs2_param_t instances, and 
 *    xavs2_encoder_opt_set(), xavs2_encoder_opt_set2()
 * to assign values by name, and 
 *    xavs2_encoder_opt_get(p, "param_name")
 * to get specific configuration value string.
 * 
 * Just treat xavs2_param_t as an opaque data structure */
typedef struct xavs2_param_t  xavs2_param_t;


/* ---------------------------------------------------------------------------
 * xavs2_image_t
 */
typedef struct xavs2_image_t {
    int      i_csp;                    /* color space */
    int      in_sample_size;           /* input sample size in byte */
    int      enc_sample_size;          /* encoding sample size in byte */
    int      i_plane;                  /* number of image planes */
    int      i_width[3];               /* widths  for each plane */
    int      i_lines[3];               /* heights for each plane */
    int      i_stride[3];              /* strides for each plane */
    uint8_t *img_planes[4];            /* pointers to each plane (planes[3]: start buffer address) */
} xavs2_image_t;


/* ---------------------------------------------------------------------------
 * xavs2_picture_t
 */
typedef struct xavs2_picture_t {
    /* [IN ]    flush or not
     * [OUT]    encoding state */
    int         i_state;
    /* [IN ]    force picture type (if not auto)
     *          if xavs2 encoder encoding parameters are violated in the forcing of picture
     *          types, xavs2 encoder will correct the input picture type and log a warning.
     *          the quality of frame type decisions may suffer if a great deal of
     *          fine-grained mixing of auto and forced frametypes is done
     * [OUT]    type of the picture encoded */
    int         i_type;
    /* [IN ]    force quantizer for != XAVS2_QP_AUTO */
    int         i_qpplus1;
    /* [OUT]    whether this frame is a keyframe. important when using modes that
     *          result in SEI recovery points being used instead of IDR frames */
    int         b_keyframe;
    /* [IN ]    user pts. [OUT]: pts of encoded picture (user) */
    int64_t     i_pts;
    /* [OUT]    frame dts. When the pts of the first frame is close to zero,
     *          initial frames may have a negative dts which must be dealt
     *          with by any muxer */
    int64_t     i_dts;
    /* [IN ]    raw data */
    xavs2_image_t  img;
    /* [IN ]    private pointer, DO NOT change it */
    void       *priv;
} xavs2_picture_t;

/* ---------------------------------------------------------------------------
 * xavs2_outpacket_t
 */
typedef struct xavs2_outpacket_t {
    void          *private_data;      /* private pointer, DONOT change it */
    const uint8_t *stream;            /* pointer to bitstream data buffer */
    int            len;               /* length  of bitstream data */
    int            state;             /* state of current frame encoded */
    int            type;              /* type  of current frame encoded */
    int64_t        pts;               /* pts   of current frame encoded */
    int64_t        dts;               /* dts   of current frame encoded */
    void           *opaque;           /* pointer to user data */
} xavs2_outpacket_t;

#if XAVS2_API_VERSION < 2
/* ---------------------------------------------------------------------------
 * type define of dump function
 */
typedef void(*xavs2_dump_func_t)(
    void *coder,                      /* handler of the encoder */
    xavs2_outpacket_t *packet);       /* output bitstream */
#endif

/**
 * ===========================================================================
 * interface function declares: parameters
 * ===========================================================================
 */

/**
 * ---------------------------------------------------------------------------
 * Function   : get buffer for the encoder caller
 * Parameters :
 *      [in ] : coder - pointer to wrapper of the xavs2 encoder
 *            : pic   - pointer to struct xavs2_picture_t
 *      [out] : none
 * Return     : zero for success, otherwise failed
 * ---------------------------------------------------------------------------
 */
XAVS2_API int
xavs2_encoder_get_buffer(void *coder, xavs2_picture_t *pic);

/**
 * ---------------------------------------------------------------------------
 * Function   : Output help parameters
 * Parameters :
 *      [in ] : param - pointer to struct xavs2_param_t
 *      [out] : none
 * Return     : none
 * ---------------------------------------------------------------------------
 */
XAVS2_API void
xavs2_encoder_opt_help(void);

/**
 * ---------------------------------------------------------------------------
 * Function   : initialize default parameters for the xavs2 video encoder
 * Parameters :
 *      [in ] : none
 * Return     : parameter handler, can be further configured
 * ---------------------------------------------------------------------------
 */
XAVS2_API xavs2_param_t *
xavs2_encoder_opt_alloc(void);

/**
 * ---------------------------------------------------------------------------
 * Function   : Parsing encoding parameters
 * Parameters :
 *      [in ] : param - pointer to struct xavs2_param_t
 *      [in ] : argc  - number of command line parameters
 *      [in ] : argv  - pointer to parameter strings
 * Return     : int   - zero for success, otherwise failed
 * ---------------------------------------------------------------------------
 */
XAVS2_API int
xavs2_encoder_opt_set(xavs2_param_t *param, int argc, char *argv[]);

/**
 * ---------------------------------------------------------------------------
 * Function   : set parameter value
 * Parameters :
 *      [in ] : param - pointer to struct xavs2_param_t
 *      [in ] : name  - name of parameter
 *      [in ] : value_string - parameter value
 * Return     : int   - zero for success, otherwise failed
 * ---------------------------------------------------------------------------
 */
XAVS2_API int
xavs2_encoder_opt_set2(xavs2_param_t *param, const char *name, const char *value_string);

/**
 * ---------------------------------------------------------------------------
 * Function   : get value of a specific parameter
 * Parameters :
 *      [in ] : param - pointer to struct xavs2_param_t
 *      [in ] : name  - name of a parameter (input, output, width, height, frames)
 * Return     : const char *: value string
 * ---------------------------------------------------------------------------
 */
XAVS2_API const char *
xavs2_encoder_opt_get(xavs2_param_t *param, const char *name);

/**
 * ---------------------------------------------------------------------------
 * Function   : free memory of parameter
 * Parameters :
 *      [in ] : none
 *      [out] : parameter handler, can be further configured
 * Return     : none
 * ---------------------------------------------------------------------------
 */
XAVS2_API void
xavs2_encoder_opt_destroy(xavs2_param_t *param);

/**
 * ===========================================================================
 * interface function declares: encoding
 * ===========================================================================
 */

/**
 * ---------------------------------------------------------------------------
 * Function   : create and initialize the xavs2 video encoder
 * Parameters :
 *      [in ] : param     - pointer to struct xavs2_param_t
 *            : dump_func - pointer to struct xavs2_dump_func_t
 *            : opaque    - user data
 *      [out] : none
 * Return     : handle of xavs2 encoder wrapper, none zero for success, otherwise false
 * ---------------------------------------------------------------------------
 */
XAVS2_API void *
#if XAVS2_API_VERSION < 2
xavs2_encoder_create(xavs2_param_t *param, xavs2_dump_func_t dump_func, void *opaque);
#else
xavs2_encoder_create(xavs2_param_t *param);
#endif

/**
 * ---------------------------------------------------------------------------
 * Function   : write (send) data to the xavs2 encoder
 * Parameters :
 *      [in ] : coder - pointer to wrapper of the xavs2 encoder
 *            : pic   - pointer to struct xavs2_picture_t
 *      [out] : none
 * Return     : zero for success, otherwise failed
 * ---------------------------------------------------------------------------
 */
XAVS2_API int
#if XAVS2_API_VERSION < 2
xavs2_encoder_encode(void *coder, xavs2_picture_t *pic);
#else
xavs2_encoder_encode(void *coder, xavs2_picture_t *pic, xavs2_outpacket_t *packet);
#endif

/**
 * ---------------------------------------------------------------------------
 * Function   : label a packet to be recycled
 * Parameters :
 *      [in ] : coder    - pointer to wrapper of the xavs2 encoder
 *            : packet   - pointer to struct xavs2_outpacket_t
 *      [out] : none
 * Return     : zero for success, otherwise failed
 * ---------------------------------------------------------------------------
 */
XAVS2_API int
xavs2_encoder_packet_unref(void *coder, xavs2_outpacket_t *packet);


/**
 * ---------------------------------------------------------------------------
 * Function   : destroy the xavs2 video encoder
 * Parameters :
 *      [in ] : coder - pointer to wrapper of the xavs2 encoder
 *      [out] : none
 * Return     : none
 * Note       : this API is *NOT* thread-safe, 
 *              and can not be called simultaneously with other APIs.
 * ---------------------------------------------------------------------------
 */
XAVS2_API void
xavs2_encoder_destroy(void *coder);

#ifdef __cplusplus
}
#endif
#endif // XAVS2_XAVS2_H
