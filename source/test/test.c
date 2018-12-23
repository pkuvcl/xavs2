/*
 * test.c
 *
 * Description of this file:
 *    Main function
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
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

#if defined(_MSC_VER)
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#include <pthread.h>
#endif

#include "xavs2.h"

/* ---------------------------------------------------------------------------
 */
static FILE *g_infile  = NULL;
static FILE *g_outfile = NULL;
const xavs2_api_t *g_api = NULL;

/* ---------------------------------------------------------------------------
 */
static void dump_encoded_data(void *coder, xavs2_outpacket_t *packet)
{
    if (g_outfile) {
        if (packet->state == XAVS2_STATE_ENCODED) {
            fwrite(packet->stream, packet->len, 1, g_outfile);
        } else if (packet->state == XAVS2_STATE_FLUSH_END) {
            fwrite(packet->stream, packet->len, 1, g_outfile);
        }
        g_api->encoder_packet_unref(coder, packet);
    }
}

/* ---------------------------------------------------------------------------
 * read one frame data from file line by line
 */
static int read_one_frame(xavs2_image_t *img, int shift_in)
{
    int k, j;
    if (img->in_sample_size != img->enc_sample_size) {
        static uint8_t p_buffer[16 * 1024];

        for (k = 0; k < img->i_plane; k++) {
            int i_width  = img->i_width[k];
            int i_stride = img->i_stride[k];

            if (img->in_sample_size == 1) {
                for (j = 0; j < img->i_lines[k]; j++) {
                    uint16_t *p_plane = (uint16_t *)&img->img_planes[k][j * i_stride];
                    int i;
                    if (fread(p_buffer, i_width, 1, g_infile) != 1) {
                        return -1;
                    }
                    memset(p_plane, 0, i_stride);
                    for (i = 0; i < i_width; i++) {
                        p_plane[i] = p_buffer[i] << shift_in;
                    }
                }
            } else {
                printf("Not supported high bit-depth for reading\n");
                return -1;
            }
        }
    } else {
        for (k = 0; k < img->i_plane; k++) {
            int size_line = img->i_width[k] * img->in_sample_size;
            for (j = 0; j < img->i_lines[k]; j++) {
                if (fread(img->img_planes[k] + img->i_stride[k] * j, size_line, 1, g_infile) != 1) {
                    return -1;
                }
            }
        }
    }
    return 0;
}

int test_encoder(const xavs2_api_t *api, xavs2_param_t *param)
{
    const char *in_file = api->opt_get(param, "input");
    const char *bs_file = api->opt_get(param, "output");
    const int shift_in  = atoi(api->opt_get(param, "SampleShift"));
    int num_frames      = atoi(api->opt_get(param, "frames"));
    xavs2_picture_t pic;
    void *encoder = NULL;
    int k;
    xavs2_outpacket_t packet = {0};

    /* open input & output files */
    if ((g_infile = fopen(in_file, "rb")) == NULL) {
        fprintf(stderr, "error opening input file: \"%s\"\n", in_file);
        return -1;
    }

    if ((g_outfile = fopen(bs_file, "wb")) == NULL) {
        fprintf(stderr, "error opening output file: \"%s\"\n", bs_file);
        fclose(g_infile);

        return -1;
    }

    if (num_frames == 0) {
        num_frames = 1 << 30;
    }

    /* create the xavs2 video encoder */
    encoder = api->encoder_create(param);

    if (encoder == NULL) {
        fprintf(stderr, "Error: Can not create encoder. Null pointer returned.\n");
        fclose(g_infile);
        fclose(g_outfile);

        return -1;
    }

    /* read frame data and send to the xavs2 video encoder */
    for (k = 0; k < num_frames; k++) {
        if (api->encoder_get_buffer(encoder, &pic) < 0) {
            fprintf(stderr, "failed to get frame buffer [%3d,%3d].\n", k, num_frames);
            break;
        }

        if (read_one_frame(&pic.img, shift_in) < 0) {
            fprintf(stderr, "failed to read one YUV frame [%3d/%3d]\n", k, num_frames);
            /* return the buffer to the encoder */
            pic.i_state = XAVS2_STATE_NO_DATA;

            api->encoder_encode(encoder, &pic, &packet);
            dump_encoded_data(encoder, &packet);
            break;
        }

        pic.i_state = 0;
        pic.i_type  = XAVS2_TYPE_AUTO;
        pic.i_pts   = k;

        api->encoder_encode(encoder, &pic, &packet);
        dump_encoded_data(encoder, &packet);
    }

    /* flush delayed frames */
    for (; packet.state != XAVS2_STATE_FLUSH_END;) {
        api->encoder_encode(encoder, NULL, &packet);
        dump_encoded_data(encoder, &packet);
    }

    /* destroy the encoder */
    api->encoder_destroy(encoder);

    return 0;
}

/* ---------------------------------------------------------------------------
 */
const xavs2_api_t *load_xavs2_library(int argc, char **argv, xavs2_param_t **p_param)
{
    const xavs2_api_t *api = NULL;
    /* encoding parameters */
    int guess_bit_depth;
    xavs2_param_t *param = NULL;

    /* get API handler */
    for (guess_bit_depth = 8; guess_bit_depth <= 10; guess_bit_depth += 2) {
        int enc_bit_depth = 0;
        api = xavs2_api_get(guess_bit_depth);

        if (api == NULL) {
            continue;
        }

        fprintf(stdout, "xavs2 lib loaded: version %s %d-bit\n",
                api->s_version_source, api->internal_bit_depth);
        fflush(stdout);

        if (argc < 2) {
            api->opt_help(); /* at lease one additional parameter needed */
            return NULL;
        }

        /* parse parameters and modify the parameters */
        param = api->opt_alloc();
        if (api->opt_set(param, argc, argv) < 0) {
            fprintf(stdout, "parse contents error.\n");
            return NULL;
        }

        enc_bit_depth = atoi(api->opt_get(param, "BitDepth"));
        if (enc_bit_depth == api->internal_bit_depth) {
            fprintf(stdout, "using xavs2 lib: version %s %d-bit success\n",
                    api->s_version_source, api->internal_bit_depth);
            break;
        }

        fprintf(stdout, "Incompatible encoding bit-depth to library: %d-bit lib for encoding %d-bit source\n",
                api->internal_bit_depth, enc_bit_depth);

        /* free spaces */
        api->opt_destroy(param);
        param = NULL;
        api = NULL;
    }

    *p_param = param;
    return api;
}

/* ---------------------------------------------------------------------------
 */
int main(int argc, char **argv)
{
    /* encoding parameters */
    xavs2_param_t *param = NULL;
    int ret;

    /* get API handler */
    g_api = load_xavs2_library(argc, argv, &param);

    if (g_api == NULL) {
        fprintf(stdout, "xavs2 lib load error\n");
        return -1;
    }
    fflush(NULL);    // flush all output streams

    /* test encoding */
    ret = test_encoder(g_api, param);

    /* free spaces */
    g_api->opt_destroy(param);

    return ret;
}
