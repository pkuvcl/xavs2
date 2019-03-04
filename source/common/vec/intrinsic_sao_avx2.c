/*
 * intrinsic_sao_avx2.c
 *
 * Description of this file:
 *    AVX2 assembly functions of SAO module of the xavs2 library
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

#include <mmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#include "../common.h"
#include "intrinsic.h"
#include "../filter.h"

/* ---------------------------------------------------------------------------
 */
void SAO_on_block_sse256(pel_t *p_dst, int i_dst, pel_t *p_src, int i_src, int i_block_w, int i_block_h,
                         int *lcu_avail, SAOBlkParam *sao_param)
{
    int start_x, end_x, start_y, end_y;
    int start_x_r0, end_x_r0, start_x_r, end_x_r, start_x_rn, end_x_rn;
    int x, y;
    // __m128i clipMin = _mm_setzero_si128();

    assert(sao_param->typeIdc != SAO_TYPE_OFF);

    switch (sao_param->typeIdc) {
    case SAO_TYPE_EO_0: {
        __m256i off;
        __m256i s0, s1, s2;
        __m256i t0, t1, t2, t3, t4, etype;
        __m128i mask, offtmp;
        __m256i c2 = _mm256_set1_epi8(2);

        int end_x_32;

        offtmp = _mm_loadu_si128((__m128i*)sao_param->offset);
        offtmp = _mm_packs_epi32(offtmp, _mm_set_epi32(0, 0, 0, sao_param->offset[4]));
        offtmp = _mm_packs_epi16(offtmp, _mm_setzero_si128());

        off = _mm256_castsi128_si256(offtmp);
        off = _mm256_inserti128_si256(off, offtmp, 1);

        start_x = lcu_avail[SAO_L] ? 0 : 1;
        end_x = lcu_avail[SAO_R] ? i_block_w : (i_block_w - 1);
        end_x_32 = end_x - ((end_x - start_x) & 0x1f);

        for (y = 0; y < i_block_h; y++) {
            for (x = start_x; x < end_x; x += 32) {
                s0 = _mm256_lddqu_si256((__m256i*)&p_src[x - 1]);
                s1 = _mm256_loadu_si256((__m256i*)&p_src[x]);
                s2 = _mm256_loadu_si256((__m256i*)&p_src[x + 1]);

                t3 = _mm256_min_epu8(s0, s1);
                t1 = _mm256_cmpeq_epi8(t3, s0);
                t2 = _mm256_cmpeq_epi8(t3, s1);
                t0 = _mm256_subs_epi8(t2, t1); //leftsign

                t3 = _mm256_min_epu8(s1, s2);
                t1 = _mm256_cmpeq_epi8(t3, s1);
                t2 = _mm256_cmpeq_epi8(t3, s2);
                t3 = _mm256_subs_epi8(t1, t2); //rightsign

                etype = _mm256_adds_epi8(t0, t3);

                etype = _mm256_adds_epi8(etype, c2);//edgetype=left + right +2

                t0 = _mm256_shuffle_epi8(off, etype);//get offset

                //convert byte to short for possible overflow
                t1 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(t0));
                t2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(t0, 1));
                t3 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(s1));
                t4 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(s1, 1));

                t1 = _mm256_adds_epi16(t1, t3);
                t2 = _mm256_adds_epi16(t2, t4);
                t0 = _mm256_packus_epi16(t1, t2); //saturated
                t0 = _mm256_permute4x64_epi64(t0, 0xd8);

                if (x != end_x_32) {
                    _mm256_storeu_si256((__m256i*)(p_dst + x), t0);
                } else {
                    if (end_x - x >= 16) {
                        _mm_storeu_si128((__m128i*)(p_dst + x), _mm256_castsi256_si128(t0));
                        if (end_x - x > 16) {
                            mask = _mm_loadu_si128((__m128i*)(intrinsic_mask[end_x - end_x_32 - 17]));
                            _mm_maskmoveu_si128(_mm256_extracti128_si256(t0, 1), mask, (char*)(p_dst + x + 16));
                        }
                    } else {
                        mask = _mm_load_si128((__m128i*)(intrinsic_mask[end_x - end_x_32 - 1]));
                        _mm_maskmoveu_si128(_mm256_castsi256_si128(t0), mask, (char*)(p_dst + x));
                    }
                    break;
                }
            }
            p_dst += i_dst;
            p_src += i_src;
        }
    }
    break;

    case SAO_TYPE_EO_90: {
        __m256i off;
        __m256i s0, s1, s2;
        __m256i t0, t1, t2, t3, t4, etype;
        __m128i mask, offtmp;
        __m256i c2 = _mm256_set1_epi8(2);
        int end_x_32 = i_block_w - (i_block_w & 0x1f);

        offtmp = _mm_loadu_si128((__m128i*)sao_param->offset);
        offtmp = _mm_packs_epi32(offtmp, _mm_set_epi32(0, 0, 0, sao_param->offset[4]));
        offtmp = _mm_packs_epi16(offtmp, _mm_setzero_si128());

        off = _mm256_castsi128_si256(offtmp);
        off = _mm256_inserti128_si256(off, offtmp, 1);

        start_y = lcu_avail[SAO_T] ? 0 : 1;
        end_y = lcu_avail[SAO_D] ? i_block_h : (i_block_h - 1);

        p_dst += start_y * i_dst;
        p_src += start_y * i_src;

        for (y = start_y; y < end_y; y++) {
            for (x = 0; x < i_block_w; x += 32) {
                s0 = _mm256_lddqu_si256((__m256i*)&p_src[x - i_src]);
                s1 = _mm256_lddqu_si256((__m256i*)&p_src[x]);
                s2 = _mm256_lddqu_si256((__m256i*)&p_src[x + i_src]);

                t3 = _mm256_min_epu8(s0, s1);
                t1 = _mm256_cmpeq_epi8(t3, s0);
                t2 = _mm256_cmpeq_epi8(t3, s1);
                t0 = _mm256_subs_epi8(t2, t1); //leftsign

                t3 = _mm256_min_epu8(s1, s2);
                t1 = _mm256_cmpeq_epi8(t3, s1);
                t2 = _mm256_cmpeq_epi8(t3, s2);
                t3 = _mm256_subs_epi8(t1, t2); //rightsign

                etype = _mm256_adds_epi8(t0, t3); //edgetype

                etype = _mm256_adds_epi8(etype, c2);

                t0 = _mm256_shuffle_epi8(off, etype);//get offset

                //convert byte to short for possible overflow
                t1 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(t0));
                t2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(t0, 1));
                t3 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(s1));
                t4 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(s1, 1));

                t1 = _mm256_adds_epi16(t1, t3);
                t2 = _mm256_adds_epi16(t2, t4);
                t0 = _mm256_packus_epi16(t1, t2); //saturated
                t0 = _mm256_permute4x64_epi64(t0, 0xd8);

                if (x != end_x_32) {
                    _mm256_storeu_si256((__m256i*)(p_dst + x), t0);
                } else {
                    if (i_block_w - x >= 16) {
                        _mm_storeu_si128((__m128i*)(p_dst + x), _mm256_castsi256_si128(t0));
                        if (i_block_w - x > 16) {
                            mask = _mm_loadu_si128((__m128i*)(intrinsic_mask[i_block_w - end_x_32 - 17]));
                            _mm_maskmoveu_si128(_mm256_extracti128_si256(t0, 1), mask, (char*)(p_dst + x + 16));
                        }
                    } else {
                        mask = _mm_load_si128((__m128i*)(intrinsic_mask[i_block_w - end_x_32 - 1]));
                        _mm_maskmoveu_si128(_mm256_castsi256_si128(t0), mask, (char*)(p_dst + x));
                    }
                    break;
                }
            }
            p_dst += i_dst;
            p_src += i_src;
        }
    }
    break;

    case SAO_TYPE_EO_135: {
        __m256i off;
        __m256i s0, s1, s2;
        __m256i t0, t1, t2, t3, t4, etype;
        __m128i mask, offtmp;
        __m256i c2 = _mm256_set1_epi8(2);
        int end_x_r0_32, end_x_r_32, end_x_rn_32;

        offtmp = _mm_loadu_si128((__m128i*)sao_param->offset);
        offtmp = _mm_packs_epi32(offtmp, _mm_set_epi32(0, 0, 0, sao_param->offset[4]));
        offtmp = _mm_packs_epi16(offtmp, _mm_setzero_si128());

        off = _mm256_castsi128_si256(offtmp);
        off = _mm256_inserti128_si256(off, offtmp, 1);

        //first row
        start_x_r0 = lcu_avail[SAO_TL] ? 0 : 1;
        end_x_r0 = lcu_avail[SAO_T] ? (lcu_avail[SAO_R] ? i_block_w : (i_block_w - 1)) : 1;
        end_x_r0_32 = end_x_r0 - ((end_x_r0 - start_x_r0) & 0x1f);
        for (x = start_x_r0; x < end_x_r0; x += 32) {
            s0 = _mm256_loadu_si256((__m256i*)&p_src[x - i_src - 1]);
            s1 = _mm256_loadu_si256((__m256i*)&p_src[x]);
            s2 = _mm256_loadu_si256((__m256i*)&p_src[x + i_src + 1]);

            t3 = _mm256_min_epu8(s0, s1);
            t1 = _mm256_cmpeq_epi8(t3, s0);
            t2 = _mm256_cmpeq_epi8(t3, s1);
            t0 = _mm256_subs_epi8(t2, t1); //upsign

            t3 = _mm256_min_epu8(s1, s2);
            t1 = _mm256_cmpeq_epi8(t3, s1);
            t2 = _mm256_cmpeq_epi8(t3, s2);
            t3 = _mm256_subs_epi8(t1, t2); //downsign

            etype = _mm256_adds_epi8(t0, t3); //edgetype
            etype = _mm256_adds_epi8(etype, c2);

            t0 = _mm256_shuffle_epi8(off, etype);//get offset
            //convert byte to short for possible overflow
            t1 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(t0));
            t2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(t0, 1));
            t3 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(s1));
            t4 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(s1, 1));

            t1 = _mm256_adds_epi16(t1, t3);
            t2 = _mm256_adds_epi16(t2, t4);
            t0 = _mm256_packus_epi16(t1, t2); //saturated
            t0 = _mm256_permute4x64_epi64(t0, 0xd8);

            if (x != end_x_r0_32) {
                _mm256_storeu_si256((__m256i*)(p_dst + x), t0);
            } else {
                if (end_x_r0 - x >= 16) {
                    _mm_storeu_si128((__m128i*)(p_dst + x), _mm256_castsi256_si128(t0));
                    if (end_x_r0 - x > 16) {
                        mask = _mm_loadu_si128((__m128i*)intrinsic_mask[end_x_r0 - end_x_r0_32 - 17]);
                        _mm_maskmoveu_si128(_mm256_extracti128_si256(t0, 1), mask, (char*)(p_dst + x + 16));
                    }
                } else {
                    mask = _mm_load_si128((__m128i*)(intrinsic_mask[end_x_r0 - end_x_r0_32 - 1]));
                    _mm_maskmoveu_si128(_mm256_castsi256_si128(t0), mask, (char*)(p_dst + x));
                }
                break;
            }
        }
        p_dst += i_dst;
        p_src += i_src;

        //middle rows
        start_x_r = lcu_avail[SAO_L] ? 0 : 1;
        end_x_r = lcu_avail[SAO_R] ? i_block_w : (i_block_w - 1);
        end_x_r_32 = end_x_r - ((end_x_r - start_x_r) & 0x1f);
        for (y = 1; y < i_block_h - 1; y++) {
            for (x = start_x_r; x < end_x_r; x += 32) {
                s0 = _mm256_loadu_si256((__m256i*)&p_src[x - i_src - 1]);
                s1 = _mm256_loadu_si256((__m256i*)&p_src[x]);
                s2 = _mm256_loadu_si256((__m256i*)&p_src[x + i_src + 1]);

                t3 = _mm256_min_epu8(s0, s1);
                t1 = _mm256_cmpeq_epi8(t3, s0);
                t2 = _mm256_cmpeq_epi8(t3, s1);
                t0 = _mm256_subs_epi8(t2, t1); //upsign

                t3 = _mm256_min_epu8(s1, s2);
                t1 = _mm256_cmpeq_epi8(t3, s1);
                t2 = _mm256_cmpeq_epi8(t3, s2);
                t3 = _mm256_subs_epi8(t1, t2); //downsign

                etype = _mm256_adds_epi8(t0, t3); //edgetype

                etype = _mm256_adds_epi8(etype, c2);

                t0 = _mm256_shuffle_epi8(off, etype);//get offset

                //convert byte to short for possible overflow
                t1 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(t0));
                t2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(t0, 1));
                t3 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(s1));
                t4 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(s1, 1));

                t1 = _mm256_adds_epi16(t1, t3);
                t2 = _mm256_adds_epi16(t2, t4);
                t0 = _mm256_packus_epi16(t1, t2); //saturated
                t0 = _mm256_permute4x64_epi64(t0, 0xd8);

                if (x != end_x_r_32) {
                    _mm256_storeu_si256((__m256i*)(p_dst + x), t0);
                } else {
                    if (end_x_r - x >= 16) {
                        _mm_storeu_si128((__m128i*)(p_dst + x), _mm256_castsi256_si128(t0));
                        if (end_x_r - x > 16) {
                            mask = _mm_load_si128((__m128i*)(intrinsic_mask[end_x_r - end_x_r_32 - 17]));
                            _mm_maskmoveu_si128(_mm256_extracti128_si256(t0, 1), mask, (char*)(p_dst + x + 16));
                        }
                    } else {
                        mask = _mm_load_si128((__m128i*)(intrinsic_mask[end_x_r - end_x_r_32 - 1]));
                        _mm_maskmoveu_si128(_mm256_castsi256_si128(t0), mask, (char*)(p_dst + x));
                    }
                    break;
                }
            }
            p_dst += i_dst;
            p_src += i_src;
        }
        //last row
        start_x_rn = lcu_avail[SAO_D] ? (lcu_avail[SAO_L] ? 0 : 1) : (i_block_w - 1);
        end_x_rn = lcu_avail[SAO_DR] ? i_block_w : (i_block_w - 1);
        end_x_rn_32 = end_x_rn - ((end_x_rn - start_x_rn) & 0x1f);
        for (x = start_x_rn; x < end_x_rn; x += 32) {
            s0 = _mm256_loadu_si256((__m256i*)&p_src[x - i_src - 1]);
            s1 = _mm256_loadu_si256((__m256i*)&p_src[x]);
            s2 = _mm256_loadu_si256((__m256i*)&p_src[x + i_src + 1]);

            t3 = _mm256_min_epu8(s0, s1);
            t1 = _mm256_cmpeq_epi8(t3, s0);
            t2 = _mm256_cmpeq_epi8(t3, s1);
            t0 = _mm256_subs_epi8(t2, t1); //upsign

            t3 = _mm256_min_epu8(s1, s2);
            t1 = _mm256_cmpeq_epi8(t3, s1);
            t2 = _mm256_cmpeq_epi8(t3, s2);
            t3 = _mm256_subs_epi8(t1, t2); //downsign

            etype = _mm256_adds_epi8(t0, t3); //edgetype

            etype = _mm256_adds_epi8(etype, c2);

            t0 = _mm256_shuffle_epi8(off, etype);//get offset

            //convert byte to short for possible overflow
            t1 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(t0));
            t2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(t0, 1));
            t3 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(s1));
            t4 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(s1, 1));

            t1 = _mm256_adds_epi16(t1, t3);
            t2 = _mm256_adds_epi16(t2, t4);
            t0 = _mm256_packus_epi16(t1, t2); //saturated
            t0 = _mm256_permute4x64_epi64(t0, 0xd8);

            if (x != end_x_rn_32) {
                _mm256_storeu_si256((__m256i*)(p_dst + x), t0);
            } else {
                if (end_x_rn - x >= 16) {
                    _mm_storeu_si128((__m128i*)(p_dst + x), _mm256_castsi256_si128(t0));
                    if (end_x_rn - x > 16) {
                        mask = _mm_load_si128((__m128i*)(intrinsic_mask[end_x_rn - end_x_rn_32 - 17]));
                        _mm_maskmoveu_si128(_mm256_extracti128_si256(t0, 1), mask, (char*)(p_dst + x + 16));
                    }
                } else {
                    mask = _mm_load_si128((__m128i*)(intrinsic_mask[end_x_rn - end_x_rn_32 - 1]));
                    _mm_maskmoveu_si128(_mm256_castsi256_si128(t0), mask, (char*)(p_dst + x));
                }
                break;
            }
        }
    }
    break;

    case SAO_TYPE_EO_45: {
        __m256i off;
        __m256i s0, s1, s2;
        __m256i t0, t1, t2, t3, t4, etype;
        __m128i mask, offtmp;
        __m256i c2 = _mm256_set1_epi8(2);
        int end_x_r0_32, end_x_r_32, end_x_rn_32;


        offtmp = _mm_loadu_si128((__m128i*)sao_param->offset);
        offtmp = _mm_packs_epi32(offtmp, _mm_set_epi32(0, 0, 0, sao_param->offset[4]));
        offtmp = _mm_packs_epi16(offtmp, _mm_setzero_si128());

        off = _mm256_castsi128_si256(offtmp);
        off = _mm256_inserti128_si256(off, offtmp, 1);

        start_x_r0 = lcu_avail[SAO_T] ? (lcu_avail[SAO_L] ? 0 : 1) : (i_block_w - 1);
        end_x_r0 = lcu_avail[SAO_TR] ? i_block_w : (i_block_w - 1);
        end_x_r0_32 = end_x_r0 - ((end_x_r0 - start_x_r0) & 0x1f);

        //first row
        for (x = start_x_r0; x < end_x_r0; x += 32) {
            s0 = _mm256_loadu_si256((__m256i*)&p_src[x - i_src + 1]);
            s1 = _mm256_loadu_si256((__m256i*)&p_src[x]);
            s2 = _mm256_loadu_si256((__m256i*)&p_src[x + i_src - 1]);

            t3 = _mm256_min_epu8(s0, s1);
            t1 = _mm256_cmpeq_epi8(t3, s0);
            t2 = _mm256_cmpeq_epi8(t3, s1);
            t0 = _mm256_subs_epi8(t2, t1); //upsign

            t3 = _mm256_min_epu8(s1, s2);
            t1 = _mm256_cmpeq_epi8(t3, s1);
            t2 = _mm256_cmpeq_epi8(t3, s2);
            t3 = _mm256_subs_epi8(t1, t2); //downsign

            etype = _mm256_adds_epi8(t0, t3); //edgetype

            etype = _mm256_adds_epi8(etype, c2);

            t0 = _mm256_shuffle_epi8(off, etype);//get offset

            //convert byte to short for possible overflow
            t1 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(t0));
            t2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(t0, 1));
            t3 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(s1));
            t4 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(s1, 1));

            t1 = _mm256_adds_epi16(t1, t3);
            t2 = _mm256_adds_epi16(t2, t4);
            t0 = _mm256_packus_epi16(t1, t2); //saturated
            t0 = _mm256_permute4x64_epi64(t0, 0xd8);

            if (x != end_x_r0_32) {
                _mm256_storeu_si256((__m256i*)(p_dst + x), t0);
            } else {
                if (end_x_r0 - x >= 16) {
                    _mm_storeu_si128((__m128i*)(p_dst + x), _mm256_castsi256_si128(t0));
                    if (end_x_r0 - x > 16) {
                        mask = _mm_load_si128((__m128i*)(intrinsic_mask[end_x_r0 - end_x_r0_32 - 17]));
                        _mm_maskmoveu_si128(_mm256_extracti128_si256(t0, 1), mask, (char*)(p_dst + x + 16));
                    }
                } else {
                    mask = _mm_load_si128((__m128i*)(intrinsic_mask[end_x_r0 - end_x_r0_32 - 1]));
                    _mm_maskmoveu_si128(_mm256_castsi256_si128(t0), mask, (char*)(p_dst + x));
                }
                break;
            }
        }
        p_dst += i_dst;
        p_src += i_src;

        //middle rows
        start_x_r = lcu_avail[SAO_L] ? 0 : 1;
        end_x_r = lcu_avail[SAO_R] ? i_block_w : (i_block_w - 1);
        end_x_r_32 = end_x_r - ((end_x_r - start_x_r) & 0x1f);
        for (y = 1; y < i_block_h - 1; y++) {
            for (x = start_x_r; x < end_x_r; x += 32) {
                s0 = _mm256_loadu_si256((__m256i*)&p_src[x - i_src + 1]);
                s1 = _mm256_loadu_si256((__m256i*)&p_src[x]);
                s2 = _mm256_loadu_si256((__m256i*)&p_src[x + i_src - 1]);

                t3 = _mm256_min_epu8(s0, s1);
                t1 = _mm256_cmpeq_epi8(t3, s0);
                t2 = _mm256_cmpeq_epi8(t3, s1);
                t0 = _mm256_subs_epi8(t2, t1); //upsign

                t3 = _mm256_min_epu8(s1, s2);
                t1 = _mm256_cmpeq_epi8(t3, s1);
                t2 = _mm256_cmpeq_epi8(t3, s2);
                t3 = _mm256_subs_epi8(t1, t2); //downsign

                etype = _mm256_adds_epi8(t0, t3); //edgetype

                etype = _mm256_adds_epi8(etype, c2);

                t0 = _mm256_shuffle_epi8(off, etype);//get offset

                //convert byte to short for possible overflow
                t1 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(t0));
                t2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(t0, 1));
                t3 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(s1));
                t4 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(s1, 1));

                t1 = _mm256_adds_epi16(t1, t3);
                t2 = _mm256_adds_epi16(t2, t4);
                t0 = _mm256_packus_epi16(t1, t2); //saturated
                t0 = _mm256_permute4x64_epi64(t0, 0xd8);

                if (x != end_x_r_32) {
                    _mm256_storeu_si256((__m256i*)(p_dst + x), t0);
                } else {
                    if (end_x_r - x >= 16) {
                        _mm_storeu_si128((__m128i*)(p_dst + x), _mm256_castsi256_si128(t0));
                        if (end_x_r - x > 16) {
                            mask = _mm_load_si128((__m128i*)(intrinsic_mask[end_x_r - end_x_r_32 - 17]));
                            _mm_maskmoveu_si128(_mm256_extracti128_si256(t0, 1), mask, (char*)(p_dst + x + 16));
                        }
                    } else {
                        mask = _mm_load_si128((__m128i*)(intrinsic_mask[end_x_r - end_x_r_32 - 1]));
                        _mm_maskmoveu_si128(_mm256_castsi256_si128(t0), mask, (char*)(p_dst + x));
                    }
                    break;
                }
            }
            p_dst += i_dst;
            p_src += i_src;
        }

        //last row
        start_x_rn = lcu_avail[SAO_DL] ? 0 : 1;
        end_x_rn = lcu_avail[SAO_D] ? (lcu_avail[SAO_R] ? i_block_w : (i_block_w - 1)) : 1;
        end_x_rn_32 = end_x_rn - ((end_x_rn - start_x_rn) & 0x1f);
        for (x = start_x_rn; x < end_x_rn; x += 32) {
            s0 = _mm256_loadu_si256((__m256i*)&p_src[x - i_src + 1]);
            s1 = _mm256_loadu_si256((__m256i*)&p_src[x]);
            s2 = _mm256_loadu_si256((__m256i*)&p_src[x + i_src - 1]);

            t3 = _mm256_min_epu8(s0, s1);
            t1 = _mm256_cmpeq_epi8(t3, s0);
            t2 = _mm256_cmpeq_epi8(t3, s1);
            t0 = _mm256_subs_epi8(t2, t1); //upsign

            t3 = _mm256_min_epu8(s1, s2);
            t1 = _mm256_cmpeq_epi8(t3, s1);
            t2 = _mm256_cmpeq_epi8(t3, s2);
            t3 = _mm256_subs_epi8(t1, t2); //downsign

            etype = _mm256_adds_epi8(t0, t3); //edgetype

            etype = _mm256_adds_epi8(etype, c2);

            t0 = _mm256_shuffle_epi8(off, etype);//get offset

            //convert byte to short for possible overflow
            t1 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(t0));
            t2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(t0, 1));
            t3 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(s1));
            t4 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(s1, 1));

            t1 = _mm256_adds_epi16(t1, t3);
            t2 = _mm256_adds_epi16(t2, t4);
            t0 = _mm256_packus_epi16(t1, t2); //saturated
            t0 = _mm256_permute4x64_epi64(t0, 0xd8);

            if (x != end_x_rn_32) {
                _mm256_storeu_si256((__m256i*)(p_dst + x), t0);
            } else {
                if (end_x_rn - x >= 16) {
                    _mm_storeu_si128((__m128i*)(p_dst + x), _mm256_castsi256_si128(t0));
                    if (end_x_rn - x > 16) {
                        mask = _mm_load_si128((__m128i*)(intrinsic_mask[end_x_rn - end_x_rn_32 - 17]));
                        _mm_maskmoveu_si128(_mm256_extracti128_si256(t0, 1), mask, (char*)(p_dst + x + 16));
                    }
                } else {
                    mask = _mm_load_si128((__m128i*)(intrinsic_mask[end_x_rn - end_x_rn_32 - 1]));
                    _mm_maskmoveu_si128(_mm256_castsi256_si128(t0), mask, (char*)(p_dst + x));
                }
                break;
            }
        }
    }
    break;

    case SAO_TYPE_BO: {
        __m256i r0, r1, r2, r3, off0, off1, off2, off3;
        __m256i t0, t1, t2, t3, t4, src0, src1;
        __m128i mask = _mm_setzero_si128();
        __m256i shift_mask = _mm256_set1_epi8(31);
        end_x = i_block_w;
        int end_x_32 = end_x - ((end_x - 0) & 0x1f);

        r0 = _mm256_set1_epi8((int8_t)(sao_param->startBand));
        r1 = _mm256_set1_epi8((int8_t)((sao_param->startBand + 1) & 31));
        r2 = _mm256_set1_epi8((int8_t)((sao_param->deltaBand + sao_param->startBand) & 31));
        r3 = _mm256_set1_epi8((int8_t)((sao_param->deltaBand + sao_param->startBand + 1) & 31));

        off0 = _mm256_set1_epi8((int8_t)sao_param->offset[sao_param->startBand]);
        off1 = _mm256_set1_epi8((int8_t)sao_param->offset[(sao_param->startBand + 1) & 31]);
        off2 = _mm256_set1_epi8((int8_t)sao_param->offset[(sao_param->deltaBand + sao_param->startBand) & 31]);
        off3 = _mm256_set1_epi8((int8_t)sao_param->offset[(sao_param->deltaBand + sao_param->startBand + 1) & 31]);

        for (y = 0; y < i_block_h; y++) {
            for (x = 0; x < i_block_w; x += 32) {
                src0 = _mm256_loadu_si256((__m256i*)&p_src[x]);
                src1 = _mm256_srli_epi16(src0, 3);
                src1 = _mm256_and_si256(src1, shift_mask);

                t0 = _mm256_cmpeq_epi8(src1, r0);
                t1 = _mm256_cmpeq_epi8(src1, r1);
                t2 = _mm256_cmpeq_epi8(src1, r2);
                t3 = _mm256_cmpeq_epi8(src1, r3);

                t0 = _mm256_and_si256(t0, off0);
                t1 = _mm256_and_si256(t1, off1);
                t2 = _mm256_and_si256(t2, off2);
                t3 = _mm256_and_si256(t3, off3);
                t0 = _mm256_or_si256(t0, t1);
                t2 = _mm256_or_si256(t2, t3);
                t0 = _mm256_or_si256(t0, t2);

                //convert byte to short for possible overflow
                t1 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(t0));
                t2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(t0, 1));
                t3 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(src0));
                t4 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(src0, 1));

                t1 = _mm256_add_epi16(t1, t3);
                t2 = _mm256_add_epi16(t2, t4);
                t0 = _mm256_packus_epi16(t1, t2); //saturated
                t0 = _mm256_permute4x64_epi64(t0, 0xd8);

                if (x < end_x_32) {
                    _mm256_storeu_si256((__m256i*)(p_dst + x), t0);
                } else {
                    if (end_x - x >= 16) {
                        _mm_storeu_si128((__m128i*)(p_dst + x), _mm256_castsi256_si128(t0));
                        if (end_x - x > 16) {
                            mask = _mm_loadu_si128((__m128i*)(intrinsic_mask[end_x - end_x_32 - 17]));
                            _mm_maskmoveu_si128(_mm256_extracti128_si256(t0, 1), mask, (char*)(p_dst + x + 16));
                        }
                    } else {
                        mask = _mm_load_si128((__m128i*)(intrinsic_mask[end_x - end_x_32 - 1]));
                        _mm_maskmoveu_si128(_mm256_castsi256_si128(t0), mask, (char*)(p_dst + x));
                    }
                    break;
                }
            }
            p_dst += i_dst;
            p_src += i_src;
        }
    }
    break;
    default: {
        xavs2_log(NULL, XAVS2_LOG_ERROR, "Invalid SAO type in %s.", __FUNCTION__);
        exit(-1);
    }
    }
}

