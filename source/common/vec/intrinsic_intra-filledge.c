/*
 * intrinsic_intra-fiiledge.c
 *
 * Description of this file:
 *   SSE assembly functions of Intra-Filledge module of the xavs2 library
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

#include "../avs2_defs.h"
#include "../basic_types.h"
#include "intrinsic.h"

#include <string.h>
#include <mmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>


/* ---------------------------------------------------------------------------
 * fill reference samples for intra prediction
 * LCU内在左边界上的PU
 */
void fill_edge_samples_0_sse128(const pel_t *pTL, int i_TL, const pel_t *pLcuEP, pel_t *EP, uint32_t i_avai, int bsx, int bsy)
{
    __m128i T0, T1;
    int i, k, j;
    int num_padding;

    UNUSED_PARAMETER(pTL);
    UNUSED_PARAMETER(i_TL);
    /* fill default value */
    k = ((bsy + bsx) << 1) + 1;
    j = (k >> 4) << 4;
    T0 = _mm_set1_epi8((uint8_t)g_dc_value);
    for (i = 0; i < j; i += 16) {
        _mm_storeu_si128((__m128i *)(&EP[-(bsy << 1)] + i), T0);
    }
    memset(&EP[-(bsy << 1)] + j, g_dc_value, k - j + 1);
    EP[2 * bsx] = (pel_t)g_dc_value;

    /* get prediction pixels ---------------------------------------
     * extra pixels          | left-down pixels   | left pixels   | top-left | top pixels  | top-right pixels  | extra pixels
     * -2*bsy-4 ... -2*bsy-1 | -bsy-bsy ... -bsy-1| -bsy -3 -2 -1 |     0    | 1 2 ... bsx | bsx+1 ... bsx+bsx | 2*bsx+1 ... 2*bsx+4
     */

    /* fill top & top-right pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        /* fill top pixels */
        if (bsx == 4) {
            memcpy(&EP[1], &pLcuEP[1], bsx * sizeof(pel_t));
        } else if (bsx == 8) {
            T1 = _mm_loadu_si128((__m128i *)&pLcuEP[1]);
            _mm_storel_epi64((__m128i *)&EP[1], T1);
        } else {
            for (i = 0; i < bsx; i += 16) {
                T1 = _mm_loadu_si128((__m128i *)(pLcuEP + i + 1));
                _mm_store_si128((__m128i *)(&EP[1] + i), T1);
            }
        }
    }

    /* fill top-right pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_RIGHT)) {
        if (bsx == 4) {
            memcpy(&EP[bsx + 1], &pLcuEP[bsx + 1], bsx * sizeof(pel_t));
        } else if (bsx == 8) {
            T1 = _mm_loadu_si128((__m128i *)&pLcuEP[bsx + 1]);
            _mm_storel_epi64((__m128i *)&EP[bsx + 1], T1);
        } else {
            for (i = 0; i < bsx; i += 16) {
                T1 = _mm_loadu_si128((__m128i *)(&pLcuEP[bsx + i + 1]));
                _mm_store_si128((__m128i *)(&EP[bsx + 1] + i), T1);
            }
        }
    } else {
        if (bsx == 4) {
            memset(&EP[bsx + 1], EP[bsx], bsx);
        } else if (bsx == 8) {
            T0 = _mm_set1_epi8(EP[bsx]);    // repeat the last pixel
            _mm_storel_epi64((__m128i *)&EP[bsx + 1], T0);
        } else {
            T0 = _mm_set1_epi8(EP[bsx]);    // repeat the last pixel
            for (i = 0; i < bsx; i += 16) {
                _mm_store_si128((__m128i *)(&EP[bsx + 1 + i]), T0);
            }
        }
    }

    /* fill extra pixels */
    num_padding = bsy * 11 / 4 - bsx + 4;
    if (num_padding > 0) {
        memset(&EP[2 * bsx + 1], EP[2 * bsx], num_padding); // from (2*bsx) to (iX + 3) = (bsy *11/4 + bsx - 1) + 3
    }

    /* fill left & left-down pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        /* fill left pixels */
        memcpy(&EP[-bsy], &pLcuEP[-bsy], bsy * sizeof(pel_t));
    }

    /* fill left-down pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT_DOWN)) {
        memcpy(&EP[-2 * bsy], &pLcuEP[-2 * bsy], bsy * sizeof(pel_t));
    } else {
        if (bsy == 4) {
            memset(&EP[-(bsy << 1)], EP[-bsy], bsy);
        } else if (bsy == 8) {
            T0 = _mm_set1_epi8(EP[-bsy]);
            _mm_storel_epi64((__m128i *)&EP[-(bsy << 1)], T0);
        } else {
            T0 = _mm_set1_epi8(EP[-bsy]);
            for (i = 0; i < bsy; i += 16) {
                _mm_storeu_si128((__m128i *)(&EP[-(bsy << 1)] + i), T0);
            }
        }
    }

    /* fill extra pixels */
    num_padding = bsx * 11 / 4 - bsy + 4;
    if (num_padding > 0) {
        memset(&EP[-2 * bsy - num_padding], EP[-2 * bsy], num_padding); // from (-2*bsy) to (-iY - 3) = -(bsx *11/4 + bsy - 1) - 3
    }

    /* fill EP[0] */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_LEFT)) {
        EP[0] = pLcuEP[0];
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        EP[0] = pLcuEP[1];
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        EP[0] = pLcuEP[-1];
    }
}

/* ---------------------------------------------------------------------------
 * fill reference samples for intra prediction
 * LCU内在左边界上的PU
 */
void fill_edge_samples_x_sse128(const pel_t *pTL, int i_TL, const pel_t *pLcuEP, pel_t *EP, uint32_t i_avai, int bsx, int bsy)
{
    __m128i T0, T1;
    int i, k, j;
    int num_padding;

    const pel_t *pL = pTL + i_TL;

    /* fill default value */
    k = ((bsy + bsx) << 1) + 1;
    j = (k >> 4) << 4;
    T0 = _mm_set1_epi8((uint8_t)g_dc_value);
    for (i = 0; i < j; i += 16) {
        _mm_storeu_si128((__m128i *)(&EP[-(bsy << 1)] + i), T0);
    }
    memset(&EP[-(bsy << 1)] + j, g_dc_value, k - j + 1);
    EP[2 * bsx] = (pel_t)g_dc_value;

    /* get prediction pixels ---------------------------------------
     * extra pixels          | left-down pixels   | left pixels   | top-left | top pixels  | top-right pixels  | extra pixels
     * -2*bsy-4 ... -2*bsy-1 | -bsy-bsy ... -bsy-1| -bsy -3 -2 -1 |     0    | 1 2 ... bsx | bsx+1 ... bsx+bsx | 2*bsx+1 ... 2*bsx+4
     */

    /* fill top & top-right pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        /* fill top pixels */
        if (bsx == 4) {
            memcpy(&EP[1], &pLcuEP[1], bsx * sizeof(pel_t));
        } else if (bsx == 8) {
            T1 = _mm_loadu_si128((__m128i *)&pLcuEP[1]);
            _mm_storel_epi64((__m128i *)&EP[1], T1);
        } else {
            for (i = 0; i < bsx; i += 16) {
                T1 = _mm_loadu_si128((__m128i *)(pLcuEP + i + 1));
                _mm_store_si128((__m128i *)(&EP[1] + i), T1);
            }
        }
    }

    /* fill top-right pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_RIGHT)) {
        if (bsx == 4) {
            memcpy(&EP[bsx + 1], &pLcuEP[bsx + 1], bsx * sizeof(pel_t));
        } else if (bsx == 8) {
            T1 = _mm_loadu_si128((__m128i *)&pLcuEP[bsx + 1]);
            _mm_storel_epi64((__m128i *)&EP[bsx + 1], T1);
        } else {
            for (i = 0; i < bsx; i += 16) {
                T1 = _mm_loadu_si128((__m128i *)(&pLcuEP[bsx + i + 1]));
                _mm_store_si128((__m128i *)(&EP[bsx + 1] + i), T1);
            }
        }
    } else {
        if (bsx == 4) {
            memset(&EP[bsx + 1], EP[bsx], bsx);
        } else if (bsx == 8) {
            T0 = _mm_set1_epi8(EP[bsx]);    // repeat the last pixel
            _mm_storel_epi64((__m128i *)&EP[bsx + 1], T0);
        } else {
            T0 = _mm_set1_epi8(EP[bsx]);    // repeat the last pixel
            for (i = 0; i < bsx; i += 16) {
                _mm_store_si128((__m128i *)(&EP[bsx + 1 + i]), T0);
            }
        }
    }

    /* fill extra pixels */
    num_padding = bsy * 11 / 4 - bsx + 4;
    if (num_padding > 0) {
        memset(&EP[2 * bsx + 1], EP[2 * bsx], num_padding); // from (2*bsx) to (iX + 3) = (bsy *11/4 + bsx - 1) + 3
    }

    /* fill left & left-down pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        const pel_t *p_l = pL;
        int y;
        /* fill left pixels */
        for (y = 0; y < bsy; y++) {
            EP[-1 - y] = *p_l;
            p_l += i_TL;
        }
    }

    /* fill left-down pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT_DOWN)) {
        const pel_t *p_l = pL + bsy * i_TL;
        int y;
        for (y = 0; y < bsy; y++) {
            EP[-bsy - 1 - y] = *p_l;
            p_l += i_TL;
        }
    } else {
        if (bsy == 4) {
            memset(&EP[-(bsy << 1)], EP[-bsy], bsy);
        } else if (bsy == 8) {
            T0 = _mm_set1_epi8(EP[-bsy]);
            _mm_storel_epi64((__m128i *)&EP[-(bsy << 1)], T0);
        } else {
            T0 = _mm_set1_epi8(EP[-bsy]);
            for (i = 0; i < bsy; i += 16) {
                _mm_storeu_si128((__m128i *)(&EP[-(bsy << 1)] + i), T0);
            }
        }
    }

    /* fill extra pixels */
    num_padding = bsx * 11 / 4 - bsy + 4;
    if (num_padding > 0) {
        memset(&EP[-2 * bsy - num_padding], EP[-2 * bsy], num_padding); // from (-2*bsy) to (-iY - 3) = -(bsx *11/4 + bsy - 1) - 3
    }

    /* fill EP[0] */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_LEFT)) {
        EP[0] = pLcuEP[0];
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        EP[0] = pLcuEP[1];
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        EP[0] = pL[0];
    }
}

/* ---------------------------------------------------------------------------
 * fill reference samples for intra prediction
 * LCU内在左边界上的PU
 */
void fill_edge_samples_y_sse128(const pel_t *pTL, int i_TL, const pel_t *pLcuEP, pel_t *EP, uint32_t i_avai, int bsx, int bsy)
{
    __m128i T0, T1;
    int i, k, j;
    int num_padding;

    const pel_t *pT = pTL + 1;
    UNUSED_PARAMETER(i_TL);

    /* fill default value */
    k = ((bsy + bsx) << 1) + 1;
    j = (k >> 4) << 4;
    T0 = _mm_set1_epi8((uint8_t)g_dc_value);
    for (i = 0; i < j; i += 16) {
        _mm_storeu_si128((__m128i *)(&EP[-(bsy << 1)] + i), T0);
    }
    memset(&EP[-(bsy << 1)] + j, g_dc_value, k - j + 1);
    EP[2 * bsx] = (pel_t)g_dc_value;

    /* get prediction pixels ---------------------------------------
     * extra pixels          | left-down pixels   | left pixels   | top-left | top pixels  | top-right pixels  | extra pixels
     * -2*bsy-4 ... -2*bsy-1 | -bsy-bsy ... -bsy-1| -bsy -3 -2 -1 |     0    | 1 2 ... bsx | bsx+1 ... bsx+bsx | 2*bsx+1 ... 2*bsx+4
     */

    /* fill top & top-right pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        /* fill top pixels */
        if (bsx == 4) {
            memcpy(&EP[1], pT, bsx * sizeof(pel_t));
        } else if (bsx == 8) {
            T1 = _mm_loadu_si128((__m128i *)pT);
            _mm_storel_epi64((__m128i *)&EP[1], T1);
        } else {
            for (i = 0; i < bsx; i += 16) {
                T1 = _mm_loadu_si128((__m128i *)(pT + i));
                _mm_store_si128((__m128i *)(&EP[1] + i), T1);
            }
        }
    }

    /* fill top-right pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_RIGHT)) {
        if (bsx == 4) {
            memcpy(&EP[bsx + 1], &pT[bsx], bsx * sizeof(pel_t));
        } else if (bsx == 8) {
            T1 = _mm_loadu_si128((__m128i *)&pT[bsx]);
            _mm_storel_epi64((__m128i *)&EP[bsx + 1], T1);
        } else {
            for (i = 0; i < bsx; i += 16) {
                T1 = _mm_loadu_si128((__m128i *)(&pT[bsx + i]));
                _mm_store_si128((__m128i *)(&EP[bsx + 1] + i), T1);
            }
        }
    } else {
        if (bsx == 4) {
            memset(&EP[bsx + 1], EP[bsx], bsx);
        } else if (bsx == 8) {
            T0 = _mm_set1_epi8(EP[bsx]);    // repeat the last pixel
            _mm_storel_epi64((__m128i *)&EP[bsx + 1], T0);
        } else {
            T0 = _mm_set1_epi8(EP[bsx]);    // repeat the last pixel
            for (i = 0; i < bsx; i += 16) {
                _mm_store_si128((__m128i *)(&EP[bsx + 1 + i]), T0);
            }
        }
    }

    /* fill extra pixels */
    num_padding = bsy * 11 / 4 - bsx + 4;
    if (num_padding > 0) {
        memset(&EP[2 * bsx + 1], EP[2 * bsx], num_padding); // from (2*bsx) to (iX + 3) = (bsy *11/4 + bsx - 1) + 3
    }

    /* fill left & left-down pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        /* fill left pixels */
        memcpy(&EP[-bsy], &pLcuEP[-bsy], bsy * sizeof(pel_t));
    }

    /* fill left-down pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT_DOWN)) {
        memcpy(&EP[-2 * bsy], &pLcuEP[-2 * bsy], bsy * sizeof(pel_t));
    } else {
        if (bsy == 4) {
            memset(&EP[-(bsy << 1)], EP[-bsy], bsy);
        } else if (bsy == 8) {
            T0 = _mm_set1_epi8(EP[-bsy]);
            _mm_storel_epi64((__m128i *)&EP[-(bsy << 1)], T0);
        } else {
            T0 = _mm_set1_epi8(EP[-bsy]);
            for (i = 0; i < bsy; i += 16) {
                _mm_storeu_si128((__m128i *)(&EP[-(bsy << 1)] + i), T0);
            }
        }
    }

    /* fill extra pixels */
    num_padding = bsx * 11 / 4 - bsy + 4;
    if (num_padding > 0) {
        memset(&EP[-2 * bsy - num_padding], EP[-2 * bsy], num_padding); // from (-2*bsy) to (-iY - 3) = -(bsx *11/4 + bsy - 1) - 3
    }

    /* fill EP[0] */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_LEFT)) {
        EP[0] = pLcuEP[0];
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        EP[0] = pT[0];
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        EP[0] = pLcuEP[-1];
    }
}

/* ---------------------------------------------------------------------------
 * fill reference samples for intra prediction
 * LCU内在左边界上的PU
 */
void fill_edge_samples_xy_sse128(const pel_t *pTL, int i_TL, const pel_t *pLcuEP, pel_t *EP, uint32_t i_avai, int bsx, int bsy)
{
    __m128i T0, T1;
    int i, k, j;
    int num_padding;

    const pel_t *pT = pTL + 1;
    const pel_t *pL = pTL + i_TL;

    UNUSED_PARAMETER(pLcuEP);
    /* fill default value */
    k = ((bsy + bsx) << 1) + 1;
    j = (k >> 4) << 4;
    T0 = _mm_set1_epi8((uint8_t)g_dc_value);
    for (i = 0; i < j; i += 16) {
        _mm_storeu_si128((__m128i *)(&EP[-(bsy << 1)] + i), T0);
    }
    memset(&EP[-(bsy << 1)] + j, g_dc_value, k - j + 1);
    EP[2 * bsx] = (pel_t)g_dc_value;

    /* get prediction pixels ---------------------------------------
     * extra pixels          | left-down pixels   | left pixels   | top-left | top pixels  | top-right pixels  | extra pixels
     * -2*bsy-4 ... -2*bsy-1 | -bsy-bsy ... -bsy-1| -bsy -3 -2 -1 |     0    | 1 2 ... bsx | bsx+1 ... bsx+bsx | 2*bsx+1 ... 2*bsx+4
     */

    /* fill top & top-right pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        /* fill top pixels */
        if (bsx == 4) {
            memcpy(&EP[1], pT, bsx * sizeof(pel_t));
        } else if (bsx == 8) {
            T1 = _mm_loadu_si128((__m128i *)pT);
            _mm_storel_epi64((__m128i *)&EP[1], T1);
        } else {
            for (i = 0; i < bsx; i += 16) {
                T1 = _mm_loadu_si128((__m128i *)(pT + i));
                _mm_store_si128((__m128i *)(&EP[1] + i), T1);
            }
        }
    }

    /* fill top-right pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_RIGHT)) {
        if (bsx == 4) {
            memcpy(&EP[bsx + 1], &pT[bsx], bsx * sizeof(pel_t));
        } else if (bsx == 8) {
            T1 = _mm_loadu_si128((__m128i *)&pT[bsx]);
            _mm_storel_epi64((__m128i *)&EP[bsx + 1], T1);
        } else {
            for (i = 0; i < bsx; i += 16) {
                T1 = _mm_loadu_si128((__m128i *)(&pT[bsx + i]));
                _mm_store_si128((__m128i *)(&EP[bsx + 1] + i), T1);
            }
        }
    } else {
        if (bsx == 4) {
            memset(&EP[bsx + 1], EP[bsx], bsx);
        } else if (bsx == 8) {
            T0 = _mm_set1_epi8(EP[bsx]);    // repeat the last pixel
            _mm_storel_epi64((__m128i *)&EP[bsx + 1], T0);
        } else {
            T0 = _mm_set1_epi8(EP[bsx]);    // repeat the last pixel
            for (i = 0; i < bsx; i += 16) {
                _mm_store_si128((__m128i *)(&EP[bsx + 1 + i]), T0);
            }
        }
    }

    /* fill extra pixels */
    num_padding = bsy * 11 / 4 - bsx + 4;
    if (num_padding > 0) {
        memset(&EP[2 * bsx + 1], EP[2 * bsx], num_padding); // from (2*bsx) to (iX + 3) = (bsy *11/4 + bsx - 1) + 3
    }

    /* fill left & left-down pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        const pel_t *p_l = pL;
        int y;
        /* fill left pixels */
        for (y = 0; y < bsy; y++) {
            EP[-1 - y] = *p_l;
            p_l += i_TL;
        }
    }

    /* fill left-down pixels */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT_DOWN)) {
        const pel_t *p_l = pL + bsy * i_TL;
        int y;
        for (y = 0; y < bsy; y++) {
            EP[-bsy - 1 - y] = *p_l;
            p_l += i_TL;
        }
    } else {
        if (bsy == 4) {
            memset(&EP[-(bsy << 1)], EP[-bsy], bsy);
        } else if (bsy == 8) {
            T0 = _mm_set1_epi8(EP[-bsy]);
            _mm_storel_epi64((__m128i *)&EP[-(bsy << 1)], T0);
        } else {
            T0 = _mm_set1_epi8(EP[-bsy]);
            for (i = 0; i < bsy; i += 16) {
                _mm_storeu_si128((__m128i *)(&EP[-(bsy << 1)] + i), T0);
            }
        }
    }

    /* fill extra pixels */
    num_padding = bsx * 11 / 4 - bsy + 4;
    if (num_padding > 0) {
        memset(&EP[-2 * bsy - num_padding], EP[-2 * bsy], num_padding); // from (-2*bsy) to (-iY - 3) = -(bsx *11/4 + bsy - 1) - 3
    }

    /* fill EP[0] */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_LEFT)) {
        EP[0] = pTL[0];
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        EP[0] = pT[0];
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        EP[0] = pL[0];
    }
}


