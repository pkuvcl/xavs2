;*****************************************************************************
;* quant8.asm: x86 quantization functions
;*****************************************************************************
;*    xavs2 - video encoder of AVS2/IEEE1857.4 video coding standard
;*    Copyright (C) 2018~ VCL, NELVT, Peking University
;*
;*    Authors: Falei LUO <falei.luo@gmail.com>
;*             Jiaqi Zhang <zhangjiaqi.cs@gmail.com>
;*
;*    Homepage1: http://vcl.idm.pku.edu.cn/xavs2
;*    Homepage2: https://github.com/pkuvcl/xavs2
;*    Homepage3: https://gitee.com/pkuvcl/xavs2
;*
;*    This program is free software; you can redistribute it and/or modify
;*    it under the terms of the GNU General Public License as published by
;*    the Free Software Foundation; either version 2 of the License, or
;*    (at your option) any later version.
;*
;*    This program is distributed in the hope that it will be useful,
;*    but WITHOUT ANY WARRANTY; without even the implied warranty of
;*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;*    GNU General Public License for more details.
;*
;*    You should have received a copy of the GNU General Public License
;*    along with this program; if not, write to the Free Software
;*    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
;*
;*    This program is also available under a commercial proprietary license.
;*    For more information, contact us at sswang @ pku.edu.cn.
;*****************************************************************************


%include "x86inc.asm"
%include "x86util.asm"


SECTION .text
cextern pw_1
cextern pd_32767
cextern pd_n32768

; ----------------------------------------------------------------------------
; int quant(coeff_t *coef, const int i_coef, const int scale, const int shift, const int add);
; ----------------------------------------------------------------------------

; ----------------------------------------------------------------------------
; quant_sse4
INIT_XMM sse4
cglobal quant, 2,3,8
;{
    movq        m4, r2mp              ; m4[0] = scale
    movq        m5, r3mp              ; m5[0] = shift
    movq        m6, r4mp              ; m6[0] = add
    mov         r2, r1                ; r2    = i_coef
    shr         r1, 3                 ; r1    = i_coef/8
    pxor        m7, m7                ; m7 <-- num_non_zero = 0
    pshufd      m4, m4, 0             ; m4[3210] = scale
    pshufd      m6, m6, 0             ; m6[3210] = add
                                      ;
.loop:                                ; >>>>> LOOP
    pmovsxwd    m0, [r0    ]          ; m0 = level, 4 coeff
    pmovsxwd    m1, [r0 + 8]          ; m1 = level, 4 coeff
                                      ;
    pabsd       m2, m0                ; m2 <--   XAVS2_ABS(coef[i])
    pabsd       m3, m1                ;
    pmulld      m2, m4                ; m2 <--   XAVS2_ABS(coef[i]) * scale
    pmulld      m3, m4                ;
    paddd       m2, m6                ; m2 <--   XAVS2_ABS(coef[i]) * scale + add
    paddd       m3, m6                ;
    psrad       m2, m5                ; m2 <--  (XAVS2_ABS(coef[i]) * scale + add) >> shift
    psrad       m3, m5                ;
    psignd      m2, m0                ; m2 <-- ((XAVS2_ABS(coef[i]) * scale + add) >> shift) * xavs2_sign2(coef[i])
    psignd      m3, m1                ;
                                      ;
    packssdw    m2, m3                ; pack to 8 coeff
                                      ;
    mova      [r0], m2                ; store
    add         r0, 16                ;
                                      ;
    pxor        m0, m0                ; m0 <-- 0
    pcmpeqw     m2, m0                ; m2[i] == 0 ?
    psubw       m7, m2                ; m7[i] <-- count the number of 0
                                      ;
    dec         r1                    ;
    jnz        .loop                  ; <<<<< LOOP
                                      ;
    packuswb    m7, m7                ;
    psadbw      m7, m0                ;
    movifnidn  rax, r2                ; eax <-- i_coef
    movq        r1, m7                ;
    sub        rax, r1                ; return value: num_non_zero
    RET                               ; return
;}


; ----------------------------------------------------------------------------
; void dequant(coeff_t *coef, const int i_coef, const int scale, const int shift);
; ----------------------------------------------------------------------------

; ----------------------------------------------------------------------------
; dequant_sse4
INIT_XMM sse4
cglobal dequant, 2,4,7
;{
    mov         r3, r3mp              ; r3  <-- shift
    movq        m4, r2mp              ; m4[0] = scale
    movq        m6, r3                ; m6[0] = shift
    dec         r3                    ; r3d <-- shift - 1
    xor         r2, r2                ; r2 <-- 0
    shr         r1, 4                 ; r1    = i_coef/16
    bts         r2, r3                ; r2 <-- add = 1 < (shift - 1)
    movq        m5, r2                ; m5[0] = add
    pshufd      m4, m4, 0             ; m4[3210] = scale
    pshufd      m5, m5, 0             ; m5[3210] = add
                                      ;
.loop:                                ;
    pmovsxwd    m0, [r0     ]         ; load 4 coeff
    pmovsxwd    m1, [r0 +  8]         ;
    pmovsxwd    m2, [r0 + 16]         ;
    pmovsxwd    m3, [r0 + 24]         ;
                                      ;
    pmulld      m0, m4                ; coef[i] * scale
    pmulld      m1, m4                ;
    pmulld      m2, m4                ;
    pmulld      m3, m4                ;
    paddd       m0, m5                ; coef[i] * scale + add
    paddd       m1, m5                ;
    paddd       m2, m5                ;
    paddd       m3, m5                ;
    psrad       m0, m6                ; (coef[i] * scale + add) >> shift
    psrad       m1, m6                ;
    psrad       m2, m6                ;
    psrad       m3, m6                ;
                                      ;
    packssdw    m0, m1                ; pack to 8 coeff
    packssdw    m2, m3                ;
                                      ;
    mova   [r0   ], m0                ; store
    mova   [r0+16], m2                ;
    add         r0, 32                ; r0 <-- coef + 16
    dec         r1                    ;
    jnz        .loop                  ;
                                      ;
    RET                               ; return
;}

; ----------------------------------------------------------------------------
; int quant(coeff_t *coef, const int i_coef, const int scale, const int shift, const int add);
; ----------------------------------------------------------------------------

; ----------------------------------------------------------------------------
; quant_avx2
INIT_YMM avx2
cglobal quant, 2,3,8
;{
    vpbroadcastd      m6,  r4m                ; m6[3210] = add
    movd              xm4, r2m                ; m4[0] = scale
    vpbroadcastd      m4,  xm4                ; m4[3210] = scale
    movd              xm5, r3m                ; m5[0] = shift
    shr               r1d,   4                ; r1    = i_coef/16
    pxor              m7,   m7                ; m7 <-- num_non_zero = 0
                                              ;
    lea               r5, [pw_1]
.loop:
    pmovsxwd          m0, [r0]                ; m0 = level, 8 coeff
    pabsd             m2, m0                  ; m2 <--   XAVS2_ABS(coef[i])
    pmulld            m2, m4                  ; m2 <--   XAVS2_ABS(coef[i]) * scale
    paddd             m2, m2, m6              ; m2 <--   XAVS2_ABS(coef[i]) * scale + add
    psrad             m2, xm5                 ; m2 <--   (XAVS2_ABS(coef[i]) * scale + add) >> shift
    psignd            m2, m0                  ; m2 <--   ((XAVS2_ABS(coef[i]) * scale + add) >> shift) * xavs2_sign2(coef[i])
                                              ;
    pmovsxwd          m1, [r0 + 16]           ;
    pabsd             m3, m1                  ;
    pmulld            m3, m4                  ;
    paddd             m3, m3, m6              ;
    psrad             m3, xm5                 ;
    psignd            m3, m1                  ;
                                              ;
    packssdw          m2, m3                  ;
    vpermq            m2, m2, q3120           ;
    vmovdqa           [r0], m2                ;
    add               r0, 32                  ; r0 = r0 + 16
                                              ;
    pminuw            m2, [r5]
    paddw             m7, m2
    dec               r1d                     ;
    jnz               .loop                   ; <<<<< LOOP
                                              ;
    xorpd             m0, m0
    psadbw            m7, m0
    vextracti128      xm1, m7, 1
    paddd             xm7, xm1
    movhlps           xm0, xm7
    paddd             xm7, xm0
    movd              eax, xm7
    RET                                     ; return
;}


; ----------------------------------------------------------------------------
; void dequant(coeff_t *coef, const int i_coef, const int scale, const int shift);
; ----------------------------------------------------------------------------

; ----------------------------------------------------------------------------
%if ARCH_X86_64
; dequant_avx2
INIT_YMM avx2
cglobal dequant, 2,4,7
;{
    movd              xm4, r2m                ; m4[0] = scale
    vpbroadcastd      m4 , xm4                ; m4[3210] = scale
    movd              xm5, r3m                ; m5[0] = shift
    shr               r1d,   4                ; r1    = i_coef/16

    xor               r2 ,  r2                ; r2 <--- 0
    dec               r3                      ; shift -= 1
    bts               r2m , r3m               ; r2 <-- add = 1 < (shift - 1)
    movd              xm6 , r2m
    vpbroadcastd      m6, xm6                 ;
    
    ;m4 <--- scale
    ;m5 <--- shift
    ;m6 <--- add

.loop:
    pmovsxwd          m0, [r0     ]          ; load 8 coeff
    pmovsxwd          m1, [r0 + 16]          ;
                                          
    pmulld            m0, m4                 ; coef[i] * scale
    pmulld            m1, m4                 ;
                                          
    paddd             m0, m0, m6             ; coef[i] * scale + add
    paddd             m1, m1, m6             ;
                    
    psrad             m0, xm5                ; (coef[i] * scale + add) >> shift
    psrad             m1, xm5                ;
    
    packssdw          m0, m1                 ; pack to 16 coeff
    vpermq            m0, m0, q3120          ;
    vmovdqa           [r0], m0               ;

    add               r0, 32
    dec               r1d
    jnz              .loop
    RET

;}
%endif
