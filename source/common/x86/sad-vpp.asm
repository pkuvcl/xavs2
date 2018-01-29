; ============================================================================
; sad-vpp.asm
; - x86 sad functions
; ----------------------------------------------------------------------------
;
; xavs2 encoder , the Chinese AVS2 video encoder library.
;
; ============================================================================

%include "x86inc.asm"
%include "x86util.asm"

; ----------------------------------------------------------------------------
; all functions in this file are same as in sad-a.asm, except the
; stride value: VPP_STRIDE
;
; ----------------------------------------------------------------------------
; functions defined in this file:
; vpp_sad_x3_8x8_xmm2
; vpp_sad_x4_8x8_xmm2
;
; vpp_sad_x3_8x8_sse2
; vpp_sad_x4_8x8_sse2
;
; vpp_sad_x4_8x8_ssse3
;
; #if ARCH_X86_64 == 1 && HIGH_BIT_DEPTH == 0
; vpp_sad_x3_8x8_avx2
; vpp_sad_x4_8x8_avx2
; #endif
; ----------------------------------------------------------------------------


SECTION_RODATA 32

SECTION .text

%assign VPP_STRIDE  8


;=============================================================================
; SAD x3/x4 MMX
;=============================================================================

%macro SAD_X3_START_1x8P 0
    movq    mm3,    [r0]
    movq    mm0,    [r1]
    movq    mm1,    [r2]
    movq    mm2,    [r3]
    psadbw  mm0,    mm3
    psadbw  mm1,    mm3
    psadbw  mm2,    mm3
%endmacro

%macro SAD_X3_1x8P 2
    movq    mm3,    [r0+%1]
    movq    mm4,    [r1+%2]
    movq    mm5,    [r2+%2]
    movq    mm6,    [r3+%2]
    psadbw  mm4,    mm3
    psadbw  mm5,    mm3
    psadbw  mm6,    mm3
    paddw   mm0,    mm4
    paddw   mm1,    mm5
    paddw   mm2,    mm6
%endmacro

%macro SAD_X3_2x8P 1
%if %1
    SAD_X3_START_1x8P
%else
    SAD_X3_1x8P 0, 0
%endif
    SAD_X3_1x8P VPP_STRIDE, r4
    add     r0, 2*VPP_STRIDE
    lea     r1, [r1+2*r4]
    lea     r2, [r2+2*r4]
    lea     r3, [r3+2*r4]
%endmacro

%macro SAD_X4_START_1x8P 0
    movq    mm7,    [r0]
    movq    mm0,    [r1]
    movq    mm1,    [r2]
    movq    mm2,    [r3]
    movq    mm3,    [r4]
    psadbw  mm0,    mm7
    psadbw  mm1,    mm7
    psadbw  mm2,    mm7
    psadbw  mm3,    mm7
%endmacro

%macro SAD_X4_1x8P 2
    movq    mm7,    [r0+%1]
    movq    mm4,    [r1+%2]
    movq    mm5,    [r2+%2]
    movq    mm6,    [r3+%2]
    psadbw  mm4,    mm7
    psadbw  mm5,    mm7
    psadbw  mm6,    mm7
    psadbw  mm7,    [r4+%2]
    paddw   mm0,    mm4
    paddw   mm1,    mm5
    paddw   mm2,    mm6
    paddw   mm3,    mm7
%endmacro

%macro SAD_X4_2x8P 1
%if %1
    SAD_X4_START_1x8P
%else
    SAD_X4_1x8P 0, 0
%endif
    SAD_X4_1x8P VPP_STRIDE, r5
    add     r0, 2*VPP_STRIDE
    lea     r1, [r1+2*r5]
    lea     r2, [r2+2*r5]
    lea     r3, [r3+2*r5]
    lea     r4, [r4+2*r5]
%endmacro

%macro SAD_X3_END 0
%if UNIX64
    movd    [r5+0], mm0
    movd    [r5+4], mm1
    movd    [r5+8], mm2
%else
    mov     r0, r5mp
    movd    [r0+0], mm0
    movd    [r0+4], mm1
    movd    [r0+8], mm2
%endif
    RET
%endmacro

%macro SAD_X4_END 0
    mov     r0, r6mp
    movd    [r0+0], mm0
    movd    [r0+4], mm1
    movd    [r0+8], mm2
    movd    [r0+12], mm3
    RET
%endmacro


; ----------------------------------------------------------------------------
; void vpp_sad_x3_8x8( uint8_t *fenc, uint8_t *pix0, uint8_t *pix1,
;                      uint8_t *pix2, intptr_t i_stride, int scores[3] )
; ----------------------------------------------------------------------------
%macro SAD_X 3
cglobal vpp_sad_x%1_%2x%3_mmx2, %1+2, %1+2
    SAD_X%1_2x%2P 1
%rep %3/2-1
    SAD_X%1_2x%2P 0
%endrep
    SAD_X%1_END
%endmacro

INIT_MMX
SAD_X 3,  8,  8
SAD_X 4,  8,  8


; ============================================================================
; SAD x3/x4 XMM
; ============================================================================

%if ARCH_X86_64
    DECLARE_REG_TMP 6
%else
    DECLARE_REG_TMP 5
%endif

%macro SAD_X3_START_2x8P_SSE2 0
    movq     m3, [r0]
    movq     m0, [r1]
    movq     m1, [r2]
    movq     m2, [r3]
    movhps   m3, [r0+VPP_STRIDE]
    movhps   m0, [r1+r4]
    movhps   m1, [r2+r4]
    movhps   m2, [r3+r4]
    psadbw   m0, m3
    psadbw   m1, m3
    psadbw   m2, m3
%endmacro

%macro SAD_X3_2x8P_SSE2 4
    movq     m6, [r0+%1]
    movq     m3, [r1+%2]
    movq     m4, [r2+%2]
    movq     m5, [r3+%2]
    movhps   m6, [r0+%3]
    movhps   m3, [r1+%4]
    movhps   m4, [r2+%4]
    movhps   m5, [r3+%4]
    psadbw   m3, m6
    psadbw   m4, m6
    psadbw   m5, m6
    paddd    m0, m3
    paddd    m1, m4
    paddd    m2, m5
%endmacro

%macro SAD_X4_START_2x8P_SSE2 0
    movq     m4, [r0]
    movq     m0, [r1]
    movq     m1, [r2]
    movq     m2, [r3]
    movq     m3, [r4]
    movhps   m4, [r0+VPP_STRIDE]
    movhps   m0, [r1+r5]
    movhps   m1, [r2+r5]
    movhps   m2, [r3+r5]
    movhps   m3, [r4+r5]
    psadbw   m0, m4
    psadbw   m1, m4
    psadbw   m2, m4
    psadbw   m3, m4
%endmacro

%macro SAD_X4_2x8P_SSE2 4
    movq     m6, [r0+%1]
    movq     m4, [r1+%2]
    movq     m5, [r2+%2]
    movhps   m6, [r0+%3]
    movhps   m4, [r1+%4]
    movhps   m5, [r2+%4]
    psadbw   m4, m6
    psadbw   m5, m6
    paddd    m0, m4
    paddd    m1, m5
    movq     m4, [r3+%2]
    movq     m5, [r4+%2]
    movhps   m4, [r3+%4]
    movhps   m5, [r4+%4]
    psadbw   m4, m6
    psadbw   m5, m6
    paddd    m2, m4
    paddd    m3, m5
%endmacro

%macro SAD_X3_4x8P_SSE2 2
%if %1==0
    lea  t0, [r4*3]
    SAD_X3_START_2x8P_SSE2
%else
    SAD_X3_2x8P_SSE2 VPP_STRIDE*(0+(%1&1)*4), r4*0, VPP_STRIDE*(1+(%1&1)*4), r4*1
%endif
    SAD_X3_2x8P_SSE2 VPP_STRIDE*(2+(%1&1)*4), r4*2, VPP_STRIDE*(3+(%1&1)*4), t0
%if %1 != %2-1
%if (%1&1) != 0
    add  r0, 8*VPP_STRIDE
%endif
    lea  r1, [r1+4*r4]
    lea  r2, [r2+4*r4]
    lea  r3, [r3+4*r4]
%endif
%endmacro

%macro SAD_X4_4x8P_SSE2 2
%if %1==0
    lea    r6, [r5*3]
    SAD_X4_START_2x8P_SSE2
%else
    SAD_X4_2x8P_SSE2 VPP_STRIDE*(0+(%1&1)*4), r5*0, VPP_STRIDE*(1+(%1&1)*4), r5*1
%endif
    SAD_X4_2x8P_SSE2 VPP_STRIDE*(2+(%1&1)*4), r5*2, VPP_STRIDE*(3+(%1&1)*4), r6
%if %1 != %2-1
%if (%1&1) != 0
    add  r0, 8*VPP_STRIDE
%endif
    lea  r1, [r1+4*r5]
    lea  r2, [r2+4*r5]
    lea  r3, [r3+4*r5]
    lea  r4, [r4+4*r5]
%endif
%endmacro

%macro SAD_X3_END_SSE2 1
    movifnidn r5, r5mp
    movhlps    m3, m0
    movhlps    m4, m1
    movhlps    m5, m2
    paddd      m0, m3
    paddd      m1, m4
    paddd      m2, m5
    movd   [r5+0], m0
    movd   [r5+4], m1
    movd   [r5+8], m2
    RET
%endmacro

%macro SAD_X4_END_SSE2 1
    mov      r0, r6mp
    psllq      m1, 32
    psllq      m3, 32
    paddd      m0, m1
    paddd      m2, m3
    movhlps    m1, m0
    movhlps    m3, m2
    paddd      m0, m1
    paddd      m2, m3
    movq   [r0+0], m0
    movq   [r0+8], m2
    RET
%endmacro


; ----------------------------------------------------------------------------
; void vpp_sad_x3_8x8( uint8_t *fenc, uint8_t *pix0, uint8_t *pix1,
;                      uint8_t *pix2, intptr_t i_stride, int scores[3] )
; ----------------------------------------------------------------------------
%macro SAD_X_SSE2 4
cglobal vpp_sad_x%1_%2x%3, 2+%1,3+%1,%4
%assign x 0
%rep %3/4
    SAD_X%1_4x%2P_SSE2 x, %3/4
%assign x x+1
%endrep
%if %3 == 64
    SAD_X%1_END_SSE2 1
%else
    SAD_X%1_END_SSE2 0
%endif
%endmacro


INIT_XMM sse2
SAD_X_SSE2  3,  8,  8,  7
SAD_X_SSE2  4,  8,  8,  7

INIT_XMM ssse3
SAD_X_SSE2  4,  8,  8,  7


%if ARCH_X86_64 == 1 && HIGH_BIT_DEPTH == 0

INIT_YMM avx2
cglobal vpp_sad_x3_8x8, 6,6,5
    xorps           m0, m0
    xorps           m1, m1

    sub             r2, r1          ; rebase on pointer r1
    sub             r3, r1
%assign x 0
%rep 4
    ; row 0
    vpbroadcastq   xm2, [r0 + 0 * VPP_STRIDE]
    movq           xm3, [r1]
    movhps         xm3, [r1 + r2]
    movq           xm4, [r1 + r3]
    psadbw         xm3, xm2
    psadbw         xm4, xm2
    paddd          xm0, xm3
    paddd          xm1, xm4
    add             r1, r4

    ; row 1
    vpbroadcastq   xm2, [r0 + 1 * VPP_STRIDE]
    movq           xm3, [r1]
    movhps         xm3, [r1 + r2]
    movq           xm4, [r1 + r3]
    psadbw         xm3, xm2
    psadbw         xm4, xm2
    paddd          xm0, xm3
    paddd          xm1, xm4

%assign x x+1
  %if x < 4
    add             r1, r4
    add             r0, 2 * VPP_STRIDE
  %endif
%endrep

    pshufd          xm0, xm0, q0020
    movq            [r5 + 0], xm0
    movd            [r5 + 8], xm1
    RET


INIT_YMM avx2
cglobal vpp_sad_x4_8x8, 7,7,5
    xorps           m0, m0
    xorps           m1, m1

    sub             r2, r1          ; rebase on pointer r1
    sub             r3, r1
    sub             r4, r1
%assign x 0
%rep 4
    ; row 0
    vpbroadcastq   xm2, [r0 + 0 * VPP_STRIDE]
    movq           xm3, [r1]
    movhps         xm3, [r1 + r2]
    movq           xm4, [r1 + r3]
    movhps         xm4, [r1 + r4]
    psadbw         xm3, xm2
    psadbw         xm4, xm2
    paddd          xm0, xm3
    paddd          xm1, xm4
    add             r1, r5

    ; row 1
    vpbroadcastq   xm2, [r0 + 1 * VPP_STRIDE]
    movq           xm3, [r1]
    movhps         xm3, [r1 + r2]
    movq           xm4, [r1 + r3]
    movhps         xm4, [r1 + r4]
    psadbw         xm3, xm2
    psadbw         xm4, xm2
    paddd          xm0, xm3
    paddd          xm1, xm4

%assign x x+1
  %if x < 4
    add             r1, r5
    add             r0, 2 * VPP_STRIDE
  %endif
%endrep

    pshufd          xm0, xm0, q0020
    pshufd          xm1, xm1, q0020
    movq            [r6 + 0], xm0
    movq            [r6 + 8], xm1
    RET

%endif
