//go:build amd64

#include "textflag.h"

// Original in-tree implementation of mathx.FDot's exact eight-lane reduction.
// Apache-2.0, like the surrounding repository. No external implementation copied.
// SSE2 baseline only: separate MULPS/ADDPS, no FMA, horizontal reduction or
// cross-lane permutation. Each lane accumulates i, i+8, ... in scalar order.
// func fDotLanesSSE2(r, x *float32, groups int, lanes *float32) bool
TEXT ·fDotLanesSSE2(SB), NOSPLIT, $0-33
    MOVQ r+0(FP), SI
    MOVQ x+8(FP), DI
    MOVQ groups+16(FP), CX
    MOVQ lanes+24(FP), DX
    XORPS X7, X7
    XORPS X8, X8
    TESTQ CX, CX
    JLE decline
loop:
    MOVUPS 0(SI), X0
    MOVUPS 16(SI), X1
    MOVUPS 0(DI), X2
    MOVUPS 16(DI), X3

    MULPS X2, X0
    MULPS X3, X1
    ADDPS X0, X7
    ADDPS X1, X8
    ADDQ $32, SI
    ADDQ $32, DI
    DECQ CX
    JNZ loop
    MOVUPS X7, 0(DX)
    MOVUPS X8, 16(DX)
    MOVB $1, ret+32(FP)
    RET
decline:
    MOVB $0, ret+32(FP)
    RET
