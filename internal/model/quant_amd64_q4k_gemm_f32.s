//go:build amd64

#include "textflag.h"

// q4kBlockDot256x2AVX2 adapts the exact accumulation pattern already used by
// q4kRowDotF32AVX2 (quant_amd64_q4k_f32.s, in-tree Apache-2.0). Its input is the
// existing scalar decoder's 256 f32 values, not a second packed-weight decode.
//
// X7 and X8 independently hold s0..s3 for token0 and token1. For each eight
// values, add the low four products then the high four, matching increasing
// i in each scalar i%4 lane. Multiplication and addition remain separate:
// no FMA, no cross-token sum, and final (s0+s1)+(s2+s3) is unchanged. The Go
// caller retains the serial super-block sum. Unaligned loads/stores are legal.
//
// func q4kBlockDot256x2AVX2(weights, x0, x1, dst *float32)
TEXT ·q4kBlockDot256x2AVX2(SB), NOSPLIT, $0-32
	MOVQ weights+0(FP), SI
	MOVQ x0+8(FP), DI
	MOVQ x1+16(FP), R8
	MOVQ dst+24(FP), R9
	MOVQ $32, CX // 32 groups of 8 = 256 values
	VXORPS X7, X7, X7
	VXORPS X8, X8, X8

pairdot:
	VMOVUPS (SI), Y0
	VMULPS (DI), Y0, Y1
	VMULPS (R8), Y0, Y2
	VEXTRACTF128 $1, Y1, X3
	VADDPS X1, X7, X7
	VADDPS X3, X7, X7
	VEXTRACTF128 $1, Y2, X3
	VADDPS X2, X8, X8
	VADDPS X3, X8, X8
	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $32, R8
	DECQ CX
	JNZ pairdot

	// Token0: use only X9..X11 as fold scratch, preserving token1 in X8.
	VPSHUFD $0x55, X7, X9
	VADDSS X9, X7, X9
	VPSHUFD $0xAA, X7, X10
	VPSHUFD $0xFF, X7, X11
	VADDSS X11, X10, X10
	VADDSS X10, X9, X9
	VMOVSS X9, 0(R9)

	// Token1: the identical pairwise fold, with its own four lane sums.
	VPSHUFD $0x55, X8, X9
	VADDSS X9, X8, X9
	VPSHUFD $0xAA, X8, X10
	VPSHUFD $0xFF, X8, X11
	VADDSS X11, X10, X10
	VADDSS X10, X9, X9
	VMOVSS X9, 4(R9)
	VZEROUPPER
	RET
