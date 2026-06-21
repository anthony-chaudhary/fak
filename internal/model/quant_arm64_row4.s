//go:build arm64

#include "textflag.h"

// qgemm8row4NEON computes one weight row against FOUR activation tokens — the load-reusing prefill
// GEMM micro-kernel. Unlike the per-cell qdot8unroll4NEON sweep (which reloads the weight block for
// every token), the weight block is loaded ONCE per K-step and reused across the 4 tokens (1.6×
// fewer loads/block). Latency is hidden by the FOUR cells running as four independent chains:
// independent SDOT scratch (V16-V19), independent token regs (V8-V15), independent accumulators
// (V0-V3). Each cell is single-accumulator in-order, so it is BIT-IDENTICAL to qgemm8cell(...,4):
// SDOT lane partials, SCVTF, broadcast-scale FMLA (one rounding), (a0+a2)+(a1+a3) reduce.
//
// Layout: weight row at qw (block b at qw+b*32, scale dw+b); token j at qx+j*in (block b at
// qx+j*in+b*32, scale dx+j*nblk+b); result token j written to dst[j*outStride] (floats).
//
// SDOT  Vd.4S, Vn.16B, Vm.16B = WORD 0x4E809400 | (Vm<<16) | (Vn<<5) | Vd
// SCVTF Vd.4S, Vn.4S          = WORD 0x4E21D800 | (Vn<<5) | Vd
// FMLA  Vd.4S, Vn.4S, V22.S[0]= WORD 0x4F801000 | (1<<20) | (6<<16) | (Vn<<5) | Vd   (Vm=22)
//
// func qgemm8row4NEON(qw, qx *int8, dw, dx *float32, in, nblk, outStride int, dst *float32)
TEXT ·qgemm8row4NEON(SB), NOSPLIT, $0-64
	MOVD qw+0(FP), R0
	MOVD qx+8(FP), R1
	MOVD dw+16(FP), R5
	MOVD dx+24(FP), R6
	MOVD in+32(FP), R13
	MOVD nblk+40(FP), R10
	MOVD outStride+48(FP), R12
	MOVD dst+56(FP), R11

	// token code ptrs R1..R4 = qx + j*in
	ADD R1, R13, R2
	ADD R2, R13, R3
	ADD R3, R13, R4
	// token scale ptrs R6..R9 = dx + j*nblk (bytes = nblk*4)
	LSL $2, R10, R14
	ADD R6, R14, R7
	ADD R7, R14, R8
	ADD R8, R14, R9
	// dst column stride bytes
	LSL $2, R12, R12

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

	CBZ R10, reduce

block:
	VLD1.P 16(R0), [V4.B16]
	VLD1.P 16(R0), [V5.B16]
	FMOVS (R5), F20
	ADD $4, R5
	VLD1.P 16(R1), [V8.B16]
	VLD1.P 16(R1), [V9.B16]
	VLD1.P 16(R2), [V10.B16]
	VLD1.P 16(R2), [V11.B16]
	VLD1.P 16(R3), [V12.B16]
	VLD1.P 16(R3), [V13.B16]
	VLD1.P 16(R4), [V14.B16]
	VLD1.P 16(R4), [V15.B16]

	// cell 0 -> V0
	VEOR V16.B16, V16.B16, V16.B16
	WORD $(0x4E809400 | (8<<16) | (4<<5) | 16)
	WORD $(0x4E809400 | (9<<16) | (5<<5) | 16)
	WORD $(0x4E21D800 | (16<<5) | 16)
	FMOVS (R6), F21
	ADD $4, R6
	FMULS F20, F21, F22
	WORD $(0x4F801000 | (1<<20) | (6<<16) | (16<<5) | 0) // FMLA V0, V16, V22.S[0]
	// cell 1 -> V1
	VEOR V17.B16, V17.B16, V17.B16
	WORD $(0x4E809400 | (10<<16) | (4<<5) | 17)
	WORD $(0x4E809400 | (11<<16) | (5<<5) | 17)
	WORD $(0x4E21D800 | (17<<5) | 17)
	FMOVS (R7), F21
	ADD $4, R7
	FMULS F20, F21, F22
	WORD $(0x4F801000 | (1<<20) | (6<<16) | (17<<5) | 1) // FMLA V1, V17, V22.S[0]
	// cell 2 -> V2
	VEOR V18.B16, V18.B16, V18.B16
	WORD $(0x4E809400 | (12<<16) | (4<<5) | 18)
	WORD $(0x4E809400 | (13<<16) | (5<<5) | 18)
	WORD $(0x4E21D800 | (18<<5) | 18)
	FMOVS (R8), F21
	ADD $4, R8
	FMULS F20, F21, F22
	WORD $(0x4F801000 | (1<<20) | (6<<16) | (18<<5) | 2) // FMLA V2, V18, V22.S[0]
	// cell 3 -> V3
	VEOR V19.B16, V19.B16, V19.B16
	WORD $(0x4E809400 | (14<<16) | (4<<5) | 19)
	WORD $(0x4E809400 | (15<<16) | (5<<5) | 19)
	WORD $(0x4E21D800 | (19<<5) | 19)
	FMOVS (R9), F21
	ADD $4, R9
	FMULS F20, F21, F22
	WORD $(0x4F801000 | (1<<20) | (6<<16) | (19<<5) | 3) // FMLA V3, V19, V22.S[0]

	SUB $1, R10
	CBNZ R10, block

reduce:
	// each acc V0..V3 -> (a0+a2)+(a1+a3), store to dst + j*outStride
	VMOV V0.S[1], V24.S[0]
	VMOV V0.S[2], V25.S[0]
	VMOV V0.S[3], V26.S[0]
	FADDS F0, F25, F25
	FADDS F24, F26, F26
	FADDS F25, F26, F23
	FMOVS F23, (R11)
	ADD R11, R12, R11
	VMOV V1.S[1], V24.S[0]
	VMOV V1.S[2], V25.S[0]
	VMOV V1.S[3], V26.S[0]
	FADDS F1, F25, F25
	FADDS F24, F26, F26
	FADDS F25, F26, F23
	FMOVS F23, (R11)
	ADD R11, R12, R11
	VMOV V2.S[1], V24.S[0]
	VMOV V2.S[2], V25.S[0]
	VMOV V2.S[3], V26.S[0]
	FADDS F2, F25, F25
	FADDS F24, F26, F26
	FADDS F25, F26, F23
	FMOVS F23, (R11)
	ADD R11, R12, R11
	VMOV V3.S[1], V24.S[0]
	VMOV V3.S[2], V25.S[0]
	VMOV V3.S[3], V26.S[0]
	FADDS F3, F25, F25
	FADDS F24, F26, F26
	FADDS F25, F26, F23
	FMOVS F23, (R11)
	RET
