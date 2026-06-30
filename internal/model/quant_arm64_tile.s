//go:build arm64 && !(fakaccel && darwin && cgo)

#include "textflag.h"

// qgemm8tileNEON computes a 4(row)×4(token) Q8_0 output tile with deferred reduction — the
// prefill GEMM micro-kernel (the arm64 twin of the AVX-512 qgemm8tile512). Weight row i, block b
// at qw + i*in + b*32 (scale dw + i*nblk + b); token j, block b at qx + j*in + b*32 (scale
// dx + j*nblk + b); result (row i, token j) written to dst[j*outStride + i] (floats). Each of the
// 16 cells folds its blocks through an independent 4-lane float accumulator (16 live in V0..V15)
// and is bit-identical to qgemm8cell(...,4): SDOT gives the four int32 lane partials, SCVTF is
// exact, the broadcast-scale VFMLA matches math.FMA (one rounding), and the (a0+a2)+(a1+a3) lane
// reduction uses the same pairwise tree. Pinned by TestQGemm8TileNEONMatchesCell.
//
// SDOT  Vd.4S, Vn.16B, Vm.16B = WORD 0x4E809400 | (Vm<<16) | (Vn<<5) | Vd
// SCVTF Vd.4S, Vn.4S          = WORD 0x4E21D800 | (Vn<<5) | Vd
//
// Register map. GPR: R0-R3 weight-row code ptrs, R4-R7 token code ptrs, R8-R11 weight-row scale
// ptrs, R12-R15 token scale ptrs, R16 block counter, R19 dst, R21 col-stride bytes, R22-R24 token
// column bases, R20/R25 prologue scratch. V0-V15 accumulators (V[i*4+j]); V16-V23 the four token
// blocks (lo,hi); V24,V25 the streamed weight row block (lo,hi); V26 SDOT/SCVTF scratch; V27
// broadcast scale; F28 dw_i, F29 dx_j, F30 s; V28-V31 reused as reduction temps after the loop.
//
// func qgemm8tileNEON(qw, qx *int8, dw, dx *float32, in, nblk, outStride int, dst *float32)
TEXT ·qgemm8tileNEON(SB), NOSPLIT, $0-64
	MOVD qw+0(FP), R0
	MOVD qx+8(FP), R4
	MOVD dw+16(FP), R8
	MOVD dx+24(FP), R12
	MOVD in+32(FP), R20    // bytes per weight/token row
	MOVD nblk+40(FP), R16
	MOVD outStride+48(FP), R21
	MOVD dst+56(FP), R19

	ADD R0, R20, R1
	ADD R1, R20, R2
	ADD R2, R20, R3
	ADD R4, R20, R5
	ADD R5, R20, R6
	ADD R6, R20, R7
	LSL $2, R16, R25       // R25 = nblk*4 (scale stride)
	ADD R8, R25, R9
	ADD R9, R25, R10
	ADD R10, R25, R11
	ADD R12, R25, R13
	ADD R13, R25, R14
	ADD R14, R25, R15
	LSL $2, R21, R21       // R21 = outStride*4 (dst column stride, bytes)
	ADD R19, R21, R22
	ADD R22, R21, R23
	ADD R23, R21, R24

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16

	CBZ R16, store

tblock:
	VLD1.P 16(R4), [V16.B16]
	VLD1.P 16(R4), [V17.B16]
	VLD1.P 16(R5), [V18.B16]
	VLD1.P 16(R5), [V19.B16]
	VLD1.P 16(R6), [V20.B16]
	VLD1.P 16(R6), [V21.B16]
	VLD1.P 16(R7), [V22.B16]
	VLD1.P 16(R7), [V23.B16]

	// row 0 (weights R0, dw R8) -> acc V0..V3
	VLD1.P 16(R0), [V24.B16]
	VLD1.P 16(R0), [V25.B16]
	FMOVS (R8), F28
	ADD $4, R8
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (16<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (17<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R12), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 0) // FMLA V0.4S, V26.4S, V30.S[0]
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (18<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (19<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R13), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 1) // FMLA V1.4S, V26.4S, V30.S[0]
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (20<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (21<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R14), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 2) // FMLA V2.4S, V26.4S, V30.S[0]
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (22<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (23<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R15), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 3) // FMLA V3.4S, V26.4S, V30.S[0]

	// row 1 (weights R1, dw R9) -> acc V4..V7
	VLD1.P 16(R1), [V24.B16]
	VLD1.P 16(R1), [V25.B16]
	FMOVS (R9), F28
	ADD $4, R9
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (16<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (17<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R12), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 4) // FMLA V4.4S, V26.4S, V30.S[0]
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (18<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (19<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R13), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 5) // FMLA V5.4S, V26.4S, V30.S[0]
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (20<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (21<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R14), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 6) // FMLA V6.4S, V26.4S, V30.S[0]
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (22<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (23<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R15), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 7) // FMLA V7.4S, V26.4S, V30.S[0]

	// row 2 (weights R2, dw R10) -> acc V8..V11
	VLD1.P 16(R2), [V24.B16]
	VLD1.P 16(R2), [V25.B16]
	FMOVS (R10), F28
	ADD $4, R10
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (16<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (17<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R12), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 8) // FMLA V8.4S, V26.4S, V30.S[0]
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (18<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (19<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R13), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 9) // FMLA V9.4S, V26.4S, V30.S[0]
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (20<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (21<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R14), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 10) // FMLA V10.4S, V26.4S, V30.S[0]
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (22<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (23<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R15), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 11) // FMLA V11.4S, V26.4S, V30.S[0]

	// row 3 (weights R3, dw R11) -> acc V12..V15
	VLD1.P 16(R3), [V24.B16]
	VLD1.P 16(R3), [V25.B16]
	FMOVS (R11), F28
	ADD $4, R11
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (16<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (17<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R12), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 12) // FMLA V12.4S, V26.4S, V30.S[0]
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (18<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (19<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R13), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 13) // FMLA V13.4S, V26.4S, V30.S[0]
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (20<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (21<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R14), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 14) // FMLA V14.4S, V26.4S, V30.S[0]
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (22<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (23<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R15), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 15) // FMLA V15.4S, V26.4S, V30.S[0]

	ADD $4, R12
	ADD $4, R13
	ADD $4, R14
	ADD $4, R15

	SUB $1, R16
	CBNZ R16, tblock

store:
	// column 0 (base R19): rows 0..3 = V0,V4,V8,V12 at +0,+4,+8,+12
	VMOV V0.S[1], V28.S[0]
	VMOV V0.S[2], V29.S[0]
	VMOV V0.S[3], V30.S[0]
	FADDS F0, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, (R19)
	VMOV V4.S[1], V28.S[0]
	VMOV V4.S[2], V29.S[0]
	VMOV V4.S[3], V30.S[0]
	FADDS F4, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 4(R19)
	VMOV V8.S[1], V28.S[0]
	VMOV V8.S[2], V29.S[0]
	VMOV V8.S[3], V30.S[0]
	FADDS F8, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 8(R19)
	VMOV V12.S[1], V28.S[0]
	VMOV V12.S[2], V29.S[0]
	VMOV V12.S[3], V30.S[0]
	FADDS F12, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 12(R19)
	// column 1 (base R22): rows 0..3 = V1,V5,V9,V13
	VMOV V1.S[1], V28.S[0]
	VMOV V1.S[2], V29.S[0]
	VMOV V1.S[3], V30.S[0]
	FADDS F1, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, (R22)
	VMOV V5.S[1], V28.S[0]
	VMOV V5.S[2], V29.S[0]
	VMOV V5.S[3], V30.S[0]
	FADDS F5, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 4(R22)
	VMOV V9.S[1], V28.S[0]
	VMOV V9.S[2], V29.S[0]
	VMOV V9.S[3], V30.S[0]
	FADDS F9, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 8(R22)
	VMOV V13.S[1], V28.S[0]
	VMOV V13.S[2], V29.S[0]
	VMOV V13.S[3], V30.S[0]
	FADDS F13, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 12(R22)
	// column 2 (base R23): rows 0..3 = V2,V6,V10,V14
	VMOV V2.S[1], V28.S[0]
	VMOV V2.S[2], V29.S[0]
	VMOV V2.S[3], V30.S[0]
	FADDS F2, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, (R23)
	VMOV V6.S[1], V28.S[0]
	VMOV V6.S[2], V29.S[0]
	VMOV V6.S[3], V30.S[0]
	FADDS F6, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 4(R23)
	VMOV V10.S[1], V28.S[0]
	VMOV V10.S[2], V29.S[0]
	VMOV V10.S[3], V30.S[0]
	FADDS F10, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 8(R23)
	VMOV V14.S[1], V28.S[0]
	VMOV V14.S[2], V29.S[0]
	VMOV V14.S[3], V30.S[0]
	FADDS F14, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 12(R23)
	// column 3 (base R24): rows 0..3 = V3,V7,V11,V15
	VMOV V3.S[1], V28.S[0]
	VMOV V3.S[2], V29.S[0]
	VMOV V3.S[3], V30.S[0]
	FADDS F3, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, (R24)
	VMOV V7.S[1], V28.S[0]
	VMOV V7.S[2], V29.S[0]
	VMOV V7.S[3], V30.S[0]
	FADDS F7, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 4(R24)
	VMOV V11.S[1], V28.S[0]
	VMOV V11.S[2], V29.S[0]
	VMOV V11.S[3], V30.S[0]
	FADDS F11, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 8(R24)
	VMOV V15.S[1], V28.S[0]
	VMOV V15.S[2], V29.S[0]
	VMOV V15.S[3], V30.S[0]
	FADDS F15, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 12(R24)
	RET
