//go:build arm64

#include "textflag.h"

// qgemm8tile2x4NEON — 2(row)×4(token) load-reusing Q8_0 prefill tile. Each 32-wide weight block of
// the 2 rows is loaded once and reused across the 4 tokens; each token block is loaded once and
// reused across the 2 rows (1.67× fewer loads/cell than the 1×4 qgemm8row4NEON). The 8 cells stay
// parallel via INDEPENDENT SDOT scratch (V20-V27) and independent accumulators (V0-V7); each cell is
// single-accumulator in-order, so it is BIT-IDENTICAL to qgemm8cell(...,4). Pinned by
// TestQGemm8Tile2x4Sane.
//
// Layout: weight row i at qw+i*in (block b at +b*32, scale dw+i*nblk+b); token j at qx+j*in (block b
// at +b*32, scale dx+j*nblk+b); result (row i, token j) at dst[j*outStride + i] (floats).
//
// SDOT  Vd.4S, Vn.16B, Vm.16B = WORD 0x4E809400 | (Vm<<16) | (Vn<<5) | Vd
// SCVTF Vd.4S, Vn.4S          = WORD 0x4E21D800 | (Vn<<5) | Vd
// FMLA  Vd.4S, Vn.4S, V30.S[0]= WORD 0x4F801000 | (1<<20) | (14<<16) | (Vn<<5) | Vd   (Vm=30)
//
// V0-V7 acc V[i*4+j]; V8-V11 weight rows (r0 lo/hi, r1 lo/hi); V12-V19 tokens (t0..t3 lo/hi);
// V20-V27 SDOT temps V[20+i*4+j]; F28 dw0, F29 dw1, F30 s, F31 dx_j.
// GPR: R0/R1 weight-row ptrs, R2-R5 token ptrs, R6/R7 weight-row scale ptrs, R8-R11 token scale
// ptrs, R12 block counter, R13 dst, R14 col-stride bytes, R15/R16 prologue scratch, R17 dst col.
//
// func qgemm8tile2x4NEON(qw, qx *int8, dw, dx *float32, in, nblk, outStride int, dst *float32)
TEXT ·qgemm8tile2x4NEON(SB), NOSPLIT, $0-64
	MOVD qw+0(FP), R0
	MOVD qx+8(FP), R2
	MOVD dw+16(FP), R6
	MOVD dx+24(FP), R8
	MOVD in+32(FP), R15
	MOVD nblk+40(FP), R12
	MOVD outStride+48(FP), R14
	MOVD dst+56(FP), R13

	ADD R0, R15, R1       // weight row1 = qw + in
	ADD R2, R15, R3       // token1
	ADD R3, R15, R4       // token2
	ADD R4, R15, R5       // token3
	LSL $2, R12, R16      // nblk*4
	ADD R6, R16, R7       // dw row1 = dw + nblk
	ADD R8, R16, R9       // dx token1
	ADD R9, R16, R10      // dx token2
	ADD R10, R16, R11     // dx token3
	LSL $2, R14, R14      // outStride*4

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16

	CBZ R12, store

block:
	VLD1.P 16(R0), [V8.B16]
	VLD1.P 16(R0), [V9.B16]
	VLD1.P 16(R1), [V10.B16]
	VLD1.P 16(R1), [V11.B16]
	VLD1.P 16(R2), [V12.B16]
	VLD1.P 16(R2), [V13.B16]
	VLD1.P 16(R3), [V14.B16]
	VLD1.P 16(R3), [V15.B16]
	VLD1.P 16(R4), [V16.B16]
	VLD1.P 16(R4), [V17.B16]
	VLD1.P 16(R5), [V18.B16]
	VLD1.P 16(R5), [V19.B16]
	FMOVS (R6), F28
	ADD $4, R6
	FMOVS (R7), F29
	ADD $4, R7

	// row 0 (weight V8,V9; dw F28) -> acc V0..V3, temps V20..V23
	VEOR V20.B16, V20.B16, V20.B16
	WORD $(0x4E809400 | (12<<16) | (8<<5) | 20)
	WORD $(0x4E809400 | (13<<16) | (9<<5) | 20)
	WORD $(0x4E21D800 | (20<<5) | 20)
	FMOVS (R8), F31
	FMULS F28, F31, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (20<<5) | 0) // FMLA V0,V20,V30.S[0]
	VEOR V21.B16, V21.B16, V21.B16
	WORD $(0x4E809400 | (14<<16) | (8<<5) | 21)
	WORD $(0x4E809400 | (15<<16) | (9<<5) | 21)
	WORD $(0x4E21D800 | (21<<5) | 21)
	FMOVS (R9), F31
	FMULS F28, F31, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (21<<5) | 1) // FMLA V1,V21,V30.S[0]
	VEOR V22.B16, V22.B16, V22.B16
	WORD $(0x4E809400 | (16<<16) | (8<<5) | 22)
	WORD $(0x4E809400 | (17<<16) | (9<<5) | 22)
	WORD $(0x4E21D800 | (22<<5) | 22)
	FMOVS (R10), F31
	FMULS F28, F31, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (22<<5) | 2) // FMLA V2,V22,V30.S[0]
	VEOR V23.B16, V23.B16, V23.B16
	WORD $(0x4E809400 | (18<<16) | (8<<5) | 23)
	WORD $(0x4E809400 | (19<<16) | (9<<5) | 23)
	WORD $(0x4E21D800 | (23<<5) | 23)
	FMOVS (R11), F31
	FMULS F28, F31, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (23<<5) | 3) // FMLA V3,V23,V30.S[0]

	// row 1 (weight V10,V11; dw F29) -> acc V4..V7, temps V24..V27
	VEOR V24.B16, V24.B16, V24.B16
	WORD $(0x4E809400 | (12<<16) | (10<<5) | 24)
	WORD $(0x4E809400 | (13<<16) | (11<<5) | 24)
	WORD $(0x4E21D800 | (24<<5) | 24)
	FMOVS (R8), F31
	ADD $4, R8
	FMULS F29, F31, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (24<<5) | 4) // FMLA V4,V24,V30.S[0]
	VEOR V25.B16, V25.B16, V25.B16
	WORD $(0x4E809400 | (14<<16) | (10<<5) | 25)
	WORD $(0x4E809400 | (15<<16) | (11<<5) | 25)
	WORD $(0x4E21D800 | (25<<5) | 25)
	FMOVS (R9), F31
	ADD $4, R9
	FMULS F29, F31, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (25<<5) | 5) // FMLA V5,V25,V30.S[0]
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (16<<16) | (10<<5) | 26)
	WORD $(0x4E809400 | (17<<16) | (11<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R10), F31
	ADD $4, R10
	FMULS F29, F31, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 6) // FMLA V6,V26,V30.S[0]
	VEOR V27.B16, V27.B16, V27.B16
	WORD $(0x4E809400 | (18<<16) | (10<<5) | 27)
	WORD $(0x4E809400 | (19<<16) | (11<<5) | 27)
	WORD $(0x4E21D800 | (27<<5) | 27)
	FMOVS (R11), F31
	ADD $4, R11
	FMULS F29, F31, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (27<<5) | 7) // FMLA V7,V27,V30.S[0]

	SUB $1, R12
	CBNZ R12, block

store:
	// reduce each acc V_k -> (a0+a2)+(a1+a3); cell (i,j) -> dst[j*outStride + i].
	// token columns: R13 (j0), R17=R13+stride (j1), then +stride for j2,j3.
	// row i offset within a column = i*4 bytes.
	// j0
	VMOV V0.S[1], V8.S[0]
	VMOV V0.S[2], V9.S[0]
	VMOV V0.S[3], V10.S[0]
	FADDS F0, F9, F9
	FADDS F8, F10, F10
	FADDS F9, F10, F11
	FMOVS F11, (R13)
	VMOV V4.S[1], V8.S[0]
	VMOV V4.S[2], V9.S[0]
	VMOV V4.S[3], V10.S[0]
	FADDS F4, F9, F9
	FADDS F8, F10, F10
	FADDS F9, F10, F11
	FMOVS F11, 4(R13)
	ADD R13, R14, R17
	// j1
	VMOV V1.S[1], V8.S[0]
	VMOV V1.S[2], V9.S[0]
	VMOV V1.S[3], V10.S[0]
	FADDS F1, F9, F9
	FADDS F8, F10, F10
	FADDS F9, F10, F11
	FMOVS F11, (R17)
	VMOV V5.S[1], V8.S[0]
	VMOV V5.S[2], V9.S[0]
	VMOV V5.S[3], V10.S[0]
	FADDS F5, F9, F9
	FADDS F8, F10, F10
	FADDS F9, F10, F11
	FMOVS F11, 4(R17)
	ADD R17, R14, R17
	// j2
	VMOV V2.S[1], V8.S[0]
	VMOV V2.S[2], V9.S[0]
	VMOV V2.S[3], V10.S[0]
	FADDS F2, F9, F9
	FADDS F8, F10, F10
	FADDS F9, F10, F11
	FMOVS F11, (R17)
	VMOV V6.S[1], V8.S[0]
	VMOV V6.S[2], V9.S[0]
	VMOV V6.S[3], V10.S[0]
	FADDS F6, F9, F9
	FADDS F8, F10, F10
	FADDS F9, F10, F11
	FMOVS F11, 4(R17)
	ADD R17, R14, R17
	// j3
	VMOV V3.S[1], V8.S[0]
	VMOV V3.S[2], V9.S[0]
	VMOV V3.S[3], V10.S[0]
	FADDS F3, F9, F9
	FADDS F8, F10, F10
	FADDS F9, F10, F11
	FMOVS F11, (R17)
	VMOV V7.S[1], V8.S[0]
	VMOV V7.S[2], V9.S[0]
	VMOV V7.S[3], V10.S[0]
	FADDS F7, F9, F9
	FADDS F8, F10, F10
	FADDS F9, F10, F11
	FMOVS F11, 4(R17)
	RET
