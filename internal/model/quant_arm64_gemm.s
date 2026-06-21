//go:build arm64

#include "textflag.h"

// quant_arm64_gemm.s — the arm64 NEON deferred-reduction Q8_0 kernels that close the prefill
// and decode throughput gap to llama.cpp on Apple Silicon. The original arm64 path streamed one
// output cell per qdot8asm call and reduced each block's int32 lanes to a scalar INSIDE the block
// loop (VADDV + a vector->GPR move + a scalar SCVTF + a serial FMADD), so the per-block horizontal
// reduce dominated the useful SDOT work and nothing was reused across the GEMM's two free axes.
//
// These kernels do the textbook GEMM micro-kernel instead, the arm64 twin of the AVX-512
// qgemm8tile512 (quant_amd64.s):
//   1. DEFERRED REDUCTION — keep each block dot's four int32 SDOT lanes in a VECTOR float
//      accumulator (one SCVTF + one broadcast-scale VFMLA per block, single rounding) and reduce
//      the four lanes to a scalar ONCE per output, after the block loop. The per-block VADDV and
//      the vector->GPR round-trip vanish from the inner loop.
//   2. REGISTER BLOCKING — compute several output cells at once (4 rows × 1 token for decode,
//      4 rows × 4 tokens for prefill) so each loaded+sign-extended int8 block feeds multiple
//      accumulators, raising register-level arithmetic intensity and giving independent FMLA
//      chains (ILP) that hide the accumulate latency.
//
// Numerics. Each cell folds its blocks in order through a single 4-lane float accumulator and
// reduces (a0+a2)+(a1+a3) at the end — bit-FOR-bit qgemm8cell(...,lanes=4): SDOT gives the same
// four int32 lane partials (lane k = byte-group 4k..4k+3 of each 16-byte half), SCVTF is exact
// (|partial| < 2^24), VFMLA matches math.FMA (one rounding), and the lane reduction uses the same
// pairwise tree. Pinned by TestQGemm8TileNEONMatchesCell / TestQMatRows4NEONMatchesCell.
//
// SDOT Vd.4S, Vn.16B, Vm.16B and SCVTF Vd.4S, Vn.4S have no Go-assembler mnemonic, so both are
// emitted via WORD:
//   SDOT  = 0x4E809400 | (Vm<<16) | (Vn<<5) | Vd
//   SCVTF = 0x4E21D800 | (Vn<<5)  | Vd        (signed int32 -> f32, vector .4S)

// qmatrows4NEON computes y[0..3] = dot(weight row i, activation) for four consecutive Q8_0 weight
// rows against one Q8_0 activation vector — the decode GEMV micro-kernel. Weight row i, block b at
// qw + i*in + b*32 (scale dw + i*nblk + b); activation block b at qx + b*32 (scale dx + b). y is
// four consecutive floats.
//
// func qmatrows4NEON(qw, qx *int8, dw, dx *float32, in, nblk int, y *float32)
TEXT ·qmatrows4NEON(SB), NOSPLIT, $0-56
	MOVD qw+0(FP), R0   // weight row 0 codes
	MOVD qx+8(FP), R4   // activation codes
	MOVD dw+16(FP), R5  // weight row 0 scales
	MOVD dx+24(FP), R9  // activation scales
	MOVD in+32(FP), R12 // inner dim (bytes per weight row)
	MOVD nblk+40(FP), R10
	MOVD y+48(FP), R11

	// Row code pointers R0..R3 = qw + i*in.
	ADD R0, R12, R1
	ADD R1, R12, R2
	ADD R2, R12, R3
	// Row scale pointers R5..R8 = dw + i*nblk (floats -> bytes = nblk*4).
	LSL $2, R10, R13    // R13 = nblk*4
	ADD R5, R13, R6
	ADD R6, R13, R7
	ADD R7, R13, R8

	// Zero the four float accumulators (V0..V3).
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

	CBZ R10, reduce

block:
	// Activation block (32 codes -> V4 low16, V5 high16) and its scale (F18), shared by all rows.
	VLD1.P 16(R4), [V4.B16]
	VLD1.P 16(R4), [V5.B16]
	FMOVS (R9), F18
	ADD $4, R9

	// --- row 0 -> V0 ---
	VLD1.P 16(R0), [V6.B16]
	VLD1.P 16(R0), [V7.B16]
	VEOR V16.B16, V16.B16, V16.B16
	WORD $(0x4E809400 | (4<<16) | (6<<5) | 16)  // SDOT V16.4S, V6.16B, V4.16B
	WORD $(0x4E809400 | (5<<16) | (7<<5) | 16)  // SDOT V16.4S, V7.16B, V5.16B
	WORD $(0x4E21D800 | (16<<5) | 16)           // SCVTF V16.4S, V16.4S
	FMOVS (R5), F19
	ADD $4, R5
	FMULS F18, F19, F20                         // s = dx[b]*dw0[b]
	VDUP V20.S[0], V17.S4
	VFMLA V17.S4, V16.S4, V0.S4                 // V0 += pf * s

	// --- row 1 -> V1 ---
	VLD1.P 16(R1), [V6.B16]
	VLD1.P 16(R1), [V7.B16]
	VEOR V16.B16, V16.B16, V16.B16
	WORD $(0x4E809400 | (4<<16) | (6<<5) | 16)
	WORD $(0x4E809400 | (5<<16) | (7<<5) | 16)
	WORD $(0x4E21D800 | (16<<5) | 16)
	FMOVS (R6), F19
	ADD $4, R6
	FMULS F18, F19, F20
	VDUP V20.S[0], V17.S4
	VFMLA V17.S4, V16.S4, V1.S4

	// --- row 2 -> V2 ---
	VLD1.P 16(R2), [V6.B16]
	VLD1.P 16(R2), [V7.B16]
	VEOR V16.B16, V16.B16, V16.B16
	WORD $(0x4E809400 | (4<<16) | (6<<5) | 16)
	WORD $(0x4E809400 | (5<<16) | (7<<5) | 16)
	WORD $(0x4E21D800 | (16<<5) | 16)
	FMOVS (R7), F19
	ADD $4, R7
	FMULS F18, F19, F20
	VDUP V20.S[0], V17.S4
	VFMLA V17.S4, V16.S4, V2.S4

	// --- row 3 -> V3 ---
	VLD1.P 16(R3), [V6.B16]
	VLD1.P 16(R3), [V7.B16]
	VEOR V16.B16, V16.B16, V16.B16
	WORD $(0x4E809400 | (4<<16) | (6<<5) | 16)
	WORD $(0x4E809400 | (5<<16) | (7<<5) | 16)
	WORD $(0x4E21D800 | (16<<5) | 16)
	FMOVS (R8), F19
	ADD $4, R8
	FMULS F18, F19, F20
	VDUP V20.S[0], V17.S4
	VFMLA V17.S4, V16.S4, V3.S4

	SUB $1, R10
	CBNZ R10, block

reduce:
	// Reduce each accumulator V_i = [a0,a1,a2,a3] to (a0+a2)+(a1+a3) and store y[i].
	// V0 -> y[0]
	VMOV V0.S[1], V28.S[0]
	VMOV V0.S[2], V29.S[0]
	VMOV V0.S[3], V30.S[0]
	FADDS F0, F29, F29   // a2 + a0
	FADDS F28, F30, F30  // a3 + a1
	FADDS F29, F30, F31
	FMOVS F31, (R11)
	// V1 -> y[1]
	VMOV V1.S[1], V28.S[0]
	VMOV V1.S[2], V29.S[0]
	VMOV V1.S[3], V30.S[0]
	FADDS F1, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 4(R11)
	// V2 -> y[2]
	VMOV V2.S[1], V28.S[0]
	VMOV V2.S[2], V29.S[0]
	VMOV V2.S[3], V30.S[0]
	FADDS F2, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 8(R11)
	// V3 -> y[3]
	VMOV V3.S[1], V28.S[0]
	VMOV V3.S[2], V29.S[0]
	VMOV V3.S[3], V30.S[0]
	FADDS F3, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 12(R11)
	RET
