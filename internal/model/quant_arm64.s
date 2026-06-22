//go:build arm64

#include "textflag.h"

// qdot8asm computes the Q8_0 inner product over nblk 32-wide blocks:
//   Σ_b ( float32(Σ_{i∈block b} qw[i]*qx[i]) * dw[b] * dx[b] )
// using NEON SDOT (FEAT_DotProd — present on all Apple Silicon and ARMv8.4+). This is the
// arm64 twin of the amd64 AVX2/AVX-512 kernel: it turns the Q8_0 byte-savings into real
// speed. The scalar int8 dot (qdot8scalar) is compute-bound — Go emits a per-byte
// sign-extend + scalar imul, so streaming 3.5× fewer weight bytes than f32 bought nothing
// on this arch (int8 measured 0.95× f32, i.e. no win) until the dot itself goes SIMD.
//
// Bit-identity: each block's int32 isum is computed by two SDOTs into a 4-lane int32
// accumulator then reduced with VADDV. Integer addition is associative with no overflow (a
// block sum is bounded by 32·127·127 ≈ 5.2e5, far inside int32), so the lane-reduced isum
// equals qdot8scalar's exactly; the per-block float combine is done in the SAME order as the
// scalar reference — (((float(isum)·dw[b])·dx[b]), accumulated block 0,1,2,…) — so this is
// BIT-IDENTICAL to qdot8scalar and TestQdot8NEONMatchesScalar pins it.
//
// SDOT Vd.4S, Vn.16B, Vm.16B is emitted via WORD (base 0x4E809400 | Rm<<16 | Rn<<5 | Rd);
// the Go arm64 assembler has no SDOT mnemonic. The encodings below are pinned by the test.
//
// func qdot8asm(qw, qx *int8, dw, dx *float32, nblk int) float32
TEXT ·qdot8asm(SB), NOSPLIT, $0-44
	MOVD qw+0(FP), R0
	MOVD qx+8(FP), R1
	MOVD dw+16(FP), R2
	MOVD dx+24(FP), R3
	MOVD nblk+32(FP), R4
	VEOR V10.B16, V10.B16, V10.B16 // acc = 0.0 (low S lane)
	CBZ  R4, done

loop:
	VLD1.P 16(R0), [V0.B16] // weights, block low 16
	VLD1.P 16(R0), [V1.B16] // weights, block high 16
	VLD1.P 16(R1), [V2.B16] // acts,    block low 16
	VLD1.P 16(R1), [V3.B16] // acts,    block high 16
	VEOR V4.B16, V4.B16, V4.B16    // isum lanes = 0
	WORD $0x4e829404               // SDOT V4.4S, V0.16B, V2.16B
	WORD $0x4e839424               // SDOT V4.4S, V1.16B, V3.16B
	VADDV V4.S4, V5                // isum = lane0+lane1+lane2+lane3 (int32) -> V5.S[0]
	VMOV  V5.S[0], R5
	SCVTFWS R5, F6                 // float32(isum)
	FMOVS (R2), F7                 // dw[b]
	FMOVS (R3), F8                 // dx[b]
	FMULS F7, F6, F6              // A = isum*dw  (standalone product, rounded — matches gc)
	// acc = A*dx + acc as ONE fused multiply-add. gc auto-fuses `acc += isum*dw*dx` into
	// FMADD on arm64 (it does NOT on amd64 — why the amd64 kernel uses a separate mul+add and
	// still matches its scalar). Matching that single-rounding fusion is what makes this
	// bit-identical to qdot8scalar on arm64. Fd = Fn*Fm + Fa: F10 = F6*F8 + F10.
	FMADDS F8, F10, F6, F10
	ADD  $4, R2
	ADD  $4, R3
	SUB  $1, R4
	CBNZ R4, loop

done:
	FMOVS F10, ret+40(FP)
	RET

// qgemm8tile4x2NEON computes a 4(row)×2(token) Q8_0 output tile with deferred reduction — a
// register-blocked prefill GEMM micro-kernel authored for issue #478, the conservative NR=2
// companion to the NR=4 qgemm8tileNEON (quant_arm64_tile.s). Where the 4×4 tile fills all 16
// 4-lane float accumulators (V0..V15) — the maximum the one-accumulator-per-cell deferred-reduction
// design admits in the 32-register NEON file — this 4×2 tile uses 8 accumulators (V0..V7), freeing
// registers so a single token block (V16..V19) is held while each of the 4 weight rows streams its
// 32-wide block ONCE (V24,V25) and is SDOT'd against both tokens. The per-block int32 dot is kept in
// V26, converted exactly (SCVTF), scaled by dw_i·dx_j, and FMLA-accumulated into the cell's 4-lane
// float accumulator; the horizontal lane reduce is paid ONCE per cell in the store phase, in the
// same (a0+a2)+(a1+a3) pairwise tree qgemm8cell(...,4) uses. Every vector op reuses the EXACT WORD
// encodings the proven 4×4 tile (and quant_arm64_amort.s) emit — SDOT 0x4E809400|Vm<<16|Vn<<5|Vd,
// SCVTF 0x4E21D800|Vn<<5|Vd, FMLA-by-elem 0x4F801000|(1<<20)|(14<<16)|Vn<<5|Vd (V30.S[0] scale) —
// so it is bit-identical to qgemm8cell(...,4); pinned by TestQGemm8Tile4x2NEONMatchesCell.
//
// Layout (same convention as qgemm8tileNEON): weight row i, block b at qw + i*in + b*32 (scale
// dw + i*nblk + b); token j, block b at qx + j*in + b*32 (scale dx + j*nblk + b); result (row i,
// token j) at dst[j*outStride + i] (floats). `in` is the per-row code stride in bytes.
//
// Register map. GPR: R0-R3 weight-row code ptrs, R4-R5 token code ptrs, R8-R11 weight-row scale
// ptrs, R12-R13 token scale ptrs, R16 block counter, R19 dst (token-0 column base), R21 col-stride
// bytes, R22 token-1 column base, R20/R25 prologue scratch. V0-V7 accumulators (V[row*2+col]);
// V16-V19 the two token blocks (lo,hi); V24,V25 the streamed weight row block (lo,hi); V26
// SDOT/SCVTF scratch; F28 dw_i, F29 dx_j, F30 s; V28-V31 reused as reduction temps after the loop.
//
// func qgemm8tile4x2NEON(qw, qx *int8, dw, dx *float32, in, nblk, outStride int, dst *float32)
TEXT ·qgemm8tile4x2NEON(SB), NOSPLIT, $0-64
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
	LSL $2, R16, R25       // R25 = nblk*4 (scale stride)
	ADD R8, R25, R9
	ADD R9, R25, R10
	ADD R10, R25, R11
	ADD R12, R25, R13
	LSL $2, R21, R21       // R21 = outStride*4 (dst column stride, bytes)
	ADD R19, R21, R22

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16

	CBZ R16, store42

tblock42:
	VLD1.P 16(R4), [V16.B16]
	VLD1.P 16(R4), [V17.B16]
	VLD1.P 16(R5), [V18.B16]
	VLD1.P 16(R5), [V19.B16]

	// row 0 (weights R0, dw R8) -> acc V0 (tok0), V1 (tok1)
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

	// row 1 (weights R1, dw R9) -> acc V2 (tok0), V3 (tok1)
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
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 2) // FMLA V2.4S, V26.4S, V30.S[0]
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (18<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (19<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R13), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 3) // FMLA V3.4S, V26.4S, V30.S[0]

	// row 2 (weights R2, dw R10) -> acc V4 (tok0), V5 (tok1)
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
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 4) // FMLA V4.4S, V26.4S, V30.S[0]
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (18<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (19<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R13), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 5) // FMLA V5.4S, V26.4S, V30.S[0]

	// row 3 (weights R3, dw R11) -> acc V6 (tok0), V7 (tok1)
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
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 6) // FMLA V6.4S, V26.4S, V30.S[0]
	VEOR V26.B16, V26.B16, V26.B16
	WORD $(0x4E809400 | (18<<16) | (24<<5) | 26)
	WORD $(0x4E809400 | (19<<16) | (25<<5) | 26)
	WORD $(0x4E21D800 | (26<<5) | 26)
	FMOVS (R13), F29
	FMULS F28, F29, F30
	WORD $(0x4F801000 | (1<<20) | (14<<16) | (26<<5) | 7) // FMLA V7.4S, V26.4S, V30.S[0]

	ADD $4, R12
	ADD $4, R13
	SUB $1, R16
	CBNZ R16, tblock42

store42:
	// token 0 column (base R19): rows 0..3 = V0,V2,V4,V6 at +0,+4,+8,+12
	VMOV V0.S[1], V28.S[0]
	VMOV V0.S[2], V29.S[0]
	VMOV V0.S[3], V30.S[0]
	FADDS F0, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, (R19)
	VMOV V2.S[1], V28.S[0]
	VMOV V2.S[2], V29.S[0]
	VMOV V2.S[3], V30.S[0]
	FADDS F2, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 4(R19)
	VMOV V4.S[1], V28.S[0]
	VMOV V4.S[2], V29.S[0]
	VMOV V4.S[3], V30.S[0]
	FADDS F4, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 8(R19)
	VMOV V6.S[1], V28.S[0]
	VMOV V6.S[2], V29.S[0]
	VMOV V6.S[3], V30.S[0]
	FADDS F6, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 12(R19)
	// token 1 column (base R22): rows 0..3 = V1,V3,V5,V7
	VMOV V1.S[1], V28.S[0]
	VMOV V1.S[2], V29.S[0]
	VMOV V1.S[3], V30.S[0]
	FADDS F1, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, (R22)
	VMOV V3.S[1], V28.S[0]
	VMOV V3.S[2], V29.S[0]
	VMOV V3.S[3], V30.S[0]
	FADDS F3, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 4(R22)
	VMOV V5.S[1], V28.S[0]
	VMOV V5.S[2], V29.S[0]
	VMOV V5.S[3], V30.S[0]
	FADDS F5, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 8(R22)
	VMOV V7.S[1], V28.S[0]
	VMOV V7.S[2], V29.S[0]
	VMOV V7.S[3], V30.S[0]
	FADDS F7, F29, F29
	FADDS F28, F30, F30
	FADDS F29, F30, F31
	FMOVS F31, 12(R22)
	RET
