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
