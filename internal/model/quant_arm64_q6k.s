//go:build arm64

#include "textflag.h"

// quant_arm64_q6k.s — the NEON SDOT integer-reduction kernel for resident Q6_K decode, the arm64
// twin of the AVX2/VNNI amd64 kernel (quant_amd64_kquant.s) and the Q4_K NEON kernel
// (quant_arm64_q4k.s). It computes, for one weight row (nblk super-blocks) and a Q8-quantized
// activation qx, the per-group reductions the shared-Go combine (q6kCombineRow) folds into the dot:
//
//	I_g = Σ_{l∈group g} q6[l]*qx[l]      (the SDOT reduction; q6 values 0..63 are positive int8)
//	S_g = Σ_{l∈group g} qx[l]            (the −32 zero-point sum, done via SDOT vs an all-ones vector)
//
// for all 16 groups of every super-block, written to IS[b*16+g] / SS[b*16+g]. This kernel owns ONLY
// the integer reductions; the float combine (d, sc_g and the −32 shift) is shared Go, so asm
// correctness reduces to the reductions matching q6kReduceRowScalar bit-for-bit — and they must:
// SDOT (int8x16→4×int32) and the ones-vector dot are associative with no overflow on these ranges
// (|I_g| <= 16*63*127 ~= 1.3e5, |S_g| <= 16*127 ~= 2.0e3), so any lane order yields the same int32.
// Pinned by TestQ6KReduceAsmMatchesScalar (arm64).
//
// SDOT Vd.4S, Vn.16B, Vm.16B has no Go-assembler mnemonic (same as the Q4_K/Q8 kernels), so it is
// emitted via WORD as 0x4E809400 | (Rm<<16) | (Rn<<5) | Rd (the A64 SDOT vector encoding, identical
// to quant_arm64_q4k.s). Every other NEON op (VLD1/VAND/VORR/VSHL/VUSHR/VEOR/VADDV/VMOV/VDUP) has a
// mnemonic.
//
// Q6_K super-block layout (q6kBlockBytes = 210): ql[0:128] | qh[128:192] | scales[192:208] | d[208:210].
// Only ql + qh are read here (scales/d feed the float combine). A super-block is two 128-weight
// chunks; each chunk × {is=0,1 scale-half} × {p=0..3 output position} = 16 reduction groups, with
// group g = chunk*8 + is + p*2 EXACTLY as q6kReduceRowScalar / q6kDequantSuperBlock index it. Each
// group's 16 lanes (l ∈ [is*16, is*16+16)) share one 16-byte activation block, so qx is a contiguous
// 16-byte load per (group); the four positions of one (chunk,is) read the same ql_lo/ql_hi/qh bytes:
//
//	p0: q6 = (ql_lo & 0x0F) | ((qh & 0x03) << 4)   activation at xbase + 0
//	p1: q6 = (ql_hi & 0x0F) | ((qh & 0x0C) << 2)   activation at xbase + 32
//	p2: q6 = (ql_lo >> 4)   |  (qh & 0x30)          activation at xbase + 64
//	p3: q6 = (ql_hi >> 4)   | ((qh & 0xC0) >> 2)    activation at xbase + 96
//
// where the qh 2-bit field is isolated with a per-position byte mask BEFORE the lane shift (so the
// result stays in-byte: bits 4-5, value ∈ {0,16,32,48}), then OR'd onto the 0..15 nibble — the same
// isolate-then-shift invariant the amd64 kernel uses. The four (chunk,is) sections differ only in
// their ql/qh/qx byte offsets and the IS/SS base group, tabulated inline below.

// BUILDQ reconstructs the four position q6 vectors (V3=p0, V5=p1, V4=p2, V6=p3) from the loaded
// ql_lo (V0), ql_hi (V1) and qh (V2), using the mask constants V12=0x0F, V16=0x03, V17=0x0C,
// V18=0x30, V19=0xC0. V7..V10 are scratch.
#define BUILDQ \
	VAND  V0.B16, V12.B16, V3.B16 \
	VUSHR $4, V0.B16, V4.B16 \
	VAND  V1.B16, V12.B16, V5.B16 \
	VUSHR $4, V1.B16, V6.B16 \
	VAND  V2.B16, V16.B16, V7.B16 \
	VSHL  $4, V7.B16, V7.B16 \
	VORR  V7.B16, V3.B16, V3.B16 \
	VAND  V2.B16, V17.B16, V8.B16 \
	VSHL  $2, V8.B16, V8.B16 \
	VORR  V8.B16, V5.B16, V5.B16 \
	VAND  V2.B16, V18.B16, V9.B16 \
	VORR  V9.B16, V4.B16, V4.B16 \
	VAND  V2.B16, V19.B16, V10.B16 \
	VUSHR $2, V10.B16, V10.B16 \
	VORR  V10.B16, V6.B16, V6.B16

// GDOT(Q) does one group: load the 16-byte activation at (R7), SDOT it against the q6 vector VQ for
// the I_g reduction and against the all-ones vector V13 for the S_g reduction, horizontal-sum each
// (VADDV) and store the int32 results at (R8)/(R9). Then advance the activation pointer by 32 bytes
// (next position's block) and the IS/SS pointers by 8 bytes (next group = +2 group slots).
#define GDOT(Q) \
	VLD1 (R7), [V21.B16] \
	VEOR V22.B16, V22.B16, V22.B16 \
	WORD $(0x4E809400 | (21<<16) | (Q<<5) | 22) \
	VADDV V22.S4, V14 \
	VEOR V23.B16, V23.B16, V23.B16 \
	WORD $(0x4E809400 | (13<<16) | (21<<5) | 23) \
	VADDV V23.S4, V15 \
	VMOV V14.S[0], R16 \
	MOVW R16, (R8) \
	VMOV V15.S[0], R16 \
	MOVW R16, (R9) \
	ADD $32, R7 \
	ADD $8, R8 \
	ADD $8, R9

// func q6kReduceRowAsm(row *byte, nblk int, qx *int8, IS, SS *int32)
TEXT ·q6kReduceRowAsm(SB), NOSPLIT, $0-40
	MOVD row+0(FP), R0   // super-block base; advances 210/super-block
	MOVD nblk+8(FP), R1  // super-blocks remaining
	MOVD qx+16(FP), R2   // activation base; advances 256/super-block
	MOVD IS+24(FP), R3   // IS write base; advances 64/super-block (16 groups × 4 B)
	MOVD SS+32(FP), R4   // SS write base; advances 64/super-block

	MOVD $0x0F, R10
	VDUP R10, V12.B16
	MOVD $1, R10
	VDUP R10, V13.B16    // all-ones (S_g = SDOT vs ones, widening to int32)
	MOVD $0x03, R10
	VDUP R10, V16.B16
	MOVD $0x0C, R10
	VDUP R10, V17.B16
	MOVD $0x30, R10
	VDUP R10, V18.B16
	MOVD $0xC0, R10
	VDUP R10, V19.B16

	CBZ R1, done

sblock:
	// Section A — chunk 0, is 0: ql_lo@+0, ql_hi@+32, qh@+128, qx@+0, groups 0,2,4,6 (IS@+0).
	MOVD R0, R5
	VLD1 (R5), [V0.B16]
	ADD  $32, R5
	VLD1 (R5), [V1.B16]
	ADD  $128, R0, R6
	VLD1 (R6), [V2.B16]
	BUILDQ
	MOVD R2, R7
	MOVD R3, R8
	MOVD R4, R9
	GDOT(3)
	GDOT(5)
	GDOT(4)
	GDOT(6)

	// Section B — chunk 0, is 1: ql_lo@+16, ql_hi@+48, qh@+144, qx@+16, groups 1,3,5,7 (IS@+4).
	ADD  $16, R0, R5
	VLD1 (R5), [V0.B16]
	ADD  $32, R5
	VLD1 (R5), [V1.B16]
	ADD  $144, R0, R6
	VLD1 (R6), [V2.B16]
	BUILDQ
	ADD  $16, R2, R7
	ADD  $4, R3, R8
	ADD  $4, R4, R9
	GDOT(3)
	GDOT(5)
	GDOT(4)
	GDOT(6)

	// Section C — chunk 1, is 0: ql_lo@+64, ql_hi@+96, qh@+160, qx@+128, groups 8,10,12,14 (IS@+32).
	ADD  $64, R0, R5
	VLD1 (R5), [V0.B16]
	ADD  $32, R5
	VLD1 (R5), [V1.B16]
	ADD  $160, R0, R6
	VLD1 (R6), [V2.B16]
	BUILDQ
	ADD  $128, R2, R7
	ADD  $32, R3, R8
	ADD  $32, R4, R9
	GDOT(3)
	GDOT(5)
	GDOT(4)
	GDOT(6)

	// Section D — chunk 1, is 1: ql_lo@+80, ql_hi@+112, qh@+176, qx@+144, groups 9,11,13,15 (IS@+36).
	ADD  $80, R0, R5
	VLD1 (R5), [V0.B16]
	ADD  $32, R5
	VLD1 (R5), [V1.B16]
	ADD  $176, R0, R6
	VLD1 (R6), [V2.B16]
	BUILDQ
	ADD  $144, R2, R7
	ADD  $36, R3, R8
	ADD  $36, R4, R9
	GDOT(3)
	GDOT(5)
	GDOT(4)
	GDOT(6)

	ADD  $210, R0         // next super-block base
	ADD  $256, R2         // next super-block activations
	ADD  $64, R3          // next super-block IS (16 groups × 4 B)
	ADD  $64, R4          // next super-block SS
	SUB  $1, R1
	CBNZ R1, sblock

done:
	RET
