//go:build arm64 && !(fakaccel && darwin && cgo)

#include "textflag.h"

// quant_arm64_q5k.s — the NEON SDOT integer-reduction kernel for resident Q5_K decode (issue #1126,
// #1124 C2). It computes, for one weight row (nblk super-blocks) and a Q8-quantized activation qx,
// the per-sub-block reductions the shared-Go combine (kQuantCombineRow) folds into the dot:
//
//	I_s = Σ_{l∈sub s} q5[l]*qx[l]      (the SDOT reduction; q5 values 0..31 are positive int8)
//	S_s = Σ_{l∈sub s} qx[l]            (the zero-point sum, done via SDOT vs an all-ones vector)
//
// for all 8 sub-blocks of every super-block, written to IS[b*8+s] / SS[b*8+s]. This kernel owns ONLY
// the integer reductions; the float combine is shared Go, so asm correctness reduces to the
// reductions matching q5kReduceRowScalar bit-for-bit — and they must: SDOT (int8x16→4×int32) and the
// ones-vector dot are associative with no overflow on these ranges (|I_s| <= 32*31*127 ~= 1.26e5,
// |S_s| <= 32*127 ~= 4.1e3), so any lane order yields the same int32.
// Pinned by TestQ5KReduceAsmMatchesScalar (arm64).
//
// SDOT Vd.4S, Vn.16B, Vm.16B has no Go-assembler mnemonic (same as the Q4_K/Q6_K/Q8 kernels), so it is
// emitted via WORD as 0x4E809400 | (Vm<<16) | (Vn<<5) | Vd (the A64 SDOT vector encoding, identical
// to quant_arm64_q4k.s and quant_arm64_q6k.s). S_s uses a second SDOT against an all-ones vector (V13).
//
// Q5_K super-block layout (q5kBlockBytes = 176):
//   header (16 B): d (2 B) + min (2 B) + scales (12 B)
//   qh (32 B) at offset 16
//   ql (128 B) at offset 48
//
// Each super-block has 4 chunks of 64 weights = 8 sub-blocks of 32. Chunk c (c=0..3) reads 32 bytes
// of ql shared across its two sub-blocks:
//   sub-block 2c:   q5 = (ql & 0x0F) | (((qh >> (2*c))   & 0x01) << 4)
//   sub-block 2c+1: q5 = (ql >> 4)   | (((qh >> (2*c+1)) & 0x01) << 4)
//
// In each chunk c, qh_lo (elements 0..15) and qh_hi (elements 16..31) provide the 5th bit for the
// 32 weights. Shifting qh_lo and qh_hi right by 2 after each chunk maps bit (2*c) to bit 0 and
// bit (2*c+1) to bit 1, making the bit extraction uniform across all 4 chunks:
//   sub-block 2c:   (qh & 0x01) << 4
//   sub-block 2c+1: (qh & 0x02) << 3
//
// func q5kReduceRowAsm(row *byte, nblk int, qx *int8, IS, SS *int32)
TEXT ·q5kReduceRowAsm(SB), NOSPLIT, $0-40
	MOVD row+0(FP), R0   // super-block base; advances 176/super-block
	MOVD nblk+8(FP), R1  // super-blocks remaining
	MOVD qx+16(FP), R2   // activation base; advances 32/sub-block (256/super-block)
	MOVD IS+24(FP), R3   // IS write ptr; advances 4/sub-block (32/super-block)
	MOVD SS+32(FP), R4   // SS write ptr; advances 4/sub-block (32/super-block)

	MOVD $0x0F, R10
	VDUP R10, V12.B16    // V12 = 16 bytes of 0x0F (low-nibble mask)
	MOVD $1, R10
	VDUP R10, V13.B16    // V13 = 16 bytes of 1    (bit 0 mask & S_s SDOT ones vector)
	MOVD $2, R10
	VDUP R10, V14.B16    // V14 = 16 bytes of 2    (bit 1 mask)

	CBZ R1, done

sblock:
	ADD    $16, R0, R5       // R5 = qh ptr (offset 16)
	VLD1.P 16(R5), [V0.B16]  // V0 = qh_lo (16 bytes, elements 0..15)
	VLD1.P 16(R5), [V1.B16]  // V1 = qh_hi (16 bytes, elements 16..31)
	// R5 is now R0 + 48 (ql field base)
	MOVD   $4, R6            // 4 chunks of 32 bytes (8 sub-blocks total)

chunk:
	// Load 32 bytes of ql into V2 (low 16) and V3 (high 16)
	VLD1.P 16(R5), [V2.B16]
	VLD1.P 16(R5), [V3.B16]

	// --- Sub-block 2c: low nibbles + qh bit 0 ---
	// Extract low nibbles
	VAND  V2.B16, V12.B16, V4.B16
	VAND  V3.B16, V12.B16, V5.B16

	// Extract 5th bit (bit 0 of current V0/V1) shifted to bit 4
	VAND  V0.B16, V13.B16, V6.B16
	VSHL  $4, V6.B16, V6.B16
	VAND  V1.B16, V13.B16, V7.B16
	VSHL  $4, V7.B16, V7.B16

	// Assemble 5-bit weights (0..31)
	VORR  V6.B16, V4.B16, V4.B16 // V4 = q5_lo (elements 0..15)
	VORR  V7.B16, V5.B16, V5.B16 // V5 = q5_hi (elements 16..31)

	// Load 32 activations for sub-block 2c
	VLD1.P 16(R2), [V8.B16]
	VLD1.P 16(R2), [V9.B16]

	// I_{2c} = Σ q5 * qx
	VEOR  V10.B16, V10.B16, V10.B16
	// SDOT V10.4S, V4.16B, V8.16B (Vd=10, Vn=4, Vm=8)
	WORD  $(0x4E809400 | (8<<16) | (4<<5) | 10)
	// SDOT V10.4S, V5.16B, V9.16B (Vd=10, Vn=5, Vm=9)
	WORD  $(0x4E809400 | (9<<16) | (5<<5) | 10)
	VADDV V10.S4, V16

	// S_{2c} = Σ qx (SDOT against all-ones vector V13)
	VEOR  V11.B16, V11.B16, V11.B16
	// SDOT V11.4S, V8.16B, V13.16B (Vd=11, Vn=8, Vm=13)
	WORD  $(0x4E809400 | (13<<16) | (8<<5) | 11)
	// SDOT V11.4S, V9.16B, V13.16B (Vd=11, Vn=9, Vm=13)
	WORD  $(0x4E809400 | (13<<16) | (9<<5) | 11)
	VADDV V11.S4, V17

	// Store I_{2c} and S_{2c}
	VMOV  V16.S[0], R7
	MOVW  R7, (R3)
	ADD   $4, R3
	VMOV  V17.S[0], R7
	MOVW  R7, (R4)
	ADD   $4, R4

	// --- Sub-block 2c+1: high nibbles + qh bit 1 ---
	// Extract high nibbles from V2 and V3
	VUSHR $4, V2.B16, V4.B16
	VUSHR $4, V3.B16, V5.B16

	// Extract 5th bit (bit 1 of current V0/V1) shifted to bit 4
	VAND  V0.B16, V14.B16, V6.B16
	VSHL  $3, V6.B16, V6.B16
	VAND  V1.B16, V14.B16, V7.B16
	VSHL  $3, V7.B16, V7.B16

	// Assemble 5-bit weights (0..31)
	VORR  V6.B16, V4.B16, V4.B16 // V4 = q5_lo (elements 0..15)
	VORR  V7.B16, V5.B16, V5.B16 // V5 = q5_hi (elements 16..31)

	// Load 32 activations for sub-block 2c+1
	VLD1.P 16(R2), [V8.B16]
	VLD1.P 16(R2), [V9.B16]

	// I_{2c+1} = Σ q5 * qx
	VEOR  V10.B16, V10.B16, V10.B16
	// SDOT V10.4S, V4.16B, V8.16B (Vd=10, Vn=4, Vm=8)
	WORD  $(0x4E809400 | (8<<16) | (4<<5) | 10)
	// SDOT V10.4S, V5.16B, V9.16B (Vd=10, Vn=5, Vm=9)
	WORD  $(0x4E809400 | (9<<16) | (5<<5) | 10)
	VADDV V10.S4, V16

	// S_{2c+1} = Σ qx (SDOT against all-ones vector V13)
	VEOR  V11.B16, V11.B16, V11.B16
	// SDOT V11.4S, V8.16B, V13.16B (Vd=11, Vn=8, Vm=13)
	WORD  $(0x4E809400 | (13<<16) | (8<<5) | 11)
	// SDOT V11.4S, V9.16B, V13.16B (Vd=11, Vn=9, Vm=13)
	WORD  $(0x4E809400 | (13<<16) | (9<<5) | 11)
	VADDV V11.S4, V17

	// Store I_{2c+1} and S_{2c+1}
	VMOV  V16.S[0], R7
	MOVW  R7, (R3)
	ADD   $4, R3
	VMOV  V17.S[0], R7
	MOVW  R7, (R4)
	ADD   $4, R4

	// Advance qh bits for next chunk: shift right by 2
	VUSHR $2, V0.B16, V0.B16
	VUSHR $2, V1.B16, V1.B16

	SUB   $1, R6
	CBNZ  R6, chunk

	ADD   $176, R0       // next super-block base (q5kBlockBytes = 176)
	SUB   $1, R1
	CBNZ  R1, sblock

done:
	RET
