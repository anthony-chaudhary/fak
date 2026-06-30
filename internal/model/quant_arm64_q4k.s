//go:build arm64 && !(fakaccel && darwin && cgo)

#include "textflag.h"

// quant_arm64_q4k.s — the NEON SDOT integer-reduction kernel for resident Q4_K decode (plan P2).
// It computes, for one weight row (nblk super-blocks) and a Q8_0-quantized activation qx, the
// per-sub-block reductions the shared-Go combine (q4kCombineRow) folds into the dot:
//
//	I_s = Σ_{l∈sub s} nibble[l]*qx[l]      (the SDOT reduction; nibbles 0..15 are positive int8)
//	S_s = Σ_{l∈sub s} qx[l]                  (the min-term sum)
//
// for all 8 sub-blocks of every super-block, written to IS[b*8+s] / SS[b*8+s]. This kernel owns
// ONLY the integer reductions; the float combine is shared Go, so asm correctness reduces to the
// reductions matching q4kReduceRowScalar bit-for-bit — and they must: SDOT (int8x8->int32) and the
// ones-vector dot that stands in for SADDLV are both associative with no overflow on these ranges
// (|I_s| <= 32*15*127 ~= 6.1e4, |S_s| <= 32*127 ~= 4.1e3), so any lane order yields the same int32.
// Pinned by TestQ4KReduceAsmMatchesScalar.
//
// SDOT Vd.4S, Vn.16B, Vm.16B has no Go-assembler mnemonic (same as the Q8 kernel), so it is
// emitted via WORD as 0x4E809400 | (Rm<<16) | (Rn<<5) | Rd (the A64 SDOT vector encoding; verified
// against the existing Q8 kernel's encodings). S_s uses a second SDOT against an all-ones vector
// (V13) rather than SADDLV, so the only non-WORD NEON ops are VAND/VDUP/VUSHR/VLD1.P/VADDV/VEOR.
//
// Sub-block layout (matches q4kDequantSuperBlock): each super-block's 128-byte q field is 4 chunks
// of 32 bytes; chunk k encodes sub-block 2k (LOW nibble of each byte) and 2k+1 (HIGH nibble). The
// activation qx is 8 contiguous 32-byte sub-blocks per super-block.
//
// func q4kReduceRowAsm(row *byte, nblk int, qx *int8, IS, SS *int32)
TEXT ·q4kReduceRowAsm(SB), NOSPLIT, $0-40
	MOVD row+0(FP), R0   // super-block base; advances 144/super-block
	MOVD nblk+8(FP), R1  // super-blocks remaining
	MOVD qx+16(FP), R2   // activation base; advances 32/sub-block (256/super-block)
	MOVD IS+24(FP), R3   // IS write ptr; advances 4/sub-block (32/super-block)
	MOVD SS+32(FP), R4   // SS write ptr; advances 4/sub-block

	MOVD $0x0F, R10
	VDUP R10, V12.B16    // V12 = 16 bytes of 0x0F (low-nibble mask)
	MOVD $1, R10
	VDUP R10, V13.B16    // V13 = 16 bytes of 1    (S_s = SDOT vs ones, widening to int32)

	CBZ R1, done

sblock:
	ADD  $16, R0, R5     // R5 = q-field ptr (skip d/min/scales: 2+2+12 = 16 B)
	MOVD $4, R6          // chunk counter (4 chunks of 32 B = 128 B q field)

chunk:
	// 32 q-bytes -> V0 (low 16), V1 (high 16). Two single-reg post-increment loads.
	VLD1.P 16(R5), [V0.B16]
	VLD1.P 16(R5), [V1.B16]
	// low nibbles (sub 2k): V2 = V0 & 0x0F, V3 = V1 & 0x0F  (Go arm64 VAND is dst-last:
	// `VAND Vn, Vm, Vd`; writing it dst-first would overwrite the mask V12 — pinned by diag.)
	VAND  V0.B16, V12.B16, V2.B16
	VAND  V1.B16, V12.B16, V3.B16
	// high nibbles (sub 2k+1): V4 = V0 >> 4, V5 = V1 >> 4 (logical; VUSHR is $imm,src,dst)
	VUSHR $4, V0.B16, V4.B16
	VUSHR $4, V1.B16, V5.B16

	// --- sub-block 2k: low nibbles V2,V3 dot activation V6,V7 ---
	VLD1.P 16(R2), [V6.B16]
	VLD1.P 16(R2), [V7.B16]
	VEOR V10.B16, V10.B16, V10.B16           // I-acc = 0
	// SDOT V10.4S, V2.16B, V6.16B (SDOT = 0x4E809400 | Vm<<16 | Vn<<5 | Vd; Vd=10,Vn=2,Vm=6)
	WORD $(0x4E809400 | (6<<16) | (2<<5) | 10)
	// SDOT V10.4S, V3.16B, V7.16B  (Vd=10, Vn=3, Vm=7)
	WORD $(0x4E809400 | (7<<16) | (3<<5) | 10)
	VADDV V10.S4, V14                         // V14.S[0] = I_{2k}
	VEOR V11.B16, V11.B16, V11.B16           // S-acc = 0
	// SDOT V11.4S, V6.16B, V13.16B (ones; Vd=11, Vn=6, Vm=13) -> lane sums of qx
	WORD $(0x4E809400 | (13<<16) | (6<<5) | 11)
	// SDOT V11.4S, V7.16B, V13.16B  (Vd=11, Vn=7, Vm=13)
	WORD $(0x4E809400 | (13<<16) | (7<<5) | 11)
	VADDV V11.S4, V15                         // V15.S[0] = S_{2k}
	VMOV V14.S[0], R7
	MOVW R7, (R3)                             // IS[b*8+2k] = I_{2k}
	ADD  $4, R3
	VMOV V15.S[0], R7
	MOVW R7, (R4)                             // SS[b*8+2k] = S_{2k}
	ADD  $4, R4

	// --- sub-block 2k+1: high nibbles V4,V5 dot activation V6,V7 (reloaded) ---
	VLD1.P 16(R2), [V6.B16]
	VLD1.P 16(R2), [V7.B16]
	VEOR V10.B16, V10.B16, V10.B16
	// SDOT V10.4S, V4.16B, V6.16B  (high nibbles; Vd=10, Vn=4, Vm=6)
	WORD $(0x4E809400 | (6<<16) | (4<<5) | 10)
	// SDOT V10.4S, V5.16B, V7.16B  (Vd=10, Vn=5, Vm=7)
	WORD $(0x4E809400 | (7<<16) | (5<<5) | 10)
	VADDV V10.S4, V14                         // V14.S[0] = I_{2k+1}
	VEOR V11.B16, V11.B16, V11.B16
	WORD $(0x4E809400 | (13<<16) | (6<<5) | 11) // SDOT V11.4S, V6.16B, V13.16B
	WORD $(0x4E809400 | (13<<16) | (7<<5) | 11) // SDOT V11.4S, V7.16B, V13.16B
	VADDV V11.S4, V15                         // V15.S[0] = S_{2k+1}
	VMOV V14.S[0], R7
	MOVW R7, (R3)                             // IS[b*8+2k+1] = I_{2k+1}
	ADD  $4, R3
	VMOV V15.S[0], R7
	MOVW R7, (R4)                             // SS[b*8+2k+1] = S_{2k+1}
	ADD  $4, R4

	SUB  $1, R6
	CBNZ R6, chunk

	ADD  $144, R0                             // next super-block base
	SUB  $1, R1
	CBNZ R1, sblock

done:
	RET
