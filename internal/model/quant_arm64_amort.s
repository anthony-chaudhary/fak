//go:build arm64

#include "textflag.h"

// qdot8amortNEON — the issue #477 amortized-FP-reduction Q8_0 decode dot. It computes the
// same Q8_0 inner product as qdot8scalar / qdot8asm:
//   Σ_b ( float32(Σ_{i∈block b} qw[i]·qx[i]) · dw[b] · dx[b] )
// but it AMORTIZES the per-block float reduction the way llama.cpp's ggml NEON kernel does
// (M3-LLAMACPP-RESULTS.md sec. 4): instead of the bit-identical qdot8asm's per-block
// VADDV→VMOV→SCVTF→FMADDS latency chain (one cross-lane integer reduce + one scalar convert
// PER block), it keeps each block's int32 dot as a 4-lane vector, converts that whole vector
// to float (one SCVTF.4S, NO horizontal reduce), and FMLAs it into a 4-lane float accumulator
// scaled by dw[b]·dx[b]. The single horizontal float reduce is paid ONCE at the end. Four
// independent accumulators (V8..V11) hide the SDOT→SCVTF→FMLA latency across blocks.
//
// WHY THIS IS PROVABLY BOUNDED (and therefore correct, just not bit-identical):
//   • Each 32-wide block is two SDOTs into a 4-lane int32 accumulator; each lane holds the sum
//     of 8 int8·int8 products, so |lane| ≤ 8·127·127 ≈ 1.29e5 and the block sum ≤ 32·127·127 ≈
//     5.2e5 — far inside int32 AND inside exact-integer float32 (2^24 ≈ 1.67e7). So SCVTF of
//     each lane is EXACT, and Σ_lanes(float(lane)) computed in f32 is exact = float(isum_b).
//   • The float COMBINE across blocks is reordered: qdot8scalar accumulates
//     (((float(isum_b)·dw_b)·dx_b)) sequentially block 0,1,2,…; this kernel accumulates four
//     lane-parallel running sums and reduces them at the end. Reordering exact-integer terms
//     scaled by per-block floats changes only the float rounding order — NOT the integer dot.
//   So this is NOT bit-identical to qdot8scalar (it must NOT be gated by
//   TestQdot8NEONMatchesScalar); its correctness contract is argmax-exact + cosine vs the
//   reference (TestQdot8AmortArgmaxAndCosine here, and the HF oracle on the M3), exactly the
//   posture the prefill GEMM / decode unroll4 paths already take.
//
// i8mm/SMMLA is the documented FOLLOW-UP (quant_arm64.go detectI8MM gates a future tier):
//   the SMMLA matrix-multiply-accumulate path needs a hardware-validated 2×8·8×2 byte layout
//   that cannot be byte-tested on the x86 authoring host, so per issue #477's honest-block
//   clause this lands the amortized-reduction variant (SDOT-based, FEAT_DotProd, provably
//   bounded) and leaves SMMLA as the next tier. The dispatch + FEAT_I8MM detection are in
//   place so the SMMLA kernel slots in without re-plumbing.
//
// All four vector ops are WORD-emitted with the SAME encodings the in-tree unroll4 kernel uses
// (quant_arm64_unroll.s — proven to assemble and run on Apple Silicon), only the register
// fields differ: the Go arm64 assembler has no SDOT/SCVTF.4S/FMLA-by-element mnemonic. SDOT
// Vd.4S,Vn.16B,Vm.16B = 0x4E809400|Rm<<16|Rn<<5|Rd; SCVTF Vd.4S,Vn.4S = 0x4E21D800|Rn<<5|Rd;
// FMLA Vd.4S,Vn.4S,Vm.S[0] = 0x4F801000|(M<<20)|(Rm<<16)|Rn<<5|Rd with Vm=(M<<4)|Rm (here the
// scale is in V3, so M=0, Rm=3); FADD Vd.4S,Vn.4S,Vm.4S = 0x4E20D400|Rm<<16|Rn<<5|Rd.
//
// func qdot8amortNEON(qw, qx *int8, dw, dx *float32, nblk int) float32
TEXT ·qdot8amortNEON(SB), NOSPLIT, $0-44
	MOVD qw+0(FP), R0
	MOVD qx+8(FP), R1
	MOVD dw+16(FP), R2
	MOVD dx+24(FP), R3
	MOVD nblk+32(FP), R4

	// four independent 4-lane float accumulators = 0.0
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16

loop4:
	CMP $4, R4
	BLT tail

	// block 0 -> acc V8, int scratch V16
	VLD1.P 16(R0), [V4.B16]
	VLD1.P 16(R0), [V5.B16]
	VLD1.P 16(R1), [V6.B16]
	VLD1.P 16(R1), [V7.B16]
	VEOR V16.B16, V16.B16, V16.B16
	WORD $(0x4E809400 | (6<<16) | (4<<5) | 16)          // SDOT  V16.4S, V4.16B, V6.16B
	WORD $(0x4E809400 | (7<<16) | (5<<5) | 16)          // SDOT  V16.4S, V5.16B, V7.16B
	WORD $(0x4E21D800 | (16<<5) | 16)                   // SCVTF V16.4S, V16.4S
	FMOVS (R2), F1                                      // dw[b]
	FMOVS (R3), F2                                      // dx[b]
	FMULS F1, F2, F3                                    // F3 = dw[b]*dx[b]  (V3 -> M=0,Rm=3)
	WORD $(0x4F801000 | (3<<16) | (16<<5) | 8)          // FMLA  V8.4S, V16.4S, V3.S[0]

	// block 1 -> acc V9, int scratch V17
	VLD1.P 16(R0), [V4.B16]
	VLD1.P 16(R0), [V5.B16]
	VLD1.P 16(R1), [V6.B16]
	VLD1.P 16(R1), [V7.B16]
	VEOR V17.B16, V17.B16, V17.B16
	WORD $(0x4E809400 | (6<<16) | (4<<5) | 17)          // SDOT  V17.4S, V4.16B, V6.16B
	WORD $(0x4E809400 | (7<<16) | (5<<5) | 17)          // SDOT  V17.4S, V5.16B, V7.16B
	WORD $(0x4E21D800 | (17<<5) | 17)                   // SCVTF V17.4S, V17.4S
	FMOVS 4(R2), F1
	FMOVS 4(R3), F2
	FMULS F1, F2, F3
	WORD $(0x4F801000 | (3<<16) | (17<<5) | 9)          // FMLA  V9.4S, V17.4S, V3.S[0]

	// block 2 -> acc V10, int scratch V18
	VLD1.P 16(R0), [V4.B16]
	VLD1.P 16(R0), [V5.B16]
	VLD1.P 16(R1), [V6.B16]
	VLD1.P 16(R1), [V7.B16]
	VEOR V18.B16, V18.B16, V18.B16
	WORD $(0x4E809400 | (6<<16) | (4<<5) | 18)          // SDOT  V18.4S, V4.16B, V6.16B
	WORD $(0x4E809400 | (7<<16) | (5<<5) | 18)          // SDOT  V18.4S, V5.16B, V7.16B
	WORD $(0x4E21D800 | (18<<5) | 18)                   // SCVTF V18.4S, V18.4S
	FMOVS 8(R2), F1
	FMOVS 8(R3), F2
	FMULS F1, F2, F3
	WORD $(0x4F801000 | (3<<16) | (18<<5) | 10)         // FMLA  V10.4S, V18.4S, V3.S[0]

	// block 3 -> acc V11, int scratch V19
	VLD1.P 16(R0), [V4.B16]
	VLD1.P 16(R0), [V5.B16]
	VLD1.P 16(R1), [V6.B16]
	VLD1.P 16(R1), [V7.B16]
	VEOR V19.B16, V19.B16, V19.B16
	WORD $(0x4E809400 | (6<<16) | (4<<5) | 19)          // SDOT  V19.4S, V4.16B, V6.16B
	WORD $(0x4E809400 | (7<<16) | (5<<5) | 19)          // SDOT  V19.4S, V5.16B, V7.16B
	WORD $(0x4E21D800 | (19<<5) | 19)                   // SCVTF V19.4S, V19.4S
	FMOVS 12(R2), F1
	FMOVS 12(R3), F2
	FMULS F1, F2, F3
	WORD $(0x4F801000 | (3<<16) | (19<<5) | 11)         // FMLA  V11.4S, V19.4S, V3.S[0]

	ADD $16, R2
	ADD $16, R3
	SUB $4, R4
	B   loop4

tail:
	// remainder blocks (0..3) folded into V8 via int scratch V16
	CBZ R4, combine
tailloop:
	VLD1.P 16(R0), [V4.B16]
	VLD1.P 16(R0), [V5.B16]
	VLD1.P 16(R1), [V6.B16]
	VLD1.P 16(R1), [V7.B16]
	VEOR V16.B16, V16.B16, V16.B16
	WORD $(0x4E809400 | (6<<16) | (4<<5) | 16)          // SDOT  V16.4S, V4.16B, V6.16B
	WORD $(0x4E809400 | (7<<16) | (5<<5) | 16)          // SDOT  V16.4S, V5.16B, V7.16B
	WORD $(0x4E21D800 | (16<<5) | 16)                   // SCVTF V16.4S, V16.4S
	FMOVS (R2), F1
	FMOVS (R3), F2
	FMULS F1, F2, F3
	WORD $(0x4F801000 | (3<<16) | (16<<5) | 8)          // FMLA  V8.4S, V16.4S, V3.S[0]
	ADD $4, R2
	ADD $4, R3
	SUB $1, R4
	CBNZ R4, tailloop

combine:
	// V8 += V9; V10 += V11; V8 += V10   (FADD Vd.4S = 0x4E20D400 | Rm<<16 | Rn<<5 | Rd)
	WORD $(0x4E20D400 | (9<<16) | (8<<5) | 8)
	WORD $(0x4E20D400 | (11<<16) | (10<<5) | 10)
	WORD $(0x4E20D400 | (10<<16) | (8<<5) | 8)
	// reduce 4 float lanes of V8 to a scalar (F8 == V8.S[0]); order not bit-exact (this kernel
	// already forfeited bit-identity) — same lane-fold the unroll4 kernel uses.
	VMOV V8.S[1], V20.S[0]
	VMOV V8.S[2], V21.S[0]
	VMOV V8.S[3], V23.S[0]
	FADDS F8, F20, F20
	FADDS F21, F23, F23
	FADDS F20, F23, F0
	FMOVS F0, ret+40(FP)
	RET
