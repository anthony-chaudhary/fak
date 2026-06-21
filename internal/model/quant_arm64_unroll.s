//go:build arm64

#include "textflag.h"

// qdot8unroll4NEON — EXPERIMENT: a single Q8_0 dot (one weight row · one activation) with the
// blocks unrolled 4-wide into FOUR INDEPENDENT float accumulators and FOUR INDEPENDENT SDOT
// scratch registers, so the SDOT→SCVTF→FMLA latency chain is hidden by inter-block ILP (the
// textbook latency-hiding GEMV; my 4×4 tile shared one SDOT scratch V26 across all cells and may
// have serialized them). Tests whether fak's NEON Q8 dot is latency-bound (fixable) or capped.
//
// NOT bit-identical to qdot8scalar (4-accumulator reduction order) — this is a throughput probe.
//
// func qdot8unroll4NEON(qw, qx *int8, dw, dx *float32, nblk int) float32
TEXT ·qdot8unroll4NEON(SB), NOSPLIT, $0-44
	MOVD qw+0(FP), R0
	MOVD qx+8(FP), R1
	MOVD dw+16(FP), R2
	MOVD dx+24(FP), R3
	MOVD nblk+32(FP), R4

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

loop4:
	CMP  $4, R4
	BLT  tail

	// block 0 -> acc V0, temp V16
	VLD1.P 16(R0), [V4.B16]
	VLD1.P 16(R0), [V5.B16]
	VLD1.P 16(R1), [V6.B16]
	VLD1.P 16(R1), [V7.B16]
	VEOR V16.B16, V16.B16, V16.B16
	WORD $(0x4E809400 | (6<<16) | (4<<5) | 16) // SDOT V16,V4,V6
	WORD $(0x4E809400 | (7<<16) | (5<<5) | 16) // SDOT V16,V5,V7
	WORD $(0x4E21D800 | (16<<5) | 16)          // SCVTF V16
	FMOVS (R2), F20
	FMOVS (R3), F21
	FMULS F20, F21, F22
	WORD $(0x4F801000 | (1<<20) | (6<<16) | (16<<5) | 0) // FMLA V0, V16, V22.S[0]

	// block 1 -> acc V1, temp V17
	VLD1.P 16(R0), [V4.B16]
	VLD1.P 16(R0), [V5.B16]
	VLD1.P 16(R1), [V6.B16]
	VLD1.P 16(R1), [V7.B16]
	VEOR V17.B16, V17.B16, V17.B16
	WORD $(0x4E809400 | (6<<16) | (4<<5) | 17)
	WORD $(0x4E809400 | (7<<16) | (5<<5) | 17)
	WORD $(0x4E21D800 | (17<<5) | 17)
	FMOVS 4(R2), F20
	FMOVS 4(R3), F21
	FMULS F20, F21, F22
	WORD $(0x4F801000 | (1<<20) | (6<<16) | (17<<5) | 1) // FMLA V1, V17, V22.S[0]

	// block 2 -> acc V2, temp V18
	VLD1.P 16(R0), [V4.B16]
	VLD1.P 16(R0), [V5.B16]
	VLD1.P 16(R1), [V6.B16]
	VLD1.P 16(R1), [V7.B16]
	VEOR V18.B16, V18.B16, V18.B16
	WORD $(0x4E809400 | (6<<16) | (4<<5) | 18)
	WORD $(0x4E809400 | (7<<16) | (5<<5) | 18)
	WORD $(0x4E21D800 | (18<<5) | 18)
	FMOVS 8(R2), F20
	FMOVS 8(R3), F21
	FMULS F20, F21, F22
	WORD $(0x4F801000 | (1<<20) | (6<<16) | (18<<5) | 2) // FMLA V2, V18, V22.S[0]

	// block 3 -> acc V3, temp V19
	VLD1.P 16(R0), [V4.B16]
	VLD1.P 16(R0), [V5.B16]
	VLD1.P 16(R1), [V6.B16]
	VLD1.P 16(R1), [V7.B16]
	VEOR V19.B16, V19.B16, V19.B16
	WORD $(0x4E809400 | (6<<16) | (4<<5) | 19)
	WORD $(0x4E809400 | (7<<16) | (5<<5) | 19)
	WORD $(0x4E21D800 | (19<<5) | 19)
	FMOVS 12(R2), F20
	FMOVS 12(R3), F21
	FMULS F20, F21, F22
	WORD $(0x4F801000 | (1<<20) | (6<<16) | (19<<5) | 3) // FMLA V3, V19, V22.S[0]

	ADD  $16, R2
	ADD  $16, R3
	SUB  $4, R4
	B    loop4

tail:
	// remainder blocks (0..3) folded into V0 via temp V16
	CBZ  R4, combine
tailloop:
	VLD1.P 16(R0), [V4.B16]
	VLD1.P 16(R0), [V5.B16]
	VLD1.P 16(R1), [V6.B16]
	VLD1.P 16(R1), [V7.B16]
	VEOR V16.B16, V16.B16, V16.B16
	WORD $(0x4E809400 | (6<<16) | (4<<5) | 16)
	WORD $(0x4E809400 | (7<<16) | (5<<5) | 16)
	WORD $(0x4E21D800 | (16<<5) | 16)
	FMOVS (R2), F20
	FMOVS (R3), F21
	FMULS F20, F21, F22
	WORD $(0x4F801000 | (1<<20) | (6<<16) | (16<<5) | 0) // FMLA V0, V16, V22.S[0]
	ADD  $4, R2
	ADD  $4, R3
	SUB  $1, R4
	CBNZ R4, tailloop

combine:
	// V0 += V1; V2 += V3; V0 += V2   (FADD Vd.4S = 0x4E20D400 | (Vm<<16)|(Vn<<5)|Vd)
	WORD $(0x4E20D400 | (1<<16) | (0<<5) | 0)
	WORD $(0x4E20D400 | (3<<16) | (2<<5) | 2)
	WORD $(0x4E20D400 | (2<<16) | (0<<5) | 0)
	// reduce 4 lanes of V0 to a scalar (sum; order not bit-exact, throughput probe)
	VMOV V0.S[1], V20.S[0]
	VMOV V0.S[2], V21.S[0]
	VMOV V0.S[3], V23.S[0]
	FADDS F0, F20, F20
	FADDS F21, F23, F23
	FADDS F20, F23, F0
	FMOVS F0, ret+40(FP)
	RET
