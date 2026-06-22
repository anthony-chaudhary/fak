//go:build arm64

#include "textflag.h"

// quantizeVecAsmNEON — the issue #476 NEON DECODE activation quantizer, authored independently of
// the production row kernel (quant_quantize_arm64.s). It quantizes one activation vector of nblk
// Q8_0 blocks (nblk*32 floats at x) into codes q (nblk*32 int8) and per-block scales d (nblk
// float32). Per 32-wide block:
//   amax = max|x|   — sign-cleared (abs) integer max, accumulated by a SEQUENTIAL fold into V16
//                     (integer max is associative+commutative, so the fold yields the SAME amax
//                     bits as quant_quantize_arm64.s's balanced tree; this is the authored
//                     difference, provably bit-equivalent), then UMAXV to a scalar.
//   dd   = amax/127 — the block scale (stored even when 0, like quantizeRowQ8scalar's dd==0 leg).
//   code = round(x*inv) — FMUL by inv=1/dd, FRINTA (round to nearest, ties AWAY — bit-matching the
//                     now-deterministic q8round), FCVTZS to int32, then saturating SQXTN narrow
//                     int32->int16->int8. amax==0: codes are explicitly zeroed (no 1/dd), exactly
//                     reproducing the scalar dd==0 case (and avoiding FCVTZS(0*inf)=saturate-to-127).
//
// Held BIT-IDENTICAL to BOTH quantizeRowQ8scalar AND the production quantizeRowAsmNEON by
// TestQuantizeVecQ8NEONMatchesScalar (a differential oracle). The instruction WORD encodings are
// the same ones quant_quantize_arm64.s verified against clang otool -t:
//   FMUL Vd.4S,Vn.4S,V31.4S = 0x6E3FDC00 | (Vn<<5)|Vd
//   FRINTA Vd.4S,Vn.4S      = 0x6E218800 | (Vn<<5)|Vd
//   FCVTZS Vd.4S,Vn.4S      = 0x4EA1B800 | (Vn<<5)|Vd
//   SQXTN  Vd.4H,Vn.4S      = 0x0E614800 | (Vn<<5)|Vd ; SQXTN2 Vd.8H,Vn.4S = 0x4E614800 |...
//   SQXTN  Vd.8B,Vn.8H      = 0x0E214800 | (Vn<<5)|Vd ; SQXTN2 Vd.16B,Vn.8H = 0x4E214800 |...
//   UMAXV  Sd,Vn.4S         = 0x6EB0A800 | (Vn<<5)|Vd
//
// func quantizeVecAsmNEON(x *float32, q *int8, d *float32, nblk int)
TEXT ·quantizeVecAsmNEON(SB), NOSPLIT, $0-32
	MOVD x+0(FP), R0
	MOVD q+8(FP), R1
	MOVD d+16(FP), R2
	MOVD nblk+24(FP), R3

	MOVD $0x7FFFFFFF, R4
	VDUP R4, V24.S4   // sign-clear (abs) mask
	MOVD $0x42FE0000, R4
	VMOV R4, V25.S[0] // 127.0
	MOVD $0x3F800000, R4
	VMOV R4, V26.S[0] // 1.0

	CBZ R3, done

block:
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R0), [V1.S4]
	VLD1.P 16(R0), [V2.S4]
	VLD1.P 16(R0), [V3.S4]
	VLD1.P 16(R0), [V4.S4]
	VLD1.P 16(R0), [V5.S4]
	VLD1.P 16(R0), [V6.S4]
	VLD1.P 16(R0), [V7.S4]

	// abs(V0) into V16, then fold abs(V1..V7) in one lane-wise integer max at a time. V0..V7 are
	// left untouched (the originals are needed for the FMUL below); V16/V17 are the only scratch.
	VAND V0.B16, V24.B16, V16.B16
	VAND V1.B16, V24.B16, V17.B16
	VUMAX V16.S4, V17.S4, V16.S4
	VAND V2.B16, V24.B16, V17.B16
	VUMAX V16.S4, V17.S4, V16.S4
	VAND V3.B16, V24.B16, V17.B16
	VUMAX V16.S4, V17.S4, V16.S4
	VAND V4.B16, V24.B16, V17.B16
	VUMAX V16.S4, V17.S4, V16.S4
	VAND V5.B16, V24.B16, V17.B16
	VUMAX V16.S4, V17.S4, V16.S4
	VAND V6.B16, V24.B16, V17.B16
	VUMAX V16.S4, V17.S4, V16.S4
	VAND V7.B16, V24.B16, V17.B16
	VUMAX V16.S4, V17.S4, V16.S4
	WORD $(0x6EB0A800 | (16<<5) | 28) // UMAXV S28, V16.4S -> amax in F28

	FDIVS F25, F28, F29 // dd = amax / 127
	FMOVS F29, (R2)     // store scale (== 0 for an all-zero/underflow block)
	ADD $4, R2
	VMOV V29.S[0], R5
	CBZ R5, zeroblk // dd == 0: zero the codes, no 1/dd (mirror scalar dd==0)

	FDIVS F29, F26, F30 // inv = 1.0 / dd
	VDUP V30.S[0], V31.S4

	WORD $(0x6E3FDC00 | (0<<5) | 0) // FMUL V_i, V_i, V31
	WORD $(0x6E3FDC00 | (1<<5) | 1)
	WORD $(0x6E3FDC00 | (2<<5) | 2)
	WORD $(0x6E3FDC00 | (3<<5) | 3)
	WORD $(0x6E3FDC00 | (4<<5) | 4)
	WORD $(0x6E3FDC00 | (5<<5) | 5)
	WORD $(0x6E3FDC00 | (6<<5) | 6)
	WORD $(0x6E3FDC00 | (7<<5) | 7)
	WORD $(0x6E218800 | (0<<5) | 0) // FRINTA V_i, V_i
	WORD $(0x6E218800 | (1<<5) | 1)
	WORD $(0x6E218800 | (2<<5) | 2)
	WORD $(0x6E218800 | (3<<5) | 3)
	WORD $(0x6E218800 | (4<<5) | 4)
	WORD $(0x6E218800 | (5<<5) | 5)
	WORD $(0x6E218800 | (6<<5) | 6)
	WORD $(0x6E218800 | (7<<5) | 7)
	WORD $(0x4EA1B800 | (0<<5) | 0) // FCVTZS V_i, V_i
	WORD $(0x4EA1B800 | (1<<5) | 1)
	WORD $(0x4EA1B800 | (2<<5) | 2)
	WORD $(0x4EA1B800 | (3<<5) | 3)
	WORD $(0x4EA1B800 | (4<<5) | 4)
	WORD $(0x4EA1B800 | (5<<5) | 5)
	WORD $(0x4EA1B800 | (6<<5) | 6)
	WORD $(0x4EA1B800 | (7<<5) | 7)
	// narrow int32 -> int16 (saturating)
	WORD $(0x0E614800 | (0<<5) | 0) // SQXTN  V0.4H,  V0.4S
	WORD $(0x4E614800 | (1<<5) | 0) // SQXTN2 V0.8H,  V1.4S
	WORD $(0x0E614800 | (2<<5) | 2) // SQXTN  V2.4H,  V2.4S
	WORD $(0x4E614800 | (3<<5) | 2) // SQXTN2 V2.8H,  V3.4S
	WORD $(0x0E614800 | (4<<5) | 4) // SQXTN  V4.4H,  V4.4S
	WORD $(0x4E614800 | (5<<5) | 4) // SQXTN2 V4.8H,  V5.4S
	WORD $(0x0E614800 | (6<<5) | 6) // SQXTN  V6.4H,  V6.4S
	WORD $(0x4E614800 | (7<<5) | 6) // SQXTN2 V6.8H,  V7.4S
	// narrow int16 -> int8 (saturating)
	WORD $(0x0E214800 | (0<<5) | 0) // SQXTN  V0.8B,  V0.8H
	WORD $(0x4E214800 | (2<<5) | 0) // SQXTN2 V0.16B, V2.8H
	WORD $(0x0E214800 | (4<<5) | 4) // SQXTN  V4.8B,  V4.8H
	WORD $(0x4E214800 | (6<<5) | 4) // SQXTN2 V4.16B, V6.8H
	VST1.P [V0.B16], 16(R1)
	VST1.P [V4.B16], 16(R1)
	B blockend

zeroblk:
	VEOR V0.B16, V0.B16, V0.B16
	VST1.P [V0.B16], 16(R1)
	VST1.P [V0.B16], 16(R1)

blockend:
	SUB $1, R3
	CBNZ R3, block

done:
	RET
