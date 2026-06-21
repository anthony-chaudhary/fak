//go:build arm64

#include "textflag.h"

// quantizeRowAsmNEON quantizes one activation row of nblk Q8_0 blocks (nblk*32 floats at x) into
// codes q (nblk*32 int8) and per-block scales d (nblk float32), with NEON. Per block: amax via a
// sign-cleared integer VUMAX-tree + UMAXV; dd = amax/127; inv = 1/dd; then FMUL·inv + FRINTA
// (round to nearest, ties away — matching the now-deterministic q8round) + FCVTZS + saturating
// SQXTN narrow to int8. Bit-identical to quantizeRowQ8scalar (with q8round //go:noinline) — pinned
// by TestQuantizeRowAsmMatchesScalar. amax==0 needs no branch: inv=+inf, 0*inf=NaN, FCVTZS(NaN)=0,
// d=0/127=0, so the block yields d=0/codes=0 exactly like the scalar's dd==0 case.
//
// Encodings (verified against clang otool -t):
//   FMUL Vd.4S,Vn.4S,V31.4S = 0x6E3FDC00 | (Vn<<5)|Vd
//   FRINTA Vd.4S,Vn.4S      = 0x6E218800 | (Vn<<5)|Vd
//   FCVTZS Vd.4S,Vn.4S      = 0x4EA1B800 | (Vn<<5)|Vd
//   SQXTN  Vd.4H,Vn.4S      = 0x0E614800 | (Vn<<5)|Vd ; SQXTN2 Vd.8H,Vn.4S = 0x4E614800 |...
//   SQXTN  Vd.8B,Vn.8H      = 0x0E214800 | (Vn<<5)|Vd ; SQXTN2 Vd.16B,Vn.8H = 0x4E214800 |...
//   UMAXV  Sd,Vn.4S         = 0x6EB0A800 | (Vn<<5)|Vd
//
// func quantizeRowAsmNEON(x *float32, q *int8, d *float32, nblk int)
TEXT ·quantizeRowAsmNEON(SB), NOSPLIT, $0-32
	MOVD x+0(FP), R0
	MOVD q+8(FP), R1
	MOVD d+16(FP), R2
	MOVD nblk+24(FP), R3

	MOVD $0x7FFFFFFF, R4
	VDUP R4, V24.S4        // sign-clear mask
	MOVD $0x42FE0000, R4
	VMOV R4, V25.S[0]      // 127.0
	MOVD $0x3F800000, R4
	VMOV R4, V26.S[0]      // 1.0

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

	VAND V0.B16, V24.B16, V16.B16
	VAND V1.B16, V24.B16, V17.B16
	VAND V2.B16, V24.B16, V18.B16
	VAND V3.B16, V24.B16, V19.B16
	VAND V4.B16, V24.B16, V20.B16
	VAND V5.B16, V24.B16, V21.B16
	VAND V6.B16, V24.B16, V22.B16
	VAND V7.B16, V24.B16, V23.B16
	VUMAX V16.S4, V17.S4, V16.S4
	VUMAX V18.S4, V19.S4, V18.S4
	VUMAX V20.S4, V21.S4, V20.S4
	VUMAX V22.S4, V23.S4, V22.S4
	VUMAX V16.S4, V18.S4, V16.S4
	VUMAX V20.S4, V22.S4, V20.S4
	VUMAX V16.S4, V20.S4, V16.S4
	WORD $(0x6EB0A800 | (16<<5) | 28) // UMAXV S28, V16.4S  -> amax in F28

	FDIVS F25, F28, F29   // dd = amax / 127
	FMOVS F29, (R2)       // store scale (dd, == 0 for an all-zero/underflow block)
	ADD $4, R2
	// dd == 0: the scalar zeros the codes (no 1/dd). Match it, else FCVTZS(0*inf=+inf)
	// would saturate to 127.
	VMOV V29.S[0], R5
	CBZ R5, zeroblk

	FDIVS F29, F26, F30   // inv = 1.0 / dd
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
	WORD $(0x0E614800 | (0<<5) | 0)  // SQXTN  V0.4H,  V0.4S
	WORD $(0x4E614800 | (1<<5) | 0)  // SQXTN2 V0.8H,  V1.4S
	WORD $(0x0E614800 | (2<<5) | 2)  // SQXTN  V2.4H,  V2.4S
	WORD $(0x4E614800 | (3<<5) | 2)  // SQXTN2 V2.8H,  V3.4S
	WORD $(0x0E614800 | (4<<5) | 4)  // SQXTN  V4.4H,  V4.4S
	WORD $(0x4E614800 | (5<<5) | 4)  // SQXTN2 V4.8H,  V5.4S
	WORD $(0x0E614800 | (6<<5) | 6)  // SQXTN  V6.4H,  V6.4S
	WORD $(0x4E614800 | (7<<5) | 6)  // SQXTN2 V6.8H,  V7.4S
	// narrow int16 -> int8 (saturating)
	WORD $(0x0E214800 | (0<<5) | 0)  // SQXTN  V0.8B,  V0.8H
	WORD $(0x4E214800 | (2<<5) | 0)  // SQXTN2 V0.16B, V2.8H
	WORD $(0x0E214800 | (4<<5) | 4)  // SQXTN  V4.8B,  V4.8H
	WORD $(0x4E214800 | (6<<5) | 4)  // SQXTN2 V4.16B, V6.8H
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
