//go:build arm64

#include "textflag.h"

// GemmTile4x4Int8NEON computes a 4x4 int8 matrix tile multiplication with int32 accumulation:
//   C[i, j] = sum_{k=0..K-1} A[i, k] * B[j, k]
// where A has 4 rows (pointers a0..a3), B has 4 rows (pointers b0..b3), and C is written
// at stride ldc (in int32 elements).
//
// func GemmTile4x4Int8NEON(a0, a1, a2, a3, b0, b1, b2, b3 *int8, c *int32, ldc, k int)
TEXT ·GemmTile4x4Int8NEON(SB), NOSPLIT, $0-88
	MOVD a0+0(FP), R0
	MOVD a1+8(FP), R1
	MOVD a2+16(FP), R2
	MOVD a3+24(FP), R3
	MOVD b0+32(FP), R4
	MOVD b1+40(FP), R5
	MOVD b2+48(FP), R6
	MOVD b3+56(FP), R7
	MOVD c+64(FP), R8
	MOVD ldc+72(FP), R9
	MOVD k+80(FP), R10

	LSR $4, R10, R10    // k_blocks = k / 16
	LSL $2, R9, R9      // ldc in bytes (ldc * 4)

	// Zero the 16 int32 vector accumulators (V0..V15)
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16

	CBZ R10, reduce_int8

loop_int8:
	// Interleaved vector loads from A and B (16 bytes = 16 int8s per row)
	VLD1.P 16(R0), [V16.B16]
	VLD1.P 16(R1), [V17.B16]
	VLD1.P 16(R4), [V20.B16]
	VLD1.P 16(R5), [V21.B16]
	VLD1.P 16(R2), [V18.B16]
	VLD1.P 16(R3), [V19.B16]
	VLD1.P 16(R6), [V22.B16]
	VLD1.P 16(R7), [V23.B16]

	// Accumulate 4x4 outer-product tile using ARM64 NEON SDOT
	// Column 0 (B0: V20) with A0..A3
	WORD $(0x4E809400 | (20<<16) | (16<<5) | 0)   // SDOT V0.4S, V16.16B, V20.16B
	WORD $(0x4E809400 | (20<<16) | (17<<5) | 4)   // SDOT V4.4S, V17.16B, V20.16B
	WORD $(0x4E809400 | (20<<16) | (18<<5) | 8)   // SDOT V8.4S, V18.16B, V20.16B
	WORD $(0x4E809400 | (20<<16) | (19<<5) | 12)  // SDOT V12.4S, V19.16B, V20.16B

	// Column 1 (B1: V21) with A0..A3
	WORD $(0x4E809400 | (21<<16) | (16<<5) | 1)   // SDOT V1.4S, V16.16B, V21.16B
	WORD $(0x4E809400 | (21<<16) | (17<<5) | 5)   // SDOT V5.4S, V17.16B, V21.16B
	WORD $(0x4E809400 | (21<<16) | (18<<5) | 9)   // SDOT V9.4S, V18.16B, V21.16B
	WORD $(0x4E809400 | (21<<16) | (19<<5) | 13)  // SDOT V13.4S, V19.16B, V21.16B

	// Column 2 (B2: V22) with A0..A3
	WORD $(0x4E809400 | (22<<16) | (16<<5) | 2)   // SDOT V2.4S, V16.16B, V22.16B
	WORD $(0x4E809400 | (22<<16) | (17<<5) | 6)   // SDOT V6.4S, V17.16B, V22.16B
	WORD $(0x4E809400 | (22<<16) | (18<<5) | 10)  // SDOT V10.4S, V18.16B, V22.16B
	WORD $(0x4E809400 | (22<<16) | (19<<5) | 14)  // SDOT V14.4S, V19.16B, V22.16B

	// Column 3 (B3: V23) with A0..A3
	WORD $(0x4E809400 | (23<<16) | (16<<5) | 3)   // SDOT V3.4S, V16.16B, V23.16B
	WORD $(0x4E809400 | (23<<16) | (17<<5) | 7)   // SDOT V7.4S, V17.16B, V23.16B
	WORD $(0x4E809400 | (23<<16) | (18<<5) | 11)  // SDOT V11.4S, V18.16B, V23.16B
	WORD $(0x4E809400 | (23<<16) | (19<<5) | 15)  // SDOT V15.4S, V19.16B, V23.16B

	SUB $1, R10
	CBNZ R10, loop_int8

reduce_int8:
	// Horizontal vector reduction VADDV for each cell, write row 0 to C
	VADDV V0.S4, V24
	VADDV V1.S4, V25
	VADDV V2.S4, V26
	VADDV V3.S4, V27
	VMOV V24.S[0], R11
	VMOV V25.S[0], R12
	VMOV V26.S[0], R13
	VMOV V27.S[0], R14
	MOVW R11, (R8)
	MOVW R12, 4(R8)
	MOVW R13, 8(R8)
	MOVW R14, 12(R8)

	// Write row 1 to C
	ADD R9, R8, R15
	VADDV V4.S4, V24
	VADDV V5.S4, V25
	VADDV V6.S4, V26
	VADDV V7.S4, V27
	VMOV V24.S[0], R11
	VMOV V25.S[0], R12
	VMOV V26.S[0], R13
	VMOV V27.S[0], R14
	MOVW R11, (R15)
	MOVW R12, 4(R15)
	MOVW R13, 8(R15)
	MOVW R14, 12(R15)

	// Write row 2 to C
	ADD R9, R15, R15
	VADDV V8.S4, V24
	VADDV V9.S4, V25
	VADDV V10.S4, V26
	VADDV V11.S4, V27
	VMOV V24.S[0], R11
	VMOV V25.S[0], R12
	VMOV V26.S[0], R13
	VMOV V27.S[0], R14
	MOVW R11, (R15)
	MOVW R12, 4(R15)
	MOVW R13, 8(R15)
	MOVW R14, 12(R15)

	// Write row 3 to C
	ADD R9, R15, R15
	VADDV V12.S4, V24
	VADDV V13.S4, V25
	VADDV V14.S4, V26
	VADDV V15.S4, V27
	VMOV V24.S[0], R11
	VMOV V25.S[0], R12
	VMOV V26.S[0], R13
	VMOV V27.S[0], R14
	MOVW R11, (R15)
	MOVW R12, 4(R15)
	MOVW R13, 8(R15)
	MOVW R14, 12(R15)

	RET

// GemmTile4x4FP16NEON computes a 4x4 fp16 matrix tile multiplication with float32 output:
//   C[i, j] = sum_{k=0..K-1} A[i, k] * B[j, k]
// where A has 4 rows (pointers a0..a3), B has 4 rows (pointers b0..b3), and C is written
// as float32 at stride ldc (in float32 elements).
//
// func GemmTile4x4FP16NEON(a0, a1, a2, a3, b0, b1, b2, b3 *uint16, c *float32, ldc, k int)
TEXT ·GemmTile4x4FP16NEON(SB), NOSPLIT, $0-88
	MOVD a0+0(FP), R0
	MOVD a1+8(FP), R1
	MOVD a2+16(FP), R2
	MOVD a3+24(FP), R3
	MOVD b0+32(FP), R4
	MOVD b1+40(FP), R5
	MOVD b2+48(FP), R6
	MOVD b3+56(FP), R7
	MOVD c+64(FP), R8
	MOVD ldc+72(FP), R9
	MOVD k+80(FP), R10

	LSR $3, R10, R10    // k_blocks = k / 8
	LSL $2, R9, R9      // ldc in bytes (ldc * 4)

	// Zero the 16 fp16 vector accumulators (V0..V15)
	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16

	CBZ R10, reduce_fp16

loop_fp16:
	// Interleaved vector loads from A and B (16 bytes = 8 fp16 halfwords per row)
	VLD1.P 16(R0), [V16.B16]
	VLD1.P 16(R1), [V17.B16]
	VLD1.P 16(R4), [V20.B16]
	VLD1.P 16(R5), [V21.B16]
	VLD1.P 16(R2), [V18.B16]
	VLD1.P 16(R3), [V19.B16]
	VLD1.P 16(R6), [V22.B16]
	VLD1.P 16(R7), [V23.B16]

	// Accumulate 4x4 outer-product tile using ARM64 NEON FMLA (.8H)
	// Column 0 (B0: V20) with A0..A3
	WORD $(0x4E400C00 | (20<<16) | (16<<5) | 0)   // FMLA V0.8H, V16.8H, V20.8H
	WORD $(0x4E400C00 | (20<<16) | (17<<5) | 4)   // FMLA V4.8H, V17.8H, V20.8H
	WORD $(0x4E400C00 | (20<<16) | (18<<5) | 8)   // FMLA V8.8H, V18.8H, V20.8H
	WORD $(0x4E400C00 | (20<<16) | (19<<5) | 12)  // FMLA V12.8H, V19.8H, V20.8H

	// Column 1 (B1: V21) with A0..A3
	WORD $(0x4E400C00 | (21<<16) | (16<<5) | 1)   // FMLA V1.8H, V16.8H, V21.8H
	WORD $(0x4E400C00 | (21<<16) | (17<<5) | 5)   // FMLA V5.8H, V17.8H, V21.8H
	WORD $(0x4E400C00 | (21<<16) | (18<<5) | 9)   // FMLA V9.8H, V18.8H, V21.8H
	WORD $(0x4E400C00 | (21<<16) | (19<<5) | 13)  // FMLA V13.8H, V19.8H, V21.8H

	// Column 2 (B2: V22) with A0..A3
	WORD $(0x4E400C00 | (22<<16) | (16<<5) | 2)   // FMLA V2.8H, V16.8H, V22.8H
	WORD $(0x4E400C00 | (22<<16) | (17<<5) | 6)   // FMLA V6.8H, V17.8H, V22.8H
	WORD $(0x4E400C00 | (22<<16) | (18<<5) | 10)  // FMLA V10.8H, V18.8H, V22.8H
	WORD $(0x4E400C00 | (22<<16) | (19<<5) | 14)  // FMLA V14.8H, V19.8H, V22.8H

	// Column 3 (B3: V23) with A0..A3
	WORD $(0x4E400C00 | (23<<16) | (16<<5) | 3)   // FMLA V3.8H, V16.8H, V23.8H
	WORD $(0x4E400C00 | (23<<16) | (17<<5) | 7)   // FMLA V7.8H, V17.8H, V23.8H
	WORD $(0x4E400C00 | (23<<16) | (18<<5) | 11)  // FMLA V11.8H, V18.8H, V23.8H
	WORD $(0x4E400C00 | (23<<16) | (19<<5) | 15)  // FMLA V15.8H, V19.8H, V23.8H

	SUB $1, R10
	CBNZ R10, loop_fp16

reduce_fp16:
	// Reduce each of V0..V15 via 3 pairwise FADDP additions, convert H0 to float32 (FCVT), write row 0
	WORD $(0x6E401400 | (0<<16) | (0<<5) | 0)
	WORD $(0x6E401400 | (0<<16) | (0<<5) | 0)
	WORD $(0x6E401400 | (0<<16) | (0<<5) | 0)
	WORD $(0x1EE24000 | (0<<5) | 0)

	WORD $(0x6E401400 | (1<<16) | (1<<5) | 1)
	WORD $(0x6E401400 | (1<<16) | (1<<5) | 1)
	WORD $(0x6E401400 | (1<<16) | (1<<5) | 1)
	WORD $(0x1EE24000 | (1<<5) | 1)

	WORD $(0x6E401400 | (2<<16) | (2<<5) | 2)
	WORD $(0x6E401400 | (2<<16) | (2<<5) | 2)
	WORD $(0x6E401400 | (2<<16) | (2<<5) | 2)
	WORD $(0x1EE24000 | (2<<5) | 2)

	WORD $(0x6E401400 | (3<<16) | (3<<5) | 3)
	WORD $(0x6E401400 | (3<<16) | (3<<5) | 3)
	WORD $(0x6E401400 | (3<<16) | (3<<5) | 3)
	WORD $(0x1EE24000 | (3<<5) | 3)

	FMOVS F0, (R8)
	FMOVS F1, 4(R8)
	FMOVS F2, 8(R8)
	FMOVS F3, 12(R8)

	// Write row 1
	ADD R9, R8, R15
	WORD $(0x6E401400 | (4<<16) | (4<<5) | 4)
	WORD $(0x6E401400 | (4<<16) | (4<<5) | 4)
	WORD $(0x6E401400 | (4<<16) | (4<<5) | 4)
	WORD $(0x1EE24000 | (4<<5) | 4)

	WORD $(0x6E401400 | (5<<16) | (5<<5) | 5)
	WORD $(0x6E401400 | (5<<16) | (5<<5) | 5)
	WORD $(0x6E401400 | (5<<16) | (5<<5) | 5)
	WORD $(0x1EE24000 | (5<<5) | 5)

	WORD $(0x6E401400 | (6<<16) | (6<<5) | 6)
	WORD $(0x6E401400 | (6<<16) | (6<<5) | 6)
	WORD $(0x6E401400 | (6<<16) | (6<<5) | 6)
	WORD $(0x1EE24000 | (6<<5) | 6)

	WORD $(0x6E401400 | (7<<16) | (7<<5) | 7)
	WORD $(0x6E401400 | (7<<16) | (7<<5) | 7)
	WORD $(0x6E401400 | (7<<16) | (7<<5) | 7)
	WORD $(0x1EE24000 | (7<<5) | 7)

	FMOVS F4, (R15)
	FMOVS F5, 4(R15)
	FMOVS F6, 8(R15)
	FMOVS F7, 12(R15)

	// Write row 2
	ADD R9, R15, R15
	WORD $(0x6E401400 | (8<<16) | (8<<5) | 8)
	WORD $(0x6E401400 | (8<<16) | (8<<5) | 8)
	WORD $(0x6E401400 | (8<<16) | (8<<5) | 8)
	WORD $(0x1EE24000 | (8<<5) | 8)

	WORD $(0x6E401400 | (9<<16) | (9<<5) | 9)
	WORD $(0x6E401400 | (9<<16) | (9<<5) | 9)
	WORD $(0x6E401400 | (9<<16) | (9<<5) | 9)
	WORD $(0x1EE24000 | (9<<5) | 9)

	WORD $(0x6E401400 | (10<<16) | (10<<5) | 10)
	WORD $(0x6E401400 | (10<<16) | (10<<5) | 10)
	WORD $(0x6E401400 | (10<<16) | (10<<5) | 10)
	WORD $(0x1EE24000 | (10<<5) | 10)

	WORD $(0x6E401400 | (11<<16) | (11<<5) | 11)
	WORD $(0x6E401400 | (11<<16) | (11<<5) | 11)
	WORD $(0x6E401400 | (11<<16) | (11<<5) | 11)
	WORD $(0x1EE24000 | (11<<5) | 11)

	FMOVS F8, (R15)
	FMOVS F9, 4(R15)
	FMOVS F10, 8(R15)
	FMOVS F11, 12(R15)

	// Write row 3
	ADD R9, R15, R15
	WORD $(0x6E401400 | (12<<16) | (12<<5) | 12)
	WORD $(0x6E401400 | (12<<16) | (12<<5) | 12)
	WORD $(0x6E401400 | (12<<16) | (12<<5) | 12)
	WORD $(0x1EE24000 | (12<<5) | 12)

	WORD $(0x6E401400 | (13<<16) | (13<<5) | 13)
	WORD $(0x6E401400 | (13<<16) | (13<<5) | 13)
	WORD $(0x6E401400 | (13<<16) | (13<<5) | 13)
	WORD $(0x1EE24000 | (13<<5) | 13)

	WORD $(0x6E401400 | (14<<16) | (14<<5) | 14)
	WORD $(0x6E401400 | (14<<16) | (14<<5) | 14)
	WORD $(0x6E401400 | (14<<16) | (14<<5) | 14)
	WORD $(0x1EE24000 | (14<<5) | 14)

	WORD $(0x6E401400 | (15<<16) | (15<<5) | 15)
	WORD $(0x6E401400 | (15<<16) | (15<<5) | 15)
	WORD $(0x6E401400 | (15<<16) | (15<<5) | 15)
	WORD $(0x1EE24000 | (15<<5) | 15)

	FMOVS F12, (R15)
	FMOVS F13, 4(R15)
	FMOVS F14, 8(R15)
	FMOVS F15, 12(R15)

	RET
