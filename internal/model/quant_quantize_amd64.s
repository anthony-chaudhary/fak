//go:build amd64

#include "textflag.h"

// Read-only constants for the quantize kernel (one f32/i32 each, broadcast into a zmm).
DATA qzc<>+0x00(SB)/4, $0x7fffffff // abs mask (clear the f32 sign bit)
DATA qzc<>+0x04(SB)/4, $0x42fe0000 // 127.0f
DATA qzc<>+0x08(SB)/4, $0x3f800000 // 1.0f
DATA qzc<>+0x0c(SB)/4, $0x3f000000 // 0.5f
DATA qzc<>+0x10(SB)/4, $0xbf000000 // -0.5f
DATA qzc<>+0x14(SB)/4, $0x00000001 // int32 1
DATA qzc<>+0x18(SB)/4, $0x0000007f // int32 127
DATA qzc<>+0x1c(SB)/4, $0xffffff81 // int32 -127
GLOBL qzc<>(SB), RODATA|NOPTR, $32

// quantizeRowAsm512 — Q8_0 quantize one row of nblk 32-float blocks, AVX-512, bit-identical
// to quantizeRowQ8scalar. Per block: amax = max|x| (16-wide abs+max, then a 16->1 reduce);
// d = amax/127 (stored); if d==0 write 32 zero codes; else inv = 1/d and each code is
// q8round(x*inv) computed the EXACT scalar way — VCVTTPS2DQ (truncate toward zero), recover
// the fractional part, bump +1 where frac>=0.5 / -1 where frac<=-0.5 (masked), clamp to
// [-127,127], pack int32->int8 with signed saturation. No FMA, no fast-rounding shortcut, so
// every code equals the scalar q8round bit-for-bit.
//
// func quantizeRowAsm512(x *float32, q *int8, d *float32, nblk int)
TEXT ·quantizeRowAsm512(SB), NOSPLIT, $0-32
	MOVQ x+0(FP), SI
	MOVQ q+8(FP), DI
	MOVQ d+16(FP), R8
	MOVQ nblk+24(FP), CX
	TESTQ CX, CX
	JLE   qzdone

	// load the broadcast constants once
	VPBROADCASTD qzc<>+0x00(SB), Z28 // abs mask
	VBROADCASTSS qzc<>+0x0c(SB), Z27 // 0.5
	VBROADCASTSS qzc<>+0x10(SB), Z26 // -0.5
	VPBROADCASTD qzc<>+0x14(SB), Z25 // int32 1
	VPBROADCASTD qzc<>+0x1c(SB), Z24 // int32 -127
	VPBROADCASTD qzc<>+0x18(SB), Z23 // int32 127
	VMOVSS       qzc<>+0x04(SB), X29 // 127.0 (scalar)
	VMOVSS       qzc<>+0x08(SB), X30 // 1.0   (scalar)
	VXORPS       X22, X22, X22       // zero (for d==0 test + zero-code store)

qzloop:
	VMOVUPS (SI), Z0
	VMOVUPS 64(SI), Z1

	// amax = max over the 32 |x|
	VANDPS Z28, Z0, Z2
	VANDPS Z28, Z1, Z3
	VMAXPS Z3, Z2, Z4
	VEXTRACTF64X4 $1, Z4, Y5
	VMAXPS Y5, Y4, Y4
	VEXTRACTF128  $1, Y4, X5
	VMAXPS X5, X4, X4
	VSHUFPS $0x0e, X4, X4, X5
	VMAXPS  X5, X4, X4
	VSHUFPS $0x01, X4, X4, X5
	VMAXPS  X5, X4, X4 // X4 lane0 = amax

	// d = amax / 127 ; store
	VDIVSS X29, X4, X6
	VMOVSS X6, (R8)

	// if d == 0 -> write 32 zero codes
	VUCOMISS X22, X6
	JE       qzzero

	// inv = 1.0 / d ; broadcast
	VDIVSS       X6, X30, X8
	VBROADCASTSS X8, Z8

	// scaled = x * inv
	VMULPS Z8, Z0, Z9
	VMULPS Z8, Z1, Z10

	// t = trunc(scaled) ; frac = scaled - float(t)
	VCVTTPS2DQ Z9, Z11
	VCVTTPS2DQ Z10, Z12
	VCVTDQ2PS  Z11, Z13
	VCVTDQ2PS  Z12, Z14
	VSUBPS Z13, Z9, Z15
	VSUBPS Z14, Z10, Z16

	// frac>=0.5 -> +1 ; frac<=-0.5 -> -1  (predicate 5 = NLT/>=, 2 = LE)
	VCMPPS $5, Z27, Z15, K1
	VCMPPS $2, Z26, Z15, K2
	VCMPPS $5, Z27, Z16, K3
	VCMPPS $2, Z26, Z16, K4
	VPADDD Z25, Z11, K1, Z11
	VPSUBD Z25, Z11, K2, Z11
	VPADDD Z25, Z12, K3, Z12
	VPSUBD Z25, Z12, K4, Z12

	// clamp to [-127, 127]
	VPMAXSD Z24, Z11, Z11
	VPMINSD Z23, Z11, Z11
	VPMAXSD Z24, Z12, Z12
	VPMINSD Z23, Z12, Z12

	// pack int32 -> int8 (signed saturate; clamp already keeps it in range) and store 32
	VPMOVSDB Z11, X17
	VPMOVSDB Z12, X18
	VMOVUPS X17, (DI)
	VMOVUPS X18, 16(DI)
	JMP qznext

qzzero:
	VMOVUPS X22, (DI)
	VMOVUPS X22, 16(DI)

qznext:
	ADDQ $128, SI // 32 floats
	ADDQ $32, DI  // 32 codes
	ADDQ $4, R8   // 1 scale
	DECQ CX
	JNZ  qzloop

qzdone:
	VZEROUPPER
	RET


// quantizeRowAsm256 adapts the in-tree AVX-512 trunc/fraction/clamp algorithm
// above to four 8-float vectors. Finite abs/max reduction is exact; no sums or
// rounding operations are reassociated and no FMA or reciprocal approximation
// is used. Reject NaN/Inf and reciprocal overflow BEFORE writing that block.
// Prior complete blocks remain valid; Go runs the original scalar suffix.
// Return done is a block count, so all pointers retain their existing layouts.
//
// func quantizeRowAsm256(x *float32, q *int8, d *float32, nblk int) (done int)
TEXT ·quantizeRowAsm256(SB), NOSPLIT, $0-40
 MOVQ x+0(FP), SI
 MOVQ q+8(FP), DI
 MOVQ d+16(FP), R8
 MOVQ nblk+24(FP), CX
 XORQ R9, R9
 TESTQ CX, CX
 JLE qz256done
 VPBROADCASTD qzc<>+0x00(SB), Y15 // abs mask
 VPBROADCASTD qzc<>+0x18(SB), Y14 // int32 127
 VPBROADCASTD qzc<>+0x1c(SB), Y13 // int32 -127
 VBROADCASTSS qzc<>+0x10(SB), Y12 // -0.5
 VBROADCASTSS qzc<>+0x0c(SB), Y11 // 0.5
 VPBROADCASTD qzc<>+0x14(SB), Y10 // int32 1
qz256loop:
 VMOVUPS (SI), Y0
 VMOVUPS 32(SI), Y1
 VMOVUPS 64(SI), Y2
 VMOVUPS 96(SI), Y3
 // Scalar amax ignores NaNs. Keep that exceptional policy in scalar rather
 // than let MAXPS operand ordering choose a different scale or NaN payload.
 VCMPPS $3, Y0, Y0, Y4
 VCMPPS $3, Y1, Y1, Y5
 VORPS Y5, Y4, Y4
 VCMPPS $3, Y2, Y2, Y5
 VORPS Y5, Y4, Y4
 VCMPPS $3, Y3, Y3, Y5
 VORPS Y5, Y4, Y4
 VMOVMSKPS Y4, AX
 TESTL AX, AX
 JNZ qz256done
 VANDPS Y15, Y0, Y4
 VANDPS Y15, Y1, Y5
 VMAXPS Y5, Y4, Y4
 VANDPS Y15, Y2, Y5
 VMAXPS Y5, Y4, Y4
 VANDPS Y15, Y3, Y5
 VMAXPS Y5, Y4, Y4
 VEXTRACTF128 $1, Y4, X5
 VMAXPS X5, X4, X4
 VSHUFPS $0x0e, X4, X4, X5
 VMAXPS X5, X4, X4
 VSHUFPS $0x01, X4, X4, X5
 VMAXPS X5, X4, X4
 VMOVD X4, AX
 CMPL AX, $0x7f800000 // positive Inf (NaNs already refused)
 JE qz256done
 VDIVSS qzc<>+0x04(SB), X4, X6 // d = amax / 127
 VMOVD X6, AX
 TESTL AX, AX
 JZ qz256zero
 VMOVSS qzc<>+0x08(SB), X7
 VDIVSS X6, X7, X7 // inv = 1.0 / d, exactly as scalar
 VMOVD X7, AX
 CMPL AX, $0x7f800000
 JE qz256done // subnormal d can overflow its reciprocal
 VMOVSS X6, (R8)
 VBROADCASTSS X7, Y7

 // Original q8round for elements 0..7, using only ordered finite values.
 VMULPS Y7, Y0, Y8
 VCVTTPS2DQ Y8, Y9
 VCVTDQ2PS Y9, Y4
 VSUBPS Y4, Y8, Y8
 VCMPPS $5, Y11, Y8, Y4 // frac >= 0.5
 VPAND Y10, Y4, Y4
 VPADDD Y4, Y9, Y9
 VCMPPS $2, Y12, Y8, Y4 // frac <= -0.5 (mutually exclusive)
 VPAND Y10, Y4, Y4
 VPSUBD Y4, Y9, Y9
 VPMAXSD Y13, Y9, Y9
 VPMINSD Y14, Y9, Y9
 // Join low/high 128-bit halves before packing: output remains scalar order.
 VEXTRACTI128 $1, Y9, X4
 VPACKSSDW X4, X9, X9
 VPACKSSWB X9, X9, X9
 VMOVQ X9, 0(DI)

 // Original q8round for elements 8..15, using only ordered finite values.
 VMULPS Y7, Y1, Y8
 VCVTTPS2DQ Y8, Y9
 VCVTDQ2PS Y9, Y4
 VSUBPS Y4, Y8, Y8
 VCMPPS $5, Y11, Y8, Y4 // frac >= 0.5
 VPAND Y10, Y4, Y4
 VPADDD Y4, Y9, Y9
 VCMPPS $2, Y12, Y8, Y4 // frac <= -0.5 (mutually exclusive)
 VPAND Y10, Y4, Y4
 VPSUBD Y4, Y9, Y9
 VPMAXSD Y13, Y9, Y9
 VPMINSD Y14, Y9, Y9
 // Join low/high 128-bit halves before packing: output remains scalar order.
 VEXTRACTI128 $1, Y9, X4
 VPACKSSDW X4, X9, X9
 VPACKSSWB X9, X9, X9
 VMOVQ X9, 8(DI)

 // Original q8round for elements 16..23, using only ordered finite values.
 VMULPS Y7, Y2, Y8
 VCVTTPS2DQ Y8, Y9
 VCVTDQ2PS Y9, Y4
 VSUBPS Y4, Y8, Y8
 VCMPPS $5, Y11, Y8, Y4 // frac >= 0.5
 VPAND Y10, Y4, Y4
 VPADDD Y4, Y9, Y9
 VCMPPS $2, Y12, Y8, Y4 // frac <= -0.5 (mutually exclusive)
 VPAND Y10, Y4, Y4
 VPSUBD Y4, Y9, Y9
 VPMAXSD Y13, Y9, Y9
 VPMINSD Y14, Y9, Y9
 // Join low/high 128-bit halves before packing: output remains scalar order.
 VEXTRACTI128 $1, Y9, X4
 VPACKSSDW X4, X9, X9
 VPACKSSWB X9, X9, X9
 VMOVQ X9, 16(DI)

 // Original q8round for elements 24..31, using only ordered finite values.
 VMULPS Y7, Y3, Y8
 VCVTTPS2DQ Y8, Y9
 VCVTDQ2PS Y9, Y4
 VSUBPS Y4, Y8, Y8
 VCMPPS $5, Y11, Y8, Y4 // frac >= 0.5
 VPAND Y10, Y4, Y4
 VPADDD Y4, Y9, Y9
 VCMPPS $2, Y12, Y8, Y4 // frac <= -0.5 (mutually exclusive)
 VPAND Y10, Y4, Y4
 VPSUBD Y4, Y9, Y9
 VPMAXSD Y13, Y9, Y9
 VPMINSD Y14, Y9, Y9
 // Join low/high 128-bit halves before packing: output remains scalar order.
 VEXTRACTI128 $1, Y9, X4
 VPACKSSDW X4, X9, X9
 VPACKSSWB X9, X9, X9
 VMOVQ X9, 24(DI)

 JMP qz256next
qz256zero:
 VMOVSS X6, (R8)
 VXORPS Y4, Y4, Y4
 VMOVUPS Y4, (DI) // all 32 reused codes must be cleared
qz256next:
 ADDQ $128, SI
 ADDQ $32, DI
 ADDQ $4, R8
 INCQ R9
 DECQ CX
 JNZ qz256loop
qz256done:
 MOVQ R9, done+32(FP)
 VZEROUPPER
 RET
