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
