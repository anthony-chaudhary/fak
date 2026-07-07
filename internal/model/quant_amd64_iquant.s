//go:build amd64

#include "textflag.h"

// func iq3xxsGroupAVX2(dst *float32, packed uint64, signMask *uint32, scale float32)
//
// One IQ3_XXS group = 8 output lanes. `packed` is (g1 | g2<<32), the two 4-byte grid words;
// its low 8 bytes zero-extend to the 8 lanes [g1b0..g1b3, g2b0..g2b3]. Each lane is scaled by
// `scale` (db) and sign-flipped by XORing the 0x80000000 mask word from signMask[lane]. This
// is bit-identical to the scalar iq3xxsDequantSuperBlock inner loop (quant_iquant.go): the
// scalar multiplies (db*byte) by ±1, and flipping an IEEE-754 sign bit == multiplying a finite
// float by -1. Verbatim port of internal/ggufload's ggufDequantIQ3XXSGroupAVX2 (load path),
// whose bit-identity is pinned by TestDequantLoadPathMatchesScalarWiredSIMD/IQ3_XXS. Pure AVX2
// (VPMOVZXBD/VCVTDQ2PS/VMULPS/VXORPS) — no VNNI, so it runs on an AVX2-only host.
TEXT ·iq3xxsGroupAVX2(SB), NOSPLIT, $0-28
	MOVQ      dst+0(FP), DI
	MOVQ      packed+8(FP), AX
	MOVQ      signMask+16(FP), SI
	MOVSS     scale+24(FP), X3
	MOVQ      AX, X1
	VPMOVZXBD X1, Y0
	VCVTDQ2PS Y0, Y0
	VPBROADCASTD X3, Y3
	VMULPS    Y3, Y0, Y0
	VMOVUPS   (SI), Y2
	VXORPS    Y2, Y0, Y0
	VMOVUPS   Y0, 0(DI)
	VZEROUPPER
	RET
