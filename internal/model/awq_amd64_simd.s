//go:build amd64

#include "textflag.h"

// awq_amd64_simd.s — AVX2 + AVX-512 Go-assembly for the AWQ 4-bit CPU path (#1124 C4 / #1128).
//
// AWQ packs two 4-bit codes per byte: code_lo = b & 0x0f (first weight), code_hi = b >> 4
// (second weight). Dequant of one weight is scale * (int16(code) - 8); the dot is
// Σ scale*(code-8)*x over the row, x in natural order paired with the unpacked codes
// (lo0,hi0,lo1,hi1,...). The hot inner unpack is the canonical SIMD shape:
//
//   raw bytes ──┬─ & 0x0f ─────────────► lo nibbles (one per byte)
//               └─ >>4 (VPSRLW), & 0x0f ► hi nibbles (one per byte)
//   VPUNPCKLBW(hi, lo) ► interleaved codes [lo0,hi0,lo1,hi1,...] in ONE 128-bit lane
//   VPMOVZXBD ► int32 lanes ─ VPSUBD 8 ─ VCVTDQ2PS ─ VMULPS scale ─► dequant floats
//   (dot) VFMADD231PS against x ─► lane accumulators ─ horizontal reduce ─► scalar
//
// Doing the byte interleave inside a single XMM (VPUNPCKLBW) before the VPMOVZXBD widen
// sidesteps the AVX2/AVX-512 cross-128-lane unpack hazard entirely: the widen reads the
// 8 (AVX2) / 16 (AVX-512) already-ordered bytes straight into dword lanes.
//
// Block granularity: AVX2 = 4 src bytes → 8 weights / iter; AVX-512 = 8 src bytes → 16
// weights / iter. The Go wrappers (awq_amd64_asm.go) pass a whole-block byte count and
// fold the sub-block tail through the scalar reference, so any row length is exact.
//
// DEQUANT is bit-identical to awqDequantRowScalar (per-element, no reduction). DOT is
// cosine-parity (per-element scale, lane-reduced sum) — the #1124 acceptance bar.

// awqc<> — 0x0F0F0F0F broadcast base for the low-nibble mask.
DATA awqc<>+0x00(SB)/4, $0x0F0F0F0F
GLOBL awqc<>(SB), RODATA|NOPTR, $4

// func awqDequantRowAsmAVX2(dst *float32, scale float32, src *byte, nbytes int)
// nbytes is a multiple of 4; writes nbytes*2 float32 into dst.
TEXT ·awqDequantRowAsmAVX2(SB), NOSPLIT, $0-32
	MOVQ  dst+0(FP), DI
	MOVSS scale+8(FP), X3
	MOVQ  src+16(FP), SI
	MOVQ  nbytes+24(FP), CX

	VPBROADCASTD awqc<>+0x00(SB), Y6 // 0x0f mask (per byte)
	VPBROADCASTD X3, Y3              // scale broadcast (float32 bits)
	MOVL         $8, AX
	MOVQ         AX, X7
	VPBROADCASTD X7, Y7 // int32 8 (zero-point)

	TESTQ CX, CX
	JLE   dqa2_done

dqa2_loop:
	MOVL        (SI), AX
	MOVQ        AX, X0     // 4 packed bytes
	VPAND       X6, X0, X1 // lo nibbles
	VPSRLW      $4, X0, X2
	VPAND       X6, X2, X2 // hi nibbles
	VPUNPCKLBW  X2, X1, X0 // [lo0,hi0,lo1,hi1,lo2,hi2,lo3,hi3]
	VPMOVZXBD   X0, Y0     // 8 int32 codes
	VPSUBD      Y7, Y0, Y0 // - 8
	VCVTDQ2PS   Y0, Y0     // -> float32
	VMULPS      Y3, Y0, Y0 // * scale
	VMOVUPS     Y0, (DI)
	ADDQ        $4, SI
	ADDQ        $32, DI // 8 floats
	SUBQ        $4, CX
	JNZ         dqa2_loop

dqa2_done:
	VZEROUPPER
	RET

// func awqDequantRowAsmAVX512(dst *float32, scale float32, src *byte, nbytes int)
// nbytes is a multiple of 8; writes nbytes*2 float32 into dst.
TEXT ·awqDequantRowAsmAVX512(SB), NOSPLIT, $0-32
	MOVQ  dst+0(FP), DI
	MOVSS scale+8(FP), X3
	MOVQ  src+16(FP), SI
	MOVQ  nbytes+24(FP), CX

	VPBROADCASTD awqc<>+0x00(SB), Y6 // 0x0f mask (used on xmm)
	VPBROADCASTD X3, Z3              // scale broadcast
	MOVL         $8, AX
	MOVQ         AX, X7
	VPBROADCASTD X7, Z7 // int32 8

	TESTQ CX, CX
	JLE   dqa512_done

dqa512_loop:
	MOVQ        (SI), AX
	MOVQ        AX, X0     // 8 packed bytes
	VPAND       X6, X0, X1 // lo nibbles
	VPSRLW      $4, X0, X2
	VPAND       X6, X2, X2 // hi nibbles
	VPUNPCKLBW  X2, X1, X0 // [lo0,hi0,...,lo7,hi7] (16 bytes)
	VPMOVZXBD   X0, Z0     // 16 int32 codes
	VPSUBD      Z7, Z0, Z0
	VCVTDQ2PS   Z0, Z0
	VMULPS      Z3, Z0, Z0
	VMOVUPS     Z0, (DI)
	ADDQ        $8, SI
	ADDQ        $64, DI // 16 floats
	SUBQ        $8, CX
	JNZ         dqa512_loop

dqa512_done:
	VZEROUPPER
	RET

// func awqDotProductAsmAVX2(src *byte, scale float32, x *float32, nbytes int) float32
// nbytes is a multiple of 4; consumes nbytes*2 float32 from x.
TEXT ·awqDotProductAsmAVX2(SB), NOSPLIT, $0-36
	MOVQ  src+0(FP), SI
	MOVSS scale+8(FP), X3
	MOVQ  x+16(FP), DX
	MOVQ  nbytes+24(FP), CX

	VPBROADCASTD awqc<>+0x00(SB), Y6
	VPBROADCASTD X3, Y3
	MOVL         $8, AX
	MOVQ         AX, X7
	VPBROADCASTD X7, Y7
	VXORPS       Y0, Y0, Y0 // acc

	TESTQ CX, CX
	JLE   dpa2_reduce

dpa2_loop:
	MOVL        (SI), AX
	MOVQ        AX, X1
	VPAND       X6, X1, X2
	VPSRLW      $4, X1, X4
	VPAND       X6, X4, X4
	VPUNPCKLBW  X4, X2, X1  // [lo0,hi0,lo1,hi1,lo2,hi2,lo3,hi3]
	VPMOVZXBD   X1, Y1      // 8 int32 codes
	VPSUBD      Y7, Y1, Y1
	VCVTDQ2PS   Y1, Y1      // weights
	VMULPS      Y3, Y1, Y1  // * scale (per element)
	VMOVUPS     (DX), Y2    // 8 activations
	VFMADD231PS Y1, Y2, Y0  // acc += w*x
	ADDQ        $4, SI
	ADDQ        $32, DX
	SUBQ        $4, CX
	JNZ         dpa2_loop

dpa2_reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VPSHUFD      $0xEE, X0, X1
	VADDPS       X1, X0, X0
	VPSHUFD      $0x55, X0, X1
	VADDSS       X1, X0, X0
	VMOVSS       X0, ret+32(FP)
	VZEROUPPER
	RET

// func awqDotProductAsmAVX512(src *byte, scale float32, x *float32, nbytes int) float32
// nbytes is a multiple of 8; consumes nbytes*2 float32 from x.
TEXT ·awqDotProductAsmAVX512(SB), NOSPLIT, $0-36
	MOVQ  src+0(FP), SI
	MOVSS scale+8(FP), X3
	MOVQ  x+16(FP), DX
	MOVQ  nbytes+24(FP), CX

	VPBROADCASTD awqc<>+0x00(SB), Y6
	VPBROADCASTD X3, Z3
	MOVL         $8, AX
	MOVQ         AX, X7
	VPBROADCASTD X7, Z7
	VXORPS       Z0, Z0, Z0 // acc (zeroes the full ZMM, incl. low YMM)

	TESTQ CX, CX
	JLE   dpa512_reduce

dpa512_loop:
	MOVQ        (SI), AX
	MOVQ        AX, X1
	VPAND       X6, X1, X2
	VPSRLW      $4, X1, X4
	VPAND       X6, X4, X4
	VPUNPCKLBW  X4, X2, X1  // [lo0,hi0,...,lo7,hi7] (16 bytes)
	VPMOVZXBD   X1, Z1      // 16 int32 codes
	VPSUBD      Z7, Z1, Z1
	VCVTDQ2PS   Z1, Z1
	VMULPS      Z3, Z1, Z1  // * scale (per element)
	VMOVUPS     (DX), Z2    // 16 activations
	VFMADD231PS Z1, Z2, Z0  // acc += w*x
	ADDQ        $8, SI
	ADDQ        $64, DX
	SUBQ        $8, CX
	JNZ         dpa512_loop

dpa512_reduce:
	VEXTRACTF64X4 $1, Z0, Y1
	VADDPS        Y1, Y0, Y0
	VEXTRACTF128  $1, Y0, X1
	VADDPS        X1, X0, X0
	VPSHUFD       $0xEE, X0, X1
	VADDPS        X1, X0, X0
	VPSHUFD       $0x55, X0, X1
	VADDSS        X1, X0, X0
	VMOVSS        X0, ret+32(FP)
	VZEROUPPER
	RET
