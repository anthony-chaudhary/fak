//go:build amd64

// Adapted verbatim (symbol renames and unused kernels removed) from in-tree
// internal/ggufload/gguf_dequant_simd_amd64.s at
// 2fb9f3bc303bcfdeaebcaebb4e9cd61f1d187484; same repository Apache-2.0 license.
// Keep MULPS then SUBPS, never FMA: model scalar decoder order is authoritative.
#include "textflag.h"

DATA kqdqc<>+0x00(SB)/4, $0x0F0F0F0F // low-nibble mask
DATA kqdqc<>+0x04(SB)/4, $0x10101010 // byte value 16
DATA kqdqc<>+0x08(SB)/4, $0x00000020 // int32 value 32
GLOBL kqdqc<>(SB), RODATA|NOPTR, $12

// kqstoreu8x32 dequantizes 32 unsigned byte codes already unpacked in Y1:
// dst[i] = scale*float32(code[i]) - min. DI=dst, Y3=scale, Y4=min.
TEXT kqstoreu8x32<>(SB), NOSPLIT, $0-0
	VPMOVZXBD X1, Y0
	VCVTDQ2PS Y0, Y0
	VMULPS    Y3, Y0, Y0
	VSUBPS    Y4, Y0, Y0
	VMOVUPS   Y0, 0(DI)

	VPSRLDQ   $8, X1, X0
	VPMOVZXBD X0, Y0
	VCVTDQ2PS Y0, Y0
	VMULPS    Y3, Y0, Y0
	VSUBPS    Y4, Y0, Y0
	VMOVUPS   Y0, 32(DI)

	VEXTRACTI128 $1, Y1, X2
	VPMOVZXBD    X2, Y0
	VCVTDQ2PS    Y0, Y0
	VMULPS       Y3, Y0, Y0
	VSUBPS       Y4, Y0, Y0
	VMOVUPS      Y0, 64(DI)

	VPSRLDQ   $8, X2, X0
	VPMOVZXBD X0, Y0
	VCVTDQ2PS Y0, Y0
	VMULPS    Y3, Y0, Y0
	VSUBPS    Y4, Y0, Y0
	VMOVUPS   Y0, 96(DI)
	RET

// kqstoreq6x16 dequantizes 16 unsigned Q6 codes already unpacked in X1:
// dst[i] = scale*float32(code[i]-32). DI=dst, Y3=scale, Y5=int32(32).
TEXT kqstoreq6x16<>(SB), NOSPLIT, $0-0
	VPMOVZXBD X1, Y0
	VPSUBD    Y5, Y0, Y0
	VCVTDQ2PS Y0, Y0
	VMULPS    Y3, Y0, Y0
	VMOVUPS   Y0, 0(DI)

	VPSRLDQ   $8, X1, X0
	VPMOVZXBD X0, Y0
	VPSUBD    Y5, Y0, Y0
	VCVTDQ2PS Y0, Y0
	VMULPS    Y3, Y0, Y0
	VMOVUPS   Y0, 32(DI)
	RET

// kq5unpackhigh turns qh&mask into byte value 0 or 16 and adds it to Y1.
// Y1=low 4-bit codes, Y14=qh bytes, Y7=byte 16, Y8=high-bit mask.
TEXT kq5unpackhigh<>(SB), NOSPLIT, $0-0
	VPAND    Y8, Y14, Y9
	VPXOR    Y10, Y10, Y10
	VPCMPEQB Y10, Y9, Y9
	VPANDN   Y7, Y9, Y9
	VPADDB   Y9, Y1, Y1
	RET

// func q5kDequantLoAVX2(dst *float32, ql, qh *byte, scale, min float32, highMask uint32)
TEXT ·q5kDequantLoAVX2(SB), NOSPLIT, $0-36
	MOVQ  dst+0(FP), DI
	MOVQ  ql+8(FP), SI
	MOVQ  qh+16(FP), DX
	MOVSS scale+24(FP), X3
	MOVSS min+28(FP), X4
	MOVL  highMask+32(FP), AX
	MOVQ  AX, X8
	VPBROADCASTD X3, Y3
	VPBROADCASTD X4, Y4
	VPBROADCASTD X8, Y8
	VPBROADCASTD kqdqc<>+0x00(SB), Y6
	VPBROADCASTD kqdqc<>+0x04(SB), Y7
	VMOVDQU (SI), Y0
	VMOVDQU (DX), Y14
	VPAND   Y6, Y0, Y1
	CALL    kq5unpackhigh<>(SB)
	CALL    kqstoreu8x32<>(SB)
	VZEROUPPER
	RET

// func q5kDequantHiAVX2(dst *float32, ql, qh *byte, scale, min float32, highMask uint32)
TEXT ·q5kDequantHiAVX2(SB), NOSPLIT, $0-36
	MOVQ  dst+0(FP), DI
	MOVQ  ql+8(FP), SI
	MOVQ  qh+16(FP), DX
	MOVSS scale+24(FP), X3
	MOVSS min+28(FP), X4
	MOVL  highMask+32(FP), AX
	MOVQ  AX, X8
	VPBROADCASTD X3, Y3
	VPBROADCASTD X4, Y4
	VPBROADCASTD X8, Y8
	VPBROADCASTD kqdqc<>+0x00(SB), Y6
	VPBROADCASTD kqdqc<>+0x04(SB), Y7
	VMOVDQU (SI), Y0
	VMOVDQU (DX), Y14
	VPSRLW  $4, Y0, Y1
	VPAND   Y6, Y1, Y1
	CALL    kq5unpackhigh<>(SB)
	CALL    kqstoreu8x32<>(SB)
	VZEROUPPER
	RET

// func q6kDequantPos0AVX2(dst *float32, ql, qh *byte, scale float32)
TEXT ·q6kDequantPos0AVX2(SB), NOSPLIT, $0-28
	MOVQ  dst+0(FP), DI
	MOVQ  ql+8(FP), SI
	MOVQ  qh+16(FP), DX
	MOVSS scale+24(FP), X3
	VPBROADCASTD X3, Y3
	VPBROADCASTD kqdqc<>+0x00(SB), X6
	VPBROADCASTD kqdqc<>+0x08(SB), Y5
	VMOVDQU (SI), X0
	VMOVDQU (DX), X14
	VPAND   X6, X0, X1
	MOVL    $0x03030303, AX
	MOVQ    AX, X8
	VPBROADCASTD X8, X8
	VPAND   X8, X14, X8
	VPSLLW  $4, X8, X8
	VPADDB  X8, X1, X1
	CALL    kqstoreq6x16<>(SB)
	VZEROUPPER
	RET

// func q6kDequantPos1AVX2(dst *float32, ql, qh *byte, scale float32)
TEXT ·q6kDequantPos1AVX2(SB), NOSPLIT, $0-28
	MOVQ  dst+0(FP), DI
	MOVQ  ql+8(FP), SI
	MOVQ  qh+16(FP), DX
	MOVSS scale+24(FP), X3
	VPBROADCASTD X3, Y3
	VPBROADCASTD kqdqc<>+0x00(SB), X6
	VPBROADCASTD kqdqc<>+0x08(SB), Y5
	VMOVDQU (SI), X0
	VMOVDQU (DX), X14
	VPAND   X6, X0, X1
	MOVL    $0x0C0C0C0C, AX
	MOVQ    AX, X8
	VPBROADCASTD X8, X8
	VPAND   X8, X14, X8
	VPSLLW  $2, X8, X8
	VPADDB  X8, X1, X1
	CALL    kqstoreq6x16<>(SB)
	VZEROUPPER
	RET

// func q6kDequantPos2AVX2(dst *float32, ql, qh *byte, scale float32)
TEXT ·q6kDequantPos2AVX2(SB), NOSPLIT, $0-28
	MOVQ  dst+0(FP), DI
	MOVQ  ql+8(FP), SI
	MOVQ  qh+16(FP), DX
	MOVSS scale+24(FP), X3
	VPBROADCASTD X3, Y3
	VPBROADCASTD kqdqc<>+0x00(SB), X6
	VPBROADCASTD kqdqc<>+0x08(SB), Y5
	VMOVDQU (SI), X0
	VMOVDQU (DX), X14
	VPSRLW  $4, X0, X1
	VPAND   X6, X1, X1
	MOVL    $0x30303030, AX
	MOVQ    AX, X8
	VPBROADCASTD X8, X8
	VPAND   X8, X14, X8
	VPADDB  X8, X1, X1
	CALL    kqstoreq6x16<>(SB)
	VZEROUPPER
	RET

// func q6kDequantPos3AVX2(dst *float32, ql, qh *byte, scale float32)
TEXT ·q6kDequantPos3AVX2(SB), NOSPLIT, $0-28
	MOVQ  dst+0(FP), DI
	MOVQ  ql+8(FP), SI
	MOVQ  qh+16(FP), DX
	MOVSS scale+24(FP), X3
	VPBROADCASTD X3, Y3
	VPBROADCASTD kqdqc<>+0x00(SB), X6
	VPBROADCASTD kqdqc<>+0x08(SB), Y5
	VMOVDQU (SI), X0
	VMOVDQU (DX), X14
	VPSRLW  $4, X0, X1
	VPAND   X6, X1, X1
	MOVL    $0xC0C0C0C0, AX
	MOVQ    AX, X8
	VPBROADCASTD X8, X8
	VPAND   X8, X14, X8
	VPSRLW  $2, X8, X8
	VPADDB  X8, X1, X1
	CALL    kqstoreq6x16<>(SB)
	VZEROUPPER
	RET

// func q4kDequantLoAVX2(dst *float32, q *byte, scale, min float32)
TEXT ·q4kDequantLoAVX2(SB), NOSPLIT, $0-24
	MOVQ  dst+0(FP), DI
	MOVQ  q+8(FP), SI
	MOVSS scale+16(FP), X3
	MOVSS min+20(FP), X4
	VPBROADCASTD X3, Y3
	VPBROADCASTD X4, Y4
	VPBROADCASTD kqdqc<>+0x00(SB), Y6
	VMOVDQU (SI), Y0
	VPAND   Y6, Y0, Y1
	CALL    kqstoreu8x32<>(SB)
	VZEROUPPER
	RET

// func q4kDequantHiAVX2(dst *float32, q *byte, scale, min float32)
TEXT ·q4kDequantHiAVX2(SB), NOSPLIT, $0-24
	MOVQ  dst+0(FP), DI
	MOVQ  q+8(FP), SI
	MOVSS scale+16(FP), X3
	MOVSS min+20(FP), X4
	VPBROADCASTD X3, Y3
	VPBROADCASTD X4, Y4
	VPBROADCASTD kqdqc<>+0x00(SB), Y6
	VMOVDQU (SI), Y0
	VPSRLW  $4, Y0, Y1
	VPAND   Y6, Y1, Y1
	CALL    kqstoreu8x32<>(SB)
	VZEROUPPER
	RET

