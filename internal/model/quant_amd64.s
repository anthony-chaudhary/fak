//go:build amd64

#include "textflag.h"

// qdot8asm computes the Q8_0 inner product over nblk 32-wide blocks:
//   Σ_b ( float32(Σ_{i∈block b} qw[i]*qx[i]) * dw[b] * dx[b] )
// using AVX2. Signed int8 × signed int8 is handled cleanly by sign-extending each 16-byte
// half to int16 (VPMOVSXBW) and multiply-add-pairing to int32 (VPMADDWD) — no unsigned
// offset/correction terms. Integer addition is associative with no overflow (a block sum
// is bounded by 32*127*127 ≈ 5.2e5), so the lane-reduced int32 equals the scalar kernel's
// isum exactly; the per-block float combine is done in the SAME order as qdot8scalar
// (((float(isum)*dw[b])*dx[b]), accumulated block 0,1,2,…), so this is BIT-IDENTICAL to
// the scalar reference and TestQdot8AsmMatchesScalar pins it.
//
// func qdot8asm(qw, qx *int8, dw, dx *float32, nblk int) float32
TEXT ·qdot8asm(SB), NOSPLIT, $0-44
	MOVQ qw+0(FP), SI
	MOVQ qx+8(FP), DI
	MOVQ dw+16(FP), R8
	MOVQ dx+24(FP), R9
	MOVQ nblk+32(FP), CX
	VXORPS X5, X5, X5      // acc = 0.0 (low lane)
	TESTQ CX, CX
	JLE   done

loop:
	// block = 32 int8 from each input; sign-extend the two 16-byte halves to int16,
	// VPMADDWD each to 8 int32 partials, add the halves -> 8 int32 in Y0.
	VPMOVSXBW (SI), Y0
	VPMOVSXBW (DI), Y1
	VPMADDWD  Y1, Y0, Y0
	VPMOVSXBW 16(SI), Y2
	VPMOVSXBW 16(DI), Y3
	VPMADDWD  Y3, Y2, Y2
	VPADDD    Y2, Y0, Y0

	// horizontal-sum the 8 int32 in Y0 down to a single int32 in X0's low lane.
	VEXTRACTI128 $1, Y0, X1
	VPADDD    X1, X0, X0       // [a,b,c,d] = lo + hi
	VPSHUFD   $0xEE, X0, X1    // [c,d,c,d]
	VPADDD    X1, X0, X0       // lane0=a+c, lane1=b+d
	VPSHUFD   $0x55, X0, X1    // [b+d, …]
	VPADDD    X1, X0, X0       // lane0 = a+b+c+d = isum

	// acc += float32(isum) * dw[b] * dx[b]  (convert int32 lane0 to float, no GPR trip)
	VCVTDQ2PS X0, X0
	VMULSS    (R8), X0, X0
	VMULSS    (R9), X0, X0
	VADDSS    X0, X5, X5

	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $4, R8
	ADDQ $4, R9
	DECQ CX
	JNZ  loop

done:
	VMOVSS X5, ret+40(FP)
	VZEROUPPER
	RET

// qdot8asm512 is the AVX-512BW variant: a 32-wide block is sign-extended to 32 int16 in a
// single zmm (VPMOVSXBW) and reduced to 16 int32 by one VPMADDWD — half the MAC
// instructions of the AVX2 path (which splits each block into two 16-wide halves). Signed
// ×signed, no correction terms, and the int32 result is identical to the scalar/AVX2
// reduction (integer assoc, no overflow), so it is bit-identical too. Reduction of the 16
// int32 lanes uses VEX ymm/xmm ops. Used when AVX-512F+BW and OS zmm state are present.
//
// func qdot8asm512(qw, qx *int8, dw, dx *float32, nblk int) float32
TEXT ·qdot8asm512(SB), NOSPLIT, $0-44
	MOVQ qw+0(FP), SI
	MOVQ qx+8(FP), DI
	MOVQ dw+16(FP), R8
	MOVQ dx+24(FP), R9
	MOVQ nblk+32(FP), CX
	VXORPS X5, X5, X5
	TESTQ CX, CX
	JLE   done512

loop512:
	VPMOVSXBW (SI), Z0        // 32 int8 -> 32 int16
	VPMOVSXBW (DI), Z1
	VPMADDWD  Z1, Z0, Z0      // -> 16 int32 (pairwise products)
	VEXTRACTI64X4 $1, Z0, Y1  // high 8 int32
	VPADDD    Y1, Y0, Y0      // 8 int32
	VEXTRACTI128 $1, Y0, X1
	VPADDD    X1, X0, X0      // 4
	VPSHUFD   $0xEE, X0, X1
	VPADDD    X1, X0, X0
	VPSHUFD   $0x55, X0, X1
	VPADDD    X1, X0, X0      // lane0 = isum
	VCVTDQ2PS X0, X0
	VMULSS    (R8), X0, X0
	VMULSS    (R9), X0, X0
	VADDSS    X0, X5, X5
	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $4, R8
	ADDQ $4, R9
	DECQ CX
	JNZ  loop512

done512:
	VMOVSS X5, ret+40(FP)
	VZEROUPPER
	RET

// qdot8gemv512 is the fast decode-GEMV dot: same Q8_0 operands as qdot8asm512, but with
// qgemm8cell's deferred AVX-512 lane reduction. Each 32-wide block contributes 16 int32
// dot lanes, scaled by dw[b]*dx[b], FMA-accumulated in Z5, and reduced to scalar only once.
// This is not bit-identical to qdot8scalar's per-block scalar combine; it is bit-identical
// to qgemm8cell(..., lanes=16), the reduction order already used by the batched Q8 GEMM.
//
// func qdot8gemv512(qw, qx *int8, dw, dx *float32, nblk int) float32
TEXT ·qdot8gemv512(SB), NOSPLIT, $0-44
	MOVQ qw+0(FP), SI
	MOVQ qx+8(FP), DI
	MOVQ dw+16(FP), R8
	MOVQ dx+24(FP), R9
	MOVQ nblk+32(FP), CX
	VPXORD Z5, Z5, Z5
	TESTQ CX, CX
	JLE   gemvdone

gemvloop:
	VPMOVSXBW (SI), Z0
	VPMOVSXBW (DI), Z1
	VPMADDWD  Z1, Z0, Z0
	VCVTDQ2PS Z0, Z0
	VMOVSS    (R8), X2
	VMULSS    (R9), X2, X2
	VBROADCASTSS X2, Z2
	VFMADD231PS Z2, Z0, Z5
	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $4, R8
	ADDQ $4, R9
	DECQ CX
	JNZ  gemvloop

gemvdone:
	VEXTRACTI64X4 $1, Z5, Y31
	VADDPS    Y31, Y5, Y5
	VEXTRACTI32X4 $1, Z5, X31
	VADDPS    X31, X5, X5
	VPSHUFD   $0xEE, X5, X31
	VADDPS    X31, X5, X5
	VPSHUFD   $0x55, X5, X31
	VADDSS    X31, X5, X5
	VMOVSS X5, ret+40(FP)
	VZEROUPPER
	RET

// qdot8gemv512q16 is qdot8gemv512 with the activation block already sign-extended to
// int16. Decode reuses the same small activation vector across every output row, so
// pre-extending it once avoids repeating VPMOVSXBW(qx) in each row's dot kernel.
//
// func qdot8gemv512q16(qw *int8, qx *int16, dw, dx *float32, nblk int) float32
TEXT ·qdot8gemv512q16(SB), NOSPLIT, $0-44
	MOVQ qw+0(FP), SI
	MOVQ qx+8(FP), DI
	MOVQ dw+16(FP), R8
	MOVQ dx+24(FP), R9
	MOVQ nblk+32(FP), CX
	VPXORD Z5, Z5, Z5
	TESTQ CX, CX
	JLE   gemvq16done

gemvq16loop:
	VPMOVSXBW (SI), Z0
	VMOVDQU64 (DI), Z1
	VPMADDWD  Z1, Z0, Z0
	VCVTDQ2PS Z0, Z0
	VMOVSS    (R8), X2
	VMULSS    (R9), X2, X2
	VBROADCASTSS X2, Z2
	VFMADD231PS Z2, Z0, Z5
	ADDQ $32, SI
	ADDQ $64, DI
	ADDQ $4, R8
	ADDQ $4, R9
	DECQ CX
	JNZ  gemvq16loop

gemvq16done:
	VEXTRACTI64X4 $1, Z5, Y31
	VADDPS    Y31, Y5, Y5
	VEXTRACTI32X4 $1, Z5, X31
	VADDPS    X31, X5, X5
	VPSHUFD   $0xEE, X5, X31
	VADDPS    X31, X5, X5
	VPSHUFD   $0x55, X5, X31
	VADDSS    X31, X5, X5
	VMOVSS X5, ret+40(FP)
	VZEROUPPER
	RET

// q8ToQ16Asm512 sign-extends nblk Q8_0 blocks (32 int8 codes each) into a contiguous int16
// activation buffer. It is used once per decode activation, then qdot8gemv512q16 reuses the
// extended vector for every output row.
//
// func q8ToQ16Asm512(q *int8, q16 *int16, nblk int)
TEXT ·q8ToQ16Asm512(SB), NOSPLIT, $0-24
	MOVQ q+0(FP), SI
	MOVQ q16+8(FP), DI
	MOVQ nblk+16(FP), CX
	TESTQ CX, CX
	JLE   q8q16done

q8q16loop:
	VPMOVSXBW (SI), Z0
	VMOVDQU64 Z0, (DI)
	ADDQ $32, SI
	ADDQ $64, DI
	DECQ CX
	JNZ  q8q16loop

q8q16done:
	VZEROUPPER
	RET

// qgemm8tile512 computes a 5(row)×4(token) Q8_0 output tile with AVX-512BW, the
// register-blocked prefill micro-kernel (see quant_gemm.go). Twenty float accumulators
// (Z0..Z19, indexed row*4+token) stay live across the whole reduction; each block folds in
// with one VPMADDWD + VCVTDQ2PS + VFMADD231PS (scale·int + acc fused, one rounding, NO
// per-block horizontal reduce), and each sign-extended weight block (Z20) feeds all 4 tokens
// while each act block (Z16..Z19) feeds all 4 rows. The 16-lane horizontal reduction runs
// ONCE per output, in the store phase (unfused VADDPS), in the same pairwise tree qgemm8cell
// uses — so the result is Float32bits-identical to qgemm8cell(...,16), whose per-block
// accumulate also uses math.FMA (TestQGemm8AsmMatchesScalar).
//
// Layout: weight row i, block b at qw + i*in + b*32 (scale dw + i*nblk + b); token j, block
// b at qx + j*in + b*32 (scale dx + j*nblk + b); output (row i, token j) at
// dst + j*outStride + i (floats). `in` is the per-row code stride in bytes (== inner dim).
//
// func qgemm8tile512(qw, qx *int8, dw, dx *float32, in, nblk, outStride int, dst *float32)
TEXT ·qgemm8tile512(SB), NOSPLIT, $0-64
	MOVQ qw+0(FP), SI
	MOVQ qx+8(FP), DI
	MOVQ dw+16(FP), R8
	MOVQ dx+24(FP), R9
	MOVQ in+32(FP), R10    // code row/token stride (bytes)
	MOVQ nblk+40(FP), CX
	// weight row offsets: R14 = 2*in, R15 = 3*in, AX = 4*in
	MOVQ R10, R14
	ADDQ R10, R14
	MOVQ R14, R15
	ADDQ R10, R15
	MOVQ R15, AX
	ADDQ R10, AX
	// scale row/token offsets (bytes): R11 = nblk*4, R12 = 2*nblk*4, R13 = 3*nblk*4, BX = 4*nblk*4
	MOVQ CX, R11
	SHLQ $2, R11
	MOVQ R11, R12
	ADDQ R11, R12
	MOVQ R12, R13
	ADDQ R11, R13
	MOVQ R13, BX
	ADDQ R11, BX
	// zero the 16 accumulators
	VPXORD Z0, Z0, Z0
	VPXORD Z1, Z1, Z1
	VPXORD Z2, Z2, Z2
	VPXORD Z3, Z3, Z3
	VPXORD Z4, Z4, Z4
	VPXORD Z5, Z5, Z5
	VPXORD Z6, Z6, Z6
	VPXORD Z7, Z7, Z7
	VPXORD Z8, Z8, Z8
	VPXORD Z9, Z9, Z9
	VPXORD Z10, Z10, Z10
	VPXORD Z11, Z11, Z11
	VPXORD Z12, Z12, Z12
	VPXORD Z13, Z13, Z13
	VPXORD Z14, Z14, Z14
	VPXORD Z15, Z15, Z15
	VPXORD Z24, Z24, Z24
	VPXORD Z25, Z25, Z25
	VPXORD Z26, Z26, Z26
	VPXORD Z27, Z27, Z27
	TESTQ CX, CX
	JLE   gemmreduce

gemmloop:
	// sign-extend the 4 token blocks (token j at DI + j*in) to int16
	VPMOVSXBW (DI), Z16
	VPMOVSXBW (DI)(R10*1), Z17
	VPMOVSXBW (DI)(R14*1), Z18
	VPMOVSXBW (DI)(R15*1), Z19
	VMOVSS    (R9), X23
	VMOVSS    (R9)(R11*1), X29
	VMOVSS    (R9)(R12*1), X30
	VMOVSS    (R9)(R13*1), X31

	// row 0: weight (SI), dw (R8)
	VPMOVSXBW (SI), Z20
	VMOVSS    (R8), X28
	VMULSS    X23, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z16, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z0
	VMULSS    X29, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z17, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z1
	VMULSS    X30, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z18, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z2
	VMULSS    X31, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z19, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z3

	// row 1: weight (SI)(R10*1), dw (R8)(R11*1)
	VPMOVSXBW (SI)(R10*1), Z20
	VMOVSS    (R8)(R11*1), X28
	VMULSS    X23, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z16, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z4
	VMULSS    X29, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z17, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z5
	VMULSS    X30, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z18, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z6
	VMULSS    X31, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z19, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z7

	// row 2: weight (SI)(R14*1), dw (R8)(R12*1)
	VPMOVSXBW (SI)(R14*1), Z20
	VMOVSS    (R8)(R12*1), X28
	VMULSS    X23, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z16, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z8
	VMULSS    X29, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z17, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z9
	VMULSS    X30, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z18, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z10
	VMULSS    X31, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z19, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z11

	// row 3: weight (SI)(R15*1), dw (R8)(R13*1)
	VPMOVSXBW (SI)(R15*1), Z20
	VMOVSS    (R8)(R13*1), X28
	VMULSS    X23, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z16, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z12
	VMULSS    X29, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z17, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z13
	VMULSS    X30, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z18, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z14
	VMULSS    X31, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z19, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z15

	// row 4: weight (SI)(AX*1), dw (R8)(BX*1)
	VPMOVSXBW (SI)(AX*1), Z20
	VMOVSS    (R8)(BX*1), X28
	VMULSS    X23, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z16, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z24
	VMULSS    X29, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z17, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z25
	VMULSS    X30, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z18, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z26
	VMULSS    X31, X28, X22
	VBROADCASTSS X22, Z22
	VPMADDWD  Z19, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z27

	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $4, R8
	ADDQ $4, R9
	DECQ CX
	JNZ  gemmloop

gemmreduce:
	// store phase: reduce each accumulator (16 lanes -> scalar) in qgemm8cell's tree, then
	// store (row i, token j) at dst + j*outStride + i. Token bases: AX,BX,R8,R9.
	MOVQ dst+56(FP), AX
	MOVQ outStride+48(FP), DX
	SHLQ $2, DX            // outStride bytes
	MOVQ AX, BX
	ADDQ DX, BX           // token1 base
	MOVQ BX, R8
	ADDQ DX, R8          // token2 base
	MOVQ R8, R9
	ADDQ DX, R9         // token3 base

	// token 0 (base AX): rows 0..4 = Z0,Z4,Z8,Z12,Z24 at +0,+4,+8,+12,+16
	VEXTRACTI64X4 $1, Z0, Y31
	VADDPS    Y31, Y0, Y0
	VEXTRACTI32X4 $1, Z0, X31
	VADDPS    X31, X0, X0
	VPSHUFD   $0xEE, X0, X31
	VADDPS    X31, X0, X0
	VPSHUFD   $0x55, X0, X31
	VADDSS    X31, X0, X0
	VMOVSS    X0, 0(AX)
	VEXTRACTI64X4 $1, Z4, Y31
	VADDPS    Y31, Y4, Y4
	VEXTRACTI32X4 $1, Z4, X31
	VADDPS    X31, X4, X4
	VPSHUFD   $0xEE, X4, X31
	VADDPS    X31, X4, X4
	VPSHUFD   $0x55, X4, X31
	VADDSS    X31, X4, X4
	VMOVSS    X4, 4(AX)
	VEXTRACTI64X4 $1, Z8, Y31
	VADDPS    Y31, Y8, Y8
	VEXTRACTI32X4 $1, Z8, X31
	VADDPS    X31, X8, X8
	VPSHUFD   $0xEE, X8, X31
	VADDPS    X31, X8, X8
	VPSHUFD   $0x55, X8, X31
	VADDSS    X31, X8, X8
	VMOVSS    X8, 8(AX)
	VEXTRACTI64X4 $1, Z12, Y31
	VADDPS    Y31, Y12, Y12
	VEXTRACTI32X4 $1, Z12, X31
	VADDPS    X31, X12, X12
	VPSHUFD   $0xEE, X12, X31
	VADDPS    X31, X12, X12
	VPSHUFD   $0x55, X12, X31
	VADDSS    X31, X12, X12
	VMOVSS    X12, 12(AX)
	VEXTRACTI64X4 $1, Z24, Y31
	VADDPS    Y31, Y24, Y24
	VEXTRACTI32X4 $1, Z24, X31
	VADDPS    X31, X24, X24
	VPSHUFD   $0xEE, X24, X31
	VADDPS    X31, X24, X24
	VPSHUFD   $0x55, X24, X31
	VADDSS    X31, X24, X24
	VMOVSS    X24, 16(AX)

	// token 1 (base BX): rows 0..4 = Z1,Z5,Z9,Z13,Z25
	VEXTRACTI64X4 $1, Z1, Y31
	VADDPS    Y31, Y1, Y1
	VEXTRACTI32X4 $1, Z1, X31
	VADDPS    X31, X1, X1
	VPSHUFD   $0xEE, X1, X31
	VADDPS    X31, X1, X1
	VPSHUFD   $0x55, X1, X31
	VADDSS    X31, X1, X1
	VMOVSS    X1, 0(BX)
	VEXTRACTI64X4 $1, Z5, Y31
	VADDPS    Y31, Y5, Y5
	VEXTRACTI32X4 $1, Z5, X31
	VADDPS    X31, X5, X5
	VPSHUFD   $0xEE, X5, X31
	VADDPS    X31, X5, X5
	VPSHUFD   $0x55, X5, X31
	VADDSS    X31, X5, X5
	VMOVSS    X5, 4(BX)
	VEXTRACTI64X4 $1, Z9, Y31
	VADDPS    Y31, Y9, Y9
	VEXTRACTI32X4 $1, Z9, X31
	VADDPS    X31, X9, X9
	VPSHUFD   $0xEE, X9, X31
	VADDPS    X31, X9, X9
	VPSHUFD   $0x55, X9, X31
	VADDSS    X31, X9, X9
	VMOVSS    X9, 8(BX)
	VEXTRACTI64X4 $1, Z13, Y31
	VADDPS    Y31, Y13, Y13
	VEXTRACTI32X4 $1, Z13, X31
	VADDPS    X31, X13, X13
	VPSHUFD   $0xEE, X13, X31
	VADDPS    X31, X13, X13
	VPSHUFD   $0x55, X13, X31
	VADDSS    X31, X13, X13
	VMOVSS    X13, 12(BX)
	VEXTRACTI64X4 $1, Z25, Y31
	VADDPS    Y31, Y25, Y25
	VEXTRACTI32X4 $1, Z25, X31
	VADDPS    X31, X25, X25
	VPSHUFD   $0xEE, X25, X31
	VADDPS    X31, X25, X25
	VPSHUFD   $0x55, X25, X31
	VADDSS    X31, X25, X25
	VMOVSS    X25, 16(BX)

	// token 2 (base R8): rows 0..4 = Z2,Z6,Z10,Z14,Z26
	VEXTRACTI64X4 $1, Z2, Y31
	VADDPS    Y31, Y2, Y2
	VEXTRACTI32X4 $1, Z2, X31
	VADDPS    X31, X2, X2
	VPSHUFD   $0xEE, X2, X31
	VADDPS    X31, X2, X2
	VPSHUFD   $0x55, X2, X31
	VADDSS    X31, X2, X2
	VMOVSS    X2, 0(R8)
	VEXTRACTI64X4 $1, Z6, Y31
	VADDPS    Y31, Y6, Y6
	VEXTRACTI32X4 $1, Z6, X31
	VADDPS    X31, X6, X6
	VPSHUFD   $0xEE, X6, X31
	VADDPS    X31, X6, X6
	VPSHUFD   $0x55, X6, X31
	VADDSS    X31, X6, X6
	VMOVSS    X6, 4(R8)
	VEXTRACTI64X4 $1, Z10, Y31
	VADDPS    Y31, Y10, Y10
	VEXTRACTI32X4 $1, Z10, X31
	VADDPS    X31, X10, X10
	VPSHUFD   $0xEE, X10, X31
	VADDPS    X31, X10, X10
	VPSHUFD   $0x55, X10, X31
	VADDSS    X31, X10, X10
	VMOVSS    X10, 8(R8)
	VEXTRACTI64X4 $1, Z14, Y31
	VADDPS    Y31, Y14, Y14
	VEXTRACTI32X4 $1, Z14, X31
	VADDPS    X31, X14, X14
	VPSHUFD   $0xEE, X14, X31
	VADDPS    X31, X14, X14
	VPSHUFD   $0x55, X14, X31
	VADDSS    X31, X14, X14
	VMOVSS    X14, 12(R8)
	VEXTRACTI64X4 $1, Z26, Y31
	VADDPS    Y31, Y26, Y26
	VEXTRACTI32X4 $1, Z26, X31
	VADDPS    X31, X26, X26
	VPSHUFD   $0xEE, X26, X31
	VADDPS    X31, X26, X26
	VPSHUFD   $0x55, X26, X31
	VADDSS    X31, X26, X26
	VMOVSS    X26, 16(R8)

	// token 3 (base R9): rows 0..4 = Z3,Z7,Z11,Z15,Z27
	VEXTRACTI64X4 $1, Z3, Y31
	VADDPS    Y31, Y3, Y3
	VEXTRACTI32X4 $1, Z3, X31
	VADDPS    X31, X3, X3
	VPSHUFD   $0xEE, X3, X31
	VADDPS    X31, X3, X3
	VPSHUFD   $0x55, X3, X31
	VADDSS    X31, X3, X3
	VMOVSS    X3, 0(R9)
	VEXTRACTI64X4 $1, Z7, Y31
	VADDPS    Y31, Y7, Y7
	VEXTRACTI32X4 $1, Z7, X31
	VADDPS    X31, X7, X7
	VPSHUFD   $0xEE, X7, X31
	VADDPS    X31, X7, X7
	VPSHUFD   $0x55, X7, X31
	VADDSS    X31, X7, X7
	VMOVSS    X7, 4(R9)
	VEXTRACTI64X4 $1, Z11, Y31
	VADDPS    Y31, Y11, Y11
	VEXTRACTI32X4 $1, Z11, X31
	VADDPS    X31, X11, X11
	VPSHUFD   $0xEE, X11, X31
	VADDPS    X31, X11, X11
	VPSHUFD   $0x55, X11, X31
	VADDSS    X31, X11, X11
	VMOVSS    X11, 8(R9)
	VEXTRACTI64X4 $1, Z15, Y31
	VADDPS    Y31, Y15, Y15
	VEXTRACTI32X4 $1, Z15, X31
	VADDPS    X31, X15, X15
	VPSHUFD   $0xEE, X15, X31
	VADDPS    X31, X15, X15
	VPSHUFD   $0x55, X15, X31
	VADDSS    X31, X15, X15
	VMOVSS    X15, 12(R9)
	VEXTRACTI64X4 $1, Z27, Y31
	VADDPS    Y31, Y27, Y27
	VEXTRACTI32X4 $1, Z27, X31
	VADDPS    X31, X27, X27
	VPSHUFD   $0xEE, X27, X31
	VADDPS    X31, X27, X27
	VPSHUFD   $0x55, X27, X31
	VADDSS    X31, X27, X27
	VMOVSS    X27, 16(R9)

	VZEROUPPER
	RET

// qgemm8tile512x1 computes a 1(row)×4(token) Q8_0 output tile. It is used only for
// qgemm8tile512's row remainder, so non-multiple-of-5 output shapes do not fall all the way
// back to qgemm8cell for every token.
//
// func qgemm8tile512x1(qw, qx *int8, dw, dx *float32, in, nblk, outStride int, dst *float32)
TEXT ·qgemm8tile512x1(SB), NOSPLIT, $0-64
	MOVQ qw+0(FP), SI
	MOVQ qx+8(FP), DI
	MOVQ dw+16(FP), R8
	MOVQ dx+24(FP), R9
	MOVQ in+32(FP), R10
	MOVQ nblk+40(FP), CX
	MOVQ R10, R14
	ADDQ R10, R14
	MOVQ R14, R15
	ADDQ R10, R15
	MOVQ CX, R11
	SHLQ $2, R11
	MOVQ R11, R12
	ADDQ R11, R12
	MOVQ R12, R13
	ADDQ R11, R13
	VPXORD Z0, Z0, Z0
	VPXORD Z1, Z1, Z1
	VPXORD Z2, Z2, Z2
	VPXORD Z3, Z3, Z3
	TESTQ CX, CX
	JLE   x1reduce

x1loop:
	VPMOVSXBW (DI), Z16
	VPMOVSXBW (DI)(R10*1), Z17
	VPMOVSXBW (DI)(R14*1), Z18
	VPMOVSXBW (DI)(R15*1), Z19

	VPMOVSXBW (SI), Z20
	VMOVSS    (R8), X28
	VMULSS       (R9), X28, X23
	VBROADCASTSS X23, Z22
	VPMADDWD  Z16, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z0
	VMULSS       (R9)(R11*1), X28, X23
	VBROADCASTSS X23, Z22
	VPMADDWD  Z17, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z1
	VMULSS       (R9)(R12*1), X28, X23
	VBROADCASTSS X23, Z22
	VPMADDWD  Z18, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z2
	VMULSS       (R9)(R13*1), X28, X23
	VBROADCASTSS X23, Z22
	VPMADDWD  Z19, Z20, Z21
	VCVTDQ2PS Z21, Z21
	VFMADD231PS Z22, Z21, Z3

	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $4, R8
	ADDQ $4, R9
	DECQ CX
	JNZ  x1loop

x1reduce:
	MOVQ dst+56(FP), AX
	MOVQ outStride+48(FP), DX
	SHLQ $2, DX
	MOVQ AX, BX
	ADDQ DX, BX
	MOVQ BX, R8
	ADDQ DX, R8
	MOVQ R8, R9
	ADDQ DX, R9

	VEXTRACTI64X4 $1, Z0, Y31
	VADDPS    Y31, Y0, Y0
	VEXTRACTI32X4 $1, Z0, X31
	VADDPS    X31, X0, X0
	VPSHUFD   $0xEE, X0, X31
	VADDPS    X31, X0, X0
	VPSHUFD   $0x55, X0, X31
	VADDSS    X31, X0, X0
	VMOVSS    X0, 0(AX)

	VEXTRACTI64X4 $1, Z1, Y31
	VADDPS    Y31, Y1, Y1
	VEXTRACTI32X4 $1, Z1, X31
	VADDPS    X31, X1, X1
	VPSHUFD   $0xEE, X1, X31
	VADDPS    X31, X1, X1
	VPSHUFD   $0x55, X1, X31
	VADDSS    X31, X1, X1
	VMOVSS    X1, 0(BX)

	VEXTRACTI64X4 $1, Z2, Y31
	VADDPS    Y31, Y2, Y2
	VEXTRACTI32X4 $1, Z2, X31
	VADDPS    X31, X2, X2
	VPSHUFD   $0xEE, X2, X31
	VADDPS    X31, X2, X2
	VPSHUFD   $0x55, X2, X31
	VADDSS    X31, X2, X2
	VMOVSS    X2, 0(R8)

	VEXTRACTI64X4 $1, Z3, Y31
	VADDPS    Y31, Y3, Y3
	VEXTRACTI32X4 $1, Z3, X31
	VADDPS    X31, X3, X3
	VPSHUFD   $0xEE, X3, X31
	VADDPS    X31, X3, X3
	VPSHUFD   $0x55, X3, X31
	VADDSS    X31, X3, X3
	VMOVSS    X3, 0(R9)

	VZEROUPPER
	RET

// func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL eaxArg+0(FP), AX
	MOVL ecxArg+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func xgetbv() (eax, edx uint32)  — reads XCR0 (ECX=0)
TEXT ·xgetbv(SB), NOSPLIT, $0-8
	XORL CX, CX
	XGETBV
	MOVL AX, eax+0(FP)
	MOVL DX, edx+4(FP)
	RET
