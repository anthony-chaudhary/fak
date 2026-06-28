//go:build amd64

#include "textflag.h"

// q5kc<> — kernel constants:
//   +0x00: 0x0F0F0F0F  low-nibble mask (broadcast to 32 bytes of 0x0F)
//   +0x04: 0x00010001  two int16 ones (broadcast to 16 lanes of +1 for the Σqx dot)
DATA q5kc<>+0x00(SB)/4, $0x0F0F0F0F
DATA q5kc<>+0x04(SB)/4, $0x00010001
GLOBL q5kc<>(SB), RODATA|NOPTR, $8

// quant_amd64_kquant.s — the AVX2 integer-reduction kernel for resident Q5_K decode, the K-quant
// sibling of quant_amd64_q4k.s. For one weight row (nblk super-blocks) and a Q8_0 activation qx it
// writes the per-sub-block reductions the shared-Go combine (kQuantCombineRow) folds into the dot:
//
//	I_s = Σ_{l∈sub s} q5[l]*qx[l]      (q5 = nibble | (qh-bit<<4), range 0..31, non-negative)
//	S_s = Σ_{l∈sub s} qx[l]
//
// into IS[b*8+s] / SS[b*8+s]. Bit-identical to q5kReduceRowScalar (pinned by
// TestQ5KReduceAsmMatchesScalar); the float combine is shared Go.
//
// Q5_K super-block layout (q5kBlockBytes = 2+2+12+32+128 = 176): 16 B header (d,min,scales),
// then qh[32] at offset 16, then ql[128] at offset 48. Each super-block has 4 chunks of 64 weights
// = 8 sub-blocks of 32. Chunk c (c=0..3) reads ql[c*32 : c*32+32] (32 bytes shared by its two
// sub-blocks) and reconstructs:
//   sub 2c   (low nibble):  q5 = (ql & 0x0f) + 16*bit(2c)   of qh[l]
//   sub 2c+1 (high nibble): q5 = (ql >> 4)   + 16*bit(2c+1) of qh[l]
// The qh 5th bit is isolated per byte with AND (1<<shift), then moved to bit 4 (value 16) with a
// constant per-chunk 16-bit-lane shift: c=0 <<4, c=1 <<2, c=2 none, c=3 >>2 (the isolated bit never
// crosses a byte boundary, so the lane shift is exact per byte). The activation qx is 8 contiguous
// 32-byte sub-blocks, read in order 0,1,2,...,7 (sequential 32-byte advances).
//
// Registers: R11=ql ptr (+32/chunk), R12=qh ptr (reloaded /super-block), DI=qx (+32/sub-block),
// R8=IS, R9=SS, CX=super-blocks left. Y6=0x0F mask, Y7=int16 ones, Y13=byte 0x01 mask base.
//
// Helper sub-block flow (inline twice per chunk): given a 32-byte weight YMM Wq (the assembled q5
// values, 0..31) and qx at (DI), compute I=ΣWq*qx and S=Σqx, store both int32, advance DI/IS/SS.
//
// func q5kReduceRowAsmAVX2(row *byte, nblk int, qx *int8, Isum, Ssum *int32)
TEXT ·q5kReduceRowAsmAVX2(SB), NOSPLIT, $0-40
	MOVQ row+0(FP), SI
	MOVQ nblk+8(FP), CX
	MOVQ qx+16(FP), DI
	MOVQ Isum+24(FP), R8
	MOVQ Ssum+32(FP), R9

	TESTQ CX, CX
	JLE   done

	VPBROADCASTD q5kc<>+0x00(SB), Y6   // 0x0F bytes
	VPBROADCASTD q5kc<>+0x04(SB), Y7   // int16 +1

sblock:
	LEAQ 16(SI), R12      // qh ptr (offset 16)
	LEAQ 48(SI), R11      // ql ptr (offset 48)
	VMOVDQU (R12), Y14    // 32 qh bytes (shared across the 4 chunks)

	// ---- chunk 0: low-bit shift <<4 (bit0->bit4), high-bit shift <<3 (bit1->bit4) ----
	VMOVDQU (R11), Y0
	// low nibble sub-block 2c
	VPAND   Y6, Y0, Y1                 // ql & 0x0f
	MOVL    $0x01010101, AX
	MOVQ    AX, X8
	VPBROADCASTD X8, Y8                // 0x01 per byte
	VPAND   Y14, Y8, Y9                // qh & bit0
	VPSLLW  $4, Y9, Y9                 // ->bit4 (value 16)
	VPADDB  Y9, Y1, Y1                 // q5 (low) 0..31
	CALL    q5kdot<>(SB)
	// high nibble sub-block 2c+1
	VPSRLW  $4, Y0, Y2
	VPAND   Y6, Y2, Y2                 // ql >> 4 (per byte)
	MOVL    $0x02020202, AX
	MOVQ    AX, X8
	VPBROADCASTD X8, Y8                // 0x02 per byte (bit1)
	VPAND   Y14, Y8, Y9
	VPSLLW  $3, Y9, Y9                 // bit1 ->bit4
	VPADDB  Y9, Y2, Y2
	VMOVDQU Y2, Y1
	CALL    q5kdot<>(SB)
	ADDQ $32, R11

	// ---- chunk 1: low bit2 <<2, high bit3 <<1 ----
	VMOVDQU (R11), Y0
	VPAND   Y6, Y0, Y1
	MOVL    $0x04040404, AX
	MOVQ    AX, X8
	VPBROADCASTD X8, Y8
	VPAND   Y14, Y8, Y9
	VPSLLW  $2, Y9, Y9
	VPADDB  Y9, Y1, Y1
	CALL    q5kdot<>(SB)
	VPSRLW  $4, Y0, Y2
	VPAND   Y6, Y2, Y2
	MOVL    $0x08080808, AX
	MOVQ    AX, X8
	VPBROADCASTD X8, Y8
	VPAND   Y14, Y8, Y9
	VPSLLW  $1, Y9, Y9
	VPADDB  Y9, Y2, Y2
	VMOVDQU Y2, Y1
	CALL    q5kdot<>(SB)
	ADDQ $32, R11

	// ---- chunk 2: low bit4 (no shift), high bit5 >>1 ----
	VMOVDQU (R11), Y0
	VPAND   Y6, Y0, Y1
	MOVL    $0x10101010, AX
	MOVQ    AX, X8
	VPBROADCASTD X8, Y8
	VPAND   Y14, Y8, Y9                // already bit4 = value 16
	VPADDB  Y9, Y1, Y1
	CALL    q5kdot<>(SB)
	VPSRLW  $4, Y0, Y2
	VPAND   Y6, Y2, Y2
	MOVL    $0x20202020, AX
	MOVQ    AX, X8
	VPBROADCASTD X8, Y8
	VPAND   Y14, Y8, Y9
	VPSRLW  $1, Y9, Y9                 // bit5 ->bit4
	VPADDB  Y9, Y2, Y2
	VMOVDQU Y2, Y1
	CALL    q5kdot<>(SB)
	ADDQ $32, R11

	// ---- chunk 3: low bit6 >>2, high bit7 >>3 ----
	VMOVDQU (R11), Y0
	VPAND   Y6, Y0, Y1
	MOVL    $0x40404040, AX
	MOVQ    AX, X8
	VPBROADCASTD X8, Y8
	VPAND   Y14, Y8, Y9
	VPSRLW  $2, Y9, Y9                 // bit6 ->bit4
	VPADDB  Y9, Y1, Y1
	CALL    q5kdot<>(SB)
	VPSRLW  $4, Y0, Y2
	VPAND   Y6, Y2, Y2
	MOVL    $0x80808080, AX
	MOVQ    AX, X8
	VPBROADCASTD X8, Y8
	VPAND   Y14, Y8, Y9
	VPSRLW  $3, Y9, Y9                 // bit7 ->bit4
	VPADDB  Y9, Y2, Y2
	VMOVDQU Y2, Y1
	CALL    q5kdot<>(SB)
	ADDQ $32, R11

	ADDQ $176, SI                      // next super-block (q5kBlockBytes = 16+32+128)
	DECQ CX
	JNZ  sblock

done:
	VZEROUPPER
	RET

// q5kdot computes I = Σ Y1*qx and S = Σ qx for one 32-wide sub-block (weights in Y1, 0..31,
// activation at (DI)), stores both int32, and advances DI/R8/R9 by 32/4/4. Clobbers Y3,Y8..Y11,X0.
// Y1 is treated as unsigned weight bytes (0..31) sign-extended to int16 (top bit always 0).
TEXT q5kdot<>(SB), NOSPLIT, $0-0
	VMOVDQU (DI), Y3
	// I = Σ w*qx : two 16-byte halves, sign-extend, VPMADDWD, add.
	VPMOVSXBW X1, Y8
	VPMOVSXBW X3, Y9
	VPMADDWD  Y9, Y8, Y8
	VEXTRACTI128 $1, Y1, X10
	VEXTRACTI128 $1, Y3, X11
	VPMOVSXBW X10, Y10
	VPMOVSXBW X11, Y11
	VPMADDWD  Y11, Y10, Y10
	VPADDD    Y10, Y8, Y8
	// S = Σ qx : VPMADDWD(qx_int16, ones16) over both halves. Re-extract BOTH qx halves fresh
	// from Y3 (untouched) — do NOT reuse X11 from the I-path: VEX 128-bit ops there leave its
	// upper state unreliable, so trusting it silently drops half the activation sum.
	VPMOVSXBW X3, Y9
	VPMADDWD  Y7, Y9, Y9
	VEXTRACTI128 $1, Y3, X12
	VPMOVSXBW X12, Y12
	VPMADDWD  Y7, Y12, Y12
	VPADDD    Y12, Y9, Y9
	// horizontal-sum Y8 -> I, store. Use X13 (not X0) for scratch: X0 aliases Y0, which the
	// caller holds the live ql register in across the per-sub-block CALL — clobbering it drops
	// the high-nibble sub-block's weights.
	VEXTRACTI128 $1, Y8, X13
	VPADDD    X13, X8, X8
	VPSHUFD   $0xEE, X8, X13
	VPADDD    X13, X8, X8
	VPSHUFD   $0x55, X8, X13
	VPADDD    X13, X8, X8
	VMOVD     X8, (R8)
	// horizontal-sum Y9 -> S, store.
	VEXTRACTI128 $1, Y9, X13
	VPADDD    X13, X9, X9
	VPSHUFD   $0xEE, X9, X13
	VPADDD    X13, X9, X9
	VPSHUFD   $0x55, X9, X13
	VPADDD    X13, X9, X9
	VMOVD     X9, (R9)
	ADDQ $32, DI
	ADDQ $4, R8
	ADDQ $4, R9
	RET
