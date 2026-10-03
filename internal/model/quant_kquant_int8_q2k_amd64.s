//go:build amd64

#include "textflag.h"

// quant_kquant_int8_q2k_amd64.s — the AVX2 integer-reduction kernel for resident Q2_K decode, the
// Q2_K sibling of quant_amd64_kquant.s (Q5_K/Q6_K). For one weight row (nblk super-blocks) and a
// Q8_0-quantized activation qx it writes the per-sub-block reductions the shared-Go combine
// (q2kCombineRow) folds into the dot:
//
//	I_s = Σ_{l∈sub s} q2[l]*qx[l]      (q2 = (code>>2j)&3, range 0..3, non-negative)
//	S_s = Σ_{l∈sub s} qx[l]
//
// into IS[b*16+s] / SS[b*16+s]. Bit-identical to q2kReduceRowScalar (pinned by
// TestQ2KReduceAsmMatchesScalar); the float combine is shared Go.
//
// MIT provenance: direct port/adapt of the integer reduction of ggml_vec_dot_q2_K_q8_K at
//   upstream:   ggml-org/llama.cpp
//   commit pin: 83078fec0db82d6b5a00d9599062c38c39145755
//   source:     ggml/src/ggml-cpu/arch/x86/quants.c
//   license:    MIT (upstream LICENSE at the pin); Copyright (c) 2023-2026 The ggml authors
// Native Go assembly; no cgo/FFI.
//
// Q2_K super-block layout (q2kBlockBytes = 16+64+2+2 = 84): scales[0:16], code bytes q[16:80],
// d=f16[80:82], min=f16[82:84]. Two 128-weight chunks (n = 0, 128). Per chunk the code pointer
// starts at q+16 (R11) and advances +32 per chunk. Within a chunk, shift = 2*j for j=0..3; each
// shift reads 32 consecutive code bytes and produces two 16-lane sub-blocks:
//   sub s0 = base + (n/128)*8 + 2*j : lanes l=0..15 read q[qi+l],    qx[xBase+n+j*32+l]
//   sub s1 = s0+1                   : lanes l=0..15 read q[qi+16+l], qx[xBase+n+j*32+16+l]
// Sub-blocks are written in strict s order (chunk, then j, then half), so DI/R8/R9 advance 16/4/4
// per helper call with no reset.
//
// Registers: SI=row (+84/super-block), DI=qx (+16/group), R8=IS, R9=SS, CX=super-blocks left,
// R11=code ptr (+32/chunk), R13=j counter, AX=shift. Y6=0x03 mask, Y7=int16 ones, Y15=byte ones
// (VNNI), Y0=this chunk's 64 code bytes (held live across the helper CALLs). Helper q2kdot16
// clobbers Y9..Y12 and reads Y6/Y7/Y15 and X8/X10 (it never touches Y0).

// q2kc<> — kernel constants for Q2_K:
//   +0x00: 0x03030303  2-bit mask (broadcast to 32 bytes of 0x03)
//   +0x04: 0x00010001  two int16 ones (16 lanes of +1 for the AVX2 Σqx VPMADDWD dot)
//   +0x08: 0x01010101  byte ones (16 bytes of 1 for the VNNI Σqx VPDPBUSD dot)
DATA q2kc<>+0x00(SB)/4, $0x03030303
DATA q2kc<>+0x04(SB)/4, $0x00010001
DATA q2kc<>+0x08(SB)/4, $0x01010101
GLOBL q2kc<>(SB), RODATA|NOPTR, $12

// func q2kReduceRowAsmAVX2(row *byte, nblk int, qx *int8, Isum, Ssum *int32)
TEXT ·q2kReduceRowAsmAVX2(SB), NOSPLIT, $0-40
	MOVQ row+0(FP), SI
	MOVQ nblk+8(FP), CX
	MOVQ qx+16(FP), DI
	MOVQ Isum+24(FP), R8
	MOVQ Ssum+32(FP), R9

	TESTQ CX, CX
	JLE   done2k

	VPBROADCASTD q2kc<>+0x00(SB), Y6   // 0x03 bytes
	VPBROADCASTD q2kc<>+0x04(SB), Y7   // int16 +1 (AVX2 Σqx)
	VPBROADCASTD q2kc<>+0x08(SB), Y15  // byte +1 (VNNI Σqx); unused on the AVX2 path

sblock2k:
	LEAQ 16(SI), R11      // code ptr (offset 16)

	// ---- chunk 0 (n = 0): 32 code bytes at q[0:32] ----
	VMOVDQU (R11), Y0     // Y0 = q[0:32] for this chunk (held live across the CALLs)

	// j=0, shift 0, base sub-block 0: shift 0 is a no-op, just mask.
	VPAND   Y6, Y0, Y1
	VEXTRACTI128 $0, Y1, X8
	CALL    q2kdot16<>(SB)
	VEXTRACTI128 $1, Y1, X8
	CALL    q2kdot16<>(SB)

	// j=1, shift 2, base sub-block 2
	VPSRLW  $2, Y0, Y1
	VPAND   Y6, Y1, Y1
	VEXTRACTI128 $0, Y1, X8
	CALL    q2kdot16<>(SB)
	VEXTRACTI128 $1, Y1, X8
	CALL    q2kdot16<>(SB)

	// j=2, shift 4, base sub-block 4
	VPSRLW  $4, Y0, Y1
	VPAND   Y6, Y1, Y1
	VEXTRACTI128 $0, Y1, X8
	CALL    q2kdot16<>(SB)
	VEXTRACTI128 $1, Y1, X8
	CALL    q2kdot16<>(SB)

	// j=3, shift 6, base sub-block 6
	VPSRLW  $6, Y0, Y1
	VPAND   Y6, Y1, Y1
	VEXTRACTI128 $0, Y1, X8
	CALL    q2kdot16<>(SB)
	VEXTRACTI128 $1, Y1, X8
	CALL    q2kdot16<>(SB)

	ADDQ $32, R11         // advance to chunk 1's code bytes (q[32:64])

	// ---- chunk 1 (n = 128): 32 code bytes at q[32:64], base sub-block 8 ----
	VMOVDQU (R11), Y0

	// j=0, shift 0, base sub-block 8
	VPAND   Y6, Y0, Y1
	VEXTRACTI128 $0, Y1, X8
	CALL    q2kdot16<>(SB)
	VEXTRACTI128 $1, Y1, X8
	CALL    q2kdot16<>(SB)

	// j=1, shift 2, base sub-block 10
	VPSRLW  $2, Y0, Y1
	VPAND   Y6, Y1, Y1
	VEXTRACTI128 $0, Y1, X8
	CALL    q2kdot16<>(SB)
	VEXTRACTI128 $1, Y1, X8
	CALL    q2kdot16<>(SB)

	// j=2, shift 4, base sub-block 12
	VPSRLW  $4, Y0, Y1
	VPAND   Y6, Y1, Y1
	VEXTRACTI128 $0, Y1, X8
	CALL    q2kdot16<>(SB)
	VEXTRACTI128 $1, Y1, X8
	CALL    q2kdot16<>(SB)

	// j=3, shift 6, base sub-block 14
	VPSRLW  $6, Y0, Y1
	VPAND   Y6, Y1, Y1
	VEXTRACTI128 $0, Y1, X8
	CALL    q2kdot16<>(SB)
	VEXTRACTI128 $1, Y1, X8
	CALL    q2kdot16<>(SB)

	ADDQ $84, SI          // next super-block (q2kBlockBytes)
	DECQ CX
	JNZ  sblock2k

done2k:
	VZEROUPPER
	RET

// q2kdot16 computes I = Σ X8*qx and S = Σ qx for one 16-lane Q2_K sub-block (q2 weights 0..3 in the
// low 16 bytes of X8, activation at (DI)), stores both int32, and advances DI/R8/R9 by 16/4/4.
// VNNI (·q2kUseVNNI set): one 128-bit VPDPBUSD per dot (q2 weights are 0..3 so the u8 operand is
// exact); else the AVX2 sign-extend + VPMADDWD path. Both produce bit-identical int32 reductions.
// Clobbers Y9..Y12; reads Y6/Y7/Y15 and X8/X10. Never touches Y0 (the caller's live code register).
TEXT q2kdot16<>(SB), NOSPLIT, $0-0
	CMPB ·q2kUseVNNI(SB), $0
	JEQ  q2kdot16AVX2
	// VNNI: I via VPDPBUSD(u8 q2, s8 qx); S via VPDPBUSD(u8 ones, s8 qx). 128-bit (16 lanes).
	VMOVDQU (DI), X10
	VPXOR    X9, X9, X9
	VPDPBUSD X10, X8, X9          // I += q2(u8)·qx(s8)
	VPXOR    X11, X11, X11
	VPDPBUSD X10, X15, X11        // S += ones(u8)·qx(s8)
	VPSHUFD  $0xEE, X9, X12
	VPADDD   X12, X9, X9
	VPSHUFD  $0x55, X9, X12
	VPADDD   X12, X9, X9
	VMOVD    X9, (R8)
	VPSHUFD  $0xEE, X11, X12
	VPADDD   X12, X11, X11
	VPSHUFD  $0x55, X11, X12
	VPADDD   X12, X11, X11
	VMOVD    X11, (R9)
	ADDQ $16, DI
	ADDQ $4, R8
	ADDQ $4, R9
	RET

q2kdot16AVX2:
	VMOVDQU (DI), X10             // 16 qx bytes (XMM load — no over-read past the row)
	VPMOVSXBW X8, Y9             // 16 q2 -> int16 (q2 0..3, sign==zero extend)
	VPMOVSXBW X10, Y11          // 16 qx -> int16
	VPMADDWD  Y11, Y9, Y9       // I: 8 int32 pairwise products
	VPMOVSXBW X10, Y11          // re-extend qx fresh for S (do NOT trust Y11's upper after the madd)
	VPMADDWD  Y7, Y11, Y11      // S: 8 int32 pairwise (ones·qx)
	VEXTRACTI128 $1, Y9, X12
	VPADDD    X12, X9, X9
	VPSHUFD   $0xEE, X9, X12
	VPADDD    X12, X9, X9
	VPSHUFD   $0x55, X9, X12
	VPADDD    X12, X9, X9
	VMOVD     X9, (R8)
	VEXTRACTI128 $1, Y11, X12
	VPADDD    X12, X11, X11
	VPSHUFD   $0xEE, X11, X12
	VPADDD    X12, X11, X11
	VPSHUFD   $0x55, X11, X12
	VPADDD    X12, X11, X11
	VMOVD     X11, (R9)
	ADDQ $16, DI
	ADDQ $4, R8
	ADDQ $4, R9
	RET
