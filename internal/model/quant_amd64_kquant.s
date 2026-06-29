//go:build amd64

#include "textflag.h"

// q6kc<> — kernel constants for Q6_K (same layout as q5kc<>):
//   +0x00: 0x0F0F0F0F  low-nibble mask (broadcast to 32 bytes of 0x0F)
//   +0x04: 0x00010001  two int16 ones (16 lanes of +1 for the AVX2 Σqx VPMADDWD dot)
//   +0x08: 0x01010101  byte ones (16 bytes of 1 for the VNNI Σqx VPDPBUSD dot)
DATA q6kc<>+0x00(SB)/4, $0x0F0F0F0F
DATA q6kc<>+0x04(SB)/4, $0x00010001
DATA q6kc<>+0x08(SB)/4, $0x01010101
GLOBL q6kc<>(SB), RODATA|NOPTR, $12

// quant_amd64_kquant.s — the AVX2 integer-reduction kernels for resident K-quant decode.
// This file contains both Q5_K and Q6_K kernels.

// q5kc<> — kernel constants for Q5_K:
//   +0x00: 0x0F0F0F0F  low-nibble mask (broadcast to 32 bytes of 0x0F)
//   +0x04: 0x00010001  two int16 ones (16 lanes of +1 for the AVX2 Σqx VPMADDWD dot)
//   +0x08: 0x01010101  byte ones (32 bytes of 1 for the VNNI Σqx VPDPBUSD dot)
DATA q5kc<>+0x00(SB)/4, $0x0F0F0F0F
DATA q5kc<>+0x04(SB)/4, $0x00010001
DATA q5kc<>+0x08(SB)/4, $0x01010101
GLOBL q5kc<>(SB), RODATA|NOPTR, $12

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
	VPBROADCASTD q5kc<>+0x04(SB), Y7   // int16 +1 (AVX2 Σqx)
	VPBROADCASTD q5kc<>+0x08(SB), Y15  // byte +1 (VNNI Σqx); unused on the AVX2 path

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
// activation at (DI)), stores both int32, and advances DI/R8/R9 by 32/4/4. Clobbers Y3,Y8..Y13,X0.
// Y1 is treated as unsigned weight bytes (0..31). When the package flag q5kUseVNNI is set (the box
// has AVX512-VNNI), it takes the one-VPDPBUSD-per-dot fast path (Y15 holds byte-ones, set by the
// caller); else the AVX2 sign-extend + VPMADDWD path. Both produce bit-identical int32 reductions.
TEXT q5kdot<>(SB), NOSPLIT, $0-0
	CMPB ·q5kUseVNNI(SB), $0
	JEQ  q5kdotAVX2
	// VNNI: I via VPDPBUSD(u8 nibble, s8 qx); S via VPDPBUSD(u8 ones, s8 qx).
	VMOVDQU (DI), Y3
	VPXOR   Y8, Y8, Y8
	VPDPBUSD Y3, Y1, Y8
	VPXOR   Y9, Y9, Y9
	VPDPBUSD Y3, Y15, Y9
	VEXTRACTI128 $1, Y8, X13
	VPADDD  X13, X8, X8
	VPSHUFD $0xEE, X8, X13
	VPADDD  X13, X8, X8
	VPSHUFD $0x55, X8, X13
	VPADDD  X13, X8, X8
	VMOVD   X8, (R8)
	VEXTRACTI128 $1, Y9, X13
	VPADDD  X13, X9, X9
	VPSHUFD $0xEE, X9, X13
	VPADDD  X13, X9, X9
	VPSHUFD $0x55, X9, X13
	VPADDD  X13, X9, X9
	VMOVD   X9, (R9)
	ADDQ $32, DI
	ADDQ $4, R8
	ADDQ $4, R9
	RET

q5kdotAVX2:
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

// q6kReduceRowAsmAVX2 — the AVX2/VNNI integer-reduction kernel for resident Q6_K decode, the third
// K-quant reducer alongside q5kReduceRowAsmAVX2 (the deferred #209 slice). For one weight row (nblk
// super-blocks) and a Q8_0-quantized activation qx it writes the per-group reductions the shared-Go
// combine (q6kCombineRow) folds into the dot:
//
//	I_g = Σ_{l∈group g} q6[l]*qx[l]      (q6 = nibble | (qh-2-bits<<4), range 0..63, non-negative)
//	S_g = Σ_{l∈group g} qx[l]
//
// into IS[b*16+g] / SS[b*16+g] for all 16 groups of every super-block, in ascending g order so the
// IS/SS writes are a simple sequential walk. Bit-identical to q6kReduceRowScalar (pinned by
// TestQ6KReduceAsmMatchesScalar); the float combine is shared Go.
//
// Q6_K super-block layout (q6kBlockBytes = qkK/2 + qkK/4 + qkK/16 + 2 = 128 + 64 + 16 + 2 = 210):
//   ql[0:128] at offset 0   qh[128:192] at offset 128   scales[192:208]   d (f16) at offset 208
//
// Two chunks of 128 weights. Within a chunk (ql 64 bytes split into the low region bytes 0..31 and
// the high region bytes 32..63; 32 qh bytes), the four positions p∈{0,1,2,3} reconstruct EXACTLY as
// q6kReduceRowScalar: p selects (ql region, nibble, qh 2-bit field) and adds p*32 to the qx offset:
//   p=0: ql_lo & 0x0f | ((qh>>0)&3)<<4   p=1: ql_hi & 0x0f | ((qh>>2)&3)<<4
//   p=2: ql_lo >> 4   | ((qh>>4)&3)<<4   p=3: ql_hi >> 4   | ((qh>>6)&3)<<4
// The 2-bit qh field is isolated with a per-position byte mask (0x03/0x0C/0x30/0xC0) FIRST, then a
// 16-bit-lane shift moves it to bit 4 (<<4/<<2/none/>>2). Isolating before the lane shift keeps each
// byte's bits inside its byte (same invariant the Q5_K kernel relies on), so the lane op is exact.
// Each 32-lane position splits into two 16-lane groups (is=0 lanes 0..15, is=1 lanes 16..31), each
// against one int8 activation block — that is the Q6_K-specific shape vs Q5_K's 32-wide sub-block.
//
// The processing order (chunk → p → is) walks qx and IS/SS strictly forward: group (chunk,is,p)
// has qx offset chunk*128 + p*32 + is*16 and IS/SS index chunk*8 + p*2 + is, so DI/R8/R9 just
// advance 16/4/4 per group and need no per-position reset.
//
// Registers: SI=row, R11=ql ptr (+64/chunk), R12=qh ptr (+32/chunk), DI=qx (+16/group), R8=IS,
// R9=SS, CX=super-blocks left, R13=chunk counter, AX=mask build. Y6=0x0F mask, Y7=int16 ones,
// Y15=byte ones (VNNI), Y14=qh bytes, Y4/Y5=ql low/high regions, Y0..Y3 scratch.
//
// Helper q6kdot16 computes I=Σq6*qx and S=Σqx for one 16-lane group (weights in X8, qx at (DI)),
// stores both int32, advances DI/IS/SS. VNNI (·q6kUseVNNI set) takes one VPDPBUSD per dot; else
// the AVX2 sign-extend + VPMADDWD path. Both bit-identical. Clobbers Y9..Y12 (X15 read-only).
//
// func q6kReduceRowAsmAVX2(row *byte, nblk int, qx *int8, Isum, Ssum *int32)
TEXT ·q6kReduceRowAsmAVX2(SB), NOSPLIT, $0-40
	MOVQ row+0(FP), SI
	MOVQ nblk+8(FP), CX
	MOVQ qx+16(FP), DI
	MOVQ Isum+24(FP), R8
	MOVQ Ssum+32(FP), R9

	TESTQ CX, CX
	JLE   done6k

	VPBROADCASTD q6kc<>+0x00(SB), Y6   // 0x0F bytes (nibble mask)
	VPBROADCASTD q6kc<>+0x04(SB), Y7   // int16 +1 (AVX2 Σqx)
	VPBROADCASTD q6kc<>+0x08(SB), Y15  // byte +1 (VNNI Σqx); unused on the AVX2 path

sblock6k:
	LEAQ 0(SI), R11       // ql ptr (offset 0)
	LEAQ 128(SI), R12     // qh ptr (offset 128)
	MOVQ $2, R13          // 2 chunks of 128 weights

chunk6k:
	VMOVDQU (R12), Y14    // 32 qh bytes for this chunk
	VMOVDQU (R11), Y4     // ql low region (bytes 0..31)
	VMOVDQU 32(R11), Y5   // ql high region (bytes 32..63)

	// ---- position 0: ql_lo low nibble | ((qh>>0)&3)<<4 ----
	VPAND   Y6, Y4, Y1             // ql_lo & 0x0f
	MOVL    $0x03030303, AX
	MOVQ    AX, X2
	VPBROADCASTD X2, Y2           // 0x03 per byte (bits 1:0)
	VPAND   Y14, Y2, Y3           // qh & 0x03
	VPSLLW  $4, Y3, Y3            // -> bits 5:4
	VPADDB  Y3, Y1, Y1           // q6 (0..63)
	VEXTRACTI128 $0, Y1, X8       // is=0 lanes 0..15
	CALL    q6kdot16<>(SB)
	VEXTRACTI128 $1, Y1, X8       // is=1 lanes 16..31
	CALL    q6kdot16<>(SB)

	// ---- position 1: ql_hi low nibble | ((qh>>2)&3)<<4 ----
	VPAND   Y6, Y5, Y1            // ql_hi & 0x0f
	MOVL    $0x0C0C0C0C, AX
	MOVQ    AX, X2
	VPBROADCASTD X2, Y2          // 0x0C per byte (bits 3:2)
	VPAND   Y14, Y2, Y3          // qh & 0x0C
	VPSLLW  $2, Y3, Y3           // -> bits 5:4
	VPADDB  Y3, Y1, Y1
	VEXTRACTI128 $0, Y1, X8
	CALL    q6kdot16<>(SB)
	VEXTRACTI128 $1, Y1, X8
	CALL    q6kdot16<>(SB)

	// ---- position 2: ql_lo high nibble | ((qh>>4)&3)<<4 ----
	VPSRLW  $4, Y4, Y1            // ql_lo >> 4 (per 16-bit lane)
	VPAND   Y6, Y1, Y1           // & 0x0f -> high nibble
	MOVL    $0x30303030, AX
	MOVQ    AX, X2
	VPBROADCASTD X2, Y2          // 0x30 per byte (bits 5:4, already in place)
	VPAND   Y14, Y2, Y3          // qh & 0x30
	VPADDB  Y3, Y1, Y1
	VEXTRACTI128 $0, Y1, X8
	CALL    q6kdot16<>(SB)
	VEXTRACTI128 $1, Y1, X8
	CALL    q6kdot16<>(SB)

	// ---- position 3: ql_hi high nibble | ((qh>>6)&3)<<4 ----
	VPSRLW  $4, Y5, Y1            // ql_hi >> 4
	VPAND   Y6, Y1, Y1
	MOVL    $0xC0C0C0C0, AX
	MOVQ    AX, X2
	VPBROADCASTD X2, Y2          // 0xC0 per byte (bits 7:6)
	VPAND   Y14, Y2, Y3          // qh & 0xC0
	VPSRLW  $2, Y3, Y3           // -> bits 5:4
	VPADDB  Y3, Y1, Y1
	VEXTRACTI128 $0, Y1, X8
	CALL    q6kdot16<>(SB)
	VEXTRACTI128 $1, Y1, X8
	CALL    q6kdot16<>(SB)

	ADDQ $64, R11                // next chunk's ql (+64)
	ADDQ $32, R12                // next chunk's qh (+32)
	DECQ R13
	JNZ  chunk6k

	ADDQ $210, SI                // next super-block (q6kBlockBytes)
	DECQ CX
	JNZ  sblock6k

done6k:
	VZEROUPPER
	RET

// q6kdot16 computes I = Σ X8*qx and S = Σ qx for one 16-lane Q6_K group (q6 weights 0..63 in the
// low 16 bytes of X8, activation at (DI)), stores both int32, and advances DI/R8/R9 by 16/4/4.
// VNNI (·q6kUseVNNI set): one 128-bit VPDPBUSD per dot (q6 weights are 0..63 so the u8 operand is
// exact); else the AVX2 sign-extend + VPMADDWD path. Both produce bit-identical int32 reductions.
// Clobbers Y9..Y12; reads X15 (byte ones, VNNI only).
TEXT q6kdot16<>(SB), NOSPLIT, $0-0
	CMPB ·q6kUseVNNI(SB), $0
	JEQ  q6kdot16AVX2
	// VNNI: I via VPDPBUSD(u8 q6, s8 qx); S via VPDPBUSD(u8 ones, s8 qx). 128-bit (16 lanes).
	VMOVDQU (DI), X10
	VPXOR    X9, X9, X9
	VPDPBUSD X10, X8, X9          // I += q6(u8)·qx(s8)
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

q6kdot16AVX2:
	VMOVDQU (DI), X10             // 16 qx bytes (XMM load — no over-read past the row)
	VPMOVSXBW X8, Y9             // 16 q6 -> int16 (q6 0..63, sign==zero extend)
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
