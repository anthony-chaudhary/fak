//go:build amd64

#include "textflag.h"

// q4kc<> — kernel constants broadcast into vector regs (house pattern, cf. quant_quantize_amd64.s):
//   +0x00: 0x0F0F0F0F  low-nibble mask (broadcast to 32 bytes of 0x0F)
//   +0x04: 0x00010001  two int16 ones (16 lanes of +1 for the AVX2/AVX512 Σqx VPMADDWD dot)
//   +0x08: 0x01010101  byte ones (32 bytes of 1 for the VNNI Σqx VPDPBUSD dot)
DATA q4kc<>+0x00(SB)/4, $0x0F0F0F0F
DATA q4kc<>+0x04(SB)/4, $0x00010001
DATA q4kc<>+0x08(SB)/4, $0x01010101
GLOBL q4kc<>(SB), RODATA|NOPTR, $12

// quant_amd64_q4k.s — the AVX2 integer-reduction kernel for resident Q4_K decode, the amd64 sibling
// of the arm64 NEON SDOT kernel (quant_arm64_q4k.s). It computes, for one weight row (nblk
// super-blocks) and a Q8_0-quantized activation qx, the per-sub-block reductions the shared-Go
// combine (q4kCombineRow) folds into the dot:
//
//	I_s = Σ_{l∈sub s} nibble[l]*qx[l]      (nibbles 0..15 are non-negative int8)
//	S_s = Σ_{l∈sub s} qx[l]                (the min-term sum, computed as a dot vs an all-ones vec)
//
// for all 8 sub-blocks of every super-block, written to IS[b*8+s] / SS[b*8+s]. This kernel owns
// ONLY the integer reductions; the float combine is shared Go, so asm correctness reduces to the
// reductions matching q4kReduceRowScalar bit-for-bit. VPMADDWD (signed int16×int16→int32 pairwise)
// and the ones-vector dot are both associative with no overflow on these ranges
// (|I_s| <= 32*15*127 ≈ 6.1e4, |S_s| <= 32*127 ≈ 4.1e3), so any lane order yields the same int32.
// Pinned by TestQ4KReduceAsmMatchesScalar. Same VPMOVSXBW/VPMADDWD idiom as qdot8asm (quant_amd64.s).
//
// Sub-block layout (matches q4kDequantSuperBlock / the arm64 kernel): each super-block's 144-byte
// record is 16 B header (d,min,scales) + 128 B q field of 4 chunks of 32 bytes; chunk k encodes
// sub-block 2k (LOW nibble of each byte) and 2k+1 (HIGH nibble). The activation qx is 8 contiguous
// 32-byte sub-blocks per super-block (256 int8). q4kBlockBytes = 2+2+12+128 = 144.
//
// Nibble unpack: VPSRLW $4 on a 16-bit lane [hi:lo] yields (hi<<8|lo)>>4, whose low byte is
// (lo>>4)|((hi&0xf)<<4) and high byte is hi>>4; masking with 0x0F then leaves exactly lo>>4 and
// hi>>4 — the per-byte high nibble, matching the scalar q[i]>>4. The low nibble is a plain & 0x0F.
//
// q4kReduceSub (local macro idea, inlined): given a 32-byte weight-nibble YMM Wreg and the 32-byte
// activation at (DI), compute I = Σ W*qx and S = Σ qx and store both as int32, advancing DI by 32
// and IS/SS by 4. Implemented inline twice per chunk (low then high nibble) to keep regs explicit.
//
// Registers: SI=q-field ptr, DI=qx ptr, R8=IS ptr, R9=SS ptr, CX=super-blocks left, R10=chunk
// counter. Y6 = low-nibble mask (0x0F bytes), Y7 = int16 ones. Y0..Y5,Y8..Y11 scratch.
//
// func q4kReduceRowAsmAVX2(row *byte, nblk int, qx *int8, Isum, Ssum *int32)
// (params named Isum/Ssum, not IS/SS: "SS" is the x86 stack-segment register, so SS+N(FP) would
// parse as a register expression — the assembler rejects it with "(register+register)".)
TEXT ·q4kReduceRowAsmAVX2(SB), NOSPLIT, $0-40
	MOVQ row+0(FP), SI
	MOVQ nblk+8(FP), CX
	MOVQ qx+16(FP), DI
	MOVQ Isum+24(FP), R8
	MOVQ Ssum+32(FP), R9

	TESTQ CX, CX
	JLE   done

	// Y6 = 32 bytes of 0x0F (low-nibble mask); Y7 = 16 int16 lanes of +1 (for the Σqx dot).
	VPBROADCASTD q4kc<>+0x00(SB), Y6
	VPBROADCASTD q4kc<>+0x04(SB), Y7

sblock:
	ADDQ $16, SI          // skip d/min/scales (2+2+12) -> q-field ptr
	MOVQ $4, R10          // 4 chunks of 32 q-bytes (128 B)

chunk:
	// Load 32 q-bytes into Y0; low nibbles -> Y1 (sub 2k), high nibbles -> Y2 (sub 2k+1).
	VMOVDQU (SI), Y0
	VPAND   Y6, Y0, Y1        // low nibbles (0..15)
	VPSRLW  $4, Y0, Y2        // >>4 on int16 lanes
	VPAND   Y6, Y2, Y2        // mask -> per-byte high nibbles (0..15)

	// ---- sub-block 2k: weight nibbles Y1, activation at (DI) ----
	VMOVDQU (DI), Y3              // 32 activation int8
	// I = Σ nibble*qx : two 16-byte halves, sign-extend to int16, VPMADDWD, add.
	VPMOVSXBW X1, Y8             // low 16 nibbles -> int16
	VPMOVSXBW X3, Y9             // low 16 qx      -> int16
	VPMADDWD  Y9, Y8, Y8        // 8 int32 partials
	VEXTRACTI128 $1, Y1, X4
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXBW X4, Y10
	VPMOVSXBW X5, Y11
	VPMADDWD  Y11, Y10, Y10
	VPADDD    Y10, Y8, Y8        // I partials (8 int32 in Y8)
	// S = Σ qx : VPMADDWD(qx_int16, ones16) over both halves.
	VPMOVSXBW X3, Y9
	VPMADDWD  Y7, Y9, Y9
	VPMOVSXBW X5, Y11
	VPMADDWD  Y7, Y11, Y11
	VPADDD    Y11, Y9, Y9        // S partials (8 int32 in Y9)
	// horizontal-sum Y8 -> I (int32), Y9 -> S (int32); store; advance DI/IS/SS.
	VEXTRACTI128 $1, Y8, X0
	VPADDD    X0, X8, X8
	VPSHUFD   $0xEE, X8, X0
	VPADDD    X0, X8, X8
	VPSHUFD   $0x55, X8, X0
	VPADDD    X0, X8, X8
	VMOVD     X8, (R8)           // IS[..2k]
	VEXTRACTI128 $1, Y9, X0
	VPADDD    X0, X9, X9
	VPSHUFD   $0xEE, X9, X0
	VPADDD    X0, X9, X9
	VPSHUFD   $0x55, X9, X0
	VPADDD    X0, X9, X9
	VMOVD     X9, (R9)           // SS[..2k]
	ADDQ $32, DI
	ADDQ $4, R8
	ADDQ $4, R9

	// ---- sub-block 2k+1: weight nibbles Y2, activation at (DI) ----
	VMOVDQU (DI), Y3
	VPMOVSXBW X2, Y8
	VPMOVSXBW X3, Y9
	VPMADDWD  Y9, Y8, Y8
	VEXTRACTI128 $1, Y2, X4
	VEXTRACTI128 $1, Y3, X5
	VPMOVSXBW X4, Y10
	VPMOVSXBW X5, Y11
	VPMADDWD  Y11, Y10, Y10
	VPADDD    Y10, Y8, Y8
	VPMOVSXBW X3, Y9
	VPMADDWD  Y7, Y9, Y9
	VPMOVSXBW X5, Y11
	VPMADDWD  Y7, Y11, Y11
	VPADDD    Y11, Y9, Y9
	VEXTRACTI128 $1, Y8, X0
	VPADDD    X0, X8, X8
	VPSHUFD   $0xEE, X8, X0
	VPADDD    X0, X8, X8
	VPSHUFD   $0x55, X8, X0
	VPADDD    X0, X8, X8
	VMOVD     X8, (R8)           // IS[..2k+1]
	VEXTRACTI128 $1, Y9, X0
	VPADDD    X0, X9, X9
	VPSHUFD   $0xEE, X9, X0
	VPADDD    X0, X9, X9
	VPSHUFD   $0x55, X9, X0
	VPADDD    X0, X9, X9
	VMOVD     X9, (R9)           // SS[..2k+1]
	ADDQ $32, DI
	ADDQ $4, R8
	ADDQ $4, R9

	ADDQ $32, SI                 // next chunk's 32 q-bytes
	DECQ R10
	JNZ  chunk

	// SI now points at the end of this super-block's q field (16 header skipped + 128 q).
	// next super-block record starts there; loop.
	DECQ CX
	JNZ  sblock

done:
	VZEROUPPER
	RET

// q4kReduceRowAsmVNNI — the AVX512-VNNI variant. Each 32-wide sub-block dot is ONE VPDPBUSD
// (uint8 weight × int8 activation → 8 int32, the multiply + pair-add + accumulate fused), so the
// per-sub-block kernel drops from the AVX2 path's 4×VPMOVSXBW + 2×VPMADDWD + VPADDD to one
// instruction for I and one for S. Nibbles are unsigned (0..15) = the natural VPDPBUSD `a` operand;
// qx is the signed `b`. Bit-identical to the scalar/AVX2 reductions (same int32, associative).
// Uses YMM VNNI (AVX512VL+AVX512_VNNI); gated by q4kVNNI (CPUID 7,0:ECX bit 11).
//
// func q4kReduceRowAsmVNNI(row *byte, nblk int, qx *int8, Isum, Ssum *int32)
TEXT ·q4kReduceRowAsmVNNI(SB), NOSPLIT, $0-40
	MOVQ row+0(FP), SI
	MOVQ nblk+8(FP), CX
	MOVQ qx+16(FP), DI
	MOVQ Isum+24(FP), R8
	MOVQ Ssum+32(FP), R9

	TESTQ CX, CX
	JLE   vnnidone

	VPBROADCASTD q4kc<>+0x00(SB), Y6   // 0x0F low-nibble mask
	VPBROADCASTD q4kc<>+0x08(SB), Y7   // 0x01 byte ones (for Σqx)

vnnisblock:
	ADDQ $16, SI
	MOVQ $4, R10

vnnichunk:
	VMOVDQU (SI), Y0
	VPAND   Y6, Y0, Y1        // low nibbles (sub 2k), unsigned 0..15
	VPSRLW  $4, Y0, Y2
	VPAND   Y6, Y2, Y2        // high nibbles (sub 2k+1)

	// sub-block 2k
	VMOVDQU (DI), Y3
	VPXOR   Y8, Y8, Y8
	VPDPBUSD Y3, Y1, Y8       // I = Σ u8(nibble)*s8(qx) -> 8 int32
	VPXOR   Y9, Y9, Y9
	VPDPBUSD Y3, Y7, Y9       // S = Σ u8(1)*s8(qx) -> 8 int32
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

	// sub-block 2k+1
	VMOVDQU (DI), Y3
	VPXOR   Y8, Y8, Y8
	VPDPBUSD Y3, Y2, Y8
	VPXOR   Y9, Y9, Y9
	VPDPBUSD Y3, Y7, Y9
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

	ADDQ $32, SI
	DECQ R10
	JNZ  vnnichunk

	DECQ CX
	JNZ  vnnisblock

vnnidone:
	VZEROUPPER
	RET
