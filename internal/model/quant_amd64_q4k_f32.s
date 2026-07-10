//go:build amd64

#include "textflag.h"

// q4kf32c<> — kernel constant: the 0x0F low-nibble mask broadcast to 32 bytes.
DATA q4kf32c<>+0x00(SB)/4, $0x0F0F0F0F
GLOBL q4kf32c<>(SB), RODATA|NOPTR, $4

// quant_amd64_q4k_f32.s — the AVX2 f32 EXACT decode kernel for resident Q4_K, the SIMD sibling
// of the scalar q4kMatRowsRange (quant_q4k.go). It computes, for ONE weight row (nblk super-blocks)
// and an f32 activation x, the dot y = Σ_i (d_s*nibble_i - m_s)*x_i — bit-for-bit identical to the
// scalar reference, so it stays a drop-in for the golden Q4_K f32 GEMV that the HAL / Metal / int8
// parity gates are all anchored to. Unlike the int8 SDOT path there is NO activation quantization,
// so no quality witness is needed — only max|Δ|==0 vs the scalar (TestQ4KRowDotF32AVX2MatchesScalar).
//
// Bit-exactness discipline. The scalar reference reduces each super-block with FOUR f32 accumulators
// at stride 4 — s0..s3, where lane k accumulates elements i≡k (mod 4) in increasing i order — then
// folds (s0+s1)+(s2+s3) into a scalar acc PER super-block (acc reset each block). Every product is a
// separate multiply then add (NO fused-multiply-add). This kernel reproduces that exactly:
//   * a 4-lane XMM accumulator X7 holds s0..s3, zeroed per super-block;
//   * each 8-wide group's products are folded LOW half then HIGH half into X7 (lanes 0..3), which is
//     the scalar's l%4 lane order because 32 (sub-block width) and 8 (group) are multiples of 4;
//   * w = d*nib - m is VMULPS then VSUBPS (two roundings, matching d1*nib - m1); the product is a
//     separate VMULPS; the lane accumulate is VADDPS — no VFMADD anywhere.
// IEEE-754 round-to-nearest f32 on VMULPS/VSUBPS/VADDPS equals Go's scalar float32 ops, and
// uint8(0..15)->f32 via VPMOVZXBD+VCVTDQ2PS is exact, so the row dot is bit-identical.
//
// The 6-bit scale/min unpack (GetScaleMinK4) + f16 d/min decode is done ONCE PER ROW in Go into
// ds[nblk*8]/ms[nblk*8] (already d*sc and min*m), so this kernel is scale-free: it just unpacks
// nibbles, applies the per-sub-block d/m broadcast, and dots. Sub-block layout matches
// q4kDequantSuperBlock / quant_amd64_q4k.s: 16 B header then a 128 B q field of 4 chunks of 32 bytes;
// chunk k encodes sub-block 2k (LOW nibble of each byte) and 2k+1 (HIGH nibble).
//
// Registers: SI=q-field ptr, R8=x ptr (f32), CX=super-blocks left, R9=ds ptr, R11=ms ptr,
// R10=chunk counter. Y6=0x0F mask, X7=per-super-block 4-lane acc, X12=scalar row acc.
// Y13=d broadcast, Y14=m broadcast. Y0..Y5,Y3,Y5 + X10,X11,X15 scratch.

// PROCGROUP folds 8 nibble bytes (in the low 64 bits of the XMM arg) into X7:
//   nib(f32) = cvt(zext(bytes)); w = d*nib - m; p = w*x; X7 += p.lo; X7 += p.hi; x += 32.
#define PROCGROUP(XB) \
	VPMOVZXBD XB, Y3         \
	VCVTDQ2PS Y3, Y3         \
	VMULPS    Y13, Y3, Y3    \
	VSUBPS    Y14, Y3, Y3    \
	VMOVUPS   (R8), Y5       \
	VMULPS    Y5, Y3, Y3     \
	VEXTRACTF128 $1, Y3, X15 \
	VADDPS    X3, X7, X7      \
	VADDPS    X15, X7, X7     \
	ADDQ      $32, R8

// PROCSUB processes one 32-nibble sub-block held in YNIB/XNIB with the ds/ms at R9/R11.
#define PROCSUB(YNIB, XNIB) \
	VBROADCASTSS (R9), Y13   \
	VBROADCASTSS (R11), Y14  \
	PROCGROUP(XNIB)          \
	VPSRLDQ   $8, XNIB, X10  \
	PROCGROUP(X10)           \
	VEXTRACTI128 $1, YNIB, X11 \
	PROCGROUP(X11)           \
	VPSRLDQ   $8, X11, X10   \
	PROCGROUP(X10)           \
	ADDQ      $4, R9         \
	ADDQ      $4, R11

// func q4kRowDotF32AVX2(row *byte, x *float32, nblk int, ds *float32, ms *float32) float32
TEXT ·q4kRowDotF32AVX2(SB), NOSPLIT, $0-44
	MOVQ row+0(FP), SI
	MOVQ x+8(FP), R8
	MOVQ nblk+16(FP), CX
	MOVQ ds+24(FP), R9
	MOVQ ms+32(FP), R11

	VXORPS X12, X12, X12        // scalar row acc = 0

	TESTQ CX, CX
	JLE   done

	VPBROADCASTD q4kf32c<>+0x00(SB), Y6   // 0x0F low-nibble mask

sblock:
	ADDQ  $16, SI              // skip d/min/scales header -> q field
	VXORPS X7, X7, X7          // per-super-block 4-lane acc (s0..s3) = 0
	MOVQ  $4, R10              // 4 chunks of 32 q-bytes

chunk:
	VMOVDQU (SI), Y0
	VPAND   Y6, Y0, Y1         // low nibbles  (sub-block 2k)
	VPSRLW  $4, Y0, Y2
	VPAND   Y6, Y2, Y2         // high nibbles (sub-block 2k+1)

	PROCSUB(Y1, X1)
	PROCSUB(Y2, X2)

	ADDQ  $32, SI             // next chunk's 32 q-bytes
	DECQ  R10
	JNZ   chunk

	// fold (s0+s1)+(s2+s3) into the scalar row acc, per super-block.
	VPSHUFD $0x55, X7, X8      // s1
	VADDSS  X8, X7, X8         // s0+s1
	VPSHUFD $0xAA, X7, X9      // s2
	VPSHUFD $0xFF, X7, X10     // s3
	VADDSS  X10, X9, X9        // s2+s3
	VADDSS  X9, X8, X8         // (s0+s1)+(s2+s3)
	VADDSS  X8, X12, X12       // acc += block sum

	DECQ  CX
	JNZ   sblock

done:
	VMOVSS X12, ret+40(FP)
	VZEROUPPER
	RET
