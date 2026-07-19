//go:build amd64

#include "textflag.h"

// q4kdc<> — kernel constant: the 0x0F low-nibble mask broadcast to 32 bytes.
DATA q4kdc<>+0x00(SB)/4, $0x0F0F0F0F
GLOBL q4kdc<>(SB), RODATA|NOPTR, $4

// quant_amd64_q4k_decode.s — the AVX2 affine-split FMA decode kernel for resident Q4_K (see
// quant_q4k_decode.go). It computes, for one weight row, y = Σ_s [ ds_s·(Σ_i nib_i·x_i) - ms_s·xsum_s ]
// over all sub-blocks s. Unlike the exact kernel (quant_amd64_q4k_f32.s) it does NOT reproduce the
// scalar reference's rounding: the d-scale and min-subtract are hoisted out of the per-weight loop,
// leaving a single FMA per weight, and the reduction runs full-width (8-lane YMM accumulate + one
// hsum per super-block) with FMA. It is a decode-only path held to a cosine/argmax gate.
//
// Layout matches q4kDequantSuperBlock / quant_amd64_q4k.s: each super-block is a 16 B header then a
// 128 B q field of 4 chunks of 32 bytes; chunk k holds sub-block 2k in the LOW nibble of each byte
// and sub-block 2k+1 in the HIGH nibble. Sub-block s dots against x[s*32:s*32+32]; ds/ms/xsum are
// indexed [b*8+s].
//
// Registers: SI=q ptr, R8=x ptr, CX=super-blocks left, R9=ds, R11=ms, R13=xsum, R10=chunk counter.
// Y6=0x0F mask. Y7=per-sub-block dot acc (zeroed each sub-block). Y8=per-super-block Σ ds_s·dot_s
// (its low half X8 is reused by the hsum). X12=scalar row acc. X14=per-super-block Σ ms_s·xsum_s.
// Y13=ds broadcast. Y3,Y5 + X10,X11,X15 scratch.

// FMAGROUP folds 8 nibble bytes (low 64 bits of XB) into Y7: Y7 += cvt(zext(bytes)) * x; x += 32.
#define FMAGROUP(XB) \
	VPMOVZXBD   XB, Y3       \
	VCVTDQ2PS   Y3, Y3       \
	VMOVUPS     (R8), Y5     \
	VFMADD231PS Y5, Y3, Y7   \
	ADDQ        $32, R8

// SUBBLOCK dots one 32-nibble sub-block (held in YNIB/XNIB) into Y7, then folds ds_s·Y7 into Y8 and
// ms_s·xsum_s into X14, advancing the ds/ms/xsum cursors by one f32.
#define SUBBLOCK(YNIB, XNIB) \
	VXORPS       Y7, Y7, Y7    \
	FMAGROUP(XNIB)             \
	VPSRLDQ      $8, XNIB, X10 \
	FMAGROUP(X10)              \
	VEXTRACTI128 $1, YNIB, X11 \
	FMAGROUP(X11)              \
	VPSRLDQ      $8, X11, X10  \
	FMAGROUP(X10)              \
	VBROADCASTSS (R9), Y13     \
	VFMADD231PS  Y13, Y7, Y8   \
	VMOVSS       (R11), X15    \
	VMOVSS       (R13), X10    \
	VMULSS       X10, X15, X15 \
	VADDSS       X15, X14, X14 \
	ADDQ         $4, R9        \
	ADDQ         $4, R11       \
	ADDQ         $4, R13

// func q4kRowDotF32FMA(row *byte, x *float32, nblk int, ds *float32, ms *float32, xsum *float32) float32
TEXT ·q4kRowDotF32FMA(SB), NOSPLIT, $0-52
	MOVQ row+0(FP), SI
	MOVQ x+8(FP), R8
	MOVQ nblk+16(FP), CX
	MOVQ ds+24(FP), R9
	MOVQ ms+32(FP), R11
	MOVQ xsum+40(FP), R13

	VXORPS X12, X12, X12        // scalar row acc = 0

	TESTQ CX, CX
	JLE   done

	VPBROADCASTD q4kdc<>+0x00(SB), Y6   // 0x0F low-nibble mask

sblock:
	ADDQ   $16, SI             // skip d/min/scales header -> q field
	VXORPS Y8, Y8, Y8          // per-super-block Σ ds_s·dot_s = 0
	VXORPS X14, X14, X14       // per-super-block Σ ms_s·xsum_s = 0
	MOVQ   $4, R10             // 4 chunks of 32 q-bytes

chunk:
	VMOVDQU (SI), Y0
	VPAND   Y6, Y0, Y1         // low nibbles  (sub-block 2k)
	VPSRLW  $4, Y0, Y2
	VPAND   Y6, Y2, Y2         // high nibbles (sub-block 2k+1)

	SUBBLOCK(Y1, X1)
	SUBBLOCK(Y2, X2)

	ADDQ $32, SI              // next chunk's 32 q-bytes
	DECQ R10
	JNZ  chunk

	// y_sb = hsum(Y8) - X14 ; acc += y_sb
	VEXTRACTF128 $1, Y8, X11
	VADDPS       X11, X8, X8
	VHADDPS      X8, X8, X8
	VHADDPS      X8, X8, X8
	VADDSS       X8, X12, X12
	VSUBSS       X14, X12, X12

	DECQ CX
	JNZ  sblock

done:
	VMOVSS X12, ret+48(FP)
	VZEROUPPER
	RET
