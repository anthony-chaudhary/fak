package compute

import (
	"encoding/binary"
	"math"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

// vecdot.go — Compute-in-Quant (CIQ) integer dot products for legacy ggml Q4_0 and Q8_0
// weight blocks (vllm.cpp borrow, INSPIRE-ONLY clean-room Go). The scalar reference CPU
// path in internal/model dequantizes each quantized weight block to float32 and dots it
// against an f32 activation, which saturates the memory bus: a Q4_0 block is 18 bytes but
// expands to 128 bytes of f32 before a single multiply. CIQ instead quantizes the
// ACTIVATION to Q8_0 once and reduces an int8×int8 product directly on the packed weight
// bytes, so the weight traffic stays at the packed width (1.125 B/weight for Q8_0, 0.5625
// B/weight for Q4_0) and no transient float32 buffer is allocated.
//
// Block layouts are the ggml legacy formats (see q8_0BlockBytes/q4_0BlockBytes in
// internal/model and ggufload's dequant routines, reproduced here so the kernel carries no
// model dependency):
//
//	Q8_0 block (34 B): d f16 [0:2], 32 int8 codes [2:34].    w[j] = d * q8[j]
//	Q4_0 block (18 B): d f16 [0:2], 16 nibble bytes [2:18].  w[j] = d * (nibble(j) - 8)
//
// The Q4_0 nibble order is the ggml INTERLEAVE: the low nibble of byte j is element j and
// the HIGH nibble of byte j is element j+16, each re-centered by -8. This deliberately
// differs from the re-quantized int4 store's packing; the two must never share a decoder.
//
// Integer reduction and range safety. With x[j] = dx * qx[j] (qx in [-127,127]) and
// w[j] = d * (q - 8):
//
//	dot = Σ w[j]·x[j] = d·dx · Σ (q[j] - 8)·qx[j]
//
// so the whole kernel reduces to ONE integer sum I = Σ q·qx per 32-wide block, scaled by
// the two f16-derived block scales. |I| <= 32·15·127 ≈ 6.1e4 for Q4_0 (nibble in [0,15])
// and <= 32·127·127 ≈ 5.2e5 for Q8_0 — both far inside int32, so the reduction is
// associative and order-free (matching the model's existing Q4_K/Q5_K int8 reducers). The
// activation quantization makes the result APPROX (not bit-exact vs the f32 reference),
// exactly the lane the model's Q8 SDOT paths already ride.
//
// This file is deliberately scalar Go, stdlib-only (no asm/cgo/unsafe), so it is the
// portable floor; an x86 AVX2/VNNI or ARM NEON acceleration of the same fixed-order
// reduction is a private backend, not a fork of this contract.

// VecDotQ8_0Q8_0 returns dot(w, x) for one Q8_0 weight row and a Q8_0-quantized activation,
// both of nblk 32-wide blocks packed in the ggml Q8_0 layout. wRaw is nblk*34 bytes; xq is
// nblk*32 activation codes and xd is the per-block activation scales (len nblk).
func VecDotQ8_0Q8_0(wRaw []byte, nblk int, xq []int8, xd []float32) float32 {
	var acc float32
	for b := 0; b < nblk; b++ {
		blk := wRaw[b*q8_0BlockBytes : (b+1)*q8_0BlockBytes]
		dw := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[0:])))
		var i int32
		code := blk[2:q8_0BlockBytes]
		xb := xq[b*32 : (b+1)*32]
		for j := 0; j < 32; j++ {
			i += int32(int8(code[j])) * int32(xb[j])
		}
		acc += (float32(i) * dw) * xd[b]
	}
	return acc
}

// VecDotQ4_0Q8_0 returns dot(w, x) for one Q4_0 weight row and a Q8_0-quantized activation,
// both nblk 32-wide blocks. wRaw is nblk*18 bytes; xq/xd are the Q8_0 activation codes and
// per-block scales (len nblk*32 and nblk). The Q4 code is re-centered by -8 before the
// integer product, so I = Σ (nibble-8)·qx.
func VecDotQ4_0Q8_0(wRaw []byte, nblk int, xq []int8, xd []float32) float32 {
	var acc float32
	for b := 0; b < nblk; b++ {
		blk := wRaw[b*q4_0BlockBytes : (b+1)*q4_0BlockBytes]
		dw := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[0:])))
		qs := blk[2:q4_0BlockBytes]
		xb := xq[b*32 : (b+1)*32]
		var i int32
		// interleave: low nibble of byte j -> element j, high nibble -> element j+16
		for j := 0; j < 16; j++ {
			i += int32(int(qs[j]&0x0f)-8) * int32(xb[j])
			i += int32(int(qs[j]>>4)-8) * int32(xb[j+16])
		}
		acc += (float32(i) * dw) * xd[b]
	}
	return acc
}

// q8_0BlockBytes and q4_0BlockBytes are the ggml legacy block sizes (34 and 18). Held
// locally so the kernel is self-contained; they are identical to internal/model's
// q8_0BlockBytes / q4_0BlockBytes.
const (
	q8_0BlockBytes = 34
	q4_0BlockBytes = 18
)
