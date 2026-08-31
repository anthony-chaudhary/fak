package model

// quant_q2_resident.go — the GGUF-direct residency form of the ternary int2 path
// (issue #4870, T2 of the ternary spine). T1 gave GGUF a ternary container —
// ggufload.TensorQ2_0, group-128 blocks of 34 bytes (one f16 scale d + 128 signed
// 2-bit codes, dequant (c-1)*d, effective code set {-1,0,+1}·d) — plus a reference
// dequant-to-f32. This file feeds those bytes DIRECTLY into the resident ternary
// GEMV: no load-time re-quantize (the prototype's f32→quantizeQ2 round trip would
// re-derive scales and shift the code offset) and no dequant-to-f32 residency
// (512 B of f32 per 34-byte block — a ~15× larger footprint).
//
// Geometry reconciliation (the T1↔prototype seam this file closes): the
// quantize-at-load prototype in quant_q2.go is group-32 with one f32 scale and code
// offset -2; the GGUF container is group-128 with one f16 scale and code offset -1.
// Rather than converting one layout into the other at load (a re-quant in disguise),
// q2Tensor carries BOTH variants: raw != nil selects this g128 GGUF-resident form and
// q2MatRowsRange routes here; raw == nil keeps the proven g32 prototype byte-for-byte
// untouched.
//
// Correctness discipline. dequantQ2G128Block is ggufload.dequantQ2_0Scalar factored
// to one block, so the resident dequant is arithmetically identical to the T1 f32
// reference path the loader would otherwise have produced — pinned by
// TestQ2ResidentMatchesDequant (resident GEMV vs T1 dequant-then-dot + the packed
// footprint). The forward consumes it on the CPU seam only (residentMatRows →
// q2MatRows); device paths are out of scope for this slice.

import (
	"encoding/binary"
	"math"
)

const (
	// qBlk2G128 is the GGUF Q2_0 group size: 128 weights share one f16 scale.
	qBlk2G128 = 128
	// q2G128BlockBytes mirrors ggufload.blockQ2_0Bytes: 2 (d f16) + 128 2-bit codes
	// packed four-per-byte (32 B) = 34 bytes per block, 0.266 B per resident weight.
	q2G128BlockBytes = 2 + qBlk2G128/4
)

// wrapQ2G128FromRaw wraps a raw GGUF Q2_0 payload as a resident g128 q2Tensor with no
// transform: the resident bytes ARE the GGUF bytes (the q4kTensor discipline). in must
// be a multiple of 128; both guards fail loudly like quantizeQ4KFromRaw's.
func wrapQ2G128FromRaw(raw []byte, out, in int) *q2Tensor {
	if in%qBlk2G128 != 0 {
		panic("model: Q2_0 reduction dim not a multiple of 128")
	}
	nblk := in / qBlk2G128
	if want := out * nblk * q2G128BlockBytes; len(raw) != want {
		panic("model: Q2_0 payload size mismatch")
	}
	return &q2Tensor{out: out, in: in, nblk: nblk, raw: raw}
}

// dequantQ2G128Block writes the 128 weights of one 34-byte GGUF Q2_0 block into dst
// (len >= 128). It is ggufload.dequantQ2_0Scalar factored to one block — f16 scale d,
// codes four-per-byte low-first, y = (c-1)*d — so the resident dequant is
// arithmetically identical to the loader's T1 f32 reference path.
func dequantQ2G128Block(dst []float32, blk []byte) {
	d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[0:])))
	qs := blk[2:q2G128BlockBytes]
	for i := 0; i < qBlk2G128/4; i++ {
		b := qs[i]
		dst[4*i+0] = d * float32(int(b&0x3)-1)
		dst[4*i+1] = d * float32(int((b>>2)&0x3)-1)
		dst[4*i+2] = d * float32(int((b>>4)&0x3)-1)
		dst[4*i+3] = d * float32(int((b>>6)&0x3)-1)
	}
}

// q2G128MatRowsRange computes y[lo:hi] over the g128 GGUF-resident variant: each
// 34-byte block dequants into a tiny L1-resident scratch, then dots against the
// matching 128-wide slice of x. Four independent accumulators per block combined in
// fixed order, summed across blocks in row order — deterministic, like
// q4kMatRowsRange, so a future SIMD/ternary-specialized kernel can be held to it.
// Reached via q2MatRowsRange's variant branch (the q2MatRows / residentMatRows seam).
func q2G128MatRowsRange(qt *q2Tensor, x, y []float32, lo, hi int) {
	blk := make([]float32, qBlk2G128)
	rowBytes := qt.nblk * q2G128BlockBytes
	for o := lo; o < hi; o++ {
		row := qt.raw[o*rowBytes:]
		var acc float32
		for b := 0; b < qt.nblk; b++ {
			dequantQ2G128Block(blk, row[b*q2G128BlockBytes:(b+1)*q2G128BlockBytes])
			xs := x[b*qBlk2G128:]
			var s0, s1, s2, s3 float32
			for i := 0; i < qBlk2G128; i += 4 {
				s0 += blk[i] * xs[i]
				s1 += blk[i+1] * xs[i+1]
				s2 += blk[i+2] * xs[i+2]
				s3 += blk[i+3] * xs[i+3]
			}
			acc += (s0 + s1) + (s2 + s3)
		}
		y[o] = acc
	}
}

// dequantQ2G128Tensor reconstructs the full [out,in] f32 matrix from the resident GGUF
// blocks — the g128 arm of dequantQ2Tensor, used by witnesses and any consumer that
// wants the dense weights back.
func dequantQ2G128Tensor(qt *q2Tensor) []float32 {
	w := make([]float32, qt.out*qt.in)
	rowBytes := qt.nblk * q2G128BlockBytes
	parFor(qt.out, currentWorkerCount(), func(lo, hi int) {
		for o := lo; o < hi; o++ {
			row := qt.raw[o*rowBytes:]
			for b := 0; b < qt.nblk; b++ {
				dequantQ2G128Block(w[o*qt.in+b*qBlk2G128:], row[b*q2G128BlockBytes:(b+1)*q2G128BlockBytes])
			}
		}
	})
	return w
}

// AddResidentQ2 stores a raw GGUF Q2_0 payload as a resident g128 q2Tensor under the
// canonical name resolved through the qwen35 source chain, skipping any f32 round
// trip. shape is the model [out, in] convention (in a multiple of 128). Idempotent for
// non-eligible names (returns nil without storing) so the loader can call it
// unconditionally on Q2_0 tensors — the exact AddResidentQ4K contract.
func (b *QuantBuilder) AddResidentQ2(canon string, shape []int, raw []byte) error {
	return b.addResidentQuant(canon, shape, func(name string) {
		if b.m.q2w == nil {
			b.m.q2w = map[string]*q2Tensor{}
		}
		b.m.q2w[name] = wrapQ2G128FromRaw(raw, shape[0], shape[1])
	})
}

// Q2Count returns how many tensors hold a resident ternary Q2_0 copy (loader diagnostic,
// the Q4KCount/KQuantCount twin for the ternary store).
func (m *Model) Q2Count() int { return len(m.q2w) }
