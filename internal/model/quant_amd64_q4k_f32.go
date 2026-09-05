//go:build amd64

package model

import (
	"encoding/binary"
	"math"
)

// quant_amd64_q4k_f32.go — amd64 dispatch for the AVX2 f32 EXACT resident-Q4_K decode GEMV. This is
// the SIMD twin of the scalar q4kMatRowsRange (quant_q4k.go), the da33 (AVX2-only EPYC, no AVX-512)
// decode lever: the FAK_Q4K path streams ~1.8x fewer bytes than lean-Q8 but its default non-int8
// EXACT kernel was pure scalar, so the byte savings were squandered on compute. The kernel is
// BIT-IDENTICAL to the scalar reference (no activation quantization, so unlike FAK_KQ_INT8 it needs
// no quality witness); the float scale/min unpack stays in Go (per row) so the asm is scale-free.

//go:noescape
func q4kRowDotF32AVX2(row *byte, x *float32, nblk int, ds *float32, ms *float32) float32

// q4kMatRowsRangeArch computes y[lo:hi] via the AVX2 f32 kernel when the resolved tier has AVX2,
// returning true when it handled the range. It returns false (so the caller runs the scalar
// q4kMatRowsRange) on a scalar-only box or when FAK_QKERNEL pins the tier below AVX2. Each output
// row's per-sub-block d*sc / min*m are precomputed here (the 6-bit GetScaleMinK4 unpack + f16 decode)
// into ds/ms scratch reused across the range, exactly the values q4kDequantSuperBlock forms inline.
func q4kMatRowsRangeArch(qt *q4kTensor, x, y []float32, lo, hi int) bool {
	return q4kMatRowsRangeArchRaw(qt.raw, qt, x, y, lo, hi)
}

func q4kMatRowsRangeArchRaw(raw []byte, qt *q4kTensor, x, y []float32, lo, hi int) bool {
	if len(raw) == 0 {
		raw = qt.raw
	}
	if qtier < tierAVX2 || qt.nblk == 0 {
		return false
	}
	nblk := qt.nblk
	ds := make([]float32, nblk*8)
	ms := make([]float32, nblk*8)
	rowBytes := qt.q4kRowBytes()
	for o := lo; o < hi; o++ {
		row := raw[o*rowBytes : o*rowBytes+rowBytes]
		for b := 0; b < nblk; b++ {
			blk := row[b*q4kBlockBytes:]
			d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[0:])))
			mn := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[2:])))
			scales := blk[4 : 4+12]
			base := b * 8
			for s := 0; s < 8; s++ {
				sc, m := GetScaleMinK4(s, scales)
				ds[base+s] = d * float32(sc)
				ms[base+s] = mn * float32(m)
			}
		}
		y[o] = q4kRowDotF32AVX2(&row[0], &x[0], nblk, &ds[0], &ms[0])
	}
	return true
}
