//go:build amd64

package model

import (
	"encoding/binary"
	"math"
)

// q4kRowDotF32FMA computes one Q4_K weight row's affine-split dot against x:
//
//	Σ_b Σ_{s=0..7} ( ds[b*8+s]·(Σ_i nib_i·x_i) - ms[b*8+s]·xsum[b*8+s] )
//
// ds/ms are the per-sub-block d·sc and min·m for this row (precomputed in Go, same unpack the exact
// arch kernel does); xsum is the per-sub-block Σx shared across all rows of the matmul. The inner
// per-weight body is a single VFMADD (nibble-as-f32 × x); the d-scale folds in per sub-block via one
// more FMA and the min term is 8 scalar corrections per super-block. AVX2, no bit-exactness contract.
//
//go:noescape
func q4kRowDotF32FMA(row *byte, x *float32, nblk int, ds *float32, ms *float32, xsum *float32) float32

// q4kMatRowsRangeAffine computes y[lo:hi] via the affine-split FMA kernel when the resolved tier has
// AVX2, returning true when it handled the range (else the caller runs the exact path). ds/ms scratch
// is reused across the range exactly as q4kMatRowsRangeArch does; xsum is the caller's shared
// per-matmul activation reduction.
func q4kMatRowsRangeAffine(qt *q4kTensor, x, y []float32, lo, hi int, xsum []float32) bool {
	if qtier < tierAVX2 || qt.nblk == 0 {
		return false
	}
	nblk := qt.nblk
	ds := make([]float32, nblk*8)
	ms := make([]float32, nblk*8)
	rowBytes := qt.q4kRowBytes()
	for o := lo; o < hi; o++ {
		row := qt.raw[o*rowBytes : o*rowBytes+rowBytes]
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
		y[o] = q4kRowDotF32FMA(&row[0], &x[0], nblk, &ds[0], &ms[0], &xsum[0])
	}
	return true
}
