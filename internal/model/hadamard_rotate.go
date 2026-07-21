package model

// hadamard_rotate.go — a pure, deterministic fast Walsh-Hadamard transform used as an
// outlier-smoothing rotation for low-bit K/V quant. Multiplying a K or V row by the
// normalized Walsh-Hadamard matrix before quant spreads a few large outlier channels
// evenly across every dimension, so an aggressive (q4) quant step keeps far less error;
// the same transform applied after dequant restores the original row.
//
// The normalized matrix is involutory (applying it twice is the identity), so a single
// function serves as both the forward rotation and its inverse. That involution is also
// what lets a RoPE K-shift still compose: dequant, un-rotate, rope, re-rotate, re-quant.
//
// This file is the pure math core only — no device, no wall clock, no K/V store wiring.
// It operates on a plain float32 row whose length is a power of two.

import (
	"fmt"
	"math"
)

// ErrNotPowerOfTwo is returned by the transform when the row length is not a power of
// two. The fast butterfly requires a power-of-two length; callers whose head dim is a
// multiple of 64 but not a power of two must split the row into power-of-two spans (the
// common real head dims 64, 128, 256, 512 satisfy both). N carries the offending length.
type ErrNotPowerOfTwo struct {
	N int
}

func (e ErrNotPowerOfTwo) Error() string {
	return fmt.Sprintf("model: hadamard transform needs a power-of-two length, got %d", e.N)
}

// isPowerOfTwo reports whether n is a positive power of two (1, 2, 4, 8, ...).
func isPowerOfTwo(n int) bool {
	return n > 0 && n&(n-1) == 0
}

// walshHadamardForward applies the unnormalized in-place fast Walsh-Hadamard transform
// (the natural-ordered butterfly). Applying it twice scales the row by n, which is why
// the exported transform folds in a 1/sqrt(n) factor to become orthogonal and involutory.
func walshHadamardForward(v []float32) {
	n := len(v)
	for span := 1; span < n; span *= 2 {
		for base := 0; base < n; base += span * 2 {
			for j := base; j < base+span; j++ {
				a := v[j]
				b := v[j+span]
				v[j] = a + b
				v[j+span] = a - b
			}
		}
	}
}

// HadamardTransform rotates vec in place by the normalized Walsh-Hadamard matrix. The
// normalization (1/sqrt(n)) makes the matrix orthogonal and self-inverse, so this one
// function is both the forward rotation and the inverse un-rotation: transforming twice
// returns the original row (within float32 rounding). It returns ErrNotPowerOfTwo when
// len(vec) is not a power of two and leaves vec untouched in that case.
func HadamardTransform(vec []float32) error {
	n := len(vec)
	if !isPowerOfTwo(n) {
		return ErrNotPowerOfTwo{N: n}
	}
	if n == 1 {
		// Normalized H_1 is [1]: an identity, nothing to spread.
		return nil
	}
	walshHadamardForward(vec)
	scale := float32(1.0 / math.Sqrt(float64(n)))
	for i := range vec {
		vec[i] *= scale
	}
	return nil
}

// HadamardRotate is the outlier-smoothing forward rotation applied before quant. It is
// an alias for HadamardTransform, named for the pre-quant side of the round trip.
func HadamardRotate(vec []float32) error { return HadamardTransform(vec) }

// HadamardInverse is the un-rotation applied after dequant to restore the original row.
// Because the normalized transform is involutory, it is the same operation as the
// forward rotation; the distinct name documents intent at the two call sites.
func HadamardInverse(vec []float32) error { return HadamardTransform(vec) }

// MaxAbs returns the largest absolute value in vec, the crude peak that low-bit quant
// must span. An outlier-heavy row has a MaxAbs far above the bulk; the rotation lowers
// it by spreading that peak across dimensions. An empty row returns 0.
func MaxAbs(vec []float32) float32 {
	var peak float32
	for _, x := range vec {
		if a := float32(math.Abs(float64(x))); a > peak {
			peak = a
		}
	}
	return peak
}

// OutlierRatio measures how concentrated a row's magnitude is: peak absolute value over
// mean absolute value. A single large spike gives a high ratio; a row whose energy is
// spread evenly gives a ratio near 1. The rotation is meant to push this ratio down, so
// the same bit budget covers the row with less error. An all-zero or empty row returns 0.
func OutlierRatio(vec []float32) float64 {
	if len(vec) == 0 {
		return 0
	}
	var peak, sum float64
	for _, x := range vec {
		a := math.Abs(float64(x))
		sum += a
		if a > peak {
			peak = a
		}
	}
	mean := sum / float64(len(vec))
	if mean == 0 {
		return 0
	}
	return peak / mean
}
