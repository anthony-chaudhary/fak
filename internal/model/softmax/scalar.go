// Package softmax contains a dependency-light, numerically-stable scalar
// softmax reference shared by the attention and router paths.
//
// The arithmetic is deliberately serial: the max subtraction, the float32
// exponent conversion, the left-to-right sum and the final division are
// load-bearing for the f32 bit-exact rungs, so this reference must NOT be
// reassociated or replaced by an online/fused/vectorized reduction. Model
// adapters own their callers; scheduling, device selection and residency stay
// with their existing owners.
package softmax

import "math"

// InPlace overwrites s with the max-subtracted softmax of its values.
//
// The reduction order (max scan, then per-element exp/sum accumulation, then a
// final normalize pass) and the float32 accumulation are the contract: every
// consumer that migrates here must observe bit-identical results to the
// pre-extraction helper. s must be nonempty; an empty slice panics on the max
// scan, matching the original in-place helper.
func InPlace(s []float32) {
	mx := s[0]
	for _, v := range s {
		if v > mx {
			mx = v
		}
	}
	var sum float32
	for i, v := range s {
		e := float32(math.Exp(float64(v - mx)))
		s[i] = e
		sum += e
	}
	for i := range s {
		s[i] /= sum
	}
}

// Copy returns the max-subtracted softmax of z in a single fresh allocation,
// leaving z unmutated. It allocates once and delegates the arithmetic to
// InPlace, so the in-place and allocating contracts cannot drift.
//
// z must be nonempty; an empty slice panics, matching the original allocating
// helper.
func Copy(z []float32) []float32 {
	out := make([]float32, len(z))
	copy(out, z)
	InPlace(out)
	return out
}
