// Package norm contains dependency-light numerical normalization references.
//
// The arithmetic here is small, independent of any model or backend policy, and
// checked with tiny arrays. Model adapters supply weights, epsilon and the
// consumer path; scheduling, device selection and residency stay with their
// existing owners.
package norm

import "math"

// SerialRMSNorm returns x / sqrt(mean(x^2)+eps) * w using the serial in-order
// float32 sum-of-squares and a single fresh output allocation.
//
// The reduction order, epsilon placement, the float32->float64 sqrt conversion
// and the left-to-right multiplication order are load-bearing for the f32
// bit-exact rungs (R2/R14): this reference must NOT be reassociated or swapped
// for a fused/vectorized reduction. The in-place quant twin rmsnormInto in the
// model package is the one that may use a vectorized dot product.
//
// x and w are read-only; none of their bytes are mutated. w must be at least
// len(x) long, matching the caller-validated geometry of the original path. For
// empty x the result is a zero-length slice, as before the extraction.
func SerialRMSNorm(x, w []float32, eps float32) []float32 {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * inv * w[i]
	}
	return out
}
