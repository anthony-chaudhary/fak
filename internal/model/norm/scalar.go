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

// SerialRMSNormInto is copy(dst, SerialRMSNorm(x, w, eps)) without the
// temporary allocation when the written destination is disjoint from x and w.
// Short destinations receive a prefix; longer destinations keep their tail.
// Overlap conservatively uses the original allocating path, preserving copy
// semantics even for shifted overlap. A short weight slice panics before any
// destination write, as evaluation of the original copy source did.
func SerialRMSNormInto(dst, x, w []float32, eps float32) {
	if len(w) < len(x) {
		_ = w[len(x)-1]
	}
	written := dst[:min(len(dst), len(x))]
	if rmsSlicesOverlap(written, x) || rmsSlicesOverlap(written, w[:len(x)]) {
		copy(dst, SerialRMSNorm(x, w, eps))
		return
	}
	SerialRMSNormIntoDisjoint(dst, x, w, eps)
}

// SerialRMSNormIntoDisjoint is copy(dst, SerialRMSNorm(x, w, eps)) for
// callers that prove the written destination shares no elements with x or w.
// This contract is unchecked: use SerialRMSNormInto for arbitrary slices.
// x and w may overlap each other. Short destinations receive a prefix; longer
// destinations keep their tail. Short weights panic before any destination
// write. Arithmetic and rounding are identical to the serial reference.
func SerialRMSNormIntoDisjoint(dst, x, w []float32, eps float32) {
	if len(w) < len(x) {
		_ = w[len(x)-1]
	}
	written := dst[:min(len(dst), len(x))]
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	for i := range written {
		written[i] = x[i] * inv * w[i]
	}
}

// Two contiguous slices overlap iff either first element lies in the other.
// Pointer equality avoids unsafe address arithmetic and allocates no storage.
func rmsSlicesOverlap(a, b []float32) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	for i := range b {
		if &a[0] == &b[i] {
			return true
		}
	}
	for i := range a {
		if &b[0] == &a[i] {
			return true
		}
	}
	return false
}
