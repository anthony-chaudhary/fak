// Package mathx holds small numeric helpers shared across packages — the kind of
// one-liner that was copy-pasted into every report builder before it had a home.
//
// Keep it to bit-exact, dependency-free primitives. A helper belongs here only when
// two or more packages need the EXACT same behavior; a variant that rounds or clamps
// differently stays local rather than threading a mode flag through a shared copy.
package mathx

import "math"

// Round3 rounds v to three decimal places using round-half-away-from-zero
// (math.Round semantics). It is the canonical rounding for report numbers that
// should read cleanly without leaking float noise.
func Round3(v float64) float64 {
	return math.Round(v*1000) / 1000
}

// Ratio divides num by den, guarding the zero-denominator case the way report
// builders want it: a positive numerator over zero is +Inf (an unbounded
// multiplier), and 0/0 is 0 (no signal), never a NaN or a panic. It is the
// exact copy that was pasted as a local `ratio`/`safeRatio` into the vcache
// report leaves; a variant that returns 1 for 0/0 (or omits the +Inf branch)
// is a DIFFERENT behavior and stays local rather than sharing this one.
func Ratio(num, den float64) float64 {
	if den == 0 {
		if num > 0 {
			return math.Inf(1)
		}
		return 0
	}
	return num / den
}

// ArgmaxF32 returns the index of the largest element of v — the canonical
// greedy-decode / logits pick that was copy-pasted as a local argmax into every
// bench and diagnostic binary. The first maximum wins on ties (lowest index),
// and an empty slice returns 0 rather than panicking so callers can pass a raw
// logits row without a length guard.
func ArgmaxF32(v []float32) int {
	if len(v) == 0 {
		return 0
	}
	bi, best := 0, v[0]
	for i, x := range v {
		if x > best {
			best, bi = x, i
		}
	}
	return bi
}

// MaxInt returns the larger integer. Equal inputs return that shared value.
func MaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// FDot is a deterministic eight-accumulator float32 inner product. The fixed
// combine order is load-bearing for bit-identical model/compute parity.
func FDot(r, x []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	n := len(r)
	i := 0
	for ; i+8 <= n; i += 8 {
		s0 += r[i] * x[i]
		s1 += r[i+1] * x[i+1]
		s2 += r[i+2] * x[i+2]
		s3 += r[i+3] * x[i+3]
		s4 += r[i+4] * x[i+4]
		s5 += r[i+5] * x[i+5]
		s6 += r[i+6] * x[i+6]
		s7 += r[i+7] * x[i+7]
	}
	s := ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
	for ; i < n; i++ {
		s += r[i] * x[i]
	}
	return s
}
