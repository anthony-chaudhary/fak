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
