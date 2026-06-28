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
