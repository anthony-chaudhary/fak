// round.go ports aiperf's fraction-aware summary rounding to Go.
//
// NOTICE: ported from ai-dynamo/aiperf, src/aiperf/kubernetes/crd_models.py
// (_round_summary, _FRACTION_KEYS), pinned at commit
// ab610add4e8876fca223197596c728e17365f381 (ai-dynamo/aiperf@ab610ad),
// Apache-2.0 licensed. The source rounds every float in a nested summary to
// 2 decimals for display, except keys naming 0..1 fractions, which keep 9
// decimals so a small error rate is neither erased to 0 nor inflated. This
// port keeps that rule, half-to-even ties and non-finite passthrough.
package perfledger

import (
	"math"
	"strconv"
)

const (
	// SummaryDisplayDigits is the decimal precision for human-scale values.
	SummaryDisplayDigits = 2
	// SummaryFractionDigits is the decimal precision for 0..1 fraction keys.
	SummaryFractionDigits = 9
)

// DefaultSummaryFractionKeys returns a fresh copy of the upstream fraction
// key set, so callers may extend it without sharing state.
func DefaultSummaryFractionKeys() map[string]bool {
	return map[string]bool{"error_rate": true}
}

// RoundSummaryValue rounds v to ndigits decimal places with half-to-even ties
// on the exact binary value. NaN and ±Inf pass through unchanged; a negative
// ndigits rounds to tens, hundreds, and so on.
func RoundSummaryValue(v float64, ndigits int) float64 {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return v
	}
	if ndigits < 0 {
		scale := math.Pow10(-ndigits)
		if math.IsInf(scale, 0) {
			return math.Copysign(0, v)
		}
		return math.RoundToEven(v/scale) * scale
	}
	// Decimal formatting rounds the exact binary expansion, so 2.675 (stored
	// just below) becomes 2.67 as in Python, where v*100 would drift.
	out, err := strconv.ParseFloat(strconv.FormatFloat(v, 'f', ndigits, 64), 64)
	if err != nil {
		return v
	}
	return out
}

// RoundSummaryMetric rounds one named summary scalar: fraction keys keep
// SummaryFractionDigits, every other key gets SummaryDisplayDigits.
func RoundSummaryMetric(key string, v float64, fractionKeys map[string]bool) float64 {
	return RoundSummaryValue(v, rsumDigitsFor(key, SummaryDisplayDigits, fractionKeys))
}

// RoundSummaryFloats rounds a flat metric map into a new map; the input is
// not modified.
func RoundSummaryFloats(m map[string]float64, fractionKeys map[string]bool) map[string]float64 {
	if m == nil {
		return nil
	}
	out := make(map[string]float64, len(m))
	for k, v := range m {
		out[k] = RoundSummaryMetric(k, v, fractionKeys)
	}
	return out
}

// RoundSummaryTree recursively rounds float64 and float32 leaves of a
// decoded JSON-like summary (map[string]any, []any, scalars) into a new tree.
// A fraction key raises precision for its whole subtree; non-float leaves
// are returned as is.
func RoundSummaryTree(obj any, fractionKeys map[string]bool) any {
	return rsumWalk(obj, SummaryDisplayDigits, fractionKeys)
}

func rsumDigitsFor(key string, inherited int, fractionKeys map[string]bool) int {
	if fractionKeys[key] {
		return SummaryFractionDigits
	}
	return inherited
}

func rsumWalk(obj any, ndigits int, fractionKeys map[string]bool) any {
	switch v := obj.(type) {
	case float64:
		return RoundSummaryValue(v, ndigits)
	case float32:
		return float32(RoundSummaryValue(float64(v), ndigits))
	case map[string]any:
		if v == nil {
			return v
		}
		out := make(map[string]any, len(v))
		for k, child := range v {
			out[k] = rsumWalk(child, rsumDigitsFor(k, ndigits, fractionKeys), fractionKeys)
		}
		return out
	case []any:
		if v == nil {
			return v
		}
		out := make([]any, len(v))
		for i, child := range v {
			out[i] = rsumWalk(child, ndigits, fractionKeys)
		}
		return out
	default:
		return obj
	}
}
