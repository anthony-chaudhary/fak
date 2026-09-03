package compute

import (
	"math"
)

// KVReuseEstimator computes an estimated reuse probability for a resident KV span (#3411, epic #2236).
type KVReuseEstimator func(s KVSpanStats) float64

// DefaultKVReuseEstimator is the canonical Laplace-smoothed frequency estimator (Hits + 1).
// Zero-value or basic inputs reduce exactly to Hits + 1.
func DefaultKVReuseEstimator(s KVSpanStats) float64 {
	if s.Hits < 0 {
		return 1.0
	}
	return float64(s.Hits + 1)
}

// KVReuseTerm computes the standard reuse probability term for span s via the unified seam.
func KVReuseTerm(s KVSpanStats) float64 {
	return DefaultKVReuseEstimator(s)
}

// KVReuseTermWithEstimator computes the reuse term using estimator, falling back to
// DefaultKVReuseEstimator if estimator is nil.
func KVReuseTermWithEstimator(s KVSpanStats, estimator KVReuseEstimator) float64 {
	if estimator == nil {
		return DefaultKVReuseEstimator(s)
	}
	return estimator(s)
}

// KVReuseEstimate is the age-conditioned hazard-rate reuse estimate (#2669/#3411).
// It replaces the memoryless Hits+1 count with (Hits+1) * decay(age, meanInterArrival).
// If s has no interval history (IntervalCount <= 0) or age <= meanInterArrival,
// it reduces EXACTLY to float64(Hits + 1).
func KVReuseEstimate(s KVSpanStats, clock uint64) float64 {
	base := float64(s.Hits + 1)
	if s.Hits < 0 {
		base = 1.0
	}
	if s.IntervalCount <= 0 || clock <= s.LastUsed {
		return base
	}
	age := clock - s.LastUsed
	meanIA := float64(s.IntervalSum) / float64(s.IntervalCount)
	if meanIA < 1.0 {
		meanIA = 1.0
	}
	if float64(age) <= meanIA {
		return base
	}
	return base * math.Exp(-(float64(age)-meanIA)/meanIA)
}
