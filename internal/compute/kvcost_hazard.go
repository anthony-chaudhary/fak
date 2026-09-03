package compute

import (
	"math"
)

// KVEvictionCostHazard computes the cost of losing a span using an age-conditioned hazard reuse estimate (#2669/#3411).
// Inherits the Bytes <= 0 fail-open (+Inf) unchanged.
func KVEvictionCostHazard(s KVSpanStats, clock uint64) float64 {
	if s.Bytes <= 0 {
		return math.Inf(1)
	}
	reuseProbability := KVReuseEstimate(s, clock)
	recomputeCost := float64(s.Tokens)
	return recomputeCost * reuseProbability / float64(s.Bytes)
}

// PickEvictionVictimHazard selects the cheapest-to-lose span per KVEvictionCostHazard (#2669/#3411).
func PickEvictionVictimHazard(spans []KVSpanStats, clock uint64) int {
	idx, _ := pickLowestCost(spans, func(s KVSpanStats) float64 {
		return KVEvictionCostHazard(s, clock)
	})
	return idx
}
