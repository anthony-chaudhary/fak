package compute

import "math"

// kvcost_fanout.go — the concurrent-sharer fan-out term #2670 adds to #2239's cost-of-losing
// function. KVEvictionCost scores a span from its own local stats (Tokens, Hits, Bytes), but
// the radix tree (internal/radixkv) knows something the cost function never sees: how many
// live sessions/workers currently share this span as a prefix. A system-prompt span shared by
// 8 concurrent sessions is worth ~8x a leaf touched by one — losing it costs 8 re-prefills,
// not one — yet today it scores identically to a one-shot leaf with the same Hits (Hits counts
// subsequent lookups on one node, not concurrent fan-out). This file adds that term as a pure
// extension of KVSpanStats/KVEvictionCost, the same no-state discipline as kvcost_aging.go and
// kvcost_pin.go.
//
// Fan-out (concurrent breadth — how many depend NOW) and the aging-clock term (temporal
// depth — how likely reuse over time) are orthogonal: both fold cleanly into recompute value
// and can compose (KVEvictionCostFanout does not touch AgeStamp/PinBoost/Hint; a caller
// wanting all of them composes the terms the same way KVEvictionCostPinned composes Hint on
// top of KVEvictionCost's base formula).

// KVEvictionCostFanout is KVEvictionCost with the recompute term weighted by concurrent
// sharer fan-out:
//
//	cost = (Tokens × max(Sharers, 1)) × reuseProbability ÷ bytes
//
// read exactly like KVEvictionCost — a HIGHER cost means more expensive to lose per byte, so
// the evictor keeps it — except a span currently shared by S live sessions has its recompute
// cost scaled by S: losing it re-prefills for every current consumer, not just one. The
// fan-out multiply saturates (saturatingMulInt64) so an unbounded Sharers count can never
// overflow the score into a false tie or a wrapped negative.
//
// Reduction: Sharers <= 1 (the zero value included) makes max(Sharers, 1) == 1, so
// KVEvictionCostFanout(s) == KVEvictionCost(s) exactly for every span with the same
// Tokens/Hits/Bytes — a strict generalization, never a divergence, on today's single-consumer
// inputs. The KVEvictionCost fail-open is inherited unchanged: a span with non-positive Bytes
// scores +Inf so an unknown-footprint span is never preferred as a victim regardless of
// fan-out.
func KVEvictionCostFanout(s KVSpanStats) float64 {
	if s.Bytes <= 0 {
		return math.Inf(1)
	}
	fanout := s.Sharers
	if fanout < 1 {
		fanout = 1
	}
	recomputeValue := saturatingMulInt64(int64(s.Tokens), int64(fanout))
	reuseProbability := KVReuseTerm(s) // Unified reuse term seam (#3411)
	return float64(recomputeValue) * reuseProbability / float64(s.Bytes)
}

// PickEvictionVictimFanout is PickEvictionVictim scored by KVEvictionCostFanout instead of
// KVEvictionCost: same Pinned/Leased hard exclusions, same LastUsed tie-break, same -1
// no-candidate signal. A span with Sharers > 1 competing against an equally-costed
// single-consumer span (same Tokens/Hits/Bytes) is preferentially KEPT — the multi-session
// prefix outranks the one-shot leaf instead of tying with it.
//
// This is the pure R1 decision primitive #2670 names; wiring radixkv.costAwareLeaf to
// populate Sharers from real ref-count/child fan-out (and the replay hit-rate witness that
// proves the aggregate win) is the R2 cross-lane follow-on.
func PickEvictionVictimFanout(spans []KVSpanStats) int {
	idx, _ := pickLowestCost(spans, KVEvictionCostFanout)
	return idx
}
