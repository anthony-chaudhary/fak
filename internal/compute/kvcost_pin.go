package compute

import "math"

// kvcost_pin.go — the TTL-decaying pin economics + adjudicated agent cache-control hint
// #2673 adds to #2239's cost-of-losing function. Today a pin is a binary hard exclusion
// (PickEvictionVictim skips any span with Pinned==true, full stop), but #2239/#2666 are
// explicit that "pins are leases with TTL, never permanent": a pin whose TTL is nearly
// expired should not hold memory hostage against live demand the way a fresh pin does, and
// an ADJUDICATED agent hint (the fak differentiator — #805 intent conduit, default-deny
// adjudication) should FEED the value function, not blindly override it. This file models a
// pin as a bounded, TTL-decaying cost boost (KVSpanStats.PinBoost) folded additively into
// the cost, and an agent cache-control hint (KVSpanStats.Hint) as a bounded multiplier on
// the reuse term — a strict generalization of the current Pinned semantics (an unexpired
// hard pin is the PinBoost = +Inf limit; no pin, no hint reduces byte-identically to
// KVEvictionCost).
//
// Pure primitive, same no-state discipline as kvcost.go: this package holds no clock, so TTL
// arrives already resolved to remaining-ms from the caller's lease ledger (operator #2211 /
// agent #2225) and PinBoostFromTTL maps it to a boost consistently. Threading real
// radixkv/native span TTLs into PinBoost and surfacing pins in the preempt metrics
// (last_pinned / pinned_skipped_total) is the R2/R3 wiring follow-on #2673 tracks.

// Hint multipliers on the reuse term — bounded (finite, strictly positive) so a hint biases
// the ranking but can never fully defeat the value function: a precious hint on a cold span
// still loses to a much hotter span, and an ephemeral hint on a hot span still beats a much
// colder one. The factors are symmetric reciprocals so precious/ephemeral move keep-value by
// the same bounded ratio in opposite directions.
const (
	hintPreciousMultiplier  = 2.0 // precious raises keep-value (bounded).
	hintEphemeralMultiplier = 0.5 // ephemeral lowers keep-value (bounded).
)

// hintReuseMultiplier maps an agent cache-control hint to its bounded reuse-term multiplier.
// HintNone (and any unknown value) is the identity 1.0, so an unhinted span scores exactly as
// under KVEvictionCost — the reduction #2673's witness proves.
func hintReuseMultiplier(h KVCacheHint) float64 {
	switch h {
	case HintPrecious:
		return hintPreciousMultiplier
	case HintEphemeral:
		return hintEphemeralMultiplier
	default:
		return 1.0
	}
}

// KVEvictionCostPinned is KVEvictionCost extended with #2673's TTL-decaying pin economics and
// bounded agent cache-control hint:
//
//	cost = (recomputeCost × reuseProbability × hint) ÷ bytes  +  PinBoost
//
// read exactly like KVEvictionCost — a HIGHER cost means more expensive to lose per byte, so
// the evictor keeps it — with two additions:
//
//   - PinBoost folds in ADDITIVELY (a decaying pin): a live pin with ample TTL carries a
//     large boost ⇒ effectively unevictable (today's hard-pin behavior); as TTL → 0 the
//     boost decays to 0 ⇒ an expiring pin on a cold span releases gracefully instead of
//     holding memory forever. The absolute-exclusion Pinned bool is the PinBoost = +Inf
//     limit of this term, so an unexpired hard pin still yields the old exclusion.
//   - Hint scales the reuse term by a BOUNDED multiplier (hintReuseMultiplier): precious
//     raises keep-value, ephemeral lowers it, both saturating so a hint can never fully
//     defeat the value function (the adjudication fence — an agent cannot game the cache
//     into starvation).
//
// The correctness floors are preserved absolutely, ahead of any economics:
//
//   - a Leased span (an in-flight request holding refs>0) returns +Inf — it is never a
//     victim regardless of boost/hint, matching radixkv's refs>0 rule. This is correctness,
//     not economics: reclaiming a span being served would corrupt an active decode.
//   - a Pinned span returns +Inf — the hard-pin exclusion, the PinBoost→+Inf limit.
//   - the KVEvictionCost fail-open is inherited: a span with non-positive Bytes scores +Inf
//     so an unknown-footprint span is never preferred as a victim.
//
// Reduction: with PinBoost == 0, Hint == HintNone, and neither Pinned nor Leased,
// KVEvictionCostPinned(s) == KVEvictionCost(s) exactly — a strict generalization, never a
// divergence, on today's inputs.
func KVEvictionCostPinned(s KVSpanStats) float64 {
	if s.Leased {
		return math.Inf(1) // in-flight lease — absolute exclusion (correctness floor).
	}
	if s.Pinned {
		return math.Inf(1) // hard pin — the PinBoost → +Inf limit.
	}
	if s.Bytes <= 0 {
		return math.Inf(1) // fail-open on unknown footprint, same as KVEvictionCost.
	}
	reuseProbability := KVReuseTerm(s) * hintReuseMultiplier(s.Hint) // Unified reuse term seam (#3411)
	recomputeCost := float64(s.Tokens)
	return recomputeCost*reuseProbability/float64(s.Bytes) + s.PinBoost
}

// PinBoostFromTTL maps a pin's remaining TTL to its bounded PinBoost via a monotone linear
// decay: a pin with a full lease remaining carries the maximum boost (maxPinBoost — large
// enough to dominate any realistic per-byte cost-of-losing, so a fresh pin is effectively
// unevictable), and the boost decays linearly to 0 as remainingMs → 0, so an expiring pin on
// a cold span becomes a candidate again. It is the consistent TTL → boost curve the caller's
// lease ledger (operator #2211 / agent #2225) uses so every resident pool derives PinBoost
// the same way; this package holds no clock, so the caller resolves remainingMs itself.
//
// Boundary behavior (all monotone non-decreasing in remainingMs):
//   - leaseMs <= 0 (no lease / not pinned): 0 — no boost.
//   - remainingMs <= 0 (expired): 0 — the pin is spent, span fully evictable.
//   - remainingMs >= leaseMs (fresh or over-clamped): maxPinBoost — full strength.
//   - in between: maxPinBoost × (remainingMs / leaseMs) — linear decay.
func PinBoostFromTTL(remainingMs, leaseMs int64) float64 {
	if leaseMs <= 0 || remainingMs <= 0 {
		return 0
	}
	if remainingMs >= leaseMs {
		return maxPinBoost
	}
	return maxPinBoost * (float64(remainingMs) / float64(leaseMs))
}

// maxPinBoost is the boost a full-TTL pin carries. A per-byte cost-of-losing (Tokens × reuse
// ÷ Bytes) is bounded by span geometry to O(hundreds) in practice — reuse counts are small
// and Bytes grows with Tokens (a span holds at least a few bytes per token), so the ratio
// stays modest. maxPinBoost is set several orders of magnitude above that so a fresh full-TTL
// pin ranks as the most-expensive-to-lose span and is never chosen as a victim while a
// non-pinned candidate exists — reproducing today's hard-pin behavior through the economics
// term rather than a boolean bypass. It is deliberately finite (not +Inf) and only ~1e6, not
// astronomically large, so the LINEAR decay reaches negligible values near expiry: a pin with
// a fraction f of its lease left carries maxPinBoost·f, which falls below a hot span's cost
// once f is tiny, letting the pin release gracefully as TTL → 0. The absolute exclusion (a pin
// that never releases while live) is reserved for the explicit Pinned bool.
const maxPinBoost = 1e6

// PickEvictionVictimPinned is PickEvictionVictim scored by KVEvictionCostPinned instead of
// KVEvictionCost: it returns the index of the LOWEST-cost evictable span, honoring the same
// Pinned/Leased hard exclusions (now redundant with the +Inf floors in KVEvictionCostPinned,
// but kept explicit so the skip is unmistakable and never depends on +Inf comparison), the
// same LastUsed tie-break (oldest first, so a uniform-cost set reduces to LRU), and the same
// -1 no-candidate signal. Unlike the boolean Pinned exclusion, a span held ONLY by a decaying
// PinBoost (no Pinned bool) IS a candidate: once its boost has decayed below a colder span's
// cost it becomes the victim — the graceful release #2673 adds that a hard boolean cannot
// express.
//
// This is the pure R1 decision primitive; wiring it into radixkv.EvictToBudget /
// modelengine.pickPreemptVictim behind the FAK_NATIVE_KV_* flags, with real lease TTLs mapped
// through PinBoostFromTTL, is the R2/R3 cross-lane follow-on #2673 tracks.
func PickEvictionVictimPinned(spans []KVSpanStats) int {
	idx, _ := pickLowestCost(spans, KVEvictionCostPinned)
	return idx
}
