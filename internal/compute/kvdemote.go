package compute

import "sort"

// kvdemote.go — min-cost KV tier ACTION: demote-not-drop (#2671, epic #2236 rows M5 lifecycle
// + M3 hierarchical tiering; concretizes #2666's offload-before-preempt bullet; extends #2239).
//
// PickEvictionVictim (kvcost.go) answers ONE question under budget pressure — "which span is
// cheapest to lose?" — and then DROPS it. But dropping is only the most expensive rung of a
// ladder fak already prices as pure planners: quantize-in-place (kvprecision.go), spill-to-host
// (kvresidency.go), and the #2169 spill-to-disk/object tiers. The crux the binary evictor gets
// wrong: a DEMOTED span's cost-to-lose is not full re-prefill — it is the restore latency from
// the tier it lands on, which is far cheaper. Dropping a span that could have spilled to host
// for a transfer-back restore throws away recompute the ladder would have avoided.
//
// This file generalizes "pick the cheapest victim to DROP" into "pick the cheapest ACTION per
// span": for each resident span and each configured rung, weigh (bytesFreed, restoreCost) and
// choose the action set that clears a byte deficit at least aggregate cost. Drop is just the
// rung whose restore cost is full re-prefill — the top of the ladder — so with no lower rung
// configured the primitive reduces EXACTLY to PickEvictionVictim (proven in kvdemote_test.go).
// Parity frame: Dynamo KVBM's cost-aware lifecycle across HBM/host/SSD/network, SGLang HiCache
// placement, LMCache's load-vs-recompute economics, vLLM's CPU-offload connector.
//
// Same pure-planner discipline as kvcost.go / kvresidency.go / kvprecision.go: no allocation,
// no byte movement, no model state. It DECIDES; the engine/kvmmu adapters move the bytes. This
// is #2671's R1 rung; consulting it from radixkv's evictToBudget victim path and the native
// preemptor (behind FAK_NATIVE_KV_* flags), with restore-on-access (#1469) closing the loop, is
// the R2 cross-lane follow-on.
//
// Ladder scope: this rung of the work prices {spill-host, evict}. ActionQuantize (the #1474
// requantize-and-keep fate) needs a density delta plus a KVSpanStats.Precision guard against
// double-quantizing an already-dense span, and the disk/object rungs wait on #2169's tier
// wiring; both append to the vocabulary without restructuring the ranking below.
//
// Fail-open, in the grain of kvresidency.go and kvprecision.go: a non-positive deficit demotes
// nothing, an unpriced or unavailable rung is treated as absent (fak never demotes into a tier
// it cannot cost — the span evicts instead), and a pinned/leased span meets NO fate at all.

// KVTierAction is the closed vocabulary of fates a resident KV span can meet under budget
// pressure, ordered hottest-kept to coldest-lost. It is the generalization of the binary
// keep/evict decision: every value names a rung of the memory ladder, and each rung carries its
// own (bytesFreed, restoreCost) economics rather than the evictor's single "drop it" fate.
type KVTierAction uint8

const (
	// ActionKeep leaves the span resident and untouched. It is the implicit default for every
	// span a demotion plan does not name, so PlanKVDemotion never emits it as a decision.
	ActionKeep KVTierAction = iota
	// ActionSpillHost relocates the span to the roomy host pool (the kvresidency.go cold tier).
	// The span leaves the pressured pool entirely, so it frees its full footprint; its restore
	// cost is the transfer back, priced per byte — far below the full re-prefill ActionEvict
	// books. This is the rung that makes demote-not-drop pay.
	ActionSpillHost
	// ActionEvict drops the span. It is the TOP of the ladder, not a separate kind of thing:
	// the rung whose restore cost is a full re-prefill of the span's tokens. When it is the
	// only configured rung the planner reduces exactly to PickEvictionVictim.
	ActionEvict
)

// String renders the action as its short selector token, or "kvaction?" for an unknown value —
// the same closed-vocabulary rendering discipline as KVPrecision.String.
func (a KVTierAction) String() string {
	switch a {
	case ActionKeep:
		return "keep"
	case ActionSpillHost:
		return "spill-host"
	case ActionEvict:
		return "evict"
	default:
		return "kvaction?"
	}
}

// isSpill reports whether the action relocates the span to a lower tier — a BYTE-priced rung
// whose restore cost is a transfer back (RestoreCostPerByte × bytes), as opposed to the
// TOKEN-priced ActionEvict rung whose restore is a full re-prefill. The disk/object rungs join
// here when #2169 lands them.
func (a KVTierAction) isSpill() bool { return a == ActionSpillHost }

// KVTierProfile prices ONE rung of the memory ladder for the demotion planner. The caller
// (which knows the box) supplies one profile per fate it is willing to let a span meet; a rung
// absent from the slice is a fate the planner will never choose.
type KVTierProfile struct {
	// Action is the fate this rung names.
	Action KVTierAction
	// Available reports that the rung is configured and permitted. It is the CALLER's
	// attestation, never something this package verifies: an unavailable rung is dropped from
	// consideration entirely. (For a future quantize rung this is where the #1474 proven-quality
	// gate — which lives in cachemeta, not compute — is asserted by the caller.)
	Available bool
	// RestoreCostPerByte is the cost, in the caller's units, to bring one byte of the span back
	// from this tier — a latency proxy. It prices the SPILL rungs only; ActionEvict is priced
	// per token by PlanKVDemotion's recomputeCostPerToken instead. A non-positive value means
	// the rung is UNPRICED: the planner treats it as unavailable rather than silently demoting
	// into a tier it cannot cost.
	RestoreCostPerByte float64
}

// KVDemotionDecision is one (span, action) verdict in a demotion plan: the fate chosen for the
// span, the bytes that fate frees from the pressured pool, and the restore cost it books.
type KVDemotionDecision struct {
	// SpanIndex indexes into the spans slice handed to PlanKVDemotion.
	SpanIndex int
	// Action is the chosen fate. Never ActionKeep — keeping is the implicit default for every
	// span the plan does not name.
	Action KVTierAction
	// BytesFreed is what this action releases from the pressured pool. Both spill and evict
	// remove the span's whole footprint, so this is the span's Bytes.
	BytesFreed int64
	// RestoreCost is the cost booked if/when the span is needed again — a full re-prefill
	// (Tokens × recomputeCostPerToken) for ActionEvict, or the transfer back
	// (Bytes × RestoreCostPerByte) for a spill rung. It is the raw restore price, NOT weighted
	// by reuse probability: reuse weights the RANKING (which span to demote), while this is what
	// the restore actually costs when it happens — the number the cache-value ledger (#1072)
	// records as OBSERVED provenance. The demote-not-drop win is the difference between the
	// RestoreCost booked here and the re-prefill a drop would have booked.
	RestoreCost float64
}

// PlanKVDemotion picks the min-cost set of (span, action) pairs that frees deficitBytes from
// the pressured pool, generalizing PickEvictionVictim's binary drop to the full tier ladder.
//
// For each evictable span s and each admissible rung t it forms a candidate whose ranking key
// is the EXPECTED LOSS PER BYTE FREED:
//
//	costPerByte = (Hits+1) × restoreCost(s, t) ÷ bytesFreed(s, t)
//
// with the Laplace-smoothed reuse probability KVEvictionCost already uses (kvcost.go). The
// restore cost is a full re-prefill for the evict rung (Tokens × recomputeCostPerToken) and a
// transfer back for a spill rung (Bytes × RestoreCostPerByte); either way the whole span leaves
// the pressured pool, so bytesFreed is s.Bytes. Candidates are taken cheapest-first until the
// running freed total clears deficitBytes, at most one fate per span (the cheapest one, since a
// span's spill and evict candidates free identical bytes and differ only in restore cost). Cost
// ties break to the OLDEST LastUsed, the same LRU rule as pickLowestCost.
//
// recomputeCostPerToken is the unit bridge: evict is priced in TOKENS of re-prefill and spill in
// BYTES of transfer, and the two cannot be compared without it. A non-positive value means the
// units are incommensurable, so the spill rungs are dropped as unpriced and the planner ranks
// evictions by KVEvictionCost directly — i.e. it reduces to drop-only rather than guessing a
// conversion. (Precedent: cachemeta.PlacementRequest.PerTokenPrefillNanos.)
//
// Reduction: with only the evict rung available, costPerByte = KVEvictionCost(s) ×
// recomputeCostPerToken — a positive constant times the existing score — so the ranking, the
// LRU tie-break, and therefore the first victim are exactly PickEvictionVictim's. Cost-aware
// demotion is a strict generalization of cost-aware eviction, never a divergence from it.
//
// Optimality: clearing a byte deficit at minimum aggregate cost is knapsack-shaped, and greedy
// cost-per-byte is the standard approximation, not a proven optimum. The witnesses fix behavior
// (the demote-not-drop flip, the reduction, fail-open), not global optimality.
//
// Fail-open floors, matching the kvresidency.go / kvprecision.go contract:
//   - a non-positive deficitBytes demotes nothing (no pressure ⇒ everything keeps);
//   - a Pinned or Leased span is excluded from EVERY rung, not just evict — a span being served
//     (radixkv's refs>0 rule) cannot be dropped, moved, or requantized mid-decode;
//   - a span with non-positive Bytes frees nothing measurable and is never demoted, mirroring
//     KVEvictionCost's +Inf fail-open on an unknown footprint;
//   - an unavailable or unpriced rung is treated as absent, so a span whose only lower tier is
//     uncosted still EVICTS. fak never demotes into a tier it cannot stand behind.
//
// The returned plan is nil when nothing is admissible. It is best-effort: if every admissible
// action together frees less than deficitBytes, the plan holds them all and the caller sees the
// shortfall by summing BytesFreed.
func PlanKVDemotion(spans []KVSpanStats, deficitBytes int64, tiers []KVTierProfile, recomputeCostPerToken float64) []KVDemotionDecision {
	if deficitBytes <= 0 || len(spans) == 0 {
		return nil
	}

	// The unit bridge. Without a positive per-token re-prefill price the byte-priced spill rungs
	// cannot be compared against the token-priced evict rung, so they are dropped as unpriced and
	// the planner degenerates to drop-only.
	spillComparable := recomputeCostPerToken > 0

	evictAvailable := false
	var spillRungs []KVTierProfile
	for _, t := range tiers {
		switch {
		case t.Action == ActionEvict:
			evictAvailable = evictAvailable || t.Available
		case t.Action.isSpill() && t.Available && t.RestoreCostPerByte > 0:
			// Available AND priced: the only way a lower tier enters consideration.
			spillRungs = append(spillRungs, t)
		}
	}

	// candidate is one (span, rung) pair the greedy ranks. bytesFreed is always the span's whole
	// footprint (both spill and evict remove it from the pressured pool), so the rungs for a span
	// differ only in restoreCost — and the cheapest is the fate that span meets.
	type candidate struct {
		spanIndex   int
		action      KVTierAction
		bytesFreed  int64
		restoreCost float64
		costPerByte float64
		lastUsed    uint64
	}
	var candidates []candidate

	for i := range spans {
		s := spans[i]
		if s.Pinned || s.Leased {
			continue // hard exclusion, extended from evict to every rung
		}
		if s.Bytes <= 0 {
			continue // unknown footprint frees nothing measurable — never a demotion candidate
		}
		reuseProbability := KVReuseTerm(s) // Unified reuse term seam (#3411)

		if spillComparable {
			for _, t := range spillRungs {
				restore := float64(s.Bytes) * t.RestoreCostPerByte
				candidates = append(candidates, candidate{
					spanIndex:   i,
					action:      t.Action,
					bytesFreed:  s.Bytes,
					restoreCost: restore,
					costPerByte: reuseProbability * restore / float64(s.Bytes),
					lastUsed:    s.LastUsed,
				})
			}
		}

		if evictAvailable {
			// Priced per token: the full re-prefill this drop throws away. Without the unit
			// bridge the score falls back to KVEvictionCost itself (an implicit unit of 1), which
			// leaves the drop-only ranking — and thus the reduction — exactly intact.
			restore := float64(s.Tokens) * recomputeCostPerToken
			costPerByte := KVEvictionCost(s)
			if spillComparable {
				costPerByte *= recomputeCostPerToken
			} else {
				restore = float64(s.Tokens)
			}
			candidates = append(candidates, candidate{
				spanIndex:   i,
				action:      ActionEvict,
				bytesFreed:  s.Bytes,
				restoreCost: restore,
				costPerByte: costPerByte,
				lastUsed:    s.LastUsed,
			})
		}
	}

	sort.SliceStable(candidates, func(a, b int) bool {
		ca, cb := candidates[a], candidates[b]
		if ca.costPerByte != cb.costPerByte {
			return ca.costPerByte < cb.costPerByte
		}
		if ca.lastUsed != cb.lastUsed {
			return ca.lastUsed < cb.lastUsed // equal cost ⇒ oldest first, the pickLowestCost rule
		}
		return ca.spanIndex < cb.spanIndex
	})

	var plan []KVDemotionDecision
	var freed int64
	assigned := make(map[int]bool, len(spans))
	for _, c := range candidates {
		if freed >= deficitBytes {
			break // deficit cleared — every remaining span keeps
		}
		if assigned[c.spanIndex] {
			continue // a span meets exactly one fate: the cheapest rung, already taken
		}
		assigned[c.spanIndex] = true
		freed += c.bytesFreed
		plan = append(plan, KVDemotionDecision{
			SpanIndex:   c.spanIndex,
			Action:      c.action,
			BytesFreed:  c.bytesFreed,
			RestoreCost: c.restoreCost,
		})
	}
	return plan
}
