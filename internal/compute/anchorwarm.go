package compute

// anchorwarm.go — HOT-ANCHOR WARM ADMISSION: the pressure-gated serve-init decision that turns
// a persisted vCache hot-anchor index (the `fak vcache score --index-out anchors.json` artifact)
// into a concrete "warm the top-K star anchors now" plan, plus the realized-coverage accounting
// that makes the warm observable as live traffic arrives (issue #1078; parent epic #1072, row F).
//
// WHY HERE — the warm-admission sibling of prewarm_admit.go. Where prewarm_admit decides whether
// to warm ONE byte-known continuation prefix during a tool-latency window, this decides HOW MANY
// of a ranked star-anchor set to warm at serve init, gated by the SAME capacity pressure the
// #1072-A capacity executor reports — so a proactive warm never evicts demand-driven residency.
// "Warming respects pressure" (the issue's words) is exactly this gate: under pressure the plan
// warms fewer anchors, or none. It consumes the index as PLAIN DATA (ranked keys + weights, never
// prompt payload) so the tier-1 compute layer never imports the vcachescore / serve layers above
// it — the same boundary prewarm_admit.go keeps by taking a bare PrewarmCandidate.
//
// WHAT THIS FILE IS. The pure, deterministic, host-free DECISION + accounting:
//   - PlanAnchorWarm selects the warm set (the front run of the ranked index) under the capacity
//     verdict and reports its EXPECTED coverage (cumulative warm weight / total index weight).
//   - AnchorWarmPlan.SummaryLine renders the exact serve-log text the #1078 acceptance names:
//     "warming N star anchors, expected coverage X%".
//   - AnchorWarmPlan.RealizedCoverage folds a set of anchor keys OBSERVED warm in live traffic
//     into the realized fraction the fak_vcache_anchor_coverage_pct metric reports — it climbs
//     from 0 as traffic confirms each planned anchor is hot.
//
// HONEST SCOPE — what this is NOT. This is the DECISION + the coverage math, NOT the live wiring:
// it moves no KV and issues no prefill, and it reads no live metric. A serve path drives the
// actual warm (via the existing radixkv Insert / vcachewarm passthrough primitives) and feeds the
// observed-anchor set in once the capacity executor is on the live decode loop (#1073, the epic's
// OPEN keystone). That wiring is why the realized signal is not yet emitted: live turns are keyed
// by session/trace id, not by anchor prefix-digest, so there is no honest per-anchor hit stream
// until #1073 lands. This file closes the decidable, host-free core ahead of that live reader the
// same way prewarm_admit.go (#810) and discard_admit.go (#808) shipped their decisions ahead of
// theirs — pure integer/float arithmetic over plain inputs, so it is byte-deterministic across
// machines.

import (
	"fmt"
	"sort"
)

// AnchorWeight is one ranked star anchor projected to the warm decision: a stable key and its
// share of total reuse weight. Payload-free — only the metadata the index artifact carries (the
// serve seam maps a vcachescore.AnchorIndexEntry's Key/Weight onto this). A non-positive Weight
// is treated as 0 so a malformed row never warps the coverage denominator.
type AnchorWeight struct {
	Key    string
	Weight float64
}

// WarmPressure is the #1072-A capacity-executor verdict the warm respects, projected to plain
// data so compute decides without importing the engine/serve layers that produce it. It is the
// quantified cousin of prewarm_admit's WarmPoolFree bool: the warm is an opportunistic bet that
// must never evict demand-driven residency to place it.
type WarmPressure uint8

const (
	// WarmPressureNone: headroom — warm the full planned top-K. The zero value, so a
	// default-constructed input warms the whole plan.
	WarmPressureNone WarmPressure = iota
	// WarmPressurePartial: limited headroom — warm at most PressureCap anchors (the executor
	// admits a bounded warm so a warm never crowds out demand residency).
	WarmPressurePartial
	// WarmPressureHigh: no headroom — defer the warm entirely (warm nothing). A skipped warm
	// only ever costs a cold first prefill (the status quo), never a wrong answer.
	WarmPressureHigh
)

// String renders the pressure verdict for logs and the witness surface.
func (p WarmPressure) String() string {
	switch p {
	case WarmPressurePartial:
		return "partial"
	case WarmPressureHigh:
		return "high"
	default:
		return "none"
	}
}

// AnchorWarmReason explains why the plan warmed the count it did — a closed vocabulary for the
// audit log and witness; policy code never parses a free-text reason.
type AnchorWarmReason string

const (
	// ReasonNoIndex: the index carried no rankable anchor, so there is nothing to warm.
	ReasonNoIndex AnchorWarmReason = "no_index"
	// ReasonPressureDeferred: capacity pressure is high — the whole warm is deferred (fence:
	// never evict demand residency for an opportunistic warm).
	ReasonPressureDeferred AnchorWarmReason = "pressure_deferred"
	// ReasonPressureCapped: limited headroom trimmed the warm below the planned top-K.
	ReasonPressureCapped AnchorWarmReason = "pressure_capped"
	// ReasonWarmedPlanned: headroom admitted the full planned top-K warm.
	ReasonWarmedPlanned AnchorWarmReason = "warmed_planned"
)

// AnchorWarmInput is the pure decision input for one serve-init warm plan. Ranked is the
// persisted index in descending-weight order (the artifact already sorts it); PlannedAnchors is
// the index's chosen top-K (its AnchorCount); TargetCoverage is the index's coverage target,
// carried only for the witness. Pressure (+ PressureCap) is the capacity-executor verdict.
type AnchorWarmInput struct {
	Ranked         []AnchorWeight
	PlannedAnchors int
	TargetCoverage float64
	Pressure       WarmPressure
	// PressureCap is the maximum anchors the capacity executor admits when Pressure is
	// WarmPressurePartial. Negative is treated as 0 (defer); it is ignored for the other verdicts.
	PressureCap int
}

// AnchorWarmPlan is the typed serve-init warm decision: the selected warm set (the front run of
// the ranked index), its size (the "N" in the acceptance line), and its EXPECTED coverage (the
// "X"). Deferred is true when pressure warmed nothing. The plan retains the warm anchors' weights
// so RealizedCoverage can fold the live-observed set against the SAME denominator.
type AnchorWarmPlan struct {
	Warm             []AnchorWeight
	StarAnchors      int
	ExpectedCoverage float64
	TargetCoverage   float64
	TotalWeight      float64
	Deferred         bool
	Reason           AnchorWarmReason
}

// PlanAnchorWarm decides the serve-init warm set from a persisted index and a capacity verdict.
// It FAILS SAFE on pressure (a deferred or capped warm only ever costs a cold prefill, never a
// wrong answer) and is a pure fold over the ranked input, so it is deterministic for a given
// input. The expected coverage is the cumulative weight of the warm set over the WHOLE index's
// total weight, so it never reads above the index's own top-K coverage.
func PlanAnchorWarm(in AnchorWarmInput) AnchorWarmPlan {
	ranked := normalizeAnchorWeights(in.Ranked)
	total := 0.0
	for _, a := range ranked {
		total += a.Weight
	}
	plan := AnchorWarmPlan{
		TargetCoverage: clampUnit(in.TargetCoverage),
		TotalWeight:    total,
	}
	if len(ranked) == 0 || total <= 0 {
		plan.Reason = ReasonNoIndex
		return plan
	}

	// The planned top-K: the index's AnchorCount, clamped to what the index actually holds. A
	// non-positive PlannedAnchors falls back to the whole ranked set (warm everything ranked).
	k := in.PlannedAnchors
	if k <= 0 || k > len(ranked) {
		k = len(ranked)
	}

	switch in.Pressure {
	case WarmPressureHigh:
		plan.Deferred = true
		plan.Reason = ReasonPressureDeferred
		return plan
	case WarmPressurePartial:
		cap := in.PressureCap
		if cap < 0 {
			cap = 0
		}
		if cap == 0 {
			plan.Deferred = true
			plan.Reason = ReasonPressureDeferred
			return plan
		}
		if cap < k {
			k = cap
			plan.Reason = ReasonPressureCapped
		} else {
			plan.Reason = ReasonWarmedPlanned
		}
	default:
		plan.Reason = ReasonWarmedPlanned
	}

	warm := make([]AnchorWeight, k)
	copy(warm, ranked[:k])
	warmWeight := 0.0
	for _, a := range warm {
		warmWeight += a.Weight
	}
	plan.Warm = warm
	plan.StarAnchors = k
	plan.ExpectedCoverage = warmWeight / total
	return plan
}

// SummaryLine renders the exact serve-log text the #1078 acceptance names:
// "warming N star anchors, expected coverage X%". The percentage uses one decimal, matching the
// `fak vcache score` hot-anchor-index line so the two surfaces read identically.
func (p AnchorWarmPlan) SummaryLine() string {
	return fmt.Sprintf("warming %d star anchors, expected coverage %.1f%%", p.StarAnchors, 100*p.ExpectedCoverage)
}

// RealizedCoverage folds the set of anchor keys OBSERVED warm in live traffic into the realized
// fraction (0..1) the fak_vcache_anchor_coverage_pct metric reports: the share of the PLANNED
// warm weight whose anchor has actually been seen hot. It is measured against the warm set's own
// weight (not the whole index), so a fully-realized plan reads 1.0 and an all-cold one reads 0.0;
// it climbs monotonically as traffic confirms each planned anchor. A nil/empty observed set (no
// traffic yet) reads 0; a deferred plan (nothing warmed) reads 0.
func (p AnchorWarmPlan) RealizedCoverage(observed map[string]bool) float64 {
	if len(p.Warm) == 0 || len(observed) == 0 {
		return 0
	}
	warmWeight, hitWeight := 0.0, 0.0
	for _, a := range p.Warm {
		warmWeight += a.Weight
		if observed[a.Key] {
			hitWeight += a.Weight
		}
	}
	if warmWeight <= 0 {
		return 0
	}
	return hitWeight / warmWeight
}

// normalizeAnchorWeights drops empty-key / non-positive-weight rows and re-sorts the survivors by
// descending weight (key as a stable tiebreak), so the plan is order-insensitive to its input and
// a malformed row never warps the coverage math. It copies, never mutating the caller's slice.
func normalizeAnchorWeights(in []AnchorWeight) []AnchorWeight {
	out := make([]AnchorWeight, 0, len(in))
	for _, a := range in {
		if a.Key == "" || a.Weight <= 0 {
			continue
		}
		out = append(out, a)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Weight == out[j].Weight {
			return out[i].Key < out[j].Key
		}
		return out[i].Weight > out[j].Weight
	})
	return out
}

// clampUnit clamps a fraction to [0,1]; a NaN or out-of-range coverage target reads as 0.
func clampUnit(x float64) float64 {
	if x <= 0 || x != x {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}
