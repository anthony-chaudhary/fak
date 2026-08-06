package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Value-aware expert residency (issue #4357, epic #3174, colibri-inspired @1bdaeee c/tier.h).
//
// pagedRing today admits every routed-expert miss and evicts by pure recency (LRU). That is
// the right prior when reuse is recency-shaped, but it thrashes the page-in path under two
// workloads a long MoE run actually exhibits: (a) small-sample routing JITTER, where a
// one-off cold expert evicts a genuinely hot resident purely because it was touched most
// recently; and (b) workload-PHASE DRIFT, where the hot set changes slowly and a frequency
// signal tracks it better than recency alone. colibri pays a few integer ops per miss to
// protect its hot set because each miss is a full SSD read; the same tradeoff transfers to
// any fak tier whose miss hits a slow backing store.
//
// This file is a host-free SIMULATION of a value-aware residency policy — a gen/second-next
// prototype deliberately kept OFF the live serve path (exactly like the pagedRing replay
// harness it reuses). It borrows two ideas from colibri, clean-room, no bytes vendored:
//
//   - HYSTERESIS: on a miss that needs eviction, admit the newcomer only if its heat beats
//     the coldest victim's heat by a margin — `hot > victim + victim/4 + 4`. A lukewarm
//     one-off cannot evict a hot resident; a genuinely-hot newcomer accumulates heat across
//     repeated (bypassed) accesses until it clears the margin and earns promotion.
//   - LFU-DECAY: every DecayEveryAccesses touches, right-shift every heat counter
//     (`heat >>= 1`). Old heat fades exponentially, so the resident set follows phase drift
//     instead of accumulating an unforgettable early hot set.
//
// The policy is benched against the canonical compute.KVEvictLRU over the SAME event stream
// and the same Belady oracle, so the comparison is apples-to-apples and the existing LRU /
// CostAware witnesses in internal/compute are left byte-identical (compatibility preserved).

// hysteresisMarginDescriptor is the human-readable admit rule, mirrored from colibri tier.h.
const hysteresisMarginDescriptor = "hot > victim + victim/4 + 4"

// defaultDecayEveryAccesses is the LFU-decay pass interval when the caller leaves it unset.
const defaultDecayEveryAccesses = 16

// ExpertResidencyLFUOptions configures the value-aware residency simulation. The hysteresis
// margin itself is fixed to colibri's `victim + victim/4 + 4` — the axis under test — so only
// the decay cadence is tunable.
type ExpertResidencyLFUOptions struct {
	// DecayEveryAccesses is how many trace touches pass between heat right-shifts. <=0 uses
	// defaultDecayEveryAccesses. A very large value degenerates to pure LFU (no decay).
	DecayEveryAccesses int
}

// ExpertResidencyPolicyRow is the LFU-decay row, shaped to line up field-for-field with the
// compute.KVReplayResult the LRU baseline reports. PageIns and Bypasses are the two counters
// the axis actually targets: a page-in is a slow backing-store read this policy tries to
// avoid, and a bypass is a thrash LRU would have paid but hysteresis declined.
type ExpertResidencyPolicyRow struct {
	HitTokens         int     `json:"hit_tokens"`
	AccessTokens      int     `json:"access_tokens"`
	PageIns           int     `json:"page_ins"`
	Evictions         int     `json:"evictions"`
	Bypasses          int     `json:"bypasses"`
	EvictionsPerHit   float64 `json:"evictions_per_hit"`
	GoodDecisionRatio float64 `json:"good_decision_ratio"`
}

// ExpertResidencyLFUReport contrasts the value-aware policy against pagedRing's LRU on one
// trace. EvictionDelta and HitDelta are the promotion evidence: EvictionDelta>0 means the
// value-aware ring thrashed the page-in path less than LRU, and HitDelta>=0 means it did so
// without giving up hits. Both flat/negative is the demotion signal.
type ExpertResidencyLFUReport struct {
	Name               string                       `json:"name"`
	Source             string                       `json:"source"`
	BudgetBytes        int64                        `json:"budget_bytes"`
	DecayEveryAccesses int                          `json:"decay_every_accesses"`
	HysteresisMargin   string                       `json:"hysteresis_margin"`
	LFUDecay           ExpertResidencyPolicyRow     `json:"lfu_decay"`
	LRU                compute.KVReplayResult       `json:"lru"`
	Oracle             compute.KVReplayOracleResult `json:"oracle"`
	// EvictionDelta = LRU.Evictions - LFUDecay.Evictions (positive = value-aware thrashes less).
	EvictionDelta int `json:"eviction_delta"`
	// HitDelta = LFUDecay.HitTokens - LRU.HitTokens (>=0 = no hit regression vs LRU).
	HitDelta       int `json:"hit_delta"`
	UnsizedTouches int `json:"unsized_touches,omitempty"`
}

// ReplayExpertResidencyLFUDecay replays the trace under the value-aware (hysteresis +
// LFU-decay) residency policy and against the canonical pagedRing LRU baseline, scoring both
// with the same offline Belady oracle. It reuses the trace's validated event stream so the
// two policies see identical expert identities and resident byte sizes.
func ReplayExpertResidencyLFUDecay(trace ExpertAccessTrace, opts ExpertResidencyLFUOptions) (ExpertResidencyLFUReport, error) {
	events, err := trace.replayEvents()
	if err != nil {
		return ExpertResidencyLFUReport{}, err
	}
	budget, err := replayInt(trace.BudgetBytes, "budget_bytes")
	if err != nil {
		return ExpertResidencyLFUReport{}, err
	}
	decayEvery := opts.DecayEveryAccesses
	if decayEvery <= 0 {
		decayEvery = defaultDecayEveryAccesses
	}

	oracle := compute.BeladyKVReplayOracle(events, budget)
	lru := compute.ReplayKVCacheMulti(events, budget, compute.KVEvictLRU)[compute.KVEvictLRU]
	row := simulateLFUDecayResidency(events, budget, decayEvery)
	row.GoodDecisionRatio = ratioAgainstOracle(row.HitTokens, oracle.HitTokens)

	return ExpertResidencyLFUReport{
		Name: trace.Name, Source: trace.Source, BudgetBytes: trace.BudgetBytes,
		DecayEveryAccesses: decayEvery, HysteresisMargin: hysteresisMarginDescriptor,
		LFUDecay: row, LRU: lru, Oracle: oracle,
		EvictionDelta:  lru.Evictions - row.Evictions,
		HitDelta:       row.HitTokens - lru.HitTokens,
		UnsizedTouches: trace.UnsizedTouches,
	}, nil
}

// simulateLFUDecayResidency is the full value-aware policy: the decaying-heat victim ranking PLUS
// the admission hysteresis. It is what this file's #4357 report scores.
func simulateLFUDecayResidency(events []compute.KVReplayEvent, budget, decayEvery int) ExpertResidencyPolicyRow {
	return simulateHeatResidency(events, budget, decayEvery, true)
}

// simulateHeatResidency is the pure accounting loop, with the admission hysteresis as a SEPARATE
// axis from the victim ranking. The split exists because the live ring (#5615,
// expert_ring_policy.go) can adopt the ranking but not yet the hysteresis — a bypassed weight still
// has to be served, and the HAL's fallback for a refused stage is permanent residency — so scoring
// the ring against the hysteresis variant would measure a policy it does not run. Callers wanting
// the shipped #4357 policy use simulateLFUDecayResidency; the ring's gate passes hysteresis=false.
//
// Heat is tracked for every span ever seen, resident or not: a shadow ("ghost") heat lets a
// repeatedly-bypassed newcomer earn promotion, and an evicted span retains its heat so re-promotion
// respects accumulated value. Victim selection and every tie-break are deterministic (lowest heat,
// then oldest use, then lowest id) so the report is reproducible under Go's randomized map
// iteration.
func simulateHeatResidency(events []compute.KVReplayEvent, budget, decayEvery int, hysteresis bool) ExpertResidencyPolicyRow {
	heat := map[int]int{}
	resident := map[int]int{} // spanID -> tokens
	lastUsed := map[int]uint64{}
	residentTokens := 0
	var clock uint64
	var row ExpertResidencyPolicyRow
	accesses := 0

	// coldestVictim returns the resident span the policy would evict first, or -1 if none.
	coldestVictim := func() int {
		victim := -1
		var vHeat int
		var vLast uint64
		for id := range resident {
			h, lu := heat[id], lastUsed[id]
			switch {
			case victim == -1, h < vHeat, h == vHeat && lu < vLast,
				h == vHeat && lu == vLast && id < victim:
				victim, vHeat, vLast = id, h, lu
			}
		}
		return victim
	}

	for _, ev := range events {
		if ev.Tokens <= 0 {
			continue
		}
		accesses++
		if decayEvery > 0 && accesses%decayEvery == 0 {
			for id := range heat {
				heat[id] >>= 1
			}
		}
		row.AccessTokens += ev.Tokens
		clock++
		heat[ev.SpanID]++ // colibri increments the heat counter on every touch, resident or not.

		if _, ok := resident[ev.SpanID]; ok {
			row.HitTokens += ev.Tokens
			lastUsed[ev.SpanID] = clock
			continue
		}
		if ev.Tokens > budget {
			continue // larger than the whole ring: a permanent miss, never resident.
		}
		if residentTokens+ev.Tokens > budget {
			// Eviction required — apply the hysteresis gate against the coldest victim, when this
			// variant carries one. Without it the newcomer is always admitted and only the VICTIM
			// ranking differs from LRU, which is what the live ring implements.
			if hysteresis {
				victim := coldestVictim()
				vHeat := heat[victim]
				if heat[ev.SpanID] <= vHeat+vHeat/4+4 {
					row.Bypasses++ // newcomer not decisively hotter: decline to thrash the hot set.
					continue
				}
			}
			for residentTokens+ev.Tokens > budget && len(resident) > 0 {
				vid := coldestVictim()
				residentTokens -= resident[vid]
				delete(resident, vid)
				delete(lastUsed, vid)
				// heat[vid] is intentionally retained as shadow/ghost heat.
				row.Evictions++
			}
		}
		resident[ev.SpanID] = ev.Tokens
		lastUsed[ev.SpanID] = clock
		residentTokens += ev.Tokens
		row.PageIns++
	}
	row.EvictionsPerHit = residencyEvictionsPerHit(row.Evictions, row.HitTokens)
	return row
}

func ratioAgainstOracle(hitTokens, oracleHitTokens int) float64 {
	if oracleHitTokens <= 0 {
		if hitTokens == 0 {
			return 1
		}
		return 0
	}
	return float64(hitTokens) / float64(oracleHitTokens)
}

func residencyEvictionsPerHit(evictions, hitTokens int) float64 {
	if hitTokens <= 0 {
		if evictions == 0 {
			return 0
		}
		return float64(evictions) // undefined ratio surfaced as the raw eviction count.
	}
	return float64(evictions) / float64(hitTokens)
}

// GenerateHotSetJitterTrace builds a deterministic trace with a small stable hot set plus a
// stream of cold one-off experts (routing jitter). It is the workload the hysteresis axis is
// designed for: LRU lets each cold one-off evict a hot resident, while the value-aware policy
// bypasses them and keeps the hot set paged in. hotExperts stay resident within budget;
// coldOneOffs are each touched once, interleaved between hot re-touches.
func GenerateHotSetJitterTrace(hotExperts, coldOneOffs int, weightBytes, budgetBytes int64) ExpertAccessTrace {
	if hotExperts < 1 {
		hotExperts = 2
	}
	if coldOneOffs < 1 {
		coldOneOffs = 6
	}
	if weightBytes <= 0 {
		weightBytes = q4kBlockBytes
	}
	if budgetBytes <= 0 {
		budgetBytes = weightBytes * int64(hotExperts)
	}
	events := make([]ExpertAccessTraceEvent, 0, hotExperts*2+coldOneOffs*(hotExperts+1))
	touchHot := func() {
		for e := 0; e < hotExperts; e++ {
			events = append(events, ExpertAccessTraceEvent{Layer: 0, Expert: e, WeightBytes: weightBytes})
		}
	}
	// Warm the hot set so it accumulates heat before the jitter starts.
	touchHot()
	touchHot()
	// Each cold one-off (ids past the hot range) is a single touch, followed by a hot re-touch.
	for c := 0; c < coldOneOffs; c++ {
		events = append(events, ExpertAccessTraceEvent{Layer: 0, Expert: hotExperts + c, WeightBytes: weightBytes})
		touchHot()
	}
	return ExpertAccessTrace{
		Schema: ExpertReplayTraceSchema, Name: "hot-set-plus-routing-jitter",
		Source:      fmt.Sprintf("synthetic-hysteresis-fixture hot=%d cold=%d", hotExperts, coldOneOffs),
		BudgetBytes: budgetBytes, Events: events,
	}
}
