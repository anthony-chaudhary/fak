package model

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// expert_ring_policy.go — R4 of the activated-expert offload ladder (#5615, epic #5606,
// docs/MOE-ACTIVATED-OFFLOAD-PLAN.md): give the bounded routed-expert ring a VICTIM-POLICY seam,
// record the live routing trace the seam must be judged on, and gate the promotion on measured
// regret rather than on the belief that recency is the right prior.
//
// Why this rung exists. The ring evicts by pure recency because it borrows polymodel.Pool's LRU,
// and LRU arrived as polymodel's DEFAULT for whole models — it was never a finding about MoE expert
// access. The literature is explicitly contested here: routing is layer-sequential, so a resident
// expert's recency carries much less information about its next use than it does for a KV span, and
// #4357 measured exactly that on this repo's own traces. Yet nothing could act on the measurement:
// simulateLFUDecayResidency (expert_residency_lfu.go) is a pure accounting loop, ReplayKVCacheMulti
// and BeladyKVReplayOracle score policies offline, and the RING has no way to be told which one to
// run. Every piece of the evidence shipped; the seam it should have moved did not exist.
//
// What this closes.
//
//	seam    — pagedRing owns victim choice under ExpertRingEvictValueAware: it selects victims by
//	          decaying heat (LFU-decay, LRU tie-break) and evicts them BEFORE polymodel.Admit, so
//	          Admit finds room and never has to choose. Under the default ExpertRingEvictLRU the
//	          ring does nothing of the sort and Admit's own coldestUnpinned choice stands — R0/R2
//	          byte-for-byte.
//	trace   — the ring records the ORDERED access sequence its own staging produced (the usage
//	          histogram R2 keeps is aggregate and loses the order Belady needs), so the gauge can be
//	          pointed at what this workload actually routed rather than at a synthetic corpus.
//	gate    — SelectExpertRingEvictPolicy replays that trace under both policies against the same
//	          offline oracle and promotes the candidate ONLY on a strict eviction win with no hit
//	          regression. A tie, a regression, or an unusable trace all keep LRU.
//
// The live policy is deliberately the SAME ranking simulateLFUDecayResidency simulates — decayed
// integer heat, ghost heat retained across an eviction, deterministic (heat, last-use, id)
// tie-breaks — because that identity is what makes the offline replay PREDICTIVE of the ring. A
// gauge that scores a policy the ring does not actually run measures nothing.
//
// What it does NOT take from #4357: the admission HYSTERESIS (bypass a lukewarm newcomer rather
// than let it evict a hot resident). In the simulation a bypass is free; in the ring the caller
// needs a handle, and weightHALStagedBounded's fallback for a refused stage is PERMANENT halW
// residency — so a bypass would promote the very expert the policy wanted to keep transient, and
// break the "no routed expert reaches halW" bound R0 witnesses. Serving a bypassed weight
// transiently (upload, use, free — pagedKernel's lifecycle) needs a staging contract the HAL does
// not have yet, so hysteresis stays simulated and only the victim RANKING goes live.

// ExpertRingEvictPolicy selects how the routed-expert ring ranks eviction victims among its
// unpinned residents. It mirrors compute.KVEvictPolicy's shape (a small enum whose zero value is
// the incumbent) so a caller reading both reads one vocabulary.
type ExpertRingEvictPolicy int

const (
	// ExpertRingEvictLRU is pure recency — polymodel.Pool's own coldestUnpinned choice, which the
	// ring has always inherited. It is the ZERO VALUE, so a session that says nothing runs exactly
	// the ring R0 shipped, down to allocating no heat state at all.
	ExpertRingEvictLRU ExpertRingEvictPolicy = iota
	// ExpertRingEvictValueAware is #4357's value-aware ranking: evict the resident with the lowest
	// DECAYING heat, tie-broken by least-recent use and then by id. Heat counts every access
	// (hit or miss) and is right-shifted every expertRingDecayEveryAccesses touches, so an early
	// burst fades and the resident set follows phase drift instead of an unforgettable prior. Heat
	// SURVIVES an eviction as ghost heat, so an expert that keeps being routed re-earns its place.
	ExpertRingEvictValueAware
)

// String names the policy for a report or a log line.
func (p ExpertRingEvictPolicy) String() string {
	switch p {
	case ExpertRingEvictValueAware:
		return "value-aware"
	default:
		return "lru"
	}
}

// expertRingDecayEveryAccesses is how many ring stagings pass between heat right-shifts, matching
// defaultDecayEveryAccesses so the live ring and the offline simulation age heat identically.
const expertRingDecayEveryAccesses = defaultDecayEveryAccesses

// expertRingTraceLimit bounds the recorded access trace. A serve runs for days; the gauge needs a
// window, not a journal. At the limit the trace stops GROWING rather than rotating, so the retained
// prefix stays a contiguous, replayable sequence — a ring buffer would splice two disjoint phases
// into one trace and the oracle would score the seam.
const expertRingTraceLimit = 1 << 16

// noteAccess folds one staging into the ring's policy state: it advances the recency clock and, when
// a value-aware policy is running, bumps this weight's heat and ages every counter on the decay
// cadence. Under the default LRU it only ticks the clock, so no heat map is ever allocated.
func (r *pagedRing) noteAccess(id polymodel.ModelID) {
	r.clock++
	if r.lastUse == nil {
		r.lastUse = map[polymodel.ModelID]uint64{}
	}
	r.lastUse[id] = r.clock
	// A prefetch (R3/#5614) is a HINT, not a demand: it is genuinely newly resident, so it takes
	// recency, but it must not earn heat. Heat ranked on the prefetcher's own guesses is
	// self-confirming — it would protect exactly what was speculated whether or not anything read it,
	// and the offline gauge would then score an access stream the workload never produced.
	if r.policy == ExpertRingEvictLRU || r.prefetching {
		return
	}
	if r.heat == nil {
		r.heat = map[polymodel.ModelID]int{}
	}
	r.accesses++
	if r.accesses%expertRingDecayEveryAccesses == 0 {
		for k := range r.heat {
			r.heat[k] >>= 1
		}
	}
	r.heat[id]++ // counted on a hit AND a miss, exactly as the simulation counts a touch
}

// evictForPolicy makes room for weightBytes by evicting the ring's OWN choice of victims, so that
// the polymodel Admit that follows fits without evicting anything and its LRU choice never applies.
// It is a no-op under the default policy, when the weight already fits, and when the weight alone
// exceeds the budget (that is polymodel's ErrTooLarge case: paging residents out for a weight that
// can never be admitted would be pure loss).
//
// The victim list is computed WITHOUT mutating anything and applied only if it actually makes room.
// That preserves stage's all-or-nothing contract: a weight that fits only by dropping a pinned
// resident (ErrPinnedNoRoom) leaves the resident set exactly as it found it, so the caller's
// fallback is unchanged and no expert is paged out for a stage that then refuses.
func (r *pagedRing) evictForPolicy(weightBytes int64) {
	if r.policy == ExpertRingEvictLRU || weightBytes <= 0 {
		return
	}
	budget := r.pool.Budget()
	if weightBytes > budget || r.pool.Used()+weightBytes <= budget {
		return
	}
	victims, ok := r.victimsByHeat(budget - weightBytes)
	if !ok {
		return // no set of unpinned residents makes room; leave the ring untouched
	}
	for _, id := range victims {
		if h, live := r.resident[id]; live {
			r.discardStaged(id)
			r.be.Free(h)
			delete(r.resident, id)
		}
		r.pool.Evict(id)
		delete(r.lastUse, id)
		// r.heat[id] is deliberately RETAINED as ghost heat: an expert that keeps being routed
		// accumulates value while evicted and re-earns residency, which is the whole point of a
		// frequency signal over a recency one.
		r.evict++
	}
}

// victimsByHeat returns the coldest unpinned residents, in eviction order, whose removal brings the
// footprint down to target — or ok=false if evicting every unpinned resident still would not. The
// ranking is (lowest heat, then least-recent use, then lowest id): fully deterministic under Go's
// randomized map iteration, and identical to simulateLFUDecayResidency's coldestVictim, which is
// what makes the offline replay predictive of this ring.
func (r *pagedRing) victimsByHeat(target int64) ([]polymodel.ModelID, bool) {
	ids := make([]polymodel.ModelID, 0, len(r.resident))
	for id := range r.resident {
		if m, ok := r.pool.Get(id); ok && m.Pinned {
			continue // durable pin (R2) or a live hold span — never a victim
		}
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool {
		hi, hj := r.heat[ids[i]], r.heat[ids[j]]
		if hi != hj {
			return hi < hj
		}
		ui, uj := r.lastUse[ids[i]], r.lastUse[ids[j]]
		if ui != uj {
			return ui < uj
		}
		return ids[i] < ids[j]
	})
	used := r.pool.Used()
	for n, id := range ids {
		if used <= target {
			return ids[:n], true
		}
		if m, ok := r.pool.Get(id); ok {
			used -= m.WeightBytes
		}
	}
	if used <= target {
		return ids, true
	}
	return nil, false
}

// observeTrace appends one routed-expert access to the ring's ordered trace — the record the offline
// gauge replays. Consecutive stagings of the SAME expert are coalesced into one access with their
// bytes summed, because one activation is three projections (gate/up/down) staged back to back and
// the policy question is about the EXPERT, not about which of its matrices was touched first.
//
// Past expertRingTraceLimit accesses the trace stops growing. Truncating rather than rotating keeps
// the retained events a contiguous prefix: a ring buffer would splice a later phase onto an earlier
// one and the Belady oracle would score a discontinuity that never happened.
func (r *pagedRing) observeTrace(layer, expert int, weightBytes int64) {
	if r == nil || weightBytes <= 0 || layer < 0 || expert < 0 {
		return
	}
	if n := len(r.trace); n > 0 {
		if last := &r.trace[n-1]; last.Layer == layer && last.Expert == expert {
			last.WeightBytes += weightBytes
			return
		}
	}
	if len(r.trace) >= expertRingTraceLimit {
		r.traceDropped++
		return
	}
	r.trace = append(r.trace, ExpertAccessTraceEvent{Layer: layer, Expert: expert, WeightBytes: weightBytes})
}

// ExpertRingTrace is this session's recorded routed-expert access sequence, shaped as the replay
// trace ReplayExpertAccessTrace / ReplayExpertResidencyLFUDecay / SelectExpertRingEvictPolicy
// consume. It is the LIVE counterpart of ExpertTraceRecorder (which observes the router) and of
// GenerateExpertReplaySyntheticTrace (which invents a corpus): this one is what the ring itself
// actually staged, at the budget it actually ran under.
//
// Sizes are normalized per (layer, expert) to the largest run recorded for that expert, because the
// replay requires one stable footprint per span and a first or last activation may be coalesced
// from a different number of projections. Normalizing UP means the replay reserves at least what
// the ring reserved, so the simulated budget can never be more generous than the real one.
//
// The zero trace (no ring, or nothing staged) is returned as-is; the replay entry points reject it
// with a clear error rather than scoring an empty window.
func (s *Session) ExpertRingTrace() ExpertAccessTrace {
	if s == nil || s.expertRing == nil || len(s.expertRing.trace) == 0 {
		return ExpertAccessTrace{}
	}
	r := s.expertRing
	// R7/#5618: the trace is ring state a peer agent may be appending to right now, so the snapshot
	// is taken inside a ring span. Inert under the per-session default.
	done := s.ringEnter(r)
	defer done()
	type key struct{ layer, expert int }
	size := map[key]int64{}
	for _, e := range r.trace {
		if k := (key{e.Layer, e.Expert}); e.WeightBytes > size[k] {
			size[k] = e.WeightBytes
		}
	}
	events := make([]ExpertAccessTraceEvent, len(r.trace))
	for i, e := range r.trace {
		events[i] = ExpertAccessTraceEvent{Layer: e.Layer, Expert: e.Expert, WeightBytes: size[key{e.Layer, e.Expert}]}
	}
	return ExpertAccessTrace{
		Schema:      ExpertReplayTraceSchema,
		Name:        "observed-expert-ring",
		Source:      fmt.Sprintf("paged-ring policy=%s budget_bytes=%d", r.policy, r.budget()),
		BudgetBytes: r.budget(),
		Events:      events,
		// A dropped access is an access the window did not see, not one that did not happen —
		// surfacing it keeps a truncated trace from reading as a complete one.
		UnsizedTouches: r.traceDropped,
	}
}

// ExpertRingEvictDecision is the promotion verdict: which policy the evidence supports, against
// what deltas, and — when the answer is "keep LRU" — why. Reason is always populated, because a
// gate that only explains its promotions turns every demotion into an unexplained default.
type ExpertRingEvictDecision struct {
	Policy   ExpertRingEvictPolicy `json:"policy"`
	Promoted bool                  `json:"promoted"`
	Reason   string                `json:"reason"`
	// EvictionDelta = LRU evictions - candidate evictions (positive = the candidate thrashed the
	// page-in path less). HitDelta = candidate hit tokens - LRU hit tokens (negative = a hit
	// regression, which blocks promotion however good the eviction delta looks).
	EvictionDelta int `json:"eviction_delta"`
	HitDelta      int `json:"hit_delta"`
	// GoodDecisionRatio of each policy against the SAME offline Belady oracle — the regret gauge
	// (#4233) the plan requires the promotion to be argued in, rather than in raw hit counts.
	LRUGoodDecisionRatio       float64 `json:"lru_good_decision_ratio"`
	CandidateGoodDecisionRatio float64 `json:"candidate_good_decision_ratio"`
}

// SelectExpertRingEvictPolicy is the promotion GATE: it replays one trace under the incumbent LRU
// and under the value-aware candidate, scores both against the same offline Belady oracle, and
// returns the policy the evidence supports. The candidate is promoted only on a STRICT eviction win
// with NO hit regression — the rule the plan states, and the reason this rung is "promote on
// measured regret, not on faith". Everything else keeps LRU: a tie is not evidence, and a policy
// that trades hits for eviction count has moved the cost rather than removed it.
//
// It deliberately does NOT call ReplayExpertResidencyLFUDecay, which scores #4357's FULL policy —
// decaying-heat ranking plus admission hysteresis. The ring runs the ranking alone (see this file's
// header for why the bypass cannot go live yet), and hysteresis is the half that does most of the
// work on a jitter trace: scoring the ring against it would promote on a win the ring cannot
// collect. simulateHeatResidency with hysteresis=false is the ring's policy exactly, so this gate
// measures what will actually run.
//
// An unusable trace (empty, unbudgeted, malformed) is an error and keeps LRU. It is not silently
// treated as a demotion, because "we could not measure" and "we measured and it lost" are different
// facts and a caller that conflates them will re-run the gate forever expecting a different answer.
func SelectExpertRingEvictPolicy(trace ExpertAccessTrace, opts ExpertResidencyLFUOptions) (ExpertRingEvictPolicy, ExpertRingEvictDecision, error) {
	events, err := trace.replayEvents()
	if err == nil {
		var budget int
		if budget, err = replayInt(trace.BudgetBytes, "budget_bytes"); err == nil {
			return decideExpertRingEvictPolicy(events, budget, opts)
		}
	}
	return ExpertRingEvictLRU, ExpertRingEvictDecision{
		Policy: ExpertRingEvictLRU, Reason: "trace not replayable: " + err.Error(),
	}, err
}

// decideExpertRingEvictPolicy is the scored comparison behind the gate: the same event stream, the
// same budget and the same offline oracle for both policies, so the deltas are attributable to the
// victim ranking and to nothing else.
func decideExpertRingEvictPolicy(events []compute.KVReplayEvent, budget int, opts ExpertResidencyLFUOptions) (ExpertRingEvictPolicy, ExpertRingEvictDecision, error) {
	decayEvery := opts.DecayEveryAccesses
	if decayEvery <= 0 {
		decayEvery = expertRingDecayEveryAccesses
	}
	oracle := compute.BeladyKVReplayOracle(events, budget)
	lru := compute.ReplayKVCacheMulti(events, budget, compute.KVEvictLRU)[compute.KVEvictLRU]
	cand := simulateHeatResidency(events, budget, decayEvery, false)
	cand.GoodDecisionRatio = ratioAgainstOracle(cand.HitTokens, oracle.HitTokens)

	d := ExpertRingEvictDecision{
		Policy:                     ExpertRingEvictLRU,
		EvictionDelta:              lru.Evictions - cand.Evictions,
		HitDelta:                   cand.HitTokens - lru.HitTokens,
		LRUGoodDecisionRatio:       lru.GoodDecisionRatio,
		CandidateGoodDecisionRatio: cand.GoodDecisionRatio,
	}
	switch {
	case d.HitDelta < 0:
		d.Reason = fmt.Sprintf("hit regression: candidate is %d hit tokens behind LRU", -d.HitDelta)
	case d.EvictionDelta <= 0:
		d.Reason = fmt.Sprintf("no eviction win: delta %d (needs > 0)", d.EvictionDelta)
	default:
		d.Policy, d.Promoted = ExpertRingEvictValueAware, true
		d.Reason = fmt.Sprintf("%d fewer evictions with no hit regression (good-decision ratio %.3f vs LRU %.3f)",
			d.EvictionDelta, d.CandidateGoodDecisionRatio, d.LRUGoodDecisionRatio)
	}
	return d.Policy, d, nil
}

// SelectExpertRingEvictPolicyForSession runs the gate over what THIS session's ring actually staged
// — the closing of the loop the rung is for: measure the live workload, decide on it, and hand back
// a policy the next session can be constructed with. It does not mutate the running ring: changing
// victim policy mid-flight would score a window under two policies and make the next measurement
// uninterpretable.
func (s *Session) SelectExpertRingEvictPolicy(opts ExpertResidencyLFUOptions) (ExpertRingEvictPolicy, ExpertRingEvictDecision, error) {
	return SelectExpertRingEvictPolicy(s.ExpertRingTrace(), opts)
}
