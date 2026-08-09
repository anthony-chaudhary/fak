package model

import (
	"fmt"
	"sort"
)

// expert_residency_report.go — R6 of the activated-expert offload ladder (#5617, epic #5606,
// docs/MOE-ACTIVATED-OFFLOAD-PLAN.md): make activated-expert residency an object an operator can
// READ, and make the reading checkable.
//
// The gap this closes. R0-R5 and R7 each shipped a counter, and every one of them stopped at the
// package boundary: pagedRing's pageIn/hit/evict were unexported, ExpertRingStats had no caller
// outside its own tests, and two finished gauges were stranded with no consumer at all —
// ScoreExpertPlacement (#3902: is the residency plan still right?) and the eviction-regret ratio
// against the Belady ceiling (#4233). By this repo's own standard — an operator can turn it on and
// inspect it — none of the ladder was first-class, and every rung's witness was a number in a PR
// body rather than something a second person could reproduce.
//
// One fold, not a new channel. MoEResidencyReport is the single view: the ring's ledger, the
// cross-agent coalescing ledger (R7) when the ring is shared, the checkpoint tier's IO (R5), the two
// stranded gauges, and the derived rates an operator actually tunes against. Every field is read
// from the component's OWN accounting; nothing here recomputes a number a rung already keeps.
//
// Reconciliation is the load-bearing part. A telemetry surface that merely restates its source can
// drift from it silently, so the ring now counts the reconciliation pair `lookups` and `refused`
// INDEPENDENTLY of the outcome counters (paging_ring.go): one increment at the top of stage() before
// anything is decided, one on the single exit that returns no handle. `Hits + PageIns + Refusals ==
// Lookups` is then an identity whose two sides come from different increments, so the check can
// FAIL, which is the only thing that makes it worth reporting. The same shape applies across the
// R7 boundary: the shared ledger's page-in bytes and refusal count are booked by the shared ring's
// hooks, the ring's by the ring, and the report checks that they agree.
//
// Zero cost when off. The report is PULL-based — nothing here runs during a decode. The three
// counters added to stage() only advance inside a ring, and a session with no ring
// (ExpertRingBytes == 0, the default) never reaches them.

// MoEResidencyOptions parameterizes the two things the report cannot know on its own.
type MoEResidencyOptions struct {
	// Tokens is how many output tokens the reported window produced. It is the denominator of
	// bytes-per-token, the number a budget is actually tuned against. Left at 0 the rate is reported
	// as 0, which reads as "not measured" rather than "free" — the ring cannot count tokens itself
	// because it never sees one.
	Tokens int64
	// Regret asks for the eviction-regret comparison (#4233): replay this window's own trace under
	// the incumbent LRU and under the value-aware candidate, both scored against the same offline
	// Belady oracle. It is OFF by default because the replay is O(trace) work and an operator polling
	// a live serve every second does not want it; the placement gauge and the counters are free.
	Regret bool
	// RegretOptions is passed through to the replay; its zero value is the ring's own decay cadence.
	RegretOptions ExpertResidencyLFUOptions
}

// MoEShape is the static routing shape the rest of the report is read against. It is here because
// every number below is meaningless without it: a 60% hit rate is excellent at k/E = 8/256 and
// unremarkable at 8/8, and ActivatedFraction is the whole premise of the ladder — modern MoE
// checkpoints fire a few percent of their parameters per token, so the stored-vs-activated gap is a
// residency problem before it is a compute one.
type MoEShape struct {
	Experts         int `json:"experts"`
	ExpertsPerToken int `json:"experts_per_token"`
	Layers          int `json:"layers"`
	// ActivatedFraction is ExpertsPerToken/Experts — the share of routed-expert parameters one token
	// actually touches. 0 when the model declares no expert count (a dense model has no ladder).
	ActivatedFraction float64 `json:"activated_fraction"`
}

// ExpertPlacementReport is #3902's coverage/drift gauge finally pointed at a live ring: how well the
// residency PLAN matches what this window's routing actually exercised.
//
// The plan is the durable pin-set (R2/#5613) when there is one, because that is the only thing in
// the system that claims "these experts should stay resident". A ring running plain LRU has no plan
// to score, so the gauge falls back to the CURRENT resident set — still the right question ("has
// residency drifted from the hot set?"), asked of a set that was chosen by recency rather than
// declared. Basis names which was used, because the two are not interchangeable evidence.
//
// The scoring unit is the (layer, expert) pair, not the expert ordinal: expert 7 in adjacent layers
// is a different weight with different heat, and folding them would report a plan that keeps layer
// 0's hot experts as if it also kept layer 30's.
type ExpertPlacementReport struct {
	// Basis is "pin-set", "resident-set", or "none" (with Reason set).
	Basis string `json:"basis"`
	// Coverage is the share of observed touches served by a planned-resident (layer, expert);
	// Drift is 1 - |planned ∩ observed hot set| / k, over k = BasisWidth. Coverage weights by
	// touch volume, Drift is unweighted set membership — they diverge exactly when the plan covers
	// most traffic but has swapped a few marginal members, which is the signal a single scalar hides.
	Coverage float64 `json:"coverage"`
	Drift    float64 `json:"drift"`
	// BasisWidth is the size of the scored plan and the width of the hot set it is compared to.
	// ObservedUnits is how many distinct (layer, expert) pairs the window touched, ObservedTouches
	// the total. A plan scored against a handful of touches is not evidence, so both travel with the
	// verdict rather than being folded into it.
	BasisWidth      int    `json:"basis_width"`
	ObservedUnits   int    `json:"observed_units"`
	ObservedTouches int64  `json:"observed_touches"`
	Reason          string `json:"reason,omitempty"`
}

// MoEResidencyRates are the derived numbers an operator tunes against. Each is a ratio of two
// counters in this same report, so any one of them can be re-derived by hand from the raw fields —
// deliberately, because a rate nobody can check is a rate nobody should trust.
type MoEResidencyRates struct {
	// Tokens is the window size the byte rates are over, echoed from the options.
	Tokens int64 `json:"tokens"`
	// HitRate is Hits/(Hits+PageIns) — the activated-set hit rate. RefusalRate is Refusals/Lookups,
	// and a non-zero one is the first thing to read: a refused staging falls back to PERMANENT halW
	// residency, so the budget is being quietly abandoned rather than enforced.
	HitRate     float64 `json:"hit_rate"`
	RefusalRate float64 `json:"refusal_rate"`
	// ActivatedResidentShare is ActivatedCovered/ActivatedExperts (R3's meter): of the experts the router
	// picked, the share the budget could hold. Below 1 the ring is too small to hold one step's
	// activated set, which a hit rate alone reports as honest misses forever.
	ActivatedResidentShare float64 `json:"activated_resident_share"`
	// AsyncOverlap is the share of fenced transfers already landed when demanded (#5627); 0 on a
	// synchronous backend means "not reportable here", not "no overlap".
	AsyncOverlap float64 `json:"async_overlap"`
	// BudgetUsed is ResidentBytes/BudgetBytes and PeakBudgetUsed the high-water equivalent. A peak
	// well under 1 means the budget is bigger than the workload needs.
	BudgetUsed     float64 `json:"budget_used"`
	PeakBudgetUsed float64 `json:"peak_budget_used"`
	// ExpertBytesPerToken is ring page-in bytes per output token — the traffic the whole ladder
	// exists to reduce. CheckpointBytesPerToken is the same for R5's checkpoint reads, which sit one
	// tier below and are counted separately because they are host IO, not device traffic.
	ExpertBytesPerToken     float64 `json:"expert_bytes_per_token"`
	CheckpointBytesPerToken float64 `json:"checkpoint_bytes_per_token"`
	// CrossAgentHitRate and AgentsPerPageIn are the R7 coalescing rates, populated only under a
	// shared ring. AgentsPerPageIn is the factor one streamed expert's cost is divided by.
	CrossAgentHitRate float64 `json:"cross_agent_hit_rate,omitempty"`
	AgentsPerPageIn   float64 `json:"agents_per_page_in,omitempty"`
}

// MoEResidencyCheck is one reconciliation invariant and its verdict. Detail is populated on failure
// AND on success, so a passing check still shows the numbers it passed on: "ok" with no evidence is
// indistinguishable from a check that was never run.
type MoEResidencyCheck struct {
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail"`
}

// MoEResidencyReconciliation is the verdict that the reported numbers ARE the components' own
// accounting. OK false means a counter pair that must agree by construction does not — a real bug in
// the ring or in a hook, never a rounding artifact, because every check compares integers.
type MoEResidencyReconciliation struct {
	OK     bool                `json:"ok"`
	Checks []MoEResidencyCheck `json:"checks"`
}

// MoEResidencyReport is the whole operator view of activated-expert residency for one session.
type MoEResidencyReport struct {
	Shape MoEShape `json:"shape"`
	// Ring is this session's bounded routed-expert residency; under a shared ring it is that ring's
	// aggregate, which is what actually bounds this session's experts.
	Ring ExpertRingStats `json:"ring"`
	// Shared is the cross-agent coalescing ledger (R7/#5618), absent on a private ring.
	Shared *SharedExpertRingStats `json:"shared,omitempty"`
	// Checkpoint is the R5 tier's host-IO ledger, Enabled=false when experts are fully resident.
	Checkpoint ExpertCheckpointStats `json:"checkpoint"`
	Placement  ExpertPlacementReport `json:"placement"`
	// Regret is the #4233 comparison, present only when MoEResidencyOptions.Regret asked for it and
	// the window produced a replayable trace.
	Regret         *ExpertRingEvictDecision   `json:"regret,omitempty"`
	Rates          MoEResidencyRates          `json:"rates"`
	Reconciliation MoEResidencyReconciliation `json:"reconciliation"`
}

// MoEResidency folds this session's activated-expert residency into the operator report.
//
// It is safe to call on any session: one with no ring reports Ring.Enabled=false and a
// reconciliation that trivially holds, which is the honest reading for the unbounded halW path
// (residency is whatever accumulated, so there is nothing to reconcile). Under a SHARED ring the
// whole gather runs inside one ring span, so the numbers are a consistent snapshot rather than
// several taken while peers moved the ring underneath.
func (s *Session) MoEResidency(opts MoEResidencyOptions) MoEResidencyReport {
	if s == nil {
		return MoEResidencyReport{}
	}
	rep := MoEResidencyReport{Shape: moeShapeOf(s.M)}
	if s.M != nil {
		rep.Checkpoint = s.M.ExpertCheckpointStats()
	}

	if r := s.expertRing; r != nil {
		var trace ExpertAccessTrace
		func() {
			// One span for every ring read below. ringEnter is reentrant, so ExpertRingTrace's own
			// span nests inside it; on a private ring the whole thing is a no-op closure.
			done := s.ringEnter(r)
			defer done()
			rep.Ring = r.stats()
			rep.Placement = r.placementReport(rep.Shape.Experts)
			if sh := s.sharedRing; sh != nil {
				st := sh.statsLocked()
				rep.Shared = &st
				rep.Ring = st.Ring
			}
			if opts.Regret {
				trace = s.ExpertRingTrace()
			}
		}()
		if opts.Regret && len(trace.Events) > 0 {
			if _, d, err := SelectExpertRingEvictPolicy(trace, opts.RegretOptions); err == nil {
				rep.Regret = &d
			}
		}
	} else {
		rep.Placement = ExpertPlacementReport{Basis: "none", Reason: "session has no routed-expert ring"}
	}

	rep.Rates = moeResidencyRates(rep, opts)
	rep.Reconciliation = reconcileMoEResidency(rep)
	return rep
}

// moeShapeOf reads the routing shape off the model config. A nil model or a dense one reports zeros,
// and ActivatedFraction stays 0 rather than dividing by an expert count of zero.
func moeShapeOf(m *Model) MoEShape {
	if m == nil {
		return MoEShape{}
	}
	sh := MoEShape{
		Experts:         m.Cfg.NumExperts,
		ExpertsPerToken: m.Cfg.NumExpertsPerTok,
		Layers:          m.Cfg.NumLayers,
	}
	if sh.Experts > 0 && sh.ExpertsPerToken > 0 {
		sh.ActivatedFraction = float64(sh.ExpertsPerToken) / float64(sh.Experts)
	}
	return sh
}

// placementReport scores the ring's residency plan against the routing this window observed. It
// takes no lock — like every other pagedRing method, it runs inside the caller's span.
func (r *pagedRing) placementReport(numExperts int) ExpertPlacementReport {
	if numExperts <= 0 {
		return ExpertPlacementReport{Basis: "none", Reason: "model declares no routed experts"}
	}
	if len(r.trace) == 0 {
		return ExpertPlacementReport{Basis: "none", Reason: "no routed-expert access observed yet"}
	}

	basis, planned := r.basisUnits()
	if len(planned) == 0 {
		return ExpertPlacementReport{
			Basis:  "none",
			Reason: "no expert is planned resident: the ring holds nothing and declares no pin-set",
		}
	}

	// Flatten (layer, expert) to one id space so the pure gauge can score it, and size the arrays to
	// the largest id either side actually uses — a plan may name a layer this window never touched.
	width := 0
	freq := map[int]int64{}
	var touches int64
	for _, e := range r.trace {
		if e.Expert < 0 || e.Expert >= numExperts || e.Layer < 0 {
			continue // not addressable in this id space; skip rather than fold two experts into one
		}
		id := e.Layer*numExperts + e.Expert
		freq[id]++
		touches++
		if id+1 > width {
			width = id + 1
		}
	}
	if touches == 0 {
		return ExpertPlacementReport{Basis: basis, Reason: "observed accesses fall outside the declared expert range"}
	}
	for _, u := range planned {
		if u.expert < 0 || u.expert >= numExperts || u.layer < 0 {
			continue
		}
		if id := u.layer*numExperts + u.expert; id+1 > width {
			width = id + 1
		}
	}

	hist := make([]int64, width)
	for id, n := range freq {
		hist[id] = n
	}
	mask := make([]bool, width)
	basisWidth := 0
	for _, u := range planned {
		if u.expert < 0 || u.expert >= numExperts || u.layer < 0 {
			continue
		}
		if id := u.layer*numExperts + u.expert; !mask[id] {
			mask[id] = true
			basisWidth++
		}
	}

	// k is the PLAN's own width: drift asks "of the k hottest units, how many did the plan keep?",
	// and scoring a k-sized plan against a different k would report drift the plan could not avoid.
	score := ScoreExpertPlacement(hist, mask, basisWidth)
	return ExpertPlacementReport{
		Basis:           basis,
		Coverage:        score.Coverage,
		Drift:           score.Drift,
		BasisWidth:      basisWidth,
		ObservedUnits:   len(freq),
		ObservedTouches: touches,
	}
}

// expertUnit is one (layer, expert) placement unit — the granularity the plan is scored at.
type expertUnit struct{ layer, expert int }

// basisUnits returns the set the placement gauge scores and the name of what it is. The durable
// pin-set wins when it holds anything, because it is the only DECLARED plan; otherwise the current
// resident set stands in, deduped across an expert's three projections (gate/up/down are one
// placement decision, not three). Resident ids are the dtype-prefixed HAL keys, and
// routedExpertIdentity parses the canonical name out of them by substring, so the prefix is
// harmless; an id that does not parse is skipped rather than counted as layer 0 expert 0.
func (r *pagedRing) basisUnits() (string, []expertUnit) {
	if r.pins != nil && r.pins.Len() > 0 {
		pins := r.pins.Pins()
		units := make([]expertUnit, 0, len(pins))
		for _, p := range pins {
			units = append(units, expertUnit{p.Layer, p.Expert})
		}
		return "pin-set", units
	}
	seen := map[expertUnit]bool{}
	units := make([]expertUnit, 0, len(r.resident))
	for id := range r.resident {
		layer, expert, ok := routedExpertIdentity(string(id))
		if !ok {
			continue
		}
		u := expertUnit{layer, expert}
		if !seen[u] {
			seen[u] = true
			units = append(units, u)
		}
	}
	// Map iteration is randomized; the gauge is order-independent but a stable order keeps two
	// reports of the same ring byte-identical, which is what makes a diff of them readable.
	sort.Slice(units, func(i, j int) bool {
		if units[i].layer != units[j].layer {
			return units[i].layer < units[j].layer
		}
		return units[i].expert < units[j].expert
	})
	return "resident-set", units
}

// moeResidencyRates derives the tuning ratios. Every denominator is guarded and answers 0 rather
// than NaN or +Inf: a JSON surface that emits NaN is unparseable by half its consumers, and an
// operator reading 0 next to the raw counters can see for themselves that nothing was measured.
func moeResidencyRates(rep MoEResidencyReport, opts MoEResidencyOptions) MoEResidencyRates {
	rt := MoEResidencyRates{Tokens: opts.Tokens}
	ring := rep.Ring
	if served := ring.Hits + ring.PageIns; served > 0 {
		rt.HitRate = float64(ring.Hits) / float64(served)
	}
	if ring.Lookups > 0 {
		rt.RefusalRate = float64(ring.Refusals) / float64(ring.Lookups)
	}
	if ring.ActivatedExperts > 0 {
		rt.ActivatedResidentShare = float64(ring.ActivatedCovered) / float64(ring.ActivatedExperts)
	}
	rt.AsyncOverlap = ring.AsyncOverlapFraction()
	if ring.BudgetBytes > 0 {
		rt.BudgetUsed = float64(ring.ResidentBytes) / float64(ring.BudgetBytes)
		rt.PeakBudgetUsed = float64(ring.PeakBytes) / float64(ring.BudgetBytes)
	}
	if opts.Tokens > 0 {
		rt.ExpertBytesPerToken = float64(ring.PageInBytes) / float64(opts.Tokens)
		rt.CheckpointBytesPerToken = float64(rep.Checkpoint.BytesRead) / float64(opts.Tokens)
	}
	if sh := rep.Shared; sh != nil {
		if sh.Demands > 0 {
			rt.CrossAgentHitRate = float64(sh.CrossAgentHits) / float64(sh.Demands)
		}
		rt.AgentsPerPageIn = sh.AgentsPerPageIn()
	}
	return rt
}

// reconcileMoEResidency runs the invariants that must hold if the report is the components' own
// accounting. Each check names both sides and how they were counted, so a failure points at the hook
// that drifted rather than at "telemetry is wrong".
//
// A session with no ring passes trivially and says so: the unbounded halW path keeps no ledger, and
// inventing a green check for it would make "reconciled" mean two different things.
func reconcileMoEResidency(rep MoEResidencyReport) MoEResidencyReconciliation {
	var rc MoEResidencyReconciliation
	add := func(name string, ok bool, format string, args ...any) {
		rc.Checks = append(rc.Checks, MoEResidencyCheck{Name: name, OK: ok, Detail: fmt.Sprintf(format, args...)})
	}
	ring := rep.Ring
	if !ring.Enabled {
		add("ring-absent", true, "no routed-expert ring on this session; the unbounded halW path keeps no ledger to reconcile")
		rc.OK = true
		return rc
	}

	// The identity the rung is for: lookups is booked at the top of stage(), the three outcome
	// counters on the three exits, so the two sides never share an increment.
	outcomes := ring.Hits + ring.PageIns + ring.Refusals
	add("lookups-identity", outcomes == ring.Lookups,
		"hits(%d) + page_ins(%d) + refusals(%d) = %d vs lookups(%d)",
		ring.Hits, ring.PageIns, ring.Refusals, outcomes, ring.Lookups)

	if ring.BudgetBytes > 0 {
		add("resident-within-budget", ring.ResidentBytes <= ring.BudgetBytes,
			"resident_bytes(%d) <= budget_bytes(%d)", ring.ResidentBytes, ring.BudgetBytes)
		add("peak-within-budget", ring.PeakBytes <= ring.BudgetBytes,
			"peak_bytes(%d) <= budget_bytes(%d)", ring.PeakBytes, ring.BudgetBytes)
	}
	add("coverage-bounded", ring.ActivatedCovered <= ring.ActivatedExperts,
		"activated_covered(%d) <= activated_experts(%d)", ring.ActivatedCovered, ring.ActivatedExperts)

	if sh := rep.Shared; sh != nil {
		// Cross-boundary checks: the shared ledger's hooks and the ring's own counters are separate
		// increments in separate files, so agreement is evidence and disagreement is a bug.
		add("shared-refusals-agree", sh.Refusals == int64(ring.Refusals),
			"shared ledger refusals(%d) vs ring refusals(%d)", sh.Refusals, ring.Refusals)
		add("shared-page-in-bytes-agree", sh.PageInBytes == ring.PageInBytes,
			"shared ledger page_in_bytes(%d) vs ring page_in_bytes(%d)", sh.PageInBytes, ring.PageInBytes)
		// Demands exclude prefetch hints, so they can only be a subset of the ring's stagings; each
		// serve books at most one distinct-agent credit, so serves cannot outrun demands.
		add("shared-demands-bounded", sh.Demands <= int64(ring.Lookups),
			"demands(%d) <= lookups(%d) — prefetch hints are excluded from demands", sh.Demands, ring.Lookups)
		add("shared-serves-bounded", sh.DistinctServes <= sh.Demands,
			"distinct_serves(%d) <= demands(%d)", sh.DistinctServes, sh.Demands)
	}

	if ck := rep.Checkpoint; ck.Enabled && ck.BudgetBytes > 0 {
		add("checkpoint-peak-within-budget", ck.PeakBytes <= ck.BudgetBytes,
			"checkpoint peak_bytes(%d) <= budget_bytes(%d)", ck.PeakBytes, ck.BudgetBytes)
	}

	rc.OK = true
	for _, c := range rc.Checks {
		if !c.OK {
			rc.OK = false
			break
		}
	}
	return rc
}
