package model

import (
	"errors"
	"strings"
)

// expert_ring_pins.go — R2 of the activated-expert offload ladder (#5613, epic #5606,
// docs/MOE-ACTIVATED-OFFLOAD-PLAN.md): give the bounded routed-expert ring (#5611) a DURABLE,
// workload-personalized pin-set instead of the static `pinned=false` R0 wired.
//
// What was missing. #4358 shipped every piece of an online-learning pin cache — a persistent
// per-(layer,expert) usage histogram (crash-safe dump, summed at boot), a warm-start selection
// (WarmStartExpertPins), and a between-turns actuator (RepinPass) — and left them all off the live
// path, because `pagedRing.pins` had no consumer: matMulStaged/stage take the caller's `pinned`
// boolean, and R0's only caller passed a constant false. So the ring was plain LRU, and every
// hot expert paid a cold page-in again after any eviction, on every turn and every restart.
//
// What this closes. Three joins, all on code that already exists:
//
//	consult   — weightHALStagedBounded derives (layer, expert) from the canonical weight name and
//	            passes r.isExpertPinned(layer, expert), so the pool's proven pinned-never-evicted
//	            invariant now protects the WORKLOAD's hot set rather than nothing;
//	observe   — every routed-expert staging folds a touch into the ring's per-turn histogram, so the
//	            prior is built from real routing rather than a profiling run;
//	persist   — ExpertRingEndTurn decays the standing heat, folds the turn in, repins under a
//	            bounded swap cap, and dumps the result crash-safely, so the NEXT process warm-starts
//	            from it.
//
// Default is OFF, byte-for-byte. A session that sets neither ExpertPinBudget nor ExpertUsagePath
// builds no pin-set: isExpertPinned is false for every weight, nothing is observed, and the ring is
// the plain-LRU R0 ring exactly as before.
//
// Still not done here: the ring's VICTIM policy among unpinned residents is still LRU (promoting a
// measured winner is R4/#5615), and a miss is still a synchronous upload (R3/#5614).

// routedExpertIdentity parses the (layer, expert) pair out of a canonical routed-expert weight name
// — model.layers.<l>.mlp.experts.<e>.<proj>.weight (expertName, moe.go). It is the bridge between
// the ring, which is keyed by NAME, and the pin-set / usage histogram, which are keyed by the
// (layer, expert) identity ExpertAccessTraceEvent and ExpertUsageHistogram share: expert 7 in
// adjacent layers names different resident weights, so neither ordinal alone identifies one.
//
// ok is false for anything that is not a routed expert projection — a shared expert
// (.mlp.shared_experts., the #3212 distinction), the router, a dense or attention weight, or a
// routed name whose ordinals do not parse. A caller must treat that as "no pin opinion" rather
// than being handed a silent (0,0), which is a REAL identity: layer 0 expert 0.
func routedExpertIdentity(name string) (layer, expert int, ok bool) {
	if !isRoutedExpertWeight(name) {
		return 0, 0, false
	}
	layer, ok = expertLayerIndex(name)
	if !ok {
		return 0, 0, false
	}
	const seg = ".mlp.experts."
	i := strings.Index(name, seg)
	if i < 0 {
		return 0, 0, false
	}
	rest := name[i+len(seg):]
	j := strings.IndexByte(rest, '.')
	if j <= 0 || j > 9 { // >9 digits is not an expert ordinal; refuse rather than overflow
		return 0, 0, false
	}
	for k := 0; k < j; k++ {
		c := rest[k]
		if c < '0' || c > '9' {
			return 0, 0, false
		}
		expert = expert*10 + int(c-'0')
	}
	return layer, expert, true
}

// expertPinsEnabled reports whether this session asked for a pin-set at all. A pin BUDGET alone
// pins the prior's hot set; a usage PATH alone still observes and dumps, which is how a cold first
// run BUILDS the prior a later run warm-starts from — so either knob is enough to turn the
// machinery on, and neither is enough to change the math.
func (s *Session) expertPinsEnabled() bool {
	return s != nil && (s.ExpertPinBudget > 0 || s.ExpertUsagePath != "")
}

// warmStartExpertPins seeds a freshly built ring's pin-set from the persisted histogram at
// ExpertUsagePath. A MISSING file is the cold first run, not an error: the pin-set is created
// empty and the turn's observations start building the prior. A present-but-corrupt file degrades
// to that same cold start rather than failing the session — but the error is retained and handed
// back at the next turn boundary, so a cache that silently stopped loading is still reportable.
//
// A caller wanting the cross-SESSION fold (several concurrent sessions each dumping their own file)
// sums those files with SumExpertUsageHistograms and points ExpertUsagePath at the result.
func (s *Session) warmStartExpertPins(r *pagedRing) {
	if s == nil || r == nil || !s.expertPinsEnabled() {
		return
	}
	hist := NewExpertUsageHistogram()
	if s.ExpertUsagePath != "" {
		loaded, err := LoadExpertUsageHistogram(s.ExpertUsagePath)
		if err != nil {
			s.expertPinErr = err
		} else {
			hist = loaded
		}
	}
	r.WarmStartPins(hist, s.ExpertPinBudget)
	r.turn = NewExpertUsageHistogram()
}

// observeExpert folds one routed-expert staging into the ring's per-turn usage histogram — the live
// signal ExpertRingEndTurn repins against. It counts a TOUCH per staged projection, so one expert
// activation contributes three (gate/up/down); the actuator ranks by relative heat, so a uniform
// per-projection weighting orders experts identically to a per-activation one.
func (r *pagedRing) observeExpert(layer, expert int) {
	if r == nil || r.turn == nil {
		return
	}
	r.turn.Observe(layer, expert, 1)
}

// fillPins pins the hottest experts carrying heat until the pin-set reaches its budget, and reports
// what it added. It exists because RepinPass only ever SWAPS — it walks the currently pinned experts
// and exchanges the coldest for a hotter candidate — so a set that starts EMPTY stays empty for the
// life of the pass no matter how much heat accumulates. Warm-starting fills the set from a prior, so
// #4358's actuator never had to: but a cold first run has no prior, and without a fill a long-lived
// serve would run its whole life on plain LRU and only start pinning after a restart. Filling an
// empty slot displaces nothing, so it is not churn and is not bounded by the swap cap; it is bounded
// by the budget the operator declared. An expert with no heat is never pinned — an unobserved pin is
// a guess, and the ring's LRU is a better one.
func (p *ExpertPinSet) fillPins() []ExpertPinSwap {
	if p == nil || p.budget <= 0 || len(p.pinned) >= p.budget {
		return nil
	}
	var added []ExpertPinSwap
	for _, k := range p.unpinnedByHeatDesc() {
		if len(p.pinned) >= p.budget {
			break
		}
		heat := p.heat.Count(k.layer, k.expert)
		if heat <= 0 {
			break // ranked hottest-first, so the first cold candidate ends the fill
		}
		p.pinned[k] = true
		added = append(added, ExpertPinSwap{
			OutLayer: -1, OutExpert: -1, // filled a free slot; nothing was displaced
			InLayer: k.layer, InExpert: k.expert, InHeat: heat,
		})
	}
	return added
}

// ExpertRingEndTurn is the between-turns quiescent boundary for the routed-expert ring: it decays
// the standing heat, folds in what THIS turn actually routed, swaps up to maxSwaps of the coldest
// pinned experts for the hottest recent ones (RepinPass, #4358), fills any pin slots still free
// (fillPins), then dumps the updated heat to ExpertUsagePath so the next process warm-starts from
// it. Call it when no forward is in flight.
//
// It returns the pin-set changes performed — the swaps first (coldest eviction first), then the
// fills, which carry OutLayer/OutExpert = -1 because they displaced nothing — so a caller can watch
// the pin-set personalize. The error joins any deferred warm-start load failure with the dump
// failure; the warm-start error is reported once here rather than at load, because a corrupt cache
// must degrade the session, not fail it. A session with no pin-set (the default) is a no-op.
//
// decay in (0,1] is the forgetting term and maxSwaps bounds the churn; both are the caller's
// policy, not the ring's. maxSwaps <= 0 still ages the heat, still fills free slots, and still dumps
// — that is the mode a cold first run uses to BUILD the prior without churning a set it has no
// basis for re-ranking yet.
func (s *Session) ExpertRingEndTurn(decay float64, maxSwaps int) ([]ExpertPinSwap, error) {
	if s == nil || s.expertRing == nil || s.expertRing.pins == nil {
		return nil, nil
	}
	r := s.expertRing
	// R7/#5618: a shared ring's pin-set and heat are cross-agent state, so the repin runs inside a
	// ring span. "No forward in flight" is this session's quiescence; a peer agent's may not be.
	done := s.ringEnter(r)
	defer done()
	swaps := r.RepinPass(r.turn, decay, maxSwaps)
	swaps = append(swaps, r.pins.fillPins()...)
	r.turn = NewExpertUsageHistogram()

	var dumpErr error
	if s.ExpertUsagePath != "" {
		dumpErr = r.pins.heat.Persist(s.ExpertUsagePath)
	}
	warmErr := s.expertPinErr
	s.expertPinErr = nil
	return swaps, errors.Join(warmErr, dumpErr)
}
