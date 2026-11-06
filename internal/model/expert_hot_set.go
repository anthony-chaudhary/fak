package model

import "sort"

// expert_hot_set.go — the ONLINE hot-set learner with hysteresis: fuses EPLB-style activation
// statistics (#3886's per-(layer,expert) activation counts) with the routed-expert residency
// cache's DURABLE pin-set (ExpertPinSet, expert_warmpins.go), so the pin/repin path is seeded by a
// hot-set the workload LEARNED rather than only by what RepinPass happened to swap last turn.
//
// Hysteresis previously existed only in the offline simulation simulateHeatResidency
// (expert_residency_lfu.go), whose admit rule is hysteresisMarginDescriptor — `hot > victim +
// victim/4 + 4`; a live bypass needed a staging contract the HAL does not have, so only the victim
// RANKING went live. But there is a softer place hysteresis can act with no HAL change: the pin-set
// SELECTION. A pin is a durable exemption from eviction, not a bypass, and re-selecting which
// experts stay pinned is a between-turns quiescent decision, exactly the boundary
// ExpertRingEndTurn already owns. So the hysteresis band protects an incumbent PIN from a
// marginally-hotter challenger, killing the oscillating pin/unpin churn the plain
// "swap the coldest for the hottest" actuator (RepinPass) exhibits on a jittery workload — without
// touching the live matMul path at all.
//
// Concretely: a small accumulator of activation statistics (heat, +1 per observed activation) that
// recomputes a committed hot-set of at most `budget` (layer,expert) units under the SAME margin as
// the offline policy. A challenger may only displace the committed COLDEST member when it clears the
// margin; inside the band the incumbent STAYS, which is what makes re-running Learn() on unchanged
// heat return zero swaps. The learned set seeds an ExpertPinSet (SeedPins), giving RepinPass and the
// ring's pinned-never-evicted invariant a learned hot-set to protect.
//
// Default is OFF, byte-for-byte: a session that leaves Session.ExpertHotSetHysteresis at 0 builds no
// learner at all. Determinism: every ranking is total-ordered under the ported rank_learned_profile
// order (resident first, heat desc, warm-start profile rank, (layer, expert) index — see
// expertUnitLess), so the result is independent of Go's randomized map iteration. With no resident
// mark and no loaded profile the decisions degenerate to the legacy ones, with two documented
// default-case deltas: HotSet displays in heat-descending order rather than identity order, and an
// exact-heat victim tie evicts the HIGHEST (layer, expert) index rather than the lowest.
//
// Exported integration point: ExpertHotSetAdvisor (expert_hot_set_advisor.go) implements
// ctxmmu.ExpertResidencyAdvisor, feeding this learner's committed hot-set and displacement band
// into ctxmmu's expert-residency decisions (decode hot-set LRU admission/eviction, prefill
// transient-ring sizing); cmd/hotsetcheck drives that loop on fixtures and reports the
// routed-expert cache hit rate [SW-VERIFIED].

// DefaultExpertHotSetHysteresis is the displacement margin FRACTION that reproduces the offline
// policy's `victim/4` band for integer heat: the offline rule computes `vHeat/4` as integer division
// while this learner scales by the fraction, so the two thresholds agree whenever heat is integral
// (the live path's case). A caller wanting another margin passes it as
// Session.ExpertHotSetHysteresis; 0 disables learning entirely.
const DefaultExpertHotSetHysteresis = 0.25

// hotSetMarginAbs is the constant ADDEND of the admit rule (the `+ 4` in `victim + victim/4 + 4`),
// named so the code and hysteresisMarginDescriptor cannot drift apart silently.
const hotSetMarginAbs = 4.0

// ExpertHotSetSwap is one hysteresis decision: a cold committed unit displaced by a challenger, or
// (for a free slot) a challenger admitted with nothing displaced. OutLayer/OutExpert are -1 on a
// fill, the same convention ExpertPinSwap uses for a free-slot admission. InHeat strictly exceeds
// OutHeat on a displacement, by at least the hysteresis margin; on a fill OutHeat is 0.
type ExpertHotSetSwap struct {
	OutLayer, OutExpert int     // the currently committed coldest unit displaced, or -1,-1 on a fill
	InLayer, InExpert   int     // the challenger admitted
	OutHeat, InHeat     float64 // their heats at the decision
}

// ExpertHotSetLearner is the online hot-set selector: it accumulates EPLB-style activation
// statistics (heat) for every observed (layer,expert) and commits at most `budget` of them as the
// hot-set the pin/repin path should protect. Unlike RepinPass, which walks the CURRENT pin-set and
// swaps pairwise, this ranks the WHOLE observed population and admits/displaces under a hysteresis
// band, so a lukewarm newcomer cannot evict a hot incumbent.
//
// State: heat is the evidence, current is the committed decision. Learn() is the only mutator of
// current; Observe/ObserveTrace are the only mutators of heat. Nothing allocates on a default
// (disabled) session because no learner is built.
type ExpertHotSetLearner struct {
	// budget is how many (layer,expert) units the committed hot-set holds. <= 0 makes Learn a no-op.
	budget int
	// hysteresis is the displacement margin FRACTION; 0 or negative means "use the default". The
	// Session knob uses 0 as "disabled" and therefore never constructs a learner with 0.
	hysteresis float64
	// heat is the accumulated activation statistics â€” the same per-(layer,expert) accumulator the
	// warm-start prior uses, so an expert's learned weight and its persisted prior share a
	// vocabulary. Observe folds one activation; the wiring folds one turn's histogram.
	heat *ExpertUsageHistogram
	// current is the committed hot-set. Membership is the decision; heat is read live from `heat`,
	// so HotSet reports each unit's current heat without a second copy to keep in sync.
	current map[expertUsageKey]bool
	// resident marks the units currently resident in the ring tier — the first partition of the
	// rank_learned_profile order Learn and HotSet rank by. nil means no unit is resident, and the
	// ranking degenerates to heat/prior/index exactly as it did before this field existed.
	resident map[expertUsageKey]bool
	// priorRank holds each unit's position in the loaded warm-start profile; a unit absent from
	// the profile carries no rank and sorts after ranked units at equal residency and heat.
	// nil means no profile was loaded.
	priorRank map[expertUsageKey]int
}

// NewExpertHotSetLearner returns a learner over `budget` hot-set slots using `hysteresis` as the
// displacement margin fraction (0 => DefaultExpertHotSetHysteresis). A budget <= 0 yields a learner
// whose Learn is a no-op — honest, not an error: a caller may build one unconditionally and let the
// budget decide.
func NewExpertHotSetLearner(budget int, hysteresis float64) *ExpertHotSetLearner {
	return &ExpertHotSetLearner{
		budget:     budget,
		hysteresis: hysteresis,
		heat:       NewExpertUsageHistogram(),
		current:    map[expertUsageKey]bool{},
	}
}

// margin is the effective displacement margin fraction: the caller's value when > 0, else the
// shipped default. Negative is treated as the default rather than as an inverted (admit-anything)
// band: a negative margin is a misconfiguration, not a policy.
func (l *ExpertHotSetLearner) margin() float64 {
	if l == nil || l.hysteresis <= 0 {
		return DefaultExpertHotSetHysteresis
	}
	return l.hysteresis
}

// Observe folds one expert activation into the heat (+1) — the EPLB activation-statistics input.
// A negative layer/expert is ignored, the same identity guard ExpertUsageHistogram.Observe applies,
// so a malformed activation cannot poison the learned hot-set.
func (l *ExpertHotSetLearner) Observe(layer, expert int) {
	if l == nil || l.heat == nil || layer < 0 || expert < 0 {
		return
	}
	l.heat.Observe(layer, expert, 1)
}

// ObserveTrace folds one recorded routed-expert window into the heat — the bridge from a recorded
// ExpertAccessTrace (or the ring's per-turn histogram) to the learned statistic.
func (l *ExpertHotSetLearner) ObserveTrace(t ExpertAccessTrace) {
	if l == nil || l.heat == nil {
		return
	}
	l.heat.ObserveTrace(t)
}

// absorb folds one already-built usage histogram (the ring's per-turn accumulator) into the heat —
// the turn-boundary twin of Observe: the live path hands over the whole turn's counters rather than
// re-observing each touch here.
func (l *ExpertHotSetLearner) absorb(h *ExpertUsageHistogram) {
	if l == nil || l.heat == nil || h == nil {
		return
	}
	l.heat.Add(h)
}

// MarkResident declares (layer, expert) resident for subsequent rankings — the first partition of
// the ported rank_learned_profile order. Negative coordinates are ignored, the same identity guard
// Observe applies; marking an already-resident unit is idempotent. Residency partitions OBSERVED
// units only: a never-observed unit carries no heat and is never admitted, so marking one changes
// nothing until the workload actually routes it.
func (l *ExpertHotSetLearner) MarkResident(layer, expert int) {
	if l == nil || layer < 0 || expert < 0 {
		return
	}
	if l.resident == nil {
		l.resident = map[expertUsageKey]bool{}
	}
	l.resident[expertUsageKey{layer, expert}] = true
}

// ClearResident removes one unit from the resident partition; unknown or malformed units are a
// no-op.
func (l *ExpertHotSetLearner) ClearResident(layer, expert int) {
	if l == nil || l.resident == nil || layer < 0 || expert < 0 {
		return
	}
	delete(l.resident, expertUsageKey{layer, expert})
}

// LoadWarmStartProfile records a saved expert profile's (layer, expert) sequence as the prior-rank
// tie-breaker of the ported order: position r in the slice is rank r, and the FIRST occurrence
// fixes a unit's rank (a later duplicate does not move it), matching upstream. Out-of-range pairs
// — negative layer or expert, this learner's identity guard — are ignored, so a malformed row
// cannot poison the table. Count is ignored; the profile carries order only.
func (l *ExpertHotSetLearner) LoadWarmStartProfile(profile []ExpertUsageCount) {
	if l == nil || len(profile) == 0 {
		return
	}
	for r, p := range profile {
		if p.Layer < 0 || p.Expert < 0 {
			continue
		}
		k := expertUsageKey{p.Layer, p.Expert}
		if _, ok := l.priorRank[k]; ok {
			continue // first occurrence already fixed this unit's rank
		}
		if l.priorRank == nil {
			l.priorRank = map[expertUsageKey]int{}
		}
		l.priorRank[k] = r
	}
}

// expertUnitLess reports whether a sorts strictly before b under the learner's total order — the
// rank_learned_profile ordering DIRECT-PORTED from Strata (Niko1221/Strata
// src/core/expert_cache.cpp:141 @ 7695859aa13afb248bdb4d36f9b2e95707db079d, MIT): resident experts
// first, then routing-frequency heat descending, then the prior saved-profile rank ascending (a
// unit absent from the profile sorts after ranked ones), then the (layer, expert) index ascending.
// It is a total order, so a warm-start profile breaks ties deterministically regardless of Go's
// randomized map iteration. Upstream bounds profile pairs by the model's n_layers/n_expert grid;
// the learner is model-agnostic and applies its established identity guard instead — negative
// coordinates are out of range and are refused at the mutators above.
func expertUnitLess(a, b ExpertUsageCount, resident map[expertUsageKey]bool, priorRank map[expertUsageKey]int) bool {
	ka, kb := expertUsageKey{a.Layer, a.Expert}, expertUsageKey{b.Layer, b.Expert}
	if ra, rb := resident[ka], resident[kb]; ra != rb {
		return ra // resident experts first
	}
	if a.Count != b.Count {
		return a.Count > b.Count // routing-frequency heat descending
	}
	pa, oka := priorRank[ka]
	pb, okb := priorRank[kb]
	switch {
	case oka && !okb:
		return true // a ranked profile entry beats an unranked one
	case !oka && okb:
		return false
	case oka && okb && pa != pb:
		return pa < pb // earlier profile rank first
	}
	if a.Layer != b.Layer {
		return a.Layer < b.Layer
	}
	return a.Expert < b.Expert
}

// rankedObserved returns every observed unit in the learner's total order — the ported
// rank_learned_profile order over the population the histogram has seen. With no resident mark and
// no loaded profile it degenerates to the legacy (heat desc, layer asc, expert asc) order, so a
// default session ranks byte-for-byte as before.
func (l *ExpertHotSetLearner) rankedObserved() []ExpertUsageCount {
	ranked := l.heat.sortedCounts()
	sort.Slice(ranked, func(i, j int) bool {
		return expertUnitLess(ranked[i], ranked[j], l.resident, l.priorRank)
	})
	return ranked
}

// Learn is THE hysteresis step: it recomputes the committed hot-set from the accumulated heat and
// returns the swaps performed, coldest displacement first. The rules, in order:
//
//   - Rank every observed unit by the learner's total order (resident first, heat desc, warm-start
//     profile rank, (layer, expert) index — total and deterministic under Go's randomized map
//     iteration; see expertUnitLess).
//   - Free slot (len(current) < budget): admit the BEST uncommitted unit under the ported order
//     (every observed unit carries positive heat, so no heat test is needed). A fill displaces
//     nothing, so it is bounded only by the declared budget, exactly as ExpertPinSet.fillPins is.
//   - Displacement (at capacity): the best uncommitted unit may replace the WORST committed member
//     (the last-committed one under the ported order — see coldestCommitted) ONLY when
//     `challengerHeat > coldHeat + coldHeat*margin + 4` — the same margin as
//     hysteresisMarginDescriptor. Inside that band the incumbent STAYS, which kills the pin/unpin
//     oscillation a pure hottest-vs-coldest swap would produce and makes a second Learn() on
//     unchanged heat return zero swaps.
//
// Termination is guaranteed: a fill strictly grows the set and a displacement strictly increases its
// total heat, so the loop is fuel-bounded; the iteration cap is belt-and-braces. Returns nil (not an
// empty non-nil slice) when nothing changed, so a caller can test length as a change signal.
func (l *ExpertHotSetLearner) Learn() []ExpertHotSetSwap {
	if l == nil || l.heat == nil || l.budget <= 0 {
		return nil
	}
	ranked := l.rankedObserved() // ported rank_learned_profile order over the observed population
	if len(ranked) == 0 {
		return nil
	}
	var swaps []ExpertHotSetSwap
	// Fuel-bounded: at most len(ranked) fills then at most len(ranked) distinct displacements.
	for fuel := 0; fuel <= l.budget+len(ranked); fuel++ {
		if len(l.current) < l.budget {
			challenger, ok := l.hottestUncommitted(ranked)
			if !ok {
				break // no observed unit left to admit
			}
			l.current[expertUsageKey{challenger.Layer, challenger.Expert}] = true
			swaps = append(swaps, ExpertHotSetSwap{
				OutLayer: -1, OutExpert: -1, // filled a free slot; nothing was displaced
				InLayer: challenger.Layer, InExpert: challenger.Expert, InHeat: challenger.Count,
			})
			continue
		}
		cold, coldHeat, ok := l.coldestCommitted()
		if !ok {
			break
		}
		challenger, ok := l.hottestUncommitted(ranked)
		if !ok {
			break
		}
		if challenger.Count <= coldHeat+coldHeat*l.margin()+hotSetMarginAbs {
			break // inside the hysteresis band: the incumbent stays, and nothing hotter is waiting
		}
		delete(l.current, cold)
		l.current[expertUsageKey{challenger.Layer, challenger.Expert}] = true
		swaps = append(swaps, ExpertHotSetSwap{
			OutLayer: cold.layer, OutExpert: cold.expert, OutHeat: coldHeat,
			InLayer: challenger.Layer, InExpert: challenger.Expert, InHeat: challenger.Count,
		})
	}
	return swaps
}

// hottestUncommitted returns the best observed unit not already committed under the learner's total
// order, or ok=false when every observed unit is committed. ranked is in the ported
// rank_learned_profile order, so the first uncommitted row is the challenger — a resident unit
// leads even ahead of a hotter non-resident, which is the point of the ordering. Every observed
// unit carries positive heat by construction (the histogram admits no non-positive weight), so no
// heat test is needed here.
func (l *ExpertHotSetLearner) hottestUncommitted(ranked []ExpertUsageCount) (ExpertUsageCount, bool) {
	for _, c := range ranked {
		if !l.current[expertUsageKey{c.Layer, c.Expert}] {
			return c, true
		}
	}
	return ExpertUsageCount{}, false
}

// coldestCommitted returns the committed unit that sorts LAST under the learner's total order — the
// deterministic eviction candidate. Because residents sort first, a committed resident is never
// chosen while any committed non-resident exists; within the same residency class the lower heat
// loses first, then the later warm-start profile rank, then the higher (layer, expert) index. Map
// iteration is randomized, so the total order (not the iteration) is what makes the choice
// reproducible.
func (l *ExpertHotSetLearner) coldestCommitted() (expertUsageKey, float64, bool) {
	var (
		cold     ExpertUsageCount
		coldHeat float64
		found    bool
	)
	for k := range l.current {
		h := l.heat.Count(k.layer, k.expert)
		cand := ExpertUsageCount{Layer: k.layer, Expert: k.expert, Count: h}
		if !found || expertUnitLess(cold, cand, l.resident, l.priorRank) {
			cold, coldHeat, found = cand, h, true
		}
	}
	if !found {
		return expertUsageKey{}, 0, false
	}
	return expertUsageKey{cold.Layer, cold.Expert}, coldHeat, true
}

// HotSet returns the committed units in the learner's total order (resident first, heat desc,
// warm-start profile rank, (layer, expert) index — see expertUnitLess), each carrying its CURRENT
// heat — the readable snapshot of what the learner would pin right now, the selection a caller
// renders, audits, or hands to SeedPins.
func (l *ExpertHotSetLearner) HotSet() []ExpertUsageCount {
	if l == nil || l.heat == nil {
		return nil
	}
	out := make([]ExpertUsageCount, 0, len(l.current))
	for k := range l.current {
		out = append(out, ExpertUsageCount{Layer: k.layer, Expert: k.expert, Count: l.heat.Count(k.layer, k.expert)})
	}
	sort.Slice(out, func(i, j int) bool {
		return expertUnitLess(out[i], out[j], l.resident, l.priorRank)
	})
	return out
}

// SeedPins seeds an ExpertPinSet from the learned hot-set: it pins every committed unit and folds
// the learner's heat into the pin-set's heat model, so the set is immediately rankable by RepinPass
// rather than starting cold. It is additive — an existing pin not in the learned set is left in
// place — because a caller may combine the learned prior with pins from another source (a persisted
// warm-start, a hold). It does not return the delta; a caller wanting the decision trace reads
// Learn()'s swaps or HotSet().
func (l *ExpertHotSetLearner) SeedPins(p *ExpertPinSet) {
	if l == nil || p == nil || l.heat == nil {
		return
	}
	if p.pinned == nil {
		p.pinned = map[expertUsageKey]bool{}
	}
	if p.heat == nil {
		p.heat = NewExpertUsageHistogram()
	}
	p.heat.Add(l.heat)
	for k := range l.current {
		p.pinned[k] = true
	}
}

// applyHotSetSwaps applies Learn()'s decision to an existing pin-set WITHOUT re-seeding: a
// displacement unpins its cold unit and pins the challenger; a fill pins the challenger. The
// challenger's heat is topped up in the pin-set's heat model only when it carries none, so RepinPass
// has a basis for it without re-adding the learner's whole cumulative heat each turn (which would
// inflate the shared heat model). This is the incremental twin of SeedPins, used by the live turn
// boundary where pins from RepinPass/fillPins must survive.
//
// It is BUDGET-CLAMPED, and that clamping is load-bearing. The learner selects its committed set from
// ITS OWN heat, while the pin-set may already hold unrelated pins from another source — the
// warm-start prior, or RepinPass/fillPins earlier this same boundary. A learner `fill` (nothing
// displaced) has no deletion to offset it, and applying it unconditionally could grow p.pinned past
// p.budget — the exact set the pool's pinned-never-evicted invariant exempts from eviction, so an
// over-budget pin-set makes every subsequent admit fail ErrPinnedNoRoom and silently fall back to
// UNBOUNDED halW residency. The clamp keeps the set within the declared budget: a fill is applied
// only while a slot is free, and a displacement (which frees its own slot first) always fits. The
// learner does not need every fill honored — its committed set is the long-run target, and the next
// turn re-proposes anything this clamp skipped.
func (p *ExpertPinSet) applyHotSetSwaps(swaps []ExpertHotSetSwap) {
	if p == nil || len(swaps) == 0 {
		return
	}
	if p.pinned == nil {
		p.pinned = map[expertUsageKey]bool{}
	}
	if p.heat == nil {
		p.heat = NewExpertUsageHistogram()
	}
	for _, sw := range swaps {
		if sw.OutLayer >= 0 {
			delete(p.pinned, expertUsageKey{sw.OutLayer, sw.OutExpert})
		} else if p.budget > 0 && len(p.pinned) >= p.budget {
			// A fill (OutLayer < 0) must not overrun the declared budget; a displacement is
			// self-bounding (it deletes before it inserts), so only fills are clamped.
			continue
		}
		k := expertUsageKey{sw.InLayer, sw.InExpert}
		p.pinned[k] = true
		if p.heat.Count(sw.InLayer, sw.InExpert) <= 0 && sw.InHeat > 0 {
			p.heat.Observe(sw.InLayer, sw.InExpert, sw.InHeat)
		}
	}
}
