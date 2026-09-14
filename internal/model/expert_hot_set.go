package model

import "sort"

// expert_hot_set.go â€” the ONLINE hot-set learner with hysteresis: the missing live leaf that fuses
// EPLB-style activation statistics (#3886's per-(layer,expert) activation counts) with the
// routed-expert residency cache's DURABLE pin-set (ExpertPinSet, expert_warmpins.go), so the pin /
// repin path is seeded by a hot-set the workload LEARNED rather than only by what RepinPass happened
// to swap last turn.
//
// The gap this closes. Admission hysteresis exists today in exactly ONE place: the offline
// simulation `simulateHeatResidency` (expert_residency_lfu.go), whose admit rule is
// hysteresisMarginDescriptor â€” `hot > victim + victim/4 + 4`. expert_ring_policy.go's header names
// the reason it never went live: a bypass needs a staging contract the HAL does not have, so only
// the victim RANKING went live and hysteresis stayed simulated. But that framing is about the RING's
// per-stage admission. There is a second, softer place hysteresis can act with no HAL change at all:
// the pin-set SELECTION. A pin is a durable exemption from eviction, not a bypass â€” the pool already
// serves a pinned weight, and re-selecting WHICH experts stay pinned is a between-turns quiescent
// decision, exactly the boundary ExpertRingEndTurn already owns. So the hysteresis band can protect
// an incumbent PIN from a marginally-hotter challenger, killing the oscillating pin/unpin churn the
// plain "swap the coldest for the hottest" actuator (RepinPass) exhibits on a jittery workload â€”
// without touching the live matMul path at all.
//
// What this file is, precisely: a small accumulator of EPLB-style activation statistics (heat, +1
// per observed activation) that recomputes a committed hot-set of at most `budget` (layer,expert)
// units under the SAME margin as the offline policy. A challenger may only displace the currently
// committed COLDEST member when it clears the margin; inside the band the incumbent STAYS. This is
// what makes re-running Learn() on unchanged heat return zero swaps (the no-oscillation property).
// The learned set can then seed an ExpertPinSet (SeedPins), giving RepinPass and the ring's
// pinned-never-evicted invariant a learned hot-set to protect.
//
// Default is OFF, byte-for-byte. A session that leaves Session.ExpertHotSetHysteresis at 0 builds no
// learner at all (routedExpertRing's pins path is unchanged), so every existing witness and every
// default session is inert. Determinism: every ranking is total-ordered (heat, then layer, then
// expert), so the result is independent of Go's randomized map iteration.

// DefaultExpertHotSetHysteresis is the displacement margin FRACTION that reproduces the offline
// policy's `victim/4` band for integer heat. The offline rule computes `vHeat/4` as integer
// division while this learner scales by the fraction, so the two thresholds agree whenever heat is
// integral (the live path's case, where heat counts activations). A caller wanting the shipped #4357
// margin passes it (or any other >0 fraction) as Session.ExpertHotSetHysteresis; 0 disables
// learning entirely.
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
// The state is deliberately small and explicit: heat is the evidence, current is the committed
// decision. Learn() is the only mutator of current; Observe/ObserveTrace are the only mutators of
// heat. Nothing here allocates on a default (disabled) session because no learner is built.
type ExpertHotSetLearner struct {
	// budget is how many (layer,expert) units the committed hot-set holds. <= 0 makes Learn a no-op.
	budget int
	// hysteresis is the displacement margin FRACTION. 0 means "use the default" (default margin =
	// the shipped victim/4 term); the Session knob uses 0 as "disabled" and therefore never
	// constructs a learner with 0. Negative is treated as the default margin.
	hysteresis float64
	// heat is the accumulated activation statistics â€” the same per-(layer,expert) accumulator the
	// warm-start prior uses, so an expert's learned weight and its persisted prior share a
	// vocabulary. Observe folds one activation; the wiring folds one turn's histogram.
	heat *ExpertUsageHistogram
	// current is the committed hot-set. Membership is the decision; heat is read live from `heat`,
	// so HotSet reports each unit's current heat without a second copy to keep in sync.
	current map[expertUsageKey]bool
}

// NewExpertHotSetLearner returns a learner over `budget` hot-set slots using `hysteresis` as the
// displacement margin fraction (0 => DefaultExpertHotSetHysteresis). A budget <= 0 yields a learner
// whose Learn is a no-op â€” honest, not an error: a caller may build one unconditionally and let the
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
// band, because a negative margin is a misconfiguration, not a policy.
func (l *ExpertHotSetLearner) margin() float64 {
	if l == nil || l.hysteresis <= 0 {
		return DefaultExpertHotSetHysteresis
	}
	return l.hysteresis
}

// Observe folds one expert activation into the heat (+1) â€” the EPLB activation-statistics input.
// A negative layer/expert is ignored, the same identity guard ExpertUsageHistogram.Observe applies,
// so a malformed activation cannot poison the learned hot-set.
func (l *ExpertHotSetLearner) Observe(layer, expert int) {
	if l == nil || l.heat == nil || layer < 0 || expert < 0 {
		return
	}
	l.heat.Observe(layer, expert, 1)
}

// ObserveTrace folds one recorded routed-expert window into the heat â€” the bridge from a recorded
// ExpertAccessTrace (or the ring's per-turn histogram) to the learned statistic.
func (l *ExpertHotSetLearner) ObserveTrace(t ExpertAccessTrace) {
	if l == nil || l.heat == nil {
		return
	}
	l.heat.ObserveTrace(t)
}

// absorb folds one already-built usage histogram (the ring's per-turn accumulator) into the heat.
// It is the turn-boundary twin of Observe: the live path stages many projections and hands over the
// whole turn's counters rather than re-observing each touch here.
func (l *ExpertHotSetLearner) absorb(h *ExpertUsageHistogram) {
	if l == nil || l.heat == nil || h == nil {
		return
	}
	l.heat.Add(h)
}

// Learn is THE hysteresis step: it recomputes the committed hot-set from the accumulated heat and
// returns the swaps performed, coldest displacement first. The rules, in order:
//
//   - Rank every observed unit by heat desc (ties by layer asc, then expert asc â€” total and
//     deterministic under Go's randomized map iteration).
//   - Free slot (len(current) < budget): admit the hottest uncommitted unit carrying heat > 0.
//     A fill displaces nothing, so it is not bounded by the hysteresis band; it is bounded by the
//     declared budget, exactly as ExpertPinSet.fillPins is.
//   - Displacement (at capacity): the hottest uncommitted unit may replace the currently committed
//     COLDEST member ONLY when `challengerHeat > coldHeat + coldHeat*margin + 4` â€” the same margin
//     as hysteresisMarginDescriptor. Inside that band the incumbent STAYS. This is what kills the
//     pin/unpin oscillation a pure hottest-vs-coldest swap would produce on a jittery workload, and
//     what makes a second Learn() on unchanged heat return zero swaps: once stable, no uncommitted
//     unit clears the margin, so the loop breaks with nothing performed.
//
// Termination is guaranteed: a fill strictly grows the set and a displacement strictly increases its
// total heat (the challenger beats the cold member it replaces), so the loop is fuel-bounded; the
// iteration cap is belt-and-braces on top of that argument. Returns nil (not an empty non-nil slice)
// when nothing changed, so a caller can test length as a change signal.
func (l *ExpertHotSetLearner) Learn() []ExpertHotSetSwap {
	if l == nil || l.heat == nil || l.budget <= 0 {
		return nil
	}
	ranked := l.heat.sortedCounts() // heat desc, layer asc, expert asc
	if len(ranked) == 0 {
		return nil
	}
	var swaps []ExpertHotSetSwap
	// fuel bounds the loop: every iteration is a fill or a strictly-heat-increasing displacement,
	// and at most len(ranked) fills then at most len(ranked) distinct displacements can occur.
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

// hottestUncommitted returns the highest-heat observed unit not already committed, or ok=false when
// every remaining unit is committed or carries no heat. ranked is heat-descending, so the first
// uncommitted row is the hottest; a heat of 0 there ends the scan because every later row is <= 0.
func (l *ExpertHotSetLearner) hottestUncommitted(ranked []ExpertUsageCount) (ExpertUsageCount, bool) {
	for _, c := range ranked {
		if c.Count <= 0 {
			return ExpertUsageCount{}, false
		}
		if !l.current[expertUsageKey{c.Layer, c.Expert}] {
			return c, true
		}
	}
	return ExpertUsageCount{}, false
}

// coldestCommitted returns the committed unit with the lowest heat, tie-broken by layer asc then
// expert asc â€” the deterministic eviction candidate. Map iteration is randomized, so the total
// order (not the iteration) is what makes the choice reproducible.
func (l *ExpertHotSetLearner) coldestCommitted() (expertUsageKey, float64, bool) {
	var (
		cold     expertUsageKey
		coldHeat float64
		found    bool
	)
	for k := range l.current {
		h := l.heat.Count(k.layer, k.expert)
		switch {
		case !found, h < coldHeat,
			h == coldHeat && k.layer < cold.layer,
			h == coldHeat && k.layer == cold.layer && k.expert < cold.expert:
			cold, coldHeat, found = k, h, true
		}
	}
	return cold, coldHeat, found
}

// HotSet returns the committed units in deterministic (layer asc, expert asc) order, each carrying
// its CURRENT heat. It is the readable snapshot of what the learner would pin right now â€” the
// selection a caller renders, audits, or hands to SeedPins.
func (l *ExpertHotSetLearner) HotSet() []ExpertUsageCount {
	if l == nil || l.heat == nil {
		return nil
	}
	out := make([]ExpertUsageCount, 0, len(l.current))
	for k := range l.current {
		out = append(out, ExpertUsageCount{Layer: k.layer, Expert: k.expert, Count: l.heat.Count(k.layer, k.expert)})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Layer != out[j].Layer {
			return out[i].Layer < out[j].Layer
		}
		return out[i].Expert < out[j].Expert
	})
	return out
}

// SeedPins seeds an ExpertPinSet from the learned hot-set: it pins every committed unit and folds
// the learner's heat into the pin-set's heat model, so the set is immediately rankable by RepinPass
// rather than starting cold. It is additive â€” an existing pin not in the learned set is left in
// place â€” because a caller may combine the learned prior with pins from another source (a persisted
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
// challenger's heat is topped up in the pin-set's heat model only when it carries none, so
// RepinPass has a basis for it without re-adding the learner's whole cumulative heat each turn (which
// would inflate the shared heat model). This is the incremental twin of SeedPins, used by the live
// turn boundary where pins from RepinPass/fillPins must survive.
//
// It is BUDGET-CLAMPED, and that clamping is load-bearing. The learner selects its committed set from
// ITS OWN heat (what it observed), while the pin-set may already hold unrelated pins from another
// source -- the warm-start prior, or RepinPass/fillPins earlier this same boundary. A learner `fill`
// (nothing displaced) therefore has no deletion to offset it, and applying it unconditionally could
// grow p.pinned past p.budget -- the exact set the pool's pinned-never-evicted invariant exempts from
// eviction, so an over-budget pin-set makes every subsequent admit fail ErrPinnedNoRoom and silently
// fall back to UNBOUNDED halW residency. The clamp keeps the set within the declared budget: a fill is
// applied only while a slot is free, and a displacement (which frees its own slot first) always fits.
// The learner does not need every fill honored -- its committed set is the long-run target, and the
// next turn re-proposes anything this clamp skipped.
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
			// A fill (OutLayer < 0) must not overrun the declared budget: when a disjoint warm prior
			// already filled every slot, pinning the challenger here would grow the set past p.budget.
			// A displacement is self-bounding (it deletes before it inserts), so only fills are clamped.
			continue
		}
		k := expertUsageKey{sw.InLayer, sw.InExpert}
		p.pinned[k] = true
		if p.heat.Count(sw.InLayer, sw.InExpert) <= 0 && sw.InHeat > 0 {
			p.heat.Observe(sw.InLayer, sw.InExpert, sw.InHeat)
		}
	}
}
