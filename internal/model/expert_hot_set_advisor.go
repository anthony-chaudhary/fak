package model

import "github.com/anthony-chaudhary/fak/internal/expertaccess"

// expert_hot_set_advisor.go — the exported integration point that feeds ExpertHotSetLearner's
// output (hot-set pins with hysteresis) into ctxmmu's expert-residency decisions
// (cluster-bringup P2-4; parent plan fak-private#cluster-bringup-rtx6000-b300-readiness-20261104).
//
// The learner (expert_hot_set.go) and the residency policy (ctxmmu.ExpertResidencyPolicy) are two
// packages that share only the primitive expertaccess descriptor: ctxmmu declares the
// consumer interface, and this file implements it without importing the policy package.
// The serve-path hookup is the documented one-line follow-up: build an advisor over the session's
// pin budget, wire it through ctxmmu.NewExpertResidencyPolicyWithAdvisor, observe routed touches
// into it, call Learn() at the turn boundary, and warm-start from its HotSet(). Until that line
// lands, the loop is fully unit-tested here and by cmd/hotsetcheck, which measures the decisive
// quantity — the routed-expert cache hit rate — on fixtures.
//
// Nothing here changes the learner itself: every method either forwards to it or applies the
// SAME displacement band Learn() commits under, so the gate a caller sees at admission time is
// the exact band the committed hot-set was selected with.

// ExpertHotSetAdvisor adapts ExpertHotSetLearner to ctxmmu.ExpertResidencyAdvisor. It owns no
// state of its own — heat, the committed set, the resident partition, and the warm-start profile
// all live on the learner — so an advisor and a raw learner built with the same inputs decide
// identically.
type ExpertHotSetAdvisor struct {
	l *ExpertHotSetLearner
}

// NewExpertHotSetAdvisor returns an advisor over `budget` hot-set slots using `hysteresis` as the
// displacement margin fraction (0 => DefaultExpertHotSetHysteresis), mirroring
// NewExpertHotSetLearner exactly. A budget <= 0 yields an advisor whose HotSet is always empty and
// whose Admit still answers from the band — honest, not an error.
func NewExpertHotSetAdvisor(budget int, hysteresis float64) *ExpertHotSetAdvisor {
	return &ExpertHotSetAdvisor{l: NewExpertHotSetLearner(budget, hysteresis)}
}

// Learner exposes the underlying learner for callers that need its native surface (Learn's swap
// trace, SeedPins, persistence) alongside the advisor view.
func (a *ExpertHotSetAdvisor) Learner() *ExpertHotSetLearner {
	if a == nil {
		return nil
	}
	return a.l
}

// Observe folds one expert activation into the heat — the EPLB activation-statistics input. A nil
// advisor is a no-op, the same guard the learner's mutators apply.
func (a *ExpertHotSetAdvisor) Observe(layer, expert int) {
	if a == nil || a.l == nil {
		return
	}
	a.l.Observe(layer, expert)
}

// ObserveTrace folds one recorded routed-expert window into the heat.
func (a *ExpertHotSetAdvisor) ObserveTrace(t ExpertAccessTrace) {
	if a == nil || a.l == nil {
		return
	}
	a.l.ObserveTrace(t)
}

// Absorb folds one already-built usage histogram (the ring's per-turn accumulator) into the heat —
// the turn-boundary twin of Observe, the live path's fold.
func (a *ExpertHotSetAdvisor) Absorb(h *ExpertUsageHistogram) {
	if a == nil || a.l == nil {
		return
	}
	a.l.absorb(h)
}

// Learn recomputes the committed hot-set under the hysteresis band and returns the swaps performed
// — the between-turns decision a caller watches for pin stability (zero swaps on unchanged heat is
// the no-thrash witness).
func (a *ExpertHotSetAdvisor) Learn() []ExpertHotSetSwap {
	if a == nil || a.l == nil {
		return nil
	}
	return a.l.Learn()
}

// Heat returns the accumulated activation heat for (layer, expert); 0 for a never-observed unit.
func (a *ExpertHotSetAdvisor) Heat(layer, expert int) float64 {
	if a == nil || a.l == nil {
		return 0
	}
	return a.l.heat.Count(layer, expert)
}

// Admit reports whether a challenger carrying chHeat may displace a resident carrying vHeat under
// the learner's displacement band: admit iff chHeat > vHeat + vHeat*margin + hotSetMarginAbs —
// the strict complement of Learn()'s stay rule (`challenger <= cold + cold*margin + 4` keeps the
// incumbent), so the admission gate and the committed set can never disagree about the band.
func (a *ExpertHotSetAdvisor) Admit(chHeat, vHeat float64) bool {
	if a == nil || a.l == nil {
		return false
	}
	m := a.l.margin()
	return chHeat > vHeat+vHeat*m+hotSetMarginAbs
}

// HotSet returns the committed units, hottest first under the learner's total order, adapted to
// the ctxmmu shape: WeightBytes is 0 because the learner tracks activations, not bytes — the
// policy loads each descriptor from backing when it seeds the ring.
func (a *ExpertHotSetAdvisor) HotSet() []expertaccess.Access {
	if a == nil || a.l == nil {
		return nil
	}
	units := a.l.HotSet()
	out := make([]expertaccess.Access, len(units))
	for i, u := range units {
		out[i] = expertaccess.Access{Layer: u.Layer, Expert: u.Expert}
	}
	return out
}

// ObservedCount is how many distinct (layer, expert) units the learner has ever observed — the
// working-set breadth ctxmmu.SizeTransientRing budgets against.
func (a *ExpertHotSetAdvisor) ObservedCount() int {
	if a == nil || a.l == nil {
		return 0
	}
	return a.l.heat.Len()
}

// MarkResident declares (layer, expert) resident for subsequent rankings — the first partition of
// the learner's total order; the policy calls it whenever a unit enters the hot ring.
func (a *ExpertHotSetAdvisor) MarkResident(layer, expert int) {
	if a == nil || a.l == nil {
		return
	}
	a.l.MarkResident(layer, expert)
}

// ClearResident removes (layer, expert) from the resident partition; the policy calls it whenever
// a unit leaves the hot ring.
func (a *ExpertHotSetAdvisor) ClearResident(layer, expert int) {
	if a == nil || a.l == nil {
		return
	}
	a.l.ClearResident(layer, expert)
}
