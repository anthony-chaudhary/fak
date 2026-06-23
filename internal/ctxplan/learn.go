package ctxplan

import (
	"math"
	"sort"
)

// Outcome is the WITNESSED result of one turn run against a Plan: which resident spans the
// turn actually referenced (Hits), which elided spans it had to demand-page back in
// (Faults — forecast MISSES; a cheap page fault, never a lost fact, because the store is
// lossless), and which resident spans it never touched (Wasted — over-resident, the
// forecast over-predicted).
//
// It is the feedback signal the planner closes its loop over. A Plan is a PREDICTION
// (which spans the next turns will reference); an Outcome is the ground truth that
// prediction is checked against. Forecast.Learn and Weights.Learn turn one Plan+Outcome
// pair into a revised forecast and revised cost constants for the NEXT turn — the online
// rung that keeps the planner honest as a session drifts, instead of frozen at the
// DefaultWeights seed. The whole loop is a forecast MISS costing one demand-page, never a
// lost fact, so learning degrades efficiency when it is wrong, never correctness.
//
// IDs in an Outcome name spans by Cell.ID (the same key Plan.Selected/Plan.Elided carry).
// An id that names no span, or a sealed/tombstoned span, teaches nothing and is skipped
// (fail-closed: a stale or fabricated id cannot poison the learner).
type Outcome struct {
	Hits   []string `json:"hits,omitempty"`   // resident spans the turn referenced (good predictions)
	Faults []string `json:"faults,omitempty"` // elided spans demand-paged back in (forecast MISSES)
	Wasted []string `json:"wasted,omitempty"` // resident spans the turn never touched (over-resident)
}

// maxLearnedIntents bounds a learned forecast's intent list so the prediction stays O(1)
// — it never grows into an unbounded memory of every span ever faulted. New fault-tokens
// that do not fit are dropped (oldest-preserved order: existing intents first).
const maxLearnedIntents = 32

// Learn returns a Forecast whose Intents have been revised from the witnessed Outcome: the
// content-tokens of every FAULTED span (a forecast MISS — the span was elided but the turn
// needed it) are PROMOTED into the intents, so the next plan predicts the span instead of
// faulting it back in. The forecast "learns what to predict" from where it was wrong.
//
// Existing intents are preserved verbatim; only NEW fault-derived content-tokens are
// appended (deduped against what the intents already cover, sorted for determinism, capped
// at maxLearnedIntents). With no faults (or faults that name no learnable span) the
// forecast is returned unchanged — a deterministic no-op, so a turn that confirmed the
// forecast touches nothing. Weights, Horizon, and Pins are carried through unchanged
// (weights are tuned separately via Weights.Learn).
func (f Forecast) Learn(o Outcome, spans []Span) Forecast {
	byID := indexByID(spans)
	// PROMOTE: the content-tokens of every faulted span (a MISS) become intent candidates.
	promoted := map[string]bool{}
	for _, id := range o.Faults {
		s, ok := byID[id]
		if !ok || s.Sealed || s.Tombstoned {
			continue
		}
		for t := range tokenSet(s.Role + " " + s.Descriptor) {
			promoted[t] = true
		}
	}
	if len(promoted) == 0 {
		return f // nothing miss-predicted -> the forecast is unchanged
	}
	// Tokens the current intents already cover (so promotion never duplicates coverage).
	covered := map[string]bool{}
	for _, ph := range f.Intents {
		for t := range tokenSet(ph) {
			covered[t] = true
		}
	}
	promo := make([]string, 0, len(promoted))
	for t := range promoted {
		if !covered[t] {
			promo = append(promo, t)
		}
	}
	if len(promo) == 0 {
		return f // the faults added no token the forecast did not already predict
	}
	sort.Strings(promo) // deterministic append order (map iteration is randomized)
	out := f
	ints := append([]string(nil), f.Intents...)
	for _, t := range promo {
		if len(ints) >= maxLearnedIntents {
			break
		}
		ints = append(ints, t)
	}
	out.Intents = ints
	return out
}

// Learning constants for Weights.Learn. learnRate is the online step size; weightMax is the
// clamp ceiling so a runaway signal (or a long adversarial session) cannot push a weight
// to dominance. Both keep the online update bounded and replay-stable.
const (
	learnRate = 1.0
	weightMax = 10.0
)

// Learn returns Weights tuned from the witnessed Outcome by one online logistic-gradient
// step. Each span the turn NEEDED (a Hit OR a Fault — both are spans the turn actually
// referenced) is a positive label (y=1); each Wasted resident span is a negative label
// (y=0). The four weight knobs move by the average gradient of the logistic loss so the
// score (Forecast.Benefit) better separates needed from wasted: a signal that correlated
// with need is up-weighted, one that did not is down-weighted.
//
// It is deterministic (no randomness, no wall clock), bounded (each weight clamped to
// [0, weightMax]), and fail-closed (a non-finite gradient collapses to a no-op clamp; ids
// naming no span or a sealed/tombstoned span are skipped). With no labeled spans the
// effective weights are returned unchanged. The feature row is Forecast.signals — the SAME
// vector Benefit scores with — so the weight an outcome tunes is exactly the weight that
// scored the span.
func (w Weights) Learn(o Outcome, spans []Span, f Forecast, maxStep int) Weights {
	cur := w.orDefault()
	byID := indexByID(spans)
	type row struct {
		s signal
		y float64
	}
	var rows []row
	add := func(ids []string, y float64) {
		for _, id := range ids {
			s, ok := byID[id]
			if !ok || s.Sealed || s.Tombstoned {
				continue
			}
			rows = append(rows, row{f.signals(s, maxStep), y})
		}
	}
	add(o.Hits, 1.0)
	add(o.Faults, 1.0) // a fault was NEEDED (just not resident) -> positive label too
	add(o.Wasted, 0.0)
	if len(rows) == 0 {
		return cur // nothing witnessed -> effective weights unchanged
	}
	// Average logistic gradient: d/dw [σ(w·x) - y]^2 ~ 2*(σ(w·x)-y)*x; the factor 2 folds
	// into learnRate, leaving (σ(w·x)-y)*x per row.
	var gRel, gUtil, gDur, gRec float64
	for _, r := range rows {
		score := cur.Relevance*r.s.Relevance + cur.Utility*r.s.Utility + cur.Durability*r.s.Durability + cur.Recency*r.s.Recency
		err := sigmoid(score) - r.y
		gRel += err * r.s.Relevance
		gUtil += err * r.s.Utility
		gDur += err * r.s.Durability
		gRec += err * r.s.Recency
	}
	n := float64(len(rows))
	return Weights{
		Relevance:  clampWeight(cur.Relevance - learnRate*(gRel/n)),
		Utility:    clampWeight(cur.Utility - learnRate*(gUtil/n)),
		Durability: clampWeight(cur.Durability - learnRate*(gDur/n)),
		Recency:    clampWeight(cur.Recency - learnRate*(gRec/n)),
	}
}

// clampWeight holds a learned weight in [0, weightMax], failing closed to 0 on NaN and to
// weightMax on +Inf. A negative or non-finite gradient (a poisoned signal) therefore
// cannot push a weight out of range or corrupt the next plan's sort.
func clampWeight(x float64) float64 {
	if math.IsNaN(x) || x < 0 {
		return 0
	}
	if math.IsInf(x, 1) || x > weightMax {
		return weightMax
	}
	return x
}

// sigmoid is the numerically stable logistic function (overflow-safe for large |score|),
// used by Weights.Learn to turn a weighted score into a [0,1] prediction.
func sigmoid(x float64) float64 {
	if x >= 0 {
		z := math.Exp(-x)
		return 1 / (1 + z)
	}
	z := math.Exp(x)
	return z / (1 + z)
}

// indexByID builds a Cell.ID -> Span lookup over a candidate set, so a learner can resolve
// the witnessed Outcome's ids to the spans (and their signals) without a second scan.
func indexByID(spans []Span) map[string]Span {
	m := make(map[string]Span, len(spans))
	for _, s := range spans {
		m[s.ID] = s
	}
	return m
}
