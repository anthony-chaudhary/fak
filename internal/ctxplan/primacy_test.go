package ctxplan

import (
	"math"
	"testing"
)

// The primacy term is the experimental 'remove-the-middle' prior: OFF by default,
// and when ON it favors BOTH ends (oldest + newest) over the middle. These tests
// pin the invariants the design rests on.

// neutralSpan is a span with no relevance/utility/durability signal, so its Benefit
// is driven purely by the step-positional terms (recency + primacy) -- the cleanest
// probe for the positional shape.
func neutralSpan(id string, step int) Span {
	return Span{ID: id, Role: "tool", Descriptor: "zzz", Step: step, Durability: DurabilityDurable}
}

// DEFAULT OFF: with the default weights (Primacy == 0), the score is byte-identical
// to what it was before the primacy term existed -- i.e. primacy contributes nothing.
func TestPrimacyDefaultOff(t *testing.T) {
	if DefaultWeights().Primacy != 0 {
		t.Fatalf("default Primacy weight must be 0 (experiment off), got %v", DefaultWeights().Primacy)
	}
	f := Forecast{Intents: []string{"refund"}} // default weights
	maxStep := 100
	old := Span{ID: "old", Role: "tool", Descriptor: "refund detail", Step: 1, Durability: DurabilitySession}
	mid := Span{ID: "mid", Role: "tool", Descriptor: "refund detail", Step: 50, Durability: DurabilitySession}
	// With Primacy off, an older span is NOT favored over a middle one on the primacy
	// axis (only recency, which favors newer). So old must score <= mid here.
	if f.Benefit(old, maxStep) > f.Benefit(mid, maxStep) {
		t.Fatalf("with primacy off, the older span must not outscore the middle: old=%v mid=%v",
			f.Benefit(old, maxStep), f.Benefit(mid, maxStep))
	}
}

// primacy() is the exact opposite-end mirror of recency over the open interval, and
// pinned to 0 at both ends.
func TestPrimacyHelperShape(t *testing.T) {
	maxStep := 10
	// degenerate ends -> 0
	if primacy(0, maxStep) != 0 {
		t.Fatalf("primacy at step 0 must be 0")
	}
	if primacy(maxStep, maxStep) != 0 {
		t.Fatalf("primacy at the newest step must be 0 (never lifts the newest span)")
	}
	if primacy(5, 0) != 0 {
		t.Fatalf("primacy with maxStep<=0 must be 0")
	}
	// interior: 1 - recency
	for _, step := range []int{1, 3, 5, 7, 9} {
		want := 1 - recency(step, maxStep)
		if math.Abs(primacy(step, maxStep)-want) > 1e-12 {
			t.Fatalf("primacy(%d,%d)=%v, want 1-recency=%v", step, maxStep, primacy(step, maxStep), want)
		}
	}
	// monotone: older (lower step) has higher primacy
	if primacy(1, maxStep) <= primacy(9, maxStep) {
		t.Fatalf("older step must have higher primacy: p(1)=%v p(9)=%v", primacy(1, maxStep), primacy(9, maxStep))
	}
}

// ON: with a non-zero primacy weight, the OLDEST span is lifted RELATIVE TO ITS OWN
// score without primacy -- the honest 'what primacy adds' comparison. (It does not by
// itself beat the middle: see the U-shape note below -- a LINEAR primacy+recency sum is
// flat, so primacy raises the old end's absolute score, restoring the symmetry recency
// alone broke.)
func TestPrimacyOnLiftsOldEnd(t *testing.T) {
	maxStep := 100
	old := neutralSpan("old", 1)
	off := Forecast{Weights: Weights{Recency: 0.2}}              // primacy OFF
	on := Forecast{Weights: Weights{Recency: 0.2, Primacy: 0.2}} // primacy ON
	if on.Benefit(old, maxStep) <= off.Benefit(old, maxStep) {
		t.Fatalf("primacy must lift the oldest span's score: off=%v on=%v",
			off.Benefit(old, maxStep), on.Benefit(old, maxStep))
	}
}

// THE HONEST SHAPE FINDING: a LINEAR primacy+recency sum at EQUAL weights is FLAT, not
// U -- because primacy == 1-recency, so w*recency + w*primacy == w for every step. A
// true U (middle strictly below both ends) needs either ASYMMETRIC weights or a CONVEX
// positional transform. This test PINS that finding so a future 'we made a U' claim
// must change the math, not just the weights. (This is the design's biggest subtlety,
// surfaced by the adversarial design pass.)
func TestLinearPrimacyRecencyIsFlatNotU(t *testing.T) {
	f := Forecast{Weights: Weights{Recency: 0.5, Primacy: 0.5}} // equal ends, linear
	maxStep := 100
	oldEnd := f.Benefit(neutralSpan("old", 1), maxStep)
	middle := f.Benefit(neutralSpan("mid", 50), maxStep)
	newEnd := f.Benefit(neutralSpan("new", 99), maxStep)
	// flat: all three within rounding of each other -- NOT a U.
	if math.Abs(oldEnd-middle) > 0.02 || math.Abs(newEnd-middle) > 0.02 {
		t.Fatalf("expected a FLAT positional sum at equal weights (linear primacy+recency), "+
			"got old=%v mid=%v new=%v -- if this changed, the U math changed", oldEnd, middle, newEnd)
	}
}

// A real U emerges only with ASYMMETRIC weights: weight the old end MORE than the new
// (or vice versa) and the middle dips below the heavier end. This is the actual lever an
// operator pulls to 'remove the middle' under the linear term.
func TestAsymmetricWeightsTiltAwayFromMiddle(t *testing.T) {
	// Heavily favor the OLD end: a steep primacy, mild recency. The middle now scores
	// below the old end (the dominant end), which is the usable 'keep the framing, drop
	// the middle' tilt.
	f := Forecast{Weights: Weights{Recency: 0.1, Primacy: 0.6}}
	maxStep := 100
	oldEnd := f.Benefit(neutralSpan("old", 1), maxStep)
	middle := f.Benefit(neutralSpan("mid", 50), maxStep)
	if middle >= oldEnd {
		t.Fatalf("with the old end weighted heavier, the middle (%v) must dip below the old end (%v)",
			middle, oldEnd)
	}
}

// INVARIANT: a sealed/tombstoned span still scores exactly 0 even with primacy on --
// the positional prior never lifts a span the trust gate refuses.
func TestPrimacyNeverLiftsSealed(t *testing.T) {
	f := Forecast{Weights: Weights{Primacy: 1.0}}
	sealed := Span{ID: "s", Step: 1, Sealed: true, Durability: DurabilityDurable}
	tomb := Span{ID: "t", Step: 1, Tombstoned: true, Durability: DurabilityDurable}
	if got := f.Benefit(sealed, 100); got != 0 {
		t.Fatalf("sealed span must score 0 even with primacy on, got %v", got)
	}
	if got := f.Benefit(tomb, 100); got != 0 {
		t.Fatalf("tombstoned span must score 0 even with primacy on, got %v", got)
	}
}

// INVARIANT: a zero-signal span (no relevance/utility/durability and step at the new
// end where primacy==0) still scores 0 -- the term is additive, never a floor.
func TestPrimacyZeroBenefitStaysZero(t *testing.T) {
	f := Forecast{Weights: Weights{Primacy: 1.0}} // primacy only
	// step == maxStep: primacy is 0 there; no other signal -> benefit 0.
	s := Span{ID: "z", Role: "x", Descriptor: "x", Step: 100, Durability: DurabilityTurn}
	// durability is 0.2 default-prior but weight is 0 here, so the only weighted term is
	// primacy, which is 0 at the newest step.
	if got := f.Benefit(s, 100); got != 0 {
		t.Fatalf("a newest-step span under primacy-only weights must score 0, got %v", got)
	}
}

// COUPLING: the learner's score mirrors Benefit (it includes the primacy term), so the
// two never drift. We assert that turning primacy on changes the learned gradient path
// -- i.e. Learn actually reads the primacy signal.
func TestLearnerSeesPrimacy(t *testing.T) {
	spans := []Span{neutralSpan("old", 1), neutralSpan("new", 99)}
	o := Outcome{Hits: []string{"old"}, Wasted: []string{"new"}}
	// With primacy on, the old span (high primacy) is a HIT and the new (low primacy) is
	// WASTED, so the gradient should not leave Primacy untouched at its seeded value.
	w := Weights{Relevance: 1.0, Recency: 0.2, Primacy: 0.2}
	learned := w.Learn(o, spans, Forecast{Weights: w}, 100)
	if learned.Primacy == w.Primacy {
		t.Fatalf("learner did not move the primacy weight; scorer/learner are drifting (was %v, still %v)",
			w.Primacy, learned.Primacy)
	}
}
