package modelroute

import (
	"errors"
	"fmt"
	"math"
	"testing"
)

// quantileOneShotScorer is the tiny inline scorer for this skeleton test: it
// hands back one fixed win rate for every subject, satisfying WinRateScorer.
type quantileOneShotScorer struct{ rate float64 }

func (s quantileOneShotScorer) CalculateStrongWinRate(Subject) (float64, error) {
	return s.rate, nil
}

// quantileEps is the comparison epsilon for calibration values: quantile
// arithmetic lands within float64 rounding noise of the contract formula, and
// 1e-9 sits far above that noise and far below any meaningful score gap here.
const quantileEps = 1e-9

// quantileSliceScorer is the deterministic multi-subject scorer for the
// quantile tests: the test encodes a subject's index in its PromptTokens
// (subject i carries PromptTokens i), and the scorer hands back scores[i] for
// that subject. Keying on the subject rather than on call order keeps
// calibration and any later Route calls over the same reference pointing at
// the same per-subject win rate regardless of call order or repeats.
type quantileSliceScorer struct{ scores []float64 }

func (s quantileSliceScorer) CalculateStrongWinRate(sub Subject) (float64, error) {
	if sub.PromptTokens < 0 || sub.PromptTokens >= len(s.scores) {
		return 0, fmt.Errorf("quantileSliceScorer: subject index %d outside 0..%d", sub.PromptTokens, len(s.scores)-1)
	}
	return s.scores[sub.PromptTokens], nil
}

// quantileReference builds an n-subject reference prompt distribution whose
// i-th subject carries PromptTokens i, matching quantileSliceScorer's index
// encoding.
func quantileReference(n int) []Subject {
	ref := make([]Subject, n)
	for i := range ref {
		ref[i] = Subject{Aspect: AspectRequest, PromptTokens: i}
	}
	return ref
}

// quantileExpect re-derives the contract's numpy-default linear-interpolation
// quantile from an ascending score slice: q = 1-strongPct, idx = q*(n-1),
// value = sorted[floor(idx)] + (idx-floor(idx))*(sorted[ceil(idx)]-sorted[floor(idx)]).
func quantileExpect(sorted []float64, strongPct float64) float64 {
	idx := (1 - strongPct) * float64(len(sorted)-1)
	lo, hi := math.Floor(idx), math.Ceil(idx)
	return sorted[int(lo)] + (idx-lo)*(sorted[int(hi)]-sorted[int(lo)])
}

// quantileErrorScorer always fails with the wrapped error, for the untouched
// propagation assertion: calibration must pass the ORIGINAL error value
// through errors.Is, not a re-wrapped copy that loses identity.
type quantileErrorScorer struct{ err error }

func (s quantileErrorScorer) CalculateStrongWinRate(Subject) (float64, error) {
	return 0, s.err
}

// quantileCountingScorer wraps an inner scorer and counts how many times the
// calibration loop actually consults it.
type quantileCountingScorer struct {
	inner WinRateScorer
	calls int
}

func (c *quantileCountingScorer) CalculateStrongWinRate(s Subject) (float64, error) {
	c.calls++
	return c.inner.CalculateStrongWinRate(s)
}

// TestThresholdQuantileCalibration is the parent test for the
// winrate_calibrate.go calibration loop; deeper quantile cases follow here.
func TestThresholdQuantileCalibration(t *testing.T) {
	t.Run("one-element reference at strong_pct 0.5 returns that score", func(t *testing.T) {
		const want = 0.42
		got, err := CalibrateThreshold(quantileOneShotScorer{rate: want}, []Subject{{}}, 0.5)
		if err != nil {
			t.Fatalf("CalibrateThreshold: %v", err)
		}
		if got != want {
			t.Fatalf("CalibrateThreshold = %v, want %v", got, want)
		}
	})

	t.Run("QuantilePositions", func(t *testing.T) {
		five := []float64{0.1, 0.3, 0.5, 0.7, 0.9}
		four := []float64{0.2, 0.4, 0.6, 0.8}
		cases := []struct {
			label     string
			scores    []float64
			strongPct float64
			want      float64
		}{
			// idx = (1-0.5)*(5-1) = 2.0 -> sorted[2] verbatim.
			{"five scores, strong_pct 0.5 lands on sorted[2.0]", five, 0.5, 0.5},
			// idx = (1-0.25)*(5-1) = 3.0 -> sorted[3] verbatim.
			{"five scores, strong_pct 0.25 lands on sorted[3.0]", five, 0.25, 0.7},
			// idx = (1-0.5)*(4-1) = 1.5 -> midpoint of sorted[1..2] = (0.4+0.6)/2.
			{"four scores, strong_pct 0.5 interpolates sorted[1.5]", four, 0.5, 0.5},
		}
		for _, c := range cases {
			got, err := CalibrateThreshold(quantileSliceScorer{scores: c.scores}, quantileReference(len(c.scores)), c.strongPct)
			if err != nil {
				t.Fatalf("%s: CalibrateThreshold: %v", c.label, err)
			}
			if want := quantileExpect(c.scores, c.strongPct); math.Abs(got-want) > quantileEps {
				t.Fatalf("%s: CalibrateThreshold = %v, want formula value %v", c.label, got, want)
			}
			if math.Abs(got-c.want) > quantileEps {
				t.Fatalf("%s: CalibrateThreshold = %v, want %v", c.label, got, c.want)
			}
		}
	})

	t.Run("DialBoundaries", func(t *testing.T) {
		scores := []float64{0.1, 0.3, 0.5, 0.7, 0.9}
		pair := ModelPair{Strong: "router-strong", Weak: "router-weak"}

		// strong_pct 1 -> q 0 -> idx 0 -> threshold = MIN: every subject meets it.
		r, err := CalibratedThresholdRouter("min-dial", quantileSliceScorer{scores: scores}, pair, quantileReference(len(scores)), 1)
		if err != nil {
			t.Fatalf("strong_pct 1: CalibratedThresholdRouter: %v", err)
		}
		if err := r.Validate(); err != nil {
			t.Fatalf("strong_pct 1: assembled router fails Validate: %v", err)
		}
		if math.Abs(r.Threshold-0.1) > quantileEps {
			t.Fatalf("strong_pct 1: threshold = %v, want MIN 0.1", r.Threshold)
		}
		for i, sub := range quantileReference(len(scores)) {
			if got, err := r.Route(sub); err != nil || got != pair.Strong {
				t.Fatalf("strong_pct 1: Route(subject %d) = (%q, %v), want strong %q", i, got, err, pair.Strong)
			}
		}

		// strong_pct 0 -> q 1 -> idx n-1 -> threshold = MAX: only the max-tied
		// subject meets it, everything else takes the weak tier.
		r, err = CalibratedThresholdRouter("max-dial", quantileSliceScorer{scores: scores}, pair, quantileReference(len(scores)), 0)
		if err != nil {
			t.Fatalf("strong_pct 0: CalibratedThresholdRouter: %v", err)
		}
		if math.Abs(r.Threshold-0.9) > quantileEps {
			t.Fatalf("strong_pct 0: threshold = %v, want MAX 0.9", r.Threshold)
		}
		for i, sub := range quantileReference(len(scores)) {
			want := pair.Weak
			if i == len(scores)-1 {
				want = pair.Strong
			}
			if got, err := r.Route(sub); err != nil || got != want {
				t.Fatalf("strong_pct 0: Route(subject %d) = (%q, %v), want %q", i, got, err, want)
			}
		}

		// strong_pct 0.5 -> idx 2.0 -> threshold 0.5: the tie at 0.5 is INCLUDED
		// (the dispatch rule is >=), so exactly 3 of 5 take the strong tier.
		r, err = CalibratedThresholdRouter("half-dial", quantileSliceScorer{scores: scores}, pair, quantileReference(len(scores)), 0.5)
		if err != nil {
			t.Fatalf("strong_pct 0.5: CalibratedThresholdRouter: %v", err)
		}
		if math.Abs(r.Threshold-0.5) > quantileEps {
			t.Fatalf("strong_pct 0.5: threshold = %v, want 0.5", r.Threshold)
		}
		strong := 0
		for i, sub := range quantileReference(len(scores)) {
			got, err := r.Route(sub)
			if err != nil {
				t.Fatalf("strong_pct 0.5: Route(subject %d): %v", i, err)
			}
			if got == pair.Strong {
				strong++
			}
		}
		if strong != 3 {
			t.Fatalf("strong_pct 0.5: %d of 5 subjects routed strong, want exactly 3 (scores 0.5, 0.7, 0.9 with the tie included)", strong)
		}
	})

	t.Run("FractionProperty", func(t *testing.T) {
		const n = 100
		scores := make([]float64, n)
		for i := range scores {
			scores[i] = 0.005 + 0.01*float64(i)
		}
		for _, strongPct := range []float64{0.2, 0.5, 0.8} {
			got, err := CalibrateThreshold(quantileSliceScorer{scores: scores}, quantileReference(n), strongPct)
			if err != nil {
				t.Fatalf("strong_pct %v: CalibrateThreshold: %v", strongPct, err)
			}
			if want := quantileExpect(scores, strongPct); math.Abs(got-want) > quantileEps {
				t.Fatalf("strong_pct %v: CalibrateThreshold = %v, want formula value %v", strongPct, got, want)
			}
			count := 0
			for _, s := range scores {
				if s >= got {
					count++
				}
			}
			if d := math.Abs(float64(count) - strongPct*100); d > 1 {
				t.Fatalf("strong_pct %v: threshold %v routes %d of %d subjects strong, want within 1 of %v", strongPct, got, count, n, strongPct*100)
			}
		}
	})

	t.Run("SingleSubject", func(t *testing.T) {
		scores := []float64{0.42}
		for _, strongPct := range []float64{0, 0.5, 1} {
			got, err := CalibrateThreshold(quantileSliceScorer{scores: scores}, quantileReference(len(scores)), strongPct)
			if err != nil {
				t.Fatalf("strong_pct %v: CalibrateThreshold: %v", strongPct, err)
			}
			if math.Abs(got-0.42) > quantileEps {
				t.Fatalf("strong_pct %v: CalibrateThreshold = %v, want 0.42", strongPct, got)
			}
		}
	})

	t.Run("Refusals", func(t *testing.T) {
		oneSubject := []Subject{{Aspect: AspectRequest, PromptTokens: 0}}

		// (a) nil scorer: the contract folds "no scorer wired" into the pair
		// refusal vocabulary (a router without a scorer cannot dispatch).
		if _, err := CalibrateThreshold(nil, oneSubject, 0.5); !errors.Is(err, ErrInvalidModelPair) {
			t.Fatalf("nil scorer: err = %v, want ErrInvalidModelPair", err)
		}

		// (b) dials outside [0,1] or non-finite are refused, not clamped.
		for _, dial := range []float64{-0.1, 1.1, math.NaN(), math.Inf(1)} {
			if _, err := CalibrateThreshold(quantileOneShotScorer{rate: 0.5}, oneSubject, dial); !errors.Is(err, ErrInvalidStrongPct) {
				t.Fatalf("dial %v: err = %v, want ErrInvalidStrongPct", dial, err)
			}
		}

		// (c) a nil or empty reference has no quantile to take.
		for _, ref := range [][]Subject{nil, {}} {
			if _, err := CalibrateThreshold(quantileOneShotScorer{rate: 0.5}, ref, 0.5); !errors.Is(err, ErrEmptyReference) {
				t.Fatalf("reference len %d: err = %v, want ErrEmptyReference", len(ref), err)
			}
		}

		// (d) scorer outputs that are not finite probabilities in [0,1] are
		// refused rather than dispatched on.
		for _, rate := range []float64{math.NaN(), math.Inf(1), -0.5, 1.5} {
			if _, err := CalibrateThreshold(quantileOneShotScorer{rate: rate}, oneSubject, 0.5); !errors.Is(err, ErrInvalidWinRate) {
				t.Fatalf("score %v: err = %v, want ErrInvalidWinRate", rate, err)
			}
		}

		// (e) a scorer failure propagates UNTOUCHED: errors.Is must match the
		// ORIGINAL error value, so wrapping must not replace the sentinel.
		original := errors.New("transport down")
		_, err := CalibrateThreshold(quantileErrorScorer{err: original}, oneSubject, 0.5)
		if !errors.Is(err, original) {
			t.Fatalf("scorer error: err = %v, want errors.Is against the original %v", err, original)
		}
	})

	t.Run("OrderInvariantSinglePass", func(t *testing.T) {
		scores := []float64{0.1, 0.9, 0.3, 0.7, 0.5}
		scorer := quantileSliceScorer{scores: scores}
		// The same score multiset presented in two subject orders: the natural
		// order and a fixed permutation. The threshold is a property of the
		// score MULTISET, so both orders must calibrate identically.
		refA := quantileReference(len(scores))
		refB := []Subject{
			{Aspect: AspectRequest, PromptTokens: 4},
			{Aspect: AspectRequest, PromptTokens: 2},
			{Aspect: AspectRequest, PromptTokens: 0},
			{Aspect: AspectRequest, PromptTokens: 3},
			{Aspect: AspectRequest, PromptTokens: 1},
		}
		for _, strongPct := range []float64{0.5, 0.25} {
			gotA, err := CalibrateThreshold(scorer, refA, strongPct)
			if err != nil {
				t.Fatalf("order A, strong_pct %v: CalibrateThreshold: %v", strongPct, err)
			}
			gotB, err := CalibrateThreshold(scorer, refB, strongPct)
			if err != nil {
				t.Fatalf("order B, strong_pct %v: CalibrateThreshold: %v", strongPct, err)
			}
			if math.Abs(gotA-gotB) > quantileEps {
				t.Fatalf("strong_pct %v: order A threshold %v != order B threshold %v", strongPct, gotA, gotB)
			}
		}

		// One calibration scores each reference subject exactly once: a
		// counting wrapper must observe exactly len(reference) scorer calls.
		ref := quantileReference(len(scores))
		counter := &quantileCountingScorer{inner: scorer}
		if _, err := CalibrateThreshold(counter, ref, 0.5); err != nil {
			t.Fatalf("counting calibration: %v", err)
		}
		if counter.calls != len(ref) {
			t.Fatalf("scorer called %d times for a %d-subject reference, want exactly %d", counter.calls, len(ref), len(ref))
		}
	})

	t.Run("AssemblyContract", func(t *testing.T) {
		scores := []float64{0.1, 0.3, 0.5, 0.7, 0.9}
		scorer := quantileSliceScorer{scores: scores}
		ref := quantileReference(len(scores))
		pair := ModelPair{Strong: "strong-a", Weak: "weak-a"}

		r, err := CalibratedThresholdRouter("cal-probe", scorer, pair, ref, 0.5)
		if err != nil {
			t.Fatalf("CalibratedThresholdRouter: %v", err)
		}
		if err := r.Validate(); err != nil {
			t.Fatalf("assembled router fails Validate: %v", err)
		}
		if r.Name != "cal-probe" {
			t.Fatalf("router Name = %q, want %q", r.Name, "cal-probe")
		}
		if r.Pair != pair {
			t.Fatalf("router Pair = %+v, want %+v", r.Pair, pair)
		}
		wantThreshold, err := CalibrateThreshold(scorer, ref, 0.5)
		if err != nil {
			t.Fatalf("CalibrateThreshold (for comparison): %v", err)
		}
		if math.Abs(r.Threshold-wantThreshold) > quantileEps {
			t.Fatalf("router Threshold = %v, want calibrated %v", r.Threshold, wantThreshold)
		}

		// Route dispatches per the inclusive rule AT the calibrated threshold:
		// the subject scoring exactly 0.5 meets it and takes the strong tier;
		// below goes weak, above goes strong.
		if got, err := r.Route(ref[2]); err != nil || got != pair.Strong {
			t.Fatalf("Route(subject at threshold 0.5) = (%q, %v), want strong %q", got, err, pair.Strong)
		}
		if got, err := r.Route(ref[1]); err != nil || got != pair.Weak {
			t.Fatalf("Route(subject below threshold) = (%q, %v), want weak %q", got, err, pair.Weak)
		}
		if got, err := r.Route(ref[3]); err != nil || got != pair.Strong {
			t.Fatalf("Route(subject above threshold) = (%q, %v), want strong %q", got, err, pair.Strong)
		}

		// Assembly refusals: an invalid pair is caught by the final Validate,
		// and the calibration refusals propagate with their own sentinels.
		for _, badPair := range []ModelPair{{}, {Strong: "same", Weak: "same"}} {
			if _, err := CalibratedThresholdRouter("bad-pair", scorer, badPair, ref, 0.5); !errors.Is(err, ErrInvalidModelPair) {
				t.Fatalf("pair %+v: err = %v, want ErrInvalidModelPair", badPair, err)
			}
		}
		if _, err := CalibratedThresholdRouter("bad-dial", scorer, pair, ref, 1.5); !errors.Is(err, ErrInvalidStrongPct) {
			t.Fatalf("dial 1.5 through assembly: err = %v, want ErrInvalidStrongPct", err)
		}
		if _, err := CalibratedThresholdRouter("empty-ref", scorer, pair, nil, 0.5); !errors.Is(err, ErrEmptyReference) {
			t.Fatalf("empty reference through assembly: err = %v, want ErrEmptyReference", err)
		}
	})
}
