package logitoracle

import (
	"errors"
	"math"
	"testing"
)

func near(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// fak-test:runtime fast est=5ms lane=default
func TestComparePositionHandComputedFixture(t *testing.T) {
	ref := []float32{3, 1, 0, -1}
	cand := []float32{1, 3, 0, -1}
	st, err := ComparePosition(ref, cand, 2, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if !near(st.Cosine, 7.0/11.0, 1e-12) || !near(st.RelL2, math.Sqrt(8.0/11.0), 1e-12) || st.MaxAbs != 2 {
		t.Fatalf("cos=%v rel=%v max=%v", st.Cosine, st.RelL2, st.MaxAbs)
	}
	if st.RefTop1 != 0 || st.CandTop1 != 1 || st.Top1Agree || st.RefGap != 2 || st.CandGap != 2 || st.RefTop2Margin != 2 {
		t.Fatalf("ranking %+v", st)
	}
	if st.Tie || st.TopKOverlap != 1 {
		t.Fatalf("tie=%v overlap=%v", st.Tie, st.TopKOverlap)
	}
	wide, err := ComparePosition(ref, cand, 2, 2)
	if err != nil || !wide.Tie {
		t.Fatalf("eps=2 must make the swap a tie: %+v %v", wide, err)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestComparePositionIdenticalIsExact(t *testing.T) {
	v := []float32{0.5, -2, 7, 7, 1}
	st, err := ComparePosition(v, v, 5, 0)
	if err != nil {
		t.Fatal(err)
	}
	if st.Cosine != 1 || st.RelL2 != 0 || st.MaxAbs != 0 || st.KL != 0 || !st.Top1Agree || st.TopKOverlap != 1 || st.RefTop1 != 2 {
		t.Fatalf("%+v", st)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestComparePositionTieNeedsBothGaps(t *testing.T) {
	ref := []float32{2.00, 1.95, 0}
	tie, err := ComparePosition(ref, []float32{1.96, 2.00, 0}, 2, 0.1)
	if err != nil || !tie.Tie || tie.Top1Agree {
		t.Fatalf("near tie: %+v %v", tie, err)
	}
	lopsided, err := ComparePosition(ref, []float32{1.0, 2.0, 0}, 2, 0.1)
	if err != nil || lopsided.Tie {
		t.Fatalf("candidate gap 1.0 is not a tie: %+v %v", lopsided, err)
	}
	if !near(lopsided.RefGap, 0.05, 1e-6) || !near(lopsided.CandGap, 1.0, 1e-6) {
		t.Fatalf("gaps %+v", lopsided)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestComparePositionKLFixture(t *testing.T) {
	st, err := ComparePosition([]float32{0, 0}, []float32{float32(math.Log(3)), 0}, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if want := 0.5 * math.Log(4.0/3.0); !near(st.KL, want, 1e-6) {
		t.Fatalf("kl=%v want %v", st.KL, want)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestComparePositionRejectsMalformedVectors(t *testing.T) {
	cases := []struct {
		ref, cand []float32
		want      error
	}{
		{nil, []float32{1}, ErrEmptyLogits},
		{[]float32{1, 2}, []float32{1}, ErrLengthMismatch},
		{[]float32{1, float32(math.NaN())}, []float32{1, 2}, ErrNonFinite},
		{[]float32{1, 2}, []float32{float32(math.Inf(1)), 2}, ErrNonFinite},
	}
	for _, c := range cases {
		if _, err := ComparePosition(c.ref, c.cand, 1, 0); !errors.Is(err, c.want) {
			t.Fatalf("got %v want %v", err, c.want)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestTopKBreaksTiesByLowerIndex(t *testing.T) {
	got := TopK([]float32{1, 5, 5, 2, 5}, 3)
	if len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 4 {
		t.Fatalf("got %v", got)
	}
	if got := TopK([]float32{4, 3}, 5); len(got) != 2 || got[0] != 0 {
		t.Fatalf("k beyond length: %v", got)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestFoldGreedyAndGateVerdicts(t *testing.T) {
	th := Thresholds{CosineMin: 0.999, RelL2Max: 0.05, TieEpsilon: 0.1, TopK: 5}
	exact := PositionStats{Cosine: 1, RelL2: 0, Top1Agree: true}
	small := PositionStats{Position: 1, Cosine: 0.9995, RelL2: 0.01, Top1Agree: true}
	tie := PositionStats{Position: 2, Cosine: 0.9999, RelL2: 0.005, Tie: true}
	real := PositionStats{Position: 3, Cosine: 0.9999, RelL2: 0.005}

	acc := NewAccumulator("p")
	for _, st := range []PositionStats{exact, small, tie} {
		acc.Add(st)
	}
	r := acc.Fold(3, 0, th)
	if r.Greedy != GreedyTieEquivalent || !r.GatePass || r.StrictPass || r.Top1Agree != 2 || len(r.Mismatches) != 1 {
		t.Fatalf("tie fold %+v", r)
	}
	if r.Worst.Position != 1 || r.CosineMin != 0.9995 || r.RelL2Max != 0.01 || r.RelL2Median != 0.005 {
		t.Fatalf("aggregates %+v", r)
	}
	acc.Add(real)
	if r := acc.Fold(4, 0, th); r.Greedy != GreedyNotEquivalent || r.GatePass {
		t.Fatalf("real mismatch must fail: %+v", r)
	}
	strict := NewAccumulator("s")
	strict.Add(exact)
	if r := strict.Fold(1, 0, th); !r.StrictPass || !r.GatePass || r.Greedy != GreedyIdentical {
		t.Fatalf("exact fold %+v", r)
	}
	loose := NewAccumulator("l")
	loose.Add(PositionStats{Cosine: 0.99, RelL2: 0.2, Top1Agree: true})
	if r := loose.Fold(1, 0, th); r.GatePass {
		t.Fatalf("threshold breach must fail: %+v", r)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestFoldReportVerdicts(t *testing.T) {
	pass := PromptReport{GatePass: true, StrictPass: true, Greedy: GreedyIdentical}
	tie := PromptReport{GatePass: true, Greedy: GreedyTieEquivalent}
	fail := PromptReport{Greedy: GreedyNotEquivalent}
	if r := FoldReport([]PromptReport{pass, tie}, FlipThresholds); r.Verdict != VerdictPass || r.StrictVerdict != VerdictFail || r.Greedy != GreedyTieEquivalent {
		t.Fatalf("%+v", r)
	}
	if r := FoldReport([]PromptReport{pass, fail}, FlipThresholds); r.Verdict != VerdictFail || r.Greedy != GreedyNotEquivalent {
		t.Fatalf("%+v", r)
	}
	partial := pass
	partial.Excluded = 3
	if r := FoldReport([]PromptReport{partial}, FlipThresholds); r.Verdict != VerdictIncomplete || r.StrictVerdict != VerdictIncomplete {
		t.Fatalf("%+v", r)
	}
	if r := FoldReport(nil, FlipThresholds); r.Verdict != VerdictFail {
		t.Fatalf("%+v", r)
	}
}
