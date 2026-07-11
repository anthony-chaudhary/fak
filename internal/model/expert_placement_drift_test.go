package model

import "testing"

// TestScoreExpertPlacementCoverageAndDrift pins the two orthogonal terms of ScoreExpertPlacement to
// their definitions on a hand-built histogram: Coverage == served/total observed touches, and
// Drift == 1 - |planned ∩ observed_topk|/k. The three placements share one freq histogram whose
// top-2 hot set is unambiguously {expert 0, expert 1} (12 > 9 > 6 > 2 > 1), so the mask==top-k,
// disjoint, and partial cases exercise Drift 0, 1, and 0.5 exactly.
func TestScoreExpertPlacementCoverageAndDrift(t *testing.T) {
	// freq indexed by expert id. Distinct values => an unambiguous top-k with no tie-break ambiguity.
	freq := []int64{12, 9, 6, 2, 1} // total = 30; top-2 = {0,1}
	const k = 2
	const total = 30

	cases := []struct {
		name        string
		mask        []bool
		wantServed  int64
		wantOverlap int // |planned ∩ observed_topk|
	}{
		// resident set == observed top-k => no drift.
		{"mask equals top-k", []bool{true, true, false, false, false}, 12 + 9, 2},
		// resident set disjoint from the top-k => full drift.
		{"mask disjoint from top-k", []bool{false, false, false, true, true}, 2 + 1, 0},
		// resident set overlaps the top-k in exactly one member => half drift.
		{"mask overlaps top-k by one", []bool{true, false, true, false, false}, 12 + 6, 1},
	}
	for _, c := range cases {
		got := ScoreExpertPlacement(freq, c.mask, k)
		wantCoverage := float64(c.wantServed) / float64(total)
		if got.Coverage != wantCoverage {
			t.Errorf("%s: Coverage = %v, want served/total = %d/%d = %v", c.name, got.Coverage, c.wantServed, total, wantCoverage)
		}
		wantDrift := 1 - float64(c.wantOverlap)/float64(k)
		if got.Drift != wantDrift {
			t.Errorf("%s: Drift = %v, want 1 - %d/%d = %v", c.name, got.Drift, c.wantOverlap, k, wantDrift)
		}
	}

	// Explicit boundary assertions the task names directly.
	if d := ScoreExpertPlacement(freq, []bool{true, true, false, false, false}, k).Drift; d != 0 {
		t.Fatalf("mask == top-k must give Drift 0, got %v", d)
	}
	if d := ScoreExpertPlacement(freq, []bool{false, false, false, true, true}, k).Drift; d != 1 {
		t.Fatalf("mask disjoint from top-k must give Drift 1, got %v", d)
	}
}

// TestScoreExpertPlacementEdgeCases guards the degenerate inputs: an empty/zero-total histogram is
// Coverage 0 (not a divide-by-zero NaN), and k <= 0 has no top-k window so Drift is 0.
func TestScoreExpertPlacementEdgeCases(t *testing.T) {
	if got := ScoreExpertPlacement(nil, nil, 2); got.Coverage != 0 || got.Drift != 1 {
		// nil freq => total 0 (Coverage 0) and empty top-k => overlap 0 => Drift 1-0/2 = 1.
		t.Fatalf("nil histogram: got %+v, want Coverage 0 Drift 1", got)
	}
	freq := []int64{5, 3, 1}
	if got := ScoreExpertPlacement(freq, []bool{true, false, false}, 0); got.Drift != 0 {
		t.Fatalf("k<=0 must give Drift 0 (no top-k window), got Drift %v", got.Drift)
	}
	// Zero-total histogram (all experts unaccessed) => Coverage 0, no NaN.
	if got := ScoreExpertPlacement([]int64{0, 0, 0}, []bool{true, true, true}, 2); got.Coverage != 0 {
		t.Fatalf("zero-total histogram must give Coverage 0, got %v", got.Coverage)
	}
}

// TestExpertAccessHistogramFold pins the trace fold: hist[e] counts every touch of expert e across
// layers, unaccessed experts keep a zero slot, and out-of-range expert ids are skipped.
func TestExpertAccessHistogramFold(t *testing.T) {
	events := []ExpertAccessTraceEvent{
		{Layer: 0, Expert: 0, WeightBytes: 100},
		{Layer: 0, Expert: 0, WeightBytes: 100}, // expert 0 twice in layer 0
		{Layer: 1, Expert: 2, WeightBytes: 100}, // same expert id, different layer, still counts
		{Layer: 0, Expert: 1, WeightBytes: 100},
		{Layer: 2, Expert: 0, WeightBytes: 100}, // expert 0 again in a later layer
		{Layer: 0, Expert: 9, WeightBytes: 100}, // out of range for numExperts=4 => skipped
	}
	got := ExpertAccessHistogram(events, 4)
	want := []int64{3, 1, 1, 0} // expert 0 hit 3×, 1 once, 2 once, 3 never
	if len(got) != len(want) {
		t.Fatalf("histogram width = %d, want %d", len(got), len(want))
	}
	for e := range want {
		if got[e] != want[e] {
			t.Errorf("hist[%d] = %d, want %d", e, got[e], want[e])
		}
	}
	if h := ExpertAccessHistogram(events, 0); h != nil {
		t.Errorf("numExperts<=0 must yield a nil histogram, got %v", h)
	}

	// The fold feeds ScoreExpertPlacement directly: expert 0 is the sole top-1 (3 touches), so a plan
	// resident on {0} covers 3/5 touches with zero drift.
	score := ScoreExpertPlacement(got, []bool{true, false, false, false}, 1)
	if wantCov := 3.0 / 5.0; score.Coverage != wantCov {
		t.Errorf("folded Coverage = %v, want %v", score.Coverage, wantCov)
	}
	if score.Drift != 0 {
		t.Errorf("plan resident on the top-1 expert must have Drift 0, got %v", score.Drift)
	}
}
