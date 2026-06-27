package main

import (
	"encoding/json"
	"testing"
	"time"
)

// TestLCGIDs pins the deterministic id generator: every id lands in [0,vocab),
// the slice is exactly n long, and the stream is reproducible (same seed path
// every call) — the property the benchmark relies on to compare reps fairly.
func TestLCGIDs(t *testing.T) {
	const n, vocab = 256, 8192
	got := lcgIDs(n, vocab)
	if len(got) != n {
		t.Fatalf("lcgIDs(%d,%d) length = %d, want %d", n, vocab, len(got), n)
	}
	for i, id := range got {
		if id < 0 || id >= vocab {
			t.Fatalf("id[%d] = %d out of range [0,%d)", i, id, vocab)
		}
	}
	// Deterministic: a second call must reproduce the first byte-for-byte.
	again := lcgIDs(n, vocab)
	for i := range got {
		if got[i] != again[i] {
			t.Fatalf("lcgIDs not deterministic at %d: %d != %d", i, got[i], again[i])
		}
	}
	// Non-degenerate: with vocab >> 1 the stream must not be a single constant.
	allSame := true
	for _, id := range got[1:] {
		if id != got[0] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Fatalf("lcgIDs produced a constant stream (%d) — LCG is dead", got[0])
	}
}

// TestMedianMS checks the median-in-milliseconds reducer and, crucially, that it
// does NOT reorder its caller's slice (it sorts a copy) — the timings slices are
// reused across reps, so an in-place sort would corrupt later measurements.
func TestMedianMS(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   []time.Duration
		want float64
	}{
		{"odd", []time.Duration{3 * time.Millisecond, time.Millisecond, 2 * time.Millisecond}, 2.0},
		{"even-upper-middle", []time.Duration{time.Millisecond, 2 * time.Millisecond, 3 * time.Millisecond, 4 * time.Millisecond}, 3.0},
		{"single", []time.Duration{5 * time.Millisecond}, 5.0},
	} {
		in := append([]time.Duration(nil), tc.in...)
		first := in[0]
		if got := medianMS(in); got != tc.want {
			t.Errorf("%s: medianMS = %v, want %v", tc.name, got, tc.want)
		}
		if in[0] != first {
			t.Errorf("%s: medianMS mutated caller slice (in[0] %v -> %v)", tc.name, first, in[0])
		}
	}
}

// TestRoundHelpers pins the half-up rounding to 2 and 3 decimals on values chosen
// to avoid binary-float representation edges, so the test stays deterministic.
func TestRoundHelpers(t *testing.T) {
	if got := round2(3.14159); got != 3.14 {
		t.Errorf("round2(3.14159) = %v, want 3.14", got)
	}
	if got := round2(0); got != 0 {
		t.Errorf("round2(0) = %v, want 0", got)
	}
	if got := round3(3.14159); got != 3.142 {
		t.Errorf("round3(3.14159) = %v, want 3.142", got)
	}
	if got := round3(2.0005); got != 2.001 {
		t.Errorf("round3(2.0005) = %v, want 2.001", got)
	}
}

// TestBisectPlanSingleVariable proves the plan reverts EXACTLY one dim per step (the definition of
// a single-variable bisection), in the documented stable order, and skips dims that already agree.
func TestBisectPlanSingleVariable(t *testing.T) {
	// The carried-P0 anchors: sweep row 4 (known-good) vs row 5 (faults). They differ in
	// layers (16->32) and topk (256->512) only, so the plan must be exactly those two steps.
	good := glmDims{Layers: 16, Hidden: 4096, Heads: 32, Inter: 14336, TopK: 256}
	bad := glmDims{Layers: 32, Hidden: 4096, Heads: 32, Inter: 14336, TopK: 512}

	plan := bisectPlan(good, bad)
	if len(plan) != 2 {
		t.Fatalf("expected 2 differing dims (layers, topk), got %d: %+v", len(plan), plan)
	}
	if plan[0].Dim != "layers" || plan[1].Dim != "index_topk" {
		t.Fatalf("expected stable order [layers, index_topk], got [%s, %s]", plan[0].Dim, plan[1].Dim)
	}
	// Step 0 reverts ONLY layers; every other dim must still hold bad's value.
	if plan[0].Layers != good.Layers {
		t.Errorf("layers step did not revert layers: got %d want %d", plan[0].Layers, good.Layers)
	}
	if plan[0].TopK != bad.TopK || plan[0].Hidden != bad.Hidden || plan[0].Heads != bad.Heads || plan[0].Inter != bad.Inter {
		t.Errorf("layers step changed more than one dim: %+v", plan[0])
	}
	// Step 1 reverts ONLY topk.
	if plan[1].TopK != good.TopK {
		t.Errorf("topk step did not revert topk: got %d want %d", plan[1].TopK, good.TopK)
	}
	if plan[1].Layers != bad.Layers || plan[1].Hidden != bad.Hidden || plan[1].Heads != bad.Heads || plan[1].Inter != bad.Inter {
		t.Errorf("topk step changed more than one dim: %+v", plan[1])
	}
}

// TestBisectPlanAllDimsAndOrder covers every dimension and pins the full stable order.
func TestBisectPlanAllDimsAndOrder(t *testing.T) {
	good := glmDims{Layers: 1, Hidden: 1, Heads: 1, Inter: 1, TopK: 1}
	bad := glmDims{Layers: 2, Hidden: 2, Heads: 2, Inter: 2, TopK: 2}
	plan := bisectPlan(good, bad)
	want := []string{"layers", "hidden", "heads", "inter", "index_topk"}
	if len(plan) != len(want) {
		t.Fatalf("expected %d steps, got %d", len(want), len(plan))
	}
	for i, dim := range want {
		if plan[i].Dim != dim {
			t.Errorf("step %d: got dim %q want %q", i, plan[i].Dim, dim)
		}
	}
}

// TestBisectPlanIdenticalIsEmpty: no differing dims => nothing to bisect.
func TestBisectPlanIdenticalIsEmpty(t *testing.T) {
	d := glmDims{Layers: 8, Hidden: 2048, Heads: 16, Inter: 8192, TopK: 256}
	if plan := bisectPlan(d, d); len(plan) != 0 {
		t.Fatalf("expected empty plan for identical configs, got %d steps", len(plan))
	}
}

// TestBisectStepJSONShape guards the GLMBISECT_JSON contract the GPU runner consumes: each step
// carries its `dim` label alongside the embedded glm dims.
func TestBisectStepJSONShape(t *testing.T) {
	good := glmDims{Layers: 16, Hidden: 4096, Heads: 32, Inter: 14336, TopK: 256}
	bad := glmDims{Layers: 32, Hidden: 4096, Heads: 32, Inter: 14336, TopK: 512}
	b, err := json.Marshal(bisectPlan(good, bad))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back []map[string]any
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(back) != 2 {
		t.Fatalf("expected 2 steps, got %d", len(back))
	}
	for _, k := range []string{"dim", "layers", "hidden", "heads", "inter", "index_topk"} {
		if _, ok := back[0][k]; !ok {
			t.Errorf("step JSON missing key %q: %v", k, back[0])
		}
	}
}
