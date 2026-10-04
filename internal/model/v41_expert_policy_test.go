package model

import "testing"

// TestV41ExpertPolicyTraceGeneratorResumable is acceptance criterion 3: a resumable per-layer
// trace generator emits per-layer histograms consumable as warm-start input.
func TestV41ExpertPolicyTraceGeneratorResumable(t *testing.T) {
	trace := ExpertAccessTrace{
		Schema: ExpertReplayTraceSchema,
		Name:   "v41-policy-test",
		Source: "synthetic",
		Events: []ExpertAccessTraceEvent{
			{Layer: 1, Expert: 0, WeightBytes: 10},
			{Layer: 0, Expert: 2, WeightBytes: 10},
			{Layer: 0, Expert: 1, WeightBytes: 10},
			{Layer: 0, Expert: 1, WeightBytes: 10},
			{Layer: 2, Expert: 3, WeightBytes: 10},
		},
	}
	g := NewV41ExpertTraceGenerator(trace, 4)

	emitted, remaining := g.Progress()
	if emitted != 0 || remaining != 3 {
		t.Fatalf("initial progress emitted=%d remaining=%d, want 0/3", emitted, remaining)
	}

	// Ascending layer order: 0, then 1, then 2.
	h0, ok := g.Next()
	if !ok || h0.Layer != 0 {
		t.Fatalf("first layer=%d ok=%t, want layer 0", h0.Layer, ok)
	}
	if h0.Counts[1] != 2 || h0.Counts[2] != 1 || h0.Touches != 3 {
		t.Fatalf("layer 0 histogram=%v touches=%d, want counts[1]=2 counts[2]=1", h0.Counts, h0.Touches)
	}
	if top := h0.TopExperts(2); len(top) != 2 || top[0] != 1 || top[1] != 2 {
		t.Fatalf("layer 0 top-2=%v, want [1 2]", top)
	}

	if emitted, remaining := g.Progress(); emitted != 1 || remaining != 2 {
		t.Fatalf("progress after 1=%d/%d, want 1/2", emitted, remaining)
	}

	if h1, _ := g.Next(); h1.Layer != 1 {
		t.Fatalf("second layer=%d, want 1", h1.Layer)
	}
	if h2, _ := g.Next(); h2.Layer != 2 {
		t.Fatalf("third layer=%d, want 2", h2.Layer)
	}
	if _, ok := g.Next(); ok {
		t.Fatal("generator emitted a fourth layer; want exhausted")
	}

	g.Reset()
	if emitted, remaining := g.Progress(); emitted != 0 || remaining != 3 {
		t.Fatalf("progress after Reset=%d/%d, want 0/3", emitted, remaining)
	}
	if h, ok := g.Next(); !ok || h.Layer != 0 {
		t.Fatalf("after Reset first layer=%d ok=%t, want layer 0", h.Layer, ok)
	}
}

// TestV41ExpertPolicyWarmStartPlan checks the deterministic per-layer plan and its inference of
// the expert width from the trace when the caller does not supply it.
func TestV41ExpertPolicyWarmStartPlan(t *testing.T) {
	trace := ExpertAccessTrace{
		Schema: ExpertReplayTraceSchema,
		Events: []ExpertAccessTraceEvent{
			{Layer: 0, Expert: 3}, {Layer: 0, Expert: 3}, {Layer: 0, Expert: 5},
			{Layer: 1, Expert: 1},
		},
	}
	plan := V41ExpertWarmStartPlan(trace, 1)
	if len(plan) != 2 {
		t.Fatalf("plan layers=%d, want 2", len(plan))
	}
	if plan[0].Layer != 0 || plan[1].Layer != 1 {
		t.Fatalf("plan layer order=%d,%d want ascending", plan[0].Layer, plan[1].Layer)
	}
	// Layer 0: expert 3 (count 2) beats expert 5 (count 1).
	if h := plan[0]; h.TopExperts(1)[0] != 3 || h.NumExperts != 6 {
		t.Fatalf("layer 0 plan top=%v width=%d, want top 3, width 6 (derived)", h.TopExperts(1), h.NumExperts)
	}
	if h := plan[1]; h.TopExperts(1)[0] != 1 {
		t.Fatalf("layer 1 plan top=%v, want 1", h.TopExperts(1))
	}
	if V41ExpertWarmStartPlan(trace, 0) != nil {
		t.Fatal("zero-budget plan must be nil")
	}
}
