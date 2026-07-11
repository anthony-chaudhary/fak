package ctxplan

import "testing"

// The #3381 DROP-vs-WAIT knob: under budget pressure the planner either accepts the
// recoverable over_budget elision (DROP, the default / never-raise behavior) or raises a
// producer-facing wait signal on the plan (WAIT_FOR_SPACE). The knob only ever adds a
// SIGNAL — it never changes which spans are selected/elided — so every test below pins the
// selection identical across policies and asserts only Plan.Backpressure differs.

// overBudgetKnapsackCands: three equal-cost spans that cannot all fit budget 8 (one is
// forced cold by the knapsack — the main over_budget path, plan.go site 3).
func overBudgetKnapsackCands() []Candidate {
	return []Candidate{
		cand("a", 0, 4, 4.0),
		cand("b", 1, 4, 3.0),
		cand("c", 2, 4, 2.0),
	}
}

func TestBackpressureDefaultZeroValueIsDrop(t *testing.T) {
	// A zero-value Budget (no Backpressure set) must behave EXACTLY as before: elide the
	// over-budget span and raise NO wait signal. This pins "no silent behavior change".
	p := Optimize(overBudgetKnapsackCands(), Budget{Tokens: 8}, nil, ObjGreedy)
	if len(p.Elided) != 1 || p.Elided[0].Reason != ElideOverBudget {
		t.Fatalf("expected exactly 1 over_budget elision, got %+v", p.Elided)
	}
	if p.Backpressure != "" {
		t.Fatalf("default (zero-value) policy must raise no wait signal, got %q", p.Backpressure)
	}
}

func TestBackpressureDropPolicyRaisesNoSignal(t *testing.T) {
	// Explicit DROP is identical to the zero value: elide, never signal.
	p := Optimize(overBudgetKnapsackCands(), Budget{Tokens: 8, Backpressure: BackpressureDrop}, nil, ObjGreedy)
	if len(p.Elided) != 1 || p.Elided[0].Reason != ElideOverBudget {
		t.Fatalf("DROP must still elide over budget, got %+v", p.Elided)
	}
	if p.Backpressure != "" {
		t.Fatalf("DROP must raise no wait signal, got %q", p.Backpressure)
	}
}

func TestBackpressureWaitRaisesSignalOnKnapsackPressure(t *testing.T) {
	drop := Optimize(overBudgetKnapsackCands(), Budget{Tokens: 8}, nil, ObjGreedy)
	wait := Optimize(overBudgetKnapsackCands(), Budget{Tokens: 8, Backpressure: BackpressureWaitForSpace}, nil, ObjGreedy)

	// The knob adds a signal, it does not change the plan: selection + elision are identical.
	if len(wait.Selected) != len(drop.Selected) || len(wait.Elided) != len(drop.Elided) {
		t.Fatalf("WAIT must not change selection: drop sel=%d eli=%d, wait sel=%d eli=%d",
			len(drop.Selected), len(drop.Elided), len(wait.Selected), len(wait.Elided))
	}
	if wait.Backpressure != BackpressureWaitForSpace {
		t.Fatalf("WAIT under real budget pressure must raise the wait signal, got %q", wait.Backpressure)
	}
}

func TestBackpressureWaitNoSignalWhenUnderBudget(t *testing.T) {
	// Everything fits (4+3 <= 8): no span is forced cold, so even WAIT raises nothing — a
	// plan that fits never asks the producer to wait, under either policy.
	cands := []Candidate{cand("a", 0, 4, 4.0), cand("b", 1, 3, 3.0)}
	for _, policy := range []string{BackpressureDrop, BackpressureWaitForSpace} {
		p := Optimize(cands, Budget{Tokens: 8, Backpressure: policy}, nil, ObjGreedy)
		if len(p.Elided) != 0 {
			t.Fatalf("policy %q: under-budget input must elide nothing, got %+v", policy, p.Elided)
		}
		if p.Backpressure != "" {
			t.Fatalf("policy %q: a plan that fits must raise no wait signal, got %q", policy, p.Backpressure)
		}
	}
}

func TestBackpressureWaitRaisesOnPinsOverrun(t *testing.T) {
	// Pins alone (20 tokens) overrun budget 5 — the pins-overrun path (plan.go site 1).
	cands := []Candidate{
		cand("p1", 0, 10, 1),
		cand("p2", 1, 10, 1),
		cand("free", 2, 1, 5),
	}
	pins := map[string]bool{"p1": true, "p2": true}

	drop := Optimize(cands, Budget{Tokens: 5}, pins, ObjGreedy)
	if !drop.OverBudget || drop.Backpressure != "" {
		t.Fatalf("DROP pins-overrun: OverBudget=%v Backpressure=%q, want true/empty", drop.OverBudget, drop.Backpressure)
	}
	wait := Optimize(cands, Budget{Tokens: 5, Backpressure: BackpressureWaitForSpace}, pins, ObjGreedy)
	if !wait.OverBudget {
		t.Fatalf("WAIT pins-overrun should still be OverBudget")
	}
	if wait.Backpressure != BackpressureWaitForSpace {
		t.Fatalf("WAIT pins-overrun must raise the wait signal, got %q", wait.Backpressure)
	}
}

func TestBackpressureWaitRaisesOnFloorOverrun(t *testing.T) {
	// The minimum-evidence floor cannot fit budget 3 — the floor-overrun path (plan.go site 2).
	cands := []Candidate{
		evidenceCand("support", 0, 4, 0, "case-oversize", EvidenceSupport),
		evidenceCand("needle", 1, 2, 10, "case-oversize", EvidenceDecisive),
		cand("optional", 2, 1, 5),
	}
	drop := Optimize(cands, Budget{Tokens: 3}, nil, ObjGreedy)
	if !drop.MinEvidenceOverBudget || drop.Backpressure != "" {
		t.Fatalf("DROP floor-overrun: MinEvidenceOverBudget=%v Backpressure=%q, want true/empty",
			drop.MinEvidenceOverBudget, drop.Backpressure)
	}
	wait := Optimize(cands, Budget{Tokens: 3, Backpressure: BackpressureWaitForSpace}, nil, ObjGreedy)
	if !wait.MinEvidenceOverBudget {
		t.Fatalf("WAIT floor-overrun should still set MinEvidenceOverBudget")
	}
	if wait.Backpressure != BackpressureWaitForSpace {
		t.Fatalf("WAIT floor-overrun must raise the wait signal, got %q", wait.Backpressure)
	}
}
