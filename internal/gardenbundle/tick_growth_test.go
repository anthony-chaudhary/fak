package gardenbundle

import "testing"

// TestPlanTickActsOnGrowthgateAction proves a growthgate member that surfaced an
// ACTION condition is bound to the ActGrowthReap edge with Perform=true — the
// act-edge that turns growthgate's reported-never-acted verdict into a collect.
func TestPlanTickActsOnGrowthgateAction(t *testing.T) {
	plan := PlanTick([]MemberResult{mkResult("growthgate", "action")}, false)
	d := decisionFor(plan, "growthgate")
	if d.Act != ActGrowthReap {
		t.Fatalf("growthgate: want act=%s, got %s", ActGrowthReap, d.Act)
	}
	if !d.Perform {
		t.Fatalf("growthgate ACTION must Perform, got perform=false (reason=%q)", d.Reason)
	}
}

// TestPlanTickGrowthgateOkInert proves an OK growthgate member is never acted on:
// a clean census has nothing to collect, so the edge stays inert (ActNone).
func TestPlanTickGrowthgateOkInert(t *testing.T) {
	plan := PlanTick([]MemberResult{mkResult("growthgate", "ok")}, false)
	d := decisionFor(plan, "growthgate")
	if d.Perform {
		t.Fatalf("an ok growthgate must not be acted on, got perform=true")
	}
	if d.Act != ActNone {
		t.Fatalf("an ok growthgate must resolve to ActNone, got %s", d.Act)
	}
}

// TestPlanTickGrowthgateDryRunNoPerform proves --dry-run on a surfaced growthgate
// ACTION still classifies the edge (ActGrowthReap) but performs nothing, inheriting
// the tick's dry-run invariant with no special-casing.
func TestPlanTickGrowthgateDryRunNoPerform(t *testing.T) {
	plan := PlanTick([]MemberResult{mkResult("growthgate", "action")}, true)
	d := decisionFor(plan, "growthgate")
	if d.Perform {
		t.Fatalf("dry-run must perform nothing, got perform=true")
	}
	if d.Act != ActGrowthReap {
		t.Fatalf("dry-run must still classify the edge, got act=%s", d.Act)
	}
	if d.Mode != "dry-run" {
		t.Fatalf("dry-run decision mode = %q, want dry-run", d.Mode)
	}
}
