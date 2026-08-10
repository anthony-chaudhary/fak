package dogfoodissues

import (
	"strings"
	"testing"
)

func TestEffectfulPlanFailsClosedWithoutProjectBaseline(t *testing.T) {
	item := qaActionItem()
	plan, skipped := BuildPlanWithOptions([]ActionItem{item}, nil, BuildOptions{Live: true, DedupeChecked: true, DedupeCap: 10})
	if len(plan) != 0 || len(skipped) != 1 || !strings.Contains(skipped[0].Reason, "ISSUE_PROJECT_WORK_MISSING") {
		t.Fatalf("plan=%+v skipped=%+v", plan, skipped)
	}
}
func TestEffectfulPlanBodyCarriesProjectWork(t *testing.T) {
	item := qaActionItem()
	opt := BuildOptions{Live: true, DedupeChecked: true, DedupeCap: 10, ParentBaseline: 20, CompletionStandard: "demo"}
	plan, skipped := BuildPlanWithOptions([]ActionItem{item}, nil, opt)
	if len(skipped) != 0 || len(plan) != 1 {
		t.Fatalf("plan=%+v skipped=%+v", plan, skipped)
	}
	for _, want := range []string{"## Work estimate", "Contribution: 3/20 points", "## Completion standard\n\ndemo"} {
		if !strings.Contains(plan[0].Body, want) {
			t.Fatalf("missing %q:\n%s", want, plan[0].Body)
		}
	}
}
func qaActionItem() ActionItem {
	return ActionItem{Key: "k", Title: "x", ParentRef: "#36", CurrentState: "x", WhyNow: "x", WorkingSpine: "x", WorkUnit: "leaf", ExpectedSteps: 3, Assumptions: []string{"x"}, ConfusionRisks: []string{"x"}, Coordination: []string{"x"}, Trigger: "x", BatchPolicy: "one batch, capped at 10 items, dedupe by key", InScope: "x", OutOfScope: "x", DoneCondition: "x", Witness: "go test ./internal/x", AcceptanceGate: "go test ./internal/x", Lane: "x", Paths: []string{"internal/x"}, Labels: []string{"gen/now", "priority/P1"}, ClosureBinding: "x"}
}
