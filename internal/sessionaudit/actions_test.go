package sessionaudit

import "testing"

func TestBuildCompactActionPlanNamesCostAndLongContextActions(t *testing.T) {
	rep := CompactReport{
		Schema:    "fak.session_audit.summary.v1",
		Generated: "2026-07-02T00:00:00Z",
		Scope:     CompactScope{NamespaceFilter: "C--work-fak", Audited: 2, Discovered: 2},
		TopLongContext: []CompactLongContext{{
			Session:            "heavy",
			Namespace:          "C--work-fak",
			TotalContextTokens: 42_000_000,
			IORatio:            500,
			TopModel:           "claude-opus-4-8",
		}},
		Recommendations: []CompactRecommendation{
			{Kind: "opus_cost_pressure", Severity: "high", Action: "keep Fable default", Reason: "cost", Evidence: "opus_cost_share=80.0%"},
			{Kind: "long_context_pressure", Severity: "high", Action: "checkpoint or reset", Reason: "context", Evidence: "session=heavy"},
		},
	}

	plan := BuildCompactActionPlan(rep)
	if plan.Schema != CompactActionPlanSchema || plan.SummarySchema != rep.Schema {
		t.Fatalf("plan schema = %q/%q", plan.Schema, plan.SummarySchema)
	}
	if plan.Counts.Total != 2 || plan.Counts.High != 2 {
		t.Fatalf("counts = %+v, want two high actions", plan.Counts)
	}
	if plan.Actions[0].ID != "keep_fable_default" || plan.Actions[0].Target != "model_route:fable_default" {
		t.Fatalf("cost action = %+v", plan.Actions[0])
	}
	if plan.Actions[1].ID != "checkpoint_reset_top_long_context" ||
		plan.Actions[1].Target != "session:C--work-fak/heavy" ||
		plan.Actions[1].Session != "heavy" ||
		len(plan.Actions[1].WitnessCommands) == 0 {
		t.Fatalf("long-context action = %+v", plan.Actions[1])
	}
	if plan.Correctness == "" {
		t.Fatal("plan must carry advisory correctness boundary")
	}
}

func TestApplyCompactActionGateRefusesAtThreshold(t *testing.T) {
	plan := CompactActionPlan{Actions: []CompactAction{
		{ID: "a", Severity: "high"},
		{ID: "b", Severity: "medium"},
	}}
	gated, ok := ApplyCompactActionGate(plan, "high")
	if !ok {
		t.Fatal("high threshold rejected as invalid")
	}
	if gated.Gate.Verdict != "refuse" || gated.Gate.Refused != 1 || gated.Gate.Threshold != "high" {
		t.Fatalf("high gate = %+v, want one refused high action", gated.Gate)
	}
	gated, ok = ApplyCompactActionGate(plan, "medium")
	if !ok {
		t.Fatal("medium threshold rejected as invalid")
	}
	if gated.Gate.Verdict != "refuse" || gated.Gate.Refused != 2 {
		t.Fatalf("medium gate = %+v, want two refused actions", gated.Gate)
	}
	gated, ok = ApplyCompactActionGate(plan, "none")
	if !ok {
		t.Fatal("none threshold rejected as invalid")
	}
	if gated.Gate.Verdict != "allow" || gated.Gate.Refused != 0 {
		t.Fatalf("disabled gate = %+v, want allow", gated.Gate)
	}
	if _, ok := ApplyCompactActionGate(plan, "urgent"); ok {
		t.Fatal("unknown threshold should be invalid")
	}
}
