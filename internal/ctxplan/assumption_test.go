package ctxplan

import "testing"

func TestAssessAssumptionsScoresSourceClasses(t *testing.T) {
	report := AssessAssumptions([]Assumption{
		{Key: "z-stale", Source: AssumptionStale, Statement: "old recalled region"},
		{Key: "a-user", Source: AssumptionUserStated, Statement: "user chose prod"},
		{Key: "b-witness", Source: AssumptionWitnessed, Statement: "test passed"},
		{Key: "c-inferred", Source: AssumptionInferred, Confidence: 0.90, Statement: "likely branch"},
		{Key: "d-unknown", Source: AssumptionUnknown, Statement: "deployment target"},
	}, DefaultAssumptionPolicy())

	if report.EffectSafe {
		t.Fatal("stale/unknown assumptions must make the report unsafe for effects")
	}
	if report.Summary.Use != 3 || report.Summary.Query != 1 || report.Summary.Refresh != 1 {
		t.Fatalf("summary=%+v, want use=3 query=1 refresh=1", report.Summary)
	}
	if got := report.Assessments[0].Key; got != "a-user" {
		t.Fatalf("assessments must be sorted by key, first=%q", got)
	}
	want := map[string]AssumptionAction{
		"a-user":     AssumptionUse,
		"b-witness":  AssumptionUse,
		"c-inferred": AssumptionUse,
		"d-unknown":  AssumptionQuery,
		"z-stale":    AssumptionRefresh,
	}
	for _, a := range report.Assessments {
		if a.Action != want[a.Key] {
			t.Fatalf("%s action=%q want %q (%+v)", a.Key, a.Action, want[a.Key], a)
		}
	}
}

func TestAssessAssumptionsLowConfidenceInferenceQueries(t *testing.T) {
	report := AssessAssumptions([]Assumption{
		{Key: "low-direct", Source: AssumptionWitnessed, Confidence: 0.20},
		{Key: "low-inferred", Source: AssumptionInferred, Confidence: 0.79},
		{Key: "bad-source", Source: AssumptionSource("maybe")},
	}, DefaultAssumptionPolicy())

	if report.EffectSafe {
		t.Fatal("low-confidence and unknown-source assumptions must not be effect safe")
	}
	if report.Summary.Query != 3 {
		t.Fatalf("query count=%d, want 3: %+v", report.Summary.Query, report.Assessments)
	}
	for _, a := range report.Assessments {
		if a.Action != AssumptionQuery {
			t.Fatalf("%s action=%q, want query", a.Key, a.Action)
		}
	}
}

func TestPlanQueryCarriesAssumptionGate(t *testing.T) {
	budget := Budget{Tokens: 64}
	view := PlanQuery{
		Intents: []string{"auth token rotation"},
		Budget:  &budget,
		Assumptions: []Assumption{
			{Key: "deploy-target", Source: AssumptionUnknown, Statement: "production or staging"},
			{Key: "ticket", Source: AssumptionUserStated, Statement: "customer ticket A"},
		},
	}.Plan(querySpans(), nil)

	if view.Assumptions == nil {
		t.Fatal("PlanView should carry an assumption report when query assumptions are supplied")
	}
	if view.Assumptions.EffectSafe {
		t.Fatalf("unknown assumption should block effect-safe plan use: %+v", view.Assumptions)
	}
	if view.Assumptions.Summary.Query != 1 || view.Assumptions.Summary.Use != 1 {
		t.Fatalf("assumption summary=%+v, want one query and one usable fact", view.Assumptions.Summary)
	}
	if !selectedHas(view, "hit") {
		t.Fatalf("assumption gate must not disturb planning selection, selected=%v", viewSelectedIDs(view))
	}
}
