package issueorchestrator

import (
	"strings"
	"testing"
)

func TestAdaptiveSplit_SinglePackageElasticity(t *testing.T) {
	iss := Issue{
		Number:        101,
		Key:           "ISSUE-101",
		Title:         "Refactor AST parser in single package",
		Lane:          "ast",
		Paths:         []string{"internal/ast/parser.go", "internal/ast/lexer.go"},
		ExpectedSteps: 18,
	}

	assessment := AssessStepBudget(iss, 25, 8)

	if !assessment.ElasticAdmitted {
		t.Fatalf("expected 18-step single-package issue to be admitted under elasticity")
	}
	if assessment.RequiresSplit {
		t.Fatalf("expected 18-step single-package issue to NOT require split")
	}
	if len(assessment.Packages) != 1 || assessment.Packages[0] != "ast" {
		t.Fatalf("expected package [ast], got %v", assessment.Packages)
	}
	if assessment.NominalSteps != 18 {
		t.Fatalf("expected nominal steps 18, got %d", assessment.NominalSteps)
	}

	expectedLog := "[advisory] issue #101 step budget (18) exceeds nominal leaf limit; admitted under step elasticity"
	if assessment.AdvisoryLog != expectedLog {
		t.Fatalf("expected advisory log %q, got %q", expectedLog, assessment.AdvisoryLog)
	}

	// Boundary check: <= 8 steps should be standard leaf without elasticity advisory
	smallIss := Issue{
		Number:        102,
		Key:           "ISSUE-102",
		Title:         "Small fix",
		Paths:         []string{"internal/ast/parser.go"},
		ExpectedSteps: 8,
	}
	smallAssessment := AssessStepBudget(smallIss, 25, 8)
	if smallAssessment.ElasticAdmitted {
		t.Fatalf("8-step issue should not be flagged as elastic-admitted")
	}
	if smallAssessment.RequiresSplit {
		t.Fatalf("8-step issue should not require split")
	}

	// Boundary check: > 25 steps should require split
	hugeIss := Issue{
		Number:        103,
		Key:           "ISSUE-103",
		Title:         "Oversized rewrite",
		Paths:         []string{"internal/ast/parser.go"},
		ExpectedSteps: 26,
	}
	hugeAssessment := AssessStepBudget(hugeIss, 25, 8)
	if !hugeAssessment.RequiresSplit {
		t.Fatalf("26-step single-package issue must require split")
	}
	if hugeAssessment.ElasticAdmitted {
		t.Fatalf("26-step single-package issue cannot be elastic-admitted")
	}
}

func TestAdaptiveSplit_MultiPackageDecomposition(t *testing.T) {
	iss := Issue{
		Number: 202,
		Key:    "EPIC-202",
		Title:  "Cross-cutting metrics overhaul",
		Lane:   "telemetry",
		Paths: []string{
			"internal/gateway/proxy.go",
			"internal/gateway/metrics.go",
			"internal/engine/eval.go",
			"internal/ctxmmu/cache.go",
		},
		ExpectedSteps: 20,
	}

	assessment := AssessStepBudget(iss, 25, 8)
	if !assessment.RequiresSplit {
		t.Fatalf("multi-package issue spanning >2 packages with >8 steps must require split")
	}
	if assessment.ElasticAdmitted {
		t.Fatalf("multi-package epic must not be admitted under elasticity")
	}
	if len(assessment.Packages) != 3 {
		t.Fatalf("expected 3 distinct packages, got %v", assessment.Packages)
	}

	children := SplitEpicArchitectural(iss)
	if len(children) != 3 {
		t.Fatalf("expected 3 children split along package boundaries, got %d", len(children))
	}

	totalChildSteps := 0
	pkgMap := make(map[string]SubdividedIssue)
	for _, child := range children {
		totalChildSteps += child.ExpectedSteps
		pkgMap[child.Package] = child
		if child.ParentNumber != 202 {
			t.Fatalf("expected child ParentNumber to be 202, got %d", child.ParentNumber)
		}
	}

	if totalChildSteps != 20 {
		t.Fatalf("expected total child steps to equal 20, got %d", totalChildSteps)
	}

	// gateway had 2 of 4 paths -> should receive 10 steps
	gwChild, ok := pkgMap["gateway"]
	if !ok {
		t.Fatalf("expected gateway child")
	}
	if len(gwChild.Paths) != 2 {
		t.Fatalf("expected gateway child to have 2 paths, got %d", len(gwChild.Paths))
	}
	if gwChild.ExpectedSteps != 10 {
		t.Fatalf("expected gateway child to have 10 steps, got %d", gwChild.ExpectedSteps)
	}

	// engine had 1 of 4 paths -> 5 steps
	engChild, ok := pkgMap["engine"]
	if !ok {
		t.Fatalf("expected engine child")
	}
	if len(engChild.Paths) != 1 {
		t.Fatalf("expected engine child to have 1 path, got %d", len(engChild.Paths))
	}
	if engChild.ExpectedSteps != 5 {
		t.Fatalf("expected engine child to have 5 steps, got %d", engChild.ExpectedSteps)
	}

	// ctxmmu had 1 of 4 paths -> 5 steps
	mmuChild, ok := pkgMap["ctxmmu"]
	if !ok {
		t.Fatalf("expected ctxmmu child")
	}
	if len(mmuChild.Paths) != 1 {
		t.Fatalf("expected ctxmmu child to have 1 path, got %d", len(mmuChild.Paths))
	}
	if mmuChild.ExpectedSteps != 5 {
		t.Fatalf("expected ctxmmu child to have 5 steps, got %d", mmuChild.ExpectedSteps)
	}
}

func TestAdaptiveSplit_AdjustSteps(t *testing.T) {
	plan := Plan{
		Schema:       WavePlanSchema,
		TotalWaves:   1,
		PlannedSteps: 10,
		Waves: []Wave{
			{
				Index:      1,
				ID:         "wave-1",
				StepBudget: 10,
				Issues: []Issue{
					{
						Number:        301,
						Key:           "ISSUE-301",
						Title:         "Initial scoped work",
						ExpectedSteps: 10,
					},
				},
			},
		},
	}

	err := AdjustIssueSteps(&plan, 301, 18, "discovered complex AST refactoring requiring elasticity")
	if err != nil {
		t.Fatalf("unexpected error adjusting steps: %v", err)
	}

	if plan.Waves[0].Issues[0].ExpectedSteps != 18 {
		t.Fatalf("expected issue expected steps to be 18, got %d", plan.Waves[0].Issues[0].ExpectedSteps)
	}
	if plan.Waves[0].StepBudget != 18 {
		t.Fatalf("expected wave step budget to be 18, got %d", plan.Waves[0].StepBudget)
	}
	if plan.PlannedSteps != 18 {
		t.Fatalf("expected plan planned steps to be 18, got %d", plan.PlannedSteps)
	}

	if plan.Diagnostics == nil || len(plan.Diagnostics.AdvisoryWarnings) == 0 {
		t.Fatalf("expected advisory warning logged for step adjustment")
	}
	lastWarning := plan.Diagnostics.AdvisoryWarnings[len(plan.Diagnostics.AdvisoryWarnings)-1]
	if !strings.Contains(lastWarning, "301") || !strings.Contains(lastWarning, "18") {
		t.Fatalf("expected warning log to mention issue 301 and 18 steps, got: %s", lastWarning)
	}

	// Non-existent issue should return an error
	errNotFound := AdjustIssueSteps(&plan, 999, 15, "no issue")
	if errNotFound == nil {
		t.Fatalf("expected error for non-existent issue")
	}

	// Negative steps should return an error
	errNegative := AdjustIssueSteps(&plan, 301, -3, "negative")
	if errNegative == nil {
		t.Fatalf("expected error for negative steps")
	}
}
