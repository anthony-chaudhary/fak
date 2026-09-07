package issueorchestrator

import (
	"fmt"
	"strings"
	"testing"
)

func TestPlanWaves_DynamicSlidingWindow(t *testing.T) {
	// Build a candidate list of 30 issues:
	// Top 20 items contain 15 unplannable entries (e.g. subdivide/triage) and 5 dispatchable leaves.
	// The next 10 items (positions 21-30) contain 5 more dispatchable leaves and 5 unplannable.
	var issues []Issue

	// Items 1 to 20:
	// 5 dispatchable (indices 0, 4, 8, 12, 16 -> numbers 1, 5, 9, 13, 17)
	// 15 unplannable (others -> ExpectedSteps: 20 which triggers isSubdivideTarget, or Dispatchability: "needs_scope")
	for i := 1; i <= 20; i++ {
		if i == 1 || i == 5 || i == 9 || i == 13 || i == 17 {
			issues = append(issues, testIssue(i, fmt.Sprintf("leaf-%d", i), fmt.Sprintf("Leaf %d", i), fmt.Sprintf("lane%d", i), []string{fmt.Sprintf("internal/pkg%d/file.go", i)}, 3))
		} else {
			// Unplannable: ExpectedSteps > 15 (subdivide target)
			iss := testIssue(i, fmt.Sprintf("epic-%d", i), fmt.Sprintf("Epic %d", i), "epiclane", []string{"internal/a/a.go", "internal/b/b.go", "internal/c/c.go"}, 25)
			iss.Dispatchability = "needs_scope"
			issues = append(issues, iss)
		}
	}

	// Items 21 to 30:
	// 5 more dispatchable leaves (numbers 21, 23, 25, 27, 29)
	// 5 unplannable (numbers 22, 24, 26, 28, 30)
	for i := 21; i <= 30; i++ {
		if i%2 == 1 {
			issues = append(issues, testIssue(i, fmt.Sprintf("leaf-%d", i), fmt.Sprintf("Leaf %d", i), fmt.Sprintf("lane%d", i), []string{fmt.Sprintf("internal/pkg%d/file.go", i)}, 2))
		} else {
			iss := testIssue(i, fmt.Sprintf("epic-%d", i), fmt.Sprintf("Epic %d", i), "epiclane", []string{"internal/a/a.go", "internal/b/b.go", "internal/c/c.go"}, 20)
			iss.Dispatchability = "needs_scope"
			issues = append(issues, iss)
		}
	}

	opts := WavePlanOptions{
		WaveSize:             4,
		TargetIssues:         10,
		MinWindow:            20,
		MaxWindow:            50,
		AutoExpand:           true,
		UnplannableWarnRatio: 0.65,
	}

	plan := PlanWaves(issues, opts)

	if plan.Diagnostics == nil {
		t.Fatalf("expected plan.Diagnostics to be non-nil")
	}

	d := plan.Diagnostics
	if d.DiscoveryWindowSize <= 20 {
		t.Errorf("expected DiscoveryWindowSize > 20, got %d", d.DiscoveryWindowSize)
	}
	if d.WindowExpansions <= 0 {
		t.Errorf("expected WindowExpansions > 0, got %d", d.WindowExpansions)
	}
	if d.UnplannableRatio <= 0.65 {
		t.Errorf("expected UnplannableRatio > 0.65, got %f", d.UnplannableRatio)
	}
	if len(d.AdvisoryWarnings) == 0 {
		t.Errorf("expected advisory warnings to be emitted, got 0")
	} else {
		found := false
		for _, w := range d.AdvisoryWarnings {
			if strings.Contains(w, "[advisory] high unplannable ratio") {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected warning containing '[advisory] high unplannable ratio', got %v", d.AdvisoryWarnings)
		}
	}

	// Should have collected 10 dispatchable leaves across waves
	if plan.Dispatchable < 10 {
		t.Errorf("expected at least 10 dispatchable leaves collected, got %d", plan.Dispatchable)
	}
	if plan.PlannedIssues < 10 {
		t.Errorf("expected at least 10 planned issues, got %d", plan.PlannedIssues)
	}
}

func TestPlanWaves_AutoExpandDisabled(t *testing.T) {
	var issues []Issue
	for i := 1; i <= 30; i++ {
		if i <= 5 {
			issues = append(issues, testIssue(i, fmt.Sprintf("leaf-%d", i), fmt.Sprintf("Leaf %d", i), "lane", []string{fmt.Sprintf("internal/pkg%d/file.go", i)}, 2))
		} else {
			iss := testIssue(i, fmt.Sprintf("epic-%d", i), fmt.Sprintf("Epic %d", i), "epiclane", []string{"internal/a/a.go", "internal/b/b.go", "internal/c/c.go"}, 25)
			issues = append(issues, iss)
		}
	}

	opts := WavePlanOptions{
		WaveSize:     4,
		TargetIssues: 10,
		Limit:        10,
		AutoExpand:   false,
	}

	plan := PlanWaves(issues, opts)
	if plan.TotalIssues != 10 {
		t.Errorf("expected 10 total issues when AutoExpand=false with Limit=10, got %d", plan.TotalIssues)
	}
	if plan.Diagnostics != nil {
		t.Errorf("expected Diagnostics to be nil when AutoExpand=false, got %+v", plan.Diagnostics)
	}
}
