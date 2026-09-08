package issueorchestrator

import (
	"fmt"
	"os"
	"path/filepath"
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

func TestPlanWaves_SameLaneDisjointPathsAdmittedInSameWave(t *testing.T) {
	issues := []Issue{
		testIssue(1, "issue-1", "Compute task A", "compute", []string{"internal/compute/a.go"}, 3),
		testIssue(2, "issue-2", "Compute task B", "compute", []string{"internal/compute/b.go"}, 3),
	}

	plan := PlanWaves(issues, WavePlanOptions{
		WaveSize: 4,
	})

	if plan.TotalWaves != 1 {
		t.Fatalf("expected 1 wave for same-lane disjoint issues, got %d waves", plan.TotalWaves)
	}
	if len(plan.Waves[0].Issues) != 2 {
		t.Fatalf("expected 2 issues in wave 1, got %d", len(plan.Waves[0].Issues))
	}
	if plan.Waves[0].Issues[0].Number != 1 || plan.Waves[0].Issues[1].Number != 2 {
		t.Errorf("expected issues 1 and 2 in wave 1, got numbers: %v", plan.Waves[0].IssueNumbers)
	}
}

func TestPlanWaves_OverlappingPathsSeparatedIntoDifferentWaves(t *testing.T) {
	// Same lane with overlapping paths
	issuesSameLane := []Issue{
		testIssue(1, "issue-1", "Compute task A", "compute", []string{"internal/compute/a.go"}, 3),
		testIssue(2, "issue-2", "Compute task A prime", "compute", []string{"internal/compute/a.go"}, 3),
	}
	planSameLane := PlanWaves(issuesSameLane, WavePlanOptions{
		WaveSize: 4,
	})
	if planSameLane.TotalWaves != 2 {
		t.Fatalf("expected 2 waves for same-lane overlapping issues, got %d", planSameLane.TotalWaves)
	}
	if len(planSameLane.Waves) != 2 || len(planSameLane.Waves[0].Issues) != 1 || len(planSameLane.Waves[1].Issues) != 1 {
		t.Fatalf("expected 1 issue per wave, got waves: %v", planSameLane.Waves)
	}

	// Cross lane with overlapping paths
	issuesCrossLane := []Issue{
		testIssue(3, "issue-3", "Cross lane 1", "compute", []string{"internal/shared/common.go"}, 3),
		testIssue(4, "issue-4", "Cross lane 2", "gateway", []string{"internal/shared/common.go"}, 3),
	}
	planCrossLane := PlanWaves(issuesCrossLane, WavePlanOptions{
		WaveSize: 4,
	})
	if planCrossLane.TotalWaves != 2 {
		t.Fatalf("expected 2 waves for cross-lane overlapping issues, got %d", planCrossLane.TotalWaves)
	}
}

func TestPlanWaves_HeldLeaseExactTreeSpecificity(t *testing.T) {
	tempDir := t.TempDir()
	dosDir := filepath.Join(tempDir, ".dos")
	if err := os.MkdirAll(dosDir, 0o755); err != nil {
		t.Fatal(err)
	}
	journal := filepath.Join(dosDir, "lane-journal.jsonl")
	// Lease held specifically on internal/compute/a.go
	record := `{"op":"ACQUIRE","lane":"compute","tree":["internal/compute/a.go"]}` + "\n"
	if err := os.WriteFile(journal, []byte(record), 0o644); err != nil {
		t.Fatal(err)
	}

	issues := []Issue{
		// Issue 1: disjoint path on b.go -> MUST be permitted (NOT held)
		testIssue(1, "issue-1", "Compute task B", "compute", []string{"internal/compute/b.go"}, 3),
		// Issue 2: exact overlapping path on a.go -> MUST be held
		testIssue(2, "issue-2", "Compute task A", "compute", []string{"internal/compute/a.go"}, 3),
		// Issue 3: ancestor wildcard overlapping path on internal/compute/** -> MUST be held
		testIssue(3, "issue-3", "Compute task wildcard", "compute", []string{"internal/compute/**"}, 3),
	}

	plan := PlanWaves(issues, WavePlanOptions{
		WaveSize:       4,
		WorkspaceRoot:  tempDir,
		AutoDetectHeld: true,
	})

	// Issue 1 should NOT be in HeldIssues
	for _, num := range plan.HeldIssues {
		if num == 1 {
			t.Errorf("issue 1 on disjoint path 'internal/compute/b.go' must NOT be held!")
		}
	}

	// Issue 2 and Issue 3 MUST be in HeldIssues
	hasIssue2 := false
	hasIssue3 := false
	for _, num := range plan.HeldIssues {
		if num == 2 {
			hasIssue2 = true
		}
		if num == 3 {
			hasIssue3 = true
		}
	}
	if !hasIssue2 {
		t.Errorf("issue 2 on exact overlapping path 'internal/compute/a.go' MUST be held!")
	}
	if !hasIssue3 {
		t.Errorf("issue 3 on wildcard path 'internal/compute/**' MUST be held!")
	}

	// Issue 1 should be scheduled in Wave 1
	if plan.PlannedIssues != 1 {
		t.Fatalf("expected exactly 1 planned issue (issue 1), got %d", plan.PlannedIssues)
	}
	if len(plan.Waves) == 0 || len(plan.Waves[0].Issues) != 1 || plan.Waves[0].Issues[0].Number != 1 {
		t.Fatalf("expected issue 1 to be planned in wave 1, got %v", plan.Waves)
	}
}
