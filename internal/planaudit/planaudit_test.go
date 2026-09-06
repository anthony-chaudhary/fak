package planaudit

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCountUnits(t *testing.T) {
	lines := []string{
		"# Plan",
		"| N | Work |",
		"|---|---|",
		"| 1 | table unit |",
		"| 2 | table unit |",
		"## 3. heading unit",
		"### 3.1 heading subunit",
		"## Not a unit",
	}
	if got := CountUnits(lines); got != 4 {
		t.Fatalf("CountUnits=%d, want 4", got)
	}
}

func TestBuildReportTaskWeightedFloor(t *testing.T) {
	report := BuildReport([]Plan{{
		ID: "PLAN-x", Name: "Plan X", File: "PLAN-x.md", TotalUnits: 5, Signal: "none", PercentComplete: 0, Status: "not_started",
	}})
	task := report.WorkUnits.TaskWeighted
	if task["total_units"] != 5 || task["done_units"] != 0 || task["coverage_plans"] != 1 {
		t.Fatalf("task weighted=%+v", task)
	}
}

func TestAuditPlanUsesHeaderMarkerOnly(t *testing.T) {
	path := filepath.Join(t.TempDir(), "PLAN-example.md")
	lines := make([]string, HeaderLines+2)
	lines[0] = "# Example plan"
	lines[1] = "## 1. first unit"
	lines[HeaderLines] = "This work shipped later."
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")), 0o644); err != nil {
		t.Fatal(err)
	}

	plan, err := AuditPlan(path)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Name != "Example plan" || plan.TotalUnits != 1 {
		t.Fatalf("plan=%+v", plan)
	}
	if plan.Signal != "none" || plan.PercentComplete != 0 || plan.Status != "not_started" {
		t.Fatalf("marker outside header changed completion: %+v", plan)
	}
}

func TestAuditPlanRecognizesHeaderMarker(t *testing.T) {
	path := filepath.Join(t.TempDir(), "BUILD-example.md")
	if err := os.WriteFile(path, []byte("# Build example\n\nStatus: completed\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	plan, err := AuditPlan(path)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Signal != "shipped-marker" || plan.PercentComplete != 100 || plan.Status != "complete" {
		t.Fatalf("plan=%+v", plan)
	}
}

func TestAuditPlan_AllUncheckedPlusIncidentalShipped(t *testing.T) {
	path := filepath.Join(t.TempDir(), "PLAN-unchecked-shipped.md")
	content := strings.Join([]string{
		"# Plan with Incidental Shipped Marker",
		"",
		"Status: shipped",
		"Phase 1 shipped yesterday, but milestone tasks are still open.",
		"",
		"## Tasks",
		"- [ ] M1: Kernel admission",
		"- [ ] M2: Pager fault handling",
		"- [ ] M3: Context MMU proof",
	}, "\n")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	plan, err := AuditPlan(path)
	if err != nil {
		t.Fatal(err)
	}
	if plan.TotalUnits != 3 {
		t.Fatalf("TotalUnits=%d, want 3", plan.TotalUnits)
	}
	if plan.DoneUnits != 0 {
		t.Fatalf("DoneUnits=%d, want 0", plan.DoneUnits)
	}
	if plan.PercentComplete != 0 {
		t.Fatalf("PercentComplete=%d, want 0", plan.PercentComplete)
	}
	if plan.Status == "complete" || plan.Status != "not_started" {
		t.Fatalf("Status=%q, want not_started (must not be complete)", plan.Status)
	}
	if plan.Signal != "task-boxes" {
		t.Fatalf("Signal=%q, want task-boxes", plan.Signal)
	}

	report := BuildReport([]Plan{plan})
	if got := report.Counts["complete"]; got != 0 {
		t.Fatalf("counts[complete]=%d, want 0", got)
	}
	if got := report.Counts["not_started"]; got != 1 {
		t.Fatalf("counts[not_started]=%d, want 1", got)
	}
	tw := report.WorkUnits.TaskWeighted
	if tw["done_units"] != 0 || tw["total_units"] != 3 {
		t.Fatalf("task_weighted 0/N mismatch: done_units=%v, total_units=%v, want 0/3", tw["done_units"], tw["total_units"])
	}
	if tw["pct_complete"] != 0.0 {
		t.Fatalf("task_weighted pct_complete=%v, want 0.0", tw["pct_complete"])
	}
}

func TestAuditPlan_MixedChecks(t *testing.T) {
	path := filepath.Join(t.TempDir(), "PLAN-mixed-checks.md")
	content := strings.Join([]string{
		"# Plan with Mixed Checked Tasks",
		"",
		"## Tasks",
		"- [x] M1: Wire API boundary",
		"- [ ] M2: Implement retry policy",
		"- [ ] M3: Verification tests",
		"- [ ] M4: Documentation",
	}, "\n")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	plan, err := AuditPlan(path)
	if err != nil {
		t.Fatal(err)
	}
	if plan.TotalUnits != 4 {
		t.Fatalf("TotalUnits=%d, want 4", plan.TotalUnits)
	}
	if plan.DoneUnits != 1 {
		t.Fatalf("DoneUnits=%d, want 1", plan.DoneUnits)
	}
	if plan.PercentComplete != 25 {
		t.Fatalf("PercentComplete=%d, want 25", plan.PercentComplete)
	}
	if plan.Status != "in_progress" {
		t.Fatalf("Status=%q, want in_progress", plan.Status)
	}
	if plan.Signal != "task-boxes" {
		t.Fatalf("Signal=%q, want task-boxes", plan.Signal)
	}

	report := BuildReport([]Plan{plan})
	if got := report.Counts["in_progress"]; got != 1 {
		t.Fatalf("counts[in_progress]=%d, want 1", got)
	}
	if got := report.Counts["complete"]; got != 0 {
		t.Fatalf("counts[complete]=%d, want 0", got)
	}
	tw := report.WorkUnits.TaskWeighted
	if tw["done_units"] != 1 || tw["total_units"] != 4 {
		t.Fatalf("task_weighted done_units=%v, total_units=%v, want 1/4", tw["done_units"], tw["total_units"])
	}
	if tw["pct_complete"] != 25.0 {
		t.Fatalf("task_weighted pct_complete=%v, want 25.0", tw["pct_complete"])
	}
	pw := report.WorkUnits.PlanWeighted
	if pw["pct_complete"] != 25.0 {
		t.Fatalf("plan_weighted pct_complete=%v, want 25.0", pw["pct_complete"])
	}
}

func TestAuditPlan_AllChecked(t *testing.T) {
	path := filepath.Join(t.TempDir(), "PLAN-all-checked.md")
	content := strings.Join([]string{
		"# Plan Fully Complete",
		"",
		"## Tasks",
		"- [x] M1: Initial design",
		"- [X] M2: Capital X check",
		"- [x] M3: Shipped implementation",
	}, "\n")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	plan, err := AuditPlan(path)
	if err != nil {
		t.Fatal(err)
	}
	if plan.TotalUnits != 3 {
		t.Fatalf("TotalUnits=%d, want 3", plan.TotalUnits)
	}
	if plan.DoneUnits != 3 {
		t.Fatalf("DoneUnits=%d, want 3", plan.DoneUnits)
	}
	if plan.PercentComplete != 100 {
		t.Fatalf("PercentComplete=%d, want 100", plan.PercentComplete)
	}
	if plan.Status != "complete" {
		t.Fatalf("Status=%q, want complete", plan.Status)
	}
	if plan.Signal != "task-boxes" {
		t.Fatalf("Signal=%q, want task-boxes", plan.Signal)
	}

	report := BuildReport([]Plan{plan})
	if got := report.Counts["complete"]; got != 1 {
		t.Fatalf("counts[complete]=%d, want 1", got)
	}
	if got := report.Counts["in_progress"]; got != 0 {
		t.Fatalf("counts[in_progress]=%d, want 0", got)
	}
	tw := report.WorkUnits.TaskWeighted
	if tw["done_units"] != 3 || tw["total_units"] != 3 {
		t.Fatalf("task_weighted done_units=%v, total_units=%v, want 3/3", tw["done_units"], tw["total_units"])
	}
	if tw["pct_complete"] != 100.0 {
		t.Fatalf("task_weighted pct_complete=%v, want 100.0", tw["pct_complete"])
	}
}

func TestAuditPlan_NoTaskBoxesMarkerFallback(t *testing.T) {
	t.Run("shipped marker present", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "PLAN-no-tasks-shipped.md")
		content := strings.Join([]string{
			"# Shipped Plan",
			"Status: shipped",
			"## 1. First Heading Unit",
			"## 2. Second Heading Unit",
		}, "\n")
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}

		plan, err := AuditPlan(path)
		if err != nil {
			t.Fatal(err)
		}
		if plan.TotalUnits != 2 {
			t.Fatalf("TotalUnits=%d, want 2", plan.TotalUnits)
		}
		if plan.DoneUnits != 2 {
			t.Fatalf("DoneUnits=%d, want 2", plan.DoneUnits)
		}
		if plan.PercentComplete != 100 {
			t.Fatalf("PercentComplete=%d, want 100", plan.PercentComplete)
		}
		if plan.Status != "complete" {
			t.Fatalf("Status=%q, want complete", plan.Status)
		}
		if plan.Signal != "shipped-marker" {
			t.Fatalf("Signal=%q, want shipped-marker", plan.Signal)
		}

		report := BuildReport([]Plan{plan})
		if got := report.Counts["complete"]; got != 1 {
			t.Fatalf("counts[complete]=%d, want 1", got)
		}
		tw := report.WorkUnits.TaskWeighted
		if tw["done_units"] != 2 || tw["total_units"] != 2 {
			t.Fatalf("task_weighted done_units=%v, total_units=%v, want 2/2", tw["done_units"], tw["total_units"])
		}
		if tw["pct_complete"] != 100.0 {
			t.Fatalf("task_weighted pct_complete=%v, want 100.0", tw["pct_complete"])
		}
	})

	t.Run("no shipped marker", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "PLAN-no-tasks-pending.md")
		content := strings.Join([]string{
			"# Pending Plan",
			"Status: draft",
			"## 1. First Heading Unit",
			"## 2. Second Heading Unit",
		}, "\n")
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}

		plan, err := AuditPlan(path)
		if err != nil {
			t.Fatal(err)
		}
		if plan.TotalUnits != 2 {
			t.Fatalf("TotalUnits=%d, want 2", plan.TotalUnits)
		}
		if plan.DoneUnits != 0 {
			t.Fatalf("DoneUnits=%d, want 0", plan.DoneUnits)
		}
		if plan.PercentComplete != 0 {
			t.Fatalf("PercentComplete=%d, want 0", plan.PercentComplete)
		}
		if plan.Status != "not_started" {
			t.Fatalf("Status=%q, want not_started", plan.Status)
		}
		if plan.Signal != "none" {
			t.Fatalf("Signal=%q, want none", plan.Signal)
		}

		report := BuildReport([]Plan{plan})
		if got := report.Counts["not_started"]; got != 1 {
			t.Fatalf("counts[not_started]=%d, want 1", got)
		}
		tw := report.WorkUnits.TaskWeighted
		if tw["done_units"] != 0 || tw["total_units"] != 2 {
			t.Fatalf("task_weighted done_units=%v, total_units=%v, want 0/2", tw["done_units"], tw["total_units"])
		}
		if tw["pct_complete"] != 0.0 {
			t.Fatalf("task_weighted pct_complete=%v, want 0.0", tw["pct_complete"])
		}
	})
}

func TestAuditPlan_Qwen38MacTop10Plan(t *testing.T) {
	path := filepath.Join("..", "..", "docs", "plans", "qwen38-mac-top10-plan.md")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("docs/plans/qwen38-mac-top10-plan.md does not exist")
	}
	plan, err := AuditPlan(path)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Signal != "task-boxes" {
		t.Fatalf("Signal=%q, want task-boxes", plan.Signal)
	}
	if plan.TotalUnits != 10 {
		t.Fatalf("TotalUnits=%d, want 10", plan.TotalUnits)
	}
	if plan.Status == "complete" {
		t.Fatalf("Status=%q, want not complete when open tasks remain", plan.Status)
	}
	if plan.DoneUnits >= plan.TotalUnits {
		t.Fatalf("DoneUnits=%d must be less than TotalUnits=%d", plan.DoneUnits, plan.TotalUnits)
	}
}

func TestAuditPlan_TableDrivenPrecedence(t *testing.T) {
	cases := []struct {
		name        string
		content     string
		wantTotal   int
		wantDone    int
		wantPct     int
		wantStatus  string
		wantSignal  string
		wantRepComp int
	}{
		{
			name: "all unchecked task boxes + incidental prose shipped",
			content: strings.Join([]string{
				"# Plan 1 - Unchecked With Incidental Shipped",
				"Status: shipped",
				"Phase A shipped in v1.0, but follow-up tasks remain open.",
				"",
				"## Tasks",
				"- [ ] M1: Task one",
				"- [ ] M2: Task two",
				"- [ ] M3: Task three",
			}, "\n"),
			wantTotal:   3,
			wantDone:    0,
			wantPct:     0,
			wantStatus:  "not_started",
			wantSignal:  "task-boxes",
			wantRepComp: 0,
		},
		{
			name: "mixed checks (1 checked out of 3)",
			content: strings.Join([]string{
				"# Plan 2 - Mixed Tasks",
				"Status: active",
				"",
				"## Tasks",
				"- [x] M1: Task one finished",
				"- [ ] M2: Task two open",
				"- [ ] M3: Task three open",
			}, "\n"),
			wantTotal:   3,
			wantDone:    1,
			wantPct:     33,
			wantStatus:  "in_progress",
			wantSignal:  "task-boxes",
			wantRepComp: 0,
		},
		{
			name: "all checked",
			content: strings.Join([]string{
				"# Plan 3 - All Checked",
				"",
				"## Tasks",
				"- [x] M1: Task one done",
				"- [x] M2: Task two done",
				"- [x] M3: Task three done",
			}, "\n"),
			wantTotal:   3,
			wantDone:    3,
			wantPct:     100,
			wantStatus:  "complete",
			wantSignal:  "task-boxes",
			wantRepComp: 1,
		},
		{
			name: "no task boxes at all with coarse shipped marker fallback",
			content: strings.Join([]string{
				"# Plan 4 - Coarse Shipped Marker",
				"Status: complete",
				"",
				"## 1. First Phase",
				"## 2. Second Phase",
			}, "\n"),
			wantTotal:   2,
			wantDone:    2,
			wantPct:     100,
			wantStatus:  "complete",
			wantSignal:  "shipped-marker",
			wantRepComp: 1,
		},
		{
			name: "no task boxes at all without shipped marker",
			content: strings.Join([]string{
				"# Plan 5 - No Marker",
				"Status: draft",
				"",
				"## 1. First Phase",
				"## 2. Second Phase",
			}, "\n"),
			wantTotal:   2,
			wantDone:    0,
			wantPct:     0,
			wantStatus:  "not_started",
			wantSignal:  "none",
			wantRepComp: 0,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "PLAN-table.md")
			if err := os.WriteFile(path, []byte(tc.content), 0o644); err != nil {
				t.Fatal(err)
			}

			plan, err := AuditPlan(path)
			if err != nil {
				t.Fatalf("AuditPlan failed: %v", err)
			}
			if plan.TotalUnits != tc.wantTotal {
				t.Errorf("TotalUnits = %d, want %d", plan.TotalUnits, tc.wantTotal)
			}
			if plan.DoneUnits != tc.wantDone {
				t.Errorf("DoneUnits = %d, want %d", plan.DoneUnits, tc.wantDone)
			}
			if plan.PercentComplete != tc.wantPct {
				t.Errorf("PercentComplete = %d, want %d", plan.PercentComplete, tc.wantPct)
			}
			if plan.Status != tc.wantStatus {
				t.Errorf("Status = %q, want %q", plan.Status, tc.wantStatus)
			}
			if plan.Signal != tc.wantSignal {
				t.Errorf("Signal = %q, want %q", plan.Signal, tc.wantSignal)
			}

			report := BuildReport([]Plan{plan})
			if got := report.Counts["complete"]; got != tc.wantRepComp {
				t.Errorf("report.Counts[complete] = %d, want %d", got, tc.wantRepComp)
			}
		})
	}
}
