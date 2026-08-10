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
