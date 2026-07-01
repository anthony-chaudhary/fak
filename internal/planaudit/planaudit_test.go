package planaudit

import "testing"

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
