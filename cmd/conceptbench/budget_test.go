package main

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestBudgetRegimeRecordedInArtifacts pins the #2740 core-budget scope line
// (mirrors cmd/modelbench): the replay report RECORDS the resolved worker
// regime, so a scheduled fleet artifact states the parallelism its numbers
// were taken at instead of leaving it implicit in the machine it ran on.
func TestBudgetRegimeRecordedInArtifacts(t *testing.T) {
	code, out := captureRun(t, []string{"--replay", replayDir()})
	if code != 0 {
		t.Fatalf("exit=%d, want 0\n%s", code, out)
	}
	var rep struct {
		Budget struct {
			Workers int    `json:"workers"`
			Source  string `json:"source"`
		} `json:"budget"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("report not valid JSON: %v\n%s", err, out)
	}
	if rep.Budget.Workers < 1 || rep.Budget.Source == "" {
		t.Fatalf("report must record the resolved budget regime, got %+v", rep.Budget)
	}
}

// TestFakWorkersOverridesBudgetFlag pins the documented knob precedence
// (FAK_WORKERS > FAK_BUDGET > -budget, internal/model/budget.go): with
// FAK_WORKERS set the -budget flag is ignored (it never reaches
// model.SetWorkerBudget, so the resolved count is untouched), and the
// artifact records BOTH knobs so the conflict stays auditable.
func TestFakWorkersOverridesBudgetFlag(t *testing.T) {
	t.Setenv("FAK_WORKERS", "3")
	before := model.NumWorkers()
	code, out := captureRun(t, []string{"--contract", "--replay", replayDir(), "--models", "opus-4.8,smollm2-135m", "--budget", "0.5"})
	if code != 0 {
		t.Fatalf("exit=%d, want 0\n%s", code, out)
	}
	if got := model.NumWorkers(); got != before {
		t.Fatalf("-budget mutated the resolved workers under FAK_WORKERS: %d -> %d", before, got)
	}
	var c struct {
		Budget struct {
			Workers    int     `json:"workers"`
			Source     string  `json:"source"`
			BudgetFlag float64 `json:"budget_flag"`
			FakWorkers string  `json:"fak_workers_env"`
		} `json:"budget"`
	}
	if err := json.Unmarshal([]byte(out), &c); err != nil {
		t.Fatalf("contract not valid JSON: %v\n%s", err, out)
	}
	if c.Budget.FakWorkers != "3" {
		t.Fatalf("contract must record the FAK_WORKERS override, got %+v", c.Budget)
	}
	if c.Budget.BudgetFlag != 0.5 {
		t.Fatalf("contract must record the ignored -budget flag for audit, got %+v", c.Budget)
	}
	if c.Budget.Workers != before {
		t.Fatalf("contract workers=%d, want the untouched resolved count %d", c.Budget.Workers, before)
	}
}
