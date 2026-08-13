package main

import (
	"strings"
	"testing"
)

func TestFoldMicroCorpusPreservesUnsupportedCostAndAblationNotYet(t *testing.T) {
	cost := 0.01
	r := foldMicroCorpus([]microCorpusCase{{
		Task: microCorpusTask{ID: "one", Complexity: "one-step", Task: "x", Expected: "x"}, ExecutionVerdict: "PASS",
		Micro:    pairedArm{Correct: true, InputTokens: 7, OutputTokens: 1, WallMS: 10, CostStatus: "provider-unsupported"},
		Baseline: pairedArm{Correct: true, InputTokens: 9, OutputTokens: 1, WallMS: 20, CostUSD: &cost, CostStatus: "provider-reported"},
	}})
	if r.Totals.MicroCostUSD != nil || r.ValueVerdict != "NOT_YET" || r.Ablations[0].Status != "NOT_YET" {
		t.Fatalf("dishonest corpus receipt: %+v", r)
	}
	if r.ExecutionVerdict != "PASS" || r.Totals.MicroCorrect != 1 || r.Totals.BaselineCorrect != 1 || r.Totals.BaselineCostUSD == nil || *r.Totals.BaselineCostUSD != cost {
		t.Fatalf("bad fold: %+v", r)
	}
	md := formatMicroCorpusReport(r)
	for _, want := range []string{"Execution: **PASS**", "Value: **NOT_YET**", "| one | one-step | true | true |", "| mode | NOT_YET |"} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q:\n%s", want, md)
		}
	}
}

func TestFoldMicroCorpusFailsMissingEvidence(t *testing.T) {
	r := foldMicroCorpus([]microCorpusCase{{Task: microCorpusTask{ID: "bad"}, ExecutionVerdict: "FAIL"}})
	if r.ExecutionVerdict != "FAIL" || r.Totals.BaselineCostUSD != nil || r.Totals.BaselineCostStatus != "provider-unreported" {
		t.Fatalf("missing evidence became pass/value: %+v", r)
	}
}
func TestPinnedMicroCorpusIsSmallAndComplexityBucketed(t *testing.T) {
	if len(pinnedMicroCorpus) != 3 {
		t.Fatalf("tasks=%d, want deliberately narrow corpus of 3", len(pinnedMicroCorpus))
	}
	seen := map[string]bool{}
	for _, task := range pinnedMicroCorpus {
		if task.ID == "" || task.Task == "" || task.Expected == "" || task.Complexity == "" {
			t.Fatalf("incomplete task: %+v", task)
		}
		if seen[task.Complexity] {
			t.Fatalf("duplicate complexity bucket %q", task.Complexity)
		}
		seen[task.Complexity] = true
	}
}
