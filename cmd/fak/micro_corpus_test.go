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

func TestApplyMeasuredAblationsPromotesOnlyWitnessedLayers(t *testing.T) {
	r := foldMicroCorpus(nil)
	r.RetryAblation = microRetryAblation{
		WithoutRetryCompleted: false,
		WithRetryCompleted:    true,
		WithoutRetryAttempts:  1,
		WithRetryAttempts:     2,
		Evidence:              []string{retryAblationFailure},
	}
	r.HistoryAblation = microHistoryAblation{
		TokenCap:                 64,
		Turns:                    24,
		NaiveRetainedPointer:     false,
		CompactedRetainedPointer: true,
		Compactions:              2,
		PeakTokens:               64,
		FinalTokens:              61,
	}
	r.ModeAblation = microModeAblation{StringCorrect: true, ToolCorrect: true, StringTokens: 23, ToolTokens: 19}
	r.VerifierAblation = microVerifierAblation{
		WithoutVerifierCompleted: true,
		WithVerifierCompleted:    false,
		WithVerifierCaught:       true,
		Readback:                 "artifact-absent",
		Evidence:                 "artifact-absent: independent readback found no claimed artifact",
	}
	applyMeasuredAblations(&r)
	if r.Ablations[0].Layer != "retry" || r.Ablations[0].Status != "PASS" {
		t.Fatalf("retry not promoted: %+v", r.Ablations)
	}
	if r.Ablations[1].Layer != "context" || r.Ablations[1].Status != "PASS" {
		t.Fatalf("context not promoted: %+v", r.Ablations)
	}
	if r.Ablations[2].Layer != "verify" || r.Ablations[2].Status != "PASS" {
		t.Fatalf("verifier not promoted: %+v", r.Ablations)
	}
	if r.Ablations[3].Layer != "mode" || r.Ablations[3].Status != "PASS" {
		t.Fatalf("mode not promoted: %+v", r.Ablations)
	}
	md := formatMicroCorpusReport(r)
	for _, want := range []string{"| retry | PASS |", "retry off: completed=false, attempts=1", "evidence re-fed verbatim: `" + retryAblationFailure + "`", "| context | PASS |", "durable pointer retained: naive=false, compacted=true", "| verify | PASS |", "independent readback: `artifact-absent`", "| mode | PASS |", "same extraction task correct: string=true, typed-tool=true"} {
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
