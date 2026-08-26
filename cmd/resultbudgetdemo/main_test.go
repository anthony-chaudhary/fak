package main

import (
	"encoding/json"
	"os/exec"
	"testing"

	"github.com/anthony-chaudhary/fak/cmd/internal/democapture"
)

func TestResultBudgetSelfcheck(t *testing.T) {
	report, err := runSelfcheck()
	if err != nil {
		t.Fatal(err)
	}
	if report.Verdict != "PASS" || report.SelfCheck != "PASS" {
		t.Fatalf("unexpected report: %+v", report)
	}
	if err := validateReport(report); err != nil {
		t.Fatal(err)
	}
}

func TestSelfcheckCapturedOutput(t *testing.T) {
	cmd := exec.Command("go", "run", ".", "-selfcheck", "-pretty")
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("selfcheck: %v", err)
	}
	var report demoReport
	if err := json.Unmarshal(out, &report); err != nil {
		t.Fatalf("decode captured output: %v\n%s", err, out)
	}
	if report.SelfCheck != "PASS" {
		t.Fatalf("captured output did not pass: %s", out)
	}
	if err := validateReport(report); err != nil {
		t.Fatalf("captured witness: %v\n%s", err, out)
	}
	if err := democapture.MatchMarkdown("EXAMPLE-OUTPUT.md", out); err != nil {
		t.Fatal(err)
	}
}

func TestAdapterFailsSafeForUnknownTool(t *testing.T) {
	policy := shapingConfig{
		Kind: "fak/tool-result-budget", Version: "1.0.0", Name: "thimble/default", Mode: modeEnforce,
		DefaultBudget: resultBudget{Items: 10},
		Contracts: []argumentContract{{
			Tool: knownTool, Argument: resultArgument, Dimension: "items",
			Maximum: 10, Minimum: 1, SafeToReduce: true,
		}},
	}
	original := json.RawMessage(`{"per_page":500}`)
	effective, receipt, err := (budgetAdapter{policy: policy}).adapt(toolCall{Tool: "unknown", Args: original})
	if err != nil {
		t.Fatal(err)
	}
	if string(effective) != string(original) || receipt.Outcome != "pass" || receipt.Reason != "unknown_tool_contract" {
		t.Fatalf("effective=%s receipt=%+v", effective, receipt)
	}
}
