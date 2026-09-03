package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestHarnessCompareCLIAll(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runHarnessCompare(&stdout, &stderr, []string{})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d; stderr: %s", rc, stderr.String())
	}
	out := stdout.String()
	for _, expected := range []string{
		"FAK NATIVE HARNESS vs NEXT-BEST ALTERNATIVE (NBA) COMPARISON",
		"OpenCode",
		"OpenAI Codex",
		"Cursor",
		"Claude Code",
		"In-kernel vDSO FastPath",
		"Proactive pre-consumption barWrite",
		"In-syscall abi.VerdictTransform",
		"[LOW-EGO FIELD-BORROWING & LEARNING POINTS]",
		"[ARCHITECTURAL SEAMS & KERNEL ADVANTAGES]",
		"[UPSTREAM ADAPTATION STATUS]",
		"[NBA BENCHMARKING TARGETS]",
	} {
		if !strings.Contains(out, expected) {
			t.Errorf("CLI output missing expected substring %q", expected)
		}
	}
}

func TestHarnessCompareJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runHarnessCompare(&stdout, &stderr, []string{"--view", "json", "--baseline", "opencode"})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d; stderr: %s", rc, stderr.String())
	}

	var report HarnessComparisonReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to parse JSON report: %v", err)
	}

	if report.Schema != "fak.harness.comparison/v1" {
		t.Errorf("unexpected schema: %q", report.Schema)
	}
	if len(report.Comparisons) != 1 {
		t.Fatalf("expected 1 comparison, got %d", len(report.Comparisons))
	}
	cmp := report.Comparisons[0]
	if cmp.Key != "opencode" || cmp.Name != "OpenCode" {
		t.Errorf("unexpected comparison entry: %+v", cmp)
	}
	if cmp.NBABenchmarkTargets.TurnEfficiencyRatio <= 0 || cmp.NBABenchmarkTargets.TokenEconomyRatio <= 0 {
		t.Errorf("invalid NBA benchmark targets: %+v", cmp.NBABenchmarkTargets)
	}
}

func TestHarnessCompareBaselineFiltering(t *testing.T) {
	baselines := []string{"opencode", "codex", "cursor", "claude"}
	for _, b := range baselines {
		var stdout, stderr bytes.Buffer
		rc := runHarnessCompare(&stdout, &stderr, []string{"--baseline", b, "--view", "json"})
		if rc != 0 {
			t.Fatalf("failed for baseline %s: rc=%d, stderr=%s", b, rc, stderr.String())
		}
		var report HarnessComparisonReport
		if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
			t.Fatalf("failed to parse JSON for baseline %s: %v", b, err)
		}
		if len(report.Comparisons) != 1 || report.Comparisons[0].Key != b {
			t.Errorf("expected baseline %s, got %+v", b, report.Comparisons)
		}
	}
}

func TestHarnessCompareDimensionsFilter(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runHarnessCompare(&stdout, &stderr, []string{"--baseline", "opencode", "--dimensions", "architecture"})
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d; stderr: %s", rc, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "[ARCHITECTURAL SEAMS & KERNEL ADVANTAGES]") {
		t.Errorf("expected architecture section in output: %s", out)
	}
	if strings.Contains(out, "[LOW-EGO FIELD-BORROWING & LEARNING POINTS]") {
		t.Errorf("did not expect learning section when filtered to architecture: %s", out)
	}
}

func TestHarnessCompareInvalidBaseline(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runHarnessCompare(&stdout, &stderr, []string{"--baseline", "nonexistent"})
	if rc != 2 {
		t.Fatalf("expected rc 2 for invalid baseline, got %d", rc)
	}
	if !strings.Contains(stderr.String(), "unknown baseline") {
		t.Errorf("expected unknown baseline error message, got: %s", stderr.String())
	}
}

func TestHarnessSubcommandDispatch(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := runHarness(&stdout, &stderr, []string{"compare", "--baseline", "codex"})
	if rc != 0 {
		t.Fatalf("expected rc 0 from runHarness dispatch, got %d; stderr: %s", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "OpenAI Codex") {
		t.Errorf("expected OpenAI Codex in output, got: %s", stdout.String())
	}
}
