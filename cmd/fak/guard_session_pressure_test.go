package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

func TestGuardSessionPressureGateRefusesHighPressure(t *testing.T) {
	root := t.TempDir()
	writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "heavy.jsonl"), []map[string]any{
		sessionAuditAssistantDetailed("opus", 200, 0, 900_000, 50_000, "claude-opus-4-8", ""),
	})
	writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "fable.jsonl"), []map[string]any{
		sessionAuditAssistantDetailed("fable", 300, 0, 20_000, 1_000, "claude-fable-5", ""),
	})

	var stderr bytes.Buffer
	reportPath := filepath.Join(t.TempDir(), "pressure", "report.json")
	rc := runGuardSessionPressureGate(&stderr, guardSessionPressureGateConfig{
		Threshold:       "high",
		SinceDays:       7,
		Max:             40,
		NamespacePrefix: "C--work-fak",
		Roots:           []string{root},
		ReportPath:      reportPath,
	})
	if rc != 1 {
		t.Fatalf("gate rc=%d stderr=%s", rc, stderr.String())
	}
	var plan sessionaudit.CompactActionPlan
	raw, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("read report: %v", err)
	}
	if err := json.Unmarshal(raw, &plan); err != nil {
		t.Fatalf("decode report: %v\n%s", err, string(raw))
	}
	if plan.Schema != sessionaudit.CompactActionPlanSchema ||
		plan.Gate.Verdict != "refuse" ||
		plan.Gate.Refused != 2 ||
		plan.Counts.High != 2 {
		t.Fatalf("report plan = %+v", plan)
	}
	for _, want := range []string{
		"session pressure gate REFUSE",
		"opus_cost_pressure",
		"long_context_pressure",
		"checkpoint_reset_top_long_context",
	} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("stderr missing %q:\n%s", want, stderr.String())
		}
	}
}

func TestGuardSessionPressureGateAllowsWhenDisabledOrClean(t *testing.T) {
	var stderr bytes.Buffer
	if rc := runGuardSessionPressureGate(&stderr, guardSessionPressureGateConfig{Threshold: "off"}); rc != 0 {
		t.Fatalf("off gate rc=%d stderr=%s", rc, stderr.String())
	}

	root := t.TempDir()
	reportPath := filepath.Join(t.TempDir(), "report.json")
	writeSessionAuditJSONL(t, filepath.Join(root, "C--work-fak", "small.jsonl"), []map[string]any{
		sessionAuditAssistantDetailed("fable", 300, 10, 20, 5, "claude-fable-5", ""),
	})
	stderr.Reset()
	rc := runGuardSessionPressureGate(&stderr, guardSessionPressureGateConfig{
		Threshold:       "high",
		SinceDays:       7,
		Max:             40,
		NamespacePrefix: "C--work-fak",
		Roots:           []string{root},
		ReportPath:      reportPath,
	})
	if rc != 0 {
		t.Fatalf("clean gate rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stderr.String(), "session pressure gate allow") {
		t.Fatalf("clean gate did not report allow:\n%s", stderr.String())
	}
	raw, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("read allow report: %v", err)
	}
	var plan sessionaudit.CompactActionPlan
	if err := json.Unmarshal(raw, &plan); err != nil {
		t.Fatalf("decode allow report: %v\n%s", err, string(raw))
	}
	if plan.Gate.Verdict != "allow" || plan.Counts.Total != 0 {
		t.Fatalf("allow report plan = %+v", plan)
	}
}

func TestGuardSessionPressureGateRejectsBadThreshold(t *testing.T) {
	var stderr bytes.Buffer
	rc := runGuardSessionPressureGate(&stderr, guardSessionPressureGateConfig{
		Threshold:       "urgent",
		NamespacePrefix: "C--work-fak",
		Roots:           []string{t.TempDir()},
	})
	if rc != 2 || !strings.Contains(stderr.String(), "invalid --session-pressure-gate") {
		t.Fatalf("bad threshold rc=%d stderr=%s", rc, stderr.String())
	}
}
