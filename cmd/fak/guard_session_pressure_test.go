package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
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
	rc := runGuardSessionPressureGate(&stderr, guardSessionPressureGateConfig{
		Threshold:       "high",
		SinceDays:       7,
		Max:             40,
		NamespacePrefix: "C--work-fak",
		Roots:           []string{root},
	})
	if rc != 1 {
		t.Fatalf("gate rc=%d stderr=%s", rc, stderr.String())
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
	})
	if rc != 0 {
		t.Fatalf("clean gate rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(stderr.String(), "session pressure gate allow") {
		t.Fatalf("clean gate did not report allow:\n%s", stderr.String())
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
