package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestGuardReadPriorRefusalCarryForwardUsesPreviousDefaultAuditSidecar(t *testing.T) {
	dir := t.TempDir()
	priorAudit := filepath.Join(dir, "interactive-old.jsonl")
	currentAudit := filepath.Join(dir, "interactive-new.jsonl")
	if err := guardWriteRefusalCarryForwardFile(priorAudit, "guard", []guardRefusalCarry{{
		Reason: "OFF_TRUNK",
		Count:  1,
		Fix:    "commit directly to main",
	}}, time.Unix(10, 0)); err != nil {
		t.Fatalf("write prior carry-forward: %v", err)
	}

	got := guardReadPriorRefusalCarryForward(currentAudit, "guard", dir)
	if len(got) != 1 || got[0].Reason != "OFF_TRUNK" || got[0].Count != 1 {
		t.Fatalf("prior carry-forward = %+v, want OFF_TRUNK x1 from previous audit path", got)
	}
}

func TestGuardReadPriorRefusalCarryForwardLatestCleanSidecarClearsOlder(t *testing.T) {
	dir := t.TempDir()
	oldAudit := filepath.Join(dir, "interactive-old.jsonl")
	cleanAudit := filepath.Join(dir, "interactive-clean.jsonl")
	currentAudit := filepath.Join(dir, "interactive-new.jsonl")
	if err := guardWriteRefusalCarryForwardFile(oldAudit, "guard", []guardRefusalCarry{{
		Reason: "OFF_TRUNK",
		Count:  1,
	}}, time.Unix(10, 0)); err != nil {
		t.Fatalf("write old carry-forward: %v", err)
	}
	if err := guardWriteRefusalCarryForwardFile(cleanAudit, "guard", nil, time.Unix(20, 0)); err != nil {
		t.Fatalf("write clean carry-forward: %v", err)
	}

	if got := guardReadPriorRefusalCarryForward(currentAudit, "guard", dir); len(got) != 0 {
		t.Fatalf("latest clean sidecar should clear older guard refusals, got %+v", got)
	}
}

func TestGuardReadPriorRefusalCarryForwardFallsBackToPreviousJournal(t *testing.T) {
	dir := t.TempDir()
	priorAudit := filepath.Join(dir, "interactive-old.jsonl")
	currentAudit := filepath.Join(dir, "interactive-new.jsonl")
	row := `{"seq":1,"kind":"DENY","trace_id":"guard","verdict":"DENY","reason":"POLICY_BLOCK"}` + "\n"
	if err := os.WriteFile(priorAudit, []byte(row), 0o644); err != nil {
		t.Fatalf("write prior journal: %v", err)
	}

	got := guardReadPriorRefusalCarryForward(currentAudit, "guard", dir)
	if len(got) != 1 || got[0].Reason != "POLICY_BLOCK" || got[0].Count != 1 {
		t.Fatalf("journal fallback carry-forward = %+v, want POLICY_BLOCK x1", got)
	}
}

func TestGuardRecoveryPromptNamesPriorRefusals(t *testing.T) {
	got := guardRecoveryPrompt([]guardRefusalCarry{{
		Reason: "OFF_TRUNK",
		Count:  2,
		Fix:    "commit directly to main",
	}})
	for _, want := range []string{
		"[fak] resume recovery",
		"recovery/debugging",
		"Do not re-propose the same refused call unchanged",
		"OFF_TRUNK x2",
		"commit directly to main",
		"Keep fak guard wrapped",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("recovery prompt missing %q:\n%s", want, got)
		}
	}
	if got := guardRecoveryPrompt([]guardRefusalCarry{{Count: 1}}); got != "" {
		t.Fatalf("empty reason prompt = %q, want empty", got)
	}
}
