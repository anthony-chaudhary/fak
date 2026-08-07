package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestS2BVerifierAcceptsCapturedArtifact(t *testing.T) {
	path := filepath.Join("..", "..", "experiments", "microcontext", "s2b-gcp-inkernel-prefix-ab-pass-2026-08-07.json")
	if err := verifyS2BArtifact(path); err != nil {
		t.Fatal(err)
	}
}

func TestS2BVerifierRejectsCounterDrift(t *testing.T) {
	path := filepath.Join("..", "..", "experiments", "microcontext", "s2b-gcp-inkernel-prefix-ab-pass-2026-08-07.json")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var report map[string]any
	if err := json.Unmarshal(b, &report); err != nil {
		t.Fatal(err)
	}
	shared, _ := object(report, "shared")
	shared["kernel_reused_tokens_delta"] = float64(1)
	if err := verifyS2BReport(report); err == nil {
		t.Fatal("expected counter refusal")
	}
}
