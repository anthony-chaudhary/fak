package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestMultiTurnDescriptorWitnessReconciles1000ByK(t *testing.T) {
	r, err := runMultiTurnDescriptor(context.Background(), 1000, 16, 3)
	if err != nil {
		t.Fatal(err)
	}
	if r.ExpectedTurns != 3000 || r.AccountedTurns != 3000 || r.Completed != 1000 || r.VerifiedRestores != 1000 {
		t.Fatalf("report=%+v", r)
	}
	path := filepath.Join(t.TempDir(), "witness.json")
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := verifyMultiTurnDescriptorArtifact(path); err != nil {
		t.Fatal(err)
	}
}

func TestMultiTurnDescriptorVerifierRejectsMissingTurn(t *testing.T) {
	r, err := runMultiTurnDescriptor(context.Background(), 1000, 8, 2)
	if err != nil {
		t.Fatal(err)
	}
	r.AccountedTurns--
	if err := verifyMultiTurnDescriptorReport(r); err == nil {
		t.Fatal("expected turn-reconciliation refusal")
	}
}
