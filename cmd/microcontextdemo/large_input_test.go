package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestLargeInputOperatorSelfcheck(t *testing.T) {
	r, err := buildLargeInputReport(context.Background(), 1000, 16)
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "PASS" {
		t.Fatalf("verdict=%q", r.Verdict)
	}
	if r.Baseline.Records != 1000 || !passAccounted(r.Baseline) {
		t.Fatalf("baseline accounting: %+v", r.Baseline)
	}
	if r.Unchanged.SemanticCalls != 0 || r.Unchanged.CacheHits != r.Baseline.SemanticCalls {
		t.Fatalf("unchanged cache witness: baseline calls=%d unchanged=%+v", r.Baseline.SemanticCalls, r.Unchanged)
	}
	if r.Mutated.SemanticCalls != 1 || r.PreciseInvalidations != 1 {
		t.Fatalf("mutation invalidation: %+v", r.Mutated)
	}
	if r.Baseline.PeakInFlight > 16 || r.Baseline.FoldMaxInput > largeInputFanIn {
		t.Fatalf("bounds: peak=%d fold=%d", r.Baseline.PeakInFlight, r.Baseline.FoldMaxInput)
	}
	if !r.OracleVerified || !sameInts(r.Mutated.CitedIssueIDs, r.Mutated.OracleIssueIDs) {
		t.Fatalf("oracle mismatch: cited=%v oracle=%v", r.Mutated.CitedIssueIDs, r.Mutated.OracleIssueIDs)
	}
}

func TestLargeInputVerifierRejectsCoverageCacheAndFoldRegressions(t *testing.T) {
	r, err := buildLargeInputReport(context.Background(), 1000, 8)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name   string
		mutate func(*largeInputReport)
	}{
		{"unaccounted source", func(r *largeInputReport) { r.SourceAccountingVerified = false }},
		{"unreused unchanged stage", func(r *largeInputReport) { r.Unchanged.SemanticCalls = 1 }},
		{"broad invalidation", func(r *largeInputReport) { r.Mutated.SemanticCalls = 2 }},
		{"unbounded reducer", func(r *largeInputReport) { r.ReducerBounded = false }},
		{"oracle mismatch", func(r *largeInputReport) { r.OracleVerified = false }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bad := r
			tt.mutate(&bad)
			if err := verifyLargeInputReport(bad); err == nil {
				t.Fatal("expected verifier refusal")
			}
		})
	}
}

func TestVerifyLargeInputArtifact(t *testing.T) {
	r, err := buildLargeInputReport(context.Background(), 1000, 8)
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "artifact.json")
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := verifyLargeInputArtifact(path); err != nil {
		t.Fatal(err)
	}
}
