package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestFilterSelectorSelfcheck(t *testing.T) {
	r, err := buildSelectorReport(context.Background(), 16)
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "PASS" || r.Confusion.Wrong != 0 {
		t.Fatalf("selector verdict/confusion: %+v", r)
	}
	if r.Adaptive.TotalCostUnits >= r.DeterministicOnly.TotalCostUnits || r.Adaptive.TotalCostUnits >= r.AlwaysRun.TotalCostUnits {
		t.Fatalf("adaptive cost=%d deterministic=%d always=%d", r.Adaptive.TotalCostUnits, r.DeterministicOnly.TotalCostUnits, r.AlwaysRun.TotalCostUnits)
	}
	if r.StageInvocations["widen(issue-neighborhood)"] != 1 || r.StageInvocations["escalate"] != 1 {
		t.Fatalf("stage counts=%v", r.StageInvocations)
	}
	if r.CacheHits != r.Adaptive.SelectorCalls || r.Replay.SelectorCalls != 0 {
		t.Fatalf("replay: hits=%d adaptive_calls=%d replay=%+v", r.CacheHits, r.Adaptive.SelectorCalls, r.Replay)
	}
}

func TestFilterSelectorVerifierRejectsQualityCostAuthorityAndReplayRegressions(t *testing.T) {
	r, err := buildSelectorReport(context.Background(), 8)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name   string
		mutate func(*selectorReport)
	}{
		{"false negative", func(r *selectorReport) { r.Adaptive.FalseNegatives++ }},
		{"cost regression", func(r *selectorReport) { r.Adaptive.TotalCostUnits = r.DeterministicOnly.TotalCostUnits }},
		{"missing widen", func(r *selectorReport) { r.StageInvocations["widen(issue-neighborhood)"] = 0 }},
		{"worker overflow", func(r *selectorReport) { r.PeakSelectorFlight = int64(r.PhysicalWorkers + 1) }},
		{"cache replay miss", func(r *selectorReport) { r.Replay.SelectorCalls = 1 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bad := r
			bad.StageInvocations = cloneIntMap(r.StageInvocations)
			tt.mutate(&bad)
			if err := verifySelectorReport(bad); err == nil {
				t.Fatal("expected verifier refusal")
			}
		})
	}
}

func TestFilterSelectorCatalogDeniesArbitraryStage(t *testing.T) {
	if selectorStageAllowed("run(shell --arbitrary)") {
		t.Fatal("arbitrary selector output acquired execution authority")
	}
}

func TestVerifyFilterSelectorArtifact(t *testing.T) {
	r, err := buildSelectorReport(context.Background(), 8)
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(compactSelectorReport(r))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "selector.json")
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := verifySelectorArtifact(path); err != nil {
		t.Fatal(err)
	}
}

func cloneIntMap(src map[string]int) map[string]int {
	dst := make(map[string]int, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}
