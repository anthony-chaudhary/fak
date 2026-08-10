package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestNaturalCorpusHasFiveHeldOutEvidenceClasses(t *testing.T) {
	root := filepath.Join("..", "..", "experiments", "microcontext")
	if err := verifyNatural(filepath.Join(root, "s8s-natural-multitool-corpus-2026-08-10.json"), filepath.Join(root, "s8s-natural-fold-2026-08-10.json")); err != nil {
		t.Fatal(err)
	}
}
func TestNaturalSurfaceKeepsQualityGateAheadOfCost(t *testing.T) {
	p := filepath.Join("..", "..", "experiments", "microcontext", "s8t-natural-multitool-surface-2026-08-10.json")
	if err := verifyNaturalSurface(p); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(p)
	var r naturalSurface
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	for _, s := range r.Summaries {
		if s.Policy == "deterministic" && s.Quality >= 1 {
			t.Fatal("expected natural planner misses")
		}
		if s.PromptTokens < 0 || s.OutputTokens < 0 {
			t.Fatal("negative usage")
		}
	}
}
