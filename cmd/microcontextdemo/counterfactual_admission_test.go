package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestCounterfactualCorpusPairsHoldContentConstant(t *testing.T) {
	root := filepath.Join("..", "..", "experiments", "microcontext")
	corpus := filepath.Join(root, "s8q-counterfactual-tool-corpus-2026-08-10.json")
	fold := filepath.Join(root, "s8q-counterfactual-fold-2026-08-10.json")
	if err := verifyCounterfactual(corpus, fold); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(corpus)
	if err != nil {
		t.Fatal(err)
	}
	var c counterfactualCorpus
	if err := json.Unmarshal(b, &c); err != nil {
		t.Fatal(err)
	}
	pairs := map[string][]counterfactualRecord{}
	for _, r := range c.Records {
		pairs[r.PairID] = append(pairs[r.PairID], r)
	}
	for id, pair := range pairs {
		if len(pair) != 2 || pair[0].Title != pair[1].Title || pair[0].Body != pair[1].Body || pair[0].Question == pair[1].Question {
			t.Fatalf("pair %s does not isolate question", id)
		}
	}
}

func TestTrueAdmissionDeclinesOnlyReadOnlyToolsAtMatchedQuality(t *testing.T) {
	p := filepath.Join("..", "..", "experiments", "microcontext", "s8r-true-tool-admission-2026-08-10.json")
	if err := verifyTrueAdmission(p); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	var r trueAdmissionReport
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	for _, x := range r.Receipts {
		if x.Policy == "true-two-stage" && x.ToolOpened != (x.Predicted == "current_state") {
			t.Fatalf("admission mismatch for %s", x.ID)
		}
	}
}
