package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestDogfoodScorecardReproducesFrozenArtifact(t *testing.T) {
	root := filepath.Join("..", "..", "experiments", "crossaudit-dogfood-3859")
	receipts, err := os.ReadFile(filepath.Join(root, "receipt-envelopes.json"))
	if err != nil {
		t.Fatal(err)
	}
	var envs []receiptEnvelope
	if err := json.Unmarshal(receipts, &envs); err != nil {
		t.Fatal(err)
	}
	got, err := deriveScorecard(envs)
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(filepath.Join(root, "scorecard.json"))
	if err != nil {
		t.Fatal(err)
	}
	var wantValue, gotValue any
	if err := json.Unmarshal(want, &wantValue); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(got, &gotValue); err != nil {
		t.Fatal(err)
	}
	w, _ := json.Marshal(wantValue)
	g, _ := json.Marshal(gotValue)
	if string(w) != string(g) {
		t.Fatal("derived scorecard differs from frozen artifact")
	}
}
