package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
)

func TestGuardStopFloorPinned(t *testing.T) {
	pred := dojo.Registry.MustPredict("guard-stop", "bad_stop_block_rate", "fraction")
	if pred.Claimed != 1 || pred.LowerIsBetter || !pred.IntentionalFloor || pred.Basis == "" {
		t.Fatalf("guard-stop floor drifted: %+v", pred)
	}
}

func TestGuardStopLeverScoresCatchRateAndBreach(t *testing.T) {
	root := t.TempDir()
	ledger := filepath.Join(root, "docs", "nightrun", "guard-stops.jsonl")
	if err := os.MkdirAll(filepath.Dir(ledger), 0o755); err != nil {
		t.Fatal(err)
	}
	rows := `{"kind":"continue","blocked":true}` + "\n" +
		`{"kind":"continue","blocked":true}` + "\n" +
		`{"kind":"failopen"}` + "\n" +
		`{"kind":"clean"}` + "\n"
	if err := os.WriteFile(ledger, []byte(rows), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := (guardStopLever{root: root}).Episodes(dojo.Scenario{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("episodes=%d, want 1", len(got))
	}
	in := got[0]
	if in.Outcome.Sample != 3 || in.Outcome.Realized != 2.0/3.0 || !in.Outcome.Measured {
		t.Fatalf("outcome=%+v, want two catches across three bad stops", in.Outcome)
	}
	ep := dojo.Score("guard-corpus", in.Prediction, in.Outcome, dojo.DefaultCalibBand())
	folded := dojo.FoldCalibrable([]dojo.Episode{ep})
	if folded.FloorBreachErr <= 0 || folded.Value != folded.FloorBreachErr {
		t.Fatalf("miss must be breach-folded, not estimate-averaged: %+v", folded)
	}
}

func TestGuardStopLeverUnmeasuredWithoutLabeledRows(t *testing.T) {
	got, err := (guardStopLever{root: t.TempDir()}).Episodes(dojo.Scenario{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Outcome.Measured || got[0].Outcome.Sample != 0 {
		t.Fatalf("episodes=%+v, want one honest UNMEASURED floor cell", got)
	}
}
