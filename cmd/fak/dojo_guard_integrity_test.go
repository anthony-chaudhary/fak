package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
)

func TestGuardIntegrityFloorPinned(t *testing.T) {
	pred := dojo.Registry.MustPredict("guard-integrity", "bad_stop_leak_rate", "fraction")
	if pred.Claimed != 0 || !pred.LowerIsBetter || !pred.IntentionalFloor || pred.Basis == "" {
		t.Fatalf("guard-integrity floor drifted: %+v", pred)
	}
}

func TestGuardIntegrityLeverScoresLeakRate(t *testing.T) {
	root := t.TempDir()
	ledger := filepath.Join(root, "docs", "nightrun", "guard-stops.jsonl")
	if err := os.MkdirAll(filepath.Dir(ledger), 0o755); err != nil {
		t.Fatal(err)
	}
	rows := "" +
		`{"kind":"continue","blocked":true}` + "\n" +
		`{"kind":"continue","blocked":true}` + "\n" +
		`{"kind":"standdown"}` + "\n" +
		`{"kind":"clean"}` + "\n" +
		`not-json` + "\n"
	if err := os.WriteFile(ledger, []byte(rows), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := (guardIntegrityLever{root: root}).Episodes(dojo.Scenario{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("episodes=%d, want 1", len(got))
	}
	in := got[0]
	if in.Outcome.Sample != 3 || in.Outcome.Realized != 1.0/3.0 || !in.Outcome.Measured {
		t.Fatalf("outcome=%+v, want one leak across three labeled bad stops", in.Outcome)
	}
	ep := dojo.Score("guard-corpus", in.Prediction, in.Outcome, dojo.DefaultCalibBand())
	if got := dojo.FloorRespectErr(ep); got <= 0 {
		t.Fatalf("leak must breach floor, respect_err=%v episode=%+v", got, ep)
	}
}

func TestGuardIntegrityLeverUnmeasuredWithoutLabeledRows(t *testing.T) {
	got, err := (guardIntegrityLever{root: t.TempDir()}).Episodes(dojo.Scenario{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Outcome.Measured || got[0].Outcome.Sample != 0 {
		t.Fatalf("episodes=%+v, want one honest UNMEASURED floor cell", got)
	}
}
