package main

import (
	"github.com/anthony-chaudhary/fak/internal/trajctl"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDispatchStatusRendersWitnessedTrajectory(t *testing.T) {
	root := t.TempDir()
	ledger := filepath.Join(root, trajctl.DefaultLedgerRel)
	if err := os.MkdirAll(filepath.Dir(ledger), 0755); err != nil {
		t.Fatal(err)
	}
	rows := []trajctl.Row{trajctl.ObjectiveRecord(trajctl.Objective{ID: "ship", Statement: "ship", Status: trajctl.StatusActive}), trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "ship", Method: "commit-progress", Version: "1", Witness: trajctl.W3, Value: .5, UnixMillis: 1})}
	for _, r := range rows {
		if err := trajctl.Append(ledger, r); err != nil {
			t.Fatal(err)
		}
	}
	got := renderDispatchStatus(dispatchStatusScan(filepath.Join(root, "runs"), root))
	for _, want := range []string{"trajectory:", "ship  HEALTHY", "witnesses: W3=1"} {
		if !strings.Contains(got, want) {
			t.Fatalf("render %q missing %q", got, want)
		}
	}
}
func TestDispatchStatusOmitsEmptyTrajectory(t *testing.T) {
	got := renderDispatchStatus(dispatchStatusScan(filepath.Join(t.TempDir(), "runs"), t.TempDir()))
	if strings.Contains(got, "trajectory:") {
		t.Fatalf("render=%q", got)
	}
}
