package main

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

func TestGuardTrajectoryWarningNamesStalledObjective(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	t.Setenv(guardTrajctlEnvLedger, ledger)
	t.Setenv(guardTrajctlEnvMode, "on")
	rows := []trajctl.Row{
		trajctl.ObjectiveRecord(trajctl.Objective{ID: "ship", Statement: "ship unattended", Status: trajctl.StatusActive}),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "ship", Method: "commit-progress", Version: "1", Witness: trajctl.W3, Value: .4, UnixMillis: 1}),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "ship", Method: "commit-progress", Version: "1", Witness: trajctl.W3, Value: .4, UnixMillis: 2}),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "ship", Method: trajctl.ActivityDivergenceScorerMethod, Version: "1", Witness: trajctl.W2, Value: 1, UnixMillis: 2}),
	}
	for _, row := range rows {
		if err := trajctl.Append(ledger, row); err != nil {
			t.Fatal(err)
		}
	}
	got := guardTrajectoryWarningLine()
	for _, want := range []string{"trajectory warning:", "objective ship", "STALL", "flat progress"} {
		if !strings.Contains(got, want) {
			t.Fatalf("warning %q missing %q", got, want)
		}
	}
}

func TestGuardTrajectoryWarningIsSilentWhenHealthyOrDisabled(t *testing.T) {
	t.Setenv(guardTrajctlEnvLedger, "")
	if got := guardTrajectoryWarningLine(); got != "" {
		t.Fatalf("disabled warning=%q", got)
	}
}
