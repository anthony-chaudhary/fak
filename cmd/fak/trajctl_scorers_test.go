package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// scorerLedger seeds a ledger with two opposite-outcome objectives and a good vs broken
// judge, returning the ledger path.
func scorerLedger(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	ledger := filepath.Join(root, filepath.FromSlash(trajctl.DefaultLedgerRel))
	rows := []trajctl.Row{
		trajctl.ObjectiveRecord(trajctl.Objective{ID: "objMet", Statement: "ship the met widget", Status: trajctl.StatusActive}),
		trajctl.ObjectiveRecord(trajctl.Objective{ID: "objFail", Statement: "ship the failed widget", Status: trajctl.StatusActive}),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "objMet", Method: trajctl.CommitScorerMethod, Version: "1", Value: 1.0, Witness: trajctl.W3, UnixMillis: 1}),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "objFail", Method: trajctl.CommitScorerMethod, Version: "1", Value: 0.0, Witness: trajctl.W3, UnixMillis: 1}),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "objMet", Method: "good-judge", Version: "1", Value: 0.9, Witness: trajctl.W1, UnixMillis: 2}),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "objFail", Method: "good-judge", Version: "1", Value: 0.1, Witness: trajctl.W1, UnixMillis: 2}),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "objMet", Method: "broken-judge", Version: "1", Value: 0.1, Witness: trajctl.W1, UnixMillis: 2}),
		trajctl.ScoreRecord(trajctl.ScoreRow{ObjectiveID: "objFail", Method: "broken-judge", Version: "1", Value: 0.9, Witness: trajctl.W1, UnixMillis: 2}),
	}
	for _, r := range rows {
		if err := trajctl.Append(ledger, r); err != nil {
			t.Fatalf("seed ledger: %v", err)
		}
	}
	return ledger
}

// TestTrajctlScorersJSONRanksBrokenLast is the end-to-end CLI witness (#2566): the
// leaderboard renders from a real ledger and the deliberately broken judge ranks last,
// annotated MISCALIBRATED, while the well-calibrated judge ranks first.
func TestTrajctlScorersJSONRanksBrokenLast(t *testing.T) {
	ledger := scorerLedger(t)
	var out, errb bytes.Buffer
	if code := runTrajctlScorers(&out, &errb, []string{"--ledger", ledger, "--json"}); code != 0 {
		t.Fatalf("scorers exit=%d stderr=%s", code, errb.String())
	}
	var rep trajctl.CalibrationReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("json: %v\n%s", err, out.String())
	}
	if rep.Schema != trajctl.CalibrationSchema {
		t.Errorf("schema = %q, want %q", rep.Schema, trajctl.CalibrationSchema)
	}
	if len(rep.Scorers) != 2 {
		t.Fatalf("want the two judges, got %d: %+v", len(rep.Scorers), rep.Scorers)
	}
	if rep.Scorers[0].Method != "good-judge" || rep.Scorers[0].Verdict != trajctl.CalibrationWell {
		t.Errorf("rank 1 should be the well-calibrated judge, got %s/%s", rep.Scorers[0].Method, rep.Scorers[0].Verdict)
	}
	if last := rep.Scorers[len(rep.Scorers)-1]; last.Method != "broken-judge" || last.Verdict != trajctl.CalibrationMiscalibrated {
		t.Errorf("broken-judge must rank last MISCALIBRATED, got %s/%s", last.Method, last.Verdict)
	}
}

// TestTrajctlScorersTextNamesRepairTarget covers the human render: it lists best-first and
// names the worst-first repair target at the foot.
func TestTrajctlScorersTextNamesRepairTarget(t *testing.T) {
	ledger := scorerLedger(t)
	var out, errb bytes.Buffer
	if code := runTrajctlScorers(&out, &errb, []string{"--ledger", ledger}); code != 0 {
		t.Fatalf("scorers exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "worst-first: repair broken-judge") {
		t.Errorf("render must name the worst-first repair target, got:\n%s", got)
	}
}
