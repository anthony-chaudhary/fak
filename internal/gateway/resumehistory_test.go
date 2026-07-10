package gateway

// resumehistory_test.go — the fak_resume_history MCP tool (resumehistory.go): the ledger
// reader mirrors the CLI's loadResumeHistory, the path resolver mirrors defaultResumeLedger,
// and ResumeHistoryFor folds one session's rows through the shared resume.FoldSelfObservation.
// The invariants under test are the fail-closed ones: no rows / no ledger fold to the honest
// floor, and the progress-earned budget reads the launch spacing the CLI would.

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// writeLedger drops a resume_ledger.jsonl fixture into a temp dir and returns its path.
func writeLedger(t *testing.T, lines string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "resume_ledger.jsonl")
	if err := os.WriteFile(path, []byte(lines), 0o644); err != nil {
		t.Fatalf("write ledger fixture: %v", err)
	}
	return path
}

// TestResumeHistoryForFoldsLedger walks the ledger read + fold: two fired launches spaced past
// the progress gap earn +1 over the base budget, the launch count excludes bookkeeping rows,
// and the resolved provenance carries the session and ledger path.
func TestResumeHistoryForFoldsLedger(t *testing.T) {
	// Two launches 700s apart (>= ProgressGapSeconds) for sid, plus a deferred bookkeeping row
	// that must NOT count as an attempt, plus an unrelated session's row that must be ignored.
	ledger := writeLedger(t, ""+
		`{"ts":"2026-07-10T10:00:00Z","session":"sid-1","phase":"launched"}`+"\n"+
		`{"ts":"2026-07-10T10:05:00Z","session":"sid-1","phase":"deferred"}`+"\n"+
		`{"ts":"2026-07-10T10:11:40Z","session":"sid-1","phase":"launched"}`+"\n"+
		`{"ts":"2026-07-10T10:00:00Z","session":"other","phase":"launched"}`+"\n")

	s := &Server{}
	rep := s.ResumeHistoryFor(ResumeHistoryRequest{Session: "sid-1", Ledger: ledger})

	if rep.Schema != resumeHistorySchema {
		t.Errorf("schema = %q, want %q", rep.Schema, resumeHistorySchema)
	}
	if !rep.Resolved || rep.LedgerPath != ledger {
		t.Errorf("resolved=%v ledger=%q, want resolved with path %q", rep.Resolved, rep.LedgerPath, ledger)
	}
	if rep.Session != "sid-1" {
		t.Errorf("session = %q, want sid-1", rep.Session)
	}
	obs := rep.Observation
	if !obs.HasHistory {
		t.Error("has_history = false, want true (sid-1 has launch rows)")
	}
	if obs.Attempts != 2 {
		t.Errorf("attempts = %d, want 2 (deferred row is bookkeeping, not a launch)", obs.Attempts)
	}
	// base DefaultMaxResumeAttempts (8) + one progress interval (700s >= 600s gap) = 9.
	if want := resume.DefaultMaxResumeAttempts + 1; obs.EarnedBudget != want {
		t.Errorf("earned_budget = %d, want %d (one progress-spaced interval)", obs.EarnedBudget, want)
	}
	if obs.NewTurns != 0 || obs.Outcome != resume.OutcomeUnknown {
		t.Errorf("ledger-only fold must be conservative: new_turns=%d outcome=%v", obs.NewTurns, obs.Outcome)
	}
	if obs.NextHint == "" {
		t.Error("next_hint is empty; the fold always carries a closed self-advice line")
	}
}

// TestResumeHistoryForOperatorSettled confirms a manual consolidate row settles the session:
// OperatorSettled is true and the settled row is not miscounted as a fired attempt.
func TestResumeHistoryForOperatorSettled(t *testing.T) {
	ledger := writeLedger(t, ""+
		`{"ts":"2026-07-10T10:00:00Z","session":"sid-2","phase":"launched"}`+"\n"+
		`{"ts":"2026-07-10T12:00:00Z","session":"sid-2","action":"consolidate: operator closed"}`+"\n")

	s := &Server{}
	obs := s.ResumeHistoryFor(ResumeHistoryRequest{Session: "sid-2", Ledger: ledger}).Observation
	if !obs.OperatorSettled {
		t.Error("operator_settled = false, want true (a consolidate row is a manual settle)")
	}
	if obs.Attempts != 1 {
		t.Errorf("attempts = %d, want 1 (the consolidate row is an override, not a launch)", obs.Attempts)
	}
}

// TestResumeHistoryForNoRowsFloor confirms the fail-closed floor: a session with no ledger rows
// folds to has_history=false with zero attempts, never a fabricated recovery.
func TestResumeHistoryForNoRowsFloor(t *testing.T) {
	ledger := writeLedger(t, `{"ts":"2026-07-10T10:00:00Z","session":"someone-else","phase":"launched"}`+"\n")
	s := &Server{}
	rep := s.ResumeHistoryFor(ResumeHistoryRequest{Session: "ghost", Ledger: ledger})
	if !rep.Resolved {
		t.Error("resolved = false, want true (the ledger exists and was read)")
	}
	if rep.Observation.HasHistory || rep.Observation.Attempts != 0 {
		t.Errorf("floor expected: has_history=%v attempts=%d", rep.Observation.HasHistory, rep.Observation.Attempts)
	}
}

// TestResumeHistoryForUnresolvedLedger confirms that with no explicit ledger and no fleet env,
// the tool fails closed: resolved=false with a reason and the honest empty observation.
func TestResumeHistoryForUnresolvedLedger(t *testing.T) {
	t.Setenv("FLEET_REG_DIR", "")
	t.Setenv("FLEET_STATE_DIR", "")
	t.Setenv("LOCALAPPDATA", t.TempDir()) // a dir with no Fleet/registry subtree -> Stat fails

	s := &Server{}
	rep := s.ResumeHistoryFor(ResumeHistoryRequest{Session: "sid-3"})
	if rep.Resolved {
		t.Errorf("resolved = true, want false (no ledger path resolvable); path=%q", rep.LedgerPath)
	}
	if rep.Reason == "" {
		t.Error("reason is empty; an unresolved ledger must say why")
	}
	if rep.Observation.HasHistory {
		t.Error("has_history = true on an unresolved ledger; must be the empty floor")
	}
}

// TestResolveResumeLedgerPath walks the resolver's precedence: FLEET_REG_DIR wins, then
// FLEET_STATE_DIR/registry.
func TestResolveResumeLedgerPath(t *testing.T) {
	t.Run("reg dir wins", func(t *testing.T) {
		t.Setenv("FLEET_REG_DIR", "/reg")
		t.Setenv("FLEET_STATE_DIR", "/state")
		if got, want := resolveResumeLedgerPath(), filepath.Join("/reg", "resume_ledger.jsonl"); got != want {
			t.Errorf("path = %q, want %q", got, want)
		}
	})
	t.Run("state dir fallback", func(t *testing.T) {
		t.Setenv("FLEET_REG_DIR", "")
		t.Setenv("FLEET_STATE_DIR", "/state")
		if got, want := resolveResumeLedgerPath(), filepath.Join("/state", "registry", "resume_ledger.jsonl"); got != want {
			t.Errorf("path = %q, want %q", got, want)
		}
	})
}
