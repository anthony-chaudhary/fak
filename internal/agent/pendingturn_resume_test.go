package agent

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// pendingturn_resume_test.go is the #4124 witness: the READ half of the #1363 write-ahead turn
// checkpoint. #4122/#4123 gave the loop a WRITER (checkpointPending -> table.SetPendingTurn) so a
// retry that dies mid-flight leaves a durable "attempt N was in progress" mark; this proves the
// loop READS that mark at entry and re-enters the checkpointed turn instead of starting a fresh
// turn-0 that has forgotten the lost attempt.

// TestRunArmResumesPendingTurnCheckpoint restores a session carrying a non-zero PendingTurn,
// starts RunArm keyed on it, and asserts the resume seam reports the exact checkpoint (attempt
// 2) — the loop OBSERVED the in-flight turn, it did not treat it as fresh. Driven by the offline
// MockPlanner, so no live upstream is touched.
func TestRunArmResumesPendingTurnCheckpoint(t *testing.T) {
	tbl := session.NewTable()
	const trace = "resume-arm"
	pt := session.PendingTurn{Attempt: 2, LastStatus: 429, StartedAtUnixNano: 1_780_000_000_000_000_000}
	if _, ok := tbl.SetPendingTurn(trace, pt); !ok {
		t.Fatal("setup: SetPendingTurn rejected on a live session")
	}

	m, err := RunArm(context.Background(), NewMockPlanner("mock"), DefaultTask, false, 20, nil,
		WithSessionTable(tbl, trace))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	if m.ResumedPendingTurn != pt {
		t.Fatalf("resume seam = %+v, want the restored checkpoint %+v", m.ResumedPendingTurn, pt)
	}
	if m.ResumedPendingTurn.IsZero() {
		t.Fatal("resumed run reported a zero checkpoint — the loop treated a restored in-flight turn as fresh turn-0")
	}
	// The resumed loop still RAN to completion — the checkpoint read informs the run, it does
	// not stall it (the out-of-scope backoff fast-forward is a separate follow-on).
	if m.FinalAnswer == "" {
		t.Fatal("resumed run produced no final answer; the read must not stall the loop")
	}
}

// TestRunArmNoPendingTurnIsFreshTurnZero is the non-vacuous mirror: with nothing checkpointed the
// resume seam stays zero, so a green TestRunArmResumesPendingTurnCheckpoint cannot be an artifact
// of the field always reporting a value. Covers both the no-session and the wired-but-empty paths.
func TestRunArmNoPendingTurnIsFreshTurnZero(t *testing.T) {
	// (a) No session wired at all — the historical loop.
	m, err := RunArm(context.Background(), NewMockPlanner("mock"), DefaultTask, false, 20, nil)
	if err != nil {
		t.Fatalf("RunArm(no session): %v", err)
	}
	if !m.ResumedPendingTurn.IsZero() {
		t.Fatalf("no-session run reported a resume checkpoint %+v, want zero", m.ResumedPendingTurn)
	}

	// (b) Session wired, budget set, but no turn was ever checkpointed for this trace.
	tbl := session.NewTable()
	const trace = "fresh-arm"
	tbl.SetBudget(trace, session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded})
	m2, err := RunArm(context.Background(), NewMockPlanner("mock"), DefaultTask, false, 20, nil,
		WithSessionTable(tbl, trace))
	if err != nil {
		t.Fatalf("RunArm(fresh session): %v", err)
	}
	if !m2.ResumedPendingTurn.IsZero() {
		t.Fatalf("fresh session reported a resume checkpoint %+v, want zero", m2.ResumedPendingTurn)
	}
}

// TestRunArmResumesPendingTurnViaGate proves the function-shaped path: the gateway native loop
// carries Decide/ResumeCheckpoint hooks, not a *session.Table, so it reads the checkpoint through
// the gate's ResumeCheckpoint twin — the same seam WithSessionGate.Checkpoint writes.
func TestRunArmResumesPendingTurnViaGate(t *testing.T) {
	const trace = "gate-arm"
	gate := SessionGate{
		// Proceed with no cap so the mock task runs to completion.
		Decide: func(string) (int, bool, int, string) { return 0, true, 0, "" },
		ResumeCheckpoint: func(tr string) (int, int, int64) {
			if tr != trace {
				return 0, 0, 0
			}
			return 2, 429, 1_780_000_000_000_000_000
		},
	}

	m, err := RunArm(context.Background(), NewMockPlanner("mock"), DefaultTask, false, 20, nil,
		WithSessionGate(gate, trace))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	want := session.PendingTurn{Attempt: 2, LastStatus: 429, StartedAtUnixNano: 1_780_000_000_000_000_000}
	if m.ResumedPendingTurn != want {
		t.Fatalf("gate resume seam = %+v, want %+v", m.ResumedPendingTurn, want)
	}
}
