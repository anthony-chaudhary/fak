package agent

import (
	"context"
	"net/http"
	"net/http/httptest"
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

// TestPendingTurnResumeReadIsNonDestructive isolates the READ-ONLY contract of the #4124 loop-entry
// resume read, and doubles as the witness the issue's documented acceptance selector
// (`go test ./internal/agent -run 'PendingTurnResume'`) actually SELECTS: the pre-existing witnesses
// are named "...ResumesPendingTurn..." / "...PendingTurnKill9Resume...", none of which the literal
// 'PendingTurnResume' pattern matches, so that command otherwise reports "no tests to run" — a
// vacuous green. Observing the checkpoint at loop entry must NOT consume it: clearing an in-flight
// turn is the turn's OWN job on completion (#4123, via the HTTP retry path), never a side effect of
// the read. The offline MockPlanner installs no checkpoint-clear hook (bindPendingCheckpoint no-ops
// off the *HTTPPlanner path), so a checkpoint that survives the whole run proves the entry read left
// it intact — a destructive read would have zeroed it.
func TestPendingTurnResumeReadIsNonDestructive(t *testing.T) {
	tbl := session.NewTable()
	const trace = "resume-readonly"
	pt := session.PendingTurn{Attempt: 3, LastStatus: 503, StartedAtUnixNano: 1_781_000_000_000_000_000}
	if _, ok := tbl.SetPendingTurn(trace, pt); !ok {
		t.Fatal("setup: SetPendingTurn rejected on a live session")
	}

	m, err := RunArm(context.Background(), NewMockPlanner("mock"), DefaultTask, false, 20, nil,
		WithSessionTable(tbl, trace))
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}
	// The loop RE-ENTERED the checkpointed turn (attempt 3), not a fresh turn-0.
	if m.ResumedPendingTurn != pt {
		t.Fatalf("resume seam = %+v, want the restored checkpoint %+v", m.ResumedPendingTurn, pt)
	}
	// ...and reading it did NOT clear it: the table still carries the exact checkpoint after the run,
	// because the mock path installs no clear hook. A destructive loop-entry read would zero this.
	if got := tbl.Get(trace).PendingTurn; got != pt {
		t.Fatalf("loop-entry read mutated the checkpoint: table now carries %+v, want the untouched %+v", got, pt)
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

// TestPendingTurnKill9ResumeReRunsExactTurn is the #1363 DoD witness (epic #1193): the whole
// reason the write-ahead turn checkpoint exists is so a kill -9 mid-retry can resume and re-run
// the EXACT turn. It composes the three cluster-C code leaves into one end-to-end assertion at the
// internal/agent layer — the LIVE-LOOP counterpart to the pure STORAGE round-trip that
// internal/session/descriptor_test.go:TestRegistryRestartReattachesPendingTurnCheckpoint proves
// (referenced here, not duplicated). Each leaf has its OWN assertion, so reverting any one turns
// this RED:
//   - WRITE (#4122): a real 429 retry writes {Attempt:2,LastStatus:429,start} at the retry boundary.
//   - kill -9: the live planner + local retryState vanish; a fresh Table over the restored State
//     (carrying the checkpoint the crashed process persisted) is process B.
//   - READ (#4124): the resumed loop re-enters the checkpointed turn — the resume seam reports the
//     EXACT checkpoint (attempt 2), not a fresh turn-0.
//   - CLEAR (#4123): the resumed turn succeeds against the recovered upstream, so the checkpoint
//     clears — a healthy turn leaves nothing in flight for the NEXT restart to re-attach.
func TestPendingTurnKill9ResumeReRunsExactTurn(t *testing.T) {
	const trace = "pt-4125-kill9"
	// The SAME task drives both the crashed attempt and the resume, so the re-run is the exact
	// turn (same messages), not a different one that merely happens to carry a checkpoint.
	const task = "book a flight from SFO to JFK"

	// --- Phase 1 (pre-crash): a REAL retry against a 429 writes the write-ahead checkpoint. ---
	// retryOnce429ThenOK (in-package) 429s once, forcing exactly one retry, so chat.go's retry
	// boundary fires PendingTurnCheckpoint(2,429,start) BEFORE the backoff sleep — the durable
	// record a kill -9 during that wait would have frozen.
	crashSrv := retryOnce429ThenOK(t)
	preTbl := session.NewTable()
	preHP := bindPendingCheckpoint(
		NewHTTPPlanner(crashSrv.URL, "m", ""),
		resolveRunConfig([]RunOption{WithSessionTable(preTbl, trace)}),
	).(*HTTPPlanner)

	// Complete recovers on the 2nd try and #4123 clears the checkpoint, but a kill -9 during the
	// backoff would have persisted exactly this streamed revision; WatchRevisions records the
	// non-zero checkpoint the write boundary emitted before the recovering 200.
	var wrote session.PendingTurn
	preTbl.WatchRevisions(func(st session.State) {
		if !st.PendingTurn.IsZero() {
			wrote = st.PendingTurn
		}
	})
	if _, err := preHP.Complete(context.Background(), []Message{{Role: RoleUser, Content: task}}, nil); err != nil {
		t.Fatalf("phase 1 Complete after one 429: %v", err)
	}
	// WRITE leaf (#4122): the retry actually wrote a checkpoint. Revert the write and this is zero —
	// the whole kill-9 witness collapses here first.
	if wrote.Attempt != 2 || wrote.LastStatus != 429 || wrote.StartedAtUnixNano <= 0 {
		t.Fatalf("phase 1 did not write the checkpoint: got %+v, want {Attempt:2,LastStatus:429,StartedAt:>0}", wrote)
	}

	// --- kill -9: process A's live planner + local retryState are GONE. A brand-new Table over the
	// SAME restored State is process B. The Registry/Store round-trip that carries the checkpoint
	// across the restart is already witnessed in descriptor_test.go — referenced, not duplicated. ---
	resumed := session.NewTable()
	if !resumed.Get(trace).PendingTurn.IsZero() {
		t.Fatal("precondition: fresh table already carried a checkpoint; the resume witness would be vacuous")
	}
	restored := resumed.Restore(trace, session.State{
		TraceID:     trace,
		Run:         session.Running,
		Budget:      session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded},
		PendingTurn: wrote, // the checkpoint the crashed process persisted mid-retry
	})
	if restored.PendingTurn != wrote {
		t.Fatalf("restore lost the checkpoint: got %+v want %+v", restored.PendingTurn, wrote)
	}

	// --- resume: drive the loop against the RECOVERED upstream (now a clean 200). ---
	okSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(okSrv.Close)

	m, err := RunArm(context.Background(), NewHTTPPlanner(okSrv.URL, "m", ""), task, false, 8, nil,
		WithSessionTable(resumed, trace))
	if err != nil {
		t.Fatalf("resume RunArm: %v", err)
	}

	// READ leaf (#4124): the resumed loop RE-ENTERED the checkpointed turn — the resume seam reports
	// the EXACT checkpoint (same attempt AND same start instant), not a fresh turn-0. Revert the
	// loop-entry read and this is the zero value.
	if m.ResumedPendingTurn != wrote {
		t.Fatalf("resumed loop did not re-enter the checkpointed turn: ResumedPendingTurn=%+v want %+v (a fresh turn-0 is zero)", m.ResumedPendingTurn, wrote)
	}
	if m.ResumedPendingTurn.Attempt != 2 {
		t.Fatalf("resumed from attempt %d, want 2 (the checkpointed retry, not a fresh turn-0)", m.ResumedPendingTurn.Attempt)
	}
	if m.FinalAnswer == "" {
		t.Fatal("resumed turn produced no final answer; the re-run did not complete against the recovered upstream")
	}

	// CLEAR leaf (#4123): the resumed turn SUCCEEDED, so the checkpoint clears — a healthy turn
	// leaves nothing in flight for the NEXT restart to re-attach. Revert the clear and this stays
	// non-zero.
	if pt := resumed.Get(trace).PendingTurn; !pt.IsZero() {
		t.Fatalf("after the resumed turn succeeded the checkpoint still carries %+v, want the zero (cleared) checkpoint", pt)
	}
}
