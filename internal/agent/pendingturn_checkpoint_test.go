package agent

// pendingturn_checkpoint_test.go — the #4122 loop-side witness (epic #1193, primitive
// #1363): a retry inside HTTPPlanner.Complete writes a WRITE-AHEAD PendingTurn checkpoint
// keyed on the run's trace, so a kill -9 mid-retry resumes the in-flight turn instead of a
// fresh turn-0. The session-layer PendingTurn primitive (SetPendingTurn/IsZero/persist) is
// already committed; this proves the agent loop actually DRIVES it at the retry boundary.
//
// Without chat.go firing p.PendingTurnCheckpoint on the retry (and loop_session.go binding
// it to the drive sink), the table/gate would never see the checkpoint and every assertion
// below reads the zero value — the RED these tests turn GREEN.

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// retryOnce429ThenOK is the minimal upstream that forces EXACTLY one retry: the first
// request 429s, the second returns a clean final answer. One retry means the checkpoint
// hook fires once, with the 1-based attempt now in progress (2) and the 429 that triggered
// it — the canonical {Attempt:2, LastStatus:429} fixture the session tests already encode.
func retryOnce429ThenOK(t *testing.T) *httptest.Server {
	t.Helper()
	var n int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n++
		if n == 1 {
			w.WriteHeader(http.StatusTooManyRequests) // 429 once
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestPendingTurnCheckpoint_RetryWritesToTable is the primary #4122 witness: driving one
// retry through the loop's concrete-table binding writes SetPendingTurn(trace, {2,429,now})
// — the write-ahead record a restart re-attaches. It also pins the race-safety design:
// bindPendingCheckpoint returns a per-run CLONE and leaves the shared planner's hook nil,
// so concurrent arms sharing one *HTTPPlanner never cross-write each other's trace.
func TestPendingTurnCheckpoint_RetryWritesToTable(t *testing.T) {
	srv := retryOnce429ThenOK(t)
	const trace = "pt-4122-table"
	tbl := session.NewTable()

	p := NewHTTPPlanner(srv.URL, "m", "")
	bound := bindPendingCheckpoint(p, resolveRunConfig([]RunOption{WithSessionTable(tbl, trace)}))
	hp, ok := bound.(*HTTPPlanner)
	if !ok {
		t.Fatalf("bindPendingCheckpoint returned %T, want *HTTPPlanner", bound)
	}
	if hp == p {
		t.Fatal("bindPendingCheckpoint mutated the shared planner in place — a per-run clone is required so concurrent arms don't cross-write traces")
	}
	if p.PendingTurnCheckpoint != nil {
		t.Fatal("the shared planner's PendingTurnCheckpoint was set — binding must touch only the clone")
	}

	// #4123 clears the checkpoint on the recovering 200, so the post-Complete Get() is zero.
	// Witness the DURABLE write at the retry boundary via the every-revision stream — a kill -9
	// during the backoff sleep would have persisted exactly this streamed revision.
	var wrote session.PendingTurn
	tbl.WatchRevisions(func(st session.State) {
		if !st.PendingTurn.IsZero() {
			wrote = st.PendingTurn
		}
	})

	if _, err := hp.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
		t.Fatalf("Complete after one 429: %v", err)
	}

	if wrote.Attempt != 2 {
		t.Errorf("streamed PendingTurn.Attempt = %d, want 2 (the 1-based attempt now in progress on the first retry)", wrote.Attempt)
	}
	if wrote.LastStatus != 429 {
		t.Errorf("streamed PendingTurn.LastStatus = %d, want 429 (the status that triggered the retry)", wrote.LastStatus)
	}
	if wrote.StartedAtUnixNano <= 0 {
		t.Errorf("streamed PendingTurn.StartedAtUnixNano = %d, want the turn's start instant (>0)", wrote.StartedAtUnixNano)
	}
	if pt := tbl.Get(trace).PendingTurn; !pt.IsZero() {
		t.Errorf("post-Complete table still carries %+v, want the zero checkpoint (#4123 clears on success)", pt)
	}
}

// TestPendingTurnCheckpoint_RetryWritesToGate witnesses the function-shaped SessionGate
// twin (the gateway native-serve path holds Decide/Debit hooks, not a *session.Table): the
// checkpoint routes through gate.Checkpoint keyed on this run's trace, preferred over any
// table exactly like the Decide/Debit seam.
func TestPendingTurnCheckpoint_RetryWritesToGate(t *testing.T) {
	srv := retryOnce429ThenOK(t)
	const trace = "pt-4122-gate"

	type call struct {
		trace               string
		attempt, lastStatus int
		start               int64
	}
	var calls []call
	gate := SessionGate{Checkpoint: func(tr string, attempt, lastStatus int, startedAtUnixNano int64) {
		calls = append(calls, call{tr, attempt, lastStatus, startedAtUnixNano})
	}}

	p := NewHTTPPlanner(srv.URL, "m", "")
	bound := bindPendingCheckpoint(p, resolveRunConfig([]RunOption{WithSessionGate(gate, trace)}))
	hp := bound.(*HTTPPlanner)
	if _, err := hp.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
		t.Fatalf("Complete after one 429: %v", err)
	}

	// The retry WRITE routes through gate.Checkpoint keyed on this run's trace; #4123's success
	// CLEAR trails it as a zero-value call on the same trace.
	if len(calls) != 2 {
		t.Fatalf("gate.Checkpoint fired %d times, want 2 (the retry write then the #4123 clear): %+v", len(calls), calls)
	}
	if calls[0].trace != trace {
		t.Errorf("checkpoint trace = %q, want %q", calls[0].trace, trace)
	}
	if calls[0].attempt != 2 || calls[0].lastStatus != 429 || calls[0].start <= 0 {
		t.Errorf("write call = {attempt:%d,status:%d,start:%d}, want {2,429,>0}", calls[0].attempt, calls[0].lastStatus, calls[0].start)
	}
	if last := calls[len(calls)-1]; last != (call{trace, 0, 0, 0}) {
		t.Errorf("last call = %+v, want the (%q) zero-value clear", last, trace)
	}
}

// TestPendingTurnClear_FirstTrySuccessNetsToZero is the bound-sink net-state twin of the raw-hook
// clear witnesses in pendingturn_write_test.go. Post-#4123 a first-try success FIRES the clear hook
// (zero value) rather than staying silent, but the drive state a restart reads through the concrete
// table nets to the zero PendingTurn all the same — a healthy turn leaves nothing in flight to
// re-attach. (Pre-#4123 this asserted the hook was retry-only; the clear-on-completion leaf reverses
// the hook contract while preserving this net-zero table invariant.)
func TestPendingTurnClear_FirstTrySuccessNetsToZero(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(srv.Close)
	const trace = "pt-4122-silent"
	tbl := session.NewTable()

	p := NewHTTPPlanner(srv.URL, "m", "")
	hp := bindPendingCheckpoint(p, resolveRunConfig([]RunOption{WithSessionTable(tbl, trace)})).(*HTTPPlanner)
	if _, err := hp.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if pt := tbl.Get(trace).PendingTurn; !pt.IsZero() {
		t.Fatalf("first-try success wrote a checkpoint %+v, want none (the hook is retry-only)", pt)
	}
}
