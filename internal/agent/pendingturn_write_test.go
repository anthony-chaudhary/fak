package agent

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// completionOKBody is a minimal, well-formed OpenAI-shape chat completion — the buffered
// success body a recovered turn returns after its retry (mirrors retry_notify_test.go).
const completionOKBody = `{"model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`

// A retry inside Complete must WRITE the #1363 write-ahead checkpoint before the silent
// backoff sleep — the durable twin of RetryNotify. Here the upstream 429s once then 200s,
// so the FIRST checkpoint call is the write, carrying the 1-based attempt now in progress
// (2), the 429 that triggered it, and a real turn-start timestamp. This is the primitive's
// first live writer: before this leaf, grep -rn PendingTurn internal/agent/ was empty. Since
// #4123 the recovering 200 fires a trailing zero-value CLEAR, so the sequence is [write, clear];
// this test pins the WRITE (calls[0]); the clear is the subject of the PendingTurnClear tests.
func TestPendingTurnCheckpoint_WrittenOnRetry(t *testing.T) {
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&n, 1) == 1 {
			w.WriteHeader(http.StatusTooManyRequests) // 429 once, before the recovered 200
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(completionOKBody))
	}))
	t.Cleanup(srv.Close)

	type checkpoint struct {
		attempt, lastStatus int
		startedAtUnixNano   int64
	}
	var calls []checkpoint
	p := NewHTTPPlanner(srv.URL, "m", "")
	p.PendingTurnCheckpoint = func(attempt, lastStatus int, startedAtUnixNano int64) {
		calls = append(calls, checkpoint{attempt, lastStatus, startedAtUnixNano})
	}

	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Complete after 1x429: %v", err)
	}
	if comp.Message.Content != "ok" {
		t.Fatalf("content = %q, want ok", comp.Message.Content)
	}
	if len(calls) != 2 {
		t.Fatalf("PendingTurnCheckpoint fired %d times, want 2 (the retry WRITE then the #4123 success CLEAR): %+v", len(calls), calls)
	}
	if calls[0].attempt != 2 {
		t.Errorf("checkpoint attempt = %d, want 2 (the 1-based try now in progress)", calls[0].attempt)
	}
	if calls[0].lastStatus != http.StatusTooManyRequests {
		t.Errorf("checkpoint lastStatus = %d, want 429", calls[0].lastStatus)
	}
	if calls[0].startedAtUnixNano <= 0 {
		t.Errorf("checkpoint startedAtUnixNano = %d, want a real turn-start timestamp", calls[0].startedAtUnixNano)
	}
	if last := calls[len(calls)-1]; last != (checkpoint{}) {
		t.Errorf("final checkpoint = %+v, want the zero-value clear that #4123 fires on the recovering 200", last)
	}
}

// TestPendingTurnClear_ClearedAfterRetryRecovery is the primary #4123 witness (C-pending-turn-2,
// the symmetric other half of the #1363 retry checkpoint): a turn that 429s then 200s must, on
// success, drop the checkpoint it wrote back to the ZERO value through the SAME hook. Recording
// every Checkpoint call, the sequence is exactly [{2,429,>0}, {0,0,0}] — the retry-boundary WRITE
// followed by the completion CLEAR — so the LAST call is the zero value. Bound to a concrete table,
// the drive state a restart would read is IsZero(): nothing left in flight to re-attach.
func TestPendingTurnClear_ClearedAfterRetryRecovery(t *testing.T) {
	newRetryServer := func() *httptest.Server {
		var n int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if atomic.AddInt32(&n, 1) == 1 {
				w.WriteHeader(http.StatusTooManyRequests) // 429 once, before the recovered 200
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(completionOKBody))
		}))
		t.Cleanup(srv.Close)
		return srv
	}

	t.Run("raw hook records write then clear", func(t *testing.T) {
		type checkpoint struct {
			attempt, lastStatus int
			startedAtUnixNano   int64
		}
		var calls []checkpoint
		p := NewHTTPPlanner(newRetryServer().URL, "m", "")
		p.PendingTurnCheckpoint = func(attempt, lastStatus int, startedAtUnixNano int64) {
			calls = append(calls, checkpoint{attempt, lastStatus, startedAtUnixNano})
		}
		if _, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
			t.Fatalf("Complete after 1x429: %v", err)
		}
		if len(calls) != 2 {
			t.Fatalf("Checkpoint fired %d times, want 2 (the retry WRITE then the success CLEAR): %+v", len(calls), calls)
		}
		if calls[0] != (checkpoint{2, http.StatusTooManyRequests, calls[0].startedAtUnixNano}) || calls[0].startedAtUnixNano <= 0 {
			t.Errorf("first call = %+v, want the retry write {2,429,>0}", calls[0])
		}
		if last := calls[len(calls)-1]; last != (checkpoint{}) {
			t.Errorf("last call = %+v, want the zero-value clear {0,0,0}", last)
		}
	})

	t.Run("bound table nets to IsZero", func(t *testing.T) {
		tbl := session.NewTable()
		tbl.Restore("trace-x", session.State{TraceID: "trace-x", Run: session.Running, Budget: session.Budget{TurnsLeft: 5, TokensLeft: 500}})
		p := NewHTTPPlanner(newRetryServer().URL, "m", "")
		bound := bindPendingCheckpoint(p, resolveRunConfig([]RunOption{WithSessionTable(tbl, "trace-x")}))
		if _, err := bound.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
			t.Fatalf("Complete after 1x429: %v", err)
		}
		if pt := tbl.Get("trace-x").PendingTurn; !pt.IsZero() {
			t.Fatalf("after a recovered turn the table still carries %+v, want the zero checkpoint (nothing in flight)", pt)
		}
	})
}

// TestPendingTurnClear_FiresOnFirstTrySuccess pins the fast-path half of #4123: even a turn that
// never retried fires exactly one Checkpoint call on success, and it is the ZERO value. This is
// what lets a RESUMED process — one that re-entered a turn carrying a restored checkpoint and then
// completed it on the first attempt (the 429 having cleared) — drop that stale checkpoint. Complete
// cannot distinguish a fresh first-try from a resumed one, so the clear is unconditional on success;
// the net drive state stays IsZero() either way.
func TestPendingTurnClear_FiresOnFirstTrySuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(completionOKBody))
	}))
	t.Cleanup(srv.Close)

	type checkpoint struct {
		attempt, lastStatus int
		startedAtUnixNano   int64
	}
	var calls []checkpoint
	p := NewHTTPPlanner(srv.URL, "m", "")
	p.PendingTurnCheckpoint = func(attempt, lastStatus int, startedAtUnixNano int64) {
		calls = append(calls, checkpoint{attempt, lastStatus, startedAtUnixNano})
	}
	if _, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(calls) != 1 || calls[0] != (checkpoint{}) {
		t.Fatalf("first-try success fired %+v, want exactly one zero-value clear {0,0,0}", calls)
	}
}

// bindPendingCheckpoint (the RunArm binding) must connect the planner's retry to the run's
// session sink KEYED ON THE TRACE, through BOTH shapes: the function-shaped SessionGate.Checkpoint
// (the gateway native loop) and the concrete session.Table.SetPendingTurn (the harness). This
// proves the loop_session.go call site that closes the #1363 zero-caller gap: after this leaf,
// grep -rn SetPendingTurn internal/agent/ finds checkpointPending.
func TestPendingTurnCheckpoint_BoundSinkKeyedOnTrace(t *testing.T) {
	newRetryServer := func() *httptest.Server {
		var n int32
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if atomic.AddInt32(&n, 1) == 1 {
				w.WriteHeader(http.StatusTooManyRequests)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(completionOKBody))
		}))
	}

	t.Run("function-shaped gate", func(t *testing.T) {
		srv := newRetryServer()
		t.Cleanup(srv.Close)
		type call struct {
			trace               string
			attempt, lastStatus int
		}
		var calls []call
		gate := SessionGate{Checkpoint: func(trace string, attempt, lastStatus int, startedAtUnixNano int64) {
			calls = append(calls, call{trace, attempt, lastStatus})
		}}
		p := NewHTTPPlanner(srv.URL, "m", "")
		bound := bindPendingCheckpoint(p, resolveRunConfig([]RunOption{WithSessionGate(gate, "trace-x")}))
		if _, err := bound.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		// The WRITE is the first call, keyed on this run's trace; #4123's clear trails it.
		if len(calls) == 0 || calls[0] != (call{"trace-x", 2, http.StatusTooManyRequests}) {
			t.Fatalf("gate.Checkpoint calls = %+v, want the write (trace-x,2,429) first", calls)
		}
		if last := calls[len(calls)-1]; last != (call{"trace-x", 0, 0}) {
			t.Fatalf("gate.Checkpoint last call = %+v, want the (trace-x) zero-value clear", last)
		}
	})

	t.Run("concrete table", func(t *testing.T) {
		srv := newRetryServer()
		t.Cleanup(srv.Close)
		tbl := session.NewTable()
		tbl.Restore("trace-x", session.State{TraceID: "trace-x", Run: session.Running, Budget: session.Budget{TurnsLeft: 5, TokensLeft: 500}})
		if !tbl.Get("trace-x").PendingTurn.IsZero() {
			t.Fatalf("precondition: fresh table already carried a checkpoint; test would be vacuous")
		}
		// The success clears the checkpoint (#4123), so the post-Complete Get() is zero. Witness the
		// DURABLE write at the moment it happened via the every-revision stream: a kill -9 during the
		// backoff sleep would have persisted exactly this streamed {2,429,>0} revision.
		var wroteCheckpoint bool
		tbl.WatchRevisions(func(st session.State) {
			if pt := st.PendingTurn; pt.Attempt == 2 && pt.LastStatus == http.StatusTooManyRequests && pt.StartedAtUnixNano > 0 {
				wroteCheckpoint = true
			}
		})
		p := NewHTTPPlanner(srv.URL, "m", "")
		bound := bindPendingCheckpoint(p, resolveRunConfig([]RunOption{WithSessionTable(tbl, "trace-x")}))
		if _, err := bound.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		if !wroteCheckpoint {
			t.Fatal("no revision carried the write-ahead checkpoint {Attempt:2 LastStatus:429 StartedAtUnixNano:>0}")
		}
		if got := tbl.Get("trace-x").PendingTurn; !got.IsZero() {
			t.Fatalf("post-Complete table still carries %+v, want the zero checkpoint (#4123 clears on success)", got)
		}
	})
}
