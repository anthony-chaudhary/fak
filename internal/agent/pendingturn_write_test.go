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
// so PendingTurnCheckpoint fires exactly once, carrying the 1-based attempt now in progress
// (2), the 429 that triggered it, and a real turn-start timestamp. This is the primitive's
// first live writer: before this leaf, grep -rn PendingTurn internal/agent/ was empty.
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
	if len(calls) != 1 {
		t.Fatalf("PendingTurnCheckpoint fired %d times, want 1 (once, on the single retry)", len(calls))
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
}

// A clean first-try success writes NO checkpoint on this leaf — the write is retry-only, so
// a healthy turn stays byte-for-byte the pre-#1363 path. (Clearing the checkpoint on success
// is the symmetric other half, C-pending-turn-2 / #4123.)
func TestPendingTurnCheckpoint_SilentOnFirstTrySuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(completionOKBody))
	}))
	t.Cleanup(srv.Close)

	var hits int32
	p := NewHTTPPlanner(srv.URL, "m", "")
	p.PendingTurnCheckpoint = func(int, int, int64) { atomic.AddInt32(&hits, 1) }
	if _, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if atomic.LoadInt32(&hits) != 0 {
		t.Fatal("PendingTurnCheckpoint fired on a first-try success (the write is retry-only)")
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
		var gotTrace string
		var gotAttempt, gotStatus int
		gate := SessionGate{Checkpoint: func(trace string, attempt, lastStatus int, startedAtUnixNano int64) {
			gotTrace, gotAttempt, gotStatus = trace, attempt, lastStatus
		}}
		p := NewHTTPPlanner(srv.URL, "m", "")
		bound := bindPendingCheckpoint(p, resolveRunConfig([]RunOption{WithSessionGate(gate, "trace-x")}))
		if _, err := bound.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		if gotTrace != "trace-x" || gotAttempt != 2 || gotStatus != http.StatusTooManyRequests {
			t.Fatalf("gate.Checkpoint got (%q,%d,%d), want (trace-x,2,429)", gotTrace, gotAttempt, gotStatus)
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
		p := NewHTTPPlanner(srv.URL, "m", "")
		bound := bindPendingCheckpoint(p, resolveRunConfig([]RunOption{WithSessionTable(tbl, "trace-x")}))
		if _, err := bound.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
			t.Fatalf("Complete: %v", err)
		}
		got := tbl.Get("trace-x").PendingTurn
		if got.Attempt != 2 || got.LastStatus != http.StatusTooManyRequests || got.StartedAtUnixNano <= 0 {
			t.Fatalf("Table.SetPendingTurn wrote %+v, want {Attempt:2 LastStatus:429 StartedAtUnixNano:>0}", got)
		}
	})
}
