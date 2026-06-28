package agent

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// Complete's 429/5xx backoff is otherwise INVISIBLE. RetryNotify must fire once per retry (on
// attempts 1..N-1, never the first try) with the triggering status and the upcoming wait, so the
// gateway can surface the silent backoff window. Here the upstream 429s twice then 200s, so the
// hook must fire exactly twice before the success.
func TestRetryNotify_FiresPerRetryWithStatus(t *testing.T) {
	var hits int32
	type call struct {
		attempt int
		status  int
		wait    time.Duration
	}
	var calls []call

	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&n, 1) <= 2 {
			w.WriteHeader(http.StatusTooManyRequests) // 429 twice
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(srv.Close)

	p := NewHTTPPlanner(srv.URL, "m", "")
	p.RetryNotify = func(attempt, status int, wait time.Duration) {
		atomic.AddInt32(&hits, 1)
		calls = append(calls, call{attempt, status, wait})
	}

	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Complete after 2x429: %v", err)
	}
	if comp.Message.Content != "ok" {
		t.Fatalf("content = %q, want ok", comp.Message.Content)
	}
	if got := atomic.LoadInt32(&hits); got != 2 {
		t.Fatalf("RetryNotify fired %d times, want 2 (one per retry before the success)", got)
	}
	// Each notify must carry the 429 that triggered it and the growing backoff wait.
	for i, c := range calls {
		if c.attempt != i+1 {
			t.Errorf("call %d attempt = %d, want %d", i, c.attempt, i+1)
		}
		if c.status != http.StatusTooManyRequests {
			t.Errorf("call %d status = %d, want 429", i, c.status)
		}
		if c.wait <= 0 {
			t.Errorf("call %d wait = %s, want a positive backoff", i, c.wait)
		}
	}
}

// A clean first-try success must NOT fire the hook — it is for retries only, so a healthy turn
// stays byte-for-byte silent.
func TestRetryNotify_SilentOnFirstTrySuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(srv.Close)

	var hits int32
	p := NewHTTPPlanner(srv.URL, "m", "")
	p.RetryNotify = func(int, int, time.Duration) { atomic.AddInt32(&hits, 1) }
	if _, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if atomic.LoadInt32(&hits) != 0 {
		t.Fatal("RetryNotify fired on a first-try success (must be retry-only)")
	}
}
