package agent

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// retryableStatus is the gate that decides whether a transient upstream failure is
// retried at all. The two additions that matter for an Anthropic-fronting fleet are 408
// (upstream timed out receiving the request) and 529 (Anthropic "Overloaded"); the rest
// of the transient family must stay retryable and every request-error 4xx must stay
// non-retryable so a futile retry never burns the budget.
func TestRetryableStatus_Membership(t *testing.T) {
	retry := []int{
		http.StatusRequestTimeout,      // 408
		http.StatusTooManyRequests,     // 429
		http.StatusInternalServerError, // 500
		http.StatusBadGateway,          // 502
		http.StatusServiceUnavailable,  // 503
		http.StatusGatewayTimeout,      // 504
		statusOverloaded,               // 529
	}
	for _, c := range retry {
		if !retryableStatus(c) {
			t.Errorf("retryableStatus(%d) = false, want true (transient/overload)", c)
		}
	}
	noRetry := []int{200, 400, 401, 403, 404, 409, 413, 422, 418, 501}
	for _, c := range noRetry {
		if retryableStatus(c) {
			t.Errorf("retryableStatus(%d) = true, want false (request error / not transient)", c)
		}
	}
}

// parseRetryAfterSeconds honors only the delta-seconds form. The HTTP-date form and any
// junk value must fall back (ok=false) so a bad header can never wedge the wait.
func TestParseRetryAfterSeconds(t *testing.T) {
	cases := []struct {
		in     string
		wantD  time.Duration
		wantOK bool
	}{
		{"0", 0, true},
		{"30", 30 * time.Second, true},
		{" 12 ", 12 * time.Second, true}, // surrounding whitespace tolerated
		{"", 0, false},
		{"-5", 0, false},                            // negative is nonsense, ignore
		{"abc", 0, false},                           // non-numeric
		{"Wed, 21 Oct 2025 07:28:00 GMT", 0, false}, // HTTP-date form is deliberately unhandled
	}
	for _, c := range cases {
		d, ok := parseRetryAfterSeconds(c.in)
		if ok != c.wantOK || (ok && d != c.wantD) {
			t.Errorf("parseRetryAfterSeconds(%q) = (%s, %v), want (%s, %v)", c.in, d, ok, c.wantD, c.wantOK)
		}
	}
}

// retryWait prefers a server-directed Retry-After (capped), and otherwise falls back to
// the jittered exponential schedule. Both branches must stay strictly positive for a real
// retry so the otherwise-invisible backoff window is always reported as a wait.
func TestRetryWait_HonorsRetryAfterElseBackoff(t *testing.T) {
	// A 5s Retry-After is honored: never less than 5s (the server asked us not to come
	// back early) and only a small upward nudge beyond it.
	for i := 0; i < 100; i++ {
		w := retryWait(1, "5")
		if w < 5*time.Second || w > 5*time.Second+5*time.Second/4 {
			t.Fatalf("retryWait honoring Retry-After=5 = %s, want [5s, 6.25s]", w)
		}
	}
	// A Retry-After above the ceiling is clamped to maxHonoredRetryAfter (plus jitter).
	big := retryWait(1, "9999")
	if big < maxHonoredRetryAfter || big > maxHonoredRetryAfter+maxHonoredRetryAfter/4 {
		t.Fatalf("retryWait honoring huge Retry-After = %s, want ~%s (clamped)", big, maxHonoredRetryAfter)
	}
	// No Retry-After: fall back to the jittered exponential base for the attempt, in
	// [base/2, base].
	for i := 0; i < 100; i++ {
		base := backoffDuration(2)
		w := retryWait(2, "")
		if w < base/2 || w > base {
			t.Fatalf("retryWait backoff fallback = %s, want [%s, %s]", w, base/2, base)
		}
	}
}

// backoffDuration must grow quadratically but never exceed the per-wait cap, so a large
// attempt budget cannot produce an unboundedly long single sleep.
func TestBackoffDuration_CappedAndMonotone(t *testing.T) {
	if got := backoffDuration(0); got != 0 {
		t.Errorf("backoffDuration(0) = %s, want 0 (no wait on first try)", got)
	}
	if got := backoffDuration(1); got != 600*time.Millisecond {
		t.Errorf("backoffDuration(1) = %s, want 600ms", got)
	}
	for attempt := 1; attempt <= 64; attempt++ {
		if got := backoffDuration(attempt); got > maxBackoff {
			t.Fatalf("backoffDuration(%d) = %s exceeds cap %s", attempt, got, maxBackoff)
		}
	}
}

// plannerMaxAttempts defaults to 8 and accepts an in-range env override; an out-of-range
// or junk override is ignored (falls back to the default) so a typo can neither disable
// retries nor wedge a turn.
func TestPlannerMaxAttempts_DefaultAndClamp(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "")
	if got := plannerMaxAttempts(); got != 8 {
		t.Errorf("default plannerMaxAttempts = %d, want 8", got)
	}
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "3")
	if got := plannerMaxAttempts(); got != 3 {
		t.Errorf("override plannerMaxAttempts = %d, want 3", got)
	}
	for _, bad := range []string{"0", "-1", "17", "abc"} {
		t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", bad)
		if got := plannerMaxAttempts(); got != 8 {
			t.Errorf("override %q: plannerMaxAttempts = %d, want 8 (ignored)", bad, got)
		}
	}
}

// Complete must now RETRY a 529 (Anthropic "Overloaded") — the headline gap — and
// succeed once the upstream recovers, instead of failing the turn on the first 529.
func TestComplete_Retries529ThenSucceeds(t *testing.T) {
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&n, 1) == 1 {
			w.WriteHeader(statusOverloaded) // 529 once
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(srv.Close)

	p := NewHTTPPlanner(srv.URL, "m", "")
	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Complete should retry 529 and succeed, got: %v", err)
	}
	if comp.Message.Content != "ok" {
		t.Fatalf("content = %q, want ok", comp.Message.Content)
	}
	if got := atomic.LoadInt32(&n); got != 2 {
		t.Fatalf("upstream hit %d times, want 2 (one 529 + one success)", got)
	}
}

// A persistently retryable upstream must fail AFTER exactly maxAttempts tries (here
// pinned low via env so the test is fast), reporting the attempt count — proving the
// budget is honored and env-tunable rather than infinite.
func TestComplete_FailsAfterMaxAttempts(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "2")
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&n, 1)
		w.WriteHeader(http.StatusServiceUnavailable) // always 503
	}))
	t.Cleanup(srv.Close)

	p := NewHTTPPlanner(srv.URL, "m", "")
	_, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("Complete against a persistent 503 should fail, got nil")
	}
	if got := atomic.LoadInt32(&n); got != 2 {
		t.Fatalf("upstream hit %d times, want 2 (FAK_PLANNER_MAX_ATTEMPTS)", got)
	}
}

// CompleteStream must now RETRY a pre-stream retryable status (here 529) and then stream
// the recovered turn. The streaming wire (fak guard -- codex/--local) previously failed the
// turn on the first transient status with no retry at all.
func TestCompleteStream_Retries529ThenStreams(t *testing.T) {
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&n, 1) == 1 {
			w.WriteHeader(statusOverloaded) // 529 once, before any byte is streamed
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":null}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)

	p := NewHTTPPlanner(srv.URL, "m", "")
	var got strings.Builder
	comp, err := p.CompleteStream(context.Background(), func(d string) error { got.WriteString(d); return nil },
		[]Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("CompleteStream should retry 529 and stream, got: %v", err)
	}
	if got.String() != "hi" || comp.Message.Content != "hi" {
		t.Fatalf("streamed=%q content=%q, want hi/hi", got.String(), comp.Message.Content)
	}
	if hits := atomic.LoadInt32(&n); hits != 2 {
		t.Fatalf("upstream hit %d times, want 2 (one 529 + one streamed success)", hits)
	}
}

// A cancelled context must abort the backoff wait promptly rather than sleeping out the
// full (now longer) schedule — so a user who Ctrl-C's a stalled turn is not held hostage
// by the retry budget.
func TestComplete_ContextCancelAbortsBackoff(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable) // always 503 -> always retries
	}))
	t.Cleanup(srv.Close)

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	p := NewHTTPPlanner(srv.URL, "m", "")
	start := time.Now()
	if _, err := p.Complete(ctx, []Message{{Role: RoleUser, Content: "hi"}}, nil); err == nil {
		t.Fatal("Complete should fail when the context is cancelled mid-backoff")
	}
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Fatalf("Complete took %s after a 200ms ctx timeout — backoff did not honor cancellation", elapsed)
	}
}
