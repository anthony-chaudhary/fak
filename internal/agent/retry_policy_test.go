package agent

import (
	"context"
	"errors"
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

// parseRetryAfter honors BOTH RFC 7231 forms: delta-seconds (as before) AND the HTTP-date
// form, resolved against an injected `now` so the test is deterministic. A date in the
// past (nothing left to wait for) or junk falls back (ok=false). This is the #1360 gap:
// the delta-only parser silently dropped a date-form window to local backoff.
func TestParseRetryAfter_BothForms(t *testing.T) {
	now := time.Date(2025, 10, 21, 7, 0, 0, 0, time.UTC)
	// delta-seconds still works
	if d, ok := parseRetryAfter("30", now); !ok || d != 30*time.Second {
		t.Errorf("parseRetryAfter delta = (%s,%v), want (30s,true)", d, ok)
	}
	// HTTP-date in the future -> remaining duration until that instant
	future := now.Add(28 * time.Minute).Format(http.TimeFormat)
	if d, ok := parseRetryAfter(future, now); !ok || d != 28*time.Minute {
		t.Errorf("parseRetryAfter future date = (%s,%v), want (28m,true)", d, ok)
	}
	// HTTP-date already in the past -> nothing to wait for
	past := now.Add(-time.Minute).Format(http.TimeFormat)
	if _, ok := parseRetryAfter(past, now); ok {
		t.Errorf("parseRetryAfter past date ok=true, want false")
	}
	// junk -> fall back
	if _, ok := parseRetryAfter("not-a-date", now); ok {
		t.Errorf("parseRetryAfter junk ok=true, want false")
	}
}

// plannerRetryBudget defaults to 4h, accepts a Go-duration override, clamps a huge value to
// maxRetryBudget, and treats 0 as "time bound disabled". A junk value falls back to the
// default so a typo cannot silently disable the retry window.
func TestPlannerRetryBudget_DefaultOverrideClamp(t *testing.T) {
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "")
	if got := plannerRetryBudget(); got != defaultRetryBudget {
		t.Errorf("default budget = %s, want %s", got, defaultRetryBudget)
	}
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "30m")
	if got := plannerRetryBudget(); got != 30*time.Minute {
		t.Errorf("override budget = %s, want 30m", got)
	}
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "0")
	if got := plannerRetryBudget(); got != 0 {
		t.Errorf("budget=0 = %s, want 0 (time bound off)", got)
	}
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "999h")
	if got := plannerRetryBudget(); got != maxRetryBudget {
		t.Errorf("huge budget = %s, want clamp %s", got, maxRetryBudget)
	}
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "junk")
	if got := plannerRetryBudget(); got != defaultRetryBudget {
		t.Errorf("junk budget = %s, want default %s", got, defaultRetryBudget)
	}
}

// retryBounds: an explicitly-PINNED attempt count stays authoritative and exact (the
// historical fast-give-up contract); when NOT pinned the time budget is the primary
// limiter and the attempt cap rises to the spin guard so the full window is reachable; a
// zero budget restores pure attempt-count behavior with the time bound off.
func TestRetryBounds_PinnedVsBudget(t *testing.T) {
	now := time.Unix(0, 0)
	// pinned attempt count -> exact, budget still on for the wait clamp
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "2")
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "4h")
	if max, _, on := retryBounds(now); max != 2 || !on {
		t.Errorf("pinned: (max=%d,on=%v), want (2,true)", max, on)
	}
	// not pinned + budget on -> attempt cap rises to the spin guard, deadline = now+budget
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "")
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "1h")
	if max, dl, on := retryBounds(now); max != retryAttemptHardCap || !on || !dl.Equal(now.Add(time.Hour)) {
		t.Errorf("budget: (max=%d,deadline=%s,on=%v), want (%d,%s,true)", max, dl, on, retryAttemptHardCap, now.Add(time.Hour))
	}
	// budget disabled -> attempt-count bound, time bound off
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "0")
	if max, _, on := retryBounds(now); max != 8 || on {
		t.Errorf("budget off: (max=%d,on=%v), want (8,false)", max, on)
	}
}

// retryWaitWithin never returns a wait that would sleep PAST the deadline, and signals
// "budget spent" with a negative value so the loop stops instead of overshooting 4h.
func TestRetryWaitWithin_ClampsToDeadline(t *testing.T) {
	now := time.Unix(1000, 0)
	// A 9999s Retry-After but only 5s of budget left -> wait at most the remaining 5s.
	if w := retryWaitWithin(1, "9999", now.Add(5*time.Second), now); w > 5*time.Second || w <= 0 {
		t.Errorf("clamp-to-remaining = %s, want (0,5s]", w)
	}
	// No budget left -> negative (caller treats as exhaustion).
	if w := retryWaitWithin(1, "30", now.Add(-time.Second), now); w >= 0 {
		t.Errorf("spent budget = %s, want negative", w)
	}
	// Zero deadline -> time bound off -> identical to retryWait (honors the 5s Retry-After).
	if w := retryWaitWithin(1, "5", time.Time{}, now); w < 5*time.Second {
		t.Errorf("no-deadline = %s, want >=5s (honors Retry-After)", w)
	}
}

// #1358 — the headline truth bug. A retryable 429 (carrying a real Retry-After) FOLLOWED by
// a transient transport error on a later attempt must NOT let the transport glitch shadow
// the real status: on exhaustion the returned error must still unwrap to the
// *UpstreamStatusError{429, Retry-After}, so the gateway surfaces a real 429 + Retry-After
// rather than an opaque 502 with the Retry-After dropped. Before the fix this returned the
// transport error and the 429 + Retry-After were lost.
func TestComplete_InterleavedTransportErrorKeepsRealStatus(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "3") // bounded + fast
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch atomic.AddInt32(&n, 1) {
		case 1:
			// A real 429 with a Retry-After — the truthful signal we must preserve.
			w.Header().Set("Retry-After", "7")
			w.WriteHeader(http.StatusTooManyRequests)
		default:
			// A transient transport error: hijack and slam the connection so the client
			// sees a (non-deterministic) read/connection error, not an HTTP status.
			hj, ok := w.(http.Hijacker)
			if !ok {
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			}
			conn, _, _ := hj.Hijack()
			_ = conn.Close()
		}
	}))
	t.Cleanup(srv.Close)

	p := NewHTTPPlanner(srv.URL, "m", "")
	_, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("Complete should fail after exhausting retries")
	}
	var se *UpstreamStatusError
	if !errors.As(err, &se) {
		t.Fatalf("err = %v, want it to unwrap to *UpstreamStatusError (the real 429), not the transport glitch", err)
	}
	if se.Status != http.StatusTooManyRequests {
		t.Fatalf("preserved status = %d, want 429", se.Status)
	}
	if se.RetryAfter != "7" {
		t.Fatalf("preserved Retry-After = %q, want \"7\" (must survive the later transport error)", se.RetryAfter)
	}
}

// #1360 — the time budget makes retries continue PAST the default attempt count. With the
// attempt count unpinned and a tiny budget + near-zero waits, the loop keeps retrying a
// persistent 503 until the budget expires, exceeding the old fixed 8-attempt ceiling, then
// exhausts cleanly (still surfacing the real 503). A zero-Retry-After keeps each wait
// near-instant so the test is fast.
func TestComplete_TimeBudgetRetriesPastAttemptCap(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "") // unpinned -> time budget is the limiter
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "300ms")
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&n, 1)
		w.Header().Set("Retry-After", "0") // zero wait -> the loop spins fast within the budget
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(srv.Close)

	p := NewHTTPPlanner(srv.URL, "m", "")
	start := time.Now()
	_, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("Complete against a persistent 503 should fail after the budget")
	}
	var se *UpstreamStatusError
	if !errors.As(err, &se) || se.Status != http.StatusServiceUnavailable {
		t.Fatalf("err = %v, want wrapped *UpstreamStatusError{503}", err)
	}
	if got := atomic.LoadInt32(&n); got <= 8 {
		t.Fatalf("upstream hit %d times, want > 8 (the time budget must retry past the old attempt cap)", got)
	}
	// The total wall time is bounded by the budget (plus a little slack for the final try).
	if elapsed > 3*time.Second {
		t.Fatalf("took %s, want ~budget (300ms) + slack — must not overshoot the deadline", elapsed)
	}
}
