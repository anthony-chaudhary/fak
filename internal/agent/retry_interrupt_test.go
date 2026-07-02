package agent

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
)

// The #2257 regression: the upstream answers a cap-classified 429 whose relayed reset is
// ~1h out, the retry loop announces and begins the honored wait, and the CLIENT cancels
// mid-sleep (the wrapped Claude Code timing out its own request at ~300s). The error the
// turn surfaces must carry the rate-limit truth — the classified *UpstreamStatusError with
// its cap kind — not collapse into the bare context error, so the FAILED line, the metric
// kind, and any supervisor reading the failure see a known rate-limit park candidate.
func TestComplete_ClientCancelDuringCapWait_CarriesClassification(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Anthropic-Ratelimit-Unified-Reset", strconv.FormatInt(time.Now().Add(70*time.Minute).Unix(), 10))
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write(anthropicLimitBody("usage limit reached"))
	}))
	t.Cleanup(srv.Close)

	p := NewHTTPPlanner(srv.URL, "m", "")
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	// Cancel the instant the retry wait is announced — the deterministic stand-in for a
	// client that hangs up partway into the honored sleep.
	var announced time.Duration
	p.RetryNotify = func(_, _ int, wait time.Duration) {
		announced = wait
		cancel()
	}
	_, err := p.Complete(ctx, []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("Complete succeeded against a capped upstream with a cancelled client")
	}

	var ri *RetryInterruptedError
	if !errors.As(err, &ri) {
		t.Fatalf("error %T does not unwrap to *RetryInterruptedError: %v", err, err)
	}
	if ri.AnnouncedWait <= 0 || ri.AnnouncedWait != announced {
		t.Fatalf("AnnouncedWait = %s, want the announced retry wait %s", ri.AnnouncedWait, announced)
	}
	if !ri.ClientGone() {
		t.Fatal("ClientGone() = false for a context.Canceled interruption")
	}
	var se *UpstreamStatusError
	if !errors.As(err, &se) {
		t.Fatalf("error %T does not unwrap to the classified *UpstreamStatusError: %v", err, err)
	}
	if se.Status != http.StatusTooManyRequests {
		t.Fatalf("Status = %d, want 429", se.Status)
	}
	if se.LimitReason != resume.LimitUsage {
		t.Fatalf("LimitReason = %q, want %q (the closed vocabulary token must survive the cancel)", se.LimitReason, resume.LimitUsage)
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatal("errors.Is(err, context.Canceled) = false — the interruption cause must stay reachable")
	}
}

// The no-behavior-change fence: a cancellation during a retry wait with NO classified
// status pending (a pure transport-glitch loop) surfaces the bare context error exactly
// as before — a genuinely-unclassified failure must still read "error" downstream.
func TestComplete_ClientCancelDuringGlitchBackoff_StaysBareContextError(t *testing.T) {
	// A hijack-and-slam server: every attempt dies on a transient transport error
	// (mid-flight reset/EOF) — never an HTTP status, so lastStatusErr stays nil.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hj, ok := w.(http.Hijacker)
		if !ok {
			t.Fatal("test writer cannot hijack")
		}
		conn, _, err := hj.Hijack()
		if err != nil {
			t.Fatalf("hijack: %v", err)
		}
		conn.Close() // slam the connection: a transient transport error, no HTTP status
	}))
	t.Cleanup(srv.Close)

	p := NewHTTPPlanner(srv.URL, "m", "")
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	p.RetryNotify = func(_, _ int, _ time.Duration) { cancel() }
	_, err := p.Complete(ctx, []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("Complete succeeded against a connection-slamming upstream")
	}
	var ri *RetryInterruptedError
	if errors.As(err, &ri) {
		t.Fatalf("unclassified glitch cancel wrapped as RetryInterruptedError: %v", err)
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want the bare context.Canceled path unchanged", err)
	}
}
