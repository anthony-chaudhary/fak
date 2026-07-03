package agent

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// okBody is the minimal OpenAI-shaped success the fake upstream returns once a transient 403 clears.
const okBody = `{"model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`

// A TRANSIENT 403 (a server-side abuse/capacity gate that clears in seconds — the 2026-07-03 gem8
// storm's recoverable population) must self-heal in place via the bounded recovery arm, not surface
// terminally and drop the live session into a spurious /login. Here the upstream 403s twice then
// 200s; Complete must return the success and fire ForbiddenRetryNotify exactly once, "recovered",
// on the CONFIRMED 200 (never optimistically on the retry decision).
func TestForbiddenRetry_TransientRecovers(t *testing.T) {
	t.Setenv("FAK_FORBIDDEN_RETRY_WINDOW", "5s") // keep the window short but real
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&n, 1) <= 2 {
			w.WriteHeader(http.StatusForbidden) // 403 twice, no permanent-signature body
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(okBody))
	}))
	t.Cleanup(srv.Close)

	var outcomes []string
	p := NewHTTPPlanner(srv.URL, "m", "")
	p.ForbiddenRetryNotify = func(outcome string, _ int) { outcomes = append(outcomes, outcome) }

	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("a transient 403 should self-heal, got: %v", err)
	}
	if comp.Message.Content != "ok" {
		t.Fatalf("content = %q, want ok", comp.Message.Content)
	}
	if len(outcomes) != 1 || outcomes[0] != ForbiddenRetryRecovered {
		t.Fatalf("want exactly one %q notification, got %v", ForbiddenRetryRecovered, outcomes)
	}
}

// A PERMANENT 403 whose body carries a hard entitlement signature must NOT be retried at all — it
// surfaces immediately with the actionable answer, and the exhausted notify does NOT fire (no
// recovery was ever attempted). This is the fast-path for a real "not entitled to this model".
func TestForbiddenRetry_PermanentSignatureFailsFast(t *testing.T) {
	t.Setenv("FAK_FORBIDDEN_RETRY_WINDOW", "5s")
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&n, 1)
		w.WriteHeader(http.StatusForbidden)
		// A body the arm recognizes as a permanent denial.
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"permission_error","message":"you do not have access to model X"}}`))
	}))
	t.Cleanup(srv.Close)

	var outcomes []string
	p := NewHTTPPlanner(srv.URL, "m", "")
	p.ForbiddenRetryNotify = func(outcome string, _ int) { outcomes = append(outcomes, outcome) }

	_, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("a permanent 403 must surface, not self-heal")
	}
	if got := atomic.LoadInt32(&n); got != 1 {
		t.Fatalf("a permanent-signature 403 must not be retried; upstream hit %d times, want 1", got)
	}
	if len(outcomes) != 0 {
		t.Fatalf("a fail-fast permanent 403 must fire no recovery notify, got %v", outcomes)
	}
}

// A 403 that PERSISTS past the bounded window (a transient-looking body that never clears) must
// surface terminally after a handful of retries and fire ForbiddenRetryNotify "exhausted" once, so
// a real permission denial is not masked and the spent self-heal is visible, not silent.
func TestForbiddenRetry_PersistentExhausts(t *testing.T) {
	t.Setenv("FAK_FORBIDDEN_RETRY_WINDOW", "1s") // small window so the test exhausts quickly
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&n, 1)
		w.WriteHeader(http.StatusForbidden) // always 403, no permanent signature
	}))
	t.Cleanup(srv.Close)

	var outcomes []string
	p := NewHTTPPlanner(srv.URL, "m", "")
	p.ForbiddenRetryNotify = func(outcome string, _ int) { outcomes = append(outcomes, outcome) }

	_, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("a persistent 403 must surface terminally after the window")
	}
	if len(outcomes) != 1 || outcomes[0] != ForbiddenRetryExhausted {
		t.Fatalf("want exactly one %q notification, got %v", ForbiddenRetryExhausted, outcomes)
	}
	// The arm must have actually retried (more than the single first hit) before giving up.
	if got := atomic.LoadInt32(&n); got < 2 {
		t.Fatalf("a persistent 403 should be retried before exhausting; upstream hit %d times", got)
	}
}

// A disabled arm (FAK_FORBIDDEN_RETRY_WINDOW=0) restores the historical terminal-on-first-403
// behavior: one upstream hit, no retry, no notify.
func TestForbiddenRetry_DisabledIsTerminalOnFirst(t *testing.T) {
	t.Setenv("FAK_FORBIDDEN_RETRY_WINDOW", "0")
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&n, 1)
		w.WriteHeader(http.StatusForbidden)
	}))
	t.Cleanup(srv.Close)

	var outcomes []string
	p := NewHTTPPlanner(srv.URL, "m", "")
	p.ForbiddenRetryNotify = func(outcome string, _ int) { outcomes = append(outcomes, outcome) }

	if _, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil); err == nil {
		t.Fatal("a 403 with the arm disabled must surface terminally")
	}
	if got := atomic.LoadInt32(&n); got != 1 {
		t.Fatalf("disabled arm must not retry; upstream hit %d times, want 1", got)
	}
	if len(outcomes) != 0 {
		t.Fatalf("disabled arm must fire no notify, got %v", outcomes)
	}
}

// forbiddenBodyIsPermanent must recognize the hard-denial signatures and treat an unlabeled or
// transient-looking body as POSSIBLY-transient (retryable). The storm's whole lesson: an unlabeled
// 403 may well clear, so only a conservative, explicit signature suppresses the retry.
func TestForbiddenBodyIsPermanent_Classifies(t *testing.T) {
	permanent := []string{
		`{"error":{"type":"permission_error"}}`,
		`your organization does not have access to this model`,
		`the account is not entitled to claude-opus`,
		`this region is not supported for the requested model`,
		`unsupported_region`,
	}
	for _, b := range permanent {
		if !forbiddenBodyIsPermanent([]byte(b)) {
			t.Errorf("should be permanent: %q", b)
		}
	}
	transient := []string{
		``,                                  // blank body — the gem8 storm case
		`{"error":{"type":"forbidden"}}`,    // bare forbidden, no permission language
		`temporarily unavailable`,           // sounds transient
		`request blocked, please try again`, // an abuse gate that clears
	}
	for _, b := range transient {
		if forbiddenBodyIsPermanent([]byte(b)) {
			t.Errorf("should be treated as possibly-transient: %q", b)
		}
	}
}

// Sanity: the bounded window resolver clamps a fat-fingered override so a permanent 403 can never
// stall for minutes before the real answer.
func TestForbiddenRetryWindow_ClampsAndDefaults(t *testing.T) {
	t.Setenv("FAK_FORBIDDEN_RETRY_WINDOW", "999h")
	if got := forbiddenRetryWindow(); got != maxForbiddenRetryWindow {
		t.Fatalf("an over-long window should clamp to %s, got %s", maxForbiddenRetryWindow, got)
	}
	t.Setenv("FAK_FORBIDDEN_RETRY_WINDOW", "")
	if got := forbiddenRetryWindow(); got != defaultForbiddenRetryWindow {
		t.Fatalf("an unset window should default to %s, got %s", defaultForbiddenRetryWindow, got)
	}
}
