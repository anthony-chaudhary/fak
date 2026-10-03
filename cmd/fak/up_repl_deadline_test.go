package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/macfit"
)

// replStubCompletion is the smallest OpenAI-shaped buffered completion the REPL
// decodes: one assistant choice plus usage telemetry.
const replStubCompletion = `{"id":"cmpl-1","object":"chat.completion","created":1,"model":"qwen3.8-27b-q4_k_m","choices":[{"index":0,"message":{"role":"assistant","content":"slow answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`

func replTestPlan(t *testing.T) macfit.TurnkeyProfile {
	t.Helper()
	plan, err := macfit.ConfigureTurnkey(36 * macfit.GiB)
	if err != nil {
		t.Fatalf("ConfigureTurnkey: %v", err)
	}
	return plan
}

// #12926: the interactive REPL requests buffered native completions, so a valid
// generation that outlasts the old fixed 30s client deadline must still reach the
// user rather than being cancelled and reported as a connection error.
//
// This test is self-contained: it names only the production entry point
// (runTurnkeyREPL) so it compiles against the parent source, where it is RED —
// the 30s client timeout cancels the 31s answer and the completion is never
// delivered.
//
// fak-test:runtime slow est=35s
func TestTurnkeyREPLDeliversCompletionPastOldClientDeadline(t *testing.T) {
	const answerDelay = 31 * time.Second
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done(): // the client cancelled or disconnected
			return
		case <-time.After(answerDelay):
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, replStubCompletion)
	}))
	defer srv.Close()

	var out bytes.Buffer
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := runTurnkeyREPL(ctx, strings.NewReader("hello\n/quit\n"), &out, srv.URL, replTestPlan(t)); err != nil {
		t.Fatalf("runTurnkeyREPL: %v", err)
	}
	got := out.String()
	if !strings.Contains(got, "slow answer") {
		t.Fatalf("a completion slower than the old 30s client deadline was not delivered:\n%s", got)
	}
	if strings.Contains(got, "connection error") {
		t.Fatalf("a valid slow generation was reported as a connection error:\n%s", got)
	}
}

// #12926: removing the fixed deadline must not remove cancellation. A parent
// context cancel still terminates an in-flight request promptly and is reported,
// so Ctrl-C / SIGTERM keeps working through the shared process context.
//
// fak-test:runtime fast est=3s
func TestTurnkeyREPLParentCancelTerminatesInFlight(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done(): // the client cancelled or disconnected
		case <-release:
		case <-time.After(5 * time.Second): // never let a stuck handler wedge cleanup
		}
		_, _ = io.WriteString(w, replStubCompletion)
	}))
	// Registered after srv.Close below so it runs FIRST (LIFO): the handler is
	// released before the server waits for in-flight requests to drain.
	defer srv.Close()
	defer close(release)

	var out bytes.Buffer
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	start := time.Now()
	go func() {
		done <- runTurnkeyREPL(ctx, strings.NewReader("hello\n/quit\n"), &out, srv.URL, replTestPlan(t))
	}()

	// Let the request reach the handler, then cancel the parent context.
	time.Sleep(150 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		if elapsed := time.Since(start); elapsed > 2*time.Second {
			t.Fatalf("REPL did not terminate promptly after parent cancel: %v", elapsed)
		}
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("runTurnkeyREPL error = %v, want context.Canceled", err)
		}
		if strings.Contains(out.String(), "slow answer") {
			t.Fatalf("REPL rendered a completion after the parent context was cancelled:\n%s", out.String())
		}
	case <-time.After(4 * time.Second):
		t.Fatal("REPL did not terminate after parent context cancellation")
	}
}
