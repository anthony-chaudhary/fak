package main

// notify_test.go - proves the #761 tiered push notifier: the native sink writes one line
// per boundary, the webhook sink POSTs the StopEvent JSON, the effort order fires both,
// and the (TraceID, Rev) dedupe fires each boundary exactly once. The session-package side
// (WHEN a transition/budget event fires) is covered by internal/session/observe_test.go;
// this file proves the host's fan-out, sinks, and idempotency.

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

func TestNotifierWebhookSinkPostsOncePerRev(t *testing.T) {
	got := make(chan StopEvent, 4)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("webhook got %s, want POST", r.Method)
		}
		body, _ := io.ReadAll(r.Body)
		var ev StopEvent
		if err := json.Unmarshal(body, &ev); err != nil {
			t.Errorf("decode StopEvent: %v (raw=%s)", err, body)
		}
		got <- ev
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	n := newNotifier(false, io.Discard, srv.URL, "")
	if n == nil {
		t.Fatal("newNotifier with a webhook URL returned nil")
	}
	ev := StopEvent{TraceID: "t", Reason: session.ReasonPaused, To: "paused", Rev: 2}
	n.fire(ev)

	select {
	case rec := <-got:
		if rec.TraceID != "t" || rec.Reason != session.ReasonPaused || rec.Rev != 2 {
			t.Fatalf("webhook received %+v, want the paused event at rev 2", rec)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("webhook never received the StopEvent")
	}

	// Re-fire the SAME rev: idempotent, no second POST.
	n.fire(ev)
	select {
	case rec := <-got:
		t.Fatalf("same-rev re-fire produced a second POST: %+v", rec)
	case <-time.After(300 * time.Millisecond):
	}

	// A new (higher) rev DOES fire.
	n.fire(StopEvent{TraceID: "t", Reason: session.ReasonStopped, To: "stopped", Rev: 3})
	select {
	case rec := <-got:
		if rec.Rev != 3 || rec.Reason != session.ReasonStopped {
			t.Fatalf("rev-3 POST = %+v, want the stopped event at rev 3", rec)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("webhook never received the rev-3 event")
	}
}

func TestNotifierNativeSinkWritesLine(t *testing.T) {
	var buf bytes.Buffer
	n := newNotifier(true, &buf, "", "")
	if n == nil {
		t.Fatal("newNotifier with native on returned nil")
	}
	n.fire(StopEvent{TraceID: "sess-9", Reason: session.ReasonDrained, To: "draining", Rev: 5})
	out := buf.String()
	if !strings.Contains(out, "sess-9") || !strings.Contains(out, "draining") || !strings.Contains(out, session.ReasonDrained) {
		t.Fatalf("native line = %q, want trace + to-token + reason", out)
	}
	if !strings.Contains(out, "rev=5") {
		t.Fatalf("native line = %q, want rev=5", out)
	}
	// Same-rev re-fire: still one line (dedupe applies to the native sink too).
	n.fire(StopEvent{TraceID: "sess-9", Reason: session.ReasonDrained, To: "draining", Rev: 5})
	if n := strings.Count(buf.String(), "sess-9"); n != 1 {
		t.Fatalf("native sink wrote %d lines for one rev, want 1", n)
	}
}

func TestNotifierEffortOrderNativeAndWebhook(t *testing.T) {
	posts := make(chan struct{}, 4)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.ReadAll(r.Body)
		posts <- struct{}{}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	var buf bytes.Buffer
	n := newNotifier(true, &buf, srv.URL, "")
	n.fire(StopEvent{TraceID: "t2", Reason: session.ReasonPaused, To: "paused", Rev: 1})

	// Native fired (inline).
	if !strings.Contains(buf.String(), "t2") {
		t.Fatalf("native sink did not fire: %q", buf.String())
	}
	// Webhook fired (async).
	select {
	case <-posts:
	case <-time.After(3 * time.Second):
		t.Fatal("webhook sink did not fire")
	}

	// Same-rev re-fire: neither sink re-invoked.
	n.fire(StopEvent{TraceID: "t2", Reason: session.ReasonPaused, To: "paused", Rev: 1})
	if c := strings.Count(buf.String(), "t2"); c != 1 {
		t.Fatalf("native re-fired on a deduped rev: %d lines", c)
	}
	select {
	case <-posts:
		t.Fatal("webhook re-fired on a deduped rev")
	case <-time.After(300 * time.Millisecond):
	}
}

func TestNotifierNilWhenNoSinks(t *testing.T) {
	if n := newNotifier(false, nil, "", ""); n != nil {
		t.Fatal("newNotifier with no sinks must return nil (the no-op seam)")
	}
}

func TestNotifierObserverAdaptersFire(t *testing.T) {
	var buf bytes.Buffer
	n := newNotifier(true, &buf, "", "")

	// The transition adapter renders the run-state token + reason.
	n.transitionObserver()(session.TransitionEvent{TraceID: "ta", To: session.Stopped, Reason: session.ReasonStopped, Rev: 1})
	if !strings.Contains(buf.String(), "stopped") || !strings.Contains(buf.String(), session.ReasonStopped) {
		t.Fatalf("transition adapter line = %q, want stopped + reason", buf.String())
	}
	// The budget adapter renders the "budget" origin token + the budget reason.
	n.budgetObserver()(session.BudgetEvent{TraceID: "ba", Reason: session.ReasonBudgetContext, Rev: 1})
	if !strings.Contains(buf.String(), "ba") || !strings.Contains(buf.String(), "budget") || !strings.Contains(buf.String(), session.ReasonBudgetContext) {
		t.Fatalf("budget adapter line = %q, want trace + budget origin + reason", buf.String())
	}
}

func TestCombineBudgetObservers(t *testing.T) {
	if combineBudgetObservers(nil, nil) != nil {
		t.Fatal("all-nil must combine to nil")
	}
	hits := 0
	one := session.BudgetObserver(func(session.BudgetEvent) { hits++ })
	combineBudgetObservers(nil, one)(session.BudgetEvent{})
	if hits != 1 {
		t.Fatalf("single non-nil combine fired %d times, want 1", hits)
	}
	hits = 0
	combineBudgetObservers(one, one)(session.BudgetEvent{})
	if hits != 2 {
		t.Fatalf("two-observer combine fired %d times, want 2", hits)
	}
}
