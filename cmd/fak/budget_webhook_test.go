package main

// budget_webhook_test.go — proves the #743 host wiring: budgetWebhookObserver POSTs each
// session.BudgetEvent to the configured operator webhook as JSON, and an empty URL yields
// the no-op (nil) seam. The session-package side (when the warn/exhaustion events fire) is
// covered by internal/session/observe_test.go; this file proves the host actually ships
// the event over HTTP.

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

func TestBudgetWebhookObserverPostsEvent(t *testing.T) {
	got := make(chan session.BudgetEvent, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("webhook got %s ct=%q, want POST application/json", r.Method, r.Header.Get("Content-Type"))
		}
		body, _ := io.ReadAll(r.Body)
		var ev session.BudgetEvent
		if err := json.Unmarshal(body, &ev); err != nil {
			t.Errorf("decode webhook body: %v (raw=%s)", err, body)
		}
		got <- ev
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	obs := budgetWebhookObserver(srv.URL)
	if obs == nil {
		t.Fatal("budgetWebhookObserver(url) returned nil, want a live observer")
	}
	obs(session.BudgetEvent{
		Kind:              session.BudgetWarn,
		TraceID:           "wh-1",
		ContextTokensLeft: 15,
		ContextTokensCap:  100,
		FractionConsumed:  0.85,
	})

	select {
	case ev := <-got:
		if ev.Kind != session.BudgetWarn || ev.TraceID != "wh-1" || ev.ContextTokensCap != 100 {
			t.Fatalf("received event = %+v, want the warn we posted", ev)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("webhook never received the budget event")
	}
}

func TestBudgetWebhookObserverEmptyURLIsNoOp(t *testing.T) {
	if obs := budgetWebhookObserver("   "); obs != nil {
		t.Fatal("empty/blank URL must return a nil observer (the no-op seam)")
	}
}
