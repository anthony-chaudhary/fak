package main

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/session"
)

func TestUnkeyedServeSessionsEachGetTheirOwnBudget(t *testing.T) {
	prev := serveUnkeyedSessionBudget.Load()
	t.Cleanup(func() { serveUnkeyedSessionBudget.Store(prev) })
	setServeUnkeyedSessionBudget(&session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: 1000})

	ctx := context.Background()
	a, b := gateway.UnkeyedTracePrefix+"t930a", gateway.UnkeyedTracePrefix+"t930b"
	if v := decideSession(ctx, a); !v.Proceed {
		t.Fatalf("first unkeyed turn refused: %+v", v.State)
	}
	debitSession(ctx, a, gateway.SessionUsage{ContextTokens: 1500})
	if v := decideSession(ctx, a); v.Proceed {
		t.Fatal("unkeyed session over its context budget still admitted; budget admission must stay fail-closed")
	} else if v.State.Reason != session.ReasonBudgetContext {
		t.Fatalf("refusal reason = %q, want %s", v.State.Reason, session.ReasonBudgetContext)
	}
	if v := decideSession(ctx, b); !v.Proceed {
		t.Fatalf("a second unkeyed connection was refused (%q): one caller's budget poisoned another", v.State.Reason)
	}
	if v := decideSession(ctx, "keyed-t930"); !v.Proceed || serveSessions.Get("keyed-t930").Budget.ContextTokensLeft != 0 {
		t.Fatal("a keyed trace must not inherit the unkeyed per-connection budget")
	}
}
