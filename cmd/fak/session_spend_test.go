package main

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// session_spend_test.go — the host spend meter (session_spend.go) and its wiring
// into the served-session debit hook: pricing resolution order (env override →
// built-in provider table → dollar-blind), the exact integer micro-cent turn
// cost model, and the end-to-end drain: a guard budget envelope's spend=$N
// ceiling refusing the session's next request once priced turns cross it.

// swapServedSpendPricing installs a deterministic pricing for one test and
// restores the previous process state on cleanup, so package-global pricing
// never leaks across tests.
func swapServedSpendPricing(t *testing.T, p gateway.CachePricing, ok bool) {
	t.Helper()
	servedSpend.mu.Lock()
	prevArmed, prevOK, prevP, prevSrc := servedSpend.armed, servedSpend.ok, servedSpend.p, servedSpend.source
	servedSpend.armed, servedSpend.ok, servedSpend.p, servedSpend.source = true, ok, p, "test"
	servedSpend.mu.Unlock()
	t.Cleanup(func() {
		servedSpend.mu.Lock()
		servedSpend.armed, servedSpend.ok, servedSpend.p, servedSpend.source = prevArmed, prevOK, prevP, prevSrc
		servedSpend.mu.Unlock()
	})
}

// TestSpendTurnMicroCentsExactAxes pins the integer micro-cent cost of each
// billable axis under Opus 4.8 pricing ({5,25} per MTok): uncached input 1.0x,
// cache read 0.1x, cache creation at the 5m write tier 1.25x, output at the
// output price — the same shape as gateway.CachePricing.CostUSD.
func TestSpendTurnMicroCentsExactAxes(t *testing.T) {
	p := gateway.CachePricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25}
	cases := []struct {
		name string
		u    gateway.SessionUsage
		want int64
	}{
		{"uncached input", gateway.SessionUsage{PromptTokens: 1000}, 500_000},
		{"cache read at 0.1x", gateway.SessionUsage{CacheReadInputTokens: 10_000}, 500_000},
		{"cache write at 1.25x", gateway.SessionUsage{CacheCreationInputTokens: 1000}, 625_000},
		{"output", gateway.SessionUsage{CompletionTokens: 1000}, 2_500_000},
		{"all axes", gateway.SessionUsage{
			PromptTokens: 1000, CacheReadInputTokens: 10_000,
			CacheCreationInputTokens: 1000, CompletionTokens: 1000,
		}, 4_125_000},
		{"zero usage", gateway.SessionUsage{}, 0},
	}
	for _, tc := range cases {
		if got := spendTurnMicroCents(p, tc.u); got != tc.want {
			t.Errorf("%s: spendTurnMicroCents = %d, want %d", tc.name, got, tc.want)
		}
	}
}

// TestResolveSpendPricingOrder pins the resolution ladder: explicit env wins,
// the built-in table prices the flagship anthropic/claude pair, and an unknown
// pair is dollar-blind (ok=false → no debit, never a guessed cost).
func TestResolveSpendPricingOrder(t *testing.T) {
	t.Run("env override wins", func(t *testing.T) {
		t.Setenv(spendInputPriceEnv, "7")
		t.Setenv(spendOutputPriceEnv, "11")
		p, source, ok := resolveSpendPricing("anthropic", "claude")
		if !ok || p.InputPerMTokUSD != 7 || p.OutputPerMTokUSD != 11 || source != spendEnvPricingSource {
			t.Fatalf("env pricing = (%+v, %q, %v), want 7/11 from env", p, source, ok)
		}
	})
	t.Run("builtin table prices guarded claude", func(t *testing.T) {
		p, _, ok := resolveSpendPricing("anthropic", "claude")
		if !ok || p.InputPerMTokUSD != gateway.ClaudeOpus48InputPerMTokUSD || p.OutputPerMTokUSD != gateway.ClaudeOpus48OutputPerMTokUSD {
			t.Fatalf("builtin pricing = (%+v, %v), want Opus 4.8 table", p, ok)
		}
	})
	t.Run("unknown pair is dollar-blind", func(t *testing.T) {
		if _, source, ok := resolveSpendPricing("openai", "codex"); ok || source != "none" {
			t.Fatalf("unknown pair = (%q, %v), want dollar-blind none/false", source, ok)
		}
	})
}

// TestDebitSessionSpendCeilingDrains is the end-to-end control: a guard budget
// envelope's spend ceiling, seeded through the SAME applyGuardSessionBudgetEnvelope
// path `fak guard --budget-envelope` uses, drains the session with the closed
// BUDGET_SPEND_EXHAUSTED reason once priced turns cross it — and the next
// request boundary refuses without minting a continuation.
func TestDebitSessionSpendCeilingDrains(t *testing.T) {
	swapServedSpendPricing(t, gateway.CachePricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25}, true)
	const trace = "serve-hook-spend-1"
	t.Cleanup(func() { serveSessions.Reset(trace) })

	env, err := session.ParseBudgetEnvelope("spend=$0.05") // 5 cents = 5,000,000 micro-cents
	if err != nil {
		t.Fatalf("ParseBudgetEnvelope: %v", err)
	}
	applyGuardSessionBudgetEnvelope(serveSessions, trace, env, true, nil, 0, 0, time.Now())

	// Turn 1: 1000 output tokens at $25/MTok = 2,500,000 micro-cents — under the cap.
	st := debitSession(context.Background(), trace, gateway.SessionUsage{CompletionTokens: 1000})
	if st.Run != "running" || st.Reason != "" {
		t.Fatalf("turn 1 state = {Run:%q Reason:%q}, want running under the cap", st.Run, st.Reason)
	}

	// Turn 2: another 1500 output tokens = 3,750,000 micro-cents — crosses $0.05.
	st = debitSession(context.Background(), trace, gateway.SessionUsage{CompletionTokens: 1500})
	if st.Run != "draining" || st.Reason != session.ReasonBudgetSpend {
		t.Fatalf("turn 2 state = {Run:%q Reason:%q}, want draining/%s", st.Run, st.Reason, session.ReasonBudgetSpend)
	}
	if st.ContinuationID != "" {
		t.Fatalf("spend drain minted continuation %q, want none — a spent cap is terminal", st.ContinuationID)
	}

	v := decideSession(context.Background(), trace)
	if v.Proceed {
		t.Fatalf("decide after spend drain = %+v, want refusal", v)
	}
}

// TestDebitSessionDollarBlindLeavesSpendUntouched: with no pricing resolvable,
// a configured spend budget is never debited a guessed cost — the meter is
// honest about what it cannot price.
func TestDebitSessionDollarBlindLeavesSpendUntouched(t *testing.T) {
	swapServedSpendPricing(t, gateway.CachePricing{}, false)
	const trace = "serve-hook-spend-blind-1"
	t.Cleanup(func() { serveSessions.Reset(trace) })

	env, err := session.ParseBudgetEnvelope("spend=$0.01")
	if err != nil {
		t.Fatalf("ParseBudgetEnvelope: %v", err)
	}
	applyGuardSessionBudgetEnvelope(serveSessions, trace, env, true, nil, 0, 0, time.Now())

	st := debitSession(context.Background(), trace, gateway.SessionUsage{CompletionTokens: 100_000})
	if st.Run != "running" || st.Reason != "" {
		t.Fatalf("dollar-blind debit = {Run:%q Reason:%q}, want running (no guessed debit)", st.Run, st.Reason)
	}
}
