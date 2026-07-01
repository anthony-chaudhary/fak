package gateway

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ratelimit"
)

// TestTokenRateGateReserveSettleAndWindowRoll is the #2019 accounting witness for the
// reserve-on-estimate / settle-on-truth cycle: an in-flight estimate holds window budget,
// the provider's real (smaller) usage returns the over-estimated headroom the moment it
// settles, and the window's settled usage expires when the minute rolls.
func TestTokenRateGateReserveSettleAndWindowRoll(t *testing.T) {
	g := NewTokenRateGate(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxInputTokens: 1000}})
	clock := time.Unix(1_700_000_000, 0)
	g.now = func() time.Time { return clock }
	g.windowStart = clock

	r1, err := g.Admit(ratelimit.NewTokenUsage(600, 0, 0))
	if err != nil || r1 == nil {
		t.Fatalf("first Admit = (%v, %v), want an admitted reservation", r1, err)
	}

	// The 600-token estimate is held in flight: a 500-token request no longer fits.
	_, err = g.Admit(ratelimit.NewTokenUsage(500, 0, 0))
	var ae *AdmissionError
	if !errors.As(err, &ae) || ae.Verdict != VerdictShed {
		t.Fatalf("over-budget Admit err = %v, want a VerdictShed *AdmissionError", err)
	}
	if !strings.Contains(ae.Reason, ratelimit.TokenCapInputTokens) {
		t.Fatalf("shed reason = %q, want the firing cap %q named", ae.Reason, ratelimit.TokenCapInputTokens)
	}

	// Truth arrives BELOW the estimate: settling returns the difference to the window,
	// so a 450-token request now fits (500 settled + 450 <= 1000 — it would not have
	// fit had the 600 estimate stuck).
	r1.Settle(ratelimit.NewTokenUsage(500, 0, 0))
	r2, err := g.Admit(ratelimit.NewTokenUsage(450, 0, 0))
	if err != nil {
		t.Fatalf("post-settle Admit err = %v, want admit after the estimate returned headroom", err)
	}
	r2.Settle(ratelimit.NewTokenUsage(450, 0, 0))

	// 950 settled this window: 100 more sheds.
	if _, err = g.Admit(ratelimit.NewTokenUsage(100, 0, 0)); err == nil {
		t.Fatal("Admit over the settled window = admit, want shed at 950/1000 + 100")
	}

	// The window rolls: settled usage expires and the same request admits.
	clock = clock.Add(defaultTokenRateWindow)
	r3, err := g.Admit(ratelimit.NewTokenUsage(100, 0, 0))
	if err != nil {
		t.Fatalf("post-roll Admit err = %v, want admit in a fresh window", err)
	}
	snap := g.Snapshot()
	if snap.Settled != (ratelimit.TokenUsage{}) || snap.InFlight != 1 {
		t.Fatalf("post-roll snapshot = %+v, want zero settled usage and 1 in flight", snap)
	}
	r3.Release()
}

// TestTokenRateGateConcurrencyAndTargetUtilization witnesses the two remaining #2019
// admission dimensions: the provider concurrency cap counts in-flight reservations, and
// TargetUtilization=0.9 sheds at the ~90% headroom target below the raw cap. It also
// pins the conservative Release edge: a reservation abandoned without truth keeps its
// estimate charged.
func TestTokenRateGateConcurrencyAndTargetUtilization(t *testing.T) {
	g := NewTokenRateGate(TokenRatePolicy{Caps: ratelimit.TokenCaps{
		MaxConcurrent:     1,
		MaxTotalTokens:    1000,
		TargetUtilization: 0.9,
	}})

	// 901 total is under the raw 1000 cap but over the 90% target (900): shed.
	_, err := g.Admit(ratelimit.NewTokenUsage(901, 0, 0))
	var ae *AdmissionError
	if !errors.As(err, &ae) || !strings.Contains(ae.Reason, ratelimit.TokenCapTotalTokens) || !strings.Contains(ae.Reason, "900") {
		t.Fatalf("target-utilization shed = %v, want %q fired at the effective limit 900", err, ratelimit.TokenCapTotalTokens)
	}

	r1, err := g.Admit(ratelimit.NewTokenUsage(890, 0, 10)) // exactly the 900 target
	if err != nil {
		t.Fatalf("at-target Admit err = %v, want admit at exactly the effective limit", err)
	}

	// One reservation is in flight and MaxConcurrent is 1: the concurrency cap fires.
	_, err = g.Admit(ratelimit.NewTokenUsage(1, 0, 0))
	if !errors.As(err, &ae) || !strings.Contains(ae.Reason, ratelimit.TokenCapConcurrency) {
		t.Fatalf("concurrent Admit = %v, want the %q cap named", err, ratelimit.TokenCapConcurrency)
	}

	// Truth frees the seat and most of the token estimate.
	r1.Settle(ratelimit.NewTokenUsage(90, 0, 10))
	r2, err := g.Admit(ratelimit.NewTokenUsage(1, 0, 0))
	if err != nil {
		t.Fatalf("post-settle Admit err = %v, want the freed seat admitted", err)
	}

	// Abandoned without truth: Release keeps the conservative estimate in the window.
	r2.Release()
	if got := g.Snapshot(); got.InFlight != 0 || got.Settled.TotalTokens() != 101 {
		t.Fatalf("post-release snapshot = %+v, want 0 in flight and 101 settled total (100 truth + 1 kept estimate)", got)
	}
}

// TestServedAdmissionTokenGateWiring is the served-path seam witness for #2019: the
// token gate composes with the scheduler slot gate through beginServedAdmission, the
// settled usage is PROVIDER-NORMALIZED from an OpenAI/Codex-shaped fixture (the cached
// portion of prompt_tokens is not charged as uncached — the issue's acceptance fixture),
// the fed-back usage drives the next admission to a typed 429, and a token shed releases
// the scheduler slot it briefly held.
func TestServedAdmissionTokenGateWiring(t *testing.T) {
	ctx := context.Background()
	srv := newTestServer(t)
	ctl := NewAdmissionController(DefaultAdmissionPolicy())
	srv.SetAdmissionController(ctl)
	gate := NewTokenRateGate(TokenRatePolicy{Caps: ratelimit.TokenCaps{MaxUncachedInputTokens: 99}})
	srv.SetTokenRateGate(gate)
	turn := servedSessionTurn{traceID: "tpm", state: SessionState{Priority: 0}, maxTokens: 1}

	lease, err := srv.beginServedAdmission(ctx, turn, nil, nil, 1)
	if err != nil || lease == nil {
		t.Fatalf("beginServedAdmission = (%v, %v), want an admitted lease under an empty window", lease, err)
	}

	// The provider answers with the OpenAI/Codex usage shape: 120 prompt tokens of which
	// 60 were cache reads (nested prompt_tokens_details), 5 completion tokens.
	lease.SettleUsage(agent.Usage{
		PromptTokens:        120,
		CompletionTokens:    5,
		PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: 60},
	})
	lease.Release()
	want := ratelimit.TokenUsage{InputTokens: 120, CachedInputTokens: 60, UncachedInputTokens: 60, OutputTokens: 5}
	if snap := gate.Snapshot(); snap.Settled != want {
		t.Fatalf("settled usage = %+v, want the provider-normalized split %+v (cached NOT charged as uncached)", snap.Settled, want)
	}

	// 60 uncached settled + a ~50-uncached-token prompt estimate exceeds the 99 cap:
	// the served path sheds as a typed 429 and the scheduler slot is released.
	msgs := []agent.Message{{Role: "user", Content: strings.Repeat("x", 196)}} // 200 chars -> 50 tokens
	_, err = srv.beginServedAdmission(ctx, turn, msgs, nil, 1)
	status, code, _, ok := admissionErrorStatus(err)
	if !ok || status != http.StatusTooManyRequests || code != "scheduler_overloaded" {
		t.Fatalf("token shed mapping = (%d, %q, %v), want (429, scheduler_overloaded, true); err=%v", status, code, ok, err)
	}
	if st := ctl.Stats(); st.Running != 0 {
		t.Fatalf("scheduler running = %d after a token shed, want 0 (the slot the shed request briefly held must be released)", st.Running)
	}
}

// TestEstimateServedTokenUsageSplitsInputOutput pins the admission-time estimate's
// input/output split: chars/4 on the prompt side charged as uncached input, max_tokens
// as the planned output (floor 1 when unset), matching the scheduler gate's heuristic.
func TestEstimateServedTokenUsageSplitsInputOutput(t *testing.T) {
	msgs := []agent.Message{{Role: "user", Content: strings.Repeat("a", 96)}} // 100 chars
	got := estimateServedTokenUsage(msgs, nil, 7)
	want := ratelimit.TokenUsage{InputTokens: 25, UncachedInputTokens: 25, OutputTokens: 7}
	if got != want {
		t.Fatalf("estimate = %+v, want %+v", got, want)
	}
	if got := estimateServedTokenUsage(nil, nil, 0); got != (ratelimit.TokenUsage{OutputTokens: 1}) {
		t.Fatalf("empty estimate = %+v, want the output floor of 1", got)
	}
}
