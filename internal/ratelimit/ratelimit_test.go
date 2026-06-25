package ratelimit

import (
	"context"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // register the Ref resolver (CAS backend)
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

func call(trace, tool, args string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool:    tool,
		TraceID: trace,
		Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(args)},
	}
}

func mustDefer(t *testing.T, v abi.Verdict, ctxMsg string) {
	t.Helper()
	if v.Kind != abi.VerdictDefer {
		t.Fatalf("%s: verdict = %v (reason %s), want Defer (abstain)", ctxMsg, v.Kind, abi.ReasonName(v.Reason))
	}
}

func mustRateLimited(t *testing.T, v abi.Verdict, ctxMsg string) {
	t.Helper()
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("%s: verdict = %v, want Deny", ctxMsg, v.Kind)
	}
	if v.Reason != abi.ReasonRateLimited {
		t.Fatalf("%s: reason = %s, want RATE_LIMITED", ctxMsg, abi.ReasonName(v.Reason))
	}
}

// TestInertUntilConfigured is the "no spontaneous refusal" contract: a limiter with
// no cap Defers on every call, however many times it is hit.
func TestInertUntilConfigured(t *testing.T) {
	ctx := context.Background()
	r := New()
	for i := 0; i < 100; i++ {
		mustDefer(t, r.Adjudicate(ctx, call("t", "tool", "{}")), "inert limiter")
	}
	if a, d, _ := r.Stats(); a != 0 || d != 0 {
		t.Fatalf("inert limiter should neither admit nor deny; admits=%d denies=%d", a, d)
	}
}

// TestQuotaDeniesOverCap is the core enforcer: the first MaxCalls calls for a key
// are admitted (Defer), and the very next one is Deny(RATE_LIMITED).
func TestQuotaDeniesOverCap(t *testing.T) {
	ctx := context.Background()
	r := New()
	r.SetLimit(Limit{MaxCalls: 3}, KeyPerTrace)

	for i := 0; i < 3; i++ {
		mustDefer(t, r.Adjudicate(ctx, call("t1", "search", "{}")), "under-cap call "+strconv.Itoa(i))
	}
	mustRateLimited(t, r.Adjudicate(ctx, call("t1", "search", "{}")), "the over-cap call")

	if a, d, _ := r.Stats(); a != 3 || d != 1 {
		t.Fatalf("admits=%d denies=%d, want 3 and 1", a, d)
	}
}

// TestDeniedCallConsumesNoBudget proves an over-cap probe is idempotent: repeatedly
// hitting an exhausted key keeps returning RATE_LIMITED and never advances the
// admit counter past the cap (a refused call consumes no budget).
func TestDeniedCallConsumesNoBudget(t *testing.T) {
	ctx := context.Background()
	r := New()
	r.SetLimit(Limit{MaxCalls: 2}, KeyPerTrace)

	mustDefer(t, r.Adjudicate(ctx, call("t", "x", "{}")), "call 1")
	mustDefer(t, r.Adjudicate(ctx, call("t", "x", "{}")), "call 2")
	for i := 0; i < 5; i++ {
		mustRateLimited(t, r.Adjudicate(ctx, call("t", "x", "{}")), "over-cap probe "+strconv.Itoa(i))
	}
	if a, _, _ := r.Stats(); a != 2 {
		t.Fatalf("admits=%d, want exactly 2 (denied probes must not consume budget)", a)
	}
}

// TestPerTraceIsolation proves the per-trace dimension caps one agent/run without
// throttling its peers: trace A can be exhausted while trace B still flows.
func TestPerTraceIsolation(t *testing.T) {
	ctx := context.Background()
	r := New()
	r.SetLimit(Limit{MaxCalls: 1}, KeyPerTrace)

	mustDefer(t, r.Adjudicate(ctx, call("A", "x", "{}")), "A first")
	mustRateLimited(t, r.Adjudicate(ctx, call("A", "x", "{}")), "A over cap")
	// B is a different trace — its budget is untouched.
	mustDefer(t, r.Adjudicate(ctx, call("B", "x", "{}")), "B first")
}

// TestPerToolMode buckets by tool name across callers.
func TestPerToolMode(t *testing.T) {
	ctx := context.Background()
	r := New()
	r.SetLimit(Limit{MaxCalls: 1}, KeyPerTool)

	mustDefer(t, r.Adjudicate(ctx, call("A", "hot", "{}")), "hot first")
	// Same tool, different trace — same bucket, so this is over cap.
	mustRateLimited(t, r.Adjudicate(ctx, call("B", "hot", "{}")), "hot from another trace")
	// A different tool is its own bucket.
	mustDefer(t, r.Adjudicate(ctx, call("A", "cold", "{}")), "cold tool")
}

// TestGlobalMode is one shared budget for the whole process.
func TestGlobalMode(t *testing.T) {
	ctx := context.Background()
	r := New()
	r.SetLimit(Limit{MaxCalls: 2}, KeyGlobal)

	mustDefer(t, r.Adjudicate(ctx, call("A", "x", "{}")), "global 1")
	mustDefer(t, r.Adjudicate(ctx, call("B", "y", "{}")), "global 2")
	mustRateLimited(t, r.Adjudicate(ctx, call("C", "z", "{}")), "global over cap")
}

// TestCostBudgetDeniesOverBudget caps cumulative cost (arg bytes), not call count:
// a key whose calls sum past MaxCost is denied even if it is under any call quota.
func TestCostBudgetDeniesOverBudget(t *testing.T) {
	ctx := context.Background()
	r := New()
	r.SetLimit(Limit{MaxCost: 10}, KeyPerTrace) // 10 bytes of args per trace

	mustDefer(t, r.Adjudicate(ctx, call("t", "x", "12345")), "5 bytes")                // cost 5, total 5
	mustDefer(t, r.Adjudicate(ctx, call("t", "x", "1234")), "4 bytes")                 // cost 4, total 9
	mustRateLimited(t, r.Adjudicate(ctx, call("t", "x", "12")), "would exceed budget") // +2 -> 11 > 10
	// A zero-cost call still fits (total stays 9).
	mustDefer(t, r.Adjudicate(ctx, call("t", "x", "")), "empty args fit")
}

// TestExplicitCostOverride proves a caller can supply a real token count via Meta,
// overriding the byte proxy.
func TestExplicitCostOverride(t *testing.T) {
	ctx := context.Background()
	r := New()
	r.SetLimit(Limit{MaxCost: 100}, KeyPerTrace)

	c := call("t", "x", "tiny") // 4 bytes, but declares cost 100
	c.Meta = map[string]string{"fak.ratelimit.cost": "100"}
	mustDefer(t, r.Adjudicate(ctx, c), "exact-budget call")
	// Any further cost exceeds the budget.
	c2 := call("t", "x", "x")
	c2.Meta = map[string]string{"fak.ratelimit.cost": "1"}
	mustRateLimited(t, r.Adjudicate(ctx, c2), "over the declared-cost budget")
}

// TestResetClearsBudget proves the lifecycle hook: resetting a key restores its
// full budget (the per-trace ledger reset issue #12 drives at trace end).
func TestResetClearsBudget(t *testing.T) {
	ctx := context.Background()
	r := New()
	r.SetLimit(Limit{MaxCalls: 1}, KeyPerTrace)

	mustDefer(t, r.Adjudicate(ctx, call("t", "x", "{}")), "first")
	mustRateLimited(t, r.Adjudicate(ctx, call("t", "x", "{}")), "over cap")
	r.Reset("trace:t")
	mustDefer(t, r.Adjudicate(ctx, call("t", "x", "{}")), "after reset")
}

// TestBoundedKeysFailOpen proves state is bounded: past the key ceiling a NEW key
// is not tracked (fail-open Defer) rather than evicting a live budget, and the drop
// is accounted. Existing keys keep enforcing.
func TestBoundedKeysFailOpen(t *testing.T) {
	ctx := context.Background()
	r := New()
	r.SetLimit(Limit{MaxCalls: 1}, KeyPerTrace)
	r.maxKeys = 2 // tiny ceiling for the test

	// Fill the two key slots and exhaust both.
	mustDefer(t, r.Adjudicate(ctx, call("k0", "x", "{}")), "k0")
	mustDefer(t, r.Adjudicate(ctx, call("k1", "x", "{}")), "k1")
	// A third, new key cannot be tracked -> fail-open Defer, counted as dropped.
	mustDefer(t, r.Adjudicate(ctx, call("k2", "x", "{}")), "k2 past ceiling")
	if _, _, dropped := r.Stats(); dropped != 1 {
		t.Fatalf("dropped=%d, want 1 (the untracked over-ceiling key)", dropped)
	}
	// An EXISTING key still enforces its cap (not bypassed by the ceiling).
	mustRateLimited(t, r.Adjudicate(ctx, call("k0", "x", "{}")), "existing key still capped")
}

// TestLimiterDenyCarriesRetryAfter is the issue-#699 retry-after witness: an
// over-cap deny carries an advisory back-off in its meta — both a Go
// time.Duration string (retry_after) and integer milliseconds (retry_after_ms) —
// while a sub-cap call Defers and carries none. The declared RetryAfter wins when
// set; an undeclared cap still carries the documented default advisory constant.
func TestLimiterDenyCarriesRetryAfter(t *testing.T) {
	ctx := context.Background()

	// Declared retry-after: the over-cap deny surfaces exactly that back-off.
	r := New()
	r.SetLimit(Limit{MaxCalls: 1, RetryAfter: 500 * time.Millisecond}, KeyPerTrace)
	mustDefer(t, r.Adjudicate(ctx, call("t", "x", "{}")), "sub-cap call Defers (no deny, no retry_after)")
	v := r.Adjudicate(ctx, call("t", "x", "{}")) // over cap
	mustRateLimited(t, v, "over-cap call")

	ra, err := time.ParseDuration(v.Meta["retry_after"])
	if err != nil || ra != 500*time.Millisecond {
		t.Fatalf("retry_after = %q (%v), want 500ms", v.Meta["retry_after"], err)
	}
	ms, err := strconv.Atoi(v.Meta["retry_after_ms"])
	if err != nil || ms != 500 {
		t.Fatalf("retry_after_ms = %q (%v), want 500", v.Meta["retry_after_ms"], err)
	}

	// Undeclared retry-after: the over-cap deny still carries the documented default
	// advisory constant (the limiter has no rate window to derive a duration from).
	r2 := New()
	r2.SetLimit(Limit{MaxCalls: 1}, KeyPerTrace)
	r2.Adjudicate(ctx, call("t2", "x", "{}")) // consume the one-call quota
	v2 := r2.Adjudicate(ctx, call("t2", "x", "{}")) // over cap
	mustRateLimited(t, v2, "over-cap call with default retry-after")
	if ra, err := time.ParseDuration(v2.Meta["retry_after"]); err != nil || ra != defaultRetryAfter {
		t.Fatalf("default retry_after = %q (%v), want %s", v2.Meta["retry_after"], err, defaultRetryAfter)
	}
	if v2.Meta["retry_after_ms"] == "" {
		t.Fatal("over-cap deny must always carry a non-empty retry_after_ms (issue #699)")
	}

	// The cap/limit/key forensics are still present alongside the new hint.
	if v.Meta["cap"] != "max_calls" || v.Meta["key"] != "trace:t" {
		t.Fatalf("deny forensics lost: cap=%q key=%q", v.Meta["cap"], v.Meta["key"])
	}
}

// --- kernel integration: the issue-#13 witness -----------------------------------

// allowAll is the affirmative-allow rung so an under-cap call resolves to Allow
// (an all-Defer chain folds to DEFAULT_DENY).
type allowAll struct{}

func (allowAll) Caps() []abi.Capability { return nil }
func (allowAll) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictAllow, By: "test"}
}

// TestRateLimitedDenySurfacesWaitDisposition is the issue-#13 acceptance witness:
// driven through a REAL kernel, a run exceeding the configured cap gets a
// RATE_LIMITED deny whose disposition is WAIT — the loop is steered to back off,
// not to burn another turn retrying. Under-cap calls are admitted (Allow); the
// over-cap call denies; the deny-as-value the next turn consumes carries WAIT.
func TestRateLimitedDenySurfacesWaitDisposition(t *testing.T) {
	ctx := context.Background()

	// Build the chain: an affirmative allow + a capped governor at its real rank.
	abi.RegisterAdjudicator(0, allowAll{})
	lim := New()
	lim.SetLimit(Limit{MaxCalls: 3}, KeyPerTrace)
	abi.RegisterAdjudicator(8, lim)

	k := kernel.New("mock") // no dispatch happens (under-cap via Submit; over-cap denies pre-dispatch)
	k.SetVDSO(false)        // every call must reach the adjudicator chain, no fast-path

	// Under cap: the first 3 calls on the trace are admitted (Allow wins the fold).
	for i := 0; i < 3; i++ {
		_, v := k.Submit(ctx, call("run-1", "tool", "{}"))
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("under-cap call %d: verdict = %v (reason %s), want Allow", i, v.Kind, abi.ReasonName(v.Reason))
		}
	}

	// Over cap: the 4th call is denied with RATE_LIMITED, and the kernel maps that
	// reason to a WAIT disposition.
	_, v := k.Submit(ctx, call("run-1", "tool", "{}"))
	mustRateLimited(t, v, "over-cap Submit")
	if d := kernel.Disposition(v.Reason); d != "WAIT" {
		t.Fatalf("disposition = %q, want WAIT (a back-off, not a wasted retry)", d)
	}

	// The deny-as-value a Syscall hands the next turn carries the same: a denied
	// call returns the structured DenyResult with reason RATE_LIMITED + WAIT, with
	// no engine dispatch.
	r, v2 := k.Syscall(ctx, call("run-1", "tool", "{}"))
	mustRateLimited(t, v2, "over-cap Syscall")
	if r == nil {
		t.Fatal("Syscall returned nil result for a denied call")
	}
	if got := r.Meta["disposition"]; got != "WAIT" {
		t.Fatalf("DenyResult disposition = %q, want WAIT", got)
	}
	if got := r.Meta["reason"]; got != "RATE_LIMITED" {
		t.Fatalf("DenyResult reason = %q, want RATE_LIMITED", got)
	}

	// A different run is unaffected — the cap is per-trace, not global.
	_, vb := k.Submit(ctx, call("run-2", "tool", "{}"))
	if vb.Kind != abi.VerdictAllow {
		t.Fatalf("a fresh run was throttled by another run's budget: verdict = %v", vb.Kind)
	}
}
