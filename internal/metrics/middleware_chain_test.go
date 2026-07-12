package metrics

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// allow / defer are the two clean (non-erroring) stub verdicts the chain tests
// compose around; the throwing enforcers are built inline per case.
var (
	cleanAllow = stubMiddleware{mode: FailClosed, verdict: abi.Verdict{Kind: abi.VerdictAllow, By: "unit"}}
	cleanDefer = stubMiddleware{mode: FailClosed, verdict: abi.Verdict{Kind: abi.VerdictDefer, By: "unit"}}
)

// TestChainFailClosedEnforcerDeniesWholeChain is the issue's acceptance lifted to
// COMPOSITION (#2904): a throwing fail-closed enforcer buried between two clean
// allows must deny the WHOLE chain, not let a neighbouring admit leak through.
func TestChainFailClosedEnforcerDeniesWholeChain(t *testing.T) {
	throwing := stubMiddleware{mode: FailClosed, err: errors.New("adjudicator exploded")}
	chain := []Middleware{cleanAllow, throwing, cleanAllow}

	got := Chain(context.Background(), chain, &abi.ToolCall{Tool: "rm"})

	if got.Kind != abi.VerdictDeny {
		t.Fatalf("chain with throwing enforcer: Kind = %v, want VerdictDeny (a broken enforcer must deny the whole chain)", got.Kind)
	}
	if got.Reason != abi.ReasonDefaultDeny {
		t.Errorf("chain deny: Reason = %v (%s), want ReasonDefaultDeny", got.Reason, abi.ReasonName(got.Reason))
	}
}

// TestChainDenyIsOrderIndependent proves the fold is a lattice max, not a
// first/last-wins scan: the throwing enforcer denies whether it sits first or last,
// so composition order can never turn a deny into an admit.
func TestChainDenyIsOrderIndependent(t *testing.T) {
	throwing := stubMiddleware{mode: FailClosed, panicV: "nil map write"}
	for _, tc := range []struct {
		name  string
		chain []Middleware
	}{
		{"enforcer first", []Middleware{throwing, cleanAllow, cleanAllow}},
		{"enforcer last", []Middleware{cleanAllow, cleanAllow, throwing}},
		{"enforcer middle", []Middleware{cleanAllow, throwing, cleanAllow}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := Chain(context.Background(), tc.chain, &abi.ToolCall{Tool: "curl"})
			if got.Kind != abi.VerdictDeny {
				t.Fatalf("%s: Kind = %v, want VerdictDeny regardless of position", tc.name, got.Kind)
			}
		})
	}
}

// TestChainFailOpenObserverDoesNotBlock proves the other half survives composition:
// a fail-open observer that throws contributes only a Defer, so it never escalates
// the chain into a Deny — a broken observer never blocks the call.
func TestChainFailOpenObserverDoesNotBlock(t *testing.T) {
	throwingObserver := stubMiddleware{mode: FailOpen, err: errors.New("telemetry sink down")}
	chain := []Middleware{throwingObserver, cleanAllow}

	got := Chain(context.Background(), chain, &abi.ToolCall{Tool: "ls"})

	if got.Kind == abi.VerdictDeny {
		t.Fatalf("chain with throwing fail-open observer denied (Kind=%v) — an observer must never block", got.Kind)
	}
}

// TestChainEmptyDefers proves the fail-closed base case: an empty chain has no
// opinion (Defer), so it can never fabricate an admit — the kernel's downstream
// default-deny resolves a call nothing in the chain adjudicated.
func TestChainEmptyDefers(t *testing.T) {
	got := Chain(context.Background(), nil, &abi.ToolCall{Tool: "rm"})
	if got.Kind != abi.VerdictDefer {
		t.Fatalf("empty chain: Kind = %v, want VerdictDefer (no opinion, never a fabricated allow)", got.Kind)
	}
}

// TestChainAllDeferDefers proves that a chain of pure no-opinion links folds to
// Defer, not Allow: abstention never widens into an admit.
func TestChainAllDeferDefers(t *testing.T) {
	got := Chain(context.Background(), []Middleware{cleanDefer, cleanDefer}, &abi.ToolCall{Tool: "rm"})
	if got.Kind != abi.VerdictDefer {
		t.Fatalf("all-defer chain: Kind = %v, want VerdictDefer", got.Kind)
	}
}

// TestChainAllAllowPassesThrough proves the fold does not over-restrict: when every
// link cleanly allows, the chain returns that Allow (the least-restrictive winner),
// carrying the deciding link's own By rather than a synthesized tag.
func TestChainAllAllowPassesThrough(t *testing.T) {
	got := Chain(context.Background(), []Middleware{cleanAllow, cleanAllow}, &abi.ToolCall{Tool: "cat"})
	if got.Kind != abi.VerdictAllow {
		t.Fatalf("all-allow chain: Kind = %v, want VerdictAllow", got.Kind)
	}
	if got.By != "unit" {
		t.Errorf("all-allow chain: By = %q, want the deciding link's own tag %q", got.By, "unit")
	}
}
