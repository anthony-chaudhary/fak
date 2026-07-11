package metrics

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// stubMiddleware is a test double whose Handle result and fail contract are set
// per-case, so each test fixes exactly one behavior (error / panic / clean).
type stubMiddleware struct {
	mode    FailMode
	verdict abi.Verdict
	err     error
	panicV  any // non-nil => Handle panics with this value
}

func (s stubMiddleware) Handle(context.Context, *abi.ToolCall) (abi.Verdict, error) {
	if s.panicV != nil {
		panic(s.panicV)
	}
	return s.verdict, s.err
}

func (s stubMiddleware) Mode() FailMode { return s.mode }

// TestThrowingEnforcerDenies is the issue's named acceptance (#2904): a
// security-path (fail-closed) middleware that ERRORS must deny the call, not
// admit it.
func TestThrowingEnforcerDenies(t *testing.T) {
	m := stubMiddleware{mode: FailClosed, err: errors.New("adjudicator exploded")}

	got := Apply(context.Background(), m, &abi.ToolCall{Tool: "rm"})

	if got.Kind != abi.VerdictDeny {
		t.Fatalf("throwing enforcer: Kind = %v, want VerdictDeny (fail-closed must not admit)", got.Kind)
	}
	if got.Reason != abi.ReasonDefaultDeny {
		t.Errorf("throwing enforcer: Reason = %v (%s), want ReasonDefaultDeny",
			got.Reason, abi.ReasonName(got.Reason))
	}
}

// TestPanickingEnforcerDenies proves the guarantee is structural: a middleware
// that THROWS (panics) is contained and denies rather than unwinding the security
// path into an accidental admit.
func TestPanickingEnforcerDenies(t *testing.T) {
	m := stubMiddleware{mode: FailClosed, panicV: "nil map write"}

	got := Apply(context.Background(), m, &abi.ToolCall{Tool: "rm"})

	if got.Kind != abi.VerdictDeny {
		t.Fatalf("panicking enforcer: Kind = %v, want VerdictDeny", got.Kind)
	}
}

// TestFailClosedIsZeroValue proves the distinction is structural: a middleware
// that never declares a mode still fails closed, so forgetting the contract can
// never silently widen authority.
func TestFailClosedIsZeroValue(t *testing.T) {
	if (FailMode(0)) != FailClosed {
		t.Fatalf("zero-value FailMode = %v, want FailClosed", FailMode(0))
	}
	// A stub with the zero-value mode and an error must still deny.
	m := stubMiddleware{err: errors.New("boom")} // mode defaults to FailClosed
	got := Apply(context.Background(), m, &abi.ToolCall{Tool: "curl"})
	if got.Kind != abi.VerdictDeny {
		t.Fatalf("zero-value-mode enforcer: Kind = %v, want VerdictDeny", got.Kind)
	}
}

// TestFailOpenObserverDefers proves the other half of the contract: a fail-open
// middleware (telemetry) that errors is skipped — it defers to the next link
// rather than blocking the call.
func TestFailOpenObserverDefers(t *testing.T) {
	m := stubMiddleware{mode: FailOpen, err: errors.New("telemetry sink down")}

	got := Apply(context.Background(), m, &abi.ToolCall{Tool: "ls"})

	if got.Kind != abi.VerdictDefer {
		t.Fatalf("fail-open middleware error: Kind = %v, want VerdictDefer (must not block)", got.Kind)
	}
	if got.Kind == abi.VerdictDeny {
		t.Error("fail-open middleware denied on error — observers must not block")
	}
}

// TestSuccessfulMiddlewarePassesThrough proves the fail contract only fires on
// error: a middleware that returns cleanly yields its own verdict verbatim.
func TestSuccessfulMiddlewarePassesThrough(t *testing.T) {
	want := abi.Verdict{Kind: abi.VerdictAllow, By: "unit"}
	m := stubMiddleware{mode: FailClosed, verdict: want}

	got := Apply(context.Background(), m, &abi.ToolCall{Tool: "cat"})

	if got.Kind != abi.VerdictAllow || got.By != "unit" {
		t.Fatalf("clean middleware: got %+v, want %+v (verdict must pass through untouched)", got, want)
	}
}

// TestFailModeString pins the contract tokens the issue names (fail_open /
// fail_closed), so a serialized or logged verdict labels the contract.
func TestFailModeString(t *testing.T) {
	if FailClosed.String() != "fail_closed" {
		t.Errorf("FailClosed.String() = %q, want fail_closed", FailClosed.String())
	}
	if FailOpen.String() != "fail_open" {
		t.Errorf("FailOpen.String() = %q, want fail_open", FailOpen.String())
	}
}
