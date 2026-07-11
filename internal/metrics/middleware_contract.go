// middleware_contract.go — the adjudicated middleware/observer extension seam
// with an explicit fail_open|fail_closed contract (#2904, epic #2871).
//
// Hermes exposes two plugin contracts around the llm/tool request-and-execution
// seam — hermes.observer.v1 (report) and hermes.middleware.v1 (change behavior,
// nested via next_call). BOTH are fail-open by design: a broken plugin never
// blocks the agent. That is correct for telemetry and wrong for enforcement.
//
// fak is the enforcement point, so the fak-does-it-better distinction is made
// STRUCTURAL here: an observer is fail-open (a broken observer never blocks a
// call), but a middleware on the security path is fail-CLOSED (if the
// adjudicating middleware errors — returns an error OR panics — the call is
// DENIED, not admitted). The distinction is not a comment or a runtime flag; it
// is the zero value of a closed enum, so a middleware that forgets to declare a
// mode denies-on-error by construction and can never silently widen authority.
//
// This mirrors the frozen-ABI principle in internal/abi: an unknown verdict kind
// resolves to FallbackDeny, "so an unknown kind can never silently widen
// authority." The seam builds on the real security-path types (abi.ToolCall,
// abi.Verdict) rather than a private shadow so a fail-closed deny here is the
// same provable VerdictDeny the adjudicator chain folds.
package metrics

import (
	"context"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// FailMode is the CLOSED fail contract that structurally separates a fail-OPEN
// observer from a fail-CLOSED security-path enforcer. The zero value is
// FailClosed on purpose: a security-path middleware that neglects to declare its
// mode denies-on-error rather than admitting, so forgetting the contract fails
// safe.
type FailMode uint8

const (
	// FailClosed is the security-path default: when the middleware errors
	// (returns a non-nil error or panics) the call is DENIED, never admitted.
	// This is the enforcement contract — a broken enforcer refuses rather than
	// waving the call through.
	FailClosed FailMode = iota
	// FailOpen is the telemetry contract (mirrors Hermes' plugins): when the
	// middleware errors, the failure is swallowed and the call proceeds to the
	// next link. A broken observer never blocks the agent.
	FailOpen
)

// String renders the mode as the issue's contract token (fail_closed/fail_open),
// so a logged or serialized verdict names the contract that produced it.
func (m FailMode) String() string {
	if m == FailOpen {
		return "fail_open"
	}
	return "fail_closed"
}

// Middleware is one link on the tool-request/adjudication path (mirrors
// hermes.middleware.v1, but carrying a fail contract). Handle returns an
// abi.Verdict for the call; a non-nil error means the middleware itself failed to
// adjudicate. How that failure resolves is governed by Mode — the structural
// distinction between an observer and an enforcer.
type Middleware interface {
	Handle(ctx context.Context, c *abi.ToolCall) (abi.Verdict, error)
	// Mode reports the fail contract. Security-path enforcers return FailClosed
	// (the zero value); pure observers wired as middleware return FailOpen.
	Mode() FailMode
}

// Observer is the report-only seam (mirrors hermes.observer.v1): it witnesses a
// call but cannot change its disposition. An observer is fail-open by definition
// — Apply contains a panic so a broken observer never blocks the call — so it
// carries no FailMode of its own.
type Observer interface {
	Observe(ctx context.Context, c *abi.ToolCall)
}

// middlewareBy is the forensic By tag stamped on a fail-contract verdict so an
// audit can tell a deny/defer synthesized by this seam apart from one a real
// adjudicator decided. It is forensics, not dispatch (see abi.Verdict.By).
const middlewareBy = "metrics/middleware"

// Apply runs a middleware under its fail contract and returns the effective
// verdict. On a nil error it returns the middleware's own verdict unchanged. On a
// non-nil error OR a recovered panic it applies the contract:
//
//   - FailClosed -> abi.Verdict{Kind: VerdictDeny, Reason: ReasonDefaultDeny}:
//     the call is refused. A broken enforcer denies; it never admits.
//   - FailOpen   -> abi.Verdict{Kind: VerdictDefer}: this link is skipped and the
//     call proceeds to the next adjudicator. A broken observer never blocks.
//
// Recovering the panic is what makes the guarantee structural rather than
// advisory: the issue's "throwing enforcer" cannot crash the security path INTO
// an admit — a recovered panic is an error, and an errored enforcer denies.
// ReasonDefaultDeny is the exact fail-closed reason ("allowed only once a policy
// affirmatively permits it; none did").
func Apply(ctx context.Context, m Middleware, c *abi.ToolCall) abi.Verdict {
	v, err := safeHandle(ctx, m, c)
	if err == nil {
		return v
	}
	if m.Mode() == FailOpen {
		return abi.Verdict{Kind: abi.VerdictDefer, By: middlewareBy}
	}
	return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: middlewareBy}
}

// safeHandle invokes m.Handle, converting a panic into a non-nil error so the
// caller's fail contract governs a throwing middleware exactly as it governs one
// that returns an error. Without this, a panicking enforcer would unwind past the
// security path — the fail-open-by-accident bug the issue exists to prevent.
func safeHandle(ctx context.Context, m Middleware, c *abi.ToolCall) (v abi.Verdict, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("middleware panicked: %v", r)
		}
	}()
	return m.Handle(ctx, c)
}
