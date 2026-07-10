package screen

// screen_test.go — the red-team floor for C1 (#4192). A read plane is an exfiltration
// surface, so these tests are adversarial by construction: they assert what MUST NOT
// happen (quarantined bytes crossing the boundary, a cross-principal read succeeding)
// with byte-exact and closed-token witnesses, not merely that the happy path works.

import (
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionread"
)

// secret is a sentinel the outbound screen must never let cross the boundary for a
// suppressed span. Every taint assertion checks this substring is absent from any bytes
// the screen returns.
var secret = []byte(`{"role":"user","content":"LAUNCH CODES: hunter2 — the dropped originating task"}`)

// TestOutboundScreenNeverEmitsQuarantinedBytes is the byte-exactness half of the
// done-condition: a sealed or tombstoned span refuses with READ_TAINT_WITHHELD and its
// bytes NEVER appear in the returned slice; a clean span is disclosed byte-for-byte.
func TestOutboundScreenNeverEmitsQuarantinedBytes(t *testing.T) {
	for _, tc := range []struct {
		name string
		span Span
	}{
		{"sealed", Span{Bytes: secret, Sealed: true}},
		{"tombstoned", Span{Bytes: secret, Tombstoned: true}},
		{"sealed-and-tombstoned", Span{Bytes: secret, Sealed: true, Tombstoned: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ScreenOutbound(tc.span)
			if err == nil {
				t.Fatalf("suppressed span disclosed with no refusal — the outbound screen leaked")
			}
			if reason := RefusalReason(err); reason != sessionread.ReasonReadTaintWithheld {
				t.Fatalf("refusal reason = %q, want %q", reason, sessionread.ReasonReadTaintWithheld)
			}
			if got != nil {
				t.Fatalf("suppressed span returned %d bytes, want nil", len(got))
			}
			// The load-bearing assertion: the secret never crossed the boundary.
			if bytes.Contains(got, secret) {
				t.Fatal("quarantined bytes appeared in the outbound read — byte-exact leak")
			}
			// The refusal message must not smuggle the bytes out either.
			if bytes.Contains([]byte(err.Error()), secret) {
				t.Fatal("the refusal message echoed the quarantined bytes")
			}
		})
	}
}

// TestCleanSpanDisclosedByteExact pins the other side: a span with no gate is disclosed
// unchanged — the screen neither mutates nor truncates a span it clears.
func TestCleanSpanDisclosedByteExact(t *testing.T) {
	clean := append([]byte(nil), secret...)
	got, err := ScreenOutbound(Span{Bytes: clean})
	if err != nil {
		t.Fatalf("clean span refused: %v", err)
	}
	if !bytes.Equal(got, clean) {
		t.Fatalf("clean span not disclosed byte-exact:\n got %q\nwant %q", got, clean)
	}
}

// TestSelfReadSucceedsCrossPrincipalRefused is the scope half of the done-condition on a
// read-self op (context-restore): the owning principal reads its own trace; a different
// principal is refused READ_SCOPE_DENIED.
func TestSelfReadSucceedsCrossPrincipalRefused(t *testing.T) {
	// Self-read: caller owns the trace it addresses → allowed.
	if err := Authorize(ScopeRequest{Op: sessionread.OpContextRestore, Caller: "traceA", TargetOwner: "traceA"}); err != nil {
		t.Fatalf("self-read of own trace refused: %v", err)
	}
	// Cross-principal: caller addresses another principal's trace → refused, closed reason.
	err := Authorize(ScopeRequest{Op: sessionread.OpContextRestore, Caller: "traceB", TargetOwner: "traceA"})
	if err == nil {
		t.Fatal("cross-principal read-self succeeded — the scope floor did not engage")
	}
	if got := RefusalReason(err); got != sessionread.ReasonReadScopeDenied {
		t.Fatalf("cross-principal refusal reason = %q, want %q", got, sessionread.ReasonReadScopeDenied)
	}
}

// TestFleetOpRequiresGrant pins the read-fleet rule: a fleet-scoped op (session-state)
// crosses the per-principal boundary and needs the fleet grant. Without it the read is
// refused — the worst-first exposure: today no principal holds the grant, so every fleet
// read is denied-by-floor until one is explicitly issued.
func TestFleetOpRequiresGrant(t *testing.T) {
	spec, ok := sessionread.Spec(sessionread.OpSessionState)
	if !ok {
		t.Fatal("precondition: OpSessionState is not a registered read op")
	}
	if spec.Capability != sessionread.CapReadFleet {
		t.Fatalf("precondition: OpSessionState capability = %q, want %q", spec.Capability, sessionread.CapReadFleet)
	}
	// No grant → denied.
	err := Authorize(ScopeRequest{Op: sessionread.OpSessionState, Caller: "traceA", FleetGrant: false})
	if err == nil {
		t.Fatal("fleet read without the grant succeeded — the floor served an unauthorized fleet read")
	}
	if got := RefusalReason(err); got != sessionread.ReasonReadScopeDenied {
		t.Fatalf("ungranted fleet refusal reason = %q, want %q", got, sessionread.ReasonReadScopeDenied)
	}
	// With the grant → allowed (a fleet read does not require Caller==TargetOwner).
	if err := Authorize(ScopeRequest{Op: sessionread.OpSessionState, Caller: "traceA", TargetOwner: "traceZ", FleetGrant: true}); err != nil {
		t.Fatalf("granted fleet read refused: %v", err)
	}
}

// TestDefaultTraceOriginatingTaskNotCrossPrincipal is the exact #4192 red-team scenario:
// on a no-RequireKey loopback guard, one principal must not read another's dropped
// originating task. It proves defense-in-depth — the read is refused at the SCOPE gate,
// and even if an attacker bypassed scope, the TAINT gate independently withholds a
// sealed span's bytes. Two independent gates, either alone sufficient.
func TestDefaultTraceOriginatingTaskNotCrossPrincipal(t *testing.T) {
	const victim = "victim-default-trace"
	const attacker = "attacker-loopback-pid"

	// Gate 1 — scope: the attacker restoring the victim's originating task is refused.
	scopeErr := Authorize(ScopeRequest{Op: sessionread.OpContextRestore, Caller: attacker, TargetOwner: victim})
	if scopeErr == nil {
		t.Fatal("attacker read the victim's default-trace task — scope floor breached")
	}
	if got := RefusalReason(scopeErr); got != sessionread.ReasonReadScopeDenied {
		t.Fatalf("scope refusal = %q, want %q", got, sessionread.ReasonReadScopeDenied)
	}

	// Gate 2 — taint: had scope been bypassed, the victim's task is a sealed span, so the
	// outbound screen still refuses and never emits its bytes.
	victimTask := Span{Bytes: secret, Sealed: true}
	body, taintErr := ScreenOutbound(victimTask)
	if taintErr == nil {
		t.Fatal("sealed originating task disclosed — taint gate breached")
	}
	if got := RefusalReason(taintErr); got != sessionread.ReasonReadTaintWithheld {
		t.Fatalf("taint refusal = %q, want %q", got, sessionread.ReasonReadTaintWithheld)
	}
	if bytes.Contains(body, secret) {
		t.Fatal("victim's originating-task bytes leaked past the taint gate")
	}
}

// TestScopeFailsClosed pins the fail-closed contract: an empty caller, an empty target on
// a read-self op, and an unknown op are all refused with a closed reason — the floor
// never defaults open on a request it cannot reason about.
func TestScopeFailsClosed(t *testing.T) {
	for _, tc := range []struct {
		name string
		req  ScopeRequest
	}{
		{"empty caller", ScopeRequest{Op: sessionread.OpContextRestore, Caller: "", TargetOwner: "traceA"}},
		{"empty target on read-self", ScopeRequest{Op: sessionread.OpContextRestore, Caller: "traceA", TargetOwner: ""}},
		{"unknown op", ScopeRequest{Op: sessionread.ReadOp("no-such-op"), Caller: "traceA", TargetOwner: "traceA"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := Authorize(tc.req)
			if err == nil {
				t.Fatal("request that should fail closed was authorized")
			}
			if got := RefusalReason(err); got != sessionread.ReasonReadScopeDenied {
				t.Fatalf("fail-closed refusal = %q, want %q", got, sessionread.ReasonReadScopeDenied)
			}
		})
	}
}

// TestScreenRefusalTokensAreClosedReadTokens grounds this package in the S vocabulary: the
// only tokens it emits are members of the closed read-refusal set. If a future edit
// introduces a bespoke token, this fails — the screen must never invent a refusal reason.
func TestScreenRefusalTokensAreClosedReadTokens(t *testing.T) {
	closed := map[string]bool{}
	for _, tok := range sessionread.ReadRefusalTokens() {
		closed[tok] = true
	}
	// Every reason this package can emit, exercised through its two entry points.
	emitted := []error{
		func() error { _, e := ScreenOutbound(Span{Sealed: true}); return e }(),
		Authorize(ScopeRequest{Op: sessionread.OpContextRestore, Caller: "x", TargetOwner: "y"}),
	}
	for _, e := range emitted {
		tok := RefusalReason(e)
		if tok == "" {
			t.Fatalf("expected a closed refusal token, got none from %v", e)
		}
		if !closed[tok] {
			t.Fatalf("emitted token %q is not in the closed read-refusal vocabulary", tok)
		}
	}
}
