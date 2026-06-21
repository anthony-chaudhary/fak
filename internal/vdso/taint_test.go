package vdso

// taint_test.go — the vDSO's own one-hop taint-laundering hole on the PAYLOAD label.
// (The complementary SESSION-LEDGER laundering on a tier-2 hit is closed in the IFC
// layer by vdsoTaintEmitter; this file covers the locally-computed tier-1/tier-3 serve.)

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"

	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

// A tier-1 pure result must not LAUNDER a quarantined input to Trusted in one hop: a
// pure function's output is at most as trustworthy as its args. Only positive proof
// (Quarantined) downgrades; an unstamped/zero-Tainted arg keeps the adjudicated-safe
// default, so internal pure calls are not over-tainted (the SinkGate zero-value rule).
func TestServedTaint_Tier1DoesNotLaunderQuarantined(t *testing.T) {
	ctx := context.Background()
	v := New(8)
	v.RegisterPure("calculate", calcSum)

	// Quarantined args -> Quarantined result (the laundering this closes).
	q := &abi.ToolCall{
		Tool: "calculate",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"a":1,"b":2}`), Taint: abi.TaintQuarantined},
		Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
	}
	res, ok := v.Lookup(ctx, q)
	if !ok {
		t.Fatalf("tier-1 calculate (quarantined args): miss, want hit")
	}
	if res.Payload.Taint != abi.TaintQuarantined {
		t.Errorf("quarantined-arg pure serve Taint=%d, want Quarantined(%d) — one-hop laundering still open",
			res.Payload.Taint, abi.TaintQuarantined)
	}

	// Unstamped (zero == Tainted) args -> Trusted: must NOT over-taint internal calls.
	res2, hit := v.Lookup(ctx, roCall("calculate", `{"a":3,"b":4}`))
	if !hit {
		t.Fatalf("tier-1 calculate (unstamped args): miss, want hit")
	}
	if res2.Payload.Taint != abi.TaintTrusted {
		t.Errorf("unstamped-arg pure serve Taint=%d, want Trusted — over-tainting internal pure calls",
			res2.Payload.Taint)
	}
}

// A tier-3 static answer is args-independent (canned), so it stays Trusted even when the
// call's args are quarantined — laundering does not apply to data the result does not
// depend on.
func TestServedTaint_Tier3StaticStaysTrusted(t *testing.T) {
	ctx := context.Background()
	v := New(8)
	v.RegisterStatic("list_all_airports", []byte(`{"airports":["SFO"]}`))

	c := &abi.ToolCall{
		Tool: "list_all_airports",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`), Taint: abi.TaintQuarantined},
	}
	res, ok := v.Lookup(ctx, c)
	if !ok {
		t.Fatalf("tier-3 static: miss, want hit")
	}
	if res.Payload.Taint != abi.TaintTrusted {
		t.Errorf("static serve Taint=%d, want Trusted (args-independent, no laundering)", res.Payload.Taint)
	}
}
