package abi_test

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/abi"
)

// denyAdj is a driver-shaped Adjudicator implemented entirely against the
// re-exported pkg/abi surface — it never names internal/abi. If this compiles
// and satisfies abi.Adjudicator, the alias identity holds.
type denyAdj struct{ tool string }

func (d denyAdj) Adjudicate(_ context.Context, c *abi.ToolCall) abi.Verdict {
	if c != nil && c.Tool == d.tool {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "extdriver-test"}
	}
	return abi.Verdict{Kind: abi.VerdictDefer}
}
func (denyAdj) Caps() []abi.Capability { return nil }

// TestVendorRegistrationThroughShim claims a vendor VerdictKind through the
// re-exported RegisterVerdictKind and asserts it resolves via the re-exported
// FoldRank — i.e. a registration made through pkg/abi lands in the same frozen
// registry internal/abi reads. This is the end-to-end witness that the shim is a
// real seam, not just a compile-time type-alias.
func TestVendorRegistrationThroughShim(t *testing.T) {
	// A kind inside the VerdictsVendor range [1024, 1<<16).
	const kind abi.VerdictKind = 1024 + 0xAB
	if !inRange(uint32(kind), abi.VerdictsVendor) {
		t.Fatalf("test kind %d not inside VerdictsVendor %+v", kind, abi.VerdictsVendor)
	}
	const wantRank = 250
	abi.RegisterVerdictKind(kind, "EXTDRIVER_TEST_DENY", wantRank, abi.FallbackDeny)

	if got := abi.FoldRank(kind); got != wantRank {
		t.Fatalf("FoldRank(%d) through shim = %d, want %d", kind, got, wantRank)
	}
}

// TestDriverInterfaceIdentity proves a driver type implemented purely against
// pkg/abi satisfies the interface and produces a well-formed Verdict via the
// re-exported constants.
func TestDriverInterfaceIdentity(t *testing.T) {
	var a abi.Adjudicator = denyAdj{tool: "refund_payment"}
	v := a.Adjudicate(context.Background(), &abi.ToolCall{Tool: "refund_payment"})
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("verdict kind = %v, want VerdictDeny", v.Kind)
	}
	if abi.ReasonName(v.Reason) != "POLICY_BLOCK" {
		t.Fatalf("ReasonName(%v) = %q, want POLICY_BLOCK", v.Reason, abi.ReasonName(v.Reason))
	}
	// A non-targeted tool must Defer (the fold identity).
	if v2 := a.Adjudicate(context.Background(), &abi.ToolCall{Tool: "read_file"}); v2.Kind != abi.VerdictDefer {
		t.Fatalf("non-targeted verdict = %v, want VerdictDefer", v2.Kind)
	}
}

// TestVendorRangesAreUsable confirms the re-exported reserved ranges carry the
// frozen bounds a vendor needs to pick a non-colliding number.
func TestVendorRangesAreUsable(t *testing.T) {
	if abi.OpsVendor.Lo != 1<<16 || abi.OpsVendor.Hi != 1<<17 {
		t.Fatalf("OpsVendor = %+v, want {65536, 131072}", abi.OpsVendor)
	}
	if abi.VerdictsVendor.Lo != 1024 || abi.VerdictsVendor.Hi != 1<<16 {
		t.Fatalf("VerdictsVendor = %+v, want {1024, 65536}", abi.VerdictsVendor)
	}
}

func inRange(n uint32, r abi.Range) bool { return n >= r.Lo && n < r.Hi }
