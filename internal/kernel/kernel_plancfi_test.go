package kernel

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// a registered open-range escalation kind, mirroring plancfi.VerdictRequireApproval
// without importing the leaf (the kernel stays driver-blind). Fold rank 50 sits
// between Quarantine(3) and Deny(100).
const testRequireApproval abi.VerdictKind = 1024

type escalateAdj struct{}

func (escalateAdj) Caps() []abi.Capability { return nil }
func (escalateAdj) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	if c.Tool == "send_email" { // the unplanned gadget
		return abi.Verdict{Kind: testRequireApproval, Reason: abi.ReasonTrustViolation, By: "plancfi"}
	}
	return abi.Verdict{Kind: abi.VerdictDefer, By: "plancfi"}
}

// TestRegisteredEscalationVerdictIsHeldFailClosed proves the kernel's fail-closed
// default: a registered, non-core restrictive verdict (RequireApproval) is HELD —
// never dispatched — even though the monitor would allow the call. A conforming
// call still dispatches. This is the CFI deviation -> escalate path end-to-end.
func TestRegisteredEscalationVerdictIsHeldFailClosed(t *testing.T) {
	setup()
	abi.RegisterVerdictKind(testRequireApproval, "RequireApproval", 50, abi.FallbackDeny)
	// a permissive monitor that would ALLOW everything...
	abi.RegisterAdjudicator(100, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}})
	// ...plus the escalating CFI gate.
	abi.RegisterAdjudicator(25, escalateAdj{})
	eng := &countEngine{}
	abi.RegisterEngine("e", eng)
	k := New("e")
	ctx := context.Background()

	// the deviation: most-restrictive fold picks RequireApproval(50) over Allow(0);
	// the kernel HOLDS it (fail-closed default), so the exfil never dispatches.
	_, v := k.Syscall(ctx, call("send_email", "{}"))
	if v.Kind != testRequireApproval {
		t.Fatalf("the fold must pick RequireApproval over Allow, got %v", v.Kind)
	}
	if atomic.LoadInt64(&eng.n) != 0 {
		t.Fatalf("an escalated call must NOT dispatch (engine n=%d)", eng.n)
	}

	// a conforming call still reaches dispatch (CFI defers; monitor allows).
	if _, v := k.Syscall(ctx, call("search_flights", "{}")); v.Kind != abi.VerdictAllow {
		t.Fatalf("a conforming call must Allow, got %v", v.Kind)
	}
	if atomic.LoadInt64(&eng.n) != 1 {
		t.Fatalf("the conforming call must dispatch exactly once (engine n=%d)", eng.n)
	}
	if c := k.Counters(); c.Denies != 1 {
		t.Fatalf("the held escalation must count as a deny, got %d", c.Denies)
	}
}
