package plancfi

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func call(tool, trace string) *abi.ToolCall {
	return &abi.ToolCall{Tool: tool, TraceID: trace,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte("{}")}}
}

// the airline-booking plan: the tools a legitimate booking task needs.
var airlinePlan = Plan{
	Tools: []string{"get_user_details", "search_flights", "read_refund_policy", "book_reservation"},
	Mode:  AllowedSet,
}

// TestNoPlanDefers — with no plan declared, CFI is inactive and Defers (it must not
// affect an unplanned flow).
func TestNoPlanDefers(t *testing.T) {
	ctx := context.Background()
	a := New(NewLedger())
	if v := a.Adjudicate(ctx, call("anything", "t")); v.Kind != abi.VerdictDefer {
		t.Fatalf("no plan => Defer, got %v", v.Kind)
	}
}

// TestConformingCallDefers — a call within the approved set Defers (CFI has no
// objection; the other gates decide).
func TestConformingCallDefers(t *testing.T) {
	ctx := context.Background()
	l := NewLedger()
	l.Declare("t", airlinePlan)
	a := New(l)
	for _, tool := range airlinePlan.Tools {
		if v := a.Adjudicate(ctx, call(tool, "t")); v.Kind != abi.VerdictDefer {
			t.Fatalf("planned tool %q must Defer, got %v", tool, v.Kind)
		}
	}
}

// TestDeviationEscalates is the headline: an injection-derailed call to a tool NOT
// in the approved plan (send_email — the exfil gadget) is trapped as a CFI
// violation and escalated for human approval.
func TestDeviationEscalates(t *testing.T) {
	ctx := context.Background()
	l := NewLedger()
	l.Declare("t", airlinePlan)
	a := New(l)
	v := a.Adjudicate(ctx, call("send_email", "t"))
	if v.Kind != VerdictRequireApproval {
		t.Fatalf("an unplanned tool must escalate (RequireApproval), got %v", v.Kind)
	}
	if v.Meta["plancfi"] != "deviation" || v.Meta["tool"] != "send_email" {
		t.Fatalf("verdict must name the deviation, got %v", v.Meta)
	}
	if wp, ok := v.Payload.(abi.WitnessPayload); !ok || wp.Claim == "" {
		t.Fatalf("deviation verdict must carry a descriptive claim, got %#v", v.Payload)
	}
}

// TestStrictModeDenies — OnDeviation=Deny turns a deviation into a hard block
// (no human in the loop), proving the escalate-vs-deny policy is a knob.
func TestStrictModeDenies(t *testing.T) {
	ctx := context.Background()
	l := NewLedger()
	l.Declare("t", airlinePlan)
	a := New(l)
	a.OnDeviation = abi.VerdictDeny
	if v := a.Adjudicate(ctx, call("delete_everything", "t")); v.Kind != abi.VerdictDeny {
		t.Fatalf("strict mode must Deny a deviation, got %v", v.Kind)
	}
}

// TestSequenceMode — calls must follow the plan order; a repeat or a prior step is
// fine, a jump ahead or an unlisted tool deviates.
func TestSequenceMode(t *testing.T) {
	ctx := context.Background()
	l := NewLedger()
	l.Declare("t", Plan{Tools: []string{"a", "b", "c"}, Mode: Sequence})
	a := New(l)
	ok := func(tool string) {
		if v := a.Adjudicate(ctx, call(tool, "t")); v.Kind != abi.VerdictDefer {
			t.Fatalf("%q should conform, got %v", tool, v.Kind)
		}
	}
	dev := func(tool string) {
		if v := a.Adjudicate(ctx, call(tool, "t")); v.Kind == abi.VerdictDefer {
			t.Fatalf("%q should deviate, got Defer", tool)
		}
	}
	ok("a")  // step 0
	ok("b")  // step 1
	ok("a")  // a prior step (re-read) is fine
	dev("z") // unlisted tool deviates
	ok("c")  // step 2 (next) is fine
}

// TestSessionIsolation — a plan on one trace does not constrain another.
func TestSessionIsolation(t *testing.T) {
	ctx := context.Background()
	l := NewLedger()
	l.Declare("planned", airlinePlan)
	a := New(l)
	// an unplanned trace is unconstrained.
	if v := a.Adjudicate(ctx, call("send_email", "free")); v.Kind != abi.VerdictDefer {
		t.Fatalf("an unplanned trace must Defer, got %v", v.Kind)
	}
	// the planned trace still traps the deviation.
	if v := a.Adjudicate(ctx, call("send_email", "planned")); v.Kind != VerdictRequireApproval {
		t.Fatalf("the planned trace must still escalate, got %v", v.Kind)
	}
}

// TestRequireApprovalRegistered — the open-range verdict is registered with the
// right fold rank + fail-closed fallback (so an unaware worker can't proceed past
// an approval gate).
func TestRequireApprovalRegistered(t *testing.T) {
	if got := abi.FoldRank(VerdictRequireApproval); got != requireApprovalFoldRank {
		t.Fatalf("RequireApproval fold rank = %d, want %d", got, requireApprovalFoldRank)
	}
	if abi.FoldRank(VerdictRequireApproval) <= abi.FoldRank(abi.VerdictQuarantine) {
		t.Fatal("RequireApproval must outrank Quarantine")
	}
	if abi.FoldRank(VerdictRequireApproval) >= abi.FoldRank(abi.VerdictDeny) {
		t.Fatal("RequireApproval must be LESS restrictive than a hard Deny")
	}
	if abi.Fallback(VerdictRequireApproval) != abi.FallbackDeny {
		t.Fatal("RequireApproval must fall back to Deny (fail-closed)")
	}
}
