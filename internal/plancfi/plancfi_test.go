package plancfi

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

func call(tool, trace string) *abi.ToolCall {
	return &abi.ToolCall{Tool: tool, TraceID: trace,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte("{}")}}
}

var airlinePlan = Plan{
	Tools: []string{"get_user_details", "search_flights", "read_refund_policy", "book_reservation"},
	Mode:  AllowedSet,
}

func TestNoPlanDefers(t *testing.T) {
	ctx := context.Background()
	a := New(NewLedger())
	if v := a.Adjudicate(ctx, call("anything", "t")); v.Kind != abi.VerdictDefer {
		t.Fatalf("no plan => Defer, got %v", v.Kind)
	}
}

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
	ok("a")
	ok("b")
	ok("a")
	dev("z")
	ok("c")
}

func TestSessionIsolation(t *testing.T) {
	ctx := context.Background()
	l := NewLedger()
	l.Declare("planned", airlinePlan)
	a := New(l)
	if v := a.Adjudicate(ctx, call("send_email", "free")); v.Kind != abi.VerdictDefer {
		t.Fatalf("an unplanned trace must Defer, got %v", v.Kind)
	}
	if v := a.Adjudicate(ctx, call("send_email", "planned")); v.Kind != VerdictRequireApproval {
		t.Fatalf("the planned trace must still escalate, got %v", v.Kind)
	}
}

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

func TestPlanCFIDeviationDisposition(t *testing.T) {
	ctx := context.Background()
	l := NewLedger()
	l.Declare("t", airlinePlan)
	a := New(l)
	c := call("send_email", "t")
	v := a.Adjudicate(ctx, c)

	if v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("deviation verdict reason = %v, want ReasonPolicyBlock (%v)", v.Reason, abi.ReasonPolicyBlock)
	}
	if v.Disposition != "ESCALATE" {
		t.Fatalf("deviation verdict disposition = %q, want ESCALATE", v.Disposition)
	}
	if v.Meta["disposition"] != "ESCALATE" {
		t.Fatalf("deviation verdict meta disposition = %q, want ESCALATE", v.Meta["disposition"])
	}

	res := kernel.DenyResult(c, v)
	if got := res.Meta["disposition"]; got != "ESCALATE" {
		t.Fatalf("DenyResult meta disposition = %q, want ESCALATE", got)
	}
	if got := res.Meta["reason"]; got != "POLICY_BLOCK" {
		t.Fatalf("DenyResult meta reason = %q, want POLICY_BLOCK", got)
	}
}
