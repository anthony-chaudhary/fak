package vdso

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

type toggleAdjudicator struct {
	deny atomic.Bool
}

func (a *toggleAdjudicator) Adjudicate(_ context.Context, _ *abi.ToolCall) abi.Verdict {
	if a.deny.Load() {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "monitor"}
	}
	return abi.Verdict{Kind: abi.VerdictAllow, By: "monitor"}
}

func (a *toggleAdjudicator) Caps() []abi.Capability { return nil }

func TestSubmitFastPathIsolationAndReload(t *testing.T) {
	EnsureRegistered()
	ctx := context.Background()

	// 1. Two kernels in the same process with different adjudicators must enforce
	// their own policies independently even for vDSO-registered tier-1 ("calculate")
	// and tier-3 ("list_all_airports") tools (#14514).
	allowAdj := &toggleAdjudicator{}
	denyAdj := &toggleAdjudicator{}
	denyAdj.deny.Store(true)

	kAllow := kernel.New("", kernel.WithAdjudicators([]abi.Adjudicator{allowAdj}))
	kDeny := kernel.New("", kernel.WithAdjudicators([]abi.Adjudicator{denyAdj}))

	calcCall := func() *abi.ToolCall {
		return &abi.ToolCall{
			Tool: "calculate",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"a":2,"b":3}`), Len: 13},
			Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
		}
	}
	airportCall := func() *abi.ToolCall {
		return &abi.ToolCall{
			Tool: "list_all_airports",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`), Len: 2},
		}
	}

	// Allowing kernel hits vDSO for both tier-1 and tier-3.
	if _, v := kAllow.Syscall(ctx, calcCall()); v.Kind != abi.VerdictAllow || v.By != "vdso" {
		t.Fatalf("kAllow calculate: got %+v, want Allow/vdso", v)
	}
	if _, v := kAllow.Syscall(ctx, airportCall()); v.Kind != abi.VerdictAllow || v.By != "vdso" {
		t.Fatalf("kAllow list_all_airports: got %+v, want Allow/vdso", v)
	}

	// Denying kernel in the same process MUST deny both despite global vDSO fast paths.
	if _, v := kDeny.Syscall(ctx, calcCall()); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("kDeny calculate: got %+v, want Deny/POLICY_BLOCK", v)
	}
	if _, v := kDeny.Syscall(ctx, airportCall()); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("kDeny list_all_airports: got %+v, want Deny/POLICY_BLOCK", v)
	}
	if kDeny.Counters().VDSOHits != 0 {
		t.Fatalf("kDeny VDSOHits = %d, want 0", kDeny.Counters().VDSOHits)
	}

	// 2. Reloading kAllow's adjudicator to deny previously-allowed vDSO tools
	// must immediately return VerdictDeny without serving from vDSO.
	allowAdj.deny.Store(true)
	if _, v := kAllow.Syscall(ctx, calcCall()); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("kAllow after policy reload (calculate): got %+v, want Deny/POLICY_BLOCK", v)
	}
	if _, v := kAllow.Syscall(ctx, airportCall()); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("kAllow after policy reload (list_all_airports): got %+v, want Deny/POLICY_BLOCK", v)
	}
}
