package computeadmit

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

// allowTail is the permissive policy tail a real chain ends in; the gate itself
// only ever denies or defers (an admission gate is not an allow-source).
type allowTail struct{}

func (allowTail) Adjudicate(context.Context, *abi.ToolCall) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictAllow, By: "test-allow"}
}
func (allowTail) Caps() []abi.Capability { return nil }

// Acceptance (#3269): two co-targeted ensemble members are serialized AT THE
// SUBMIT SEAM with a COLLISION_RISK, not silently co-scheduled — and the second
// admits once the first releases (a decision, not a scheduler).
func TestSubmitSerializesCoTargetedEnsembleMembers(t *testing.T) {
	gate := NewSubmitAdmitter(Taxonomy{})
	// modelroute writes ToolCall.Engine pre-Submit; the wiring layer binds each
	// engine route to the device region it occupies.
	device0 := dispatchorder.ComputeClaim{Class: ClassDevice, Range: "0"}
	gate.BindRoute("ensemble-m1", device0)
	gate.BindRoute("ensemble-m2", device0)

	k := kernel.New("test-engine", kernel.WithAdjudicators([]abi.Adjudicator{gate, allowTail{}}))
	ctx := context.Background()

	_, v1 := k.Submit(ctx, &abi.ToolCall{Tool: "generate", Engine: "ensemble-m1", TraceID: "member-1"})
	if v1.Kind != abi.VerdictAllow {
		t.Fatalf("first member verdict = %v, want allow", v1.Kind)
	}

	_, v2 := k.Submit(ctx, &abi.ToolCall{Tool: "generate", Engine: "ensemble-m2", TraceID: "member-2"})
	if v2.Kind != abi.VerdictDeny {
		t.Fatalf("co-targeted second member verdict = %v, want deny", v2.Kind)
	}
	if v2.Reason != ReasonComputeCollision {
		t.Fatalf("deny reason code = %d, want %d", v2.Reason, ReasonComputeCollision)
	}
	if name := abi.ReasonName(v2.Reason); name != "COLLISION_RISK" {
		t.Fatalf("deny reason name = %q, want COLLISION_RISK", name)
	}
	if v2.Meta["rung"] != RungRegionCollision || v2.Meta["conflict"] != "member-1" {
		t.Fatalf("deny meta = %v, want rung=%s conflict=member-1", v2.Meta, RungRegionCollision)
	}

	// A re-submit by the SAME holder is a renewal, never self-serialized.
	if _, v := k.Submit(ctx, &abi.ToolCall{Tool: "generate", Engine: "ensemble-m1", TraceID: "member-1"}); v.Kind != abi.VerdictAllow {
		t.Fatalf("same-holder renewal verdict = %v, want allow", v.Kind)
	}

	// Release frees the region and the serialized member admits — decision-only
	// serialization, exactly like two agents on one file tree.
	gate.Release("member-1")
	if _, v := k.Submit(ctx, &abi.ToolCall{Tool: "generate", Engine: "ensemble-m2", TraceID: "member-2"}); v.Kind != abi.VerdictAllow {
		t.Fatalf("post-release verdict = %v (reason %s), want allow", v.Kind, abi.ReasonName(v.Reason))
	}
	if live := gate.Live(); len(live) != 1 || live[0].ID != "member-2" {
		t.Fatalf("live leases after handoff = %+v, want [member-2]", live)
	}
}

// A claim-less call is untouched (the gate defers), keeping every existing
// chain's verdicts byte-identical — the additive-no-regression guarantee.
func TestSubmitAdmitterIgnoresClaimlessCalls(t *testing.T) {
	gate := NewSubmitAdmitter(Taxonomy{})
	k := kernel.New("test-engine", kernel.WithAdjudicators([]abi.Adjudicator{gate, allowTail{}}))

	if _, v := k.Submit(context.Background(), &abi.ToolCall{Tool: "read", TraceID: "plain"}); v.Kind != abi.VerdictAllow {
		t.Fatalf("claim-less call verdict = %v, want allow via chain tail", v.Kind)
	}
	if live := gate.Live(); len(live) != 0 {
		t.Fatalf("claim-less call recorded a lease: %+v", live)
	}
}

// Meta-declared claims ride the same pricing as engine-bound ones, and an
// out-of-taxonomy claim refuses POLICY_BLOCK at the seam.
func TestSubmitAdmitterMetaClaimAndTaxonomy(t *testing.T) {
	gate := NewSubmitAdmitter(DecodeTaxonomy(4))
	k := kernel.New("test-engine", kernel.WithAdjudicators([]abi.Adjudicator{gate, allowTail{}}))
	ctx := context.Background()

	_, v1 := k.Submit(ctx, &abi.ToolCall{Tool: "generate", TraceID: "d-1",
		Meta: map[string]string{MetaComputeClass: ClassPhase, MetaComputeRange: "1"}})
	if v1.Kind != abi.VerdictAllow {
		t.Fatalf("meta-claimed worker 1 verdict = %v, want allow", v1.Kind)
	}
	_, v2 := k.Submit(ctx, &abi.ToolCall{Tool: "generate", TraceID: "d-2",
		Meta: map[string]string{MetaComputeClass: ClassPhase, MetaComputeRange: "1"}})
	if v2.Kind != abi.VerdictDeny || v2.Reason != ReasonComputeCollision {
		t.Fatalf("co-targeted meta claim: kind=%v reason=%s, want deny COLLISION_RISK", v2.Kind, abi.ReasonName(v2.Reason))
	}
	_, v3 := k.Submit(ctx, &abi.ToolCall{Tool: "generate", TraceID: "d-3",
		Meta: map[string]string{MetaComputeClass: ClassPhase, MetaComputeRange: "9"}})
	if v3.Kind != abi.VerdictDeny || v3.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("out-of-pool claim: kind=%v reason=%s, want deny POLICY_BLOCK", v3.Kind, abi.ReasonName(v3.Reason))
	}
	if v3.Meta["rung"] != RungOutOfTaxonomy {
		t.Fatalf("out-of-pool rung = %q, want %q", v3.Meta["rung"], RungOutOfTaxonomy)
	}
}
