package shipgate

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func shipCall(tool, claim string) *abi.ToolCall {
	c := &abi.ToolCall{Tool: tool}
	if claim != "" {
		c.Meta = map[string]string{"witness": claim}
	}
	return c
}

// A ship-shaped call is lifted to RequireWitness, carrying the caller's claimed
// effect for the kernel's witness fold to corroborate.
func TestShipGateRequiresWitnessForShipTools(t *testing.T) {
	ctx := context.Background()
	for _, tool := range []string{"ship", "release", "ship_release", "publish", "deploy"} {
		v := DefaultAdjudicator.Adjudicate(ctx, shipCall(tool, "ancestor:HEAD"))
		if v.Kind != abi.VerdictRequireWitness {
			t.Fatalf("%s: got %v, want RequireWitness", tool, v.Kind)
		}
		wp, ok := v.Payload.(abi.WitnessPayload)
		if !ok || wp.Claim != "ancestor:HEAD" {
			t.Fatalf("%s: claim not threaded into the verdict, got %+v", tool, v.Payload)
		}
		if v.Meta["tool"] != tool {
			t.Fatalf("%s: meta tool = %q", tool, v.Meta["tool"])
		}
	}
}

// A non-ship call gets no opinion from the gate (Defer).
func TestShipGateDefersNonShip(t *testing.T) {
	ctx := context.Background()
	for _, tool := range []string{"read_file", "git_status", "get_user", "write_file"} {
		if v := DefaultAdjudicator.Adjudicate(ctx, shipCall(tool, "")); v.Kind != abi.VerdictDefer {
			t.Fatalf("%s: got %v, want Defer", tool, v.Kind)
		}
	}
}

// A claim-less ship still fires RequireWitness (with an empty claim) — so the kernel
// fail-closes it to UNWITNESSED rather than silently letting an unproven ship through.
func TestShipGateClaimlessStillRequiresWitness(t *testing.T) {
	v := DefaultAdjudicator.Adjudicate(context.Background(), shipCall("ship_release", ""))
	if v.Kind != abi.VerdictRequireWitness {
		t.Fatalf("claimless ship: got %v, want RequireWitness", v.Kind)
	}
	if wp, ok := v.Payload.(abi.WitnessPayload); !ok || wp.Claim != "" {
		t.Fatalf("claimless ship: expected empty claim, got %+v", v.Payload)
	}
}

func TestShipGateNilCallDefers(t *testing.T) {
	if v := DefaultAdjudicator.Adjudicate(context.Background(), nil); v.Kind != abi.VerdictDefer {
		t.Fatalf("nil call: got %v, want Defer", v.Kind)
	}
}
