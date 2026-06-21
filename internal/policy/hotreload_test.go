package policy

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// TestHotReloadSwapsLiveFloorWithoutRestart is the issue #493 leg 1 witness: a
// policy reload — the exact manifest -> Parse -> SetPolicy path the host CLI's
// reloadPolicy and the /v1/fak/policy/reload route run — swaps the adjudicator's
// live capability floor on the SAME instance, with no rebuild and no restart.
//
// The issue's claim is that SetPolicy (decide.go) is RW-mutex-safe and makes the
// atomic floor swap trivial, but nothing in production exercised it: the floor was
// installed once at serve boot, so rotating it meant a rolling restart. This test
// proves the swap is live: before the reload the SAME instance denies a tool;
// after SetPolicy it allows it; rotating back denies it again — all on one
// adjudicator pointer, never a fresh process.
func TestHotReloadSwapsLiveFloorWithoutRestart(t *testing.T) {
	ctx := context.Background()
	call := &abi.ToolCall{Tool: "search_flights", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)}}

	// Floor A: nothing affirmatively allows search_flights -> DEFAULT_DENY.
	floorA, err := Parse([]byte(`{"version":"fak-policy/v1","posture":"fail_closed"}`))
	if err != nil {
		t.Fatalf("parse floor A: %v", err)
	}
	// Floor B: the rotated floor that admits search_flights.
	floorB, err := Parse([]byte(`{"version":"fak-policy/v1","allow":["search_flights"]}`))
	if err != nil {
		t.Fatalf("parse floor B: %v", err)
	}

	adj := adjudicator.New(floorA)
	if v := adj.Adjudicate(ctx, call); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("under floor A: got %v/%s, want Deny/DEFAULT_DENY", v.Kind, abi.ReasonName(v.Reason))
	}

	// Hot reload: swap the floor on the LIVE instance — the call reloadPolicy makes.
	adj.SetPolicy(floorB)
	if v := adj.Adjudicate(ctx, call); v.Kind != abi.VerdictAllow {
		t.Fatalf("after hot reload: got %v/%s, want Allow (live floor swapped, no restart)", v.Kind, abi.ReasonName(v.Reason))
	}

	// The swap is reversible on the same instance — rotate back to the deny floor.
	adj.SetPolicy(floorA)
	if v := adj.Adjudicate(ctx, call); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("after rotating back: got %v/%s, want Deny/DEFAULT_DENY", v.Kind, abi.ReasonName(v.Reason))
	}
}

// TestParseRoundTripsAdmitAndLogPosture is a focused leg-4 reload witness: a
// reloaded manifest can select the opt-in admit-and-log posture, and a fresh
// adjudicator built from that parsed floor admits a benign read-shaped call that
// the fail-closed floor would default-deny — so an operator can rotate a batch
// run into the fail-open posture through the same Parse path a reload uses,
// without recompiling. (The posture decision itself is witnessed end-to-end in
// adjudicator.TestAdmitAndLogPostureAllowsOnlyReadShapedDefaultDeny; this pins
// the manifest -> live-floor seam the reload route depends on.)
func TestParseRoundTripsAdmitAndLogPosture(t *testing.T) {
	ctx := context.Background()
	read := &abi.ToolCall{Tool: "read_dashboard", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)}}

	failClosed, err := Parse([]byte(`{"version":"fak-policy/v1","posture":"fail_closed"}`))
	if err != nil {
		t.Fatalf("parse fail_closed: %v", err)
	}
	admitLog, err := Parse([]byte(`{"version":"fak-policy/v1","posture":"admit_and_log"}`))
	if err != nil {
		t.Fatalf("parse admit_and_log: %v", err)
	}

	adj := adjudicator.New(failClosed)
	if v := adj.Adjudicate(ctx, read); v.Kind != abi.VerdictDeny {
		t.Fatalf("fail_closed floor must deny an unallowed read, got %v", v.Kind)
	}

	// Reload into the opt-in fail-open posture: the benign read now completes.
	adj.SetPolicy(admitLog)
	v := adj.Adjudicate(ctx, read)
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("admit_and_log floor must admit a benign read, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
	if v.Meta["posture"] != "admit_and_log" || v.Meta["would_deny"] != "DEFAULT_DENY" {
		t.Fatalf("admitted read must carry forensic posture metadata, got %v", v.Meta)
	}
}
