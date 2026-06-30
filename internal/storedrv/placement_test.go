package storedrv

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// TestPutPlacedDemotesUnderHotPressure is the MLCACHE5 (#1471) witness: under high
// hot-tier pressure a payload that the SIZE heuristic (and PutHinted's durability
// routing) would keep hot instead routes to the durable tier when the caller supplies a
// cachemeta placement verdict. The refute condition from the spine doc — that this only
// duplicates PutHinted's size/durability routing — is killed by the second half: the
// SAME small, non-durable payload+hint stays hot under a no-pressure verdict, so the
// demote above was produced by PRESSURE (via the real cachemeta.PlanPlacement policy),
// a route neither size nor durability could have produced.
func TestPutPlacedDemotesUnderHotPressure(t *testing.T) {
	ctx := context.Background()

	// A small, non-durable, agent-scoped payload: BOTH the size heuristic (< threshold ->
	// hot) AND PutHinted's durability routing (turn-durability, agent scope, untainted ->
	// hot) would keep this hot. Nothing about the bytes or the hint asks for durable.
	small := payload(1000, 'h')
	hint := Hint{Durability: "turn", Scope: abi.ScopeAgent, Taint: abi.TaintTainted}

	profiles := cachemeta.DefaultTierProfiles()

	// (1) HOT-TIER FULL: the placement policy, given live HBM pressure 1.0 and a colder
	// DRAM tier with room, DEMOTES — the canonical demote-not-evict verdict the epic
	// centers on. Derive it from the real policy so the witness proves the router honors
	// cachemeta, not a hand-built decision.
	demote := cachemeta.PlanPlacement(cachemeta.PlacementRequest{
		Lifecycle:            cachemeta.Lifecycle{State: cachemeta.StateResident, Tier: cachemeta.TierHBM},
		SizeBytes:            int64(len(small)),
		Tokens:               100_000, // an expensive-to-rebuild span...
		PerTokenPrefillNanos: 100_000, // ...so retain (demote) beats recompute
		Profiles:             profiles,
		Pressure:             cachemeta.TierPressure{cachemeta.TierHBM: 1.0},
	})
	if demote.Action != cachemeta.ActionDemote {
		t.Fatalf("precondition: expected the policy to DEMOTE under hot pressure, got %q (%s)", demote.Action, demote.Reason)
	}

	r, mem, disk := memDiskRouter(t, false)
	if _, err := r.PutPlaced(ctx, small, hint, demote); err != nil {
		t.Fatalf("PutPlaced under pressure: %v", err)
	}
	if c, _, _ := disk.Resident(); c != 1 {
		t.Fatalf("pressure-aware placement did not route to the durable tier: disk=%d", c)
	}
	if mem.Len() != 0 {
		t.Fatalf("payload stayed hot despite hot-tier pressure: mem.Len=%d", mem.Len())
	}

	// (2) REFUTE GUARD — HOT TIER HAS ROOM: the same payload+hint with a no-pressure
	// verdict (ActionKeep) stays hot. This is the route the size heuristic alone produces;
	// proving it diverges from (1) shows PutPlaced added genuine tier-awareness, not a
	// duplicate of PutHinted's size/durability routing.
	keep := cachemeta.PlanPlacement(cachemeta.PlacementRequest{
		Lifecycle: cachemeta.Lifecycle{State: cachemeta.StateResident, Tier: cachemeta.TierHBM},
		SizeBytes: int64(len(small)),
		Profiles:  profiles,
		Pressure:  cachemeta.TierPressure{}, // hot tier has room
	})
	if keep.Action != cachemeta.ActionKeep {
		t.Fatalf("precondition: expected the policy to KEEP with room, got %q (%s)", keep.Action, keep.Reason)
	}

	r2, mem2, disk2 := memDiskRouter(t, false)
	if _, err := r2.PutPlaced(ctx, small, hint, keep); err != nil {
		t.Fatalf("PutPlaced with room: %v", err)
	}
	if mem2.Len() != 1 {
		t.Fatalf("no-pressure placement should keep the payload hot: mem.Len=%d", mem2.Len())
	}
	if c, _, _ := disk2.Resident(); c != 0 {
		t.Fatalf("no-pressure placement leaked to the durable tier: disk=%d", c)
	}
}

// TestPutPlacedZeroVerdictMatchesPutHinted proves PutPlaced is a safe superset of
// PutHinted: a zero PlacementDecision (a caller that supplied no real verdict) routes
// exactly like the size/durability path, so opting in never changes behavior until a
// real pressure-aware decision is handed in.
func TestPutPlacedZeroVerdictMatchesPutHinted(t *testing.T) {
	ctx := context.Background()
	small := payload(1000, 'z') // < threshold -> hot by size

	r, mem, disk := memDiskRouter(t, false)
	if _, err := r.PutPlaced(ctx, small, Hint{}, cachemeta.PlacementDecision{}); err != nil {
		t.Fatalf("PutPlaced zero verdict: %v", err)
	}
	if mem.Len() != 1 {
		t.Fatalf("zero verdict should defer to the size route (hot): mem.Len=%d", mem.Len())
	}
	if c, _, _ := disk.Resident(); c != 0 {
		t.Fatalf("zero verdict leaked to durable: disk=%d", c)
	}

	// A quarantined hint still routes durable through the size/durability fallback, exactly
	// as PutHinted would — the zero verdict does not suppress the existing routing.
	r2, mem2, disk2 := memDiskRouter(t, false)
	if _, err := r2.PutPlaced(ctx, small, Hint{Taint: abi.TaintQuarantined}, cachemeta.PlacementDecision{}); err != nil {
		t.Fatalf("PutPlaced quarantined zero verdict: %v", err)
	}
	if c, _, _ := disk2.Resident(); c != 1 {
		t.Fatalf("quarantined hint should still route durable via the fallback: disk=%d", c)
	}
	if mem2.Len() != 0 {
		t.Fatalf("quarantined payload leaked to volatile hot tier: mem.Len=%d", mem2.Len())
	}
}
