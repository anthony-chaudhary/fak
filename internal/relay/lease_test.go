package relay

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

// Issue #1909 (rung J7) done condition: "A test asserts the relay re-acquires a
// free region and defers on a held one." These are that witness, exercised as a
// dos-arbitrate-level simulation — no real multi-process fleet, just the relay's
// leg-boundary Reacquire (lease.go, rung H2 #1895) over the lane arbiter with a
// simulated peer (run: `go test ./internal/relay -run LeaseConcurrency`).
//
// The baton is the one leg N hands to leg N+1: its ProgressCursor.HeldRegion is
// the lease region the successor must re-acquire before writing.

// relayTax is the dos.toml lane taxonomy the simulation arbitrates over. The
// relay lane is not yet declared in dos.toml, so the re-acquire is a TREE-ONLY
// request (geometry decides); a named lane would only narrow the same decision.
func relayTax() laneadmit.Taxonomy {
	return laneadmit.Taxonomy{
		Loaded:    true,
		Exclusive: map[string]bool{"release": true, "global": true, "abi": true, "dos": true},
		Trees: map[string][]string{
			"session": {"internal/session/**"},
			"gateway": {"internal/gateway/**"},
			"release": {"VERSION", "docs/releases/**"},
		},
	}
}

// handoffBaton is leg N's baton: the successor re-acquires its HeldRegion.
func handoffBaton() Baton {
	return Baton{
		Schema:  Schema,
		RelayID: "RLY-20260704-1909",
		Leg:     3,
		ProgressCursor: ProgressCursor{
			StartSHA:   "0123456789abcdef0123456789abcdef01234567",
			HeldRegion: []string{"internal/relay/**"},
		},
	}
}

// TestLeaseConcurrencyReacquiresFreeRegion asserts the free-region half of the
// done condition: with no peer holding an overlapping tree, leg N+1 re-acquires
// the region and is admitted to write.
func TestLeaseConcurrencyReacquiresFreeRegion(t *testing.T) {
	v := Reacquire(handoffBaton(), nil, relayTax())
	if !v.Admit {
		t.Fatalf("leg N+1 must re-acquire a free region, got refusal %+v", v)
	}
	if v.Reason != "" {
		t.Fatalf("an admitted re-acquire carries no reason, got %q", v.Reason)
	}
	if len(v.Conflicts) != 0 {
		t.Fatalf("a free region has no conflicts, got %+v", v.Conflicts)
	}
}

// TestLeaseConcurrencyDefersOnHeldRegion asserts the held-region half: a simulated
// peer holding an OVERLAPPING tree refuses the re-acquire with the arbiter's closed
// lane-conflict reason (COLLISION_RISK), naming the holder — so leg N+1 defers
// instead of colliding with the peer on the shared tree.
func TestLeaseConcurrencyDefersOnHeldRegion(t *testing.T) {
	peer := []laneadmit.Lease{{
		ID:     "loop-lane-relay",
		Lane:   "loop",
		Tree:   []string{"internal/relay/**"},
		Holder: "peer-worker-7",
	}}
	v := Reacquire(handoffBaton(), peer, relayTax())
	if v.Admit {
		t.Fatal("leg N+1 must DEFER when a peer holds an overlapping tree, got admit")
	}
	if v.Reason != laneadmit.ReasonCollisionRisk {
		t.Fatalf("refusal reason = %q, want the closed lane-conflict reason %q",
			v.Reason, laneadmit.ReasonCollisionRisk)
	}
	if len(v.Conflicts) == 0 || v.Conflicts[0].Holder != "peer-worker-7" {
		t.Fatalf("refusal must name the holding peer, got conflicts %+v", v.Conflicts)
	}
}

// TestLeaseConcurrencyOwnPriorLegNeverConflicts asserts the continuity property:
// the relay re-acquires under a lease id STABLE across legs (LeaseID(RelayID)), so
// the prior leg's own lease — even over the SAME overlapping tree — is the caller's
// own and never conflicts. Leg N+1 is a renew of the same lease, not a second one
// racing it; that is what "no collision across legs" means.
func TestLeaseConcurrencyOwnPriorLegNeverConflicts(t *testing.T) {
	b := handoffBaton()
	own := []laneadmit.Lease{{
		ID:     LeaseID(b.RelayID),
		Lane:   "loop",
		Tree:   []string{"internal/relay/**"},
		Holder: b.RelayID,
	}}
	v := Reacquire(b, own, relayTax())
	if !v.Admit {
		t.Fatalf("the relay's own prior-leg lease must not conflict (continuity), got %+v", v)
	}
}
