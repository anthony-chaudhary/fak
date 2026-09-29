package gateway

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"testing"
)

// LB-16 (public-router hardening). The placement primitive used to be a mutable
// round-robin cursor (`FleetMembership.rr`), which made a selection a function of
// history: replay was impossible and the same registry state could pick different
// workers on different runs. The ticket replaces it with a keyed rendezvous
// (`hrwScore(key, memberID)`), which is pure. This file pins the properties the
// cursor could not promise:
//
//  1. Replay: two identical `(key, set)` selections are byte-identical, with no
//     counter state consulted.
//  2. Spread: distinct keys distribute across the declared set (a keyed hash, not
//     a fixed sentinel).
//  3. Failover is preserved: a persistently unhealthy worker is still avoided, and
//     all-failing still yields the typed ErrNoHealthyWorker (never a silent drop).
//
// The cursor itself is gone, so this replaces the old cursor-parity differential:
// the new algorithm is deliberately NOT round-robin, and the compatibility claim
// that matters is the TYPED VERDICT and the failover loop, not the cursor value.

// TestFleetMembershipKeyedSelectionIsReplayable is the LB-16 witness: over a
// randomized stream of probe flips, drains and re-registrations, the placement for
// a fixed key is identical across two independent registries at every step, and
// re-running the same key twice on ONE registry gives the same worker (no cursor
// to advance, so a repeated call is a no-op on state).
func TestFleetMembershipKeyedSelectionIsReplayable(t *testing.T) {
	const steps = 3000
	ids := []string{"w1", "w2", "w3", "w4", "w5"}
	ctx := context.Background()

	for _, seed := range []int64{1, 7, 42, 1337, 20260806} {
		rng := rand.New(rand.NewSource(seed))
		spA, spB := newScriptedProbe(), newScriptedProbe()
		mA := NewFleetMembership(MembershipConfig{HealthyAfter: 2, UnhealthyAfter: 2, Probe: spA.probe})
		mB := NewFleetMembership(MembershipConfig{HealthyAfter: 2, UnhealthyAfter: 2, Probe: spB.probe})
		for _, id := range ids {
			mustAdd(t, mA, WorkerSpec{ID: id, Endpoint: id})
			mustAdd(t, mB, WorkerSpec{ID: id, Endpoint: id})
		}

		for s := 0; s < steps; s++ {
			switch rng.Intn(10) {
			case 0, 1, 2: // flip one worker's probe verdict, then tick both loops
				id := ids[rng.Intn(len(ids))]
				ok := rng.Intn(2) == 0
				spA.set(id, ok)
				spB.set(id, ok)
				mA.ProbeOnce(ctx)
				mB.ProbeOnce(ctx)
			case 3: // drain (removes the worker once idle, mutating order)
				id := ids[rng.Intn(len(ids))]
				_ = mA.Drain(id)
				_ = mB.Drain(id)
			case 4: // re-register a previously drained worker
				id := ids[rng.Intn(len(ids))]
				_ = mA.Add(WorkerSpec{ID: id, Endpoint: id})
				_ = mB.Add(WorkerSpec{ID: id, Endpoint: id})
			default: // place a request through both and compare
				specA, okA := mA.Pick()
				specB, okB := mB.Pick()
				if okA != okB || specA.ID != specB.ID {
					t.Fatalf("seed %d step %d: Pick() = (%q, %v); mirror registry = (%q, %v) — placement is not replayable",
						seed, s, specA.ID, okA, specB.ID, okB)
				}
				if okA {
					// A second identical call on the SAME registry must not move
					// (there is no cursor to advance).
					again, okAgain := mA.Pick()
					if !okAgain || again.ID != specA.ID {
						t.Fatalf("seed %d step %d: repeated Pick() = (%q, %v), want (%q, true) — a cursor leaked back in",
							seed, s, again.ID, okAgain, specA.ID)
					}
				}
			}
		}
	}
}

// TestFleetMembershipKeyedSelectionSpreadsAndPreservesFailover proves the keyed
// algorithm still distributes distinct keys across the admissible set (so it is a
// working balancer, not a fixed sentinel) and that the failover loop and its typed
// verdicts are unchanged.
func TestFleetMembershipKeyedSelectionSpreadsAndPreservesFailover(t *testing.T) {
	ctx := context.Background()
	ids := []string{"w1", "w2", "w3", "w4"}

	t.Run("distinct keys spread across the set", func(t *testing.T) {
		m := NewFleetMembership(MembershipConfig{HealthyAfter: 1, Probe: func(context.Context, WorkerSpec) bool { return true }})
		for _, id := range ids {
			mustAdd(t, m, WorkerSpec{ID: id, Endpoint: id})
		}
		m.ProbeOnce(ctx)

		seen := map[string]int{}
		const keys = 4000
		for i := 0; i < keys; i++ {
			spec, err := m.pickKeyedForModel("", nil, fmt.Sprintf("request-%d", i))
			if err != nil {
				t.Fatalf("pickKeyedForModel(%d): %v", i, err)
			}
			seen[spec.ID]++
		}
		if len(seen) != len(ids) {
			t.Fatalf("keyed rendezvous used %d/%d workers: %v", len(seen), len(ids), seen)
		}
		// Loose proportionality: every worker lands within a wide band of the mean.
		mean := keys / len(ids)
		for id, n := range seen {
			if n < mean/3 || n > mean*3 {
				t.Fatalf("worker %s got %d of %d keys (mean %d), grossly disproportionate: %v", id, n, keys, mean, seen)
			}
		}
	})

	t.Run("unhealthy worker is avoided, all-fail is the typed verdict", func(t *testing.T) {
		sp := newScriptedProbe()
		m := NewFleetMembership(MembershipConfig{HealthyAfter: 1, UnhealthyAfter: 1, Probe: sp.probe})
		for _, id := range ids {
			mustAdd(t, m, WorkerSpec{ID: id, Endpoint: id})
		}
		m.ProbeOnce(ctx)

		// Drive w1 permanently unhealthy.
		sp.set("w1", false)
		m.ProbeOnce(ctx)

		for i := 0; i < 200; i++ {
			spec, err := m.pickKeyedForModel("", nil, fmt.Sprintf("k-%d", i))
			if err != nil {
				t.Fatalf("pickKeyedForModel: %v", err)
			}
			if spec.ID == "w1" {
				t.Fatalf("an unhealthy worker was selected at key %d", i)
			}
		}

		// Every admissible worker fails the send: the typed verdict, never a drop.
		var lastErr error
		_, err := m.Dispatch(ctx, func(_ context.Context, _ WorkerSpec) error {
			lastErr = errors.New("down")
			return lastErr
		})
		if !errors.Is(err, ErrNoHealthyWorker) {
			t.Fatalf("all-fail Dispatch err = %v, want ErrNoHealthyWorker", err)
		}
	})

	t.Run("empty registry keeps the typed verdict", func(t *testing.T) {
		empty := NewFleetMembership(MembershipConfig{})
		if _, err := empty.Dispatch(ctx, func(context.Context, WorkerSpec) error { return nil }); !errors.Is(err, ErrNoHealthyWorker) {
			t.Fatalf("empty Dispatch err = %v, want ErrNoHealthyWorker", err)
		}
		if _, ok := empty.Pick(); ok {
			t.Fatal("empty Pick() reported ok")
		}
	})
}
