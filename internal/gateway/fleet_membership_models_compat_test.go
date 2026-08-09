package gateway

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"testing"
)

// Issue #5635, backward-compatibility half. Adding WorkerSpec.Models rewrote the
// one placement primitive every un-modeled caller already goes through, so
// "empty Models keeps today's behavior" is not a claim a hand-written expectation
// can settle — a rotation that drifts only after a particular drain/health
// sequence would pass every fixed-sequence test in the package.
//
// The reference below is the VERBATIM HEAD algorithm (internal/gateway
// fleet_membership.go at 3a1b7320f0, funcs admissibleIDsLocked + pickExcept),
// copied unchanged. The differential then drives the same randomized stream of
// probe flips, drains and re-registrations through both and compares the served
// worker AND the round-robin cursor at every single step — a cursor that diverges
// silently is exactly the failure a spot check misses.

// headAdmissibleIDsLocked is HEAD's admissibleIDsLocked, verbatim.
func headAdmissibleIDsLocked(m *FleetMembership) []string {
	var ids []string
	for _, id := range m.order {
		if w := m.workers[id]; w != nil && w.admissible() {
			ids = append(ids, id)
		}
	}
	return ids
}

// headPickExcept is HEAD's pickExcept, verbatim (receiver lowered to a parameter
// so it can live beside the current method without colliding with it).
func headPickExcept(m *FleetMembership, skip map[string]struct{}) (WorkerSpec, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	adm := headAdmissibleIDsLocked(m)
	n := uint64(len(adm))
	if n == 0 {
		return WorkerSpec{}, false
	}
	for i := uint64(0); i < n; i++ {
		id := adm[int((m.rr+i)%n)]
		if _, done := skip[id]; done {
			continue
		}
		m.rr += i + 1
		return m.workers[id].spec, true
	}
	return WorkerSpec{}, false
}

// headDispatch is HEAD's Dispatch, verbatim, driven by headPickExcept — the
// failover loop the un-modeled callers used before DispatchForModel existed.
func headDispatch(ctx context.Context, m *FleetMembership, send func(ctx context.Context, spec WorkerSpec) error) (WorkerSpec, error) {
	tried := make(map[string]struct{})
	var lastErr error
	for {
		spec, ok := headPickExcept(m, tried)
		if !ok {
			if lastErr != nil {
				return WorkerSpec{}, fmt.Errorf("%w: every admissible worker failed: %w", ErrNoHealthyWorker, lastErr)
			}
			return WorkerSpec{}, ErrNoHealthyWorker
		}
		if err := m.Acquire(spec.ID); err != nil {
			// Worker went non-admissible between pick and acquire — skip it.
			tried[spec.ID] = struct{}{}
			continue
		}
		err := send(ctx, spec)
		m.Release(spec.ID)
		if err == nil {
			return spec, nil
		}
		lastErr = err
		tried[spec.ID] = struct{}{}
		m.markDispatchFailure(spec.ID)
	}
}

// errSendFailedForCompat is the single deterministic send failure both sides of
// the differential return, so the wrapped error text is comparable.
var errSendFailedForCompat = errors.New("compat: upstream send failed")

// cursorOf reads the round-robin cursor under the registry lock.
func cursorOf(m *FleetMembership) uint64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.rr
}

// TestFleetMembershipUnlabeledPlacementMatchesHeadExactly is the load-bearing
// compatibility proof: over randomized op streams, an unlabeled roster placed by
// the NEW model-aware primitive is indistinguishable from the same roster placed
// by HEAD's primitive — same worker, same cursor, every step. It also proves the
// three un-modeled entry points (Pick, PickForModel(""), and PickForModel of an
// arbitrary model over an unlabeled fleet) are the same function.
func TestFleetMembershipUnlabeledPlacementMatchesHeadExactly(t *testing.T) {
	const steps = 3000
	ids := []string{"w1", "w2", "w3", "w4", "w5"}
	ctx := context.Background()

	for _, seed := range []int64{1, 7, 42, 1337, 20260806} {
		rng := rand.New(rand.NewSource(seed))
		spNew, spRef := newScriptedProbe(), newScriptedProbe()
		// Identical config; the ONLY difference between the two registries is
		// which placement primitive the test calls on them.
		mNew := NewFleetMembership(MembershipConfig{HealthyAfter: 2, UnhealthyAfter: 2, Probe: spNew.probe})
		mRef := NewFleetMembership(MembershipConfig{HealthyAfter: 2, UnhealthyAfter: 2, Probe: spRef.probe})
		for _, id := range ids {
			// No Models: the pre-labeling registration shape, verbatim.
			mustAdd(t, mNew, WorkerSpec{ID: id, Endpoint: id})
			mustAdd(t, mRef, WorkerSpec{ID: id, Endpoint: id})
		}

		for s := 0; s < steps; s++ {
			switch rng.Intn(10) {
			case 0, 1, 2: // flip one worker's probe verdict, then tick both loops
				id := ids[rng.Intn(len(ids))]
				ok := rng.Intn(2) == 0
				spNew.set(id, ok)
				spRef.set(id, ok)
				mNew.ProbeOnce(ctx)
				mRef.ProbeOnce(ctx)
			case 3: // drain (removes the worker once idle, mutating m.order)
				id := ids[rng.Intn(len(ids))]
				_ = mNew.Drain(id)
				_ = mRef.Drain(id)
			case 4: // re-register a previously drained worker
				id := ids[rng.Intn(len(ids))]
				_ = mNew.Add(WorkerSpec{ID: id, Endpoint: id})
				_ = mRef.Add(WorkerSpec{ID: id, Endpoint: id})
			default: // place a request through both and compare
				var (
					gotSpec WorkerSpec
					gotOK   bool
					via     string
				)
				switch rng.Intn(3) {
				case 0:
					gotSpec, gotOK = mNew.Pick()
					via = `Pick()`
				case 1:
					spec, err := mNew.PickForModel("")
					gotSpec, gotOK, via = spec, err == nil, `PickForModel("")`
				default:
					spec, err := mNew.PickForModel("a-model-nobody-labeled")
					gotSpec, gotOK, via = spec, err == nil, `PickForModel(unlabeled-fleet)`
				}
				wantSpec, wantOK := headPickExcept(mRef, nil)
				if gotOK != wantOK || gotSpec.ID != wantSpec.ID {
					t.Fatalf("seed %d step %d: %s = (%q, %v); HEAD pickExcept = (%q, %v)",
						seed, s, via, gotSpec.ID, gotOK, wantSpec.ID, wantOK)
				}
			}
			// The cursor is the piece that would drift silently: a mismatch here
			// changes a LATER pick, not this one.
			if rrNew, rrRef := cursorOf(mNew), cursorOf(mRef); rrNew != rrRef {
				t.Fatalf("seed %d step %d: round-robin cursor %d != HEAD cursor %d", seed, s, rrNew, rrRef)
			}
			// And the admissible view must agree, since the router reads it directly.
			gotAdm, wantAdm := admissibleIDs(mNew), admissibleIDs(mRef)
			if len(gotAdm) != len(wantAdm) {
				t.Fatalf("seed %d step %d: Admissible() = %v, HEAD = %v", seed, s, gotAdm, wantAdm)
			}
			for id := range wantAdm {
				if !gotAdm[id] {
					t.Fatalf("seed %d step %d: Admissible() = %v, HEAD = %v", seed, s, gotAdm, wantAdm)
				}
			}
		}
	}
}

// TestFleetMembershipUnlabeledDispatchMatchesHeadExactly is the same differential
// for the failover path: Dispatch (un-modeled) must visit the same workers in the
// same order and return the same typed verdict as HEAD's loop over pickExcept,
// including when every admissible worker fails and the error is wrapped.
func TestFleetMembershipUnlabeledDispatchMatchesHeadExactly(t *testing.T) {
	ctx := context.Background()
	ids := []string{"w1", "w2", "w3", "w4"}

	for _, seed := range []int64{3, 11, 99, 20260806} {
		rng := rand.New(rand.NewSource(seed))
		spNew, spRef := newScriptedProbe(), newScriptedProbe()
		mNew := NewFleetMembership(MembershipConfig{HealthyAfter: 1, UnhealthyAfter: 3, Probe: spNew.probe})
		mRef := NewFleetMembership(MembershipConfig{HealthyAfter: 1, UnhealthyAfter: 3, Probe: spRef.probe})
		for _, id := range ids {
			mustAdd(t, mNew, WorkerSpec{ID: id, Endpoint: id})
			mustAdd(t, mRef, WorkerSpec{ID: id, Endpoint: id})
		}

		for s := 0; s < 400; s++ {
			// Same health picture on both sides.
			for _, id := range ids {
				ok := rng.Intn(3) != 0
				spNew.set(id, ok)
				spRef.set(id, ok)
			}
			mNew.ProbeOnce(ctx)
			mRef.ProbeOnce(ctx)

			// The same deterministic send outcome per worker on both sides.
			failing := map[string]bool{}
			for _, id := range ids {
				failing[id] = rng.Intn(3) == 0
			}
			send := func(visited *[]string) func(context.Context, WorkerSpec) error {
				return func(_ context.Context, spec WorkerSpec) error {
					*visited = append(*visited, spec.ID)
					if failing[spec.ID] {
						return errSendFailedForCompat
					}
					return nil
				}
			}

			var gotVisited, wantVisited []string
			gotSpec, gotErr := mNew.Dispatch(ctx, send(&gotVisited))
			wantSpec, wantErr := headDispatch(ctx, mRef, send(&wantVisited))

			if (gotErr == nil) != (wantErr == nil) || gotSpec.ID != wantSpec.ID {
				t.Fatalf("seed %d step %d: Dispatch = (%q, %v); HEAD = (%q, %v)",
					seed, s, gotSpec.ID, gotErr, wantSpec.ID, wantErr)
			}
			if gotErr != nil && wantErr != nil && gotErr.Error() != wantErr.Error() {
				t.Fatalf("seed %d step %d: Dispatch err = %q; HEAD err = %q", seed, s, gotErr, wantErr)
			}
			if len(gotVisited) != len(wantVisited) {
				t.Fatalf("seed %d step %d: visited %v; HEAD visited %v", seed, s, gotVisited, wantVisited)
			}
			for i := range wantVisited {
				if gotVisited[i] != wantVisited[i] {
					t.Fatalf("seed %d step %d: visited %v; HEAD visited %v", seed, s, gotVisited, wantVisited)
				}
			}
			if rrNew, rrRef := cursorOf(mNew), cursorOf(mRef); rrNew != rrRef {
				t.Fatalf("seed %d step %d: cursor %d != HEAD cursor %d", seed, s, rrNew, rrRef)
			}
		}
	}
}
