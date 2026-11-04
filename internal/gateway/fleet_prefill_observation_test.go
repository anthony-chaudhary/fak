package gateway

import (
	"context"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"sync"
	"testing"
)

func observedTestFleet(t *testing.T) *FleetMembership {
	t.Helper()
	m := NewFleetMembership(MembershipConfig{Probe: func(context.Context, WorkerSpec) bool { return true }, HealthyAfter: 1})
	for _, id := range []string{"a", "b"} {
		if err := m.Add(WorkerSpec{ID: id, Engine: EngineNative}); err != nil {
			t.Fatal(err)
		}
	}
	m.ProbeOnce(context.Background())
	return m
}
func observedTestReserve(t *testing.T, m *FleetMembership, id string) *fleetReservation {
	t.Helper()
	r, err := m.reserveForModel("", EngineNative, nil, nil, func(statuses []WorkerStatus) (fleetReservationPickResult, bool) {
		for _, s := range statuses {
			if s.Spec.ID == id {
				return fleetReservationPickResult{workerID: id, bookedOutputBlocks: 3}, true
			}
		}
		return fleetReservationPickResult{}, false
	})
	if err != nil {
		t.Fatal(err)
	}
	return r
}
func observedTestStatus(t *testing.T, m *FleetMembership, id string) WorkerStatus {
	t.Helper()
	for _, s := range m.Snapshot() {
		if s.Spec.ID == id {
			return s
		}
	}
	t.Fatal("missing worker", id)
	return WorkerStatus{}
}

// fak-test:runtime fast est=10ms lane=default
func TestObservedPrefillLifecycleAndDecodeIndependence(t *testing.T) {
	m := observedTestFleet(t)
	r := observedTestReserve(t, m, "a")
	defer r.Release()
	for _, completed := range []bool{true, false} {
		finish := r.startObservedPrefill(r.worker, r.prefillEpoch, 4000, 3900)
		s := observedTestStatus(t, m, "a")
		if s.ObservedPrefillTokens != 100 || s.ObservedPrefillAttempts != 1 || s.Inflight != 1 || s.BookedOutputBlocks != 3 {
			t.Fatal(s)
		}
		finish(completed)
		finish(!completed)
		s = observedTestStatus(t, m, "a")
		if s.ObservedPrefillTokens != 0 || s.ObservedPrefillAttempts != 0 || s.Inflight != 1 || s.BookedOutputBlocks != 3 {
			t.Fatal(s)
		}
	}
	z := r.startObservedPrefill(r.worker, r.prefillEpoch, 12, 12)
	s := observedTestStatus(t, m, "a")
	if s.ObservedPrefillTokens != 0 || s.ObservedPrefillAttempts != 1 {
		t.Fatal(s)
	}
	z(true)
	for _, v := range [][2]int{{-1, 0}, {12, -1}, {12, 13}} {
		if r.startObservedPrefill(r.worker, r.prefillEpoch, v[0], v[1]) != nil {
			t.Fatal("invalid count accepted")
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestObservedPrefillRetargetAndLateCompletion(t *testing.T) {
	m := observedTestFleet(t)
	r := observedTestReserve(t, m, "a")
	old := r.worker
	finishOld := r.startObservedPrefill(old, r.prefillEpoch, 100, 0)
	_, err := r.Retarget("", nil, map[string]struct{}{"a": {}}, func([]WorkerStatus) (fleetReservationPickResult, bool) {
		return fleetReservationPickResult{workerID: "b", bookedOutputBlocks: 3}, true
	})
	if err != nil {
		t.Fatal(err)
	}
	if r.startObservedPrefill(old, r.prefillEpoch, 100, 0) != nil {
		t.Fatal("stale start accepted")
	}
	finishNew := r.startObservedPrefill(r.worker, r.prefillEpoch, 200, 50)
	finishOld(false)
	finishOld(false)
	a, b := observedTestStatus(t, m, "a"), observedTestStatus(t, m, "b")
	if a.ObservedPrefillTokens != 0 || a.Inflight != 0 || b.ObservedPrefillTokens != 150 || b.Inflight != 1 {
		t.Fatal(a, b)
	}
	finishNew(true)
	r.Release()
	if s := observedTestStatus(t, m, "b"); s.ObservedPrefillTokens != 0 || s.Inflight != 0 {
		t.Fatal(s)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestObservedPrefillReplacementAndReleasedReservation(t *testing.T) {
	m := observedTestFleet(t)
	r := observedTestReserve(t, m, "a")
	finishOld := r.startObservedPrefill(r.worker, r.prefillEpoch, 100, 0)
	if err := m.Remove("a"); err != nil {
		t.Fatal(err)
	}
	if err := m.Add(WorkerSpec{ID: "a", Engine: EngineNative}); err != nil {
		t.Fatal(err)
	}
	m.ProbeOnce(context.Background())
	replacement := observedTestReserve(t, m, "a")
	finishNew := replacement.startObservedPrefill(replacement.worker, replacement.prefillEpoch, 200, 0)
	finishOld(false)
	r.Release()
	if s := observedTestStatus(t, m, "a"); s.ObservedPrefillTokens != 200 || s.Inflight != 1 {
		t.Fatal(s)
	}
	replacement.Release()
	if replacement.startObservedPrefill(replacement.worker, replacement.prefillEpoch, 50, 0) != nil {
		t.Fatal("released start accepted")
	}
	// Releasing occupancy/cancelling a request cannot assert physical work stopped.
	if s := observedTestStatus(t, m, "a"); s.ObservedPrefillTokens != 200 || s.Inflight != 0 {
		t.Fatal(s)
	}
	finishNew(false)
	if s := observedTestStatus(t, m, "a"); s.ObservedPrefillTokens != 0 {
		t.Fatal(s)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestObservedPrefillSharedConcurrentAndOverflow(t *testing.T) {
	m := observedTestFleet(t)
	r1, r2 := observedTestReserve(t, m, "a"), observedTestReserve(t, m, "a")
	defer r1.Release()
	defer r2.Release()
	const n = 32
	ready := make(chan func(bool), n)
	release := make(chan struct{})
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		r := r1
		if i%2 != 0 {
			r = r2
		}
		wg.Add(1)
		go func(r *fleetReservation) {
			defer wg.Done()
			f := r.startObservedPrefill(r.worker, r.prefillEpoch, 100, 50)
			ready <- f
			<-release
			f(true)
			f(false)
		}(r)
	}
	for i := 0; i < n; i++ {
		if <-ready == nil {
			t.Fatal("missing charge")
		}
	}
	if s := observedTestStatus(t, m, "a"); s.ObservedPrefillTokens != n*50 || s.ObservedPrefillAttempts != n || s.Inflight != 2 {
		t.Fatal(s)
	}
	close(release)
	wg.Wait()
	max := int(^uint(0) >> 1)
	large := r1.startObservedPrefill(r1.worker, r1.prefillEpoch, max, 0)
	small := r2.startObservedPrefill(r2.worker, r2.prefillEpoch, 1, 0)
	if s := observedTestStatus(t, m, "a"); s.ObservedPrefillTokens != max || !s.ObservedPrefillOverflow {
		t.Fatal(s)
	}
	large(true)
	if s := observedTestStatus(t, m, "a"); s.ObservedPrefillTokens != 1 || s.ObservedPrefillOverflow {
		t.Fatal(s)
	}
	small(false)
}

// fak-test:runtime fast est=10ms lane=default
func TestObservedPrefillHedgeChargesPhysicalWorkers(t *testing.T) {
	m := observedTestFleet(t)
	a, b := observedTestReserve(t, m, "a"), observedTestReserve(t, m, "b")
	defer a.Release()
	defer b.Release()
	fa, fb := a.startObservedPrefill(a.worker, a.prefillEpoch, 100, 0), b.startObservedPrefill(b.worker, b.prefillEpoch, 100, 20)
	if sa, sb := observedTestStatus(t, m, "a"), observedTestStatus(t, m, "b"); sa.ObservedPrefillTokens != 100 || sb.ObservedPrefillTokens != 80 {
		t.Fatal(sa, sb)
	}
	fa(true)
	if s := observedTestStatus(t, m, "b"); s.ObservedPrefillTokens != 80 {
		t.Fatal(s)
	}
	fb(false)
}

// fak-test:runtime fast est=10ms lane=default
func TestObservedPrefillBindingPreservesCallerAndUnknown(t *testing.T) {
	m := observedTestFleet(t)
	r := observedTestReserve(t, m, "a")
	defer r.Release()
	p := &agent.InKernelPlanner{}
	route := reservedPlannerReplica{replica: PlannerReplica{Name: "a", Planner: p}, reservation: r}
	ctx := context.Background()
	bound := route.withObservedPrefill(ctx)
	if bound == ctx {
		t.Fatal("native reservation not bound")
	}
	if _, ok := p.TryWithOwnedPrefillObservation(bound, func(int, int) func(bool) { return nil }); ok {
		t.Fatal("binding absent")
	}
	existing := p.WithOwnedPrefillObservation(ctx, func(int, int) func(bool) { return nil })
	if route.withObservedPrefill(existing) != existing {
		t.Fatal("caller instrumentation replaced")
	}
	if s := observedTestStatus(t, m, "a"); s.ObservedPrefillAttempts != 0 {
		t.Fatal("binding invented token counts", s)
	}
	route.replica.Name = "b"
	if route.withObservedPrefill(ctx) != ctx {
		t.Fatal("mismatched worker binding")
	}
	route.replica.Name = "a"
	route.reservation = nil
	if route.withObservedPrefill(ctx) != ctx {
		t.Fatal("no-membership binding")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestObservedPrefillPlacementEpochFencesABA(t *testing.T) {
	m := observedTestFleet(t)
	r := observedTestReserve(t, m, "a")
	defer r.Release()
	oldWorker, oldEpoch := r.worker, r.prefillEpoch
	finishOld := r.startObservedPrefill(oldWorker, oldEpoch, 100, 0)
	pick := func(id string) fleetReservationPick {
		return func([]WorkerStatus) (fleetReservationPickResult, bool) {
			return fleetReservationPickResult{workerID: id}, true
		}
	}
	if _, err := r.Retarget("", nil, nil, pick("b")); err != nil {
		t.Fatal(err)
	}
	// Restore A's health, deliberately allowing API-level same-worker reuse.
	m.ProbeOnce(context.Background())
	if _, err := r.Retarget("", nil, nil, pick("a")); err != nil {
		t.Fatal(err)
	}
	if r.worker != oldWorker {
		t.Fatal("fixture did not return to same worker incarnation")
	}
	if r.prefillEpoch == oldEpoch {
		t.Fatal("placement epoch reused")
	}
	if r.startObservedPrefill(oldWorker, oldEpoch, 500, 0) != nil {
		t.Fatal("stale A-B-A callback admitted")
	}
	finishNew := r.startObservedPrefill(r.worker, r.prefillEpoch, 200, 0)
	finishOld(false)
	if s := observedTestStatus(t, m, "a"); s.ObservedPrefillTokens != 200 || s.ObservedPrefillAttempts != 1 {
		t.Fatal(s)
	}
	finishNew(true)
}
