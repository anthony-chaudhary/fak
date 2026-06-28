package gateway

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

// scriptedProbe returns a Probe whose per-worker result is read from a shared map
// the test mutates between ticks, so a test can flip a worker healthy/unhealthy
// deterministically and count how many ticks the loop takes to react.
type scriptedProbe struct {
	mu     sync.Mutex
	result map[string]bool // worker id -> next probe result (absent => true)
	calls  map[string]int
}

func newScriptedProbe() *scriptedProbe {
	return &scriptedProbe{result: map[string]bool{}, calls: map[string]int{}}
}

func (s *scriptedProbe) set(id string, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.result[id] = ok
}

func (s *scriptedProbe) probe(_ context.Context, spec WorkerSpec) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls[spec.ID]++
	ok, set := s.result[spec.ID]
	return !set || ok // default healthy until the test says otherwise
}

func admissibleIDs(m *FleetMembership) map[string]bool {
	out := map[string]bool{}
	for _, s := range m.Admissible() {
		out[s.ID] = true
	}
	return out
}

// Acceptance: a membership registry exists that the router reads; entries carry
// id/role/engine/health; an unknown worker is not admissible until it probes
// healthy, and the registry replaces a flat endpoint list as the live fleet view.
func TestFleetMembershipRegistryAndInitialAdmission(t *testing.T) {
	sp := newScriptedProbe()
	m := NewFleetMembership(MembershipConfig{Probe: sp.probe})
	if err := m.Add(WorkerSpec{ID: "w1", Role: RolePrefill, Engine: EngineExternal, Endpoint: "h1:8000"}); err != nil {
		t.Fatalf("Add w1: %v", err)
	}
	if err := m.Add(WorkerSpec{ID: "w2", Role: RoleDecode, Engine: EngineNative, Endpoint: "h2:8000"}); err != nil {
		t.Fatalf("Add w2: %v", err)
	}
	if err := m.Add(WorkerSpec{ID: "w1"}); !errors.Is(err, ErrWorkerExists) {
		t.Fatalf("duplicate Add error = %v, want ErrWorkerExists", err)
	}

	// Before any probe both workers are unknown -> not admissible.
	if got := m.Admissible(); len(got) != 0 {
		t.Fatalf("unprobed fleet admissible = %+v, want none", got)
	}

	m.ProbeOnce(context.Background()) // both default-healthy
	adm := admissibleIDs(m)
	if !adm["w1"] || !adm["w2"] {
		t.Fatalf("after probe admissible = %v, want both", adm)
	}

	// The snapshot carries the per-worker identity a router reads.
	snap := m.Snapshot()
	if len(snap) != 2 {
		t.Fatalf("snapshot len = %d, want 2", len(snap))
	}
	if snap[0].Spec.Role != RolePrefill || snap[0].Spec.Engine != EngineExternal || snap[0].Health != HealthHealthy {
		t.Fatalf("w1 status = %+v, want prefill/external/healthy", snap[0])
	}
	if snap[1].Spec.Role != RoleDecode || snap[1].Spec.Engine != EngineNative {
		t.Fatalf("w2 status = %+v, want decode/native", snap[1])
	}
}

// Acceptance: the router drops an unhealthy worker within the bounded health
// interval (unhealthyAfter ticks), AND a single transient failure does not flap a
// worker out (hysteresis).
func TestFleetMembershipDropsUnhealthyWithinBoundAndHysteresis(t *testing.T) {
	sp := newScriptedProbe()
	const unhealthyAfter = 3
	m := NewFleetMembership(MembershipConfig{UnhealthyAfter: unhealthyAfter, Probe: sp.probe})
	mustAdd(t, m, WorkerSpec{ID: "w1"})
	m.ProbeOnce(context.Background())
	if !admissibleIDs(m)["w1"] {
		t.Fatalf("w1 not admissible after a healthy probe")
	}

	// Hysteresis: a single failure between healthy probes must NOT drop w1.
	sp.set("w1", false)
	m.ProbeOnce(context.Background())
	if !admissibleIDs(m)["w1"] {
		t.Fatalf("single transient failure flapped w1 out (failStreak=1 < %d)", unhealthyAfter)
	}
	sp.set("w1", true)
	m.ProbeOnce(context.Background()) // recovery resets the streak
	if !admissibleIDs(m)["w1"] {
		t.Fatalf("w1 dropped after recovery")
	}

	// Sustained failure: w1 must leave the admissible set within unhealthyAfter ticks.
	sp.set("w1", false)
	for tick := 1; tick <= unhealthyAfter; tick++ {
		m.ProbeOnce(context.Background())
		admitted := admissibleIDs(m)["w1"]
		switch {
		case tick < unhealthyAfter && !admitted:
			t.Fatalf("w1 dropped after %d failed ticks, want it to survive until %d (hysteresis)", tick, unhealthyAfter)
		case tick == unhealthyAfter && admitted:
			t.Fatalf("w1 still admissible after %d failed ticks, want it dropped within the bound", unhealthyAfter)
		}
	}
}

// Acceptance: Drain marks a worker non-admissible (router routes no NEW work to
// it) while existing in-flight requests are allowed to finish before removal.
func TestFleetMembershipDrainFinishesInflightBeforeRemoval(t *testing.T) {
	sp := newScriptedProbe()
	m := NewFleetMembership(MembershipConfig{Probe: sp.probe})
	mustAdd(t, m, WorkerSpec{ID: "w1"})
	mustAdd(t, m, WorkerSpec{ID: "w2"})
	m.ProbeOnce(context.Background())

	// A request is in flight on w1.
	if err := m.Acquire("w1"); err != nil {
		t.Fatalf("Acquire w1: %v", err)
	}

	// Drain w1: it must drop out of the admissible set at once, but NOT be removed
	// while a request is still in flight.
	if err := m.Drain("w1"); err != nil {
		t.Fatalf("Drain w1: %v", err)
	}
	if admissibleIDs(m)["w1"] {
		t.Fatalf("drained w1 still admissible for NEW work")
	}
	if err := m.Acquire("w1"); !errors.Is(err, ErrNoHealthyWorker) {
		t.Fatalf("Acquire on draining w1 = %v, want ErrNoHealthyWorker (no new work)", err)
	}
	if len(m.Snapshot()) != 2 {
		t.Fatalf("w1 removed before its in-flight request finished")
	}

	// The in-flight request finishes -> w1 is removed (drain-before-remove done).
	m.Release("w1")
	snap := m.Snapshot()
	if len(snap) != 1 || snap[0].Spec.ID != "w2" {
		t.Fatalf("after Release snapshot = %+v, want only w2", snap)
	}
}

// Acceptance: an in-flight request on a lost worker is re-routed to a healthy
// worker (failover), and there is no silent drop — when all admissible workers
// fail the caller gets a typed verdict.
func TestFleetMembershipDispatchFailoverAndTypedVerdict(t *testing.T) {
	sp := newScriptedProbe()
	m := NewFleetMembership(MembershipConfig{Probe: sp.probe})
	mustAdd(t, m, WorkerSpec{ID: "w1"})
	mustAdd(t, m, WorkerSpec{ID: "w2"})
	m.ProbeOnce(context.Background())

	// w1 fails the send; the request must fail over to w2 and be served there.
	var served []string
	got, err := m.Dispatch(context.Background(), func(_ context.Context, spec WorkerSpec) error {
		served = append(served, spec.ID)
		if spec.ID == "w1" {
			return errors.New("connection refused")
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Dispatch with one healthy fallback: %v", err)
	}
	if got.ID != "w2" {
		t.Fatalf("served by %q, want failover to w2", got.ID)
	}
	if len(served) != 2 || served[0] != "w1" || served[1] != "w2" {
		t.Fatalf("send order = %v, want [w1 w2] (tried w1 then failed over)", served)
	}
	// A failover transition was recorded against the worker we moved off of.
	if !hasEvent(m.DrainEvents(), EventFailover, "w1") {
		t.Fatalf("no failover event recorded for w1")
	}

	// Now make EVERY admissible worker fail the send: no silent drop, typed verdict.
	_, err = m.Dispatch(context.Background(), func(_ context.Context, _ WorkerSpec) error {
		return errors.New("down")
	})
	if !errors.Is(err, ErrNoHealthyWorker) {
		t.Fatalf("all-fail Dispatch err = %v, want ErrNoHealthyWorker", err)
	}

	// And with zero admissible workers, Dispatch refuses rather than dropping.
	empty := NewFleetMembership(MembershipConfig{Probe: sp.probe})
	if _, err := empty.Dispatch(context.Background(), func(context.Context, WorkerSpec) error { return nil }); !errors.Is(err, ErrNoHealthyWorker) {
		t.Fatalf("empty Dispatch err = %v, want ErrNoHealthyWorker", err)
	}
}

// Acceptance: the loop works identically for an external-adapter worker and a
// fak-native worker.
func TestFleetMembershipUniformAcrossEngineKinds(t *testing.T) {
	for _, kind := range []EngineKind{EngineExternal, EngineNative} {
		kind := kind
		t.Run(string(kind), func(t *testing.T) {
			sp := newScriptedProbe()
			m := NewFleetMembership(MembershipConfig{UnhealthyAfter: 2, Probe: sp.probe})
			mustAdd(t, m, WorkerSpec{ID: "w", Engine: kind})
			m.ProbeOnce(context.Background())
			if !admissibleIDs(m)["w"] {
				t.Fatalf("%s worker not admissible after healthy probe", kind)
			}
			sp.set("w", false)
			m.ProbeOnce(context.Background())
			m.ProbeOnce(context.Background())
			if admissibleIDs(m)["w"] {
				t.Fatalf("%s worker not evicted after sustained failure", kind)
			}
		})
	}
}

// Acceptance: membership/health/drain/failover transitions are emitted with a
// per-worker label.
func TestFleetMembershipEmitsPerWorkerTransitions(t *testing.T) {
	sp := newScriptedProbe()
	m := NewFleetMembership(MembershipConfig{HealthyAfter: 1, UnhealthyAfter: 1, Probe: sp.probe})
	mustAdd(t, m, WorkerSpec{ID: "w1"})
	m.ProbeOnce(context.Background()) // unknown -> healthy
	sp.set("w1", false)
	m.ProbeOnce(context.Background()) // healthy -> unhealthy
	_ = m.Drain("w1")                 // drain (idle -> removed)

	events := m.DrainEvents()
	wantKinds := []MembershipKind{EventAdded, EventHealthChanged, EventHealthChanged, EventDrainStarted, EventRemoved}
	if len(events) != len(wantKinds) {
		t.Fatalf("events = %+v, want %d transitions", events, len(wantKinds))
	}
	for i, ev := range events {
		if ev.Kind != wantKinds[i] {
			t.Fatalf("event[%d].Kind = %q, want %q (%+v)", i, ev.Kind, wantKinds[i], events)
		}
		if ev.WorkerID != "w1" {
			t.Fatalf("event[%d] missing per-worker label: %+v", i, ev)
		}
	}
	// DrainEvents clears the log.
	if got := m.DrainEvents(); len(got) != 0 {
		t.Fatalf("DrainEvents did not clear: %+v", got)
	}
}

// The continuous loop's real-time driver probes immediately and returns on cancel.
func TestFleetMembershipRunHealthLoopProbesThenStops(t *testing.T) {
	sp := newScriptedProbe()
	m := NewFleetMembership(MembershipConfig{Probe: sp.probe})
	mustAdd(t, m, WorkerSpec{ID: "w1"})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel up front: the immediate probe still runs, then the loop returns.
	done := make(chan struct{})
	go func() {
		m.RunHealthLoop(ctx, time.Hour)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("RunHealthLoop did not return on context cancel")
	}
	if !admissibleIDs(m)["w1"] {
		t.Fatalf("immediate probe did not run before the loop returned")
	}
}

func mustAdd(t *testing.T, m *FleetMembership, spec WorkerSpec) {
	t.Helper()
	if err := m.Add(spec); err != nil {
		t.Fatalf("Add %q: %v", spec.ID, err)
	}
}

func hasEvent(events []MembershipEvent, kind MembershipKind, id string) bool {
	for _, ev := range events {
		if ev.Kind == kind && ev.WorkerID == id {
			return true
		}
	}
	return false
}
