package gateway

import (
	"context"
	"math"
	"testing"
	"time"
)

// fak-test:runtime fast est=100ms lane=default
func TestTopologyPressureScrapeLag(t *testing.T) {
	now := time.Date(2026, 11, 2, 5, 0, 30, 0, time.UTC)
	m := NewFleetMembership(MembershipConfig{Probe: func(context.Context, WorkerSpec) bool { return true }})
	add := func() {
		t.Helper()
		if err := m.Add(WorkerSpec{ID: "a"}); err != nil {
			t.Fatal(err)
		}
		m.ProbeOnce(t.Context())
	}
	add()
	pick := func([]WorkerStatus) (fleetReservationPickResult, bool) {
		return fleetReservationPickResult{workerID: "a", bookedOutputBlocks: 2}, true
	}
	book := func() *fleetReservation {
		t.Helper()
		r, e := m.reserveForModel("", "", nil, nil, pick)
		if e != nil {
			t.Fatal(e)
		}
		return r
	}
	r1, r2, r3 := book(), book(), book()
	waiting, running, kv := 1.0, 1.0, .3
	o := TopologyPressureObservation{WorkerID: "a", Incarnation: m.Snapshot()[0].Incarnation, Epoch: 1, Source: "fixture", ObservedAt: now, TTLSeconds: 10, Waiting: &waiting, Running: &running, KVUtilization: &kv}
	if p := topologyPressure(now, 1, m.Snapshot()[0], o); !p.Known || p.QueueDepth != 2 {
		t.Fatalf("lag %+v", p)
	}
	r1.Release()
	r1.Release()
	if p := topologyPressure(now, 1, m.Snapshot()[0], o); p.QueueDepth != 1 {
		t.Fatalf("double-count %+v", p)
	}
	for _, mutate := range []func(*TopologyPressureObservation){func(x *TopologyPressureObservation) { x.Waiting = nil }, func(x *TopologyPressureObservation) { v := math.NaN(); x.Running = &v }, func(x *TopologyPressureObservation) { x.ObservedAt = now.Add(time.Second) }, func(x *TopologyPressureObservation) { x.ObservedAt = now.Add(-10 * time.Second) }, func(x *TopologyPressureObservation) { x.Incarnation++ }, func(x *TopologyPressureObservation) { x.Source = "" }} {
		x := o
		mutate(&x)
		if topologyPressure(now, 1, m.Snapshot()[0], x).Known {
			t.Fatal("unknown accepted")
		}
	}
	// An old completion/fallback must neither consume nor poison a replacement.
	if err := m.Remove("a"); err != nil {
		t.Fatal(err)
	}
	add()
	fresh := book()
	r2.Release()
	_, _ = r3.Retarget("", nil, map[string]struct{}{"a": {}}, pick)
	r3.Release()
	st := m.Snapshot()[0]
	if st.Inflight != 1 || st.BookedOutputBlocks != 2 || st.Health != HealthHealthy {
		t.Fatalf("replacement charged %+v", st)
	}
	fresh.Release()
	// Captured DispatchForModel completion is also incarnation-fenced.
	_, _ = m.DispatchForModel(t.Context(), "", func(ctx context.Context, _ WorkerSpec) error {
		_ = m.Remove("a")
		add()
		fresh = book()
		return context.Canceled
	})
	if st = m.Snapshot()[0]; st.Inflight != 1 || st.Health != HealthHealthy {
		t.Fatalf("dispatch poisoned replacement %+v", st)
	}
	fresh.Release()
	// ID-only legacy callers cannot overlap a replacement they cannot identify.
	if e := m.Acquire("a"); e != nil {
		t.Fatal(e)
	}
	_ = m.Remove("a")
	if e := m.Add(WorkerSpec{ID: "a"}); e == nil {
		t.Fatal("legacy replacement admitted")
	}
	m.Release("a")
	add()
	// A probe that finishes after remove/re-add cannot make the new worker healthy.
	m.probe = func(context.Context, WorkerSpec) bool {
		_ = m.Remove("a")
		if e := m.Add(WorkerSpec{ID: "a"}); e != nil {
			t.Fatal(e)
		}
		return true
	}
	m.ProbeOnce(t.Context())
	if m.Snapshot()[0].Health != HealthUnknown {
		t.Fatal("old probe applied to replacement")
	}
}
