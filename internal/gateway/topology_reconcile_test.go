package gateway

import (
	"context"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/inferencetopology"
)

// fak-test:runtime fast est=100ms lane=default
func TestTopologyReconcileObservedCapacity(t *testing.T) {
	now := time.Date(2026, 11, 2, 5, 0, 30, 0, time.UTC)
	f, e := os.Open("../inferencetopology/testdata/runtime_search.json")
	if e != nil {
		t.Fatal(e)
	}
	defer f.Close()
	search, e := inferencetopology.DecodeRuntimeSearch(f)
	if e != nil {
		t.Fatal(e)
	}
	search.Workload.Concurrency = 4
	selected, e := inferencetopology.SearchRuntimeTopology(t.Context(), search, now)
	if e != nil || selected.Chosen == nil {
		t.Fatalf("setup %v", e)
	}
	m := NewFleetMembership(MembershipConfig{UnhealthyAfter: 1, Probe: func(context.Context, WorkerSpec) bool { return true }})
	for _, id := range []string{"r0", "r1", "r2"} {
		if e = m.Add(WorkerSpec{ID: id}); e != nil {
			t.Fatal(e)
		}
	}
	m.ProbeOnce(t.Context())
	in := TopologyShadowInput{Schema: TopologyShadowSchema, Search: search, Incumbent: 3, HysteresisTicks: 2, Objective: PoolObjective{MinReplicas: 1, MaxReplicas: 3, ScaleStep: 1, QueueHigh: 4, QueueLow: 1, KVUtilHigh: .9, KVUtilLow: .5}}
	waiting, running, kv := 0.0, 0.0, .2
	for _, st := range m.Snapshot() {
		in.Workers = append(in.Workers, TopologyPressureObservation{TopologyID: selected.Chosen.Config.TopologyID, RankID: st.Spec.ID, WorkerID: st.Spec.ID, Incarnation: st.Incarnation, Epoch: selected.Chosen.ObservationEpoch, Source: "fixture", ModelRevision: search.ModelRevision, Backend: search.Backend, Kernel: search.Kernel, LayoutDigest: selected.Chosen.LayoutDigest, LinkDigest: selected.Chosen.LinkDigest, ObservedAt: now, TTLSeconds: 60, Waiting: &waiting, Running: &running, KVUtilization: &kv})
	}
	s := &Server{fleet: m}
	if e = s.SetTopologyShadow(in, nil); e != nil {
		t.Fatal(e)
	}
	before := m.Snapshot()
	tick := func(at time.Time) TopologyShadowReport {
		t.Helper()
		r, ok := s.ReconcileTopologyShadow(t.Context(), at)
		if !ok {
			t.Fatal("missing report")
		}
		return r
	}
	if r := tick(now); r.Decision.DesiredReplicas != 3 || r.Decision.CurrentReplicas != 3 {
		t.Fatalf("hysteresis %+v", r)
	}
	if r := tick(now.Add(time.Second)); r.Decision.DesiredReplicas != 2 || r.Decision.AppliedReplicas != 3 || r.Decision.Action != ScaleActionHold {
		t.Fatalf("shadow %+v", r)
	}
	if !reflect.DeepEqual(before, m.Snapshot()) {
		t.Fatal("shadow mutated routing/bookings")
	}
	// Explicit unhealthy capacity is excluded; unknown replacement must retain
	// incumbent until a new supplied binding/epoch is accepted.
	m.markDispatchFailure("r2", nil)
	if r := tick(now.Add(2 * time.Second)); r.Decision.CurrentReplicas != 2 {
		t.Fatalf("unhealthy counted %+v", r)
	}
	_ = m.Remove("r2")
	_ = m.Add(WorkerSpec{ID: "r2"})
	m.ProbeOnce(t.Context())
	if r := tick(now.Add(3 * time.Second)); !r.Backpressure || r.Decision.DesiredReplicas != 2 || r.Decision.CurrentReplicas != 2 {
		t.Fatalf("replacement identity %+v", r)
	}
	if r := tick(now.Add(61 * time.Second)); !r.Backpressure || r.Decision.DesiredReplicas != 2 {
		t.Fatalf("stale scaled %+v", r)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	r, _ := s.ReconcileTopologyShadow(ctx, now)
	if !r.Backpressure || r.Decision.DesiredReplicas != 2 {
		t.Fatal("cancel changed incumbent")
	}
	// A valid refreshed incarnation can re-enter the same existing owner.
	for i, st := range m.Snapshot() {
		in.Workers[i].Incarnation = st.Incarnation
	}
	in.Incumbent = 2
	if e = s.SetTopologyShadow(in, nil); e != nil {
		t.Fatal(e)
	}
	if r = tick(now); r.Decision.CurrentReplicas != 3 {
		t.Fatal("replacement did not re-enter")
	}
	// Cancellation / unknown ticks cannot advance the active controller's state.
	active := NewServingAutoscaler(m, nil, ServingAutoscalerConfig{HysteresisTicks: 2})
	state := active.state
	_ = active.shadowDecision(now, 3, 3, true, PoolSignals{QueueDepth: 100}, in.Objective)
	if len(state) != 0 {
		t.Fatal("shadow primed live hysteresis")
	}
}
