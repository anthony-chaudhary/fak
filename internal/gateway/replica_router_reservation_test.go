package gateway

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

var errReservationTestEndpoint = errors.New("reservation test endpoint failed")

type reservationTestPlanner struct {
	worker     string
	membership *FleetMembership
	err        error
	started    chan struct{}
	wait       <-chan struct{}
	wantBooked int

	startOnce sync.Once
	calls     atomic.Int32
	unbooked  atomic.Bool
}

func (p *reservationTestPlanner) Model() string { return "Qwen3.8" }

func (p *reservationTestPlanner) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls.Add(1)
	wantBooked := p.wantBooked
	if wantBooked < 1 {
		wantBooked = 1
	}
	if status, ok := fleetWorkerStatus(p.membership, p.worker); !ok || status.Inflight < wantBooked {
		p.unbooked.Store(true)
	}
	if p.started != nil {
		p.startOnce.Do(func() { close(p.started) })
	}
	if p.wait != nil {
		select {
		case <-p.wait:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if p.err != nil {
		return nil, p.err
	}
	return &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, Content: p.worker},
		Model:   "Qwen3.8",
	}, nil
}

func fleetWorkerStatus(m *FleetMembership, id string) (WorkerStatus, bool) {
	if m == nil {
		return WorkerStatus{}, false
	}
	for _, status := range m.Snapshot() {
		if status.Spec.ID == id {
			return status, true
		}
	}
	return WorkerStatus{}, false
}

func nativeReservationFleet(t *testing.T, ids ...string) *FleetMembership {
	t.Helper()
	m := NewFleetMembership(MembershipConfig{
		HealthyAfter:   1,
		UnhealthyAfter: 2,
		Probe:          func(context.Context, WorkerSpec) bool { return true },
	})
	for _, id := range ids {
		mustAdd(t, m, WorkerSpec{
			ID:     id,
			Engine: EngineNative,
			Models: []string{"Qwen3.8"},
		})
	}
	m.ProbeOnce(context.Background())
	return m
}

func nativeReservationRouter(t *testing.T, m *FleetMembership, policy PickPolicy, planners ...*reservationTestPlanner) *ReplicaRouter {
	t.Helper()
	replicas := make([]PlannerReplica, len(planners))
	for i, planner := range planners {
		replicas[i] = PlannerReplica{Name: planner.worker, Planner: planner}
	}
	r, err := NewReplicaRouter("Qwen3.8", replicas)
	if err != nil {
		t.Fatalf("NewReplicaRouter: %v", err)
	}
	r.WithMembership(m).WithPickPolicy(policy)
	return r
}

func assertReservationFleetClean(t *testing.T, m *FleetMembership) {
	t.Helper()
	for _, status := range m.Snapshot() {
		if status.Inflight < 0 {
			t.Fatalf("worker %q occupancy went negative: %+v", status.Spec.ID, status)
		}
		if status.Inflight != 0 {
			t.Fatalf("worker %q leaked occupancy: %+v", status.Spec.ID, status)
		}
	}
}

// The winner must be booked before the membership lock is released. Otherwise a
// concurrent drain removes it while its planner is already serving the request.
func TestReplicaRouterReservationKeepsWinnerUntilSendFinishes(t *testing.T) {
	m := nativeReservationFleet(t, "w1")
	release := make(chan struct{})
	planner := &reservationTestPlanner{
		worker:     "w1",
		membership: m,
		started:    make(chan struct{}),
		wait:       release,
	}
	r := nativeReservationRouter(t, m, NewCacheAwarePolicy(nil, DefaultSkewThreshold()), planner)

	done := make(chan error, 1)
	go func() {
		_, err := r.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: "winner-loss"}}, nil)
		done <- err
	}()
	<-planner.started

	if err := m.Drain("w1"); err != nil {
		t.Fatalf("Drain(w1): %v", err)
	}
	status, ok := fleetWorkerStatus(m, "w1")
	if !ok || !status.Draining || status.Inflight != 1 {
		t.Fatalf("winner was lost before send finished: status=%+v present=%v", status, ok)
	}

	close(release)
	if err := <-done; err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got := m.Snapshot(); len(got) != 0 {
		t.Fatalf("drained winner survived reservation release: %+v", got)
	}
	if planner.unbooked.Load() {
		t.Fatal("winner planner was called without an occupancy booking")
	}
}

type unbookablePickPolicy struct {
	replica PlannerReplica
}

func (p unbookablePickPolicy) Pick([]PlannerReplica, []string, func(string) int) (PlannerReplica, bool) {
	return p.replica, true
}

type namedPickPolicy string

func (p namedPickPolicy) Pick(candidates []PlannerReplica, _ []string, _ func(string) int) (PlannerReplica, bool) {
	for _, candidate := range candidates {
		if candidate.Name == string(p) {
			return candidate, true
		}
	}
	return PlannerReplica{}, false
}

// A policy result that cannot be booked must be rejected before the send and fall
// back to a real admissible replica under the same membership lock.
func TestReplicaRouterBookingFailureFallsBackBeforeSend(t *testing.T) {
	m := nativeReservationFleet(t, "w1")
	booked := &reservationTestPlanner{worker: "w1", membership: m}
	ghost := &reservationTestPlanner{worker: "ghost", membership: m}
	policy := unbookablePickPolicy{replica: PlannerReplica{Name: "ghost", Planner: ghost}}
	r := nativeReservationRouter(t, m, policy, booked)

	got, err := r.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: "booking-failure"}}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got.Message.Content != "w1" {
		t.Fatalf("completion worker = %q, want booked fallback w1", got.Message.Content)
	}
	if ghost.calls.Load() != 0 || booked.calls.Load() != 1 {
		t.Fatalf("planner calls ghost=%d booked=%d, want 0/1", ghost.calls.Load(), booked.calls.Load())
	}
	if booked.unbooked.Load() {
		t.Fatal("fallback planner was called without an occupancy booking")
	}
	assertReservationFleetClean(t, m)
}

// Cancellation is terminal for the request: release its one booking exactly once
// and do not re-send the canceled request to a fallback.
func TestReplicaRouterCancellationReleasesWithoutFallback(t *testing.T) {
	m := nativeReservationFleet(t, "w1", "w2")
	if err := m.Acquire("w1"); err != nil {
		t.Fatalf("sentinel Acquire(w1): %v", err)
	}
	first := &reservationTestPlanner{
		worker:     "w1",
		membership: m,
		started:    make(chan struct{}),
		wait:       make(chan struct{}),
		wantBooked: 2,
	}
	second := &reservationTestPlanner{worker: "w2", membership: m}
	r := nativeReservationRouter(t, m, namedPickPolicy("w1"), first, second)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := r.Complete(ctx, []agent.Message{{Role: agent.RoleUser, Content: "cancel"}}, nil)
		done <- err
	}()
	<-first.started
	if status, ok := fleetWorkerStatus(m, "w1"); !ok || status.Inflight != 2 {
		t.Fatalf("running cancellation occupancy = %+v present=%v, want 2", status, ok)
	}
	cancel()

	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("Complete error = %v, want context.Canceled", err)
	}
	if first.calls.Load() != 1 || second.calls.Load() != 0 {
		t.Fatalf("planner calls first=%d fallback=%d, want 1/0", first.calls.Load(), second.calls.Load())
	}
	if first.unbooked.Load() {
		t.Fatal("canceled planner was called without an occupancy booking")
	}
	if status, ok := fleetWorkerStatus(m, "w1"); !ok || status.Inflight != 1 {
		t.Fatalf("cancellation released other occupancy or leaked its own: status=%+v present=%v", status, ok)
	}
	m.Release("w1")
	assertReservationFleetClean(t, m)
}

// Endpoint failure retargets the same logical reservation to an untried worker.
// Both physical sends must see a booking, and the final release must leave zero
// (never negative) occupancy on every worker.
func TestReplicaRouterEndpointFailureRetargetsBookedFallback(t *testing.T) {
	m := nativeReservationFleet(t, "w1", "w3")
	mustAdd(t, m, WorkerSpec{
		ID:     "w2",
		Engine: EngineExternal,
		Models: []string{"Qwen3.8"},
	})
	m.ProbeOnce(context.Background())
	for _, id := range []string{"w1", "w2", "w3"} {
		if err := m.Acquire(id); err != nil {
			t.Fatalf("sentinel Acquire(%s): %v", id, err)
		}
	}
	first := &reservationTestPlanner{worker: "w1", membership: m, err: errReservationTestEndpoint, wantBooked: 2}
	external := &reservationTestPlanner{worker: "w2", membership: m, wantBooked: 2}
	fallback := &reservationTestPlanner{worker: "w3", membership: m, wantBooked: 2}
	r := nativeReservationRouter(t, m, NewCacheAwarePolicy(nil, DefaultSkewThreshold()), first, external, fallback)

	got, err := r.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: "fallback"}}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got.Message.Content != "w3" {
		t.Fatalf("completion worker = %q, want fak-native fallback w3", got.Message.Content)
	}
	if first.calls.Load() != 1 || external.calls.Load() != 0 || fallback.calls.Load() != 1 {
		t.Fatalf("planner calls first=%d external=%d fallback=%d, want 1/0/1", first.calls.Load(), external.calls.Load(), fallback.calls.Load())
	}
	if first.unbooked.Load() || fallback.unbooked.Load() {
		t.Fatalf("unbooked send first=%v fallback=%v", first.unbooked.Load(), fallback.unbooked.Load())
	}
	if !hasEvent(m.DrainEvents(), EventFailover, "w1") {
		t.Fatal("fallback did not record failover from w1")
	}
	for _, id := range []string{"w1", "w2", "w3"} {
		if status, ok := fleetWorkerStatus(m, id); !ok || status.Inflight != 1 {
			t.Fatalf("fallback released other occupancy or leaked its own on %s: status=%+v present=%v", id, status, ok)
		}
		m.Release(id)
	}
	assertReservationFleetClean(t, m)
}

func TestReplicaRouterHedgeBooksBothPhysicalAttempts(t *testing.T) {
	m := nativeReservationFleet(t, "w1", "w2")
	primary := &reservationTestPlanner{
		worker:     "w1",
		membership: m,
		started:    make(chan struct{}),
		wait:       make(chan struct{}),
	}
	alternate := &reservationTestPlanner{worker: "w2", membership: m}
	r := nativeReservationRouter(t, m, NewCacheAwarePolicy(nil, DefaultSkewThreshold()), primary, alternate)
	r.Hedge = eligibleHedgePolicy(nil)

	got, err := r.Complete(
		context.Background(),
		[]agent.Message{{Role: agent.RoleUser, Content: "hedge"}},
		nil,
		zeroTemperatureOpt(),
	)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got.Message.Content != "w2" {
		t.Fatalf("completion worker = %q, want hedged worker w2", got.Message.Content)
	}
	if primary.calls.Load() != 1 || alternate.calls.Load() != 1 {
		t.Fatalf("planner calls primary=%d hedge=%d, want 1/1", primary.calls.Load(), alternate.calls.Load())
	}
	if primary.unbooked.Load() || alternate.unbooked.Load() {
		t.Fatalf("unbooked hedge send primary=%v hedge=%v", primary.unbooked.Load(), alternate.unbooked.Load())
	}
	assertReservationFleetClean(t, m)
}
