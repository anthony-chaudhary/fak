package gateway

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fleetWireReplica is a local Planner fake: it echoes its own name back as the
// completion content so a test can tally which replica served each Complete. It
// deliberately does NOT expose ProbeReachability, so *fleetWireReplica satisfies
// agent.Planner alone and newReplicaHealthProbe must fail it closed.
type fleetWireReplica struct {
	name string

	mu        sync.Mutex
	completeN int
}

func (p *fleetWireReplica) Model() string { return p.name }

func (p *fleetWireReplica) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.mu.Lock()
	p.completeN++
	p.mu.Unlock()
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: p.name}, Model: p.name}, nil
}

func (p *fleetWireReplica) count() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.completeN
}

// fleetWireProbeReplica is a fleetWireReplica that ALSO exposes the readiness
// surface newReplicaHealthProbe looks for, so it can be scripted healthy/unhealthy.
type fleetWireProbeReplica struct {
	fleetWireReplica

	mu          sync.Mutex
	probeErr    error
	probeStatus int
	probeCalls  int
}

func (p *fleetWireProbeReplica) ProbeReachability(ctx context.Context) (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.probeCalls++
	status := p.probeStatus
	if status == 0 {
		status = 200
	}
	return status, p.probeErr
}

func (p *fleetWireProbeReplica) setProbeErr(err error) {
	p.mu.Lock()
	p.probeErr = err
	p.mu.Unlock()
}

func (p *fleetWireProbeReplica) probes() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.probeCalls
}

// fleetWireTally runs n Completes and tallies which replica served each.
func fleetWireTally(t *testing.T, router *ReplicaRouter, n int) map[string]int {
	t.Helper()
	got := make(map[string]int)
	for i := 0; i < n; i++ {
		comp, err := router.Complete(context.Background(), nil, nil)
		if err != nil {
			t.Fatalf("Complete(%d): %v", i, err)
		}
		got[comp.Message.Content]++
	}
	return got
}

// TestReplicaMembershipWireUnhealthyReplicaDropsFromRotation is the acceptance test:
// with a live membership attached, a two-replica router spreads traffic across both
// while healthy, then sends EVERY subsequent request to the survivor once peer B
// crosses to unhealthy — and records which planner served each call.
func TestReplicaMembershipWireUnhealthyReplicaDropsFromRotation(t *testing.T) {
	a := &fleetWireReplica{name: "ra"}
	b := &fleetWireReplica{name: "rb"}
	router, err := NewReplicaRouter("fleet", []PlannerReplica{
		{Name: "w-a", Planner: a},
		{Name: "w-b", Planner: b},
	})
	if err != nil {
		t.Fatalf("NewReplicaRouter: %v", err)
	}

	var mu sync.Mutex
	healthy := map[string]bool{"w-a": true, "w-b": true}
	fm := NewFleetMembership(MembershipConfig{
		HealthyAfter:   1,
		UnhealthyAfter: 1,
		Probe: func(_ context.Context, s WorkerSpec) bool {
			mu.Lock()
			defer mu.Unlock()
			return healthy[s.ID]
		},
	})
	for _, id := range []string{"w-a", "w-b"} {
		if err := fm.Add(WorkerSpec{ID: id, Endpoint: id, Models: []string{"fleet"}}); err != nil {
			t.Fatalf("Add(%s): %v", id, err)
		}
	}
	router.WithMembership(fm)
	ctx := context.Background()
	fm.ProbeOnce(ctx) // admit both

	if got := fleetWireTally(t, router, 4); got["ra"] != 2 || got["rb"] != 2 {
		t.Fatalf("healthy fleet round-robin = %v, want ra:2 rb:2", got)
	}

	mu.Lock()
	healthy["w-b"] = false
	mu.Unlock()
	fm.ProbeOnce(ctx)

	got := fleetWireTally(t, router, 20)
	if got["rb"] != 0 {
		t.Fatalf("unhealthy replica B received %d of 20 calls, want 0 (tally %v)", got["rb"], got)
	}
	if got["ra"] != 20 {
		t.Fatalf("healthy replica A served %d of 20 calls, want 20 (tally %v)", got["ra"], got)
	}
}

// TestReplicaMembershipWireDrainExcludesReplica pins the drain arm: draining B (idle,
// healthy) drops it from rotation and A carries every call.
func TestReplicaMembershipWireDrainExcludesReplica(t *testing.T) {
	a := &fleetWireReplica{name: "ra"}
	b := &fleetWireReplica{name: "rb"}
	router, err := NewReplicaRouter("fleet", []PlannerReplica{
		{Name: "w-a", Planner: a},
		{Name: "w-b", Planner: b},
	})
	if err != nil {
		t.Fatalf("NewReplicaRouter: %v", err)
	}
	fm := NewFleetMembership(MembershipConfig{
		HealthyAfter:   1,
		UnhealthyAfter: 1,
		Probe:          func(context.Context, WorkerSpec) bool { return true },
	})
	for _, id := range []string{"w-a", "w-b"} {
		if err := fm.Add(WorkerSpec{ID: id, Endpoint: id, Models: []string{"fleet"}}); err != nil {
			t.Fatalf("Add(%s): %v", id, err)
		}
	}
	router.WithMembership(fm)
	fm.ProbeOnce(context.Background())

	if err := fm.Drain("w-b"); err != nil {
		t.Fatalf("Drain(w-b): %v", err)
	}
	got := fleetWireTally(t, router, 20)
	if got["rb"] != 0 || got["ra"] != 20 {
		t.Fatalf("drained replica still served: %v, want ra:20 rb:0", got)
	}
}

// TestReplicaMembershipWireAllUnhealthyVerdict asserts that emptying the
// admissible set yields ErrNoHealthyWorker — a typed verdict, never a silent drop
// or a nil completion.
func TestReplicaMembershipWireAllUnhealthyVerdict(t *testing.T) {
	a := &fleetWireReplica{name: "ra"}
	b := &fleetWireReplica{name: "rb"}
	router, err := NewReplicaRouter("fleet", []PlannerReplica{
		{Name: "w-a", Planner: a},
		{Name: "w-b", Planner: b},
	})
	if err != nil {
		t.Fatalf("NewReplicaRouter: %v", err)
	}
	fm := NewFleetMembership(MembershipConfig{
		HealthyAfter:   1,
		UnhealthyAfter: 1,
		Probe:          func(context.Context, WorkerSpec) bool { return false },
	})
	for _, id := range []string{"w-a", "w-b"} {
		if err := fm.Add(WorkerSpec{ID: id, Endpoint: id, Models: []string{"fleet"}}); err != nil {
			t.Fatalf("Add(%s): %v", id, err)
		}
	}
	router.WithMembership(fm)
	fm.ProbeOnce(context.Background()) // both cross to unhealthy

	comp, err := router.Complete(context.Background(), nil, nil)
	if err == nil {
		t.Fatalf("all-unhealthy Complete returned comp=%+v err=nil, want ErrNoHealthyWorker", comp)
	}
	if !errors.Is(err, ErrNoHealthyWorker) {
		t.Fatalf("all-unhealthy Complete err = %v, want ErrNoHealthyWorker", err)
	}
	if comp != nil {
		t.Fatalf("all-unhealthy Complete returned non-nil completion alongside the error")
	}
}

// TestReplicaMembershipWireOptOutStaysBlind pins the opt-in contract: a router with
// no membership (or WithMembership(nil)) keeps the blind round-robin over BOTH
// replicas regardless of any fleet state.
func TestReplicaMembershipWireOptOutStaysBlind(t *testing.T) {
	a := &fleetWireReplica{name: "ra"}
	b := &fleetWireReplica{name: "rb"}
	router, err := NewReplicaRouter("fleet", []PlannerReplica{
		{Name: "w-a", Planner: a},
		{Name: "w-b", Planner: b},
	})
	if err != nil {
		t.Fatalf("NewReplicaRouter: %v", err)
	}
	if got := fleetWireTally(t, router, 4); got["ra"] != 2 || got["rb"] != 2 {
		t.Fatalf("no-membership blind round-robin = %v, want ra:2 rb:2", got)
	}

	// Explicit nil restores the blind rotation.
	router.WithMembership(nil)
	if got := fleetWireTally(t, router, 4); got["ra"] != 2 || got["rb"] != 2 {
		t.Fatalf("WithMembership(nil) blind round-robin = %v, want ra:2 rb:2", got)
	}
}

// TestReplicaMembershipWireSingleUpstreamBuildsNoFleet is the reachability guard: a
// single base URL must yield a planner whose replicaRouterOf is nil — proving the
// single-upstream path builds no membership at all.
func TestReplicaMembershipWireSingleUpstreamBuildsNoFleet(t *testing.T) {
	planner, err := newProxyPlanner(Config{Provider: "openai", APIKey: "k"}, "m", []string{"http://127.0.0.1:1/v1"})
	if err != nil {
		t.Fatalf("newProxyPlanner(single): %v", err)
	}
	if router := replicaRouterOf(planner); router != nil {
		t.Fatalf("single-upstream replicaRouterOf = %+v, want nil (no fleet)", router)
	}
	if fm := plannerFleetMembership(planner); fm != nil {
		t.Fatalf("single-upstream planner carried a fleet membership %+v, want nil", fm)
	}
}

func plannerFleetMembership(p agent.Planner) *FleetMembership {
	if r := replicaRouterOf(p); r != nil {
		return r.FleetMembership()
	}
	return nil
}

// TestReplicaMembershipWireBuildRegistersReplicaIDs is the production-wiring
// reachability check: buildReplicaMembership over two hand-built replicas must
// register exactly two workers whose IDs EQUAL the replica Names, and a router armed
// with the returned membership must honor a scripted unhealthy flip.
func TestReplicaMembershipWireBuildRegistersReplicaIDs(t *testing.T) {
	a := &fleetWireProbeReplica{fleetWireReplica: fleetWireReplica{name: "ra"}}
	b := &fleetWireProbeReplica{fleetWireReplica: fleetWireReplica{name: "rb"}}
	replicas := []PlannerReplica{
		{Name: "w-a", Planner: a, Endpoint: "http://a/v1"},
		{Name: "w-b", Planner: b, Endpoint: "http://b/v1"},
	}
	fm, err := buildReplicaMembership(replicas, "fleet")
	if err != nil {
		t.Fatalf("buildReplicaMembership: %v", err)
	}

	snap := fm.Snapshot()
	if len(snap) != 2 {
		t.Fatalf("Snapshot() registered %d workers, want 2", len(snap))
	}
	ids := map[string]bool{}
	for _, st := range snap {
		ids[st.Spec.ID] = true
	}
	for _, want := range []string{"w-a", "w-b"} {
		if !ids[want] {
			t.Fatalf("Snapshot() ids = %v, missing replica Name %q", ids, want)
		}
	}

	router, err := NewReplicaRouter("fleet", replicas)
	if err != nil {
		t.Fatalf("NewReplicaRouter: %v", err)
	}
	fm.ProbeOnce(context.Background()) // both probe healthy (200)
	router.WithMembership(fm)
	if got := fleetWireTally(t, router, 4); got["ra"] != 2 || got["rb"] != 2 {
		t.Fatalf("built membership healthy round-robin = %v, want ra:2 rb:2", got)
	}

	b.setProbeErr(errors.New("upstream refused"))
	// buildReplicaMembership installs the membership with a Probe only, so the
	// hysteresis defaults apply: UnhealthyAfter=2 means ONE failed beat must not
	// evict a healthy worker. Two consecutive failures cross B to unhealthy.
	fm.ProbeOnce(context.Background())
	if got := fleetWireTally(t, router, 4); got["rb"] == 0 {
		t.Fatalf("a single failed beat flapped replica B out of rotation: %v", got)
	}
	fm.ProbeOnce(context.Background())
	if b.probes() < 3 {
		t.Fatalf("probe was not driven through buildReplicaMembership's probe (%d calls)", b.probes())
	}
	got := fleetWireTally(t, router, 20)
	if got["rb"] != 0 || got["ra"] != 20 {
		t.Fatalf("built membership unhealthy flip not honored: %v, want ra:20 rb:0", got)
	}
}

// TestReplicaMembershipWireProbeFailsClosedWithoutReachability pins the fail-closed
// probe edge: a replica whose planner does NOT implement ProbeReachability is never
// admitted, while a replica that does is admitted when the probe succeeds.
func TestReplicaMembershipWireProbeFailsClosedWithoutReachability(t *testing.T) {
	noProbe := &fleetWireReplica{name: "no-probe"}
	prober := &fleetWireProbeReplica{fleetWireReplica: fleetWireReplica{name: "prober"}}

	probe := newReplicaHealthProbe([]PlannerReplica{
		{Name: "w-noprobe", Planner: noProbe, Endpoint: "http://n/v1"},
		{Name: "w-prober", Planner: prober, Endpoint: "http://p/v1"},
	})
	fm := NewFleetMembership(MembershipConfig{HealthyAfter: 1, UnhealthyAfter: 1, Probe: probe})
	for _, id := range []string{"w-noprobe", "w-prober"} {
		if err := fm.Add(WorkerSpec{ID: id, Endpoint: id, Models: []string{"fleet"}}); err != nil {
			t.Fatalf("Add(%s): %v", id, err)
		}
	}
	fm.ProbeOnce(context.Background())

	admissible := map[string]bool{}
	for _, spec := range fm.Admissible() {
		admissible[spec.ID] = true
	}
	if admissible["w-noprobe"] {
		t.Fatalf("worker without ProbeReachability was admitted (presence != invokability)")
	}
	if !admissible["w-prober"] {
		t.Fatalf("worker with a successful ProbeReachability was NOT admitted: %v", admissible)
	}

	// Flip the prober unhealthy: the probe must now fail it too.
	prober.setProbeErr(errors.New("503"))
	fm.ProbeOnce(context.Background())
	for _, spec := range fm.Admissible() {
		if spec.ID == "w-prober" {
			t.Fatalf("prober stayed admissible after a failing reachability probe")
		}
	}
}
