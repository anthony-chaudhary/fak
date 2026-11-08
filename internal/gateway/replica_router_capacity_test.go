package gateway

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestCapacityAwarePolicyNormalizedLoad pins the borrowed heterogeneous-capacity
// rule: placement cost is load/CAPACITY, not raw load. A replica with half the
// learned throughput must win on twice the raw load when its normalized cost is
// lower, and lose when it is higher.
func TestCapacityAwarePolicyNormalizedLoad(t *testing.T) {
	p := NewCapacityAwarePolicy(nil)
	// fast learns 40 tok/s; slow learns 10 tok/s.
	p.ObserveServedOutput("fast", 400, 10*time.Second)
	p.ObserveServedOutput("slow", 100, 10*time.Second)

	// fast carries load 1 (cost 1/40 = 0.025); slow carries load 2 (cost 2/10 = 0.2).
	load := func(name string) int {
		if name == "fast" {
			return 1
		}
		return 2
	}
	got, ok := p.Pick([]PlannerReplica{{Name: "slow"}, {Name: "fast"}}, nil, load)
	if !ok || got.Name != "fast" {
		t.Fatalf("expected faster replica despite higher raw load, got %q (ok=%v)", got.Name, ok)
	}

	// Inverting the loads inverts the verdict: slow now has load 1 (0.1) vs
	// fast's load 4 (0.1) — equal scores fall to the lower raw load.
	loadInv := func(name string) int {
		if name == "fast" {
			return 4
		}
		return 1
	}
	got, ok = p.Pick([]PlannerReplica{{Name: "fast"}, {Name: "slow"}}, nil, loadInv)
	if !ok || got.Name != "slow" {
		t.Fatalf("expected slower replica at equal normalized cost via load tiebreak, got %q (ok=%v)", got.Name, ok)
	}
}

// TestCapacityAwarePolicyKVPenalty pins the (1 + α·kvUsed²) pressure factor: at
// equal load and capacity, the replica with the fuller KV cache loses, and the
// penalty grows with the SQUARE of usage.
func TestCapacityAwarePolicyKVPenalty(t *testing.T) {
	kv := map[string]float64{"calm": 0.1, "hot": 0.6}
	p := NewCapacityAwarePolicy(func(name string) (float64, bool) {
		v, ok := kv[name]
		return v, ok
	})
	p.ObserveServedOutput("calm", 200, 10*time.Second) // 20 tok/s
	p.ObserveServedOutput("hot", 200, 10*time.Second)  // 20 tok/s

	got, ok := p.Pick([]PlannerReplica{{Name: "hot"}, {Name: "calm"}}, nil, func(string) int { return 1 })
	if !ok || got.Name != "calm" {
		t.Fatalf("expected lower-KV-pressure replica, got %q (ok=%v)", got.Name, ok)
	}

	// Both equally pressured: the factor cancels and the name tiebreak decides.
	kv["calm"] = 0.4
	kv["hot"] = 0.4
	got, ok = p.Pick([]PlannerReplica{{Name: "hot"}, {Name: "calm"}}, nil, func(string) int { return 1 })
	if !ok || got.Name != "calm" {
		t.Fatalf("expected deterministic name tiebreak at equal scores, got %q (ok=%v)", got.Name, ok)
	}
}

// TestCapacityAwarePolicyCriticalFreeKVVeto pins the critical-free-KV guard: a
// replica whose free-KV fraction drops below the floor scores +Inf and is
// skipped even at zero load; when EVERY candidate is vetoed the dispatch still
// places on the least-loaded last resort instead of stranding the request.
func TestCapacityAwarePolicyCriticalFreeKVVeto(t *testing.T) {
	healthy := map[string]float64{"sick": 0.97, "well": 0.3} // sick: 3% free < 5% floor
	p := NewCapacityAwarePolicy(func(name string) (float64, bool) {
		v, ok := healthy[name]
		return v, ok
	})
	p.ObserveServedOutput("sick", 400, 10*time.Second)
	p.ObserveServedOutput("well", 100, 10*time.Second)

	got, ok := p.Pick([]PlannerReplica{{Name: "sick"}, {Name: "well"}}, nil, func(string) int { return 1 })
	if !ok || got.Name != "well" {
		t.Fatalf("expected vetoed replica skipped in favor of healthy one, got %q (ok=%v)", got.Name, ok)
	}

	// All candidates below the floor: still place, on the least-loaded.
	allSick := map[string]float64{"sick": 0.97, "worse": 0.99}
	pAll := NewCapacityAwarePolicy(func(name string) (float64, bool) {
		v, ok := allSick[name]
		return v, ok
	})
	pAll.ObserveServedOutput("sick", 400, 10*time.Second)
	pAll.ObserveServedOutput("worse", 400, 10*time.Second)
	got, ok = pAll.Pick([]PlannerReplica{{Name: "worse"}, {Name: "sick"}}, nil, func(name string) int {
		if name == "sick" {
			return 1
		}
		return 9
	})
	if !ok || got.Name != "sick" {
		t.Fatalf("expected least-loaded last-resort placement when all vetoed, got %q (ok=%v)", got.Name, ok)
	}
}

// TestCapacityAwarePolicyUnknownCapacityBaseline pins the cold-start contract:
// a replica without observations is imputed at the cohort's mean learned
// capacity, so it competes on the same scale as proven replicas. Here the
// unproven replica wins only because its imputed capacity normalizes its load
// below the proven replica's cost — raw-load scoring would have picked the
// other way.
func TestCapacityAwarePolicyUnknownCapacityBaseline(t *testing.T) {
	p := NewCapacityAwarePolicy(nil)
	// Only "proven" has observations: 40 tok/s. "fresh" is unknown and is
	// imputed at the cohort baseline (40).
	p.ObserveServedOutput("proven", 400, 10*time.Second)

	// proven: load 4 / 40 = 0.1. fresh: load 1 / 40 (imputed) = 0.025.
	got, ok := p.Pick([]PlannerReplica{{Name: "proven"}, {Name: "fresh"}}, nil, func(name string) int {
		if name == "proven" {
			return 4
		}
		return 1
	})
	if !ok || got.Name != "fresh" {
		t.Fatalf("expected baseline-imputed unknown replica to beat raw-load intuition, got %q (ok=%v)", got.Name, ok)
	}

	// No observations anywhere: every score degrades to raw load, ties by name.
	pEmpty := NewCapacityAwarePolicy(nil)
	got, ok = pEmpty.Pick([]PlannerReplica{{Name: "zeta"}, {Name: "alpha"}}, nil, func(name string) int {
		if name == "zeta" {
			return 1
		}
		return 2
	})
	if !ok || got.Name != "zeta" {
		t.Fatalf("expected least-loaded placement with no learned capacities, got %q (ok=%v)", got.Name, ok)
	}
}

// TestCapacityAwarePolicyDeterministicTieBreak pins closed-form determinism:
// identical inputs yield the lexicographically smaller name, repeated picks
// agree, and an empty candidate set declines so the keyed rendezvous fallback
// takes over.
func TestCapacityAwarePolicyDeterministicTieBreak(t *testing.T) {
	p := NewCapacityAwarePolicy(nil)
	p.ObserveServedOutput("b", 200, 10*time.Second)
	p.ObserveServedOutput("a", 200, 10*time.Second)

	first, ok := p.Pick([]PlannerReplica{{Name: "b"}, {Name: "a"}}, nil, func(string) int { return 3 })
	if !ok || first.Name != "a" {
		t.Fatalf("expected name tiebreak winner, got %q (ok=%v)", first.Name, ok)
	}
	for i := 0; i < 16; i++ {
		again, ok := p.Pick([]PlannerReplica{{Name: "b"}, {Name: "a"}}, nil, func(string) int { return 3 })
		if !ok || again.Name != first.Name {
			t.Fatalf("pick %d drifted: got %q want %q", i, again.Name, first.Name)
		}
	}

	if _, ok := p.Pick(nil, nil, func(string) int { return 0 }); ok {
		t.Fatal("empty candidate set must decline (false) for the keyed fallback")
	}
}

// holdingPlanner blocks inside Complete until its gate closes, holding the
// dispatch's reservation (and thus its inflight load) for the call's duration.
type holdingPlanner struct {
	name    string
	gate    chan struct{}
	mu      sync.Mutex
	started []chan struct{}
	calls   int
}

func newHoldingPlanner(name string, expectedCalls int) *holdingPlanner {
	h := &holdingPlanner{name: name, gate: make(chan struct{})}
	h.started = make([]chan struct{}, expectedCalls)
	for i := range h.started {
		h.started[i] = make(chan struct{})
	}
	return h
}

func (h *holdingPlanner) waitCall(i int, t *testing.T) {
	t.Helper()
	select {
	case <-h.started[i]:
	case <-time.After(5 * time.Second):
		t.Fatalf("%s: call %d did not start within 5s", h.name, i)
	}
}

func (h *holdingPlanner) Model() string { return h.name }

func (h *holdingPlanner) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	h.mu.Lock()
	idx := h.calls
	h.calls++
	h.mu.Unlock()
	if idx < len(h.started) {
		close(h.started[idx])
	}
	<-h.gate
	// The planner answers with its own name so the test can witness which
	// replica served the turn (the dispatch does not stamp completions).
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: h.name}, Usage: agent.Usage{CompletionTokens: 10}}, nil
}

// TestReplicaDispatchCapacitySteeringEndToEnd pins the closed contract through
// the full dispatch path: with two turns held in flight (one per replica), the
// third turn must route by LEARNED CAPACITY-NORMALIZED load, not raw load or
// name. r0 is seeded slow (10 tok/s) and r1 fast (40 tok/s); both carry equal
// raw load, so only the borrowed load/capacity rule can pick r1 — a raw-load
// or name tiebreak would pick r0.
func TestReplicaDispatchCapacitySteeringEndToEnd(t *testing.T) {
	h0 := newHoldingPlanner("r0", 1)
	h1 := newHoldingPlanner("r1", 2)

	router, err := NewReplicaDispatch("fleet", []PlannerReplica{
		{Name: "r0", Planner: h0},
		{Name: "r1", Planner: h1},
	})
	if err != nil {
		t.Fatalf("NewReplicaDispatch: %v", err)
	}
	mem := NewFleetMembership(MembershipConfig{
		HealthyAfter:   1,
		UnhealthyAfter: 2,
		Probe:          func(context.Context, WorkerSpec) bool { return true },
	})
	for _, id := range []string{"r0", "r1"} {
		if err := mem.Add(WorkerSpec{ID: id, Endpoint: id}); err != nil {
			t.Fatalf("Add(%s): %v", id, err)
		}
	}
	p := NewCapacityAwarePolicy(nil)
	router = router.WithMembership(mem).WithPickPolicy(p)

	ctx := context.Background()
	mem.ProbeOnce(ctx)

	// Seed the learned capacities: r0 slow, r1 fast.
	p.ObserveServedOutput("r0", 100, 10*time.Second) // 10 tok/s
	p.ObserveServedOutput("r1", 400, 10*time.Second) // 40 tok/s

	released := false
	release := func() {
		if !released {
			released = true
			close(h0.gate)
			close(h1.gate)
		}
	}
	defer release()

	msgs := []agent.Message{{Role: agent.RoleUser, Content: "hello"}}
	type result struct {
		replica string
		err     error
	}
	var g1, g2, g3 result
	var wg sync.WaitGroup
	wg.Add(3)

	go func() {
		defer wg.Done()
		comp, err := router.Complete(ctx, msgs, nil)
		if err != nil {
			g1 = result{err: err}
			return
		}
		g1 = result{replica: comp.Message.Content}
	}()
	h0.waitCall(0, t) // r0 now holds one in-flight reservation

	go func() {
		defer wg.Done()
		comp, err := router.Complete(ctx, msgs, nil)
		if err != nil {
			g2 = result{err: err}
			return
		}
		g2 = result{replica: comp.Message.Content}
	}()
	h1.waitCall(0, t) // r1 now holds one in-flight reservation too

	go func() {
		defer wg.Done()
		comp, err := router.Complete(ctx, msgs, nil)
		if err != nil {
			g3 = result{err: err}
			return
		}
		g3 = result{replica: comp.Message.Content}
	}()
	h1.waitCall(1, t) // the deciding call landed on r1

	release()
	wg.Wait()

	for i, r := range []result{g1, g2, g3} {
		if r.err != nil {
			t.Fatalf("turn %d: %v", i, r.err)
		}
	}
	if g1.replica != "r0" {
		t.Fatalf("turn 1: idle tie should land on r0 by name, got %q", g1.replica)
	}
	if g2.replica != "r1" {
		t.Fatalf("turn 2: idle replica should take the second turn, got %q", g2.replica)
	}
	if g3.replica != "r1" {
		t.Fatalf("turn 3: equal raw loads must route by learned capacity (r1 at 40 tok/s beats r0 at 10), got %q", g3.replica)
	}
}
