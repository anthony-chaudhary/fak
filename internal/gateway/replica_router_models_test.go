package gateway

import (
	"context"
	"errors"
	"testing"
)

// Issue #5635, router half. newProxyPlanner hands the SAME model string to every
// configured base URL, so the router's placement path had no way to keep a request
// off a replica serving something else. With membership carrying per-worker Models
// the router's candidate set is filtered by model BEFORE health, and the two empty
// outcomes stay typed apart.

// modelFleet builds a membership whose worker ids match replica names, each labeled
// with the model it holds, all probed healthy.
func modelFleet(t *testing.T, holdings map[string][]string) *FleetMembership {
	t.Helper()
	m := NewFleetMembership(MembershipConfig{
		HealthyAfter: 1, UnhealthyAfter: 2,
		Probe: func(context.Context, WorkerSpec) bool { return true },
	})
	// Deterministic registration order: glm-w before qwen-w, matching the replica
	// order the tests declare.
	for _, id := range []string{"glm-w", "qwen-w"} {
		if models, ok := holdings[id]; ok {
			mustAdd(t, m, WorkerSpec{ID: id, Endpoint: id, Models: models})
		}
	}
	m.ProbeOnce(context.Background())
	return m
}

func glmQwenRouter(t *testing.T, model string) (*ReplicaRouter, *replicaRouterTestPlanner, *replicaRouterTestPlanner) {
	t.Helper()
	glm := &replicaRouterTestPlanner{name: "glm-upstream"}
	qwen := &replicaRouterTestPlanner{name: "qwen-upstream"}
	router, err := NewReplicaRouter(model, []PlannerReplica{
		{Name: "glm-w", Planner: glm},
		{Name: "qwen-w", Planner: qwen},
	})
	if err != nil {
		t.Fatalf("NewReplicaRouter(%s): %v", model, err)
	}
	return router, glm, qwen
}

// TestReplicaRouterRoutesOnlyToWorkersHoldingItsModel is the router-level form of the
// bug: a heterogeneous fleet must never serve this router's model from the worker
// holding the other one.
func TestReplicaRouterRoutesOnlyToWorkersHoldingItsModel(t *testing.T) {
	router, glm, qwen := glmQwenRouter(t, "glm-4.6")
	router.WithMembership(modelFleet(t, map[string][]string{
		"glm-w":  {"glm-4.6"},
		"qwen-w": {"qwen3-32b"},
	}))

	got := dispatchCounts(t, router, 6)
	if got["qwen-upstream"] != 0 {
		t.Fatalf("router for glm-4.6 dialed the qwen upstream %d time(s): %v", got["qwen-upstream"], got)
	}
	if got["glm-upstream"] != 6 {
		t.Fatalf("dispatch counts = %v, want all 6 on glm-upstream", got)
	}
	if c, _ := qwen.counts(); c != 0 {
		t.Fatalf("qwen planner Complete count = %d, want 0", c)
	}
	if c, _ := glm.counts(); c != 6 {
		t.Fatalf("glm planner Complete count = %d, want 6", c)
	}
}

// TestReplicaRouterModelVerdictIsDistinctFromOutage pins that the router propagates
// membership's two verdicts unmerged — the single most expensive confusion in this
// area is a configuration mistake that reads as a fleet outage.
func TestReplicaRouterModelVerdictIsDistinctFromOutage(t *testing.T) {
	ctx := context.Background()

	t.Run("no worker holds the router model", func(t *testing.T) {
		router, glm, qwen := glmQwenRouter(t, "llama-3.1-70b")
		router.WithMembership(modelFleet(t, map[string][]string{
			"glm-w":  {"glm-4.6"},
			"qwen-w": {"qwen3-32b"},
		}))
		_, err := router.Complete(ctx, nil, nil)
		if !errors.Is(err, ErrNoWorkerForModel) {
			t.Fatalf("Complete err = %v, want ErrNoWorkerForModel", err)
		}
		if errors.Is(err, ErrNoHealthyWorker) {
			t.Fatalf("router merged the configuration verdict into the outage verdict: %v", err)
		}
		// Nothing was dialed on the way to that verdict.
		if c, _ := glm.counts(); c != 0 {
			t.Fatalf("glm planner dialed %d time(s) for an unheld model", c)
		}
		if c, _ := qwen.counts(); c != 0 {
			t.Fatalf("qwen planner dialed %d time(s) for an unheld model", c)
		}
	})

	t.Run("the holder is unhealthy", func(t *testing.T) {
		sp := newScriptedProbe()
		mem := NewFleetMembership(MembershipConfig{HealthyAfter: 1, UnhealthyAfter: 1, Probe: sp.probe})
		mustAdd(t, mem, WorkerSpec{ID: "glm-w", Endpoint: "glm-w", Models: []string{"glm-4.6"}})
		mustAdd(t, mem, WorkerSpec{ID: "qwen-w", Endpoint: "qwen-w", Models: []string{"qwen3-32b"}})
		mem.ProbeOnce(ctx)
		router, _, qwen := glmQwenRouter(t, "glm-4.6")
		router.WithMembership(mem)

		sp.set("glm-w", false)
		mem.ProbeOnce(ctx)

		_, err := router.Complete(ctx, nil, nil)
		if !errors.Is(err, ErrNoHealthyWorker) {
			t.Fatalf("holder down: Complete err = %v, want ErrNoHealthyWorker", err)
		}
		if errors.Is(err, ErrNoWorkerForModel) {
			t.Fatalf("an outage was typed as a configuration fault: %v", err)
		}
		// And it did NOT quietly fall through to the healthy wrong-model upstream.
		if c, _ := qwen.counts(); c != 0 {
			t.Fatalf("router fell through to the qwen upstream %d time(s)", c)
		}
	})
}

// TestReplicaRouterUnlabeledMembershipIsUnchanged is the compatibility arm at the
// router seam: membership registered the pre-labeling way (no Models) keeps the exact
// health-gated round-robin the router has always had.
func TestReplicaRouterUnlabeledMembershipIsUnchanged(t *testing.T) {
	router, _, _ := glmQwenRouter(t, "fleet")
	router.WithMembership(modelFleet(t, map[string][]string{
		"glm-w":  nil,
		"qwen-w": nil,
	}))
	got := dispatchCounts(t, router, 4)
	if got["glm-upstream"] != 2 || got["qwen-upstream"] != 2 {
		t.Fatalf("unlabeled membership round-robin = %v, want 2 each", got)
	}
}

// pickFirstPolicy is a minimal PickPolicy that always takes the first candidate the
// router offers it — enough to prove the model filter narrows the CANDIDATE SET the
// policy sees, not just the round-robin fallback.
type pickFirstPolicy struct{ saw [][]string }

func (p *pickFirstPolicy) Pick(candidates []PlannerReplica, _ []string, _ func(string) int) (PlannerReplica, bool) {
	var names []string
	for _, c := range candidates {
		names = append(names, c.Name)
	}
	p.saw = append(p.saw, names)
	if len(candidates) == 0 {
		return PlannerReplica{}, false
	}
	return candidates[0], true
}

// TestReplicaRouterPolicyPathIsModelFiltered covers the second placement path: a
// cache-aware policy must be offered only candidates that hold the router's model,
// and must never be consulted at all when nothing holds it.
func TestReplicaRouterPolicyPathIsModelFiltered(t *testing.T) {
	ctx := context.Background()
	holdings := map[string][]string{"glm-w": {"glm-4.6"}, "qwen-w": {"qwen3-32b"}}

	router, _, qwen := glmQwenRouter(t, "glm-4.6")
	pol := &pickFirstPolicy{}
	router.WithMembership(modelFleet(t, holdings)).WithPickPolicy(pol)
	if _, err := router.Complete(ctx, nil, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(pol.saw) != 1 || len(pol.saw[0]) != 1 || pol.saw[0][0] != "glm-w" {
		t.Fatalf("policy saw candidates %v, want exactly [[glm-w]]", pol.saw)
	}
	if c, _ := qwen.counts(); c != 0 {
		t.Fatalf("policy path dialed the qwen upstream %d time(s)", c)
	}

	// An unheld model short-circuits to the typed verdict before the policy runs.
	unheld, _, _ := glmQwenRouter(t, "llama-3.1-70b")
	pol2 := &pickFirstPolicy{}
	unheld.WithMembership(modelFleet(t, holdings)).WithPickPolicy(pol2)
	if _, err := unheld.Complete(ctx, nil, nil); !errors.Is(err, ErrNoWorkerForModel) {
		t.Fatalf("policy path unheld model: err = %v, want ErrNoWorkerForModel", err)
	}
	if len(pol2.saw) != 0 {
		t.Fatalf("policy was consulted %d time(s) for a model no worker holds", len(pol2.saw))
	}
}
