package gateway

import "testing"

func cacheAwareTestRouter(t *testing.T) *ReplicaDispatch {
	t.Helper()
	router, err := NewReplicaDispatch("fleet", []PlannerReplica{
		{Name: "w0", Planner: &replicaDispatchTestPlanner{name: "r0"}},
		{Name: "w1", Planner: &replicaDispatchTestPlanner{name: "r1"}},
		{Name: "w2", Planner: &replicaDispatchTestPlanner{name: "r2"}},
	})
	if err != nil {
		t.Fatalf("NewReplicaDispatch: %v", err)
	}
	return router
}

// Option off: pick() is exactly the HRW rendezvous.
func TestReplicaCacheAwareOffIsHRW(t *testing.T) {
	r := cacheAwareTestRouter(t)
	for _, p := range [][]string{nil, {"a"}, {"a", "b"}, {"x", "y", "z"}} {
		got, err := r.pick(p)
		want, werr := r.pickKeyed(p)
		if err != nil || werr != nil || got.Name != want.Name {
			t.Fatalf("prefix %v: pick=%q,%v want HRW %q,%v", p, got.Name, err, want.Name, werr)
		}
	}
}

// Option on: a matched prefix routes to its owner on every repeat.
func TestReplicaCacheAwareOnRoutesMatchedPrefixToOwner(t *testing.T) {
	r := cacheAwareTestRouter(t).WithCacheAwareRouting(SkewThreshold{})
	prefix := []string{"sys", "turn1"}
	first, err := r.pick(prefix)
	if err != nil {
		t.Fatalf("pick: %v", err)
	}
	for i := 0; i < 5; i++ {
		got, err := r.pick(append(append([]string{}, prefix...), "more"))
		if err != nil || got.Name != first.Name {
			t.Fatalf("repeat %d: got %q,%v want owner %q", i, got.Name, err, first.Name)
		}
	}
}

// Skew beyond both thresholds drops locality and falls back to least-loaded.
func TestReplicaCacheAwareSkewFallsBackToLeastLoaded(t *testing.T) {
	r := cacheAwareTestRouter(t)
	p := NewCacheAwarePolicy(NewPrefixResidencyIndex(8), DefaultSkewThreshold())
	r.WithPickPolicy(p)
	prefix := []string{"hot"}
	owner, _ := p.Pick(r.replicas, prefix, nil)
	load := func(name string) int {
		if name == owner.Name {
			return 100
		}
		return 1
	}
	got, ok := p.Pick(r.replicas, prefix, load)
	if !ok || got.Name == owner.Name {
		t.Fatalf("skewed owner %q should lose to least-loaded, got %q ok=%v", owner.Name, got.Name, ok)
	}
}
