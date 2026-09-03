package gateway

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestPrefixResidencyIndexFoldsTrackAEventStream is the AC-1 witness: the residency
// index aggregates per-worker resident-prefix signal from a synthetic external-engine
// KV-cache-event stream (the Track-A path — vLLM KV-events / SGLang radix signal lower
// into KVCacheEvent), including evictions and the per-worker LRU bound.
func TestPrefixResidencyIndexFoldsTrackAEventStream(t *testing.T) {
	idx := NewPrefixResidencyIndex(2) // each worker retains 2 distinct resident prefixes
	p1 := []string{"A", "B", "C"}
	p2 := []string{"A", "B", "D"}
	p3 := []string{"X"}

	// A synthetic stream as an external engine would emit it.
	stream := []KVCacheEvent{
		{Worker: "w0", Op: ResidentAdd, Prefix: p1},
		{Worker: "w1", Op: ResidentAdd, Prefix: p1},
	}
	for _, ev := range stream {
		idx.Ingest(ev)
	}

	// Longest resident leading run, not just exact membership.
	if got := idx.Overlap("w0", []string{"A", "B", "C", "E"}); got != 3 {
		t.Errorf("Overlap(w0, ABCE) = %d, want 3", got)
	}
	if got := idx.Overlap("w0", p2); got != 2 { // shares A,B with p1
		t.Errorf("Overlap(w0, ABD) = %d, want 2", got)
	}
	if got := idx.Overlap("w2", p1); got != 0 { // an unseen worker holds nothing
		t.Errorf("Overlap(unseen, ABC) = %d, want 0", got)
	}

	// Admit p2 on w0 (held = {p1,p2}), then p3 evicts the LRU (p1) past capacity 2.
	idx.Ingest(KVCacheEvent{Worker: "w0", Op: ResidentAdd, Prefix: p2})
	if got := idx.Overlap("w0", p2); got != 3 {
		t.Errorf("Overlap(w0, ABD) after admit = %d, want 3", got)
	}
	idx.Ingest(KVCacheEvent{Worker: "w0", Op: ResidentAdd, Prefix: p3})
	if got := idx.Occupancy("w0"); got != 2 {
		t.Fatalf("Occupancy(w0) = %d, want 2 (LRU bound)", got)
	}
	if got := idx.Overlap("w0", p1); got != 2 { // p1 evicted; p2 still shares A,B
		t.Errorf("Overlap(w0, ABC) after LRU evict of p1 = %d, want 2", got)
	}
	if got := idx.Overlap("w0", p3); got != 1 {
		t.Errorf("Overlap(w0, X) = %d, want 1", got)
	}

	// An explicit eviction event removes the span from the routing view.
	idx.Ingest(KVCacheEvent{Worker: "w0", Op: ResidentDrop, Prefix: p2})
	if got := idx.Overlap("w0", p2); got != 0 {
		t.Errorf("Overlap(w0, ABD) after ResidentDrop = %d, want 0", got)
	}
}

type fakeResidentSource struct{ prefixes [][]string }

func (f fakeResidentSource) ResidentPrefixes() [][]string { return f.prefixes }

// TestPrefixResidencyIndexTrackBEmitter is the AC-5 witness: the native (Track-B)
// emitter path — resident prefixes pulled from a native source (internal/radixkv via
// the ResidentPrefixSource seam) — folds into the SAME index schema as Track A.
func TestPrefixResidencyIndexTrackBEmitter(t *testing.T) {
	idx := NewPrefixResidencyIndex(8)
	src := fakeResidentSource{prefixes: [][]string{
		{"sys", "tools", "turn1"},
		{"sys", "tools", "turn2"},
		{}, // empty is skipped
	}}
	if n := idx.IngestResidentPrefixes("native-w", src); n != 2 {
		t.Fatalf("IngestResidentPrefixes admitted %d, want 2", n)
	}
	if got := idx.Overlap("native-w", []string{"sys", "tools", "turn1", "x"}); got != 3 {
		t.Errorf("Overlap(native-w) = %d, want 3", got)
	}
	if n := idx.IngestResidentPrefixes("native-w", nil); n != 0 {
		t.Errorf("nil source admitted %d, want 0", n)
	}
}

// TestResidencyViewContractForPDSeed is the AC-4 witness: the index satisfies the
// documented ResidencyView contract the P/D-orchestration seed consumes, ranking the
// candidate workers best-overlap-first so the orchestrator can co-locate decode with
// the replica holding the prefill KV.
func TestResidencyViewContractForPDSeed(t *testing.T) {
	var view ResidencyView = NewPrefixResidencyIndex(8) // compile-time contract check
	idx := view.(*PrefixResidencyIndex)
	idx.Observe("w0", []string{"A", "B"})
	idx.Observe("w1", []string{"A", "B", "C", "D"})
	idx.Observe("w2", []string{"Z"})

	got := view.ResidentWorkers([]string{"A", "B", "C", "E"})
	want := []ResidentWorker{
		{Worker: "w1", Overlap: 3}, // A,B,C
		{Worker: "w0", Overlap: 2}, // A,B
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ResidentWorkers = %+v, want %+v", got, want)
	}
}

// TestCacheAwareRoutingBeatsRoundRobin is the AC-2 witness: power-of-two-choices
// cache-aware routing MEASURABLY beats round-robin on prefix-cache hit rate on the
// shared-prefix workload — a number from a measured run, not an asserted one. The
// pinned values keep the committed witness reproducible (the same isolation the
// existing TestKVAwareFleetRoutingHitRate gives the FleetCacheRouter row).
func TestCacheAwareRoutingBeatsRoundRobin(t *testing.T) {
	res := MeasureCacheAwareRouting(DefaultKVFleetWorkload)

	const (
		wantRequests = 464
		wantRRHit    = 0.8512931034482759 // cache-blind round-robin
		wantCAHit    = 0.9396551724137931 // cache-aware power-of-two
		wantLift     = 1.1037974683544303 // cache-aware / round-robin
		eps          = 1e-9
	)
	if res.Requests != wantRequests {
		t.Fatalf("requests = %d, want %d (the standing workload stream)", res.Requests, wantRequests)
	}
	// Pin the deterministic measured rates so the committed witness cannot silently
	// drift from the policy that produced them.
	if d := res.RoundRobinHitRate - wantRRHit; d > eps || d < -eps {
		t.Errorf("round-robin hit-rate = %.16f, want %.16f", res.RoundRobinHitRate, wantRRHit)
	}
	if d := res.CacheAwareHitRate - wantCAHit; d > eps || d < -eps {
		t.Errorf("cache-aware hit-rate = %.16f, want %.16f", res.CacheAwareHitRate, wantCAHit)
	}
	if d := res.HitRateLift - wantLift; d > eps || d < -eps {
		t.Errorf("hit-rate lift = %.16f, want %.16f", res.HitRateLift, wantLift)
	}
	// The whole point: cache-aware placement must beat the cache-blind baseline on the
	// same stream and same residency mechanics, so the lift isolates the PICK policy.
	if res.CacheAwareHitRate <= res.RoundRobinHitRate {
		t.Errorf("cache-aware hit-rate %.4f must exceed round-robin %.4f",
			res.CacheAwareHitRate, res.RoundRobinHitRate)
	}
	if res.HitRateLift <= 1.0 {
		t.Errorf("hit-rate lift = %.4f, want > 1.0", res.HitRateLift)
	}
	b, _ := json.MarshalIndent(res, "", "  ")
	t.Logf("measured cache-aware vs round-robin:\n%s", b)
}

// TestCacheAwareSkewFallbackAvoidsHerding is the AC-3 witness: past the documented skew
// threshold the policy falls back to plain load-balancing, so a hot shared prefix's
// traffic spreads instead of herding onto the one replica that first cached it. The
// contrast arm (skew effectively disabled) shows that locality scoring ALONE herds —
// which is exactly why the balancing-threshold fallback exists.
func TestCacheAwareSkewFallbackAvoidsHerding(t *testing.T) {
	workers := []string{"w0", "w1", "w2"}
	hot := []string{"hot-shared-prefix"}
	const n = 30

	// route runs n requests for the hot prefix through a policy, modelling each pick as
	// adding one unit of in-flight load to the chosen worker (an external load source
	// the live router reads from membership), and returns the per-worker pick counts.
	route := func(p *CacheAwarePolicy) map[string]int {
		picks := map[string]int{}
		ext := func(name string) int { return picks[name] }
		for i := 0; i < n; i++ {
			chosen, _, ok := p.pickWorker(workers, hot, ext)
			if !ok {
				t.Fatalf("pickWorker returned ok=false")
			}
			picks[chosen]++
		}
		return picks
	}

	// Skew fallback active (tight threshold): a 2-request load gap trips the balancing
	// fallback, so traffic must spread across more than one worker.
	withSkew := route(NewCacheAwarePolicy(NewPrefixResidencyIndex(8), SkewThreshold{AbsLoad: 2, RelLoad: 1.0}))
	spread := 0
	for _, c := range withSkew {
		if c > 0 {
			spread++
		}
	}
	if spread < 2 {
		t.Errorf("skew fallback should spread the hot prefix across >=2 workers, got %v", withSkew)
	}
	if withSkew["w0"] == n {
		t.Errorf("skew fallback still herded all %d requests onto w0: %v", n, withSkew)
	}

	// Skew fallback effectively disabled (unreachable threshold): locality scoring alone
	// herds every recurrence onto the first holder — the failure mode the fallback fixes.
	noSkew := route(NewCacheAwarePolicy(NewPrefixResidencyIndex(8), SkewThreshold{AbsLoad: 1 << 30, RelLoad: 1 << 30}))
	if noSkew["w0"] != n {
		t.Errorf("without skew fallback the hot prefix should herd onto w0, got %v", noSkew)
	}
}

// TestReplicaRouterCacheAwarePolicyPinsByResidency is the AC-6 witness: the index +
// scorer are reachable from the live ReplicaRouter skeleton (build-on #45) — attaching
// a CacheAwarePolicy makes pick() route each distinct request prefix to a stable home
// replica (residency locality) rather than blind round-robin, while distinct prefixes
// land on distinct homes (load spreading).
func TestReplicaRouterCacheAwarePolicyPinsByResidency(t *testing.T) {
	a := &replicaRouterTestPlanner{name: "r1"}
	b := &replicaRouterTestPlanner{name: "r2"}
	c := &replicaRouterTestPlanner{name: "r3"}
	router, err := NewReplicaRouter("fleet", []PlannerReplica{
		{Name: "w0", Planner: a},
		{Name: "w1", Planner: b},
		{Name: "w2", Planner: c},
	})
	if err != nil {
		t.Fatalf("NewReplicaRouter: %v", err)
	}
	router.WithPickPolicy(NewCacheAwarePolicy(NewPrefixResidencyIndex(8), DefaultSkewThreshold()))

	convoA := []agent.Message{{Role: agent.RoleSystem, Content: "shared scaffold A"}, {Role: agent.RoleUser, Content: "alpha"}}
	convoB := []agent.Message{{Role: agent.RoleSystem, Content: "shared scaffold B"}, {Role: agent.RoleUser, Content: "beta"}}

	served := func(messages []agent.Message) string {
		comp, err := router.Complete(context.Background(), messages, nil)
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		return comp.Message.Content
	}

	// First touch of each conversation homes it; every later touch must return to the
	// SAME replica because that replica now holds the prefix KV.
	homeA := served(convoA)
	homeB := served(convoB)
	if homeA == homeB {
		t.Fatalf("distinct prefixes should land on distinct homes; both got %q", homeA)
	}
	for i := 0; i < 5; i++ {
		if got := served(convoA); got != homeA {
			t.Fatalf("convoA touch %d routed to %q, want stable home %q", i, got, homeA)
		}
		if got := served(convoB); got != homeB {
			t.Fatalf("convoB touch %d routed to %q, want stable home %q", i, got, homeB)
		}
	}
}

func TestCacheAwarePolicyTierWeightedLiveRouting(t *testing.T) {
	idx := NewPrefixResidencyIndex(8)
	prefix := []string{"a", "b"}
	idx.ObserveTier("device", prefix, TierDevice)
	idx.ObserveTier("host", prefix, TierHost)
	idx.ObserveTier("disk", prefix, TierDisk)
	p := NewCacheAwarePolicy(idx, DefaultSkewThreshold()).WithTierWeightedOverlap(true, DefaultTierWeights())
	if got := p.localityTarget([]string{"disk", "host", "device"}, prefix, nil); got != "device" {
		t.Fatalf("tier route=%q, want device", got)
	}
	d := p.TierContribution("device", prefix, nil)
	h := p.TierContribution("host", prefix, nil)
	k := p.TierContribution("disk", prefix, nil)
	if !(d.Score > h.Score && h.Score > k.Score) {
		t.Fatalf("scores device=%v host=%v disk=%v", d.Score, h.Score, k.Score)
	}
	if d.Engine != "fak_native" || d.Model != "Qwen3.8" || !d.Enabled {
		t.Fatalf("receipt=%+v", d)
	}
}

func TestCacheAwarePolicyTierUnknownZeroAndRollback(t *testing.T) {
	idx := NewPrefixResidencyIndex(8)
	prefix := []string{"shared"}
	idx.Ingest(KVCacheEvent{Worker: "unknown", Prefix: prefix, Tier: OverlapTier(99), Op: ResidentAdd})
	idx.ObserveTier("host", prefix, TierHost)
	zero := DefaultTierWeights()
	zero.Host = 0
	p := NewCacheAwarePolicy(idx, DefaultSkewThreshold()).WithTierWeightedOverlap(true, zero)
	if got := p.TierContribution("unknown", prefix, nil); got.Credit != 0 || got.Score != 0 {
		t.Fatalf("unknown credited: %+v", got)
	}
	if got := p.TierContribution("host", prefix, nil); got.Credit != 0 || got.Score != 0 {
		t.Fatalf("zero tier credited: %+v", got)
	}
	p.WithTierWeightedOverlap(false, zero)
	if got := p.TierContribution("unknown", prefix, nil); got.Credit != 1 || got.Score <= 0 || got.Enabled {
		t.Fatalf("rollback=%+v", got)
	}
}

func TestPrefixResidencyTierRemovedWithPrefix(t *testing.T) {
	idx := NewPrefixResidencyIndex(8)
	p := []string{"a", "b"}
	idx.ObserveTier("w", p, TierDevice)
	if got := idx.TierOverlap("w", p); len(got) != 1 || got[0].Blocks != 2 {
		t.Fatalf("held=%+v", got)
	}
	idx.Ingest(KVCacheEvent{Worker: "w", Prefix: p, Op: ResidentDrop})
	if got := idx.TierOverlap("w", p); len(got) != 0 {
		t.Fatalf("stale tier=%+v", got)
	}
}

func TestSessionAffinityStickyMultiTurn(t *testing.T) {
	idx := NewPrefixResidencyIndex(8)
	idx.Observe("w0", []string{"common", "prefix"})
	p := NewCacheAwarePolicy(idx, DefaultSkewThreshold())

	candidates := []PlannerReplica{
		{Name: "w0"},
		{Name: "w1"},
		{Name: "w2"},
	}

	// Turn 1: routes to w0 due to prefix overlap.
	chosen, ok := p.PickWithAffinity(candidates, []string{"common", "prefix", "turn1"}, "sess-1", nil)
	if !ok || chosen.Name != "w0" {
		t.Fatalf("turn 1 pick = %q, ok = %v, want w0", chosen.Name, ok)
	}

	// Verify session affinity is recorded.
	aff, ok := p.GetAffinity("sess-1")
	if !ok || aff != "w0" {
		t.Fatalf("session affinity = %q, ok = %v, want w0", aff, ok)
	}

	// Now seed w1 with another prefix so w1 would normally win for that prefix.
	idx.Observe("w1", []string{"other", "prefix"})

	// Turn 2 with sess-1: should stick to w0 even though w1 has better overlap for this prefix.
	chosen2, ok := p.PickWithAffinity(candidates, []string{"other", "prefix"}, "sess-1", nil)
	if !ok || chosen2.Name != "w0" {
		t.Fatalf("turn 2 pick with session = %q, want sticky w0", chosen2.Name)
	}
}

func TestSessionAffinityExpiration(t *testing.T) {
	simTime := time.Unix(1000, 0)
	p := NewCacheAwarePolicy(nil, DefaultSkewThreshold()).
		WithClock(func() time.Time { return simTime }).
		WithAffinityTTL(2 * time.Minute)

	candidates := []PlannerReplica{
		{Name: "w0"},
		{Name: "w1"},
	}

	// Seed w0 so it is picked first.
	p.Index().Observe("w0", []string{"prompt"})
	chosen, ok := p.PickWithAffinity(candidates, []string{"prompt"}, "sess-exp", nil)
	if !ok || chosen.Name != "w0" {
		t.Fatalf("initial pick = %q, want w0", chosen.Name)
	}

	if aff, ok := p.GetAffinity("sess-exp"); !ok || aff != "w0" {
		t.Fatalf("affinity = %q, want w0", aff)
	}

	// Advance time past TTL (2 minutes).
	simTime = simTime.Add(3 * time.Minute)

	// Now seed w1 with longer overlap for new request.
	p.Index().Observe("w1", []string{"prompt", "extension"})

	// Affinity expired, so it should re-evaluate and choose w1.
	chosen2, ok := p.PickWithAffinity(candidates, []string{"prompt", "extension"}, "sess-exp", nil)
	if !ok || chosen2.Name != "w1" {
		t.Fatalf("post-expiry pick = %q, want w1", chosen2.Name)
	}

	// Affinity should be updated to w1.
	if aff, ok := p.GetAffinity("sess-exp"); !ok || aff != "w1" {
		t.Fatalf("updated affinity = %q, want w1", aff)
	}
}

func TestSessionAffinityFallbackOnUnavailable(t *testing.T) {
	p := NewCacheAwarePolicy(nil, DefaultSkewThreshold())
	p.SetAffinity("sess-avail", "w0")

	// Pinned worker w0 is removed from candidates (unavailable).
	candidates := []PlannerReplica{
		{Name: "w1"},
		{Name: "w2"},
	}

	chosen, ok := p.PickWithAffinity(candidates, []string{"prompt"}, "sess-avail", nil)
	if !ok {
		t.Fatalf("pick failed on fallback candidates")
	}
	if chosen.Name != "w1" && chosen.Name != "w2" {
		t.Fatalf("chosen = %q, want w1 or w2", chosen.Name)
	}

	// Affinity is updated to the newly chosen available worker.
	if aff, ok := p.GetAffinity("sess-avail"); !ok || aff != chosen.Name {
		t.Fatalf("affinity after fallback = %q, want %q", aff, chosen.Name)
	}

	// When w0 returns, the session sticks to the updated affinity.
	allCandidates := []PlannerReplica{
		{Name: "w0"},
		{Name: "w1"},
		{Name: "w2"},
	}
	chosen2, ok := p.PickWithAffinity(allCandidates, []string{"prompt"}, "sess-avail", nil)
	if !ok || chosen2.Name != chosen.Name {
		t.Fatalf("subsequent pick = %q, want %q", chosen2.Name, chosen.Name)
	}
}

func TestSessionAffinityIndependentSessions(t *testing.T) {
	p := NewCacheAwarePolicy(nil, DefaultSkewThreshold())
	candidates := []PlannerReplica{
		{Name: "w0"},
		{Name: "w1"},
	}

	p.SetAffinity("sess-1", "w0")
	p.SetAffinity("sess-2", "w1")

	c1, ok1 := p.PickWithAffinity(candidates, []string{"p1"}, "sess-1", nil)
	c2, ok2 := p.PickWithAffinity(candidates, []string{"p2"}, "sess-2", nil)
	if !ok1 || c1.Name != "w0" {
		t.Fatalf("sess-1 pick = %q, want w0", c1.Name)
	}
	if !ok2 || c2.Name != "w1" {
		t.Fatalf("sess-2 pick = %q, want w1", c2.Name)
	}

	// Clear only sess-1.
	p.ClearAffinity("sess-1")
	if _, ok := p.GetAffinity("sess-1"); ok {
		t.Fatalf("sess-1 should be cleared")
	}
	if aff, ok := p.GetAffinity("sess-2"); !ok || aff != "w1" {
		t.Fatalf("sess-2 affinity = %q, want w1", aff)
	}
}

func TestSessionAffinityEmptySessionID(t *testing.T) {
	p := NewCacheAwarePolicy(nil, DefaultSkewThreshold())
	candidates := []PlannerReplica{
		{Name: "w0"},
		{Name: "w1"},
	}

	chosen, ok := p.Pick(candidates, []string{"unkeyed"}, nil)
	if !ok {
		t.Fatalf("Pick failed")
	}
	if chosen.Name == "" {
		t.Fatalf("chosen replica has empty name")
	}

	if _, ok := p.GetAffinity(""); ok {
		t.Fatalf("GetAffinity(\"\") returned true")
	}
}

func TestSessionAffinitySeverelySkewedFallback(t *testing.T) {
	p := NewCacheAwarePolicy(nil, SkewThreshold{AbsLoad: 4, RelLoad: 1.5})
	candidates := []PlannerReplica{
		{Name: "w0"},
		{Name: "w1"},
	}
	p.SetAffinity("sess-skew", "w0")

	// w0 is severely overloaded (load 20 vs 0).
	loadFunc := func(name string) int {
		if name == "w0" {
			return 20
		}
		return 0
	}

	chosen, ok := p.PickWithAffinity(candidates, []string{"prompt"}, "sess-skew", loadFunc)
	if !ok || chosen.Name != "w1" {
		t.Fatalf("severely skewed affinity target should fall back to w1, got %q", chosen.Name)
	}

	// Affinity should now be updated to w1.
	if aff, ok := p.GetAffinity("sess-skew"); !ok || aff != "w1" {
		t.Fatalf("affinity after skew fallback = %q, want w1", aff)
	}
}
