package radixkv

import (
	"math"
	"testing"
)

func touchPure(t *Tree, req []int) {
	_, leaf := servePure(t, req)
	t.Done(leaf)
}

func TestCostAwareEvictionKeepsReusedSpan(t *testing.T) {
	a, b, c := distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)

	lru := New(20)
	touchPure(lru, a)
	touchPure(lru, a) // refresh recency before b arrives; LRU should still later evict a.
	touchPure(lru, b)
	touchPure(lru, c)

	if m := lru.MatchLen(a); m != 0 {
		t.Fatalf("LRU should evict the older hot span once it becomes least-recently-used, matched %d", m)
	}

	costAware := NewWithEvictionPolicy(20, EvictionCostAware)
	costAware.SetAdmissionEnabled(false) // isolate the eviction selector from #9311 admission
	touchPure(costAware, a)
	touchPure(costAware, a)
	touchPure(costAware, b)
	touchPure(costAware, c)

	if m := costAware.MatchLen(a); m != len(a) {
		t.Fatalf("cost-aware should keep reused a, matched %d/%d", m, len(a))
	}
	if m := costAware.MatchLen(b); m != 0 {
		t.Fatalf("cost-aware should evict one-shot b, matched %d", m)
	}
	if m := costAware.MatchLen(c); m != len(c) {
		t.Fatalf("newly inserted c should survive, matched %d/%d", m, len(c))
	}
}

func TestCostAwareEvictionStatsExposeVictimTelemetry(t *testing.T) {
	tree := NewWithEvictionPolicy(20, EvictionCostAware)
	tree.SetAdmissionEnabled(false) // this test exercises eviction telemetry, not admission
	a, b, c := distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)

	touchPure(tree, a)
	touchPure(tree, a)
	touchPure(tree, b)
	touchPure(tree, c)

	st := tree.Stats()
	if st.EvictionPolicy != "cost-aware" {
		t.Fatalf("EvictionPolicy=%q, want cost-aware", st.EvictionPolicy)
	}
	if st.Evictions != 1 || st.CostEvictions != 1 {
		t.Fatalf("evictions=%d costEvictions=%d, want 1/1", st.Evictions, st.CostEvictions)
	}
	if st.ReuseHits == 0 {
		t.Fatal("ReuseHits=0, want observed demand reuse to feed the victim rule")
	}
	if st.LastEvictPolicy != "cost-aware" {
		t.Fatalf("LastEvictPolicy=%q, want cost-aware", st.LastEvictPolicy)
	}
	if st.LastEvictCandidates != 3 || st.LastEvictLocked != 1 {
		t.Fatalf("last eviction candidates/locked=%d/%d, want 3/1", st.LastEvictCandidates, st.LastEvictLocked)
	}
	if st.LastEvictVictimHits != 0 {
		t.Fatalf("victim hits=%d, want one-shot victim", st.LastEvictVictimHits)
	}
	if st.LastEvictVictimTokens != len(b) || st.LastEvictVictimPrefix != len(b) {
		t.Fatalf("victim tokens/prefix=%d/%d, want %d/%d",
			st.LastEvictVictimTokens, st.LastEvictVictimPrefix, len(b), len(b))
	}
	if math.Abs(st.LastEvictVictimCost-1) > 1e-9 {
		t.Fatalf("victim cost=%v, want 1", st.LastEvictVictimCost)
	}
}

func TestCostAwareEvictionRespectsLeases(t *testing.T) {
	tree := NewWithEvictionPolicy(20, EvictionCostAware)
	tree.SetAdmissionEnabled(false) // preserve the insert-always setup needed to pressure leases
	a, b, c := distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10)

	_, leasedA := servePure(tree, a)
	touchPure(tree, b)
	_, leasedC := servePure(tree, c)
	tree.Done(leasedC)

	if m := tree.MatchLen(a); m != len(a) {
		t.Fatalf("leased a should survive cost-aware eviction, matched %d/%d", m, len(a))
	}
	if m := tree.MatchLen(b); m != 0 {
		t.Fatalf("unleased b should be evicted, matched %d", m)
	}
	if m := tree.MatchLen(c); m != len(c) {
		t.Fatalf("leased c should survive its insert-time eviction pass, matched %d/%d", m, len(c))
	}
	if st := tree.Stats(); st.LastEvictLocked != 2 {
		t.Fatalf("last eviction locked=%d, want 2 (a and c)", st.LastEvictLocked)
	}

	tree.Done(leasedA)
}

func TestSetEvictionPolicyUnknownFallsBackToLRU(t *testing.T) {
	tree := NewWithEvictionPolicy(0, EvictionPolicy(99))
	if got := tree.Stats().EvictionPolicy; got != "lru" {
		t.Fatalf("unknown policy = %q, want lru fallback", got)
	}
}
