package radixkv

import "testing"

// TestSLRUScanResistanceProtectsHotPrefix is #3890's end-to-end acceptance witness for the
// open eviction seam's first new policy: SGLang's scan-resistant SLRU (SLRUStrategy,
// comparator (is_protected = hit_count >= 2, last_access_time)). The invariant proven here
// mirrors SGLang's test_radix_cache_slru_accuracy: a probationary node ALWAYS evicts before
// any protected node REGARDLESS of recency, so a frequently-hit prefix survives a burst of
// one-off cold inserts (scan resistance) — and it survives even though it is the OLDEST
// resident span, the exact case pure LRU gets wrong.
func TestSLRUScanResistanceProtectsHotPrefix(t *testing.T) {
	// Budget = 20 tokens = room for two 10-token spans at a time.
	hot, c1, c2, c3 := distinctReq(0, 10), distinctReq(1, 10), distinctReq(2, 10), distinctReq(3, 10)

	slru, err := NewWithEvictionStrategy(20, "slru")
	if err != nil {
		t.Fatalf("NewWithEvictionStrategy(slru): %v", err)
	}
	// Warm the hot prefix past SLRU's protection bar (hit_count >= 2): the first serve
	// inserts it (0 hits), the next two are demand reuses that each bump its hit count.
	touchPure(slru, hot)
	touchPure(slru, hot) // hits -> 1
	touchPure(slru, hot) // hits -> 2 : now PROTECTED
	if m := slru.MatchLen(hot); m != len(hot) {
		t.Fatalf("hot prefix should be resident before the scan, matched %d/%d", m, len(hot))
	}

	// Scan: a burst of one-off cold inserts, each newer than the hot prefix. Under pure LRU
	// the hot prefix — now the OLDEST span — would be the victim; SLRU keeps it because a
	// probationary span always precedes a protected one.
	touchPure(slru, c1) // hot + c1 = 20 tokens, fits, no eviction
	touchPure(slru, c2) // 30 > 20 -> evict a probationary span (c1), not protected hot
	touchPure(slru, c3) // evict c2; hot still protected

	if m := slru.MatchLen(hot); m != len(hot) {
		t.Fatalf("SLRU must protect the frequently-hit prefix through a cold scan, matched %d/%d", m, len(hot))
	}
	if m := slru.MatchLen(c1); m != 0 {
		t.Fatalf("probationary c1 should have been evicted first, matched %d", m)
	}
	if m := slru.MatchLen(c2); m != 0 {
		t.Fatalf("probationary c2 should have been evicted, matched %d", m)
	}
	if m := slru.MatchLen(c3); m != len(c3) {
		t.Fatalf("most-recent cold insert c3 should be resident, matched %d/%d", m, len(c3))
	}
	if st := slru.Stats(); st.EvictionPolicy != "slru" {
		t.Fatalf("EvictionPolicy=%q, want slru", st.EvictionPolicy)
	}

	// Contrast: pure LRU flushes the hot prefix on the SAME workload (the scan the SLRU
	// segment exists to resist), proving the two policies genuinely diverge here.
	lru := New(20)
	touchPure(lru, hot)
	touchPure(lru, hot)
	touchPure(lru, hot)
	touchPure(lru, c1)
	touchPure(lru, c2)
	touchPure(lru, c3)
	if m := lru.MatchLen(hot); m != 0 {
		t.Fatalf("pure LRU should evict the oldest (hot) span under the cold scan, matched %d/%d", m, len(hot))
	}
}

// TestGetEvictionStrategyFailsClosed proves the registry rejects an unknown policy name
// (SGLang's get_eviction_strategy raise) rather than silently widening to a default — the
// fail-closed operator-knob contract, and it leaves the prior strategy in place.
func TestGetEvictionStrategyFailsClosed(t *testing.T) {
	if _, err := NewWithEvictionStrategy(0, "nope"); err == nil {
		t.Fatal("unknown eviction strategy name must fail closed, got nil error")
	}
	tree := New(20)
	if err := tree.SetEvictionStrategy("slru"); err != nil {
		t.Fatalf("SetEvictionStrategy(slru): %v", err)
	}
	if err := tree.SetEvictionStrategy("bogus"); err == nil {
		t.Fatal("SetEvictionStrategy(bogus) must return an error")
	}
	if got := tree.Stats().EvictionPolicy; got != "slru" {
		t.Fatalf("rejected name must not change the active policy, EvictionPolicy=%q want slru", got)
	}
}

// TestEvictionSeedStrategiesUnchanged pins that routing lru / cost-aware through the open seam
// reports the same policy names the closed enum did — the behavior-unchanged guarantee for the
// two seed strategies.
func TestEvictionSeedStrategiesUnchanged(t *testing.T) {
	if got := New(20).Stats().EvictionPolicy; got != "lru" {
		t.Fatalf("bare New tree EvictionPolicy=%q, want lru", got)
	}
	if got := NewWithEvictionPolicy(20, EvictionCostAware).Stats().EvictionPolicy; got != "cost-aware" {
		t.Fatalf("cost-aware EvictionPolicy=%q, want cost-aware", got)
	}
	lru, err := NewWithEvictionStrategy(20, "lru")
	if err != nil {
		t.Fatalf("NewWithEvictionStrategy(lru): %v", err)
	}
	if got := lru.Stats().EvictionPolicy; got != "lru" {
		t.Fatalf("string lru EvictionPolicy=%q, want lru", got)
	}
}
