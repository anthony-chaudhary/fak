package radixkv

import "testing"

// thrash_test.go — the evict→reuse-gap thrash detector (#3393). Every test drives the
// tree's own deterministic logical clock (t.clock advances once per Lookup and once per
// suffix-attaching Insert); no wall clock, no sleeps.

// evictOne serves three disjoint 10-token requests into a 20-token budget so the THIRD
// insert evicts the FIRST (LRU) — the canonical "K was evicted under budget pressure"
// setup. It returns the tree plus K (the victim's token path).
func evictOne(t *testing.T) (*Tree, []int) {
	t.Helper()
	tree := New(20)
	k := distinctReq(0, 10)
	touchPure(tree, k)
	touchPure(tree, distinctReq(1, 10))
	touchPure(tree, distinctReq(2, 10)) // budget 30>20 → evicts k (LRU)
	if m := tree.MatchLen(k); m != 0 {
		t.Fatalf("setup: k should be evicted, matched %d", m)
	}
	if st := tree.Stats(); st.Evictions != 1 || st.ThrashTracked != 1 {
		t.Fatalf("setup: evictions=%d tracked=%d, want 1/1", st.Evictions, st.ThrashTracked)
	}
	return tree, k
}

// (a) evict K then re-demand K within the window → one thrash, and the recorded gap is
// exactly the logical-tick delta between the eviction and the re-demanding Lookup.
func TestThrashDetectsEvictThenReinsert(t *testing.T) {
	tree, k := evictOne(t)
	evictClock := tree.clock // the eviction happened inside the last insert's pass, at this tick

	// Advance the clock deterministically with resident touches (1 tick each, no eviction).
	resident := distinctReq(1, 10)
	for i := 0; i < 3; i++ {
		touchPure(tree, resident)
	}

	wantGap := tree.clock + 1 - evictClock // Lookup ticks once before it probes
	b, m := tree.Lookup(k)                 // the probe fires here: demand for k is back
	st := tree.Stats()
	if st.ThrashReuses != 1 {
		t.Fatalf("ThrashReuses=%d, want 1", st.ThrashReuses)
	}
	if st.ThrashGapLast != wantGap || st.ThrashGapTotal != wantGap || st.ThrashGapMax != wantGap {
		t.Fatalf("gap last/total/max=%d/%d/%d, want %d (tick delta)",
			st.ThrashGapLast, st.ThrashGapTotal, st.ThrashGapMax, wantGap)
	}
	if st.ThrashTokens != len(k) {
		t.Fatalf("ThrashTokens=%d, want %d (the victim's edge)", st.ThrashTokens, len(k))
	}
	if st.ThrashTracked != 0 {
		t.Fatalf("ThrashTracked=%d, want 0 (entry consumed once)", st.ThrashTracked)
	}

	// Completing the re-insert must NOT double-count: the Lookup consumed the entry.
	leaf := tree.Insert(b, k[m:], nil)
	tree.Done(leaf)
	if st := tree.Stats(); st.ThrashReuses != 1 {
		t.Fatalf("ThrashReuses=%d after re-insert, want still 1 (consumed once)", st.ThrashReuses)
	}
}

// The attachLeaf probe stands alone: a WarmInsert (no Lookup on its path) re-creating a
// just-evicted key is detected too, with the exact same-tick gap of 0.
func TestThrashDetectsWarmReinsert(t *testing.T) {
	tree, k := evictOne(t)
	tree.WarmInsert(k, nil) // no clock tick on the warm path → gap 0
	st := tree.Stats()
	if st.ThrashReuses != 1 || st.ThrashGapLast != 0 || st.ThrashGapMax != 0 {
		t.Fatalf("warm re-insert: reuses/gapLast/gapMax=%d/%d/%d, want 1/0/0",
			st.ThrashReuses, st.ThrashGapLast, st.ThrashGapMax)
	}
	if st.ThrashTokens != len(k) {
		t.Fatalf("ThrashTokens=%d, want %d", st.ThrashTokens, len(k))
	}
}

// (b) capacity ageing: push thrashCap further evictions after K's, so K's record is the
// oldest and falls off the ring — then a re-insert of K is a COLD MISS, not a thrash.
func TestThrashAgedOutByCapIsNotCounted(t *testing.T) {
	tree, k := evictOne(t)
	// Each further distinct serve evicts exactly one LRU leaf (one record); after
	// thrashCap of them K's record has been popped for room.
	for i := 3; i < 3+thrashCap; i++ {
		touchPure(tree, distinctReq(i, 10))
	}
	if tree.thrashCount != thrashCap || len(tree.thrashIndex) != thrashCap {
		t.Fatalf("ring/index=%d/%d, want full at the %d bound",
			tree.thrashCount, len(tree.thrashIndex), thrashCap)
	}
	touchPure(tree, k) // re-insert K after it aged out of the ring
	if st := tree.Stats(); st.ThrashReuses != 0 {
		t.Fatalf("ThrashReuses=%d, want 0 (aged out by capacity → cold miss)", st.ThrashReuses)
	}
}

// (b') window ageing: keep K's record resident in the ring but advance the logical clock
// past thrashWindow — the late re-demand consumes the entry as a cold miss, not a thrash.
func TestThrashAgedOutByWindowIsNotCounted(t *testing.T) {
	tree, k := evictOne(t)
	resident := distinctReq(1, 10)
	for i := uint64(0); i < thrashWindow; i++ { // 1 tick per resident touch, no eviction
		touchPure(tree, resident)
	}
	touchPure(tree, k) // gap = thrashWindow+1 > thrashWindow
	st := tree.Stats()
	if st.ThrashReuses != 0 || st.ThrashGapTotal != 0 {
		t.Fatalf("reuses/gapTotal=%d/%d, want 0/0 (past the window → cold miss)",
			st.ThrashReuses, st.ThrashGapTotal)
	}
}

// (c) a key evicted but never re-demanded is not a thrash — the entry just sits tracked
// until it ages out.
func TestThrashEvictedNeverReusedIsNotCounted(t *testing.T) {
	tree, _ := evictOne(t)
	resident := distinctReq(2, 10)
	for i := 0; i < 5; i++ {
		touchPure(tree, resident) // unrelated demand only
	}
	st := tree.Stats()
	if st.ThrashReuses != 0 || st.ThrashGapTotal != 0 {
		t.Fatalf("reuses/gapTotal=%d/%d, want 0/0 (never re-demanded)",
			st.ThrashReuses, st.ThrashGapTotal)
	}
	if st.ThrashTracked != 1 {
		t.Fatalf("ThrashTracked=%d, want 1 (still armed)", st.ThrashTracked)
	}
}

// (d) the side-map never exceeds its documented bound (thrashCap entries) under sustained
// eviction pressure, and distinct one-shot keys never count as thrash.
func TestThrashSideMapBounded(t *testing.T) {
	tree := New(20)
	for i := 0; i < 3*thrashCap; i++ {
		touchPure(tree, distinctReq(i, 10))
		if tree.thrashCount > thrashCap || len(tree.thrashIndex) > thrashCap {
			t.Fatalf("after %d serves: ring/index=%d/%d exceed the %d bound",
				i+1, tree.thrashCount, len(tree.thrashIndex), thrashCap)
		}
	}
	st := tree.Stats()
	if st.ThrashTracked > thrashCap {
		t.Fatalf("ThrashTracked=%d exceeds the %d bound", st.ThrashTracked, thrashCap)
	}
	if st.ThrashReuses != 0 {
		t.Fatalf("ThrashReuses=%d, want 0 (all keys one-shot)", st.ThrashReuses)
	}
	if st.Evictions == 0 {
		t.Fatal("setup: expected sustained evictions to pressure the side-map")
	}
}

// POLICY eviction (quarantine) never arms the detector: a quarantined prefix coming back
// is not a budget signal.
func TestThrashIgnoresPolicyEviction(t *testing.T) {
	tree := New(0) // unbounded — no budget pressure anywhere in this test
	k := distinctReq(0, 10)
	touchPure(tree, k)
	if freed := tree.EvictPrefix(k); freed != len(k) {
		t.Fatalf("EvictPrefix freed %d, want %d", freed, len(k))
	}
	touchPure(tree, k) // re-insert the policy-evicted key
	st := tree.Stats()
	if st.ThrashReuses != 0 || st.ThrashTracked != 0 {
		t.Fatalf("reuses/tracked=%d/%d, want 0/0 (policy eviction is not thrash)",
			st.ThrashReuses, st.ThrashTracked)
	}
}
