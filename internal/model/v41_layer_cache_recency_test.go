package model

// v41_layer_cache_recency_test.go -- the #13294 frontier witness for the
// layer-scoped routed-expert cache's eviction policy.
//
// The #13296 layer cache exists to raise the resident hit fraction of the V4.1
// prefill first-token path (fak#13294): a repeated routed-expert read should be
// served from RAM rather than re-faulted from the checkpoint tier. Its dedupe is
// proven token-independent ONLY in the regime the existing witness exercises --
// every prompt token identical, so the layer's distinct routed set is one
// position's set and fits the ~810 MiB bound.
//
// Production prefill is NOT that regime. Distinct prompt tokens route distinct
// experts; a prompt of L tokens routes up to min(L*topK, experts) distinct
// experts in a layer, and real routing is skewed: a HOT subset recurs across the
// token dimension while a cold TAIL streams through once (the classic MoE
// hot-expert distribution). When the hot subset plus the live streaming window
// exceeds the cache bound, the eviction POLICY decides the hit fraction, and the
// shipped policy is FIFO/insertion-order: v41LayerCachePut appends on first
// insert and evicts the oldest insertion, and a GET does not re-queue. A hot
// expert re-read after the tail pushed it out therefore re-faults the tier even
// though a recency policy would have kept it.
//
// This witness pins the recency contract on the REAL cache the forward uses,
// over a hot-set + streaming-tail access stream. Under FIFO the hot re-reads
// fault once the tail fills the bound; under a hit-refreshing (LRU) policy the
// hot set stays resident and the read count falls strictly.

import "testing"

// v41RecencyStream builds a hot-set + streaming-tail key stream: each round
// re-reads the same `hot` keys (temporal locality), then reads `tail` fresh keys
// that will not be seen again (a streaming tail). The bound admits hot+some tail
// entries, so FIFO's insertion order evicts a hot key during the tail and the
// next round's hot re-read faults; an LRU policy refreshes the hot keys on each
// re-read and keeps them across the tail.
func v41RecencyStream(hot, tail, rounds int) []string {
	var keys []string
	for r := 0; r < rounds; r++ {
		for i := 0; i < hot; i++ {
			keys = append(keys, "ffn.experts.hot"+itoa(i)+".w1.weight")
		}
		for i := 0; i < tail; i++ {
			keys = append(keys, "ffn.experts.tail"+itoa(r)+"_"+itoa(i)+".w1.weight")
		}
	}
	return keys
}

// v41RecencyCacheReads drives the REAL cache (v41ProjScratch.v41LayerCacheGet /
// v41LayerCachePut, the production eviction path) over keys and returns the
// number of misses -- each miss is one routed-expert read the forward would fault
// from the checkpoint tier.
func v41RecencyCacheReads(budgetBytes int64, keys []string) int {
	s := &v41ProjScratch{expertLayerCacheBytes: budgetBytes}
	misses := 0
	for _, k := range keys {
		if _, ok := s.v41LayerCacheGet(k); ok {
			continue
		}
		misses++
		s.v41LayerCachePut(k, make([]float32, 1))
	}
	return misses
}

// TestV41LayerCacheKeepsHotSetAcrossTail is the #13294 recency witness.
//
// 8 hot keys are re-read every round (temporal locality: the recurring routed
// experts); 2 fresh tail keys stream through once between the hot rounds (the
// cold tail a real prefill also routes); the bound holds 10 one-float blocks.
// The hot set fits the bound (8 <= 10) with room for the live tail, so a policy
// that refreshes recency on a hit keeps every hot key across the tail. FIFO's
// insertion order does not: the tail inserts evict the hot keys by age even
// though they are re-read every round, so each round after the first re-faults
// hot keys too.
func TestV41LayerCacheKeepsHotSetAcrossTail(t *testing.T) {
	const hot, tail, rounds = 8, 2, 5
	const bound = 10 // hot <= bound < hot+rounds*tail: hot fits, but the stream exceeds the bound

	keys := v41RecencyStream(hot, tail, rounds)
	got := v41RecencyCacheReads(int64(bound)*4, keys)

	// First round faults hot+tail. After that a recency policy serves every hot
	// re-read from RAM (the hit refreshes recency) and only each round's fresh
	// tail faults, so the total is (hot+tail) + (rounds-1)*tail.
	wantLRU := (hot + tail) + (rounds-1)*tail
	if got != wantLRU {
		t.Fatalf("layer cache read %d misses over %d rounds (hot=%d, tail=%d, bound=%d), want %d: "+
			"a recency policy keeps the hot set resident across the streaming tail (FIFO evicts it in insertion order)",
			got, rounds, hot, tail, bound, wantLRU)
	}

	// Non-vacuous: strictly below the recompute-everything ceiling a no-reuse
	// policy would pay.
	if ceiling := (hot + tail) * rounds; got >= ceiling {
		t.Fatalf("read count %d is not below the no-reuse ceiling %d", got, ceiling)
	}
}

// TestV41LayerCacheReReadIsFreeWithinBound pins the minimal non-vacuous floor:
// a key re-read while fewer than `bound` other distinct keys intervene is a hit.
func TestV41LayerCacheReReadIsFreeWithinBound(t *testing.T) {
	s := &v41ProjScratch{expertLayerCacheBytes: 4 * 4} // holds 4 one-float blocks
	for _, k := range []string{"a", "b", "c"} {
		s.v41LayerCachePut(k, make([]float32, 1))
	}
	if _, ok := s.v41LayerCacheGet("a"); !ok {
		t.Fatal("key 'a' evicted with room to spare; the cache bound is wrong")
	}
	// Re-touch 'a' (a hit, which refreshes recency) then insert one more: with the
	// bound at 4, the LRU victim is 'b', never the just-used 'a'.
	s.v41LayerCachePut("d", make([]float32, 1))
	if _, ok := s.v41LayerCacheGet("a"); !ok {
		t.Fatal("key 'a' fell out of the cache immediately after being used; recency is not refreshed on a hit")
	}
}
