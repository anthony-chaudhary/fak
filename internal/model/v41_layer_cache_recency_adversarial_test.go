package model

// v41_layer_cache_recency_adversarial_test.go -- adversarial edge cases for the
// #13294 LRU refresh in v41ProjScratch. Authored independently of the fix to
// falsify it: ordering/aliasing, exact-boundary eviction, oversized blocks,
// duplicate insert, and the recency invariant.

import (
	"reflect"
	"testing"
)

// TestAdversarialLayerCacheReturnsExactBytes pins that a hit returns the SAME
// bytes that were put, and a re-read after a refresh still does: the refresh
// must move bookkeeping, never data.
func TestAdversarialLayerCacheReturnsExactBytes(t *testing.T) {
	s := &v41ProjScratch{expertLayerCacheBytes: 16 * 4}
	payload := []float32{1, 2, 3, 4}
	s.v41LayerCachePut("a", payload)
	s.v41LayerCachePut("b", make([]float32, 4))
	if _, ok := s.v41LayerCacheGet("a"); !ok {
		t.Fatal("'a' not resident")
	}
	got, ok := s.v41LayerCacheGet("a")
	if !ok {
		t.Fatal("second read of 'a' missed")
	}
	if !reflect.DeepEqual(got, payload) {
		t.Fatalf("hit returned %v, want %v", got, payload)
	}
	// A hit must not alias the put-time slice in a way that lets a caller mutate
	// the cache: the same backing array is expected (documented read-only use),
	// but the VALUES must be unchanged after an unrelated get.
	_ = got
	if !reflect.DeepEqual(s.layerExperts["a"], payload) {
		t.Fatalf("cache content changed to %v", s.layerExperts["a"])
	}
}

// TestAdversarialLayerCacheExactBoundaryEviction pins LRU victim selection at
// the exact byte boundary: with the bound full, inserting one more evicts the
// least-recently-used key, NOT the just-refreshed one.
func TestAdversarialLayerCacheExactBoundaryEviction(t *testing.T) {
	s := &v41ProjScratch{expertLayerCacheBytes: 3 * 4} // exactly 3 one-float blocks
	s.v41LayerCachePut("a", make([]float32, 1))
	s.v41LayerCachePut("b", make([]float32, 1))
	s.v41LayerCachePut("c", make([]float32, 1))
	// Refresh 'a' so it is MRU; LRU order is now b, c, a.
	if _, ok := s.v41LayerCacheGet("a"); !ok {
		t.Fatal("'a' missing before refresh")
	}
	s.v41LayerCachePut("d", make([]float32, 1)) // must evict 'b' (LRU), not 'a'
	if _, ok := s.v41LayerCacheGet("a"); !ok {
		t.Fatal("refreshed key 'a' was evicted; recency not honored")
	}
	if _, ok := s.v41LayerCacheGet("b"); ok {
		t.Fatal("least-recently-used key 'b' survived; eviction is not LRU")
	}
	if _, ok := s.v41LayerCacheGet("c"); !ok {
		t.Fatal("key 'c' (more recent than 'b') was evicted")
	}
	if _, ok := s.v41LayerCacheGet("d"); !ok {
		t.Fatal("newly inserted key 'd' missing")
	}
}

// TestAdversarialLayerCacheOversizedBlockNotCached pins that a block larger than
// the whole bound is refused WITHOUT evicting the current residents.
func TestAdversarialLayerCacheOversizedBlockNotCached(t *testing.T) {
	s := &v41ProjScratch{expertLayerCacheBytes: 4 * 4}
	s.v41LayerCachePut("a", make([]float32, 1))
	s.v41LayerCachePut("huge", make([]float32, 100)) // > bound
	if _, ok := s.v41LayerCacheGet("huge"); ok {
		t.Fatal("oversized block was cached")
	}
	if _, ok := s.v41LayerCacheGet("a"); !ok {
		t.Fatal("oversized insert evicted a legitimate resident")
	}
}

// TestAdversarialLayerCacheDuplicateInsertKeepsBytes pins that re-putting an
// existing key refreshes recency and does NOT double-count bytes (which would
// corrupt the byte accounting and cause premature eviction).
func TestAdversarialLayerCacheDuplicateInsertKeepsBytes(t *testing.T) {
	s := &v41ProjScratch{expertLayerCacheBytes: 3 * 4}
	s.v41LayerCachePut("a", make([]float32, 1))
	s.v41LayerCachePut("b", make([]float32, 1))
	before := s.layerExpertBytes
	s.v41LayerCachePut("a", make([]float32, 1)) // duplicate
	if s.layerExpertBytes != before {
		t.Fatalf("duplicate insert changed byte count %d -> %d", before, s.layerExpertBytes)
	}
	if len(s.layerExpertFIFO) != 2 {
		t.Fatalf("duplicate insert inflated the queue to %d entries", len(s.layerExpertFIFO))
	}
	// 'a' is now MRU, so inserting 'c' evicts 'b'.
	s.v41LayerCachePut("c", make([]float32, 1))
	if _, ok := s.v41LayerCacheGet("a"); !ok {
		t.Fatal("re-put key 'a' was not treated as most-recently-used")
	}
}

// TestAdversarialLayerCacheResetClearsQueue pins that a layer reset drops both
// the map and the queue, so no stale ordering leaks into the next layer.
func TestAdversarialLayerCacheResetClearsQueue(t *testing.T) {
	s := &v41ProjScratch{expertLayerCacheBytes: 8 * 4}
	s.v41LayerCachePut("a", make([]float32, 1))
	s.v41LayerCachePut("b", make([]float32, 1))
	s.v41LayerCacheReset()
	if s.layerExperts != nil || s.layerExpertFIFO != nil || s.layerExpertBytes != 0 {
		t.Fatalf("reset left state: experts=%v fifo=%v bytes=%d", s.layerExperts, s.layerExpertFIFO, s.layerExpertBytes)
	}
	// A get after reset must miss, not panic on the (nil) queue.
	if _, ok := s.v41LayerCacheGet("a"); ok {
		t.Fatal("reset cache served a stale hit")
	}
}

// TestAdversarialLayerCacheZeroBudgetInert pins the default-off contract: a zero
// budget caches nothing and a get never touches a nil queue.
func TestAdversarialLayerCacheZeroBudgetInert(t *testing.T) {
	var s v41ProjScratch // expertLayerCacheBytes == 0
	s.v41LayerCachePut("a", make([]float32, 1))
	if _, ok := s.v41LayerCacheGet("a"); ok {
		t.Fatal("zero-budget cache retained a block")
	}
}

// TestAdversarialLayerCacheNilReceiverSafe pins that a nil scratch never panics
// on any cache method (the historical default path).
func TestAdversarialLayerCacheNilReceiverSafe(t *testing.T) {
	var s *v41ProjScratch
	if _, ok := s.v41LayerCacheGet("a"); ok {
		t.Fatal("nil scratch served a hit")
	}
	s.v41LayerCachePut("a", make([]float32, 1))
	s.v41LayerCacheReset()
}
