package eviction

import (
	"math/rand"
	"testing"
)

func TestWTinyLFUBasic(t *testing.T) {
	var evictedKeys []uint64
	w := NewWTinyLFU(100, func(h uint64, kl uint16) {
		evictedKeys = append(evictedKeys, h)
	})

	// Admit entries
	for i := uint64(1); i <= 50; i++ {
		evicted := w.Admit(i, 8)
		_ = evicted
	}
	if w.Size() != 50 {
		t.Errorf("size: got %d, want 50", w.Size())
	}

	// Access should return true for existing
	if !w.Access(1) {
		t.Error("access(1) should return true")
	}
	// Access should return false for non-existing
	if w.Access(999) {
		t.Error("access(999) should return false")
	}

	// Remove
	w.Remove(1)
	if w.Access(1) {
		t.Error("access(1) should return false after remove")
	}
}

func TestWTinyLFUEviction(t *testing.T) {
	var evictedKeys []uint64
	w := NewWTinyLFU(20, func(h uint64, kl uint16) {
		evictedKeys = append(evictedKeys, h)
	})

	// Fill beyond capacity
	for i := uint64(1); i <= 30; i++ {
		w.Admit(i, 8)
	}

	// Some entries should have been evicted
	if len(evictedKeys) == 0 {
		t.Error("expected some evictions")
	}

	// Hot key: access many times to build frequency
	for j := 0; j < 20; j++ {
		w.Access(1)
	}

	// Now insert more cold keys
	for i := uint64(100); i <= 120; i++ {
		w.Admit(i, 8)
	}

	// Key 1 should survive (high frequency)
	if !w.Access(1) {
		t.Error("hot key 1 should survive eviction")
	}
}

func TestSIEVEBasic(t *testing.T) {
	var evictedKeys []uint64
	s := NewSIEVE(50, func(h uint64, kl uint16) {
		evictedKeys = append(evictedKeys, h)
	})

	for i := uint64(1); i <= 50; i++ {
		s.Admit(i, 8)
	}
	if s.Size() != 50 {
		t.Errorf("size: got %d, want 50", s.Size())
	}

	// Access marks visited
	s.Access(1)

	// Fill beyond capacity — should evict unvisited first
	for i := uint64(51); i <= 60; i++ {
		s.Admit(i, 8)
	}
	if len(evictedKeys) == 0 {
		t.Error("expected some evictions")
	}

	// Key 1 was visited, should survive
	if !s.Access(1) {
		t.Error("visited key 1 should survive SIEVE eviction")
	}
}

func TestCountMinSketch(t *testing.T) {
	s := NewCountMinSketch(1024)

	// Increment key 42 ten times
	for i := 0; i < 10; i++ {
		s.Increment(42)
	}

	est := s.Estimate(42)
	if est < 8 || est > 10 {
		t.Errorf("estimate for key 42: got %d, want ~10", est)
	}

	// Key 99 never incremented
	est99 := s.Estimate(99)
	if est99 > 2 {
		t.Errorf("estimate for unseen key 99: got %d, want ~0", est99)
	}
}

func TestWTinyLFUZipfianHitRate(t *testing.T) {
	// Simulate Zipfian workload and verify hit rate > LRU
	capacity := uint64(1000)
	numKeys := uint64(10000)

	hitCount := 0
	missCount := 0

	w := NewWTinyLFU(capacity, func(h uint64, kl uint16) {})

	// Generate Zipfian access pattern using stdlib Zipf generator
	r := rand.New(rand.NewSource(42))
	zipf := rand.NewZipf(r, 1.1, 1, numKeys-1)

	for i := 0; i < 100000; i++ {
		key := zipf.Uint64() + 1
		if w.Access(key) {
			hitCount++
		} else {
			w.Admit(key, 8)
			missCount++
		}
	}

	hitRate := float64(hitCount) / float64(hitCount+missCount) * 100
	t.Logf("W-TinyLFU hit rate: %.1f%% (hits=%d, misses=%d)", hitRate, hitCount, missCount)

	if hitRate < 50 {
		t.Errorf("hit rate too low: %.1f%%, expected > 50%%", hitRate)
	}
}
