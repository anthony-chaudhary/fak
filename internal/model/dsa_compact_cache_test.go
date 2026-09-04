package model

import (
	"errors"
	"math/rand"
	"reflect"
	"testing"
)

func TestCompactLatentCache_AppendAndGet(t *testing.T) {
	dim := 64
	maxRollback := 16
	cache := NewCompactLatentCache(dim, maxRollback)

	if cache.Dim() != dim {
		t.Fatalf("expected dim %d, got %d", dim, cache.Dim())
	}
	if cache.Capacity() != DefaultCacheCapacity {
		t.Fatalf("expected capacity %d, got %d", DefaultCacheCapacity, cache.Capacity())
	}
	if cache.MaxRollback() != maxRollback {
		t.Fatalf("expected maxRollback %d, got %d", maxRollback, cache.MaxRollback())
	}

	// Appending vectors with distinct values
	v0 := make([]float32, dim)
	for i := range v0 {
		v0[i] = float32(i) * 1.5
	}
	v1 := make([]float32, dim)
	for i := range v1 {
		v1[i] = float32(i)*2.0 + 0.5
	}

	cache.Append(0, v0)
	cache.Append(1, v1)

	// Get existing positions
	got0, ok0 := cache.Get(0)
	if !ok0 {
		t.Fatalf("expected Get(0) to succeed")
	}
	if !reflect.DeepEqual(got0, v0) {
		t.Fatalf("Get(0) mismatch:\ngot  %v\nwant %v", got0, v0)
	}

	got1, ok1 := cache.Get(1)
	if !ok1 {
		t.Fatalf("expected Get(1) to succeed")
	}
	if !reflect.DeepEqual(got1, v1) {
		t.Fatalf("Get(1) mismatch:\ngot  %v\nwant %v", got1, v1)
	}

	// Verify defensive copy: mutating got0 must not alter cache
	got0[0] = 9999.0
	got0After, _ := cache.Get(0)
	if got0After[0] == 9999.0 {
		t.Fatalf("Get(0) returned mutable reference to internal cache pool")
	}

	// Get unwritten or negative positions
	if _, ok := cache.Get(2); ok {
		t.Fatalf("expected Get(2) to return false for unwritten position")
	}
	if _, ok := cache.Get(-1); ok {
		t.Fatalf("expected Get(-1) to return false for negative position")
	}
	if _, ok := cache.Get(100); ok {
		t.Fatalf("expected Get(100) to return false for unwritten position")
	}

	// Panic tests for invalid arguments
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected Append with wrong dimension to panic")
		}
	}()
	cache.Append(2, make([]float32, dim+1))
}

func TestCompactLatentCache_RollbackJournal(t *testing.T) {
	dim := 32
	maxRollback := 16
	cache := NewCompactLatentCache(dim, maxRollback)

	// Append 10 tokens
	tokens := make([][]float32, 10)
	for i := 0; i < 10; i++ {
		tokens[i] = make([]float32, dim)
		for d := 0; d < dim; d++ {
			tokens[i][d] = float32(i*100 + d)
		}
		cache.Append(i, tokens[i])
	}

	if depth := cache.JournalDepth(); depth != 10 {
		t.Fatalf("expected journal depth 10, got %d", depth)
	}

	// Rollback 4 tokens (tokens 6..9 undone)
	if err := cache.Rollback(4); err != nil {
		t.Fatalf("Rollback(4) failed: %v", err)
	}

	if depth := cache.JournalDepth(); depth != 6 {
		t.Fatalf("expected journal depth 6 after rolling back 4, got %d", depth)
	}

	// Verify tokens 6..9 are no longer retrievable
	for i := 6; i < 10; i++ {
		if _, ok := cache.Get(i); ok {
			t.Fatalf("token %d should have been rolled back", i)
		}
	}

	// Verify tokens 0..5 remain intact
	for i := 0; i < 6; i++ {
		got, ok := cache.Get(i)
		if !ok {
			t.Fatalf("token %d should remain present after rollback", i)
		}
		if !reflect.DeepEqual(got, tokens[i]) {
			t.Fatalf("token %d corrupted after rollback", i)
		}
	}

	// Append 20 new tokens from pos 6 to 25 (rolling past 16)
	newTokens := make([][]float32, 26)
	for i := 0; i < 6; i++ {
		newTokens[i] = tokens[i]
	}
	for i := 6; i < 26; i++ {
		newTokens[i] = make([]float32, dim)
		for d := 0; d < dim; d++ {
			newTokens[i][d] = float32(i*1000 + d)
		}
		cache.Append(i, newTokens[i])
	}

	// Journal should be capped at maxRollback = 16
	if depth := cache.JournalDepth(); depth != 16 {
		t.Fatalf("expected journal depth capped at %d, got %d", maxRollback, depth)
	}

	// Attempting rollback of 17 tokens MUST be refused
	err := cache.Rollback(17)
	if err == nil {
		t.Fatalf("expected Rollback(17) to fail when journal depth is 16")
	}
	if !errors.Is(err, ErrRollbackExceedsJournal) {
		t.Fatalf("expected error wrapping ErrRollbackExceedsJournal, got %v", err)
	}

	// Verify state is unchanged after refused rollback
	for i := 0; i < 26; i++ {
		got, ok := cache.Get(i)
		if !ok {
			t.Fatalf("token %d should still be present after refused rollback", i)
		}
		if !reflect.DeepEqual(got, newTokens[i]) {
			t.Fatalf("token %d altered by refused rollback", i)
		}
	}

	// Rollback 16 tokens (tokens 10..25 undone, tokens 0..9 remain)
	if err := cache.Rollback(16); err != nil {
		t.Fatalf("Rollback(16) failed: %v", err)
	}

	if depth := cache.JournalDepth(); depth != 0 {
		t.Fatalf("expected journal depth 0 after rolling back 16, got %d", depth)
	}

	for i := 10; i < 26; i++ {
		if _, ok := cache.Get(i); ok {
			t.Fatalf("token %d should have been rolled back", i)
		}
	}
	for i := 0; i < 10; i++ {
		got, ok := cache.Get(i)
		if !ok {
			t.Fatalf("token %d should still be present", i)
		}
		if !reflect.DeepEqual(got, newTokens[i]) {
			t.Fatalf("token %d corrupted", i)
		}
	}

	// Rollback on empty journal must fail
	if err := cache.Rollback(1); !errors.Is(err, ErrRollbackExceedsJournal) {
		t.Fatalf("expected ErrRollbackExceedsJournal on empty journal, got %v", err)
	}

	// Rollback 0 is a no-op
	if err := cache.Rollback(0); err != nil {
		t.Fatalf("Rollback(0) should succeed as no-op, got %v", err)
	}

	// Rollback < 0 fails
	if err := cache.Rollback(-1); err == nil {
		t.Fatalf("expected negative rollback to fail")
	}
}

func TestCompactLatentCache_RollbackRestoresOverwritten(t *testing.T) {
	// Small capacity cache to verify circular overwrite and rollback restoration
	dim := 8
	capacity := 4
	maxRollback := 4
	cache := NewCompactLatentCache(dim, maxRollback, capacity)

	v0 := []float32{0, 1, 2, 3, 4, 5, 6, 7}
	v1 := []float32{10, 11, 12, 13, 14, 15, 16, 17}
	v2 := []float32{20, 21, 22, 23, 24, 25, 26, 27}
	v3 := []float32{30, 31, 32, 33, 34, 35, 36, 37}
	v4 := []float32{40, 41, 42, 43, 44, 45, 46, 47} // Overwrites slot 0 (pos 0)

	cache.Append(0, v0)
	cache.Append(1, v1)
	cache.Append(2, v2)
	cache.Append(3, v3)
	cache.Append(4, v4) // pos 4 % 4 == 0; overwrites pos 0 in slot 0

	// Pos 4 is present; pos 0 was evicted/overwritten
	if _, ok := cache.Get(0); ok {
		t.Fatalf("pos 0 should be evicted by pos 4")
	}
	got4, ok4 := cache.Get(4)
	if !ok4 || !reflect.DeepEqual(got4, v4) {
		t.Fatalf("pos 4 should be retrievable")
	}

	// Rollback pos 4: pos 0 should be restored from the reversion journal!
	if err := cache.Rollback(1); err != nil {
		t.Fatalf("Rollback(1) failed: %v", err)
	}

	if _, ok := cache.Get(4); ok {
		t.Fatalf("pos 4 should have been rolled back")
	}
	got0, ok0 := cache.Get(0)
	if !ok0 {
		t.Fatalf("pos 0 should have been restored by rollback")
	}
	if !reflect.DeepEqual(got0, v0) {
		t.Fatalf("restored pos 0 mismatch:\ngot  %v\nwant %v", got0, v0)
	}
}

func TestCompactLatentCache_MemoryReductionAndWindowParity(t *testing.T) {
	// DeepSeek MLA kv_lora_rank dimension
	dim := 512
	maxRollback := 16
	numTokens := 10000

	cache := NewCompactLatentCache(dim, maxRollback)
	windowSize := cache.Capacity() // 1024 tokens

	// Deterministic pseudo-random generation for reference parity
	rng := rand.New(rand.NewSource(42))
	fullHistory := make([][]float32, numTokens)

	for pos := 0; pos < numTokens; pos++ {
		vec := make([]float32, dim)
		for d := 0; d < dim; d++ {
			vec[d] = rng.Float32()
		}
		fullHistory[pos] = vec
		cache.Append(pos, vec)
	}

	// 1. Memory reduction verification:
	// Standard full-history storage retains all 10,000 uncompressed vectors.
	fullHistoryBytes := int64(numTokens * dim * 4)
	compactCacheBytes := cache.MemoryBytes()

	reduction := float64(fullHistoryBytes-compactCacheBytes) / float64(fullHistoryBytes)
	reductionPct := reduction * 100.0

	t.Logf("Full history memory (%d tokens, dim %d): %d bytes (%.2f MB)",
		numTokens, dim, fullHistoryBytes, float64(fullHistoryBytes)/(1024*1024))
	t.Logf("CompactLatentCache memory (cap %d, rollback %d): %d bytes (%.2f MB)",
		cache.Capacity(), maxRollback, compactCacheBytes, float64(compactCacheBytes)/(1024*1024))
	t.Logf("Memory reduction: %.2f%% (requirement: >= 80%%)", reductionPct)

	if reduction < 0.80 {
		t.Fatalf("memory reduction %.2f%% is below the required >= 80%% threshold", reductionPct)
	}

	// 2. Exact retrieval parity for the recent window [numTokens - windowSize, numTokens - 1]
	startPos := numTokens - windowSize
	for pos := startPos; pos < numTokens; pos++ {
		got, ok := cache.Get(pos)
		if !ok {
			t.Fatalf("recent window position %d missing from cache", pos)
		}
		if !reflect.DeepEqual(got, fullHistory[pos]) {
			t.Fatalf("exact parity failure at recent window position %d", pos)
		}
	}

	// 3. Eviction verification for tokens older than the circular window
	sampleEvictedPositions := []int{0, 1, 100, 500, 1000, 5000, startPos - 1}
	for _, pos := range sampleEvictedPositions {
		if _, ok := cache.Get(pos); ok {
			t.Fatalf("position %d should have been evicted from circular buffer", pos)
		}
	}
}
