package ctxmmu

import (
	"errors"
	"testing"
)

// TestEvictUnreferencedSessions_UnreferencedOnly proves the M3 unreferenced-only
// session eviction rule: a referenced (refCount>0) session's warm pages are never
// evicted while unreferenced sessions' KV can still be reclaimed, even when the
// eviction pressure exceeds the unreferenced supply.
func TestEvictUnreferencedSessions_UnreferencedOnly(t *testing.T) {
	const total = 10

	newPool := func(t *testing.T) *KVPool {
		t.Helper()
		cfg := KVPoolConfig{
			TokensPerPage: 16,
			NumLayers:     4,
			NumKVHeads:    4,
			HeadDim:       64,
			DType:         KVDTypeFP16,
			Topology:      KVTopologyGQA,
			TotalPages:    total,
		}
		pool, err := NewKVPool(cfg)
		if err != nil {
			t.Fatalf("NewKVPool failed: %v", err)
		}
		return pool
	}

	// makeColdSupply allocates n physical blocks and drops their refcount to 0
	// while leaving them allocated — the exact state a released session's KV is in
	// before reclamation. Returns their allocation IDs.
	makeColdSupply := func(t *testing.T, pool *KVPool, n int) []int {
		t.Helper()
		ids := make([]int, 0, n)
		for i := 0; i < n; i++ {
			b, err := pool.Allocator().Allocate()
			if err != nil {
				t.Fatalf("cold Allocate %d failed: %v", i, err)
			}
			if got := b.Release(); got != 0 {
				t.Fatalf("cold block %d Release returned %d, want 0", b.ID, got)
			}
			ids = append(ids, b.ID)
		}
		return ids
	}

	t.Run("PressureExceedsUnreferencedSupply_WarmStaysResident", func(t *testing.T) {
		pool := newPool(t)

		// Warm referenced session: one page, refCount=1.
		seq, err := pool.CreateSequence("warm")
		if err != nil {
			t.Fatalf("CreateSequence warm failed: %v", err)
		}
		if err := pool.AppendTokens("warm", 16); err != nil {
			t.Fatalf("AppendTokens warm failed: %v", err)
		}
		warmPages := seq.Pages()
		if len(warmPages) != 1 {
			t.Fatalf("expected warm session to hold 1 page, got %d", len(warmPages))
		}
		warmID := warmPages[0]
		warmBlock, err := pool.Allocator().GetBlock(warmID)
		if err != nil {
			t.Fatalf("GetBlock(warm) failed: %v", err)
		}
		if warmBlock.RefCount() != 1 {
			t.Fatalf("warm block refCount = %d, want 1", warmBlock.RefCount())
		}

		// Cold unreferenced supply: 3 allocated blocks at refCount 0.
		coldIDs := makeColdSupply(t, pool, 3)

		// Sanity: 1 warm + 3 cold allocated, 6 free.
		if got := pool.PhysicalBlocksAllocated(); got != 4 {
			t.Fatalf("allocated = %d, want 4", got)
		}

		// Eviction pressure (5) EXCEEDS the unreferenced supply (3).
		const target = 5
		reclaimed, err := pool.EvictUnreferencedSessions(target)
		if err != nil {
			t.Fatalf("EvictUnreferencedSessions failed: %v", err)
		}

		// Reclaimed must be capped at the unreferenced supply, strictly below target.
		if reclaimed != 3 {
			t.Fatalf("reclaimed = %d, want 3 (capped at unreferenced supply)", reclaimed)
		}
		if reclaimed >= target {
			t.Fatalf("reclaimed = %d reached target %d; warm KV was evicted", reclaimed, target)
		}

		// The referenced session's warm page must stay resident: still allocated,
		// still referenced, and NOT swapped out.
		warmBlock, err = pool.Allocator().GetBlock(warmID)
		if err != nil {
			t.Fatalf("warm block was evicted: GetBlock(warm) = %v", err)
		}
		if warmBlock.RefCount() != 1 {
			t.Fatalf("warm block refCount = %d after eviction, want 1", warmBlock.RefCount())
		}
		if warmBlock.Swapped() {
			t.Fatalf("warm block was swapped out under eviction pressure")
		}
		if got := seq.Pages(); len(got) != 1 || got[0] != warmID {
			t.Fatalf("warm page table changed after eviction: %v", got)
		}
		if _, _, err := pool.Translate("warm", 0); err != nil {
			t.Fatalf("warm token no longer translatable after eviction: %v", err)
		}

		// Every cold block must have been reclaimed (freed).
		for _, id := range coldIDs {
			if _, err := pool.Allocator().GetBlock(id); !errors.Is(err, ErrBlockNotFound) {
				t.Fatalf("cold block %d still allocated after eviction, err=%v", id, err)
			}
		}
	})

	t.Run("OnlyReferencedSessionsRemain_ReclaimsZero", func(t *testing.T) {
		pool := newPool(t)

		// A single referenced session, no unreferenced supply.
		if _, err := pool.CreateSequence("warm"); err != nil {
			t.Fatalf("CreateSequence failed: %v", err)
		}
		if err := pool.AppendTokens("warm", 16); err != nil {
			t.Fatalf("AppendTokens failed: %v", err)
		}
		warmSeq, ok := pool.GetSequence("warm")
		if !ok {
			t.Fatalf("GetSequence(warm) not found")
		}
		warmID := warmSeq.Pages()[0]

		// Pressure far exceeds any (zero) unreferenced supply.
		reclaimed, err := pool.EvictUnreferencedSessions(8)
		if err != nil {
			t.Fatalf("EvictUnreferencedSessions failed: %v", err)
		}
		if reclaimed != 0 {
			t.Fatalf("reclaimed = %d with only referenced sessions, want 0", reclaimed)
		}

		// Warm KV stays resident.
		b, err := pool.Allocator().GetBlock(warmID)
		if err != nil {
			t.Fatalf("warm block evicted: %v", err)
		}
		if b.RefCount() != 1 || b.Swapped() {
			t.Fatalf("warm block refCount=%d swapped=%v, want refCount=1 swapped=false",
				b.RefCount(), b.Swapped())
		}
	})
}
