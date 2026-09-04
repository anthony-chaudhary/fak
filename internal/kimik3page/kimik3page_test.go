package kimik3page

import (
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.BlockTokens != DefaultBlockTokens {
		t.Errorf("BlockTokens = %d, want %d", cfg.BlockTokens, DefaultBlockTokens)
	}
	if cfg.TotalLayers != TotalLayers {
		t.Errorf("TotalLayers = %d, want %d", cfg.TotalLayers, TotalLayers)
	}
	if cfg.KDASlices != KDASubLayers || cfg.MLASlices != MLASubLayers {
		t.Errorf("Layer slices KDA=%d MLA=%d, want %d, %d", cfg.KDASlices, cfg.MLASlices, KDASubLayers, MLASubLayers)
	}
	if cfg.BytesPerTok <= 0 {
		t.Errorf("BytesPerTok = %d, want > 0", cfg.BytesPerTok)
	}
}

func TestNewPagePoolValidation(t *testing.T) {
	cfg := DefaultConfig()
	if _, err := NewPagePool(cfg, 0); err == nil {
		t.Error("expected error for capacity 0")
	}

	badCfg := cfg
	badCfg.BlockTokens = 7 // not multiple of 8
	if _, err := NewPagePool(badCfg, 10); err != ErrInvalidBlockSize {
		t.Errorf("got error %v, want ErrInvalidBlockSize", err)
	}
}

func TestPagePoolAllocationAndRecycle(t *testing.T) {
	cfg := DefaultConfig()
	pool, err := NewPagePool(cfg, 4)
	if err != nil {
		t.Fatalf("NewPagePool failed: %v", err)
	}

	if pool.FreeBlocks() != 4 || pool.AllocatedBlocks() != 0 {
		t.Errorf("initial state mismatch: free=%d, allocated=%d", pool.FreeBlocks(), pool.AllocatedBlocks())
	}

	b0, err := pool.Allocate()
	if err != nil {
		t.Fatalf("allocate b0 failed: %v", err)
	}
	b1, err := pool.Allocate()
	if err != nil {
		t.Fatalf("allocate b1 failed: %v", err)
	}

	if pool.FreeBlocks() != 2 || pool.AllocatedBlocks() != 2 {
		t.Errorf("mid-state mismatch: free=%d, allocated=%d", pool.FreeBlocks(), pool.AllocatedBlocks())
	}

	// Test Retain & Release reference counting
	if err := pool.Retain(b0); err != nil {
		t.Fatalf("retain b0 failed: %v", err)
	}

	// Release once - still retained (refcount 1)
	if err := pool.Release(b0); err != nil {
		t.Fatalf("release b0 (1) failed: %v", err)
	}
	if pool.AllocatedBlocks() != 2 {
		t.Errorf("b0 should still be allocated, got %d", pool.AllocatedBlocks())
	}

	// Release again - freed (refcount 0)
	if err := pool.Release(b0); err != nil {
		t.Fatalf("release b0 (2) failed: %v", err)
	}
	if pool.AllocatedBlocks() != 1 {
		t.Errorf("allocated blocks should be 1, got %d", pool.AllocatedBlocks())
	}

	// Double free should error
	if err := pool.Release(b0); err == nil {
		t.Error("expected error on double free")
	}

	// Drain remaining pool
	_, _ = pool.Allocate()
	_, _ = pool.Allocate()
	_, _ = pool.Allocate()
	if _, err := pool.Allocate(); err != ErrBlockExhaustion {
		t.Errorf("expected ErrBlockExhaustion, got %v", err)
	}

	// Release b1 and check bounds
	if err := pool.Release(999); err != ErrInvalidBlockIndex {
		t.Errorf("expected ErrInvalidBlockIndex for 999, got %v", err)
	}
	if err := pool.Retain(-1); err != ErrInvalidBlockIndex {
		t.Errorf("expected ErrInvalidBlockIndex for -1, got %v", err)
	}
	if err := pool.Release(b1); err != nil {
		t.Errorf("release b1 failed: %v", err)
	}
}

func TestBlockTableLifecycleAndAddressing(t *testing.T) {
	cfg := DefaultConfig()
	pool, err := NewPagePool(cfg, 10)
	if err != nil {
		t.Fatalf("NewPagePool: %v", err)
	}

	table, err := NewBlockTable(pool)
	if err != nil {
		t.Fatalf("NewBlockTable: %v", err)
	}

	if _, err := NewBlockTable(nil); err != ErrNilTable {
		t.Errorf("expected ErrNilTable, got %v", err)
	}

	// Append 25 tokens with block size 16 -> should allocate 2 blocks (capacity 32)
	if err := table.AppendTokens(25); err != nil {
		t.Fatalf("AppendTokens(25) failed: %v", err)
	}

	if table.TokenCount() != 25 {
		t.Errorf("TokenCount = %d, want 25", table.TokenCount())
	}

	blocks := table.PhysicalBlocks()
	if len(blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(blocks))
	}

	// Check token addressing
	// Token 0 -> block 0, offset 0
	bID, offset, err := table.PhysicalBlockForToken(0)
	if err != nil || bID != blocks[0] || offset != 0 {
		t.Errorf("pos 0: got block %d offset %d err %v, want block %d offset 0", bID, offset, err, blocks[0])
	}

	// Token 16 -> block 1, offset 0
	bID, offset, err = table.PhysicalBlockForToken(16)
	if err != nil || bID != blocks[1] || offset != 0 {
		t.Errorf("pos 16: got block %d offset %d err %v, want block %d offset 0", bID, offset, err, blocks[1])
	}

	// Token 24 -> block 1, offset 8
	bID, offset, err = table.PhysicalBlockForToken(24)
	if err != nil || bID != blocks[1] || offset != 8 {
		t.Errorf("pos 24: got block %d offset %d err %v, want block %d offset 8", bID, offset, err, blocks[1])
	}

	// Out of bounds pos 25
	if _, _, err := table.PhysicalBlockForToken(25); err == nil {
		t.Error("expected error for token pos 25 (bounds [0, 25))")
	}

	// Append 5 more tokens -> fits in already allocated 2 blocks (total 30 <= 32)
	if err := table.AppendTokens(5); err != nil {
		t.Fatalf("AppendTokens(5) failed: %v", err)
	}
	if len(table.PhysicalBlocks()) != 2 {
		t.Errorf("expected still 2 blocks, got %d", len(table.PhysicalBlocks()))
	}

	// Fork table (prefix sharing / CoW)
	forked, err := table.Fork()
	if err != nil {
		t.Fatalf("Fork failed: %v", err)
	}
	if forked.TokenCount() != 30 {
		t.Errorf("forked token count = %d, want 30", forked.TokenCount())
	}

	// Releasing original table leaves physical blocks referenced by forked
	table.Release()
	if pool.AllocatedBlocks() != 2 {
		t.Errorf("forked table should keep 2 blocks allocated, got %d", pool.AllocatedBlocks())
	}

	forked.Release()
	if pool.AllocatedBlocks() != 0 {
		t.Errorf("all blocks should be freed, got %d allocated", pool.AllocatedBlocks())
	}
}

func BenchmarkBlockTableAppend(b *testing.B) {
	cfg := DefaultConfig()
	pool, err := NewPagePool(cfg, 100000)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		table, _ := NewBlockTable(pool)
		_ = table.AppendTokens(128)
		table.Release()
	}
}
