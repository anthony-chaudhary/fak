package model

import (
	"slices"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestPagedPrefixOwnerPoolBlockGeometry(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name        string
		supplied    bool
		requested   int
		invalidPool bool
		wantError   bool
		wantSize    int
	}{
		{"no_pool_default", false, 0, false, false, 16},
		{"no_pool_negative", false, -1, false, false, 16},
		{"no_pool_explicit", false, 2, false, false, 2},
		{"supplied_omitted", true, 0, false, false, 1},
		{"supplied_negative", true, -1, false, false, 1},
		{"supplied_matching", true, 1, false, false, 1},
		{"supplied_mismatch", true, 2, false, true, 0},
		{"supplied_invalid", true, 0, true, true, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{NumLayers: 1, NumKVHeads: 1, HeadDim: 1}
			var pool *PagedBlockPool
			var victim *PagedPrefixBlock
			var before []float32
			if tc.supplied {
				pool = NewPagedBlockPool(cfg, 1, false)
				victim = pool.Alloc()
				victim.WriteToken(0, [][]float32{{71}}, [][]float32{{72}}, [][]float32{{73}})
				before = slices.Clone(victim.Data())
				defer pool.Release(victim.ID())
				// Keep a reusable page too, so rejection must preserve the free list.
				pool.Release(pool.Alloc().ID())
				if tc.invalidPool {
					pool.blockTokens = 0
				}
			}
			var allocatedBefore, blocksBefore int
			var freeBefore []int
			if pool != nil {
				allocatedBefore = pool.TotalAllocated()
				blocksBefore = len(pool.blocks)
				freeBefore = slices.Clone(pool.freeList)
			}
			owner, err := NewPagedPrefixOwner(PagedPrefixOwnerConfig{
				Config: cfg, Pool: pool, BlockTokens: tc.requested, Tokens: 2,
				K:    [][][]float32{{{11}}, {{21}}},
				Kraw: [][][]float32{{{12}}, {{22}}},
				V:    [][][]float32{{{13}}, {{23}}},
			})
			if tc.wantError {
				if err == nil || owner != nil {
					t.Fatalf("constructor returned (%v, %v), want (nil, error)", owner, err)
				}
				if pool.TotalAllocated() != allocatedBefore || len(pool.blocks) != blocksBefore || !slices.Equal(pool.freeList, freeBefore) {
					t.Error("rejected geometry changed pool allocation or free-list state")
				}
			} else {
				if err != nil || owner == nil {
					t.Fatalf("constructor returned (%v, %v)", owner, err)
				}
				defer owner.Release()
				if pool != nil && owner.Pool() != pool {
					t.Fatal("constructor replaced the supplied pool")
				}
				pool = owner.Pool()
				if owner.blockTokens != tc.wantSize || pool.blockTokens != tc.wantSize {
					t.Fatalf("owner/pool block sizes=%d/%d, want %d", owner.blockTokens, pool.blockTokens, tc.wantSize)
				}
				child, err := owner.ForkSession("geometry")
				if err != nil {
					t.Fatal(err)
				}
				defer child.Release()
				if err := owner.Release(); err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(child.GatherK(0), []float32{11, 21}) ||
					!slices.Equal(child.GatherKraw(0), []float32{12, 22}) ||
					!slices.Equal(child.GatherV(0), []float32{13, 23}) {
					t.Error("forked payload differs after owner release")
				}
				if err := child.Release(); err != nil {
					t.Fatal(err)
				}
			}
			wantActive := 0
			if victim != nil {
				wantActive = 1
				if victim.Refcount() != 1 || victim.IsFreed() || !slices.Equal(victim.Data(), before) {
					t.Error("constructor or owner cleanup changed the existing live page")
				}
			}
			if pool.ActivePages() != wantActive {
				t.Errorf("active pages=%d, want %d", pool.ActivePages(), wantActive)
			}
		})
	}
}
