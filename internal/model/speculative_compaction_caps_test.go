package model

import (
	"context"
	"testing"
)

func TestKVTreeCompactionBranchCacheCapturesBeforeCompactionWithinByteCap(t *testing.T) {
	m := NewSynthetic(cfgV(32, 2, 2, 1, 16, 64))
	target := m.NewSession()
	prompt := []int{1, 2, 3}
	last := target.Prefill(prompt)
	root := argmaxF32(last)

	// The root candidate is accepted; its sibling is discarded. A retained entry can
	// only contain that sibling's KV if preservation ran before tree compaction removed it.
	tree := &CandidateTree{Nodes: []CandidateNode{
		{Token: root, Parent: -1, Depth: 0},
		{Token: (root + 1) % m.Cfg.VocabSize, Parent: -1, Depth: 0},
	}}
	proposal := DraftProposal{Tree: tree, Tokens: tree.Tokens()}
	const byteCap = int64(64 << 10)
	cache := NewSpeculativeBranchCache(SpeculativeBranchCacheConfig{
		MaxEntries:   16,
		MaxBytes:     byteCap,
		MaxBranchLen: 4,
	})
	AttachBranchCache(target, cache)
	t.Cleanup(func() { DetachBranchCache(target) })

	result, err := ParallelVerifyKernel(context.Background(), target, prompt, proposal, last, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.NumAccepted != 1 || target.Cache.Len() != len(prompt)+1 {
		t.Fatalf("accepted/cache length=%d/%d, want 1/%d", result.NumAccepted, target.Cache.Len(), len(prompt)+1)
	}
	stats := cache.Stats()
	if stats.PreservedBranches == 0 || cache.Len() == 0 {
		t.Fatalf("discarded sibling was not retained before compaction: stats=%+v len=%d", stats, cache.Len())
	}
	if cache.Bytes() <= 0 || cache.Bytes() > byteCap || stats.BytesUsed != cache.Bytes() {
		t.Fatalf("retained bytes=%d stats=%d, want 0 < bytes <= %d", cache.Bytes(), stats.BytesUsed, byteCap)
	}
	entry, ok := cache.Probe(prompt, []int{tree.Nodes[1].Token})
	if !ok || entry == nil || len(entry.K) == 0 || len(entry.Kraw) == 0 || len(entry.V) == 0 {
		t.Fatal("discarded sibling entry lacks the pre-compaction KV planes")
	}
}

func TestKVTreeCompactionBranchCacheByteCapEvictsOldest(t *testing.T) {
	prompt := []int{7, 8}
	entryBytes := branchCacheEntryBytes(t, prompt, []int{10, 11})
	cache := NewSpeculativeBranchCache(SpeculativeBranchCacheConfig{
		MaxEntries:   16,
		MaxBytes:     entryBytes,
		MaxBranchLen: 4,
	})
	preserveBranchForCapTest(t, cache, prompt, []int{10, 11})
	preserveBranchForCapTest(t, cache, prompt, []int{20, 21})

	if cache.Bytes() > entryBytes {
		t.Fatalf("retained bytes=%d exceed configured cap=%d", cache.Bytes(), entryBytes)
	}
	if cache.Len() != 1 || cache.Stats().Evictions != 1 {
		t.Fatalf("byte-driven eviction len/evictions=%d/%d, want 1/1", cache.Len(), cache.Stats().Evictions)
	}
	if _, ok := cache.Probe(prompt, []int{10, 11}); ok {
		t.Fatal("oldest branch survived byte-driven eviction")
	}
	if _, ok := cache.Probe(prompt, []int{20, 21}); !ok {
		t.Fatal("newest branch was not retained after byte-driven eviction")
	}
}

func TestKVTreeCompactionBranchCacheOversizedEntryIsNotRetained(t *testing.T) {
	prompt := []int{7, 8}
	branch := []int{30, 31}
	entryBytes := branchCacheEntryBytes(t, prompt, branch)
	cache := NewSpeculativeBranchCache(SpeculativeBranchCacheConfig{
		MaxEntries:   16,
		MaxBytes:     entryBytes - 1,
		MaxBranchLen: 4,
	})
	preserveBranchForCapTest(t, cache, prompt, branch)

	if cache.Len() != 0 || cache.Bytes() != 0 {
		t.Fatalf("oversized entry remained resident: len=%d bytes=%d cap=%d", cache.Len(), cache.Bytes(), entryBytes-1)
	}
	stats := cache.Stats()
	if stats.PreservedBranches != 1 || stats.Evictions != 1 || stats.BytesUsed != 0 {
		t.Fatalf("oversized-entry accounting=%+v, want one attempted preservation and immediate eviction", stats)
	}
}

func branchCacheEntryBytes(t *testing.T, prompt, branch []int) int64 {
	t.Helper()
	cache := NewSpeculativeBranchCache(SpeculativeBranchCacheConfig{MaxEntries: 16, MaxBranchLen: 4})
	preserveBranchForCapTest(t, cache, prompt, branch)
	if cache.Len() != 1 || cache.Bytes() <= 0 {
		t.Fatalf("calibration entry len/bytes=%d/%d, want 1/positive", cache.Len(), cache.Bytes())
	}
	return cache.Bytes()
}

func preserveBranchForCapTest(t *testing.T, cache *SpeculativeBranchCache, prompt, branch []int) {
	t.Helper()
	tree, err := BuildCandidateTreeFromBranches([][]int{branch})
	if err != nil {
		t.Fatal(err)
	}
	rows := make([][]float32, len(tree.Nodes))
	for i := range rows {
		rows[i] = []float32{float32(i), float32(i + 1), float32(i + 2), float32(i + 3)}
	}
	if got := cache.PreserveUnacceptedBranches(nil, prompt, tree, nil, 0, rows); got != 1 {
		t.Fatalf("preserved branches=%d, want 1", got)
	}
}
