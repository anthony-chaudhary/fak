package model

import "testing"

// Dynamic cache selection must respect target retirement without owning shared cache data.
// fak-test:runtime fast est=1ms lane=default
func TestSpeculativeEngineSetBranchCacheRespectsSessionLifetime(t *testing.T) {
	first, second := &Session{}, &Session{}
	t.Cleanup(func() {
		DetachBranchCache(first)
		DetachBranchCache(second)
		first.Close()
		second.Close()
	})
	cache := NewSpeculativeBranchCache(SpeculativeBranchCacheConfig{
		MaxEntries: 1, MaxBytes: 1024, MaxBranchLen: 1,
	})
	prefix, branch := []int{1}, []int{2}
	tree, err := BuildCandidateTreeFromBranches([][]int{branch})
	if err != nil {
		t.Fatal(err)
	}
	if n := cache.PreserveUnacceptedBranches(nil, prefix, tree, nil, 0, [][]float32{{0.25, 0.75}}); n != 1 {
		t.Fatalf("preserved branches=%d, want one caller-owned entry", n)
	}
	bytes := cache.Bytes()
	AttachBranchCache(second, cache)
	engine := NewSpeculativeEngine(first, nil, SpeculativeEngineConfig{})
	engine.SetBranchCache(cache)
	if engine.BranchCache() != cache || GetSessionBranchCache(first) != cache {
		t.Fatal("SetBranchCache did not attach the cache to its live target")
	}
	first.Close()
	engine.SetBranchCache(cache)
	if GetSessionBranchCache(first) != nil {
		t.Error("SetBranchCache reattached its retired target")
	}
	if engine.BranchCache() != cache || GetSessionBranchCache(second) != cache {
		t.Error("SetBranchCache changed shared caller-owned cache associations")
	}
	entry, ok := cache.Probe(prefix, branch)
	if bytes <= 0 || !ok || entry == nil || len(entry.Logits) != 1 || len(entry.Logits[0]) != 2 ||
		entry.Logits[0][0] != 0.25 || entry.Logits[0][1] != 0.75 || cache.Len() != 1 || cache.Bytes() != bytes {
		t.Error("SetBranchCache changed the caller-owned cache contents")
	}
}
