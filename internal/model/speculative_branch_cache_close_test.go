package model

import "testing"

// Closing one session must retire its registry link without destroying a
// caller-owned branch cache that remains attached to another session.
// fak-test:runtime fast est=1ms lane=default
func TestSessionCloseDetachesSharedBranchCache(t *testing.T) {
	m := &Model{Cfg: Config{NumLayers: 1, NumKVHeads: 1, HeadDim: 1}}
	first, second := m.NewSession(), m.NewSession()
	t.Cleanup(func() {
		// Contain the expected pre-fix registry leak even when an assertion fails.
		DetachBranchCache(first)
		DetachBranchCache(second)
		first.Close()
		second.Close()
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	cache := NewSpeculativeBranchCache(SpeculativeBranchCacheConfig{
		MaxEntries: 4, MaxBytes: 1024, MaxBranchLen: 2,
	})
	prefix, branch := []int{1}, []int{2}
	tree, err := BuildCandidateTreeFromBranches([][]int{branch})
	if err != nil {
		t.Fatal(err)
	}
	if n := cache.PreserveUnacceptedBranches(nil, prefix, tree, nil, 0, [][]float32{{0.25, 0.75}}); n != 1 {
		t.Fatalf("preserved branches=%d, want one nonempty caller-owned entry", n)
	}
	bytes := cache.Bytes()
	if bytes <= 0 {
		t.Fatal("caller-owned cache fixture is empty")
	}
	for _, session := range []*Session{first, second} {
		engine := NewSpeculativeEngine(session, nil, SpeculativeEngineConfig{BranchCache: cache})
		if engine.BranchCache() != cache || GetSessionBranchCache(session) != cache {
			t.Fatal("engine did not automatically attach the caller-owned cache")
		}
	}
	assertUsable := func() {
		t.Helper()
		entry, ok := cache.Probe(prefix, branch)
		if !ok || entry == nil || len(entry.Logits) != 1 || len(entry.Logits[0]) != 2 ||
			entry.Logits[0][0] != 0.25 || entry.Logits[0][1] != 0.75 || cache.Len() != 1 || cache.Bytes() != bytes {
			t.Error("session detachment changed the caller-owned cache or its usable entry")
		}
	}
	first.Close()
	first.Close()
	if GetSessionBranchCache(second) != cache {
		t.Error("closing one session detached another session's cache")
	}
	assertUsable()

	DetachBranchCache(second)
	DetachBranchCache(second)
	second.Close()
	second.Close()
	if GetSessionBranchCache(second) != nil {
		t.Error("explicit detachment followed by close was not idempotent")
	}
	assertUsable()
	if GetSessionBranchCache(first) != nil {
		t.Error("closed session retained its branch-cache registration")
	}
}
