package model

import "testing"

// A retired session must not regain a strong registry link through automatic attachment.
// fak-test:runtime fast est=1ms lane=default
func TestSessionCloseRejectsBranchCacheReattachment(t *testing.T) {
	m := &Model{Cfg: Config{NumLayers: 1, NumKVHeads: 1, HeadDim: 1}}
	s := m.NewSession()
	t.Cleanup(func() {
		DetachBranchCache(s)
		s.Close()
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
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
	s.Close()
	engine := NewSpeculativeEngine(s, nil, SpeculativeEngineConfig{BranchCache: cache})
	reattached := GetSessionBranchCache(s) != nil
	s.Close()
	if engine.BranchCache() != cache {
		t.Error("engine lost its caller-owned cache")
	}
	entry, ok := cache.Probe(prefix, branch)
	if bytes <= 0 || !ok || entry == nil || len(entry.Logits) != 1 || len(entry.Logits[0]) != 2 ||
		entry.Logits[0][0] != 0.25 || entry.Logits[0][1] != 0.75 || cache.Len() != 1 || cache.Bytes() != bytes {
		t.Error("closed-session attachment changed caller-owned cache contents")
	}
	if reattached || GetSessionBranchCache(s) != nil {
		t.Error("retired session accepted a branch-cache registration")
	}
}
