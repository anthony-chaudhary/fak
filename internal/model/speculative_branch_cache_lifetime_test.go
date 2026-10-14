package model

import "testing"

// Explicit detachment is reversible while the session is live.
// fak-test:runtime fast est=1ms lane=default
func TestBranchCacheDetachAllowsLiveReattachment(t *testing.T) {
	s := &Session{}
	t.Cleanup(func() { DetachBranchCache(s); s.Close() })
	first := NewSpeculativeBranchCache(SpeculativeBranchCacheConfig{MaxEntries: 1})
	second := NewSpeculativeBranchCache(SpeculativeBranchCacheConfig{MaxEntries: 1})
	AttachBranchCache(s, first)
	if GetSessionBranchCache(s) != first {
		t.Fatal("fresh session rejected its branch cache")
	}
	DetachBranchCache(s)
	if GetSessionBranchCache(s) != nil {
		t.Fatal("explicit detachment retained its branch cache")
	}
	AttachBranchCache(s, second)
	if GetSessionBranchCache(s) != second {
		t.Fatal("explicit detachment retired a live session")
	}
	AttachBranchCache(s, nil)
	if GetSessionBranchCache(s) != nil {
		t.Fatal("nil-cache detachment retained its branch cache")
	}
	AttachBranchCache(s, first)
	if GetSessionBranchCache(s) != first {
		t.Fatal("nil-cache detachment retired a live session")
	}
	s.Close()
	if GetSessionBranchCache(s) != nil {
		t.Fatal("close retained a reattached branch cache")
	}
}
