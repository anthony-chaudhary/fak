package radixkv

import "testing"

// fak-test:runtime fast est=10ms lane=default

func sessTree(t *testing.T, seqs ...[]int) *Tree {
	t.Helper()
	tr := New(0)
	for _, s := range seqs {
		b, m := tr.Lookup(s)
		if m < len(s) {
			tr.Insert(b, s[m:], nil)
		}
	}
	return tr
}

func nodeAt(tr *Tree, toks []int) *node {
	n, l := tr.Lookup(toks)
	if l != len(toks) {
		return nil
	}
	tr.Done(n)
	return n
}

func TestSessionSharedAncestorLastHolderRelease(t *testing.T) {
	tr := sessTree(t, []int{1, 2, 3, 4}, []int{1, 2, 5, 6})
	a, b := tr.OpenSession("A"), tr.OpenSession("B")
	if tr.RegisterSessionLeaf("", "A", a, []int{1, 2, 3, 4}) != 4 || tr.RegisterSessionLeaf("", "B", b, []int{1, 2, 5, 6}) != 4 {
		t.Fatal("register did not cover full path")
	}
	anc := nodeAt(tr, []int{1, 2})
	if anc == nil || anc.holderRefs != 2 {
		t.Fatalf("ancestor refs: %+v", anc)
	}
	tr.ReleaseSession("A")
	if !tr.isNodePinnedOrImmune(anc) {
		t.Fatal("ancestor must stay immune while B holds it")
	}
	if tr.isNodePinnedOrImmune(nodeAt(tr, []int{1, 2, 3, 4})) {
		t.Fatal("A's leaf must be evictable")
	}
	tr.ReleaseSession("B")
	if tr.isNodePinnedOrImmune(anc) || tr.SessionImmuneNodes() != 0 {
		t.Fatal("ancestor must be evictable after last holder releases")
	}
}

func TestSessionReleaseDoesNotClearTier0Pin(t *testing.T) {
	tr := sessTree(t, []int{1, 2, 3})
	tr.PinPrefix([]int{1, 2, 3})
	g := tr.OpenSession("A")
	tr.RegisterSessionLeaf("", "A", g, []int{1, 2, 3})
	tr.ReleaseSession("A")
	if !tr.isNodePinnedOrImmune(nodeAt(tr, []int{1, 2, 3})) {
		t.Fatal("Tier0 pin lost after ReleaseSession")
	}
}

func TestSessionLateFinishAfterCloseIgnored(t *testing.T) {
	tr := sessTree(t, []int{1, 2, 3})
	g := tr.OpenSession("A")
	tr.ReleaseSession("A")
	if tr.RegisterSessionLeaf("", "A", g, []int{1, 2, 3}) != 0 || tr.SessionImmuneNodes() != 0 {
		t.Fatal("late register after close must be ignored")
	}
}

func TestSessionStaleGenerationIgnored(t *testing.T) {
	tr := sessTree(t, []int{1, 2, 3})
	old := tr.OpenSession("A")
	cur := tr.OpenSession("A")
	if tr.RegisterSessionLeaf("", "A", old, []int{1, 2, 3}) != 0 || tr.SessionImmuneNodes() != 0 {
		t.Fatal("stale generation must be ignored")
	}
	if tr.RegisterSessionLeaf("", "A", cur, []int{1, 2, 3}) != 3 {
		t.Fatal("current generation must register")
	}
}

func TestSessionTombstoneBounded(t *testing.T) {
	tr := New(0)
	for i := 0; i < sessionTombstoneCap+100; i++ {
		id := string(rune('a'+i%26)) + string(rune(i))
		tr.OpenSession(id)
		tr.ReleaseSession(id)
	}
	if got := tr.sessions.tombLRU.Len(); got != sessionTombstoneCap || len(tr.sessions.tombs) != sessionTombstoneCap {
		t.Fatalf("tombstone size %d, want %d", got, sessionTombstoneCap)
	}
}

func TestSessionAdvanceCoverageNoDoubleCount(t *testing.T) {
	tr := sessTree(t, []int{1, 2, 3, 4, 5})
	g := tr.OpenSession("A")
	tr.RegisterSessionLeaf("", "A", g, []int{1, 2, 3})
	tr.RegisterSessionLeaf("", "A", g, []int{1, 2, 3, 4, 5})
	tr.RegisterSessionLeaf("", "A", g, []int{1, 2, 3, 4, 5})
	root := tr.root
	for p := nodeAt(tr, []int{1, 2, 3, 4, 5}); p != nil && p != root; p = p.parent {
		if p.holderRefs != 1 {
			t.Fatalf("node plen=%d refs=%d, want 1", p.plen, p.holderRefs)
		}
	}
	tr.ReleaseSession("A")
	if tr.SessionImmuneNodes() != 0 {
		t.Fatal("refs leaked after release")
	}
}
