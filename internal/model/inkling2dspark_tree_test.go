package model

import "testing"

// Width 1 is Draft's greedy chain; a wider tree keeps that chain's first token
// as its best root, orders every parent before its children and honours the
// node budget.
func TestInkling2DSparkDraftTreeMatchesReference(t *testing.T) {
	features := inkling2DSparkTestFeatures(9, 3, 16, 1)
	draft := func() *Inkling2DSparkDraft {
		d := inkling2DSparkTestDraft(t, []bool{false, true}, 4)
		for i, f := range features {
			if err := d.Commit(i, f); err != nil {
				t.Fatal(err)
			}
		}
		return d
	}
	pos := len(features)
	d := draft()
	greedy, err := d.Draft(5, pos)
	if err != nil {
		t.Fatal(err)
	}
	chain, err := d.DraftTree(5, pos, 1, 100)
	if err != nil {
		t.Fatal(err)
	}
	if len(chain) != len(greedy) {
		t.Fatalf("width-1 tree has %d nodes, greedy drafted %d", len(chain), len(greedy))
	}
	for i, n := range chain {
		if n.Token != greedy[i] || n.Parent != i-1 {
			t.Fatalf("width-1 node %d = %+v, want token %d parent %d (greedy %v)", i, n, greedy[i], i-1, greedy)
		}
	}

	const width = 4
	tree, err := draft().DraftTree(5, pos, width, 100)
	if err != nil {
		t.Fatal(err)
	}
	depth := make([]int, len(tree))
	perDepth := map[int]int{}
	for i, n := range tree {
		if n.Parent < -1 || n.Parent >= i {
			t.Fatalf("node %d parent %d does not precede it", i, n.Parent)
		}
		if n.Parent >= 0 {
			depth[i] = depth[n.Parent] + 1
		}
		if i > 0 && depth[i] < depth[i-1] {
			t.Fatalf("node %d at depth %d after depth %d", i, depth[i], depth[i-1])
		}
		perDepth[depth[i]]++
	}
	if len(tree) != width*len(greedy) || perDepth[0] != width {
		t.Fatalf("tree of %d nodes, depths %v; want %d per depth over %d depths", len(tree), perDepth, width, len(greedy))
	}
	if tree[0].Token != greedy[0] {
		t.Fatalf("best root %d, greedy first token %d", tree[0].Token, greedy[0])
	}
	seen := map[int]bool{}
	for _, n := range tree[:width] {
		if seen[n.Token] {
			t.Fatalf("duplicate root token %d", n.Token)
		}
		seen[n.Token] = true
	}

	small, err := draft().DraftTree(5, pos, width, width+1)
	if err != nil {
		t.Fatal(err)
	}
	if len(small) != width+1 || small[width].Parent < 0 {
		t.Fatalf("budget %d gave %d nodes (%v)", width+1, len(small), small)
	}
	for i := range small {
		if small[i] != tree[i] {
			t.Fatalf("budgeted node %d = %+v, unbudgeted %+v", i, small[i], tree[i])
		}
	}
	if _, err := d.DraftTree(5, pos, 0, 4); err == nil {
		t.Fatal("width 0 accepted")
	}
}
