package ggufload

import (
	"math/rand"
	"reflect"
	"testing"
)

func ink2TreeTestCand(rng *rand.Rand, rows, k int) (path, cand []int) {
	path, cand = make([]int, rows), make([]int, rows*k)
	for r := range rows {
		perm := rng.Perm(50)
		for j := range k {
			cand[r*k+j] = 100*r + perm[j]
		}
		path[r] = cand[r*k+rng.Intn(k)]
	}
	return path, cand
}

func TestInk2TreeWidthOneIsTheChain(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	path, cand := ink2TreeTestCand(rng, 7, 16)
	for _, budget := range []int{1, 4, 8, 16} {
		tr, err := buildInk2Tree(9, path, cand, 16, nil, 1, budget)
		if err != nil {
			t.Fatal(err)
		}
		n := min(budget, len(path)+1)
		if tr.size() != n {
			t.Fatalf("budget %d: %d nodes, want %d", budget, tr.size(), n)
		}
		for i := range n {
			want := 9
			if i > 0 {
				want = path[i-1]
			}
			if int(tr.tokens[i]) != want || int(tr.parent[i]) != i-1 || int(tr.depth[i]) != i || tr.anc[i] != uint32(1)<<(i+1)-1 {
				t.Fatalf("budget %d node %d: token %d parent %d depth %d anc %b", budget, i, tr.tokens[i], tr.parent[i], tr.depth[i], tr.anc[i])
			}
		}
		// The walk is the chain loop's greedy prefix match.
		for a := 0; a < n; a++ {
			argmax := make([]int, n)
			for i := range n {
				argmax[i] = -5 - i
				if i < a {
					argmax[i] = path[i]
				}
			}
			nodes, bonus := tr.accept(argmax)
			if len(nodes) != a+1 || bonus != argmax[a] {
				t.Fatalf("budget %d, %d matches: kept %v, bonus %d", budget, a, nodes, bonus)
			}
		}
	}
}

func TestInk2TreeShapeAndDeterminism(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for trial := range 50 {
		rows, k := 1+rng.Intn(7), 1+rng.Intn(16)
		path, cand := ink2TreeTestCand(rng, rows, k)
		var scores []float32
		if trial%2 == 0 {
			scores = make([]float32, rows*k)
			for i := range scores {
				scores[i] = float32(rng.NormFloat64() * 3)
			}
		}
		width, budget := 1+rng.Intn(4), 1+rng.Intn(32)
		tr, err := buildInk2Tree(3, path, cand, k, scores, width, budget)
		if err != nil {
			t.Fatal(err)
		}
		again, _ := buildInk2Tree(3, path, cand, k, scores, width, budget)
		if !reflect.DeepEqual(tr, again) {
			t.Fatal("two builds of one draft differ")
		}
		if tr.size() > budget {
			t.Fatalf("%d nodes over a budget of %d", tr.size(), budget)
		}
		kids := map[int]map[int32]bool{}
		for i := 1; i < tr.size(); i++ {
			p := int(tr.parent[i])
			if p < 0 || p >= i || tr.depth[i] != tr.depth[p]+1 || tr.anc[i] != tr.anc[p]|1<<i || tr.est[i] > tr.est[p] {
				t.Fatalf("node %d: parent %d depth %d anc %b est %v", i, p, tr.depth[i], tr.anc[i], tr.est[i])
			}
			if tr.est[i] > tr.est[i-1] {
				t.Fatalf("node %d estimate %v above its predecessor's %v", i, tr.est[i], tr.est[i-1])
			}
			if kids[p] == nil {
				kids[p] = map[int32]bool{}
			}
			if kids[p][tr.tokens[i]] {
				t.Fatalf("node %d repeats a sibling's token", i)
			}
			kids[p][tr.tokens[i]] = true
			if len(kids[p]) > width {
				t.Fatalf("node %d has more than %d children", p, width)
			}
			if d := int(tr.depth[i]); tr.ancestorAt(i, d-1) != p || tr.ancestorAt(i, 0) != 0 {
				t.Fatalf("node %d ancestors", i)
			}
		}
		// The greedy chain is always a root path: the chain's tokens come first among siblings.
		at := 0
		for d := 0; d < rows && tr.size() > 1; d++ {
			next := -1
			for c := 1; c < tr.size(); c++ {
				if int(tr.parent[c]) == at && int(tr.tokens[c]) == path[d] {
					next = c
				}
			}
			if next < 0 {
				break
			}
			at = next
		}
		if want := min(budget, rows+1); width == 1 && int(tr.depth[at])+1 != want {
			t.Fatalf("width 1 chain reaches depth %d, want %d", tr.depth[at], want-1)
		}
	}
}

func TestInk2TreeAcceptsABranch(t *testing.T) {
	// rows of 4 candidates; the chain takes candidate 0 each row
	path := []int{10, 20, 30}
	cand := []int{10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33}
	tr, err := buildInk2Tree(1, path, cand, 4, nil, 4, 16)
	if err != nil {
		t.Fatal(err)
	}
	if tr.size() != 16 {
		t.Fatalf("%d nodes", tr.size())
	}
	// target rejects the chain's first token but takes candidate 11, then 20
	argmax := make([]int, tr.size())
	for i := range argmax {
		argmax[i] = -1
	}
	argmax[0] = 11
	var n11 int
	for i := range tr.size() {
		if tr.tokens[i] == 11 && tr.parent[i] == 0 {
			n11 = i
		}
	}
	if n11 == 0 {
		t.Fatal("no node for the second candidate under the root")
	}
	argmax[n11] = 20
	nodes, bonus := tr.accept(argmax)
	hasChild := false
	for i := range tr.size() {
		hasChild = hasChild || (int(tr.parent[i]) == n11 && tr.tokens[i] == 20)
	}
	want := 2
	if hasChild {
		want = 3
	}
	if len(nodes) != want || nodes[1] != n11 {
		t.Fatalf("kept %v, want root, node %d and its child when present (%v)", nodes, n11, hasChild)
	}
	if want == 2 && bonus != 20 {
		t.Fatalf("bonus %d", bonus)
	}
	// prefix trees stay trees and the best row count follows the table
	p := tr.prefix(5)
	if p.size() != 5 {
		t.Fatal("prefix size")
	}
	flat := []float64{0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10}
	if r := ink2TreeBestRows(tr, flat); r != tr.size() {
		t.Fatalf("free rows pick %d", r)
	}
	steep := []float64{0, 10, 1000}
	if r := ink2TreeBestRows(tr, steep); r != 1 {
		t.Fatalf("dear rows pick %d", r)
	}
}

func TestInk2TreeRejectsBadShapes(t *testing.T) {
	for _, c := range []struct{ width, budget, k int }{{0, 8, 2}, {5, 8, 2}, {2, 0, 2}, {2, 33, 2}, {2, 8, 0}} {
		if _, err := buildInk2Tree(0, []int{1}, make([]int, max(c.k, 1)), c.k, nil, c.width, c.budget); err == nil {
			t.Fatalf("accepted %+v", c)
		}
	}
}
