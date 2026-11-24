package model

import (
	"math"
	"testing"
)

// Every tree node's logits and feature taps equal a serial decode of its root
// path bitwise, and committing one path (or none) leaves the host decoding
// exactly as if only that path had ever been stepped.
func TestInkling2TreeVerifyMatchesSerialDecode(t *testing.T) {
	hp, w := inkling2TinyModel()
	prefix := []int{1, 4, 2}
	// Depth 4 crosses the 3-token sliding window and the 3-lag conv history.
	nodes := []Inkling2TreeNode{{3, -1}, {5, 0}, {2, 0}, {7, 1}, {1, -1}, {6, 4}, {8, 3}, {9, 6}}
	host := func() *Inkling2LoopHost {
		h, err := NewInkling2LoopHost(hp, w)
		if err != nil {
			t.Fatal(err)
		}
		h.FeatureVisits = []int{1, 3, h.Visits()}
		for _, tok := range prefix {
			h.Step(tok)
		}
		return h
	}
	path := func(i int) []int {
		var toks []int
		for ; i >= 0; i = nodes[i].Parent {
			toks = append([]int{nodes[i].Token}, toks...)
		}
		return toks
	}
	same := func(what string, a, b []float32) {
		t.Helper()
		if len(a) != len(b) {
			t.Fatalf("%s: len %d vs %d", what, len(a), len(b))
		}
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Fatalf("%s[%d]: %v vs serial %v", what, i, a[i], b[i])
			}
		}
	}

	tree := host()
	v, err := tree.VerifyTree(nodes)
	if err != nil {
		t.Fatal(err)
	}
	if tree.pos != len(prefix) {
		t.Fatalf("verify left host at %d, want the committed prefix %d", tree.pos, len(prefix))
	}
	for i := range nodes {
		serial := host()
		var logits []float32
		for _, tok := range path(i) {
			logits = serial.Step(tok)
		}
		same("node logits", v.Logits[i], logits)
		for f := range serial.Features {
			same("node features", v.Features[i][f], serial.Features[f])
		}
	}

	for _, commit := range []int{7, 2, -1} {
		tree := host()
		v, err := tree.VerifyTree(nodes)
		if err != nil {
			t.Fatal(err)
		}
		if err := v.Commit(commit); err != nil {
			t.Fatal(err)
		}
		if err := v.Commit(commit); err == nil {
			t.Fatal("second commit accepted")
		}
		serial := host()
		var walked []int
		if commit >= 0 {
			walked = path(commit)
		}
		for _, tok := range walked {
			serial.Step(tok)
		}
		for s := range serial.keys {
			if len(tree.keys[s]) != len(serial.keys[s]) {
				t.Fatalf("commit %d slot %d holds %d keys, serial %d", commit, s, len(tree.keys[s]), len(serial.keys[s]))
			}
		}
		for _, tok := range []int{4, 0, 6, 5} {
			same("post-commit logits", tree.Step(tok), serial.Step(tok))
		}
	}

	stale := host()
	v, err = stale.VerifyTree(nodes)
	if err != nil {
		t.Fatal(err)
	}
	stale.Step(1)
	if err := v.Commit(0); err == nil {
		t.Fatal("commit after the host advanced was accepted")
	}
	if _, err := stale.VerifyTree([]Inkling2TreeNode{{1, 0}}); err == nil {
		t.Fatal("node naming itself as parent was accepted")
	}
}
