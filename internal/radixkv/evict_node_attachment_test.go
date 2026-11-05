package radixkv

import "testing"

// fak-test:runtime fast est=10ms lane=default
// Estimate only: tiny accounting-mode trees, with no model or device work.
func TestEvictNodeRequiresAttachedHandle(t *testing.T) {
	t.Parallel()
	for _, ns := range []string{"", "tenant"} {
		t.Run("namespace="+ns, func(t *testing.T) {
			tree, other := New(0), New(0)
			insert := func(owner *Tree, tokens []int) *node {
				boundary, matched := owner.LookupNS(ns, tokens)
				return owner.Insert(boundary, tokens[matched:], nil)
			}
			ancestor := insert(tree, []int{1, 2})
			tree.Done(ancestor)
			child := insert(tree, []int{1, 2, 3, 4})
			sibling := insert(tree, []int{9, 10})
			tree.Done(sibling)
			foreign := insert(other, []int{1, 2, 3, 4})
			other.Done(foreign)
			if child.parent != ancestor || child.refs != 1 {
				t.Fatal("fixture must retain an attached, leased descendant")
			}

			// Policy eviction still removes a valid subtree despite a live lease.
			if got := tree.EvictNode(ancestor); got != 4 {
				t.Fatalf("attached subtree: freed %d, want 4", got)
			}
			if tree.tokens != 2 || tree.policyEvictions != 1 || child.refs != 1 {
				t.Fatalf("first eviction: tokens=%d policies=%d child refs=%d", tree.tokens, tree.policyEvictions, child.refs)
			}
			tree.Done(child)
			if child.refs != 0 {
				t.Fatal("evicted lease must remain releasable")
			}
			replacement := insert(tree, []int{1, 2, 3, 4})
			tree.Done(replacement)
			if replacement == ancestor || replacement == child || ancestor.children[3] != child {
				t.Fatal("fixture must keep a stale subtree beside a fresh replacement")
			}

			for _, tc := range []struct {
				name string
				n    *node
			}{
				{"nil", nil},
				{"root", tree.rootForRead(ns)},
				{"detached ancestor", ancestor},
				{"detached descendant", child},
				{"foreign tree", foreign},
			} {
				if got := tree.EvictNode(tc.n); got != 0 {
					t.Fatalf("%s: freed %d, want no-op", tc.name, got)
				}
				if tree.tokens != 6 || tree.policyEvictions != 1 || other.tokens != 4 || other.policyEvictions != 0 {
					t.Fatalf("%s changed accounting: local=(%d,%d) foreign=(%d,%d)", tc.name, tree.tokens, tree.policyEvictions, other.tokens, other.policyEvictions)
				}
				if !tree.nodeAttached(replacement) || !tree.nodeAttached(sibling) || !other.nodeAttached(foreign) || ancestor.children[3] != child {
					t.Fatalf("%s changed local, foreign, or detached topology", tc.name)
				}
			}

			// The guard must not disable fresh local or namespace-rooted handles.
			if got := tree.EvictNode(replacement); got != 4 || tree.tokens != 2 || tree.policyEvictions != 2 || !tree.nodeAttached(sibling) {
				t.Fatalf("replacement eviction: freed=%d tokens=%d policies=%d", got, tree.tokens, tree.policyEvictions)
			}
			if got := other.EvictNode(foreign); got != 4 || other.tokens != 0 || other.policyEvictions != 1 {
				t.Fatalf("foreign owner's own eviction: freed=%d tokens=%d policies=%d", got, other.tokens, other.policyEvictions)
			}
		})
	}
}
