package radixkv

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// keyA is a complete materialization binding for "model-A". Every axis the #432 key
// enumerates is populated so MaterializationKey.Complete() holds.
func keyA() cachemeta.MaterializationKey {
	return cachemeta.MaterializationKey{
		ModelID:         "model-A",
		TokenizerID:     "tok-A",
		SerializerID:    "serde-1",
		PositionRegime:  "rope:theta=1e4",
		PolicyVersion:   "policy-7",
		AdmitterVersion: "admitter-3",
	}
}

// seed inserts the token run as a single cached prefix (pure-accounting mode, nil KV)
// so the tree holds one edge to match/split against.
func seed(t *testing.T, b *BoundTree, tokens []int) {
	t.Helper()
	n, m := b.Tree.Lookup(tokens)
	leaf := b.Tree.Insert(n, tokens[m:], nil)
	b.Tree.Done(leaf)
}

// TestBoundTreeServesOnlyItsOwnBinding witnesses #432 at the KV REUSE POINT: a radix
// tree pinned to model-A serves a model-A request but FAILS CLOSED for a model-B
// request that shares a token prefix — without taking a lease or splitting an edge —
// so a KV span computed under one model is never reused under another.
func TestBoundTreeServesOnlyItsOwnBinding(t *testing.T) {
	b := NewBound(0, keyA())
	seed(t, b, []int{1, 2, 3, 4, 5})

	// A model-B request keyed to a different model. It shares the prefix [1,2,3] and
	// then diverges MID-EDGE at offset 3, so a binding-blind Lookup would SPLIT the
	// edge and lease the new boundary — the exact cross-model reuse #432 forbids.
	keyB := keyA()
	keyB.ModelID = "model-B"

	before := b.Tree.Stats()
	n, matched, ok := b.Lookup(keyB, []int{1, 2, 3, 9})
	if ok || n != nil || matched != 0 {
		t.Fatalf("a model-B request must fail closed against a model-A tree: ok=%v n=%v matched=%d", ok, n, matched)
	}
	if admit, reason := b.Reusable(keyB); admit || reason != cachemeta.ReasonModelMismatch {
		t.Fatalf("Reusable(model-B) must refuse with model_mismatch, got admit=%v reason=%q", admit, reason)
	}
	// The refused lookup must NOT have mutated the tree: no split, no new node, so no
	// lease can be dangling against the budget.
	if after := b.Tree.Stats(); after.Splits != before.Splits || after.Nodes != before.Nodes {
		t.Fatalf("a refused cross-binding lookup must not split or grow the tree: before=%+v after=%+v", before, after)
	}

	// The SAME query under the tree's OWN binding is admitted: it reuses the [1,2,3]
	// prefix, which (because it diverges mid-edge) splits the edge — proving the guard,
	// not a benign query, is what gated the mutation above.
	n2, matched2, ok2 := b.Lookup(keyA(), []int{1, 2, 3, 9})
	if !ok2 || n2 == nil || matched2 != 3 {
		t.Fatalf("a matching request must reuse the shared prefix: ok=%v n=%v matched=%d", ok2, n2, matched2)
	}
	b.Tree.Done(n2)
	if got := b.Tree.Stats().Splits; got != before.Splits+1 {
		t.Fatalf("the admitted lookup should have split the diverging edge exactly once, got Splits=%d", got)
	}
}

// TestBoundTreeEveryAxisIsLoadBearing witnesses #432 acceptance #2 at the reuse
// point: flipping ANY one binding axis alone fails the reuse closed with its typed
// reason, and clearing any axis (an incomplete key) fails closed too.
func TestBoundTreeEveryAxisIsLoadBearing(t *testing.T) {
	b := NewBound(0, keyA())

	if admit, reason := b.Reusable(keyA()); !admit || reason != cachemeta.ReasonNone {
		t.Fatalf("an identical binding must be reusable with ReasonNone, got admit=%v reason=%q", admit, reason)
	}

	cases := []struct {
		name   string
		mutate func(cachemeta.MaterializationKey) cachemeta.MaterializationKey
		want   cachemeta.LookupReason
	}{
		{"model", func(k cachemeta.MaterializationKey) cachemeta.MaterializationKey { k.ModelID = "x"; return k }, cachemeta.ReasonModelMismatch},
		{"tokenizer", func(k cachemeta.MaterializationKey) cachemeta.MaterializationKey { k.TokenizerID = "x"; return k }, cachemeta.ReasonTokenizerMismatch},
		{"serializer", func(k cachemeta.MaterializationKey) cachemeta.MaterializationKey { k.SerializerID = "x"; return k }, cachemeta.ReasonSerializerMismatch},
		{"position", func(k cachemeta.MaterializationKey) cachemeta.MaterializationKey { k.PositionRegime = "x"; return k }, cachemeta.ReasonPositionMismatch},
		{"policy", func(k cachemeta.MaterializationKey) cachemeta.MaterializationKey { k.PolicyVersion = "x"; return k }, cachemeta.ReasonPolicyMismatch},
		{"admitter", func(k cachemeta.MaterializationKey) cachemeta.MaterializationKey { k.AdmitterVersion = "x"; return k }, cachemeta.ReasonAdmitterMismatch},
	}
	for _, tc := range cases {
		if admit, reason := b.Reusable(tc.mutate(keyA())); admit || reason != tc.want {
			t.Fatalf("%s axis: a divergent binding must refuse with %q, got admit=%v reason=%q", tc.name, tc.want, admit, reason)
		}
	}

	// An incomplete requesting key cannot prove a match — fail closed.
	var empty cachemeta.MaterializationKey
	if admit, _ := b.Reusable(empty); admit {
		t.Fatal("an incomplete requesting key must fail closed")
	}
	// A tree bound to an incomplete key never admits reuse, even against itself.
	unbound := NewBound(0, cachemeta.MaterializationKey{ModelID: "only-model"})
	if admit, _ := unbound.Reusable(keyA()); admit {
		t.Fatal("a tree with an incomplete binding must fail closed")
	}
}
