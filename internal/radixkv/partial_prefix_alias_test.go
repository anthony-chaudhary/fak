package radixkv

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestInsertWithLogitsAliasesTheCallerCache pins a load-bearing hazard for the
// chunk-level coalescing work in fak#13661.
//
// Tree.InsertWithLogits stores the caller's *model.KVCache POINTER
// (radixkv.go: attachLeafWithChunk assigns `kv: kv` with no clone). A radix node
// therefore does NOT own an immutable snapshot of the cache it was inserted
// with: it keeps observing that same cache as the owning lane appends to it —
// during decode Step() or during a further chunked-prefill chunk.
//
// That is benign for the one call site that exists today,
// modelengine.(*NativeScheduler).maybeInsertRadixKV, which inserts only on the
// FINAL chunk of a whole prompt: the node's token key and its KV length then
// agree, and the node growing alongside its own request's decode is the
// intended prefix-cache behaviour.
//
// It is NOT benign for publishing a PARTIAL prefix between chunks, which is the
// obvious cheap way to let a cold twin adopt a chunked lane's in-flight prefix:
// the node's token key would name `end` tokens while its KV keeps growing
// toward the full prompt, so a later Lookup that matches the shorter key can
// hand an adopting lane a KV LONGER than its prompt cursor. This test makes that
// aliasing explicit and measurable so the next implementer does not have to
// rediscover it, and so any future change that starts cloning on insert is caught
// here as a behavioural difference rather than silently shifting memory cost.
//
// fak-test:runtime fast est=200ms
func TestInsertWithLogitsAliasesTheCallerCache(t *testing.T) {
	cfg := model.Config{
		HiddenSize: 1, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 1,
		IntermediateSize: 1, VocabSize: 8,
	}
	m := model.NewSynthetic(cfg)
	sess := m.NewSession()
	defer sess.Close()

	prompt := []int{1, 2, 3}
	sess.Prefill(prompt)
	live := sess.Cache
	if live.Len() != len(prompt) {
		t.Fatalf("prefilled cache length = %d, want %d", live.Len(), len(prompt))
	}

	tree := New(0)
	root, _ := tree.Lookup(prompt)
	leaf := tree.InsertWithLogits(root, prompt, live, nil)
	tree.Done(leaf)

	node, _ := tree.Lookup(prompt)
	if node == nil {
		t.Fatal("inserted prefix did not resolve on lookup")
	}
	defer tree.Done(node)
	if node.KV() != live {
		t.Fatal("node does not hold the caller's cache object; this test pins the aliasing contract and must be revisited if Insert starts cloning")
	}
	if got := node.KV().Len(); got != len(prompt) {
		t.Fatalf("node KV length after insert = %d, want %d", got, len(prompt))
	}

	// Grow the caller's cache exactly as the owning lane does: one decode step
	// appends a position.
	sess.Step(1)
	grown := live.Len()
	if grown == len(prompt) {
		t.Skipf("decode step did not grow the cache; cannot witness the alias on this config")
	}
	if got := node.KV().Len(); got != grown {
		t.Fatalf("node KV length = %d while the caller's cache is %d; the node no longer tracks the caller, so this characterization changed", got, grown)
	}
	t.Logf("ALIAS CONFIRMED: node token key covers %d tokens while its KV now holds %d; "+
		"partial-prefix publication between chunks would therefore hand an adopting lane a KV longer than its prompt cursor",
		len(prompt), grown)
}
