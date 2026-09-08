package radixkv

import (
	"github.com/anthony-chaudhary/fak/internal/model"
	"testing"
)

func TestCPUCacheBudgetProtectsPinnedPayloadBeforeClone(t *testing.T) {
	tree := New(0)
	tree.SetCPUCacheByteBudget(28)
	kv := model.NewKVCache(model.Config{NumLayers: 1})
	kv.K[0] = []float32{1, 2, 3, 4, 5, 6}
	b, _ := tree.Lookup([]int{1, 2})
	leaf := tree.InsertCloneWithLogits(b, []int{1, 2}, kv, []float32{7})
	if tree.Stats().CPUCacheBytes != 28 {
		t.Fatal("first payload not retained")
	}
	b, _ = tree.Lookup([]int{3})
	got := tree.InsertCloneWithLogits(b, []int{3}, kv, []float32{9})
	if got != b || tree.Stats().CPUCacheLastBypass != "pinned" || leaf.KV() == nil {
		t.Fatal("pinned payload was displaced")
	}
	tree.Done(got)
	// A split would clone the full backing arrays, even for a short prefix.
	split, _ := tree.Lookup([]int{1, 9})
	if split.KV() != nil || tree.Stats().CPUCacheBytes != 28 {
		t.Fatal("split exceeded byte budget")
	}
	tree.Done(split)
	// Exact-hit logits growth is refused before replacing the old payload.
	got = tree.InsertWithLogits(leaf, nil, nil, []float32{1, 2})
	if got != leaf || leaf.logits[0] != 7 || tree.Stats().CPUCacheBytes != 28 {
		t.Fatal("logits replacement exceeded cap")
	}
	tree.Done(leaf)
	b, _ = tree.Lookup([]int{3})
	got = tree.InsertCloneWithLogits(b, []int{3}, kv, []float32{9})
	tree.Done(got)
	if tree.Stats().CPUCacheBytes != 28 || got.KV() == nil || leaf.KV() != nil {
		t.Fatal("released victim was not reclaimed")
	}
	kv.K[0][0] = 99
	if got.KV().K[0][0] != 1 {
		t.Fatal("cache aliases session source")
	}
}
