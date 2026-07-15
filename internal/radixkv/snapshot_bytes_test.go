package radixkv

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func sizedHostSnapshot(t *testing.T, cfg model.Config, tokens int) *model.PrefixSnapshot {
	t.Helper()
	cache := model.NewKVCache(cfg)
	// The cache payload is intentionally populated directly by a real model cache clone
	// path in model tests; here logits provide a deterministic cross-package byte payload.
	return model.NewHostPrefixSnapshotForTest(cache)
}

func TestSnapshotByteBudgetEvictsUnleasedSnapshot(t *testing.T) {
	first := model.NewHostPrefixSnapshotForTest(model.NewKVCache(model.Config{}))
	second := model.NewHostPrefixSnapshotForTest(model.NewKVCache(model.Config{}))
	logits := make([]float32, 32) // 128 resident bytes per complete snapshot
	tree := NewWithBudgets(1000, 128)

	root, _ := tree.Lookup(nil)
	leaf1, err := tree.InsertSnapshot(root, []int{1}, first, logits)
	if err != nil {
		t.Fatal(err)
	}
	tree.Done(leaf1)
	root, _ = tree.Lookup(nil)
	leaf2, err := tree.InsertSnapshot(root, []int{2}, second, logits)
	if err != nil {
		t.Fatal(err)
	}
	tree.Done(leaf2)

	stats := tree.Stats()
	if stats.SnapshotBytes != 128 || stats.MaxSnapshotBytes != 128 {
		t.Fatalf("snapshot byte stats = %d/%d, want 128/128", stats.SnapshotBytes, stats.MaxSnapshotBytes)
	}
	_, got1, _, err := tree.LookupSnapshot([]int{1})
	if err != nil {
		t.Fatal(err)
	}
	if got1 != nil {
		got1.Close()
		t.Fatal("old unleased snapshot survived byte-budget eviction")
	}
	_, got2, _, err := tree.LookupSnapshot([]int{2})
	if err != nil {
		t.Fatal(err)
	}
	if got2 == nil {
		t.Fatal("new snapshot was not admitted")
	}
	got2.Close()
}

func TestSnapshotByteBudgetRejectsLeasedWithoutOwnershipTransfer(t *testing.T) {
	tree := NewWithBudgets(1000, 64)
	logits := make([]float32, 16)
	first := model.NewHostPrefixSnapshotForTest(model.NewKVCache(model.Config{}))
	root, _ := tree.Lookup(nil)
	leaf, err := tree.InsertSnapshot(root, []int{1}, first, logits)
	if err != nil {
		t.Fatal(err)
	}
	// Keep leaf leased so it cannot be evicted to admit another complete snapshot.
	second := model.NewHostPrefixSnapshotForTest(model.NewKVCache(model.Config{}))
	root2, _ := tree.Lookup(nil)
	rejected, err := tree.InsertSnapshot(root2, []int{2}, second, logits)
	if rejected != nil {
		tree.Done(rejected)
	}
	if !errors.Is(err, ErrSnapshotByteBudget) {
		t.Fatalf("error = %v", err)
	}
	if tree.Stats().SnapshotBytes != 64 {
		t.Fatalf("bytes changed after rejection: %d", tree.Stats().SnapshotBytes)
	}
	if second.ResidentBytes() != 0 {
		t.Fatalf("caller-owned rejected snapshot mutated")
	}
	second.Close()
	tree.Done(leaf)
}
