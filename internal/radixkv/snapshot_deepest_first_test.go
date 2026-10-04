package radixkv

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// A shared-prefix (system/tools) boundary snapshot is admitted BEFORE the deeper
// per-request snapshot below it, so when an unrelated request applies byte pressure
// pure LRU evicts the ancestor first. Snapshot byte pressure must instead reclaim
// the deepest snapshot: the ancestor restores every sibling, and recurrent state
// cannot be rebuilt by truncating a leaf.
// fak-test:runtime fast est=1s
func TestSnapshotByteBudgetEvictsDeepestBeforeSharedAncestor(t *testing.T) {
	logits := make([]float32, 32) // 128 resident bytes per complete snapshot
	tree := NewWithBudgets(1000, 256)
	admit := func(tokens []int) {
		t.Helper()
		b, m := tree.Lookup(tokens)
		leaf, err := tree.InsertSnapshot(b, tokens[m:], model.NewHostPrefixSnapshotForTest(model.NewKVCache(model.Config{})), logits)
		if err != nil {
			t.Fatalf("admit %v: %v", tokens, err)
		}
		tree.Done(leaf)
	}
	has := func(tokens []int) bool {
		t.Helper()
		n, snap, matched, err := tree.LookupSnapshot(tokens)
		tree.Done(n)
		if err != nil {
			t.Fatalf("lookup %v: %v", tokens, err)
		}
		if snap == nil {
			return false
		}
		snap.Close()
		return matched == len(tokens)
	}

	shared := []int{1, 2}
	admit(shared)         // parent: shared boundary (oldest)
	admit([]int{1, 2, 3}) // parent: full prompt
	admit([]int{9})       // unrelated request forces one eviction
	if !has(shared) {
		t.Fatal("byte pressure evicted the shared ancestor snapshot before a deeper leaf")
	}
	if has([]int{1, 2, 3}) {
		t.Fatal("deeper parent leaf survived while the budget holds only two snapshots")
	}
	if !has([]int{9}) {
		t.Fatal("newly admitted snapshot missing")
	}
}
