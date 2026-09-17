package radixkv_test

import (
	"testing"
)

// TestTransientCheckpointHashRetraction is the witness for
// "drop a transient checkpoint's published prefix hash before reuse overwrites it".
//
// A node that carries no direct physical blocks can still PUBLISH a usable block
// chain derived from a descendant (resolveBlockIDsLocked step 2, the post-split
// derivation). When that descendant is a transient request-local checkpoint whose
// physical slots are about to be overwritten by a later write, the ancestor must
// stop advertising the derived chain BEFORE the overwrite: otherwise a subsequent
// BindPrefix/ResolveMetalPageTable keeps matching a published hash that names slots
// which no longer hold the ancestor prefix - a silent wrong-reuse.
func TestTransientCheckpointHashRetraction(t *testing.T) {
	pagedRadix, kvPool := makeTestPool(t, 64, 16)
	allocator := kvPool.Allocator()

	blockA, err := allocator.Allocate()
	if err != nil {
		t.Fatalf("allocate blockA: %v", err)
	}
	blockB, err := allocator.Allocate()
	if err != nil {
		t.Fatalf("allocate blockB: %v", err)
	}

	ancestorTokens := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	boundary, _ := pagedRadix.Tree.Lookup(nil)
	ancestor, err := pagedRadix.InsertPaged(boundary, ancestorTokens, []int{blockA.ID})
	if err != nil {
		t.Fatalf("InsertPaged ancestor: %v", err)
	}

	descTokens := []int{101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116}
	descendant, err := pagedRadix.InsertPaged(ancestor, descTokens, []int{blockB.ID})
	if err != nil {
		t.Fatalf("InsertPaged descendant: %v", err)
	}

	if got := pagedRadix.ResolveMetalPageTable(ancestor); len(got) != 1 {
		t.Fatalf("precondition: ancestor ResolveMetalPageTable = %v, want one published block", got)
	}

	pagedRadix.RetractPublishedHash(descendant)

	if got := pagedRadix.ResolveMetalPageTable(descendant); got != nil {
		t.Fatalf("retracted descendant ResolveMetalPageTable = %v, want nil until re-published", got)
	}

	full := append(append([]int(nil), ancestorTokens...), descTokens...)
	if metal := pagedRadix.ResolvePublishedChain(full); metal != nil {
		t.Fatalf("ResolvePublishedChain(retracted) = %v, want nil", metal)
	}

	newBlock, err := allocator.Allocate()
	if err != nil {
		t.Fatalf("allocate newBlock: %v", err)
	}
	if err := pagedRadix.SetNodeBlocks(descendant, []int{newBlock.ID}); err != nil {
		t.Fatalf("re-publish descendant: %v", err)
	}
	if got := pagedRadix.ResolveMetalPageTable(descendant); len(got) != 1 {
		t.Fatalf("re-published descendant ResolveMetalPageTable = %v, want one block", got)
	}
}
