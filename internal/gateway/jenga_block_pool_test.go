package gateway

import (
	"testing"
)

func TestJengaBlockBankWitness(t *testing.T) {
	// First witness requirements (#9920):
	// 1. Model two leaf geometries with different entry sizes (e.g. leaf_alpha: 512B, leaf_beta: 256B).
	// 2. Allocate all capacity to Leaf A until bank exhausts.
	// 3. Reclaim Leaf A entries.
	// 4. Prove reclamation raises usable capacity: huge blocks are reassigned to Leaf B without OOM.
	// 5. Zero cross-leaf aliasing and prefix hash integrity verified.

	hugeBlockBytes := 1024
	numBlocks := 2 // Total bank capacity: 2048 bytes

	bank, err := NewJengaBlockBank(hugeBlockBytes, numBlocks)
	if err != nil {
		t.Fatalf("NewJengaBlockBank failed: %v", err)
	}

	geomA := JengaLeafGeometry{LeafType: "leaf_alpha", EntryBytes: 512} // 2 entries per block -> 4 total
	geomB := JengaLeafGeometry{LeafType: "leaf_beta", EntryBytes: 256}  // 4 entries per block -> 8 total

	// 2. Allocate all capacity to Leaf A (2 blocks * 2 entries = 4 entries)
	var entriesA [][2]int
	var digestsA []string
	for i := 0; i < 4; i++ {
		bID, eID, digest, err := bank.AllocateEntry(geomA, "shared_prefix_0")
		if err != nil {
			t.Fatalf("alloc A %d failed: %v", i, err)
		}
		entriesA = append(entriesA, [2]int{bID, eID})
		digestsA = append(digestsA, digest)
	}

	// 5th allocation must fail (bank exhausted)
	if _, _, _, err := bank.AllocateEntry(geomA, "shared_prefix_0"); err == nil {
		t.Fatal("expected bank exhaustion on 5th allocation of Leaf A")
	}

	// Also cannot allocate Leaf B because no huge block is free or has Leaf B type
	if _, _, _, err := bank.AllocateEntry(geomB, "shared_prefix_0"); err == nil {
		t.Fatal("expected allocation of Leaf B to fail while all blocks held by Leaf A")
	}

	// 3. Reclaim all entries of huge block 0 (entriesA[0] and entriesA[1])
	if err := bank.ReclaimEntry(entriesA[0][0], entriesA[0][1]); err != nil {
		t.Fatalf("reclaim A[0] failed: %v", err)
	}
	if err := bank.ReclaimEntry(entriesA[1][0], entriesA[1][1]); err != nil {
		t.Fatalf("reclaim A[1] failed: %v", err)
	}

	// 4. Prove reclamation raises usable capacity for Leaf B!
	// Huge block 0 is now unassigned and can be claimed by Leaf B (4 entries of 256B)
	var entriesB [][2]int
	var digestsB []string
	for i := 0; i < 4; i++ {
		bID, eID, digest, err := bank.AllocateEntry(geomB, "shared_prefix_0")
		if err != nil {
			t.Fatalf("alloc B %d failed after A reclaim: %v", i, err)
		}
		if bID != 0 {
			t.Fatalf("expected Leaf B to allocate into reclaimed block 0, got block %d", bID)
		}
		entriesB = append(entriesB, [2]int{bID, eID})
		digestsB = append(digestsB, digest)
	}

	// 5. Verify zero cross-leaf aliasing:
	// Block 0 is now Leaf B, Block 1 is still Leaf A (entriesA[2] and entriesA[3])
	if bank.Blocks[0].LeafType != "leaf_beta" {
		t.Fatalf("block 0 leaf type = %s, want leaf_beta", bank.Blocks[0].LeafType)
	}
	if bank.Blocks[1].LeafType != "leaf_alpha" {
		t.Fatalf("block 1 leaf type = %s, want leaf_alpha", bank.Blocks[1].LeafType)
	}

	// Verify prefix hash integrity: Leaf A and Leaf B digests on same prefix are distinct
	seenDigests := make(map[string]bool)
	for _, d := range digestsA {
		seenDigests[d] = true
	}
	for _, d := range digestsB {
		if seenDigests[d] {
			t.Fatalf("prefix digest collision between Leaf A and Leaf B: %s", d)
		}
	}

	if bank.TotalReclaims != 2 {
		t.Fatalf("expected 2 total reclaims, got %d", bank.TotalReclaims)
	}
}
