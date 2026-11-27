package radixkv_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// TestCheckpointRing is the witness for the bounded per-lane KV checkpoint ring
// (oss-port-llamacpp-lane-ctx-checkpoint-20261106), the host-side port of
// llama.cpp's per-slot context checkpoints (--ctx-checkpoints 32,
// --checkpoint-min-step 8192, save/restore/erase; PR #15293).
//
//	go test -v ./internal/radixkv -run TestCheckpointRing -count=1
//
// It verifies:
//  1. Save at a token boundary snapshots the lane's page-set descriptor (block
//     IDs + refcounts) without copying KV bytes.
//  2. Restore re-pins the checkpoint's page set and truncates the lane back to
//     the checkpoint's token boundary, releasing the lane's longer page set.
//  3. The ring is bounded: when full, the oldest checkpoint is evicted and its
//     retained references are released.
//  4. Restore/erase of an unknown checkpoint fails closed.
func TestCheckpointRing(t *testing.T) {
	t.Run("SaveSnapshotsPageSetDescriptor", func(t *testing.T) {
		pagedRadix, kvPool := makeTestPool(t, 64, 16)
		alloc := kvPool.Allocator()
		ring := radixkv.NewCheckpointRing(alloc, 0, 0) // defaults: cap 32, min-step 8192
		if ring.Capacity() != 32 {
			t.Fatalf("default Capacity = %d, want 32", ring.Capacity())
		}
		if ring.MinStep() != 8192 {
			t.Fatalf("default MinStep = %d, want 8192", ring.MinStep())
		}

		// Save before the lane has advanced must fail closed.
		if _, err := ring.Save("lane-a"); err != radixkv.ErrLaneNotAdvanced {
			t.Fatalf("Save before Advance = %v, want ErrLaneNotAdvanced", err)
		}

		// Advance the lane to a 16-token boundary backed by one physical block.
		boundary, _ := pagedRadix.Tree.Lookup(nil)
		_, blocks, err := pagedRadix.InsertWithAllocatedTokens(boundary, []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
		if err != nil {
			t.Fatalf("InsertWithAllocatedTokens: %v", err)
		}
		if len(blocks) != 1 {
			t.Fatalf("allocated %d blocks, want 1", len(blocks))
		}
		if err := ring.Advance("lane-a", 16, blocks); err != nil {
			t.Fatalf("Advance: %v", err)
		}
		// The ring retains the lane's page set on top of the tree's reference.
		blk, err := alloc.GetBlock(blocks[0])
		if err != nil {
			t.Fatalf("GetBlock: %v", err)
		}
		if ref := blk.RefCount(); ref != 2 {
			t.Fatalf("block refcount after Advance = %d, want 2 (tree + ring)", ref)
		}

		id, err := ring.Save("lane-a")
		if err != nil {
			t.Fatalf("Save: %v", err)
		}
		if ring.Len("lane-a") != 1 {
			t.Fatalf("Len after Save = %d, want 1", ring.Len("lane-a"))
		}
		latestID, latestPos, ok := ring.Latest("lane-a")
		if !ok || latestID != id || latestPos != 16 {
			t.Fatalf("Latest = (%v, %d, %v), want (%v, 16, true)", latestID, latestPos, ok, id)
		}
		// Save retains the snapshot's page set independently of the lane hold.
		if ref := blk.RefCount(); ref != 3 {
			t.Fatalf("block refcount after Save = %d, want 3 (tree + lane + checkpoint)", ref)
		}

		// A second save at the same boundary is closer than the min step.
		if _, err := ring.Save("lane-a"); err != radixkv.ErrCheckpointTooClose {
			t.Fatalf("Save at same boundary = %v, want ErrCheckpointTooClose", err)
		}
	})

	t.Run("RestoreRepinsAndTruncates", func(t *testing.T) {
		pagedRadix, kvPool := makeTestPool(t, 64, 16)
		alloc := kvPool.Allocator()
		// minStep 16 so consecutive 16-token boundaries are savable.
		ring := radixkv.NewCheckpointRing(alloc, 0, 16)

		boundary, _ := pagedRadix.Tree.Lookup(nil)
		_, blocksA, err := pagedRadix.InsertWithAllocatedTokens(boundary, []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
		if err != nil {
			t.Fatalf("InsertWithAllocatedTokens A: %v", err)
		}
		if err := ring.Advance("lane-b", 16, blocksA); err != nil {
			t.Fatalf("Advance A: %v", err)
		}
		idA, err := ring.Save("lane-b")
		if err != nil {
			t.Fatalf("Save A: %v", err)
		}

		// The lane grows past the checkpoint to a 32-token boundary.
		leafA, _ := pagedRadix.Tree.Lookup([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
		_, blocksB, err := pagedRadix.InsertWithAllocatedTokens(leafA, []int{17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32})
		if err != nil {
			t.Fatalf("InsertWithAllocatedTokens B: %v", err)
		}
		if err := ring.Advance("lane-b", 32, append(append([]int(nil), blocksA...), blocksB...)); err != nil {
			t.Fatalf("Advance B: %v", err)
		}
		blkA, err := alloc.GetBlock(blocksA[0])
		if err != nil {
			t.Fatalf("GetBlock A: %v", err)
		}
		blkB, err := alloc.GetBlock(blocksB[0])
		if err != nil {
			t.Fatalf("GetBlock B: %v", err)
		}
		// Before restore: A held by tree + checkpoint + lane; B by tree + lane.
		if ref := blkA.RefCount(); ref != 3 {
			t.Fatalf("blockA refcount pre-restore = %d, want 3", ref)
		}
		if ref := blkB.RefCount(); ref != 2 {
			t.Fatalf("blockB refcount pre-restore = %d, want 2", ref)
		}

		pos, err := ring.Restore("lane-b", idA)
		if err != nil {
			t.Fatalf("Restore: %v", err)
		}
		if pos != 16 {
			t.Fatalf("Restore position = %d, want 16", pos)
		}
		if got, ok := ring.TokenPosition("lane-b"); !ok || got != 16 {
			t.Fatalf("TokenPosition after Restore = (%d, %v), want (16, true)", got, ok)
		}
		// Restore re-pinned A (lane hold back) and truncated the lane, releasing
		// its hold on B: A back to 3, B down to the tree's 1.
		if ref := blkA.RefCount(); ref != 3 {
			t.Fatalf("blockA refcount post-restore = %d, want 3 (tree + checkpoint + lane)", ref)
		}
		if ref := blkB.RefCount(); ref != 1 {
			t.Fatalf("blockB refcount post-restore = %d, want 1 (tree only, lane truncated)", ref)
		}

		// The lane can advance again from the restored boundary.
		leafA2, _ := pagedRadix.Tree.Lookup([]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
		_, blocksC, err := pagedRadix.InsertWithAllocatedTokens(leafA2, []int{33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48})
		if err != nil {
			t.Fatalf("InsertWithAllocatedTokens C: %v", err)
		}
		if err := ring.Advance("lane-b", 32, append(append([]int(nil), blocksA...), blocksC...)); err != nil {
			t.Fatalf("Advance C: %v", err)
		}
		if got, _ := ring.TokenPosition("lane-b"); got != 32 {
			t.Fatalf("TokenPosition after re-advance = %d, want 32", got)
		}
	})

	t.Run("RingEvictsOldestWhenFull", func(t *testing.T) {
		// Drive the ring against the raw allocator (no tree) so the ring is the
		// sole retainer of each block and refcounts are deterministic: a live
		// checkpoint holds exactly one reference, an evicted one holds zero.
		_, kvPool := makeTestPool(t, 64, 16)
		alloc := kvPool.Allocator()
		ring := radixkv.NewCheckpointRing(alloc, 3, 16) // bounded to 3 checkpoints

		var ids []radixkv.CheckpointID
		var allBlocks [][]int
		for i := 0; i < 5; i++ {
			blk, err := alloc.Allocate()
			if err != nil {
				t.Fatalf("Allocate %d: %v", i, err)
			}
			blocks := []int{blk.ID}
			allBlocks = append(allBlocks, blocks)
			// Advance the lane to a fresh 16-token boundary backed by this block.
			if err := ring.Advance("lane-c", 16*(i+1), blocks); err != nil {
				t.Fatalf("Advance %d: %v", i, err)
			}
			id, err := ring.Save("lane-c")
			if err != nil {
				t.Fatalf("Save %d: %v", i, err)
			}
			ids = append(ids, id)
		}

		if got := ring.Len("lane-c"); got != 3 {
			t.Fatalf("Len after 5 saves with capacity 3 = %d, want 3", got)
		}
		// The two oldest checkpoints (ids[0], ids[1]) were evicted: their blocks
		// dropped to the lane's single reference (the lane now holds block 4).
		// The three live checkpoints (ids[2], ids[3], ids[4]) each retain their
		// block on top of the lane's hold.
		for i, blocks := range allBlocks {
			blk, err := alloc.GetBlock(blocks[0])
			if err != nil {
				t.Fatalf("GetBlock %d: %v", i, err)
			}
			want := int32(1) // base allocator ref only (evicted, lane no longer holds)
			if i == 4 {
				want = 3 // base + lane hold (current page set) + live checkpoint ids[4]
			}
			if i == 2 || i == 3 {
				want = 2 // base + live checkpoint ids[i] (lane released on advance)
			}
			if ref := blk.RefCount(); ref != want {
				t.Fatalf("block %d refcount = %d, want %d (live=%v)", i, ref, want, i >= 2)
			}
		}
		// Restoring an evicted checkpoint fails closed.
		if _, err := ring.Restore("lane-c", ids[0]); err != radixkv.ErrCheckpointUnknown {
			t.Fatalf("Restore evicted = %v, want ErrCheckpointUnknown", err)
		}
		if _, err := ring.Restore("lane-c", ids[1]); err != radixkv.ErrCheckpointUnknown {
			t.Fatalf("Restore evicted = %v, want ErrCheckpointUnknown", err)
		}
		// The newest checkpoint is still restorable and truncates the lane to it.
		if pos, err := ring.Restore("lane-c", ids[4]); err != nil || pos != 80 {
			t.Fatalf("Restore newest = (%d, %v), want (80, nil)", pos, err)
		}
	})

	t.Run("UnknownCheckpointFailsClosed", func(t *testing.T) {
		pagedRadix, kvPool := makeTestPool(t, 64, 16)
		alloc := kvPool.Allocator()
		ring := radixkv.NewCheckpointRing(alloc, 0, 16)

		boundary, _ := pagedRadix.Tree.Lookup(nil)
		_, blocks, err := pagedRadix.InsertWithAllocatedTokens(boundary, []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
		if err != nil {
			t.Fatalf("InsertWithAllocatedTokens: %v", err)
		}
		if err := ring.Advance("lane-d", 16, blocks); err != nil {
			t.Fatalf("Advance: %v", err)
		}
		id, err := ring.Save("lane-d")
		if err != nil {
			t.Fatalf("Save: %v", err)
		}

		// Restore of a never-minted id fails closed.
		if _, err := ring.Restore("lane-d", id+1); err != radixkv.ErrCheckpointUnknown {
			t.Fatalf("Restore unknown = %v, want ErrCheckpointUnknown", err)
		}
		// Restore of a valid id in a different lane fails closed.
		if _, err := ring.Restore("other-lane", id); err != radixkv.ErrCheckpointUnknown {
			t.Fatalf("Restore cross-lane = %v, want ErrCheckpointUnknown", err)
		}
		// Erase of an unknown id fails closed.
		if err := ring.Erase("lane-d", id+1); err != radixkv.ErrCheckpointUnknown {
			t.Fatalf("Erase unknown = %v, want ErrCheckpointUnknown", err)
		}
		// Erase of the live checkpoint releases its retained page set.
		blk, err := alloc.GetBlock(blocks[0])
		if err != nil {
			t.Fatalf("GetBlock: %v", err)
		}
		if ref := blk.RefCount(); ref != 3 {
			t.Fatalf("precondition refcount = %d, want 3", ref)
		}
		if err := ring.Erase("lane-d", id); err != nil {
			t.Fatalf("Erase: %v", err)
		}
		if ring.Len("lane-d") != 0 {
			t.Fatalf("Len after Erase = %d, want 0", ring.Len("lane-d"))
		}
		if ref := blk.RefCount(); ref != 2 {
			t.Fatalf("refcount after Erase = %d, want 2 (tree + lane)", ref)
		}

		// EraseLane releases the lane's remaining hold.
		ring.EraseLane("lane-d")
		if ref := blk.RefCount(); ref != 1 {
			t.Fatalf("refcount after EraseLane = %d, want 1 (tree only)", ref)
		}
		if got, ok := ring.TokenPosition("lane-d"); ok {
			t.Fatalf("TokenPosition after EraseLane = (%d, true), want not-ok", got)
		}
	})
}

// TestCheckpointRingLaneRegression pins that a lane cannot Advance backwards:
// moving a lane to an earlier boundary is a Restore, not an Advance.
func TestCheckpointRingLaneRegression(t *testing.T) {
	_, kvPool := makeTestPool(t, 64, 16)
	alloc := kvPool.Allocator()
	ring := radixkv.NewCheckpointRing(alloc, 0, 16)

	blk, err := alloc.Allocate()
	if err != nil {
		t.Fatalf("Allocate: %v", err)
	}
	if err := ring.Advance("lane-e", 32, []int{blk.ID}); err != nil {
		t.Fatalf("Advance 32: %v", err)
	}
	if err := ring.Advance("lane-e", 16, []int{blk.ID}); err != radixkv.ErrLaneRegression {
		t.Fatalf("Advance backwards = %v, want ErrLaneRegression", err)
	}
}

// TestCheckpointRingBranchAndRestoreLatest verifies O(checkpoint) branching into a
// new lane without mutating the source lane, and RestoreLatest for undo/retry.
func TestCheckpointRingBranchAndRestoreLatest(t *testing.T) {
	_, kvPool := makeTestPool(t, 64, 16)
	alloc := kvPool.Allocator()
	ring := radixkv.NewCheckpointRing(alloc, 4, 16)

	blkA, err := alloc.Allocate()
	if err != nil {
		t.Fatalf("Allocate A: %v", err)
	}
	blkB, err := alloc.Allocate()
	if err != nil {
		t.Fatalf("Allocate B: %v", err)
	}

	if err := ring.Advance("src", 16, []int{blkA.ID}); err != nil {
		t.Fatalf("Advance src 16: %v", err)
	}
	idA, err := ring.Save("src")
	if err != nil {
		t.Fatalf("Save src 16: %v", err)
	}
	if err := ring.Advance("src", 32, []int{blkA.ID, blkB.ID}); err != nil {
		t.Fatalf("Advance src 32: %v", err)
	}

	// Same-lane or unknown-id branch must fail closed.
	if _, err := ring.Branch("src", "src", idA); err != radixkv.ErrCheckpointBranchSameLane {
		t.Fatalf("Branch same lane = %v, want ErrCheckpointBranchSameLane", err)
	}
	if _, err := ring.Branch("src", "dst", idA+99); err != radixkv.ErrCheckpointUnknown {
		t.Fatalf("Branch unknown id = %v, want ErrCheckpointUnknown", err)
	}

	// Branch dst from src@16: src remains at 32, dst starts at 16 sharing blkA.
	pos, err := ring.Branch("src", "dst", idA)
	if err != nil || pos != 16 {
		t.Fatalf("Branch = (%d, %v), want (16, nil)", pos, err)
	}
	if srcPos, ok := ring.TokenPosition("src"); !ok || srcPos != 32 {
		t.Fatalf("src TokenPosition = (%d, %v), want (32, true)", srcPos, ok)
	}
	if dstPos, ok := ring.TokenPosition("dst"); !ok || dstPos != 16 {
		t.Fatalf("dst TokenPosition = (%d, %v), want (16, true)", dstPos, ok)
	}
	// blkA refcount: 1 (alloc) + 1 (src lane) + 1 (idA checkpoint) + 1 (dst lane) = 4.
	if ref := blkA.RefCount(); ref != 4 {
		t.Fatalf("blkA RefCount after Branch = %d, want 4", ref)
	}

	// RestoreLatest on src rolls back src from 32 to 16 (blkB released back to 1).
	restoredID, restoredPos, err := ring.RestoreLatest("src")
	if err != nil || restoredID != idA || restoredPos != 16 {
		t.Fatalf("RestoreLatest = (%v, %d, %v), want (%v, 16, nil)", restoredID, restoredPos, err, idA)
	}
	if ref := blkB.RefCount(); ref != 1 {
		t.Fatalf("blkB RefCount after RestoreLatest = %d, want 1", ref)
	}
}
