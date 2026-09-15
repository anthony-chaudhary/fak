// Package compute implements hardware abstraction, tensor computation, memory slab management,
// and zero-copy device interconnect acceleration for the fak agent kernel.
package compute

import (
	"errors"
	"testing"
)

// newResidencyCoord builds a coordinator with a small resident frame count so
// freshly allocated layer blocks spill to NVMe and can be batch-restored.
func newResidencyCoord(t *testing.T, frames int, queueDepth int) *BaMKVPagingCoordinator {
	t.Helper()
	hal := NewAMDGPUDirectHAL(AMDGPUDirectConfig{})
	cfg := BaMKVPagingConfig{
		TotalModelLayers:  8,
		TokensPerBlock:    256,
		BytesPerBlock:     16 * 1024,
		MaxResidentFrames: frames,
		TotalNVMeLBAs:     1024,
		SectorSizeBytes:   4096,
		QueueDepth:        queueDepth,
		PrefetchDistance:  1,
	}
	coord, err := NewBaMKVPagingCoordinator(hal, cfg)
	if err != nil {
		t.Fatalf("NewBaMKVPagingCoordinator failed: %v", err)
	}
	return coord
}

// allocateLayerBlocks allocates n distinct blocks on layer and returns their IDs.
func allocateLayerBlocks(t *testing.T, coord *BaMKVPagingCoordinator, layer int, n int) []uint64 {
	t.Helper()
	ids := make([]uint64, 0, n)
	for i := 0; i < n; i++ {
		data := make([]byte, 16*1024)
		data[0] = byte(i & 0xFF)
		data[len(data)-1] = 0xAA
		bid, err := coord.AllocateBlock(layer, 256, data)
		if err != nil {
			t.Fatalf("AllocateBlock layer %d #%d failed: %v", layer, i, err)
		}
		ids = append(ids, bid)
	}
	return ids
}

// TestKVResidency_BufferColdPrefixOnePollPerBatch asserts BufferColdPrefix issues
// exactly one PollCompletions for a batch of N resident cold-prefix blocks,
// rather than one poll per block.
func TestKVResidency_BufferColdPrefixOnePollPerBatch(t *testing.T) {
	const n = 8
	coord := newResidencyCoord(t, n, 128)
	ids := allocateLayerBlocks(t, coord, 0, n)

	// Pin every block so Phase 2's per-frame offload (a separate NVMe write
	// path) does not fire; the poll count then isolates the batch write drain.
	for _, bid := range ids {
		if err := coord.Pin(bid); err != nil {
			t.Fatalf("Pin(%d) failed: %v", bid, err)
		}
	}

	before := coord.PollCallCount()
	if err := coord.BufferColdPrefix(ids); err != nil {
		t.Fatalf("BufferColdPrefix failed: %v", err)
	}
	polls := coord.PollCallCount() - before
	if polls != 1 {
		t.Fatalf("BufferColdPrefix polls = %d, want 1 (batched O(1))", polls)
	}

	stats := coord.Stats()
	if stats.ColdPrefixBlocks != n {
		t.Fatalf("ColdPrefixBlocks = %d, want %d", stats.ColdPrefixBlocks, n)
	}
	if stats.NVMeWriteCount != n {
		t.Fatalf("NVMeWriteCount = %d, want %d", stats.NVMeWriteCount, n)
	}
	if stats.StagingCopies != 0 {
		t.Fatalf("StagingCopies = %d, want 0", stats.StagingCopies)
	}

	// The batch drain must leave every pinned block resident (pinning blocks the
	// Phase 2 offload), and each block must still read back its bytes.
	for _, bid := range ids {
		entry, err := coord.GetPageDirectoryEntry(bid)
		if err != nil {
			t.Fatalf("GetPageDirectoryEntry(%d) failed: %v", bid, err)
		}
		if !entry.Resident {
			t.Fatalf("pinned cold-prefix block %d offloaded despite pin", bid)
		}
		if _, err := coord.ReadBlock(bid); err != nil {
			t.Fatalf("ReadBlock(%d) after batched cold prefix failed: %v", bid, err)
		}
	}
}

// TestKVResidency_PrefetchLayerAheadOnePollPerBatch forces N blocks of the
// target layer offloaded, then asserts the prefetch drains them with a single
// poll and restores residency.
func TestKVResidency_PrefetchLayerAheadOnePollPerBatch(t *testing.T) {
	const n = 6
	// Enough frames for every block, so the batch restore needs no eviction;
	// all n are then explicitly offloaded, giving n missing blocks to restore.
	coord := newResidencyCoord(t, n, 128)
	ids := allocateLayerBlocks(t, coord, 1, n)

	for _, bid := range ids {
		if err := coord.OffloadBlock(bid); err != nil && !errors.Is(err, ErrBlockPinned) {
			t.Fatalf("OffloadBlock(%d) failed: %v", bid, err)
		}
	}

	missing := 0
	for _, bid := range ids {
		entry, err := coord.GetPageDirectoryEntry(bid)
		if err != nil {
			t.Fatalf("GetPageDirectoryEntry(%d) failed: %v", bid, err)
		}
		if !entry.Resident {
			missing++
		}
	}
	if missing < 4 {
		t.Fatalf("expected >= 4 offloaded blocks, got %d", missing)
	}

	before := coord.PollCallCount()
	if err := coord.PrefetchLayerAhead(0, 1); err != nil {
		t.Fatalf("PrefetchLayerAhead failed: %v", err)
	}
	polls := coord.PollCallCount() - before
	if polls != 1 {
		t.Fatalf("PrefetchLayerAhead polls = %d, want 1 (batched O(1))", polls)
	}

	for _, bid := range ids {
		entry, err := coord.GetPageDirectoryEntry(bid)
		if err != nil {
			t.Fatalf("GetPageDirectoryEntry(%d) failed: %v", bid, err)
		}
		if !entry.Resident {
			t.Fatalf("block %d not resident after prefetch", bid)
		}
		ok, err := coord.VerifyDataIntegrity(bid)
		if err != nil || !ok {
			t.Fatalf("VerifyDataIntegrity(%d) = %v, %v; want true, nil", bid, ok, err)
		}
	}
}

// TestKVResidency_PrefetchFallbackOnQueueExhaustion drives a queue-depth-1
// pipeline: the single batched submit of N reads cannot admit all of them, so
// the batched phase fails for the later blocks and must fall back to per-block
// RestoreBlock. Correctness is preserved: every block ends resident and the
// pipeline is fully drained, with ExhaustionCount witnessing the batch error.
func TestKVResidency_PrefetchFallbackOnQueueExhaustion(t *testing.T) {
	const n = 5
	coord := newResidencyCoord(t, n, 1)
	ids := allocateLayerBlocks(t, coord, 1, n)
	for _, bid := range ids {
		if err := coord.OffloadBlock(bid); err != nil && !errors.Is(err, ErrBlockPinned) {
			t.Fatalf("OffloadBlock(%d) failed: %v", bid, err)
		}
	}

	if err := coord.PrefetchLayerAhead(0, 1); err != nil {
		t.Fatalf("PrefetchLayerAhead fallback returned error: %v", err)
	}

	// Queue depth 1 means the batched submit hit ErrQueueExhausted at least once.
	if got := coord.pipeline.ExhaustionCount(); got == 0 {
		t.Fatalf("ExhaustionCount = 0; batch never exercised the submission-error fallback")
	}
	// Nothing stranded in flight, and every block restored by the fallback.
	if got := coord.pipeline.InFlightCount(); got != 0 {
		t.Fatalf("in-flight after fallback = %d, want 0", got)
	}
	for _, bid := range ids {
		entry, err := coord.GetPageDirectoryEntry(bid)
		if err != nil {
			t.Fatalf("GetPageDirectoryEntry(%d) failed: %v", bid, err)
		}
		if !entry.Resident {
			t.Fatalf("block %d not resident after fallback restore", bid)
		}
	}
	if coord.StagingCopyCount() != 0 {
		t.Fatalf("StagingCopyCount = %d, want 0", coord.StagingCopyCount())
	}
}

// TestKVResidency_BufferColdPrefixFallbackOnQueueExhaustion asserts the write
// batch surfaces a submission error and drains without stranding commands.
func TestKVResidency_BufferColdPrefixFallbackOnQueueExhaustion(t *testing.T) {
	const n = 5
	coord := newResidencyCoord(t, n, 1)
	ids := allocateLayerBlocks(t, coord, 0, n)

	err := coord.BufferColdPrefix(ids)
	if err == nil {
		t.Fatalf("expected submission error under queue depth 1, got nil")
	}
	if got := coord.pipeline.InFlightCount(); got != 0 {
		t.Fatalf("in-flight after failed batch = %d, want 0", got)
	}
}
