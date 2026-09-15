package compute

import (
	"errors"
	"fmt"
	"sync/atomic"
	"time"
)

// pendingRestore tracks a frame claimed for a not-yet-finalized batched read.
type pendingRestore struct {
	entry    *KVPageDirectoryEntry
	frameIdx int
}

// BufferColdPrefix ensures prefix blocks are committed to NVMe storage and marks them cold.
//
// The batch is submitted as a whole (every SubmitWrite issued back-to-back) and
// drained with a single PollCompletions(len(batch)), collapsing N serial NVMe
// round-trips into O(1) polls. A submission failure or an offload on any block
// still drains the outstanding batched commands before the per-block fallback,
// so an in-flight entry is never stranded or stolen by a stray serial poll.
func (c *BaMKVPagingCoordinator) BufferColdPrefix(blockIDs []uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Pass 1: submit every eligible block write without waiting on completions.
	pending := make([]*KVPageDirectoryEntry, 0, len(blockIDs))
	for _, blockID := range blockIDs {
		entry, exists := c.directory[blockID]
		if !exists {
			return fmt.Errorf("amddirect: cold prefix block %d not found", blockID)
		}
		if !entry.IsColdPrefix {
			entry.IsColdPrefix = true
			c.stats.ColdPrefixBlocks++
		}

		if !entry.Resident || entry.FrameID < 0 || entry.FrameID >= len(c.frames) {
			continue
		}

		frame := c.frames[entry.FrameID]
		storageBuf := make([]byte, frame.SizeBytes)
		copy(storageBuf, frame.Data)
		c.storageMu.Lock()
		c.storageBacking[entry.NVMeLBA] = storageBuf
		c.storageMu.Unlock()

		if _, err := c.pipeline.SubmitWrite(entry.NVMeLBA, uint16(entry.LBACount), frame.VRAMAddress, frame.SizeBytes); err != nil {
			// Per-block fallback: drain the batched submits already in flight so
			// they are not stranded, then surface the error, preserving the
			// early-return-on-error contract.
			if len(pending) > 0 {
				c.PollCompletions(len(pending))
			}
			return err
		}
		pending = append(pending, entry)
	}

	// Drain the whole batch with a single poll before any offload runs.
	if len(pending) > 0 {
		c.PollCompletions(len(pending))
	}

	// Pass 2: the batch write persisted every frame's bytes, so complete each
	// unpinned frame's offload with bookkeeping only. Using the write-free
	// completion keeps the whole call at one poll: re-issuing a serial
	// offloadFrameLocked submit+poll here would add N polls and could steal a
	// still-in-flight batched entry.
	for _, entry := range pending {
		frame := c.frames[entry.FrameID]
		c.stats.NVMeWriteCount++
		c.stats.BytesWrittenNVMe += frame.SizeBytes

		if frame.PinCount == 0 {
			_ = c.markFrameOffloadedLocked(entry.FrameID)
		}
	}
	return nil
}

// PrefetchLayerAhead brings forward KV blocks for upcoming layers into VRAM before compute reaches them.
//
// Non-resident blocks are read as a batch: every SubmitRead is issued first, then a
// single PollCompletions(missing) drains the whole layer's missing set. This removes
// the per-block submit→blocking-poll serialization where block i+1's IO could only
// start after block i completed. A block whose allocation or submission failed is
// restored serially through the per-block fallback so one bad block cannot drop the
// whole layer.
func (c *BaMKVPagingCoordinator) PrefetchLayerAhead(currentLayer int, prefetchDistance int) error {
	if prefetchDistance <= 0 {
		prefetchDistance = c.cfg.PrefetchDistance
	}
	targetLayer := currentLayer + prefetchDistance
	if targetLayer >= c.cfg.TotalModelLayers {
		return nil
	}

	c.mu.RLock()
	blocks := append([]uint64(nil), c.layerBlocks[targetLayer]...)
	c.mu.RUnlock()

	if len(blocks) == 0 {
		return nil
	}

	// Classify: residents are hits; missing blocks are queued for one batched read.
	missing := make([]uint64, 0, len(blocks))
	c.mu.Lock()
	for _, blockID := range blocks {
		entry, exists := c.directory[blockID]
		if !exists {
			continue
		}
		if entry.Resident {
			c.stats.PrefetchHits++
		} else {
			c.stats.PrefetchMisses++
			missing = append(missing, blockID)
		}
	}

	// Phase 1: allocate a frame and submit every read without polling.
	claimed := make([]pendingRestore, 0, len(missing))
	failed := make([]uint64, 0)
	for _, blockID := range missing {
		entry, exists := c.directory[blockID]
		if !exists || entry.Resident {
			continue
		}
		frameIdx, err := c.beginRestoreLocked(entry)
		if err != nil {
			failed = append(failed, blockID)
			continue
		}
		claimed = append(claimed, pendingRestore{entry: entry, frameIdx: frameIdx})
	}

	// Phase 2: one drain for the whole missing set, then hydrate each frame.
	if len(claimed) > 0 {
		c.PollCompletions(len(claimed))
		for _, p := range claimed {
			c.finishRestoreLocked(p.entry, p.frameIdx)
		}
	}
	c.mu.Unlock()

	// Per-block fallback for blocks whose batched submit failed.
	for _, blockID := range failed {
		if err := c.RestoreBlock(blockID); err != nil {
			return fmt.Errorf("amddirect: prefetch failed restoring block %d: %w", blockID, err)
		}
	}

	c.mu.Lock()
	totalPrefetch := c.stats.PrefetchHits + c.stats.PrefetchMisses
	if totalPrefetch > 0 {
		c.stats.PrefetchHitRate = float64(c.stats.PrefetchHits) / float64(totalPrefetch)
	}
	c.mu.Unlock()

	return nil
}

// beginRestoreLocked claims a VRAM frame for entry and issues its NVMe read
// without waiting for the completion. It mirrors the serial allocation + submit
// body of RestoreBlock so the batched path can issue reads back-to-back. On
// error the claimed frame is returned to the free pool.
func (c *BaMKVPagingCoordinator) beginRestoreLocked(entry *KVPageDirectoryEntry) (int, error) {
	var frameIdx int
	if len(c.freeFrameIndices) > 0 {
		frameIdx = c.freeFrameIndices[0]
		c.freeFrameIndices = c.freeFrameIndices[1:]
	} else {
		chosenIdx, err := c.selectUnpinnedFrameLocked()
		if err != nil {
			return -1, err
		}
		if err := c.offloadFrameLocked(chosenIdx); err != nil {
			return -1, err
		}
		frameIdx = c.freeFrameIndices[0]
		c.freeFrameIndices = c.freeFrameIndices[1:]
	}

	frame := c.frames[frameIdx]
	// Reserve the frame immediately so a later block in this same batch cannot
	// re-select it via selectUnpinnedFrameLocked while its read is in flight.
	// finishRestoreLocked re-marks it InUse once the batch drains.
	frame.InUse = false
	frame.BlockID = 0
	frame.PinCount = 0
	frame.IsDirty = false

	if _, err := c.pipeline.SubmitRead(entry.NVMeLBA, uint16(entry.LBACount), frame.VRAMAddress, entry.SizeBytes); err != nil {
		c.freeFrameIndices = append(c.freeFrameIndices, frameIdx)
		return -1, err
	}
	return frameIdx, nil
}

// finishRestoreLocked hydrates a frame and applies residency bookkeeping after
// its batched NVMe read has been drained. It mirrors the post-poll body of
// RestoreBlock so both paths converge on identical state.
func (c *BaMKVPagingCoordinator) finishRestoreLocked(entry *KVPageDirectoryEntry, frameIdx int) {
	frame := c.frames[frameIdx]

	var data []byte
	if c.dirtyRing != nil {
		if d, err := c.dirtyRing.ReadBlock(entry.NVMeLBA); err == nil {
			data = d
		}
	}
	if data == nil {
		c.storageMu.RLock()
		storageBuf := c.storageBacking[entry.NVMeLBA]
		c.storageMu.RUnlock()
		data = storageBuf
	}
	if len(data) > 0 {
		copy(frame.Data, data)
	}

	frame.BlockID = entry.BlockID
	frame.LayerID = entry.LayerID
	frame.InUse = true
	frame.PinCount = 0
	frame.IsDirty = false
	frame.LastAccessNano = time.Now().UnixNano()

	entry.Resident = true
	entry.FrameID = frameIdx
	entry.VRAMAddress = frame.VRAMAddress
	entry.LastAccessNano = frame.LastAccessNano

	c.stats.ResidentBlocks++
	c.stats.OffloadedBlocks--
	c.stats.NVMeReadCount++
	c.stats.BytesReadNVMe += entry.SizeBytes
}

// SimulateLayerComputeWithPrefetch models compute and prefetch overlap to calculate hiding percentage.
func (c *BaMKVPagingCoordinator) SimulateLayerComputeWithPrefetch(layerID int, computeTimeNs int64) (PrefetchOverlapMetrics, error) {
	if computeTimeNs <= 0 {
		computeTimeNs = 500000
	}

	targetLayer := layerID + c.cfg.PrefetchDistance
	offloadedCount := 0
	c.mu.RLock()
	if targetLayer < c.cfg.TotalModelLayers {
		for _, bID := range c.layerBlocks[targetLayer] {
			if entry, ok := c.directory[bID]; ok && !entry.Resident {
				offloadedCount++
			}
		}
	}
	c.mu.RUnlock()

	var prefetchNs int64
	if offloadedCount > 0 {
		prefetchNs = int64(offloadedCount) * 10000
	} else {
		prefetchNs = 5000
	}

	if err := c.PrefetchLayerAhead(layerID, c.cfg.PrefetchDistance); err != nil {
		return PrefetchOverlapMetrics{}, err
	}

	var hiddenNs int64
	var overlapPct float64

	if prefetchNs <= computeTimeNs {
		hiddenNs = prefetchNs
		overlapPct = 100.0
	} else {
		hiddenNs = computeTimeNs
		overlapPct = (float64(computeTimeNs) / float64(prefetchNs)) * 100.0
	}

	hiding := overlapPct >= 90.0
	allDone := (hiddenNs == prefetchNs)
	_ = allDone

	return PrefetchOverlapMetrics{
		LayerIndex:         layerID,
		ComputeDurationNs:  computeTimeNs,
		PrefetchDurationNs: prefetchNs,
		HiddenDurationNs:   hiddenNs,
		OverlapPercentage:  overlapPct,
		HidingAchieved:     hiding,
	}, nil
}

// VerifyDataIntegrity checks block contents against its initial checksum for bit-rot or corruption.
func (c *BaMKVPagingCoordinator) VerifyDataIntegrity(blockID uint64) (bool, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, exists := c.directory[blockID]
	if !exists {
		return false, ErrBlockNotFound
	}

	var data []byte
	if entry.Resident {
		if entry.FrameID < 0 || entry.FrameID >= len(c.frames) {
			return false, errors.New("amddirect: corrupted frame index in directory")
		}
		data = c.frames[entry.FrameID].Data
	} else {
		if c.dirtyRing != nil {
			if d, err := c.dirtyRing.ReadBlock(entry.NVMeLBA); err == nil {
				data = d
			}
		}
		if data == nil {
			c.storageMu.RLock()
			storageBuf, ok := c.storageBacking[entry.NVMeLBA]
			c.storageMu.RUnlock()
			if !ok {
				return false, errors.New("amddirect: storage backing missing for offloaded block")
			}
			data = storageBuf
		}
	}

	currentChk := computeChecksum(data)
	return currentChk == entry.Checksum, nil
}

// GetPageDirectoryEntry returns a copy of the page directory entry for blockID.
func (c *BaMKVPagingCoordinator) GetPageDirectoryEntry(blockID uint64) (*KVPageDirectoryEntry, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, exists := c.directory[blockID]
	if !exists {
		return nil, ErrBlockNotFound
	}
	cp := *entry
	return &cp, nil
}

// TotalTokens reports the sum of token counts across all registered blocks.
func (c *BaMKVPagingCoordinator) TotalTokens() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	total := 0
	for _, entry := range c.directory {
		total += entry.TokenCount
	}
	return total
}

// PollCompletions resolves finished NVMe transfers.
func (c *BaMKVPagingCoordinator) PollCompletions(maxEntries int) int {
	return c.pipeline.PollCompletions(maxEntries)
}

// PollCallCount reports the total number of pipeline poll calls made by this
// coordinator. Batched block paths issue O(1) polls per batch; this is the
// witness seam for the SPOB collapse on the KV paging path.
func (c *BaMKVPagingCoordinator) PollCallCount() uint64 {
	return c.pipeline.PollCallCount()
}

// Stats returns a point-in-time telemetry snapshot of the KV paging subsystem.
func (c *BaMKVPagingCoordinator) Stats() BaMKVPagingStats {
	c.mu.RLock()
	defer c.mu.RUnlock()

	snap := c.stats
	snap.NVMeWriteCount += atomic.LoadUint64(&c.ringNVMeWrites)
	snap.BytesWrittenNVMe += atomic.LoadUint64(&c.ringBytesWritten)
	snap.QueueExhaustionCount = c.pipeline.ExhaustionCount()
	snap.StagingCopies = 0

	totalPrefetch := snap.PrefetchHits + snap.PrefetchMisses
	if totalPrefetch > 0 {
		snap.PrefetchHitRate = float64(snap.PrefetchHits) / float64(totalPrefetch)
	}
	return snap
}

// StagingCopyCount returns the number of host DRAM bounce buffer copies. Invariant: always 0.
func (c *BaMKVPagingCoordinator) StagingCopyCount() int {
	return 0
}

// DirtyRing returns the associated UMA DRAM write-back dirty ring, or nil if disabled.
func (c *BaMKVPagingCoordinator) DirtyRing() *UMADRAMWriteBackRing {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.dirtyRing
}

// FlushDirtyRing flushes pending coalesced extents from the write-back dirty ring to storage.
func (c *BaMKVPagingCoordinator) FlushDirtyRing() (uint64, int, error) {
	c.mu.RLock()
	ring := c.dirtyRing
	c.mu.RUnlock()

	if ring == nil {
		return 0, 0, nil
	}
	return ring.Flush()
}

// DirtyRingStats returns the telemetry stats of the dirty ring, or nil if disabled.
func (c *BaMKVPagingCoordinator) DirtyRingStats() *UMADRAMRingStats {
	c.mu.RLock()
	ring := c.dirtyRing
	c.mu.RUnlock()

	if ring == nil {
		return nil
	}
	s := ring.Stats()
	return &s
}
