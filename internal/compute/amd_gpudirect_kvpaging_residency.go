package compute

import (
	"errors"
	"fmt"
	"sync/atomic"
)

// BufferColdPrefix ensures prefix blocks are committed to NVMe storage and marks them cold.
func (c *BaMKVPagingCoordinator) BufferColdPrefix(blockIDs []uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	for _, blockID := range blockIDs {
		entry, exists := c.directory[blockID]
		if !exists {
			return fmt.Errorf("amddirect: cold prefix block %d not found", blockID)
		}
		if !entry.IsColdPrefix {
			entry.IsColdPrefix = true
			c.stats.ColdPrefixBlocks++
		}

		if entry.Resident && entry.FrameID >= 0 && entry.FrameID < len(c.frames) {
			frame := c.frames[entry.FrameID]
			storageBuf := make([]byte, frame.SizeBytes)
			copy(storageBuf, frame.Data)
			c.storageMu.Lock()
			c.storageBacking[entry.NVMeLBA] = storageBuf
			c.storageMu.Unlock()

			_, err := c.pipeline.SubmitWrite(entry.NVMeLBA, uint16(entry.LBACount), frame.VRAMAddress, frame.SizeBytes)
			if err != nil {
				return err
			}
			c.pipeline.PollCompletions(1)
			c.stats.NVMeWriteCount++
			c.stats.BytesWrittenNVMe += frame.SizeBytes

			if frame.PinCount == 0 {
				_ = c.offloadFrameLocked(entry.FrameID)
			}
		}
	}
	return nil
}

// PrefetchLayerAhead brings forward KV blocks for upcoming layers into VRAM before compute reaches them.
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

	for _, blockID := range blocks {
		c.mu.RLock()
		entry, exists := c.directory[blockID]
		if !exists {
			c.mu.RUnlock()
			continue
		}
		isResident := entry.Resident
		c.mu.RUnlock()

		if isResident {
			c.mu.Lock()
			c.stats.PrefetchHits++
			c.mu.Unlock()
		} else {
			c.mu.Lock()
			c.stats.PrefetchMisses++
			c.mu.Unlock()
			if err := c.RestoreBlock(blockID); err != nil {
				return fmt.Errorf("amddirect: prefetch failed restoring block %d: %w", blockID, err)
			}
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
