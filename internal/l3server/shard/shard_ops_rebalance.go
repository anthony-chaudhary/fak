package shard

import (
	"log"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/alloc"
	"github.com/anthony-chaudhary/fak/internal/l3server/index"
)

// finalizeMigration completes the migration: swaps allocator, notifies listeners, cleans up.
func (s *Shard) finalizeMigration() {
	if s.migrate == nil {
		return
	}
	ms := s.migrate

	// Wait for MR pre-registration to complete while keeping the shard responsive.
	// This drains ops from the channel so clients don't time out during the wait.
	preRegStart := time.Now()
	opsDrained := 0
	for i, ch := range ms.preRegDone {
		for {
			select {
			case <-ch:
				goto next
			case op := <-s.ops:
				s.handleOp(op)
				opsDrained++
			case <-time.After(60 * time.Second):
				log.Printf("[rebalance] shard %d: WARNING â€” MR pre-registration %d/%d timed out after 60s",
					s.id, i+1, len(ms.preRegDone))
				goto next
			}
		}
	next:
	}
	preRegWait := time.Since(preRegStart)
	if preRegWait > 100*time.Millisecond {
		log.Printf("[rebalance] shard %d: pre-registration wait=%s (drained %d ops)",
			s.id, preRegWait.Truncate(time.Millisecond), opsDrained)
	}

	// Swap allocator pointer
	s.allocPtr.Store(&allocBox{a: ms.newAlloc})
	s.notifyAllocListeners(ms.oldAlloc, ms.newAlloc)

	// Clear FlagMigrated from all entries
	s.idx.ClearFlagAll(index.FlagMigrated)

	// Update tracking state: only freeze if this was an auto-detect migration,
	// not a vacuum rebalance (which resets the tracker to allow re-detection).
	if ms.freezeAfter {
		s.sizeTracker.frozen = true
		s.sizeTracker.updateCachedSnapshot()
	}
	s.pressureTracker.reset(ms.newAlloc.NumClasses())

	if s.config.UseHugePages {
		_, _, regular := ms.newAlloc.HugepageSummary()
		if regular > 0 {
			log.Printf("[l3server] shard %d: WARNING â€” %d region(s) fell back to regular 4KB pages (hugepage pressure?)",
				s.id, regular)
		}
	}

	elapsed := time.Since(ms.startTime)
	s.metrics.SetMigrationActive(0)
	s.metrics.SetMigrationDurationMs(elapsed.Milliseconds())
	s.metrics.SetMigrationEntries(int64(ms.migrated))
	s.metrics.SetMigrationPreRegWaitMs(preRegWait.Milliseconds())
	s.metrics.IncrMigrationsTotal()

	log.Printf("[rebalance] shard %d: COMPLETED â€” migrated %d entries in %s (prereg_wait=%s, ops_drained=%d)",
		s.id, ms.migrated, elapsed.Truncate(time.Millisecond), preRegWait.Truncate(time.Millisecond), opsDrained)

	s.migrate = nil
	s.releaseMigrateSem()
}

// abortMigration cancels an in-progress migration, closing the new allocator.
func (s *Shard) abortMigration() {
	if s.migrate == nil {
		return
	}
	ms := s.migrate

	// Wait for pre-registration goroutines to finish (with 5s timeout) before cleanup.
	for _, ch := range ms.preRegDone {
		select {
		case <-ch:
		case <-time.After(5 * time.Second):
		}
	}

	// Entries already migrated (FlagMigrated set) have their data in newAlloc.
	// Fast abort: evict non-migrated entries from the index, then swap to newAlloc.
	// This avoids a synchronous full-Iter copy that can block for minutes.
	if ms.migrated > 0 {
		evicted := 0
		s.idx.Iter(func(_ uint64, e index.Entry) bool {
			if e.Flags&index.FlagMigrated != 0 {
				return true // already in newAlloc â€” keep
			}
			s.freeEntryKey(e)   // frees from oldAlloc via allocForEntry
			s.freeEntryValue(e) // frees from oldAlloc via allocForEntry
			s.idx.Delete(e.KeyHash, e.KeyLen)
			s.eviction.Remove(e.KeyHash)
			s.metrics.IncrEvictions()
			s.metrics.IncrEvictionsRebalance()
			evicted++
			return true
		})
		s.allocPtr.Store(&allocBox{a: ms.newAlloc})
		// Pre-registered MRs will be consumed by notifyAllocListeners â†’ OnAllocatorChanged
		s.notifyAllocListeners(ms.oldAlloc, ms.newAlloc)
		s.idx.ClearFlagAll(index.FlagMigrated)
		s.pressureTracker.reset(ms.newAlloc.NumClasses())
		log.Printf("[rebalance] shard %d: ABORTED â€” kept %d migrated, evicted %d non-migrated",
			s.id, ms.migrated, evicted)
	} else {
		// No entries migrated yet â€” discard pre-registered MRs and close newAlloc
		for _, l := range s.allocListeners {
			if pr, ok := l.(AllocPreRegisterer); ok {
				pr.DiscardPreRegistered(s.id)
			}
		}
		ms.newAlloc.Close()
		log.Printf("[rebalance] shard %d: ABORTED â€” no entries migrated", s.id)
	}
	s.metrics.SetMigrationActive(0)
	s.migrate = nil
	s.releaseMigrateSem()
}

// releaseMigrateSem releases the migration semaphore if one was acquired.
func (s *Shard) releaseMigrateSem() {
	if s.config.MigrateSem != nil {
		select {
		case <-s.config.MigrateSem:
		default:
		}
	}
}

// handleRebalance performs a ZeroLatencyBalance rebalance.
// Unlike FLUSH, this preserves all cached data via startMigration() (async batched migration).
// When ClassWeights is set, uses pressure-derived weights.
// Requires auto_tune_slabs=true and (detection completed OR ClassWeights provided).
func (s *Shard) handleRebalance(op ShardOp) OpResult {
	if !s.config.AutoTuneSlabs {
		return OpResult{Err: errorString("rebalance requires auto_tune_slabs=true")}
	}

	// Skip if migration already in progress â€” vacuum will re-evaluate next tick
	if s.migrate != nil {
		log.Printf("[rebalance] shard %d: SKIPPED â€” migration already in progress", s.id)
		return OpResult{OK: true}
	}

	// Skip if async allocator construction is in progress
	if s.allocBuilding.Load() {
		log.Printf("[rebalance] shard %d: SKIPPED â€” allocator construction in progress", s.id)
		return OpResult{OK: true}
	}

	// Pressure-driven rebalance: ClassWeights provided by vacuum coordinator.
	// Don't reset sizeTracker â€” pressure changed weights, not dominant size.
	// Reset only pressure counters so re-evaluation starts fresh.
	if op.ClassWeights != nil {
		s.startMigration(op.ClassWeights, false)
		s.pressureTracker.reset(s.allocPtr.Load().a.NumClasses())
		return OpResult{OK: true}
	}

	// Legacy rebalance: requires size detection
	if !s.sizeTracker.detected && !s.sizeTracker.frozen {
		return OpResult{Err: errorString("rebalance requires size detection to have completed")}
	}

	// If frozen (already rebuilt for detected workload), skip redundant migration
	if s.sizeTracker.frozen {
		log.Printf("[rebalance] shard %d: SKIPPED â€” allocator already tuned (frozen)", s.id)
		return OpResult{OK: true}
	}

	s.startMigration(nil, false)

	// Reset size tracker so re-detection can happen after workload changes
	s.sizeTracker.reset()

	return OpResult{OK: true}
}

// handlePageSizeHint applies a client-reported model page size.
// If the hinted size differs from the currently tracked optimal size,
// it triggers a slab rebuild to concentrate memory on the correct class.
//
// Fast path: when the shard is empty (no entries yet â€” typical at startup when
// the connector sends an eager hint from register_mem_pool_host), the old
// allocator is closed and replaced synchronously.  This avoids the full
// ZeroLatencyBalance migration that would otherwise race with the AGGRESSIVE
// warmup phase writes.
func (s *Shard) handlePageSizeHint(op ShardOp) OpResult {
	hintBytes := op.HintBytes
	if hintBytes == 0 {
		return OpResult{OK: true}
	}
	if s.config.AllocatorMode == "offset" {
		return OpResult{OK: true} // offset allocator handles all sizes natively
	}

	// Skip if already detected with the same size
	if s.sizeTracker.detected && s.sizeTracker.optimalSize == hintBytes {
		return OpResult{OK: true}
	}

	s.sizeTracker.optimalSize = hintBytes
	s.sizeTracker.detected = true

	// Fast path: empty shard â€” build new allocator synchronously and swap.
	// No data to migrate, so we avoid the async build + migration machinery
	// entirely.  This lets the first writes land in optimised slab classes
	// even during the AGGRESSIVE warmup phase.
	if s.idx.Count() == 0 && s.migrate == nil && !s.allocBuilding.Load() {
		// Auto-promote to dedicated mode when slab_distribution is "auto" and a
		// page-size hint arrives. The hint signals a single-value-size workload
		// (SGLang KV cache) â€” allocating ~28 classes would waste ~39% of memory.
		isDedicated := s.config.SlabDistribution == "dedicated" || s.config.SlabDistribution == "auto"
		newCfg := alloc.SlabConfig{
			MaxMemoryBytes: s.config.MaxMemoryBytes,
			HugePageSizeKB: s.config.HugePageSizeKB,
			ModelPageBytes: hintBytes,
			Dedicated:      isDedicated,
		}
		newAlloc, err := alloc.NewSlabAllocator(newCfg)
		if err != nil {
			log.Printf("[l3server] shard %d: page-size hint fast-swap failed: %v (keeping old allocator)", s.id, err)
		} else {
			oldAlloc := s.allocPtr.Load().a
			s.allocPtr.Store(&allocBox{a: newAlloc})
			s.pressureTracker.reset(newAlloc.NumClasses())
			s.sizeTracker.setSlotUtilization(newAlloc.SlotUtilization(hintBytes))
			s.sizeTracker.frozen = true
			s.sizeTracker.updateCachedSnapshot()
			// Notify RDMA listeners so MR registrations target the new allocator
			for _, l := range s.allocListeners {
				l.OnAllocatorChanged(AllocatorChange{
					ShardID:      s.id,
					OldAllocator: oldAlloc,
					NewAllocator: newAlloc,
				})
			}
			oldAlloc.Close()
			mode := "model-weighted"
			if isDedicated {
				mode = "dedicated (2 classes, 95/5 split)"
			}
			log.Printf("[l3server] shard %d: page-size hint %d bytes â€” fast-swapped allocator (%s, empty shard)", s.id, hintBytes, mode)
			return OpResult{OK: true}
		}
	}

	// Slow path: shard has data â€” full ZeroLatencyBalance migration
	if s.config.VerboseShardLogging {
		log.Printf("[l3server] shard %d: page-size hint %d bytes â€” triggering slab rebuild", s.id, hintBytes)
	}

	// Compute slot utilization for the hinted size
	sa := s.allocPtr.Load().a
	s.sizeTracker.setSlotUtilization(sa.SlotUtilization(hintBytes))

	// Abort any in-progress migration before starting a new one
	if s.migrate != nil {
		s.abortMigration()
	}
	s.startMigration(nil, true)
	return OpResult{OK: true}
}

// handleForceAutoTune performs on-demand auto-tune: force detection + rebuild.
func (s *Shard) handleForceAutoTune(op ShardOp) OpResult {
	if !s.config.AutoTuneSlabs {
		return OpResult{Err: errorString("auto_tune_slabs is disabled")}
	}
	if s.config.AllocatorMode == "offset" {
		return OpResult{OK: true, Info: map[string]interface{}{"skip": "offset allocator"}}
	}

	// Force early detection if warmup hasn't completed
	if !s.sizeTracker.detected && !s.sizeTracker.frozen {
		if s.sizeTracker.totalSets == 0 {
			return OpResult{Err: errorString("no SETs recorded â€” nothing to detect")}
		}
		if op.ForceDetect {
			s.sizeTracker.detect()
			if s.sizeTracker.detected {
				sa := s.allocPtr.Load().a
				s.sizeTracker.setSlotUtilization(sa.SlotUtilization(s.sizeTracker.optimalSize))
			}
		} else {
			return OpResult{Err: errorString("detection not complete; use force=true to detect early")}
		}
	}

	if !s.sizeTracker.detected && !s.sizeTracker.frozen {
		return OpResult{Err: errorString("detection failed")}
	}

	// Abort any in-progress migration before starting a new one
	if s.migrate != nil {
		s.abortMigration()
	}
	s.startMigration(nil, true)
	snap := s.sizeTracker.snapshot()
	return OpResult{OK: true, Info: map[string]interface{}{"detection": snap}}
}
