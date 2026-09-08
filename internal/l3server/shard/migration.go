package shard

import (
	"log"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/alloc"
	"github.com/anthony-chaudhary/fak/internal/l3server/index"
)

// startMigration begins a ZeroLatencyBalance migration asynchronously.
// The expensive allocator construction (mmap+MAP_POPULATE, potentially 30s+) runs
// in a background goroutine. The shard's run loop picks up the result via
// pendingAlloc channel and calls commitMigration to start the actual data migration,
// keeping the shard fully responsive during allocator construction.
//
// weights is nil for auto-tune rebuilds, non-nil for pressure-driven rebuilds.
// freezeAfter controls whether finalizeMigration marks the sizeTracker as frozen
// (used for auto-detect rebuilds so we don't re-detect the same workload).
// Returns true if construction was started, false if deferred/skipped.
func (s *Shard) startMigration(weights map[uint64]float64, freezeAfter bool) bool {
	if s.config.AllocatorMode == "offset" {
		return false // offset allocator handles all sizes natively
	}

	// Acquire migration semaphore if available
	if s.config.MigrateSem != nil {
		select {
		case s.config.MigrateSem <- struct{}{}:
			// acquired
		default:
			// semaphore full â€” defer migration to next opportunity
			log.Printf("[rebalance] shard %d: DEFERRED â€” concurrency limit reached", s.id)
			return false
		}
	}

	s.allocBuilding.Store(true)

	optimalSize := s.sizeTracker.optimalSize
	if optimalSize == 0 {
		optimalSize = s.config.ModelPageBytes
	}

	// Use dedicated mode for auto/dedicated distributions, but only when
	// weights are nil (pressure-driven rebalance supplies custom weights
	// and should use normal multi-class distribution).
	isDedicated := (s.config.SlabDistribution == "dedicated" || s.config.SlabDistribution == "auto") && weights == nil
	newCfg := alloc.SlabConfig{
		MaxMemoryBytes: s.config.MaxMemoryBytes,
		HugePageSizeKB: s.config.HugePageSizeKB,
		ModelPageBytes: optimalSize,
		ClassWeights:   weights,
		Dedicated:      isDedicated,
	}

	// Memory guard: check free hugepages BEFORE starting the expensive
	// background allocation. Without sufficient hugepage headroom, the new
	// allocator's mmap(MAP_HUGETLB) will fail and fall back to regular pages,
	// leaving the reserved hugepages unused while consuming additional RAM â€”
	// this is the primary cause of the ~2Ã— memory balloon.
	if s.config.UseHugePages {
		freeHP := alloc.FreeHugepageBytes()
		if freeHP > 0 && freeHP < s.config.MaxMemoryBytes {
			log.Printf("[rebalance] shard %d: DEFERRED â€” insufficient hugepage headroom (free=%d MB, need=%d MB). "+
				"Without headroom, migration would fall back to regular pages and nearly double physical RAM usage.",
				s.id, freeHP/(1024*1024), s.config.MaxMemoryBytes/(1024*1024))
			s.allocBuilding.Store(false)
			s.releaseMigrateSem()
			return false
		}
	}

	// Build allocator in a background goroutine so the shard's op goroutine
	// continues servicing ops during the potentially 30s+ mmap(MAP_POPULATE).
	shardID := s.id
	go func() {
		allocStart := time.Now()
		newAlloc, err := alloc.NewSlabAllocator(newCfg)
		if err != nil {
			log.Printf("[l3server] shard %d: async allocator construction failed: %v (keeping old allocator)", shardID, err)
			s.allocBuilding.Store(false)
			s.releaseMigrateSem()
			return
		}
		log.Printf("[rebalance] shard %d: allocator created in %s (async, shard stayed responsive)",
			shardID, time.Since(allocStart).Truncate(time.Millisecond))

		// Deliver to shard's run loop â€” non-blocking in case shard is shutting down
		select {
		case s.pendingAlloc <- &pendingAllocResult{newAlloc: newAlloc, weights: weights, freezeAfter: freezeAfter}:
		default:
			log.Printf("[rebalance] shard %d: allocator discarded (shard busy or shutting down)", shardID)
			newAlloc.Close()
			s.allocBuilding.Store(false)
			s.releaseMigrateSem()
		}
	}()
	return true
}

// commitMigration is called by the shard's run loop when a pendingAllocResult
// arrives from the async allocator construction goroutine. It wires up the
// migrateState so the normal migration batch loop takes over.
func (s *Shard) commitMigration(result *pendingAllocResult) {
	s.allocBuilding.Store(false)

	// If a migration or FLUSH started in the meantime, discard
	if s.migrate != nil {
		log.Printf("[rebalance] shard %d: discarding async allocator (migration already active)", s.id)
		result.newAlloc.Close()
		s.releaseMigrateSem()
		return
	}

	oldAlloc := s.allocPtr.Load().a

	batchSize := s.config.MigrateBatchSize
	if batchSize <= 0 {
		batchSize = 512
	}

	now := time.Now()
	entries := s.idx.Count()
	ms := &migrateState{
		oldAlloc:     oldAlloc,
		newAlloc:     result.newAlloc,
		cursor:       0,
		batch:        batchSize,
		migrated:     0,
		weights:      result.weights,
		freezeAfter:  result.freezeAfter,
		startTime:    now,
		totalEntries: entries,
		lastLogTime:  now,
	}

	s.migrate = ms
	s.metrics.SetMigrationActive(1)

	// Trigger MR pre-registration on listeners that support it.
	// Pre-registration runs in background goroutines, overlapping with batch migration.
	for _, l := range s.allocListeners {
		if pr, ok := l.(AllocPreRegisterer); ok {
			ch := pr.PreRegisterAllocator(s.id, result.newAlloc)
			s.migrate.preRegDone = append(s.migrate.preRegDone, ch)
		}
	}

	log.Printf("[rebalance] shard %d: STARTED â€” %d entries, batch=%d, pre_reg_listeners=%d",
		s.id, entries, batchSize, len(s.migrate.preRegDone))
}

// migrateBatch migrates up to batchSize entries from old to new allocator.
// Returns true when migration is complete (all entries processed).
func (s *Shard) migrateBatch() bool {
	if s.migrate == nil {
		return true
	}

	ms := s.migrate
	processed := 0

	nextCursor, done := s.idx.IterFrom(ms.cursor, func(idx uint64, e index.Entry) bool {
		// Skip already-migrated entries (e.g. new SETs during migration)
		if e.Flags&index.FlagMigrated != 0 {
			return true
		}

		// Read key from old allocator (use stored class index if available)
		oldKeyClass := int(e.KeyClassIdx)
		if e.Flags&index.FlagHasClassIdx == 0 {
			oldKeyClass = ms.oldAlloc.FindClass(uint64(e.KeyLen))
		}
		if oldKeyClass < 0 || oldKeyClass >= ms.oldAlloc.NumClasses() {
			return true // skip orphaned entry
		}
		keyData := ms.oldAlloc.Read(alloc.Allocation{ClassIdx: oldKeyClass, Offset: e.KeyOffset, Size: ms.oldAlloc.ClassSize(oldKeyClass)})
		keyCopy := make([]byte, e.KeyLen)
		copy(keyCopy, keyData[:e.KeyLen])

		// Read value from old allocator (use stored class index if available)
		oldValClass := int(e.ValueClassIdx)
		if e.Flags&index.FlagHasClassIdx == 0 {
			oldValClass = ms.oldAlloc.FindClass(uint64(e.ValueLen))
		}
		if oldValClass < 0 || oldValClass >= ms.oldAlloc.NumClasses() {
			return true
		}
		valData := ms.oldAlloc.Read(alloc.Allocation{ClassIdx: oldValClass, Offset: e.ValueOffset, Size: ms.oldAlloc.ClassSize(oldValClass)})
		valCopy := make([]byte, e.ValueLen)
		copy(valCopy, valData[:e.ValueLen])

		// Allocate in new allocator
		newKeyAlloc, kerr := ms.newAlloc.Alloc(uint64(e.KeyLen))
		if kerr != nil {
			log.Printf("[l3server] shard %d: migration batch failed on key alloc: %v (aborting)", s.id, kerr)
			return false
		}
		ms.newAlloc.Write(newKeyAlloc, keyCopy)

		newValAlloc, verr := ms.newAlloc.Alloc(uint64(e.ValueLen))
		if verr != nil {
			ms.newAlloc.Free(newKeyAlloc)
			log.Printf("[l3server] shard %d: migration batch failed on value alloc: %v (aborting)", s.id, verr)
			return false
		}
		ms.newAlloc.Write(newValAlloc, valCopy)

		// Free from old allocator
		ms.oldAlloc.Free(alloc.Allocation{ClassIdx: oldKeyClass, Offset: e.KeyOffset, Size: ms.oldAlloc.ClassSize(oldKeyClass)})
		ms.oldAlloc.Free(alloc.Allocation{ClassIdx: oldValClass, Offset: e.ValueOffset, Size: ms.oldAlloc.ClassSize(oldValClass)})

		// Update index entry with new offsets and set FlagMigrated
		updated := index.Entry{
			KeyHash:       e.KeyHash,
			KeyLen:        e.KeyLen,
			KeyOffset:     newKeyAlloc.Offset,
			ValueOffset:   newValAlloc.Offset,
			ValueLen:      e.ValueLen,
			TTL:           e.TTL,
			RefCount:      e.RefCount,
			Flags:         e.Flags | index.FlagMigrated | index.FlagHasClassIdx,
			ValueClassIdx: uint8(newValAlloc.ClassIdx),
			KeyClassIdx:   uint8(newKeyAlloc.ClassIdx),
		}
		s.idx.Insert(e.KeyHash, updated)
		ms.migrated++
		processed++

		return processed < ms.batch
	})

	ms.cursor = nextCursor

	// Periodic progress logging (~every 2 seconds)
	if now := time.Now(); now.Sub(ms.lastLogTime) >= 2*time.Second {
		pct := float64(0)
		if ms.totalEntries > 0 {
			pct = float64(ms.migrated) / float64(ms.totalEntries) * 100
		}
		log.Printf("[rebalance] shard %d: PROGRESS â€” %d/%d entries (%.1f%%), ops_queued=%d, elapsed=%s",
			s.id, ms.migrated, ms.totalEntries, pct, len(s.ops), now.Sub(ms.startTime).Truncate(time.Millisecond))
		ms.lastLogTime = now
	}

	// If IterFrom stopped early (fn returned false) and we processed a full batch, not done yet
	if !done && processed >= ms.batch {
		return false
	}

	// Check if iteration stopped due to allocation failure (processed < batch but not done)
	if !done && processed < ms.batch {
		// Allocation failure â€” abort migration
		log.Printf("[l3server] shard %d: migration aborted at entry %d (alloc failure)", s.id, ms.migrated)
		s.abortMigration()
		return true // migration "complete" (aborted)
	}

	return done
}
