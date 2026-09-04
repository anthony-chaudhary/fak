package shard

import (
	"fmt"
	"log"
	"regexp"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/alloc"
	"github.com/anthony-chaudhary/fak/internal/l3server/eviction"
	"github.com/anthony-chaudhary/fak/internal/l3server/index"
	"github.com/anthony-chaudhary/fak/internal/l3server/lease"
	"github.com/anthony-chaudhary/fak/internal/l3server/snapshot"
)

func (s *Shard) handleOp(op ShardOp) {
	var result OpResult

	switch op.Type {
	case OpGet:
		result = s.handleGet(op)
	case OpSet:
		result = s.handleSet(op)
	case OpDelete:
		result = s.handleDelete(op)
	case OpMGet:
		result = s.handleMGet(op)
	case OpMSet:
		result = s.handleMSet(op)
	case OpTest:
		result = s.handleTest(op)
	case OpLease:
		result = s.handleLease(op)
	case OpPin:
		result = s.handlePin(op)
	case OpUnpin:
		result = s.handleUnpin(op)
	case OpInfo:
		result = s.handleInfo(op)
	case OpMDel:
		result = s.handleMDel(op)
	case OpKeys:
		result = s.handleKeys(op)
	case OpFlush:
		result = s.handleFlush(op)
	case OpRebalance:
		result = s.handleRebalance(op)
	case OpPageSizeHint:
		result = s.handlePageSizeHint(op)
	case OpForceAutoTune:
		result = s.handleForceAutoTune(op)
	case OpSnapshot:
		result = s.handleSnapshot(op)
	case OpRestore:
		result = s.handleRestore(op)
	case OpMGetWithAlloc:
		result = s.handleMGetWithAlloc(op)
	case OpBatchLease:
		result = s.handleBatchLease(op)
	default:
		result = OpResult{Err: errUnknownOp}
	}

	if op.Result != nil {
		op.Result <- result
	}
}

var errUnknownOp = errorString("unknown operation")

const maxEvictRetries = 5
const maxEvictRetriesUnderPressure = 10

// EvictionReason tags why an eviction was triggered during allocation.
type EvictionReason uint8

const (
	EvictionKeyAlloc   EvictionReason = iota // key buffer allocation pressure
	EvictionValueAlloc                       // value buffer allocation pressure
)

// allocWithEvictionIn tries to allocate from a specific allocator, force-evicting on failure.
// Promotion path: when best-fit class is exhausted, tries larger classes before evicting.
func (s *Shard) allocWithEvictionIn(sa alloc.Allocator, size uint64, reason EvictionReason) (alloc.Allocation, error) {
	start := time.Now()
	defer func() { s.latency.allocDur.record(time.Since(start)) }()
	// Record alloc attempt for pressure tracking
	classIdx := sa.FindClass(size)
	if classIdx >= 0 {
		s.pressureTracker.recordAllocOp(classIdx)
	}

	// Step 1: Try best-fit
	a, err := sa.Alloc(size)
	if err == nil {
		s.allocSucceeded()
		return a, nil
	}

	// Step 2: Promote to larger class (zero-eviction path)
	maxCI := s.promotionMaxClass(sa, classIdx)
	a, promoted, err := sa.AllocWithPromotion(size, maxCI)
	if err == nil {
		if promoted {
			s.metrics.IncrPromotions()
			s.pressureTracker.recordPromotion(classIdx, a.ClassIdx)
		}
		s.allocSucceeded()
		return a, nil
	}

	// Under system memory pressure, try harder before failing:
	// - Larger TTL sweep batch (500 vs 100) to reclaim more expired entries
	// - More eviction retries (10 vs 5) to free fragmented classes
	underPressure := s.systemPressureLevel != nil && s.systemPressureLevel.Load() >= 2
	sweepBatch := 100
	evictRetries := maxEvictRetries
	if underPressure {
		sweepBatch = 500
		evictRetries = maxEvictRetriesUnderPressure
	}

	// Step 2.5: OOM-triggered mini TTL sweep â€” reclaim expired entries before
	// evicting live data (inspired by SGLang's pre-emptive cache eviction).
	swept := s.sweepExpiredBatch(sweepBatch)
	if swept > 0 {
		// Retry after sweep freed some slots
		a, err = sa.Alloc(size)
		if err == nil {
			s.allocSucceeded()
			return a, nil
		}
		a, promoted, err = sa.AllocWithPromotion(size, maxCI)
		if err == nil {
			if promoted {
				s.metrics.IncrPromotions()
				s.pressureTracker.recordPromotion(classIdx, a.ClassIdx)
			}
			s.allocSucceeded()
			return a, nil
		}
	}

	// Step 3: Eviction loop (only when no free space exists anywhere)
	for i := 0; i < evictRetries; i++ {
		if !s.eviction.EvictOne() {
			break // nothing left to evict
		}
		// Tag the successful eviction with its pressure source
		switch reason {
		case EvictionKeyAlloc:
			s.metrics.IncrEvictionsKeyPressure()
		case EvictionValueAlloc:
			s.metrics.IncrEvictionsValuePressure()
		}
		// After eviction freed a slot, try best-fit then promotion
		a, err = sa.Alloc(size)
		if err == nil {
			s.allocSucceeded()
			return a, nil
		}
		a, promoted, err = sa.AllocWithPromotion(size, maxCI)
		if err == nil {
			if promoted {
				s.metrics.IncrPromotions()
				s.pressureTracker.recordPromotion(classIdx, a.ClassIdx)
			}
			s.allocSucceeded()
			return a, nil
		}
	}
	// Record alloc failure for pressure tracking
	if classIdx >= 0 {
		s.pressureTracker.recordAllocFailure(classIdx)
	}
	s.metrics.IncrEvictionsFailed()
	s.allocFailed()
	log.Printf("[shard %d] WARNING: allocation failed after %d eviction retries (requested %d bytes, class %d): %v â€” SET will be rejected",
		s.id, evictRetries, size, classIdx, err)
	return alloc.Allocation{}, err
}

// allocSucceeded resets the consecutive alloc failure counter and clears OOM state.
func (s *Shard) allocSucceeded() {
	if s.consecAllocFails > 0 || s.oomActive {
		s.consecAllocFails = 0
		if s.oomActive {
			s.oomActive = false
			s.metrics.SetMemoryPressure(0)
			log.Printf("[shard %d] memory pressure cleared â€” accepting SETs again", s.id)
		}
	}
}

// allocFailed increments the consecutive alloc failure counter.
// When the threshold is reached, enters OOM state for fast SET rejection.
func (s *Shard) allocFailed() {
	if s.oomThreshold <= 0 {
		return // OOM admission control disabled
	}
	s.consecAllocFails++
	if s.consecAllocFails >= s.oomThreshold && !s.oomActive {
		if !s.allClassesSaturated(0.90) {
			log.Printf("[shard %d] %d consecutive allocation failures but not all slab classes are saturated (>90%%) â€” likely fragmentation, not entering OOM rejection mode",
				s.id, s.consecAllocFails)
			return
		}
		s.oomActive = true
		s.metrics.SetMemoryPressure(1)
		log.Printf("[shard %d] CRITICAL: %d consecutive allocation failures â€” entering OOM rejection mode (SETs rejected without eviction attempts)",
			s.id, s.consecAllocFails)
	}
}

// checkOOM returns an OOM error with diagnostics if the shard is in memory pressure state.
// Used as a pre-check in handleSet/handleMSet to skip the expensive eviction loop entirely.
//
// Checks three sources (in priority order):
//  1. Per-shard OOM: consecutive alloc failures exceeded oom_reject_after_fails threshold.
//  2. Tiered system pressure: syshealth monitor's MemPressureLevel (0-4).
//     At High (level 2): probabilistically accept ~50% of SETs.
//     At Critical (level 3): probabilistically accept ~10%.
//     At Emergency (level 4): reject all.
//  3. Legacy boolean: systemOOMFlag (backwards compat, superseded by tiered).
func (s *Shard) checkOOM() error {
	// Per-shard OOM (existing allocation-failure-based detection)
	if s.oomActive {
		s.metrics.IncrOOMRejections()
		a := s.allocPtr.Load().a
		allocated := a.AllocatedBytes()
		var totalBytes uint64
		for _, r := range a.Regions() {
			totalBytes += r.Region.Size()
		}
		return fmt.Errorf("%w: shard %d at %.0f%% capacity (%d/%d bytes) â€” try DELETE/FLUSH to free space or increase max_memory_gb",
			ErrOOM, s.id, float64(allocated)/float64(totalBytes)*100, allocated, totalBytes)
	}

	// Tiered system memory pressure â€” probabilistic backpressure.
	// Each level accepts a random fraction of SETs using a per-shard PRNG
	// (fair to burst patterns, no counter drift, no goroutine blocking).
	if s.systemPressureLevel != nil {
		level := s.systemPressureLevel.Load()
		if level >= 4 {
			// Emergency: reject all SETs
			s.metrics.IncrOOMRejections()
			return fmt.Errorf("%w: system memory emergency â€” MemAvailable critically low; all SETs rejected until memory recovers",
				ErrOOM)
		}
		if level >= 2 {
			var acceptPct uint64
			if level == 2 {
				acceptPct = 50 // High: accept ~50% of SETs
			} else {
				acceptPct = 10 // Critical: accept ~10% of SETs
			}
			if s.rng.next()%100 >= acceptPct {
				s.metrics.IncrOOMRejections()
				return fmt.Errorf("%w: system memory pressure (level %d) â€” backpressure active, retry later",
					ErrOOM, level)
			}
			// Accepted â€” fall through to normal allocation path
		}
		// Level 0-1: no restriction (level 1 = elevated, warn only)
		return nil
	}

	// Legacy boolean fallback (when tiered pressure is not wired)
	if s.systemOOMFlag != nil && s.systemOOMFlag.Load() {
		s.metrics.IncrOOMRejections()
		return fmt.Errorf("%w: system memory pressure â€” MemAvailable below mem_pressure_threshold_pct; try DELETE/FLUSH or free system memory",
			ErrOOM)
	}
	return nil
}

// sweepExpiredBatch sweeps up to limit expired TTL entries. Returns count swept.
// Used as a pre-emptive reclaim before evicting live data.
func (s *Shard) sweepExpiredBatch(limit int) int {
	now := time.Now().UnixMilli()
	swept := 0
	s.idx.Iter(func(_ uint64, e index.Entry) bool {
		if swept >= limit {
			return false
		}
		if e.TTL > 0 && now > e.TTL {
			s.freeEntryKey(e)
			s.freeEntryValue(e)
			s.idx.Delete(e.KeyHash, e.KeyLen)
			s.eviction.Remove(e.KeyHash)
			s.metrics.IncrTTLExpirations()
			swept++
		}
		return true
	})
	return swept
}

// promotionMaxClass returns the maximum class index to search during promotion.
// During warmup (sizes unknown), promotion is uncapped (-1).
// After detection, caps at 2x the best-fit class size to limit waste.
func (s *Shard) promotionMaxClass(sa alloc.Allocator, bestFitIdx int) int {
	if bestFitIdx < 0 {
		return -1
	}
	if !s.sizeTracker.detected {
		return -1 // warmup: uncapped, try ALL classes
	}
	// Steady state: cap at 2x waste (next class up from best-fit)
	bestSize := sa.ClassSize(bestFitIdx)
	maxSize := bestSize * 2
	for ci := bestFitIdx + 1; ci < sa.NumClasses(); ci++ {
		if sa.ClassSize(ci) > maxSize {
			return ci - 1
		}
	}
	return sa.NumClasses() - 1
}

type errorString string

func (e errorString) Error() string { return string(e) }

func (s *Shard) handleGet(op ShardOp) OpResult {
	s.metrics.IncrGets()

	entry, _, found := s.idx.Lookup(op.KeyHash, uint16(len(op.Key)))
	if !found {
		s.metrics.IncrMisses()
		return OpResult{Found: false}
	}

	// Check TTL
	if entry.TTL > 0 && time.Now().UnixMilli() > entry.TTL {
		// Expired â€” delete it
		s.idx.Delete(op.KeyHash, uint16(len(op.Key)))
		s.eviction.Remove(op.KeyHash)
		s.metrics.IncrMisses()
		s.metrics.IncrTTLExpirations()
		return OpResult{Found: false}
	}

	// Read value from correct allocator (migration-aware)
	a := s.allocForEntry(entry)
	valCI := int(entry.ValueClassIdx)
	if entry.Flags&index.FlagHasClassIdx == 0 {
		valCI = findAllocClassIn(a, uint64(entry.ValueLen))
	}
	if valCI < 0 || valCI >= a.NumClasses() {
		return OpResult{Found: false}
	}
	valAlloc := alloc.Allocation{ClassIdx: valCI, Offset: entry.ValueOffset, Size: a.ClassSize(valCI)}
	valData := a.Read(valAlloc)
	if uint32(len(valData)) < entry.ValueLen {
		log.Printf("WARN: shard %d: value data shorter than entry.ValueLen (%d < %d) â€” treating as miss",
			s.id, len(valData), entry.ValueLen)
		return OpResult{Found: false}
	}
	value := make([]byte, entry.ValueLen)
	copy(value, valData[:entry.ValueLen])

	s.eviction.Access(op.KeyHash)
	s.metrics.IncrHits()
	s.metrics.AddBytesOut(int64(entry.ValueLen))
	return OpResult{
		Value: value,
		Found: true,
		OK:    true,
		AllocInfo: &AllocMeta{
			ClassIdx: valCI,
			Offset:   entry.ValueOffset,
			Size:     uint64(entry.ValueLen),
		},
	}
}

func (s *Shard) handleSet(op ShardOp) OpResult {
	s.metrics.IncrSets()
	s.metrics.AddBytesIn(int64(len(op.Key)) + int64(len(op.Value)))
	s.metrics.AddKeyBytesIn(int64(len(op.Key)))
	s.metrics.AddValueBytesIn(int64(len(op.Value)))

	// OOM admission control: fast-reject before attempting expensive eviction loop
	if err := s.checkOOM(); err != nil {
		return OpResult{Err: err}
	}

	// Track value size for auto-detection
	sa := s.allocPtr.Load().a
	valSize := uint64(len(op.Value))
	if !s.sizeTracker.detected && !s.sizeTracker.frozen {
		s.sizeTracker.record(valSize)
		if s.sizeTracker.detected {
			s.sizeTracker.setSlotUtilization(sa.SlotUtilization(s.sizeTracker.optimalSize))
		}
	}

	// Auto-rebuild after detection (no FLUSH required) â€” slab mode only
	// During an active migration or allocator construction, skip triggering another rebuild
	if s.migrate == nil && !s.allocBuilding.Load() && s.config.AllocatorMode != "offset" && s.config.AutoTuneSlabs && s.sizeTracker.justDetected {
		// Skip if allocator is already tuned for this size (no-op migration)
		if s.sizeTracker.optimalSize == s.config.ModelPageBytes && s.config.ModelPageBytes > 0 {
			s.sizeTracker.justDetected = false
			s.sizeTracker.frozen = true
			s.sizeTracker.updateCachedSnapshot()
		} else if s.startMigration(nil, true) {
			// Only clear justDetected if migration actually started (not deferred by sem)
			s.sizeTracker.justDetected = false
		}
	}

	var nowMs int64
	if op.TTLMs > 0 {
		nowMs = time.Now().UnixMilli()
	}
	if err := s.doSetEntry(sa, op.Key, op.KeyHash, op.Value, op.TTLMs, nowMs); err != nil {
		return OpResult{Err: err}
	}
	return OpResult{OK: true}
}

// doSetEntry performs the core SET logic for a single key-value pair:
// allocate/migrate key + value, insert into index, update eviction.
// The caller handles metrics, size tracking, and error disposition.
func (s *Shard) doSetEntry(
	sa alloc.Allocator, key []byte, keyHash uint64, value []byte,
	ttlMs int64, nowMs int64,
) error {
	keyLen := uint16(len(key))
	valLen := uint32(len(value))

	// During migration, new SETs go to the new allocator
	targetAlloc := sa
	var flags uint16
	if s.migrate != nil {
		targetAlloc = s.migrate.newAlloc
		flags = index.FlagMigrated
	}

	// Check if key already exists â€” if so, free old allocations
	oldEntry, _, oldFound := s.idx.Lookup(keyHash, keyLen)
	if oldFound {
		s.freeEntryValue(oldEntry)
		if oldEntry.KeyLen != keyLen {
			s.freeEntryKey(oldEntry)
		}
	}

	// Allocate for key (if new or key size changed)
	var keyAlloc alloc.Allocation
	keyAllocFailed := false
	if !oldFound || oldEntry.KeyLen != keyLen {
		var err error
		keyAlloc, err = s.allocWithEvictionIn(targetAlloc, uint64(keyLen), EvictionKeyAlloc)
		if err != nil {
			return fmt.Errorf("key alloc failed (shard %d, key_size=%d): %w", s.id, keyLen, err)
		}
		targetAlloc.Write(keyAlloc, key)
	} else {
		// Reuse old key allocation â€” but if migrating and old key is in old alloc, need to move it
		if s.migrate != nil && oldEntry.Flags&index.FlagMigrated == 0 {
			oldKeyAlloc := s.allocForEntry(oldEntry)
			oldKeyClass := int(oldEntry.KeyClassIdx)
			if oldEntry.Flags&index.FlagHasClassIdx == 0 {
				oldKeyClass = findAllocClassIn(oldKeyAlloc, uint64(keyLen))
			}
			if oldKeyClass >= 0 {
				keyData := oldKeyAlloc.Read(alloc.Allocation{ClassIdx: oldKeyClass, Offset: oldEntry.KeyOffset, Size: oldKeyAlloc.ClassSize(oldKeyClass)})
				keyCopy := make([]byte, keyLen)
				copy(keyCopy, keyData[:keyLen])
				var err error
				keyAlloc, err = targetAlloc.Alloc(uint64(keyLen))
				if err != nil {
					return fmt.Errorf("key migration alloc failed (shard %d): %w", s.id, err)
				}
				targetAlloc.Write(keyAlloc, keyCopy)
				oldKeyAlloc.Free(alloc.Allocation{ClassIdx: oldKeyClass, Offset: oldEntry.KeyOffset, Size: oldKeyAlloc.ClassSize(oldKeyClass)})
			}
		} else {
			reuseAlloc := targetAlloc
			if s.migrate != nil && oldEntry.Flags&index.FlagMigrated != 0 {
				reuseAlloc = s.migrate.newAlloc
			}
			reuseCI := int(oldEntry.KeyClassIdx)
			if oldEntry.Flags&index.FlagHasClassIdx == 0 {
				reuseCI = findAllocClassIn(reuseAlloc, uint64(keyLen))
			}
			keyAlloc = alloc.Allocation{ClassIdx: reuseCI, Offset: oldEntry.KeyOffset, Size: reuseAlloc.ClassSize(reuseCI)}
		}
	}

	// Allocate for value
	valAlloc, err := s.allocWithEvictionIn(targetAlloc, uint64(valLen), EvictionValueAlloc)
	if err != nil && s.migrate != nil && targetAlloc == s.migrate.newAlloc {
		// newAlloc full â€” fall back to oldAlloc so migrateBatch moves it later
		fallbackAlloc := s.allocPtr.Load().a
		if !oldFound || oldEntry.KeyLen != keyLen {
			s.migrate.newAlloc.Free(keyAlloc)
			keyAlloc, err = s.allocWithEvictionIn(fallbackAlloc, uint64(keyLen), EvictionKeyAlloc)
			if err != nil {
				keyAllocFailed = true
			} else {
				fallbackAlloc.Write(keyAlloc, key)
			}
		}
		if !keyAllocFailed {
			valAlloc, err = s.allocWithEvictionIn(fallbackAlloc, uint64(valLen), EvictionValueAlloc)
			if err == nil {
				targetAlloc = fallbackAlloc
				flags = 0 // stays in oldAlloc; migrateBatch will move it
			}
		}
	}
	if err != nil {
		if !oldFound || oldEntry.KeyLen != keyLen {
			if !keyAllocFailed {
				targetAlloc.Free(keyAlloc)
			}
		}
		return fmt.Errorf("alloc failed (shard %d, val_size=%d): %w", s.id, valLen, err)
	}
	targetAlloc.Write(valAlloc, value)

	// Compute TTL
	var ttl int64
	if ttlMs > 0 {
		ttl = nowMs + ttlMs
	}

	// Insert into index â€” store actual class indices for correct free during promotion/migration
	entryFlags := flags | index.FlagHasClassIdx
	if valAlloc.ClassIdx != targetAlloc.FindClass(uint64(valLen)) {
		entryFlags |= index.FlagPromoted
	}
	entry := index.Entry{
		KeyHash:       keyHash,
		KeyLen:        keyLen,
		KeyOffset:     keyAlloc.Offset,
		ValueOffset:   valAlloc.Offset,
		ValueLen:      valLen,
		TTL:           ttl,
		Flags:         entryFlags,
		ValueClassIdx: uint8(valAlloc.ClassIdx),
		KeyClassIdx:   uint8(keyAlloc.ClassIdx),
	}
	// L2: If index is at max capacity and this is a new key, trigger eviction
	if !oldFound && s.idx.IsFull() {
		s.eviction.EvictOne()
	}
	s.idx.Insert(keyHash, entry)

	// Admit to eviction tracker
	if !oldFound {
		_ = s.eviction.Admit(keyHash, keyLen)
		s.eviction.Access(keyHash) // mark visited so SIEVE doesn't evict on first hand pass
	} else {
		s.eviction.Access(keyHash)
	}

	return nil
}

func (s *Shard) handleDelete(op ShardOp) OpResult {
	keyLen := uint16(len(op.Key))
	entry, _, found := s.idx.Lookup(op.KeyHash, keyLen)
	if !found {
		return OpResult{OK: false}
	}

	// Free allocations
	s.freeEntryKey(entry)
	s.freeEntryValue(entry)

	s.idx.Delete(op.KeyHash, keyLen)
	s.eviction.Remove(op.KeyHash)
	s.metrics.IncrDeletes()

	// Clear OOM state: a delete freed space
	s.allocSucceeded()

	return OpResult{OK: true}
}

func (s *Shard) handleMGet(op ShardOp) OpResult {
	values := make([][]byte, len(op.Keys))
	founds := make([]bool, len(op.Keys))

	now := time.Now().UnixMilli()
	sa := s.allocPtr.Load().a
	var totalBytesOut int64
	var hits, misses int64

	for i, key := range op.Keys {
		keyHash := op.KeyHashes[i]
		entry, _, found := s.idx.Lookup(keyHash, uint16(len(key)))
		if !found {
			misses++
			continue
		}
		if entry.TTL > 0 && now > entry.TTL {
			s.idx.Delete(keyHash, uint16(len(key)))
			s.eviction.Remove(keyHash)
			s.metrics.IncrTTLExpirations()
			misses++
			continue
		}

		a := s.allocForEntryOr(entry, sa)
		valCI := int(entry.ValueClassIdx)
		if entry.Flags&index.FlagHasClassIdx == 0 {
			valCI = findAllocClassIn(a, uint64(entry.ValueLen))
		}
		if valCI < 0 || valCI >= a.NumClasses() {
			misses++
			continue
		}
		valData := a.Read(alloc.Allocation{ClassIdx: valCI, Offset: entry.ValueOffset, Size: a.ClassSize(valCI)})
		if uint32(len(valData)) < entry.ValueLen {
			misses++ // corrupted entry â€” skip
			continue
		}
		value := make([]byte, entry.ValueLen)
		copy(value, valData[:entry.ValueLen])

		values[i] = value
		founds[i] = true
		totalBytesOut += int64(entry.ValueLen)
		hits++
		s.eviction.Access(keyHash)
	}

	s.metrics.AddGets(int64(len(op.Keys)))
	s.metrics.AddHits(hits)
	s.metrics.AddMisses(misses)
	s.metrics.AddBytesOut(totalBytesOut)
	return OpResult{Values: values, Founds: founds, OK: true}
}

func (s *Shard) handleMGetWithAlloc(op ShardOp) OpResult {
	n := len(op.Keys)
	values := make([][]byte, n)
	founds := make([]bool, n)
	allocInfos := make([]*AllocMeta, n)

	now := time.Now().UnixMilli()
	sa := s.allocPtr.Load().a
	var totalBytesOut int64
	var hits, misses int64

	for i, key := range op.Keys {
		keyHash := op.KeyHashes[i]
		entry, _, found := s.idx.Lookup(keyHash, uint16(len(key)))
		if !found {
			misses++
			continue
		}
		if entry.TTL > 0 && now > entry.TTL {
			s.idx.Delete(keyHash, uint16(len(key)))
			s.eviction.Remove(keyHash)
			s.metrics.IncrTTLExpirations()
			misses++
			continue
		}

		a := s.allocForEntryOr(entry, sa)
		valCI := int(entry.ValueClassIdx)
		if entry.Flags&index.FlagHasClassIdx == 0 {
			valCI = findAllocClassIn(a, uint64(entry.ValueLen))
		}
		if valCI < 0 || valCI >= a.NumClasses() {
			misses++
			continue
		}
		valData := a.Read(alloc.Allocation{ClassIdx: valCI, Offset: entry.ValueOffset, Size: a.ClassSize(valCI)})
		if uint32(len(valData)) < entry.ValueLen {
			misses++ // corrupted entry â€” skip
			continue
		}
		value := make([]byte, entry.ValueLen)
		copy(value, valData[:entry.ValueLen])

		values[i] = value
		founds[i] = true
		allocInfos[i] = &AllocMeta{
			ClassIdx: valCI,
			Offset:   entry.ValueOffset,
			Size:     uint64(entry.ValueLen),
		}
		totalBytesOut += int64(entry.ValueLen)
		hits++
		s.eviction.Access(keyHash)
	}

	s.metrics.AddGets(int64(len(op.Keys)))
	s.metrics.AddHits(hits)
	s.metrics.AddMisses(misses)
	s.metrics.AddBytesOut(totalBytesOut)
	return OpResult{Values: values, Founds: founds, AllocInfos: allocInfos, OK: true}
}

func (s *Shard) handleBatchLease(op ShardOp) OpResult {
	for _, kh := range op.KeyHashes {
		s.leases.Grant(kh, op.LeaseMs)
	}
	return OpResult{OK: true}
}

func (s *Shard) handleMSet(op ShardOp) OpResult {
	n := len(op.Keys)
	if n == 0 {
		return OpResult{OK: true}
	}

	// OOM admission control: reject entire batch before attempting allocation
	if err := s.checkOOM(); err != nil {
		return OpResult{Err: err}
	}

	// --- Load shared state ONCE ---
	sa := s.allocPtr.Load().a
	var now int64
	if op.TTLMs > 0 {
		now = time.Now().UnixMilli()
	}

	// --- Batch size tracking ---
	if !s.sizeTracker.detected && !s.sizeTracker.frozen {
		sizes := make([]uint64, n)
		for i, v := range op.Values {
			sizes[i] = uint64(len(v))
		}
		s.sizeTracker.recordBatch(sizes)
		if s.sizeTracker.detected {
			s.sizeTracker.setSlotUtilization(sa.SlotUtilization(s.sizeTracker.optimalSize))
		}
	}

	// --- Auto-rebuild trigger (once per batch) ---
	if s.migrate == nil && !s.allocBuilding.Load() && s.config.AllocatorMode != "offset" && s.config.AutoTuneSlabs && s.sizeTracker.justDetected {
		if s.sizeTracker.optimalSize == s.config.ModelPageBytes && s.config.ModelPageBytes > 0 {
			s.sizeTracker.justDetected = false
			s.sizeTracker.frozen = true
			s.sizeTracker.updateCachedSnapshot()
		} else if s.startMigration(nil, true) {
			s.sizeTracker.justDetected = false
		}
	}

	// --- Per-key loop (continue on error, collect per-key statuses) ---
	var totalBytesIn, totalKeyBytes, totalValBytes int64
	statuses := make([]byte, n)
	var failed int

	for i, key := range op.Keys {
		value := op.Values[i]
		keyHash := op.KeyHashes[i]

		totalBytesIn += int64(len(key)) + int64(len(value))
		totalKeyBytes += int64(len(key))
		totalValBytes += int64(len(value))

		if err := s.doSetEntry(sa, key, keyHash, value, op.TTLMs, now); err != nil {
			statuses[i] = 1
			failed++
		}
	}

	// --- Flush metrics ONCE ---
	s.metrics.AddSets(int64(n))
	s.metrics.AddBytesIn(totalBytesIn)
	s.metrics.AddKeyBytesIn(totalKeyBytes)
	s.metrics.AddValueBytesIn(totalValBytes)

	return OpResult{OK: failed == 0, SetStatuses: statuses}
}

func (s *Shard) handleTest(op ShardOp) OpResult {
	// Normalize single-key to batch of 1
	keys := op.Keys
	hashes := op.KeyHashes
	single := len(keys) == 0
	if single {
		keys = [][]byte{op.Key}
		hashes = []uint64{op.KeyHash}
	}

	now := time.Now().UnixMilli()
	founds := make([]bool, len(keys))
	var hits, misses int64
	for i, key := range keys {
		entry, _, found := s.idx.Lookup(hashes[i], uint16(len(key)))
		if !found {
			misses++
			continue
		}
		if entry.TTL > 0 && now > entry.TTL {
			s.idx.Delete(hashes[i], uint16(len(key)))
			s.eviction.Remove(hashes[i])
			misses++
			s.metrics.IncrTTLExpirations()
			continue
		}
		s.eviction.Access(hashes[i])
		hits++
		founds[i] = true
	}

	// Batch metrics â€” single atomic add instead of per-key
	s.metrics.AddExists(int64(len(keys)))
	s.metrics.AddExistsHits(hits)
	s.metrics.AddExistsMisses(misses)

	if single {
		return OpResult{Found: founds[0], OK: true}
	}
	return OpResult{Founds: founds, OK: true}
}

func (s *Shard) handleLease(op ShardOp) OpResult {
	s.leases.Grant(op.KeyHash, op.LeaseMs)
	return OpResult{OK: true}
}

func (s *Shard) handlePin(op ShardOp) OpResult {
	s.leases.Pin(op.KeyHash)
	return OpResult{OK: true}
}

func (s *Shard) handleUnpin(op ShardOp) OpResult {
	s.leases.Unpin(op.KeyHash)
	return OpResult{OK: true}
}

func (s *Shard) handleInfo(op ShardOp) OpResult {
	a := s.allocPtr.Load().a
	info := map[string]interface{}{
		"shard_id":      s.id,
		"index_count":   s.idx.Count(),
		"index_cap":     s.idx.Capacity(),
		"eviction_size": s.eviction.Size(),
		"alloc_bytes":   a.AllocatedBytes(),
		"alloc_classes": a.NumClasses(),
	}
	return OpResult{Info: info, OK: true}
}

func (s *Shard) handleMDel(op ShardOp) OpResult {
	for i, key := range op.Keys {
		delOp := ShardOp{
			Type:    OpDelete,
			Key:     key,
			KeyHash: op.KeyHashes[i],
		}
		s.handleDelete(delOp)
	}
	return OpResult{OK: true}
}

func (s *Shard) handleFlush(op ShardOp) OpResult {
	// Discard any pending async allocator â€” FLUSH invalidates whatever it's building
	select {
	case pa := <-s.pendingAlloc:
		pa.newAlloc.Close()
		s.allocBuilding.Store(false)
		s.releaseMigrateSem()
		log.Printf("[rebalance] shard %d: pending async allocator discarded by FLUSH", s.id)
	default:
	}

	// Abort any in-progress migration â€” FLUSH destroys everything.
	// Fast-path: no synchronous consolidation needed since FLUSH clears all data.
	if s.migrate != nil {
		ms := s.migrate
		// Wait briefly for pre-registration goroutines, then discard
		for _, ch := range ms.preRegDone {
			select {
			case <-ch:
			case <-time.After(5 * time.Second):
			}
		}
		for _, l := range s.allocListeners {
			if pr, ok := l.(AllocPreRegisterer); ok {
				pr.DiscardPreRegistered(s.id)
			}
		}
		ms.newAlloc.Close()
		s.metrics.SetMigrationActive(0)
		log.Printf("[rebalance] shard %d: migration cancelled by FLUSH", s.id)
		s.migrate = nil
		s.releaseMigrateSem()
	}

	// Free all key and value allocations
	s.idx.Iter(func(_ uint64, e index.Entry) bool {
		s.freeEntryKey(e)
		s.freeEntryValue(e)
		return true
	})

	// Auto-tune: rebuild slab allocator with detected optimal classes (slab mode only)
	if s.config.AllocatorMode != "offset" && s.config.AutoTuneSlabs && s.sizeTracker.detected {
		flushDedicated := s.config.SlabDistribution == "dedicated" || s.config.SlabDistribution == "auto"
		newCfg := alloc.SlabConfig{
			MaxMemoryBytes: s.config.MaxMemoryBytes,
			HugePageSizeKB: s.config.HugePageSizeKB,
			ModelPageBytes: s.sizeTracker.optimalSize,
			Dedicated:      flushDedicated,
		}

		oldAlloc := s.allocPtr.Load().a

		// Optimization: when no RDMA listeners, close old allocator BEFORE creating the new one.
		// FLUSH already freed all entries, so the old allocator's memory is unused.
		// Closing first frees hugepages for the new allocator, avoiding a 2Ã— spike.
		if len(s.allocListeners) == 0 {
			oldAlloc.Close()
			newAlloc, err := alloc.NewSlabAllocator(newCfg)
			if err != nil {
				// Old allocator is gone â€” must create a fallback with default config
				log.Printf("[cama] shard %d: auto-tune failed after close: %v (creating default allocator)", s.id, err)
				fallbackCfg := alloc.SlabConfig{
					MaxMemoryBytes: s.config.MaxMemoryBytes,
					HugePageSizeKB: s.config.HugePageSizeKB,
					ModelPageBytes: s.config.ModelPageBytes,
				}
				fallback, ferr := alloc.NewSlabAllocator(fallbackCfg)
				if ferr != nil {
					log.Printf("[cama] shard %d: CRITICAL â€” fallback allocator also failed: %v", s.id, ferr)
				} else {
					s.allocPtr.Store(&allocBox{a: fallback})
				}
			} else {
				s.allocPtr.Store(&allocBox{a: newAlloc})
				if s.config.VerboseShardLogging {
					slots, classSize := newAlloc.ModelClassCapacity(s.sizeTracker.optimalSize)
					log.Printf("[cama] shard %d: auto-tuned slabs (in-place) â€” model_page_bytes=%d, model_slots=%d (class=%d)",
						s.id, s.sizeTracker.optimalSize, slots, classSize)
				}
			}
		} else {
			// RDMA listeners need both allocators to coexist during MR deregistration
			newAlloc, err := alloc.NewSlabAllocator(newCfg)
			if err != nil {
				log.Printf("[cama] shard %d: auto-tune failed: %v (keeping old allocator)", s.id, err)
			} else {
				s.allocPtr.Store(&allocBox{a: newAlloc})
				s.notifyAllocListeners(oldAlloc, newAlloc)
				if s.config.VerboseShardLogging {
					slots, classSize := newAlloc.ModelClassCapacity(s.sizeTracker.optimalSize)
					log.Printf("[cama] shard %d: auto-tuned slabs â€” model_page_bytes=%d, model_slots=%d (class=%d)",
						s.id, s.sizeTracker.optimalSize, slots, classSize)
				}
			}
		}
	}

	// Replace index with a fresh table
	idxCap := s.config.IndexCapacity
	if idxCap == 0 {
		idxCap = DefaultIndexCapacity
	}
	s.idx = index.NewTable(idxCap)

	// Replace eviction engine with a fresh instance
	maxKeys := idxCap * index.MaxLoadNumerator / index.MaxLoadDenominator
	s.eviction = eviction.NewPolicy(s.config.EvictionPolicy, maxKeys, func(keyHash uint64, keyLen uint16) { s.evictKey(keyHash, keyLen) })

	// Replace lease table
	maxLease := s.config.MaxLeaseDurMs
	if maxLease == 0 {
		maxLease = 30000
	}
	s.leases = lease.NewTable(maxLease)

	s.metrics.ResetEpoch()

	// Reset size tracker so it can re-detect after workload changes
	s.sizeTracker.reset()

	// Reset pressure tracker for the (possibly new) allocator
	s.pressureTracker.reset(s.allocPtr.Load().a.NumClasses())

	// Clear OOM state: FLUSH freed everything
	s.consecAllocFails = 0
	s.oomActive = false
	s.metrics.SetMemoryPressure(0)

	return OpResult{OK: true}
}

// notifyAllocListeners notifies listeners of an allocator swap and manages the old allocator lifecycle.
// If listeners are registered, they own the old allocator lifecycle (e.g., RDMA deferred close).
// If no listeners, the old allocator is closed immediately.
func (s *Shard) notifyAllocListeners(oldAlloc, newAlloc alloc.Allocator) {
	if len(s.allocListeners) > 0 {
		change := AllocatorChange{
			ShardID:      s.id,
			OldAllocator: oldAlloc,
			NewAllocator: newAlloc,
		}
		for _, l := range s.allocListeners {
			l.OnAllocatorChanged(change)
		}
	} else {
		// No RDMA listeners â€” safe to close immediately
		oldAlloc.Close()
	}
}

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
			log.Printf("[cama] shard %d: async allocator construction failed: %v (keeping old allocator)", shardID, err)
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
			log.Printf("[cama] shard %d: migration batch failed on key alloc: %v (aborting)", s.id, kerr)
			return false
		}
		ms.newAlloc.Write(newKeyAlloc, keyCopy)

		newValAlloc, verr := ms.newAlloc.Alloc(uint64(e.ValueLen))
		if verr != nil {
			ms.newAlloc.Free(newKeyAlloc)
			log.Printf("[cama] shard %d: migration batch failed on value alloc: %v (aborting)", s.id, verr)
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
		log.Printf("[cama] shard %d: migration aborted at entry %d (alloc failure)", s.id, ms.migrated)
		s.abortMigration()
		return true // migration "complete" (aborted)
	}

	return done
}

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
			log.Printf("[cama] shard %d: WARNING â€” %d region(s) fell back to regular 4KB pages (hugepage pressure?)",
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
			log.Printf("[cama] shard %d: page-size hint fast-swap failed: %v (keeping old allocator)", s.id, err)
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
			log.Printf("[cama] shard %d: page-size hint %d bytes â€” fast-swapped allocator (%s, empty shard)", s.id, hintBytes, mode)
			return OpResult{OK: true}
		}
	}

	// Slow path: shard has data â€” full ZeroLatencyBalance migration
	if s.config.VerboseShardLogging {
		log.Printf("[cama] shard %d: page-size hint %d bytes â€” triggering slab rebuild", s.id, hintBytes)
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

func (s *Shard) handleKeys(op ShardOp) OpResult {
	pattern := string(op.Pattern)
	re, err := regexp.Compile(pattern)
	if err != nil {
		return OpResult{Err: errorString("invalid key pattern: " + err.Error())}
	}

	var matched [][]byte
	s.idx.Iter(func(_ uint64, e index.Entry) bool {
		if e.KeyOffset == 0 || e.KeyLen == 0 {
			return true
		}
		// Check TTL
		if e.TTL > 0 && time.Now().UnixMilli() > e.TTL {
			return true
		}
		a := s.allocForEntry(e)
		keyClassIdx := findAllocClassIn(a, uint64(e.KeyLen))
		if keyClassIdx < 0 {
			return true
		}
		keyData := a.Read(alloc.Allocation{ClassIdx: keyClassIdx, Offset: e.KeyOffset, Size: a.ClassSize(keyClassIdx)})
		key := make([]byte, e.KeyLen)
		copy(key, keyData[:e.KeyLen])
		if re.Match(key) {
			matched = append(matched, key)
		}
		return true
	})
	return OpResult{MatchedKeys: matched, OK: true}
}

// handleSnapshot iterates all index entries and returns KVEntries for serialization.
func (s *Shard) handleSnapshot(op ShardOp) OpResult {
	now := time.Now().UnixMilli()
	var entries []snapshot.KVEntry

	s.idx.Iter(func(_ uint64, e index.Entry) bool {
		// Skip expired entries
		if e.TTL > 0 && now > e.TTL {
			return true
		}

		a := s.allocForEntry(e)

		// Read key
		keyClassIdx := findAllocClassIn(a, uint64(e.KeyLen))
		if keyClassIdx < 0 {
			return true
		}
		keyData := a.Read(alloc.Allocation{ClassIdx: keyClassIdx, Offset: e.KeyOffset, Size: a.ClassSize(keyClassIdx)})
		key := make([]byte, e.KeyLen)
		copy(key, keyData[:e.KeyLen])

		// Read value
		valClassIdx := findAllocClassIn(a, uint64(e.ValueLen))
		if valClassIdx < 0 {
			return true
		}
		valData := a.Read(alloc.Allocation{ClassIdx: valClassIdx, Offset: e.ValueOffset, Size: a.ClassSize(valClassIdx)})
		value := make([]byte, e.ValueLen)
		copy(value, valData[:e.ValueLen])

		// Compute remaining TTL
		var ttlMs int64
		if e.TTL > 0 {
			ttlMs = e.TTL - now
			if ttlMs <= 0 {
				return true // expired during iteration
			}
		}

		entries = append(entries, snapshot.KVEntry{Key: key, Value: value, TTLMs: ttlMs})
		return true
	})

	return OpResult{SnapshotEntries: entries, OK: true}
}

// handleRestore bulk-loads entries from a snapshot. Mirrors handleSet+doSetEntry:
// uses eviction-aware allocation so post-boot admin restore makes room under pressure,
// frees pre-existing entries before overwriting to avoid alloc leak / double-Admit, and
// updates metrics + size tracker so Prometheus and auto-tune see restored data.
func (s *Shard) handleRestore(op ShardOp) OpResult {
	loaded := 0
	now := time.Now().UnixMilli()

	for _, entry := range op.RestoreEntries {
		var ttlAbs int64
		if entry.TTLMs > 0 {
			ttlAbs = now + entry.TTLMs
		}

		keyHash := index.KeyHash(entry.Key)
		keyLen := uint16(len(entry.Key))
		valLen := uint32(len(entry.Value))

		sa := s.allocPtr.Load().a

		// Free any pre-existing entry for this key so we don't leak its allocation
		// or double-Admit on the eviction tracker (admin restore can hit a live cache).
		oldEntry, _, oldFound := s.idx.Lookup(keyHash, keyLen)
		if oldFound {
			s.freeEntryValue(oldEntry)
			s.freeEntryKey(oldEntry)
		}

		keyAlloc, err := s.allocWithEvictionIn(sa, uint64(keyLen), EvictionKeyAlloc)
		if err != nil {
			continue // out of space even after eviction
		}
		sa.Write(keyAlloc, entry.Key)

		valAlloc, err := s.allocWithEvictionIn(sa, uint64(valLen), EvictionValueAlloc)
		if err != nil {
			sa.Free(keyAlloc)
			continue
		}
		sa.Write(valAlloc, entry.Value)

		entryFlags := uint16(index.FlagHasClassIdx)
		if valAlloc.ClassIdx != sa.FindClass(uint64(valLen)) {
			entryFlags |= index.FlagPromoted
		}
		e := index.Entry{
			KeyHash:       keyHash,
			KeyLen:        keyLen,
			KeyOffset:     keyAlloc.Offset,
			ValueOffset:   valAlloc.Offset,
			ValueLen:      valLen,
			TTL:           ttlAbs,
			Flags:         entryFlags,
			ValueClassIdx: uint8(valAlloc.ClassIdx),
			KeyClassIdx:   uint8(keyAlloc.ClassIdx),
		}
		s.idx.Insert(keyHash, e)

		if oldFound {
			s.eviction.Access(keyHash)
		} else {
			s.eviction.Admit(keyHash, keyLen)
			s.eviction.Access(keyHash)
		}

		s.metrics.IncrSets()
		s.metrics.AddBytesIn(int64(keyLen) + int64(valLen))
		s.metrics.AddKeyBytesIn(int64(keyLen))
		s.metrics.AddValueBytesIn(int64(valLen))
		if !s.sizeTracker.detected && !s.sizeTracker.frozen {
			s.sizeTracker.record(uint64(valLen))
		}

		loaded++
	}

	return OpResult{Loaded: loaded, OK: true}
}
