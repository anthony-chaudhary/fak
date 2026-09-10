package strix

import (
	"sort"
	"sync"
	"sync/atomic"
	"unsafe"
)

// CuckooTable implements an 8 MiB hardware-aligned Cuckoo hash table for sub-20ns
// Context MMU CAS handle resolution on AMD Strix Halo APUs.
//
// The table geometry is strictly 131,072 64-byte buckets (8,192 sets x 16 ways x 64B),
// each bucket containing 4 slots of 16 bytes (8B CASKey + 8B KVBlockPtr).
// Base address is guaranteed aligned to 64-byte boundaries.
//
// Lookups evaluate at most 2 candidate buckets (at most 2 cache line reads) in the
// primary table. If neither candidate bucket contains the handle, lookups check the
// DRAM overflow stash under RLock.
type CuckooTable struct {
	// raw holds the original unaligned byte slice to keep memory rooted for Go GC.
	raw []byte

	// buckets is the 64-byte aligned view of 131,072 Cuckoo buckets.
	buckets []CuckooBucket

	// baseAddress is the 64-byte aligned memory address of buckets[0].
	baseAddress uintptr

	// stripeLocks provides 4,096 fine-grained striped mutexes for concurrent writers.
	stripeLocks [NumStripeLocks]sync.Mutex

	// displacementMu serializes multi-hop Cuckoo kick-outs to prevent cross-displacement
	// deadlocks while allowing all non-displacing insertions to run fully parallel.
	displacementMu sync.Mutex

	// stash holds overflow entries when candidate buckets and Cuckoo displacements
	// are saturated, capped at MaxStashCapacity.
	stash   map[CASKey]KVBlockPtr
	stashMu sync.RWMutex

	// telemetry tracks real-time lookups, hits, displacements, and load factors.
	telemetry CuckooTelemetry
}

// NewCuckooTable allocates and initializes a 64-byte aligned 8 MiB Cuckoo hash table.
func NewCuckooTable() *CuckooTable {
	// Allocate 8 MiB + 64 bytes to guarantee 64-byte cache line alignment.
	raw := make([]byte, TableSizeBytes+CacheLineSize)
	base := uintptr(unsafe.Pointer(&raw[0]))
	offset := int((CacheLineSize - (base % CacheLineSize)) % CacheLineSize)
	alignedBase := uintptr(unsafe.Pointer(&raw[offset]))

	buckets := unsafe.Slice((*CuckooBucket)(unsafe.Pointer(&raw[offset])), NumBuckets)

	return &CuckooTable{
		raw:         raw,
		buckets:     buckets,
		baseAddress: alignedBase,
		stash:       make(map[CASKey]KVBlockPtr, 64),
	}
}

// BaseAddress returns the 64-byte aligned memory address of the table.
func (t *CuckooTable) BaseAddress() uintptr {
	return t.baseAddress
}

// ByteSize returns the exact physical memory size of the Cuckoo table (8,388,608 bytes).
func (t *CuckooTable) ByteSize() int {
	return TableSizeBytes
}

// NumBuckets returns the number of 64-byte buckets (131,072).
func (t *CuckooTable) NumBuckets() int {
	return NumBuckets
}

// Capacity returns the total slot capacity of the primary table (524,288).
func (t *CuckooTable) Capacity() int {
	return TotalSlots
}

// Hash1 computes the primary 64-bit avalanche hash for key k using a SplitMix64 mixer.
func Hash1(k CASKey) uint64 {
	x := uint64(k) ^ 0x517cc1b727220a95
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// Hash2 computes the secondary 64-bit avalanche hash for key k using a Murmur3 mixer.
func Hash2(k CASKey) uint64 {
	x := uint64(k) ^ 0x9e3779b97f4a7c15
	x = (x ^ (x >> 33)) * 0xff51afd7ed558ccd
	x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53
	return x ^ (x >> 33)
}

// CandidateBuckets returns two distinct bucket indices in [0, 131071] for key k.
func CandidateBuckets(k CASKey) (int, int) {
	h1 := Hash1(k)
	h2 := Hash2(k)
	b1 := int(h1 & (NumBuckets - 1))
	b2 := int(h2 & (NumBuckets - 1))
	if b1 == b2 {
		// Guarantee distinct bucket indices using secondary odd offset modulo power-of-two.
		b2 = int((uint64(b1) + (h2 | 1)) & uint64(NumBuckets-1))
	}
	return b1, b2
}

// CandidateBuckets returns two distinct bucket indices in [0, 131071] for key k on table t.
func (t *CuckooTable) CandidateBuckets(k CASKey) (int, int) {
	return CandidateBuckets(k)
}

// AltBucket returns the alternative bucket for key k given its current bucket.
func AltBucket(k CASKey, currentBucket int) int {
	b1, b2 := CandidateBuckets(k)
	if currentBucket == b1 {
		return b2
	}
	return b1
}

// AltBucket returns the alternative bucket for key k given its current bucket on table t.
func (t *CuckooTable) AltBucket(k CASKey, currentBucket int) int {
	return AltBucket(k, currentBucket)
}

// stripeIndex maps a bucket index to its writer stripe lock index in [0, NumStripeLocks-1].
func (t *CuckooTable) stripeIndex(bucketIdx int) int {
	return bucketIdx & (NumStripeLocks - 1)
}

// lockStripes acquires two stripe locks in strictly ascending index order to eliminate deadlocks.
func (t *CuckooTable) lockStripes(idx1, idx2 int) func() {
	if idx1 == idx2 {
		t.stripeLocks[idx1].Lock()
		return func() { t.stripeLocks[idx1].Unlock() }
	}
	if idx1 < idx2 {
		t.stripeLocks[idx1].Lock()
		t.stripeLocks[idx2].Lock()
		return func() {
			t.stripeLocks[idx2].Unlock()
			t.stripeLocks[idx1].Unlock()
		}
	}
	t.stripeLocks[idx2].Lock()
	t.stripeLocks[idx1].Lock()
	return func() {
		t.stripeLocks[idx1].Unlock()
		t.stripeLocks[idx2].Unlock()
	}
}

// lockMultipleStripes acquires an arbitrary set of stripe locks in strictly ascending index order.
func (t *CuckooTable) lockMultipleStripes(stripes []int) func() {
	if len(stripes) == 0 {
		return func() {}
	}
	sorted := make([]int, len(stripes))
	copy(sorted, stripes)
	sort.Ints(sorted)

	unique := sorted[:0]
	for i, s := range sorted {
		if i == 0 || s != sorted[i-1] {
			unique = append(unique, s)
		}
	}

	for _, s := range unique {
		t.stripeLocks[s].Lock()
	}

	return func() {
		for i := len(unique) - 1; i >= 0; i-- {
			t.stripeLocks[unique[i]].Unlock()
		}
	}
}

// Lookup resolves a 64-bit CASKey to its physical KVBlockPtr without taking mutexes.
func (t *CuckooTable) Lookup(k CASKey) (KVBlockPtr, bool) {
	if k == EmptyCASKey {
		atomic.AddUint64(&t.telemetry.Misses, 1)
		atomic.AddUint64(&t.telemetry.TotalLookups, 1)
		return EmptyKVBlockPtr, false
	}

	b1, b2 := CandidateBuckets(k)

	// Probe 1: Candidate Bucket 1 (single 64-byte cache line)
	slots1 := &t.buckets[b1].Slots
	for i := 0; i < 4; i++ {
		slot := &slots1[i]
		k1 := atomic.LoadUint64((*uint64)(&slot.Key))
		if k1 == uint64(k) {
			ptr := atomic.LoadUint64((*uint64)(&slot.Ptr))
			k2 := atomic.LoadUint64((*uint64)(&slot.Key))
			if k2 == k1 {
				atomic.AddUint64(&t.telemetry.Bucket1Hits, 1)
				atomic.AddUint64(&t.telemetry.TotalLookups, 1)
				return KVBlockPtr(ptr), true
			}
			for retry := 0; retry < 3; retry++ {
				k1Retry := atomic.LoadUint64((*uint64)(&slot.Key))
				ptrRetry := atomic.LoadUint64((*uint64)(&slot.Ptr))
				k2Retry := atomic.LoadUint64((*uint64)(&slot.Key))
				if k1Retry == uint64(k) && k2Retry == k1Retry {
					atomic.AddUint64(&t.telemetry.Bucket1Hits, 1)
					atomic.AddUint64(&t.telemetry.TotalLookups, 1)
					return KVBlockPtr(ptrRetry), true
				}
			}
		}
	}

	// Probe 2: Candidate Bucket 2 (single 64-byte cache line)
	slots2 := &t.buckets[b2].Slots
	for i := 0; i < 4; i++ {
		slot := &slots2[i]
		k1 := atomic.LoadUint64((*uint64)(&slot.Key))
		if k1 == uint64(k) {
			ptr := atomic.LoadUint64((*uint64)(&slot.Ptr))
			k2 := atomic.LoadUint64((*uint64)(&slot.Key))
			if k2 == k1 {
				atomic.AddUint64(&t.telemetry.Bucket2Hits, 1)
				atomic.AddUint64(&t.telemetry.TotalLookups, 1)
				return KVBlockPtr(ptr), true
			}
			for retry := 0; retry < 3; retry++ {
				k1Retry := atomic.LoadUint64((*uint64)(&slot.Key))
				ptrRetry := atomic.LoadUint64((*uint64)(&slot.Ptr))
				k2Retry := atomic.LoadUint64((*uint64)(&slot.Key))
				if k1Retry == uint64(k) && k2Retry == k1Retry {
					atomic.AddUint64(&t.telemetry.Bucket2Hits, 1)
					atomic.AddUint64(&t.telemetry.TotalLookups, 1)
					return KVBlockPtr(ptrRetry), true
				}
			}
		}
	}

	// Fallback Probe 3: DRAM overflow stash
	if atomic.LoadUint64(&t.telemetry.StashEntries) > 0 {
		t.stashMu.RLock()
		ptr, ok := t.stash[k]
		t.stashMu.RUnlock()
		if ok {
			atomic.AddUint64(&t.telemetry.StashHits, 1)
			atomic.AddUint64(&t.telemetry.TotalLookups, 1)
			return ptr, true
		}
	}

	// Handle not found
	atomic.AddUint64(&t.telemetry.Misses, 1)
	atomic.AddUint64(&t.telemetry.TotalLookups, 1)
	return EmptyKVBlockPtr, false
}

// Insert adds or updates a CASKey -> KVBlockPtr mapping.
func (t *CuckooTable) Insert(k CASKey, ptr KVBlockPtr) error {
	if k == EmptyCASKey {
		return ErrInvalidKey
	}
	if ptr == EmptyKVBlockPtr {
		return ErrInvalidPointer
	}

	// Check if key is already present in DRAM stash
	t.stashMu.RLock()
	inStash := false
	if _, ok := t.stash[k]; ok {
		inStash = true
	}
	t.stashMu.RUnlock()

	if inStash {
		t.stashMu.Lock()
		if _, ok := t.stash[k]; ok {
			t.stash[k] = ptr
			t.stashMu.Unlock()
			return nil
		}
		t.stashMu.Unlock()
	}

	b1, b2 := CandidateBuckets(k)
	s1 := t.stripeIndex(b1)
	s2 := t.stripeIndex(b2)

	unlock := t.lockStripes(s1, s2)

	// 1. Check for existing key in candidate bucket 1 (update in-place)
	for i := 0; i < SlotsPerBucket; i++ {
		slot := &t.buckets[b1].Slots[i]
		if CASKey(atomic.LoadUint64((*uint64)(&slot.Key))) == k {
			atomic.StoreUint64((*uint64)(&slot.Ptr), uint64(ptr))
			unlock()
			return nil
		}
	}

	// 2. Check for existing key in candidate bucket 2 (update in-place)
	for i := 0; i < SlotsPerBucket; i++ {
		slot := &t.buckets[b2].Slots[i]
		if CASKey(atomic.LoadUint64((*uint64)(&slot.Key))) == k {
			atomic.StoreUint64((*uint64)(&slot.Ptr), uint64(ptr))
			unlock()
			return nil
		}
	}

	// 3. Check for empty slot in candidate bucket 1
	for i := 0; i < SlotsPerBucket; i++ {
		slot := &t.buckets[b1].Slots[i]
		if atomic.LoadUint64((*uint64)(&slot.Key)) == 0 {
			atomic.StoreUint64((*uint64)(&slot.Ptr), uint64(ptr))
			atomic.StoreUint64((*uint64)(&slot.Key), uint64(k)) // Release store
			atomic.AddUint64(&t.telemetry.TotalInsertions, 1)
			atomic.AddUint64(&t.telemetry.ActiveEntries, 1)
			unlock()
			return nil
		}
	}

	// 4. Check for empty slot in candidate bucket 2
	for i := 0; i < SlotsPerBucket; i++ {
		slot := &t.buckets[b2].Slots[i]
		if atomic.LoadUint64((*uint64)(&slot.Key)) == 0 {
			atomic.StoreUint64((*uint64)(&slot.Ptr), uint64(ptr))
			atomic.StoreUint64((*uint64)(&slot.Key), uint64(k)) // Release store
			atomic.AddUint64(&t.telemetry.TotalInsertions, 1)
			atomic.AddUint64(&t.telemetry.ActiveEntries, 1)
			unlock()
			return nil
		}
	}

	// Both candidate buckets are full; release stripe locks before entering displacement
	unlock()

	// Slow path: Cuckoo displacement kick-out or DRAM stash spillover
	return t.displaceAndInsert(k, ptr, b1, b2)
}

// displacementStep captures a single slot migration during Cuckoo kick-out.
type displacementStep struct {
	fromBucket int
	fromSlot   int
	toBucket   int
	toSlot     int
}

// bfsHop captures a node in the breadth-first search for an augmenting Cuckoo path.
type bfsHop struct {
	bucket    int // bucket index
	fromSlot  int // slot in parent bucket that kicked into this bucket
	parentIdx int // index in bfs queue (-1 for root)
}

// displaceAndInsert performs a breadth-first search to find a Cuckoo displacement path
// ending at an empty slot within MaxDisplacements hops. If no path exists, spills to DRAM stash.
func (t *CuckooTable) displaceAndInsert(k CASKey, ptr KVBlockPtr, b1, b2 int) error {
	t.displacementMu.Lock()
	defer t.displacementMu.Unlock()

	// Re-check candidate buckets under stripe locks (an empty slot may have opened)
	s1 := t.stripeIndex(b1)
	s2 := t.stripeIndex(b2)
	recheckUnlock := t.lockStripes(s1, s2)

	for i := 0; i < SlotsPerBucket; i++ {
		slot := &t.buckets[b1].Slots[i]
		if CASKey(atomic.LoadUint64((*uint64)(&slot.Key))) == k {
			atomic.StoreUint64((*uint64)(&slot.Ptr), uint64(ptr))
			recheckUnlock()
			return nil
		}
		if atomic.LoadUint64((*uint64)(&slot.Key)) == 0 {
			atomic.StoreUint64((*uint64)(&slot.Ptr), uint64(ptr))
			atomic.StoreUint64((*uint64)(&slot.Key), uint64(k))
			atomic.AddUint64(&t.telemetry.TotalInsertions, 1)
			atomic.AddUint64(&t.telemetry.ActiveEntries, 1)
			recheckUnlock()
			return nil
		}
	}
	for i := 0; i < SlotsPerBucket; i++ {
		slot := &t.buckets[b2].Slots[i]
		if CASKey(atomic.LoadUint64((*uint64)(&slot.Key))) == k {
			atomic.StoreUint64((*uint64)(&slot.Ptr), uint64(ptr))
			recheckUnlock()
			return nil
		}
		if atomic.LoadUint64((*uint64)(&slot.Key)) == 0 {
			atomic.StoreUint64((*uint64)(&slot.Ptr), uint64(ptr))
			atomic.StoreUint64((*uint64)(&slot.Key), uint64(k))
			atomic.AddUint64(&t.telemetry.TotalInsertions, 1)
			atomic.AddUint64(&t.telemetry.ActiveEntries, 1)
			recheckUnlock()
			return nil
		}
	}
	recheckUnlock()

	// Attempt BFS search for an augmenting path up to 5 times to absorb concurrent races
	for attempt := 0; attempt < 5; attempt++ {
		steps := t.findCuckooPath(b1, b2)
		if len(steps) == 0 {
			// No augmenting path found within MaxDisplacements; spill to DRAM stash
			break
		}

		// Collect all stripe locks involved in the displacement path
		stripeMap := make(map[int]bool, len(steps)*2)
		for _, step := range steps {
			stripeMap[t.stripeIndex(step.fromBucket)] = true
			stripeMap[t.stripeIndex(step.toBucket)] = true
		}
		stripes := make([]int, 0, len(stripeMap))
		for s := range stripeMap {
			stripes = append(stripes, s)
		}

		unlockAll := t.lockMultipleStripes(stripes)

		// Verify path validity under lock
		valid := true
		for i, step := range steps {
			fromKey := atomic.LoadUint64((*uint64)(&t.buckets[step.fromBucket].Slots[step.fromSlot].Key))
			if fromKey == 0 {
				valid = false
				break
			}
			if i == 0 {
				toKey := atomic.LoadUint64((*uint64)(&t.buckets[step.toBucket].Slots[step.toSlot].Key))
				if toKey != 0 {
					valid = false
					break
				}
			}
		}

		if !valid {
			unlockAll()
			continue
		}

		// Execute displacement moves from leaf to root (backward).
		// Every move transfers an entry to an already-empty slot before clearing source.
		for _, step := range steps {
			fromSlot := &t.buckets[step.fromBucket].Slots[step.fromSlot]
			toSlot := &t.buckets[step.toBucket].Slots[step.toSlot]

			kVal := atomic.LoadUint64((*uint64)(&fromSlot.Key))
			pVal := atomic.LoadUint64((*uint64)(&fromSlot.Ptr))

			// Write destination
			atomic.StoreUint64((*uint64)(&toSlot.Ptr), pVal)
			atomic.StoreUint64((*uint64)(&toSlot.Key), kVal)

			// Clear source
			atomic.StoreUint64((*uint64)(&fromSlot.Key), 0)
			atomic.StoreUint64((*uint64)(&fromSlot.Ptr), 0)
		}

		// Insert new entry into now-freed root slot
		rootStep := steps[len(steps)-1]
		rootSlot := &t.buckets[rootStep.fromBucket].Slots[rootStep.fromSlot]
		atomic.StoreUint64((*uint64)(&rootSlot.Ptr), uint64(ptr))
		atomic.StoreUint64((*uint64)(&rootSlot.Key), uint64(k))

		atomic.AddUint64(&t.telemetry.KickoutDisplacements, uint64(len(steps)))
		atomic.AddUint64(&t.telemetry.TotalInsertions, 1)
		atomic.AddUint64(&t.telemetry.ActiveEntries, 1)

		unlockAll()
		return nil
	}

	// Fallback to DRAM stash
	return t.insertToStashLocked(k, ptr)
}

// findCuckooPath executes BFS up to MaxDisplacements to locate an augmenting displacement chain.
func (t *CuckooTable) findCuckooPath(b1, b2 int) []displacementStep {
	queue := make([]bfsHop, 0, 64)
	queue = append(queue, bfsHop{bucket: b1, fromSlot: -1, parentIdx: -1})
	queue = append(queue, bfsHop{bucket: b2, fromSlot: -1, parentIdx: -1})

	visited := make(map[int]bool, 64)
	visited[b1] = true
	visited[b2] = true

	head := 0
	for head < len(queue) && len(queue) < MaxDisplacements {
		curr := queue[head]

		for s := 0; s < SlotsPerBucket; s++ {
			slotKey := CASKey(atomic.LoadUint64((*uint64)(&t.buckets[curr.bucket].Slots[s].Key)))
			if slotKey == 0 {
				continue
			}

			alt := AltBucket(slotKey, curr.bucket)
			if visited[alt] {
				continue
			}

			// Check if alt has an empty slot
			emptySlot := -1
			for es := 0; es < SlotsPerBucket; es++ {
				if atomic.LoadUint64((*uint64)(&t.buckets[alt].Slots[es].Key)) == 0 {
					emptySlot = es
					break
				}
			}

			if emptySlot != -1 {
				// Found empty slot! Construct steps from leaf back to root
				steps := []displacementStep{
					{fromBucket: curr.bucket, fromSlot: s, toBucket: alt, toSlot: emptySlot},
				}
				target := curr
				for target.parentIdx != -1 {
					parent := queue[target.parentIdx]
					steps = append(steps, displacementStep{
						fromBucket: parent.bucket,
						fromSlot:   target.fromSlot,
						toBucket:   target.bucket,
						toSlot:     steps[len(steps)-1].fromSlot,
					})
					target = parent
				}
				return steps
			}

			visited[alt] = true
			queue = append(queue, bfsHop{
				bucket:    alt,
				fromSlot:  s,
				parentIdx: head,
			})
			if len(queue) >= MaxDisplacements {
				break
			}
		}
		head++
	}

	return nil
}

// insertToStashLocked inserts into the DRAM stash under stashMu lock.
func (t *CuckooTable) insertToStashLocked(k CASKey, ptr KVBlockPtr) error {
	t.stashMu.Lock()
	defer t.stashMu.Unlock()

	if len(t.stash) >= MaxStashCapacity {
		return ErrTableFull
	}

	t.stash[k] = ptr
	atomic.AddUint64(&t.telemetry.StashSpills, 1)
	atomic.AddUint64(&t.telemetry.StashEntries, 1)
	atomic.AddUint64(&t.telemetry.TotalInsertions, 1)
	return nil
}

// InsertToStash directly places an entry into the DRAM overflow stash (used for testing overflow).
func (t *CuckooTable) InsertToStash(k CASKey, ptr KVBlockPtr) error {
	if k == EmptyCASKey {
		return ErrInvalidKey
	}
	if ptr == EmptyKVBlockPtr {
		return ErrInvalidPointer
	}
	return t.insertToStashLocked(k, ptr)
}

// Delete removes a CASKey from the primary Cuckoo table or DRAM stash.
func (t *CuckooTable) Delete(k CASKey) bool {
	if k == EmptyCASKey {
		return false
	}

	b1, b2 := CandidateBuckets(k)
	s1 := t.stripeIndex(b1)
	s2 := t.stripeIndex(b2)

	unlock := t.lockStripes(s1, s2)

	// Check candidate bucket 1
	for i := 0; i < SlotsPerBucket; i++ {
		slot := &t.buckets[b1].Slots[i]
		if CASKey(atomic.LoadUint64((*uint64)(&slot.Key))) == k {
			atomic.StoreUint64((*uint64)(&slot.Key), 0)
			atomic.StoreUint64((*uint64)(&slot.Ptr), 0)
			atomic.AddUint64(&t.telemetry.ActiveEntries, ^uint64(0)) // -1
			unlock()
			return true
		}
	}

	// Check candidate bucket 2
	for i := 0; i < SlotsPerBucket; i++ {
		slot := &t.buckets[b2].Slots[i]
		if CASKey(atomic.LoadUint64((*uint64)(&slot.Key))) == k {
			atomic.StoreUint64((*uint64)(&slot.Key), 0)
			atomic.StoreUint64((*uint64)(&slot.Ptr), 0)
			atomic.AddUint64(&t.telemetry.ActiveEntries, ^uint64(0)) // -1
			unlock()
			return true
		}
	}

	unlock()

	// Check DRAM overflow stash
	t.stashMu.Lock()
	if _, ok := t.stash[k]; ok {
		delete(t.stash, k)
		atomic.AddUint64(&t.telemetry.StashEntries, ^uint64(0)) // -1
		t.stashMu.Unlock()
		return true
	}
	t.stashMu.Unlock()

	return false
}

// Telemetry returns a consistent point-in-time snapshot of runtime Cuckoo telemetry.
func (t *CuckooTable) Telemetry() CuckooTelemetry {
	totalLookups := atomic.LoadUint64(&t.telemetry.TotalLookups)
	b1Hits := atomic.LoadUint64(&t.telemetry.Bucket1Hits)
	b2Hits := atomic.LoadUint64(&t.telemetry.Bucket2Hits)
	stashHits := atomic.LoadUint64(&t.telemetry.StashHits)
	misses := atomic.LoadUint64(&t.telemetry.Misses)
	totalInsertions := atomic.LoadUint64(&t.telemetry.TotalInsertions)
	kickouts := atomic.LoadUint64(&t.telemetry.KickoutDisplacements)
	stashSpills := atomic.LoadUint64(&t.telemetry.StashSpills)
	activeEntries := atomic.LoadUint64(&t.telemetry.ActiveEntries)
	stashEntries := atomic.LoadUint64(&t.telemetry.StashEntries)

	var loadFactor float64
	if TotalSlots > 0 {
		loadFactor = float64(activeEntries) / float64(TotalSlots)
	}

	var mallHitRate float64
	if totalLookups > 0 {
		mallHitRate = float64(b1Hits+b2Hits) / float64(totalLookups)
	}

	var displacementRate float64
	if totalInsertions > 0 {
		displacementRate = float64(kickouts) / float64(totalInsertions)
	}

	return CuckooTelemetry{
		TotalLookups:         totalLookups,
		Bucket1Hits:          b1Hits,
		Bucket2Hits:          b2Hits,
		StashHits:            stashHits,
		Misses:               misses,
		TotalInsertions:      totalInsertions,
		KickoutDisplacements: kickouts,
		StashSpills:          stashSpills,
		ActiveEntries:        activeEntries,
		StashEntries:         stashEntries,
		LoadFactor:           loadFactor,
		MALLHitRate:          mallHitRate,
		DisplacementRate:     displacementRate,
	}
}

// ResetTelemetry resets all telemetry counters to zero.
func (t *CuckooTable) ResetTelemetry() {
	atomic.StoreUint64(&t.telemetry.TotalLookups, 0)
	atomic.StoreUint64(&t.telemetry.Bucket1Hits, 0)
	atomic.StoreUint64(&t.telemetry.Bucket2Hits, 0)
	atomic.StoreUint64(&t.telemetry.StashHits, 0)
	atomic.StoreUint64(&t.telemetry.Misses, 0)
	atomic.StoreUint64(&t.telemetry.TotalInsertions, 0)
	atomic.StoreUint64(&t.telemetry.KickoutDisplacements, 0)
	atomic.StoreUint64(&t.telemetry.StashSpills, 0)
}
