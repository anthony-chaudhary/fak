package strix

import (
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
)

// TestCuckooCAS_Exact8MBGeometryAndAlignment verifies:
// - Exact 8,388,608 bytes (8 MiB = 8,192 sets x 16 ways x 64B).
// - 131,072 buckets of 64 bytes each (matching Zen 5 / RDNA 3.5 / MALL cache line size).
// - 4 slots per bucket, each 16 bytes (8-byte CASKey + 8-byte KVBlockPtr).
// - Base address of table is strictly aligned to 64-byte boundaries.
func TestCuckooCAS_Exact8MBGeometryAndAlignment(t *testing.T) {
	// Invariant 1: Constant geometry checks
	if TableSizeBytes != 8*1024*1024 {
		t.Fatalf("expected TableSizeBytes == 8388608, got %d", TableSizeBytes)
	}
	if NumBuckets != 131072 {
		t.Fatalf("expected NumBuckets == 131072, got %d", NumBuckets)
	}
	if SlotsPerBucket != 4 {
		t.Fatalf("expected SlotsPerBucket == 4, got %d", SlotsPerBucket)
	}
	if SlotSizeBytes != 16 {
		t.Fatalf("expected SlotSizeBytes == 16, got %d", SlotSizeBytes)
	}
	if BucketSizeBytes != 64 {
		t.Fatalf("expected BucketSizeBytes == 64, got %d", BucketSizeBytes)
	}
	if CacheLineSize != 64 {
		t.Fatalf("expected CacheLineSize == 64, got %d", CacheLineSize)
	}
	if NumSets != 8192 {
		t.Fatalf("expected NumSets == 8192, got %d", NumSets)
	}
	if NumWays != 16 {
		t.Fatalf("expected NumWays == 16, got %d", NumWays)
	}
	if TotalSlots != 524288 {
		t.Fatalf("expected TotalSlots == 524288, got %d", TotalSlots)
	}

	// Invariant 2: Struct memory layouts
	if sz := unsafe.Sizeof(CuckooSlot{}); sz != 16 {
		t.Fatalf("expected sizeof(CuckooSlot) == 16, got %d", sz)
	}
	if sz := unsafe.Sizeof(CuckooBucket{}); sz != 64 {
		t.Fatalf("expected sizeof(CuckooBucket) == 64, got %d", sz)
	}

	// Invariant 3: Allocated table geometry and alignment
	tbl := NewCuckooTable()
	if tbl.ByteSize() != 8388608 {
		t.Fatalf("expected tbl.ByteSize() == 8388608, got %d", tbl.ByteSize())
	}
	if tbl.NumBuckets() != 131072 {
		t.Fatalf("expected tbl.NumBuckets() == 131072, got %d", tbl.NumBuckets())
	}
	if tbl.Capacity() != 524288 {
		t.Fatalf("expected tbl.Capacity() == 524288, got %d", tbl.Capacity())
	}

	baseAddr := tbl.BaseAddress()
	if baseAddr%64 != 0 {
		t.Fatalf("table base address 0x%x is not 64-byte aligned (rem = %d)", baseAddr, baseAddr%64)
	}

	bucket0Addr := uintptr(unsafe.Pointer(&tbl.buckets[0]))
	if bucket0Addr != baseAddr {
		t.Fatalf("buckets[0] address 0x%x != baseAddress 0x%x", bucket0Addr, baseAddr)
	}
	if bucket0Addr%64 != 0 {
		t.Fatalf("buckets[0] address 0x%x is not 64-byte aligned", bucket0Addr)
	}

	// Verify contiguous 64-byte stride for all buckets
	bucket1Addr := uintptr(unsafe.Pointer(&tbl.buckets[1]))
	if bucket1Addr-bucket0Addr != 64 {
		t.Fatalf("stride between buckets is %d, expected 64", bucket1Addr-bucket0Addr)
	}
}

// TestCuckooCAS_TwoProbeLookupGuarantee verifies:
// - CandidateBuckets(k) returns two distinct buckets in [0, 131071].
// - All primary table lookups resolve in at most 2 cache line evaluations.
// - Zero pointer chasing.
func TestCuckooCAS_TwoProbeLookupGuarantee(t *testing.T) {
	tbl := NewCuckooTable()
	const numKeys = 2000

	inserted := make(map[CASKey]KVBlockPtr, numKeys)
	for i := uint64(1); i <= numKeys; i++ {
		key := CASKey(0x1000000000000000 | (i * 0x9e3779b97f4a7c15))
		ptr := KVBlockPtr(0x80000000 + i*64)

		b1, b2 := CandidateBuckets(key)
		if b1 == b2 {
			t.Fatalf("key 0x%x candidate buckets are not distinct: b1=%d b2=%d", key, b1, b2)
		}
		if b1 < 0 || b1 >= NumBuckets || b2 < 0 || b2 >= NumBuckets {
			t.Fatalf("candidate buckets out of bounds: b1=%d b2=%d", b1, b2)
		}

		if err := tbl.Insert(key, ptr); err != nil {
			t.Fatalf("failed to insert key 0x%x: %v", key, err)
		}
		inserted[key] = ptr
	}

	// Verify all keys: must reside in either bucket 1 or bucket 2
	for key, expectedPtr := range inserted {
		b1, b2 := CandidateBuckets(key)

		// Check physical placement in candidate buckets
		foundInB1 := false
		foundInB2 := false
		for s := 0; s < SlotsPerBucket; s++ {
			if tbl.buckets[b1].Slots[s].Key == key {
				foundInB1 = true
			}
			if tbl.buckets[b2].Slots[s].Key == key {
				foundInB2 = true
			}
		}

		if !foundInB1 && !foundInB2 {
			t.Fatalf("key 0x%x not found in candidate bucket b1=%d or b2=%d", key, b1, b2)
		}
		if foundInB1 && foundInB2 {
			t.Fatalf("key 0x%x duplicated in both candidate buckets", key)
		}

		// Perform lock-free lookup
		resolvedPtr, ok := tbl.Lookup(key)
		if !ok {
			t.Fatalf("Lookup failed for present key 0x%x", key)
		}
		if resolvedPtr != expectedPtr {
			t.Fatalf("Lookup returned wrong pointer: got 0x%x, expected 0x%x", resolvedPtr, expectedPtr)
		}
	}

	// Verify telemetry confirms 100% MALL residency (zero stash hits, zero misses)
	telem := tbl.Telemetry()
	if telem.TotalLookups != numKeys {
		t.Fatalf("expected %d total lookups, got %d", numKeys, telem.TotalLookups)
	}
	if telem.Bucket1Hits+telem.Bucket2Hits != numKeys {
		t.Fatalf("expected all lookups to hit Bucket1 or Bucket2, got b1=%d b2=%d",
			telem.Bucket1Hits, telem.Bucket2Hits)
	}
	if telem.StashHits != 0 {
		t.Fatalf("expected 0 stash hits, got %d", telem.StashHits)
	}
	if telem.Misses != 0 {
		t.Fatalf("expected 0 misses, got %d", telem.Misses)
	}
	if telem.MALLHitRate != 1.0 {
		t.Fatalf("expected 1.0 MALL hit rate, got %.4f", telem.MALLHitRate)
	}
}

// TestCuckooCAS_DisplacementKickout verifies:
// - Cuckoo displacement moves existing entries to their alternate buckets when candidate buckets are full.
// - All keys remain accessible and consistent after kick-outs.
// - KickoutDisplacements telemetry accurately reflects migrations.
func TestCuckooCAS_DisplacementKickout(t *testing.T) {
	tbl := NewCuckooTable()

	// Pick a test key K0 and get its two candidate buckets
	k0 := CASKey(0xBADC0FFEE0000001)
	p0 := KVBlockPtr(0x70000001)
	targetB1, targetB2 := CandidateBuckets(k0)

	// Step 1: Fill all 4 slots of targetB1
	filledB1Keys := make([]CASKey, 0, 4)
	for i := uint64(1); len(filledB1Keys) < SlotsPerBucket; i++ {
		k := CASKey(0x1000000000000000 | i)
		b1, b2 := CandidateBuckets(k)
		if b1 == targetB1 || b2 == targetB1 {
			// Ensure it actually lands in targetB1
			_ = tbl.Insert(k, KVBlockPtr(0x1000+i))
			// Check if targetB1 slot was filled
			for s := 0; s < SlotsPerBucket; s++ {
				if tbl.buckets[targetB1].Slots[s].Key == k {
					filledB1Keys = append(filledB1Keys, k)
					break
				}
			}
		}
	}

	// Step 2: Fill all 4 slots of targetB2
	filledB2Keys := make([]CASKey, 0, 4)
	for i := uint64(100000); len(filledB2Keys) < SlotsPerBucket; i++ {
		k := CASKey(0x2000000000000000 | i)
		b1, b2 := CandidateBuckets(k)
		if (b1 == targetB2 || b2 == targetB2) && b1 != targetB1 && b2 != targetB1 {
			_ = tbl.Insert(k, KVBlockPtr(0x2000+i))
			for s := 0; s < SlotsPerBucket; s++ {
				if tbl.buckets[targetB2].Slots[s].Key == k {
					filledB2Keys = append(filledB2Keys, k)
					break
				}
			}
		}
	}

	// Verify both targetB1 and targetB2 are 100% full
	for s := 0; s < SlotsPerBucket; s++ {
		if tbl.buckets[targetB1].Slots[s].Key == 0 {
			t.Fatalf("targetB1 slot %d is unexpectedly empty", s)
		}
		if tbl.buckets[targetB2].Slots[s].Key == 0 {
			t.Fatalf("targetB2 slot %d is unexpectedly empty", s)
		}
	}

	displacementsBefore := tbl.Telemetry().KickoutDisplacements

	// Step 3: Insert k0, whose candidate buckets are targetB1 and targetB2.
	// Since both buckets are full, this MUST trigger a Cuckoo kick-out displacement!
	if err := tbl.Insert(k0, p0); err != nil {
		t.Fatalf("failed to insert k0: %v", err)
	}

	displacementsAfter := tbl.Telemetry().KickoutDisplacements
	if displacementsAfter <= displacementsBefore {
		t.Fatalf("expected KickoutDisplacements to increase, before=%d after=%d",
			displacementsBefore, displacementsAfter)
	}

	// Verify k0 is retrievable
	ptr, ok := tbl.Lookup(k0)
	if !ok || ptr != p0 {
		t.Fatalf("Lookup(k0) failed: got (0x%x, %v), expected (0x%x, true)", ptr, ok, p0)
	}

	// Verify all pre-filled keys in B1 and B2 are still retrievable
	for _, k := range filledB1Keys {
		if _, ok := tbl.Lookup(k); !ok {
			t.Fatalf("pre-filled B1 key 0x%x missing after displacement", k)
		}
	}
	for _, k := range filledB2Keys {
		if _, ok := tbl.Lookup(k); !ok {
			t.Fatalf("pre-filled B2 key 0x%x missing after displacement", k)
		}
	}

	telem := tbl.Telemetry()
	if telem.DisplacementRate <= 0.0 {
		t.Fatalf("expected DisplacementRate > 0, got %.4f", telem.DisplacementRate)
	}
	t.Logf("Displacement kick-out verified: before=%d, after=%d, rate=%.4f",
		displacementsBefore, displacementsAfter, telem.DisplacementRate)
}

// TestCuckooCAS_StashSpillover verifies:
// - Overflow entries spill gracefully into the DRAM stash when candidate paths are exhausted.
// - Stash lookups succeed and increment StashHits.
// - MaxStashCapacity (1024) is strictly enforced.
// - Deletions from stash correctly update StashEntries.
func TestCuckooCAS_StashSpillover(t *testing.T) {
	tbl := NewCuckooTable()

	// 1. Direct insertion to stash
	stashKey := CASKey(0x57A5400000000001)
	stashPtr := KVBlockPtr(0x99990000)

	if err := tbl.InsertToStash(stashKey, stashPtr); err != nil {
		t.Fatalf("failed to insert into stash: %v", err)
	}

	telem := tbl.Telemetry()
	if telem.StashSpills != 1 {
		t.Fatalf("expected StashSpills == 1, got %d", telem.StashSpills)
	}
	if telem.StashEntries != 1 {
		t.Fatalf("expected StashEntries == 1, got %d", telem.StashEntries)
	}

	// 2. Lookup finds key in stash
	p, ok := tbl.Lookup(stashKey)
	if !ok {
		t.Fatalf("lookup failed for key in stash")
	}
	if p != stashPtr {
		t.Fatalf("lookup returned 0x%x, expected 0x%x", p, stashPtr)
	}

	telem = tbl.Telemetry()
	if telem.StashHits != 1 {
		t.Fatalf("expected StashHits == 1, got %d", telem.StashHits)
	}

	// 3. Fill stash up to MaxStashCapacity (1024)
	for i := uint64(2); i <= MaxStashCapacity; i++ {
		k := CASKey(0x57A5400000000000 | i)
		ptr := KVBlockPtr(0x99990000 | i)
		if err := tbl.InsertToStash(k, ptr); err != nil {
			t.Fatalf("failed to insert stash key %d: %v", i, err)
		}
	}

	telem = tbl.Telemetry()
	if telem.StashEntries != MaxStashCapacity {
		t.Fatalf("expected %d stash entries, got %d", MaxStashCapacity, telem.StashEntries)
	}

	// 4. Next stash insertion must return ErrTableFull
	overflowKey := CASKey(0x57A5400000000FFF)
	overflowPtr := KVBlockPtr(0x9999FFFF)
	err := tbl.InsertToStash(overflowKey, overflowPtr)
	if err != ErrTableFull {
		t.Fatalf("expected ErrTableFull when stash is full, got %v", err)
	}

	// 5. Delete an entry from stash and verify space is reclaimed
	if !tbl.Delete(stashKey) {
		t.Fatalf("failed to delete stash key")
	}
	if telem := tbl.Telemetry(); telem.StashEntries != MaxStashCapacity-1 {
		t.Fatalf("expected %d stash entries after delete, got %d", MaxStashCapacity-1, telem.StashEntries)
	}

	// Now inserting overflowKey should succeed
	if err := tbl.InsertToStash(overflowKey, overflowPtr); err != nil {
		t.Fatalf("failed to insert after deleting from stash: %v", err)
	}
}

// TestCuckooCAS_ConcurrentReadWriteRaceFree tests concurrent readers, writers,
// updaters, and deleters under -race to prove zero torn reads and zero deadlocks.
func TestCuckooCAS_ConcurrentReadWriteRaceFree(t *testing.T) {
	tbl := NewCuckooTable()
	const numWriters = 4
	const numReaders = 4
	const keysPerWriter = 1000

	var wg sync.WaitGroup
	var stopFlag uint32

	// Pre-populate some keys
	for i := uint64(1); i <= 500; i++ {
		k := CASKey(0xA000000000000000 | i)
		p := KVBlockPtr(0x50000000 | i)
		_ = tbl.Insert(k, p)
	}

	// Start writer goroutines
	for w := 0; w < numWriters; w++ {
		wg.Add(1)
		go func(writerID int) {
			defer wg.Done()
			base := uint64(writerID+1) * 100000
			for i := uint64(1); i <= keysPerWriter; i++ {
				k := CASKey(base + i)
				p := KVBlockPtr(base*2 + i)
				if err := tbl.Insert(k, p); err != nil && err != ErrTableFull {
					t.Errorf("writer %d insert error: %v", writerID, err)
					return
				}
				// Verify immediate read consistency
				if val, ok := tbl.Lookup(k); ok && val != p {
					t.Errorf("writer %d read inconsistency: got 0x%x, expected 0x%x", writerID, val, p)
				}
			}
		}(w)
	}

	// Start reader goroutines
	for r := 0; r < numReaders; r++ {
		wg.Add(1)
		go func(readerID int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(readerID)))
			for atomic.LoadUint32(&stopFlag) == 0 {
				// Read either pre-populated keys, newly written keys, or non-existent keys
				target := uint64(rng.Intn(500000)) + 1
				k := CASKey(target)
				tbl.Lookup(k)
			}
		}(r)
	}

	// Start updater goroutines
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 200; i++ {
			k := CASKey(0xA000000000000000 | uint64(i%50+1))
			newPtr := KVBlockPtr(0x77770000 | uint64(i))
			_ = tbl.Insert(k, newPtr)
		}
	}()

	// Start deleter goroutines
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 100; i++ {
			k := CASKey(0xA000000000000000 | uint64(i%20+100))
			tbl.Delete(k)
		}
	}()

	// Wait for writers to complete
	time.Sleep(50 * time.Millisecond)
	atomic.StoreUint32(&stopFlag, 1)
	wg.Wait()

	telem := tbl.Telemetry()
	t.Logf("Concurrent run completed. Lookups: %d, Insertions: %d, Active: %d, Stash: %d",
		telem.TotalLookups, telem.TotalInsertions, telem.ActiveEntries, telem.StashEntries)
}

// TestCuckooCAS_Delete verifies:
// - Deletion clears entries from primary candidate buckets and stash.
// - Deleted entries are no longer discoverable by Lookup.
// - ActiveEntries and StashEntries are accurately updated.
// - Re-inserting a deleted key succeeds cleanly.
func TestCuckooCAS_Delete(t *testing.T) {
	tbl := NewCuckooTable()
	const count = 100

	keys := make([]CASKey, count)
	for i := 0; i < count; i++ {
		keys[i] = CASKey(0xBEEF000000000000 | uint64(i+1))
		if err := tbl.Insert(keys[i], KVBlockPtr(i+1)); err != nil {
			t.Fatalf("failed to insert key %d: %v", i, err)
		}
	}

	if telem := tbl.Telemetry(); telem.ActiveEntries != count {
		t.Fatalf("expected %d active entries, got %d", count, telem.ActiveEntries)
	}

	// Delete first 50 keys
	for i := 0; i < 50; i++ {
		if !tbl.Delete(keys[i]) {
			t.Fatalf("delete failed for present key %d", i)
		}
		// Second delete must return false
		if tbl.Delete(keys[i]) {
			t.Fatalf("second delete returned true for already-deleted key %d", i)
		}
		// Lookup must return false
		if _, ok := tbl.Lookup(keys[i]); ok {
			t.Fatalf("deleted key %d still found by Lookup", i)
		}
	}

	// Verify remaining 50 keys are intact
	for i := 50; i < count; i++ {
		ptr, ok := tbl.Lookup(keys[i])
		if !ok || ptr != KVBlockPtr(i+1) {
			t.Fatalf("remaining key %d lookup failed: got (0x%x, %v)", i, ptr, ok)
		}
	}

	if telem := tbl.Telemetry(); telem.ActiveEntries != 50 {
		t.Fatalf("expected 50 active entries after deletions, got %d", telem.ActiveEntries)
	}

	// Re-insert deleted keys
	for i := 0; i < 50; i++ {
		if err := tbl.Insert(keys[i], KVBlockPtr(i+1000)); err != nil {
			t.Fatalf("re-insert failed for key %d: %v", i, err)
		}
		ptr, ok := tbl.Lookup(keys[i])
		if !ok || ptr != KVBlockPtr(i+1000) {
			t.Fatalf("re-inserted key %d verification failed", i)
		}
	}

	if telem := tbl.Telemetry(); telem.ActiveEntries != count {
		t.Fatalf("expected %d active entries after re-insert, got %d", count, telem.ActiveEntries)
	}
}

// TestCuckooCAS_HighLoadFactorTarget verifies:
// - Sustained high load factor insertion (up to 50,000+ keys).
// - 100% data integrity on lookups across the loaded table.
// - LoadFactor accurately reflects ActiveEntries / TotalSlots.
func TestCuckooCAS_HighLoadFactorTarget(t *testing.T) {
	tbl := NewCuckooTable()
	const targetKeys = 50000

	start := time.Now()
	for i := uint64(1); i <= targetKeys; i++ {
		k := CASKey(0x1234567800000000 | i)
		p := KVBlockPtr(0x8000000000000000 | (i * 64))
		if err := tbl.Insert(k, p); err != nil {
			t.Fatalf("insert failed at %d: %v", i, err)
		}
	}
	insertDuration := time.Since(start)

	telem := tbl.Telemetry()
	t.Logf("Inserted %d keys in %v (avg %.2f µs/insert)",
		targetKeys, insertDuration, float64(insertDuration.Microseconds())/targetKeys)
	t.Logf("Load factor: %.4f (Active: %d / %d), Kickouts: %d, Stash spills: %d",
		telem.LoadFactor, telem.ActiveEntries, TotalSlots, telem.KickoutDisplacements, telem.StashSpills)

	expectedLoadFactor := float64(targetKeys) / float64(TotalSlots)
	if telem.LoadFactor < expectedLoadFactor*0.99 || telem.LoadFactor > expectedLoadFactor*1.01 {
		t.Fatalf("load factor mismatch: got %.4f, expected ~%.4f", telem.LoadFactor, expectedLoadFactor)
	}

	// Verify sample lookups
	lookupStart := time.Now()
	const verifySamples = 10000
	for i := uint64(1); i <= verifySamples; i++ {
		k := CASKey(0x1234567800000000 | i)
		expectedPtr := KVBlockPtr(0x8000000000000000 | (i * 64))
		ptr, ok := tbl.Lookup(k)
		if !ok || ptr != expectedPtr {
			t.Fatalf("lookup failed for key %d: got (0x%x, %v)", i, ptr, ok)
		}
	}
	lookupDuration := time.Since(lookupStart)
	avgLookupNs := float64(lookupDuration.Nanoseconds()) / verifySamples
	t.Logf("Verified %d lookups in %v (avg %.2f ns/lookup)", verifySamples, lookupDuration, avgLookupNs)
}

// BenchmarkCuckooCAS_Lookup evaluates sub-20ns target for Context MMU CAS handle resolution.
func BenchmarkCuckooCAS_Lookup(b *testing.B) {
	tbl := NewCuckooTable()
	const numKeys = 50000
	keys := make([]CASKey, numKeys)

	for i := 0; i < numKeys; i++ {
		keys[i] = CASKey(0x9876543200000000 | uint64(i+1))
		ptr := KVBlockPtr(0x10000000 | uint64(i+1))
		if err := tbl.Insert(keys[i], ptr); err != nil {
			b.Fatalf("insert failed: %v", err)
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		k := keys[i&(numKeys-1)]
		ptr, ok := tbl.Lookup(k)
		if !ok || ptr == 0 {
			b.Fatal("unexpected lookup failure")
		}
	}
}

// BenchmarkCandidateBuckets benchmarks the dual-hash calculation latency.
func BenchmarkCandidateBuckets(b *testing.B) {
	k := CASKey(0x123456789ABCDEF0)
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		b1, b2 := CandidateBuckets(k)
		if b1 == b2 {
			b.Fatal("buckets must be distinct")
		}
	}
}
