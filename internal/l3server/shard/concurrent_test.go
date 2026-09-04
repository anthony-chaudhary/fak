package shard

import (
	"bytes"
	"fmt"
	"sync"
	"testing"
)

// TestConcurrentSetGetIntegrity runs 8 goroutines with disjoint key ranges,
// then verifies all values are correct. Designed for -race.
func TestConcurrentSetGetIntegrity(t *testing.T) {
	mgr, err := NewManager(ManagerConfig{
		NumShards:      4,
		MaxMemoryGB:    1,
		EvictionPolicy: "wtinylfu",
		IndexCapacity:  4096,
	})
	if err != nil {
		t.Fatal(err)
	}
	mgr.Start()
	t.Cleanup(mgr.Stop)

	const numGoroutines = 8
	const keysPerGoroutine = 100
	const valSize = 1024

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		g := g
		go func() {
			defer wg.Done()
			marker := byte(g)
			val := bytes.Repeat([]byte{marker}, valSize)
			for i := 0; i < keysPerGoroutine; i++ {
				key := []byte(fmt.Sprintf("g%d-%04d", g, i))
				mgr.Set(key, val, 0)
			}
		}()
	}
	wg.Wait()

	// Single-threaded verification
	corrupted := 0
	missing := 0
	for g := 0; g < numGoroutines; g++ {
		marker := byte(g)
		for i := 0; i < keysPerGoroutine; i++ {
			key := []byte(fmt.Sprintf("g%d-%04d", g, i))
			val, found := mgr.Get(key)
			if !found {
				missing++
				continue
			}
			if len(val) != valSize {
				t.Errorf("key %s: length %d, want %d", key, len(val), valSize)
				corrupted++
				continue
			}
			for j, b := range val {
				if b != marker {
					t.Errorf("key %s: byte[%d]=%d, want %d", key, j, b, marker)
					corrupted++
					break
				}
			}
		}
	}

	total := numGoroutines * keysPerGoroutine
	t.Logf("verified %d keys: missing=%d corrupted=%d", total, missing, corrupted)
	if missing > 0 {
		t.Errorf("%d keys missing (unexpected with 4096 index capacity)", missing)
	}
	if corrupted > 0 {
		t.Errorf("%d values corrupted under concurrent writes", corrupted)
	}
}

// TestConcurrentOverwriteLastWriterWins verifies that concurrent writes to
// the SAME key never produce a mixed/corrupted value.
func TestConcurrentOverwriteLastWriterWins(t *testing.T) {
	mgr, err := NewManager(ManagerConfig{
		NumShards:      4,
		MaxMemoryGB:    1,
		EvictionPolicy: "wtinylfu",
		IndexCapacity:  4096,
	})
	if err != nil {
		t.Fatal(err)
	}
	mgr.Start()
	t.Cleanup(mgr.Stop)

	const numGoroutines = 4
	const writesPerGoroutine = 50
	const valSize = 1024
	key := []byte("contested-key")

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		g := g
		go func() {
			defer wg.Done()
			val := bytes.Repeat([]byte{byte(g)}, valSize)
			for i := 0; i < writesPerGoroutine; i++ {
				mgr.Set(key, val, 0)
			}
		}()
	}
	wg.Wait()

	// The final value must be exactly valSize bytes of a single marker
	val, found := mgr.Get(key)
	if !found {
		t.Fatal("contested-key: not found after concurrent writes")
	}
	if len(val) != valSize {
		t.Fatalf("contested-key: length %d, want %d", len(val), valSize)
	}

	// Every byte must be the same (atomic write, no mixing)
	first := val[0]
	for j := 1; j < len(val); j++ {
		if val[j] != first {
			t.Fatalf("contested-key: byte[%d]=%d != byte[0]=%d â€” value is a mix of writes (CORRUPTION)", j, val[j], first)
		}
	}
	t.Logf("last writer was goroutine %d", first)
}

// TestSubmitPoolStress verifies that channel/timer pooling in Submit()
// never leaks a stale result to a subsequent op. 8 goroutines Ã— 10K ops,
// each verifying the value matches its unique key.
func TestSubmitPoolStress(t *testing.T) {
	mgr, err := NewManager(ManagerConfig{
		NumShards:      4,
		MaxMemoryGB:    1,
		EvictionPolicy: "wtinylfu",
		IndexCapacity:  131072,
	})
	if err != nil {
		t.Fatal(err)
	}
	mgr.Start()
	t.Cleanup(mgr.Stop)

	const numGoroutines = 8
	const opsPerGoroutine = 10000
	const valSize = 64

	var wg sync.WaitGroup
	wg.Add(numGoroutines)
	errors := make([]int, numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		g := g
		go func() {
			defer wg.Done()
			marker := byte(g)
			val := bytes.Repeat([]byte{marker}, valSize)
			for i := 0; i < opsPerGoroutine; i++ {
				key := []byte(fmt.Sprintf("pool-%d-%06d", g, i))
				mgr.Set(key, val, 0)
				got, found := mgr.Get(key)
				if !found {
					errors[g]++
					continue
				}
				if len(got) != valSize || got[0] != marker {
					errors[g]++
				}
			}
		}()
	}
	wg.Wait()

	total := 0
	for g := 0; g < numGoroutines; g++ {
		total += errors[g]
	}
	if total > 0 {
		t.Errorf("pool stress: %d ops returned wrong result (stale channel leak?)", total)
	}
	t.Logf("pool stress: %d goroutines Ã— %d ops = %d total, %d errors",
		numGoroutines, opsPerGoroutine, numGoroutines*opsPerGoroutine, total)
}

// TestSubmitTimeoutNoStaleResult verifies that a timed-out Submit never
// delivers a stale result to a subsequent caller via a recycled channel.
// This is a regression test for the channel pool cross-contamination bug
// where defer putResultChan(ch) on the timeout path returned a channel
// while the shard goroutine still held a reference.
func TestSubmitTimeoutNoStaleResult(t *testing.T) {
	mgr, err := NewManager(ManagerConfig{
		NumShards:         1,
		MaxMemoryGB:       1,
		EvictionPolicy:    "wtinylfu",
		IndexCapacity:     4096,
		DispatchTimeoutMs: 1, // 1ms timeout â€” forces rapid timeouts
	})
	if err != nil {
		t.Fatal(err)
	}
	mgr.Start()
	t.Cleanup(mgr.Stop)

	const numGoroutines = 8
	const opsPerGoroutine = 500

	var wg sync.WaitGroup
	wg.Add(numGoroutines)
	crossContaminated := make([]int, numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		g := g
		go func() {
			defer wg.Done()
			marker := byte(g + 1) // nonzero
			val := bytes.Repeat([]byte{marker}, 64)
			for i := 0; i < opsPerGoroutine; i++ {
				key := []byte(fmt.Sprintf("to-%d-%04d", g, i))
				// SET may or may not timeout
				mgr.Set(key, val, 0)

				// GET â€” the result (if any) must belong to this key
				got, found := mgr.Get(key)
				if !found {
					continue // timeout or evicted, ok
				}
				if len(got) != 64 {
					crossContaminated[g]++
					continue
				}
				// Every byte must be our marker
				for _, b := range got {
					if b != marker {
						crossContaminated[g]++
						break
					}
				}
			}
		}()
	}
	wg.Wait()

	total := 0
	for g := 0; g < numGoroutines; g++ {
		total += crossContaminated[g]
	}
	if total > 0 {
		t.Errorf("channel pool cross-contamination: %d ops received stale/wrong result", total)
	}
	t.Logf("timeout stress: %d goroutines Ã— %d ops, %d cross-contaminated",
		numGoroutines, opsPerGoroutine, total)
}

// TestConcurrentSetWithRebalance writes keys while a rebalance runs mid-flight
// and verifies all pre-rebalance keys survive.
func TestConcurrentSetWithRebalance(t *testing.T) {
	mgr, err := NewManager(ManagerConfig{
		NumShards:      1,
		MaxMemoryGB:    1,
		EvictionPolicy: "wtinylfu",
		IndexCapacity:  4096,
		WarmupOps:      5,
		AutoTuneSlabs:  true,
	})
	if err != nil {
		t.Fatal(err)
	}
	mgr.Start()
	t.Cleanup(mgr.Stop)

	const valSize = 4096
	type kv struct {
		key []byte
		val []byte
	}

	// Pre-rebalance entries (enough to trigger detection with warmup=5)
	preEntries := make([]kv, 10)
	for i := range preEntries {
		k := []byte(fmt.Sprintf("pre-%04d", i))
		v := bytes.Repeat([]byte{byte(i + 10)}, valSize)
		preEntries[i] = kv{key: k, val: v}
		mgr.Set(k, v, 0)
	}

	var wg sync.WaitGroup

	// Background writer
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 200; i++ {
			k := []byte(fmt.Sprintf("bg-%04d", i))
			v := bytes.Repeat([]byte{byte(i % 256)}, valSize)
			mgr.Set(k, v, 0)
		}
	}()

	// Trigger rebalance on shard 0
	s := mgr.Shard(0)
	result := s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})
	if result.Err != nil {
		t.Logf("rebalance during concurrent writes: %v (may be expected)", result.Err)
	}

	wg.Wait()

	// Verify pre-rebalance entries
	for _, e := range preEntries {
		val, found := mgr.Get(e.key)
		if !found {
			t.Errorf("GET %s: not found after concurrent rebalance", e.key)
			continue
		}
		if !bytes.Equal(val, e.val) {
			t.Errorf("GET %s: value corrupted after concurrent rebalance", e.key)
		}
	}
}

func TestShardCountCappedByMemory(t *testing.T) {
	const GB = 1 << 30
	tests := []struct {
		name       string
		totalMem   uint64
		numShards  int
		wantShards int
	}{
		{"512GB/64â†’64", 512 * GB, 64, 64},   // 8GB/shard = 8GB â†’ no cap (warns <16GB)
		{"512GB/32â†’32", 512 * GB, 32, 32},    // 16GB/shard = 16GB â†’ no cap
		{"256GB/32â†’32", 256 * GB, 32, 32},    // 8GB/shard = 8GB â†’ no cap (warns <16GB)
		{"128GB/16â†’16", 128 * GB, 16, 16},    // 8GB/shard = 8GB â†’ no cap (warns <16GB)
		{"32GB/8â†’4", 32 * GB, 8, 4},          // 4GB/shard < 8GB â†’ cap
		{"16GB/1â†’1", 16 * GB, 1, 1},          // 16GB/shard â†’ no cap
		{"8GB/4â†’1", 8 * GB, 4, 1},            // 2GB/shard < 8GB â†’ cap (total=8GB â†’ cap at 1)
		{"1GB/4â†’4", 1 * GB, 4, 4},            // total < 8GB â†’ skip
		{"1024GB/64â†’64", 1024 * GB, 64, 64},  // 16GB/shard â†’ no cap, no warning
		{"512GB/128â†’64", 512 * GB, 128, 64},  // 4GB/shard < 8GB â†’ cap
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := capShardsForMemory(tt.numShards, tt.totalMem)
			if got != tt.wantShards {
				t.Errorf("capShardsForMemory(%d, %dGB) = %d, want %d",
					tt.numShards, tt.totalMem/GB, got, tt.wantShards)
			}
		})
	}
}

// TestNewManagerDefaultShards verifies NumShards=0 defaults to 8.
func TestNewManagerDefaultShards(t *testing.T) {
	mgr, err := NewManager(ManagerConfig{
		NumShards:      0, // should default to 16
		MaxMemoryGB:    128,
		EvictionPolicy: "wtinylfu",
		IndexCapacity:  4096,
	})
	if err != nil {
		t.Fatal(err)
	}
	mgr.Start()
	t.Cleanup(mgr.Stop)
	if mgr.NumShards() != 16 {
		t.Errorf("NumShards=0 should default to 16, got %d", mgr.NumShards())
	}
}
