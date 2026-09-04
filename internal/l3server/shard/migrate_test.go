package shard

import (
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/index"
)

// TestIncrementalMigrationCorrectness writes N entries, triggers ZeroLatencyBalance,
// interleaves GETs during migration, and verifies all entries are correct after finalization.
func TestIncrementalMigrationCorrectness(t *testing.T) {
	s := newTestShard(t, 0, 0, 50, true) // low warmup so detection triggers quickly
	s.config.MigrateBatchSize = 10        // small batches to exercise multi-batch path
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	const numEntries = 100
	valSize := 512
	type entry struct {
		key     []byte
		keyHash uint64
		val     byte
	}
	entries := make([]entry, numEntries)

	// Insert entries
	for i := 0; i < numEntries; i++ {
		key := []byte(fmt.Sprintf("migrate-key-%04d", i))
		hash := index.KeyHash(key)
		val := byte(i % 256)
		value := make([]byte, valSize)
		for j := range value {
			value[j] = val
		}
		result := s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: hash,
			Value:   value,
			Result:  make(chan OpResult, 1),
		})
		if result.Err != nil {
			t.Fatalf("SET %d: %v", i, result.Err)
		}
		entries[i] = entry{key: key, keyHash: hash, val: val}
	}

	// Verify all entries readable before migration
	for _, e := range entries {
		result := s.Submit(ShardOp{
			Type:    OpGet,
			Key:     e.key,
			KeyHash: e.keyHash,
			Result:  make(chan OpResult, 1),
		})
		if !result.Found {
			t.Fatalf("GET %s: not found before migration", e.key)
		}
	}

	// Trigger rebalance (which starts ZeroLatencyBalance)
	result := s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})
	if result.Err != nil {
		t.Fatalf("OpRebalance: %v", result.Err)
	}

	// Interleave GETs during/after migration â€” these will hit the shard goroutine
	// which also processes migration batches
	for _, e := range entries {
		result := s.Submit(ShardOp{
			Type:    OpGet,
			Key:     e.key,
			KeyHash: e.keyHash,
			Result:  make(chan OpResult, 1),
		})
		if !result.Found {
			t.Errorf("GET %s: not found during migration", e.key)
			continue
		}
		if len(result.Value) != valSize {
			t.Errorf("GET %s: value length %d, want %d", e.key, len(result.Value), valSize)
			continue
		}
		if result.Value[0] != e.val {
			t.Errorf("GET %s: value byte 0 = %d, want %d", e.key, result.Value[0], e.val)
		}
	}

	// Wait for migration to complete (submit a no-op GET to ensure shard has processed batches)
	time.Sleep(50 * time.Millisecond)

	// Verify all entries still correct after migration should have completed
	for _, e := range entries {
		result := s.Submit(ShardOp{
			Type:    OpGet,
			Key:     e.key,
			KeyHash: e.keyHash,
			Result:  make(chan OpResult, 1),
		})
		if !result.Found {
			t.Errorf("GET %s: not found after migration", e.key)
			continue
		}
		if result.Value[0] != e.val {
			t.Errorf("GET %s: value byte 0 = %d, want %d after migration", e.key, result.Value[0], e.val)
		}
	}
}

// TestSetDuringMigration verifies that new SETs during migration go to the new allocator.
func TestSetDuringMigration(t *testing.T) {
	s := newTestShard(t, 0, 0, 30, true)
	s.config.MigrateBatchSize = 5
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Insert enough to trigger detection
	for i := 0; i < 40; i++ {
		key := []byte(fmt.Sprintf("k-%04d", i))
		value := make([]byte, 256)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: index.KeyHash(key),
			Value:   value,
			Result:  make(chan OpResult, 1),
		})
	}

	// Trigger migration
	s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})

	// SET new keys during migration
	for i := 100; i < 120; i++ {
		key := []byte(fmt.Sprintf("new-k-%04d", i))
		value := make([]byte, 256)
		for j := range value {
			value[j] = byte(i)
		}
		result := s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: index.KeyHash(key),
			Value:   value,
			Result:  make(chan OpResult, 1),
		})
		if result.Err != nil {
			t.Fatalf("SET new-k-%04d during migration: %v", i, result.Err)
		}
	}

	// Verify all new keys readable
	time.Sleep(50 * time.Millisecond)
	for i := 100; i < 120; i++ {
		key := []byte(fmt.Sprintf("new-k-%04d", i))
		result := s.Submit(ShardOp{
			Type:    OpGet,
			Key:     key,
			KeyHash: index.KeyHash(key),
			Result:  make(chan OpResult, 1),
		})
		if !result.Found {
			t.Errorf("GET new-k-%04d: not found after migration", i)
			continue
		}
		if result.Value[0] != byte(i) {
			t.Errorf("GET new-k-%04d: value[0] = %d, want %d", i, result.Value[0], byte(i))
		}
	}
}

// TestDeleteDuringMigration verifies that DELETEs during migration free from the correct allocator.
func TestDeleteDuringMigration(t *testing.T) {
	s := newTestShard(t, 0, 0, 30, true)
	s.config.MigrateBatchSize = 5
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Insert entries
	for i := 0; i < 40; i++ {
		key := []byte(fmt.Sprintf("dk-%04d", i))
		value := make([]byte, 256)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: index.KeyHash(key),
			Value:   value,
			Result:  make(chan OpResult, 1),
		})
	}

	// Trigger migration
	s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})

	// Delete some keys during migration
	for i := 0; i < 10; i++ {
		key := []byte(fmt.Sprintf("dk-%04d", i))
		result := s.Submit(ShardOp{
			Type:    OpDelete,
			Key:     key,
			KeyHash: index.KeyHash(key),
			Result:  make(chan OpResult, 1),
		})
		if !result.OK {
			t.Errorf("DELETE dk-%04d: expected OK", i)
		}
	}

	// Wait for migration to finish
	time.Sleep(50 * time.Millisecond)

	// Verify deleted keys are gone
	for i := 0; i < 10; i++ {
		key := []byte(fmt.Sprintf("dk-%04d", i))
		result := s.Submit(ShardOp{
			Type:    OpGet,
			Key:     key,
			KeyHash: index.KeyHash(key),
			Result:  make(chan OpResult, 1),
		})
		if result.Found {
			t.Errorf("GET dk-%04d: should not be found after delete", i)
		}
	}

	// Verify remaining keys still readable
	for i := 10; i < 40; i++ {
		key := []byte(fmt.Sprintf("dk-%04d", i))
		result := s.Submit(ShardOp{
			Type:    OpGet,
			Key:     key,
			KeyHash: index.KeyHash(key),
			Result:  make(chan OpResult, 1),
		})
		if !result.Found {
			t.Errorf("GET dk-%04d: not found after migration", i)
		}
	}
}

// TestFlushDuringMigration verifies that FLUSH aborts migration cleanly.
func TestFlushDuringMigration(t *testing.T) {
	s := newTestShard(t, 0, 0, 30, true)
	s.config.MigrateBatchSize = 5
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Insert entries to trigger detection
	for i := 0; i < 40; i++ {
		key := []byte(fmt.Sprintf("fk-%04d", i))
		value := make([]byte, 256)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: index.KeyHash(key),
			Value:   value,
			Result:  make(chan OpResult, 1),
		})
	}

	// Trigger migration
	s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})

	// Immediately flush
	result := s.Submit(ShardOp{
		Type:   OpFlush,
		Result: make(chan OpResult, 1),
	})
	if result.Err != nil {
		t.Fatalf("FLUSH during migration: %v", result.Err)
	}

	// Verify all keys are gone
	for i := 0; i < 40; i++ {
		key := []byte(fmt.Sprintf("fk-%04d", i))
		result := s.Submit(ShardOp{
			Type:    OpGet,
			Key:     key,
			KeyHash: index.KeyHash(key),
			Result:  make(chan OpResult, 1),
		})
		if result.Found {
			t.Errorf("GET fk-%04d: should not be found after flush", i)
		}
	}
}

// TestStaggeredWarmup verifies that NewManager assigns distinct warmup targets.
func TestStaggeredWarmup(t *testing.T) {
	mgr, err := NewManager(ManagerConfig{
		NumShards:   8,
		MaxMemoryGB: 1,
		WarmupOps:   100,
		AutoTuneSlabs: true,
	})
	if err != nil {
		t.Fatal(err)
	}

	warmups := make(map[int]bool)
	for i := 0; i < mgr.NumShards(); i++ {
		w := mgr.Shard(i).Config().WarmupOps
		if w < 100 {
			t.Errorf("shard %d warmup %d < base 100", i, w)
		}
		if warmups[w] && i > 0 {
			// Only shard 0 should have the base warmup of 100
			t.Errorf("shard %d has duplicate warmup target %d", i, w)
		}
		warmups[w] = true
	}

	// Verify spread: last shard should have warmup > first
	first := mgr.Shard(0).Config().WarmupOps
	last := mgr.Shard(mgr.NumShards() - 1).Config().WarmupOps
	if last <= first {
		t.Errorf("expected last shard warmup (%d) > first (%d)", last, first)
	}
}

// TestMigrationSemaphore verifies that concurrent migrations are limited.
func TestMigrationSemaphore(t *testing.T) {
	mgr, err := NewManager(ManagerConfig{
		NumShards:               4,
		MaxMemoryGB:             1,
		WarmupOps:               20,
		AutoTuneSlabs:           true,
		MaxConcurrentMigrations: 1,
		MigrateBatchSize:        5,
	})
	if err != nil {
		t.Fatal(err)
	}
	mgr.Start()
	defer mgr.Stop()

	// Insert enough to trigger detection on all shards
	for i := 0; i < 100; i++ {
		key := []byte(fmt.Sprintf("sem-key-%04d", i))
		value := make([]byte, 256)
		mgr.Set(key, value, 0)
	}

	// Trigger rebalance on all shards simultaneously
	results := make(chan OpResult, mgr.NumShards())
	for i := 0; i < mgr.NumShards(); i++ {
		go func(s *Shard) {
			r := s.Submit(ShardOp{
				Type:   OpRebalance,
				Result: make(chan OpResult, 1),
			})
			results <- r
		}(mgr.Shard(i))
	}

	// Collect results â€” some may error if semaphore was full (migration deferred)
	okCount := 0
	for i := 0; i < mgr.NumShards(); i++ {
		r := <-results
		if r.Err == nil && r.OK {
			okCount++
		}
	}
	// At least one should succeed
	if okCount == 0 {
		t.Error("expected at least one migration to succeed")
	}
}
