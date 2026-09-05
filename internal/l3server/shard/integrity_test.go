package shard

import (
	"bytes"
	"fmt"
	"testing"
	"time"
)

// TestEvictionPressureIntegrity fills a shard past capacity and verifies that
// surviving values are byte-for-byte correct â€” eviction must not corrupt neighbors.
func TestEvictionPressureIntegrity(t *testing.T) {
	cfg := ShardConfig{
		ID:             0,
		IndexCapacity:  256,
		MaxMemoryBytes: 16 * 1024 * 1024, // 16MB
		EvictionPolicy: "wtinylfu",
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// maxKeys = IndexCapacity * 7/8 = 224
	total := 400
	valSize := 4096
	type entry struct {
		key     []byte
		keyHash uint64
		marker  byte
	}
	entries := make([]entry, total)
	for i := 0; i < total; i++ {
		k := []byte(fmt.Sprintf("ep-%04d", i))
		marker := byte(i % 256)
		v := bytes.Repeat([]byte{marker}, valSize)
		entries[i] = entry{key: k, keyHash: uint64(i + 1), marker: marker}
		result := s.Submit(ShardOp{
			Type:    OpSet,
			Key:     k,
			KeyHash: uint64(i + 1),
			Value:   v,
			Result:  make(chan OpResult, 1),
		})
		if result.Err != nil {
			t.Fatalf("SET %s: %v", k, result.Err)
		}
	}

	found, missing, corrupted := 0, 0, 0
	for _, e := range entries {
		result := s.Submit(ShardOp{
			Type:    OpGet,
			Key:     e.key,
			KeyHash: e.keyHash,
			Result:  make(chan OpResult, 1),
		})
		if !result.Found {
			missing++
			continue
		}
		found++
		if len(result.Value) != valSize {
			t.Errorf("GET %s: length %d, want %d", e.key, len(result.Value), valSize)
			corrupted++
			continue
		}
		for j, b := range result.Value {
			if b != e.marker {
				t.Errorf("GET %s: byte[%d]=%d, want %d (CORRUPTION)", e.key, j, b, e.marker)
				corrupted++
				break
			}
		}
	}

	t.Logf("found=%d missing=%d corrupted=%d total=%d", found, missing, corrupted, total)
	if found+missing != total {
		t.Errorf("found+missing=%d, want %d", found+missing, total)
	}
	if found == 0 {
		t.Error("no keys found â€” shard is broken")
	}
	if missing == 0 {
		t.Error("no evictions happened â€” increase total or decrease IndexCapacity")
	}
	if corrupted > 0 {
		t.Errorf("%d values corrupted after eviction", corrupted)
	}
}

// TestSetOverwriteIntegrity verifies SET-overwrite returns the new value and
// handles size changes (grow + shrink) without corruption.
func TestSetOverwriteIntegrity(t *testing.T) {
	s := newTestShard(t, 0, 0, 0, false)
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	key := []byte("overwrite-key")
	keyHash := uint64(42)

	tests := []struct {
		name string
		size int
	}{
		{"initial 1KB", 1024},
		{"grow to 4KB", 4096},
		{"shrink to 256B", 256},
		{"same size 256B", 256},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			marker := byte(tc.size % 256)
			val := bytes.Repeat([]byte{marker}, tc.size)

			result := s.Submit(ShardOp{
				Type:    OpSet,
				Key:     key,
				KeyHash: keyHash,
				Value:   val,
				Result:  make(chan OpResult, 1),
			})
			if result.Err != nil {
				t.Fatalf("SET: %v", result.Err)
			}

			result = s.Submit(ShardOp{
				Type:    OpGet,
				Key:     key,
				KeyHash: keyHash,
				Result:  make(chan OpResult, 1),
			})
			if !result.Found {
				t.Fatal("GET: not found after SET")
			}
			if len(result.Value) != tc.size {
				t.Fatalf("GET: length %d, want %d", len(result.Value), tc.size)
			}
			for j, b := range result.Value {
				if b != marker {
					t.Fatalf("GET: byte[%d]=%d, want %d", j, b, marker)
				}
			}
		})
	}
}

// TestFlushClearsAllKeys verifies that OpFlush removes all keys and
// IndexCount drops to zero.
func TestFlushClearsAllKeys(t *testing.T) {
	s := newTestShard(t, 0, 0, 0, false)
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	numKeys := 100
	for i := 0; i < numKeys; i++ {
		k := []byte(fmt.Sprintf("flush-%04d", i))
		v := bytes.Repeat([]byte{byte(i)}, 512)
		result := s.Submit(ShardOp{
			Type:    OpSet,
			Key:     k,
			KeyHash: uint64(i + 1),
			Value:   v,
			Result:  make(chan OpResult, 1),
		})
		if result.Err != nil {
			t.Fatalf("SET %s: %v", k, result.Err)
		}
	}

	// Flush
	result := s.Submit(ShardOp{
		Type:   OpFlush,
		Result: make(chan OpResult, 1),
	})
	if result.Err != nil {
		t.Fatalf("FLUSH: %v", result.Err)
	}

	// Verify all keys are gone
	for i := 0; i < numKeys; i++ {
		k := []byte(fmt.Sprintf("flush-%04d", i))
		result := s.Submit(ShardOp{
			Type:    OpGet,
			Key:     k,
			KeyHash: uint64(i + 1),
			Result:  make(chan OpResult, 1),
		})
		if result.Found {
			t.Errorf("GET %s: found after FLUSH", k)
		}
	}

	if count := s.IndexCount(); count != 0 {
		t.Errorf("IndexCount after FLUSH: %d, want 0", count)
	}
}

// TestTTLValueIntegrity verifies TTL expiration doesn't corrupt neighboring entries.
func TestTTLValueIntegrity(t *testing.T) {
	s := newTestShard(t, 0, 0, 0, false)
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Key with long TTL â€” should survive
	longVal := bytes.Repeat([]byte{0xAA}, 1024)
	s.Submit(ShardOp{
		Type:    OpSet,
		Key:     []byte("long-ttl"),
		KeyHash: 1,
		Value:   longVal,
		TTLMs:   60000,
		Result:  make(chan OpResult, 1),
	})

	// Key with no TTL â€” should survive
	noTTLVal := bytes.Repeat([]byte{0xBB}, 1024)
	s.Submit(ShardOp{
		Type:    OpSet,
		Key:     []byte("no-ttl"),
		KeyHash: 2,
		Value:   noTTLVal,
		Result:  make(chan OpResult, 1),
	})

	// Key with very short TTL â€” should expire
	shortVal := bytes.Repeat([]byte{0xCC}, 1024)
	s.Submit(ShardOp{
		Type:    OpSet,
		Key:     []byte("short-ttl"),
		KeyHash: 3,
		Value:   shortVal,
		TTLMs:   1,
		Result:  make(chan OpResult, 1),
	})

	// Immediate GET should find short-ttl (maybe)
	result := s.Submit(ShardOp{
		Type:    OpGet,
		Key:     []byte("short-ttl"),
		KeyHash: 3,
		Result:  make(chan OpResult, 1),
	})
	if result.Found {
		// If found, value must be correct
		for j, b := range result.Value {
			if b != 0xCC {
				t.Fatalf("short-ttl byte[%d]=%d, want 0xCC", j, b)
			}
		}
	}

	// Wait for expiry
	time.Sleep(20 * time.Millisecond)

	// short-ttl should be gone
	result = s.Submit(ShardOp{
		Type:    OpGet,
		Key:     []byte("short-ttl"),
		KeyHash: 3,
		Result:  make(chan OpResult, 1),
	})
	if result.Found {
		t.Error("short-ttl: still found after expiry")
	}

	// Verify long-ttl is intact
	result = s.Submit(ShardOp{
		Type:    OpGet,
		Key:     []byte("long-ttl"),
		KeyHash: 1,
		Result:  make(chan OpResult, 1),
	})
	if !result.Found {
		t.Fatal("long-ttl: not found")
	}
	if !bytes.Equal(result.Value, longVal) {
		t.Error("long-ttl: value corrupted after neighbor expiry")
	}

	// Verify no-ttl is intact
	result = s.Submit(ShardOp{
		Type:    OpGet,
		Key:     []byte("no-ttl"),
		KeyHash: 2,
		Result:  make(chan OpResult, 1),
	})
	if !result.Found {
		t.Fatal("no-ttl: not found")
	}
	if !bytes.Equal(result.Value, noTTLVal) {
		t.Error("no-ttl: value corrupted after neighbor expiry")
	}
}

// TestMultiRebalanceIntegrity verifies that two consecutive rebalances
// don't lose or corrupt data.
func TestMultiRebalanceIntegrity(t *testing.T) {
	// Need auto-tune + warmup for rebalance to work
	s := newTestShard(t, 0, 0, 3, true)
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	valSize := 4096
	type kv struct {
		key     []byte
		keyHash uint64
		value   []byte
	}

	// Insert first batch (enough to trigger detection)
	batch1 := make([]kv, 5)
	for i := range batch1 {
		k := []byte(fmt.Sprintf("mr-a-%04d", i))
		v := bytes.Repeat([]byte{byte(i + 10)}, valSize)
		batch1[i] = kv{key: k, keyHash: uint64(i + 1), value: v}
		result := s.Submit(ShardOp{
			Type:    OpSet,
			Key:     k,
			KeyHash: uint64(i + 1),
			Value:   v,
			Result:  make(chan OpResult, 1),
		})
		if result.Err != nil {
			t.Fatalf("SET %s: %v", k, result.Err)
		}
	}

	// First rebalance
	result := s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})
	if result.Err != nil {
		t.Fatalf("rebalance 1: %v", result.Err)
	}

	// Insert second batch
	batch2 := make([]kv, 5)
	for i := range batch2 {
		k := []byte(fmt.Sprintf("mr-b-%04d", i))
		v := bytes.Repeat([]byte{byte(i + 50)}, valSize)
		batch2[i] = kv{key: k, keyHash: uint64(i + 100), value: v}
		result := s.Submit(ShardOp{
			Type:    OpSet,
			Key:     k,
			KeyHash: uint64(i + 100),
			Value:   v,
			Result:  make(chan OpResult, 1),
		})
		if result.Err != nil {
			t.Fatalf("SET %s: %v", k, result.Err)
		}
	}

	// Need enough ops for re-detection before second rebalance
	for i := 0; i < 5; i++ {
		s.Submit(ShardOp{
			Type:    OpGet,
			Key:     batch1[0].key,
			KeyHash: batch1[0].keyHash,
			Result:  make(chan OpResult, 1),
		})
	}

	// Second rebalance
	result = s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})
	if result.Err != nil {
		// Rebalance may fail if detection hasn't completed after reset â€” acceptable
		t.Logf("rebalance 2: %v (may be expected if re-warmup incomplete)", result.Err)
	}

	// Verify all entries from both batches
	all := append(batch1, batch2...)
	for _, e := range all {
		result := s.Submit(ShardOp{
			Type:    OpGet,
			Key:     e.key,
			KeyHash: e.keyHash,
			Result:  make(chan OpResult, 1),
		})
		if !result.Found {
			t.Errorf("GET %s: not found after double rebalance", e.key)
			continue
		}
		if !bytes.Equal(result.Value, e.value) {
			t.Errorf("GET %s: value corrupted after double rebalance", e.key)
		}
	}
}

// TestEvictionDoesNotCorruptNeighbors inserts keys A, B, C, pins B,
// then triggers evictions and verifies B is intact.
func TestEvictionDoesNotCorruptNeighbors(t *testing.T) {
	cfg := ShardConfig{
		ID:             0,
		IndexCapacity:  256,
		MaxMemoryBytes: 16 * 1024 * 1024,
		EvictionPolicy: "wtinylfu",
		MaxLeaseDurMs:  60000,
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	valSize := 4096
	pinnedKey := []byte("pinned-B")
	pinnedHash := uint64(999)
	pinnedVal := bytes.Repeat([]byte{0xBB}, valSize)

	// Insert A, B, C
	for _, e := range []struct {
		key  []byte
		hash uint64
		val  byte
	}{
		{[]byte("neighbor-A"), 998, 0xAA},
		{pinnedKey, pinnedHash, 0xBB},
		{[]byte("neighbor-C"), 1000, 0xCC},
	} {
		v := bytes.Repeat([]byte{e.val}, valSize)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     e.key,
			KeyHash: e.hash,
			Value:   v,
			Result:  make(chan OpResult, 1),
		})
	}

	// Pin key B
	s.Submit(ShardOp{
		Type:    OpPin,
		Key:     pinnedKey,
		KeyHash: pinnedHash,
		Result:  make(chan OpResult, 1),
	})

	// Fill shard to trigger evictions
	for i := 0; i < 400; i++ {
		k := []byte(fmt.Sprintf("filler-%04d", i))
		v := bytes.Repeat([]byte{byte(i % 256)}, valSize)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     k,
			KeyHash: uint64(i + 2000),
			Value:   v,
			Result:  make(chan OpResult, 1),
		})
	}

	// Verify pinned key B
	result := s.Submit(ShardOp{
		Type:    OpGet,
		Key:     pinnedKey,
		KeyHash: pinnedHash,
		Result:  make(chan OpResult, 1),
	})
	if !result.Found {
		t.Fatal("pinned key B: not found after eviction pressure")
	}
	if !bytes.Equal(result.Value, pinnedVal) {
		t.Error("pinned key B: value corrupted after eviction pressure")
	}
}
