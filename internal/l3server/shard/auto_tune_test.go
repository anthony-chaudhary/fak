package shard

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

// TestAutoRebuildAfterDetection verifies that after warmup SETs, the allocator
// is automatically rebuilt and all entries are still readable.
func TestAutoRebuildAfterDetection(t *testing.T) {
	cfg := ShardConfig{
		ID:             0,
		IndexCapacity:  1024,
		MaxMemoryBytes: 64 * 1024 * 1024, // 64MB
		EvictionPolicy: "wtinylfu",
		WarmupOps:      10,
		AutoTuneSlabs:  true,
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

	// Insert 10 entries of 4096 bytes (uniform size)
	valSize := 4096
	for i := 0; i < 10; i++ {
		key := []byte(fmt.Sprintf("key-%04d", i))
		value := make([]byte, valSize)
		for j := range value {
			value[j] = byte(i)
		}
		result := s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 1), // non-zero
			Value:   value,
			Result:  make(chan OpResult, 1),
		})
		if result.Err != nil {
			t.Fatalf("SET key-%04d: %v", i, result.Err)
		}
	}

	// Verify detection occurred and ZeroLatencyBalance completed.
	// ZeroLatencyBalance is incremental â€” give the shard goroutine time to finish.
	var snap DetectionSnapshot
	for i := 0; i < 50; i++ {
		snap = s.SizeDetectionSnapshot()
		if snap.Status == "rebuilt" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if snap.Status != "rebuilt" {
		t.Errorf("expected status 'rebuilt', got %q", snap.Status)
	}
	if snap.DominantValueSize != uint64(valSize) {
		t.Errorf("expected dominant size %d, got %d", valSize, snap.DominantValueSize)
	}

	// Verify all entries are still readable
	for i := 0; i < 10; i++ {
		key := []byte(fmt.Sprintf("key-%04d", i))
		result := s.Submit(ShardOp{
			Type:    OpGet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Result:  make(chan OpResult, 1),
		})
		if !result.Found {
			t.Errorf("GET key-%04d: not found after rebuild", i)
			continue
		}
		if len(result.Value) != valSize {
			t.Errorf("GET key-%04d: expected value len %d, got %d", i, valSize, len(result.Value))
			continue
		}
		// Verify value contents
		for j, b := range result.Value {
			if b != byte(i) {
				t.Errorf("GET key-%04d: byte %d expected %d, got %d", i, j, byte(i), b)
				break
			}
		}
	}
}

// TestSnapshotThreadSafety verifies concurrent snapshot reads + shard SET writes are -race clean.
func TestSnapshotThreadSafety(t *testing.T) {
	cfg := ShardConfig{
		ID:             0,
		IndexCapacity:  1024,
		MaxMemoryBytes: 64 * 1024 * 1024,
		EvictionPolicy: "wtinylfu",
		WarmupOps:      100,
		AutoTuneSlabs:  true,
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

	var wg sync.WaitGroup

	// Goroutine 1: read snapshots continuously
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 500; i++ {
			snap := s.SizeDetectionSnapshot()
			_ = snap.Status // access fields to trigger race detector
			_ = snap.DominantValueSize
		}
	}()

	// Goroutine 2: read allocator continuously
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 500; i++ {
			a := s.Allocator()
			_ = a.NumClasses()
		}
	}()

	// Main: perform SETs to trigger detection
	for i := 0; i < 200; i++ {
		key := []byte(fmt.Sprintf("key-%04d", i))
		value := make([]byte, 1024)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Value:   value,
			Result:  make(chan OpResult, 1),
		})
	}

	wg.Wait()
}

// TestAllocatorAtomicSwap verifies concurrent Allocator() reads + FLUSH are -race clean.
func TestAllocatorAtomicSwap(t *testing.T) {
	cfg := ShardConfig{
		ID:             0,
		IndexCapacity:  1024,
		MaxMemoryBytes: 64 * 1024 * 1024,
		EvictionPolicy: "wtinylfu",
		WarmupOps:      10,
		AutoTuneSlabs:  true,
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

	// Insert enough to trigger detection
	for i := 0; i < 10; i++ {
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     []byte(fmt.Sprintf("k%d", i)),
			KeyHash: uint64(i + 1),
			Value:   make([]byte, 2048),
			Result:  make(chan OpResult, 1),
		})
	}

	var wg sync.WaitGroup

	// Concurrent reads of allocator
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 200; i++ {
			a := s.Allocator()
			_ = a.AllocatedBytes()
			_ = a.NumClasses()
		}
	}()

	// FLUSH triggers allocator swap
	s.Submit(ShardOp{
		Type:   OpFlush,
		Result: make(chan OpResult, 1),
	})

	wg.Wait()
}

// mockAllocListener records allocator change notifications.
type mockAllocListener struct {
	mu      sync.Mutex
	changes []AllocatorChange
}

func (m *mockAllocListener) OnAllocatorChanged(change AllocatorChange) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.changes = append(m.changes, change)
}

func (m *mockAllocListener) getChanges() []AllocatorChange {
	m.mu.Lock()
	defer m.mu.Unlock()
	result := make([]AllocatorChange, len(m.changes))
	copy(result, m.changes)
	return result
}

// TestListenerNotification verifies that listeners receive correct old/new allocators on rebuild.
func TestListenerNotification(t *testing.T) {
	cfg := ShardConfig{
		ID:             0,
		IndexCapacity:  1024,
		MaxMemoryBytes: 64 * 1024 * 1024,
		EvictionPolicy: "wtinylfu",
		WarmupOps:      5,
		AutoTuneSlabs:  true,
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	listener := &mockAllocListener{}
	s.RegisterAllocListener(listener)

	origAlloc := s.Allocator()
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Insert enough to trigger auto-rebuild
	for i := 0; i < 5; i++ {
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     []byte(fmt.Sprintf("k%d", i)),
			KeyHash: uint64(i + 1),
			Value:   make([]byte, 8192),
			Result:  make(chan OpResult, 1),
		})
	}

	// Insert one more SET to trigger the rebuild (detection happens at warmup, rebuild at next SET)
	s.Submit(ShardOp{
		Type:    OpSet,
		Key:     []byte("trigger"),
		KeyHash: 999,
		Value:   make([]byte, 8192),
		Result:  make(chan OpResult, 1),
	})

	// Wait for async allocator construction + migration to complete
	var changes []AllocatorChange
	for i := 0; i < 100; i++ {
		changes = listener.getChanges()
		if len(changes) > 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if len(changes) == 0 {
		t.Fatal("expected at least one allocator change notification")
	}

	change := changes[0]
	if change.ShardID != 0 {
		t.Errorf("expected ShardID=0, got %d", change.ShardID)
	}
	if change.OldAllocator != origAlloc {
		t.Error("OldAllocator should match the original allocator")
	}
	if change.NewAllocator == nil {
		t.Error("NewAllocator should not be nil")
	}
	if change.NewAllocator == origAlloc {
		t.Error("NewAllocator should differ from old")
	}

	// Don't close old allocator â€” the listener owns it (in production, RDMA deferred cleanup)
	change.OldAllocator.Close()
}

// TestMigrationPreservesData inserts known keys, triggers rebuild, then verifies values.
func TestMigrationPreservesData(t *testing.T) {
	cfg := ShardConfig{
		ID:             0,
		IndexCapacity:  1024,
		MaxMemoryBytes: 64 * 1024 * 1024,
		EvictionPolicy: "wtinylfu",
		WarmupOps:      5,
		AutoTuneSlabs:  true,
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

	// Store entries with distinct values
	type kv struct {
		key     []byte
		keyHash uint64
		value   []byte
	}
	entries := make([]kv, 5)
	for i := range entries {
		k := []byte(fmt.Sprintf("mig-%04d", i))
		v := make([]byte, 2048)
		for j := range v {
			v[j] = byte(i + 42)
		}
		entries[i] = kv{key: k, keyHash: uint64(i + 100), value: v}
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

	// One more SET to trigger rebuild
	s.Submit(ShardOp{
		Type:    OpSet,
		Key:     []byte("extra"),
		KeyHash: 999,
		Value:   make([]byte, 2048),
		Result:  make(chan OpResult, 1),
	})

	// Verify all original entries are intact
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
		if len(result.Value) != len(e.value) {
			t.Errorf("GET %s: value length %d, want %d", e.key, len(result.Value), len(e.value))
			continue
		}
		for j := range e.value {
			if result.Value[j] != e.value[j] {
				t.Errorf("GET %s: byte %d = %d, want %d", e.key, j, result.Value[j], e.value[j])
				break
			}
		}
	}
}

// TestFlushResetsDetection verifies that after rebuild, FLUSH clears frozen so re-detection works.
func TestFlushResetsDetection(t *testing.T) {
	cfg := ShardConfig{
		ID:             0,
		IndexCapacity:  1024,
		MaxMemoryBytes: 64 * 1024 * 1024,
		EvictionPolicy: "wtinylfu",
		WarmupOps:      5,
		AutoTuneSlabs:  true,
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

	// Trigger detection + rebuild
	for i := 0; i < 6; i++ {
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     []byte(fmt.Sprintf("k%d", i)),
			KeyHash: uint64(i + 1),
			Value:   make([]byte, 4096),
			Result:  make(chan OpResult, 1),
		})
	}

	// Wait for async allocator construction + migration to complete
	var snap DetectionSnapshot
	for i := 0; i < 100; i++ {
		snap = s.SizeDetectionSnapshot()
		if snap.Status == "rebuilt" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if snap.Status != "rebuilt" {
		t.Fatalf("expected 'rebuilt', got %q", snap.Status)
	}

	// FLUSH should reset detection
	result := s.Submit(ShardOp{
		Type:   OpFlush,
		Result: make(chan OpResult, 1),
	})
	if result.Err != nil {
		t.Fatalf("FLUSH: %v", result.Err)
	}

	snap = s.SizeDetectionSnapshot()
	if snap.Status != "warming_up" {
		t.Errorf("expected 'warming_up' after FLUSH, got %q", snap.Status)
	}

	// Re-detect with different size
	for i := 0; i < 6; i++ {
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     []byte(fmt.Sprintf("k%d", i)),
			KeyHash: uint64(i + 1),
			Value:   make([]byte, 8192),
			Result:  make(chan OpResult, 1),
		})
	}

	// Wait for async allocator construction + migration
	for i := 0; i < 100; i++ {
		snap = s.SizeDetectionSnapshot()
		if snap.Status == "rebuilt" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if snap.Status != "rebuilt" {
		t.Errorf("expected re-detection to 'rebuilt', got %q", snap.Status)
	}
	if snap.DominantValueSize != 8192 {
		t.Errorf("expected dominant size 8192, got %d", snap.DominantValueSize)
	}
}
