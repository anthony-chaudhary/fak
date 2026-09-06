package compute

import (
	"bytes"
	"fmt"
	"sync"
	"testing"
	"time"
)

type mockStorageSink struct {
	mu     sync.Mutex
	writes map[string][]byte
	order  []string
}

func newMockStorageSink() *mockStorageSink {
	return &mockStorageSink{
		writes: make(map[string][]byte),
	}
}

func (m *mockStorageSink) Sink(pageID string, data []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	cp := make([]byte, len(data))
	copy(cp, data)
	m.writes[pageID] = cp
	m.order = append(m.order, pageID)
	return nil
}

func (m *mockStorageSink) count() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.writes)
}

func (m *mockStorageSink) totalWrites() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.order)
}

func (m *mockStorageSink) get(pageID string) ([]byte, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	d, ok := m.writes[pageID]
	return d, ok
}

// TestDRAMStagingRingCoalesce proves that pages modified or invalidated (via Drop)
// within the cooldown period emit zero storage writes.
func TestDRAMStagingRingCoalesce(t *testing.T) {
	sink := newMockStorageSink()
	t0 := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	currTime := t0

	ring := NewDRAMStagingRing(DRAMStagingRingConfig{
		CapacityBytes: 10 * 1024 * 1024,
		Cooldown:      60 * time.Second,
		StorageSink:   sink.Sink,
		TimeSource:    func() time.Time { return currTime },
	})

	// Part 1: Write coalescing via repeated modifications within cooldown.
	dataV1 := []byte("intermediate-reasoning-step-v1")
	if err := ring.Put("page-coalesce", dataV1, true); err != nil {
		t.Fatalf("failed to Put v1: %v", err)
	}

	// Advance time by 30s (within 60s cooldown).
	currTime = t0.Add(30 * time.Second)
	flushed, err := ring.FlushExpired(currTime)
	if err != nil {
		t.Fatalf("FlushExpired error: %v", err)
	}
	if flushed != 0 {
		t.Fatalf("expected 0 flushed pages at 30s, got %d", flushed)
	}
	if sink.totalWrites() != 0 {
		t.Fatalf("expected 0 sink writes at 30s, got %d", sink.totalWrites())
	}

	// Modify page within cooldown (at 30s): reset dwell timer.
	dataV2 := []byte("coalesced-turn-output-v2")
	if err := ring.Put("page-coalesce", dataV2, true); err != nil {
		t.Fatalf("failed to Put v2: %v", err)
	}

	// Advance time to 70s from t0 (40s from v2 update; < 60s cooldown from last modification).
	currTime = t0.Add(70 * time.Second)
	flushed, err = ring.FlushExpired(currTime)
	if err != nil {
		t.Fatalf("FlushExpired error at 70s: %v", err)
	}
	if flushed != 0 {
		t.Fatalf("expected 0 flushed pages at 70s (coalesced), got %d", flushed)
	}
	if sink.totalWrites() != 0 {
		t.Fatalf("expected 0 sink writes at 70s, got %d", sink.totalWrites())
	}

	// Advance time to 100s from t0 (70s from v2 update; >= 60s cooldown).
	currTime = t0.Add(100 * time.Second)
	flushed, err = ring.FlushExpired(currTime)
	if err != nil {
		t.Fatalf("FlushExpired error at 100s: %v", err)
	}
	if flushed != 1 {
		t.Fatalf("expected 1 flushed page at 100s, got %d", flushed)
	}
	if sink.totalWrites() != 1 {
		t.Fatalf("expected exactly 1 coalesced sink write, got %d", sink.totalWrites())
	}
	writtenData, ok := sink.get("page-coalesce")
	if !ok || !bytes.Equal(writtenData, dataV2) {
		t.Fatalf("expected flushed data to be v2 %q, got %q", dataV2, writtenData)
	}

	// Part 2: Page invalidated via Drop within cooldown emits ZERO storage writes.
	dataDrop := []byte("ephemeral-scratchpad-to-drop")
	currTime = t0.Add(110 * time.Second)
	if err := ring.Put("page-drop", dataDrop, true); err != nil {
		t.Fatalf("failed to Put page-drop: %v", err)
	}

	// Verify page is resident in DRAM.
	gotData, found := ring.Get("page-drop")
	if !found || !bytes.Equal(gotData, dataDrop) {
		t.Fatalf("expected page-drop in DRAM, got found=%v, data=%q", found, gotData)
	}

	// Invalidate page within cooldown (at 130s).
	currTime = t0.Add(130 * time.Second)
	ring.Drop("page-drop")

	// Verify page is gone from DRAM.
	if _, found := ring.Get("page-drop"); found {
		t.Fatalf("expected page-drop to be dropped from DRAM")
	}

	// Advance time well past cooldown (e.g. 250s).
	currTime = t0.Add(250 * time.Second)
	flushed, err = ring.FlushExpired(currTime)
	if err != nil {
		t.Fatalf("FlushExpired error at 250s: %v", err)
	}
	if flushed != 0 {
		t.Fatalf("expected 0 flushed pages at 250s for dropped page, got %d", flushed)
	}

	// Verify storage sink never received any write for "page-drop".
	if _, found := sink.get("page-drop"); found {
		t.Fatalf("expected zero storage writes for dropped page, but found write in sink")
	}
	if sink.totalWrites() != 1 {
		t.Fatalf("expected total storage writes to remain 1, got %d", sink.totalWrites())
	}
}

// TestDRAMStagingRingCapacityEviction tests clean page eviction without writes,
// and dirty page write-back upon capacity pressure.
func TestDRAMStagingRingCapacityEviction(t *testing.T) {
	sink := newMockStorageSink()
	t0 := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	currTime := t0

	// 1000-byte capacity limit.
	ring := NewDRAMStagingRing(DRAMStagingRingConfig{
		CapacityBytes: 1000,
		Cooldown:      60 * time.Second,
		StorageSink:   sink.Sink,
		TimeSource:    func() time.Time { return currTime },
	})

	// 1. Clean pages capacity eviction (should emit ZERO storage writes).
	data400 := make([]byte, 400)
	for i := range data400 {
		data400[i] = byte(i)
	}

	if err := ring.Put("clean-1", data400, false); err != nil {
		t.Fatalf("Put clean-1 failed: %v", err)
	}
	if err := ring.Put("clean-2", data400, false); err != nil {
		t.Fatalf("Put clean-2 failed: %v", err)
	}
	if ring.UsedBytes() != 800 {
		t.Fatalf("expected 800 used bytes, got %d", ring.UsedBytes())
	}

	// Adding clean-3 (400 bytes) exceeds 1000 capacity (800+400=1200 > 1000).
	// clean-1 (oldest clean page) must be evicted.
	if err := ring.Put("clean-3", data400, false); err != nil {
		t.Fatalf("Put clean-3 failed: %v", err)
	}
	if _, found := ring.Get("clean-1"); found {
		t.Fatalf("expected clean-1 to be evicted")
	}
	if _, found := ring.Get("clean-2"); !found {
		t.Fatalf("expected clean-2 to remain resident")
	}
	if _, found := ring.Get("clean-3"); !found {
		t.Fatalf("expected clean-3 to remain resident")
	}
	if ring.UsedBytes() != 800 {
		t.Fatalf("expected 800 used bytes after eviction, got %d", ring.UsedBytes())
	}
	if sink.totalWrites() != 0 {
		t.Fatalf("expected 0 writes to sink for clean page eviction, got %d", sink.totalWrites())
	}

	// 2. Dirty pages capacity eviction (write-back before eviction).
	// Drop remaining clean pages.
	ring.Drop("clean-2")
	ring.Drop("clean-3")
	if ring.UsedBytes() != 0 {
		t.Fatalf("expected 0 used bytes after drop, got %d", ring.UsedBytes())
	}

	dirtyA := []byte("dirty-page-A-payload-400-bytes---------------------------------")
	dirtyA = append(dirtyA, make([]byte, 400-len(dirtyA))...)
	dirtyB := []byte("dirty-page-B-payload-400-bytes---------------------------------")
	dirtyB = append(dirtyB, make([]byte, 400-len(dirtyB))...)
	dirtyC := []byte("dirty-page-C-payload-400-bytes---------------------------------")
	dirtyC = append(dirtyC, make([]byte, 400-len(dirtyC))...)

	if err := ring.Put("dirty-A", dirtyA, true); err != nil {
		t.Fatalf("Put dirty-A failed: %v", err)
	}
	if err := ring.Put("dirty-B", dirtyB, true); err != nil {
		t.Fatalf("Put dirty-B failed: %v", err)
	}

	// Adding dirty-C (400 bytes) requires evicting dirty-A.
	// Since dirty-A is dirty, it must be flushed to the storage sink before eviction.
	if err := ring.Put("dirty-C", dirtyC, true); err != nil {
		t.Fatalf("Put dirty-C failed: %v", err)
	}

	if _, found := ring.Get("dirty-A"); found {
		t.Fatalf("expected dirty-A to be evicted from DRAM")
	}
	if _, found := ring.Get("dirty-B"); !found {
		t.Fatalf("expected dirty-B to remain in DRAM")
	}
	if _, found := ring.Get("dirty-C"); !found {
		t.Fatalf("expected dirty-C to remain in DRAM")
	}

	// Check that dirty-A was written back to sink upon eviction.
	writtenA, ok := sink.get("dirty-A")
	if !ok {
		t.Fatalf("expected dirty-A to be flushed to sink upon capacity eviction")
	}
	if !bytes.Equal(writtenA, dirtyA) {
		t.Fatalf("written dirty-A data mismatch")
	}
}

// TestDRAMStagingRingDirtyFlushExpired tests dirty page flushing upon cooldown expiration.
func TestDRAMStagingRingDirtyFlushExpired(t *testing.T) {
	sink := newMockStorageSink()
	t0 := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	currTime := t0

	ring := NewDRAMStagingRing(DRAMStagingRingConfig{
		CapacityBytes: 10 * 1024 * 1024,
		Cooldown:      90 * time.Second,
		StorageSink:   sink.Sink,
		TimeSource:    func() time.Time { return currTime },
	})

	data1 := []byte("page-1-data")
	data2 := []byte("page-2-data")
	cleanData := []byte("clean-data")

	if err := ring.Put("p1", data1, true); err != nil {
		t.Fatalf("Put p1 failed: %v", err)
	}
	if err := ring.Put("clean", cleanData, false); err != nil {
		t.Fatalf("Put clean failed: %v", err)
	}

	// Advance time to t0 + 30s.
	currTime = t0.Add(30 * time.Second)
	if err := ring.Put("p2", data2, true); err != nil {
		t.Fatalf("Put p2 failed: %v", err)
	}

	// At t0 + 60s:
	// p1 age: 60s < 90s cooldown -> not expired.
	// p2 age: 30s < 90s cooldown -> not expired.
	currTime = t0.Add(60 * time.Second)
	flushed, err := ring.FlushExpired(currTime)
	if err != nil {
		t.Fatalf("FlushExpired failed: %v", err)
	}
	if flushed != 0 {
		t.Fatalf("expected 0 flushed, got %d", flushed)
	}

	// At t0 + 95s:
	// p1 age: 95s >= 90s -> EXPIRED!
	// p2 age: 65s < 90s -> not expired.
	// clean -> never flushed.
	currTime = t0.Add(95 * time.Second)
	flushed, err = ring.FlushExpired(currTime)
	if err != nil {
		t.Fatalf("FlushExpired failed: %v", err)
	}
	if flushed != 1 {
		t.Fatalf("expected 1 flushed (p1), got %d", flushed)
	}
	if sink.totalWrites() != 1 {
		t.Fatalf("expected 1 write in sink, got %d", sink.totalWrites())
	}
	if _, ok := sink.get("p1"); !ok {
		t.Fatalf("expected p1 to be in sink")
	}

	// p1 is now clean in DRAM. It should still be retrievable.
	p1Got, found := ring.Get("p1")
	if !found || !bytes.Equal(p1Got, data1) {
		t.Fatalf("p1 should remain resident and clean in DRAM, got found=%v, data=%q", found, p1Got)
	}

	// At t0 + 125s:
	// p2 age: 95s >= 90s -> EXPIRED!
	// p1 is already clean -> not flushed again.
	currTime = t0.Add(125 * time.Second)
	flushed, err = ring.FlushExpired(currTime)
	if err != nil {
		t.Fatalf("FlushExpired failed: %v", err)
	}
	if flushed != 1 {
		t.Fatalf("expected 1 flushed (p2), got %d", flushed)
	}
	if sink.totalWrites() != 2 {
		t.Fatalf("expected 2 total writes in sink, got %d", sink.totalWrites())
	}
	if _, ok := sink.get("p2"); !ok {
		t.Fatalf("expected p2 to be in sink")
	}

	// Clean page was never flushed.
	if _, ok := sink.get("clean"); ok {
		t.Fatalf("clean page should never be flushed to storage sink")
	}
}

// TestDRAMStagingRingDefaults verifies default capacity (2 GiB) and cooldown (60s).
func TestDRAMStagingRingDefaults(t *testing.T) {
	ring := NewDRAMStagingRing(DRAMStagingRingConfig{})
	if ring.Capacity() != DefaultDRAMStagingCapacity {
		t.Fatalf("expected capacity %d, got %d", DefaultDRAMStagingCapacity, ring.Capacity())
	}
	if ring.Cooldown() != DefaultDRAMStagingCooldown {
		t.Fatalf("expected cooldown %v, got %v", DefaultDRAMStagingCooldown, ring.Cooldown())
	}
}

// TestDRAMStagingRingConcurrent validates thread safety under concurrent access.
func TestDRAMStagingRingConcurrent(t *testing.T) {
	sink := newMockStorageSink()
	ring := NewDRAMStagingRing(DRAMStagingRingConfig{
		CapacityBytes: 10 * 1024 * 1024,
		Cooldown:      10 * time.Millisecond,
		StorageSink:   sink.Sink,
	})

	var wg sync.WaitGroup
	workers := 8
	iterations := 100

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				pageID := fmt.Sprintf("worker-%d-page-%d", workerID, i%10)
				data := []byte(fmt.Sprintf("data-%d-%d", workerID, i))
				_ = ring.Put(pageID, data, i%2 == 0)
				_, _ = ring.Get(pageID)
				if i%5 == 0 {
					ring.Drop(pageID)
				}
				if i%10 == 0 {
					_, _ = ring.FlushExpired(time.Now())
				}
			}
		}(w)
	}

	wg.Wait()
}
