package gateway_test

import (
	"context"
	"errors"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	directio "github.com/anthony-chaudhary/fak/internal/kv/direct_io"
)

// mockKVEvictor tracks eviction calls and simulates block freeing.
type mockKVEvictor struct {
	evictCount atomic.Int64
	freedCount atomic.Int64
	failErr    error
}

func (m *mockKVEvictor) EvictKV(ctx context.Context) (int, error) {
	if m.failErr != nil {
		return 0, m.failErr
	}
	m.evictCount.Add(1)
	freed := int(m.freedCount.Add(8))
	_ = freed
	return 8, nil
}

// TestKVTierIdleGatedEvictionBlocksWhileBusy satisfies requirement 4:
// Verifying idle-gated eviction blocks while busy and drains cleanly when idle.
func TestKVTierIdleGatedEvictionBlocksWhileBusy(t *testing.T) {
	tracker := gateway.NewActiveRequestTracker()
	mock := &mockKVEvictor{}
	evictor := gateway.NewIdleGatedEvictor(tracker, mock)

	ctx := context.Background()

	// 1. Simulate server being busy (active requests = 2)
	done1 := tracker.Track()
	defer done1()
	done2 := tracker.Track()

	if tracker.IsIdle() {
		t.Fatalf("expected tracker to be busy")
	}
	if tracker.ActiveRequests() != 2 {
		t.Fatalf("active requests = %d; want 2", tracker.ActiveRequests())
	}

	// 2. Attempt eviction while busy — must be blocked!
	res, err := evictor.TryEvict(ctx)
	if !errors.Is(err, gateway.ErrEvictionBlockedBusy) {
		t.Fatalf("TryEvict while busy: got err %v; want ErrEvictionBlockedBusy", err)
	}
	if res.Outcome != gateway.EvictionOutcomeBlocked {
		t.Fatalf("res.Outcome = %q; want %q", res.Outcome, gateway.EvictionOutcomeBlocked)
	}
	if res.FreedBlocks != 0 {
		t.Fatalf("res.FreedBlocks = %d; want 0", res.FreedBlocks)
	}
	if mock.evictCount.Load() != 0 {
		t.Fatalf("mock evictor called %d times; want 0 (mid-turn race prevention)", mock.evictCount.Load())
	}

	// 3. DrainWhenIdle with timeout while busy should time out
	resWait, errWait := evictor.DrainWhenIdle(ctx, 20*time.Millisecond)
	if !errors.Is(errWait, gateway.ErrEvictionBlockedBusy) {
		t.Fatalf("DrainWhenIdle while busy: got err %v; want ErrEvictionBlockedBusy", errWait)
	}
	if resWait.Outcome != gateway.EvictionOutcomeBlocked {
		t.Fatalf("resWait.Outcome = %q; want %q", resWait.Outcome, gateway.EvictionOutcomeBlocked)
	}

	// 4. Release requests so server becomes idle
	done2()
	done1()

	if !tracker.IsIdle() {
		t.Fatalf("expected tracker to be idle after releasing requests")
	}

	// 5. TryEvict now that server is idle — must drain cleanly!
	resIdle, errIdle := evictor.TryEvict(ctx)
	if errIdle != nil {
		t.Fatalf("TryEvict while idle: unexpected err %v", errIdle)
	}
	if resIdle.Outcome != gateway.EvictionOutcomeDrained {
		t.Fatalf("resIdle.Outcome = %q; want %q", resIdle.Outcome, gateway.EvictionOutcomeDrained)
	}
	if resIdle.FreedBlocks != 8 {
		t.Fatalf("resIdle.FreedBlocks = %d; want 8", resIdle.FreedBlocks)
	}
	if mock.evictCount.Load() != 1 {
		t.Fatalf("mock evictor called %d times; want 1", mock.evictCount.Load())
	}

	// Verify stats
	stats := evictor.Stats()
	if stats.BlockedAttempts < 2 {
		t.Errorf("stats.BlockedAttempts = %d; want >= 2", stats.BlockedAttempts)
	}
	if stats.SuccessfulDrains != 1 {
		t.Errorf("stats.SuccessfulDrains = %d; want 1", stats.SuccessfulDrains)
	}
	if stats.TotalFreedBlocks != 8 {
		t.Errorf("stats.TotalFreedBlocks = %d; want 8", stats.TotalFreedBlocks)
	}
}

// TestKVTierIdleGatedEvictionDrainsCleanlyWhenIdle verifies asynchronous transition from busy to idle.
func TestKVTierIdleGatedEvictionDrainsCleanlyWhenIdle(t *testing.T) {
	tracker := gateway.NewActiveRequestTracker()
	mock := &mockKVEvictor{}
	evictor := gateway.NewIdleGatedEvictor(tracker, mock)

	// Start in busy state
	done := tracker.Track()

	// Launch background worker that finishes request after 30ms
	go func() {
		time.Sleep(30 * time.Millisecond)
		done()
	}()

	ctx := context.Background()
	start := time.Now()
	res, err := evictor.DrainWhenIdle(ctx, 500*time.Millisecond)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("DrainWhenIdle failed: %v", err)
	}
	if res.Outcome != gateway.EvictionOutcomeDrained {
		t.Fatalf("res.Outcome = %q; want %q", res.Outcome, gateway.EvictionOutcomeDrained)
	}
	if res.FreedBlocks != 8 {
		t.Fatalf("freed blocks = %d; want 8", res.FreedBlocks)
	}
	if elapsed < 25*time.Millisecond {
		t.Fatalf("DrainWhenIdle returned in %v; expected it to wait for active request to finish", elapsed)
	}
}

// TestKVTierBackgroundEvictionLoop verifies periodic background pruning loop with idle gating.
func TestKVTierBackgroundEvictionLoop(t *testing.T) {
	tracker := gateway.NewActiveRequestTracker()
	mock := &mockKVEvictor{}
	evictor := gateway.NewIdleGatedEvictor(tracker, mock, gateway.WithCheckInterval(10*time.Millisecond))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Keep tracker busy initially
	done := tracker.Track()
	evictor.Start(ctx)
	defer evictor.Stop()

	// Wait 40ms while busy — no eviction should fire
	time.Sleep(40 * time.Millisecond)
	if mock.evictCount.Load() != 0 {
		t.Fatalf("evictions occurred while server was busy: %d", mock.evictCount.Load())
	}

	// Release request — server becomes idle
	done()

	// Trigger immediate check and wait briefly for ticker
	evictor.Trigger()
	time.Sleep(50 * time.Millisecond)

	if mock.evictCount.Load() == 0 {
		t.Fatalf("expected background loop to evict once idle")
	}

	evictor.Stop()
	if evictor.IsRunning() {
		t.Fatalf("evictor should not be running after Stop()")
	}
}

// TestKVTierLoopbackKVStoreIntegration tests directio.LoopbackKVStore integrated with IdleGatedEvictor.
func TestKVTierLoopbackKVStoreIntegration(t *testing.T) {
	tmpDir := t.TempDir()
	storePath := filepath.Join(tmpDir, "tier_store.kv")

	store, err := directio.NewLoopbackKVStore(storePath, 32)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()

	// Allocate 10 blocks in store
	for i := 0; i < 10; i++ {
		bid, err := store.AllocateBlock(directio.BlockMetadata{
			SessionID: "sess-tier",
			Turn:      1,
		})
		if err != nil {
			t.Fatalf("alloc: %v", err)
		}
		data := make([]byte, 1024)
		data[0] = byte(i)
		_ = store.WriteBlock(bid, data)
	}

	tracker := gateway.NewActiveRequestTracker()
	// store satisfies gateway.KVEvictor directly via EvictKV method
	evictor := gateway.NewIdleGatedEvictor(tracker, store)

	ctx := context.Background()

	// While busy:
	done := tracker.Track()
	_, err = evictor.TryEvict(ctx)
	if !errors.Is(err, gateway.ErrEvictionBlockedBusy) {
		t.Fatalf("expected ErrEvictionBlockedBusy, got %v", err)
	}
	if store.AllocatedCount() != 10 {
		t.Fatalf("store blocks evicted while busy: got %d, want 10", store.AllocatedCount())
	}

	// Become idle:
	done()
	res, err := evictor.TryEvict(ctx)
	if err != nil {
		t.Fatalf("TryEvict while idle: %v", err)
	}
	if res.Outcome != gateway.EvictionOutcomeDrained {
		t.Fatalf("res.Outcome = %q; want %q", res.Outcome, gateway.EvictionOutcomeDrained)
	}
	if res.FreedBlocks != 10 {
		t.Fatalf("freed blocks = %d; want 10", res.FreedBlocks)
	}
	if store.AllocatedCount() != 0 {
		t.Fatalf("store remaining blocks = %d; want 0", store.AllocatedCount())
	}
}
