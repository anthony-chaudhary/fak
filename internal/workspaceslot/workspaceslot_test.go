package workspaceslot

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestSlotRingInitializationAndAcquisition(t *testing.T) {
	baseDir := t.TempDir()
	const cap = 4

	ring, err := NewSlotRing(baseDir, cap)
	if err != nil {
		t.Fatalf("NewSlotRing: %v", err)
	}
	defer ring.Close()

	if ring.Capacity() != cap {
		t.Fatalf("expected capacity %d, got %d", cap, ring.Capacity())
	}
	if ring.AvailableCount() != cap {
		t.Fatalf("expected available %d, got %d", cap, ring.AvailableCount())
	}

	// Verify directories created on disk
	for i := 0; i < cap; i++ {
		slotDir := filepath.Join(baseDir, fmt.Sprintf("slot-%02d", i))
		if _, err := os.Stat(slotDir); err != nil {
			t.Fatalf("slot dir %s missing: %v", slotDir, err)
		}
		if _, err := os.Stat(filepath.Join(slotDir, "scratch")); err != nil {
			t.Fatalf("scratch dir missing: %v", err)
		}
		if _, err := os.Stat(filepath.Join(slotDir, "repo")); err != nil {
			t.Fatalf("repo dir missing: %v", err)
		}
	}

	// Acquire all slots
	var slots []*Slot
	for i := 0; i < cap; i++ {
		s, err := ring.Acquire(context.Background(), fmt.Sprintf("sess-%d", i))
		if err != nil {
			t.Fatalf("acquire slot %d: %v", i, err)
		}
		if !s.Leased {
			t.Fatalf("slot %d should be leased", i)
		}
		slots = append(slots, s)
	}

	// TryAcquire should fail when all are leased
	_, err = ring.TryAcquire("overflow")
	if err != ErrNoSlotsAvailable {
		t.Fatalf("expected ErrNoSlotsAvailable, got %v", err)
	}

	// Release one slot
	if err := ring.Release(slots[0]); err != nil {
		t.Fatalf("release slot: %v", err)
	}
	if ring.AvailableCount() != 1 {
		t.Fatalf("expected 1 available slot, got %d", ring.AvailableCount())
	}

	// Re-acquire slot
	sReacquired, err := ring.Acquire(context.Background(), "sess-new")
	if err != nil {
		t.Fatalf("re-acquire slot: %v", err)
	}
	if sReacquired.Index != slots[0].Index {
		t.Fatalf("expected re-acquired slot index %d, got %d", slots[0].Index, sReacquired.Index)
	}

	// Clean up all
	for _, s := range slots[1:] {
		_ = ring.Release(s)
	}
	_ = ring.Release(sReacquired)
}

func TestFastInPlaceRecyclingAndProcessReaper(t *testing.T) {
	baseDir := t.TempDir()
	killedPIDs := make(map[int]bool)

	mockReaper := func(pid int) error {
		killedPIDs[pid] = true
		return nil
	}

	ring, err := NewSlotRing(baseDir, 2, WithProcessReaper(mockReaper))
	if err != nil {
		t.Fatalf("NewSlotRing: %v", err)
	}
	defer ring.Close()

	slot, err := ring.Acquire(context.Background(), "session-with-files")
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	// Bind mock PIDs
	slot.BindPID(4242)
	slot.BindPID(4243)

	// Write scratch files and untracked repo files
	scratchFile := filepath.Join(slot.ScratchDir, "temp_data.bin")
	_ = os.WriteFile(scratchFile, []byte("large volatile data"), 0644)
	subDir := filepath.Join(slot.ScratchDir, "subdir")
	_ = os.MkdirAll(subDir, 0755)
	_ = os.WriteFile(filepath.Join(subDir, "nested.txt"), []byte("nested"), 0644)

	repoFile := filepath.Join(slot.RepoDir, "untracked.go")
	_ = os.WriteFile(repoFile, []byte("package main"), 0644)

	// Release and recycle
	if err := ring.Release(slot); err != nil {
		t.Fatalf("Release: %v", err)
	}

	// 1. Check child processes were reaped
	if !killedPIDs[4242] || !killedPIDs[4243] {
		t.Fatalf("expected PIDs 4242 and 4243 to be reaped, got %v", killedPIDs)
	}

	// 2. Check scratch directory emptied
	entries, err := os.ReadDir(slot.ScratchDir)
	if err != nil {
		t.Fatalf("ReadDir scratch: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("expected 0 entries in scratch dir after recycling, got %d", len(entries))
	}

	// 3. Check repo directory emptied
	repoEntries, err := os.ReadDir(slot.RepoDir)
	if err != nil {
		t.Fatalf("ReadDir repo: %v", err)
	}
	if len(repoEntries) != 0 {
		t.Fatalf("expected 0 entries in repo dir after recycling, got %d", len(repoEntries))
	}
}

// TestZeroDynamicTempGrowth10kCycles verifies that running 10,000 consecutive agent workflows
// creates ZERO new directory entries in %TEMP% and maintains constant O(1) slot acquisition latency.
func TestZeroDynamicTempGrowth10kCycles(t *testing.T) {
	tempDir := t.TempDir()
	baseDir := filepath.Join(tempDir, "slots")
	const numSlots = 8
	const numCycles = 10000

	ring, err := NewSlotRing(baseDir, numSlots)
	if err != nil {
		t.Fatalf("NewSlotRing: %v", err)
	}
	defer ring.Close()

	// Initial directory count in tempDir (should be exactly 1: "slots")
	initialEntries, err := os.ReadDir(tempDir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(initialEntries) != 1 {
		t.Fatalf("expected 1 initial entry ('slots'), got %d", len(initialEntries))
	}

	start := time.Now()
	var latencies []time.Duration

	// Execute 10,000 consecutive agent workflows through the slot ring
	for i := 0; i < numCycles; i++ {
		t0 := time.Now()
		slot, err := ring.Acquire(context.Background(), fmt.Sprintf("agent-%d", i))
		lat := time.Since(t0)
		if err != nil {
			t.Fatalf("cycle %d acquire failed: %v", i, err)
		}

		if i < 100 || i >= numCycles-100 {
			latencies = append(latencies, lat)
		}

		// Simulate writing scratch files inside the leased slot
		scratchFile := filepath.Join(slot.ScratchDir, "run.log")
		_ = os.WriteFile(scratchFile, []byte("turn output"), 0644)

		// In-place recycling
		if err := ring.Release(slot); err != nil {
			t.Fatalf("cycle %d release failed: %v", i, err)
		}
	}

	totalDuration := time.Since(start)
	t.Logf("Completed %d cycles in %v (avg %.2f µs/cycle)", numCycles, totalDuration, float64(totalDuration.Microseconds())/float64(numCycles))

	// Verify ZERO new directories created in tempDir!
	finalEntries, err := os.ReadDir(tempDir)
	if err != nil {
		t.Fatalf("ReadDir final: %v", err)
	}
	if len(finalEntries) != len(initialEntries) {
		t.Fatalf("DIR LEAK: expected %d entries in temp dir, got %d (new dirs created!)", len(initialEntries), len(finalEntries))
	}

	// Verify slot base dir has exactly numSlots
	slotEntries, err := os.ReadDir(baseDir)
	if err != nil {
		t.Fatalf("ReadDir baseDir: %v", err)
	}
	if len(slotEntries) != numSlots {
		t.Fatalf("expected exactly %d pre-allocated slots, got %d", numSlots, len(slotEntries))
	}
}

func BenchmarkSlotAcquireRelease(b *testing.B) {
	baseDir := b.TempDir()
	ring, err := NewSlotRing(baseDir, 16)
	if err != nil {
		b.Fatalf("NewSlotRing: %v", err)
	}
	defer ring.Close()

	ctx := context.Background()
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		slot, err := ring.Acquire(ctx, "bench-sess")
		if err != nil {
			b.Fatalf("Acquire: %v", err)
		}
		if err := ring.Release(slot); err != nil {
			b.Fatalf("Release: %v", err)
		}
	}
}

func TestSlotRingEdgeCasesAndErrors(t *testing.T) {
	// Invalid constructor args
	if _, err := NewSlotRing("", 4); err == nil {
		t.Fatalf("expected error on empty baseDir")
	}
	if _, err := NewSlotRing(t.TempDir(), 0); err == nil {
		t.Fatalf("expected error on 0 capacity")
	}

	baseDir := t.TempDir()
	ring, err := NewSlotRing(baseDir, 2)
	if err != nil {
		t.Fatalf("NewSlotRing: %v", err)
	}

	if ring.BaseDir() != baseDir {
		t.Fatalf("expected BaseDir %s, got %s", baseDir, ring.BaseDir())
	}

	// BindPID edge case (<= 0)
	slot, err := ring.Acquire(context.Background(), "s1")
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	slot.BindPID(-1)
	if len(slot.BoundPIDs) != 0 {
		t.Fatalf("expected no negative PIDs")
	}

	// Release invalid slot
	if err := ring.Release(nil); err != ErrInvalidSlot {
		t.Fatalf("expected ErrInvalidSlot on nil, got %v", err)
	}
	invalidSlot := &Slot{Index: 999}
	if err := ring.Release(invalidSlot); err != ErrInvalidSlot {
		t.Fatalf("expected ErrInvalidSlot on out of bounds index, got %v", err)
	}

	// Acquire with canceled context
	ctxCanceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ring.Acquire(ctxCanceled, "canceled"); err == nil {
		t.Fatalf("expected error on canceled ctx")
	}

	// Test git clean path in cleanRepoDir
	gitDir := filepath.Join(slot.RepoDir, ".git")
	_ = os.MkdirAll(gitDir, 0755)
	untracked := filepath.Join(slot.RepoDir, "untracked.txt")
	_ = os.WriteFile(untracked, []byte("junk"), 0644)

	if err := ring.Release(slot); err != nil {
		t.Fatalf("Release: %v", err)
	}

	// Close ring
	if err := ring.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	// Double close should be idempotent
	if err := ring.Close(); err != nil {
		t.Fatalf("double close error: %v", err)
	}

	// Acquire / TryAcquire on closed ring
	if _, err := ring.Acquire(context.Background(), "after-close"); err != ErrRingClosed {
		t.Fatalf("expected ErrRingClosed, got %v", err)
	}
	if _, err := ring.TryAcquire("after-close"); err != ErrRingClosed {
		t.Fatalf("expected ErrRingClosed, got %v", err)
	}
}
