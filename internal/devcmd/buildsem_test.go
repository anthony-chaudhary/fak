package devcmd

import (
	"context"
	"errors"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestBuildSemConcurrentAcquisitionsUpToLimit verifies that up to `limit` slots can
// be acquired simultaneously without error, and that a subsequent acquisition fails
// with ErrBuildSlotTimeout when all slots are occupied.
func TestBuildSemConcurrentAcquisitionsUpToLimit(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "sem")
	const limit = 3
	sem := NewBuildSemaphore(dir, limit)

	ctx := context.Background()
	releases := make([]func(), limit)

	for i := 0; i < limit; i++ {
		rel, err := sem.Acquire(ctx, 100*time.Millisecond)
		if err != nil {
			t.Fatalf("acquire slot %d: %v", i, err)
		}
		if rel == nil {
			t.Fatalf("slot %d release func is nil", i)
		}
		releases[i] = rel
	}

	// All slots are now occupied. An acquisition with timeout=0 should immediately fail closed.
	_, err := sem.Acquire(ctx, 0)
	if !errors.Is(err, ErrBuildSlotTimeout) {
		t.Fatalf("expected ErrBuildSlotTimeout for slot %d, got %v", limit, err)
	}

	// Releasing one slot allows another acquisition to succeed.
	releases[0]()
	rel, err := sem.Acquire(ctx, 100*time.Millisecond)
	if err != nil {
		t.Fatalf("acquire after release: %v", err)
	}
	defer rel()

	for i := 1; i < limit; i++ {
		releases[i]()
	}
}

// TestBuildSemWaitsUntilReleased verifies that an acquisition blocks and waits when all
// slots are occupied, and unblocks promptly once a slot is released.
func TestBuildSemWaitsUntilReleased(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "sem")
	sem := NewBuildSemaphore(dir, 1)

	ctx := context.Background()
	rel1, err := sem.Acquire(ctx, 50*time.Millisecond)
	if err != nil {
		t.Fatalf("initial acquire: %v", err)
	}

	released := make(chan struct{})
	go func() {
		time.Sleep(60 * time.Millisecond)
		rel1()
		close(released)
	}()

	start := time.Now()
	rel2, err := sem.Acquire(ctx, 500*time.Millisecond)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("queued acquire: %v", err)
	}
	defer rel2()

	if elapsed < 40*time.Millisecond {
		t.Fatalf("acquisition returned too early (%v), did not wait for slot release", elapsed)
	}

	select {
	case <-released:
	default:
		t.Fatal("acquisition succeeded before background slot was released")
	}
}

// TestBuildSemTimeoutBehavior verifies that when slots remain busy past the timeout,
// the acquisition returns ErrBuildSlotTimeout and respects the deadline.
func TestBuildSemTimeoutBehavior(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "sem")
	sem := NewBuildSemaphore(dir, 1)

	ctx := context.Background()
	rel, err := sem.Acquire(ctx, 50*time.Millisecond)
	if err != nil {
		t.Fatalf("initial acquire: %v", err)
	}
	defer rel()

	timeout := 75 * time.Millisecond
	start := time.Now()
	_, err = sem.Acquire(ctx, timeout)
	elapsed := time.Since(start)

	if !errors.Is(err, ErrBuildSlotTimeout) {
		t.Fatalf("expected ErrBuildSlotTimeout, got %v", err)
	}
	if elapsed < 50*time.Millisecond {
		t.Fatalf("timeout returned prematurely after %v, expected at least ~50ms", elapsed)
	}
}

// TestBuildSemContextCancellation verifies that canceling the context aborts waiting
// and returns the context error.
func TestBuildSemContextCancellation(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "sem")
	sem := NewBuildSemaphore(dir, 1)

	ctx, cancel := context.WithCancel(context.Background())
	rel, err := sem.Acquire(ctx, 0)
	if err != nil {
		t.Fatalf("initial acquire: %v", err)
	}
	defer rel()

	go func() {
		time.Sleep(30 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	_, err = sem.Acquire(ctx, 5*time.Second)
	elapsed := time.Since(start)

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if elapsed > 500*time.Millisecond {
		t.Fatalf("context cancel took too long to unblock: %v", elapsed)
	}
}

// TestBuildSemHighConcurrencyStress verifies that multiple concurrent workers arbitrate
// cleanly without deadlocking or exceeding the configured concurrency limit.
func TestBuildSemHighConcurrencyStress(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "sem")
	const limit = 2
	const workers = 8
	sem := NewBuildSemaphore(dir, limit)

	var active int32
	var maxObserved int32
	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rel, err := sem.Acquire(context.Background(), 2*time.Second)
			if err != nil {
				t.Errorf("worker acquire failed: %v", err)
				return
			}
			cur := atomic.AddInt32(&active, 1)
			for {
				old := atomic.LoadInt32(&maxObserved)
				if cur <= old || atomic.CompareAndSwapInt32(&maxObserved, old, cur) {
					break
				}
			}
			time.Sleep(20 * time.Millisecond)
			atomic.AddInt32(&active, -1)
			rel()
		}()
	}

	wg.Wait()

	if maxObserved > limit {
		t.Fatalf("max concurrent holders %d exceeded limit %d", maxObserved, limit)
	}
}

// TestBuildSemDefaultAcquireAndConfig verifies package-level AcquireBuildSlot and configuration setters.
func TestBuildSemDefaultAcquireAndConfig(t *testing.T) {
	origDir := defaultBuildSemaphoreDir()
	origLimit := GetDefaultBuildConcurrency()
	tmpDir := t.TempDir()

	SetDefaultBuildSemaphoreDir(tmpDir)
	SetDefaultBuildConcurrency(1)
	t.Cleanup(func() {
		SetDefaultBuildSemaphoreDir(origDir)
		SetDefaultBuildConcurrency(origLimit)
	})

	if GetDefaultBuildConcurrency() != 1 {
		t.Fatalf("GetDefaultBuildConcurrency() = %d, want 1", GetDefaultBuildConcurrency())
	}

	ctx := context.Background()
	rel, err := AcquireBuildSlot(ctx, 100*time.Millisecond)
	if err != nil {
		t.Fatalf("AcquireBuildSlot: %v", err)
	}

	// Calling release multiple times must be idempotent.
	rel()
	rel()

	// After release, slot should be immediately acquirable again.
	rel2, err := AcquireBuildSlot(ctx, 100*time.Millisecond)
	if err != nil {
		t.Fatalf("AcquireBuildSlot after release: %v", err)
	}
	rel2()
}
