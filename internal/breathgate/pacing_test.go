package breathgate

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestMinIntervalPacing(t *testing.T) {
	interval := 40 * time.Millisecond
	gate := New(Config{
		MaxConcurrent: 2,
		MinInterval:   interval,
	})

	ctx := context.Background()

	// Turn 1
	start1 := time.Now()
	if err := gate.Acquire(ctx); err != nil {
		t.Fatalf("first acquire failed: %v", err)
	}
	t1 := time.Now()
	gate.Release()

	// Turn 2 immediately requested
	if err := gate.Acquire(ctx); err != nil {
		t.Fatalf("second acquire failed: %v", err)
	}
	t2 := time.Now()
	gate.Release()

	elapsedBetweenTurns := t2.Sub(t1)
	if elapsedBetweenTurns < 30*time.Millisecond {
		t.Errorf("inter-turn pacing violated: expected at least ~30ms spacing, got %v (total from start: %v)", elapsedBetweenTurns, t2.Sub(start1))
	}
}

func TestConcurrencyThrottling(t *testing.T) {
	gate := New(Config{
		MaxConcurrent: 2,
		MinInterval:   0,
	})

	ctx := context.Background()

	// Fill slots 1 and 2
	if err := gate.Acquire(ctx); err != nil {
		t.Fatalf("acquire slot 1 failed: %v", err)
	}
	if err := gate.Acquire(ctx); err != nil {
		t.Fatalf("acquire slot 2 failed: %v", err)
	}

	if h := gate.Metrics().ActiveHolders; h != 2 {
		t.Fatalf("expected 2 active holders, got %d", h)
	}

	// Slot 3 should block
	acquired3 := int32(0)
	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()
		if err := gate.Acquire(ctx); err != nil {
			t.Errorf("third acquire failed: %v", err)
			return
		}
		atomic.StoreInt32(&acquired3, 1)
		gate.Release()
	}()

	// Verify that slot 3 is blocked
	time.Sleep(25 * time.Millisecond)
	if atomic.LoadInt32(&acquired3) != 0 {
		t.Fatal("third slot acquired before release, exceeding MaxConcurrent limit")
	}

	// Release one slot to unblock slot 3
	gate.Release()
	wg.Wait()

	if atomic.LoadInt32(&acquired3) != 1 {
		t.Fatal("third slot was not acquired after release")
	}

	// Release remaining slot 2
	gate.Release()

	m := gate.Metrics()
	if m.ActiveHolders != 0 {
		t.Errorf("expected 0 active holders at end, got %d", m.ActiveHolders)
	}
	if m.TotalAcquires != 3 || m.TotalReleases != 3 {
		t.Errorf("expected 3 acquires and 3 releases, got %+v", m)
	}
	if m.Throttles < 1 {
		t.Errorf("expected at least 1 throttle event recorded, got %d", m.Throttles)
	}
}

func TestConcurrentPacingBounds(t *testing.T) {
	const maxConcurrent = 3
	const numWorkers = 8
	const turnsPerWorker = 2
	const minInterval = 15 * time.Millisecond

	gate := New(Config{
		MaxConcurrent: maxConcurrent,
		MinInterval:   minInterval,
	})

	var currentActive int64
	var maxObservedActive int64
	var totalSuccessful int64

	var mu sync.Mutex
	var acquireTimes []time.Time

	var wg sync.WaitGroup
	ctx := context.Background()

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for turn := 0; turn < turnsPerWorker; turn++ {
				if err := gate.Acquire(ctx); err != nil {
					t.Errorf("concurrent acquire failed: %v", err)
					return
				}

				acqTime := time.Now()
				mu.Lock()
				acquireTimes = append(acquireTimes, acqTime)
				mu.Unlock()

				curr := atomic.AddInt64(&currentActive, 1)
				for {
					prev := atomic.LoadInt64(&maxObservedActive)
					if curr <= prev || atomic.CompareAndSwapInt64(&maxObservedActive, prev, curr) {
						break
					}
				}

				if curr > maxConcurrent {
					t.Errorf("active holders exceeded max: %d > %d", curr, maxConcurrent)
				}

				// Simulate minimal work
				time.Sleep(5 * time.Millisecond)

				atomic.AddInt64(&currentActive, -1)
				atomic.AddInt64(&totalSuccessful, 1)
				gate.Release()
			}
		}()
	}

	wg.Wait()

	expectedTotal := int64(numWorkers * turnsPerWorker)
	if total := atomic.LoadInt64(&totalSuccessful); total != expectedTotal {
		t.Fatalf("expected %d successful turns, got %d", expectedTotal, total)
	}

	if maxObs := atomic.LoadInt64(&maxObservedActive); maxObs > maxConcurrent {
		t.Fatalf("observed concurrency %d exceeded ceiling %d", maxObs, maxConcurrent)
	}

	m := gate.Metrics()
	if m.ActiveHolders != 0 {
		t.Errorf("expected 0 active holders at completion, got %d", m.ActiveHolders)
	}
	if m.TotalAcquires != expectedTotal || m.TotalReleases != expectedTotal {
		t.Errorf("mismatch in acquires/releases: %+v", m)
	}
}

func TestWaitDurationAccounting(t *testing.T) {
	gate := New(Config{
		MaxConcurrent: 1,
		MinInterval:   30 * time.Millisecond,
	})

	ctx := context.Background()

	// First acquire: immediate
	if err := gate.Acquire(ctx); err != nil {
		t.Fatalf("first acquire failed: %v", err)
	}
	gate.Release()

	// Second acquire: should wait for MinInterval
	start := time.Now()
	if err := gate.Acquire(ctx); err != nil {
		t.Fatalf("second acquire failed: %v", err)
	}
	elapsed := time.Since(start)
	gate.Release()

	if elapsed < 20*time.Millisecond {
		t.Errorf("expected second acquire to wait, took %v", elapsed)
	}

	m := gate.Metrics()
	if m.Throttles < 1 {
		t.Errorf("expected Throttles >= 1, got %d", m.Throttles)
	}
	if m.TotalWaitDuration <= 0 {
		t.Errorf("expected TotalWaitDuration > 0, got %v", m.TotalWaitDuration)
	}
}
