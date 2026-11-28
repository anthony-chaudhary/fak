// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func waitForWaiter(t *testing.T, lim *DynamicConcurrencyLimit) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		lim.mu.Lock()
		waiting := lim.waitCh != nil
		lim.mu.Unlock()
		if waiting {
			return
		}
		runtime.Gosched()
	}
	t.Fatal("timed out waiting for Acquire goroutine to park on waitCh")
}

func assertStillWaiting(t *testing.T, lim *DynamicConcurrencyLimit, acquired <-chan struct{}) {
	t.Helper()
	lim.mu.Lock()
	waiting := lim.waitCh != nil
	lim.mu.Unlock()
	if !waiting {
		t.Fatal("waitCh cleared prematurely while debt remained")
	}
	select {
	case <-acquired:
		t.Fatal("Acquire returned prematurely before debt drained")
	default:
	}
}

func assertConcurrencyStats(t *testing.T, got, want ConcurrencyStats) {
	t.Helper()
	assertConcurrencyInvariants(t, got)
	if got != want {
		t.Fatalf("stats=%+v want %+v", got, want)
	}
}

func assertConcurrencyInvariants(t *testing.T, s ConcurrencyStats) {
	t.Helper()
	if s.Limit < 0 || s.Active < 0 || s.Available < 0 || s.Debt < 0 {
		t.Fatalf("negative concurrency counter: %+v", s)
	}
	// Conservation law: active + available - debt == limit (active + available == limit + debt).
	if s.Active+s.Available-s.Debt != s.Limit {
		t.Fatalf("semaphore conservation violated: active(%d)+available(%d)-debt(%d) != limit(%d) (%+v)",
			s.Active, s.Available, s.Debt, s.Limit, s)
	}
	// Available slots and debt cannot coexist.
	if s.Available > 0 && s.Debt > 0 {
		t.Fatalf("available and debt simultaneously >0: %+v", s)
	}
	wantAvailable := max(0, s.Limit-s.Active)
	wantDebt := max(0, s.Active-s.Limit)
	if s.Available != wantAvailable || s.Debt != wantDebt {
		t.Fatalf("unexpected available/debt split in %+v: want available=%d debt=%d", s, wantAvailable, wantDebt)
	}
}

// fak-test:runtime fast est=1s
// TestDebtDrain pins the dynamic concurrency limiter contract adapted from
// limit decreases drain available slots first and record any remainder as debt
// absorbed by subsequent releases or limit increases without violating the
// non-negative semaphore or conservation invariants.
func TestDebtDrain(t *testing.T) {
	t.Run("validationSentinels", func(t *testing.T) {
		if _, err := NewDynamicConcurrencyLimit(-1); !errors.Is(err, errNegativeConcurrencyLimit) {
			t.Fatalf("NewDynamicConcurrencyLimit(-1) err=%v want %v", err, errNegativeConcurrencyLimit)
		}

		zero, err := NewDynamicConcurrencyLimit(0)
		if err != nil {
			t.Fatalf("NewDynamicConcurrencyLimit(0) unexpected err=%v", err)
		}
		assertConcurrencyStats(t, zero.Stats(), ConcurrencyStats{Limit: 0, Active: 0, Available: 0, Debt: 0})
		if zero.TryAcquire() {
			t.Fatal("TryAcquire succeeded at limit=0")
		}
		if err := zero.SetLimit(-2); !errors.Is(err, errNegativeConcurrencyLimit) {
			t.Fatalf("SetLimit(-2) err=%v want %v", err, errNegativeConcurrencyLimit)
		}
		if err := zero.Release(); !errors.Is(err, errReleaseWithoutActive) {
			t.Fatalf("Release() with no active err=%v want %v", err, errReleaseWithoutActive)
		}
		assertConcurrencyStats(t, zero.Stats(), ConcurrencyStats{Limit: 0, Active: 0, Available: 0, Debt: 0})
	})

	t.Run("decreaseDrainsAvailableThenAccruesDebt", func(t *testing.T) {
		lim, err := NewDynamicConcurrencyLimit(5)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 4; i++ {
			if !lim.TryAcquire() {
				t.Fatalf("TryAcquire %d failed", i)
			}
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 5, Active: 4, Available: 1, Debt: 0})

		// Decrease by 3 (from 5 to 2): drains 1 available slot first, converts remaining 2 to debt.
		if err := lim.SetLimit(2); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 2, Active: 4, Available: 0, Debt: 2})
		if lim.TryAcquire() {
			t.Fatal("TryAcquire must fail while debt is non-zero")
		}

		// First release is absorbed by debt; available stays 0.
		if err := lim.Release(); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 2, Active: 3, Available: 0, Debt: 1})
		if lim.TryAcquire() {
			t.Fatal("TryAcquire must fail while debt=1")
		}

		// Second release absorbs remaining debt; available stays 0 because active == limit.
		if err := lim.Release(); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 2, Active: 2, Available: 0, Debt: 0})
		if lim.TryAcquire() {
			t.Fatal("TryAcquire must fail while active == limit")
		}

		// Third release sees debt=0 and restores one available slot.
		if err := lim.Release(); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 2, Active: 1, Available: 1, Debt: 0})

		// A new acquire succeeds under the reduced limit.
		if !lim.TryAcquire() {
			t.Fatal("TryAcquire should succeed once active < limit and debt == 0")
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 2, Active: 2, Available: 0, Debt: 0})

		for i := 0; i < 2; i++ {
			if err := lim.Release(); err != nil {
				t.Fatal(err)
			}
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 2, Active: 0, Available: 2, Debt: 0})
	})

	t.Run("decreaseWithinAvailableAccruesNoDebt", func(t *testing.T) {
		lim, err := NewDynamicConcurrencyLimit(5)
		if err != nil {
			t.Fatal(err)
		}
		if !lim.TryAcquire() {
			t.Fatal("TryAcquire failed")
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 5, Active: 1, Available: 4, Debt: 0})

		if err := lim.SetLimit(3); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 3, Active: 1, Available: 2, Debt: 0})

		if err := lim.SetLimit(1); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 1, Active: 1, Available: 0, Debt: 0})

		if err := lim.Release(); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 1, Active: 0, Available: 1, Debt: 0})
	})

	t.Run("increasePaysDownDebtBeforeAddingAvailable", func(t *testing.T) {
		lim, err := NewDynamicConcurrencyLimit(6)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 5; i++ {
			if !lim.TryAcquire() {
				t.Fatalf("TryAcquire %d failed", i)
			}
		}
		if err := lim.SetLimit(1); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 1, Active: 5, Available: 0, Debt: 4})

		// Increase by 2 (< debt 4): reduces debt to 2 without opening any slots.
		if err := lim.SetLimit(3); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 3, Active: 5, Available: 0, Debt: 2})
		if lim.TryAcquire() {
			t.Fatal("TryAcquire must fail while active(5) > limit(3)")
		}

		// No-op SetLimit keeps state intact.
		if err := lim.SetLimit(3); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 3, Active: 5, Available: 0, Debt: 2})

		// Increase by 4 (> debt 2): pays remaining 2 debt and opens 2 available slots.
		if err := lim.SetLimit(7); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 7, Active: 5, Available: 2, Debt: 0})

		if !lim.TryAcquire() || !lim.TryAcquire() {
			t.Fatal("expected two available slots after debt was paid off")
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 7, Active: 7, Available: 0, Debt: 0})
		if lim.TryAcquire() {
			t.Fatal("TryAcquire must fail at limit=7")
		}
	})

	t.Run("decreaseToZeroDrainsCompletely", func(t *testing.T) {
		lim, err := NewDynamicConcurrencyLimit(3)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 3; i++ {
			if !lim.TryAcquire() {
				t.Fatalf("TryAcquire %d failed", i)
			}
		}
		if err := lim.SetLimit(0); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 0, Active: 3, Available: 0, Debt: 3})

		for want := 2; want >= 0; want-- {
			if err := lim.Release(); err != nil {
				t.Fatal(err)
			}
			assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 0, Active: want, Available: 0, Debt: want})
		}

		if err := lim.SetLimit(2); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 2, Active: 0, Available: 2, Debt: 0})
	})

	t.Run("blockedAcquireWaitsThroughDebtAndWakesOnAvailable", func(t *testing.T) {
		lim, err := NewDynamicConcurrencyLimit(3)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 3; i++ {
			if !lim.TryAcquire() {
				t.Fatalf("TryAcquire %d failed", i)
			}
		}
		if err := lim.SetLimit(1); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 1, Active: 3, Available: 0, Debt: 2})

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		var acquiredDuringDebt atomic.Bool
		acquiredCh := make(chan struct{})
		go func() {
			if lim.Acquire(ctx) {
				s := lim.Stats()
				if s.Debt > 0 || s.Active > s.Limit {
					acquiredDuringDebt.Store(true)
				}
				close(acquiredCh)
			}
		}()

		waitForWaiter(t, lim)

		// First two releases are absorbed by debt=2; waiter stays blocked.
		if err := lim.Release(); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 1, Active: 2, Available: 0, Debt: 1})
		assertStillWaiting(t, lim, acquiredCh)
		if err := lim.Release(); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 1, Active: 1, Available: 0, Debt: 0})
		assertStillWaiting(t, lim, acquiredCh)

		// Third release frees the single permitted slot and wakes the blocked waiter.
		if err := lim.Release(); err != nil {
			t.Fatal(err)
		}
		select {
		case <-acquiredCh:
		case <-ctx.Done():
			t.Fatalf("blocked Acquire did not wake after debt drained: %v", ctx.Err())
		}
		if acquiredDuringDebt.Load() {
			t.Fatal("Acquire returned while debt > 0 or active > limit")
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 1, Active: 1, Available: 0, Debt: 0})
		if err := lim.Release(); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 1, Active: 0, Available: 1, Debt: 0})
	})

	t.Run("blockedAcquireWakesOnIncreaseAndRespectsCancel", func(t *testing.T) {
		lim, err := NewDynamicConcurrencyLimit(2)
		if err != nil {
			t.Fatal(err)
		}
		if !lim.TryAcquire() || !lim.TryAcquire() {
			t.Fatal("initial TryAcquire failed")
		}
		if err := lim.SetLimit(1); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 1, Active: 2, Available: 0, Debt: 1})

		// Pre-canceled context never acquires even if slots later exist.
		canceledCtx, cancelImmediate := context.WithCancel(context.Background())
		cancelImmediate()
		if lim.Acquire(canceledCtx) {
			t.Fatal("pre-canceled Acquire must return false")
		}

		// Canceled blocked waiter returns false cleanly.
		waitCtx, waitCancel := context.WithCancel(context.Background())
		waitDone := make(chan bool, 1)
		go func() {
			waitDone <- lim.Acquire(waitCtx)
		}()
		waitForWaiter(t, lim)
		waitCancel()
		select {
		case ok := <-waitDone:
			if ok {
				t.Fatal("canceled Acquire returned true")
			}
		case <-time.After(2 * time.Second):
			t.Fatal("canceled Acquire did not exit promptly")
		}

		// Waiter blocked behind debt wakes when limit rises past active.
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		acquiredCh := make(chan bool, 1)
		go func() {
			acquiredCh <- lim.Acquire(ctx)
		}()
		waitForWaiter(t, lim)

		// Raising to 2 only pays off debt=1; waiter must wait for limit=3.
		if err := lim.SetLimit(2); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 2, Active: 2, Available: 0, Debt: 0})
		select {
		case <-acquiredCh:
			t.Fatal("Acquire returned when SetLimit(2) only paid off debt")
		default:
		}

		if err := lim.SetLimit(3); err != nil {
			t.Fatal(err)
		}
		select {
		case ok := <-acquiredCh:
			if !ok {
				t.Fatal("Acquire returned false after limit raised above active")
			}
		case <-ctx.Done():
			t.Fatalf("Acquire did not wake on SetLimit increase: %v", ctx.Err())
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 3, Active: 3, Available: 0, Debt: 0})
		for i := 0; i < 3; i++ {
			if err := lim.Release(); err != nil {
				t.Fatal(err)
			}
		}
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 3, Active: 0, Available: 3, Debt: 0})
	})

	t.Run("twoTierPrefillDebtDrain", func(t *testing.T) {
		inFlight := newClientInFlight(4, 3)
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		for i := 0; i < 3; i++ {
			if !inFlight.beginPrefill(ctx) {
				t.Fatalf("beginPrefill %d failed", i)
			}
		}
		if err := inFlight.setPrefillLimit(1); err != nil {
			t.Fatal(err)
		}
		assertConcurrencyStats(t, inFlight.prefill.Stats(), ConcurrencyStats{Limit: 1, Active: 3, Available: 0, Debt: 2})

		entered := make(chan bool, 1)
		go func() {
			entered <- inFlight.beginPrefill(ctx)
		}()
		waitForWaiter(t, inFlight.prefill)

		// Releases 1 and 2 pay off debt=2; release 3 opens the slot for the 4th caller.
		inFlight.endPrefill()
		assertConcurrencyStats(t, inFlight.prefill.Stats(), ConcurrencyStats{Limit: 1, Active: 2, Available: 0, Debt: 1})
		inFlight.endPrefill()
		assertConcurrencyStats(t, inFlight.prefill.Stats(), ConcurrencyStats{Limit: 1, Active: 1, Available: 0, Debt: 0})
		select {
		case <-entered:
			t.Fatal("4th beginPrefill entered before slot freed past limit=1")
		default:
		}
		inFlight.endPrefill()

		select {
		case ok := <-entered:
			if !ok {
				t.Fatal("4th beginPrefill returned false")
			}
		case <-ctx.Done():
			t.Fatalf("4th beginPrefill stalled after debt drain: %v", ctx.Err())
		}
		inFlight.endPrefill()
		assertConcurrencyStats(t, inFlight.prefill.Stats(), ConcurrencyStats{Limit: 1, Active: 0, Available: 1, Debt: 0})
	})

	t.Run("concurrentChangesPreserveInvariants", func(t *testing.T) {
		lim, err := NewDynamicConcurrencyLimit(4)
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()

		var wg sync.WaitGroup
		const workers = 8
		const opsPerWorker = 40
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := 0; i < opsPerWorker; i++ {
					if !lim.Acquire(ctx) {
						t.Errorf("Acquire failed under active context: %v", ctx.Err())
						return
					}
					assertConcurrencyInvariants(t, lim.Stats())
					if err := lim.Release(); err != nil {
						t.Errorf("Release failed: %v", err)
						return
					}
					assertConcurrencyInvariants(t, lim.Stats())
				}
			}()
		}

		wg.Add(1)
		go func() {
			defer wg.Done()
			schedule := []int{2, 0, 5, 1, 8, 3, 4}
			for i := 0; i < 35; i++ {
				target := schedule[i%len(schedule)]
				if target == 0 {
					// Keep nonzero capacity quickly restored so workers complete deterministically.
					if err := lim.SetLimit(0); err != nil {
						t.Errorf("SetLimit(0) failed: %v", err)
					}
					assertConcurrencyInvariants(t, lim.Stats())
					target = 2
				}
				if err := lim.SetLimit(target); err != nil {
					t.Errorf("SetLimit(%d) failed: %v", target, err)
				}
				assertConcurrencyInvariants(t, lim.Stats())
			}
			if err := lim.SetLimit(4); err != nil {
				t.Errorf("final SetLimit(4) failed: %v", err)
			}
		}()

		wg.Wait()
		assertConcurrencyStats(t, lim.Stats(), ConcurrencyStats{Limit: 4, Active: 0, Available: 4, Debt: 0})
	})
}
