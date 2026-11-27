// Dynamic concurrency limiter with debt tracking.
//
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from ai-dynamo/aiperf (DynamicConcurrencyLimit,
// src/aiperf/timing/concurrency.py:41 @
// ab610add4e8876fca223197596c728e17365f381, Apache-2.0); ADAPT disposition:
// re-expressed in idiomatic Go with a mutex and closed-channel broadcast
// wakeups; no upstream code is copied.
//
// Why debt exists: when a live concurrency limit decreases while more slots are
// already active than the new limit permits, in-flight requests cannot be
// preempted, and a semaphore's available count cannot become negative.
// Decreasing the limit drains available slots first and converts any remainder
// into debt. Subsequent Release calls (and later SetLimit increases) absorb
// debt before restoring available slots, preserving both the non-negative
// semaphore and the conservation invariant:
//
//	active + available - debt == limit   (equivalently active + available == limit + debt)
//	available >= 0, debt >= 0, available == 0 || debt == 0
package main

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

var (
	errNegativeConcurrencyLimit = errors.New("loadgen/concurrency: limit must be non-negative")
	errReleaseWithoutActive     = errors.New("loadgen/concurrency: release with no active slots")
)

// ConcurrencyStats is an atomic snapshot of a DynamicConcurrencyLimit's state.
type ConcurrencyStats struct {
	Limit     int
	Active    int
	Available int
	Debt      int
}

// DynamicConcurrencyLimit is a context-aware concurrency semaphore whose limit
// can be raised or lowered at runtime without violating the semaphore
// invariant or aborting in-flight holders.
type DynamicConcurrencyLimit struct {
	mu        sync.Mutex
	limit     int
	active    int
	available int
	debt      int
	waitCh    chan struct{} // closed and cleared whenever available becomes > 0
}

// NewDynamicConcurrencyLimit builds a limiter initialized to limit (must be >= 0).
// A limit of 0 admits no acquires until raised via SetLimit.
func NewDynamicConcurrencyLimit(limit int) (*DynamicConcurrencyLimit, error) {
	if limit < 0 {
		return nil, fmt.Errorf("%w: got %d", errNegativeConcurrencyLimit, limit)
	}
	return &DynamicConcurrencyLimit{
		limit:     limit,
		available: limit,
	}, nil
}

// Acquire blocks until a slot is available or ctx finishes, returning true when
// a slot was acquired. An already-canceled context never acquires a slot even
// when one is available.
func (l *DynamicConcurrencyLimit) Acquire(ctx context.Context) bool {
	if ctx == nil {
		ctx = context.Background()
	}
	for {
		if ctx.Err() != nil {
			return false
		}
		l.mu.Lock()
		if l.available > 0 {
			l.available--
			l.active++
			l.mu.Unlock()
			return true
		}
		if l.waitCh == nil {
			l.waitCh = make(chan struct{})
		}
		ch := l.waitCh
		l.mu.Unlock()

		select {
		case <-ctx.Done():
			return false
		case <-ch:
		}
	}
}

// TryAcquire acquires a slot without blocking, returning false if none is
// currently available.
func (l *DynamicConcurrencyLimit) TryAcquire() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.available <= 0 {
		return false
	}
	l.available--
	l.active++
	return true
}

// Release returns one active slot. If prior limit decreases left outstanding
// debt, the release is absorbed into debt rather than restoring an available
// slot; otherwise available increments and blocked Acquire callers are woken.
func (l *DynamicConcurrencyLimit) Release() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.active <= 0 {
		return errReleaseWithoutActive
	}
	l.active--
	if l.debt > 0 {
		l.debt--
		return nil
	}
	l.available++
	l.notifyLocked()
	return nil
}

// SetLimit adjusts the concurrency ceiling at runtime.
//
//   - Decrease: drains available slots first without blocking; any remaining
//     reduction becomes debt absorbed by future Release or SetLimit calls.
//   - Increase: pays down outstanding debt first; any remaining capacity raises
//     available slots and wakes blocked Acquire callers.
func (l *DynamicConcurrencyLimit) SetLimit(newLimit int) error {
	if newLimit < 0 {
		return fmt.Errorf("%w: got %d", errNegativeConcurrencyLimit, newLimit)
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	delta := newLimit - l.limit
	l.limit = newLimit
	if delta == 0 {
		return nil
	}
	if delta > 0 {
		pay := min(delta, l.debt)
		l.debt -= pay
		rem := delta - pay
		if rem > 0 {
			l.available += rem
			l.notifyLocked()
		}
		return nil
	}
	reduce := -delta
	drain := min(reduce, l.available)
	l.available -= drain
	rem := reduce - drain
	l.debt += rem
	return nil
}

// Stats returns an atomic snapshot of the limiter's counters.
func (l *DynamicConcurrencyLimit) Stats() ConcurrencyStats {
	l.mu.Lock()
	defer l.mu.Unlock()
	return ConcurrencyStats{
		Limit:     l.limit,
		Active:    l.active,
		Available: l.available,
		Debt:      l.debt,
	}
}

func (l *DynamicConcurrencyLimit) notifyLocked() {
	if l.waitCh != nil && l.available > 0 {
		close(l.waitCh)
		l.waitCh = nil
	}
}
