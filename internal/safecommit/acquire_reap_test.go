package safecommit

import (
	"errors"
	"testing"
	"time"
)

// fakeClock advances only when sleep is called, so acquireWithReap's bounded wait is
// exercised deterministically with no real time spent.
type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time        { return c.t }
func (c *fakeClock) sleep(d time.Duration) { c.t = c.t.Add(d) }
func newFakeClock() *fakeClock             { return &fakeClock{t: time.Unix(1_800_000_000, 0)} }

// TestAcquireWithReapBreaksMidWaitDeath proves the "waiters re-check liveness inside the
// wait loop" half of #2339: a holder that is busy for the first attempts but whose lock
// the reap eventually breaks is acquired within a few poll intervals — not stalled for
// the whole timeout. reap runs before every attempt.
func TestAcquireWithReapBreaksMidWaitDeath(t *testing.T) {
	clock := newFakeClock()
	var reapCalls, acqCalls int
	acquire := func(noWait bool) (func(), error) {
		acqCalls++
		if acqCalls < 3 { // busy until the reap "breaks" the dead holder's lock
			return nil, ErrLockBusy
		}
		return func() {}, nil
	}
	reap := func() { reapCalls++ }

	release, err := acquireWithReap(acquire, reap, false, 10*time.Second, 250*time.Millisecond, clock.now, clock.sleep)
	if err != nil {
		t.Fatalf("acquireWithReap: %v, want success once the lock is reaped", err)
	}
	if release == nil {
		t.Fatal("release is nil on success")
	}
	if acqCalls != 3 {
		t.Fatalf("acqCalls = %d, want 3 (busy, busy, acquired)", acqCalls)
	}
	if reapCalls != 3 {
		t.Fatalf("reapCalls = %d, want 3 (reap re-runs before every attempt)", reapCalls)
	}
}

// TestAcquireWithReapNoWait proves NoWait makes exactly one reap + one attempt, then
// LOCK_BUSY — never a wait.
func TestAcquireWithReapNoWait(t *testing.T) {
	clock := newFakeClock()
	var reapCalls, acqCalls, sleeps int
	acquire := func(noWait bool) (func(), error) { acqCalls++; return nil, ErrLockBusy }
	reap := func() { reapCalls++ }
	sleep := func(d time.Duration) { sleeps++; clock.sleep(d) }

	_, err := acquireWithReap(acquire, reap, true, 10*time.Second, 250*time.Millisecond, clock.now, sleep)
	if !errors.Is(err, ErrLockBusy) {
		t.Fatalf("NoWait busy: err = %v, want ErrLockBusy", err)
	}
	if acqCalls != 1 || reapCalls != 1 || sleeps != 0 {
		t.Fatalf("NoWait made acq=%d reap=%d sleeps=%d, want 1/1/0", acqCalls, reapCalls, sleeps)
	}
}

// TestAcquireWithReapTimeout proves an always-busy lock surfaces LOCK_BUSY once the
// deadline passes, with a bounded number of attempts (never an unbounded spin).
func TestAcquireWithReapTimeout(t *testing.T) {
	clock := newFakeClock()
	var acqCalls int
	acquire := func(noWait bool) (func(), error) { acqCalls++; return nil, ErrLockBusy }

	_, err := acquireWithReap(acquire, func() {}, false, 1*time.Second, 250*time.Millisecond, clock.now, clock.sleep)
	if !errors.Is(err, ErrLockBusy) {
		t.Fatalf("timeout: err = %v, want ErrLockBusy", err)
	}
	// 1s / 250ms => ~4 waits, so ~5 attempts. Bound loosely to catch an unbounded spin.
	if acqCalls < 4 || acqCalls > 6 {
		t.Fatalf("acqCalls = %d, want ~5 (1s/250ms bounded)", acqCalls)
	}
}

// TestAcquireWithReapPropagatesNonBusyError proves an infrastructure error from acquire
// (not ErrLockBusy) propagates immediately — it is never mistaken for contention.
func TestAcquireWithReapPropagatesNonBusyError(t *testing.T) {
	clock := newFakeClock()
	boom := errors.New("gpulease: open denied")
	var acqCalls int
	acquire := func(noWait bool) (func(), error) { acqCalls++; return nil, boom }

	_, err := acquireWithReap(acquire, func() {}, false, 10*time.Second, 250*time.Millisecond, clock.now, clock.sleep)
	if !errors.Is(err, boom) {
		t.Fatalf("non-busy error: err = %v, want boom propagated", err)
	}
	if acqCalls != 1 {
		t.Fatalf("acqCalls = %d, want 1 (propagate immediately)", acqCalls)
	}
}
