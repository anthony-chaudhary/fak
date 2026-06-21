package leakcheck_test

// leakcheck_test.go — the helpers must (a) PASS on leak-free code and (b) FIRE on a real
// leak. We prove (b) with a recording fake TB so the injected leak does not fail the parent
// test — a guard that never fails is worthless, so each helper is exercised both ways.

import (
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leakcheck"
)

// fakeTB records Errorf/Fatalf instead of failing the test, so we can assert a guard fired.
type fakeTB struct {
	failed bool
	msgs   []string
}

func (f *fakeTB) Helper()                           {}
func (f *fakeTB) Errorf(format string, args ...any) { f.failed = true }
func (f *fakeTB) Fatalf(format string, args ...any) { f.failed = true }

// --- Stable -------------------------------------------------------------------------

func TestStablePassesOnCleanBody(t *testing.T) {
	// A body that spawns a goroutine and JOINS it leaks nothing.
	body := func(i int) {
		var wg sync.WaitGroup
		wg.Add(1)
		go func() { defer wg.Done(); _ = i * i }()
		wg.Wait()
	}
	base, final := leakcheck.Stable(t, leakcheck.StableOpts{Iters: 30, Settle: time.Second}, body)
	if final > base+4 {
		t.Fatalf("clean body should be stable: base=%d final=%d", base, final)
	}
}

func TestStableFiresOnGoroutineLeak(t *testing.T) {
	release := make(chan struct{})
	defer close(release) // unblock the leaked goroutines so they don't pollute later tests
	body := func(i int) {
		go func() { <-release }() // never returns until the test ends → leaks one per call
	}
	f := &fakeTB{}
	leakcheck.Stable(f, leakcheck.StableOpts{Iters: 20, Slack: 3, Settle: 250 * time.Millisecond}, body)
	if !f.failed {
		t.Fatalf("Stable must FIRE when each call leaks a goroutine")
	}
}

// --- AllocScaling -------------------------------------------------------------------

func TestAllocScalingPassesOnReusedScratch(t *testing.T) {
	// setup(size) returns an op that allocates a FIXED amount per call regardless of the
	// size knob (the reused-scratch shape), so bytes/op is independent of size → ratio ≈ 1.
	setup := func(size int) func() {
		_ = size
		return func() {
			b := make([]byte, 256) // fixed alloc, independent of the size knob
			b[0] = 1
			_ = b
		}
	}
	ratio := leakcheck.AllocScaling(t, leakcheck.ScalingOpts{
		Small: 64, Large: 4096, Iters: 64, MaxRatio: 2.0}, setup)
	if ratio > 2.0 {
		t.Fatalf("reused-scratch op should not scale: ratio=%.2f", ratio)
	}
}

func TestAllocScalingFiresOnPerOpScaling(t *testing.T) {
	// setup(size) returns an op that allocates O(size) EVERY call — bytes/op tracks size,
	// so the ratio ≈ Large/Small and the guard must fire.
	setup := func(size int) func() {
		return func() {
			b := make([]byte, size)
			b[0] = 1
			_ = b
		}
	}
	f := &fakeTB{}
	ratio := leakcheck.AllocScaling(f, leakcheck.ScalingOpts{
		Small: 64, Large: 8192, Iters: 64, MaxRatio: 2.0}, setup)
	if !f.failed {
		t.Fatalf("AllocScaling must FIRE on an O(size) per-op buffer (ratio=%.2f)", ratio)
	}
}

// --- BoundedSize --------------------------------------------------------------------

func TestBoundedSizePassesOnBoundedMap(t *testing.T) {
	const cap = 16
	m := map[int]struct{}{}
	order := []int{}
	step := func(i int) {
		m[i] = struct{}{}
		order = append(order, i)
		for len(m) > cap {
			delete(m, order[0])
			order = order[1:]
		}
	}
	leakcheck.BoundedSize(t, 1000, cap, step, func() int { return len(m) })
}

func TestBoundedSizeFiresOnUnboundedMap(t *testing.T) {
	m := map[int]struct{}{}
	step := func(i int) { m[i] = struct{}{} } // never evicts → grows forever
	f := &fakeTB{}
	leakcheck.BoundedSize(f, 1000, 16, step, func() int { return len(m) })
	if !f.failed {
		t.Fatalf("BoundedSize must FIRE on a map that only grows")
	}
}
