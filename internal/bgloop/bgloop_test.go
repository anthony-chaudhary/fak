package bgloop

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"
)

func noopTick(context.Context) error { return nil }

var errTick = errors.New("boom-err")

// --- pure / lifecycle validation (no virtual time) ---------------------------

func TestRegisterValidation(t *testing.T) {
	s := New()
	if err := s.Register(Loop{Name: "  ", Tick: noopTick}); err == nil {
		t.Error("blank name should be rejected")
	}
	if err := s.Register(Loop{Name: "a", Tick: nil}); err == nil {
		t.Error("nil Tick should be rejected")
	}
	if err := s.Register(Loop{Name: "a", Tick: noopTick}); err != nil {
		t.Fatalf("valid register: %v", err)
	}
	if err := s.Register(Loop{Name: "a", Tick: noopTick}); err == nil {
		t.Error("duplicate name should be rejected")
	}
	if s.Len() != 1 {
		t.Errorf("Len=%d want 1", s.Len())
	}
}

func TestRegisterAfterStartRejected(t *testing.T) {
	s := New()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s.Start(ctx)
	if err := s.Register(Loop{Name: "late", Tick: noopTick}); err == nil {
		t.Error("Register after Start should be rejected")
	}
	if err := s.Shutdown(context.Background()); err != nil {
		t.Errorf("shutdown: %v", err)
	}
}

func TestSnapshotBeforeStart(t *testing.T) {
	s := New()
	_ = s.Register(Loop{Name: "z", Interval: 5 * time.Second, Tick: noopTick})
	_ = s.Register(Loop{Name: "a", Tick: noopTick}) // continuous
	snap := s.Snapshot()
	if len(snap) != 2 {
		t.Fatalf("len=%d want 2", len(snap))
	}
	if snap[0].Name != "a" || snap[1].Name != "z" {
		t.Errorf("snapshot not sorted by name: %q,%q", snap[0].Name, snap[1].Name)
	}
	if snap[0].Interval != "continuous" {
		t.Errorf("continuous interval=%q", snap[0].Interval)
	}
	if snap[1].Interval != "5s" {
		t.Errorf("interval=%q want 5s", snap[1].Interval)
	}
	if snap[0].State != StateIdle || snap[0].Ticks != 0 {
		t.Errorf("pre-start loop should be idle/zero, got %q/%d", snap[0].State, snap[0].Ticks)
	}
}

func TestNextBackoff(t *testing.T) {
	if got := nextBackoff(time.Second, time.Minute); got != 2*time.Second {
		t.Errorf("double: got %v want 2s", got)
	}
	if got := nextBackoff(40*time.Second, time.Minute); got != time.Minute {
		t.Errorf("cap: got %v want 1m", got)
	}
	if got := nextBackoff(time.Minute, time.Minute); got != time.Minute {
		t.Errorf("stay capped: got %v want 1m", got)
	}
}

// TestEqualJitter pins the pure jitter math: the realized wait lands in [base/2, base]
// for any frac in [0,1), is non-decreasing in frac, and a non-positive base passes
// through unchanged (no panic, no negative sleep).
func TestEqualJitter(t *testing.T) {
	const base = 100 * time.Millisecond
	if got := equalJitter(base, 0); got != base/2 {
		t.Errorf("frac=0: got %v want %v (base/2 floor)", got, base/2)
	}
	if got := equalJitter(base, 0.25); got > equalJitter(base, 0.75) {
		t.Error("equalJitter must be non-decreasing in frac")
	}
	for _, f := range []float64{0, 0.1, 0.33, 0.5, 0.9, 0.9999} {
		if w := equalJitter(base, f); w < base/2 || w > base {
			t.Errorf("frac=%v: wait %v out of [%v, %v]", f, w, base/2, base)
		}
	}
	if got := equalJitter(0, 0.5); got != 0 {
		t.Errorf("base=0: got %v want 0", got)
	}
	if got := equalJitter(-time.Second, 0.5); got != -time.Second {
		t.Errorf("negative base must pass through unchanged: got %v", got)
	}
}

// TestBackoffJitterApplied is the differential witness that the realized sleep — not
// the fixed schedule base — is what jitter perturbs: the same always-failing loop with
// a deterministic low jitter (wait ~ base/2) must retry strictly more often in the same
// virtual window than one with a high jitter (wait ~ base). backoffMin==backoffMax pins
// the schedule, isolating the effect to the jitter fraction.
func TestBackoffJitterApplied(t *testing.T) {
	runErrs := func(frac float64) uint64 {
		var n uint64
		synctest.Test(t, func(t *testing.T) {
			s := New(WithBackoff(20*time.Millisecond, 20*time.Millisecond))
			s.rng = func() float64 { return frac } // white-box: deterministic jitter
			_ = s.Register(Loop{Name: "errs", Tick: func(context.Context) error { return errTick }})
			ctx, cancel := context.WithCancel(context.Background())
			s.Start(ctx)
			time.Sleep(200 * time.Millisecond)
			synctest.Wait()
			st, _ := s.Get("errs")
			n = st.Errors
			cancel()
			synctest.Wait()
			_ = s.Shutdown(context.Background())
		})
		return n
	}
	low := runErrs(0)      // wait = 10ms -> ~20 retries in 200ms
	high := runErrs(0.999) // wait ~ 20ms -> ~10 retries in 200ms
	if low <= high {
		t.Errorf("jitter not applied to the sleep: low-jitter errors=%d should exceed high-jitter errors=%d", low, high)
	}
}

// --- supervised behavior under virtual time (testing/synctest) ---------------

// TestKeepsProgressing is the PROGRESS witness: while the kernel is up, an interval
// loop keeps ticking on the clock and its observable snapshot advances.
func TestKeepsProgressing(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var n atomic.Int64
		s := New()
		if err := s.Register(Loop{Name: "counter", Interval: 100 * time.Millisecond, Tick: func(context.Context) error {
			n.Add(1)
			return nil
		}}); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		s.Start(ctx)

		synctest.Wait()
		if got := n.Load(); got != 1 {
			t.Fatalf("after start: ticks=%d want 1 (immediate first tick)", got)
		}

		time.Sleep(550 * time.Millisecond)
		synctest.Wait()
		if got := n.Load(); got < 6 {
			t.Fatalf("after 550ms at 100ms interval: ticks=%d want >=6", got)
		}

		st, ok := s.Get("counter")
		if !ok || st.Ticks < 6 {
			t.Fatalf("snapshot ticks=%d want >=6", st.Ticks)
		}
		if !st.NextTickAt.After(st.StartedAt) {
			t.Errorf("NextTickAt %v should be after StartedAt %v", st.NextTickAt, st.StartedAt)
		}

		cancel()
		synctest.Wait()
		if st, _ := s.Get("counter"); st.State != StateStopped {
			t.Errorf("after cancel: state=%q want stopped", st.State)
		}
		if err := s.Shutdown(context.Background()); err != nil {
			t.Errorf("shutdown: %v", err)
		}
	})
}

// TestPanicContainmentKernelStaysUp is the CONTAINMENT witness: a loop that panics
// every tick keeps restarting under backoff (it never crashes the supervisor) AND a
// sibling healthy loop is wholly unaffected.
func TestPanicContainmentKernelStaysUp(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var healthy atomic.Int64
		s := New(WithBackoff(10*time.Millisecond, 40*time.Millisecond))
		_ = s.Register(Loop{Name: "boom", Interval: 100 * time.Millisecond, Tick: func(context.Context) error {
			panic("kaboom")
		}})
		_ = s.Register(Loop{Name: "healthy", Interval: 100 * time.Millisecond, Tick: func(context.Context) error {
			healthy.Add(1)
			return nil
		}})
		ctx, cancel := context.WithCancel(context.Background())
		s.Start(ctx)

		time.Sleep(500 * time.Millisecond)
		synctest.Wait()

		boom, _ := s.Get("boom")
		if boom.Panics < 3 {
			t.Errorf("boom panics=%d want >=3 (loop must keep restarting)", boom.Panics)
		}
		if boom.Restarts < 3 {
			t.Errorf("boom restarts=%d want >=3", boom.Restarts)
		}
		if boom.Ticks != 0 {
			t.Errorf("boom ticks=%d want 0 (every tick panics)", boom.Ticks)
		}
		if !strings.Contains(boom.LastErr, "kaboom") {
			t.Errorf("boom lastErr=%q want to contain panic value", boom.LastErr)
		}

		if h, _ := s.Get("healthy"); h.Ticks < 4 {
			t.Errorf("healthy ticks=%d want >=4 (a sibling panic must not stop it)", h.Ticks)
		}

		cancel()
		synctest.Wait()
		_ = s.Shutdown(context.Background())
	})
}

// TestErrorBackoff: a tick that errors is recorded and retried under backoff; errors
// and restarts move together and panics stay zero.
func TestErrorBackoff(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s := New(WithBackoff(10*time.Millisecond, 40*time.Millisecond))
		_ = s.Register(Loop{Name: "errs", Tick: func(context.Context) error { return errTick }})
		ctx, cancel := context.WithCancel(context.Background())
		s.Start(ctx)

		time.Sleep(200 * time.Millisecond)
		synctest.Wait()

		st, _ := s.Get("errs")
		if st.Errors < 2 {
			t.Errorf("errors=%d want >=2", st.Errors)
		}
		if st.Restarts != st.Errors {
			t.Errorf("restarts=%d errors=%d want equal", st.Restarts, st.Errors)
		}
		if st.Panics != 0 {
			t.Errorf("panics=%d want 0", st.Panics)
		}
		if st.LastErr != "boom-err" {
			t.Errorf("lastErr=%q want boom-err", st.LastErr)
		}

		cancel()
		synctest.Wait()
		_ = s.Shutdown(context.Background())
	})
}

// TestAdmitGatePausesAndResumes: the admit seam holds fires (StatePaused, Pauses
// climbs, no ticks) and the loop resumes ticking the moment admission flips.
func TestAdmitGatePausesAndResumes(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var allow atomic.Bool
		var n atomic.Int64
		s := New(WithAdmit(func(string) (bool, string) {
			if allow.Load() {
				return true, ""
			}
			return false, "held"
		}))
		_ = s.Register(Loop{Name: "gated", Interval: 50 * time.Millisecond, Tick: func(context.Context) error {
			n.Add(1)
			return nil
		}})
		ctx, cancel := context.WithCancel(context.Background())
		s.Start(ctx)

		time.Sleep(200 * time.Millisecond)
		synctest.Wait()
		if n.Load() != 0 {
			t.Fatalf("ticks ran while paused: n=%d", n.Load())
		}
		st, _ := s.Get("gated")
		if st.State != StatePaused {
			t.Errorf("state=%q want paused", st.State)
		}
		if st.Pauses < 2 {
			t.Errorf("pauses=%d want >=2", st.Pauses)
		}

		allow.Store(true)
		time.Sleep(200 * time.Millisecond)
		synctest.Wait()
		if n.Load() < 2 {
			t.Errorf("after resume: ticks=%d want >=2", n.Load())
		}

		cancel()
		synctest.Wait()
		_ = s.Shutdown(context.Background())
	})
}

// TestObserverFires: the push seam delivers a Status after each tick with advancing
// counters (the loopmgr-ledger / metrics fan-out hook).
func TestObserverFires(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var mu sync.Mutex
		var calls int
		var maxTicks uint64
		s := New(WithObserver(func(st Status) {
			mu.Lock()
			calls++
			if st.Ticks > maxTicks {
				maxTicks = st.Ticks
			}
			mu.Unlock()
		}))
		_ = s.Register(Loop{Name: "obs", Interval: 100 * time.Millisecond, Tick: noopTick})
		ctx, cancel := context.WithCancel(context.Background())
		s.Start(ctx)

		time.Sleep(350 * time.Millisecond)
		synctest.Wait()

		mu.Lock()
		c, m := calls, maxTicks
		mu.Unlock()
		if c < 3 {
			t.Errorf("observer calls=%d want >=3", c)
		}
		if m < 3 {
			t.Errorf("observer saw maxTicks=%d want >=3", m)
		}

		cancel()
		synctest.Wait()
		_ = s.Shutdown(context.Background())
	})
}

// TestContinuousLoopRespectsCancel: an Interval<=0 loop whose Tick paces itself keeps
// progressing and stops cleanly on cancellation.
func TestContinuousLoopRespectsCancel(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var n atomic.Int64
		s := New()
		_ = s.Register(Loop{Name: "cont", Tick: func(ctx context.Context) error {
			tmr := time.NewTimer(20 * time.Millisecond)
			defer tmr.Stop()
			select {
			case <-ctx.Done():
				return nil
			case <-tmr.C:
				n.Add(1)
				return nil
			}
		}})
		ctx, cancel := context.WithCancel(context.Background())
		s.Start(ctx)

		time.Sleep(100 * time.Millisecond)
		synctest.Wait()
		if n.Load() < 3 {
			t.Errorf("continuous ticks=%d want >=3", n.Load())
		}

		cancel()
		synctest.Wait()
		if st, _ := s.Get("cont"); st.State != StateStopped {
			t.Errorf("state=%q want stopped", st.State)
		}
		if err := s.Shutdown(context.Background()); err != nil {
			t.Errorf("shutdown: %v", err)
		}
	})
}

// --- injectable clock seam (deterministic time-travel, #1192) ----------------

// testClock is a deterministic, manually-advanced time source for the injectable `now`
// seam (#1192): a test places it at any instant — minutes to months from an epoch — and
// the supervisor reads every recorded timestamp back through it, with no wall clock and
// no real wait. It is the bgloop end of the one clock the dormancy harness drives; the
// mutex makes it safe if a future multi-loop test shares one across goroutines.
type testClock struct {
	mu  sync.Mutex
	cur time.Time
}

func newTestClock(start time.Time) *testClock { return &testClock{cur: start} }

func (c *testClock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.cur
}

func (c *testClock) advance(d time.Duration) {
	c.mu.Lock()
	c.cur = c.cur.Add(d)
	c.mu.Unlock()
}

// TestInjectedClockDrivesTimestamps is the time-travel witness the #1192 contract names
// for internal/bgloop: with an injected clock the supervisor records EVERY timestamp from
// that clock, not the wall clock, so a single test fast-forwards a loop's observable
// dormancy past 90 days with no real sleep. The clock starts at a 2020 epoch — far from
// any real run — so a leaked time.Now() (a 2026 wall clock) fails the year assertion. Each
// tick advances the injected clock one day; synctest paces the loop so ~90+ days elapse on
// the injected timeline in microseconds of real time.
func TestInjectedClockDrivesTimestamps(t *testing.T) {
	epoch := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
	const (
		step     = 24 * time.Hour   // each tick fast-forwards the injected clock one day
		interval = time.Millisecond // virtual pacing under synctest; no real wait
	)
	synctest.Test(t, func(t *testing.T) {
		clk := newTestClock(epoch)
		s := New(WithClock(clk.now))
		if err := s.Register(Loop{Name: "dorm", Interval: interval, Tick: func(context.Context) error {
			clk.advance(step)
			return nil
		}}); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		s.Start(ctx)

		// >180 virtual intervals => >180 ticks => >180 days advanced on the injected clock.
		time.Sleep(200 * interval)
		synctest.Wait()
		st, _ := s.Get("dorm")

		cancel()
		synctest.Wait()
		_ = s.Shutdown(context.Background())

		// StartedAt is read from the injected clock at loop begin — exactly the epoch,
		// before any tick advanced it. An equal check catches a wall-clock leak outright.
		if !st.StartedAt.Equal(epoch) {
			t.Errorf("StartedAt=%v want injected epoch %v (wall clock leaked into begin?)", st.StartedAt, epoch)
		}
		// Every recorded instant sits on the injected 2020 timeline, never the real clock.
		if st.LastTickAt.Year() != 2020 {
			t.Errorf("LastTickAt=%v not on the injected 2020 timeline (wall clock leaked into recordTick)", st.LastTickAt)
		}
		if st.NextTickAt.Year() != 2020 {
			t.Errorf("NextTickAt=%v not on the injected 2020 timeline (wall clock leaked into next-idle)", st.NextTickAt)
		}
		// The observable dormancy the injected clock simulated is well past 90 days — a
		// horizon no real test could wait out.
		if gap := st.LastTickAt.Sub(st.StartedAt); gap < 90*24*time.Hour {
			t.Errorf("simulated dormancy gap=%v want >= 90d (injected fast-forward did not drive the clock)", gap)
		}
		// After a clean tick the next-idle instant is exactly the last injected instant plus
		// one interval — proving both the tick-start and next-idle sites read the SAME seam.
		if want := st.LastTickAt.Add(interval); !st.NextTickAt.Equal(want) {
			t.Errorf("NextTickAt=%v want LastTickAt+interval=%v", st.NextTickAt, want)
		}
	})
}

// TestInjectedClockDrivesBackoffTimestamp covers the third seam site named in the contract
// — the backoff NextTickAt. An always-failing loop schedules its retry through the injected
// clock, so the recorded backoff instant is on the simulated timeline, not the wall clock.
func TestInjectedClockDrivesBackoffTimestamp(t *testing.T) {
	epoch := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
	const step = 24 * time.Hour
	synctest.Test(t, func(t *testing.T) {
		clk := newTestClock(epoch)
		s := New(WithClock(clk.now), WithBackoff(20*time.Millisecond, 20*time.Millisecond))
		if err := s.Register(Loop{Name: "errs", Tick: func(context.Context) error {
			clk.advance(step)
			return errTick
		}}); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		s.Start(ctx)

		time.Sleep(200 * time.Millisecond)
		synctest.Wait()
		st, _ := s.Get("errs")

		cancel()
		synctest.Wait()
		_ = s.Shutdown(context.Background())

		if st.Errors == 0 {
			t.Fatal("loop never errored; backoff path not exercised")
		}
		if !st.StartedAt.Equal(epoch) {
			t.Errorf("StartedAt=%v want injected epoch %v", st.StartedAt, epoch)
		}
		// The backoff retry instant (markBackoff) and the error instant (recordTick) both
		// come from the injected clock — on the 2020 timeline, never the wall clock.
		if st.NextTickAt.Year() != 2020 {
			t.Errorf("backoff NextTickAt=%v not on the injected 2020 timeline (wall clock leaked into markBackoff)", st.NextTickAt)
		}
		if st.LastErrAt.Year() != 2020 {
			t.Errorf("LastErrAt=%v not on the injected 2020 timeline", st.LastErrAt)
		}
	})
}

// --- clean shutdown / no-leak (real time) ------------------------------------

// TestShutdownJoinsCleanly: Shutdown returns nil only after every loop goroutine has
// exited (s.wg drained) — the no-leak witness — and every loop reads back Stopped.
func TestShutdownJoinsCleanly(t *testing.T) {
	s := New()
	for _, name := range []string{"a", "b", "c"} {
		_ = s.Register(Loop{Name: name, Interval: 10 * time.Millisecond, Tick: noopTick})
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s.Start(ctx)

	shctx, shcancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer shcancel()
	if err := s.Shutdown(shctx); err != nil {
		t.Fatalf("clean shutdown returned error: %v", err)
	}
	for _, st := range s.Snapshot() {
		if st.State != StateStopped {
			t.Errorf("loop %q state=%q want stopped", st.Name, st.State)
		}
	}
}

// TestShutdownTimesOutOnStuckTick: a Tick that ignores cancellation makes Shutdown
// return a timeout error naming the loop, rather than hanging silently.
func TestShutdownTimesOutOnStuckTick(t *testing.T) {
	release := make(chan struct{})
	s := New()
	_ = s.Register(Loop{Name: "stuck", Tick: func(context.Context) error {
		<-release // deliberately ignores ctx
		return nil
	}})
	ctx, cancel := context.WithCancel(context.Background())
	s.Start(ctx)

	if !waitFor(2*time.Second, func() bool {
		st, _ := s.Get("stuck")
		return st.State == StateRunning
	}) {
		t.Fatal("loop never entered the stuck tick")
	}

	shctx, shcancel := context.WithTimeout(context.Background(), 80*time.Millisecond)
	defer shcancel()
	err := s.Shutdown(shctx)
	if err == nil {
		t.Fatal("expected a shutdown timeout error")
	}
	if !strings.Contains(err.Error(), "stuck") || !strings.Contains(err.Error(), "timed out") {
		t.Errorf("err=%q want to name the stuck loop and 'timed out'", err.Error())
	}

	// Release so the goroutine exits and the test does not leak it.
	close(release)
	cancel()
	if err := s.Shutdown(context.Background()); err != nil {
		t.Errorf("final shutdown after release: %v", err)
	}
}

func waitFor(d time.Duration, cond func() bool) bool {
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if cond() {
			return true
		}
		time.Sleep(time.Millisecond)
	}
	return cond()
}
