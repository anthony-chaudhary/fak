package servelaunch

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"
	"time"
)

// fakeClock pins the wait's origin; progression comes from the tick
// timestamps the fake ticker delivers, so nothing here ever sleeps.
type fakeClock struct{ now time.Time }

func (c *fakeClock) Now() time.Time { return c.now }

// fakeTicker delivers ticks the test enqueues by hand.
type fakeTicker struct {
	ch      chan time.Time
	stopped bool
}

func newFakeTicker(buffer int) *fakeTicker {
	return &fakeTicker{ch: make(chan time.Time, buffer)}
}

func (f *fakeTicker) Channel() <-chan time.Time { return f.ch }
func (f *fakeTicker) Stop()                     { f.stopped = true }

// prime enqueues n ticks at absolute timestamps origin+i*interval, without
// touching the clock: Wait consumes exactly as many as it needs and reports
// elapsed time from the delivered timestamps.
func (f *fakeTicker) prime(origin *fakeClock, n int, interval time.Duration) {
	for i := 1; i <= n; i++ {
		f.ch <- origin.now.Add(time.Duration(i) * interval)
	}
}

// harness builds a Config wired to hand-driven time. Ticks are pre-queued in
// generous excess: Wait consumes exactly as many as it needs and returns before
// touching the rest, so no test ever waits on real wall-clock time.
func harness(t *testing.T, healthURL string, interval, warm, cold time.Duration, state CacheState) (*fakeClock, *fakeTicker, Config) {
	t.Helper()
	clock := &fakeClock{now: time.Date(2026, 11, 9, 12, 0, 0, 0, time.UTC)}
	ticker := newFakeTicker(128)
	cfg := Config{
		HealthURL:  healthURL,
		Interval:   interval,
		Budgets:    Budgets{Warm: warm, Cold: cold},
		CacheState: func() CacheState { return state },
		Clock:      clock,
		NewTicker:  func(time.Duration) Ticker { return ticker },
	}
	return clock, ticker, cfg
}

func unhealthyServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestReadinessReadyAfterFailedPolls proves the happy path: two failed polls,
// then a healthy answer, yields the Ready sentinel with exact accounting.
// fak-test:runtime fast est=50ms lane=default
func TestReadinessReadyAfterFailedPolls(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		if calls < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)

	clock, ticker, cfg := harness(t, srv.URL, time.Second, 10*time.Second, 10*time.Second, CacheWarm)
	ticker.prime(clock, 4, time.Second)

	res, err := Wait(context.Background(), cfg)
	if !errors.Is(err, ErrReady) {
		t.Fatalf("err = %v, want errors.Is(err, ErrReady)", err)
	}
	if res.Verdict != VerdictReady {
		t.Errorf("Verdict = %q, want %q", res.Verdict, VerdictReady)
	}
	if res.Attempts != 3 {
		t.Errorf("Attempts = %d, want 3", res.Attempts)
	}
	if res.Elapsed != 2*time.Second {
		t.Errorf("Elapsed = %s, want 2s", res.Elapsed)
	}
	if len(res.CrashLines) != 0 {
		t.Errorf("CrashLines = %v, want none for Ready", res.CrashLines)
	}
	if res.CacheState != CacheWarm || res.Budget != 10*time.Second {
		t.Errorf("CacheState/Budget = %v/%s, want warm/10s", res.CacheState, res.Budget)
	}
}

// TestReadinessTimeoutWhenHealthNeverAnswers proves the deadline branch with
// the crash side-channel disabled (zero-value Crash): the wait rides out the
// full budget and reports Timeout, not Crashed.
// fak-test:runtime fast est=50ms lane=default
func TestReadinessTimeoutWhenHealthNeverAnswers(t *testing.T) {
	srv := unhealthyServer(t)

	clock, ticker, cfg := harness(t, srv.URL, time.Second, 4*time.Second, 4*time.Second, CacheCold)
	ticker.prime(clock, 8, time.Second)

	res, err := Wait(context.Background(), cfg)
	if !errors.Is(err, ErrTimeout) {
		t.Fatalf("err = %v, want errors.Is(err, ErrTimeout)", err)
	}
	if res.Verdict != VerdictTimeout {
		t.Errorf("Verdict = %q, want %q", res.Verdict, VerdictTimeout)
	}
	if res.Attempts != 4 {
		t.Errorf("Attempts = %d, want 4 (polls at t=0..3s, timeout at the 4s tick)", res.Attempts)
	}
	if res.Elapsed != 4*time.Second {
		t.Errorf("Elapsed = %s, want 4s", res.Elapsed)
	}
	if len(res.CrashLines) != 0 {
		t.Errorf("CrashLines = %v, want none without a side-channel", res.CrashLines)
	}
}

// TestReadinessCrashBeatsDeadline proves the core race: with an unhealthy
// server and a 10s budget, an exit signal arriving after the first poll
// returns Crashed with the fetched last-log lines, far inside the budget.
// fak-test:runtime fast est=50ms lane=default
func TestReadinessCrashBeatsDeadline(t *testing.T) {
	firstHit := make(chan struct{})
	var once sync.Once
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		once.Do(func() { close(firstHit) })
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(srv.Close)

	_, _, cfg := harness(t, srv.URL, time.Second, 10*time.Second, 10*time.Second, CacheWarm)
	crashCh := make(chan struct{}, 1)
	wantLines := []string{"panic: runtime.out of memory", "goroutine 1 [running]: main.serve"}
	cfg.Crash = CrashSideChannel{
		Exit:    crashCh,
		LogTail: func() []string { return wantLines },
	}

	type outcome struct {
		res *Result
		err error
	}
	done := make(chan outcome, 1)
	go func() {
		res, err := Wait(context.Background(), cfg)
		done <- outcome{res, err}
	}()

	select {
	case <-firstHit:
	case <-time.After(5 * time.Second): // deadlock guard only; the SUT advances on channels, not wall clock
		t.Fatal("first health poll never reached the server")
	}
	crashCh <- struct{}{}

	select {
	case out := <-done:
		if !errors.Is(out.err, ErrCrashed) {
			t.Fatalf("err = %v, want errors.Is(err, ErrCrashed)", out.err)
		}
		if out.res.Verdict != VerdictCrashed {
			t.Errorf("Verdict = %q, want %q", out.res.Verdict, VerdictCrashed)
		}
		if !reflect.DeepEqual(out.res.CrashLines, wantLines) {
			t.Errorf("CrashLines = %v, want %v", out.res.CrashLines, wantLines)
		}
		if out.res.Elapsed >= out.res.Budget {
			t.Errorf("Elapsed = %s with Budget = %s, want the crash to beat the deadline", out.res.Elapsed, out.res.Budget)
		}
	case <-time.After(5 * time.Second): // deadlock guard only
		t.Fatal("Wait did not return after the exit signal fired")
	}
}

// TestReadinessCrashWinsOverDeadlineTick proves deterministic precedence: a
// crash value already queued AND a tick landing exactly on the deadline both
// present — the verdict must be Crashed, never Timeout.
// fak-test:runtime fast est=50ms lane=default
func TestReadinessCrashWinsOverDeadlineTick(t *testing.T) {
	srv := unhealthyServer(t)

	clock, ticker, cfg := harness(t, srv.URL, time.Second, 4*time.Second, 4*time.Second, CacheCold)
	ticker.prime(clock, 4, time.Second) // last tick lands exactly on the 4s deadline
	crashCh := make(chan struct{}, 1)
	crashCh <- struct{}{}
	cfg.Crash = CrashSideChannel{Exit: crashCh}

	res, err := Wait(context.Background(), cfg)
	if !errors.Is(err, ErrCrashed) {
		t.Fatalf("err = %v, want errors.Is(err, ErrCrashed) even though the deadline tick was also due", err)
	}
	if res.Verdict != VerdictCrashed {
		t.Errorf("Verdict = %q, want %q", res.Verdict, VerdictCrashed)
	}
	if res.Attempts != 0 {
		t.Errorf("Attempts = %d, want 0 (crash checked before the first poll)", res.Attempts)
	}
}

// TestReadinessHeartbeatAtEachInterval proves OnHeartbeat fires exactly once
// per surviving tick at the configured interval, and NOT on the terminating
// deadline tick.
// fak-test:runtime fast est=50ms lane=default
func TestReadinessHeartbeatAtEachInterval(t *testing.T) {
	srv := unhealthyServer(t)

	clock, ticker, cfg := harness(t, srv.URL, time.Second, 4*time.Second, 4*time.Second, CacheWarm)
	var beats []time.Duration
	cfg.OnHeartbeat = func(elapsed time.Duration) { beats = append(beats, elapsed) }
	ticker.prime(clock, 8, time.Second)

	res, err := Wait(context.Background(), cfg)
	if !errors.Is(err, ErrTimeout) {
		t.Fatalf("err = %v, want errors.Is(err, ErrTimeout)", err)
	}
	want := []time.Duration{1 * time.Second, 2 * time.Second, 3 * time.Second}
	if !reflect.DeepEqual(beats, want) {
		t.Errorf("heartbeats = %v, want %v (one per interval, none on the deadline tick)", beats, want)
	}
	if res.Verdict != VerdictTimeout {
		t.Errorf("Verdict = %q, want %q", res.Verdict, VerdictTimeout)
	}
}

// TestReadinessDeadlineBudgetPureFunction pins the pure selection contract:
// warm state takes the warm budget, cold state takes the cold budget.
// fak-test:runtime fast est=1ms lane=default
func TestReadinessDeadlineBudgetPureFunction(t *testing.T) {
	b := Budgets{Warm: 30 * time.Second, Cold: 3 * time.Minute}
	cases := []struct {
		state CacheState
		want  time.Duration
	}{
		{CacheWarm, 30 * time.Second},
		{CacheCold, 3 * time.Minute},
	}
	for _, tc := range cases {
		if got := DeadlineBudget(b, tc.state); got != tc.want {
			t.Errorf("DeadlineBudget(%+v, %v) = %s, want %s", b, tc.state, got, tc.want)
		}
	}
}

// TestReadinessBudgetSelectionBothBranches proves Wait applies the probe
// result end to end: a cold probe runs under the cold budget, a warm probe
// under the warm budget, recorded on the Result either way.
// fak-test:runtime fast est=50ms lane=default
func TestReadinessBudgetSelectionBothBranches(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)

	const (
		warm = 30 * time.Second
		cold = 3 * time.Minute
	)
	for _, tc := range []struct {
		name  string
		state CacheState
		want  time.Duration
	}{
		{"cold probe takes cold budget", CacheCold, cold},
		{"warm probe takes warm budget", CacheWarm, warm},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, _, cfg := harness(t, srv.URL, time.Second, warm, cold, tc.state)
			res, err := Wait(context.Background(), cfg)
			if !errors.Is(err, ErrReady) {
				t.Fatalf("err = %v, want errors.Is(err, ErrReady)", err)
			}
			if res.CacheState != tc.state {
				t.Errorf("CacheState = %v, want %v", res.CacheState, tc.state)
			}
			if res.Budget != tc.want {
				t.Errorf("Budget = %s, want %s", res.Budget, tc.want)
			}
		})
	}
}

// TestReadinessContextCancellation proves cancellation aborts the wait with a
// context error that matches NONE of the three verdict sentinels.
// fak-test:runtime fast est=50ms lane=default
func TestReadinessContextCancellation(t *testing.T) {
	srv := unhealthyServer(t)

	t.Run("pre-cancelled context", func(t *testing.T) {
		_, _, cfg := harness(t, srv.URL, time.Second, 10*time.Second, 10*time.Second, CacheWarm)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		res, err := Wait(ctx, cfg)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want errors.Is(err, context.Canceled)", err)
		}
		for _, s := range []error{ErrReady, ErrTimeout, ErrCrashed} {
			if errors.Is(err, s) {
				t.Fatalf("cancellation error must not match verdict sentinel %v", s)
			}
		}
		if res.Verdict != "" {
			t.Errorf("Verdict = %q, want empty (no verdict on cancellation)", res.Verdict)
		}
	})

	t.Run("cancelled mid-wait", func(t *testing.T) {
		firstHit := make(chan struct{})
		var once sync.Once
		srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			once.Do(func() { close(firstHit) })
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		t.Cleanup(srv2.Close)

		_, _, cfg := harness(t, srv2.URL, time.Second, 10*time.Second, 10*time.Second, CacheWarm)
		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan error, 1)
		go func() {
			_, err := Wait(ctx, cfg)
			done <- err
		}()
		select {
		case <-firstHit:
		case <-time.After(5 * time.Second): // deadlock guard only
			t.Fatal("first health poll never reached the server")
		}
		cancel()
		select {
		case err := <-done:
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("err = %v, want errors.Is(err, context.Canceled)", err)
			}
		case <-time.After(5 * time.Second): // deadlock guard only
			t.Fatal("Wait did not return after context cancellation")
		}
	})
}

// TestReadinessRefusesEmptyHealthURL proves a missing URL is refused up front
// with a config error that matches none of the verdict sentinels.
// fak-test:runtime fast est=1ms lane=default
func TestReadinessRefusesEmptyHealthURL(t *testing.T) {
	_, _, cfg := harness(t, "", time.Second, time.Second, time.Second, CacheWarm)
	res, err := Wait(context.Background(), cfg)
	if err == nil {
		t.Fatal("expected a refusal error for an empty health URL")
	}
	if res != nil {
		t.Errorf("Result = %+v, want nil on refusal", res)
	}
	for _, s := range []error{ErrReady, ErrTimeout, ErrCrashed} {
		if errors.Is(err, s) {
			t.Fatalf("config refusal must not match verdict sentinel %v", s)
		}
	}
}
