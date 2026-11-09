// Package servelaunch is the readiness-probe composite for benchmark-wave
// server launches: after a bench harness starts its server process, this
// package waits until the health URL answers, while staying honest about what
// actually happened if it never does.
//
// The composite races three outcomes against each other:
//   - Ready:    the health URL answers 2xx within the deadline budget.
//   - Timeout:  the deadline budget passes with no healthy answer.
//   - Crashed:  the caller-supplied crash side-channel (process-exit signal
//     plus last-log-lines fetcher) fires first, so a fast crash returns in
//     seconds instead of riding the full timeout.
//
// The deadline budget itself is chosen from a caller-supplied cache-state
// probe: cold state (weights not resident, first load dominates startup) gets
// the long cold budget, warm state gets the short warm budget. The selection
// is a pure function over the probe result (DeadlineBudget), so both branches
// are directly testable without any clock or network.
//
// Everything time-ordered goes through an injectable Clock and Ticker seam:
// the Clock supplies the wait's origin, and the loop tracks "now" from the
// timestamps the ticker delivers (exactly what time.NewTicker delivers in
// production). Tests drive fake ticks by hand, and no code path here ever
// sleeps. Elapsed figures are therefore reported at tick granularity.
package servelaunch

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Closed outcome vocabulary and their sentinel errors. Callers branch on the
// sentinels with errors.Is, never on error prose: the prose is free to change,
// the identity is the contract.
const (
	// VerdictReady reports a healthy answer inside the deadline budget.
	VerdictReady Verdict = "ready"
	// VerdictTimeout reports the deadline budget passing with no healthy answer.
	VerdictTimeout Verdict = "timeout"
	// VerdictCrashed reports the launched process exiting before readiness.
	VerdictCrashed Verdict = "crashed"
)

// Verdict is the closed outcome vocabulary of a readiness wait.
type Verdict string

var (
	// ErrReady is the sentinel for the Ready verdict.
	ErrReady = errors.New("servelaunch: ready")
	// ErrTimeout is the sentinel for the Timeout verdict.
	ErrTimeout = errors.New("servelaunch: readiness deadline exceeded")
	// ErrCrashed is the sentinel for the Crashed verdict.
	ErrCrashed = errors.New("servelaunch: server process exited during readiness wait")
)

// CacheState is the closed vocabulary the cache-state probe reports.
type CacheState int

const (
	// CacheCold: weights are not cache-resident; the first load dominates
	// startup, so the wait deserves the long budget.
	CacheCold CacheState = iota
	// CacheWarm: weights are cache-resident; startup is short, so the wait
	// gets the short budget and a hang is suspect sooner.
	CacheWarm
)

// CacheStateProbe is the caller-supplied cache-state probe. A nil probe means
// "unknown", which Wait treats as cold: patience over premature timeout.
type CacheStateProbe func() CacheState

// Budgets pairs the two deadline budgets the wait may run under.
type Budgets struct {
	Warm time.Duration
	Cold time.Duration
}

// DeadlineBudget is the pure selection: warm state takes the warm budget,
// everything else (cold, unknown) takes the cold budget. No I/O, no clock —
// the only input is the probe result, so both branches are trivially tested.
func DeadlineBudget(b Budgets, state CacheState) time.Duration {
	if state == CacheWarm {
		return b.Warm
	}
	return b.Cold
}

// CrashSideChannel is the caller-supplied observation of the launched
// process's fate. Exit receives one value when the process goes away (closing
// it also counts as firing); LogTail fetches its last log lines for the crash
// evidence. A nil Exit disables crash detection entirely: the wait then runs
// to Ready or Timeout alone.
type CrashSideChannel struct {
	Exit    <-chan struct{}
	LogTail func() []string
}

// Clock is the injectable time source that supplies the wait's origin. The
// loop tracks subsequent time from the timestamps its ticker delivers, so a
// test clock only needs a fixed origin — no advancing, no sleeping.
type Clock interface {
	Now() time.Time
}

type realClock struct{}

func (realClock) Now() time.Time { return time.Now() }

// Ticker is the minimal interval-tick surface the wait loop drives.
type Ticker interface {
	Channel() <-chan time.Time
	Stop()
}

// TickerFactory builds the interval ticker. Production adapts time.NewTicker;
// tests return a fake whose channel they feed by hand.
type TickerFactory func(d time.Duration) Ticker

type stdTicker struct{ t *time.Ticker }

// DefaultTickerFactory adapts time.NewTicker to the Ticker seam.
func DefaultTickerFactory(d time.Duration) Ticker { return &stdTicker{t: time.NewTicker(d)} }

func (s *stdTicker) Channel() <-chan time.Time { return s.t.C }
func (s *stdTicker) Stop()                     { s.t.Stop() }

// Config bounds one readiness wait. Zero fields fall back to the defaults
// below; Clock and NewTicker exist purely so tests can drive time by hand.
type Config struct {
	// HealthURL is polled with GET; a 2xx status means ready.
	HealthURL string
	// Client performs the health requests. Nil uses a package default.
	Client *http.Client
	// Interval between polls and heartbeat cadence.
	Interval time.Duration
	// Budgets supplies the warm/cold deadline budgets.
	Budgets Budgets
	// CacheState picks the budget via DeadlineBudget. Nil means cold.
	CacheState CacheStateProbe
	// Crash is the process-exit side-channel. Zero value disables it.
	Crash CrashSideChannel
	// OnHeartbeat, if non-nil, is invoked once per surviving tick with the
	// elapsed time since the wait started.
	OnHeartbeat func(elapsed time.Duration)
	// Clock and NewTicker are the injectable time seams.
	Clock     Clock
	NewTicker TickerFactory
}

// Defaults keep a zero-ish Config usable without a caller thinking about it.
const (
	// DefaultInterval is the poll/heartbeat cadence when Config.Interval <= 0.
	DefaultInterval = 250 * time.Millisecond
	// DefaultWarmBudget bounds a warm-cache launch.
	DefaultWarmBudget = 30 * time.Second
	// DefaultColdBudget bounds a cold-cache launch (first model load included).
	DefaultColdBudget = 3 * time.Minute
	// defaultAttemptTimeout bounds one individual health request.
	defaultAttemptTimeout = 2 * time.Second
)

var defaultClient = &http.Client{}

// Result is the machine-readable readiness outcome. Verdict and Budget record
// which branch ran; Attempts and Elapsed are the wait's own accounting;
// CrashLines carries the fetched last-log lines and is populated only for
// VerdictCrashed.
type Result struct {
	Verdict    Verdict
	CacheState CacheState
	Budget     time.Duration
	Attempts   int
	Elapsed    time.Duration
	CrashLines []string
}

// Wait polls HealthURL at Config.Interval until it answers 2xx, the selected
// deadline budget passes, or the crash side-channel fires. It returns a
// Result plus an error whose identity is exactly one of the three sentinels:
// branch with errors.Is(err, ErrReady | ErrTimeout | ErrCrashed), never prose.
//
// Ordering guarantees: the crash side-channel is checked before every poll and
// again before every deadline evaluation, so a fired crash always beats both
// the next poll and the deadline — a fast crash returns Crashed in seconds,
// never Timeout. Context cancellation aborts the wait and wraps ctx.Err();
// that error matches none of the three sentinels.
func Wait(ctx context.Context, cfg Config) (*Result, error) {
	if strings.TrimSpace(cfg.HealthURL) == "" {
		return nil, errors.New("servelaunch: health URL is required")
	}
	cfg = normalize(cfg)

	state := CacheCold
	if cfg.CacheState != nil {
		state = cfg.CacheState()
	}
	budget := DeadlineBudget(cfg.Budgets, state)

	start := cfg.Clock.Now()
	deadline := start.Add(budget)
	now := start
	res := &Result{CacheState: state, Budget: budget}

	ticker := cfg.NewTicker(cfg.Interval)
	defer ticker.Stop()

	// A nil channel blocks forever in a select, so the same loop body serves
	// both "side-channel armed" and "side-channel disabled".
	crashCh := cfg.Crash.Exit
	crashFired := func() bool {
		if crashCh == nil {
			return false
		}
		select {
		case <-crashCh:
			return true
		default:
			return false
		}
	}

	for {
		if err := ctx.Err(); err != nil {
			return res, fmt.Errorf("servelaunch: wait aborted: %w", err)
		}
		if crashFired() {
			return crashResult(res, cfg, now.Sub(start))
		}
		res.Attempts++
		if err := probeHealth(ctx, cfg); err == nil {
			res.Verdict = VerdictReady
			res.Elapsed = now.Sub(start)
			return res, fmt.Errorf("%w: %d attempt(s) in %s", ErrReady, res.Attempts, res.Elapsed.Round(time.Millisecond))
		}
		select {
		case <-crashCh:
			return crashResult(res, cfg, now.Sub(start))
		case tv := <-ticker.Channel():
			// Re-check before the deadline evaluation so a crash that lands on
			// the same tick still wins the race.
			if crashFired() {
				return crashResult(res, cfg, now.Sub(start))
			}
			now = tv
			if !now.Before(deadline) {
				res.Verdict = VerdictTimeout
				res.Elapsed = now.Sub(start)
				return res, fmt.Errorf("%w: %d attempt(s) in %s", ErrTimeout, res.Attempts, res.Elapsed.Round(time.Millisecond))
			}
			if cfg.OnHeartbeat != nil {
				cfg.OnHeartbeat(now.Sub(start))
			}
		case <-ctx.Done():
			return res, fmt.Errorf("servelaunch: wait aborted: %w", ctx.Err())
		}
	}
}

// crashResult fills the Crashed outcome, including the fetched last-log lines.
func crashResult(res *Result, cfg Config, elapsed time.Duration) (*Result, error) {
	res.Verdict = VerdictCrashed
	res.Elapsed = elapsed
	if cfg.Crash.LogTail != nil {
		res.CrashLines = append([]string(nil), cfg.Crash.LogTail()...)
	}
	return res, fmt.Errorf("%w: %d attempt(s) in %s", ErrCrashed, res.Attempts, res.Elapsed.Round(time.Millisecond))
}

// probeHealth performs one bounded GET; 2xx is success, anything else (refused
// connection, 4xx, 5xx) means "not ready yet".
func probeHealth(ctx context.Context, cfg Config) error {
	reqCtx, cancel := context.WithTimeout(ctx, defaultAttemptTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, cfg.HealthURL, nil)
	if err != nil {
		return err
	}
	client := cfg.Client
	if client == nil {
		client = defaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4<<10))
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return fmt.Errorf("health endpoint answered with status %d", resp.StatusCode)
	}
	return nil
}

func normalize(cfg Config) Config {
	if cfg.Interval <= 0 {
		cfg.Interval = DefaultInterval
	}
	if cfg.Budgets.Warm <= 0 {
		cfg.Budgets.Warm = DefaultWarmBudget
	}
	if cfg.Budgets.Cold <= 0 {
		cfg.Budgets.Cold = DefaultColdBudget
	}
	if cfg.Clock == nil {
		cfg.Clock = realClock{}
	}
	if cfg.NewTicker == nil {
		cfg.NewTicker = DefaultTickerFactory
	}
	return cfg
}
