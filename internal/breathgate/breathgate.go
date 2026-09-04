// Package breathgate provides turn pacing, pause control, debounce, and cooldown
// mechanisms for autonomous agent loops.
//
// In multi-turn agent execution, runaway loops can rapidly exhaust API rate limits,
// deplete token spend, or cause destructive thrashing. A breath gate introduces
// deliberate "breathing room" between autonomous operations, enforcing minimum
// intervals, bounded jitter to avoid thundering-herd synchronization, and automatic
// or explicit cooldown periods.
//
// All methods on Gate are safe for concurrent use by multiple goroutines.
package breathgate

import (
	"context"
	"math/rand/v2"
	"sync"
	"time"
)

// Config specifies pacing intervals, jitter factor, and cooldown behavior for a Gate.
type Config struct {
	// MinInterval is the minimum pause required between consecutive turns.
	// If non-positive, no pacing interval is enforced between turns.
	MinInterval time.Duration

	// MaxInterval is the upper bound for jittered or adaptive pacing intervals.
	// If non-positive or less than MinInterval, MinInterval is treated as the ceiling.
	MaxInterval time.Duration

	// Cooldown is the duration of an enforced pause when a burst limit is exceeded
	// or an explicit cooldown is requested.
	Cooldown time.Duration

	// Jitter is a scaling coefficient in the range [0.0, 1.0] used to add
	// pseudo-random variance to pacing intervals, preventing thundering herds.
	// A value of 0.10 adds up to 10% additional random delay to MinInterval.
	Jitter float64

	// BurstLimit is the maximum number of rapid turns allowed within a burst window
	// before triggering an automatic Cooldown. If non-positive, burst detection is disabled.
	BurstLimit int

	// BurstWindow is the maximum elapsed duration between turns to be considered
	// consecutive for burst limit tracking. If non-positive, defaults to
	// 2 * max(MinInterval, MaxInterval).
	BurstWindow time.Duration

	// Clock is an optional custom time provider (defaults to time.Now if nil).
	Clock func() time.Time

	// RNG is an optional pseudo-random generator returning a float64 in [0.0, 1.0).
	// Defaults to rand.Float64 if nil.
	RNG func() float64

	// Sleep is an optional context-aware sleeper function (defaults to timer-based wait if nil).
	Sleep func(ctx context.Context, d time.Duration) error
}

// DefaultConfig returns a recommended production configuration with sensible
// pacing defaults: 100ms minimum interval, 500ms max interval, 5s cooldown,
// 10% jitter, and a burst limit of 20 turns.
func DefaultConfig() Config {
	return Config{
		MinInterval: 100 * time.Millisecond,
		MaxInterval: 500 * time.Millisecond,
		Cooldown:    5 * time.Second,
		Jitter:      0.10,
		BurstLimit:  20,
	}
}

// Stats holds an immutable snapshot of operational metrics for a Gate.
type Stats struct {
	// TotalTurns is the cumulative number of turns recorded by the gate.
	TotalTurns int64

	// ThrottledTurns is the count of turns that experienced a non-zero wait delay.
	ThrottledTurns int64

	// CooldownCount is the number of times the gate has entered a cooldown state.
	CooldownCount int64

	// TotalWaitTime is the cumulative time callers spent waiting in Wait.
	TotalWaitTime time.Duration

	// LastTurn is the timestamp when the most recent turn was recorded.
	LastTurn time.Time

	// InCooldown indicates whether the gate is actively in a cooldown period.
	InCooldown bool

	// ConsecutiveTurns is the current run of consecutive turns within the burst window.
	ConsecutiveTurns int
}

// Gate manages pacing intervals, pause control, debounce, and cooldowns.
// It enforces minimum breathing intervals between autonomous tool turns,
// serializes bursts, and protects against runaway thrashing.
//
// Contract:
//   - Safe for concurrent use across multiple goroutines.
//   - Check() is non-blocking and reports whether a turn can run immediately.
//   - Wait(ctx) blocks until the pacing interval and cooldown have elapsed, or ctx cancels.
//   - RecordTurn() logs turn completion and triggers cooldown if burst limits are exceeded.
//   - Reset() clears all cooldowns and throttling, returning the gate to initial ready state.
type Gate struct {
	mu sync.Mutex

	cfg Config

	lastTurn         time.Time
	nextAllowed      time.Time
	cooldownUntil    time.Time
	consecutiveTurns int

	totalTurns     int64
	throttledTurns int64
	cooldownCount  int64
	totalWaitTime  time.Duration

	nowFn   func() time.Time
	rngFn   func() float64
	sleepFn func(ctx context.Context, d time.Duration) error
}

// New constructs a new Gate with the provided Config. Negative intervals and
// out-of-range jitter factors are automatically sanitized to safe bounds.
func New(cfg Config) *Gate {
	if cfg.MinInterval < 0 {
		cfg.MinInterval = 0
	}
	if cfg.MaxInterval < 0 {
		cfg.MaxInterval = 0
	}
	if cfg.MaxInterval > 0 && cfg.MaxInterval < cfg.MinInterval {
		cfg.MaxInterval = cfg.MinInterval
	}
	if cfg.Cooldown < 0 {
		cfg.Cooldown = 0
	}
	if cfg.Jitter < 0 {
		cfg.Jitter = 0
	}
	if cfg.Jitter > 1.0 {
		cfg.Jitter = 1.0
	}
	if cfg.BurstLimit < 0 {
		cfg.BurstLimit = 0
	}
	if cfg.BurstWindow <= 0 {
		window := cfg.MaxInterval
		if cfg.MinInterval > window {
			window = cfg.MinInterval
		}
		cfg.BurstWindow = 2 * window
		if cfg.BurstWindow <= 0 {
			cfg.BurstWindow = time.Second
		}
	}

	g := &Gate{
		cfg:     cfg,
		nowFn:   cfg.Clock,
		rngFn:   cfg.RNG,
		sleepFn: cfg.Sleep,
	}

	if g.nowFn == nil {
		g.nowFn = time.Now
	}
	if g.rngFn == nil {
		g.rngFn = rand.Float64
	}
	if g.sleepFn == nil {
		g.sleepFn = defaultSleep
	}

	return g
}

// defaultSleep waits for duration d or until ctx is canceled.
func defaultSleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// now returns the current time using the configured clock.
func (g *Gate) now() time.Time {
	return g.nowFn()
}

// computeInterval calculates the required interval including jitter, clamped to MaxInterval.
// Must be called with g.mu held or on immutable config.
func (g *Gate) computeInterval() time.Duration {
	base := g.cfg.MinInterval
	if base <= 0 {
		return 0
	}
	if g.cfg.Jitter <= 0 {
		return base
	}

	delta := time.Duration(float64(base) * g.cfg.Jitter * g.rngFn())
	total := base + delta
	if g.cfg.MaxInterval > 0 && total > g.cfg.MaxInterval {
		return g.cfg.MaxInterval
	}
	return total
}

// Check reports whether the gate is ready to admit the next turn immediately
// without waiting. It returns true if both the pacing interval and any active
// cooldown have elapsed. Check does not block and does not modify turn counters.
func (g *Gate) Check() bool {
	g.mu.Lock()
	defer g.mu.Unlock()

	now := g.now()
	if now.Before(g.cooldownUntil) {
		return false
	}
	if now.Before(g.nextAllowed) {
		return false
	}
	return true
}

// Remaining returns the duration until the gate will be ready to admit the next turn.
// Returns 0 if the gate is ready immediately.
func (g *Gate) Remaining() time.Duration {
	g.mu.Lock()
	defer g.mu.Unlock()

	now := g.now()
	var d time.Duration
	if now.Before(g.cooldownUntil) {
		d = g.cooldownUntil.Sub(now)
	}
	if now.Before(g.nextAllowed) {
		pacingDelay := g.nextAllowed.Sub(now)
		if pacingDelay > d {
			d = pacingDelay
		}
	}
	return d
}

// Wait blocks until the required pacing interval and any active cooldown period
// have elapsed, or until the provided Context is canceled. Concurrent callers are
// serialized and scheduled in FIFO order to prevent thundering herds.
//
// Returns nil on successful wait or immediate admission, or ctx.Err() if canceled.
func (g *Gate) Wait(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	g.mu.Lock()
	now := g.now()

	// Determine target time: maximum of now, cooldown, and nextAllowed.
	target := now
	if g.cooldownUntil.After(target) {
		target = g.cooldownUntil
	}
	if g.nextAllowed.After(target) {
		target = g.nextAllowed
	}

	var d time.Duration
	if target.After(now) {
		d = target.Sub(now)
	}

	// Reserve the next pacing slot for following callers.
	interval := g.computeInterval()
	g.nextAllowed = target.Add(interval)

	if d <= 0 {
		g.mu.Unlock()
		return nil
	}

	g.mu.Unlock()

	// Sleep until the allocated turn time arrives.
	if err := g.sleepFn(ctx, d); err != nil {
		return err
	}

	g.mu.Lock()
	g.throttledTurns++
	g.totalWaitTime += d
	g.mu.Unlock()

	return nil
}

// RecordTurn logs the completion of a turn, updating timestamps and evaluating
// burst limits. If the number of rapid consecutive turns meets or exceeds BurstLimit,
// an automatic Cooldown is triggered.
func (g *Gate) RecordTurn() {
	g.mu.Lock()
	defer g.mu.Unlock()

	now := g.now()
	g.totalTurns++

	// Evaluate burst tracking if BurstLimit is configured.
	if g.cfg.BurstLimit > 0 {
		if !g.lastTurn.IsZero() && now.Sub(g.lastTurn) <= g.cfg.BurstWindow {
			g.consecutiveTurns++
		} else {
			g.consecutiveTurns = 1
		}

		if g.consecutiveTurns >= g.cfg.BurstLimit {
			g.triggerCooldownLocked(now, g.cfg.Cooldown)
			g.lastTurn = now
			return
		}
	}

	g.lastTurn = now

	// Ensure the next turn obeys the pacing interval relative to turn completion.
	interval := g.computeInterval()
	targetNext := now.Add(interval)
	if targetNext.After(g.nextAllowed) {
		g.nextAllowed = targetNext
	}
}

// triggerCooldownLocked arms cooldown until now + d and resets burst counters.
// Must be called with g.mu held.
func (g *Gate) triggerCooldownLocked(now time.Time, d time.Duration) {
	if d <= 0 {
		return
	}
	g.cooldownUntil = now.Add(d)
	g.cooldownCount++
	g.consecutiveTurns = 0
	if g.cooldownUntil.After(g.nextAllowed) {
		g.nextAllowed = g.cooldownUntil
	}
}

// TriggerCooldown forces the gate into an active cooldown state for the duration
// configured in Config.Cooldown.
func (g *Gate) TriggerCooldown() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.triggerCooldownLocked(g.now(), g.cfg.Cooldown)
}

// TriggerCooldownFor forces the gate into an active cooldown state for the specified
// duration. Useful when handling external API rate limit responses such as HTTP 429
// Retry-After headers.
func (g *Gate) TriggerCooldownFor(d time.Duration) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.triggerCooldownLocked(g.now(), d)
}

// Reset clears all cooldowns, pacing reservations, and consecutive turn counters,
// immediately restoring the gate to its initial ready state.
func (g *Gate) Reset() {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.lastTurn = time.Time{}
	g.nextAllowed = time.Time{}
	g.cooldownUntil = time.Time{}
	g.consecutiveTurns = 0
}

// Stats returns an atomic snapshot of current gate metrics and operational status.
func (g *Gate) Stats() Stats {
	g.mu.Lock()
	defer g.mu.Unlock()

	now := g.now()
	return Stats{
		TotalTurns:       g.totalTurns,
		ThrottledTurns:   g.throttledTurns,
		CooldownCount:    g.cooldownCount,
		TotalWaitTime:    g.totalWaitTime,
		LastTurn:         g.lastTurn,
		InCooldown:       now.Before(g.cooldownUntil),
		ConsecutiveTurns: g.consecutiveTurns,
	}
}

// Config returns a copy of the configuration parameters governing this Gate.
func (g *Gate) Config() Config {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.cfg
}
