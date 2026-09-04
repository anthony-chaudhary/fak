// Package breathgate implements rate-limiting, pacing, and cooldown gating
// between tool invocations and autonomous agent turns.
//
// Autonomous agents in multi-turn environments require bounded execution
// characteristics: rate-limiting prevents external API exhaustion, inter-turn
// pacing enforces a minimum breathing room between consecutive turns, and
// cooldown gating handles backoff when downstream providers throttle or error.
//
// # Synchronization and Invariants
//
// All methods on Gate are thread-safe and safe for concurrent invocation by
// multiple worker goroutines. Concurrency is guarded by mutual exclusion and
// channel-based waiter notification.
//
// Invariant: activeCount >= 0 && activeCount <= config.MaxConcurrent
// Invariant: lastAcquire is monotonic non-decreasing across admissions
// Invariant: cooldownUntil represents the earliest timestamp next admission is permitted
// Guard: fail-closed admission: nil context, canceled context, or closed state denies entry.
package breathgate

import (
	"context"
	"errors"
	"sync"
	"time"
)

// ErrNilContext is returned when Acquire is invoked with a nil context.
// Guard: fail-closed admission: nil context is rejected immediately without state mutation.
var ErrNilContext = errors.New("breathgate: context must not be nil")

// ErrClosed is returned when an operation is attempted on a closed Gate.
// Guard: fail-closed admission: closed gate denies all subsequent acquisitions.
var ErrClosed = errors.New("breathgate: gate is closed")

// Config defines operational parameters for Gate concurrency, pacing, and cooldown backoff.
type Config struct {
	// MaxConcurrent is the maximum number of concurrent requests or agent turns permitted.
	// Values <= 0 default to 1 (strict serialization).
	MaxConcurrent int `json:"max_concurrent"`

	// MinInterval is the minimum required pacing duration between consecutive turn starts.
	// Values <= 0 disable inter-turn pacing.
	MinInterval time.Duration `json:"min_interval"`

	// BackoffOnThrottle is the cooldown duration applied when throttling or when
	// Cooldown is invoked with a non-positive duration.
	// Values <= 0 default to 100 Milliseconds.
	BackoffOnThrottle time.Duration `json:"backoff_on_throttle"`
}

// DefaultConfig returns safe baseline configuration values:
// MaxConcurrent=1, MinInterval=50ms, BackoffOnThrottle=100ms.
func DefaultConfig() Config {
	return Config{
		MaxConcurrent:     1,
		MinInterval:       50 * time.Millisecond,
		BackoffOnThrottle: 100 * time.Millisecond,
	}
}

// Metrics holds point-in-time observational counters and status for Gate telemetry.
type Metrics struct {
	// TotalAcquires is the cumulative count of successful admissions through the gate.
	TotalAcquires int64 `json:"total_acquires"`

	// TotalReleases is the cumulative count of slots released back to the gate.
	TotalReleases int64 `json:"total_releases"`

	// ActiveHolders is the current count of concurrent admissions actively held.
	ActiveHolders int64 `json:"active_holders"`

	// Throttles is the number of times an acquisition was delayed by pacing, limits, or cooldown.
	Throttles int64 `json:"throttles"`

	// Cooldowns is the number of cooldown periods enforced on the gate.
	Cooldowns int64 `json:"cooldowns"`

	// TotalWaitDuration accumulates total time callers spent blocked waiting in Acquire.
	TotalWaitDuration time.Duration `json:"total_wait_duration"`

	// LastAcquire is the timestamp of the most recent successful acquisition.
	LastAcquire time.Time `json:"last_acquire"`

	// CooldownUntil is the expiration timestamp of any active cooldown window.
	CooldownUntil time.Time `json:"cooldown_until"`
}

// Gate coordinates rate-limiting, pacing, and cooldown gating across tool calls and agent turns.
//
// Invariant: activeCount >= 0 && activeCount <= config.MaxConcurrent
// Invariant: lastAcquire is monotonic non-decreasing across admissions
// Invariant: cooldownUntil represents the earliest timestamp the next admission is permitted
// Guard: fail-closed admission: invalid context, canceled context, or closed state denies entry.
type Gate struct {
	mu            sync.Mutex
	config        Config
	activeCount   int
	lastAcquire   time.Time
	cooldownUntil time.Time
	waiters       []chan struct{}
	metrics       Metrics
	closed        bool
	now           func() time.Time
}

// New creates a new Gate initialized with the supplied configuration.
// If non-positive configuration values are supplied for MaxConcurrent or BackoffOnThrottle,
// safe defaults are applied.
func New(cfg Config) *Gate {
	if cfg.MaxConcurrent <= 0 {
		cfg.MaxConcurrent = 1
	}
	if cfg.BackoffOnThrottle <= 0 {
		cfg.BackoffOnThrottle = 100 * time.Millisecond
	}
	if cfg.MinInterval < 0 {
		cfg.MinInterval = 0
	}

	return &Gate{
		config: cfg,
	}
}

// Acquire blocks until a concurrency slot is available, inter-turn pacing is satisfied,
// and any active cooldown has expired, or until ctx is canceled.
//
// Invariant: upon successful return, activeCount <= config.MaxConcurrent.
// Guard: fail-closed admission: nil or canceled context immediately denies admission without side effects.
func (g *Gate) Acquire(ctx context.Context) error {
	// Guard: fail-closed admission
	if ctx == nil {
		return ErrNilContext
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	start := g.timeNow()
	throttled := false

	for {
		g.mu.Lock()
		if g.closed {
			g.mu.Unlock()
			// Guard: fail-closed admission
			return ErrClosed
		}

		now := g.timeNow()

		// 1. Evaluate active cooldown.
		var waitDur time.Duration
		if now.Before(g.cooldownUntil) {
			waitDur = g.cooldownUntil.Sub(now)
		}

		// 2. Evaluate minimum interval pacing requirement.
		if g.config.MinInterval > 0 && !g.lastAcquire.IsZero() {
			elapsed := now.Sub(g.lastAcquire)
			if elapsed < g.config.MinInterval {
				intervalWait := g.config.MinInterval - elapsed
				if intervalWait > waitDur {
					waitDur = intervalWait
				}
			}
		}

		// 3. Evaluate concurrency limits.
		concurrencyFull := g.activeCount >= g.config.MaxConcurrent

		if waitDur <= 0 && !concurrencyFull {
			// Invariant: activeCount < config.MaxConcurrent
			g.activeCount++
			g.lastAcquire = now
			g.metrics.TotalAcquires++
			g.metrics.ActiveHolders = int64(g.activeCount)
			g.metrics.LastAcquire = now
			if throttled {
				g.metrics.TotalWaitDuration += g.timeNow().Sub(start)
			}
			g.mu.Unlock()
			return nil
		}

		if !throttled {
			throttled = true
			g.metrics.Throttles++
		}

		ch := make(chan struct{}, 1)
		g.waiters = append(g.waiters, ch)
		g.mu.Unlock()

		var timer *time.Timer
		var timerCh <-chan time.Time
		if waitDur > 0 {
			timer = time.NewTimer(waitDur)
			timerCh = timer.C
		}

		select {
		case <-ctx.Done():
			if timer != nil {
				if !timer.Stop() {
					select {
					case <-timer.C:
					default:
					}
				}
			}
			g.mu.Lock()
			g.removeWaiterLocked(ch)
			if throttled {
				g.metrics.TotalWaitDuration += g.timeNow().Sub(start)
			}
			g.mu.Unlock()
			// Guard: fail-closed admission
			return ctx.Err()

		case <-timerCh:
			g.mu.Lock()
			g.removeWaiterLocked(ch)
			g.mu.Unlock()

		case <-ch:
			if timer != nil {
				if !timer.Stop() {
					select {
					case <-timer.C:
					default:
					}
				}
			}
		}
	}
}

// Release relinquishes one active concurrency slot and notifies waiting callers.
// If Release is called when activeCount is 0, it safely no-ops to prevent counter underflow.
//
// Invariant: activeCount >= 0 && activeCount <= config.MaxConcurrent
func (g *Gate) Release() {
	g.mu.Lock()
	defer g.mu.Unlock()

	// Invariant: activeCount >= 0
	if g.activeCount <= 0 {
		return
	}
	g.activeCount--
	g.metrics.TotalReleases++
	g.metrics.ActiveHolders = int64(g.activeCount)
	g.broadcastLocked()
}

// Cooldown enforces a quiet period of duration d during which subsequent Acquire calls will block.
// If d <= 0, the configured BackoffOnThrottle duration is applied.
// If an existing cooldown extends beyond the newly specified duration, the later expiration is preserved.
//
// Guard: fail-closed admission: during cooldown, all acquisition attempts are delayed until expiration.
func (g *Gate) Cooldown(d time.Duration) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if d <= 0 {
		d = g.config.BackoffOnThrottle
	}
	if d <= 0 {
		return
	}

	now := g.timeNow()
	until := now.Add(d)
	if until.After(g.cooldownUntil) {
		g.cooldownUntil = until
	}
	g.metrics.Cooldowns++
	g.metrics.CooldownUntil = g.cooldownUntil
}

// Throttle triggers a cooldown period using the configured BackoffOnThrottle duration.
func (g *Gate) Throttle() {
	g.Cooldown(0)
}

// Metrics returns a point-in-time snapshot copy of the gate's observational telemetry.
func (g *Gate) Metrics() Metrics {
	g.mu.Lock()
	defer g.mu.Unlock()

	m := g.metrics
	m.ActiveHolders = int64(g.activeCount)
	m.CooldownUntil = g.cooldownUntil
	return m
}

// IsCoolingDown reports whether the gate is currently within an active cooldown window.
func (g *Gate) IsCoolingDown() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.timeNow().Before(g.cooldownUntil)
}

// Config returns a copy of the gate's operational configuration.
func (g *Gate) Config() Config {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.config
}

// Close permanently closes the gate. Any ongoing or future Acquire calls
// will immediately return ErrClosed.
//
// Guard: fail-closed admission: closed gates reject all admissions.
func (g *Gate) Close() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.closed {
		return nil
	}
	g.closed = true
	g.broadcastLocked()
	return nil
}

// Reset restores the gate to an idle unthrottled state, clearing active cooldowns
// and pacing timestamps, and waking waiting callers.
func (g *Gate) Reset() {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.cooldownUntil = time.Time{}
	g.lastAcquire = time.Time{}
	g.broadcastLocked()
}

func (g *Gate) broadcastLocked() {
	for _, ch := range g.waiters {
		select {
		case ch <- struct{}{}:
		default:
		}
	}
	g.waiters = nil
}

func (g *Gate) removeWaiterLocked(target chan struct{}) {
	for i, ch := range g.waiters {
		if ch == target {
			g.waiters = append(g.waiters[:i], g.waiters[i+1:]...)
			return
		}
	}
}

func (g *Gate) timeNow() time.Time {
	if g.now != nil {
		return g.now()
	}
	return time.Now()
}
