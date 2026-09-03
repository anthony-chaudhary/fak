package radixkv

import (
	"context"
	"errors"
	"sync"
	"time"
)

// BreakerState represents the operational state of the RemoteL3Breaker.
type BreakerState string

const (
	BreakerClosed   BreakerState = "closed"
	BreakerOpen     BreakerState = "open"
	BreakerHalfOpen BreakerState = "half_open"
)

func (s BreakerState) String() string {
	switch s {
	case BreakerOpen:
		return "open"
	case BreakerHalfOpen:
		return "half_open"
	default:
		return "closed"
	}
}

const (
	DefaultBreakerFaultThreshold = 5
	DefaultBreakerCooldown       = 30 * time.Second
)

// BreakerConfig configures the failure threshold and recovery cooldown for RemoteL3Breaker.
type BreakerConfig struct {
	FaultThreshold int              `json:"fault_threshold"`
	Cooldown       time.Duration    `json:"cooldown"`
	Now            func() time.Time `json:"-"`
}

// DefaultBreakerConfig returns standard defaults for the remote L3 circuit breaker.
func DefaultBreakerConfig() BreakerConfig {
	return BreakerConfig{
		FaultThreshold: DefaultBreakerFaultThreshold,
		Cooldown:       DefaultBreakerCooldown,
	}
}

// BreakerStats captures a point-in-time snapshot of breaker metrics.
type BreakerStats struct {
	State             BreakerState  `json:"state"`
	ConsecutiveFaults int           `json:"consecutive_faults"`
	TotalFaults       int           `json:"total_faults"`
	OpenSkips         int           `json:"open_skips"`
	ProbesAttempted   int           `json:"probes_attempted"`
	ProbeRecoveries   int           `json:"probe_recoveries"`
	ProbeFailures     int           `json:"probe_failures"`
	OpenedAt          time.Time     `json:"opened_at,omitempty"`
	FaultThreshold    int           `json:"fault_threshold"`
	Cooldown          time.Duration `json:"cooldown"`
}

// RemoteL3Breaker provides bounded circuit breaking over optional remote-L3 snapshot reads.
// It prevents fault storms from degrading local prefix lookups when the external store is down.
type RemoteL3Breaker struct {
	mu sync.Mutex

	cfg   BreakerConfig
	now   func() time.Time
	state BreakerState

	consecutiveFaults int
	totalFaults       int
	openSkips         int
	probesAttempted   int
	probeRecoveries   int
	probeFailures     int
	openedAt          time.Time
}

// NewRemoteL3Breaker constructs a breaker with the supplied configuration.
func NewRemoteL3Breaker(cfg BreakerConfig) *RemoteL3Breaker {
	if cfg.FaultThreshold <= 0 {
		cfg.FaultThreshold = DefaultBreakerFaultThreshold
	}
	if cfg.Cooldown <= 0 {
		cfg.Cooldown = DefaultBreakerCooldown
	}
	nowFn := cfg.Now
	if nowFn == nil {
		nowFn = time.Now
	}
	return &RemoteL3Breaker{
		cfg:   cfg,
		now:   nowFn,
		state: BreakerClosed,
	}
}

// SetClock injects a deterministic time source for testing.
func (b *RemoteL3Breaker) SetClock(now func() time.Time) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if now == nil {
		b.now = time.Now
	} else {
		b.now = now
	}
}

func (b *RemoteL3Breaker) nowLocked() time.Time {
	if b.now != nil {
		return b.now()
	}
	return time.Now()
}

func (b *RemoteL3Breaker) cooldownLocked() time.Duration {
	if b.cfg.Cooldown > 0 {
		return b.cfg.Cooldown
	}
	return DefaultBreakerCooldown
}

func (b *RemoteL3Breaker) faultThresholdLocked() int {
	if b.cfg.FaultThreshold > 0 {
		return b.cfg.FaultThreshold
	}
	return DefaultBreakerFaultThreshold
}

// Allow decides whether an incoming remote read may proceed.
// Returns (allowed, isProbe).
//   - If Closed: returns true, false.
//   - If Open: if time.Since(openedAt) >= Cooldown: transitions to HalfOpen, admits
//     exactly 1 probe (returns true, true). Other concurrent callers get false, false.
//   - If HalfOpen: probe is in flight, other callers get false, false.
func (b *RemoteL3Breaker) Allow() (bool, bool) {
	if b == nil {
		return true, false
	}
	b.mu.Lock()
	defer b.mu.Unlock()

	state := b.state
	if state == "" {
		state = BreakerClosed
	}

	switch state {
	case BreakerClosed:
		return true, false

	case BreakerOpen:
		now := b.nowLocked()
		cooldown := b.cooldownLocked()
		if !b.openedAt.IsZero() && now.Sub(b.openedAt) >= cooldown {
			b.state = BreakerHalfOpen
			b.probesAttempted++
			return true, true
		}
		b.openSkips++
		return false, false

	case BreakerHalfOpen:
		b.openSkips++
		return false, false

	default:
		return true, false
	}
}

// RecordResult records the outcome of a remote read:
//   - If err is nil: if HalfOpen -> transitions to Closed, resets consecutive faults,
//     increments probe recoveries. If Closed -> resets consecutive faults.
//   - If err is context.Canceled: ignored (caller cancellation is not a backend fault).
//     If this was a probe, returns state to Open without penalty so another probe can run.
//   - If err is backend fault:
//   - If HalfOpen -> transitions to Open, restarts cooldown, increments probe failures.
//   - If Closed -> increments consecutive faults. If consecutive faults >= FaultThreshold
//     -> transitions to Open, records openedAt.
func (b *RemoteL3Breaker) RecordResult(err error, isProbe bool) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()

	state := b.state
	if state == "" {
		state = BreakerClosed
	}

	if errors.Is(err, context.Canceled) {
		if isProbe && state == BreakerHalfOpen {
			b.state = BreakerOpen
		}
		return
	}

	if err == nil {
		switch state {
		case BreakerHalfOpen:
			b.state = BreakerClosed
			b.consecutiveFaults = 0
			b.probeRecoveries++
			b.openedAt = time.Time{}
		case BreakerClosed:
			b.consecutiveFaults = 0
		case BreakerOpen:
			b.state = BreakerClosed
			b.consecutiveFaults = 0
			b.openedAt = time.Time{}
		}
		return
	}

	// Backend fault.
	b.totalFaults++
	now := b.nowLocked()

	if state == BreakerHalfOpen || isProbe {
		b.state = BreakerOpen
		b.openedAt = now
		b.probeFailures++
		b.consecutiveFaults++
		return
	}

	if state == BreakerClosed {
		b.consecutiveFaults++
		if b.consecutiveFaults >= b.faultThresholdLocked() {
			b.state = BreakerOpen
			b.openedAt = now
		}
		return
	}

	b.consecutiveFaults++
}

// State returns the current breaker state.
func (b *RemoteL3Breaker) State() BreakerState {
	if b == nil {
		return BreakerClosed
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.state == "" {
		return BreakerClosed
	}
	return b.state
}

// Stats returns a copy of current breaker counters.
func (b *RemoteL3Breaker) Stats() BreakerStats {
	if b == nil {
		return BreakerStats{
			State:          BreakerClosed,
			FaultThreshold: DefaultBreakerFaultThreshold,
			Cooldown:       DefaultBreakerCooldown,
		}
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	state := b.state
	if state == "" {
		state = BreakerClosed
	}
	return BreakerStats{
		State:             state,
		ConsecutiveFaults: b.consecutiveFaults,
		TotalFaults:       b.totalFaults,
		OpenSkips:         b.openSkips,
		ProbesAttempted:   b.probesAttempted,
		ProbeRecoveries:   b.probeRecoveries,
		ProbeFailures:     b.probeFailures,
		OpenedAt:          b.openedAt,
		FaultThreshold:    b.faultThresholdLocked(),
		Cooldown:          b.cooldownLocked(),
	}
}

// Reset restores the breaker to its initial closed state.
func (b *RemoteL3Breaker) Reset() {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.state = BreakerClosed
	b.consecutiveFaults = 0
	b.totalFaults = 0
	b.openSkips = 0
	b.probesAttempted = 0
	b.probeRecoveries = 0
	b.probeFailures = 0
	b.openedAt = time.Time{}
}

// ConsecutiveFaults returns consecutive backend faults recorded.
func (b *RemoteL3Breaker) ConsecutiveFaults() int {
	if b == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.consecutiveFaults
}

// TotalFaults returns total backend faults recorded over the breaker's lifetime.
func (b *RemoteL3Breaker) TotalFaults() int {
	if b == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.totalFaults
}

// OpenSkips returns the count of calls rejected by an open/half-open breaker.
func (b *RemoteL3Breaker) OpenSkips() int {
	if b == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.openSkips
}

// ProbesAttempted returns the count of half-open probe reads initiated.
func (b *RemoteL3Breaker) ProbesAttempted() int {
	if b == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.probesAttempted
}

// ProbeRecoveries returns the count of probe reads that succeeded and closed the breaker.
func (b *RemoteL3Breaker) ProbeRecoveries() int {
	if b == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.probeRecoveries
}

// ProbeFailures returns the count of probe reads that faulted.
func (b *RemoteL3Breaker) ProbeFailures() int {
	if b == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.probeFailures
}

// OpenedAt returns the timestamp when the breaker transitioned to Open.
func (b *RemoteL3Breaker) OpenedAt() time.Time {
	if b == nil {
		return time.Time{}
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.openedAt
}
