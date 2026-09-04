package gateway

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

// IdlenessChecker reports whether the server or subsystem is idle and the count of active in-flight requests.
type IdlenessChecker interface {
	IsIdle() bool
	ActiveRequests() int
}

// ActiveRequestTracker provides thread-safe tracking of in-flight active requests.
type ActiveRequestTracker struct {
	mu     sync.RWMutex
	active int
}

// NewActiveRequestTracker creates an initialized request tracker.
func NewActiveRequestTracker() *ActiveRequestTracker {
	return &ActiveRequestTracker{}
}

// Incr increments active requests count.
func (t *ActiveRequestTracker) Incr() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.active++
	return t.active
}

// Decr decrements active requests count.
func (t *ActiveRequestTracker) Decr() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.active > 0 {
		t.active--
	}
	return t.active
}

// Track marks a request active and returns a done function to be deferred.
func (t *ActiveRequestTracker) Track() func() {
	t.Incr()
	var once sync.Once
	return func() {
		once.Do(func() {
			t.Decr()
		})
	}
}

// ActiveRequests returns the number of requests currently in-flight.
func (t *ActiveRequestTracker) ActiveRequests() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.active
}

// IsIdle returns true when there are zero active requests.
func (t *ActiveRequestTracker) IsIdle() bool {
	return t.ActiveRequests() == 0
}

// IdleCondition is a function predicate returning true when idle.
type IdleCondition func() bool

// IdleConditionChecker adapts an IdleCondition function to an IdlenessChecker.
type IdleConditionChecker struct {
	fn IdleCondition
}

// NewIdleConditionChecker constructs an IdlenessChecker from an IdleCondition function.
func NewIdleConditionChecker(fn IdleCondition) *IdleConditionChecker {
	return &IdleConditionChecker{fn: fn}
}

// IsIdle checks whether the condition reports idle.
func (c *IdleConditionChecker) IsIdle() bool {
	if c.fn == nil {
		return true
	}
	return c.fn()
}

// ActiveRequests returns 0 if idle, or 1 if busy.
func (c *IdleConditionChecker) ActiveRequests() int {
	if c.IsIdle() {
		return 0
	}
	return 1
}

// KVEvictor executes the physical eviction or pruning on the KV tier.
type KVEvictor interface {
	EvictKV(ctx context.Context) (freedBlocks int, err error)
}

// KVEvictorFunc adapts a bare function to the KVEvictor interface.
type KVEvictorFunc func(ctx context.Context) (int, error)

// EvictKV calls the wrapped function.
func (f KVEvictorFunc) EvictKV(ctx context.Context) (int, error) {
	return f(ctx)
}

// EvictionOutcome categorizes the result of an eviction attempt.
type EvictionOutcome string

const (
	EvictionOutcomeDrained EvictionOutcome = "drained"
	EvictionOutcomeBlocked EvictionOutcome = "blocked_busy"
	EvictionOutcomeSkipped EvictionOutcome = "skipped"
	EvictionOutcomeFailed  EvictionOutcome = "failed"
)

// EvictionResult records the outcome and telemetry of an eviction attempt.
type EvictionResult struct {
	Outcome        EvictionOutcome `json:"outcome"`
	FreedBlocks    int             `json:"freed_blocks"`
	ActiveRequests int             `json:"active_requests"`
	Reason         string          `json:"reason,omitempty"`
	Duration       time.Duration   `json:"duration"`
	Timestamp      time.Time       `json:"timestamp"`
	Err            error           `json:"-"`
}

var (
	// ErrEvictionBlockedBusy is returned when an eviction attempt is blocked because the server has active requests.
	ErrEvictionBlockedBusy = errors.New("gateway: kv tier eviction blocked: server is not idle (active requests > 0)")
	// ErrEvictorStopped is returned when an eviction waiting on idle is aborted because the evictor was stopped.
	ErrEvictorStopped = errors.New("gateway: idle-gated evictor stopped")
)

// IdleGatedEvictorConfig configures the eviction policies and timing.
type IdleGatedEvictorConfig struct {
	CheckInterval    time.Duration
	IdleHoldoff      time.Duration
	MaxActiveAllowed int
}

// DefaultIdleGatedEvictorConfig returns sane defaults for the evictor.
func DefaultIdleGatedEvictorConfig() IdleGatedEvictorConfig {
	return IdleGatedEvictorConfig{
		CheckInterval:    50 * time.Millisecond,
		IdleHoldoff:      0,
		MaxActiveAllowed: 0,
	}
}

// EvictorOption allows fine-tuning IdleGatedEvictor settings.
type EvictorOption func(*IdleGatedEvictorConfig)

// WithCheckInterval configures the periodic background tick interval.
func WithCheckInterval(d time.Duration) EvictorOption {
	return func(cfg *IdleGatedEvictorConfig) {
		cfg.CheckInterval = d
	}
}

// WithIdleHoldoff configures minimum quiet duration before eviction triggers.
func WithIdleHoldoff(d time.Duration) EvictorOption {
	return func(cfg *IdleGatedEvictorConfig) {
		cfg.IdleHoldoff = d
	}
}

// WithMaxActiveAllowed configures the threshold for considering the server idle.
func WithMaxActiveAllowed(n int) EvictorOption {
	return func(cfg *IdleGatedEvictorConfig) {
		cfg.MaxActiveAllowed = n
	}
}

// EvictorStats tracks cumulative eviction behavior.
type EvictorStats struct {
	TotalAttempts    int64     `json:"total_attempts"`
	BlockedAttempts  int64     `json:"blocked_attempts"`
	SuccessfulDrains int64     `json:"successful_drains"`
	TotalFreedBlocks int64     `json:"total_freed_blocks"`
	FailedAttempts   int64     `json:"failed_attempts"`
	LastEvictionTime time.Time `json:"last_eviction_time,omitempty"`
	LastBlockedTime  time.Time `json:"last_blocked_time,omitempty"`
}

// IdleGatedEvictor checks if server idleness condition is met before executing background KV tier eviction or pruning,
// preventing mid-turn race conditions.
type IdleGatedEvictor struct {
	mu        sync.Mutex
	checker   IdlenessChecker
	evictor   KVEvictor
	cfg       IdleGatedEvictorConfig
	stopCh    chan struct{}
	triggerCh chan struct{}
	running   bool
	evictMu   sync.Mutex
	statsMu   sync.RWMutex
	stats     EvictorStats
}

// NewIdleGatedEvictor constructs a new idle-gated evictor.
func NewIdleGatedEvictor(checker IdlenessChecker, evictor KVEvictor, opts ...EvictorOption) *IdleGatedEvictor {
	cfg := DefaultIdleGatedEvictorConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	return &IdleGatedEvictor{
		checker:   checker,
		evictor:   evictor,
		cfg:       cfg,
		triggerCh: make(chan struct{}, 1),
	}
}

// TryEvict attempts an eviction immediately if the server is idle.
// If active requests > cfg.MaxActiveAllowed or !checker.IsIdle(), it refuses and returns ErrEvictionBlockedBusy.
func (e *IdleGatedEvictor) TryEvict(ctx context.Context) (EvictionResult, error) {
	start := time.Now()
	e.statsMu.Lock()
	e.stats.TotalAttempts++
	e.statsMu.Unlock()

	active := 0
	if e.checker != nil {
		active = e.checker.ActiveRequests()
		if active > e.cfg.MaxActiveAllowed || !e.checker.IsIdle() {
			e.statsMu.Lock()
			e.stats.BlockedAttempts++
			e.stats.LastBlockedTime = time.Now()
			e.statsMu.Unlock()

			return EvictionResult{
				Outcome:        EvictionOutcomeBlocked,
				FreedBlocks:    0,
				ActiveRequests: active,
				Reason:         "server busy: active requests > 0",
				Duration:       time.Since(start),
				Timestamp:      time.Now(),
				Err:            ErrEvictionBlockedBusy,
			}, ErrEvictionBlockedBusy
		}
	}

	// Idleness condition is met; acquire eviction lock to execute
	e.evictMu.Lock()
	defer e.evictMu.Unlock()

	// Re-check idleness after acquiring evictMu in case a request arrived in the window
	if e.checker != nil {
		active = e.checker.ActiveRequests()
		if active > e.cfg.MaxActiveAllowed || !e.checker.IsIdle() {
			e.statsMu.Lock()
			e.stats.BlockedAttempts++
			e.stats.LastBlockedTime = time.Now()
			e.statsMu.Unlock()

			return EvictionResult{
				Outcome:        EvictionOutcomeBlocked,
				FreedBlocks:    0,
				ActiveRequests: active,
				Reason:         "server busy: request arrived before eviction lock",
				Duration:       time.Since(start),
				Timestamp:      time.Now(),
				Err:            ErrEvictionBlockedBusy,
			}, ErrEvictionBlockedBusy
		}
	}

	if e.evictor == nil {
		return EvictionResult{
			Outcome:        EvictionOutcomeSkipped,
			FreedBlocks:    0,
			ActiveRequests: active,
			Reason:         "no kv evictor configured",
			Duration:       time.Since(start),
			Timestamp:      time.Now(),
		}, nil
	}

	freed, err := e.evictor.EvictKV(ctx)
	elapsed := time.Since(start)
	if err != nil {
		e.statsMu.Lock()
		e.stats.FailedAttempts++
		e.statsMu.Unlock()

		return EvictionResult{
			Outcome:        EvictionOutcomeFailed,
			FreedBlocks:    freed,
			ActiveRequests: 0,
			Reason:         err.Error(),
			Duration:       elapsed,
			Timestamp:      time.Now(),
			Err:            err,
		}, err
	}

	e.statsMu.Lock()
	e.stats.SuccessfulDrains++
	e.stats.TotalFreedBlocks += int64(freed)
	e.stats.LastEvictionTime = time.Now()
	e.statsMu.Unlock()

	return EvictionResult{
		Outcome:        EvictionOutcomeDrained,
		FreedBlocks:    freed,
		ActiveRequests: 0,
		Duration:       elapsed,
		Timestamp:      time.Now(),
	}, nil
}

// DrainWhenIdle blocks until the server is idle, then executes eviction cleanly.
func (e *IdleGatedEvictor) DrainWhenIdle(ctx context.Context, timeout time.Duration) (EvictionResult, error) {
	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(5 * time.Millisecond)
	defer ticker.Stop()

	for {
		res, err := e.TryEvict(ctx)
		if err == nil {
			return res, nil
		}
		if !errors.Is(err, ErrEvictionBlockedBusy) {
			return res, err
		}

		select {
		case <-ctx.Done():
			active := 0
			if e.checker != nil {
				active = e.checker.ActiveRequests()
			}
			return EvictionResult{
				Outcome:        EvictionOutcomeBlocked,
				ActiveRequests: active,
				Reason:         "context cancelled while waiting for idle",
				Timestamp:      time.Now(),
				Err:            ctx.Err(),
			}, ctx.Err()
		case <-ticker.C:
			if time.Now().After(deadline) {
				active := 0
				if e.checker != nil {
					active = e.checker.ActiveRequests()
				}
				return EvictionResult{
					Outcome:        EvictionOutcomeBlocked,
					ActiveRequests: active,
					Reason:         "timeout exceeded while waiting for idle",
					Timestamp:      time.Now(),
					Err:            ErrEvictionBlockedBusy,
				}, ErrEvictionBlockedBusy
			}
		}
	}
}

// WaitAndEvict is an alias for DrainWhenIdle.
func (e *IdleGatedEvictor) WaitAndEvict(ctx context.Context, timeout time.Duration) (EvictionResult, error) {
	return e.DrainWhenIdle(ctx, timeout)
}

// Start launches a background goroutine that periodically triggers eviction when idle.
func (e *IdleGatedEvictor) Start(ctx context.Context) {
	e.mu.Lock()
	if e.running {
		e.mu.Unlock()
		return
	}
	e.running = true
	e.stopCh = make(chan struct{})
	e.mu.Unlock()

	go e.runLoop(ctx)
}

// Stop terminates the background eviction goroutine.
func (e *IdleGatedEvictor) Stop() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.running {
		return
	}
	e.running = false
	close(e.stopCh)
}

// IsRunning reports whether the background worker is currently active.
func (e *IdleGatedEvictor) IsRunning() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.running
}

// Trigger signals an immediate background eviction attempt.
func (e *IdleGatedEvictor) Trigger() {
	select {
	case e.triggerCh <- struct{}{}:
	default:
	}
}

func (e *IdleGatedEvictor) runLoop(ctx context.Context) {
	interval := e.cfg.CheckInterval
	if interval <= 0 {
		interval = 50 * time.Millisecond
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-e.stopCh:
			return
		case <-ticker.C:
			_, _ = e.TryEvict(ctx)
		case <-e.triggerCh:
			_, _ = e.TryEvict(ctx)
		}
	}
}

// Stats returns a snapshot of cumulative eviction stats.
func (e *IdleGatedEvictor) Stats() EvictorStats {
	e.statsMu.RLock()
	defer e.statsMu.RUnlock()
	return e.stats
}

// ResetStats zeroes out all eviction metrics counters.
func (e *IdleGatedEvictor) ResetStats() {
	e.statsMu.Lock()
	defer e.statsMu.Unlock()
	e.stats = EvictorStats{}
}

// SetIdleGatedEvictor installs the idle-gated evictor for the gateway server.
func (s *Server) SetIdleGatedEvictor(e *IdleGatedEvictor) {
	if s == nil {
		return
	}
	s.kvEvictorMu.Lock()
	s.kvEvictor = e
	s.kvEvictorMu.Unlock()
}

// IdleGatedEvictor returns the installed idle-gated evictor, or nil if none is set.
func (s *Server) IdleGatedEvictor() *IdleGatedEvictor {
	if s == nil {
		return nil
	}
	s.kvEvictorMu.RLock()
	defer s.kvEvictorMu.RUnlock()
	return s.kvEvictor
}

// ActiveRequests reports the count of in-flight HTTP requests.
func (s *Server) ActiveRequests() int {
	if s == nil || s.metrics == nil {
		return 0
	}
	return int(atomic.LoadInt64(&s.metrics.inflight))
}

// IsIdle reports whether the server has zero active requests in flight.
func (s *Server) IsIdle() bool {
	return s.ActiveRequests() == 0
}
