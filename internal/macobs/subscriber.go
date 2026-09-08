package macobs

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"
)

// MemoryPressureLevel represents Darwin kernel memory pressure states.
type MemoryPressureLevel string

const (
	PressureNormal   MemoryPressureLevel = "NORMAL"
	PressureWarn     MemoryPressureLevel = "WARN"
	PressureCritical MemoryPressureLevel = "CRITICAL"
	PressureUnknown  MemoryPressureLevel = "UNKNOWN"

	// Aliases for backward and cross-subsystem compatibility.
	MemoryPressureNormal   = PressureNormal
	MemoryPressureWarn     = PressureWarn
	MemoryPressureCritical = PressureCritical
	MemoryPressureUnknown  = PressureUnknown
)

// Valid returns true if the pressure level is within the known closed set.
func (l MemoryPressureLevel) Valid() bool {
	switch l {
	case PressureNormal, PressureWarn, PressureCritical, PressureUnknown:
		return true
	default:
		return false
	}
}

// Severity returns an integer ordering of memory pressure urgency:
// Normal (0) < Warn (1) < Critical (2), Unknown (-1).
func (l MemoryPressureLevel) Severity() int {
	switch l {
	case PressureNormal:
		return 0
	case PressureWarn:
		return 1
	case PressureCritical:
		return 2
	default:
		return -1
	}
}

// PressureEvent represents an observed or simulated memory pressure notification.
type PressureEvent struct {
	Level     MemoryPressureLevel `json:"level"`
	Timestamp time.Time           `json:"timestamp"`
	RawFlags  uint64              `json:"raw_flags,omitempty"`
	Source    string              `json:"source"`
	Details   string              `json:"details,omitempty"`
}

// JSON returns a JSON string representation of the pressure event.
func (e PressureEvent) JSON() string {
	b, err := json.Marshal(e)
	if err != nil {
		return fmt.Sprintf(`{"level":%q,"error":"marshal failed"}`, e.Level)
	}
	return string(b)
}

// PressureHandler is a callback function invoked when a memory pressure event occurs.
type PressureHandler func(event PressureEvent)

// MemoryPressureSubscriber listens for real-time OS memory pressure changes.
type MemoryPressureSubscriber interface {
	Start(ctx context.Context) error
	Stop() error
	CurrentLevel() MemoryPressureLevel
	Subscribe(handler PressureHandler) func()
	SimulatePressure(level MemoryPressureLevel)
}

// KVCacheEvictor abstracts a KV cache or prefix cache (such as RadixAttention) capable of
// evicting unpinned cache pages while protecting active decoding sessions.
type KVCacheEvictor interface {
	// EvictOldestPages evicts up to targetPages oldest unpinned cache pages.
	// It must preserve pinned pages belonging to active decoding sessions.
	// Returns number of pages evicted and bytes reclaimed.
	EvictOldestPages(targetPages int) (evictedPages int, bytesReclaimed uint64)
	// TotalPages returns total cached pages (pinned + unpinned).
	TotalPages() int
	// ActiveSessions returns the number of active decoding sessions.
	ActiveSessions() int
}

// AdmissionState represents the request queue admission and concurrency throttling state.
type AdmissionState string

const (
	AdmissionNormal    AdmissionState = "NORMAL"
	AdmissionThrottled AdmissionState = "THROTTLED"
	AdmissionPaused    AdmissionState = "PAUSED"
)

// PressureEventSchemaV1 is the canonical schema identifier for memory governor pressure events.
const PressureEventSchemaV1 = "fak.macobs.pressure_event.v1"

// MemoryGovernorEventLog records an action taken by the MemoryGovernor in response to pressure.
type MemoryGovernorEventLog struct {
	Schema         string              `json:"schema"`
	Timestamp      time.Time           `json:"timestamp"`
	PressureLevel  MemoryPressureLevel `json:"pressure_level"`
	AdmissionState AdmissionState      `json:"admission_state"`
	EvictedPages   int                 `json:"evicted_pages"`
	BytesReclaimed uint64              `json:"bytes_reclaimed"`
	ActiveSessions int                 `json:"active_sessions"`
	RemainingPages int                 `json:"remaining_pages"`
	DurationNs     int64               `json:"duration_ns"`
	Message        string              `json:"message"`
}

// GovernorConfig holds tuning parameters for memory pressure reactions.
type GovernorConfig struct {
	// WarnEvictFraction is the fraction of total cached pages to evict upon entering WARN (default: 0.30).
	WarnEvictFraction float64 `json:"warn_evict_fraction"`
	// CriticalEvictFraction is the fraction of total cached pages to evict upon entering CRITICAL (default: 0.75).
	CriticalEvictFraction float64 `json:"critical_evict_fraction"`
	// MinEvictPages is the minimum number of pages to evict if cache is non-empty (default: 1).
	MinEvictPages int `json:"min_evict_pages"`
	// BytesPerPage is the estimated byte size of a KV page (default: 16384 for Apple Silicon).
	BytesPerPage uint64 `json:"bytes_per_page"`
	// LogEmitter optionally receives formatted JSON event logs upon WARN/CRITICAL/NORMAL transitions.
	LogEmitter io.Writer `json:"-"`
}

// DefaultGovernorConfig returns recommended production defaults for Apple Silicon.
func DefaultGovernorConfig() GovernorConfig {
	return GovernorConfig{
		WarnEvictFraction:     0.30,
		CriticalEvictFraction: 0.75,
		MinEvictPages:         1,
		BytesPerPage:          16384,
	}
}

// GovernorStats holds runtime metrics collected by the MemoryGovernor.
type GovernorStats struct {
	CurrentLevel       MemoryPressureLevel `json:"current_level"`
	AdmissionState     AdmissionState      `json:"admission_state"`
	TotalEvictions     uint64              `json:"total_evictions"`
	TotalPagesEvicted  uint64              `json:"total_pages_evicted"`
	TotalBytesFreed    uint64              `json:"total_bytes_freed"`
	LastEvictCount     int                 `json:"last_evict_count"`
	LastBytesFreed     uint64              `json:"last_bytes_freed"`
	LastEvictLatencyNs int64               `json:"last_evict_latency_ns"`
}

// MemoryGovernor bridges a MemoryPressureSubscriber with a KVCacheEvictor to perform
// deterministic proactive page eviction and request admission throttling under OS pressure.
type MemoryGovernor struct {
	mu             sync.RWMutex
	config         GovernorConfig
	subscriber     MemoryPressureSubscriber
	evictor        KVCacheEvictor
	currentLevel   MemoryPressureLevel
	admissionState AdmissionState
	stats          GovernorStats
	lastEventLog   *MemoryGovernorEventLog
	history        []MemoryGovernorEventLog
	unsubscribe    func()
	running        bool
}

// NewMemoryGovernor creates a MemoryGovernor linked to the specified subscriber and cache evictor.
func NewMemoryGovernor(subscriber MemoryPressureSubscriber, evictor KVCacheEvictor, cfg GovernorConfig) *MemoryGovernor {
	if cfg.WarnEvictFraction <= 0 || cfg.WarnEvictFraction > 1.0 {
		cfg.WarnEvictFraction = 0.30
	}
	if cfg.CriticalEvictFraction <= 0 || cfg.CriticalEvictFraction > 1.0 {
		cfg.CriticalEvictFraction = 0.75
	}
	if cfg.MinEvictPages <= 0 {
		cfg.MinEvictPages = 1
	}
	if cfg.BytesPerPage == 0 {
		cfg.BytesPerPage = 16384
	}

	initialLevel := PressureNormal
	if subscriber != nil {
		initialLevel = subscriber.CurrentLevel()
	}

	return &MemoryGovernor{
		config:         cfg,
		subscriber:     subscriber,
		evictor:        evictor,
		currentLevel:   initialLevel,
		admissionState: AdmissionNormal,
		stats: GovernorStats{
			CurrentLevel:   initialLevel,
			AdmissionState: AdmissionNormal,
		},
		history: make([]MemoryGovernorEventLog, 0, 16),
	}
}

// Start begins listening to memory pressure events and initiates proactive throttling.
func (g *MemoryGovernor) Start(ctx context.Context) error {
	g.mu.Lock()
	if g.running {
		g.mu.Unlock()
		return nil
	}
	g.running = true

	if g.subscriber != nil {
		g.unsubscribe = g.subscriber.Subscribe(g.OnPressureEvent)
		g.mu.Unlock()
		return g.subscriber.Start(ctx)
	}
	g.mu.Unlock()
	return nil
}

// Stop unsubscribes from pressure events and terminates governor operations.
func (g *MemoryGovernor) Stop() error {
	g.mu.Lock()
	if !g.running {
		g.mu.Unlock()
		return nil
	}
	g.running = false
	unsub := g.unsubscribe
	g.unsubscribe = nil
	sub := g.subscriber
	g.mu.Unlock()

	if unsub != nil {
		unsub()
	}
	if sub != nil {
		return sub.Stop()
	}
	return nil
}

// OnPressureEvent handles an incoming memory pressure transition from the subscriber.
func (g *MemoryGovernor) OnPressureEvent(evt PressureEvent) {
	start := time.Now()
	g.mu.Lock()
	defer g.mu.Unlock()

	prevLevel := g.currentLevel
	g.currentLevel = evt.Level
	g.stats.CurrentLevel = evt.Level

	var evicted int
	var freedBytes uint64

	switch evt.Level {
	case PressureWarn:
		g.admissionState = AdmissionThrottled
		g.stats.AdmissionState = AdmissionThrottled
		if g.evictor != nil {
			total := g.evictor.TotalPages()
			target := int(float64(total) * g.config.WarnEvictFraction)
			if target < g.config.MinEvictPages && total > 0 {
				target = g.config.MinEvictPages
			}
			if target > 0 {
				evicted, freedBytes = g.evictor.EvictOldestPages(target)
			}
		}

	case PressureCritical:
		g.admissionState = AdmissionPaused
		g.stats.AdmissionState = AdmissionPaused
		if g.evictor != nil {
			total := g.evictor.TotalPages()
			target := int(float64(total) * g.config.CriticalEvictFraction)
			if target < g.config.MinEvictPages && total > 0 {
				target = g.config.MinEvictPages
			}
			if target > 0 {
				evicted, freedBytes = g.evictor.EvictOldestPages(target)
			}
		}

	case PressureNormal:
		g.admissionState = AdmissionNormal
		g.stats.AdmissionState = AdmissionNormal
		// No eviction on recovery to normal; active decoding sessions resume uninhibited.

	default:
		// Unknown level: preserve current state without panic.
		return
	}

	duration := time.Since(start)

	if evicted > 0 {
		g.stats.TotalEvictions++
		g.stats.TotalPagesEvicted += uint64(evicted)
		g.stats.TotalBytesFreed += freedBytes
		g.stats.LastEvictCount = evicted
		g.stats.LastBytesFreed = freedBytes
		g.stats.LastEvictLatencyNs = duration.Nanoseconds()
	}

	// Emit structured JSON event log on WARN, CRITICAL, or state transition from degraded.
	shouldLog := evt.Level == PressureWarn || evt.Level == PressureCritical || (prevLevel != PressureNormal && evt.Level == PressureNormal)
	if shouldLog {
		remaining := 0
		active := 0
		if g.evictor != nil {
			remaining = g.evictor.TotalPages()
			active = g.evictor.ActiveSessions()
		}

		var msg string
		switch evt.Level {
		case PressureWarn:
			msg = fmt.Sprintf("macOS memory pressure WARN: evicted %d oldest KV cache pages (%d bytes reclaimed), admission throttled", evicted, freedBytes)
		case PressureCritical:
			msg = fmt.Sprintf("macOS memory pressure CRITICAL: evicted %d KV cache pages (%d bytes reclaimed), request queue paused", evicted, freedBytes)
		case PressureNormal:
			msg = "macOS memory pressure recovered to NORMAL: resumed request admission and restored normal operation"
		}

		logEntry := MemoryGovernorEventLog{
			Schema:         PressureEventSchemaV1,
			Timestamp:      evt.Timestamp,
			PressureLevel:  evt.Level,
			AdmissionState: g.admissionState,
			EvictedPages:   evicted,
			BytesReclaimed: freedBytes,
			ActiveSessions: active,
			RemainingPages: remaining,
			DurationNs:     duration.Nanoseconds(),
			Message:        msg,
		}

		g.lastEventLog = &logEntry
		g.history = append(g.history, logEntry)

		if g.config.LogEmitter != nil {
			if b, err := json.Marshal(logEntry); err == nil {
				_, _ = g.config.LogEmitter.Write(append(b, '\n'))
			}
		}
	}
}

// CurrentLevel returns the governor's observed memory pressure level.
func (g *MemoryGovernor) CurrentLevel() MemoryPressureLevel {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.currentLevel
}

// AdmissionState returns the current admission throttle or pause state.
func (g *MemoryGovernor) AdmissionState() AdmissionState {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.admissionState
}

// Stats returns a point-in-time copy of memory governor metrics.
func (g *MemoryGovernor) Stats() GovernorStats {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.stats
}

// LastEventLog returns the most recent structured event log, or nil if none.
func (g *MemoryGovernor) LastEventLog() *MemoryGovernorEventLog {
	g.mu.RLock()
	defer g.mu.RUnlock()
	if g.lastEventLog == nil {
		return nil
	}
	cp := *g.lastEventLog
	return &cp
}

// History returns an immutable copy of all recorded memory pressure event logs.
func (g *MemoryGovernor) History() []MemoryGovernorEventLog {
	g.mu.RLock()
	defer g.mu.RUnlock()
	out := make([]MemoryGovernorEventLog, len(g.history))
	copy(out, g.history)
	return out
}

// SimulatedMemoryPressureSubscriber provides a deterministic, thread-safe memory pressure subscriber
// suitable for cross-platform simulation and unit testing.
type SimulatedMemoryPressureSubscriber struct {
	mu           sync.RWMutex
	currentLevel MemoryPressureLevel
	handlers     map[uint64]PressureHandler
	nextID       uint64
	running      bool
}

// NewSimulatedSubscriber creates a new SimulatedMemoryPressureSubscriber.
func NewSimulatedSubscriber() *SimulatedMemoryPressureSubscriber {
	return &SimulatedMemoryPressureSubscriber{
		currentLevel: PressureNormal,
		handlers:     make(map[uint64]PressureHandler),
	}
}

// Start marks the simulated subscriber as active.
func (s *SimulatedMemoryPressureSubscriber) Start(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.running = true
	return nil
}

// Stop terminates the simulated subscriber.
func (s *SimulatedMemoryPressureSubscriber) Stop() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.running = false
	return nil
}

// CurrentLevel returns the currently simulated memory pressure level.
func (s *SimulatedMemoryPressureSubscriber) CurrentLevel() MemoryPressureLevel {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.currentLevel
}

// Subscribe registers a PressureHandler and returns an unsubscribe function.
func (s *SimulatedMemoryPressureSubscriber) Subscribe(handler PressureHandler) func() {
	s.mu.Lock()
	id := s.nextID
	s.nextID++
	s.handlers[id] = handler
	s.mu.Unlock()

	return func() {
		s.mu.Lock()
		delete(s.handlers, id)
		s.mu.Unlock()
	}
}

// SimulatePressure dispatches a simulated memory pressure transition to all subscribers.
func (s *SimulatedMemoryPressureSubscriber) SimulatePressure(level MemoryPressureLevel) {
	s.mu.Lock()
	s.currentLevel = level
	handlers := make([]PressureHandler, 0, len(s.handlers))
	for _, h := range s.handlers {
		handlers = append(handlers, h)
	}
	s.mu.Unlock()

	evt := PressureEvent{
		Level:     level,
		Timestamp: time.Now().UTC(),
		Source:    "simulated",
		Details:   fmt.Sprintf("simulated memory pressure transition to %s", level),
	}

	for _, h := range handlers {
		h(evt)
	}
}

// newPlatformSubscriber is overridden by pressure_darwin.go on Darwin platforms.
var newPlatformSubscriber = func() MemoryPressureSubscriber {
	return NewSimulatedSubscriber()
}

// NewPlatformMemoryPressureSubscriber returns the host-native subscriber (Darwin GCD on macOS, simulated elsewhere).
func NewPlatformMemoryPressureSubscriber() MemoryPressureSubscriber {
	return newPlatformSubscriber()
}
