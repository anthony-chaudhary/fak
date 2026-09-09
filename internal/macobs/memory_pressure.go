package macobs

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// StatusCapacityBackoff is the HTTP status code returned to OpenCode under critical memory pressure.
	StatusCapacityBackoff = http.StatusTooManyRequests // 429

	// CodeCapacityBackoff is the structured error code identifying memory governor backpressure.
	CodeCapacityBackoff = "CAPACITY_BACKOFF"

	// ReasonCapacityBackoff identifies batch rejection due to critical memory pressure.
	ReasonCapacityBackoff = "CAPACITY_BACKOFF"

	// UnifiedMemoryGovernorSchemaV1 is the canonical schema identifier for unified memory governor logs.
	UnifiedMemoryGovernorSchemaV1 = "fak.macobs.unified_memory_governor.v1"
)

// BackpressureResponse models the structured 429 response emitted to OpenCode when admissions are paused.
type BackpressureResponse struct {
	StatusCode     int                 `json:"status_code"`
	ErrorCode      string              `json:"error_code"`
	Message        string              `json:"message"`
	PressureLevel  MemoryPressureLevel `json:"pressure_level"`
	RetryAfterSec  int                 `json:"retry_after_sec"`
	ReclaimedBytes uint64              `json:"reclaimed_bytes"`
	MTPDepth       int                 `json:"mtp_depth"`
	ActiveSessions int                 `json:"active_sessions"`
	Timestamp      time.Time           `json:"timestamp"`
}

// Error implements the error interface for BackpressureResponse.
func (b *BackpressureResponse) Error() string {
	return fmt.Sprintf("HTTP %d (%s): %s (retry after %ds)", b.StatusCode, b.ErrorCode, b.Message, b.RetryAfterSec)
}

// OpenCodeBatchRequest represents an incoming batch or token generation request from OpenCode.
type OpenCodeBatchRequest struct {
	SessionID   string `json:"session_id"`
	Tokens      int    `json:"tokens"`
	Priority    int    `json:"priority"`
	IsHeartbeat bool   `json:"is_heartbeat"`
}

// OpenCodeBatchAdmission represents the admission decision for an incoming OpenCode batch.
type OpenCodeBatchAdmission struct {
	Admitted     bool                  `json:"admitted"`
	StatusCode   int                   `json:"status_code"`
	Reason       string                `json:"reason"`
	Backpressure *BackpressureResponse `json:"backpressure,omitempty"`
}

// OpenCodePageManager abstracts dynamic unpinned page reclamation and cold prefix disk page-out.
type OpenCodePageManager interface {
	// ReclaimUnpinnedPages executes dynamic in-memory unpinned block reclamation, returning reclaimed bytes.
	ReclaimUnpinnedPages(targetBytes uint64) (reclaimedBytes uint64, err error)

	// PageOutColdPrefixes serializes and offloads cold prefix blocks to disk storage,
	// protecting active session hot context. Returns paged-out block count and reclaimed bytes.
	PageOutColdPrefixes(maxBlocks int) (pagedBlocks int, reclaimedBytes uint64, err error)

	// TotalPages returns total managed memory pages.
	TotalPages() int

	// ActiveSessions returns the count of currently active decoding sessions.
	ActiveSessions() int
}

// MTPDepthGovernor controls speculative Multi-Token Prediction (MTP) candidate depth.
type MTPDepthGovernor interface {
	// CurrentDepth returns the currently permitted speculative candidate depth.
	CurrentDepth() int

	// SetDepth adjusts the speculative candidate depth within configured bounds.
	SetDepth(depth int)

	// NominalDepth returns the baseline unconstrained candidate depth configured for normal memory conditions.
	NominalDepth() int

	// ThrottledDepth returns the constrained candidate depth enforced during memory pressure warnings to conserve DRAM.
	ThrottledDepth() int

	// CriticalDepth returns the minimal candidate depth permitted under critical memory pressure to prevent allocation spikes.
	CriticalDepth() int
}

// DefaultMTPDepthGovernor implements MTPDepthGovernor with thread-safe clamping.
type DefaultMTPDepthGovernor struct {
	mu        sync.RWMutex
	current   int
	nominal   int
	throttled int
	critical  int
}

// NewDefaultMTPDepthGovernor creates a new DefaultMTPDepthGovernor.
func NewDefaultMTPDepthGovernor(nominal, throttled, critical int) *DefaultMTPDepthGovernor {
	if nominal < 0 {
		nominal = 3
	}
	if throttled < 0 {
		throttled = 1
	}
	if critical < 0 {
		critical = 0
	}
	return &DefaultMTPDepthGovernor{
		current:   nominal,
		nominal:   nominal,
		throttled: throttled,
		critical:  critical,
	}
}

// CurrentDepth returns the currently permitted candidate depth.
func (m *DefaultMTPDepthGovernor) CurrentDepth() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.current
}

// SetDepth clamps and updates the candidate depth.
func (m *DefaultMTPDepthGovernor) SetDepth(depth int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if depth < 0 {
		depth = 0
	}
	if depth > m.nominal {
		depth = m.nominal
	}
	m.current = depth
}

// NominalDepth returns the baseline unconstrained candidate depth configured for normal memory conditions.
func (m *DefaultMTPDepthGovernor) NominalDepth() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.nominal
}

// ThrottledDepth returns the constrained candidate depth enforced during memory pressure warnings to conserve DRAM.
func (m *DefaultMTPDepthGovernor) ThrottledDepth() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.throttled
}

// CriticalDepth returns the minimal candidate depth permitted under critical memory pressure to prevent allocation spikes.
func (m *DefaultMTPDepthGovernor) CriticalDepth() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.critical
}

// UnifiedMemoryGovernorConfig configures tuning parameters for Apple Silicon memory governor reactions.
type UnifiedMemoryGovernorConfig struct {
	NominalMTPDepth      int       `json:"nominal_mtp_depth"`
	ThrottledMTPDepth    int       `json:"throttled_mtp_depth"`
	CriticalMTPDepth     int       `json:"critical_mtp_depth"`
	WarnReclaimBytes     uint64    `json:"warn_reclaim_bytes"`
	WarnPrefixPageOutMax int       `json:"warn_prefix_page_out_max"`
	CritReclaimBytes     uint64    `json:"crit_reclaim_bytes"`
	CritPrefixPageOutMax int       `json:"crit_prefix_page_out_max"`
	BackpressureRetrySec int       `json:"backpressure_retry_sec"`
	LogEmitter           io.Writer `json:"-"`
}

// DefaultUnifiedMemoryGovernorConfig returns recommended production defaults for Apple Silicon.
func DefaultUnifiedMemoryGovernorConfig() UnifiedMemoryGovernorConfig {
	return UnifiedMemoryGovernorConfig{
		NominalMTPDepth:      3,
		ThrottledMTPDepth:    1,
		CriticalMTPDepth:     0,
		WarnReclaimBytes:     512 * 1024 * 1024,      // 512 MB
		WarnPrefixPageOutMax: 32,                     // 32 prefix blocks
		CritReclaimBytes:     2 * 1024 * 1024 * 1024, // 2 GB
		CritPrefixPageOutMax: 128,                    // 128 prefix blocks
		BackpressureRetrySec: 2,                      // 2 second retry-after suggestion
	}
}

// UnifiedGovernorTelemetry models the runtime status and metrics of the governor.
type UnifiedGovernorTelemetry struct {
	PressureLevel           MemoryPressureLevel `json:"pressure_level"`
	AdmissionState          AdmissionState      `json:"admission_state"`
	MTPDepth                int                 `json:"mtp_depth"`
	TotalReclaimedBytes     uint64              `json:"total_reclaimed_bytes"`
	TotalPrefixesPagedOut   int                 `json:"total_prefixes_paged_out"`
	TotalBackpressure429s   uint64              `json:"total_backpressure_429s"`
	ActiveSessions          int                 `json:"active_sessions"`
	LastEventTimestamp      time.Time           `json:"last_event_timestamp"`
	ZeroSwapBoundGuaranteed bool                `json:"zero_swap_bound_guaranteed"`
}

// UnifiedMemoryGovernorEventLog records an action taken by the governor under memory pressure.
type UnifiedMemoryGovernorEventLog struct {
	Schema         string              `json:"schema"`
	Timestamp      time.Time           `json:"timestamp"`
	PressureLevel  MemoryPressureLevel `json:"pressure_level"`
	AdmissionState AdmissionState      `json:"admission_state"`
	MTPDepth       int                 `json:"mtp_depth"`
	ReclaimedBytes uint64              `json:"reclaimed_bytes"`
	PagedOutBlocks int                 `json:"paged_out_blocks"`
	DurationNs     int64               `json:"duration_ns"`
	Message        string              `json:"message"`
}

// UnifiedMemoryPressureGovernor enforces real-time Apple Silicon unified memory pressure governance
// for OpenCode local serving, protecting against macOS Jetsam SIGKILL termination.
type UnifiedMemoryPressureGovernor struct {
	mu                    sync.RWMutex
	config                UnifiedMemoryGovernorConfig
	subscriber            MemoryPressureSubscriber
	pageManager           OpenCodePageManager
	mtpGovernor           MTPDepthGovernor
	currentLevel          MemoryPressureLevel
	admissionState        AdmissionState
	totalReclaimedBytes   uint64
	totalPrefixesPagedOut int
	backpressure429Count  uint64
	lastEventTimestamp    time.Time
	unsubscribe           func()
	running               bool
	history               []UnifiedMemoryGovernorEventLog
}

// OpenCodeServingGovernor is an alias for UnifiedMemoryPressureGovernor.
type OpenCodeServingGovernor = UnifiedMemoryPressureGovernor

// NewUnifiedMemoryPressureGovernor creates a new UnifiedMemoryPressureGovernor.
func NewUnifiedMemoryPressureGovernor(
	sub MemoryPressureSubscriber,
	pm OpenCodePageManager,
	mtpGov MTPDepthGovernor,
	cfg UnifiedMemoryGovernorConfig,
) *UnifiedMemoryPressureGovernor {
	if sub == nil {
		sub = NewPlatformMemoryPressureSubscriber()
	}
	if mtpGov == nil {
		mtpGov = NewDefaultMTPDepthGovernor(cfg.NominalMTPDepth, cfg.ThrottledMTPDepth, cfg.CriticalMTPDepth)
	}

	initialLevel := PressureNormal
	if sub != nil {
		initialLevel = sub.CurrentLevel()
	}

	return &UnifiedMemoryPressureGovernor{
		config:             cfg,
		subscriber:         sub,
		pageManager:        pm,
		mtpGovernor:        mtpGov,
		currentLevel:       initialLevel,
		admissionState:     AdmissionNormal,
		lastEventTimestamp: time.Now().UTC(),
		history:            make([]UnifiedMemoryGovernorEventLog, 0, 16),
	}
}

// Start registers the pressure subscriber callback and begins listening for OS memory pressure notifications.
func (g *UnifiedMemoryPressureGovernor) Start(ctx context.Context) error {
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

// Stop unregisters subscriber callbacks and shuts down the governor.
func (g *UnifiedMemoryPressureGovernor) Stop() error {
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

// CurrentLevel returns the governor's observed memory pressure level.
func (g *UnifiedMemoryPressureGovernor) CurrentLevel() MemoryPressureLevel {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.currentLevel
}

// AdmissionState returns the current batch admission state (NORMAL, THROTTLED, PAUSED).
func (g *UnifiedMemoryPressureGovernor) AdmissionState() AdmissionState {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.admissionState
}

// EvaluateBatchAdmission evaluates an incoming batch or token generation request from OpenCode.
// Under CRITICAL pressure, it gracefully pauses admissions and returns structured 429 CAPACITY_BACKOFF.
func (g *UnifiedMemoryPressureGovernor) EvaluateBatchAdmission(req OpenCodeBatchRequest) OpenCodeBatchAdmission {
	g.mu.RLock()
	defer g.mu.RUnlock()

	if g.admissionState == AdmissionPaused || g.currentLevel == PressureCritical {
		atomic.AddUint64(&g.backpressure429Count, 1)
		active := 0
		if g.pageManager != nil {
			active = g.pageManager.ActiveSessions()
		}
		mtpDepth := 0
		if g.mtpGovernor != nil {
			mtpDepth = g.mtpGovernor.CurrentDepth()
		}

		bp := &BackpressureResponse{
			StatusCode:     StatusCapacityBackoff,
			ErrorCode:      CodeCapacityBackoff,
			Message:        "macOS kernel memory pressure critical (DISPATCH_MEMORYPRESSURE_CRITICAL); batch admissions paused to prevent Jetsam SIGKILL",
			PressureLevel:  g.currentLevel,
			RetryAfterSec:  g.config.BackpressureRetrySec,
			ReclaimedBytes: g.totalReclaimedBytes,
			MTPDepth:       mtpDepth,
			ActiveSessions: active,
			Timestamp:      time.Now().UTC(),
		}

		return OpenCodeBatchAdmission{
			Admitted:     false,
			StatusCode:   StatusCapacityBackoff,
			Reason:       ReasonCapacityBackoff,
			Backpressure: bp,
		}
	}

	if g.admissionState == AdmissionThrottled || g.currentLevel == PressureWarn {
		return OpenCodeBatchAdmission{
			Admitted:   true,
			StatusCode: http.StatusOK,
			Reason:     "ADMITTED_THROTTLED_MTP",
		}
	}

	return OpenCodeBatchAdmission{
		Admitted:   true,
		StatusCode: http.StatusOK,
		Reason:     "ADMITTED_NORMAL",
	}
}

// OnPressureEvent handles real-time Darwin memory pressure transitions from the subscriber.
func (g *UnifiedMemoryPressureGovernor) OnPressureEvent(evt PressureEvent) {
	t0 := time.Now()
	g.mu.Lock()
	defer g.mu.Unlock()

	g.currentLevel = evt.Level
	g.lastEventTimestamp = evt.Timestamp

	var recBytes uint64
	var pagedCount int

	switch evt.Level {
	case PressureWarn:
		g.admissionState = AdmissionThrottled
		if g.mtpGovernor != nil {
			g.mtpGovernor.SetDepth(g.config.ThrottledMTPDepth)
		}
		if g.pageManager != nil {
			if r, err := g.pageManager.ReclaimUnpinnedPages(g.config.WarnReclaimBytes); err == nil {
				recBytes = r
				g.totalReclaimedBytes += r
			}
			if p, _, err := g.pageManager.PageOutColdPrefixes(g.config.WarnPrefixPageOutMax); err == nil {
				pagedCount = p
				g.totalPrefixesPagedOut += p
			}
		}

	case PressureCritical:
		g.admissionState = AdmissionPaused
		if g.mtpGovernor != nil {
			g.mtpGovernor.SetDepth(g.config.CriticalMTPDepth)
		}
		if g.pageManager != nil {
			if r, err := g.pageManager.ReclaimUnpinnedPages(g.config.CritReclaimBytes); err == nil {
				recBytes = r
				g.totalReclaimedBytes += r
			}
			if p, _, err := g.pageManager.PageOutColdPrefixes(g.config.CritPrefixPageOutMax); err == nil {
				pagedCount = p
				g.totalPrefixesPagedOut += p
			}
		}

	case PressureNormal:
		g.admissionState = AdmissionNormal
		if g.mtpGovernor != nil {
			g.mtpGovernor.SetDepth(g.config.NominalMTPDepth)
		}

	default:
		return
	}

	dur := time.Since(t0)
	depth := 0
	if g.mtpGovernor != nil {
		depth = g.mtpGovernor.CurrentDepth()
	}

	var note string
	switch evt.Level {
	case PressureWarn:
		note = fmt.Sprintf("macOS memory pressure WARN: reclaimed %d bytes unpinned pages, paged out %d cold prefix blocks, throttled MTP depth to %d", recBytes, pagedCount, depth)
	case PressureCritical:
		note = fmt.Sprintf("macOS memory pressure CRITICAL: reclaimed %d bytes unpinned pages, paged out %d prefix blocks, paused batch admissions (429 CAPACITY_BACKOFF)", recBytes, pagedCount)
	case PressureNormal:
		note = fmt.Sprintf("macOS memory pressure recovered to NORMAL: unpaused batch admissions, restored nominal MTP depth to %d", depth)
	}

	entry := UnifiedMemoryGovernorEventLog{
		Schema:         UnifiedMemoryGovernorSchemaV1,
		Timestamp:      evt.Timestamp,
		PressureLevel:  evt.Level,
		AdmissionState: g.admissionState,
		MTPDepth:       depth,
		ReclaimedBytes: recBytes,
		PagedOutBlocks: pagedCount,
		DurationNs:     dur.Nanoseconds(),
		Message:        note,
	}

	g.history = append(g.history, entry)

	if g.config.LogEmitter != nil {
		if payload, err := json.Marshal(entry); err == nil {
			_, _ = g.config.LogEmitter.Write(append(payload, '\n'))
		}
	}
}

// Telemetry snapshots the real-time status and cumulative metrics of the governor.
func (g *UnifiedMemoryPressureGovernor) Telemetry() UnifiedGovernorTelemetry {
	g.mu.RLock()
	defer g.mu.RUnlock()

	active := 0
	if g.pageManager != nil {
		active = g.pageManager.ActiveSessions()
	}
	mtpDepth := 0
	if g.mtpGovernor != nil {
		mtpDepth = g.mtpGovernor.CurrentDepth()
	}

	zeroSwap := g.currentLevel != PressureCritical

	return UnifiedGovernorTelemetry{
		PressureLevel:           g.currentLevel,
		AdmissionState:          g.admissionState,
		MTPDepth:                mtpDepth,
		TotalReclaimedBytes:     g.totalReclaimedBytes,
		TotalPrefixesPagedOut:   g.totalPrefixesPagedOut,
		TotalBackpressure429s:   atomic.LoadUint64(&g.backpressure429Count),
		ActiveSessions:          active,
		LastEventTimestamp:      g.lastEventTimestamp,
		ZeroSwapBoundGuaranteed: zeroSwap,
	}
}
