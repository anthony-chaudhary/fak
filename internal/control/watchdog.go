package control

import (
	"fmt"
	"sync"
	"time"
)

// Standard trigger identifiers for automated canary rollback.
const (
	TriggerSpeculativeCollapse  = "speculative_acceptance_collapse"
	TriggerLatencySLABreach     = "ttft_p99_sla_breach"
	Trigger5xxErrorRateExceeded = "error_5xx_rate_exceeded"
)

// CanaryState represents the lifecycle phase of a newly applied configuration.
type CanaryState string

const (
	CanaryStateIdle       CanaryState = "IDLE"
	CanaryStateEvaluating CanaryState = "EVALUATING"
	CanaryStateStabilized CanaryState = "STABILIZED"
	CanaryStateRolledBack CanaryState = "ROLLED_BACK"
)

// TelemetrySample captures serving health indicators for watchdog evaluation.
type TelemetrySample struct {
	Timestamp                 time.Time `json:"timestamp"`
	SpeculativeAcceptanceRate float64   `json:"speculative_acceptance_rate"`
	TTFTp99MS                 float64   `json:"ttft_p99_ms"`
	Error5xxRate              float64   `json:"error_5xx_rate"`
	TotalRequests             uint64    `json:"total_requests,omitempty"`
	Errors5xx                 uint64    `json:"errors_5xx,omitempty"`
}

// WatchdogConfig specifies operational tolerances and time horizons for canary monitoring.
type WatchdogConfig struct {
	StabilizationWindow      time.Duration `json:"stabilization_window"`
	Max5xxErrorRate          float64       `json:"max_5xx_error_rate"`
	MaxAcceptanceDropRatio   float64       `json:"max_acceptance_drop_ratio"`
	DefaultDeclaredLatencyMS float64       `json:"default_declared_latency_ms"`
}

// DefaultWatchdogConfig returns production-calibrated defaults:
// 30s window, 0.1% max 5xx rate, 50% acceptance drop ceiling.
func DefaultWatchdogConfig() WatchdogConfig {
	return WatchdogConfig{
		StabilizationWindow:      30 * time.Second,
		Max5xxErrorRate:          0.001, // 0.1%
		MaxAcceptanceDropRatio:   0.50,  // 50% collapse
		DefaultDeclaredLatencyMS: 250.0,
	}
}

// RollbackHandler is invoked when a canary anomaly triggers an automated rollback.
type RollbackHandler func(trigger, detail string) error

// PromoteHandler is invoked when a canary configuration stabilizes and becomes LKG.
type PromoteHandler func(epoch uint64) error

// Watchdog continuously guards applied configurations against performance degradation.
type Watchdog struct {
	mu    sync.Mutex
	cfg   WatchdogConfig
	state CanaryState
	epoch uint64
	slaMS float64

	baselineAcceptance float64
	evaluatedEpoch     uint64
	startedAt          time.Time

	onRollback RollbackHandler
	onPromote  PromoteHandler
}

// NewWatchdog creates a Watchdog with the provided configuration and hooks.
func NewWatchdog(cfg WatchdogConfig, onRollback RollbackHandler, onPromote PromoteHandler) *Watchdog {
	if cfg.StabilizationWindow <= 0 {
		cfg.StabilizationWindow = 30 * time.Second
	}
	if cfg.Max5xxErrorRate <= 0 {
		cfg.Max5xxErrorRate = 0.001
	}
	if cfg.MaxAcceptanceDropRatio <= 0 {
		cfg.MaxAcceptanceDropRatio = 0.50
	}
	if cfg.DefaultDeclaredLatencyMS <= 0 {
		cfg.DefaultDeclaredLatencyMS = 250.0
	}

	return &Watchdog{
		cfg:        cfg,
		state:      CanaryStateIdle,
		onRollback: onRollback,
		onPromote:  onPromote,
	}
}

// StartEvaluation arms the watchdog for a newly applied candidate epoch.
func (w *Watchdog) StartEvaluation(epoch uint64, slaMS float64, initialAcceptance float64) {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.evaluatedEpoch = epoch
	w.state = CanaryStateEvaluating
	w.startedAt = time.Now().UTC()

	if slaMS > 0 {
		w.slaMS = slaMS
	} else {
		w.slaMS = w.cfg.DefaultDeclaredLatencyMS
	}

	if initialAcceptance > 0 {
		w.baselineAcceptance = initialAcceptance
	}
}

// IngestTelemetry evaluates a single telemetry observation against rollback criteria.
// Returns (triggered, triggerName, detail).
func (w *Watchdog) IngestTelemetry(sample TelemetrySample) (bool, string, string) {
	w.mu.Lock()

	if w.state != CanaryStateEvaluating {
		w.mu.Unlock()
		return false, "", ""
	}

	// Compute effective 5xx rate if counts are supplied
	if sample.TotalRequests > 0 && sample.Errors5xx > 0 && sample.Error5xxRate == 0 {
		sample.Error5xxRate = float64(sample.Errors5xx) / float64(sample.TotalRequests)
	}

	var (
		doRollback     bool
		triggerName    string
		detail         string
		rollbackFn     RollbackHandler
		doPromote      bool
		evaluatedEpoch uint64
		promoteFn      PromoteHandler
	)

	// Check 1: 5xx error rate exceeds threshold (default 0.1%)
	if sample.Error5xxRate > w.cfg.Max5xxErrorRate {
		triggerName = Trigger5xxErrorRateExceeded
		detail = fmt.Sprintf("5xx error rate (%.4f%%) exceeded ceiling (%.4f%%)", sample.Error5xxRate*100, w.cfg.Max5xxErrorRate*100)
		w.state = CanaryStateRolledBack
		doRollback = true
		rollbackFn = w.onRollback
	} else if sample.TTFTp99MS > 0 && w.slaMS > 0 && sample.TTFTp99MS > w.slaMS {
		// Check 2: TTFT p99 latency breaches declared SLA
		triggerName = TriggerLatencySLABreach
		detail = fmt.Sprintf("TTFT p99 (%.2f ms) breached declared SLA (%.2f ms)", sample.TTFTp99MS, w.slaMS)
		w.state = CanaryStateRolledBack
		doRollback = true
		rollbackFn = w.onRollback
	} else if sample.SpeculativeAcceptanceRate > 0 {
		// Check 3: Speculative acceptance rate collapses by > 50%
		if w.baselineAcceptance == 0 {
			w.baselineAcceptance = sample.SpeculativeAcceptanceRate
		} else {
			collapseThreshold := w.baselineAcceptance * (1.0 - w.cfg.MaxAcceptanceDropRatio)
			if sample.SpeculativeAcceptanceRate < collapseThreshold {
				triggerName = TriggerSpeculativeCollapse
				detail = fmt.Sprintf("speculative acceptance rate collapsed to %.2f (baseline: %.2f, drop > %.0f%%)",
					sample.SpeculativeAcceptanceRate, w.baselineAcceptance, w.cfg.MaxAcceptanceDropRatio*100)
				w.state = CanaryStateRolledBack
				doRollback = true
				rollbackFn = w.onRollback
			}
		}
	}

	if !doRollback {
		// Check stabilization window completion
		now := time.Now().UTC()
		if !w.startedAt.IsZero() && now.Sub(w.startedAt) >= w.cfg.StabilizationWindow {
			w.state = CanaryStateStabilized
			doPromote = true
			evaluatedEpoch = w.evaluatedEpoch
			promoteFn = w.onPromote
		}
	}

	w.mu.Unlock()

	if doRollback {
		if rollbackFn != nil {
			_ = rollbackFn(triggerName, detail)
		}
		return true, triggerName, detail
	}

	if doPromote {
		if promoteFn != nil {
			_ = promoteFn(evaluatedEpoch)
		}
	}

	return false, "", ""
}

// CheckStabilization checks if the evaluation period has successfully elapsed without anomaly.
func (w *Watchdog) CheckStabilization(now time.Time) bool {
	w.mu.Lock()

	if w.state != CanaryStateEvaluating {
		stabilized := (w.state == CanaryStateStabilized)
		w.mu.Unlock()
		return stabilized
	}

	if now.Sub(w.startedAt) >= w.cfg.StabilizationWindow {
		w.state = CanaryStateStabilized
		evaluatedEpoch := w.evaluatedEpoch
		promoteFn := w.onPromote
		w.mu.Unlock()

		if promoteFn != nil {
			_ = promoteFn(evaluatedEpoch)
		}
		return true
	}

	w.mu.Unlock()
	return false
}

// Status returns current canary evaluation telemetry.
func (w *Watchdog) Status() (CanaryState, uint64, float64, time.Duration) {
	w.mu.Lock()
	defer w.mu.Unlock()

	var elapsed time.Duration
	if !w.startedAt.IsZero() {
		elapsed = time.Since(w.startedAt)
	}
	return w.state, w.evaluatedEpoch, w.baselineAcceptance, elapsed
}
