package telemetry

import (
	"fmt"
	"io"
	"strings"
	"sync"
	"time"
)

// MetalMTPAcceptanceStats mirrors the runtime stats shape reported by the
// Metal MTP coordinator (internal/model.MetalMTPAcceptanceStats), enabling
// loose coupling without package dependency cycles.
type MetalMTPAcceptanceStats struct {
	TotalProposed   int     `json:"total_proposed"`
	TotalAccepted   int     `json:"total_accepted"`
	TotalRollbacks  int     `json:"total_rollbacks"`
	WindowProposed  int     `json:"window_proposed"`
	WindowAccepted  int     `json:"window_accepted"`
	RollingRate     float64 `json:"rolling_acceptance_rate"`
	LifetimeRate    float64 `json:"lifetime_acceptance_rate"`
	InFallback      bool    `json:"in_fallback"`
	FallbackReason  string  `json:"fallback_reason,omitempty"`
	TripwireTripped bool    `json:"tripwire_tripped"`
	TripwireReason  string  `json:"tripwire_reason,omitempty"`
	CommittedPages  int     `json:"committed_pages"`
	FreedPages      int     `json:"freed_pages"`
	TotalGenerated  int     `json:"total_generated"`
}

// MTPMetrics captures a point-in-time snapshot of live speculative decoding metrics.
type MTPMetrics struct {
	TotalProposed        int64     `json:"total_proposed"`
	TotalAccepted        int64     `json:"total_accepted"`
	TotalRollbacks       int64     `json:"total_rollbacks"`
	WindowProposed       int64     `json:"window_proposed"`
	WindowAccepted       int64     `json:"window_accepted"`
	RollingRate          float64   `json:"rolling_acceptance_rate"`
	LifetimeRate         float64   `json:"lifetime_acceptance_rate"`
	AcceptanceRate       float64   `json:"mtp_acceptance_rate"` // rolling acceptance rate alias matching witness contract
	DecodeSpeedup        float64   `json:"mtp_decode_speedup"`  // effective decode tok/s multiplier
	RollbackCount        int64     `json:"mtp_rollback_count"`  // total rollbacks alias matching witness contract
	CommittedPages       int64     `json:"committed_pages"`
	FreedPages           int64     `json:"freed_pages"`
	TotalGenerated       int64     `json:"total_generated"`
	InFallback           bool      `json:"in_fallback"`
	FallbackReason       string    `json:"fallback_reason,omitempty"`
	TripwireTripped      bool      `json:"tripwire_tripped"`
	TripwireReason       string    `json:"tripwire_reason,omitempty"`
	TokensSavedPerMinute float64   `json:"tokens_saved_per_minute"`
	StartTime            time.Time `json:"start_time"`
	UpdatedAt            time.Time `json:"updated_at"`
}

// TUISummary returns a compact single-line summary formatted for status lines.
func (m MTPMetrics) TUISummary() string {
	if m.InFallback {
		reason := m.FallbackReason
		if reason == "" {
			reason = "serial fallback"
		}
		return fmt.Sprintf("mtp: fallback (%s) · %d rollbacks", reason, m.RollbackCount)
	}
	return fmt.Sprintf("mtp: %.1f%% acc · ×%.2f · %d rollbacks · pages %d/%d",
		m.AcceptanceRate*100, m.DecodeSpeedup, m.RollbackCount, m.CommittedPages, m.FreedPages)
}

// TUIBlock returns formatted multi-line dashboard rows including a horizontal gauge bar.
func (m MTPMetrics) TUIBlock(gaugeWidth int) []string {
	if gaugeWidth <= 0 {
		gaugeWidth = 10
	}
	if m.InFallback {
		reason := m.FallbackReason
		if reason == "" {
			reason = "serial fallback"
		}
		return []string{
			fmt.Sprintf(" mtp    [FALLBACK] serial decode (%s)", reason),
			fmt.Sprintf("        %d rollbacks · pages %d committed, %d freed", m.RollbackCount, m.CommittedPages, m.FreedPages),
		}
	}

	bar := formatProgressBar(m.AcceptanceRate, gaugeWidth)
	line1 := fmt.Sprintf(" mtp    %s %.1f%% acc · ×%.2f speedup · life %.1f%%",
		bar, m.AcceptanceRate*100, m.DecodeSpeedup, m.LifetimeRate*100)
	line2 := fmt.Sprintf("        %d rollbacks · pages %d committed, %d freed · %.1f tok/min saved",
		m.RollbackCount, m.CommittedPages, m.FreedPages, m.TokensSavedPerMinute)
	return []string{line1, line2}
}

func formatProgressBar(frac float64, width int) string {
	if width <= 0 {
		return ""
	}
	if frac < 0 {
		frac = 0
	}
	if frac > 1 {
		frac = 1
	}
	filled := int(frac * float64(width))
	if filled > width {
		filled = width
	}
	empty := width - filled
	return "[" + strings.Repeat("█", filled) + strings.Repeat("░", empty) + "]"
}

// Prometheus returns OpenMetrics/Prometheus formatted exposition text.
func (m MTPMetrics) Prometheus() string {
	var b strings.Builder
	writePromHelp(&b, "fak_mtp_rolling_acceptance_rate", "Rolling acceptance rate of MTP speculative draft tokens", "gauge")
	fmt.Fprintf(&b, "fak_mtp_rolling_acceptance_rate %0.6f\n", m.RollingRate)

	writePromHelp(&b, "fak_mtp_lifetime_acceptance_rate", "Lifetime acceptance rate of MTP speculative draft tokens", "gauge")
	fmt.Fprintf(&b, "fak_mtp_lifetime_acceptance_rate %0.6f\n", m.LifetimeRate)

	writePromHelp(&b, "fak_mtp_acceptance_rate", "Real-time MTP speculative acceptance rate (alias matching witness)", "gauge")
	fmt.Fprintf(&b, "fak_mtp_acceptance_rate %0.6f\n", m.AcceptanceRate)

	writePromHelp(&b, "fak_mtp_decode_speedup", "Effective decode speedup multiplier from MTP speculative execution", "gauge")
	fmt.Fprintf(&b, "fak_mtp_decode_speedup %0.6f\n", m.DecodeSpeedup)

	writePromHelp(&b, "fak_mtp_rollback_count", "Total count of MTP speculative draft rollbacks", "counter")
	fmt.Fprintf(&b, "fak_mtp_rollback_count %d\n", m.RollbackCount)

	writePromHelp(&b, "fak_mtp_tokens_proposed_total", "Total count of MTP draft tokens proposed", "counter")
	fmt.Fprintf(&b, "fak_mtp_tokens_proposed_total %d\n", m.TotalProposed)

	writePromHelp(&b, "fak_mtp_tokens_accepted_total", "Total count of MTP draft tokens accepted", "counter")
	fmt.Fprintf(&b, "fak_mtp_tokens_accepted_total %d\n", m.TotalAccepted)

	writePromHelp(&b, "fak_mtp_tokens_generated_total", "Total tokens generated including speculative acceptance and verification", "counter")
	fmt.Fprintf(&b, "fak_mtp_tokens_generated_total %d\n", m.TotalGenerated)

	writePromHelp(&b, "fak_mtp_pages_committed_total", "Total speculative KV cache pages committed", "counter")
	fmt.Fprintf(&b, "fak_mtp_pages_committed_total %d\n", m.CommittedPages)

	writePromHelp(&b, "fak_mtp_pages_freed_total", "Total speculative KV cache pages freed on rollback", "counter")
	fmt.Fprintf(&b, "fak_mtp_pages_freed_total %d\n", m.FreedPages)

	writePromHelp(&b, "fak_mtp_fallback_active", "Whether MTP is currently in serial fallback mode (1 for true, 0 for false)", "gauge")
	fallbackVal := 0
	if m.InFallback {
		fallbackVal = 1
	}
	fmt.Fprintf(&b, "fak_mtp_fallback_active %d\n", fallbackVal)

	writePromHelp(&b, "fak_mtp_tripwire_tripped", "Whether MTP greedy temperature zero tripwire was tripped (1 for true, 0 for false)", "gauge")
	tripwireVal := 0
	if m.TripwireTripped {
		tripwireVal = 1
	}
	fmt.Fprintf(&b, "fak_mtp_tripwire_tripped %d\n", tripwireVal)

	writePromHelp(&b, "fak_mtp_tokens_saved_per_minute", "Estimated tokens saved per minute by speculative execution", "gauge")
	fmt.Fprintf(&b, "fak_mtp_tokens_saved_per_minute %0.6f\n", m.TokensSavedPerMinute)

	return b.String()
}

// WritePrometheus writes Prometheus text format to w.
func (m MTPMetrics) WritePrometheus(w io.Writer) error {
	_, err := io.WriteString(w, m.Prometheus())
	return err
}

func writePromHelp(b *strings.Builder, name, help, mType string) {
	b.WriteString("# HELP ")
	b.WriteString(name)
	b.WriteString(" ")
	b.WriteString(help)
	b.WriteString("\n# TYPE ")
	b.WriteString(name)
	b.WriteString(" ")
	b.WriteString(mType)
	b.WriteString("\n")
}

type windowStep struct {
	proposed int
	accepted int
}

// Collector aggregates live speculative decoding observations and exposes thread-safe metrics.
type Collector struct {
	mu sync.RWMutex

	startTime   time.Time
	lastUpdated time.Time

	windowCapacity int
	window         []windowStep

	totalSteps      int64
	totalProposed   int64
	totalAccepted   int64
	totalRollbacks  int64
	totalGenerated  int64
	committedPages  int64
	freedPages      int64
	inFallback      bool
	fallbackReason  string
	tripwireTripped bool
	tripwireReason  string

	explicitSpeedup float64
}

// NewCollector creates an initialized Collector with a specified sliding window capacity.
func NewCollector(windowSize int) *Collector {
	if windowSize <= 0 {
		windowSize = 32
	}
	now := time.Now()
	return &Collector{
		startTime:      now,
		lastUpdated:    now,
		windowCapacity: windowSize,
		window:         make([]windowStep, 0, windowSize),
	}
}

var (
	defaultCollectorMu sync.RWMutex
	defaultCollector   = NewCollector(32)
)

// DefaultCollector returns the global default telemetry Collector.
func DefaultCollector() *Collector {
	defaultCollectorMu.RLock()
	defer defaultCollectorMu.RUnlock()
	return defaultCollector
}

// ResetDefaultCollector resets the default global collector (useful in tests).
func ResetDefaultCollector() {
	defaultCollectorMu.Lock()
	defer defaultCollectorMu.Unlock()
	defaultCollector = NewCollector(32)
}

// RecordStep records an individual speculative decoding step.
func (c *Collector) RecordStep(proposed, accepted, committedPages, freedPages int, rollback bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.lastUpdated = time.Now()
	c.totalSteps++
	c.totalProposed += int64(proposed)
	c.totalAccepted += int64(accepted)
	// Each verification step produces accepted draft tokens + 1 verification token.
	c.totalGenerated += int64(accepted + 1)
	c.committedPages += int64(committedPages)
	c.freedPages += int64(freedPages)
	if rollback {
		c.totalRollbacks++
	}

	if len(c.window) >= c.windowCapacity {
		c.window = c.window[1:]
	}
	c.window = append(c.window, windowStep{proposed: proposed, accepted: accepted})
}

// RecordMetalStats incorporates a stats snapshot from the Metal MTP coordinator.
func (c *Collector) RecordMetalStats(stats MetalMTPAcceptanceStats) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.lastUpdated = time.Now()
	c.totalProposed = int64(stats.TotalProposed)
	c.totalAccepted = int64(stats.TotalAccepted)
	c.totalRollbacks = int64(stats.TotalRollbacks)
	c.committedPages = int64(stats.CommittedPages)
	c.freedPages = int64(stats.FreedPages)
	c.totalGenerated = int64(stats.TotalGenerated)
	c.inFallback = stats.InFallback
	c.fallbackReason = stats.FallbackReason
	c.tripwireTripped = stats.TripwireTripped
	c.tripwireReason = stats.TripwireReason

	// If coordinator reports window stats directly, synchronize them
	if stats.WindowProposed > 0 {
		c.window = c.window[:0]
		c.window = append(c.window, windowStep{
			proposed: stats.WindowProposed,
			accepted: stats.WindowAccepted,
		})
	}
}

// SetFallback sets the current serial fallback state and reason.
func (c *Collector) SetFallback(inFallback bool, reason string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.inFallback = inFallback
	c.fallbackReason = reason
	c.lastUpdated = time.Now()
}

// SetTripwire sets the current greedy tripwire state and reason.
func (c *Collector) SetTripwire(tripped bool, reason string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.tripwireTripped = tripped
	c.tripwireReason = reason
	c.lastUpdated = time.Now()
}

// SetExplicitSpeedup sets an override decode speedup multiplier (e.g. if measured directly).
func (c *Collector) SetExplicitSpeedup(speedup float64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.explicitSpeedup = speedup
	c.lastUpdated = time.Now()
}

// Snapshot returns a point-in-time MTPMetrics snapshot.
func (c *Collector) Snapshot() MTPMetrics {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var winProposed, winAccepted int64
	for _, w := range c.window {
		winProposed += int64(w.proposed)
		winAccepted += int64(w.accepted)
	}

	rollingRate := 1.0
	if winProposed > 0 {
		rollingRate = float64(winAccepted) / float64(winProposed)
	}

	lifetimeRate := 1.0
	if c.totalProposed > 0 {
		lifetimeRate = float64(c.totalAccepted) / float64(c.totalProposed)
	}

	speedup := 1.0
	if c.explicitSpeedup > 0 {
		speedup = c.explicitSpeedup
	} else if c.inFallback {
		speedup = 1.0
	} else if c.totalSteps > 0 && c.totalProposed > 0 {
		// Analytical speedup: (accepted + rounds) / (rounds + c_draft * proposed)
		// with c_draft = 0.25
		tokensProduced := float64(c.totalAccepted + c.totalSteps)
		costFactor := float64(c.totalSteps) + 0.25*float64(c.totalProposed)
		if costFactor > 0 {
			speedup = tokensProduced / costFactor
		}
	} else if c.totalAccepted > 0 && c.totalProposed > 0 {
		tokensProduced := float64(c.totalAccepted + 1)
		costFactor := 1.0 + 0.25*float64(c.totalProposed)
		speedup = tokensProduced / costFactor
	}

	now := time.Now()
	elapsed := now.Sub(c.startTime)
	var tokensSavedPerMin float64
	if elapsed >= time.Second && c.totalAccepted > 0 {
		tokensSavedPerMin = float64(c.totalAccepted) / elapsed.Minutes()
	}

	totalGen := c.totalGenerated
	if totalGen == 0 && c.totalAccepted > 0 {
		totalGen = c.totalAccepted + c.totalSteps
	}

	return MTPMetrics{
		TotalProposed:        c.totalProposed,
		TotalAccepted:        c.totalAccepted,
		TotalRollbacks:       c.totalRollbacks,
		WindowProposed:       winProposed,
		WindowAccepted:       winAccepted,
		RollingRate:          rollingRate,
		LifetimeRate:         lifetimeRate,
		AcceptanceRate:       rollingRate,
		DecodeSpeedup:        speedup,
		RollbackCount:        c.totalRollbacks,
		CommittedPages:       c.committedPages,
		FreedPages:           c.freedPages,
		TotalGenerated:       totalGen,
		InFallback:           c.inFallback,
		FallbackReason:       c.fallbackReason,
		TripwireTripped:      c.tripwireTripped,
		TripwireReason:       c.tripwireReason,
		TokensSavedPerMinute: tokensSavedPerMin,
		StartTime:            c.startTime,
		UpdatedAt:            c.lastUpdated,
	}
}

// Prometheus returns Prometheus exposition text for the collector's current state.
func (c *Collector) Prometheus() string {
	return c.Snapshot().Prometheus()
}
