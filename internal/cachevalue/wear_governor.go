// Package cachevalue provides SSD endurance rate governance and SMART write
// amplification factor (WAF) tracking for local high-volume caching (#10964, #11076).
package cachevalue

import (
	"errors"
	"sync"
	"time"
)

// GovernorState represents the wear governor throttling state.
type GovernorState string

const (
	// GovernorGreen indicates normal operation (<80% of daily quota consumed).
	// Default admission filtering and standard quantization applied.
	GovernorGreen GovernorState = "GovernorGreen"

	// GovernorYellow indicates cautionary operation (80% to 100% consumed).
	// Tightened admission filtering and aggressive quantization enforced.
	GovernorYellow GovernorState = "GovernorYellow"

	// GovernorRed indicates daily quota exhaustion (>100% consumed).
	// Hard freeze: all local SSD staging writes are denied until the sliding
	// window recovers; caching falls back to DRAM LRU or recompute.
	GovernorRed GovernorState = "GovernorRed"
)

func (s GovernorState) String() string {
	return string(s)
}

func (s GovernorState) IsGreen() bool {
	return s == GovernorGreen
}

func (s GovernorState) IsYellow() bool {
	return s == GovernorYellow
}

func (s GovernorState) IsRed() bool {
	return s == GovernorRed
}

const (
	// DefaultTargetDays is the enterprise target operational lifespan (5 years = 1,825 days).
	DefaultTargetDays = 1825

	// DefaultTBWRatingBytes is the baseline endurance rating for a 1 TB TLC SSD (600 TBW).
	DefaultTBWRatingBytes int64 = 600 * 1000 * 1000 * 1000 * 1000

	// DefaultYellowThresholdRatio is the fraction of daily quota triggering GovernorYellow (80%).
	DefaultYellowThresholdRatio = 0.80

	// DefaultRedThresholdRatio is the fraction of daily quota triggering GovernorRed (100%).
	DefaultRedThresholdRatio = 1.00

	// DefaultSlidingWindowDuration is the duration over which write wear is bounded (24 hours).
	DefaultSlidingWindowDuration = 24 * time.Hour
)

// SMARTTelemetry captures NVMe hardware health metrics and empirical write amplification.
type SMARTTelemetry struct {
	NANDBytesWritten int64     `json:"nand_bytes_written"`
	HostBytesWritten int64     `json:"host_bytes_written"`
	DeltaNANDBytes   int64     `json:"delta_nand_bytes"`
	DeltaHostBytes   int64     `json:"delta_host_bytes"`
	EmpiricalWAF     float64   `json:"empirical_waf"`
	CumulativeWAF    float64   `json:"cumulative_waf"`
	UpdatedAt        time.Time `json:"updated_at"`
}

// SMARTCollector gathers raw NVMe SMART write metrics from the operating system or controller.
type SMARTCollector interface {
	CollectSMART() (nandBytesWritten, hostBytesWritten int64, err error)
}

// SMARTCollectorFunc adapts an ordinary function to the SMARTCollector interface.
type SMARTCollectorFunc func() (int64, int64, error)

func (f SMARTCollectorFunc) CollectSMART() (int64, int64, error) {
	return f()
}

// WearGovernorConfig defines endurance targets and sliding-window parameters.
type WearGovernorConfig struct {
	// TBWRatingBytes is the total manufacturer warrantied endurance in bytes (e.g. 600 TB for 1 TB TLC).
	TBWRatingBytes int64

	// TargetDays is the desired drive lifespan in days (defaults to 1825 = 5 years).
	TargetDays int

	// DailyQuotaBytes explicitly overrides TBWRatingBytes / TargetDays if > 0 (e.g. 300 GB/day).
	DailyQuotaBytes int64

	// WindowDuration specifies the sliding window duration (defaults to 24 hours).
	WindowDuration time.Duration

	// YellowThresholdRatio is the ratio of daily quota triggering GovernorYellow (defaults to 0.80).
	YellowThresholdRatio float64

	// RedThresholdRatio is the ratio of daily quota triggering GovernorRed (defaults to 1.00).
	RedThresholdRatio float64

	// NowFunc provides a mockable time source for deterministic testing.
	NowFunc func() time.Time
}

// DefaultWearGovernorConfig returns standard 5-year 600 TBW parameters for a 1 TB TLC SSD.
func DefaultWearGovernorConfig() WearGovernorConfig {
	return WearGovernorConfig{
		TBWRatingBytes:       DefaultTBWRatingBytes,
		TargetDays:           DefaultTargetDays,
		DailyQuotaBytes:      0,
		WindowDuration:       DefaultSlidingWindowDuration,
		YellowThresholdRatio: DefaultYellowThresholdRatio,
		RedThresholdRatio:    DefaultRedThresholdRatio,
	}
}

type writeRecord struct {
	timestamp time.Time
	bytes     int64
}

// WearGovernor manages SSD endurance budgets via a 24-hour sliding-window leaky bucket
// and tracks real-time empirical Write Amplification Factor (WAF) via NVMe SMART telemetry.
type WearGovernor struct {
	mu sync.Mutex

	dailyQuotaBytes int64
	windowDuration  time.Duration
	yellowRatio     float64
	redRatio        float64

	records     []writeRecord
	windowBytes int64

	lastNANDBytes       int64
	lastHostBytes       int64
	hasSMARTBaseline    bool
	telemetry           SMARTTelemetry
	cumulativeNANDDelta int64
	cumulativeHostDelta int64

	nowFunc    func() time.Time
	timeOffset time.Duration
}

// NewWearGovernor constructs an initialized wear governor from the provided configuration.
func NewWearGovernor(cfg WearGovernorConfig) *WearGovernor {
	dailyQuota := cfg.DailyQuotaBytes
	if dailyQuota <= 0 {
		tbw := cfg.TBWRatingBytes
		if tbw <= 0 {
			tbw = DefaultTBWRatingBytes
		}
		days := cfg.TargetDays
		if days <= 0 {
			days = DefaultTargetDays
		}
		dailyQuota = tbw / int64(days)
	}

	window := cfg.WindowDuration
	if window <= 0 {
		window = DefaultSlidingWindowDuration
	}

	yellow := cfg.YellowThresholdRatio
	if yellow <= 0 {
		yellow = DefaultYellowThresholdRatio
	}

	red := cfg.RedThresholdRatio
	if red <= 0 {
		red = DefaultRedThresholdRatio
	}

	nowFn := cfg.NowFunc
	if nowFn == nil {
		nowFn = time.Now
	}

	return &WearGovernor{
		dailyQuotaBytes: dailyQuota,
		windowDuration:  window,
		yellowRatio:     yellow,
		redRatio:        red,
		nowFunc:         nowFn,
		telemetry: SMARTTelemetry{
			EmpiricalWAF:  1.0,
			CumulativeWAF: 1.0,
		},
	}
}

func (g *WearGovernor) nowLocked() time.Time {
	base := g.nowFunc()
	if g.timeOffset != 0 {
		base = base.Add(g.timeOffset)
	}
	return base
}

func (g *WearGovernor) pruneExpiredLocked(now time.Time) {
	cutoff := now.Add(-g.windowDuration)
	prunedIdx := 0
	for prunedIdx < len(g.records) && g.records[prunedIdx].timestamp.Before(cutoff) {
		g.windowBytes -= g.records[prunedIdx].bytes
		prunedIdx++
	}
	if prunedIdx > 0 {
		if g.windowBytes < 0 {
			g.windowBytes = 0
		}
		g.records = g.records[prunedIdx:]
		if len(g.records) == 0 {
			g.records = nil
		} else if cap(g.records) > 2048 && len(g.records) < cap(g.records)/4 {
			fresh := make([]writeRecord, len(g.records))
			copy(fresh, g.records)
			g.records = fresh
		}
	}
}

func (g *WearGovernor) stateLocked() GovernorState {
	yellowThreshold := int64(float64(g.dailyQuotaBytes) * g.yellowRatio)
	redThreshold := int64(float64(g.dailyQuotaBytes) * g.redRatio)

	if g.windowBytes >= redThreshold {
		return GovernorRed
	}
	if g.windowBytes >= yellowThreshold {
		return GovernorYellow
	}
	return GovernorGreen
}

// RequestWritePermit adjudicates whether a proposed write of bytesToWrite should be admitted.
// Returns allowed=false and GovernorRed when the daily quota is exhausted (hard freeze) or if
// admitting bytesToWrite would breach the daily quota.
func (g *WearGovernor) RequestWritePermit(bytesToWrite int64) (allowed bool, state GovernorState) {
	g.mu.Lock()
	defer g.mu.Unlock()

	now := g.nowLocked()
	g.pruneExpiredLocked(now)

	state = g.stateLocked()
	if state == GovernorRed {
		return false, GovernorRed
	}

	redThreshold := int64(float64(g.dailyQuotaBytes) * g.redRatio)
	if bytesToWrite > 0 && g.windowBytes+bytesToWrite > redThreshold {
		return false, GovernorRed
	}

	return true, state
}

// RecordHostWrite records that bytesWritten have been physically issued to the host storage layer,
// adding the write event to the 24-hour sliding window.
func (g *WearGovernor) RecordHostWrite(bytesWritten int64) {
	if bytesWritten <= 0 {
		return
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	now := g.nowLocked()
	g.pruneExpiredLocked(now)

	g.records = append(g.records, writeRecord{
		timestamp: now,
		bytes:     bytesWritten,
	})
	g.windowBytes += bytesWritten
}

// UpdateSMARTTelemetry ingests raw NAND and Host write counters from NVMe SMART health logs
// and recalculates empirical WAF = Delta_NAND / Delta_Host.
func (g *WearGovernor) UpdateSMARTTelemetry(nandBytesWritten, hostBytesWritten int64) {
	g.mu.Lock()
	defer g.mu.Unlock()

	now := g.nowLocked()
	var deltaNAND, deltaHost int64

	if !g.hasSMARTBaseline {
		deltaNAND = nandBytesWritten
		deltaHost = hostBytesWritten
		g.lastNANDBytes = nandBytesWritten
		g.lastHostBytes = hostBytesWritten
		g.hasSMARTBaseline = true
	} else {
		if nandBytesWritten >= g.lastNANDBytes && hostBytesWritten >= g.lastHostBytes {
			deltaNAND = nandBytesWritten - g.lastNANDBytes
			deltaHost = hostBytesWritten - g.lastHostBytes
		} else {
			// Non-monotonic update: caller passed deltas directly
			deltaNAND = nandBytesWritten
			deltaHost = hostBytesWritten
		}
		g.lastNANDBytes = nandBytesWritten
		g.lastHostBytes = hostBytesWritten
	}

	waf := 1.0
	if deltaHost > 0 {
		waf = float64(deltaNAND) / float64(deltaHost)
	}

	if deltaNAND > 0 {
		g.cumulativeNANDDelta += deltaNAND
	}
	if deltaHost > 0 {
		g.cumulativeHostDelta += deltaHost
	}

	cumWAF := 1.0
	if g.cumulativeHostDelta > 0 {
		cumWAF = float64(g.cumulativeNANDDelta) / float64(g.cumulativeHostDelta)
	}

	g.telemetry = SMARTTelemetry{
		NANDBytesWritten: nandBytesWritten,
		HostBytesWritten: hostBytesWritten,
		DeltaNANDBytes:   deltaNAND,
		DeltaHostBytes:   deltaHost,
		EmpiricalWAF:     waf,
		CumulativeWAF:    cumWAF,
		UpdatedAt:        now,
	}
}

// SetSMARTBaseline explicitly establishes starting baseline counters for drives already in service.
func (g *WearGovernor) SetSMARTBaseline(nandBytesWritten, hostBytesWritten int64) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.lastNANDBytes = nandBytesWritten
	g.lastHostBytes = hostBytesWritten
	g.hasSMARTBaseline = true
}

// PollSMART reads from a SMARTCollector and refreshes empirical WAF telemetry.
func (g *WearGovernor) PollSMART(collector SMARTCollector) error {
	if collector == nil {
		return errors.New("cachevalue: nil SMARTCollector")
	}
	nand, host, err := collector.CollectSMART()
	if err != nil {
		return err
	}
	g.UpdateSMARTTelemetry(nand, host)
	return nil
}

// State returns the current throttling state of the governor.
func (g *WearGovernor) State() GovernorState {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.pruneExpiredLocked(g.nowLocked())
	return g.stateLocked()
}

// DailyQuotaBytes returns the active daily write budget in bytes.
func (g *WearGovernor) DailyQuotaBytes() int64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.dailyQuotaBytes
}

// WindowBytes returns total bytes written in the active 24-hour sliding window.
func (g *WearGovernor) WindowBytes() int64 {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.pruneExpiredLocked(g.nowLocked())
	return g.windowBytes
}

// WindowUtilization returns the fraction of daily quota currently consumed in the sliding window.
func (g *WearGovernor) WindowUtilization() float64 {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.dailyQuotaBytes <= 0 {
		return 0
	}
	g.pruneExpiredLocked(g.nowLocked())
	return float64(g.windowBytes) / float64(g.dailyQuotaBytes)
}

// EmpiricalWAF returns the most recently measured empirical Write Amplification Factor (Delta_NAND / Delta_Host).
func (g *WearGovernor) EmpiricalWAF() float64 {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.telemetry.EmpiricalWAF > 0 {
		return g.telemetry.EmpiricalWAF
	}
	return 1.0
}

// WAF is a convenient alias for EmpiricalWAF.
func (g *WearGovernor) WAF() float64 {
	return g.EmpiricalWAF()
}

// CumulativeWAF returns the cumulative empirical Write Amplification Factor since governor inception.
func (g *WearGovernor) CumulativeWAF() float64 {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.telemetry.CumulativeWAF > 0 {
		return g.telemetry.CumulativeWAF
	}
	return 1.0
}

// Telemetry returns a snapshot of the latest SMART write and WAF telemetry.
func (g *WearGovernor) Telemetry() SMARTTelemetry {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.telemetry
}

// AdvanceTime advances the internal simulated clock offset by duration d (for deterministic testing).
func (g *WearGovernor) AdvanceTime(d time.Duration) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.timeOffset += d
}

// SetNowFunc overrides the clock source for deterministic testing.
func (g *WearGovernor) SetNowFunc(fn func() time.Time) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.nowFunc = fn
}
