package cacheprice

import (
	"sync"
	"time"
)

// dedup_check.go implements self-disabling remote existence checks when check overhead
// exceeds avoided transfer savings (#10274). An optimization must pay rent continuously,
// not only in the benchmark that justified its initial default: checking whether a remote
// cache contains an item avoids payload transfer on a hit, but pays check RPC latency on
// every trial. When losing streaks or measured overhead make checks a net loss on the same
// net-true time basis, the controller disables ordinary checks while admitting sparse probes
// for recovery, and fails zero or unknown telemetry to a declared conservative default.

// DedupAction is the controller's verdict on whether to perform a remote dedup existence check.
type DedupAction int

const (
	// ActionSkip skips the remote existence check and proceeds directly to transfer.
	// Defined as the zero value so uninitialized / default configurations fail conservatively.
	ActionSkip DedupAction = iota
	// ActionCheck permits an ordinary remote dedup existence check.
	ActionCheck
	// ActionProbe admits a sparse test check while ordinary checks are disabled.
	ActionProbe
)

// String returns the canonical name of the dedup decision.
func (d DedupAction) String() string {
	switch d {
	case ActionSkip:
		return "skip"
	case ActionCheck:
		return "check"
	case ActionProbe:
		return "probe"
	default:
		return "unknown"
	}
}

// ShouldCheck reports whether the decision permits executing a check (ordinary or probe).
func (d DedupAction) ShouldCheck() bool {
	return d == ActionCheck || d == ActionProbe
}

// DedupReason documents the operational basis for a dedup check decision.
type DedupReason string

const (
	// ReasonSavingsFavorable indicates estimated savings strictly exceeds check cost.
	ReasonSavingsFavorable DedupReason = "savings_favorable"
	// ReasonNegativeSavings indicates check cost meets or exceeds avoided transfer savings.
	ReasonNegativeSavings DedupReason = "negative_savings"
	// ReasonLosingStreak indicates ordinary checks are disabled after consecutive check misses.
	ReasonLosingStreak DedupReason = "losing_streak"
	// ReasonDisabled indicates ordinary checks remain disabled.
	ReasonDisabled DedupReason = "checks_disabled"
	// ReasonProbeAdmitted indicates a sparse check is admitted to probe for cache recovery.
	ReasonProbeAdmitted DedupReason = "sparse_probe_admitted"
	// ReasonZeroTelemetry indicates telemetry is missing or non-positive, falling back to conservative default.
	ReasonZeroTelemetry DedupReason = "zero_telemetry_conservative_default"
	// ReasonConservativeDefault indicates explicit fallback to the declared conservative default.
	ReasonConservativeDefault DedupReason = "conservative_default"
)

// DedupCheckInput provides candidate metadata for deciding whether to perform a dedup check.
// Content keys (hashes, file paths, URLs, cache keys) are DELIBERATELY EXCLUDED to preserve privacy.
type DedupCheckInput struct {
	// CheckOverhead is the estimated or measured duration of performing the remote existence check.
	CheckOverhead time.Duration `json:"check_overhead"`

	// AvoidedTransfer is the estimated or measured duration of transferring the payload
	// across the network if the item is not deduplicated.
	AvoidedTransfer time.Duration `json:"avoided_transfer"`

	// PayloadBytes is an optional size of the candidate payload in bytes.
	PayloadBytes int64 `json:"payload_bytes,omitempty"`

	// TransferBandwidthBytesPerSec is an optional network transfer rate in bytes per second.
	// If AvoidedTransfer is zero and PayloadBytes and TransferBandwidthBytesPerSec are > 0,
	// AvoidedTransfer is derived from PayloadBytes / TransferBandwidthBytesPerSec.
	TransferBandwidthBytesPerSec float64 `json:"transfer_bandwidth_bytes_per_sec,omitempty"`
}

// EffectiveAvoidedTransfer returns the avoided transfer duration, using AvoidedTransfer
// if positive, or deriving it from PayloadBytes and TransferBandwidthBytesPerSec if available.
func (in DedupCheckInput) EffectiveAvoidedTransfer() time.Duration {
	if in.AvoidedTransfer > 0 {
		return in.AvoidedTransfer
	}
	if in.PayloadBytes > 0 && in.TransferBandwidthBytesPerSec > 0 {
		return TransferDuration(in.PayloadBytes, in.TransferBandwidthBytesPerSec)
	}
	return 0
}

// HasTelemetry reports whether the input carries valid, positive telemetry for both
// check overhead and avoided transfer duration.
func (in DedupCheckInput) HasTelemetry() bool {
	return in.CheckOverhead > 0 && in.EffectiveAvoidedTransfer() > 0
}

// DedupCheckReceipt records the audited inputs, decision, reason, and net cumulative savings
// for a single dedup existence check evaluation. Deliberately carries NO content keys or digests.
type DedupCheckReceipt struct {
	Action            DedupAction     `json:"action"`
	ActionName        string          `json:"action_name"`
	Reason            DedupReason     `json:"reason"`
	Admitted          bool            `json:"admitted"`
	IsProbe           bool            `json:"is_probe"`
	Inputs            DedupCheckInput `json:"inputs"`
	CumulativeSavings time.Duration   `json:"cumulative_savings"`
	ConsecutiveMisses int             `json:"consecutive_misses"`
	ControllerEnabled bool            `json:"controller_enabled"`
}

// DedupObservation reports the measured outcome of an executed dedup check.
// Content keys are deliberately omitted.
type DedupObservation struct {
	// Hit indicates whether the remote cache contained the item (avoiding transfer).
	Hit bool `json:"hit"`

	// CheckOverhead is the measured duration spent performing the remote check.
	CheckOverhead time.Duration `json:"check_overhead"`

	// AvoidedTransfer is the measured or estimated transfer duration saved if Hit is true.
	AvoidedTransfer time.Duration `json:"avoided_transfer"`

	// IsProbe indicates whether this check was executed as a sparse probe.
	IsProbe bool `json:"is_probe"`
}

// DedupCheckConfig configures self-disabling dedup check economics and recovery.
type DedupCheckConfig struct {
	// MaxLosingStreak is the maximum number of consecutive check misses permitted
	// before ordinary checks are disabled. Defaults to 5.
	MaxLosingStreak int `json:"max_losing_streak"`

	// ProbeInterval is the number of skipped requests between sparse probe checks
	// while ordinary checks are disabled. Defaults to 16.
	ProbeInterval int `json:"probe_interval"`

	// ConservativeDefault is the decision emitted when candidate or historical telemetry
	// is zero, missing, or non-positive. Defaults to ActionSkip.
	ConservativeDefault DedupAction `json:"conservative_default"`

	// MinTransferSavingsRatio is the minimum ratio of avoided transfer to check overhead
	// required for a check to be considered favorable (avoidedTransfer >= ratio * checkOverhead).
	// Defaults to 1.0 (strict break-even).
	MinTransferSavingsRatio float64 `json:"min_transfer_savings_ratio"`

	// MinChecksForSavings is the minimum number of observed checks before negative cumulative
	// savings triggers self-disabling (allowing an initial sample to establish). Defaults to MaxLosingStreak.
	MinChecksForSavings int `json:"min_checks_for_savings,omitempty"`

	// StartDisabled starts the controller in disabled state, requiring a favorable probe to enable.
	StartDisabled bool `json:"start_disabled,omitempty"`

	// DefaultCheckOverhead is an optional fallback check overhead when per-call telemetry is zero.
	DefaultCheckOverhead time.Duration `json:"default_check_overhead,omitempty"`

	// DefaultAvoidedTransfer is an optional fallback transfer duration when per-call telemetry is zero.
	DefaultAvoidedTransfer time.Duration `json:"default_avoided_transfer,omitempty"`
}

// DefaultDedupCheckConfig returns the canonical production configuration:
// 5 consecutive misses disable ordinary checks, 1 in 16 skipped checks probes for recovery,
// and zero telemetry fails to ActionSkip.
func DefaultDedupCheckConfig() DedupCheckConfig {
	return DedupCheckConfig{
		MaxLosingStreak:         5,
		ProbeInterval:           16,
		ConservativeDefault:     ActionSkip,
		MinTransferSavingsRatio: 1.0,
		MinChecksForSavings:     5,
	}
}

// DedupControllerStats captures a point-in-time snapshot of controller telemetry and savings.
type DedupControllerStats struct {
	Enabled                   bool          `json:"enabled"`
	ConsecutiveMisses         int           `json:"consecutive_misses"`
	TotalChecks               int64         `json:"total_checks"`
	TotalHits                 int64         `json:"total_hits"`
	TotalMisses               int64         `json:"total_misses"`
	TotalProbes               int64         `json:"total_probes"`
	ProbeHits                 int64         `json:"probe_hits"`
	CumulativeOverhead        time.Duration `json:"cumulative_overhead"`
	CumulativeAvoidedTransfer time.Duration `json:"cumulative_avoided_transfer"`
	CumulativeSavings         time.Duration `json:"cumulative_savings"`
	HitRate                   float64       `json:"hit_rate"`
}

// DedupCheckController coordinates remote existence checks by self-disabling when check cost
// exceeds avoided transfer, admitting sparse recovery probes while disabled, and failing
// unknown telemetry to a declared conservative default.
type DedupCheckController struct {
	mu                        sync.Mutex
	cfg                       DedupCheckConfig
	enabled                   bool
	consecutiveMisses         int
	skipCountSinceLastProbe   int
	totalChecks               int64
	totalHits                 int64
	totalMisses               int64
	totalProbes               int64
	probeHits                 int64
	cumulativeOverhead        time.Duration
	cumulativeAvoidedTransfer time.Duration
	cumulativeSavings         time.Duration
	lastReason                DedupReason
}

// NewDedupCheckController returns a controller initialized with the provided configuration.
// Zero or invalid settings are defensively defaulted.
func NewDedupCheckController(cfg DedupCheckConfig) *DedupCheckController {
	if cfg.MaxLosingStreak <= 0 {
		cfg.MaxLosingStreak = 5
	}
	if cfg.ProbeInterval <= 0 {
		cfg.ProbeInterval = 16
	}
	if cfg.MinTransferSavingsRatio <= 0 {
		cfg.MinTransferSavingsRatio = 1.0
	}
	if cfg.MinChecksForSavings <= 0 {
		cfg.MinChecksForSavings = cfg.MaxLosingStreak
	}

	enabled := !cfg.StartDisabled
	lastReason := ReasonSavingsFavorable
	if !enabled {
		lastReason = ReasonDisabled
	}

	return &DedupCheckController{
		cfg:        cfg,
		enabled:    enabled,
		lastReason: lastReason,
	}
}

// ShouldCheck evaluates whether to perform a remote dedup check for the given input.
// Returns a bounded receipt recording the input, decision, reason, and cumulative savings.
// Contains no content keys.
func (c *DedupCheckController) ShouldCheck(input DedupCheckInput) DedupCheckReceipt {
	if c == nil {
		return DedupCheckReceipt{
			Action:     ActionSkip,
			ActionName: ActionSkip.String(),
			Reason:     ReasonConservativeDefault,
			Admitted:   false,
			Inputs:     input,
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Resolve effective telemetry incorporating any defaults.
	overhead := input.CheckOverhead
	if overhead <= 0 && c.cfg.DefaultCheckOverhead > 0 {
		overhead = c.cfg.DefaultCheckOverhead
	}
	avoided := input.EffectiveAvoidedTransfer()
	if avoided <= 0 && c.cfg.DefaultAvoidedTransfer > 0 {
		avoided = c.cfg.DefaultAvoidedTransfer
	}

	// 1. Zero / unknown telemetry: fails to declared conservative default.
	if overhead <= 0 || avoided <= 0 {
		action := c.cfg.ConservativeDefault
		return DedupCheckReceipt{
			Action:            action,
			ActionName:        action.String(),
			Reason:            ReasonZeroTelemetry,
			Admitted:          action.ShouldCheck(),
			IsProbe:           action == ActionProbe,
			Inputs:            input,
			CumulativeSavings: c.cumulativeSavings,
			ConsecutiveMisses: c.consecutiveMisses,
			ControllerEnabled: c.enabled,
		}
	}

	// 2. Candidate economics: does this check cost more than the transfer it avoids?
	if !DedupCheckWorthwhile(overhead, avoided, c.cfg.MinTransferSavingsRatio) {
		return DedupCheckReceipt{
			Action:            ActionSkip,
			ActionName:        ActionSkip.String(),
			Reason:            ReasonNegativeSavings,
			Admitted:          false,
			IsProbe:           false,
			Inputs:            input,
			CumulativeSavings: c.cumulativeSavings,
			ConsecutiveMisses: c.consecutiveMisses,
			ControllerEnabled: c.enabled,
		}
	}

	// 3. Controller state: evaluate self-disabling conditions.
	if c.enabled {
		if c.consecutiveMisses >= c.cfg.MaxLosingStreak {
			c.enabled = false
			c.lastReason = ReasonLosingStreak
		} else if c.totalChecks >= int64(c.cfg.MinChecksForSavings) && c.cumulativeSavings < 0 {
			c.enabled = false
			c.lastReason = ReasonNegativeSavings
		}
	}

	// 4. If disabled, admit sparse recovery probes.
	if !c.enabled {
		c.skipCountSinceLastProbe++
		if c.skipCountSinceLastProbe >= c.cfg.ProbeInterval {
			c.skipCountSinceLastProbe = 0
			return DedupCheckReceipt{
				Action:            ActionProbe,
				ActionName:        ActionProbe.String(),
				Reason:            ReasonProbeAdmitted,
				Admitted:          true,
				IsProbe:           true,
				Inputs:            input,
				CumulativeSavings: c.cumulativeSavings,
				ConsecutiveMisses: c.consecutiveMisses,
				ControllerEnabled: false,
			}
		}

		reason := c.lastReason
		if reason == "" {
			reason = ReasonDisabled
		}
		return DedupCheckReceipt{
			Action:            ActionSkip,
			ActionName:        ActionSkip.String(),
			Reason:            reason,
			Admitted:          false,
			IsProbe:           false,
			Inputs:            input,
			CumulativeSavings: c.cumulativeSavings,
			ConsecutiveMisses: c.consecutiveMisses,
			ControllerEnabled: false,
		}
	}

	// 5. Ordinary check admitted.
	return DedupCheckReceipt{
		Action:            ActionCheck,
		ActionName:        ActionCheck.String(),
		Reason:            ReasonSavingsFavorable,
		Admitted:          true,
		IsProbe:           false,
		Inputs:            input,
		CumulativeSavings: c.cumulativeSavings,
		ConsecutiveMisses: c.consecutiveMisses,
		ControllerEnabled: true,
	}
}

// Evaluate is an alias for ShouldCheck.
func (c *DedupCheckController) Evaluate(input DedupCheckInput) DedupCheckReceipt {
	return c.ShouldCheck(input)
}

// Observe records the outcome of an executed dedup check, updating cumulative savings,
// losing streak counters, and state transitions. Returns the net savings realized by this
// check and whether the controller is enabled following this observation.
func (c *DedupCheckController) Observe(obs DedupObservation) (netSavings time.Duration, enabled bool) {
	if c == nil {
		return 0, false
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if obs.CheckOverhead < 0 {
		obs.CheckOverhead = 0
	}
	if obs.AvoidedTransfer < 0 {
		obs.AvoidedTransfer = 0
	}

	netSavings = DedupNetSavings(obs.Hit, obs.CheckOverhead, obs.AvoidedTransfer)

	c.totalChecks++
	c.cumulativeOverhead += obs.CheckOverhead
	if obs.Hit {
		c.totalHits++
		c.cumulativeAvoidedTransfer += obs.AvoidedTransfer
		c.consecutiveMisses = 0
	} else {
		c.totalMisses++
		c.consecutiveMisses++
	}
	c.cumulativeSavings += netSavings

	if obs.IsProbe {
		c.totalProbes++
		if obs.Hit {
			c.probeHits++
		}
	}

	if !c.enabled {
		// Recovery probe check: favorable probe re-enables checks.
		favorable := obs.IsProbe && obs.Hit && DedupCheckWorthwhile(obs.CheckOverhead, obs.AvoidedTransfer, c.cfg.MinTransferSavingsRatio)
		if favorable {
			c.enabled = true
			c.consecutiveMisses = 0
			c.skipCountSinceLastProbe = 0
			c.lastReason = ReasonSavingsFavorable
			if c.cumulativeSavings < 0 {
				c.cumulativeSavings = 0 // forgive past deficit on witnessed recovery
			}
		}
	} else {
		// Ordinary checks enabled: check if losing streak or negative cumulative savings disables.
		if c.consecutiveMisses >= c.cfg.MaxLosingStreak {
			c.enabled = false
			c.lastReason = ReasonLosingStreak
		} else if c.totalChecks >= int64(c.cfg.MinChecksForSavings) && c.cumulativeSavings < 0 {
			c.enabled = false
			c.lastReason = ReasonNegativeSavings
		}
	}

	return netSavings, c.enabled
}

// IsEnabled reports whether ordinary dedup checks are currently enabled.
func (c *DedupCheckController) IsEnabled() bool {
	if c == nil {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.enabled
}

// CumulativeSavings returns the signed net time saved across all observed checks.
func (c *DedupCheckController) CumulativeSavings() time.Duration {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.cumulativeSavings
}

// ConsecutiveMisses returns the current number of consecutive check misses.
func (c *DedupCheckController) ConsecutiveMisses() int {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.consecutiveMisses
}

// Stats returns a telemetry snapshot of the controller.
func (c *DedupCheckController) Stats() DedupControllerStats {
	if c == nil {
		return DedupControllerStats{}
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	var hitRate float64
	if c.totalChecks > 0 {
		hitRate = float64(c.totalHits) / float64(c.totalChecks)
	}

	return DedupControllerStats{
		Enabled:                   c.enabled,
		ConsecutiveMisses:         c.consecutiveMisses,
		TotalChecks:               c.totalChecks,
		TotalHits:                 c.totalHits,
		TotalMisses:               c.totalMisses,
		TotalProbes:               c.totalProbes,
		ProbeHits:                 c.probeHits,
		CumulativeOverhead:        c.cumulativeOverhead,
		CumulativeAvoidedTransfer: c.cumulativeAvoidedTransfer,
		CumulativeSavings:         c.cumulativeSavings,
		HitRate:                   hitRate,
	}
}

// Reset clears controller statistics and resets state according to configuration.
func (c *DedupCheckController) Reset() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	c.enabled = !c.cfg.StartDisabled
	c.consecutiveMisses = 0
	c.skipCountSinceLastProbe = 0
	c.totalChecks = 0
	c.totalHits = 0
	c.totalMisses = 0
	c.totalProbes = 0
	c.probeHits = 0
	c.cumulativeOverhead = 0
	c.cumulativeAvoidedTransfer = 0
	c.cumulativeSavings = 0
	if c.enabled {
		c.lastReason = ReasonSavingsFavorable
	} else {
		c.lastReason = ReasonDisabled
	}
}

// DedupNetSavings returns the SIGNED net-true time saved by a dedup check.
// If hit is true, avoidedTransfer was saved at the cost of checkOverhead:
//
//	netSavings = avoidedTransfer - checkOverhead
//
// If hit is false, checkOverhead was spent without avoiding any transfer:
//
//	netSavings = -checkOverhead
//
// Inputs clamp defensively to non-negative durations.
func DedupNetSavings(hit bool, checkOverhead, avoidedTransfer time.Duration) time.Duration {
	if checkOverhead < 0 {
		checkOverhead = 0
	}
	if avoidedTransfer < 0 {
		avoidedTransfer = 0
	}
	if !hit {
		return -checkOverhead
	}
	return avoidedTransfer - checkOverhead
}

// DedupCheckWorthwhile reports whether an avoided transfer duration strictly justifies
// the check overhead on the same net-true time basis:
//
//	avoidedTransfer >= minRatio * checkOverhead
//
// A non-positive minRatio defaults to 1.0 (strict break-even). Non-positive checkOverhead
// or avoidedTransfer returns false.
func DedupCheckWorthwhile(checkOverhead, avoidedTransfer time.Duration, minRatio float64) bool {
	if checkOverhead <= 0 || avoidedTransfer <= 0 {
		return false
	}
	if minRatio <= 0 {
		minRatio = 1.0
	}
	return float64(avoidedTransfer) >= float64(checkOverhead)*minRatio
}

// TransferDuration calculates the expected wire transfer duration for a payload
// of payloadBytes given a network bandwidth of bytesPerSec. Returns 0 if either
// argument is non-positive.
func TransferDuration(payloadBytes int64, bytesPerSec float64) time.Duration {
	if payloadBytes <= 0 || bytesPerSec <= 0 {
		return 0
	}
	sec := float64(payloadBytes) / bytesPerSec
	return time.Duration(sec * float64(time.Second))
}
