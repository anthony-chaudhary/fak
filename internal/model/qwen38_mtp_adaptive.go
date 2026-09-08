package model

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

const (
	// Qwen38AdaptiveDepthReceiptSchema identifies the schema version for adaptive depth receipts.
	Qwen38AdaptiveDepthReceiptSchema = "fak/qwen38-mtp-adaptive-receipt/v1"
)

// Qwen38AdaptiveConfig defines configuration bounds and parameters for the depth governor.
type Qwen38AdaptiveConfig struct {
	MaxDepth             int     `json:"max_depth"`              // Default 4
	ColdStartDepth       int     `json:"cold_start_depth"`       // Default 2
	AlphaEMA             float64 `json:"alpha_ema"`              // Default 0.1
	HysteresisSteps      int     `json:"hysteresis_steps"`       // Default 3
	ProbeInterval        int     `json:"probe_interval"`         // Default 10 (target-only steps before probing depth 1)
	SpeedupThresholdHigh float64 `json:"speedup_threshold_high"` // Default 1.20
	SpeedupThresholdLow  float64 `json:"speedup_threshold_low"`  // Default 1.05
	TargetOnlyThreshold  float64 `json:"target_only_threshold"`  // Default 1.00 (below this is net-negative)
	MinAcceptanceRate    float64 `json:"min_acceptance_rate"`    // Default 0.40
}

// DefaultQwen38AdaptiveConfig returns standard production defaults for Qwen3.8 MTP depth governance.
func DefaultQwen38AdaptiveConfig() Qwen38AdaptiveConfig {
	return Qwen38AdaptiveConfig{
		MaxDepth:             4,
		ColdStartDepth:       2,
		AlphaEMA:             0.1,
		HysteresisSteps:      3,
		ProbeInterval:        10,
		SpeedupThresholdHigh: 1.20,
		SpeedupThresholdLow:  1.05,
		TargetOnlyThreshold:  1.00,
		MinAcceptanceRate:    0.40,
	}
}

// Validate checks whether the config satisfies all operational invariants.
func (cfg Qwen38AdaptiveConfig) Validate() error {
	if cfg.MaxDepth < 1 {
		return errors.New("model: max_depth must be at least 1")
	}
	if cfg.ColdStartDepth < 0 || cfg.ColdStartDepth > cfg.MaxDepth {
		return fmt.Errorf("model: cold_start_depth %d outside [0, %d]", cfg.ColdStartDepth, cfg.MaxDepth)
	}
	if cfg.AlphaEMA <= 0 || cfg.AlphaEMA > 1.0 {
		return fmt.Errorf("model: alpha_ema %g outside (0, 1]", cfg.AlphaEMA)
	}
	if cfg.HysteresisSteps < 1 {
		return errors.New("model: hysteresis_steps must be at least 1")
	}
	if cfg.ProbeInterval < 1 {
		return errors.New("model: probe_interval must be at least 1")
	}
	if cfg.TargetOnlyThreshold <= 0 {
		return errors.New("model: target_only_threshold must be positive")
	}
	if cfg.SpeedupThresholdLow < cfg.TargetOnlyThreshold {
		return fmt.Errorf("model: speedup_threshold_low %g cannot be below target_only_threshold %g",
			cfg.SpeedupThresholdLow, cfg.TargetOnlyThreshold)
	}
	if cfg.SpeedupThresholdHigh <= cfg.SpeedupThresholdLow {
		return fmt.Errorf("model: speedup_threshold_high %g must exceed speedup_threshold_low %g",
			cfg.SpeedupThresholdHigh, cfg.SpeedupThresholdLow)
	}
	return nil
}

// Qwen38AdaptiveStepObservation records the measured performance of one decode step.
type Qwen38AdaptiveStepObservation struct {
	ProposedTokens int           `json:"proposed_tokens"`
	AcceptedTokens int           `json:"accepted_tokens"`
	StepLatency    time.Duration `json:"step_latency"`
	TargetLatency  time.Duration `json:"target_latency"`
	Speedup        float64       `json:"speedup,omitempty"`
}

// ComputeSpeedup computes the effective speedup relative to target-only decode.
func (obs Qwen38AdaptiveStepObservation) ComputeSpeedup() float64 {
	if obs.Speedup > 0 {
		return obs.Speedup
	}
	tokensProduced := obs.AcceptedTokens + 1
	if obs.ProposedTokens == 0 {
		tokensProduced = 1
	}
	if obs.TargetLatency > 0 && obs.StepLatency > 0 {
		costPerToken := float64(obs.StepLatency.Nanoseconds()) / float64(tokensProduced)
		return float64(obs.TargetLatency.Nanoseconds()) / costPerToken
	}
	// Fallback analytical model if latency not provided:
	// speedup = (accepted + 1) / (1 + 0.25 * proposed)
	costFactor := 1.0 + 0.25*float64(obs.ProposedTokens)
	return float64(tokensProduced) / costFactor
}

// Qwen38AdaptiveTraceRecord records an audited, deterministic step decision.
// The NetNegative field explicitly prevents hiding negative net performance.
type Qwen38AdaptiveTraceRecord struct {
	Step               int           `json:"step"`
	DepthBefore        int           `json:"depth_before"`
	DepthAfter         int           `json:"depth_after"`
	ProposedTokens     int           `json:"proposed_tokens"`
	AcceptedTokens     int           `json:"accepted_tokens"`
	AcceptanceRate     float64       `json:"acceptance_rate"`
	EMA_AcceptanceRate float64       `json:"ema_acceptance_rate"`
	StepLatency        time.Duration `json:"step_latency"`
	TargetLatency      time.Duration `json:"target_latency"`
	Speedup            float64       `json:"speedup"`
	EMA_Speedup        float64       `json:"ema_speedup"`
	Action             string        `json:"action"`
	Reason             string        `json:"reason"`
	NetNegative        bool          `json:"net_negative"`
}

// Qwen38AdaptiveDepthReceipt witnesses the adaptive governor's lifetime or session performance.
type Qwen38AdaptiveDepthReceipt struct {
	SchemaVersion      string                      `json:"schema_version"`
	Engine             Qwen38MTPEngine             `json:"engine"`
	MaxDepth           int                         `json:"max_depth"`
	InitialDepth       int                         `json:"initial_depth"`
	FinalDepth         int                         `json:"final_depth"`
	TotalSteps         int                         `json:"total_steps"`
	TargetOnlyEscapes  int                         `json:"target_only_escapes"`
	ProbesExecuted     int                         `json:"probes_executed"`
	NetNegativeSteps   int                         `json:"net_negative_steps"`
	MeanSpeedup        float64                     `json:"mean_speedup"`
	MeanAcceptanceRate float64                     `json:"mean_acceptance_rate"`
	HysteresisSteps    int                         `json:"hysteresis_steps"`
	ProbeInterval      int                         `json:"probe_interval"`
	Trace              []Qwen38AdaptiveTraceRecord `json:"trace,omitempty"`
}

// Validate ensures the receipt invariants hold.
func (r Qwen38AdaptiveDepthReceipt) Validate() error {
	if r.SchemaVersion != Qwen38AdaptiveDepthReceiptSchema {
		return fmt.Errorf("model: adaptive receipt schema %q, want %q", r.SchemaVersion, Qwen38AdaptiveDepthReceiptSchema)
	}
	if r.Engine != Qwen38EngineMTP {
		return fmt.Errorf("model: adaptive receipt engine %q is not fak-native MTP", r.Engine)
	}
	if r.FinalDepth < 0 || r.FinalDepth > r.MaxDepth {
		return fmt.Errorf("model: impossible final depth %d for max %d", r.FinalDepth, r.MaxDepth)
	}
	if r.TotalSteps < 0 || r.TargetOnlyEscapes < 0 || r.NetNegativeSteps < 0 {
		return fmt.Errorf("model: negative step counters")
	}
	return nil
}

// Qwen38MTPAdaptiveDepthGovernor dynamically controls speculative draft depth K in [0, MaxDepth]
// based on exponential moving averages of acceptance rate and net speedup.
// Features target-only escape when speedup < 1.0, probe re-evaluation, and full deterministic audit traces.
type Qwen38MTPAdaptiveDepthGovernor struct {
	mu                sync.Mutex
	config            Qwen38AdaptiveConfig
	currentDepth      int
	initialized       bool
	emaAcceptanceRate float64
	emaSpeedup        float64
	emaStepLatency    time.Duration
	highStreak        int
	lowStreak         int
	stepsInTargetOnly int
	isProbing         bool
	stepCount         int
	targetOnlyEscapes int
	probesExecuted    int
	netNegativeSteps  int
	sumSpeedup        float64
	sumAcceptanceRate float64
	trace             []Qwen38AdaptiveTraceRecord
}

// NewQwen38MTPAdaptiveDepthGovernor creates and initializes an adaptive depth governor.
func NewQwen38MTPAdaptiveDepthGovernor(cfg Qwen38AdaptiveConfig) (*Qwen38MTPAdaptiveDepthGovernor, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Qwen38MTPAdaptiveDepthGovernor{
		config:       cfg,
		currentDepth: cfg.ColdStartDepth,
		trace:        make([]Qwen38AdaptiveTraceRecord, 0),
	}, nil
}

// Config returns the governor's configuration.
func (g *Qwen38MTPAdaptiveDepthGovernor) Config() Qwen38AdaptiveConfig {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.config
}

// CurrentDepth returns the currently adjudicated draft depth K in [0, MaxDepth].
func (g *Qwen38MTPAdaptiveDepthGovernor) CurrentDepth() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.currentDepth
}

// DowngradeReason returns Qwen38MTPNetLatencyRegressed when depth is 0 (target-only escape),
// or Qwen38MTPEligible when drafting is active.
func (g *Qwen38MTPAdaptiveDepthGovernor) DowngradeReason() Qwen38MTPDowngradeReason {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.currentDepth == 0 {
		return Qwen38MTPNetLatencyRegressed
	}
	return Qwen38MTPEligible
}

// ObserveStep processes an execution step observation, updates EMAs, evaluates target-only escape
// and hysteresis transitions, and returns the next draft depth along with the deterministic trace record.
func (g *Qwen38MTPAdaptiveDepthGovernor) ObserveStep(obs Qwen38AdaptiveStepObservation) (int, Qwen38AdaptiveTraceRecord, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.stepCount++
	depthBefore := g.currentDepth

	// 1. Calculate step metrics
	speedup := obs.ComputeSpeedup()
	acceptanceRate := 0.0
	if obs.ProposedTokens > 0 {
		acceptanceRate = float64(obs.AcceptedTokens) / float64(obs.ProposedTokens)
	}

	g.sumSpeedup += speedup
	g.sumAcceptanceRate += acceptanceRate

	// 2. Update EMAs
	if !g.initialized {
		g.emaAcceptanceRate = acceptanceRate
		g.emaSpeedup = speedup
		g.emaStepLatency = obs.StepLatency
		g.initialized = true
	} else {
		alpha := g.config.AlphaEMA
		g.emaAcceptanceRate = alpha*acceptanceRate + (1.0-alpha)*g.emaAcceptanceRate
		g.emaSpeedup = alpha*speedup + (1.0-alpha)*g.emaSpeedup
		g.emaStepLatency = time.Duration(alpha*float64(obs.StepLatency) + (1.0-alpha)*float64(g.emaStepLatency))
	}

	netNegative := speedup < g.config.TargetOnlyThreshold
	if netNegative {
		g.netNegativeSteps++
	}

	action := "hold"
	reason := "normal"

	// 3. State-dependent adjudication
	if depthBefore == 0 {
		// In target-only escape state
		if !g.isProbing {
			g.stepsInTargetOnly++
			if g.stepsInTargetOnly >= g.config.ProbeInterval {
				// Probe period elapsed: probe depth 1 to test if drafting conditions have improved
				g.currentDepth = 1
				g.isProbing = true
				g.probesExecuted++
				action = "probe"
				reason = "probe_interval_elapsed_probing_depth_1"
			} else {
				action = "hold"
				reason = fmt.Sprintf("target_only_hold_step_%d_of_%d", g.stepsInTargetOnly, g.config.ProbeInterval)
			}
		} else {
			// We just observed a probe step at depth 1
			g.isProbing = false
			if speedup >= g.config.TargetOnlyThreshold {
				// Probe succeeded: drafting is net-positive again!
				g.currentDepth = 1
				g.stepsInTargetOnly = 0
				action = "resume"
				reason = "probe_succeeded_speedup_positive"
			} else {
				// Probe failed: still net-negative
				g.currentDepth = 0
				g.stepsInTargetOnly = 0
				g.targetOnlyEscapes++
				action = "escape"
				reason = "probe_failed_net_negative_escape_to_target_only"
			}
		}
	} else {
		// In active speculative drafting state (depthBefore > 0)
		if g.isProbing {
			// Handled probe step
			g.isProbing = false
			if speedup < g.config.TargetOnlyThreshold {
				g.currentDepth = 0
				g.targetOnlyEscapes++
				g.stepsInTargetOnly = 0
				action = "escape"
				reason = "probe_failed_net_negative_escape_to_target_only"
			} else {
				g.currentDepth = 1
				g.stepsInTargetOnly = 0
				action = "resume"
				reason = "probe_succeeded_speedup_positive"
			}
		} else if speedup < g.config.TargetOnlyThreshold {
			// Target-only escape: drafting is net-negative!
			g.currentDepth = 0
			g.targetOnlyEscapes++
			g.stepsInTargetOnly = 0
			g.highStreak = 0
			g.lowStreak = 0
			action = "escape"
			reason = "target_only_escape_speedup_subunitary"
		} else {
			// Check if already at MaxDepth
			if g.currentDepth >= g.config.MaxDepth && speedup >= g.config.SpeedupThresholdLow {
				g.highStreak = 0
				g.lowStreak = 0
				action = "hold"
				reason = "at_max_depth"
			} else if speedup >= g.config.SpeedupThresholdHigh && g.emaSpeedup >= g.config.SpeedupThresholdHigh && g.emaAcceptanceRate >= g.config.MinAcceptanceRate {
				g.highStreak++
				g.lowStreak = 0
				if g.highStreak >= g.config.HysteresisSteps {
					if g.currentDepth < g.config.MaxDepth {
						g.currentDepth++
						g.highStreak = 0
						action = "ramp_up"
						reason = fmt.Sprintf("high_speedup_ema_%0.2f_ramp_depth_%d", g.emaSpeedup, g.currentDepth)
					} else {
						action = "hold"
						reason = "at_max_depth"
					}
				} else {
					action = "hold"
					reason = fmt.Sprintf("hysteresis_ramp_pending_%d_of_%d", g.highStreak, g.config.HysteresisSteps)
				}
			} else if speedup <= g.config.SpeedupThresholdLow || g.emaSpeedup <= g.config.SpeedupThresholdLow || g.emaAcceptanceRate < g.config.MinAcceptanceRate {
				g.lowStreak++
				g.highStreak = 0
				if g.lowStreak >= g.config.HysteresisSteps {
					if g.currentDepth > 1 {
						g.currentDepth--
						g.lowStreak = 0
						action = "drop"
						reason = fmt.Sprintf("low_speedup_ema_%0.2f_drop_depth_%d", g.emaSpeedup, g.currentDepth)
					} else if g.currentDepth == 1 {
						// At depth 1, if performance is poor, escape to target-only
						if speedup < g.config.TargetOnlyThreshold {
							g.currentDepth = 0
							g.targetOnlyEscapes++
							g.stepsInTargetOnly = 0
							action = "escape"
							reason = "target_only_escape_depth1_regressed"
						} else {
							action = "hold"
							reason = "at_min_active_depth"
						}
					}
				} else {
					action = "hold"
					reason = fmt.Sprintf("hysteresis_drop_pending_%d_of_%d", g.lowStreak, g.config.HysteresisSteps)
				}
			} else {
				// In deadband
				g.highStreak = 0
				g.lowStreak = 0
				action = "hold"
				reason = "deadband_stable"
			}
		}
	}

	rec := Qwen38AdaptiveTraceRecord{
		Step:               g.stepCount,
		DepthBefore:        depthBefore,
		DepthAfter:         g.currentDepth,
		ProposedTokens:     obs.ProposedTokens,
		AcceptedTokens:     obs.AcceptedTokens,
		AcceptanceRate:     acceptanceRate,
		EMA_AcceptanceRate: g.emaAcceptanceRate,
		StepLatency:        obs.StepLatency,
		TargetLatency:      obs.TargetLatency,
		Speedup:            speedup,
		EMA_Speedup:        g.emaSpeedup,
		Action:             action,
		Reason:             reason,
		NetNegative:        netNegative,
	}

	g.trace = append(g.trace, rec)
	return g.currentDepth, rec, nil
}

// Trace returns an immutable copy of all recorded step transitions.
func (g *Qwen38MTPAdaptiveDepthGovernor) Trace() []Qwen38AdaptiveTraceRecord {
	g.mu.Lock()
	defer g.mu.Unlock()
	copied := make([]Qwen38AdaptiveTraceRecord, len(g.trace))
	copy(copied, g.trace)
	return copied
}

// Receipt generates a verified receipt summarizing governor decisions and performance.
func (g *Qwen38MTPAdaptiveDepthGovernor) Receipt() Qwen38AdaptiveDepthReceipt {
	g.mu.Lock()
	defer g.mu.Unlock()

	meanSpeedup := 0.0
	meanAcceptance := 0.0
	if g.stepCount > 0 {
		meanSpeedup = g.sumSpeedup / float64(g.stepCount)
		meanAcceptance = g.sumAcceptanceRate / float64(g.stepCount)
	}

	copiedTrace := make([]Qwen38AdaptiveTraceRecord, len(g.trace))
	copy(copiedTrace, g.trace)

	return Qwen38AdaptiveDepthReceipt{
		SchemaVersion:      Qwen38AdaptiveDepthReceiptSchema,
		Engine:             Qwen38EngineMTP,
		MaxDepth:           g.config.MaxDepth,
		InitialDepth:       g.config.ColdStartDepth,
		FinalDepth:         g.currentDepth,
		TotalSteps:         g.stepCount,
		TargetOnlyEscapes:  g.targetOnlyEscapes,
		ProbesExecuted:     g.probesExecuted,
		NetNegativeSteps:   g.netNegativeSteps,
		MeanSpeedup:        meanSpeedup,
		MeanAcceptanceRate: meanAcceptance,
		HysteresisSteps:    g.config.HysteresisSteps,
		ProbeInterval:      g.config.ProbeInterval,
		Trace:              copiedTrace,
	}
}

// Reset clears recorded steps and restarts from cold start depth.
func (g *Qwen38MTPAdaptiveDepthGovernor) Reset() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.currentDepth = g.config.ColdStartDepth
	g.initialized = false
	g.emaAcceptanceRate = 0
	g.emaSpeedup = 0
	g.emaStepLatency = 0
	g.highStreak = 0
	g.lowStreak = 0
	g.stepsInTargetOnly = 0
	g.isProbing = false
	g.stepCount = 0
	g.targetOnlyEscapes = 0
	g.probesExecuted = 0
	g.netNegativeSteps = 0
	g.sumSpeedup = 0
	g.sumAcceptanceRate = 0
	g.trace = make([]Qwen38AdaptiveTraceRecord, 0)
}

// SimulateSteps runs a sequence of observations and returns the final receipt.
func (g *Qwen38MTPAdaptiveDepthGovernor) SimulateSteps(observations []Qwen38AdaptiveStepObservation) (Qwen38AdaptiveDepthReceipt, error) {
	for _, obs := range observations {
		if _, _, err := g.ObserveStep(obs); err != nil {
			return Qwen38AdaptiveDepthReceipt{}, err
		}
	}
	return g.Receipt(), nil
}
