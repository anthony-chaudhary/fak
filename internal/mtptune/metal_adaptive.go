package mtptune

import (
	"sync"
)

// TuningAction indicates the adjustment action taken on draft depth.
type TuningAction string

const (
	ActionHold      TuningAction = "hold"
	ActionScaleUp   TuningAction = "scale_up"
	ActionScaleDown TuningAction = "scale_down"
)

const (
	// DefaultAppleSiliconMTPWindowSize is the rolling evaluation window size (32 tokens).
	DefaultAppleSiliconMTPWindowSize = 32

	// DefaultAppleSiliconMTPMinK is the minimum draft depth floor (1).
	DefaultAppleSiliconMTPMinK = 1

	// DefaultAppleSiliconMTPMaxK is the maximum draft depth ceiling on Apple Silicon Metal (4).
	DefaultAppleSiliconMTPMaxK = 4

	// DefaultAppleSiliconMTPUpscaleThreshold is the rolling acceptance threshold to scale depth up (0.82).
	DefaultAppleSiliconMTPUpscaleThreshold = 0.82

	// DefaultAppleSiliconMTPDownscaleThreshold is the rolling acceptance threshold to scale depth down (0.60).
	DefaultAppleSiliconMTPDownscaleThreshold = 0.60

	// DefaultAppleSiliconMTPCooldownTokens is the minimum token observation period between depth adjustments.
	DefaultAppleSiliconMTPCooldownTokens = 32
)

// AppleSiliconMTPTunerConfig configures the dynamic adaptive depth tuner.
type AppleSiliconMTPTunerConfig struct {
	MinK               int     `json:"min_k"`
	MaxK               int     `json:"max_k"`
	InitialK           int     `json:"initial_k"`
	WindowSize         int     `json:"window_size"`
	UpscaleThreshold   float64 `json:"upscale_threshold"`
	DownscaleThreshold float64 `json:"downscale_threshold"`
	CooldownTokens     int     `json:"cooldown_tokens"`
	MinTokensForScale  int     `json:"min_tokens_for_scale"`
}

// DefaultAppleSiliconMTPTunerConfig returns canonical defaults for Apple Silicon Metal MTP depth tuning.
func DefaultAppleSiliconMTPTunerConfig() AppleSiliconMTPTunerConfig {
	return AppleSiliconMTPTunerConfig{
		MinK:               DefaultAppleSiliconMTPMinK,
		MaxK:               DefaultAppleSiliconMTPMaxK,
		InitialK:           DefaultAppleSiliconMTPMinK,
		WindowSize:         DefaultAppleSiliconMTPWindowSize,
		UpscaleThreshold:   DefaultAppleSiliconMTPUpscaleThreshold,
		DownscaleThreshold: DefaultAppleSiliconMTPDownscaleThreshold,
		CooldownTokens:     DefaultAppleSiliconMTPCooldownTokens,
		MinTokensForScale:  DefaultAppleSiliconMTPWindowSize,
	}
}

// StepReport captures the outcome of an observed speculative decode step.
type StepReport struct {
	DepthBefore    int          `json:"depth_before"`
	DepthAfter     int          `json:"depth_after"`
	Proposed       int          `json:"proposed"`
	Accepted       int          `json:"accepted"`
	AcceptanceRate float64      `json:"acceptance_rate"`
	Action         TuningAction `json:"action"`
	InCooldown     bool         `json:"in_cooldown"`
}

// AppleSiliconMTPDepthTuner dynamically auto-tunes speculative draft depth K in [1, 4]
// on Apple Silicon Metal by monitoring rolling acceptance rate over a 32-token window
// with hysteresis to prevent rapid oscillations.
type AppleSiliconMTPDepthTuner struct {
	mu               sync.RWMutex
	cfg              AppleSiliconMTPTunerConfig
	currentK         int
	window           []bool
	windowHead       int
	windowCount      int
	acceptedCount    int
	totalTokens      int
	totalAccepted    int
	totalProposed    int
	tokensSinceScale int
	scaleUpCount     int
	scaleDownCount   int
	lastAction       TuningAction
}

// NewAppleSiliconMTPDepthTuner constructs a new Apple Silicon MTP depth tuner.
func NewAppleSiliconMTPDepthTuner(cfgs ...AppleSiliconMTPTunerConfig) *AppleSiliconMTPDepthTuner {
	cfg := DefaultAppleSiliconMTPTunerConfig()
	if len(cfgs) > 0 {
		userCfg := cfgs[0]
		if userCfg.MinK > 0 {
			cfg.MinK = userCfg.MinK
		}
		if userCfg.MaxK >= cfg.MinK {
			cfg.MaxK = userCfg.MaxK
		}
		if userCfg.InitialK >= cfg.MinK && userCfg.InitialK <= cfg.MaxK {
			cfg.InitialK = userCfg.InitialK
		}
		if userCfg.WindowSize > 0 {
			cfg.WindowSize = userCfg.WindowSize
		}
		if userCfg.UpscaleThreshold > 0 {
			cfg.UpscaleThreshold = userCfg.UpscaleThreshold
		}
		if userCfg.DownscaleThreshold > 0 {
			cfg.DownscaleThreshold = userCfg.DownscaleThreshold
		}
		if userCfg.CooldownTokens > 0 {
			cfg.CooldownTokens = userCfg.CooldownTokens
		}
		if userCfg.MinTokensForScale > 0 {
			cfg.MinTokensForScale = userCfg.MinTokensForScale
		}
	}

	return &AppleSiliconMTPDepthTuner{
		cfg:              cfg,
		currentK:         cfg.InitialK,
		window:           make([]bool, cfg.WindowSize),
		tokensSinceScale: cfg.CooldownTokens,
		lastAction:       ActionHold,
	}
}

// NewAppleSiliconMTPDepthTunerWithDepth constructs a tuner with a specific initial depth K.
func NewAppleSiliconMTPDepthTunerWithDepth(initialK int) *AppleSiliconMTPDepthTuner {
	cfg := DefaultAppleSiliconMTPTunerConfig()
	if initialK >= cfg.MinK && initialK <= cfg.MaxK {
		cfg.InitialK = initialK
	}
	return NewAppleSiliconMTPDepthTuner(cfg)
}

func (t *AppleSiliconMTPDepthTuner) ensureInitLocked() {
	if t.window == nil {
		cfg := t.cfg
		defaultCfg := DefaultAppleSiliconMTPTunerConfig()
		if cfg.MinK <= 0 {
			cfg.MinK = defaultCfg.MinK
		}
		if cfg.MaxK < cfg.MinK {
			cfg.MaxK = defaultCfg.MaxK
		}
		if cfg.InitialK < cfg.MinK || cfg.InitialK > cfg.MaxK {
			cfg.InitialK = cfg.MinK
		}
		if cfg.WindowSize <= 0 {
			cfg.WindowSize = defaultCfg.WindowSize
		}
		if cfg.UpscaleThreshold <= 0 {
			cfg.UpscaleThreshold = defaultCfg.UpscaleThreshold
		}
		if cfg.DownscaleThreshold <= 0 {
			cfg.DownscaleThreshold = defaultCfg.DownscaleThreshold
		}
		if cfg.CooldownTokens <= 0 {
			cfg.CooldownTokens = defaultCfg.CooldownTokens
		}
		if cfg.MinTokensForScale <= 0 {
			cfg.MinTokensForScale = defaultCfg.MinTokensForScale
		}
		t.cfg = cfg
		if t.currentK < cfg.MinK || t.currentK > cfg.MaxK {
			t.currentK = cfg.InitialK
		}
		t.window = make([]bool, cfg.WindowSize)
		t.tokensSinceScale = cfg.CooldownTokens
		t.lastAction = ActionHold
	}
}

// CurrentK returns the currently configured draft depth K in [MinK, MaxK].
func (t *AppleSiliconMTPDepthTuner) CurrentK() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.window == nil {
		if t.cfg.InitialK >= DefaultAppleSiliconMTPMinK && t.cfg.InitialK <= DefaultAppleSiliconMTPMaxK {
			return t.cfg.InitialK
		}
		return DefaultAppleSiliconMTPMinK
	}
	return t.currentK
}

// K is an alias for CurrentK.
func (t *AppleSiliconMTPDepthTuner) K() int {
	return t.CurrentK()
}

// RollingAcceptanceRate returns the rolling acceptance rate over the current window.
// Returns 1.0 when no tokens have been recorded yet.
func (t *AppleSiliconMTPDepthTuner) RollingAcceptanceRate() float64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.windowCount == 0 {
		return 1.0
	}
	return float64(t.acceptedCount) / float64(t.windowCount)
}

// AcceptanceRate is an alias for RollingAcceptanceRate.
func (t *AppleSiliconMTPDepthTuner) AcceptanceRate() float64 {
	return t.RollingAcceptanceRate()
}

// RecordStep records proposed and accepted token counts for a speculative step,
// updates the rolling 32-token window, applies hysteresis, adjusts K, and returns current K.
func (t *AppleSiliconMTPDepthTuner) RecordStep(proposed, accepted int) int {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.ensureInitLocked()

	if proposed <= 0 {
		return t.currentK
	}
	if accepted > proposed {
		accepted = proposed
	}
	if accepted < 0 {
		accepted = 0
	}

	t.totalProposed += proposed
	t.totalAccepted += accepted

	for i := 0; i < proposed; i++ {
		isAcc := i < accepted
		t.recordSingleTokenLocked(isAcc)
	}

	t.evaluateScalingLocked()

	return t.currentK
}

// ObserveStep processes an execution step and returns a comprehensive StepReport.
func (t *AppleSiliconMTPDepthTuner) ObserveStep(proposed, accepted int) StepReport {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.ensureInitLocked()

	depthBefore := t.currentK

	if proposed <= 0 {
		rate := 1.0
		if t.windowCount > 0 {
			rate = float64(t.acceptedCount) / float64(t.windowCount)
		}
		return StepReport{
			DepthBefore:    depthBefore,
			DepthAfter:     depthBefore,
			Proposed:       0,
			Accepted:       0,
			AcceptanceRate: rate,
			Action:         ActionHold,
			InCooldown:     t.tokensSinceScale < t.cfg.CooldownTokens,
		}
	}

	if accepted > proposed {
		accepted = proposed
	}
	if accepted < 0 {
		accepted = 0
	}

	t.totalProposed += proposed
	t.totalAccepted += accepted

	for i := 0; i < proposed; i++ {
		isAcc := i < accepted
		t.recordSingleTokenLocked(isAcc)
	}

	t.evaluateScalingLocked()

	rate := 1.0
	if t.windowCount > 0 {
		rate = float64(t.acceptedCount) / float64(t.windowCount)
	}

	return StepReport{
		DepthBefore:    depthBefore,
		DepthAfter:     t.currentK,
		Proposed:       proposed,
		Accepted:       accepted,
		AcceptanceRate: rate,
		Action:         t.lastAction,
		InCooldown:     t.tokensSinceScale < t.cfg.CooldownTokens,
	}
}

func (t *AppleSiliconMTPDepthTuner) recordSingleTokenLocked(isAccepted bool) {
	t.totalTokens++
	t.tokensSinceScale++

	if t.windowCount < t.cfg.WindowSize {
		t.window[t.windowCount] = isAccepted
		t.windowCount++
		if isAccepted {
			t.acceptedCount++
		}
	} else {
		old := t.window[t.windowHead]
		if old {
			t.acceptedCount--
		}
		t.window[t.windowHead] = isAccepted
		if isAccepted {
			t.acceptedCount++
		}
		t.windowHead = (t.windowHead + 1) % t.cfg.WindowSize
	}
}

func (t *AppleSiliconMTPDepthTuner) evaluateScalingLocked() {
	if t.windowCount < t.cfg.MinTokensForScale || t.tokensSinceScale < t.cfg.CooldownTokens {
		t.lastAction = ActionHold
		return
	}

	rate := float64(t.acceptedCount) / float64(t.windowCount)
	if rate >= t.cfg.UpscaleThreshold && t.currentK < t.cfg.MaxK {
		t.currentK++
		t.tokensSinceScale = 0
		t.scaleUpCount++
		t.lastAction = ActionScaleUp
	} else if rate <= t.cfg.DownscaleThreshold && t.currentK > t.cfg.MinK {
		t.currentK--
		t.tokensSinceScale = 0
		t.scaleDownCount++
		t.lastAction = ActionScaleDown
	} else {
		t.lastAction = ActionHold
	}
}

// Record is an alias for RecordStep.
func (t *AppleSiliconMTPDepthTuner) Record(proposed, accepted int) int {
	return t.RecordStep(proposed, accepted)
}

// RecordTokens records accepted tokens out of a total count.
func (t *AppleSiliconMTPDepthTuner) RecordTokens(accepted, total int) int {
	return t.RecordStep(total, accepted)
}

// RecordToken records a single token outcome (accepted or rejected).
func (t *AppleSiliconMTPDepthTuner) RecordToken(accepted bool) int {
	if accepted {
		return t.RecordStep(1, 1)
	}
	return t.RecordStep(1, 0)
}

// RecordOutcome is an alias for RecordToken.
func (t *AppleSiliconMTPDepthTuner) RecordOutcome(accepted bool) int {
	return t.RecordToken(accepted)
}

// SetK manually sets the current draft depth K, clamped to [MinK, MaxK].
func (t *AppleSiliconMTPDepthTuner) SetK(k int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.ensureInitLocked()
	if k < t.cfg.MinK {
		k = t.cfg.MinK
	}
	if k > t.cfg.MaxK {
		k = t.cfg.MaxK
	}
	t.currentK = k
	t.tokensSinceScale = 0
}

// SetDepth is an alias for SetK.
func (t *AppleSiliconMTPDepthTuner) SetDepth(k int) {
	t.SetK(k)
}

// Reset clears the rolling window and resets depth to InitialK.
func (t *AppleSiliconMTPDepthTuner) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.ensureInitLocked()
	t.currentK = t.cfg.InitialK
	t.windowHead = 0
	t.windowCount = 0
	t.acceptedCount = 0
	t.totalTokens = 0
	t.totalAccepted = 0
	t.totalProposed = 0
	t.tokensSinceScale = t.cfg.CooldownTokens
	t.scaleUpCount = 0
	t.scaleDownCount = 0
	t.lastAction = ActionHold
	for i := range t.window {
		t.window[i] = false
	}
}

// TotalTokens returns the total number of proposed tokens observed across the tuner lifetime.
func (t *AppleSiliconMTPDepthTuner) TotalTokens() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.totalTokens
}

// TotalAccepted returns the total number of accepted tokens observed across the tuner lifetime.
func (t *AppleSiliconMTPDepthTuner) TotalAccepted() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.totalAccepted
}

// TotalProposed returns the total number of proposed tokens observed across the tuner lifetime.
func (t *AppleSiliconMTPDepthTuner) TotalProposed() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.totalProposed
}

// WindowCount returns the number of tokens currently in the rolling window (0..WindowSize).
func (t *AppleSiliconMTPDepthTuner) WindowCount() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.windowCount
}

// AcceptedInWindow returns the number of accepted tokens in the rolling window.
func (t *AppleSiliconMTPDepthTuner) AcceptedInWindow() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.acceptedCount
}

// TokensSinceScale returns the number of tokens observed since the last depth adjustment.
func (t *AppleSiliconMTPDepthTuner) TokensSinceScale() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.tokensSinceScale
}

// InCooldown returns true if the tuner is within the hysteresis cooldown window.
func (t *AppleSiliconMTPDepthTuner) InCooldown() bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.tokensSinceScale < t.cfg.CooldownTokens
}

// ScaleUpCount returns the total number of depth upscaling events.
func (t *AppleSiliconMTPDepthTuner) ScaleUpCount() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.scaleUpCount
}

// ScaleDownCount returns the total number of depth downscaling events.
func (t *AppleSiliconMTPDepthTuner) ScaleDownCount() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.scaleDownCount
}

// LastAction returns the most recent tuning action.
func (t *AppleSiliconMTPDepthTuner) LastAction() TuningAction {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.lastAction
}

// EffectiveThroughput calculates the modeled decode throughput (tok/s) on Apple Silicon Metal
// based on current draft depth K and rolling acceptance rate.
func (t *AppleSiliconMTPDepthTuner) EffectiveThroughput() float64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	rate := 1.0
	if t.windowCount > 0 {
		rate = float64(t.acceptedCount) / float64(t.windowCount)
	}
	return EstimateThroughput(t.currentK, rate, AppleSiliconMetalSweepConfig())
}

// Config returns a copy of the tuner configuration.
func (t *AppleSiliconMTPDepthTuner) Config() AppleSiliconMTPTunerConfig {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.cfg
}
