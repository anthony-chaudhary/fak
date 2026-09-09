package mtptune

import (
	"errors"
	"fmt"
	"math"
	"sync"
)

// DepthThreshold defines the hysteresis bounds for scaling draft depth K.
type DepthThreshold struct {
	Upscale   float64 `json:"upscale"`   // Rolling acceptance rate at or above which K scales up
	Downscale float64 `json:"downscale"` // Rolling acceptance rate below which K scales down
}

// DefaultLevelThresholds returns calibrated hysteresis thresholds for K in [1, 4].
// Provides deadbands between upscale and downscale to prevent depth flapping:
// - K=1: upscale at >= 0.70
// - K=2: upscale at >= 0.80, downscale at < 0.55 (deadband: [0.55, 0.80))
// - K=3: upscale at >= 0.85, downscale at < 0.65 (deadband: [0.65, 0.85))
// - K=4: downscale at < 0.75 (deadband: [0.75, 1.00])
func DefaultLevelThresholds() map[int]DepthThreshold {
	return map[int]DepthThreshold{
		1: {Upscale: 0.70, Downscale: 0.00},
		2: {Upscale: 0.80, Downscale: 0.55},
		3: {Upscale: 0.85, Downscale: 0.65},
		4: {Upscale: 1.00, Downscale: 0.75},
	}
}

// AdaptiveDepthConfig configures dynamic draft depth auto-tuning for MTP speculative decoding.
type AdaptiveDepthConfig struct {
	MinK                 int                    `json:"min_k"`                   // Minimum draft depth (default 1)
	MaxK                 int                    `json:"max_k"`                   // Maximum draft depth (default 4)
	InitialK             int                    `json:"initial_k"`               // Starting draft depth (default 1)
	WindowSize           int                    `json:"window_size"`             // Rolling evaluation window size in tokens (default 32)
	MinTokensBeforeScale int                    `json:"min_tokens_before_scale"` // Minimum tokens observed before scaling (default 8)
	UpscaleThreshold     float64                `json:"upscale_threshold"`       // General fallback upscale threshold (default 0.85)
	DownscaleThreshold   float64                `json:"downscale_threshold"`     // General fallback downscale threshold (default 0.60)
	LevelThresholds      map[int]DepthThreshold `json:"level_thresholds"`        // Per-level hysteresis thresholds
	Hardware             SweepConfig            `json:"hardware"`                // Hardware parameters for throughput estimation
}

// DefaultAdaptiveDepthConfig returns standard configuration for dynamic MTP draft depth tuning.
func DefaultAdaptiveDepthConfig() AdaptiveDepthConfig {
	return AdaptiveDepthConfig{
		MinK:                 1,
		MaxK:                 4,
		InitialK:             1,
		WindowSize:           32,
		MinTokensBeforeScale: 8,
		UpscaleThreshold:     0.85,
		DownscaleThreshold:   0.60,
		LevelThresholds:      DefaultLevelThresholds(),
		Hardware:             DefaultSweepConfig(),
	}
}

// AppleSiliconMetalAdaptiveConfig returns production configuration calibrated for Apple Silicon Metal (M3/M4).
func AppleSiliconMetalAdaptiveConfig() AdaptiveDepthConfig {
	cfg := DefaultAdaptiveDepthConfig()
	cfg.Hardware = AppleSiliconMetalSweepConfig()
	return cfg
}

// Validate checks the adaptive depth configuration for consistency.
func (cfg *AdaptiveDepthConfig) Validate() error {
	if cfg.MinK < 1 || cfg.MaxK < cfg.MinK {
		return fmt.Errorf("invalid K range: [%d, %d]", cfg.MinK, cfg.MaxK)
	}
	if cfg.WindowSize <= 0 {
		return errors.New("window size must be positive")
	}
	if cfg.MinTokensBeforeScale < 1 || cfg.MinTokensBeforeScale > cfg.WindowSize {
		return fmt.Errorf("invalid min tokens before scale: %d (window size: %d)", cfg.MinTokensBeforeScale, cfg.WindowSize)
	}
	if cfg.UpscaleThreshold <= cfg.DownscaleThreshold {
		return fmt.Errorf("upscale threshold (%.2f) must be strictly greater than downscale threshold (%.2f)",
			cfg.UpscaleThreshold, cfg.DownscaleThreshold)
	}
	for k, thresh := range cfg.LevelThresholds {
		if k > cfg.MinK && thresh.Downscale >= thresh.Upscale && thresh.Upscale < 1.0 {
			return fmt.Errorf("level %d: downscale threshold (%.2f) must be less than upscale threshold (%.2f)",
				k, thresh.Downscale, thresh.Upscale)
		}
	}
	return nil
}

// AdaptiveTunerStats snapshots runtime metrics of the adaptive depth auto-tuner.
type AdaptiveTunerStats struct {
	CurrentK            int     `json:"current_k"`
	RollingRate         float64 `json:"rolling_acceptance_rate"`
	LifetimeRate        float64 `json:"lifetime_acceptance_rate"`
	WindowTokens        int     `json:"window_tokens"`
	WindowAccepted      int     `json:"window_accepted"`
	TotalProposed       int     `json:"total_proposed"`
	TotalAccepted       int     `json:"total_accepted"`
	TotalRollbacks      int     `json:"total_rollbacks"`
	UpscaleCount        int     `json:"upscale_count"`
	DownscaleCount      int     `json:"downscale_count"`
	EffectiveThroughput float64 `json:"effective_throughput_tps"`
}

// rollingWindow maintains a fixed-capacity ring buffer of token acceptance outcomes.
type rollingWindow struct {
	outcomes []bool
	head     int
	size     int
	accCount int
}

func newRollingWindow(size int) *rollingWindow {
	if size <= 0 {
		size = 32
	}
	return &rollingWindow{
		outcomes: make([]bool, 0, size),
		size:     size,
	}
}

func (w *rollingWindow) push(accepted bool) {
	if len(w.outcomes) < w.size {
		w.outcomes = append(w.outcomes, accepted)
		if accepted {
			w.accCount++
		}
	} else {
		old := w.outcomes[w.head]
		w.outcomes[w.head] = accepted
		if old && !accepted {
			w.accCount--
		} else if !old && accepted {
			w.accCount++
		}
		w.head = (w.head + 1) % w.size
	}
}

func (w *rollingWindow) rate() float64 {
	if len(w.outcomes) == 0 {
		return 1.0
	}
	return float64(w.accCount) / float64(len(w.outcomes))
}

func (w *rollingWindow) count() int {
	return len(w.outcomes)
}

func (w *rollingWindow) acceptedCount() int {
	return w.accCount
}

func (w *rollingWindow) reset() {
	w.outcomes = w.outcomes[:0]
	w.head = 0
	w.accCount = 0
}

// AdaptiveDepthTuner dynamically adjusts MTP speculative draft depth K based on
// rolling acceptance rates, using hysteresis thresholds to maximize effective decode
// throughput while preventing wasted base forward passes when acceptance drops.
type AdaptiveDepthTuner struct {
	mu             sync.RWMutex
	cfg            AdaptiveDepthConfig
	currentK       int
	window         *rollingWindow
	totalProposed  int
	totalAccepted  int
	totalRollbacks int
	upscaleCount   int
	downscaleCount int
}

// NewAdaptiveDepthTuner creates an auto-tuner initialized with the given configuration.
func NewAdaptiveDepthTuner(cfgs ...AdaptiveDepthConfig) *AdaptiveDepthTuner {
	cfg := AppleSiliconMetalAdaptiveConfig()
	if len(cfgs) > 0 {
		cfg = cfgs[0]
	}
	if cfg.MinK < 1 {
		cfg.MinK = 1
	}
	if cfg.MaxK < cfg.MinK {
		cfg.MaxK = 4
	}
	if cfg.InitialK < cfg.MinK || cfg.InitialK > cfg.MaxK {
		cfg.InitialK = cfg.MinK
	}
	if cfg.WindowSize <= 0 {
		cfg.WindowSize = 32
	}
	if cfg.MinTokensBeforeScale <= 0 {
		cfg.MinTokensBeforeScale = 8
	}
	if cfg.LevelThresholds == nil {
		cfg.LevelThresholds = DefaultLevelThresholds()
	}

	return &AdaptiveDepthTuner{
		cfg:      cfg,
		currentK: cfg.InitialK,
		window:   newRollingWindow(cfg.WindowSize),
	}
}

// Config returns a copy of the tuner configuration.
func (t *AdaptiveDepthTuner) Config() AdaptiveDepthConfig {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.cfg
}

// CurrentK returns the current speculative draft depth.
func (t *AdaptiveDepthTuner) CurrentK() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.currentK
}

// RollingAcceptanceRate returns the acceptance rate over the rolling window.
func (t *AdaptiveDepthTuner) RollingAcceptanceRate() float64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.window.rate()
}

// LifetimeAcceptanceRate returns the lifetime acceptance rate across all recorded tokens.
func (t *AdaptiveDepthTuner) LifetimeAcceptanceRate() float64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.totalProposed == 0 {
		return 1.0
	}
	return float64(t.totalAccepted) / float64(t.totalProposed)
}

// RecordStep records the outcome of a speculative verification step:
// accepted tokens out of proposed tokens. Evaluates hysteresis thresholds and updates K.
// Returns the newly updated draft depth K.
func (t *AdaptiveDepthTuner) RecordStep(accepted, proposed int) int {
	t.mu.Lock()
	defer t.mu.Unlock()

	if proposed <= 0 {
		return t.currentK
	}
	if accepted < 0 {
		accepted = 0
	}
	if accepted > proposed {
		accepted = proposed
	}

	t.totalProposed += proposed
	t.totalAccepted += accepted
	if accepted < proposed {
		t.totalRollbacks += (proposed - accepted)
	}

	for i := 0; i < proposed; i++ {
		t.window.push(i < accepted)
	}

	t.evaluateLocked()
	return t.currentK
}

// RecordOutcome records a single token outcome (accepted or rejected) and updates K.
func (t *AdaptiveDepthTuner) RecordOutcome(accepted bool) int {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.totalProposed++
	if accepted {
		t.totalAccepted++
	} else {
		t.totalRollbacks++
	}

	t.window.push(accepted)
	t.evaluateLocked()
	return t.currentK
}

func (t *AdaptiveDepthTuner) evaluateLocked() {
	if t.window.count() < t.cfg.MinTokensBeforeScale {
		return
	}

	rate := t.window.rate()
	k := t.currentK

	var upThresh, downThresh float64
	if thresh, ok := t.cfg.LevelThresholds[k]; ok {
		upThresh = thresh.Upscale
		downThresh = thresh.Downscale
	} else {
		upThresh = t.cfg.UpscaleThreshold
		downThresh = t.cfg.DownscaleThreshold
	}

	// Upscale if rolling rate meets or exceeds the upscale threshold
	if rate >= upThresh && k < t.cfg.MaxK {
		t.currentK++
		t.upscaleCount++
	} else if rate < downThresh && k > t.cfg.MinK {
		// Gracefully downscale if rolling rate drops below the downscale threshold
		t.currentK--
		t.downscaleCount++
	}
}

// EffectiveThroughput calculates the modeled decode throughput (tok/s) on Apple Silicon Metal
// based on current draft depth K and rolling acceptance rate.
func (t *AdaptiveDepthTuner) EffectiveThroughput() float64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return EstimateThroughput(t.currentK, t.window.rate(), t.cfg.Hardware)
}

// Stats returns a snapshot of runtime metrics.
func (t *AdaptiveDepthTuner) Stats() AdaptiveTunerStats {
	t.mu.RLock()
	defer t.mu.RUnlock()

	lifetime := 1.0
	if t.totalProposed > 0 {
		lifetime = float64(t.totalAccepted) / float64(t.totalProposed)
	}

	return AdaptiveTunerStats{
		CurrentK:            t.currentK,
		RollingRate:         t.window.rate(),
		LifetimeRate:        lifetime,
		WindowTokens:        t.window.count(),
		WindowAccepted:      t.window.acceptedCount(),
		TotalProposed:       t.totalProposed,
		TotalAccepted:       t.totalAccepted,
		TotalRollbacks:      t.totalRollbacks,
		UpscaleCount:        t.upscaleCount,
		DownscaleCount:      t.downscaleCount,
		EffectiveThroughput: EstimateThroughput(t.currentK, t.window.rate(), t.cfg.Hardware),
	}
}

// Reset clears the rolling window and counters, resetting draft depth to InitialK.
func (t *AdaptiveDepthTuner) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.currentK = t.cfg.InitialK
	t.window.reset()
	t.totalProposed = 0
	t.totalAccepted = 0
	t.totalRollbacks = 0
	t.upscaleCount = 0
	t.downscaleCount = 0
}

// EstimateThroughput models decode throughput (tok/s) on unified memory architectures
// for a given draft depth K, acceptance rate, and hardware configuration.
func EstimateThroughput(k int, acceptanceRate float64, hw SweepConfig) float64 {
	if k < 1 {
		k = 1
	}
	if hw.ModelWeightGB <= 0 {
		hw = AppleSiliconMetalSweepConfig()
	}
	acceptanceRate = math.Max(0.0, math.Min(1.0, acceptanceRate))

	// Step-wise expected accepted draft tokens
	expectedAcceptedDrafts := 0.0
	cumProb := 1.0
	for i := 0; i < k; i++ {
		cumProb *= acceptanceRate
		expectedAcceptedDrafts += cumProb
	}
	// 1 target token + expected accepted drafts
	expectedTotalAccepted := 1.0 + expectedAcceptedDrafts

	// Rollback penalty occurs when draft tokens are rejected
	rollbackPenaltyGB := 0.30 * (1.0 - acceptanceRate) * float64(k)
	totalMemoryGB := hw.ModelWeightGB + float64(k)*hw.MTPHeadWeightGB + float64(k+1)*hw.KVTrafficPerTokGB + rollbackPenaltyGB

	// Streaming time across unified memory bus
	if hw.BusBandwidthGBs <= 0 {
		hw.BusBandwidthGBs = 150.0
	}
	tBusSec := totalMemoryGB / hw.BusBandwidthGBs

	// Compute time (base forward pass + draft steps)
	tComputeSec := (hw.BaseComputeMs + float64(k)*hw.DraftStepComputeMs) / 1000.0

	tStepSec := tBusSec + tComputeSec
	if tStepSec <= 0 {
		return 0.0
	}
	return expectedTotalAccepted / tStepSec
}

// FormatAdaptiveStats returns an aligned text summary of the tuner stats.
func FormatAdaptiveStats(s AdaptiveTunerStats) string {
	return fmt.Sprintf("MTP Adaptive Depth: K=%d | Rolling Accept: %.1f%% | Lifetime: %.1f%% | Tokens: %d/%d | Rollbacks: %d | Throughput: %.2f tok/s (Upscales: %d, Downscales: %d)",
		s.CurrentK, s.RollingRate*100, s.LifetimeRate*100, s.WindowAccepted, s.WindowTokens, s.TotalRollbacks, s.EffectiveThroughput, s.UpscaleCount, s.DownscaleCount)
}
