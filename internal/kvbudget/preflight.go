// Structural adaptation of Ollama scheduler load-path capacity planning
package kvbudget

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
)

const (
	// DefaultPreflightUtilization is the 0.80 VRAM ceiling before load is refused.
	DefaultPreflightUtilization = 0.80
	// DefaultRecoveryFraction is the 0.75 fraction of estimated usage that must
	// return before driver-reported free VRAM is trusted.
	DefaultRecoveryFraction = 0.75
)

var (
	ErrPreflightOverCeiling = errors.New("kvbudget: predicted load VRAM exceeds utilization ceiling")
	ErrShrinkExhausted      = errors.New("kvbudget: load shrink-retry ladder exhausted")
	ErrVRAMRecoveryStale    = errors.New("kvbudget: driver free VRAM has not recovered to threshold")
)

// PreflightEnabledFromEnv reports whether opt-in preflight gating is enabled.
// Default is false so existing load paths are unchanged.
func PreflightEnabledFromEnv() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("FAK_KVBUDGET_PREFLIGHT")))
	return v == "1" || v == "true" || v == "on"
}

// LoadPreflightSpec describes the predicted VRAM footprint for a model load.
type LoadPreflightSpec struct {
	FreeBytes        uint64  `json:"free_bytes"`
	WeightsBytes     uint64  `json:"weights_bytes"`
	Ctx              int     `json:"ctx"`
	MinCtx           int     `json:"min_ctx"`
	NumParallel      int     `json:"num_parallel"`
	BytesPerToken    uint64  `json:"bytes_per_token"`
	BatchSize        int     `json:"batch_size"`
	MinBatchSize     int     `json:"min_batch_size"`
	BytesPerBatchTok uint64  `json:"bytes_per_batch_tok"`
	Utilization      float64 `json:"utilization,omitempty"`
}

func (s LoadPreflightSpec) utilization() float64 {
	if s.Utilization <= 0 {
		return DefaultPreflightUtilization
	}
	return s.Utilization
}

// PredictedBytes returns weights + ctx*numParallel*bytesPerToken + batchSize*bytesPerBatchTok.
func (s LoadPreflightSpec) PredictedBytes() uint64 {
	kv := uint64(max(0, s.Ctx)) * uint64(max(1, s.NumParallel)) * s.BytesPerToken
	batch := uint64(max(0, s.BatchSize)) * s.BytesPerBatchTok
	return s.WeightsBytes + kv + batch
}

// CheckLoadPreflight verifies predicted VRAM stays within utilization * FreeBytes.
func CheckLoadPreflight(s LoadPreflightSpec) error {
	u := s.utilization()
	if !(u > 0 && u <= 1) || math.IsNaN(u) {
		return ErrUtilizationOutOfRange
	}
	ceiling := uint64(math.Floor(float64(s.FreeBytes) * u))
	pred := s.PredictedBytes()
	if pred > ceiling {
		return fmt.Errorf("%w: predicted=%d ceiling=%d", ErrPreflightOverCeiling, pred, ceiling)
	}
	return nil
}

// ShrinkAction names one step in the bounded OOM shrink-retry ladder.
type ShrinkAction string

const (
	ShrinkCtx         ShrinkAction = "shrink_ctx"
	ShrinkBatch       ShrinkAction = "shrink_batch"
	ShrinkEvictCached ShrinkAction = "evict_cached"
)

// ShrinkStepResult records the adjusted spec after one shrink-retry attempt.
type ShrinkStepResult struct {
	Action ShrinkAction      `json:"action"`
	Spec   LoadPreflightSpec `json:"spec"`
}

// NextLoadShrinkStep advances the bounded shrink-retry ladder on load OOM:
// 1. Halve Ctx down to MinCtx;
// 2. Halve BatchSize down to MinBatchSize;
// 3. Evict cached tiers if canEvictCached is true;
// 4. Otherwise fail with ErrShrinkExhausted.
func NextLoadShrinkStep(s LoadPreflightSpec, canEvictCached bool) (ShrinkStepResult, error) {
	minCtx := max(1, s.MinCtx)
	if s.Ctx > minCtx {
		next := max(minCtx, s.Ctx/2)
		s.Ctx = next
		return ShrinkStepResult{Action: ShrinkCtx, Spec: s}, nil
	}
	minBatch := max(1, s.MinBatchSize)
	if s.BatchSize > minBatch {
		next := max(minBatch, s.BatchSize/2)
		s.BatchSize = next
		return ShrinkStepResult{Action: ShrinkBatch, Spec: s}, nil
	}
	if canEvictCached {
		return ShrinkStepResult{Action: ShrinkEvictCached, Spec: s}, nil
	}
	return ShrinkStepResult{}, ErrShrinkExhausted
}

// WaitForVRAMRecovery accepts driver-reported free-byte samples only once at
// least DefaultRecoveryFraction (75%) of expectedUsageBytes has returned
// above baselineFreeBytes.
func WaitForVRAMRecovery(baselineFreeBytes, expectedUsageBytes uint64, samples []uint64) (uint64, error) {
	needed := uint64(math.Ceil(float64(expectedUsageBytes) * DefaultRecoveryFraction))
	for _, sample := range samples {
		if sample >= baselineFreeBytes && sample-baselineFreeBytes >= needed {
			return sample, nil
		}
	}
	return 0, fmt.Errorf("%w: needed +%d bytes recovery", ErrVRAMRecoveryStale, needed)
}
