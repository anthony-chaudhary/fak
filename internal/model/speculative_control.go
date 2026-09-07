package model

import (
	"errors"
	"fmt"
	"math"
	"sync/atomic"
)

// ErrInvalidAcceptanceThreshold indicates that a requested speculative acceptance threshold
// is not within the valid normalized range [0.0, 1.0] or is NaN.
var ErrInvalidAcceptanceThreshold = errors.New("model: speculative acceptance threshold must be between 0.0 and 1.0")

// SpeculativeControlSnapshot captures an immutable point-in-time state of the speculative controller.
type SpeculativeControlSnapshot struct {
	KMax                int     `json:"k_max"`
	Depth               int     `json:"depth"`
	AcceptanceThreshold float64 `json:"acceptance_threshold"`
	IsActive            bool    `json:"is_active"`
}

// SpeculativeControl provides thread-safe runtime control over speculative decoding parameters.
// It pre-allocates buffer capacities sized for a verification ceiling KMax to avoid any reallocation
// or CUDA graph invalidation during execution. The active draft depth K is stored in an atomic scalar
// constrained to 0 <= K <= KMax. When K = 0, the controller provides a clean bypass/fallback to
// autoregressive target decode with zero verification overhead. The acceptance threshold is hot-swappable
// via an atomic scalar.
type SpeculativeControl struct {
	kMax      int
	depth     atomic.Uint32 // active draft depth K in [0, kMax]
	threshold atomic.Uint64 // math.Float64bits of threshold in [0.0, 1.0]

	// Pre-allocated buffers sized for verification ceiling kMax to prevent CUDA graph invalidation
	draftTokensBuffer       []int32
	acceptedTokensBuffer    []int32
	draftTokensIntBuffer    []int
	acceptedTokensIntBuffer []int
	acceptanceMaskBuffer    []bool
}

// NewSpeculativeControl creates a new SpeculativeControl instance with buffer capacities pre-allocated
// to the ceiling kMax. The defaultK is clamped between 0 and kMax, and defaultThreshold is clamped
// between 0.0 and 1.0.
func NewSpeculativeControl(kMax int, defaultK int, defaultThreshold float64) *SpeculativeControl {
	if kMax < 0 {
		kMax = 0
	}
	ctrl := &SpeculativeControl{
		kMax:                    kMax,
		draftTokensBuffer:       make([]int32, kMax),
		acceptedTokensBuffer:    make([]int32, kMax+1),
		draftTokensIntBuffer:    make([]int, kMax),
		acceptedTokensIntBuffer: make([]int, kMax+1),
		acceptanceMaskBuffer:    make([]bool, kMax),
	}

	clampedK := ctrl.Clamp(defaultK)
	ctrl.depth.Store(uint32(clampedK))

	if math.IsNaN(defaultThreshold) || defaultThreshold < 0.0 {
		defaultThreshold = 0.0
	} else if defaultThreshold > 1.0 {
		defaultThreshold = 1.0
	}
	ctrl.threshold.Store(math.Float64bits(defaultThreshold))

	return ctrl
}

// GetDepth returns the active draft depth K.
func (c *SpeculativeControl) GetDepth() int {
	if c == nil {
		return 0
	}
	return int(c.depth.Load())
}

// SetDepth updates the active draft depth K, clamping it within [0, KMax].
// This operation is thread-safe and lock-free.
func (c *SpeculativeControl) SetDepth(k int) error {
	if c == nil {
		return errors.New("model: nil speculative control")
	}
	clamped := c.Clamp(k)
	c.depth.Store(uint32(clamped))
	return nil
}

// GetAcceptanceThreshold returns the currently active acceptance probability/entropy threshold.
func (c *SpeculativeControl) GetAcceptanceThreshold() float64 {
	if c == nil {
		return 0.0
	}
	bits := c.threshold.Load()
	return math.Float64frombits(bits)
}

// SetAcceptanceThreshold dynamically hot-swaps the speculative acceptance threshold.
// Validates that threshold is in [0.0, 1.0] and not NaN. This operation is thread-safe and lock-free.
func (c *SpeculativeControl) SetAcceptanceThreshold(threshold float64) error {
	if c == nil {
		return errors.New("model: nil speculative control")
	}
	if math.IsNaN(threshold) || threshold < 0.0 || threshold > 1.0 {
		return fmt.Errorf("%w: got %v", ErrInvalidAcceptanceThreshold, threshold)
	}
	c.threshold.Store(math.Float64bits(threshold))
	return nil
}

// KMax returns the verification ceiling KMax configured at construction.
func (c *SpeculativeControl) KMax() int {
	if c == nil {
		return 0
	}
	return c.kMax
}

// IsActive returns true if the active draft depth K > 0.
func (c *SpeculativeControl) IsActive() bool {
	if c == nil {
		return false
	}
	return c.GetDepth() > 0
}

// Clamp bounds requestedK within [0, KMax].
func (c *SpeculativeControl) Clamp(requestedK int) int {
	if c == nil {
		return 0
	}
	if requestedK < 0 {
		return 0
	}
	if requestedK > c.kMax {
		return c.kMax
	}
	return requestedK
}

// Snapshot returns an immutable point-in-time snapshot of the control state.
func (c *SpeculativeControl) Snapshot() SpeculativeControlSnapshot {
	if c == nil {
		return SpeculativeControlSnapshot{}
	}
	depth := c.GetDepth()
	threshold := c.GetAcceptanceThreshold()
	return SpeculativeControlSnapshot{
		KMax:                c.kMax,
		Depth:               depth,
		AcceptanceThreshold: threshold,
		IsActive:            depth > 0,
	}
}

// ShouldBypass returns true when active depth is 0, indicating that speculative
// drafting and verification should be bypassed in favor of pure target decode.
func (c *SpeculativeControl) ShouldBypass() bool {
	return !c.IsActive()
}

// Bypass is an alias for ShouldBypass.
func (c *SpeculativeControl) Bypass() bool {
	return !c.IsActive()
}

// StepOrBypass dispatches either speculative decode or pure target decode fallback.
// When K = 0, fallbackTargetFn is executed immediately with zero verification overhead.
func (c *SpeculativeControl) StepOrBypass(
	speculativeFn func(k int, threshold float64) error,
	fallbackTargetFn func() error,
) error {
	k := c.GetDepth()
	if k == 0 {
		if fallbackTargetFn != nil {
			return fallbackTargetFn()
		}
		return nil
	}
	if speculativeFn != nil {
		return speculativeFn(k, c.GetAcceptanceThreshold())
	}
	return nil
}

// ExecuteWithBypass executes speculativeFn when K > 0, or bypasses directly to fallbackTargetFn when K = 0.
func (c *SpeculativeControl) ExecuteWithBypass(
	speculativeFn func(k int, threshold float64) ([]int, error),
	fallbackTargetFn func() ([]int, error),
) ([]int, error) {
	k := c.GetDepth()
	if k == 0 {
		if fallbackTargetFn != nil {
			return fallbackTargetFn()
		}
		return nil, nil
	}
	if speculativeFn != nil {
		return speculativeFn(k, c.GetAcceptanceThreshold())
	}
	return nil, nil
}

// DraftTokensBuffer returns the pre-allocated int32 draft tokens buffer sized to KMax.
func (c *SpeculativeControl) DraftTokensBuffer() []int32 {
	if c == nil {
		return nil
	}
	return c.draftTokensBuffer
}

// AcceptedTokensBuffer returns the pre-allocated int32 accepted tokens buffer sized to KMax+1.
func (c *SpeculativeControl) AcceptedTokensBuffer() []int32 {
	if c == nil {
		return nil
	}
	return c.acceptedTokensBuffer
}

// DraftTokensIntBuffer returns the pre-allocated int draft tokens buffer sized to KMax.
func (c *SpeculativeControl) DraftTokensIntBuffer() []int {
	if c == nil {
		return nil
	}
	return c.draftTokensIntBuffer
}

// AcceptedTokensIntBuffer returns the pre-allocated int accepted tokens buffer sized to KMax+1.
func (c *SpeculativeControl) AcceptedTokensIntBuffer() []int {
	if c == nil {
		return nil
	}
	return c.acceptedTokensIntBuffer
}

// AcceptanceMaskBuffer returns the pre-allocated verification boolean mask buffer sized to KMax.
func (c *SpeculativeControl) AcceptanceMaskBuffer() []bool {
	if c == nil {
		return nil
	}
	return c.acceptanceMaskBuffer
}

// ActiveDraftSlice returns the pre-allocated int buffer sliced to current active depth K.
func (c *SpeculativeControl) ActiveDraftSlice() []int {
	if c == nil {
		return nil
	}
	k := c.GetDepth()
	return c.draftTokensIntBuffer[:k]
}

// ActiveDraftSlice32 returns the pre-allocated int32 buffer sliced to current active depth K.
func (c *SpeculativeControl) ActiveDraftSlice32() []int32 {
	if c == nil {
		return nil
	}
	k := c.GetDepth()
	return c.draftTokensBuffer[:k]
}
