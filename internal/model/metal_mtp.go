package model

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"
)

var (
	// ErrMetalMTPNilTarget is returned when MetalMTPCoordinator is passed a nil target session.
	ErrMetalMTPNilTarget = errors.New("model: MetalMTPCoordinator requires non-nil target session")

	// ErrMetalMTPTripwireDiverged is returned when greedy temperature-zero tripwire detects sampling divergence.
	ErrMetalMTPTripwireDiverged = errors.New("model: Metal MTP temperature-zero tripwire detected sampling divergence")

	// ErrMetalMTPNonFiniteLogits is returned when candidate verification encounters NaN or Inf logits.
	ErrMetalMTPNonFiniteLogits = errors.New("model: Metal MTP verification encountered non-finite logits")

	// ErrMetalMTPClosed is returned when operations are attempted on a closed coordinator.
	ErrMetalMTPClosed = errors.New("model: MetalMTPCoordinator session is closed")

	// ErrMetalMTPInvalidDraftDepth is returned when draft depth is outside supported bounds [1, 4].
	ErrMetalMTPInvalidDraftDepth = errors.New("model: Metal MTP draft depth must be between 1 and 4")
)

// StepCostFn calculates or overrides step and target latencies or speedup for an adaptive governor observation.
type StepCostFn func(proposed, accepted int, base Qwen38AdaptiveStepObservation) Qwen38AdaptiveStepObservation

// MetalMTPConfig configures the in-kernel Metal MTP draft-verify-rollback execution loop.
type MetalMTPConfig struct {
	// DraftDepth is the speculative draft depth K (1..4, default 4).
	DraftDepth int `json:"draft_depth"`

	// MinAcceptanceRate is the threshold (default 0.50) below which the coordinator smoothly
	// falls back to serial autoregressive decode without dropping or corrupting the session.
	MinAcceptanceRate float64 `json:"min_acceptance_rate"`

	// WindowSize is the rolling evaluation window size in tokens (default 32).
	WindowSize int `json:"window_size"`

	// EnforceGreedyTripwire enforces temperature=0.0 and repeat_penalty=1.0 during speculative
	// verification passes to protect draft token acceptance from sampling distortion.
	EnforceGreedyTripwire bool `json:"enforce_greedy_tripwire"`

	// FallbackToSerial enables fail-closed fallback to unassisted serial decode.
	FallbackToSerial bool `json:"fallback_to_serial"`

	// Adaptive enables dynamic adaptive draft depth auto-tuning via Qwen38MTPAdaptiveDepthGovernor.
	Adaptive bool `json:"adaptive"`

	// AdaptiveConfig specifies custom configuration for the adaptive depth governor.
	AdaptiveConfig *Qwen38AdaptiveConfig `json:"adaptive_config,omitempty"`
}

// DefaultMetalMTPConfig returns production defaults for the Metal MTP execution loop.
func DefaultMetalMTPConfig() MetalMTPConfig {
	return MetalMTPConfig{
		DraftDepth:            4,
		MinAcceptanceRate:     0.50,
		WindowSize:            32,
		EnforceGreedyTripwire: true,
		FallbackToSerial:      true,
	}
}

// MetalMTPAcceptanceStats reports runtime metrics for the MTP speculative loop.
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

	// Adaptive depth auto-tuning state
	ActiveDraftDepth int                         `json:"active_draft_depth,omitempty"`
	AdaptiveReceipt  *Qwen38AdaptiveDepthReceipt `json:"adaptive_receipt,omitempty"`
	DowngradeReason  Qwen38MTPDowngradeReason    `json:"downgrade_reason,omitempty"`
}

// MTPCheckpointRecorder records speculative candidate draft tokens and atomic page commit/rollback.
type MTPCheckpointRecorder interface {
	RecordMTPDraft(sessionID string, tokens []int32) error
	CommitMTPDraft(sessionID string, accepted int) (int, int, error)
	RollbackMTPDraft(sessionID string) (int, error)
}

// MTPDraftTracker records speculative candidate draft tokens, depth, and page state.
type MTPDraftTracker struct {
	DraftDepth       int     `json:"draft_depth"`
	DraftTokens      []int32 `json:"draft_tokens"`
	AcceptedCount    int     `json:"accepted_count"`
	RollbackOccurred bool    `json:"rollback_occurred"`
	AllocatedPages   int     `json:"allocated_pages"`
	CommittedPages   int     `json:"committed_pages"`
	FreedPages       int     `json:"freed_pages"`
}

// RecordDraft registers candidate draft tokens.
func (s *MTPDraftTracker) RecordDraft(tokens []int32) {
	s.DraftDepth = len(tokens)
	s.DraftTokens = append([]int32(nil), tokens...)
	s.AcceptedCount = 0
	s.RollbackOccurred = false
	s.AllocatedPages += len(tokens)
}

// CommitDraft atomically commits accepted draft tokens and tracks freed pages.
func (s *MTPDraftTracker) CommitDraft(accepted int) (committedPages int, freedPages int, err error) {
	if accepted < 0 || accepted > len(s.DraftTokens) {
		return 0, 0, fmt.Errorf("model: accepted %d outside range [0, %d]", accepted, len(s.DraftTokens))
	}
	s.AcceptedCount = accepted
	s.RollbackOccurred = accepted < len(s.DraftTokens)
	s.CommittedPages += accepted
	freed := len(s.DraftTokens) - accepted
	s.FreedPages += freed
	return accepted, freed, nil
}

// RollbackDraft frees all speculative draft pages.
func (s *MTPDraftTracker) RollbackDraft() (freedPages int, err error) {
	_, freed, err := s.CommitDraft(0)
	return freed, err
}

// MetalMTPCoordinator coordinates resident MTP candidate proposal generation,
// wide-M Metal verification dispatch, atomic Context-MMU page commit/rollback,
// and greedy temperature-zero tripwires into an autonomous speculative decode loop.
type MetalMTPCoordinator struct {
	mu sync.Mutex

	target   *Session
	cfg      MetalMTPConfig
	drafter  ProposalGenerator
	draftSes *Qwen35MTPDraftSession

	// Adaptive depth governance
	governor      *Qwen38MTPAdaptiveDepthGovernor
	stepCostFn    StepCostFn
	targetLatency time.Duration

	// Context-MMU state
	checkpointMgr MTPCheckpointRecorder
	sessionID     string
	draftState    *MTPDraftTracker

	// Rolling acceptance monitoring (32-token window)
	windowOutcomes []bool
	windowHead     int
	totalProposed  int
	totalAccepted  int
	totalRollbacks int

	// Fallback state
	inFallback     bool
	fallbackReason string

	// Tripwire state
	tripwireTripped bool
	tripwireReason  string

	totalGenerated int
	closed         bool
}

// NewMetalMTPCoordinator constructs a new MetalMTPCoordinator.
func NewMetalMTPCoordinator(target *Session, cfgs ...MetalMTPConfig) (*MetalMTPCoordinator, error) {
	cfg := DefaultMetalMTPConfig()
	if len(cfgs) > 0 {
		cfg = cfgs[0]
	}
	if cfg.DraftDepth <= 0 {
		cfg.DraftDepth = 4
	} else if cfg.DraftDepth < 1 {
		cfg.DraftDepth = 1
	} else if cfg.DraftDepth > 4 {
		cfg.DraftDepth = 4
	}

	if cfg.WindowSize <= 0 {
		cfg.WindowSize = 32
	}
	if cfg.MinAcceptanceRate <= 0 {
		cfg.MinAcceptanceRate = 0.50
	}

	c := &MetalMTPCoordinator{
		target:         target,
		cfg:            cfg,
		draftState:     &MTPDraftTracker{DraftDepth: cfg.DraftDepth},
		windowOutcomes: make([]bool, 0, cfg.WindowSize),
	}

	if cfg.Adaptive || cfg.AdaptiveConfig != nil {
		adCfg := DefaultQwen38AdaptiveConfig()
		if cfg.AdaptiveConfig != nil {
			adCfg = *cfg.AdaptiveConfig
		}
		if cfg.DraftDepth > 0 && cfg.AdaptiveConfig == nil {
			adCfg.MaxDepth = cfg.DraftDepth
		}
		gov, err := NewQwen38MTPAdaptiveDepthGovernor(adCfg)
		if err == nil {
			c.governor = gov
			c.draftState.DraftDepth = gov.CurrentDepth()
		}
	}

	if target != nil {
		target.captureTargetHidden = true
		c.ensureDrafterLocked()
	}

	return c, nil
}

// NewMetalMTPCoordinator binds an in-kernel Metal MTP draft-verify-rollback coordinator to this session.
func (s *Session) NewMetalMTPCoordinator(cfgs ...MetalMTPConfig) (*MetalMTPCoordinator, error) {
	return NewMetalMTPCoordinator(s, cfgs...)
}

// NewMetalMTPCoordinator constructs a MetalMTPCoordinator bound to the model and session.
func (m *Model) NewMetalMTPCoordinator(s *Session, cfgs ...MetalMTPConfig) (*MetalMTPCoordinator, error) {
	if s == nil {
		s = m.NewSession()
	}
	return NewMetalMTPCoordinator(s, cfgs...)
}

func (c *MetalMTPCoordinator) ensureDrafterLocked() {
	if c.drafter != nil {
		return
	}
	if c.target == nil || c.target.M == nil {
		return
	}
	c.target.captureTargetHidden = true
	if c.target.Cache == nil {
		c.target.Cache = NewKVCache(c.target.M.Cfg)
	}
	if c.target.M.Cfg.IsQwen35Hybrid() {
		ds, err := NewQwen35MTPDraftSession(c.target, c.cfg.DraftDepth)
		if err == nil {
			c.draftSes = ds
			c.drafter = NewMTPProposalGenerator(ds)
		}
	}
}

// SetDrafter injects an explicit ProposalGenerator (e.g. resident MTP, sidecar, or test mock).
func (c *MetalMTPCoordinator) SetDrafter(p ProposalGenerator) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.drafter = p
}

// SetMMU wires a Context-MMU CheckpointManager and session identifier.
func (c *MetalMTPCoordinator) SetMMU(cm MTPCheckpointRecorder, sessionID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.checkpointMgr = cm
	c.sessionID = sessionID
}

// TargetSession returns the underlying target model session.
func (c *MetalMTPCoordinator) TargetSession() *Session {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.target
}

// SetTargetSession updates the target model session.
func (c *MetalMTPCoordinator) SetTargetSession(s *Session) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.target = s
}

// Config returns the coordinator's active configuration.
func (c *MetalMTPCoordinator) Config() MetalMTPConfig {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.cfg
}

// InFallback reports whether the coordinator is currently operating in serial decode fallback mode.
func (c *MetalMTPCoordinator) InFallback() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.governor != nil && c.governor.CurrentDepth() == 0 {
		return true
	}
	return c.inFallback
}

// ResetFallback resets the fallback state back to speculative decoding.
func (c *MetalMTPCoordinator) ResetFallback() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.inFallback = false
	c.fallbackReason = ""
	c.windowOutcomes = c.windowOutcomes[:0]
	c.windowHead = 0
	if c.governor != nil {
		c.governor.Reset()
	}
}

// CheckSamplingTripwire validates sampling parameters against the greedy temperature-zero tripwire.
func (c *MetalMTPCoordinator) CheckSamplingTripwire(temperature float64, repeatPenalty float64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.cfg.EnforceGreedyTripwire {
		return nil
	}
	if temperature > 1e-6 || (repeatPenalty > 0 && math.Abs(repeatPenalty-1.0) > 1e-6) {
		c.tripwireTripped = true
		c.tripwireReason = fmt.Sprintf("greedy temperature-zero tripwire triggered: temperature=%.3f, repeat_penalty=%.3f", temperature, repeatPenalty)
		return ErrMetalMTPTripwireDiverged
	}
	return nil
}

// recordAcceptanceLocked updates rolling 32-token window and lifetime metrics under lock.
func (c *MetalMTPCoordinator) recordAcceptanceLocked(proposed, accepted int) {
	c.totalProposed += proposed
	c.totalAccepted += accepted
	if accepted < proposed {
		c.totalRollbacks += (proposed - accepted)
	}

	for i := 0; i < proposed; i++ {
		isAcc := i < accepted
		if len(c.windowOutcomes) < c.cfg.WindowSize {
			c.windowOutcomes = append(c.windowOutcomes, isAcc)
		} else {
			c.windowOutcomes[c.windowHead] = isAcc
			c.windowHead = (c.windowHead + 1) % c.cfg.WindowSize
		}
	}

	// Smoothly fall back to serial autoregressive decode if rolling acceptance < 50%
	// (only active when adaptive governor is not managing dynamic depth and target-only escape)
	if c.governor == nil && c.cfg.FallbackToSerial && len(c.windowOutcomes) >= 8 {
		rate := c.windowRateLocked()
		if rate < c.cfg.MinAcceptanceRate {
			c.inFallback = true
			c.fallbackReason = fmt.Sprintf("rolling acceptance rate %.1f%% dropped below %.1f%% threshold across %d tokens",
				rate*100, c.cfg.MinAcceptanceRate*100, len(c.windowOutcomes))
		}
	}
}

func (c *MetalMTPCoordinator) windowRateLocked() float64 {
	if len(c.windowOutcomes) == 0 {
		return 1.0
	}
	acc := 0
	for _, ok := range c.windowOutcomes {
		if ok {
			acc++
		}
	}
	return float64(acc) / float64(len(c.windowOutcomes))
}

// Stats returns a snapshot of runtime metrics.
func (c *MetalMTPCoordinator) Stats() MetalMTPAcceptanceStats {
	c.mu.Lock()
	defer c.mu.Unlock()

	var winAcc int
	for _, ok := range c.windowOutcomes {
		if ok {
			winAcc++
		}
	}

	rollingRate := 1.0
	if len(c.windowOutcomes) > 0 {
		rollingRate = float64(winAcc) / float64(len(c.windowOutcomes))
	}

	lifetimeRate := 1.0
	if c.totalProposed > 0 {
		lifetimeRate = float64(c.totalAccepted) / float64(c.totalProposed)
	}

	var committedPages, freedPages int
	if c.draftState != nil {
		committedPages = c.draftState.CommittedPages
		freedPages = c.draftState.FreedPages
	}

	stats := MetalMTPAcceptanceStats{
		TotalProposed:   c.totalProposed,
		TotalAccepted:   c.totalAccepted,
		TotalRollbacks:  c.totalRollbacks,
		WindowProposed:  len(c.windowOutcomes),
		WindowAccepted:  winAcc,
		RollingRate:     rollingRate,
		LifetimeRate:    lifetimeRate,
		InFallback:      c.inFallback,
		FallbackReason:  c.fallbackReason,
		TripwireTripped: c.tripwireTripped,
		TripwireReason:  c.tripwireReason,
		CommittedPages:  committedPages,
		FreedPages:      freedPages,
		TotalGenerated:  c.totalGenerated,
	}

	if c.governor != nil {
		stats.ActiveDraftDepth = c.governor.CurrentDepth()
		stats.DowngradeReason = c.governor.DowngradeReason()
		receipt := c.governor.Receipt()
		stats.AdaptiveReceipt = &receipt
		if c.governor.CurrentDepth() == 0 {
			stats.InFallback = true
			if stats.FallbackReason == "" {
				stats.FallbackReason = string(c.governor.DowngradeReason())
			}
		}
	} else {
		stats.ActiveDraftDepth = c.cfg.DraftDepth
		if c.inFallback {
			stats.DowngradeReason = Qwen38MTPNetLatencyRegressed
		} else {
			stats.DowngradeReason = Qwen38MTPEligible
		}
	}

	return stats
}

// StepRound executes one speculative draft-verify-rollback cycle or serial step.
// Given committed token sequence and current boundary logits:
//  1. Verifies logit finiteness and temperature-zero tripwire.
//  2. If in fallback mode or drafter absent: performs serial autoregressive Step.
//  3. Otherwise:
//     a. Generates K draft tokens from resident MTP head (K=2..4).
//     b. Validates vocabulary bounds of draft tokens.
//     c. Records draft in Context-MMU with speculative page tracking.
//     d. Evaluates draft tokens via wide-M verification dispatch in 1 pass.
//     e. Evaluates greedy acceptance against target argmax up to first divergence.
//     f. Atomically commits accepted token pages and immediately frees rejected pages in Context-MMU.
//     g. Rolls back unaccepted KV cache positions in target session.
//     h. Advances target session with the bonus token.
//     i. Records rolling acceptance rate (falling back to serial if < 50%).
func (c *MetalMTPCoordinator) StepRound(ctx context.Context, committed []int, boundaryLogits []float32) (accepted []int, bonus int, nextLogits []float32, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, -1, nil, ErrMetalMTPClosed
	}
	if c.target == nil {
		return nil, -1, nil, ErrMetalMTPNilTarget
	}
	if err := ctx.Err(); err != nil {
		return nil, -1, nil, err
	}

	// 1. Temperature-zero logit validity tripwire
	if c.cfg.EnforceGreedyTripwire {
		for _, l := range boundaryLogits {
			if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
				c.tripwireTripped = true
				c.tripwireReason = "non-finite logits in boundary"
				return nil, -1, nil, ErrMetalMTPNonFiniteLogits
			}
		}
	}

	target0 := argmaxF32(boundaryLogits)

	activeDepth := c.cfg.DraftDepth
	if c.governor != nil {
		activeDepth = c.governor.CurrentDepth()
	}

	// Ensure resident drafter is initialized
	c.ensureDrafterLocked()

	// 2. Fallback or target-only mode: smoothly execute single serial step
	if (c.governor != nil && activeDepth == 0) || c.inFallback || c.drafter == nil || activeDepth < 1 {
		start := time.Now()
		nextLogits = c.target.Step(target0)
		elapsed := time.Since(start)
		c.totalGenerated++
		if c.governor != nil {
			obs := Qwen38AdaptiveStepObservation{
				ProposedTokens: 0,
				AcceptedTokens: 0,
				StepLatency:    elapsed,
				TargetLatency:  c.targetLatency,
			}
			if c.stepCostFn != nil {
				obs = c.stepCostFn(0, 0, obs)
			}
			_, _, _ = c.governor.ObserveStep(obs)
		}
		return []int{target0}, -1, nextLogits, nil
	}

	// 3. Propose K draft tokens from resident MTP head (zero host memory copies)
	start := time.Now()
	prop, pErr := c.drafter.Propose(ctx, committed, activeDepth)
	if pErr != nil || len(prop.Tokens) < 1 {
		nextLogits = c.target.Step(target0)
		elapsed := time.Since(start)
		c.totalGenerated++
		if c.governor != nil {
			obs := Qwen38AdaptiveStepObservation{
				ProposedTokens: 0,
				AcceptedTokens: 0,
				StepLatency:    elapsed,
				TargetLatency:  c.targetLatency,
			}
			if c.stepCostFn != nil {
				obs = c.stepCostFn(0, 0, obs)
			}
			_, _, _ = c.governor.ObserveStep(obs)
		}
		return []int{target0}, -1, nextLogits, nil
	}

	drafts := prop.Tokens
	if len(drafts) > activeDepth {
		drafts = drafts[:activeDepth]
	}

	draftTokens32 := make([]int32, len(drafts))
	for i, t := range drafts {
		draftTokens32[i] = int32(t)
	}

	// Vocabulary sanity check: reject out-of-vocab candidates
	vocabSize := c.target.M.Cfg.VocabSize
	for _, tok := range drafts {
		if tok < 0 || (vocabSize > 0 && tok >= vocabSize) {
			c.recordAcceptanceLocked(len(drafts), 0)
			c.draftState.RecordDraft(draftTokens32)
			_, _, _ = c.draftState.CommitDraft(0)
			if c.checkpointMgr != nil && c.sessionID != "" {
				_ = c.checkpointMgr.RecordMTPDraft(c.sessionID, draftTokens32)
				_, _, _ = c.checkpointMgr.CommitMTPDraft(c.sessionID, 0)
			}
			nextLogits = c.target.Step(target0)
			elapsed := time.Since(start)
			c.totalGenerated++
			if c.governor != nil {
				obs := Qwen38AdaptiveStepObservation{
					ProposedTokens: len(drafts),
					AcceptedTokens: 0,
					StepLatency:    elapsed,
					TargetLatency:  c.targetLatency,
				}
				if c.stepCostFn != nil {
					obs = c.stepCostFn(len(drafts), 0, obs)
				}
				_, _, _ = c.governor.ObserveStep(obs)
			}
			return nil, target0, nextLogits, nil
		}
	}

	// 4. Capture pre-round verified snapshot for exact rollback
	snap, snapErr := c.target.PrefixSnapshot()
	if snapErr != nil {
		nextLogits = c.target.Step(target0)
		elapsed := time.Since(start)
		c.totalGenerated++
		if c.governor != nil {
			obs := Qwen38AdaptiveStepObservation{
				ProposedTokens: 0,
				AcceptedTokens: 0,
				StepLatency:    elapsed,
				TargetLatency:  c.targetLatency,
			}
			if c.stepCostFn != nil {
				obs = c.stepCostFn(0, 0, obs)
			}
			_, _, _ = c.governor.ObserveStep(obs)
		}
		return []int{target0}, -1, nextLogits, nil
	}
	defer snap.Close()

	// Record draft in Context-MMU with speculative page tracking
	c.draftState.RecordDraft(draftTokens32)
	if c.checkpointMgr != nil && c.sessionID != "" {
		_ = c.checkpointMgr.RecordMTPDraft(c.sessionID, draftTokens32)
	}

	// 5. Wide-M Metal verification dispatch (evaluates K candidate tokens in 1 weight-streaming pass)
	baseLen := c.target.Cache.Len()
	_ = baseLen
	var rows [][]float32
	if verifyForwardBatchedOK(c.target) {
		rawRows := c.target.VerifyForward(drafts, nil, nil)
		rows = make([][]float32, len(rawRows))
		for i, r := range rawRows {
			rows[i] = append([]float32(nil), r...)
		}
	} else {
		rows = make([][]float32, len(drafts))
		for i, tok := range drafts {
			stepLogits := c.target.Step(tok)
			rows[i] = append([]float32(nil), stepLogits...)
		}
	}

	if len(rows) != len(drafts) {
		// Verification returned mismatched rows: fail closed and rollback
		_, _ = c.draftState.RollbackDraft()
		if c.checkpointMgr != nil && c.sessionID != "" {
			_, _ = c.checkpointMgr.RollbackMTPDraft(c.sessionID)
		}
		_ = snap.Restore(c.target)
		nextLogits = c.target.Step(target0)
		elapsed := time.Since(start)
		c.totalGenerated++
		if c.governor != nil {
			obs := Qwen38AdaptiveStepObservation{
				ProposedTokens: len(drafts),
				AcceptedTokens: 0,
				StepLatency:    elapsed,
				TargetLatency:  c.targetLatency,
			}
			if c.stepCostFn != nil {
				obs = c.stepCostFn(len(drafts), 0, obs)
			}
			_, _, _ = c.governor.ObserveStep(obs)
		}
		return []int{target0}, -1, nextLogits, nil
	}

	// 6. Greedy temperature-zero verification tripwire
	if c.cfg.EnforceGreedyTripwire {
		for _, row := range rows {
			for _, l := range row {
				if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
					c.tripwireTripped = true
					c.tripwireReason = "non-finite logits in verification rows"
					_, _ = c.draftState.RollbackDraft()
					if c.checkpointMgr != nil && c.sessionID != "" {
						_, _ = c.checkpointMgr.RollbackMTPDraft(c.sessionID)
					}
					_ = snap.Restore(c.target)
					return nil, -1, nil, ErrMetalMTPNonFiniteLogits
				}
			}
		}
	}

	// Construct target argmax sequence: targetArgmax[i+1] is argmax of rows[i]
	targetArgmax := make([]int, len(drafts)+1)
	targetArgmax[0] = target0
	for i, r := range rows {
		targetArgmax[i+1] = argmaxF32(r)
	}

	// 7. Verify acceptance against target argmax
	accTokens, bonusTok, tripErr := TripwireVerify(drafts, targetArgmax, boundaryLogits, rows)
	if tripErr != nil {
		c.tripwireTripped = true
		c.tripwireReason = tripErr.Error()
		_, _ = c.draftState.RollbackDraft()
		if c.checkpointMgr != nil && c.sessionID != "" {
			_, _ = c.checkpointMgr.RollbackMTPDraft(c.sessionID)
		}
		_ = snap.Restore(c.target)
		return nil, -1, nil, tripErr
	}

	numAccepted := len(accTokens)

	// 8. Atomic Context-MMU page commit & rollback:
	// Commits accepted token pages and immediately frees rejected pages without memory leaks
	_, _, _ = c.draftState.CommitDraft(numAccepted)
	if c.checkpointMgr != nil && c.sessionID != "" {
		_, _, _ = c.checkpointMgr.CommitMTPDraft(c.sessionID, numAccepted)
	}

	// Roll back unaccepted KV cache positions and recurrent state in the target session
	if numAccepted < len(drafts) {
		clone, cErr := snap.Clone()
		if cErr == nil {
			_ = clone.Restore(c.target)
		} else {
			_ = snap.Restore(c.target)
		}
		for _, tok := range accTokens {
			c.target.Step(tok)
		}
	}

	// 9. Advance target session with the bonus token
	nextLogits = c.target.Step(bonusTok)
	elapsed := time.Since(start)
	c.totalGenerated += numAccepted + 1

	// 10. Update rolling acceptance monitoring
	c.recordAcceptanceLocked(len(drafts), numAccepted)

	// 11. Update adaptive depth governor if connected
	if c.governor != nil {
		obs := Qwen38AdaptiveStepObservation{
			ProposedTokens: len(drafts),
			AcceptedTokens: numAccepted,
			StepLatency:    elapsed,
			TargetLatency:  c.targetLatency,
		}
		if c.stepCostFn != nil {
			obs = c.stepCostFn(len(drafts), numAccepted, obs)
		}
		_, _, _ = c.governor.ObserveStep(obs)
	}

	return accTokens, bonusTok, nextLogits, nil
}

// Generate drives the complete in-kernel speculative generation loop from prompt to maxNew tokens.
// Output token sequence identity is 100% bit-exact with non-speculative autoregressive decode at temperature zero.
func (c *MetalMTPCoordinator) Generate(ctx context.Context, prompt []int, maxNew int) ([]int, error) {
	if c == nil {
		return nil, ErrMetalMTPNilTarget
	}
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil, ErrMetalMTPClosed
	}
	if c.target == nil {
		c.mu.Unlock()
		return nil, ErrMetalMTPNilTarget
	}
	c.mu.Unlock()

	if len(prompt) == 0 {
		return nil, errors.New("model: empty prompt in MetalMTPCoordinator.Generate")
	}
	if maxNew <= 0 {
		return nil, nil
	}

	boundaryLogits := c.target.Prefill(prompt)
	committed := append([]int(nil), prompt...)
	output := make([]int, 0, maxNew)
	eos := c.target.M.Cfg.EOSTokenID

	for len(output) < maxNew {
		if err := ctx.Err(); err != nil {
			return output, err
		}

		accTokens, bonusTok, nextLogits, err := c.StepRound(ctx, committed, boundaryLogits)
		if err != nil {
			return output, err
		}

		stopped := false
		for _, tok := range accTokens {
			if len(output) >= maxNew || (eos >= 0 && tok == eos) {
				stopped = true
				break
			}
			output = append(output, tok)
			committed = append(committed, tok)
		}
		if stopped || len(output) >= maxNew {
			break
		}

		if bonusTok >= 0 {
			if eos >= 0 && bonusTok == eos {
				break
			}
			output = append(output, bonusTok)
			committed = append(committed, bonusTok)
		}
		if len(output) >= maxNew {
			break
		}

		boundaryLogits = nextLogits
	}

	return output, nil
}

// Close releases the coordinator and any underlying draft session resources.
func (c *MetalMTPCoordinator) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}
	c.closed = true

	if c.draftState != nil {
		_, _ = c.draftState.RollbackDraft()
	}
	if c.draftSes != nil {
		c.draftSes.Close()
		c.draftSes = nil
	}
	return nil
}

// CurrentDraftDepth returns the currently adjudicated speculative draft depth K (0..MaxDepth).
func (c *MetalMTPCoordinator) CurrentDraftDepth() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.governor != nil {
		return c.governor.CurrentDepth()
	}
	if c.inFallback {
		return 0
	}
	return c.cfg.DraftDepth
}

// AdaptiveGovernor returns the currently configured adaptive depth governor, or nil if none.
func (c *MetalMTPCoordinator) AdaptiveGovernor() *Qwen38MTPAdaptiveDepthGovernor {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.governor
}

// SetAdaptiveGovernor binds an adaptive depth governor to dynamically control draft depth K.
func (c *MetalMTPCoordinator) SetAdaptiveGovernor(gov *Qwen38MTPAdaptiveDepthGovernor) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.governor = gov
	if gov != nil && c.draftState != nil {
		c.draftState.DraftDepth = gov.CurrentDepth()
	}
}

// SetStepCostFn registers a custom cost function for deterministic testing or profiling.
func (c *MetalMTPCoordinator) SetStepCostFn(fn StepCostFn) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.stepCostFn = fn
}

// SetTargetLatency sets the baseline serial target decode latency for speedup calculations.
func (c *MetalMTPCoordinator) SetTargetLatency(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.targetLatency = d
}

// DowngradeReason reports the typed downgrade reason (e.g. Qwen38MTPNetLatencyRegressed on target-only escape).
func (c *MetalMTPCoordinator) DowngradeReason() Qwen38MTPDowngradeReason {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.governor != nil {
		return c.governor.DowngradeReason()
	}
	if c.inFallback {
		return Qwen38MTPNetLatencyRegressed
	}
	return Qwen38MTPEligible
}

// Engine reports the active execution engine (Qwen38EngineMTP or Qwen38EngineTargetDecode).
func (c *MetalMTPCoordinator) Engine() Qwen38MTPEngine {
	c.mu.Lock()
	defer c.mu.Unlock()
	if (c.governor != nil && c.governor.CurrentDepth() == 0) || c.inFallback {
		return Qwen38EngineTargetDecode
	}
	return Qwen38EngineMTP
}
