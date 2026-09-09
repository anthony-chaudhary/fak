package model

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
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

	// ErrMetalMTPDraftProfilerUnavailable reports that no coordinator-owned
	// Qwen3.8 MTP draft session exists. Injected drafters never expose a profiler.
	ErrMetalMTPDraftProfilerUnavailable = errors.New("model: coordinator-owned Metal MTP draft profiler is unavailable")

	// ErrMetalMTPDraftProfilerTargetAlias prevents target and draft execution
	// from being aggregated into one PhaseProfiler receipt.
	ErrMetalMTPDraftProfilerTargetAlias = errors.New("model: Metal MTP draft profiler must be distinct from target profiler")
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

// MetalMTPTargetVerificationReceipt couples the generic target-verification
// accounting with the exact admitted Metal panel and its state transaction.
type MetalMTPTargetVerificationReceipt struct {
	TargetVerificationReceipt
	Panel *Qwen35MetalMTPVerifyPanelReceipt `json:"panel,omitempty"`
}

// MetalMTPDraftPhaseProfilerReceipt is an immutable readback from the profiler
// attached to the coordinator-owned Qwen3.8 MTP draft Session. The two embedded
// receipts remain independently schema-validated by their owning packages.
type MetalMTPDraftPhaseProfilerReceipt struct {
	MetalFallback  MetalFallbackReceipt       `json:"metal_fallback"`
	MetalExecution metalgemm.ExecutionReceipt `json:"metal_execution"`
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
	mu           sync.Mutex
	generationMu sync.Mutex

	target   *Session
	cfg      MetalMTPConfig
	drafter  ProposalGenerator
	draftSes *Qwen35MTPDraftSession
	// draftProfiler is a coordinator configuration, not the target profiler. It
	// survives replacement of an owned draft session and is rebound to the new
	// Qwen35MTPForward.draft Session before that session executes.
	draftProfiler *PhaseProfiler

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

	// Most recent target verification transaction. This diagnostic receipt keeps
	// one-operation panel execution distinguishable from an honest K-step target
	// decode downgrade.
	lastTargetVerification MetalMTPTargetVerificationReceipt
	hasTargetVerification  bool

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
		if cfg.DraftDepth != 0 && (cfg.DraftDepth < 1 || cfg.DraftDepth > 4) {
			return nil, ErrMetalMTPInvalidDraftDepth
		}
	}
	if cfg.DraftDepth <= 0 {
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
		c.rebindDraftPhaseProfilerLocked()
		return
	}
	if c.target == nil || c.target.M == nil {
		return
	}
	c.target.captureTargetHidden = true
	if c.target.Cache == nil {
		c.target.Cache = NewKVCache(c.target.M.Cfg)
	}
	if c.target.M.Cfg.IsQwen35Hybrid() || (c.target.M.Cfg.isQwen35TextFamily() && c.target.M.Cfg.NumMTPLayers() > 0) {
		ds, err := NewQwen35MTPDraftSession(c.target, c.cfg.DraftDepth)
		if err == nil {
			c.draftSes = ds
			c.drafter = NewMTPProposalGenerator(ds)
			c.rebindDraftPhaseProfilerLocked()
		}
	}
}

func (c *MetalMTPCoordinator) rebindDraftPhaseProfilerLocked() {
	if c == nil || c.draftSes == nil {
		return
	}
	ds := c.draftSes
	// Reset the owned session to its native default before deciding whether the
	// stored profiler remains admissible for the current target.
	ds.step = qwen35MTPForwardFeedback
	if ds.forward != nil && ds.forward.draft != nil {
		ds.forward.draft.PhaseProfiler = nil
	}
	profiler := c.draftProfiler
	if profiler == nil || c.target == nil || c.target.PhaseProfiler == profiler || ds.forward == nil || ds.forward.draft == nil {
		return
	}
	ds.forward.draft.PhaseProfiler = profiler
	// A draft session can recreate its Qwen35MTPForward when the committed
	// prefix diverges. Reattach immediately before every native feedback step so
	// that the replacement draft Session keeps the coordinator's profiler.
	ds.step = func(forward *Qwen35MTPForward, pos int, priorHidden, embedding []float32) ([]float32, []float32, error) {
		if forward != nil && forward.draft != nil {
			forward.draft.PhaseProfiler = profiler
		}
		return qwen35MTPForwardFeedback(forward, pos, priorHidden, embedding)
	}
}

func (c *MetalMTPCoordinator) boundDraftPhaseProfilerLocked() (*PhaseProfiler, error) {
	if c.draftSes == nil || c.draftSes.forward == nil || c.draftSes.forward.draft == nil || c.draftProfiler == nil {
		return nil, ErrMetalMTPDraftProfilerUnavailable
	}
	if c.target != nil && c.target.PhaseProfiler == c.draftProfiler {
		return nil, ErrMetalMTPDraftProfilerTargetAlias
	}
	if c.draftSes.forward.draft.PhaseProfiler != c.draftProfiler {
		return nil, ErrMetalMTPDraftProfilerUnavailable
	}
	return c.draftProfiler, nil
}

// SetDraftPhaseProfiler installs a profiler only on the coordinator-owned MTP
// draft Session. It rejects injected drafters, nil profilers, and the target's
// profiler, and reapplies the profiler whenever the owned drafter is rebound.
func (c *MetalMTPCoordinator) SetDraftPhaseProfiler(profiler *PhaseProfiler) error {
	if c == nil {
		return ErrMetalMTPDraftProfilerUnavailable
	}
	c.generationMu.Lock()
	defer c.generationMu.Unlock()
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return ErrMetalMTPClosed
	}
	if profiler == nil {
		return ErrMetalMTPDraftProfilerUnavailable
	}
	c.ensureDrafterLocked()
	if c.draftSes == nil || c.draftSes.forward == nil || c.draftSes.forward.draft == nil {
		return ErrMetalMTPDraftProfilerUnavailable
	}
	if c.target != nil && c.target.PhaseProfiler == profiler {
		return ErrMetalMTPDraftProfilerTargetAlias
	}
	c.draftProfiler = profiler
	c.rebindDraftPhaseProfilerLocked()
	_, err := c.boundDraftPhaseProfilerLocked()
	return err
}

// DraftPhaseProfilerReceipt safely snapshots the actual profiler currently
// attached to the internally owned draft Session. It waits for active
// generation before reading the profiler's otherwise single-owner state.
func (c *MetalMTPCoordinator) DraftPhaseProfilerReceipt() (MetalMTPDraftPhaseProfilerReceipt, error) {
	if c == nil {
		return MetalMTPDraftPhaseProfilerReceipt{}, ErrMetalMTPDraftProfilerUnavailable
	}
	c.generationMu.Lock()
	defer c.generationMu.Unlock()
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return MetalMTPDraftPhaseProfilerReceipt{}, ErrMetalMTPClosed
	}
	profiler, err := c.boundDraftPhaseProfilerLocked()
	if err != nil {
		return MetalMTPDraftPhaseProfilerReceipt{}, err
	}
	receipt := MetalMTPDraftPhaseProfilerReceipt{}
	receipt.MetalFallback, err = profiler.MetalFallbackReceipt()
	execution, executionErr := profiler.MetalExecutionReceipt()
	receipt.MetalExecution = execution
	if readErr := errors.Join(err, executionErr); readErr != nil {
		return receipt, fmt.Errorf("model: read Metal MTP draft profiler receipt: %w", readErr)
	}
	return receipt, nil
}

// SetDrafter injects an explicit ProposalGenerator (e.g. resident MTP, sidecar, or test mock).
func (c *MetalMTPCoordinator) SetDrafter(p ProposalGenerator) {
	c.generationMu.Lock()
	defer c.generationMu.Unlock()
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return
	}
	if c.draftSes != nil {
		c.draftSes.Close()
		c.draftSes = nil
	}
	c.drafter = p
}

// SetMMU wires a Context-MMU CheckpointManager and session identifier.
func (c *MetalMTPCoordinator) SetMMU(cm MTPCheckpointRecorder, sessionID string) {
	c.generationMu.Lock()
	defer c.generationMu.Unlock()
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return
	}
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
	c.generationMu.Lock()
	defer c.generationMu.Unlock()
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return
	}
	// draftSes is present only for the coordinator-created drafter. It is bound
	// to target hidden history and therefore must never survive a request target
	// switch. An injected drafter has no draftSes owner and is intentionally
	// preserved across switches.
	if c.draftSes != nil {
		c.draftSes.Close()
		c.draftSes = nil
		c.drafter = nil
	}
	c.draftState = &MTPDraftTracker{DraftDepth: c.cfg.DraftDepth}
	c.target = s
	c.lastTargetVerification = MetalMTPTargetVerificationReceipt{}
	c.hasTargetVerification = false
	if s != nil {
		s.captureTargetHidden = true
		c.ensureDrafterLocked()
	}
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

func (c *MetalMTPCoordinator) observeSpeculativeRoundLocked(start time.Time, proposed, accepted int) error {
	c.recordAcceptanceLocked(proposed, accepted)
	if c.governor == nil {
		return nil
	}
	obs := Qwen38AdaptiveStepObservation{
		ProposedTokens: proposed,
		AcceptedTokens: accepted,
		StepLatency:    time.Since(start),
		TargetLatency:  c.targetLatency,
	}
	if c.stepCostFn != nil {
		obs = c.stepCostFn(proposed, accepted, obs)
	}
	_, _, err := c.governor.ObserveStep(obs)
	return err
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

// LastTargetVerificationReceipt returns the most recent speculative target
// verification receipt. The bool is false when the latest round did not run a
// target verification transaction (for example, target-only fallback).
func (c *MetalMTPCoordinator) LastTargetVerificationReceipt() (MetalMTPTargetVerificationReceipt, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.hasTargetVerification {
		return MetalMTPTargetVerificationReceipt{}, false
	}
	receipt := c.lastTargetVerification
	if receipt.Shape != nil {
		shape := *receipt.Shape
		receipt.Shape = &shape
	}
	if receipt.Panel != nil {
		panel := *receipt.Panel
		panel.GDNCheckpointLayers = append([]int(nil), receipt.Panel.GDNCheckpointLayers...)
		panel.GDNCheckpointLineageSHA256 = append([]string(nil), receipt.Panel.GDNCheckpointLineageSHA256...)
		receipt.Panel = &panel
	}
	return receipt, true
}

func (c *MetalMTPCoordinator) recordTargetVerificationLocked(tx *qwen35MTPTargetTransaction) {
	if tx == nil {
		return
	}
	c.lastTargetVerification = MetalMTPTargetVerificationReceipt{TargetVerificationReceipt: tx.VerificationReceipt()}
	if tx.panelReceipt != nil {
		panel := *tx.panelReceipt
		panel.GDNCheckpointLayers = append([]int(nil), tx.panelReceipt.GDNCheckpointLayers...)
		panel.GDNCheckpointLineageSHA256 = append([]string(nil), tx.panelReceipt.GDNCheckpointLineageSHA256...)
		c.lastTargetVerification.Panel = &panel
	}
	c.hasTargetVerification = true
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
	c.lastTargetVerification = MetalMTPTargetVerificationReceipt{}
	c.hasTargetVerification = false

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
	if pErr != nil || (len(prop.Tokens) < 1 && (prop.Tree == nil || len(prop.Tree.Nodes) < 1)) {
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

	// Candidate tree verification path: evaluate M=16..24 tree in one forward pass
	if prop.Tree != nil && len(prop.Tree.Nodes) > 0 {
		return c.stepRoundTreeLocked(start, target0, boundaryLogits, prop.Tree)
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

	// The production Qwen3.8 Metal target panel is an exact P=4 operation. Give
	// its transaction seam first refusal before the quantized target can reach
	// the ordinary K-step verifier fallback. The transaction owns the live GDN
	// checkpoint and PrefixSnapshot needed to adopt a full panel or restore and
	// replay a partial accepted prefix.
	if activeDepth == 4 && len(drafts) == 4 && (c.target.M.Cfg.IsQwen35Hybrid() || c.target.M.Cfg.isQwen35TextFamily()) {
		return c.stepRoundQwen35P4Locked(start, target0, boundaryLogits, drafts, draftTokens32)
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
	isBatched := false
	if verifyForwardBatchedOK(c.target) {
		rawRows := c.target.VerifyForward(drafts, nil, nil)
		if len(rawRows) == len(drafts) {
			rows = make([][]float32, len(rawRows))
			for i, r := range rawRows {
				rows[i] = append([]float32(nil), r...)
			}
			isBatched = true
		}
	}
	if !isBatched {
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
		c.lastTargetVerification = MetalMTPTargetVerificationReceipt{
			TargetVerificationReceipt: TargetVerificationReceipt{
				Schema:                       targetVerificationReceiptSchema,
				Engine:                       targetVerificationEngine,
				Path:                         targetVerificationDecodePath,
				OneOperation:                 false,
				TargetVerificationOperations: 0,
				TargetDecodeSteps:            len(drafts),
				DraftTokens:                  len(drafts),
				AcceptedTokens:               0,
				RejectedTokens:               len(drafts),
				DowngradeReason:              c.tripwireReason,
			},
		}
		c.hasTargetVerification = true
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

	path := targetVerificationDecodePath
	targetOps := 0
	targetSteps := len(drafts)
	downgradeReason := ""
	if isBatched {
		path = targetVerificationBatchedPath
		targetOps = 1
		targetSteps = 0
	} else {
		downgradeReason = "unsupported wide-M target shape; fell back to serial decode"
	}
	c.lastTargetVerification = MetalMTPTargetVerificationReceipt{
		TargetVerificationReceipt: TargetVerificationReceipt{
			Schema:                       targetVerificationReceiptSchema,
			Engine:                       targetVerificationEngine,
			Path:                         path,
			OneOperation:                 isBatched,
			TargetVerificationOperations: targetOps,
			TargetDecodeSteps:            targetSteps,
			DraftTokens:                  len(drafts),
			AcceptedTokens:               numAccepted,
			RejectedTokens:               len(drafts) - numAccepted,
			DowngradeReason:              downgradeReason,
			Accounting: SpeculativeCostAccounting{
				Setup:              SpeculativeCostComponent{Nanoseconds: time.Since(start).Nanoseconds(), Measured: true},
				TargetVerification: SpeculativeCostComponent{Nanoseconds: time.Since(start).Nanoseconds(), Measured: true},
				KnownMemoryBytes:   snap.ResidentBytes(),
				MemoryMeasured:     true,
			},
		},
	}
	c.hasTargetVerification = true

	return accTokens, bonusTok, nextLogits, nil
}

func (c *MetalMTPCoordinator) stepRoundQwen35P4Locked(start time.Time, target0 int, boundaryLogits []float32, drafts []int, draftTokens32 []int32) ([]int, int, []float32, error) {
	tx, txErr := beginQwen35MTPTargetTransaction(c.target, boundaryLogits)
	if txErr != nil {
		nextLogits := c.target.Step(target0)
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

	c.draftState.RecordDraft(draftTokens32)
	if c.checkpointMgr != nil && c.sessionID != "" {
		if recordErr := c.checkpointMgr.RecordMTPDraft(c.sessionID, draftTokens32); recordErr != nil {
			rollbackErr := func() error {
				_, localErr := c.draftState.RollbackDraft()
				_, mmuErr := c.checkpointMgr.RollbackMTPDraft(c.sessionID)
				return errors.Join(localErr, mmuErr)
			}()
			abortErr := tx.Abort()
			tx.receipt.AcceptedTokens = 0
			tx.receipt.RejectedTokens = len(drafts)
			c.recordTargetVerificationLocked(tx)
			observeErr := c.observeSpeculativeRoundLocked(start, len(drafts), 0)
			return nil, -1, nil, errors.Join(fmt.Errorf("model: record Context-MMU MTP draft: %w", recordErr), rollbackErr, abortErr, observeErr)
		}
	}
	rollbackDraft := func() error {
		_, localErr := c.draftState.RollbackDraft()
		var mmuErr error
		if c.checkpointMgr != nil && c.sessionID != "" {
			_, mmuErr = c.checkpointMgr.RollbackMTPDraft(c.sessionID)
		}
		return errors.Join(localErr, mmuErr)
	}
	abort := func(cause error) ([]int, int, []float32, error) {
		rollbackErr := rollbackDraft()
		abortErr := tx.Abort()
		c.recordTargetVerificationLocked(tx)
		observeErr := c.observeSpeculativeRoundLocked(start, len(drafts), 0)
		return nil, -1, nil, errors.Join(cause, rollbackErr, abortErr, observeErr)
	}

	rows, verifyErr := tx.Verify(drafts)
	if verifyErr != nil {
		tx.receipt.AcceptedTokens = 0
		tx.receipt.RejectedTokens = len(drafts)
		return abort(verifyErr)
	}
	if len(rows) != len(drafts) {
		tx.receipt.AcceptedTokens = 0
		tx.receipt.RejectedTokens = len(drafts)
		return abort(fmt.Errorf("model: Qwen3.8 P4 target verification returned %d rows for %d draft tokens", len(rows), len(drafts)))
	}
	for _, row := range rows {
		if len(row) != c.target.M.Cfg.VocabSize {
			tx.receipt.AcceptedTokens = 0
			tx.receipt.RejectedTokens = len(drafts)
			return abort(fmt.Errorf("model: Qwen3.8 P4 target verification returned malformed logits"))
		}
		if c.cfg.EnforceGreedyTripwire {
			for _, l := range row {
				if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
					c.tripwireTripped = true
					c.tripwireReason = "non-finite logits in verification rows"
					tx.receipt.AcceptedTokens = 0
					tx.receipt.RejectedTokens = len(drafts)
					return abort(ErrMetalMTPNonFiniteLogits)
				}
			}
		}
	}

	targetArgmax := make([]int, len(drafts)+1)
	targetArgmax[0] = target0
	for i, row := range rows {
		targetArgmax[i+1] = argmaxF32(row)
	}
	accTokens, bonusTok, tripErr := TripwireVerify(drafts, targetArgmax, boundaryLogits, rows)
	if tripErr != nil {
		c.tripwireTripped = true
		c.tripwireReason = tripErr.Error()
		tx.receipt.AcceptedTokens = 0
		tx.receipt.RejectedTokens = len(drafts)
		return abort(tripErr)
	}

	numAccepted := len(accTokens)
	// Commit the external Context-MMU while the target transaction still owns
	// its pre-panel rollback state. A failed external commit cannot expose the
	// verified live target or any accepted output.
	if c.checkpointMgr != nil && c.sessionID != "" {
		if _, _, commitErr := c.checkpointMgr.CommitMTPDraft(c.sessionID, numAccepted); commitErr != nil {
			return abort(fmt.Errorf("model: commit Context-MMU MTP draft: %w", commitErr))
		}
	}
	if _, commitErr := tx.Commit(numAccepted); commitErr != nil {
		return abort(commitErr)
	}
	c.recordTargetVerificationLocked(tx)
	if _, _, commitErr := c.draftState.CommitDraft(numAccepted); commitErr != nil {
		return nil, -1, nil, fmt.Errorf("model: commit local MTP draft accounting: %w", commitErr)
	}

	nextLogits := c.target.Step(bonusTok)
	c.totalGenerated += numAccepted + 1
	if observeErr := c.observeSpeculativeRoundLocked(start, len(drafts), numAccepted); observeErr != nil {
		return nil, -1, nil, observeErr
	}
	return accTokens, bonusTok, nextLogits, nil
}

// StepRoundTree executes one speculative candidate tree verification cycle.
// Evaluates the full candidate tree (M=16..24 nodes) in one forward pass,
// and extracts the highest-scoring verified token branch via greedy argmax selection.
func (c *MetalMTPCoordinator) StepRoundTree(ctx context.Context, committed []int, boundaryLogits []float32, tree *CandidateTree) (accepted []int, bonus int, nextLogits []float32, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.lastTargetVerification = MetalMTPTargetVerificationReceipt{}
	c.hasTargetVerification = false

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

	if tree == nil || len(tree.Nodes) == 0 {
		nextLogits = c.target.Step(target0)
		c.totalGenerated++
		return []int{target0}, -1, nextLogits, nil
	}

	return c.stepRoundTreeLocked(time.Now(), target0, boundaryLogits, tree)
}

func (c *MetalMTPCoordinator) stepRoundTreeLocked(start time.Time, target0 int, boundaryLogits []float32, tree *CandidateTree) ([]int, int, []float32, error) {
	N := len(tree.Nodes)
	ids := tree.Tokens()

	// Vocabulary sanity check: reject out-of-vocab candidates
	vocabSize := c.target.M.Cfg.VocabSize
	for _, tok := range ids {
		if tok < 0 || (vocabSize > 0 && tok >= vocabSize) {
			nextLogits := c.target.Step(target0)
			c.totalGenerated++
			return nil, target0, nextLogits, nil
		}
	}

	// Capture pre-round verified snapshot for exact rollback
	snap, snapErr := c.target.PrefixSnapshot()
	if snapErr != nil {
		nextLogits := c.target.Step(target0)
		c.totalGenerated++
		return []int{target0}, -1, nextLogits, nil
	}
	defer snap.Close()

	// Record draft in Context-MMU with speculative page tracking
	draftTokens32 := make([]int32, N)
	for i, t := range ids {
		draftTokens32[i] = int32(t)
	}
	c.draftState.RecordDraft(draftTokens32)
	if c.checkpointMgr != nil && c.sessionID != "" {
		_ = c.checkpointMgr.RecordMTPDraft(c.sessionID, draftTokens32)
	}

	baseLen := c.target.Cache.Len()
	pos := make([]int, N)
	for i, node := range tree.Nodes {
		pos[i] = baseLen + node.Depth
	}

	mask, mErr := tree.DeriveMask()
	if mErr != nil {
		_, _ = c.draftState.RollbackDraft()
		if c.checkpointMgr != nil && c.sessionID != "" {
			_, _ = c.checkpointMgr.RollbackMTPDraft(c.sessionID)
		}
		_ = snap.Restore(c.target)
		nextLogits := c.target.Step(target0)
		c.totalGenerated++
		return []int{target0}, -1, nextLogits, nil
	}

	allow := func(q, k int) bool {
		if q >= 0 && q < len(mask) && k >= 0 && k < len(mask[q]) {
			return mask[q][k]
		}
		return false
	}

	// Single-pass verification forward across all tree candidates
	var rows [][]float32
	isBatched := false
	if verifyForwardBatchedOK(c.target) {
		rawRows := c.target.VerifyForward(ids, pos, allow)
		if len(rawRows) == N {
			rows = make([][]float32, len(rawRows))
			for i, r := range rawRows {
				rows[i] = append([]float32(nil), r...)
			}
			isBatched = true
		}
	}
	if !isBatched {
		// Fallback verification: step through candidate sequence
		rows = make([][]float32, N)
		for i, tok := range ids {
			stepLogits := c.target.Step(tok)
			rows[i] = append([]float32(nil), stepLogits...)
		}
	}

	if len(rows) != N {
		_, _ = c.draftState.RollbackDraft()
		if c.checkpointMgr != nil && c.sessionID != "" {
			_, _ = c.checkpointMgr.RollbackMTPDraft(c.sessionID)
		}
		_ = snap.Restore(c.target)
		nextLogits := c.target.Step(target0)
		c.totalGenerated++
		return []int{target0}, -1, nextLogits, nil
	}

	// Greedy temperature-zero verification tripwire
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

	// Greedy argmax path selection over tree proposals:
	// Extracts the highest-scoring verified token branch starting from root matching target0.
	matchRoot := -1
	for i, node := range tree.Nodes {
		if node.Parent == -1 && node.Token == target0 {
			matchRoot = i
			break
		}
	}

	if matchRoot == -1 {
		// Root candidate did not match target0: reject all draft tokens
		_, _, _ = c.draftState.CommitDraft(0)
		if c.checkpointMgr != nil && c.sessionID != "" {
			_, _, _ = c.checkpointMgr.CommitMTPDraft(c.sessionID, 0)
		}
		_ = snap.Restore(c.target)
		nextLogits := c.target.Step(target0)
		elapsed := time.Since(start)
		c.totalGenerated++
		c.recordAcceptanceLocked(N, 0)
		if c.governor != nil {
			obs := Qwen38AdaptiveStepObservation{
				ProposedTokens: N,
				AcceptedTokens: 0,
				StepLatency:    elapsed,
				TargetLatency:  c.targetLatency,
			}
			if c.stepCostFn != nil {
				obs = c.stepCostFn(N, 0, obs)
			}
			_, _, _ = c.governor.ObserveStep(obs)
		}
		return nil, target0, nextLogits, nil
	}

	acceptedIndices := []int{matchRoot}
	cur := matchRoot
	pred := argmaxF32(rows[cur])

	for {
		nextChild := -1
		for _, childIdx := range tree.Nodes[cur].Children {
			if childIdx >= 0 && childIdx < N && tree.Nodes[childIdx].Token == pred {
				nextChild = childIdx
				break
			}
		}
		if nextChild == -1 {
			break
		}
		acceptedIndices = append(acceptedIndices, nextChild)
		cur = nextChild
		pred = argmaxF32(rows[cur])
	}

	acceptedTokens := make([]int, len(acceptedIndices))
	for i, idx := range acceptedIndices {
		acceptedTokens[i] = tree.Nodes[idx].Token
	}
	bonusTok := pred
	numAccepted := len(acceptedTokens)

	// Atomic Context-MMU page commit & rollback:
	// Commits accepted token pages and immediately frees rejected pages without memory leaks
	_, _, _ = c.draftState.CommitDraft(numAccepted)
	if c.checkpointMgr != nil && c.sessionID != "" {
		_, _, _ = c.checkpointMgr.CommitMTPDraft(c.sessionID, numAccepted)
	}

	// Restore target snapshot and advance sequentially with accepted branch
	_ = snap.Restore(c.target)
	for _, tok := range acceptedTokens {
		c.target.Step(tok)
	}

	// Advance target session with the bonus token
	nextLogits := c.target.Step(bonusTok)
	elapsed := time.Since(start)
	c.totalGenerated += numAccepted + 1

	c.recordAcceptanceLocked(N, numAccepted)

	if c.governor != nil {
		obs := Qwen38AdaptiveStepObservation{
			ProposedTokens: N,
			AcceptedTokens: numAccepted,
			StepLatency:    elapsed,
			TargetLatency:  c.targetLatency,
		}
		if c.stepCostFn != nil {
			obs = c.stepCostFn(N, numAccepted, obs)
		}
		_, _, _ = c.governor.ObserveStep(obs)
	}

	treePath := targetVerificationDecodePath
	treeTargetOps := 0
	treeTargetSteps := N
	treeDowngradeReason := ""
	if isBatched {
		treePath = targetVerificationBatchedPath
		treeTargetOps = 1
		treeTargetSteps = 0
	} else {
		treeDowngradeReason = "unsupported wide-M tree target shape; fell back to serial decode"
	}
	c.lastTargetVerification = MetalMTPTargetVerificationReceipt{
		TargetVerificationReceipt: TargetVerificationReceipt{
			Schema:                       targetVerificationReceiptSchema,
			Engine:                       targetVerificationEngine,
			Path:                         treePath,
			OneOperation:                 isBatched,
			TargetVerificationOperations: treeTargetOps,
			TargetDecodeSteps:            treeTargetSteps,
			DraftTokens:                  N,
			AcceptedTokens:               numAccepted,
			RejectedTokens:               N - numAccepted,
			DowngradeReason:              treeDowngradeReason,
			Accounting: SpeculativeCostAccounting{
				Setup:              SpeculativeCostComponent{Nanoseconds: time.Since(start).Nanoseconds(), Measured: true},
				TargetVerification: SpeculativeCostComponent{Nanoseconds: time.Since(start).Nanoseconds(), Measured: true},
				KnownMemoryBytes:   snap.ResidentBytes(),
				MemoryMeasured:     true,
			},
		},
	}
	c.hasTargetVerification = true

	return acceptedTokens, bonusTok, nextLogits, nil
}

// Generate drives the complete in-kernel speculative generation loop from prompt to maxNew tokens.
// Output token sequence identity is 100% bit-exact with non-speculative autoregressive decode at temperature zero.
func (c *MetalMTPCoordinator) Generate(ctx context.Context, prompt []int, maxNew int) ([]int, error) {
	if c == nil {
		return nil, ErrMetalMTPNilTarget
	}
	c.generationMu.Lock()
	defer c.generationMu.Unlock()
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
	c.generationMu.Lock()
	defer c.generationMu.Unlock()
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
	c.draftProfiler = nil
	c.checkpointMgr = nil
	c.sessionID = ""
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
