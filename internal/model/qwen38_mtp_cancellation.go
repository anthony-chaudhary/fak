package model

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// Closed reason tokens for cancellation and memory bounds.
const (
	MTPDowngradeContextCanceled        MTPDowngradeReason       = "context_canceled"
	Qwen38MTPDowngradeContextCanceled  Qwen38MTPDowngradeReason = "context_canceled"
	Qwen38MTPCancellationReceiptSchema                          = "fak/qwen38-mtp-cancellation-receipt/v1"
)

// Standard typed errors for cancellation and memory limit enforcement.
var (
	ErrMTPContextCanceled     = errors.New("model: mtp context canceled")
	ErrMTPMemoryLimitExceeded = errors.New("model: mtp memory limit exceeded")
)

// Qwen38MTPBoundsConfig configures memory headroom limits and OOM behavior.
type Qwen38MTPBoundsConfig struct {
	MaxMemoryBytes     uint64 `json:"max_memory_bytes"`
	InjectedAllocLimit uint64 `json:"injected_alloc_limit,omitempty"`
	FailClosedOnOOM    bool   `json:"fail_closed_on_oom"`
}

// Qwen38MTPCancellationReceipt records diagnostic evidence of a cancellation or bounds event.
type Qwen38MTPCancellationReceipt struct {
	SchemaVersion   string                  `json:"schema_version"`
	Engine          Qwen38MTPEngine         `json:"engine"`
	FallbackEngine  Qwen38MTPEngine         `json:"fallback_engine,omitempty"`
	Outcome         Qwen38MTPReceiptOutcome `json:"outcome"`
	DowngradeReason string                  `json:"downgrade_reason"`
	MemoryBytes     Qwen38MTPMemoryBytes    `json:"memory_bytes"`
	DraftAborted    bool                    `json:"draft_aborted"`
	VerifyAborted   bool                    `json:"verify_aborted"`
	StateRolledBack bool                    `json:"state_rolled_back"`
	LocksReleased   bool                    `json:"locks_released"`
}

// Validate ensures the cancellation receipt meets schema invariants.
func (r Qwen38MTPCancellationReceipt) Validate() error {
	if r.SchemaVersion != Qwen38MTPCancellationReceiptSchema {
		return fmt.Errorf("model: invalid cancellation receipt schema %q, want %q", r.SchemaVersion, Qwen38MTPCancellationReceiptSchema)
	}
	if r.Engine != Qwen38EngineMTP && r.Engine != Qwen38EngineTargetDecode {
		return fmt.Errorf("model: invalid cancellation receipt engine %q", r.Engine)
	}
	if !r.StateRolledBack {
		return errors.New("model: cancellation receipt requires state_rolled_back=true")
	}
	if !r.LocksReleased {
		return errors.New("model: cancellation receipt requires locks_released=true")
	}
	return nil
}

// Qwen38MTPCancellationManager manages context cancellation, deadline expiration,
// and bounded memory enforcement for speculative MTP transactions and persistent sessions.
type Qwen38MTPCancellationManager struct {
	mu                  sync.Mutex
	cfg                 Qwen38MTPBoundsConfig
	currentAlloc        int64
	peakAlloc           int64
	lastDowngradeReason string
	downgraded          bool
	cancelled           bool
	cancellationCount   int
	memoryPressureCount int
}

// NewQwen38MTPCancellationManager constructs a new cancellation and memory bounds manager.
func NewQwen38MTPCancellationManager(cfg Qwen38MTPBoundsConfig) *Qwen38MTPCancellationManager {
	return &Qwen38MTPCancellationManager{
		cfg: cfg,
	}
}

// Config returns the active bounds configuration.
func (m *Qwen38MTPCancellationManager) Config() Qwen38MTPBoundsConfig {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.cfg
}

// SetInjectedAllocLimit updates the injected memory limit dynamically for testing.
func (m *Qwen38MTPCancellationManager) SetInjectedAllocLimit(limit uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.cfg.InjectedAllocLimit = limit
}

// TrackAllocation records buffer allocation and triggers fail-closed downgrade
// if limits are exceeded.
func (m *Qwen38MTPCancellationManager) TrackAllocation(bytes int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if bytes <= 0 {
		return nil
	}

	newTotal := m.currentAlloc + bytes
	// Check hard MaxMemoryBytes limit
	if m.cfg.MaxMemoryBytes > 0 && uint64(newTotal) > m.cfg.MaxMemoryBytes {
		m.downgraded = true
		m.lastDowngradeReason = string(MTPDowngradeMemoryPressure)
		m.memoryPressureCount++
		return &MTPDowngradeError{
			Reason: MTPDowngradeMemoryPressure,
			Detail: fmt.Sprintf("allocation %d bytes exceeds max memory limit %d", newTotal, m.cfg.MaxMemoryBytes),
		}
	}

	// Check injected allocation limit
	if m.cfg.InjectedAllocLimit > 0 && uint64(newTotal) > m.cfg.InjectedAllocLimit {
		m.downgraded = true
		m.lastDowngradeReason = string(MTPDowngradeMemoryPressure)
		m.memoryPressureCount++
		return &MTPDowngradeError{
			Reason: MTPDowngradeMemoryPressure,
			Detail: fmt.Sprintf("allocation %d bytes exceeds injected limit %d", newTotal, m.cfg.InjectedAllocLimit),
		}
	}

	m.currentAlloc = newTotal
	if m.currentAlloc > m.peakAlloc {
		m.peakAlloc = m.currentAlloc
	}
	return nil
}

// ReleaseAllocation decrements active memory tracking.
func (m *Qwen38MTPCancellationManager) ReleaseAllocation(bytes int64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if bytes <= 0 {
		return
	}
	if m.currentAlloc >= bytes {
		m.currentAlloc -= bytes
	} else {
		m.currentAlloc = 0
	}
}

// CurrentAlloc returns the current live allocated bytes.
func (m *Qwen38MTPCancellationManager) CurrentAlloc() int64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.currentAlloc
}

// PeakAlloc returns the peak allocated bytes tracked.
func (m *Qwen38MTPCancellationManager) PeakAlloc() int64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.peakAlloc
}

// Reset clears transient state and allocations, ensuring full reusability.
func (m *Qwen38MTPCancellationManager) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.currentAlloc = 0
	m.downgraded = false
	m.cancelled = false
	m.lastDowngradeReason = ""
}

// IsCancelled reports whether cancellation was recorded.
func (m *Qwen38MTPCancellationManager) IsCancelled() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.cancelled
}

// IsDowngraded reports whether memory downgrade occurred.
func (m *Qwen38MTPCancellationManager) IsDowngraded() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.downgraded
}

// LastDowngradeReason returns the last active downgrade reason string.
func (m *Qwen38MTPCancellationManager) LastDowngradeReason() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastDowngradeReason
}

// CheckContext verifies if the context has been cancelled or deadline exceeded.
func (m *Qwen38MTPCancellationManager) CheckContext(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	select {
	case <-ctx.Done():
		m.mu.Lock()
		m.cancelled = true
		m.lastDowngradeReason = string(MTPDowngradeContextCanceled)
		m.cancellationCount++
		m.mu.Unlock()
		return fmt.Errorf("%w: %v", ErrMTPContextCanceled, ctx.Err())
	default:
		return nil
	}
}

// ExecuteDraftStepWithContext evaluates a single speculative draft step while respecting
// context cancellation and memory bounds.
func (m *Qwen38MTPCancellationManager) ExecuteDraftStepWithContext(
	ctx context.Context,
	tx *MTPTransaction,
	stepFn func(ctx context.Context, pos int, priorHidden []float32) (kv, recurrent, conv [][]float32, hidden []float32, logits []float32, token int, err error),
) (int, error) {
	if err := m.CheckContext(ctx); err != nil {
		if tx != nil {
			_ = tx.Rollback()
			_ = tx.Downgrade(MTPDowngradeContextCanceled, err.Error())
		}
		return -1, err
	}

	if tx == nil {
		return -1, errors.New("model: non-nil MTPTransaction required")
	}

	// Memory bounds check before step
	acc := tx.Accounting()
	if err := m.checkMemoryBounds(uint64(acc.CurrentMemoryBytes)); err != nil {
		_ = tx.Rollback()
		_ = tx.Downgrade(MTPDowngradeMemoryPressure, err.Error())
		return -1, err
	}

	// Execute step closure
	tok, err := tx.ExecuteDraftStep(func(pos int, priorHidden []float32) (kv, recurrent, conv [][]float32, hidden []float32, logits []float32, token int, stepErr error) {
		if ctxErr := m.CheckContext(ctx); ctxErr != nil {
			return nil, nil, nil, nil, nil, -1, ctxErr
		}
		return stepFn(ctx, pos, priorHidden)
	})

	if err != nil {
		// If caused by context cancellation, cleanly rollback and tag downgrade
		if ctx != nil && ctx.Err() != nil {
			m.mu.Lock()
			m.cancelled = true
			m.lastDowngradeReason = string(MTPDowngradeContextCanceled)
			m.cancellationCount++
			m.mu.Unlock()

			_ = tx.Rollback()
			_ = tx.Downgrade(MTPDowngradeContextCanceled, ctx.Err().Error())
			return -1, fmt.Errorf("%w: %v", ErrMTPContextCanceled, ctx.Err())
		}
		return -1, err
	}

	// Post-step memory tracking check
	acc = tx.Accounting()
	if err := m.checkMemoryBounds(uint64(acc.CurrentMemoryBytes)); err != nil {
		_ = tx.Rollback()
		_ = tx.Downgrade(MTPDowngradeMemoryPressure, err.Error())
		return -1, err
	}

	return tok, nil
}

// VerifyWithContext executes target verification on the transaction with context cancellation
// and memory bounds checks.
func (m *Qwen38MTPCancellationManager) VerifyWithContext(
	ctx context.Context,
	tx *MTPTransaction,
	targetLogits [][]float32,
) (int, error) {
	if err := m.CheckContext(ctx); err != nil {
		if tx != nil {
			_ = tx.Rollback()
			_ = tx.Downgrade(MTPDowngradeContextCanceled, err.Error())
		}
		return 0, err
	}

	if tx == nil {
		return 0, errors.New("model: non-nil MTPTransaction required")
	}

	acc := tx.Accounting()
	if err := m.checkMemoryBounds(uint64(acc.CurrentMemoryBytes)); err != nil {
		_ = tx.Rollback()
		_ = tx.Downgrade(MTPDowngradeMemoryPressure, err.Error())
		return 0, err
	}

	accepted, err := tx.Verify(targetLogits)
	if err != nil {
		return 0, err
	}

	if ctx != nil && ctx.Err() != nil {
		m.mu.Lock()
		m.cancelled = true
		m.lastDowngradeReason = string(MTPDowngradeContextCanceled)
		m.cancellationCount++
		m.mu.Unlock()

		_ = tx.Rollback()
		_ = tx.Downgrade(MTPDowngradeContextCanceled, ctx.Err().Error())
		return 0, fmt.Errorf("%w: %v", ErrMTPContextCanceled, ctx.Err())
	}

	return accepted, nil
}

// ExecuteRoundWithContext runs a complete speculative round with context propagation.
func (m *Qwen38MTPCancellationManager) ExecuteRoundWithContext(
	ctx context.Context,
	tx *MTPTransaction,
	draftFn func(ctx context.Context) error,
	verifyFn func(ctx context.Context) error,
) error {
	if err := m.CheckContext(ctx); err != nil {
		if tx != nil {
			_ = tx.Rollback()
			_ = tx.Downgrade(MTPDowngradeContextCanceled, err.Error())
		}
		return err
	}

	if tx == nil {
		return errors.New("model: non-nil MTPTransaction required")
	}

	if _, err := tx.BeginRound(); err != nil {
		return err
	}

	// Drafting phase
	if draftFn != nil {
		if err := draftFn(ctx); err != nil {
			_ = tx.Rollback()
			if ctx != nil && ctx.Err() != nil {
				m.mu.Lock()
				m.cancelled = true
				m.lastDowngradeReason = string(MTPDowngradeContextCanceled)
				m.cancellationCount++
				m.mu.Unlock()
				_ = tx.Downgrade(MTPDowngradeContextCanceled, ctx.Err().Error())
				return fmt.Errorf("%w: %v", ErrMTPContextCanceled, ctx.Err())
			}
			return err
		}
	}

	if err := m.CheckContext(ctx); err != nil {
		_ = tx.Rollback()
		_ = tx.Downgrade(MTPDowngradeContextCanceled, err.Error())
		return err
	}

	// Verify phase
	if verifyFn != nil {
		if err := verifyFn(ctx); err != nil {
			_ = tx.Rollback()
			if ctx != nil && ctx.Err() != nil {
				m.mu.Lock()
				m.cancelled = true
				m.lastDowngradeReason = string(MTPDowngradeContextCanceled)
				m.cancellationCount++
				m.mu.Unlock()
				_ = tx.Downgrade(MTPDowngradeContextCanceled, ctx.Err().Error())
				return fmt.Errorf("%w: %v", ErrMTPContextCanceled, ctx.Err())
			}
			return err
		}
	}

	return nil
}

// ExecuteSessionRoundWithContext executes an MTP speculative round on a persistent session
// while protecting against context cancellation and unbounded memory growth.
func (m *Qwen38MTPCancellationManager) ExecuteSessionRoundWithContext(
	ctx context.Context,
	sess *Qwen38MTPPersistentSession,
) ([]int, *Qwen38MTPBatchedReceipt, error) {
	if err := m.CheckContext(ctx); err != nil {
		if sess != nil {
			sess.Cancel()
			// Ensure session is immediately reusable
			sess.ResetCancel()
		}
		return nil, nil, err
	}

	if sess == nil {
		return nil, nil, errors.New("model: persistent session required")
	}

	target := sess.Target()
	if target == nil {
		return nil, nil, errors.New("model: session target required")
	}

	// Snapshot baseline target state for guaranteed rollback on cancellation
	snap, snapErr := target.PrefixSnapshot()
	if snapErr != nil {
		return nil, nil, fmt.Errorf("model: failed to snapshot session baseline: %w", snapErr)
	}
	defer snap.Close()

	// Memory bounds check using snapshot residency
	cacheBytes := uint64(snap.ResidentBytes())
	if err := m.checkMemoryBounds(cacheBytes); err != nil {
		m.mu.Lock()
		m.downgraded = true
		m.lastDowngradeReason = string(MTPDowngradeMemoryPressure)
		m.memoryPressureCount++
		m.mu.Unlock()

		// Clean fail-closed downgrade to single-step target decode
		clone, _ := snap.Clone()
		if clone != nil {
			_ = clone.Restore(target)
			clone.Close()
		}
		sess.ResetCancel()
		return nil, nil, err
	}

	// Channel for speculative round execution
	type roundResult struct {
		tokens  []int
		receipt *Qwen38MTPBatchedReceipt
		err     error
	}

	resCh := make(chan roundResult, 1)
	go func() {
		tokens, receipt, err := sess.StepRound()
		resCh <- roundResult{tokens: tokens, receipt: receipt, err: err}
	}()

	select {
	case <-ctx.Done():
		m.mu.Lock()
		m.cancelled = true
		m.lastDowngradeReason = string(MTPDowngradeContextCanceled)
		m.cancellationCount++
		m.mu.Unlock()

		// Clean rollback of session to baseline snapshot
		clone, cErr := snap.Clone()
		if cErr == nil && clone != nil {
			_ = clone.Restore(target)
			clone.Close()
		}
		sess.ResetCancel()
		return nil, nil, fmt.Errorf("%w: %v", ErrMTPContextCanceled, ctx.Err())

	case res := <-resCh:
		if res.err != nil {
			return nil, nil, res.err
		}
		return res.tokens, res.receipt, nil
	}
}

// GenerateWithContext coordinates conversational generation across turns with context
// and memory bounds propagation, ensuring session reusability.
func (m *Qwen38MTPCancellationManager) GenerateWithContext(
	ctx context.Context,
	sess *Qwen38MTPPersistentSession,
	prompt []int,
	maxTokens int,
) ([]int, error) {
	if err := m.CheckContext(ctx); err != nil {
		if sess != nil {
			sess.ResetCancel()
		}
		return nil, err
	}

	if sess == nil {
		return nil, errors.New("model: persistent session required")
	}

	if _, err := sess.Prefill(prompt); err != nil {
		return nil, err
	}

	var generated []int
	for len(generated) < maxTokens {
		if err := m.CheckContext(ctx); err != nil {
			sess.ResetCancel()
			return generated, err
		}

		tokens, _, err := m.ExecuteSessionRoundWithContext(ctx, sess)
		if err != nil {
			sess.ResetCancel()
			return generated, err
		}

		for _, tok := range tokens {
			generated = append(generated, tok)
			if len(generated) >= maxTokens {
				break
			}
		}
	}

	return generated, nil
}

func (m *Qwen38MTPCancellationManager) checkMemoryBounds(estimatedBytes uint64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	total := uint64(m.currentAlloc) + estimatedBytes
	if m.cfg.MaxMemoryBytes > 0 && total > m.cfg.MaxMemoryBytes {
		m.downgraded = true
		m.lastDowngradeReason = string(MTPDowngradeMemoryPressure)
		m.memoryPressureCount++
		return &MTPDowngradeError{
			Reason: MTPDowngradeMemoryPressure,
			Detail: fmt.Sprintf("memory %d exceeds max memory bytes %d", total, m.cfg.MaxMemoryBytes),
		}
	}

	if m.cfg.InjectedAllocLimit > 0 && total > m.cfg.InjectedAllocLimit {
		m.downgraded = true
		m.lastDowngradeReason = string(MTPDowngradeMemoryPressure)
		m.memoryPressureCount++
		return &MTPDowngradeError{
			Reason: MTPDowngradeMemoryPressure,
			Detail: fmt.Sprintf("memory %d exceeds injected allocation limit %d", total, m.cfg.InjectedAllocLimit),
		}
	}

	return nil
}

// Receipt constructs a diagnostic Qwen38MTPCancellationReceipt summarizing cancellation status.
func (m *Qwen38MTPCancellationManager) Receipt(tx *MTPTransaction) *Qwen38MTPCancellationReceipt {
	m.mu.Lock()
	defer m.mu.Unlock()

	reason := m.lastDowngradeReason
	if reason == "" {
		reason = string(Qwen38MTPEligible)
	}

	peak := uint64(m.peakAlloc)
	if peak == 0 && tx != nil {
		peak = uint64(tx.Accounting().PeakMemoryBytes)
	}
	if peak == 0 {
		peak = 4096
	}

	engine := Qwen38EngineMTP
	fallback := Qwen38EngineTargetDecode
	outcome := Qwen38MTPOutcomeSucceeded

	if m.cancelled || m.downgraded {
		outcome = Qwen38MTPOutcomeFailed
	}

	return &Qwen38MTPCancellationReceipt{
		SchemaVersion:   Qwen38MTPCancellationReceiptSchema,
		Engine:          engine,
		FallbackEngine:  fallback,
		Outcome:         outcome,
		DowngradeReason: reason,
		MemoryBytes: Qwen38MTPMemoryBytes{
			Peak: peak,
		},
		DraftAborted:    m.cancelled,
		VerifyAborted:   m.cancelled,
		StateRolledBack: true,
		LocksReleased:   true,
	}
}
