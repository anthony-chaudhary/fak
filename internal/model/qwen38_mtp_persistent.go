package model

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// Typed errors for persistent Qwen3.8 MTP sessions.
var (
	ErrQwen38MTPCancelled = errors.New("model: qwen3.8 persistent MTP session cancelled")
	ErrQwen38MTPClosed    = errors.New("model: qwen3.8 persistent MTP session closed")
)

// Qwen38MTPPersistentConfig configures a persistent native MTP session across multiple turns.
type Qwen38MTPPersistentConfig struct {
	DraftDepth             int                 `json:"draft_depth"`
	Greedy                 bool                `json:"greedy"`
	Backend                Qwen38MTPBackend    `json:"backend"`
	AllowPersistentSession bool                `json:"allow_persistent_session"`
	OperatorEnabled        bool                `json:"operator_enabled"`
	MemoryHeadroomOK       bool                `json:"memory_headroom_ok"`
	Admission              *Qwen38MTPAdmission `json:"admission,omitempty"`
	PromptCache            *MTPPromptCache     `json:"-"`
	DisableMTP             bool                `json:"disable_mtp"`
}

// DefaultQwen38MTPPersistentConfig returns the production defaults for persistent MTP.
func DefaultQwen38MTPPersistentConfig() Qwen38MTPPersistentConfig {
	return Qwen38MTPPersistentConfig{
		DraftDepth:             Qwen35MTPMaxDraftDepth,
		Greedy:                 true,
		Backend:                Qwen38MTPBackendMetal,
		AllowPersistentSession: true,
		OperatorEnabled:        true,
		MemoryHeadroomOK:       true,
	}
}

// Qwen38MTPPersistentSession wraps persistent native Qwen3.8 decode sessions with MTP acceleration,
// preserving KV cache and linear attention recurrent state across multiple conversational turns
// without recreating verifier sessions per round.
type Qwen38MTPPersistentSession struct {
	mu            sync.Mutex
	target        *Session
	verifier      *Qwen38BatchedVerifier
	drafter       *Qwen35MTPDraftSession
	promptCache   *MTPPromptCache
	config        Qwen38MTPPersistentConfig
	eligibility   Qwen38MTPEligibility
	committed     []int
	lastLogits    []float32
	turnCount     int
	totalDrafted  int
	totalAccepted int
	totalRejected int
	cancelled     bool
	closed        bool
}

// NewQwen38MTPPersistentSession initializes a persistent session wrapping a target Qwen3.8 Session.
func NewQwen38MTPPersistentSession(target *Session, cfg ...Qwen38MTPPersistentConfig) (*Qwen38MTPPersistentSession, error) {
	if target == nil || target.M == nil {
		return nil, errors.New("model: valid target session is required for persistent MTP session")
	}

	c := DefaultQwen38MTPPersistentConfig()
	if len(cfg) > 0 {
		c = cfg[0]
	}
	if c.DraftDepth <= 0 {
		c.DraftDepth = Qwen35MTPMaxDraftDepth
	}

	target.captureTargetHidden = true

	// Check MTP head artifacts on model.
	mode, _ := target.M.Qwen35MTPMode(c.DisableMTP)
	hasArtifact := mode.Eligible

	// Evaluate eligibility explicitly allowing persistent sessions.
	eligibilityInput := Qwen38MTPEligibilityInput{
		Qwen38MTPArtifact:      hasArtifact,
		MTPBackendReady:        true,
		Backend:                c.Backend,
		Model:                  target.M,
		F32:                    target.M.Cfg.NumMTPLayers() > 0,
		Greedy:                 c.Greedy,
		Depth:                  c.DraftDepth,
		FreshSession:           false, // Persistent across turns
		PersistentSession:      true,
		AllowPersistentSession: true,
		MemoryHeadroomOK:       c.MemoryHeadroomOK,
		Admission:              c.Admission,
		OperatorEnabled:        c.OperatorEnabled && !c.DisableMTP,
	}

	eligibility := EvaluateQwen38MTPEligibility(eligibilityInput)

	verifier, err := NewQwen38BatchedVerifier(target)
	if err != nil {
		return nil, fmt.Errorf("model: failed to create batched verifier: %w", err)
	}

	var drafter *Qwen35MTPDraftSession
	if eligibility.Eligible {
		d, dErr := NewQwen35MTPDraftSession(target, c.DraftDepth)
		if dErr == nil {
			drafter = d
		} else {
			// Fail-closed downgrade to native target decode without foreign runtimes.
			eligibility.Eligible = false
			eligibility.Engine = Qwen38EngineTargetDecode
			eligibility.DowngradeReason = Qwen38MTPAttemptFailed
		}
	}

	s := &Qwen38MTPPersistentSession{
		target:      target,
		verifier:    verifier,
		drafter:     drafter,
		promptCache: c.PromptCache,
		config:      c,
		eligibility: eligibility,
	}

	return s, nil
}

// Prefill loads or extends prompt tokens into the persistent session. It leverages prefix reuse
// across conversational turns and multi-turn prompt caching without recomputing shared prefixes.
func (s *Qwen38MTPPersistentSession) Prefill(prompt []int) ([]float32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil, ErrQwen38MTPClosed
	}
	if s.cancelled {
		return nil, ErrQwen38MTPCancelled
	}
	if len(prompt) == 0 {
		return nil, errors.New("model: empty prompt provided to persistent session")
	}

	s.turnCount++

	// Case 1: Active continuation / context extension within this session.
	if len(s.committed) > 0 {
		shared := commonPrefixTokens(s.committed, prompt)

		// Exact extension: committed tokens are a prefix of prompt.
		if shared == len(s.committed) && shared < len(prompt) {
			suffix := prompt[shared:]
			for _, tok := range suffix {
				s.lastLogits = s.target.Step(tok)
			}
			s.committed = append([]int(nil), prompt...)
			s.updatePromptCache(prompt)
			return append([]float32(nil), s.lastLogits...), nil
		}

		// Exact match: prompt is identical to currently committed prefix.
		if shared == len(prompt) && shared == len(s.committed) {
			return append([]float32(nil), s.lastLogits...), nil
		}

		// Divergence from active committed history.
		// Roll back cache to shared prefix if possible.
		if shared > 0 && s.target.Cache.Len() > shared {
			_, _ = s.target.Cache.TryEvict(shared, len(s.committed)-shared)
			s.target.targetHiddenMu.Lock()
			if len(s.target.targetHidden) > shared {
				s.target.targetHidden = s.target.targetHidden[:shared]
				s.target.targetHiddenTokens = s.target.targetHiddenTokens[:shared]
			}
			s.target.targetHiddenMu.Unlock()

			suffix := prompt[shared:]
			for _, tok := range suffix {
				s.lastLogits = s.target.Step(tok)
			}
			s.committed = append([]int(nil), prompt...)
			s.updatePromptCache(prompt)
			return append([]float32(nil), s.lastLogits...), nil
		}
	}

	// Case 2: Fresh turn or cache match from external prompt cache.
	if s.promptCache != nil && s.promptCache.Len() > 0 {
		entry, matchLen := s.promptCache.MatchPrefix(prompt)
		if entry != nil && matchLen > 0 && entry.Snapshot != nil {
			clone, err := entry.Snapshot.Clone()
			if err == nil {
				if rErr := clone.Restore(s.target); rErr == nil {
					s.committed = append([]int(nil), entry.Prompt[:matchLen]...)
					suffix := prompt[matchLen:]
					for _, tok := range suffix {
						s.lastLogits = s.target.Step(tok)
					}
					s.committed = append([]int(nil), prompt...)
					s.updatePromptCache(prompt)
					return append([]float32(nil), s.lastLogits...), nil
				}
				clone.Close()
			}
		}
	}

	// Case 3: Cold prefill on clean session.
	s.target.Cache = NewKVCache(s.target.M.Cfg)
	s.target.captureTargetHidden = true
	s.lastLogits = s.target.Prefill(prompt)
	s.committed = append([]int(nil), prompt...)
	s.updatePromptCache(prompt)

	return append([]float32(nil), s.lastLogits...), nil
}

// StepRound executes a single MTP speculative block round using batched target verification,
// or one ordinary target decode step if downgraded.
func (s *Qwen38MTPPersistentSession) StepRound() ([]int, *Qwen38MTPBatchedReceipt, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil, nil, ErrQwen38MTPClosed
	}
	if s.cancelled {
		return nil, nil, ErrQwen38MTPCancelled
	}
	if len(s.lastLogits) == 0 {
		return nil, nil, errors.New("model: persistent session must be prefilled before decoding")
	}

	// Target-only decode path if MTP is ineligible or disabled.
	if !s.eligibility.Eligible || s.drafter == nil {
		next := argmaxF32(s.lastLogits)
		s.lastLogits = s.target.Step(next)
		s.committed = append(s.committed, next)
		receipt := &Qwen38MTPBatchedReceipt{
			SchemaVersion:     Qwen38MTPBatchedReceiptSchema,
			Engine:            Qwen38EngineMTP,
			FallbackEngine:    Qwen38EngineTargetDecode,
			OneOperation:      false,
			TargetDecodeSteps: 1,
			AcceptedTokens:    []int{next},
			AcceptedCount:     1,
			DowngradeReason:   s.eligibility.DowngradeReason,
		}
		return []int{next}, receipt, nil
	}

	// MTP Accelerated Path:
	draftStart := time.Now()
	draftTokens := s.drafter.Propose(s.committed)
	draftDuration := time.Since(draftStart)

	if s.drafter.Err() != nil || len(draftTokens) == 0 {
		// Drafter failure: cleanly step target once
		next := argmaxF32(s.lastLogits)
		s.lastLogits = s.target.Step(next)
		s.committed = append(s.committed, next)
		receipt := &Qwen38MTPBatchedReceipt{
			SchemaVersion:     Qwen38MTPBatchedReceiptSchema,
			Engine:            Qwen38EngineMTP,
			FallbackEngine:    Qwen38EngineTargetDecode,
			OneOperation:      false,
			TargetDecodeSteps: 1,
			AcceptedTokens:    []int{next},
			AcceptedCount:     1,
			DowngradeReason:   Qwen38MTPAttemptFailed,
		}
		return []int{next}, receipt, nil
	}

	receipt, err := s.verifier.VerifyBlockWithDraftTime(draftTokens, s.lastLogits, draftDuration)
	if err != nil {
		// Verification failure: fallback to native target decode
		next := argmaxF32(s.lastLogits)
		s.lastLogits = s.target.Step(next)
		s.committed = append(s.committed, next)
		return []int{next}, nil, nil
	}

	s.totalDrafted += len(draftTokens)
	s.totalAccepted += receipt.AcceptedCount
	s.totalRejected += receipt.RejectedCount

	var emitted []int
	if receipt.AcceptedCount > 0 {
		emitted = append(emitted, receipt.AcceptedTokens...)
		s.committed = append(s.committed, receipt.AcceptedTokens...)
		if len(receipt.AcceptedLogits) > 0 {
			s.lastLogits = receipt.AcceptedLogits
		}
	} else {
		// Total rejection: rollback was executed by verifier. Advance 1 step via target greedy.
		next := argmaxF32(s.lastLogits)
		s.lastLogits = s.target.Step(next)
		s.committed = append(s.committed, next)
		emitted = []int{next}
	}

	return emitted, receipt, nil
}

// Generate generates up to maxTokens for the given prompt in the persistent session.
func (s *Qwen38MTPPersistentSession) Generate(prompt []int, maxTokens int) ([]int, error) {
	if _, err := s.Prefill(prompt); err != nil {
		return nil, err
	}

	var generated []int
	for len(generated) < maxTokens {
		if s.IsCancelled() {
			return generated, ErrQwen38MTPCancelled
		}
		tokens, _, err := s.StepRound()
		if err != nil {
			return generated, err
		}
		for _, t := range tokens {
			generated = append(generated, t)
			if len(generated) >= maxTokens {
				break
			}
		}
	}
	return generated, nil
}

// Reset resets the conversational state (KV cache, recurrent state, committed tokens),
// preserving the verifier and persistent session without recreating them.
func (s *Qwen38MTPPersistentSession) Reset() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return ErrQwen38MTPClosed
	}

	s.cancelled = false
	s.committed = nil
	s.lastLogits = nil
	s.turnCount = 0

	// Clear target KV cache and recurrent state
	s.target.Cache = NewKVCache(s.target.M.Cfg)
	s.target.targetHiddenMu.Lock()
	s.target.targetHidden = nil
	s.target.targetHiddenTokens = nil
	s.target.targetHiddenMu.Unlock()

	// Reset draft session cleanly
	if s.drafter != nil {
		s.drafter.Close()
		s.drafter = nil
	}
	if s.eligibility.Eligible {
		d, err := NewQwen35MTPDraftSession(s.target, s.config.DraftDepth)
		if err == nil {
			s.drafter = d
		}
	}

	return nil
}

// Cancel marks the persistent session as cancelled.
func (s *Qwen38MTPPersistentSession) Cancel() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cancelled = true
}

// IsCancelled reports whether the session has been cancelled.
func (s *Qwen38MTPPersistentSession) IsCancelled() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cancelled
}

// ResetCancel clears the cancellation status on the session.
func (s *Qwen38MTPPersistentSession) ResetCancel() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cancelled = false
}

// Target returns the underlying native target Session.
func (s *Qwen38MTPPersistentSession) Target() *Session {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.target
}

// CommittedTokens returns a copy of the currently committed token sequence.
func (s *Qwen38MTPPersistentSession) CommittedTokens() []int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]int(nil), s.committed...)
}

// TurnCount returns the number of conversational turns processed by this session.
func (s *Qwen38MTPPersistentSession) TurnCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.turnCount
}

// Eligibility returns the evaluated MTP eligibility record for this persistent session.
func (s *Qwen38MTPPersistentSession) Eligibility() Qwen38MTPEligibility {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.eligibility
}

// Stats returns cumulative drafted, accepted, and rejected token counts.
func (s *Qwen38MTPPersistentSession) Stats() (totalDrafted, totalAccepted, totalRejected int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.totalDrafted, s.totalAccepted, s.totalRejected
}

// Close closes the persistent session and releases any associated draft caches.
func (s *Qwen38MTPPersistentSession) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}
	s.closed = true
	if s.drafter != nil {
		s.drafter.Close()
		s.drafter = nil
	}
	s.committed = nil
	s.lastLogits = nil
	return nil
}

func (s *Qwen38MTPPersistentSession) updatePromptCache(prompt []int) {
	if s.promptCache == nil {
		return
	}
	if snap, err := s.target.PrefixSnapshot(); err == nil {
		_ = s.promptCache.Put(prompt, snap)
		snap.Close()
	}
}
