package model

import (
	"errors"
	"fmt"
	"sync"
)

// ErrQwen35GDNSequenceUnsupported reports that an optional sequence provider
// cannot execute the requested preprojected recurrence. Callers must not retry
// the operation through a host fallback after the provider has accepted it.
var ErrQwen35GDNSequenceUnsupported = errors.New("model: Qwen GDN sequence capability unsupported")

// Qwen35GDNSequenceConfig identifies the complete recurrence state a session
// asks a provider to own. State is allocated only after Preflight succeeds.
type Qwen35GDNSequenceConfig struct {
	Layers    int
	StateSize int
}

// Qwen35GDNSequenceState is one provider-owned, per-layer recurrence handle.
// The model package treats Value as opaque and preserves its identity.
type Qwen35GDNSequenceState struct {
	Layer int
	Value any
}

// Qwen35GDNPreprojectedSequence is the provider-neutral input to one accepted
// sequence operation. Projection remains outside this contract.
type Qwen35GDNPreprojectedSequence struct {
	Tokens int
	Values []float32
}

// Qwen35GDNSequenceCapability is optional and independent of compute.Backend.
// Preflight is the capability admission boundary. Once RunSequence is called,
// its result is final: the session does not attempt a host fallback.
type Qwen35GDNSequenceCapability interface {
	PreflightQwen35GDNSequence(Qwen35GDNSequenceConfig) error
	NewQwen35GDNSequenceState(Qwen35GDNSequenceConfig) ([]Qwen35GDNSequenceState, error)
	RunQwen35GDNPreprojectedSequence([]Qwen35GDNSequenceState, Qwen35GDNPreprojectedSequence) error
	FreeQwen35GDNSequenceState([]Qwen35GDNSequenceState)
}

type qwen35GDNSequenceOwner struct {
	mu     sync.Mutex
	cap    Qwen35GDNSequenceCapability
	states []Qwen35GDNSequenceState
	closed bool
}

// InitQwen35GDNSequence admits and initializes an optional sequence provider.
// candidate is deliberately independent of Session.Backend so backend-nil
// production sessions can own the auxiliary state without widening the HAL.
func (s *Session) InitQwen35GDNSequence(candidate any, cfg Qwen35GDNSequenceConfig) error {
	if s == nil {
		return errors.New("model: nil session")
	}
	capability, ok := candidate.(Qwen35GDNSequenceCapability)
	if !ok || capability == nil {
		return ErrQwen35GDNSequenceUnsupported
	}
	if cfg.Layers <= 0 || cfg.StateSize <= 0 {
		return fmt.Errorf("%w: invalid state geometry", ErrQwen35GDNSequenceUnsupported)
	}
	if err := capability.PreflightQwen35GDNSequence(cfg); err != nil {
		return fmt.Errorf("%w: %v", ErrQwen35GDNSequenceUnsupported, err)
	}
	states, err := capability.NewQwen35GDNSequenceState(cfg)
	if err != nil {
		return err
	}
	if len(states) != cfg.Layers {
		capability.FreeQwen35GDNSequenceState(states)
		return fmt.Errorf("model: Qwen GDN sequence provider allocated %d states for %d layers", len(states), cfg.Layers)
	}
	owner := &qwen35GDNSequenceOwner{cap: capability, states: states}
	s.qwen35GDNSequenceMu.Lock()
	if s.qwen35GDNSequence != nil {
		s.qwen35GDNSequenceMu.Unlock()
		owner.close()
		return errors.New("model: Qwen GDN sequence state already initialized")
	}
	s.qwen35GDNSequence = owner
	s.qwen35GDNSequenceMu.Unlock()
	return nil
}

// RunQwen35GDNPreprojectedSequence submits one accepted operation. Any submit
// failure tears down the owned state exactly once and is returned unchanged.
func (s *Session) RunQwen35GDNPreprojectedSequence(seq Qwen35GDNPreprojectedSequence) error {
	if s == nil {
		return ErrQwen35GDNSequenceUnsupported
	}
	s.qwen35GDNSequenceMu.Lock()
	owner := s.qwen35GDNSequence
	s.qwen35GDNSequenceMu.Unlock()
	if owner == nil {
		return ErrQwen35GDNSequenceUnsupported
	}
	owner.mu.Lock()
	if owner.closed {
		owner.mu.Unlock()
		return ErrQwen35GDNSequenceUnsupported
	}
	err := owner.cap.RunQwen35GDNPreprojectedSequence(owner.states, seq)
	owner.mu.Unlock()
	if err != nil {
		s.closeQwen35GDNSequence()
	}
	return err
}

// ResetQwen35GDNSequence releases optional recurrence state. A later use must
// explicitly preflight and initialize a fresh owner.
func (s *Session) ResetQwen35GDNSequence() { s.closeQwen35GDNSequence() }

func (s *Session) closeQwen35GDNSequence() {
	if s == nil {
		return
	}
	s.qwen35GDNSequenceMu.Lock()
	owner := s.qwen35GDNSequence
	s.qwen35GDNSequence = nil
	s.qwen35GDNSequenceMu.Unlock()
	if owner != nil {
		owner.close()
	}
}

func (o *qwen35GDNSequenceOwner) close() {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.closed {
		return
	}
	o.closed = true
	o.cap.FreeQwen35GDNSequenceState(o.states)
	o.states = nil
}
