package microagent

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/idempotency"
)

var ErrEffectConflict = errors.New("microagent: effect resource busy")
var ErrAuthorityRefused = errors.New("microagent: effect capability denied")

type EffectIntent struct{ ContextID, Capability, Resource, Operation, IdempotencyToken string }
type EffectOutcome struct {
	Result             string
	Replayed, Verified bool
}
type EffectReadback func(context.Context, EffectIntent, string) error

// EffectCoordinator separates effect admission from model scheduling. Resource
// ownership is nonblocking, capability authority is explicit, landed effects
// dedupe durably, and an independently supplied readback gates the result.
type EffectCoordinator struct {
	store   *idempotency.Store
	mu      sync.Mutex
	held    map[string]string
	jmu     sync.Mutex
	journal []EffectIntent
}

func NewEffectCoordinator(s *idempotency.Store) *EffectCoordinator {
	return &EffectCoordinator{store: s, held: map[string]string{}}
}
func (e *EffectCoordinator) Run(ctx context.Context, in EffectIntent, allowed []string, apply func() (string, error), read EffectReadback) (EffectOutcome, error) {
	if e == nil || e.store == nil || apply == nil || read == nil {
		return EffectOutcome{}, errors.New("microagent: incomplete effect coordinator")
	}
	if !contains(allowed, in.Capability) {
		return EffectOutcome{}, ErrAuthorityRefused
	}
	if in.Resource == "" || in.Operation == "" || in.IdempotencyToken == "" {
		return EffectOutcome{}, errors.New("microagent: incomplete effect intent")
	}
	e.mu.Lock()
	if owner, ok := e.held[in.Resource]; ok && owner != in.ContextID {
		e.mu.Unlock()
		return EffectOutcome{}, fmt.Errorf("%w: %s held by %s", ErrEffectConflict, in.Resource, owner)
	}
	e.held[in.Resource] = in.ContextID
	e.mu.Unlock()
	defer func() { e.mu.Lock(); delete(e.held, in.Resource); e.mu.Unlock() }()
	e.jmu.Lock()
	e.journal = append(e.journal, in)
	e.jmu.Unlock()
	res, replayed, err := e.store.Do(idempotency.Key(in.Operation, in.IdempotencyToken), in.Operation, apply)
	if err != nil {
		return EffectOutcome{}, err
	}
	if err = read(ctx, in, res); err != nil {
		return EffectOutcome{Result: res, Replayed: replayed}, &VerificationError{Evidence: err}
	}
	return EffectOutcome{Result: res, Replayed: replayed, Verified: true}, nil
}
func (e *EffectCoordinator) Journal() []EffectIntent {
	e.jmu.Lock()
	defer e.jmu.Unlock()
	return append([]EffectIntent(nil), e.journal...)
}
func contains(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}
