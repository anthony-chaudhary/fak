package ggufload

import (
	"context"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Inkling2LoopResidentContext is the token capacity (prompt plus generated) of one
// resident device session (FAK_INKLING2_CTX, default 4096); the host path has no
// fixed bound.
var Inkling2LoopResidentContext = inkling2ResidentCtx

// Inkling2LoopSession is one request's decode state over a fresh KV cache.
// Prefill consumes the whole prompt and returns the greedy next token; Step
// consumes one token and returns the greedy next token. A resident session
// captures a CUDA graph bound to the OS thread that first steps it, so every
// call on one session must come from a single locked thread.
type Inkling2LoopSession interface {
	Prefill(prompt []int) (int, error)
	Step(token int) (int, error)
	Close() error
}

// NewInkling2LoopSession opens a session on the resident device forward when the
// backend and weights qualify, else on the host-driven forward.
func NewInkling2LoopSession(m *Inkling2LoopModel, be compute.Backend) (Inkling2LoopSession, error) {
	if inkling2ResidentEligible(m, be) {
		s, err := newInkling2Resident(m, be)
		if err != nil {
			return nil, err
		}
		return &inkling2ResidentSession{s: s}, nil
	}
	return newInkling2HostSession(m, be)
}

type inkling2HostSession struct {
	h       *model.Inkling2LoopHost
	metal   bool
	recycle func()
}

func newInkling2HostSession(m *Inkling2LoopModel, be compute.Backend) (*inkling2HostSession, error) {
	h, err := model.NewInkling2LoopHost(m.HP, m.Weights)
	if err != nil {
		return nil, err
	}
	_, metal := m.Weights.Batcher.(*inkling2MetalBatch)
	s := &inkling2HostSession{h: h, metal: metal, recycle: func() {}}
	if r, ok := be.(interface{ Recycle() }); ok {
		s.recycle = r.Recycle
	}
	return s, nil
}

func (s *inkling2HostSession) Prefill(prompt []int) (int, error) {
	if len(prompt) == 0 {
		return 0, fmt.Errorf("inkling2_loop: empty prompt")
	}
	var logits []float32
	var err error
	if s.metal {
		if logits, err = Inkling2LoopPrefill(context.Background(), s.h, prompt); err != nil {
			return 0, err
		}
		return inkling2Argmax(logits), nil
	}
	for i, t := range prompt {
		if logits, err = Inkling2LoopPrefillToken(s.h, t, i == len(prompt)-1); err != nil {
			return 0, err
		}
		s.recycle()
	}
	return inkling2Argmax(logits), nil
}

func (s *inkling2HostSession) Step(token int) (int, error) {
	logits, err := Inkling2LoopStep(s.h, token)
	if err != nil {
		return 0, err
	}
	s.recycle()
	return inkling2Argmax(logits), nil
}

func (s *inkling2HostSession) Close() error { return nil }

// inkling2Argmax keeps the lowest index on ties, matching the device argmax.
func inkling2Argmax(logits []float32) int {
	best := 0
	for i, v := range logits {
		if v > logits[best] {
			best = i
		}
	}
	return best
}

type inkling2ResidentSession struct{ s *inkling2Resident }

func (r *inkling2ResidentSession) Prefill(prompt []int) (int, error) {
	if len(prompt) == 0 {
		return 0, fmt.Errorf("inkling2_loop: empty prompt")
	}
	if r.s.batched {
		return r.s.prefill(prompt)
	}
	next := 0
	for _, t := range prompt {
		var err error
		if next, err = r.s.step(t); err != nil {
			return 0, err
		}
	}
	return next, nil
}

func (r *inkling2ResidentSession) Step(token int) (int, error) { return r.s.step(token) }

func (r *inkling2ResidentSession) Close() error {
	r.s.close()
	return nil
}
