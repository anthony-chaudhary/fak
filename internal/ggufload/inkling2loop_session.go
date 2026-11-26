package ggufload

import (
	"context"
	"fmt"
	"math"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Inkling2LoopResidentContext is the token capacity (prompt plus generated) of one
// resident device session (FAK_INKLING2_CTX, default 4096), already clamped to the
// bound of the build's resident backend; the host path has no fixed bound.
var Inkling2LoopResidentContext = inkling2ResidentCtx

// Inkling2LoopSession is one request's decode state over a fresh KV cache.
// Prefill consumes the whole prompt and returns the greedy next token; Step
// consumes one token and returns the greedy next token. A resident session
// captures a CUDA graph bound to the OS thread that first steps it, so every
// call on one session must come from a single locked thread. A Prefill that ctx
// ends leaves the session partly consumed; the caller must drop it, not reuse it.
type Inkling2LoopSession interface {
	Prefill(ctx context.Context, prompt []int) (int, error)
	Step(token int) (int, error)
	Close() error
}

// Inkling2LoopRewinder is implemented by sessions that can keep their cache across
// requests. Prefill may be called repeatedly to append; Pos advances by the tokens
// consumed. Restore returns to exactly the state saved by the last Checkpoint and
// keeps that checkpoint as long as at most horizon tokens ran since it; Reset returns to position 0 with fresh history, keeps
// device buffers, and drops the checkpoint.
type Inkling2LoopRewinder interface {
	Pos() int
	Checkpoint(horizon int) error
	Restore() error
	Reset() error
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
	ck      *model.Inkling2LoopHostState
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

func (s *inkling2HostSession) Prefill(ctx context.Context, prompt []int) (int, error) {
	if len(prompt) == 0 {
		return 0, fmt.Errorf("inkling2_loop: empty prompt")
	}
	var logits []float32
	var err error
	if s.metal {
		if logits, err = Inkling2LoopPrefill(ctx, s.h, prompt); err != nil {
			return 0, err
		}
		return inkling2Argmax(logits), nil
	}
	for i, t := range prompt {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
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

type inkling2ResidentSession struct {
	s  *inkling2Resident
	ck *ink2Rewind
}

// inkling2GraphOwner is the resident session whose captures the device's
// process-wide graph execs hold: a capture on one session replaces another's, and
// closing any session destroys them all.
var inkling2GraphOwner atomic.Pointer[inkling2Resident]

// own makes r the graph owner before it runs. A session taking ownership from
// another recaptures after one eager token instead of replaying an exec bound to
// the other session's buffers.
func (r *inkling2ResidentSession) own() {
	if inkling2GraphOwner.Swap(r.s) == r.s {
		return
	}
	r.s.dropCaptures()
}

// HeldBytes is the device memory the session owns: its cache, arenas, uploaded
// small weights and checkpoint buffers, not the model weights it shares.
func (r *inkling2ResidentSession) HeldBytes() int64 {
	n := inkling2TensorBytes(r.s.owned)
	if r.ck != nil {
		n += inkling2TensorBytes(r.ck.base) + inkling2TensorBytes(r.ck.grown)
	}
	return n
}

func inkling2TensorBytes(ts []compute.Tensor) int64 {
	var n int64
	for _, t := range ts {
		n += int64(t.Numel()) * int64(t.Dtype.Bytes())
	}
	return n
}

func (r *inkling2ResidentSession) Prefill(ctx context.Context, prompt []int) (int, error) {
	if len(prompt) == 0 {
		return 0, fmt.Errorf("inkling2_loop: empty prompt")
	}
	r.own()
	r.s.settle()
	if r.s.batched {
		next, err := r.s.prefill(ctx, prompt)
		if err == nil {
			r.s.pipeExpect(next)
		}
		return next, err
	}
	next := 0
	for _, t := range prompt {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		var err error
		if next, err = r.s.step(t); err != nil {
			return 0, err
		}
	}
	return next, nil
}

func (r *inkling2ResidentSession) Step(token int) (int, error) {
	r.own()
	return r.s.step(token)
}

func (r *inkling2ResidentSession) Close() error {
	if r.ck != nil {
		r.freeGroup(&r.ck.grown)
		r.freeGroup(&r.ck.base)
		r.ck = nil
	}
	r.s.close()
	inkling2GraphOwner.Store(nil)
	return nil
}

func (r *inkling2ResidentSession) Pos() int { return r.s.pos }

// checkpointable is rewindable plus the global-cache copies a checkpoint makes, which
// read f16 caches in float32 element strides and so refuse FAK_INKLING2_KV=f16.
func (s *inkling2Resident) checkpointable() error {
	if s.kvHalf {
		return fmt.Errorf("inkling2 resident: f16 global KV (FAK_INKLING2_KV=f16) cannot be checkpointed")
	}
	return s.rewindable()
}

func (s *inkling2Resident) rewindable() error {
	// every rewind entry reads or replaces the state a speculative forward changed
	s.settle()
	if s.draft != nil || s.specMax > 0 {
		return fmt.Errorf("inkling2 resident: speculative state cannot be checkpointed")
	}
	return s.err
}

func (s *inkling2Resident) histFloats() int {
	kvDim := s.hp.KVHeads * s.hp.HeadDim
	return (2*kvDim + 2*s.hp.Hidden) * (s.hp.ConvKernel - 1)
}

type ink2DeviceOnlyAllocator interface {
	Ink2AllocDeviceOnly(n int) (compute.Tensor, error)
}

// allocInto allocates n floats without touching the session's sticky error, and
// without a managed-memory fallback, so a checkpoint that does not fit leaves the
// session as it was.
func (r *inkling2ResidentSession) allocInto(group *[]compute.Tensor, n int) (ink2Ref, error) {
	var t compute.Tensor
	var err error
	if d, ok := r.s.dev.(ink2DeviceOnlyAllocator); ok {
		t, err = d.Ink2AllocDeviceOnly(n)
	} else {
		t, err = r.s.dev.Ink2Alloc(n)
	}
	if err != nil {
		return ink2Ref{}, err
	}
	*group = append(*group, t)
	return compute.Ink2At(t, 0), nil
}

func (r *inkling2ResidentSession) freeGroup(group *[]compute.Tensor) {
	for _, t := range *group {
		r.s.be.Free(t)
	}
	*group = nil
}

func (r *inkling2ResidentSession) negZeroFill(n int) (ink2Ref, []compute.Tensor, error) {
	var g []compute.Tensor
	ref, err := r.allocInto(&g, n)
	if err == nil {
		nz := make([]float32, n)
		for i := range nz {
			nz[i] = float32(math.Copysign(0, -1))
		}
		err = r.s.dev.Ink2Write(ref, nz)
	}
	if err != nil {
		r.freeGroup(&g)
	}
	return ref, g, err
}

// baseBuffers allocates the conv history copies and fills once per session.
func (r *inkling2ResidentSession) baseBuffers() (*ink2Rewind, error) {
	if r.ck != nil {
		return r.ck, nil
	}
	s := r.s
	hf := s.histFloats()
	ck := &ink2Rewind{hist: make([]ink2Ref, len(s.hist))}
	var err error
	for slot := 0; slot < len(s.hist) && err == nil; slot++ {
		ck.hist[slot], err = r.allocInto(&ck.base, hf)
	}
	if err == nil {
		ck.zero, err = r.allocInto(&ck.base, hf)
	}
	if err == nil {
		var g []compute.Tensor
		ck.histFill, g, err = r.negZeroFill(hf)
		ck.base = append(ck.base, g...)
	}
	if err != nil {
		r.freeGroup(&ck.base)
		return nil, fmt.Errorf("%w: %v", ErrInkling2CheckpointUnavailable, err)
	}
	r.ck = ck
	return ck, nil
}

// growRows makes every local slot's side buffers hold at least rows cache rows.
func (r *inkling2ResidentSession) growRows(ck *ink2Rewind, rows int) error {
	if rows <= ck.rows {
		return nil
	}
	s := r.s
	kvDim := s.hp.KVHeads * s.hp.HeadDim
	r.freeGroup(&ck.grown)
	ck.rows, ck.k, ck.v = 0, make([]ink2Ref, len(s.hist)), make([]ink2Ref, len(s.hist))
	var err error
	for slot := 0; slot < len(s.hist) && err == nil; slot++ {
		if !s.localAt(slot) {
			continue
		}
		if ck.k[slot], err = r.allocInto(&ck.grown, rows*kvDim); err == nil {
			ck.v[slot], err = r.allocInto(&ck.grown, rows*kvDim)
		}
	}
	if err == nil {
		var g []compute.Tensor
		ck.rowFill, g, err = r.negZeroFill(rows * kvDim)
		ck.grown = append(ck.grown, g...)
	}
	if err != nil {
		r.freeGroup(&ck.grown)
		return fmt.Errorf("%w: %v", ErrInkling2CheckpointUnavailable, err)
	}
	ck.rows = rows
	return nil
}

func (r *inkling2ResidentSession) copyDev(dst, src, fill ink2Ref, n int) {
	if n > 0 && r.s.err == nil {
		r.s.fail(r.s.dev.Ink2Add(src, fill, dst, n))
	}
}

// Checkpoint saves the state at the current position for a Restore after at most
// horizon more tokens.
func (r *inkling2ResidentSession) Checkpoint(horizon int) error {
	s := r.s
	if err := s.checkpointable(); err != nil {
		return err
	}
	ck, err := r.baseBuffers()
	if err != nil {
		return err
	}
	ck.valid = false
	ck.pos, ck.horizon = s.pos, max(horizon, 1)
	ck.lo, ck.hi = 0, 0
	for slot := range s.hist {
		if !s.localAt(slot) {
			continue
		}
		lo, hi := inkling2SavedSpan(s.pos, s.hp.SlidingWindow, s.caps[slot], ck.horizon)
		if hi == lo {
			continue
		}
		if ck.hi == ck.lo {
			ck.lo, ck.hi = lo, hi
		} else {
			ck.lo, ck.hi = min(ck.lo, lo), max(ck.hi, hi)
		}
	}
	if err := r.growRows(ck, ck.hi-ck.lo); err != nil {
		return err
	}
	kvDim := s.hp.KVHeads * s.hp.HeadDim
	hf := s.histFloats()
	for slot := range s.hist {
		r.copyDev(ck.hist[slot], s.hist[slot], ck.histFill, hf)
		if !s.localAt(slot) {
			continue
		}
		for _, sp := range inkling2RingSpans(ck.lo, ck.hi, s.caps[slot]) {
			r.copyDev(at(ck.k[slot], sp.off*kvDim), at(s.kc[slot], sp.ring*kvDim), ck.rowFill, sp.n*kvDim)
			r.copyDev(at(ck.v[slot], sp.off*kvDim), at(s.vc[slot], sp.ring*kvDim), ck.rowFill, sp.n*kvDim)
		}
	}
	ck.valid = s.err == nil
	return s.err
}

func (r *inkling2ResidentSession) Restore() error {
	s := r.s
	if err := s.checkpointable(); err != nil {
		return err
	}
	ck := r.ck
	if ck == nil || !ck.valid || s.pos-ck.pos > ck.horizon {
		if ck != nil {
			ck.valid = false
		}
		return ErrInkling2NoCheckpoint
	}
	kvDim := s.hp.KVHeads * s.hp.HeadDim
	hf := s.histFloats()
	for slot := range s.hist {
		r.copyDev(s.hist[slot], ck.hist[slot], ck.histFill, hf)
		if !s.localAt(slot) {
			continue
		}
		for _, sp := range inkling2RingSpans(ck.lo, ck.hi, s.caps[slot]) {
			r.copyDev(at(s.kc[slot], sp.ring*kvDim), at(ck.k[slot], sp.off*kvDim), ck.rowFill, sp.n*kvDim)
			r.copyDev(at(s.vc[slot], sp.ring*kvDim), at(ck.v[slot], sp.off*kvDim), ck.rowFill, sp.n*kvDim)
		}
	}
	if s.err != nil {
		return s.err
	}
	s.pos = ck.pos
	return nil
}

// Reset zeroes the conv histories and rewinds the position. Cache rows are left
// as they are: a token at position p reads only rows at positions <= p, which the
// new sequence has already written. The captured step graph reads the position
// from the input row, so it stays valid.
func (r *inkling2ResidentSession) Reset() error {
	s := r.s
	if err := s.rewindable(); err != nil {
		return err
	}
	ck, err := r.baseBuffers()
	if err != nil {
		return err
	}
	ck.valid = false
	hf := s.histFloats()
	for slot := range s.hist {
		r.copyDev(s.hist[slot], ck.zero, ck.histFill, hf)
	}
	if s.err != nil {
		return s.err
	}
	s.pos = 0
	return nil
}
