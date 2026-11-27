package ggufload

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync/atomic"
	"time"

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

// Inkling2LoopLogitSession is an optional interface implemented by sessions
// that expose the softcapped logits from the most recent Prefill or Step call
// for host-side non-greedy sampling.
type Inkling2LoopLogitSession interface {
	LastLogits() ([]float32, error)
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

// Inkling2LoopSnapshotter is an optional interface implemented by sessions that
// can capture and restore a portable Inkling2StateSnapshot at a snapshot boundary.
type Inkling2LoopSnapshotter interface {
	SnapshotState(boundary model.Inkling2SnapshotBoundary, horizon int, modelIdentity string) (*model.Inkling2StateSnapshot, error)
	RestoreState(snap *model.Inkling2StateSnapshot) error
}

// Inkling2LoopSpecSession extends Inkling2LoopSession with batched speculative
// verification and drafting for greedy decoding in fak serve.
type Inkling2LoopSpecSession interface {
	Inkling2LoopSession
	SpecMaxDraft() int
	SpecPlainUsesVerify() bool
	SpecStart(prompt []int) error
	SpecWarm(next int) error
	SpecDraft(history []int, k int) []int
	SpecVerify(tokens []int) ([]int, error)
	SpecCommitVerify(T, keep int) error
	SpecObserve(fed []int, keep int) error
	SpecObserveCost(width int, took time.Duration)
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
	h          *model.Inkling2LoopHost
	metal      bool
	recycle    func()
	ck         *model.Inkling2LoopHostState
	lastLogits []float32
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
		s.lastLogits = logits
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
	s.lastLogits = logits
	return inkling2Argmax(logits), nil
}

func (s *inkling2HostSession) Step(token int) (int, error) {
	logits, err := Inkling2LoopStep(s.h, token)
	if err != nil {
		return 0, err
	}
	s.recycle()
	s.lastLogits = logits
	return inkling2Argmax(logits), nil
}

func (s *inkling2HostSession) LastLogits() ([]float32, error) {
	if len(s.lastLogits) == 0 {
		return nil, fmt.Errorf("inkling2_loop: no logits available before Prefill or Step")
	}
	return s.lastLogits, nil
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
	s           *inkling2Resident
	ck          *ink2Rewind
	devOwner    *Inkling2Device
	specDepth   int
	specDrafter inkling2Drafter
	specLookup  bool
	specCosts   inkling2CostObserver
	specCleanup func()
}

// inkling2GraphOwner mirrors the default device's resident graph owner atomic
// for backward compatibility with existing graph-owner tests; all ownership
// transitions go through Inkling2Device.
var inkling2GraphOwner atomic.Pointer[inkling2Resident]

func (r *inkling2ResidentSession) device() *Inkling2Device {
	if r.devOwner != nil {
		return r.devOwner
	}
	if r.s != nil && r.s.dev != nil {
		return inkling2DeviceFor(r.s.dev)
	}
	return defaultInkling2Device
}

// own ensures r is the current graph owner on its device before it runs. A
// session taking ownership from another session, from a cohort, or after a
// reset drops its local capture flags and recaptures after one eager token.
func (r *inkling2ResidentSession) own() {
	if !r.device().Claim(r.s) {
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

// PrefillArenaBytes is the subset of HeldBytes currently allocated for lazy
// batched-prefill arenas (keep, scratch, and input staging).
func (r *inkling2ResidentSession) PrefillArenaBytes() int64 {
	if r.s == nil || r.s.prefillCap == 0 {
		return 0
	}
	return inkling2TensorBytes([]compute.Tensor{r.s.prefillKeep.T, r.s.prefillScr.T, r.s.prefillIn.T})
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
	if err := ctx.Err(); err != nil {
		return 0, err
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

func (r *inkling2ResidentSession) LastLogits() ([]float32, error) {
	if r.s == nil {
		return nil, fmt.Errorf("inkling2_loop: closed session")
	}
	r.own()
	return r.s.lastLogits()
}

func (r *inkling2ResidentSession) Close() error {
	if r == nil || r.s == nil {
		return nil
	}
	if r.ck != nil {
		r.freeGroup(&r.ck.grown)
		r.freeGroup(&r.ck.base)
		r.ck = nil
	}
	if r.specCleanup != nil {
		r.specCleanup()
		r.specCleanup = nil
	}
	dev := r.device()
	be := r.s.be
	r.s.close()
	r.s = nil
	dev.Reset()
	inkling2TrimIdle(be)
	return nil
}

func (r *inkling2ResidentSession) Pos() int { return r.s.pos }

// checkpointable settles state shared by FP32, FP16, FP8, and NVFP4 sessions. Global
// cache rows stay in their resident allocation across a rewind; only local FP32 ring
// rows that the horizon can overwrite are copied into checkpoint side buffers.
func (s *inkling2Resident) checkpointable() error {
	return s.rewindable()
}

func (s *inkling2Resident) rewindable() error {
	s.settle()
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

// checkpointRowWords is the 32-bit storage width of one cache row copied by a
// scalar checkpoint. Global rows return zero because they never wrap and stay in
// place. kvHalfSlot, kvFP8Slot, and kvNVFP4Slot are global-only, so every copied
// local row is FP32; copyDev's exact x + -0 operation must never be used for packed
// half, fp8, or nvfp4 words.
func (s *inkling2Resident) checkpointRowWords(slot int) int {
	if !s.localAt(slot) {
		return 0
	}
	return s.hp.KVHeads * s.hp.HeadDim
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

// growRows makes every local slot's and attached draft layer's side buffers hold
// at least rows and draftRows cache rows.
func (r *inkling2ResidentSession) growRows(ck *ink2Rewind, rows, draftRows int, draftSlots []ink2DraftLayerSlot) error {
	if rows <= ck.rows && draftRows <= ck.draftRows && len(ck.draft) == len(draftSlots) {
		return nil
	}
	s := r.s
	r.freeGroup(&ck.grown)
	ck.rows, ck.k, ck.v = 0, make([]ink2Ref, len(s.hist)), make([]ink2Ref, len(s.hist))
	ck.draftRows, ck.draft = 0, make([]ink2DraftRewind, len(draftSlots))
	maxWords := 0
	var err error
	if rows > 0 {
		for slot := 0; slot < len(s.hist) && err == nil; slot++ {
			words := s.checkpointRowWords(slot)
			if words == 0 {
				continue
			}
			maxWords = max(maxWords, rows*words)
			if ck.k[slot], err = r.allocInto(&ck.grown, rows*words); err == nil {
				ck.v[slot], err = r.allocInto(&ck.grown, rows*words)
			}
		}
	}
	for i, ds := range draftSlots {
		if err != nil {
			break
		}
		if draftRows <= 0 || ds.words <= 0 {
			continue
		}
		maxWords = max(maxWords, draftRows*ds.words)
		if ck.draft[i].k, err = r.allocInto(&ck.grown, draftRows*ds.words); err == nil {
			ck.draft[i].v, err = r.allocInto(&ck.grown, draftRows*ds.words)
		}
	}
	if err == nil && maxWords > 0 {
		var g []compute.Tensor
		ck.rowFill, g, err = r.negZeroFill(maxWords)
		ck.grown = append(ck.grown, g...)
	}
	if err != nil {
		r.freeGroup(&ck.grown)
		ck.draft = nil
		return fmt.Errorf("%w: %v", ErrInkling2CheckpointUnavailable, err)
	}
	ck.rows, ck.draftRows = rows, draftRows
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
	draftSlots := s.draftLayerSlots()
	var draftLo, draftHi []int
	draftRows := 0
	if len(draftSlots) > 0 {
		draftLo = make([]int, len(draftSlots))
		draftHi = make([]int, len(draftSlots))
		for i, ds := range draftSlots {
			lo, hi := inkling2SavedSpan(s.pos, ds.window, ds.cap, ck.horizon)
			draftLo[i], draftHi[i] = lo, hi
			draftRows = max(draftRows, hi-lo)
		}
	}
	if err := r.growRows(ck, ck.hi-ck.lo, draftRows, draftSlots); err != nil {
		return err
	}
	hf := s.histFloats()
	for slot := range s.hist {
		r.copyDev(ck.hist[slot], s.hist[slot], ck.histFill, hf)
		words := s.checkpointRowWords(slot)
		if words == 0 {
			continue
		}
		for _, sp := range inkling2RingSpans(ck.lo, ck.hi, s.caps[slot]) {
			r.copyDev(at(ck.k[slot], sp.off*words), at(s.kc[slot], sp.ring*words), ck.rowFill, sp.n*words)
			r.copyDev(at(ck.v[slot], sp.off*words), at(s.vc[slot], sp.ring*words), ck.rowFill, sp.n*words)
		}
	}
	for i, ds := range draftSlots {
		ck.draft[i].lo, ck.draft[i].hi = draftLo[i], draftHi[i]
		for _, sp := range inkling2RingSpans(draftLo[i], draftHi[i], ds.cap) {
			r.copyDev(at(ck.draft[i].k, sp.off*ds.words), at(ds.kc, sp.ring*ds.words), ck.rowFill, sp.n*ds.words)
			r.copyDev(at(ck.draft[i].v, sp.off*ds.words), at(ds.vc, sp.ring*ds.words), ck.rowFill, sp.n*ds.words)
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
	hf := s.histFloats()
	for slot := range s.hist {
		r.copyDev(s.hist[slot], ck.hist[slot], ck.histFill, hf)
		words := s.checkpointRowWords(slot)
		if words == 0 {
			continue
		}
		for _, sp := range inkling2RingSpans(ck.lo, ck.hi, s.caps[slot]) {
			r.copyDev(at(s.kc[slot], sp.ring*words), at(ck.k[slot], sp.off*words), ck.rowFill, sp.n*words)
			r.copyDev(at(s.vc[slot], sp.ring*words), at(ck.v[slot], sp.off*words), ck.rowFill, sp.n*words)
		}
	}
	for i, ds := range s.draftLayerSlots() {
		if i >= len(ck.draft) {
			break
		}
		dr := ck.draft[i]
		for _, sp := range inkling2RingSpans(dr.lo, dr.hi, ds.cap) {
			r.copyDev(at(ds.kc, sp.ring*ds.words), at(dr.k, sp.off*ds.words), ck.rowFill, sp.n*ds.words)
			r.copyDev(at(ds.vc, sp.ring*ds.words), at(dr.v, sp.off*ds.words), ck.rowFill, sp.n*ds.words)
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

func (s *inkling2HostSession) SnapshotState(boundary model.Inkling2SnapshotBoundary, horizon int, modelIdentity string) (*model.Inkling2StateSnapshot, error) {
	return s.h.StateSnapshot(boundary, horizon, modelIdentity)
}

func (s *inkling2HostSession) RestoreState(snap *model.Inkling2StateSnapshot) error {
	if err := s.h.RestoreStateSnapshot(snap); err != nil {
		return err
	}
	s.ck = nil
	s.lastLogits = nil
	return nil
}

func (r *inkling2ResidentSession) SnapshotState(boundary model.Inkling2SnapshotBoundary, horizon int, modelIdentity string) (*model.Inkling2StateSnapshot, error) {
	s := r.s
	if err := s.rewindable(); err != nil {
		return nil, err
	}
	if err := boundary.Validate(s.pos); err != nil {
		return nil, err
	}
	if s.kvHalf || s.kvFP8 || s.kvNVFP4 {
		return nil, fmt.Errorf("inkling2 resident: quantized global KV (%s) cannot be snapshotted to host F32", s.kvDType())
	}
	hf := s.histFloats()
	kvDim := s.hp.KVHeads * s.hp.HeadDim
	snap := &model.Inkling2StateSnapshot{
		ModelIdentity: modelIdentity,
		HP:            s.hp,
		Pos:           s.pos,
		Boundary:      boundary,
		Horizon:       max(horizon, 0),
		Conv:          make([][]float32, len(s.hist)),
	}
	for slot := range s.hist {
		convBuf, err := s.dev.Ink2Read(s.hist[slot], hf)
		if err != nil {
			return nil, err
		}
		snap.Conv[slot] = append([]float32(nil), convBuf...)

		if !s.localAt(slot) {
			gs := model.Inkling2SlotKVSnapshot{
				Slot:      slot,
				Positions: make([]int, s.pos),
				K:         make([][]float32, s.pos),
				V:         make([][]float32, s.pos),
			}
			if s.pos > 0 {
				kAll, err := s.dev.Ink2Read(s.kc[slot], s.pos*kvDim)
				if err != nil {
					return nil, err
				}
				vAll, err := s.dev.Ink2Read(s.vc[slot], s.pos*kvDim)
				if err != nil {
					return nil, err
				}
				kCopy := append([]float32(nil), kAll...)
				vCopy := append([]float32(nil), vAll...)
				for p := 0; p < s.pos; p++ {
					gs.Positions[p] = p
					gs.K[p] = kCopy[p*kvDim : (p+1)*kvDim]
					gs.V[p] = vCopy[p*kvDim : (p+1)*kvDim]
				}
			}
			snap.GlobalSlots = append(snap.GlobalSlots, gs)
			continue
		}

		lo := max(0, s.pos-s.hp.SlidingWindow)
		hi := s.pos
		rows := hi - lo
		ls := model.Inkling2RingKVSnapshot{
			Slot:      slot,
			Cap:       s.caps[slot],
			Lo:        lo,
			Hi:        hi,
			Positions: make([]int, rows),
			K:         make([][]float32, rows),
			V:         make([][]float32, rows),
		}
		for p := lo; p < hi; p++ {
			ls.Positions[p-lo] = p
		}
		kRing := make([]float32, rows*kvDim)
		vRing := make([]float32, rows*kvDim)
		for _, sp := range inkling2RingSpans(lo, hi, s.caps[slot]) {
			kSpan, err := s.dev.Ink2Read(at(s.kc[slot], sp.ring*kvDim), sp.n*kvDim)
			if err != nil {
				return nil, err
			}
			copy(kRing[sp.off*kvDim:(sp.off+sp.n)*kvDim], kSpan)
			vSpan, err := s.dev.Ink2Read(at(s.vc[slot], sp.ring*kvDim), sp.n*kvDim)
			if err != nil {
				return nil, err
			}
			copy(vRing[sp.off*kvDim:(sp.off+sp.n)*kvDim], vSpan)
		}
		for i := 0; i < rows; i++ {
			ls.K[i] = kRing[i*kvDim : (i+1)*kvDim]
			ls.V[i] = vRing[i*kvDim : (i+1)*kvDim]
		}
		snap.LocalRingSlots = append(snap.LocalRingSlots, ls)
	}
	return snap, nil
}

func (r *inkling2ResidentSession) RestoreState(snap *model.Inkling2StateSnapshot) error {
	s := r.s
	if err := s.rewindable(); err != nil {
		return err
	}
	if err := snap.Validate(); err != nil {
		return err
	}
	if s.kvHalf || s.kvFP8 || s.kvNVFP4 {
		return fmt.Errorf("inkling2 resident: quantized global KV (%s) cannot restore host F32 snapshot", s.kvDType())
	}
	if snap.Pos > s.ctx {
		return fmt.Errorf("inkling2 resident: snapshot pos=%d exceeds ctx=%d", snap.Pos, s.ctx)
	}
	wantHP, err := json.Marshal(s.hp)
	if err != nil {
		return err
	}
	gotHP, err := json.Marshal(snap.HP)
	if err != nil {
		return err
	}
	if !bytes.Equal(wantHP, gotHP) {
		return fmt.Errorf("%w: resident hparams mismatch", model.ErrInkling2StateSnapshotScope)
	}
	if len(snap.Conv) != len(s.hist) {
		return fmt.Errorf("%w: conv slots=%d, want %d", model.ErrInkling2StateSnapshotScope, len(snap.Conv), len(s.hist))
	}
	r.own()
	if err := r.Reset(); err != nil {
		return err
	}
	kvDim := s.hp.KVHeads * s.hp.HeadDim
	for slot, c := range snap.Conv {
		if len(c) != s.histFloats() {
			return fmt.Errorf("%w: conv slot %d len=%d, want %d", model.ErrInkling2StateSnapshotInvalid, slot, len(c), s.histFloats())
		}
		if err := s.dev.Ink2Write(s.hist[slot], c); err != nil {
			return err
		}
	}
	for _, gs := range snap.GlobalSlots {
		if gs.Slot < 0 || gs.Slot >= len(s.kc) || s.localAt(gs.Slot) {
			return fmt.Errorf("%w: invalid global slot %d", model.ErrInkling2StateSnapshotInvalid, gs.Slot)
		}
		if len(gs.K) > 0 {
			kFlat := make([]float32, len(gs.K)*kvDim)
			vFlat := make([]float32, len(gs.V)*kvDim)
			for i := range gs.K {
				copy(kFlat[i*kvDim:(i+1)*kvDim], gs.K[i])
				copy(vFlat[i*kvDim:(i+1)*kvDim], gs.V[i])
			}
			if err := s.dev.Ink2Write(s.kc[gs.Slot], kFlat); err != nil {
				return err
			}
			if err := s.dev.Ink2Write(s.vc[gs.Slot], vFlat); err != nil {
				return err
			}
		}
	}
	for _, ls := range snap.LocalRingSlots {
		if ls.Slot < 0 || ls.Slot >= len(s.kc) || !s.localAt(ls.Slot) {
			return fmt.Errorf("%w: invalid local slot %d", model.ErrInkling2StateSnapshotInvalid, ls.Slot)
		}
		rows := ls.Hi - ls.Lo
		if rows != len(ls.K) || rows != len(ls.V) {
			return fmt.Errorf("%w: local slot %d row count mismatch", model.ErrInkling2StateSnapshotInvalid, ls.Slot)
		}
		if rows > 0 {
			kFlat := make([]float32, rows*kvDim)
			vFlat := make([]float32, rows*kvDim)
			for i := 0; i < rows; i++ {
				copy(kFlat[i*kvDim:(i+1)*kvDim], ls.K[i])
				copy(vFlat[i*kvDim:(i+1)*kvDim], ls.V[i])
			}
			for _, sp := range inkling2RingSpans(ls.Lo, ls.Hi, s.caps[ls.Slot]) {
				if err := s.dev.Ink2Write(at(s.kc[ls.Slot], sp.ring*kvDim), kFlat[sp.off*kvDim:(sp.off+sp.n)*kvDim]); err != nil {
					return err
				}
				if err := s.dev.Ink2Write(at(s.vc[ls.Slot], sp.ring*kvDim), vFlat[sp.off*kvDim:(sp.off+sp.n)*kvDim]); err != nil {
					return err
				}
			}
		}
	}
	if r.ck != nil {
		r.ck.valid = false
	}
	s.pos = snap.Pos
	return s.err
}

type ink2DraftLayerSlot struct {
	kc, vc      ink2Ref
	cap, window int
	words       int
}

func (s *inkling2Resident) draftLayerSlots() []ink2DraftLayerSlot {
	switch d := s.draft.(type) {
	case *ink2DeviceDraft:
		if d == nil {
			return nil
		}
		words := d.hp.KVHeads * d.hp.HeadDim
		slots := make([]ink2DraftLayerSlot, len(d.layers))
		for i, l := range d.layers {
			w := l.window
			if w <= 0 {
				w = l.cap
			}
			slots[i] = ink2DraftLayerSlot{kc: l.kc, vc: l.vc, cap: l.cap, window: w, words: words}
		}
		return slots
	case *ink2DFlash2Draft:
		if d == nil {
			return nil
		}
		words := d.cfg.KVHeads * d.cfg.HeadDim
		slots := make([]ink2DraftLayerSlot, len(d.layers))
		for i, l := range d.layers {
			slots[i] = ink2DraftLayerSlot{kc: l.kc, vc: l.vc, cap: inkling2DFlash2Ring, window: inkling2DFlash2Ring, words: words}
		}
		return slots
	default:
		return nil
	}
}

func newInkling2LoopSpecResidentSession(m *Inkling2LoopModel, be compute.Backend, depth int, kind string, ds *Inkling2DSparkModel, df *Inkling2DFlash2Model) (Inkling2LoopSession, error) {
	s, err := newInkling2ResidentWithSpec(m, be, true)
	if err != nil {
		return nil, err
	}
	if err := s.enableSpec(depth); err != nil {
		s.close()
		return nil, err
	}
	r := &inkling2ResidentSession{s: s, specDepth: depth}
	switch kind {
	case inkling2DraftLookup:
		d := newInkling2LookupDrafter(nil)
		r.specLookup = true
		r.specDrafter = d
		r.specCosts = d
	case inkling2DraftDSpark:
		d, err := s.attachDraft(ds)
		if err != nil {
			s.close()
			return nil, err
		}
		d.steps = depth
		r.specDrafter = inkling2DSparkDrafter{d}
	case inkling2DraftDFlash:
		d, err := s.attachDFlash2(df)
		if err != nil {
			if d != nil {
				d.close()
			}
			s.close()
			return nil, err
		}
		r.specDrafter = inkling2DFlash2Drafter{d}
		r.specCleanup = d.close
	default:
		s.close()
		return nil, fmt.Errorf("inkling2 spec: unsupported draft kind %q", kind)
	}
	return r, nil
}

func (r *inkling2ResidentSession) SpecMaxDraft() int {
	if r == nil || r.s == nil || r.s.specMax <= 1 {
		return 0
	}
	return r.specDepth
}

func (r *inkling2ResidentSession) SpecPlainUsesVerify() bool {
	return r != nil && r.s != nil && inkling2VerifyMMA(r.s.be)
}

func (r *inkling2ResidentSession) SpecStart(prompt []int) error {
	if r == nil || r.s == nil || r.specDepth <= 0 {
		return nil
	}
	r.own()
	if r.specLookup {
		d := newInkling2LookupDrafter(prompt)
		r.specDrafter = d
		r.specCosts = d
	} else if df, ok := r.specDrafter.(inkling2DFlash2Drafter); ok && df.d != nil {
		df.d.width = nil
		df.d.run2.drafted = time.Time{}
	}
	if r.specDrafter == nil {
		return nil
	}
	return r.specDrafter.Observe(r.s, prompt, len(prompt))
}

func (r *inkling2ResidentSession) SpecWarm(next int) error {
	if r == nil || r.s == nil || !r.s.graph || r.s.verifyWarm || r.specDepth <= 0 {
		return nil
	}
	ctx := r.s.ctx
	if ctx <= 0 {
		ctx = inkling2ResidentCtx
	}
	if r.s.pos+r.specDepth+1 > ctx {
		return nil
	}
	r.own()
	warm := make([]int, r.specDepth+1)
	for i := range warm {
		warm[i] = next
	}
	if _, err := r.s.verify(warm); err != nil {
		return err
	}
	return r.s.commitVerify(len(warm), 0)
}

func (r *inkling2ResidentSession) SpecDraft(history []int, k int) []int {
	if r == nil || r.s == nil || r.specDrafter == nil || k <= 0 {
		return nil
	}
	r.own()
	return r.specDrafter.Draft(history, min(k, r.specDepth))
}

func (r *inkling2ResidentSession) SpecVerify(tokens []int) ([]int, error) {
	if r == nil || r.s == nil {
		return nil, fmt.Errorf("inkling2_loop: closed session")
	}
	r.own()
	return r.s.verify(tokens)
}

func (r *inkling2ResidentSession) SpecCommitVerify(T, keep int) error {
	if r == nil || r.s == nil {
		return fmt.Errorf("inkling2_loop: closed session")
	}
	r.own()
	return r.s.commitVerify(T, keep)
}

func (r *inkling2ResidentSession) SpecObserve(fed []int, keep int) error {
	if r == nil || r.s == nil || r.specDrafter == nil {
		return nil
	}
	return r.specDrafter.Observe(r.s, fed, keep)
}

func (r *inkling2ResidentSession) SpecObserveCost(width int, took time.Duration) {
	if r != nil && r.specCosts != nil {
		r.specCosts.ObserveCost(width, took)
	}
}
