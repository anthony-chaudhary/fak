package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// The ordinary callback and its optional canonical core form one immutable
// binding. Replacing a callback never transfers permission to borrow inputs,
// even when the replacement is the same function or wraps the old callback.
// State/scratch copies share this binding; prefix snapshots omit it.
type v41IndexerScoreBinding struct {
	ordinary v41IndexerScoreFunc
	native   v41BoundIndexerScore
}

type v41IndexerScoreCore func(int, []float32, []float32, []float32, int, int, int, v41IndexerScoreKeys) ([]float32, error)

type v41BoundIndexerScore struct {
	ordinary v41IndexerScoreFunc
	core     v41IndexerScoreCore
	owner    *Session
	model    *Model
	backend  compute.Backend
}

func newV41IndexerScoreBinding(score v41IndexerScoreFunc) *v41IndexerScoreBinding {
	if score == nil {
		return nil
	}
	return &v41IndexerScoreBinding{ordinary: score}
}

func (b *v41IndexerScoreBinding) scoreFunc() v41IndexerScoreFunc {
	if b == nil {
		return nil
	}
	return b.ordinary
}

func (st *v41ForwardState) indexerScore() v41IndexerScoreFunc {
	return st.indexerScoreBinding.scoreFunc()
}

func (scratch *v41ProjScratch) indexerScore() v41IndexerScoreFunc {
	return scratch.indexerScoreBinding.scoreFunc()
}

func (st *v41ForwardState) setIndexerScore(score v41IndexerScoreFunc) {
	st.indexerScoreBinding = newV41IndexerScoreBinding(score)
}

func (st *v41ForwardState) bindCanonicalIndexerScore(score v41BoundIndexerScore) {
	b := newV41IndexerScoreBinding(score.ordinary)
	if b != nil {
		b.native = score
	}
	st.indexerScoreBinding = b
}

func (b *v41IndexerScoreBinding) nativeOwner(m *Model, st *v41ForwardState, scratch *v41ProjScratch) *Session {
	if b == nil || b.ordinary == nil || b.native.core == nil || st == nil || scratch == nil {
		return nil
	}
	n := b.native
	s := n.owner
	if s == nil || n.model != m || s.M != m || s.Backend != n.backend || s.halClosed || s.halFailure != nil || !s.modelWeightsHeld ||
		st.callbackOwner != s || st.indexerScoreBinding != b || scratch.indexerScoreBinding != b ||
		s.v41Forward == nil || s.v41Forward.callbackOwner != s || s.v41Forward.indexerScoreBinding != b {
		return nil
	}
	// The public execution boundary has already acquired this backend's owner.
	// Checking capability and membership does not probe or re-admit the scorer.
	if _, ok := n.backend.(compute.ForwardExecutionOwnerBackend); !ok {
		return nil
	}
	scope := s.backendExecutionScope
	if scope == nil || !scope.live || scope.owner == nil {
		return nil
	}
	for _, owned := range scope.sessions {
		if owned == s {
			return s
		}
	}
	return nil
}

// A borrow belongs only to the synchronous cold query loop. It contains no
// input storage or device tensor. The current query freezes its prefix before
// projections, then rechecks authority after health immediately before scoring.
type v41IndexerScoreBorrow struct {
	binding *v41IndexerScoreBinding
	model   *Model
	state   *v41ForwardState
	scratch *v41ProjScratch
	forward *v41ForwardState
	scope   *backendExecutionScope
	epoch   uint64
	revoked bool
}

func (b *v41IndexerScoreBinding) prepareBorrow(m *Model, st *v41ForwardState, scratch *v41ProjScratch) v41IndexerScoreBorrow {
	s := b.nativeOwner(m, st, scratch)
	if s == nil {
		return v41IndexerScoreBorrow{}
	}
	return v41IndexerScoreBorrow{binding: b, model: m, state: st, scratch: scratch,
		forward: s.v41Forward, scope: s.backendExecutionScope, epoch: s.cacheGeometryEpoch}
}

func (b *v41IndexerScoreBorrow) valid() bool {
	if b == nil || b.binding == nil || b.revoked {
		return false
	}
	s := b.binding.nativeOwner(b.model, b.state, b.scratch)
	return s != nil && s.v41Forward == b.forward && s.backendExecutionScope == b.scope && s.cacheGeometryEpoch == b.epoch
}

func (b *v41IndexerScoreBorrow) resolve(keys []float32) (v41IndexerScoreFunc, []float32, v41IndexerScoreKeys) {
	if b.valid() {
		return b.binding.scoreFunc(), keys, v41IndexerScoreKeys{borrow: b, frozen: keys}
	}
	// Never re-read source rows after projections/health have run, or dispatch
	// a replacement callback. One revoked query needs one exact prefix copy;
	// its peak is panel + visible prefix, at most two complete panel payloads.
	// The caller drops the panel before the next query, preventing accumulation.
	b.revoked = true
	private := make([]float32, len(keys))
	copy(private, keys)
	return b.binding.scoreFunc(), private, v41IndexerScoreKeys{}
}

// The value carrier is local to one query. It does not retain input storage in
// the binding, Session or continuation state. A zero carrier is the unchanged
// ordinary path, including a prefix already made private after health.
type v41IndexerScoreKeys struct {
	borrow *v41IndexerScoreBorrow
	frozen []float32
}

func (k v41IndexerScoreKeys) beforeUpload(ordinary []float32) []float32 {
	if k.borrow == nil {
		return ordinary
	}
	if k.borrow.valid() {
		return k.frozen
	}
	// Admission and query upload are existing callbacks and may revoke the
	// earlier decision. Preserve its pre-projection snapshot, copy just once,
	// and permanently end borrowing for this cold query loop.
	k.borrow.revoked = true
	private := make([]float32, len(k.frozen))
	copy(private, k.frozen)
	return private
}
